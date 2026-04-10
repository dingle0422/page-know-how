import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# 匹配 "1." / "1.1." / "1.1.1." 等格式，后接一个或多个空白字符（空格、Tab 等），再接标题文字
HEADING_PATTERN = re.compile(r'^(\d+(?:\.\d+)*)\.\s+(.+)$')


@dataclass
class HeadingNode:
    number: str                          # "1.1.2" 或 "0"（文档标题）
    title: str
    content: str = ""
    children: list['HeadingNode'] = field(default_factory=list)

    @property
    def depth(self) -> int:
        if not self.number or self.number == "0":
            return 0
        return len(self.number.split('.'))

    @property
    def folder_name(self) -> str:
        from utils.helpers import sanitize_filename
        return f"{self.number}_{sanitize_filename(self.title)}"


def build_heading_tree(lines: list) -> list[HeadingNode]:
    """
    逐行扫描识别标题，优先级：
    1. 文本匹配 '1.' / '1.1.' / '1.1.1.' + 空白 + 文字 → 文本模式标题（兼容 txt）
    2. 段落带有 Word Heading 样式（heading_level > 0）且文本不匹配模式 → 样式模式标题（兼容 docx）
    3. 文档第一个非空行若不是标题 → 作为文档标题（编号 "0"）
    4. 其余行归入最近标题的 content
    """
    title_node: HeadingNode | None = None
    root_nodes: list[HeadingNode] = []
    stack: list[HeadingNode] = []
    current_node: HeadingNode | None = None

    # 用于样式模式下自动生成层级编号（如 "1", "1.2", "1.2.3"）
    style_counters: dict[int, int] = {}

    def _next_style_number(depth: int) -> str:
        """递增指定深度的计数器，重置更深层级，返回形如 '1.2.3' 的编号"""
        style_counters[depth] = style_counters.get(depth, 0) + 1
        for d in list(style_counters.keys()):
            if d > depth:
                del style_counters[d]
        return '.'.join(str(style_counters.get(d, 1)) for d in range(1, depth + 1))

    def _make_node(text: str, heading_level: int) -> HeadingNode | None:
        """
        尝试从文本或 Word 样式层级创建标题节点。
        - 文本匹配 HEADING_PATTERN → 优先使用文本中的编号
        - 否则若有样式层级 → 使用自动编号
        - 否则返回 None（普通内容行）
        """
        if not text:
            return None
        m = HEADING_PATTERN.match(text)
        if m:
            number = m.group(1)
            title = m.group(2).strip()
            return HeadingNode(number=number, title=title)
        if heading_level > 0:
            number = _next_style_number(heading_level)
            return HeadingNode(number=number, title=text)
        return None

    for item in lines:
        text = item.text.strip() if hasattr(item, 'text') else str(item).strip()
        heading_level = getattr(item, 'heading_level', 0)

        if title_node is None and not text:
            continue

        # 处理文档第一个非空行
        if title_node is None and text:
            node = _make_node(text, heading_level)
            if node is not None:
                root_nodes.append(node)
                stack.append(node)
                current_node = node
            else:
                title_node = HeadingNode(number="0", title=text)
                current_node = title_node
            continue

        if not text:
            if current_node is not None:
                current_node.content += "\n"
            continue

        node = _make_node(text, heading_level)
        if node is not None:
            new_depth = node.depth
            while stack and stack[-1].depth >= new_depth:
                stack.pop()
            if stack:
                stack[-1].children.append(node)
            else:
                root_nodes.append(node)
            stack.append(node)
            current_node = node
        else:
            if current_node is not None:
                if current_node.content:
                    current_node.content += "\n" + text
                else:
                    current_node.content = text

    all_nodes = []
    if title_node is not None:
        title_node.content = title_node.content.strip()
        all_nodes.append(title_node)
    for node in root_nodes:
        _strip_content(node)
    all_nodes.extend(root_nodes)

    return all_nodes


def _strip_content(node: HeadingNode):
    """递归清理每个节点的 content 首尾空白"""
    node.content = node.content.strip()
    for child in node.children:
        _strip_content(child)


def build_tree_from_clauses(clauses: list[dict]) -> list[HeadingNode]:
    """
    从已结构化的 clause_list 构建 HeadingNode 树。

    clause_list 中每个 clause 已包含 number / level / path / content，
    可直接利用 path 字段确定父子关系，无需再做标题识别。

    参数:
        clauses: parse_clause_json() 返回的 Clause 列表

    返回:
        HeadingNode 树的根节点列表
    """
    if not clauses:
        return []

    node_map: dict[str, HeadingNode] = {}
    root_nodes: list[HeadingNode] = []

    for clause in clauses:
        number = clause['number']
        content = clause['content']
        level = clause['level']
        path = clause.get('path', '')

        title_match = HEADING_PATTERN.match(content)
        if title_match:
            title = title_match.group(2).strip()
        else:
            title = content.split('\n', 1)[0].strip()

        node = HeadingNode(number=number, title=title, content=content)
        node_map[number] = node

        if not path:
            root_nodes.append(node)
        else:
            parent = node_map.get(path)
            if parent is not None:
                parent.children.append(node)
            else:
                logger.warning(f"条款 {number} 的父路径 '{path}' 未找到对应节点，作为根节点处理")
                root_nodes.append(node)

    for node in root_nodes:
        _strip_content(node)

    return root_nodes
