import os
import time
import logging

from extractor.parser import parse_document, parse_clause_json
from extractor.heading_tree import build_heading_tree, build_tree_from_clauses, HeadingNode
from utils.helpers import truncate_text

logger = logging.getLogger(__name__)

CHILD_SUMMARY_HINT = (
    "\n\n> 以上仅为摘要，具体细节可能遗漏。"
    "建议打开对应子文件夹内的 knowledge.md 进行渐进式探索，"
    "以获取更完整的细节信息。"
)


def _build_knowledge_md(node: HeadingNode, abs_dir: str) -> str:
    """为单个节点生成 knowledge.md 的内容"""
    sections = []

    sections.append(f"## 当前路径\n\n{abs_dir}")

    if node.content:
        sections.append(f"## 本章节内容\n\n{node.content}")
    else:
        sections.append("## 本章节内容\n\n（本章节无直接内容，请查看子目录）")

    if node.children:
        child_lines = []
        for child in node.children:
            summary = truncate_text(child.content, 200) if child.content else "（无直接内容，包含更细分的子章节）"
            child_lines.append(f"- **{child.folder_name}**: {summary}")
        children_text = "\n".join(child_lines) + CHILD_SUMMARY_HINT
        sections.append(f"## 子目录概览\n\n{children_text}")

    return "\n\n".join(sections) + "\n"


def _build_root_knowledge_md(nodes: list[HeadingNode], abs_root: str) -> str:
    """为知识目录的根目录生成 knowledge.md"""
    sections = []
    sections.append(f"## 当前路径\n\n{abs_root}")
    sections.append("## 文档概览\n\n本目录为文档知识的结构化根目录，包含以下一级章节：")

    child_lines = []
    for node in nodes:
        summary = truncate_text(node.content, 200) if node.content else "（无直接内容，包含更细分的子章节）"
        child_lines.append(f"- **{node.folder_name}**: {summary}")
    children_text = "\n".join(child_lines) + CHILD_SUMMARY_HINT
    sections.append(f"## 子目录概览\n\n{children_text}")

    return "\n\n".join(sections) + "\n"


def _build_dirs_recursive(nodes: list[HeadingNode], base_dir: str):
    """递归构建目录结构和 knowledge.md"""
    for node in nodes:
        folder_name = node.folder_name
        node_dir = os.path.join(base_dir, folder_name)
        os.makedirs(node_dir, exist_ok=True)

        abs_dir = os.path.abspath(node_dir)
        knowledge_content = _build_knowledge_md(node, abs_dir)
        knowledge_path = os.path.join(node_dir, "knowledge.md")
        with open(knowledge_path, "w", encoding="utf-8") as f:
            f.write(knowledge_content)

        if node.children:
            _build_dirs_recursive(node.children, node_dir)


def extract(filepath: str) -> str:
    """
    知识抽取主入口。
    1. 解析文档为文本
    2. 构建标题树
    3. 在 page_knowledge/{filename}_{timestamp_ms}/ 下构建目录
    返回生成的知识目录根路径。
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    page_knowledge_dir = os.path.join(project_root, "page_knowledge")
    os.makedirs(page_knowledge_dir, exist_ok=True)

    filename = os.path.splitext(os.path.basename(filepath))[0]
    timestamp_ms = int(time.time() * 1000)
    knowledge_root = os.path.join(page_knowledge_dir, f"{filename}_{timestamp_ms}")
    os.makedirs(knowledge_root, exist_ok=True)

    ext = os.path.splitext(filepath)[1].lower()
    logger.info(f"开始解析文档: {filepath}")

    if ext == '.json':
        clauses = parse_clause_json(filepath)
        logger.info("从 clause_list 构建标题树...")
        tree = build_tree_from_clauses(clauses)
    else:
        parsed_lines = parse_document(filepath)
        logger.info("构建标题树...")
        tree = build_heading_tree(parsed_lines)

    if not tree:
        logger.warning("未识别到任何标准标题，知识目录将为空")
        return knowledge_root

    logger.info(f"识别到 {len(tree)} 个一级标题，开始构建目录结构...")
    _build_dirs_recursive(tree, knowledge_root)

    root_knowledge = _build_root_knowledge_md(tree, os.path.abspath(knowledge_root))
    with open(os.path.join(knowledge_root, "knowledge.md"), "w", encoding="utf-8") as f:
        f.write(root_knowledge)

    logger.info(f"知识抽取完成，输出目录: {knowledge_root}")
    return knowledge_root
