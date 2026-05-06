import json
import os
import re
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from reasoner._sort_utils import natural_dir_sort_key

if TYPE_CHECKING:
    from reasoner._registries import RelationFragment

logger = logging.getLogger(__name__)

__all__ = [
    "KnowledgeChunk",
    "build_knowledge_chunks",
    "split_relations_into_chunks",
    "build_parent_location_label",
    "build_target_location_label",
    "natural_dir_sort_key",
]


def build_target_location_label(fragment: "RelationFragment") -> str:
    """公开入口：构造 RelationFragment 的本体业务定位（见 _build_target_location_label）。"""
    return _build_target_location_label(fragment)

_KNOWLEDGE_NAME_CACHE: dict[str, str] = {}


def _read_knowledge_name(knowledge_root: str) -> str:
    """读取知识包的业务名称（如 "农产品精简版"），带进程级缓存。

    解析优先级：
        1. <page_knowledge_dir>/_policy_index.json 中 entry.root == basename(knowledge_root)
           的 entry.name；
        2. knowledge_root 下任一子目录 clause.json 的 searchLabels[0]，按 "|" 切第一段；
        3. 兜底：knowledge_root 的 basename（可能带时间戳）。
    """
    if not knowledge_root:
        return ""
    key = os.path.normpath(knowledge_root)
    cached = _KNOWLEDGE_NAME_CACHE.get(key)
    if cached is not None:
        return cached

    name = ""
    base = os.path.basename(key)

    try:
        page_dir = os.path.dirname(key)
        idx_path = os.path.join(page_dir, "_policy_index.json")
        if os.path.isfile(idx_path):
            with open(idx_path, "r", encoding="utf-8") as f:
                idx = json.load(f)
            for entry in (idx or {}).values():
                if not isinstance(entry, dict):
                    continue
                if entry.get("root") == base and entry.get("name"):
                    name = str(entry["name"]).strip()
                    break
    except Exception as e:
        logger.debug(f"[KnowledgeName] 读取 _policy_index.json 失败: {e}")

    if not name:
        try:
            for d in sorted(os.listdir(key)):
                sub = os.path.join(key, d)
                if not os.path.isdir(sub):
                    continue
                cj = os.path.join(sub, "clause.json")
                if not os.path.isfile(cj):
                    continue
                with open(cj, "r", encoding="utf-8") as f:
                    data = json.load(f)
                labels = data.get("searchLabels") or []
                if labels and isinstance(labels[0], str) and "|" in labels[0]:
                    name = labels[0].split("|", 1)[0].strip()
                    if name:
                        break
        except Exception as e:
            logger.debug(f"[KnowledgeName] 回退读取子 clause.json 失败: {e}")

    if not name:
        name = base

    _KNOWLEDGE_NAME_CACHE[key] = name
    return name


def build_parent_location_label(knowledge_root: str, parent_dir: str) -> str:
    """把 parent_dir 组装为业务化可读的定位串：

        "<知识名> > <一级章节> > ... > <末级章节>"

    - 最根部加上知识名（来自 _policy_index.json 或子 clause.json 的 searchLabels）；
    - 章节段沿用目录名（保留 "2_涉税处理" 这类业务编号），便于 LLM 回贴原文；
    - parent_dir 为空或无法相对化时退化为目录名列表或空串。
    """
    if not parent_dir:
        return ""
    try:
        rel = os.path.relpath(parent_dir, knowledge_root) if knowledge_root else parent_dir
    except ValueError:
        rel = parent_dir
    segs = [s for s in rel.replace("\\", "/").split("/") if s and s != "."]
    kn = _read_knowledge_name(knowledge_root) if knowledge_root else ""
    if kn:
        return " > ".join([kn, *segs])
    return " > ".join(segs) if segs else ""

_IGNORED_DIRS = frozenset({
    "__pycache__", ".ipynb_checkpoints", ".git", ".svn",
    "node_modules", ".venv", "venv", ".tox",
})

_SECTION_HEADERS_TO_STRIP = ("## 当前路径", "## 项目条款名称", "## 全称", "## 子目录概览")

_DIR_NUMBER_RE = re.compile(r'^(\d+(?:\.\d+)*)_')


@dataclass
class KnowledgeChunk:
    index: int
    content: str
    heading_paths: list[list[str]] = field(default_factory=list)
    directories: list[str] = field(default_factory=list)
    # 派生 chunk 标记：来自原 chunk 的关联展开（多跳条款）。default None 不影响原始 chunk。
    parent_chunk_index: int | None = None
    derived_seq: int = 0
    # 该 chunk 命中的关联条款 (policy_id, clause_id) 列表，用于 trace。原始 chunk 为空。
    relation_keys: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class _KnowledgeNode:
    """A single node in the knowledge directory tree."""
    dir_path: str
    dir_name: str
    depth: int
    ancestors: list[str]
    body: str


def _list_subdirs(directory: str) -> list[str]:
    if not os.path.isdir(directory):
        return []
    candidates = [
        d for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d))
        and not d.startswith(".")
        and d not in _IGNORED_DIRS
    ]
    return sorted(candidates, key=lambda d: (natural_dir_sort_key(d), d))


def _read_knowledge_md(directory: str) -> str:
    km_path = os.path.join(directory, "knowledge.md")
    if os.path.exists(km_path):
        with open(km_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def _strip_metadata_sections(content: str) -> str:
    """Remove metadata sections, keep only the substantive body."""
    for heading in _SECTION_HEADERS_TO_STRIP:
        idx = content.find(heading)
        if idx == -1:
            continue
        next_heading_idx = content.find("\n## ", idx + len(heading))
        if heading == "## 子目录概览":
            content = content[:idx].rstrip()
        elif next_heading_idx != -1:
            content = content[:idx].rstrip() + "\n\n" + content[next_heading_idx + 1:]
        else:
            content = content[:idx].rstrip()

    body_marker = "## 本章节内容"
    idx = content.find(body_marker)
    if idx != -1:
        content = content[idx + len(body_marker):].strip()

    return content.strip()


def _get_dir_depth(dir_name: str) -> int:
    """Return the hierarchy depth based on the directory number prefix (e.g. '1' -> 1, '1.2' -> 2)."""
    m = _DIR_NUMBER_RE.match(dir_name)
    if not m:
        return 0
    return len(m.group(1).split('.'))


def _is_top_level_dir(dir_name: str) -> bool:
    return _get_dir_depth(dir_name) == 1


def _format_heading_label(ancestors: list[str], dir_name: str) -> str:
    parts = ancestors + [dir_name]
    return "【" + " > ".join(parts) + "】"


def _walk_knowledge_tree(
    directory: str,
    ancestors: list[str],
    depth: int,
) -> list[_KnowledgeNode]:
    """Depth-first traversal of the knowledge tree, yielding nodes in order."""
    nodes: list[_KnowledgeNode] = []

    raw_content = _read_knowledge_md(directory)
    body = _strip_metadata_sections(raw_content) if raw_content else ""

    dir_name = os.path.basename(directory)
    nodes.append(_KnowledgeNode(
        dir_path=directory,
        dir_name=dir_name,
        depth=depth,
        ancestors=list(ancestors),
        body=body,
    ))

    subdirs = _list_subdirs(directory)
    for subdir in subdirs:
        subdir_path = os.path.join(directory, subdir)
        child_ancestors = ancestors + [dir_name] if depth > 0 else [subdir]
        child_nodes = _walk_knowledge_tree(
            subdir_path,
            ancestors=child_ancestors if depth > 0 else [],
            depth=depth + 1,
        )
        nodes.extend(child_nodes)

    return nodes


def build_knowledge_chunks(
    knowledge_root: str,
    chunk_size: int = 5000,
) -> list[KnowledgeChunk]:
    """
    Traverse the knowledge directory tree depth-first and split into chunks
    of approximately `chunk_size` characters each.

    Rules:
    - Each top-level directory (depth==1, e.g. "1_会计处理") forces a new chunk.
    - When appending a node would exceed chunk_size and the current chunk is
      non-empty, start a new chunk.
    - Each new chunk that doesn't start at a top-level directory gets a heading
      label prefix showing the full upstream path for semantic completeness.
    """
    all_nodes = _walk_knowledge_tree(knowledge_root, ancestors=[], depth=0)

    if not all_nodes:
        logger.warning("[ChunkBuilder] 知识目录树为空")
        return []

    # Skip the root node (depth==0) as it's just an overview
    content_nodes = [n for n in all_nodes if n.depth > 0]

    if not content_nodes:
        logger.warning("[ChunkBuilder] 知识目录树无内容节点")
        return []

    chunks: list[KnowledgeChunk] = []
    current_text_parts: list[str] = []
    current_length = 0
    current_heading_paths: list[list[str]] = []
    current_directories: list[str] = []

    def _finalize_chunk():
        nonlocal current_text_parts, current_length, current_heading_paths, current_directories
        if current_text_parts:
            chunks.append(KnowledgeChunk(
                index=len(chunks) + 1,
                content="\n\n".join(current_text_parts),
                heading_paths=list(current_heading_paths),
                directories=list(current_directories),
            ))
        current_text_parts = []
        current_length = 0
        current_heading_paths = []
        current_directories = []

    for node in content_nodes:
        if not node.body:
            continue

        is_top = _is_top_level_dir(node.dir_name)

        if is_top and current_text_parts:
            _finalize_chunk()

        heading_label = _format_heading_label(node.ancestors, node.dir_name)
        node_text = f"{heading_label}\n{node.body}"
        node_len = len(node_text)

        if current_text_parts and (current_length + node_len + 2) > chunk_size:
            _finalize_chunk()

        if not current_text_parts and node.ancestors:
            upstream_label = "【" + " > ".join(node.ancestors) + "】"
            if not node_text.startswith(upstream_label):
                pass

        current_text_parts.append(node_text)
        current_length += node_len + 2
        current_heading_paths.append(node.ancestors + [node.dir_name])
        current_directories.append(node.dir_path)

    _finalize_chunk()

    logger.info(
        f"[ChunkBuilder] 知识分块完成：共 {len(chunks)} 个块，"
        f"平均长度 {sum(len(c.content) for c in chunks) // max(len(chunks), 1)} 字符"
    )
    for i, chunk in enumerate(chunks):
        logger.debug(
            f"[ChunkBuilder] Chunk {i+1}: {len(chunk.content)} 字符, "
            f"{len(chunk.directories)} 个目录节点"
        )

    return chunks


def _format_relation_fragment_text(
    fragment: "RelationFragment",
    knowledge_root: str = "",
) -> str:
    """把单个 RelationFragment 渲染为用于派生 chunk 的字符串块。

    头部三行是同构的三条业务坐标，**格式完全一致**（都是"【标签 · 内容】"）：
      - 【来自父章节 · ...】触发本次关联展开的上游章节路径；
      - 【命中关键词 · ...】父章节里那段被高亮、并把本条款牵出来的 highlightedContent；
      - 【关联条款位置 · ...】本条款自身在知识库里的章节路径；
    三者同构是刻意设计——LLM 能一眼看出"父章节 → 关键词 → 本条款"这条溯源链条上的三个坐标点，
    从而明确"这块关联知识属于 A 章节、但它是因为 B 章节正文里高亮的某词才被拉进来的"。

    故意**不把** hop_depth / source / policyId / clauseId 放进 prompt：这些是内部去重与 trace 用的
    技术标识（UUID、数字跳深），对 LLM 的业务推理没有任何帮助，反而挤占 token 与干扰注意力。
    """
    target_label = _build_target_location_label(fragment)

    heading_lines: list[str] = []
    parent_label = build_parent_location_label(knowledge_root, fragment.parent_dir)
    if parent_label:
        heading_lines.append(f"【来自父章节 · {parent_label}】")
    if fragment.highlighted:
        heading_lines.append(f"【命中关键词 · {fragment.highlighted}】")
    heading_lines.append(f"【关联条款位置 · {target_label}】")

    meta_lines: list[str] = []
    if fragment.parent_assessment:
        meta_lines.append(f"> 上层关联性判定: {_one_line(fragment.parent_assessment, 200)}")

    if fragment.highlighted:
        intro = f"**{fragment.highlighted}的关联知识细节如下：**"
    else:
        intro = "**关联知识细节如下：**"

    body = (fragment.content or "").strip() or "（条款内容为空）"
    return "\n".join([*heading_lines, *meta_lines, "", intro, body])


def _build_target_location_label(fragment: "RelationFragment") -> str:
    """为 RelationFragment 构造"本条款在知识库中的业务定位"字符串。

    优先级（与 parent_label 逻辑对称）：
      1. target_dir + target_knowledge_root 都非空：走 build_parent_location_label，
         得到 "知识名 > 2_涉税处理 > 2.1_增值税 > ..."（带业务编号）。
      2. 任一缺失（远程兜底节点无 dir_abspath）：回退到 heading_path 拼接；
         heading_path 是名称段（无编号），由 ClauseLocator 从相对路径去编号产出，
         此时无法带编号，只能退化为可读名链。
      3. 全部缺失：退化到 clause_full_name 或 clause_id。
    """
    if fragment.target_dir and fragment.target_knowledge_root:
        label = build_parent_location_label(
            fragment.target_knowledge_root, fragment.target_dir,
        )
        if label:
            return label
    heading_parts = list(fragment.heading_path)
    if heading_parts:
        return " > ".join(heading_parts)
    return fragment.clause_full_name or fragment.clause_id


def _one_line(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= limit else text[:limit] + "…"


def split_relations_into_chunks(
    fragments: list["RelationFragment"],
    chunk_size: int,
    parent_chunk: KnowledgeChunk,
    start_derived_seq: int = 1,
    knowledge_root: str = "",
) -> list[KnowledgeChunk]:
    """把一批关联条款片段按 chunk_size 切分为派生 KnowledgeChunk 列表。

    切分规则（与原始 chunk 思路对齐）：
      - 不切碎单个 fragment：哪怕单个 fragment 自身超过 chunk_size 也独立成一个 chunk，
        避免 markdown 表格/段落被截断破坏语义；
      - 多个小 fragment 在不超 chunk_size 的前提下合并到同一 chunk；
      - 派生 chunk 共享 parent_chunk_index = parent_chunk.index；
      - 派生 chunk 的 directories / heading_paths 仅记录 parent_chunk 的归属（用于 trace），
        关联条款本身的 heading 信息已经写入 content 内的【关联条款 · ...】标签，不再重复进入
        parent dirs，避免污染 khObj 的章节回填；
      - index 暂用 0 占位，由调用方（AgentGraph）扁平拼接到原始 chunks 后统一重排。

    返回的 list 顺序即派生 chunk 在原始 chunk 之后的逻辑顺序。
    """
    if not fragments:
        return []

    chunks: list[KnowledgeChunk] = []
    current_parts: list[str] = []
    current_len = 0
    current_keys: list[tuple[str, str]] = []
    seq = start_derived_seq

    def _finalize():
        nonlocal current_parts, current_len, current_keys, seq
        if not current_parts:
            return
        chunks.append(KnowledgeChunk(
            index=0,  # 占位，调用方重排
            content="\n\n".join(current_parts),
            heading_paths=list(parent_chunk.heading_paths),
            directories=list(parent_chunk.directories),
            parent_chunk_index=parent_chunk.index,
            derived_seq=seq,
            relation_keys=list(current_keys),
        ))
        current_parts = []
        current_len = 0
        current_keys = []
        seq += 1

    for frag in fragments:
        text = _format_relation_fragment_text(frag, knowledge_root=knowledge_root)
        text_len = len(text)
        # 单 fragment 超 chunk_size：先 flush 当前累积，再让该 fragment 独立成一个 chunk
        if text_len >= chunk_size:
            _finalize()
            chunks.append(KnowledgeChunk(
                index=0,
                content=text,
                heading_paths=list(parent_chunk.heading_paths),
                directories=list(parent_chunk.directories),
                parent_chunk_index=parent_chunk.index,
                derived_seq=seq,
                relation_keys=[(frag.policy_id, frag.clause_id)],
            ))
            seq += 1
            continue
        if current_parts and (current_len + text_len + 2) > chunk_size:
            _finalize()
        current_parts.append(text)
        current_len += text_len + 2
        current_keys.append((frag.policy_id, frag.clause_id))

    _finalize()
    return chunks
