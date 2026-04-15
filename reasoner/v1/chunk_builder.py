import os
import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

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
    return [
        d for d in sorted(os.listdir(directory))
        if os.path.isdir(os.path.join(directory, d))
        and not d.startswith(".")
        and d not in _IGNORED_DIRS
    ]


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
