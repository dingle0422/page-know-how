"""
基础设施：跨版本共享的注册表与数据结构。

设计意图：
- 这些类是纯基础设施（线程安全的注册表 + dataclass），不属于 v0 或 v1 版本特有的业务逻辑
- v0 / v1 都从这里 import，避免 v1.agent_graph 反向依赖 v0.agent_graph 造成版本耦合
- 与 reasoner/_sort_utils.py 共享工具的设计保持一致
"""

import os
import threading
from dataclasses import dataclass


class ExploredRegistry:
    """线程安全的已探索节点注册表，同一问题的所有子智能体共享"""

    def __init__(self):
        self._lock = threading.Lock()
        self._explored: dict[str, str] = {}

    def try_claim(self, dir_path: str, agent_id: str) -> bool:
        with self._lock:
            if dir_path in self._explored:
                return False
            self._explored[dir_path] = agent_id
            return True

    def is_explored(self, dir_path: str) -> bool:
        with self._lock:
            return dir_path in self._explored

    def get_explorer(self, dir_path: str) -> str | None:
        with self._lock:
            return self._explored.get(dir_path)


class PitfallsRegistry:
    """线程安全的全局易错点注册表，同一问题的所有子智能体共享"""

    def __init__(self):
        self._lock = threading.Lock()
        self._pitfalls: list[str] = []
        self._dir_pitfalls: dict[str, list[str]] = {}

    def add(self, pitfalls: list[str], directory: str = "") -> None:
        with self._lock:
            for p in pitfalls:
                if p and p not in self._pitfalls:
                    self._pitfalls.append(p)
            if directory:
                norm_dir = os.path.normpath(directory)
                if norm_dir not in self._dir_pitfalls:
                    self._dir_pitfalls[norm_dir] = []
                for p in pitfalls:
                    if p and p not in self._dir_pitfalls[norm_dir]:
                        self._dir_pitfalls[norm_dir].append(p)

    def get_all(self) -> list[str]:
        with self._lock:
            return list(self._pitfalls)

    def get_by_dir(self, directory: str) -> list[str]:
        with self._lock:
            norm_dir = os.path.normpath(directory)
            return list(self._dir_pitfalls.get(norm_dir, []))

    def format_context(self) -> str:
        items = self.get_all()
        if not items:
            return "（暂无易错点）"
        return "\n".join(f"- {p}" for p in items)


@dataclass
class KnowledgeFragment:
    """召回模式中收集的知识片段"""
    content: str
    heading_path: list[str]
    directory_path: str


class RetrievalKnowledgeRegistry:
    """线程安全的全局知识片段收集池，召回模式下所有子智能体共享"""

    def __init__(self):
        self._lock = threading.Lock()
        self._fragments: list[KnowledgeFragment] = []
        self._seen_dirs: set[str] = set()

    def add(self, fragment: KnowledgeFragment) -> bool:
        normalized = os.path.normpath(fragment.directory_path)
        with self._lock:
            if normalized in self._seen_dirs:
                return False
            self._seen_dirs.add(normalized)
            self._fragments.append(fragment)
            return True

    def get_all(self) -> list[KnowledgeFragment]:
        with self._lock:
            return list(self._fragments)


@dataclass
class RelationFragment:
    """关联展开命中的外部条款片段。

    数据来源：clause.json.references → resolvedClauses 引用图中被 LLM 判定相关的节点。
    chunk 模式下被切分为派生 KnowledgeChunk；retrieval/standard 模式下 inline 追加到对应 fragment。
    """
    policy_id: str
    clause_id: str
    clause_number: str             # 用于排序/章节展示，如 "2.1.3"；远程兜底时可能为空
    clause_full_name: str          # searchLabels[0] 或拼接的可读名
    heading_path: list[str]        # 完整层级路径，本地解析时按目录拼出，远程时退化为 [full_name]
    content: str                   # markdownified clauseContent
    highlighted: str               # 上一跳引用本节点时的 highlightedContent
    parent_assessment: str         # 上一层 LLM 判定的 reason 摘要（用于 trace + 下一跳上下文）
    hop_depth: int                 # 1=直接关联, 2=二跳, ...
    source: str                    # "local" | "remote" | "missing"
    parent_chunk_index: int        # chunk 模式：触发本次展开的原始 chunk index；副路径为 -1
    parent_dir: str                # 触发本次展开的原始 dir 绝对路径
    target_dir: str = ""           # 本条款在本地知识库中的 dir 绝对路径（远程兜底时为空）
    target_knowledge_root: str = ""  # 本条款所属 policy 的 knowledge_root 绝对路径（page_knowledge_dir 下一层）


class RelationRegistry:
    """线程安全的关联展开命中池。

    全局按 (policy_id, clause_id) 去重，避免跨 chunk / 跨 dir 重复 LLM 推理；
    同时按 parent_chunk_index 与 parent_dir 维护二级索引，便于：
      - chunk 主路径：派生 chunk 切分时回溯归属父 chunk；
      - retrieval/standard 副路径：把命中条款 inline 追加到对应 fragment.content 末尾。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._fragments: list[RelationFragment] = []
        self._seen_keys: set[tuple[str, str]] = set()
        self._by_chunk: dict[int, list[RelationFragment]] = {}
        self._by_dir: dict[str, list[RelationFragment]] = {}

    def add(self, fragment: RelationFragment) -> bool:
        key = (fragment.policy_id, fragment.clause_id)
        with self._lock:
            if key in self._seen_keys:
                return False
            self._seen_keys.add(key)
            self._fragments.append(fragment)
            self._by_chunk.setdefault(fragment.parent_chunk_index, []).append(fragment)
            if fragment.parent_dir:
                self._by_dir.setdefault(
                    os.path.normpath(fragment.parent_dir), []
                ).append(fragment)
            return True

    def get_by_chunk(self, chunk_index: int) -> list[RelationFragment]:
        with self._lock:
            return list(self._by_chunk.get(chunk_index, []))

    def get_by_dir(self, dir_path: str) -> list[RelationFragment]:
        with self._lock:
            return list(self._by_dir.get(os.path.normpath(dir_path), []))

    def get_all(self) -> list[RelationFragment]:
        with self._lock:
            return list(self._fragments)

    def has_any(self) -> bool:
        with self._lock:
            return bool(self._fragments)

    def has(self, policy_id: str, clause_id: str) -> bool:
        """快速判断 (policy_id, clause_id) 是否已注册。

        供 RelationCrawler BFS 跨 chunk 去重使用：当某个候选条款已经被其他
        chunk 的展开评估并注册过，再次出现在新 chunk 的 BFS 队列中时直接跳过，
        避免重复 LLM 评估调用。线程安全，只读。
        """
        with self._lock:
            return (policy_id, clause_id) in self._seen_keys
