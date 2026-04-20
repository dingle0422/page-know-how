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
