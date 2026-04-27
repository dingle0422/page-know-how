"""跨策略条款定位器：服务于 v1 关联展开（RelationCrawler）。

核心职责：
    给定 (policy_id, clause_id) → 返回标准化 clause dict（含 markdownified 内容、
    完整 heading_path、references 列表、本地 dir 绝对路径、来源标记）。

查找优先级：
    1. 进程级缓存（命中即返回，避免重复 IO/网络）
    2. 本地：通过 page_knowledge/_policy_index.json → entry.clauses[clause_id]
       → 加载本地 clause.json，使用相对路径切片重建 heading_path
    3. 远程：调用 extractor.parser._fetch_single_clause（DEFAULT_CLAUSE_API_URL），
       同步把 clauseContent HTML 转为 markdown，heading_path 退化为
       [clauseFullName]（远程接口无层级上下文）
    4. 仍然找不到：返回 (None, "missing")

线程安全：locate 用 RLock 保护缓存；远程请求本身在 _fetch_single_clause 内部走 requests
（无共享状态）。AgentGraph 在初始化时构造单例并在 RelationCrawler 多线程间共享。
"""

import os
import json
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)


def _safe_import_remote_fetcher():
    """延迟导入避免与 extractor 包循环依赖。"""
    from extractor.parser import _fetch_single_clause, _convert_html_to_markdown, extract_clause_references
    from extractor.parser import DEFAULT_CLAUSE_API_URL
    return _fetch_single_clause, _convert_html_to_markdown, extract_clause_references, DEFAULT_CLAUSE_API_URL


class ClauseLocator:
    """支持 (policy_id, clause_id) → clause dict 的本地+远程定位器。

    返回的 clause dict 形如：
        {
            "policy_id": str,
            "clause_id": str,
            "clause_number": str,
            "clause_full_name": str,
            "heading_path": list[str],
            "content": str,                # markdown
            "highlighted": str,            # 由调用方在外部填入（locate 不知道上一跳上下文）
            "references": list[dict],      # 同 clause.json.references 结构（含 resolvedClauses）
            "dir_abspath": str,            # 本地命中时为目录绝对路径；远程时为 ""
        }
    """

    _MISSING_SENTINEL = object()

    def __init__(
        self,
        page_knowledge_dir: str,
        policy_index_path: str,
        api_url: Optional[str] = None,
        remote_timeout: float = 5.0,
    ):
        self.page_knowledge_dir = os.path.abspath(page_knowledge_dir)
        self.policy_index_path = policy_index_path
        self.remote_timeout = remote_timeout

        _fetch, _md, _extract_refs, default_api = _safe_import_remote_fetcher()
        self._fetch_remote = _fetch
        self._html_to_md = _md
        self._extract_refs = _extract_refs
        self.api_url = api_url or default_api

        self._lock = threading.RLock()
        self._cache: dict[tuple[str, str], object] = {}
        self._policy_index: dict[str, dict] | None = None

    # ---------- 内部：policy index 懒加载 ----------

    def _get_policy_index(self) -> dict[str, dict]:
        """懒加载并 cache 整份 _policy_index.json（关联展开期间不需要热更新）。"""
        if self._policy_index is not None:
            return self._policy_index
        with self._lock:
            if self._policy_index is not None:
                return self._policy_index
            from extractor.policy_index import load_index
            self._policy_index = load_index(self.policy_index_path)
            return self._policy_index

    # ---------- 内部：本地解析 ----------

    def _try_local(self, policy_id: str, clause_id: str) -> Optional[dict]:
        """命中本地索引 → 读 clause.json → 组装标准化 dict；任何环节失败返回 None。"""
        index = self._get_policy_index()
        entry = index.get(policy_id)
        if not entry:
            return None
        root_dirname = entry.get("root") or ""
        clauses = entry.get("clauses") or {}
        relpath = clauses.get(clause_id)
        if not root_dirname or not relpath:
            return None
        clause_dir = os.path.join(self.page_knowledge_dir, root_dirname, *relpath.split("/"))
        clause_json_path = os.path.join(clause_dir, "clause.json")
        if not os.path.isfile(clause_json_path):
            return None
        try:
            with open(clause_json_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"[ClauseLocator] 本地 clause.json 解析失败 {clause_json_path}: {e}")
            return None

        heading_path = self._heading_path_from_relpath(relpath)
        # 复用条款抽取阶段已经写盘的 markdown（knowledge.md 与 clause.json 同目录）
        knowledge_md_path = os.path.join(clause_dir, "knowledge.md")
        content_md = ""
        if os.path.isfile(knowledge_md_path):
            try:
                with open(knowledge_md_path, "r", encoding="utf-8") as f:
                    content_md = f.read().strip()
            except OSError:
                content_md = ""
        if not content_md:
            content_md = self._html_to_md(raw.get("clauseContent", "") or "")

        return {
            "policy_id": policy_id,
            "clause_id": clause_id,
            "clause_number": raw.get("clauseNumber") or "",
            "clause_full_name": raw.get("clauseName") or self._guess_full_name(raw),
            "heading_path": heading_path,
            "content": content_md,
            "highlighted": "",
            "references": list(raw.get("references") or []),
            "dir_abspath": os.path.abspath(clause_dir),
        }

    @staticmethod
    def _heading_path_from_relpath(relpath: str) -> list[str]:
        """目录 relpath '1_xxx/1.3_yyy/1.3.5_zzz' → ['xxx', 'yyy', 'zzz']。

        前缀编号（1, 1.3, 1.3.5）由 LLM 输出 relevant_headings 时另行展示，这里只保留可读名。
        """
        out: list[str] = []
        for seg in relpath.replace("\\", "/").split("/"):
            if not seg:
                continue
            if "_" in seg:
                _, name = seg.split("_", 1)
                out.append(name.strip())
            else:
                out.append(seg)
        return out

    @staticmethod
    def _guess_full_name(raw: dict) -> str:
        labels = raw.get("searchLabels") or []
        if labels:
            return str(labels[0])
        return raw.get("clauseName") or raw.get("clauseId") or ""

    # ---------- 内部：远程兜底 ----------

    def _try_remote(self, policy_id: str, clause_id: str) -> Optional[dict]:
        # 把 self.remote_timeout 透传到 extractor.parser._fetch_single_clause,
        # 防止下游 requests 在 read 阶段永久 hang、拖住 executor 线程不退出。
        # 下游接受 float（按 read 处理、connect 自动取 min(10, T)）或 (connect, read) tuple。
        raw = self._fetch_remote(policy_id, clause_id, self.api_url, timeout=self.remote_timeout)
        if raw is None:
            return None
        html = raw.get("clauseContent", "") or ""
        content_md = self._html_to_md(html)
        refs = self._extract_refs(html)  # 远程兜底节点的 references 不递归展开（避免运行时雪崩）
        full_name = self._guess_full_name(raw)
        return {
            "policy_id": policy_id,
            "clause_id": clause_id,
            "clause_number": raw.get("clauseNumber") or "",
            "clause_full_name": full_name,
            "heading_path": [full_name] if full_name else [],
            "content": content_md,
            "highlighted": "",
            "references": refs,
            "dir_abspath": "",
        }

    # ---------- 对外接口 ----------

    def locate(self, policy_id: str, clause_id: str) -> tuple[Optional[dict], str]:
        """返回 (clause_dict | None, source)，source ∈ {'local','remote','missing'}。

        缓存策略：locate 结果（含 missing）按 (policy_id, clause_id) 全局 memo，
        relation crawl 期间命中复用，避免对同一节点的重复 IO/网络。
        """
        if not policy_id or not clause_id:
            return None, "missing"
        key = (policy_id, clause_id)
        with self._lock:
            cached = self._cache.get(key, self._MISSING_SENTINEL)
        if cached is not self._MISSING_SENTINEL:
            if cached is None:
                return None, "missing"
            payload, src = cached  # type: ignore
            return dict(payload), src

        local = self._try_local(policy_id, clause_id)
        if local is not None:
            with self._lock:
                self._cache[key] = (local, "local")
            return dict(local), "local"

        remote = self._try_remote(policy_id, clause_id)
        if remote is not None:
            with self._lock:
                self._cache[key] = (remote, "remote")
            return dict(remote), "remote"

        with self._lock:
            self._cache[key] = None
        return None, "missing"
