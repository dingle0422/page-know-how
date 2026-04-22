"""关联条款 Crawler：BFS 多跳、并发、上下文相关性判定、终止护栏。

输入：(source_chunk_index, source_dir, parent_assessment)
    - source_dir 必须含 clause.json；其 references 是抽取阶段已经预展开的
      跨条款/跨策略引用图（见 extractor/parser.resolve_references）。
    - parent_assessment 是触发本次展开的上一层 LLM 判定理由（chunk 模式下
      取自 chunk LLM 的 analysis 摘要，副路径取 _assess_* 的 conclusion 摘要）。

行为：
    1. 读 clause.json.references → 把每个 ref + 其 resolvedClauses 扁平化为 (policy_id, clause_id,
       highlightedContent) 候选队列（首跳）。
    2. 用 ClauseLocator 拉每个候选条款的真实内容（local-first，remote fallback）。
    3. 调 LLM (RELATION_RELEVANCE_PROMPT) 判定 is_relevant + should_descend，传入：
         - 用户问题
         - 上一层 assessment（首跳为 source 的 parent_assessment，深跳为父节点的 reason）
         - 上一层引用本节点时的 highlightedContent
         - 当前条款完整 markdown 内容
    4. is_relevant=True → 入 RelationFragment 注册到全局 RelationRegistry，并按 hop_depth < max_depth 向下递归展开。
    5. 子节点用线程池并发评估。
    6. 护栏：max_depth、max_nodes、(policy_id, clause_id) 全局去重、cycle 检测、
       missing/error 短路。

输出：本次 crawl 命中的扁平 RelationFragment 列表（按 BFS 顺序）。已 add 到 registry。
"""

import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

from llm.client import chat
from reasoner._registries import RelationFragment, RelationRegistry
from reasoner.v1.clause_locator import ClauseLocator
from reasoner.v1.prompts import RELATION_RELEVANCE_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class _CandidateRef:
    """BFS 队列单元：一次待评估的条款引用（已携带上层上下文）。"""
    policy_id: str
    clause_id: str
    highlighted: str
    parent_assessment: str
    hop_depth: int


class RelationCrawler:
    """有状态、线程安全。AgentGraph 持有单例，跨多个原始 chunk / 副路径节点共享。

    通过 RelationRegistry 全局去重 (policy_id, clause_id)，避免不同 chunk 命中同一条款时
    重复 LLM 评估。
    """

    def __init__(
        self,
        question: str,
        registry: RelationRegistry,
        locator: ClauseLocator,
        executor: ThreadPoolExecutor,
        vendor: str = "qwen3.5-122b-a10b",
        model: str = "Qwen3.5-122B-A10B",
        max_depth: int = 3,
        max_nodes: int = 50,
    ):
        self.question = question
        self.registry = registry
        self.locator = locator
        self.executor = executor
        self.vendor = vendor
        self.model = model
        self.max_depth = max(1, int(max_depth))
        self.max_nodes = max(1, int(max_nodes))

        # 单次 crawl 内的 visited（policy_id, clause_id）防止同一 source 内的环
        # 跨 source 的去重交给 RelationRegistry
        self._lock = threading.Lock()

    # ---------- 公共入口 ----------

    def crawl(
        self,
        source_chunk_index: int,
        source_dir: str,
        parent_assessment: str,
    ) -> list[RelationFragment]:
        """从 source_dir/clause.json 起步做 BFS。返回本次新增的 RelationFragment 列表。"""
        clause_json = os.path.join(source_dir, "clause.json")
        if not os.path.isfile(clause_json):
            logger.debug(f"[RelationCrawler] {source_dir} 无 clause.json，跳过")
            return []
        try:
            with open(clause_json, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"[RelationCrawler] 读取 clause.json 失败 {clause_json}: {e}")
            return []

        first_hop: list[_CandidateRef] = []
        for ref in raw.get("references", []) or []:
            highlighted = ref.get("highlightedContent", "") or ""
            for child in ref.get("resolvedClauses", []) or []:
                pid = child.get("policyId", "")
                cid = child.get("clauseId", "")
                if not pid or not cid:
                    continue
                if child.get("cycle") or child.get("missing"):
                    continue
                first_hop.append(_CandidateRef(
                    policy_id=pid,
                    clause_id=cid,
                    highlighted=highlighted,
                    parent_assessment=parent_assessment,
                    hop_depth=1,
                ))

        if not first_hop:
            return []

        new_fragments: list[RelationFragment] = []
        visited: set[tuple[str, str]] = set()
        node_budget = [self.max_nodes]  # 用 list 包装以便闭包内修改

        self._bfs_evaluate(
            queue=first_hop,
            source_chunk_index=source_chunk_index,
            source_dir=os.path.abspath(source_dir),
            visited=visited,
            node_budget=node_budget,
            new_fragments=new_fragments,
        )

        if new_fragments:
            logger.info(
                f"[RelationCrawler] source_dir={source_dir} 命中 {len(new_fragments)} 个关联条款"
                f"（max_depth={self.max_depth}, budget_used={self.max_nodes - node_budget[0]})"
            )
        return new_fragments

    # ---------- 内部：BFS 调度 ----------

    def _bfs_evaluate(
        self,
        queue: list[_CandidateRef],
        source_chunk_index: int,
        source_dir: str,
        visited: set[tuple[str, str]],
        node_budget: list[int],
        new_fragments: list[RelationFragment],
    ):
        """逐层 BFS：每一层并发评估当前批，把判定为 relevant 的节点按 hop_depth < max_depth
        条件扩展其子引用到下一层 queue。
        """
        current_layer = queue
        while current_layer and node_budget[0] > 0:
            # 同层内按 (policy_id, clause_id) 去重 + visited 去重 + budget 截断
            # 风险2修复：先查 RelationRegistry 是否已有该 key，有则跳过（避免重复 LLM 调用）
            dedup: list[_CandidateRef] = []
            for cand in current_layer:
                if node_budget[0] <= 0:
                    break
                key = (cand.policy_id, cand.clause_id)
                # 跨 chunk 去重：registry 已有的 key 直接跳过，不浪费 LLM 调用
                if self.registry.has(cand.policy_id, cand.clause_id):
                    continue
                with self._lock:
                    if key in visited:
                        continue
                    visited.add(key)
                dedup.append(cand)

            if not dedup:
                break

            futures = {
                self.executor.submit(self._evaluate_single, cand, source_chunk_index, source_dir): cand
                for cand in dedup
            }

            next_layer: list[_CandidateRef] = []
            for fut in as_completed(futures):
                cand = futures[fut]
                try:
                    fragment, descend_children = fut.result()
                except Exception as e:
                    logger.warning(
                        f"[RelationCrawler] 评估失败 policy={cand.policy_id} clause={cand.clause_id}: {e}"
                    )
                    continue
                if fragment is None:
                    continue
                # 注册（全局去重）；风险1修复：仅当真正注册成功时才扣 node_budget
                added = self.registry.add(fragment)
                if added:
                    node_budget[0] -= 1
                    new_fragments.append(fragment)
                # 是否扩展下一跳
                if descend_children and fragment.hop_depth < self.max_depth:
                    next_layer.extend(descend_children)

            current_layer = next_layer

        if node_budget[0] <= 0:
            logger.info(f"[RelationCrawler] 已达 max_nodes={self.max_nodes} 上限，停止扩展")

    # ---------- 内部：单节点评估 ----------

    def _evaluate_single(
        self,
        cand: _CandidateRef,
        source_chunk_index: int,
        source_dir: str,
    ) -> tuple[Optional[RelationFragment], list[_CandidateRef]]:
        """对单个候选条款：定位 → LLM 判定 → 返回 (fragment | None, 下一跳候选)。"""
        clause_dict, source = self.locator.locate(cand.policy_id, cand.clause_id)
        if clause_dict is None:
            logger.debug(
                f"[RelationCrawler] miss policy={cand.policy_id} clause={cand.clause_id} (source=missing)"
            )
            return None, []

        heading_label = " > ".join(clause_dict.get("heading_path") or []) or clause_dict.get("clause_full_name", "")

        prompt = RELATION_RELEVANCE_PROMPT.format(
            question=self.question,
            parent_assessment=cand.parent_assessment or "（无）",
            parent_highlighted=cand.highlighted or "（无）",
            policy_id=cand.policy_id,
            clause_id=cand.clause_id,
            heading_label=heading_label or "（无）",
            hop_depth=cand.hop_depth,
            current_content=clause_dict.get("content") or "（空）",
        )

        decision = _call_llm_json(prompt, vendor=self.vendor, model=self.model)
        if decision is None:
            return None, []
        is_relevant = bool(decision.get("is_relevant"))
        reason = (decision.get("reason") or "").strip()

        if not is_relevant:
            logger.debug(
                f"[RelationCrawler] reject hop={cand.hop_depth} policy={cand.policy_id} "
                f"clause={cand.clause_id}: {reason[:80]}"
            )
            return None, []

        fragment = RelationFragment(
            policy_id=cand.policy_id,
            clause_id=cand.clause_id,
            clause_number=clause_dict.get("clause_number", ""),
            clause_full_name=clause_dict.get("clause_full_name", ""),
            heading_path=list(clause_dict.get("heading_path") or []),
            content=clause_dict.get("content", ""),
            highlighted=cand.highlighted,
            parent_assessment=cand.parent_assessment,
            hop_depth=cand.hop_depth,
            source=source,
            parent_chunk_index=source_chunk_index,
            parent_dir=source_dir,
        )

        next_layer: list[_CandidateRef] = []
        # 只要 is_relevant=True，就按 hop_depth < max_depth 决定是否下钻
        if is_relevant:
            next_assessment = reason if reason else cand.parent_assessment
            for ref in clause_dict.get("references", []) or []:
                highlighted = ref.get("highlightedContent", "") or ""
                # 本地节点 ref 可能含 resolvedClauses；远程兜底节点没有 resolvedClauses，
                # 此时 ref 自身的 (policyId, clauseId) 即下一跳目标
                children = ref.get("resolvedClauses") or []
                if children:
                    for child in children:
                        pid = child.get("policyId", "")
                        cid = child.get("clauseId", "")
                        if not pid or not cid:
                            continue
                        if child.get("cycle") or child.get("missing"):
                            continue
                        next_layer.append(_CandidateRef(
                            policy_id=pid,
                            clause_id=cid,
                            highlighted=highlighted,
                            parent_assessment=next_assessment,
                            hop_depth=cand.hop_depth + 1,
                        ))
                else:
                    pid = ref.get("policyId", "")
                    cid = ref.get("clauseId", "")
                    if pid and cid:
                        next_layer.append(_CandidateRef(
                            policy_id=pid,
                            clause_id=cid,
                            highlighted=highlighted,
                            parent_assessment=next_assessment,
                            hop_depth=cand.hop_depth + 1,
                        ))

        return fragment, next_layer


def _call_llm_json(prompt: str, vendor: str, model: str) -> Optional[dict]:
    """复用 react_agent 中的轻量 JSON LLM 调用模式（剥 ```json 外壳 + json.loads）。"""
    try:
        response = chat(prompt, vendor=vendor, model=model)
    except Exception as e:
        logger.warning(f"[RelationCrawler] LLM 调用失败: {e}")
        return None
    cleaned = response.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            import json5
            return json5.loads(cleaned)
        except Exception as e:
            logger.warning(f"[RelationCrawler] LLM JSON 解析失败: {e}; raw={cleaned[:200]}")
            return None
