"""HighlightPrecheck：reasoning 启动时的"关联关键词一次性预判"。

目的：
    - chunk 相关性判定 / agent 探索判定**只会从问题出发找"直接相关"的父章节**，
      容易漏掉那些"父章节本身不直接相关，但它引用的外部条款是关键证据"的场景。
    - 例如："2.1.2.1.2 流通环节销售蔬菜" 的父 chunk 如果被判定无关，那它里面
      指向《蔬菜主要品种目录》的关联就也会跟着一起漏掉——即使《蔬菜主要品种目录》
      才是回答问题的关键。
    - 本模块在 reasoning 启动时与 skill 判定、chunk/agent 判定**并行**跑一次：
      把当前知识包里**所有外链 highlightedContent** 打成一张清单喂给 LLM，
      让它从关键词语义层面**主动**挑出"可能值得展开"的那些，然后立刻触发
      RelationCrawler 强制展开其目标条款（按 ref_filter 精确只展开选中的）。
    - 命中结果写入共享的 RelationRegistry；无论父 chunk 后续是否被判定相关，
      这些关联条款都会经由已有的渲染路径（_format_relation_fragment_text /
      _render_inline_relations / 以及 chunk 模式的 _build_chunk_orphan_part 兜底）
      进入 summary 链路，与其他 part 按 batch_size 规则统一参与 BATCH_SUMMARY
      或直进 final merge。

设计要点：
    - 候选按 (highlighted, target_policy_id, target_clause_id) 全局去重，
      避免同一外链因多父章节重复出现而挤占 LLM 上下文。
    - 只触发 first-hop BFS 种子；后续深跳仍受 RelationCrawler 的 max_depth /
      max_nodes / RelationRegistry 去重约束，不会引发爆炸式展开。
    - 返回统计信息供 trace 使用。
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

from llm.client import chat
from utils.verbose_logger import step_scope
from reasoner.v1.chunk_builder import build_parent_location_label
from reasoner.v1.prompts import HIGHLIGHT_PRECHECK_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class HighlightCandidate:
    """单条高亮外链候选（用于 LLM 预判 + 后续强制展开的 ref_filter）。"""
    index: int                      # 1-based 展示编号
    parent_dir: str                 # 含 clause.json 的父目录绝对路径
    parent_label: str               # 业务化父章节定位（知识名 > ... > 末级章节）
    highlighted: str                # 关键词（ref.highlightedContent）
    target_policy_id: str           # 目标条款 policy id
    target_clause_id: str           # 目标条款 clause id


def collect_highlight_candidates(knowledge_root: str) -> list[HighlightCandidate]:
    """扫描 knowledge_root 下所有 clause.json，展开 references 为候选列表。

    去重键：(highlighted, target_policy_id, target_clause_id)。同一个关键词/目标
    组合即使出现在多个父章节也只保留首次遇到的那个作为代表。
    """
    candidates: list[HighlightCandidate] = []
    if not knowledge_root or not os.path.isdir(knowledge_root):
        return candidates

    seen: set[tuple[str, str, str]] = set()
    idx = 0

    for root, _dirs, files in os.walk(knowledge_root):
        if "clause.json" not in files:
            continue
        cj = os.path.join(root, "clause.json")
        try:
            with open(cj, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.debug(f"[HighlightPrecheck] 读取 {cj} 失败: {e}")
            continue

        refs = raw.get("references") or []
        if not refs:
            continue

        parent_label = build_parent_location_label(knowledge_root, root)

        for ref in refs:
            highlighted = (ref.get("highlightedContent") or "").strip()
            if not highlighted:
                continue
            children = ref.get("resolvedClauses") or []
            targets = children if children else [ref]

            for tgt in targets:
                if tgt.get("cycle") or tgt.get("missing"):
                    continue
                pid = (tgt.get("policyId") or "").strip()
                cid = (tgt.get("clauseId") or "").strip()
                if not pid or not cid:
                    continue
                key = (highlighted, pid, cid)
                if key in seen:
                    continue
                seen.add(key)
                idx += 1
                candidates.append(HighlightCandidate(
                    index=idx,
                    parent_dir=os.path.abspath(root),
                    parent_label=parent_label,
                    highlighted=highlighted,
                    target_policy_id=pid,
                    target_clause_id=cid,
                ))

    return candidates


def _build_prompt(question: str, candidates: list[HighlightCandidate]) -> str:
    lines = []
    for c in candidates:
        loc = c.parent_label or c.parent_dir
        lines.append(f"[{c.index}] 「{c.highlighted}」 — 位于：{loc}")
    candidate_block = "\n".join(lines)
    return HIGHLIGHT_PRECHECK_PROMPT.format(
        question=question,
        candidate_block=candidate_block,
        total=len(candidates),
    )


def _parse_selected(response: str, max_idx: int) -> tuple[list[int], str]:
    """解析 LLM 返回的 selected / reason，宽松处理 code fence 与数字格式。"""
    cleaned = (response or "").strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    data: Optional[dict] = None
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            import json5
            data = json5.loads(cleaned)
        except Exception:
            logger.warning(
                f"[HighlightPrecheck] 解析 LLM JSON 失败，raw={cleaned[:200]}"
            )
            return [], ""

    if not isinstance(data, dict):
        return [], ""

    selected_raw = data.get("selected") or []
    reason = str(data.get("reason") or "").strip()

    normalized: list[int] = []
    if isinstance(selected_raw, list):
        for v in selected_raw:
            try:
                i = int(v)
            except (ValueError, TypeError):
                continue
            if 1 <= i <= max_idx and i not in normalized:
                normalized.append(i)

    return normalized, reason


class HighlightPrecheck:
    """一次性 LLM 判定 + 强制触发 RelationCrawler 展开。

    线程安全假设：本类在单线程中被调用一次（作为 AgentGraph 启动时的一个并行任务），
    内部对 RelationCrawler 的调用依赖 crawler 自身的线程安全（RelationRegistry 全局锁）。
    """

    def __init__(
        self,
        question: str,
        knowledge_root: str,
        crawler,
        vendor: str,
        model: str,
    ):
        self.question = question
        self.knowledge_root = knowledge_root
        self.crawler = crawler
        self.vendor = vendor
        self.model = model
        self.last_stats: dict = {}

    def run(self) -> dict:
        """执行一次预判 + 强制展开。返回统计字典：

            {
                "total_candidates": 候选关键词总数,
                "selected": LLM 选中的序号数量,
                "fragments": 本次新增的 RelationFragment 数量,
                "reason": LLM 给出的整体挑选理由（截断后）,
            }
        """
        result = {
            "total_candidates": 0,
            "selected": 0,
            "fragments": 0,
            "reason": "",
        }
        if self.crawler is None:
            logger.debug("[HighlightPrecheck] crawler=None，跳过")
            self.last_stats = result
            return result

        candidates = collect_highlight_candidates(self.knowledge_root)
        result["total_candidates"] = len(candidates)
        if not candidates:
            logger.info(
                "[HighlightPrecheck] 当前知识根下无外链 highlightedContent，跳过"
            )
            self.last_stats = result
            return result

        prompt = _build_prompt(self.question, candidates)
        logger.info(
            f"[HighlightPrecheck] 候选关键词 {len(candidates)} 条，"
            f"prompt 长度 {len(prompt)} 字符，发起 LLM 一次性判定"
        )

        try:
            with step_scope("highlight_precheck"):
                response = chat(prompt, vendor=self.vendor, model=self.model)
        except Exception as e:
            logger.warning(f"[HighlightPrecheck] LLM 调用失败: {e}")
            self.last_stats = result
            return result

        selected_idx, reason = _parse_selected(response, max_idx=len(candidates))
        result["selected"] = len(selected_idx)
        result["reason"] = reason

        if not selected_idx:
            logger.info(
                f"[HighlightPrecheck] LLM 未选中任何关键词，reason={reason[:120]}"
            )
            self.last_stats = result
            return result

        selected = [candidates[i - 1] for i in selected_idx]
        logger.info(
            f"[HighlightPrecheck] LLM 选中 {len(selected)}/{len(candidates)} 条，"
            f"开始强制触发关联展开。reason={reason[:120]}"
        )

        # 按 parent_dir 分组，减少 crawl 调用次数：一个父 clause 只跑一次 BFS 种子。
        by_parent: dict[str, list[HighlightCandidate]] = {}
        for c in selected:
            by_parent.setdefault(c.parent_dir, []).append(c)

        assessment_base = (
            "HighlightPrecheck 判定这些高亮关键词与用户问题相关，强制展开其关联条款"
        )
        assessment = (
            f"{assessment_base}。LLM 简述：{reason[:200]}"
            if reason else
            f"{assessment_base}。"
        )

        total_frags = 0
        for parent_dir, group in by_parent.items():
            allowed = {
                (c.highlighted, c.target_policy_id, c.target_clause_id)
                for c in group
            }

            def _filter(h, pid, cid, _allowed=allowed):
                return (h, pid, cid) in _allowed

            try:
                new_frags = self.crawler.crawl(
                    source_chunk_index=-1,
                    source_dir=parent_dir,
                    parent_assessment=assessment,
                    ref_filter=_filter,
                )
            except Exception as e:
                logger.warning(
                    f"[HighlightPrecheck] crawl 失败 parent_dir={parent_dir}: {e}"
                )
                continue
            total_frags += len(new_frags)

        result["fragments"] = total_frags
        logger.info(
            f"[HighlightPrecheck] 完成：新增 {total_frags} 个 RelationFragment "
            f"(跨 {len(by_parent)} 个父章节，已注册到 RelationRegistry)"
        )
        self.last_stats = result
        return result
