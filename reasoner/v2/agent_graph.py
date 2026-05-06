import os
import re
import json
import asyncio
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable

from llm.client import chat
from utils.verbose_logger import agent_scope, step_scope
from utils.helpers import split_think_block
from reasoner.v2.react_agent import ReactAgent, AgentResult
from reasoner.v2.prompts import (
    SUMMARY_PROMPT,
    SUMMARY_AND_CLEAN_PROMPT,
    SUMMARY_AND_CLEAN_THINK_PROMPT,
    SUMMARY_AND_CLEAN_THINK_HTML_PROMPT,
    SUMMARY_EXTRACT_SYSTEM_PROMPT,
    BATCH_REDUCE_SYSTEM_PROMPT,
    SUMMARY_ANSWER_SYSTEM_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
    CLEAN_ANSWER_PROMPT,
    BATCH_SUMMARY_PROMPT,
    BATCH_MERGE_PROMPT,
    BATCH_MERGE_AND_CLEAN_PROMPT,
    BATCH_MERGE_AND_CLEAN_THINK_PROMPT,
    BATCH_MERGE_AND_CLEAN_THINK_HTML_PROMPT,
    RETRIEVAL_SUMMARY_PROMPT,
    RETRIEVAL_SUMMARY_AND_CLEAN_PROMPT,
    RETRIEVAL_SUMMARY_AND_CLEAN_THINK_PROMPT,
    RETRIEVAL_SUMMARY_AND_CLEAN_THINK_HTML_PROMPT,
    RETRIEVAL_BATCH_SUMMARY_PROMPT,
    RETRIEVAL_BATCH_MERGE_PROMPT,
    RETRIEVAL_BATCH_MERGE_AND_CLEAN_PROMPT,
    RETRIEVAL_BATCH_MERGE_AND_CLEAN_THINK_PROMPT,
    RETRIEVAL_BATCH_MERGE_AND_CLEAN_THINK_HTML_PROMPT,
    BATCH_SUMMARY_SYSTEM_PROMPT,
    BATCH_MERGE_AND_CLEAN_THINK_SYSTEM_PROMPT,
    BATCH_MERGE_AND_CLEAN_THINK_HTML_SYSTEM_PROMPT,
    CHUNK_REASONING_PROMPT,
    CHUNK_REASONING_WITH_PITFALLS_PROMPT,
    CHUNK_REASONING_SYSTEM_PROMPT,
    CHUNK_REASONING_WITH_PITFALLS_SYSTEM_PROMPT,
    ALL_IN_ANSWER_PROMPT,
    ALL_IN_ANSWER_HTML_PROMPT,
    ALL_IN_ANSWER_SYSTEM_PROMPT,
    ALL_IN_ANSWER_HTML_SYSTEM_PROMPT,
    PURE_MODEL_REQUEST_PROMPT,
    PURE_MODEL_REFERENCE_EXTRACT_INSTRUCTIONS,
    PURE_MODEL_REFERENCE_ANSWER_INSTRUCTIONS,
    ANSWER_REFINE_PROMPT,
    ANSWER_REFINE_SYSTEM_PROMPT,
)
from reasoner.v2.chunk_builder import (
    build_knowledge_chunks,
    build_parent_location_label,
    build_target_location_label,
    split_relations_into_chunks,
    natural_dir_sort_key,
    KnowledgeChunk,
)
from reasoner.v2.clause_locator import ClauseLocator
from reasoner.v2.relation_crawler import RelationCrawler
from reasoner.v2.highlight_precheck import HighlightPrecheck
from reasoner.v2.reduce_pipeline import ReducePart, ReducePipeline

from reasoner._registries import (
    ExploredRegistry,
    PitfallsRegistry,
    KnowledgeFragment,
    RetrievalKnowledgeRegistry,
    RelationRegistry,
    RelationFragment,
)

from skills import (
    SkillRunner,
    SkillResultRegistry,
    SkillRecord,
)
from reasoner.v2.skill_evaluator import evaluate_and_run, select_extra_skills

logger = logging.getLogger(__name__)


def _try_loads_to_dict(s: str) -> dict | None:
    """容忍多种 JSON 变体（标准 / json5 / 内嵌 markdown 围栏 / raw 内首个 {} 块）。"""
    if not s or not s.strip():
        return None
    s = s.strip()
    if s.startswith("```json"):
        s = s[len("```json"):].strip()
    elif s.startswith("```"):
        s = s[3:].strip()
    if s.endswith("```"):
        s = s[:-3].strip()

    try:
        o = json.loads(s)
        if isinstance(o, dict):
            return o
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        import json5  # type: ignore[import-not-found]

        o = json5.loads(s)
        if isinstance(o, dict):
            return o
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if m:
        try:
            o = json.loads(m.group(0))
            if isinstance(o, dict):
                return o
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _validate_think_answer_json(raw: str) -> dict | None:
    """严格校验最终 think_mode 输出：必须是 dict，且 think、answer 两字段都是非空字符串。

    - 兼容历史字段名：think 缺失时回落到 analysis；answer 缺失时回落到 concise_answer。
    - 通过返回标准化 {"think": str, "answer": str}；任一关键校验不过返回 None。
    """
    parsed = _try_loads_to_dict(raw)
    if parsed is None:
        return None
    think = parsed.get("think")
    if not (isinstance(think, str) and think.strip()):
        legacy_think = parsed.get("analysis")
        if isinstance(legacy_think, str) and legacy_think.strip():
            think = legacy_think
        else:
            return None
    answer = parsed.get("answer")
    if not (isinstance(answer, str) and answer.strip()):
        legacy_answer = parsed.get("concise_answer")
        if isinstance(legacy_answer, str) and legacy_answer.strip():
            answer = legacy_answer
        else:
            return None
    return {"think": think.strip(), "answer": answer.strip()}


def _extract_answer_from_think_answer_json(raw: str) -> str | None:
    """从最终 think/answer JSON 中提取 answer 字段（兼容旧 analysis/concise_answer 字段）；
    解析失败或字段全空返回 None。"""
    parsed = _try_loads_to_dict(raw)
    if not parsed:
        return None
    ans = parsed.get("answer")
    if isinstance(ans, str) and ans.strip():
        return ans.strip()
    legacy = parsed.get("concise_answer")
    if isinstance(legacy, str) and legacy.strip():
        return legacy.strip()
    return None


# ---- HTML 标签解析所需的基础正则（拆成"开/闭标签独立匹配"，便于按位置切片） ----
# 锚点：</think>(可选空白)<answer>，是 think→answer 的硬边界。
# 即使 think 内复述了 prompt 给出的 schema（含 <think>/</think>/<answer>/</answer>
# 字面量），真正的边界一定是字符串里**最后一处**紧贴的 </think>\s*<answer>。
_HTML_BOUNDARY_RE = re.compile(
    r"</think\s*>\s*<answer(?:\s[^>]*)?>",
    re.IGNORECASE,
)
_HTML_THINK_OPEN_RE = re.compile(r"<think(?:\s[^>]*)?>", re.IGNORECASE)
_HTML_THINK_CLOSE_RE = re.compile(r"</think\s*>", re.IGNORECASE)
_HTML_ANSWER_OPEN_RE = re.compile(r"<answer(?:\s[^>]*)?>", re.IGNORECASE)
_HTML_ANSWER_CLOSE_RE = re.compile(r"</answer\s*>", re.IGNORECASE)

# answer body 实质字符数下限：低于该值视作 schema 复述（如 `...`、`……`、
# `示例答案` 等），降级到下一档；阈值 20 远低于业务侧任何合规客服回答的最低字数，
# 不会误杀真实答案。`\w` 在 Python re 默认 unicode 模式下会匹配 CJK 字符，
# 足够覆盖中英文混排正文。
_MIN_ANSWER_CONTENT_CHARS = 20
_ANSWER_CONTENT_CHAR_RE = re.compile(r"\w")


def _is_substantive_answer_body(text: str) -> bool:
    """answer body 是否包含足量实质字符（剔除空白/标点/省略号后）。

    用于 HTML 解析 tier 1/2 命中后过滤掉"schema 复述"误命中，例如：
        <answer>...</answer>
        <answer>面向用户表述</answer>
        <answer>……</answer>
    这类 body 实质字符数都 ≤ 20，会被判为不合格、降级到下一档。
    """
    if not text:
        return False
    return len(_ANSWER_CONTENT_CHAR_RE.findall(text)) > _MIN_ANSWER_CONTENT_CHARS


def _take_answer_body(body: str, answer_open_end: int) -> str:
    """从 <answer> 开标签结束位置往后取 answer body：
    优先到最近的 </answer>，无闭合则到 EOF（兼容 deepseek 偶发末尾漏闭合的情况）。
    """
    rest = body[answer_open_end:]
    cm = _HTML_ANSWER_CLOSE_RE.search(rest)
    if cm:
        return rest[:cm.start()].strip()
    return rest.strip()


def _extract_think_segment(section: str, fallback_think: str) -> str:
    """从 think 段（边界 </think> 之前的内容）抽取 think 文本。

    - 若 section 以 <think> 开标签开头（含前导空白），认为 body 内含完整 <think>...，
      剥离开标签后即可作为 think；fallback_think 已被吸收，此时无需再拼接。
    - 否则视为：上游 split_think_block 已把开头的 <think>...</think> 切走、
      fallback_think 就是被切走那段，section 则是被切走之后剩下的 think 后半截；
      两者拼接才是完整 think（典型场景：think 内复述 schema 含字面量 </think>，
      被 split_think_block 的非贪婪匹配在 schema 处提前截断）。
    """
    section = section or ""
    leading_open = re.match(r"\A\s*<think(?:\s[^>]*)?>", section, flags=re.IGNORECASE)
    if leading_open:
        return section[leading_open.end():].strip()
    fb = (fallback_think or "").strip()
    sec = section.strip()
    if fb and sec:
        return f"{fb}\n{sec}"
    return fb or sec


def _parse_think_answer_html(body: str, fallback_think: str = "") -> dict | None:
    """从 body 中尽力提取 <think>...</think> + <answer>...</answer>，按两档兜底逐档尝试。

    设计动机：deepseek 等模型在 HTML 模式下偶尔会在 think 内复述 prompt 给出的输出
    格式 schema（含 `<think>...</think><answer>...</answer>` 字面量），导致简单的
    "第一个完整 <answer> 块"匹配会落到 schema 复述上，把真实答案截成 `...`，
    think 也会被对应截断。本函数按以下优先级处理：

      Tier 1 — 锚点边界（最稳健）：
        在 body 中找**最后一处** `</think>\\s*<answer>`，作为 think→answer 的硬边界。
        即使 think 内复述了 schema，真实边界一定是最后一处紧贴的 </think><answer>。
        命中后 answer 取该 <answer> 到最近 </answer>（或 EOF）。
        answer body 必须通过 `_is_substantive_answer_body` 长度判定（>20 实质字符），
        否则视作 schema 复述误命中，降级到 tier 2。

      Tier 2 — 末尾配对（容忍 boundary 缺失/带文字间隔）：
        没有 </think><answer> 紧贴或 tier 1 长度不达标时，倒序遍历所有 <answer>
        开标签，逐个尝试「<answer> 开标签 → 紧随 </answer> 或 EOF」作为候选 answer，
        通过长度判定即接受。think 取候选 <answer> 之前最后一个 </think> 之前的内容
        （与 fallback_think 配合 `_extract_think_segment` 拼接补齐）。
        覆盖原"开标签到 EOF"漏闭合形态（N=10 实测约 10%），无需独立 tier。

      全失败返回 None，由上层走 coercion 兜底。

    fallback_think 兼容上游 split_think_block 已切走 body 开头 <think>...</think>
    的形态：body 内若仍残留完整 <think>...</think> 片段则优先使用 body 内的内容；
    否则与 fallback_think 拼接得到完整 think。
    """
    if not body:
        return None

    # ===== Tier 1: 锚点边界 </think>\s*<answer>（取最后一处） =====
    boundaries = list(_HTML_BOUNDARY_RE.finditer(body))
    if boundaries:
        b = boundaries[-1]
        # 在 boundary match 内重新定位 </think> 与 <answer> 子段（按构造永远命中），
        # 转换为 body 中的绝对位置。
        think_close_local = _HTML_THINK_CLOSE_RE.search(b.group(0))
        answer_open_local = _HTML_ANSWER_OPEN_RE.search(b.group(0))
        think_close_start = b.start() + think_close_local.start()
        answer_open_end = b.start() + answer_open_local.end()
        candidate_answer = _take_answer_body(body, answer_open_end)
        if _is_substantive_answer_body(candidate_answer):
            think = _extract_think_segment(body[:think_close_start], fallback_think)
            return {"think": think, "answer": candidate_answer}
        # 锚点的 answer body 不够实质（疑似 schema 复述误命中），降级到 tier 2

    # ===== Tier 2: 末尾配对 — 倒序遍历 <answer> 开标签，选第一个 body 通过长度判定的 =====
    answer_opens = list(_HTML_ANSWER_OPEN_RE.finditer(body))
    for am in reversed(answer_opens):
        candidate = _take_answer_body(body, am.end())
        if not _is_substantive_answer_body(candidate):
            continue
        # 选定该 <answer>：在它起点之前找最后一个 </think>，之前的内容作为 think 段
        pre = body[:am.start()]
        closes = list(_HTML_THINK_CLOSE_RE.finditer(pre))
        section = pre[:closes[-1].start()] if closes else pre
        think = _extract_think_segment(section, fallback_think)
        return {"think": think, "answer": candidate}

    return None


def _mirror_fill_think_answer(
    think: str, answer: str, full_text: str = ""
) -> tuple[str, str]:
    """coercion 兜底专用的 think/answer 互填策略，保证下游字段尽量都非空。

    填充规则：
      - 仅 think 非空 → answer 复用 think 内容（让用户至少看到一段内容）
      - 仅 answer 非空 → think 复用 answer 内容（保持 think_mode 字段非空）
      - 两者都非空 → 原值不变
      - 两者都空：
          * full_text 非空 → 用 full_text 同时填入 think 和 answer
            （覆盖"模型只输出空 JSON `{}`"或 `<answer></answer>` 这类极端形态）
          * full_text 也为空 → 维持双空字符串（罕见的模型完全空响应场景）
    """
    think = (think or "").strip()
    answer = (answer or "").strip()
    if think and not answer:
        return think, think
    if answer and not think:
        return answer, answer
    if not think and not answer:
        ft = (full_text or "").strip()
        if ft:
            return ft, ft
        return "", ""
    return think, answer


def _coerce_to_think_answer_json(body: str, fallback_think: str = "") -> str:
    """三层兜底的最末层：模型两次都没有给出可解析的 think/answer 结构时，
    把现有内容硬封装成下游统一的 think/answer JSON 字符串，确保 app.py 解析侧
    能拿到稳定的字段。

    填充策略（保证 think 与 answer 字段尽量都非空）：
      1) body 本身是 JSON dict（含 think/analysis、answer/concise_answer 任一非空字段）：
         提取后 mirror-fill 互填空字段。注意此处比 _validate_think_answer_json 宽松——
         主校验路径要求两字段都非空（防止把半成品当成功），但兜底层只要有"任意一个
         字段有值"就尽力挽救。
      2) body 是 HTML <think>/<answer>（含未闭合 <answer> 兜底，think 缺失时回落到
         fallback_think）：提取后 mirror-fill 互填。
      3) 最末兜底：think = fallback_think、answer = body 全文；mirror-fill 互填；
         若两者都为空且 fallback_think+body 拼起来非空，用拼接全文覆盖两个字段。
    """
    full_text = (body or "").strip() or (fallback_think or "").strip()

    parsed_dict = _try_loads_to_dict(body)
    if parsed_dict:
        think_val = parsed_dict.get("think")
        if not (isinstance(think_val, str) and think_val.strip()):
            legacy = parsed_dict.get("analysis")
            think_val = legacy if isinstance(legacy, str) else ""
        answer_val = parsed_dict.get("answer")
        if not (isinstance(answer_val, str) and answer_val.strip()):
            legacy = parsed_dict.get("concise_answer")
            answer_val = legacy if isinstance(legacy, str) else ""
        think_val = think_val.strip() if isinstance(think_val, str) else ""
        answer_val = answer_val.strip() if isinstance(answer_val, str) else ""
        if think_val or answer_val:
            think, answer = _mirror_fill_think_answer(think_val, answer_val, full_text=full_text)
            return json.dumps({"think": think, "answer": answer}, ensure_ascii=False)

    parsed_html = _parse_think_answer_html(body, fallback_think=fallback_think)
    if parsed_html:
        think, answer = _mirror_fill_think_answer(
            parsed_html["think"], parsed_html["answer"], full_text=full_text
        )
        return json.dumps({"think": think, "answer": answer}, ensure_ascii=False)

    think, answer = _mirror_fill_think_answer(
        fallback_think or "", body or "", full_text=full_text
    )
    return json.dumps({"think": think, "answer": answer}, ensure_ascii=False)


@dataclass
class _OrderedSlot:
    """流式 chunk 调度的状态机单元（每个原始 chunk 一份）。

    生命周期 (按 Event 触发顺序)：
        self_done    ← 原始 chunk LLM 完成
        relation_done← 关联展开（crawl + 派生 chunk 切分）完成；不需要展开则与 self_done 同时 set
        derived_done ← 所有派生 chunk LLM 完成；无派生则与 relation_done 同时 set

    调度线程按 slot 顺序 wait derived_done，依次把 self + derived parts 喂给 batch summary。
    """
    parent_index: int
    self_chunk: KnowledgeChunk | None = None
    self_result: dict | None = None
    derived_chunks: list[KnowledgeChunk] = field(default_factory=list)
    derived_results: dict[int, dict] = field(default_factory=dict)  # derived_seq -> result
    relation_fragments: list[RelationFragment] = field(default_factory=list)

    self_done: threading.Event = field(default_factory=threading.Event)
    relation_done: threading.Event = field(default_factory=threading.Event)
    derived_done: threading.Event = field(default_factory=threading.Event)

    pending_derived: int = 0
    completed_derived: int = 0
    derived_lock: threading.Lock = field(default_factory=threading.Lock)


class AgentGraph:
    """管理子智能体的衍生、并行执行和结果汇聚（v1：无 BacktrackIntent 机制）"""

    def __init__(
        self,
        question: str,
        knowledge_root: str,
        max_rounds: int = 5,
        vendor: str = "qwen3.5-122b-a10b",
        model: str = "Qwen3.5-122B-A10B",
        clean_answer: bool = False,
        summary_batch_size: int = 0,
        retrieval_mode: bool = False,
        check_pitfalls: bool = False,
        chunk_size: int = 0,
        enable_skills: bool = True,
        summary_clean_answer: bool = False,
        answer_system_prompt: str | None = None,
        think_mode: bool = False,
        last_think: bool = False,
        enable_relations: bool = False,
        relation_max_depth: int = 5,
        relation_max_nodes: int = 50,
        relation_workers: int = 8,
        relation_remote_timeout: float = 5.0,
        relations_expansion_mode: str = "all",
        summary_pipeline_mode: str = "layered",
        reduce_max_part_depth: int = 5,
        page_knowledge_dir: str | None = None,
        policy_index_path: str | None = None,
        pure_model_result: bool = False,
        answer_refine: bool = False,
    ):
        self.question = question
        self.knowledge_root = knowledge_root
        self.max_rounds = max_rounds
        self.vendor = vendor
        self.model = model
        self.clean_answer = clean_answer
        self.summary_batch_size = summary_batch_size
        self.retrieval_mode = retrieval_mode
        self.check_pitfalls = check_pitfalls
        self.chunk_size = chunk_size
        self.enable_skills = enable_skills
        self.summary_clean_answer = summary_clean_answer
        # think_mode：作用于【所有最终节点】的 *_AND_CLEAN_PROMPT，统一切换为
        # *_AND_CLEAN_THINK_PROMPT，要求模型按 <think>/<answer> 双标签输出。
        # 覆盖范围（不受分批/召回/chunk 影响）：
        #   - 非分批：SUMMARY_AND_CLEAN / RETRIEVAL_SUMMARY_AND_CLEAN
        #   - 分批  ：BATCH_MERGE_AND_CLEAN / RETRIEVAL_BATCH_MERGE_AND_CLEAN
        #             （chunk 模式最终也落到 batch_final_merge）
        # 中间提炼 prompt（BATCH_SUMMARY / RETRIEVAL_BATCH_SUMMARY / CHUNK_REASONING_*）
        # 始终保持原样，不受 think_mode 影响。
        self.think_mode = think_mode
        if self.think_mode and not self.summary_clean_answer:
            logger.warning(
                "[ThinkMode] think_mode=True 但 summary_clean_answer=False，"
                "本次将不会生效（think 版 prompt 仅作用于 *_AND_CLEAN 体系的最终总结）"
            )
        # last_think：在【全流程最后一步总结】阶段把 chat(enable_thinking=True) 打开，
        # 让底层模型返回思考轨迹（qwen3.5/3.6 会把 <think>…</think> 写进 content；
        # deepseek-reasoner / deepseek-v3.2 等会放到 message.reasoning_content，
        # llm/client.py 已统一回注到 content 前缀）。
        # 作用范围（仅这些最终节点，中间 batch/chunk/探索阶段都不开）：
        #   - _all_in_answer / _final_summary / _batch_final_merge
        #   - _retrieval_final_summary / _retrieval_batch_final_merge
        #   - _clean_answer（独立 clean 阶段）
        # 与 think_mode 是正交开关：think_mode 只改 prompt 模板（要求 JSON 输出），
        # last_think 只改 chat_template_kwargs.enable_thinking（开启推理轨迹）。
        self.last_think = bool(last_think)
        # 最终作答阶段的 system prompt：调用方（CLI/HTTP）可自定义；
        # 仅作用于【最终总结/作答】节点（_all_in_answer / _final_summary /
        # _batch_final_merge / _retrieval_final_summary / _retrieval_batch_final_merge）；
        # 中间提炼层使用 SUMMARY_EXTRACT_SYSTEM_PROMPT（chunk 级单块判定/摘录）和
        # BATCH_REDUCE_SYSTEM_PROMPT（多份前序结果的中间提炼/逻辑归并），不受此参数影响。
        #
        # 拼接策略（避免调用方覆盖掉内置的财税推理范式）：
        #   - 未传 / 传空：直接使用默认 SUMMARY_ANSWER_SYSTEM_PROMPT。
        #   - 传入非空：在 SUMMARY_ANSWER_SYSTEM_PROMPT 之前以「## 最高行为准则」标题挂上，
        #     并把默认部分放在「## 默认作答规范」标题下，二者用大标题显式区分；
        #     语义上自定义部分优先级最高（与默认冲突时以自定义为准）。
        custom = (answer_system_prompt or "").strip() if answer_system_prompt is not None else ""
        if custom:
            self.answer_system_prompt_var_names = [
                "answerSystemPrompt",
                "SUMMARY_ANSWER_SYSTEM_PROMPT",
            ]
            self.answer_system_prompt = (
                "## 最高行为准则\n"
                "（以下规则由调用方在本次请求中传入，优先级高于下方默认规范；"
                "若与默认规范冲突，以本节为准）\n\n"
                f"{custom}\n\n"
                "## 默认作答规范\n"
                f"{SUMMARY_ANSWER_SYSTEM_PROMPT}"
            )
            logger.info(
                f"[AnswerSystemPrompt] 检测到调用方自定义版本（{len(custom)} 字符），"
                f"已与默认 SUMMARY_ANSWER_SYSTEM_PROMPT 按【最高行为准则 + 默认作答规范】结构拼接，"
                f"最终长度: {len(self.answer_system_prompt)} 字符"
            )
        else:
            self.answer_system_prompt_var_names = ["SUMMARY_ANSWER_SYSTEM_PROMPT"]
            self.answer_system_prompt = SUMMARY_ANSWER_SYSTEM_PROMPT
            logger.info(
                f"[AnswerSystemPrompt] 使用默认 SUMMARY_ANSWER_SYSTEM_PROMPT，"
                f"长度: {len(self.answer_system_prompt)} 字符"
            )
        self.registry = ExploredRegistry()
        self.pitfalls_registry = PitfallsRegistry()
        self.retrieval_registry = RetrievalKnowledgeRegistry() if retrieval_mode else None
        self.skill_registry: SkillResultRegistry | None = (
            SkillResultRegistry() if enable_skills else None
        )
        self.skill_runner: SkillRunner | None = SkillRunner() if enable_skills else None
        self.all_results: list[AgentResult] = []
        self._chunk_directories: list[str] = []
        self._chunk_relevant_headings: list[str] = []
        self._chunk_reasoning_results: list[dict] = []
        self._chunk_slots: list[_OrderedSlot] = []  # 流式调度产物，trace 阶段引用
        self._reduce_pipeline: ReducePipeline | None = None  # reduce_queue 模式下保留以便 trace 阶段拿轨迹

        # ---------- 外部大模型原生回答（pure_model_result 开关） ----------
        # 开启时：推理流程启动（run() 入口）就向 deepseek-v4-pro 并行发起一次裸模型作答，
        # 在 batch summary / final summary 阶段阻塞等待该结果（最多 PURE_MODEL_WAIT_TIMEOUT
        # 秒），命中则以「## 参考回答」小节形式注入到 user prompt 紧随问题之后，并给对应
        # system prompt 追加 PURE_MODEL_REFERENCE_*_INSTRUCTIONS 以约束参考策略。
        # 关闭时（默认）：任何注入点均返回空串，完整保持原有行为。
        self.pure_model_result = bool(pure_model_result)
        self.pure_model_vendor = "deepseek-v4-pro"
        self.pure_model_wait_timeout_seconds = 60.0
        self._pure_model_executor: ThreadPoolExecutor | None = None
        self._pure_model_future = None
        self._pure_model_reference: str | None = None
        self._pure_model_lock = threading.Lock()

        # ---------- answer-refine（流水线最末一步独立后置精简） ----------
        # 开启时（默认推荐开启）：在 standard / retrieval / chunk 三种模式产出最终
        # answer 后、返回 result 字典之前，追加一次「结论先行 + 核心证据/因果逻辑/
        # 注意事项」结构化精简。与 cleanAnswer / summaryCleanAnswer / thinkMode /
        # lastThink 完全正交。
        # think_mode=True 时：原 answer 内容会迁移到响应 think 字段，refine 结果
        # 写入 answer 字段；think_mode=False 时直接覆盖最终 answer。
        # 关闭时：流水线行为保持原样，无任何额外 LLM 调用。
        self.answer_refine = bool(answer_refine)

        # ---------- 关联展开（可选） ----------
        self.enable_relations = enable_relations
        self.relation_max_depth = max(1, int(relation_max_depth))
        self.relation_max_nodes = max(1, int(relation_max_nodes))
        self.relation_workers = max(1, int(relation_workers))
        self.relation_remote_timeout = float(relation_remote_timeout)
        # relations 展开模式：
        #   "all"   - 默认；跳过 LLM 二次判定，定位成功即命中并按深度全展开
        #             （省 N 次 LLM RT，触发率 100%，token 略增）
        #   "smart" - 旧行为；每个候选条款用 LLM 判 is_relevant，准确率高但触发率
        #             受 chunk LLM 与 relation LLM 双重采样波动影响
        mode = (relations_expansion_mode or "all").strip().lower()
        if mode not in ("all", "smart"):
            logger.warning(
                f"[Relations] 未知 relations_expansion_mode='{relations_expansion_mode}'，回退到 'all'"
            )
            mode = "all"
        self.relations_expansion_mode = mode

        # ---------- batch summary 流水线模式 ----------
        # "layered"      - 默认；保留原行为：chunk+relations 走 _chunk_streaming_pipeline
        #                  的"按 slot 顺序流式 batch + 后续递归压缩按层同步"；其他入口走
        #                  _recursive_batch_reduce 同步分层。
        # "reduce_queue" - 所有压缩任务统一进 ReducePipeline，凑批+回灌，无层间同步点。
        #                  适合 chunk 数大、batch 长尾差异显著的场景。代价是早 flush 的
        #                  part 经过更多次中间压缩；用 reduce_max_part_depth 兜底护栏。
        sp_mode = (summary_pipeline_mode or "layered").strip().lower()
        if sp_mode not in ("layered", "reduce_queue"):
            logger.warning(
                f"[ReducePipeline] 未知 summary_pipeline_mode='{summary_pipeline_mode}'，"
                f"回退到 'layered'"
            )
            sp_mode = "layered"
        self.summary_pipeline_mode = sp_mode
        self.reduce_max_part_depth = max(1, int(reduce_max_part_depth))
        if self.summary_pipeline_mode == "reduce_queue":
            logger.info(
                f"[ReducePipeline] 已启用 reduce_queue 模式："
                f"max_part_depth={self.reduce_max_part_depth}, "
                f"batch_size={self.summary_batch_size}"
            )

        # 推导 page_knowledge 目录与索引路径（默认从 knowledge_root 的 parent 推断）。
        # 调用方（如 stress test）可显式传入 override。
        self.page_knowledge_dir = (
            os.path.abspath(page_knowledge_dir)
            if page_knowledge_dir
            else os.path.abspath(os.path.dirname(self.knowledge_root))
        )
        self.policy_index_path = (
            policy_index_path
            if policy_index_path
            else os.path.join(self.page_knowledge_dir, "_policy_index.json")
        )

        self.relation_registry: RelationRegistry | None = None
        self.clause_locator: ClauseLocator | None = None
        self.relation_crawler: RelationCrawler | None = None
        self.highlight_precheck: HighlightPrecheck | None = None
        self._highlight_precheck_stats: dict = {}
        self._relation_eval_executor: ThreadPoolExecutor | None = None
        self._relation_dispatch_executor: ThreadPoolExecutor | None = None
        # 副路径（standard / retrieval）on_hit 触发的 crawl future 收集器
        self._react_hit_dirs: set[str] = set()
        self._react_relation_futures: list = []
        self._react_hit_lock = threading.Lock()

        if self.enable_relations:
            self.relation_registry = RelationRegistry()
            self.clause_locator = ClauseLocator(
                page_knowledge_dir=self.page_knowledge_dir,
                policy_index_path=self.policy_index_path,
                remote_timeout=self.relation_remote_timeout,
            )
            # 两个独立池，避免 BFS 内嵌套同池等待导致死锁：
            #   - eval 池：单候选 (_evaluate_single) 的 LLM 评估
            #   - dispatch 池：每个 source_dir 的 crawl() 整体调度
            self._relation_eval_executor = ThreadPoolExecutor(
                max_workers=self.relation_workers * 2,
                thread_name_prefix="rel-eval",
            )
            self._relation_dispatch_executor = ThreadPoolExecutor(
                max_workers=self.relation_workers,
                thread_name_prefix="rel-dispatch",
            )
            self.relation_crawler = RelationCrawler(
                question=self.question,
                registry=self.relation_registry,
                locator=self.clause_locator,
                executor=self._relation_eval_executor,
                vendor=self.vendor,
                model=self.model,
                max_depth=self.relation_max_depth,
                max_nodes=self.relation_max_nodes,
                expand_all=(self.relations_expansion_mode == "all"),
            )
            self.highlight_precheck = HighlightPrecheck(
                question=self.question,
                knowledge_root=self.knowledge_root,
                crawler=self.relation_crawler,
                vendor=self.vendor,
                model=self.model,
            )
            logger.info(
                f"[Relations] 已启用关联展开：mode={self.relations_expansion_mode}, "
                f"max_depth={self.relation_max_depth}, max_nodes={self.relation_max_nodes}, "
                f"workers={self.relation_workers}, page_knowledge_dir={self.page_knowledge_dir}"
            )
            logger.info(
                "[Relations] HighlightPrecheck 已注册：reasoning 启动时将与 "
                "skill 判定、chunk/agent 相关性判定并行跑一次"
            )

    def _shutdown_relation_executors(self) -> None:
        """关联展开相关线程池在推理结束时统一释放。重复调用幂等。

        pure_model 线程池在本函数内一并释放（在此收口而不是新开函数，让外部的
        推理收尾路径只用记住一个 shutdown 入口）。
        """
        for attr in ("_relation_eval_executor", "_relation_dispatch_executor", "_pure_model_executor"):
            ex = getattr(self, attr, None)
            if ex is not None:
                try:
                    ex.shutdown(wait=True)
                except Exception:
                    pass
                setattr(self, attr, None)

    # ============================= 外部大模型原生回答 =============================

    def _start_pure_model_request_async(self) -> None:
        """推理流程启动时向 deepseek-v4-pro 并行发起一次纯模型作答。

        底层 chat 开启 enable_thinking=True（利于模型推理质量）；思考轨迹不入「参考回答」——
        `_get_pure_model_reference` 内对返回值做 split_think_block，仅缓存正文。

        开关关闭 / 已经启动过 → 直接返回（幂等）。异常只打日志不抛出——下游取值时
        会走超时/失败降级路径，保证主推理链路不会因外部模型故障而失败。
        """
        if not self.pure_model_result:
            return
        if self._pure_model_future is not None:
            return
        prompt = PURE_MODEL_REQUEST_PROMPT.format(question=self.question)
        logger.info(
            f"[PureModel] 启动 {self.pure_model_vendor} 并行作答，"
            f"prompt 长度: {len(prompt)} 字符"
        )
        self._pure_model_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="pure-model",
        )

        def _run() -> str:
            try:
                with step_scope(
                    "pure_model_reasoning",
                    prompt_vars={
                        "user": "PURE_MODEL_REQUEST_PROMPT",
                        "system": "SUMMARY_SYSTEM_PROMPT",
                    },
                ):
                    return chat(
                        prompt,
                        vendor=self.pure_model_vendor,
                        model=self.pure_model_vendor,
                        system=SUMMARY_SYSTEM_PROMPT,
                        enable_thinking=True,
                    )
            except Exception as e:
                logger.error(f"[PureModel] 外部模型调用失败: {e}")
                return ""

        self._pure_model_future = self._pure_model_executor.submit(_run)

    def _get_pure_model_reference(self) -> str:
        """batch summary / final summary 阶段阻塞取值。

        - 开关关闭：直接返回空串
        - 首次调用：最多等 pure_model_wait_timeout_seconds 秒；超时 / 失败 / 空回复
          视为"无外部参考"，缓存空串防止后续节点重复阻塞
        - 再次调用：直接返回缓存值（pure-model 回答只对同一请求生效一次）
        """
        if not self.pure_model_result:
            return ""
        with self._pure_model_lock:
            if self._pure_model_reference is not None:
                return self._pure_model_reference
            fut = self._pure_model_future
            if fut is None:
                self._pure_model_reference = ""
                return ""
            try:
                text = fut.result(timeout=self.pure_model_wait_timeout_seconds)
            except Exception as e:
                logger.warning(
                    f"[PureModel] 等待外部模型回答失败/超时 "
                    f"({self.pure_model_wait_timeout_seconds}s 上限): {e}，"
                    f"本次退化为'无外部参考'"
                )
                text = ""
            # 底层 chat() 若模型走 thinking 模式可能前缀带 <think>…</think>，
            # 参考块要展示给后续 LLM，统一剥掉思考块只留正式作答正文。
            _, body = split_think_block(text or "")
            cleaned = (body or "").strip()
            if cleaned:
                preview = cleaned[:200].replace("\n", " ")
                logger.info(
                    f"[PureModel] 外部模型回答就绪（{len(cleaned)} 字符）: "
                    f"{preview}{'…' if len(cleaned) > 200 else ''}"
                )
            else:
                logger.warning("[PureModel] 外部模型回答为空，本次退化为'无外部参考'")
            self._pure_model_reference = cleaned
            return cleaned

    @staticmethod
    def _append_pure_model_reference_to_prompt(prompt: str, reference: str) -> str:
        """把外部模型回答以「## 参考回答」小节形式插入到 `## 用户问题\\n{question}`
        紧随其后（下一个 `#`/`##` 小节之前）。

        锚点策略与 _append_skill_context_to_prompt 完全同构（皆以"用户问题段尾部、
        下个 heading 之前"为插入位置），两者叠加时 skill_context 先插入 → reference
        紧随其后，最终 prompt 呈现顺序为：用户问题 → Skill 参考事实 → 外部参考回答 →
        后续知识 / 输出要求段。
        """
        if not reference:
            return prompt
        block = "## 参考回答\n" + reference
        match = re.search(r"## 用户问题\n.+?\n\n(?=#)", prompt, flags=re.DOTALL)
        if match:
            insert_pos = match.end()
            return prompt[:insert_pos] + block + "\n\n" + prompt[insert_pos:]
        marker = "\n---\n"
        idx = prompt.rfind(marker)
        if idx < 0:
            return prompt + "\n\n" + block
        return prompt[:idx] + "\n\n" + block + prompt[idx:]

    def _inject_pure_model_reference(self, prompt: str) -> str:
        """封装"取外部回答 + 按需注入 user prompt"的一站式入口。关闭/无回答时原样返回。"""
        ref = self._get_pure_model_reference()
        if not ref:
            return prompt
        return self._append_pure_model_reference_to_prompt(prompt, ref)

    def _bake_pure_model_reference_into_template(self, template: str) -> str:
        """把「## 参考回答」块预先烘焙进 prompt 模板，再交给后续 `.format(...)` 渲染。

        专供 ReducePipeline 使用：pipeline 在内部按 `intermediate_prompt.format(...)`
        统一渲染，不方便把"渲染后再注入"的 hook 接进去；因此这里在模板层就把参考
        块插到 `## 用户问题\\n{question}` 后、下一个 `#` 小节前。为了让 `.format()`
        能完整保留参考块原文，参考文本中的花括号会被转义（`{` → `{{`，`}` → `}}`）。

        pure_model 关闭 / 无回答：原样返回模板，不做任何修改。
        """
        ref = self._get_pure_model_reference()
        if not ref:
            return template
        ref_escaped = ref.replace("{", "{{").replace("}", "}}")
        block = "## 参考回答\n" + ref_escaped + "\n\n"
        anchor = re.compile(r"(?<=## 用户问题\n\{question\}\n\n)(?=#)")
        m = anchor.search(template)
        if m:
            return template[: m.start()] + block + template[m.start():]
        marker = "\n---\n"
        idx = template.rfind(marker)
        if idx < 0:
            return template + "\n\n" + block
        return template[:idx] + "\n\n" + block + template[idx:]

    def _augment_system_for_extract(
        self, role_header: str, format_body: str = ""
    ) -> str:
        """batch summary 中间阶段的 system prompt 组装。

        组装顺序固定为：
            role_header
            └─（若 pure_model_result=True 且参考回答非空）PURE_MODEL_REFERENCE_EXTRACT_INSTRUCTIONS
                └─ format_body（含「输出格式 / 动作规范」等内容；可为空）

        即：始终把【外部参考回答处理策略】放在【格式要求 / 动作规范】**上方**，
        避免末尾追加把它压到了输出 schema 之后导致优先级被弱化。
        """
        parts: list[str] = [role_header]
        if self.pure_model_result and self._get_pure_model_reference():
            parts.append(PURE_MODEL_REFERENCE_EXTRACT_INSTRUCTIONS)
        if format_body:
            parts.append(format_body)
        return "\n\n".join(parts)

    def _augment_system_for_answer(
        self, role_header: str, format_body: str = ""
    ) -> str:
        """final summary 阶段的 system prompt 组装。

        组装顺序与 `_augment_system_for_extract` 完全一致，只是追加的指令换成
        PURE_MODEL_REFERENCE_ANSWER_INSTRUCTIONS：
            role_header
            └─（若 pure_model_result=True 且参考回答非空）PURE_MODEL_REFERENCE_ANSWER_INSTRUCTIONS
                └─ format_body（含「输出格式（必须严格遵守）」JSON schema 等；可为空）
        """
        parts: list[str] = [role_header]
        if self.pure_model_result and self._get_pure_model_reference():
            parts.append(PURE_MODEL_REFERENCE_ANSWER_INSTRUCTIONS)
        if format_body:
            parts.append(format_body)
        return "\n\n".join(parts)

    def _extract_system_prompt_vars(self, *format_prompt_vars: str) -> list[str]:
        names = ["BATCH_REDUCE_SYSTEM_PROMPT"]
        if self.pure_model_result and self._pure_model_reference:
            names.append("PURE_MODEL_REFERENCE_EXTRACT_INSTRUCTIONS")
        names.extend(name for name in format_prompt_vars if name)
        return names

    def _answer_system_prompt_vars(self, *format_prompt_vars: str) -> list[str]:
        names = list(self.answer_system_prompt_var_names)
        if self.pure_model_result and self._pure_model_reference:
            names.append("PURE_MODEL_REFERENCE_ANSWER_INSTRUCTIONS")
        names.extend(name for name in format_prompt_vars if name)
        return names

    def _run_skill_evaluation(self) -> None:
        """在独立线程中跑 evaluator 的 asyncio 事件循环"""
        try:
            asyncio.run(evaluate_and_run(
                question=self.question,
                registry=self.skill_registry,
                runner=self.skill_runner,
                vendor=self.vendor,
                model=self.model,
            ))
        except Exception as e:
            logger.exception(f"[Skill] 预评估阶段异常: {e}")

    def _run_highlight_precheck(self) -> None:
        """在独立线程中跑 HighlightPrecheck。

        把当前知识包所有外链 highlightedContent 一次性喂给 LLM 判定是否需要展开，
        命中结果通过 RelationCrawler 直接写入 RelationRegistry。统计信息存到
        self._highlight_precheck_stats 供 trace 使用。

        与 skill 判定 / chunk & agent 相关性判定并行执行；失败不影响主推理链路。
        """
        if self.highlight_precheck is None:
            return
        try:
            self._highlight_precheck_stats = self.highlight_precheck.run()
        except Exception as e:
            logger.exception(f"[HighlightPrecheck] 异常: {e}")
            self._highlight_precheck_stats = {
                "total_candidates": 0,
                "selected": 0,
                "fragments": 0,
                "reason": f"异常: {e}",
            }

    def _run_root_agent_and_flatten(self) -> None:
        """主推理：根智能体 -> 子智能体并行 -> 展平结果。

        当 enable_relations=True 时，把 self._react_on_hit 作为 on_hit 回调注入到
        ReactAgent；任何 _assess_* 命中都会异步触发 RelationCrawler.crawl，
        crawl 结果在后续 _organize_fragments / _build_evidence_parts 阶段被 inline 渲染。
        """
        on_hit_cb = self._react_on_hit if self.enable_relations else None
        root_agent = ReactAgent(
            question=self.question,
            knowledge_root=self.knowledge_root,
            current_dir=self.knowledge_root,
            upstream_path=[self.knowledge_root],
            parent_summary="",
            registry=self.registry,
            pitfalls_registry=self.pitfalls_registry,
            max_rounds=self.max_rounds,
            vendor=self.vendor,
            model=self.model,
            retrieval_mode=self.retrieval_mode,
            retrieval_registry=self.retrieval_registry,
            subtree_root=self.knowledge_root,
            on_hit=on_hit_cb,
        )
        root_result = root_agent.run()
        self.all_results = self._flatten_results(root_result)
        logger.info(f"第一轮汇聚完成，共 {len(self.all_results)} 个智能体结果")

    # ===== 副路径（standard / retrieval）关联展开钩子 =====

    def _react_on_hit(self, directory: str, assessment: dict) -> None:
        """ReactAgent 命中相关知识时调用：异步触发 RelationCrawler.crawl。

        - 不阻塞调用方（提交到 _relation_dispatch_executor 后立即返回）
        - 同一 directory 多次命中只 schedule 一次（用 _react_hit_dirs 去重）
        - parent_assessment 取 assessment 中的 reason / conclusion / summary
        """
        if not self.enable_relations or self.relation_crawler is None:
            return
        if self._relation_dispatch_executor is None:
            return
        if not os.path.isfile(os.path.join(directory, "clause.json")):
            return  # 无 clause.json 即无 references，不浪费 crawl

        with self._react_hit_lock:
            if directory in self._react_hit_dirs:
                return
            self._react_hit_dirs.add(directory)

        parent_assessment = (
            assessment.get("reason")
            or assessment.get("conclusion")
            or assessment.get("summary")
            or ""
        )
        if isinstance(parent_assessment, list):
            parent_assessment = " | ".join(str(x) for x in parent_assessment)
        parent_assessment = (parent_assessment or "")[:400]

        fut = self._relation_dispatch_executor.submit(
            self.relation_crawler.crawl,
            -1, directory, parent_assessment,
        )
        with self._react_hit_lock:
            self._react_relation_futures.append(fut)

    def _wait_react_relations(self) -> None:
        """阻塞等待 ReactAgent 触发的所有关联 crawl 完成；pipeline 进入前调用。"""
        if not self._react_relation_futures:
            return
        logger.info(
            f"[Relations] 等待副路径关联展开 {len(self._react_relation_futures)} 个 crawl 完成..."
        )
        for fut in self._react_relation_futures:
            try:
                fut.result()
            except Exception as e:
                logger.warning(f"[Relations] 副路径 crawl 异常已忽略: {e}")
        if self.relation_registry:
            logger.info(
                f"[Relations] 副路径 crawl 全部完成，共命中 "
                f"{len(self.relation_registry.get_all())} 个关联条款"
            )

    # ========================= Skill 上下文 & double-check 并行编排 =========================

    @staticmethod
    def _render_skill_records(records: list[SkillRecord]) -> str:
        """将 SkillRecord 列表渲染为可读文本（与 SkillResultRegistry.format_context 对齐，但只取指定 records）。

        命令行不展示，避免污染面向 LLM 的事实段；只保留 skill 名与结果/失败信息。
        """
        if not records:
            return ""
        lines: list[str] = []
        for i, rec in enumerate(records, 1):
            lines.append(f"【Skill 调用 {i}】{rec.skill_name}")
            if rec.result.success:
                lines.append("结果:")
                lines.append(rec.result.stdout or "（无输出）")
            else:
                lines.append(f"调用失败 (exit={rec.result.exit_code}):")
                if rec.result.stderr:
                    lines.append(rec.result.stderr)
            lines.append("")
        return "\n".join(lines).rstrip()

    def _build_skill_context_for_summary(self, records: list[SkillRecord]) -> str:
        """把 skill records 包装成可插入到 summary prompt 头部的参考事实段。空 records → 空串。

        只给一个节标题；每条 skill 结果的适用范围/免责声明由各 skill 的 format_result
        内部自行附带（见 e.g. standard_product_name_verification/service.py 中的
        _RESULT_DISCLAIMER），避免在不同层级上重复同一句话。
        """
        body = self._render_skill_records(records)
        if not body:
            return ""
        return "## 参考事实（外部 Skill 结果）\n" + body

    @staticmethod
    def _append_skill_context_to_prompt(prompt: str, skill_context: str) -> str:
        """把 skill_context 插入到 prompt 的「用户问题」段之后、下一个小节标题之前，
        让 Skill 结果作为"参考事实"紧跟在用户问题下方出现。

        所有最终 summary/merge 模板都遵循
            "## 用户问题\\n{question}\\n\\n{下一个 # 或 ## 小节}\\n...\\n---\\n{输出要求}"
        结构，以"紧跟在 `## 用户问题` 段之后的第一个 heading"作为插入锚点；
        若意外缺失锚点，退化为插入到最后一个 "---" 之前；仍失败则追加到末尾。
        """
        if not skill_context:
            return prompt
        match = re.search(r"## 用户问题\n.+?\n\n(?=#)", prompt, flags=re.DOTALL)
        if match:
            insert_pos = match.end()
            return prompt[:insert_pos] + skill_context + "\n\n" + prompt[insert_pos:]
        marker = "\n---\n"
        idx = prompt.rfind(marker)
        if idx < 0:
            return prompt + "\n\n" + skill_context
        return prompt[:idx] + "\n\n" + skill_context + prompt[idx:]

    @staticmethod
    def _format_batch_summaries_for_merge(summaries: list[str]) -> str:
        """把多份中间摘要拼接成最终合并 prompt 中的 batch_summaries 段。

        使用强分隔哨兵（BEGIN_BATCH_SUMMARY n / END_BATCH_SUMMARY n）包围每条
        摘要，避免摘要正文里的 Markdown 标题（##/###）与外层 prompt 的 ##/###
        小节标题竞争层级——后者负责"用户问题/各批次摘要/输出格式约束"等外层
        结构，前者只能出现在哨兵块内部，模型可以一眼区分边界归属。

        每个摘要块结构：
            ===== BEGIN_BATCH_SUMMARY {i} =====
            [摘要序号] {i}
            [摘要内容开始]
            {正文原样保留，rstrip 去尾部空白}
            [摘要内容结束]
            ===== END_BATCH_SUMMARY {i} =====
        块之间用空行分隔。
        """
        blocks: list[str] = []
        for i, s in enumerate(summaries, start=1):
            body = (s or "").rstrip()
            blocks.append(
                f"===== BEGIN_BATCH_SUMMARY {i} =====\n"
                f"[摘要序号] {i}\n"
                f"[摘要内容开始]\n"
                f"{body}\n"
                f"[摘要内容结束]\n"
                f"===== END_BATCH_SUMMARY {i} ====="
            )
        return "\n\n".join(blocks)

    def _judge_extra_skills(
        self, exclude: set[str], evidence: str | None = None
    ) -> list[str]:
        """让 LLM 基于 question + 已完成 skill + （可选）浓缩证据二次判定还需哪些 skill。

        与旧版的差异：evidence 不再为空——final summary 即将吃下去的 numbered summaries
        / agent_parts / organized_parts 会作为浓缩中间事实喂给 LLM，让它能基于"已经被
        提炼出来的关键事实"判定是否仍存在必须靠 skill 才能闭合的关键缺口。
        """
        try:
            return select_extra_skills(
                question=self.question,
                exclude=exclude,
                vendor=self.vendor,
                model=self.model,
                evidence=evidence,
            )
        except Exception as e:
            logger.exception(f"[DoubleCheck] 二次判定 extra skill 异常，视为无新需求: {e}")
            return []

    def _run_extra_skills(self, skill_names: list[str]) -> list[SkillRecord]:
        """跑指定的 skill，返回本次新增的 records（不含调用前已存在的）"""
        if not skill_names or not self.skill_registry or not self.skill_runner:
            return []
        before = len(self.skill_registry.get_all())
        try:
            asyncio.run(evaluate_and_run(
                question=self.question,
                registry=self.skill_registry,
                runner=self.skill_runner,
                vendor=self.vendor,
                model=self.model,
                skill_names=skill_names,
            ))
        except Exception as e:
            logger.exception(f"[DoubleCheck] 补充执行 skill 异常: {e}")
        after_records = self.skill_registry.get_all()
        return after_records[before:]

    def _all_in_answer(
        self, final_summary: str, extra_records: list[SkillRecord]
    ) -> str:
        """慢速路径专用：在 final_summary（已带 phase-0 注入 + 清洗）之后，
        把"本轮 judge 才补跑出来的 extra skill records"作为权威事实，最小修订草稿。

        不再覆盖 phase-0 records（它们已经在 final_summary 里被吸收了），避免重复 load。
        """
        skill_context = self._render_skill_records(extra_records)
        if not skill_context:
            return final_summary
        prompt_kwargs = dict(
            question=self.question,
            final_summary=final_summary,
            skill_context=skill_context,
        )
        prompt = ALL_IN_ANSWER_PROMPT.format(**prompt_kwargs)
        prompt_html = ALL_IN_ANSWER_HTML_PROMPT.format(**prompt_kwargs)
        system_prompt = (
            self.answer_system_prompt
            + "\n\n## 输出格式约束\n"
            + ALL_IN_ANSWER_SYSTEM_PROMPT
        )
        system_prompt_html = (
            self.answer_system_prompt
            + "\n\n## 输出格式约束\n"
            + ALL_IN_ANSWER_HTML_SYSTEM_PROMPT
        )
        logger.info(f"[AllInAnswer] prompt 长度: {len(prompt)} 字符")
        logger.info(f"[AllInAnswer] prompt 内容:\n{prompt}")
        try:
            answer_json = self._chat_final_with_format_retry(
                prompt=prompt,
                system_prompt=system_prompt,
                step_label="all_in_answer",
                expects_think_answer_json=True,
                html_retry_prompt=prompt_html,
                html_retry_system=system_prompt_html,
                prompt_vars={
                    "user": "ALL_IN_ANSWER_PROMPT",
                    "system": self._answer_system_prompt_vars("ALL_IN_ANSWER_SYSTEM_PROMPT"),
                },
                html_retry_prompt_vars={
                    "user": "ALL_IN_ANSWER_HTML_PROMPT",
                    "system": self._answer_system_prompt_vars("ALL_IN_ANSWER_HTML_SYSTEM_PROMPT"),
                },
            )
            if self.think_mode and answer_json:
                # 保留完整结构交给 app.py 映射到 ReasonData.think / answer。
                return answer_json
            extracted = _extract_answer_from_think_answer_json(answer_json)
            if extracted:
                return extracted
            if not answer_json:
                logger.warning("[AllInAnswer] LLM 返回空，沿用 final summary 原文")
                return final_summary
            # coercion 兜底也至少是 {"think": ..., "answer": ...} JSON；走到这里属于极端情形，
            # 直接把 JSON 串原样返回也不至于让 final summary 丢失（下游 _split_analysis_concise_answer
            # 会再做一次解析）。
            return answer_json
        except Exception as e:
            logger.error(f"[AllInAnswer] LLM 调用失败，沿用 final summary 原文: {e}")
            return final_summary

    def _finalize_with_double_check(
        self,
        final_summary_callable,
        judge_evidence: str,
        stage_label: str = "FinalSummary",
    ) -> str:
        """新版 double-check 编排：在每个最终 summary / merge 步骤内部使用。

        - final_summary_callable: 已经把 phase-0 skill_context 拼好的 final summary 调用
          （带清洗或不带清洗，由调用方决定）
        - judge_evidence: 喂给 judge 的浓缩证据（即将进入 final summary 的 numbered
          summaries / agent_parts / organized_parts）
        - 编排：summary || judge 并行；judge 无需求 → 直接返回 summary；judge 有需求 →
          起 extra-skill 并行跑，等 summary 与 extra-skill 都完成后走 _all_in_answer 拼接
        """
        if not self.enable_skills or not self.skill_registry:
            return final_summary_callable()

        done_records_snapshot = self.skill_registry.get_all()
        done_skill_names = {r.skill_name for r in done_records_snapshot}

        with ThreadPoolExecutor(max_workers=3) as executor:
            summary_future = executor.submit(final_summary_callable)
            judge_future = executor.submit(
                self._judge_extra_skills, done_skill_names, judge_evidence,
            )

            try:
                extra_needed = judge_future.result()
            except Exception as e:
                logger.error(f"[{stage_label}] judge 异常，按无新需求处理: {e}")
                extra_needed = []

            extra_future = (
                executor.submit(self._run_extra_skills, extra_needed)
                if extra_needed else None
            )

            final_summary = summary_future.result()

            if extra_future is None:
                logger.info(
                    f"[{stage_label}] judge 无新 skill 需求"
                    f"（phase-0 已完成 {len(done_records_snapshot)} 个 skill 并已注入），"
                    f"直接采用 final summary 输出"
                )
                return final_summary

            new_records = extra_future.result()

        if not new_records:
            logger.warning(
                f"[{stage_label}] 补跑 skill 后未产出有效 records，沿用 final summary 原文"
            )
            return final_summary

        logger.info(
            f"[{stage_label}] 用 {len(new_records)} 条新增 skill records 走 all-in-answer 拼接"
            f"（老的 {len(done_records_snapshot)} 条已在 final summary 里被吸收）"
        )
        return self._all_in_answer(final_summary, new_records)

    def _postprocess_final_chat(self, raw: str, step_label: str) -> str:
        """剥离最终节点 chat() 返回里可能存在的 <think>…</think> 前缀。

        - 当 last_think=True（或某些 qwen 模型强行吐 think）时，llm/client.py 会把
          reasoning_content 以 <think>…</think> 形式前缀回注到 content。
        - 业务层要求 answer 保持"纯业务格式"：think_mode=True 时应为合法 JSON；
          think_mode=False 时应为自然语言客服回答。因此在每个最终节点统一剥掉。
        - 剥下来的 think 块打到 INFO 日志（截断预览），完整内容保留在 verbose_log
          的 LLM 响应记录里（llm/client.py 已写入），不会丢失。
        - 幂等：raw 里没有 <think> 前缀时原样返回。
        """
        if not raw:
            return raw or ""
        think, body = split_think_block(raw)
        if think:
            preview = think[:300].replace("\n", " ")
            logger.info(
                f"[FinalChat·{step_label}] 剥离 <think> 前缀 {len(think)} 字符："
                f"{preview}{'…' if len(think) > 300 else ''}"
            )
        return body

    def _is_html_first_for_final(self) -> bool:
        """判定最终节点（_all_in_answer / *_final_summary / *_batch_final_merge）
        在双格式兜底链路中是否优先用 HTML、再回落到 JSON。

        背景（参见 tests_logs/format_priority_compare_20260505_003502.log，N=10 实测）：
          - deepseek-v3.2：JSON 模式平均 91.8s（reasoning 平均 3,901 字符，模型对 JSON
            字符串转义的反复斟酌占用了大量推理 token）；HTML 模式平均 38.3s（reasoning
            1,255 字符），通过率 90%（含未闭合 <answer> 兜底后 100%）。HTML 首选可节省
            ~54s/次（约 -58% 时延）。
          - qwen3.6-35b-a3b：JSON 模式 100%/8.4s，HTML 模式只有 56% 通过率（剩余直接
            吐纯散文）。必须保持 JSON 首选 + HTML 兜底。

        判定逻辑：vendor 或 model 名包含 "deepseek" 即视作 deepseek 系列，无论走哪条
        通道（aliyun mudgate / servyou 内网 / mudgate v4-pro/flash），也兼容未来版本。
        """
        v = (self.vendor or "").lower()
        m = (self.model or "").lower()
        return "deepseek" in v or "deepseek" in m

    def _final_chat_and_parse_once(
        self,
        *,
        prompt: str,
        system_prompt: str | None,
        label: str,
        fmt: str,  # "json" / "html"
        enable_thinking: bool,
        prompt_vars: dict | list | str | None = None,
    ) -> tuple[dict | None, str, str]:
        """单次 chat + 按指定格式解析。返回 (parsed_or_none, body, think_prefix)。"""
        with step_scope(label, prompt_vars=prompt_vars):
            raw = chat(
                prompt, vendor=self.vendor, model=self.model,
                system=system_prompt,
                enable_thinking=enable_thinking,
            )
        think_prefix, body = split_think_block(raw or "")
        if think_prefix:
            preview = think_prefix[:300].replace("\n", " ")
            logger.info(
                f"[FinalChat·{label}] 剥离 <think> 前缀 {len(think_prefix)} 字符："
                f"{preview}{'…' if len(think_prefix) > 300 else ''}"
            )
        if fmt == "json":
            parsed = _validate_think_answer_json(body)
        else:
            parsed = _parse_think_answer_html(body, fallback_think=think_prefix)
        return parsed, body, think_prefix

    def _chat_final_with_format_retry(
        self,
        *,
        prompt: str,
        system_prompt: str | None,
        step_label: str,
        expects_think_answer_json: bool,
        enable_thinking: bool | None = None,
        html_retry_prompt: str | None = None,
        html_retry_system: str | None = None,
        prompt_vars: dict | list | str | None = None,
        html_retry_prompt_vars: dict | list | str | None = None,
    ) -> str:
        """统一最终节点 chat 调用入口；自带「首选格式 → 兜底格式 → coercion」三层格式兜底。

        参数：
          - prompt:                JSON 版完整 user prompt（已注入 skill_context、参考回答等）
          - system_prompt:         JSON 版 system prompt（已 augment 完毕）；可为 None
          - step_label:            verbose trace 与日志中显示的节点名
          - expects_think_answer_json:
              True  → 期望模型最终给出严格的 `{think, answer}` 结构（即 think_mode +
                       summary_clean_answer 同时开启的最终节点 / `_all_in_answer`）。按
                       `_is_html_first_for_final()` 决定首选格式：
                         * deepseek 系列 → HTML 首选 + JSON 兜底（实测 -58% 时延）
                         * 其他（qwen 等）→ JSON 首选 + HTML 兜底（保持原行为）
                       两次都不过则交叉兜底（首轮 body 是不是其实是另一格式）+ coercion。
                       返回值恒为合法的 `{"think": str, "answer": str}` JSON 字符串。
              False → 不做格式校验、不重试，原样返回 split_think_block 后的 body（自然语言）。
          - enable_thinking:       默认随 self.last_think；少数节点（如 clean_answer）可显式覆盖。
          - html_retry_prompt:     HTML 完整 user prompt（已 .format 完毕、已注入 skill_context /
                                    参考回答等），内容/呈现段与 prompt 100% 一致，仅"输出格式"段
                                    切换为 HTML 双标签。expects_think_answer_json=True 时强烈建议
                                    传入；缺省时退化为单格式（不进行兜底重试，直接 coercion）。
          - html_retry_system:     HTML 版 system prompt（已 augment 完毕），格式段为 HTML。
                                    缺省时复用 system_prompt（仅当 system 不携带格式段时安全）。

        异常时直接抛给上层（保持与原 chat 调用相同的失败语义），由各节点自身的 try/except 兜住。
        """
        if enable_thinking is None:
            enable_thinking = self.last_think

        if not expects_think_answer_json:
            with step_scope(step_label, prompt_vars=prompt_vars):
                raw = chat(
                    prompt, vendor=self.vendor, model=self.model,
                    system=system_prompt,
                    enable_thinking=enable_thinking,
                )
            think_prefix, body = split_think_block(raw or "")
            if think_prefix:
                preview = think_prefix[:300].replace("\n", " ")
                logger.info(
                    f"[FinalChat·{step_label}] 剥离 <think> 前缀 {len(think_prefix)} 字符："
                    f"{preview}{'…' if len(think_prefix) > 300 else ''}"
                )
            return body

        # 决定首选 / 兜底格式：deepseek 系列优先 HTML（reasoning_content 自然承载 think，
        # 跳过 JSON 转义反复斟酌，节省 ~58% 时延）；其他保持 JSON 首选 + HTML 兜底。
        html_first = self._is_html_first_for_final() and html_retry_prompt is not None

        if html_first:
            primary_fmt = "html"
            primary_prompt = html_retry_prompt
            primary_system = html_retry_system if html_retry_system is not None else system_prompt
            primary_prompt_vars = html_retry_prompt_vars or prompt_vars
            primary_label = step_label
            secondary_fmt = "json"
            secondary_prompt: str | None = prompt
            secondary_system: str | None = system_prompt
            secondary_prompt_vars = prompt_vars
            secondary_label = f"{step_label}·json_retry"
        else:
            primary_fmt = "json"
            primary_prompt = prompt
            primary_system = system_prompt
            primary_prompt_vars = prompt_vars
            primary_label = step_label
            if html_retry_prompt is not None:
                secondary_fmt = "html"
                secondary_prompt = html_retry_prompt
                secondary_system = html_retry_system if html_retry_system is not None else system_prompt
                secondary_prompt_vars = html_retry_prompt_vars or prompt_vars
                secondary_label = f"{step_label}·html_retry"
            else:
                secondary_fmt = ""
                secondary_prompt = None
                secondary_system = None
                secondary_prompt_vars = None
                secondary_label = ""

        # ===== 首轮：首选格式 =====
        parsed1, body1, think1 = self._final_chat_and_parse_once(
            prompt=primary_prompt,
            system_prompt=primary_system,
            label=primary_label,
            fmt=primary_fmt,
            enable_thinking=enable_thinking,
            prompt_vars=primary_prompt_vars,
        )
        if parsed1:
            return json.dumps(parsed1, ensure_ascii=False)

        body1_preview = (body1 or "").replace("\n", " ")[:200]

        if not secondary_prompt:
            logger.warning(
                f"[FinalChat·{primary_label}] {primary_fmt.upper()} 校验失败且未提供"
                f" {('JSON' if primary_fmt == 'html' else 'HTML')} 兜底 prompt，"
                f"直接走 coercion 兜底封装。 body 预览: {body1_preview}"
            )
            return _coerce_to_think_answer_json(body1, fallback_think=think1)

        logger.warning(
            f"[FinalChat·{primary_label}] {primary_fmt.upper()} 校验失败，发起 "
            f"{secondary_fmt.upper()} 兜底重试。 body 预览: {body1_preview}"
        )

        # ===== 兜底：另一种格式 =====
        try:
            parsed2, body2, think2 = self._final_chat_and_parse_once(
                prompt=secondary_prompt,
                system_prompt=secondary_system,
                label=secondary_label,
                fmt=secondary_fmt,
                enable_thinking=enable_thinking,
                prompt_vars=secondary_prompt_vars,
            )
        except Exception as e:
            logger.error(
                f"[FinalChat·{secondary_label}] {secondary_fmt.upper()} 兜底 chat 调用异常 "
                f"{type(e).__name__}: {e}，转入 coercion 兜底封装"
            )
            return _coerce_to_think_answer_json(body1, fallback_think=think1)

        if parsed2:
            logger.info(
                f"[FinalChat·{secondary_label}] {secondary_fmt.upper()} 解析成功"
                f"（answer 长度 {len(parsed2['answer'])}，think 长度 {len(parsed2['think'])}）"
            )
            return json.dumps(parsed2, ensure_ascii=False)

        # ===== 交叉兜底：兜底也失败时，再回头看看首轮 body 是不是其实是另一种格式 =====
        # （deepseek 偶尔会在 HTML schema 下吐 JSON，反之亦然；免得白白丢掉一次推理）
        if secondary_fmt == "html":
            parsed1_cross = _parse_think_answer_html(body1, fallback_think=think1)
        else:
            parsed1_cross = _validate_think_answer_json(body1)
        if parsed1_cross:
            logger.warning(
                f"[FinalChat·{primary_label}] {secondary_fmt.upper()} 兜底也失败，"
                f"但首轮 body 含可解析 {secondary_fmt.upper()}，使用首轮结果"
                f"（answer 长度 {len(parsed1_cross['answer'])}）"
            )
            return json.dumps(parsed1_cross, ensure_ascii=False)

        body2_preview = (body2 or "").replace("\n", " ")[:200]
        logger.warning(
            f"[FinalChat·{secondary_label}] {secondary_fmt.upper()} 兜底仍未通过校验，"
            f"触发 coercion 兜底封装。 body2 预览: {body2_preview}"
        )
        fallback_think = think2 or think1
        fallback_body = body2 if (body2 and body2.strip()) else body1
        return _coerce_to_think_answer_json(fallback_body, fallback_think=fallback_think)

    def run(self) -> dict:
        with agent_scope(agent_id="graph-root"):
            # 外部大模型并行作答尽可能早启动，与 skill 判定 / 主推理同步并发，
            # 以便在后续 batch summary / final summary 阶段直接就绪。
            self._start_pure_model_request_async()
            return self._run_impl()

    def _run_impl(self) -> dict:
        if self.chunk_size > 0:
            return self._run_chunk_mode()

        mode_label = "召回模式" if self.retrieval_mode else "标准模式"
        skill_label = "Skill开启" if self.enable_skills else "Skill关闭"
        precheck_label = (
            "HighlightPrecheck开启" if self.highlight_precheck is not None
            else "HighlightPrecheck关闭"
        )
        logger.info(
            f"AgentGraph 启动 [{mode_label} | {skill_label} | {precheck_label}]: "
            f"问题={self.question[:50]}..."
        )

        # 启动阶段的并行任务：skill 判定 + highlight 关键词预判 + 主推理（agent 相关性判定）
        side_tasks: list[tuple[str, Callable]] = []
        if self.enable_skills:
            side_tasks.append(("skill", self._run_skill_evaluation))
        if self.highlight_precheck is not None:
            side_tasks.append(("highlight_precheck", self._run_highlight_precheck))

        if side_tasks:
            with ThreadPoolExecutor(max_workers=len(side_tasks) + 1) as executor:
                side_futures = {
                    name: executor.submit(fn) for name, fn in side_tasks
                }
                reasoning_future = executor.submit(self._run_root_agent_and_flatten)
                reasoning_future.result()
                for name, fut in side_futures.items():
                    try:
                        fut.result()
                    except Exception as e:
                        logger.exception(f"[Parallel·{name}] 启动阶段任务异常: {e}")
        else:
            self._run_root_agent_and_flatten()

        # 副路径关联展开 crawl 在子智能体命中时已开始；此处 join 后再进入 pipeline，
        # 保证 _organize_fragments / _build_evidence_parts 能拿到完整的 RelationRegistry。
        if self.enable_relations:
            self._wait_react_relations()

        # phase-0 完成的 skill 结果（快照）会被前置注入到最终 summary/merge 的 prompt；
        # judge 在 final 步骤内部触发，不在这里编排。
        if self.enable_skills and self.skill_registry:
            done_records_snapshot = self.skill_registry.get_all()
        else:
            done_records_snapshot = []
        summary_skill_context = self._build_skill_context_for_summary(done_records_snapshot)

        if self.retrieval_mode:
            answer = self._retrieval_pipeline(skill_context=summary_skill_context)
        else:
            answer = self._standard_pipeline(skill_context=summary_skill_context)

        if self.summary_clean_answer:
            logger.info(
                "[CleanAnswer] summary_clean_answer 已启用，"
                "已在最终 summary/merge 阶段一并完成清洗，跳过独立清洗调用"
            )
        elif self.clean_answer:
            answer = self._clean_answer(answer)

        if self.answer_refine:
            answer = self._refine_answer(answer)

        trace_log = self._build_trace_log()
        relevant_chapters = self._collect_relevant_chapters()

        self._shutdown_relation_executors()

        return {
            "answer": answer,
            "trace_log": trace_log,
            "relevant_chapters": relevant_chapters,
        }

    # ========================= Chunk 模式 =========================

    @staticmethod
    def _join_chunk_side_futures(
        skill_future: Future | None,
        precheck_future: Future | None,
    ) -> None:
        """等待 chunk 并行侧任务完成（与 Thread.join 等价，供 finally 兜底）。"""
        if skill_future is not None:
            skill_future.result()
        if precheck_future is not None:
            precheck_future.result()

    def _run_chunk_mode(self) -> dict:
        """Chunk 模式：程序化分块 -> 并行推理（可选关联展开 + 流式调度）-> 分批汇总。

        流式调度（enable_relations=True 时）：
          1. 所有原始 chunk 并发跑 _reason_on_chunk LLM（chunk_pool）
          2. LLM 命中 relevant_headings 且 chunk 含 clause.json → 触发 RelationCrawler
             多跳并发拉取关联条款（relation_pool）
          3. 关联条款按 chunk_size 切分为派生 chunk → 提交到 chunk_pool 同样跑 LLM
          4. 调度线程按原始 chunk 顺序消费 ready slot 的 parts（self + derived），
             凑够 summary_batch_size 即提交 batch summary（batch_pool）
          5. 不带关联的 chunk 不会被带关联的 chunk 阻塞——只要 prefix 全部 ready
             就立刻喂入下一个 batch
          6. 所有 batch 完成后递归合并到最终 answer
        """
        logger.info(
            f"AgentGraph 启动 [Chunk模式 chunk_size={self.chunk_size}, "
            f"relations={self.enable_relations}]: 问题={self.question[:50]}..."
        )

        chunks = build_knowledge_chunks(self.knowledge_root, self.chunk_size)

        for chunk in chunks:
            logger.info(
                f"[Chunk] === 第 {chunk.index} 块知识内容 "
                f"({len(chunk.content)} 字符, {len(chunk.directories)} 个目录) ===\n"
                f"{chunk.content}\n"
                f"[Chunk] === 第 {chunk.index} 块结束 ==="
            )

        if not chunks:
            logger.warning("[Chunk] 未生成任何知识块，回退到标准模式")
            self._shutdown_relation_executors()
            return {
                "answer": "知识目录为空，无法生成回答。",
                "trace_log": "",
                "relevant_chapters": [],
            }

        self._chunk_directories = []
        for chunk in chunks:
            self._chunk_directories.extend(chunk.directories)

        # 使用 ThreadPoolExecutor.submit（与 verbose_logger 的 copy_context patch 对齐），
        # 使 Phase-0 skill / HighlightPrecheck 与主会话共用 ContextVar（verbose trace 等）。
        side_workers = (1 if self.enable_skills else 0) + (
            1 if self.highlight_precheck is not None else 0
        )
        side_pool: ThreadPoolExecutor | None = None
        skill_future: Future | None = None
        precheck_future: Future | None = None
        if side_workers > 0:
            side_pool = ThreadPoolExecutor(
                max_workers=side_workers, thread_name_prefix="chunk-side",
            )
            if self.enable_skills:
                skill_future = side_pool.submit(self._run_skill_evaluation)
            if self.highlight_precheck is not None:
                precheck_future = side_pool.submit(self._run_highlight_precheck)

        # 三种走法：
        #   (A) reduce_queue 模式且 batch_size > 0：走统一 ReducePipeline，
        #       生产侧（chunk LLM + 关联展开 + 派生 chunk LLM）边产出边入队，
        #       中间 batch summary 与生产侧并发，收口时 final merge 内部 join skill 线程。
        #   (B) 关联展开 + layered：走原 _chunk_streaming_pipeline。
        #   (C) 其他：走原 _chunk_reason_phase 同步等齐。
        use_reduce_queue = (
            self.summary_pipeline_mode == "reduce_queue"
            and self.summary_batch_size > 0
        )
        if (
            self.summary_pipeline_mode == "reduce_queue"
            and self.summary_batch_size <= 0
        ):
            logger.warning(
                "[ReducePipeline] summary_batch_size<=0，reduce_queue 模式无法激活，"
                "回退到 layered"
            )

        try:
            if use_reduce_queue:
                # ReducePipeline 自己管 final merge，且会在 final merge 触发时 join skill；
                # HighlightPrecheck 的 orphan 注入也下沉到 pipeline 内部（需要在 wait_and_finalize
                # 收口之前完成 part 投递），以保持与分批压缩的顺序一致。
                try:
                    answer = self._chunk_reduce_queue_pipeline(
                        chunks,
                        skill_future=skill_future,
                        precheck_future=precheck_future,
                    )
                finally:
                    self._join_chunk_side_futures(skill_future, precheck_future)
            else:
                try:
                    if self.enable_relations and self.relation_crawler is not None:
                        batch_outputs, ordered_parts = self._chunk_streaming_pipeline(chunks)
                    else:
                        ordered_parts = self._chunk_reason_phase(chunks)
                        batch_outputs = None
                finally:
                    # 必须先等齐 precheck 才能保证 RelationRegistry 稳定、
                    # _build_chunk_orphan_part 能看到完整的 orphan 集合。
                    self._join_chunk_side_futures(skill_future, precheck_future)

                if self.enable_skills and self.skill_registry:
                    done_records_snapshot = self.skill_registry.get_all()
                else:
                    done_records_snapshot = []
                summary_skill_context = self._build_skill_context_for_summary(done_records_snapshot)

                # 把 HighlightPrecheck 主动判定命中、但未被任何 chunk slot 收纳的 RelationFragment
                # 作为额外 part 追加到 summary 递归入口：
                #   - 若 batch_outputs 非空（走过流式 BATCH_SUMMARY），追加到 batch_outputs 尾部，
                #     随后与其他 batch 摘要一起进 _recursive_batch_reduce(layer=2)：总数 ≤ batch_size
                #     时 orphan 原文直进 final merge 的 prompt，否则再过一轮 BATCH_SUMMARY。
                #   - 若未过流式 BATCH_SUMMARY（summary_batch_size==0 等路径），直接追加到
                #     ordered_parts，由 _chunk_finalize_summary 决定 final merge 时的处理。
                orphan_part = self._build_chunk_orphan_part()
                if orphan_part:
                    if batch_outputs is not None:
                        batch_outputs.append(orphan_part)
                    else:
                        ordered_parts.append(orphan_part)

                if batch_outputs is not None:
                    # 流式调度已经把 ordered_parts 凑批跑完 BATCH_SUMMARY，
                    # 此处只剩"对 batch_outputs 做 final merge"。
                    # 把 batch_outputs 喂回 _recursive_batch_reduce，逻辑会自动判断
                    # 是否需要再压一层（>batch_size 时递归，否则直接 final merge）。
                    answer = self._recursive_batch_reduce(
                        batch_outputs, layer=2, skill_context=summary_skill_context,
                    )
                else:
                    answer = self._chunk_finalize_summary(ordered_parts, summary_skill_context)
        finally:
            if side_pool is not None:
                side_pool.shutdown(wait=True)

        if self.summary_clean_answer:
            logger.info(
                "[CleanAnswer] summary_clean_answer 已启用，"
                "已在最终 summary/merge 阶段一并完成清洗，跳过独立清洗调用"
            )
        elif self.clean_answer:
            answer = self._clean_answer(answer)

        if self.answer_refine:
            answer = self._refine_answer(answer)

        trace_log = self._build_chunk_trace_log(chunks)
        relevant_chapters = self._collect_relevant_chapters()

        self._shutdown_relation_executors()

        return {
            "answer": answer,
            "trace_log": trace_log,
            "relevant_chapters": relevant_chapters,
        }

    # ------------------- 流式调度（enable_relations=True） -------------------

    def _chunk_streaming_pipeline(
        self, chunks: list[KnowledgeChunk]
    ) -> tuple[list[str] | None, list[str]]:
        """三相并发流式 pipeline。

        返回 (batch_outputs, ordered_parts)：
          - summary_batch_size > 0 时：batch_outputs 为按顺序完成的 batch summaries 列表
            （已经被分批 LLM 压缩过），ordered_parts 同样填充以便 trace；
          - summary_batch_size == 0 时：batch_outputs 为 None，ordered_parts 是按顺序
            的全部 parts（原始 + 派生），由调用方自行做 final merge。
        """
        total_chunks = len(chunks)
        slots = [_OrderedSlot(parent_index=c.index, self_chunk=c) for c in chunks]
        self._chunk_slots = slots

        # chunk_pool 容量需同时容纳原始 + 估计派生 chunk 数
        chunk_pool_size = min(max(total_chunks * 2, 16), 100)
        chunk_pool = ThreadPoolExecutor(
            max_workers=chunk_pool_size, thread_name_prefix="chunk-llm",
        )

        batch_size = max(0, int(self.summary_batch_size))
        bs_pool: ThreadPoolExecutor | None = None
        if batch_size > 0:
            bs_pool = ThreadPoolExecutor(max_workers=5, thread_name_prefix="bs-stream")

        # 原始 chunk LLM 提交
        for chunk, slot in zip(chunks, slots):
            chunk_pool.submit(
                self._streaming_run_self, chunk, slot, total_chunks, chunk_pool,
            )

        ordered_parts: list[str] = []
        batch_outputs: list[str] = []

        if bs_pool is not None:
            ordered_parts, batch_outputs = self._streaming_orchestrate(
                slots, batch_size, bs_pool,
            )
        else:
            # 无分批：等所有 slot 完成后，按顺序拉 parts
            for slot in slots:
                slot.derived_done.wait()
            for slot in slots:
                ordered_parts.extend(self._render_slot_parts(slot, total_chunks))

        chunk_pool.shutdown(wait=True)
        if bs_pool is not None:
            bs_pool.shutdown(wait=True)

        # 汇总 _chunk_relevant_headings / _chunk_reasoning_results 与 trace_log 兼容
        self._chunk_reasoning_results = [
            slot.self_result if slot.self_result else {
                "relevant_headings": [], "analysis": "", "pitfalls": [],
            }
            for slot in slots
        ]
        for slot in slots:
            if slot.self_result:
                hs = slot.self_result.get("relevant_headings", []) or []
                self._chunk_relevant_headings.extend(hs)
            for d_idx in sorted(slot.derived_results.keys()):
                d_res = slot.derived_results[d_idx]
                hs = d_res.get("relevant_headings", []) or []
                self._chunk_relevant_headings.extend(hs)

        if bs_pool is not None:
            return batch_outputs, ordered_parts
        return None, ordered_parts

    def _streaming_run_self(
        self,
        chunk: KnowledgeChunk,
        slot: _OrderedSlot,
        total_chunks: int,
        chunk_pool: ThreadPoolExecutor,
    ) -> None:
        """worker：原始 chunk 自身 LLM → 触发关联展开 → 派生 chunk 入 chunk_pool。"""
        try:
            result = self._reason_on_chunk(chunk, total_chunks)
        except Exception as e:
            logger.error(f"[Streaming] chunk {chunk.index} 自身推理异常: {e}")
            result = {
                "relevant_headings": [],
                "analysis": f"（chunk {chunk.index} 推理失败: {e}）",
                "pitfalls": [],
            }
        slot.self_result = result
        slot.self_done.set()

        headings = result.get("relevant_headings") or []
        if not headings:
            slot.relation_done.set()
            slot.derived_done.set()
            return

        # 触发关联展开：把 relevant_headings 映射回 chunk 内带 clause.json 的目录
        targets = self._resolve_relation_targets(chunk, headings)
        if not targets:
            slot.relation_done.set()
            slot.derived_done.set()
            return

        parent_assessment = self._summarize_assessment_for_crawl(result)
        # crawl 在独立 dispatch 池跑（避免阻塞 chunk_pool worker）
        self._relation_dispatch_executor.submit(  # type: ignore[union-attr]
            self._streaming_run_relations,
            chunk, slot, targets, parent_assessment, chunk_pool,
        )

    def _streaming_run_relations(
        self,
        chunk: KnowledgeChunk,
        slot: _OrderedSlot,
        targets: list[str],
        parent_assessment: str,
        chunk_pool: ThreadPoolExecutor,
    ) -> None:
        """worker：对 chunk 内每个命中目录跑 RelationCrawler.crawl，合并 fragments。"""
        all_fragments: list[RelationFragment] = []
        try:
            for target_dir in targets:
                fragments = self.relation_crawler.crawl(  # type: ignore[union-attr]
                    source_chunk_index=chunk.index,
                    source_dir=target_dir,
                    parent_assessment=parent_assessment,
                )
                all_fragments.extend(fragments)
        except Exception as e:
            logger.error(f"[Streaming] chunk {chunk.index} 关联展开异常: {e}")
            all_fragments = []

        slot.relation_fragments = all_fragments

        if not all_fragments:
            slot.relation_done.set()
            slot.derived_done.set()
            return

        derived_chunks = split_relations_into_chunks(
            fragments=all_fragments,
            chunk_size=self.chunk_size,
            parent_chunk=chunk,
            start_derived_seq=1,
            knowledge_root=self.knowledge_root,
        )
        slot.derived_chunks = derived_chunks

        if not derived_chunks:
            slot.relation_done.set()
            slot.derived_done.set()
            return

        with slot.derived_lock:
            slot.pending_derived = len(derived_chunks)
            slot.completed_derived = 0
        slot.relation_done.set()

        # 派生 chunk 提交到 chunk_pool 跑同款 LLM
        for d_chunk in derived_chunks:
            chunk_pool.submit(
                self._streaming_run_derived, chunk, slot, d_chunk,
            )

    def _streaming_run_derived(
        self,
        parent_chunk: KnowledgeChunk,
        slot: _OrderedSlot,
        d_chunk: KnowledgeChunk,
    ) -> None:
        """worker：派生 chunk LLM 推理。"""
        try:
            # total_chunks 用 parent_index 配合 derived_seq 渲染（不影响 LLM 行为，仅日志）
            result = self._reason_on_chunk(d_chunk, total_chunks=0)
        except Exception as e:
            logger.error(
                f"[Streaming] derived chunk parent={parent_chunk.index} "
                f"seq={d_chunk.derived_seq} 推理异常: {e}"
            )
            result = {
                "relevant_headings": [],
                "analysis": f"（派生 chunk {parent_chunk.index}.{d_chunk.derived_seq} 推理失败: {e}）",
                "pitfalls": [],
            }
        slot.derived_results[d_chunk.derived_seq] = result
        with slot.derived_lock:
            slot.completed_derived += 1
            done = slot.completed_derived >= slot.pending_derived
        if done:
            slot.derived_done.set()

    def _streaming_orchestrate(
        self,
        slots: list[_OrderedSlot],
        batch_size: int,
        bs_pool: ThreadPoolExecutor,
    ) -> tuple[list[str], list[str]]:
        """head-of-line 调度：按 slot 顺序消费 ready parts，凑够 batch_size 提交 BATCH_SUMMARY。

        每个 slot 必须等三个 Event 全部 set 才算 ready；不 ready 时阻塞，但已就绪的前缀
        chunk 仍会立刻被打包为 batch（因为我们逐 slot 顺序处理：等到 slot[k] 全 ready 后，
        slot[0..k] 的 parts 都已纳入 pending_parts，凑够即 submit）。
        """
        total_chunks_estimate = len(slots)  # 仅用于 part 头部的 "i/N" 标签
        ordered_parts: list[str] = []
        pending_parts: list[str] = []
        batch_futures = []
        batch_index = 0

        def _flush_batch():
            nonlocal batch_index
            if not pending_parts:
                return
            batch_index += 1
            this_parts = pending_parts.copy()
            pending_parts.clear()
            fut = bs_pool.submit(
                self._streaming_submit_batch, batch_index, this_parts,
            )
            batch_futures.append(fut)

        for slot in slots:
            slot.derived_done.wait()
            slot_parts = self._render_slot_parts(slot, total_chunks_estimate)
            ordered_parts.extend(slot_parts)
            for part in slot_parts:
                pending_parts.append(part)
                if len(pending_parts) >= batch_size:
                    _flush_batch()

        _flush_batch()  # tail

        batch_outputs: list[str] = []
        for fut in batch_futures:
            try:
                batch_outputs.append(fut.result())
            except Exception as e:
                logger.error(f"[Streaming] batch summary future 异常: {e}")
                batch_outputs.append(f"（流式 batch summary 异常: {e}）")

        logger.info(
            f"[Streaming] 流式 batch 调度完成：共 {len(batch_futures)} 个 batch，"
            f"原始+派生 parts 总计 {len(ordered_parts)} 条"
        )
        return ordered_parts, batch_outputs

    # ------------------- reduce_queue 模式（chunk + relations 通用） -------------------

    def _chunk_reduce_queue_pipeline(
        self,
        chunks: list[KnowledgeChunk],
        skill_future: Future | None,
        precheck_future: Future | None = None,
    ) -> str:
        """reduce_queue 模式的 chunk 入口：复用 chunk LLM / RelationCrawler / 派生 chunk LLM
        三相生产侧逻辑，但用 ReducePipeline 接管所有 batch summary 调度。

        与 _chunk_streaming_pipeline 的差异：
          - 不再按 slot 顺序 head-of-line 凑批；任何 part 完成即 submit_part 入队，
            队列长度凑齐 batch_size 立即 flush 一批中间 BATCH_SUMMARY。
          - 中间 BATCH_SUMMARY 输出回灌当新 part 继续凑批；多次压缩在同一 pipeline 内完成，
            层间无同步点。
          - final merge 触发时由 pipeline 调用闭包，闭包内部 join skill 线程并构造 skill_context。
          - _chunk_slots 仍按原方式填充，trace 渲染逻辑无需改动；
            self._reduce_pipeline 引用保留，trace 末尾追加流水线轨迹。
        """
        total_chunks = len(chunks)
        slots = [_OrderedSlot(parent_index=c.index, self_chunk=c) for c in chunks]
        self._chunk_slots = slots

        chunk_pool_size = min(max(total_chunks * 2, 16), 100)
        chunk_pool = ThreadPoolExecutor(
            max_workers=chunk_pool_size, thread_name_prefix="chunk-llm",
        )

        has_relations = self.enable_relations and self.relation_crawler is not None

        def _final_merge(parts: list[ReducePart]) -> str:
            # final merge 内部等待 skill，确保 skill_context 完整
            if skill_future is not None:
                skill_future.result()
            if self.enable_skills and self.skill_registry:
                done_records = self.skill_registry.get_all()
            else:
                done_records = []
            skill_ctx = self._build_skill_context_for_summary(done_records)
            summaries = [p.text for p in parts]
            return self._batch_final_merge(
                summaries, layer=0, skill_context=skill_ctx,
            )

        intermediate_prompt = self._bake_pure_model_reference_into_template(
            BATCH_SUMMARY_PROMPT
        )
        intermediate_system = self._augment_system_for_extract(
            BATCH_REDUCE_SYSTEM_PROMPT, BATCH_SUMMARY_SYSTEM_PROMPT
        )
        pipeline = ReducePipeline(
            batch_size=self.summary_batch_size,
            intermediate_prompt=intermediate_prompt,
            intermediate_system_prompt=intermediate_system,
            final_merge_callable=_final_merge,
            question=self.question,
            vendor=self.vendor,
            model=self.model,
            max_part_depth=self.reduce_max_part_depth,
            intermediate_prompt_vars={
                "user": "BATCH_SUMMARY_PROMPT",
                "system": self._extract_system_prompt_vars("BATCH_SUMMARY_SYSTEM_PROMPT"),
            },
            logger_label="ReduceQueue·Chunk",
        )
        self._reduce_pipeline = pipeline

        # HighlightPrecheck 的 orphan 注入：在 pipeline 收口前以"独立生产者"的方式投递，
        # 为其保留一个 producer slot（在任意 chunk producer_done 前就 inc，避免
        # active_producers 提前归零被 _is_settled_locked 误判为已收口）。
        orphan_reserved = precheck_future is not None
        if orphan_reserved:
            pipeline.producer_inc(1)

        try:
            for chunk, slot in zip(chunks, slots):
                pipeline.producer_inc(1)
                chunk_pool.submit(
                    self._reduce_run_self,
                    chunk, slot, total_chunks, chunk_pool, pipeline, has_relations,
                )

            if orphan_reserved:
                def _inject_orphan():
                    try:
                        if precheck_future is not None:
                            precheck_future.result()
                        orphan_text = self._build_chunk_orphan_part()
                        if orphan_text:
                            pipeline.submit_part(ReducePart(
                                text=orphan_text,
                                source_label="highlight-precheck-orphan",
                                depth=0,
                            ))
                    finally:
                        pipeline.producer_done(1)

                threading.Thread(
                    target=_inject_orphan,
                    name="precheck-orphan-inject",
                    daemon=True,
                ).start()

            answer = pipeline.wait_and_finalize()
        finally:
            chunk_pool.shutdown(wait=True)

        # 与流式版本一致：回填 trace 用的 _chunk_reasoning_results / _chunk_relevant_headings
        self._chunk_reasoning_results = [
            slot.self_result if slot.self_result else {
                "relevant_headings": [], "analysis": "", "pitfalls": [],
            }
            for slot in slots
        ]
        for slot in slots:
            if slot.self_result:
                hs = slot.self_result.get("relevant_headings", []) or []
                self._chunk_relevant_headings.extend(hs)
            for d_idx in sorted(slot.derived_results.keys()):
                d_res = slot.derived_results[d_idx]
                hs = d_res.get("relevant_headings", []) or []
                self._chunk_relevant_headings.extend(hs)

        return answer

    def _reduce_run_self(
        self,
        chunk: KnowledgeChunk,
        slot: _OrderedSlot,
        total_chunks: int,
        chunk_pool: ThreadPoolExecutor,
        pipeline: ReducePipeline,
        has_relations: bool,
    ) -> None:
        """worker：根 chunk LLM → 投递 self part → 决定派生 token 调度。

        producer token 流转规则：入口已 producer_inc(1) 占位；
          - 无关联 / 无命中 / 无 targets：归还该 1，结束；
          - 有派生 M 个：先 producer_inc(M)，再归还根 token (净增 M-1)；
            每个派生 worker 完成时各归还 1。
        """
        try:
            result = self._reason_on_chunk(chunk, total_chunks)
        except Exception as e:
            logger.error(f"[ReduceQueue] chunk {chunk.index} 自身推理异常: {e}")
            result = {
                "relevant_headings": [],
                "analysis": f"（chunk {chunk.index} 推理失败: {e}）",
                "pitfalls": [],
            }
        slot.self_result = result
        slot.self_done.set()

        # 立即投递 self part 到 reduce 队列
        self_part_text = self._render_chunk_part(
            chunk_label=f"{slot.parent_index}/{total_chunks}",
            result=result,
        )
        pipeline.submit_part(ReducePart(
            text=self_part_text,
            source_label=f"chunk-{slot.parent_index}",
            depth=0,
        ))

        headings = result.get("relevant_headings") or []
        if not has_relations or not headings:
            slot.relation_done.set()
            slot.derived_done.set()
            pipeline.producer_done(1)
            return

        targets = self._resolve_relation_targets(chunk, headings)
        if not targets:
            slot.relation_done.set()
            slot.derived_done.set()
            pipeline.producer_done(1)
            return

        parent_assessment = self._summarize_assessment_for_crawl(result)
        # crawl 在独立 dispatch 池跑，避免阻塞 chunk_pool worker
        self._relation_dispatch_executor.submit(  # type: ignore[union-attr]
            self._reduce_run_relations,
            chunk, slot, targets, parent_assessment, chunk_pool, pipeline,
        )

    def _reduce_run_relations(
        self,
        chunk: KnowledgeChunk,
        slot: _OrderedSlot,
        targets: list[str],
        parent_assessment: str,
        chunk_pool: ThreadPoolExecutor,
        pipeline: ReducePipeline,
    ) -> None:
        """worker：crawl + 切派生 chunk → 决定是否生成派生 token。"""
        all_fragments: list[RelationFragment] = []
        try:
            for target_dir in targets:
                fragments = self.relation_crawler.crawl(  # type: ignore[union-attr]
                    source_chunk_index=chunk.index,
                    source_dir=target_dir,
                    parent_assessment=parent_assessment,
                )
                all_fragments.extend(fragments)
        except Exception as e:
            logger.error(f"[ReduceQueue] chunk {chunk.index} 关联展开异常: {e}")
            all_fragments = []

        slot.relation_fragments = all_fragments

        if not all_fragments:
            slot.relation_done.set()
            slot.derived_done.set()
            pipeline.producer_done(1)
            return

        derived_chunks = split_relations_into_chunks(
            fragments=all_fragments,
            chunk_size=self.chunk_size,
            parent_chunk=chunk,
            start_derived_seq=1,
            knowledge_root=self.knowledge_root,
        )
        slot.derived_chunks = derived_chunks

        if not derived_chunks:
            slot.relation_done.set()
            slot.derived_done.set()
            pipeline.producer_done(1)
            return

        with slot.derived_lock:
            slot.pending_derived = len(derived_chunks)
            slot.completed_derived = 0
        slot.relation_done.set()

        # 把根 chunk 的 1 token 替换为 M 个派生 token：
        #   先 +M（avoid race window where producers 暂时归零触发误收口），再归还原 1
        m = len(derived_chunks)
        pipeline.producer_inc(m)
        pipeline.producer_done(1)

        for d_chunk in derived_chunks:
            chunk_pool.submit(
                self._reduce_run_derived, chunk, slot, d_chunk, pipeline,
            )

    def _reduce_run_derived(
        self,
        parent_chunk: KnowledgeChunk,
        slot: _OrderedSlot,
        d_chunk: KnowledgeChunk,
        pipeline: ReducePipeline,
    ) -> None:
        """worker：派生 chunk LLM → 投递 derived part → 归还派生 token。"""
        try:
            result = self._reason_on_chunk(d_chunk, total_chunks=0)
        except Exception as e:
            logger.error(
                f"[ReduceQueue] derived chunk parent={parent_chunk.index} "
                f"seq={d_chunk.derived_seq} 推理异常: {e}"
            )
            result = {
                "relevant_headings": [],
                "analysis": f"（派生 chunk {parent_chunk.index}.{d_chunk.derived_seq} 推理失败: {e}）",
                "pitfalls": [],
            }
        slot.derived_results[d_chunk.derived_seq] = result

        derived_part_text = self._render_chunk_part(
            chunk_label=f"{parent_chunk.index}.{d_chunk.derived_seq} (关联派生)",
            result=result,
        )
        pipeline.submit_part(ReducePart(
            text=derived_part_text,
            source_label=f"derived-{parent_chunk.index}.{d_chunk.derived_seq}",
            depth=0,
        ))

        with slot.derived_lock:
            slot.completed_derived += 1
            done = slot.completed_derived >= slot.pending_derived
        if done:
            slot.derived_done.set()
        pipeline.producer_done(1)

    def _streaming_submit_batch(self, batch_index: int, parts: list[str]) -> str:
        """流式 batch summary：复用 BATCH_SUMMARY_PROMPT 模板（中间提炼层，不带 clean）。"""
        batch_content = "\n\n".join(parts)
        # total_batches 在流式下未知；用占位"?"，模板对该字段无强约束
        prompt = BATCH_SUMMARY_PROMPT.format(
            batch_index=batch_index,
            total_batches="?",
            question=self.question,
            batch_content=batch_content,
        )
        prompt = self._inject_pure_model_reference(prompt)
        system_prompt = self._augment_system_for_extract(
            BATCH_REDUCE_SYSTEM_PROMPT, BATCH_SUMMARY_SYSTEM_PROMPT
        )
        logger.info(
            f"[StreamingBatch] 第 {batch_index} 批 prompt 长度: {len(prompt)} 字符 "
            f"({len(parts)} 个 parts)"
        )
        try:
            with step_scope(
                f"streaming_batch_summary#{batch_index}",
                prompt_vars={
                    "user": "BATCH_SUMMARY_PROMPT",
                    "system": self._extract_system_prompt_vars("BATCH_SUMMARY_SYSTEM_PROMPT"),
                },
            ):
                return chat(
                    prompt, vendor=self.vendor, model=self.model,
                    system=system_prompt,
                )
        except Exception as e:
            logger.error(f"[StreamingBatch] 第 {batch_index} 批失败: {e}")
            return f"（第 {batch_index} 批流式压缩失败）\n原始内容:\n{batch_content}"

    # ------------------- 流式调度·辅助 -------------------

    def _render_slot_parts(
        self, slot: _OrderedSlot, total_chunks: int,
    ) -> list[str]:
        """把一个已完成的 slot 渲染为按顺序的 parts 列表（self + derived[1..M]）。"""
        parts: list[str] = []
        if slot.self_chunk and slot.self_result:
            parts.append(self._render_chunk_part(
                chunk_label=f"{slot.parent_index}/{total_chunks}",
                result=slot.self_result,
            ))
        for d_seq in sorted(slot.derived_results.keys()):
            d_res = slot.derived_results[d_seq]
            parts.append(self._render_chunk_part(
                chunk_label=f"{slot.parent_index}.{d_seq} (关联派生)",
                result=d_res,
            ))
        return parts

    @staticmethod
    def _render_chunk_part(chunk_label: str, result: dict) -> str:
        """与原 _chunk_reason_phase 内 part 渲染保持一致（label 之外完全同构）。"""
        analysis = result.get("analysis", "") or ""
        headings = result.get("relevant_headings", []) or []
        pitfalls = result.get("pitfalls", []) or []
        lines = [f"### 知识块 {chunk_label}"]
        if headings:
            lines.append(f"**引用章节：** {' | '.join(headings)}")
        lines.append(analysis)
        if pitfalls:
            lines.append("\n**【易错点提醒】**")
            for p in pitfalls:
                lines.append(f"- {p}")
        return "\n".join(lines)

    def _resolve_relation_targets(
        self, chunk: KnowledgeChunk, relevant_headings: list[str],
    ) -> list[str]:
        """把 LLM 输出的 relevant_headings 映射回 chunk 内带 clause.json 的目录绝对路径。

        匹配策略：把 chunk.heading_paths 与 chunk.directories 一一对应组成 (label, dir) 表，
        用 ' > '.join(hp) 与 cleaned heading 字符串相等命中。仅返回包含 clause.json 的 dir。
        """
        if not chunk.directories:
            return []
        # 构建查找表：label → dir
        label_to_dir: dict[str, str] = {}
        for hp, d in zip(chunk.heading_paths, chunk.directories):
            label = " > ".join(hp)
            label_to_dir[label] = d

        seen_dirs: set[str] = set()
        targets: list[str] = []
        for raw_h in relevant_headings:
            cleaned = (raw_h or "").strip().strip("【】").strip()
            d = label_to_dir.get(cleaned)
            if not d:
                continue
            if d in seen_dirs:
                continue
            if not os.path.isfile(os.path.join(d, "clause.json")):
                continue
            seen_dirs.add(d)
            targets.append(d)
        return targets

    @staticmethod
    def _summarize_assessment_for_crawl(result: dict, limit: int = 400) -> str:
        """把 chunk LLM 的 analysis 截断为给关联 crawl 用的 parent_assessment。

        优先用 analysis；analysis 为空时退化为 relevant_headings 拼接。
        """
        analysis = (result.get("analysis") or "").strip()
        if analysis:
            return analysis if len(analysis) <= limit else analysis[:limit] + "…"
        headings = result.get("relevant_headings") or []
        return " | ".join(headings) if headings else ""

    def _chunk_reason_phase(self, chunks) -> list[str]:
        """Chunk 模式 phase 1：并行推理每个 chunk → 解析 JSON → 拼成 summary_parts。

        与 skill_eval 并行执行，不做最终合并（合并交给 orchestrator 与 skill 上下文一并处理）。
        """
        total_chunks = len(chunks)
        logger.info(f"[Chunk] 开始并行推理 {total_chunks} 个知识块")

        raw_results = [None] * total_chunks
        with ThreadPoolExecutor(max_workers=min(total_chunks, 100)) as executor:
            futures = {
                executor.submit(
                    self._reason_on_chunk, chunk, total_chunks
                ): chunk.index - 1
                for chunk in chunks
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    raw_results[idx] = future.result()
                except Exception as e:
                    logger.error(f"[Chunk] 第 {idx+1} 块推理异常: {e}")
                    raw_results[idx] = {
                        "relevant_headings": [],
                        "analysis": f"（第 {idx+1} 块推理失败: {e}）",
                        "pitfalls": [],
                    }

        self._chunk_reasoning_results = raw_results

        summary_parts = []
        for i, result in enumerate(raw_results):
            analysis = result.get("analysis", "")
            headings = result.get("relevant_headings", [])
            pitfalls = result.get("pitfalls", [])

            if headings:
                self._chunk_relevant_headings.extend(headings)

            part_lines = [f"### 知识块 {i+1}/{total_chunks}"]
            if headings:
                part_lines.append(f"**引用章节：** {' | '.join(headings)}")
            part_lines.append(analysis)
            if pitfalls:
                part_lines.append("\n**【易错点提醒】**")
                for p in pitfalls:
                    part_lines.append(f"- {p}")

            summary_parts.append("\n".join(part_lines))

        logger.info(
            f"[Chunk] 所有 {total_chunks} 个知识块推理完成，"
            f"共引用 {len(self._chunk_relevant_headings)} 个相关章节"
        )
        return summary_parts

    def _chunk_finalize_summary(self, summary_parts: list[str], skill_context: str = "") -> str:
        """Chunk 模式 phase 2：把 phase 1 的 summary_parts 汇总为最终 answer。

        skill_context 会被注入到最终合并 prompt，与 _orchestrate_summary_with_double_check 配合使用。
        """
        if self.summary_batch_size > 0:
            return self._recursive_batch_reduce(summary_parts, layer=1, skill_context=skill_context)
        else:
            return self._batch_final_merge(summary_parts, layer=1, skill_context=skill_context)

    def _reason_on_chunk(self, chunk, total_chunks: int) -> dict:
        """对单个 chunk 调用 LLM 推理，返回解析后的 JSON dict"""
        template = (CHUNK_REASONING_WITH_PITFALLS_PROMPT
                    if self.check_pitfalls
                    else CHUNK_REASONING_PROMPT)
        system_template = (CHUNK_REASONING_WITH_PITFALLS_SYSTEM_PROMPT
                           if self.check_pitfalls
                           else CHUNK_REASONING_SYSTEM_PROMPT)
        prompt = template.format(
            question=self.question,
            chunk_index=chunk.index,
            total_chunks=total_chunks,
            chunk_content=chunk.content,
        )
        mode_label = "推理+易错点" if self.check_pitfalls else "推理"
        logger.info(
            f"[Chunk] 第 {chunk.index}/{total_chunks} 块{mode_label} "
            f"prompt 长度: {len(prompt)} 字符"
        )
        try:
            prompt_var = (
                "CHUNK_REASONING_WITH_PITFALLS_PROMPT"
                if self.check_pitfalls
                else "CHUNK_REASONING_PROMPT"
            )
            system_prompt_var = (
                "CHUNK_REASONING_WITH_PITFALLS_SYSTEM_PROMPT"
                if self.check_pitfalls
                else "CHUNK_REASONING_SYSTEM_PROMPT"
            )
            with step_scope(
                f"chunk_reason#{chunk.index}",
                prompt_vars={
                    "user": prompt_var,
                    "system": ["SUMMARY_EXTRACT_SYSTEM_PROMPT", system_prompt_var],
                },
            ):
                response = chat(prompt, vendor=self.vendor, model=self.model,
                                system=SUMMARY_EXTRACT_SYSTEM_PROMPT + "\n\n" + system_template)
            return self._parse_chunk_json(response, chunk.index)
        except Exception as e:
            logger.error(f"[Chunk] 第 {chunk.index} 块 LLM 调用失败: {e}")
            return {
                "relevant_headings": [],
                "analysis": f"（第 {chunk.index} 块推理失败: {e}）",
                "pitfalls": [],
            }

    @staticmethod
    def _parse_chunk_json(response: str, chunk_index: int) -> dict:
        """解析 chunk 推理的 JSON 响应，解析失败时降级为纯文本"""
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        try:
            result = json.loads(cleaned.strip())
            if not isinstance(result.get("relevant_headings"), list):
                result["relevant_headings"] = []
            if not isinstance(result.get("analysis"), str):
                result["analysis"] = str(result.get("analysis", ""))
            if not isinstance(result.get("pitfalls"), list):
                result["pitfalls"] = []
            return result
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(
                f"[Chunk] 第 {chunk_index} 块 JSON 解析失败，降级为纯文本: {e}"
            )
            return {
                "relevant_headings": [],
                "analysis": response,
                "pitfalls": [],
            }

    def _build_chunk_trace_log(self, chunks) -> str:
        """为 chunk 模式构建追踪日志，包含各块引用的相关章节、关联展开命中条款、派生 chunk。"""
        lines = [
            f"=== Chunk 模式 (chunk_size={self.chunk_size}, "
            f"relations={self.enable_relations}) ===",
            "",
        ]

        results = getattr(self, '_chunk_reasoning_results', None) or []
        slots = getattr(self, '_chunk_slots', None) or []
        slots_by_index = {s.parent_index: s for s in slots}

        for chunk in chunks:
            lines.append(f"--- Chunk {chunk.index} ({len(chunk.content)} 字符) ---")
            lines.append(f"  包含目录节点:")
            for hp in chunk.heading_paths:
                lines.append(f"    - {' > '.join(hp)}")

            idx = chunk.index - 1
            if idx < len(results) and results[idx]:
                result = results[idx]
                headings = result.get("relevant_headings", [])
                pitfalls = result.get("pitfalls", [])
                if headings:
                    lines.append(f"  LLM 判定相关章节:")
                    for h in headings:
                        lines.append(f"    * {h}")
                else:
                    lines.append(f"  LLM 判定: 本块与问题无关")
                if pitfalls:
                    lines.append(f"  识别易错点: {len(pitfalls)} 条")

            slot = slots_by_index.get(chunk.index)
            if slot and slot.relation_fragments:
                lines.append(
                    f"  关联展开命中: {len(slot.relation_fragments)} 个条款 "
                    f"(派生 chunk {len(slot.derived_chunks)} 个)"
                )
                for frag in slot.relation_fragments:
                    heading_label = " > ".join(frag.heading_path) or frag.clause_full_name
                    reason = (frag.parent_assessment or "")[:80]
                    lines.append(
                        f"    * [hop={frag.hop_depth}/{frag.source}] "
                        f"{heading_label} (policy={frag.policy_id} clause={frag.clause_id})"
                    )
                    if reason:
                        lines.append(f"        上层关联性: {reason}")
                for d_chunk in slot.derived_chunks:
                    d_res = slot.derived_results.get(d_chunk.derived_seq, {})
                    d_headings = d_res.get("relevant_headings", []) or []
                    lines.append(
                        f"    - 派生 chunk {chunk.index}.{d_chunk.derived_seq} "
                        f"({len(d_chunk.content)} 字符, 命中 {len(d_headings)} 个 sub-heading)"
                    )
            lines.append("")

        # reduce_queue 模式：把 ReducePipeline 的中间 batch 轨迹追加到 trace 末尾
        if (
            self.summary_pipeline_mode == "reduce_queue"
            and self._reduce_pipeline is not None
        ):
            lines.append("")
            lines.append(self._reduce_pipeline.render_trace_section())

        return "\n".join(lines)

    def _standard_pipeline(self, skill_context: str = "") -> str:
        """标准模式的后处理管线：直接汇总（无意图解决）"""
        if self.summary_batch_size > 0:
            return self._batched_summary(skill_context=skill_context)
        else:
            return self._final_summary(skill_context=skill_context)

    def _retrieval_pipeline(self, skill_context: str = "") -> str:
        """召回模式的后处理管线：程序化梳理 -> 总结"""
        organized_parts = self._organize_fragments()

        if not organized_parts:
            logger.warning("[Retrieval] 未召回任何相关知识片段，回退到标准模式汇总")
            return self._standard_pipeline(skill_context=skill_context)

        logger.info(f"[Retrieval] 程序化梳理完成，共 {len(organized_parts)} 个知识片段")

        if self.summary_batch_size > 0:
            return self._retrieval_batched_summary(organized_parts, skill_context=skill_context)
        else:
            return self._retrieval_final_summary(organized_parts, skill_context=skill_context)

    def _organize_fragments(self) -> list[str]:
        """程序化梳理：按层级排序、去重、格式化（不调用 LLM）。

        启用 enable_relations 时，把每个 fragment 对应的 RelationRegistry.by_dir 命中
        条款 inline 渲染到 fragment.content 末尾，作为同一个 organized part 投递到
        retrieval pipeline；这样关联条款不会破坏 batch_size 切分逻辑，但完整保留了
        引用上下文。
        """
        fragments = self.retrieval_registry.get_all()
        if not fragments:
            return []

        sorted_fragments = sorted(
            fragments,
            key=lambda f: [
                (natural_dir_sort_key(seg), seg) for seg in f.heading_path
            ],
        )

        organized = []
        rendered_dirs: set[str] = set()
        for frag in sorted_fragments:
            heading_label = " > ".join(frag.heading_path)
            text = f"【{heading_label}】\n{frag.content}"
            if self.check_pitfalls:
                dir_pitfalls = self.pitfalls_registry.get_by_dir(frag.directory_path)
                if dir_pitfalls:
                    pitfalls_str = "\n".join(f"- {p}" for p in dir_pitfalls)
                    text += f"\n\n**易错点提醒：**\n{pitfalls_str}"

            relation_block = self._render_inline_relations(frag.directory_path)
            if relation_block:
                text += "\n\n" + relation_block
            rendered_dirs.add(os.path.normpath(frag.directory_path))

            organized.append(text)

        orphan_part = self._build_orphan_relation_part(rendered_dirs)
        if orphan_part:
            organized.append(orphan_part)

        logger.info(
            f"[Retrieval] 知识片段梳理：{len(fragments)} 个片段，"
            f"层级深度范围 {min(len(f.heading_path) for f in fragments)}"
            f"-{max(len(f.heading_path) for f in fragments)}"
        )
        return organized

    def _render_inline_relations(self, directory: str) -> str:
        """把 RelationRegistry 中按 dir 命中的 RelationFragment 渲染为追加到 fragment 末尾的字符串。

        与 chunk 派生 chunk 不同：副路径下关联条款不会单独成 part（避免破坏 batch_size 切分），
        而是 inline 追加到对应 fragment 的同一 part 内。呈现规则：
          - 父章节定位只在 header 显示一次（同一 directory 下多条关联共享同一父章节）；
          - 每条关联用三行同构的【标签 · 内容】块呈现一条"父章节 → 命中关键词 → 本条款"的溯源链，
            其中父章节已在 header 出现，每条关联自身只需再补【命中关键词】+【关联条款位置】；
          - 三条坐标格式完全一致，LLM 能识别出它们是同一坐标系下的三个不同点；
          - 不输出 hop / source / policyId / clauseId 等技术元数据——对业务推理无帮助。
        """
        if not self.enable_relations or self.relation_registry is None:
            return ""
        rels = self.relation_registry.get_by_dir(directory)
        if not rels:
            return ""
        parent_label = build_parent_location_label(self.knowledge_root, directory)
        lines = ["**关联条款命中：**"]
        if parent_label:
            lines.append(f"【来自父章节 · {parent_label}】")
        for r in rels:
            target_label = build_target_location_label(r)
            lines.append("")
            if r.highlighted:
                lines.append(f"【命中关键词 · {r.highlighted}】")
            lines.append(f"【关联条款位置 · {target_label}】")
            if r.highlighted:
                lines.append(f"**{r.highlighted}的关联知识细节如下：**")
            else:
                lines.append("**关联知识细节如下：**")
            lines.append(r.content or "（关联条款内容为空）")
        return "\n".join(lines)

    def _collect_relevant_chapters(self) -> list[str]:
        """从所有智能体结果或 chunk 目录中提取去重的相关章节序号"""
        chapters = set()

        if self._chunk_relevant_headings:
            for heading in self._chunk_relevant_headings:
                cleaned = heading.strip().strip("【】")
                parts = [p.strip() for p in cleaned.split(">")]
                for part in parts:
                    if "_" in part:
                        chapter_num = part.split("_")[0]
                        if chapter_num and chapter_num[0].isdigit():
                            chapters.add(chapter_num)
        else:
            all_dirs = []
            for r in self.all_results:
                all_dirs.extend(r.relevant_dirs)

            if self.retrieval_mode and self.retrieval_registry:
                for frag in self.retrieval_registry.get_all():
                    if frag.directory_path not in all_dirs:
                        all_dirs.append(frag.directory_path)

            for dir_path in all_dirs:
                rel = os.path.relpath(dir_path, self.knowledge_root)
                if rel == ".":
                    continue
                parts = rel.replace("\\", "/").split("/")
                leaf = parts[-1]
                if "_" in leaf:
                    chapter_num = leaf.split("_")[0]
                    chapters.add(chapter_num)

        return sorted(chapters, key=natural_dir_sort_key)

    def _flatten_results(self, result: AgentResult) -> list[AgentResult]:
        flat = [result]
        for child in result.child_results:
            flat.extend(self._flatten_results(child))
        return flat

    def _build_evidence_parts(self) -> list[str]:
        """构建子智能体结果的文本片段列表。

        启用关联展开时：
          - 把 r.relevant_dirs 中每个目录命中的 RelationFragment inline 追加到对应 part 末尾；
          - 所有 r.relevant_dirs 之外残留的 RelationFragment（主要来源：HighlightPrecheck
            主动预判命中、但其父章节未被任何子智能体探索到）补一个独立的 "额外关联" part，
            确保这些关联条款仍能进入 final summary 的 prompt。
        """
        agent_parts = []
        rendered_dirs: set[str] = set()
        for r in self.all_results:
            evidence_str = "\n".join(f"  - {e}" for e in r.evidence) if r.evidence else "  （无证据）"
            part = (
                f"### {r.agent_id} | 探索目录: {r.explored_dir}\n"
                f"- 结论: {r.conclusion}\n"
                f"- 证据:\n{evidence_str}"
            )
            if self.check_pitfalls and r.pitfalls:
                pitfalls_str = "\n".join(f"  - {p}" for p in r.pitfalls)
                part += f"\n- 易错点提醒:\n{pitfalls_str}"

            if self.enable_relations and self.relation_registry is not None:
                rel_blocks = [
                    self._render_inline_relations(d) for d in r.relevant_dirs
                ]
                rel_blocks = [b for b in rel_blocks if b]
                if rel_blocks:
                    part += "\n\n" + "\n\n".join(rel_blocks)
                for d in r.relevant_dirs:
                    rendered_dirs.add(os.path.normpath(d))
            agent_parts.append(part)

        orphan_part = self._build_orphan_relation_part(rendered_dirs)
        if orphan_part:
            agent_parts.append(orphan_part)

        return agent_parts

    def _build_chunk_orphan_part(self) -> str:
        """chunk 模式专用：汇总所有未被任何 chunk slot 收纳的 RelationFragment。

        Chunk 模式的"主循环"是派生 chunk：每个 chunk 的 _streaming_run_relations 把它自己
        crawl 到的 fragments 切成派生 chunk 跑 LLM。HighlightPrecheck 在 RelationRegistry
        里先行占位的 (policy_id, clause_id)，会使后续 chunk 的 crawl 去重命中后直接丢弃
        （registry.has 返回 True），因此这些 fragment 既不会出现在任何 slot.relation_fragments
        中，也不会成为派生 chunk 的输入——不在这里单独兜底就会在 summary 阶段丢失。

        本方法把这些 orphan 按 `_format_relation_fragment_text` 统一渲染，拼成一个字符串 part，
        作为额外 part 追加到 summary 递归入口（layered 模式下进 _recursive_batch_reduce
        layer=2；reduce_queue 模式下进 pipeline.submit_part），**不再经过 chunk LLM 提炼**，
        但仍会按 summary 链路的普通 part 参与后续可能的 BATCH_SUMMARY / final merge 调度：
          - 如果 summary 入口的 part 总数 ≤ batch_size，orphan 原文直进 final merge 的 prompt；
          - 否则 orphan 会跟其他已压缩过的摘要一起再走一轮 BATCH_SUMMARY_PROMPT，然后递归收口。
        """
        if not self.enable_relations or self.relation_registry is None:
            return ""
        all_frags = self.relation_registry.get_all()
        if not all_frags:
            return ""

        rendered_keys: set[tuple[str, str]] = set()
        slots = getattr(self, "_chunk_slots", None) or []
        for slot in slots:
            for f in slot.relation_fragments:
                rendered_keys.add((f.policy_id, f.clause_id))

        orphan_frags = [
            f for f in all_frags
            if (f.policy_id, f.clause_id) not in rendered_keys
        ]
        if not orphan_frags:
            return ""

        from reasoner.v2.chunk_builder import _format_relation_fragment_text
        blocks = [
            _format_relation_fragment_text(f, knowledge_root=self.knowledge_root)
            for f in orphan_frags
        ]
        logger.info(
            f"[HighlightPrecheck] chunk 模式补回 {len(orphan_frags)} 个未被任何 chunk slot "
            f"收纳的 RelationFragment（原文未经 chunk LLM / StreamingBatch 提炼，"
            f"作为额外 part 追加到 summary 递归入口，后续按 batch_size 规则"
            f"可能参与 BATCH_SUMMARY 或直进 final merge）"
        )
        return (
            "### HighlightPrecheck 额外发现的关联条款"
            "（父 chunk 未被判定相关，但关键词预判认为可能对答案有帮助；原文未经 chunk LLM 提炼）\n\n"
            + "\n\n".join(blocks)
        )

    def _build_orphan_relation_part(self, rendered_dirs: set[str]) -> str:
        """把 RelationRegistry 中父章节不在 rendered_dirs 的关联条款汇总为一个独立 part。

        设计动机：RelationRegistry 由三类来源写入——
          - chunk LLM 命中后触发的 crawl（parent_chunk_index=chunk.index）；
          - ReactAgent 命中后触发的 crawl（parent_chunk_index=-1）；
          - HighlightPrecheck 在启动阶段主动判定需要展开的 crawl（parent_chunk_index=-1）。
        前两类的 parent_dir 天然落在某个 agent 的 relevant_dirs 里，会被 inline 渲染；
        HighlightPrecheck 的核心价值恰恰是"父 chunk / agent 都没判相关、但关联知识可能
        依然重要"——这类 fragment 的 parent_dir 不在 rendered_dirs 里，若不额外兜底就
        会在 final summary prompt 里丢失。这里按 parent_dir 分组走相同的 inline 渲染器，
        保持与主循环一致的呈现风格，只是统一挂在"额外关联"标题下。
        """
        if not self.enable_relations or self.relation_registry is None:
            return ""
        all_frags = self.relation_registry.get_all()
        if not all_frags:
            return ""

        orphan_dirs: list[str] = []
        seen_norm: set[str] = set()
        for f in all_frags:
            if not f.parent_dir:
                continue
            norm = os.path.normpath(f.parent_dir)
            if norm in rendered_dirs or norm in seen_norm:
                continue
            seen_norm.add(norm)
            orphan_dirs.append(f.parent_dir)

        if not orphan_dirs:
            return ""

        blocks: list[str] = []
        for d in orphan_dirs:
            block = self._render_inline_relations(d)
            if block:
                blocks.append(block)
        if not blocks:
            return ""

        header = (
            "### HighlightPrecheck 额外发现的关联条款"
            "（其父章节未被任何子智能体主动探索到，但 LLM 关键词预判认为可能对答案有帮助）"
        )
        return header + "\n\n" + "\n\n".join(blocks)

    def _final_summary(self, skill_context: str = "") -> str:
        """调用 LLM 做最终汇总（单次全量），并行触发 judge → 按需 all-in-answer"""
        agent_parts = self._build_evidence_parts()
        agent_results_text = "\n\n".join(agent_parts) if agent_parts else "（无）"

        html_template: str | None = None
        if self.summary_clean_answer:
            if self.think_mode:
                template = SUMMARY_AND_CLEAN_THINK_PROMPT
                html_template = SUMMARY_AND_CLEAN_THINK_HTML_PROMPT
                prompt_var = "SUMMARY_AND_CLEAN_THINK_PROMPT"
                html_prompt_var = "SUMMARY_AND_CLEAN_THINK_HTML_PROMPT"
                mode_label = "汇总+清洗一体·Think"
            else:
                template = SUMMARY_AND_CLEAN_PROMPT
                prompt_var = "SUMMARY_AND_CLEAN_PROMPT"
                html_prompt_var = None
                mode_label = "汇总+清洗一体"
        else:
            template = SUMMARY_PROMPT
            prompt_var = "SUMMARY_PROMPT"
            html_prompt_var = None
            mode_label = "纯汇总"
        fmt_kwargs = dict(question=self.question, agent_results=agent_results_text)
        prompt = template.format(**fmt_kwargs)
        prompt = self._append_skill_context_to_prompt(prompt, skill_context)
        prompt = self._inject_pure_model_reference(prompt)
        system_prompt = self._augment_system_for_answer(self.answer_system_prompt)

        # think_mode 期望 JSON；同步构造对应的 HTML 完整 prompt 用于格式兜底重试，
        # 让重试链路与首轮 100% 共享内容/呈现段，仅切换"输出格式"段。
        prompt_html: str | None = None
        if html_template is not None:
            prompt_html = html_template.format(**fmt_kwargs)
            prompt_html = self._append_skill_context_to_prompt(prompt_html, skill_context)
            prompt_html = self._inject_pure_model_reference(prompt_html)

        logger.info(f"[Summary·{mode_label}] system prompt 长度: {len(system_prompt)} 字符")
        logger.info(f"[Summary·{mode_label}] user prompt 长度: {len(prompt)} 字符")
        logger.info(f"[Summary·{mode_label}] user prompt 内容:\n{prompt}")

        expects_think_answer_json = self.summary_clean_answer and self.think_mode

        def _do_summary() -> str:
            try:
                return self._chat_final_with_format_retry(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    step_label="final_summary",
                    expects_think_answer_json=expects_think_answer_json,
                    html_retry_prompt=prompt_html,
                    prompt_vars={
                        "user": prompt_var,
                        "system": self._answer_system_prompt_vars(),
                    },
                    html_retry_prompt_vars={
                        "user": html_prompt_var,
                        "system": self._answer_system_prompt_vars(),
                    } if html_prompt_var else None,
                )
            except Exception as e:
                logger.error(f"最终汇总 LLM 调用失败: {e}")
                fallback_parts = []
                for r in self.all_results:
                    if r.conclusion:
                        fallback_parts.append(f"- [{r.agent_id}] {r.conclusion}")
                return "最终汇总生成失败，以下为各智能体的原始结论：\n" + "\n".join(fallback_parts)

        return self._finalize_with_double_check(
            final_summary_callable=_do_summary,
            judge_evidence=agent_results_text,
            stage_label=f"Summary·{mode_label}",
        )

    def _reduce_queue_run_over_parts(
        self,
        *,
        parts: list[str],
        source_label_prefix: str,
        intermediate_prompt: str,
        final_merge: Callable[[list[str]], str],
        logger_label: str,
        intermediate_system_prompt: str | None = None,
    ) -> str:
        """退化场景的 reduce_queue：parts 已经全部就位，按完成顺序入队 → 凑批 → 回灌 → final merge。

        与 chunk 模式相比，本路径没有"动态产出"，但仍享受统一架构 + 后续中间 batch 之间
        的回灌并发性。standard / retrieval 两个入口共享本 helper，区别只在
        intermediate_prompt 与 final_merge 的具体实现。

        intermediate_system_prompt：调用方可显式覆盖 ReducePipeline 的默认
        `BATCH_REDUCE_SYSTEM_PROMPT + BATCH_SUMMARY_SYSTEM_PROMPT`，用于在 pure_model
        开关启用时把 EXTRACT 指令插到 BATCH_REDUCE 与 BATCH_SUMMARY（含输出格式说明）
        之间——`_augment_system_for_extract(role_header, format_body)` 已保证此顺序。
        None 时保持 pipeline 内置默认。
        """
        if not parts:
            return final_merge([])

        pipeline_kwargs = dict(
            batch_size=self.summary_batch_size,
            intermediate_prompt=intermediate_prompt,
            final_merge_callable=lambda rps: final_merge([p.text for p in rps]),
            question=self.question,
            vendor=self.vendor,
            model=self.model,
            max_part_depth=self.reduce_max_part_depth,
            intermediate_prompt_vars={
                "user": (
                    "RETRIEVAL_BATCH_SUMMARY_PROMPT"
                    if "Retrieval" in logger_label
                    else "BATCH_SUMMARY_PROMPT"
                ),
                "system": self._extract_system_prompt_vars("BATCH_SUMMARY_SYSTEM_PROMPT"),
            },
            logger_label=logger_label,
        )
        if intermediate_system_prompt is not None:
            pipeline_kwargs["intermediate_system_prompt"] = intermediate_system_prompt
        pipeline = ReducePipeline(**pipeline_kwargs)
        # 这里 parts 已就绪，没有"未完成生产者"，直接全部 submit_part 即可
        for i, text in enumerate(parts, 1):
            pipeline.submit_part(ReducePart(
                text=text,
                source_label=f"{source_label_prefix}-{i}",
                depth=0,
            ))
        # 把 pipeline 暂存以便事后追加 trace（chunk 模式独占 self._reduce_pipeline，
        # standard / retrieval 不写 _chunk_slots，trace 章节由各自模块决定是否渲染）
        self._reduce_pipeline = pipeline
        return pipeline.wait_and_finalize()

    def _batched_summary(self, skill_context: str = "") -> str:
        """分批并行压缩总结。
        reduce_queue 模式下走 ReducePipeline（统一流水线，输出回灌、无层间同步）；
        layered 模式仍走 _recursive_batch_reduce 同步分层。
        """
        agent_parts = self._build_evidence_parts()
        if (
            self.summary_pipeline_mode == "reduce_queue"
            and self.summary_batch_size > 0
        ):
            # pure_model 开启时把参考块预烘焙进 intermediate 模板、并覆盖系统提示
            # 追加 EXTRACT 指令；关闭时原样通过。
            intermediate_prompt = self._bake_pure_model_reference_into_template(
                BATCH_SUMMARY_PROMPT
            )
            intermediate_system = self._augment_system_for_extract(
                BATCH_REDUCE_SYSTEM_PROMPT, BATCH_SUMMARY_SYSTEM_PROMPT
            )
            return self._reduce_queue_run_over_parts(
                parts=agent_parts,
                source_label_prefix="agent",
                intermediate_prompt=intermediate_prompt,
                final_merge=lambda summaries: self._batch_final_merge(
                    summaries, layer=0, skill_context=skill_context,
                ),
                logger_label="ReduceQueue·Standard",
                intermediate_system_prompt=intermediate_system,
            )
        return self._recursive_batch_reduce(agent_parts, layer=1, skill_context=skill_context)

    def _recursive_batch_reduce(self, parts: list[str], layer: int, skill_context: str = "") -> str:
        batch_size = self.summary_batch_size

        if len(parts) <= batch_size:
            return self._batch_final_merge(parts, layer, skill_context=skill_context)

        batches = [
            parts[i:i + batch_size]
            for i in range(0, len(parts), batch_size)
        ]
        total_batches = len(batches)

        logger.info(
            f"[BatchSummary] 第 {layer} 层：{len(parts)} 条 → "
            f"分为 {total_batches} 批（batch_size={batch_size}）"
        )

        def _summarize_batch(batch_index: int, batch_parts: list[str]) -> str:
            batch_content = "\n\n".join(batch_parts)
            prompt = BATCH_SUMMARY_PROMPT.format(
                batch_index=batch_index,
                total_batches=total_batches,
                question=self.question,
                batch_content=batch_content,
            )
            prompt = self._inject_pure_model_reference(prompt)
            system_prompt = self._augment_system_for_extract(
                BATCH_REDUCE_SYSTEM_PROMPT, BATCH_SUMMARY_SYSTEM_PROMPT
            )
            logger.info(
                f"[BatchSummary] 第 {layer} 层 第 {batch_index}/{total_batches} 批 "
                f"prompt 长度: {len(prompt)} 字符"
            )
            try:
                with step_scope(
                    f"batch_summary·L{layer}·b{batch_index}",
                    prompt_vars={
                        "user": "BATCH_SUMMARY_PROMPT",
                        "system": self._extract_system_prompt_vars("BATCH_SUMMARY_SYSTEM_PROMPT"),
                    },
                ):
                    return chat(prompt, vendor=self.vendor, model=self.model,
                                system=system_prompt)
            except Exception as e:
                logger.error(f"[BatchSummary] 第 {layer} 层 第 {batch_index} 批失败: {e}")
                return f"（第 {batch_index} 批压缩失败）\n原始内容:\n{batch_content}"

        batch_summaries = [None] * total_batches
        with ThreadPoolExecutor(max_workers=min(total_batches, 5)) as executor:
            futures = {
                executor.submit(_summarize_batch, i + 1, batch): i
                for i, batch in enumerate(batches)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    batch_summaries[idx] = future.result()
                except Exception as e:
                    logger.error(f"[BatchSummary] 第 {layer} 层 batch {idx+1} 异常: {e}")
                    batch_summaries[idx] = f"（第 {idx+1} 批执行异常: {e}）"

        logger.info(
            f"[BatchSummary] 第 {layer} 层完成，产出 {total_batches} 个摘要，"
            f"进入下一层"
        )

        return self._recursive_batch_reduce(batch_summaries, layer + 1, skill_context=skill_context)

    def _batch_final_merge(self, summaries: list[str], layer: int, skill_context: str = "") -> str:
        numbered = self._format_batch_summaries_for_merge(summaries)
        html_template: str | None = None
        if self.summary_clean_answer:
            if self.think_mode:
                template = BATCH_MERGE_AND_CLEAN_THINK_PROMPT
                html_template = BATCH_MERGE_AND_CLEAN_THINK_HTML_PROMPT
                prompt_var = "BATCH_MERGE_AND_CLEAN_THINK_PROMPT"
                html_prompt_var = "BATCH_MERGE_AND_CLEAN_THINK_HTML_PROMPT"
                mode_label = "合并+清洗一体·Think"
            else:
                template = BATCH_MERGE_AND_CLEAN_PROMPT
                prompt_var = "BATCH_MERGE_AND_CLEAN_PROMPT"
                html_prompt_var = None
                mode_label = "合并+清洗一体"
        else:
            template = BATCH_MERGE_PROMPT
            prompt_var = "BATCH_MERGE_PROMPT"
            html_prompt_var = None
            mode_label = "纯合并"
        fmt_kwargs = dict(question=self.question, batch_summaries=numbered)
        merge_prompt = template.format(**fmt_kwargs)
        merge_prompt = self._append_skill_context_to_prompt(merge_prompt, skill_context)
        merge_prompt = self._inject_pure_model_reference(merge_prompt)

        # think_mode 期望 JSON；同步构造对应的 HTML 完整 prompt（user + system 都换 schema）
        # 让 HTML 兜底重试与首轮共享内容/呈现段，仅"输出格式"段切换。
        merge_prompt_html: str | None = None
        if html_template is not None:
            merge_prompt_html = html_template.format(**fmt_kwargs)
            merge_prompt_html = self._append_skill_context_to_prompt(merge_prompt_html, skill_context)
            merge_prompt_html = self._inject_pure_model_reference(merge_prompt_html)

        logger.info(
            f"[BatchMerge] 第 {layer} 层（最终合并·{mode_label}）："
            f"{len(summaries)} 条摘要，prompt 长度: {len(merge_prompt)} 字符"
        )
        logger.info(f"[BatchMerge] prompt 内容:\n{merge_prompt}")

        expects_think_answer_json = self.summary_clean_answer and self.think_mode

        def _do_merge() -> str:
            try:
                if expects_think_answer_json:
                    system_prompt = self._augment_system_for_answer(
                        self.answer_system_prompt,
                        "## 输出格式约束\n" + BATCH_MERGE_AND_CLEAN_THINK_SYSTEM_PROMPT,
                    )
                    system_prompt_html = self._augment_system_for_answer(
                        self.answer_system_prompt,
                        "## 输出格式约束\n" + BATCH_MERGE_AND_CLEAN_THINK_HTML_SYSTEM_PROMPT,
                    )
                else:
                    system_prompt = self._augment_system_for_answer(
                        self.answer_system_prompt
                    )
                    system_prompt_html = None
                return self._chat_final_with_format_retry(
                    prompt=merge_prompt,
                    system_prompt=system_prompt,
                    step_label=f"batch_final_merge·L{layer}",
                    expects_think_answer_json=expects_think_answer_json,
                    html_retry_prompt=merge_prompt_html,
                    html_retry_system=system_prompt_html,
                    prompt_vars={
                        "user": prompt_var,
                        "system": self._answer_system_prompt_vars(
                            "BATCH_MERGE_AND_CLEAN_THINK_SYSTEM_PROMPT"
                            if expects_think_answer_json else ""
                        ),
                    },
                    html_retry_prompt_vars={
                        "user": html_prompt_var,
                        "system": self._answer_system_prompt_vars(
                            "BATCH_MERGE_AND_CLEAN_THINK_HTML_SYSTEM_PROMPT"
                        ),
                    } if html_prompt_var else None,
                )
            except Exception as e:
                logger.error(f"[BatchMerge] 最终合并失败: {e}")
                return "分批合并失败，以下为各摘要：\n" + numbered

        return self._finalize_with_double_check(
            final_summary_callable=_do_merge,
            judge_evidence=numbered,
            stage_label=f"BatchMerge·L{layer}·{mode_label}",
        )

    def _retrieval_final_summary(self, organized_parts: list[str], skill_context: str = "") -> str:
        knowledge_text = "\n\n".join(organized_parts)
        html_template: str | None = None
        if self.summary_clean_answer:
            if self.think_mode:
                template = RETRIEVAL_SUMMARY_AND_CLEAN_THINK_PROMPT
                html_template = RETRIEVAL_SUMMARY_AND_CLEAN_THINK_HTML_PROMPT
                prompt_var = "RETRIEVAL_SUMMARY_AND_CLEAN_THINK_PROMPT"
                html_prompt_var = "RETRIEVAL_SUMMARY_AND_CLEAN_THINK_HTML_PROMPT"
                mode_label = "汇总+清洗一体·Think"
            else:
                template = RETRIEVAL_SUMMARY_AND_CLEAN_PROMPT
                prompt_var = "RETRIEVAL_SUMMARY_AND_CLEAN_PROMPT"
                html_prompt_var = None
                mode_label = "汇总+清洗一体"
        else:
            template = RETRIEVAL_SUMMARY_PROMPT
            prompt_var = "RETRIEVAL_SUMMARY_PROMPT"
            html_prompt_var = None
            mode_label = "纯汇总"
        fmt_kwargs = dict(question=self.question, organized_knowledge=knowledge_text)
        prompt = template.format(**fmt_kwargs)
        prompt = self._append_skill_context_to_prompt(prompt, skill_context)
        prompt = self._inject_pure_model_reference(prompt)
        system_prompt = self._augment_system_for_answer(self.answer_system_prompt)

        prompt_html: str | None = None
        if html_template is not None:
            prompt_html = html_template.format(**fmt_kwargs)
            prompt_html = self._append_skill_context_to_prompt(prompt_html, skill_context)
            prompt_html = self._inject_pure_model_reference(prompt_html)

        logger.info(f"[RetrievalSummary·{mode_label}] prompt 长度: {len(prompt)} 字符")
        logger.info(f"[RetrievalSummary·{mode_label}] prompt 内容:\n{prompt}")

        expects_think_answer_json = self.summary_clean_answer and self.think_mode

        def _do_summary() -> str:
            try:
                return self._chat_final_with_format_retry(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    step_label="retrieval_final_summary",
                    expects_think_answer_json=expects_think_answer_json,
                    html_retry_prompt=prompt_html,
                    prompt_vars={
                        "user": prompt_var,
                        "system": self._answer_system_prompt_vars(),
                    },
                    html_retry_prompt_vars={
                        "user": html_prompt_var,
                        "system": self._answer_system_prompt_vars(),
                    } if html_prompt_var else None,
                )
            except Exception as e:
                logger.error(f"[RetrievalSummary] 召回总结失败: {e}")
                return "召回总结生成失败，以下为召回的知识片段：\n" + knowledge_text

        return self._finalize_with_double_check(
            final_summary_callable=_do_summary,
            judge_evidence=knowledge_text,
            stage_label=f"RetrievalSummary·{mode_label}",
        )

    def _retrieval_batched_summary(self, organized_parts: list[str], skill_context: str = "") -> str:
        if (
            self.summary_pipeline_mode == "reduce_queue"
            and self.summary_batch_size > 0
        ):
            intermediate_prompt = self._bake_pure_model_reference_into_template(
                RETRIEVAL_BATCH_SUMMARY_PROMPT
            )
            intermediate_system = self._augment_system_for_extract(
                BATCH_REDUCE_SYSTEM_PROMPT, BATCH_SUMMARY_SYSTEM_PROMPT
            )
            return self._reduce_queue_run_over_parts(
                parts=organized_parts,
                source_label_prefix="frag",
                intermediate_prompt=intermediate_prompt,
                final_merge=lambda summaries: self._retrieval_batch_final_merge(
                    summaries, layer=0, skill_context=skill_context,
                ),
                logger_label="ReduceQueue·Retrieval",
                intermediate_system_prompt=intermediate_system,
            )
        return self._retrieval_recursive_batch_reduce(organized_parts, layer=1, skill_context=skill_context)

    def _retrieval_recursive_batch_reduce(self, parts: list[str], layer: int, skill_context: str = "") -> str:
        batch_size = self.summary_batch_size

        if len(parts) <= batch_size:
            return self._retrieval_batch_final_merge(parts, layer, skill_context=skill_context)

        batches = [
            parts[i:i + batch_size]
            for i in range(0, len(parts), batch_size)
        ]
        total_batches = len(batches)

        logger.info(
            f"[RetrievalBatch] 第 {layer} 层：{len(parts)} 条 → "
            f"分为 {total_batches} 批（batch_size={batch_size}）"
        )

        def _summarize_batch(batch_index: int, batch_parts: list[str]) -> str:
            batch_content = "\n\n".join(batch_parts)
            prompt = RETRIEVAL_BATCH_SUMMARY_PROMPT.format(
                batch_index=batch_index,
                total_batches=total_batches,
                question=self.question,
                batch_content=batch_content,
            )
            prompt = self._inject_pure_model_reference(prompt)
            system_prompt = self._augment_system_for_extract(
                BATCH_REDUCE_SYSTEM_PROMPT, BATCH_SUMMARY_SYSTEM_PROMPT
            )
            logger.info(
                f"[RetrievalBatch] 第 {layer} 层 第 {batch_index}/{total_batches} 批 "
                f"prompt 长度: {len(prompt)} 字符"
            )
            try:
                with step_scope(
                    f"retrieval_batch_summary·L{layer}·b{batch_index}",
                    prompt_vars={
                        "user": "RETRIEVAL_BATCH_SUMMARY_PROMPT",
                        "system": self._extract_system_prompt_vars("BATCH_SUMMARY_SYSTEM_PROMPT"),
                    },
                ):
                    return chat(prompt, vendor=self.vendor, model=self.model,
                                system=system_prompt)
            except Exception as e:
                logger.error(f"[RetrievalBatch] 第 {layer} 层 第 {batch_index} 批失败: {e}")
                return f"（第 {batch_index} 批压缩失败）\n原始内容:\n{batch_content}"

        batch_summaries = [None] * total_batches
        with ThreadPoolExecutor(max_workers=min(total_batches, 5)) as executor:
            futures = {
                executor.submit(_summarize_batch, i + 1, batch): i
                for i, batch in enumerate(batches)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    batch_summaries[idx] = future.result()
                except Exception as e:
                    logger.error(f"[RetrievalBatch] 第 {layer} 层 batch {idx+1} 异常: {e}")
                    batch_summaries[idx] = f"（第 {idx+1} 批执行异常: {e}）"

        logger.info(
            f"[RetrievalBatch] 第 {layer} 层完成，产出 {total_batches} 个摘要，进入下一层"
        )
        return self._retrieval_recursive_batch_reduce(batch_summaries, layer + 1, skill_context=skill_context)

    def _retrieval_batch_final_merge(self, summaries: list[str], layer: int, skill_context: str = "") -> str:
        numbered = self._format_batch_summaries_for_merge(summaries)
        html_template: str | None = None
        if self.summary_clean_answer:
            if self.think_mode:
                template = RETRIEVAL_BATCH_MERGE_AND_CLEAN_THINK_PROMPT
                html_template = RETRIEVAL_BATCH_MERGE_AND_CLEAN_THINK_HTML_PROMPT
                prompt_var = "RETRIEVAL_BATCH_MERGE_AND_CLEAN_THINK_PROMPT"
                html_prompt_var = "RETRIEVAL_BATCH_MERGE_AND_CLEAN_THINK_HTML_PROMPT"
                mode_label = "合并+清洗一体·Think"
            else:
                template = RETRIEVAL_BATCH_MERGE_AND_CLEAN_PROMPT
                prompt_var = "RETRIEVAL_BATCH_MERGE_AND_CLEAN_PROMPT"
                html_prompt_var = None
                mode_label = "合并+清洗一体"
        else:
            template = RETRIEVAL_BATCH_MERGE_PROMPT
            prompt_var = "RETRIEVAL_BATCH_MERGE_PROMPT"
            html_prompt_var = None
            mode_label = "纯合并"
        fmt_kwargs = dict(question=self.question, batch_summaries=numbered)
        merge_prompt = template.format(**fmt_kwargs)
        merge_prompt = self._append_skill_context_to_prompt(merge_prompt, skill_context)
        merge_prompt = self._inject_pure_model_reference(merge_prompt)
        system_prompt = self._augment_system_for_answer(self.answer_system_prompt)

        merge_prompt_html: str | None = None
        if html_template is not None:
            merge_prompt_html = html_template.format(**fmt_kwargs)
            merge_prompt_html = self._append_skill_context_to_prompt(merge_prompt_html, skill_context)
            merge_prompt_html = self._inject_pure_model_reference(merge_prompt_html)

        logger.info(
            f"[RetrievalBatchMerge] 第 {layer} 层（最终合并·{mode_label}）："
            f"{len(summaries)} 条摘要，prompt 长度: {len(merge_prompt)} 字符"
        )
        logger.info(f"[RetrievalBatchMerge] prompt 内容:\n{merge_prompt}")

        expects_think_answer_json = self.summary_clean_answer and self.think_mode

        def _do_merge() -> str:
            try:
                return self._chat_final_with_format_retry(
                    prompt=merge_prompt,
                    system_prompt=system_prompt,
                    step_label=f"retrieval_batch_final_merge·L{layer}",
                    expects_think_answer_json=expects_think_answer_json,
                    html_retry_prompt=merge_prompt_html,
                    prompt_vars={
                        "user": prompt_var,
                        "system": self._answer_system_prompt_vars(),
                    },
                    html_retry_prompt_vars={
                        "user": html_prompt_var,
                        "system": self._answer_system_prompt_vars(),
                    } if html_prompt_var else None,
                )
            except Exception as e:
                logger.error(f"[RetrievalBatchMerge] 最终合并失败: {e}")
                return "召回分批合并失败，以下为各摘要：\n" + numbered

        return self._finalize_with_double_check(
            final_summary_callable=_do_merge,
            judge_evidence=numbered,
            stage_label=f"RetrievalBatchMerge·L{layer}·{mode_label}",
        )

    def _clean_answer(self, raw_answer: str) -> str:
        prompt = CLEAN_ANSWER_PROMPT.format(
            question=self.question,
            raw_answer=raw_answer,
        )
        logger.info(f"[CleanAnswer] prompt 长度: {len(prompt)} 字符")
        logger.info(f"[CleanAnswer] prompt 内容:\n{prompt}")
        try:
            with step_scope("clean_answer", prompt_vars={"user": "CLEAN_ANSWER_PROMPT"}):
                cleaned = chat(prompt, vendor=self.vendor, model=self.model,
                               enable_thinking=self.last_think)
            cleaned = self._postprocess_final_chat(cleaned, "clean_answer")
            logger.info("答案清洗完成")
            return cleaned
        except Exception as e:
            logger.error(f"答案清洗 LLM 调用失败，返回原始 summary: {e}")
            return raw_answer

    def _extract_user_facing_answer(self, raw: str) -> str:
        """从前序最终 answer 抽取出「面向用户的纯文本 answer 部分」。

        覆盖三种历史形态：
          1) JSON `{think, answer}`（含旧字段名 analysis/concise_answer）：
             复用 `_extract_answer_from_think_answer_json`。
          2) HTML `<think>...</think><answer>...</answer>`（首轮 JSON 失败、HTML
             兜底成功的形态）：复用 `_parse_think_answer_html` 取 answer 段。
          3) 纯文本（think_mode=False / coercion 兜底失败 / 老版本路径）：原样返回。

        任何形态都至少返回一个非 None 的字符串（空串也算合法），便于下游 refine
        prompt 直接 .format 使用。
        """
        if not raw:
            return ""
        text = raw.strip()
        if not text:
            return ""

        json_extracted = _extract_answer_from_think_answer_json(text)
        if json_extracted:
            return json_extracted

        html_parsed = _parse_think_answer_html(text)
        if html_parsed:
            ans = (html_parsed.get("answer") or "").strip()
            if ans:
                return ans

        return text

    def _refine_answer(self, raw_answer: str) -> str:
        """流水线最末一步：对最终 answer 做「结论先行 + 核心证据/因果逻辑/注意事项」
        结构化精简。

        - 输入：上一阶段最终 answer（可能是 think_mode JSON / HTML / 纯文本）。
        - LLM 仅看 user-facing answer 部分（不喂 think 草稿）。
        - 输出（写回 result["answer"]）：
            * think_mode=True ：返回 `{"think": <精简前的原 answer>, "answer": <refine 结果>}`
              JSON 字符串；下游 app.py._split_analysis_concise_answer 会自然映射，
              使响应里 think=精简前完整答案、answer=精简后结构化答案。
            * think_mode=False：直接返回 refine 后的纯文本，覆盖原 answer。
        - 失败兜底：任何异常都打 ERROR 后返回 raw_answer 原样，避免 refine 节点
          拖垮主流程。
        """
        original_answer = self._extract_user_facing_answer(raw_answer)
        if not original_answer.strip():
            logger.warning("[AnswerRefine] 抽取出的原 answer 为空，跳过 refine 直接返回原值")
            return raw_answer

        prompt = ANSWER_REFINE_PROMPT.format(
            question=self.question,
            raw_answer=original_answer,
        )
        logger.info(f"[AnswerRefine] prompt 长度: {len(prompt)} 字符")
        logger.info(f"[AnswerRefine] prompt 内容:\n{prompt}")
        try:
            with step_scope(
                "answer_refine",
                prompt_vars={
                    "user": "ANSWER_REFINE_PROMPT",
                    "system": "ANSWER_REFINE_SYSTEM_PROMPT",
                },
            ):
                refined_raw = chat(
                    prompt,
                    vendor=self.vendor,
                    model=self.model,
                    system=ANSWER_REFINE_SYSTEM_PROMPT,
                    enable_thinking=self.last_think,
                )
            refined = self._postprocess_final_chat(refined_raw, "answer_refine")
            refined = (refined or "").strip()
            if not refined:
                logger.warning("[AnswerRefine] LLM 返回空内容，沿用原 answer")
                return raw_answer

            logger.info(
                f"[AnswerRefine] 完成：原 answer {len(original_answer)} 字符 → "
                f"精简后 {len(refined)} 字符"
                f"（保留比 {len(refined) / max(len(original_answer), 1):.0%}）"
            )

            if self.think_mode:
                # think_mode：原完整 answer 归入 think 字段，refine 结果归入 answer 字段，
                # 让 app.py 的 think_mode 解析链路自然落到响应的 think / answer。
                return json.dumps(
                    {"think": original_answer, "answer": refined},
                    ensure_ascii=False,
                )
            return refined
        except Exception as e:
            logger.error(f"[AnswerRefine] LLM 调用失败，沿用原 answer: {e}")
            return raw_answer

    def _build_trace_log(self) -> str:
        """工程化拼接所有智能体的游走路径和返回内容"""
        lines = []

        for r in self.all_results:
            lines.append(f"--- {r.agent_id} | 起始目录: {r.explored_dir} ---")
            for i, step in enumerate(r.trace, 1):
                lines.append(f"  [轮次{i}] {step.action:<15} {step.directory}")
                lines.append(f"           → {step.observation}")

            if r.conclusion:
                lines.append(f"  结论: {r.conclusion}")
            if r.evidence:
                lines.append(f"  证据: {'; '.join(r.evidence)}")
            lines.append("")

        return "\n".join(lines)
