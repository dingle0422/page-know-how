"""Skill double-check 阶段

在 summary pipeline 之后、_clean_answer 之前调用。两条路径：
- 若 registry 已有 skill 结果 → 让 LLM 判断这些结果是否能改进当前 answer，能则生成增强版
- 若 registry 为空            → 让 LLM 评估当前 answer 是否仍需 skill 验证；
                                  如需则复用 evaluator 流程补一轮，再做注入优化
"""

import asyncio
import json
import logging
import re

from llm.client import chat

from .evaluator import evaluate_and_run, _list_available_skills, _load_index_doc
from .registry import SkillResultRegistry
from .runner import SkillRunner

logger = logging.getLogger(__name__)


_ENHANCE_PROMPT = """你正在为税务问答系统做最后一轮答案校验。下面是当前的初步回答与已经执行过的 skill 结果，请你严格基于 skill 结果对回答做最终修正。

==== 用户问题 ====
{question}

==== 当前回答（草稿） ====
{raw_answer}

==== 已执行的 Skill 结果 ====
{skill_context}

请判断 skill 结果是否对当前回答有帮助：
- 如果有帮助：用 skill 结果中的标准名称、税率、编码等关键事实**校正/补充**当前回答中可能不准确或缺失的部分，输出**完整的最终回答**（保持原回答的结构与口径，仅做事实层面的修正与补充，不要保留"根据 skill 结果"之类的元描述）
- 如果无帮助：原样输出当前回答的全文，不做任何修改

直接输出最终回答正文，不要任何额外解释、不要 markdown 代码块。"""


_NEED_SKILL_PROMPT = """你正在为税务问答系统做最后一轮答案校验。下面是当前的初步回答与所有可用的 skill 索引：

==== 用户问题 ====
{question}

==== 当前回答（草稿） ====
{raw_answer}

==== 可用 Skill 索引 ====
{index_doc}

请判断这个回答中是否还存在**必须依赖 skill 才能可靠确认**的事实点（例如商品的标准税收名称、适用税率等明显可由 skill 给出权威结果的内容）。
- 如果需要：输出对应 skill 名称的 JSON 数组，例如 ["standard_product_name_verification"]
- 如果不需要：输出 []

严格只输出 JSON 数组，不要任何额外解释、不要 markdown 代码块。"""


_JSON_ARRAY_RE = re.compile(r"\[.*?\]", re.DOTALL)


def _extract_json_array(text: str) -> list[str]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(x) for x in data]
    except json.JSONDecodeError:
        pass
    m = _JSON_ARRAY_RE.search(text)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, list):
                return [str(x) for x in data]
        except json.JSONDecodeError:
            pass
    return []


def _enhance_with_skill_results(
    question: str,
    raw_answer: str,
    skill_context: str,
    vendor: str,
    model: str,
) -> str:
    prompt = _ENHANCE_PROMPT.format(
        question=question, raw_answer=raw_answer, skill_context=skill_context,
    )
    try:
        enhanced = chat(prompt, vendor=vendor, model=model)
        enhanced = enhanced.strip()
        if not enhanced:
            logger.warning("[DoubleCheck] LLM 返回空，沿用原回答")
            return raw_answer
        return enhanced
    except Exception as e:
        logger.error("[DoubleCheck] 增强答案 LLM 调用失败，沿用原回答: %s", e)
        return raw_answer


def _judge_need_skill(question: str, raw_answer: str, vendor: str, model: str) -> list[str]:
    index_doc = _load_index_doc()
    if not index_doc:
        return []
    prompt = _NEED_SKILL_PROMPT.format(
        question=question, raw_answer=raw_answer, index_doc=index_doc,
    )
    try:
        resp = chat(prompt, vendor=vendor, model=model)
    except Exception as e:
        logger.error("[DoubleCheck] 判断是否需要 skill LLM 调用失败: %s", e)
        return []
    selected = _extract_json_array(resp)
    available = _list_available_skills()
    return [s for s in selected if s in available]


async def check_and_enhance(
    question: str,
    raw_answer: str,
    registry: SkillResultRegistry,
    runner: SkillRunner,
    vendor: str = "aliyun",
    model: str = "deepseek-v3.2",
) -> str:
    """double-check 入口：返回（可能已增强的）最终 answer"""
    if registry.has_any():
        skill_context = registry.format_context()
        logger.info("[DoubleCheck] registry 已有 skill 结果，进入增强流程")
        return await asyncio.to_thread(
            _enhance_with_skill_results,
            question, raw_answer, skill_context, vendor, model,
        )

    logger.info("[DoubleCheck] registry 为空，让 LLM 判断是否需要补 skill 调用")
    needed = await asyncio.to_thread(_judge_need_skill, question, raw_answer, vendor, model)
    if not needed:
        logger.info("[DoubleCheck] 当前回答无需 skill 验证，原样返回")
        return raw_answer

    logger.info("[DoubleCheck] 补充执行 skill: %s", needed)
    await evaluate_and_run(
        question=question,
        registry=registry,
        runner=runner,
        vendor=vendor,
        model=model,
        skill_names=needed,
    )

    if not registry.has_any():
        logger.warning("[DoubleCheck] 补充执行后仍无结果，原样返回")
        return raw_answer

    skill_context = registry.format_context()
    return await asyncio.to_thread(
        _enhance_with_skill_results,
        question, raw_answer, skill_context, vendor, model,
    )
