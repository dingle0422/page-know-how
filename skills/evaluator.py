"""Skill 两步式评估器

请求进入时调用一次：
- Step 1: LLM 读外层 skills/skills.md，决定本问题需要哪些 skill（可能为空）
- Step 2: 对每个选中 skill，加载其内层 skills.md，让 LLM 生成 bash 命令
- Step 3: 沙箱执行 bash 命令，结果写入 SkillResultRegistry

Step 2 的多个 skill 之间互相独立，并行 await 提升吞吐。
"""

import asyncio
import json
import logging
import os
import re

from llm.client import chat

from .registry import SkillResultRegistry
from .runner import SkillRunner

logger = logging.getLogger(__name__)

_SKILLS_DIR = os.path.dirname(os.path.abspath(__file__))
_INDEX_DOC = os.path.join(_SKILLS_DIR, "skills.md")


_SELECT_PROMPT = """你是一个税务问答系统的 skill 调度器。下面是当前可用的 skill 索引：

==== skills/skills.md ====
{index_doc}
==== END ====

用户问题：
{question}

请判断回答这个问题前，**必须先调用**哪些 skill 来获取关键事实信息（例如标准商品名、税率、外部数据等）。
- 仅在 skill 能为问题提供关键事实支撑时才选中；如果问题靠常识或纯文本知识就能回答，返回空数组
- 同一个 skill 不要重复选中
- 严格按照下面的 JSON 数组格式输出，不要输出任何额外解释、不要包裹 markdown 代码块

输出格式（仅 JSON 数组）：
["skill_name_1", "skill_name_2"]
或
[]
"""


_GENERATE_CMD_PROMPT = """你是一个税务问答系统的 skill 命令生成器。下面是 skill `{skill_name}` 的详细文档：

==== skills/{skill_name}/skills.md ====
{detail_doc}
==== END ====

用户问题：
{question}

请基于上述文档，为这个问题生成**一行可直接在 bash 中执行**的命令，要求：
- 严格按照文档中的"调用方式"示例编写
- 把用户问题中涉及的实体（如商品名、服务名）作为参数传入
- 多个参数请用双引号包裹，包括包含中文/空格/特殊字符的参数
- 仅输出命令本身，不要任何解释、不要 markdown 代码块、不要前后缀
- 如果一行无法表达，使用 ` && ` 连接，不要换行

只输出命令本身："""


_JSON_ARRAY_RE = re.compile(r"\[.*?\]", re.DOTALL)


def _extract_json_array(text: str) -> list[str]:
    """从 LLM 响应中尽力提取 JSON 数组"""
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

    logger.warning("[Evaluator] 无法解析 LLM 选 skill 的输出，原文: %s", text[:200])
    return []


def _strip_command(text: str) -> str:
    """LLM 偶尔会带 markdown 围栏或解释，清洗为纯命令字符串"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    text = " ".join(line.strip() for line in text.splitlines() if line.strip())
    return text


def _load_index_doc() -> str:
    if not os.path.exists(_INDEX_DOC):
        logger.error("[Evaluator] skills 索引文件不存在: %s", _INDEX_DOC)
        return ""
    with open(_INDEX_DOC, "r", encoding="utf-8") as f:
        return f.read()


def _load_detail_doc(skill_name: str) -> str:
    path = os.path.join(_SKILLS_DIR, skill_name, "skills.md")
    if not os.path.exists(path):
        logger.warning("[Evaluator] skill 详细文档不存在: %s", path)
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _list_available_skills() -> set[str]:
    """扫描 skills/ 下含 skills.md 的子目录，作为合法 skill 名"""
    available: set[str] = set()
    for entry in os.listdir(_SKILLS_DIR):
        full = os.path.join(_SKILLS_DIR, entry)
        if os.path.isdir(full) and os.path.exists(os.path.join(full, "skills.md")):
            available.add(entry)
    return available


def _select_skills(question: str, vendor: str, model: str) -> list[str]:
    """Step 1: LLM 选 skill"""
    index_doc = _load_index_doc()
    if not index_doc:
        return []

    prompt = _SELECT_PROMPT.format(index_doc=index_doc, question=question)
    try:
        resp = chat(prompt, vendor=vendor, model=model)
    except Exception as e:
        logger.error("[Evaluator] Step1 LLM 调用失败: %s", e)
        return []

    selected = _extract_json_array(resp)
    available = _list_available_skills()
    valid = [s for s in selected if s in available]
    invalid = set(selected) - available
    if invalid:
        logger.warning("[Evaluator] LLM 选中了不存在的 skill，已忽略: %s", invalid)

    logger.info("[Evaluator] Step1 选中 skill: %s", valid)
    return valid


def _generate_command(skill_name: str, question: str, vendor: str, model: str) -> str:
    """Step 2: LLM 为指定 skill 生成 bash 命令"""
    detail_doc = _load_detail_doc(skill_name)
    if not detail_doc:
        return ""

    prompt = _GENERATE_CMD_PROMPT.format(
        skill_name=skill_name, detail_doc=detail_doc, question=question,
    )
    try:
        resp = chat(prompt, vendor=vendor, model=model)
    except Exception as e:
        logger.error("[Evaluator] Step2 LLM 调用失败 (skill=%s): %s", skill_name, e)
        return ""

    cmd = _strip_command(resp)
    logger.info("[Evaluator] Step2 skill=%s 生成命令: %s", skill_name, cmd[:200])
    return cmd


async def _run_one_skill(
    skill_name: str,
    question: str,
    runner: SkillRunner,
    registry: SkillResultRegistry,
    vendor: str,
    model: str,
) -> None:
    cmd = await asyncio.to_thread(_generate_command, skill_name, question, vendor, model)
    if not cmd:
        logger.warning("[Evaluator] skill=%s 命令生成失败，跳过执行", skill_name)
        return

    result = await runner.execute(cmd)
    registry.add(skill_name, cmd, result)


async def evaluate_and_run(
    question: str,
    registry: SkillResultRegistry,
    runner: SkillRunner,
    vendor: str = "aliyun",
    model: str = "deepseek-v3.2",
    skill_names: list[str] | None = None,
) -> list[str]:
    """对单个用户问题做 skill 评估与执行。

    Args:
        question:    用户问题
        registry:    结果注册表（执行结果会写入此处）
        runner:      bash 沙箱执行器
        vendor/model: LLM 厂商与模型
        skill_names: 可选；若提供则跳过 Step1，直接对这些 skill 走 Step2+执行
                     （供 double_check 阶段复用 evaluator 的 Step2 逻辑）

    Returns:
        实际执行的 skill 名列表
    """
    if skill_names is None:
        selected = await asyncio.to_thread(_select_skills, question, vendor, model)
    else:
        available = _list_available_skills()
        selected = [s for s in skill_names if s in available]

    if not selected:
        logger.info("[Evaluator] 本次问题无需调用任何 skill")
        return []

    await asyncio.gather(*[
        _run_one_skill(name, question, runner, registry, vendor, model)
        for name in selected
    ])

    return selected
