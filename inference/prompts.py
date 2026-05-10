"""inference 阶段使用的系统/用户 prompt。

设计原则：

- ``CORPUS_SYSTEM_PROMPT`` / ``CORPUS_USER_PROMPT`` 直接 import 自 ``reasoner/v3/prompts.py``，
  不复制不修改，避免两份口径漂移。
- ``REACT_SYSTEM_PROMPT`` 在 ``CORPUS_SYSTEM_PROMPT`` 上**前置**一段 ReAct 指令头：
  本轮只能基于已加载的部分知识，必须用 ``<verdict>complete|incomplete</verdict>``
  自我判定是否还需要继续读后续证据。
- ``REACT_USER_PROMPT`` 在 ``CORPUS_USER_PROMPT`` 之后追加 ``prev_think`` / ``preview``
  / ``skills`` / ``round_status`` 四个上下文块，便于多轮迭代。
"""

from __future__ import annotations

from reasoner.v3.prompts import (
    CORPUS_SYSTEM_PROMPT,
    CORPUS_USER_PROMPT,
)

__all__ = [
    "CORPUS_SYSTEM_PROMPT",
    "CORPUS_USER_PROMPT",
    "PREVIEW_SYSTEM_PROMPT",
    "PREVIEW_USER_PROMPT",
    "REACT_SYSTEM_PROMPT",
    "REACT_USER_PROMPT",
    "format_react_user_prompt",
    "format_skill_block",
    "format_preview_block",
]


# -------------------------------------------------------------- preview prompt

PREVIEW_SYSTEM_PROMPT = """\
你是一个资深财税咨询专家，正在为接下来的多轮 ReAct 推理做"前置预答"。

请基于自身常识，先做一次轻量分析（不要等到检索结果到齐）：
- <think> 段：从财税实务角度对问题做拆解，列出涉及的知识体系与回答逻辑。
- <answer> 段：给出一个保守、范围明确的预答（不超过 200 字），允许标注"待结合检索后修正"。

【绝对约束】
- 仅输出一段 <think>...</think> 和一段 <answer>...</answer>，不要任何其他文字。
- <answer> 不要编造具体政策文号或法规细节，遇到不确定处直接说"待结合检索"。
"""

PREVIEW_USER_PROMPT = """## 用户问题
{question}
"""


# -------------------------------------------------------------- react prompt

_REACT_HEADER = """\
你正在以 ReAct 多轮模式回答用户问题。每一轮只会喂给你一部分检索到的知识，你必须：

1. 先在 <think>...</think> 里说明：本轮观察到了什么、是否已经能给出最终答案。
2. 在 <verdict>...</verdict> 里给出本轮判定，仅允许 ``complete`` 或 ``incomplete``：
   - 当本轮证据已足够回答问题时输出 ``complete``；
   - 否则输出 ``incomplete``，并在 <think> 中描述"目前缺什么、希望下一轮看到什么"。
3. 仅当 verdict==complete 时，才在 <answer>...</answer> 里给出最终答案，
   并严格遵循下文 CORPUS 系统提示词中关于答案结构、术语、引用、风险提示的全部规定。
4. 当本轮被外部强制为最终轮（``round_status=final``）时，必须按 ``complete``
   产出完整 <answer>，不允许再返回 incomplete。

【绝对约束】
- 输出顺序固定：<think>...</think><verdict>...</verdict>[<answer>...</answer>]。
- <verdict> 标签内除了 ``complete`` / ``incomplete`` 不得有任何其他字符。
- 中间轮（``incomplete``）禁止输出 <answer> 段。

——以下是问答样本规范（沿用 CORPUS 提示词，严格遵守）——
"""


REACT_SYSTEM_PROMPT = _REACT_HEADER + "\n" + CORPUS_SYSTEM_PROMPT


REACT_USER_PROMPT = """## 【用户问题】：
{question}

## 【前置预答】（来自 preview 阶段，可能未结束，仅供参考）
{preview_block}

## 【本轮可用的事实依据】（来自 skills 阶段）
{skill_context_block}

## 【历史轮次思考摘要】（仅供参考，禁止直接复述）
{prev_think}

## 【本轮新增的检索证据】
{evidence}

## 【本轮状态】
{round_status}
"""


# -------------------------------------------------------------- helpers


def format_skill_block(skills: list[dict] | None) -> str:
    """把 redis 快照里的 skills 列表格式化为 prompt 用的事实依据块。

    入参形如 ``[{"name": "...", "success": true, "stdout": "...", "exitCode": 0}, ...]``。
    与 ``skills.registry.SkillResultRegistry.format_context`` 输出风格保持一致。
    """

    skills = skills or []
    if not skills:
        return "（暂无事实依据）"
    lines: list[str] = []
    for i, rec in enumerate(skills, 1):
        lines.append(f"【依据{i}】")
        if rec.get("success"):
            stdout = (rec.get("stdout") or "").strip()
            lines.append(stdout or "（该依据未返回有效内容）")
        else:
            exit_code = rec.get("exitCode")
            lines.append(f"该依据获取失败（exit={exit_code}）。")
            stderr = (rec.get("stderr") or "").strip()
            if stderr:
                lines.append(stderr)
        lines.append("")
    return "\n".join(lines).rstrip()


def format_preview_block(preview: dict | None) -> str:
    """把 redis 快照里的 preview 子结构格式化为 prompt 用的预答块。"""

    preview = preview or {}
    think = (preview.get("think") or "").strip()
    answer = (preview.get("answer") or "").strip()
    done = bool(preview.get("done"))

    if not think and not answer:
        return "（preview 暂无产出）"
    parts: list[str] = []
    if think:
        parts.append(f"<preview_think>\n{think}\n</preview_think>")
    if answer:
        parts.append(f"<preview_answer>\n{answer}\n</preview_answer>")
    parts.append(f"<preview_done>{'true' if done else 'false'}</preview_done>")
    return "\n".join(parts)


def format_react_user_prompt(
    *,
    question: str,
    evidence: str,
    prev_think: str,
    preview: dict | None,
    skills: list[dict] | None,
    is_last_chunk: bool,
) -> str:
    """组装 ReAct 单轮 user prompt。"""

    round_status = "final" if is_last_chunk else "intermediate"
    prev_text = (prev_think or "").strip() or "（首轮，无历史思考）"
    return REACT_USER_PROMPT.format(
        question=question,
        evidence=evidence,
        prev_think=prev_text,
        preview_block=format_preview_block(preview),
        skill_context_block=format_skill_block(skills),
        round_status=round_status,
    )
