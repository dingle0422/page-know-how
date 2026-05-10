"""inference 阶段使用的系统/用户 prompt。

设计原则：

- ``CORPUS_SYSTEM_PROMPT`` / ``CORPUS_USER_PROMPT`` 直接 import 自 ``reasoner/v3/prompts.py``，
  不复制不修改，避免两份口径漂移。
- **中间轮**（``incomplete`` 判定阶段）：用独立的 ``REACT_INTERMEDIATE_*`` prompt，
  不依赖 CORPUS_*，本轮只判定 "是否能完美回答" + 自反思，**禁止** 产出 <answer>。
- **最终轮**（``complete`` 收尾或被强制最终轮）：直接用 **纯**
  ``CORPUS_SYSTEM_PROMPT``，不再叠加任何 ReAct 指令头；
  user prompt 走 ``CORPUS_USER_PROMPT`` 模板，把 preview / 历史轮次思考 /
  本轮新增检索证据都按 markdown 三级标题拼到 ``evidence`` 变量内。
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
    "REACT_INTERMEDIATE_SYSTEM_PROMPT",
    "REACT_INTERMEDIATE_USER_PROMPT",
    "REACT_FINAL_SYSTEM_PROMPT",
    "format_react_intermediate_user_prompt",
    "format_react_final_user_prompt",
    "select_react_prompt",
    "format_skill_block",
    "format_preview_block",
    "format_evidence_for_final",
]


# -------------------------------------------------------------- preview prompt

PREVIEW_SYSTEM_PROMPT = """\
你是一个资深财税实务咨询专家。

请基于自身常识，先做一次轻量分析：
- <think> 标签内容：从财税实务角度对问题做拆解，列出涉及的知识体系与回答逻辑。
- <answer> 标签内容：给出一个保守、范围明确的预答，允许标注"待结合检索后修正"。

【绝对约束】
- 仅输出一段 <think>...</think> 和一段 <answer>...</answer>，不要任何其他文字。
- <answer> 不要编造具体政策文号或法规细节，并且注意政策法规的时效性，遇到不确定处直接说"待结合检索"。
"""

PREVIEW_USER_PROMPT = """## 用户问题
{question}
"""


# -------------------------------------------------------------- react: 中间轮
#
# 中间轮目标只有一个：判断当前已掌握的信息能否完美回答问题。
# **禁止输出 <answer>**，最终答案由"最终轮"用纯 CORPUS prompt 重新生成。

REACT_INTERMEDIATE_SYSTEM_PROMPT = """\
你是一个资深财税实务专家。你的目标是判断当前掌握的信息是否已经足以**完美回答**用户问题，

请严格按以下结构输出：

1. <think>...</think>：复盘本轮观察到了什么、与问题的关联，是否已经覆盖
   核心结论 / 关键依据 / 必要的操作或风险提示；若仍不充分，明确说明
   "还缺什么、希望下一轮看到什么类型的知识"。
2. <verdict>...</verdict>：仅允许 ``complete`` 或 ``incomplete``：
   - ``complete``：已经能完美回答用户问题（覆盖核心结论、依据、必要的操作或风险提示等）。
   - ``incomplete``：仍需更多检索证据，请继续。

【绝对约束】
- 输出顺序固定：<think>...</think><verdict>...</verdict>。
- **禁止输出 <answer> 段**，最终答案不在本轮产生。
- <verdict> 标签内只能是 ``complete`` 或 ``incomplete``，不得有空白或其他字符。
"""


REACT_INTERMEDIATE_USER_PROMPT = """## 用户问题
{question}

## 历史轮次思考摘要
{prev_think}

## 本轮新增的检索证据
{evidence}

## 前置预答（可能存在过期信息，仅供参考，请注意政策法规的时效性）
{preview_block}

## 参考依据
{skill_context_block}
"""


# -------------------------------------------------------------- react: 最终轮
#
# 最终轮直接复用 v3 的 CORPUS_SYSTEM_PROMPT / CORPUS_USER_PROMPT，
# 不再叠加任何 ReAct 指令头。preview / 历史思考 / 本轮新增检索证据全部通过
# :func:`format_evidence_for_final` 拼到 CORPUS_USER_PROMPT 的 evidence 槽位里，
# 内部用 ### 三级标题，避免与 ## 【内化原生知识】抢标题层级。

REACT_FINAL_SYSTEM_PROMPT = CORPUS_SYSTEM_PROMPT


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
    """把 redis 快照里的 preview 子结构格式化为中间轮 prompt 用的预答块（带标签）。"""

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


def _format_preview_block_markdown(preview: dict | None) -> str:
    """最终轮专用：把 preview 子结构渲染为纯 markdown（无标签），便于嵌入 evidence。"""

    preview = preview or {}
    think = (preview.get("think") or "").strip()
    answer = (preview.get("answer") or "").strip()
    if not think and not answer:
        return ""
    parts: list[str] = []
    if think:
        parts.append("**preview think**\n\n" + think)
    if answer:
        parts.append("**preview answer**\n\n" + answer)
    return "\n\n".join(parts)


def format_evidence_for_final(
    *,
    prev_think: str,
    preview: dict | None,
    new_evidence: str,
) -> str:
    """组装最终轮 ``CORPUS_USER_PROMPT.evidence`` 的内容主体。

    注意：``CORPUS_USER_PROMPT`` 内 evidence 是接在 ``## 【内化原生知识】：`` 标题下方，
    所以本函数内部一律用 ``### `` 三级标题，避免抢层级。
    """

    sections: list[str] = []

    pt = (prev_think or "").strip()
    if pt:
        sections.append("### 历史轮次推理摘要\n" + pt)

    preview_md = _format_preview_block_markdown(preview)
    if preview_md:
        sections.append("### 前置预答（来自 preview 阶段，可能存在过期信息，仅供参考）\n" + preview_md)

    ne = (new_evidence or "").strip()
    if ne:
        sections.append("### 本轮检索证据\n" + ne)
    else:
        sections.append("### 本轮检索证据\n（本次未检索到任何证据，请基于通识做最终回答。）")

    return "\n\n".join(sections)


def format_react_intermediate_user_prompt(
    *,
    question: str,
    evidence: str,
    prev_think: str,
    preview: dict | None,
    skills: list[dict] | None,
) -> str:
    """组装中间轮 user prompt（自反思 + verdict 判定，禁止 <answer>）。"""

    prev_text = (prev_think or "").strip() or "（首轮，无历史思考）"
    return REACT_INTERMEDIATE_USER_PROMPT.format(
        question=question,
        evidence=(evidence or "").strip() or "（本轮未提供新增检索证据）",
        prev_think=prev_text,
        preview_block=format_preview_block(preview),
        skill_context_block=format_skill_block(skills),
    )


def format_react_final_user_prompt(
    *,
    question: str,
    evidence: str,
    prev_think: str,
    preview: dict | None,
    skills: list[dict] | None,
) -> str:
    """组装最终轮 user prompt（纯 ``CORPUS_USER_PROMPT``，evidence 内塞所有上下文）。

    skills 渲染口径与 ``reasoner/v3/agent_graph._build_skill_context_for_summary``
    保持一致：``## 事实依据\\n<body>\\n\\n``，空依据时整段省略。
    """

    skill_text = format_skill_block(skills)
    if skill_text and skill_text != "（暂无事实依据）":
        skill_block = "## 事实依据\n" + skill_text + "\n\n"
    else:
        skill_block = ""

    merged_evidence = format_evidence_for_final(
        prev_think=prev_think,
        preview=preview,
        new_evidence=evidence,
    )
    return CORPUS_USER_PROMPT.format(
        question=question,
        skill_context_block=skill_block,
        evidence=merged_evidence,
    )


def select_react_prompt(
    *,
    is_last_chunk: bool,
    question: str,
    evidence: str,
    prev_think: str,
    preview: dict | None,
    skills: list[dict] | None,
) -> tuple[str, str]:
    """按是否为最终轮，返回 ``(system_prompt, user_prompt)``。

    - 中间轮：``REACT_INTERMEDIATE_SYSTEM_PROMPT`` + ``REACT_INTERMEDIATE_USER_PROMPT``。
    - 最终轮：纯 ``CORPUS_SYSTEM_PROMPT`` + ``CORPUS_USER_PROMPT``（evidence 内塞所有上下文）。
    """

    if is_last_chunk:
        return REACT_FINAL_SYSTEM_PROMPT, format_react_final_user_prompt(
            question=question,
            evidence=evidence,
            prev_think=prev_think,
            preview=preview,
            skills=skills,
        )
    return REACT_INTERMEDIATE_SYSTEM_PROMPT, format_react_intermediate_user_prompt(
        question=question,
        evidence=evidence,
        prev_think=prev_think,
        preview=preview,
        skills=skills,
    )
