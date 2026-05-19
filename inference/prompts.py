"""inference 阶段使用的系统/用户 prompt。

设计原则：

- ``CORPUS_SYSTEM_PROMPT`` / ``CORPUS_USER_PROMPT`` 直接 import 自 ``reasoner/v3/prompts.py``，
  不复制不修改，避免两份口径漂移。
- **中间轮**（``incomplete`` 判定阶段）：用独立的 ``REACT_INTERMEDIATE_*`` prompt，
  不依赖 CORPUS_*，本轮只做两件事：
    1. ``<think>`` 内自反思本轮收集到了什么、还缺什么 —— 作为 ``prev_think`` 传给下一轮；
    2. ``<answer>`` 内仅输出 ``complete`` 或 ``incomplete`` —— **这一项就是本轮的 verdict**,
       由 :mod:`inference.react_loop` 据此决定是否再开一轮。
  历史 ``<verdict>`` 标签已弃用；不再依赖 ``StreamTagRouter.verdict`` 解析中间轮裁决。
- **最终轮**（``complete`` 收尾或被强制最终轮）：直接用 **纯**
  ``CORPUS_SYSTEM_PROMPT``，不再叠加任何 ReAct 指令头；
  user prompt 走 ``CORPUS_USER_PROMPT`` 模板，把 preview / 历史轮次思考 /
  本轮新增检索证据都按 markdown 三级标题拼到 ``evidence`` 变量内。
  最终轮 ``<answer>`` 内容才是真正面向用户的答案。
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
    "PREVIEW_SYSTEM_PROMPT_WITH_TGK",
    "PREVIEW_USER_PROMPT_WITH_TGK",
    "REACT_INTERMEDIATE_SYSTEM_PROMPT",
    "REACT_INTERMEDIATE_USER_PROMPT",
    "REACT_FINAL_SYSTEM_PROMPT",
    "select_preview_prompt",
    "format_preview_user_prompt",
    "format_react_intermediate_user_prompt",
    "format_react_final_user_prompt",
    "select_react_prompt",
    "format_skill_block",
    "format_preview_block",
    "format_evidence_for_final",
]


# -------------------------------------------------------------- preview prompt
#
# 双套 prompt 并存，按 ``topic_general_knowledge`` 是否提供路由（见
# :func:`select_preview_prompt`）：
#
# - 老（``PREVIEW_SYSTEM_PROMPT`` / ``PREVIEW_USER_PROMPT``）：
#   仅基于模型自身常识做轻量分析，user prompt 只有【用户问题】一段。
#   入参缺少 ``topic_general_knowledge`` 时使用，向后兼容原有调用。
# - 新（``PREVIEW_SYSTEM_PROMPT_WITH_TGK`` / ``PREVIEW_USER_PROMPT_WITH_TGK``）：
#   要求模型"遵循专题通用知识 + 自身常识"，user prompt 多一段【专题通用知识】。
#   对接外层 ``ReasonRequest.answerSystemPrompt``，让客户传入的专题背景知识
#   参与 preview 预答。两个常量必须成对启用，避免出现 system 让模型遵循
#   一段不存在的小节这种矛盾。

PREVIEW_SYSTEM_PROMPT = """\
你是一个资深财税实务咨询专家。

请基于自身常识，对用户问题做一次轻量分析
- <think> 标签内容：从财税实务角度对问题做拆解，列出涉及的知识体系与回答逻辑。（500字以内）
- <answer> 标签内容：给出还需要进一步验证的关键点（100字以内）

【绝对约束】
- 仅输出一段 <think>...</think> 和一段 <answer>...</answer>，不要任何其他文字。
- 不确定的思路和常识，允许标注"待结合检索后修正"
- 不用考虑可能的风险点
- 严禁输出具体政策名称、内容及其相关要求（你的信息是过时的）
- 严禁给出问题的答案，只要输出解答思路
"""

PREVIEW_USER_PROMPT = """## 【用户问题】
{question}
"""


PREVIEW_SYSTEM_PROMPT_WITH_TGK = """\
你是一个资深财税实务咨询专家。

请遵循**专题通用知识**，并基于自身常识，对**用户问题**做一次轻量分析
- <think> 标签内容：从财税实务角度对问题做拆解，列出涉及的知识体系与回答逻辑。（500字以内）
- <answer> 标签内容：列出与问题相关的专题通用知识、以及还需要进一步验证的关键点。（200字以内）

【绝对约束】
- 仅输出一段 <think>...</think> 和一段 <answer>...</answer>，不要任何其他文字。
- 不确定的思路和常识，允许标注"待结合检索后修正"
- 不用考虑可能的风险点
- 严禁输出具体政策名称、内容及其相关要求（你的自身常识是过时的）
- 严禁给出问题的答案，只要输出解答思路
- 专题通用知识是绝对真理，可以放心输出
"""

PREVIEW_USER_PROMPT_WITH_TGK = """## 【用户问题】
{question}

## 【专题通用知识】
{topic_general_knowledge}
"""


def select_preview_prompt(
    *,
    question: str,
    topic_general_knowledge: str | None = None,
) -> tuple[str, str]:
    """根据 ``topic_general_knowledge`` 是否提供，路由到老/新两套 preview prompt。

    - ``topic_general_knowledge`` 非空（``str.strip()`` 后仍有内容）：
      返回 ``(PREVIEW_SYSTEM_PROMPT_WITH_TGK, 渲染后的 PREVIEW_USER_PROMPT_WITH_TGK)``,
      user prompt 注入【专题通用知识】小节，system 同步切到要求"遵循专题通用知识"的版本。
    - 否则：返回 ``(PREVIEW_SYSTEM_PROMPT, 渲染后的 PREVIEW_USER_PROMPT)``,
      与改造前的老版 preview 完全一致（只有【用户问题】、system 要求"基于自身常识"）。

    返回 ``(system_prompt, user_prompt)`` tuple，结构对齐 :func:`select_react_prompt`。
    """

    tgk = (topic_general_knowledge or "").strip()
    if tgk:
        return (
            PREVIEW_SYSTEM_PROMPT_WITH_TGK,
            PREVIEW_USER_PROMPT_WITH_TGK.format(
                question=question,
                topic_general_knowledge=tgk,
            ),
        )
    return (
        PREVIEW_SYSTEM_PROMPT,
        PREVIEW_USER_PROMPT.format(question=question),
    )


def format_preview_user_prompt(
    *,
    question: str,
    topic_general_knowledge: str | None = None,
) -> str:
    """组装 preview 阶段 user prompt（仅返回 user 段；如需 system 请用 :func:`select_preview_prompt`）。

    内部直接复用 :func:`select_preview_prompt` 的路由，行为完全等价于取其 ``[1]``。
    """

    _, user_prompt = select_preview_prompt(
        question=question,
        topic_general_knowledge=topic_general_knowledge,
    )
    return user_prompt


# -------------------------------------------------------------- react: 中间轮
#
# 中间轮目标只有一个：判断当前已掌握的信息能否完美回答问题。

REACT_INTERMEDIATE_SYSTEM_PROMPT = """\
你是一个资深财税实务专家。你的目标是判断当前掌握的信息是否已经足以**完美回答**用户问题，

请严格按以下结构输出：

1. <think>...</think>：复盘本轮观察到了什么、与问题的关联，是否已经覆盖
   核心结论 / 关键依据 / 必要的操作或风险提示；若仍不充分，明确说明
   "还缺什么、希望下一轮看到什么类型的知识"。
2. <answer>...</answer>：仅允许 ``complete`` 或 ``incomplete``：
   - ``complete``：已经能完美回答用户问题（覆盖核心结论、直接证据、依据、必要的操作或风险提示等）。
   - ``incomplete``：仍需更多检索证据，请继续。

【绝对约束】
- 禁止类比推理，必须有信息明文给出直接证据
- 输出顺序固定：<think>...</think><answer>...</answer>。
- <answer> 标签内只能是 ``complete`` 或 ``incomplete``，不得有空白或其他字符。
"""


REACT_INTERMEDIATE_USER_PROMPT = """## 【用户问题】
{question}

## 【参考依据】
{skill_context_block}

## 【前置预答】（可能存在过期信息，若采纳政策条款信息，则必须在摘要或证据中得到印证）
{preview_block}

## 【历史轮次思考摘要】
{prev_think}

## 【本轮新增的检索证据】
{evidence}
"""


# -------------------------------------------------------------- react: 最终轮
#
# 最终轮直接复用 v3 的 CORPUS_SYSTEM_PROMPT / CORPUS_USER_PROMPT，
# 不再叠加任何 ReAct 指令头。preview / 历史思考 / 本轮新增检索证据全部通过
# :func:`format_evidence_for_final` 拼到 CORPUS_USER_PROMPT 的 evidence 槽位里，
# 内部用 ### 三级标题，避免与 ## 【已知信息】抢标题层级。

REACT_FINAL_SYSTEM_PROMPT = CORPUS_SYSTEM_PROMPT


# -------------------------------------------------------------- helpers


def format_skill_block(skills: list[dict] | None) -> str:
    """把 redis 快照里的 skills 列表格式化为 prompt 用的参考依据块。

    入参形如 ``[{"name": "...", "success": true, "stdout": "...", "exitCode": 0}, ...]``。
    与 ``skills.registry.SkillResultRegistry.format_context`` 输出风格保持一致。
    """

    skills = skills or []
    if not skills:
        return "（暂无参考依据）"
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

    注意：``CORPUS_USER_PROMPT`` 内 evidence 是接在 ``## 【已知信息】：`` 标题下方，
    所以本函数内部一律用 ``### `` 三级标题，避免抢层级。
    """

    sections: list[str] = []
    
    # 2026.05.12：在此处将前置预答和历史轮次摘要做了位置互换
    preview_md = _format_preview_block_markdown(preview)
    if preview_md:
        sections.append("### 前置预答（可能存在过期信息，若采纳政策条款信息，则必须在摘要或证据中得到印证）\n" + preview_md)

    pt = (prev_think or "").strip()
    if pt:
        sections.append("### 历史轮次推理摘要\n" + pt)

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
    """组装中间轮 user prompt。

    模型应输出 ``<think>...</think><answer>complete|incomplete</answer>``：
    ``<think>`` 是下一轮的 ``prev_think``，``<answer>`` 内容就是本轮 verdict。
    """

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
    保持一致：``## 参考依据\\n<body>\\n\\n``，空依据时整段省略。
    """

    skill_text = format_skill_block(skills)
    if skill_text and skill_text != "（暂无参考依据）":
        skill_block = "## 【参考依据】\n" + skill_text + "\n\n"
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
