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
- **最终轮虚拟 skill "当前真实日期"**：每次组装最终轮 user prompt 时,
  :func:`format_react_final_user_prompt` 会调用 :func:`_make_current_date_skill_record`
  在 skills 列表尾部追加一条系统时间记录。该记录结构与 ``inference.skills_runner._serialize_registry``
  输出完全一致（``name/command/success/stdout/stderr/exitCode`` 六个字段），
  让它能复用同一个 :func:`format_skill_block`，最终在 prompt 内的呈现与其他真实 skill
  ``【依据N】`` 段无差别。该记录**不写回 redis 快照**：避免污染
  ``_v4_skills_result_from_snapshot`` 给客户端返回的 ``skillsResult``、也不会被中间轮
  prompt 重复看到（中间轮仍只看真实 skill 列表）。
"""

from __future__ import annotations

import datetime as _dt

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
    "PREVIEW_SYSTEM_PROMPT_WITH_CASES",
    "PREVIEW_USER_PROMPT_WITH_CASES",
    "PREVIEW_SYSTEM_PROMPT_WITH_TGK_AND_CASES",
    "PREVIEW_USER_PROMPT_WITH_TGK_AND_CASES",
    "REACT_INTERMEDIATE_SYSTEM_PROMPT",
    "REACT_INTERMEDIATE_USER_PROMPT",
    "REACT_FINAL_SYSTEM_PROMPT",
    "select_preview_prompt",
    "format_preview_user_prompt",
    "format_related_cases_block",
    "format_react_intermediate_user_prompt",
    "format_react_final_user_prompt",
    "select_react_prompt",
    "format_skill_block",
    "format_preview_block",
    "format_evidence_for_final",
]


# -------------------------------------------------------------- preview prompt
#
# 四套 prompt 并存，按 ``(topic_general_knowledge, related_cases)`` 两维路由
# （见 :func:`select_preview_prompt`）：
#
# - 原 ``PREVIEW_SYSTEM_PROMPT`` / ``PREVIEW_USER_PROMPT``：
#   仅基于模型自身常识做轻量分析，user prompt 只有【用户问题】一段。
#   ``topic_general_knowledge`` 与 ``related_cases`` 都为空时使用，与改造前完全等价。
# - ``PREVIEW_SYSTEM_PROMPT_WITH_TGK`` / ``PREVIEW_USER_PROMPT_WITH_TGK``：
#   要求模型"遵循专题通用知识 + 自身常识"，user prompt 多一段【专题通用知识】。
#   对接外层 ``ReasonRequest.answerSystemPrompt``，让客户传入的专题背景知识
#   参与 preview 预答。
# - ``PREVIEW_SYSTEM_PROMPT_WITH_CASES`` / ``PREVIEW_USER_PROMPT_WITH_CASES``：
#   要求模型"参考历史经验 + 自身常识"，user prompt 多一段【历史经验】。
#   案例来源见 :mod:`inference.retrieval.case_search`（LanceDB case 库纯向量检索,
#   按 cosine_similarity 阈值过滤后取 top-k）。
# - ``PREVIEW_SYSTEM_PROMPT_WITH_TGK_AND_CASES`` / ``PREVIEW_USER_PROMPT_WITH_TGK_AND_CASES``：
#   两者并存：system 要求"遵循专题通用知识 + 参考历史经验 + 自身常识"，
#   user prompt 同时包含【专题通用知识】与【历史经验】两个小节。
#
# 关键语义：``related_cases`` 为空（含"topC=0 关闭检索"和"召回 0 条达阈值"两种情况）时,
# 一律回退到不带【历史经验】段的版本，**绝不**渲染"暂无相关案例"占位串，
# 避免空段落给模型带来无意义干扰。

PREVIEW_SYSTEM_PROMPT = """\
你是一个资深财税实务咨询专家。

请基于自身常识，对用户问题做一次轻量分析
- <think> 标签内容：从财税实务角度对问题做拆解，列出涉及的知识体系与回答逻辑。（500字以内）
- <answer> 标签内容：给出还需要进一步验证的关键点（200字以内）

【绝对约束】
- 不确定的思路和常识，允许标注"待结合检索后修正"
- 不用考虑可能的风险点
- 严禁输出具体政策名称、内容及其相关要求（你的信息是过时的）
- 严禁给出问题的答案，只要输出解答思路
- 必须输出一段 <think>...</think> 和一段 <answer>...</answer>，不要任何其他标签内容。
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
- 不确定的思路和常识，允许标注"待结合检索后修正"
- 不用考虑可能的风险点
- 严禁输出具体政策名称、内容及其相关要求（你的自身常识是过时的）
- 严禁给出问题的答案，只要输出解答思路
- 专题通用知识是绝对真理，可以放心输出
- 必须输出一段 <think>...</think> 和一段 <answer>...</answer>，不要任何其他标签内容。
"""

PREVIEW_USER_PROMPT_WITH_TGK = """## 【专题通用知识】
{topic_general_knowledge}

## 【用户问题】
{question}
"""


PREVIEW_SYSTEM_PROMPT_WITH_CASES = """\
你是一个资深财税实务咨询专家。

请参考**历史经验**，并基于自身常识，对**用户问题**做一次轻量分析
- <think> 标签内容：从财税实务角度对问题做拆解，列出涉及的知识体系与回答逻辑。（500字以内）
- <answer> 标签内容：列出与问题相关的历史经验、以及还需要进一步验证的关键点。（200字以内）

【绝对约束】
- 不确定的思路和常识，允许标注"待结合检索后修正"
- 不用考虑可能的风险点
- 严禁输出具体政策名称、内容及其相关要求（你的信息是过时的）
- 严禁给出问题的答案，只要输出解答思路
- 如何使用历史经验：
    - 先严谨对比用户问题和案例问题的业务实质差异，若不相关则直接忽略；
    - 若相关，则识别出与**用户问题密切相关**的'关键判定逻辑'或'易错点'。
- 必须输出一段 <think>...</think> 和一段 <answer>...</answer>，不要任何其他标签内容。
"""

PREVIEW_USER_PROMPT_WITH_CASES = """## 【用户问题】
{question}

## 【历史经验】
{related_cases_block}
"""


PREVIEW_SYSTEM_PROMPT_WITH_TGK_AND_CASES = """\
你是一个资深财税实务咨询专家。

请遵循**专题通用知识**，参考**历史经验**，并基于自身常识，对**用户问题**做一次轻量分析
- <think> 标签内容：从财税实务角度对问题做拆解，列出涉及的知识体系与回答逻辑。（500字以内）
- <answer> 标签内容：列出与问题相关的专题通用知识、历史经验、以及还需要进一步验证的关键点。（200字以内）

【绝对约束】
- 不确定的思路和常识，允许标注"待结合检索后修正"
- 不用考虑可能的风险点
- 严禁输出具体政策名称、内容及其相关要求（你的自身常识是过时的）
- 严禁给出问题的答案，只要输出解答思路
- 专题通用知识是绝对真理，可以放心输出
- 如何使用历史经验：
    - 先严谨对比用户问题和案例问题的业务实质差异，若不相关则直接忽略；
    - 若相关，则识别出与**用户问题密切相关**的'关键判定逻辑'或'易错点'。
- 必须输出一段 <think>...</think> 和一段 <answer>...</answer>，不要任何其他标签内容。
"""

PREVIEW_USER_PROMPT_WITH_TGK_AND_CASES = """## 【专题通用知识】
{topic_general_knowledge}

## 【用户问题】
{question}

## 【历史经验】
{related_cases_block}
"""


def format_related_cases_block(related_cases: list | None) -> str:
    """把 ``list[CaseHit]`` 渲染为 preview user prompt 的【历史经验】段正文。

    入参为 :class:`inference.retrieval.case_search.CaseHit` 列表（也兼容 dict
    形态，方便单测 / 离线脚本直接喂构造好的 dict）。返回 markdown 文本块,
    用三级标题 ``### 案例N`` 列出每条。

    入参为空（None / 空 list）时返回空串——**select_preview_prompt 据此回退
    到不带【历史经验】段的 prompt**，不会在 prompt 里渲染空段。
    """

    cases = list(related_cases or [])
    if not cases:
        return ""

    lines: list[str] = []
    for i, case in enumerate(cases, 1):
        if isinstance(case, dict):
            question = (case.get("question") or "").strip()
            knowledge = (case.get("knowledge") or "").strip()
        else:
            question = (getattr(case, "question", "") or "").strip()
            knowledge = (getattr(case, "knowledge", "") or "").strip()

        lines.append(f"### 案例{i}")
        if question:
            lines.append(f"**案例问题**：{question}")
        if knowledge:
            lines.append(f"**案例知识**：{knowledge}")
        lines.append("")
    return "\n".join(lines).rstrip()


def select_preview_prompt(
    *,
    question: str,
    topic_general_knowledge: str | None = None,
    related_cases: list | None = None,
) -> tuple[str, str]:
    """按 ``(topic_general_knowledge, related_cases)`` 两维路由到 4 套 preview prompt 之一。

    路由表：

    +----------+----------+--------------------------------------+
    | tgk 非空 | cases 非空 | 路由到                              |
    +==========+==========+======================================+
    | 否       | 否       | ``PREVIEW_*``                        |
    | 是       | 否       | ``PREVIEW_*_WITH_TGK``               |
    | 否       | 是       | ``PREVIEW_*_WITH_CASES``             |
    | 是       | 是       | ``PREVIEW_*_WITH_TGK_AND_CASES``     |
    +----------+----------+--------------------------------------+

    ``related_cases`` 为空（含 ``topC=0`` 关闭检索 / 召回 0 条达阈值两种情况）时
    一律回退到不带【历史经验】段的版本——这保证了"关闭 case 模式"和
    "开启但无命中"对模型的可见 prompt 完全一致，避免空段落干扰预答质量。

    返回 ``(system_prompt, user_prompt)`` tuple，结构对齐 :func:`select_react_prompt`。
    """

    tgk = (topic_general_knowledge or "").strip()
    cases_block = format_related_cases_block(related_cases)
    has_cases = bool(cases_block)

    if tgk and has_cases:
        return (
            PREVIEW_SYSTEM_PROMPT_WITH_TGK_AND_CASES,
            PREVIEW_USER_PROMPT_WITH_TGK_AND_CASES.format(
                question=question,
                topic_general_knowledge=tgk,
                related_cases_block=cases_block,
            ),
        )
    if has_cases:
        return (
            PREVIEW_SYSTEM_PROMPT_WITH_CASES,
            PREVIEW_USER_PROMPT_WITH_CASES.format(
                question=question,
                related_cases_block=cases_block,
            ),
        )
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
    related_cases: list | None = None,
) -> str:
    """组装 preview 阶段 user prompt（仅返回 user 段；如需 system 请用 :func:`select_preview_prompt`）。

    内部直接复用 :func:`select_preview_prompt` 的路由，行为完全等价于取其 ``[1]``。
    """

    _, user_prompt = select_preview_prompt(
        question=question,
        topic_general_knowledge=topic_general_knowledge,
        related_cases=related_cases,
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


_WEEKDAY_CN = ("一", "二", "三", "四", "五", "六", "日")


def _make_current_date_skill_record() -> dict:
    """构造一条"当前真实日期"虚拟 skill 记录，结构对齐 ``inference.skills_runner._serialize_registry``。

    每次调用都重新读取系统本地时间（``datetime.now()``），保证最终轮 prompt 拿到的是
    本次请求处理时的当下日期，而不是进程启动时的快照。返回 dict 字段集合与真实 skill
    完全一致（``name/command/success/stdout/stderr/exitCode``），这样后续 :func:`format_skill_block`
    把它当成普通 skill 渲染时，呈现到 prompt 内的格式与其他 ``【依据N】`` 段无差别。

    刻意把日期写在 stdout 内部的自然语言里（而非仅靠 ``name`` 字段），是因为
    ``format_skill_block`` 不会输出 skill 的 ``name``——只输出 ``【依据N】`` + ``stdout``,
    所以语义信息必须自带在 stdout 文本里，模型才能识别这是日期。
    """

    now = _dt.datetime.now()
    weekday_cn = _WEEKDAY_CN[now.weekday()]
    stdout = (
        f"当前真实日期为 {now.year}年{now.month}月{now.day}日（星期{weekday_cn}）。"
        "请基于该日期判断与时间相关的政策时效、申报周期或纳税期限。"
    )
    return {
        "name": "当前真实日期",
        "command": "",
        "success": True,
        "stdout": stdout,
        "stderr": "",
        "exitCode": 0,
    }


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

    最终轮 prompt 一律在 skills 尾部追加一条"当前真实日期"虚拟记录（结构与真实 skill
    完全一致，详见 :func:`_make_current_date_skill_record`），让模型在写最终答案时
    始终能感知请求处理当下的真实日期。该记录只在本函数局部追加、**不**回写 redis 快照,
    因此不会影响 ``skillsResult`` 字段、也不会被中间轮 prompt 看到。注入后 skills 列表
    永远非空，原本"空依据时整段省略"的分支实际不会再触达，但保留分支结构以兼容
    ``format_skill_block`` 返回占位串的极端路径。
    """

    skills_with_date = list(skills or [])
    skills_with_date.append(_make_current_date_skill_record())

    skill_text = format_skill_block(skills_with_date)
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
