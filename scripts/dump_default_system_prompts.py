"""一次性脚本：把 v2 默认参数下 chunk_reason / batch_summary / streaming_batch_summary /
batch_final_merge 这四个阶段实际下发给 LLM 的 system prompt 拼接结果导出到 markdown，
便于核对（拼接逻辑由 agent_graph.py 中各阶段决定，本脚本与其保持一致）。

默认参数前提以 `app.py` 中 Pydantic 的 Field 默认值为准（这是 web 服务实际生效的默认；
请求体经 `req.model_dump()` 走一遍 Pydantic 后字段都会被填进去，因此后端
`request_payload.get(key, fallback)` 的 `fallback` 在这条调用路径上几乎不会被命中）：

    version              = "v2"
    retrievalMode        = True
    checkPitfalls        = True   ⚠ 影响 chunk_reason
    chunkSize            = 3000
    summaryBatchSize     = 3
    enableSkills         = True
    summaryCleanAnswer   = True   ⚠ 影响 batch_final_merge
    thinkMode            = True   ⚠ 影响 batch_final_merge
    lastThink            = True
    enableRelations      = True
    summaryPipelineMode  = "layered"
    pureModelResult      = True   ⚠ 影响 batch_summary / streaming_batch_summary / batch_final_merge
    answerSystemPrompt   = None
"""

from __future__ import annotations

import os

from reasoner.v2.prompts import (
    BATCH_MERGE_AND_CLEAN_THINK_SYSTEM_PROMPT,
    BATCH_REDUCE_SYSTEM_PROMPT,
    BATCH_SUMMARY_SYSTEM_PROMPT,
    CHUNK_REASONING_SYSTEM_PROMPT,
    CHUNK_REASONING_WITH_PITFALLS_SYSTEM_PROMPT,
    PURE_MODEL_REFERENCE_ANSWER_INSTRUCTIONS,
    PURE_MODEL_REFERENCE_EXTRACT_INSTRUCTIONS,
    SUMMARY_ANSWER_SYSTEM_PROMPT,
    SUMMARY_EXTRACT_SYSTEM_PROMPT,
)


DEFAULTS_TABLE = """| Pydantic 字段 | app.py 默认 | 对四阶段 system prompt 的影响 |
| --- | --- | --- |
| `version` | "v2" | 走 `reasoner/v2/agent_graph.py` 这条链路 |
| `retrievalMode` | True | 子智能体只做相关性判定+召回知识；不直接影响这四阶段的 system 拼接 |
| `checkPitfalls` | **True** | chunk_reason 切到 `CHUNK_REASONING_WITH_PITFALLS_SYSTEM_PROMPT`（输出 schema 多一段 `pitfalls`） |
| `chunkSize` | 3000 | 启用 chunk 模式 → 走 `_reason_on_chunk` + `_streaming_submit_batch` |
| `summaryBatchSize` | 3 | 启用分批中间提炼 → 走 `_recursive_batch_reduce`/`_streaming_submit_batch` |
| `enableSkills` | True | skill 评估阶段独立 LLM 调用，不影响这四阶段 system 拼接 |
| `summaryCleanAnswer` | **True** | batch_final_merge 启用合并+清洗一体化（须配合 `thinkMode`） |
| `thinkMode` | **True** | 与 `summaryCleanAnswer=True` 联动后，batch_final_merge 在 `SUMMARY_ANSWER_SYSTEM_PROMPT` 末尾追加 `BATCH_MERGE_AND_CLEAN_THINK_SYSTEM_PROMPT` |
| `lastThink` | True | 仅切换底层 chat `enable_thinking=True`，不改 system prompt 文本 |
| `enableRelations` | True | 关联条款展开走 `_reason_on_chunk` 同一通路，system prompt 与 chunk_reason 一致 |
| `summaryPipelineMode` | "layered" | streaming_batch_summary 走 `_streaming_submit_batch`；batch_summary 走 `_recursive_batch_reduce` 内的 `_summarize_batch`；两者 system prompt 一致 |
| `pureModelResult` | **True** | 在【拿到 deepseek-v4-pro 的非空参考回答】时：batch_summary / streaming_batch_summary 把 `PURE_MODEL_REFERENCE_EXTRACT_INSTRUCTIONS` **插入到 `BATCH_REDUCE` 与 `BATCH_SUMMARY` 之间**（位于格式说明上方）；batch_final_merge 把 `PURE_MODEL_REFERENCE_ANSWER_INSTRUCTIONS` **插入到 `SUMMARY_ANSWER` 与 `## 输出格式约束 + BATCH_MERGE_AND_CLEAN_THINK` 之间**（位于 JSON schema 上方）。若超时/失败/空回复则该段缺失 |
| `answerSystemPrompt` | None | batch_final_merge 使用内置 `SUMMARY_ANSWER_SYSTEM_PROMPT`（再叠加上述追加项） |
"""


# chunk_reason 阶段：_reason_on_chunk 内部直接拼，不经过 _augment_system_for_extract，
# 因此即使 pureModelResult=True 也不会在此阶段出现 PURE_MODEL_* 指令
CHUNK_REASON_FINAL = (
    SUMMARY_EXTRACT_SYSTEM_PROMPT + "\n\n" + CHUNK_REASONING_WITH_PITFALLS_SYSTEM_PROMPT
)

# batch_summary / streaming_batch_summary 阶段：
#   _augment_system_for_extract(role_header=BATCH_REDUCE, format_body=BATCH_SUMMARY)
# pureModelResult=True 且参考回答非空时，PURE_MODEL_REFERENCE_EXTRACT_INSTRUCTIONS
# 被插到 role_header（角色 + 处理目标 + 红线）与 format_body（动作规范 + 输出格式说明）
# 之间，确保「外部参考回答处理策略」位于「格式要求」上方
BATCH_SUMMARY_FINAL = "\n\n".join([
    BATCH_REDUCE_SYSTEM_PROMPT,
    PURE_MODEL_REFERENCE_EXTRACT_INSTRUCTIONS,
    BATCH_SUMMARY_SYSTEM_PROMPT,
])

# batch_final_merge 阶段（summaryCleanAnswer=True + thinkMode=True）：
#   _augment_system_for_answer(
#       role_header=SUMMARY_ANSWER_SYSTEM_PROMPT,
#       format_body="## 输出格式约束\n" + BATCH_MERGE_AND_CLEAN_THINK_SYSTEM_PROMPT,
#   )
# pureModelResult=True 且参考回答非空时，PURE_MODEL_REFERENCE_ANSWER_INSTRUCTIONS 同样
# 被插到 role_header 与 format_body 之间——保证它压在「## 输出格式约束 + JSON schema」上方
BATCH_FINAL_MERGE_FORMAT_BODY = (
    "## 输出格式约束\n" + BATCH_MERGE_AND_CLEAN_THINK_SYSTEM_PROMPT
)
BATCH_FINAL_MERGE_FINAL = "\n\n".join([
    SUMMARY_ANSWER_SYSTEM_PROMPT,
    PURE_MODEL_REFERENCE_ANSWER_INSTRUCTIONS,
    BATCH_FINAL_MERGE_FORMAT_BODY,
])


STAGES = [
    {
        "name": "chunk_reason",
        "desc": "单个知识块的相关性判定 + 原文摘录 + 易错点识别（`_reason_on_chunk`）",
        "concat": (
            'SUMMARY_EXTRACT_SYSTEM_PROMPT + "\\n\\n" + '
            "CHUNK_REASONING_WITH_PITFALLS_SYSTEM_PROMPT"
        ),
        "note": (
            "`checkPitfalls=True` 触发；不经过 `_augment_system_for_extract`，"
            "因此即使 `pureModelResult=True` 也不会在此阶段追加 PURE_MODEL_* 指令。"
        ),
        "pieces": [
            ("SUMMARY_EXTRACT_SYSTEM_PROMPT", SUMMARY_EXTRACT_SYSTEM_PROMPT),
            (
                "CHUNK_REASONING_WITH_PITFALLS_SYSTEM_PROMPT",
                CHUNK_REASONING_WITH_PITFALLS_SYSTEM_PROMPT,
            ),
        ],
        "final": CHUNK_REASON_FINAL,
    },
    {
        "name": "batch_summary",
        "desc": "分批中间提炼·layered 同步分层（`_recursive_batch_reduce` → `_summarize_batch`）",
        "concat": (
            'BATCH_REDUCE_SYSTEM_PROMPT + "\\n\\n" '
            '+ PURE_MODEL_REFERENCE_EXTRACT_INSTRUCTIONS + "\\n\\n" '
            "+ BATCH_SUMMARY_SYSTEM_PROMPT"
        ),
        "note": (
            "由 `_augment_system_for_extract(role_header=BATCH_REDUCE, format_body=BATCH_SUMMARY)` 组装。"
            "中间一段 `PURE_MODEL_REFERENCE_EXTRACT_INSTRUCTIONS` 只在 deepseek-v4-pro 在 60s 超时内返回"
            "非空参考回答时插入，确保它位于「输出格式说明（在 BATCH_SUMMARY_SYSTEM_PROMPT 内）」**上方**；"
            "若超时/失败/空回复则该段缺失，最终拼接退化为 `BATCH_REDUCE + BATCH_SUMMARY`。"
        ),
        "pieces": [
            ("BATCH_REDUCE_SYSTEM_PROMPT（角色头）", BATCH_REDUCE_SYSTEM_PROMPT),
            (
                "PURE_MODEL_REFERENCE_EXTRACT_INSTRUCTIONS（外部参考回答处理策略）",
                PURE_MODEL_REFERENCE_EXTRACT_INSTRUCTIONS,
            ),
            ("BATCH_SUMMARY_SYSTEM_PROMPT（动作规范 + 输出格式说明）", BATCH_SUMMARY_SYSTEM_PROMPT),
        ],
        "final": BATCH_SUMMARY_FINAL,
    },
    {
        "name": "streaming_batch_summary",
        "desc": "流式 chunk 调度下按 slot 顺序的中间提炼（`_streaming_submit_batch`）",
        "concat": (
            'BATCH_REDUCE_SYSTEM_PROMPT + "\\n\\n" '
            '+ PURE_MODEL_REFERENCE_EXTRACT_INSTRUCTIONS + "\\n\\n" '
            "+ BATCH_SUMMARY_SYSTEM_PROMPT"
        ),
        "note": (
            "与 `batch_summary` 完全同构，差别只在触发位置（chunk 流式）。`_augment_system_for_extract` "
            "的插入规则一致：外部参考回答处理策略放在格式说明上方。"
        ),
        "pieces": [
            ("BATCH_REDUCE_SYSTEM_PROMPT（角色头）", BATCH_REDUCE_SYSTEM_PROMPT),
            (
                "PURE_MODEL_REFERENCE_EXTRACT_INSTRUCTIONS（外部参考回答处理策略）",
                PURE_MODEL_REFERENCE_EXTRACT_INSTRUCTIONS,
            ),
            ("BATCH_SUMMARY_SYSTEM_PROMPT（动作规范 + 输出格式说明）", BATCH_SUMMARY_SYSTEM_PROMPT),
        ],
        "final": BATCH_SUMMARY_FINAL,
    },
    {
        "name": "batch_final_merge",
        "desc": "所有中间摘要的最终合并 + 清洗一体化作答（`_batch_final_merge`）",
        "concat": (
            'SUMMARY_ANSWER_SYSTEM_PROMPT + "\\n\\n" '
            '+ PURE_MODEL_REFERENCE_ANSWER_INSTRUCTIONS + "\\n\\n" '
            '+ "## 输出格式约束\\n" + BATCH_MERGE_AND_CLEAN_THINK_SYSTEM_PROMPT'
        ),
        "note": (
            "由 `_augment_system_for_answer(role_header=SUMMARY_ANSWER, "
            "format_body=\"## 输出格式约束\\n\" + BATCH_MERGE_AND_CLEAN_THINK)` 组装"
            "（`summaryCleanAnswer=True` + `thinkMode=True` 触发拼上 BATCH_MERGE_AND_CLEAN_THINK）。"
            "中间一段 `PURE_MODEL_REFERENCE_ANSWER_INSTRUCTIONS` 仅在外部参考回答非空时插入，"
            "确保它位于「## 输出格式约束 + JSON schema」**上方**；若外部参考为空则该段缺失。"
        ),
        "pieces": [
            ("SUMMARY_ANSWER_SYSTEM_PROMPT（角色 + 推理范式）", SUMMARY_ANSWER_SYSTEM_PROMPT),
            (
                "PURE_MODEL_REFERENCE_ANSWER_INSTRUCTIONS（外部参考回答处理策略）",
                PURE_MODEL_REFERENCE_ANSWER_INSTRUCTIONS,
            ),
            (
                '"## 输出格式约束\\n" + BATCH_MERGE_AND_CLEAN_THINK_SYSTEM_PROMPT（输出格式 / JSON schema）',
                BATCH_FINAL_MERGE_FORMAT_BODY,
            ),
        ],
        "final": BATCH_FINAL_MERGE_FINAL,
    },
]


def main() -> None:
    lines: list[str] = []
    lines.append("# 默认参数下四阶段拼接后的 system prompt（按 app.py 默认）")
    lines.append("")
    lines.append(
        "> 本文件由 `scripts/dump_default_system_prompts.py` 自动导出，便于核对每个阶段实际下发给 "
        "LLM 的 system prompt。**默认值以 `app.py` 中 Pydantic Field 的 default 为准**——"
        "这是 web 服务实际生效的默认参数。"
    )
    lines.append("")
    lines.append("## 默认参数前提（来自 app.py）")
    lines.append("")
    lines.append(DEFAULTS_TABLE)
    lines.append("")
    lines.append("> 备注：")
    lines.append(">")
    lines.append(
        "> - `_augment_system_for_extract(role_header, format_body)` 与 "
        "`_augment_system_for_answer(role_header, format_body)` 的组装顺序固定为 "
        "`role_header` → `PURE_MODEL_REFERENCE_*_INSTRUCTIONS`（仅当 `pureModelResult=True` "
        "**且** deepseek-v4-pro 在 60s 超时内返回非空参考回答时插入）→ `format_body`，"
        "确保「外部参考回答处理策略」始终位于「格式要求 / 输出 schema」**上方**。"
    )
    lines.append(
        "> - 若外部参考回答为空（超时 / 失败 / 空回复 / `pureModelResult=False`），中间一段会缺失，"
        "最终拼接退化为 `role_header + format_body` 两段。"
    )
    lines.append(
        "> - chunk_reason 阶段（`_reason_on_chunk`）**没有**经过 `_augment_system_for_extract`，"
        "因此即使 `pureModelResult=True` 也不会出现 PURE_MODEL_* 指令。"
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 四阶段拼接对照表")
    lines.append("")
    lines.append("| 阶段 | 拼接公式 | 字符数 |")
    lines.append("| --- | --- | --- |")
    for s in STAGES:
        lines.append(f"| `{s['name']}` | {s['concat']} | {len(s['final'])} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    for idx, s in enumerate(STAGES, 1):
        lines.append(f"## {idx}. `{s['name']}`")
        lines.append("")
        lines.append(f"**用途**：{s['desc']}")
        lines.append("")
        lines.append(f"**拼接公式**：`{s['concat']}`")
        lines.append("")
        lines.append(f"**触发条件备注**：{s['note']}")
        lines.append("")

        if len(s["pieces"]) > 1:
            lines.append("### 组成部件")
            lines.append("")
            for name, body in s["pieces"]:
                lines.append(f"#### `{name}`")
                lines.append("")
                lines.append("```text")
                lines.append(body.rstrip())
                lines.append("```")
                lines.append("")

        lines.append("### 最终拼接结果（实际下发给 LLM 的 system 内容）")
        lines.append("")
        lines.append("```text")
        lines.append(s["final"].rstrip())
        lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")

    out_dir = "docs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "system_prompts_default.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {out_path}: {sum(len(x) for x in lines)} chars, {len(lines)} lines")


if __name__ == "__main__":
    main()
