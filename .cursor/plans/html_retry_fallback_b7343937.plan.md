---
name: HTML Retry Fallback
overview: 在 v2 最终 summary 的 JSON 输出失败后，先用同一份最终提示词改成 HTML 标签输出重试；若 HTML 解析仍失败，再执行你指定的兜底：把已存在字段内容放入 answer。
todos:
  - id: add-parsers
    content: 添加严格 JSON 与 HTML analysis/answer 解析 helper
    status: pending
  - id: wire-v2-retry
    content: 把 v2 最终 summary/merge 调用统一接入 JSON 失败后的 HTML retry
    status: pending
  - id: update-app-fallback
    content: 更新 app.py 的 HTML 解析和字段缺失兜底策略
    status: pending
  - id: validate
    content: 执行语法检查和 linter 诊断
    status: pending
isProject: false
---

# HTML Retry Fallback

## Scope
- 修改 `reasoner/v2/agent_graph.py`：只作用于 `summary_clean_answer=True` 且 `think_mode=True` 的 v2 最终节点，包括 `_final_summary`、`_batch_final_merge`、`_retrieval_final_summary`、`_retrieval_batch_final_merge`，以及 extra skill 后的 `_all_in_answer`。
- 修改 `app.py`：让响应层的 `_split_analysis_concise_answer()` 支持最终 HTML 标签解析，并调整字段缺失兜底逻辑。

## Implementation
- 在 `reasoner/v2/agent_graph.py` 增加统一最终调用 helper：第一次仍使用现有 JSON prompt；返回后严格解析 `analysis`/`answer`，判定 JSON 无法解析、字段缺失或字段内容为空为失败。
- 失败时发起第二次 LLM 请求：复用第一次已经格式化、已注入 skill/pure-model 的原始 final prompt，只替换“输出格式”要求为 `<analysis>...</analysis><answer>...</answer>`。除输出格式段外，用户问题、证据、内容层面和呈现层面要求保持一致。
- HTML retry 返回后先解析 `<analysis>` 和 `<answer>` 标签；成功时将其规范化成 JSON 字符串返回给上游，保持现有 `app.py` 的 `think <- analysis`、`answer <- answer` 映射稳定。
- 在 `app.py` 增加 HTML 解析兜底：如果未来上游直接返回 HTML 标签，也能解析出 `analysis`/`answer`。
- 调整 `app.py` 字段缺失兜底：如果只拿到 `analysis`，则 `think=analysis` 且 `answer=analysis`；如果只拿到 `answer`，则 `think=""` 且 `answer=answer`；如果只拿到旧 `concise_answer`，仍作为 `answer` 兼容；如果全部缺失或无法解析，才 `think=""`、`answer=raw`。

## Validation
- 用小的单元式样例验证解析函数：合法 JSON、代码块 JSON、缺 `answer`、缺 `analysis`、合法 HTML、HTML 缺标签、完全非结构化文本。
- 运行 `python -m py_compile app.py reasoner/v2/agent_graph.py`。
- 用 `ReadLints` 检查 `app.py` 和 `reasoner/v2/agent_graph.py`。