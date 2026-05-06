---
name: v3 final-summary heading restructure
overview: 将 v3 chunk+layered+relations 默认链路的 final-summary 拼装方式改造为「按来源分组、每组独立 reduce 至 1 条 summary、最终 prompt 仅保留最细粒度章节标题/高亮关键词标识」。同时把 answerRefine 在 _reason_executor 中的兜底改为 False，让 final summary 阶段保持「think 来自 reasoning_content、answer 来自 <answer> 标签」的现有行为。
todos:
  - id: group-relations-by-keyword
    content: _streaming_run_relations 按 frag.highlighted 分桶，_OrderedSlot 增 derived_chunks_by_keyword / derived_results_by_keyword，兼容现有 trace 用的聚合字段
    status: completed
  - id: orchestrate-by-group
    content: 新增 _streaming_orchestrate_by_group：按 self 组 + 每个 keyword 组收齐 part，每组独立提交 bs_pool；HighlightPrecheck orphan 按 highlighted 分派到对应组
    status: completed
  - id: reduce-group-to-single
    content: 新增 _reduce_group_to_single 递归调用 BATCH_SUMMARY_PROMPT 将单组压至 1 条，不走 _recursive_batch_reduce/_batch_final_merge
    status: completed
  - id: assemble-final-evidence
    content: 新增 _format_groups_for_corpus_evidence + _chunk_streaming_assemble_final：拼接 【知识块章节】 / 【高亮关键词】 区块，调 _build_corpus_final_prompt 走 corpus final 并保留 double-check
    status: completed
  - id: clean-heading-util
    content: 新增 _clean_heading_path：按 『 > 』切段、每段去掉 ^[\w\.]+_ 前缀，提供给 evidence 拼装使用
    status: completed
  - id: rewire-chunk-mode
    content: _run_chunk_mode 在 layered+relations 分支改调 _streaming_orchestrate_by_group 返回的 groups，取代原 _recursive_batch_reduce(batch_outputs, layer=2)，其他路径保持不变
    status: completed
  - id: trace-compat
    content: 检查 _build_chunk_trace_log 调用 slot.derived_chunks/derived_results 的地方仍能拿到所有派生 chunk；补一段「各组 final summary 长度」总览
    status: completed
  - id: answer-refine-default-off
    content: app.py:1008 把 answer_refine=request_payload.get('answerRefine', True) 改为 False
    status: completed
  - id: self-verify
    content: 确认 _corpus_chat_to_think_answer_json 仍然负责拆 reasoning_content 与 <answer>；answerRefine 关闭后不会被 _refine_answer 覆盖
    status: completed
isProject: false
---

# v3 推理链路 final-summary 输入重构 + answerRefine 默认关闭

## 语境锁定（默认参数）

默认值以 [app.py:158-322](app.py) 的 `ReasonRequest` 字段定义为准（不是 `_reason_executor` 里的兜底）：

- `version="v3"`、`vendor="aliyun"`、`model="deepseek-v3.2"`、`maxRounds=10`
- `chunkSize=3000`（>0 → 进入 chunk 模式）、`summaryBatchSize=3`、`summaryPipelineMode="layered"`
- `enableRelations=True`、`relationsExpansionMode="all"`、`relationMaxDepth=5`、`relationMaxNodes=999`、`relationWorkers=8`
- `retrievalMode=True`（chunk 模式下不接管 pipeline，仅对 standard 路径生效）
- `thinkMode=True`、`lastThink=True`、`summaryCleanAnswer=True`、`cleanAnswer=False`、`checkPitfalls=True`
- `enableSkills=True`、`pureModelResult=False`、**`answerRefine=False`**（字段层默认就关闭）

由于 `version="v3"` 且 `chunkSize>0`，请求落到 `reasoner/v3/agent_graph.py::AgentGraph` 的 `_run_chunk_mode()`；relations + layered 进入 `_chunk_streaming_pipeline`（[reasoner/v3/agent_graph.py:1725-1762](reasoner/v3/agent_graph.py)）。其它模式（reduce_queue、retrieval-only、standard）本次不动。

关于 `answerRefine`：FastAPI 通过 pydantic 解析后 `request_payload["answerRefine"]` 已经是 `False`，所以「实际跑出来的默认行为」就是不调 `_refine_answer`。[app.py:1008](app.py) 那个 `True` 兜底只在「上游绕过 pydantic 直接喂 dict」时才生效，目前没有该路径；本次把它改成 `False` 是消除歧义性的字段层对齐，**不会改变现有 API 实际行为**。

## 目标 evidence 段（CORPUS_USER_PROMPT.evidence 占位符）

```
【参考知识】

{self 组 reduce 至 1 条的最终 summary 正文}

---

【关键词】土地使用税
【关键词相关知识】
- 城镇土地使用税 > 征收范围
- 城镇土地使用税 > 税率

{该关键词组 reduce 至 1 条的最终 summary 正文}

---

【关键词】免税
【关键词相关知识】
- 免税情形 > 农业生产

{该关键词组 reduce 至 1 条的最终 summary 正文}
```

布局规则：
- self 组：仅保留 `【参考知识】` 小节标题 + summary 正文，**不再列章节 bullet**。
- keyword 组：`【关键词】X` + `【关键词相关知识】` + 该关键词覆盖到的最细粒度章节 bullet + summary 正文。
- 章节 bullet 路径剥掉 `^[\w\.]+_` 前缀（"2.1.3_视同销售" → "视同销售"）。
- 完全去掉所有 `===== BEGIN_BATCH_SUMMARY n =====` / `[摘要序号]` / `[摘要内容开始/结束]` 等内部哨兵。
- 区与区之间用 `\n\n---\n\n` 分隔。

## 改造点

### 1. `reasoner/v3/agent_graph.py` — 派生 chunk 按高亮关键词分桶

- 在 `_OrderedSlot` 上新增 `derived_chunks_by_keyword: dict[str, list[KnowledgeChunk]]` 与 `derived_results_by_keyword: dict[str, dict[int, dict]]`，沿用既有 `derived_done` Event 表示该 slot 的全部派生 chunk 已完成。
- `_streaming_run_relations`（[reasoner/v3/agent_graph.py:1903-1955](reasoner/v3/agent_graph.py)）改造：先按 `frag.highlighted` 把 `all_fragments` 分桶（空 `highlighted` 归到 `__no_keyword__` 兜底桶），逐桶调 `split_relations_into_chunks` 得到 `derived_chunks_by_keyword`；保留 `slot.derived_chunks` 字段（合并所有桶）以兼容 trace 渲染。`_streaming_run_derived` 把 result 同时写入 `derived_results` 与 `derived_results_by_keyword[keyword][derived_seq]`。
- `_reduce_run_relations` 在 reduce_queue 路径下不属于本次改造范围，可保持现状（默认参数 `summaryPipelineMode=layered` 不会进入）。

### 2. `reasoner/v3/agent_graph.py` — 用「按组独立 reduce」替换 `_streaming_orchestrate`

新增 `_streaming_orchestrate_by_group(slots, batch_size, bs_pool)`，思路：

- 等所有 slot.derived_done。
- 构建分组列表 `groups: list[_StreamGroup]`：
  - `self` 组：所有 slot.self_part（按 slot 顺序），收集每个 chunk 结果里的 `relevant_headings`。
  - 每个 highlight keyword 组：跨所有 slot 把 `derived_chunks_by_keyword[keyword]` 对应的 part 收齐，收集这些 chunk 结果里的 `relevant_headings`。
- 把 HighlightPrecheck 兜底产生的 `_build_chunk_orphan_part`（[reasoner/v3/agent_graph.py:2831-2881](reasoner/v3/agent_graph.py)）按 fragment.highlighted 拆分到对应关键词组（已存在的关键词组追加；不存在则新建关键词组），不再单独追加到 `batch_outputs`。
- 每组提交一个 `bs_pool` 任务跑「递归 batch reduce 至 1 条」（见第 3 点）。`bs_pool` 的 `max_workers` 从 5 调为 `min(8, len(groups)+5)` 以容纳更多关键词组。
- 等所有组完成 → 返回 `list[_StreamGroup]`，每个 group 含 `(kind, keyword, cleaned_headings, final_summary)`。

`_chunk_streaming_pipeline` 不再返回 `batch_outputs/ordered_parts`，改为直接返回 `groups`；`_run_chunk_mode` 路径在 layered+relations 分支下调用新的 `_chunk_streaming_assemble_final(groups, summary_skill_context)` 收口。

### 3. `reasoner/v3/agent_graph.py` — 单组内的递归压缩到 1 条

新增 `_reduce_group_to_single(parts: list[str]) -> str`：

- `len(parts) == 0`：返回空串（外层会跳过该组渲染）。
- `len(parts) == 1`：直接返回 parts[0]，不调 LLM（仍然是 chunk LLM 输出的 analysis 段，已经足够压缩）。
- `len(parts) > 1`：复用 `BATCH_SUMMARY_PROMPT` + `_streaming_submit_batch` 思路，每 `summary_batch_size` 一批并发跑 BATCH_SUMMARY；产物再次进入 `_reduce_group_to_single`，直到 `len == 1`。每层 batch 调用复用 bs_pool。

不调用现有的 `_recursive_batch_reduce`，避免它走 `_batch_final_merge` → CORPUS final 而提前终止。

### 4. `reasoner/v3/agent_graph.py` — 新的 final assemble + corpus 调用

新增 `_chunk_streaming_assemble_final(groups, skill_context)`：

- 用 `_format_groups_for_corpus_evidence(groups)`（新工具函数）拼成上面示例的 evidence 字符串：
  - self 组：`【参考知识】\n\n{summary}`（**不列章节 bullet**；summary 为空时整组省略不渲染）。
  - keyword 组：`【关键词】{keyword}\n【关键词相关知识】\n- {h}\n...\n\n{summary}`；headings 列表去重（保留首次出现顺序）；headings 全空时省略 `【关键词相关知识】` 子段；keyword 为空（兜底桶 `""`）时回落为 self 组同款 `【参考知识】` 标签，避免出现空关键词标题。
  - 区间用 `\n\n---\n\n` 分隔。
- 调 `_build_corpus_final_prompt(evidence)` 拿 `(system_prompt, prompt)`。
- 走 `_finalize_with_double_check(final_summary_callable=_do_merge, judge_evidence=evidence, stage_label="ChunkStreamingFinal·Corpus")`，`_do_merge` 复用 `_corpus_chat_to_think_answer_json`，保持「think=reasoning_content、answer=<answer> 段」输出。

### 5. `reasoner/v3/agent_graph.py` — 章节标题清洗工具

在 `_format_groups_for_corpus_evidence` 同文件中新增 `_clean_heading_path(raw: str) -> str`：按 ` > ` 切段，每段用 `re.sub(r"^[\w\.]+_", "", segment)` 去掉数字/字母编号前缀，再用 ` > ` 拼回。组装 evidence 前，对每个 chunk 的 `relevant_headings` 全量展开 + 跨 chunk 去重（保留首次出现顺序）。

### 6. `app.py` — answerRefine 兜底对齐字段层默认

[app.py:1008](app.py) 把 `answer_refine=request_payload.get("answerRefine", True)` 改为 `False`，与 `ReasonRequest.answerRefine` 字段 `default=False`（[app.py:299-308](app.py)）对齐。

注意（与 ReasonRequest 字段默认一致后）：
- API 入口（`/api/reason`、`/api/reason/submit`）走 pydantic 解析，`request_payload["answerRefine"]` 已经是 `False`，本次改动**不改变实际行为**，只是消除"误导性的 True 兜底"。
- 真正确保了：在 v3 默认链路里，`_refine_answer` 不会再被调用，`_corpus_chat_to_think_answer_json` 构造好的 `{think, answer}` JSON 直接进入响应解析，think=reasoning_content、answer=`<answer>` 标签内部。

### 7. final summary think/answer 提取（确认现状即可）

`_corpus_chat_to_think_answer_json`（[reasoner/v3/agent_graph.py:865-910](reasoner/v3/agent_graph.py)）已经做：

- `reasoning, body = split_think_block(raw)` → think 取 `reasoning_content`。
- `answer = self._extract_user_facing_answer(body)` → answer 取 `<answer>...</answer>` 内部。
- 输出 `{"think": reasoning, "answer": answer}` JSON 串，`_split_analysis_concise_answer`（[app.py:700-769](app.py)）映射到 `ReasonData.think / answer`。

只要第 6 点把 answerRefine 默认关掉，不会再有覆盖；本节无代码改动，仅作收尾自检。

### 8. trace 日志兼容

`_build_chunk_trace_log`（[reasoner/v3/agent_graph.py:2582-2646](reasoner/v3/agent_graph.py)）目前消费 `slot.derived_chunks / slot.derived_results`，改造时保留这两个聚合字段（在新数据结构基础上补一份），trace 渲染保持原样。新增一段「按高亮关键词的最终 summary 长度」概览即可。

## 风险与回退点

- 关键词数量极多时 final prompt 会变长；可后续视情况加上限（暂不处理）。
- 派生 chunk 内 fragment 实际只持一个 `highlighted` 字段（`RelationFragment.highlighted` 是单值），所以「同一 chunk 跨关键词」的歧义在现有数据模型下不会出现；不再单独考虑。
- 仅 `summary_pipeline_mode=layered` + relations + chunk 路径有改动，其他模式（reduce_queue、retrieval、standard）保持原状，回退面较小。
