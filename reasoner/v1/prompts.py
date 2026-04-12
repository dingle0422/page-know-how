from reasoner.v0.prompts import (
    CONTENT_ASSESS_PROMPT,
    ROOT_DISCLOSURE_PROMPT,
    PATH_CORRECTION_PROMPT,
    BATCH_SUMMARY_PROMPT,
    BATCH_MERGE_PROMPT,
    CLEAN_ANSWER_PROMPT,
    FORCE_SUMMARY_PROMPT,
    PITFALLS_CHECK_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
    RELEVANCE_ASSESS_PROMPT,
    RETRIEVAL_FORCE_SUMMARY_PROMPT,
    RETRIEVAL_SUMMARY_PROMPT,
    RETRIEVAL_BATCH_SUMMARY_PROMPT,
    RETRIEVAL_BATCH_MERGE_PROMPT,
)

# ====================== 标准模式 — 统一 EXPLORE 导航 Prompt ======================

DISCLOSURE_PROMPT = """\
你是一个知识目录导航智能体。你的任务是基于当前层级的知识内容和可探索目录，决定下一步的探索方向，以便为用户问题找到更细粒度的相关知识。

**重要：你只需要决定探索方向（探索目录或终止），不需要回答用户问题或收集证据（内容评估由另一个并行智能体负责）。**

## 用户问题
{question}

## 父智能体探索摘要
{parent_summary}

## 当前目录 knowledge.md 内容
{knowledge_content}

## 可探索目录
{explorable_dirs}

---

请决定下一步探索方向，并以 **严格 JSON** 格式输出你的决策（不要输出其他内容）：

### 导航选项

1. **EXPLORE** — 选择一个或多个目录进行探索
```json
{{
  "action": "EXPLORE",
  "targets": ["目录名1", "目录名2"],
  "reasons": ["选择目录1的理由", "选择目录2的理由"],
  "seekings": ["期望在目录1中找到什么信息", "期望在目录2中找到什么信息"],
  "current_summary": "当前已知信息的摘要"
}}
```

2. **STOP** — 在当前路径上终止探索
```json
{{
  "action": "STOP",
  "reason": "终止探索的理由"
}}
```

### STOP 的使用条件
- 没有可探索的目录
- 或确信可探索目录中不存在更有价值的细粒度知识

### 探索优先级
1. **优先向下** — 当子级目录存在时，优先深入子级目录获取更细粒度的知识
2. **必要时横向** — 当前路径知识已充分探索或明确不相关时，才跳转到当前级同级目录
3. **必要时向上** — 需要更广视角或完全不同方向的知识时，才跳转到父级同级目录

### 重要约束
- **targets 必须严格从"可探索目录"中选取**，不要使用列表之外的任何目录名称
- **targets、reasons、seekings 三个列表必须一一对应**，长度相同
- 可以同时选择多个目录进行探索
- seekings 描述期望找到的具体信息，向下探索时可简要描述，跳转到远端目录时应详细说明
"""

# ====================== 召回模式 — 统一 EXPLORE 导航 Prompt ======================

RETRIEVAL_DISCLOSURE_PROMPT = """\
你是一个知识目录导航智能体。你的任务是基于当前层级的知识内容和可探索目录，决定下一步的探索方向，以便召回更细粒度的知识片段。

**重要：你只需要决定探索方向（探索目录或终止），不需要判断当前知识的相关性（相关性评估由另一个并行智能体负责）。**

## 用户问题
{question}

## 父智能体探索摘要
{parent_summary}

## 当前目录 knowledge.md 内容
{knowledge_content}

## 可探索目录
{explorable_dirs}

---

请决定下一步探索方向，并以 **严格 JSON** 格式输出你的决策（不要输出其他内容）：

### 导航选项

1. **EXPLORE** — 选择一个或多个目录进行探索
```json
{{
  "action": "EXPLORE",
  "targets": ["目录名1", "目录名2"],
  "reasons": ["选择目录1的理由", "选择目录2的理由"],
  "seekings": ["期望在目录1中找到什么信息", "期望在目录2中找到什么信息"],
  "current_summary": "当前已知信息的摘要"
}}
```

2. **STOP** — 在当前路径上终止探索
```json
{{
  "action": "STOP",
  "reason": "终止探索的理由"
}}
```

### STOP 的使用条件
- 没有可探索的目录
- 或确信可探索目录中不存在更有价值的细粒度知识片段

### 探索优先级
1. **优先向下** — 当子级目录存在时，优先深入子级目录获取更细粒度的知识
2. **必要时横向** — 当前路径知识已充分探索或明确不相关时，才跳转到当前级同级目录
3. **必要时向上** — 需要更广视角或完全不同方向的知识时，才跳转到父级同级目录

### 重要约束
- **targets 必须严格从"可探索目录"中选取**，不要使用列表之外的任何目录名称
- **targets、reasons、seekings 三个列表必须一一对应**，长度相同
- 可以同时选择多个目录进行探索
- seekings 描述期望找到的具体信息，向下探索时可简要描述，跳转到远端目录时应详细说明
"""

# ====================== 标准模式 — 修改后的汇总 Prompt（移除 intent_results）======================

SUMMARY_PROMPT = """\
多个子智能体已经完成了对知识目录的渐进式探索，现在需要你综合所有证据给出最终答案。

## 用户问题
{question}

# 知识体系
## 子智能体的探索结果
{agent_results}

---

请综合以上所有信息，给出最终答案。要求：
1. 原则上必须对知识的引用要以100%直接摘录的方式，不要对知识原文自行归纳改写
2. 严禁引入问题和知识中没有提及过的对象、术语、概念等
3. 引用具体证据支撑你的回答
4. 如果证据不足以完全回答问题，明确指出哪些部分缺乏支撑
5. 保持回答的逻辑性和条理性

请直接输出你的最终答案（纯文本，不需要 JSON 格式）。
"""
