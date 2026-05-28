---
name: preview related cases
overview: 在 preview 阶段引入 case 库纯向量检索：InferenceRequest 新增 `topC` 与 `caseSimThreshold` 两个请求参数，`topC>0` 时按阈值取 top-c 案例注入 preview prompt，`topC=0` 时关闭 case 模式并使用原 2 套 prompt（不带 `{related_cases_block}` 槽位，0 侵入原行为）。
todos:
  - id: request_params
    content: 在 app.py InferenceRequest 新增 topC / caseSimThreshold 两个字段（含 Pydantic 校验与 description），_build_inference_options 透传给 _InferenceOptions，session_meta 也带上
    status: completed
  - id: options_field
    content: 在 inference/pipeline.py InferenceOptions 新增 case_top_k / case_sim_threshold 字段（默认 3 / 0.85）
    status: completed
  - id: config_defaults
    content: 在 inference/config.py 新增 CASE_SEARCH_TOP_K_DEFAULT / CASE_SEARCH_THRESHOLD_DEFAULT 两个默认常量，供 BaseModel 用作 default
    status: completed
  - id: client_v2
    content: 在 inference/retrieval/client.py 新增 vector_search_v2() 方法，兜底 404/网络异常返回 []
    status: completed
  - id: case_search
    content: 新建 inference/retrieval/case_search.py：CaseHit dataclass + search_cases(question, policy_id, threshold, top_k)，含 collection_id 解析（剥 __cs 后缀 + split _ 取 kh_code）、embedding、阈值过滤、按 cosine 降序截 top-k
    status: completed
  - id: prompts
    content: 改 inference/prompts.py：原 2 套 PREVIEW_*（PREVIEW_* / PREVIEW_*_WITH_TGK）一字不改；新增 2 套 PREVIEW_*_WITH_CASES / PREVIEW_*_WITH_TGK_AND_CASES（system 与 user 各一对）；新增 format_related_cases_block() helper；select_preview_prompt() 接受 related_cases 参数，按 (tgk, cases) 两维路由到 4 套之一
    status: completed
  - id: preview
    content: 改 inference/preview.py：run() 新增 policy_id / case_top_k / case_sim_threshold 入参，topC>0 时调 search_cases，把结果传给 select_preview_prompt
    status: completed
  - id: pipeline
    content: 改 inference/pipeline.py：把 policy_id 与 case 配置透传给 run_preview()
    status: completed
isProject: false
---

## 1. 改动总览

- **请求参数新增**：[app.py](app.py) `InferenceRequest` 新增 `topC: int = 3`（0 关闭 case 检索）与 `caseSimThreshold: float = 0.85`（`topC=0` 时此字段不生效）；`_build_inference_options` / `_build_inference_session_meta` 同步透传
- **InferenceOptions 新增字段**：[inference/pipeline.py](inference/pipeline.py) `InferenceOptions` 新增 `case_top_k: int = 3` / `case_sim_threshold: float = 0.85`
- **新增 case 检索模块**：[inference/retrieval/case_search.py](inference/retrieval/case_search.py) 封装 cosine_similarity 阈值过滤 + top-k
- **扩展 retrieval client**：[inference/retrieval/client.py](inference/retrieval/client.py) 新增 v2 端点 `vector_search_v2()`（现有方法都打 `/v1/...`）
- **prompt 改为 4 套**：[inference/prompts.py](inference/prompts.py) 原 2 套 PREVIEW prompt 一字不改；新增 2 套带 `{related_cases_block}` 的版本；`select_preview_prompt()` 按 `(tgk_present, cases_present)` 二维路由
- **preview / pipeline 透传**：[inference/preview.py](inference/preview.py) `run()` 新增 `policy_id` / `case_top_k` / `case_sim_threshold` 参数；[inference/pipeline.py](inference/pipeline.py) 在调 `run_preview()` 时透传
- **inference/config.py**：只放默认常量供 `InferenceRequest` 的 Pydantic default 引用，**不放全局 enabled 开关**（启停由 `topC` 控制）

向后兼容：
- 老请求方未传 `topC` → 默认 3，走 case 检索；如希望保留旧行为可显式传 `topC=0`
- preview / pipeline 新增参数都有默认值，单测可不传

## 2. 数据流

```mermaid
flowchart LR
    Req["InferenceRequest<br/>topC, caseSimThreshold"] --> Opts[InferenceOptions]
    Opts --> P[pipeline.run]
    P -->|"policy_id + case cfg"| PV[preview.run]
    PV -->|topC > 0| CS[case_search.search_cases]
    PV -->|topC == 0| Skip[skip, cases=[]]
    CS --> EM[embedding_client.embed_texts]
    CS --> LDB["LanceDB v2 /v2/search<br/>top_m=0, query_tokenized=''<br/>collection_id=case_kh_code"]
    LDB -->|"hits with cosine_similarity"| CS
    CS -->|"filter >= threshold, sort desc, top_c"| PV
    PV --> SP[select_preview_prompt]
    SP -->|"cases present"| W4["PREVIEW_*_WITH_CASES /<br/>_WITH_TGK_AND_CASES"]
    SP -->|"cases empty"| W2["PREVIEW_* /<br/>PREVIEW_*_WITH_TGK<br/>(original, unchanged)"]
    W4 --> LLM[preview LLM stream]
    W2 --> LLM
```

## 3. 关键细节

### 3.1 InferenceRequest 新增字段

[app.py](app.py) `InferenceRequest`：

```python
topC: int = Field(
    default=_inference_config.CASE_SEARCH_TOP_K_DEFAULT,
    ge=0,
    le=20,
    description=(
        "preview 阶段 case 库召回数量。"
        "0 表示关闭 case 检索，preview 使用原 2 套 prompt（不带【相关案例经验】段）；"
        ">0 表示按 caseSimThreshold 过滤后取 cosine_similarity 最高的前 N 条，"
        "走带【相关案例经验】的新 prompt。"
    ),
)
caseSimThreshold: float = Field(
    default=_inference_config.CASE_SEARCH_THRESHOLD_DEFAULT,
    ge=0.0,
    le=1.0,
    description=(
        "case 库召回的 cosine_similarity 过滤阈值（默认 0.85）。"
        "仅当 topC>0 时生效；topC=0 时该字段被忽略。"
    ),
)
```

`_build_inference_options` 透传：

```python
return _InferenceOptions(
    ...,
    case_top_k=int(req.topC),
    case_sim_threshold=float(req.caseSimThreshold),
)
```

`_build_inference_session_meta` 加 `"topC": int(req.topC), "caseSimThreshold": float(req.caseSimThreshold)`，便于线下 verbose 复盘。

### 3.2 InferenceOptions 字段

[inference/pipeline.py](inference/pipeline.py)：

```python
@dataclass
class InferenceOptions:
    ...
    case_top_k: int = 3                    # 0 = 关闭
    case_sim_threshold: float = 0.85
```

### 3.3 inference/config.py 默认常量

只放默认值，不放 enabled 开关：

```python
CASE_SEARCH_TOP_K_DEFAULT: int = max(
    0, int(os.getenv("INFERENCE_CASE_SEARCH_TOP_K_DEFAULT", "3"))
)
CASE_SEARCH_THRESHOLD_DEFAULT: float = float(
    os.getenv("INFERENCE_CASE_SEARCH_THRESHOLD_DEFAULT", "0.85")
)
```

env 仅影响"未传 `topC` 时的默认值"；运行时启停以请求体为准。

### 3.4 collection_id 解析

`case_search.py` 内部：

```python
def _resolve_collection_id(policy_id: str) -> str:
    base = (policy_id or "").split("__cs")[0]
    kh_code = base.split("_")[0] if base else ""
    return f"case_{kh_code}" if kh_code else ""
```

剥 `__cs{N}` 后缀是为了兼容 pipeline 现在传入的是 `index_policy_id`（见 [app.py:2312](app.py)：`_inference_run(task_id, question, index_policy_id, rs, ...)`）。

### 3.5 纯向量检索调用

参考 [docs/lancedb_v2_api.md](docs/lancedb_v2_api.md) 第 390–410 行，调 `POST /v2/collections/{case_kh_code}/search`：

```json
{
  "query_tokenized": "",
  "query_vector": [...],
  "top_n": 9,
  "top_m": 0,
  "include_content": true,
  "include_derived": true,
  "strategy": "legacy_hybrid"
}
```

`top_n` 实际取 `max(top_k * 3, top_k + 5)` 多召回一些，方便阈值过滤后仍能凑齐 top-k；最终按 `hit["cosine_similarity"] >= threshold` 过滤并按相似度降序截 top-k。

### 3.6 case hit 字段映射

参考 `.cursor/plans/case_refinery_service_54a867e7.plan.md` 第 117–151 行的 schema：

- 主展示字段：`metadata.refined_knowledge`
- 副字段（refined 为空时 fallback）：`metadata.answer_content` + `metadata.thinking`
- 标签：`metadata.case_polarity` (positive/negative)
- 原问题：`content`（即 questionContent）

返回 dataclass：

```python
@dataclass
class CaseHit:
    cosine_similarity: float
    question: str
    knowledge: str         # refined_knowledge or fallback 拼接
    polarity: str          # "positive" | "negative" | ""
```

### 3.7 PREVIEW prompt 改为 4 套

**原 2 套保留不变**：`PREVIEW_SYSTEM_PROMPT` / `PREVIEW_USER_PROMPT` / `PREVIEW_SYSTEM_PROMPT_WITH_TGK` / `PREVIEW_USER_PROMPT_WITH_TGK`（共 4 个常量，原文不动）。

**新增 2 套**（user 新增 `{related_cases_block}` 槽位，system 在"请基于自身常识"之外多一句"参考**相关案例经验**"）：

```python
PREVIEW_SYSTEM_PROMPT_WITH_CASES = """\
你是一个资深财税实务咨询专家。

请参考**相关案例经验**与自身常识，对用户问题做一次轻量分析
- <think> 标签内容：从财税实务角度对问题做拆解，列出涉及的知识体系与回答逻辑。（500字以内）
- <answer> 标签内容：给出还需要进一步验证的关键点（100字以内）

【绝对约束】
- 仅输出一段 <think>...</think> 和一段 <answer>...</answer>，不要任何其他文字。
- 不确定的思路和常识，允许标注"待结合检索后修正"
- 不用考虑可能的风险点
- 严禁输出具体政策名称、内容及其相关要求（你的信息是过时的）
- 严禁给出问题的答案，只要输出解答思路
- 相关案例经验可作为推理参考，但不可直接抄录其结论
"""

PREVIEW_USER_PROMPT_WITH_CASES = """## 【用户问题】
{question}

## 【相关案例经验】
{related_cases_block}
"""

PREVIEW_SYSTEM_PROMPT_WITH_TGK_AND_CASES = """\
你是一个资深财税实务咨询专家。

请遵循**专题通用知识**，参考**相关案例经验**与自身常识，对**用户问题**做一次轻量分析
- <think> 标签内容：从财税实务角度对问题做拆解，列出涉及的知识体系与回答逻辑。（500字以内）
- <answer> 标签内容：列出与问题相关的专题通用知识、以及还需要进一步验证的关键点。（200字以内）

【绝对约束】
- 仅输出一段 <think>...</think> 和一段 <answer>...</answer>，不要任何其他文字。
- 不确定的思路和常识，允许标注"待结合检索后修正"
- 不用考虑可能的风险点
- 严禁输出具体政策名称、内容及其相关要求（你的自身常识是过时的）
- 严禁给出问题的答案，只要输出解答思路
- 专题通用知识是绝对真理，可以放心输出
- 相关案例经验可作为推理参考，但不可直接抄录其结论
"""

PREVIEW_USER_PROMPT_WITH_TGK_AND_CASES = """## 【专题通用知识】
{topic_general_knowledge}

## 【用户问题】
{question}

## 【相关案例经验】
{related_cases_block}
"""
```

**format_related_cases_block(cases)**：

- `cases` 为空 → 返回 `""`（**且 select_preview_prompt 会因此路由到原 2 套 prompt，不会渲染空段**）
- 非空 → 渲染为 markdown 三级标题列表，如：

```
### 案例1（相似度 0.92，正向案例）
**原问题**：...
**案例知识**：...

### 案例2 ...
```

**select_preview_prompt(question, topic_general_knowledge, related_cases=None)** 路由表：

| tgk 非空 | cases 非空 | 路由到 |
| --- | --- | --- |
| 否 | 否 | PREVIEW_* （现有，不变） |
| 是 | 否 | PREVIEW_*_WITH_TGK （现有，不变） |
| 否 | 是 | PREVIEW_*_WITH_CASES （新增） |
| 是 | 是 | PREVIEW_*_WITH_TGK_AND_CASES （新增） |

注意：**当 `topC>0` 但实际召回 0 条 ≥ 阈值时，cases 为空 → 自动回退到原 PREVIEW_*，不会显示"暂无相关案例"占位**。这是相对上一版 plan 的关键调整，符合"topC=0 与无案例命中体验一致"的语义。

### 3.8 preview 集成

[inference/preview.py](inference/preview.py) 的 `run()` 新增三个可选入参，主流程伪代码：

```python
async def run(
    task_id, question, redis_stream,
    *,
    vendor=..., model=...,
    system_prompt=None, user_prompt=None,
    topic_general_knowledge=None,
    policy_id: Optional[str] = None,
    case_top_k: int = 0,
    case_sim_threshold: float = 0.85,
    tps=config.PREVIEW_TPS,
) -> None:
    related_cases: list[CaseHit] = []
    if case_top_k > 0 and policy_id:
        try:
            from .retrieval.case_search import search_cases
            related_cases = await search_cases(
                question, policy_id,
                threshold=case_sim_threshold,
                top_k=case_top_k,
            )
        except Exception as e:
            logger.warning("[InferencePreview] case_search 失败（忽略）: %s", e)

    default_sys_p, default_usr_p = select_preview_prompt(
        question=question,
        topic_general_knowledge=topic_general_knowledge,
        related_cases=related_cases,
    )
    ...
```

显式传 `system_prompt` / `user_prompt` 时仍优先用入参，保持现有行为。

### 3.9 pipeline 透传

[inference/pipeline.py](inference/pipeline.py) 第 79–87 行调用处改成：

```python
run_preview(
    task_id, question, redis_stream,
    vendor=opts.vendor, model=opts.model,
    topic_general_knowledge=opts.topic_general_knowledge,
    policy_id=policy_id,
    case_top_k=opts.case_top_k,
    case_sim_threshold=opts.case_sim_threshold,
),
```

`pipeline.run(task_id, question, policy_id, ...)` 第三个位置参数已经是 policy_id（实为 `index_policy_id`，由 `case_search` 内部剥 `__cs{N}` 后缀），直接透传。

### 3.10 失败降级矩阵

| 失败场景 | 行为 |
| --- | --- |
| `topC=0` | 跳过 case 检索，走原 2 套 prompt（与改动前完全等价） |
| `policy_id` 为空 / 解析不出 kh_code | 跳过，cases=[]，回退原 2 套 prompt |
| embedding 服务不可用 | 跳过，cases=[]，记 warning |
| LanceDB 服务 404（case 集合不存在） | 跳过，cases=[]，记 info（case_refinery 未上线是预期） |
| LanceDB 5xx / 超时 | 跳过，cases=[]，记 warning |
| 命中 0 条 ≥ 阈值 | cases=[]，回退原 2 套 prompt（不显示占位） |

所有降级路径绝不抛出阻塞 preview。

## 4. RetrievalServiceClient 扩展

[inference/retrieval/client.py](inference/retrieval/client.py) 当前所有方法都打 `/v1/...`，case 库走 `/v2/...`，base_url 与 v1 共用（`http://mlp.paas.dc.servyou-it.com/kh-lancedb`）。新增方法：

```python
async def vector_search_v2(
    self,
    collection_id: str,
    *,
    query_vector: list[float],
    top_n: int,
    include_content: bool = True,
) -> list[dict]:
    body = {
        "query_tokenized": "",
        "query_vector": list(query_vector),
        "top_n": int(top_n),
        "top_m": 0,
        "include_content": bool(include_content),
        "include_derived": True,
        "strategy": "legacy_hybrid",
    }
    try:
        data = await self._request(
            "POST", f"/v2/collections/{_quote(collection_id)}/search", json=body,
        )
    except RetrievalServiceUnavailable:
        return []
    except RetrievalServiceError as e:
        if "404" in str(e):
            return []
        raise
    return list(data.get("hits") or [])
```

返回 raw hits（保留 `cosine_similarity` / `content` / `metadata`），由 `case_search.py` 解析成 `CaseHit`。

## 5. 不做的事

- 不动 react_loop / 最终轮 prompt：用户只要求 preview 注入
- 不做 case 检索结果缓存：preview 每次都重算
- 不动 query_vector 在 hybrid_search 与 case_search 间复用：当前两者并发跑
- 不动 case_refinery 服务（独立迭代，本改动只读它产出的 collection）
