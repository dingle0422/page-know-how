# Page Know-How 推理服务 —— 接口文档

面向调用方的简易版接口说明。覆盖 `app.py` 当前暴露的全部 HTTP 接口。

---

## 0. 全局约定

### 0.1 响应体统一结构

所有业务接口都返回如下外壳（HTTP 层永远是 200，业务成败看 `status_code`）：

```json
{
  "data": { "...": "..." } ,
  "status_code": 200,
  "message": "success"
}
```

| 字段 | 说明 |
| --- | --- |
| `data` | 具体业务数据；失败时通常为 `null` |
| `status_code` | 业务状态码：`200` 成功；`404` 资源不存在；`500` 服务端失败；`503` 依赖服务未就绪 |
| `message` | 人类可读的结果描述 |

### 0.2 架构一览（跟调用方有关的部分）

- 推理请求走 **异步化通道**：`submit` 先入 redis 队列，内置 worker pool 消费，客户端轮询 `result`；
- `MAX_CONCURRENT_REASONING`（默认 10）= worker 数量 = 同时推理上限；
- 批量场景**强烈建议**用 `submit + result` 取代同步 `/api/reason`，避免中间代理长连接超时导致 `NoHttpResponseException`。

---

## 1. 推理接口

### 1.1 `POST /api/reason/submit` — 提交推理任务（推荐）

提交一个推理任务，立刻返回 `taskId`。实际推理在后台 worker 里异步执行。

**Request Body**：参见下方 [ReasonRequest](#21-reasonrequest-字段表)。仅 `policyId` + `question` 为必填，其余均有默认值。

**Response**（`ReasonSubmitResponse`）：

```json
{
  "data": {
    "taskId": "6e3afb6f-db0f-4f8e-bb94-f1cdbced9536",
    "status": "pending",
    "enqueueTime": 1777273678.948
  },
  "status_code": 200,
  "message": "success"
}
```

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `data.taskId` | string | 任务 UUID，轮询 `/api/reason/result/{taskId}` 取结果 |
| `data.status` | string | 提交成功后固定为 `pending` |
| `data.enqueueTime` | float | 任务入队时间（epoch 秒） |

**典型失败**：`status_code=503` redis_server 未就绪；`status_code=500` 入队异常。

---

### 1.2 `GET /api/reason/result/{taskId}` — 查询任务状态 / 结果

按 `taskId` 查询进度。调用方自行轮询，推荐间隔 **1~2 秒**。

**Path Param**：`taskId` — submit 返回的 UUID。

**Response**（`ReasonResultResponse`）：

```json
{
  "data": {
    "taskId": "6e3afb6f-db0f-4f8e-bb94-f1cdbced9536",
    "status": "done",
    "enqueueTime": 1777273678.948,
    "startTime":  1777273679.003,
    "endTime":    1777273728.714,
    "result": {
      "khObj":      "{\"增值税\":\"2.1\"}",
      "policyId":   "5efd92_v1",
      "answer":     "根据…… (最终客服回答)",
      "think":      "……",
      "skillsResult": {},
      "sessionId":  "sess-XXXXX",
      "logName":    "20260427_150810_sess-XXXXX.jsonl"
    },
    "error": null
  },
  "status_code": 200,
  "message": "success"
}
```

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `data.status` | string | `pending` / `running` / `done` / `failed` |
| `data.enqueueTime` | float | submit 时间（epoch 秒） |
| `data.startTime` | float \| null | worker 拉起时间；未开始时 `null` |
| `data.endTime` | float \| null | 结束时间；未完成时 `null` |
| `data.result` | object \| null | `status=done` 时为推理结果（结构见 [ReasonData](#22-reasondata-响应字段)）；其他状态为 `null` |
| `data.error` | string \| null | `status=failed` 时的错误摘要 |

**特殊返回**：
- `status_code=404`：`taskId` 不存在或已过期（默认 24h 后清理，可通过 `REASON_TASK_TTL_SECONDS` 配置）；
- `status_code=503`：redis_server 未就绪。

---

### 1.3 `POST /api/reason` — 同步兼容接口

请求体 / 响应体**与历史版本完全一致**，方便老调用方无感升级。内部实现已改为 `submit + 服务端内部轮询`，所以同样享受队列治理、但连接保持时间由 `/api/reason/submit` 的处理流程决定。**不推荐新接入方使用**。

**Request Body**：同 [ReasonRequest](#21-reasonrequest-字段表)。

**Response**（`ReasonResponse`）：

```json
{
  "data": {
    "khObj":      "{\"增值税\":\"2.1\"}",
    "policyId":   "5efd92_v1",
    "answer":     "……",
    "think":      "……",
    "skillsResult": {},
    "sessionId":  "sess-XXXXX",
    "logName":    "20260427_150810_sess-XXXXX.jsonl"
  },
  "status_code": 200,
  "message": "success"
}
```

- 推理失败：`status_code=500`，`data.answer` 填入 `"推理失败: <原因>"`；
- 客户端连接中途断开：**任务不会中断**，worker 依旧跑完，结果可用同一 `policyId+question` 再次 submit 查回（新的 taskId），或由内部日志检索。

---

## 2. 推理请求 / 响应字段详解

### 2.1 `ReasonRequest` 字段表

#### 核心字段（最常用）

| 字段 | 类型 | 必填 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `policyId` | string | ✅ |  | 政策 / 知识库 ID，对应 `page_knowledge/` 下的目录 |
| `question` | string | ✅ |  | 用户问题 |
| `verbose` | bool | - | `VERBOSE_TRACE` 环境变量 | 打开完整 LLM 输入输出落盘；日志在 `verbose_logs/` |
| `sessionId` | string \| null | - | `null` | 手动指定 verbose session id；不传会自动生成 `sess-xxxxx`（仅 verbose=True 生效） |

#### 模型 / 流水线参数（按需调）

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `version` | string | `"v1"` | 推理引擎版本，`v0`=原始版本；`v1`=统一 EXPLORE + 三层目录树 |
| `vendor` | string | `"aliyun"` | LLM 供应商 |
| `model` | string | `"deepseek-v3.2"` | LLM 模型名 |
| `maxRounds` | int | `10` | 每个子智能体最大 ReAct 轮次 |
| `retrievalMode` | bool | `true` | 召回模式：子智能体只做相关性判定 + 收集原始知识，抗信息畸变 |
| `enableSkills` | bool | `true` | 开启 skill 评估 / double-check |
| `checkPitfalls` | bool | `true` | 一层推理时让 LLM 同步产出易错点，注入总结阶段 |
| `enableRelations` | bool | `true` | 关联条款展开（BFS 多跳拉 `clause.json`），v1 生效 |
| `thinkMode` | bool | `true` | 使用 `*_AND_CLEAN_THINK` prompt，返回 `{analysis, answer}` JSON（兼容 `concise_answer`）；需 `summaryCleanAnswer=true`；v1 生效 |
| `lastThink` | bool | `true` | 最终节点开启底层 LLM 的 `enable_thinking=True`，把推理轨迹回注到 `content` |
| `summaryCleanAnswer` | bool | `true` | summary+clean 一体化，省一次串行 LLM |
| `cleanAnswer` | bool | `false` | summary 后再追一轮 LLM 清洗（与 `summaryCleanAnswer` 相互排斥，优先级较低） |
| `answerSystemPrompt` | string \| null | `null` | 最终作答阶段自定义 system prompt；v1 生效 |

#### 分块 / 分批 / 关联展开调优

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `chunkSize` | int | `3000` | 知识分块字符数上限；0 表示关闭分块 |
| `summaryBatchSize` | int | `3` | 分批总结：每批证据条数；0 表示不分批 |
| `summaryPipelineMode` | string | `"layered"` | `layered`=分层同步；`reduce_queue`=全量入队、无层间同步点（chunk 长尾差异大时更快） |
| `reduceMaxPartDepth` | int | `4` | `reduce_queue` 模式下单 part 最多经过几次中间 BATCH_SUMMARY；命中后 frozen 到 final merge |
| `relationMaxDepth` | int | `5` | 关联展开 BFS 最大跳深 |
| `relationMaxNodes` | int | `999` | 单次 chunk/子智能体触发的关联 BFS 节点总数上限 |
| `relationWorkers` | int | `8` | 关联展开调度线程数 |
| `relationsExpansionMode` | string | `"all"` | `all`=跳过 LLM 二次判定，定位到即展开；`smart`=每个候选 LLM 判 is_relevant |

---

### 2.2 `ReasonData` 响应字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `khObj` | string (JSON) | 知识点名称 → 章节编号的映射，已序列化为 JSON 字符串 |
| `policyId` | string | 回显请求字段 |
| `answer` | string | 面向用户的客服回答；`thinkMode=true` 时取 LLM JSON 输出的 `answer`（兼容 `concise_answer`） |
| `think` | string | `thinkMode=true` 时取 LLM JSON 输出的 `analysis`（完整客服回答 ≤500 字）；未开启时为 `""` |
| `skillsResult` | object | `{skill_name: stdout}`；未触发 skill 时为 `{}` |
| `sessionId` | string \| null | verbose session id；`verbose=false` 时回显请求的 sessionId（可为 null） |
| `logName` | string \| null | verbose 日志文件名（`verbose_logs/<logName>`）；`verbose=false` 为 `null` |

---

## 3. 运维 / 管理接口

### 3.1 `GET /api/requestQueueStatus` — 推理队列状态

从 redis_server 实时读取当前待执行 / 执行中的任务快照，用于监控、排队观察。

**Response**（`QueueStatusResponse`）：

```json
{
  "data": {
    "maxConcurrent": 10,
    "runningCount": 3,
    "queuedCount":  2,
    "running": [ { "...": "QueueEntry" } ],
    "queued":  [ { "...": "QueueEntry" } ]
  },
  "status_code": 200,
  "message": "success"
}
```

`QueueEntry` 字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `taskId` | string | 任务 UUID |
| `question` | string | 原始问题 |
| `policyId` | string | 原始 policyId |
| `state` | string | `queued`=在 redis 队列；`running`=worker 已拉起正在推理 |
| `enqueueTime` | float | 入队时间（epoch 秒） |
| `startTime` | float \| null | 开始推理时间；`queued` 阶段为 `null` |
| `waitingSeconds` | float | `running` 表示排队耗时；`queued` 表示当前已等时长 |
| `runningSeconds` | float \| null | `running` 阶段已推理秒数；`queued` 为 `null` |

> ⚠️ **字段变更**：旧版 `requestId (int)` 已重命名为 `taskId (string)`。如有老监控面板需要同步改字段名 / 类型。

---

### 3.2 `POST /api/kh/update` — 强制重建知识

强制重新抽取并落盘某个 `khId + version` 对应的知识目录（失效本地缓存 / 覆盖 `_policy_index.json`）。

**Request Body**：

| 字段 | 类型 | 必填 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `khId` | string | ✅ |  | 知识主键 |
| `version` | string | ✅ |  | 版本号 |
| `typeId` | string | - | `"0"` | `0`=行业经验、`1`=基础知识、`2`=三方知识 等 |

> 内部会拼成 `policyId = f"{khId}_{version}"` 使用。

**Response**（`KhUpdateResponse`）：

```json
{
  "data": {
    "policyId": "5efd92_v1",
    "update_time": "1777273800123"
  },
  "status_code": 200,
  "message": "success"
}
```

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `data.policyId` | string | 合成后的 policyId |
| `data.update_time` | string | 毫秒级时间戳 |

失败返回 `status_code=500`，`data=null`，`message` 带原因。

---

### 3.3 健康检查

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/example` | 只要进程能响应就返回 `{"status":"ok", "message":"Server is running."}`；供 PaaS readiness 探针使用 |
| GET | `/` | 同上；供 liveness 探针 / 根路径检测使用 |

两者都**不**访问 redis_server 或其他依赖，纯进程存活探针。

---

## 4. 推荐调用姿势

### 4.1 单次请求（对延迟不敏感）
直接 `POST /api/reason`；等待返回即可。

### 4.2 批量调用 / 长耗时请求（推荐）

```text
step 1  POST /api/reason/submit  → 拿 taskId
step 2  循环 GET /api/reason/result/{taskId} ，间隔 1~2s
        直到 data.status ∈ {done, failed}
step 3  status=done → 读 data.result；status=failed → 读 data.error
```

优点：
- 客户端 HTTP 连接短平快，天然避开代理 `proxy_read_timeout` 之类的强制断流；
- 服务端 redis 承接队列，就算客户端断了重连，任务也不丢，24h 内能查回；
- `/api/requestQueueStatus` 可实时看到队列压力。

### 4.3 客户端最佳实践

- **轮询间隔 1~2s** 起步，拿到 `running` 后可以适当放宽到 3~5s；
- `taskId` **本地持久化**，服务重启 / 客户端重连都能接回；
- 对 `status_code=404` 视为"已过期"，需要重新 submit；
- 对 `status_code=503` 退避重试（redis_server 未就绪）。

---

## 5. 相关环境变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `REDIS_SERVER_URL` | `http://127.0.0.1:5000` | redis_server 地址；支持带 path 前缀 |
| `REDIS_SERVER_AUTH_TOKEN` | 空 | redis_server 鉴权 token（若服务端开启） |
| `MAX_CONCURRENT_REASONING` | `10` | worker 数 = 推理并发上限 |
| `REASON_TASK_TTL_SECONDS` | `86400` | 单条任务在 redis 存活秒数（即结果最长可被查到的时间） |
| `REASON_SYNC_POLL_INTERVAL` | `0.5` | `/api/reason` 同步包装内部轮询间隔 |
| `REASON_BLPOP_TIMEOUT_SECONDS` | `10` | worker 每轮 BLPOP 阻塞时长；**必须小于链路上最短的网关 `proxy_read_timeout`**，否则会吃 504 Gateway Timeout |
| `VERBOSE_TRACE` | `false` | `verbose` 字段未显式传入时的默认值 |
