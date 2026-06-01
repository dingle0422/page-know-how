"""inference 模块的默认参数与开关。

所有常量集中在此，方便后续上 env / 配置文件覆盖。
保持纯 dataclass / 模块常量，避免任何重副作用导入。
"""

from __future__ import annotations

import os

# --- ReAct 主循环参数 -----------------------------------------------------

CHUNK_SIZE: int = 5000
"""单轮 ReAct 喂给模型的拼接证据上限（字符数）。"""

INFERENCE_DEFAULT_CHUNK_SIZE: int = int(
    os.getenv("INFERENCE_DEFAULT_CHUNK_SIZE", "500")
)
"""离线建索引（``build_knowledge_chunks`` / ``split_relations_into_chunks``）默认 chunk_size。

与 ``CHUNK_SIZE``（ReAct 单轮证据拼接上限）是两个不同概念：

- ``CHUNK_SIZE``：检索后把 KnowledgeChunk 列表打包成"轮组"喂给 LLM 的字符上限；
- ``INFERENCE_DEFAULT_CHUNK_SIZE``：建索引时切原文 chunk 的字符上限，
  逻辑沿用 V3：单节点超 chunk_size 独立成块、不超则与前块拼接。

``/api/inference/stream`` 的 ``chunkSize`` 入参未传时取本值；
同一 ``policyId`` 下不同 chunkSize 会落到不同的 LanceDB 表
（表名 ``{policyId}__cs{chunkSize}``）。
"""

REACT_MAX_ROUNDS: int = 5
"""ReAct 主循环硬上限：到顶后强制以"最终轮"语义出 answer。"""

REACT_INTERMEDIATE_THINK_ENABLED: bool = (
    os.getenv("INFERENCE_REACT_INTERMEDIATE_THINK", "0").lower() in {"1", "true", "yes", "on"}
)
"""中间 ReAct 轮是否开启 ``enable_thinking``（默认关闭）。

- ``False``（默认）：仅最后一轮 ``enable_thinking=True``。中间轮 ``chunk.think=""``，
  非最终轮的 ``chunk.answer`` 仅作为"前一轮的 answer"在最终轮前被并入接口 ``think``。
- ``True``：中间轮也开 think。聚合时把所有非最终轮的 ``think+answer`` 全拼接进接口 ``think``，
  最终轮 ``think`` 仍并入接口 ``think``，最终轮 ``answer`` 进接口 ``answer``。
"""

# --- 检索参数 -------------------------------------------------------------

TOP_N: int = 20
"""semantic 召回上限。"""

TOP_M: int = 20
"""BM25 召回上限。"""

RRF_K: int = 60
"""RRF 融合 k 常量。"""

# --- preview 节流 ---------------------------------------------------------

PREVIEW_TPS: int = 5
"""preview 阶段写 redis 的目标速率（每秒最多写多少次/字符）。"""

# --- preview case 检索（历史经验） --------------------------------------

CASE_SEARCH_TOP_K_DEFAULT: int = max(
    0, int(os.getenv("INFERENCE_CASE_SEARCH_TOP_K_DEFAULT", "1"))
)
"""``InferenceRequest.topC`` 未传时的默认值（>0 表示开启 case 检索）。

仅作 Pydantic ``Field.default`` 的取值源，不充当全局启停开关——
运行时是否开 case 检索完全以请求体 ``topC`` 为准（0=关闭）。
"""

CASE_SEARCH_THRESHOLD_DEFAULT: float = float(
    os.getenv("INFERENCE_CASE_SEARCH_THRESHOLD_DEFAULT", "0.85")
)
"""``InferenceRequest.caseSimThreshold`` 未传时的默认 cosine_similarity 阈值。

仅在 ``topC>0`` 时生效；``topC=0`` 时该值被忽略。
"""

# --- react 节流 -----------------------------------------------------------

REACT_TPS: int = max(1, int(os.getenv("INFERENCE_REACT_TPS", "10")))
"""react 单轮流式写 redis 的目标速率（每秒最多 flush 多少次）。

与 :data:`PREVIEW_TPS` 同义：把"每 LLM SSE delta 一次 ``RedisStream.update``"
折叠为"每 ``1/REACT_TPS`` 秒批量一次 ``update``"，避免单 token 速度被远程
``redis_server`` HTTP 的 RTT × per-task ``asyncio.Lock`` 串行 卡到上界
``1/(2×RTT)``：

- 单次 ``update`` 在 lock 内必须等一次 ``GET`` + 一次 ``SET`` 远程 HTTP RTT；
- 原"每 delta 一次"模式下，单路 token TPS 上限 ≈ ``1/(2×RTT)``（RTT=15ms
  时只能到 ~33 tokens/s，低于上游 LLM 能提供的速度），多余 token 会反压到
  ``chat_stream`` 的 TCP 接收 buffer，进而让上游"看起来"也变慢——
  在 verbose log 中表现为 react 单轮 ``elapsed_ms`` 变长 + 前端逐字变慢；
- 改成"每 ``1/REACT_TPS`` 秒批量 flush 一次"后，单路 Redis 写频率从
  数十–百 RPS 降到 ~``2*REACT_TPS`` RPS（think/answer 各一次），
  瓶颈被搬离 redis_server 网关的 RTT。

取值参考：

- 应不大于 ``SSE_TICK_MS`` 决定的上界（默认 100ms → 10 Hz）：SSE relay 每
  100ms 才读一次快照，flush 更快也不会更早被前端看到；
- 一般 5–20 之间；过小（如 1）前端逐字吐字会像批量；过大（如 50）会让
  update 频率回到与原方案接近，远程 RTT 压不下来；
- 默认 10 与 ``SSE_TICK_MS=100`` 对齐，可通过环境变量
  ``INFERENCE_REACT_TPS`` 运行时覆盖。
"""

# --- SSE 转发 -------------------------------------------------------------

SSE_TICK_MS: int = 100
"""接口主协程每多少毫秒读一次 redis 快照。"""

SSE_HEARTBEAT_S: float = 15.0
"""SSE 静默期内每多少秒发一次注释心跳，避免被网关 idle timeout 掐断。
建议设小于网关 ``proxy_read_timeout`` 的一半（常见网关 30s/60s）。"""

# --- redis ---------------------------------------------------------------

TASK_KEY_PREFIX: str = "inference:task:"
TASK_TTL_SECONDS: float = 86400.0

# --- 高亮外链关联：离线索引侧静态展开 -----------------------------------------
#
# inference 走的是 retrieval_service（LanceDB）里的 BM25 + 向量混合检索，与
# reasoner 在线模式里 HighlightPrecheck/RelationCrawler 那条实时通路独立。
# 为了在 inference 模式下也能命中"父章节高亮词 -> 关联条款"语义，建索引时
# 复用同一套 RelationCrawler（expand_all=True，纯静态展开，不走 LLM），把
# 命中的 RelationFragment 渲染成派生 chunk 一起 upsert（含 relation_keys 列）。
#
# 关掉本开关时全链路退化为旧行为：不展开关联、跨 policy cascade 也不会触发，
# 便于灰度回滚。
INCLUDE_HIGHLIGHTED_RELATIONS_IN_INDEX: bool = (
    os.getenv("INFERENCE_INCLUDE_HIGHLIGHTED_RELATIONS", "1").lower()
    in {"1", "true", "yes", "on"}
)
"""是否在 inference 离线索引中烘入高亮外链派生 chunk。"""

HIGHLIGHT_INDEX_MAX_DEPTH: int = int(
    os.getenv("INFERENCE_HIGHLIGHT_INDEX_MAX_DEPTH", "5")
)
"""RelationCrawler 多跳 BFS 最大深度，与 reasoner 默认一致。"""

HIGHLIGHT_INDEX_MAX_NODES: int = int(
    os.getenv("INFERENCE_HIGHLIGHT_INDEX_MAX_NODES", "50")
)
"""RelationCrawler 单次 crawl 节点预算，防止极端引用图爆炸。"""

HIGHLIGHT_INDEX_ALLOW_REMOTE: bool = (
    os.getenv("INFERENCE_HIGHLIGHT_INDEX_ALLOW_REMOTE", "1").lower()
    in {"1", "true", "yes", "on"}
)
"""离线建索引时，本地未命中的 (policy_id, clause_id) 是否允许走远程兜底。
关掉则该条 fragment 直接跳过；级联 stale 机制会在目标 policy 落盘后兜底刷新。"""

HIGHLIGHT_INDEX_REMOTE_TIMEOUT: float = float(
    os.getenv("INFERENCE_HIGHLIGHT_INDEX_REMOTE_TIMEOUT", "5.0")
)
"""ClauseLocator._try_remote 单条超时（秒）。"""

INDEX_SCHEMA_VERSION: int = 3
"""离线索引 schema 版本。升级时旧索引会被 _inference_artifacts_stale 判定为过期
触发自动重建。

- v1 = 仅 knowledge.md；
- v2 = v1 + 高亮外链派生 chunk；
- v3 = 三件套（_chunks.jsonl/_bm25.pkl/_embeddings.npy）整体迁移到 retrieval_service
       (LanceDB)；schema 增加 kind/parent_chunk_index/derived_seq/relation_keys 等列。
"""

# --- retrieval_service HTTP 客户端 ---------------------------------------
#
# inference 检索后端从"本地三件套"切换到独立 FastAPI 服务。默认指向已部署的生产地址，
# 本地起服务时通过环境变量 RETRIEVAL_SERVICE_URL=http://127.0.0.1:8088 覆盖即可。
RETRIEVAL_SERVICE_URL: str = os.getenv(
    "RETRIEVAL_SERVICE_URL", "http://mlp.paas.dc.servyou-it.com/kh-lancedb"
).strip().rstrip("/")
"""retrieval_service 基础 URL。"""

RETRIEVAL_SERVICE_API_KEY: str = os.getenv("RETRIEVAL_SERVICE_API_KEY", "")
"""retrieval_service 鉴权 key（X-API-Key 或 Authorization: Bearer）。空表示关闭鉴权。"""

RETRIEVAL_SERVICE_TIMEOUT: float = float(
    os.getenv("RETRIEVAL_SERVICE_TIMEOUT", "30.0")
)
"""单次请求超时（秒）。upsert 大批量时可适当调大。"""
