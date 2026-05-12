"""inference 模块的默认参数与开关。

所有常量集中在此，方便后续上 env / 配置文件覆盖。
保持纯 dataclass / 模块常量，避免任何重副作用导入。
"""

from __future__ import annotations

import os

# --- ReAct 主循环参数 -----------------------------------------------------

CHUNK_SIZE: int = 5000
"""单轮 ReAct 喂给模型的拼接证据上限（字符数）。"""

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
# inference 走的是离线索引（_chunks.jsonl + _bm25.pkl + _embeddings.npy），与
# reasoner 在线模式里 HighlightPrecheck/RelationCrawler 那条实时通路独立。
# 为了在 inference 模式下也能命中"父章节高亮词 -> 关联条款"语义，建索引时
# 复用同一套 RelationCrawler（expand_all=True，纯静态展开，不走 LLM），把
# 命中的 RelationFragment 渲染成派生 chunk 一起喂给 BM25/embedding。
#
# 关掉本开关时全链路退化为旧行为：不展开关联、不写 _relation_targets.json、
# 不级联其他 root，便于灰度回滚。
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

INDEX_SCHEMA_VERSION: int = 2
"""离线索引 schema 版本。升级时旧索引会被 _inference_artifacts_stale 判定为过期
触发自动重建。v1 = 仅 knowledge.md；v2 = v1 + 高亮外链派生 chunk。"""
