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

# --- redis ---------------------------------------------------------------

TASK_KEY_PREFIX: str = "inference:task:"
TASK_TTL_SECONDS: float = 86400.0
