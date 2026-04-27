"""HTTP 接口的请求 / 响应 pydantic 模型。

统一风格：
- 所有响应都带 status_code 与 message 字段，与本项目 app.py 保持一致；
- 具体数据装在 data 字段里，避免未来扩展额外 meta 字段时破坏兼容。
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ----------------------------------------------------------------- 通用响应


class BaseResp(BaseModel):
    status_code: int = 200
    message: str = "success"


# --------------------------------------------------------------------- KV


class SetReq(BaseModel):
    key: str = Field(min_length=1)
    value: Any = Field(description="任意 JSON 可序列化值：str / number / bool / dict / list / null")
    ttl_seconds: Optional[float] = Field(
        default=None,
        description="过期秒数；None 或 <=0 表示永不过期",
    )


class GetResp(BaseResp):
    data: dict[str, Any] = Field(
        default_factory=dict,
        description='{"exists": bool, "value": Any}',
    )


class DeleteResp(BaseResp):
    data: dict[str, Any] = Field(
        default_factory=dict,
        description='{"deleted": bool}',
    )


class ExistsResp(BaseResp):
    data: dict[str, Any] = Field(
        default_factory=dict,
        description='{"exists": bool}',
    )


class ExpireReq(BaseModel):
    key: str = Field(min_length=1)
    ttl_seconds: float = Field(gt=0)


class ExpireResp(BaseResp):
    data: dict[str, Any] = Field(
        default_factory=dict,
        description='{"applied": bool}  —— key 不存在时 False',
    )


class TtlResp(BaseResp):
    data: dict[str, Any] = Field(
        default_factory=dict,
        description='{"ttl_seconds": float|None}  —— None=不存在; -1=永不过期',
    )


class KeysResp(BaseResp):
    data: dict[str, Any] = Field(
        default_factory=dict,
        description='{"keys": list[str]}',
    )


# ------------------------------------------------------------------- Queue


class PushReq(BaseModel):
    queue: str = Field(min_length=1)
    value: Any


class PushResp(BaseResp):
    data: dict[str, Any] = Field(
        default_factory=dict,
        description='{"length": int}  —— rpush 后的队列长度',
    )


class PopReq(BaseModel):
    queue: str = Field(min_length=1)


class BPopReq(BaseModel):
    queue: str = Field(min_length=1)
    timeout_seconds: float = Field(
        default=0,
        description="阻塞等待秒数；<=0 表示一直等到有值为止",
    )


class PopResp(BaseResp):
    data: dict[str, Any] = Field(
        default_factory=dict,
        description='{"ok": bool, "value": Any}  —— ok=False 表示队列空 / 超时',
    )


class LLenResp(BaseResp):
    data: dict[str, Any] = Field(
        default_factory=dict,
        description='{"length": int}',
    )


class LRangeResp(BaseResp):
    data: dict[str, Any] = Field(
        default_factory=dict,
        description='{"items": list[Any]}  —— 闭区间、支持负索引',
    )


class LRemReq(BaseModel):
    queue: str = Field(min_length=1)
    value: Any
    count: int = Field(
        default=0,
        description=">0 从头删 count 个；<0 从尾删 |count| 个；=0 删全部",
    )


class LRemResp(BaseResp):
    data: dict[str, Any] = Field(
        default_factory=dict,
        description='{"removed": int}',
    )


# -------------------------------------------------------------------- Admin


class StatsResp(BaseResp):
    data: dict[str, Any] = Field(default_factory=dict)


class SnapshotResp(BaseResp):
    data: dict[str, Any] = Field(
        default_factory=dict,
        description='{"path": str, "saved_at": float}',
    )


class FlushResp(BaseResp):
    data: dict[str, Any] = Field(default_factory=dict)
