"""redis_server 的 Python 异步客户端。

**注意**：本模块此前是一个走 HTTP（httpx）调用自研轻量 KV+Queue 服务的薄封装，
现已无缝切换为**直连专业 Redis Sentinel 集群**（哨兵 + 主从）。

对上层而言，`RedisServerClient` 的公开方法签名与语义完全保持不变：
    - KV    : set / get / delete / exists / expire / ttl / keys
    - Queue : rpush / lpop / blpop / llen / lrange / lrem
    - Admin : health / stats / snapshot / flushall
因此 app.py / task_queue.py / inference/redis_stream.py 等调用方**无需任何改动**。

值语义说明
----------
旧的自研服务端直接存取 Python 对象（dict / list / str / ...）。Redis 只能存字符串，
因此本客户端在**写入时统一 ``json.dumps``、读取时统一 ``json.loads``**，对外暴露的
依旧是原始 Python 对象，行为与旧实现一致（dict 进 → dict 出）。``set(key, value)``
里的 value 可以是任意可 JSON 序列化对象。队列同理（rpush 的 value、lrange/lpop/blpop
返回值均经 JSON 编解码）。

连接配置（环境变量优先，默认值即下方集群）
------------------------------------------------
    REDIS_SENTINELS    : "host:port,host:port,..."（哨兵节点列表）
    REDIS_MASTER_NAME  : Sentinel 监控的 master 名
    REDIS_PASSWORD     : 连接 Redis 数据节点的密码
    REDIS_SENTINEL_PASSWORD : 连接 Sentinel 节点的密码（可选，默认不配=不对哨兵鉴权）
    REDIS_USERNAME     : 连接 Redis 数据节点的 ACL 用户名（可选）
    REDIS_SENTINEL_USERNAME : 连接 Sentinel 节点的 ACL 用户名（可选）
    REDIS_KEY_PREFIX   : 可选的全局 key 前缀（默认空，完全沿用旧 key 命名）
    REDIS_ALLOW_FLUSH  : 置 "1" 才允许 flushall 真正清库（防止误清共享集群）

示例:
    from redis_server.client import RedisServerClient

    async with RedisServerClient() as cli:
        await cli.set("task:abc", {"status": "pending"}, ttl_seconds=86400)
        await cli.rpush("queue:reason:pending", "abc")
        value, ok = await cli.lpop("queue:reason:pending")
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Optional

import redis.asyncio as aioredis
from redis import exceptions as redis_exceptions
from redis.asyncio.sentinel import Sentinel

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- 集群默认配置
# 全部可被同名环境变量覆盖。默认值即用户提供的专业 Sentinel 集群。

_DEFAULT_SENTINELS = (
    "10.64.2.100:7560,"
    "10.64.2.55:7564,"
    "10.64.2.56:7565,"
    "10.64.2.57:7566,"
    "10.64.2.99:7567"
)
_DEFAULT_MASTER_NAME = "g-alg-solution-sen-6464"
_DEFAULT_PASSWORD = "0d18966e5876fb3155ddfa5c0a7a25be"
TARGET_DB = 0


def _parse_sentinels(raw: str) -> list[tuple[str, int]]:
    """把 "host:port,host:port" 解析成 [(host, port), ...]。"""
    out: list[tuple[str, int]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        host, _, port = item.rpartition(":")
        if not host:
            raise ValueError(f"非法 sentinel 配置项（缺少端口）: {item!r}")
        out.append((host.strip(), int(port.strip())))
    if not out:
        raise ValueError("REDIS_SENTINELS 解析后为空")
    return out


# 连接 Redis / Sentinel 链路上的瞬时错误：连接被对端关闭、读超时、哨兵故障切换
# 窗口期暂时取不到 master 等。这些情况下重试通常即可恢复（命令幂等性见各方法注释）。
_RETRYABLE_REDIS_ERRORS: tuple[type[BaseException], ...] = (
    redis_exceptions.ConnectionError,
    redis_exceptions.TimeoutError,
    # 故障切换瞬间 master 暂不可达 / 哨兵还没选出新 master。
    redis_exceptions.ReadOnlyError,
)


class RedisServerError(RuntimeError):
    """与 Redis 交互失败（连接、超时、命令错误等）时抛出。

    保留此类型名以兼容历史调用方（task_queue 把它归入瞬时网络错误做高层重试）。
    """


class RedisServerClient:
    """异步 Redis 客户端（Sentinel 主从），对外 API 与旧 HTTP 客户端一致。

    **内部维护两个独立连接（各自连接池）**：

    - ``_client``：服务于普通短请求（set / get / rpush / lrange / lpop / lrem ...）。
    - ``_blpop_client``：服务于 ``blpop`` 阻塞长轮询。BLPOP 每次会长期占住一个连接，
      与短请求池分开，避免 worker 把池子占满后 set/get/rpush 等接口被挨饿。

    构造参数尽量与旧 ``RedisServerClient`` 对齐（``base_url`` / ``auth_token`` 仅为
    向后兼容保留，不再使用——实际连接信息来自环境变量 / 默认集群配置）。
    """

    def __init__(
        self,
        base_url: str = "",
        auth_token: str = "",
        timeout_seconds: float = 10.0,
        connect_timeout_seconds: float = 3.0,
        *,
        pool_timeout_seconds: float = 2.0,
        max_connections: int = 64,
        max_keepalive_connections: int = 32,
        keepalive_expiry_seconds: float = 15.0,
        long_poll_max_connections: int = 32,
        transient_retries: int = 1,
        # 以下为 Sentinel 专用、可选覆盖；不传则读环境变量 / 默认集群。
        sentinels: Optional[str] = None,
        master_name: Optional[str] = None,
        password: Optional[str] = None,
        sentinel_password: Optional[str] = None,
        username: Optional[str] = None,
        sentinel_username: Optional[str] = None,
        key_prefix: Optional[str] = None,
        sentinel_socket_timeout: float = 0.5,
    ) -> None:
        # base_url / auth_token / max_keepalive_connections / keepalive_expiry_seconds
        # 仅为兼容旧签名保留，Sentinel 链路不使用它们。
        self.base_url = base_url
        self.auth_token = auth_token
        self._transient_retries = max(int(transient_retries), 0)

        sentinels_raw = (
            sentinels
            if sentinels is not None
            else os.environ.get("REDIS_SENTINELS", _DEFAULT_SENTINELS)
        )
        self._master_name = (
            master_name
            if master_name is not None
            else os.environ.get("REDIS_MASTER_NAME", _DEFAULT_MASTER_NAME)
        )
        self._password = (
            password
            if password is not None
            else os.environ.get("REDIS_PASSWORD", _DEFAULT_PASSWORD)
        )
        if not self._password:
            raise ValueError("REDIS_PASSWORD 未配置，拒绝使用空密码连接 Redis Sentinel")
        self._username = (
            username
            if username is not None
            else os.environ.get("REDIS_USERNAME", "")
        ).strip() or None
        # Sentinel 鉴权策略（默认不鉴权）：
        # - 多数 Sentinel 部署本身不开密码，因此默认**不向哨兵发送 AUTH**，
        #   避免哨兵无密码时报 "Client sent AUTH but no password is set" / AuthenticationError；
        # - 仅当显式传入 sentinel_password 或设置环境变量 REDIS_SENTINEL_PASSWORD 时，
        #   才对哨兵做鉴权（值为空串同样视为不鉴权）。
        if sentinel_password is not None:
            sentinel_pwd_raw = sentinel_password
        else:
            sentinel_pwd_raw = os.environ.get("REDIS_SENTINEL_PASSWORD", "")
        self._sentinel_password = (sentinel_pwd_raw or "").strip() or None
        self._sentinel_username = (
            sentinel_username
            if sentinel_username is not None
            else os.environ.get("REDIS_SENTINEL_USERNAME", "")
        ).strip() or None
        self._prefix = (
            key_prefix
            if key_prefix is not None
            else os.environ.get("REDIS_KEY_PREFIX", "")
        )

        sentinel_hosts = _parse_sentinels(sentinels_raw)

        # 哨兵连接：socket_timeout 取较短值（探活类调用），并把密码同时下发给
        # 哨兵节点（sentinel_kwargs）和数据节点（顶层 connection_kwargs）。
        common_kwargs: dict[str, Any] = {
            "password": self._password,
            # 部分线上 Redis/Sentinel 版本不支持 RESP3（HELLO 3）。
            # 显式固定 RESP2，避免握手阶段触发 "unknown command HELLO"。
            "protocol": 2,
            "socket_connect_timeout": connect_timeout_seconds,
            "socket_keepalive": True,
        }
        if self._username:
            common_kwargs["username"] = self._username
        sentinel_kwargs: dict[str, Any] = {
            # Sentinel 节点连接同样固定 RESP2，避免 HELLO 3。
            "protocol": 2,
        }
        if self._sentinel_username:
            sentinel_kwargs["username"] = self._sentinel_username
        if self._sentinel_password:
            sentinel_kwargs["password"] = self._sentinel_password
        self._sentinel = Sentinel(
            sentinel_hosts,
            socket_timeout=sentinel_socket_timeout,
            sentinel_kwargs=sentinel_kwargs,
            **common_kwargs,
        )

        # 短请求 master 客户端：socket_timeout 控制单命令读超时，
        # 避免链路抖动时把请求卡满。decode_responses=True → 取回 str，再 json.loads。
        self._client = self._sentinel.master_for(
            self._master_name,
            redis_class=aioredis.Redis,
            db=TARGET_DB,
            decode_responses=True,
            socket_timeout=timeout_seconds,
            max_connections=max_connections,
            health_check_interval=30,
            **common_kwargs,
        )

        # 长轮询 master 客户端：BLPOP 会按调用方传入的 timeout 阻塞，必须让 socket
        # 读超时**大于**阻塞时长，否则 socket 会先于 BLPOP 超时。这里干脆不设
        # socket_timeout（None = 不超时），依赖 socket_keepalive + 健康检查兜底。
        self._blpop_client = self._sentinel.master_for(
            self._master_name,
            redis_class=aioredis.Redis,
            decode_responses=True,
            socket_timeout=None,
            max_connections=long_poll_max_connections,
            health_check_interval=30,
            **common_kwargs,
        )

    async def __aenter__(self) -> "RedisServerClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def close(self) -> None:
        # 关闭两个数据客户端 + 哨兵客户端。aclose 在新版 redis-py 上替代 close。
        for cli in (self._client, self._blpop_client):
            try:
                await cli.aclose()
            except AttributeError:
                await cli.close()  # type: ignore[attr-defined]
            except Exception:
                pass
        # 哨兵实例本身也持有到各 sentinel 节点的连接，需要一并释放。
        try:
            for sent in getattr(self._sentinel, "sentinels", []) or []:
                try:
                    await sent.aclose()
                except AttributeError:
                    await sent.close()  # type: ignore[attr-defined]
                except Exception:
                    pass
        except Exception:
            pass

    # ----------------------------------------------------------- 内部工具

    def _k(self, key: str) -> str:
        """套上可选的全局前缀。默认前缀为空 → 完全沿用旧 key 命名。"""
        return f"{self._prefix}{key}" if self._prefix else key

    @staticmethod
    def _dump(value: Any) -> str:
        """统一 JSON 编码。``sort_keys`` 保证同一对象编码稳定，使 lrem 按值匹配可靠。"""
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _load(raw: Optional[str]) -> Any:
        """统一 JSON 解码；解析失败时退化为返回原始字符串（兼容脏数据）。"""
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return raw

    async def _run(self, op_name: str, coro_factory, *, retries: Optional[int] = None):
        """统一命令出口：捕获瞬时链路错误重试，最终失败包装成 RedisServerError。

        ``coro_factory`` 必须是零参函数（每次调用返回**新的** awaitable），
        以便重试时重新发起命令，而不是二次 await 同一个协程。
        """
        attempts = (self._transient_retries if retries is None else retries) + 1
        last_err: Optional[BaseException] = None
        for attempt in range(attempts):
            try:
                return await coro_factory()
            except _RETRYABLE_REDIS_ERRORS as e:
                last_err = e
                if attempt == attempts - 1:
                    break
                logger.warning(
                    "[RedisServerClient] %s 遇到 %s，第 %d/%d 次重试",
                    op_name, type(e).__name__, attempt + 1, attempts - 1,
                )
                await asyncio.sleep(0.05 * (attempt + 1))
            except redis_exceptions.RedisError as e:
                # 非瞬时的 Redis 错误（命令语义错误等）：直接包装上抛，不重试。
                if "No master found for" in str(e):
                    logger.error(
                        "[RedisServerClient] %s 失败：无法从 Sentinel 发现 master=%s。"
                        "请检查 REDIS_MASTER_NAME 是否正确，及 Sentinel 鉴权（REDIS_SENTINEL_PASSWORD/USERNAME）是否匹配。",
                        op_name,
                        self._master_name,
                    )
                raise RedisServerError(f"{op_name} 失败: {e}") from e
        raise RedisServerError(f"{op_name} 失败（已重试 {attempts - 1} 次）: {last_err}")

    # -------------------------------------------------------------- KV

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[float] = None,
    ) -> None:
        px = int(ttl_seconds * 1000) if ttl_seconds and ttl_seconds > 0 else None
        await self._run(
            f"set {key}",
            lambda: self._client.set(self._k(key), self._dump(value), px=px),
        )

    async def get(self, key: str) -> tuple[Any, bool]:
        raw = await self._run(f"get {key}", lambda: self._client.get(self._k(key)))
        if raw is None:
            return None, False
        return self._load(raw), True

    async def delete(self, key: str) -> bool:
        n = await self._run(f"delete {key}", lambda: self._client.delete(self._k(key)))
        return bool(n)

    async def exists(self, key: str) -> bool:
        n = await self._run(f"exists {key}", lambda: self._client.exists(self._k(key)))
        return bool(n)

    async def expire(self, key: str, ttl_seconds: float) -> bool:
        # 用 pexpire（毫秒）保留亚秒级 TTL 精度，与旧实现的 float 语义一致。
        ms = max(int(ttl_seconds * 1000), 1)
        ok = await self._run(
            f"expire {key}", lambda: self._client.pexpire(self._k(key), ms)
        )
        return bool(ok)

    async def ttl(self, key: str) -> Optional[float]:
        # Redis PTTL: -2 不存在, -1 永不过期, 否则剩余毫秒。
        # 旧语义: None 不存在, -1.0 永不过期, 否则剩余秒。
        ms = await self._run(f"ttl {key}", lambda: self._client.pttl(self._k(key)))
        if ms == -2:
            return None
        if ms == -1:
            return -1.0
        return ms / 1000.0

    async def keys(self, prefix: str = "") -> list[str]:
        # SCAN 非阻塞遍历；返回 key 时剥掉内部全局前缀，保证调用方拿到的是原始 key。
        match = f"{self._k(prefix)}*"

        async def _scan() -> list[str]:
            found: list[str] = []
            async for k in self._client.scan_iter(match=match, count=500):
                found.append(k)
            return found

        keys = await self._run(f"keys {prefix}", _scan)
        if self._prefix:
            plen = len(self._prefix)
            return [k[plen:] if k.startswith(self._prefix) else k for k in keys]
        return keys

    # ------------------------------------------------------------ Queue

    async def rpush(self, queue: str, value: Any) -> int:
        n = await self._run(
            f"rpush {queue}",
            lambda: self._client.rpush(self._k(queue), self._dump(value)),
        )
        return int(n or 0)

    async def lpop(self, queue: str) -> tuple[Any, bool]:
        raw = await self._run(f"lpop {queue}", lambda: self._client.lpop(self._k(queue)))
        if raw is None:
            return None, False
        return self._load(raw), True

    async def blpop(
        self,
        queue: str,
        timeout_seconds: float = 30.0,
    ) -> tuple[Any, bool]:
        """阻塞弹出。``timeout_seconds<=0`` 表示一直阻塞直到有值。

        显式走 ``_blpop_client``：BLPOP 会长期占住一个连接，必须与短请求池分开，
        否则 worker 把池子占满后 set/get/rpush 等接口会全部超时。
        """
        # Redis BLPOP: timeout=0 表示无限阻塞，正好对应旧语义 timeout<=0。
        # 部分 Redis 版本只接受**整数**秒超时（传 float 会报
        # "timeout is not an integer or out of range"），这里统一取整为 int。
        # 正超时至少取 1s，避免 0<raw<1 被取整成 0 而变成永久阻塞。
        raw_timeout = float(timeout_seconds) if timeout_seconds and timeout_seconds > 0 else 0.0
        block = max(1, int(round(raw_timeout))) if raw_timeout > 0 else 0

        result = await self._run(
            f"blpop {queue}",
            lambda: self._blpop_client.blpop([self._k(queue)], timeout=block),
        )
        if not result:
            return None, False
        # 返回 (key, value)；我们只阻塞单个队列，取 value 即可。
        return self._load(result[1]), True

    async def llen(self, queue: str) -> int:
        n = await self._run(f"llen {queue}", lambda: self._client.llen(self._k(queue)))
        return int(n or 0)

    async def lrange(self, queue: str, start: int = 0, stop: int = -1) -> list[Any]:
        items = await self._run(
            f"lrange {queue}",
            lambda: self._client.lrange(self._k(queue), start, stop),
        )
        return [self._load(x) for x in (items or [])]

    async def lrem(self, queue: str, value: Any, count: int = 0) -> int:
        # redis-py 签名为 lrem(name, count, value)。count 语义与旧实现一致：
        # >0 从头删、<0 从尾删、0 删全部。按值匹配依赖 _dump 的稳定编码。
        removed = await self._run(
            f"lrem {queue}",
            lambda: self._client.lrem(self._k(queue), count, self._dump(value)),
        )
        return int(removed or 0)

    # ------------------------------------------------------------ Admin

    async def health(self) -> dict:
        """PING 探活，返回与旧 /health 兼容的结构（含 status 字段）。"""
        await self._run("health(ping)", lambda: self._client.ping(), retries=0)
        return {
            "status": "ok",
            "backend": "redis-sentinel",
            "master_name": self._master_name,
        }

    async def stats(self) -> dict:
        """返回简要统计（key 数量等），兼容旧 stats 调用点（仅冒烟测试使用）。"""
        size = await self._run("stats(dbsize)", lambda: self._client.dbsize())
        return {"kv_count": int(size or 0), "backend": "redis-sentinel"}

    async def snapshot(self) -> dict:
        """旧实现会触发 JSON 快照落盘；Redis 集群自带持久化，这里为 no-op。

        不主动触发 BGSAVE，避免在共享集群上越权或造成额外负载。
        """
        return {"path": None, "note": "persistence managed by redis cluster"}

    async def flushall(self) -> None:
        """清空当前 DB（**危险**：共享集群慎用）。

        为防止误清生产共享集群，默认禁用；需显式设置环境变量
        ``REDIS_ALLOW_FLUSH=1`` 才会真正执行 FLUSHDB。
        """
        if os.environ.get("REDIS_ALLOW_FLUSH", "").strip() not in ("1", "true", "yes", "on"):
            raise RedisServerError(
                "flushall 已被禁用以保护共享 Redis 集群；"
                "如确需清库请设置环境变量 REDIS_ALLOW_FLUSH=1"
            )
        await self._run("flushall", lambda: self._client.flushdb())
