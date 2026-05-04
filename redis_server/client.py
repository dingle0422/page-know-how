"""redis_server 的 Python 异步客户端（httpx）。

提供薄封装，签名与服务端语义一一对应；上层不需要关心 HTTP 细节。
设计上刻意做成极小面，方便 app.py / 推理 worker 直接 import 使用。

示例:
    from redis_server.client import RedisServerClient

    async with RedisServerClient("http://127.0.0.1:6390") as cli:
        await cli.set("task:abc", {"status": "pending"}, ttl_seconds=86400)
        await cli.rpush("queue:reason:pending", "abc")
        value, ok = await cli.lpop("queue:reason:pending")
"""

from __future__ import annotations

from typing import Any, Optional

import httpx


class RedisServerError(RuntimeError):
    """服务端返回非 2xx 或业务 status_code 非 200 时抛出。"""


class RedisServerClient:
    """异步客户端。**内部维护两个独立的 httpx 连接池**：

    - ``_client``：服务于普通短请求（set / get / rpush / lrange / lpop / lrem ...），
      pool 超时较短（默认 2s），让 /api/reason/submit、/api/requestQueueStatus
      等路径在链路抖动时**快速失败**而不是卡 15 秒，便于上层重试或返回 503。
    - ``_long_poll_client``：服务于 ``blpop`` 长轮询（每次最长占用一个连接 ~timeout 秒）。
      worker 池里 N 个 worker 会持续把这个 pool 占住，把它独立出来后**就不会挨饿短请求**。

    背景：之前所有调用共享同一个 client，10 个 worker 长期占着 BLPOP 连接 + 偶发的
    底层 RemoteProtocolError/ReadError 会让 httpcore 池状态错乱，submit/queue_status
    等接口随后开始稳定 ``httpx.PoolTimeout``。拆双 pool + 短 pool timeout 同时根治
    挨饿与卡住两类症状。
    """

    def __init__(
        self,
        base_url: str,
        auth_token: str = "",
        timeout_seconds: float = 10.0,
        connect_timeout_seconds: float = 3.0,
        *,
        pool_timeout_seconds: float = 2.0,
        max_connections: int = 64,
        max_keepalive_connections: int = 32,
        keepalive_expiry_seconds: float = 30.0,
        long_poll_max_connections: int = 32,
    ) -> None:
        # 注意：不走 httpx 的 base_url 机制。httpx 在 base_url 带 path 前缀时，
        # 用户侧传入 "/xxx" 会导致 path 被当成绝对路径、吃掉 base_url 的前缀
        # （例如 base=http://host/redis-server + path=/kv/set -> http://host/kv/set）。
        # 这里改为显式字符串拼接：_full_url = base_url.rstrip("/") + path，
        # path 必须以 "/" 开头，路由统一干净。
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        headers = {"X-Auth-Token": auth_token} if auth_token else {}

        # 短请求 client：set/get/rpush/lrange/... 都走它。
        # pool 超时刻意设短：链路抖出 PoolTimeout 让上层立即知道并重试，
        # 而不是把整条 HTTP 请求卡满 timeout_seconds（默认 10~15s）才返回 500。
        self._timeout = httpx.Timeout(
            timeout=timeout_seconds,
            connect=connect_timeout_seconds,
            pool=pool_timeout_seconds,
        )
        self._client = httpx.AsyncClient(
            headers=headers,
            timeout=self._timeout,
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive_connections,
                keepalive_expiry=keepalive_expiry_seconds,
            ),
        )

        # 长轮询专用 client：BLPOP 调用每次会占一个连接长达 ~blpop_timeout+5s,
        # 单独走一个池子，确保即便 worker 把池子占满，也不会挨饿短请求 client。
        # pool 超时默认放宽（≈ 短请求池超时 + 1 个 BLPOP 周期），
        # 避免 worker 启动瞬间彼此抢同一个连接时偶发地误报 PoolTimeout。
        self._long_poll_timeout = httpx.Timeout(
            timeout=timeout_seconds,
            connect=connect_timeout_seconds,
            pool=max(pool_timeout_seconds, timeout_seconds + 5.0),
        )
        self._long_poll_client = httpx.AsyncClient(
            headers=headers,
            timeout=self._long_poll_timeout,
            limits=httpx.Limits(
                max_connections=long_poll_max_connections,
                # 长轮询连接周转慢，keepalive 池跟总池一样大即可，避免反复重建。
                max_keepalive_connections=long_poll_max_connections,
                keepalive_expiry=keepalive_expiry_seconds,
            ),
        )

    async def __aenter__(self) -> "RedisServerClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def close(self) -> None:
        # 两个 client 都要关。aclose 内部已经做幂等，先后顺序无所谓。
        await self._client.aclose()
        await self._long_poll_client.aclose()

    # ----------------------------------------------------------- 内部工具

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        params: Optional[dict] = None,
        timeout: Optional[float] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> dict:
        """统一的请求出口。

        ``client`` 显式指定走哪个连接池——短请求传 None（默认 ``self._client``）,
        BLPOP 等长轮询调用传 ``self._long_poll_client``，避免它们相互挨饿。
        """
        target_client = client if client is not None else self._client
        # 自定义 timeout 时，pool / connect 字段沿用所属 client 的设置，
        # 避免一刀切超时把网关延迟正常的请求误判为失败。
        base_timeout = (
            self._long_poll_timeout
            if target_client is self._long_poll_client
            else self._timeout
        )
        kwargs: dict[str, Any] = {}
        if json is not None:
            kwargs["json"] = json
        if params is not None:
            kwargs["params"] = params
        if timeout is not None:
            kwargs["timeout"] = httpx.Timeout(
                timeout=timeout,
                connect=base_timeout.connect,
                pool=base_timeout.pool,
            )
        url = self.base_url + path
        resp = await target_client.request(method, url, **kwargs)
        if resp.status_code >= 400:
            raise RedisServerError(
                f"{method} {url} http={resp.status_code} body={resp.text[:500]}"
            )
        payload = resp.json()
        if payload.get("status_code", 200) != 200:
            raise RedisServerError(
                f"{method} {url} biz_status={payload.get('status_code')} "
                f"msg={payload.get('message')}"
            )
        return payload.get("data") or {}

    # -------------------------------------------------------------- KV

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[float] = None,
    ) -> None:
        await self._request(
            "POST", "/kv/set",
            json={"key": key, "value": value, "ttl_seconds": ttl_seconds},
        )

    async def get(self, key: str) -> tuple[Any, bool]:
        data = await self._request("GET", f"/kv/get/{key}")
        return data.get("value"), bool(data.get("exists"))

    async def delete(self, key: str) -> bool:
        data = await self._request("DELETE", f"/kv/del/{key}")
        return bool(data.get("deleted"))

    async def exists(self, key: str) -> bool:
        data = await self._request("GET", f"/kv/exists/{key}")
        return bool(data.get("exists"))

    async def expire(self, key: str, ttl_seconds: float) -> bool:
        data = await self._request(
            "POST", "/kv/expire",
            json={"key": key, "ttl_seconds": ttl_seconds},
        )
        return bool(data.get("applied"))

    async def ttl(self, key: str) -> Optional[float]:
        data = await self._request("GET", f"/kv/ttl/{key}")
        return data.get("ttl_seconds")

    async def keys(self, prefix: str = "") -> list[str]:
        data = await self._request("GET", "/kv/keys", params={"prefix": prefix})
        return data.get("keys") or []

    # ------------------------------------------------------------ Queue

    async def rpush(self, queue: str, value: Any) -> int:
        data = await self._request(
            "POST", "/queue/rpush",
            json={"queue": queue, "value": value},
        )
        return int(data.get("length") or 0)

    async def lpop(self, queue: str) -> tuple[Any, bool]:
        data = await self._request(
            "POST", "/queue/lpop",
            json={"queue": queue},
        )
        return data.get("value"), bool(data.get("ok"))

    async def blpop(
        self,
        queue: str,
        timeout_seconds: float = 30.0,
    ) -> tuple[Any, bool]:
        """阻塞弹出。

        客户端 HTTP 超时在服务端阻塞时长之上 + 5s 冗余，
        避免因 client 超时导致服务端仍在 wait 的假失败。

        显式走 ``_long_poll_client``：BLPOP 会长期占住一个连接，必须与短请求池
        分开，否则 worker 把池子占满后 set/get/rpush 等接口会全部 PoolTimeout。
        """
        http_timeout = max(5.0, float(timeout_seconds)) + 5.0
        data = await self._request(
            "POST", "/queue/blpop",
            json={"queue": queue, "timeout_seconds": timeout_seconds},
            timeout=http_timeout,
            client=self._long_poll_client,
        )
        return data.get("value"), bool(data.get("ok"))

    async def llen(self, queue: str) -> int:
        data = await self._request("GET", f"/queue/llen/{queue}")
        return int(data.get("length") or 0)

    async def lrange(self, queue: str, start: int = 0, stop: int = -1) -> list[Any]:
        data = await self._request(
            "GET", f"/queue/lrange/{queue}",
            params={"start": start, "stop": stop},
        )
        return data.get("items") or []

    async def lrem(self, queue: str, value: Any, count: int = 0) -> int:
        data = await self._request(
            "POST", "/queue/lrem",
            json={"queue": queue, "value": value, "count": count},
        )
        return int(data.get("removed") or 0)

    # ------------------------------------------------------------ Admin

    async def health(self) -> dict:
        resp = await self._client.get(self.base_url + "/health")
        resp.raise_for_status()
        return resp.json()

    async def stats(self) -> dict:
        return await self._request("GET", "/admin/stats")

    async def snapshot(self) -> dict:
        return await self._request("POST", "/admin/snapshot")

    async def flushall(self) -> None:
        await self._request("POST", "/admin/flushall")
