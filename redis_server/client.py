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
    def __init__(
        self,
        base_url: str,
        auth_token: str = "",
        timeout_seconds: float = 10.0,
        connect_timeout_seconds: float = 3.0,
    ) -> None:
        # 注意：不走 httpx 的 base_url 机制。httpx 在 base_url 带 path 前缀时，
        # 用户侧传入 "/xxx" 会导致 path 被当成绝对路径、吃掉 base_url 的前缀
        # （例如 base=http://host/redis-server + path=/kv/set -> http://host/kv/set）。
        # 这里改为显式字符串拼接：_full_url = base_url.rstrip("/") + path，
        # path 必须以 "/" 开头，路由统一干净。
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        headers = {"X-Auth-Token": auth_token} if auth_token else {}
        self._timeout = httpx.Timeout(
            timeout=timeout_seconds,
            connect=connect_timeout_seconds,
        )
        self._client = httpx.AsyncClient(
            headers=headers,
            timeout=self._timeout,
        )

    async def __aenter__(self) -> "RedisServerClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    # ----------------------------------------------------------- 内部工具

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        params: Optional[dict] = None,
        timeout: Optional[float] = None,
    ) -> dict:
        kwargs: dict[str, Any] = {}
        if json is not None:
            kwargs["json"] = json
        if params is not None:
            kwargs["params"] = params
        if timeout is not None:
            kwargs["timeout"] = httpx.Timeout(timeout=timeout, connect=self._timeout.connect)
        url = self.base_url + path
        resp = await self._client.request(method, url, **kwargs)
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
        """
        http_timeout = max(5.0, float(timeout_seconds)) + 5.0
        data = await self._request(
            "POST", "/queue/blpop",
            json={"queue": queue, "timeout_seconds": timeout_seconds},
            timeout=http_timeout,
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
