"""redis_server 冒烟测试。

依赖服务已在 127.0.0.1:6390 启动。运行：
    python -m redis_server.smoke_test
"""

from __future__ import annotations

import asyncio
import time

from redis_server.client import RedisServerClient


async def main() -> None:
    async with RedisServerClient("http://127.0.0.1:6390") as cli:
        print("--- health ---")
        print(await cli.health())

        print("\n--- KV 基础 ---")
        await cli.set("hello", "world")
        v, ok = await cli.get("hello")
        assert ok and v == "world", (v, ok)
        print("  set/get ok")

        await cli.set("obj", {"a": 1, "b": [1, 2, 3]}, ttl_seconds=2)
        v, ok = await cli.get("obj")
        assert ok and v == {"a": 1, "b": [1, 2, 3]}
        print("  nested json ok, ttl=", await cli.ttl("obj"))

        print("  等待 2.5s 让 obj TTL 过期…")
        await asyncio.sleep(2.5)
        v, ok = await cli.get("obj")
        assert not ok, f"obj 应该已过期，但拿到 {v}"
        print("  ttl 过期回收 ok")

        print("\n--- Queue 基础 ---")
        await cli.rpush("qa", "t1")
        await cli.rpush("qa", "t2")
        await cli.rpush("qa", {"id": "t3"})
        assert await cli.llen("qa") == 3
        items = await cli.lrange("qa")
        print("  lrange:", items)
        v, ok = await cli.lpop("qa")
        assert ok and v == "t1"
        v, ok = await cli.lpop("qa")
        assert ok and v == "t2"
        v, ok = await cli.lpop("qa")
        assert ok and v == {"id": "t3"}
        v, ok = await cli.lpop("qa")
        assert not ok
        print("  rpush/lpop FIFO ok")

        print("\n--- BLPOP 阻塞+生产者唤醒 ---")

        async def producer():
            await asyncio.sleep(0.5)
            n = await cli.rpush("qb", {"task_id": "T-1"})
            print(f"  producer pushed, length={n}")

        t0 = time.time()
        pt = asyncio.create_task(producer())
        v, ok = await cli.blpop("qb", timeout_seconds=5)
        elapsed = time.time() - t0
        await pt
        assert ok and v == {"task_id": "T-1"}, (v, ok)
        assert 0.3 < elapsed < 2.0, f"唤醒不在合理时间内: {elapsed:.2f}s"
        print(f"  blpop 唤醒耗时 {elapsed:.2f}s ok")

        print("\n--- BLPOP 超时 ---")
        t0 = time.time()
        v, ok = await cli.blpop("empty-queue", timeout_seconds=1)
        elapsed = time.time() - t0
        assert not ok
        assert 0.8 < elapsed < 3.0, f"超时时间异常: {elapsed:.2f}s"
        print(f"  blpop 超时 {elapsed:.2f}s ok")

        print("\n--- 主动 snapshot ---")
        print(" ", await cli.snapshot())

        print("\n--- stats ---")
        print(" ", await cli.stats())

        print("\n全部冒烟通过")


if __name__ == "__main__":
    asyncio.run(main())
