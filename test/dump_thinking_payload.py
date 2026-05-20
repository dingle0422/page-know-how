"""一次性脚本：dump 修改后两个调用点对 servyou + deepseek-v3.2 的 PAYLOAD。

不真正发请求，只跑 vendor 路由逻辑，方便人工核对：
- 顶层是否含 ``enable_thinking: true``（servyou 私有协议）
- 是否还残留 ``chat_template_kwargs.enable_thinking``（不应有）

同时对比 enable_thinking=False / 其他 vendor 的 payload，确保改动没碰到非目标分支。
"""

from __future__ import annotations

import json
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _dump_client_chat_payload(vendor: str, model: str, enable_thinking: bool) -> dict:
    """直接复刻 llm/client.py:chat 的 payload 组装逻辑（不发请求）。

    这里没办法直接 import chat() 后拦截 requests.post——会真的发出去。
    用 monkeypatch 把 requests.post 替换成一个抓 payload 的 stub。
    """

    import llm.client as client

    captured: dict = {}

    class _StubResp:
        def json(self_inner):
            return {
                "choices": [{"message": {"role": "assistant", "content": "stub"}}]
            }

    def _stub_post(url, data=None, headers=None, timeout=None, **kwargs):
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = json.loads(data) if isinstance(data, (str, bytes)) else data
        return _StubResp()

    orig_post = client.requests.post
    client.requests.post = _stub_post
    try:
        client.chat(
            messages="hello",
            vendor=vendor,
            model=model,
            system=None,
            enable_thinking=enable_thinking,
        )
    finally:
        client.requests.post = orig_post

    return captured


def _dump_stream_request(vendor: str, model: str, enable_thinking: bool) -> dict:
    """直接调 inference/llm_stream.py:_build_request 拿 (url, headers, payload)。"""

    from inference.llm_stream import _build_request

    messages_payload = [{"role": "user", "content": "hello"}]
    url, headers, payload = _build_request(
        messages_payload,
        vendor=vendor,
        model=model,
        enable_thinking=enable_thinking,
    )
    return {"url": url, "headers": headers, "payload": payload}


def _print_section(title: str, info: dict) -> None:
    print("=" * 80)
    print(f"# {title}")
    print("-" * 80)
    print(f"URL    : {info.get('url')}")
    print(f"HEADERS: {info.get('headers')}")
    print(f"PAYLOAD:")
    print(json.dumps(info.get("payload"), ensure_ascii=False, indent=2))


def main() -> int:
    cases = [
        # 主目标：servyou + deepseek-v3.2 + thinking ON
        ("servyou", "deepseek-v3.2-1163259bcc6c", True),
        # 对照：同 vendor 关 thinking，确认无副作用
        ("servyou", "deepseek-v3.2-1163259bcc6c", False),
        # 对照：其他走 mudgate 的 vendor 不应受影响
        ("deepseek-v4-pro", "deepseek-v4-pro", True),
        # 对照：qwen 系列继续走 chat_template_kwargs
        ("qwen3.6-35b-a3b", "Qwen/Qwen3.6-35B-A3B", True),
    ]

    for vendor, model, et in cases:
        title = f"client.chat   vendor={vendor} model={model} enable_thinking={et}"
        _print_section(title, _dump_client_chat_payload(vendor, model, et))
        title = f"llm_stream    vendor={vendor} model={model} enable_thinking={et}"
        _print_section(title, _dump_stream_request(vendor, model, et))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
