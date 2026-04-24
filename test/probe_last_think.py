"""探测：各 vendor / model 组合对 enable_thinking=True 的实际支持情况。

用法：
    python test/probe_last_think.py                     # 跑默认 vendor 矩阵
    python test/probe_last_think.py --vendor servyou --model deepseek-v3.2
    python test/probe_last_think.py --timeout 60

输出每个 vendor 的：
    - 请求是否 HTTP 200 成功
    - 响应 content 是否含 <think>...</think>
    - 原始 message.reasoning_content 是否非空
    - 前 200 字 preview（用于人工核对）

运行前请确保能访问：
    - qwen3.5-122b-a10b: http://211.137.21.19:17860
    - qwen3.6-35b-a3b / qwen3.5-27b: http://mlp.paas.dc.servyou-it.com
    - mudgate / servyou: 对应内网

注意：本脚本绕过了 llm/client.py 的 content 归一化逻辑，直接打印原始响应，
方便看清不同 vendor 到底把 think 放在 content 还是 reasoning_content。
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import logging
import dataclasses
from typing import Any

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger("probe_last_think")

PROMPT = (
    "请逐步推理后回答：一个农产品批发企业销售自产的圣女果给商超，"
    "是否需要缴纳增值税？用一句话给出结论。"
)

# 默认矩阵：可以命令行 --vendor/--model 覆盖单测
DEFAULT_MATRIX: list[dict[str, str]] = [
    {"vendor": "qwen3.5-122b-a10b", "model": "Qwen3.5-122B-A10B"},
    {"vendor": "qwen3.6-35b-a3b", "model": "Qwen/Qwen3.6-35B-A3B"},
    {"vendor": "qwen3.5-27b", "model": "Qwen/Qwen3.5-27B"},
    # mudgate 下常见模型；如失败请把 vendor/model 改成环境中真实可用的 slug。
    {"vendor": "servyou", "model": "deepseek-v3.2"},
    {"vendor": "deepseek-v3.2", "model": "deepseek-v3.2"},
    {"vendor": "aliyun", "model": "qwen-plus"},
]


@dataclasses.dataclass
class Probe:
    vendor: str
    model: str
    ok: bool = False
    http_status: int | None = None
    has_think_tag: bool = False
    reasoning_content_len: int = 0
    content_len: int = 0
    elapsed_ms: int = 0
    error: str = ""
    content_preview: str = ""
    reasoning_preview: str = ""


def _build_request(vendor: str, model: str, prompt: str, enable_thinking: bool) -> tuple[str, dict, dict]:
    """复刻 llm/client.py 的路由逻辑，但返回 URL / headers / payload 三元组，方便调试。"""
    messages = [{"role": "user", "content": prompt}]

    if vendor == "qwen3.5-122b-a10b":
        url = "http://211.137.21.19:17860/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": "Qwen3.5-122B-A10B",
            "messages": messages,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
    elif vendor == "qwen3.6-35b-a3b":
        url = "http://mlp.paas.dc.servyou-it.com/qwen3.6-35b-a3b/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": "Qwen/Qwen3.6-35B-A3B",
            "messages": messages,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
    elif vendor == "qwen3.5-27b":
        url = "http://mlp.paas.dc.servyou-it.com/qwen3.5-27b/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": "Qwen/Qwen3.5-27B",
            "messages": messages,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
    else:
        if vendor == "servyou":
            url = f"http://10.199.0.7:5000/api/llm/{vendor}/v1/chat/completions"
            app_id = "sk-d75b519b704d4d348245efe435f08ff3"
        else:
            url = f"http://mlp.paas.dc.servyou-it.com/mudgate/api/llm/{vendor}/v1/chat/completions"
            app_id = "sk-0609aa6d08de4413a72e14b3fb8fbab1"
        headers = {"Content-Type": "application/json", "Authorization": app_id}
        payload = {
            "appId": app_id,
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if enable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": True}

    return url, headers, payload


def _probe_one(vendor: str, model: str, prompt: str, timeout: float) -> Probe:
    p = Probe(vendor=vendor, model=model)
    try:
        url, headers, payload = _build_request(vendor, model, prompt, enable_thinking=True)
    except Exception as e:
        p.error = f"payload 构造失败: {e!r}"
        return p

    t0 = time.time()
    try:
        resp = requests.post(url, data=json.dumps(payload), headers=headers, timeout=timeout)
        p.http_status = resp.status_code
        data = resp.json()
    except Exception as e:
        p.elapsed_ms = int((time.time() - t0) * 1000)
        p.error = f"请求/解析失败: {e!r}"
        return p
    p.elapsed_ms = int((time.time() - t0) * 1000)

    # 服务端业务错误
    if isinstance(data, dict) and "success" in data and data.get("success") is False:
        p.error = f"服务端业务错误: {data.get('errorContext') or data}"
        return p

    try:
        msg: dict[str, Any] = data["choices"][0]["message"]
    except Exception:
        p.error = f"非 OpenAI 兼容响应: {str(data)[:200]}"
        return p

    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    p.content_len = len(content)
    p.reasoning_content_len = len(reasoning)
    p.has_think_tag = "<think>" in content
    p.content_preview = content[:200].replace("\n", "\\n")
    p.reasoning_preview = reasoning[:200].replace("\n", "\\n") if reasoning else ""
    p.ok = True
    return p


def _format_table(results: list[Probe]) -> str:
    headers = ["vendor", "model", "ok", "http", "ms",
               "content_len", "<think>tag",
               "reasoning_len"]
    widths = [24, 26, 4, 5, 6, 12, 11, 14]
    lines: list[str] = []
    row = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    lines.append(row)
    lines.append("-+-".join("-" * w for w in widths))
    for p in results:
        row = " | ".join([
            p.vendor.ljust(widths[0])[:widths[0]],
            p.model.ljust(widths[1])[:widths[1]],
            ("Y" if p.ok else "N").ljust(widths[2]),
            (str(p.http_status) if p.http_status is not None else "-").ljust(widths[3]),
            str(p.elapsed_ms).ljust(widths[4]),
            str(p.content_len).ljust(widths[5]),
            ("Y" if p.has_think_tag else "N").ljust(widths[6]),
            str(p.reasoning_content_len).ljust(widths[7]),
        ])
        lines.append(row)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vendor", default=None, help="只测指定 vendor（与 --model 搭配）")
    parser.add_argument("--model", default=None, help="只测指定 model")
    parser.add_argument("--timeout", type=float, default=120.0, help="单次 HTTP 请求超时秒数，默认 120")
    parser.add_argument("--prompt", default=PROMPT, help="测试用 prompt（默认使用内置税务问题）")
    parser.add_argument(
        "--disable-think", action="store_true",
        help="关闭 enable_thinking（用于与开启时对照，观察 content 是否变化）"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.vendor and args.model:
        matrix = [{"vendor": args.vendor, "model": args.model}]
    else:
        matrix = DEFAULT_MATRIX

    results: list[Probe] = []
    for item in matrix:
        v, m = item["vendor"], item["model"]
        logger.info(f"-> probing vendor={v} model={m} enable_thinking={not args.disable_think}")
        if args.disable_think:
            # 关掉 thinking 作对照
            try:
                url, headers, payload = _build_request(v, m, args.prompt, enable_thinking=False)
                t0 = time.time()
                resp = requests.post(url, data=json.dumps(payload), headers=headers, timeout=args.timeout)
                data = resp.json()
                msg = data["choices"][0]["message"]
                content = msg.get("content") or ""
                reasoning = msg.get("reasoning_content") or ""
                p = Probe(
                    vendor=v, model=m, ok=True, http_status=resp.status_code,
                    content_len=len(content), reasoning_content_len=len(reasoning),
                    has_think_tag="<think>" in content,
                    elapsed_ms=int((time.time() - t0) * 1000),
                    content_preview=content[:200].replace("\n", "\\n"),
                    reasoning_preview=(reasoning[:200].replace("\n", "\\n") if reasoning else ""),
                )
            except Exception as e:
                p = Probe(vendor=v, model=m, error=repr(e))
        else:
            p = _probe_one(v, m, args.prompt, args.timeout)
        results.append(p)
        logger.info(
            f"   ok={p.ok} http={p.http_status} ms={p.elapsed_ms} "
            f"content_len={p.content_len} <think>={p.has_think_tag} "
            f"reasoning_len={p.reasoning_content_len} err={p.error[:80] if p.error else '-'}"
        )

    print("\n=== 汇总表 (enable_thinking=%s) ===" % (not args.disable_think))
    print(_format_table(results))

    print("\n=== 详细 preview ===")
    for p in results:
        print(f"\n--- {p.vendor} / {p.model} ---")
        if p.error:
            print(f"ERROR: {p.error}")
            continue
        print(f"content[:200]      : {p.content_preview}")
        if p.reasoning_preview:
            print(f"reasoning_content  : {p.reasoning_preview}")
        else:
            print("reasoning_content  : <空>")


if __name__ == "__main__":
    main()
