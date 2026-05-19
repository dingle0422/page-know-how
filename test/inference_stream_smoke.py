"""POST /api/inference/stream 的轻量 smoke 脚本。

功能：

- 对一个已抽取过的 policy 发请求，订阅 SSE，按事件落到 stdout；
- 支持 ``--reconnect`` 在中途断流后凭同一 ``taskId`` 重新订阅，验证后台不死、可续接；
- 默认仅打印 think/answer 长度 + status 增量，加 ``--verbose`` 输出完整快照内容。

用法示例（PowerShell）::

    python test/inference_stream_smoke.py \
        --url http://127.0.0.1:5000/api/inference/stream \
        --policy-id 12345 \
        --question "请问蔬菜流通环节增值税如何处理？" \
        --reconnect-after 3.0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Iterator
from uuid import uuid4

import requests


def _iter_sse(resp: requests.Response) -> Iterator[tuple[str, dict[str, Any]]]:
    """SSE 行解析器：按 ``event:`` / ``data:`` 块输出 ``(event, payload_dict)``。"""

    event = "message"
    data_lines: list[str] = []
    for raw in resp.iter_lines(decode_unicode=True):
        if raw is None:
            continue
        if raw == "":
            if data_lines:
                payload_text = "\n".join(data_lines)
                try:
                    payload = json.loads(payload_text)
                except json.JSONDecodeError:
                    payload = {"_raw": payload_text}
                yield event, payload
                event = "message"
                data_lines = []
            continue
        if raw.startswith(":"):
            continue
        if raw.startswith("event:"):
            event = raw[len("event:"):].strip()
        elif raw.startswith("data:"):
            data_lines.append(raw[len("data:"):].lstrip())
    if data_lines:
        payload_text = "\n".join(data_lines)
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            payload = {"_raw": payload_text}
        yield event, payload


def _print_event(event: str, payload: dict, verbose: bool) -> None:
    if event != "snapshot":
        print(f"[event={event}] {payload}")
        return
    think = payload.get("think") or ""
    answer = payload.get("answer") or ""
    if verbose:
        print(
            f"[snapshot] status={payload.get('status')} "
            f"round={payload.get('reactRound')} previewDone={payload.get('previewDone')}\n"
            f"--- think ({len(think)}) ---\n{think}\n"
            f"--- answer ({len(answer)}) ---\n{answer}\n"
        )
    else:
        print(
            f"[snapshot] status={payload.get('status')} "
            f"round={payload.get('reactRound')} "
            f"think_len={len(think)} answer_len={len(answer)}"
        )
    if payload.get("error"):
        print(f"  error={payload['error']}")


def _stream_once(
    url: str,
    body: dict,
    *,
    timeout: float,
    verbose: bool,
    abort_after: float | None,
) -> tuple[str | None, bool]:
    """发起一次 SSE 订阅，返回 ``(taskId, finished_normally)``。"""

    start = time.time()
    task_id: str | None = body.get("taskId")
    finished_normally = False
    print(f"[connect] POST {url} (taskId={task_id or '<new>'})")
    with requests.post(url, json=body, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        for event, payload in _iter_sse(resp):
            if isinstance(payload, dict) and not task_id:
                task_id = payload.get("taskId")
            _print_event(event, payload, verbose)
            if event == "snapshot" and payload.get("status") in {"done", "failed"}:
                finished_normally = True
                # 服务端会自己再推一帧并断流，这里继续读到 EOF 为止
            if abort_after is not None and (time.time() - start) >= abort_after:
                print(f"[abort] 主动断开（已耗时 {time.time() - start:.2f}s）")
                return task_id, False
    return task_id, finished_normally


def main() -> int:
    parser = argparse.ArgumentParser(description="inference SSE smoke test")
    parser.add_argument(
        "--url", default="http://127.0.0.1:5000/api/inference/stream",
        help="SSE 接口地址",
    )
    parser.add_argument("--policy-id", required=True, help="已抽取过的 policyId")
    parser.add_argument("--question", required=True, help="测试问题")
    parser.add_argument("--task-id", default=None, help="复用 taskId（不传则随机生成）")
    parser.add_argument("--vendor", default="servyou")
    parser.add_argument("--model", default="deepseek-v3.2-1163259bcc6c")
    parser.add_argument(
        "--reconnect-after", type=float, default=None,
        help="在 N 秒后主动断流并凭同一 taskId 重连，验证后台不死",
    )
    parser.add_argument(
        "--intermediate-think", choices=["on", "off"], default=None,
        help="覆盖 INFERENCE_REACT_INTERMEDIATE_THINK；不传则跟随服务端默认",
    )
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--top-m", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--verbose", action="store_true", help="打印完整 think/answer 内容")
    args = parser.parse_args()

    task_id = args.task_id or str(uuid4())
    body: dict[str, Any] = {
        "taskId": task_id,
        "policyId": args.policy_id,
        "question": args.question,
        "vendor": args.vendor,
        "model": args.model,
        "topN": args.top_n,
        "topM": args.top_m,
    }
    if args.intermediate_think is not None:
        body["intermediateThinkEnabled"] = args.intermediate_think == "on"

    tid, ok = _stream_once(
        args.url, body,
        timeout=args.timeout, verbose=args.verbose,
        abort_after=args.reconnect_after,
    )
    if args.reconnect_after is not None and tid:
        print(f"\n[reconnect] 等待 1.0s 后用同 taskId={tid} 重连...")
        time.sleep(1.0)
        body["taskId"] = tid
        _, ok = _stream_once(
            args.url, body,
            timeout=args.timeout, verbose=args.verbose,
            abort_after=None,
        )
    print(f"[summary] taskId={tid} finished={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
