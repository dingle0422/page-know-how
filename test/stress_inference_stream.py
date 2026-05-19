"""POST /api/inference/stream 接口并发吞吐量压测。

对指定 URL 以固定入参发起 N 路真并发 SSE 订阅，逐条记录入参、输出、总耗时、
大模型首字响应时间（TTFT，SSE 快照中 think/answer 首次出现非空内容），
并按并发档位（默认 16 / 32 / 64 / 128）分别写入 test/ 目录下的 xlsx 文件。

用法示例（PowerShell）::

    python test/stress_inference_stream.py

    python test/stress_inference_stream.py `
        --url http://mlp.paas.dc.servyou-it.com/kh-deepthink/api/inference/stream `
        --concurrency 16 32 `
        --timeout 900
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterator
from uuid import uuid4

import pandas as pd
import requests

DEFAULT_URL = "http://mlp.paas.dc.servyou-it.com/kh-deepthink/api/inference/stream"
DEFAULT_PAYLOAD: dict[str, Any] = {
    "policyId": "KH1493204307733168128_20260519101916",
    "question": "咸鸭蛋、松花蛋在开具发票时是否可以享受免税政策？",
}
DEFAULT_CONCURRENCY = (16, 32, 64, 128)

_print_lock = threading.Lock()


def safe_print(*args: Any, **kwargs: Any) -> None:
    with _print_lock:
        print(*args, **kwargs, flush=True)


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
                yield event, payload if isinstance(payload, dict) else {"_raw": payload}
                event = "message"
                data_lines = []
            continue
        if raw.startswith(":"):
            continue
        if raw.startswith("event:"):
            event = raw[len("event:") :].strip()
        elif raw.startswith("data:"):
            data_lines.append(raw[len("data:") :].lstrip())
    if data_lines:
        payload_text = "\n".join(data_lines)
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            payload = {"_raw": payload_text}
        yield event, payload if isinstance(payload, dict) else {"_raw": payload}


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


@dataclass
class StreamRequestResult:
    """单路并发请求的完整压测记录。"""

    concurrency: int
    req_idx: int
    task_id: str
    policy_id: str
    question: str
    input_json: str
    http_status: int | None
    final_status: str | None
    preview_done: bool | None
    react_round: int | None
    answerable: bool | None
    think: str
    answer: str
    error: str | None
    ok: bool
    elapsed_sec: float
    http_connected_sec: float | None
    first_snapshot_sec: float | None
    ttft_sec: float | None
    started_at: str
    finished_at: str
    snapshot_count: int = 0


@dataclass
class ConcurrencyRun:
    concurrency: int
    started_at: float
    finished_at: float
    results: list[StreamRequestResult] = field(default_factory=list)

    @property
    def wall_time(self) -> float:
        return self.finished_at - self.started_at


def _record_ttft(
    t0: float,
    think: str,
    answer: str,
    ttft_sec: float | None,
) -> float | None:
    """首字时间：SSE 快照中 think/answer 首次出现非空内容（含 preview / ReAct）。"""

    if ttft_sec is not None:
        return ttft_sec
    if (think and think.strip()) or (answer and answer.strip()):
        return time.perf_counter() - t0
    return None


def do_stream_request(
    url: str,
    base_payload: dict[str, Any],
    concurrency: int,
    req_idx: int,
    timeout: float,
    verbose: bool,
) -> StreamRequestResult:
    task_id = str(uuid4())
    body = {**base_payload, "taskId": task_id}
    input_json = json.dumps(body, ensure_ascii=False)
    policy_id = str(body.get("policyId") or "")
    question = str(body.get("question") or "")
    started_dt = datetime.now()
    t0 = time.perf_counter()

    http_status: int | None = None
    final_status: str | None = None
    preview_done: bool | None = None
    react_round: int | None = None
    answerable: bool | None = None
    think = ""
    answer = ""
    error: str | None = None
    ok = False
    snapshot_count = 0
    http_connected_sec: float | None = None
    first_snapshot_sec: float | None = None
    ttft_sec: float | None = None

    try:
        with requests.post(url, json=body, stream=True, timeout=timeout) as resp:
            http_connected_sec = time.perf_counter() - t0
            http_status = resp.status_code
            if http_status != 200:
                error = f"HTTP {http_status}: {resp.text[:500]}"
            else:
                for event, payload in _iter_sse(resp):
                    if event != "snapshot" or not isinstance(payload, dict):
                        continue
                    if first_snapshot_sec is None:
                        first_snapshot_sec = time.perf_counter() - t0
                    snapshot_count += 1
                    if payload.get("taskId"):
                        task_id = str(payload["taskId"])
                    new_think = str(payload.get("think") or "")
                    new_answer = str(payload.get("answer") or "")
                    ttft_sec = _record_ttft(t0, new_think, new_answer, ttft_sec)
                    think = new_think or think
                    answer = new_answer or answer
                    if "previewDone" in payload:
                        preview_done = bool(payload["previewDone"])
                    if payload.get("reactRound") is not None:
                        react_round = int(payload["reactRound"])
                    if "answerable" in payload:
                        ans = payload["answerable"]
                        answerable = bool(ans) if ans is not None else None
                    st = payload.get("status")
                    if st:
                        final_status = str(st)
                    if payload.get("error"):
                        error = str(payload["error"])
                    if final_status in {"done", "failed"}:
                        ok = final_status == "done" and not error
                        break
    except requests.RequestException as e:
        error = f"{type(e).__name__}: {e}"
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"

    elapsed = time.perf_counter() - t0
    finished_dt = datetime.now()

    if http_status == 200 and final_status is None and not error:
        error = "流结束但未收到 status=done/failed 的 snapshot"
        ok = False
    elif http_status == 200 and final_status == "done" and not error:
        ok = True

    result = StreamRequestResult(
        concurrency=concurrency,
        req_idx=req_idx,
        task_id=task_id,
        policy_id=policy_id,
        question=question,
        input_json=input_json,
        http_status=http_status,
        final_status=final_status,
        preview_done=preview_done,
        react_round=react_round,
        answerable=answerable,
        think=think,
        answer=answer,
        error=error,
        ok=ok,
        elapsed_sec=elapsed,
        http_connected_sec=http_connected_sec,
        first_snapshot_sec=first_snapshot_sec,
        ttft_sec=ttft_sec,
        started_at=started_dt.isoformat(timespec="seconds"),
        finished_at=finished_dt.isoformat(timespec="seconds"),
        snapshot_count=snapshot_count,
    )

    if verbose:
        tag = "OK " if ok else "ERR"
        ttft_disp = f"{ttft_sec:.2f}s" if ttft_sec is not None else "n/a"
        safe_print(
            f"[c={concurrency} req {req_idx:>4}] {tag} taskId={task_id[:8]}... "
            f"http={http_status} status={final_status} "
            f"ttft={ttft_disp} elapsed={elapsed:.2f}s "
            f"think_len={len(think)} answer_len={len(answer)}"
            + (f" err={error}" if error else "")
        )

    return result


def run_concurrency_level(
    url: str,
    base_payload: dict[str, Any],
    concurrency: int,
    timeout: float,
    verbose: bool,
) -> ConcurrencyRun:
    safe_print(f"\n>>> 并发档位 {concurrency}：同时发起 {concurrency} 个 stream 请求 ...")
    started_at = time.perf_counter()
    results: list[StreamRequestResult] = []

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                do_stream_request, url, base_payload, concurrency, i + 1, timeout, verbose,
            )
            for i in range(concurrency)
        ]
        for fut in as_completed(futures):
            results.append(fut.result())

    finished_at = time.perf_counter()
    results.sort(key=lambda r: r.req_idx)
    run = ConcurrencyRun(
        concurrency=concurrency,
        started_at=started_at,
        finished_at=finished_at,
        results=results,
    )
    _print_run_summary(run)
    return run


def _print_run_summary(run: ConcurrencyRun) -> None:
    n = len(run.results)
    succ = sum(1 for r in run.results if r.ok)
    elapses = [r.elapsed_sec for r in run.results]
    safe_print(
        f"    完成: 总数={n} 成功={succ} 失败={n - succ} "
        f"墙钟={run.wall_time:.2f}s QPS={n / run.wall_time:.3f}"
        if run.wall_time > 0
        else f"    完成: 总数={n} 成功={succ} 失败={n - succ}"
    )
    if elapses:
        safe_print(
            f"    总耗时: min={min(elapses):.2f}s avg={statistics.mean(elapses):.2f}s "
            f"p50={percentile(elapses, 50):.2f}s p95={percentile(elapses, 95):.2f}s "
            f"max={max(elapses):.2f}s"
        )
    ttfts = [r.ttft_sec for r in run.results if r.ttft_sec is not None]
    if ttfts:
        safe_print(
            f"    首字TTFT: min={min(ttfts):.2f}s avg={statistics.mean(ttfts):.2f}s "
            f"p50={percentile(ttfts, 50):.2f}s p95={percentile(ttfts, 95):.2f}s "
            f"max={max(ttfts):.2f}s"
        )


def _build_detail_rows(run: ConcurrencyRun) -> list[dict[str, Any]]:
    """每条并发一行，字段覆盖入参、时序、状态与完整输出。"""
    rows: list[dict[str, Any]] = []
    for r in run.results:
        rows.append(
            {
                "concurrency": r.concurrency,
                "req_idx": r.req_idx,
                "task_id": r.task_id,
                "policy_id": r.policy_id,
                "question": r.question,
                "input_json": r.input_json,
                "http_status": r.http_status,
                "final_status": r.final_status,
                "preview_done": r.preview_done,
                "react_round": r.react_round,
                "answerable": r.answerable,
                "ok": r.ok,
                "error": r.error,
                "http_connected_sec": round(r.http_connected_sec, 4)
                if r.http_connected_sec is not None
                else None,
                "first_snapshot_sec": round(r.first_snapshot_sec, 4)
                if r.first_snapshot_sec is not None
                else None,
                "ttft_sec": round(r.ttft_sec, 4) if r.ttft_sec is not None else None,
                "elapsed_sec": round(r.elapsed_sec, 4),
                "started_at": r.started_at,
                "finished_at": r.finished_at,
                "snapshot_count": r.snapshot_count,
                "think_len": len(r.think),
                "answer_len": len(r.answer),
                "think": r.think,
                "answer": r.answer,
            }
        )
    return rows


def _build_summary_row(
    run: ConcurrencyRun,
    url: str,
    base_payload: dict[str, Any],
    ran_at: str,
) -> dict[str, Any]:
    n = len(run.results)
    succ = sum(1 for r in run.results if r.ok)
    elapses = [r.elapsed_sec for r in run.results]
    succ_elapses = [r.elapsed_sec for r in run.results if r.ok]
    ttfts = [r.ttft_sec for r in run.results if r.ttft_sec is not None]
    succ_ttfts = [r.ttft_sec for r in run.results if r.ok and r.ttft_sec is not None]
    return {
        "ran_at": ran_at,
        "url": url,
        "concurrency": run.concurrency,
        "total_requests": n,
        "success_count": succ,
        "fail_count": n - succ,
        "success_rate_pct": round(succ / n * 100, 2) if n else 0.0,
        "wall_time_sec": round(run.wall_time, 4),
        "qps": round(n / run.wall_time, 4) if run.wall_time > 0 else 0.0,
        "payload_json": json.dumps(base_payload, ensure_ascii=False),
        "elapsed_min": round(min(elapses), 4) if elapses else None,
        "elapsed_avg": round(statistics.mean(elapses), 4) if elapses else None,
        "elapsed_p50": round(percentile(elapses, 50), 4) if elapses else None,
        "elapsed_p90": round(percentile(elapses, 90), 4) if elapses else None,
        "elapsed_p95": round(percentile(elapses, 95), 4) if elapses else None,
        "elapsed_p99": round(percentile(elapses, 99), 4) if elapses else None,
        "elapsed_max": round(max(elapses), 4) if elapses else None,
        "success_elapsed_avg": round(statistics.mean(succ_elapses), 4) if succ_elapses else None,
        "ttft_count": len(ttfts),
        "ttft_min": round(min(ttfts), 4) if ttfts else None,
        "ttft_avg": round(statistics.mean(ttfts), 4) if ttfts else None,
        "ttft_p50": round(percentile(ttfts, 50), 4) if ttfts else None,
        "ttft_p90": round(percentile(ttfts, 90), 4) if ttfts else None,
        "ttft_p95": round(percentile(ttfts, 95), 4) if ttfts else None,
        "ttft_p99": round(percentile(ttfts, 99), 4) if ttfts else None,
        "ttft_max": round(max(ttfts), 4) if ttfts else None,
        "success_ttft_avg": round(statistics.mean(succ_ttfts), 4) if succ_ttfts else None,
    }


def export_xlsx(
    out_path: str,
    run: ConcurrencyRun,
    url: str,
    base_payload: dict[str, Any],
    ran_at: str,
) -> None:
    detail_df = pd.DataFrame(_build_detail_rows(run))
    summary_df = pd.DataFrame([_build_summary_row(run, url, base_payload, ran_at)])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        detail_df.to_excel(writer, sheet_name="requests", index=False)
    safe_print(f"    结果已写入: {out_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="对 /api/inference/stream 进行分档位并发压测并导出 xlsx",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--url", default=DEFAULT_URL, help="stream SSE 接口地址")
    p.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=list(DEFAULT_CONCURRENCY),
        help="并发档位列表，每个档位各发 N 个并发请求",
    )
    p.add_argument("--timeout", type=float, default=900.0, help="单请求 HTTP 超时（秒）")
    p.add_argument(
        "--cooldown",
        type=float,
        default=5.0,
        help="相邻并发档位之间的间隔秒数",
    )
    p.add_argument("--verbose", action="store_true", help="打印每个请求的进度")
    p.add_argument(
        "--out-dir",
        default="",
        help="xlsx 输出目录，默认为本脚本所在目录（test/）",
    )
    p.add_argument(
        "--payload-file",
        help="从 JSON 文件读取请求体（需含 policyId、question；taskId 由脚本自动生成）",
    )
    p.add_argument("--policy-id", help="覆盖默认 policyId")
    p.add_argument("--question", help="覆盖默认 question")
    return p.parse_args(argv)


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    if args.payload_file:
        with open(args.payload_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = dict(DEFAULT_PAYLOAD)
        if args.policy_id:
            payload["policyId"] = args.policy_id
        if args.question:
            payload["question"] = args.question
    payload.pop("taskId", None)
    if "policyId" not in payload or "question" not in payload:
        raise ValueError("请求体必须包含 policyId 与 question")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(__file__))

    try:
        base_payload = build_payload(args)
    except (ValueError, OSError) as e:
        safe_print(f"构造请求体失败: {e}")
        return 2

    levels = sorted(set(args.concurrency))
    if any(c <= 0 for c in levels):
        safe_print("错误：并发数必须为正整数")
        return 2

    ran_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_print("=" * 80)
    safe_print(f"压测目标: {args.url}")
    safe_print(f"并发档位: {levels}")
    safe_print(f"单请求超时: {args.timeout}s   档位间隔: {args.cooldown}s")
    safe_print(f"输出目录: {out_dir}")
    safe_print("基础请求体（taskId 由脚本为每个请求单独生成）:")
    safe_print(json.dumps(base_payload, ensure_ascii=False, indent=2))
    safe_print("=" * 80)

    all_ok = True
    try:
        for i, level in enumerate(levels):
            run = run_concurrency_level(
                url=args.url,
                base_payload=base_payload,
                concurrency=level,
                timeout=args.timeout,
                verbose=args.verbose,
            )
            out_path = os.path.join(
                out_dir,
                f"stream_stress_concurrency_{level}_{ran_at}.xlsx",
            )
            export_xlsx(out_path, run, args.url, base_payload, ran_at)
            if not all(r.ok for r in run.results):
                all_ok = False
            if i < len(levels) - 1 and args.cooldown > 0:
                safe_print(f"--- 等待 {args.cooldown}s 后进入下一档位 ---")
                time.sleep(args.cooldown)
    except KeyboardInterrupt:
        safe_print("\n[中断] 用户取消")
        return 130

    safe_print("\n全部档位压测完成。")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
