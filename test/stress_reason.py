"""
/api/reason 接口并发压测脚本

用同一组入参，分批并发发送 batch_size * batches 个请求：
- 每批同时发出 batch_size 个请求（线程池真并发）
- 共发送 batches 批
- 批与批之间可设置间隔时间（默认 0，背靠背）

用法示例（PowerShell）：
    python test/stress_reason.py `
        --url http://localhost:5000/api/reason `
        --policy-id 12345 `
        --question "请问蔬菜流通环节增值税如何处理？" `
        --batch-size 5 `
        --batches 4 `
        --interval 0

可通过 --payload-file 直接读取一个 JSON 文件作为请求体（覆盖命令行的 reason 入参）。
统计指标：成功率、错误码分布、各分位耗时、QPS、按批次明细。
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
from typing import Any

import requests


@dataclass
class RequestResult:
    batch_idx: int
    req_idx: int
    start_ts: float
    elapsed: float
    http_status: int | None
    biz_status: int | None
    ok: bool
    error: str | None = None
    answer_preview: str = ""


@dataclass
class BatchStats:
    batch_idx: int
    started_at: float
    finished_at: float
    results: list[RequestResult] = field(default_factory=list)

    @property
    def wall_time(self) -> float:
        return self.finished_at - self.started_at

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results if r.ok)

    @property
    def fail_count(self) -> int:
        return len(self.results) - self.success_count


_print_lock = threading.Lock()


def safe_print(*args: Any, **kwargs: Any) -> None:
    with _print_lock:
        print(*args, **kwargs, flush=True)


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    if args.payload_file:
        with open(args.payload_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if "policyId" not in payload or "question" not in payload:
            raise ValueError("payload-file 中必须包含 policyId 与 question 字段")
        return payload

    if not args.policy_id or not args.question:
        raise ValueError("必须指定 --policy-id 与 --question，或通过 --payload-file 提供")

    payload: dict[str, Any] = {
        "policyId": args.policy_id,
        "question": args.question,
        "maxRounds": args.max_rounds,
        "vendor": args.vendor,
        "model": args.model,
        "cleanAnswer": args.clean_answer,
        "summaryBatchSize": args.summary_batch_size,
        "retrievalMode": args.retrieval_mode,
        "checkPitfalls": args.check_pitfalls,
        "chunkSize": args.chunk_size,
        "version": args.version,
        "enableSkills": args.enable_skills,
        "summaryCleanAnswer": args.summary_clean_answer,
        "thinkMode": args.think_mode,
    }
    if args.answer_system_prompt is not None:
        payload["answerSystemPrompt"] = args.answer_system_prompt
    return payload


def do_request(
    url: str,
    payload: dict[str, Any],
    batch_idx: int,
    req_idx: int,
    timeout: float,
    verbose: bool,
) -> RequestResult:
    start_ts = time.time()
    t0 = time.perf_counter()
    http_status: int | None = None
    biz_status: int | None = None
    ok = False
    error: str | None = None
    answer_preview = ""

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        http_status = resp.status_code
        try:
            body = resp.json()
            biz_status = body.get("status_code")
            data = body.get("data") or {}
            answer = data.get("answer") or ""
            answer_preview = answer.replace("\n", " ")[:60]
            ok = (http_status == 200) and (biz_status == 200)
            if not ok:
                error = f"biz_msg={body.get('message')}"
        except ValueError:
            error = f"非 JSON 响应: {resp.text[:200]}"
    except requests.RequestException as e:
        error = f"{type(e).__name__}: {e}"
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"

    elapsed = time.perf_counter() - t0
    result = RequestResult(
        batch_idx=batch_idx,
        req_idx=req_idx,
        start_ts=start_ts,
        elapsed=elapsed,
        http_status=http_status,
        biz_status=biz_status,
        ok=ok,
        error=error,
        answer_preview=answer_preview,
    )

    if verbose:
        tag = "OK " if ok else "ERR"
        extra = f" answer={answer_preview!r}" if ok else f" err={error}"
        safe_print(
            f"[batch {batch_idx:>3} req {req_idx:>3}] {tag} "
            f"http={http_status} biz={biz_status} elapsed={elapsed:.2f}s{extra}"
        )

    return result


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


def summarize_batch(b: BatchStats) -> str:
    elapses = [r.elapsed for r in b.results]
    if not elapses:
        return f"批 {b.batch_idx}: 无结果"
    return (
        f"批 {b.batch_idx:>3} | 总数={len(b.results):>3} "
        f"成功={b.success_count:>3} 失败={b.fail_count:>3} "
        f"墙钟={b.wall_time:6.2f}s "
        f"min={min(elapses):5.2f}s avg={statistics.mean(elapses):5.2f}s "
        f"p50={percentile(elapses,50):5.2f}s p95={percentile(elapses,95):5.2f}s "
        f"max={max(elapses):5.2f}s"
    )


def summarize_overall(all_results: list[RequestResult], total_wall: float) -> str:
    n = len(all_results)
    if n == 0:
        return "未收集到任何结果"
    succ = [r for r in all_results if r.ok]
    fail = [r for r in all_results if not r.ok]
    elapses = [r.elapsed for r in all_results]
    succ_elapses = [r.elapsed for r in succ]

    err_count: dict[str, int] = {}
    for r in fail:
        key = f"http={r.http_status}|biz={r.biz_status}|err={r.error}"
        err_count[key] = err_count.get(key, 0) + 1

    lines = [
        "=" * 80,
        "整体压测汇总",
        "=" * 80,
        f"总请求数        : {n}",
        f"成功 / 失败     : {len(succ)} / {len(fail)}  (成功率 {len(succ)/n*100:.2f}%)",
        f"总墙钟时间      : {total_wall:.2f}s",
        f"整体吞吐 (QPS)  : {n/total_wall:.2f}",
        f"成功吞吐 (QPS)  : {len(succ)/total_wall:.2f}",
        "-" * 80,
        "全部请求耗时分布:",
        f"  min={min(elapses):.2f}s  avg={statistics.mean(elapses):.2f}s  "
        f"p50={percentile(elapses,50):.2f}s  p90={percentile(elapses,90):.2f}s  "
        f"p95={percentile(elapses,95):.2f}s  p99={percentile(elapses,99):.2f}s  "
        f"max={max(elapses):.2f}s",
    ]
    if succ_elapses:
        lines += [
            "成功请求耗时分布:",
            f"  min={min(succ_elapses):.2f}s  avg={statistics.mean(succ_elapses):.2f}s  "
            f"p50={percentile(succ_elapses,50):.2f}s  p90={percentile(succ_elapses,90):.2f}s  "
            f"p95={percentile(succ_elapses,95):.2f}s  p99={percentile(succ_elapses,99):.2f}s  "
            f"max={max(succ_elapses):.2f}s",
        ]
    if err_count:
        lines.append("-" * 80)
        lines.append("错误分组：")
        for k, v in sorted(err_count.items(), key=lambda x: -x[1]):
            lines.append(f"  [{v}]  {k}")
    lines.append("=" * 80)
    return "\n".join(lines)


def run_batch(
    url: str,
    payload: dict[str, Any],
    batch_idx: int,
    batch_size: int,
    timeout: float,
    verbose: bool,
) -> BatchStats:
    safe_print(f"\n>>> 启动第 {batch_idx} 批：并发 {batch_size} 个请求 ...")
    started_at = time.perf_counter()
    results: list[RequestResult] = []

    with ThreadPoolExecutor(max_workers=batch_size) as pool:
        futures = [
            pool.submit(do_request, url, payload, batch_idx, i + 1, timeout, verbose)
            for i in range(batch_size)
        ]
        for fut in as_completed(futures):
            results.append(fut.result())

    finished_at = time.perf_counter()
    results.sort(key=lambda r: r.req_idx)
    batch = BatchStats(
        batch_idx=batch_idx,
        started_at=started_at,
        finished_at=finished_at,
        results=results,
    )
    safe_print(summarize_batch(batch))
    return batch


def export_results(
    out_path: str,
    payload: dict[str, Any],
    batches: list[BatchStats],
    total_wall: float,
) -> None:
    flat = []
    for b in batches:
        for r in b.results:
            flat.append({
                "batch_idx": r.batch_idx,
                "req_idx": r.req_idx,
                "start_ts": r.start_ts,
                "elapsed_sec": round(r.elapsed, 4),
                "http_status": r.http_status,
                "biz_status": r.biz_status,
                "ok": r.ok,
                "error": r.error,
                "answer_preview": r.answer_preview,
            })
    payload_for_dump = {k: v for k, v in payload.items() if k != "answerSystemPrompt"}
    out = {
        "url": None,
        "ran_at": datetime.now().isoformat(timespec="seconds"),
        "total_wall_sec": round(total_wall, 4),
        "request_payload": payload_for_dump,
        "batch_summaries": [
            {
                "batch_idx": b.batch_idx,
                "wall_time_sec": round(b.wall_time, 4),
                "success": b.success_count,
                "fail": b.fail_count,
            }
            for b in batches
        ],
        "results": flat,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    safe_print(f"\n详细结果已写入: {out_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="对 /api/reason 接口进行分批并发压测",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--url", default="http://localhost:5000/api/reason", help="reason 接口地址")
    p.add_argument("-n", "--batch-size", type=int, default=5, help="每批并发请求数 n")
    p.add_argument("-m", "--batches", type=int, default=2, help="批次数量 m，总请求数=n*m")
    p.add_argument("--interval", type=float, default=0.0, help="批次之间间隔秒数")
    p.add_argument("--timeout", type=float, default=600.0, help="单请求 HTTP 超时秒数")
    p.add_argument("--verbose", action="store_true", help="打印每个请求的明细日志")
    p.add_argument("--out", default="", help="结果导出 JSON 路径，默认 test/stress_result_<时间戳>.json")

    p.add_argument("--payload-file", help="直接从 JSON 文件读取请求体（覆盖下方 reason 参数）")

    p.add_argument("--policy-id", help="reason 入参：policyId")
    p.add_argument("--question", help="reason 入参：question")
    p.add_argument("--max-rounds", type=int, default=10)
    p.add_argument("--vendor", default="qwen3.5-122b-a10b")
    p.add_argument("--model", default="Qwen3.5-122B-A10B")
    p.add_argument("--clean-answer", action="store_true", default=False)
    p.add_argument("--summary-batch-size", type=int, default=3)
    p.add_argument("--retrieval-mode", action="store_true", default=True)
    p.add_argument("--check-pitfalls", action="store_true", default=True)
    p.add_argument("--chunk-size", type=int, default=3000)
    p.add_argument("--version", default="v1", choices=["v0", "v1"])
    p.add_argument("--enable-skills", action="store_true", default=True)
    p.add_argument("--summary-clean-answer", action="store_true", default=True)
    p.add_argument("--think-mode", action="store_true", default=True)
    p.add_argument("--answer-system-prompt", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.batch_size <= 0 or args.batches <= 0:
        safe_print("错误：--batch-size 与 --batches 必须为正整数")
        return 2

    try:
        payload = build_payload(args)
    except (ValueError, OSError) as e:
        safe_print(f"构造请求体失败: {e}")
        return 2

    safe_print("=" * 80)
    safe_print(f"压测目标: {args.url}")
    safe_print(f"分批方案: 每批 {args.batch_size} 并发 × {args.batches} 批 = {args.batch_size * args.batches} 次请求")
    safe_print(f"批间隔  : {args.interval}s   单请求超时: {args.timeout}s")
    safe_print("请求体  :")
    safe_print(json.dumps(payload, ensure_ascii=False, indent=2))
    safe_print("=" * 80)

    batches: list[BatchStats] = []
    overall_t0 = time.perf_counter()

    try:
        for i in range(1, args.batches + 1):
            batch = run_batch(
                url=args.url,
                payload=payload,
                batch_idx=i,
                batch_size=args.batch_size,
                timeout=args.timeout,
                verbose=args.verbose,
            )
            batches.append(batch)
            if i < args.batches and args.interval > 0:
                safe_print(f"--- 等待 {args.interval}s 进入下一批 ---")
                time.sleep(args.interval)
    except KeyboardInterrupt:
        safe_print("\n[中断] 用户取消，输出已收集的结果 ...")

    total_wall = time.perf_counter() - overall_t0
    all_results = [r for b in batches for r in b.results]

    safe_print("\n各批次明细：")
    for b in batches:
        safe_print(summarize_batch(b))

    safe_print("\n" + summarize_overall(all_results, total_wall))

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"stress_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    export_results(out_path, payload, batches, total_wall)
    return 0 if all(r.ok for r in all_results) else 1


if __name__ == "__main__":
    sys.exit(main())
    # 示例：运行 20 并发 × 10 批 的压测命令（可直接粘贴到终端）
    # Linux/macOS:
    #   python test/stress_reason.py --url http://localhost:5000/api/reason --policy-id 12345 --question "请问蔬菜流通环节增值税如何处理？" --batch-size 20 --batches 10 --interval 0
    # python test/stress_reason.py --url http://mlp.paas.dc.servyou-it.com/kh-server/api/reason --policy-id KH1493204307733168128_20260417131730 --question "批发要售圣女果是否免征增值税" --batch-size 20 --batches 10 --interval 0