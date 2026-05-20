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
import random
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
DEFAULT_CONCURRENCY = (16, 32, 48, 64)
DEFAULT_QUESTIONS_FILE = "ncp_test_0423.csv"
"""默认问题集文件名（相对脚本所在目录）。

模拟真实场景：每条并发的 ``question`` 都来自该文件，互不重复，
更贴近线上 SSE 流量的多 policyId/多问题分布。
"""

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


def load_questions(path: str) -> list[str]:
    """从 CSV / xlsx 读取问题集，返回去重后的列表（保留原顺序）。

    兼容列名 ``问题`` / ``question`` / ``Q``，至少需要其中之一。
    """

    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path, engine="openpyxl")
    else:
        raise ValueError(f"不支持的问题集文件类型: {ext}，仅支持 .csv / .xlsx")

    cols = list(df.columns)
    col = None
    for cand in ("问题", "question", "Question", "Q"):
        if cand in cols:
            col = cand
            break
    if col is None:
        raise ValueError(f"问题集缺少 '问题' 或 'question' 列，实际表头={cols}")

    questions: list[str] = []
    seen: set[str] = set()
    for raw in df[col].tolist():
        if raw is None:
            continue
        q = str(raw).strip()
        if not q or q.lower() == "nan":
            continue
        if q in seen:
            continue
        seen.add(q)
        questions.append(q)
    if not questions:
        raise ValueError(f"问题集为空: {path}")
    return questions


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
    # 流被网关 / 中间链路截断的次数（含「流自然结束但未收到 done/failed」）。
    cut_count: int = 0
    # 实际触发的重连次数；恒等于 attempts - 1。
    reconnect_count: int = 0


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


_CUT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ConnectionError,
)


def _one_sse_pass(
    url: str,
    body: dict[str, Any],
    t0: float,
    state: dict[str, Any],
    timeout: float,
) -> tuple[bool, bool, str | None]:
    """读一次 SSE 流，**原地**更新 ``state``。

    服务端约定：``taskId`` 不变时，``/api/inference/stream`` 会继续转发同一份
    Redis 快照（``inference:task:{taskId}`` 仍在 TTL 内），新连接的第一帧
    就会带上累积的 think/answer。因此本函数被多次调用以实现「断流重连」。

    Returns
    -------
    terminated : bool
        是否走到了 ``status in {done, failed}``，即流已经看到完整终态。
    was_cut : bool
        是否因传输被截断（``ChunkedEncodingError`` / ``ConnectionError`` /
        流自然结束但未收到 done/failed）而提前终止——这类情况外层会按策略重连。
    err_str : str | None
        若非空，用于打印 / 写入 ``state['error']``。
    """

    try:
        with requests.post(url, json=body, stream=True, timeout=timeout) as resp:
            if state["http_connected_sec"] is None:
                state["http_connected_sec"] = time.perf_counter() - t0
            state["http_status"] = resp.status_code
            if resp.status_code != 200:
                return False, False, f"HTTP {resp.status_code}: {resp.text[:500]}"
            for event, payload in _iter_sse(resp):
                if event != "snapshot" or not isinstance(payload, dict):
                    continue
                if state["first_snapshot_sec"] is None:
                    state["first_snapshot_sec"] = time.perf_counter() - t0
                state["snapshot_count"] += 1
                if payload.get("taskId"):
                    state["task_id"] = str(payload["taskId"])
                new_think = str(payload.get("think") or "")
                new_answer = str(payload.get("answer") or "")
                state["ttft_sec"] = _record_ttft(
                    t0, new_think, new_answer, state["ttft_sec"]
                )
                # 服务端 snapshot 里 think/answer 是 append-only 的全量，
                # 重连后第一帧就会带上之前累计的内容，覆盖式赋值即可。
                state["think"] = new_think or state["think"]
                state["answer"] = new_answer or state["answer"]
                if "previewDone" in payload:
                    state["preview_done"] = bool(payload["previewDone"])
                if payload.get("reactRound") is not None:
                    state["react_round"] = int(payload["reactRound"])
                if "answerable" in payload:
                    ans = payload["answerable"]
                    state["answerable"] = bool(ans) if ans is not None else None
                st = payload.get("status")
                if st:
                    state["final_status"] = str(st)
                if payload.get("error"):
                    state["error"] = str(payload["error"])
                if state["final_status"] in {"done", "failed"}:
                    return True, False, None
            # 走到这说明 iter_lines 干净结束但没看到 done/failed：
            # 典型是网关 / 中间链路把 chunked 末尾的 0\r\n\r\n 也吞了，
            # 表现为 requests 不抛异常但 SSE 半截没了——按「被切」处理重连。
            return False, True, "SSE 流被对端无标记关闭（可能是网关 hard timeout）"
    except _CUT_EXCEPTIONS as e:
        return False, True, f"{type(e).__name__}: {e}"
    except requests.RequestException as e:
        return False, False, f"{type(e).__name__}: {e}"
    except Exception as e:  # noqa: BLE001
        return False, False, f"{type(e).__name__}: {e}"


def do_stream_request(
    url: str,
    base_payload: dict[str, Any],
    concurrency: int,
    req_idx: int,
    timeout: float,
    verbose: bool,
    *,
    question_override: str | None = None,
    reconnect_on_cut: bool = True,
    max_reconnects: int = 3,
    reconnect_backoff_sec: float = 1.0,
) -> StreamRequestResult:
    task_id = str(uuid4())
    body = {**base_payload, "taskId": task_id}
    # question_override 让本路并发使用题库里专属的一条问题——
    # 在 main 里按"每档 N 条互不重复"的策略已经分配好。
    if question_override is not None:
        body["question"] = question_override
    input_json = json.dumps(body, ensure_ascii=False)
    policy_id = str(body.get("policyId") or "")
    question = str(body.get("question") or "")
    started_dt = datetime.now()
    t0 = time.perf_counter()

    state: dict[str, Any] = {
        "http_status": None,
        "final_status": None,
        "preview_done": None,
        "react_round": None,
        "answerable": None,
        "think": "",
        "answer": "",
        "error": None,
        "snapshot_count": 0,
        "http_connected_sec": None,
        "first_snapshot_sec": None,
        "ttft_sec": None,
        "task_id": task_id,
    }

    cut_count = 0
    reconnect_count = 0
    attempts_max = 1 + (max_reconnects if reconnect_on_cut else 0)
    last_cut_err: str | None = None
    terminated = False

    for attempt in range(attempts_max):
        if attempt > 0:
            reconnect_count += 1
            if verbose:
                safe_print(
                    f"[c={concurrency} req {req_idx:>4}] 第 {reconnect_count} 次重连 "
                    f"taskId={state['task_id'][:8]}... cut_count={cut_count} "
                    f"last_err={last_cut_err}"
                )
            time.sleep(reconnect_backoff_sec)
            # 重连必须带同一 taskId，让服务端把后续帧接力推过来。
            body["taskId"] = state["task_id"]

        terminated, was_cut, err_str = _one_sse_pass(url, body, t0, state, timeout)

        if terminated:
            break
        if was_cut:
            cut_count += 1
            last_cut_err = err_str
            if attempt + 1 < attempts_max:
                continue
            state["error"] = (
                f"流连续 {cut_count} 次被截断（最后一次：{err_str}）"
            )
            break
        if err_str and not state["error"]:
            state["error"] = err_str
        break

    elapsed = time.perf_counter() - t0
    finished_dt = datetime.now()

    if (
        state["http_status"] == 200
        and state["final_status"] is None
        and not state["error"]
    ):
        state["error"] = "流结束但未收到 status=done/failed 的 snapshot"

    ok = (
        state["http_status"] == 200
        and state["final_status"] == "done"
        and not state["error"]
    )

    result = StreamRequestResult(
        concurrency=concurrency,
        req_idx=req_idx,
        task_id=state["task_id"],
        policy_id=policy_id,
        question=question,
        input_json=input_json,
        http_status=state["http_status"],
        final_status=state["final_status"],
        preview_done=state["preview_done"],
        react_round=state["react_round"],
        answerable=state["answerable"],
        think=state["think"],
        answer=state["answer"],
        error=state["error"],
        ok=ok,
        elapsed_sec=elapsed,
        http_connected_sec=state["http_connected_sec"],
        first_snapshot_sec=state["first_snapshot_sec"],
        ttft_sec=state["ttft_sec"],
        started_at=started_dt.isoformat(timespec="seconds"),
        finished_at=finished_dt.isoformat(timespec="seconds"),
        snapshot_count=state["snapshot_count"],
        cut_count=cut_count,
        reconnect_count=reconnect_count,
    )

    if verbose:
        tag = "OK " if ok else "ERR"
        ttft_disp = f"{result.ttft_sec:.2f}s" if result.ttft_sec is not None else "n/a"
        rec_disp = f" rec={reconnect_count}/{cut_count}" if cut_count else ""
        safe_print(
            f"[c={concurrency} req {req_idx:>4}] {tag} taskId={result.task_id[:8]}... "
            f"http={result.http_status} status={result.final_status} "
            f"ttft={ttft_disp} elapsed={elapsed:.2f}s"
            f"{rec_disp} "
            f"think_len={len(result.think)} answer_len={len(result.answer)}"
            + (f" err={result.error}" if result.error else "")
        )

    return result


def run_concurrency_level(
    url: str,
    base_payload: dict[str, Any],
    concurrency: int,
    timeout: float,
    verbose: bool,
    *,
    questions: list[str] | None = None,
    reconnect_on_cut: bool = True,
    max_reconnects: int = 3,
    reconnect_backoff_sec: float = 1.0,
) -> ConcurrencyRun:
    if questions is not None and len(questions) != concurrency:
        raise ValueError(
            f"questions 长度 {len(questions)} 与并发数 {concurrency} 不一致"
        )

    safe_print(f"\n>>> 并发档位 {concurrency}：同时发起 {concurrency} 个 stream 请求 ...")
    if questions is not None:
        safe_print(
            f"    使用 {len(questions)} 条互不重复的题库问题（首条："
            f"{questions[0][:30]}{'...' if len(questions[0]) > 30 else ''}）"
        )
    started_at = time.perf_counter()
    results: list[StreamRequestResult] = []

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                do_stream_request,
                url,
                base_payload,
                concurrency,
                i + 1,
                timeout,
                verbose,
                question_override=(questions[i] if questions is not None else None),
                reconnect_on_cut=reconnect_on_cut,
                max_reconnects=max_reconnects,
                reconnect_backoff_sec=reconnect_backoff_sec,
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
    cut_total = sum(r.cut_count for r in run.results)
    reconnected = sum(1 for r in run.results if r.reconnect_count > 0)
    safe_print(
        f"    完成: 总数={n} 成功={succ} 失败={n - succ} "
        f"墙钟={run.wall_time:.2f}s QPS={n / run.wall_time:.3f}"
        if run.wall_time > 0
        else f"    完成: 总数={n} 成功={succ} 失败={n - succ}"
    )
    if cut_total > 0 or reconnected > 0:
        safe_print(
            f"    断流统计: 被切总次数={cut_total} 触发重连的请求数={reconnected}"
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
                "cut_count": r.cut_count,
                "reconnect_count": r.reconnect_count,
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
        "cut_total": sum(r.cut_count for r in run.results),
        "reconnected_requests": sum(1 for r in run.results if r.reconnect_count > 0),
        "reconnect_total": sum(r.reconnect_count for r in run.results),
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
    p.add_argument(
        "--question",
        help="覆盖默认 question；显式传入后所有并发都使用同一条问题，"
        "忽略 --questions-file",
    )
    p.add_argument(
        "--questions-file",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            DEFAULT_QUESTIONS_FILE,
        ),
        help="问题集文件路径（CSV / xlsx，需含 '问题' 或 'question' 列）。"
        "每档会随机抽取与并发数相同数量的不重复问题——更贴近真实场景。",
    )
    p.add_argument(
        "--no-questions-file",
        action="store_true",
        help="禁用问题集，所有并发统一使用 --question / 默认 payload 里的固定问题",
    )
    p.add_argument(
        "--questions-seed",
        type=int,
        default=None,
        help="问题抽取的随机种子；不传则每次完全随机",
    )
    p.add_argument(
        "--reconnect-on-cut",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="SSE 被网关 / 中间链路截断时，凭同一 taskId 自动重连续接（默认开启）",
    )
    p.add_argument(
        "--max-reconnects",
        type=int,
        default=3,
        help="单条请求最多重连多少次；--no-reconnect-on-cut 时本参数被忽略",
    )
    p.add_argument(
        "--reconnect-backoff",
        type=float,
        default=1.0,
        help="每次重连前等待秒数",
    )
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

    # 问题集加载策略：
    #   - --no-questions-file / --question / --payload-file 任一显式给出固定问题，关闭题库；
    #   - 否则尝试加载 --questions-file（默认 test/ncp_test_0423.csv）。
    questions_pool: list[str] | None = None
    use_pool = (
        not args.no_questions_file
        and not args.question
        and not args.payload_file
    )
    if use_pool:
        try:
            questions_pool = load_questions(args.questions_file)
        except (ValueError, OSError, FileNotFoundError) as e:
            safe_print(f"加载问题集失败，回落到固定 question：{e}")
            questions_pool = None

    if questions_pool is not None:
        max_level = max(levels)
        if max_level > len(questions_pool):
            safe_print(
                f"错误：最高并发档位 {max_level} > 题库容量 {len(questions_pool)}，"
                f"无法保证每条并发问题都不同。"
                f"请缩小档位或扩充 {args.questions_file}。"
            )
            return 2

    ran_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_print("=" * 80)
    safe_print(f"压测目标: {args.url}")
    safe_print(f"并发档位: {levels}")
    safe_print(f"单请求超时: {args.timeout}s   档位间隔: {args.cooldown}s")
    if questions_pool is not None:
        seed_disp = args.questions_seed if args.questions_seed is not None else "随机"
        safe_print(
            f"问题集    : {args.questions_file} (共 {len(questions_pool)} 条不重复问题；"
            f"种子={seed_disp})"
        )
    else:
        safe_print("问题集    : 关闭（所有并发使用同一条 question）")
    if args.reconnect_on_cut:
        safe_print(
            f"断流重连: 开启（最多 {args.max_reconnects} 次，"
            f"退避 {args.reconnect_backoff}s）"
        )
    else:
        safe_print("断流重连: 关闭")
    safe_print(f"输出目录: {out_dir}")
    safe_print("基础请求体（taskId 由脚本生成；question 受题库控制时按档位覆盖）:")
    safe_print(json.dumps(base_payload, ensure_ascii=False, indent=2))
    safe_print("=" * 80)

    rng = (
        random.Random(args.questions_seed)
        if args.questions_seed is not None
        else random.Random()
    )

    all_ok = True
    try:
        for i, level in enumerate(levels):
            questions_for_level: list[str] | None = None
            if questions_pool is not None:
                questions_for_level = rng.sample(questions_pool, level)

            run = run_concurrency_level(
                url=args.url,
                base_payload=base_payload,
                concurrency=level,
                timeout=args.timeout,
                verbose=args.verbose,
                questions=questions_for_level,
                reconnect_on_cut=args.reconnect_on_cut,
                max_reconnects=args.max_reconnects,
                reconnect_backoff_sec=args.reconnect_backoff,
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
