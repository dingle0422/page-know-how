"""Locust-based load test for OpenAI-like streaming chat endpoint.

This file serves two roles:
1) Locust file (`locust -f test/stress_llm_locust.py`) with `LlmStreamUser`.
2) Convenience runner (`python test/stress_llm_locust.py`) that executes
   multiple concurrency levels and exports summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

if __name__ != "__main__":
    from locust import HttpUser, constant, events, task
else:
    # Avoid importing gevent/locust in runner mode to prevent shutdown noise.
    class HttpUser:  # pragma: no cover
        pass

    def constant(_seconds: float):  # pragma: no cover
        return lambda: 0

    def task(fn):  # pragma: no cover
        return fn

    class _DummyTestStop:  # pragma: no cover
        @staticmethod
        def add_listener(fn):
            return fn

    class _DummyEvents:  # pragma: no cover
        test_stop = _DummyTestStop()

    events = _DummyEvents()

DEFAULT_URL = "http://mlp.paas.dc.servyou-it.com"
DEFAULT_ENDPOINT = "/mudgate/api/llm/servyou/v1/chat/completions"
DEFAULT_MODEL = "deepseek-v3.2-1163259bcc6c"
DEFAULT_CONCURRENCY = (64, 72, 80, 88, 96)
DEFAULT_PROMPT = "请简要介绍增值税发票开具注意事项。"
API_KEY = "sk-a57093c05ed94f37a7c845ff3fd688e2"


def _resolve_api_key(cli_value: str = "", env_name: str = "SERVYOU_APP_ID") -> str:
    return (cli_value or os.environ.get("LLM_API_KEY") or os.environ.get(env_name) or API_KEY).strip()


def _resolve_app_id(
    cli_value: str = "",
    *,
    env_name: str = "SERVYOU_APP_ID",
    api_key: str = "",
) -> str:
    key = (cli_value or os.environ.get("LLM_APP_ID") or os.environ.get(env_name) or api_key or API_KEY).strip()
    return key


def safe_print(*args: Any, **kwargs: Any) -> None:
    print(*args, **kwargs, flush=True)


def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _redact_secret(text: str) -> str:
    if not text:
        return "<REDACTED>"
    if len(text) <= 8:
        return "<REDACTED>"
    return f"{text[:3]}***{text[-3:]}"


def _redact_command(cmd: list[str]) -> str:
    out: list[str] = []
    i = 0
    while i < len(cmd):
        cur = cmd[i]
        out.append(cur)
        if cur in {"--api-key"} and i + 1 < len(cmd):
            out.append(_redact_secret(cmd[i + 1]))
            i += 2
            continue
        i += 1
    return shlex.join(out)


def _load_prompts(path: str) -> list[str]:
    if not path:
        return [DEFAULT_PROMPT]
    ext = os.path.splitext(path)[1].lower()
    prompts: list[str] = []
    if ext in {".txt", ".md"}:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text:
                    prompts.append(text)
    elif ext == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            cols = list(reader.fieldnames or [])
            target = None
            for cand in ("prompt", "question", "问题", "text"):
                if cand in cols:
                    target = cand
                    break
            if target is None:
                raise ValueError(f"csv prompt file missing expected column, found={cols}")
            for row in reader:
                text = str(row.get(target, "")).strip()
                if text:
                    prompts.append(text)
    else:
        raise ValueError(f"unsupported prompt file extension: {ext}")
    if not prompts:
        raise ValueError(f"prompt file has no valid prompt: {path}")
    return prompts


# ----------------------------- Locust worker side -----------------------------

_metrics_lock = threading.Lock()
_metrics_rows: list[dict[str, Any]] = []


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _build_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    raw_headers = os.environ.get("LLM_HEADERS_JSON", "")
    if raw_headers:
        try:
            parsed = json.loads(raw_headers)
            if isinstance(parsed, dict):
                for k, v in parsed.items():
                    headers[str(k)] = str(v)
        except json.JSONDecodeError:
            pass

    api_key = _resolve_api_key()
    auth_mode = os.environ.get("LLM_AUTH_MODE", "raw")
    if api_key and auth_mode != "none":
        if auth_mode == "bearer":
            headers["Authorization"] = f"Bearer {api_key}"
        else:
            headers["Authorization"] = api_key
    if _env_bool("LLM_ADD_ACCEPT_SSE", False):
        headers["Accept"] = "text/event-stream"
    return headers


def _build_payload(prompt: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": os.environ.get("LLM_MODEL", DEFAULT_MODEL),
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    api_key = _resolve_api_key()
    app_id = _resolve_app_id(api_key=api_key)
    if app_id:
        payload["appId"] = app_id
    if _env_bool("LLM_ENABLE_THINKING", False):
        payload["enable_thinking"] = True
    if _env_bool("LLM_CHAT_TEMPLATE_ENABLE_THINKING", False):
        payload["chat_template_kwargs"] = {"enable_thinking": True}

    max_tokens = os.environ.get("LLM_MAX_TOKENS", "")
    if max_tokens:
        try:
            payload["max_tokens"] = int(max_tokens)
        except ValueError:
            pass
    temperature = os.environ.get("LLM_TEMPERATURE", "")
    if temperature:
        try:
            payload["temperature"] = float(temperature)
        except ValueError:
            pass
    top_p = os.environ.get("LLM_TOP_P", "")
    if top_p:
        try:
            payload["top_p"] = float(top_p)
        except ValueError:
            pass
    return payload


def _truncate_text(text: str, max_len: int = 180) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_len:
        return normalized
    return normalized[:max_len] + "..."


def _response_error_detail(resp: Any) -> str:
    status_code = int(getattr(resp, "status_code", 0) or 0)
    err_obj = getattr(resp, "error", None)
    if err_obj:
        return _truncate_text(str(err_obj))

    body = ""
    try:
        body = str(getattr(resp, "text", "") or "")
    except Exception:  # noqa: BLE001
        body = ""
    body = body.strip()
    if body:
        return _truncate_text(body)
    return f"http_{status_code}"


def _parse_sse_stream(resp) -> tuple[bool, float | None, int, str | None]:
    t0 = time.perf_counter()
    ttft_ms: float | None = None
    out_chars = 0
    done = False
    err: str | None = None
    for raw in resp.iter_lines(decode_unicode=True):
        if raw is None:
            continue
        line = str(raw).strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            done = True
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            err = "invalid_json_chunk"
            continue
        choices = chunk.get("choices") or []
        for item in choices:
            delta = (item or {}).get("delta") or {}
            content = delta.get("content")
            reasoning = delta.get("reasoning_content")
            for piece in (reasoning, content):
                if isinstance(piece, str) and piece:
                    out_chars += len(piece)
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000.0
    return done, ttft_ms, out_chars, err


class LlmStreamUser(HttpUser):
    wait_time = constant(0)

    def on_start(self) -> None:
        prompt_list = os.environ.get("LLM_PROMPTS_JSON", "")
        try:
            parsed = json.loads(prompt_list) if prompt_list else [DEFAULT_PROMPT]
        except json.JSONDecodeError:
            parsed = [DEFAULT_PROMPT]
        self.prompts = [str(x) for x in parsed if str(x).strip()] or [DEFAULT_PROMPT]

    @task
    def chat_stream(self) -> None:
        endpoint = os.environ.get("LLM_ENDPOINT", DEFAULT_ENDPOINT)
        request_name = os.environ.get("LLM_REQUEST_NAME", "chat_stream")
        timeout_sec = _to_float(os.environ.get("LLM_TIMEOUT_SECONDS", "120")) or 120.0
        prompt = random.choice(self.prompts)
        payload = _build_payload(prompt)
        headers = _build_headers()

        elapsed_ms = 0.0
        ttft_ms: float | None = None
        out_chars = 0
        ok = False
        err: str | None = None
        t0 = time.perf_counter()
        with self.client.post(
            endpoint,
            json=payload,
            headers=headers,
            timeout=timeout_sec,
            stream=True,
            catch_response=True,
            name=request_name,
        ) as resp:
            try:
                status_code = int(getattr(resp, "status_code", 0) or 0)
                if status_code != 200:
                    detail = _response_error_detail(resp)
                    if detail.startswith(f"http_{status_code}"):
                        err = detail
                    else:
                        err = f"http_{status_code}: {detail}"
                    resp.failure(err)
                else:
                    done, ttft_ms, out_chars, stream_err = _parse_sse_stream(resp)
                    if stream_err and not done:
                        err = stream_err
                    if done:
                        ok = True
                        resp.success()
                    else:
                        if err is None:
                            err = "missing_done"
                        resp.failure(err)
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
                resp.failure(err)
            finally:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

        with _metrics_lock:
            _metrics_rows.append(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "ok": ok,
                    "error": err,
                    "elapsed_ms": round(elapsed_ms, 4),
                    "ttft_ms": round(ttft_ms, 4) if ttft_ms is not None else None,
                    "output_chars": out_chars,
                }
            )


@events.test_stop.add_listener
def _on_test_stop(environment, **kwargs) -> None:  # noqa: ARG001
    metrics_csv = os.environ.get("LLM_METRICS_CSV", "")
    if not metrics_csv:
        return
    rows = []
    with _metrics_lock:
        rows = list(_metrics_rows)
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(metrics_csv)), exist_ok=True)
    with open(metrics_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamp", "ok", "error", "elapsed_ms", "ttft_ms", "output_chars"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ----------------------------- Runner side -----------------------------


def parse_locust_stats(stats_csv_path: str) -> dict[str, Any]:
    if not os.path.exists(stats_csv_path):
        return {}
    with open(stats_csv_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    agg = None
    for row in rows:
        if (row.get("Name") or "") == "Aggregated":
            agg = row
            break
    if agg is None:
        agg = rows[-1]

    req = int(float(agg.get("Request Count") or 0))
    fail = int(float(agg.get("Failure Count") or 0))
    return {
        "request_count": req,
        "failure_count": fail,
        "success_rate_pct": round((req - fail) * 100.0 / req, 4) if req else None,
        "avg_response_ms": _to_float(agg.get("Average Response Time")),
        "p95_response_ms": _to_float(agg.get("95%")),
        "rps": _to_float(agg.get("Requests/s")),
    }


def parse_metrics_csv(metrics_csv_path: str) -> dict[str, Any]:
    if not os.path.exists(metrics_csv_path):
        return {}
    with open(metrics_csv_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    ttfts = [_to_float(r.get("ttft_ms")) for r in rows]
    ttft_vals = [x for x in ttfts if x is not None]
    output_chars = [_to_float(r.get("output_chars")) for r in rows]
    out_vals = [x for x in output_chars if x is not None]
    return {
        "samples": len(rows),
        "ttft_avg_ms": round(sum(ttft_vals) / len(ttft_vals), 4) if ttft_vals else None,
        "ttft_p95_ms": round(_percentile(ttft_vals, 95), 4) if ttft_vals else None,
        "output_chars_avg": round(sum(out_vals) / len(out_vals), 2) if out_vals else None,
    }


@dataclass
class LevelRunResult:
    concurrency: int
    ok: bool
    exit_code: int
    elapsed_sec: float
    command: str
    run_dir: str
    locust_stats_csv: str
    metrics_csv: str
    stdout_log: str
    stderr_log: str
    error: str | None
    locust_stats: dict[str, Any]
    metrics_stats: dict[str, Any]


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run streaming LLM pressure test using Locust (headless).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--url", default=DEFAULT_URL, help="LLM service base URL")
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="chat completions path")
    p.add_argument("--model", default=DEFAULT_MODEL, help="target model name")
    p.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=list(DEFAULT_CONCURRENCY),
        help="users per level",
    )
    p.add_argument("--spawn-rate", type=float, default=0.0, help="0 means use users value")
    p.add_argument("--run-time", default="30s", help="locust run time, e.g. 30s, 2m")
    p.add_argument("--stop-timeout", type=int, default=10)
    p.add_argument("--cooldown", type=float, default=3.0)
    p.add_argument("--timeout-seconds", type=float, default=180.0, help="single request timeout")
    p.add_argument("--request-name", default="chat_stream")

    p.add_argument("--prompt-file", default="", help="txt/md/csv prompt file")
    p.add_argument("--prompt", default=DEFAULT_PROMPT, help="single prompt fallback")

    p.add_argument("--api-key", default=API_KEY, help="auth secret (default: API_KEY in script)")
    p.add_argument("--api-key-env", default="SERVYOU_APP_ID")
    p.add_argument("--auth-mode", choices=("raw", "bearer", "none"), default="raw")
    p.add_argument("--add-accept-sse", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--app-id", default="", help="add appId to payload")
    p.add_argument(
        "--app-id-env",
        default="SERVYOU_APP_ID",
        help="fallback env var for appId payload",
    )
    p.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--chat-template-enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)

    p.add_argument(
        "--headers-json",
        default="",
        help='extra headers JSON object, e.g. {"X-App":"foo"}',
    )
    p.add_argument("--out-dir", default="", help="output directory")
    p.add_argument("--report-prefix", default="", help="report filename prefix")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--exit-code-on-error", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def run_one_level(
    args: argparse.Namespace,
    *,
    concurrency: int,
    run_dir: str,
    prompts: list[str],
    script_path: str,
) -> LevelRunResult:
    os.makedirs(run_dir, exist_ok=True)
    locust_csv_prefix = os.path.join(run_dir, "locust")
    locust_stats_csv = f"{locust_csv_prefix}_stats.csv"
    metrics_csv = os.path.join(run_dir, "stream_metrics.csv")
    stdout_log = os.path.join(run_dir, "stdout.log")
    stderr_log = os.path.join(run_dir, "stderr.log")

    api_key = _resolve_api_key(args.api_key, args.api_key_env)
    app_id = _resolve_app_id(args.app_id, env_name=args.app_id_env, api_key=api_key)
    spawn_rate = args.spawn_rate if args.spawn_rate > 0 else float(concurrency)
    if spawn_rate <= 0:
        spawn_rate = 1.0

    env = dict(os.environ)
    env["LLM_ENDPOINT"] = args.endpoint
    env["LLM_MODEL"] = args.model
    env["LLM_PROMPTS_JSON"] = json.dumps(prompts, ensure_ascii=False)
    env["LLM_API_KEY"] = api_key
    env["LLM_AUTH_MODE"] = args.auth_mode
    env["LLM_METRICS_CSV"] = metrics_csv
    env["LLM_TIMEOUT_SECONDS"] = str(args.timeout_seconds)
    env["LLM_REQUEST_NAME"] = args.request_name
    env["LLM_ADD_ACCEPT_SSE"] = "true" if args.add_accept_sse else "false"
    env["LLM_APP_ID"] = app_id
    env["LLM_ENABLE_THINKING"] = "true" if args.enable_thinking else "false"
    env["LLM_CHAT_TEMPLATE_ENABLE_THINKING"] = (
        "true" if args.chat_template_enable_thinking else "false"
    )
    env["LLM_MAX_TOKENS"] = str(args.max_tokens) if args.max_tokens is not None else ""
    env["LLM_TEMPERATURE"] = str(args.temperature) if args.temperature is not None else ""
    env["LLM_TOP_P"] = str(args.top_p) if args.top_p is not None else ""
    env["LLM_HEADERS_JSON"] = args.headers_json

    cmd = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        script_path,
        "--headless",
        "--host",
        args.url,
        "--users",
        str(concurrency),
        "--spawn-rate",
        str(spawn_rate),
        "--run-time",
        args.run_time,
        "--stop-timeout",
        str(args.stop_timeout),
        "--csv",
        locust_csv_prefix,
        "--only-summary",
        "--exit-code-on-error",
        str(args.exit_code_on_error),
    ]

    t0 = time.perf_counter()
    if args.dry_run:
        return LevelRunResult(
            concurrency=concurrency,
            ok=True,
            exit_code=0,
            elapsed_sec=0.0,
            command=_redact_command(cmd),
            run_dir=run_dir,
            locust_stats_csv=locust_stats_csv,
            metrics_csv=metrics_csv,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            error=None,
            locust_stats={},
            metrics_stats={},
        )

    cp = subprocess.run(
        cmd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = time.perf_counter() - t0

    with open(stdout_log, "w", encoding="utf-8") as f:
        f.write(cp.stdout or "")
    with open(stderr_log, "w", encoding="utf-8") as f:
        f.write(cp.stderr or "")

    error: str | None = None
    if cp.returncode != 0:
        error = f"locust exit {cp.returncode}"
    if not os.path.exists(locust_stats_csv):
        error = (error + "; " if error else "") + "missing locust_stats.csv"
    if not os.path.exists(metrics_csv):
        error = (error + "; " if error else "") + "missing stream_metrics.csv"

    locust_stats = parse_locust_stats(locust_stats_csv)
    metrics_stats = parse_metrics_csv(metrics_csv)
    return LevelRunResult(
        concurrency=concurrency,
        ok=error is None,
        exit_code=cp.returncode,
        elapsed_sec=elapsed,
        command=_redact_command(cmd),
        run_dir=run_dir,
        locust_stats_csv=locust_stats_csv,
        metrics_csv=metrics_csv,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
        error=error,
        locust_stats=locust_stats,
        metrics_stats=metrics_stats,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    levels = sorted(set(args.concurrency))
    if any(x <= 0 for x in levels):
        safe_print("error: concurrency must be positive integers")
        return 2
    api_key = _resolve_api_key(args.api_key, args.api_key_env)
    app_id = _resolve_app_id(args.app_id, env_name=args.app_id_env, api_key=api_key)
    if "/mudgate/api/llm/" in args.endpoint and args.auth_mode != "none" and not api_key:
        safe_print("error: missing api key for mudgate endpoint (set API_KEY in script)")
        return 2

    script_path = os.path.abspath(__file__)
    out_dir = args.out_dir or os.path.dirname(script_path)
    run_ts = _now_ts()
    prefix = args.report_prefix or f"llm_locust_{run_ts}"
    run_root = os.path.join(out_dir, f"{prefix}_artifacts")
    summary_csv = os.path.join(out_dir, f"{prefix}_summary.csv")
    details_csv = os.path.join(out_dir, f"{prefix}_details.csv")
    os.makedirs(run_root, exist_ok=True)

    prompts = [args.prompt]
    if args.prompt_file:
        prompts = _load_prompts(args.prompt_file)
    if not prompts:
        prompts = [DEFAULT_PROMPT]

    safe_print("=" * 80)
    safe_print(f"framework         : locust")
    safe_print(f"url               : {args.url}")
    safe_print(f"endpoint          : {args.endpoint}")
    safe_print(f"model             : {args.model}")
    safe_print(f"concurrency levels: {levels}")
    safe_print(f"run_time          : {args.run_time}")
    safe_print(f"prompts           : {len(prompts)}")
    safe_print(f"output root       : {run_root}")
    safe_print("=" * 80)

    all_results: list[LevelRunResult] = []
    all_ok = True
    try:
        for idx, level in enumerate(levels):
            safe_print(f"\n>>> level={level}")
            run_dir = os.path.join(run_root, f"c{level}")
            result = run_one_level(
                args,
                concurrency=level,
                run_dir=run_dir,
                prompts=prompts,
                script_path=script_path,
            )
            all_results.append(result)
            all_ok = all_ok and result.ok
            safe_print(f"command: {result.command}")
            if result.ok:
                safe_print(
                    "ok:"
                    f" elapsed={result.elapsed_sec:.2f}s"
                    f" req={result.locust_stats.get('request_count')}"
                    f" fail={result.locust_stats.get('failure_count')}"
                    f" ttft_avg_ms={result.metrics_stats.get('ttft_avg_ms')}"
                    f" rps={result.locust_stats.get('rps')}"
                )
            else:
                safe_print(
                    f"failed: exit={result.exit_code} error={result.error} "
                    f"stderr={result.stderr_log}"
                )
                if args.fail_fast:
                    break
            if idx < len(levels) - 1 and args.cooldown > 0 and not args.dry_run:
                safe_print(f"cooldown {args.cooldown}s ...")
                time.sleep(args.cooldown)
    except KeyboardInterrupt:
        safe_print("\ninterrupted by user")
        return 130

    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for r in all_results:
        row = {
            "concurrency": r.concurrency,
            "ok": r.ok,
            "exit_code": r.exit_code,
            "elapsed_sec": round(r.elapsed_sec, 4),
            "error": r.error,
            "run_dir": r.run_dir,
            "locust_stats_csv": r.locust_stats_csv,
            "metrics_csv": r.metrics_csv,
            "stdout_log": r.stdout_log,
            "stderr_log": r.stderr_log,
        }
        row.update(r.locust_stats)
        row.update(r.metrics_stats)
        summary_rows.append(row)

        detail_rows.append(
            {
                "concurrency": r.concurrency,
                "command": r.command,
                "stdout_log": r.stdout_log,
                "stderr_log": r.stderr_log,
                "error": r.error,
            }
        )

    _write_csv(summary_csv, summary_rows)
    _write_csv(details_csv, detail_rows)
    safe_print(f"\nsummary: {summary_csv}")
    safe_print(f"details: {details_csv}")
    safe_print("done.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
