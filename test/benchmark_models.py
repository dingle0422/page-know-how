"""
Qwen3.5-122B-A10B vs Qwen3.6-35B-A3B 对比基准脚本

- 阶段 1: LLM 直连对比
    对每个 vendor，用 短(~100字) / 中(~1000字) / 长(~3000字) 三种 prompt 串行各跑 N 次，
    记录单次端到端响应耗时（含网络）。
- 阶段 2: CLI 端到端对比
    通过 subprocess 调用 `python main.py reason --single-question ... --knowledge-dir ... --vendor xxx`，
    每个 vendor 跑 K 次，记录端到端 wall time。

输出：控制台表格 + 落盘 JSON。

用法（PowerShell）:
    python test/benchmark_models.py --llm-rounds 5 --cli-rounds 3
    # 仅跑某一阶段:
    python test/benchmark_models.py --skip-cli
    python test/benchmark_models.py --skip-llm
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from llm.client import chat  # noqa: E402  (路径调整后才能导入)


VENDORS: list[dict[str, str]] = [
    {"vendor": "qwen3.5-122b-a10b", "model": "Qwen3.5-122B-A10B"},
    {"vendor": "qwen3.6-35b-a3b", "model": "Qwen/Qwen3.6-35B-A3B"},
]

DEFAULT_KNOWLEDGE_DIR = os.path.join(
    PROJECT_ROOT, "page_knowledge", "农产品精简版V3.0-new_20260414165344"
)
DEFAULT_E2E_QUESTION = "批发要售圣女果是否免征增值税"


# ---------------- prompt 构造 ----------------

_BASE_FACT = (
    "假设你是一名增值税顾问。下面给出政策片段：纳税人销售或者进口下列货物，"
    "按照低税率征收增值税：粮食、食用植物油、自来水、暖气、冷气、热水、煤气、"
    "石油液化气、天然气、沼气、居民用煤炭制品、图书、报纸、杂志、饲料、化肥、"
    "农药、农机、农膜、农业产品。农业生产者销售的自产农业产品免征增值税。"
)


def build_prompt(target_chars: int) -> str:
    """构造一个目标长度的 prompt，并提一个相同的问题，便于公平对比。"""
    question = (
        "请基于上述政策回答：批发环节销售自产圣女果是否可以免征增值税？"
        "请给出适用条款依据并简要说明理由。"
    )

    # 反复填充政策片段，直到达到目标长度
    body = _BASE_FACT
    while len(body) < target_chars:
        body += "\n补充说明：" + _BASE_FACT
    body = body[:target_chars]
    return body + "\n\n" + question


# ---------------- 结果数据结构 ----------------

@dataclass
class LLMTrial:
    vendor: str
    model: str
    size_label: str
    prompt_chars: int
    round_idx: int
    elapsed_sec: float
    ok: bool
    error: str = ""
    answer_preview: str = ""
    answer_chars: int = 0


@dataclass
class CLITrial:
    vendor: str
    model: str
    round_idx: int
    elapsed_sec: float
    ok: bool
    return_code: int
    stdout_tail: str = ""
    stderr_tail: str = ""


@dataclass
class Stat:
    n: int
    success: int
    fail: int
    min: float = 0.0
    avg: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    max: float = 0.0


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


def stats_of(elapses_ok: list[float], total: int) -> Stat:
    if not elapses_ok:
        return Stat(n=total, success=0, fail=total)
    return Stat(
        n=total,
        success=len(elapses_ok),
        fail=total - len(elapses_ok),
        min=min(elapses_ok),
        avg=statistics.mean(elapses_ok),
        p50=percentile(elapses_ok, 50),
        p95=percentile(elapses_ok, 95),
        max=max(elapses_ok),
    )


# ---------------- 阶段 1: LLM 直连 ----------------

def run_llm_phase(
    rounds: int,
    sizes: list[tuple[str, int]],
    enable_thinking: bool,
) -> list[LLMTrial]:
    trials: list[LLMTrial] = []
    print("\n" + "=" * 80)
    print(f"阶段 1：LLM 直连对比  rounds={rounds}  sizes={sizes}  enable_thinking={enable_thinking}")
    print("=" * 80)

    for size_label, target_chars in sizes:
        prompt = build_prompt(target_chars)
        actual = len(prompt)
        print(f"\n--- 长度档：{size_label}  (实际 {actual} 字) ---")
        for v in VENDORS:
            for r in range(1, rounds + 1):
                t0 = time.perf_counter()
                ok = False
                err = ""
                answer = ""
                try:
                    answer = chat(
                        messages=prompt,
                        vendor=v["vendor"],
                        model=v["model"],
                        enable_thinking=enable_thinking,
                    )
                    ok = True
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {e}"
                elapsed = time.perf_counter() - t0
                preview = answer.replace("\n", " ")[:80] if answer else ""
                trial = LLMTrial(
                    vendor=v["vendor"],
                    model=v["model"],
                    size_label=size_label,
                    prompt_chars=actual,
                    round_idx=r,
                    elapsed_sec=round(elapsed, 3),
                    ok=ok,
                    error=err,
                    answer_preview=preview,
                    answer_chars=len(answer) if answer else 0,
                )
                trials.append(trial)
                tag = "OK " if ok else "ERR"
                extra = (
                    f" ans_chars={trial.answer_chars}"
                    if ok else f" err={err[:120]}"
                )
                print(
                    f"  [{v['vendor']:<20s}] round {r}/{rounds} "
                    f"{tag} elapsed={elapsed:6.2f}s{extra}"
                )
    return trials


def summarize_llm(trials: list[LLMTrial]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    sizes = sorted({(t.size_label, t.prompt_chars) for t in trials}, key=lambda x: x[1])
    for size_label, _chars in sizes:
        summary[size_label] = {}
        for v in VENDORS:
            subset = [t for t in trials if t.size_label == size_label and t.vendor == v["vendor"]]
            ok_elapses = [t.elapsed_sec for t in subset if t.ok]
            avg_ans = (
                round(statistics.mean([t.answer_chars for t in subset if t.ok]), 1)
                if ok_elapses else 0
            )
            s = stats_of(ok_elapses, total=len(subset))
            summary[size_label][v["vendor"]] = {
                **asdict(s),
                "avg_answer_chars": avg_ans,
            }
    return summary


def print_llm_table(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 100)
    print("LLM 直连汇总（单位：秒，仅统计成功样本）")
    print("=" * 100)
    header = (
        f"{'size':<8} {'vendor':<22} {'n':>3} {'ok':>3} {'fail':>4} "
        f"{'min':>7} {'avg':>7} {'p50':>7} {'p95':>7} {'max':>7} {'ans_chars':>10}"
    )
    print(header)
    print("-" * len(header))
    for size_label, vendors_map in summary.items():
        for vendor, s in vendors_map.items():
            print(
                f"{size_label:<8} {vendor:<22} "
                f"{s['n']:>3} {s['success']:>3} {s['fail']:>4} "
                f"{s['min']:>7.2f} {s['avg']:>7.2f} {s['p50']:>7.2f} "
                f"{s['p95']:>7.2f} {s['max']:>7.2f} {s['avg_answer_chars']:>10}"
            )


# ---------------- 阶段 2: CLI 端到端 ----------------

def run_cli_phase(
    rounds: int,
    knowledge_dir: str,
    question: str,
    extra_args: list[str],
) -> list[CLITrial]:
    trials: list[CLITrial] = []
    print("\n" + "=" * 80)
    print(f"阶段 2：CLI 端到端对比  rounds/vendor={rounds}")
    print(f"knowledge_dir = {knowledge_dir}")
    print(f"question      = {question}")
    print(f"extra_args    = {extra_args}")
    print("=" * 80)

    if not os.path.isdir(knowledge_dir):
        print(f"!! knowledge_dir 不存在，跳过 CLI 阶段: {knowledge_dir}")
        return trials

    main_py = os.path.join(PROJECT_ROOT, "main.py")
    for v in VENDORS:
        for r in range(1, rounds + 1):
            cmd = [
                sys.executable, main_py, "reason",
                "--single-question", question,
                "--knowledge-dir", knowledge_dir,
                "--vendor", v["vendor"],
                "--model", v["model"],
                *extra_args,
            ]
            print(f"\n  [{v['vendor']}] round {r}/{rounds} -> 启动 subprocess ...")
            t0 = time.perf_counter()
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=1800,
                )
                ok = proc.returncode == 0
                rc = proc.returncode
                stdout_tail = (proc.stdout or "")[-500:]
                stderr_tail = (proc.stderr or "")[-500:]
            except subprocess.TimeoutExpired as e:
                ok = False
                rc = -1
                stdout_tail = (e.stdout or "")[-500:] if isinstance(e.stdout, str) else ""
                stderr_tail = f"TIMEOUT: {e}"
            elapsed = time.perf_counter() - t0
            tag = "OK " if ok else "ERR"
            print(f"    {tag} rc={rc} elapsed={elapsed:.2f}s")
            if not ok:
                print(f"    stderr_tail: {stderr_tail[:200]}")
            trials.append(CLITrial(
                vendor=v["vendor"],
                model=v["model"],
                round_idx=r,
                elapsed_sec=round(elapsed, 3),
                ok=ok,
                return_code=rc,
                stdout_tail=stdout_tail,
                stderr_tail=stderr_tail,
            ))
    return trials


def summarize_cli(trials: list[CLITrial]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for v in VENDORS:
        subset = [t for t in trials if t.vendor == v["vendor"]]
        ok_elapses = [t.elapsed_sec for t in subset if t.ok]
        s = stats_of(ok_elapses, total=len(subset))
        summary[v["vendor"]] = asdict(s)
    return summary


def print_cli_table(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 100)
    print("CLI 端到端汇总（单位：秒，含进程冷启动 + 推理 + 退出）")
    print("=" * 100)
    header = (
        f"{'vendor':<22} {'n':>3} {'ok':>3} {'fail':>4} "
        f"{'min':>8} {'avg':>8} {'p50':>8} {'p95':>8} {'max':>8}"
    )
    print(header)
    print("-" * len(header))
    for vendor, s in summary.items():
        print(
            f"{vendor:<22} {s['n']:>3} {s['success']:>3} {s['fail']:>4} "
            f"{s['min']:>8.2f} {s['avg']:>8.2f} {s['p50']:>8.2f} "
            f"{s['p95']:>8.2f} {s['max']:>8.2f}"
        )


# ---------------- main ----------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="对比 qwen3.5-122b-a10b 与 qwen3.6-35b-a3b 的响应时间",
    )
    p.add_argument("--llm-rounds", type=int, default=5, help="LLM 直连每档每模型测试次数")
    p.add_argument("--cli-rounds", type=int, default=3, help="CLI 端到端每模型测试次数")
    p.add_argument(
        "--sizes", default="short:100,medium:1000,long:3000",
        help="LLM 直连测试 prompt 长度档，格式 label:chars,...",
    )
    p.add_argument("--enable-thinking", action="store_true", default=False)
    p.add_argument("--skip-llm", action="store_true", default=False)
    p.add_argument("--skip-cli", action="store_true", default=False)
    p.add_argument("--knowledge-dir", default=DEFAULT_KNOWLEDGE_DIR)
    p.add_argument("--question", default=DEFAULT_E2E_QUESTION)
    p.add_argument(
        "--cli-extra", default="",
        help="额外透传给 main.py reason 的参数（用空格分隔），"
             "默认参考 stress_reason.py 默认值添加常用开关",
    )
    p.add_argument("--out", default="", help="结果 JSON 输出路径")
    return p.parse_args()


def parse_sizes(spec: str) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"sizes 段非法: {chunk}")
        label, chars = chunk.split(":", 1)
        out.append((label.strip(), int(chars.strip())))
    return out


def main() -> int:
    args = parse_args()
    sizes = parse_sizes(args.sizes)

    if args.cli_extra.strip():
        cli_extra = args.cli_extra.split()
    else:
        cli_extra = [
            "--retrieval-mode",
            "--check-pitfalls",
            "--summary-batch-size", "3",
            "--chunk-size", "3000",
            "--summary-clean-answer",
            "--think-mode",
        ]

    started = datetime.now().isoformat(timespec="seconds")
    overall_t0 = time.perf_counter()

    llm_trials: list[LLMTrial] = []
    cli_trials: list[CLITrial] = []

    if not args.skip_llm:
        llm_trials = run_llm_phase(
            rounds=args.llm_rounds,
            sizes=sizes,
            enable_thinking=args.enable_thinking,
        )
        llm_summary = summarize_llm(llm_trials)
        print_llm_table(llm_summary)
    else:
        llm_summary = {}
        print("\n[已跳过] LLM 直连阶段")

    if not args.skip_cli:
        cli_trials = run_cli_phase(
            rounds=args.cli_rounds,
            knowledge_dir=args.knowledge_dir,
            question=args.question,
            extra_args=cli_extra,
        )
        cli_summary = summarize_cli(cli_trials)
        print_cli_table(cli_summary)
    else:
        cli_summary = {}
        print("\n[已跳过] CLI 端到端阶段")

    total_wall = time.perf_counter() - overall_t0

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"benchmark_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    payload = {
        "started_at": started,
        "total_wall_sec": round(total_wall, 3),
        "config": {
            "llm_rounds": args.llm_rounds,
            "cli_rounds": args.cli_rounds,
            "sizes": sizes,
            "enable_thinking": args.enable_thinking,
            "knowledge_dir": args.knowledge_dir,
            "question": args.question,
            "cli_extra": cli_extra,
            "vendors": VENDORS,
        },
        "llm_summary": llm_summary,
        "cli_summary": cli_summary,
        "llm_trials": [asdict(t) for t in llm_trials],
        "cli_trials": [asdict(t) for t in cli_trials],
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已写入: {out_path}")
    print(f"总耗时: {total_wall:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
