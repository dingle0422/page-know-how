"""
针对 ncp 结果 CSV 中失败的行进行重跑，并把新答案写回原文件。

判定失败的标准：biz_status != "200" 或 error 非空。
重跑时会带串行重试（默认 3 次，指数退避），降低 RemoteDisconnected 概率。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime

from run_ncp_reason import call_reason, OUTPUT_FIELDS

DEFAULT_RESULT_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ncp_test_0423_result.csv"
)
DEFAULT_URL = "http://mlp.paas.dc.servyou-it.com/kh-server/api/reason"
DEFAULT_POLICY_ID = "KH1493204307733168128_20260423163645"


def is_failed(row: dict, include_empty_kh: bool = False) -> bool:
    biz = (row.get("biz_status") or "").strip()
    err = (row.get("error") or "").strip()
    if biz != "200" or err:
        return True
    if include_empty_kh:
        kh = (row.get("khObj") or "").strip()
        if kh in ("", "{}"):
            return True
    return False


def call_with_retry(
    url: str,
    policy_id: str,
    question: str,
    timeout: float,
    retries: int,
    backoff: float,
    require_non_empty_kh: bool = False,
) -> dict:
    last: dict = {}
    for attempt in range(1, retries + 1):
        res = call_reason(url, policy_id, question, timeout)
        biz_ok = res.get("biz_status") == 200 and not res.get("error")
        kh = (res.get("khObj") or "").strip()
        kh_ok = kh not in ("", "{}") if require_non_empty_kh else True
        ok = biz_ok and kh_ok
        last = res
        preview = (res.get("answer") or res.get("error") or "").replace("\n", " ")[:60]
        kh_tag = "" if not require_non_empty_kh else f" kh_empty={not kh_ok}"
        print(
            f"  attempt={attempt}/{retries} ok={ok} biz={res.get('biz_status')} "
            f"http={res.get('http_status')} {res.get('elapsed_sec')}s{kh_tag} | {preview}",
            flush=True,
        )
        if ok:
            return res
        if attempt < retries:
            sleep_s = backoff * (2 ** (attempt - 1))
            print(f"  -> {sleep_s:.1f}s 后重试", flush=True)
            time.sleep(sleep_s)
    return last


def main() -> int:
    parser = argparse.ArgumentParser(description="重跑 ncp 结果 CSV 中的失败行")
    parser.add_argument("--csv", default=DEFAULT_RESULT_CSV, help="待修补的结果 CSV 路径")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--policy-id", default=DEFAULT_POLICY_ID)
    parser.add_argument("--timeout", type=float, default=1200.0, help="单次请求超时(s)")
    parser.add_argument("--retries", type=int, default=3, help="单题最大尝试次数")
    parser.add_argument("--backoff", type=float, default=5.0, help="重试初始等待(s)，指数退避")
    parser.add_argument(
        "--include-empty-kh",
        action="store_true",
        help="将 khObj 为空（'' 或 '{}'）的行也视为待重跑",
    )
    parser.add_argument(
        "--require-non-empty-kh",
        action="store_true",
        help="重试判定 ok 时要求 khObj 非空（避免反复返回空 khObj 仍算成功）",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"[ERROR] 找不到结果文件: {args.csv}")
        return 1

    with open(args.csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or OUTPUT_FIELDS
        rows = list(reader)

    failed_idx = [i for i, r in enumerate(rows) if is_failed(r, args.include_empty_kh)]
    if not failed_idx:
        print("[INFO] 未发现失败行，无需重跑")
        return 0

    print(
        f"[INFO] 总行数={len(rows)}, 失败={len(failed_idx)}, "
        f"timeout={args.timeout}s, retries={args.retries}"
    )
    for i in failed_idx:
        print(f"  - idx={rows[i].get('idx')}: {rows[i].get('question','')[:60]}")

    fixed = 0
    for n, i in enumerate(failed_idx, start=1):
        row = rows[i]
        question = row.get("question", "")
        print(f"\n[{n}/{len(failed_idx)}] 重跑 idx={row.get('idx')}: {question[:60]}", flush=True)
        res = call_with_retry(
            args.url,
            args.policy_id,
            question,
            args.timeout,
            args.retries,
            args.backoff,
            require_non_empty_kh=args.require_non_empty_kh,
        )
        new_row = {
            "idx": row.get("idx"),
            "question": question,
            "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **res,
        }
        for k in OUTPUT_FIELDS:
            row[k] = new_row.get(k, row.get(k, ""))
        biz_ok = res.get("biz_status") == 200 and not res.get("error")
        kh = (res.get("khObj") or "").strip()
        kh_ok = kh not in ("", "{}") if args.require_non_empty_kh else True
        if biz_ok and kh_ok:
            fixed += 1

        with open(args.csv, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  -> 已写回 {args.csv}", flush=True)

    print(
        f"\n[DONE] 失败={len(failed_idx)} 修复成功={fixed} "
        f"仍失败={len(failed_idx) - fixed}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
