"""端到端验证：think_mode + 新 JSON 解析逻辑后，
_run_reasoning 是否能正确分离 analysis -> think、concise_answer -> answer。
"""

import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tmp_test_think_json")

from app import _run_reasoning, _split_analysis_concise_answer  # noqa: E402

KNOWLEDGE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "page_knowledge",
    "农产品精简版V3.0-new_20260414165344",
)


def _check_e2e(label: str, result: dict) -> dict:
    print()
    print("=" * 80)
    print(f"[CASE-E2E] {label}")
    print("=" * 80)
    think = result.get("think", "")
    answer = result.get("answer", "")
    print(f"think len  = {len(think)}")
    print(f"answer len = {len(answer)}")
    print("--- think (analysis) 预览 ---")
    print(think[:300] + ("\n  ...(truncated)" if len(think) > 300 else ""))
    print("--- answer (concise_answer) ---")
    print(answer)

    report = {
        "label": label,
        "think_len": len(think),
        "answer_len": len(answer),
        "think_nonempty": bool(think),
        "answer_nonempty": bool(answer),
        # 期望：think 是较长的客服回答(≤500)，answer 是简短核心结论
        "looks_correct_split": bool(think) and bool(answer) and len(think) > len(answer),
    }
    return report


def _check_unit():
    """单元测试 _split_analysis_concise_answer 的各种异常分支"""
    print("\n" + "=" * 80)
    print("[UNIT] _split_analysis_concise_answer 容错分支")
    print("=" * 80)

    cases = [
        ("正常 JSON",
         '{"analysis": "完整客服回答 abc", "concise_answer": "核心结论 xyz"}',
         ("完整客服回答 abc", "核心结论 xyz")),
        ("被 markdown ```json 代码块包裹",
         '```json\n{"analysis": "A", "concise_answer": "B"}\n```',
         ("A", "B")),
        ("被 ``` 代码块（无 json 标识）包裹",
         '```\n{"analysis": "A2", "concise_answer": "B2"}\n```',
         ("A2", "B2")),
        ("前后混入了说明文字",
         '好的，下面是输出：{"analysis": "A3", "concise_answer": "B3"} 请查收',
         ("A3", "B3")),
        ("非严格 JSON：尾随逗号 + 单引号 + 注释（json5 兜底）",
         "{'analysis': 'A4', 'concise_answer': 'B4',  // trailing\n}",
         ("A4", "B4")),
        ("缺 concise_answer 字段",
         '{"analysis": "only A"}',
         ("only A", "")),
        ("缺 analysis 字段",
         '{"concise_answer": "only B"}',
         ("", "only B")),
        ("两个字段都没有",
         '{"foo": "bar"}',
         ("", '{"foo": "bar"}')),
        ("完全无法解析（非 JSON）",
         "这只是普通自然语言文本，不是 JSON",
         ("", "这只是普通自然语言文本，不是 JSON")),
        ("think_mode=False 直接透传",
         "any text",
         ("", "any text")),
    ]

    all_ok = True
    for i, (name, raw, expected) in enumerate(cases):
        think_mode = name != "think_mode=False 直接透传"
        got = _split_analysis_concise_answer(raw, think_mode)
        ok = got == expected
        all_ok &= ok
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if not ok:
            print(f"    expected={expected!r}")
            print(f"    got     ={got!r}")
    print(f"[UNIT result] {'ALL PASS' if all_ok else 'HAS FAILURES'}")
    return all_ok


def run_case(label: str, **kwargs) -> dict:
    t0 = time.time()
    print(f"\n[RUN] {label} kwargs={ {k:v for k,v in kwargs.items() if k!='question'} }")
    try:
        result = _run_reasoning(
            knowledge_dir=KNOWLEDGE_DIR,
            version="v1",
            max_rounds=2,
            vendor="qwen3.5-122b-a10b",
            model="Qwen3.5-122B-A10B",
            enable_skills=False,
            summary_clean_answer=True,
            think_mode=True,
            **kwargs,
        )
    except Exception as e:
        logger.exception(f"[{label}] 运行异常")
        return {"label": label, "error": repr(e)}
    dt = time.time() - t0
    print(f"[RUN done] {label}, elapsed={dt:.1f}s")
    return _check_e2e(label, result)


def main():
    if not os.path.isdir(KNOWLEDGE_DIR):
        print(f"知识目录不存在: {KNOWLEDGE_DIR}")
        sys.exit(1)

    # 先跑容错单元测试（不调 LLM，毫秒级）
    unit_ok = _check_unit()

    # 再跑一条 LLM 端到端
    question = "购买农产品时，可以抵扣的进项税额怎么计算？"
    e2e_report = run_case(
        "retrieval_no_batch",
        question=question,
        retrieval_mode=True,
        summary_batch_size=0,
    )

    print("\n\n========== SUMMARY ==========")
    print(f"unit_test_all_pass = {unit_ok}")
    print(json.dumps(e2e_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
