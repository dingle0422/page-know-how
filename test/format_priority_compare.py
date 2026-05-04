"""
格式优先级对照测试：在生产链路（_batch_final_merge）的真实 prompt 形态下，
比较 deepseek-v3.2 与 qwen3.6-35b-a3b 对 JSON / HTML 两种输出 schema 的首轮遵循度。

为何要做：
    最终节点要求模型严格输出 {think, answer}，目前默认首选 JSON、兜底 HTML 重试。
    单点 baseline 已观察到 qwen3.6 在 HTML 模式下倾向于直接吐纯散文，但样本太少（N=1），
    需要多轮多问题确认是不是稳定现象，以决定是否要切换默认首选格式或按 vendor 自动切换。

实验设计：
    - 模型矩阵：deepseek-v3.2（vendor=aliyun）+ qwen3.6-35b-a3b（vendor 同名）
    - 输出格式：JSON / HTML（直接复用 prompts.py 中已拆好的 *_THINK_PROMPT / *_THINK_HTML_PROMPT）
    - 问题集：3 个真实税务问题，每个问题配套手工编写的 numbered batch_summaries
              （模拟 _batch_final_merge 上游交付的中间产物形态）
    - 轮数：每个 cell 默认 3 轮 → 3q × 2model × 2fmt × 3 = 36 次调用
    - enable_thinking=True，与生产 last_think 保持一致

度量：
    1. 首轮通过率（核心）：
       - JSON: _validate_think_answer_json 通过且 think/answer 都非空
       - HTML: _parse_think_answer_html 通过且 answer 非空
    2. think / answer 长度（字段是否被实质填充）
    3. 平均响应耗时
    4. reasoning_content 长度（thinking 模式下副产品）
    5. 失败样例 body 头部预览（用于诊断）

输出：
    - 终端打印汇总表 + 详细 cell 统计
    - 写一份完整日志到 tests_logs/format_priority_compare_<ts>.log

用法：
    python test/format_priority_compare.py                    # 默认 N=3 全量
    python test/format_priority_compare.py --rounds 1         # 连通性 sanity（4 次/q × 3q = 12 次）
    python test/format_priority_compare.py --rounds 5         # 更稳的 60 次
    python test/format_priority_compare.py --workers 4        # 4 路并发（默认 1 串行）
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# 让脚本能直接运行（不依赖 PYTHONPATH 设置）
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import re

from llm.client import chat
from utils.helpers import split_think_block
from reasoner.v2.prompts import (
    BATCH_MERGE_AND_CLEAN_THINK_PROMPT,
    BATCH_MERGE_AND_CLEAN_THINK_HTML_PROMPT,
    BATCH_MERGE_AND_CLEAN_THINK_SYSTEM_PROMPT,
    BATCH_MERGE_AND_CLEAN_THINK_HTML_SYSTEM_PROMPT,
    SUMMARY_ANSWER_SYSTEM_PROMPT,
)
from reasoner.v2.agent_graph import (
    _validate_think_answer_json,
    _parse_think_answer_html,
)


# ===== 实验性宽松解析：当生产解析器 _parse_think_answer_html 因 <answer> 缺闭合而失败时，
#       尝试匹配"<answer> 开标签到 EOF"作为答案。仅用于本测试脚本量化"加上未闭合兜底"的潜力。
_HTML_ANSWER_OPEN_TO_EOF_RE = re.compile(
    r"<answer\b[^>]*>(?P<body>.*)\Z", re.DOTALL | re.IGNORECASE
)
_HTML_THINK_BLOCK_RE = re.compile(
    r"<think\b[^>]*>(?P<body>.*?)</think\s*>", re.DOTALL | re.IGNORECASE
)


def _parse_think_answer_html_lenient(body: str, fallback_think: str) -> dict | None:
    """先试生产严格解析器；失败时再尝试'<answer> 未闭合到 EOF'兜底。"""
    strict = _parse_think_answer_html(body, fallback_think=fallback_think)
    if strict:
        return strict
    if not body:
        return None
    m = _HTML_ANSWER_OPEN_TO_EOF_RE.search(body)
    if not m:
        return None
    answer = (m.group("body") or "").strip()
    if not answer:
        return None
    tm = _HTML_THINK_BLOCK_RE.search(body)
    if tm and (tm.group("body") or "").strip():
        think = (tm.group("body") or "").strip()
    else:
        think = (fallback_think or "").strip()
    return {"think": think, "answer": answer}

logger = logging.getLogger("format_priority_compare")


# ===== 模型矩阵 =====
MODELS: list[dict[str, str]] = [
    # deepseek v3.2 走 aliyun mudgate 通道（与生产 main.py default 一致）
    {"display": "deepseek-v3.2", "vendor": "aliyun", "model": "deepseek-v3.2"},
    # qwen3.6 直连 mlp 自部署（client.py 里已有显式分支）
    {"display": "qwen3.6-35b-a3b", "vendor": "qwen3.6-35b-a3b", "model": "Qwen/Qwen3.6-35B-A3B"},
]

# ===== 测试问题与对应的 numbered batch_summaries =====
# 每条 summary 模拟 _batch_final_merge 上游产出的中间稿（章节路径 + 原文摘录 + 易错点）。
# 内容保持与生产数据形态一致：含【XXX > YYY】层级标签、引号包裹的原文片段、易错点提醒块。
QUESTIONS: list[dict] = [
    {
        "id": "Q1_农产品税率栏",
        "question": "自产农产品销售开票时税率栏应该怎么选？",
        "summaries": [
            (
                "**章节路径**：【3.2.2.1_数电票特定业务 / 3.2.2.3_开票规范】\n"
                "**原文摘录**：\n"
                "- 「针对'自产农产品销售'这一特定业务标签，开具时选择此标签，税率栏选'免税'。」\n"
                "- 「免税发票：税率栏显示'免税'，不得显示'0%'。」\n"
                "- 「农产品销售发票（数电票）：必须通过特定业务模块开具，并选择'自产农产品销售'标签，否则购买方无法抵扣。」\n"
                "**易错点提醒**：\n"
                "- '0%' 适用于出口业务，用于内销免税业务属于不合规发票，购买方无法抵扣。"
            ),
            (
                "**章节路径**：【3.2.2.4_开票常见问题及处理方式】\n"
                "**原文摘录**：\n"
                "- 「错误场景：自产农产品销售，开票时税率栏选了'0%'；正确处理方式：应作废或红冲，重新开具税率栏为'免税'的发票。」\n"
                "- 「销售方有自产能力，但未使用'自产农产品销售'标签开具数电票...需联系销售方红冲重开，必须使用特定业务标签，否则购买方无法勾选抵扣进项税。」\n"
                "**易错点提醒**：\n"
                "- 风险提示：未使用此标签开具的免税数电票，购买方无法抵扣进项税。\n"
                "- 农产品收购发票若错误开具为'0%'或'9%'，也需红冲后重开，且税率栏必须为'免税'。"
            ),
        ],
    },
    {
        "id": "Q2_即征即退",
        "question": "增值税即征即退是什么？需要满足哪些条件才能享受？",
        "summaries": [
            (
                "**章节路径**：【2.3_增值税优惠政策 / 2.3.4_即征即退】\n"
                "**原文摘录**：\n"
                "- 「即征即退：纳税人按照规定缴纳税款后，由税务机关按规定的程序审核确认后将所缴税款全部或部分退还。」\n"
                "- 「常见适用对象包括软件产品、动漫企业、资源综合利用产品、安置残疾人就业单位等。」\n"
                "- 「软件产品按 13% 税率征收增值税后，对实际税负超过 3% 的部分实行即征即退。」\n"
                "**易错点提醒**：\n"
                "- 即征即退不同于直接减免：必须先足额缴纳税款，才能进入退税流程。\n"
                "- 退税仅限于增值税本税，不退附加税费。"
            ),
            (
                "**章节路径**：【2.3.4.2_软件产品即征即退备案要件】\n"
                "**原文摘录**：\n"
                "- 「享受软件产品增值税即征即退政策的纳税人，应单独核算软件产品的销售额和应纳税额。」\n"
                "- 「未单独核算或核算不清的，不得享受增值税即征即退政策。」\n"
                "- 「需取得省级软件产业主管部门认可的软件检测机构出具的检测证明材料，并取得软件产业主管部门颁发的《软件产品登记证书》或著作权行政管理部门颁发的《计算机软件著作权登记证书》。」\n"
                "**易错点提醒**：\n"
                "- 嵌入式软件产品的销售额计算需将硬件部分剔除后单独计算。"
            ),
        ],
    },
    {
        "id": "Q3_个税年终汇算",
        "question": "个税年终汇算什么时候开始？补税和退税的处理方式有什么不同？",
        "summaries": [
            (
                "**章节路径**：【4.1_个人所得税 / 4.1.5_综合所得汇算清缴】\n"
                "**原文摘录**：\n"
                "- 「居民个人取得综合所得，需要在取得所得的次年 3 月 1 日至 6 月 30 日内办理汇算清缴。」\n"
                "- 「综合所得包括：工资薪金、劳务报酬、稿酬、特许权使用费四项所得。」\n"
                "- 「年度汇算应纳税款=（综合所得收入额-60000元-专项扣除-专项附加扣除-依法确定的其他扣除-公益慈善事业捐赠）×适用税率-速算扣除数」\n"
                "**易错点提醒**：\n"
                "- 不需要办理汇算的情形：年度汇算需补税但综合所得收入全年不超过 12 万元的；年度汇算需补税金额不超过 400 元的；已预缴税额与年度应纳税额一致或不申请年度汇算退税的。"
            ),
            (
                "**章节路径**：【4.1.5.2_汇算清缴退补税操作】\n"
                "**原文摘录**：\n"
                "- 「需要补税的：通过个人所得税 APP、电子税务局、办税服务厅等渠道办理。可选择银行卡、第三方支付等方式缴款。」\n"
                "- 「申请退税的：纳税人需要提供本人在中国境内开设的符合条件的银行账户。退税款将由税务机关审核确认后退至该账户。」\n"
                "- 「未按期办理汇算的法律责任：需补税未补的，加收滞纳金并依法追究法律责任；申请退税无强制时间，但建议在汇算期内完成。」\n"
                "**易错点提醒**：\n"
                "- 退税账户必须是纳税人本人在中国境内开设的银行账户，不能用他人账户。\n"
                "- 补税未按期缴纳将按日加收万分之五的滞纳金。"
            ),
        ],
    },
]


@dataclass
class CallResult:
    question_id: str
    model_display: str
    fmt: str  # "json" / "html"
    round_idx: int

    elapsed_ms: int = 0
    raw_len: int = 0
    body_len: int = 0
    reasoning_len: int = 0
    has_think_prefix: bool = False  # body 中是否有 <think> 前缀

    parse_ok: bool = False  # 严格模式（生产解析器）
    parse_reason: str = ""
    think_len: int = 0
    answer_len: int = 0

    parse_ok_lenient: bool = False  # 实验：含"未闭合 <answer> 到 EOF"兜底
    parse_reason_lenient: str = ""

    has_answer_open: bool = False
    has_answer_close: bool = False
    has_think_in_body: bool = False

    body_preview: str = ""  # 现扩到 2000 字符
    error: str = ""


def _build_system(fmt: str) -> str:
    """复刻 _batch_final_merge 中的 system 拼装方式：
        SUMMARY_ANSWER_SYSTEM_PROMPT + "\n\n## 输出格式约束\n" + <格式段 SYSTEM PROMPT>
    """
    if fmt == "json":
        schema_section = BATCH_MERGE_AND_CLEAN_THINK_SYSTEM_PROMPT
    else:
        schema_section = BATCH_MERGE_AND_CLEAN_THINK_HTML_SYSTEM_PROMPT
    return SUMMARY_ANSWER_SYSTEM_PROMPT + "\n\n## 输出格式约束\n" + schema_section


def _build_user_prompt(fmt: str, question: str, batch_summaries: list[str]) -> str:
    """user prompt 复刻 _batch_final_merge 的 .format 调用方式。"""
    numbered = "\n\n".join(
        f"### 摘要 {i+1}\n{s}" for i, s in enumerate(batch_summaries)
    )
    template = (
        BATCH_MERGE_AND_CLEAN_THINK_PROMPT if fmt == "json"
        else BATCH_MERGE_AND_CLEAN_THINK_HTML_PROMPT
    )
    return template.format(question=question, batch_summaries=numbered)


def _run_once(question: dict, model_spec: dict, fmt: str, round_idx: int, timeout: float) -> CallResult:
    user = _build_user_prompt(fmt, question["question"], question["summaries"])
    system = _build_system(fmt)
    res = CallResult(
        question_id=question["id"],
        model_display=model_spec["display"],
        fmt=fmt,
        round_idx=round_idx,
    )
    t0 = time.time()
    try:
        raw = chat(
            user, vendor=model_spec["vendor"], model=model_spec["model"],
            system=system, enable_thinking=True,
        )
    except Exception as e:
        res.elapsed_ms = int((time.time() - t0) * 1000)
        res.error = f"{type(e).__name__}: {e}"
        return res
    res.elapsed_ms = int((time.time() - t0) * 1000)
    res.raw_len = len(raw or "")

    # client.py 已把 reasoning_content 以 <think>...</think> 前缀回注到 content；
    # 用 split_think_block 拆出"思考前缀"和"实际 body"，与生产 _chat_final_with_format_retry 一致。
    think_prefix, body = split_think_block(raw or "")
    res.body_len = len(body or "")
    res.reasoning_len = len(think_prefix or "")
    res.has_think_prefix = bool(think_prefix)
    # 扩到 2000 字符以容纳完整 deepseek body 看 </answer> 是否真的存在
    res.body_preview = (body or "")[:2000].replace("\n", "\\n")

    # 标签存在性诊断（同时记录 strict/lenient 两套解析结果）
    if body:
        res.has_answer_open = bool(re.search(r"<answer\b", body, re.IGNORECASE))
        res.has_answer_close = bool(re.search(r"</answer\s*>", body, re.IGNORECASE))
        res.has_think_in_body = bool(re.search(r"<think\b", body, re.IGNORECASE))

    if fmt == "json":
        parsed = _validate_think_answer_json(body)
        if parsed:
            res.parse_ok = True
            res.parse_ok_lenient = True
            res.parse_reason = "json_valid"
            res.parse_reason_lenient = "json_valid"
            res.think_len = len(parsed["think"])
            res.answer_len = len(parsed["answer"])
        else:
            res.parse_reason = "json_invalid_or_empty_field"
            res.parse_reason_lenient = "json_invalid_or_empty_field"
    else:
        # HTML 模式下，think 优先取 body 内的 <think>...</think>；缺失时回落到 think_prefix
        parsed_strict = _parse_think_answer_html(body, fallback_think=think_prefix)
        if parsed_strict:
            res.parse_ok = True
            res.parse_reason = "html_valid"
            res.think_len = len(parsed_strict["think"])
            res.answer_len = len(parsed_strict["answer"])
        else:
            res.parse_reason = "html_no_closed_answer_tag"

        parsed_lenient = _parse_think_answer_html_lenient(body, fallback_think=think_prefix)
        if parsed_lenient:
            res.parse_ok_lenient = True
            if parsed_strict:
                res.parse_reason_lenient = "html_valid"
            else:
                res.parse_reason_lenient = "html_unclosed_answer_salvaged"
            # lenient 模式下，think_len/answer_len 取 lenient 解析结果（更准）
            res.think_len = len(parsed_lenient["think"])
            res.answer_len = len(parsed_lenient["answer"])
        else:
            res.parse_reason_lenient = "html_no_answer_open_tag"

    return res


def _summarize(results: list[CallResult]) -> str:
    """按 (model, fmt) 聚合主指标，输出 markdown 表格。"""
    cells: dict[tuple[str, str], list[CallResult]] = {}
    for r in results:
        cells.setdefault((r.model_display, r.fmt), []).append(r)

    lines = []
    lines.append("## 汇总：按 (model × fmt) 聚合\n")
    lines.append("| model | fmt | n | strict_pass | strict% | lenient_pass | lenient% | avg_ms | think_avg | answer_avg | reasoning_avg |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (model, fmt), rs in sorted(cells.items()):
        n = len(rs)
        passed = sum(1 for r in rs if r.parse_ok)
        passed_lenient = sum(1 for r in rs if r.parse_ok_lenient)
        avg_ms = sum(r.elapsed_ms for r in rs) / n if n else 0
        think_avg = sum(r.think_len for r in rs) / n if n else 0
        answer_avg = sum(r.answer_len for r in rs) / n if n else 0
        reasoning_avg = sum(r.reasoning_len for r in rs) / n if n else 0
        lines.append(
            f"| {model} | {fmt} | {n} | {passed} | {passed/n*100:.0f}% | "
            f"{passed_lenient} | {passed_lenient/n*100:.0f}% | "
            f"{avg_ms:.0f} | {think_avg:.0f} | {answer_avg:.0f} | {reasoning_avg:.0f} |"
        )

    lines.append("\n## 按 (model × fmt × question) 拆解通过率（strict / lenient）\n")
    cells2: dict[tuple[str, str, str], list[CallResult]] = {}
    for r in results:
        cells2.setdefault((r.model_display, r.fmt, r.question_id), []).append(r)
    lines.append("| model | fmt | question | n | strict_pass% | lenient_pass% |")
    lines.append("|---|---|---|---:|---:|---:|")
    for (model, fmt, qid), rs in sorted(cells2.items()):
        n = len(rs)
        passed = sum(1 for r in rs if r.parse_ok)
        passed_lenient = sum(1 for r in rs if r.parse_ok_lenient)
        lines.append(
            f"| {model} | {fmt} | {qid} | {n} | "
            f"{passed/n*100:.0f}% | {passed_lenient/n*100:.0f}% |"
        )

    lines.append("\n## (deepseek/html) 标签形态诊断（验证『reasoning_content 才是真 think』）\n")
    deep_html = [r for r in results if r.model_display.startswith("deepseek") and r.fmt == "html"]
    if deep_html:
        n = len(deep_html)
        no_think_in_body = sum(1 for r in deep_html if not r.has_think_in_body)
        has_open = sum(1 for r in deep_html if r.has_answer_open)
        has_close = sum(1 for r in deep_html if r.has_answer_close)
        unclosed = sum(1 for r in deep_html if r.has_answer_open and not r.has_answer_close)
        lines.append(f"- 样本数：{n}")
        lines.append(f"- body 内无 `<think>` 标签：{no_think_in_body}/{n} ({no_think_in_body/n*100:.0f}%)")
        lines.append(f"- 含 `<answer>` 开标签：{has_open}/{n} ({has_open/n*100:.0f}%)")
        lines.append(f"- 含 `</answer>` 闭标签：{has_close}/{n} ({has_close/n*100:.0f}%)")
        lines.append(f"- **未闭合 `<answer>`（开有/闭无）：{unclosed}/{n} ({unclosed/n*100:.0f}%)** ← 加 lenient 兜底可救")

    return "\n".join(lines)


def _format_failures(results: list[CallResult], max_per_cell: int = 2) -> str:
    """按 (model, fmt) 列出失败样例 body 头部预览。"""
    cells: dict[tuple[str, str], list[CallResult]] = {}
    for r in results:
        if r.parse_ok or r.error:
            # error 单独一节
            continue
        cells.setdefault((r.model_display, r.fmt), []).append(r)

    if not cells:
        return ""
    lines = ["\n## 解析失败样例 body 头部预览\n"]
    for (model, fmt), rs in sorted(cells.items()):
        lines.append(f"### {model} / {fmt}（共 {len(rs)} 例失败）\n")
        for r in rs[:max_per_cell]:
            lines.append(
                f"- [{r.question_id} round={r.round_idx}] reason={r.parse_reason} "
                f"body_len={r.body_len} reasoning_len={r.reasoning_len} elapsed={r.elapsed_ms}ms"
            )
            lines.append(f"  body[:280]: {r.body_preview}")
        lines.append("")
    return "\n".join(lines)


def _format_errors(results: list[CallResult]) -> str:
    errors = [r for r in results if r.error]
    if not errors:
        return ""
    lines = ["\n## 网络/服务错误样例\n"]
    for r in errors[:8]:
        lines.append(
            f"- [{r.model_display}/{r.fmt}/{r.question_id} round={r.round_idx}] "
            f"elapsed={r.elapsed_ms}ms err={r.error}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rounds", type=int, default=3, help="每个 cell 的轮数（默认 3）")
    parser.add_argument("--workers", type=int, default=1, help="并发线程数（默认 1 串行；调大可加速但也加重服务端压力）")
    parser.add_argument("--timeout", type=float, default=180.0, help="单次 chat 超时秒数（默认 180）")
    parser.add_argument(
        "--questions", type=str, default="all",
        help="逗号分隔问题 id，如 Q1_农产品税率栏；默认 all 全部",
    )
    parser.add_argument(
        "--models", type=str, default="all",
        help="逗号分隔 display 名，如 deepseek-v3.2；默认 all",
    )
    parser.add_argument(
        "--log-dir", type=str, default="tests_logs",
        help="日志输出目录（默认 tests_logs）",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    selected_questions = QUESTIONS if args.questions == "all" else [
        q for q in QUESTIONS if q["id"] in {x.strip() for x in args.questions.split(",")}
    ]
    selected_models = MODELS if args.models == "all" else [
        m for m in MODELS if m["display"] in {x.strip() for x in args.models.split(",")}
    ]
    if not selected_questions or not selected_models:
        raise SystemExit("没有匹配到任何问题或模型，检查 --questions / --models 参数")

    formats = ["json", "html"]
    total = len(selected_questions) * len(selected_models) * len(formats) * args.rounds
    logger.info(
        f"启动：questions={len(selected_questions)} models={len(selected_models)} "
        f"formats={len(formats)} rounds={args.rounds} → 共 {total} 次调用，"
        f"并发={args.workers}"
    )

    tasks = []
    for q in selected_questions:
        for m in selected_models:
            for fmt in formats:
                for r in range(1, args.rounds + 1):
                    tasks.append((q, m, fmt, r))

    results: list[CallResult] = []
    if args.workers <= 1:
        for i, (q, m, fmt, r) in enumerate(tasks, 1):
            logger.info(f"[{i}/{total}] q={q['id']} model={m['display']} fmt={fmt} round={r}")
            res = _run_once(q, m, fmt, r, args.timeout)
            logger.info(
                f"  -> ok={res.parse_ok} reason={res.parse_reason} elapsed={res.elapsed_ms}ms "
                f"think_len={res.think_len} answer_len={res.answer_len} err={res.error[:80] if res.error else '-'}"
            )
            results.append(res)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            future_map = {
                ex.submit(_run_once, q, m, fmt, r, args.timeout): (q, m, fmt, r)
                for (q, m, fmt, r) in tasks
            }
            done = 0
            for fut in as_completed(future_map):
                done += 1
                q, m, fmt, r = future_map[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    res = CallResult(
                        question_id=q["id"], model_display=m["display"],
                        fmt=fmt, round_idx=r, error=f"{type(e).__name__}: {e}",
                    )
                logger.info(
                    f"[{done}/{total}] q={res.question_id} model={res.model_display} "
                    f"fmt={res.fmt} round={res.round_idx} -> ok={res.parse_ok} "
                    f"reason={res.parse_reason} elapsed={res.elapsed_ms}ms"
                )
                results.append(res)

    summary = _summarize(results)
    failures = _format_failures(results)
    errors = _format_errors(results)

    print("\n" + "=" * 60)
    print(summary)
    if failures:
        print(failures)
    if errors:
        print(errors)
    print("=" * 60)

    # 落盘
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"format_priority_compare_{ts}.log"
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"# format_priority_compare run at {ts}\n\n")
        f.write(f"args: {vars(args)}\n\n")
        f.write(summary + "\n")
        if failures:
            f.write(failures + "\n")
        if errors:
            f.write(errors + "\n")
        f.write("\n## 全部 raw 结果（JSON 行）\n")
        for r in results:
            f.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")
    logger.info(f"日志已落盘：{log_path}")


if __name__ == "__main__":
    main()
