#!/usr/bin/env python3
"""批量测试 inference preview 四套 prompt，每种跑 N 次并写入 markdown 报告。

用法:
    python test/run_preview_prompt_batch.py
    python test/run_preview_prompt_batch.py --runs 5
    python test/run_preview_prompt_batch.py --type with_tgk_and_cases --runs 3
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference.llm_stream import StreamTagRouter, chat_stream
from inference.prompts import select_preview_prompt

VENDOR = "servyou"
MODEL = "deepseek-v3.2-1163259bcc6c"

QUESTION = "咸鸭蛋、松花蛋在开具发票时是否可以享受免税政策？"

TOPIC_GENERAL_KNOWLEDGE = """\
本专题讨论**自产农产品销售环节**的增值税处理要点：

1. **自产认定**：「自产」指纳税人自己种植、养殖、捕捞或加工取得的初级农产品。外购后简单包装、贴牌再销售一般不构成自产；委托养殖、代养模式需核查合同与实质控制权。

2. **注释范围**：销售自产农产品免税的前提是产品属于《农业产品注释范围》。蛋类（鲜蛋）属于明确的初级农产品；咸蛋、皮蛋/松花蛋若仅经腌制、变蛋等传统简单加工，通常仍视为初级农产品的延伸形态。

3. **加工深度边界**：「简单加工」指不改变农产品基本物理形态和本质属性的加工。若加工成蛋粉、蛋液、复合调味蛋制品等，可能超出初级农产品范畴，不再适用自产免税。

4. **开票规范**：适用免税政策的，应开具增值税普通发票，税率栏选择「免税」，不得误选 9% 或其他征收率。混营自产与外购产品的，须分别核算、分别开票。

5. **与外购抵扣路径区分**：外购农产品再销售不适用自产免税，可走购进农产品计算抵扣或按适用税率计税，两条路径不可混用。
"""

# 每条案例知识约 500～1000 字，贴近真实 case 库体量
CASE_1_KNOWLEDGE = """\
**案例背景**：某食品公司自有养殖基地，从鸭苗孵化、成鸭饲养到鲜蛋产出均由公司完成，随后将自产鲜鸭蛋经传统盐水腌制工艺制成咸鸭蛋对外销售。税务机关在辅导期发现其部分发票税率栏选择了 9%，部分选择了免税，存在口径不一致。

**判定过程**：
第一步，核实「自产」链条。公司提供了养殖台账、饲料采购记录、鸭舍资产清单及员工名册，证明从养殖到产蛋环节均由本公司控制并完成，不存在外购鲜蛋再加工的情形，满足自产认定。

第二步，产品定性。咸鸭蛋是在鲜蛋基础上经腌制处理的蛋类制品。实务中普遍认为，传统腌制未添加复杂辅料、未改变蛋类基本形态（仍为完整蛋体），属于初级农产品的简单加工，仍在注释范围内。

第三步，政策匹配。同时满足「自产」+「注释范围内初级农产品（含简单加工）」两个条件，可适用销售自产农产品增值税免税政策。

第四步，开票纠正。辅导后统一改为开具免税普通发票，税率栏填「免税」，并补做前期差异说明。

**关键判定逻辑**：
- 自产链条必须完整可追溯，任何一环外购（如外购鲜蛋）即打破自产认定；
- 咸鸭蛋的传统腌制一般视为简单加工，但具体工艺（是否添加防腐剂、是否改变产品形态）仍需个案核实；
- 开票口径必须与适用政策一致，混用 9% 与免税存在合规风险。

**易错点**：
- 将「外购鲜蛋腌制」误判为自产；
- 将咸鸭蛋与深加工蛋制品（如蛋黄酱、蛋粉）混为一谈；
- 自产免税与外购农产品抵扣两条路径混淆使用。
"""

CASE_2_KNOWLEDGE = """\
**案例背景**：某商贸公司主要从农户及小型加工厂批量收购松花蛋（皮蛋），经分拣、包装后销往超市及餐饮渠道。公司财务人员认为松花蛋属于农产品，遂按「自产农产品免税」开具免税普通发票。主管税务所核查后认定其不适用自产免税，要求更正并按适用税率补税及滞纳金。

**判定过程**：
第一步，业务实质核查。公司通过采购合同、入库单、付款流水确认：松花蛋全部来自外部供应商，公司仅承担收购、仓储、分拣、贴标包装及销售职能，未参与任何养殖或初加工环节，不构成「自产」。

第二步，路径区分。外购农产品再销售不能适用「销售自产农产品免税」政策。若符合购进农产品计算抵扣条件，可按相关规定抵扣进项税额后按适用税率销售；若不符合抵扣条件，则按适用征收率或税率全额计税。

第三步，产品属性补充分析。即便松花蛋本身在注释范围内属于蛋类简单加工品，「产品属性符合注释范围」≠「销售方可享受自产免税」。免税的关键前提是「自产」，外购路径下此条件不成立。

第四步，整改要求。公司更正开票方式，对历史免税发票进行自查补税，并建立「自产业务」与「商贸流通业务」分开核算、分开开票的内部制度。

**关键判定逻辑**：
- 外购再销售一律不适用自产农产品免税，无论产品本身是否为农产品；
- 「收购农产品抵扣」与「自产免税」是两条独立路径，适用条件完全不同；
- 商贸流通企业尤其容易将「产品属于农产品」等同于「可以免税开票」。

**易错点**：
- 看到松花蛋是「农产品」就直接开免税票；
- 未区分自产与外购的业务模式；
- 将农户免税政策（若涉及）与企业销售环节政策混谈；
- 包装、分拣等轻加工被误认为「自产加工」。
"""

RELATED_CASES = [
    {"question": "公司自养鸭子产咸鸭蛋对外销售，能否按自产农产品免税开票？", "knowledge": CASE_1_KNOWLEDGE},
    {"question": "从农户收购松花蛋再销售，能否享受农产品免税？", "knowledge": CASE_2_KNOWLEDGE},
]

PREVIEW_TYPES: dict[str, dict] = {
    "base": {
        "label": "PREVIEW（仅用户问题）",
        "topic_general_knowledge": None,
        "related_cases": None,
        "output": "preview_results_base.md",
    },
    "with_tgk": {
        "label": "PREVIEW_WITH_TGK（专题通用知识）",
        "topic_general_knowledge": TOPIC_GENERAL_KNOWLEDGE,
        "related_cases": None,
        "output": "preview_results_with_tgk.md",
    },
    "with_cases": {
        "label": "PREVIEW_WITH_CASES（历史经验）",
        "topic_general_knowledge": None,
        "related_cases": RELATED_CASES,
        "output": "preview_results_with_cases.md",
    },
    "with_tgk_and_cases": {
        "label": "PREVIEW_WITH_TGK_AND_CASES（专题通用知识 + 历史经验）",
        "topic_general_knowledge": TOPIC_GENERAL_KNOWLEDGE,
        "related_cases": RELATED_CASES,
        "output": "preview_results_with_tgk_and_cases.md",
    },
}


async def run_single_preview(sys_p: str, usr_p: str) -> tuple[str, str, float, str | None]:
    """调用 LLM 一次，返回 (think, answer, elapsed_sec, error)。"""
    router = StreamTagRouter()
    native_think: list[str] = []
    routed_think: list[str] = []
    routed_answer: list[str] = []
    t0 = time.time()
    err: str | None = None

    try:
        async for channel, delta in chat_stream(
            usr_p,
            vendor=VENDOR,
            model=MODEL,
            system=sys_p,
            enable_thinking=True,
        ):
            if channel == "think":
                native_think.append(delta)
            else:
                router.feed(
                    delta,
                    on_think=lambda s: routed_think.append(s),
                    on_answer=lambda s: routed_answer.append(s),
                )
    except Exception as e:
        err = f"{type(e).__name__}: {e}"

    elapsed = time.time() - t0
    think = ("".join(native_think) + "".join(routed_think)).strip()
    answer = "".join(routed_answer).strip()
    return think, answer, elapsed, err


def _md_escape_block(text: str) -> str:
    return text or "（空）"


def build_report_md(
    *,
    type_key: str,
    cfg: dict,
    runs: list[dict],
    runs_count: int,
) -> str:
    sys_p, usr_p = select_preview_prompt(
        question=QUESTION,
        topic_general_knowledge=cfg["topic_general_knowledge"],
        related_cases=cfg["related_cases"],
    )
    case_lens = [len(c["knowledge"]) for c in (cfg["related_cases"] or [])]

    lines = [
        f"# Preview 测试报告：{cfg['label']}",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 类型键：`{type_key}`",
        f"- 运行次数：{runs_count}",
        f"- 模型：`{VENDOR}` / `{MODEL}`",
        f"- 用户问题：{QUESTION}",
        "",
    ]
    if cfg["topic_general_knowledge"]:
        lines += [
            f"- 专题通用知识字数：{len(cfg['topic_general_knowledge'])}",
        ]
    if case_lens:
        lines += [
            f"- 历史经验案例数：{len(case_lens)}",
            f"- 各案例知识字数：{case_lens}",
        ]
    lines += [
        "",
        "---",
        "",
        "## 输入 Prompt",
        "",
        "### System Prompt",
        "",
        "```",
        sys_p,
        "```",
        "",
        "### User Prompt",
        "",
        "```",
        usr_p,
        "```",
        "",
        "---",
        "",
        "## 运行结果",
        "",
    ]

    for r in runs:
        idx = r["index"]
        lines += [
            f"### 第 {idx} 次（耗时 {r['elapsed_sec']:.1f}s）",
            "",
        ]
        if r["error"]:
            lines += [f"**错误**：{r['error']}", ""]
        lines += [
            f"- think 字数：{r['think_len']}",
            f"- answer 字数：{r['answer_len']}",
            "",
            "#### `<think>`",
            "",
            _md_escape_block(r["think"]),
            "",
            "#### `<answer>`",
            "",
            _md_escape_block(r["answer"]),
            "",
            "---",
            "",
        ]

    return "\n".join(lines)


async def run_batch(*, type_keys: list[str], runs_count: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for type_key in type_keys:
        cfg = PREVIEW_TYPES[type_key]
        sys_p, usr_p = select_preview_prompt(
            question=QUESTION,
            topic_general_knowledge=cfg["topic_general_knowledge"],
            related_cases=cfg["related_cases"],
        )
        print(f"\n{'=' * 60}")
        print(f"开始测试：{cfg['label']} × {runs_count} 次")
        print(f"{'=' * 60}")

        runs: list[dict] = []
        for i in range(1, runs_count + 1):
            print(f"  [{type_key}] 第 {i}/{runs_count} 次...", flush=True)
            think, answer, elapsed, err = await run_single_preview(sys_p, usr_p)
            runs.append({
                "index": i,
                "think": think,
                "answer": answer,
                "think_len": len(think),
                "answer_len": len(answer),
                "elapsed_sec": elapsed,
                "error": err,
            })
            status = "OK" if not err else f"ERR: {err[:80]}"
            print(f"    -> {status} | think={len(think)} answer={len(answer)} | {elapsed:.1f}s")

        md = build_report_md(type_key=type_key, cfg=cfg, runs=runs, runs_count=runs_count)
        out_path = out_dir / cfg["output"]
        out_path.write_text(md, encoding="utf-8")
        print(f"  已写入：{out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="批量测试 preview 四套 prompt")
    parser.add_argument(
        "--runs", type=int, default=5, help="每种 prompt 运行次数（默认 5）"
    )
    parser.add_argument(
        "--type",
        choices=list(PREVIEW_TYPES.keys()),
        action="append",
        dest="types",
        help="仅跑指定类型，可重复指定；默认跑全部 4 种",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "test",
        help="markdown 输出目录（默认 test/）",
    )
    args = parser.parse_args()
    type_keys = args.types or list(PREVIEW_TYPES.keys())

    print(f"Preview 批量测试：{len(type_keys)} 种 × {args.runs} 次 = {len(type_keys) * args.runs} 次 LLM 调用")
    asyncio.run(run_batch(type_keys=type_keys, runs_count=args.runs, out_dir=args.out_dir))


if __name__ == "__main__":
    main()
