"""标准商品/服务名称验证 — CLI 入口

设计目的：让大模型直接以 bash 命令的形式调用本技能，无需生成 Python 代码。

用法示例：
    python -m skills.standard_product_name_verification "马粪"
    python -m skills.standard_product_name_verification "马粪" "电脑" "理发服务"
    python -m skills.standard_product_name_verification --json "马粪"
    python -m skills.standard_product_name_verification --timeout 30 "马粪"

退出码：
    0 — 所有查询均成功（每条至少返回一个候选或正常的空结果）
    1 — 任意一条因网络 / 解析失败而无候选返回
"""

import argparse
import json
import sys

from .service import format_result, verify_product_names


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m skills.standard_product_name_verification",
        description=(
            "将商品/服务名称匹配到国家税收分类编码体系的标准名称、类型，"
            "并返回候选项的标准名称、简称、税率与匹配度。"
        ),
    )
    parser.add_argument(
        "names",
        nargs="+",
        metavar="NAME",
        help="待验证的商品或服务名称，可一次传入多个（建议用双引号包裹）",
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="以 JSON 数组输出结构化结果（默认输出自然语言文本，便于大模型直接阅读）",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=15,
        help="单次请求超时秒数，默认 15",
    )
    return parser


def _to_dict(results) -> list[dict]:
    return [
        {
            "query": r.query,
            "message": r.message,
            "candidates": [
                {
                    "standard_name": c.standard_name,
                    "abbreviation": c.abbreviation,
                    "tax_rate": c.tax_rate,
                    "confidence": c.confidence,
                    "product_code": c.product_code,
                }
                for c in r.candidates
            ],
        }
        for r in results
    ]


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    results = verify_product_names(args.names, timeout=args.timeout)

    if args.as_json:
        print(json.dumps(_to_dict(results), ensure_ascii=False, indent=2))
    else:
        print("\n\n".join(format_result(r) for r in results))

    has_failure = any(not r.candidates and r.message for r in results)
    return 1 if has_failure else 0


if __name__ == "__main__":
    sys.exit(main())
