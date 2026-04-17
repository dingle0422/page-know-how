"""标准商品/服务名称验证服务

调用税务智能分类模型，将用户输入的商品或服务名称匹配到税收分类编码体系中的标准名称，
并返回对应税率。服务会返回多个候选结果及其匹配度，由调用方自行判断采纳程度。
"""

import logging
from dataclasses import dataclass, field

import requests

logger = logging.getLogger(__name__)

_API_URL = "http://mlp.paas.dc.servyou-it.com/serve-1813414490354499584/predict_znfm"
_REQUEST_TIMEOUT = 15


@dataclass
class CandidateMatch:
    """单个候选匹配项"""
    standard_name: str        # spmc - 标准商品/服务名称
    abbreviation: str         # jc   - 简称
    tax_rate: str             # slv  - 税率
    confidence: float         # ppd  - 匹配度
    product_code: str         # spbm - 税收分类编码


@dataclass
class VerificationResult:
    """单条查询的完整验证结果，包含所有候选匹配项"""
    query: str
    candidates: list[CandidateMatch] = field(default_factory=list)
    message: str = ""

    @property
    def best(self) -> CandidateMatch | None:
        return self.candidates[0] if self.candidates else None


def _parse_candidates(pmxx_out: list[dict]) -> list[CandidateMatch]:
    """解析 API 返回的候选列表"""
    return [
        CandidateMatch(
            standard_name=item["spmc"],
            abbreviation=item.get("jc", ""),
            tax_rate=item["slv"],
            confidence=float(item["ppd"]),
            product_code=item.get("spbm", ""),
        )
        for item in pmxx_out
    ]


def verify_product_name(product_name: str, timeout: int = _REQUEST_TIMEOUT) -> VerificationResult:
    """验证单个商品/服务名称，返回包含多个候选项的结果。

    Args:
        product_name: 用户输入的商品或服务名称
        timeout: 请求超时秒数

    Returns:
        VerificationResult，其中 candidates 按匹配度从高到低排列
    """
    results = verify_product_names([product_name], timeout=timeout)
    return results[0]


def verify_product_names(
    product_names: list[str],
    timeout: int = _REQUEST_TIMEOUT,
) -> list[VerificationResult]:
    """批量验证商品/服务名称。

    Args:
        product_names: 商品或服务名称列表
        timeout: 请求超时秒数

    Returns:
        与输入顺序一致的 VerificationResult 列表
    """
    if not product_names:
        return []

    payload = {
        "dataArr": [
            {"spmc": name, "spbm": ""}
            for name in product_names
        ]
    }

    try:
        resp = requests.post(_API_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        api_results = resp.json()
    except requests.RequestException as e:
        logger.error(f"标准商品名称验证服务请求失败: {e}")
        return [
            VerificationResult(query=name, message=f"服务请求失败: {e}")
            for name in product_names
        ]

    results: list[VerificationResult] = []
    for i, name in enumerate(product_names):
        try:
            pmxx_out = api_results[i]["pmxx_out"]
            candidates = _parse_candidates(pmxx_out)
            results.append(VerificationResult(query=name, candidates=candidates))
        except (IndexError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"解析第 {i} 条结果失败: {e}")
            results.append(VerificationResult(query=name, message=f"结果解析失败: {e}"))

    return results


def format_result(result: VerificationResult) -> str:
    """将验证结果格式化为可直接嵌入回答的自然语言文本"""
    if not result.candidates:
        if result.message:
            return f"「{result.query}」的标准名称验证失败: {result.message}"
        return f"「{result.query}」未返回任何候选结果。"

    lines = [f"「{result.query}」在税收分类编码体系中的候选匹配："]
    for c in result.candidates:
        lines.append(
            f"  - {c.standard_name}（{c.abbreviation}），税率 {c.tax_rate}，匹配度 {c.confidence}"
        )
    return "\n".join(lines)
