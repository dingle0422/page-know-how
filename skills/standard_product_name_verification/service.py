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


# top-1 同时满足"高置信"且"显著领先次优"时，视为可直接采纳的单一权威结论；
# 注入到 LLM prompt 时不再列出剩余低分候选，避免给 LLM "分情况讨论"的把手。
_TOP1_CONFIDENT_THRESHOLD = 0.9
_TOP1_LEADING_GAP = 0.3


def format_result(
    result: VerificationResult,
    top1_threshold: float = _TOP1_CONFIDENT_THRESHOLD,
    leading_gap: float = _TOP1_LEADING_GAP,
) -> str:
    """将验证结果格式化为可直接嵌入 LLM 回答的自然语言文本。

    输出策略：
    - 无候选 → 失败/空结果提示
    - top-1 置信度 ≥ top1_threshold 且 (top1 - top2) ≥ leading_gap（或仅有 1 个候选）
      → 只输出 top-1，并明确告知 LLM "应直接采纳，无需再分情况猜测其他归类"
    - 其余情况 → 列出全部候选（保留旧行为，由调用方/LLM 自行权衡）

    设计动机：当 skill 给出单一确定性结论时，prompt 里再附带低分候选会诱导 LLM
    把"匹配度 0.4 的水果"也当成需要讨论的合法分支，最终输出"若属于X / 若属于Y"
    的摇摆答案；而高置信 + 显著领先 在分类问题中实际等价于"已识别"，应避免
    把工程上的多候选缓冲传递成业务上的歧义。
    """
    if not result.candidates:
        if result.message:
            return f"「{result.query}」的标准名称验证失败: {result.message}"
        return f"「{result.query}」未返回任何候选结果。"

    best = result.candidates[0]
    second_conf = result.candidates[1].confidence if len(result.candidates) > 1 else 0.0
    is_confident_top1 = (
        best.confidence >= top1_threshold
        and (best.confidence - second_conf) >= leading_gap
    )

    if is_confident_top1:
        return (
            f"「{result.query}」标准商品名称已识别为：{best.standard_name}"
            f"（简称：{best.abbreviation or '无'}），适用税率 {best.tax_rate}，"
            f"匹配置信度 {best.confidence}（显著高于其余候选）。\n"
            f"该归类为权威识别结果，应直接据此进行后续涉税判断，"
            f"不要再使用「若属于X / 若属于Y」等句式对该商品的归类做二次猜测。"
        )

    lines = [f"「{result.query}」在税收分类编码体系中的候选匹配："]
    for c in result.candidates:
        lines.append(
            f"  - {c.standard_name}（{c.abbreviation}），税率 {c.tax_rate}，匹配度 {c.confidence}"
        )
    return "\n".join(lines)
