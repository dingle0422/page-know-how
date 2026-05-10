"""Reciprocal Rank Fusion。

接受多个 ``[(item_id, score), ...]`` 排好序的列表，输出按融合分数降序的 item_id 列表。

公式：``rrf_score(item) = sum_over_lists(1 / (k + rank_in_that_list))``，
``rank`` 从 1 起；item 不在某列表里就不贡献该项分数。
"""

from __future__ import annotations

from typing import Iterable

from .. import config


def reciprocal_rank_fusion(
    rank_lists: Iterable[list[tuple[int, float]]],
    *,
    k: int = config.RRF_K,
    top_k: int | None = None,
) -> list[tuple[int, float]]:
    """返回 ``[(item_id, fused_score), ...]``，按融合分数降序。

    - ``rank_lists`` 内每个子列表已按各自打分降序；
    - ``top_k=None`` 时返回全量融合结果。
    """

    fused: dict[int, float] = {}
    for sub in rank_lists:
        for rank, (item_id, _score) in enumerate(sub or [], start=1):
            fused[item_id] = fused.get(item_id, 0.0) + 1.0 / (k + rank)
    pairs = list(fused.items())
    pairs.sort(key=lambda x: x[1], reverse=True)
    if top_k is not None:
        pairs = pairs[: max(int(top_k), 0)]
    return pairs
