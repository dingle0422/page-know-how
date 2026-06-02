"""`inference.retrieval.case_search.search_cases_multi` 多专题 fan-out 单测。

覆盖点（topic locator 多候选时的 case 检索 fan-out 行为）：

1. 多专题并发对各自 ``case_{khCode}`` collection 独立召回；
2. 每个 collection 保留 positive / negative 分桶 + cosine ≥ threshold 过滤 +
   每极性 topC 截断；
3. 跨 collection 合并后按内容键全量去重（同一条样本只保留一份，取 cosine 更高者）；
4. 同 khCode 的多个 policyId 去重，只查一次 collection；
5. 0/1 个有效 collection 时回落到单集合 ``search_cases``；
6. ``_merge_dedup_cases`` 纯函数：positive 段在前、negative 段在后、组内按相似度降序。

retrieval client 与 embedding 全部 mock，不依赖外部服务。

运行方式：

- ``pytest test/test_case_search_multi.py -v``
- 或直接 ``python test/test_case_search_multi.py``（内置 runner，无需 pytest）。
"""

from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import patch

# 允许 `python test/xxx.py` 直接跑时能 import 到仓库内的 inference 包。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.retrieval import case_search  # noqa: E402
from inference.retrieval.case_search import (  # noqa: E402
    CaseHit,
    _merge_dedup_cases,
    search_cases_multi,
)

# case_search 用到的稳定兜底列名（meta 解析不到时会回退到它）。
_POLARITY_COL = "md_case_polarity_af1bc6c9"
_TOMBSTONE_COL = "md_tombstoned_45c36238"


def _async_return(value):
    """构造一个忽略入参、返回固定值的协程函数（兼容 Py3.7，无需 AsyncMock）。"""

    async def _coro(*args, **kwargs):
        return value

    return _coro


def _hit(question: str, knowledge: str, polarity: str, sim: float) -> dict:
    """构造一条 vector_search_v2 返回的 raw hit。"""

    return {
        "cosine_similarity": sim,
        "content": question,
        "metadata": {
            "question_content": question,
            "refined_knowledge": knowledge,
            "case_polarity": polarity,
        },
    }


class _FakeClient:
    """按 (collection_id, where 里的极性) 返回预置 hits 的假 retrieval client。"""

    def __init__(self, data: dict[str, dict[str, list[dict]]]) -> None:
        # data: {collection_id: {"positive": [hit...], "negative": [hit...]}}
        self.data = data
        self.search_calls: list[tuple[str, str | None]] = []

    async def get_collection_meta_v2(self, collection_id: str) -> dict:
        return {"filterable_fields": [_POLARITY_COL, _TOMBSTONE_COL]}

    async def vector_search_v2(
        self,
        collection_id: str,
        *,
        query_vector,
        top_n,
        where=None,
        include_content=True,
        include_derived=True,
    ) -> list[dict]:
        self.search_calls.append((collection_id, where))
        w = where or ""
        if "'positive'" in w:
            polarity = "positive"
        elif "'negative'" in w:
            polarity = "negative"
        else:
            polarity = None
        bucket = self.data.get(collection_id, {})
        if polarity is None:
            return bucket.get("positive", []) + bucket.get("negative", [])
        return bucket.get(polarity, [])


def _run_multi(policy_ids, data, *, threshold=0.85, top_k=5):
    """在干净的列名缓存下跑一次 search_cases_multi（mock embedding + client）。"""

    case_search._COLUMN_CACHE.clear()
    fake = _FakeClient(data)
    with patch(
        "inference.embedding_client.embed_texts",
        new=_async_return([[0.1, 0.2, 0.3]]),
    ), patch(
        "inference.retrieval.client.get_default_client",
        new=_async_return(fake),
    ):
        result = asyncio.run(
            search_cases_multi(
                "某个问题", policy_ids, threshold=threshold, top_k=top_k
            )
        )
    return result, fake


# --------------------------------------------------------------------------- #
# 1) fan-out + 合并 + 跨 collection 去重 + 阈值过滤
# --------------------------------------------------------------------------- #

def test_fanout_merge_dedup_and_threshold():
    data = {
        "case_1001": {
            "positive": [
                _hit("Q-P1", "K-P1", "positive", 0.95),
                _hit("Q-P2", "K-P2", "positive", 0.90),
            ],
            "negative": [
                _hit("Q-N1", "K-N1", "negative", 0.88),
            ],
        },
        "case_1002": {
            "positive": [
                # 与 case_1001 的 P1 完全同内容 → 跨 collection 应去重，保留高分一份。
                _hit("Q-P1", "K-P1", "positive", 0.95),
                _hit("Q-P3", "K-P3", "positive", 0.92),
            ],
            "negative": [
                # 0.80 < 阈值 0.85 → 必须被过滤掉。
                _hit("Q-N2", "K-N2", "negative", 0.80),
            ],
        },
    }

    result, fake = _run_multi(
        ["1001_a", "1002_b"], data, threshold=0.85, top_k=5
    )

    questions = [c.question for c in result]
    # positive 段在前（P1 0.95 > P3 0.92 > P2 0.90），negative 段在后（N1 0.88）。
    assert questions == ["Q-P1", "Q-P3", "Q-P2", "Q-N1"], questions
    # P1 跨两个 collection 命中，去重后只保留一份。
    assert questions.count("Q-P1") == 1
    # N2 (0.80) 被阈值过滤。
    assert "Q-N2" not in questions
    # 极性分组：所有 positive 都排在所有 negative 之前。
    polarities = [c.polarity for c in result]
    assert polarities == ["positive", "positive", "positive", "negative"], polarities

    # 两个 collection 都被并发查询，且各自跑了 positive + negative 两个桶。
    queried = {cid for cid, _ in fake.search_calls}
    assert queried == {"case_1001", "case_1002"}
    assert len(fake.search_calls) == 4


# --------------------------------------------------------------------------- #
# 2) 同 khCode 的多个 policyId 去重 → 只查一次 collection
# --------------------------------------------------------------------------- #

def test_same_khcode_collection_dedup():
    data = {
        "case_1001": {
            "positive": [_hit("Q-P1", "K-P1", "positive", 0.95)],
            "negative": [],
        },
    }
    # 三个 policyId 实际都映射到 case_1001（剥 __cs 后取 split('_')[0]）。
    result, fake = _run_multi(
        ["1001_a", "1001_b__cs256", "1001"], data, threshold=0.85, top_k=5
    )

    assert [c.question for c in result] == ["Q-P1"]
    # 只查 case_1001 一个 collection（positive + negative 两个桶）。
    queried = {cid for cid, _ in fake.search_calls}
    assert queried == {"case_1001"}


# --------------------------------------------------------------------------- #
# 3) 单专题列表 → 回落到单集合检索（结果等价）
# --------------------------------------------------------------------------- #

def test_single_collection_falls_back():
    data = {
        "case_2002": {
            "positive": [_hit("Q-A", "K-A", "positive", 0.91)],
            "negative": [_hit("Q-B", "K-B", "negative", 0.90)],
        },
    }
    result, fake = _run_multi(["2002_x"], data, threshold=0.85, top_k=5)

    assert [c.question for c in result] == ["Q-A", "Q-B"]
    assert {cid for cid, _ in fake.search_calls} == {"case_2002"}


# --------------------------------------------------------------------------- #
# 4) 空 / 无效候选 → 直接返回 []
# --------------------------------------------------------------------------- #

def test_empty_policy_ids_returns_empty():
    result, fake = _run_multi([], {}, threshold=0.85, top_k=5)
    assert result == []
    assert fake.search_calls == []


# --------------------------------------------------------------------------- #
# 5) _merge_dedup_cases 纯函数：分组顺序 + 去重保留高分 + 组内降序
# --------------------------------------------------------------------------- #

def test_merge_dedup_pure():
    buckets = [
        [
            CaseHit(0.90, "Q1", "K1", "positive"),
            CaseHit(0.70, "Q2", "K2", "negative"),
        ],
        [
            # Q1 重复，分数更高 → 应保留 0.96 这条。
            CaseHit(0.96, "Q1", "K1", "positive"),
            CaseHit(0.85, "Q3", "K3", "positive"),
            CaseHit(0.60, "Q4", "K4", ""),  # 未知极性 → 排在最后
        ],
    ]
    merged = _merge_dedup_cases(buckets)

    assert [c.question for c in merged] == ["Q1", "Q3", "Q2", "Q4"]
    # Q1 去重后取高分。
    q1 = next(c for c in merged if c.question == "Q1")
    assert q1.cosine_similarity == 0.96
    # positive 段（Q1,Q3）→ negative 段（Q2）→ 未知极性（Q4）。
    assert [c.polarity for c in merged] == ["positive", "positive", "negative", ""]


def _main() -> int:
    tests = [
        test_fanout_merge_dedup_and_threshold,
        test_same_khcode_collection_dedup,
        test_single_collection_falls_back,
        test_empty_policy_ids_returns_empty,
        test_merge_dedup_pure,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {t.__name__}: {e}")
        except Exception as e:  # pragma: no cover
            failed += 1
            print(f"[ERROR] {t.__name__}: {e!r}")
        else:
            print(f"[ok]   {t.__name__}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
