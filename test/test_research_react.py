"""reSearch 模式离线冒烟测试（不依赖 LLM / redis / retrieval_service）。

覆盖点：

1. ``parse_research_answer``：新子标签解析、complete 忽略 action/query、旧协议回退、
   解析失败兜底 paginate；
2. ``_chunk_dedup_key``：归一化正文去重；
3. ``select_react_prompt``：``re_search_enabled`` 切换 system prompt；
4. ``_run_research_loop`` 状态机（mock run_one_round / search）：
   - incomplete + research → 新召回替换证据集 → complete → forced final；
   - paginate 耗尽 → final；
   - ``max_rounds=1`` 预算用尽 → 立即 final；
   - research 召回与已看过块重复 → 单向去重后为空 → 降级 paginate → final；
5. ``run(re_search_enabled=False)`` 走旧主循环分支（不调用 ``_run_research_loop``）。

运行方式：

- ``pytest test/test_research_react.py -v``
- 或直接 ``python test/test_research_react.py``
"""

from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_core.chunk_builder import KnowledgeChunk  # noqa: E402

from inference import prompts  # noqa: E402
from inference import react_loop as RL  # noqa: E402


def _chunk(content: str, *, index: int = 0) -> KnowledgeChunk:
    return KnowledgeChunk(index=index, content=content, heading_paths=[["1_测试章节"]])


# --------------------------------------------------------------------------- #
# 1) parse_research_answer
# --------------------------------------------------------------------------- #


def test_parse_complete_ignores_action_and_query():
    raw = (
        "<completion>complete</completion>"
        "<action>research</action>"
        "<search-query>增值税 小规模</search-query>"
    )
    d = RL.parse_research_answer(raw)
    assert d == {"completion": "complete", "action": "paginate", "search_query": ""}


def test_parse_incomplete_research_with_query():
    raw = (
        "<completion>incomplete</completion>"
        "<action>research</action>"
        "<search-query>增值税  小规模\n纳税人</search-query>"
    )
    d = RL.parse_research_answer(raw)
    assert d["completion"] == "incomplete"
    assert d["action"] == "research"
    assert d["search_query"] == "增值税 小规模 纳税人"


def test_parse_incomplete_paginate():
    d = RL.parse_research_answer(
        "<completion>incomplete</completion><action>paginate</action>"
    )
    assert d["completion"] == "incomplete"
    assert d["action"] == "paginate"
    assert d["search_query"] == ""


def test_parse_research_without_query_downgrades_paginate():
    d = RL.parse_research_answer(
        "<completion>incomplete</completion><action>research</action>"
    )
    assert d["action"] == "paginate"


def test_parse_legacy_fallback_incomplete():
    assert RL.parse_research_answer("incomplete")["completion"] == "incomplete"
    assert RL.parse_research_answer("complete")["completion"] == "complete"


def test_parse_garbage_fallback_paginate():
    d = RL.parse_research_answer("<<>> random noise")
    assert d == {"completion": "incomplete", "action": "paginate", "search_query": ""}


# --------------------------------------------------------------------------- #
# 2) dedup key
# --------------------------------------------------------------------------- #


def test_chunk_dedup_key_normalizes_whitespace():
    a = _chunk("  增值税   政策 ")
    b = _chunk("增值税 政策")
    assert RL._chunk_dedup_key(a) == RL._chunk_dedup_key(b)


# --------------------------------------------------------------------------- #
# 3) prompt routing
# --------------------------------------------------------------------------- #


def test_select_react_prompt_research_system():
    sys_p, _ = prompts.select_react_prompt(
        is_last_chunk=False,
        question="Q",
        evidence="E",
        prev_think="",
        preview=None,
        skills=None,
        re_search_enabled=True,
    )
    assert sys_p is prompts.REACT_INTERMEDIATE_RESEARCH_SYSTEM_PROMPT
    assert "<completion>" in sys_p
    assert "<search-query>" in sys_p


def test_select_react_prompt_legacy_system():
    sys_p, _ = prompts.select_react_prompt(
        is_last_chunk=False,
        question="Q",
        evidence="E",
        prev_think="",
        preview=None,
        skills=None,
        re_search_enabled=False,
    )
    assert sys_p is prompts.REACT_INTERMEDIATE_SYSTEM_PROMPT


# --------------------------------------------------------------------------- #
# 4) _run_research_loop (mocked)
# --------------------------------------------------------------------------- #


class _FakeRedisStream:
    """最小 RedisStream 替身：research 状态机测试够用；legacy 分流测试再补方法。"""

    def __init__(self) -> None:
        self.headings: list[str] | None = None

    async def set_status(self, *a, **k) -> None:
        pass

    async def ensure_react_chunk(self, *a, **k) -> None:
        pass

    async def get(self, *a, **k) -> dict:
        return {}

    async def set_react_chunk_complete(self, *a, **k) -> None:
        pass

    async def set_used_headings(self, _tid: str, headings: list[str]) -> None:
        self.headings = headings

    async def set_react_intermediate_locked(self, *a, **k) -> None:
        pass

    async def set_react_final_locked(self, *a, **k) -> None:
        pass

    async def append_react_chunk_delta(self, *a, **k) -> None:
        pass


def _run_loop(
    *,
    chunks: list[KnowledgeChunk],
    decisions: list[str],
    search_results: list[list[KnowledgeChunk]] | None = None,
    max_rounds: int = 5,
) -> tuple[dict, list[tuple], _FakeRedisStream]:
    calls: list[tuple] = []
    search_calls: list[str] = []
    di = {"i": 0}
    si = {"i": 0}

    async def fake_search(query: str) -> list[KnowledgeChunk]:
        search_calls.append(query)
        if search_results is None:
            return [_chunk("NEW-" + query, index=99)]
        idx = si["i"]
        si["i"] += 1
        if idx < len(search_results):
            return search_results[idx]
        return []

    async def prepare_final(idx: int) -> None:
        calls.append(("prepare_final", idx))

    async def lock_final() -> None:
        calls.append(("lock_final",))

    async def run_one_round(
        *,
        round_idx: int,
        group_text: str,
        is_last_chunk: bool,
        prev_think_val: str,
        step_kind: str,
        research_mode: bool = False,
    ) -> tuple[str, str, str]:
        calls.append(
            ("round", round_idx, step_kind, is_last_chunk, research_mode, group_text[:20])
        )
        if is_last_chunk:
            return ("complete", "final-think", "FINAL_ANSWER")
        ans = decisions[di["i"]]
        di["i"] += 1
        return ("", f"think-{round_idx}", ans)

    rs = _FakeRedisStream()

    async def _go():
        return await RL._run_research_loop(
            task_id="smoke-t",
            question="测试问题",
            chunks=chunks,
            redis_stream=rs,
            chunk_size=5000,
            max_rounds=max_rounds,
            policy_id="policy__cs500",
            top_n=20,
            top_m=20,
            search_fn=fake_search,
            prepare_final=prepare_final,
            lock_final=lock_final,
            run_one_round=run_one_round,
        )

    res = asyncio.run(_go())
    return res, calls, rs, search_calls


def test_research_loop_research_then_complete():
    res, calls, rs, search_calls = _run_loop(
        chunks=[_chunk("OLD chunk A"), _chunk("OLD chunk B", index=1)],
        decisions=[
            "<completion>incomplete</completion>"
            "<action>research</action><search-query>q2</search-query>",
            "<completion>complete</completion>",
        ],
    )
    assert res["answer"] == "FINAL_ANSWER"
    assert res["verdict"] == "complete"
    assert search_calls == ["q2"]
    assert ("prepare_final", 2) in calls
    assert ("lock_final",) in calls
    assert rs.headings is not None


def test_research_loop_paginate_exhaust_then_final():
    res, calls, _, _ = _run_loop(
        chunks=[_chunk("only one group chunk")],
        decisions=[
            "<completion>incomplete</completion><action>paginate</action>",
        ],
    )
    assert res["answer"] == "FINAL_ANSWER"
    assert ("prepare_final", 1) in calls


def test_research_loop_max_rounds_budget_forces_final():
    res, calls, _, _ = _run_loop(
        chunks=[_chunk("g")],
        decisions=[
            "<completion>incomplete</completion><action>paginate</action>",
        ],
        max_rounds=1,
    )
    assert res["answer"] == "FINAL_ANSWER"
    assert res["rounds"] == 1
    # budget_last：第一轮就直接 final，不应再跑 intermediate。
    round_kinds = [c[2] for c in calls if c[0] == "round"]
    assert round_kinds == ["final"]


def test_research_loop_dedup_empty_research_fallback_paginate():
    """新召回块与已看过块正文相同 → 单向去重后为空 → 降级 paginate → 证据耗尽 → final。"""
    same = _chunk("SAME content block")
    res, calls, _, search_calls = _run_loop(
        chunks=[same],
        decisions=[
            "<completion>incomplete</completion>"
            "<action>research</action><search-query>dup query</search-query>",
        ],
        search_results=[[same]],  # 与已看过块重复，去重后为空
    )
    assert res["answer"] == "FINAL_ANSWER"
    assert search_calls == ["dup query"]
    assert any(c[0] == "round" and c[2] == "intermediate" for c in calls)


# --------------------------------------------------------------------------- #
# 5) run() 分流：re_search_enabled=False 不进入 research 状态机
# --------------------------------------------------------------------------- #


def test_run_true_dispatches_research_loop():
    """re_search_enabled=True 时必须走 _run_research_loop，不进入旧 for 循环。"""
    with patch.object(RL, "_run_research_loop", new_callable=AsyncMock) as mock_research:
        mock_research.return_value = {"rounds": 0, "verdict": "complete", "answer": "x"}
        out = asyncio.run(
            RL.run(
                "t",
                "Q",
                [_chunk("x")],
                _FakeRedisStream(),
                re_search_enabled=True,
                policy_id="p__cs500",
            )
        )
        mock_research.assert_called_once()
    assert out["answer"] == "x"


def test_run_false_skips_research_loop():
    """re_search_enabled=False 时不得调用 _run_research_loop（旧主循环零侵入）。"""
    async def _fake_stream(*a, **k):
        return ("complete", "think", "用户答案")

    with patch.object(RL, "_run_research_loop", new_callable=AsyncMock) as mock_research:
        with patch.object(RL, "_stream_react_round", side_effect=_fake_stream):
            with patch.object(RL, "pack_chunks_by_size", return_value=["证据组"]):
                with patch.object(RL, "_pack_chunks_with_indices", return_value=[[0]]):
                    out = asyncio.run(
                        RL.run(
                            "t",
                            "Q",
                            [_chunk("x")],
                            _FakeRedisStream(),
                            re_search_enabled=False,
                            max_rounds=5,
                        )
                    )
        mock_research.assert_not_called()
    assert out["answer"] == "用户答案"
    assert out["verdict"] == "complete"


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #

_ALL_TESTS = [
    test_parse_complete_ignores_action_and_query,
    test_parse_incomplete_research_with_query,
    test_parse_incomplete_paginate,
    test_parse_research_without_query_downgrades_paginate,
    test_parse_legacy_fallback_incomplete,
    test_parse_garbage_fallback_paginate,
    test_chunk_dedup_key_normalizes_whitespace,
    test_select_react_prompt_research_system,
    test_select_react_prompt_legacy_system,
    test_research_loop_research_then_complete,
    test_research_loop_paginate_exhaust_then_final,
    test_research_loop_max_rounds_budget_forces_final,
    test_research_loop_dedup_empty_research_fallback_paginate,
    test_run_true_dispatches_research_loop,
    test_run_false_skips_research_loop,
]


def _main() -> int:
    failed = 0
    for t in _ALL_TESTS:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"[ERROR] {t.__name__}: {e!r}")
        else:
            print(f"[ok]   {t.__name__}")
    print(f"\n{len(_ALL_TESTS) - failed}/{len(_ALL_TESTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
