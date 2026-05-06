"""验证 _merge_orphan_groups_into_groups 跨组并行：

构造 2 个新建 keyword 组的注入场景，把 _reduce_group_to_single 替换为带 sleep
的桩函数；记录每个组实际开始/结束时间，确认两组同时启动而非依次串行。
"""

import os
import sys
import time
import threading
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reasoner.v3.agent_graph import AgentGraph, _StreamGroup


def main() -> None:
    g_self = _StreamGroup(kind="self", keyword="", headings=[], parts=[], summary="")
    groups = [g_self]
    orphan_by_kw = {
        "蔬菜": ["**蔬菜 关联知识** ...sample fragment A..."],
        "农产品初加工": ["**农产品初加工 关联知识** ...sample fragment B..."],
    }

    timeline: list[tuple[str, str, float, str]] = []
    timeline_lock = threading.Lock()

    def _stub_reduce(self, parts, group_label, force_compress=False):
        t0 = time.monotonic()
        with timeline_lock:
            timeline.append((group_label, "start", t0, threading.current_thread().name))
        time.sleep(2.0)
        t1 = time.monotonic()
        with timeline_lock:
            timeline.append((group_label, "end", t1, threading.current_thread().name))
        return f"[stub-summary] {group_label} (force_compress={force_compress})"

    graph = AgentGraph.__new__(AgentGraph)
    graph.summary_batch_size = 3

    with patch.object(AgentGraph, "_reduce_group_to_single", _stub_reduce):
        wall_t0 = time.monotonic()
        result_groups = AgentGraph._merge_orphan_groups_into_groups(
            graph, groups, orphan_by_kw,
        )
        wall_t1 = time.monotonic()

    print(f"\n[result] groups returned: {len(result_groups)}")
    for g in result_groups:
        kind_kw = f"{g.kind}/{g.keyword!r}"
        print(f"  - kind={kind_kw} parts={len(g.parts)} summary={g.summary!r}")

    base = timeline[0][2] if timeline else 0.0
    print(f"\n[timeline]")
    for label, evt, t, th in timeline:
        rel = (t - base) * 1000
        print(f"  +{rel:7.1f}ms {evt:5s} group={label:30s} thread={th}")

    starts = sorted([t for _, evt, t, _ in timeline if evt == "start"])
    if len(starts) >= 2:
        gap_ms = (starts[1] - starts[0]) * 1000
        print(f"\n[gap] 第二组启动时间 - 第一组启动时间 = {gap_ms:.1f}ms")
        print(f"[wall] 总耗时 = {(wall_t1 - wall_t0) * 1000:.1f}ms (串行应≈4000ms，并行应≈2000ms)")
        if gap_ms < 200 and (wall_t1 - wall_t0) < 3.0:
            print("[verdict] PASS: 两组同时启动，总耗时接近单组耗时 → 跨组并行已生效")
        else:
            print("[verdict] FAIL: 仍存在串行迹象")


if __name__ == "__main__":
    main()
