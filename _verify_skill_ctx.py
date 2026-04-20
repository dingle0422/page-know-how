import sys
sys.stdout.reconfigure(encoding="utf-8")

from reasoner.v1.agent_graph import AgentGraph
from skills.runner import SkillExecutionResult as SkillResult
from skills import SkillRecord
from reasoner.v1.prompts import (
    SUMMARY_AND_CLEAN_PROMPT, BATCH_MERGE_AND_CLEAN_PROMPT,
    RETRIEVAL_SUMMARY_AND_CLEAN_PROMPT, RETRIEVAL_BATCH_MERGE_AND_CLEAN_PROMPT,
    SUMMARY_PROMPT, BATCH_MERGE_PROMPT, RETRIEVAL_SUMMARY_PROMPT, RETRIEVAL_BATCH_MERGE_PROMPT,
)

records = [
    SkillRecord(
        skill_name="standard_product_name_verification",
        command='python -m skills.standard_product_name_verification "圣女果"',
        result=SkillResult(
            success=True, exit_code=0,
            stdout=(
                "「圣女果」候选匹配:\n"
                "  - 番茄等茄果类蔬菜（蔬菜），税率 9%，匹配度 1.0\n"
                "  - 其他未列明水果（水果），税率 9%，匹配度 0.463"
            ),
            stderr="",
        ),
    ),
]

rendered = AgentGraph._render_skill_records(records)
print("=== _render_skill_records 输出 ===")
print(rendered)
print()
assert "命令:" not in rendered, "FAIL: 仍包含 命令: 行"
assert "【Skill 调用 1】standard_product_name_verification" in rendered
assert "结果:" in rendered
print("CHECK 1 PASS: 命令: 行已移除\n")

skill_ctx = (
    "\n\n## 外部 Skill 已确认的权威事实"
    "（请在最终回答中以此为准；若与上文知识/摘要冲突，必须以本节为准重新推理并覆盖结论）\n"
    + rendered
)

cases = [
    ("SUMMARY_AND_CLEAN_PROMPT", SUMMARY_AND_CLEAN_PROMPT),
    ("BATCH_MERGE_AND_CLEAN_PROMPT", BATCH_MERGE_AND_CLEAN_PROMPT),
    ("RETRIEVAL_SUMMARY_AND_CLEAN_PROMPT", RETRIEVAL_SUMMARY_AND_CLEAN_PROMPT),
    ("RETRIEVAL_BATCH_MERGE_AND_CLEAN_PROMPT", RETRIEVAL_BATCH_MERGE_AND_CLEAN_PROMPT),
    ("SUMMARY_PROMPT", SUMMARY_PROMPT),
    ("BATCH_MERGE_PROMPT", BATCH_MERGE_PROMPT),
    ("RETRIEVAL_SUMMARY_PROMPT", RETRIEVAL_SUMMARY_PROMPT),
    ("RETRIEVAL_BATCH_MERGE_PROMPT", RETRIEVAL_BATCH_MERGE_PROMPT),
]
fields = {"question": "Q", "agent_results": "AR", "organized_knowledge": "OK", "batch_summaries": "BS"}

print("=== _append_skill_context_to_prompt 顺序校验 ===")
for name, tmpl in cases:
    safe = {k: v for k, v in fields.items() if "{" + k + "}" in tmpl}
    p0 = tmpl.format(**safe)
    p1 = AgentGraph._append_skill_context_to_prompt(p0, skill_ctx)
    skill_idx = p1.find("## 外部 Skill 已确认的权威事实")
    sep_idx = p1.rfind("\n---\n")
    out_candidates = [p1.find(s) for s in ("### 输出要求", "请综合", "请基于以上")]
    output_idx = max([i for i in out_candidates if i >= 0] or [-1])
    ok = skill_idx > 0 and sep_idx > 0 and skill_idx < sep_idx < output_idx
    print(f"{name:42s} skill@{skill_idx:5d} sep@{sep_idx:5d} out@{output_idx:5d} {'PASS' if ok else 'FAIL'}")
    assert ok, f"{name} 顺序错误"

print("\nCHECK 2 PASS: skill 段插入到 --- 之前，所有 8 个模板均生效\n")

print("=== RETRIEVAL_BATCH_MERGE_AND_CLEAN_PROMPT 拼接后 skill 段周边 (600 字) ===")
sample = AgentGraph._append_skill_context_to_prompt(
    RETRIEVAL_BATCH_MERGE_AND_CLEAN_PROMPT.format(question="Q", batch_summaries="BS"),
    skill_ctx,
)
sk_idx = sample.find("## 外部 Skill")
print(sample[max(0, sk_idx - 80):sk_idx + 700])
