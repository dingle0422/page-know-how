import os
import json
import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from llm.client import chat
from reasoner.v1.react_agent import ReactAgent, AgentResult
from reasoner.v1.prompts import (
    SUMMARY_PROMPT,
    SUMMARY_AND_CLEAN_PROMPT,
    SUMMARY_AND_CLEAN_THINK_PROMPT,
    SUMMARY_EXTRACT_SYSTEM_PROMPT,
    SUMMARY_ANSWER_SYSTEM_PROMPT,
    CLEAN_ANSWER_PROMPT,
    BATCH_SUMMARY_PROMPT,
    BATCH_MERGE_PROMPT,
    BATCH_MERGE_AND_CLEAN_PROMPT,
    BATCH_MERGE_AND_CLEAN_THINK_PROMPT,
    RETRIEVAL_SUMMARY_PROMPT,
    RETRIEVAL_SUMMARY_AND_CLEAN_PROMPT,
    RETRIEVAL_SUMMARY_AND_CLEAN_THINK_PROMPT,
    RETRIEVAL_BATCH_SUMMARY_PROMPT,
    RETRIEVAL_BATCH_MERGE_PROMPT,
    RETRIEVAL_BATCH_MERGE_AND_CLEAN_PROMPT,
    RETRIEVAL_BATCH_MERGE_AND_CLEAN_THINK_PROMPT,
    CHUNK_REASONING_PROMPT,
    CHUNK_REASONING_WITH_PITFALLS_PROMPT,
    ALL_IN_ANSWER_PROMPT,
)
from reasoner.v1.chunk_builder import (
    build_knowledge_chunks,
    split_relations_into_chunks,
    natural_dir_sort_key,
    KnowledgeChunk,
)
from reasoner.v1.clause_locator import ClauseLocator
from reasoner.v1.relation_crawler import RelationCrawler

from reasoner._registries import (
    ExploredRegistry,
    PitfallsRegistry,
    KnowledgeFragment,
    RetrievalKnowledgeRegistry,
    RelationRegistry,
    RelationFragment,
)

from skills import (
    SkillRunner,
    SkillResultRegistry,
    SkillRecord,
    evaluate_and_run,
    select_extra_skills,
)

logger = logging.getLogger(__name__)


@dataclass
class _OrderedSlot:
    """流式 chunk 调度的状态机单元（每个原始 chunk 一份）。

    生命周期 (按 Event 触发顺序)：
        self_done    ← 原始 chunk LLM 完成
        relation_done← 关联展开（crawl + 派生 chunk 切分）完成；不需要展开则与 self_done 同时 set
        derived_done ← 所有派生 chunk LLM 完成；无派生则与 relation_done 同时 set

    调度线程按 slot 顺序 wait derived_done，依次把 self + derived parts 喂给 batch summary。
    """
    parent_index: int
    self_chunk: KnowledgeChunk | None = None
    self_result: dict | None = None
    derived_chunks: list[KnowledgeChunk] = field(default_factory=list)
    derived_results: dict[int, dict] = field(default_factory=dict)  # derived_seq -> result
    relation_fragments: list[RelationFragment] = field(default_factory=list)

    self_done: threading.Event = field(default_factory=threading.Event)
    relation_done: threading.Event = field(default_factory=threading.Event)
    derived_done: threading.Event = field(default_factory=threading.Event)

    pending_derived: int = 0
    completed_derived: int = 0
    derived_lock: threading.Lock = field(default_factory=threading.Lock)


class AgentGraph:
    """管理子智能体的衍生、并行执行和结果汇聚（v1：无 BacktrackIntent 机制）"""

    def __init__(
        self,
        question: str,
        knowledge_root: str,
        max_rounds: int = 5,
        vendor: str = "qwen3.5-122b-a10b",
        model: str = "Qwen3.5-122B-A10B",
        clean_answer: bool = False,
        summary_batch_size: int = 0,
        retrieval_mode: bool = False,
        check_pitfalls: bool = False,
        chunk_size: int = 0,
        enable_skills: bool = True,
        summary_clean_answer: bool = False,
        answer_system_prompt: str | None = None,
        think_mode: bool = False,
        enable_relations: bool = False,
        relation_max_depth: int = 3,
        relation_max_nodes: int = 50,
        relation_workers: int = 8,
        relation_remote_timeout: float = 5.0,
        page_knowledge_dir: str | None = None,
        policy_index_path: str | None = None,
    ):
        self.question = question
        self.knowledge_root = knowledge_root
        self.max_rounds = max_rounds
        self.vendor = vendor
        self.model = model
        self.clean_answer = clean_answer
        self.summary_batch_size = summary_batch_size
        self.retrieval_mode = retrieval_mode
        self.check_pitfalls = check_pitfalls
        self.chunk_size = chunk_size
        self.enable_skills = enable_skills
        self.summary_clean_answer = summary_clean_answer
        # think_mode：作用于【所有最终节点】的 *_AND_CLEAN_PROMPT，统一切换为
        # *_AND_CLEAN_THINK_PROMPT，要求模型按 <think>/<answer> 双标签输出。
        # 覆盖范围（不受分批/召回/chunk 影响）：
        #   - 非分批：SUMMARY_AND_CLEAN / RETRIEVAL_SUMMARY_AND_CLEAN
        #   - 分批  ：BATCH_MERGE_AND_CLEAN / RETRIEVAL_BATCH_MERGE_AND_CLEAN
        #             （chunk 模式最终也落到 batch_final_merge）
        # 中间提炼 prompt（BATCH_SUMMARY / RETRIEVAL_BATCH_SUMMARY / CHUNK_REASONING_*）
        # 始终保持原样，不受 think_mode 影响。
        self.think_mode = think_mode
        if self.think_mode and not self.summary_clean_answer:
            logger.warning(
                "[ThinkMode] think_mode=True 但 summary_clean_answer=False，"
                "本次将不会生效（think 版 prompt 仅作用于 *_AND_CLEAN 体系的最终总结）"
            )
        # 最终作答阶段的 system prompt：调用方（CLI/HTTP）可自定义；
        # 未传入或传入空字符串时回退到默认的 SUMMARY_ANSWER_SYSTEM_PROMPT。
        # 中间提炼层始终使用 SUMMARY_EXTRACT_SYSTEM_PROMPT，不受此参数影响。
        custom = (answer_system_prompt or "").strip() if answer_system_prompt is not None else ""
        self.answer_system_prompt = custom if custom else SUMMARY_ANSWER_SYSTEM_PROMPT
        if custom:
            logger.info(
                f"[AnswerSystemPrompt] 使用调用方自定义版本，长度: {len(self.answer_system_prompt)} 字符"
            )
        else:
            logger.info(
                f"[AnswerSystemPrompt] 使用默认 SUMMARY_ANSWER_SYSTEM_PROMPT，长度: {len(self.answer_system_prompt)} 字符"
            )
        self.registry = ExploredRegistry()
        self.pitfalls_registry = PitfallsRegistry()
        self.retrieval_registry = RetrievalKnowledgeRegistry() if retrieval_mode else None
        self.skill_registry: SkillResultRegistry | None = (
            SkillResultRegistry() if enable_skills else None
        )
        self.skill_runner: SkillRunner | None = SkillRunner() if enable_skills else None
        self.all_results: list[AgentResult] = []
        self._chunk_directories: list[str] = []
        self._chunk_relevant_headings: list[str] = []
        self._chunk_reasoning_results: list[dict] = []
        self._chunk_slots: list[_OrderedSlot] = []  # 流式调度产物，trace 阶段引用

        # ---------- 关联展开（可选） ----------
        self.enable_relations = enable_relations
        self.relation_max_depth = max(1, int(relation_max_depth))
        self.relation_max_nodes = max(1, int(relation_max_nodes))
        self.relation_workers = max(1, int(relation_workers))
        self.relation_remote_timeout = float(relation_remote_timeout)

        # 推导 page_knowledge 目录与索引路径（默认从 knowledge_root 的 parent 推断）。
        # 调用方（如 stress test）可显式传入 override。
        self.page_knowledge_dir = (
            os.path.abspath(page_knowledge_dir)
            if page_knowledge_dir
            else os.path.abspath(os.path.dirname(self.knowledge_root))
        )
        self.policy_index_path = (
            policy_index_path
            if policy_index_path
            else os.path.join(self.page_knowledge_dir, "_policy_index.json")
        )

        self.relation_registry: RelationRegistry | None = None
        self.clause_locator: ClauseLocator | None = None
        self.relation_crawler: RelationCrawler | None = None
        self._relation_eval_executor: ThreadPoolExecutor | None = None
        self._relation_dispatch_executor: ThreadPoolExecutor | None = None
        # 副路径（standard / retrieval）on_hit 触发的 crawl future 收集器
        self._react_hit_dirs: set[str] = set()
        self._react_relation_futures: list = []
        self._react_hit_lock = threading.Lock()

        if self.enable_relations:
            self.relation_registry = RelationRegistry()
            self.clause_locator = ClauseLocator(
                page_knowledge_dir=self.page_knowledge_dir,
                policy_index_path=self.policy_index_path,
                remote_timeout=self.relation_remote_timeout,
            )
            # 两个独立池，避免 BFS 内嵌套同池等待导致死锁：
            #   - eval 池：单候选 (_evaluate_single) 的 LLM 评估
            #   - dispatch 池：每个 source_dir 的 crawl() 整体调度
            self._relation_eval_executor = ThreadPoolExecutor(
                max_workers=self.relation_workers * 2,
                thread_name_prefix="rel-eval",
            )
            self._relation_dispatch_executor = ThreadPoolExecutor(
                max_workers=self.relation_workers,
                thread_name_prefix="rel-dispatch",
            )
            self.relation_crawler = RelationCrawler(
                question=self.question,
                registry=self.relation_registry,
                locator=self.clause_locator,
                executor=self._relation_eval_executor,
                vendor=self.vendor,
                model=self.model,
                max_depth=self.relation_max_depth,
                max_nodes=self.relation_max_nodes,
            )
            logger.info(
                f"[Relations] 已启用关联展开：max_depth={self.relation_max_depth}, "
                f"max_nodes={self.relation_max_nodes}, workers={self.relation_workers}, "
                f"page_knowledge_dir={self.page_knowledge_dir}"
            )

    def _shutdown_relation_executors(self) -> None:
        """关联展开相关线程池在推理结束时统一释放。重复调用幂等。"""
        for attr in ("_relation_eval_executor", "_relation_dispatch_executor"):
            ex = getattr(self, attr, None)
            if ex is not None:
                try:
                    ex.shutdown(wait=True)
                except Exception:
                    pass
                setattr(self, attr, None)

    def _run_skill_evaluation(self) -> None:
        """在独立线程中跑 evaluator 的 asyncio 事件循环"""
        try:
            asyncio.run(evaluate_and_run(
                question=self.question,
                registry=self.skill_registry,
                runner=self.skill_runner,
                vendor=self.vendor,
                model=self.model,
            ))
        except Exception as e:
            logger.exception(f"[Skill] 预评估阶段异常: {e}")

    def _run_root_agent_and_flatten(self) -> None:
        """主推理：根智能体 -> 子智能体并行 -> 展平结果。

        当 enable_relations=True 时，把 self._react_on_hit 作为 on_hit 回调注入到
        ReactAgent；任何 _assess_* 命中都会异步触发 RelationCrawler.crawl，
        crawl 结果在后续 _organize_fragments / _build_evidence_parts 阶段被 inline 渲染。
        """
        on_hit_cb = self._react_on_hit if self.enable_relations else None
        root_agent = ReactAgent(
            question=self.question,
            knowledge_root=self.knowledge_root,
            current_dir=self.knowledge_root,
            upstream_path=[self.knowledge_root],
            parent_summary="",
            registry=self.registry,
            pitfalls_registry=self.pitfalls_registry,
            max_rounds=self.max_rounds,
            vendor=self.vendor,
            model=self.model,
            retrieval_mode=self.retrieval_mode,
            retrieval_registry=self.retrieval_registry,
            subtree_root=self.knowledge_root,
            on_hit=on_hit_cb,
        )
        root_result = root_agent.run()
        self.all_results = self._flatten_results(root_result)
        logger.info(f"第一轮汇聚完成，共 {len(self.all_results)} 个智能体结果")

    # ===== 副路径（standard / retrieval）关联展开钩子 =====

    def _react_on_hit(self, directory: str, assessment: dict) -> None:
        """ReactAgent 命中相关知识时调用：异步触发 RelationCrawler.crawl。

        - 不阻塞调用方（提交到 _relation_dispatch_executor 后立即返回）
        - 同一 directory 多次命中只 schedule 一次（用 _react_hit_dirs 去重）
        - parent_assessment 取 assessment 中的 reason / conclusion / summary
        """
        if not self.enable_relations or self.relation_crawler is None:
            return
        if self._relation_dispatch_executor is None:
            return
        if not os.path.isfile(os.path.join(directory, "clause.json")):
            return  # 无 clause.json 即无 references，不浪费 crawl

        with self._react_hit_lock:
            if directory in self._react_hit_dirs:
                return
            self._react_hit_dirs.add(directory)

        parent_assessment = (
            assessment.get("reason")
            or assessment.get("conclusion")
            or assessment.get("summary")
            or ""
        )
        if isinstance(parent_assessment, list):
            parent_assessment = " | ".join(str(x) for x in parent_assessment)
        parent_assessment = (parent_assessment or "")[:400]

        fut = self._relation_dispatch_executor.submit(
            self.relation_crawler.crawl,
            -1, directory, parent_assessment,
        )
        with self._react_hit_lock:
            self._react_relation_futures.append(fut)

    def _wait_react_relations(self) -> None:
        """阻塞等待 ReactAgent 触发的所有关联 crawl 完成；pipeline 进入前调用。"""
        if not self._react_relation_futures:
            return
        logger.info(
            f"[Relations] 等待副路径关联展开 {len(self._react_relation_futures)} 个 crawl 完成..."
        )
        for fut in self._react_relation_futures:
            try:
                fut.result()
            except Exception as e:
                logger.warning(f"[Relations] 副路径 crawl 异常已忽略: {e}")
        if self.relation_registry:
            logger.info(
                f"[Relations] 副路径 crawl 全部完成，共命中 "
                f"{len(self.relation_registry.get_all())} 个关联条款"
            )

    # ========================= Skill 上下文 & double-check 并行编排 =========================

    @staticmethod
    def _render_skill_records(records: list[SkillRecord]) -> str:
        """将 SkillRecord 列表渲染为可读文本（与 SkillResultRegistry.format_context 对齐，但只取指定 records）。

        命令行不展示，避免污染面向 LLM 的事实段；只保留 skill 名与结果/失败信息。
        """
        if not records:
            return ""
        lines: list[str] = []
        for i, rec in enumerate(records, 1):
            lines.append(f"【Skill 调用 {i}】{rec.skill_name}")
            if rec.result.success:
                lines.append("结果:")
                lines.append(rec.result.stdout or "（无输出）")
            else:
                lines.append(f"调用失败 (exit={rec.result.exit_code}):")
                if rec.result.stderr:
                    lines.append(rec.result.stderr)
            lines.append("")
        return "\n".join(lines).rstrip()

    def _build_skill_context_for_summary(self, records: list[SkillRecord]) -> str:
        """把 skill records 包装成可追加到 summary prompt 末尾的授权事实段。空 records → 空串。"""
        body = self._render_skill_records(records)
        if not body:
            return ""
        return (
            "\n\n## 外部 Skill 已确认的权威事实"
            "（请在最终回答中以此为准；若与上文知识/摘要冲突，必须以本节为准重新推理并覆盖结论）\n"
            + body
        )

    @staticmethod
    def _append_skill_context_to_prompt(prompt: str, skill_context: str) -> str:
        """把 skill_context 插入到 prompt 的"---"分隔符之前（即输出要求之前），
        让 skill 段与上下文知识/摘要并列作为输入事实，而非附加在输出指令之后。

        所有最终 summary/merge 模板都遵循 "{输入}\\n\\n---\\n\\n{输出要求}" 结构，
        以最后一个 "---" 作为切分点；若意外缺失，退化为追加到末尾。
        """
        if not skill_context:
            return prompt
        marker = "\n---\n"
        idx = prompt.rfind(marker)
        if idx < 0:
            return prompt + skill_context
        return prompt[:idx] + skill_context + prompt[idx:]

    def _judge_extra_skills(
        self, exclude: set[str], evidence: str | None = None
    ) -> list[str]:
        """让 LLM 基于 question + 已完成 skill + （可选）浓缩证据二次判定还需哪些 skill。

        与旧版的差异：evidence 不再为空——final summary 即将吃下去的 numbered summaries
        / agent_parts / organized_parts 会作为浓缩中间事实喂给 LLM，让它能基于"已经被
        提炼出来的关键事实"判定是否仍存在必须靠 skill 才能闭合的关键缺口。
        """
        try:
            return select_extra_skills(
                question=self.question,
                exclude=exclude,
                vendor=self.vendor,
                model=self.model,
                evidence=evidence,
            )
        except Exception as e:
            logger.exception(f"[DoubleCheck] 二次判定 extra skill 异常，视为无新需求: {e}")
            return []

    def _run_extra_skills(self, skill_names: list[str]) -> list[SkillRecord]:
        """跑指定的 skill，返回本次新增的 records（不含调用前已存在的）"""
        if not skill_names or not self.skill_registry or not self.skill_runner:
            return []
        before = len(self.skill_registry.get_all())
        try:
            asyncio.run(evaluate_and_run(
                question=self.question,
                registry=self.skill_registry,
                runner=self.skill_runner,
                vendor=self.vendor,
                model=self.model,
                skill_names=skill_names,
            ))
        except Exception as e:
            logger.exception(f"[DoubleCheck] 补充执行 skill 异常: {e}")
        after_records = self.skill_registry.get_all()
        return after_records[before:]

    def _all_in_answer(
        self, final_summary: str, extra_records: list[SkillRecord]
    ) -> str:
        """慢速路径专用：在 final_summary（已带 phase-0 注入 + 清洗）之后，
        把"本轮 judge 才补跑出来的 extra skill records"作为权威事实，最小修订草稿。

        不再覆盖 phase-0 records（它们已经在 final_summary 里被吸收了），避免重复 load。
        """
        skill_context = self._render_skill_records(extra_records)
        if not skill_context:
            return final_summary
        prompt = ALL_IN_ANSWER_PROMPT.format(
            question=self.question,
            final_summary=final_summary,
            skill_context=skill_context,
        )
        logger.info(f"[AllInAnswer] prompt 长度: {len(prompt)} 字符")
        logger.info(f"[AllInAnswer] prompt 内容:\n{prompt}")
        try:
            answer = chat(
                prompt, vendor=self.vendor, model=self.model,
                system=self.answer_system_prompt,
            )
            answer = (answer or "").strip()
            if not answer:
                logger.warning("[AllInAnswer] LLM 返回空，沿用 final summary 原文")
                return final_summary
            return answer
        except Exception as e:
            logger.error(f"[AllInAnswer] LLM 调用失败，沿用 final summary 原文: {e}")
            return final_summary

    def _finalize_with_double_check(
        self,
        final_summary_callable,
        judge_evidence: str,
        stage_label: str = "FinalSummary",
    ) -> str:
        """新版 double-check 编排：在每个最终 summary / merge 步骤内部使用。

        - final_summary_callable: 已经把 phase-0 skill_context 拼好的 final summary 调用
          （带清洗或不带清洗，由调用方决定）
        - judge_evidence: 喂给 judge 的浓缩证据（即将进入 final summary 的 numbered
          summaries / agent_parts / organized_parts）
        - 编排：summary || judge 并行；judge 无需求 → 直接返回 summary；judge 有需求 →
          起 extra-skill 并行跑，等 summary 与 extra-skill 都完成后走 _all_in_answer 拼接
        """
        if not self.enable_skills or not self.skill_registry:
            return final_summary_callable()

        done_records_snapshot = self.skill_registry.get_all()
        done_skill_names = {r.skill_name for r in done_records_snapshot}

        with ThreadPoolExecutor(max_workers=3) as executor:
            summary_future = executor.submit(final_summary_callable)
            judge_future = executor.submit(
                self._judge_extra_skills, done_skill_names, judge_evidence,
            )

            try:
                extra_needed = judge_future.result()
            except Exception as e:
                logger.error(f"[{stage_label}] judge 异常，按无新需求处理: {e}")
                extra_needed = []

            extra_future = (
                executor.submit(self._run_extra_skills, extra_needed)
                if extra_needed else None
            )

            final_summary = summary_future.result()

            if extra_future is None:
                logger.info(
                    f"[{stage_label}] judge 无新 skill 需求"
                    f"（phase-0 已完成 {len(done_records_snapshot)} 个 skill 并已注入），"
                    f"直接采用 final summary 输出"
                )
                return final_summary

            new_records = extra_future.result()

        if not new_records:
            logger.warning(
                f"[{stage_label}] 补跑 skill 后未产出有效 records，沿用 final summary 原文"
            )
            return final_summary

        logger.info(
            f"[{stage_label}] 用 {len(new_records)} 条新增 skill records 走 all-in-answer 拼接"
            f"（老的 {len(done_records_snapshot)} 条已在 final summary 里被吸收）"
        )
        return self._all_in_answer(final_summary, new_records)

    def run(self) -> dict:
        if self.chunk_size > 0:
            return self._run_chunk_mode()

        mode_label = "召回模式" if self.retrieval_mode else "标准模式"
        skill_label = "Skill开启" if self.enable_skills else "Skill关闭"
        logger.info(
            f"AgentGraph 启动 [{mode_label} | {skill_label}]: 问题={self.question[:50]}..."
        )

        if self.enable_skills:
            with ThreadPoolExecutor(max_workers=2) as executor:
                skill_future = executor.submit(self._run_skill_evaluation)
                reasoning_future = executor.submit(self._run_root_agent_and_flatten)
                reasoning_future.result()
                skill_future.result()
        else:
            self._run_root_agent_and_flatten()

        # 副路径关联展开 crawl 在子智能体命中时已开始；此处 join 后再进入 pipeline，
        # 保证 _organize_fragments / _build_evidence_parts 能拿到完整的 RelationRegistry。
        if self.enable_relations:
            self._wait_react_relations()

        # phase-0 完成的 skill 结果（快照）会被前置注入到最终 summary/merge 的 prompt；
        # judge 在 final 步骤内部触发，不在这里编排。
        if self.enable_skills and self.skill_registry:
            done_records_snapshot = self.skill_registry.get_all()
        else:
            done_records_snapshot = []
        summary_skill_context = self._build_skill_context_for_summary(done_records_snapshot)

        if self.retrieval_mode:
            answer = self._retrieval_pipeline(skill_context=summary_skill_context)
        else:
            answer = self._standard_pipeline(skill_context=summary_skill_context)

        if self.summary_clean_answer:
            logger.info(
                "[CleanAnswer] summary_clean_answer 已启用，"
                "已在最终 summary/merge 阶段一并完成清洗，跳过独立清洗调用"
            )
        elif self.clean_answer:
            answer = self._clean_answer(answer)

        trace_log = self._build_trace_log()
        relevant_chapters = self._collect_relevant_chapters()

        self._shutdown_relation_executors()

        return {
            "answer": answer,
            "trace_log": trace_log,
            "relevant_chapters": relevant_chapters,
        }

    # ========================= Chunk 模式 =========================

    def _run_chunk_mode(self) -> dict:
        """Chunk 模式：程序化分块 -> 并行推理（可选关联展开 + 流式调度）-> 分批汇总。

        流式调度（enable_relations=True 时）：
          1. 所有原始 chunk 并发跑 _reason_on_chunk LLM（chunk_pool）
          2. LLM 命中 relevant_headings 且 chunk 含 clause.json → 触发 RelationCrawler
             多跳并发拉取关联条款（relation_pool）
          3. 关联条款按 chunk_size 切分为派生 chunk → 提交到 chunk_pool 同样跑 LLM
          4. 调度线程按原始 chunk 顺序消费 ready slot 的 parts（self + derived），
             凑够 summary_batch_size 即提交 batch summary（batch_pool）
          5. 不带关联的 chunk 不会被带关联的 chunk 阻塞——只要 prefix 全部 ready
             就立刻喂入下一个 batch
          6. 所有 batch 完成后递归合并到最终 answer
        """
        logger.info(
            f"AgentGraph 启动 [Chunk模式 chunk_size={self.chunk_size}, "
            f"relations={self.enable_relations}]: 问题={self.question[:50]}..."
        )

        chunks = build_knowledge_chunks(self.knowledge_root, self.chunk_size)

        for chunk in chunks:
            logger.info(
                f"[Chunk] === 第 {chunk.index} 块知识内容 "
                f"({len(chunk.content)} 字符, {len(chunk.directories)} 个目录) ===\n"
                f"{chunk.content}\n"
                f"[Chunk] === 第 {chunk.index} 块结束 ==="
            )

        if not chunks:
            logger.warning("[Chunk] 未生成任何知识块，回退到标准模式")
            self._shutdown_relation_executors()
            return {
                "answer": "知识目录为空，无法生成回答。",
                "trace_log": "",
                "relevant_chapters": [],
            }

        self._chunk_directories = []
        for chunk in chunks:
            self._chunk_directories.extend(chunk.directories)

        skill_thread: threading.Thread | None = None
        if self.enable_skills:
            skill_thread = threading.Thread(
                target=self._run_skill_evaluation,
                name="skill-eval",
                daemon=True,
            )
            skill_thread.start()

        try:
            if self.enable_relations and self.relation_crawler is not None:
                batch_outputs, ordered_parts = self._chunk_streaming_pipeline(chunks)
            else:
                ordered_parts = self._chunk_reason_phase(chunks)
                batch_outputs = None
        finally:
            if skill_thread is not None:
                skill_thread.join()

        if self.enable_skills and self.skill_registry:
            done_records_snapshot = self.skill_registry.get_all()
        else:
            done_records_snapshot = []
        summary_skill_context = self._build_skill_context_for_summary(done_records_snapshot)

        if batch_outputs is not None:
            # 流式调度已经把 ordered_parts 凑批跑完 BATCH_SUMMARY，
            # 此处只剩"对 batch_outputs 做 final merge"。
            # 把 batch_outputs 喂回 _recursive_batch_reduce，逻辑会自动判断
            # 是否需要再压一层（>batch_size 时递归，否则直接 final merge）。
            answer = self._recursive_batch_reduce(
                batch_outputs, layer=2, skill_context=summary_skill_context,
            )
        else:
            answer = self._chunk_finalize_summary(ordered_parts, summary_skill_context)

        if self.summary_clean_answer:
            logger.info(
                "[CleanAnswer] summary_clean_answer 已启用，"
                "已在最终 summary/merge 阶段一并完成清洗，跳过独立清洗调用"
            )
        elif self.clean_answer:
            answer = self._clean_answer(answer)

        trace_log = self._build_chunk_trace_log(chunks)
        relevant_chapters = self._collect_relevant_chapters()

        self._shutdown_relation_executors()

        return {
            "answer": answer,
            "trace_log": trace_log,
            "relevant_chapters": relevant_chapters,
        }

    # ------------------- 流式调度（enable_relations=True） -------------------

    def _chunk_streaming_pipeline(
        self, chunks: list[KnowledgeChunk]
    ) -> tuple[list[str] | None, list[str]]:
        """三相并发流式 pipeline。

        返回 (batch_outputs, ordered_parts)：
          - summary_batch_size > 0 时：batch_outputs 为按顺序完成的 batch summaries 列表
            （已经被分批 LLM 压缩过），ordered_parts 同样填充以便 trace；
          - summary_batch_size == 0 时：batch_outputs 为 None，ordered_parts 是按顺序
            的全部 parts（原始 + 派生），由调用方自行做 final merge。
        """
        total_chunks = len(chunks)
        slots = [_OrderedSlot(parent_index=c.index, self_chunk=c) for c in chunks]
        self._chunk_slots = slots

        # chunk_pool 容量需同时容纳原始 + 估计派生 chunk 数
        chunk_pool_size = min(max(total_chunks * 2, 16), 100)
        chunk_pool = ThreadPoolExecutor(
            max_workers=chunk_pool_size, thread_name_prefix="chunk-llm",
        )

        batch_size = max(0, int(self.summary_batch_size))
        bs_pool: ThreadPoolExecutor | None = None
        if batch_size > 0:
            bs_pool = ThreadPoolExecutor(max_workers=5, thread_name_prefix="bs-stream")

        # 原始 chunk LLM 提交
        for chunk, slot in zip(chunks, slots):
            chunk_pool.submit(
                self._streaming_run_self, chunk, slot, total_chunks, chunk_pool,
            )

        ordered_parts: list[str] = []
        batch_outputs: list[str] = []

        if bs_pool is not None:
            ordered_parts, batch_outputs = self._streaming_orchestrate(
                slots, batch_size, bs_pool,
            )
        else:
            # 无分批：等所有 slot 完成后，按顺序拉 parts
            for slot in slots:
                slot.derived_done.wait()
            for slot in slots:
                ordered_parts.extend(self._render_slot_parts(slot, total_chunks))

        chunk_pool.shutdown(wait=True)
        if bs_pool is not None:
            bs_pool.shutdown(wait=True)

        # 汇总 _chunk_relevant_headings / _chunk_reasoning_results 与 trace_log 兼容
        self._chunk_reasoning_results = [
            slot.self_result if slot.self_result else {
                "relevant_headings": [], "analysis": "", "pitfalls": [],
            }
            for slot in slots
        ]
        for slot in slots:
            if slot.self_result:
                hs = slot.self_result.get("relevant_headings", []) or []
                self._chunk_relevant_headings.extend(hs)
            for d_idx in sorted(slot.derived_results.keys()):
                d_res = slot.derived_results[d_idx]
                hs = d_res.get("relevant_headings", []) or []
                self._chunk_relevant_headings.extend(hs)

        if bs_pool is not None:
            return batch_outputs, ordered_parts
        return None, ordered_parts

    def _streaming_run_self(
        self,
        chunk: KnowledgeChunk,
        slot: _OrderedSlot,
        total_chunks: int,
        chunk_pool: ThreadPoolExecutor,
    ) -> None:
        """worker：原始 chunk 自身 LLM → 触发关联展开 → 派生 chunk 入 chunk_pool。"""
        try:
            result = self._reason_on_chunk(chunk, total_chunks)
        except Exception as e:
            logger.error(f"[Streaming] chunk {chunk.index} 自身推理异常: {e}")
            result = {
                "relevant_headings": [],
                "analysis": f"（chunk {chunk.index} 推理失败: {e}）",
                "pitfalls": [],
            }
        slot.self_result = result
        slot.self_done.set()

        headings = result.get("relevant_headings") or []
        if not headings:
            slot.relation_done.set()
            slot.derived_done.set()
            return

        # 触发关联展开：把 relevant_headings 映射回 chunk 内带 clause.json 的目录
        targets = self._resolve_relation_targets(chunk, headings)
        if not targets:
            slot.relation_done.set()
            slot.derived_done.set()
            return

        parent_assessment = self._summarize_assessment_for_crawl(result)
        # crawl 在独立 dispatch 池跑（避免阻塞 chunk_pool worker）
        self._relation_dispatch_executor.submit(  # type: ignore[union-attr]
            self._streaming_run_relations,
            chunk, slot, targets, parent_assessment, chunk_pool,
        )

    def _streaming_run_relations(
        self,
        chunk: KnowledgeChunk,
        slot: _OrderedSlot,
        targets: list[str],
        parent_assessment: str,
        chunk_pool: ThreadPoolExecutor,
    ) -> None:
        """worker：对 chunk 内每个命中目录跑 RelationCrawler.crawl，合并 fragments。"""
        all_fragments: list[RelationFragment] = []
        try:
            for target_dir in targets:
                fragments = self.relation_crawler.crawl(  # type: ignore[union-attr]
                    source_chunk_index=chunk.index,
                    source_dir=target_dir,
                    parent_assessment=parent_assessment,
                )
                all_fragments.extend(fragments)
        except Exception as e:
            logger.error(f"[Streaming] chunk {chunk.index} 关联展开异常: {e}")
            all_fragments = []

        slot.relation_fragments = all_fragments

        if not all_fragments:
            slot.relation_done.set()
            slot.derived_done.set()
            return

        derived_chunks = split_relations_into_chunks(
            fragments=all_fragments,
            chunk_size=self.chunk_size,
            parent_chunk=chunk,
            start_derived_seq=1,
        )
        slot.derived_chunks = derived_chunks

        if not derived_chunks:
            slot.relation_done.set()
            slot.derived_done.set()
            return

        with slot.derived_lock:
            slot.pending_derived = len(derived_chunks)
            slot.completed_derived = 0
        slot.relation_done.set()

        # 派生 chunk 提交到 chunk_pool 跑同款 LLM
        for d_chunk in derived_chunks:
            chunk_pool.submit(
                self._streaming_run_derived, chunk, slot, d_chunk,
            )

    def _streaming_run_derived(
        self,
        parent_chunk: KnowledgeChunk,
        slot: _OrderedSlot,
        d_chunk: KnowledgeChunk,
    ) -> None:
        """worker：派生 chunk LLM 推理。"""
        try:
            # total_chunks 用 parent_index 配合 derived_seq 渲染（不影响 LLM 行为，仅日志）
            result = self._reason_on_chunk(d_chunk, total_chunks=0)
        except Exception as e:
            logger.error(
                f"[Streaming] derived chunk parent={parent_chunk.index} "
                f"seq={d_chunk.derived_seq} 推理异常: {e}"
            )
            result = {
                "relevant_headings": [],
                "analysis": f"（派生 chunk {parent_chunk.index}.{d_chunk.derived_seq} 推理失败: {e}）",
                "pitfalls": [],
            }
        slot.derived_results[d_chunk.derived_seq] = result
        with slot.derived_lock:
            slot.completed_derived += 1
            done = slot.completed_derived >= slot.pending_derived
        if done:
            slot.derived_done.set()

    def _streaming_orchestrate(
        self,
        slots: list[_OrderedSlot],
        batch_size: int,
        bs_pool: ThreadPoolExecutor,
    ) -> tuple[list[str], list[str]]:
        """head-of-line 调度：按 slot 顺序消费 ready parts，凑够 batch_size 提交 BATCH_SUMMARY。

        每个 slot 必须等三个 Event 全部 set 才算 ready；不 ready 时阻塞，但已就绪的前缀
        chunk 仍会立刻被打包为 batch（因为我们逐 slot 顺序处理：等到 slot[k] 全 ready 后，
        slot[0..k] 的 parts 都已纳入 pending_parts，凑够即 submit）。
        """
        total_chunks_estimate = len(slots)  # 仅用于 part 头部的 "i/N" 标签
        ordered_parts: list[str] = []
        pending_parts: list[str] = []
        batch_futures = []
        batch_index = 0

        def _flush_batch():
            nonlocal batch_index
            if not pending_parts:
                return
            batch_index += 1
            this_parts = pending_parts.copy()
            pending_parts.clear()
            fut = bs_pool.submit(
                self._streaming_submit_batch, batch_index, this_parts,
            )
            batch_futures.append(fut)

        for slot in slots:
            slot.derived_done.wait()
            slot_parts = self._render_slot_parts(slot, total_chunks_estimate)
            ordered_parts.extend(slot_parts)
            for part in slot_parts:
                pending_parts.append(part)
                if len(pending_parts) >= batch_size:
                    _flush_batch()

        _flush_batch()  # tail

        batch_outputs: list[str] = []
        for fut in batch_futures:
            try:
                batch_outputs.append(fut.result())
            except Exception as e:
                logger.error(f"[Streaming] batch summary future 异常: {e}")
                batch_outputs.append(f"（流式 batch summary 异常: {e}）")

        logger.info(
            f"[Streaming] 流式 batch 调度完成：共 {len(batch_futures)} 个 batch，"
            f"原始+派生 parts 总计 {len(ordered_parts)} 条"
        )
        return ordered_parts, batch_outputs

    def _streaming_submit_batch(self, batch_index: int, parts: list[str]) -> str:
        """流式 batch summary：复用 BATCH_SUMMARY_PROMPT 模板（中间提炼层，不带 clean）。"""
        batch_content = "\n\n".join(parts)
        # total_batches 在流式下未知；用占位"?"，模板对该字段无强约束
        prompt = BATCH_SUMMARY_PROMPT.format(
            batch_index=batch_index,
            total_batches="?",
            question=self.question,
            batch_content=batch_content,
        )
        logger.info(
            f"[StreamingBatch] 第 {batch_index} 批 prompt 长度: {len(prompt)} 字符 "
            f"({len(parts)} 个 parts)"
        )
        try:
            return chat(
                prompt, vendor=self.vendor, model=self.model,
                system=SUMMARY_EXTRACT_SYSTEM_PROMPT,
            )
        except Exception as e:
            logger.error(f"[StreamingBatch] 第 {batch_index} 批失败: {e}")
            return f"（第 {batch_index} 批流式压缩失败）\n原始内容:\n{batch_content}"

    # ------------------- 流式调度·辅助 -------------------

    def _render_slot_parts(
        self, slot: _OrderedSlot, total_chunks: int,
    ) -> list[str]:
        """把一个已完成的 slot 渲染为按顺序的 parts 列表（self + derived[1..M]）。"""
        parts: list[str] = []
        if slot.self_chunk and slot.self_result:
            parts.append(self._render_chunk_part(
                chunk_label=f"{slot.parent_index}/{total_chunks}",
                result=slot.self_result,
            ))
        for d_seq in sorted(slot.derived_results.keys()):
            d_res = slot.derived_results[d_seq]
            parts.append(self._render_chunk_part(
                chunk_label=f"{slot.parent_index}.{d_seq} (关联派生)",
                result=d_res,
            ))
        return parts

    @staticmethod
    def _render_chunk_part(chunk_label: str, result: dict) -> str:
        """与原 _chunk_reason_phase 内 part 渲染保持一致（label 之外完全同构）。"""
        analysis = result.get("analysis", "") or ""
        headings = result.get("relevant_headings", []) or []
        pitfalls = result.get("pitfalls", []) or []
        lines = [f"### 知识块 {chunk_label}"]
        if headings:
            lines.append(f"**引用章节：** {' | '.join(headings)}")
        lines.append(analysis)
        if pitfalls:
            lines.append("\n**【易错点提醒】**")
            for p in pitfalls:
                lines.append(f"- {p}")
        return "\n".join(lines)

    def _resolve_relation_targets(
        self, chunk: KnowledgeChunk, relevant_headings: list[str],
    ) -> list[str]:
        """把 LLM 输出的 relevant_headings 映射回 chunk 内带 clause.json 的目录绝对路径。

        匹配策略：把 chunk.heading_paths 与 chunk.directories 一一对应组成 (label, dir) 表，
        用 ' > '.join(hp) 与 cleaned heading 字符串相等命中。仅返回包含 clause.json 的 dir。
        """
        if not chunk.directories:
            return []
        # 构建查找表：label → dir
        label_to_dir: dict[str, str] = {}
        for hp, d in zip(chunk.heading_paths, chunk.directories):
            label = " > ".join(hp)
            label_to_dir[label] = d

        seen_dirs: set[str] = set()
        targets: list[str] = []
        for raw_h in relevant_headings:
            cleaned = (raw_h or "").strip().strip("【】").strip()
            d = label_to_dir.get(cleaned)
            if not d:
                continue
            if d in seen_dirs:
                continue
            if not os.path.isfile(os.path.join(d, "clause.json")):
                continue
            seen_dirs.add(d)
            targets.append(d)
        return targets

    @staticmethod
    def _summarize_assessment_for_crawl(result: dict, limit: int = 400) -> str:
        """把 chunk LLM 的 analysis 截断为给关联 crawl 用的 parent_assessment。

        优先用 analysis；analysis 为空时退化为 relevant_headings 拼接。
        """
        analysis = (result.get("analysis") or "").strip()
        if analysis:
            return analysis if len(analysis) <= limit else analysis[:limit] + "…"
        headings = result.get("relevant_headings") or []
        return " | ".join(headings) if headings else ""

    def _chunk_reason_phase(self, chunks) -> list[str]:
        """Chunk 模式 phase 1：并行推理每个 chunk → 解析 JSON → 拼成 summary_parts。

        与 skill_eval 并行执行，不做最终合并（合并交给 orchestrator 与 skill 上下文一并处理）。
        """
        total_chunks = len(chunks)
        logger.info(f"[Chunk] 开始并行推理 {total_chunks} 个知识块")

        raw_results = [None] * total_chunks
        with ThreadPoolExecutor(max_workers=min(total_chunks, 100)) as executor:
            futures = {
                executor.submit(
                    self._reason_on_chunk, chunk, total_chunks
                ): chunk.index - 1
                for chunk in chunks
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    raw_results[idx] = future.result()
                except Exception as e:
                    logger.error(f"[Chunk] 第 {idx+1} 块推理异常: {e}")
                    raw_results[idx] = {
                        "relevant_headings": [],
                        "analysis": f"（第 {idx+1} 块推理失败: {e}）",
                        "pitfalls": [],
                    }

        self._chunk_reasoning_results = raw_results

        summary_parts = []
        for i, result in enumerate(raw_results):
            analysis = result.get("analysis", "")
            headings = result.get("relevant_headings", [])
            pitfalls = result.get("pitfalls", [])

            if headings:
                self._chunk_relevant_headings.extend(headings)

            part_lines = [f"### 知识块 {i+1}/{total_chunks}"]
            if headings:
                part_lines.append(f"**引用章节：** {' | '.join(headings)}")
            part_lines.append(analysis)
            if pitfalls:
                part_lines.append("\n**【易错点提醒】**")
                for p in pitfalls:
                    part_lines.append(f"- {p}")

            summary_parts.append("\n".join(part_lines))

        logger.info(
            f"[Chunk] 所有 {total_chunks} 个知识块推理完成，"
            f"共引用 {len(self._chunk_relevant_headings)} 个相关章节"
        )
        return summary_parts

    def _chunk_finalize_summary(self, summary_parts: list[str], skill_context: str = "") -> str:
        """Chunk 模式 phase 2：把 phase 1 的 summary_parts 汇总为最终 answer。

        skill_context 会被注入到最终合并 prompt，与 _orchestrate_summary_with_double_check 配合使用。
        """
        if self.summary_batch_size > 0:
            return self._recursive_batch_reduce(summary_parts, layer=1, skill_context=skill_context)
        else:
            return self._batch_final_merge(summary_parts, layer=1, skill_context=skill_context)

    def _reason_on_chunk(self, chunk, total_chunks: int) -> dict:
        """对单个 chunk 调用 LLM 推理，返回解析后的 JSON dict"""
        template = (CHUNK_REASONING_WITH_PITFALLS_PROMPT
                    if self.check_pitfalls
                    else CHUNK_REASONING_PROMPT)
        prompt = template.format(
            question=self.question,
            chunk_index=chunk.index,
            total_chunks=total_chunks,
            chunk_content=chunk.content,
        )
        mode_label = "推理+易错点" if self.check_pitfalls else "推理"
        logger.info(
            f"[Chunk] 第 {chunk.index}/{total_chunks} 块{mode_label} "
            f"prompt 长度: {len(prompt)} 字符"
        )
        try:
            response = chat(prompt, vendor=self.vendor, model=self.model,
                            system=SUMMARY_EXTRACT_SYSTEM_PROMPT)
            return self._parse_chunk_json(response, chunk.index)
        except Exception as e:
            logger.error(f"[Chunk] 第 {chunk.index} 块 LLM 调用失败: {e}")
            return {
                "relevant_headings": [],
                "analysis": f"（第 {chunk.index} 块推理失败: {e}）",
                "pitfalls": [],
            }

    @staticmethod
    def _parse_chunk_json(response: str, chunk_index: int) -> dict:
        """解析 chunk 推理的 JSON 响应，解析失败时降级为纯文本"""
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        try:
            result = json.loads(cleaned.strip())
            if not isinstance(result.get("relevant_headings"), list):
                result["relevant_headings"] = []
            if not isinstance(result.get("analysis"), str):
                result["analysis"] = str(result.get("analysis", ""))
            if not isinstance(result.get("pitfalls"), list):
                result["pitfalls"] = []
            return result
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(
                f"[Chunk] 第 {chunk_index} 块 JSON 解析失败，降级为纯文本: {e}"
            )
            return {
                "relevant_headings": [],
                "analysis": response,
                "pitfalls": [],
            }

    def _build_chunk_trace_log(self, chunks) -> str:
        """为 chunk 模式构建追踪日志，包含各块引用的相关章节、关联展开命中条款、派生 chunk。"""
        lines = [
            f"=== Chunk 模式 (chunk_size={self.chunk_size}, "
            f"relations={self.enable_relations}) ===",
            "",
        ]

        results = getattr(self, '_chunk_reasoning_results', None) or []
        slots = getattr(self, '_chunk_slots', None) or []
        slots_by_index = {s.parent_index: s for s in slots}

        for chunk in chunks:
            lines.append(f"--- Chunk {chunk.index} ({len(chunk.content)} 字符) ---")
            lines.append(f"  包含目录节点:")
            for hp in chunk.heading_paths:
                lines.append(f"    - {' > '.join(hp)}")

            idx = chunk.index - 1
            if idx < len(results) and results[idx]:
                result = results[idx]
                headings = result.get("relevant_headings", [])
                pitfalls = result.get("pitfalls", [])
                if headings:
                    lines.append(f"  LLM 判定相关章节:")
                    for h in headings:
                        lines.append(f"    * {h}")
                else:
                    lines.append(f"  LLM 判定: 本块与问题无关")
                if pitfalls:
                    lines.append(f"  识别易错点: {len(pitfalls)} 条")

            slot = slots_by_index.get(chunk.index)
            if slot and slot.relation_fragments:
                lines.append(
                    f"  关联展开命中: {len(slot.relation_fragments)} 个条款 "
                    f"(派生 chunk {len(slot.derived_chunks)} 个)"
                )
                for frag in slot.relation_fragments:
                    heading_label = " > ".join(frag.heading_path) or frag.clause_full_name
                    reason = (frag.parent_assessment or "")[:80]
                    lines.append(
                        f"    * [hop={frag.hop_depth}/{frag.source}] "
                        f"{heading_label} (policy={frag.policy_id} clause={frag.clause_id})"
                    )
                    if reason:
                        lines.append(f"        上层关联性: {reason}")
                for d_chunk in slot.derived_chunks:
                    d_res = slot.derived_results.get(d_chunk.derived_seq, {})
                    d_headings = d_res.get("relevant_headings", []) or []
                    lines.append(
                        f"    - 派生 chunk {chunk.index}.{d_chunk.derived_seq} "
                        f"({len(d_chunk.content)} 字符, 命中 {len(d_headings)} 个 sub-heading)"
                    )
            lines.append("")

        return "\n".join(lines)

    def _standard_pipeline(self, skill_context: str = "") -> str:
        """标准模式的后处理管线：直接汇总（无意图解决）"""
        if self.summary_batch_size > 0:
            return self._batched_summary(skill_context=skill_context)
        else:
            return self._final_summary(skill_context=skill_context)

    def _retrieval_pipeline(self, skill_context: str = "") -> str:
        """召回模式的后处理管线：程序化梳理 -> 总结"""
        organized_parts = self._organize_fragments()

        if not organized_parts:
            logger.warning("[Retrieval] 未召回任何相关知识片段，回退到标准模式汇总")
            return self._standard_pipeline(skill_context=skill_context)

        logger.info(f"[Retrieval] 程序化梳理完成，共 {len(organized_parts)} 个知识片段")

        if self.summary_batch_size > 0:
            return self._retrieval_batched_summary(organized_parts, skill_context=skill_context)
        else:
            return self._retrieval_final_summary(organized_parts, skill_context=skill_context)

    def _organize_fragments(self) -> list[str]:
        """程序化梳理：按层级排序、去重、格式化（不调用 LLM）。

        启用 enable_relations 时，把每个 fragment 对应的 RelationRegistry.by_dir 命中
        条款 inline 渲染到 fragment.content 末尾，作为同一个 organized part 投递到
        retrieval pipeline；这样关联条款不会破坏 batch_size 切分逻辑，但完整保留了
        引用上下文。
        """
        fragments = self.retrieval_registry.get_all()
        if not fragments:
            return []

        sorted_fragments = sorted(
            fragments,
            key=lambda f: [
                (natural_dir_sort_key(seg), seg) for seg in f.heading_path
            ],
        )

        organized = []
        for frag in sorted_fragments:
            heading_label = " > ".join(frag.heading_path)
            text = f"【{heading_label}】\n{frag.content}"
            if self.check_pitfalls:
                dir_pitfalls = self.pitfalls_registry.get_by_dir(frag.directory_path)
                if dir_pitfalls:
                    pitfalls_str = "\n".join(f"- {p}" for p in dir_pitfalls)
                    text += f"\n\n**易错点提醒：**\n{pitfalls_str}"

            relation_block = self._render_inline_relations(frag.directory_path)
            if relation_block:
                text += "\n\n" + relation_block

            organized.append(text)

        logger.info(
            f"[Retrieval] 知识片段梳理：{len(fragments)} 个片段，"
            f"层级深度范围 {min(len(f.heading_path) for f in fragments)}"
            f"-{max(len(f.heading_path) for f in fragments)}"
        )
        return organized

    def _render_inline_relations(self, directory: str) -> str:
        """把 RelationRegistry 中按 dir 命中的 RelationFragment 渲染为追加到 fragment 末尾的字符串。

        与 chunk 派生 chunk 不同：副路径下关联条款不会单独成 part（避免破坏 batch_size 切分），
        而是 inline 追加到对应 fragment 的同一 part 内。
        """
        if not self.enable_relations or self.relation_registry is None:
            return ""
        rels = self.relation_registry.get_by_dir(directory)
        if not rels:
            return ""
        lines = ["**关联条款命中：**"]
        for r in rels:
            heading = " > ".join(r.heading_path) or r.clause_full_name
            lines.append(
                f"\n**[hop={r.hop_depth}/{r.source}] {heading}** "
                f"(policy={r.policy_id} clause={r.clause_id})"
            )
            if r.highlighted:
                lines.append(f"> 上层引用高亮: {r.highlighted}")
            lines.append(r.content or "（关联条款内容为空）")
        return "\n".join(lines)

    def _collect_relevant_chapters(self) -> list[str]:
        """从所有智能体结果或 chunk 目录中提取去重的相关章节序号"""
        chapters = set()

        if self._chunk_relevant_headings:
            for heading in self._chunk_relevant_headings:
                cleaned = heading.strip().strip("【】")
                parts = [p.strip() for p in cleaned.split(">")]
                for part in parts:
                    if "_" in part:
                        chapter_num = part.split("_")[0]
                        if chapter_num and chapter_num[0].isdigit():
                            chapters.add(chapter_num)
        else:
            all_dirs = []
            for r in self.all_results:
                all_dirs.extend(r.relevant_dirs)

            if self.retrieval_mode and self.retrieval_registry:
                for frag in self.retrieval_registry.get_all():
                    if frag.directory_path not in all_dirs:
                        all_dirs.append(frag.directory_path)

            for dir_path in all_dirs:
                rel = os.path.relpath(dir_path, self.knowledge_root)
                if rel == ".":
                    continue
                parts = rel.replace("\\", "/").split("/")
                leaf = parts[-1]
                if "_" in leaf:
                    chapter_num = leaf.split("_")[0]
                    chapters.add(chapter_num)

        return sorted(chapters, key=natural_dir_sort_key)

    def _flatten_results(self, result: AgentResult) -> list[AgentResult]:
        flat = [result]
        for child in result.child_results:
            flat.extend(self._flatten_results(child))
        return flat

    def _build_evidence_parts(self) -> list[str]:
        """构建子智能体结果的文本片段列表。

        启用关联展开时，把 r.relevant_dirs 中每个目录命中的 RelationFragment inline 追加
        到对应 part 末尾，不另起 part。
        """
        agent_parts = []
        for r in self.all_results:
            evidence_str = "\n".join(f"  - {e}" for e in r.evidence) if r.evidence else "  （无证据）"
            part = (
                f"### {r.agent_id} | 探索目录: {r.explored_dir}\n"
                f"- 结论: {r.conclusion}\n"
                f"- 证据:\n{evidence_str}"
            )
            if self.check_pitfalls and r.pitfalls:
                pitfalls_str = "\n".join(f"  - {p}" for p in r.pitfalls)
                part += f"\n- 易错点提醒:\n{pitfalls_str}"

            if self.enable_relations and self.relation_registry is not None:
                rel_blocks = [
                    self._render_inline_relations(d) for d in r.relevant_dirs
                ]
                rel_blocks = [b for b in rel_blocks if b]
                if rel_blocks:
                    part += "\n\n" + "\n\n".join(rel_blocks)
            agent_parts.append(part)
        return agent_parts

    def _final_summary(self, skill_context: str = "") -> str:
        """调用 LLM 做最终汇总（单次全量），并行触发 judge → 按需 all-in-answer"""
        agent_parts = self._build_evidence_parts()
        agent_results_text = "\n\n".join(agent_parts) if agent_parts else "（无）"

        if self.summary_clean_answer:
            if self.think_mode:
                template = SUMMARY_AND_CLEAN_THINK_PROMPT
                mode_label = "汇总+清洗一体·Think"
            else:
                template = SUMMARY_AND_CLEAN_PROMPT
                mode_label = "汇总+清洗一体"
        else:
            template = SUMMARY_PROMPT
            mode_label = "纯汇总"
        prompt = template.format(
            question=self.question,
            agent_results=agent_results_text,
        )
        prompt = self._append_skill_context_to_prompt(prompt, skill_context)

        logger.info(f"[Summary·{mode_label}] system prompt 长度: {len(self.answer_system_prompt)} 字符")
        logger.info(f"[Summary·{mode_label}] user prompt 长度: {len(prompt)} 字符")
        logger.info(f"[Summary·{mode_label}] user prompt 内容:\n{prompt}")

        def _do_summary() -> str:
            try:
                return chat(
                    prompt, vendor=self.vendor, model=self.model,
                    system=self.answer_system_prompt,
                )
            except Exception as e:
                logger.error(f"最终汇总 LLM 调用失败: {e}")
                fallback_parts = []
                for r in self.all_results:
                    if r.conclusion:
                        fallback_parts.append(f"- [{r.agent_id}] {r.conclusion}")
                return "最终汇总生成失败，以下为各智能体的原始结论：\n" + "\n".join(fallback_parts)

        return self._finalize_with_double_check(
            final_summary_callable=_do_summary,
            judge_evidence=agent_results_text,
            stage_label=f"Summary·{mode_label}",
        )

    def _batched_summary(self, skill_context: str = "") -> str:
        """分批并行压缩总结"""
        agent_parts = self._build_evidence_parts()
        return self._recursive_batch_reduce(agent_parts, layer=1, skill_context=skill_context)

    def _recursive_batch_reduce(self, parts: list[str], layer: int, skill_context: str = "") -> str:
        batch_size = self.summary_batch_size

        if len(parts) <= batch_size:
            return self._batch_final_merge(parts, layer, skill_context=skill_context)

        batches = [
            parts[i:i + batch_size]
            for i in range(0, len(parts), batch_size)
        ]
        total_batches = len(batches)

        logger.info(
            f"[BatchSummary] 第 {layer} 层：{len(parts)} 条 → "
            f"分为 {total_batches} 批（batch_size={batch_size}）"
        )

        def _summarize_batch(batch_index: int, batch_parts: list[str]) -> str:
            batch_content = "\n\n".join(batch_parts)
            prompt = BATCH_SUMMARY_PROMPT.format(
                batch_index=batch_index,
                total_batches=total_batches,
                question=self.question,
                batch_content=batch_content,
            )
            logger.info(
                f"[BatchSummary] 第 {layer} 层 第 {batch_index}/{total_batches} 批 "
                f"prompt 长度: {len(prompt)} 字符"
            )
            try:
                return chat(prompt, vendor=self.vendor, model=self.model,
                            system=SUMMARY_EXTRACT_SYSTEM_PROMPT)
            except Exception as e:
                logger.error(f"[BatchSummary] 第 {layer} 层 第 {batch_index} 批失败: {e}")
                return f"（第 {batch_index} 批压缩失败）\n原始内容:\n{batch_content}"

        batch_summaries = [None] * total_batches
        with ThreadPoolExecutor(max_workers=min(total_batches, 5)) as executor:
            futures = {
                executor.submit(_summarize_batch, i + 1, batch): i
                for i, batch in enumerate(batches)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    batch_summaries[idx] = future.result()
                except Exception as e:
                    logger.error(f"[BatchSummary] 第 {layer} 层 batch {idx+1} 异常: {e}")
                    batch_summaries[idx] = f"（第 {idx+1} 批执行异常: {e}）"

        logger.info(
            f"[BatchSummary] 第 {layer} 层完成，产出 {total_batches} 个摘要，"
            f"进入下一层"
        )

        return self._recursive_batch_reduce(batch_summaries, layer + 1, skill_context=skill_context)

    def _batch_final_merge(self, summaries: list[str], layer: int, skill_context: str = "") -> str:
        numbered = "\n\n".join(
            f"### 摘要 {i+1}\n{s}" for i, s in enumerate(summaries)
        )
        if self.summary_clean_answer:
            if self.think_mode:
                template = BATCH_MERGE_AND_CLEAN_THINK_PROMPT
                mode_label = "合并+清洗一体·Think"
            else:
                template = BATCH_MERGE_AND_CLEAN_PROMPT
                mode_label = "合并+清洗一体"
        else:
            template = BATCH_MERGE_PROMPT
            mode_label = "纯合并"
        merge_prompt = template.format(
            question=self.question,
            batch_summaries=numbered,
        )
        merge_prompt = self._append_skill_context_to_prompt(merge_prompt, skill_context)

        logger.info(
            f"[BatchMerge] 第 {layer} 层（最终合并·{mode_label}）："
            f"{len(summaries)} 条摘要，prompt 长度: {len(merge_prompt)} 字符"
        )
        logger.info(f"[BatchMerge] prompt 内容:\n{merge_prompt}")

        def _do_merge() -> str:
            try:
                return chat(merge_prompt, vendor=self.vendor, model=self.model,
                            system=self.answer_system_prompt)
            except Exception as e:
                logger.error(f"[BatchMerge] 最终合并失败: {e}")
                return "分批合并失败，以下为各摘要：\n" + numbered

        return self._finalize_with_double_check(
            final_summary_callable=_do_merge,
            judge_evidence=numbered,
            stage_label=f"BatchMerge·L{layer}·{mode_label}",
        )

    def _retrieval_final_summary(self, organized_parts: list[str], skill_context: str = "") -> str:
        knowledge_text = "\n\n".join(organized_parts)
        if self.summary_clean_answer:
            if self.think_mode:
                template = RETRIEVAL_SUMMARY_AND_CLEAN_THINK_PROMPT
                mode_label = "汇总+清洗一体·Think"
            else:
                template = RETRIEVAL_SUMMARY_AND_CLEAN_PROMPT
                mode_label = "汇总+清洗一体"
        else:
            template = RETRIEVAL_SUMMARY_PROMPT
            mode_label = "纯汇总"
        prompt = template.format(
            question=self.question,
            organized_knowledge=knowledge_text,
        )
        prompt = self._append_skill_context_to_prompt(prompt, skill_context)

        logger.info(f"[RetrievalSummary·{mode_label}] prompt 长度: {len(prompt)} 字符")
        logger.info(f"[RetrievalSummary·{mode_label}] prompt 内容:\n{prompt}")

        def _do_summary() -> str:
            try:
                return chat(prompt, vendor=self.vendor, model=self.model,
                            system=self.answer_system_prompt)
            except Exception as e:
                logger.error(f"[RetrievalSummary] 召回总结失败: {e}")
                return "召回总结生成失败，以下为召回的知识片段：\n" + knowledge_text

        return self._finalize_with_double_check(
            final_summary_callable=_do_summary,
            judge_evidence=knowledge_text,
            stage_label=f"RetrievalSummary·{mode_label}",
        )

    def _retrieval_batched_summary(self, organized_parts: list[str], skill_context: str = "") -> str:
        return self._retrieval_recursive_batch_reduce(organized_parts, layer=1, skill_context=skill_context)

    def _retrieval_recursive_batch_reduce(self, parts: list[str], layer: int, skill_context: str = "") -> str:
        batch_size = self.summary_batch_size

        if len(parts) <= batch_size:
            return self._retrieval_batch_final_merge(parts, layer, skill_context=skill_context)

        batches = [
            parts[i:i + batch_size]
            for i in range(0, len(parts), batch_size)
        ]
        total_batches = len(batches)

        logger.info(
            f"[RetrievalBatch] 第 {layer} 层：{len(parts)} 条 → "
            f"分为 {total_batches} 批（batch_size={batch_size}）"
        )

        def _summarize_batch(batch_index: int, batch_parts: list[str]) -> str:
            batch_content = "\n\n".join(batch_parts)
            prompt = RETRIEVAL_BATCH_SUMMARY_PROMPT.format(
                batch_index=batch_index,
                total_batches=total_batches,
                question=self.question,
                batch_content=batch_content,
            )
            logger.info(
                f"[RetrievalBatch] 第 {layer} 层 第 {batch_index}/{total_batches} 批 "
                f"prompt 长度: {len(prompt)} 字符"
            )
            try:
                return chat(prompt, vendor=self.vendor, model=self.model,
                            system=SUMMARY_EXTRACT_SYSTEM_PROMPT)
            except Exception as e:
                logger.error(f"[RetrievalBatch] 第 {layer} 层 第 {batch_index} 批失败: {e}")
                return f"（第 {batch_index} 批压缩失败）\n原始内容:\n{batch_content}"

        batch_summaries = [None] * total_batches
        with ThreadPoolExecutor(max_workers=min(total_batches, 5)) as executor:
            futures = {
                executor.submit(_summarize_batch, i + 1, batch): i
                for i, batch in enumerate(batches)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    batch_summaries[idx] = future.result()
                except Exception as e:
                    logger.error(f"[RetrievalBatch] 第 {layer} 层 batch {idx+1} 异常: {e}")
                    batch_summaries[idx] = f"（第 {idx+1} 批执行异常: {e}）"

        logger.info(
            f"[RetrievalBatch] 第 {layer} 层完成，产出 {total_batches} 个摘要，进入下一层"
        )
        return self._retrieval_recursive_batch_reduce(batch_summaries, layer + 1, skill_context=skill_context)

    def _retrieval_batch_final_merge(self, summaries: list[str], layer: int, skill_context: str = "") -> str:
        numbered = "\n\n".join(
            f"### 摘要 {i+1}\n{s}" for i, s in enumerate(summaries)
        )
        if self.summary_clean_answer:
            if self.think_mode:
                template = RETRIEVAL_BATCH_MERGE_AND_CLEAN_THINK_PROMPT
                mode_label = "合并+清洗一体·Think"
            else:
                template = RETRIEVAL_BATCH_MERGE_AND_CLEAN_PROMPT
                mode_label = "合并+清洗一体"
        else:
            template = RETRIEVAL_BATCH_MERGE_PROMPT
            mode_label = "纯合并"
        merge_prompt = template.format(
            question=self.question,
            batch_summaries=numbered,
        )
        merge_prompt = self._append_skill_context_to_prompt(merge_prompt, skill_context)

        logger.info(
            f"[RetrievalBatchMerge] 第 {layer} 层（最终合并·{mode_label}）："
            f"{len(summaries)} 条摘要，prompt 长度: {len(merge_prompt)} 字符"
        )
        logger.info(f"[RetrievalBatchMerge] prompt 内容:\n{merge_prompt}")

        def _do_merge() -> str:
            try:
                return chat(merge_prompt, vendor=self.vendor, model=self.model,
                            system=self.answer_system_prompt)
            except Exception as e:
                logger.error(f"[RetrievalBatchMerge] 最终合并失败: {e}")
                return "召回分批合并失败，以下为各摘要：\n" + numbered

        return self._finalize_with_double_check(
            final_summary_callable=_do_merge,
            judge_evidence=numbered,
            stage_label=f"RetrievalBatchMerge·L{layer}·{mode_label}",
        )

    def _clean_answer(self, raw_answer: str) -> str:
        prompt = CLEAN_ANSWER_PROMPT.format(
            question=self.question,
            raw_answer=raw_answer,
        )
        logger.info(f"[CleanAnswer] prompt 长度: {len(prompt)} 字符")
        logger.info(f"[CleanAnswer] prompt 内容:\n{prompt}")
        try:
            cleaned = chat(prompt, vendor=self.vendor, model=self.model)
            logger.info("答案清洗完成")
            return cleaned
        except Exception as e:
            logger.error(f"答案清洗 LLM 调用失败，返回原始 summary: {e}")
            return raw_answer

    def _build_trace_log(self) -> str:
        """工程化拼接所有智能体的游走路径和返回内容"""
        lines = []

        for r in self.all_results:
            lines.append(f"--- {r.agent_id} | 起始目录: {r.explored_dir} ---")
            for i, step in enumerate(r.trace, 1):
                lines.append(f"  [轮次{i}] {step.action:<15} {step.directory}")
                lines.append(f"           → {step.observation}")

            if r.conclusion:
                lines.append(f"  结论: {r.conclusion}")
            if r.evidence:
                lines.append(f"  证据: {'; '.join(r.evidence)}")
            lines.append("")

        return "\n".join(lines)
