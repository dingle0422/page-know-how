import os
import json
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm.client import chat
from reasoner.v1.react_agent import ReactAgent, AgentResult
from reasoner.v1.prompts import (
    SUMMARY_PROMPT,
    SUMMARY_AND_CLEAN_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
    CLEAN_ANSWER_PROMPT,
    BATCH_SUMMARY_PROMPT,
    BATCH_MERGE_PROMPT,
    BATCH_MERGE_AND_CLEAN_PROMPT,
    RETRIEVAL_SUMMARY_PROMPT,
    RETRIEVAL_SUMMARY_AND_CLEAN_PROMPT,
    RETRIEVAL_BATCH_SUMMARY_PROMPT,
    RETRIEVAL_BATCH_MERGE_PROMPT,
    RETRIEVAL_BATCH_MERGE_AND_CLEAN_PROMPT,
    CHUNK_REASONING_PROMPT,
    CHUNK_REASONING_WITH_PITFALLS_PROMPT,
)
from reasoner.v1.chunk_builder import build_knowledge_chunks, natural_dir_sort_key

from reasoner._registries import (
    ExploredRegistry,
    PitfallsRegistry,
    KnowledgeFragment,
    RetrievalKnowledgeRegistry,
)

from skills import (
    SkillRunner,
    SkillResultRegistry,
    SkillRecord,
    evaluate_and_run,
    select_extra_skills,
    check_and_enhance,
)
from skills.double_check import _enhance_with_skill_results

logger = logging.getLogger(__name__)


class AgentGraph:
    """管理子智能体的衍生、并行执行和结果汇聚（v1：无 BacktrackIntent 机制）"""

    def __init__(
        self,
        question: str,
        knowledge_root: str,
        max_rounds: int = 5,
        vendor: str = "aliyun",
        model: str = "deepseek-v3.2",
        clean_answer: bool = False,
        summary_batch_size: int = 0,
        retrieval_mode: bool = False,
        check_pitfalls: bool = False,
        chunk_size: int = 0,
        enable_skills: bool = True,
        summary_clean_answer: bool = False,
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
        """主推理：根智能体 -> 子智能体并行 -> 展平结果"""
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
        )
        root_result = root_agent.run()
        self.all_results = self._flatten_results(root_result)
        logger.info(f"第一轮汇聚完成，共 {len(self.all_results)} 个智能体结果")

    def _run_double_check(self, raw_answer: str) -> str:
        """[兼容入口] 旧版串行 double-check：summary 完成后再做 skill 注入 / 补充。

        新流程已改用 _orchestrate_summary_with_double_check 实现 summary || double-check 并行；
        本方法保留以备外部调用。
        """
        try:
            return asyncio.run(check_and_enhance(
                question=self.question,
                raw_answer=raw_answer,
                registry=self.skill_registry,
                runner=self.skill_runner,
                vendor=self.vendor,
                model=self.model,
            ))
        except Exception as e:
            logger.exception(f"[Skill] double-check 阶段异常，沿用原回答: {e}")
            return raw_answer

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

    def _judge_extra_skills(self, exclude: set[str]) -> list[str]:
        """让 LLM 基于 question 二次判定还需哪些 skill；已完成的从可选项隐去"""
        try:
            return select_extra_skills(
                question=self.question,
                exclude=exclude,
                vendor=self.vendor,
                model=self.model,
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

    def _enhance_with_new_records(
        self, raw_answer: str, new_records: list[SkillRecord]
    ) -> str:
        """仅基于"新跑出来的 skill records"做 enhance；已被 summary 吸收的老 records 不再重复 load"""
        skill_context = self._render_skill_records(new_records)
        if not skill_context:
            return raw_answer
        try:
            return _enhance_with_skill_results(
                question=self.question,
                raw_answer=raw_answer,
                skill_context=skill_context,
                vendor=self.vendor,
                model=self.model,
            )
        except Exception as e:
            logger.error(f"[DoubleCheck] enhance 调用失败，沿用 summary 原文: {e}")
            return raw_answer

    def _orchestrate_summary_with_double_check(self, summary_callable) -> str:
        """三路并行：summary 主线 || extra-skill 判定 || （按需）执行新 skill。

        语义：
        - summary 主线已经把 phase-0 完成的 skill 结果一并喂进了最终合并 prompt
        - 与此同时让 LLM 仅基于 question 判定"还差哪些 skill"（已完成的从可选项隐去）
          - 若无新需求 → summary 完成即返回，省掉一次 enhance LLM 调用
          - 若有新需求 → 立刻并行启动这些 skill 的 step2+exec；summary 完成后等 skill，
                       再用"新增的 records"做 enhance，避免重复 load 已经被 summary 看过的老结果
        """
        if not self.enable_skills or not self.skill_registry:
            return summary_callable()

        done_records_snapshot = self.skill_registry.get_all()
        done_skill_names = {r.skill_name for r in done_records_snapshot}

        with ThreadPoolExecutor(max_workers=3) as executor:
            summary_future = executor.submit(summary_callable)
            judge_future = executor.submit(self._judge_extra_skills, done_skill_names)

            try:
                extra_needed = judge_future.result()
            except Exception as e:
                logger.error(f"[DoubleCheck] 判定异常，按无新需求处理: {e}")
                extra_needed = []

            extra_future = (
                executor.submit(self._run_extra_skills, extra_needed)
                if extra_needed else None
            )

            raw_answer = summary_future.result()

            if extra_future is None:
                logger.info(
                    "[DoubleCheck] 无新 skill 需求（已完成的 %d 个 skill 已在 summary 里被 load），"
                    "省掉 enhance 调用直接采用 summary 输出",
                    len(done_records_snapshot),
                )
                return raw_answer

            new_records = extra_future.result()

        if not new_records:
            logger.warning("[DoubleCheck] 补跑 skill 后未产出有效 records，沿用 summary 原文")
            return raw_answer

        logger.info(
            "[DoubleCheck] 用 %d 条新增 skill records 做 enhance（老的 %d 条已在 summary 里）",
            len(new_records), len(done_records_snapshot),
        )
        return self._enhance_with_new_records(raw_answer, new_records)

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

        # phase-0 完成的 skill 结果（快照），用于：
        # (1) 喂给 summary 最终合并 prompt
        # (2) 让 double-check 在 judge 阶段把这些 skill 隐去，避免重复选取
        if self.enable_skills and self.skill_registry:
            done_records_snapshot = self.skill_registry.get_all()
        else:
            done_records_snapshot = []
        summary_skill_context = self._build_skill_context_for_summary(done_records_snapshot)

        def _summary_callable():
            if self.retrieval_mode:
                return self._retrieval_pipeline(skill_context=summary_skill_context)
            return self._standard_pipeline(skill_context=summary_skill_context)

        answer = self._orchestrate_summary_with_double_check(_summary_callable)

        if self.summary_clean_answer:
            logger.info(
                "[CleanAnswer] summary_clean_answer 已启用，"
                "已在最终 summary/merge 阶段一并完成清洗，跳过独立清洗调用"
            )
        elif self.clean_answer:
            answer = self._clean_answer(answer)

        trace_log = self._build_trace_log()
        relevant_chapters = self._collect_relevant_chapters()

        return {
            "answer": answer,
            "trace_log": trace_log,
            "relevant_chapters": relevant_chapters,
        }

    # ========================= Chunk 模式 =========================

    def _run_chunk_mode(self) -> dict:
        """Chunk 模式：程序化分块 -> 并行推理 -> 分批汇总"""
        logger.info(
            f"AgentGraph 启动 [Chunk模式 chunk_size={self.chunk_size}]: "
            f"问题={self.question[:50]}..."
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
            return self.run.__wrapped__(self) if hasattr(self.run, '__wrapped__') else {
                "answer": "知识目录为空，无法生成回答。",
                "trace_log": "",
                "relevant_chapters": [],
            }

        self._chunk_directories = []
        for chunk in chunks:
            self._chunk_directories.extend(chunk.directories)

        if self.enable_skills:
            with ThreadPoolExecutor(max_workers=2) as executor:
                skill_future = executor.submit(self._run_skill_evaluation)
                reason_future = executor.submit(self._chunk_reason_phase, chunks)
                summary_parts = reason_future.result()
                skill_future.result()
        else:
            summary_parts = self._chunk_reason_phase(chunks)

        if self.enable_skills and self.skill_registry:
            done_records_snapshot = self.skill_registry.get_all()
        else:
            done_records_snapshot = []
        summary_skill_context = self._build_skill_context_for_summary(done_records_snapshot)

        def _chunk_summary_callable():
            return self._chunk_finalize_summary(summary_parts, summary_skill_context)

        answer = self._orchestrate_summary_with_double_check(_chunk_summary_callable)

        if self.summary_clean_answer:
            logger.info(
                "[CleanAnswer] summary_clean_answer 已启用，"
                "已在最终 summary/merge 阶段一并完成清洗，跳过独立清洗调用"
            )
        elif self.clean_answer:
            answer = self._clean_answer(answer)

        trace_log = self._build_chunk_trace_log(chunks)
        relevant_chapters = self._collect_relevant_chapters()

        return {
            "answer": answer,
            "trace_log": trace_log,
            "relevant_chapters": relevant_chapters,
        }

    def _chunk_reason_phase(self, chunks) -> list[str]:
        """Chunk 模式 phase 1：并行推理每个 chunk → 解析 JSON → 拼成 summary_parts。

        与 skill_eval 并行执行，不做最终合并（合并交给 orchestrator 与 skill 上下文一并处理）。
        """
        total_chunks = len(chunks)
        logger.info(f"[Chunk] 开始并行推理 {total_chunks} 个知识块")

        raw_results = [None] * total_chunks
        with ThreadPoolExecutor(max_workers=min(total_chunks, 10)) as executor:
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
                            system=SUMMARY_SYSTEM_PROMPT)
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
        """为 chunk 模式构建追踪日志，包含各块引用的相关章节"""
        lines = [f"=== Chunk 模式 (chunk_size={self.chunk_size}) ===", ""]

        results = getattr(self, '_chunk_reasoning_results', None) or []

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
        """程序化梳理：按层级排序、去重、格式化（不调用 LLM）"""
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
            organized.append(text)

        logger.info(
            f"[Retrieval] 知识片段梳理：{len(fragments)} 个片段，"
            f"层级深度范围 {min(len(f.heading_path) for f in fragments)}"
            f"-{max(len(f.heading_path) for f in fragments)}"
        )
        return organized

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
        """构建子智能体结果的文本片段列表"""
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
            agent_parts.append(part)
        return agent_parts

    def _final_summary(self, skill_context: str = "") -> str:
        """调用 LLM 做最终汇总（单次全量）"""
        agent_parts = self._build_evidence_parts()

        if self.summary_clean_answer:
            template = SUMMARY_AND_CLEAN_PROMPT
            mode_label = "汇总+清洗一体"
        else:
            template = SUMMARY_PROMPT
            mode_label = "纯汇总"
        prompt = template.format(
            question=self.question,
            agent_results="\n\n".join(agent_parts) if agent_parts else "（无）",
        )
        prompt = self._append_skill_context_to_prompt(prompt, skill_context)

        logger.info(f"[Summary·{mode_label}] system prompt 长度: {len(SUMMARY_SYSTEM_PROMPT)} 字符")
        logger.info(f"[Summary·{mode_label}] user prompt 长度: {len(prompt)} 字符")
        logger.info(f"[Summary·{mode_label}] user prompt 内容:\n{prompt}")

        try:
            return chat(prompt, vendor=self.vendor, model=self.model, system=SUMMARY_SYSTEM_PROMPT)
        except Exception as e:
            logger.error(f"最终汇总 LLM 调用失败: {e}")
            parts = []
            for r in self.all_results:
                if r.conclusion:
                    parts.append(f"- [{r.agent_id}] {r.conclusion}")
            return "最终汇总生成失败，以下为各智能体的原始结论：\n" + "\n".join(parts)

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
                            system=SUMMARY_SYSTEM_PROMPT)
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

        try:
            return chat(merge_prompt, vendor=self.vendor, model=self.model,
                        system=SUMMARY_SYSTEM_PROMPT)
        except Exception as e:
            logger.error(f"[BatchMerge] 最终合并失败: {e}")
            return "分批合并失败，以下为各摘要：\n" + numbered

    def _retrieval_final_summary(self, organized_parts: list[str], skill_context: str = "") -> str:
        knowledge_text = "\n\n".join(organized_parts)
        if self.summary_clean_answer:
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

        try:
            return chat(prompt, vendor=self.vendor, model=self.model,
                        system=SUMMARY_SYSTEM_PROMPT)
        except Exception as e:
            logger.error(f"[RetrievalSummary] 召回总结失败: {e}")
            return "召回总结生成失败，以下为召回的知识片段：\n" + knowledge_text

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
                            system=SUMMARY_SYSTEM_PROMPT)
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

        try:
            return chat(merge_prompt, vendor=self.vendor, model=self.model,
                        system=SUMMARY_SYSTEM_PROMPT)
        except Exception as e:
            logger.error(f"[RetrievalBatchMerge] 最终合并失败: {e}")
            return "召回分批合并失败，以下为各摘要：\n" + numbered

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
