import os
import json
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm.client import chat
from reasoner.v1.react_agent import ReactAgent, AgentResult
from reasoner.v1.prompts import (
    SUMMARY_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
    CLEAN_ANSWER_PROMPT,
    BATCH_SUMMARY_PROMPT,
    BATCH_MERGE_PROMPT,
    RETRIEVAL_SUMMARY_PROMPT,
    RETRIEVAL_BATCH_SUMMARY_PROMPT,
    RETRIEVAL_BATCH_MERGE_PROMPT,
    CHUNK_REASONING_PROMPT,
    CHUNK_REASONING_WITH_PITFALLS_PROMPT,
)
from reasoner.v1.chunk_builder import build_knowledge_chunks, natural_dir_sort_key

from reasoner.v0.agent_graph import (
    ExploredRegistry,
    PitfallsRegistry,
    KnowledgeFragment,
    RetrievalKnowledgeRegistry,
)

from skills import (
    SkillRunner,
    SkillResultRegistry,
    evaluate_and_run,
    check_and_enhance,
)

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
        """skill 结果注入 / 补充优化"""
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

        if self.retrieval_mode:
            answer = self._retrieval_pipeline()
        else:
            answer = self._standard_pipeline()

        if self.enable_skills:
            answer = self._run_double_check(answer)

        if self.clean_answer:
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
                chunk_future = executor.submit(self._chunk_pipeline, chunks)
                answer = chunk_future.result()
                skill_future.result()
        else:
            answer = self._chunk_pipeline(chunks)

        if self.enable_skills:
            answer = self._run_double_check(answer)

        if self.clean_answer:
            answer = self._clean_answer(answer)

        trace_log = self._build_chunk_trace_log(chunks)
        relevant_chapters = self._collect_relevant_chapters()

        return {
            "answer": answer,
            "trace_log": trace_log,
            "relevant_chapters": relevant_chapters,
        }

    def _chunk_pipeline(self, chunks) -> str:
        """并行推理每个 chunk，解析 JSON 结果，然后分批汇总"""
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
            f"共引用 {len(self._chunk_relevant_headings)} 个相关章节，"
            f"进入汇总阶段"
        )

        if self.summary_batch_size > 0:
            return self._recursive_batch_reduce(summary_parts, layer=1)
        else:
            return self._batch_final_merge(summary_parts, layer=1)

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

    def _standard_pipeline(self) -> str:
        """标准模式的后处理管线：直接汇总（无意图解决）"""
        if self.summary_batch_size > 0:
            return self._batched_summary()
        else:
            return self._final_summary()

    def _retrieval_pipeline(self) -> str:
        """召回模式的后处理管线：程序化梳理 -> 总结"""
        organized_parts = self._organize_fragments()

        if not organized_parts:
            logger.warning("[Retrieval] 未召回任何相关知识片段，回退到标准模式汇总")
            return self._standard_pipeline()

        logger.info(f"[Retrieval] 程序化梳理完成，共 {len(organized_parts)} 个知识片段")

        if self.summary_batch_size > 0:
            return self._retrieval_batched_summary(organized_parts)
        else:
            return self._retrieval_final_summary(organized_parts)

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

    def _final_summary(self) -> str:
        """调用 LLM 做最终汇总（单次全量）"""
        agent_parts = self._build_evidence_parts()

        prompt = SUMMARY_PROMPT.format(
            question=self.question,
            agent_results="\n\n".join(agent_parts) if agent_parts else "（无）",
        )

        logger.info(f"[Summary] system prompt 长度: {len(SUMMARY_SYSTEM_PROMPT)} 字符")
        logger.info(f"[Summary] user prompt 长度: {len(prompt)} 字符")
        logger.info(f"[Summary] user prompt 内容:\n{prompt}")

        try:
            return chat(prompt, vendor=self.vendor, model=self.model, system=SUMMARY_SYSTEM_PROMPT)
        except Exception as e:
            logger.error(f"最终汇总 LLM 调用失败: {e}")
            parts = []
            for r in self.all_results:
                if r.conclusion:
                    parts.append(f"- [{r.agent_id}] {r.conclusion}")
            return "最终汇总生成失败，以下为各智能体的原始结论：\n" + "\n".join(parts)

    def _batched_summary(self) -> str:
        """分批并行压缩总结"""
        agent_parts = self._build_evidence_parts()
        return self._recursive_batch_reduce(agent_parts, layer=1)

    def _recursive_batch_reduce(self, parts: list[str], layer: int) -> str:
        batch_size = self.summary_batch_size

        if len(parts) <= batch_size:
            return self._batch_final_merge(parts, layer)

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

        return self._recursive_batch_reduce(batch_summaries, layer + 1)

    def _batch_final_merge(self, summaries: list[str], layer: int) -> str:
        numbered = "\n\n".join(
            f"### 摘要 {i+1}\n{s}" for i, s in enumerate(summaries)
        )
        merge_prompt = BATCH_MERGE_PROMPT.format(
            question=self.question,
            batch_summaries=numbered,
        )

        logger.info(
            f"[BatchMerge] 第 {layer} 层（最终合并）："
            f"{len(summaries)} 条摘要，prompt 长度: {len(merge_prompt)} 字符"
        )
        logger.info(f"[BatchMerge] prompt 内容:\n{merge_prompt}")

        try:
            return chat(merge_prompt, vendor=self.vendor, model=self.model,
                        system=SUMMARY_SYSTEM_PROMPT)
        except Exception as e:
            logger.error(f"[BatchMerge] 最终合并失败: {e}")
            return "分批合并失败，以下为各摘要：\n" + numbered

    def _retrieval_final_summary(self, organized_parts: list[str]) -> str:
        knowledge_text = "\n\n".join(organized_parts)
        prompt = RETRIEVAL_SUMMARY_PROMPT.format(
            question=self.question,
            organized_knowledge=knowledge_text,
        )

        logger.info(f"[RetrievalSummary] prompt 长度: {len(prompt)} 字符")
        logger.info(f"[RetrievalSummary] prompt 内容:\n{prompt}")

        try:
            return chat(prompt, vendor=self.vendor, model=self.model,
                        system=SUMMARY_SYSTEM_PROMPT)
        except Exception as e:
            logger.error(f"[RetrievalSummary] 召回总结失败: {e}")
            return "召回总结生成失败，以下为召回的知识片段：\n" + knowledge_text

    def _retrieval_batched_summary(self, organized_parts: list[str]) -> str:
        return self._retrieval_recursive_batch_reduce(organized_parts, layer=1)

    def _retrieval_recursive_batch_reduce(self, parts: list[str], layer: int) -> str:
        batch_size = self.summary_batch_size

        if len(parts) <= batch_size:
            return self._retrieval_batch_final_merge(parts, layer)

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
        return self._retrieval_recursive_batch_reduce(batch_summaries, layer + 1)

    def _retrieval_batch_final_merge(self, summaries: list[str], layer: int) -> str:
        numbered = "\n\n".join(
            f"### 摘要 {i+1}\n{s}" for i, s in enumerate(summaries)
        )
        merge_prompt = RETRIEVAL_BATCH_MERGE_PROMPT.format(
            question=self.question,
            batch_summaries=numbered,
        )

        logger.info(
            f"[RetrievalBatchMerge] 第 {layer} 层（最终合并）："
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
