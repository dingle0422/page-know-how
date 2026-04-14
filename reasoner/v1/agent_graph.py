import os
import json
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
)

from reasoner.v0.agent_graph import (
    ExploredRegistry,
    PitfallsRegistry,
    KnowledgeFragment,
    RetrievalKnowledgeRegistry,
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
        self.registry = ExploredRegistry()
        self.pitfalls_registry = PitfallsRegistry()
        self.retrieval_registry = RetrievalKnowledgeRegistry() if retrieval_mode else None
        self.all_results: list[AgentResult] = []

    def run(self) -> dict:
        mode_label = "召回模式" if self.retrieval_mode else "标准模式"
        logger.info(f"AgentGraph 启动 [{mode_label}]: 问题={self.question[:50]}...")

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

        if self.retrieval_mode:
            answer = self._retrieval_pipeline()
        else:
            answer = self._standard_pipeline()

        if self.clean_answer:
            answer = self._clean_answer(answer)

        trace_log = self._build_trace_log()
        relevant_chapters = self._collect_relevant_chapters()

        return {
            "answer": answer,
            "trace_log": trace_log,
            "relevant_chapters": relevant_chapters,
        }

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

        sorted_fragments = sorted(fragments, key=lambda f: f.heading_path)

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
        """从所有智能体结果中提取去重的相关章节序号"""
        all_dirs = []
        for r in self.all_results:
            all_dirs.extend(r.relevant_dirs)

        if self.retrieval_mode:
            for frag in self.retrieval_registry.get_all():
                if frag.directory_path not in all_dirs:
                    all_dirs.append(frag.directory_path)

        chapters = set()
        for dir_path in all_dirs:
            rel = os.path.relpath(dir_path, self.knowledge_root)
            if rel == ".":
                continue
            parts = rel.replace("\\", "/").split("/")
            leaf = parts[-1]
            if "_" in leaf:
                chapter_num = leaf.split("_")[0]
                chapters.add(chapter_num)

        def _chapter_sort_key(ch: str) -> list:
            segments = ch.split(".")
            result = []
            for s in segments:
                try:
                    result.append(int(s))
                except ValueError:
                    result.append(float("inf"))
            return result

        return sorted(chapters, key=_chapter_sort_key)

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
