import os
import json
import asyncio
import threading
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm.client import chat
from reasoner._sort_utils import natural_dir_sort_key
from reasoner._registries import (
    ExploredRegistry,
    PitfallsRegistry,
    KnowledgeFragment,
    RetrievalKnowledgeRegistry,
)
from reasoner.v0.react_agent import ReactAgent, AgentResult, BacktrackIntent
from reasoner.v0.prompts import (
    INTENT_RESOLVE_PROMPT,
    SUMMARY_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
    CLEAN_ANSWER_PROMPT,
    BATCH_SUMMARY_PROMPT,
    BATCH_MERGE_PROMPT,
    RETRIEVAL_SUMMARY_PROMPT,
    RETRIEVAL_BATCH_SUMMARY_PROMPT,
    RETRIEVAL_BATCH_MERGE_PROMPT,
)

from skills import (
    SkillRunner,
    SkillResultRegistry,
    evaluate_and_run,
    check_and_enhance,
)

logger = logging.getLogger(__name__)


class AgentGraph:
    """管理子智能体的衍生、并行执行、意图解决和结果汇聚"""

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
        self.enable_skills = enable_skills
        self.registry = ExploredRegistry()
        self.pitfalls_registry = PitfallsRegistry()
        self.retrieval_registry = RetrievalKnowledgeRegistry() if retrieval_mode else None
        self.skill_registry: SkillResultRegistry | None = (
            SkillResultRegistry() if enable_skills else None
        )
        self.skill_runner: SkillRunner | None = SkillRunner() if enable_skills else None
        self.all_results: list[AgentResult] = []
        self.intent_resolve_results: list[dict] = []

    def _run_skill_evaluation(self) -> None:
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
        """
        从根目录开始执行完整推理流程。
        返回 {"answer": str, "trace_log": str}
        """
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

    def _standard_pipeline(self) -> str:
        """标准模式的后处理管线：意图解决 -> 汇总"""
        intents = self._find_backtrack_intents()
        if intents:
            logger.info(f"发现 {len(intents)} 个未完成的回溯意图，开始处理...")
            self.intent_resolve_results = self._resolve_intents(intents)

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

        return sorted(chapters, key=natural_dir_sort_key)

    def _flatten_results(self, result: AgentResult) -> list[AgentResult]:
        """递归展平所有子智能体的结果"""
        flat = [result]
        for child in result.child_results:
            flat.extend(self._flatten_results(child))
        return flat

    def _find_backtrack_intents(self) -> list[tuple[AgentResult, AgentResult]]:
        """找出所有携带 backtrack_intent 的结果，并与已探索方配对"""
        pairs = []
        for result_a in self.all_results:
            if result_a.backtrack_intent is None:
                continue
            target_dir = result_a.backtrack_intent.target_dir
            for result_b in self.all_results:
                if result_b.agent_id == result_a.agent_id:
                    continue
                if result_b.explored_dir == target_dir:
                    pairs.append((result_a, result_b))
                    break
                for step in result_b.trace:
                    if step.directory == target_dir:
                        pairs.append((result_a, result_b))
                        break
        return pairs

    def _resolve_intents(
        self, pairs: list[tuple[AgentResult, AgentResult]]
    ) -> list[dict]:
        """并行执行意图解决子智能体"""
        results = []

        def _resolve(pair):
            agent_a, agent_b = pair
            intent = agent_a.backtrack_intent
            prompt = INTENT_RESOLVE_PROMPT.format(
                question=self.question,
                agent_a_dir=agent_a.explored_dir,
                agent_a_evidence="\n".join(f"- {e}" for e in agent_a.evidence) or "（无）",
                agent_a_conclusion=agent_a.conclusion,
                intent_target_dir=intent.target_dir,
                intent_reason=intent.reason,
                intent_seeking=intent.seeking,
                intent_current_summary=intent.current_summary or "（无摘要）",
                agent_b_dir=agent_b.explored_dir,
                agent_b_evidence="\n".join(f"- {e}" for e in agent_b.evidence) or "（无）",
                agent_b_conclusion=agent_b.conclusion,
            )
            try:
                response = chat(prompt, vendor=self.vendor, model=self.model)
                cleaned = response.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                if cleaned.startswith("```"):
                    cleaned = cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                resolve_result = json.loads(cleaned.strip())
                resolve_result["agent_a_id"] = agent_a.agent_id
                resolve_result["agent_b_id"] = agent_b.agent_id
                return resolve_result
            except Exception as e:
                logger.error(f"IntentResolver 失败 ({agent_a.agent_id} <-> {agent_b.agent_id}): {e}")
                return {
                    "agent_a_id": agent_a.agent_id,
                    "agent_b_id": agent_b.agent_id,
                    "intent_satisfied": False,
                    "reasoning": f"解析失败: {e}",
                    "conclusion": "意图解决失败",
                    "combined_evidence": [],
                }

        with ThreadPoolExecutor(max_workers=min(len(pairs), 5)) as executor:
            futures = [executor.submit(_resolve, p) for p in pairs]
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"IntentResolver 执行异常: {e}")

        return results

    def _build_evidence_parts(self) -> tuple[list[str], list[str]]:
        """构建子智能体结果和意图解决结果的文本片段列表"""
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

        intent_parts = []
        for ir in self.intent_resolve_results:
            combined = "\n".join(f"  - {e}" for e in ir.get("combined_evidence", [])) or "  （无）"
            intent_parts.append(
                f"### IntentResolver | {ir['agent_a_id']} ↔ {ir['agent_b_id']}\n"
                f"- 意图是否满足: {ir.get('intent_satisfied', False)}\n"
                f"- 推理: {ir.get('reasoning', '')}\n"
                f"- 结论: {ir.get('conclusion', '')}\n"
                f"- 结合证据:\n{combined}"
            )
        return agent_parts, intent_parts

    def _final_summary(self) -> str:
        """调用 LLM 做最终汇总（单次全量）"""
        agent_parts, intent_parts = self._build_evidence_parts()

        prompt = SUMMARY_PROMPT.format(
            question=self.question,
            agent_results="\n\n".join(agent_parts) if agent_parts else "（无）",
            intent_results="\n\n".join(intent_parts) if intent_parts else "（无意图关联）",
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
        """分批并行压缩总结：递归分层，直到收敛到 batch_size 条以内再做最终合并"""
        agent_parts, intent_parts = self._build_evidence_parts()
        all_parts = agent_parts + intent_parts
        return self._recursive_batch_reduce(all_parts, layer=1)

    def _recursive_batch_reduce(self, parts: list[str], layer: int) -> str:
        """递归分批压缩：每层并行压缩，直到摘要数 ≤ batch_size 时做最终合并"""
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
        """分批压缩的最终合并层"""
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
        """召回模式：单次全量总结"""
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
        """召回模式：分批并行压缩总结"""
        return self._retrieval_recursive_batch_reduce(organized_parts, layer=1)

    def _retrieval_recursive_batch_reduce(self, parts: list[str], layer: int) -> str:
        """召回模式专用递归分批压缩"""
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
        """召回模式分批压缩的最终合并层"""
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
        """调用 LLM 对 summary 结果做信息清洗，转为面向用户的客服口吻回答"""
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
            if r.backtrack_intent:
                intent = r.backtrack_intent
                lines.append(f"  回溯意图: 目标={intent.target_dir}, 原因={intent.reason}, 期望={intent.seeking}")
                if intent.current_summary:
                    lines.append(f"  回溯携带摘要: {intent.current_summary}")
            lines.append("")

        for ir in self.intent_resolve_results:
            lines.append(f"--- IntentResolver | 关联: {ir['agent_a_id']} ↔ {ir['agent_b_id']} ---")
            lines.append(f"  意图是否满足: {ir.get('intent_satisfied', False)}")
            lines.append(f"  推理: {ir.get('reasoning', '')}")
            lines.append(f"  补全结论: {ir.get('conclusion', '')}")
            if ir.get("combined_evidence"):
                lines.append(f"  结合证据: {'; '.join(ir['combined_evidence'])}")
            lines.append("")

        return "\n".join(lines)
