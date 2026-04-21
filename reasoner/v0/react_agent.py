import os
import json
import uuid
import logging
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm.client import chat
from reasoner._sort_utils import natural_dir_sort_key
from reasoner._registries import KnowledgeFragment
from reasoner.v0.prompts import (
    DISCLOSURE_PROMPT,
    CONTENT_ASSESS_PROMPT,
    FORCE_SUMMARY_PROMPT,
    ROOT_DISCLOSURE_PROMPT,
    PATH_CORRECTION_PROMPT,
    RETRIEVAL_DISCLOSURE_PROMPT,
    RELEVANCE_ASSESS_PROMPT,
    RETRIEVAL_FORCE_SUMMARY_PROMPT,
)

logger = logging.getLogger(__name__)


@dataclass
class BacktrackIntent:
    target_dir: str
    reason: str
    seeking: str
    current_summary: str = ""


@dataclass
class TraceStep:
    directory: str
    action: str
    observation: str


@dataclass
class AgentResult:
    agent_id: str
    explored_dir: str
    evidence: list[str] = field(default_factory=list)
    relevant_dirs: list[str] = field(default_factory=list)
    reasoning_chain: str = ""
    conclusion: str = ""
    trace: list[TraceStep] = field(default_factory=list)
    backtrack_intent: BacktrackIntent | None = None
    child_results: list['AgentResult'] = field(default_factory=list)
    pitfalls: list[str] = field(default_factory=list)


class ReactAgent:
    def __init__(
        self,
        question: str,
        knowledge_root: str,
        current_dir: str,
        upstream_path: list[str],
        parent_summary: str = "",
        registry=None,
        pitfalls_registry=None,
        max_rounds: int = 5,
        vendor: str = "qwen3.5-122b-a10b",
        model: str = "Qwen3.5-122B-A10B",
        retrieval_mode: bool = False,
        retrieval_registry=None,
        subtree_root: str = "",
    ):
        self.agent_id = f"Agent-{uuid.uuid4().hex[:6]}"
        self.question = question
        self.knowledge_root = knowledge_root
        self.current_dir = current_dir
        self.upstream_path = upstream_path
        self.parent_summary = parent_summary
        self.registry = registry
        self.pitfalls_registry = pitfalls_registry
        self.max_rounds = max_rounds
        self.vendor = vendor
        self.model = model
        self.retrieval_mode = retrieval_mode
        self.retrieval_registry = retrieval_registry
        self.subtree_root = subtree_root or knowledge_root

        self.evidence: list[str] = []
        self.relevant_dirs: list[str] = []
        self.trace: list[TraceStep] = []
        self.reasoning_parts: list[str] = []
        self.content_conclusion: str = ""
        self.local_pitfalls: list[str] = []

    def run(self) -> AgentResult:
        """执行 ReAct 循环"""
        logger.info(f"[{self.agent_id}] 开始探索: {self.current_dir}")

        if self.registry:
            self.registry.try_claim(self.current_dir, self.agent_id)

        current_dir = self.current_dir
        all_child_results: list[AgentResult] = []

        for round_num in range(1, self.max_rounds + 1):
            logger.info(f"[{self.agent_id}] 轮次 {round_num}/{self.max_rounds}")

            knowledge_content = self._read_knowledge_md(current_dir)
            available_subdirs = self._list_subdirs(current_dir)

            self.trace.append(TraceStep(
                directory=current_dir,
                action="READ",
                observation=f"读取 knowledge.md，发现 {len(available_subdirs)} 个子目录: {', '.join(available_subdirs) if available_subdirs else '无'}"
            ))

            if round_num >= self.max_rounds:
                result = self._force_summary(current_dir)
                result.child_results = all_child_results
                return result

            # ===== 根节点：单次 LLM 调用，全面探索所有一级子目录 =====
            if self._is_root_level(current_dir) and available_subdirs:
                decision = self._ask_root_llm(knowledge_content, available_subdirs)
                if decision is None:
                    decision = {
                        "action": "EXPLORE",
                        "targets": available_subdirs,
                        "current_summary": "",
                    }
                    logger.warning(f"[{self.agent_id}] 根节点 LLM 解析失败，回退为全面探索所有一级子目录")

                current_summary = decision.get("current_summary", "") or decision.get("summary", "")
                llm_targets = decision.get("targets", [])
                missing = [d for d in available_subdirs if d not in llm_targets]
                if missing:
                    logger.info(
                        f"[{self.agent_id}] 根节点 LLM 遗漏了 {len(missing)} 个一级子目录，自动补全: {missing}"
                    )
                all_targets = available_subdirs

                self.reasoning_parts.append(
                    f"[ROOT_EXPLORE] 根节点全面探索所有 {len(all_targets)} 个一级子目录"
                )
                self.trace.append(TraceStep(
                    directory=current_dir,
                    action="ROOT_EXPLORE",
                    observation=f"根节点全面探索所有一级子目录: {', '.join(all_targets)}"
                ))

                valid_targets = []
                for subdir in all_targets:
                    target_path = os.path.join(current_dir, subdir)
                    if os.path.isdir(target_path):
                        if self.registry:
                            self.registry.try_claim(target_path, self.agent_id)
                        valid_targets.append(target_path)

                if valid_targets:
                    child_results = self._spawn_child_agents(
                        valid_targets, current_summary, current_dir
                    )
                    all_child_results.extend(child_results)
                    for cr in child_results:
                        self.evidence.extend(cr.evidence)
                        self.relevant_dirs.extend(cr.relevant_dirs)
                    return self._build_result(
                        conclusion=f"根节点已衍生 {len(child_results)} 个子智能体全面探索所有一级目录",
                        child_results=all_child_results,
                    )
                continue

            # ===== 非根节点：并行发起「内容评估」和「导航决策」两个 LLM 请求 =====
            assessment, navigation = self._parallel_decide(
                knowledge_content, available_subdirs, current_dir
            )

            if assessment:
                self._collect_pitfalls(assessment, directory=current_dir)

            # ------ 处理内容评估结果 ------
            if self.retrieval_mode:
                if assessment:
                    is_relevant = assessment.get("is_relevant", False)
                    assess_reason = assessment.get("reason", "")

                    if is_relevant:
                        self.relevant_dirs.append(current_dir)
                        if self.retrieval_registry:
                            heading_path = self._compute_heading_path(current_dir)
                            fragment = KnowledgeFragment(
                                content=knowledge_content,
                                heading_path=heading_path,
                                directory_path=current_dir,
                            )
                            added = self.retrieval_registry.add(fragment)
                        logger.info(
                            f"[{self.agent_id}] 相关性评估: 相关 "
                            f"(层级={'> '.join(heading_path)}, 新增={added})"
                        )
                    else:
                        logger.info(f"[{self.agent_id}] 相关性评估: 无关 - {assess_reason}")

                    self.reasoning_parts.append(
                        f"[RELEVANCE_ASSESS] is_relevant={is_relevant}, reason={assess_reason}"
                    )
                    self.trace.append(TraceStep(
                        directory=current_dir,
                        action="RELEVANCE_ASSESS",
                        observation=f"相关性评估: {'相关' if is_relevant else '无关'}, 理由: {assess_reason}"
                    ))
                else:
                    logger.warning(f"[{self.agent_id}] 相关性评估 LLM 解析失败，跳过本层级评估")
            else:
                if assessment:
                    has_relevant = assessment.get("has_relevant_content", False)

                    if has_relevant:
                        self.evidence.extend(assessment.get("evidence", []))
                        self.relevant_dirs.append(current_dir)
                        conclusion = assessment.get("conclusion", "")
                        summary = assessment.get("summary", "")
                        if conclusion:
                            self.content_conclusion = conclusion
                        self.reasoning_parts.append(f"[CONTENT_ASSESS] 发现相关内容: {summary}")
                        self.trace.append(TraceStep(
                            directory=current_dir,
                            action="CONTENT_ASSESS",
                            observation=f"发现相关内容，结论: {conclusion}"
                        ))
                    else:
                        self.reasoning_parts.append("[CONTENT_ASSESS] 当前层级未发现直接相关内容")
                        self.trace.append(TraceStep(
                            directory=current_dir,
                            action="CONTENT_ASSESS",
                            observation="当前层级未发现与问题直接相关的内容"
                        ))
                else:
                    logger.warning(f"[{self.agent_id}] 内容评估 LLM 解析失败，跳过本层级评估")

            # ------ 处理导航决策结果 ------
            if navigation is None:
                logger.error(f"[{self.agent_id}] 导航决策 LLM 解析失败，返回已有证据")
                return self._build_result(
                    conclusion=self.content_conclusion or "导航决策解析失败，返回已有证据",
                    child_results=all_child_results
                )

            nav_action = navigation.get("action", "").upper()

            if nav_action == "STOP":
                stop_reason = navigation.get("reason", "无更多可探索的子目录")
                self.reasoning_parts.append(f"[STOP] {stop_reason}")
                self.trace.append(TraceStep(
                    directory=current_dir,
                    action="STOP",
                    observation=f"终止探索: {stop_reason}"
                ))
                conclusion_text = self.content_conclusion or "探索完成"
                return self._build_result(
                    conclusion=conclusion_text,
                    child_results=all_child_results
                )

            elif nav_action == "EXPLORE":
                targets = navigation.get("targets", [])
                reason = navigation.get("reason", "")
                current_summary = navigation.get("current_summary", "")

                targets = self._correct_targets(targets, available_subdirs, current_dir)

                self.reasoning_parts.append(f"[EXPLORE] {reason} -> {targets}")
                self.trace.append(TraceStep(
                    directory=current_dir,
                    action="EXPLORE",
                    observation=f"选择子目录: {', '.join(targets)}，原因: {reason}"
                ))

                valid_targets = []
                for t in targets:
                    target_path = os.path.join(current_dir, t)
                    if not os.path.isdir(target_path):
                        logger.warning(f"[{self.agent_id}] 子目录不存在: {target_path}")
                        continue
                    if self.registry and not self.registry.try_claim(target_path, self.agent_id):
                        logger.info(f"[{self.agent_id}] 子目录已被认领，跳过: {target_path}")
                        continue
                    valid_targets.append(target_path)

                if not valid_targets:
                    self.reasoning_parts.append("[EXPLORE] 所有目标子目录均已被其他智能体认领或不存在")
                    continue

                if len(valid_targets) == 1:
                    current_dir = valid_targets[0]
                    self.upstream_path = self.upstream_path + [current_dir]
                    self.parent_summary = current_summary
                    continue

                child_results = self._spawn_child_agents(
                    valid_targets, current_summary, current_dir
                )
                all_child_results.extend(child_results)

                for cr in child_results:
                    self.evidence.extend(cr.evidence)
                    self.relevant_dirs.extend(cr.relevant_dirs)

                conclusion_text = self.content_conclusion or f"已衍生 {len(child_results)} 个子智能体探索"
                return self._build_result(
                    conclusion=conclusion_text,
                    child_results=all_child_results
                )

            elif nav_action == "BACKTRACK":
                target_dir = navigation.get("target_dir", "")
                reason = navigation.get("reason", "")
                seeking = navigation.get("seeking", "")
                backtrack_summary = navigation.get("current_summary", "")
                self.reasoning_parts.append(f"[BACKTRACK] -> {target_dir}: {reason}")

                if not os.path.isdir(target_dir):
                    target_dir = self._correct_backtrack_target(target_dir)

                if not self._is_within_subtree(target_dir):
                    logger.warning(
                        f"[{self.agent_id}] 回溯目标超出子树边界，忽略: "
                        f"'{target_dir}' 不在 '{self.subtree_root}' 内"
                    )
                    self.trace.append(TraceStep(
                        directory=current_dir,
                        action="BACKTRACK_BLOCKED",
                        observation=f"目标 {target_dir} 超出子树边界 {self.subtree_root}，已忽略"
                    ))
                    continue

                if self.registry and self.registry.is_explored(target_dir):
                    logger.info(f"[{self.agent_id}] 回溯目标已被探索: {target_dir}，终止并记录意图")
                    self.trace.append(TraceStep(
                        directory=current_dir,
                        action="BACKTRACK",
                        observation=f"目标 {target_dir} 已被其他智能体探索，终止并记录回溯意图"
                    ))
                    return self._build_result(
                        conclusion=self.content_conclusion or f"回溯目标 {target_dir} 已被探索，携带意图返回",
                        backtrack_intent=BacktrackIntent(
                            target_dir=target_dir,
                            reason=reason,
                            seeking=seeking,
                            current_summary=backtrack_summary,
                        ),
                        child_results=all_child_results,
                    )

                if self.registry:
                    self.registry.try_claim(target_dir, self.agent_id)

                self.trace.append(TraceStep(
                    directory=current_dir,
                    action="BACKTRACK",
                    observation=f"回溯到: {target_dir}，原因: {reason}"
                ))

                new_upstream = self._truncate_upstream(target_dir)
                self.upstream_path = new_upstream
                current_dir = target_dir
                continue

            else:
                logger.warning(f"[{self.agent_id}] 未知导航动作: {nav_action}")
                continue

        return self._build_result(
            conclusion="达到最大轮次，返回已有证据",
            child_results=all_child_results
        )

    def _collect_pitfalls(self, decision: dict, directory: str = "") -> None:
        """从 LLM 决策中提取 pitfalls，注册到全局缓存并本地收集"""
        pitfalls = decision.get("pitfalls", [])
        if not pitfalls:
            return
        if isinstance(pitfalls, str):
            pitfalls = [pitfalls]
        for p in pitfalls:
            if p and p not in self.local_pitfalls:
                self.local_pitfalls.append(p)
        if self.pitfalls_registry:
            self.pitfalls_registry.add(pitfalls, directory=directory)
        logger.info(f"[{self.agent_id}] 提取 {len(pitfalls)} 条易错点")

    def _is_root_level(self, current_dir: str) -> bool:
        """判断当前目录是否为知识目录根节点"""
        return os.path.normpath(current_dir) == os.path.normpath(self.knowledge_root)

    def _is_within_subtree(self, path: str) -> bool:
        """判断路径是否在当前智能体的子树边界内"""
        path_norm = os.path.normpath(path)
        subtree_norm = os.path.normpath(self.subtree_root)
        return path_norm == subtree_norm or path_norm.startswith(subtree_norm + os.sep)

    def _read_knowledge_md(self, directory: str) -> str:
        km_path = os.path.join(directory, "knowledge.md")
        if os.path.exists(km_path):
            with open(km_path, "r", encoding="utf-8") as f:
                return f.read()
        return "（当前目录无 knowledge.md 文件）"

    _IGNORED_DIRS = frozenset({
        "__pycache__", ".ipynb_checkpoints", ".git", ".svn",
        "node_modules", ".venv", "venv", ".tox",
    })

    def _list_subdirs(self, directory: str) -> list[str]:
        if not os.path.isdir(directory):
            return []
        candidates = [
            d for d in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, d))
            and not d.startswith(".")
            and d not in self._IGNORED_DIRS
        ]
        return sorted(candidates, key=lambda d: (natural_dir_sort_key(d), d))

    @staticmethod
    def _strip_section(content: str, heading: str) -> str:
        """去除 knowledge.md 中指定 ## 段落（从该标题到下一个 ## 标题之间的内容）"""
        idx = content.find(heading)
        if idx == -1:
            return content
        next_heading_idx = content.find("\n## ", idx + len(heading))
        if next_heading_idx != -1:
            return content[:idx].rstrip() + "\n\n" + content[next_heading_idx + 1:]
        return content[:idx].rstrip()

    @staticmethod
    def _strip_subdirs_overview(content: str) -> str:
        """去除 knowledge.md 中的 '## 子目录概览'、'## 项目条款名称'、'## 全称' 段落，减少冗余信息"""
        for heading in ("## 项目条款名称", "## 全称"):
            content = ReactAgent._strip_section(content, heading)
        marker = "## 子目录概览"
        idx = content.find(marker)
        if idx != -1:
            content = content[:idx].rstrip()
        return content

    def _build_upstream_tree(self, current_dir: str) -> str:
        """构建上游路径树，每层附带未探索的平级目录；无可探索平级的上级层级整体移除。
        平级目录限定在 subtree_root 内，防止跨知识库串台。
        """
        lines = []
        for dir_path in self.upstream_path:
            name = os.path.basename(dir_path)
            is_current = os.path.normpath(dir_path) == os.path.normpath(current_dir)

            parent = os.path.dirname(dir_path)
            unexplored = []
            if parent and os.path.isdir(parent):
                siblings = self._list_subdirs(parent)
                unexplored = [
                    s for s in siblings
                    if s != name
                    and self._is_within_subtree(os.path.join(parent, s))
                    and not (
                        self.registry and self.registry.is_explored(
                            os.path.join(parent, s)
                        )
                    )
                ]

            if not is_current and not unexplored:
                continue

            indent = "  " * len(lines)
            label = f"{indent}→ {name}" + (" ← 当前位置" if is_current else "")
            lines.append(label)

            if unexplored:
                lines.append(f"{indent}  可回溯探索的平级目录: {', '.join(unexplored)}")

        return "\n".join(lines) if lines else "（所有上游路径均已探索）"

    def _filter_explored_subdirs(self, available_subdirs: list[str], current_dir: str) -> list[str]:
        """过滤掉已被探索的子目录，防止 LLM 重复选择"""
        if not self.registry:
            return available_subdirs
        return [
            d for d in available_subdirs
            if not self.registry.is_explored(os.path.join(current_dir, d))
        ]

    def _correct_backtrack_target(self, target_dir: str) -> str:
        """当 LLM 返回的 BACKTRACK target_dir 不是有效目录时，尝试修正。
        优先按名称在各上游层级的平级目录中精确拼接，再回退到模糊匹配。
        所有候选路径必须在 subtree_root 内，防止跨知识库串台。
        """
        for up_dir in self.upstream_path:
            parent = os.path.dirname(up_dir)
            if parent and os.path.isdir(parent):
                candidate = os.path.join(parent, target_dir)
                if os.path.isdir(candidate) and self._is_within_subtree(candidate):
                    logger.info(f"[{self.agent_id}] 回溯目标修正(名称拼接): '{target_dir}' -> '{candidate}'")
                    return candidate

        for up_dir in reversed(self.upstream_path):
            if not self._is_within_subtree(up_dir):
                continue
            if target_dir in up_dir or os.path.basename(up_dir).startswith(
                target_dir.split('_')[0] if '_' in target_dir else target_dir
            ):
                logger.info(f"[{self.agent_id}] 回溯目标修正(上游匹配): '{target_dir}' -> '{up_dir}'")
                return up_dir

        for up_dir in self.upstream_path:
            parent = os.path.dirname(up_dir)
            if parent and os.path.isdir(parent):
                for sibling in self._list_subdirs(parent):
                    sibling_path = os.path.join(parent, sibling)
                    if not self._is_within_subtree(sibling_path):
                        continue
                    if target_dir in sibling or sibling.startswith(
                        target_dir.split('_')[0] if '_' in target_dir else target_dir
                    ):
                        logger.info(f"[{self.agent_id}] 回溯目标修正(平级模糊): '{target_dir}' -> '{sibling_path}'")
                        return sibling_path

        logger.warning(f"[{self.agent_id}] 回溯目标无法修正: '{target_dir}'，保持原值")
        return target_dir

    def _truncate_upstream(self, target_dir: str) -> list[str]:
        """根据 BACKTRACK target_dir 截断 upstream_path。
        支持两种情况：target 是上游路径自身，或 target 是某上游路径的平级目录。
        """
        target_norm = os.path.normpath(target_dir)
        target_parent_norm = os.path.normpath(os.path.dirname(target_dir))

        new_upstream = []
        for p in self.upstream_path:
            p_norm = os.path.normpath(p)
            new_upstream.append(p)
            if p_norm == target_norm:
                return new_upstream
            if p_norm == target_parent_norm:
                new_upstream.append(target_dir)
                return new_upstream

        new_upstream.append(target_dir)
        return new_upstream

    def _correct_targets(
        self, targets: list[str], available_subdirs: list[str], current_dir: str
    ) -> list[str]:
        """校正 LLM 返回的 EXPLORE targets，确保它们在当前可用子目录中"""
        if not targets or not available_subdirs:
            return targets

        corrected = []
        need_llm_correction: list[str] = []

        for t in targets:
            if t in available_subdirs:
                corrected.append(t)
                continue

            match = self._fuzzy_match_subdir(t, available_subdirs)
            if match:
                logger.info(f"[{self.agent_id}] 子目录名模糊修正: '{t}' -> '{match}'")
                if match not in corrected:
                    corrected.append(match)
            else:
                need_llm_correction.append(t)

        if need_llm_correction:
            already_matched = set(corrected)
            remaining_subdirs = [s for s in available_subdirs if s not in already_matched]

            if remaining_subdirs:
                llm_corrections = self._llm_correct_paths(
                    need_llm_correction, remaining_subdirs, current_dir
                )
                for invalid_name, fixed_name in llm_corrections.items():
                    if fixed_name and fixed_name in available_subdirs and fixed_name not in corrected:
                        logger.info(f"[{self.agent_id}] 子目录名LLM修正: '{invalid_name}' -> '{fixed_name}'")
                        corrected.append(fixed_name)
                    else:
                        logger.warning(
                            f"[{self.agent_id}] 子目录 '{invalid_name}' 不在当前目录 {os.path.basename(current_dir)} 的可用子目录中，已跳过"
                        )
            else:
                for inv in need_llm_correction:
                    logger.warning(
                        f"[{self.agent_id}] 子目录 '{inv}' 不在当前目录 {os.path.basename(current_dir)} 的可用子目录中，已跳过"
                    )

        return corrected

    @staticmethod
    def _fuzzy_match_subdir(target: str, available: list[str]) -> str | None:
        """基于编号前缀和子串进行模糊匹配"""
        target_stripped = target.strip()

        if '_' in target_stripped:
            prefix = target_stripped.split('_')[0]
            prefix_matches = [s for s in available if s.split('_')[0] == prefix]
            if len(prefix_matches) == 1:
                return prefix_matches[0]

        for s in available:
            if target_stripped in s or s in target_stripped:
                return s

        target_lower = target_stripped.lower()
        for s in available:
            if target_lower in s.lower() or s.lower() in target_lower:
                return s

        return None

    def _llm_correct_paths(
        self, invalid_targets: list[str], available_subdirs: list[str], current_dir: str
    ) -> dict[str, str | None]:
        """调用 LLM 根据实际目录列表修正无效的子目录名"""
        invalid_str = "\n".join(f"- {t}" for t in invalid_targets)
        avail_str = "\n".join(f"- {s}" for s in available_subdirs)

        prompt = PATH_CORRECTION_PROMPT.format(
            invalid_targets=invalid_str,
            available_subdirs=avail_str,
            current_dir=current_dir,
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
            result = json.loads(cleaned.strip())
            corrections = result.get("corrections", {})
            return {k: v for k, v in corrections.items() if v is None or v in available_subdirs}
        except Exception as e:
            logger.error(f"[{self.agent_id}] LLM 路径修正失败: {e}")
            return {}

    def _call_llm_json(self, prompt: str) -> dict | None:
        """调用 LLM 并解析 JSON 响应"""
        try:
            response = chat(prompt, vendor=self.vendor, model=self.model)
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            return json.loads(cleaned.strip())
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"[{self.agent_id}] LLM 响应解析失败: {e}")
            return None

    def _ask_root_llm(self, knowledge_content: str, available_subdirs: list[str]) -> dict | None:
        """根节点专用：单次 LLM 调用获取全面探索决策"""
        subdirs_str = "\n".join(f"- {d}" for d in available_subdirs) if available_subdirs else "（无子目录）"
        stripped_content = self._strip_subdirs_overview(knowledge_content)
        prompt = ROOT_DISCLOSURE_PROMPT.format(
            question=self.question,
            knowledge_content=stripped_content,
            available_subdirs=subdirs_str,
        )
        return self._call_llm_json(prompt)

    def _assess_content(self, knowledge_content: str, current_dir: str) -> dict | None:
        """标准模式：评估当前层级知识是否包含与用户问题相关的证据"""
        stripped_content = self._strip_subdirs_overview(knowledge_content)
        prompt = CONTENT_ASSESS_PROMPT.format(
            question=self.question,
            parent_summary=self.parent_summary or "（无上游摘要）",
            knowledge_content=stripped_content,
        )
        return self._call_llm_json(prompt)

    def _assess_relevance(self, knowledge_content: str, current_dir: str) -> dict | None:
        """召回模式：评估当前层级知识是否与用户问题相关"""
        stripped_content = self._strip_subdirs_overview(knowledge_content)
        prompt = RELEVANCE_ASSESS_PROMPT.format(
            question=self.question,
            parent_summary=self.parent_summary or "（无上游摘要）",
            knowledge_content=stripped_content,
        )
        return self._call_llm_json(prompt)

    def _ask_navigation(self, knowledge_content: str, available_subdirs: list[str], current_dir: str) -> dict | None:
        """非根节点：决定探索方向（EXPLORE / BACKTRACK / STOP）"""
        upstream_tree = self._build_upstream_tree(current_dir)
        filtered_subdirs = self._filter_explored_subdirs(available_subdirs, current_dir)
        subdirs_str = "\n".join(f"- {d}" for d in filtered_subdirs) if filtered_subdirs else "（无子目录）"
        stripped_content = self._strip_subdirs_overview(knowledge_content)
        if self.retrieval_mode:
            prompt = RETRIEVAL_DISCLOSURE_PROMPT.format(
                question=self.question,
                upstream_path=upstream_tree,
                parent_summary=self.parent_summary or "（无上游摘要）",
                knowledge_content=stripped_content,
                available_subdirs=subdirs_str,
            )
        else:
            prompt = DISCLOSURE_PROMPT.format(
                question=self.question,
                upstream_path=upstream_tree,
                parent_summary=self.parent_summary or "（无上游摘要）",
                knowledge_content=stripped_content,
                available_subdirs=subdirs_str,
            )
        return self._call_llm_json(prompt)

    def _parallel_decide(
        self, knowledge_content: str, available_subdirs: list[str], current_dir: str
    ) -> tuple[dict | None, dict | None]:
        """非根节点：并行执行内容评估和导航决策两个独立 LLM 请求"""
        with ThreadPoolExecutor(max_workers=2) as executor:
            if self.retrieval_mode:
                assess_future = executor.submit(
                    self._assess_relevance, knowledge_content, current_dir
                )
            else:
                assess_future = executor.submit(
                    self._assess_content, knowledge_content, current_dir
                )
            nav_future = executor.submit(
                self._ask_navigation, knowledge_content, available_subdirs, current_dir
            )

            try:
                assessment = assess_future.result()
            except Exception as e:
                logger.error(f"[{self.agent_id}] 内容评估 LLM 调用异常: {e}")
                assessment = None

            try:
                navigation = nav_future.result()
            except Exception as e:
                logger.error(f"[{self.agent_id}] 导航决策 LLM 调用异常: {e}")
                navigation = None

        return assessment, navigation

    def _force_summary(self, current_dir: str) -> AgentResult:
        """达到轮次上限，强制总结（标准模式）或强制相关性判定（召回模式）"""
        if self.retrieval_mode:
            return self._force_retrieval_judge(current_dir)

        evidence_str = "\n".join(f"- {e}" for e in self.evidence) if self.evidence else "（无证据）"
        path_str = " -> ".join(self.upstream_path) if self.upstream_path else current_dir

        prompt = FORCE_SUMMARY_PROMPT.format(
            question=self.question,
            evidence=evidence_str,
            exploration_path=path_str,
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
            decision = json.loads(cleaned.strip())
            self.evidence.extend(decision.get("evidence", []))
            conclusion = decision.get("conclusion", "达到轮次上限的总结")
            self.trace.append(TraceStep(
                directory=current_dir,
                action="FORCE_SUMMARY",
                observation=f"强制总结: {conclusion}"
            ))
        except Exception as e:
            logger.error(f"[{self.agent_id}] 强制总结解析失败: {e}")
            conclusion = "达到轮次上限，无法完成总结"
            self.trace.append(TraceStep(
                directory=current_dir,
                action="FORCE_SUMMARY",
                observation=f"强制总结失败: {e}"
            ))

        return self._build_result(conclusion=conclusion)

    def _force_retrieval_judge(self, current_dir: str) -> AgentResult:
        """召回模式下轮次用尽时的强制相关性判定"""
        knowledge_content = self._read_knowledge_md(current_dir)
        stripped_content = self._strip_subdirs_overview(knowledge_content)
        path_str = " -> ".join(self.upstream_path) if self.upstream_path else current_dir

        prompt = RETRIEVAL_FORCE_SUMMARY_PROMPT.format(
            question=self.question,
            knowledge_content=stripped_content,
            exploration_path=path_str,
        )

        is_relevant = False
        try:
            response = chat(prompt, vendor=self.vendor, model=self.model)
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            decision = json.loads(cleaned.strip())
            is_relevant = decision.get("is_relevant", False)
            reason = decision.get("reason", "")

            if is_relevant:
                self.relevant_dirs.append(current_dir)
                if self.retrieval_registry:
                    heading_path = self._compute_heading_path(current_dir)
                    fragment = KnowledgeFragment(
                        content=knowledge_content,
                        heading_path=heading_path,
                        directory_path=current_dir,
                    )
                    self.retrieval_registry.add(fragment)

            self.trace.append(TraceStep(
                directory=current_dir,
                action="FORCE_RELEVANCE_JUDGE",
                observation=f"强制相关性判定: {'相关' if is_relevant else '无关'}, 理由: {reason}"
            ))
        except Exception as e:
            logger.error(f"[{self.agent_id}] 强制相关性判定解析失败: {e}")
            self.trace.append(TraceStep(
                directory=current_dir,
                action="FORCE_RELEVANCE_JUDGE",
                observation=f"强制相关性判定失败: {e}"
            ))

        return self._build_result(
            conclusion=f"轮次用尽，相关性判定: {'相关' if is_relevant else '无关'}"
        )

    def _spawn_child_agents(
        self, target_paths: list[str], current_summary: str, parent_dir: str
    ) -> list[AgentResult]:
        """为多个目标子目录并行创建子智能体"""
        results = []

        def _run_child(target_path):
            child = ReactAgent(
                question=self.question,
                knowledge_root=self.knowledge_root,
                current_dir=target_path,
                upstream_path=self.upstream_path + [target_path],
                parent_summary=current_summary,
                registry=self.registry,
                pitfalls_registry=self.pitfalls_registry,
                max_rounds=self.max_rounds,
                vendor=self.vendor,
                model=self.model,
                retrieval_mode=self.retrieval_mode,
                retrieval_registry=self.retrieval_registry,
                subtree_root=self.subtree_root,
            )
            return child.run()

        with ThreadPoolExecutor(max_workers=min(len(target_paths), 5)) as executor:
            futures = {
                executor.submit(_run_child, tp): tp
                for tp in target_paths
            }
            for future in as_completed(futures):
                tp = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"[{self.agent_id}] 子智能体执行失败 ({tp}): {e}")
                    results.append(AgentResult(
                        agent_id=f"Agent-ERROR-{uuid.uuid4().hex[:4]}",
                        explored_dir=tp,
                        conclusion=f"执行失败: {e}",
                    ))

        return results

    def _compute_heading_path(self, directory: str) -> list[str]:
        """从目录路径计算层级标题路径（相对于 knowledge_root）"""
        rel = os.path.relpath(directory, self.knowledge_root)
        if rel == ".":
            return []
        return rel.replace("\\", "/").split("/")

    def _build_result(
        self,
        conclusion: str = "",
        backtrack_intent: BacktrackIntent | None = None,
        child_results: list[AgentResult] | None = None,
    ) -> AgentResult:
        return AgentResult(
            agent_id=self.agent_id,
            explored_dir=self.current_dir,
            evidence=list(self.evidence),
            relevant_dirs=list(self.relevant_dirs),
            reasoning_chain="\n".join(self.reasoning_parts),
            conclusion=conclusion,
            trace=list(self.trace),
            backtrack_intent=backtrack_intent,
            child_results=child_results or [],
            pitfalls=list(self.local_pitfalls),
        )
