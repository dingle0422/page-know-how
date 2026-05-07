"""SkillResultRegistry — 线程安全的 skill 执行结果注册表

存储「skill 名 + 实际执行的 bash 命令 + 执行结果」的三元组，供推理流程的
double-check 阶段复用。
"""

import logging
import threading
from dataclasses import dataclass

from .runner import SkillExecutionResult

logger = logging.getLogger(__name__)


@dataclass
class SkillRecord:
    """单条 skill 调用记录"""
    skill_name: str
    command: str
    result: SkillExecutionResult


class SkillResultRegistry:
    """同一推理请求内的全局 skill 结果注册表，线程安全"""

    def __init__(self):
        self._lock = threading.Lock()
        self._records: list[SkillRecord] = []

    def add(self, skill_name: str, command: str, result: SkillExecutionResult) -> None:
        with self._lock:
            self._records.append(SkillRecord(
                skill_name=skill_name, command=command, result=result,
            ))
            logger.info(
                "[SkillRegistry] 写入: skill=%s, exit=%d, success=%s",
                skill_name, result.exit_code, result.success,
            )

    def get(self, skill_name: str) -> list[SkillRecord]:
        """取某个 skill 的所有调用记录（同一 skill 可能被多次调用）"""
        with self._lock:
            return [r for r in self._records if r.skill_name == skill_name]

    def get_all(self) -> list[SkillRecord]:
        with self._lock:
            return list(self._records)

    def has_any(self) -> bool:
        with self._lock:
            return len(self._records) > 0

    def has(self, skill_name: str) -> bool:
        with self._lock:
            return any(r.skill_name == skill_name for r in self._records)

    def format_context(self) -> str:
        """将所有 skill 结果格式化为可注入 LLM prompt 的事实依据文本"""
        records = self.get_all()
        if not records:
            return "（暂无事实依据）"

        lines: list[str] = []
        for i, rec in enumerate(records, 1):
            lines.append(f"【依据{i}】")
            if rec.result.success:
                lines.append((rec.result.stdout or "（该依据未返回有效内容）").strip())
            else:
                lines.append(f"该依据获取失败（exit={rec.result.exit_code}）。")
                if rec.result.stderr:
                    lines.append(rec.result.stderr.strip())
            lines.append("")

        return "\n".join(lines).rstrip()
