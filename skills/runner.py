"""Skill Bash 沙箱执行模块

每次 execute() = 一个独立 bash subprocess + 独立临时 cwd + 受控 env，
专为「大模型直接产出 bash 命令」的调用范式设计。

并发隔离保证（避免不同请求结果互相混淆）：
- cwd 隔离 :  每次调用单独 mkdtemp，结束后 rmtree，文件副作用互不可见
- IO 隔离  :  stdout / stderr 各自独占 PIPE，输出不会串流
- env 隔离 :  父进程 env 仅按白名单 copy + 注入 PYTHONPATH，子进程改 env 不影响父进程
- 并发控制 :  asyncio.Semaphore 限制同时存在的子进程数量
- 超时保护 :  wait_for 超时即 kill 子进程并回收

两种调用：
- execute(command)              — 任意 bash 命令字符串（推荐）
- run_skill(skill_module, *args) — 便利封装，等价于 `python -m skills.<module> <args...>`
"""

import asyncio
import json
import logging
import os
import shlex
import shutil
import sys
import tempfile
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 透传到子进程的环境变量白名单：避免父进程敏感变量泄漏，
# 但保留 shell / Python / 编码 / 临时目录所需的最小集合
_DEFAULT_ENV_PASSTHROUGH = (
    # 通用
    "PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ",
    "HOME", "USER", "LOGNAME",
    # Windows
    "USERNAME", "USERPROFILE", "TEMP", "TMP",
    "SystemRoot", "SystemDrive", "ComSpec", "windir",
    "PATHEXT", "APPDATA", "LOCALAPPDATA",
)


def _resolve_shell() -> str:
    """解析可用的 shell：SKILL_BASH_PATH > bash > sh"""
    explicit = os.environ.get("SKILL_BASH_PATH")
    if explicit and os.path.exists(explicit):
        return explicit
    for candidate in ("bash", "sh"):
        found = shutil.which(candidate)
        if found:
            return found
    raise RuntimeError(
        "未找到 bash/sh 可执行文件。请安装 bash 或通过 SKILL_BASH_PATH 环境变量指定路径。"
    )


@dataclass
class SkillExecutionResult:
    """单次 bash 执行结果"""
    success: bool
    stdout: str
    stderr: str
    exit_code: int

    @property
    def output(self) -> str:
        """兼容旧字段名"""
        return self.stdout

    @property
    def error(self) -> str:
        """兼容旧字段名"""
        return self.stderr

    def as_json(self):
        """尝试将 stdout 解析为 JSON，失败返回 None"""
        try:
            return json.loads(self.stdout)
        except (json.JSONDecodeError, TypeError):
            return None


class SkillRunner:
    """Skill Bash 沙箱执行器

    Args:
        max_concurrency: 最大并发 subprocess 数；
                         也可通过环境变量 SKILL_MAX_CONCURRENCY 覆盖（默认 10）
        default_timeout: 默认超时秒数
        shell_path:      指定 shell 可执行文件路径，默认自动解析（bash > sh）

    每次 execute() 之间完全互不干扰：临时目录 + 独占 PIPE + env 副本 + 并发信号量。
    """

    def __init__(
        self,
        max_concurrency: int | None = None,
        default_timeout: int = 30,
        shell_path: str | None = None,
    ):
        concurrency = max_concurrency or int(
            os.environ.get("SKILL_MAX_CONCURRENCY", "10")
        )
        self._max_concurrency = concurrency
        self._default_timeout = default_timeout
        self._shell_path = shell_path or _resolve_shell()
        self._semaphore: asyncio.Semaphore | None = None
        logger.info(
            "SkillRunner 初始化: shell=%s, 并发上限=%d, 超时=%ds",
            self._shell_path, concurrency, default_timeout,
        )

    def _get_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrency)
        return self._semaphore

    def _build_env(self, extra_env: dict[str, str] | None = None) -> dict[str, str]:
        """构造子进程环境变量（白名单 copy + 必要注入）"""
        env: dict[str, str] = {
            k: os.environ[k]
            for k in _DEFAULT_ENV_PASSTHROUGH
            if k in os.environ
        }
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            _PROJECT_ROOT + os.pathsep + existing_pp if existing_pp else _PROJECT_ROOT
        )
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        if extra_env:
            env.update(extra_env)
        return env

    async def execute(
        self,
        command: str,
        timeout: int | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> SkillExecutionResult:
        """在独立 bash subprocess 中执行命令。

        Args:
            command:   bash 命令字符串（多条命令可用 ; / && / 换行 拼接）
            timeout:   超时秒数，None 则使用 default_timeout
            extra_env: 追加 / 覆盖环境变量，仅作用于本次 subprocess

        Returns:
            SkillExecutionResult，包含 success / stdout / stderr / exit_code
        """
        timeout = timeout or self._default_timeout
        env = self._build_env(extra_env)

        sem = self._get_semaphore()
        async with sem:
            return await self._run_in_isolated_subprocess(command, timeout, env)

    async def run_skill(
        self,
        skill_module: str,
        *args: str,
        timeout: int | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> SkillExecutionResult:
        """便利封装：等价于 `python -m skills.<skill_module> <args...>`

        所有 args 会被自动 shell-quote，无需调用方自行处理空格 / 引号。

        Example:
            await runner.run_skill(
                "standard_product_name_verification",
                "马粪", "电脑", "--json",
            )
        """
        py = shlex.quote(sys.executable)
        quoted_args = " ".join(shlex.quote(a) for a in args)
        command = f"{py} -m skills.{skill_module} {quoted_args}".strip()
        return await self.execute(command, timeout=timeout, extra_env=extra_env)

    async def _run_in_isolated_subprocess(
        self,
        command: str,
        timeout: int,
        env: dict[str, str],
    ) -> SkillExecutionResult:
        """实际启动 subprocess，独立 cwd + 独占 PIPE，结束后清理"""
        # 每个请求一个独立工作目录 → 文件副作用互不可见
        workdir = tempfile.mkdtemp(prefix="skill_run_")
        try:
            # 把命令写入 cwd 下的脚本文件再执行，而非走 `bash -c "..."`：
            # - 通过文件传递命令体，彻底避开 Windows 上 git-bash / mingw runtime
            #   对含 Unicode / 复杂引号的 -c 参数的解析问题
            # - 文件强制 UTF-8 + LF 行尾，bash 跨平台一致
            # - 脚本随 cwd 一起 rmtree 清理，不留垃圾
            script_path = os.path.join(workdir, "__skill_cmd.sh")
            with open(script_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(command)

            proc = await asyncio.create_subprocess_exec(
                self._shell_path, script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workdir,
                env=env,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                # 超时：kill 子进程并尽量回收，避免孤儿进程残留
                proc.kill()
                try:
                    await asyncio.wait_for(proc.communicate(), timeout=2)
                except asyncio.TimeoutError:
                    pass
                logger.warning("Skill 执行超时 (%ds): %s", timeout, command[:160])
                return SkillExecutionResult(
                    success=False,
                    stdout="",
                    stderr=f"[执行超时 {timeout}s] 命令: {command}",
                    exit_code=-1,
                )

            stdout_str = stdout.decode("utf-8", errors="replace").strip()
            stderr_str = stderr.decode("utf-8", errors="replace").strip()
            exit_code = proc.returncode if proc.returncode is not None else -1
            success = exit_code == 0

            if not success:
                logger.warning(
                    "Skill 执行失败 (exit=%d): %s\nstderr: %s",
                    exit_code, command[:160], stderr_str[:300],
                )

            return SkillExecutionResult(
                success=success,
                stdout=stdout_str,
                stderr=stderr_str,
                exit_code=exit_code,
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
