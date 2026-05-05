"""Verbose 模式日志模块。

功能：
1. 每个请求一个 session，对应一个独立 JSONL 日志文件，完整记录
   根智能体 / 子智能体 / 汇总层 每一步 LLM 的输入（system+user prompt）与输出。
2. 通过 ContextVar + ThreadPoolExecutor.submit 的 monkey-patch 传递会话上下文，
   无须在每个 chat() 调用点显式传参。
3. 达到每日 20MB 阈值时，自动打包已关闭日志为 zip，并以"起止时间范围"命名，
   防止磁盘无限增长。

日志存放：
  <project_root>/verbose_logs/
    ├── 20260424_093015_<session_id>.jsonl      # 活跃 / 最近 session
    ├── 20260424_095012_<session_id>.jsonl
    └── archives/
        └── verbose_logs_20260424_093015__20260424_181524.zip
"""
from __future__ import annotations

import os
import json
import time
import uuid
import logging
import threading
import contextvars
import zipfile
import contextlib
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Context vars：session_id / agent_id / step_label / prompt_vars
# -------------------------------------------------------------------
_session_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "verbose_session_id", default=None
)
_agent_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "verbose_agent_id", default=None
)
_step_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "verbose_step_label", default=None
)
_prompt_vars_var: contextvars.ContextVar[dict | list | str | None] = contextvars.ContextVar(
    "verbose_prompt_vars", default=None
)
_PROMPT_VARS_UNSET = object()

# -------------------------------------------------------------------
# ThreadPoolExecutor.submit monkey-patch：让 ContextVar 跨线程继承
# -------------------------------------------------------------------
# Python stdlib 的 ThreadPoolExecutor.submit 不会把调用方的 ContextVar
# 复制到 worker 线程。这里做幂等 monkey-patch：把 contextvars.copy_context()
# 捕获的快照交给 worker 调用，从而让 session_id / agent_id 等跨池传播。
# asyncio.to_thread 本身已复制 context，这里补齐显式 ThreadPoolExecutor 场景。
_SUBMIT_PATCHED = False
_ORIG_SUBMIT = ThreadPoolExecutor.submit


def _patch_executor_submit() -> None:
    global _SUBMIT_PATCHED
    if _SUBMIT_PATCHED:
        return

    def _context_aware_submit(self, fn, /, *args, **kwargs):
        ctx = contextvars.copy_context()

        def _runner(*a, **kw):
            return ctx.run(fn, *a, **kw)

        return _ORIG_SUBMIT(self, _runner, *args, **kwargs)

    ThreadPoolExecutor.submit = _context_aware_submit  # type: ignore[assignment]
    _SUBMIT_PATCHED = True
    logger.debug("[Verbose] ThreadPoolExecutor.submit monkey-patched for ctx propagation")


# -------------------------------------------------------------------
# 路径 / 阈值配置
# -------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_LOG_DIR = _PROJECT_ROOT / "verbose_logs"
_ARCHIVE_SUBDIR = "archives"
_DEFAULT_THRESHOLD_BYTES = 20 * 1024 * 1024  # 20 MB


def _env_bool(key: str, default: bool = True) -> bool:
    raw = os.environ.get(key, "")
    if not raw:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


VERBOSE_DEFAULT_ENABLED = _env_bool("VERBOSE_TRACE", True)

_LOG_DIR = Path(os.environ.get("VERBOSE_LOG_DIR", str(_DEFAULT_LOG_DIR)))
try:
    _THRESHOLD_BYTES = int(os.environ.get("VERBOSE_ARCHIVE_MB", "20")) * 1024 * 1024
except ValueError:
    _THRESHOLD_BYTES = _DEFAULT_THRESHOLD_BYTES


# -------------------------------------------------------------------
# Session
# -------------------------------------------------------------------
class _Session:
    """单请求的日志容器。线程安全追加写入 JSONL。"""

    __slots__ = ("session_id", "file_path", "fp", "lock", "opened_at",
                 "event_seq", "meta")

    def __init__(self, session_id: str, file_path: Path, meta: dict | None = None):
        self.session_id = session_id
        self.file_path = file_path
        self.meta = meta or {}
        self.lock = threading.Lock()
        self.fp = file_path.open("a", encoding="utf-8", buffering=1)
        self.opened_at = time.time()
        self.event_seq = 0

    def write_event(self, payload: dict) -> None:
        with self.lock:
            self.event_seq += 1
            payload.setdefault("seq", self.event_seq)
            line = json.dumps(payload, ensure_ascii=False)
            self.fp.write(line)
            self.fp.write("\n")

    def close(self) -> None:
        try:
            self.fp.flush()
            self.fp.close()
        except Exception:
            pass


_sessions_lock = threading.Lock()
_sessions: dict[str, _Session] = {}
_archive_lock = threading.Lock()


def _ensure_log_dirs() -> Path:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    (_LOG_DIR / _ARCHIVE_SUBDIR).mkdir(parents=True, exist_ok=True)
    return _LOG_DIR


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


# -------------------------------------------------------------------
# 公共 API
# -------------------------------------------------------------------
def is_session_active() -> bool:
    """当前执行上下文是否处于活跃 verbose session。"""
    return _session_var.get() is not None


def current_session_id() -> str | None:
    return _session_var.get()


def current_agent_id() -> str | None:
    return _agent_var.get()


def current_step() -> str | None:
    return _step_var.get()


def current_prompt_vars() -> dict | list | str | None:
    return _prompt_vars_var.get()


@contextlib.contextmanager
def open_session(
    session_id: str | None = None,
    meta: dict | None = None,
    enabled: bool = True,
):
    """开启 / 关闭一次请求级 session。

    - enabled=False：直接走空 context，没有任何磁盘副作用（兼容默认关闭）。
    - enabled=True：
        * 首次调用会幂等 monkey-patch ThreadPoolExecutor.submit
        * 生成 session_id（默认自动，也可显式传入，便于跨服务串联）
        * 建立 <timestamp>_<session_id>.jsonl 日志文件
        * 退出时写入 session_end 并触发归档检查
    """
    if not enabled:
        yield None
        return

    _patch_executor_submit()
    _ensure_log_dirs()

    sid = session_id or f"sess-{uuid.uuid4().hex[:12]}"
    safe_sid = sid.replace("/", "_").replace("\\", "_").replace(":", "_")
    ts_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = _LOG_DIR / f"{ts_prefix}_{safe_sid}.jsonl"

    session = _Session(sid, file_path, meta=meta)
    with _sessions_lock:
        _sessions[sid] = session

    token = _session_var.set(sid)
    try:
        session.write_event({
            "type": "session_start",
            "timestamp": _now_iso(),
            "session_id": sid,
            "meta": meta or {},
        })
        logger.info(
            f"[Verbose] session 开启: id={sid}, file={file_path.name}"
        )
        yield session
    except Exception as e:
        # 记录异常到日志再向上抛
        try:
            session.write_event({
                "type": "session_error",
                "timestamp": _now_iso(),
                "session_id": sid,
                "error": repr(e),
            })
        except Exception:
            pass
        raise
    finally:
        try:
            session.write_event({
                "type": "session_end",
                "timestamp": _now_iso(),
                "session_id": sid,
                "duration_ms": int((time.time() - session.opened_at) * 1000),
                "total_events": session.event_seq,
            })
        except Exception:
            pass
        with _sessions_lock:
            _sessions.pop(sid, None)
        session.close()
        _session_var.reset(token)
        try:
            maybe_archive_logs()
        except Exception as e:
            logger.warning(f"[Verbose] 归档检查异常（不影响主流程）: {e}")


@contextlib.contextmanager
def agent_scope(
    agent_id: str | None = None,
    step: str | None = None,
    prompt_vars: dict | list | str | None | object = _PROMPT_VARS_UNSET,
):
    """为当前执行上下文绑定 agent_id 与 step 标签（嵌套安全）。

    verbose 未开启时退化为空 context，不产生任何副作用。
    """
    if _session_var.get() is None:
        yield
        return

    tokens: list[tuple[str, object]] = []
    if agent_id is not None:
        tokens.append(("agent", _agent_var.set(agent_id)))
    if step is not None:
        tokens.append(("step", _step_var.set(step)))
    if prompt_vars is not _PROMPT_VARS_UNSET:
        tokens.append(("prompt_vars", _prompt_vars_var.set(prompt_vars)))
    try:
        yield
    finally:
        for name, tk in reversed(tokens):
            if name == "agent":
                _agent_var.reset(tk)  # type: ignore[arg-type]
            elif name == "step":
                _step_var.reset(tk)  # type: ignore[arg-type]
            elif name == "prompt_vars":
                _prompt_vars_var.reset(tk)  # type: ignore[arg-type]


def step_scope(step: str, prompt_vars: dict | list | str | None = None):
    """只改 step 的便捷作用域，可顺带记录该步使用的 prompt 变量名。"""
    return agent_scope(step=step, prompt_vars=prompt_vars)


def log_llm_call(
    *,
    prompt: str,
    response: str,
    system: str | None = None,
    vendor: str | None = None,
    model: str | None = None,
    elapsed_ms: int | None = None,
    extra: dict | None = None,
) -> None:
    """由 llm.client.chat 成功返回后统一调用；session 未开启时直接 no-op。"""
    sid = _session_var.get()
    if sid is None:
        return
    with _sessions_lock:
        session = _sessions.get(sid)
    if session is None:
        return
    payload = {
        "type": "llm_call",
        "timestamp": _now_iso(),
        "session_id": sid,
        "agent_id": _agent_var.get(),
        "step": _step_var.get(),
        "thread": threading.current_thread().name,
        "vendor": vendor,
        "model": model,
        "elapsed_ms": elapsed_ms,
        "prompt_vars": _prompt_vars_var.get(),
        "system": system,
        "prompt": prompt,
        "response": response,
    }
    if extra:
        payload["extra"] = extra
    try:
        session.write_event(payload)
    except Exception as e:
        logger.debug(f"[Verbose] write_event 失败（忽略）: {e}")


def log_llm_error(
    *,
    prompt: str,
    error: str,
    system: str | None = None,
    vendor: str | None = None,
    model: str | None = None,
    elapsed_ms: int | None = None,
    extra: dict | None = None,
) -> None:
    """LLM 调用失败时写入错误事件。"""
    sid = _session_var.get()
    if sid is None:
        return
    with _sessions_lock:
        session = _sessions.get(sid)
    if session is None:
        return
    payload = {
        "type": "llm_error",
        "timestamp": _now_iso(),
        "session_id": sid,
        "agent_id": _agent_var.get(),
        "step": _step_var.get(),
        "thread": threading.current_thread().name,
        "vendor": vendor,
        "model": model,
        "elapsed_ms": elapsed_ms,
        "prompt_vars": _prompt_vars_var.get(),
        "system": system,
        "prompt": prompt,
        "error": error,
    }
    if extra:
        payload["extra"] = extra
    try:
        session.write_event(payload)
    except Exception:
        pass


def log_event(event_type: str, **fields) -> None:
    """通用事件日志（非 LLM 调用）。"""
    sid = _session_var.get()
    if sid is None:
        return
    with _sessions_lock:
        session = _sessions.get(sid)
    if session is None:
        return
    payload = {
        "type": event_type,
        "timestamp": _now_iso(),
        "session_id": sid,
        "agent_id": _agent_var.get(),
        "step": _step_var.get(),
        "prompt_vars": _prompt_vars_var.get(),
        "thread": threading.current_thread().name,
    }
    payload.update(fields)
    try:
        session.write_event(payload)
    except Exception:
        pass


# -------------------------------------------------------------------
# 归档：当可归档日志累计体积 ≥ 20MB，打包到 archives/<起止时间>.zip
# -------------------------------------------------------------------
def _collect_archivable_files() -> list[Path]:
    """非活跃 session 对应的 *.jsonl 才参与归档（避免打包正在写入的文件）。"""
    if not _LOG_DIR.exists():
        return []
    with _sessions_lock:
        active_paths = {s.file_path.resolve() for s in _sessions.values()}
    result: list[Path] = []
    for p in _LOG_DIR.glob("*.jsonl"):
        try:
            if p.resolve() in active_paths:
                continue
        except Exception:
            continue
        result.append(p)
    return sorted(result, key=lambda x: x.stat().st_mtime)


def maybe_archive_logs(threshold_bytes: int | None = None) -> str | None:
    """累计体积超过阈值时，把所有可归档日志打包为 zip，并删除原文件。

    返回 zip 路径；未触发则返回 None。
    """
    if threshold_bytes is None:
        threshold_bytes = _THRESHOLD_BYTES
    with _archive_lock:
        files = _collect_archivable_files()
        if not files:
            return None
        total = sum(f.stat().st_size for f in files)
        if total < threshold_bytes:
            return None

        first_mtime = datetime.fromtimestamp(files[0].stat().st_mtime)
        last_mtime = datetime.fromtimestamp(files[-1].stat().st_mtime)
        first_s = first_mtime.strftime("%Y%m%d_%H%M%S")
        last_s = last_mtime.strftime("%Y%m%d_%H%M%S")

        archive_dir = _LOG_DIR / _ARCHIVE_SUBDIR
        archive_dir.mkdir(parents=True, exist_ok=True)
        zip_path = archive_dir / f"verbose_logs_{first_s}__{last_s}.zip"
        suffix = 1
        while zip_path.exists():
            zip_path = archive_dir / f"verbose_logs_{first_s}__{last_s}_{suffix}.zip"
            suffix += 1

        logger.info(
            f"[Verbose] 触发归档：累计 {total / 1024 / 1024:.2f}MB ≥ "
            f"{threshold_bytes / 1024 / 1024:.0f}MB，打包 {len(files)} 个日志 → "
            f"{zip_path.name}"
        )
        try:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
                for f in files:
                    zf.write(f, arcname=f.name)
            for f in files:
                try:
                    f.unlink()
                except Exception as e:
                    logger.warning(f"[Verbose] 删除原始日志 {f.name} 失败: {e}")
            return str(zip_path)
        except Exception as e:
            logger.exception(f"[Verbose] 归档失败: {e}")
            try:
                if zip_path.exists():
                    zip_path.unlink()
            except Exception:
                pass
            return None


def get_log_dir() -> Path:
    """外部工具 / 测试可读取当前日志目录。"""
    return _LOG_DIR
