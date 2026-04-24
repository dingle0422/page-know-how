import re
import time
import functools
import logging

logger = logging.getLogger(__name__)


def retry(max_retries: int = 3, sleep_seconds: float = 5.0):
    """重试装饰器，在函数抛出异常时自动重试"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(
                        f"[retry] {func.__name__} 第{attempt}次调用失败: {e}"
                    )
                    if attempt < max_retries:
                        time.sleep(sleep_seconds)
            raise last_exception
        return wrapper
    return decorator


def sanitize_filename(name: str, max_length: int = 50) -> str:
    """将标题中不适合作为文件夹名的特殊字符替换为下划线，并截断过长的名称"""
    sanitized = re.sub(r'[\x00-\x1f\x7f\\/:*?"<>|]', '_', name)
    sanitized = re.sub(r'_+', '_', sanitized)
    sanitized = sanitized.strip().strip('_').rstrip('.')
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip('_').rstrip('.')
    return sanitized or "untitled"


def truncate_text(text: str, max_length: int = 200) -> str:
    """截断文本到指定长度，用于生成摘要"""
    text = text.strip()
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


# 匹配最开头（允许前置空白 / BOM）的一段 <think>...</think>，支持多行、non-greedy。
# 仅剥离**文首的第一段**，不动后面正文里可能出现的 <think> 字样。
_THINK_BLOCK_RE = re.compile(
    r"\A[\s\ufeff]*<think[^>]*>(?P<body>.*?)</think>\s*",
    re.DOTALL | re.IGNORECASE,
)


def split_think_block(text: str) -> tuple[str, str]:
    """把 LLM 返回的带 <think>...</think> 前缀的文本拆成 (think, answer)。

    - 兼容 qwen3.5/3.6 原生把 <think> 写进 content 的风格，
      以及 deepseek-v3.2 / deepseek-reasoner 由 llm/client.py 统一回注的前缀。
    - 仅剥离文首的第一段 think 块；如果 text 开头没有 <think>，返回 ("", text) 不改动。
    - 幂等：对已经不含 think 前缀的字符串重复调用是安全的。

    返回：
        (think_content, answer_body)
        think_content: <think> 标签之间的内容（已 strip；不含标签本身），可能是空串
        answer_body:   剩余的业务主体（已 strip）
    """
    if not text:
        return "", text or ""
    m = _THINK_BLOCK_RE.match(text)
    if not m:
        return "", text
    think = (m.group("body") or "").strip()
    answer = text[m.end():].lstrip()
    return think, answer
