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
