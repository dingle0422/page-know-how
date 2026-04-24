import json
import time
import logging
import requests

from utils.helpers import retry
from utils.verbose_logger import (
    is_session_active,
    log_llm_call,
    log_llm_error,
)

logger = logging.getLogger(__name__)


@retry(max_retries=3, sleep_seconds=5.0)
def chat(
    messages: str,
    vendor: str = "qwen3.5-122b-a10b",
    model: str = "Qwen3.5-122B-A10B",
    system: str = None,
    enable_thinking: bool = False,
) -> str:
    """
    调用自有模型服务。
    返回 response["choices"][0]["message"]["content"]。

    - vendor="qwen3.5-122b-a10b" (默认): 直连 Qwen3.5-122B-A10B 自部署服务
      （OpenAI 兼容协议，无鉴权，可通过 enable_thinking 控制思考模式）
    - vendor="qwen3.5-27b": 直连 MLP 上的 Qwen3.5-27B 自部署服务
    - vendor="servyou": 内网直连
    - 其他: 走 mudgate 网关
    """
    messages_payload = []
    if system:
        messages_payload.append({"role": "system", "content": system})
    messages_payload.append({"role": "user", "content": messages})

    if vendor == "qwen3.5-122b-a10b":
        URL = "http://211.137.21.19:17860/v1/chat/completions"
        HEADERS = {"Content-Type": "application/json"}
        PAYLOAD = {
            "model": "Qwen3.5-122B-A10B",
            "messages": messages_payload,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
            # "temperature": 0.5,
        }
    elif vendor == "qwen3.6-35b-a3b":
        URL = "http://mlp.paas.dc.servyou-it.com/qwen3.6-35b-a3b/v1/chat/completions"
        HEADERS = {"Content-Type": "application/json"}
        PAYLOAD = {
            "model": "Qwen/Qwen3.6-35B-A3B",
            "messages": messages_payload,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
            # "temperature": 0.5,
        }
       
    elif vendor == "qwen3.5-27b":
        URL = "http://mlp.paas.dc.servyou-it.com/qwen3.5-27b/v1/chat/completions"
        HEADERS = {"Content-Type": "application/json"}
        PAYLOAD = {
            "model": "Qwen/Qwen3.5-27B",
            "messages": messages_payload,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
    else:
        if vendor == "servyou":
            URL = f"http://10.199.0.7:5000/api/llm/{vendor}/v1/chat/completions"
            app_id = "sk-d75b519b704d4d348245efe435f08ff3"
        else:
            URL = f"http://mlp.paas.dc.servyou-it.com/mudgate/api/llm/{vendor}/v1/chat/completions"
            app_id = "sk-0609aa6d08de4413a72e14b3fb8fbab1"

        HEADERS = {"Content-Type": "application/json", "Authorization": app_id}
        PAYLOAD = {
            "appId": app_id,
            "model": model,
            "messages": messages_payload,
            "stream": False,
            # "top_p": 0.7,
            # "temperature": 0.5,
        }
        if enable_thinking:
            PAYLOAD["chat_template_kwargs"] = {"enable_thinking": True}

    logger.debug(f"LLM 请求 [{vendor}/{model}]: {messages[:100]}...")

    verbose_on = is_session_active()
    t0 = time.time() if verbose_on else None

    try:
        response = requests.post(
            URL, data=json.dumps(PAYLOAD), headers=HEADERS, timeout=(30, 360)
        ).json()
    except Exception as e:
        if verbose_on:
            elapsed_ms = int((time.time() - t0) * 1000) if t0 is not None else None
            try:
                log_llm_error(
                    prompt=messages,
                    error=repr(e),
                    system=system,
                    vendor=vendor,
                    model=model,
                    elapsed_ms=elapsed_ms,
                    extra={"enable_thinking": enable_thinking},
                )
            except Exception:
                pass
        raise

    if "success" in response:
        err = response.get("errorContext", "未知错误")
        if verbose_on:
            elapsed_ms = int((time.time() - t0) * 1000) if t0 is not None else None
            try:
                log_llm_error(
                    prompt=messages,
                    error=str(err),
                    system=system,
                    vendor=vendor,
                    model=model,
                    elapsed_ms=elapsed_ms,
                    extra={"enable_thinking": enable_thinking},
                )
            except Exception:
                pass
        raise Exception(err)

    result = response["choices"][0]["message"]
    content = result["content"] if isinstance(result, dict) else str(result)
    # 部分 OpenAI 兼容服务（如 deepseek-reasoner / deepseek-v3.2 的 thinking 模式）
    # 会把思考过程单独放到 message.reasoning_content 字段，而不是塞回 content。
    # 统一把非空 reasoning_content 以 <think>…</think> 前缀并入 content，保持
    # 与 qwen 系列（原生把 <think> 写进 content）一致的下游体验。
    if isinstance(result, dict):
        reasoning = result.get("reasoning_content")
        if reasoning and isinstance(reasoning, str) and reasoning.strip():
            if "<think>" not in (content or ""):
                content = f"<think>{reasoning.strip()}</think>\n{content or ''}"
    logger.debug(f"LLM 响应: {content[:100]}...")

    if verbose_on:
        elapsed_ms = int((time.time() - t0) * 1000) if t0 is not None else None
        try:
            log_llm_call(
                prompt=messages,
                response=content,
                system=system,
                vendor=vendor,
                model=model,
                elapsed_ms=elapsed_ms,
                extra={"enable_thinking": enable_thinking},
            )
        except Exception:
            pass

    return content
