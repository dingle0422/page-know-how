import json
import logging
import requests

from utils.helpers import retry

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
            "temperature": 0.5,
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
            "top_p": 0.7,
            "temperature": 0.5,
        }

    logger.debug(f"LLM 请求 [{vendor}/{model}]: {messages[:100]}...")
    response = requests.post(URL, data=json.dumps(PAYLOAD), headers=HEADERS, timeout=(30, 360)).json()

    if "success" in response:
        raise Exception(response.get("errorContext", "未知错误"))

    result = response["choices"][0]["message"]
    content = result["content"] if isinstance(result, dict) else str(result)
    logger.debug(f"LLM 响应: {content[:100]}...")
    return content
