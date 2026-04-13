import json
import logging
import requests

from utils.helpers import retry

logger = logging.getLogger(__name__)


@retry(max_retries=3, sleep_seconds=5.0)
def chat(messages: str, vendor: str = "aliyun", model: str = "qwen3.6-plus", system: str = None) -> str:
    """
    调用自有模型服务。
    返回 response["choices"][0]["message"]["content"]。
    """
    if vendor == "servyou":
        URL = f"http://10.199.0.7:5000/api/llm/{vendor}/v1/chat/completions"
        app_id = "sk-d75b519b704d4d348245efe435f08ff3"
    else:
        URL = f"http://mlp.paas.dc.servyou-it.com/mudgate/api/llm/{vendor}/v1/chat/completions"
        app_id = "sk-0609aa6d08de4413a72e14b3fb8fbab1"

    HEADERS = {"Content-Type": "application/json", "Authorization": app_id}
    messages_payload = []
    if system:
        messages_payload.append({"role": "system", "content": system})
    messages_payload.append({"role": "user", "content": messages})
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
