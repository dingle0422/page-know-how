"""服务端配置，全部通过环境变量 / .env 文件注入。"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """运行时配置。

    - ``STORE_DIR``：LanceDB 数据目录，每个 policy 一张子目录 ``{policy_id}.lance``。
    - ``API_KEY``：客户端 X-API-Key / Bearer Token 校验值；为空时关闭鉴权（仅供本地开发）。
    - ``ENABLE_SCALAR_INDEX``：是否在 ``kind`` / ``parent_chunk_index`` 上建标量索引。
      极小表（< 1k chunks）可关掉以省构建时间。
    - ``RRF_K``：RRF 融合常量，与主项目 ``inference/config.py::RRF_K`` 默认对齐。
    """

    store_dir: str = "./data"
    api_key: str = ""
    host: str = "0.0.0.0"
    port: int = 8088
    log_level: str = "INFO"
    enable_scalar_index: bool = True
    rrf_k: int = 60

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
