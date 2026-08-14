"""集中配置管理 — 所有密钥/配置统一从项目根目录的 .env 文件读取。

约定：
  - 真实密钥只放在本地的 `.env` 文件（已加入 .gitignore，不入库）。
  - 仓库只提交 `.env.example` 模板，新用户复制为 `.env` 后填写即可。
  - 各模块通过本文件的函数读取配置，不要在别处直接 os.getenv。

配置项一览（见 .env.example）：
  - OPENAI_API_KEY            必填，OpenAI 兼容 API 密钥
  - OPENAI_BASE_URL           可选，自定义 API 端点（默认官方）
  - MODEL                     可选，模型名（默认 gpt-4o）
  - TAVILY_API_KEY            可选，Tavily 联网搜索密钥（未配置时 WebSearchTool 报错）
  - MOKIO_CONTEXT_TOKEN_LIMIT 可选，上下文压缩阈值（默认 400000）
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# 模块导入时加载 .env（幂等，重复调用无副作用）
load_dotenv()


@dataclass
class OpenAIConfig:
    """OpenAI 兼容 LLM 配置。"""

    api_key: str
    base_url: str | None
    model: str


def get_openai_config() -> OpenAIConfig:
    """读取 OpenAI 相关配置；OPENAI_API_KEY 未配置时抛异常。"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Copy .env.example to .env and fill in your key."
        )
    return OpenAIConfig(
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL"),
        model=os.getenv("MODEL", "gpt-4o"),
    )


def get_tavily_api_key() -> str | None:
    """读取 Tavily 搜索密钥；未配置返回 None（WebSearchTool 会返回友好错误）。"""
    return os.getenv("TAVILY_API_KEY")


def get_context_token_limit(default: int = 400000) -> int:
    """读取上下文压缩阈值 MOKIO_CONTEXT_TOKEN_LIMIT；非法值回退到 default。"""
    raw = os.getenv("MOKIO_CONTEXT_TOKEN_LIMIT", str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default
