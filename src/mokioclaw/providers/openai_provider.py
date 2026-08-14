"""LLM provider factory — creates a ChatOpenAI instance from .env config."""

from langchain_openai import ChatOpenAI

from mokioclaw.config import get_openai_config


def create_model() -> ChatOpenAI:
    """Create a ChatOpenAI model from centralized .env config.

    Reads via mokioclaw.config: OPENAI_API_KEY / OPENAI_BASE_URL / MODEL.

    Returns:
        Configured ChatOpenAI instance with tool-calling enabled.
    """
    config = get_openai_config()

    kwargs: dict = {
        "api_key": config.api_key,
        "model": config.model,
        "temperature": 0,
    }
    if config.base_url:
        kwargs["base_url"] = config.base_url

    return ChatOpenAI(**kwargs)
