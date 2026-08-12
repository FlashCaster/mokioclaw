"""LLM provider factory — creates a ChatOpenAI instance from .env config."""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def create_model() -> ChatOpenAI:
    """Create a ChatOpenAI model from environment variables.

    Reads:
        OPENAI_API_KEY (required) — API key
        OPENAI_BASE_URL (optional) — custom base URL
        MODEL (optional, default "gpt-4o") — model name

    Returns:
        Configured ChatOpenAI instance with tool-calling enabled.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Copy .env.example to .env and fill in your key."
        )

    base_url = os.getenv("OPENAI_BASE_URL")
    model_name = os.getenv("MODEL", "gpt-4o")

    kwargs: dict = {
        "api_key": api_key,
        "model": model_name,
        "temperature": 0,
    }
    if base_url:
        kwargs["base_url"] = base_url

    return ChatOpenAI(**kwargs)
