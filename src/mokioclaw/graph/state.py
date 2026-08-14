"""LangGraph state definition for the planner→actor→verifier workflow (M4)."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

from mokioclaw.core.state import RuntimeState


class TodoItem(TypedDict):
    id: str
    content: str
    status: str
    note: str


class VerificationCheck(TypedDict, total=False):
    name: str
    passed: bool
    detail: str


class SourceItem(TypedDict, total=False):
    title: str
    url: str
    content: str
    score: float


class AgentHandoff(TypedDict, total=False):
    from_agent: str
    to_agent: str
    instruction: str
    result: str


class MokioGraphState(TypedDict, total=False):
    # 输入
    task: str
    runtime: RuntimeState
    # 消息（add_messages 自动追加合并）
    messages: Annotated[list[BaseMessage], add_messages]
    # 计划（planner 产出，actor 消费）
    plan_summary: str
    todos: list[TodoItem]
    acceptance_criteria: list[str]
    verification_commands: list[str]
    # 验证（verifier 产出）
    passed: bool
    attempts: int
    max_attempts: int
    verifier_summary: str
    verification_checks: list[VerificationCheck]
    recommended_next_instruction: str
    # 多智能体（M6）
    research_notes: str
    sources: list[SourceItem]
    agent_handoffs: list[AgentHandoff]
    code_agent_summary: str
    # 记忆 / 上下文压缩（M5）
    context_summary: str
    context_token_count: int
    context_token_limit: int
    context_should_compress: bool
    context_next_node: str
    compression_events: list[dict]
    history_summary: str
    memory_snapshot: dict
    last_error: str
    # 收尾
    final_answer: str
