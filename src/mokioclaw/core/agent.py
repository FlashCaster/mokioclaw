"""ReAct agent loop — the heart of Stage 1.

stream_agent_events() runs the observe→think→act→observe loop,
yielding structured events for the CLI layer to display.
"""

from typing import Iterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from mokioclaw.core.state import RuntimeState
from mokioclaw.providers.openai_provider import create_model
from mokioclaw.tools.registry import build_tools

ACTOR_PROMPT = """You are the actor node in MokioClaw's ReAct workflow.

You implement the user's task using tools. Work inside the workspace only.

Rules:
- Use FileWriteTool for new files.
- Use FileReadTool before editing existing files.
- Use FileEditTool for focused edits.
- Use BashTool to run commands and test results.
- BashTool already runs inside the workspace. Use relative paths, never "cd /workspace".
- End with a concise summary of files changed and commands run.
"""


def stream_agent_events(
    task: str,
    state: RuntimeState,
    *,
    max_loops: int = 10,
) -> Iterator[dict]:
    """Run the ReAct loop and yield structured events.

    Event types yielded:
        {"type": "ai_message", "content": str}
        {"type": "tool_call", "name": str, "args": dict}
        {"type": "tool_result", "name": str, "result": dict}
        {"type": "final_answer", "content": str}
        {"type": "error", "message": str}

    Args:
        task: User's task description.
        state: RuntimeState with workspace path.
        max_loops: Maximum ReAct iterations before forcing stop.

    Yields:
        Dict events for each step in the agent's execution.
    """
    model = create_model()
    tools = build_tools(state)
    agent = model.bind_tools(tools)

    messages: list = [
        SystemMessage(content=ACTOR_PROMPT),
        HumanMessage(content=task),
    ]

    for loop_idx in range(max_loops):
        response = agent.invoke(messages)
        messages.append(response)

        # Extract text content for display
        text_content = ""
        if hasattr(response, "content") and isinstance(response.content, str):
            text_content = response.content
        elif hasattr(response, "content") and isinstance(response.content, list):
            # content blocks — collect text parts
            parts = []
            for block in response.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            text_content = "\n".join(parts)

        tool_calls = getattr(response, "tool_calls", None) or []

        if not tool_calls:
            # No tool calls → agent is done
            yield {"type": "final_answer", "content": text_content}
            return

        yield {
            "type": "ai_message",
            "content": text_content,
            "loop": loop_idx + 1,
        }

        for tc in tool_calls:
            tool_name = tc.get("name", "unknown")
            tool_args = tc.get("args", {})

            yield {
                "type": "tool_call",
                "name": tool_name,
                "args": tool_args,
            }

            # Find and execute the matching tool
            tool_map = {t.name: t for t in tools}
            tool = tool_map.get(tool_name)
            if tool is None:
                result = {"ok": False, "error": f"Unknown tool: {tool_name}"}
            else:
                try:
                    result = tool.invoke(tool_args)
                except Exception as e:
                    result = {"ok": False, "error": str(e)}

            yield {
                "type": "tool_result",
                "name": tool_name,
                "result": result,
            }

            # Format result for ToolMessage
            import json
            result_str = json.dumps(result, ensure_ascii=False, default=str)
            messages.append(
                ToolMessage(content=result_str, tool_call_id=tc["id"])
            )

    # Exhausted max loops
    yield {
        "type": "error",
        "message": f"Agent reached max loops ({max_loops}) without finishing.",
    }
