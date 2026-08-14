"""Tool registry: builds the StructuredTool list for model.bind_tools()."""

from langchain_core.tools import StructuredTool

from mokioclaw.core.state import RuntimeState
from mokioclaw.tools.bash_tool import run_bash
from mokioclaw.tools.file_tools import edit_file, read_file, write_file
from mokioclaw.tools.grep_tool import grep


def build_tools(state: RuntimeState) -> list[StructuredTool]:
    """Build the list of StructuredTools bound to the given RuntimeState.

    Each tool is a closure over `state`, so all file operations are
    automatically sandboxed inside `state.workspace`.

    Returns:
        List of StructuredTool instances ready for model.bind_tools().
    """
    return [
        StructuredTool.from_function(
            name="FileReadTool",
            description=(
                "Read a text file from the workspace. "
                "Use before editing existing files to understand their contents."
            ),
            func=lambda file_path, offset=0, limit=2000: read_file(
                state, file_path, offset, limit
            ),
        ),
        StructuredTool.from_function(
            name="FileWriteTool",
            description=(
                "Create a new file or overwrite an existing file in the workspace. "
                "Provide the full file content."
            ),
            func=lambda file_path, content: write_file(state, file_path, content),
        ),
        StructuredTool.from_function(
            name="FileEditTool",
            description=(
                "Replace a single, unique text fragment in an existing file. "
                "old_text must match exactly one location — make it specific. "
                "Use FileReadTool first to find the exact text to replace."
            ),
            func=lambda file_path, old_text, new_text: edit_file(
                state, file_path, old_text, new_text
            ),
        ),
        StructuredTool.from_function(
            name="GrepTool",
            description=(
                "Search for a regex pattern in workspace files. "
                "Use to locate code, find definitions, or check for patterns."
            ),
            func=lambda pattern, path=".", glob="*", head_limit=20, ignore_case=True: grep(
                state, pattern, path, glob, head_limit, ignore_case
            ),
        ),
        StructuredTool.from_function(
            name="BashTool",
            description=(
                "Execute a shell command inside the workspace directory. "
                "Use to run code, install dependencies, or check results. "
                "Command runs from workspace root — use relative paths. "
                "Set timeout_seconds for long-running commands (default 120)."
            ),
            func=lambda command, timeout_seconds=120: run_bash(
                state, command, timeout_seconds
            ),
        ),
    ]


def build_read_only_tools(state: RuntimeState) -> list[StructuredTool]:
    """Build read-only tools for the verifier (no write/edit tools).

    The verifier may read files, grep, and run shell checks, but cannot
    modify the workspace — preventing "check" from becoming "fix in disguise".
    """
    return [
        StructuredTool.from_function(
            name="FileReadTool",
            description="Read a text file from the workspace. Use to inspect results.",
            func=lambda file_path, offset=0, limit=2000: read_file(
                state, file_path, offset, limit
            ),
        ),
        StructuredTool.from_function(
            name="GrepTool",
            description=(
                "Search for a regex pattern in workspace files. "
                "Use to locate code, find definitions, or check for patterns."
            ),
            func=lambda pattern, path=".", glob="*", head_limit=20, ignore_case=True: grep(
                state, pattern, path, glob, head_limit, ignore_case
            ),
        ),
        StructuredTool.from_function(
            name="BashTool",
            description=(
                "Execute a read-only shell command inside the workspace. "
                "Use to run verification commands and check results. "
                "Do NOT use this to modify files."
            ),
            func=lambda command, timeout_seconds=120: run_bash(
                state, command, timeout_seconds
            ),
        ),
    ]
