"""Bash tool: execute shell commands within the workspace."""

import subprocess
import time

from mokioclaw.core.state import RuntimeState


def run_bash(
    state: RuntimeState,
    command: str,
    timeout_seconds: int | None = None,
) -> dict:
    """Execute a shell command inside the workspace directory.

    Args:
        command: Shell command to execute.
        timeout_seconds: Max execution time (default 120).

    Returns:
        dict with keys: ok, command, exit_code, stdout, stderr, duration_seconds.
    """
    if timeout_seconds is None:
        timeout_seconds = 120

    start = time.monotonic()
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=str(state.workspace),
        )
        duration = round(time.monotonic() - start, 3)

        return {
            "ok": result.returncode == 0,
            "command": command,
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration_seconds": duration,
        }
    except subprocess.TimeoutExpired:
        duration = round(time.monotonic() - start, 3)
        return {
            "ok": False,
            "command": command,
            "exit_code": -1,
            "stdout": "",
            "stderr": f"Command timed out after {timeout_seconds}s",
            "duration_seconds": duration,
        }
    except Exception as e:
        duration = round(time.monotonic() - start, 3)
        return {
            "ok": False,
            "command": command,
            "exit_code": -1,
            "stdout": "",
            "stderr": str(e),
            "duration_seconds": duration,
        }
