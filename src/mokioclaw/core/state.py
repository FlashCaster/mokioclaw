"""Runtime state holding workspace context for all tools."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mokioclaw.core.trace import TraceWriter


@dataclass
class RuntimeState:
    """Shared runtime state accessible by all tools and agent components.

    All file operations are sandboxed inside `workspace`.
    """

    workspace: Path = field(default_factory=Path.cwd)
    approval_mode: str = "inline"
    checkpoint_mode: str = "light"  # light / strict / off
    trace: Any = None  # TraceWriter | None（旁路观测）
    approval_callback: Any = None  # callable(command)->bool | None（TUI 审批桥）

    @property
    def workspace_str(self) -> str:
        return str(self.workspace.resolve())
