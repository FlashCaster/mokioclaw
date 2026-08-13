"""Runtime state holding workspace context for all tools."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RuntimeState:
    """Shared runtime state accessible by all tools and agent components.

    All file operations are sandboxed inside `workspace`.
    """

    workspace: Path = field(default_factory=Path.cwd)
    approval_mode: str = "inline"

    @property
    def workspace_str(self) -> str:
        return str(self.workspace.resolve())
