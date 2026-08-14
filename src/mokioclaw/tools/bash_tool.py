"""Bash tool: execute shell commands within the workspace.

安全模型：文件工具靠 paths.py 的路径校验强制沙箱；
BashTool 是命令字符串，无法靠路径解析强制，只能靠「风险分类 + 审批」兜底。
"""

import subprocess
import time
import re

from pathlib import Path
from mokioclaw.core.state import RuntimeState


# 高危命令黑名单（正则）。
# 命中任意一条 → classify_risk 返回 "high"。
# 设计取舍：黑名单偏保守，宁可误拦也不漏拦；M2 阶段刻意保持简单。
DENY_PATTERNS = [
    r"rm\s+-rf\s+[~/]",                        # 删除家目录/根：rm -rf ~ 或 rm -rf /
    r"mkfs|fdisk|dd\s+of=",                    # 磁盘危险操作：格式化/分区/写盘
    r"curl|wget",                              # 网络请求：外发数据
    r"pip\s+install|npm\s+install|uv\s+add",   # 安装依赖：引入外部代码
]
# 单次命令输出的最大字符数
MAX_OUTPUT_CHARS = 6000


def classify_risk(command: str) -> str:
    """判断命令风险等级。

    遍历黑名单，命中任意一条正则即判定高危，否则低危。

    Args:
        command: 待执行的 shell 命令字符串。

    Returns:
        "high"（命中黑名单）或 "low"（未命中）。
    """
    for pat in DENY_PATTERNS:
        if re.search(pat, command):
            return "high"
    return "low"


def _resolve_approval(state: RuntimeState, command: str) -> tuple[bool, str]:
    """根据审批模式决定是否放行高危命令。

    三种模式（state.approval_mode）：
      - auto   ：全部放行（开发期方便，失去安全意义）
      - deny   ：高危命令直接拒绝（最严格）
      - inline ：高危命令交互式询问用户 [y/N]（默认，推荐）

    判定顺序（放行优先 → 拦截其次 → 询问兜底）：
      1. 低风险 或 auto 模式 → 放行
      2. deny 模式 → 拒绝
      3. inline 模式 → 问用户

    Args:
        state: 运行时状态（含 approval_mode 字段）。
        command: 待执行的 shell 命令。

    Returns:
        (是否批准, 原因说明) 元组。
    """
    mode = state.approval_mode
    risk = classify_risk(command)

    # 低风险命令永远放行；auto 模式下全部放行（两者结果相同，用 or 合并）
    if risk == "low" or mode == "auto":
        return True, "放行"

    # deny 模式：高危直接拒绝，不打扰用户
    if mode == "deny":
        return False, f"高危命令被拒绝(deny 模式): {command}"

    # inline 模式：优先用注入的审批回调（TUI 模式），否则交互询问。
    # [y/N] 大写 N 表示默认拒绝（直接回车 = 拒绝，安全优先）
    if getattr(state, "approval_callback", None) is not None:
        approved = bool(state.approval_callback(command))
        return approved, "已批准" if approved else "用户拒绝"
    answer = input(f"高危命令，批准执行? [y/N] {command}\n> ").strip().lower()
    return answer in ("y", "yes"), "已批准" if answer in ("y", "yes") else "用户拒绝"

def _truncate(text: str, command: str, ws: Path) -> str:
    # 判断：如果没超限，直接原样返回（不截断）
    if len(text) <= MAX_OUTPUT_CHARS:
        return text

    # 超限了：先建落盘目录 .mokioclaw/bash-outputs
    log_dir = ws / ".mokioclaw" / "bash-outputs"
    log_dir.mkdir(parents=True, exist_ok=True)
    # 生成唯一文件名，避免两次截断互相覆盖
    fname = log_dir / f"{int(time.time())}.out"
    # 把完整 text 写入文件
    fname.write_text(
       f"# command: {command}\n"
       f"# time: {int(time.time())}\n"
       f"{'=' * 60}\n\n"
       f"{text}",
       encoding="utf-8",
   )
    # 返回截断版 + 指向完整文件的提示
    return text[:MAX_OUTPUT_CHARS] + f"\n...（输出截断，完整见 {fname}）"

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

    approved, reason = _resolve_approval(state, command)
    if not approved:
        return {
            "ok": False,
            "command": command,
            "exit_code": -1,
            "stdout": "",
            "stderr": reason,
            "duration_seconds": 0.0
        }

    start = time.monotonic()
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,   # 捕获 stdout/stderr，不让命令直接刷屏
            text=True,             # 以文本（非字节）返回输出
            timeout=timeout_seconds,
            cwd=str(state.workspace),  # 沙箱关键：命令只能从 workspace 内启动
        )
        duration = round(time.monotonic() - start, 3)

        return {
            "ok": result.returncode == 0,
            "command": command,
            "exit_code": result.returncode,
            "stdout": _truncate(result.stdout, command, state.workspace),
            "stderr": _truncate(result.stderr, command, state.workspace),
            "duration_seconds": duration,
        }
    except subprocess.TimeoutExpired:
        # 超时：返回结构化失败，exit_code 用 -1 表示"无正常退出码"
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
        # 其他异常：同样结构化返回，不向上抛，保证 agent 循环不中断
        duration = round(time.monotonic() - start, 3)
        return {
            "ok": False,
            "command": command,
            "exit_code": -1,
            "stdout": "",
            "stderr": str(e),
            "duration_seconds": duration,
        }
