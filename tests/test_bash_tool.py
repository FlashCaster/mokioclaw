from mokioclaw.tools.bash_tool import run_bash
from mokioclaw.core.state import RuntimeState

def test_low_risk_passes(tmp_path):
    state = RuntimeState(workspace=tmp_path, approval_mode="deny")
    result = run_bash(state, "echo hello")

    assert result["ok"] is True
    assert "hello" in result["stdout"]
    assert result["exit_code"] == 0

def test_high_risk_denied(tmp_path):
    # state 用 deny 模式
    state = RuntimeState(workspace=tmp_path, approval_mode="deny")
    # 跑高危命令（不会真的删东西，因为会被拒）
    result = run_bash(state, "rm -rf ~/xxx")

    assert result["ok"] is False
    assert "拒绝" in result["stderr"]
    assert result["exit_code"] == -1

def test_high_risk_inline_prompt(tmp_path, monkeypatch):
    state = RuntimeState(workspace=tmp_path, approval_mode="inline")
    # 把内置 input 替换成"总是返回 'n'"的假函数
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    result = run_bash(state, "rm -rf ~/xxx")

    assert result["ok"] is False
    assert "拒绝" in result["stderr"]
    assert result["exit_code"] == -1

def test_output_truncated(tmp_path):
    state = RuntimeState(workspace=tmp_path, approval_mode="auto")
    # 造一条输出10000个字符的命令
    result = run_bash(state, "yes A | head -c 10000")

    assert len(result["stdout"]) < 10000
    assert "完整见" in result["stdout"]
    out_dir = tmp_path / ".mokioclaw" / "bash-outputs"
    files = list(out_dir.glob("*.out"))
    assert len(files) == 1 

def test_workspace_escape_not_blocked(tmp_path):
    """已知局限：黑名单不覆盖 cd，逃逸当前拦不住。TODO: 白名单重构。"""
    state = RuntimeState(workspace=tmp_path, approval_mode="deny")
    result = run_bash(state, "cd /etc")
    assert result["ok"] is True