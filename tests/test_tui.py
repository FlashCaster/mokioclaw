"""M7c 测试：TUI 组件（只测 import 与 Modal 逻辑，不实际启动 TUI）。"""


def test_tui_modules_importable():
    from mokioclaw.tui.app import ApprovalModal, MokioClawApp

    assert ApprovalModal.__name__ == "ApprovalModal"
    assert MokioClawApp.__name__ == "MokioClawApp"


def test_approval_modal_holds_command():
    from mokioclaw.tui.app import ApprovalModal

    modal = ApprovalModal("pip install requests")
    assert modal.command == "pip install requests"
    # 默认拒绝绑定存在（y 批准 / n 拒绝 / escape 拒绝）——BINDINGS 为 (key, action, desc) 元组
    binding_keys = {binding[0] for binding in modal.BINDINGS}
    assert "y" in binding_keys
    assert "n" in binding_keys
    assert "escape" in binding_keys
