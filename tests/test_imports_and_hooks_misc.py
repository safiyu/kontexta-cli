import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ORDERS = ["myagy", "myagy.agents", "myagy.core", "myagy.core.coordinator", "myagy.agents.subagent", "myagy.terminal", "myagy.core.session", "myagy.terminal.cli", "myagy.agents.brainstorm", "myagy.agents.compactor", "myagy.context", "myagy.tools", "myagy.context.hooks_loader", "myagy.context.kontexta"]


@pytest.mark.parametrize("first", ORDERS)
def test_import_order_independent(first):
    code = (
        f"import sys; sys.path.insert(0, {os.path.join(ROOT, 'tests')!r}); import sdk_stub; sdk_stub.install()\n"
        f"sys.path.insert(0, {os.path.dirname(ROOT)!r})\n"
        f"import importlib; importlib.import_module({first!r})\n"
        "from myagy import MultiGpuHybridSession; import myagy.core as c; assert c.MultiGpuHybridSession is MultiGpuHybridSession\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr


def test_core_leaf_modules_import_without_sdk():
    code = (
        f"import sys; sys.path.insert(0, {os.path.dirname(ROOT)!r})\n"
        "sys.modules['google'] = None\n"
        "from myagy.core.coordinator import CudaCoordinator\n"
        "from myagy.core.laya import LayaDecisionEngine\n"
        "from myagy.core.checkpoints import Checkpoints\n"
        "from myagy.core.prefs import load_prefs\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr


def test_hook_returning_non_dict_json_is_tolerated(tmp_path):
    from myagy.context.hooks_loader import ExternalHooksManager
    res = ExternalHooksManager.execute_command_hook("echo '[1,2]'", str(tmp_path), {})
    assert isinstance(res, dict) and res["raw_output"] == "[1,2]"


def test_loop_guard_blocks_third_identical_call(session):
    from google.antigravity import types
    session.json_output = True
    guard = session.build_tool_hooks()[0]
    session._recent_tool_calls.clear()
    call = types.ToolCall(name="read", args={"p": 1})
    assert guard(call).allow and guard(call).allow
    assert guard(call).allow is False


def test_loop_guard_records_tool_calls_for_json_output(session):
    from google.antigravity import types
    session.json_output = True
    guard = session.build_tool_hooks()[0]
    session._turn_executed_tools = []
    guard(types.ToolCall(name="grep", args={"q": "x"}))
    assert session._turn_executed_tools == [{"name": "grep", "args": {"q": "x"}}]
    assert session._last_tool_call["name"] == "grep"


def test_reload_command_targets_real_entry_script():
    cli_file = os.path.join(ROOT, "terminal", "cli.py")
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(cli_file)))
    assert os.path.isfile(os.path.join(pkg_root, "antigravity_agent.py"))
