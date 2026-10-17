import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BOOT = (
    "import sys, importlib.util\n"
    f"spec = importlib.util.spec_from_file_location('kontexta_cli', {os.path.join(ROOT, '__init__.py')!r}, submodule_search_locations=[{ROOT!r}])\n"
    "mod = importlib.util.module_from_spec(spec); sys.modules['kontexta_cli'] = mod; spec.loader.exec_module(mod)\n"
)

ORDERS = ["kontexta_cli", "kontexta_cli.agents", "kontexta_cli.core", "kontexta_cli.core.coordinator", "kontexta_cli.agents.subagent", "kontexta_cli.terminal", "kontexta_cli.core.session", "kontexta_cli.terminal.cli", "kontexta_cli.agents.brainstorm", "kontexta_cli.agents.compactor", "kontexta_cli.context", "kontexta_cli.tools", "kontexta_cli.context.hooks_loader", "kontexta_cli.context.kontexta"]


@pytest.mark.parametrize("first", ORDERS)
def test_import_order_independent(first):
    code = (
        f"import sys; sys.path.insert(0, {os.path.join(ROOT, 'tests')!r}); import sdk_stub; sdk_stub.install()\n"
        + BOOT +
        f"import importlib; importlib.import_module({first!r})\n"
        "from kontexta_cli import MultiGpuHybridSession; import kontexta_cli.core as c; assert c.MultiGpuHybridSession is MultiGpuHybridSession\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr


def test_core_leaf_modules_import_without_sdk():
    code = (
        "import sys; sys.modules['google'] = None\n"
        + BOOT +
        "from kontexta_cli.core.coordinator import CudaCoordinator\n"
        "from kontexta_cli.core.laya import LayaDecisionEngine\n"
        "from kontexta_cli.core.checkpoints import Checkpoints\n"
        "from kontexta_cli.core.prefs import load_prefs\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr


def test_hook_returning_non_dict_json_is_tolerated(tmp_path):
    from kontexta_cli.context.hooks_loader import ExternalHooksManager
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
