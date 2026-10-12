import json
import os
import subprocess
import sys
import time

import pytest

from myagy.context import hooks_loader
from myagy.context.hooks_loader import ExternalHooksManager, normalize_spec
from myagy.context.instructions import ProjectInstructions
from myagy.context.repomap import RepoMap
from myagy.tools.web_search import WebSearchEngine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def reset_instruction_cache():
    ProjectInstructions._cached_content = None
    ProjectInstructions._cached_files = []
    ProjectInstructions._cached_paths = []
    ProjectInstructions._cached_mtime = 0.0


def test_instructions_cache_hits_on_second_call(tmp_path, monkeypatch):
    (tmp_path / "GEMINI.md").write_text("rules")
    first, _ = ProjectInstructions.load_instructions(str(tmp_path))
    monkeypatch.setattr("builtins.open", lambda *a, **k: pytest.fail("cache miss re-read the file"))
    second, files = ProjectInstructions.load_instructions(str(tmp_path))
    assert second == first and files == ["GEMINI.md"]


def test_instructions_dedup_case_variants(tmp_path):
    (tmp_path / "GEMINI.md").write_text("rules")
    found = ProjectInstructions.find_instruction_files(str(tmp_path))
    assert len([f for f in found if f.lower().endswith("gemini.md")]) == 1
    text, _ = ProjectInstructions.load_instructions(str(tmp_path))
    assert text.count("rules") == 1


def _mcp_setup(monkeypatch, tmp_path, data):
    from myagy.context import mcp_loader
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps(data))
    monkeypatch.setattr(mcp_loader, "MCP_CONFIG_PATH", str(cfg))
    return mcp_loader


def test_mcp_bad_entry_does_not_drop_others(monkeypatch, tmp_path):
    loader = _mcp_setup(monkeypatch, tmp_path, {"mcpServers": {
        "bad": {"args": []},
        "good": {"command": "node", "args": ["x.js"]},
    }})
    names = [s.name for s in loader.load_mcp_servers(enable_kontexta=False)]
    assert names == ["good"]


def test_mcp_bad_entry_first_or_last(monkeypatch, tmp_path):
    loader = _mcp_setup(monkeypatch, tmp_path, {"mcpServers": {
        "good": {"command": "node"},
        "bad": "not-a-dict",
        "also_good": {"command": "node"},
    }})
    names = [s.name for s in loader.load_mcp_servers(enable_kontexta=False)]
    assert names == ["good", "also_good"]


def test_mcp_mcpservers_wins_over_vscode_servers(monkeypatch, tmp_path):
    loader = _mcp_setup(monkeypatch, tmp_path, {
        "mcpServers": {"dup": {"command": "primary"}},
        "servers": {"dup": {"command": "secondary"}, "extra": {"command": "e"}},
    })
    servers = loader.load_mcp_servers(enable_kontexta=False)
    assert [(s.name, s.command) for s in servers] == [("dup", "primary"), ("extra", "e")]


def test_normalize_spec_handles_odd_shapes():
    spec = normalize_spec({
        "PreToolUse": None,
        "PostToolUse": {"matcher": "x", "hooks": [{"command": "echo"}]},
        "Stop": "echo stop",
        "PreTurn": [{"hooks": "echo a"}, {"hooks": [{"command": ""}, 5]}, 7],
        "PostTurn": [{"matcher": 3, "hooks": [{"command": "c", "timeout": "bad"}]}],
        "enabled": True,
    })
    assert spec["PreToolUse"] == []
    assert spec["PostToolUse"] == [{"matcher": "x", "hooks": [{"command": "echo", "timeout": 30}]}]
    assert spec["Stop"] == [{"matcher": "*", "hooks": [{"command": "echo stop", "timeout": 30}]}]
    assert spec["PreTurn"] == [{"matcher": "*", "hooks": [{"command": "echo a", "timeout": 30}]}]
    assert spec["PostTurn"][0]["matcher"] == "*" and spec["PostTurn"][0]["hooks"][0]["timeout"] == 30
    assert spec["enabled"] is True


class _Sess:
    active_target = "t"
    json_output = True
    verbose = False
    conversation_id = "c"


def _hooks_file(tmp_path, monkeypatch, body):
    (tmp_path / ".agents").mkdir()
    (tmp_path / ".agents" / "hooks.json").write_text(json.dumps(body))
    monkeypatch.chdir(tmp_path)


def test_build_hooks_survives_malformed_spec(tmp_path, monkeypatch):
    _hooks_file(tmp_path, monkeypatch, {"h": {"PreToolUse": None, "PostToolUse": {"hooks": "echo hi"}}})
    built = ExternalHooksManager.build_lifecycle_hooks(_Sess())
    assert len(built) == 1


def test_pre_tool_non_string_decision_does_not_crash(tmp_path, monkeypatch):
    script = tmp_path / "h.sh"
    script.write_text('#!/bin/sh\ncat >/dev/null\necho \'{"decision": true}\'\n')
    script.chmod(0o755)
    _hooks_file(tmp_path, monkeypatch, {"h": {"PreToolUse": [{"hooks": [{"command": str(script)}]}]}})
    (decide,) = ExternalHooksManager.build_lifecycle_hooks(_Sess())
    from myagy.context.hooks_loader import types
    assert decide(types.ToolCall(name="grep", args={})).allow


def test_hook_timeout_kills_process_group(tmp_path):
    marker = tmp_path / "survivor"
    cmd = f"(sleep 2; touch {marker}) & wait"
    t0 = time.time()
    res = ExternalHooksManager.execute_command_hook(cmd, str(tmp_path), {}, timeout=0.5)
    assert "timed out" in res["error"] and time.time() - t0 < 1.9
    time.sleep(2.3)
    assert not marker.exists()


def test_hook_command_returns_json_and_raw(tmp_path):
    ok = ExternalHooksManager.execute_command_hook("cat >/dev/null; echo '{\"decision\":\"deny\"}'", str(tmp_path), {})
    assert ok == {"decision": "deny"}
    raw = ExternalHooksManager.execute_command_hook("echo plain", str(tmp_path), {})
    assert raw["raw_output"] == "plain" and raw["exit_code"] == 0


def test_repomap_cache_detects_rename(tmp_path):
    (tmp_path / "a.py").write_text("def alpha():\n    pass\n")
    first = RepoMap.get_cached_map(str(tmp_path))
    assert "a.py" in first
    os.rename(tmp_path / "a.py", tmp_path / "b.py")
    second = RepoMap.get_cached_map(str(tmp_path))
    assert "b.py" in second and "a.py" not in second


def test_ddg_snippets_align_with_titles_when_one_missing():
    body = (
        '<h2 class="result__title"><a href="//x?uddg=https%3A%2F%2Fone.com&rut=1">One</a></h2>'
        '<h2 class="result__title"><a href="//x?uddg=https%3A%2F%2Ftwo.com&rut=2">Two</a></h2>'
        '<a class="result__snippet" href="#">snippet two</a>'
    )
    res = WebSearchEngine._parse_ddg(body, 5)
    assert [(r["url"], r["snippet"]) for r in res] == [("https://one.com", ""), ("https://two.com", "snippet two")]


class _Resp:
    def __init__(self, body=b"", ctype="text/html"):
        self._body, self.headers = body, {"Content-Type": ctype}
        self.read_sizes = []

    def read(self, n=-1):
        self.read_sizes.append(n)
        return self._body if n < 0 else self._body[:n]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_search_falls_back_to_github_on_empty_ddg(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Resp(b"<html>captcha</html>"))
    monkeypatch.setattr(WebSearchEngine, "search_github", classmethod(lambda cls, q, max_results=5: [{"title": "gh", "url": "u", "snippet": "s"}]))
    assert WebSearchEngine.search("q")[0]["title"] == "gh"


def test_fetch_url_rejects_binary_and_caps_read(monkeypatch):
    binary = _Resp(b"%PDF", "application/pdf")
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: binary)
    assert "unsupported content type" in WebSearchEngine.fetch_url("example.com")
    assert binary.read_sizes == []

    page = _Resp(b"<p>hello</p>", "text/html; charset=utf-8")
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: page)
    assert "hello" in WebSearchEngine.fetch_url("example.com")
    assert page.read_sizes == [WebSearchEngine.MAX_READ_BYTES]


def test_config_env_overrides():
    code = (
        f"import sys; sys.path.insert(0, {os.path.dirname(ROOT)!r}); sys.modules['google'] = None\n"
        "from myagy import config\n"
        "print(config.DEFAULT_MODEL_DIR, config.KONTEXTA_MCP_PATH, config.KONTEXTA_DATA_DIR)\n"
    )
    env = {**os.environ, "MYAGY_MODEL_DIR": "/m", "MYAGY_KONTEXTA_MCP": "/k/index.js", "MYAGY_KONTEXTA_DATA": "/d"}
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, env=env, timeout=60)
    assert out.stdout.split() == ["/m", "/k/index.js", "/d"], out.stderr


def test_wrapper_import_is_lazy():
    code = (
        f"import sys; sys.path.insert(0, {os.path.join(ROOT, 'tests')!r}); import sdk_stub; sdk_stub.install()\n"
        f"sys.path.insert(0, {os.path.dirname(ROOT)!r})\n"
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('antigravity_agent', {os.path.join(ROOT, 'antigravity_agent.py')!r})\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "assert 'myagy.core.session' not in sys.modules\n"
        "assert m.PORT_ROCM == 9000\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
