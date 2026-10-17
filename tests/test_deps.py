import os
import types

import pytest

from kontexta_cli import deps, config


class FakeRun:
    def __init__(self, returncodes=None, stdout="22"):
        self.calls = []
        self.returncodes = list(returncodes or [])
        self.stdout = stdout

    def __call__(self, cmd, **kw):
        self.calls.append((cmd, kw))
        code = self.returncodes.pop(0) if self.returncodes else 0
        return types.SimpleNamespace(returncode=code, stdout=self.stdout)


def make_env(run=None, which=None, answers=()):
    answers = list(answers)
    out = []
    env = deps.Env(
        run=run or FakeRun(),
        which=which or (lambda name: f"/usr/bin/{name}"),
        ask=lambda prompt: answers.pop(0) if answers else "y",
        out=out.append,
    )
    return env, out


@pytest.fixture
def all_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PROJECTS_DIR", str(tmp_path))
    monkeypatch.setattr(deps, "check_antigravity", lambda: (False, "missing"))
    monkeypatch.setattr(deps, "check_kontexta", lambda: (False, "missing"))
    monkeypatch.setattr(deps, "check_llamashift", lambda: (False, "missing"))
    monkeypatch.setattr(deps, "check_laya", lambda: (False, "missing"))
    for d in deps.DEPENDENCIES:
        monkeypatch.setattr(d, "check", getattr(deps, "check_" + d.name))


def test_install_antigravity_uses_pinned_pip_requirement():
    run = FakeRun()
    env, _ = make_env(run)
    assert deps.install_antigravity(env) is True
    cmd = run.calls[0][0]
    assert cmd[1:4] == ["-m", "pip", "install"] and cmd[-1] == deps.ANTIGRAVITY_REQUIREMENT


def test_install_kontexta_needs_recent_node():
    env, out = make_env(FakeRun(stdout="18"))
    assert deps.install_kontexta(env) is False
    assert "Node 22+" in out[0]


def test_install_kontexta_runs_global_npm_install():
    run = FakeRun(stdout="22")
    env, _ = make_env(run)
    assert deps.install_kontexta(env) is True
    assert run.calls[-1][0] == ["/usr/bin/npm", "install", "-g", deps.KONTEXTA_NPM_PACKAGE]


def test_install_llamashift_uses_pip_spec():
    run = FakeRun()
    env, out = make_env(run)
    assert deps.install_llamashift(env) is True
    cmd = run.calls[0][0]
    assert cmd[1:5] == ["-m", "pip", "install", "--upgrade"] and cmd[-1] == deps.LLAMASHIFT_PIP_SPEC
    assert any("llamashift" in line for line in out)


def test_install_llamashift_failure_is_reported():
    env, _ = make_env(FakeRun(returncodes=[1]))
    assert deps.install_llamashift(env) is False


def _serve_llamashift(monkeypatch, handler_body):
    import http.server, threading

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            code, body = handler_body(self.path)
            self.send_response(code)
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(config, "DEFAULT_LLAMASHIFT_URL", f"http://127.0.0.1:{srv.server_address[1]}")
    return srv


def test_check_llamashift_accepts_supported_api(monkeypatch):
    srv = _serve_llamashift(monkeypatch, lambda p: (200, '{"version": "1.0.0", "apiVersion": 1}'))
    try:
        ok, detail = deps.check_llamashift()
        assert ok and "API v1" in detail
    finally:
        srv.shutdown()


def test_check_llamashift_rejects_unsupported_api(monkeypatch):
    srv = _serve_llamashift(monkeypatch, lambda p: (200, '{"version": "9.0.0", "apiVersion": 9}'))
    try:
        ok, detail = deps.check_llamashift()
        assert not ok and "supports v1 to v1" in detail
    finally:
        srv.shutdown()


def test_check_llamashift_flags_old_server_without_version_api(monkeypatch):
    srv = _serve_llamashift(monkeypatch, lambda p: (404, "{}"))
    try:
        ok, detail = deps.check_llamashift()
        assert not ok and "predates the version API" in detail
    finally:
        srv.shutdown()


def test_check_llamashift_not_running(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DEFAULT_LLAMASHIFT_URL", "http://127.0.0.1:9")
    monkeypatch.setattr(config, "PROJECTS_DIR", str(tmp_path))
    monkeypatch.setattr(deps.shutil, "which", lambda n: None)
    ok, detail = deps.check_llamashift()
    assert not ok and "not installed" in detail
    monkeypatch.setattr(deps.shutil, "which", lambda n: "/x/llamashift" if n == "llamashift" else None)
    ok, detail = deps.check_llamashift()
    assert not ok and "start it with `llamashift`" in detail


def test_setup_declined_required_dependency_fails(all_missing):
    env, out = make_env(answers=["n", "n", "n"])
    assert deps.setup(env=env) == 1


def test_setup_yes_installs_and_reports(all_missing, monkeypatch):
    state = {"installed": False}
    ag = next(d for d in deps.DEPENDENCIES if d.name == "antigravity")
    monkeypatch.setattr(ag, "check", lambda: (state["installed"], "ok" if state["installed"] else "missing"))
    monkeypatch.setattr(ag, "install", lambda env: state.update(installed=True) or True)
    env, out = make_env()
    code = deps.setup(only=["antigravity"], assume_yes=True, env=env)
    assert code == 0 and any("[done] antigravity" in line for line in out)


def test_setup_failed_required_install_returns_nonzero(all_missing, monkeypatch):
    ag = next(d for d in deps.DEPENDENCIES if d.name == "antigravity")
    monkeypatch.setattr(ag, "install", lambda env: False)
    env, out = make_env()
    assert deps.setup(only=["antigravity"], assume_yes=True, env=env) == 1
    assert any("[FAIL] antigravity" in line for line in out)


def test_laya_has_no_installer_and_is_not_required():
    laya = next(d for d in deps.DEPENDENCIES if d.name == "laya")
    assert laya.install is None and laya.required is False


def test_setup_unknown_name(all_missing):
    env, out = make_env()
    assert deps.setup(only=["nope"], env=env) == 2


def test_doctor_exit_code_tracks_required_only(all_missing, monkeypatch):
    lines = []
    assert deps.doctor(out=lines.append) == 1
    ag = next(d for d in deps.DEPENDENCIES if d.name == "antigravity")
    monkeypatch.setattr(ag, "check", lambda: (True, "ok"))
    assert deps.doctor(out=lines.append) == 0


def test_kontexta_found_via_npm_bin(monkeypatch, tmp_path):
    real = tmp_path / "lib" / "dist" / "index.js"
    real.parent.mkdir(parents=True)
    real.write_text("")
    link = tmp_path / "bin" / "kxta"
    link.parent.mkdir()
    os.symlink(real, link)
    monkeypatch.delenv("KXTC_KONTEXTA_MCP", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(config, "PROJECTS_DIR", str(tmp_path / "none"))
    monkeypatch.setattr(config.shutil, "which", lambda n: str(link) if n == "kxta" else None)
    assert config.find_kontexta_mcp_path() == str(real.resolve())
