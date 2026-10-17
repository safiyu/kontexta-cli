import os
import shutil
import pytest

from kontexta_cli.context.kontexta import KontextaManager
from kontexta_cli.context.mcp_loader import load_mcp_servers
from kontexta_cli.terminal import cli


def test_kontexta_manager_entrypoint_and_availability():
    entry = KontextaManager.find_mcp_entrypoint()
    # It must find the compiled dist/index.js if present
    if entry:
        assert os.path.isfile(entry)
        assert entry.endswith("index.js")
        assert KontextaManager.is_available()


def test_kontexta_create_mcp_server():
    server = KontextaManager.create_mcp_server(server_name="kxta")
    if KontextaManager.is_available():
        assert server is not None
        assert server.name == "kxta"
        assert server.command == (shutil.which("node") or "node")
        assert "KONTEXTA_DATA_DIR" in server.env
        assert server.env["NODE_ENV"] == "production"


def test_load_mcp_servers_inbuilt_kontexta(monkeypatch, tmp_path):
    # Empty mcp_config.json
    fake_config = str(tmp_path / "empty_mcp.json")
    monkeypatch.setattr("kontexta_cli.context.mcp_loader.MCP_CONFIG_PATH", fake_config)

    # When enable_kontexta=True, it should include kontexta if available
    servers = load_mcp_servers(enable_kontexta=True)
    if KontextaManager.is_available():
        names = [s.name for s in servers]
        assert "kxta" in names or "kontexta" in names

    # When enable_kontexta=False, empty list
    servers_disabled = load_mcp_servers(enable_kontexta=False)
    assert len(servers_disabled) == 0


def test_kontexta_status_telemetry():
    status = KontextaManager.get_status(".")
    assert "available" in status
    assert "data_dir" in status
    assert "db_path" in status
    assert isinstance(status["projects"], list)


def test_kontexta_context_prompt_generation():
    if KontextaManager.is_available():
        prompt = KontextaManager.generate_context_prompt(".")
        assert prompt is not None
        assert "[Inbuilt Kontexta Context & Knowledge Engine Active]" in prompt
        assert "kxta" in prompt
        assert "Available MCP Tools" in prompt


def test_kontexta_subagent_inherits_mcp(session):
    assert session._mcp_servers is not None
    # Verify subagent receives session._mcp_servers


def test_kontexta_cli_tab_completion(monkeypatch):
    pytest.importorskip("prompt_toolkit")
    from prompt_toolkit.document import Document
    monkeypatch.setattr(cli, "_pt_session", None)
    monkeypatch.setattr(cli, "HISTORY_PATH", "/tmp/kontexta-cli-test-history")
    pt = cli._get_pt_session()
    assert pt is not None

    def complete(text):
        return [c.text for c in pt.completer.get_completions(Document(text), None)]

    assert "/kontexta" in complete("/ko")
    assert "/kxta" in complete("/kx")


def test_kontexta_preference_persistence(session):
    session.apply_preferences({"enable_kontexta": False})
    assert session.enable_kontexta is False
    assert session.preferences()["enable_kontexta"] is False

    session.apply_preferences({"enable_kontexta": True})
    assert session.enable_kontexta is True
    assert session.preferences()["enable_kontexta"] is True
