import os
import sys
import tempfile
import pytest

from kontexta_cli.config import (
    setup_antigravity_sdk,
    _find_antigravity_sdk_dir,
    _find_localharness_bin,
    _find_antigravity_cli_dir,
    _find_oauth_token_path,
    _find_mcp_config_path,
)


def test_dynamic_sdk_discovery(tmp_path, monkeypatch):
    fake_sdk = tmp_path / "custom_sdk" / "google" / "antigravity"
    fake_sdk.mkdir(parents=True)
    (fake_sdk / "__init__.py").write_text("# fake sdk")

    # Override env var to custom path
    monkeypatch.setenv("ANTIGRAVITY_SDK_PATH", str(fake_sdk))
    found = _find_antigravity_sdk_dir()
    assert found == str(fake_sdk)

    # Calling setup_antigravity_sdk should register root in sys.path
    root_to_add = str(tmp_path / "custom_sdk")
    setup_antigravity_sdk(str(fake_sdk))
    assert root_to_add in sys.path


def test_dynamic_harness_binary_discovery(tmp_path, monkeypatch):
    fake_bin = tmp_path / "bin" / "localharness"
    fake_bin.parent.mkdir(parents=True)
    fake_bin.write_text("#!/bin/sh\nexit 0")
    fake_bin.chmod(0o755)

    monkeypatch.setenv("ANTIGRAVITY_HARNESS_PATH", str(fake_bin))
    found = _find_localharness_bin()
    assert found == str(fake_bin)


def test_dynamic_cli_dir_and_oauth_discovery(tmp_path, monkeypatch):
    fake_cli = tmp_path / "gemini_cli_custom"
    fake_cli.mkdir(parents=True)
    token_file = fake_cli / "antigravity-oauth-token"
    token_file.write_text("oauth_fake_secret_token")

    monkeypatch.setenv("ANTIGRAVITY_CLI_DIR", str(fake_cli))
    cli_dir = _find_antigravity_cli_dir()
    assert cli_dir == str(fake_cli)

    oauth_path = _find_oauth_token_path(cli_dir)
    assert oauth_path == str(token_file)


def test_dynamic_mcp_config_discovery(tmp_path, monkeypatch):
    fake_mcp = tmp_path / "custom_mcp.json"
    fake_mcp.write_text("{}")

    monkeypatch.setenv("ANTIGRAVITY_MCP_CONFIG_PATH", str(fake_mcp))
    found = _find_mcp_config_path()
    assert found == str(fake_mcp)
