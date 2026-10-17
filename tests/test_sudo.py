"""Unit tests for SudoManager, askpass helper, privilege elevation, and password entry."""

import os
import time
import subprocess
import types
from unittest.mock import patch, MagicMock

try:
    import pytest
except ImportError:
    pytest = None

from kontexta_cli.core.sudo_manager import SudoManager
from kontexta_cli.core.session import MultiGpuHybridSession


def test_is_sudo_command():
    assert SudoManager.is_sudo_command("sudo apt update") is True
    assert SudoManager.is_sudo_command("/usr/bin/sudo systemctl restart nginx") is True
    assert SudoManager.is_sudo_command("cd /var/log && sudo rm -f test.log") is True
    assert SudoManager.is_sudo_command("echo hello | sudo tee /etc/test") is True
    assert SudoManager.is_sudo_command("true; sudo make install") is True

    # Non-sudo commands
    assert SudoManager.is_sudo_command("ls -la") is False
    assert SudoManager.is_sudo_command("cat pseudo_code.py") is False
    assert SudoManager.is_sudo_command("git commit -m 'sudo feature'") is False
    assert SudoManager.is_sudo_command("") is False


def test_sudo_manager_lifecycle():
    SudoManager.cleanup()
    assert SudoManager.is_authenticated() is False
    assert SudoManager.time_remaining() == 0.0

    # Set password
    with patch("subprocess.run") as mock_sub:
        mock_sub.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        SudoManager.set_password("secret_pass_123")

    assert SudoManager.is_authenticated() is True
    assert SudoManager.time_remaining() > 0.0
    assert SudoManager._pass_file is not None
    assert os.path.exists(SudoManager._pass_file)

    # Check file permissions are restricted to owner only (0o600)
    mode = os.stat(SudoManager._pass_file).st_mode & 0o777
    assert mode == 0o600

    # Verify askpass helper outputs the secret
    askpass = SudoManager._askpass_file
    res = subprocess.run([askpass], capture_output=True, text=True)
    assert res.stdout.strip() == "secret_pass_123"

    # Clear credentials
    with patch("subprocess.run") as mock_sub:
        mock_sub.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        SudoManager.clear_credentials()

    assert SudoManager.is_authenticated() is False
    SudoManager.cleanup()


def test_verify_and_authenticate():
    SudoManager.cleanup()

    # Success case
    with patch("subprocess.run") as mock_sub:
        mock_sub.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout=b"", stderr=b"")
        ok, msg = SudoManager.verify_and_authenticate("valid_pass")
        assert ok is True
        assert SudoManager.is_authenticated() is True

    # Failure case
    with patch("subprocess.run") as mock_sub:
        mock_sub.return_value = subprocess.CompletedProcess(args=[], returncode=1, stdout=b"", stderr=b"Sorry, try again.")
        ok, msg = SudoManager.verify_and_authenticate("wrong_pass")
        assert ok is False

    SudoManager.cleanup()


def test_prompt_password_entry_success(monkeypatch):
    SudoManager.cleanup()
    monkeypatch.setattr("getpass.getpass", lambda prompt="": "my_root_pass")

    with patch.object(SudoManager, "verify_and_authenticate", return_value=(True, "Authenticated")):
        ok = SudoManager.prompt_password_entry(reason="sudo apt install htop")
        assert ok is True

    SudoManager.cleanup()


def test_prompt_password_entry_cancellation(monkeypatch):
    SudoManager.cleanup()
    monkeypatch.setattr("getpass.getpass", lambda prompt="": "")

    ok = SudoManager.prompt_password_entry(reason="sudo systemctl restart")
    assert ok is False
    SudoManager.cleanup()


def test_session_permission_handler_sudo_unauthenticated(session, monkeypatch):
    SudoManager.cleanup()
    tool = types.SimpleNamespace(name="run_command")
    args = {"CommandLine": "sudo apt-get update"}

    # User approves and enters valid password
    monkeypatch.setattr("kontexta_cli.core.session.prompt_input", lambda p="": "y")
    monkeypatch.setattr(SudoManager, "prompt_password_entry", lambda reason=None: True)

    allowed = session.permission_prompt_handler(tool, args)
    assert allowed is True

    SudoManager.cleanup()


def test_session_permission_handler_sudo_cached(session, monkeypatch):
    SudoManager.cleanup()
    with patch("subprocess.run"):
        SudoManager.set_password("cached_pass")

    tool = types.SimpleNamespace(name="run_command")
    args = {"CommandLine": "sudo systemctl restart"}

    # User confirms execution with Enter or 'y'
    monkeypatch.setattr("kontexta_cli.core.session.prompt_input", lambda p="": "y")

    allowed = session.permission_prompt_handler(tool, args)
    assert allowed is True

    SudoManager.cleanup()
