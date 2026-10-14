import os
import select
import signal
import sys
import time

import pytest

pty = pytest.importorskip("pty")

CHILD = r'''
import sys, time, importlib.util
spec = importlib.util.spec_from_file_location("esc", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with m.EscListener(on_escape=lambda: print("ESC!", flush=True)) as l:
    time.sleep(0.3)
    got = m.prompt_input("Authorize? [y/N]: ")
    print("\nINPUT_GOT=%r" % got, flush=True)
    time.sleep(0.5)
print("DONE interrupted=%s" % l.interrupted, flush=True)
'''


def _run_child(tmp_path, keystrokes, esc_after=True):
    script = tmp_path / "child.py"
    script.write_text(CHILD)
    listener_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "terminal", "esc_listener.py")
    pid, fd = pty.fork()
    if pid == 0:
        os.execvp(sys.executable, [sys.executable, str(script), listener_path])
    out = b""
    try:
        time.sleep(1.0)
        for ch in keystrokes:
            os.write(fd, bytes([ch]))
            time.sleep(0.05)
        if esc_after:
            time.sleep(0.1)
            os.write(fd, b"\x1b")
        end = time.time() + 5
        while time.time() < end and b"DONE" not in out:
            r, _, _ = select.select([fd], [], [], 0.2)
            if r:
                try:
                    out += os.read(fd, 4096)
                except OSError:
                    break
    finally:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
        os.waitpid(pid, 0)
    return out.decode(errors="replace")


def test_prompt_receives_typed_input_and_esc_still_works_after(tmp_path):
    out = _run_child(tmp_path, b"yes\n")
    assert "INPUT_GOT='yes'" in out
    assert "ESC!" in out and "interrupted=True" in out


def test_arrow_key_sequences_do_not_interrupt(tmp_path):
    script = tmp_path / "child2.py"
    script.write_text(CHILD.replace('got = m.prompt_input("Authorize? [y/N]: ")', 'time.sleep(1.5); got = None'))
    listener_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "terminal", "esc_listener.py")
    pid, fd = pty.fork()
    if pid == 0:
        os.execvp(sys.executable, [sys.executable, str(script), listener_path])
    out = b""
    try:
        time.sleep(1.0)
        os.write(fd, b"\x1b[A")  # up arrow: ESC followed immediately by bytes
        end = time.time() + 5
        while time.time() < end and b"DONE" not in out:
            r, _, _ = select.select([fd], [], [], 0.2)
            if r:
                try:
                    out += os.read(fd, 4096)
                except OSError:
                    break
    finally:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
        os.waitpid(pid, 0)
    assert "interrupted=False" in out.decode(errors="replace")


def test_prompt_input_strips_ansi_on_non_tty(monkeypatch, capsys):
    from myagy.terminal.esc_listener import prompt_input
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    monkeypatch.setattr("builtins.input", lambda: "ok")
    res = prompt_input("\033[1;38;5;214mAuthorize?\033[0m ")
    captured = capsys.readouterr()
    assert res == "ok"
    assert captured.out == "Authorize? "
    assert "\033" not in captured.out
    assert "[1;38;5;214m" not in captured.out


def test_prompt_input_preserves_ansi_on_tty(monkeypatch, capsys):
    from myagy.terminal.esc_listener import prompt_input
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda: "yes")
    res = prompt_input("\033[1;38;5;214mAuthorize?\033[0m ")
    captured = capsys.readouterr()
    assert res == "yes"
    assert captured.out == "\033[1;38;5;214mAuthorize?\033[0m "
