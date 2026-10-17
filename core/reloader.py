# Projects/myagy/core/reloader.py
from __future__ import annotations

import json
import os
import sys
import tempfile
import termios
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class ReloadSnapshot:
    version: int = 1
    timestamp: float = field(default_factory=time.time)
    cwd: str = field(default_factory=os.getcwd)

    # Core Routing & Mode State
    active_target: str = "rocm"
    active_models: dict[str, str] = field(default_factory=dict)
    auto_route: bool = True
    permission_mode: str = "auto"
    markdown_enabled: bool = True
    repomap_enabled: bool = True
    curation_enabled: bool = True
    kontexta_enabled: bool = True

    # Conversation & Working Memory
    chat_history: list[dict[str, Any]] = field(default_factory=list)
    compacted_working_memory: str = ""
    frozen_epochs: list[dict[str, Any]] = field(default_factory=list)

    # Checkpoints (/diff, /undo) & Session Usage Ledger (/stats)
    checkpoint_trees: list[dict[str, Any]] = field(default_factory=list)
    stats_ledger: dict[str, Any] = field(default_factory=dict)

    # Pending FIFO Subagent Queue (/tasks)
    pending_subagent_tasks: list[dict[str, Any]] = field(default_factory=list)


class ReloadRequested(Exception):
    """Signal raised by /reload to unwind the REPL loop cleanly before os.execv."""

    def __init__(self, shm_path: Path) -> None:
        super().__init__(str(shm_path))
        self.shm_path = Path(shm_path)


def _get_shm_dir() -> Path:
    """Prefers Linux RAM disk (/dev/shm) for sub-millisecond I/O; falls back to /tmp."""
    shm = Path("/dev/shm")
    if shm.is_dir() and os.access(shm, os.W_OK):
        return shm
    return Path(tempfile.gettempdir())


def write_reload_snapshot(snapshot: ReloadSnapshot) -> Path:
    """Writes session state to /dev/shm with strict 0600 permissions."""
    shm_dir = _get_shm_dir()
    shm_path = shm_dir / f"myagy_reload_{os.getpid()}.json"

    payload = json.dumps(asdict(snapshot), ensure_ascii=False).encode("utf-8")

    # Open with O_CREAT | O_WRONLY | O_TRUNC and 0600 permissions (owner read/write only)
    fd = os.open(
        str(shm_path),
        os.O_CREAT | os.O_WRONLY | os.O_TRUNC,
        0o600,
    )
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)

    return shm_path


def consume_reload_snapshot(shm_path_str: str) -> ReloadSnapshot | None:
    """Reads and immediately unlinks the /dev/shm snapshot file on startup."""
    shm_path = Path(shm_path_str)
    if not shm_path.exists():
        return None

    try:
        raw = shm_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        return ReloadSnapshot(**data)
    except Exception:
        return None
    finally:
        # Always unlink immediately so sensitive context never lingers in /dev/shm
        try:
            shm_path.unlink(missing_ok=True)
        except OSError:
            pass


def _restore_tty_sane() -> None:
    """Ensures stdin echo and canonical mode are restored before execv."""
    if not sys.stdin.isatty():
        return
    try:
        fd = sys.stdin.fileno()
        attrs = termios.tcgetattr(fd)
        # Re-enable canonical input (ICANON) and terminal echo (ECHO)
        attrs[3] |= termios.ICANON | termios.ECHO
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
    except Exception:
        pass


def perform_execv_reload(shm_path: Path) -> None:
    """Flushes stdio, restores TTY, and replaces current process via os.execv."""
    sys.stdout.flush()
    sys.stderr.flush()
    _restore_tty_sane()

    # Strip any previous --restore-shm flags from sys.argv to keep argv clean
    clean_argv: list[str] = []
    skip_next = False
    for arg in sys.argv:
        if skip_next:
            skip_next = False
            continue
        if arg == "--restore-shm":
            skip_next = True
            continue
        if arg.startswith("--restore-shm="):
            continue
        clean_argv.append(arg)

    new_argv = [
        sys.executable,
        *clean_argv,
        "--restore-shm",
        str(shm_path),
    ]

    # Replace current process image in-place (same PID, zero leaked Python objects)
    os.execv(sys.executable, new_argv)
