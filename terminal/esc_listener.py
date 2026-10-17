"""Instant ESC-key interruption listener for terminal streaming and tool execution.

Similar to Claude Code and Antigravity CLI, this listener monitors stdin in cbreak
mode during model generation or tool execution. Tapping the ESC key immediately
aborts the active turn without needing Ctrl+C or triggering OS SIGINT signals.
Bulletproofed with loop.add_reader when an asyncio event loop is active, prompt_toolkit
KeyBindings, and guaranteed termios restoration on finally/atexit to protect the shell.
"""

import asyncio
import os
import re
import sys
import shutil
import select
import threading
import contextlib
from typing import Callable, Optional, Any

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")

try:
    import termios
    import tty
    TERMIOS_AVAILABLE = True
except ImportError:
    TERMIOS_AVAILABLE = False


# ── Global Terminal Restoration (Prevents broken bash shell on crash) ─────────
_GLOBAL_SAVED_SETTINGS = None
_GLOBAL_SAVED_FD = None


def _restore_global_terminal():
    """Emergency restore registered with atexit to guarantee echo is restored."""
    global _GLOBAL_SAVED_SETTINGS, _GLOBAL_SAVED_FD
    if _GLOBAL_SAVED_FD is not None and _GLOBAL_SAVED_SETTINGS is not None and TERMIOS_AVAILABLE:
        try:
            termios.tcflush(_GLOBAL_SAVED_FD, termios.TCIFLUSH)
            termios.tcsetattr(_GLOBAL_SAVED_FD, termios.TCSANOW, _GLOBAL_SAVED_SETTINGS)
        except Exception:
            pass


if TERMIOS_AVAILABLE and sys.stdin.isatty():
    try:
        import atexit
        _GLOBAL_SAVED_FD = sys.stdin.fileno()
        _GLOBAL_SAVED_SETTINGS = termios.tcgetattr(_GLOBAL_SAVED_FD)
        atexit.register(_restore_global_terminal)
    except Exception:
        pass


class EscListener:
    """
    Context manager that listens for the standalone ESC key (ASCII 27 / \\x1b).
    Uses asyncio loop.add_reader when an event loop is running, with graceful thread
    fallback and strict termios restoration in finally.

    Usage:
        listener = EscListener(on_escape=lambda: current_task.cancel())
        with listener:
            await current_task
        if listener.interrupted:
            print("Interrupted by user.")
    """

    def __init__(self, on_escape: Optional[Callable[[], None]] = None, session: Any = None):
        self.on_escape = on_escape
        self.session = session
        self.interrupted: bool = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._reader_added: bool = False
        self._old_settings = None
        self._cbreak_settings = None
        self._fd: Optional[int] = None
        self._io_lock = threading.Lock()
        self._paused = False
        # Queue-ahead typing: typed chars accumulate; drained after the turn ends.
        self.buffered_text: bytearray = bytearray()
        self.submit_pending: bool = False
        self._box_drawn: bool = False

    # Active listener, so prompts can pause it without a reference
    _current: Optional["EscListener"] = None

    @classmethod
    def clear_box(cls):
        """Clears the live prompt box if currently visible."""
        listener = cls._current
        if listener and getattr(listener, "_box_drawn", False) and sys.stdout.isatty():
            try:
                sys.stdout.write("\033[1A\r\033[J")
                sys.stdout.flush()
            except Exception:
                pass
            listener._box_drawn = False

    def _render_live_input(self):
        """Live echo for characters typed while tasks or tools are running."""
        if not sys.stdout.isatty() or self._paused or not sys.stdin.isatty():
            return
        try:
            from .ui import UI
            width = min(shutil.get_terminal_size((80, 24)).columns, 80)
            rule = f"{UI.DARK_GRAY}{'─' * width}{UI.RST}"
            text = bytes(self.buffered_text).decode("utf-8", errors="replace")

            target = getattr(self.session, "active_target", "9000") if self.session else "9000"
            target_col = UI.target_color(target) if UI._use_color else ""

            tag = f" {UI.AMBER}[queued]{UI.RST}" if self.submit_pending else ""
            line = f"{target_col}❯{UI.RST} {text}{tag}"

            if not self._box_drawn:
                sys.stdout.write(f"\n{rule}\n{line}\n{rule}\033[1A\r\033[{len(text) + 3}C")
                self._box_drawn = True
            else:
                sys.stdout.write(f"\r\033[K{line}\n{rule}\033[1A\r\033[{len(text) + 3}C")
            sys.stdout.flush()
        except Exception:
            pass

    def _process_input_byte(self, b: bytes) -> bool:
        """Processes one byte from stdin. Returns True if turn was interrupted."""
        if not b:
            return False

        if b == b"\x1b":
            # Check if subsequent bytes follow (e.g., arrow key sequences \x1b[A)
            r_trailing, _, _ = select.select([self._fd], [], [], 0.04)
            if r_trailing:
                try:
                    os.read(self._fd, 32)
                except Exception:
                    pass
                return False

            # Standalone ESC: abort active turn
            self.buffered_text = bytearray()
            self.submit_pending = False
            self.interrupted = True
            self._stop_event.set()
            if self._box_drawn:
                self.clear_box()
            if self.on_escape:
                try:
                    self.on_escape()
                except Exception:
                    pass
            return True

        elif b == b"\x03":
            # Ctrl+C
            self.buffered_text = bytearray()
            self.submit_pending = False
            self.interrupted = True
            self._stop_event.set()
            if self._box_drawn:
                self.clear_box()
            if self.on_escape:
                try:
                    self.on_escape()
                except Exception:
                    pass
            return True

        else:
            prev_len = len(self.buffered_text)
            self.submit_pending = _apply_buffer_byte(
                self.buffered_text, self.submit_pending, b[0]
            )
            if len(self.buffered_text) != prev_len or self.submit_pending:
                self._render_live_input()
            return False

    def _on_stdin_event(self):
        """Callback for asyncio loop.add_reader."""
        if self._paused or self.interrupted or self._stop_event.is_set() or self._fd is None:
            return
        try:
            b = os.read(self._fd, 1)
            interrupted = self._process_input_byte(b)
            if interrupted and self._loop and self._reader_added:
                try:
                    self._loop.remove_reader(self._fd)
                except Exception:
                    pass
                self._reader_added = False
        except Exception:
            pass

    def __enter__(self):
        self.interrupted = False
        self._stop_event.clear()
        self.buffered_text = bytearray()
        self.submit_pending = False
        self._box_drawn = False
        EscListener._current = self

        # Check if stdin is an interactive terminal and termios is available
        if not TERMIOS_AVAILABLE or not sys.stdin.isatty():
            return self

        try:
            self._fd = sys.stdin.fileno()
            self._old_settings = termios.tcgetattr(self._fd)

            # Configure terminal: cbreak mode + disable ECHO
            new_settings = termios.tcgetattr(self._fd)
            new_settings[3] = new_settings[3] & ~termios.ICANON
            new_settings[3] = new_settings[3] & ~termios.ECHO
            new_settings[6][termios.VMIN] = 1
            new_settings[6][termios.VTIME] = 0
            termios.tcsetattr(self._fd, termios.TCSANOW, new_settings)
            self._cbreak_settings = new_settings

            # Use asyncio reader if a running loop is detected on the thread
            loop = None
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                pass

            if loop and loop.is_running():
                self._loop = loop
                loop.add_reader(self._fd, self._on_stdin_event)
                self._reader_added = True
            else:
                self._thread = threading.Thread(target=self._listen_loop, daemon=True, name="EscListenerThread")
                self._thread.start()
        except Exception:
            self._cleanup_terminal()

        return self

    def _listen_loop(self):
        if self._fd is None:
            return

        while not self._stop_event.is_set():
            if self._paused:
                self._stop_event.wait(0.02)
                continue
            try:
                with self._io_lock:
                    if self._paused:
                        continue
                    r, _, _ = select.select([self._fd], [], [], 0.05)
                    if not r or self._stop_event.is_set():
                        continue
                    b = os.read(self._fd, 1)

                if self._process_input_byte(b):
                    break

            except Exception:
                break

    def pause(self):
        """Stops reading stdin and restores cooked mode so input() works normally."""
        if self._fd is None or self._old_settings is None or self._paused:
            return
        self._paused = True
        if self._loop and self._reader_added and self._fd is not None:
            try:
                self._loop.remove_reader(self._fd)
            except Exception:
                pass
            self._reader_added = False

        with self._io_lock:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
            except Exception:
                pass

    def resume(self):
        """Re-enters cbreak mode and restarts ESC detection after a prompt."""
        if not self._paused:
            return
        if self._fd is not None and self._cbreak_settings is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSANOW, self._cbreak_settings)
            except Exception:
                pass
            if self._loop and not self._reader_added and self._loop.is_running():
                try:
                    self._loop.add_reader(self._fd, self._on_stdin_event)
                    self._reader_added = True
                except Exception:
                    pass
        self._paused = False

    @classmethod
    @contextlib.contextmanager
    def suspended(cls):
        """Pauses the active listener (if any) for the duration of a prompt."""
        listener = cls._current
        if listener:
            listener.pause()
        try:
            yield
        finally:
            if listener:
                listener.resume()

    def _cleanup_terminal(self):
        if self._fd is not None and self._old_settings is not None:
            try:
                termios.tcflush(self._fd, termios.TCIFLUSH)
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
            except Exception:
                pass
            self._old_settings = None

    def drain(self) -> tuple:
        """Returns (text, submit_pending) and clears the queued-input buffer."""
        text = bytes(self.buffered_text).decode("utf-8", errors="replace")
        submit = self.submit_pending
        self.buffered_text = bytearray()
        self.submit_pending = False
        return text, submit

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self._stop_event.set()
            if self._box_drawn:
                self.clear_box()
            if self._loop and self._reader_added and self._fd is not None:
                try:
                    self._loop.remove_reader(self._fd)
                except Exception:
                    pass
                self._reader_added = False
            if self._thread and self._thread.is_alive() and self._thread is not threading.current_thread():
                self._thread.join(timeout=0.3)
        finally:
            self._cleanup_terminal()
            if EscListener._current is self:
                EscListener._current = None
            self._paused = False

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return self.__exit__(exc_type, exc_val, exc_tb)


def _apply_buffer_byte(buffer: bytearray, submit_pending: bool, b: int) -> bool:
    """Processes one input byte seen during a turn and returns the new submit flag.

    Mutates `buffer` in place. Once submit_pending is True, further input is ignored
    until the turn ends and the buffer is drained.
    """
    if submit_pending:
        return True
    if b in (0x0A, 0x0D):  # Enter (LF or CR)
        return True
    if b in (0x7F, 0x08):  # Backspace (DEL or BS)
        if buffer:
            buffer.pop()
        return False
    if b == 0x15:  # Ctrl-U clears the line
        buffer.clear()
        return False
    if 0x20 <= b <= 0x7E or b >= 0x80:  # printable ASCII + any high byte (UTF-8)
        buffer.append(b)
        return False
    return False  # ignore other control chars (Tab, Ctrl-A, etc.)


def prompt_input(prompt: str = "") -> str:
    """input() that is safe to call while an ESC listener is active."""
    try:
        from .ui import stop_status
        stop_status()
    except Exception:
        pass

    with EscListener.suspended():
        if prompt:
            if not sys.stdout.isatty():
                prompt = _ANSI_RE.sub("", prompt)
            sys.stdout.write(prompt)
            sys.stdout.flush()
            return input()
        return input()


@contextlib.asynccontextmanager
async def listen_for_esc(cancel_ev: asyncio.Event):
    """Context manager setting cancel_ev when Esc is pressed."""
    listener = EscListener(on_escape=lambda: cancel_ev.set())
    async with listener:
        yield listener

