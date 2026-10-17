"""
Sudo Privilege Elevation & Askpass Management:
Enables seamless execution of administrative/root commands inside Multi-GPU sessions,
subagents, and direct terminal executions with interactive password entry and secure caching.
"""

import os
import time
import atexit
import shutil
import getpass
import tempfile
import threading
import subprocess
from typing import Optional, Tuple

from ..terminal.ui import UI
from ..terminal import say
from . import shellparse

try:
    import termios
    _TTY_ERRORS = (OSError, termios.error)
except ImportError:
    _TTY_ERRORS = (OSError,)


class SudoManager:
    """
    Manages session-level sudo privilege elevation, secure askpass helpers,
    and interactive password verification for autonomous agent tool execution.
    """

    _password: Optional[str] = None
    _authenticated_at: Optional[float] = None
    _timeout_seconds: float = 900.0  # 15 minutes default timeout
    _runtime_dir: Optional[str] = None
    _pass_file: Optional[str] = None
    _askpass_file: Optional[str] = None
    _shim_file: Optional[str] = None
    _initialized: bool = False
    _atexit_registered: bool = False
    _expiry_timer: Optional[threading.Timer] = None
    _EXPIRY_GRACE: float = 1.0

    @classmethod
    def _ensure_initialized(cls):
        """Prepares private runtime directory, askpass script, and sudo shim."""
        if cls._initialized and cls._runtime_dir and os.path.exists(cls._runtime_dir):
            return

        runtime_dir = tempfile.mkdtemp(prefix="kontexta_cli_sudo_")
        os.chmod(runtime_dir, 0o700)
        cls._runtime_dir = runtime_dir

        cls._pass_file = os.path.join(runtime_dir, ".sudo_secret")
        # Touch file with restrictive permissions
        with open(cls._pass_file, "w", encoding="utf-8") as f:
            f.write("")
        os.chmod(cls._pass_file, 0o600)

        # 1. Askpass script: reads from the secure token file
        cls._askpass_file = os.path.join(runtime_dir, "kontexta_cli_askpass.sh")
        with open(cls._askpass_file, "w", encoding="utf-8") as f:
            f.write(
                "#!/bin/sh\n"
                f'if [ -f "{cls._pass_file}" ]; then\n'
                f'    cat "{cls._pass_file}"\n'
                "fi\n"
            )
        os.chmod(cls._askpass_file, 0o700)

        # 2. Transparent Sudo shim: intercepts sudo calls and routes to -A with SUDO_ASKPASS
        cls._shim_file = os.path.join(runtime_dir, "sudo")
        with open(cls._shim_file, "w", encoding="utf-8") as f:
            f.write(
                "#!/bin/bash\n"
                'REAL_SUDO="/usr/bin/sudo"\n'
                'if [ -n "$SUDO_ASKPASS" ] && [ -x "$SUDO_ASKPASS" ]; then\n'
                '    exec "$REAL_SUDO" -A "$@"\n'
                "else\n"
                '    exec "$REAL_SUDO" "$@"\n'
                "fi\n"
            )
        os.chmod(cls._shim_file, 0o700)

        # Export into process environment so children inherit seamlessly
        cur_path = os.environ.get("PATH", "")
        if not cur_path.startswith(runtime_dir):
            os.environ["PATH"] = f"{runtime_dir}:{cur_path}"
        os.environ["SUDO_ASKPASS"] = cls._askpass_file

        if not cls._atexit_registered:
            atexit.register(cls.cleanup)
            cls._atexit_registered = True
        cls._initialized = True

    @classmethod
    def cleanup(cls):
        """Wipes the cached password file and resets system sudo credentials."""
        cls._cancel_expiry_timer()
        try:
            if cls._pass_file and os.path.exists(cls._pass_file):
                with open(cls._pass_file, "w", encoding="utf-8") as f:
                    f.write("\x00" * 64)
                os.remove(cls._pass_file)
        except Exception:
            pass

        cls._password = None
        cls._authenticated_at = None

        try:
            if cls._runtime_dir and os.path.exists(cls._runtime_dir):
                shutil.rmtree(cls._runtime_dir, ignore_errors=True)
        except Exception:
            pass

        cls._initialized = False

        # Reset host kernel sudo timestamp cache
        try:
            subprocess.run(["/usr/bin/sudo", "-k"], capture_output=True, timeout=2)
        except Exception:
            pass

    @classmethod
    def is_authenticated(cls) -> bool:
        """Returns True if a valid sudo password is currently cached and within timeout."""
        if not cls._password or cls._authenticated_at is None:
            return False
        if (time.time() - cls._authenticated_at) > cls._timeout_seconds:
            cls.clear_credentials()
            return False
        return True

    @classmethod
    def time_remaining(cls) -> float:
        """Returns remaining seconds for cached sudo credentials, or 0.0 if expired."""
        if not cls.is_authenticated():
            return 0.0
        elapsed = time.time() - cls._authenticated_at
        return max(0.0, cls._timeout_seconds - elapsed)

    @classmethod
    def set_password(cls, password: str):
        """Stores password in volatile memory and the private askpass file."""
        cls._ensure_initialized()
        cleaned_pass = password.strip("\r\n")
        cls._password = cleaned_pass
        cls._authenticated_at = time.time()

        if cls._pass_file:
            with open(cls._pass_file, "w", encoding="utf-8") as f:
                f.write(cleaned_pass + "\n")
            os.chmod(cls._pass_file, 0o600)

        cls._start_expiry_timer()

    @classmethod
    def _cancel_expiry_timer(cls):
        timer, cls._expiry_timer = cls._expiry_timer, None
        if timer is not None:
            timer.cancel()

    @classmethod
    def _start_expiry_timer(cls):
        cls._cancel_expiry_timer()
        # is_authenticated wipes the secret on expiry, so the file does not outlive the timeout
        timer = threading.Timer(cls._timeout_seconds + cls._EXPIRY_GRACE, cls.is_authenticated)
        timer.daemon = True
        cls._expiry_timer = timer
        timer.start()

    @classmethod
    def clear_credentials(cls):
        """Clears cached credentials and resets sudo state."""
        cls._cancel_expiry_timer()
        cls._password = None
        cls._authenticated_at = None
        if cls._pass_file and os.path.exists(cls._pass_file):
            try:
                with open(cls._pass_file, "w", encoding="utf-8") as f:
                    f.write("\x00" * 64)
                with open(cls._pass_file, "w", encoding="utf-8") as f:
                    f.write("")
            except Exception:
                pass
        try:
            subprocess.run(["/usr/bin/sudo", "-k"], capture_output=True, timeout=2)
        except Exception:
            pass

    @classmethod
    def verify_and_authenticate(cls, password: str) -> Tuple[bool, str]:
        """Validates password against sudo -S -v and caches upon success."""
        cls._ensure_initialized()
        cleaned_pass = password.strip("\r\n")
        if not cleaned_pass:
            return False, "Password cannot be empty."

        try:
            proc = subprocess.run(
                ["/usr/bin/sudo", "-S", "-v"],
                input=(cleaned_pass + "\n").encode("utf-8"),
                capture_output=True,
                timeout=8,
            )
            if proc.returncode == 0:
                cls.set_password(cleaned_pass)
                return True, "Sudo access authenticated successfully."
            err = proc.stderr.decode("utf-8", errors="replace").strip()
            return False, err or "Incorrect sudo password."
        except subprocess.TimeoutExpired:
            return False, "Sudo authentication check timed out."
        except Exception as ex:
            return False, f"Sudo verification error: {ex}"

    @classmethod
    def prompt_password_entry(cls, reason: Optional[str] = None, max_attempts: int = 3) -> bool:
        """
        Interactive password entry section with masked input, validation,
        and clean visual feedback.
        """
        user = os.environ.get("USER", "root")
        rows = [("user", user)]
        if reason:
            short_reason = reason.strip().replace("\r\n", " ").replace("\n", " ")
            term_cols = shutil.get_terminal_size((100, 24)).columns
            max_reason = max(80, term_cols - 25)
            if len(short_reason) > max_reason:
                short_reason = short_reason[:max_reason - 3] + "..."
            rows.append(("command", short_reason))
        rows.append(("reason", "this command needs root privileges"))
        rows.append(("cache", "15 minutes (revoke with /sudo clear)"))
        say.card("sudo required", rows, actions="password prompt follows", style="danger")

        for attempt in range(max_attempts):
            try:
                prompt_label = f"{UI.AMBER_BOLD}[sudo] password for {user}: {UI.RST}"
                # Attempt getpass for secure hidden entry
                try:
                    pw = getpass.getpass(prompt_label)
                except _TTY_ERRORS:
                    from ..terminal.esc_listener import prompt_input
                    pw = prompt_input(prompt_label)
            except (KeyboardInterrupt, EOFError):
                say.err("Password entry cancelled.")
                return False

            if not pw or not pw.strip():
                say.err("Empty password provided. Entry aborted.")
                return False

            ok, msg = cls.verify_and_authenticate(pw)
            if ok:
                say.ok("Sudo access authenticated and unlocked for 15 minutes.")
                return True
            else:
                rem = max_attempts - attempt - 1
                if rem > 0:
                    say.err(f"{msg} ({rem} attempt(s) remaining)")
                else:
                    say.err("Sudo authentication failed: maximum attempts exceeded.")

        return False

    @classmethod
    def is_sudo_command(cls, cmd: str) -> bool:
        """Determines if a command line string requires root / sudo elevation."""
        return shellparse.elevates(cmd)
