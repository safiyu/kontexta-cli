"""Pushes brainstorm and commit summaries to the Kontexta journal for future sessions."""

import asyncio
import os
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

from .kontexta import KontextaManager

PUSH_TIMEOUT = 25.0
MAX_ENTRY_CHARS = 4000
MAX_FILES_LISTED = 25

_COMMIT_RE = re.compile(r"\bgit\s+(?:-[^\s]+\s+(?:\S+\s+)?)*commit\b")
_SECRET_PATTERNS = [
    re.compile(r"(?i)\b(password|passwd|secret|token|api[_-]?key|authorization)\b(\s*[:=]\s*)(?:(?:bearer|basic)\s+)?\S+"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{12,}"),
    re.compile(r"\b(sk-[A-Za-z0-9_-]{16,}|AIza[0-9A-Za-z_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|xox[abprs]-[A-Za-z0-9-]{10,})\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
]

_pending: Set["asyncio.Task"] = set()
_synced_commits: Set[str] = set()


def redact(text: str) -> str:
    """Masks credentials before anything leaves the process."""
    for pat in _SECRET_PATTERNS:
        if pat.groups >= 2:
            text = pat.sub(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]", text)
        else:
            text = pat.sub("[REDACTED]", text)
    return text


def is_commit_call(call: Dict[str, Any]) -> bool:
    """True when a recorded tool call ran `git commit`."""
    return bool(_COMMIT_RE.search(str(call.get("args", ""))))


def enabled(session: Any, repo_root: str = ".") -> bool:
    """Sync runs only when opted in, Kontexta is installed, and the repo is a registered project."""
    if not getattr(session, "kontexta_sync", True):
        return False
    if not KontextaManager.is_available():
        return False
    try:
        return KontextaManager.find_project_for_dir(repo_root) is not None
    except Exception:
        return False


def _git(args: List[str], cwd: str) -> str:
    try:
        res = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=15)
        return res.stdout if res.returncode == 0 else ""
    except Exception:
        return ""


def commit_info(repo_root: str, rev: str = "HEAD") -> Optional[Dict[str, Any]]:
    """Hash, subject and per-file line counts of a commit, or None outside a repo."""
    head = _git(["log", "-1", "--format=%h%x00%s", rev], repo_root).strip()
    if "\x00" not in head:
        return None
    short, subject = head.split("\x00", 1)
    files = []
    for line in _git(["show", "--numstat", "--format=", rev], repo_root).splitlines():
        parts = line.split("\t")
        if len(parts) == 3:
            files.append({"path": parts[2], "added": parts[0], "removed": parts[1]})
    return {"hash": short, "subject": subject, "files": files}


def build_entry(kind: str, title: str, repo_root: str, *, commit: Optional[Dict[str, Any]] = None,
                outcome: str = "", spec_path: str = "", notes: Optional[List[str]] = None) -> Tuple[str, List[str]]:
    """Builds the journal text and tags for a brainstorm or commit event."""
    project = os.path.basename(os.path.abspath(repo_root))
    lines = [f"[kontexta-cli {kind}] {title}", f"Project: {project}"]
    if outcome:
        lines.append(f"Outcome: {outcome}")
    if spec_path:
        lines.append(f"Spec: {spec_path}")
    if commit:
        lines.append(f"Commit: {commit['hash']} {commit['subject']}")
        files = commit.get("files", [])
        for f in files[:MAX_FILES_LISTED]:
            lines.append(f"  {f['path']} (+{f['added']} -{f['removed']})")
        if len(files) > MAX_FILES_LISTED:
            lines.append(f"  +{len(files) - MAX_FILES_LISTED} more files")
    for note in notes or []:
        lines.append(note)
    text = redact("\n".join(lines))
    if len(text) > MAX_ENTRY_CHARS:
        text = text[:MAX_ENTRY_CHARS] + "\n[truncated]"
    return text, ["kontexta-cli", kind, project]


async def push(text: str, tags: List[str], repo_root: str = ".", timeout: float = PUSH_TIMEOUT) -> bool:
    """Writes one journal entry and one tagged note via the Kontexta MCP server; never raises."""
    entry = KontextaManager.find_mcp_entrypoint()
    node = shutil.which("node")
    if not entry or not node:
        return False
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError:
        return False

    async def _run() -> bool:
        params = StdioServerParameters(
            command=node, args=[entry], cwd=os.path.abspath(repo_root),
            env={**os.environ, "KONTEXTA_DATA_DIR": KontextaManager.get_data_dir(), "NODE_ENV": "production"},
        )
        async with stdio_client(params, errlog=open(os.devnull, "w")) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                first = await client.call_tool("journal_write", {"kind": "append", "text": text})
                await client.call_tool("journal_write", {"kind": "note", "text": text.splitlines()[0], "tags": tags})
                return not first.is_error

    try:
        return await asyncio.wait_for(_run(), timeout)
    except Exception:
        return False


def schedule(coro_factory) -> None:
    """Runs a push in the background; a failure never reaches the user's turn."""
    try:
        task = asyncio.get_running_loop().create_task(coro_factory())
    except RuntimeError:
        return
    _pending.add(task)
    task.add_done_callback(_pending.discard)


async def drain(timeout: float = 8.0) -> None:
    """Gives in-flight pushes a short window to finish before the session exits."""
    tasks = list(_pending)
    if tasks:
        await asyncio.wait(tasks, timeout=timeout)


def sync_event(session: Any, kind: str, title: str, repo_root: str = ".", *, commit: Optional[Dict[str, Any]] = None,
               outcome: str = "", spec_path: str = "", notes: Optional[List[str]] = None, quiet: bool = False) -> bool:
    """Queues a Kontexta push if enabled; returns whether one was queued."""
    if commit:
        if commit["hash"] in _synced_commits:
            return False
        _synced_commits.add(commit["hash"])
    if not enabled(session, repo_root):
        return False
    text, tags = build_entry(kind, title, repo_root, commit=commit, outcome=outcome, spec_path=spec_path, notes=notes)

    async def _go():
        ok = await push(text, tags, repo_root)
        if not ok and not quiet and not getattr(session, "json_output", False):
            print("[kontexta] journal sync skipped (server unavailable)", file=sys.stderr)

    schedule(_go)
    return True
