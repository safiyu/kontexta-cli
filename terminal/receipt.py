"""Compact per-turn receipt: what actually changed, ran and took how long."""

import sys
from typing import Dict, List, Optional

from .say import fmt_elapsed as _fmt_time, rule_line
from .ui import UI

MAX_COMMANDS = 5


def files_from_numstat(text: str) -> List[Dict]:
    """Parses `git diff --numstat` output; binary files report 0/0."""
    files = []
    for line in (text or "").splitlines():
        parts = line.split("\t", 2)
        if len(parts) != 3:
            continue
        added, removed, path = parts
        files.append({
            "path": path.strip(),
            "added": int(added) if added.isdigit() else 0,
            "removed": int(removed) if removed.isdigit() else 0,
        })
    return files


def _one_line(cmd: str, limit: int = 80) -> str:
    flat = " ".join(str(cmd).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "..."


def build_receipt(files: List[Dict], commands: List[str], tests: Optional[str],
                  elapsed: float, tok_s: Optional[float], target: str) -> Optional[str]:
    """Returns the receipt text, or None when the turn changed and ran nothing."""
    files, commands = files or [], commands or []
    if not files and not commands:
        return None

    d, r = UI.DARK_GRAY, UI.RST
    lines = []
    if files:
        total_add = sum(f.get("added", 0) for f in files)
        total_rm = sum(f.get("removed", 0) for f in files)
        noun = "file" if len(files) == 1 else "files"
        lines.append(f"{UI.GRAY}changed {len(files)} {noun} {UI.GREEN}+{total_add}{r} {UI.RED}-{total_rm}{r}")
        for f in files:
            lines.append(f"{d}  {r}{UI.GRAY}{f['path']}{r} {UI.GREEN}+{f.get('added', 0)}{r} {UI.RED}-{f.get('removed', 0)}{r}")
    if commands:
        lines.append(f"{UI.GRAY}ran {len(commands)} command{'s' if len(commands) != 1 else ''}{r}")
        for c in commands[:MAX_COMMANDS]:
            lines.append(f"{d}  $ {_one_line(c)}{r}")
        if len(commands) > MAX_COMMANDS:
            lines.append(f"{d}  +{len(commands) - MAX_COMMANDS} more{r}")
    if tests:
        ok = str(tests).strip().lower() in ("pass", "passed", "ok")
        mark = f"{UI.GREEN}pass{r}" if ok else f"{UI.RED}{tests}{r}"
        lines.append(f"{UI.GRAY}tests {mark}")

    foot = [str(target), _fmt_time(elapsed)]
    if tok_s:
        foot.append(f"{tok_s:.1f} tok/s")
    lines.append(f"{d}{' · '.join(foot)}{r}")

    head = rule_line("receipt", d, 40)
    return "\n".join([head, *lines])


def render_receipt(files: List[Dict], commands: List[str], tests: Optional[str],
                   elapsed: float, tok_s: Optional[float], target: str, file=None) -> bool:
    """Prints the receipt; returns False when there was nothing to show."""
    text = build_receipt(files, commands, tests, elapsed, tok_s, target)
    if text is None:
        return False
    print(text, file=file or sys.stdout, flush=True)
    return True
