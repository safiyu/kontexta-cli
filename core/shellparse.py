"""Dependency-free shell command inspection used by the sudo gate and Laya's offline safety fallback."""

import os
import re
import shlex
from typing import List

ELEVATORS = {"sudo", "su", "doas", "pkexec"}
SHELLS = {"sh", "bash", "zsh", "dash", "ksh", "fish"}
WRAPPERS = {
    "env", "xargs", "nohup", "nice", "ionice", "time", "timeout", "stdbuf", "setsid",
    "command", "exec", "watch", "find", "parallel", "builtin", "strace",
}
KEYWORDS = {"if", "then", "else", "elif", "do", "while", "until", "!", "{", "}"}
READ_ONLY_COMMANDS = {
    "ls", "cat", "head", "tail", "pwd", "echo", "printf", "grep", "egrep", "fgrep", "rg", "wc",
    "which", "whoami", "date", "tree", "stat", "file", "sort", "uniq", "diff", "du", "df",
    "basename", "dirname", "realpath", "readlink", "uname", "id", "hostname", "true", "false",
    "test", "[", "cut", "tr", "nl", "cd",
}
READ_ONLY_GIT = {"status", "diff", "log", "show", "rev-parse", "ls-files", "blame"}
UNSAFE_FIND_FLAGS = {"-exec", "-execdir", "-ok", "-okdir", "-delete", "-fprint", "-fprintf", "-fls"}

_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_MAX_DEPTH = 5


def split_segments(cmd: str) -> List[str]:
    """Splits a command line into simple commands at ; & | newline ( ) backtick and $( outside single quotes."""
    segs: List[str] = []
    buf: List[str] = []
    quote = None
    i, n = 0, len(cmd)

    def flush():
        if buf:
            segs.append("".join(buf))
            buf.clear()

    while i < n:
        ch = cmd[i]
        if quote == "'":
            buf.append(ch)
            if ch == "'":
                quote = None
        elif ch == "\\" and i + 1 < n:
            buf.append(ch)
            buf.append(cmd[i + 1])
            i += 1
        elif quote == '"':
            if ch == '"':
                quote = None
                buf.append(ch)
            elif ch == "`" or cmd.startswith("$(", i):
                flush()
                if ch == "$":
                    i += 1
            else:
                buf.append(ch)
        elif ch in "'\"":
            quote = ch
            buf.append(ch)
        elif ch in ";&|\n()`":
            flush()
        else:
            buf.append(ch)
        i += 1
    flush()
    return [s for s in segs if s.strip()]


def _tokens(segment: str) -> List[str]:
    try:
        return shlex.split(segment, posix=True)
    except ValueError:
        return segment.split()


def _segment_elevates(tokens: List[str], depth: int) -> bool:
    i = 0
    while i < len(tokens) and (tokens[i] in KEYWORDS or _ASSIGN.match(tokens[i])):
        i += 1
    if i >= len(tokens):
        return False
    base = os.path.basename(tokens[i])
    rest = tokens[i + 1:]
    if base in ELEVATORS:
        return True
    if depth >= _MAX_DEPTH:
        return False
    if base in SHELLS:
        for j, t in enumerate(rest):
            if t.startswith("-") and not t.startswith("--") and "c" in t[1:] and j + 1 < len(rest):
                return elevates(rest[j + 1], depth + 1)
        return False
    if base == "eval":
        return elevates(" ".join(rest), depth + 1)
    if base in WRAPPERS:
        for k, t in enumerate(rest):
            tb = os.path.basename(t)
            if tb in ELEVATORS:
                return True
            if tb in SHELLS or tb == "eval":
                return _segment_elevates(rest[k:], depth + 1)
    return False


def elevates(cmd: str, depth: int = 0) -> bool:
    """True if any simple command in `cmd` (including nested shells, eval, substitutions) runs sudo/su/doas/pkexec."""
    if not cmd or not isinstance(cmd, str):
        return False
    return any(_segment_elevates(_tokens(seg), depth) for seg in split_segments(cmd))


def is_read_only(cmd: str) -> bool:
    """True only when every simple command is on a small read-only allowlist and nothing redirects or substitutes."""
    if not cmd or not isinstance(cmd, str):
        return False
    if "$(" in cmd or "`" in cmd or ">" in cmd or "<(" in cmd:
        return False
    segments = split_segments(cmd)
    if not segments:
        return False
    for seg in segments:
        toks = _tokens(seg)
        while toks and toks[0] in KEYWORDS:
            toks = toks[1:]
        if not toks or _ASSIGN.match(toks[0]):
            return False
        base = os.path.basename(toks[0])
        if base == "git":
            sub = next((t for t in toks[1:] if not t.startswith("-")), "")
            if sub not in READ_ONLY_GIT:
                return False
        elif base == "find":
            if any(t in UNSAFE_FIND_FLAGS or t.startswith("-fprint") for t in toks[1:]):
                return False
        elif base not in READ_ONLY_COMMANDS:
            return False
    return True
