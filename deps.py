"""Dependency checks and installers behind `kxtc setup` and `kxtc doctor`.

Pure stdlib on purpose: it must run before the Antigravity SDK is installed.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional

from . import config

ANTIGRAVITY_REQUIREMENT = "google-antigravity>=0.1.21,<0.2"
KONTEXTA_NPM_PACKAGE = "kontexta-mcp"
LLAMASHIFT_REPO = os.environ.get("KXTC_LLAMASHIFT_REPO", "https://github.com/safiyu/llamashift.git")
# pip spec for LlamaShift; set KXTC_LLAMASHIFT_PIP=llamashift once it is published on PyPI
LLAMASHIFT_PIP_SPEC = os.environ.get("KXTC_LLAMASHIFT_PIP", f"git+{LLAMASHIFT_REPO}")
# LlamaShift management-API versions this client works with (its /api/version apiVersion)
LLAMASHIFT_API_MIN = 1
LLAMASHIFT_API_MAX = 1
NODE_MIN_MAJOR = 22


@dataclass
class Dependency:
    name: str
    purpose: str
    required: bool
    check: Callable[[], tuple[bool, str]]
    install: Optional[Callable[["Env"], bool]]
    manual: str


@dataclass
class Env:
    """Injectable side effects so installers are testable without touching the machine."""
    run: Callable = subprocess.run
    which: Callable = shutil.which
    ask: Callable = input
    out: Callable = print


def _reachable(url: str, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout):
            return True
    except urllib.error.HTTPError:
        return True  # the server answered, just not with 2xx
    except Exception:
        return False


def _llamashift_dir() -> str:
    return os.path.join(config.PROJECTS_DIR, "llamashift")


# ── checks ────────────────────────────────────────────────────────────

def check_antigravity() -> tuple[bool, str]:
    try:
        found = importlib.util.find_spec("google.antigravity") is not None
    except (ImportError, ValueError):
        found = False
    return found, "google.antigravity importable" if found else "python package google-antigravity not installed"


def check_kontexta() -> tuple[bool, str]:
    path = config.find_kontexta_mcp_path()
    if os.path.isfile(path):
        return True, path
    return False, "no Kontexta MCP build found (npm package kontexta-mcp or a local checkout)"


def _llamashift_api_version() -> Optional[int]:
    """apiVersion from GET /api/version; None when the server is old (404) or unreachable."""
    try:
        with urllib.request.urlopen(config.DEFAULT_LLAMASHIFT_URL + "/api/version", timeout=1.0) as resp:
            return int(json.loads(resp.read().decode("utf-8")).get("apiVersion"))
    except Exception:
        return None


def _llamashift_installed(which: Optional[Callable] = None) -> bool:
    return bool((which or shutil.which)("llamashift")) or os.path.isdir(_llamashift_dir())


def check_llamashift() -> tuple[bool, str]:
    url = config.DEFAULT_LLAMASHIFT_URL
    if _reachable(url):
        api = _llamashift_api_version()
        if api is None:
            return False, f"running at {url} but it predates the version API; upgrade LlamaShift"
        if not (LLAMASHIFT_API_MIN <= api <= LLAMASHIFT_API_MAX):
            return False, f"running at {url} with API v{api}; kontexta-cli supports v{LLAMASHIFT_API_MIN} to v{LLAMASHIFT_API_MAX}"
        return True, f"running at {url} (API v{api})"
    if _llamashift_installed():
        return False, f"installed but not running on {url}; start it with `llamashift`"
    return False, f"not installed and nothing running on {url}"


def check_laya() -> tuple[bool, str]:
    if _reachable(config.LAYA_BASE_URL):
        return True, f"running at {config.LAYA_BASE_URL}"
    return False, f"not running on {config.LAYA_BASE_URL} (built-in heuristic routing is used instead)"


# ── installers ────────────────────────────────────────────────────────

def install_antigravity(env: Env) -> bool:
    res = env.run([sys.executable, "-m", "pip", "install", ANTIGRAVITY_REQUIREMENT])
    return res.returncode == 0


def _node_major(env: Env) -> Optional[int]:
    node = env.which("node")
    if not node:
        return None
    try:
        out = env.run([node, "-p", "process.versions.node.split('.')[0]"], capture_output=True, text=True)
        return int(out.stdout.strip())
    except Exception:
        return None


def install_kontexta(env: Env) -> bool:
    major = _node_major(env)
    if major is None or major < NODE_MIN_MAJOR:
        env.out(f"  Kontexta needs Node {NODE_MIN_MAJOR}+ (found {'none' if major is None else major}). Install Node, then rerun `kxtc setup`.")
        return False
    npm = env.which("npm")
    if not npm:
        env.out("  npm not found. Install Node with npm, then rerun `kxtc setup`.")
        return False
    return env.run([npm, "install", "-g", KONTEXTA_NPM_PACKAGE]).returncode == 0


def install_llamashift(env: Env) -> bool:
    """pip-installs LlamaShift next to kontexta-cli; running it as a system service stays opt-in."""
    ok = env.run([sys.executable, "-m", "pip", "install", "--upgrade", LLAMASHIFT_PIP_SPEC]).returncode == 0
    if ok:
        env.out("  Start it with `llamashift` (data in ~/.llamashift, dashboard on "
                f"{config.DEFAULT_LLAMASHIFT_URL}). To run it as a service, use the installer in its repository.")
    return ok


DEPENDENCIES: list[Dependency] = [
    Dependency("antigravity", "Antigravity Python SDK (agent runtime)", True, check_antigravity, install_antigravity,
               f"pip install '{ANTIGRAVITY_REQUIREMENT}'"),
    Dependency("kontexta", "Kontexta MCP server (knowledge base, journal)", False, check_kontexta, install_kontexta,
               f"npm install -g {KONTEXTA_NPM_PACKAGE}   (needs Node {NODE_MIN_MAJOR}+)"),
    Dependency("llamashift", "LlamaShift model manager (local model switching)", False, check_llamashift, install_llamashift,
               f"pip install '{LLAMASHIFT_PIP_SPEC}' && llamashift"),
    Dependency("laya", "Laya System 1 decision server (fast routing and safety gate)", False, check_laya, None,
               f"start your Laya server on {config.LAYA_BASE_URL} or set KXTC_LAYA_BASE_URL; there is no installer for it here"),
]


def doctor(out: Callable = print) -> int:
    """Prints one line per dependency; returns 1 if a required one is missing."""
    missing_required = 0
    for dep in DEPENDENCIES:
        ok, detail = dep.check()
        mark = "OK  " if ok else ("MISS" if dep.required else "warn")
        out(f"[{mark}] {dep.name:<11} {dep.purpose}")
        out(f"       {detail}")
        if not ok:
            out(f"       fix: {dep.manual}")
            missing_required += 1 if dep.required else 0
    return 1 if missing_required else 0


def setup(only: Optional[list[str]] = None, assume_yes: bool = False, env: Optional[Env] = None) -> int:
    """Installs whatever `doctor` reports missing; returns 1 if a required dependency is still missing."""
    env = env or Env()
    unknown = set(only or []) - {d.name for d in DEPENDENCIES}
    if unknown:
        env.out(f"Unknown dependency: {', '.join(sorted(unknown))}. Choose from: {', '.join(d.name for d in DEPENDENCIES)}")
        return 2
    failed_required = False
    for dep in DEPENDENCIES:
        if only and dep.name not in only:
            continue
        ok, detail = dep.check()
        if ok:
            env.out(f"[OK  ] {dep.name}: {detail}")
            continue
        if dep.install is None:
            env.out(f"[skip] {dep.name}: {detail}\n       {dep.manual}")
            continue
        if not assume_yes:
            answer = env.ask(f"Install {dep.name} ({dep.purpose})? [Y/n] ").strip().lower()
            if answer in ("n", "no"):
                env.out(f"[skip] {dep.name}")
                failed_required |= dep.required
                continue
        env.out(f"[....] installing {dep.name}")
        installed = False
        try:
            installed = dep.install(env)
        except Exception as exc:  # keep going so one failure does not block the rest
            env.out(f"       installer error: {exc}")
        ok, detail = dep.check()
        if installed and (ok or (dep.name == "llamashift" and _llamashift_installed(env.which))):
            env.out(f"[done] {dep.name}: {detail}")
        else:
            env.out(f"[FAIL] {dep.name}: {detail}\n       manual: {dep.manual}")
            failed_required |= dep.required
    return 1 if failed_required else 0
