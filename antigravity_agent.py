#!/usr/bin/env python3
"""
Antigravity Multi-GPU Hybrid Agent Runner
Thin entry point delegating to the `kontexta_cli` package:
  - config.py: Central paths, ports, endpoint definitions
  - core/: session orchestrator, Laya engine, CUDA coordinator, llama-shift client
  - agents/: subagent queue, context compactor, /brainstorm workflow
  - context/: repo map, project instructions, hooks.json, MCP discovery
  - tools/: web search and page fetch
  - terminal/: REPL (cli.py), UI styling, ESC listener
"""

import asyncio
import importlib
import importlib.util
import os
import sys

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_NAME = __package__ or "kontexta_cli"

# The checkout directory (e.g. "kontexta-cli") need not be a valid package name, so load it by path under _PKG_NAME
if not __package__:
    _PARENT_DIR = os.path.dirname(_PKG_DIR)
    if _PARENT_DIR not in sys.path:
        sys.path.insert(0, _PARENT_DIR)
    if importlib.util.find_spec(_PKG_NAME) is None:
        _spec = importlib.util.spec_from_file_location(
            _PKG_NAME, os.path.join(_PKG_DIR, "__init__.py"), submodule_search_locations=[_PKG_DIR]
        )
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules[_PKG_NAME] = _mod
        _spec.loader.exec_module(_mod)


def __getattr__(name):
    # Backward-compat re-exports resolve lazily so importing this wrapper stays cheap
    return getattr(importlib.import_module(_PKG_NAME), name)


def _run_dependency_command(argv):
    """`kxtc setup [--yes] [--only NAME ...]` and `kxtc doctor`; runs before the SDK is imported."""
    deps = importlib.import_module(f"{_PKG_NAME}.deps")
    if argv[0] == "doctor":
        return deps.doctor()
    only, assume_yes = [], False
    rest = iter(argv[1:])
    for arg in rest:
        if arg in ("-y", "--yes"):
            assume_yes = True
        elif arg == "--only":
            only.append(next(rest, ""))
        else:
            print(f"Unknown option for setup: {arg}")
            return 2
    return deps.setup(only=only or None, assume_yes=assume_yes)


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("setup", "doctor"):
        sys.exit(_run_dependency_command(sys.argv[1:]))
    try:
        cli_mod = importlib.import_module(f"{_PKG_NAME}.terminal.cli")
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] in ("google", "httpx", "rich", "prompt_toolkit"):
            print(f"Missing dependency '{exc.name}'. Run `kxtc setup` (or `python3 {os.path.abspath(__file__)} setup`) to install everything.")
            sys.exit(1)
        raise
    reloader_mod = importlib.import_module(f"{_PKG_NAME}.core.reloader")
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        return cli_mod.main()
    shm_path = asyncio.run(cli_mod.main())
    if shm_path is not None:
        reloader_mod.perform_execv_reload(shm_path)


if __name__ == "__main__":
    main()
