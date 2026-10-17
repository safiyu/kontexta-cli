#!/usr/bin/env python3
"""
Antigravity Multi-GPU Hybrid Agent Runner
Thin entry point delegating to the `myagy` package:
  - config.py: Central paths, ports, endpoint definitions
  - core/: session orchestrator, Laya engine, CUDA coordinator, llama-shift client
  - agents/: subagent queue, context compactor, /brainstorm workflow
  - context/: repo map, project instructions, hooks.json, MCP discovery
  - tools/: web search and page fetch
  - terminal/: REPL (cli.py), UI styling, ESC listener
"""

import asyncio
import importlib
import os
import sys

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_NAME = __package__ or os.path.basename(_PKG_DIR)

# Only the parent dir goes on the path so subpackages stay namespaced under the package name
if not __package__:
    _PARENT_DIR = os.path.dirname(_PKG_DIR)
    if _PARENT_DIR not in sys.path:
        sys.path.insert(0, _PARENT_DIR)


def __getattr__(name):
    # Backward-compat re-exports resolve lazily so importing this wrapper stays cheap
    return getattr(importlib.import_module(_PKG_NAME), name)


def main():
    cli_mod = importlib.import_module(f"{_PKG_NAME}.terminal.cli")
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
