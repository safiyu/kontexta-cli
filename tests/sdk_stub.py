"""Minimal stand-in for the google.antigravity SDK so tests run without it."""

import sys
import types as pytypes
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HookResult:
    allow: bool = True
    message: str = ""


@dataclass
class ToolCall:
    name: str = ""
    args: dict = field(default_factory=dict)


@dataclass
class ToolResult:
    result: Any = None
    error: Any = None
    call: Any = None


class McpStdioServer:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class McpStreamableHttpServer(McpStdioServer):
    pass


class _Cfg:
    def __init__(self, *a, **kw):
        self.args, self.kwargs = a, kw


class Agent:
    def __init__(self, config):
        self.config = config

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def install():
    if "google.antigravity" in sys.modules and getattr(sys.modules["google.antigravity"], "_is_stub", False):
        return

    google = sys.modules.get("google") or pytypes.ModuleType("google")
    ag = pytypes.ModuleType("google.antigravity")
    ag._is_stub = True
    types_mod = pytypes.ModuleType("google.antigravity.types")
    types_mod.HookResult, types_mod.ToolCall, types_mod.ToolResult = HookResult, ToolCall, ToolResult
    types_mod.McpStdioServer, types_mod.McpStreamableHttpServer = McpStdioServer, McpStreamableHttpServer

    hooks_mod = pytypes.ModuleType("google.antigravity.hooks")
    hooks_mod.pre_tool_call_decide = lambda f: f
    hooks_mod.post_tool_call = lambda f: f
    policy_mod = pytypes.SimpleNamespace(
        allow_all=lambda: "allow_all",
        confirm_run_command=lambda handler: ["confirm", handler],
    )
    hooks_mod.policy = policy_mod

    ag.Agent, ag.LocalAgentConfig, ag.LocalOpenAIAgentConfig, ag.CapabilitiesConfig = Agent, _Cfg, _Cfg, _Cfg
    ag.types, ag.hooks = types_mod, hooks_mod
    sys.modules.update({
        "google": google,
        "google.antigravity": ag,
        "google.antigravity.types": types_mod,
        "google.antigravity.hooks": hooks_mod,
    })
