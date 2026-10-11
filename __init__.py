"""
myagy - Antigravity Multi-GPU Hybrid Agent Package
Dual-GPU executive pipeline (AMD ROCm + NVIDIA CUDA) with continuous background hippocampus curation,
Laya System 1 decision engine (~33ms ModernBERT), and Cloud Gemini fallback.
"""

from .config import (
    PORT_ROCM,
    PORT_CUDA,
    PORT_LLAMASHIFT,
    PORT_LAYA,
    DEFAULT_CLOUD_MODEL,
    DEFAULT_LAYA_ENDPOINT,
    DEFAULT_LLAMASHIFT_URL,
    SESSIONS_DIR,
    MCP_CONFIG_PATH,
    OAUTH_TOKEN_PATH,
)

# Heavy names load on first access so leaf modules (core.coordinator, core.prefs, ...) import without the SDK
_LAZY = {
    **{n: ".terminal" for n in ("UI", "RICH_AVAILABLE", "console", "EscListener", "read_input_prompt", "interactive_loop", "main")},
    **{n: ".core" for n in ("MultiGpuHybridSession", "CudaCoordinator", "LayaDecisionEngine", "detect_port_model", "query_endpoint_model", "trigger_llamashift_switch", "get_llamashift_active", "get_llamashift_models")},
    **{n: ".agents" for n in ("SubagentManager", "SubagentTask", "ContextCurator", "BrainstormWorkflow")},
    **{n: ".context" for n in ("RepoMap", "ProjectInstructions", "ExternalHooksManager", "load_mcp_servers", "check_oauth_available", "McpStdioServer", "KontextaManager")},
    "WebSearchEngine": ".tools",
}


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value
    return value


__all__ = [
    # Core
    "MultiGpuHybridSession",
    "CudaCoordinator",
    "LayaDecisionEngine",
    "detect_port_model",
    "query_endpoint_model",
    "trigger_llamashift_switch",
    "get_llamashift_active",
    "get_llamashift_models",
    # Agents
    "SubagentManager",
    "SubagentTask",
    "ContextCurator",
    "BrainstormWorkflow",
    # Context
    "RepoMap",
    "ProjectInstructions",
    "ExternalHooksManager",
    "load_mcp_servers",
    "check_oauth_available",
    "McpStdioServer",
    "KontextaManager",
    # Tools
    "WebSearchEngine",
    # Terminal & UI
    "UI",
    "RICH_AVAILABLE",
    "console",
    "EscListener",
    "read_input_prompt",
    "interactive_loop",
    "main",
    # Config
    "PORT_ROCM",
    "PORT_CUDA",
    "PORT_LLAMASHIFT",
    "PORT_LAYA",
    "DEFAULT_CLOUD_MODEL",
    "DEFAULT_LAYA_ENDPOINT",
    "DEFAULT_LLAMASHIFT_URL",
    "SESSIONS_DIR",
    "MCP_CONFIG_PATH",
    "OAUTH_TOKEN_PATH",
]
