"""Configuration constants and paths for Antigravity Hybrid Agent."""

import os

# Filesystem & Persistence Directories
SESSIONS_DIR = os.path.expanduser("~/.gemini/antigravity-cli/sessions")
MCP_CONFIG_PATH = os.path.expanduser("~/.gemini/config/mcp_config.json")
KONTEXTA_MCP_PATH = os.path.expanduser(os.environ.get("MYAGY_KONTEXTA_MCP", "~/Projects/kontexta/apps/mcp/dist/index.js"))
KONTEXTA_DATA_DIR = os.path.expanduser(os.environ.get("MYAGY_KONTEXTA_DATA", "~/.local/share/kontexta"))
DEFAULT_MODEL_DIR = os.path.expanduser(os.environ.get("MYAGY_MODEL_DIR", "/home/safiyu/models"))
OAUTH_TOKEN_PATH = os.path.expanduser("~/.gemini/antigravity-cli/antigravity-oauth-token")
PLANS_DIR = os.path.expanduser("~/.myagy/plans")
PREFS_PATH = os.path.expanduser("~/.myagy/config.json")
HISTORY_PATH = os.path.expanduser("~/.myagy/prompt_history")

# Default Service Endpoints & Ports
PORT_ROCM = 9000
PORT_CUDA = 9001
PORT_LLAMASHIFT = 8002
PORT_LAYA = 8003

DEFAULT_CLOUD_MODEL = "gemini-3.8-flash-high"
DEFAULT_LAYA_ENDPOINT = "http://localhost:8003/v1/systemone"
DEFAULT_LLAMASHIFT_URL = "http://localhost:8002"

# Local Dynamic Model Profiles for ROCm (Port 9000)
FAST_GIT_MODEL = "gemma4"      # Gemma 4 26B A4B MoE (~4B active params, ultra-fast git/tools)
DENSE_CODE_MODEL = "qwen27"    # Qwen 3.8 27B Dense (deep reasoning, architecture, heavy code)

# Endpoints Registry Initial Schema
DEFAULT_ENDPOINTS = {
    "9000": {
        "port": PORT_ROCM,
        "name": "ROCm:9000",
        "desc": "AMD ROCm GPU (Primary Executive)",
        "url": f"http://localhost:{PORT_ROCM}/v1",
        "model": "",
        "filename": "",
        "name_friendly": "",
        "status": "offline",
        "source": "none",
    },
    "9001": {
        "port": PORT_CUDA,
        "name": "CUDA:9001 (Compactor)",
        "desc": "NVIDIA RTX 4060 Context Synthesis Engine",
        "url": f"http://localhost:{PORT_CUDA}/v1",
        "model": "",
        "filename": "",
        "name_friendly": "",
        "status": "offline",
        "source": "none",
    },
}
