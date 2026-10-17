"""Configuration constants and paths for Antigravity Hybrid Agent."""

import os
import sys
import shutil
import importlib.util

# ── Dynamic Antigravity SDK & Binary Discovery ────────────────────────

def _find_antigravity_sdk_dir() -> str | None:
    """Discovers the Antigravity Python SDK directory dynamically."""
    # 1. Explicit environment variables
    for env_var in ("ANTIGRAVITY_SDK_PATH", "MYAGY_SDK_PATH", "GOOGLE_ANTIGRAVITY_PATH"):
        custom_path = os.environ.get(env_var)
        if custom_path:
            expanded = os.path.expanduser(custom_path)
            if os.path.isdir(expanded):
                return os.path.abspath(expanded)

    # 2. Check if google.antigravity is already importable / installed
    try:
        spec = importlib.util.find_spec("google.antigravity")
        if spec and spec.origin:
            return os.path.abspath(os.path.dirname(spec.origin))
    except Exception:
        pass

    # 3. Candidate search directories across system and user paths
    candidates = [
        "~/.gemini/antigravity/sdk",
        "~/.gemini/antigravity-cli/sdk",
        "~/.gemini/antigravity-ide/sdk",
        "~/Projects/antigravity-sdk-python",
        "~/Projects/google-antigravity",
        "~/.local/lib/python3.12/site-packages/google/antigravity",
    ]
    for c in candidates:
        exp = os.path.expanduser(c)
        if os.path.isdir(exp) and (os.path.isfile(os.path.join(exp, "__init__.py")) or os.path.isfile(os.path.join(exp, "agent.py"))):
            return os.path.abspath(exp)

    return None


def _find_localharness_bin(sdk_dir: str | None = None) -> str | None:
    """Discovers the compiled localharness binary dynamically."""
    # 1. Existing ANTIGRAVITY_HARNESS_PATH env var
    env_bin = os.environ.get("ANTIGRAVITY_HARNESS_PATH")
    if env_bin and os.path.isfile(os.path.expanduser(env_bin)):
        return os.path.abspath(os.path.expanduser(env_bin))

    # 2. Check within discovered SDK directory
    if sdk_dir:
        candidates = [
            os.path.join(sdk_dir, "bin", "localharness"),
            os.path.join(os.path.dirname(sdk_dir), "bin", "localharness"),
            os.path.join(sdk_dir, "localharness"),
        ]
        for c in candidates:
            if os.path.isfile(c) and os.access(c, os.X_OK):
                return os.path.abspath(c)

    # 3. Check system PATH
    which_bin = shutil.which("localharness")
    if which_bin:
        return os.path.abspath(which_bin)

    # 4. Check known Antigravity runtime locations
    known_bins = [
        "~/.gemini/antigravity/bin/localharness",
        "~/.gemini/antigravity-cli/bin/localharness",
        "~/.gemini/antigravity-ide/bin/localharness",
        "~/.local/bin/localharness",
    ]
    for kb in known_bins:
        exp = os.path.expanduser(kb)
        if os.path.isfile(exp) and os.access(exp, os.X_OK):
            return os.path.abspath(exp)

    return None


def setup_antigravity_sdk(custom_path: str | None = None) -> tuple[str | None, str | None]:
    """
    Dynamically registers the Antigravity SDK in sys.path and configures
    ANTIGRAVITY_HARNESS_PATH if not already set.
    """
    if custom_path:
        os.environ["ANTIGRAVITY_SDK_PATH"] = custom_path

    sdk_dir = _find_antigravity_sdk_dir()
    if sdk_dir:
        parent = os.path.dirname(sdk_dir)
        grandparent = os.path.dirname(parent)
        target_path = None
        if os.path.basename(parent) == "google":
            target_path = grandparent
        elif os.path.isdir(os.path.join(sdk_dir, "google")):
            target_path = sdk_dir
        elif parent not in sys.path:
            target_path = parent

        if target_path and target_path not in sys.path:
            sys.path.insert(0, target_path)

    harness_bin = _find_localharness_bin(sdk_dir)
    if harness_bin and not os.environ.get("ANTIGRAVITY_HARNESS_PATH"):
        os.environ["ANTIGRAVITY_HARNESS_PATH"] = harness_bin

    return sdk_dir, harness_bin


# Execute discovery on import
ANTIGRAVITY_SDK_DIR, ANTIGRAVITY_HARNESS_PATH = setup_antigravity_sdk()


# ── Filesystem & Persistence Directories ─────────────────────────────

def _find_antigravity_cli_dir() -> str:
    """Discovers the Antigravity CLI / runtime home directory dynamically."""
    for env_var in ("ANTIGRAVITY_CLI_DIR", "ANTIGRAVITY_HOME", "GEMINI_CLI_DIR"):
        v = os.environ.get(env_var)
        if v:
            return os.path.abspath(os.path.expanduser(v))

    candidates = [
        "~/.gemini/antigravity-cli",
        "~/.gemini/antigravity",
        "~/.gemini/antigravity-ide",
        "~/.config/antigravity",
    ]
    for c in candidates:
        exp = os.path.abspath(os.path.expanduser(c))
        if os.path.isdir(exp):
            return exp

    return os.path.abspath(os.path.expanduser("~/.gemini/antigravity-cli"))


def _find_oauth_token_path(cli_dir: str) -> str:
    """Discovers the active Google Antigravity OAuth token file dynamically."""
    for env_var in ("MYAGY_OAUTH_TOKEN", "ANTIGRAVITY_OAUTH_TOKEN_PATH"):
        v = os.environ.get(env_var)
        if v:
            return os.path.abspath(os.path.expanduser(v))

    candidates = [
        os.path.join(cli_dir, "antigravity-oauth-token"),
        os.path.expanduser("~/.gemini/antigravity-cli/antigravity-oauth-token"),
        os.path.expanduser("~/.gemini/antigravity/antigravity-oauth-token"),
        os.path.expanduser("~/.gemini/antigravity-ide/antigravity-oauth-token"),
        os.path.expanduser("~/.config/antigravity/oauth-token"),
    ]
    for c in candidates:
        if os.path.isfile(c) and os.path.getsize(c) > 0:
            return os.path.abspath(c)

    return os.path.abspath(candidates[0])


def _find_mcp_config_path() -> str:
    """Discovers the MCP config file dynamically."""
    for env_var in ("MYAGY_MCP_CONFIG", "ANTIGRAVITY_MCP_CONFIG_PATH"):
        v = os.environ.get(env_var)
        if v:
            return os.path.abspath(os.path.expanduser(v))

    candidates = [
        os.path.expanduser("~/.gemini/config/mcp_config.json"),
        os.path.expanduser("~/.gemini/antigravity-cli/mcp/mcp_config.json"),
        os.path.expanduser("~/.gemini/antigravity-ide/mcp/mcp_config.json"),
        os.path.abspath(".agents/mcp_config.json"),
        os.path.abspath(".gemini/mcp_config.json"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)

    return os.path.abspath(candidates[0])


ANTIGRAVITY_CLI_DIR = _find_antigravity_cli_dir()
SESSIONS_DIR = os.path.abspath(os.path.expanduser(os.environ.get("MYAGY_SESSIONS_DIR", os.path.join(ANTIGRAVITY_CLI_DIR, "sessions"))))
MCP_CONFIG_PATH = _find_mcp_config_path()
KONTEXTA_MCP_PATH = os.path.expanduser(os.environ.get("MYAGY_KONTEXTA_MCP", "~/Projects/kontexta/apps/mcp/dist/index.js"))
KONTEXTA_DATA_DIR = os.path.expanduser(os.environ.get("MYAGY_KONTEXTA_DATA", "~/.local/share/kontexta"))
DEFAULT_MODEL_DIR = os.path.expanduser(os.environ.get("MYAGY_MODEL_DIR", "~/models"))
OAUTH_TOKEN_PATH = _find_oauth_token_path(ANTIGRAVITY_CLI_DIR)
PLANS_SUBDIR = os.path.join("docs", "plans")
PREFS_PATH = os.path.expanduser("~/.myagy/config.json")
HISTORY_PATH = os.path.expanduser("~/.myagy/prompt_history")

# LlamaShift profile id for the CUDA compactor and subagents
COMPACTOR_MODEL_ID = os.environ.get("MYAGY_COMPACTOR_MODEL", "granite42_8b")

# Default Service Endpoints & Ports
PORT_ROCM = 9000
PORT_CUDA = 9002
PORT_LLAMASHIFT = 8002
PORT_LAYA = 8003

CUDA_BASE_URL = os.environ.get("MYAGY_CUDA_BASE_URL", "http://127.0.0.1:9002/v1")
ROCM_BASE_URL = os.environ.get("MYAGY_ROCM_BASE_URL", "http://127.0.0.1:9000/v1")
LAYA_BASE_URL = os.environ.get("MYAGY_LAYA_BASE_URL", "http://127.0.0.1:8003")

DEFAULT_CLOUD_MODEL = "gemini-3.8-flash-high"
DEFAULT_CLOUD_IDLE_TIMEOUT = 1200  # seconds (20 min); set to 0 to disable idle timeout
DEFAULT_LAYA_ENDPOINT = "http://localhost:8003/v1/systemone"
DEFAULT_LLAMASHIFT_URL = "http://localhost:8002"

# Local Dynamic Model Profiles for ROCm (Port 9000)
FAST_GIT_MODEL = os.environ.get("MYAGY_FAST_MODEL", "gemma4")      # Gemma 4 26B A4B MoE (~4B active params, ultra-fast git/tools)
DENSE_CODE_MODEL = os.environ.get("MYAGY_DENSE_MODEL", "qwen27")    # Qwen 3.8 27B Dense (deep reasoning, architecture, heavy code)

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
    "9002": {
        "port": PORT_CUDA,
        "name": "CUDA:9002 (Compactor)",
        "desc": "NVIDIA RTX 4060 Context Synthesis Engine",
        "url": f"http://localhost:{PORT_CUDA}/v1",
        "model": "",
        "filename": "",
        "name_friendly": "",
        "status": "offline",
        "source": "none",
    },
}
