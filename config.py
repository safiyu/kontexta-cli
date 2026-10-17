"""Configuration constants and paths for Antigravity Hybrid Agent."""

import os

import re
import sys
import shutil
import importlib.util


# Directory holding sibling checkouts (SDK, Kontexta, LlamaShift). A source checkout uses its parent
# directory; an installed copy (site-packages) uses <state dir>/deps. Override with KXTC_PROJECTS_DIR.
_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_INSTALLED = any(part in _PKG_PARENT for part in ("site-packages", "dist-packages"))
PROJECTS_DIR = os.path.expanduser(os.environ.get(
    "KXTC_PROJECTS_DIR",
    os.path.join(os.path.expanduser(os.environ.get("KXTC_HOME", "~/.kontexta-cli")), "deps") if _INSTALLED else _PKG_PARENT,
))


def _env(name: str, default: str) -> str:
    """KXTC_<name> from the environment, else default."""
    return os.environ.get("KXTC_" + name, default)

# ── Dynamic Antigravity SDK & Binary Discovery ────────────────────────

def _find_antigravity_sdk_dir() -> str | None:
    """Discovers the Antigravity Python SDK directory dynamically."""
    # 1. Explicit environment variables
    for env_var in ("ANTIGRAVITY_SDK_PATH", "KXTC_SDK_PATH", "GOOGLE_ANTIGRAVITY_PATH"):
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
        os.path.join(PROJECTS_DIR, "antigravity-sdk-python"),
        os.path.join(PROJECTS_DIR, "google-antigravity"),
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
    for env_var in ("KXTC_OAUTH_TOKEN", "ANTIGRAVITY_OAUTH_TOKEN_PATH"):
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
    for env_var in ("KXTC_MCP_CONFIG", "ANTIGRAVITY_MCP_CONFIG_PATH"):
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
SESSIONS_DIR = os.path.abspath(os.path.expanduser(os.environ.get("KXTC_SESSIONS_DIR", os.path.join(ANTIGRAVITY_CLI_DIR, "sessions"))))
MCP_CONFIG_PATH = _find_mcp_config_path()
def find_kontexta_mcp_path() -> str:
    """KXTC_KONTEXTA_MCP if set, else a sibling/~/Projects build, else the npm-installed kxta bin."""
    explicit = os.environ.get("KXTC_KONTEXTA_MCP")
    if explicit:
        return os.path.expanduser(explicit)
    tail = os.path.join("kontexta", "apps", "mcp", "dist", "index.js")
    candidates = [os.path.join(PROJECTS_DIR, tail), os.path.expanduser(os.path.join("~/Projects", tail))]
    for bin_name in ("kxta", "kontexta-mcp"):
        found = shutil.which(bin_name)
        if found:
            candidates.append(os.path.realpath(found))
    return next((c for c in candidates if os.path.isfile(c)), candidates[0])


KONTEXTA_MCP_PATH = find_kontexta_mcp_path()
KONTEXTA_DATA_DIR = os.path.expanduser(_env("KONTEXTA_DATA", "~/.local/share/kontexta"))
DEFAULT_MODEL_DIR = os.path.expanduser(_env("MODEL_DIR", "~/models"))
OAUTH_TOKEN_PATH = _find_oauth_token_path(ANTIGRAVITY_CLI_DIR)
PLANS_SUBDIR = os.path.join("docs", "plans")

# Per-user state (prefs, prompt history, repomap cache); override with KXTC_HOME
STATE_DIR = os.path.expanduser(_env("HOME", "~/.kontexta-cli"))
PREFS_PATH = os.path.join(STATE_DIR, "config.json")
HISTORY_PATH = os.path.join(STATE_DIR, "prompt_history")

# LlamaShift profile id and GGUF filename for the CUDA compactor and subagents
COMPACTOR_MODEL_ID = _env("COMPACTOR_MODEL", "granite42_8b")
COMPACTOR_MODEL_FILE = _env("COMPACTOR_FILE", "granite-4.2-8b-instruct-Q4_K_M.gguf")

# Service host and ports. The "9000"/"9002" endpoint ids stay stable target names even if ports change.
SERVICE_HOST = _env("HOST", "127.0.0.1")
PORT_ROCM = int(_env("ROCM_PORT", "9000"))
PORT_CUDA = int(_env("CUDA_PORT", "9002"))
PORT_LLAMASHIFT = int(_env("LLAMASHIFT_PORT", "8002"))
PORT_LAYA = int(_env("LAYA_PORT", "8003"))


def service_url(port: int | str, path: str = "") -> str:
    """http://<host>:<port><path> for a local service."""
    return f"http://{SERVICE_HOST}:{int(port)}{path}"


CUDA_BASE_URL = _env("CUDA_BASE_URL", service_url(PORT_CUDA, "/v1"))
ROCM_BASE_URL = _env("ROCM_BASE_URL", service_url(PORT_ROCM, "/v1"))
LAYA_BASE_URL = _env("LAYA_BASE_URL", service_url(PORT_LAYA))

DEFAULT_CLOUD_MODEL = _env("CLOUD_MODEL", "gemini-3.8-flash-high")
DEFAULT_CLOUD_IDLE_TIMEOUT = 1200  # seconds (20 min); set to 0 to disable idle timeout
DEFAULT_LAYA_ENDPOINT = _env("LAYA_ENDPOINT", service_url(PORT_LAYA, "/v1/systemone"))
DEFAULT_LLAMASHIFT_URL = _env("LLAMASHIFT_URL", service_url(PORT_LLAMASHIFT))

# Local dynamic model profiles for ROCm; the family keyword is matched against the loaded model name
FAST_GIT_MODEL = _env("FAST_MODEL", "gemma4")      # ultra-fast git/tools model
DENSE_CODE_MODEL = _env("DENSE_MODEL", "qwen27")   # deep reasoning, architecture, heavy code


def _family(model_id: str) -> str:
    m = re.match(r"[A-Za-z]+", model_id)
    return (m.group(0) if m else model_id).lower()


FAST_GIT_FAMILY = _family(FAST_GIT_MODEL)
DENSE_CODE_FAMILY = _family(DENSE_CODE_MODEL)

# Endpoints Registry Initial Schema
DEFAULT_ENDPOINTS = {
    "9000": {
        "port": PORT_ROCM,
        "name": f"ROCm:{PORT_ROCM}",
        "desc": "AMD ROCm GPU (Primary Executive)",
        "url": ROCM_BASE_URL,
        "model": "",
        "filename": "",
        "name_friendly": "",
        "status": "offline",
        "source": "none",
    },
    "9002": {
        "port": PORT_CUDA,
        "name": f"CUDA:{PORT_CUDA} (Compactor)",
        "desc": "NVIDIA RTX 4060 Context Synthesis Engine",
        "url": CUDA_BASE_URL,
        "model": "",
        "filename": "",
        "name_friendly": "",
        "status": "offline",
        "source": "none",
    },
}
