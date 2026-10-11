"""Persisted user preferences (~/.myagy/config.json)."""

import os
import json
from typing import Any, Dict

from ..config import PREFS_PATH, DEFAULT_CLOUD_MODEL

# Session attribute -> default value; only these keys are persisted
PREF_DEFAULTS: Dict[str, Any] = {
    "active_target": "9000",
    "cloud_model": DEFAULT_CLOUD_MODEL,
    "use_laya_adaptive_permissions": True,
    "dangerously_skip_permissions": True,
    "verbose": False,
    "show_metrics": True,
    "multiline_input": True,
    "autocompact_threshold": 20,
    "continuous_curation": True,
    "auto_repomap": True,
    "auto_instructions": True,
    "enable_hooks": True,
    "enable_mcp": True,
    "enable_kontexta": True,
    "max_tool_steps_per_turn": 0,
    "render_markdown": True,
}


def load_prefs(path: str = PREFS_PATH) -> Dict[str, Any]:
    """Returns saved preferences, ignoring unknown keys and wrong types."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    return {
        k: v for k, v in raw.items()
        if k in PREF_DEFAULTS and isinstance(v, type(PREF_DEFAULTS[k]))
    }


def save_prefs(values: Dict[str, Any], path: str = PREFS_PATH) -> bool:
    """Writes preferences atomically; returns False if the write failed."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({k: values[k] for k in PREF_DEFAULTS if k in values}, f, indent=2)
        os.replace(tmp, path)
        return True
    except Exception:
        return False


def reset_prefs(path: str = PREFS_PATH) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
