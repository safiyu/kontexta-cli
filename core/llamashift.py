"""LlamaShift client: cached status, mode awareness, readiness waits, and model hot-swap."""

import os
import re
import json
import time
import threading
import urllib.request
import urllib.error
from typing import Optional, List, Dict, Any, Callable, Tuple

from ..config import DEFAULT_MODEL_DIR, DEFAULT_LLAMASHIFT_URL

_BASE_HEADERS = {"User-Agent": "Antigravity/1.0"}
STATUS_TTL = 2.0
CONFIG_TTL = 30.0
DOWN_TTL = 3.0
GPU_TTL = 5.0
GPU_FAIL_TTL = 15.0
GPU_MAX_AGE = 30.0
START_TIMEOUT = 130.0  # a swap can wait out the 60s drain plus ~20s for the port to free

_lock = threading.Lock()
_cache: Dict[str, Any] = {"at": 0.0, "data": None, "down_until": 0.0, "state": "down"}
_cfg_cache: Dict[str, Any] = {"at": 0.0, "master_port": None}
_gpu_cache: Dict[str, Any] = {"at": 0.0, "data": None, "ttl": GPU_TTL, "refreshing": False}


def _headers(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Request headers; adds the management key when LLAMASHIFT_API_KEY is set (read per call)."""
    headers = dict(_BASE_HEADERS)
    key = os.environ.get("LLAMASHIFT_API_KEY", "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    if extra:
        headers.update(extra)
    return headers


def _get_json(url: str, timeout: float) -> Any:
    req = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def invalidate_cache():
    with _lock:
        _cache.update(at=0.0, data=None, down_until=0.0)
        _cfg_cache.update(at=0.0, master_port=None)
        _gpu_cache.update(at=0.0, data=None, ttl=GPU_TTL)


def _probe_failure_state() -> str:
    """Tells 'starting' (health 503) apart from 'down' once per failure window."""
    try:
        req = urllib.request.Request(f"{DEFAULT_LLAMASHIFT_URL}/api/health", headers=_headers())
        with urllib.request.urlopen(req, timeout=0.5):
            return "down"
    except urllib.error.HTTPError as e:
        return "starting" if e.code == 503 else "down"
    except Exception:
        return "down"


def get_status(max_age: float = STATUS_TTL) -> Optional[Dict[str, Any]]:
    """One cached GET /api/status; None while LlamaShift is down (failures are cached briefly too)."""
    now = time.monotonic()
    with _lock:
        if _cache["data"] is not None and now - _cache["at"] < max_age:
            return _cache["data"]
        if now < _cache["down_until"]:
            return None
    try:
        data = _get_json(f"{DEFAULT_LLAMASHIFT_URL}/api/status", 1.0)
        if not isinstance(data, dict):
            raise ValueError("unexpected status payload")
    except Exception as exc:
        unauthorized = isinstance(exc, urllib.error.HTTPError) and exc.code == 401
        state = "unauthorized" if unauthorized else _probe_failure_state()
        with _lock:
            _cache.update(data=None, down_until=time.monotonic() + DOWN_TTL, state=state)
        return None
    with _lock:
        _cache.update(at=time.monotonic(), data=data, down_until=0.0, state="up")
    return data


def shift_state() -> str:
    """'up', 'starting', 'unauthorized' (API key missing or wrong) or 'down'."""
    return "up" if get_status() is not None else _cache["state"]


def get_gpu_vram() -> Optional[List[Dict[str, Any]]]:
    """GET /api/gpu as [{vendor, name, used_mib, total_mib}]; None when unavailable."""
    try:
        raw = _get_json(f"{DEFAULT_LLAMASHIFT_URL}/api/gpu", 1.5)
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    out = []
    for vendor in ("nvidia", "amd"):
        gpu = raw.get(vendor)
        if not isinstance(gpu, dict):
            continue
        try:
            total, used = float(gpu.get("mem_total") or 0), float(gpu.get("mem_used") or 0)
        except (TypeError, ValueError):
            continue
        if total > 0:
            out.append({"vendor": vendor, "name": str(gpu.get("name", "")), "used_mib": used, "total_mib": total})
    return out or None


def _refresh_gpu():
    data = get_gpu_vram()
    with _lock:
        _gpu_cache.update(at=time.monotonic(), data=data, ttl=GPU_TTL if data else GPU_FAIL_TTL, refreshing=False)


def vram_snapshot() -> Optional[List[Dict[str, Any]]]:
    """Cached VRAM for UI chrome: never blocks, a stale value starts a background refresh, old data is dropped."""
    now = time.monotonic()
    with _lock:
        start = now - _gpu_cache["at"] >= _gpu_cache["ttl"] and not _gpu_cache["refreshing"]
        if start:
            _gpu_cache["refreshing"] = True
        data = _gpu_cache["data"] if now - _gpu_cache["at"] < GPU_MAX_AGE else None
    if start:
        threading.Thread(target=_refresh_gpu, daemon=True, name="vram-refresh").start()
    return data


def get_mode() -> Optional[str]:
    status = get_status()
    return status.get("mode", "single_port") if status else None


def get_master_port(status: Optional[Dict[str, Any]] = None) -> str:
    """masterPort lives in /api/config, not /api/status; cached because it rarely changes."""
    if status and status.get("masterPort") is not None:
        return str(status["masterPort"])
    now = time.monotonic()
    with _lock:
        if _cfg_cache["master_port"] is not None and now - _cfg_cache["at"] < CONFIG_TTL:
            return _cfg_cache["master_port"]
    try:
        port = str(_get_json(f"{DEFAULT_LLAMASHIFT_URL}/api/config", 1.0).get("masterPort", 9000))
    except Exception:
        return "9000"
    with _lock:
        _cfg_cache.update(at=time.monotonic(), master_port=port)
    return port


def _normalize_models(status: Dict[str, Any]) -> List[Dict[str, Any]]:
    mode = status.get("mode", "single_port")
    master_port = None
    out = []
    for m in status.get("models", []):
        is_dedicated = bool(m.get("dedicated", False))
        if m.get("effectivePort") is not None:
            effective_port = str(m["effectivePort"])
        else:
            master_port = master_port or get_master_port(status)
            effective_port = master_port if mode == "single_port" and not is_dedicated else str(m.get("port", master_port))
        try:
            ctx_size = int(m.get("ctxSize") or 0)
        except (TypeError, ValueError):
            ctx_size = 0
        out.append({
            "id": m.get("id", ""),
            "name": m.get("name", ""),
            "filename": m.get("filename", ""),
            "port": effective_port,
            "raw_port": str(m.get("port", "")),
            "dedicated": is_dedicated,
            "status": m.get("status", "stopped"),
            "gpu": m.get("gpu", ""),
            "devices": m.get("devices", []),
            "size": m.get("size", ""),
            "desc": m.get("desc", ""),
            "ctx_size": ctx_size,
        })
    return out


def get_llamashift_models() -> List[Dict[str, Any]]:
    """All configured model profiles (running and stopped) from the cached /api/status."""
    status = get_status()
    return _normalize_models(status) if status else []


def get_llamashift_active() -> List[Dict[str, Any]]:
    """Currently running profiles, derived from the cached /api/status."""
    return [m for m in get_llamashift_models() if m["status"] == "running"]


def ctx_size_for_port(port: int | str) -> Optional[int]:
    """Context size LlamaShift configured for the model running on this port."""
    for m in get_llamashift_active():
        if m["port"] == str(port) and m["ctx_size"] > 0:
            return m["ctx_size"]
    return None


def detect_port_model(port: int | str, fallback_url: Optional[str] = None) -> Dict[str, Any]:
    """Finds the model running on a port: LlamaShift status first, then the llama-server itself."""
    port_int = int(port)
    model_dir = DEFAULT_MODEL_DIR

    for m in get_llamashift_active():
        if m["port"] == str(port_int):
            fname = m.get("filename") or ""
            full_p = os.path.join(model_dir, fname) if fname and os.path.exists(os.path.join(model_dir, fname)) else fname
            return {
                "model_path": full_p or fname,
                "filename": fname or os.path.basename(full_p),
                "name": m.get("name") or fname,
                "id": m.get("id", ""),
                "status": "online",
                "source": "llamashift",
                "port": port_int,
                "gpu": m.get("gpu", ""),
            }

    url = fallback_url or f"http://localhost:{port_int}/v1"
    model_id = query_endpoint_model(url, timeout=0.6)
    if model_id:
        fname = os.path.basename(model_id)
        return {
            "model_path": model_id,
            "filename": fname,
            "name": fname,
            "id": fname,
            "status": "online",
            "source": "llama-server",
            "port": port_int,
            "gpu": "",
        }

    return {
        "model_path": "",
        "filename": "[Offline / No Model Loaded]",
        "name": "Offline",
        "id": "",
        "status": "offline",
        "source": "none",
        "port": port_int,
        "gpu": "",
    }


def query_endpoint_model(url: str, timeout: float = 1.0) -> Optional[str]:
    """Compatibility wrapper that queries a model URL."""
    try:
        req = urllib.request.Request(f"{url.rstrip('/')}/models", headers=_BASE_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
                return data["data"][0].get("id") or data["data"][0].get("name")
            if "models" in data and isinstance(data["models"], list) and len(data["models"]) > 0:
                return data["models"][0].get("name") or data["models"][0].get("model")
    except Exception:
        pass
    return None


def _is_timeout(exc: BaseException) -> bool:
    return isinstance(exc, TimeoutError) or isinstance(getattr(exc, "reason", None), TimeoutError)


def trigger_llamashift_switch(model_id: str) -> Dict[str, Any]:
    """Starts or switches a model via LlamaShift; the result may carry `warnings`, or `timeout` when it kept working past our wait."""
    try:
        payload = json.dumps({"model": model_id}).encode("utf-8")
        req = urllib.request.Request(
            f"{DEFAULT_LLAMASHIFT_URL}/api/start",
            data=payload,
            headers=_headers({"Content-Type": "application/json"}),
        )
        with urllib.request.urlopen(req, timeout=START_TIMEOUT) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode("utf-8"))
        except Exception:
            body = {}
        detail = str(body.get("error") or e.reason)
        if e.code == 401:
            detail += " (set LLAMASHIFT_API_KEY)"
        result = {"error": detail}
        if isinstance(body.get("warnings"), list):
            result["warnings"] = body["warnings"]
    except Exception as e:
        result = {"error": str(e)}
        if _is_timeout(e):
            result["timeout"] = True
    invalidate_cache()
    return result


def resolve_model_id(configured: str) -> Optional[str]:
    """Maps a configured model id to a real LlamaShift profile id, or None if no profile fits."""
    status = get_status()
    if status is None:
        return configured
    models = _normalize_models(status)
    if any(m["id"] == configured for m in models):
        return configured
    stem = re.split(r"[\d_.\-]", configured.lower())[0]
    if len(stem) >= 4:
        for m in models:
            if stem in f"{m['id']} {m['name']} {m['filename']}".lower():
                return m["id"]
    return None


def can_wake_secondary(model_id: str, port: int | str) -> Tuple[bool, str]:
    """A second model may start only if LlamaShift runs it on its own port (multi_port, or dedicated in single_port)."""
    status = get_status()
    if status is None:
        return False, "LlamaShift is not reachable"
    profile = next((m for m in _normalize_models(status) if m["id"] == model_id), None)
    if profile is None:
        return False, f"model '{model_id}' is not defined in LlamaShift"
    if profile["port"] != str(port):
        if status.get("mode", "single_port") == "single_port" and not profile["dedicated"]:
            return False, (f"'{model_id}' shares port {profile['port']} in single_port mode, so starting it would stop the main model "
                           f"(set \"dedicated\": true on it in LlamaShift)")
        return False, f"'{model_id}' is configured for port {profile['port']}, not {port}"
    return True, ""


def get_logs_tail(model_id: str, lines: int = 12) -> str:
    """Last log lines for a model from /api/logs; empty if unavailable."""
    try:
        data = _get_json(f"{DEFAULT_LLAMASHIFT_URL}/api/logs?model={model_id}&lines={int(lines)}", 1.5)
        logs = data.get("logs", "")
        text = "\n".join(logs) if isinstance(logs, list) else str(logs)
        return "" if text.startswith("Server logs will populate") else text
    except Exception:
        return ""


def model_is_running(model_id: str) -> Optional[bool]:
    """Fresh LlamaShift view of whether a profile is running; None when it cannot be told."""
    status = get_status(max_age=0.0)
    profile = next((m for m in _normalize_models(status) if m["id"] == model_id), None) if status else None
    return None if profile is None else profile["status"] == "running"


def _port_ready(port: int | str) -> bool:
    try:
        with urllib.request.urlopen(f"http://localhost:{int(port)}/health", timeout=1.0) as resp:
            return resp.status == 200
    except Exception:
        return False


def ensure_ready(model_id: str, port: int | str, timeout: float = 45.0,
                 on_progress: Optional[Callable[[float], None]] = None) -> Dict[str, Any]:
    """Starts a model and waits until its llama-server answers /health; returns {ok, reason, log_tail, elapsed, warnings}."""
    t0 = time.monotonic()
    warnings: List[str] = []

    def fail(reason: str) -> Dict[str, Any]:
        return {"ok": False, "reason": reason, "log_tail": get_logs_tail(model_id),
                "elapsed": time.monotonic() - t0, "warnings": warnings}

    res = trigger_llamashift_switch(model_id)
    warnings = [str(w) for w in res.get("warnings", [])]
    if "error" in res and not res.get("timeout"):
        return fail(f"start request failed: {res['error']}")
    if res.get("success") is False:
        return fail(str(res.get("message") or "start refused"))

    # A timed-out start call means LlamaShift is still draining or spawning, so "not running yet" is not a crash.
    start_pending = bool(res.get("timeout"))
    seen_running = False
    t_wait = time.monotonic()
    last_check = 0.0
    while True:
        waited = time.monotonic() - t_wait
        # On a shared port the old model still answers /health, so health alone is not proof the new one is up.
        if _port_ready(port) and model_is_running(model_id) is not False:
            invalidate_cache()
            return {"ok": True, "reason": "", "log_tail": "", "elapsed": time.monotonic() - t0, "warnings": warnings}
        if waited >= timeout:
            return fail(f"not ready after {timeout:.0f}s")
        if waited > 3.0 and waited - last_check >= 2.0:
            last_check = waited
            running = model_is_running(model_id)
            if running:
                seen_running = True
            elif running is False and (seen_running or not start_pending):
                return fail("the model process exited while loading")
        if on_progress:
            on_progress(time.monotonic() - t0)
        time.sleep(0.5)


def wake_compactor(port: int | str = 9001, timeout: float = 45.0,
                   on_progress: Optional[Callable[[float], None]] = None) -> Dict[str, Any]:
    """Brings the compactor model up on its own port, refusing when that would disturb the main model."""
    from ..config import COMPACTOR_MODEL_ID
    model_id = resolve_model_id(COMPACTOR_MODEL_ID)
    if model_id is None:
        return {"ok": False, "reason": f"no LlamaShift profile matches '{COMPACTOR_MODEL_ID}' (set MYAGY_COMPACTOR_MODEL)", "log_tail": "", "elapsed": 0.0, "warnings": []}
    allowed, why = can_wake_secondary(model_id, port)
    if not allowed:
        return {"ok": False, "reason": why, "log_tail": "", "elapsed": 0.0, "warnings": []}
    return ensure_ready(model_id, port, timeout, on_progress)


