import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from myagy.core import llamashift as ls
from myagy.agents import compactor
from myagy.agents.compactor import ContextCurator


def _status(mode="multi_port", granite_port=9002, running=("qwen27",), granite_dedicated=False, effective=False):
    def model(mid, port, ctx, fname, dedicated=False):
        return {"id": mid, "name": mid.upper(), "filename": fname, "port": port, "ctxSize": ctx, "dedicated": dedicated,
                "status": "running" if mid in running else "stopped", "gpu": "GPU", "size": "8B"}
    models = [
        model("qwen27", 9000, 65152, "qwen27.gguf"),
        model("granite_8b_q4", granite_port, 26384, "granite-4.2-8b.gguf", granite_dedicated),
    ]
    if effective:
        for m in models:
            m["effectivePort"] = 19000 if mode == "single_port" and not m["dedicated"] else m["port"]
    return {"mode": mode, "models": models}


class FakeShift:
    """Serves LlamaShift's /api/* plus a llama-server style /health on the same port."""

    def __init__(self, status, health_503_first=0, health_ok=True, master_port=9000, logs=None,
                 api_key=None, start_delay=0.0, start_warnings=None, start_code=200, running_after=0.0,
                 gpu="default", gpu_delay=0.0):
        self.status, self.paths = status, []
        self.api_key, self.start_delay = api_key, start_delay
        self.running_after, self.started = running_after, {}
        self.gpu_delay = gpu_delay
        self.gpu = {
            "nvidia": {"name": "RTX 4060", "temp": 50.0, "util": 10.0, "mem_used": 2970.0, "mem_total": 8188.0, "power": 30.0},
            "amd": {"name": "R9700", "temp": 60.0, "util": 20.0, "mem_used": 14540.0, "mem_total": 24560.0, "power": 90.0},
        } if gpu == "default" else gpu
        self.start_warnings, self.start_code = start_warnings or [], start_code
        self.config = {"mode": status.get("mode", "single_port"), "masterPort": master_port}
        self.logs = ["line one", "line two"] if logs is None else logs
        self.health_left, self.health_ok = health_503_first, health_ok
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body):
                raw = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _authorized(self):
                if not outer.api_key or not self.path.startswith("/api/") or self.path.startswith("/api/health"):
                    return True
                if self.headers.get("Authorization") == f"Bearer {outer.api_key}":
                    return True
                self._send(401, {"error": "Unauthorized: a valid management API key or PIN session is required"})
                return False

            def do_GET(self):
                outer.paths.append(self.path)
                if not self._authorized():
                    return
                if self.path.startswith("/api/status"):
                    now = time.monotonic()
                    for m in outer.status["models"]:
                        t = outer.started.get(m["id"])
                        if t is not None and now - t >= outer.running_after:
                            m["status"] = "running"
                    self._send(200, outer.status)
                elif self.path.startswith("/api/config"):
                    self._send(200, outer.config)
                elif self.path.startswith("/api/gpu"):
                    if outer.gpu_delay:
                        time.sleep(outer.gpu_delay)
                    self._send(200, outer.gpu)
                elif self.path.startswith("/api/logs"):
                    self._send(200, {"model": "x", "logs": outer.logs})
                elif self.path == "/health":
                    if outer.health_left > 0 or not outer.health_ok:
                        outer.health_left -= 1
                        self._send(503, {"status": "loading"})
                    else:
                        self._send(200, {"status": "ok"})
                else:
                    self._send(404, {})

            def do_POST(self):
                outer.paths.append(self.path)
                if not self._authorized():
                    return
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if outer.start_delay:
                    time.sleep(outer.start_delay)
                if body.get("model") == "missing":
                    self._send(400, {"error": "Invalid or missing 'model' parameter"})
                elif outer.start_code == 409:
                    self._send(409, {"error": "Not enough free VRAM. Send force to start anyway.", "warnings": outer.start_warnings})
                else:
                    outer.started[body.get("model")] = time.monotonic()
                    self._send(200, {"success": True, "warnings": outer.start_warnings})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture
def shift(monkeypatch):
    made = []

    def make(status=None, **kw):
        fake = FakeShift(status or _status(), **kw)
        made.append(fake)
        monkeypatch.setattr(ls, "DEFAULT_LLAMASHIFT_URL", fake.url)
        ls.invalidate_cache()
        return fake

    yield make
    for f in made:
        f.close()
    ls.invalidate_cache()


def test_status_is_cached_and_never_uses_api_active(shift):
    fake = shift()
    ls.get_llamashift_models()
    ls.get_llamashift_active()
    ls.detect_port_model(9000)
    assert fake.paths.count("/api/status") == 1
    assert not any("/api/active" in p for p in fake.paths)


def test_running_models_and_ctx_size(shift):
    shift()
    assert [m["id"] for m in ls.get_llamashift_active()] == ["qwen27"]
    assert ls.ctx_size_for_port(9000) == 65152
    assert ls.ctx_size_for_port(9002) is None
    assert ls.detect_port_model(9000)["status"] == "online"


def test_master_port_comes_from_api_config(shift):
    fake = shift(_status(mode="single_port"), master_port=19000)
    assert {m["port"] for m in ls.get_llamashift_models()} == {"19000"}
    ls.get_llamashift_models()
    assert fake.paths.count("/api/config") == 1
    assert ls.ctx_size_for_port(19000) == 65152


def test_master_port_falls_back_to_9000_when_config_unreachable(shift, monkeypatch):
    shift(_status(mode="single_port"))
    real = ls._get_json
    monkeypatch.setattr(ls, "_get_json", lambda url, t: (_ for _ in ()).throw(OSError("x")) if "/api/config" in url else real(url, t))
    assert {m["port"] for m in ls.get_llamashift_models()} == {"9000"}


def test_log_placeholder_is_ignored(shift):
    shift(logs="Server logs will populate here once the model starts.")
    assert ls.get_logs_tail("granite_8b_q4") == ""


def test_single_port_maps_every_model_to_master_port(shift):
    shift(_status(mode="single_port"))
    assert {m["port"] for m in ls.get_llamashift_models()} == {"9000"}


def test_down_is_cached_and_not_probed_repeatedly(monkeypatch):
    ls.invalidate_cache()
    monkeypatch.setattr(ls, "DEFAULT_LLAMASHIFT_URL", "http://127.0.0.1:9")
    calls = []
    real = ls._get_json
    monkeypatch.setattr(ls, "_get_json", lambda *a, **k: calls.append(a) or real(*a, **k))
    assert ls.get_status() is None and ls.shift_state() == "down"
    assert ls.get_status() is None
    assert len(calls) == 1
    ls.invalidate_cache()


def test_starting_state_from_health_503(monkeypatch):
    ls.invalidate_cache()
    monkeypatch.setattr(ls, "_get_json", lambda *a, **k: (_ for _ in ()).throw(OSError("refused")))
    monkeypatch.setattr(ls, "_probe_failure_state", lambda: "starting")
    assert ls.shift_state() == "starting"
    ls.invalidate_cache()


def test_resolve_model_id(shift):
    shift()
    assert ls.resolve_model_id("granite_8b_q4") == "granite_8b_q4"
    assert ls.resolve_model_id("granite42_8b") == "granite_8b_q4"
    assert ls.resolve_model_id("llama99") is None


def test_wake_refused_in_single_port_and_never_calls_start(shift, monkeypatch):
    fake = shift(_status(mode="single_port"))
    monkeypatch.setattr("myagy.config.COMPACTOR_MODEL_ID", "granite_8b_q4")
    res = ls.wake_compactor(9002, timeout=2)
    assert not res["ok"] and "single_port" in res["reason"] and "dedicated" in res["reason"]
    assert "/api/start" not in fake.paths


def test_wake_allowed_for_dedicated_model_in_single_port(shift):
    shift(_status(mode="single_port", granite_dedicated=True))
    assert ls.can_wake_secondary("granite_8b_q4", 9002) == (True, "")


def test_wake_refused_when_profile_is_on_another_port(shift, monkeypatch):
    shift(_status(granite_port=9005))
    monkeypatch.setattr("myagy.config.COMPACTOR_MODEL_ID", "granite_8b_q4")
    res = ls.wake_compactor(9002, timeout=2)
    assert not res["ok"] and "9005" in res["reason"]


def test_wake_refused_when_no_profile_matches(shift, monkeypatch):
    shift()
    monkeypatch.setattr("myagy.config.COMPACTOR_MODEL_ID", "nosuchmodel")
    res = ls.wake_compactor(9002, timeout=2)
    assert not res["ok"] and "MYAGY_COMPACTOR_MODEL" in res["reason"]


def test_ensure_ready_waits_for_health(shift):
    fake = shift(health_503_first=2)
    res = ls.ensure_ready("granite_8b_q4", fake.port, timeout=10)
    assert res["ok"] and res["elapsed"] >= 0.9
    assert "/api/start" in fake.paths


def test_ensure_ready_timeout_returns_log_tail(shift):
    fake = shift(health_ok=False)
    res = ls.ensure_ready("granite_8b_q4", fake.port, timeout=1.2)
    assert not res["ok"] and "not ready" in res["reason"]
    assert "line two" in res["log_tail"]


def test_start_error_body_is_reported(shift):
    shift()
    res = ls.trigger_llamashift_switch("missing")
    assert "Invalid or missing" in res["error"]
    res = ls.ensure_ready("missing", 9999, timeout=1)
    assert not res["ok"] and "Invalid or missing" in res["reason"]


def test_context_metrics_use_llamashift_ctx_size(session, monkeypatch):
    monkeypatch.setattr(compactor, "ctx_size_for_port", lambda port: 26384)
    # Force the live /slots query to fail so the ctx_size_for_port fallback is exercised
    # regardless of whether a real llama-server is running on the port.
    def _no_slots(*a, **k):
        raise OSError("slots query disabled in test")
    monkeypatch.setattr(compactor.urllib.request, "urlopen", _no_slots)
    metrics = ContextCurator.get_context_metrics(session, "9000")
    assert metrics["max_ctx"] == 26384


def test_wake_skip_note_is_shown_once(capsys):
    compactor._noted_wake_reasons.clear()
    compactor._note_wake_skipped("single_port mode")
    compactor._note_wake_skipped("single_port mode")
    out = capsys.readouterr()
    assert (out.out + out.err).count("single_port mode") == 1
    compactor._noted_wake_reasons.clear()


def test_refresh_endpoints_announces_transitions_once(monkeypatch, capsys):
    from myagy.core import session as sess_mod
    state = {"v": "down"}
    monkeypatch.setattr(sess_mod, "shift_state", lambda: state["v"])
    monkeypatch.setattr(sess_mod, "get_llamashift_active", lambda: [])
    offline = {"model_path": "", "filename": "x", "name": "x", "status": "offline", "source": "none", "port": 0, "gpu": ""}
    monkeypatch.setattr(sess_mod, "detect_port_model", lambda *a, **k: offline)
    s = sess_mod.MultiGpuHybridSession(enable_mcp=False, json_output=False)
    s.refresh_endpoints()
    s.refresh_endpoints()
    state["v"] = "up"
    s.refresh_endpoints()
    out = capsys.readouterr()
    text = out.out + out.err
    assert text.count("not reachable") == 1
    assert text.count("reachable again") == 1


def test_api_key_is_sent_when_configured(shift, monkeypatch):
    shift(api_key="s3cret")
    monkeypatch.setenv("LLAMASHIFT_API_KEY", "s3cret")
    assert ls.shift_state() == "up"
    assert [m["id"] for m in ls.get_llamashift_active()] == ["qwen27"]


def test_missing_key_is_reported_as_unauthorized_not_down(shift, monkeypatch):
    shift(api_key="s3cret")
    monkeypatch.delenv("LLAMASHIFT_API_KEY", raising=False)
    assert ls.get_status() is None
    assert ls.shift_state() == "unauthorized"
    res = ls.trigger_llamashift_switch("granite_8b_q4")
    assert "LLAMASHIFT_API_KEY" in res["error"]


def test_wrong_key_is_unauthorized(shift, monkeypatch):
    shift(api_key="s3cret")
    monkeypatch.setenv("LLAMASHIFT_API_KEY", "nope")
    assert ls.shift_state() == "unauthorized"


def test_slow_start_timeout_keeps_polling_instead_of_failing(shift, monkeypatch):
    fake = shift(start_delay=1.5)
    monkeypatch.setattr(ls, "START_TIMEOUT", 0.4)
    res = ls.trigger_llamashift_switch("granite_8b_q4")
    assert res.get("timeout") is True
    res = ls.ensure_ready("granite_8b_q4", fake.port, timeout=10)
    assert res["ok"], res


def test_readiness_window_starts_after_the_start_call_returns(shift, monkeypatch):
    fake = shift(start_delay=1.0, health_503_first=2)
    res = ls.ensure_ready("granite_8b_q4", fake.port, timeout=5)
    assert res["ok"] and res["elapsed"] >= 1.0


def test_start_warnings_are_returned(shift):
    fake = shift(start_warnings=["Model may not fit: 9.1 GB needed, 6.0 GB free"])
    res = ls.ensure_ready("granite_8b_q4", fake.port, timeout=5)
    assert res["ok"] and res["warnings"] == ["Model may not fit: 9.1 GB needed, 6.0 GB free"]


def test_strict_vram_409_surfaces_message_and_warnings(shift):
    fake = shift(start_code=409, start_warnings=["Model may not fit"])
    res = ls.ensure_ready("granite_8b_q4", fake.port, timeout=2)
    assert not res["ok"] and "Not enough free VRAM" in res["reason"]
    assert res["warnings"] == ["Model may not fit"]


def test_effective_port_from_status_needs_no_config_call(shift):
    fake = shift(_status(mode="single_port", effective=True), master_port=1234)
    ports = {m["id"]: m["port"] for m in ls.get_llamashift_models()}
    assert ports == {"qwen27": "19000", "granite_8b_q4": "19000"}
    assert "/api/config" not in fake.paths
    assert ls.ctx_size_for_port(19000) == 65152


def test_wake_result_always_carries_warnings(shift, monkeypatch):
    shift(_status(mode="single_port"))
    monkeypatch.setattr("myagy.config.COMPACTOR_MODEL_ID", "granite_8b_q4")
    assert ls.wake_compactor(9002, timeout=1)["warnings"] == []


def test_refresh_endpoints_explains_unauthorized(monkeypatch, capsys):
    from myagy.core import session as sess_mod
    monkeypatch.setattr(sess_mod, "shift_state", lambda: "unauthorized")
    monkeypatch.setattr(sess_mod, "get_llamashift_active", lambda: [])
    offline = {"model_path": "", "filename": "x", "name": "x", "status": "offline", "source": "none", "port": 0, "gpu": ""}
    monkeypatch.setattr(sess_mod, "detect_port_model", lambda *a, **k: offline)
    s = sess_mod.MultiGpuHybridSession(enable_mcp=False, json_output=False)
    s.refresh_endpoints()
    out = capsys.readouterr()
    assert "LLAMASHIFT_API_KEY" in out.out + out.err


def test_compactor_start_command_shows_vram_warnings(monkeypatch, capsys):
    from myagy.terminal import cli

    monkeypatch.setattr(cli, "wake_compactor", lambda port=9002, **k: {
        "ok": True, "reason": "", "log_tail": "", "elapsed": 2.0, "warnings": ["Model may not fit: 9 GB needed"]})

    class Fake:
        json_output = False
        endpoints = {}

        def refresh_endpoints(self):
            return {}

    asyncio.run(cli.COMMANDS["/compactor"](Fake(), "start", None))
    out = capsys.readouterr()
    assert "Model may not fit" in out.out + out.err


def test_ensure_ready_waits_for_status_when_old_model_still_answers_health(shift):
    fake = shift(_status(running=("qwen27",)), running_after=1.5)
    t0 = time.monotonic()
    res = ls.ensure_ready("granite_8b_q4", fake.port, timeout=10)
    assert res["ok"]
    assert time.monotonic() - t0 >= 1.4


def test_draining_swap_is_not_mistaken_for_a_crash(shift, monkeypatch):
    fake = shift(_status(running=("qwen27",)), start_delay=0.8, running_after=4.5)
    monkeypatch.setattr(ls, "START_TIMEOUT", 0.3)
    res = ls.ensure_ready("granite_8b_q4", fake.port, timeout=15)
    assert res["ok"], res


def test_real_crash_is_still_detected(shift):
    fake = shift(_status(running=("qwen27",)), health_ok=False, running_after=999)
    res = ls.ensure_ready("granite_8b_q4", fake.port, timeout=8)
    assert not res["ok"] and "exited" in res["reason"]


def _wait_for(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = cond()
        if value:
            return value
        time.sleep(0.05)
    return None


def test_gpu_vram_parses_both_vendors(shift):
    shift()
    gpus = ls.get_gpu_vram()
    assert [(g["vendor"], g["used_mib"], g["total_mib"]) for g in gpus] == [("nvidia", 2970.0, 8188.0), ("amd", 14540.0, 24560.0)]


def test_gpu_vram_skips_missing_and_zero_total(shift):
    shift(gpu={"nvidia": None, "amd": {"name": "x", "mem_used": 1, "mem_total": 0}})
    assert ls.get_gpu_vram() is None


def test_vram_snapshot_never_blocks_and_fills_in_the_background(shift):
    shift(gpu_delay=0.6)
    t0 = time.monotonic()
    assert ls.vram_snapshot() is None
    assert time.monotonic() - t0 < 0.3
    data = _wait_for(ls.vram_snapshot)
    assert data and {g["vendor"] for g in data} == {"nvidia", "amd"}


def test_vram_snapshot_is_cached_between_calls(shift):
    fake = shift()
    _wait_for(ls.vram_snapshot)
    for _ in range(5):
        ls.vram_snapshot()
    assert sum(1 for p in fake.paths if p.startswith("/api/gpu")) == 1


def test_vram_snapshot_is_none_when_llamashift_is_down(monkeypatch):
    ls.invalidate_cache()
    monkeypatch.setattr(ls, "DEFAULT_LLAMASHIFT_URL", "http://127.0.0.1:9")
    ls.vram_snapshot()
    time.sleep(0.4)
    assert ls.vram_snapshot() is None
    ls.invalidate_cache()


def test_vram_snapshot_drops_data_that_is_too_old(shift, monkeypatch):
    shift()
    assert _wait_for(ls.vram_snapshot)
    monkeypatch.setattr(ls, "GPU_MAX_AGE", 0.0)
    monkeypatch.setitem(ls._gpu_cache, "refreshing", True)
    assert ls.vram_snapshot() is None


GPUS = [
    {"vendor": "nvidia", "name": "RTX 4060", "used_mib": 2970.0, "total_mib": 8188.0},
    {"vendor": "amd", "name": "R9700", "used_mib": 14540.0, "total_mib": 24560.0},
]


def _toolbar_text(session, monkeypatch, target, gpus=GPUS):
    from myagy.core import session as sess_mod
    monkeypatch.setattr(sess_mod, "vram_snapshot", lambda: gpus)
    session.active_target = target
    return "".join(t for _, t in session.toolbar_fragments())


def test_toolbar_shows_rocm_vram_for_9000(session, monkeypatch):
    text = _toolbar_text(session, monkeypatch, "9000")
    assert "vram 14.2/24G" in text and "2.9" not in text


def test_toolbar_shows_cuda_vram_for_9002(session, monkeypatch):
    assert "vram 2.9/8.0G" in _toolbar_text(session, monkeypatch, "9002")


def test_toolbar_shows_both_gpus_under_auto(session, monkeypatch):
    text = _toolbar_text(session, monkeypatch, "auto")
    assert "rocm 14.2/24G" in text and "cuda 2.9/8.0G" in text
    assert text.index("rocm") < text.index("cuda")


def test_toolbar_has_no_vram_for_cloud_or_when_unknown(session, monkeypatch):
    assert "vram" not in _toolbar_text(session, monkeypatch, "cloud")
    assert "vram" not in _toolbar_text(session, monkeypatch, "9000", gpus=None)
