import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from myagy.core import llamashift as ls
from myagy.agents import compactor
from myagy.agents.compactor import ContextCurator


def _status(mode="multi_port", granite_port=9001, running=("qwen27",), granite_dedicated=False):
    def model(mid, port, ctx, fname, dedicated=False):
        return {"id": mid, "name": mid.upper(), "filename": fname, "port": port, "ctxSize": ctx, "dedicated": dedicated,
                "status": "running" if mid in running else "stopped", "gpu": "GPU", "size": "8B"}
    return {"mode": mode, "models": [
        model("qwen27", 9000, 65152, "qwen27.gguf"),
        model("granite_8b_q4", granite_port, 26384, "granite-4.2-8b.gguf", granite_dedicated),
    ]}


class FakeShift:
    """Serves LlamaShift's /api/* plus a llama-server style /health on the same port."""

    def __init__(self, status, health_503_first=0, health_ok=True, master_port=9000, logs=None):
        self.status, self.paths = status, []
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

            def do_GET(self):
                outer.paths.append(self.path)
                if self.path.startswith("/api/status"):
                    self._send(200, outer.status)
                elif self.path.startswith("/api/config"):
                    self._send(200, outer.config)
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
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if body.get("model") == "missing":
                    self._send(400, {"error": "Invalid or missing 'model' parameter"})
                else:
                    self._send(200, {"success": True})

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
    assert ls.ctx_size_for_port(9001) is None
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
    res = ls.wake_compactor(9001, timeout=2)
    assert not res["ok"] and "single_port" in res["reason"] and "dedicated" in res["reason"]
    assert "/api/start" not in fake.paths


def test_wake_allowed_for_dedicated_model_in_single_port(shift):
    shift(_status(mode="single_port", granite_dedicated=True))
    assert ls.can_wake_secondary("granite_8b_q4", 9001) == (True, "")


def test_wake_refused_when_profile_is_on_another_port(shift, monkeypatch):
    shift(_status(granite_port=9005))
    monkeypatch.setattr("myagy.config.COMPACTOR_MODEL_ID", "granite_8b_q4")
    res = ls.wake_compactor(9001, timeout=2)
    assert not res["ok"] and "9005" in res["reason"]


def test_wake_refused_when_no_profile_matches(shift, monkeypatch):
    shift()
    monkeypatch.setattr("myagy.config.COMPACTOR_MODEL_ID", "nosuchmodel")
    res = ls.wake_compactor(9001, timeout=2)
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
