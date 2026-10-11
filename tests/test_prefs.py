import json

from myagy.core.prefs import PREF_DEFAULTS, load_prefs, save_prefs, reset_prefs


def test_round_trip(tmp_path):
    path = str(tmp_path / "cfg" / "config.json")
    assert save_prefs({"verbose": True, "autocompact_threshold": 7, "bogus": 1}, path)
    assert load_prefs(path) == {"verbose": True, "autocompact_threshold": 7}


def test_wrong_types_and_garbage_ignored(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"verbose": "yes", "show_metrics": False, "unknown": 5}))
    assert load_prefs(str(path)) == {"show_metrics": False}
    path.write_text("not json")
    assert load_prefs(str(path)) == {}
    assert load_prefs(str(tmp_path / "missing.json")) == {}


def test_reset_removes_file(tmp_path):
    path = str(tmp_path / "config.json")
    save_prefs({"verbose": True}, path)
    reset_prefs(path)
    assert load_prefs(path) == {}
    reset_prefs(path)  # idempotent


def test_session_apply_and_snapshot(session):
    session.apply_preferences({
        "verbose": True,
        "autocompact_threshold": 5,
        "dangerously_skip_permissions": False,
        "active_target": "cloud",
        "cloud_model": "gemini-2.5-pro",
        "nope": 1,
    })
    prefs = session.preferences()
    assert prefs["verbose"] is True and prefs["autocompact_threshold"] == 5
    assert prefs["dangerously_skip_permissions"] is False
    assert prefs["active_target"] == "cloud"
    assert prefs["cloud_model"] == "gemini-2.5-pro"
    assert session.cloud_model == "gemini-2.5-pro"
    assert set(prefs) <= set(PREF_DEFAULTS)

