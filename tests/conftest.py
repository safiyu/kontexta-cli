import os
import sys
import subprocess
import importlib.util

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import sdk_stub  # noqa: E402

sdk_stub.install()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Make `import kontexta_cli` work even if the checkout directory has a different name
if importlib.util.find_spec("kontexta_cli") is None:
    spec = importlib.util.spec_from_file_location("kontexta_cli", os.path.join(ROOT, "__init__.py"), submodule_search_locations=[ROOT])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kontexta_cli"] = mod
    spec.loader.exec_module(mod)
else:
    sys.path.insert(0, os.path.dirname(ROOT))


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch, tmp_path):
    """Reset class-level singletons and keep real user hooks/prefs out of tests."""
    from kontexta_cli.core.coordinator import CudaCoordinator
    from kontexta_cli.context import hooks_loader
    from kontexta_cli.context.hooks_loader import ExternalHooksManager

    CudaCoordinator._state = "idle"
    CudaCoordinator._active_subagents = 0
    CudaCoordinator._compactor_task = None
    ExternalHooksManager._cached_configs = None
    ExternalHooksManager._cached_sources = []
    ExternalHooksManager._cached_mtime = 0.0
    monkeypatch.setattr(hooks_loader, "HOOKS_CANDIDATE_PATHS", [os.path.join(".agents", "hooks.json"), "hooks.json"])
    monkeypatch.setattr("kontexta_cli.core.session.save_prefs", lambda values, path=None: True)


@pytest.fixture
def git_repo(tmp_path, monkeypatch):
    """A temp git repo with one commit; cwd is switched into it."""
    def run(*a):
        return subprocess.run(["git", *a], cwd=tmp_path, capture_output=True, text=True, check=True)
    run("init", "-q")
    run("config", "user.email", "t@example.com")
    run("config", "user.name", "tester")
    (tmp_path / "a.txt").write_text("a\n")
    run("add", ".")
    run("commit", "-qm", "init")
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def session(monkeypatch):
    """A real MultiGpuHybridSession with network discovery disabled."""
    from kontexta_cli.core.session import MultiGpuHybridSession
    monkeypatch.setattr(MultiGpuHybridSession, "refresh_endpoints", lambda self, *a, **k: self.endpoints)
    s = MultiGpuHybridSession(enable_mcp=False, json_output=True)
    s.auto_repomap = False
    s.auto_instructions = False
    return s
