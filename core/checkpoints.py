"""Per-turn working-tree checkpoints (git tree snapshots) powering /diff and /undo."""

import os
import subprocess
import tempfile
from typing import Dict, List, Optional


class Checkpoints:
    """Snapshots the working tree as git tree objects without touching the real index."""

    MAX_TURNS = 20
    MAX_UNTRACKED_BYTES = 5 * 1024 * 1024

    @staticmethod
    def _git(args: List[str], cwd: str, env: Optional[dict] = None, check: bool = True) -> str:
        res = subprocess.run(
            ["git", *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=60,
        )
        if check and res.returncode != 0:
            raise RuntimeError(res.stderr.strip() or f"git {' '.join(args)} failed")
        return res.stdout

    @classmethod
    def repo_root(cls, cwd: str = ".") -> Optional[str]:
        try:
            return cls._git(["rev-parse", "--show-toplevel"], cwd).strip() or None
        except Exception:
            return None

    @classmethod
    def _oversized_untracked(cls, root: str) -> List[str]:
        """Untracked, non-ignored files above the size cap; snapshotting them would bloat .git/objects."""
        out = cls._git(["ls-files", "--others", "--exclude-standard", "-z"], root)
        big = []
        for rel in out.split("\0"):
            if not rel:
                continue
            try:
                if os.path.getsize(os.path.join(root, rel)) > cls.MAX_UNTRACKED_BYTES:
                    big.append(rel)
            except OSError:
                pass
        return big

    @classmethod
    def snapshot(cls, root: str) -> Optional[str]:
        """Returns a tree hash of the current working tree (tracked + untracked, not ignored)."""
        tmpdir = tempfile.mkdtemp(prefix="myagy-idx-")
        env = dict(os.environ, GIT_INDEX_FILE=os.path.join(tmpdir, "index"))
        try:
            excludes = [f":(exclude,literal){p}" for p in cls._oversized_untracked(root)]
            cls._git(["add", "-A", "--", ".", *excludes], root, env)
            return cls._git(["write-tree"], root, env).strip() or None
        except Exception:
            return None
        finally:
            try:
                for fn in os.listdir(tmpdir):
                    os.remove(os.path.join(tmpdir, fn))
                os.rmdir(tmpdir)
            except OSError:
                pass

    @classmethod
    def changed(cls, root: str, before: str, after: str) -> Dict[str, str]:
        """Maps path -> status letter (A/M/D) between two snapshots."""
        out = cls._git(["diff", "--name-status", "--no-renames", before, after], root)
        result: Dict[str, str] = {}
        for line in out.splitlines():
            status, _, path = line.partition("\t")
            if path:
                result[path] = status[:1]
        return result

    @classmethod
    def diff(cls, root: str, before: str, after: str, stat: bool = False, numstat: bool = False) -> str:
        args = ["diff", "--no-renames", "--no-color"]
        if stat:
            args.append("--stat")
        if numstat:
            args.append("--numstat")
        return cls._git([*args, before, after], root)

    @classmethod
    def restore(cls, root: str, before: str, current: str) -> List[str]:
        """Reverts the working tree from `current` back to `before`; returns touched paths."""
        touched = []
        for path, status in cls.changed(root, before, current).items():
            full = os.path.join(root, path)
            if status == "A":
                try:
                    os.remove(full)
                    touched.append(path)
                except OSError:
                    pass
            else:
                cls._git(["restore", f"--source={before}", "--worktree", "--", path], root)
                touched.append(path)
        return touched
