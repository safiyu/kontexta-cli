"""Per-turn working-tree checkpoints (git tree snapshots) powering /diff and /undo."""
from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class CheckpointRecord:
    turn_id: int
    tree_sha: str
    parent_tree_sha: str
    files_changed: list[str]


class GitCheckpointer:
    """
    Manages per-turn git tree snapshots using shadow indexes (GIT_INDEX_FILE)
    to support instant /diff and /undo without altering .git/index.
    """

    def __init__(self, repo_root: Path | str | None = None) -> None:
        self.repo_root = Path(repo_root or os.getcwd()).resolve()
        self._shadow_dir = Path(tempfile.mkdtemp(prefix="myagy_shadow_idx_"))
        self._shadow_index = self._shadow_dir / "index"

        self.checkpoints: list[CheckpointRecord] = []
        self._pre_turn_tree: str | None = None
        self._turn_counter = 0

    def close(self) -> None:
        """Clean up temporary shadow index directory on shutdown."""
        try:
            for child in self._shadow_dir.glob("*"):
                child.unlink(missing_ok=True)
            self._shadow_dir.rmdir()
        except OSError:
            pass

    def _run_git(self, args: list[str]) -> tuple[str, str, int]:
        """Runs a git command with GIT_INDEX_FILE pointed to the shadow index."""
        env = os.environ.copy()
        env["GIT_INDEX_FILE"] = str(self._shadow_index)

        res = subprocess.run(
            ["git", *args],
            cwd=str(self.repo_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return res.stdout.strip(), res.stderr.strip(), res.returncode

    def snapshot_pre_turn(self) -> None:
        """
        Called at the start of a turn (PreTurn hook).
        Populates the shadow index from HEAD and writes the starting tree SHA.
        """
        # 1. Initialize shadow index from HEAD (or empty if brand new repo)
        _, _, code = self._run_git(["rev-parse", "--verify", "HEAD"])
        if code == 0:
            self._run_git(["read-tree", "HEAD"])
        else:
            self._shadow_index.unlink(missing_ok=True)

        # 2. Add all current working directory state to shadow index
        self._run_git(["add", "-A"])

        # 3. Write tree object for pre-turn state
        stdout, _, code = self._run_git(["write-tree"])
        self._pre_turn_tree = stdout if code == 0 else None

    def commit_post_turn_snapshot(self) -> int:
        """
        Called at the end of a turn (PostTurn).
        Snapshots the new state, computes changed files, and stores checkpoint.
        """
        if self._pre_turn_tree is None:
            return 0

        # Refresh shadow index with post-turn working tree state
        self._run_git(["add", "-A"])
        post_tree, _, code = self._run_git(["write-tree"])
        if code != 0 or post_tree == self._pre_turn_tree:
            return 0  # No file changes occurred this turn

        # Determine exact list of changed files between pre and post trees
        stdout, _, _ = self._run_git(["diff-tree", "--no-commit-id", "--name-only", "-r", self._pre_turn_tree, post_tree])
        files_changed = [line.strip() for line in stdout.splitlines() if line.strip()]

        if not files_changed:
            return 0

        self._turn_counter += 1
        record = CheckpointRecord(
            turn_id=self._turn_counter,
            tree_sha=post_tree,
            parent_tree_sha=self._pre_turn_tree,
            files_changed=files_changed,
        )
        self.checkpoints.append(record)

        # Anchor tree sha under refs/myagy/checkpoints/ to prevent git gc pruning
        self._run_git(["update-ref", f"refs/myagy/checkpoints/turn_{self._turn_counter}", post_tree])

        return len(files_changed)

    # -------------------------------------------------------------------------
    # /diff [n] and /undo Implementation
    # -------------------------------------------------------------------------

    def get_diff(self, nth_latest: int = 1) -> str:
        """Implements `/diff [n]` — shows file changes from the nth latest turn."""
        if not self.checkpoints:
            return "No file modifications recorded in current session."

        idx = max(0, len(self.checkpoints) - nth_latest)
        chk = self.checkpoints[idx]

        stdout, _, code = self._run_git([
            "diff", "--color=always", chk.parent_tree_sha, chk.tree_sha
        ])
        if code != 0 or not stdout:
            return f"No diff available for turn {chk.turn_id}."

        header = f"[dim]── Turn {chk.turn_id} Diff ({len(chk.files_changed)} files changed) ──[/dim]\n"
        return header + stdout

    def undo_latest(self) -> tuple[bool, str, list[str]]:
        """
        Implements `/undo` — reverts working directory and index to the
        parent tree of the latest turn checkpoint.
        Returns: (success_bool, message, files_reverted)
        """
        if not self.checkpoints:
            return False, "Nothing to undo (no checkpoints in session stack).", []

        chk = self.checkpoints.pop()

        # Revert working tree files to parent_tree_sha state
        # 1. Read parent tree into shadow index
        _, err, code = self._run_git(["read-tree", chk.parent_tree_sha])
        if code != 0:
            return False, f"Failed to read parent git tree: {err}", []

        # 2. Checkout shadow index into working directory (-f overwrites modifications)
        _, err, code = self._run_git(["checkout-index", "-a", "-f"])
        if code != 0:
            return False, f"Failed to checkout working tree: {err}", []

        # 3. Clean up newly created files that did not exist in parent tree
        stdout, _, _ = self._run_git(["diff-tree", "--no-commit-id", "--name-status", "-r", chk.parent_tree_sha, chk.tree_sha])
        for line in stdout.splitlines():
            parts = line.strip().split("\t", 1)
            if len(parts) == 2 and parts[0] == "A":
                added_file = self.repo_root / parts[1]
                if added_file.is_file():
                    try:
                        added_file.unlink()
                    except OSError:
                        pass

        # 4. Clean up the git ref anchor
        self._run_git(["update-ref", "-d", f"refs/myagy/checkpoints/turn_{chk.turn_id}"])

        msg = f"Successfully reverted Turn {chk.turn_id} ({len(chk.files_changed)} files restored)."
        return True, msg, chk.files_changed

    # -------------------------------------------------------------------------
    # /reload State Persistence Serialization
    # -------------------------------------------------------------------------

    def export_stack(self) -> list[dict[str, Any]]:
        return [
            {
                "turn_id": c.turn_id,
                "tree_sha": c.tree_sha,
                "parent_tree_sha": c.parent_tree_sha,
                "files_changed": c.files_changed,
            }
            for c in self.checkpoints
        ]

    def restore_stack(self, data: list[dict[str, Any]]) -> None:
        self.checkpoints = [
            CheckpointRecord(
                turn_id=d["turn_id"],
                tree_sha=d["tree_sha"],
                parent_tree_sha=d["parent_tree_sha"],
                files_changed=d["files_changed"],
            )
            for d in data
            if isinstance(d, dict) and "turn_id" in d
        ]
        if self.checkpoints:
            self._turn_counter = self.checkpoints[-1].turn_id


# -----------------------------------------------------------------------------
# Backward-Compatible Static API for Legacy Checkpoints
# -----------------------------------------------------------------------------

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
