import os

from myagy.core.checkpoints import Checkpoints


def test_snapshot_diff_and_restore(git_repo):
    root = Checkpoints.repo_root(str(git_repo))
    before = Checkpoints.snapshot(root)
    (git_repo / "a.txt").write_text("changed\n")
    (git_repo / "new.py").write_text("print(1)\n")
    after = Checkpoints.snapshot(root)
    assert before and after and before != after
    assert Checkpoints.changed(root, before, after) == {"a.txt": "M", "new.py": "A"}
    assert "+changed" in Checkpoints.diff(root, before, after)

    touched = Checkpoints.restore(root, before, after)
    assert sorted(touched) == ["a.txt", "new.py"]
    assert (git_repo / "a.txt").read_text() == "a\n"
    assert not (git_repo / "new.py").exists()


def test_restore_deleted_file(git_repo):
    root = Checkpoints.repo_root(str(git_repo))
    before = Checkpoints.snapshot(root)
    os.remove(git_repo / "a.txt")
    after = Checkpoints.snapshot(root)
    assert Checkpoints.changed(root, before, after) == {"a.txt": "D"}
    Checkpoints.restore(root, before, after)
    assert (git_repo / "a.txt").read_text() == "a\n"


def test_snapshot_does_not_touch_real_index(git_repo):
    import subprocess
    root = Checkpoints.repo_root(str(git_repo))
    (git_repo / "untracked.txt").write_text("x")
    Checkpoints.snapshot(root)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=git_repo, capture_output=True, text=True).stdout
    assert "?? untracked.txt" in status


def test_not_a_repo(tmp_path):
    assert Checkpoints.repo_root(str(tmp_path)) is None


def test_git_checkpointer_shadow_index_diff_and_undo(git_repo):
    import subprocess
    from myagy.core.checkpoints import GitCheckpointer, CheckpointRecord

    checkpointer = GitCheckpointer(repo_root=git_repo)
    assert checkpointer._shadow_dir.exists()

    # Pre-turn snapshot
    checkpointer.snapshot_pre_turn()
    assert checkpointer._pre_turn_tree is not None

    # Mutate files
    (git_repo / "a.txt").write_text("modified text\n")
    (git_repo / "created.py").write_text("print('hello')\n")

    # Real git index should still be untouched
    status = subprocess.run(["git", "status", "--porcelain"], cwd=git_repo, capture_output=True, text=True).stdout
    assert "M a.txt" in status or " M a.txt" in status
    assert "?? created.py" in status

    # Post-turn commit snapshot
    changed_count = checkpointer.commit_post_turn_snapshot()
    assert changed_count == 2
    assert len(checkpointer.checkpoints) == 1

    chk = checkpointer.checkpoints[0]
    assert chk.turn_id == 1
    assert sorted(chk.files_changed) == ["a.txt", "created.py"]

    # Verify git ref anchor was written
    ref_check = subprocess.run(["git", "rev-parse", "refs/myagy/checkpoints/turn_1"], cwd=git_repo, capture_output=True, text=True)
    assert ref_check.returncode == 0
    assert ref_check.stdout.strip() == chk.tree_sha

    # Verify get_diff
    diff_text = checkpointer.get_diff(nth_latest=1)
    assert "Turn 1 Diff (2 files changed)" in diff_text
    assert "modified text" in diff_text

    # Test export_stack and restore_stack
    stack = checkpointer.export_stack()
    assert len(stack) == 1
    assert stack[0]["turn_id"] == 1

    new_checkpointer = GitCheckpointer(repo_root=git_repo)
    new_checkpointer.restore_stack(stack)
    assert len(new_checkpointer.checkpoints) == 1
    assert new_checkpointer._turn_counter == 1
    new_checkpointer.close()

    # Test undo_latest
    success, msg, files = checkpointer.undo_latest()
    assert success is True
    assert "Successfully reverted Turn 1" in msg
    assert sorted(files) == ["a.txt", "created.py"]

    # Verify files restored
    assert (git_repo / "a.txt").read_text() == "a\n"
    assert not (git_repo / "created.py").exists()

    # Verify ref anchor removed
    ref_gone = subprocess.run(["git", "rev-parse", "refs/myagy/checkpoints/turn_1"], cwd=git_repo, capture_output=True, text=True)
    assert ref_gone.returncode != 0

    checkpointer.close()
    assert not checkpointer._shadow_dir.exists()

