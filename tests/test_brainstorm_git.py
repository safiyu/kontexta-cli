import subprocess

from kontexta_cli.agents.brainstorm import BrainstormWorkflow as B


def test_untracked_files_appear_in_diff(git_repo):
    (git_repo / "brand_new.py").write_text("print('hi')\n")
    diff = B._get_git_diff(str(git_repo))
    assert "brand_new.py" in diff and "+print('hi')" in diff


def test_commit_selection_excludes_preexisting_dirty_files(git_repo):
    (git_repo / "pre.txt").write_text("p\n")
    subprocess.run(["git", "add", "pre.txt"], cwd=git_repo, check=True)
    subprocess.run(["git", "commit", "-qm", "pre"], cwd=git_repo, check=True)
    (git_repo / "pre.txt").write_text("dirty before run\n")
    before = B._git_paths(str(git_repo))

    (git_repo / "a.txt").write_text("agent edit\n")
    (git_repo / "created.py").write_text("x\n")
    assert sorted(B._git_paths(str(git_repo)) - before) == ["a.txt", "created.py"]


def test_detect_test_command(tmp_path):
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    assert B._detect_test_command(str(tmp_path)) == "pytest"
