"""Tests for apply_solution.py, on a small fake repository in a temporary folder."""

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "apply_solution", Path(__file__).resolve().parent.parent / "apply_solution.py")
apply_solution_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(apply_solution_module)
apply_solution = apply_solution_module.apply_solution


def fake_repo(tmp_path):
    (tmp_path / "mdsim").mkdir()
    (tmp_path / "mdsim" / "forces_numpy.py").write_text("student version\n")
    (tmp_path / "mdsim" / "system.py").write_text("untouched\n")
    solution = tmp_path / "solutions" / "problem_2a" / "mdsim"
    solution.mkdir(parents=True)
    (solution / "forces_numpy.py").write_text("solution\n")
    return tmp_path


def never_asked(prompt):
    raise AssertionError("should not ask")


def test_copies_solution_over_the_file(tmp_path):
    root = fake_repo(tmp_path)
    copied = apply_solution("2a", root=root, ask=never_asked)
    assert copied == [Path("mdsim/forces_numpy.py")]
    assert (root / "mdsim" / "forces_numpy.py").read_text() == "solution\n"
    assert (root / "mdsim" / "system.py").read_text() == "untouched\n"


def test_second_run_does_nothing(tmp_path):
    root = fake_repo(tmp_path)
    apply_solution("2a", root=root, ask=never_asked)
    assert apply_solution("2a", root=root, ask=never_asked) == []


def test_missing_solution_stops_with_a_message(tmp_path):
    root = fake_repo(tmp_path)
    with pytest.raises(SystemExit, match="git pull"):
        apply_solution("2b", root=root, ask=never_asked)


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_asks_before_replacing_uncommitted_work(tmp_path):
    root = fake_repo(tmp_path)
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], cwd=root, check=True)
    subprocess.run([*git, "add", "."], cwd=root, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "start"], cwd=root, check=True)
    (root / "mdsim" / "forces_numpy.py").write_text("work in progress\n")

    with pytest.raises(SystemExit, match="Nothing was changed"):
        apply_solution("2a", root=root, ask=lambda prompt: "n")
    assert (root / "mdsim" / "forces_numpy.py").read_text() == "work in progress\n"

    apply_solution("2a", root=root, ask=lambda prompt: "y")
    assert (root / "mdsim" / "forces_numpy.py").read_text() == "solution\n"
