"""Put the released solution of one problem into your code.

Usage, from the repository folder:

    git pull                        # get the newest solutions first
    python apply_solution.py 2      # or 3a, 3b

Every file in solutions/problem_<N>/ is copied over the file with the same
path in the repository, for example solutions/problem_2/mdsim/viz.py over
mdsim/viz.py. Your own version of those files is replaced. If you have
changed one of them and not committed it yet, the script asks first.
Afterwards check the result with the problem's test, e.g.
`pytest tests/test_problem_2.py`, and commit.
"""

import filecmp
import shutil
import subprocess
import sys
from pathlib import Path

PROBLEMS = ["2", "3a", "3b"]
ROOT = Path(__file__).resolve().parent


def solution_files(folder):
    """Paths of all files in a solution folder, relative to that folder."""
    return sorted(p.relative_to(folder) for p in folder.rglob("*")
                  if p.is_file() and "__pycache__" not in p.parts)


def uncommitted(root, paths):
    """Those of `paths` that have changes not committed yet (empty if Git is unavailable)."""
    try:
        result = subprocess.run(["git", "status", "--porcelain", "--", *map(str, paths)],
                                cwd=root, capture_output=True, text=True)
    except OSError:
        return []
    if result.returncode != 0:
        return []
    return [line[3:] for line in result.stdout.splitlines()]


def apply_solution(problem, root=ROOT, ask=input):
    """Copy solutions/problem_<problem>/ over the repository. Returns the copied paths."""
    folder = root / "solutions" / f"problem_{problem}"
    if not folder.is_dir():
        sys.exit(f"The solution of problem {problem} is not here yet ({folder.relative_to(root)} "
                 "is missing). Run `git pull` and try again; if it is still missing, it has not "
                 "been released yet.")

    files = solution_files(folder)
    changed = [rel for rel in files
               if not (root / rel).exists() or not filecmp.cmp(folder / rel, root / rel, shallow=False)]
    if not changed:
        print(f"Your files already match the solution of problem {problem}. Nothing to do.")
        return []

    not_committed = uncommitted(root, changed)
    if not_committed:
        print("These files have changes that are not committed yet; they will be replaced:")
        for path in not_committed:
            print(f"  {path}")
        if ask("Replace them anyway? [y/N] ").strip().lower() not in ("y", "yes"):
            sys.exit("Nothing was changed. Commit your work first (git add, git commit), then run this again.")

    for rel in changed:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(folder / rel, root / rel)
        print(f"  {rel.as_posix()}  <-  solutions/problem_{problem}/{rel.as_posix()}")
    print(f"Done. Check it with:  pytest tests/test_problem_{problem}.py")
    print(f'Then commit it:      git commit -am "Solution of problem {problem}"')
    return changed


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in PROBLEMS:
        sys.exit(f"Usage: python apply_solution.py <problem>, where <problem> is one of {', '.join(PROBLEMS)}")
    apply_solution(sys.argv[1])


if __name__ == "__main__":
    main()
