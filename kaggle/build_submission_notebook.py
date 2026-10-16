"""Build a self-contained Kaggle notebook from the maintained solver package."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "arc_solver"
OUTPUT = ROOT / "kaggle" / "arc_prize_2026_submission.ipynb"
MODULES = ("__init__.py", "grids.py", "operations.py", "inference.py", "submission.py")


def build_notebook() -> None:
    sources = {
        name: (PACKAGE / name).read_text(encoding="utf-8")
        for name in MODULES
    }
    bootstrap = (
        "from pathlib import Path\n"
        "import sys\n\n"
        "PACKAGE_DIR = Path('/kaggle/working/arc_solver')\n"
        "PACKAGE_DIR.mkdir(parents=True, exist_ok=True)\n"
        f"SOURCES = {sources!r}\n"
        "for filename, source in SOURCES.items():\n"
        "    (PACKAGE_DIR / filename).write_text(source, encoding='utf-8')\n"
        "sys.path.insert(0, '/kaggle/working')\n"
    )
    run_submission = (
        "import json\n"
        "from pathlib import Path\n"
        "from arc_solver.submission import make_submission\n\n"
        "input_files = list(Path('/kaggle/input').rglob('arc-agi_test_challenges.json'))\n"
        "if len(input_files) != 1:\n"
        "    raise FileNotFoundError(\n"
        "        f'Expected exactly one arc-agi_test_challenges.json; found {len(input_files)}.'\n"
        "    )\n"
        "with input_files[0].open(encoding='utf-8') as stream:\n"
        "    challenges = json.load(stream)\n"
        "submission = make_submission(challenges)\n"
        "submission_path = Path('/kaggle/working/submission.json')\n"
        "with submission_path.open('w', encoding='utf-8') as stream:\n"
        "    json.dump(submission, stream, separators=(',', ':'))\n"
        "print(f'Wrote {len(submission)} task predictions to {submission_path}')\n"
    )
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# ARC Prize 2026 - ARC-AGI-2\n",
                    "\n",
                    "Offline, task-level symbolic program synthesis. The source package is embedded by `kaggle/build_submission_notebook.py` so this notebook is self-contained when submitted.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": bootstrap.splitlines(keepends=True),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": run_submission.splitlines(keepends=True),
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"Built {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    build_notebook()
