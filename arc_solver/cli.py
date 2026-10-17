"""Command-line runner for local validation and Kaggle notebook use."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .submission import exact_match_score, make_submission


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve ARC-AGI task JSON files.")
    parser.add_argument("--challenges", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("submission.json"))
    parser.add_argument("--solutions", type=Path)
    parser.add_argument("--max-depth", type=int, default=2, choices=(1, 2, 3))
    args = parser.parse_args()

    challenges = _read_json(args.challenges)
    submission = make_submission(challenges, max_depth=args.max_depth)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(submission, stream, separators=(",", ":"))
    if args.solutions is not None:
        solved, total = exact_match_score(submission, _read_json(args.solutions))
        percentage = 100 * solved / total if total else 0.0
        print(f"Exact output grids: {solved}/{total} ({percentage:.2f}%)")


if __name__ == "__main__":
    main()
