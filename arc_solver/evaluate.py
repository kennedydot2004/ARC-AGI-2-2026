"""One-pass exact-match evaluation for ARC task directories with labels."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .grids import as_grid
from .inference import solve_task
from .proposer import ProgramProposer


def evaluate_directory(
    directory: Path,
    max_depth: int = 2,
    limit: int | None = None,
    proposer: ProgramProposer | None = None,
    proposal_count: int = 3,
) -> dict[str, float | int]:
    task_files = sorted(directory.glob("*.json"))
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive.")
        task_files = task_files[:limit]
    if not task_files:
        raise FileNotFoundError(f"No task JSON files found in {directory}.")

    total_outputs = 0
    solved_outputs = 0
    total_tasks = 0
    solved_tasks = 0
    tasks_with_program = 0
    model_proposals = 0
    model_accepted = 0
    model_rejected = 0
    started = time.perf_counter()
    for path in task_files:
        with path.open(encoding="utf-8") as stream:
            task = json.load(stream)
        if any("output" not in pair for pair in task.get("test", [])):
            raise ValueError(f"Task labels are missing from {path}.")
        predictions = solve_task(
            task,
            max_depth=max_depth,
            proposer=proposer,
            proposal_count=proposal_count,
        )
        if len(predictions) != len(task["test"]):
            raise ValueError(f"Prediction/test count mismatch for {path}.")
        task_solved = True
        tasks_with_program += int(any(prediction.consistent_program_count for prediction in predictions))
        if predictions:
            model_proposals += predictions[0].model_proposal_count
            model_accepted += predictions[0].model_accepted_count
            model_rejected += predictions[0].model_rejected_count
        for prediction, test_pair in zip(predictions, task["test"]):
            target = as_grid(test_pair["output"])
            correct = prediction.attempt_1 == target or prediction.attempt_2 == target
            solved_outputs += int(correct)
            total_outputs += 1
            task_solved = task_solved and correct
        total_tasks += 1
        solved_tasks += int(task_solved)

    elapsed = time.perf_counter() - started
    return {
        "tasks": total_tasks,
        "tasks_solved": solved_tasks,
        "task_accuracy_percent": 100 * solved_tasks / total_tasks,
        "outputs": total_outputs,
        "outputs_solved": solved_outputs,
        "output_accuracy_percent": 100 * solved_outputs / total_outputs,
        "tasks_with_consistent_program": tasks_with_program,
        "model_proposals": model_proposals,
        "model_programs_accepted": model_accepted,
        "model_programs_rejected": model_rejected,
        "elapsed_seconds": round(elapsed, 2),
        "milliseconds_per_task": round(1000 * elapsed / total_tasks, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on labeled ARC JSON task files.")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--max-depth", type=int, choices=(1, 2), default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--proposals", type=int, default=3)
    args = parser.parse_args()
    proposer = None
    if args.model_path is not None:
        from .local_model import LocalQwenProgramProposer

        proposer = LocalQwenProgramProposer.from_pretrained(args.model_path)
    for key, value in evaluate_directory(
        args.directory,
        args.max_depth,
        args.limit,
        proposer,
        args.proposals,
    ).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
