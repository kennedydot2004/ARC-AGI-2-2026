"""Kaggle challenge-to-submission adapter and exact-match scoring helpers."""

from __future__ import annotations

from typing import Any

from .grids import Grid, as_grid, to_json_grid
from .inference import solve_task
from .proposer import ProgramProposer


def make_submission(
    challenges: dict[str, Any],
    max_depth: int = 2,
    proposer: ProgramProposer | None = None,
    proposal_count: int = 3,
    diagnostics: list[dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    submission: dict[str, list[dict[str, Any]]] = {}
    for task_id, task in challenges.items():
        predictions = solve_task(
            task,
            max_depth=max_depth,
            proposer=proposer,
            proposal_count=proposal_count,
        )
        if diagnostics is not None:
            first = predictions[0]
            diagnostics.append(
                {
                    "task_id": task_id,
                    "consistent_programs": first.consistent_program_count,
                    "model_proposals": first.model_proposal_count,
                    "model_accepted": first.model_accepted_count,
                    "model_rejected": first.model_rejected_count,
                    "rejection_reasons": first.model_rejection_reasons,
                }
            )
        submission[task_id] = [
            {
                "attempt_1": to_json_grid(prediction.attempt_1),
                "attempt_2": to_json_grid(prediction.attempt_2),
            }
            for prediction in predictions
        ]
    return submission


def exact_match_score(
    submission: dict[str, list[dict[str, Any]]],
    solutions: dict[str, Any],
) -> tuple[int, int]:
    """Return number of solved output grids and total output grids."""
    solved = 0
    total = 0
    for task_id, expected_values in solutions.items():
        predictions = submission[task_id]
        if isinstance(expected_values, dict):
            expected_values = expected_values.get("test", expected_values)
        if not isinstance(expected_values, list) or len(expected_values) != len(predictions):
            raise ValueError(f"Solution/prediction count mismatch for task {task_id}.")
        for expected, prediction in zip(expected_values, predictions):
            target: Grid = as_grid(expected)
            total += 1
            if (
                as_grid(prediction["attempt_1"]) == target
                or as_grid(prediction["attempt_2"]) == target
            ):
                solved += 1
    return solved, total
