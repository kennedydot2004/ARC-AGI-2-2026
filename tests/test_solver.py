from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from arc_solver.grids import as_grid
from arc_solver.inference import solve_task
from arc_solver.local_model import find_local_qwen_model
from arc_solver.operations import build_operations
from arc_solver.proposer import ProgramProposal, build_program_prompt, parse_program_proposal
from arc_solver.submission import exact_match_score, make_submission


class FixedProposer:
    def __init__(self, *proposals: ProgramProposal) -> None:
        self.proposals = list(proposals)

    def propose(self, task, operations, count):
        self.task = task
        self.operations = operations
        return self.proposals[:count]


class GridTests(unittest.TestCase):
    def test_grid_validation_checks_shape_and_colors(self) -> None:
        self.assertEqual(as_grid([[0, 1], [2, 3]]), ((0, 1), (2, 3)))
        with self.assertRaises(ValueError):
            as_grid([[0, 1], [2]])
        with self.assertRaises(ValueError):
            as_grid([[10]])
        with self.assertRaises(ValueError):
            as_grid([[True]])

    def test_parse_program_proposal_rejects_unknown_operations(self) -> None:
        proposal = parse_program_proposal(
            'Reasoning... {"program":["rotate_90","run_code"],"rationale":"test"}',
            {"identity", "rotate_90"},
        )
        self.assertIsNone(proposal.operation_names)
        self.assertIn("Unknown operation", proposal.error or "")

    def test_prompt_and_local_model_discovery_are_offline_safe(self) -> None:
        task = {
            "train": [{"input": [[1]], "output": [[1]]}],
            "test": [{"input": [[2]]}],
        }
        operations = build_operations([(((1,),), ((1,),))])
        prompt = build_program_prompt(task, operations)
        self.assertIn("ALLOWED_OPERATIONS=", prompt)
        self.assertIn("Do not write or execute code", prompt)
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary) / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(
                json.dumps({"model_type": "qwen3_moe"}),
                encoding="utf-8",
            )
            (model_dir / "model.safetensors").touch()
            self.assertEqual(find_local_qwen_model(Path(temporary)), model_dir.resolve())


class SolverTests(unittest.TestCase):
    def test_infers_rotation_from_all_demonstrations(self) -> None:
        task = {
            "train": [
                {"input": [[1, 0], [2, 3]], "output": [[2, 1], [3, 0]]},
                {"input": [[4, 5], [6, 7]], "output": [[6, 4], [7, 5]]},
            ],
            "test": [{"input": [[1, 2], [3, 4]]}],
        }
        result = solve_task(task)[0]
        self.assertEqual(result.attempt_1, ((3, 1), (4, 2)))
        self.assertGreater(result.consistent_program_count, 0)

    def test_infers_color_mapping_and_preserves_unseen_colors(self) -> None:
        task = {
            "train": [
                {"input": [[1, 2], [2, 1]], "output": [[3, 4], [4, 3]]},
                {"input": [[2, 1], [1, 2]], "output": [[4, 3], [3, 4]]},
            ],
            "test": [{"input": [[1, 2], [5, 1]]}],
        }
        result = solve_task(task)[0]
        self.assertEqual(result.attempt_1, ((3, 4), (5, 3)))

    def test_infers_translation(self) -> None:
        task = {
            "train": [
                {
                    "input": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                    "output": [[0, 0, 0], [0, 0, 1], [0, 0, 0]],
                },
                {
                    "input": [[0, 2, 0], [0, 0, 0], [0, 0, 0]],
                    "output": [[0, 0, 2], [0, 0, 0], [0, 0, 0]],
                },
            ],
            "test": [{"input": [[0, 0, 0], [3, 0, 0], [0, 0, 0]]}],
        }
        result = solve_task(task)[0]
        self.assertEqual(result.attempt_1, ((0, 0, 0), (0, 3, 0), (0, 0, 0)))

    def test_composes_crop_and_rotation(self) -> None:
        task = {
            "train": [
                {
                    "input": [[0, 0, 0], [0, 1, 2], [0, 3, 4]],
                    "output": [[3, 1], [4, 2]],
                },
                {
                    "input": [[0, 0, 0], [0, 5, 6], [0, 7, 8]],
                    "output": [[7, 5], [8, 6]],
                },
            ],
            "test": [{"input": [[0, 0, 0], [0, 2, 1], [0, 4, 3]]}],
        }
        result = solve_task(task)[0]
        self.assertEqual(result.attempt_1, ((4, 2), (3, 1)))

    def test_exact_demonstration_match_is_preserved(self) -> None:
        task = {
            "train": [{"input": [[1, 2]], "output": [[2, 1]]}],
            "test": [{"input": [[1, 2]]}],
        }
        result = solve_task(task)[0]
        self.assertEqual(result.attempt_1, ((2, 1),))
        self.assertEqual(result.program_names, ("exact_demonstration_match",))

    def test_model_proposed_three_step_program_must_fit_all_examples(self) -> None:
        task = {
            "train": [
                {
                    "input": [
                        [0, 0, 0, 0],
                        [0, 1, 2, 0],
                        [0, 3, 4, 0],
                        [0, 0, 0, 0],
                    ],
                    "output": [
                        [3, 1, 3, 1],
                        [4, 2, 4, 2],
                        [3, 1, 3, 1],
                        [4, 2, 4, 2],
                    ],
                },
                {
                    "input": [
                        [0, 0, 0, 0],
                        [0, 5, 6, 0],
                        [0, 7, 8, 0],
                        [0, 0, 0, 0],
                    ],
                    "output": [
                        [7, 5, 7, 5],
                        [8, 6, 8, 6],
                        [7, 5, 7, 5],
                        [8, 6, 8, 6],
                    ],
                },
            ],
            "test": [
                {
                    "input": [
                        [0, 0, 0, 0],
                        [0, 9, 1, 0],
                        [0, 2, 3, 0],
                        [0, 0, 0, 0],
                    ],
                    "output": [
                        [2, 9, 2, 9],
                        [3, 1, 3, 1],
                        [2, 9, 2, 9],
                        [3, 1, 3, 1],
                    ],
                }
            ],
        }
        proposer = FixedProposer(
            ProgramProposal(
                ("crop_background_0", "rotate_90", "tile_2x2"),
                "Crop, rotate, then repeat.",
            )
        )
        result = solve_task(task, proposer=proposer)[0]
        self.assertEqual(
            result.attempt_1,
            ((2, 9, 2, 9), (3, 1, 3, 1), (2, 9, 2, 9), (3, 1, 3, 1)),
        )
        self.assertEqual(result.model_proposal_count, 1)
        self.assertEqual(result.model_accepted_count, 1)
        self.assertNotIn("output", proposer.task["test"][0])

    def test_inconsistent_model_proposal_is_rejected(self) -> None:
        task = {
            "train": [
                {"input": [[1, 2], [2, 1]], "output": [[1, 2], [2, 1]]},
                {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]},
            ],
            "test": [{"input": [[3, 4], [5, 6]]}],
        }
        proposer = FixedProposer(ProgramProposal(("rotate_180",), "fits only one example"))
        result = solve_task(task, proposer=proposer)[0]
        self.assertEqual(result.attempt_1, ((3, 4), (5, 6)))
        self.assertEqual(result.model_accepted_count, 0)
        self.assertEqual(result.model_rejected_count, 1)
        self.assertIn("failed exact demonstration", result.model_rejection_reasons[0])

    def test_verified_model_program_ranks_ahead_of_symbolic_alternatives(self) -> None:
        test_input = [[7, 8]]
        task = {
            "train": [
                {"input": [[1]], "output": [[1]]},
                {"input": [[2]], "output": [[2]]},
            ],
            "test": [{"input": test_input}],
        }
        proposer = FixedProposer(ProgramProposal(("identity",), "Preserve the grid."))
        result = solve_task(task, proposer=proposer)[0]
        self.assertEqual(result.attempt_1, as_grid(test_input))
        self.assertEqual(result.model_accepted_count, 1)

    def test_submission_order_and_two_attempt_contract(self) -> None:
        challenges = {
            "task-a": {
                "train": [{"input": [[1]], "output": [[1]]}],
                "test": [{"input": [[2]]}, {"input": [[3]]}],
            }
        }
        output = make_submission(challenges)
        self.assertEqual(list(output), ["task-a"])
        self.assertEqual(len(output["task-a"]), 2)
        self.assertEqual(set(output["task-a"][0]), {"attempt_1", "attempt_2"})
        self.assertEqual(output["task-a"][0]["attempt_1"], [[2]])
        score, total = exact_match_score(output, {"task-a": [[[2]], [[0]]]})
        self.assertEqual((score, total), (1, 2))
        self.assertEqual(json.loads(json.dumps(output)), output)


if __name__ == "__main__":
    unittest.main()
