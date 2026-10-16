from __future__ import annotations

import json
import unittest

from arc_solver.grids import as_grid
from arc_solver.inference import solve_task
from arc_solver.submission import exact_match_score, make_submission


class GridTests(unittest.TestCase):
    def test_grid_validation_checks_shape_and_colors(self) -> None:
        self.assertEqual(as_grid([[0, 1], [2, 3]]), ((0, 1), (2, 3)))
        with self.assertRaises(ValueError):
            as_grid([[0, 1], [2]])
        with self.assertRaises(ValueError):
            as_grid([[10]])
        with self.assertRaises(ValueError):
            as_grid([[True]])


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
