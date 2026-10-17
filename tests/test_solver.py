from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from arc_solver.grids import as_grid
from arc_solver.inference import solve_task
from arc_solver.local_model import LocalQwenProgramProposer, find_local_qwen_model
from arc_solver.operations import build_operations
from arc_solver.proposer import (
    ProgramProposal,
    build_program_prompt,
    parse_program_proposal,
    parse_program_proposals,
)
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

    def test_parse_multiple_candidate_programs_and_fill_missing_slots(self) -> None:
        proposals = parse_program_proposals(
            '{"programs":[["identity"],["rotate_90"]]}',
            {"identity", "rotate_90"},
            count=3,
        )
        self.assertEqual(proposals[0].operation_names, ("identity",))
        self.assertEqual(proposals[1].operation_names, ("rotate_90",))
        self.assertIn("2 programs; 3 were requested", proposals[2].error or "")

    def test_parse_programs_from_truncated_response(self) -> None:
        proposals = parse_program_proposals(
            '{"programs":[["identity"],["rotate_90"]],"calibration_output":["1"],'
            '"test_outputs":[["2"]',
            {"identity", "rotate_90"},
            count=2,
            test_count=1,
            require_calibration=True,
        )
        self.assertEqual(proposals[0].operation_names, ("identity",))
        self.assertEqual(proposals[1].operation_names, ("rotate_90",))
        self.assertEqual(proposals[0].calibration_output, ((1,),))
        self.assertIn("test_outputs list", proposals[0].direct_output_error or "")

    def test_parse_calibration_and_direct_test_grids(self) -> None:
        proposals = parse_program_proposals(
            json.dumps(
                {
                    "programs": [["identity"]],
                    "calibration_output": ["12", "30"],
                    "test_outputs": [["45", "67"]],
                }
            ),
            {"identity"},
            count=1,
            test_count=1,
            require_calibration=True,
        )
        self.assertEqual(proposals[0].calibration_output, ((1, 2), (3, 0)))
        self.assertEqual(proposals[0].direct_test_outputs, (((4, 5), (6, 7)),))
        self.assertIsNone(proposals[0].direct_output_error)

    def test_prompt_and_local_model_discovery_are_offline_safe(self) -> None:
        task = {
            "train": [{"input": [[1]], "output": [[1]]}],
            "test": [{"input": [[2]]}],
        }
        operations = build_operations([(((1,),), ((1,),))])
        prompt = build_program_prompt(task, operations)
        self.assertIn("ALLOWED_OPERATIONS=", prompt)
        self.assertIn("Do not write or execute code", prompt)
        self.assertIn('"programs":', prompt)
        calibrated_prompt = build_program_prompt(
            task,
            operations,
            calibration_input=((2,),),
        )
        self.assertIn("CALIBRATION_INPUT=[[2]]", calibrated_prompt)
        self.assertNotIn('"output":[[2]]', calibrated_prompt)
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary) / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(
                json.dumps({"model_type": "qwen3_moe"}),
                encoding="utf-8",
            )
            (model_dir / "model.safetensors").touch()
            self.assertEqual(find_local_qwen_model(Path(temporary)), model_dir.resolve())

    def test_local_proposer_passes_tensor_ids_to_generation(self) -> None:
        class FakeTensor:
            shape = (1, 4)

            def to(self, device):
                self.device = device
                return self

        class FakeBatchEncoding(dict):
            pass

        class FakeGenerated:
            def __getitem__(self, key):
                return "generated-token-slice"

        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.template_kwargs = kwargs
                self.prompt = messages[0]["content"]
                return "rendered chat prompt"

            def __call__(self, text, **kwargs):
                self.tokenizer_kwargs = kwargs
                self.tokenized_text = text
                return FakeBatchEncoding(
                    input_ids=FakeTensor(),
                    attention_mask=FakeTensor(),
                )

            def decode(self, token_ids, **kwargs):
                self.decoded_tokens = token_ids
                return (
                    '{"programs":[["identity"]],"calibration_output":["2"],'
                    '"test_outputs":[["2"]]}'
                )

            eos_token_id = 0

        class FakeModel:
            def get_input_embeddings(self):
                return types.SimpleNamespace(
                    weight=types.SimpleNamespace(device="cpu")
                )

            def generate(self, input_ids, **kwargs):
                self.generated_input = input_ids
                self.generate_kwargs = kwargs
                return FakeGenerated()

        fake_torch = types.ModuleType("torch")
        fake_torch.manual_seed = lambda seed: None
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        tokenizer = FakeTokenizer()
        model = FakeModel()
        proposer = LocalQwenProgramProposer(tokenizer, model)
        task = {
            "train": [
                {"input": [[1]], "output": [[1]]},
                {"input": [[2]], "output": [[2]]},
            ],
            "test": [{"input": [[2]]}],
        }

        with patch.dict("sys.modules", {"torch": fake_torch}):
            proposals = proposer.propose(
                task,
                build_operations([(((1,),), ((1,),))]),
                count=1,
            )

        self.assertEqual(tokenizer.template_kwargs["tokenize"], False)
        self.assertIs(tokenizer.tokenizer_kwargs["add_special_tokens"], False)
        self.assertIsInstance(model.generated_input, FakeTensor)
        self.assertIsInstance(model.generate_kwargs["attention_mask"], FakeTensor)
        self.assertIn("CALIBRATION_INPUT=[[2]]", tokenizer.prompt)
        self.assertNotIn('"output":[[2]]', tokenizer.prompt)
        self.assertEqual(tokenizer.decoded_tokens, "generated-token-slice")
        self.assertEqual(proposals[0].operation_names, ("identity",))
        self.assertEqual(proposals[0].calibration_output, ((2,),))
        self.assertEqual(proposals[0].direct_test_outputs, (((2,),),))


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

    def test_synthesizes_three_step_crop_rotate_and_tile_program(self) -> None:
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
                    ]
                }
            ],
        }
        result = solve_task(task, max_depth=3)[0]
        self.assertEqual(
            result.attempt_1,
            ((2, 9, 2, 9), (3, 1, 3, 1), (2, 9, 2, 9), (3, 1, 3, 1)),
        )

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

    def test_direct_prediction_is_used_only_after_exact_calibration(self) -> None:
        task = {
            "train": [
                {"input": [[1]], "output": [[1]]},
                {"input": [[2]], "output": [[2]]},
            ],
            "test": [{"input": [[7, 8]]}],
        }
        direct = ((8, 7),)
        verified = FixedProposer(
            ProgramProposal(
                ("identity",),
                "keep the grid",
                calibration_output=((2,),),
                direct_test_outputs=(direct,),
            )
        )
        result = solve_task(task, proposer=verified)[0]
        self.assertEqual(result.attempt_1, direct)
        self.assertTrue(result.model_calibration_passed)

        uncalibrated = FixedProposer(
            ProgramProposal(
                ("identity",),
                "keep the grid",
                calibration_output=((9,),),
                direct_test_outputs=(direct,),
            )
        )
        rejected_result = solve_task(task, proposer=uncalibrated)[0]
        self.assertNotIn(direct, (rejected_result.attempt_1, rejected_result.attempt_2))
        self.assertFalse(rejected_result.model_calibration_passed)

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


class ExtraOperationTests(unittest.TestCase):
    def _solve(self, pairs, test_input):
        task = {
            "train": [{"input": a, "output": b} for a, b in pairs],
            "test": [{"input": test_input}],
        }
        return solve_task(task)[0]

    def test_gravity_down(self):
        pairs = [
            ([[1, 0], [0, 0], [0, 2]], [[0, 0], [0, 0], [1, 2]]),
            ([[3, 0, 0], [0, 0, 0]], [[0, 0, 0], [3, 0, 0]]),
        ]
        result = self._solve(pairs, [[0, 4], [0, 0]])
        self.assertEqual(result.attempt_1, ((0, 0), (0, 4)))

    def test_fill_enclosed(self):
        ring = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
        filled = [[1, 1, 1], [1, 5, 1], [1, 1, 1]]
        ring2 = [[0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 0, 1], [0, 1, 1, 1]]
        filled2 = [[0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 5, 1], [0, 1, 1, 1]]
        result = self._solve([(ring, filled), (ring2, filled2)], [[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        self.assertEqual(result.attempt_1, tuple(tuple(r) for r in filled))

    def test_split_boolean_rule(self):
        def pair(left, right, out):
            return ([a + [5] + b for a, b in zip(left, right)], out)

        pairs = [
            pair([[1, 0], [1, 1]], [[2, 2], [0, 2]], [[3, 0], [0, 3]]),
            pair([[0, 1], [1, 0]], [[2, 0], [2, 0]], [[0, 0], [3, 0]]),
            pair([[1, 1], [0, 0]], [[0, 2], [0, 2]], [[0, 3], [0, 0]]),
        ]
        result = self._solve(pairs, [[1, 1, 5, 2, 0], [0, 1, 5, 2, 2]])
        self.assertEqual(result.attempt_1, ((3, 0), (0, 3)))


if __name__ == "__main__":
    unittest.main()
