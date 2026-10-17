"""Task-level program synthesis and prediction ranking."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .grids import Grid, as_grid
from .operations import Operation, build_operations
from .proposer import ProgramProposer

_SYNTHESIS_BEAM_WIDTH = 128
_PATHS_PER_STATE = 3
_EXTRA_OPERATION_COST = 4
_LEARNED_PREFIXES = ("local_rule_", "split_")


@dataclass(frozen=True)
class Program:
    operations: tuple[Operation, ...]

    @property
    def cost(self) -> int:
        return sum(operation.cost for operation in self.operations)

    @property
    def name(self) -> str:
        return " -> ".join(operation.name for operation in self.operations)

    def apply(self, grid: Grid) -> Grid | None:
        result: Grid | None = grid
        for operation in self.operations:
            if result is None:
                return None
            result = operation.apply(result)
            if result is not None and (
                not result
                or not result[0]
                or len(result) > 30
                or len(result[0]) > 30
                or any(len(row) != len(result[0]) for row in result)
                or any(cell not in range(10) for row in result for cell in row)
            ):
                return None
        return result


def _program_signature(program: Program) -> tuple[str, ...]:
    return tuple(operation.name for operation in program.operations)


@dataclass(frozen=True)
class SolveResult:
    attempt_1: Grid
    attempt_2: Grid
    program_names: tuple[str, ...]
    consistent_program_count: int
    model_proposal_count: int = 0
    model_accepted_count: int = 0
    model_rejected_count: int = 0
    model_rejection_reasons: tuple[str, ...] = ()
    model_calibration_passed: bool = False
    model_direct_output_error: str | None = None


def _beam_candidate_programs(
    examples: list[tuple[Grid, Grid]],
    max_depth: int,
    operations: list[Operation],
) -> list[Program]:
    sources = tuple(source for source, _ in examples)
    targets = tuple(target for _, target in examples)
    frontier: dict[tuple[Grid, ...], list[tuple[Operation, ...]]] = {
        sources: [()]
    }
    visited_depth = {sources: 0}

    def distance(state: tuple[Grid, ...]) -> int:
        total = 0
        for grid, target in zip(state, targets):
            height, width = len(grid), len(grid[0])
            target_height, target_width = len(target), len(target[0])
            total += 1000 * (
                abs(height - target_height) + abs(width - target_width)
            )
            total += sum(
                grid[row][column] != target[row][column]
                for row in range(min(height, target_height))
                for column in range(min(width, target_width))
            )
        return total

    def valid_grid(grid: Grid | None) -> bool:
        return (
            grid is not None
            and 0 < len(grid) <= 30
            and 0 < len(grid[0]) <= 30
            and all(len(row) == len(grid[0]) for row in grid)
            and all(cell in range(10) for row in grid for cell in row)
        )

    for depth in range(1, max_depth + 1):
        next_paths: dict[tuple[Grid, ...], list[tuple[Operation, ...]]] = defaultdict(list)
        exact_programs: dict[tuple[str, ...], Program] = {}
        for state, prefixes in frontier.items():
            for operation in operations:
                if operation.name == "identity":
                    continue
                transformed_grids = tuple(operation.apply(grid) for grid in state)
                if not all(valid_grid(grid) for grid in transformed_grids):
                    continue
                transformed = tuple(
                    grid for grid in transformed_grids if grid is not None
                )
                if transformed == state:
                    continue

                for prefix in prefixes:
                    sequence = prefix + (operation,)
                    if transformed == targets:
                        program = Program(sequence)
                        exact_programs[_program_signature(program)] = program
                        continue

                    previous_depth = visited_depth.get(transformed)
                    if previous_depth is not None and previous_depth < depth:
                        continue
                    visited_depth[transformed] = depth
                    paths = next_paths[transformed]
                    signature = tuple(item.name for item in sequence)
                    if any(
                        tuple(item.name for item in existing) == signature
                        for existing in paths
                    ):
                        continue
                    paths.append(sequence)

        if exact_programs:
            return sorted(
                exact_programs.values(),
                key=lambda program: (program.cost, program.name),
            )
        if depth == max_depth:
            break

        ranked_states = sorted(
            next_paths.items(),
            key=lambda item: (
                distance(item[0]),
                min(sum(operation.cost for operation in path) for path in item[1]),
                min(" -> ".join(operation.name for operation in path) for path in item[1]),
            ),
        )[:_SYNTHESIS_BEAM_WIDTH]
        frontier = {}
        for state, paths in ranked_states:
            frontier[state] = sorted(
                paths,
                key=lambda path: (
                    sum(operation.cost for operation in path),
                    " -> ".join(operation.name for operation in path),
                ),
            )[:_PATHS_PER_STATE]

    return []


def _candidate_programs(
    examples: list[tuple[Grid, Grid]],
    max_depth: int,
    operations: list[Operation],
) -> list[Program]:
    first_input, first_output = examples[0]
    one_step: list[tuple[Operation, Grid | None]] = [
        (operation, operation.apply(first_input)) for operation in operations
    ]
    exact_programs: list[Program] = [
        Program((operation,))
        for operation, result in one_step
        if result == first_output
    ]

    if max_depth >= 2:
        intermediates: dict[Grid, list[Operation]] = defaultdict(list)
        for operation, result in one_step:
            if (
                result is not None
                and result != first_output
                and operation.name != "identity"
            ):
                intermediates[result].append(operation)
        for intermediate, prefixes in intermediates.items():
            for suffix in operations:
                if suffix.name == "identity":
                    continue
                if suffix.apply(intermediate) == first_output:
                    exact_programs.extend(
                        Program((prefix, suffix)) for prefix in prefixes
                    )

    consistent: dict[tuple[str, ...], Program] = {}
    for program in exact_programs:
        if all(program.apply(source) == target for source, target in examples):
            consistent[_program_signature(program)] = program
    ranked = sorted(
        consistent.values(),
        key=lambda program: (program.cost, program.name),
    )
    if max_depth == 3 and not any(
        not any(
            operation.name.startswith(_LEARNED_PREFIXES)
            for operation in program.operations
        )
        for program in ranked
    ):
        base_operations = [
            operation for operation in operations if operation.cost < _EXTRA_OPERATION_COST
        ]
        return ranked + _beam_candidate_programs(examples, max_depth, base_operations)
    return ranked


def solve_task(
    task: dict[str, Any],
    max_depth: int = 2,
    proposer: ProgramProposer | None = None,
    proposal_count: int = 3,
) -> list[SolveResult]:
    """Infer a transformation from all demonstrations and predict each test input."""
    if max_depth not in (1, 2, 3):
        raise ValueError("max_depth must be 1, 2, or 3.")
    if proposal_count < 1:
        raise ValueError("proposal_count must be positive.")
    raw_train = task.get("train")
    raw_test = task.get("test")
    if not isinstance(raw_train, list) or not raw_train:
        raise ValueError("Task must contain at least one training pair.")
    if not isinstance(raw_test, list) or not raw_test:
        raise ValueError("Task must contain at least one test input.")

    examples: list[tuple[Grid, Grid]] = []
    for pair in raw_train:
        examples.append((as_grid(pair["input"]), as_grid(pair["output"])))
    operations = build_operations(examples)
    programs = _candidate_programs(examples, max_depth, operations)
    test_inputs = [as_grid(pair["input"]) for pair in raw_test]
    model_proposal_count = 0
    model_accepted_count = 0
    model_rejection_reasons: list[str] = []
    model_program_signatures: set[tuple[str, ...]] = set()
    model_direct_outputs: tuple[Grid, ...] | None = None
    model_calibration_passed = False
    model_direct_output_error: str | None = None
    if proposer is not None:
        safe_task = {
            "train": [
                {"input": source, "output": target}
                for source, target in examples
            ],
            "test": [{"input": grid} for grid in test_inputs],
        }
        proposals = proposer.propose(safe_task, operations, proposal_count)
        model_proposal_count = len(proposals)
        operations_by_name = {operation.name: operation for operation in operations}
        model_programs: dict[tuple[str, ...], Program] = {}
        for proposal in proposals:
            if proposal.direct_output_error is not None:
                model_direct_output_error = proposal.direct_output_error
            if (
                proposal.calibration_output is not None
                and proposal.direct_test_outputs is not None
            ):
                if proposal.calibration_output == examples[-1][1]:
                    model_direct_outputs = proposal.direct_test_outputs
                    model_calibration_passed = True
            if proposal.error is not None:
                model_rejection_reasons.append(proposal.error)
                continue
            if proposal.operation_names is None:
                model_rejection_reasons.append("Proposal contained no operation sequence.")
                continue
            if not proposal.operation_names or len(proposal.operation_names) > 5:
                model_rejection_reasons.append(
                    "Program must contain between one and five operations."
                )
                continue
            if any(not isinstance(name, str) for name in proposal.operation_names):
                model_rejection_reasons.append(
                    "Every proposed operation name must be a string."
                )
                continue
            selected_operations = [
                operations_by_name.get(name) for name in proposal.operation_names
            ]
            if any(operation is None for operation in selected_operations):
                model_rejection_reasons.append("Proposal referenced an unavailable operation.")
                continue
            program = Program(
                tuple(
                    operation
                    for operation in selected_operations
                    if operation is not None
                )
            )
            if not all(program.apply(source) == target for source, target in examples):
                model_rejection_reasons.append(
                    f"Program failed exact demonstration verification: {program.name}"
                )
                continue
            signature = _program_signature(program)
            model_programs[signature] = program
            model_program_signatures.add(signature)
        model_accepted_count = len(model_programs)
        programs_by_signature = {
            _program_signature(program): program
            for program in programs + list(model_programs.values())
        }
        programs = sorted(
            programs_by_signature.values(),
            key=lambda program: (program.cost, program.name),
        )
    results: list[SolveResult] = []
    for test_input in test_inputs:
        votes: dict[Grid, list[Program]] = defaultdict(list)
        for program in programs:
            prediction = program.apply(test_input)
            if prediction is not None:
                votes[prediction].append(program)
        ranked = sorted(
            votes.items(),
            key=lambda item: (
                -len(
                    {
                        _program_signature(program)
                        for program in item[1]
                        if _program_signature(program) in model_program_signatures
                    }
                ),
                -len(
                    {
                        program.operations[0].name
                        for program in item[1]
                        if program.operations[0].cost < _EXTRA_OPERATION_COST
                    }
                ),
                min(program.cost for program in item[1]),
                min(program.name for program in item[1]),
            ),
        )
        matching_examples = {
            target
            for source, target in examples
            if source == test_input
        }
        if len(matching_examples) == 1:
            exact_example = next(iter(matching_examples))
            results.append(
                SolveResult(
                    exact_example,
                    exact_example,
                    ("exact_demonstration_match",),
                    len(programs),
                    model_proposal_count,
                    model_accepted_count,
                    len(model_rejection_reasons),
                    tuple(model_rejection_reasons),
                    model_calibration_passed,
                    model_direct_output_error,
                )
            )
            continue
        if ranked:
            symbolic_first = ranked[0][0]
            symbolic_second = ranked[1][0] if len(ranked) > 1 else symbolic_first
            names = tuple(
                min((program.name for program in group), key=lambda name: (len(name), name))
                for _, group in ranked[:2]
            )
            if model_direct_outputs is not None:
                first = model_direct_outputs[len(results)]
                second = (
                    symbolic_first
                    if symbolic_first != first
                    else symbolic_second
                )
                program_names = ("calibrated_model_prediction",) + names[:1]
            else:
                first = symbolic_first
                second = symbolic_second
                program_names = names
            results.append(
                SolveResult(
                    first,
                    second,
                    program_names,
                    len(programs),
                    model_proposal_count,
                    model_accepted_count,
                    len(model_rejection_reasons),
                    tuple(model_rejection_reasons),
                    model_calibration_passed,
                    model_direct_output_error,
                )
            )
        elif model_direct_outputs is not None:
            first = model_direct_outputs[len(results)]
            second = test_input if test_input != first else first
            results.append(
                SolveResult(
                    first,
                    second,
                    ("calibrated_model_prediction",),
                    0,
                    model_proposal_count,
                    model_accepted_count,
                    len(model_rejection_reasons),
                    tuple(model_rejection_reasons),
                    model_calibration_passed,
                    model_direct_output_error,
                )
            )
        else:
            results.append(
                SolveResult(
                    test_input,
                    test_input,
                    ("fallback_identity",),
                    0,
                    model_proposal_count,
                    model_accepted_count,
                    len(model_rejection_reasons),
                    tuple(model_rejection_reasons),
                    model_calibration_passed,
                    model_direct_output_error,
                )
            )
    return results
