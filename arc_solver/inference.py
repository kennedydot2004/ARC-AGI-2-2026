"""Task-level program synthesis and prediction ranking."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .grids import Grid, as_grid
from .operations import Operation, build_operations


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


@dataclass(frozen=True)
class SolveResult:
    attempt_1: Grid
    attempt_2: Grid
    program_names: tuple[str, ...]
    consistent_program_count: int


def _candidate_programs(
    examples: list[tuple[Grid, Grid]],
    max_depth: int,
) -> list[Program]:
    operations = build_operations(examples)
    first_input, first_output = examples[0]
    one_step: list[tuple[Operation, Grid | None]] = [
        (operation, operation.apply(first_input)) for operation in operations
    ]
    exact_programs: list[Program] = []
    for operation, result in one_step:
        if result == first_output:
            exact_programs.append(Program((operation,)))

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
            consistent[tuple(op.name for op in program.operations)] = program
    return sorted(consistent.values(), key=lambda program: (program.cost, program.name))


def solve_task(task: dict[str, Any], max_depth: int = 2) -> list[SolveResult]:
    """Infer a transformation from all demonstrations and predict each test input."""
    if max_depth not in (1, 2):
        raise ValueError("max_depth must be 1 or 2.")
    raw_train = task.get("train")
    raw_test = task.get("test")
    if not isinstance(raw_train, list) or not raw_train:
        raise ValueError("Task must contain at least one training pair.")
    if not isinstance(raw_test, list) or not raw_test:
        raise ValueError("Task must contain at least one test input.")

    examples: list[tuple[Grid, Grid]] = []
    for pair in raw_train:
        examples.append((as_grid(pair["input"]), as_grid(pair["output"])))
    programs = _candidate_programs(examples, max_depth)
    test_inputs = [as_grid(pair["input"]) for pair in raw_test]
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
                -len({program.operations[0].name for program in item[1]}),
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
                )
            )
            continue
        if ranked:
            first = ranked[0][0]
            second = ranked[1][0] if len(ranked) > 1 else first
            names = tuple(
                min((program.name for program in group), key=lambda name: (len(name), name))
                for _, group in ranked[:2]
            )
            results.append(SolveResult(first, second, names, len(programs)))
        else:
            results.append(SolveResult(test_input, test_input, ("fallback_identity",), 0))
    return results
