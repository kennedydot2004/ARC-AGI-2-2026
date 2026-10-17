"""Structured program proposals and validation-safe prompts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from typing import Any, Protocol

from .grids import Grid, as_grid, to_json_grid
from .operations import Operation


@dataclass(frozen=True)
class ProgramProposal:
    operation_names: tuple[str, ...] | None
    rationale: str = ""
    error: str | None = None
    calibration_output: Grid | None = None
    direct_test_outputs: tuple[Grid, ...] | None = None
    direct_output_error: str | None = None


class ProgramProposer(Protocol):
    def propose(
        self,
        task: dict[str, Any],
        operations: list[Operation],
        count: int,
    ) -> list[ProgramProposal]: ...


def describe_operation(name: str) -> str:
    descriptions = {
        "identity": "leave the grid unchanged",
        "rotate_90": "rotate the whole grid 90 degrees clockwise",
        "rotate_180": "rotate the whole grid 180 degrees",
        "rotate_270": "rotate the whole grid 90 degrees counterclockwise",
        "flip_horizontal": "mirror the whole grid left to right",
        "flip_vertical": "mirror the whole grid top to bottom",
        "transpose": "transpose rows and columns",
        "anti_transpose": "reflect across the top-right to bottom-left diagonal",
        "global_color_map": "apply one globally consistent color remapping inferred from the examples",
    }
    if name in descriptions:
        return descriptions[name]
    match = re.fullmatch(r"crop_background_(\d+)", name)
    if match:
        return f"crop to the tight bounding box of cells not color {match.group(1)}"
    match = re.fullmatch(r"crop_excluding_color_(\d+)", name)
    if match:
        return f"crop to the tight bounding box of cells not color {match.group(1)}"
    match = re.fullmatch(
        r"crop_(largest|smallest|topmost|bottommost|leftmost|rightmost)_component_bg(\d+)_isolate([01])",
        name,
    )
    if match:
        selected, background, isolate = match.groups()
        detail = "keep only that component" if isolate == "1" else "keep the selected crop contents"
        return (
            f"select the {selected} connected component excluding background color "
            f"{background}, crop to its bounds, and {detail}"
        )
    match = re.fullmatch(r"(isolate|remove)_color_(\d+)", name)
    if match:
        action, color = match.groups()
        if action == "isolate":
            return f"keep color {color} and replace all other cells with the grid's modal color"
        return f"replace color {color} with the grid's modal color"
    match = re.fullmatch(r"translate_x(-?\d+)_y(-?\d+)", name)
    if match:
        dx, dy = match.groups()
        return f"shift all cells by {dx} columns and {dy} rows, filling exposed cells with the modal color"
    match = re.fullmatch(r"(tile|scale)_(\d+)x(\d+)", name)
    if match:
        operation, height, width = match.groups()
        action = "repeat the full grid" if operation == "tile" else "repeat each cell"
        return f"{action} {height} times vertically and {width} times horizontally"
    return f"apply the predefined operation named {name}"


def build_program_prompt(
    task: dict[str, Any],
    operations: list[Operation],
    candidate_count: int = 3,
    calibration_input: Grid | None = None,
) -> str:
    if candidate_count < 1:
        raise ValueError("At least one candidate program is required.")
    training_examples = [
        {
            "input": to_json_grid(pair["input"]),
            "output": to_json_grid(pair["output"]),
        }
        for pair in task["train"]
    ]
    test_inputs = [{"input": to_json_grid(pair["input"])} for pair in task["test"]]
    operation_catalog = [
        {"name": operation.name, "meaning": describe_operation(operation.name)}
        for operation in operations
    ]
    calibration_section = ""
    response_fields = ""
    program_schema = ",".join(
        '["operation_name"]' for _ in range(candidate_count)
    )
    if calibration_input is not None:
        calibration_section = (
            f"CALIBRATION_INPUT={json.dumps(to_json_grid(calibration_input), separators=(',', ':'))}\n"
            "Predict this held-out training example using only TRAINING_EXAMPLES, "
            "then predict TEST_INPUTS. Write each output as a list of digit strings, "
            "one string per row. The solver will use the calibration prediction only "
            "if it exactly matches the withheld training output.\n"
        )
        response_fields = (
            ',"calibration_output":["012","345"],'
            '"test_outputs":[["012","345"]]'
        )
    return (
        "Infer the single transformation rule demonstrated by these ARC examples. "
        "Choose a sequence of 1 to 5 operations only from the allowed catalog. "
        "The sequence is applied left to right to each test input. Do not write or "
        "execute code, do not invent operations, and do not use outside information. "
        "Prefer the shortest rule that explains every demonstration. Test inputs "
        "are included only to clarify which grids the rule will be applied to. "
        f"Propose exactly {candidate_count} distinct candidate programs. "
        "Return exactly one JSON object and no other text. Do not include "
        "rationales. Use only operation names from the catalog. "
        f'Output schema: {{"programs":[{program_schema}]{response_fields}}}.\n\n'
        f"TRAINING_EXAMPLES={json.dumps(training_examples, separators=(',', ':'))}\n"
        f"{calibration_section}"
        f"TEST_INPUTS={json.dumps(test_inputs, separators=(',', ':'))}\n"
        f"ALLOWED_OPERATIONS={json.dumps(operation_catalog, separators=(',', ':'))}"
    )


def _find_program_object(response: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    parsed: dict[str, Any] | None = None
    for position, character in enumerate(response):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(response, position)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and (
            "programs" in value or "program" in value
        ):
            parsed = value
    return parsed


def _find_json_field(response: str, field: str) -> Any | None:
    decoder = json.JSONDecoder()
    pattern = re.compile(rf'"{re.escape(field)}"\s*:')
    value: Any | None = None
    for match in pattern.finditer(response):
        position = match.end()
        while position < len(response) and response[position].isspace():
            position += 1
        try:
            value, _ = decoder.raw_decode(response, position)
        except json.JSONDecodeError:
            continue
    return value


def _parse_program_fields(
    raw_program: Any,
    rationale: Any,
    allowed_operation_names: set[str],
    max_length: int,
) -> ProgramProposal:
    if not isinstance(raw_program, list) or not raw_program:
        return ProgramProposal(
            None,
            error="The program must be a non-empty list of operation names.",
        )
    if len(raw_program) > max_length:
        return ProgramProposal(
            None,
            error=f"The program exceeds the {max_length}-operation limit.",
        )
    if any(not isinstance(name, str) for name in raw_program):
        return ProgramProposal(
            None,
            error="Every program operation must be a string.",
        )
    names = tuple(raw_program)
    unknown = sorted(set(names) - allowed_operation_names)
    if unknown:
        return ProgramProposal(
            None,
            error=f"Unknown operation name(s): {', '.join(unknown)}.",
        )
    if not isinstance(rationale, str):
        return ProgramProposal(None, error="The rationale field must be a string.")
    return ProgramProposal(names, rationale)


def _parse_output_grid(value: Any) -> Grid:
    if isinstance(value, list) and value and all(isinstance(row, str) for row in value):
        if any(not row or any(character not in "0123456789" for character in row) for row in value):
            raise ValueError("Grid rows must be non-empty strings of ARC color digits.")
        value = [[int(character) for character in row] for row in value]
    return as_grid(value)


def parse_program_proposals(
    response: str,
    allowed_operation_names: set[str],
    count: int,
    max_length: int = 5,
    test_count: int = 0,
    require_calibration: bool = False,
) -> list[ProgramProposal]:
    if count < 1:
        raise ValueError("At least one candidate program is required.")
    parsed = _find_program_object(response)
    if parsed is None:
        raw_programs = _find_json_field(response, "programs")
        raw_program = _find_json_field(response, "program")
        if raw_programs is None and raw_program is None:
            return [
                ProgramProposal(
                    None,
                    error="Response did not contain a JSON program object.",
                )
                for _ in range(count)
            ]
        parsed = {}
        if raw_programs is not None:
            parsed["programs"] = raw_programs
        if raw_program is not None:
            parsed["program"] = raw_program
        calibration_output = _find_json_field(response, "calibration_output")
        test_outputs = _find_json_field(response, "test_outputs")
        if calibration_output is not None:
            parsed["calibration_output"] = calibration_output
        if test_outputs is not None:
            parsed["test_outputs"] = test_outputs

    raw_programs = parsed.get("programs")
    if raw_programs is None:
        raw_programs = [parsed.get("program")]
    if not isinstance(raw_programs, list):
        raw_programs = []

    proposals: list[ProgramProposal] = []
    for raw_program in raw_programs[:count]:
        if isinstance(raw_program, dict):
            proposals.append(
                _parse_program_fields(
                    raw_program.get("program"),
                    raw_program.get("rationale", ""),
                    allowed_operation_names,
                    max_length,
                )
            )
        else:
            proposals.append(
                _parse_program_fields(
                    raw_program,
                    "",
                    allowed_operation_names,
                    max_length,
                )
            )

    while len(proposals) < count:
        proposals.append(
            ProgramProposal(
                None,
                error=(
                    f"Response returned {len(raw_programs)} programs; "
                    f"{count} were requested."
                ),
            )
        )

    if require_calibration:
        direct_output_error: str | None = None
        calibration_output: Grid | None = None
        direct_test_outputs: tuple[Grid, ...] | None = None
        try:
            if "calibration_output" not in parsed:
                raise ValueError("Response did not include calibration_output.")
            calibration_output = _parse_output_grid(parsed["calibration_output"])
            raw_test_outputs = parsed.get("test_outputs")
            if not isinstance(raw_test_outputs, list):
                raise ValueError("Response did not include a test_outputs list.")
            if len(raw_test_outputs) != test_count:
                raise ValueError(
                    f"Expected {test_count} test outputs, got {len(raw_test_outputs)}."
                )
            direct_test_outputs = tuple(
                _parse_output_grid(output) for output in raw_test_outputs
            )
        except ValueError as error:
            direct_output_error = str(error)
        if proposals:
            proposals[0] = replace(
                proposals[0],
                calibration_output=calibration_output,
                direct_test_outputs=direct_test_outputs,
                direct_output_error=direct_output_error,
            )
    return proposals


def parse_program_proposal(
    response: str,
    allowed_operation_names: set[str],
    max_length: int = 5,
) -> ProgramProposal:
    return parse_program_proposals(
        response,
        allowed_operation_names,
        count=1,
        max_length=max_length,
    )[0]
