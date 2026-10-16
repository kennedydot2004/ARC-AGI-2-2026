"""Structured program proposals and validation-safe prompts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from .grids import to_json_grid
from .operations import Operation


@dataclass(frozen=True)
class ProgramProposal:
    operation_names: tuple[str, ...] | None
    rationale: str = ""
    error: str | None = None


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


def build_program_prompt(task: dict[str, Any], operations: list[Operation]) -> str:
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
    return (
        "Infer the single transformation rule demonstrated by these ARC examples. "
        "Choose a sequence of 1 to 5 operations only from the allowed catalog. "
        "The sequence is applied left to right to each test input. Do not write or "
        "execute code, do not invent operations, and do not use outside information. "
        "Prefer the shortest rule that explains every demonstration. Test inputs "
        "are included only to clarify which grids the rule will be applied to.\n\n"
        f"TRAINING_EXAMPLES={json.dumps(training_examples, separators=(',', ':'))}\n"
        f"TEST_INPUTS={json.dumps(test_inputs, separators=(',', ':'))}\n"
        f"ALLOWED_OPERATIONS={json.dumps(operation_catalog, separators=(',', ':'))}\n\n"
        'Return exactly one JSON object: {"program":["operation_name",...],'
        '"rationale":"brief rule description"}.'
    )


def parse_program_proposal(
    response: str,
    allowed_operation_names: set[str],
    max_length: int = 5,
) -> ProgramProposal:
    decoder = json.JSONDecoder()
    parsed: dict[str, Any] | None = None
    for position, character in enumerate(response):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(response, position)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "program" in value:
            parsed = value
    if parsed is None:
        return ProgramProposal(None, error="Response did not contain a JSON program object.")

    raw_program = parsed["program"]
    if not isinstance(raw_program, list) or not raw_program:
        return ProgramProposal(None, error="The program must be a non-empty list of operation names.")
    if len(raw_program) > max_length:
        return ProgramProposal(None, error=f"The program exceeds the {max_length}-operation limit.")
    if any(not isinstance(name, str) for name in raw_program):
        return ProgramProposal(None, error="Every program operation must be a string.")
    names = tuple(raw_program)
    unknown = sorted(set(names) - allowed_operation_names)
    if unknown:
        return ProgramProposal(None, error=f"Unknown operation name(s): {', '.join(unknown)}.")
    rationale = parsed.get("rationale", "")
    if not isinstance(rationale, str):
        return ProgramProposal(None, error="The rationale field must be a string.")
    return ProgramProposal(names, rationale)
