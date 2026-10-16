"""Small, data-independent transformations used by the synthesis baseline."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

from .grids import Grid, bounding_box, crop, most_common_color, shape

GridTransform = Callable[[Grid], Grid | None]


@dataclass(frozen=True)
class Operation:
    name: str
    apply: GridTransform
    cost: int = 1


def _rotate(grid: Grid, turns: int) -> Grid:
    result = grid
    for _ in range(turns):
        result = tuple(tuple(row) for row in zip(*result[::-1]))
    return result


def _flip_horizontal(grid: Grid) -> Grid:
    return tuple(tuple(reversed(row)) for row in grid)


def _flip_vertical(grid: Grid) -> Grid:
    return tuple(reversed(grid))


def _translate(grid: Grid, dx: int, dy: int) -> Grid:
    height, width = shape(grid)
    background = most_common_color(grid)
    rows = [[background] * width for _ in range(height)]
    for r in range(height):
        for c in range(width):
            nr, nc = r + dy, c + dx
            if 0 <= nr < height and 0 <= nc < width:
                rows[nr][nc] = grid[r][c]
    return tuple(tuple(row) for row in rows)


def _infer_translations(examples: list[tuple[Grid, Grid]]) -> list[tuple[int, int]]:
    source, target = examples[0]
    if shape(source) != shape(target):
        return []
    background = most_common_color(source)
    anchors = [
        (r, c, source[r][c])
        for r, row in enumerate(source)
        for c in range(len(row))
        if source[r][c] != background
    ]
    if not anchors:
        return []
    color_counts = Counter(color for _, _, color in anchors)
    anchor_color = min(color_counts, key=lambda color: (color_counts[color], color))
    anchor_row, anchor_col, _ = next(item for item in anchors if item[2] == anchor_color)
    target_positions = [
        (r, c)
        for r, row in enumerate(target)
        for c, color in enumerate(row)
        if color == anchor_color
    ]
    return sorted(
        (col - anchor_col, row - anchor_row)
        for row, col in target_positions
        if (row, col) != (anchor_row, anchor_col)
        and all(
            _translate(example, col - anchor_col, row - anchor_row) == expected
            for example, expected in examples
        )
    )


def _background_candidates(examples: list[tuple[Grid, Grid]]) -> list[int]:
    candidates = {0}
    for grid, _ in examples:
        candidates.add(most_common_color(grid))
        border = (
            list(grid[0])
            + list(grid[-1])
            + [row[0] for row in grid[1:-1]]
            + [row[-1] for row in grid[1:-1]]
        )
        if border:
            counts = Counter(border)
            candidates.add(min(counts, key=lambda color: (-counts[color], color)))
    return sorted(candidates)


@lru_cache(maxsize=256)
def _isolate_color(grid: Grid, color: int) -> Grid:
    background = most_common_color(grid)
    return tuple(
        tuple(cell if cell == color else background for cell in row)
        for row in grid
    )


def _remove_color(grid: Grid, color: int) -> Grid:
    background = most_common_color(grid)
    return tuple(
        tuple(background if cell == color else cell for cell in row)
        for row in grid
    )


@lru_cache(maxsize=256)
def _components(
    grid: Grid,
    background: int,
) -> tuple[tuple[int, int, int, int, frozenset[tuple[int, int]]], ...]:
    height, width = shape(grid)
    seen: set[tuple[int, int]] = set()
    components: list[tuple[int, int, int, int, frozenset[tuple[int, int]]]] = []
    for r in range(height):
        for c in range(width):
            color = grid[r][c]
            if color == background or (r, c) in seen:
                continue
            queue = deque([(r, c)])
            seen.add((r, c))
            cells: set[tuple[int, int]] = set()
            while queue:
                cr, cc = queue.popleft()
                cells.add((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and grid[nr][nc] == color
                        and (nr, nc) not in seen
                    ):
                        seen.add((nr, nc))
                        queue.append((nr, nc))
            rows, cols = zip(*cells)
            components.append(
                (color, len(cells), min(rows), min(cols), frozenset(cells))
            )
    return tuple(components)


def _component_crop(
    grid: Grid,
    background: int,
    ordering: str,
    isolate: bool,
) -> Grid | None:
    components = _components(grid, background)
    if not components:
        return None
    sort_keys = {
        "largest": lambda obj: (-obj[1], obj[2], obj[3], obj[0]),
        "smallest": lambda obj: (obj[1], obj[2], obj[3], obj[0]),
        "topmost": lambda obj: (obj[2], obj[3], -obj[1], obj[0]),
        "bottommost": lambda obj: (-obj[2], obj[3], -obj[1], obj[0]),
        "leftmost": lambda obj: (obj[3], obj[2], -obj[1], obj[0]),
        "rightmost": lambda obj: (-obj[3], obj[2], -obj[1], obj[0]),
    }
    selected = sorted(components, key=sort_keys[ordering])[0]
    _, _, top, left, cells = selected
    bottom = max(r for r, _ in cells) + 1
    right = max(c for _, c in cells) + 1
    if isolate:
        return tuple(
            tuple(grid[r][c] if (r, c) in cells else background for c in range(left, right))
            for r in range(top, bottom)
        )
    return crop(grid, top, left, bottom, right)


def _learn_color_map(examples: list[tuple[Grid, Grid]]) -> dict[int, int] | None:
    mapping: dict[int, int] = {}
    for source, target in examples:
        if shape(source) != shape(target):
            return None
        for srow, trow in zip(source, target):
            for old, new in zip(srow, trow):
                if old in mapping and mapping[old] != new:
                    return None
                mapping[old] = new
    return mapping if any(old != new for old, new in mapping.items()) else None


def build_operations(examples: list[tuple[Grid, Grid]]) -> list[Operation]:
    operations: list[Operation] = [
        Operation("identity", lambda grid: grid, 0),
        Operation("rotate_90", lambda grid: _rotate(grid, 1)),
        Operation("rotate_180", lambda grid: _rotate(grid, 2)),
        Operation("rotate_270", lambda grid: _rotate(grid, 3)),
        Operation("flip_horizontal", _flip_horizontal),
        Operation("flip_vertical", _flip_vertical),
        Operation("transpose", lambda grid: tuple(tuple(row) for row in zip(*grid))),
        Operation("anti_transpose", lambda grid: _rotate(_flip_horizontal(grid), 1)),
    ]

    color_map = _learn_color_map(examples)
    if color_map is not None:
        operations.append(
            Operation(
                "global_color_map",
                lambda grid, mapping=color_map: tuple(
                    tuple(mapping.get(cell, cell) for cell in row) for row in grid
                ),
                2,
            )
        )

    for background in _background_candidates(examples):
        operations.append(
            Operation(
                f"crop_background_{background}",
                lambda grid, bg=background: (
                    None
                    if (box := bounding_box(grid, bg)) is None
                    else crop(grid, *box)
                ),
                2,
            )
        )
        for ordering in ("largest", "smallest", "topmost", "bottommost", "leftmost", "rightmost"):
            for isolate in (False, True):
                operations.append(
                    Operation(
                        f"crop_{ordering}_component_bg{background}_isolate{int(isolate)}",
                        lambda grid, bg=background, order=ordering, only=isolate: _component_crop(
                            grid, bg, order, only
                        ),
                        3,
                    )
                )

    colors = sorted({cell for source, _ in examples for row in source for cell in row})
    for color in colors:
        operations.append(
            Operation(
                f"isolate_color_{color}",
                lambda grid, selected=color: _isolate_color(grid, selected),
                2,
            )
        )
        operations.append(
            Operation(
                f"remove_color_{color}",
                lambda grid, removed=color: _remove_color(grid, removed),
                2,
            )
        )
    for dx, dy in _infer_translations(examples):
        operations.append(
            Operation(
                f"translate_x{dx}_y{dy}",
                lambda grid, x=dx, y=dy: _translate(grid, x, y),
                2,
            )
        )

    for factor_y in range(2, 4):
        for factor_x in range(2, 4):
            operations.extend(
                [
                    Operation(
                        f"tile_{factor_y}x{factor_x}",
                        lambda grid, fy=factor_y, fx=factor_x: tuple(
                            row
                            for _ in range(fy)
                            for source_row in grid
                            for row in (source_row * fx,)
                        ),
                        3,
                    ),
                    Operation(
                        f"scale_{factor_y}x{factor_x}",
                        lambda grid, fy=factor_y, fx=factor_x: tuple(
                            expanded_row
                            for source_row in grid
                            for expanded_row in (
                                tuple(cell for cell in source_row for _ in range(fx)),
                            )
                            for _ in range(fy)
                        ),
                        3,
                    ),
                ]
            )
    return operations
