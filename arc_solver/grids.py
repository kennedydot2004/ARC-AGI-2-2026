"""Grid validation and small immutable-grid utilities."""

from __future__ import annotations

from collections import Counter
from typing import TypeAlias

Grid: TypeAlias = tuple[tuple[int, ...], ...]
RawGrid: TypeAlias = list[list[int]]


def as_grid(value: object) -> Grid:
    """Validate a JSON-compatible ARC grid and return an immutable copy."""
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("A grid must be a non-empty list of rows.")
    rows: list[tuple[int, ...]] = []
    width: int | None = None
    for row in value:
        if not isinstance(row, (list, tuple)) or not row:
            raise ValueError("Grid rows must be non-empty lists.")
        converted: list[int] = []
        for cell in row:
            if isinstance(cell, bool) or not isinstance(cell, int) or not 0 <= cell <= 9:
                raise ValueError(f"Invalid ARC cell value: {cell!r}")
            converted.append(cell)
        if width is None:
            width = len(converted)
        elif len(converted) != width:
            raise ValueError("Grid rows must all have the same width.")
        rows.append(tuple(converted))
    if len(rows) > 30 or width is None or width > 30:
        raise ValueError("ARC grids must be no larger than 30 by 30.")
    return tuple(rows)


def to_json_grid(grid: Grid) -> RawGrid:
    return [list(row) for row in grid]


def shape(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0])


def most_common_color(grid: Grid) -> int:
    counts = Counter(cell for row in grid for cell in row)
    return min(counts, key=lambda color: (-counts[color], color))


def crop(grid: Grid, top: int, left: int, bottom: int, right: int) -> Grid:
    return tuple(row[left:right] for row in grid[top:bottom])


def bounding_box(grid: Grid, background: int) -> tuple[int, int, int, int] | None:
    cells = [
        (r, c)
        for r, row in enumerate(grid)
        for c, color in enumerate(row)
        if color != background
    ]
    if not cells:
        return None
    rows, cols = zip(*cells)
    return min(rows), min(cols), max(rows) + 1, max(cols) + 1
