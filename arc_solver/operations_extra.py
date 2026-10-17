"""Additional exact transformations: symmetry, gravity, fills, grid splits and learned local rules."""

from __future__ import annotations

from collections import Counter, deque
from typing import Callable

from .grids import Grid, most_common_color, shape
from .operations import Operation


def _hflip(grid: Grid) -> Grid:
    return tuple(tuple(reversed(row)) for row in grid)


def _vflip(grid: Grid) -> Grid:
    return tuple(reversed(grid))


def _hconcat(left: Grid, right: Grid) -> Grid:
    return tuple(a + b for a, b in zip(left, right))


def _symmetry_ops() -> list[Operation]:
    return [
        Operation("mirror_right", lambda g: _hconcat(g, _hflip(g)), 3),
        Operation("mirror_left", lambda g: _hconcat(_hflip(g), g), 3),
        Operation("mirror_down", lambda g: g + _vflip(g), 3),
        Operation("mirror_up", lambda g: _vflip(g) + g, 3),
        Operation(
            "mirror_quad",
            lambda g: (lambda top: top + _vflip(top))(_hconcat(g, _hflip(g))),
            3,
        ),
        Operation("keep_left_half", lambda g: tuple(r[: len(r) // 2] for r in g) if len(g[0]) > 1 else None, 3),
        Operation("keep_right_half", lambda g: tuple(r[(len(r) + 1) // 2 :] for r in g) if len(g[0]) > 1 else None, 3),
        Operation("keep_top_half", lambda g: g[: len(g) // 2] if len(g) > 1 else None, 3),
        Operation("keep_bottom_half", lambda g: g[(len(g) + 1) // 2 :] if len(g) > 1 else None, 3),
        Operation(
            "complete_symmetry",
            lambda g: _complete_symmetry(g),
            3,
        ),
        Operation("self_tile", _self_tile, 3),
        Operation("self_tile_inverse", lambda g: _self_tile(g, inverse=True), 3),
    ]


def _self_tile(grid: Grid, inverse: bool = False) -> Grid | None:
    height, width = shape(grid)
    if height * height > 30 or width * width > 30:
        return None
    background = most_common_color(grid)
    rows = []
    for r in range(height):
        for tr in range(height):
            row: list[int] = []
            for c in range(width):
                use = (grid[r][c] != background) != inverse
                row.extend(grid[tr] if use else (background,) * width)
            rows.append(tuple(row))
    return tuple(rows)


def _complete_symmetry(grid: Grid) -> Grid | None:
    """Fill the most common colour cells from their mirror positions."""
    height, width = shape(grid)
    noise = most_common_color(grid)
    counts = Counter(cell for row in grid for cell in row)
    if len(counts) < 2:
        return None
    least = min(counts, key=lambda color: (counts[color], color))
    rows = [list(row) for row in grid]
    changed = False
    for r in range(height):
        for c in range(width):
            if grid[r][c] != least:
                continue
            for mr, mc in ((r, width - 1 - c), (height - 1 - r, c), (height - 1 - r, width - 1 - c)):
                if grid[mr][mc] != least:
                    rows[r][c] = grid[mr][mc]
                    changed = True
                    break
    del noise
    return tuple(tuple(row) for row in rows) if changed else None


def _gravity(grid: Grid, direction: str) -> Grid:
    background = most_common_color(grid)
    height, width = shape(grid)
    rows = [[background] * width for _ in range(height)]
    if direction in ("down", "up"):
        for c in range(width):
            column = [grid[r][c] for r in range(height) if grid[r][c] != background]
            if direction == "down":
                for index, value in enumerate(reversed(column)):
                    rows[height - 1 - index][c] = value
            else:
                for index, value in enumerate(column):
                    rows[index][c] = value
    else:
        for r in range(height):
            line = [cell for cell in grid[r] if cell != background]
            if direction == "right":
                for index, value in enumerate(reversed(line)):
                    rows[r][width - 1 - index] = value
            else:
                for index, value in enumerate(line):
                    rows[r][index] = value
    return tuple(tuple(row) for row in rows)


def _enclosed_cells(grid: Grid, background: int) -> set[tuple[int, int]]:
    height, width = shape(grid)
    seen: set[tuple[int, int]] = set()
    queue: deque[tuple[int, int]] = deque()
    for r in range(height):
        for c in range(width):
            if (r in (0, height - 1) or c in (0, width - 1)) and grid[r][c] == background:
                seen.add((r, c))
                queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in seen and grid[nr][nc] == background:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return {
        (r, c)
        for r in range(height)
        for c in range(width)
        if grid[r][c] == background and (r, c) not in seen
    }


def _fill_enclosed(grid: Grid, fill: int) -> Grid | None:
    background = most_common_color(grid)
    cells = _enclosed_cells(grid, background)
    if not cells:
        return None
    return tuple(
        tuple(fill if (r, c) in cells else cell for c, cell in enumerate(row))
        for r, row in enumerate(grid)
    )


def _split_parts(grid: Grid, axis: str, parts: int) -> list[Grid] | None:
    height, width = shape(grid)
    size = height if axis == "v" else width
    if axis == "v":
        uniform = {r for r in range(height) if len(set(grid[r])) == 1}
        colors = {grid[r][0] for r in uniform}
    else:
        uniform = {c for c in range(width) if len({grid[r][c] for r in range(height)}) == 1}
        colors = {grid[0][c] for c in uniform}
    bounds: list[tuple[int, int]] | None = None
    if (size - (parts - 1)) % parts == 0 and size > parts:
        step = (size - (parts - 1)) // parts
        lines = [(i + 1) * step + i for i in range(parts - 1)]
        if all(line in uniform for line in lines) and len({_line_color(grid, axis, l) for l in lines}) == 1:
            bounds = [(i * (step + 1), i * (step + 1) + step) for i in range(parts)]
    if bounds is None:
        del colors
        if size % parts:
            return None
        step = size // parts
        bounds = [(i * step, (i + 1) * step) for i in range(parts)]
    pieces: list[Grid] = []
    for start, end in bounds:
        if end <= start:
            return None
        pieces.append(grid[start:end] if axis == "v" else tuple(row[start:end] for row in grid))
    return pieces


def _line_color(grid: Grid, axis: str, index: int) -> int:
    return grid[index][0] if axis == "v" else grid[0][index]


def _learn_split_rule(
    examples: list[tuple[Grid, Grid]], axis: str, parts: int
) -> Callable[[Grid], Grid | None] | None:
    table: dict[tuple[int, ...], int] = {}
    backgrounds: set[int] = set()
    for source, target in examples:
        pieces = _split_parts(source, axis, parts)
        if pieces is None or shape(pieces[0]) != shape(target):
            return None
        for piece in pieces:
            backgrounds.add(most_common_color(source))
        height, width = shape(target)
        for r in range(height):
            for c in range(width):
                key = tuple(int(piece[r][c] != 0) for piece in pieces)
                value = target[r][c]
                if table.setdefault(key, value) != value:
                    return None
    if len(table) < 2:
        return None

    def apply(grid: Grid) -> Grid | None:
        pieces = _split_parts(grid, axis, parts)
        if pieces is None:
            return None
        height, width = shape(pieces[0])
        rows = []
        for r in range(height):
            row = []
            for c in range(width):
                key = tuple(int(piece[r][c] != 0) for piece in pieces)
                if key not in table:
                    return None
                row.append(table[key])
            rows.append(tuple(row))
        return tuple(rows)

    return apply


def _patch(grid: Grid, r: int, c: int, kind: str) -> tuple[int, ...]:
    height, width = shape(grid)

    def at(rr: int, cc: int) -> int:
        return grid[rr][cc] if 0 <= rr < height and 0 <= cc < width else -1

    if kind == "cross":
        return (at(r, c), at(r - 1, c), at(r + 1, c), at(r, c - 1), at(r, c + 1))
    if kind == "square":
        return tuple(at(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1))
    if kind == "row_col":
        row_colors = tuple(sorted(set(grid[r]) - {most_common_color(grid)}))
        col_colors = tuple(sorted({grid[i][c] for i in range(height)} - {most_common_color(grid)}))
        return (at(r, c),) + row_colors + (-2,) + col_colors
    return (at(r, c),)


def _learn_local_rule(
    examples: list[tuple[Grid, Grid]], kind: str
) -> Callable[[Grid], Grid | None] | None:
    table: dict[tuple[int, ...], int] = {}
    for source, target in examples:
        if shape(source) != shape(target):
            return None
        for r, row in enumerate(source):
            for c in range(len(row)):
                key = _patch(source, r, c, kind)
                value = target[r][c]
                if table.setdefault(key, value) != value:
                    return None
    if all(key[0] == value for key, value in table.items()):
        return None

    def apply(grid: Grid) -> Grid | None:
        rows = []
        for r, row in enumerate(grid):
            new_row = []
            for c in range(len(row)):
                key = _patch(grid, r, c, kind)
                if key not in table:
                    return None
                new_row.append(table[key])
            rows.append(tuple(new_row))
        return tuple(rows)

    return apply


def _downscale(grid: Grid, fy: int, fx: int) -> Grid | None:
    height, width = shape(grid)
    if height % fy or width % fx:
        return None
    return tuple(tuple(grid[r][c] for c in range(0, width, fx)) for r in range(0, height, fy))


def _remove_noise(grid: Grid) -> Grid | None:
    background = most_common_color(grid)
    height, width = shape(grid)
    rows = [list(row) for row in grid]
    changed = False
    for r in range(height):
        for c in range(width):
            color = grid[r][c]
            if color == background:
                continue
            neighbours = [
                grid[nr][nc]
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                if 0 <= nr < height and 0 <= nc < width
            ]
            if color not in neighbours:
                rows[r][c] = background
                changed = True
    return tuple(tuple(row) for row in rows) if changed else None


def build_extra_operations(examples: list[tuple[Grid, Grid]]) -> list[Operation]:
    operations = _symmetry_ops()
    for direction in ("down", "up", "left", "right"):
        operations.append(Operation(f"gravity_{direction}", lambda g, d=direction: _gravity(g, d), 3))
    operations.append(Operation("remove_noise", _remove_noise, 3))
    for factor_y in range(1, 4):
        for factor_x in range(1, 4):
            if (factor_y, factor_x) != (1, 1):
                operations.append(
                    Operation(
                        f"downscale_{factor_y}x{factor_x}",
                        lambda g, fy=factor_y, fx=factor_x: _downscale(g, fy, fx),
                        3,
                    )
                )

    output_colors = sorted({cell for _, target in examples for row in target for cell in row})
    for color in output_colors:
        operations.append(
            Operation(f"fill_enclosed_{color}", lambda g, fill=color: _fill_enclosed(g, fill), 3)
        )

    for axis in ("v", "h"):
        for parts in (2, 3, 4):
            rule = _learn_split_rule(examples, axis, parts)
            if rule is not None:
                operations.append(Operation(f"split_{axis}{parts}_boolean_rule", rule, 5))

    for kind in ("cross", "square", "row_col", "cell"):
        rule = _learn_local_rule(examples, kind)
        if rule is not None:
            operations.append(Operation(f"local_rule_{kind}", rule, 5))
    return [Operation(o.name, o.apply, max(o.cost, 4)) for o in operations]
