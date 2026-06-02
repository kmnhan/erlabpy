"""Axes selection helpers for Figure Composer."""

from __future__ import annotations

import ast
import typing

if typing.TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

    from erlab.interactive._figurecomposer._state import FigureSubplotsState


def _all_axes(setup: FigureSubplotsState) -> tuple[tuple[int, int], ...]:
    return tuple((row, col) for row in range(setup.nrows) for col in range(setup.ncols))


def _all_axes_for_shape(nrows: int, ncols: int) -> tuple[tuple[int, int], ...]:
    return tuple((row, col) for row in range(nrows) for col in range(ncols))


def _compact_axes_code(
    axes: Sequence[tuple[int, int]],
    *,
    nrows: int | None = None,
    ncols: int | None = None,
) -> str | None:
    axes_tuple = tuple(axes)
    if not axes_tuple:
        return None
    if len(axes_tuple) == 1:
        row, col = axes_tuple[0]
        return f"axs[{row}, {col}]"
    if (
        nrows is not None
        and ncols is not None
        and tuple(axes_tuple) == _all_axes_for_shape(nrows, ncols)
    ):
        return "axs"

    rows = tuple(sorted({row for row, _col in axes_tuple}))
    cols = tuple(sorted({col for _row, col in axes_tuple}))
    if not _is_contiguous(rows) or not _is_contiguous(cols):
        return None
    if axes_tuple != tuple((row, col) for row in rows for col in cols):
        return None
    return (
        "axs["
        f"{_axis_index_code(rows[0], rows[-1] + 1, nrows)}, "
        f"{_axis_index_code(cols[0], cols[-1] + 1, ncols)}"
        "]"
    )


def _compact_axes_iterable_code(
    axes: Sequence[tuple[int, int]],
    *,
    nrows: int,
    ncols: int,
) -> str | None:
    axes_tuple = tuple(axes)
    if not axes_tuple:
        return None
    if len(axes_tuple) == 1:
        row, col = axes_tuple[0]
        return f"(axs[{row}, {col}],)"
    compact = _compact_axes_code(axes_tuple, nrows=nrows, ncols=ncols)
    if compact is not None:
        return f"{compact}.flat"
    items = ", ".join(f"axs[{row}, {col}]" for row, col in axes_tuple)
    return f"({items})"


def _is_contiguous(values: Sequence[int]) -> bool:
    return bool(values) and list(values) == list(range(values[0], values[-1] + 1))


def _axis_index_code(start: int, stop: int, size: int | None) -> str:
    if stop == start + 1:
        return str(start)
    if start == 0 and size is not None and stop == size:
        return ":"
    if start == 0:
        return f":{stop}"
    return f"{start}:{stop}"


def _axes_expression_value(expression: str, axs: np.ndarray) -> object:
    def eval_node(node: ast.AST) -> object:
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Name):
            if node.id == "axs":
                return axs
            if node.id == "ax":
                return axs[0, 0]
            raise ValueError(f"Unsupported axes name {node.id!r}")
        if isinstance(node, ast.Constant):
            if isinstance(node.value, int) or node.value is None:
                return node.value
            raise ValueError("Axes expressions only support integer indices")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            operand = eval_node(node.operand)
            if isinstance(operand, int):
                return -operand
            raise ValueError("Axes expressions only support integer indices")
        if isinstance(node, ast.Slice):
            return slice(
                eval_node(node.lower) if node.lower is not None else None,
                eval_node(node.upper) if node.upper is not None else None,
                eval_node(node.step) if node.step is not None else None,
            )
        if isinstance(node, ast.Tuple):
            return tuple(eval_node(elt) for elt in node.elts)
        if isinstance(node, ast.List):
            return [eval_node(elt) for elt in node.elts]
        if isinstance(node, ast.Subscript):
            subscriptable = typing.cast("typing.Any", eval_node(node.value))
            return subscriptable[eval_node(node.slice)]
        raise ValueError("Unsupported axes expression")

    return eval_node(ast.parse(expression, mode="eval"))
