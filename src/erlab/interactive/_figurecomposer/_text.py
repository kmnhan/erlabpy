"""Text parsing and generated-code literal helpers."""

from __future__ import annotations

import ast
import typing

import numpy as np

import erlab.interactive.utils
from erlab.interactive._figurecomposer._axes import _compact_axes_code

if typing.TYPE_CHECKING:
    from collections.abc import Sequence

    import xarray as xr

    from erlab.interactive._figurecomposer._state import FigureLimit


def _float_pair_from_text(text: str) -> tuple[float, float] | None:
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        return None
    if len(parts) != 2:
        raise ValueError("Expected two comma-separated values")
    return (float(parts[0]), float(parts[1]))


def _plot_limit_from_text(text: str) -> FigureLimit | None:
    stripped = text.strip()
    if not stripped:
        return None
    value = ast.literal_eval(stripped)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, list | tuple):
        if len(value) == 1:
            return float(value[0])
        if len(value) == 2:
            return (float(value[0]), float(value[1]))
    raise ValueError("Expected one number or two comma-separated numbers")


def _float_tuple_from_text(text: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in text.split(",") if part.strip())


def _literal_sequence_from_text(text: str) -> tuple[typing.Any, ...]:
    stripped = text.strip()
    if not stripped:
        return ()
    if stripped[0] in "[(":
        value = ast.literal_eval(stripped)
        if isinstance(value, (list, tuple)):
            return tuple(value)
        return (value,)
    value = ast.literal_eval(f"({stripped},)")
    if not isinstance(value, tuple):
        return (value,)
    return value


def _format_literal_sequence(value: Sequence[typing.Any]) -> str:
    return ", ".join(erlab.interactive.utils._parse_single_arg(item) for item in value)


def _string_tuple_from_text(text: str) -> tuple[str, ...]:
    stripped = text.strip()
    if not stripped:
        return ()
    if stripped[0] in "[(":
        value = ast.literal_eval(stripped)
        if isinstance(value, str):
            return (value,)
        if not isinstance(value, (list, tuple)):
            raise TypeError("Expected a string, list, or tuple literal")
        return tuple(str(item) for item in value)
    return tuple(part.strip() for part in text.split(",") if part.strip())


def _text_tuple_from_text(
    text: str, *, preserve_empty: bool = False
) -> tuple[str, ...]:
    if preserve_empty:
        return tuple(part.strip() for part in text.split("\n"))
    return tuple(part.strip() for part in text.splitlines() if part.strip())


def _literal_from_ast(node: ast.AST, *, allow_slice: bool = False) -> typing.Any:
    if (
        allow_slice
        and isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "slice"
        and not node.keywords
        and len(node.args) <= 3
    ):
        return slice(
            *(_literal_from_ast(arg, allow_slice=allow_slice) for arg in node.args)
        )
    return ast.literal_eval(node)


def _dict_from_text(text: str, *, allow_slice: bool = False) -> dict[str, typing.Any]:
    stripped = text.strip()
    if not stripped:
        return {}
    expression = ast.parse(
        f"_kwargs({stripped})" if not stripped.startswith("{") else stripped,
        mode="eval",
    ).body
    if isinstance(expression, ast.Call):
        if expression.args:
            raise TypeError("Expected a dictionary literal or keyword arguments")
        kwargs: dict[str, typing.Any] = {}
        for keyword in expression.keywords:
            if keyword.arg is None:
                raise TypeError("Expected explicit keyword arguments")
            kwargs[keyword.arg] = _literal_from_ast(
                keyword.value, allow_slice=allow_slice
            )
        return kwargs
    if not isinstance(expression, ast.Dict):
        raise TypeError("Expected a dictionary literal or keyword arguments")
    dict_kwargs: dict[typing.Any, typing.Any] = {}
    for key_node, value_node in zip(expression.keys, expression.values, strict=True):
        if key_node is None:
            raise TypeError("Expected explicit dictionary keys")
        dict_kwargs[_literal_from_ast(key_node, allow_slice=allow_slice)] = (
            _literal_from_ast(value_node, allow_slice=allow_slice)
        )
    return dict_kwargs


def _format_pair(value: tuple[float, float] | None) -> str:
    if value is None:
        return ""
    return f"{value[0]:g}, {value[1]:g}"


def _format_plot_limit(value: FigureLimit | None) -> str:
    if value is None:
        return ""
    if isinstance(value, int | float):
        return f"{value:g}"
    return _format_pair(value)


def _format_tuple(value: Sequence[float]) -> str:
    return ", ".join(f"{item:g}" for item in value)


def _format_string_tuple(value: Sequence[str]) -> str:
    return ", ".join(value)


def _format_axes_tuple(
    value: Sequence[tuple[int, int]],
    *,
    nrows: int | None = None,
    ncols: int | None = None,
) -> str:
    if not value:
        return "none"
    compact = _compact_axes_code(value, nrows=nrows, ncols=ncols)
    if compact is not None:
        return compact
    return ", ".join(f"axs[{row}, {col}]" for row, col in value)


def _format_dict(value: dict[str, typing.Any]) -> str:
    if not value:
        return ""
    return erlab.interactive.utils.format_kwargs(value)


def _format_dim_sizes(data: xr.DataArray) -> str:
    return ", ".join(f"{dim}={data.sizes[dim]}" for dim in data.dims)


def _selection_value_count(value: typing.Any) -> int | None:
    if isinstance(value, (str, bytes, slice)):
        return None
    if isinstance(value, np.ndarray):
        return int(value.size)
    if isinstance(value, (list, tuple)):
        return len(value)
    return None


def _code_kwargs(kwargs: dict[str, typing.Any]) -> str:
    if not kwargs:
        return ""
    return erlab.interactive.utils.format_call_kwargs(kwargs)


def _code_args(args: Sequence[typing.Any]) -> str:
    return ", ".join(erlab.interactive.utils._parse_single_arg(arg) for arg in args)


class _RawCode:
    def __init__(self, code: str) -> None:
        self.code = code

    def __repr__(self) -> str:
        return self.code

    def __str__(self) -> str:
        return self.code
