"""Text parsing and generated-code literal helpers."""

from __future__ import annotations

import ast
import typing

import numpy as np

import erlab.utils._code
from erlab.interactive._figurecomposer._exceptions import FigureComposerInputError
from erlab.interactive._figurecomposer._model._axes import _compact_axes_code

if typing.TYPE_CHECKING:
    from collections.abc import Sequence

    import xarray as xr

    from erlab.interactive._figurecomposer._model._state import FigureLimit


_DICT_INPUT_MESSAGE = (
    "Enter keyword arguments like alpha=0.5, or a dictionary literal like "
    "{'alpha': 0.5}."
)
_LITERAL_INPUT_MESSAGE = "Enter a valid Python literal, such as 1, 'text', or [1, 2]."


def _input_error(message: str) -> FigureComposerInputError:
    return FigureComposerInputError(message)


def _literal_from_text(
    text: str, *, message: str = _LITERAL_INPUT_MESSAGE
) -> typing.Any:
    try:
        return ast.literal_eval(text)
    except (SyntaxError, TypeError, ValueError) as exc:
        raise _input_error(message) from exc


def _float_pair_from_text(text: str) -> tuple[float, float] | None:
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        return None
    if len(parts) != 2:
        raise _input_error("Enter exactly two comma-separated numbers.")
    try:
        return (float(parts[0]), float(parts[1]))
    except ValueError as exc:
        raise _input_error("Enter exactly two comma-separated numbers.") from exc


def _limit_bound_from_value(value: typing.Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _limit_pair_from_value(value: typing.Any) -> tuple[float | None, float | None]:
    if isinstance(value, str | bytes) or not isinstance(value, list | tuple):
        raise _input_error(
            "Enter two comma-separated numbers or None, such as 0, None."
        )
    if len(value) != 2:
        raise _input_error(
            "Enter two comma-separated numbers or None, such as 0, None."
        )
    try:
        return (_limit_bound_from_value(value[0]), _limit_bound_from_value(value[1]))
    except (TypeError, ValueError) as exc:
        raise _input_error(
            "Enter two comma-separated numbers or None, such as 0, None."
        ) from exc


def _limit_pair_from_text(text: str) -> tuple[float | None, float | None] | None:
    stripped = text.strip()
    if not stripped:
        return None
    value = _literal_from_text(
        stripped,
        message="Enter two comma-separated numbers or None, such as 0, None.",
    )
    return _limit_pair_from_value(value)


def _plot_limit_from_text(text: str) -> FigureLimit | None:
    stripped = text.strip()
    if not stripped:
        return None
    value = _literal_from_text(
        stripped,
        message=(
            "Enter one number or one pair of numbers or None, such as -1, 1 or 0, None."
        ),
    )
    try:
        if value is None:
            return None
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, list | tuple):
            if len(value) == 1:
                return _limit_bound_from_value(value[0])
            if len(value) == 2:
                return _limit_pair_from_value(value)
    except (TypeError, ValueError) as exc:
        raise _input_error(
            "Enter one number or one pair of numbers or None, such as -1, 1 or 0, None."
        ) from exc
    raise _input_error(
        "Enter one number or one pair of numbers or None, such as -1, 1 or 0, None."
    )


def _float_tuple_from_text(text: str) -> tuple[float, ...]:
    try:
        return tuple(float(part.strip()) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise _input_error("Enter comma-separated numbers.") from exc


def _literal_sequence_from_text(text: str) -> tuple[typing.Any, ...]:
    stripped = text.strip()
    if not stripped:
        return ()
    if stripped[0] in "[(":
        value = _literal_from_text(
            stripped,
            message=(
                "Enter comma-separated literal values, or a Python list/tuple literal."
            ),
        )
        if isinstance(value, (list, tuple)):
            return tuple(value)
        return (value,)
    return _literal_from_text(
        f"({stripped},)",
        message="Enter comma-separated literal values.",
    )


def _format_literal_sequence(value: Sequence[typing.Any]) -> str:
    return ", ".join(erlab.utils._code._parse_single_arg(item) for item in value)


def _string_tuple_from_text(text: str) -> tuple[str, ...]:
    stripped = text.strip()
    if not stripped:
        return ()
    if stripped[0] in "[(":
        value = _literal_from_text(
            stripped,
            message="Enter text values separated by commas, or a string list literal.",
        )
        if isinstance(value, str):
            return (value,)
        if not isinstance(value, (list, tuple)):
            raise _input_error(
                "Enter text values separated by commas, or a string list literal."
            )
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
    try:
        return ast.literal_eval(node)
    except (TypeError, ValueError) as exc:
        raise _input_error(_LITERAL_INPUT_MESSAGE) from exc


def _dict_from_text(text: str, *, allow_slice: bool = False) -> dict[str, typing.Any]:
    stripped = text.strip()
    if not stripped:
        return {}
    try:
        expression = ast.parse(
            f"_kwargs({stripped})" if not stripped.startswith("{") else stripped,
            mode="eval",
        ).body
    except SyntaxError as exc:
        raise _input_error(_DICT_INPUT_MESSAGE) from exc
    if isinstance(expression, ast.Call):
        if expression.args:
            raise _input_error(_DICT_INPUT_MESSAGE)
        kwargs: dict[str, typing.Any] = {}
        for keyword in expression.keywords:
            if keyword.arg is None:
                raise _input_error(
                    "Enter explicit keyword arguments; avoid ** unpacking."
                )
            try:
                kwargs[keyword.arg] = _literal_from_ast(
                    keyword.value, allow_slice=allow_slice
                )
            except FigureComposerInputError as exc:
                raise _input_error(_DICT_INPUT_MESSAGE) from exc
        return kwargs
    if not isinstance(expression, ast.Dict):
        raise _input_error(_DICT_INPUT_MESSAGE)
    dict_kwargs: dict[typing.Any, typing.Any] = {}
    for key_node, value_node in zip(expression.keys, expression.values, strict=True):
        if key_node is None:
            raise _input_error("Enter explicit dictionary keys; avoid ** unpacking.")
        try:
            key = _literal_from_ast(key_node, allow_slice=allow_slice)
            value = _literal_from_ast(value_node, allow_slice=allow_slice)
        except FigureComposerInputError as exc:
            raise _input_error(_DICT_INPUT_MESSAGE) from exc
        dict_kwargs[key] = value
    return dict_kwargs


def _format_pair(value: tuple[float, float] | None) -> str:
    if value is None:
        return ""
    return f"{value[0]:g}, {value[1]:g}"


def _format_limit_bound(value: float | None) -> str:
    return "None" if value is None else f"{value:g}"


def _format_limit_pair(value: tuple[float | None, float | None] | None) -> str:
    if value is None:
        return ""
    return f"{_format_limit_bound(value[0])}, {_format_limit_bound(value[1])}"


def _format_plot_limit(value: FigureLimit | None) -> str:
    if value is None:
        return ""
    if isinstance(value, int | float):
        return f"{value:g}"
    return _format_limit_pair(value)


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
    return erlab.utils._code.format_kwargs(value)


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
    return erlab.utils._code.format_call_kwargs(kwargs)


def _code_args(args: Sequence[typing.Any]) -> str:
    return ", ".join(erlab.utils._code._parse_single_arg(arg) for arg in args)


class _RawCode:
    def __init__(self, code: str) -> None:
        self.code = code

    def __repr__(self) -> str:
        return self.code

    def __str__(self) -> str:
        return self.code
