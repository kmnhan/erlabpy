"""Pure helpers for generating executable Python expressions."""

from __future__ import annotations

import json
import sys
import typing

import numpy as np
import numpy.typing as npt

import erlab.utils.misc

if typing.TYPE_CHECKING:
    from collections.abc import Mapping


def format_kwargs(d: Mapping[typing.Any, typing.Any]) -> str:
    """Format a mapping as keyword arguments or a dictionary literal.

    Identifier keys are emitted as ordinary keyword arguments. Other keys are
    preserved in an executable dictionary literal.

    Parameters
    ----------
    d
        Mapping of argument names or keys to values.

    """
    if all(erlab.utils.misc._is_valid_identifier(key) for key in d):
        return ", ".join(
            f"{key}={_parse_single_arg(value)}" for key, value in d.items()
        )
    entries = ", ".join(
        f"{_parse_single_arg(key)}: {_parse_single_arg(value)}"
        for key, value in d.items()
    )
    return "{" + entries + "}"


def format_call_kwargs(d: Mapping[typing.Any, typing.Any]) -> str:
    """Format a mapping for use as arguments at a call site.

    String keys that are not identifiers use ``**{...}`` so they retain keyword-call
    semantics. Mappings with non-string keys remain positional dictionary literals.
    """
    string_keys = [key for key in d if isinstance(key, str)]
    if len(string_keys) == len(d):
        if all(erlab.utils.misc._is_valid_identifier(key) for key in string_keys):
            return format_kwargs(d)
        return f"**{format_kwargs(d)}"
    return format_kwargs(d)


def _parse_single_arg(arg: typing.Any) -> str:
    arg = erlab.utils.misc._convert_to_native(arg)

    if isinstance(arg, str):
        if arg.startswith("|") and arg.endswith("|"):
            arg = arg[1:-1]
        else:
            arg = json.dumps(arg, ensure_ascii=False)
    elif isinstance(arg, tuple):
        inner = ", ".join(_parse_single_arg(item) for item in arg)
        if len(arg) == 1:
            inner += ","
        arg = f"({inner})"
    elif isinstance(arg, list):
        arg = "[" + ", ".join(_parse_single_arg(item) for item in arg) + "]"
    elif isinstance(arg, dict):
        converted = {
            erlab.utils.misc._convert_to_native(key): (
                erlab.utils.misc._convert_to_native(value)
            )
            for key, value in arg.items()
        }
        arg = (
            "{"
            + ", ".join(
                f"{_parse_single_arg(key)}: {_parse_single_arg(value)}"
                for key, value in converted.items()
            )
            + "}"
        )
    elif isinstance(arg, slice):
        start, stop, step = arg.start, arg.stop, arg.step
        if step is not None:
            args = [start, stop, step]
        elif start is None:
            args = [stop]
        else:
            args = [start, stop]
        return f"slice({', '.join(_parse_single_arg(value) for value in args)})"
    elif isinstance(arg, np.ndarray):
        array = np.array2string(
            arg,
            separator=", ",
            threshold=sys.maxsize,
            formatter={
                "float_kind": lambda value: np.format_float_positional(value, trim="-")
            },
        ).replace("\n", "")
        arg = f"np.array({array})"

    return str(arg)


def format_1d_numeric_array_code(values: npt.ArrayLike) -> str:
    """Format one-dimensional numeric values as compact executable Python code.

    Uniform integer-like grids use ``np.arange`` and uniform floating-point grids use
    ``np.linspace``. Other inputs use the general array literal formatter.
    """
    values = np.asarray(values)
    values_code = _parse_single_arg(values)
    if values.ndim != 1 or values.size == 0:
        return values_code
    if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
        values.dtype, np.complexfloating
    ):
        return values_code

    numeric = values.astype(np.float64, copy=False)
    if not np.all(np.isfinite(numeric)):
        return values_code

    if numeric.size == 1:
        value_code = _parse_single_arg(float(numeric[0]))
        return f"np.linspace({value_code}, {value_code}, 1)"

    diffs = np.diff(numeric)
    step = float(diffs[0])
    if not np.allclose(diffs, step, rtol=1e-12, atol=1e-12):
        return values_code

    rounded = np.rint(numeric)
    if np.allclose(numeric, rounded, rtol=0.0, atol=1e-12):
        int_step = int(rounded[1] - rounded[0])
        if int_step != 0:
            start = int(rounded[0])
            stop = int(rounded[-1] + int_step)
            return f"np.arange({start}, {stop}, {int_step})"

    start_code = _parse_single_arg(float(numeric[0]))
    stop_code = _parse_single_arg(float(numeric[-1]))
    return f"np.linspace({start_code}, {stop_code}, {numeric.size})"
