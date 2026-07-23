"""DataArray-backed plot and errorbar value semantics."""

from __future__ import annotations

import typing

import erlab.utils._code
from erlab.interactive._figurecomposer._code import _needs_squeeze_drop
from erlab.interactive._figurecomposer._model._sources import (
    _public_source_data,
    _valid_source_variable,
)
from erlab.interactive._figurecomposer._model._state import (
    FigureMethodPlotValueState,
    FigureOperationState,
)
from erlab.interactive._figurecomposer._operations._method._state import (
    _is_axes_errorbar_method,
)
from erlab.interactive._figurecomposer._text import _RawCode

if typing.TYPE_CHECKING:
    from collections.abc import Mapping

    import xarray as xr

    from erlab.interactive._figurecomposer._operations._method._catalog import (
        MethodSpec,
    )


_PLOT_DATA_VALUE: tuple[str, str | None] = ("data", None)


_PLOT_DATA_AXES: typing.TypeAlias = typing.Literal["x", "y", "xerr", "yerr"]


def _plot_method_call_name(spec: MethodSpec) -> str:
    return f"ax.{spec.call_name}"


def _plot_axis_label(axis: _PLOT_DATA_AXES) -> str:
    return {
        "x": "X",
        "y": "Y",
        "xerr": "X error",
        "yerr": "Y error",
    }[axis]


def _plot_axis_required(spec: MethodSpec, axis: _PLOT_DATA_AXES) -> bool:
    return axis == "y" or (axis == "x" and _is_axes_errorbar_method(spec))


def _plot_axis_field(axis: _PLOT_DATA_AXES) -> str:
    return {
        "x": "method_plot_x",
        "y": "method_plot_y",
        "xerr": "method_plot_xerr",
        "yerr": "method_plot_yerr",
    }[axis]


def _plot_axis_value_state(
    operation: FigureOperationState, axis: _PLOT_DATA_AXES
) -> FigureMethodPlotValueState | None:
    return typing.cast(
        "FigureMethodPlotValueState | None",
        getattr(operation, _plot_axis_field(axis)),
    )


def _plot_axis_source(
    operation: FigureOperationState, axis: _PLOT_DATA_AXES
) -> str | None:
    state = _plot_axis_value_state(operation, axis)
    return None if state is None else state.source


def _plot_coord_by_name(
    data: xr.DataArray, name: str
) -> tuple[typing.Hashable, xr.DataArray] | None:
    coord = data.coords.get(name)
    if coord is not None:
        return name, coord
    for coord_name, coord_data in data.coords.items():
        if str(coord_name) == name:
            return coord_name, coord_data
    return None


def _plot_value_options(
    source_data: Mapping[str, xr.DataArray], source: str | None
) -> tuple[tuple[str, tuple[str, str | None]], ...]:
    if source is None:
        return ()
    data = source_data.get(source)
    if data is None:
        return ()
    data = _public_source_data(data)
    options: list[tuple[str, tuple[str, str | None]]] = []
    if data.squeeze(drop=True).ndim == 1:
        options.append(("Data values", _PLOT_DATA_VALUE))
    seen = {_PLOT_DATA_VALUE}
    for coord_name, coord in data.coords.items():
        combo_data = ("coord", str(coord_name))
        if combo_data in seen or coord.squeeze(drop=True).ndim != 1:
            continue
        seen.add(combo_data)
        options.append((str(coord_name), combo_data))
    return tuple(options)


def _default_plot_value_state(
    source_data: Mapping[str, xr.DataArray], source: str
) -> FigureMethodPlotValueState:
    options = _plot_value_options(source_data, source)
    if options:
        kind, name = options[0][1]
        return FigureMethodPlotValueState(
            source=source,
            kind=typing.cast("typing.Literal['data', 'coord']", kind),
            name=name,
        )
    return FigureMethodPlotValueState(source=source, kind="data")


def _plot_value_data(
    source_data: Mapping[str, xr.DataArray], state: FigureMethodPlotValueState
) -> xr.DataArray:
    source = source_data.get(state.source)
    if source is None:
        raise ValueError(f"DataArray {state.source!r} is not available")
    data = _public_source_data(source)
    if state.kind == "data":
        value = data.squeeze(drop=True)
        if value.ndim != 1:
            raise ValueError("Picked plot data values must be one-dimensional")
        return value
    if state.name is None:
        raise ValueError("Choose a coordinate for picked plot data")
    coord = _plot_coord_by_name(data, state.name)
    if coord is None:
        raise ValueError(
            f"Coordinate {state.name!r} is not available in DataArray {state.source!r}"
        )
    _coord_key, coord_data = coord
    value = coord_data.squeeze(drop=True)
    if value.ndim != 1:
        raise ValueError("Picked plot coordinates must be one-dimensional")
    return value


def _plot_value_code_and_data(
    source_data: Mapping[str, xr.DataArray], state: FigureMethodPlotValueState
) -> tuple[_RawCode, xr.DataArray]:
    source = source_data.get(state.source)
    if source is None:
        raise ValueError(f"DataArray {state.source!r} is not available")
    data = _public_source_data(source)
    source_code = _valid_source_variable(state.source)
    if state.kind == "data":
        value = data.squeeze(drop=True)
        if value.ndim != 1:
            raise ValueError("Picked plot data values must be one-dimensional")
        code = source_code
        if _needs_squeeze_drop(data):
            code = f"{code}.squeeze(drop=True)"
        return _RawCode(f"{code}.values"), value
    if state.name is None:
        raise ValueError("Choose a coordinate for picked plot data")
    coord = _plot_coord_by_name(data, state.name)
    if coord is None:
        raise ValueError(
            f"Coordinate {state.name!r} is not available in DataArray {state.source!r}"
        )
    coord_key, coord_data = coord
    value = coord_data.squeeze(drop=True)
    if value.ndim != 1:
        raise ValueError("Picked plot coordinates must be one-dimensional")
    code = f"{source_code}.coords[{erlab.utils._code._parse_single_arg(coord_key)}]"
    if _needs_squeeze_drop(coord_data):
        code = f"{code}.squeeze(drop=True)"
    return _RawCode(f"{code}.values"), value


def _validate_plot_value_lengths(
    x_value: xr.DataArray | None,
    y_value: xr.DataArray,
    *error_values: tuple[str, xr.DataArray],
) -> None:
    if x_value is not None and x_value.size != y_value.size:
        raise ValueError("Picked plot X and Y values must have the same length")
    for label, value in error_values:
        if value.size != y_value.size:
            raise ValueError(
                f"Picked ax.errorbar {label} and Y values must have the same length"
            )


def _picked_plot_args(
    source_data: Mapping[str, xr.DataArray],
    operation: FigureOperationState,
    spec: MethodSpec,
) -> tuple[typing.Any, ...]:
    return _picked_plot_args_from_states(
        source_data,
        x_state=operation.method_plot_x,
        y_state=operation.method_plot_y,
        spec=spec,
    )


def _picked_plot_args_from_states(
    source_data: Mapping[str, xr.DataArray],
    *,
    x_state: FigureMethodPlotValueState | None,
    y_state: FigureMethodPlotValueState | None,
    spec: MethodSpec,
) -> tuple[typing.Any, ...]:
    if y_state is None:
        raise ValueError(f"Choose Y values for {_plot_method_call_name(spec)}")
    y_value = _plot_value_data(source_data, y_state)
    x_value = None if x_state is None else _plot_value_data(source_data, x_state)
    if x_value is None and _plot_axis_required(spec, "x"):
        raise ValueError(f"Choose X values for {_plot_method_call_name(spec)}")
    _validate_plot_value_lengths(x_value, y_value)
    if x_value is None:
        return (y_value.values,)
    return x_value.values, y_value.values


def _picked_plot_code_args(
    source_data: Mapping[str, xr.DataArray],
    operation: FigureOperationState,
    spec: MethodSpec,
) -> tuple[typing.Any, ...]:
    if operation.method_plot_y is None:
        raise ValueError(f"Choose Y values for {_plot_method_call_name(spec)}")
    y_code, y_value = _plot_value_code_and_data(source_data, operation.method_plot_y)
    if operation.method_plot_x is None:
        if _plot_axis_required(spec, "x"):
            raise ValueError(f"Choose X values for {_plot_method_call_name(spec)}")
        return (y_code,)
    x_code, x_value = _plot_value_code_and_data(source_data, operation.method_plot_x)
    _validate_plot_value_lengths(x_value, y_value)
    return x_code, y_code


def _validate_entered_errorbar_args(args: tuple[typing.Any, ...]) -> None:
    if len(args) < 2 or any(
        value is None or (isinstance(value, (list, tuple)) and len(value) == 0)
        for value in args[:2]
    ):
        raise ValueError("Enter X and Y values for ax.errorbar")


def _picked_plot_error_kwargs(
    source_data: Mapping[str, xr.DataArray], operation: FigureOperationState
) -> dict[str, typing.Any]:
    return _picked_plot_error_kwargs_from_states(
        source_data,
        y_state=operation.method_plot_y,
        xerr_state=operation.method_plot_xerr,
        yerr_state=operation.method_plot_yerr,
    )


def _picked_plot_error_kwargs_from_states(
    source_data: Mapping[str, xr.DataArray],
    *,
    y_state: FigureMethodPlotValueState | None,
    xerr_state: FigureMethodPlotValueState | None,
    yerr_state: FigureMethodPlotValueState | None,
) -> dict[str, typing.Any]:
    if y_state is None:
        return {}
    y_value = _plot_value_data(source_data, y_state)
    kwargs: dict[str, typing.Any] = {}
    error_values = []
    states: tuple[tuple[_PLOT_DATA_AXES, FigureMethodPlotValueState | None], ...] = (
        ("xerr", xerr_state),
        ("yerr", yerr_state),
    )
    for axis, state in states:
        if state is None:
            continue
        value = _plot_value_data(source_data, state)
        error_values.append((_plot_axis_label(axis), value))
        kwargs[axis] = value.values
    _validate_plot_value_lengths(None, y_value, *error_values)
    return kwargs


def _picked_plot_error_code_kwargs(
    source_data: Mapping[str, xr.DataArray], operation: FigureOperationState
) -> dict[str, typing.Any]:
    if operation.method_plot_y is None:
        return {}
    _y_code, y_value = _plot_value_code_and_data(source_data, operation.method_plot_y)
    kwargs: dict[str, typing.Any] = {}
    error_values = []
    for axis in ("xerr", "yerr"):
        state = _plot_axis_value_state(operation, axis)
        if state is None:
            continue
        code, value = _plot_value_code_and_data(source_data, state)
        error_values.append((_plot_axis_label(axis), value))
        kwargs[axis] = code
    _validate_plot_value_lengths(None, y_value, *error_values)
    return kwargs
