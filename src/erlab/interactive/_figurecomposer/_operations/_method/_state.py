"""Recipe-state semantics for curated Figure Composer methods."""

from __future__ import annotations

import contextlib
import typing

from matplotlib.figure import Figure

import erlab.interactive.utils
from erlab.interactive._figurecomposer._model._operation_metadata import (
    is_axes_errorbar_data_method,
    is_axes_plot_data_method,
)
from erlab.interactive._figurecomposer._model._state import (
    FigureAxesSelectionState,
    FigureMethodFamily,
    FigureOperationState,
)
from erlab.interactive._figurecomposer._operations._method._catalog import (
    _LAYOUT_ENGINE_KWARGS,
    MethodCallPolicy,
    MethodControlKind,
    MethodControlSpec,
    MethodSpec,
    MethodTargetDomain,
    MethodTextValuesPolicy,
    _method_spec,
)
from erlab.interactive._figurecomposer._text import (
    _code_args,
    _dict_from_text,
    _format_dict,
    _format_literal_sequence,
    _literal_from_text,
    _string_tuple_from_text,
)

if typing.TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from matplotlib.axes import Axes

    from erlab.interactive._figurecomposer._tool import FigureComposerTool


def _limit_method_default_args(
    spec: MethodSpec,
    axis: Axes | None,
) -> tuple[typing.Any, ...]:
    if axis is None:
        return ()
    limits = axis.get_xlim() if spec.name == "set_xlim" else axis.get_ylim()
    return float(limits[0]), float(limits[1])


def _default_method_args(
    spec: MethodSpec,
    axis: Axes | None,
) -> tuple[typing.Any, ...]:
    if spec.family == FigureMethodFamily.AXES and spec.name in {"set_xlim", "set_ylim"}:
        return _limit_method_default_args(spec, axis)
    return spec.default_args


def _empty_text_as_none(text: str) -> str | None:
    return text or None


def _string_tuple_from_text_or_none(text: str) -> tuple[str, ...] | None:
    return _string_tuple_from_text(text) or None


def _method_arg_value(
    operation: FigureOperationState,
    spec: MethodSpec,
    index: int,
    default: typing.Any,
) -> typing.Any:
    args = _method_args(operation, spec)
    return args[index] if index < len(args) else default


def _method_kwarg_value(
    operation: FigureOperationState, key: str, default: typing.Any
) -> typing.Any:
    return operation.method_kwargs.get(key, default)


def _is_axes_plot_method(spec: MethodSpec) -> bool:
    return is_axes_plot_data_method(spec.family, spec.name)


def _is_axes_errorbar_method(spec: MethodSpec) -> bool:
    return is_axes_errorbar_data_method(spec.family, spec.name)


def _tool_subplots_adjust_defaults(
    tool: FigureComposerTool,
) -> dict[str, float]:
    figure_window = tool._figure_window
    if figure_window is not None and erlab.interactive.utils.qt_is_valid(figure_window):
        subplotpars = figure_window.figure.subplotpars
    else:
        figure = Figure(
            figsize=tool.tool_status.setup.figsize,
            dpi=tool.tool_status.setup.dpi,
            layout=typing.cast("typing.Any", tool.tool_status.setup.layout),
        )
        subplotpars = figure.subplotpars
    return {
        key: float(getattr(subplotpars, key))
        for key in ("left", "bottom", "right", "top", "wspace", "hspace")
    }


def _subplots_adjust_values(
    operation: FigureOperationState, defaults: Mapping[str, float]
) -> dict[str, float]:
    values: dict[str, float] = {}
    for key in ("left", "bottom", "right", "top", "wspace", "hspace"):
        default = defaults[key]
        value = _method_kwarg_value(operation, key, default)
        with contextlib.suppress(TypeError, ValueError):
            values[key] = float(value)
            continue
        values[key] = default
    return values


def _is_layout_engine_method(spec: MethodSpec) -> bool:
    return spec.family == FigureMethodFamily.FIGURE and spec.name == "set_layout_engine"


def _layout_engine_name(operation: FigureOperationState, spec: MethodSpec) -> str:
    args = _method_args(operation, spec)
    if not args:
        return "default"
    layout = args[0]
    return layout if isinstance(layout, str) else "default"


def _layout_engine_kwarg_keys(layout: str) -> frozenset[str]:
    return _LAYOUT_ENGINE_KWARGS.get(layout, frozenset[str]())


def _filter_layout_engine_kwargs(
    args: Sequence[typing.Any], kwargs: dict[str, typing.Any]
) -> dict[str, typing.Any]:
    if not args or not isinstance(args[0], str):
        return kwargs
    allowed_keys = _layout_engine_kwarg_keys(args[0])
    return {key: value for key, value in kwargs.items() if key in allowed_keys}


def _is_subplots_adjust_method(spec: MethodSpec) -> bool:
    return spec.family == FigureMethodFamily.FIGURE and spec.name == "subplots_adjust"


def _format_int_value(value: typing.Any) -> str:
    return "" if value is None else str(int(value))


def _format_float_value(value: typing.Any) -> str:
    return "" if value is None else f"{float(value):g}"


def _format_literal_value(value: typing.Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return _format_dict(value)
    if isinstance(value, (list, tuple)):
        return _format_literal_sequence(value)
    return _code_args((value,))


def _format_aspect_value(value: typing.Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, int | float):
        return f"{float(value):g}"
    return _code_args((value,))


def _literal_value_from_text(text: str) -> typing.Any:
    stripped = text.strip()
    if not stripped:
        return None
    if stripped.startswith("{") or "=" in stripped:
        return _dict_from_text(stripped)
    return _literal_from_text(stripped)


def _aspect_value_from_text(text: str) -> str | float | None:
    stripped = text.strip()
    if not stripped:
        return None
    if stripped in {"auto", "equal"}:
        return stripped
    try:
        value = _literal_from_text(stripped)
    except ValueError:
        return stripped
    if isinstance(value, str):
        return value
    if not isinstance(value, int | float):
        return stripped
    return float(value)


def _optional_literal_from_text(text: str) -> typing.Any:
    stripped = text.strip()
    if not stripped:
        return None
    return _literal_value_from_text(stripped)


def _optional_float_from_text(text: str) -> float | None:
    stripped = text.strip()
    return None if not stripped else float(stripped)


def _optional_int_from_text(text: str) -> int | None:
    stripped = text.strip()
    return None if not stripped else int(stripped)


def _method_args(
    operation: FigureOperationState,
    spec: MethodSpec,
    *,
    default_args: Sequence[typing.Any] | None = None,
) -> tuple[typing.Any, ...]:
    if operation.method_args:
        return operation.method_args
    return tuple(spec.default_args if default_args is None else default_args)


def _label_values(operation: FigureOperationState) -> str | list[str]:
    values = list(operation.text_values)
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    return values


def _method_has_transform_control(spec: MethodSpec) -> bool:
    return any(control.kind == MethodControlKind.TRANSFORM for control in spec.controls)


def _control_accepts_value(control: MethodControlSpec, value: typing.Any) -> bool:
    if control.kind in {
        MethodControlKind.ARG_COMBO,
        MethodControlKind.KWARG_COMBO,
    }:
        if value is None:
            return control.none_label is not None
        return str(value) in control.options
    if control.kind in {
        MethodControlKind.BOOL_ARG_COMBO,
        MethodControlKind.BOOL_KWARG_COMBO,
    }:
        return isinstance(value, bool)
    if control.kind == MethodControlKind.OPTIONAL_BOOL_KWARG_COMBO:
        return value is None or isinstance(value, bool)
    return True


def _transfer_axis_label_loc(
    source_spec: MethodSpec, target_spec: MethodSpec, value: typing.Any
) -> typing.Any:
    if source_spec.name == "set_xlabel" and target_spec.name == "set_ylabel":
        return {"left": "bottom", "center": "center", "right": "top"}.get(value, value)
    if source_spec.name == "set_ylabel" and target_spec.name == "set_xlabel":
        return {"bottom": "left", "center": "center", "top": "right"}.get(value, value)
    return value


def _method_transfer_updates(
    operation: FigureOperationState,
    target_spec: MethodSpec,
    *,
    default_axis: Axes | None,
) -> dict[str, typing.Any]:
    source_spec = _method_spec(operation)
    axes = (
        operation.axes
        if target_spec.target_domain == MethodTargetDomain.AXES
        else FigureAxesSelectionState(axes=())
    )
    source_args = {
        control.arg_index: control
        for control in source_spec.controls
        if control.arg_index is not None
    }
    target_args = {
        control.arg_index: control
        for control in target_spec.controls
        if control.arg_index is not None
    }
    args = list(_default_method_args(target_spec, default_axis))
    source_kinds = {control.kind for control in source_spec.controls}
    target_kinds = {control.kind for control in target_spec.controls}
    if (
        operation.method_args
        and MethodControlKind.FLOAT_PAIR_ARGS in source_kinds
        and MethodControlKind.FLOAT_PAIR_ARGS in target_kinds
    ) or (
        operation.method_args
        and MethodControlKind.PLOT_DATA_ARGS in source_kinds
        and MethodControlKind.PLOT_DATA_ARGS in target_kinds
    ):
        args = list(operation.method_args)
    elif operation.method_args:
        for index, target_control in target_args.items():
            source_control = source_args.get(index)
            if (
                source_control is None
                or source_control.kind != target_control.kind
                or index >= len(operation.method_args)
            ):
                continue
            value = operation.method_args[index]
            if (
                index < len(source_spec.default_args)
                and value == source_spec.default_args[index]
            ):
                continue
            if not _control_accepts_value(target_control, value):
                continue
            while len(args) <= index:
                args.append(None)
            args[index] = value

    source_kwargs = {
        control.key: control
        for control in source_spec.controls
        if control.key is not None
    }
    target_kwargs = {
        control.key: control
        for control in target_spec.controls
        if control.key is not None
    }
    source_signature = tuple(
        (control.kind, control.arg_index, control.key, control.none_label is not None)
        for control in source_spec.controls
        if control.kind != MethodControlKind.TRANSFORM
    )
    target_signature = tuple(
        (control.kind, control.arg_index, control.key, control.none_label is not None)
        for control in target_spec.controls
        if control.kind != MethodControlKind.TRANSFORM
    )
    transfer_extra_kwargs = (
        source_spec.family == target_spec.family
        and source_spec.target_domain == target_spec.target_domain
        and source_spec.allow_extra_kwargs
        and target_spec.allow_extra_kwargs
        and source_signature == target_signature
    )
    kwargs: dict[str, typing.Any] = {}
    for key, value in operation.method_kwargs.items():
        target_control = target_kwargs.get(key)
        if target_control is None:
            if key not in source_kwargs and transfer_extra_kwargs:
                kwargs[key] = value
            continue
        source_control = source_kwargs.get(key)
        if source_control is None or source_control.kind != target_control.kind:
            continue
        if key == "loc":
            value = _transfer_axis_label_loc(source_spec, target_spec, value)
        if _control_accepts_value(target_control, value):
            kwargs[key] = value

    method_call_policy = None
    if operation.method_call_policy is not None:
        with contextlib.suppress(ValueError):
            policy = MethodCallPolicy(operation.method_call_policy)
            if policy in target_spec.selectable_call_policies:
                method_call_policy = (
                    None if policy == target_spec.call_policy else policy.value
                )

    text_values = ()
    if (
        source_spec.text_values_policy != MethodTextValuesPolicy.NONE
        and source_spec.text_values_policy == target_spec.text_values_policy
        and (
            source_spec.text_values_policy != MethodTextValuesPolicy.KWARG
            or source_spec.text_values_kwarg == target_spec.text_values_kwarg
        )
    ):
        text_values = operation.text_values

    updates: dict[str, typing.Any] = {
        "label": target_spec.label,
        "method_name": target_spec.name,
        "method_args": tuple(args),
        "method_kwargs": kwargs,
        "method_call_policy": method_call_policy,
        "method_plot_data_mode": "entered",
        "method_plot_x": None,
        "method_plot_y": None,
        "method_plot_xerr": None,
        "method_plot_yerr": None,
        "text_values": text_values,
        "method_transform": "data",
        "method_transform_x": "data",
        "method_transform_y": "axes",
        "method_transform_expression": "",
        "axes": axes,
    }
    if _method_has_transform_control(source_spec) and _method_has_transform_control(
        target_spec
    ):
        updates.update(
            {
                "method_transform": operation.method_transform,
                "method_transform_x": operation.method_transform_x,
                "method_transform_y": operation.method_transform_y,
                "method_transform_expression": operation.method_transform_expression,
            }
        )
    if _is_axes_plot_method(source_spec) and _is_axes_plot_method(target_spec):
        updates.update(
            {
                "method_plot_data_mode": operation.method_plot_data_mode,
                "method_plot_x": operation.method_plot_x,
                "method_plot_y": operation.method_plot_y,
            }
        )
        if _is_axes_errorbar_method(source_spec) and _is_axes_errorbar_method(
            target_spec
        ):
            updates.update(
                {
                    "method_plot_xerr": operation.method_plot_xerr,
                    "method_plot_yerr": operation.method_plot_yerr,
                }
            )
    if not _is_axes_errorbar_method(target_spec):
        kwargs.pop("xerr", None)
        kwargs.pop("yerr", None)
    return updates


def _loaded_operation(operation: FigureOperationState) -> FigureOperationState:
    if operation.method_transform == "custom":
        return operation.model_copy(update={"trusted": False})
    return operation
