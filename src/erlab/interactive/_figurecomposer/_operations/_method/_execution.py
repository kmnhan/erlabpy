"""Rendering and generated-code execution for method operations."""

from __future__ import annotations

import dataclasses
import typing

import matplotlib.transforms as mtransforms
from matplotlib.figure import Figure

import erlab.interactive.utils
import erlab.plotting as eplt
from erlab.interactive._figurecomposer._code import _axes_code, _axes_sequence_code
from erlab.interactive._figurecomposer._model._gridspec import (
    _gridspec_all_axes_ids,
    _gridspec_valid_axes_ids,
)
from erlab.interactive._figurecomposer._model._state import (
    FigureAxesSelectionState,
    FigureMethodFamily,
    FigureMethodPlotValueState,
    FigureOperationState,
)
from erlab.interactive._figurecomposer._operations._method._catalog import (
    MethodCallPolicy,
    MethodSpec,
    MethodTargetDomain,
    MethodTextValuesPolicy,
    _effective_call_policy,
    _method_spec,
)
from erlab.interactive._figurecomposer._operations._method._plot_data import (
    _picked_plot_args_from_states,
    _picked_plot_code_args,
    _picked_plot_error_code_kwargs,
    _picked_plot_error_kwargs_from_states,
    _validate_entered_errorbar_args,
)
from erlab.interactive._figurecomposer._operations._method._state import (
    _default_method_args,
    _filter_layout_engine_kwargs,
    _is_axes_errorbar_method,
    _is_axes_plot_method,
    _is_layout_engine_method,
    _is_subplots_adjust_method,
    _label_values,
    _method_args,
    _method_has_transform_control,
    _subplots_adjust_values,
)
from erlab.interactive._figurecomposer._rendering import (
    _axes_from_selection,
    _iter_axes,
    _live_layout_axes,
)
from erlab.interactive._figurecomposer._subplot_adjust import (
    normalize_subplots_adjust_kwargs,
)
from erlab.interactive._figurecomposer._text import _code_args, _code_kwargs, _RawCode

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from matplotlib.axes import Axes

    from erlab.interactive._figurecomposer._tool import FigureComposerTool


_TransformComponent = typing.Literal["data", "axes", "figure", "dpi"]


@dataclasses.dataclass(frozen=True)
class _MethodPlotDataPlan:
    """Semantic source selections used by a data-backed method call."""

    family: FigureMethodFamily
    name: str
    mode: typing.Literal["entered", "from_data"]
    x: FigureMethodPlotValueState | None
    y: FigureMethodPlotValueState | None
    xerr: FigureMethodPlotValueState | None
    yerr: FigureMethodPlotValueState | None

    @classmethod
    def from_operation(cls, operation: FigureOperationState) -> _MethodPlotDataPlan:
        return cls(
            family=operation.method_family,
            name=operation.method_name,
            mode=operation.method_plot_data_mode,
            x=operation.method_plot_x,
            y=operation.method_plot_y,
            xerr=operation.method_plot_xerr,
            yerr=operation.method_plot_yerr,
        )

    def prepare(
        self,
        tool: FigureComposerTool,
        spec: MethodSpec,
    ) -> tuple[tuple[typing.Any, ...], dict[str, typing.Any]]:
        args = _picked_plot_args_from_states(
            tool._document.source_data,
            x_state=self.x,
            y_state=self.y,
            spec=spec,
        )
        error_kwargs = (
            _picked_plot_error_kwargs_from_states(
                tool._document.source_data,
                y_state=self.y,
                xerr_state=self.xerr,
                yerr_state=self.yerr,
            )
            if _is_axes_errorbar_method(spec)
            else {}
        )
        return args, error_kwargs


def _picked_plot_render_data(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    spec: MethodSpec,
) -> tuple[tuple[typing.Any, ...], dict[str, typing.Any]]:
    plan = _MethodPlotDataPlan.from_operation(operation)
    return tool._cached_render_data(
        "method-plot-data",
        plan,
        lambda: plan.prepare(tool, spec),
    )


def _first_live_axis(
    tool: FigureComposerTool, selection: FigureAxesSelectionState
) -> Axes | None:
    layout_axes = _live_layout_axes(tool, render_if_missing=True)
    if layout_axes is None:
        return None
    if isinstance(layout_axes, dict) and not selection.axes_ids:
        selection = selection.model_copy(
            update={
                "axes_ids": _gridspec_valid_axes_ids(
                    tool._document.recipe.setup,
                    _gridspec_all_axes_ids(tool._document.recipe.setup),
                )[:1]
            }
        )
    try:
        selected_axes = _axes_from_selection(
            tool, selection, layout_axes, for_plot_slices=False
        )
    except (IndexError, TypeError, ValueError):
        return None
    axes = _iter_axes(selected_axes)
    return axes[0] if axes else None


def _method_call_args(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    spec: MethodSpec,
    *,
    default_axis: Axes | None = None,
) -> tuple[typing.Any, ...]:
    if _is_axes_plot_method(spec) and operation.method_plot_data_mode == "from_data":
        return _picked_plot_render_data(tool, operation, spec)[0]
    if (
        default_axis is None
        and spec.family == FigureMethodFamily.AXES
        and spec.name in {"set_xlim", "set_ylim"}
    ):
        default_axis = _first_live_axis(tool, operation.axes)
    args = _method_args(
        operation,
        spec,
        default_args=_default_method_args(spec, default_axis),
    )
    if _is_axes_errorbar_method(spec):
        _validate_entered_errorbar_args(args)
    return args


def _method_code_call_args(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    spec: MethodSpec,
) -> tuple[typing.Any, ...]:
    if _is_axes_plot_method(spec) and operation.method_plot_data_mode == "from_data":
        return _picked_plot_code_args(tool._document.source_data, operation, spec)
    default_axis = None
    if spec.family == FigureMethodFamily.AXES and spec.name in {"set_xlim", "set_ylim"}:
        default_axis = _first_live_axis(tool, operation.axes)
    args = _method_args(
        operation,
        spec,
        default_args=_default_method_args(spec, default_axis),
    )
    if _is_axes_errorbar_method(spec):
        _validate_entered_errorbar_args(args)
    return args


def _transform_component(
    figure: Figure, axis: Axes, component: _TransformComponent
) -> mtransforms.Transform:
    match component:
        case "data":
            return axis.transData
        case "axes":
            return axis.transAxes
        case "figure":
            return figure.transFigure
        case "dpi":
            return figure.dpi_scale_trans


def _transform_component_code(
    component: _TransformComponent, *, axis_code: str = "ax"
) -> str:
    match component:
        case "data":
            return f"{axis_code}.transData"
        case "axes":
            return f"{axis_code}.transAxes"
        case "figure":
            return "fig.transFigure"
        case "dpi":
            return "fig.dpi_scale_trans"


def _render_method_transform(
    operation: FigureOperationState,
    spec: MethodSpec,
    *,
    figure: Figure,
    axis: Axes,
) -> mtransforms.Transform | None:
    if not _method_has_transform_control(spec):
        return None
    match operation.method_transform:
        case "data":
            return None
        case "axes":
            return axis.transAxes
        case "figure":
            return figure.transFigure
        case "dpi":
            return figure.dpi_scale_trans
        case "xaxis":
            return axis.get_xaxis_transform()
        case "yaxis":
            return axis.get_yaxis_transform()
        case "blend":
            return mtransforms.blended_transform_factory(
                _transform_component(figure, axis, operation.method_transform_x),
                _transform_component(figure, axis, operation.method_transform_y),
            )
        case "custom":
            if not operation.trusted:
                raise ValueError("Custom transform expression is not trusted")
            expression = operation.method_transform_expression.strip()
            if not expression:
                raise ValueError("Custom transform expression is empty")
            namespace = {"ax": axis, "fig": figure, "mtransforms": mtransforms}
            transform = eval(expression, {"__builtins__": {}}, namespace)  # noqa: S307
            if not isinstance(transform, mtransforms.Transform):
                raise TypeError("Custom transform expression must return a Transform")
            return transform


def _method_transform_code(
    operation: FigureOperationState, spec: MethodSpec, *, axis_code: str = "ax"
) -> _RawCode | None:
    if not _method_has_transform_control(spec):
        return None
    match operation.method_transform:
        case "data":
            return None
        case "axes":
            return _RawCode(f"{axis_code}.transAxes")
        case "figure":
            return _RawCode("fig.transFigure")
        case "dpi":
            return _RawCode("fig.dpi_scale_trans")
        case "xaxis":
            return _RawCode(f"{axis_code}.get_xaxis_transform()")
        case "yaxis":
            return _RawCode(f"{axis_code}.get_yaxis_transform()")
        case "blend":
            x_transform = _transform_component_code(
                operation.method_transform_x, axis_code=axis_code
            )
            y_transform = _transform_component_code(
                operation.method_transform_y, axis_code=axis_code
            )
            return _RawCode(
                f"mtransforms.blended_transform_factory({x_transform}, {y_transform})"
            )
        case "custom":
            if not operation.trusted:
                raise ValueError("Custom transform expression is not trusted")
            expression = operation.method_transform_expression.strip()
            if not expression:
                raise ValueError("Custom transform expression is empty")
            return _RawCode(expression)


def _render_args_kwargs(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    spec: MethodSpec,
    *,
    figure: Figure | None = None,
    axis: Axes | None = None,
    default_axis: Axes | None = None,
) -> tuple[tuple[typing.Any, ...], dict[str, typing.Any]]:
    args = list(
        _method_call_args(
            tool,
            operation,
            spec,
            default_axis=axis if default_axis is None else default_axis,
        )
    )
    kwargs = dict(spec.default_kwargs)
    kwargs.update(operation.method_kwargs)
    if (
        _is_axes_errorbar_method(spec)
        and operation.method_plot_data_mode == "from_data"
    ):
        kwargs.pop("xerr", None)
        kwargs.pop("yerr", None)
        kwargs.update(_picked_plot_render_data(tool, operation, spec)[1])
    if _method_has_transform_control(spec):
        kwargs.pop("transform", None)
    if spec.text_values_policy == MethodTextValuesPolicy.POSITIONAL:
        args.append(_label_values(operation))
    elif (
        spec.text_values_policy == MethodTextValuesPolicy.KWARG
        and operation.text_values
    ):
        kwargs.setdefault(spec.text_values_kwarg, list(operation.text_values))
    if axis is not None and figure is not None:
        transform = _render_method_transform(operation, spec, figure=figure, axis=axis)
        if transform is not None:
            kwargs["transform"] = transform
    if _is_layout_engine_method(spec):
        kwargs = _filter_layout_engine_kwargs(args, kwargs)
    elif _is_subplots_adjust_method(spec):
        kwargs = normalize_subplots_adjust_kwargs(
            kwargs,
            defaults=_subplots_adjust_values(
                operation, _tool_subplots_adjust_defaults(tool)
            ),
        )
    return tuple(args), kwargs


def _code_args_kwargs(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    spec: MethodSpec,
    *,
    axis_code: str | None = None,
) -> tuple[tuple[typing.Any, ...], dict[str, typing.Any]]:
    args = list(_method_code_call_args(tool, operation, spec))
    kwargs = dict(spec.default_kwargs)
    kwargs.update(operation.method_kwargs)
    if (
        _is_axes_errorbar_method(spec)
        and operation.method_plot_data_mode == "from_data"
    ):
        kwargs.pop("xerr", None)
        kwargs.pop("yerr", None)
        kwargs.update(
            _picked_plot_error_code_kwargs(tool._document.source_data, operation)
        )
    if _method_has_transform_control(spec):
        kwargs.pop("transform", None)
    if spec.text_values_policy == MethodTextValuesPolicy.POSITIONAL:
        args.append(_label_values(operation))
    elif (
        spec.text_values_policy == MethodTextValuesPolicy.KWARG
        and operation.text_values
    ):
        kwargs.setdefault(spec.text_values_kwarg, list(operation.text_values))
    transform_code = _method_transform_code(
        operation, spec, axis_code=axis_code or "ax"
    )
    if axis_code is not None and transform_code is not None:
        kwargs["transform"] = transform_code
    if _is_layout_engine_method(spec):
        kwargs = _filter_layout_engine_kwargs(args, kwargs)
    elif _is_subplots_adjust_method(spec):
        kwargs = normalize_subplots_adjust_kwargs(
            kwargs,
            defaults=_subplots_adjust_values(
                operation, _tool_subplots_adjust_defaults(tool)
            ),
        )
    return tuple(args), kwargs


def _erlab_callable(spec: MethodSpec) -> Callable[..., typing.Any]:
    return typing.cast("Callable[..., typing.Any]", getattr(eplt, spec.call_name))


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


def _render_method(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    figure: Figure,
    axs: typing.Any,
) -> None:
    spec = _method_spec(operation)
    call_policy = _effective_call_policy(operation, spec)
    axes = None
    if spec.target_domain == MethodTargetDomain.AXES:
        axes = _axes_from_selection(tool, operation.axes, axs, for_plot_slices=False)
    default_axis = None
    if axes is not None:
        selected_axes = _iter_axes(axes)
        if selected_axes:
            default_axis = selected_axes[0]

    match call_policy:
        case MethodCallPolicy.BOUND_EACH_AXIS:
            if axes is None:
                return
            for axis in _iter_axes(axes):
                args, kwargs = _render_args_kwargs(
                    tool, operation, spec, figure=figure, axis=axis
                )
                getattr(axis, spec.call_name)(*args, **kwargs)
        case MethodCallPolicy.AXES_POSITIONAL:
            if axes is None:
                return
            args, kwargs = _render_args_kwargs(
                tool, operation, spec, default_axis=default_axis
            )
            _erlab_callable(spec)(axes, *args, **kwargs)
        case MethodCallPolicy.AX_KEYWORD:
            if axes is None:
                return
            args, kwargs = _render_args_kwargs(
                tool, operation, spec, default_axis=default_axis
            )
            _erlab_callable(spec)(*args, ax=axes, **kwargs)
        case MethodCallPolicy.EACH_AXIS_AX_KEYWORD:
            if axes is None:
                return
            for axis in _iter_axes(axes):
                args, kwargs = _render_args_kwargs(
                    tool, operation, spec, default_axis=axis
                )
                _erlab_callable(spec)(*args, ax=axis, **kwargs)
        case MethodCallPolicy.BOUND_FIGURE:
            args, kwargs = _render_args_kwargs(
                tool, operation, spec, default_axis=default_axis
            )
            getattr(figure, spec.call_name)(*args, **kwargs)
        case MethodCallPolicy.FIG_KEYWORD:
            args, kwargs = _render_args_kwargs(
                tool, operation, spec, default_axis=default_axis
            )
            _erlab_callable(spec)(*args, fig=figure, **kwargs)
        case MethodCallPolicy.PLAIN_CALL:
            args, kwargs = _render_args_kwargs(
                tool, operation, spec, default_axis=default_axis
            )
            _erlab_callable(spec)(*args, **kwargs)


def _call_code(
    name: str, args: Sequence[typing.Any], kwargs: dict[str, typing.Any]
) -> str:
    args_text = _code_args(args)
    kwargs_text = _code_kwargs(kwargs)
    call_parts = [part for part in (args_text, kwargs_text) if part]
    return f"{name}({', '.join(call_parts)})"


def _method_code(
    tool: FigureComposerTool, operation: FigureOperationState
) -> list[str]:
    spec = _method_spec(operation)
    call_policy = _effective_call_policy(operation, spec)
    match call_policy:
        case MethodCallPolicy.BOUND_EACH_AXIS:
            axis_code = _single_axis_code(tool, operation)
            if axis_code is not None:
                args, kwargs = _code_args_kwargs(
                    tool, operation, spec, axis_code=axis_code
                )
                return [_call_code(f"{axis_code}.{spec.call_name}", args, kwargs)]
            args, kwargs = _code_args_kwargs(tool, operation, spec, axis_code="ax")
            call = _call_code(f"ax.{spec.call_name}", args, kwargs)
            return [
                f"for ax in {_axes_sequence_code(tool._document, operation.axes)}:",
                f"    {call}",
            ]
        case MethodCallPolicy.AXES_POSITIONAL:
            axes_code = _axes_code(
                tool._document, operation.axes, for_plot_slices=False
            )
            args, kwargs = _code_args_kwargs(tool, operation, spec)
            args_text = _code_args((_RawCode(axes_code), *args))
            kwargs_text = _code_kwargs(kwargs)
            parts = [part for part in (args_text, kwargs_text) if part]
            return [f"eplt.{spec.call_name}({', '.join(parts)})"]
        case MethodCallPolicy.AX_KEYWORD:
            axes_code = _axes_code(
                tool._document, operation.axes, for_plot_slices=False
            )
            args, kwargs = _code_args_kwargs(tool, operation, spec)
            kwargs["ax"] = _RawCode(axes_code)
            return [f"eplt.{spec.call_name}({_call_parts(args, kwargs)})"]
        case MethodCallPolicy.EACH_AXIS_AX_KEYWORD:
            axis_code = _single_axis_code(tool, operation)
            if axis_code is not None:
                args, kwargs = _code_args_kwargs(
                    tool, operation, spec, axis_code=axis_code
                )
                kwargs["ax"] = _RawCode(axis_code)
                return [f"eplt.{spec.call_name}({_call_parts(args, kwargs)})"]
            args, kwargs = _code_args_kwargs(tool, operation, spec, axis_code="ax")
            kwargs["ax"] = _RawCode("ax")
            call = f"eplt.{spec.call_name}({_call_parts(args, kwargs)})"
            return [
                f"for ax in {_axes_sequence_code(tool._document, operation.axes)}:",
                f"    {call}",
            ]
        case MethodCallPolicy.BOUND_FIGURE:
            args, kwargs = _code_args_kwargs(tool, operation, spec)
            return [_call_code(f"fig.{spec.call_name}", args, kwargs)]
        case MethodCallPolicy.FIG_KEYWORD:
            args, kwargs = _code_args_kwargs(tool, operation, spec)
            kwargs["fig"] = _RawCode("fig")
            return [f"eplt.{spec.call_name}({_call_parts(args, kwargs)})"]
        case MethodCallPolicy.PLAIN_CALL:
            args, kwargs = _code_args_kwargs(tool, operation, spec)
            return [f"eplt.{spec.call_name}({_call_parts(args, kwargs)})"]


def _single_axis_code(
    tool: FigureComposerTool, operation: FigureOperationState
) -> str | None:
    if operation.method_transform == "custom":
        return None
    if operation.axes.expression:
        return None
    setup = tool._document.recipe.setup
    if setup.layout_mode == "gridspec":
        if len(_gridspec_valid_axes_ids(setup, operation.axes.axes_ids)) != 1:
            return None
    elif len(operation.axes.valid_axes(setup)) != 1:
        return None
    return _axes_code(tool._document, operation.axes, for_plot_slices=False)


def _call_parts(args: Sequence[typing.Any], kwargs: dict[str, typing.Any]) -> str:
    return ", ".join(part for part in (_code_args(args), _code_kwargs(kwargs)) if part)


def _method_requires_transform_import(operation: FigureOperationState) -> bool:
    spec = _method_spec(operation)
    return _method_has_transform_control(spec) and (
        operation.method_transform == "blend"
        or (operation.method_transform == "custom" and operation.trusted)
    )


def _required_imports(
    _tool: FigureComposerTool, operation: FigureOperationState
) -> Sequence[str]:
    imports: list[str] = []
    spec = _method_spec(operation)
    if spec.family == FigureMethodFamily.ERLAB:
        imports.append("import erlab.plotting as eplt")
    if _method_requires_transform_import(operation):
        imports.append("import matplotlib.transforms as mtransforms")
    return tuple(imports)
