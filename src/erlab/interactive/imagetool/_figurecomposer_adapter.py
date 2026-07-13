"""Translate ImageTool plot state into Figure Composer operations."""

from __future__ import annotations

import ast
import contextlib
import typing

import numpy as np
from qtpy import QtWidgets

import erlab
from erlab.interactive._figurecomposer._exceptions import (
    PLOT_SLICES_SELECTION_ERROR_TITLE,
    FigureComposerPlotSlicesSelectionError,
)
from erlab.interactive._figurecomposer._labels import default_label_text
from erlab.interactive._figurecomposer._norms import _norm_updates_from_kwargs
from erlab.interactive._figurecomposer._state import (
    FigureDataSelectionState,
    FigureOperationState,
)

if typing.TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

    import matplotlib.colors

    from erlab.interactive.imagetool.slicer import ArraySlicer
    from erlab.interactive.imagetool.viewer import ImageSlicerArea


class _ViewBoxState(typing.Protocol):
    @property
    def state(self) -> dict[str, typing.Any]: ...


class _SlicerDataItemState(typing.Protocol):
    @property
    def normalize(self) -> bool: ...


class PlotOperationSource(typing.Protocol):
    """Read-only ImageTool plot state required by the adapter."""

    @property
    def is_image(self) -> bool: ...

    @property
    def display_axis(self) -> tuple[int, ...]: ...

    @property
    def axis_dims(self) -> tuple[str | None, str | None]: ...

    @property
    def axis_dims_uniform(self) -> tuple[str | None, str | None]: ...

    @property
    def slicer_area(self) -> ImageSlicerArea: ...

    @property
    def array_slicer(self) -> ArraySlicer: ...

    @property
    def slicer_data_items(self) -> Sequence[_SlicerDataItemState]: ...

    @property
    def vb(self) -> _ViewBoxState: ...


def build_figure_composer_operation(
    source: PlotOperationSource, *, source_name: str
) -> FigureOperationState:
    """Build an operation from the current ImageTool plot state."""
    return _PlotOperationBuilder(source).build(source_name=source_name)


def show_plot_slices_selection_error(
    parent: QtWidgets.QWidget | None, error: Exception
) -> None:
    """Show an ImageTool selection error using the Figure Composer wording."""
    QtWidgets.QMessageBox.warning(parent, PLOT_SLICES_SELECTION_ERROR_TITLE, str(error))


def _plain_value(value: typing.Any) -> typing.Any:
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, list):
        return [_plain_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_plain_value(item) for item in value)
    if isinstance(value, dict):
        return {_plain_value(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, float):
        return value
    with contextlib.suppress(TypeError, ValueError):
        return float(value)
    return value


def _indexer_state(indexer: typing.Any) -> typing.Any:
    if isinstance(indexer, slice):
        return {
            "kind": "slice",
            "start": indexer.start,
            "stop": indexer.stop,
            "step": indexer.step,
        }
    return indexer


def _multicursor_variable_key(varying: list[Hashable]) -> Hashable | None:
    if len(varying) > 1 and not (
        len(varying) == 2 and any(f"{key}_width" in varying for key in varying)
    ):
        raise ValueError(
            "Cannot plot when more than one dimension has differing values "
            f"across cursors: {sorted(map(str, varying))}"
        )
    if len(varying) == 1 and str(varying[0]).endswith("_width"):
        raise ValueError(
            "Cannot plot when all cursor positions are the same but widths differ."
        )
    for key in varying:
        if not str(key).endswith("_width"):
            return key
    return None


def _qsel_key_is_editable(key: Hashable) -> bool:
    if not isinstance(key, str):
        return False
    dim_name = key.removesuffix("_width") if key.endswith("_width") else key
    return bool(dim_name) and not dim_name.endswith("_idx")


def _norm_updates(norm_code: str) -> dict[str, typing.Any] | None:
    expression = norm_code.strip()
    if expression.startswith("|") and expression.endswith("|"):
        expression = expression[1:-1]
    try:
        call = ast.parse(expression, mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
        return None
    if not isinstance(call.func.value, ast.Name) or call.func.value.id != "eplt":
        return None
    norm_name = call.func.attr
    if norm_name not in {
        "CenteredInversePowerNorm",
        "InversePowerNorm",
        "CenteredPowerNorm",
    }:
        return None
    try:
        args = [ast.literal_eval(arg) for arg in call.args]
        kwargs: dict[str, typing.Any] = {}
        for keyword in call.keywords:
            if keyword.arg is None:
                return None
            kwargs[keyword.arg] = ast.literal_eval(keyword.value)
    except (TypeError, ValueError):
        return None
    updates = _norm_updates_from_kwargs(kwargs)
    updates["norm_name"] = norm_name
    if args:
        updates["norm_gamma"] = args[0]
    return updates


def _operation_updates(
    plot_kwargs: dict[str, typing.Any],
) -> dict[str, typing.Any] | None:
    field_names = {
        "transpose",
        "xlim",
        "ylim",
        "crop",
        "same_limits",
        "axis",
        "show_all_labels",
        "colorbar",
        "hide_colorbar_ticks",
        "annotate",
        "cmap",
        "gamma",
        "norm_name",
        "norm_gamma",
        "norm_kwargs",
        "slice_kwargs",
        "vmin",
        "vmax",
        "vcenter",
        "halfrange",
        "order",
        "cmap_order",
        "norm_order",
        "gradient",
        "gradient_kw",
        "subplot_kw",
        "annotate_kw",
        "colorbar_kw",
    }
    updates: dict[str, typing.Any] = {}
    extra_kwargs: dict[str, typing.Any] = {}
    for key, value in plot_kwargs.items():
        if key == "norm":
            if isinstance(value, str):
                parsed_norm = _norm_updates(value)
                if parsed_norm is not None:
                    updates.update(parsed_norm)
            continue
        plain_value = _plain_value(value)
        if (
            isinstance(plain_value, str)
            and plain_value.startswith("|")
            and plain_value.endswith("|")
        ):
            continue
        if key in field_names:
            if key == "gamma":
                updates["norm_name"] = "PowerNorm"
                updates["norm_gamma"] = plain_value
            else:
                updates[key] = plain_value
        else:
            extra_kwargs[key] = plain_value
    if not updates and not extra_kwargs:
        return None
    updates["extra_kwargs"] = extra_kwargs
    return updates


class _PlotOperationBuilder:
    def __init__(self, source: PlotOperationSource) -> None:
        self.source = source

    @property
    def slicer_area(self) -> ImageSlicerArea:
        return self.source.slicer_area

    @property
    def array_slicer(self) -> ArraySlicer:
        return self.source.array_slicer

    def build(self, *, source_name: str) -> FigureOperationState:
        non_display_axes = tuple(
            sorted(
                set(range(self.slicer_area.data.ndim)) - set(self.source.display_axis)
            )
        )
        has_nonuniform_axes = any(
            axis in self.array_slicer._nonuniform_axes for axis in non_display_axes
        )
        try:
            qsel_kwargs, selection_exprs, variable_dim, selected_dims = (
                self.multicursor_selection_plan(
                    data_name=source_name,
                    non_display_axes=non_display_axes,
                    has_nonuniform_non_display_axes=has_nonuniform_axes,
                )
            )
            selection_count = None
        except ValueError as exc:
            if self.source.is_image:
                raise FigureComposerPlotSlicesSelectionError(str(exc)) from exc
            qsel_kwargs = None
            selection_exprs = [""]
            variable_dim = None
            selected_dims = {self.selection_dim_name(axis) for axis in non_display_axes}
            selection_count = self.slicer_area.n_cursors

        invalid_qsel_keys = (
            tuple(key for key in qsel_kwargs if not _qsel_key_is_editable(key))
            if qsel_kwargs is not None
            else ()
        )
        if self.source.is_image and (
            selection_exprs is not None or qsel_kwargs is None or invalid_qsel_keys
        ):
            if invalid_qsel_keys:
                detail = "Unsupported qsel selection keys: " + ", ".join(
                    repr(key) for key in invalid_qsel_keys
                )
            elif selection_exprs is not None:
                detail = (
                    "Selection requires per-cursor expressions that cannot be "
                    "edited by plot_slices."
                )
            else:
                detail = "Selection did not produce qsel coordinates for plot_slices."
            raise FigureComposerPlotSlicesSelectionError(detail)

        map_selections = (
            self.map_selections(
                source_name=source_name,
                non_display_axes=non_display_axes,
                variable_dim=variable_dim,
                selection_count=selection_count,
            )
            if selection_exprs is not None
            else ()
        )
        dim_order_plot = [
            self.selection_dim_name(axis) for axis in range(self.slicer_area.data.ndim)
        ]
        for key in selected_dims:
            if key in dim_order_plot:
                dim_order_plot.remove(key)
        dim_order_plot.reverse()

        if self.source.is_image:
            if self.should_use_plot_array(qsel_kwargs):
                selections = (
                    self.map_selections(
                        source_name=source_name,
                        non_display_axes=non_display_axes,
                        variable_dim=variable_dim,
                        selection_count=1,
                    )
                    if qsel_kwargs
                    else ()
                )
                return self.plot_array_operation(
                    source_name=source_name,
                    dim_order_plot=dim_order_plot,
                    map_selections=selections,
                )
            return self.plot_slices_operation(
                source_name=source_name,
                variable_dim=variable_dim,
                dim_order_plot=dim_order_plot,
                qsel_kwargs=qsel_kwargs,
                selected_maps=selection_exprs,
                map_selections=map_selections,
            )
        return self.line_operation(
            source_name=source_name,
            variable_dim=variable_dim,
            x_dim=dim_order_plot[0],
            qsel_kwargs=qsel_kwargs,
            selected_lines=selection_exprs,
            map_selections=map_selections,
        )

    def qsel_kwargs_from_cursor_kwargs(
        self, all_qsel_kwargs: list[dict[Hashable, float]]
    ) -> tuple[dict[Hashable, float | list[float]], Hashable | None]:
        all_keys: set[Hashable] = set().union(
            *(kwargs.keys() for kwargs in all_qsel_kwargs)
        )
        result: dict[Hashable, float | list[float]] = {}
        varying: list[Hashable] = []
        for key in all_keys:
            values = (
                [kwargs.get(key, 0.0) for kwargs in all_qsel_kwargs]
                if str(key).endswith("_width")
                else [kwargs[key] for kwargs in all_qsel_kwargs]
            )
            if len(set(values)) == 1:
                result[key] = values[0]
            else:
                varying.append(key)
                result[key] = values
        variable_dim = _multicursor_variable_key(varying)
        variable_keys = (variable_dim, f"{variable_dim}_width")
        other_keys = sorted(
            (key for key in result if key not in variable_keys), key=str
        )
        ordered_keys = [key for key in variable_keys if key in result] + other_keys
        return {key: result[key] for key in ordered_keys}, variable_dim

    def uniform_qsel_kwargs_multicursor(
        self,
    ) -> tuple[dict[Hashable, float | list[float]], Hashable | None]:
        if any(
            axis in self.array_slicer._nonuniform_axes
            for axis in set(range(self.slicer_area.data.ndim))
            - set(self.source.display_axis)
        ):
            raise ValueError(
                "Cannot generate uniform qsel kwargs when indexing along "
                "non-uniform axes."
            )
        return self.qsel_kwargs_from_cursor_kwargs(
            [
                self.array_slicer.qsel_args(cursor, self.source.display_axis)
                for cursor in range(self.slicer_area.n_cursors)
            ]
        )

    def public_nonuniform_qsel_kwargs_multicursor(
        self, non_display_axes: tuple[int, ...]
    ) -> tuple[dict[Hashable, float | list[float]], Hashable | None]:
        public_data = erlab.interactive.imagetool.slicer.restore_nonuniform_dims(
            self.slicer_area._tool_source_parent_data()
        )
        all_qsel_kwargs: list[dict[Hashable, float]] = []
        for cursor in range(self.slicer_area.n_cursors):
            cursor_qsel_kwargs: dict[Hashable, float] = {}
            binned = self.array_slicer.get_binned(cursor)
            for axis in non_display_axes:
                if axis in self.array_slicer._nonuniform_axes:
                    dim_name = self.selection_dim_name(axis)
                    indexer = self.array_slicer._bin_slice(
                        cursor, axis, int_if_one=True
                    )
                    binned_dims = (dim_name,) if binned[axis] else ()
                    cursor_qsel_kwargs.update(
                        erlab.interactive.imagetool.slicer.qsel_args_from_indexers(
                            public_data, {dim_name: indexer}, binned_dims
                        )
                    )
                else:
                    display_axes = tuple(
                        index
                        for index in range(self.slicer_area.data.ndim)
                        if index != axis
                    )
                    cursor_qsel_kwargs.update(
                        self.array_slicer.qsel_args(cursor, display_axes)
                    )
            all_qsel_kwargs.append(cursor_qsel_kwargs)
        return self.qsel_kwargs_from_cursor_kwargs(all_qsel_kwargs)

    def multicursor_selection_plan(
        self,
        *,
        data_name: str,
        non_display_axes: tuple[int, ...],
        has_nonuniform_non_display_axes: bool,
    ) -> tuple[
        dict[Hashable, float | list[float]] | None,
        list[str] | None,
        Hashable | None,
        set[Hashable],
    ]:
        if has_nonuniform_non_display_axes:
            with contextlib.suppress(ValueError):
                qsel_kwargs, variable_dim = (
                    self.public_nonuniform_qsel_kwargs_multicursor(non_display_axes)
                )
                return qsel_kwargs, None, variable_dim, set(qsel_kwargs)
            varying: list[Hashable] = []
            for axis in non_display_axes:
                dim_name = self.selection_dim_name(axis)
                centers = [
                    self.array_slicer.get_index(cursor, axis)
                    for cursor in range(self.slicer_area.n_cursors)
                ]
                widths = [
                    self.array_slicer.get_bins(cursor)[axis] if binned[axis] else 0
                    for cursor in range(self.slicer_area.n_cursors)
                    for binned in (self.array_slicer.get_binned(cursor),)
                ]
                if len(set(centers)) > 1:
                    varying.append(dim_name)
                if len(set(widths)) > 1:
                    varying.append(f"{dim_name}_width")
            variable_dim = _multicursor_variable_key(varying)
            expressions = [
                self.selection_expr_for_cursor(data_name, cursor, non_display_axes)
                for cursor in range(self.slicer_area.n_cursors)
            ]
            if variable_dim is None:
                expressions = expressions[:1]
            return (
                None,
                expressions,
                variable_dim,
                {self.selection_dim_name(axis) for axis in non_display_axes},
            )
        qsel_kwargs, variable_dim = self.uniform_qsel_kwargs_multicursor()
        return qsel_kwargs, None, variable_dim, set(qsel_kwargs)

    def selection_dim_name(self, axis: int) -> Hashable:
        dim_name: Hashable = self.slicer_area.data.dims[axis]
        if (
            axis in self.array_slicer._nonuniform_axes
            and isinstance(dim_name, str)
            and dim_name.endswith("_idx")
        ):
            return dim_name.removesuffix("_idx")
        return dim_name

    def selection_expr_for_cursor(
        self, data_name: str, cursor: int, non_display_axes: tuple[int, ...]
    ) -> str:
        isel_kwargs: dict[Hashable, slice | int] = {}
        qsel_kwargs: dict[Hashable, float] = {}
        mean_dims: list[Hashable] = []
        binned = self.array_slicer.get_binned(cursor)
        uses_nonuniform_axes = False
        for axis in non_display_axes:
            if axis in self.array_slicer._nonuniform_axes:
                uses_nonuniform_axes = True
                dim_name = self.selection_dim_name(axis)
                isel_kwargs[dim_name] = self.array_slicer._bin_slice(
                    cursor, axis, int_if_one=True
                )
                if binned[axis]:
                    mean_dims.append(dim_name)
            else:
                display_axes = tuple(
                    index
                    for index in range(self.slicer_area.data.ndim)
                    if index != axis
                )
                qsel_kwargs.update(self.array_slicer.qsel_args(cursor, display_axes))
        selected = (
            f"erlab.interactive.imagetool.slicer.restore_nonuniform_dims({data_name})"
            if uses_nonuniform_axes
            else data_name
        )
        if isel_kwargs:
            selected += f".isel({erlab.interactive.utils.format_kwargs(isel_kwargs)})"
        if qsel_kwargs:
            selected += f".qsel({erlab.interactive.utils.format_kwargs(qsel_kwargs)})"
        if mean_dims:
            mean_arg = erlab.interactive.utils._parse_single_arg(
                mean_dims[0] if len(mean_dims) == 1 else tuple(mean_dims)
            )
            selected += f".qsel.mean({mean_arg})"
        return selected

    def map_selections(
        self,
        *,
        source_name: str,
        non_display_axes: tuple[int, ...],
        variable_dim: Hashable | None,
        selection_count: int | None = None,
    ) -> tuple[FigureDataSelectionState, ...]:
        if selection_count is None:
            selection_count = (
                self.slicer_area.n_cursors if variable_dim is not None else 1
            )
        selections: list[FigureDataSelectionState] = []
        for cursor in range(selection_count):
            isel_kwargs: dict[str, typing.Any] = {}
            qsel_kwargs: dict[str, typing.Any] = {}
            mean_dims: list[str] = []
            binned = self.array_slicer.get_binned(cursor)
            for axis in non_display_axes:
                if axis in self.array_slicer._nonuniform_axes:
                    dim_name = str(self.selection_dim_name(axis))
                    isel_kwargs[dim_name] = _indexer_state(
                        self.array_slicer._bin_slice(cursor, axis, int_if_one=True)
                    )
                    if binned[axis]:
                        mean_dims.append(dim_name)
                else:
                    display_axes = tuple(
                        index
                        for index in range(self.slicer_area.data.ndim)
                        if index != axis
                    )
                    qsel_kwargs.update(
                        {
                            str(key): _plain_value(value)
                            for key, value in self.array_slicer.qsel_args(
                                cursor, display_axes
                            ).items()
                        }
                    )
            selections.append(
                FigureDataSelectionState(
                    source=source_name,
                    isel=isel_kwargs,
                    qsel=qsel_kwargs,
                    mean_dims=tuple(mean_dims),
                )
            )
        return tuple(selections)

    @staticmethod
    def should_use_plot_array(
        qsel_kwargs: dict[Hashable, float | list[float]] | None,
    ) -> bool:
        if qsel_kwargs is None:
            return True
        return not any(
            isinstance(value := _plain_value(item), list) and len(value) > 1
            for item in qsel_kwargs.values()
        )

    def effective_crop_indexers(self) -> dict[Hashable, slice]:
        limits = {
            key: list(value) for key, value in self.slicer_area.manual_limits.items()
        }
        for dim, auto, value_range in zip(
            self.source.axis_dims_uniform,
            self.source.vb.state["autoRange"],
            self.source.vb.state["viewRange"],
            strict=True,
        ):
            if dim is None:
                continue
            if auto:
                limits.pop(dim, None)
                continue
            try:
                axis = self.slicer_area.data.dims.index(dim)
            except ValueError:
                continue
            data_min, data_max = sorted(self.array_slicer.lims_uniform[axis])
            view_min, view_max = sorted(float(value) for value in value_range)
            if view_min > data_min or view_max < data_max:
                limits[dim] = [view_min, view_max]
            else:
                limits.pop(dim, None)

        slices: dict[Hashable, slice] = {}
        for dim, values in limits.items():
            if dim not in self.slicer_area.data.dims:
                continue
            axis = self.slicer_area.data.dims.index(dim)
            digits = self.array_slicer.get_significant(axis, uniform=True)
            bounds = (
                float(np.round(values[0], digits)),
                float(np.round(values[1], digits)),
            )
            start, stop = min(bounds), max(bounds)
            if self.array_slicer.incs_uniform[axis] < 0:
                start, stop = stop, start
            if digits == 0:
                start, stop = int(start), int(stop)
            slices[dim] = slice(start, stop)
        return slices

    def plot_slices_kwargs(
        self, dim_order_plot: list[Hashable]
    ) -> dict[str, typing.Any]:
        plot_kwargs: dict[str, typing.Any] = {}
        plot_dims = list(dim_order_plot)
        if plot_dims[0] != self.source.axis_dims[0]:
            plot_kwargs["transpose"] = True
            plot_dims.reverse()
        for dim, value in self.effective_crop_indexers().items():
            if dim == plot_dims[0]:
                plot_kwargs["xlim"] = (float(value.start), float(value.stop))
            elif dim == plot_dims[1]:
                plot_kwargs["ylim"] = (float(value.start), float(value.stop))
        if self.source.vb.state["aspectLocked"]:
            plot_kwargs["axis"] = "image"
        plot_kwargs.update(self.colormap_kwargs())
        return plot_kwargs

    def colormap_kwargs(self) -> dict[str, typing.Any]:
        properties = self.slicer_area.colormap_properties.copy()
        plot_kwargs: dict[str, typing.Any] = {"cmap": properties["cmap"]}
        levels: tuple[float, float] | None = None
        if properties["levels_locked"]:
            plot_kwargs["same_limits"] = True
            levels = properties.get("levels")
        if properties["reverse"] and isinstance(plot_kwargs["cmap"], str):
            plot_kwargs["cmap"] = f"{plot_kwargs['cmap']}_r"
        norm_kwargs: dict[str, typing.Any] = {}
        if levels is not None:
            if properties["zero_centered"]:
                vmin, vmax = levels
                norm_kwargs["vcenter"] = 0.5 * (vmin + vmax)
                norm_kwargs["halfrange"] = (vmax - vmin) / 2
            else:
                norm_kwargs["vmin"], norm_kwargs["vmax"] = levels
        norm_class: type[matplotlib.colors.Normalize] | None
        if properties["high_contrast"]:
            norm_class = (
                erlab.plotting.CenteredInversePowerNorm
                if properties["zero_centered"]
                else erlab.plotting.InversePowerNorm
            )
        elif properties["zero_centered"]:
            norm_class = erlab.plotting.CenteredPowerNorm
        else:
            norm_class = None
        if norm_class is None:
            plot_kwargs["gamma"] = properties["gamma"]
            plot_kwargs.update(norm_kwargs)
        else:
            norm_code = erlab.interactive.utils.generate_code(
                norm_class,
                args=[properties["gamma"]],
                kwargs=norm_kwargs,
                module="eplt",
            )
            plot_kwargs["norm"] = f"|{norm_code}|"
        return plot_kwargs

    def line_style_updates(self) -> dict[str, typing.Any]:
        colors = tuple(
            self.slicer_area.cursor_colors[index].name()
            for index in range(self.slicer_area.n_cursors)
        )
        defaults = erlab.interactive._options.schema.ColorOptions.model_fields[
            "cursors"
        ].default
        return {"line_colors": colors} if any(c not in defaults for c in colors) else {}

    def line_limit_updates(self, x_dim: Hashable) -> dict[str, typing.Any]:
        crop_indexers = self.effective_crop_indexers()
        if x_dim not in crop_indexers:
            return {}
        selected = crop_indexers[x_dim]
        return {"xlim": (float(selected.start), float(selected.stop))}

    def plot_slices_operation(
        self,
        *,
        source_name: str,
        variable_dim: Hashable | None,
        dim_order_plot: list[Hashable],
        qsel_kwargs: dict[Hashable, float | list[float]] | None = None,
        selected_maps: list[str] | None = None,
        map_selections: tuple[FigureDataSelectionState, ...] = (),
    ) -> FigureOperationState:
        updates = _operation_updates(self.plot_slices_kwargs(dim_order_plot)) or {}
        line_plot = len(dim_order_plot) == 1
        label_updates: dict[str, typing.Any] = {}
        if selected_maps is not None:
            if line_plot:
                label_updates["line_label_text"] = default_label_text(
                    str(variable_dim) if variable_dim is not None else None,
                    fallback="slice {number}",
                    source_count=max(len(map_selections), 1),
                )
            return FigureOperationState.plot_slices(
                label="plot_slices",
                sources=(source_name,),
                map_selections=map_selections,
            ).model_copy(update={**updates, **label_updates})
        if qsel_kwargs is None or any(not isinstance(key, str) for key in qsel_kwargs):
            if qsel_kwargs is not None:
                count = self.slicer_area.n_cursors if variable_dim is not None else 1
                map_selections = tuple(
                    FigureDataSelectionState(
                        source=source_name,
                        qsel={
                            str(key): _plain_value(
                                value[cursor] if isinstance(value, list) else value
                            )
                            for key, value in qsel_kwargs.items()
                        },
                    )
                    for cursor in range(count)
                )
            if line_plot:
                label_updates["line_label_text"] = default_label_text(
                    str(variable_dim) if variable_dim is not None else None,
                    fallback="slice {number}",
                    source_count=max(len(map_selections), 1),
                )
            return FigureOperationState.plot_slices(
                label="plot_slices",
                sources=(source_name,),
                map_selections=map_selections,
            ).model_copy(update={**updates, **label_updates})

        slice_dim = str(variable_dim) if variable_dim is not None else None
        slice_values: tuple[float, ...] = ()
        slice_width = None
        slice_kwargs: dict[str, typing.Any] = {}
        for key, value in qsel_kwargs.items():
            key_text = str(key)
            plain_value = _plain_value(value)
            if slice_dim == key_text and isinstance(plain_value, list):
                slice_values = tuple(float(item) for item in plain_value)
            elif slice_dim is not None and key_text == f"{slice_dim}_width":
                if isinstance(plain_value, list):
                    widths = {float(item) for item in plain_value}
                    if len(widths) == 1:
                        slice_width = widths.pop()
                    else:
                        slice_kwargs[key_text] = plain_value
                else:
                    slice_width = float(plain_value)
            else:
                slice_kwargs[key_text] = plain_value
        if slice_dim is None:
            candidates: list[tuple[str, tuple[float, ...]]] = []
            for key, value in slice_kwargs.items():
                if key.endswith("_width"):
                    continue
                try:
                    values = value if isinstance(value, list) else [value]
                    candidates.append((key, tuple(float(item) for item in values)))
                except (TypeError, ValueError):
                    continue
            if len(candidates) == 1:
                slice_dim, slice_values = candidates[0]
                slice_kwargs.pop(slice_dim)
                width_key = f"{slice_dim}_width"
                width_value = slice_kwargs.get(width_key)
                if width_value is not None:
                    try:
                        widths = {
                            float(item)
                            for item in (
                                width_value
                                if isinstance(width_value, list)
                                else [width_value]
                            )
                        }
                    except (TypeError, ValueError):
                        pass
                    else:
                        if len(widths) == 1:
                            slice_width = widths.pop()
                            slice_kwargs.pop(width_key)
        extra_kwargs = dict(updates.pop("extra_kwargs", {}))
        if line_plot:
            label_updates["line_label_text"] = default_label_text(
                slice_dim, slice_values, fallback="slice {number}"
            )
        return FigureOperationState.plot_slices(
            label="plot_slices",
            sources=(source_name,),
            slice_dim=slice_dim,
            slice_values=slice_values,
        ).model_copy(
            update={
                **updates,
                "slice_width": slice_width,
                "slice_kwargs": slice_kwargs,
                "extra_kwargs": extra_kwargs,
                **label_updates,
            }
        )

    def plot_array_operation(
        self,
        *,
        source_name: str,
        dim_order_plot: list[Hashable],
        map_selections: tuple[FigureDataSelectionState, ...] = (),
    ) -> FigureOperationState:
        updates = _operation_updates(self.plot_slices_kwargs(dim_order_plot)) or {}
        extra_kwargs = dict(updates.pop("extra_kwargs", {}))
        if updates.pop("axis", None) == "image":
            extra_kwargs.setdefault("aspect", "equal")
        for key in (
            "same_limits",
            "show_all_labels",
            "annotate",
            "order",
            "cmap_order",
            "norm_order",
            "subplot_kw",
            "annotate_kw",
        ):
            updates.pop(key, None)
        return FigureOperationState.plot_array(
            label="plot_array",
            source=source_name,
            map_selections=map_selections,
        ).model_copy(update={**updates, "extra_kwargs": extra_kwargs})

    def line_operation(
        self,
        *,
        source_name: str,
        variable_dim: Hashable | None,
        x_dim: Hashable,
        qsel_kwargs: dict[Hashable, float | list[float]] | None = None,
        selected_lines: list[str] | None = None,
        map_selections: tuple[FigureDataSelectionState, ...] = (),
    ) -> FigureOperationState:
        normalize = (
            "mean"
            if self.source.slicer_data_items[self.slicer_area.current_cursor].normalize
            else "none"
        )
        style_updates = {
            "line_normalize": normalize,
            **self.line_style_updates(),
            **self.line_limit_updates(x_dim),
        }
        if selected_lines is not None:
            return FigureOperationState.line(
                label="line", source=source_name
            ).model_copy(
                update={
                    "line_x": str(x_dim),
                    "map_selections": map_selections,
                    "line_label_text": default_label_text(
                        None,
                        fallback="profile {number}",
                        source_count=max(len(map_selections), 1),
                    ),
                    **style_updates,
                }
            )
        if qsel_kwargs is None:
            line_selection: dict[str, typing.Any] = {}
            line_iter_dim = None
        elif any(not isinstance(key, str) for key in qsel_kwargs):
            count = self.slicer_area.n_cursors if variable_dim is not None else 1
            selections = tuple(
                FigureDataSelectionState(
                    source=source_name,
                    qsel={
                        str(key): _plain_value(
                            value[cursor] if isinstance(value, list) else value
                        )
                        for key, value in qsel_kwargs.items()
                    },
                )
                for cursor in range(count)
            )
            return FigureOperationState.line(
                label="line", source=source_name
            ).model_copy(
                update={
                    "line_x": str(x_dim),
                    "map_selections": selections,
                    "line_label_text": default_label_text(
                        str(variable_dim) if variable_dim is not None else None,
                        fallback="profile {number}",
                        source_count=max(len(selections), 1),
                    ),
                    **style_updates,
                }
            )
        else:
            line_selection = {
                str(key): _plain_value(value) for key, value in qsel_kwargs.items()
            }
            line_iter_dim = str(variable_dim) if variable_dim is not None else None
        if line_iter_dim is None:
            label_text = default_label_text(None, fallback="profile {number}")
        else:
            label_values = line_selection.get(line_iter_dim, ())
            if not isinstance(label_values, list | tuple):
                label_values = (label_values,)
            label_text = default_label_text(
                line_iter_dim, label_values, fallback="profile {number}"
            )
        return FigureOperationState.line(label="line", source=source_name).model_copy(
            update={
                "line_x": str(x_dim),
                "line_selection": dict(line_selection),
                "line_iter_dim": line_iter_dim,
                "line_label_text": label_text,
                **style_updates,
            }
        )
