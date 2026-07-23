"""Register the plot-slices operation with Figure Composer."""

from __future__ import annotations

import typing

from erlab.interactive._figurecomposer._line_colormap import (
    effective_line_color_coord,
    line_colormap_active,
)
from erlab.interactive._figurecomposer._line_style import line_kw_text
from erlab.interactive._figurecomposer._line_transform import line_transform_active
from erlab.interactive._figurecomposer._model._state import (
    FigureOperationKind,
    FigureOperationState,
)
from erlab.interactive._figurecomposer._norms import (
    _MATPLOTLIB_NORM_NAMES,
    _effective_norm_name,
    _use_powernorm_plot_kwargs,
)
from erlab.interactive._figurecomposer._operations._base import (
    AddStepActionSpec,
    OperationSpec,
    _always_render_cache_safe,
)
from erlab.interactive._figurecomposer._operations._plot_slices._codegen import (
    _panel_norm_uses_matplotlib_colors,
    _plot_slices_code_lines,
)
from erlab.interactive._figurecomposer._operations._plot_slices._editor import (
    _build_plot_slices_editor,
    _build_source_editor,
)
from erlab.interactive._figurecomposer._operations._plot_slices._model import (
    _PLOT_SLICES_PANEL_LINE,
    _PLOT_SLICES_PANEL_MIXED,
    _effective_extra_kwargs,
    _effective_slice_values,
    _normalized_selection_operation,
    _plot_slices_batch_panel_kind,
    _plot_slices_line_colormap_active,
    _plot_slices_selection_sources,
    _plot_slices_shape,
    _plot_slices_uses_transformed_line_maps,
)
from erlab.interactive._figurecomposer._operations._plot_slices._render import (
    _render_plot_slices,
)
from erlab.interactive._figurecomposer._ui._operation_editor import StepSection

if typing.TYPE_CHECKING:
    from erlab.interactive._figurecomposer._model._document import FigureRecipeContext
    from erlab.interactive._figurecomposer._tool import FigureComposerTool
    from erlab.interactive._figurecomposer._ui._operation_editor import (
        FigureOperationEditor,
    )

_SECTION_TOOLTIPS = {
    "selection": "Choose dimension, values, and extraction options.",
    "view": "Set orientation, axis limits, labels, and annotation behavior.",
    "colors": "Set image color scaling or line styling for this plot_slices step.",
    "transform": "Normalize, scale, and offset 1D line slices before plotting.",
    "advanced": "Pass advanced keyword arguments to plot_slices.",
}


def _create_plot_slices_operation(tool: FigureComposerTool) -> FigureOperationState:
    source_names = tool._document.source_names()
    first_source = (
        source_names[0] if source_names else tool._document.recipe.primary_source
    )
    return FigureOperationState.plot_slices(
        label="plot_slices",
        sources=(first_source,),
        axes=tool._selected_axes_state(),
    )


def _display_text(tool: FigureComposerTool, operation: FigureOperationState) -> str:
    operation = _normalized_selection_operation(tool._document, operation)
    source_text = ", ".join(
        tool._source_display_names(_plot_slices_selection_sources(operation))
    )
    if not source_text:
        source_text = "missing source"
    shape = _plot_slices_shape(tool._document, operation)
    plot_kind = "Line slices" if shape.plot_ndim == 1 else "Image slices"
    slice_values = _effective_slice_values(tool._document, operation)
    if operation.slice_dim and slice_values:
        selection_text = f"{operation.slice_dim} = {len(slice_values)} values"
    else:
        selection_text = "current selection"
    return f"{plot_kind}: {source_text}, {selection_text}"


def _tooltip(tool: FigureComposerTool, operation: FigureOperationState) -> str:
    return (
        "Runs erlab.plotting.plot_slices.\n"
        f"Targets: {tool._axes_target_text(operation.axes)}"
    )


def _has_invalid_target(
    context: FigureRecipeContext, operation: FigureOperationState
) -> bool:
    return context.axes_selection_has_invalid_target(operation.axes)


def _editor_sections(
    editor: FigureOperationEditor, operation: FigureOperationState
) -> tuple[StepSection, ...]:
    return tuple(
        StepSection(key, title, page, _SECTION_TOOLTIPS[key])
        for key, title, page in _build_plot_slices_editor(editor, operation)
    )


def _section_summary(
    tool: FigureComposerTool, key: str, operation: FigureOperationState
) -> str:
    operation = _normalized_selection_operation(tool._document, operation)
    match key:
        case "sources":
            return (
                ", ".join(
                    tool._source_display_names(
                        _plot_slices_selection_sources(operation)
                    )
                )
                or "none"
            )
        case "axes":
            return tool._axes_target_text(operation.axes)
        case "selection":
            slice_values = _effective_slice_values(tool._document, operation)
            if operation.slice_dim and slice_values:
                return f"{operation.slice_dim}, {len(slice_values)}"
            if operation.slice_kwargs:
                return "additional"
            return "none"
        case "view":
            labels = [
                label
                for label, value in (("x", operation.xlim), ("y", operation.ylim))
                if value is not None
            ]
            return ", ".join(labels) if labels else "auto"
        case "colors":
            panel_kind = _plot_slices_batch_panel_kind(
                tool._document, tool._editable_operations(), operation
            )
            if panel_kind == _PLOT_SLICES_PANEL_MIXED:
                return "mixed"
            if panel_kind == _PLOT_SLICES_PANEL_LINE:
                if line_colormap_active(operation):
                    coord = effective_line_color_coord(operation, operation.slice_dim)
                    return f"by {coord}" if coord else "by coordinate"
                return line_kw_text(operation, "color", "c") or "line"
            if operation.panel_styles_enabled and operation.panel_styles:
                return "per-panel"
            return operation.cmap or "default"
        case "transform":
            return "set" if line_transform_active(operation) else ""
        case "advanced":
            return "set" if _effective_extra_kwargs(tool._document, operation) else ""
    return ""


def _required_imports(
    tool: FigureComposerTool, operation: FigureOperationState
) -> tuple[str, ...]:
    imports = ["import erlab.plotting as eplt"]
    if (
        operation.enabled
        and _plot_slices_uses_transformed_line_maps(tool, operation)
        and operation.slice_dim
        and _effective_slice_values(tool._document, operation)
    ):
        imports.append("import xarray as xr")
    if (
        operation.enabled
        and _plot_slices_shape(tool._document, operation).plot_ndim != 1
        and (
            _panel_norm_uses_matplotlib_colors(tool, operation)
            or (
                not _use_powernorm_plot_kwargs(operation)
                and _effective_norm_name(operation.norm_name) in _MATPLOTLIB_NORM_NAMES
            )
        )
    ):
        imports.append("import matplotlib.colors as mcolors")
    if operation.enabled and _plot_slices_line_colormap_active(
        tool._document, operation
    ):
        imports.append("import matplotlib.colors as mcolors")
    return tuple(imports)


SPEC = OperationSpec(
    kind=FigureOperationKind.PLOT_SLICES,
    add_actions=(
        AddStepActionSpec(
            action_id=FigureOperationKind.PLOT_SLICES.value,
            text="Slice Plot",
            tooltip="Add an editable erlab.plotting.plot_slices step.",
            create_operation=_create_plot_slices_operation,
        ),
    ),
    display_text=_display_text,
    tooltip=_tooltip,
    target_text=lambda tool, operation: tool._axes_target_text(operation.axes),
    has_invalid_target=_has_invalid_target,
    uses_source_section=lambda _operation: True,
    build_source_editor=_build_source_editor,
    build_editor_sections=_editor_sections,
    section_summary=_section_summary,
    render=_render_plot_slices,
    code_lines=_plot_slices_code_lines,
    render_cache_safe=_always_render_cache_safe,
    required_imports=_required_imports,
)
