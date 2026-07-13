"""Brillouin-zone overlay operation editor, renderer, and code generation."""

from __future__ import annotations

import typing

import numpy as np
from qtpy import QtCore, QtWidgets

import erlab
import erlab.plotting as eplt
from erlab.interactive._figurecomposer._code import _axes_code, _axes_sequence_code
from erlab.interactive._figurecomposer._gridspec import _gridspec_valid_axes_ids
from erlab.interactive._figurecomposer._line_style import (
    LINE_STYLE_DEFAULT_LABEL,
    LINE_STYLE_OPTIONS,
    color_kw_value_from_text,
    line_kw_float,
    line_kw_style_value,
    line_kw_text,
    optional_positive_spinbox,
    optional_positive_spinbox_value,
    update_current_line_kw,
)
from erlab.interactive._figurecomposer._operations._base import (
    AddStepActionSpec,
    OperationSpec,
    StepSection,
    _empty_source_editor,
    _uses_no_source_section,
)
from erlab.interactive._figurecomposer._rendering import (
    _axes_from_selection,
    _iter_axes,
)
from erlab.interactive._figurecomposer._state import (
    FigureOperationKind,
    FigureOperationState,
)
from erlab.interactive._figurecomposer._text import (
    _code_kwargs,
    _dict_from_text,
    _format_dict,
    _input_error,
    _literal_from_text,
    _RawCode,
)
from erlab.interactive._figurecomposer._widgets import _ColorLineEditWidget

if typing.TYPE_CHECKING:
    from collections.abc import Callable

    from erlab.interactive._figurecomposer._document import FigureRecipeContext
    from erlab.interactive._figurecomposer._tool import FigureComposerTool


_CENTERING_TYPES = ("P", "A", "B", "C", "F", "I", "R")
_MODE_LABELS = {
    "in_plane": "In-plane",
    "out_of_plane": "Out-of-plane",
}
_SECTION_TOOLTIPS = {
    "slice": "Choose the Brillouin-zone slice geometry and plotted bounds.",
    "lattice": "Choose conventional-cell lattice parameters and centering.",
    "style": "Set line styling and optional vertex or midpoint markers.",
}


def _mode_text(mode: str) -> str:
    return _MODE_LABELS.get(mode, _MODE_LABELS["in_plane"])


def _mode_from_text(text: str) -> typing.Literal["in_plane", "out_of_plane"]:
    for mode, label in _MODE_LABELS.items():
        if text == label:
            return typing.cast("typing.Literal['in_plane', 'out_of_plane']", mode)
    return "in_plane"


def _format_bounds(bounds: tuple[float, float, float, float] | None) -> str:
    if bounds is None:
        return ""
    return ", ".join(f"{value:g}" for value in bounds)


def _bounds_from_text(text: str) -> tuple[float, float, float, float] | None:
    stripped = text.strip()
    if not stripped:
        return None
    value = _literal_from_text(
        stripped,
        message=(
            "Enter four comma-separated numbers, such as -1, 1, -1, 1, "
            "or leave blank to infer from axes."
        ),
    )
    if isinstance(value, str | bytes) or not isinstance(value, list | tuple):
        raise _input_error("Enter four comma-separated numbers.")
    if len(value) != 4:
        raise _input_error("Enter four comma-separated numbers.")
    try:
        return (
            float(value[0]),
            float(value[1]),
            float(value[2]),
            float(value[3]),
        )
    except (TypeError, ValueError) as exc:
        raise _input_error("Enter four comma-separated numbers.") from exc


def _bz_avec(operation: FigureOperationState) -> np.ndarray:
    avec = erlab.lattice.abc2avec(
        operation.bz_a,
        operation.bz_b,
        operation.bz_c,
        operation.bz_alpha,
        operation.bz_beta,
        operation.bz_gamma,
    )
    if operation.bz_centering_type != "P":
        avec = erlab.lattice.to_primitive(
            avec, centering_type=operation.bz_centering_type
        )
    return avec


def _bz_bvec(operation: FigureOperationState) -> np.ndarray:
    return erlab.lattice.to_reciprocal(_bz_avec(operation))


def _render_bz_overlay(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    axs: typing.Any,
) -> None:
    axes = _iter_axes(
        _axes_from_selection(tool, operation.axes, axs, for_plot_slices=False)
    )
    if not axes:
        return

    bvec = _bz_bvec(operation)
    common_kwargs = {
        "angle": operation.bz_angle,
        "bounds": operation.bz_bounds,
        "vertices": operation.bz_vertices,
        "midpoints": operation.bz_midpoints,
        "vertex_kwargs": dict(operation.bz_vertex_kw) or None,
        "midpoint_kwargs": dict(operation.bz_midpoint_kw) or None,
        **operation.line_kw,
    }
    if operation.bz_mode == "out_of_plane":
        for axis in axes:
            eplt.plot_out_of_plane_bz(
                bvec,
                k_parallel=operation.bz_k_parallel,
                ax=axis,
                **common_kwargs,
            )
        return

    kz = operation.bz_kz_pi_over_c * np.pi / operation.bz_c
    for axis in axes:
        eplt.plot_in_plane_bz(
            bvec,
            kz=kz,
            ax=axis,
            **common_kwargs,
        )


def _lattice_code_lines(operation: FigureOperationState) -> list[str]:
    args = (
        operation.bz_a,
        operation.bz_b,
        operation.bz_c,
        operation.bz_alpha,
        operation.bz_beta,
        operation.bz_gamma,
    )
    lines = [
        f"avec = erlab.lattice.abc2avec({', '.join(f'{value!r}' for value in args)})"
    ]
    if operation.bz_centering_type != "P":
        lines.append(
            "avec = erlab.lattice.to_primitive("
            f'avec, centering_type="{operation.bz_centering_type}")'
        )
    lines.append("bvec = erlab.lattice.to_reciprocal(avec)")
    return lines


def _plot_kwargs(
    operation: FigureOperationState,
    *,
    axis_code: str,
) -> dict[str, typing.Any]:
    kwargs: dict[str, typing.Any] = {
        "angle": operation.bz_angle,
        "bounds": operation.bz_bounds,
        "ax": _RawCode(axis_code),
    }
    if operation.bz_vertices:
        kwargs["vertices"] = True
    if operation.bz_midpoints:
        kwargs["midpoints"] = True
    if operation.bz_vertex_kw:
        kwargs["vertex_kwargs"] = dict(operation.bz_vertex_kw)
    if operation.bz_midpoint_kw:
        kwargs["midpoint_kwargs"] = dict(operation.bz_midpoint_kw)
    kwargs.update(operation.line_kw)
    return kwargs


def _plot_call_code(
    operation: FigureOperationState,
    *,
    axis_code: str,
) -> str:
    if operation.bz_mode == "out_of_plane":
        kwargs = {
            "k_parallel": operation.bz_k_parallel,
            **_plot_kwargs(operation, axis_code=axis_code),
        }
        return f"eplt.plot_out_of_plane_bz(bvec, {_code_kwargs(kwargs)})"

    kwargs = {"kz": _RawCode("kz"), **_plot_kwargs(operation, axis_code=axis_code)}
    return f"eplt.plot_in_plane_bz(bvec, {_code_kwargs(kwargs)})"


def _single_axis_code(
    tool: FigureComposerTool, operation: FigureOperationState
) -> str | None:
    if operation.axes.expression:
        return None
    setup = tool._document.recipe.setup
    if setup.layout_mode == "gridspec":
        if len(_gridspec_valid_axes_ids(setup, operation.axes.axes_ids)) != 1:
            return None
    elif len(operation.axes.valid_axes(setup)) != 1:
        return None
    return _axes_code(tool._document, operation.axes, for_plot_slices=False)


def _bz_code_lines(
    tool: FigureComposerTool, operation: FigureOperationState
) -> list[str]:
    lines = _lattice_code_lines(operation)
    if operation.bz_mode == "in_plane":
        lines.append(f"kz = {operation.bz_kz_pi_over_c!r} * np.pi / {operation.bz_c!r}")

    axis_code = _single_axis_code(tool, operation)
    if axis_code is not None:
        lines.append(_plot_call_code(operation, axis_code=axis_code))
        return lines

    call = _plot_call_code(operation, axis_code="ax")
    lines.extend(
        (
            f"for ax in {_axes_sequence_code(tool._document, operation.axes)}:",
            f"    {call}",
        )
    )
    return lines


def _spinbox(
    value: float,
    *,
    parent: QtWidgets.QWidget,
    minimum: float = -1_000_000.0,
    maximum: float = 1_000_000.0,
    step: float = 0.1,
    decimals: int = 4,
    suffix: str = "",
) -> QtWidgets.QDoubleSpinBox:
    spinbox = QtWidgets.QDoubleSpinBox(parent)
    spinbox.setRange(minimum, maximum)
    spinbox.setDecimals(decimals)
    spinbox.setSingleStep(step)
    spinbox.setSuffix(suffix)
    spinbox.setKeyboardTracking(False)
    spinbox.setValue(float(value))
    return spinbox


def _kz_absolute(kz_pi_over_c: float, c: float) -> float:
    return kz_pi_over_c * np.pi / c


def _kz_pi_over_c(kz_absolute: float, c: float) -> float:
    return kz_absolute * c / np.pi


def _operation_kz_absolute(operation: FigureOperationState) -> float:
    return _kz_absolute(operation.bz_kz_pi_over_c, operation.bz_c)


def _set_spinbox_value(spinbox: QtWidgets.QDoubleSpinBox, value: float) -> None:
    with QtCore.QSignalBlocker(spinbox):
        spinbox.setValue(float(value))


def _current_bz_operation(
    tool: FigureComposerTool, fallback: FigureOperationState
) -> FigureOperationState:
    current = tool._current_operation()
    if current is None:
        return fallback
    return current[1]


def _update_bz_kz_pi_over_c(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    spinbox: QtWidgets.QDoubleSpinBox,
    value: float,
) -> None:
    operation = _current_bz_operation(tool, operation)
    _set_spinbox_value(spinbox, _kz_absolute(value, operation.bz_c))
    tool._update_current_operation(bz_kz_pi_over_c=value)


def _update_bz_kz_absolute(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    spinbox: QtWidgets.QDoubleSpinBox,
    value: float,
) -> None:
    operation = _current_bz_operation(tool, operation)
    _set_spinbox_value(spinbox, _kz_pi_over_c(value, operation.bz_c))
    tool._update_operations(
        lambda _index, target: target.model_copy(
            update={"bz_kz_pi_over_c": _kz_pi_over_c(value, target.bz_c)}
        )
    )


def _update_bz_c_preserve_kz_absolute(tool: FigureComposerTool, value: float) -> None:
    new_c = float(value)

    def update_c(_index: int, operation: FigureOperationState) -> FigureOperationState:
        kz_absolute = _operation_kz_absolute(operation)
        return operation.model_copy(
            update={
                "bz_c": new_c,
                "bz_kz_pi_over_c": _kz_pi_over_c(kz_absolute, new_c),
            }
        )

    tool._update_operations(update_c, rebuild_editor=True)


def _lattice_spinbox(
    operation: FigureOperationState,
    field: str,
    *,
    parent: QtWidgets.QWidget,
) -> QtWidgets.QDoubleSpinBox:
    minimum = 0.0001 if field in {"bz_a", "bz_b", "bz_c"} else 0.0
    maximum = 1_000_000.0 if field in {"bz_a", "bz_b", "bz_c"} else 180.0
    step = 0.1 if field in {"bz_a", "bz_b", "bz_c"} else 1.0
    decimals = 4 if field in {"bz_a", "bz_b", "bz_c"} else 3
    suffix = " Å" if field in {"bz_a", "bz_b", "bz_c"} else "°"
    return _spinbox(
        typing.cast("float", getattr(operation, field)),
        parent=parent,
        minimum=minimum,
        maximum=maximum,
        step=step,
        decimals=decimals,
        suffix=suffix,
    )


def _add_spinbox_row(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    layout: QtWidgets.QFormLayout,
    *,
    label: str,
    field: str,
    object_name: str,
    tooltip: str,
    parent: QtWidgets.QWidget,
    spinbox_factory: Callable[..., QtWidgets.QDoubleSpinBox] | None = None,
    suffix: str = "",
    update: Callable[[float], None] | None = None,
) -> None:
    mixed = tool._batch_is_mixed(operation, lambda target: getattr(target, field))
    if spinbox_factory is None:
        spinbox = _spinbox(float(getattr(operation, field)), parent=parent)
    else:
        spinbox = spinbox_factory(operation, field, parent=parent)
    if suffix:
        spinbox.setSuffix(suffix)
    spinbox.setObjectName(object_name)
    tool._connect_value_signal(
        spinbox,
        spinbox.valueChanged,
        float,
        update
        if update is not None
        else lambda value: tool._update_current_operation(**{field: value}),
    )
    tool._add_form_row(
        layout,
        label,
        tool._mixed_value_widget(spinbox, mixed=mixed, parent=parent),
        tooltip,
    )


def _build_kz_row(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    page: QtWidgets.QWidget,
    layout: QtWidgets.QFormLayout,
) -> None:
    kz_pi_mixed = tool._batch_is_mixed(operation, lambda target: target.bz_kz_pi_over_c)
    c_mixed = tool._batch_is_mixed(operation, lambda target: target.bz_c)
    kz_pi_spin = _spinbox(
        operation.bz_kz_pi_over_c,
        parent=page,
        step=0.1,
        decimals=4,
        suffix=" π/c",
    )
    kz_pi_spin.setObjectName("figureComposerBZKzSpin")
    kz_absolute_spin = _spinbox(
        _operation_kz_absolute(operation),
        parent=page,
        step=0.01,
        decimals=4,
        suffix=" Å⁻¹",
    )
    kz_absolute_spin.setObjectName("figureComposerBZKzAbsoluteSpin")

    tool._connect_value_signal(
        kz_pi_spin,
        kz_pi_spin.valueChanged,
        float,
        lambda value: _update_bz_kz_pi_over_c(tool, operation, kz_absolute_spin, value),
    )
    tool._connect_value_signal(
        kz_absolute_spin,
        kz_absolute_spin.valueChanged,
        float,
        lambda value: _update_bz_kz_absolute(tool, operation, kz_pi_spin, value),
    )

    row = QtWidgets.QWidget(page)
    row_layout = QtWidgets.QHBoxLayout(row)
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.setSpacing(6)
    row_layout.addWidget(
        tool._mixed_value_widget(kz_pi_spin, mixed=kz_pi_mixed, parent=row),
        1,
    )
    row_layout.addWidget(
        tool._mixed_value_widget(
            kz_absolute_spin, mixed=kz_pi_mixed or c_mixed, parent=row
        ),
        1,
    )
    tool._add_form_row(
        layout,
        "kz",
        row,
        "Fixed out-of-plane momentum for in-plane slices.",
    )


def _build_slice_editor(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    page: QtWidgets.QWidget,
    layout: QtWidgets.QFormLayout,
) -> None:
    mode_mixed = tool._batch_is_mixed(operation, lambda target: target.bz_mode)
    mode_combo = tool._combo(
        tuple(_MODE_LABELS.values()),
        None if mode_mixed else _mode_text(operation.bz_mode),
        lambda text: tool._update_current_operation(bz_mode=_mode_from_text(text)),
        parent=page,
        mixed=mode_mixed,
    )
    mode_combo.setObjectName("figureComposerBZModeCombo")
    tool._add_form_row(
        layout,
        "Mode",
        mode_combo,
        "Choose the BZ slice orientation.",
    )

    _add_spinbox_row(
        tool,
        operation,
        layout,
        label="Angle",
        field="bz_angle",
        object_name="figureComposerBZAngleSpin",
        tooltip="Rotation angle in degrees.",
        parent=page,
        suffix="°",
    )
    _build_kz_row(tool, operation, page, layout)
    _add_spinbox_row(
        tool,
        operation,
        layout,
        label="k parallel",
        field="bz_k_parallel",
        object_name="figureComposerBZKParallelSpin",
        tooltip="Fixed in-plane momentum component for out-of-plane slices.",
        parent=page,
        suffix=" Å⁻¹",
    )

    bounds_text, bounds_mixed = tool._batch_text(
        operation, lambda target: target.bz_bounds, _format_bounds
    )
    bounds_edit = tool._line_edit(bounds_text, parent=page)
    bounds_edit.setObjectName("figureComposerBZBoundsEdit")
    tool._apply_mixed_line_edit(bounds_edit, bounds_mixed)
    tool._connect_line_edit_finished(
        bounds_edit,
        lambda text: tool._update_current_operation(bz_bounds=_bounds_from_text(text)),
    )
    tool._add_form_row(
        layout,
        "Bounds",
        bounds_edit,
        "Optional bounds as xmin, xmax, ymin, ymax. Leave blank to infer from axes.",
    )


def _build_lattice_editor(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    page: QtWidgets.QWidget,
    layout: QtWidgets.QFormLayout,
) -> None:
    def update_c(value: float) -> None:
        _update_bz_c_preserve_kz_absolute(tool, value)

    for label, field, object_name in (
        ("a", "bz_a", "figureComposerBZAEdit"),
        ("b", "bz_b", "figureComposerBZBEdit"),
        ("c", "bz_c", "figureComposerBZCEdit"),
        ("α", "bz_alpha", "figureComposerBZAlphaEdit"),
        ("β", "bz_beta", "figureComposerBZBetaEdit"),
        ("γ", "bz_gamma", "figureComposerBZGammaEdit"),
    ):
        _add_spinbox_row(
            tool,
            operation,
            layout,
            label=label,
            field=field,
            object_name=object_name,
            tooltip=f"Lattice parameter {label}.",
            parent=page,
            spinbox_factory=_lattice_spinbox,
            update=update_c if field == "bz_c" else None,
        )

    centering_mixed = tool._batch_is_mixed(
        operation, lambda target: target.bz_centering_type
    )
    centering_combo = tool._combo(
        _CENTERING_TYPES,
        None if centering_mixed else operation.bz_centering_type,
        lambda text: tool._update_current_operation(bz_centering_type=text),
        parent=page,
        mixed=centering_mixed,
    )
    centering_combo.setObjectName("figureComposerBZCenteringCombo")
    tool._add_form_row(
        layout,
        "Centering",
        centering_combo,
        "Conventional-cell centering type. P leaves the cell unchanged.",
    )


def _build_style_editor(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    page: QtWidgets.QWidget,
    layout: QtWidgets.QFormLayout,
) -> None:
    color_text, color_mixed = tool._batch_text(
        operation,
        lambda target: line_kw_text(target, "color", "c") or "",
        str,
    )
    color_edit = _ColorLineEditWidget(color_text, parent=page)
    color_edit.setLineEditObjectName("figureComposerBZColorEdit")
    color_edit.setColorButtonObjectName("figureComposerBZColorButton")
    tool._apply_mixed_line_edit(color_edit.line_edit, color_mixed)
    tool._connect_value_signal(
        color_edit,
        color_edit.editingFinished,
        color_edit.text,
        lambda text: update_current_line_kw(
            tool, "color", color_kw_value_from_text(text), aliases=("c",)
        ),
        unchanged_mixed=lambda: tool._line_edit_batch_unchanged(color_edit.line_edit),
    )
    tool._add_form_row(
        layout,
        "Line color",
        color_edit,
        "Matplotlib color for BZ boundary lines.",
    )

    line_style_mixed = tool._batch_is_mixed(
        operation, lambda target: line_kw_style_value(target, "linestyle", "ls")
    )
    line_style_combo = tool._optional_name_combo(
        LINE_STYLE_OPTIONS,
        None if line_style_mixed else line_kw_style_value(operation, "linestyle", "ls"),
        LINE_STYLE_DEFAULT_LABEL,
        lambda text: update_current_line_kw(tool, "linestyle", text, aliases=("ls",)),
        parent=page,
        mixed=line_style_mixed,
    )
    line_style_combo.setObjectName("figureComposerBZLineStyleCombo")

    line_width_mixed = tool._batch_is_mixed(
        operation, lambda target: line_kw_text(target, "linewidth", "lw")
    )
    line_width_spin = optional_positive_spinbox(
        None if line_width_mixed else line_kw_float(operation, "linewidth", "lw"),
        parent=page,
    )
    line_width_spin.setObjectName("figureComposerBZLineWidthSpin")
    tool._connect_editor_signal(
        line_width_spin,
        line_width_spin.valueChanged,
        lambda value: update_current_line_kw(
            tool,
            "linewidth",
            optional_positive_spinbox_value(value),
            aliases=("lw",),
        ),
    )
    tool._add_compound_form_row(
        layout,
        "Line",
        (
            ("Style", line_style_combo, "Matplotlib linestyle for BZ boundaries."),
            (
                "Width",
                tool._mixed_value_widget(
                    line_width_spin, mixed=line_width_mixed, parent=page
                ),
                "Matplotlib linewidth for BZ boundaries.",
            ),
        ),
        "Line style controls for BZ boundaries.",
    )

    vertices_mixed = tool._batch_is_mixed(operation, lambda target: target.bz_vertices)
    vertices_check = tool._check_box(
        operation.bz_vertices,
        lambda checked: tool._update_current_operation(bz_vertices=checked),
        parent=page,
        mixed=vertices_mixed,
    )
    vertices_check.setObjectName("figureComposerBZVerticesCheck")
    vertices_check.setText("")
    midpoints_mixed = tool._batch_is_mixed(
        operation, lambda target: target.bz_midpoints
    )
    midpoints_check = tool._check_box(
        operation.bz_midpoints,
        lambda checked: tool._update_current_operation(bz_midpoints=checked),
        parent=page,
        mixed=midpoints_mixed,
    )
    midpoints_check.setObjectName("figureComposerBZMidpointsCheck")
    midpoints_check.setText("")
    tool._add_compound_form_row(
        layout,
        "Points",
        (
            ("Vertices", vertices_check, "Draw BZ vertex markers."),
            ("Midpoints", midpoints_check, "Draw BZ segment midpoint markers."),
        ),
        "Optional point overlays.",
    )

    for label, field, object_name, tooltip in (
        (
            "Vertex kwargs",
            "bz_vertex_kw",
            "figureComposerBZVertexKwEdit",
            "Keyword arguments forwarded to the vertex scatter artist.",
        ),
        (
            "Midpoint kwargs",
            "bz_midpoint_kw",
            "figureComposerBZMidpointKwEdit",
            "Keyword arguments forwarded to the midpoint scatter artist.",
        ),
    ):

        def field_getter(
            target: FigureOperationState, field: str = field
        ) -> typing.Any:
            return getattr(target, field)

        def update_field(value: str, field: str = field) -> None:
            tool._update_current_operation(**{field: _dict_from_text(value)})

        text, mixed = tool._batch_text(
            operation,
            field_getter,
            _format_dict,
        )
        edit = tool._line_edit(text, parent=page)
        edit.setObjectName(object_name)
        tool._apply_mixed_line_edit(edit, mixed)
        tool._connect_line_edit_finished(edit, update_field)
        tool._add_form_row(layout, label, edit, tooltip)


def _build_editor(
    tool: FigureComposerTool, operation: FigureOperationState
) -> list[tuple[str, str, QtWidgets.QWidget]]:
    slice_page, slice_layout = tool._new_step_form_page("figureComposerBZSlicePage")
    lattice_page, lattice_layout = tool._new_step_form_page(
        "figureComposerBZLatticePage"
    )
    style_page, style_layout = tool._new_step_form_page("figureComposerBZStylePage")
    tool.operation_editor = slice_page
    tool.operation_editor_layout = slice_layout

    _build_slice_editor(tool, operation, slice_page, slice_layout)
    _build_lattice_editor(tool, operation, lattice_page, lattice_layout)
    _build_style_editor(tool, operation, style_page, style_layout)

    return [
        ("slice", "Slice", slice_page),
        ("lattice", "Lattice", lattice_page),
        ("style", "Style", style_page),
    ]


def _editor_sections(
    tool: FigureComposerTool, operation: FigureOperationState
) -> tuple[StepSection, ...]:
    return tuple(
        StepSection(key, title, page, _SECTION_TOOLTIPS[key])
        for key, title, page in _build_editor(tool, operation)
    )


def _create_operation(tool: FigureComposerTool) -> FigureOperationState:
    from erlab.interactive._figurecomposer._seeding import _default_bz_updates

    return FigureOperationState.bz_overlay(axes=tool._selected_axes_state()).model_copy(
        update=_default_bz_updates()
    )


def _display_text(tool: FigureComposerTool, operation: FigureOperationState) -> str:
    return f"BZ Overlay: {_mode_text(operation.bz_mode)}"


def _tooltip(tool: FigureComposerTool, operation: FigureOperationState) -> str:
    return (
        "Draws a Brillouin-zone slice overlay.\n"
        f"Targets: {tool._axes_target_text(operation.axes)}"
    )


def _has_invalid_target(
    context: FigureRecipeContext, operation: FigureOperationState
) -> bool:
    return context.axes_selection_has_invalid_target(operation.axes)


def _section_summary(
    tool: FigureComposerTool, key: str, operation: FigureOperationState
) -> str:
    match key:
        case "axes":
            return tool._axes_target_text(operation.axes)
        case "slice":
            if operation.bz_mode == "out_of_plane":
                return f"OOP, k={operation.bz_k_parallel:g} Å⁻¹"
            return (
                f"IP, kz={operation.bz_kz_pi_over_c:g} π/c; "
                f"{_operation_kz_absolute(operation):g} Å⁻¹"
            )
        case "lattice":
            return (
                f"a={operation.bz_a:g} Å, b={operation.bz_b:g} Å, "
                f"c={operation.bz_c:g} Å; α={operation.bz_alpha:g}°, "
                f"β={operation.bz_beta:g}°, γ={operation.bz_gamma:g}°; "
                f"{operation.bz_centering_type}"
            )
        case "style":
            labels = []
            if operation.bz_vertices:
                labels.append("vertices")
            if operation.bz_midpoints:
                labels.append("midpoints")
            return ", ".join(labels) or line_kw_text(operation, "color", "c")
    return ""


def _required_imports(
    _tool: FigureComposerTool, operation: FigureOperationState
) -> tuple[str, ...]:
    imports = ["import erlab", "import erlab.plotting as eplt"]
    if operation.bz_mode == "in_plane":
        imports.append("import numpy as np")
    return tuple(imports)


SPEC = OperationSpec(
    kind=FigureOperationKind.BZ_OVERLAY,
    add_actions=(
        AddStepActionSpec(
            action_id=FigureOperationKind.BZ_OVERLAY.value,
            text="BZ Overlay",
            tooltip="Add a Brillouin-zone slice overlay step.",
            create_operation=_create_operation,
        ),
    ),
    display_text=_display_text,
    tooltip=_tooltip,
    target_text=lambda tool, operation: tool._axes_target_text(operation.axes),
    has_invalid_target=_has_invalid_target,
    uses_axes=lambda _operation: True,
    uses_source_section=_uses_no_source_section,
    build_source_editor=_empty_source_editor,
    build_editor_sections=_editor_sections,
    section_summary=_section_summary,
    render=lambda tool, operation, _figure, axs: _render_bz_overlay(
        tool, operation, axs
    ),
    code_lines=_bz_code_lines,
    required_imports=_required_imports,
)
