"""Photon-energy overlay operation editor, renderer, and code generation."""

from __future__ import annotations

import contextlib
import dataclasses
import typing

from erlab.interactive._figurecomposer._code import _axes_code, _axes_sequence_code
from erlab.interactive._figurecomposer._line_style import (
    LINE_STYLE_DEFAULT_LABEL,
    LINE_STYLE_OPTIONS,
    color_kw_value_from_text,
    line_kw_float,
    line_kw_style_value,
    line_kw_text,
)
from erlab.interactive._figurecomposer._model._gridspec import _gridspec_valid_axes_ids
from erlab.interactive._figurecomposer._model._sources import (
    _public_source_data,
    _valid_source_variable,
)
from erlab.interactive._figurecomposer._model._state import (
    FigureOperationKind,
    FigureOperationState,
)
from erlab.interactive._figurecomposer._operations._base import (
    AddStepActionSpec,
    OperationSpec,
    _always_render_cache_safe,
)
from erlab.interactive._figurecomposer._rendering import (
    _axes_from_selection,
    _iter_axes,
)
from erlab.interactive._figurecomposer._text import (
    _code_kwargs,
    _dict_from_text,
    _float_tuple_from_text,
    _format_dict,
    _format_tuple,
    _input_error,
    _RawCode,
)
from erlab.interactive._figurecomposer._ui._color_widgets import _ColorLineEditWidget
from erlab.interactive._figurecomposer._ui._line_style import (
    optional_positive_spinbox,
    optional_positive_spinbox_value,
    update_current_line_kw,
)
from erlab.interactive._figurecomposer._ui._operation_editor import StepSection

if typing.TYPE_CHECKING:
    import xarray as xr
    from qtpy import QtWidgets

    from erlab.interactive._figurecomposer._model._document import FigureRecipeContext
    from erlab.interactive._figurecomposer._tool import FigureComposerTool
    from erlab.interactive._figurecomposer._ui._operation_editor import (
        FigureOperationEditor,
    )


_DEFAULT_LABEL_TEMPLATE = r"$h\nu = {hv:g}$ eV"
_SECTION_TOOLTIPS = {
    "photon": "Choose photon energies and the electron binding-energy slice.",
    "style": "Set curve styling and legend labels.",
}


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:g}"


def _optional_float_from_text(text: str) -> float | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError as exc:
        raise _input_error("Enter one number, or leave blank.") from exc


def _photon_energies_from_text(text: str) -> tuple[float, ...]:
    return _float_tuple_from_text(text)


def _photon_energies_code(values: tuple[float, ...]) -> str:
    return "[" + ", ".join(f"{value:g}" for value in values) + "]"


@dataclasses.dataclass(frozen=True)
class _PhotonEnergyPlan:
    """Semantic inputs used to prepare photon-energy overlay curves."""

    source: str | None
    photon_energies: tuple[float, ...]
    binding_energy: float | None

    @classmethod
    def from_operation(cls, operation: FigureOperationState) -> _PhotonEnergyPlan:
        return cls(
            source=operation.hv_overlay_source,
            photon_energies=operation.photon_energies,
            binding_energy=operation.binding_energy,
        )


def _source_data_from_name(
    tool: FigureComposerTool, source: str | None
) -> xr.DataArray:
    if source is None:
        raise ValueError("Select a source data array for the photon-energy overlay")
    data = tool._document.source_data.get(source)
    if data is None:
        source_name = tool._source_display_name(source)
        raise ValueError(f"Source {source_name!r} is missing")
    return _public_source_data(data)


def _source_data(
    tool: FigureComposerTool, operation: FigureOperationState
) -> xr.DataArray:
    return _source_data_from_name(tool, operation.hv_overlay_source)


def _source_is_kparallel_kz(data: xr.DataArray) -> bool:
    dims = {str(dim) for dim in data.squeeze(drop=True).dims}
    return "kz" in dims and bool({"kx", "ky"} & dims)


def _photon_x_dim(data: xr.DataArray, kz_values: xr.DataArray) -> str:
    candidates: list[str] = []
    with contextlib.suppress(AttributeError, KeyError, ValueError):
        candidates.append(str(data.kspace.slit_axis))
    candidates.extend(dim for dim in ("kx", "ky") if dim not in candidates)
    for dim in candidates:
        if dim in kz_values.dims and dim in kz_values.coords:
            return dim
    raise ValueError(
        "Photon-energy overlays require a kx-kz or ky-kz source data array"
    )


def _kz_values(
    data: xr.DataArray, operation: FigureOperationState
) -> tuple[xr.DataArray, str]:
    return _kz_values_from_plan(data, _PhotonEnergyPlan.from_operation(operation))


def _kz_values_from_plan(
    data: xr.DataArray, plan: _PhotonEnergyPlan
) -> tuple[xr.DataArray, str]:
    if not plan.photon_energies:
        raise ValueError("Enter at least one photon energy")
    if not _source_is_kparallel_kz(data):
        raise ValueError(
            "Photon-energy overlays require a kx-kz or ky-kz source data array"
        )
    kz_values = data.kspace.hv_to_kz(list(plan.photon_energies))
    if plan.binding_energy is not None:
        kz_values = kz_values.qsel(eV=plan.binding_energy)
    x_dim = _photon_x_dim(data, kz_values)
    if "eV" in kz_values.dims:
        raise ValueError(
            "Select a binding energy before plotting photon-energy overlays"
        )
    extra_dims = {str(dim) for dim in kz_values.dims} - {"hv", x_dim}
    if "hv" not in kz_values.dims or extra_dims:
        raise ValueError(
            "Photon-energy overlay curves must reduce to hv and one momentum dimension"
        )
    return kz_values, x_dim


def _label_text(operation: FigureOperationState, kz: xr.DataArray) -> str:
    return operation.label_template.format(hv=kz.hv)


def _line_kwargs(
    operation: FigureOperationState, kz: xr.DataArray
) -> dict[str, typing.Any]:
    kwargs = {key: value for key, value in operation.line_kw.items() if key != "label"}
    return {"label": _label_text(operation, kz), **kwargs}


def _render_photon_energy_overlay(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    axs: typing.Any,
) -> None:
    axes = _iter_axes(
        _axes_from_selection(tool, operation.axes, axs, for_plot_slices=False)
    )
    if not axes:
        return

    kz_values, x_dim = _photon_energy_render_data(tool, operation)
    for axis in axes:
        for index in range(kz_values.sizes["hv"]):
            kz = kz_values.isel(hv=index)
            axis.plot(kz[x_dim], kz, **_line_kwargs(operation, kz))
        if operation.show_legend:
            axis.legend(**operation.legend_kw)


def _photon_energy_render_data(
    tool: FigureComposerTool, operation: FigureOperationState
) -> tuple[xr.DataArray, str]:
    plan = _PhotonEnergyPlan.from_operation(operation)
    return tool._cached_render_data(
        "photon-energy-curves",
        plan,
        lambda: _kz_values_from_plan(
            _source_data_from_name(tool, plan.source),
            plan,
        ),
    )


def _label_code(operation: FigureOperationState) -> str:
    if operation.label_template == _DEFAULT_LABEL_TEMPLATE:
        return 'rf"$h\\nu = {kz.hv:g}$ eV"'
    return f"{operation.label_template!r}.format(hv=kz.hv)"


def _plot_call_code(
    operation: FigureOperationState, *, axis_name: str, x_dim: str
) -> str:
    line_kwargs = {
        key: value for key, value in operation.line_kw.items() if key != "label"
    }
    kwargs = {"label": _RawCode(_label_code(operation)), **line_kwargs}
    kwargs_text = _code_kwargs(kwargs)
    return f"{axis_name}.plot(kz.{x_dim}, kz, {kwargs_text})"


def _legend_call_code(operation: FigureOperationState, *, axis_name: str) -> str:
    kwargs_text = _code_kwargs(operation.legend_kw)
    if kwargs_text:
        return f"{axis_name}.legend({kwargs_text})"
    return f"{axis_name}.legend()"


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


def _photon_energy_code_lines(
    tool: FigureComposerTool, operation: FigureOperationState
) -> list[str]:
    if operation.hv_overlay_source is None:
        raise ValueError("Select a source data array for the photon-energy overlay")
    _, x_dim = _photon_energy_render_data(tool, operation)
    source_code = _valid_source_variable(operation.hv_overlay_source)
    values_code = _photon_energies_code(operation.photon_energies)
    call_code = f"{source_code}.kspace.hv_to_kz({values_code})"
    if operation.binding_energy is not None:
        call_code = f"{call_code}.qsel(eV={operation.binding_energy:g})"
    lines = [f"kz_values = {call_code}", ""]

    axis_code = _single_axis_code(tool, operation)
    if axis_code is not None:
        lines.extend(
            (
                "for i in range(len(kz_values.hv)):",
                "    kz = kz_values.isel(hv=i)",
                f"    {_plot_call_code(operation, axis_name=axis_code, x_dim=x_dim)}",
            )
        )
        if operation.show_legend:
            lines.append(_legend_call_code(operation, axis_name=axis_code))
        return lines

    lines.append(f"for ax in {_axes_sequence_code(tool._document, operation.axes)}:")
    lines.extend(
        (
            "    for i in range(len(kz_values.hv)):",
            "        kz = kz_values.isel(hv=i)",
            f"        {_plot_call_code(operation, axis_name='ax', x_dim=x_dim)}",
        )
    )
    if operation.show_legend:
        lines.append(f"    {_legend_call_code(operation, axis_name='ax')}")
    return lines


def _build_photon_editor(
    editor: FigureOperationEditor,
    operation: FigureOperationState,
    page: QtWidgets.QWidget,
    layout: QtWidgets.QFormLayout,
) -> None:
    photon_text, photon_mixed = editor.batch_text(
        operation, lambda target: target.photon_energies, _format_tuple
    )
    photon_edit = editor.line_edit(photon_text, parent=page)
    photon_edit.setObjectName("figureComposerPhotonEnergyValuesEdit")
    photon_edit.setPlaceholderText("Enter photon energies")
    editor.apply_mixed_line_edit(photon_edit, photon_mixed)
    editor.connect_line_edit_finished(
        photon_edit,
        lambda text: editor.request_update(
            photon_energies=_photon_energies_from_text(text)
        ),
    )
    editor.add_form_row(
        layout,
        "hν",
        photon_edit,
        "Photon energies in eV, entered as comma-separated numbers.",
    )

    binding_text, binding_mixed = editor.batch_text(
        operation, lambda target: target.binding_energy, _format_optional_float
    )
    binding_edit = editor.line_edit(binding_text, parent=page)
    binding_edit.setObjectName("figureComposerPhotonEnergyBindingEnergyEdit")
    editor.apply_mixed_line_edit(binding_edit, binding_mixed)
    editor.connect_line_edit_finished(
        binding_edit,
        lambda text: editor.request_update(
            binding_energy=_optional_float_from_text(text)
        ),
    )
    editor.add_form_row(
        layout,
        "Binding energy",
        binding_edit,
        "Electron binding energy in eV. Leave blank only when the overlay data "
        "already has no eV dimension.",
    )


def _build_style_editor(
    editor: FigureOperationEditor,
    operation: FigureOperationState,
    page: QtWidgets.QWidget,
    layout: QtWidgets.QFormLayout,
) -> None:
    color_text, color_mixed = editor.batch_text(
        operation,
        lambda target: line_kw_text(target, "color", "c") or "",
        str,
    )
    color_edit = _ColorLineEditWidget(color_text, parent=page)
    color_edit.setLineEditObjectName("figureComposerPhotonEnergyColorEdit")
    color_edit.setColorButtonObjectName("figureComposerPhotonEnergyColorButton")
    editor.apply_mixed_line_edit(color_edit.line_edit, color_mixed)
    editor.connect_value_signal(
        color_edit,
        color_edit.editingFinished,
        color_edit.text,
        lambda text: update_current_line_kw(
            editor, "color", color_kw_value_from_text(text), aliases=("c",)
        ),
        unchanged_mixed=lambda: editor.line_edit_batch_unchanged(color_edit.line_edit),
    )
    editor.add_form_row(
        layout,
        "Line color",
        color_edit,
        "Matplotlib color for photon-energy annotation curves.",
    )

    line_style_mixed = editor.batch_is_mixed(
        operation, lambda target: line_kw_style_value(target, "linestyle", "ls")
    )
    line_style_combo = editor.optional_name_combo(
        LINE_STYLE_OPTIONS,
        None if line_style_mixed else line_kw_style_value(operation, "linestyle", "ls"),
        LINE_STYLE_DEFAULT_LABEL,
        lambda text: update_current_line_kw(editor, "linestyle", text, aliases=("ls",)),
        parent=page,
        mixed=line_style_mixed,
    )
    line_style_combo.setObjectName("figureComposerPhotonEnergyLineStyleCombo")

    line_width_mixed = editor.batch_is_mixed(
        operation, lambda target: line_kw_text(target, "linewidth", "lw")
    )
    line_width_spin = optional_positive_spinbox(
        None if line_width_mixed else line_kw_float(operation, "linewidth", "lw"),
        parent=page,
    )
    line_width_spin.setObjectName("figureComposerPhotonEnergyLineWidthSpin")
    editor.connect_signal(
        line_width_spin,
        line_width_spin.valueChanged,
        lambda value: update_current_line_kw(
            editor,
            "linewidth",
            optional_positive_spinbox_value(value),
            aliases=("lw",),
        ),
    )
    editor.add_compound_form_row(
        layout,
        "Line",
        (
            (
                "Style",
                line_style_combo,
                "Matplotlib linestyle for photon-energy annotation curves.",
            ),
            (
                "Width",
                editor.mixed_value_widget(
                    line_width_spin, mixed=line_width_mixed, parent=page
                ),
                "Matplotlib linewidth for photon-energy annotation curves.",
            ),
        ),
        "Line style controls for photon-energy annotation curves.",
    )

    legend_mixed = editor.batch_is_mixed(operation, lambda target: target.show_legend)
    legend_check = editor.check_box(
        operation.show_legend,
        lambda checked: editor.request_update(show_legend=checked),
        parent=page,
        mixed=legend_mixed,
    )
    legend_check.setObjectName("figureComposerPhotonEnergyLegendCheck")
    legend_check.setText("")
    editor.add_form_row(
        layout,
        "Legend",
        legend_check,
        "Add or update the axes legend after plotting the photon-energy curves.",
    )

    legend_kw_text, legend_kw_mixed = editor.batch_text(
        operation, lambda target: target.legend_kw, _format_dict
    )
    legend_kw_edit = editor.line_edit(legend_kw_text, parent=page)
    legend_kw_edit.setObjectName("figureComposerPhotonEnergyLegendKwEdit")
    editor.apply_mixed_line_edit(legend_kw_edit, legend_kw_mixed)
    editor.connect_line_edit_finished(
        legend_kw_edit,
        lambda text: editor.request_update(legend_kw=_dict_from_text(text)),
    )
    editor.add_form_row(
        layout,
        "Legend kwargs",
        legend_kw_edit,
        "Keyword arguments forwarded to the Matplotlib legend.",
    )

    label_text, label_mixed = editor.batch_text(
        operation, lambda target: target.label_template, str
    )
    label_edit = editor.line_edit(label_text, parent=page)
    label_edit.setObjectName("figureComposerPhotonEnergyLabelTemplateEdit")
    editor.apply_mixed_line_edit(label_edit, label_mixed)
    editor.connect_line_edit_finished(
        label_edit,
        lambda text: editor.request_update(label_template=text),
    )
    editor.add_form_row(
        layout,
        "Label",
        label_edit,
        "Legend label template. Use {hv} for the photon energy.",
    )


def _build_editor(
    editor: FigureOperationEditor, operation: FigureOperationState
) -> list[tuple[str, str, QtWidgets.QWidget]]:
    photon_page, photon_layout = editor.new_form_page("figureComposerPhotonEnergyPage")
    style_page, style_layout = editor.new_form_page(
        "figureComposerPhotonEnergyStylePage"
    )
    _build_photon_editor(editor, operation, photon_page, photon_layout)
    _build_style_editor(editor, operation, style_page, style_layout)

    return [
        ("photon", "hν", photon_page),
        ("style", "Style", style_page),
    ]


def _editor_sections(
    editor: FigureOperationEditor, operation: FigureOperationState
) -> tuple[StepSection, ...]:
    return tuple(
        StepSection(key, title, page, _SECTION_TOOLTIPS[key])
        for key, title, page in _build_editor(editor, operation)
    )


def _seed_source(tool: FigureComposerTool) -> str | None:
    current = tool._current_operation()
    if current is not None:
        sources = tool._selected_sources_for_operation(current[1])
        if sources:
            return sources[0]
    source_names = tuple(tool._document.source_names())
    if tool._document.recipe.primary_source in source_names:
        return tool._document.recipe.primary_source
    return source_names[0] if source_names else None


def _seed_binding_energy(tool: FigureComposerTool) -> float | None:
    current = tool._current_operation()
    if current is None:
        return None
    operation = current[1]
    if (
        operation.kind == FigureOperationKind.PLOT_SLICES
        and operation.slice_dim == "eV"
        and len(operation.slice_values) == 1
    ):
        return operation.slice_values[0]
    return None


def _create_operation(tool: FigureComposerTool) -> FigureOperationState:
    return FigureOperationState.photon_energy_overlay(
        source=_seed_source(tool),
        axes=tool._selected_axes_state(),
        binding_energy=_seed_binding_energy(tool),
    )


def _display_text(tool: FigureComposerTool, operation: FigureOperationState) -> str:
    source = (
        tool._source_display_name(operation.hv_overlay_source)
        if operation.hv_overlay_source is not None
        else "missing source"
    )
    return f"Photon Energy Overlay: {source}"


def _tooltip(tool: FigureComposerTool, operation: FigureOperationState) -> str:
    return (
        "Draws constant-photon-energy curves on k_parallel-kz plots.\n"
        f"Targets: {tool._axes_target_text(operation.axes)}"
    )


def _has_invalid_target(
    context: FigureRecipeContext, operation: FigureOperationState
) -> bool:
    return context.axes_selection_has_invalid_target(operation.axes)


def _build_source_editor(
    editor: FigureOperationEditor, operation: FigureOperationState
) -> None:
    source_mixed = editor.batch_is_mixed(
        operation, lambda target: target.hv_overlay_source
    )
    source_combo = editor.source_combo(
        editor.context.source_names(),
        None if source_mixed else operation.hv_overlay_source,
        lambda source: editor.request_update_rebuild(hv_overlay_source=source),
        parent=editor.source_controls,
        mixed=source_mixed,
    )
    source_combo.setObjectName("figureComposerPhotonEnergySourceCombo")
    editor.add_source_row(
        "Overlay data",
        source_combo,
        "Data array used to compute the photon-energy annotation curves.",
    )


def _section_summary(
    tool: FigureComposerTool, key: str, operation: FigureOperationState
) -> str:
    match key:
        case "sources":
            if operation.hv_overlay_source is None:
                return "none"
            return tool._source_display_name(operation.hv_overlay_source)
        case "axes":
            return tool._axes_target_text(operation.axes)
        case "photon":
            energies = _format_tuple(operation.photon_energies) or "no energies"
            if operation.binding_energy is None:
                return energies
            return f"{energies}; eV={operation.binding_energy:g}"
        case "style":
            if operation.show_legend:
                if operation.legend_kw:
                    return "legend kwargs"
                return line_kw_text(operation, "color", "c") or "legend"
            return line_kw_text(operation, "color", "c") or "no legend"
    return ""


def _required_imports(
    _tool: FigureComposerTool, _operation: FigureOperationState
) -> tuple[str, ...]:
    return ("import erlab",)


SPEC = OperationSpec(
    kind=FigureOperationKind.PHOTON_ENERGY_OVERLAY,
    add_actions=(
        AddStepActionSpec(
            action_id=FigureOperationKind.PHOTON_ENERGY_OVERLAY.value,
            text="Photon Energy Overlay",
            tooltip="Add photon-energy annotation curves to a k_parallel-kz plot.",
            create_operation=_create_operation,
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
    render=lambda tool, operation, _figure, axs: _render_photon_energy_overlay(
        tool, operation, axs
    ),
    code_lines=_photon_energy_code_lines,
    render_cache_safe=_always_render_cache_safe,
    required_imports=_required_imports,
)
