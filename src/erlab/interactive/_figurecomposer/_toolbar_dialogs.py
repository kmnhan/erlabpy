"""Recipe-backed dialogs opened from the Figure Composer plot toolbar."""

from __future__ import annotations

import typing

import matplotlib.scale as mscale
import numpy as np
from qtpy import QtCore, QtWidgets

import erlab
from erlab.interactive._figurecomposer._gridspec import (
    _gridspec_all_axes_ids,
    _gridspec_axis_display_name,
    _gridspec_valid_axes_ids,
)
from erlab.interactive._figurecomposer._rendering import (
    _axes_from_selection,
    _iter_axes,
    _render_preview,
)
from erlab.interactive._figurecomposer._state import (
    FigureAxesSelectionState,
    FigureMethodFamily,
    FigureOperationKind,
    FigureOperationState,
)
from erlab.interactive._figurecomposer._widgets import (
    _AxesSelectorWidget,
    _GridSpecViewWidget,
)

if typing.TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from matplotlib.axes import Axes

    from erlab.interactive._figurecomposer._tool import FigureComposerTool


_LAYOUT_ENGINE_OPTIONS = ("none", "tight", "constrained", "compressed")


def show_subplot_adjust_dialog(tool: FigureComposerTool) -> None:
    """Show the toolbar subplot adjustment dialog for *tool*."""
    if _show_existing_dialog(tool, "_subplot_adjust_dialog"):
        return

    dialog = QtWidgets.QDialog(_dialog_parent(tool))
    dialog.setObjectName("figureComposerToolbarSubplotAdjustDialog")
    dialog.setWindowTitle("Adjust Subplots")
    dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
    root_layout = QtWidgets.QVBoxLayout(dialog)

    engine_row = QtWidgets.QHBoxLayout()
    engine_label = QtWidgets.QLabel("Layout engine", dialog)
    engine_combo = QtWidgets.QComboBox(dialog)
    engine_combo.setObjectName("figureComposerToolbarLayoutEngineCombo")
    engine_combo.addItems(_LAYOUT_ENGINE_OPTIONS)
    engine_combo.setCurrentText(_layout_engine_text(tool))
    engine_combo.setToolTip(
        "Recipe step for fig.set_layout_engine.\n"
        "Use none when manually adjusting subplot borders and spacing."
    )
    engine_row.addWidget(engine_label)
    engine_row.addWidget(engine_combo, 1)
    root_layout.addLayout(engine_row)

    body_layout = QtWidgets.QHBoxLayout()
    root_layout.addLayout(body_layout)
    border_group = QtWidgets.QGroupBox("Borders", dialog)
    border_layout = QtWidgets.QFormLayout(border_group)
    spacing_group = QtWidgets.QGroupBox("Spacing", dialog)
    spacing_layout = QtWidgets.QFormLayout(spacing_group)
    body_layout.addWidget(border_group)
    body_layout.addWidget(spacing_group)

    subplotpars = tool.figure.subplotpars
    spinboxes: dict[str, QtWidgets.QDoubleSpinBox] = {}

    def add_spinbox(key: str, label: str, layout: QtWidgets.QFormLayout) -> None:
        spinbox = QtWidgets.QDoubleSpinBox(dialog)
        spinbox.setObjectName(f"figureComposerToolbarSubplotAdjust_{key}")
        spinbox.setRange(0.0, 1.0)
        spinbox.setDecimals(3)
        spinbox.setSingleStep(0.005)
        spinbox.setKeyboardTracking(False)
        spinbox.setValue(float(getattr(subplotpars, key)))
        spinbox.setToolTip(
            "Recipe value passed to fig.subplots_adjust.\n"
            "Changes update the figure immediately."
        )
        spinboxes[key] = spinbox
        layout.addRow(label, spinbox)

    for key in ("top", "bottom", "left", "right"):
        add_spinbox(key, key, border_layout)
    for key in ("hspace", "wspace"):
        add_spinbox(key, key, spacing_layout)

    def adjust_enabled() -> bool:
        return engine_combo.currentText() == "none"

    def update_adjust_enabled() -> None:
        enabled = adjust_enabled()
        tooltip = (
            "Manual subplot adjustment is active."
            if enabled
            else "Set the layout engine to none to edit subplot spacing manually."
        )
        for spinbox in spinboxes.values():
            spinbox.setEnabled(enabled)
            spinbox.setToolTip(tooltip)

    def update_adjust_operation() -> None:
        if tool._updating_controls or not adjust_enabled():
            return
        _upsert_method_operation(
            tool,
            FigureMethodFamily.FIGURE,
            "subplots_adjust",
            label="Adjust subplots",
            kwargs={key: spinbox.value() for key, spinbox in spinboxes.items()},
            enabled=True,
        )

    def update_engine(text: str) -> None:
        if tool._updating_controls:
            return
        _upsert_method_operation(
            tool,
            FigureMethodFamily.FIGURE,
            "set_layout_engine",
            label="Set layout engine",
            args=(text,),
        )
        if text == "none":
            update_adjust_enabled()
            update_adjust_operation()
        else:
            _set_method_operation_enabled(
                tool,
                FigureMethodFamily.FIGURE,
                "subplots_adjust",
                axes=None,
                enabled=False,
            )
            update_adjust_enabled()

    def engine_activated(_index: int) -> None:
        update_engine(engine_combo.currentText())

    def adjust_spinbox_changed(_value: float) -> None:
        update_adjust_operation()

    engine_combo.activated.connect(engine_activated)
    for spinbox in spinboxes.values():
        spinbox.valueChanged.connect(adjust_spinbox_changed)
    update_adjust_enabled()

    button_box = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.StandardButton.Close,
        dialog,
    )
    button_box.rejected.connect(dialog.close)
    root_layout.addWidget(button_box)

    _show_toolbar_dialog(tool, "_subplot_adjust_dialog", dialog)


def show_axes_customize_dialog(tool: FigureComposerTool) -> None:
    """Show the toolbar axes customization dialog for *tool*."""
    if _show_existing_dialog(tool, "_axes_customize_dialog"):
        return

    dialog = QtWidgets.QDialog(_dialog_parent(tool))
    dialog.setObjectName("figureComposerToolbarAxesCustomizeDialog")
    dialog.setWindowTitle("Customize Axes")
    dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
    root_layout = QtWidgets.QVBoxLayout(dialog)

    selector = _selector_widget(tool, dialog)
    root_layout.addWidget(selector)

    form_host = QtWidgets.QWidget(dialog)
    form_layout = QtWidgets.QGridLayout(form_host)
    form_layout.setContentsMargins(0, 0, 0, 0)
    form_layout.setColumnStretch(1, 1)
    form_layout.setColumnStretch(3, 1)
    root_layout.addWidget(form_host)

    title_edit = QtWidgets.QLineEdit(form_host)
    title_edit.setObjectName("figureComposerToolbarAxesTitleEdit")
    xlabel_edit = QtWidgets.QLineEdit(form_host)
    xlabel_edit.setObjectName("figureComposerToolbarAxesXLabelEdit")
    ylabel_edit = QtWidgets.QLineEdit(form_host)
    ylabel_edit.setObjectName("figureComposerToolbarAxesYLabelEdit")
    xlim_edit = QtWidgets.QLineEdit(form_host)
    xlim_edit.setObjectName("figureComposerToolbarAxesXLimEdit")
    ylim_edit = QtWidgets.QLineEdit(form_host)
    ylim_edit.setObjectName("figureComposerToolbarAxesYLimEdit")
    aspect_edit = QtWidgets.QLineEdit(form_host)
    aspect_edit.setObjectName("figureComposerToolbarAxesAspectEdit")

    xscale_combo = QtWidgets.QComboBox(form_host)
    xscale_combo.setObjectName("figureComposerToolbarAxesXScaleCombo")
    yscale_combo = QtWidgets.QComboBox(form_host)
    yscale_combo.setObjectName("figureComposerToolbarAxesYScaleCombo")
    scale_names = tuple(mscale.get_scale_names())
    xscale_combo.addItems(scale_names)
    yscale_combo.addItems(scale_names)

    grid_check = QtWidgets.QCheckBox(form_host)
    grid_check.setObjectName("figureComposerToolbarAxesGridCheck")
    grid_axis_combo = QtWidgets.QComboBox(form_host)
    grid_axis_combo.setObjectName("figureComposerToolbarAxesGridAxisCombo")
    grid_axis_combo.addItems(["both", "x", "y"])
    grid_which_combo = QtWidgets.QComboBox(form_host)
    grid_which_combo.setObjectName("figureComposerToolbarAxesGridWhichCombo")
    grid_which_combo.addItems(["major", "minor", "both"])

    style_combo = QtWidgets.QComboBox(form_host)
    style_combo.setObjectName("figureComposerToolbarAxesStyleStepCombo")
    style_button = QtWidgets.QPushButton("Select Step", form_host)
    style_button.setObjectName("figureComposerToolbarAxesStyleStepButton")
    style_button.setToolTip(
        "Select the recipe step that draws on these axes so its detailed\n"
        "line, marker, image, and color controls are available in the editor."
    )

    rows = (
        ("Title", title_edit, "X label", xlabel_edit),
        ("Y label", ylabel_edit, "Aspect", aspect_edit),
        ("X limits", xlim_edit, "Y limits", ylim_edit),
        ("X scale", xscale_combo, "Y scale", yscale_combo),
        ("Grid", grid_check, "Grid axis", grid_axis_combo),
        ("Grid ticks", grid_which_combo, "Plot/style step", style_combo),
    )
    for row, (left_label, left_widget, right_label, right_widget) in enumerate(rows):
        form_layout.addWidget(QtWidgets.QLabel(left_label, form_host), row, 0)
        form_layout.addWidget(left_widget, row, 1)
        form_layout.addWidget(QtWidgets.QLabel(right_label, form_host), row, 2)
        form_layout.addWidget(right_widget, row, 3)
    form_layout.addWidget(style_button, len(rows), 3)

    for edit in (
        title_edit,
        xlabel_edit,
        ylabel_edit,
        xlim_edit,
        ylim_edit,
        aspect_edit,
    ):
        edit.setToolTip(
            "Editing this value adds or updates the matching ax.* recipe step\n"
            "for the selected axes."
        )
    for combo in (xscale_combo, yscale_combo, grid_axis_combo, grid_which_combo):
        combo.setToolTip(
            "Changing this value adds or updates the matching ax.* recipe step\n"
            "for the selected axes."
        )

    updating = False

    def current_selection() -> FigureAxesSelectionState:
        return _selector_selection(tool, selector)

    def upsert_axis_method(
        name: str,
        *,
        args: Sequence[typing.Any] = (),
        kwargs: Mapping[str, typing.Any] | None = None,
    ) -> None:
        _upsert_method_operation(
            tool,
            FigureMethodFamily.AXES,
            name,
            axes=current_selection(),
            args=args,
            kwargs=kwargs,
        )

    def refresh_style_steps() -> None:
        selected_axes = set(_axes_for_selection(tool, current_selection()))
        style_combo.blockSignals(True)
        try:
            style_combo.clear()
            if selected_axes:
                for index, operation in enumerate(tool._recipe.operations):
                    if operation.kind not in {
                        FigureOperationKind.PLOT_SLICES,
                        FigureOperationKind.LINE,
                    }:
                        continue
                    if selected_axes & set(_axes_for_selection(tool, operation.axes)):
                        style_combo.addItem(
                            tool._operation_display_text(operation), index
                        )
            style_button.setEnabled(style_combo.count() > 0)
        finally:
            style_combo.blockSignals(False)

    def refresh_from_axis() -> None:
        nonlocal updating
        axis = _first_axis(tool, current_selection())
        updating = True
        try:
            if axis is None:
                for widget in (
                    title_edit,
                    xlabel_edit,
                    ylabel_edit,
                    xlim_edit,
                    ylim_edit,
                    aspect_edit,
                ):
                    widget.clear()
                    widget.setEnabled(False)
                for combo in (xscale_combo, yscale_combo):
                    combo.setEnabled(False)
                return
            title_edit.setEnabled(True)
            xlabel_edit.setEnabled(True)
            ylabel_edit.setEnabled(True)
            xlim_edit.setEnabled(True)
            ylim_edit.setEnabled(True)
            aspect_edit.setEnabled(True)
            xscale_combo.setEnabled(True)
            yscale_combo.setEnabled(True)
            title_edit.setText(axis.get_title())
            xlabel_edit.setText(axis.get_xlabel())
            ylabel_edit.setText(axis.get_ylabel())
            xlim_edit.setText(_float_pair_text(axis.get_xlim()))
            ylim_edit.setText(_float_pair_text(axis.get_ylim()))
            aspect_edit.setText(_aspect_text(axis.get_aspect()))
            xscale_combo.setCurrentText(axis.get_xscale())
            yscale_combo.setCurrentText(axis.get_yscale())
            for edit in (
                title_edit,
                xlabel_edit,
                ylabel_edit,
                xlim_edit,
                ylim_edit,
                aspect_edit,
            ):
                edit.setModified(False)
        finally:
            updating = False
        refresh_style_steps()

    def update_text_method(edit: QtWidgets.QLineEdit, name: str) -> None:
        if updating or not edit.isModified():
            return
        upsert_axis_method(name, args=(edit.text(),))
        edit.setModified(False)

    def update_limit_method(edit: QtWidgets.QLineEdit, name: str) -> None:
        if updating or not edit.isModified():
            return
        limits = _float_pair_from_text(edit.text())
        if limits is None:
            return
        upsert_axis_method(name, args=limits)
        edit.setModified(False)

    def update_aspect() -> None:
        if updating or not aspect_edit.isModified():
            return
        try:
            aspect = _aspect_value(aspect_edit.text())
        except ValueError:
            return
        upsert_axis_method("set_aspect", args=(aspect,))
        aspect_edit.setModified(False)

    def update_grid() -> None:
        if updating:
            return
        upsert_axis_method(
            "grid",
            args=(grid_check.isChecked(),),
            kwargs={
                "which": grid_which_combo.currentText(),
                "axis": grid_axis_combo.currentText(),
            },
        )

    def open_style_step() -> None:
        index = style_combo.currentData()
        if not isinstance(index, int):
            return
        tool.operation_list.setCurrentRow(index)
        if tool.step_section_keys:
            tool._select_step_section(tool.step_section_keys[0])
        tool.raise_()
        tool.activateWindow()

    def title_finished() -> None:
        update_text_method(title_edit, "set_title")

    def xlabel_finished() -> None:
        update_text_method(xlabel_edit, "set_xlabel")

    def ylabel_finished() -> None:
        update_text_method(ylabel_edit, "set_ylabel")

    def xlim_finished() -> None:
        update_limit_method(xlim_edit, "set_xlim")

    def ylim_finished() -> None:
        update_limit_method(ylim_edit, "set_ylim")

    def xscale_activated(_index: int) -> None:
        upsert_axis_method("set_xscale", args=(xscale_combo.currentText(),))

    def yscale_activated(_index: int) -> None:
        upsert_axis_method("set_yscale", args=(yscale_combo.currentText(),))

    def grid_toggled(_checked: bool) -> None:
        update_grid()

    def grid_combo_activated(_index: int) -> None:
        update_grid()

    def selection_changed(_selection: object) -> None:
        refresh_from_axis()

    title_edit.editingFinished.connect(title_finished)
    xlabel_edit.editingFinished.connect(xlabel_finished)
    ylabel_edit.editingFinished.connect(ylabel_finished)
    xlim_edit.editingFinished.connect(xlim_finished)
    ylim_edit.editingFinished.connect(ylim_finished)
    aspect_edit.editingFinished.connect(update_aspect)
    xscale_combo.activated.connect(xscale_activated)
    yscale_combo.activated.connect(yscale_activated)
    grid_check.toggled.connect(grid_toggled)
    grid_axis_combo.activated.connect(grid_combo_activated)
    grid_which_combo.activated.connect(grid_combo_activated)
    style_button.clicked.connect(open_style_step)
    selector.sigSelectionChanged.connect(selection_changed)
    refresh_from_axis()

    button_box = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.StandardButton.Close,
        dialog,
    )
    button_box.rejected.connect(dialog.close)
    root_layout.addWidget(button_box)

    _show_toolbar_dialog(tool, "_axes_customize_dialog", dialog)


def close_toolbar_dialogs(tool: FigureComposerTool) -> None:
    """Close modeless toolbar dialogs owned by *tool*."""
    for attr_name in ("_subplot_adjust_dialog", "_axes_customize_dialog"):
        dialog = getattr(tool, attr_name)
        if isinstance(
            dialog, QtWidgets.QDialog
        ) and erlab.interactive.utils.qt_is_valid(dialog):
            dialog.close()
        setattr(tool, attr_name, None)


def _dialog_parent(tool: FigureComposerTool) -> QtWidgets.QWidget:
    if tool._figure_window is not None and erlab.interactive.utils.qt_is_valid(
        tool._figure_window
    ):
        return tool._figure_window
    return tool


def _show_existing_dialog(tool: FigureComposerTool, attr_name: str) -> bool:
    existing = getattr(tool, attr_name)
    if not isinstance(existing, QtWidgets.QDialog):
        return False
    if not erlab.interactive.utils.qt_is_valid(existing):
        return False
    _show_toolbar_dialog(tool, attr_name, existing)
    return True


def _show_toolbar_dialog(
    tool: FigureComposerTool, attr_name: str, dialog: QtWidgets.QDialog
) -> None:
    setattr(tool, attr_name, dialog)

    def clear_reference(_obj: QtCore.QObject | None = None) -> None:
        if not erlab.interactive.utils.qt_is_valid(tool):
            return
        if getattr(tool, attr_name) is dialog:
            setattr(tool, attr_name, None)

    dialog.destroyed.connect(clear_reference)
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()


def _method_axes_match(
    operation_axes: FigureAxesSelectionState,
    axes: FigureAxesSelectionState | None,
) -> bool:
    if axes is None:
        return True
    return (
        operation_axes.axes == axes.axes
        and operation_axes.axes_ids == axes.axes_ids
        and operation_axes.expression == axes.expression
    )


def _upsert_method_operation(
    tool: FigureComposerTool,
    family: FigureMethodFamily,
    name: str,
    *,
    label: str | None = None,
    axes: FigureAxesSelectionState | None = None,
    args: Sequence[typing.Any] = (),
    kwargs: Mapping[str, typing.Any] | None = None,
    enabled: bool | None = None,
) -> int:
    current = tool._current_operation()
    current_id = current[1].operation_id if current is not None else None
    selected_ids = tool._selected_operation_ids()
    operations = list(tool._recipe.operations)
    updates: dict[str, typing.Any] = {
        "label": label or name,
        "method_args": tuple(args),
        "method_kwargs": dict(kwargs or {}),
    }
    if axes is not None:
        updates["axes"] = axes
    if enabled is not None:
        updates["enabled"] = enabled
    for index, operation in enumerate(operations):
        if (
            operation.kind == FigureOperationKind.METHOD
            and operation.method_family == family
            and operation.method_name == name
            and _method_axes_match(operation.axes, axes)
        ):
            operations[index] = operation.model_copy(update=updates)
            break
    else:
        new_operation = FigureOperationState.method(
            family=family,
            name=name,
            label=label,
            axes=axes,
            args=args,
            kwargs=kwargs,
        )
        if enabled is not None:
            new_operation = new_operation.model_copy(update={"enabled": enabled})
        operations.append(new_operation)
        index = len(operations) - 1
    _set_operations(tool, tuple(operations), current_id, selected_ids)
    return index


def _set_method_operation_enabled(
    tool: FigureComposerTool,
    family: FigureMethodFamily,
    name: str,
    *,
    axes: FigureAxesSelectionState | None,
    enabled: bool,
) -> None:
    current = tool._current_operation()
    current_id = current[1].operation_id if current is not None else None
    selected_ids = tool._selected_operation_ids()
    operations = list(tool._recipe.operations)
    changed = False
    for index, operation in enumerate(operations):
        if (
            operation.kind == FigureOperationKind.METHOD
            and operation.method_family == family
            and operation.method_name == name
            and _method_axes_match(operation.axes, axes)
            and operation.enabled != enabled
        ):
            operations[index] = operation.model_copy(update={"enabled": enabled})
            changed = True
    if changed:
        _set_operations(tool, tuple(operations), current_id, selected_ids)


def _set_operations(
    tool: FigureComposerTool,
    operations: tuple[FigureOperationState, ...],
    current_id: str | None,
    selected_ids: set[str],
) -> None:
    tool._recipe = tool._recipe.model_copy(update={"operations": operations})
    tool._refresh_operation_list()
    if selected_ids:
        tool._set_selected_operation_ids_silent(selected_ids)
    if current_id is not None:
        for row, operation in enumerate(tool._recipe.operations):
            if operation.operation_id == current_id:
                tool._set_current_operation_row_silent(row)
                break
    tool._sync_axes_selector()
    tool._update_step_action_buttons()
    tool._refresh_step_section_button_texts()
    current = tool._current_operation()
    tool._update_source_status(current[1] if current is not None else None)
    tool._update_operation_editor_safely()
    _render_preview(tool)
    tool.sigInfoChanged.emit()


def _layout_axes(tool: FigureComposerTool) -> np.ndarray | dict[str, Axes] | None:
    setup = tool._recipe.setup
    if setup.layout_mode == "gridspec":
        axes_ids = _gridspec_valid_axes_ids(setup, _gridspec_all_axes_ids(setup))
        axes = tool.figure.axes[: len(axes_ids)]
        if len(axes) < len(axes_ids):
            return None
        return dict(zip(axes_ids, axes, strict=True))
    count = setup.nrows * setup.ncols
    axes = tool.figure.axes[:count]
    if len(axes) < count:
        return None
    return np.asarray(axes, dtype=object).reshape(setup.nrows, setup.ncols)


def _axes_for_selection(
    tool: FigureComposerTool, selection: FigureAxesSelectionState
) -> tuple[Axes, ...]:
    layout_axes = _layout_axes(tool)
    if layout_axes is None:
        return ()
    try:
        axes_obj = _axes_from_selection(
            tool, selection, layout_axes, for_plot_slices=False
        )
    except (IndexError, TypeError, ValueError):
        return ()
    return _iter_axes(axes_obj)


def _first_axis(
    tool: FigureComposerTool, selection: FigureAxesSelectionState
) -> Axes | None:
    axes = _axes_for_selection(tool, selection)
    return axes[0] if axes else None


def _layout_engine_text(tool: FigureComposerTool) -> str:
    for operation in reversed(tool._recipe.operations):
        if (
            operation.enabled
            and operation.kind == FigureOperationKind.METHOD
            and operation.method_family == FigureMethodFamily.FIGURE
            and operation.method_name == "set_layout_engine"
            and operation.method_args
            and isinstance(operation.method_args[0], str)
        ):
            return operation.method_args[0]
    return tool._recipe.setup.layout or "none"


def _float_pair_text(values: Sequence[float]) -> str:
    if len(values) < 2:
        return ""
    return f"{float(values[0]):g}, {float(values[1]):g}"


def _float_pair_from_text(text: str) -> tuple[float, float] | None:
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) != 2:
        return None
    return float(parts[0]), float(parts[1])


def _aspect_text(value: typing.Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int | float):
        return f"{float(value):g}"
    return str(value)


def _aspect_value(text: str) -> str | float:
    stripped = text.strip()
    if stripped in {"auto", "equal"}:
        return stripped
    return float(stripped)


def _selector_widget(
    tool: FigureComposerTool, parent: QtWidgets.QWidget
) -> _AxesSelectorWidget | _GridSpecViewWidget:
    setup = tool._recipe.setup
    if setup.layout_mode == "gridspec":
        selector = _GridSpecViewWidget(parent, mode="select")
        labels = {
            axes_id: _gridspec_axis_display_name(setup, axes_id)
            for axes_id in _gridspec_all_axes_ids(setup)
        }
        selector.set_layout(setup.gridspec.root, labels)
        selected_axes = tool._selected_axes_state().axes_ids
        if not selected_axes:
            selected_axes = _gridspec_valid_axes_ids(
                setup, _gridspec_all_axes_ids(setup)
            )[:1]
        selector.set_selected_axes_ids(selected_axes)
        return selector
    selector = _AxesSelectorWidget(parent)
    selector.set_grid(setup.nrows, setup.ncols)
    selected_axes = tool._selected_axes_state().valid_axes(setup)
    selector.set_selected_axes(selected_axes or ((0, 0),))
    return selector


def _selector_selection(
    tool: FigureComposerTool, selector: _AxesSelectorWidget | _GridSpecViewWidget
) -> FigureAxesSelectionState:
    if isinstance(selector, _GridSpecViewWidget):
        axes_ids = selector.selected_axes_ids()
        if not axes_ids:
            axes_ids = _gridspec_valid_axes_ids(
                tool._recipe.setup,
                _gridspec_all_axes_ids(tool._recipe.setup),
            )[:1]
        return FigureAxesSelectionState(axes_ids=axes_ids)
    axes = selector.selected_axes()
    return FigureAxesSelectionState(axes=axes or ((0, 0),))
