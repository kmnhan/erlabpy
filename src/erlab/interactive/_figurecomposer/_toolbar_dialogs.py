"""Recipe-backed dialogs opened from the Figure Composer plot toolbar."""

from __future__ import annotations

import functools
import typing
import weakref

import matplotlib.scale as mscale
import numpy as np
from qtpy import QtCore, QtWidgets

import erlab
from erlab.interactive._figurecomposer._editor_controls import (
    MIXED_VALUE,
    MIXED_VALUES_TEXT,
)
from erlab.interactive._figurecomposer._gridspec import (
    _gridspec_all_axes_ids,
    _gridspec_axis_display_name,
    _gridspec_valid_axes_ids,
)
from erlab.interactive._figurecomposer._line_colormap import (
    LINE_COLOR_CMAP_TRIM_MAX,
    effective_line_color_coord,
    line_color_cmap_trim_control_value,
    line_colormap_active,
)
from erlab.interactive._figurecomposer._line_style import (
    CONTROLLED_LINE_KW_KEYS,
    LINE_MARKER_OPTIONS,
    LINE_STYLE_OPTIONS,
    color_kw_value_from_text,
    configure_style_combo,
    extra_line_kw,
    line_kw_float,
    line_kw_style_value,
    line_kw_text,
    optional_positive_spinbox,
    optional_positive_spinbox_value,
    style_combo_value,
)
from erlab.interactive._figurecomposer._norms import (
    _NORM_CHOICES,
    _cmap_base_and_reverse,
    _cmap_with_reverse,
    _effective_norm_name,
    _norm_kwarg_fields,
)
from erlab.interactive._figurecomposer._operations._line_profile import (
    _available_line_color_coords,
    _line_color_mode_from_text,
    _line_color_mode_text,
)
from erlab.interactive._figurecomposer._operations._plot_slices import (
    _PLOT_SLICES_PANEL_IMAGE,
    _PLOT_SLICES_PANEL_LINE,
    _available_plot_slices_line_color_coords,
    _norm_clip_from_text,
    _norm_clip_text,
    _PanelLineStyleEditorWidget,
    _PanelStyleEditorWidget,
    _plot_slices_panel_keys,
    _plot_slices_panel_kind,
    _plot_slices_shape,
    _PlotSlicesPanelKey,
)
from erlab.interactive._figurecomposer._operations._plot_slices import (
    _line_color_mode_from_text as _plot_slices_line_color_mode_from_text,
)
from erlab.interactive._figurecomposer._operations._plot_slices import (
    _line_color_mode_text as _plot_slices_line_color_mode_text,
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
    FigurePlotSlicesPanelStyleState,
)
from erlab.interactive._figurecomposer._subplot_adjust import (
    SUBPLOTS_ADJUST_SPINBOX_DECIMALS,
    SUBPLOTS_ADJUST_SPINBOX_MAXIMUM,
    SUBPLOTS_ADJUST_SPINBOX_MINIMUM,
    SUBPLOTS_ADJUST_SPINBOX_STEP,
    normalize_subplots_adjust_kwargs,
    subplots_adjust_spinbox_range,
)
from erlab.interactive._figurecomposer._text import (
    FigureComposerInputError,
    _dict_from_text,
    _format_dict,
    _limit_pair_from_text,
)
from erlab.interactive._figurecomposer._widgets import (
    _AxesSelectorWidget,
    _ColorLineEditWidget,
    _ColorListEditorWidget,
    _GridSpecViewWidget,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from matplotlib.axes import Axes

    from erlab.interactive._figurecomposer._tool import FigureComposerTool


_LAYOUT_ENGINE_OPTIONS = ("default", "none", "tight", "constrained", "compressed")
_STYLE_TARGET_PLOT_SLICES = "plot_slices"
_STYLE_TARGET_LINE = "line"
_STYLE_TARGET_COMBO_MINIMUM_CONTENTS = 28


class _StyleTarget(typing.NamedTuple):
    operation_id: str
    operation_index: int
    target_kind: str
    label: str
    panel_keys: tuple[_PlotSlicesPanelKey, ...] = ()
    tooltip: str = ""


class _AxisValueState(typing.NamedTuple):
    value: typing.Any = None
    mixed: bool = False
    available: bool = False


class _ElidingComboBox(QtWidgets.QComboBox):
    """Combo box whose closed label elides instead of driving dialog width."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumContentsLength(_STYLE_TARGET_COMBO_MINIMUM_CONTENTS)
        self.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        view = self.view()
        if view is not None:  # pragma: no branch - Qt creates the popup view.
            view.setTextElideMode(QtCore.Qt.TextElideMode.ElideMiddle)


def _add_axis_form_row(
    layout: QtWidgets.QFormLayout,
    label: str,
    widget: QtWidgets.QWidget,
    tooltip: str,
) -> None:
    widget.setToolTip(tooltip)
    layout.addRow(label, widget)
    label_widget = layout.labelForField(widget)
    if label_widget is not None:
        label_widget.setToolTip(tooltip)


def _add_axis_compound_form_row(
    layout: QtWidgets.QFormLayout,
    label: str,
    object_name: str,
    controls: Sequence[tuple[str, QtWidgets.QWidget, str]],
    tooltip: str,
) -> None:
    parent = layout.parentWidget()
    row_widget = QtWidgets.QWidget(parent)
    row_widget.setObjectName(object_name)
    row_widget.setToolTip(tooltip)
    row_layout = QtWidgets.QHBoxLayout(row_widget)
    row_layout.setContentsMargins(0, 0, 0, 0)
    for control_label, widget, control_tooltip in controls:
        if control_label:
            label_widget = QtWidgets.QLabel(control_label, row_widget)
            label_widget.setBuddy(widget)
            label_widget.setToolTip(control_tooltip)
            row_layout.addWidget(label_widget)
        widget.setToolTip(control_tooltip)
        row_layout.addWidget(
            widget, 0 if isinstance(widget, QtWidgets.QCheckBox) else 1
        )
    layout.addRow(label, row_widget)
    form_label = layout.labelForField(row_widget)
    if form_label is not None:
        form_label.setToolTip(tooltip)


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
        "Figure layout engine from the Layout tab.\n"
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
        spinbox.setRange(
            SUBPLOTS_ADJUST_SPINBOX_MINIMUM,
            SUBPLOTS_ADJUST_SPINBOX_MAXIMUM,
        )
        spinbox.setDecimals(SUBPLOTS_ADJUST_SPINBOX_DECIMALS)
        spinbox.setSingleStep(SUBPLOTS_ADJUST_SPINBOX_STEP)
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

    def spinbox_values() -> dict[str, float]:
        return {key: spinbox.value() for key, spinbox in spinboxes.items()}

    def update_spinbox_ranges() -> None:
        for key, spinbox in spinboxes.items():
            values = spinbox_values()
            minimum, maximum = subplots_adjust_spinbox_range(key, values)
            with QtCore.QSignalBlocker(spinbox):
                spinbox.setRange(minimum, maximum)

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

    def update_adjust_operation(*, changed_key: str | None = None) -> None:
        if tool._updating_controls or not adjust_enabled():
            return
        _upsert_method_operation(
            tool,
            FigureMethodFamily.FIGURE,
            "subplots_adjust",
            label="Adjust subplots",
            kwargs=normalize_subplots_adjust_kwargs(
                spinbox_values(),
                changed_key=changed_key,
            ),
            enabled=True,
        )

    def update_engine(text: str) -> None:
        if tool._updating_controls:
            return
        if text == "none":
            _set_setup_layout_engine(tool, text)
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
            _set_setup_layout_engine(tool, text)
            update_adjust_enabled()

    def engine_activated(_index: int) -> None:
        update_engine(engine_combo.currentText())

    def adjust_spinbox_changed(key: str, _value: float) -> None:
        update_spinbox_ranges()
        update_adjust_operation(changed_key=key)

    engine_combo.activated.connect(engine_activated)
    for key, spinbox in spinboxes.items():
        spinbox.valueChanged.connect(functools.partial(adjust_spinbox_changed, key))
    update_spinbox_ranges()
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

    tab_widget = QtWidgets.QTabWidget(dialog)
    tab_widget.setObjectName("figureComposerToolbarCustomizeTabs")
    root_layout.addWidget(tab_widget, 1)

    axes_page = QtWidgets.QWidget(tab_widget)
    form_layout = QtWidgets.QFormLayout(axes_page)
    form_layout.setFieldGrowthPolicy(
        QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
    )
    tab_widget.addTab(axes_page, "Axes")

    curves_page, curves_combo, curves_layout = _style_tab_page(
        tab_widget,
        combo_object_name="figureComposerToolbarCurveTargetCombo",
    )
    images_page, images_combo, images_layout = _style_tab_page(
        tab_widget,
        combo_object_name="figureComposerToolbarImageTargetCombo",
    )
    curves_tab_index = tab_widget.addTab(curves_page, "Curves")
    images_tab_index = tab_widget.addTab(images_page, "Images")

    title_edit = QtWidgets.QLineEdit(axes_page)
    title_edit.setObjectName("figureComposerToolbarAxesTitleEdit")
    xlabel_edit = QtWidgets.QLineEdit(axes_page)
    xlabel_edit.setObjectName("figureComposerToolbarAxesXLabelEdit")
    ylabel_edit = QtWidgets.QLineEdit(axes_page)
    ylabel_edit.setObjectName("figureComposerToolbarAxesYLabelEdit")
    xlim_edit = QtWidgets.QLineEdit(axes_page)
    xlim_edit.setObjectName("figureComposerToolbarAxesXLimEdit")
    ylim_edit = QtWidgets.QLineEdit(axes_page)
    ylim_edit.setObjectName("figureComposerToolbarAxesYLimEdit")
    aspect_edit = QtWidgets.QLineEdit(axes_page)
    aspect_edit.setObjectName("figureComposerToolbarAxesAspectEdit")

    xscale_combo = QtWidgets.QComboBox(axes_page)
    xscale_combo.setObjectName("figureComposerToolbarAxesXScaleCombo")
    yscale_combo = QtWidgets.QComboBox(axes_page)
    yscale_combo.setObjectName("figureComposerToolbarAxesYScaleCombo")
    scale_names = tuple(mscale.get_scale_names())
    xscale_combo.addItems(scale_names)
    yscale_combo.addItems(scale_names)

    grid_check = QtWidgets.QCheckBox(axes_page)
    grid_check.setObjectName("figureComposerToolbarAxesGridCheck")
    grid_check.setText("Show")
    grid_axis_combo = QtWidgets.QComboBox(axes_page)
    grid_axis_combo.setObjectName("figureComposerToolbarAxesGridAxisCombo")
    grid_axis_combo.addItems(["both", "x", "y"])
    grid_which_combo = QtWidgets.QComboBox(axes_page)
    grid_which_combo.setObjectName("figureComposerToolbarAxesGridWhichCombo")
    grid_which_combo.addItems(["major", "minor", "both"])

    title_tooltip = (
        "Set ax.set_title(...) on every selected axis.\n"
        "Use an empty value to clear titles."
    )
    xlabel_tooltip = (
        "Set ax.set_xlabel(...) on every selected axis.\n"
        "Use an empty value to clear x labels."
    )
    ylabel_tooltip = (
        "Set ax.set_ylabel(...) on every selected axis.\n"
        "Use an empty value to clear y labels."
    )
    xlim_tooltip = (
        "Set ax.set_xlim(left, right) on every selected axis.\n"
        "Enter two comma-separated numbers."
    )
    ylim_tooltip = (
        "Set ax.set_ylim(bottom, top) on every selected axis.\n"
        "Enter two comma-separated numbers."
    )
    xscale_tooltip = (
        "Set ax.set_xscale(...) on every selected axis.\n"
        "Choices come from matplotlib.scale.get_scale_names()."
    )
    yscale_tooltip = (
        "Set ax.set_yscale(...) on every selected axis.\n"
        "Choices come from matplotlib.scale.get_scale_names()."
    )
    aspect_tooltip = (
        "Set ax.set_aspect(...) on every selected axis.\n"
        "Use auto, equal, or a numeric aspect ratio."
    )
    grid_check_tooltip = (
        "Turn grid lines on or off for every selected axis,\n"
        "using the Axis and Ticks choices in this row."
    )
    grid_axis_tooltip = "Choose the x, y, or both axes for ax.grid(..., axis=...)."
    grid_which_tooltip = (
        "Choose major ticks, minor ticks, or both for ax.grid(..., which=...)."
    )

    _add_axis_form_row(form_layout, "Title", title_edit, title_tooltip)
    _add_axis_compound_form_row(
        form_layout,
        "Labels",
        "figureComposerToolbarAxesLabelsRow",
        (
            ("x", xlabel_edit, xlabel_tooltip),
            ("y", ylabel_edit, ylabel_tooltip),
        ),
        "Axis label text for every selected axis.",
    )
    _add_axis_compound_form_row(
        form_layout,
        "Limits",
        "figureComposerToolbarAxesLimitsRow",
        (
            ("x", xlim_edit, xlim_tooltip),
            ("y", ylim_edit, ylim_tooltip),
        ),
        "Numeric x/y view limits for every selected axis.",
    )
    _add_axis_compound_form_row(
        form_layout,
        "Scales",
        "figureComposerToolbarAxesScalesRow",
        (
            ("x", xscale_combo, xscale_tooltip),
            ("y", yscale_combo, yscale_tooltip),
        ),
        "Matplotlib x/y scale names for every selected axis.",
    )
    _add_axis_form_row(form_layout, "Aspect", aspect_edit, aspect_tooltip)
    _add_axis_compound_form_row(
        form_layout,
        "Grid",
        "figureComposerToolbarAxesGridRow",
        (
            ("", grid_check, grid_check_tooltip),
            ("Axis", grid_axis_combo, grid_axis_tooltip),
            ("Ticks", grid_which_combo, grid_which_tooltip),
        ),
        "Grid visibility and target tick lines for every selected axis.",
    )

    updating = False
    curve_targets: list[_StyleTarget] = []
    image_targets: list[_StyleTarget] = []

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

    def refresh_style_targets() -> None:
        nonlocal curve_targets, image_targets
        selection = current_selection()
        curve_targets = _curve_style_targets(tool, selection)
        image_targets = _image_style_targets(tool, selection)
        _populate_style_target_combo(curves_combo, curve_targets)
        _populate_style_target_combo(images_combo, image_targets)
        tab_widget.setTabEnabled(curves_tab_index, bool(curve_targets))
        tab_widget.setTabEnabled(images_tab_index, bool(image_targets))
        rebuild_curve_editor()
        rebuild_image_editor()

    def rebuild_curve_editor() -> None:
        _clear_layout(curves_layout)
        target = _current_style_target(curves_combo, curve_targets)
        if target is None:
            _add_style_placeholder(curves_layout, curves_page)
            return
        operation = _operation_by_id(tool, target.operation_id)
        if operation is None:
            _add_style_placeholder(curves_layout, curves_page)
            return
        if target.target_kind == _STYLE_TARGET_PLOT_SLICES:
            editor = _PlotSlicesLineOperationStyleWidget(
                tool,
                operation,
                target.panel_keys,
                _connect_panel_editor_signal,
                curves_page,
            )
            expected_operation = operation

            def apply_panel_line_styles(
                styles: Sequence[FigurePlotSlicesPanelStyleState],
                *,
                operation_id: str = target.operation_id,
                panel_keys: tuple[_PlotSlicesPanelKey, ...] = target.panel_keys,
            ) -> None:
                nonlocal expected_operation
                updated = _update_plot_slices_panel_styles(
                    tool,
                    operation_id,
                    panel_keys,
                    styles,
                    expected_operation=expected_operation,
                )
                if updated is not None:
                    expected_operation = updated

            _connect_panel_editor_signal(
                editor, editor.sigPanelStylesChanged, apply_panel_line_styles
            )

            def apply_plot_slices_line_operation(
                updated: FigureOperationState,
                *,
                operation_id: str = target.operation_id,
            ) -> None:
                nonlocal expected_operation
                _replace_operation_by_id(tool, operation_id, updated)
                expected_operation = updated

            _connect_panel_editor_signal(
                editor, editor.sigOperationChanged, apply_plot_slices_line_operation
            )
        else:
            editor = _LineOperationStyleWidget(tool, operation, curves_page)

            def apply_line_operation(
                updated: FigureOperationState,
                *,
                operation_id: str = target.operation_id,
            ) -> None:
                _replace_operation_by_id(tool, operation_id, updated)

            _connect_panel_editor_signal(
                editor, editor.sigOperationChanged, apply_line_operation
            )
        curves_layout.addWidget(editor)

    def rebuild_image_editor() -> None:
        _clear_layout(images_layout)
        target = _current_style_target(images_combo, image_targets)
        operation = _operation_by_id(tool, target.operation_id) if target else None
        if target is None or operation is None:
            _add_style_placeholder(images_layout, images_page)
            return
        if _is_single_image_plot_slices_target(tool, operation, target):
            editor = _ImageOperationStyleWidget(operation, images_page)

            def apply_image_operation(
                updated: FigureOperationState,
                *,
                operation_id: str = target.operation_id,
            ) -> None:
                _replace_operation_by_id(tool, operation_id, updated)

            _connect_panel_editor_signal(
                editor, editor.sigOperationChanged, apply_image_operation
            )
            images_layout.addWidget(editor)
            return

        editor = _PanelStyleEditorWidget(
            operation,
            target.panel_keys,
            _connect_panel_editor_signal,
            images_page,
        )

        expected_operation = operation

        def apply_panel_image_styles(
            styles: Sequence[FigurePlotSlicesPanelStyleState],
            *,
            operation_id: str = target.operation_id,
            panel_keys: tuple[_PlotSlicesPanelKey, ...] = target.panel_keys,
        ) -> None:
            nonlocal expected_operation
            updated = _update_plot_slices_panel_styles(
                tool,
                operation_id,
                panel_keys,
                styles,
                expected_operation=expected_operation,
            )
            if updated is not None:
                expected_operation = updated

        _connect_panel_editor_signal(
            editor, editor.sigPanelStylesChanged, apply_panel_image_styles
        )
        images_layout.addWidget(editor)

    def refresh_from_axis() -> None:
        nonlocal updating
        axes = _axes_for_selection(tool, current_selection())
        updating = True
        try:
            if not axes:
                unavailable_state = _AxisValueState()
                for widget in (
                    title_edit,
                    xlabel_edit,
                    ylabel_edit,
                    xlim_edit,
                    ylim_edit,
                    aspect_edit,
                ):
                    _apply_axis_line_edit_state(widget, unavailable_state)
                for combo in (xscale_combo, yscale_combo):
                    _apply_axis_combo_state(combo, unavailable_state)
                for combo in (grid_axis_combo, grid_which_combo):
                    combo.setEnabled(False)
                _apply_axis_check_state(grid_check, unavailable_state)
                return
            _apply_axis_line_edit_state(
                title_edit, _axis_value_state(axes, lambda axis: axis.get_title())
            )
            _apply_axis_line_edit_state(
                xlabel_edit, _axis_value_state(axes, lambda axis: axis.get_xlabel())
            )
            _apply_axis_line_edit_state(
                ylabel_edit, _axis_value_state(axes, lambda axis: axis.get_ylabel())
            )
            _apply_axis_line_edit_state(
                xlim_edit,
                _axis_value_state(axes, lambda axis: _float_pair_text(axis.get_xlim())),
            )
            _apply_axis_line_edit_state(
                ylim_edit,
                _axis_value_state(axes, lambda axis: _float_pair_text(axis.get_ylim())),
            )
            _apply_axis_line_edit_state(
                aspect_edit,
                _axis_value_state(axes, lambda axis: _aspect_text(axis.get_aspect())),
            )
            _apply_axis_combo_state(
                xscale_combo,
                _axis_value_state(axes, lambda axis: axis.get_xscale()),
            )
            _apply_axis_combo_state(
                yscale_combo,
                _axis_value_state(axes, lambda axis: axis.get_yscale()),
            )
            for combo in (grid_axis_combo, grid_which_combo):
                combo.setEnabled(True)
            _apply_axis_check_state(
                grid_check,
                _axis_value_state(
                    axes,
                    lambda axis: _axis_grid_visible(
                        axis,
                        which=grid_which_combo.currentText(),
                        axis_name=grid_axis_combo.currentText(),
                    ),
                ),
            )
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
        refresh_style_targets()

    def update_text_method(edit: QtWidgets.QLineEdit, name: str) -> None:
        if updating or _line_edit_mixed_unchanged(edit) or not edit.isModified():
            return
        upsert_axis_method(name, args=(edit.text(),))
        edit.setModified(False)

    def update_limit_method(edit: QtWidgets.QLineEdit, name: str) -> None:
        if updating or _line_edit_mixed_unchanged(edit) or not edit.isModified():
            return
        try:
            limits = _limit_pair_from_text(edit.text())
        except FigureComposerInputError:
            return
        if limits is None:
            return
        upsert_axis_method(name, args=limits)
        edit.setModified(False)

    def update_aspect() -> None:
        if (
            updating
            or _line_edit_mixed_unchanged(aspect_edit)
            or not aspect_edit.isModified()
        ):
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
        if xscale_combo.currentData() is MIXED_VALUE:
            return
        upsert_axis_method("set_xscale", args=(xscale_combo.currentText(),))

    def yscale_activated(_index: int) -> None:
        if yscale_combo.currentData() is MIXED_VALUE:
            return
        upsert_axis_method("set_yscale", args=(yscale_combo.currentText(),))

    def grid_state_changed(state: int) -> None:
        if QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.PartiallyChecked:
            return
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
    grid_check.stateChanged.connect(grid_state_changed)
    grid_axis_combo.activated.connect(grid_combo_activated)
    grid_which_combo.activated.connect(grid_combo_activated)
    curves_combo.activated.connect(lambda _index: rebuild_curve_editor())
    images_combo.activated.connect(lambda _index: rebuild_image_editor())
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


class _LineOperationStyleWidget(QtWidgets.QWidget):
    """Recipe-backed line style editor for Line/Profile operations."""

    sigOperationChanged = QtCore.Signal(object)

    def __init__(
        self,
        tool: FigureComposerTool,
        operation: FigureOperationState,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._operation = operation

        self.color_mode_widget = _LineColorModeWidget(
            tool,
            operation,
            line_kind="profile",
            parent=self,
        )
        self.color_mode_widget.sigOperationChanged.connect(self._set_operation)

        self.colors_widget = _ColorListEditorWidget(operation.line_colors, self)
        self.colors_widget.setObjectName("figureComposerToolbarCurveColorsWidget")
        self.colors_widget.setMainEditObjectName("figureComposerToolbarCurveColorsEdit")
        self.colors_widget.setToolTip(
            "Color for the plotted profiles.\n"
            "Use one color for all profiles or one color per profile."
        )
        self.colors_widget.setVisible(not line_colormap_active(operation))

        self.style_combo = QtWidgets.QComboBox(self)
        self.style_combo.setObjectName("figureComposerToolbarCurveLineStyleCombo")
        configure_style_combo(
            self.style_combo,
            LINE_STYLE_OPTIONS,
            line_kw_style_value(operation, "linestyle", "ls"),
        )
        self.style_combo.setToolTip("Matplotlib line style for these profiles.")

        self.width_spin = optional_positive_spinbox(
            line_kw_float(operation, "linewidth", "lw"),
            parent=self,
        )
        self.width_spin.setObjectName("figureComposerToolbarCurveLineWidthSpin")
        self.width_spin.setToolTip("Matplotlib line width for these profiles.")

        self.marker_combo = QtWidgets.QComboBox(self)
        self.marker_combo.setObjectName("figureComposerToolbarCurveMarkerCombo")
        configure_style_combo(
            self.marker_combo,
            LINE_MARKER_OPTIONS,
            line_kw_style_value(operation, "marker"),
        )
        self.marker_combo.setToolTip("Matplotlib marker style for these profiles.")

        self.marker_size_spin = optional_positive_spinbox(
            line_kw_float(operation, "markersize", "ms"),
            parent=self,
        )
        self.marker_size_spin.setObjectName("figureComposerToolbarCurveMarkerSizeSpin")
        self.marker_size_spin.setToolTip("Matplotlib marker size for these profiles.")

        self.marker_face_edit = _ColorLineEditWidget(
            line_kw_text(operation, "markerfacecolor", "mfc"),
            self,
            inherited_color=operation.line_colors[0] if operation.line_colors else None,
        )
        self.marker_face_edit.setLineEditObjectName(
            "figureComposerToolbarCurveMarkerFaceEdit"
        )
        self.marker_face_edit.setColorButtonObjectName(
            "figureComposerToolbarCurveMarkerFaceButton"
        )
        self.marker_face_edit.setToolTip(
            "Marker face color for these profiles.\n"
            "Leave blank to use Matplotlib defaults."
        )

        self.marker_edge_edit = _ColorLineEditWidget(
            line_kw_text(operation, "markeredgecolor", "mec"),
            self,
            inherited_color=operation.line_colors[0] if operation.line_colors else None,
        )
        self.marker_edge_edit.setLineEditObjectName(
            "figureComposerToolbarCurveMarkerEdgeEdit"
        )
        self.marker_edge_edit.setColorButtonObjectName(
            "figureComposerToolbarCurveMarkerEdgeButton"
        )
        self.marker_edge_edit.setToolTip(
            "Marker edge color for these profiles.\n"
            "Leave blank to use Matplotlib defaults."
        )

        self.extra_kwargs_edit = QtWidgets.QLineEdit(
            _format_dict(extra_line_kw(operation)),
            self,
        )
        self.extra_kwargs_edit.setObjectName("figureComposerToolbarCurveLineKwEdit")
        self.extra_kwargs_edit.setPlaceholderText("optional")
        self.extra_kwargs_edit.setToolTip(
            "Additional Matplotlib Line2D keyword arguments.\n"
            "Controlled style keys above are stored in their dedicated controls."
        )

        layout = QtWidgets.QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )
        layout.addRow("Color lines", self.color_mode_widget)
        layout.addRow("Colors", self.colors_widget)
        self.colors_label = layout.labelForField(self.colors_widget)
        self._sync_color_widgets()
        layout.addRow("Line style", self.style_combo)
        layout.addRow("Line width", self.width_spin)
        layout.addRow("Marker", self.marker_combo)
        layout.addRow("Marker size", self.marker_size_spin)
        layout.addRow("Marker face", self.marker_face_edit)
        layout.addRow("Marker edge", self.marker_edge_edit)
        layout.addRow("Line kwargs", self.extra_kwargs_edit)

        self.colors_widget.colorsChanged.connect(self._colors_changed)
        self.style_combo.activated.connect(self._style_changed)
        self.width_spin.valueChanged.connect(self._width_changed)
        self.marker_combo.activated.connect(self._marker_changed)
        self.marker_size_spin.valueChanged.connect(self._marker_size_changed)
        self.marker_face_edit.editingFinished.connect(self._marker_face_changed)
        self.marker_edge_edit.editingFinished.connect(self._marker_edge_changed)
        self.extra_kwargs_edit.editingFinished.connect(self._extra_kwargs_changed)

    def _set_operation(self, operation: FigureOperationState) -> None:
        self._operation = operation
        self._sync_color_widgets()
        self.sigOperationChanged.emit(operation)

    def _sync_color_widgets(self) -> None:
        visible = not line_colormap_active(self._operation)
        self.colors_widget.setVisible(visible)
        if self.colors_label is not None:
            self.colors_label.setVisible(visible)

    def _colors_changed(self, colors: Sequence[str]) -> None:
        self._set_operation(
            self._operation.model_copy(
                update={"line_color_mode": "manual", "line_colors": tuple(colors)}
            )
        )

    def _style_changed(self, _index: int) -> None:
        self._update_line_kw(
            "linestyle",
            style_combo_value(self.style_combo),
            aliases=("ls",),
        )

    def _width_changed(self, value: float) -> None:
        self._update_line_kw(
            "linewidth",
            optional_positive_spinbox_value(value),
            aliases=("lw",),
        )

    def _marker_changed(self, _index: int) -> None:
        self._update_line_kw("marker", style_combo_value(self.marker_combo))

    def _marker_size_changed(self, value: float) -> None:
        self._update_line_kw(
            "markersize",
            optional_positive_spinbox_value(value),
            aliases=("ms",),
        )

    def _marker_face_changed(self) -> None:
        self._update_line_kw(
            "markerfacecolor",
            color_kw_value_from_text(self.marker_face_edit.text()),
            aliases=("mfc",),
        )

    def _marker_edge_changed(self) -> None:
        self._update_line_kw(
            "markeredgecolor",
            color_kw_value_from_text(self.marker_edge_edit.text()),
            aliases=("mec",),
        )

    def _extra_kwargs_changed(self) -> None:
        line_kw = {
            key: value
            for key, value in self._operation.line_kw.items()
            if key in CONTROLLED_LINE_KW_KEYS
        }
        line_kw.update(_dict_from_text(self.extra_kwargs_edit.text()))
        self._set_operation(self._operation.model_copy(update={"line_kw": line_kw}))

    def _update_line_kw(
        self,
        key: str,
        value: typing.Any,
        *,
        aliases: tuple[str, ...] = (),
    ) -> None:
        line_kw = dict(self._operation.line_kw)
        for candidate in (key, *aliases):
            line_kw.pop(candidate, None)
        if value is not None:
            line_kw[key] = value
        self._set_operation(self._operation.model_copy(update={"line_kw": line_kw}))


class _PlotSlicesLineOperationStyleWidget(QtWidgets.QWidget):
    """Recipe-backed line style editor for 1D plot_slices operations."""

    sigOperationChanged = QtCore.Signal(object)
    sigPanelStylesChanged = QtCore.Signal(object)

    def __init__(
        self,
        tool: FigureComposerTool,
        operation: FigureOperationState,
        panel_keys: tuple[_PlotSlicesPanelKey, ...],
        connect_signal: Callable[
            [QtWidgets.QWidget, typing.Any, Callable[..., None]], None
        ],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._operation = operation

        self.color_mode_widget = _LineColorModeWidget(
            tool,
            operation,
            line_kind="plot_slices",
            parent=self,
        )
        self.panel_editor = _PanelLineStyleEditorWidget(
            operation,
            panel_keys,
            connect_signal,
            self,
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )
        form.addRow("Color lines", self.color_mode_widget)
        layout.addLayout(form)
        layout.addWidget(self.panel_editor)

        connect_signal(
            self,
            self.color_mode_widget.sigOperationChanged,
            self._operation_changed,
        )
        connect_signal(
            self,
            self.panel_editor.sigPanelStylesChanged,
            self.sigPanelStylesChanged.emit,
        )

    def _operation_changed(self, operation: FigureOperationState) -> None:
        self._operation = operation
        self.sigOperationChanged.emit(operation)


class _LineColorModeWidget(QtWidgets.QWidget):
    """Toolbar color-mode controls for multi-line operations."""

    sigOperationChanged = QtCore.Signal(object)

    def __init__(
        self,
        tool: FigureComposerTool,
        operation: FigureOperationState,
        *,
        line_kind: typing.Literal["profile", "plot_slices"],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._tool = tool
        self._operation = operation
        self._line_kind = line_kind
        self._updating = False

        self.mode_combo = QtWidgets.QComboBox(self)
        self.mode_combo.setObjectName("figureComposerToolbarCurveColorModeCombo")
        self.mode_combo.addItem("Manual", "manual")
        self.mode_combo.addItem("By coordinate", "coordinate")
        mode_text = (
            _plot_slices_line_color_mode_text(operation.line_color_mode)
            if line_kind == "plot_slices"
            else _line_color_mode_text(operation.line_color_mode)
        )
        self.mode_combo.setCurrentText(mode_text)
        self.mode_combo.setToolTip(
            "Manual: use the typed line colors.\n"
            "By coordinate: map one numeric coordinate value per line through a "
            "colormap."
        )

        self.coordinate_page = QtWidgets.QWidget(self)
        coordinate_layout = QtWidgets.QVBoxLayout(self.coordinate_page)
        coordinate_layout.setContentsMargins(0, 0, 0, 0)
        coordinate_layout.setSpacing(4)

        self.coord_combo = QtWidgets.QComboBox(self.coordinate_page)
        self.coord_combo.setObjectName("figureComposerToolbarCurveColorCoordCombo")
        self.coord_combo.setToolTip(
            "Numeric scalar coordinate used to color each plotted line."
        )

        cmap_row = QtWidgets.QWidget(self.coordinate_page)
        cmap_layout = QtWidgets.QHBoxLayout(cmap_row)
        cmap_layout.setContentsMargins(0, 0, 0, 0)
        cmap_layout.setSpacing(6)
        self.cmap_combo = erlab.interactive.colors.ColorMapComboBox(cmap_row)
        self.cmap_combo.setObjectName("figureComposerToolbarCurveColorCmapCombo")
        self.cmap_combo.setToolTip("Colormap used for coordinate-colored lines.")
        self.cmap_combo.ensure_populated()
        self.reverse_check = QtWidgets.QCheckBox("Reverse", cmap_row)
        self.reverse_check.setObjectName(
            "figureComposerToolbarCurveColorCmapReverseCheck"
        )
        self.reverse_check.setToolTip("Reverse the selected line colormap.")
        cmap_layout.addWidget(self.cmap_combo, 1)
        cmap_layout.addWidget(self.reverse_check)

        trim_row = QtWidgets.QWidget(self.coordinate_page)
        trim_layout = QtWidgets.QHBoxLayout(trim_row)
        trim_layout.setContentsMargins(0, 0, 0, 0)
        trim_layout.setSpacing(6)
        trim_tooltip = (
            "Skip this fraction at both ends of the colormap.\n"
            "For example, 0.10 samples the middle 80%."
        )
        trim_label = QtWidgets.QLabel("Trim", trim_row)
        trim_label.setToolTip(trim_tooltip)
        self.trim_spin = QtWidgets.QDoubleSpinBox(trim_row)
        self.trim_spin.setObjectName("figureComposerToolbarCurveColorCmapTrimSpin")
        self.trim_spin.setRange(0.0, LINE_COLOR_CMAP_TRIM_MAX)
        self.trim_spin.setDecimals(2)
        self.trim_spin.setSingleStep(0.05)
        self.trim_spin.setKeyboardTracking(False)
        self.trim_spin.setToolTip(trim_tooltip)
        if self.trim_spin.lineEdit() is not None:
            self.trim_spin.lineEdit().setToolTip(trim_tooltip)
        trim_layout.addWidget(trim_label)
        trim_layout.addWidget(self.trim_spin)
        trim_layout.addStretch(1)

        coordinate_layout.addWidget(self.coord_combo)
        coordinate_layout.addWidget(cmap_row)
        coordinate_layout.addWidget(trim_row)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.mode_combo)
        layout.addWidget(self.coordinate_page)

        self.mode_combo.activated.connect(self._mode_changed)
        self.coord_combo.activated.connect(self._coord_changed)
        self.cmap_combo.activated.connect(self._cmap_changed)
        self.reverse_check.stateChanged.connect(self._reverse_changed)
        self.trim_spin.valueChanged.connect(self._trim_changed)
        self._sync_controls()

    def _available_coords(self) -> list[str]:
        if self._line_kind == "plot_slices":
            return _available_plot_slices_line_color_coords(self._tool, self._operation)
        return _available_line_color_coords(self._tool, self._operation)

    def _default_coord(self) -> str | None:
        if self._line_kind == "plot_slices":
            return self._operation.slice_dim
        return self._operation.line_iter_dim

    def _mode_from_text(self, text: str) -> typing.Literal["manual", "coordinate"]:
        data = self.mode_combo.currentData()
        if data in ("manual", "coordinate"):
            return typing.cast("typing.Literal['manual', 'coordinate']", data)
        if self._line_kind == "plot_slices":
            return _plot_slices_line_color_mode_from_text(text)
        return _line_color_mode_from_text(text)

    def _set_operation(self, operation: FigureOperationState) -> None:
        self._operation = operation
        self.sigOperationChanged.emit(operation)
        self._sync_controls()

    def _sync_controls(self) -> None:
        self._updating = True
        try:
            with QtCore.QSignalBlocker(self.mode_combo):
                self.mode_combo.setCurrentText(
                    _plot_slices_line_color_mode_text(self._operation.line_color_mode)
                    if self._line_kind == "plot_slices"
                    else _line_color_mode_text(self._operation.line_color_mode)
                )
            active = line_colormap_active(self._operation)
            self.coordinate_page.setVisible(active)

            coords = self._available_coords()
            selected = effective_line_color_coord(
                self._operation, self._default_coord()
            )
            with QtCore.QSignalBlocker(self.coord_combo):
                self.coord_combo.clear()
                for coord in coords:
                    self.coord_combo.addItem(coord, coord)
                if selected and self.coord_combo.findData(selected) < 0:
                    self.coord_combo.addItem(selected, selected)
                if selected:
                    self.coord_combo.setCurrentIndex(
                        self.coord_combo.findData(selected)
                    )
                elif self.coord_combo.count():
                    self.coord_combo.setCurrentIndex(0)
            self.coord_combo.setEnabled(self.coord_combo.count() > 0)

            cmap_base, cmap_reverse = _cmap_base_and_reverse(
                self._operation.line_color_cmap
            )
            with QtCore.QSignalBlocker(self.cmap_combo):
                self.cmap_combo.setCurrentText(cmap_base)
            with QtCore.QSignalBlocker(self.reverse_check):
                self.reverse_check.setChecked(
                    self._operation.line_color_cmap_reverse or cmap_reverse
                )
            with QtCore.QSignalBlocker(self.trim_spin):
                self.trim_spin.setValue(
                    line_color_cmap_trim_control_value(self._operation)
                )
        finally:
            self._updating = False

    def _mode_changed(self, _index: int) -> None:
        if self._updating:
            return
        mode = self._mode_from_text(self.mode_combo.currentText())
        updates: dict[str, typing.Any] = {"line_color_mode": mode}
        if mode == "coordinate":
            coord = self.coord_combo.currentData()
            updates["line_color_coord"] = (
                coord if isinstance(coord, str) and coord else self._default_coord()
            )
        self._set_operation(self._operation.model_copy(update=updates))

    def _coord_changed(self, _index: int) -> None:
        if self._updating:
            return
        coord = self.coord_combo.currentData()
        self._set_operation(
            self._operation.model_copy(
                update={"line_color_coord": coord if isinstance(coord, str) else None}
            )
        )

    def _cmap_changed(self, _index: int) -> None:
        if self._updating:
            return
        self._set_operation(
            self._operation.model_copy(
                update={
                    "line_color_cmap": _cmap_with_reverse(
                        self.cmap_combo.currentText(), False
                    ),
                    "line_color_cmap_reverse": self.reverse_check.isChecked(),
                }
            )
        )

    def _reverse_changed(self, state: int) -> None:
        if self._updating:
            return
        self._set_operation(
            self._operation.model_copy(
                update={
                    "line_color_cmap": _cmap_with_reverse(
                        self.cmap_combo.currentText(), False
                    ),
                    "line_color_cmap_reverse": (
                        QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked
                    ),
                }
            )
        )

    def _trim_changed(self, value: float) -> None:
        if self._updating:
            return
        self._set_operation(
            self._operation.model_copy(update={"line_color_cmap_trim": value})
        )


class _ImageOperationStyleWidget(QtWidgets.QWidget):
    """Recipe-backed image style editor for a single image plot."""

    sigOperationChanged = QtCore.Signal(object)

    def __init__(
        self,
        operation: FigureOperationState,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._operation = operation
        self._updating = False

        self.cmap_combo = erlab.interactive.colors.ColorMapComboBox(self)
        self.cmap_combo.setObjectName("figureComposerPanelCmapCombo")
        self.cmap_combo.setToolTip("Colormap stored on this image step.")
        self.cmap_combo.ensure_populated()
        self.cmap_reverse_check = QtWidgets.QCheckBox("Reverse", self)
        self.cmap_reverse_check.setObjectName("figureComposerPanelCmapReverseCheck")
        self.cmap_reverse_check.setToolTip("Append _r to this image colormap.")

        self.norm_combo = QtWidgets.QComboBox(self)
        self.norm_combo.setObjectName("figureComposerPanelNormCombo")
        self.norm_combo.addItems(list(_NORM_CHOICES))
        self.norm_combo.setToolTip("Normalization class stored on this image step.")

        self.gamma_edit = self._number_edit("figureComposerPanelGammaEdit")
        self.vmin_edit = self._number_edit("figureComposerPanelVminEdit")
        self.vmax_edit = self._number_edit("figureComposerPanelVmaxEdit")
        self.vcenter_edit = self._number_edit("figureComposerPanelVcenterEdit")
        self.halfrange_edit = self._number_edit("figureComposerPanelHalfrangeEdit")
        self.clip_combo = QtWidgets.QComboBox(self)
        self.clip_combo.setObjectName("figureComposerPanelClipCombo")
        self.clip_combo.addItems(["default", "False", "True"])
        self.clip_combo.setToolTip("Norm clip argument for this image step.")
        self.norm_kwargs_edit = QtWidgets.QLineEdit(self)
        self.norm_kwargs_edit.setObjectName("figureComposerPanelNormKwargsEdit")
        self.norm_kwargs_edit.setToolTip(
            "Extra keyword arguments for this image step's norm constructor."
        )

        layout = QtWidgets.QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )
        cmap_row = QtWidgets.QWidget(self)
        cmap_layout = QtWidgets.QHBoxLayout(cmap_row)
        cmap_layout.setContentsMargins(0, 0, 0, 0)
        cmap_layout.setSpacing(6)
        cmap_layout.addWidget(self.cmap_combo, 1)
        cmap_layout.addWidget(self.cmap_reverse_check)
        layout.addRow("Colormap", cmap_row)
        layout.addRow("Norm", self.norm_combo)
        for label, widget in (
            ("Gamma", self.gamma_edit),
            ("vmin", self.vmin_edit),
            ("vmax", self.vmax_edit),
            ("vcenter", self.vcenter_edit),
            ("halfrange", self.halfrange_edit),
            ("Clip", self.clip_combo),
            ("Norm kwargs", self.norm_kwargs_edit),
        ):
            layout.addRow(label, widget)

        self.cmap_combo.activated.connect(self._cmap_changed)
        self.cmap_reverse_check.stateChanged.connect(self._cmap_reverse_changed)
        self.norm_combo.activated.connect(self._norm_changed)
        for attr, edit in (
            ("norm_gamma", self.gamma_edit),
            ("vmin", self.vmin_edit),
            ("vmax", self.vmax_edit),
            ("vcenter", self.vcenter_edit),
            ("halfrange", self.halfrange_edit),
        ):
            edit.editingFinished.connect(
                lambda attr=attr, edit=edit: self._number_changed(attr, edit)
            )
        self.clip_combo.activated.connect(self._clip_changed)
        self.norm_kwargs_edit.editingFinished.connect(self._norm_kwargs_changed)
        self._sync_controls()

    @staticmethod
    def _number_edit(object_name: str) -> QtWidgets.QLineEdit:
        edit = QtWidgets.QLineEdit()
        edit.setObjectName(object_name)
        edit.setToolTip("Leave blank to use the default value.")
        return edit

    def _set_operation(self, operation: FigureOperationState) -> None:
        self._operation = operation
        self.sigOperationChanged.emit(operation)
        self._sync_controls()

    def _sync_controls(self) -> None:
        self._updating = True
        try:
            cmap_base, cmap_reverse = _cmap_base_and_reverse(self._operation.cmap)
            with QtCore.QSignalBlocker(self.cmap_combo):
                self.cmap_combo.setCurrentText(cmap_base)
            with QtCore.QSignalBlocker(self.cmap_reverse_check):
                self.cmap_reverse_check.setChecked(cmap_reverse)
            norm_name = _effective_norm_name(self._operation.norm_name)
            with QtCore.QSignalBlocker(self.norm_combo):
                self.norm_combo.setCurrentText(norm_name)
            self._sync_norm_fields(norm_name)
        finally:
            self._updating = False

    def _sync_norm_fields(self, norm_name: str) -> None:
        norm_fields = set(_norm_kwarg_fields(norm_name))
        for attr, edit in (
            ("norm_gamma", self.gamma_edit),
            ("vmin", self.vmin_edit),
            ("vmax", self.vmax_edit),
            ("vcenter", self.vcenter_edit),
            ("halfrange", self.halfrange_edit),
        ):
            field_name = "gamma" if attr == "norm_gamma" else attr
            edit.setEnabled(field_name in norm_fields)
            value = getattr(self._operation, attr)
            with QtCore.QSignalBlocker(edit):
                edit.setText("" if value is None else f"{value:g}")
                edit.setModified(False)
        self.clip_combo.setEnabled("clip" in norm_fields)
        with QtCore.QSignalBlocker(self.clip_combo):
            self.clip_combo.setCurrentText(_norm_clip_text(self._operation.norm_clip))
        self.norm_kwargs_edit.setEnabled(True)
        with QtCore.QSignalBlocker(self.norm_kwargs_edit):
            self.norm_kwargs_edit.setText(_format_dict(self._operation.norm_kwargs))
            self.norm_kwargs_edit.setModified(False)

    def _cmap_changed(self, _index: int) -> None:
        if self._updating:
            return
        self._set_operation(
            self._operation.model_copy(
                update={
                    "cmap": _cmap_with_reverse(
                        self.cmap_combo.currentText(),
                        self.cmap_reverse_check.isChecked(),
                    )
                }
            )
        )

    def _cmap_reverse_changed(self, state: int) -> None:
        if self._updating:
            return
        self._set_operation(
            self._operation.model_copy(
                update={
                    "cmap": _cmap_with_reverse(
                        self.cmap_combo.currentText(),
                        QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked,
                    )
                }
            )
        )

    def _norm_changed(self, _index: int) -> None:
        if self._updating:
            return
        self._set_operation(
            self._operation.model_copy(
                update={"norm_name": self.norm_combo.currentText()}
            )
        )

    def _number_changed(self, attr: str, edit: QtWidgets.QLineEdit) -> None:
        if self._updating:
            return
        text = edit.text().strip()
        self._set_operation(
            self._operation.model_copy(update={attr: float(text) if text else None})
        )

    def _clip_changed(self, _index: int) -> None:
        if self._updating:
            return
        self._set_operation(
            self._operation.model_copy(
                update={
                    "norm_clip": _norm_clip_from_text(self.clip_combo.currentText())
                }
            )
        )

    def _norm_kwargs_changed(self) -> None:
        if self._updating:
            return
        self._set_operation(
            self._operation.model_copy(
                update={"norm_kwargs": _dict_from_text(self.norm_kwargs_edit.text())}
            )
        )


def _style_tab_page(
    parent: QtWidgets.QWidget,
    *,
    combo_object_name: str,
) -> tuple[QtWidgets.QWidget, QtWidgets.QComboBox, QtWidgets.QVBoxLayout]:
    page = QtWidgets.QWidget(parent)
    layout = QtWidgets.QVBoxLayout(page)
    combo = _ElidingComboBox(page)
    combo.setObjectName(combo_object_name)
    combo.setToolTip("Choose the plotted item to customize.")
    combo.currentIndexChanged.connect(
        lambda _index, combo=combo: _update_style_target_combo_tooltip(combo)
    )
    layout.addWidget(combo)
    editor_layout = QtWidgets.QVBoxLayout()
    editor_layout.setContentsMargins(0, 0, 0, 0)
    layout.addLayout(editor_layout)
    layout.addStretch(1)
    return page, combo, editor_layout


def _connect_panel_editor_signal(
    owner: QtWidgets.QWidget,
    signal: typing.Any,
    slot: Callable[..., None],
) -> None:
    owner_ref = weakref.ref(owner)

    def guarded_slot(*args: typing.Any) -> None:
        owner_widget = owner_ref()
        if owner_widget is None or not erlab.interactive.utils.qt_is_valid(
            owner_widget
        ):
            return
        slot(*args)

    signal.connect(guarded_slot)


def _clear_layout(layout: QtWidgets.QLayout) -> None:
    while (item := layout.takeAt(0)) is not None:
        widget = item.widget()
        if widget is not None:
            widget.setParent(None)
            widget.deleteLater()
        child_layout = item.layout()
        if child_layout is not None:
            _clear_layout(child_layout)


def _add_style_placeholder(
    layout: QtWidgets.QVBoxLayout, parent: QtWidgets.QWidget
) -> None:
    label = QtWidgets.QLabel("No matching plotted items on the selected axes.", parent)
    label.setObjectName("figureComposerToolbarStylePlaceholder")
    label.setWordWrap(True)
    layout.addWidget(label)


def _populate_style_target_combo(
    combo: QtWidgets.QComboBox, targets: Sequence[_StyleTarget]
) -> None:
    with QtCore.QSignalBlocker(combo):
        combo.clear()
        for index, target in enumerate(targets):
            combo.addItem(target.label, index)
            combo.setItemData(
                index,
                target.tooltip or target.label,
                QtCore.Qt.ItemDataRole.ToolTipRole,
            )
        combo.setEnabled(bool(targets))
        _update_style_target_combo_tooltip(combo)


def _update_style_target_combo_tooltip(combo: QtWidgets.QComboBox) -> None:
    tooltip = combo.itemData(combo.currentIndex(), QtCore.Qt.ItemDataRole.ToolTipRole)
    combo.setToolTip(
        tooltip
        if isinstance(tooltip, str) and tooltip
        else "Choose the plotted item to customize."
    )


def _current_style_target(
    combo: QtWidgets.QComboBox, targets: Sequence[_StyleTarget]
) -> _StyleTarget | None:
    index = combo.currentData()
    if isinstance(index, int) and 0 <= index < len(targets):
        return targets[index]
    return None


def _curve_style_targets(
    tool: FigureComposerTool, selection: FigureAxesSelectionState
) -> list[_StyleTarget]:
    selected_axis_ids = _axis_identity_set(tool, selection)
    if not selected_axis_ids:
        return []
    targets: list[_StyleTarget] = []
    for index, operation in enumerate(tool._recipe.operations):
        if not operation.enabled:
            continue
        if operation.kind == FigureOperationKind.LINE:
            if _operation_hits_axes(tool, operation, selected_axis_ids):
                targets.append(
                    _StyleTarget(
                        operation.operation_id,
                        index,
                        _STYLE_TARGET_LINE,
                        tool._operation_display_text(operation),
                    )
                )
            continue
        if operation.kind != FigureOperationKind.PLOT_SLICES:
            continue
        if _plot_slices_panel_kind(_plot_slices_shape(tool, operation)) != (
            _PLOT_SLICES_PANEL_LINE
        ):
            continue
        panel_keys = _selected_plot_slices_panel_keys(
            tool, operation, selected_axis_ids
        )
        if panel_keys:
            targets.append(
                _StyleTarget(
                    operation.operation_id,
                    index,
                    _STYLE_TARGET_PLOT_SLICES,
                    _style_target_label(tool, operation, panel_keys),
                    panel_keys,
                )
            )
    return targets


def _image_style_targets(
    tool: FigureComposerTool, selection: FigureAxesSelectionState
) -> list[_StyleTarget]:
    selected_axis_ids = _axis_identity_set(tool, selection)
    if not selected_axis_ids:
        return []
    targets: list[_StyleTarget] = []
    for index, operation in enumerate(tool._recipe.operations):
        if (
            not operation.enabled
            or operation.kind != FigureOperationKind.PLOT_SLICES
            or _plot_slices_panel_kind(_plot_slices_shape(tool, operation))
            != _PLOT_SLICES_PANEL_IMAGE
        ):
            continue
        panel_keys = _selected_plot_slices_panel_keys(
            tool, operation, selected_axis_ids
        )
        if panel_keys:
            full_label = _style_target_label(tool, operation, panel_keys)
            targets.append(
                _StyleTarget(
                    operation.operation_id,
                    index,
                    _STYLE_TARGET_PLOT_SLICES,
                    _image_style_target_label(index, operation, panel_keys),
                    panel_keys,
                    full_label,
                )
            )
    return targets


def _axis_identity_set(
    tool: FigureComposerTool, selection: FigureAxesSelectionState
) -> set[int]:
    return {id(axis) for axis in _axes_for_selection(tool, selection)}


def _operation_hits_axes(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    selected_axis_ids: set[int],
) -> bool:
    operation_axis_ids = {
        id(axis) for axis in _axes_for_selection(tool, operation.axes)
    }
    return bool(selected_axis_ids & operation_axis_ids)


def _selected_plot_slices_panel_keys(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    selected_axis_ids: set[int],
) -> tuple[_PlotSlicesPanelKey, ...]:
    operation_axes = _axes_for_selection(tool, operation.axes)
    operation_axis_ids = {id(axis) for axis in operation_axes}
    if not (selected_axis_ids & operation_axis_ids):
        return ()
    panel_keys = _plot_slices_panel_keys(tool, operation)
    if len(operation_axes) != len(panel_keys):
        return panel_keys
    return tuple(
        key
        for key, axis in zip(panel_keys, operation_axes, strict=True)
        if id(axis) in selected_axis_ids
    )


def _style_target_label(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    panel_keys: Sequence[_PlotSlicesPanelKey],
) -> str:
    text = tool._operation_display_text(operation)
    if len(panel_keys) == 1:
        return f"{text}: {panel_keys[0].label}"
    return f"{text}: {len(panel_keys)} panels"


def _image_style_target_label(
    operation_index: int,
    operation: FigureOperationState,
    panel_keys: Sequence[_PlotSlicesPanelKey],
) -> str:
    label = operation.label.strip() or "Image slices"
    if label == "plot_slices":
        label = "Image slices"
    text = f"Step {operation_index + 1}: {label}"
    if len(panel_keys) == 1:
        key = panel_keys[0]
        panel_text = (
            f"panel {key.map_index + 1}"
            if key.slice_index == 0
            else f"panel {key.map_index + 1}.{key.slice_index + 1}"
        )
        return f"{text}: {panel_text}"
    return f"{text}: {len(panel_keys)} panels"


def _operation_by_id(
    tool: FigureComposerTool, operation_id: str
) -> FigureOperationState | None:
    for operation in tool._recipe.operations:
        if operation.operation_id == operation_id:
            return operation
    return None


def _replace_operation_by_id(
    tool: FigureComposerTool,
    operation_id: str,
    updated: FigureOperationState,
    *,
    rebuild_editor: bool = False,
) -> None:
    for index, operation in enumerate(tool._recipe.operations):
        if operation.operation_id == operation_id:
            _replace_recipe_operation(
                tool,
                index,
                updated,
                rebuild_editor=rebuild_editor,
            )
            return


def _is_single_image_plot_slices_target(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    target: _StyleTarget,
) -> bool:
    if (
        operation.kind != FigureOperationKind.PLOT_SLICES
        or _plot_slices_panel_kind(_plot_slices_shape(tool, operation))
        != _PLOT_SLICES_PANEL_IMAGE
    ):
        return False
    return (
        len(_plot_slices_panel_keys(tool, operation)) == 1
        and len(target.panel_keys) == 1
    )


def _update_plot_slices_panel_styles(
    tool: FigureComposerTool,
    operation_id: str,
    edited_panel_keys: Sequence[_PlotSlicesPanelKey],
    styles: Sequence[FigurePlotSlicesPanelStyleState],
    *,
    expected_operation: FigureOperationState | None = None,
) -> FigureOperationState | None:
    operation = _operation_by_id(tool, operation_id)
    if operation is None or operation.kind != FigureOperationKind.PLOT_SLICES:
        return None
    if expected_operation is not None and operation != expected_operation:
        return None
    edited_keys = {(key.map_index, key.slice_index) for key in edited_panel_keys}
    incoming_styles = tuple(styles)
    merged = (
        *(
            style
            for style in operation.panel_styles
            if (style.map_index, style.slice_index) not in edited_keys
        ),
        *incoming_styles,
    )
    merged = tuple(
        sorted(merged, key=lambda style: (style.map_index, style.slice_index))
    )
    updated = operation.model_copy(
        update={"panel_styles_enabled": bool(merged), "panel_styles": merged}
    )
    _replace_operation_by_id(
        tool,
        operation_id,
        updated,
        rebuild_editor=_current_operation_id(tool) == operation_id,
    )
    return updated


def _replace_recipe_operation(
    tool: FigureComposerTool,
    index: int,
    operation: FigureOperationState,
    *,
    rebuild_editor: bool = False,
) -> None:
    if index < 0 or index >= len(tool._recipe.operations):
        return
    current = tool._current_operation()
    current_id = current[1].operation_id if current is not None else None
    selected_ids = tool._selected_operation_ids()
    operations = list(tool._recipe.operations)
    operations[index] = operation
    _set_operations(
        tool,
        tuple(operations),
        current_id,
        selected_ids,
        rebuild_editor=rebuild_editor,
    )


def _current_operation_id(tool: FigureComposerTool) -> str | None:
    current = tool._current_operation()
    return current[1].operation_id if current is not None else None


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
    *,
    rebuild_editor: bool = True,
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
    if rebuild_editor:
        tool._update_operation_editor_safely()
    _render_preview(tool)
    tool.sigInfoChanged.emit()
    tool._write_state()


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


def _layout_engine_text(tool: FigureComposerTool) -> str:
    return tool._recipe.setup.layout or "default"


def _set_setup_layout_engine(tool: FigureComposerTool, text: str) -> None:
    with QtCore.QSignalBlocker(tool.layout_combo):
        tool.layout_combo.setCurrentText(text)
    tool._setup_controls_changed()


def _float_pair_text(values: Sequence[float]) -> str:
    if len(values) < 2:
        return ""
    return f"{float(values[0]):g}, {float(values[1]):g}"


def _float_pair_from_text(text: str) -> tuple[float, float] | None:
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) != 2:
        return None
    return float(parts[0]), float(parts[1])


def _axis_value_state(
    axes: Sequence[Axes],
    value_getter: Callable[[Axes], typing.Any],
) -> _AxisValueState:
    if not axes:
        return _AxisValueState()
    values = tuple(value_getter(axis) for axis in axes)
    if any(value is None for value in values):
        return _AxisValueState(mixed=True, available=True)
    first = values[0]
    if all(value == first for value in values):
        return _AxisValueState(value=first, available=True)
    return _AxisValueState(mixed=True, available=True)


def _apply_axis_line_edit_state(
    edit: QtWidgets.QLineEdit, state: _AxisValueState
) -> None:
    with QtCore.QSignalBlocker(edit):
        edit.setEnabled(state.available)
        edit.setProperty("batch_mixed", state.mixed)
        if not state.available:
            edit.clear()
            edit.setPlaceholderText("")
        elif state.mixed:
            edit.clear()
            edit.setPlaceholderText(MIXED_VALUES_TEXT)
        else:
            edit.setPlaceholderText("")
            edit.setText(str(state.value))
        edit.setModified(False)


def _line_edit_mixed_unchanged(edit: QtWidgets.QLineEdit) -> bool:
    return bool(edit.property("batch_mixed")) and not edit.isModified()


def _apply_axis_combo_state(combo: QtWidgets.QComboBox, state: _AxisValueState) -> None:
    with QtCore.QSignalBlocker(combo):
        _remove_mixed_combo_placeholder(combo)
        combo.setEnabled(state.available)
        if not state.available:
            return
        if state.mixed:
            combo.insertItem(0, MIXED_VALUES_TEXT, MIXED_VALUE)
            item = typing.cast("typing.Any", combo.model()).item(0)
            if item is not None:
                item.setEnabled(False)
            combo.setCurrentIndex(0)
        else:
            combo.setCurrentText(str(state.value))


def _remove_mixed_combo_placeholder(combo: QtWidgets.QComboBox) -> None:
    for index in reversed(range(combo.count())):
        if combo.itemData(index) is MIXED_VALUE:
            combo.removeItem(index)


def _apply_axis_check_state(check: QtWidgets.QCheckBox, state: _AxisValueState) -> None:
    with QtCore.QSignalBlocker(check):
        check.setEnabled(state.available)
        if not state.available:
            check.setTristate(False)
            check.setChecked(False)
        elif state.mixed:
            check.setTristate(True)
            check.setCheckState(QtCore.Qt.CheckState.PartiallyChecked)
        else:
            check.setTristate(False)
            check.setChecked(bool(state.value))


def _axis_grid_visible(axis: Axes, *, which: str, axis_name: str) -> bool | None:
    states: list[bool | None] = []
    if axis_name in {"x", "both"}:
        states.append(_axis_direction_grid_visible(axis.xaxis, which))
    if axis_name in {"y", "both"}:
        states.append(_axis_direction_grid_visible(axis.yaxis, which))
    return _merge_grid_states(states)


def _axis_direction_grid_visible(axis: typing.Any, which: str) -> bool | None:
    states: list[bool] = []
    if which in {"major", "both"}:
        states.append(_gridlines_visible(axis.get_gridlines()))
    if which in {"minor", "both"}:
        states.append(
            _gridlines_visible(tick.gridline for tick in axis.get_minor_ticks())
        )
    return _merge_grid_states(states)


def _gridlines_visible(lines: Iterable[typing.Any]) -> bool:
    return any(line.get_visible() for line in lines)


def _merge_grid_states(states: Sequence[bool | None]) -> bool | None:
    if not states or any(state is None for state in states):
        return None
    first = states[0]
    if all(state == first for state in states):
        return first
    return None


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
            axes_id: _gridspec_axis_display_name(
                setup, axes_id, reserved_names=tool._source_names()
            )
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

    def grow_subplot_grid(direction: typing.Literal["row", "column"]) -> None:
        selected = selector.selected_axes()
        if not tool._grow_subplot_grid(direction):
            return
        updated_setup = tool._recipe.setup
        selector.set_grid(updated_setup.nrows, updated_setup.ncols)
        selector.set_selected_axes(selected or ((0, 0),), emit=True)

    selector.sigAddRowRequested.connect(functools.partial(grow_subplot_grid, "row"))
    selector.sigAddColumnRequested.connect(
        functools.partial(grow_subplot_grid, "column")
    )
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
