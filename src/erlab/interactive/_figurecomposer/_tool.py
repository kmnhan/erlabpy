"""Manager-facing Figure Composer tool window."""

from __future__ import annotations

import math
import textwrap
import typing
import uuid

# Matplotlib's Qt backend should see the qtpy-selected binding first.
# isort: off
from qtpy import QtCore, QtGui, QtWidgets

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
# isort: on

import erlab
from erlab.interactive._figurecomposer import _codegen, _rendering
from erlab.interactive._figurecomposer._axes import _all_axes
from erlab.interactive._figurecomposer._defaults import (
    _MM_PER_INCH,
    _figure_draw_context,
    _figure_style_context,
)
from erlab.interactive._figurecomposer._operations import _registry
from erlab.interactive._figurecomposer._operations._base import (
    COMMON_AXES_SECTION_TOOLTIP,
    COMMON_SOURCE_SECTION_TOOLTIP,
    StepSection,
)
from erlab.interactive._figurecomposer._sources import (
    _default_plot_operation,
    _default_setup_for_data,
    _source_label,
    _source_name,
)
from erlab.interactive._figurecomposer._state import (
    FigureAxesSelectionState,
    FigureOperationState,
    FigureRecipeState,
    FigureSourceState,
    FigureSubplotsState,
)
from erlab.interactive._figurecomposer._text import (
    _format_axes_tuple,
    _format_tuple,
    _literal_sequence_from_text,
)
from erlab.interactive._figurecomposer._widgets import (
    _AxesSelectorWidget,
    _FigureComposerDisplayWindow,
    _step_toolbar_button,
)
from erlab.interactive.imagetool import provenance

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import xarray as xr
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


class FigureComposerTool(erlab.interactive.utils.ToolWindow[FigureRecipeState]):
    """Editable Matplotlib figure recipe window."""

    tool_name = "fig"
    manager_collection = "figures"

    StateModel = FigureRecipeState

    def __init__(
        self,
        data: xr.DataArray,
        *,
        recipe: FigureRecipeState | None = None,
        source_data: Mapping[str, xr.DataArray] | None = None,
    ) -> None:
        super().__init__()
        self._updating_controls = False
        self._rendering = False
        self._source_data: dict[str, xr.DataArray] = {}
        self._recipe = recipe or self._default_recipe(data)
        self._figure_window: _FigureComposerDisplayWindow | None = None

        if source_data is not None:
            self.set_source_data(source_data)
        elif self._recipe.primary_source in {
            source.name for source in self._recipe.sources
        }:
            self._source_data[self._recipe.primary_source] = data
        else:
            source_name = _source_name(data)
            self._source_data[source_name] = data

        if self._recipe.primary_source not in self._source_data:
            self._source_data[self._recipe.primary_source] = data

        self._current_step_section_key = "sources"
        self._build_ui()
        self._apply_recipe_to_controls()
        _rendering._render_preview(self, show_window=False)

    @staticmethod
    def _default_recipe(data: xr.DataArray) -> FigureRecipeState:
        source_name = _source_name(data)
        setup = _default_setup_for_data(data)
        source = FigureSourceState(name=source_name, label=_source_label(data))
        return FigureRecipeState(
            setup=setup,
            sources=(source,),
            operations=(_default_plot_operation(source_name, data, setup=setup),),
            primary_source=source_name,
        )

    @classmethod
    def from_sources(
        cls,
        source_data: Mapping[str, xr.DataArray],
        *,
        sources: tuple[FigureSourceState, ...],
        operations: tuple[FigureOperationState, ...] | None = None,
        setup: FigureSubplotsState | None = None,
        primary_source: str | None = None,
    ) -> FigureComposerTool:
        if not source_data:
            raise ValueError("At least one source is required")
        primary = primary_source or next(iter(source_data))
        resolved_setup = setup or FigureSubplotsState()
        recipe = FigureRecipeState(
            setup=resolved_setup,
            sources=sources,
            operations=operations
            if operations is not None
            else tuple(
                _default_plot_operation(name, data, setup=resolved_setup)
                for name, data in source_data.items()
            ),
            primary_source=primary,
        )
        return cls(source_data[primary], recipe=recipe, source_data=source_data)

    @property
    def figure_window(self) -> _FigureComposerDisplayWindow:
        return self._ensure_figure_window()

    @property
    def figure(self) -> Figure:
        return self.figure_window.figure

    @property
    def canvas(self) -> FigureCanvas:
        return self.figure_window.canvas

    def _ensure_figure_window(self) -> _FigureComposerDisplayWindow:
        if self._figure_window is None or not erlab.interactive.utils.qt_is_valid(
            self._figure_window
        ):
            self._figure_window = _FigureComposerDisplayWindow(self._recipe.setup)
            self._figure_window.sigCanvasSizeChanged.connect(
                self._figure_window_canvas_size_changed
            )
        self._figure_window.setWindowTitle(self._figure_window_title())
        return self._figure_window

    def _figure_window_title(self) -> str:
        display_name = self._tool_display_name
        if display_name:
            return f"{self.tool_name}: {display_name}"
        title = self.windowTitle()
        return title if title and title != self.tool_name else "Figure"

    @QtCore.Slot()
    def _show_figure_window_requested(self) -> None:
        self.show_figure_window(activate=True)

    def show_figure_window(self, *, activate: bool = True) -> None:
        self.figure_window.show_for_setup(
            self._recipe.setup, self._figure_window_title(), activate=activate
        )
        self.canvas.flush_events()
        self._sync_recipe_figsize_to_canvas(draw=False, emit_info=False)
        _rendering._render_preview(self, show_window=False)
        self.canvas.draw()
        self.canvas.flush_events()

    @QtCore.Slot(float, float)
    def _figure_window_canvas_size_changed(
        self, width_inches: float, height_inches: float
    ) -> None:
        if self._updating_controls:
            return
        self._set_recipe_figsize_from_canvas(
            width_inches, height_inches, draw=True, emit_info=True
        )

    def _sync_recipe_figsize_to_canvas(self, *, draw: bool, emit_info: bool) -> bool:
        canvas_size = self.canvas.size()
        if canvas_size.isEmpty():
            return False
        dpi = float(typing.cast("typing.Any", self.figure)._original_dpi)
        if dpi <= 0.0:
            return False
        return self._set_recipe_figsize_from_canvas(
            canvas_size.width() / dpi,
            canvas_size.height() / dpi,
            draw=draw,
            emit_info=emit_info,
        )

    def _set_recipe_figsize_from_canvas(
        self,
        width_inches: float,
        height_inches: float,
        *,
        draw: bool,
        emit_info: bool,
    ) -> bool:
        setup = self._recipe.setup
        if math.isclose(width_inches, setup.figsize[0], abs_tol=0.005) and math.isclose(
            height_inches, setup.figsize[1], abs_tol=0.005
        ):
            return False
        figsize = (round(width_inches, 4), round(height_inches, 4))
        self._recipe = self._recipe.model_copy(
            update={"setup": setup.model_copy(update={"figsize": figsize})}
        )
        self.figure.set_size_inches(figsize, forward=False)
        self._updating_controls = True
        try:
            self.width_spin.setValue(figsize[0])
            self.height_spin.setValue(figsize[1])
            self._sync_size_mm_controls(figsize)
        finally:
            self._updating_controls = False
        if draw:
            self.canvas.draw()
        if emit_info:
            self.sigInfoChanged.emit()
        return True

    def _hide_figure_window(self) -> None:
        if self._figure_window is not None and erlab.interactive.utils.qt_is_valid(
            self._figure_window
        ):
            self._figure_window.hide()

    def _close_figure_window(self) -> None:
        if self._figure_window is None or not erlab.interactive.utils.qt_is_valid(
            self._figure_window
        ):
            self._figure_window = None
            return
        window = self._figure_window
        self._figure_window = None
        window.close_from_owner()
        window.deleteLater()

    def showEvent(self, event: QtGui.QShowEvent | None) -> None:
        if event is not None:
            super().showEvent(event)
        self.show_figure_window(activate=False)

    def hideEvent(self, event: QtGui.QHideEvent | None) -> None:
        self._hide_figure_window()
        if event is not None:
            super().hideEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:
        self._close_figure_window()
        if event is not None:
            super().closeEvent(event)

    def _build_ui(self) -> None:
        root = QtWidgets.QWidget(self)
        root_layout = QtWidgets.QVBoxLayout(root)
        root_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.setSpacing(4)

        self.editor_tabs = QtWidgets.QTabWidget(root)
        self.editor_tabs.setObjectName("figureComposerEditorTabs")

        recipe_page = QtWidgets.QWidget(self.editor_tabs)
        recipe_page.setObjectName("figureComposerRecipePage")
        self.recipe_page = recipe_page
        recipe_layout = QtWidgets.QVBoxLayout(recipe_page)
        recipe_layout.setContentsMargins(6, 6, 6, 6)
        recipe_layout.setSpacing(4)

        action_layout = QtWidgets.QVBoxLayout()
        action_layout.setSpacing(2)

        self.add_step_button = _step_toolbar_button(
            recipe_page,
            "figureComposerAddStepButton",
            "Add Step ▾",
            "Add a plotting, ERLab method, Axes method, or Python step.",
        )
        self.add_step_button.setProperty("uses_inline_menu_arrow", True)
        self.add_step_menu = QtWidgets.QMenu(self.add_step_button)
        self.add_step_menu.setObjectName("figureComposerAddStepMenu")
        for action_spec in _registry.add_step_actions():
            action = QtWidgets.QAction(action_spec.text, self.add_step_menu)
            action.setData(action_spec.action_id)
            action.setToolTip(action_spec.tooltip)
            action.triggered.connect(
                lambda _checked=False, action_id=action_spec.action_id: (
                    self._add_operation(action_id)
                )
            )
            self.add_step_menu.addAction(action)
        self.add_step_button.clicked.connect(self._show_add_step_menu)
        self.remove_operation_button = _step_toolbar_button(
            recipe_page,
            "figureComposerDeleteStepButton",
            "Delete Step",
            "Remove the selected recipe step.",
        )
        self.remove_operation_button.clicked.connect(self._remove_current_operation)
        self.duplicate_operation_button = _step_toolbar_button(
            recipe_page,
            "figureComposerDuplicateStepButton",
            "Duplicate",
            "Copy the selected step and insert it immediately after the original.",
        )
        self.duplicate_operation_button.clicked.connect(
            self._duplicate_current_operation
        )
        self.move_operation_up_button = _step_toolbar_button(
            recipe_page,
            "figureComposerMoveStepUpButton",
            "Up",
            "Move the selected step earlier in the recipe.",
        )
        self.move_operation_up_button.clicked.connect(self._move_current_operation_up)
        self.move_operation_down_button = _step_toolbar_button(
            recipe_page,
            "figureComposerMoveStepDownButton",
            "Down",
            "Move the selected step later in the recipe.",
        )
        self.move_operation_down_button.clicked.connect(
            self._move_current_operation_down
        )
        self.show_figure_button = QtWidgets.QPushButton("Show Plot Window", root)
        self.show_figure_button.setObjectName("figureComposerShowFigureButton")
        self.show_figure_button.setToolTip(
            "Open or raise the separate Matplotlib plot window."
        )
        self.show_figure_button.clicked.connect(self._show_figure_window_requested)
        copy_button = QtWidgets.QPushButton("Copy Python", root)
        copy_button.setToolTip("Copy standalone Python code for this figure recipe.")
        copy_button.clicked.connect(self.copy_code)
        export_button = QtWidgets.QPushButton("Export", root)
        export_button.setToolTip("Save the rendered Matplotlib figure to a file.")
        export_button.clicked.connect(self.export_figure)

        output_action_layout = QtWidgets.QHBoxLayout()
        output_action_layout.setSpacing(4)
        output_action_layout.addWidget(self.show_figure_button)
        output_action_layout.addWidget(copy_button)
        output_action_layout.addWidget(export_button)
        output_action_layout.addStretch(1)
        action_layout.addLayout(output_action_layout)
        root_layout.addLayout(action_layout)
        root_layout.addWidget(self.editor_tabs, 1)

        selected_step_action_layout = QtWidgets.QHBoxLayout()
        selected_step_action_layout.setSpacing(4)
        selected_step_action_layout.addWidget(self.add_step_button)
        selected_step_action_layout.addWidget(self.duplicate_operation_button)
        selected_step_action_layout.addWidget(self.move_operation_up_button)
        selected_step_action_layout.addWidget(self.move_operation_down_button)
        selected_step_action_layout.addWidget(self.remove_operation_button)
        selected_step_action_layout.addStretch(1)
        recipe_layout.addLayout(selected_step_action_layout)

        self.operation_list = QtWidgets.QListWidget(recipe_page)
        self.operation_list.setObjectName("figureComposerOperationList")
        self.operation_list.currentRowChanged.connect(self._operation_selection_changed)
        self.operation_list.itemChanged.connect(self._operation_item_changed)
        self.operation_list.setMaximumHeight(130)
        self.operation_list.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.operation_list.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.operation_list.setToolTip(
            "Checked steps run from top to bottom to build the figure."
        )
        recipe_layout.addWidget(self.operation_list)

        self.step_inspector = QtWidgets.QWidget(recipe_page)
        self.step_inspector.setObjectName("figureComposerStepInspector")
        step_inspector_layout = QtWidgets.QHBoxLayout(self.step_inspector)
        step_inspector_layout.setContentsMargins(0, 0, 0, 0)
        step_inspector_layout.setSpacing(6)
        recipe_layout.addWidget(self.step_inspector, 1)

        self.step_navigator = QtWidgets.QWidget(self.step_inspector)
        self.step_navigator.setObjectName("figureComposerStepNavigator")
        self.step_navigator.setFixedWidth(150)
        self.step_navigator_layout = QtWidgets.QVBoxLayout(self.step_navigator)
        self.step_navigator_layout.setContentsMargins(0, 0, 0, 0)
        self.step_navigator_layout.setSpacing(3)
        self.step_section_buttons: dict[str, QtWidgets.QToolButton] = {}
        self.step_section_keys: list[str] = []
        step_inspector_layout.addWidget(self.step_navigator)

        self.step_editor_stack = QtWidgets.QStackedWidget(self.step_inspector)
        self.step_editor_stack.setObjectName("figureComposerStepSectionStack")
        step_inspector_layout.addWidget(self.step_editor_stack, 1)
        self._operation_editor_pages: list[QtWidgets.QWidget] = []

        self.step_sources_page = QtWidgets.QWidget(self.step_editor_stack)
        self.step_sources_page.setObjectName("figureComposerStepSourcesPage")
        sources_layout = QtWidgets.QVBoxLayout(self.step_sources_page)
        sources_layout.setContentsMargins(6, 6, 6, 6)
        sources_layout.setSpacing(4)
        self.step_source_controls = QtWidgets.QWidget(self.step_sources_page)
        self.step_source_controls_layout = QtWidgets.QFormLayout(
            self.step_source_controls
        )
        self.step_source_controls_layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        sources_layout.addWidget(self.step_source_controls)
        self.source_status_label = QtWidgets.QLabel(self.step_sources_page)
        self.source_status_label.setObjectName("figureComposerSourceStatus")
        self.source_status_label.setWordWrap(True)
        sources_layout.addWidget(self.source_status_label)
        self.source_list = QtWidgets.QListWidget(self.step_sources_page)
        self.source_list.setObjectName("figureComposerSourceList")
        self.source_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection
        )
        self.source_list.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.source_list.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.source_list.setToolTip("Data available to the selected step.")
        sources_layout.addWidget(self.source_list, 1)

        self.target_axes_page = QtWidgets.QWidget(self.step_editor_stack)
        self.target_axes_page.setObjectName("figureComposerTargetAxesPage")
        target_axes_layout = QtWidgets.QVBoxLayout(self.target_axes_page)
        target_axes_layout.setContentsMargins(6, 6, 6, 6)
        target_axes_layout.setSpacing(4)
        self.axes_selector = _AxesSelectorWidget(self.target_axes_page)
        self.axes_selector.sigSelectionChanged.connect(self._axes_selection_changed)
        target_axes_layout.addWidget(self.axes_selector)
        self.target_axes_status_label = QtWidgets.QLabel(self.target_axes_page)
        self.target_axes_status_label.setObjectName("figureComposerAxesStatus")
        self.target_axes_status_label.setWordWrap(True)
        target_axes_layout.addWidget(self.target_axes_status_label)
        target_axes_action_layout = QtWidgets.QHBoxLayout()
        self.use_all_axes_button = QtWidgets.QPushButton(
            "Use All Axes", self.target_axes_page
        )
        self.use_all_axes_button.setToolTip(
            "Retarget the selected step to every axis in the current figure grid."
        )
        self.use_all_axes_button.clicked.connect(
            self._target_current_operation_all_axes
        )
        self.keep_valid_axes_button = QtWidgets.QPushButton(
            "Drop Removed Axes", self.target_axes_page
        )
        self.keep_valid_axes_button.setToolTip(
            "Keep only selected axes that still exist in the current figure grid."
        )
        self.keep_valid_axes_button.clicked.connect(
            self._target_current_operation_valid_axes
        )
        target_axes_action_layout.addWidget(self.use_all_axes_button)
        target_axes_action_layout.addWidget(self.keep_valid_axes_button)
        target_axes_action_layout.addStretch(1)
        target_axes_layout.addLayout(target_axes_action_layout)
        self.axes_expression_edit = QtWidgets.QLineEdit(self.target_axes_page)
        self.axes_expression_edit.setPlaceholderText("Advanced: axs[0, :] or axs[:, 0]")
        self.axes_expression_edit.setToolTip(
            "Optional advanced axes expression used verbatim in copied Python code."
        )
        self.axes_expression_edit.editingFinished.connect(self._axes_expression_changed)
        target_axes_layout.addWidget(self.axes_expression_edit)
        target_axes_layout.addStretch(1)
        self.operation_editor = QtWidgets.QWidget(self.step_editor_stack)
        self.operation_editor_layout = QtWidgets.QFormLayout(self.operation_editor)

        layout_page = QtWidgets.QWidget(self.editor_tabs)
        layout_page.setObjectName("figureComposerLayoutPage")
        self.layout_page = layout_page
        setup_layout = QtWidgets.QGridLayout(layout_page)
        setup_layout.setContentsMargins(6, 6, 6, 6)
        setup_layout.setHorizontalSpacing(8)
        setup_layout.setVerticalSpacing(6)
        setup_layout.setColumnStretch(2, 1)
        setup_layout.setColumnStretch(4, 1)
        self.nrows_spin = QtWidgets.QSpinBox(layout_page)
        self.nrows_spin.setRange(1, 12)
        self.ncols_spin = QtWidgets.QSpinBox(layout_page)
        self.ncols_spin.setRange(1, 12)
        self.width_spin = QtWidgets.QDoubleSpinBox(layout_page)
        self.width_spin.setRange(0.25, 100.0)
        self.width_spin.setDecimals(3)
        self.width_spin.setSingleStep(0.25)
        self.height_spin = QtWidgets.QDoubleSpinBox(layout_page)
        self.height_spin.setRange(0.25, 100.0)
        self.height_spin.setDecimals(3)
        self.height_spin.setSingleStep(0.25)
        self.width_mm_spin = QtWidgets.QDoubleSpinBox(layout_page)
        self.width_mm_spin.setRange(0.25 * _MM_PER_INCH, 100.0 * _MM_PER_INCH)
        self.width_mm_spin.setDecimals(2)
        self.width_mm_spin.setSingleStep(1.0)
        self.height_mm_spin = QtWidgets.QDoubleSpinBox(layout_page)
        self.height_mm_spin.setRange(0.25 * _MM_PER_INCH, 100.0 * _MM_PER_INCH)
        self.height_mm_spin.setDecimals(2)
        self.height_mm_spin.setSingleStep(1.0)
        self.layout_combo = QtWidgets.QComboBox(layout_page)
        self.layout_combo.addItems(["constrained", "compressed", "tight", "none"])
        self.sharex_combo = QtWidgets.QComboBox(layout_page)
        self.sharex_combo.addItems(["False", "True", "row", "col", "all"])
        self.sharey_combo = QtWidgets.QComboBox(layout_page)
        self.sharey_combo.addItems(["False", "True", "row", "col", "all"])
        self.width_ratios_edit = QtWidgets.QLineEdit(layout_page)
        self.width_ratios_edit.setObjectName("figureComposerWidthRatiosEdit")
        self.height_ratios_edit = QtWidgets.QLineEdit(layout_page)
        self.height_ratios_edit.setObjectName("figureComposerHeightRatiosEdit")

        def add_grid_pair_row(
            row: int,
            row_label_text: str,
            row_object_name: str,
            row_tooltip: str,
            first_label: str,
            first_widget: QtWidgets.QWidget,
            first_tooltip: str,
            second_label: str,
            second_widget: QtWidgets.QWidget,
            second_tooltip: str,
        ) -> None:
            row_label = QtWidgets.QLabel(row_label_text, layout_page)
            row_label.setObjectName(row_object_name)
            row_label.setToolTip(row_tooltip)
            row_label.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignRight
                | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
            setup_layout.addWidget(row_label, row, 0)
            for column, label_text, widget, tooltip in (
                (1, first_label, first_widget, first_tooltip),
                (3, second_label, second_widget, second_tooltip),
            ):
                label = QtWidgets.QLabel(label_text, layout_page)
                label.setBuddy(widget)
                label.setToolTip(tooltip)
                widget.setToolTip(tooltip)
                setup_layout.addWidget(label, row, column)
                setup_layout.addWidget(widget, row, column + 1)

        add_grid_pair_row(
            0,
            "Grid",
            "figureComposerGridControls",
            "Subplot grid created by plt.subplots.",
            "Rows",
            self.nrows_spin,
            "Number of subplot rows created by plt.subplots.",
            "Columns",
            self.ncols_spin,
            "Number of subplot columns created by plt.subplots.",
        )
        add_grid_pair_row(
            1,
            "Size (in)",
            "figureComposerSizeControls",
            "Figure size in inches. The plot window follows this size.",
            "Width",
            self.width_spin,
            "Figure width in inches. The plot window follows this size.",
            "Height",
            self.height_spin,
            "Figure height in inches. The plot window follows this size.",
        )
        add_grid_pair_row(
            2,
            "Size (mm)",
            "figureComposerSizeMmControls",
            "Figure size in millimeters, synced with the inch controls.",
            "Width",
            self.width_mm_spin,
            "Figure width in millimeters. Converted to inches for Matplotlib.",
            "Height",
            self.height_mm_spin,
            "Figure height in millimeters. Converted to inches for Matplotlib.",
        )
        layout_row_label = QtWidgets.QLabel("Layout engine", layout_page)
        layout_row_label.setObjectName("figureComposerLayoutControls")
        layout_row_label.setToolTip("Matplotlib layout engine passed to plt.subplots.")
        layout_row_label.setBuddy(self.layout_combo)
        layout_row_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        self.layout_combo.setToolTip(
            "Matplotlib layout engine passed to plt.subplots.",
        )
        setup_layout.addWidget(layout_row_label, 3, 0, 1, 2)
        setup_layout.addWidget(self.layout_combo, 3, 2, 1, 3)
        add_grid_pair_row(
            4,
            "Share axes",
            "figureComposerShareControls",
            "Matplotlib shared-axis settings passed to plt.subplots.",
            "x",
            self.sharex_combo,
            "Matplotlib sharex setting passed to plt.subplots.",
            "y",
            self.sharey_combo,
            "Matplotlib sharey setting passed to plt.subplots.",
        )
        add_grid_pair_row(
            5,
            "Ratios",
            "figureComposerRatioControls",
            "Optional subplot width and height ratios passed to plt.subplots.",
            "Widths",
            self.width_ratios_edit,
            "Optional width_ratios values, one positive number per column.",
            "Heights",
            self.height_ratios_edit,
            "Optional height_ratios values, one positive number per row.",
        )
        setup_layout.setRowStretch(6, 1)

        layout_index = self.editor_tabs.addTab(layout_page, "Layout")
        self.editor_tabs.setTabToolTip(
            layout_index, "Subplot grid, figure size, and shared axes."
        )
        recipe_index = self.editor_tabs.addTab(recipe_page, "Recipe")
        self.editor_tabs.setTabToolTip(
            recipe_index,
            "Ordered plotting steps and controls for the selected step.",
        )
        self.editor_tabs.setCurrentWidget(recipe_page)

        for widget in (
            self.nrows_spin,
            self.ncols_spin,
            self.width_spin,
            self.height_spin,
        ):
            typing.cast("QtWidgets.QAbstractSpinBox", widget).editingFinished.connect(
                self._setup_controls_changed
            )
        for widget in (self.width_mm_spin, self.height_mm_spin):
            typing.cast("QtWidgets.QAbstractSpinBox", widget).editingFinished.connect(
                self._size_mm_controls_changed
            )
        for widget in (self.width_ratios_edit, self.height_ratios_edit):
            widget.editingFinished.connect(self._setup_controls_changed)
        self.layout_combo.currentTextChanged.connect(self._setup_controls_changed)
        self.sharex_combo.currentTextChanged.connect(self._setup_controls_changed)
        self.sharey_combo.currentTextChanged.connect(self._setup_controls_changed)

        self.setCentralWidget(root)
        self.setWindowTitle("Figure Composer")

    def _apply_recipe_to_controls(self) -> None:
        self._updating_controls = True
        try:
            setup = self._recipe.setup
            self.nrows_spin.setValue(setup.nrows)
            self.ncols_spin.setValue(setup.ncols)
            self.width_spin.setValue(setup.figsize[0])
            self.height_spin.setValue(setup.figsize[1])
            self._sync_size_mm_controls(setup.figsize)
            self._set_combo_value(self.layout_combo, setup.layout or "none")
            self._set_combo_value(self.sharex_combo, str(setup.sharex))
            self._set_combo_value(self.sharey_combo, str(setup.sharey))
            self.width_ratios_edit.setText(_format_tuple(setup.width_ratios))
            self.height_ratios_edit.setText(_format_tuple(setup.height_ratios))
            self._refresh_source_list()
            self._rebuild_axes_grid()
            self._refresh_operation_list()
        finally:
            self._updating_controls = False
        if self.operation_list.count() and self.operation_list.currentRow() < 0:
            self.operation_list.setCurrentRow(0)
        self._update_operation_editor()

    @staticmethod
    def _set_combo_value(combo: QtWidgets.QComboBox, value: str) -> None:
        index = combo.findText(value)
        if index < 0 and value == "False":
            index = combo.findText("False")
        if index < 0 and value == "True":
            index = combo.findText("True")
        combo.setCurrentIndex(max(index, 0))

    def _sync_size_mm_controls(self, figsize: tuple[float, float]) -> None:
        self.width_mm_spin.setValue(figsize[0] * _MM_PER_INCH)
        self.height_mm_spin.setValue(figsize[1] * _MM_PER_INCH)

    @staticmethod
    def _ratio_tuple_from_text(text: str) -> tuple[float, ...]:
        values = tuple(float(value) for value in _literal_sequence_from_text(text))
        if any(value <= 0.0 for value in values):
            raise ValueError("subplot ratios must be positive")
        return values

    def _refresh_source_list(self) -> None:
        self.source_list.clear()
        source_by_name = {source.name: source for source in self._recipe.sources}
        for name, data in self._source_data.items():
            source = source_by_name.get(name)
            label = source.label if source is not None else name
            dims = " x ".join(f"{dim}:{data.sizes[dim]}" for dim in data.dims)
            item = QtWidgets.QListWidgetItem(f"{name}  {label}  ({dims})")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, name)
            if source is None:
                item.setForeground(QtGui.QBrush(QtGui.QColor("darkRed")))
            self.source_list.addItem(item)

        missing = [
            source
            for source in self._recipe.sources
            if source.name not in self._source_data
        ]
        for source in missing:
            item = QtWidgets.QListWidgetItem(
                f"{source.name}  {source.label}  (missing)"
            )
            item.setData(QtCore.Qt.ItemDataRole.UserRole, source.name)
            item.setForeground(QtGui.QBrush(QtGui.QColor("darkRed")))
            self.source_list.addItem(item)

    def _rebuild_axes_grid(self) -> None:
        self.axes_selector.set_grid(
            self._recipe.setup.nrows,
            self._recipe.setup.ncols,
        )
        self._sync_axes_selector()

    def _sync_axes_selector(self) -> None:
        current = self._current_operation()
        selected_axes = set()
        expression = ""
        invalid_axes: tuple[tuple[int, int], ...] = ()
        if current is not None:
            _, operation = current
            spec = _registry.spec_for(operation.kind)
            if spec.uses_axes(operation):
                selected_axes = set(operation.axes.valid_axes(self._recipe.setup))
                invalid_axes = operation.axes.invalid_axes(self._recipe.setup)
                expression = operation.axes.expression

        self._updating_controls = True
        try:
            self.axes_selector.set_selected_axes(tuple(sorted(selected_axes)))
            if self.axes_expression_edit.text() != expression:
                blocker = QtCore.QSignalBlocker(self.axes_expression_edit)
                self.axes_expression_edit.setText(expression)
                del blocker
            self.keep_valid_axes_button.setEnabled(bool(invalid_axes))
            if current is None:
                self.target_axes_status_label.setText("Select a step to choose axes.")
            elif not _registry.spec_for(current[1].kind).uses_axes(current[1]):
                self.target_axes_status_label.setText(
                    _registry.spec_for(current[1].kind).target_text(self, current[1])
                )
            elif expression:
                self.target_axes_status_label.setText(
                    "Using the advanced axes expression instead of grid selection."
                )
            elif invalid_axes:
                self.target_axes_status_label.setText(
                    "Target axes removed by the current layout: "
                    f"{_format_axes_tuple(invalid_axes)}"
                )
            else:
                axes_text = _format_axes_tuple(
                    tuple(sorted(selected_axes)),
                    nrows=self._recipe.setup.nrows,
                    ncols=self._recipe.setup.ncols,
                )
                self.target_axes_status_label.setText(f"Targets: {axes_text}")
        finally:
            self._updating_controls = False

    def _refresh_operation_list(self) -> None:
        current_id = None
        current = self._current_operation()
        if current is not None:
            current_id = current[1].operation_id
        self.operation_list.blockSignals(True)
        try:
            self.operation_list.clear()
            for operation in self._recipe.operations:
                item = QtWidgets.QListWidgetItem(
                    self._operation_display_text(operation)
                )
                item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    QtCore.Qt.CheckState.Checked
                    if operation.enabled
                    else QtCore.Qt.CheckState.Unchecked
                )
                item.setData(QtCore.Qt.ItemDataRole.UserRole, operation.operation_id)
                item.setToolTip(self._operation_tooltip(operation))
                if self._operation_has_invalid_axes(operation):
                    item.setForeground(QtGui.QBrush(QtGui.QColor("darkRed")))
                self.operation_list.addItem(item)
                if operation.operation_id == current_id:
                    self.operation_list.setCurrentItem(item)
        finally:
            self.operation_list.blockSignals(False)

    def _operation_display_text(self, operation: FigureOperationState) -> str:
        return _registry.spec_for(operation.kind).display_text(self, operation)

    def _operation_tooltip(self, operation: FigureOperationState) -> str:
        return _registry.spec_for(operation.kind).tooltip(self, operation)

    def _axes_target_text(self, selection: FigureAxesSelectionState) -> str:
        if selection.expression:
            return selection.expression
        invalid_axes = selection.invalid_axes(self._recipe.setup)
        if invalid_axes:
            return f"removed axes {_format_axes_tuple(invalid_axes)}"
        valid_axes = selection.valid_axes(self._recipe.setup)
        if not valid_axes:
            return "none"
        return _format_axes_tuple(
            valid_axes,
            nrows=self._recipe.setup.nrows,
            ncols=self._recipe.setup.ncols,
        )

    def _operation_has_invalid_axes(self, operation: FigureOperationState) -> bool:
        return _registry.spec_for(operation.kind).has_invalid_target(self, operation)

    def _invalid_operation_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, operation in enumerate(self._recipe.operations)
            if operation.enabled and self._operation_has_invalid_axes(operation)
        )

    def _warn_invalid_operation_targets(self) -> bool:
        indices = self._invalid_operation_indices()
        if not indices:
            return False
        self.editor_tabs.setCurrentWidget(self.recipe_page)
        self.operation_list.setCurrentRow(indices[0])
        self._select_step_section("axes")
        QtWidgets.QMessageBox.warning(
            self,
            "Retarget Figure Steps",
            "Some enabled steps target axes that no longer exist in the current "
            "figure layout. Use All Axes or Drop Removed Axes before copying, "
            "exporting, or rendering the full recipe.",
        )
        return True

    def _current_operation(self) -> tuple[int, FigureOperationState] | None:
        row = self.operation_list.currentRow()
        if row < 0 or row >= len(self._recipe.operations):
            return None
        return row, self._recipe.operations[row]

    def _replace_operation(
        self,
        index: int,
        operation: FigureOperationState,
        *,
        render: bool = True,
        rebuild_editor: bool = False,
        sync_axes: bool = True,
    ) -> None:
        operations = list(self._recipe.operations)
        operations[index] = operation
        self._recipe = self._recipe.model_copy(update={"operations": tuple(operations)})
        self._refresh_operation_list()
        self.operation_list.setCurrentRow(index)
        if sync_axes:
            self._sync_axes_selector()
        self._refresh_step_section_button_texts()
        self._update_source_status(operation)
        if rebuild_editor:
            self._update_operation_editor()
        if render:
            _rendering._render_preview(self)
            self.sigInfoChanged.emit()

    def _update_current_operation(self, **updates: typing.Any) -> None:
        current = self._current_operation()
        if current is None:
            return
        index, operation = current
        self._replace_operation(index, operation.model_copy(update=updates))

    def _update_current_operation_rebuild(self, **updates: typing.Any) -> None:
        current = self._current_operation()
        if current is None:
            return
        index, operation = current
        self._replace_operation(
            index,
            operation.model_copy(update=updates),
            rebuild_editor=True,
        )

    def _update_current_operation_in_place(self, **updates: typing.Any) -> None:
        current = self._current_operation()
        if current is None:
            return
        index, operation = current
        operations = list(self._recipe.operations)
        operations[index] = operation.model_copy(update=updates)
        self._recipe = self._recipe.model_copy(update={"operations": tuple(operations)})
        self._refresh_operation_list()
        self.operation_list.setCurrentRow(index)
        self._refresh_step_section_button_texts()
        erlab.interactive.utils.single_shot(
            self, 0, lambda: _rendering._render_preview(self)
        )
        self.sigInfoChanged.emit()

    def _update_step_action_buttons(self) -> None:
        current = self._current_operation()
        if current is None:
            self.remove_operation_button.setEnabled(False)
            self.duplicate_operation_button.setEnabled(False)
            self.move_operation_up_button.setEnabled(False)
            self.move_operation_down_button.setEnabled(False)
            return
        index, _operation = current
        self.remove_operation_button.setEnabled(True)
        self.duplicate_operation_button.setEnabled(True)
        self.move_operation_up_button.setEnabled(index > 0)
        self.move_operation_down_button.setEnabled(
            index < len(self._recipe.operations) - 1
        )

    @QtCore.Slot()
    def _setup_controls_changed(self) -> None:
        if self._updating_controls:
            return
        try:
            setup = FigureSubplotsState(
                nrows=self.nrows_spin.value(),
                ncols=self.ncols_spin.value(),
                figsize=(self.width_spin.value(), self.height_spin.value()),
                dpi=self._recipe.setup.dpi,
                layout=self._layout_combo_value(),
                sharex=self._combo_bool_or_text(self.sharex_combo),
                sharey=self._combo_bool_or_text(self.sharey_combo),
                width_ratios=self._ratio_tuple_from_text(self.width_ratios_edit.text()),
                height_ratios=self._ratio_tuple_from_text(
                    self.height_ratios_edit.text()
                ),
            )
        except ValueError:
            return
        if setup == self._recipe.setup:
            return
        self._recipe = self._recipe.model_copy(update={"setup": setup})
        self._updating_controls = True
        try:
            self._sync_size_mm_controls(setup.figsize)
        finally:
            self._updating_controls = False
        if (
            self._figure_window is not None
            and erlab.interactive.utils.qt_is_valid(self._figure_window)
            and self._figure_window.isVisible()
        ):
            self._figure_window.resize_to_setup(setup)
            self.canvas.flush_events()
            self._sync_recipe_figsize_to_canvas(draw=False, emit_info=False)
        self._rebuild_axes_grid()
        self._refresh_operation_list()
        self._update_operation_editor()
        if self._invalid_operation_indices():
            self.editor_tabs.setCurrentWidget(self.recipe_page)
            self._select_step_section("axes")
        _rendering._render_preview(self)
        self.sigInfoChanged.emit()

    @QtCore.Slot()
    def _size_mm_controls_changed(self) -> None:
        if self._updating_controls:
            return
        self._updating_controls = True
        try:
            self.width_spin.setValue(self.width_mm_spin.value() / _MM_PER_INCH)
            self.height_spin.setValue(self.height_mm_spin.value() / _MM_PER_INCH)
        finally:
            self._updating_controls = False
        self._setup_controls_changed()

    def _layout_combo_value(
        self,
    ) -> typing.Literal["constrained", "compressed", "tight"] | None:
        text = self.layout_combo.currentText()
        if text == "none":
            return None
        return typing.cast(
            'typing.Literal["constrained", "compressed", "tight"]',
            text,
        )

    @staticmethod
    def _combo_bool_or_text(
        combo: QtWidgets.QComboBox,
    ) -> bool | typing.Literal["none", "all", "row", "col"]:
        text = combo.currentText()
        if text == "True":
            return True
        if text == "False":
            return False
        return typing.cast('typing.Literal["none", "all", "row", "col"]', text)

    @QtCore.Slot(object)
    def _axes_selection_changed(self, axes_obj: object) -> None:
        if self._updating_controls:
            return
        current = self._current_operation()
        if current is None:
            return
        if not _registry.spec_for(current[1].kind).uses_axes(current[1]):
            return
        axes = typing.cast("tuple[tuple[int, int], ...]", axes_obj)
        if not axes:
            axes = ((0, 0),)
        index, operation = current
        selection = operation.axes.model_copy(update={"axes": axes, "expression": ""})
        self._replace_operation(
            index,
            operation.model_copy(update={"axes": selection}),
            sync_axes=False,
        )
        erlab.interactive.utils.single_shot(self, 0, self._sync_axes_selector)

    @QtCore.Slot()
    def _axes_expression_changed(self) -> None:
        if self._updating_controls:
            return
        current = self._current_operation()
        if current is None:
            return
        if not _registry.spec_for(current[1].kind).uses_axes(current[1]):
            return
        index, operation = current
        selection = operation.axes.model_copy(
            update={"expression": self.axes_expression_edit.text().strip()}
        )
        self._replace_operation(
            index,
            operation.model_copy(update={"axes": selection}),
            sync_axes=False,
        )
        erlab.interactive.utils.single_shot(self, 0, self._sync_axes_selector)

    @QtCore.Slot(int)
    def _operation_selection_changed(self, _row: int) -> None:
        self._sync_axes_selector()
        self._update_operation_editor()

    @QtCore.Slot(QtWidgets.QListWidgetItem)
    def _operation_item_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        if self._updating_controls:
            return
        operation_id = item.data(QtCore.Qt.ItemDataRole.UserRole)
        for index, operation in enumerate(self._recipe.operations):
            if operation.operation_id == operation_id:
                updated = operation.model_copy(
                    update={
                        "enabled": item.checkState() == QtCore.Qt.CheckState.Checked,
                    }
                )
                operations = list(self._recipe.operations)
                operations[index] = updated
                self._recipe = self._recipe.model_copy(
                    update={"operations": tuple(operations)}
                )
                if index == self.operation_list.currentRow():
                    self._sync_axes_selector()
                    self._update_source_status(updated)
                self._refresh_step_section_button_texts()
                _rendering._render_preview(self)
                self.sigInfoChanged.emit()
                return

    @QtCore.Slot()
    def _target_current_operation_all_axes(self) -> None:
        current = self._current_operation()
        if current is None:
            return
        index, operation = current
        if not _registry.spec_for(operation.kind).uses_axes(operation):
            return
        selection = operation.axes.model_copy(
            update={"axes": _all_axes(self._recipe.setup), "expression": ""}
        )
        self._replace_operation(index, operation.model_copy(update={"axes": selection}))

    @QtCore.Slot()
    def _target_current_operation_valid_axes(self) -> None:
        current = self._current_operation()
        if current is None:
            return
        index, operation = current
        if not _registry.spec_for(operation.kind).uses_axes(operation):
            return
        axes = operation.axes.valid_axes(self._recipe.setup)
        if not axes:
            axes = ((0, 0),)
        selection = operation.axes.model_copy(update={"axes": axes, "expression": ""})
        self._replace_operation(index, operation.model_copy(update={"axes": selection}))

    def _source_names(self) -> tuple[str, ...]:
        names = tuple(source.name for source in self._recipe.sources)
        if names:
            return names
        return tuple(self._source_data)

    def _selected_axes_state(self) -> FigureAxesSelectionState:
        axes = self.axes_selector.selected_axes()
        if not axes:
            axes = ((0, 0),)
        return FigureAxesSelectionState(
            axes=axes,
            expression=self.axes_expression_edit.text().strip(),
        )

    @QtCore.Slot()
    def _show_add_step_menu(self) -> None:
        self.add_step_menu.popup(
            self.add_step_button.mapToGlobal(
                QtCore.QPoint(0, self.add_step_button.height())
            )
        )

    def _add_operation(self, action_id: str) -> None:
        operation = _registry.spec_for_action(action_id).create_operation(self)
        operations = (*self._recipe.operations, operation)
        self._recipe = self._recipe.model_copy(update={"operations": operations})
        self._refresh_operation_list()
        self.operation_list.setCurrentRow(len(operations) - 1)
        _rendering._render_preview(self)
        self.sigInfoChanged.emit()

    @QtCore.Slot()
    def _remove_current_operation(self) -> None:
        current = self._current_operation()
        if current is None:
            return
        index, _operation = current
        operations = list(self._recipe.operations)
        operations.pop(index)
        self._recipe = self._recipe.model_copy(update={"operations": tuple(operations)})
        self._refresh_operation_list()
        if operations:
            self.operation_list.setCurrentRow(min(index, len(operations) - 1))
        self._update_operation_editor()
        _rendering._render_preview(self)
        self.sigInfoChanged.emit()

    @QtCore.Slot()
    def _duplicate_current_operation(self) -> None:
        current = self._current_operation()
        if current is None:
            return
        index, operation = current
        duplicate = operation.model_copy(
            update={"operation_id": uuid.uuid4().hex},
            deep=True,
        )
        operations = list(self._recipe.operations)
        operations.insert(index + 1, duplicate)
        self._recipe = self._recipe.model_copy(update={"operations": tuple(operations)})
        self._refresh_operation_list()
        self.operation_list.setCurrentRow(index + 1)
        self._sync_axes_selector()
        self._update_operation_editor()
        _rendering._render_preview(self)
        self.sigInfoChanged.emit()

    def _move_current_operation(self, offset: int) -> None:
        current = self._current_operation()
        if current is None:
            return
        index, _operation = current
        new_index = index + offset
        if new_index < 0 or new_index >= len(self._recipe.operations):
            return
        operations = list(self._recipe.operations)
        operations[index], operations[new_index] = (
            operations[new_index],
            operations[index],
        )
        self._recipe = self._recipe.model_copy(update={"operations": tuple(operations)})
        self._refresh_operation_list()
        self.operation_list.setCurrentRow(new_index)
        self._sync_axes_selector()
        self._update_operation_editor()
        _rendering._render_preview(self)
        self.sigInfoChanged.emit()

    @QtCore.Slot()
    def _move_current_operation_up(self) -> None:
        self._move_current_operation(-1)

    @QtCore.Slot()
    def _move_current_operation_down(self) -> None:
        self._move_current_operation(1)

    def add_operation(self, operation: FigureOperationState) -> None:
        """Append an operation to the recipe."""
        self._recipe = self._recipe.model_copy(
            update={"operations": (*self._recipe.operations, operation)}
        )
        self._apply_recipe_to_controls()
        self.operation_list.setCurrentRow(len(self._recipe.operations) - 1)
        _rendering._render_preview(self)
        self.sigInfoChanged.emit()

    def add_sources(
        self,
        sources: Sequence[FigureSourceState],
        source_data: Mapping[str, xr.DataArray],
    ) -> None:
        """Add or refresh recipe sources used by appended operations."""
        existing = {source.name: source for source in self._recipe.sources}
        for source in sources:
            existing[source.name] = source
        self._source_data.update(source_data)
        ordered_sources = tuple(existing[name] for name in existing)
        self._recipe = self._recipe.model_copy(update={"sources": ordered_sources})
        self._refresh_source_list()
        self._update_source_section()
        _rendering._render_preview(self)
        self.sigInfoChanged.emit()

    @staticmethod
    def _clear_form_layout(layout: QtWidgets.QFormLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _clear_operation_editor(self) -> None:
        for page in self._operation_editor_pages:
            self.step_editor_stack.removeWidget(page)
            page.deleteLater()
        self._operation_editor_pages.clear()

    def _clear_step_source_controls(self) -> None:
        self._clear_form_layout(self.step_source_controls_layout)

    def _new_step_form_page(
        self, object_name: str
    ) -> tuple[QtWidgets.QWidget, QtWidgets.QFormLayout]:
        page = QtWidgets.QWidget(self.step_editor_stack)
        page.setObjectName(object_name)
        layout = QtWidgets.QFormLayout(page)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self._operation_editor_pages.append(page)
        return page, layout

    def _set_step_sections(self, sections: Sequence[StepSection]) -> None:
        while self.step_navigator_layout.count():
            item = self.step_navigator_layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.step_section_buttons.clear()
        self.step_section_keys = [section.key for section in sections]

        while self.step_editor_stack.count():
            self.step_editor_stack.removeWidget(self.step_editor_stack.widget(0))

        for index, section in enumerate(sections):
            self.step_editor_stack.addWidget(section.page)
            button = QtWidgets.QToolButton(self.step_navigator)
            button.setText(self._section_button_text(section.key, section.title))
            button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
            button.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            button.setObjectName(f"figureComposerSection_{section.key}")
            button.setProperty("section_title", section.title)
            button.setProperty("section_tooltip", section.tooltip)
            button.setToolTip(section.tooltip)
            button.clicked.connect(
                lambda _checked=False, key=section.key: self._select_step_section(key)
            )
            self.step_navigator_layout.insertWidget(index, button)
            self.step_section_buttons[section.key] = button
        self.step_navigator_layout.addStretch(1)

        key = (
            self._current_step_section_key
            if self._current_step_section_key in self.step_section_keys
            else self.step_section_keys[0]
            if self.step_section_keys
            else ""
        )
        if key:
            self._select_step_section(key)

    def _select_step_section(self, key: str) -> None:
        if key not in self.step_section_keys:
            return
        self._current_step_section_key = key
        index = self.step_section_keys.index(key)
        self.step_editor_stack.setCurrentIndex(index)
        for button_key, button in self.step_section_buttons.items():
            font = button.font()
            font.setBold(button_key == key)
            button.setFont(font)

    def _refresh_step_section_button_texts(self) -> None:
        for key, button in self.step_section_buttons.items():
            title = button.property("section_title")
            if not isinstance(title, str):
                title = key
            button.setText(self._section_button_text(key, title))

    def _section_button_text(self, key: str, title: str) -> str:
        current = self._current_operation()
        if current is None:
            return title
        _index, operation = current
        summary = self._section_summary(key, operation)
        return f"{title}: {summary}" if summary else title

    def _section_summary(self, key: str, operation: FigureOperationState) -> str:
        return _registry.spec_for(operation.kind).section_summary(self, key, operation)

    def _selected_sources_for_operation(
        self, operation: FigureOperationState
    ) -> tuple[str, ...]:
        return _registry.spec_for(operation.kind).source_names(operation)

    def _update_source_status(self, operation: FigureOperationState | None) -> None:
        if operation is None:
            self.source_status_label.setText("Select a step to choose data sources.")
            return
        selected_sources = self._selected_sources_for_operation(operation)
        missing = [
            source for source in selected_sources if source not in self._source_data
        ]
        if missing:
            self.source_status_label.setText("Missing sources: " + ", ".join(missing))
        elif selected_sources:
            self.source_status_label.setText(
                "Selected sources: " + ", ".join(selected_sources)
            )
        else:
            self.source_status_label.setText("This step does not read a data source.")

    def _update_source_section(self) -> None:
        self._clear_step_source_controls()
        current = self._current_operation()
        if current is None:
            self._update_source_status(None)
            return
        _index, operation = current

        _registry.spec_for(operation.kind).build_source_editor(self, operation)
        self._update_source_status(operation)

    def _update_operation_editor(self) -> None:
        section_key = self._current_step_section_key
        self._clear_operation_editor()
        self._update_source_section()
        current = self._current_operation()
        self._update_step_action_buttons()
        if current is None:
            page, layout = self._new_step_form_page("figureComposerNoStepPage")
            layout.addRow(QtWidgets.QLabel("Select a recipe step to edit.", page))
            self._set_step_sections(
                (StepSection("empty", "Step", page, "Select a recipe step."),)
            )
            return
        _index, operation = current

        spec = _registry.spec_for(operation.kind)
        sections: list[StepSection] = []
        if spec.uses_source_section(operation):
            sections.append(
                StepSection(
                    "sources",
                    "Sources",
                    self.step_sources_page,
                    COMMON_SOURCE_SECTION_TOOLTIP,
                )
            )
        if spec.uses_axes(operation):
            sections.append(
                StepSection(
                    "axes",
                    "Axes",
                    self.target_axes_page,
                    COMMON_AXES_SECTION_TOOLTIP,
                )
            )
        sections.extend(spec.build_editor_sections(self, operation))
        self._current_step_section_key = section_key
        self._set_step_sections(sections)

    def _line_edit(
        self, text: str, *, parent: QtWidgets.QWidget | None = None
    ) -> QtWidgets.QLineEdit:
        edit = QtWidgets.QLineEdit(parent or self.operation_editor)
        edit.setText(text)
        return edit

    def _combo(
        self,
        values: Sequence[str],
        current: str | None,
        changed: Callable[[str], None],
        *,
        parent: QtWidgets.QWidget | None = None,
    ) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox(parent or self.operation_editor)
        combo.addItems(list(values))
        if current is not None:
            self._set_combo_value(combo, current)
        combo.currentTextChanged.connect(changed)
        return combo

    def _optional_name_combo(
        self,
        values: Sequence[str],
        current: str | None,
        none_label: str,
        changed: Callable[[str | None], None],
        *,
        parent: QtWidgets.QWidget | None = None,
    ) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox(parent or self.operation_editor)
        combo.addItem(none_label, None)
        for value in values:
            combo.addItem(value, value)
        if current is not None and current not in values:
            combo.addItem(current, current)
        for index in range(combo.count()):
            if combo.itemData(index) == current:
                combo.setCurrentIndex(index)
                break
        combo.currentIndexChanged.connect(
            lambda _index, combo=combo: changed(
                typing.cast("str | None", combo.currentData())
            )
        )
        return combo

    def _check_box(
        self,
        checked: bool,
        changed: Callable[[bool], None],
        *,
        parent: QtWidgets.QWidget | None = None,
    ) -> QtWidgets.QCheckBox:
        check = QtWidgets.QCheckBox(parent or self.operation_editor)
        check.setChecked(checked)
        check.toggled.connect(changed)
        return check

    @staticmethod
    def _add_form_row(
        layout: QtWidgets.QFormLayout,
        label: str,
        widget: QtWidgets.QWidget,
        tooltip: str,
    ) -> None:
        if "\n" not in tooltip:
            tooltip = "\n".join(
                textwrap.wrap(tooltip, width=58, break_long_words=False)
            )
        widget.setToolTip(tooltip)
        layout.addRow(label, widget)
        label_widget = layout.labelForField(widget)
        if label_widget is not None:
            label_widget.setToolTip(tooltip)

    def generated_code(self) -> str:
        return _codegen.generated_code(self)

    @QtCore.Slot()
    def copy_code(self) -> None:
        if self._warn_invalid_operation_targets():
            return
        erlab.interactive.utils.copy_to_clipboard(self.generated_code())

    @QtCore.Slot()
    def export_figure(self) -> None:
        if self._warn_invalid_operation_targets():
            return
        filename, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Figure",
            "",
            "Images (*.png *.pdf *.svg);;All files (*)",
        )
        if not filename:
            return
        _rendering._render_preview(self, show_window=False)
        with _figure_draw_context():
            self.figure.savefig(
                filename,
                dpi=self._recipe.export.dpi,
                transparent=self._recipe.export.transparent,
                bbox_inches=self._recipe.export.bbox_inches,
            )

    @property
    def tool_data(self) -> xr.DataArray:
        if self._recipe.primary_source in self._source_data:
            return self._source_data[self._recipe.primary_source]
        return next(iter(self._source_data.values()))

    @property
    def tool_status(self) -> FigureRecipeState:
        return self._recipe

    @tool_status.setter
    def tool_status(self, status: FigureRecipeState) -> None:
        operations = tuple(
            _registry.spec_for(operation.kind).loaded_operation(operation)
            for operation in status.operations
        )
        self._recipe = status.model_copy(update={"operations": operations})
        self._apply_recipe_to_controls()
        _rendering._render_preview(self)

    def set_source_data(self, source_data: Mapping[str, xr.DataArray]) -> None:
        self._source_data = dict(source_data)

    @property
    def preview_pixmap(self) -> QtGui.QPixmap | None:
        if not self._recipe.operations:
            return None
        with _figure_style_context():
            figure = Figure(
                figsize=self._recipe.setup.figsize,
                dpi=self._recipe.setup.dpi,
                layout=self._recipe.setup.layout,
            )
            canvas = FigureCanvasAgg(figure)
            try:
                _rendering._render_into_figure(self, figure, sync_visible=False)
                with _figure_draw_context():
                    canvas.draw()
                width, height = canvas.get_width_height()
                if width <= 0 or height <= 0:
                    return None
                image = QtGui.QImage(
                    canvas.buffer_rgba(),
                    width,
                    height,
                    QtGui.QImage.Format.Format_RGBA8888,
                )
                return QtGui.QPixmap.fromImage(image.copy())
            except Exception:
                return None
            finally:
                figure.clear()

    def source_states(self) -> tuple[FigureSourceState, ...]:
        return self._recipe.sources

    def source_data(self) -> dict[str, xr.DataArray]:
        return dict(self._source_data)

    def current_provenance_spec(self) -> provenance.ToolProvenanceSpec | None:
        script_inputs = tuple(
            script_input
            for source in self._recipe.sources
            if (script_input := source.to_script_input()) is not None
        )
        if not script_inputs:
            return None
        return provenance.script(
            start_label="Figure",
            script_inputs=script_inputs,
        )

    def set_missing_sources(self, names: set[str]) -> None:
        if not names:
            return
        for source in self._recipe.sources:
            if source.name in names:
                self._source_data.pop(source.name, None)
        self._refresh_source_list()
        self._update_source_section()
        _rendering._render_preview(self)

    def refresh_from_sources(self, source_data: Mapping[str, xr.DataArray]) -> None:
        self._source_data.update(source_data)
        self._refresh_source_list()
        self._update_source_section()
        _rendering._render_preview(self)
