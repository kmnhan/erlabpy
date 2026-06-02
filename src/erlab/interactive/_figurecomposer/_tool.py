"""Manager-facing Figure Composer tool window."""

from __future__ import annotations

import json
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

import xarray as xr

import erlab
from erlab.interactive._figurecomposer import _codegen, _rendering
from erlab.interactive._figurecomposer._axes import _all_axes
from erlab.interactive._figurecomposer._defaults import (
    _MM_PER_INCH,
    _figure_draw_context,
    _figure_style_context,
)
from erlab.interactive._figurecomposer._gridspec import (
    _gridspec_all_axes_ids,
    _gridspec_axes_subplot_targets,
    _gridspec_axis_display_name,
    _gridspec_axis_display_names,
    _gridspec_grid_by_id,
    _gridspec_grid_path,
    _gridspec_has_invalid_regions,
    _gridspec_invalid_axes_ids,
    _gridspec_region_label,
    _gridspec_region_overlaps,
    _gridspec_region_valid,
    _gridspec_remove_region,
    _gridspec_replace_grid,
    _gridspec_setup_from_subplots,
    _gridspec_update_axis_label,
    _gridspec_valid_axes_ids,
    _subplots_setup_from_gridspec,
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
    _source_data_from_blob,
    _source_data_to_blob,
    _source_label,
    _source_name,
)
from erlab.interactive._figurecomposer._state import (
    FigureAxesSelectionState,
    FigureGridSpecAxesState,
    FigureGridSpecGridState,
    FigureGridSpecSpanState,
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
    _GridSpecLayoutWidget,
    _GridSpecRegionInfo,
    _step_toolbar_button,
)
from erlab.interactive.imagetool import provenance

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


_OPERATION_EDITOR_UPDATE_DELAY_MS = 25
_RETIRED_EDITOR_DRAIN_DELAY_MS = 100
_MIXED_VALUES_TEXT = "(multiple values)"
_MIXED_VALUE = object()
_PERSISTED_SOURCE_MAP_ATTR = "_figure_composer_source_payloads"
_PERSISTED_SOURCE_VAR_PREFIX = "_figure_composer_source_payload_"
_PERSISTED_SOURCE_DIM_PREFIX = "_figure_composer_source_payload_bytes_"


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
        self._operation_editor_update_pending = False
        self._retired_editor_drain_pending = False
        self._operation_multi_select_event = False
        self._operation_list_viewport: QtWidgets.QWidget | None = None
        self._retired_editor_widgets: list[QtWidgets.QWidget] = []
        self._operation_render_errors: dict[str, str] = {}
        self._source_data: dict[str, xr.DataArray] = {}
        self._recipe = recipe or self._default_recipe(data)
        self._active_gridspec_grid_id = self._recipe.setup.gridspec.root.grid_id
        self._gridspec_breadcrumb_buttons: list[QtWidgets.QToolButton] = []
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
        self._remove_operation_list_event_filter()
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
            "Remove the selected recipe step or steps.",
        )
        self.remove_operation_button.clicked.connect(self._remove_current_operation)
        self.duplicate_operation_button = _step_toolbar_button(
            recipe_page,
            "figureComposerDuplicateStepButton",
            "Duplicate",
            "Copy the selected step or steps after the last selected step.",
        )
        self.duplicate_operation_button.clicked.connect(
            self._duplicate_current_operation
        )
        self.move_operation_up_button = _step_toolbar_button(
            recipe_page,
            "figureComposerMoveStepUpButton",
            "Up",
            "Move the selected recipe step or steps earlier.",
        )
        self.move_operation_up_button.clicked.connect(self._move_current_operation_up)
        self.move_operation_down_button = _step_toolbar_button(
            recipe_page,
            "figureComposerMoveStepDownButton",
            "Down",
            "Move the selected recipe step or steps later.",
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
        self._operation_list_viewport = self.operation_list.viewport()
        if self._operation_list_viewport is not None:
            self._operation_list_viewport.installEventFilter(self)
        self.operation_list.currentRowChanged.connect(self._operation_selection_changed)
        self.operation_list.itemSelectionChanged.connect(
            self._operation_selection_changed
        )
        self.operation_list.itemChanged.connect(self._operation_item_changed)
        self.operation_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
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
        self.gridspec_axes_list = QtWidgets.QListWidget(self.target_axes_page)
        self.gridspec_axes_list.setObjectName("figureComposerGridSpecAxesList")
        self.gridspec_axes_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.gridspec_axes_list.setAlternatingRowColors(True)
        self.gridspec_axes_list.setUniformItemSizes(True)
        self.gridspec_axes_list.setToolTip(
            "Axes created in the GridSpec layout.\n"
            "Use Shift or Command/Ctrl to select multiple targets."
        )
        self.gridspec_axes_list.itemSelectionChanged.connect(
            self._gridspec_axes_selection_changed
        )
        target_axes_layout.addWidget(self.gridspec_axes_list)
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
        self.layout_mode_combo = QtWidgets.QComboBox(layout_page)
        self.layout_mode_combo.setObjectName("figureComposerLayoutModeCombo")
        self.layout_mode_combo.addItems(["subplots", "gridspec"])
        self.layout_mode_combo.setToolTip(
            "Choose regular plt.subplots layout or custom GridSpec regions."
        )

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

        mode_label = QtWidgets.QLabel("Layout mode", layout_page)
        mode_label.setObjectName("figureComposerLayoutModeControls")
        mode_label.setBuddy(self.layout_mode_combo)
        mode_label.setToolTip(self.layout_mode_combo.toolTip())
        mode_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        setup_layout.addWidget(mode_label, 0, 0, 1, 2)
        setup_layout.addWidget(self.layout_mode_combo, 0, 2, 1, 3)

        add_grid_pair_row(
            1,
            "Grid",
            "figureComposerGridControls",
            "Subplot grid or active GridSpec grid.",
            "Rows",
            self.nrows_spin,
            "Number of rows in the subplot grid or active GridSpec grid.",
            "Columns",
            self.ncols_spin,
            "Number of columns in the subplot grid or active GridSpec grid.",
        )
        add_grid_pair_row(
            2,
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
            3,
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
        setup_layout.addWidget(layout_row_label, 4, 0, 1, 2)
        setup_layout.addWidget(self.layout_combo, 4, 2, 1, 3)
        add_grid_pair_row(
            5,
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
            6,
            "Ratios",
            "figureComposerRatioControls",
            "Optional width and height ratios for subplots or active GridSpec grid.",
            "Widths",
            self.width_ratios_edit,
            "Optional width ratios, one positive number per column.",
            "Heights",
            self.height_ratios_edit,
            "Optional height ratios, one positive number per row.",
        )

        self.gridspec_editor_widget = QtWidgets.QWidget(layout_page)
        self.gridspec_editor_widget.setObjectName("figureComposerGridSpecEditor")
        gridspec_editor_layout = QtWidgets.QVBoxLayout(self.gridspec_editor_widget)
        gridspec_editor_layout.setContentsMargins(0, 0, 0, 0)
        gridspec_editor_layout.setSpacing(4)
        self.gridspec_breadcrumb_widget = QtWidgets.QWidget(self.gridspec_editor_widget)
        self.gridspec_breadcrumb_layout = QtWidgets.QHBoxLayout(
            self.gridspec_breadcrumb_widget
        )
        self.gridspec_breadcrumb_layout.setContentsMargins(0, 0, 0, 0)
        self.gridspec_breadcrumb_layout.setSpacing(2)
        gridspec_editor_layout.addWidget(self.gridspec_breadcrumb_widget)

        gridspec_action_layout = QtWidgets.QHBoxLayout()
        self.gridspec_region_kind_combo = QtWidgets.QComboBox(
            self.gridspec_editor_widget
        )
        self.gridspec_region_kind_combo.setObjectName(
            "figureComposerGridSpecRegionKindCombo"
        )
        self.gridspec_region_kind_combo.addItem("Axes", "axes")
        self.gridspec_region_kind_combo.addItem("Nested Grid", "grid")
        self.gridspec_region_kind_combo.setToolTip(
            "Kind of region created by the next drag gesture."
        )
        self.gridspec_parent_grid_button = _step_toolbar_button(
            self.gridspec_editor_widget,
            "figureComposerGridSpecParentButton",
            "Parent",
            "Return to the parent GridSpec grid.",
        )
        self.gridspec_parent_grid_button.clicked.connect(
            self._gridspec_open_parent_grid
        )
        draw_label = QtWidgets.QLabel("Draw", self.gridspec_editor_widget)
        draw_label.setBuddy(self.gridspec_region_kind_combo)
        gridspec_action_layout.addWidget(draw_label)
        gridspec_action_layout.addWidget(self.gridspec_region_kind_combo)
        gridspec_action_layout.addWidget(self.gridspec_parent_grid_button)
        gridspec_action_layout.addStretch(1)
        gridspec_editor_layout.addLayout(gridspec_action_layout)

        self.gridspec_layout_widget = _GridSpecLayoutWidget(self.gridspec_editor_widget)
        self.gridspec_layout_widget.set_creation_kind("axes")
        self.gridspec_layout_widget.sigRegionCreated.connect(
            self._gridspec_region_created
        )
        self.gridspec_layout_widget.sigRegionChanged.connect(
            self._gridspec_region_changed
        )
        self.gridspec_layout_widget.sigRegionSelected.connect(
            self._gridspec_region_selected
        )
        self.gridspec_layout_widget.sigNestedGridActivated.connect(
            self._gridspec_open_grid
        )
        self.gridspec_region_kind_combo.currentIndexChanged.connect(
            self._gridspec_region_kind_changed
        )
        gridspec_editor_layout.addWidget(self.gridspec_layout_widget)

        gridspec_region_layout = QtWidgets.QHBoxLayout()
        self.gridspec_region_label_edit = QtWidgets.QLineEdit(
            self.gridspec_editor_widget
        )
        self.gridspec_region_label_edit.setObjectName(
            "figureComposerGridSpecRegionLabelEdit"
        )
        self.gridspec_region_label_edit.setPlaceholderText("Optional axes name")
        self.gridspec_region_label_edit.setToolTip(
            "Optional label used in the GUI and generated axis variable name."
        )
        self.gridspec_region_label_edit.editingFinished.connect(
            self._gridspec_region_label_changed
        )
        self.gridspec_open_grid_button = _step_toolbar_button(
            self.gridspec_editor_widget,
            "figureComposerGridSpecOpenButton",
            "Open",
            "Edit the selected nested grid.",
        )
        self.gridspec_open_grid_button.clicked.connect(
            self._gridspec_open_selected_grid
        )
        self.gridspec_delete_region_button = _step_toolbar_button(
            self.gridspec_editor_widget,
            "figureComposerGridSpecDeleteButton",
            "Delete",
            "Delete the selected axes or nested grid region.",
        )
        self.gridspec_delete_region_button.clicked.connect(
            self._gridspec_delete_selected_region
        )
        name_label = QtWidgets.QLabel("Name", self.gridspec_editor_widget)
        name_label.setBuddy(self.gridspec_region_label_edit)
        gridspec_region_layout.addWidget(name_label)
        gridspec_region_layout.addWidget(self.gridspec_region_label_edit, 1)
        gridspec_region_layout.addWidget(self.gridspec_open_grid_button)
        gridspec_region_layout.addWidget(self.gridspec_delete_region_button)
        gridspec_editor_layout.addLayout(gridspec_region_layout)

        self.gridspec_status_label = QtWidgets.QLabel(self.gridspec_editor_widget)
        self.gridspec_status_label.setObjectName("figureComposerGridSpecStatus")
        self.gridspec_status_label.setWordWrap(True)
        gridspec_editor_layout.addWidget(self.gridspec_status_label)
        setup_layout.addWidget(self.gridspec_editor_widget, 7, 0, 1, 5)
        setup_layout.setRowStretch(8, 1)

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
        self.layout_mode_combo.currentTextChanged.connect(self._layout_mode_changed)
        self.layout_combo.currentTextChanged.connect(self._setup_controls_changed)
        self.sharex_combo.currentTextChanged.connect(self._setup_controls_changed)
        self.sharey_combo.currentTextChanged.connect(self._setup_controls_changed)

        self.setCentralWidget(root)
        self.setWindowTitle("Figure Composer")

    def _apply_recipe_to_controls(self) -> None:
        self._updating_controls = True
        try:
            setup = self._recipe.setup
            self._set_combo_value(self.layout_mode_combo, setup.layout_mode)
            self._sync_active_grid_controls(setup)
            self.width_spin.setValue(setup.figsize[0])
            self.height_spin.setValue(setup.figsize[1])
            self._sync_size_mm_controls(setup.figsize)
            self._set_combo_value(self.layout_combo, setup.layout or "none")
            self._set_combo_value(self.sharex_combo, str(setup.sharex))
            self._set_combo_value(self.sharey_combo, str(setup.sharey))
            self._refresh_source_list()
            self._rebuild_axes_grid()
            self._refresh_operation_list()
            if self.operation_list.count() and self.operation_list.currentRow() < 0:
                self.operation_list.setCurrentRow(0)
            self._update_operation_editor()
        finally:
            self._updating_controls = False

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

    def _sync_active_grid_controls(self, setup: FigureSubplotsState) -> None:
        if setup.layout_mode == "gridspec":
            active_grid = _gridspec_grid_by_id(setup, self._active_gridspec_grid_id)
            if active_grid is None:
                self._active_gridspec_grid_id = setup.gridspec.root.grid_id
                active_grid = setup.gridspec.root
            self.nrows_spin.setValue(active_grid.nrows)
            self.ncols_spin.setValue(active_grid.ncols)
            self.width_ratios_edit.setText(_format_tuple(active_grid.width_ratios))
            self.height_ratios_edit.setText(_format_tuple(active_grid.height_ratios))
        else:
            self.nrows_spin.setValue(setup.nrows)
            self.ncols_spin.setValue(setup.ncols)
            self.width_ratios_edit.setText(_format_tuple(setup.width_ratios))
            self.height_ratios_edit.setText(_format_tuple(setup.height_ratios))
        self.gridspec_editor_widget.setVisible(setup.layout_mode == "gridspec")
        self.sharex_combo.setEnabled(setup.layout_mode == "subplots")
        self.sharey_combo.setEnabled(setup.layout_mode == "subplots")

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
        self._refresh_gridspec_editor()
        self._sync_axes_selector()

    def _refresh_gridspec_editor(self) -> None:
        setup = self._recipe.setup
        if setup.layout_mode != "gridspec":
            self.gridspec_editor_widget.setVisible(False)
            return
        self.gridspec_editor_widget.setVisible(True)
        grid = _gridspec_grid_by_id(setup, self._active_gridspec_grid_id)
        if grid is None:
            self._active_gridspec_grid_id = setup.gridspec.root.grid_id
            grid = setup.gridspec.root
        regions = [
            _GridSpecRegionInfo(
                axis.axes_id,
                "axes",
                axis.span,
                axis.label.strip() or _gridspec_axis_display_name(setup, axis.axes_id),
                _gridspec_region_valid(grid, axis.span),
            )
            for axis in grid.axes
        ]
        regions.extend(
            _GridSpecRegionInfo(
                child.grid_id,
                "grid",
                child.span,
                child.label.strip() or "Grid",
                _gridspec_region_valid(grid, child.span),
            )
            for child in grid.child_grids
            if child.span is not None
        )
        self.gridspec_layout_widget.set_grid(grid.nrows, grid.ncols, regions)
        self._refresh_gridspec_breadcrumbs()
        self._refresh_gridspec_region_controls()
        self._refresh_gridspec_axes_list()
        self._refresh_gridspec_status(grid)

    def _refresh_gridspec_status(self, grid: FigureGridSpecGridState) -> None:
        invalid_regions = [
            _gridspec_region_label(self._recipe.setup, grid, axis.axes_id)
            for axis in grid.axes
            if not _gridspec_region_valid(grid, axis.span)
        ]
        invalid_regions.extend(
            _gridspec_region_label(self._recipe.setup, grid, child.grid_id)
            for child in grid.child_grids
            if child.span is None or not _gridspec_region_valid(grid, child.span)
        )
        if invalid_regions:
            self.gridspec_status_label.setText(
                "Some regions are outside the active grid: "
                + ", ".join(invalid_regions)
                + ". Increase rows/columns or delete and redraw them."
            )
            return
        if _gridspec_has_invalid_regions(self._recipe.setup.gridspec.root):
            self.gridspec_status_label.setText(
                "A nested grid contains regions outside its bounds."
            )
            return
        kind = self.gridspec_region_kind_combo.currentText()
        self.gridspec_status_label.setText(
            f"Drag cells to create {kind.lower()} regions. "
            "Double-click a nested grid to edit it."
        )

    def _refresh_gridspec_breadcrumbs(self) -> None:
        for button in self._gridspec_breadcrumb_buttons:
            button.deleteLater()
        self._gridspec_breadcrumb_buttons.clear()
        while self.gridspec_breadcrumb_layout.count():
            item = self.gridspec_breadcrumb_layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        path = _gridspec_grid_path(self._recipe.setup, self._active_gridspec_grid_id)
        for index, grid in enumerate(path):
            if index:
                separator = QtWidgets.QLabel(">", self.gridspec_breadcrumb_widget)
                self.gridspec_breadcrumb_layout.addWidget(separator)
            button = QtWidgets.QToolButton(self.gridspec_breadcrumb_widget)
            button.setText(grid.label.strip() or ("Root" if index == 0 else "Grid"))
            button.setToolTip("Open this GridSpec grid.")
            button.clicked.connect(
                lambda _checked=False, grid_id=grid.grid_id: self._gridspec_open_grid(
                    grid_id
                )
            )
            self._gridspec_breadcrumb_buttons.append(button)
            self.gridspec_breadcrumb_layout.addWidget(button)
        self.gridspec_breadcrumb_layout.addStretch(1)
        self.gridspec_parent_grid_button.setEnabled(len(path) > 1)

    def _refresh_gridspec_region_controls(self) -> None:
        region_id = self.gridspec_layout_widget.selected_region_id()
        grid = _gridspec_grid_by_id(self._recipe.setup, self._active_gridspec_grid_id)
        label = ""
        kind = ""
        if grid is not None and region_id:
            for axis in grid.axes:
                if axis.axes_id == region_id:
                    label = axis.label
                    kind = "axes"
                    break
            if not kind:
                for child in grid.child_grids:
                    if child.grid_id == region_id:
                        label = child.label
                        kind = "grid"
                        break
        blocker = QtCore.QSignalBlocker(self.gridspec_region_label_edit)
        self.gridspec_region_label_edit.setText(label)
        del blocker
        has_region = bool(kind)
        self.gridspec_region_label_edit.setPlaceholderText(
            "Optional axes name" if kind != "grid" else "Optional grid name"
        )
        self.gridspec_region_label_edit.setEnabled(has_region)
        self.gridspec_delete_region_button.setEnabled(has_region)
        self.gridspec_open_grid_button.setEnabled(kind == "grid")
        self.gridspec_open_grid_button.setToolTip(
            "Edit the selected nested grid."
            if kind == "grid"
            else "Select a nested grid region to edit it."
        )
        self.gridspec_delete_region_button.setToolTip(
            "Delete the selected axes or nested grid region."
            if has_region
            else "Select an axes or nested grid region to delete it."
        )

    def _refresh_gridspec_axes_list(self) -> None:
        selected_ids: set[str] = set()
        current = self._current_operation()
        if current is not None and _registry.spec_for(current[1].kind).uses_axes(
            current[1]
        ):
            selected_ids = set(current[1].axes.axes_ids)
        blocker = QtCore.QSignalBlocker(self.gridspec_axes_list)
        self.gridspec_axes_list.clear()
        valid_axis_ids = set(
            _gridspec_valid_axes_ids(
                self._recipe.setup,
                _gridspec_all_axes_ids(self._recipe.setup),
            )
        )
        for axes_id in _gridspec_all_axes_ids(self._recipe.setup):
            is_valid = axes_id in valid_axis_ids
            item = QtWidgets.QListWidgetItem(
                _gridspec_axis_display_name(self._recipe.setup, axes_id)
                + ("" if is_valid else " (outside grid)")
            )
            item.setData(QtCore.Qt.ItemDataRole.UserRole, axes_id)
            item.setData(QtCore.Qt.ItemDataRole.UserRole + 1, is_valid)
            if is_valid:
                item.setToolTip("Select this GridSpec axes as an operation target.")
            else:
                item.setForeground(QtGui.QBrush(QtGui.QColor("darkRed")))
                item.setToolTip(
                    "This axes is outside its current GridSpec grid. "
                    "Increase the grid size or drop removed axes."
                )
            self.gridspec_axes_list.addItem(item)
            item.setSelected(axes_id in selected_ids)
        del blocker

    def _sync_axes_selector(self) -> None:
        current = self._current_operation()
        selected_axes = set()
        selected_axes_ids: tuple[str, ...] = ()
        expression = ""
        invalid_axes: tuple[tuple[int, int], ...] = ()
        invalid_axes_ids: tuple[str, ...] = ()
        if current is not None:
            _, operation = current
            spec = _registry.spec_for(operation.kind)
            if spec.uses_axes(operation):
                if self._recipe.setup.layout_mode == "gridspec":
                    selected_axes_ids = _gridspec_valid_axes_ids(
                        self._recipe.setup, operation.axes.axes_ids
                    )
                    invalid_axes_ids = _gridspec_invalid_axes_ids(
                        self._recipe.setup, operation.axes.axes_ids
                    )
                else:
                    selected_axes = set(operation.axes.valid_axes(self._recipe.setup))
                    invalid_axes = operation.axes.invalid_axes(self._recipe.setup)
                expression = operation.axes.expression

        self._updating_controls = True
        try:
            grid_mode = self._recipe.setup.layout_mode == "gridspec"
            self.axes_selector.setVisible(not grid_mode)
            self.axes_expression_edit.setVisible(not grid_mode)
            self.gridspec_axes_list.setVisible(grid_mode)
            self.axes_selector.set_selected_axes(tuple(sorted(selected_axes)))
            self._refresh_gridspec_axes_list()
            if self.axes_expression_edit.text() != expression:
                blocker = QtCore.QSignalBlocker(self.axes_expression_edit)
                self.axes_expression_edit.setText(expression)
                del blocker
            self.keep_valid_axes_button.setEnabled(
                bool(invalid_axes_ids) if grid_mode else bool(invalid_axes)
            )
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
            elif invalid_axes_ids:
                self.target_axes_status_label.setText(
                    "Target axes removed by the current GridSpec layout: "
                    + ", ".join(invalid_axes_ids)
                )
            else:
                if grid_mode:
                    axes_text = ", ".join(
                        _gridspec_axis_display_names(
                            self._recipe.setup, selected_axes_ids
                        )
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
        selected_ids = self._selected_operation_ids()
        if not selected_ids and current_id is not None:
            selected_ids = {current_id}
        self.operation_list.blockSignals(True)
        try:
            self.operation_list.clear()
            for operation in self._recipe.operations:
                render_error = self._operation_render_errors.get(operation.operation_id)
                text = self._operation_display_text(operation)
                if render_error is not None:
                    text = f"{text} (render error)"
                item = QtWidgets.QListWidgetItem(text)
                item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    QtCore.Qt.CheckState.Checked
                    if operation.enabled
                    else QtCore.Qt.CheckState.Unchecked
                )
                item.setData(QtCore.Qt.ItemDataRole.UserRole, operation.operation_id)
                tooltip = self._operation_tooltip(operation)
                if render_error is not None:
                    tooltip = f"{tooltip}\n\nRender error: {render_error}"
                item.setToolTip(tooltip)
                if (
                    self._operation_has_invalid_axes(operation)
                    or render_error is not None
                ):
                    item.setForeground(QtGui.QBrush(QtGui.QColor("darkRed")))
                self.operation_list.addItem(item)
                if operation.operation_id in selected_ids:
                    item.setSelected(True)
                if operation.operation_id == current_id:
                    self.operation_list.setCurrentItem(item)
        finally:
            self.operation_list.blockSignals(False)

    def _set_current_operation_row_silent(
        self, index: int, *, preserve_selection: bool = True
    ) -> None:
        selected_ids = self._selected_operation_ids() if preserve_selection else set()
        was_blocked = self.operation_list.blockSignals(True)
        try:
            self.operation_list.setCurrentRow(index)
            if preserve_selection and selected_ids:
                self._set_selected_operation_ids_silent(selected_ids)
        finally:
            self.operation_list.blockSignals(was_blocked)

    def _operation_id_for_item(self, item: QtWidgets.QListWidgetItem) -> str | None:
        operation_id = item.data(QtCore.Qt.ItemDataRole.UserRole)
        return operation_id if isinstance(operation_id, str) else None

    def _selected_operation_ids(self) -> set[str]:
        return {
            operation_id
            for item in self.operation_list.selectedItems()
            if (operation_id := self._operation_id_for_item(item)) is not None
        }

    def _set_selected_operation_ids_silent(self, operation_ids: set[str]) -> None:
        was_blocked = self.operation_list.blockSignals(True)
        try:
            for row in range(self.operation_list.count()):
                item = self.operation_list.item(row)
                if item is None:
                    continue
                item.setSelected(self._operation_id_for_item(item) in operation_ids)
        finally:
            self.operation_list.blockSignals(was_blocked)

    def _operation_display_text(self, operation: FigureOperationState) -> str:
        return _registry.spec_for(operation.kind).display_text(self, operation)

    def _operation_tooltip(self, operation: FigureOperationState) -> str:
        return _registry.spec_for(operation.kind).tooltip(self, operation)

    def _set_operation_render_errors(self, errors: Mapping[str, str]) -> None:
        render_errors = dict(errors)
        if render_errors == self._operation_render_errors:
            return
        self._operation_render_errors = render_errors
        self._refresh_operation_list()
        self._refresh_step_section_button_texts()
        current = self._current_operation()
        self._update_source_status(current[1] if current is not None else None)

    def _axes_target_text(self, selection: FigureAxesSelectionState) -> str:
        if selection.expression:
            return selection.expression
        if self._recipe.setup.layout_mode == "gridspec":
            invalid_ids = _gridspec_invalid_axes_ids(
                self._recipe.setup, selection.axes_ids
            )
            if invalid_ids:
                return "removed axes " + ", ".join(invalid_ids)
            valid_ids = _gridspec_valid_axes_ids(self._recipe.setup, selection.axes_ids)
            if not valid_ids:
                return "none"
            return ", ".join(
                _gridspec_axis_display_names(self._recipe.setup, valid_ids)
            )
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

    def _axes_selection_has_invalid_target(
        self, selection: FigureAxesSelectionState
    ) -> bool:
        if selection.expression:
            return False
        if self._recipe.setup.layout_mode == "gridspec":
            if not _gridspec_valid_axes_ids(self._recipe.setup, selection.axes_ids):
                return True
            return bool(
                _gridspec_invalid_axes_ids(self._recipe.setup, selection.axes_ids)
            )
        return bool(selection.invalid_axes(self._recipe.setup)) or not bool(
            selection.valid_axes(self._recipe.setup)
        )

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

    def _selected_operation_indices(self) -> tuple[int, ...]:
        selected_ids = self._selected_operation_ids()
        if not selected_ids:
            current = self._current_operation()
            return () if current is None else (current[0],)
        return tuple(
            index
            for index, operation in enumerate(self._recipe.operations)
            if operation.operation_id in selected_ids
        )

    @staticmethod
    def _operation_editor_schema_key(
        operation: FigureOperationState,
    ) -> tuple[object, ...]:
        if operation.kind.value == "method":
            return (operation.kind, operation.method_family, operation.method_name)
        return (operation.kind,)

    def _selected_operations_are_compatible(self) -> bool:
        indices = self._selected_operation_indices()
        if len(indices) <= 1:
            return True
        keys = {
            self._operation_editor_schema_key(self._recipe.operations[index])
            for index in indices
        }
        return len(keys) == 1

    def _editable_operation_indices(self) -> tuple[int, ...]:
        indices = self._selected_operation_indices()
        if len(indices) <= 1:
            return indices
        return indices if self._selected_operations_are_compatible() else ()

    def _editable_operations(self) -> tuple[tuple[int, FigureOperationState], ...]:
        return tuple(
            (index, self._recipe.operations[index])
            for index in self._editable_operation_indices()
        )

    def _batch_value(
        self,
        operation: FigureOperationState,
        getter: Callable[[FigureOperationState], typing.Any],
    ) -> typing.Any:
        editable = self._editable_operations()
        if len(editable) <= 1:
            return getter(operation)
        values = [getter(target) for _index, target in editable]
        first = values[0]
        if all(value == first for value in values[1:]):
            return first
        return _MIXED_VALUE

    def _batch_is_mixed(
        self,
        operation: FigureOperationState,
        getter: Callable[[FigureOperationState], typing.Any],
    ) -> bool:
        return self._batch_value(operation, getter) is _MIXED_VALUE

    def _batch_text(
        self,
        operation: FigureOperationState,
        getter: Callable[[FigureOperationState], typing.Any],
        formatter: Callable[[typing.Any], str],
    ) -> tuple[str, bool]:
        value = self._batch_value(operation, getter)
        if value is _MIXED_VALUE:
            return "", True
        return formatter(value), False

    def _batch_combo_text(
        self,
        operation: FigureOperationState,
        getter: Callable[[FigureOperationState], typing.Any],
        formatter: Callable[[typing.Any], str] = str,
    ) -> str | None:
        value = self._batch_value(operation, getter)
        if value is _MIXED_VALUE:
            return None
        return formatter(value)

    def _batch_options_match(
        self,
        operation: FigureOperationState,
        options_getter: Callable[[FigureOperationState], Sequence[str]],
    ) -> bool:
        editable = self._editable_operations()
        if len(editable) <= 1:
            return True
        expected = tuple(options_getter(operation))
        return all(
            tuple(options_getter(target)) == expected for _index, target in editable
        )

    @staticmethod
    def _line_edit_batch_unchanged(edit: QtWidgets.QLineEdit) -> bool:
        return bool(edit.property("batch_mixed")) and not edit.isModified()

    @staticmethod
    def _apply_mixed_line_edit(edit: QtWidgets.QLineEdit, mixed: bool) -> None:
        if mixed:
            edit.setPlaceholderText(_MIXED_VALUES_TEXT)
            edit.setProperty("batch_mixed", True)
            edit.setModified(False)
        else:
            edit.setProperty("batch_mixed", False)

    @staticmethod
    def _plain_text_batch_unchanged(edit: QtWidgets.QPlainTextEdit) -> bool:
        document = edit.document()
        return (
            bool(edit.property("batch_mixed"))
            and document is not None
            and not document.isModified()
        )

    @staticmethod
    def _apply_mixed_plain_text_edit(
        edit: QtWidgets.QPlainTextEdit, mixed: bool
    ) -> None:
        if mixed:
            edit.setPlaceholderText(_MIXED_VALUES_TEXT)
            edit.setProperty("batch_mixed", True)
            document = edit.document()
            if document is not None:
                document.setModified(False)
        else:
            edit.setProperty("batch_mixed", False)

    @staticmethod
    def _set_combo_mixed_placeholder(combo: QtWidgets.QComboBox) -> None:
        combo.insertItem(0, _MIXED_VALUES_TEXT)
        item = typing.cast("typing.Any", combo.model()).item(0)
        if item is not None:
            item.setEnabled(False)
        combo.setCurrentIndex(0)

    @staticmethod
    def _mixed_combo_text(text: str) -> bool:
        return text == _MIXED_VALUES_TEXT

    def _update_operations(
        self,
        updater: Callable[[int, FigureOperationState], FigureOperationState],
        *,
        render: bool = True,
        rebuild_editor: bool = False,
        defer_editor_rebuild: bool = False,
        sync_axes: bool = True,
    ) -> None:
        editable = self._editable_operations()
        if not editable:
            return
        current = self._current_operation()
        operations = list(self._recipe.operations)
        for index, operation in editable:
            operations[index] = updater(index, operation)
        self._recipe = self._recipe.model_copy(update={"operations": tuple(operations)})
        self._refresh_operation_list()
        if current is not None:
            self._set_current_operation_row_silent(current[0])
        current_operation = self._current_operation()
        if sync_axes:
            self._sync_axes_selector()
        self._update_step_action_buttons()
        self._refresh_step_section_button_texts()
        self._update_source_status(current_operation[1] if current_operation else None)
        if rebuild_editor:
            if defer_editor_rebuild:
                self._queue_operation_editor_update()
            else:
                self._update_operation_editor_safely()
        if render:
            _rendering._render_preview(self)
            self.sigInfoChanged.emit()

    def _replace_operation(
        self,
        index: int,
        operation: FigureOperationState,
        *,
        render: bool = True,
        rebuild_editor: bool = False,
        defer_editor_rebuild: bool = False,
        sync_axes: bool = True,
    ) -> None:
        operations = list(self._recipe.operations)
        operations[index] = operation
        self._recipe = self._recipe.model_copy(update={"operations": tuple(operations)})
        self._refresh_operation_list()
        self._set_current_operation_row_silent(index)
        if sync_axes:
            self._sync_axes_selector()
        self._update_step_action_buttons()
        self._refresh_step_section_button_texts()
        self._update_source_status(operation)
        if rebuild_editor:
            if defer_editor_rebuild:
                self._queue_operation_editor_update()
            else:
                self._update_operation_editor_safely()
        if render:
            _rendering._render_preview(self)
            self.sigInfoChanged.emit()

    def _update_current_operation(self, **updates: typing.Any) -> None:
        if self._updating_controls:
            return
        self._update_operations(
            lambda _index, operation: operation.model_copy(update=updates)
        )

    def _update_current_operation_rebuild(self, **updates: typing.Any) -> None:
        if self._updating_controls:
            return
        self._update_operations(
            lambda _index, operation: operation.model_copy(update=updates),
            rebuild_editor=True,
            defer_editor_rebuild=True,
        )

    def _update_current_operation_in_place(self, **updates: typing.Any) -> None:
        if self._updating_controls:
            return
        self._update_operations(
            lambda _index, operation: operation.model_copy(update=updates),
            render=False,
        )
        self._update_step_action_buttons()
        self._refresh_step_section_button_texts()
        erlab.interactive.utils.single_shot(
            self, 0, lambda: _rendering._render_preview(self)
        )
        self.sigInfoChanged.emit()

    def _update_step_action_buttons(self) -> None:
        indices = self._selected_operation_indices()
        if not indices:
            self.remove_operation_button.setEnabled(False)
            self.duplicate_operation_button.setEnabled(False)
            self.move_operation_up_button.setEnabled(False)
            self.move_operation_down_button.setEnabled(False)
            return
        index_set = set(indices)
        self.remove_operation_button.setEnabled(True)
        self.duplicate_operation_button.setEnabled(True)
        self.move_operation_up_button.setEnabled(
            any(index > 0 and index - 1 not in index_set for index in indices)
        )
        self.move_operation_down_button.setEnabled(
            any(
                index < len(self._recipe.operations) - 1 and index + 1 not in index_set
                for index in indices
            )
        )

    @QtCore.Slot()
    def _setup_controls_changed(self) -> None:
        if self._updating_controls:
            return
        try:
            width_ratios = self._ratio_tuple_from_text(self.width_ratios_edit.text())
            height_ratios = self._ratio_tuple_from_text(self.height_ratios_edit.text())
            if self._recipe.setup.layout_mode == "gridspec":
                setup = self._recipe.setup.model_copy(
                    update={
                        "figsize": (self.width_spin.value(), self.height_spin.value()),
                        "dpi": self._recipe.setup.dpi,
                        "layout": self._layout_combo_value(),
                    }
                )

                def update_grid(
                    grid: FigureGridSpecGridState,
                ) -> FigureGridSpecGridState:
                    return grid.model_copy(
                        update={
                            "nrows": self.nrows_spin.value(),
                            "ncols": self.ncols_spin.value(),
                            "width_ratios": width_ratios,
                            "height_ratios": height_ratios,
                        }
                    )

                setup = _gridspec_replace_grid(
                    setup, self._active_gridspec_grid_id, update_grid
                )
            else:
                setup = FigureSubplotsState(
                    layout_mode="subplots",
                    nrows=self.nrows_spin.value(),
                    ncols=self.ncols_spin.value(),
                    figsize=(self.width_spin.value(), self.height_spin.value()),
                    dpi=self._recipe.setup.dpi,
                    layout=self._layout_combo_value(),
                    sharex=self._combo_bool_or_text(self.sharex_combo),
                    sharey=self._combo_bool_or_text(self.sharey_combo),
                    width_ratios=width_ratios,
                    height_ratios=height_ratios,
                    gridspec=self._recipe.setup.gridspec,
                )
        except ValueError:
            return
        if setup == self._recipe.setup:
            return
        self._recipe = self._recipe.model_copy(update={"setup": setup})
        self._updating_controls = True
        try:
            self._sync_size_mm_controls(setup.figsize)
            self._sync_active_grid_controls(setup)
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

    @QtCore.Slot()
    def _layout_mode_changed(self) -> None:
        if self._updating_controls:
            return
        mode = self.layout_mode_combo.currentText()
        if mode == self._recipe.setup.layout_mode:
            return
        if mode == "gridspec":
            setup = _gridspec_setup_from_subplots(self._recipe.setup)
            axes_ids = _gridspec_all_axes_ids(setup)
            axis_id_by_tuple = {
                (row, col): axes_ids[row * setup.ncols + col]
                for row in range(setup.nrows)
                for col in range(setup.ncols)
                if row * setup.ncols + col < len(axes_ids)
            }
            operations = tuple(
                operation.model_copy(
                    update={
                        "axes": operation.axes.model_copy(
                            update={
                                "axes_ids": tuple(
                                    axis_id_by_tuple[axis]
                                    for axis in operation.axes.axes
                                    if axis in axis_id_by_tuple
                                ),
                                "expression": "",
                            }
                        )
                    }
                )
                if _registry.spec_for(operation.kind).uses_axes(operation)
                else operation
                for operation in self._recipe.operations
            )
            self._active_gridspec_grid_id = setup.gridspec.root.grid_id
        else:
            setup = _subplots_setup_from_gridspec(self._recipe.setup)
            operations = tuple(
                operation.model_copy(
                    update={
                        "axes": operation.axes.model_copy(
                            update={
                                "axes": _gridspec_axes_subplot_targets(
                                    self._recipe.setup, operation.axes.axes_ids
                                )
                                or operation.axes.axes
                                or ((0, 0),),
                                "expression": "",
                            }
                        )
                    }
                )
                if _registry.spec_for(operation.kind).uses_axes(operation)
                else operation
                for operation in self._recipe.operations
            )
        self._recipe = self._recipe.model_copy(
            update={"setup": setup, "operations": operations}
        )
        self._updating_controls = True
        try:
            self._sync_active_grid_controls(setup)
        finally:
            self._updating_controls = False
        self._rebuild_axes_grid()
        self._refresh_operation_list()
        self._update_operation_editor()
        _rendering._render_preview(self)
        self.sigInfoChanged.emit()

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
    def _gridspec_axes_selection_changed(self) -> None:
        if self._updating_controls:
            return
        current = self._current_operation()
        if current is None:
            return
        if not _registry.spec_for(current[1].kind).uses_axes(current[1]):
            return
        axes_ids = tuple(
            typing.cast("str", item.data(QtCore.Qt.ItemDataRole.UserRole))
            for item in self.gridspec_axes_list.selectedItems()
        )
        index, operation = current
        selection = operation.axes.model_copy(
            update={"axes_ids": axes_ids, "expression": ""}
        )
        self._replace_operation(
            index,
            operation.model_copy(update={"axes": selection}),
            sync_axes=False,
        )
        erlab.interactive.utils.single_shot(self, 0, self._sync_axes_selector)

    @QtCore.Slot()
    @QtCore.Slot(int)
    def _gridspec_region_kind_changed(self, _index: int | None = None) -> None:
        kind = typing.cast(
            'typing.Literal["axes", "grid"]',
            self.gridspec_region_kind_combo.currentData(),
        )
        self.gridspec_layout_widget.set_creation_kind(kind)
        grid = _gridspec_grid_by_id(self._recipe.setup, self._active_gridspec_grid_id)
        if grid is not None:
            self._refresh_gridspec_status(grid)

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

    @QtCore.Slot()
    @QtCore.Slot(int)
    def _operation_selection_changed(self, _row: int | None = None) -> None:
        if _row is not None and not self._operation_multi_select_event:
            item = self.operation_list.item(_row)
            if item is not None:
                operation_id = self._operation_id_for_item(item)
                if operation_id is not None:
                    self._set_selected_operation_ids_silent({operation_id})
        self._sync_axes_selector()
        self._update_operation_editor()

    def eventFilter(
        self, watched: QtCore.QObject | None, event: QtCore.QEvent | None
    ) -> bool:
        operation_list_viewport = self._operation_list_viewport
        if (
            operation_list_viewport is not None
            and watched is operation_list_viewport
            and event is not None
            and event.type()
            in {
                QtCore.QEvent.Type.MouseButtonPress,
                QtCore.QEvent.Type.MouseButtonDblClick,
                QtCore.QEvent.Type.KeyPress,
            }
        ):
            input_event = typing.cast("QtGui.QInputEvent", event)
            self._operation_multi_select_event = (
                self._operation_modifiers_enable_multi_selection(
                    input_event.modifiers()
                )
            )
            erlab.interactive.utils.single_shot(
                self, 0, self._clear_operation_multi_select_event
            )
        return super().eventFilter(watched, event)

    def _remove_operation_list_event_filter(self) -> None:
        viewport = self._operation_list_viewport
        self._operation_list_viewport = None
        self._operation_multi_select_event = False
        if viewport is not None and erlab.interactive.utils.qt_is_valid(self, viewport):
            viewport.removeEventFilter(self)

    def _clear_operation_multi_select_event(self) -> None:
        if not erlab.interactive.utils.qt_is_valid(self):
            return
        self._operation_multi_select_event = False

    @staticmethod
    def _operation_modifiers_enable_multi_selection(
        modifiers: QtCore.Qt.KeyboardModifier,
    ) -> bool:
        multi_modifiers = (
            QtCore.Qt.KeyboardModifier.ShiftModifier
            | QtCore.Qt.KeyboardModifier.ControlModifier
            | QtCore.Qt.KeyboardModifier.MetaModifier
        )
        return bool(modifiers & multi_modifiers)

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
        if self._recipe.setup.layout_mode == "gridspec":
            selection = operation.axes.model_copy(
                update={
                    "axes_ids": _gridspec_all_axes_ids(self._recipe.setup),
                    "expression": "",
                }
            )
            self._replace_operation(
                index, operation.model_copy(update={"axes": selection})
            )
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
        if self._recipe.setup.layout_mode == "gridspec":
            axes_ids = _gridspec_valid_axes_ids(
                self._recipe.setup, operation.axes.axes_ids
            )
            if not axes_ids:
                axes_ids = _gridspec_all_axes_ids(self._recipe.setup)[:1]
            selection = operation.axes.model_copy(
                update={"axes_ids": axes_ids, "expression": ""}
            )
            self._replace_operation(
                index, operation.model_copy(update={"axes": selection})
            )
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
        if self._recipe.setup.layout_mode == "gridspec":
            axes_ids = tuple(
                typing.cast("str", item.data(QtCore.Qt.ItemDataRole.UserRole))
                for item in self.gridspec_axes_list.selectedItems()
            )
            if not axes_ids:
                axes_ids = _gridspec_all_axes_ids(self._recipe.setup)[:1]
            return FigureAxesSelectionState(axes_ids=axes_ids)
        axes = self.axes_selector.selected_axes()
        if not axes:
            axes = ((0, 0),)
        return FigureAxesSelectionState(
            axes=axes,
            expression=self.axes_expression_edit.text().strip(),
        )

    @QtCore.Slot(object, str)
    def _gridspec_region_created(self, span_obj: object, kind_text: str) -> None:
        kind = typing.cast(
            'typing.Literal["axes", "grid"]',
            kind_text,
        )
        self._add_gridspec_region(
            typing.cast("FigureGridSpecSpanState", span_obj), kind
        )

    @QtCore.Slot(str, object)
    def _gridspec_region_changed(self, region_id: str, span_obj: object) -> None:
        span = typing.cast("FigureGridSpecSpanState", span_obj)
        grid = _gridspec_grid_by_id(self._recipe.setup, self._active_gridspec_grid_id)
        if grid is None or not _gridspec_region_valid(grid, span):
            self.gridspec_status_label.setText("Region is outside the active grid.")
            return
        if _gridspec_region_overlaps(
            grid,
            span,
            ignore_axes_id=region_id,
            ignore_grid_id=region_id,
        ):
            self.gridspec_status_label.setText("Regions cannot overlap.")
            return

        def update_grid(grid_state: FigureGridSpecGridState) -> FigureGridSpecGridState:
            axes = tuple(
                axis.model_copy(update={"span": span})
                if axis.axes_id == region_id
                else axis
                for axis in grid_state.axes
            )
            children = tuple(
                child.model_copy(update={"span": span})
                if child.grid_id == region_id
                else child
                for child in grid_state.child_grids
            )
            return grid_state.model_copy(update={"axes": axes, "child_grids": children})

        setup = _gridspec_replace_grid(
            self._recipe.setup, self._active_gridspec_grid_id, update_grid
        )
        self._apply_gridspec_setup(setup, selected_region_id=region_id)

    @QtCore.Slot(str, str)
    def _gridspec_region_selected(self, region_id: str, _kind: str) -> None:
        self.gridspec_layout_widget.set_selected_region(region_id)
        self._refresh_gridspec_region_controls()

    @QtCore.Slot(str)
    def _gridspec_open_grid(self, grid_id: str) -> None:
        if _gridspec_grid_by_id(self._recipe.setup, grid_id) is None:
            return
        self._active_gridspec_grid_id = grid_id
        self._updating_controls = True
        try:
            self._sync_active_grid_controls(self._recipe.setup)
        finally:
            self._updating_controls = False
        self._refresh_gridspec_editor()

    @QtCore.Slot()
    def _gridspec_open_selected_grid(self) -> None:
        region_id = self.gridspec_layout_widget.selected_region_id()
        if region_id:
            self._gridspec_open_grid(region_id)

    @QtCore.Slot()
    def _gridspec_open_parent_grid(self) -> None:
        path = _gridspec_grid_path(self._recipe.setup, self._active_gridspec_grid_id)
        if len(path) > 1:
            self._gridspec_open_grid(path[-2].grid_id)

    @QtCore.Slot()
    def _gridspec_delete_selected_region(self) -> None:
        region_id = self.gridspec_layout_widget.selected_region_id()
        if not region_id:
            return
        setup = _gridspec_remove_region(
            self._recipe.setup, self._active_gridspec_grid_id, region_id
        )
        self._apply_gridspec_setup(setup, selected_region_id="")

    @QtCore.Slot()
    def _gridspec_region_label_changed(self) -> None:
        region_id = self.gridspec_layout_widget.selected_region_id()
        if not region_id:
            return
        label = self.gridspec_region_label_edit.text().strip()
        setup = _gridspec_update_axis_label(self._recipe.setup, region_id, label)
        if setup == self._recipe.setup:
            setup = self._update_gridspec_grid_label(region_id, label)
        self._apply_gridspec_setup(setup, selected_region_id=region_id)

    def _add_gridspec_region(
        self,
        span: FigureGridSpecSpanState,
        kind: typing.Literal["axes", "grid"],
    ) -> None:
        grid = _gridspec_grid_by_id(self._recipe.setup, self._active_gridspec_grid_id)
        if grid is None:
            return
        if not _gridspec_region_valid(grid, span):
            self.gridspec_status_label.setText("Region is outside the active grid.")
            return
        if _gridspec_region_overlaps(grid, span):
            self.gridspec_status_label.setText("Regions cannot overlap.")
            return

        if kind == "axes":
            region = FigureGridSpecAxesState(span=span)
            selected_region_id = region.axes_id

            def update_grid(
                grid_state: FigureGridSpecGridState,
            ) -> FigureGridSpecGridState:
                return grid_state.model_copy(
                    update={"axes": (*grid_state.axes, region)}
                )

        else:
            region = FigureGridSpecGridState(span=span, label="Grid")
            selected_region_id = region.grid_id

            def update_grid(
                grid_state: FigureGridSpecGridState,
            ) -> FigureGridSpecGridState:
                return grid_state.model_copy(
                    update={"child_grids": (*grid_state.child_grids, region)}
                )

        setup = _gridspec_replace_grid(
            self._recipe.setup, self._active_gridspec_grid_id, update_grid
        )
        self._apply_gridspec_setup(setup, selected_region_id=selected_region_id)

    def _update_gridspec_grid_label(
        self, grid_id: str, label: str
    ) -> FigureSubplotsState:
        def update_grid(grid: FigureGridSpecGridState) -> FigureGridSpecGridState:
            return grid.model_copy(update={"label": label})

        return _gridspec_replace_grid(self._recipe.setup, grid_id, update_grid)

    def _apply_gridspec_setup(
        self, setup: FigureSubplotsState, *, selected_region_id: str
    ) -> None:
        self._recipe = self._recipe.model_copy(update={"setup": setup})
        self._updating_controls = True
        try:
            self._sync_active_grid_controls(setup)
        finally:
            self._updating_controls = False
        self._refresh_gridspec_editor()
        self.gridspec_layout_widget.set_selected_region(selected_region_id)
        self._refresh_gridspec_region_controls()
        self._sync_axes_selector()
        self._refresh_operation_list()
        self._refresh_step_section_button_texts()
        self._update_operation_editor()
        _rendering._render_preview(self)
        self.sigInfoChanged.emit()

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
        indices = self._selected_operation_indices()
        if not indices:
            return
        index_set = set(indices)
        operations = [
            operation
            for index, operation in enumerate(self._recipe.operations)
            if index not in index_set
        ]
        self._recipe = self._recipe.model_copy(update={"operations": tuple(operations)})
        selected_ids: set[str] = set()
        current_id: str | None = None
        if operations:
            current_index = min(min(indices), len(operations) - 1)
            current_id = operations[current_index].operation_id
            selected_ids = {current_id}
        self._finish_operation_structure_change(selected_ids, current_id)

    @QtCore.Slot()
    def _duplicate_current_operation(self) -> None:
        indices = self._selected_operation_indices()
        if not indices:
            return
        operations = list(self._recipe.operations)
        duplicates = [
            operations[index].model_copy(
                update={"operation_id": uuid.uuid4().hex},
                deep=True,
            )
            for index in indices
        ]
        insert_index = max(indices) + 1
        operations[insert_index:insert_index] = duplicates
        self._recipe = self._recipe.model_copy(update={"operations": tuple(operations)})
        self._finish_operation_structure_change(
            {operation.operation_id for operation in duplicates},
            duplicates[0].operation_id,
        )

    def _move_current_operation(self, offset: int) -> None:
        indices = self._selected_operation_indices()
        if not indices:
            return
        operations = list(self._recipe.operations)
        index_set = set(indices)
        selected_ids = {operations[index].operation_id for index in indices}
        moved = False
        if offset < 0:
            for index in indices:
                if index > 0 and index - 1 not in index_set:
                    operations[index - 1], operations[index] = (
                        operations[index],
                        operations[index - 1],
                    )
                    index_set.remove(index)
                    index_set.add(index - 1)
                    moved = True
        else:
            for index in reversed(indices):
                if index < len(operations) - 1 and index + 1 not in index_set:
                    operations[index + 1], operations[index] = (
                        operations[index],
                        operations[index + 1],
                    )
                    index_set.remove(index)
                    index_set.add(index + 1)
                    moved = True
        if not moved:
            return
        current = self._current_operation()
        current_id = (
            current[1].operation_id
            if current is not None and current[1].operation_id in selected_ids
            else next(
                operation.operation_id
                for operation in operations
                if operation.operation_id in selected_ids
            )
        )
        self._recipe = self._recipe.model_copy(update={"operations": tuple(operations)})
        self._finish_operation_structure_change(selected_ids, current_id)

    def _finish_operation_structure_change(
        self, selected_ids: set[str], current_id: str | None
    ) -> None:
        self._refresh_operation_list()
        if selected_ids:
            self._set_selected_operation_ids_silent(selected_ids)
        if current_id is not None:
            for row, operation in enumerate(self._recipe.operations):
                if operation.operation_id == current_id:
                    self._set_current_operation_row_silent(row)
                    break
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
    def _remove_posted_events_recursive(widget: QtWidgets.QWidget) -> None:
        for child in widget.findChildren(QtCore.QObject):
            QtCore.QCoreApplication.removePostedEvents(child)
        QtCore.QCoreApplication.removePostedEvents(widget)

    @staticmethod
    def _block_signals_recursive(widget: QtWidgets.QWidget) -> None:
        widget.blockSignals(True)
        for child in widget.findChildren(QtCore.QObject):
            if erlab.interactive.utils.qt_is_valid(child):
                child.blockSignals(True)

    def _retire_editor_widget(self, widget: QtWidgets.QWidget) -> None:
        if not erlab.interactive.utils.qt_is_valid(widget):
            return
        self._block_signals_recursive(widget)
        widget.setEnabled(False)
        widget.hide()
        if widget not in self._retired_editor_widgets:
            self._retired_editor_widgets.append(widget)
        self._queue_retired_editor_drain()

    def _queue_retired_editor_drain(self) -> None:
        if self._retired_editor_drain_pending:
            return
        self._retired_editor_drain_pending = True
        erlab.interactive.utils.single_shot(
            self,
            _RETIRED_EDITOR_DRAIN_DELAY_MS,
            self._drain_retired_editor_widgets,
        )

    def _drain_retired_editor_widgets(self) -> None:
        self._retired_editor_drain_pending = False
        if not erlab.interactive.utils.qt_is_valid(self):
            return
        if (
            not self._retired_editor_widgets
            or self._operation_editor_rebuild_must_wait()
            or self._retired_editor_has_focus()
        ):
            if self._retired_editor_widgets:
                self._queue_retired_editor_drain()
            return

        retired_widgets = self._retired_editor_widgets
        self._retired_editor_widgets = []
        for widget in retired_widgets:
            if not erlab.interactive.utils.qt_is_valid(widget):
                continue
            self._remove_posted_events_recursive(widget)
            widget.setParent(None)
            widget.deleteLater()

    def _retired_editor_has_focus(self) -> bool:
        focus_widget = QtWidgets.QApplication.focusWidget()
        return (
            focus_widget is not None
            and erlab.interactive.utils.qt_is_valid(focus_widget)
            and any(
                widget is focus_widget or widget.isAncestorOf(focus_widget)
                for widget in self._retired_editor_widgets
                if erlab.interactive.utils.qt_is_valid(widget)
            )
        )

    def _clear_form_layout(self, layout: QtWidgets.QFormLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                self._retire_editor_widget(widget)

    def _clear_operation_editor(self) -> None:
        for page in self._operation_editor_pages:
            self.step_editor_stack.removeWidget(page)
            self._retire_editor_widget(page)
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
                self._retire_editor_widget(widget)
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
        if (
            render_error := self._operation_render_errors.get(operation.operation_id)
        ) is not None:
            self.source_status_label.setText(f"Render error: {render_error}")
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
        selected_indices = self._selected_operation_indices()
        if len(selected_indices) > 1 and not self._selected_operations_are_compatible():
            page, layout = self._new_step_form_page(
                "figureComposerIncompatibleBatchPage"
            )
            label = QtWidgets.QLabel(
                "Select steps with the same editor type to batch edit them.",
                page,
            )
            label.setWordWrap(True)
            label.setEnabled(False)
            layout.addRow(QtWidgets.QLabel("Batch edit", page), label)
            self._set_step_sections(
                (
                    StepSection(
                        "batch",
                        "Batch",
                        page,
                        "The selected steps do not share one editor schema.",
                    ),
                )
            )
            return
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

    def _update_operation_editor_safely(self) -> None:
        if self._operation_editor_sender_requires_deferred_rebuild():
            self._queue_operation_editor_update()
            return
        self._update_operation_editor()

    def _operation_editor_sender_requires_deferred_rebuild(self) -> bool:
        sender = self.sender()
        if not isinstance(sender, QtWidgets.QWidget):
            return False
        editor_roots = (
            self.step_editor_stack,
            self.step_source_controls,
            self.source_list,
            self.target_axes_page,
            self.step_navigator,
        )
        return any(root is sender or root.isAncestorOf(sender) for root in editor_roots)

    def _queue_operation_editor_update(self) -> None:
        if self._operation_editor_update_pending:
            return
        self._operation_editor_update_pending = True
        self._schedule_queued_operation_editor_update()

    def _schedule_queued_operation_editor_update(self) -> None:
        erlab.interactive.utils.single_shot(
            self,
            _OPERATION_EDITOR_UPDATE_DELAY_MS,
            self._run_queued_operation_editor_update,
        )

    def _run_queued_operation_editor_update(self) -> None:
        if not erlab.interactive.utils.qt_is_valid(self):
            return
        if self._operation_editor_rebuild_must_wait():
            self._schedule_queued_operation_editor_update()
            return
        self._operation_editor_update_pending = False
        self._update_operation_editor()

    @staticmethod
    def _operation_editor_rebuild_must_wait() -> bool:
        popup = QtWidgets.QApplication.activePopupWidget()
        return popup is not None and erlab.interactive.utils.qt_is_valid(popup)

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
        mixed: bool = False,
        enabled: bool = True,
    ) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox(parent or self.operation_editor)
        combo.addItems(list(values))
        if mixed:
            self._set_combo_mixed_placeholder(combo)
        elif current is not None:
            self._set_combo_value(combo, current)
        combo.setEnabled(enabled)
        combo.currentTextChanged.connect(
            lambda text: None if self._mixed_combo_text(text) else changed(text)
        )
        return combo

    def _optional_name_combo(
        self,
        values: Sequence[str],
        current: str | None,
        none_label: str,
        changed: Callable[[str | None], None],
        *,
        parent: QtWidgets.QWidget | None = None,
        mixed: bool = False,
        enabled: bool = True,
    ) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox(parent or self.operation_editor)
        if mixed:
            combo.addItem(_MIXED_VALUES_TEXT, _MIXED_VALUE)
        combo.addItem(none_label, None)
        for value in values:
            combo.addItem(value, value)
        if current is not None and current not in values and not mixed:
            combo.addItem(current, current)
        if mixed:
            item = typing.cast("typing.Any", combo.model()).item(0)
            if item is not None:
                item.setEnabled(False)
            combo.setCurrentIndex(0)
        else:
            for index in range(combo.count()):
                if combo.itemData(index) == current:
                    combo.setCurrentIndex(index)
                    break
        combo.setEnabled(enabled)
        combo.currentIndexChanged.connect(
            lambda _index, combo=combo: (
                None
                if combo.currentData() is _MIXED_VALUE
                else changed(typing.cast("str | None", combo.currentData()))
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
        self._ensure_primary_source_data()
        self._apply_recipe_to_controls()
        _rendering._render_preview(self)

    def set_source_data(self, source_data: Mapping[str, xr.DataArray]) -> None:
        self._source_data = dict(source_data)

    def _ensure_primary_source_data(self) -> None:
        if self._recipe.primary_source in self._source_data or not self._source_data:
            return
        recipe_sources = {source.name for source in self._recipe.sources}
        fallback_name, fallback_data = next(iter(self._source_data.items()))
        self._source_data[self._recipe.primary_source] = fallback_data
        if fallback_name not in recipe_sources:
            del self._source_data[fallback_name]

    def _append_persistence_payload(self, ds: xr.Dataset) -> xr.Dataset:
        """Persist secondary figure sources without merging their coordinates."""
        source_entries: list[dict[str, str]] = []
        saved = ds.copy()
        for index, (source_name, source_data) in enumerate(self._source_data.items()):
            if source_name == self._recipe.primary_source:
                continue
            variable_name = f"{_PERSISTED_SOURCE_VAR_PREFIX}{index}"
            saved[variable_name] = xr.DataArray(
                _source_data_to_blob(source_data),
                dims=(f"{_PERSISTED_SOURCE_DIM_PREFIX}{index}",),
            )
            source_entries.append(
                {
                    "source": source_name,
                    "variable": variable_name,
                }
            )
        if source_entries:
            saved.attrs[_PERSISTED_SOURCE_MAP_ATTR] = json.dumps(source_entries)
        return saved

    def _restore_persistence_payload(self, ds: xr.Dataset) -> None:
        payload_json = ds.attrs.get(_PERSISTED_SOURCE_MAP_ATTR)
        if not isinstance(payload_json, str):
            return
        try:
            payloads = json.loads(payload_json)
        except json.JSONDecodeError:
            return
        if not isinstance(payloads, list):
            return
        source_data = dict(self._source_data)
        for payload in payloads:
            if not isinstance(payload, dict):
                continue
            source_name = payload.get("source")
            variable_name = payload.get("variable")
            if (
                not isinstance(source_name, str)
                or not isinstance(variable_name, str)
                or variable_name not in ds
            ):
                continue
            try:
                source_data[source_name] = _source_data_from_blob(
                    ds[variable_name].values
                )
            except (OSError, TypeError, ValueError, KeyError):
                continue
        self.set_source_data(source_data)
        self._apply_recipe_to_controls()
        _rendering._render_preview(self, show_window=False)

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
