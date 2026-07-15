"""Source-list and source-detail presentation for Figure Composer."""

from __future__ import annotations

import typing

from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive._figurecomposer._model._sources import _public_source_data
from erlab.interactive._figurecomposer._ui._editor_controls import (
    MIXED_VALUE,
    MIXED_VALUES_TEXT,
    LineEditControlAdapter,
)
from erlab.interactive._figurecomposer._ui._panel_controls import _step_toolbar_button
from erlab.interactive._figurecomposer._ui._reorder_list import (
    ReorderList,
    event_requests_context_menu,
)
from erlab.interactive._figurecomposer._ui._source_inspector import (
    SourceInspectorWidget,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping, Sequence

    import xarray as xr


_SOURCE_COLUMN = 0
_SHAPE_COLUMN = 1
_USED_ROLE = QtCore.Qt.ItemDataRole.UserRole + 1


class FigureSourceRow(typing.NamedTuple):
    """Immutable presentation state for one source-list row."""

    name: str
    display: str
    tooltip: str
    data: xr.DataArray | None
    missing: bool
    used: bool


class FigureSourceDetail(typing.NamedTuple):
    """Presentation state for the selected source's detail pane."""

    name: str
    data: xr.DataArray | None
    context_lines: tuple[str, ...]
    usage_count: int
    origin: str
    alias_enabled: bool


class FigureSourceSelectionRow(typing.NamedTuple):
    """Presentation state for one editable source-selection dimension."""

    dimension: str
    tooltip: str
    mode: str | None
    mode_mixed: bool
    value_text: str
    value_mixed: bool
    width_text: str
    width_mixed: bool


class _FigureSourceSelectionWidgets(typing.NamedTuple):
    label: QtWidgets.QLabel
    row: QtWidgets.QWidget
    mode: QtWidgets.QComboBox
    value: QtWidgets.QLineEdit
    width: QtWidgets.QLineEdit


class _FigureSourceList(ReorderList):
    context_menu_requested = QtCore.Signal(QtCore.QPoint)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(_SOURCE_COLUMN, parent)
        self.setColumnCount(2)
        self.setHeaderLabels(("Alias", "Shape"))
        self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.context_menu_requested)

    def keyPressEvent(self, event: QtGui.QKeyEvent | None) -> None:
        if event is None:
            return
        if event_requests_context_menu(event):
            item = self.currentItem()
            rect = self.visualItemRect(item) if item is not None else QtCore.QRect()
            self.context_menu_requested.emit(rect.center())
            event.accept()
            return
        super().keyPressEvent(event)


class FigureSourcePanel(QtWidgets.QWidget):
    """Concrete source editor that emits source-domain user intentions."""

    selection_changed = QtCore.Signal(object, object)
    add_requested = QtCore.Signal()
    remove_requested = QtCore.Signal(object)
    refresh_requested = QtCore.Signal(object)
    refresh_all_requested = QtCore.Signal()
    reveal_requested = QtCore.Signal(object)
    rename_requested = QtCore.Signal(str, str)
    duplicate_requested = QtCore.Signal(object)
    move_requested = QtCore.Signal(object, int)
    reorder_requested = QtCore.Signal(object, object, object)
    selection_dimension_requested = QtCore.Signal(object, str, str, str, str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("figureComposerSourcesPage")
        self._synchronizing_selection = False
        self._context_menu: QtWidgets.QMenu | None = None
        self._refresh_all_available = False
        self._move_up_available = False
        self._move_down_available = False
        self._duplicate_available = False
        self._can_drop_source: Callable[[QtCore.QMimeData], bool] | None = None
        self._drop_source: Callable[[QtCore.QMimeData], bool] | None = None
        self._selection_editor_updating = False
        self._selection_widgets: list[_FigureSourceSelectionWidgets] = []
        self._build_ui()
        self.setAcceptDrops(True)
        for target in self.drop_targets():
            if target is not self:
                self.install_drop_target(target)

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.source_status_label = QtWidgets.QLabel(self)
        self.source_status_label.setObjectName("figureComposerSourceStatus")
        self.source_status_label.setWordWrap(True)
        self.source_status_label.setVisible(False)

        self.source_actions = QtWidgets.QWidget(self)
        self.source_actions.setObjectName("figureComposerSourceActions")
        action_layout = QtWidgets.QHBoxLayout(self.source_actions)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(4)
        self.add_source_button = _step_toolbar_button(
            self.source_actions,
            "figureComposerAddSourceButton",
            "Add…",
            "Add ImageTool data from the manager as figure sources.",
        )
        self.add_source_button.clicked.connect(self.add_requested)
        action_layout.addWidget(self.add_source_button)
        self.remove_selected_source_button = _step_toolbar_button(
            self.source_actions,
            "figureComposerRemoveSelectedSourceButton",
            "Remove",
            "Remove the selected unused source or sources.",
        )
        self.remove_selected_source_button.clicked.connect(
            lambda: self.remove_requested.emit(self.selected_names())
        )
        action_layout.addWidget(self.remove_selected_source_button)
        self.reveal_sources_button = _step_toolbar_button(
            self.source_actions,
            "figureComposerRevealSourcesButton",
            "Reveal in Manager",
            "Reveal selected sources in ImageTool Manager",
        )
        self.reveal_sources_button.setAccessibleName(
            "Reveal Selected Sources in ImageTool Manager"
        )
        self.reveal_sources_button.clicked.connect(
            lambda: self.reveal_requested.emit(self.selected_names())
        )
        action_layout.addWidget(self.reveal_sources_button)
        action_layout.addStretch(1)
        self.refresh_sources_button = QtWidgets.QToolButton(self.source_actions)
        self.refresh_sources_button.setObjectName("figureComposerRefreshSourcesButton")
        self.refresh_sources_button.setText("Refresh")
        self.refresh_sources_button.setAccessibleName("Refresh Selected Sources")
        self.refresh_sources_button.setIcon(QtGui.QIcon.fromTheme("view-refresh"))
        self.refresh_sources_button.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.refresh_sources_button.setAutoRaise(True)
        self.refresh_sources_button.clicked.connect(
            lambda: self.refresh_requested.emit(self.selected_names())
        )
        action_layout.addWidget(self.refresh_sources_button)
        layout.addWidget(self.source_actions)
        layout.addWidget(self.source_status_label)

        self.source_splitter = QtWidgets.QSplitter(
            QtCore.Qt.Orientation.Horizontal, self
        )
        self.source_splitter.setObjectName("figureComposerSourceSplitter")
        self.source_splitter.setChildrenCollapsible(False)
        self.source_list = _FigureSourceList(self.source_splitter)
        self.source_list.setObjectName("figureComposerSourceList")
        self.source_list.setAccessibleName("Figure Sources")
        self.source_list.context_menu_requested.connect(self._show_context_menu)
        self.source_list.rows_reordered.connect(self.reorder_requested)
        self.source_list.setRootIsDecorated(False)
        self.source_list.setIndentation(0)
        self.source_list.setUniformRowHeights(True)
        self.source_list.setAlternatingRowColors(True)
        self.source_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.source_list.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.source_list.currentItemChanged.connect(self._selection_did_change)
        self.source_list.itemSelectionChanged.connect(self._selection_did_change)
        self.source_list.itemDoubleClicked.connect(self._item_double_clicked)
        self.rename_source_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key.Key_F2), self.source_list
        )
        self.rename_source_shortcut.setObjectName("figureComposerRenameSourceShortcut")
        self.rename_source_shortcut.activated.connect(self.focus_alias_editor)
        self.source_list.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.source_list.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        header = typing.cast("QtWidgets.QHeaderView", self.source_list.header())
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(80)
        header.setSectionResizeMode(
            _SOURCE_COLUMN, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        header.setSectionResizeMode(
            _SHAPE_COLUMN, QtWidgets.QHeaderView.ResizeMode.Interactive
        )
        header.resizeSection(_SHAPE_COLUMN, 150)
        self.source_splitter.addWidget(self.source_list)

        self.source_detail_scroll = QtWidgets.QScrollArea(self.source_splitter)
        self.source_detail_scroll.setObjectName("figureComposerSourceDetailScroll")
        self.source_detail_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.source_detail_scroll.setWidgetResizable(True)
        self.source_detail_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.source_detail_content = QtWidgets.QWidget(self.source_detail_scroll)
        self.source_detail_content.setObjectName("figureComposerSourceDetailContent")
        detail_layout = QtWidgets.QVBoxLayout(self.source_detail_content)
        detail_layout.setContentsMargins(8, 0, 0, 0)
        detail_layout.setSpacing(6)
        self.source_editor_state_label = QtWidgets.QLabel(
            "Select a source to inspect or edit it.", self.source_detail_content
        )
        self.source_editor_state_label.setObjectName("figureComposerSourceEditorState")
        self.source_editor_state_label.setWordWrap(True)
        detail_layout.addWidget(self.source_editor_state_label)
        self.source_alias_controls = QtWidgets.QWidget(self.source_detail_content)
        self.source_alias_controls.setObjectName("figureComposerSourceAliasControls")
        alias_layout = QtWidgets.QFormLayout(self.source_alias_controls)
        alias_layout.setContentsMargins(0, 0, 0, 0)
        alias_layout.setSpacing(4)
        alias_layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self.source_alias_edit = QtWidgets.QLineEdit(self.source_alias_controls)
        self.source_alias_edit.setObjectName("figureComposerSourceAliasEdit")
        self.source_alias_edit.setToolTip("Rename this source variable.")
        self.source_alias_edit.editingFinished.connect(self._alias_edit_finished)
        alias_layout.addRow("Alias", self.source_alias_edit)
        detail_layout.addWidget(self.source_alias_controls)
        self.source_inspector = SourceInspectorWidget(self.source_detail_content)
        self.source_inspector.setAcceptDrops(True)
        detail_layout.addWidget(self.source_inspector)
        self.source_selection_controls = QtWidgets.QWidget(self.source_detail_content)
        self.source_selection_controls.setObjectName(
            "figureComposerSourceSelectionControls"
        )
        self.source_selection_controls_layout = QtWidgets.QFormLayout(
            self.source_selection_controls
        )
        self.source_selection_controls_layout.setContentsMargins(0, 0, 0, 0)
        self.source_selection_controls_layout.setSpacing(4)
        self.source_selection_controls_layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self._selection_section = self._create_selection_section()
        self._selection_section.setVisible(False)
        self._selection_message = QtWidgets.QLabel(self.source_selection_controls)
        self._selection_message.setObjectName("figureComposerSourceSelectionMessage")
        self._selection_message.setWordWrap(True)
        self._selection_message.setVisible(False)
        self._selection_message_label = QtWidgets.QLabel(
            "Dimensions", self.source_selection_controls
        )
        self._selection_message_label.setVisible(False)
        self.source_selection_controls.setAcceptDrops(True)
        detail_layout.addWidget(self.source_selection_controls)
        self.source_validation_label = QtWidgets.QLabel(self.source_detail_content)
        self.source_validation_label.setObjectName(
            "figureComposerSourceValidationStatus"
        )
        self.source_validation_label.setWordWrap(True)
        self.source_validation_label.setVisible(False)
        detail_layout.addWidget(self.source_validation_label)
        detail_layout.addStretch(1)
        self.source_detail_scroll.setWidget(self.source_detail_content)
        self.source_detail_content.setAutoFillBackground(False)
        viewport = typing.cast(
            "QtWidgets.QWidget", self.source_detail_scroll.viewport()
        )
        viewport.setAutoFillBackground(False)
        self.source_splitter.addWidget(self.source_detail_scroll)
        self.source_splitter.setStretchFactor(0, 2)
        self.source_splitter.setStretchFactor(1, 3)
        self.source_splitter.setSizes((400, 600))
        layout.addWidget(self.source_splitter, 1)

    def _create_selection_section(self) -> QtWidgets.QWidget:
        section = QtWidgets.QWidget(self.source_selection_controls)
        section.setObjectName("figureComposerSourceSelectionSection")
        section.setProperty("figureComposerSectionHeader", True)
        layout = QtWidgets.QHBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 2)
        layout.setSpacing(6)
        label = QtWidgets.QLabel("Selection", section)
        label.setProperty("figureComposerSectionHeaderLabel", True)
        font = label.font()
        font.setBold(True)
        label.setFont(font)
        line = QtWidgets.QFrame(section)
        line.setObjectName("figureComposerSourceSelectionSectionLine")
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        layout.addWidget(label)
        layout.addWidget(line, 1)
        return section

    def set_detail(
        self,
        detail: FigureSourceDetail | None,
        *,
        selection_count: int,
    ) -> None:
        """Render the source-detail pane from application presentation state."""
        content = self.source_detail_content
        content.setProperty("figureComposerSourceSelectionCount", selection_count)
        if selection_count == 0:
            content.setProperty("figureComposerSourceEditorMode", "empty")
            content.setProperty("figureComposerSourceUsageCount", 0)
            content.setProperty("figureComposerSourceOrigin", "")
            self.source_editor_state_label.setText(
                "Select a source to inspect or edit it."
            )
            self.source_editor_state_label.setVisible(True)
            self.source_alias_controls.setVisible(False)
            self.source_inspector.setVisible(False)
            self.source_selection_controls.setVisible(False)
            self.source_inspector.set_context(source_name=None, data=None)
            return
        if selection_count > 1:
            content.setProperty("figureComposerSourceEditorMode", "multiple")
            content.setProperty("figureComposerSourceUsageCount", 0)
            content.setProperty("figureComposerSourceOrigin", "")
            self.source_editor_state_label.setText(
                f"{selection_count} sources selected\n"
                "Selection changes apply to all compatible selected sources."
            )
            self.source_editor_state_label.setVisible(True)
            self.source_alias_controls.setVisible(False)
            self.source_inspector.setVisible(False)
            self.source_selection_controls.setVisible(True)
            self.source_inspector.set_context(source_name=None, data=None)
            return
        if detail is None:
            raise ValueError("single-source detail state is required")
        content.setProperty("figureComposerSourceEditorMode", "single")
        content.setProperty("figureComposerSourceUsageCount", detail.usage_count)
        content.setProperty("figureComposerSourceOrigin", detail.origin)
        self.source_editor_state_label.setVisible(False)
        self.source_alias_controls.setVisible(True)
        self.set_alias(detail.name, enabled=detail.alias_enabled)
        self.source_inspector.setVisible(True)
        self.source_selection_controls.setVisible(True)
        self.source_inspector.set_context(
            source_name=detail.name,
            data=detail.data,
            context_lines=detail.context_lines,
        )

    def set_selection_editor(
        self,
        rows: Sequence[FigureSourceSelectionRow],
        *,
        message: str | None = None,
        message_tooltip: str = "",
    ) -> None:
        """Reconcile reusable selection controls with current dimension state."""
        layout = self.source_selection_controls_layout
        while layout.count():
            item = layout.takeAt(0)
            if item is not None and (widget := item.widget()) is not None:
                widget.hide()
        if not rows and message is None:
            return

        self._selection_section.show()
        layout.addRow(self._selection_section)
        if message is not None:
            self._selection_message.setText(message)
            self._selection_message.setToolTip(message_tooltip)
            self._selection_message_label.setToolTip(message_tooltip)
            self._selection_message.show()
            self._selection_message_label.show()
            layout.addRow(self._selection_message_label, self._selection_message)
            return

        for index, row_state in enumerate(rows):
            widgets = self._selection_widgets_for(index)
            self._configure_selection_widgets(widgets, row_state, index=index)
            layout.addRow(widgets.label, widgets.row)

    def _selection_widgets_for(self, index: int) -> _FigureSourceSelectionWidgets:
        if index < len(self._selection_widgets):
            return self._selection_widgets[index]
        label = QtWidgets.QLabel(self.source_selection_controls)
        row = QtWidgets.QWidget(self.source_selection_controls)
        row_layout = QtWidgets.QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)
        mode = QtWidgets.QComboBox(row)
        value = QtWidgets.QLineEdit(row)
        width = QtWidgets.QLineEdit(row)
        row_layout.addWidget(mode)
        row_layout.addWidget(value, 1)
        row_layout.addWidget(width, 1)
        widgets = _FigureSourceSelectionWidgets(label, row, mode, value, width)
        self._selection_widgets.append(widgets)
        self._connect_selection_widgets(widgets)
        return widgets

    def _configure_selection_widgets(
        self,
        widgets: _FigureSourceSelectionWidgets,
        state: FigureSourceSelectionRow,
        *,
        index: int,
    ) -> None:
        self._selection_editor_updating = True
        try:
            blockers = tuple(
                QtCore.QSignalBlocker(widget)
                for widget in (widgets.mode, widgets.value, widgets.width)
            )
            widgets.label.setText(state.dimension)
            widgets.label.setToolTip(state.tooltip)
            widgets.row.setObjectName(f"figureComposerSourceSelectionDimRow{index}")
            widgets.row.setToolTip(state.tooltip)
            widgets.mode.setObjectName(f"figureComposerSourceSelectionModeCombo{index}")
            widgets.value.setObjectName(
                f"figureComposerSourceSelectionValueEdit{index}"
            )
            widgets.width.setObjectName(
                f"figureComposerSourceSelectionWidthEdit{index}"
            )
            for widget in (widgets.mode, widgets.value, widgets.width):
                widget.setProperty(
                    "figure_composer_source_selection_dim", state.dimension
                )
            widgets.value.setProperty("figure_composer_source_selection_field", "value")
            widgets.width.setProperty("figure_composer_source_selection_field", "width")
            self._configure_selection_mode(widgets.mode, state)
            widgets.mode.setToolTip(
                f"{state.tooltip}\n\nChoose None, isel, qsel, or Mean for this "
                "dimension."
            )
            widgets.value.setPlaceholderText("value")
            widgets.value.setToolTip(
                f"{state.tooltip}\n\nisel accepts integer positions and index "
                "slices. qsel accepts coordinate values and coordinate slices."
            )
            widgets.value.setText("" if state.value_mixed else state.value_text)
            widgets.value.setModified(False)
            LineEditControlAdapter(widgets.value).set_mixed(state.value_mixed)
            widgets.width.setPlaceholderText("width")
            widgets.width.setToolTip(
                f"{state.tooltip}\n\nOptional qsel width for centered averaging. "
                "Leave blank to select the nearest coordinate."
            )
            widgets.width.setText("" if state.width_mixed else state.width_text)
            widgets.width.setModified(False)
            LineEditControlAdapter(widgets.width).set_mixed(state.width_mixed)
            widgets.value.setVisible(state.mode in {"isel", "qsel"})
            widgets.width.setVisible(state.mode == "qsel")
            widgets.mode.setProperty(
                "figure_composer_source_selection_previous_mode",
                widgets.mode.currentData(),
            )
            for widget in (widgets.label, widgets.row, widgets.mode):
                widget.show()
            del blockers
        finally:
            self._selection_editor_updating = False

    @staticmethod
    def _configure_selection_mode(
        combo: QtWidgets.QComboBox, state: FigureSourceSelectionRow
    ) -> None:
        combo.clear()
        if state.mode_mixed:
            combo.addItem(MIXED_VALUES_TEXT, MIXED_VALUE)
        for mode, label, tooltip in (
            ("keep", "None", "Leave this dimension unchanged."),
            (
                "isel",
                "isel",
                "Select this dimension by integer position or index slice.",
            ),
            (
                "qsel",
                "qsel",
                "Select this dimension by coordinate value or coordinate slice.",
            ),
            ("mean", "Mean", "Average over and remove this dimension."),
        ):
            combo.addItem(label, mode)
            combo.setItemData(
                combo.count() - 1,
                tooltip,
                QtCore.Qt.ItemDataRole.ToolTipRole,
            )
        if state.mode_mixed:
            item = typing.cast("typing.Any", combo.model()).item(0)
            if item is not None:
                item.setEnabled(False)
            combo.setCurrentIndex(0)
        elif state.mode is not None:
            combo.setCurrentIndex(max(combo.findData(state.mode), 0))

    def _connect_selection_widgets(
        self,
        widgets: _FigureSourceSelectionWidgets,
    ) -> None:
        def update_from_controls() -> None:
            if self._selection_editor_updating:
                return
            mode = widgets.mode.currentData()
            if not isinstance(mode, str) or mode not in {
                "keep",
                "isel",
                "qsel",
                "mean",
            }:
                return
            if LineEditControlAdapter(widgets.value).unchanged_mixed() and mode in {
                "isel",
                "qsel",
            }:
                return
            if (
                LineEditControlAdapter(widgets.width).unchanged_mixed()
                and mode == "qsel"
            ):
                return
            dimension = widgets.mode.property("figure_composer_source_selection_dim")
            if not isinstance(dimension, str):
                return
            self.selection_dimension_requested.emit(
                self.selected_names(),
                dimension,
                mode,
                widgets.value.text(),
                widgets.width.text(),
            )

        def mode_changed(_index: int) -> None:
            mode = widgets.mode.currentData()
            mode_text = mode if isinstance(mode, str) else ""
            widgets.value.setVisible(mode_text in {"isel", "qsel"})
            widgets.width.setVisible(mode_text == "qsel")
            previous_mode = widgets.mode.property(
                "figure_composer_source_selection_previous_mode"
            )
            changed = previous_mode != mode_text
            if changed or mode_text not in {"isel", "qsel"}:
                widgets.value.clear()
                widgets.value.setModified(False)
            if changed or mode_text != "qsel":
                widgets.width.clear()
                widgets.width.setModified(False)
            widgets.mode.setProperty(
                "figure_composer_source_selection_previous_mode", mode_text
            )
            if mode_text in {"isel", "qsel"} and not widgets.value.text().strip():
                return
            update_from_controls()

        widgets.mode.activated.connect(mode_changed)
        widgets.value.editingFinished.connect(update_from_controls)
        widgets.width.editingFinished.connect(update_from_controls)

    def drop_targets(self) -> tuple[QtWidgets.QWidget, ...]:
        """Widgets that participate in synchronous source drag and drop."""
        candidates = (
            self,
            self.source_list,
            self.source_list.viewport(),
            self.source_detail_scroll,
            self.source_detail_scroll.viewport(),
            self.source_detail_content,
            self.source_inspector,
            self.source_selection_controls,
        )
        return tuple(widget for widget in candidates if widget is not None)

    def set_drop_handlers(
        self,
        can_drop: Callable[[QtCore.QMimeData], bool],
        drop: Callable[[QtCore.QMimeData], bool],
    ) -> None:
        """Set synchronous source-drop handlers owned by the application layer."""
        self._can_drop_source = can_drop
        self._drop_source = drop

    def install_drop_target(self, target: QtWidgets.QWidget) -> None:
        """Route one widget's drag events through this panel."""
        target.setAcceptDrops(True)
        target.installEventFilter(self)

    def _handle_drag_event(self, event: QtCore.QEvent | None) -> bool:
        if event is None or event.type() not in {
            QtCore.QEvent.Type.DragEnter,
            QtCore.QEvent.Type.DragMove,
            QtCore.QEvent.Type.Drop,
        }:
            return False
        if not isinstance(
            event, (QtGui.QDragEnterEvent, QtGui.QDragMoveEvent, QtGui.QDropEvent)
        ):
            return False
        mime = event.mimeData()
        if (
            mime is None
            or self._can_drop_source is None
            or not self._can_drop_source(mime)
        ):
            return False
        if event.type() == QtCore.QEvent.Type.Drop and (
            self._drop_source is None or not self._drop_source(mime)
        ):
            return False
        event.setDropAction(QtCore.Qt.DropAction.CopyAction)
        event.accept()
        return True

    def eventFilter(
        self, watched: QtCore.QObject | None, event: QtCore.QEvent | None
    ) -> bool:
        if self._handle_drag_event(event):
            return True
        return super().eventFilter(watched, event)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent | None) -> None:
        if not self._handle_drag_event(event):
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent | None) -> None:
        if not self._handle_drag_event(event):
            super().dragMoveEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent | None) -> None:
        if not self._handle_drag_event(event):
            super().dropEvent(event)

    @staticmethod
    def source_name_from_item(item: QtWidgets.QTreeWidgetItem | None) -> str | None:
        if item is None:
            return None
        source_name = item.data(_SOURCE_COLUMN, QtCore.Qt.ItemDataRole.UserRole)
        return source_name if isinstance(source_name, str) else None

    def selected_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for item in self.source_list.selectedItems():
            source_name = self.source_name_from_item(item)
            if source_name is not None and source_name not in names:
                names.append(source_name)
        return tuple(names)

    def current_name(self) -> str | None:
        return self.source_name_from_item(self.source_list.currentItem())

    def set_sources(
        self,
        rows: Sequence[FigureSourceRow],
        *,
        selected_names: Collection[str],
        current_name: str | None,
    ) -> None:
        """Replace source rows while preserving selection by stable source name."""
        self.source_inspector.invalidate_details()
        self._clear_rows()
        for row in rows:
            self._add_row(row)
        self.set_selected_names(selected_names, current_name=current_name)

    def _clear_rows(self) -> None:
        for row in range(self.source_list.topLevelItemCount()):
            item = typing.cast(
                "QtWidgets.QTreeWidgetItem", self.source_list.topLevelItem(row)
            )
            for column in range(self.source_list.columnCount()):
                widget = self.source_list.itemWidget(item, column)
                if widget is None:
                    continue
                self.source_list.removeItemWidget(item, column)
                widget.setParent(None)
                widget.deleteLater()
        self.source_list.clear()

    def _add_row(self, row: FigureSourceRow) -> None:
        item = QtWidgets.QTreeWidgetItem(
            [row.display, "Data unavailable" if row.data is None else ""]
        )
        item.setData(_SOURCE_COLUMN, QtCore.Qt.ItemDataRole.UserRole, row.name)
        item.setData(_SOURCE_COLUMN, _USED_ROLE, row.used)
        item.setFlags(
            QtCore.Qt.ItemFlag.ItemIsEnabled
            | QtCore.Qt.ItemFlag.ItemIsSelectable
            | QtCore.Qt.ItemFlag.ItemIsDragEnabled
        )
        item.setToolTip(_SOURCE_COLUMN, row.tooltip)
        item.setToolTip(_SHAPE_COLUMN, row.tooltip)
        item.setData(
            _SOURCE_COLUMN,
            QtCore.Qt.ItemDataRole.AccessibleDescriptionRole,
            row.tooltip,
        )
        if row.missing:
            brush = QtGui.QBrush(QtGui.QColor("darkRed"))
            item.setForeground(_SOURCE_COLUMN, brush)
            item.setForeground(_SHAPE_COLUMN, brush)
        self.source_list.addTopLevelItem(item)
        if row.data is None:
            return
        shape_label = QtWidgets.QLabel(
            erlab.interactive.utils._apply_qt_accent_color(
                erlab.utils.formatting.format_darr_shape_html(
                    _public_source_data(row.data).rename(None), show_size=False
                )
            ),
            self.source_list,
        )
        shape_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        shape_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.NoTextInteraction
        )
        shape_label.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        shape_label.setContentsMargins(4, 0, 4, 0)
        if row.missing:
            palette = shape_label.palette()
            palette.setColor(
                QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("darkRed")
            )
            shape_label.setPalette(palette)
        item.setSizeHint(_SHAPE_COLUMN, shape_label.sizeHint())
        self.source_list.setItemWidget(item, _SHAPE_COLUMN, shape_label)

    def update_tooltips(self, tooltips: Mapping[str, str]) -> None:
        for index in range(self.source_list.topLevelItemCount()):
            item = self.source_list.topLevelItem(index)
            source_name = self.source_name_from_item(item)
            if item is None or source_name is None:
                continue
            tooltip = tooltips.get(source_name, "")
            item.setToolTip(_SOURCE_COLUMN, tooltip)
            item.setToolTip(_SHAPE_COLUMN, tooltip)
            item.setData(
                _SOURCE_COLUMN,
                QtCore.Qt.ItemDataRole.AccessibleDescriptionRole,
                tooltip,
            )

    def set_used_sources(self, names: Collection[str]) -> None:
        for index in range(self.source_list.topLevelItemCount()):
            item = self.source_list.topLevelItem(index)
            source_name = self.source_name_from_item(item)
            if item is not None and source_name is not None:
                item.setData(_SOURCE_COLUMN, _USED_ROLE, source_name in names)

    def set_selected_names(
        self, names: Collection[str], *, current_name: str | None
    ) -> None:
        selected = set(names)
        self._synchronizing_selection = True
        try:
            self.source_list.clearSelection()
            if current_name is None:
                self.source_list.setCurrentIndex(QtCore.QModelIndex())
            current_item: QtWidgets.QTreeWidgetItem | None = None
            for row in range(self.source_list.topLevelItemCount()):
                item = typing.cast(
                    "QtWidgets.QTreeWidgetItem", self.source_list.topLevelItem(row)
                )
                if self.source_name_from_item(item) == current_name:
                    current_item = item
            if current_item is not None:
                self.source_list.setCurrentItem(current_item)
            for row in range(self.source_list.topLevelItemCount()):
                item = typing.cast(
                    "QtWidgets.QTreeWidgetItem", self.source_list.topLevelItem(row)
                )
                item.setSelected(self.source_name_from_item(item) in selected)
        finally:
            self._synchronizing_selection = False

    def set_status(self, text: str | None) -> None:
        self.source_status_label.setText("" if text is None else text)
        self.source_status_label.setVisible(bool(text))

    def set_validation(self, text: str | None) -> None:
        self.source_validation_label.setText("" if text is None else text)
        self.source_validation_label.setVisible(bool(text))

    def set_action_availability(
        self,
        *,
        add: bool,
        refresh: bool,
        reveal: bool,
        remove: bool,
        refresh_all: bool,
        duplicate: bool,
        move_up: bool,
        move_down: bool,
    ) -> None:
        self.add_source_button.setEnabled(add)
        add_tip = (
            "Add ImageTool data from the manager as figure sources"
            if add
            else "Open this Figure Composer from ImageTool Manager to add sources"
        )
        self.add_source_button.setToolTip(add_tip)
        self.add_source_button.setStatusTip(add_tip)
        self.refresh_sources_button.setEnabled(refresh)
        refresh_tip = (
            "Refresh selected sources from their ImageTools"
            if refresh
            else "No selected sources can be refreshed"
        )
        self.refresh_sources_button.setToolTip(refresh_tip)
        self.refresh_sources_button.setStatusTip(refresh_tip)
        self.reveal_sources_button.setEnabled(reveal)
        reveal_tip = (
            "Reveal selected sources in ImageTool Manager"
            if reveal
            else "No selected sources are associated with an ImageTool Manager row"
        )
        self.reveal_sources_button.setToolTip(reveal_tip)
        self.reveal_sources_button.setStatusTip(reveal_tip)
        self.remove_selected_source_button.setEnabled(remove)
        remove_tip = (
            "Remove the selected unused source or sources"
            if remove
            else "Selected sources are in use or cannot be removed"
        )
        self.remove_selected_source_button.setToolTip(remove_tip)
        self.remove_selected_source_button.setStatusTip(remove_tip)
        self._refresh_all_available = refresh_all
        self._duplicate_available = duplicate
        self._move_up_available = move_up
        self._move_down_available = move_down

    def set_alias(self, name: str, *, enabled: bool) -> None:
        self.source_alias_edit.setEnabled(enabled)
        self.source_alias_edit.setText(name)
        self.source_alias_edit.setProperty(
            "figure_composer_source_alias_original", name
        )

    def reset_alias(self, name: str) -> None:
        self.set_alias(name, enabled=self.source_alias_edit.isEnabled())

    @QtCore.Slot()
    def focus_alias_editor(self) -> None:
        if len(self.selected_names()) != 1 or not self.source_alias_edit.isEnabled():
            return
        self.source_alias_edit.setFocus(QtCore.Qt.FocusReason.ShortcutFocusReason)
        self.source_alias_edit.selectAll()

    @QtCore.Slot()
    def _alias_edit_finished(self) -> None:
        original = self.source_alias_edit.property(
            "figure_composer_source_alias_original"
        )
        if not isinstance(original, str):
            return
        alias = self.source_alias_edit.text().strip()
        if alias == original:
            self.source_alias_edit.setText(original)
            self.set_validation(None)
            return
        self.rename_requested.emit(original, alias)

    @QtCore.Slot()
    def _selection_did_change(self, *_args: object) -> None:
        if not self._synchronizing_selection:
            self.selection_changed.emit(self.selected_names(), self.current_name())

    @QtCore.Slot(QtWidgets.QTreeWidgetItem, int)
    def _item_double_clicked(
        self, _item: QtWidgets.QTreeWidgetItem, column: int
    ) -> None:
        if column == _SOURCE_COLUMN:
            self.focus_alias_editor()

    @QtCore.Slot(QtCore.QPoint)
    def _show_context_menu(self, position: QtCore.QPoint) -> None:
        menu = QtWidgets.QMenu("Sources", self.source_list)
        self._context_menu = menu

        def clear_menu(_destroyed: object | None = None) -> None:
            if self._context_menu is menu:
                self._context_menu = None

        menu.destroyed.connect(clear_menu)
        menu.aboutToHide.connect(menu.deleteLater)
        add_action = QtGui.QAction("Add…", menu)
        add_action.setObjectName("figureComposerContextAddSourceAction")
        add_action.setEnabled(self.add_source_button.isEnabled())
        add_action.triggered.connect(self.add_requested)
        menu.addAction(add_action)
        menu.addSeparator()
        rename_action = QtGui.QAction("Rename Alias", menu)
        rename_action.setObjectName("figureComposerContextRenameSourceAction")
        rename_action.setEnabled(
            len(self.selected_names()) == 1 and self.source_alias_edit.isEnabled()
        )
        rename_action.triggered.connect(self.focus_alias_editor)
        menu.addAction(rename_action)
        duplicate_action = QtGui.QAction("Duplicate", menu)
        duplicate_action.setObjectName("figureComposerContextDuplicateSourceAction")
        duplicate_action.setEnabled(self._duplicate_available)
        duplicate_action.triggered.connect(
            lambda: self.duplicate_requested.emit(self.selected_names())
        )
        menu.addAction(duplicate_action)
        for text, object_name, offset, enabled in (
            (
                "Move Up",
                "figureComposerContextMoveSourceUpAction",
                -1,
                self._move_up_available,
            ),
            (
                "Move Down",
                "figureComposerContextMoveSourceDownAction",
                1,
                self._move_down_available,
            ),
        ):
            action = QtGui.QAction(text, menu)
            action.setObjectName(object_name)
            action.setEnabled(enabled)
            action.triggered.connect(
                lambda _checked=False, direction=offset: self.move_requested.emit(
                    self.selected_names(), direction
                )
            )
            menu.addAction(action)
        menu.addSeparator()
        reveal_action = QtGui.QAction("Reveal in Manager", menu)
        reveal_action.setObjectName("figureComposerContextRevealSourceAction")
        reveal_action.setEnabled(self.reveal_sources_button.isEnabled())
        reveal_action.triggered.connect(
            lambda: self.reveal_requested.emit(self.selected_names())
        )
        menu.addAction(reveal_action)
        refresh_action = QtGui.QAction("Refresh Selected", menu)
        refresh_action.setObjectName("figureComposerContextRefreshSourceAction")
        refresh_action.setEnabled(self.refresh_sources_button.isEnabled())
        refresh_action.triggered.connect(
            lambda: self.refresh_requested.emit(self.selected_names())
        )
        menu.addAction(refresh_action)
        refresh_all_action = QtGui.QAction("Refresh All", menu)
        refresh_all_action.setObjectName("figureComposerContextRefreshAllSourcesAction")
        refresh_all_action.setEnabled(self._refresh_all_available)
        refresh_all_action.triggered.connect(self.refresh_all_requested)
        menu.addAction(refresh_all_action)
        remove_action = QtGui.QAction("Remove", menu)
        remove_action.setObjectName("figureComposerContextRemoveSourceAction")
        remove_action.setEnabled(self.remove_selected_source_button.isEnabled())
        remove_action.triggered.connect(
            lambda: self.remove_requested.emit(self.selected_names())
        )
        menu.addAction(remove_action)
        viewport = typing.cast("QtWidgets.QWidget", self.source_list.viewport())
        menu.popup(viewport.mapToGlobal(position))
