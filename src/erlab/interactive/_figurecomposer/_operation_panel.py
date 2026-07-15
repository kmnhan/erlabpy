"""Operation-list and generic editor presentation for Figure Composer."""

from __future__ import annotations

import typing
import weakref

from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive._figurecomposer._reorder_list import (
    ReorderList,
    event_requests_context_menu,
)
from erlab.interactive._figurecomposer._widgets import (
    _AxesTargetItemDelegate,
    _step_toolbar_button,
)

if typing.TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    from erlab.interactive._figurecomposer._operations._base import StepSection


_OPERATION_LIST_STEP_COLUMN = 0
_OPERATION_LIST_TARGET_COLUMN = 1
_OPERATION_LIST_STATUS_COLUMN = 2
_OPERATION_LIST_TARGET_ROLE = QtCore.Qt.ItemDataRole.UserRole + 1
_OPERATION_LIST_STATUS_ROLE = QtCore.Qt.ItemDataRole.UserRole + 2
_RETIRED_EDITOR_DRAIN_DELAY_MS = 100
_COMBO_POPUP_REBUILD_GRACE_MS = 150
_COMBO_INTERACTION_REBUILD_GRACE_MS = 250
_COMBO_TRACKED_PROPERTY = "figure_composer_combo_tracked"
_COMBO_POPUP_GUARD_ID_PROPERTY = "figure_composer_combo_popup_guard_id"


class FigureOperationAction(typing.NamedTuple):
    """Presentation state for one Add Step menu action."""

    action_id: str
    text: str
    tooltip: str


class FigureOperationRow(typing.NamedTuple):
    """Immutable presentation state for one recipe-step row."""

    operation_id: str
    display: str
    enabled: bool
    tooltip: str
    target_descriptor: tuple[object, ...]
    target_description: str
    status: str
    status_codes: tuple[str, ...]
    status_tooltip: str


class _FigureComposerStepEditorScroll(QtWidgets.QScrollArea):
    """Scroll vertically without allowing editor content to clip horizontally."""

    def minimumSizeHint(self) -> QtCore.QSize:
        hint = super().minimumSizeHint()
        content = self.widget()
        if content is None:
            return hint
        width = content.minimumSizeHint().width() + 2 * self.frameWidth()
        if (
            self.verticalScrollBarPolicy()
            != QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        ):
            scrollbar = typing.cast("QtWidgets.QScrollBar", self.verticalScrollBar())
            width += scrollbar.sizeHint().width()
        return QtCore.QSize(max(hint.width(), width), hint.height())


class _FigureComposerStepEditorPage(QtWidgets.QWidget):
    """Editor page that clears itself with the enclosing tab pane background."""

    def __init__(
        self,
        editor_tabs: QtWidgets.QTabWidget,
        parent: QtWidgets.QWidget,
    ) -> None:
        super().__init__(parent)
        self._editor_tabs = editor_tabs
        self._background_color = self.palette().color(QtGui.QPalette.ColorRole.Window)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

    def refresh_background(self) -> None:
        source = self._editor_tabs
        sample_point = source.rect().center()
        background = QtGui.QPixmap(1, 1)
        background.fill(self._background_color)
        source.render(
            background,
            QtCore.QPoint(),
            QtGui.QRegion(QtCore.QRect(sample_point, QtCore.QSize(1, 1))),
            QtWidgets.QWidget.RenderFlag.DrawWindowBackground,
        )
        color = background.toImage().pixelColor(0, 0)
        if color.isValid() and color.alpha() != 0:
            self._background_color = color
            self.update()

    def paintEvent(self, event: QtGui.QPaintEvent | None) -> None:
        painter = QtGui.QPainter(self)
        try:
            painter.fillRect(self.rect(), self._background_color)
        finally:
            painter.end()
        if event is not None:
            super().paintEvent(event)

    def changeEvent(self, event: QtCore.QEvent | None) -> None:
        if event is not None:
            super().changeEvent(event)
        if event is not None and event.type() in {
            QtCore.QEvent.Type.ApplicationPaletteChange,
            QtCore.QEvent.Type.PaletteChange,
            QtCore.QEvent.Type.StyleChange,
        }:
            self._background_color = self.palette().color(
                QtGui.QPalette.ColorRole.Window
            )
            erlab.interactive.utils.single_shot(self, 0, self.refresh_background)


class _FigureComposerOperationList(ReorderList):
    copy_requested = QtCore.Signal()
    cut_requested = QtCore.Signal()
    paste_requested = QtCore.Signal()
    context_menu_requested = QtCore.Signal(QtCore.QPoint)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(_OPERATION_LIST_STEP_COLUMN, parent)
        self.setColumnCount(3)
        self.setHeaderLabels(("Step", "Target", "Status"))
        self.setRootIsDecorated(False)
        self.setItemsExpandable(False)
        self.setIndentation(0)
        self.setUniformRowHeights(True)
        self.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.context_menu_requested)

    def _operation_ids(self) -> tuple[str, ...]:
        return self._row_ids()

    def keyPressEvent(self, event: QtGui.QKeyEvent | None) -> None:
        if event is None:
            return
        if event_requests_context_menu(event):
            item = self.currentItem()
            rect = self.visualItemRect(item) if item is not None else QtCore.QRect()
            self.context_menu_requested.emit(rect.center())
            event.accept()
            return
        if event.matches(QtGui.QKeySequence.StandardKey.Copy):
            self.copy_requested.emit()
            event.accept()
            return
        if event.matches(QtGui.QKeySequence.StandardKey.Cut):
            self.cut_requested.emit()
            event.accept()
            return
        if event.matches(QtGui.QKeySequence.StandardKey.Paste):
            self.paste_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class FigureOperationPanel(QtWidgets.QWidget):
    """Recipe-step view that emits stable-ID operation intentions."""

    add_requested = QtCore.Signal(str)
    copy_requested = QtCore.Signal()
    cut_requested = QtCore.Signal()
    paste_requested = QtCore.Signal()
    delete_requested = QtCore.Signal()
    duplicate_requested = QtCore.Signal()
    move_requested = QtCore.Signal(int)
    reorder_requested = QtCore.Signal(object, object, object)
    enabled_requested = QtCore.Signal(str, bool)
    selection_changed = QtCore.Signal(object, object)

    def __init__(
        self,
        editor_tabs: QtWidgets.QTabWidget,
        add_actions: Sequence[FigureOperationAction],
    ) -> None:
        super().__init__(editor_tabs)
        self._operation_viewport: QtWidgets.QWidget | None = None
        self.setObjectName("figureComposerRecipePage")
        self._editor_tabs = editor_tabs
        self._selection_input_event = False
        self._multi_select_event = False
        self._emitted_selection_state: tuple[str | None, frozenset[str]] | None = None
        self._selection_notification_pending = False
        self._context_menu: QtWidgets.QMenu | None = None
        self._can_duplicate = False
        self._can_move_up = False
        self._can_move_down = False
        self._target_delegate: _AxesTargetItemDelegate | None = None
        self._sections: tuple[StepSection, ...] = ()
        self._section_buttons: dict[str, QtWidgets.QToolButton] = {}
        self._current_section_key = "sources"
        self._section_tab_stop_refs: dict[
            str, weakref.ReferenceType[QtWidgets.QWidget]
        ] = {}
        self._step_tab_order_update_pending = False
        self._transient_editor_pages: list[QtWidgets.QWidget] = []
        self._retired_editor_widgets: list[QtWidgets.QWidget] = []
        self._retired_editor_drain_pending = False
        self._combo_popup_guard_tokens: set[int] = set()
        self._next_combo_popup_guard_token = 0
        self._tracked_combo_refs: list[weakref.ReferenceType[QtWidgets.QComboBox]] = []
        self._closing = False
        self._build_ui(add_actions)

    def _build_ui(self, add_actions: Sequence[FigureOperationAction]) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        action_layout = QtWidgets.QHBoxLayout()
        action_layout.setSpacing(4)
        self.add_step_button = _step_toolbar_button(
            self,
            "figureComposerAddStepButton",
            "Add Step ▾",
            "Add a plotting, ERLab method, Axes method, or Python step.",
        )
        self.add_step_button.setProperty("uses_inline_menu_arrow", True)
        self.add_step_menu = QtWidgets.QMenu(self.add_step_button)
        self.add_step_menu.setObjectName("figureComposerAddStepMenu")
        for action_state in add_actions:
            action = QtGui.QAction(action_state.text, self.add_step_menu)
            action.setData(action_state.action_id)
            action.setToolTip(action_state.tooltip)
            action.triggered.connect(
                lambda _checked=False, action_id=action_state.action_id: (
                    self.add_requested.emit(action_id)
                )
            )
            self.add_step_menu.addAction(action)
        self.add_step_button.clicked.connect(self._show_add_menu)
        action_layout.addWidget(self.add_step_button)

        self.copy_button = _step_toolbar_button(
            self,
            "figureComposerCopyStepButton",
            "Copy",
            "Copy the selected recipe step or steps.",
        )
        self.copy_button.clicked.connect(self.copy_requested)
        action_layout.addWidget(self.copy_button)
        self.cut_button = _step_toolbar_button(
            self,
            "figureComposerCutStepButton",
            "Cut",
            "Cut the selected recipe step or steps.",
        )
        self.cut_button.clicked.connect(self.cut_requested)
        action_layout.addWidget(self.cut_button)
        self.paste_button = _step_toolbar_button(
            self,
            "figureComposerPasteStepButton",
            "Paste",
            "Paste copied recipe steps after the current selection.",
        )
        self.paste_button.clicked.connect(self.paste_requested)
        action_layout.addWidget(self.paste_button)
        self.delete_button = _step_toolbar_button(
            self,
            "figureComposerDeleteStepButton",
            "Delete",
            "Remove the selected recipe step or steps.",
        )
        self.delete_button.clicked.connect(self.delete_requested)
        action_layout.addWidget(self.delete_button)
        action_layout.addStretch(1)
        layout.addLayout(action_layout)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.splitter.setObjectName("figureComposerRecipeSplitter")
        self.splitter.setChildrenCollapsible(False)
        layout.addWidget(self.splitter, 1)

        self.operation_list = _FigureComposerOperationList(self)
        self.operation_list.setObjectName("figureComposerOperationList")
        self.operation_list.copy_requested.connect(self.copy_requested)
        self.operation_list.cut_requested.connect(self.cut_requested)
        self.operation_list.paste_requested.connect(self.paste_requested)
        self.operation_list.context_menu_requested.connect(self._show_context_menu)
        self.operation_list.rows_reordered.connect(self.reorder_requested)
        self.operation_list.currentItemChanged.connect(self._current_item_changed)
        self.operation_list.itemSelectionChanged.connect(self._selection_did_change)
        self.operation_list.itemChanged.connect(self._item_changed)
        self.operation_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.operation_list.setMinimumHeight(140)
        self.operation_list.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.operation_list.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        header = typing.cast("QtWidgets.QHeaderView", self.operation_list.header())
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(48)
        header.setSectionResizeMode(
            _OPERATION_LIST_STEP_COLUMN,
            QtWidgets.QHeaderView.ResizeMode.Stretch,
        )
        header.setSectionResizeMode(
            _OPERATION_LIST_TARGET_COLUMN,
            QtWidgets.QHeaderView.ResizeMode.Interactive,
        )
        header.setSectionResizeMode(
            _OPERATION_LIST_STATUS_COLUMN,
            QtWidgets.QHeaderView.ResizeMode.Interactive,
        )
        header.resizeSection(_OPERATION_LIST_TARGET_COLUMN, 72)
        header.resizeSection(_OPERATION_LIST_STATUS_COLUMN, 88)
        self.splitter.addWidget(self.operation_list)

        self.inspector = QtWidgets.QWidget(self)
        self.inspector.setObjectName("figureComposerStepInspector")
        self.inspector.setAutoFillBackground(False)
        inspector_layout = QtWidgets.QHBoxLayout(self.inspector)
        inspector_layout.setContentsMargins(0, 0, 0, 0)
        inspector_layout.setSpacing(6)
        self.splitter.addWidget(self.inspector)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes((140, 410))

        self.navigator = QtWidgets.QWidget(self.inspector)
        self.navigator.setObjectName("figureComposerStepNavigator")
        self.navigator.setFixedWidth(150)
        self.navigator_layout = QtWidgets.QVBoxLayout(self.navigator)
        self.navigator_layout.setContentsMargins(0, 0, 0, 0)
        self.navigator_layout.setSpacing(3)
        self.navigator_layout.addStretch(1)
        inspector_layout.addWidget(self.navigator)

        self.editor_scroll = _FigureComposerStepEditorScroll(self.inspector)
        self.editor_scroll.setObjectName("figureComposerStepEditorScroll")
        self.editor_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.editor_scroll.setWidgetResizable(True)
        self.editor_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.editor_scroll.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.editor_scroll.setAutoFillBackground(False)
        viewport = typing.cast("QtWidgets.QWidget", self.editor_scroll.viewport())
        viewport.setObjectName("figureComposerStepEditorViewport")
        viewport.setAutoFillBackground(False)
        inspector_layout.addWidget(self.editor_scroll, 1)

        self.editor_stack = QtWidgets.QStackedWidget()
        self.editor_stack.setObjectName("figureComposerStepSectionStack")
        self.editor_stack.setAutoFillBackground(False)
        self.editor_scroll.setWidget(self.editor_stack)

        operation_viewport = typing.cast(
            "QtWidgets.QWidget", self.operation_list.viewport()
        )
        self._operation_viewport = operation_viewport
        operation_viewport.installEventFilter(self)

    def create_editor_page(
        self, object_name: str, *, transient: bool = False
    ) -> _FigureComposerStepEditorPage:
        """Create an editor page owned by this panel.

        Transient pages are retired safely when the next editor is prepared;
        persistent pages can be reused across operation selections.
        """
        page = _FigureComposerStepEditorPage(self._editor_tabs, self.editor_stack)
        page.setObjectName(object_name)
        if transient:
            self._transient_editor_pages.append(page)
        return page

    @property
    def section_keys(self) -> tuple[str, ...]:
        """Keys of the currently mounted editor sections."""
        return tuple(section.key for section in self._sections)

    def _current_editor_page(self) -> QtWidgets.QWidget | None:
        return self.editor_stack.currentWidget()

    def refresh_current_editor_background(self) -> None:
        """Refresh the current editor page after a palette or window change."""
        current_page = self._current_editor_page()
        if isinstance(current_page, _FigureComposerStepEditorPage):
            current_page.refresh_background()

    def prepare_editor_rebuild(self) -> None:
        """Unmount the current editor and safely retire transient pages."""
        self._section_tab_stop_refs.clear()
        while self.editor_stack.count():
            page = typing.cast("QtWidgets.QWidget", self.editor_stack.widget(0))
            page.hide()
            self.editor_stack.removeWidget(page)
        pages = self._transient_editor_pages
        self._transient_editor_pages = []
        for page in pages:
            self._retire_editor_widget(page)

    def replace_sections(
        self,
        sections: Sequence[StepSection],
        *,
        summaries: Mapping[str, str],
    ) -> None:
        """Mount a complete editor-section presentation atomically."""
        for button in self._section_buttons.values():
            button.setObjectName("")
            self._retire_editor_widget(button)
        self._section_buttons.clear()
        self._sections = tuple(sections)

        for index, section in enumerate(self._sections):
            self.editor_stack.addWidget(section.page)
            button = QtWidgets.QToolButton(self.navigator)
            button.setText(self._section_button_text(section, summaries))
            button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
            button.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
            button.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            button.setObjectName(f"figureComposerSection_{section.key}")
            button.setProperty("section_title", section.title)
            button.setProperty("section_tooltip", section.tooltip)
            button.setToolTip(section.tooltip)
            button.clicked.connect(
                lambda _checked=False, key=section.key: self.select_section(key)
            )
            self.navigator_layout.insertWidget(index, button)
            self._section_buttons[section.key] = button

        existing_key = self._current_section_key
        keys = self.section_keys
        key = existing_key if existing_key in keys else keys[0] if keys else ""
        if key:
            self.select_section(key)
        self.editor_scroll.updateGeometry()
        self.inspector.updateGeometry()

    def select_section(self, key: str) -> None:
        """Show one mounted editor section by semantic key."""
        keys = self.section_keys
        if key not in keys:
            return
        self._current_section_key = key
        self.editor_stack.setCurrentIndex(keys.index(key))
        for button_key, button in self._section_buttons.items():
            selected = button_key == key
            font = button.font()
            if font.bold() != selected:
                font.setBold(selected)
                button.setFont(font)
        self.refresh_current_editor_background()
        self._queue_step_tab_order_refresh()

    def set_section_summaries(self, summaries: Mapping[str, str]) -> None:
        """Refresh navigation labels without rebuilding editor pages."""
        for section in self._sections:
            button = self._section_buttons.get(section.key)
            if button is not None:
                button.setText(self._section_button_text(section, summaries))

    @staticmethod
    def _section_button_text(section: StepSection, summaries: Mapping[str, str]) -> str:
        summary = summaries.get(section.key, "")
        return f"{section.title}: {summary}" if summary else section.title

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
        """Disable and defer deletion of a replaced editor widget safely."""
        if not erlab.interactive.utils.qt_is_valid(widget):
            return
        self._block_signals_recursive(widget)
        widget.setEnabled(False)
        widget.hide()
        if widget not in self._retired_editor_widgets:
            self._retired_editor_widgets.append(widget)
        self._queue_retired_editor_drain()

    def retire_form_controls(self, layout: QtWidgets.QFormLayout) -> None:
        """Retire every widget currently owned by a dynamic form layout."""
        while layout.count():
            item = layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                self._retire_editor_widget(widget)

    def _queue_retired_editor_drain(self) -> None:
        if self._retired_editor_drain_pending or self._closing:
            return
        self._retired_editor_drain_pending = True
        erlab.interactive.utils.single_shot(
            self,
            _RETIRED_EDITOR_DRAIN_DELAY_MS,
            self._drain_retired_editor_widgets,
        )

    def _drain_retired_editor_widgets(self) -> None:
        self._retired_editor_drain_pending = False
        if not erlab.interactive.utils.qt_is_valid(self) or self._closing:
            return
        if (
            not self._retired_editor_widgets
            or self.editor_rebuild_must_wait()
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

    def track_editor_controls(self, widget: QtWidgets.QWidget) -> None:
        """Track combo popups that can make an editor rebuild unsafe."""
        if isinstance(widget, QtWidgets.QComboBox):
            self._track_combo_interaction(widget)
        for combo in widget.findChildren(QtWidgets.QComboBox):
            self._track_combo_interaction(combo)

    def _track_combo_interaction(self, combo: QtWidgets.QComboBox) -> None:
        if combo.property(_COMBO_TRACKED_PROPERTY):
            return
        combo.setProperty(_COMBO_TRACKED_PROPERTY, True)
        combo.installEventFilter(self)
        view = combo.view()
        if view is not None and erlab.interactive.utils.qt_is_valid(view):
            view.installEventFilter(self)
        self._tracked_combo_refs.append(weakref.ref(combo))

    def _combo_for_popup_view(
        self, view: QtWidgets.QAbstractItemView
    ) -> QtWidgets.QComboBox | None:
        matched_combo: QtWidgets.QComboBox | None = None
        for combo in self._live_tracked_combos():
            combo_view = combo.view()
            if combo_view is not None and combo_view is view:
                matched_combo = combo
        return matched_combo

    def _live_tracked_combos(self) -> tuple[QtWidgets.QComboBox, ...]:
        live_refs: list[weakref.ReferenceType[QtWidgets.QComboBox]] = []
        live_combos: list[QtWidgets.QComboBox] = []
        for combo_ref in self._tracked_combo_refs:
            combo = combo_ref()
            if combo is None or not erlab.interactive.utils.qt_is_valid(combo):
                continue
            live_refs.append(combo_ref)
            live_combos.append(combo)
        self._tracked_combo_refs = live_refs
        return tuple(live_combos)

    def _tracked_combo_popup_is_visible(self) -> bool:
        for combo in self._live_tracked_combos():
            view = combo.view()
            if view is not None and view.isVisible():
                return True
        return False

    def _begin_transient_combo_popup_guard(self) -> None:
        token = self._begin_combo_popup_guard()
        self._schedule_combo_popup_guard_end(token, _COMBO_INTERACTION_REBUILD_GRACE_MS)

    def _begin_combo_popup_view_guard(self, view: QtWidgets.QAbstractItemView) -> None:
        if isinstance(view.property(_COMBO_POPUP_GUARD_ID_PROPERTY), int):
            return
        view.setProperty(
            _COMBO_POPUP_GUARD_ID_PROPERTY,
            self._begin_combo_popup_guard(),
        )

    def _end_combo_popup_view_guard(self, view: QtWidgets.QAbstractItemView) -> None:
        token = view.property(_COMBO_POPUP_GUARD_ID_PROPERTY)
        if not isinstance(token, int):
            return
        view.setProperty(_COMBO_POPUP_GUARD_ID_PROPERTY, None)
        self._schedule_combo_popup_guard_end(token, _COMBO_POPUP_REBUILD_GRACE_MS)

    def _begin_combo_popup_guard(self) -> int:
        token = self._next_combo_popup_guard_token
        self._next_combo_popup_guard_token += 1
        self._combo_popup_guard_tokens.add(token)
        return token

    def _schedule_combo_popup_guard_end(self, token: int, delay_ms: int) -> None:
        def discard_guard() -> None:
            self._combo_popup_guard_tokens.discard(token)

        erlab.interactive.utils.single_shot(self, delay_ms, discard_guard)

    def editor_rebuild_must_wait(self) -> bool:
        """Return whether a live popup or interaction makes rebuilding unsafe."""
        if self._combo_popup_guard_tokens or self._tracked_combo_popup_is_visible():
            return True
        popup = QtWidgets.QApplication.activePopupWidget()
        return popup is not None and erlab.interactive.utils.qt_is_valid(popup)

    def contains_editor_widget(self, widget: QtWidgets.QWidget) -> bool:
        """Return whether a widget belongs to the mounted operation editor."""
        roots = (self.editor_stack, self.navigator)
        return any(root is widget or root.isAncestorOf(widget) for root in roots)

    @staticmethod
    def _accepts_tab_focus(widget: QtWidgets.QWidget) -> bool:
        return bool(widget.focusPolicy() & QtCore.Qt.FocusPolicy.TabFocus)

    def _first_editor_tab_stop(self) -> QtWidgets.QWidget | None:
        current_page = self._current_editor_page()
        if current_page is None:
            return None
        cache_key = self._current_section_key
        cached_ref = self._section_tab_stop_refs.get(cache_key)
        if cached_ref is not None:
            cached_widget = cached_ref()
            if (
                cached_widget is not None
                and erlab.interactive.utils.qt_is_valid(cached_widget)
                and cached_widget.isEnabled()
                and self._accepts_tab_focus(cached_widget)
                and (
                    cached_widget is current_page
                    or current_page.isAncestorOf(cached_widget)
                )
            ):
                return cached_widget
            self._section_tab_stop_refs.pop(cache_key, None)
        if (
            erlab.interactive.utils.qt_is_valid(current_page)
            and current_page.isEnabled()
            and self._accepts_tab_focus(current_page)
        ):
            self._section_tab_stop_refs[cache_key] = weakref.ref(current_page)
            return current_page
        for widget in current_page.findChildren(QtWidgets.QWidget):
            if (
                erlab.interactive.utils.qt_is_valid(widget)
                and widget.isEnabled()
                and self._accepts_tab_focus(widget)
            ):
                self._section_tab_stop_refs[cache_key] = weakref.ref(widget)
                return widget
        return None

    def _refresh_step_tab_order(self) -> None:
        buttons = list(self._section_buttons.values())
        if not buttons:
            return
        tab_chain = [
            self.add_step_button,
            self.copy_button,
            self.cut_button,
            self.paste_button,
            self.delete_button,
            self.operation_list,
            *buttons,
        ]
        next_widget = self._first_editor_tab_stop()
        if next_widget is not None:
            tab_chain.append(next_widget)
        for index, widget in enumerate(tab_chain[:-1]):
            QtWidgets.QWidget.setTabOrder(widget, tab_chain[index + 1])

    def _queue_step_tab_order_refresh(self) -> None:
        if self._step_tab_order_update_pending or self._closing:
            return
        self._step_tab_order_update_pending = True
        erlab.interactive.utils.single_shot(
            self, 0, self._run_queued_step_tab_order_refresh
        )

    def _run_queued_step_tab_order_refresh(self) -> None:
        if not erlab.interactive.utils.qt_is_valid(self):
            return
        self._step_tab_order_update_pending = False
        self._refresh_step_tab_order()

    @staticmethod
    def _operation_id_for_item(
        item: QtWidgets.QTreeWidgetItem | None,
    ) -> str | None:
        if item is None:
            return None
        operation_id = item.data(
            _OPERATION_LIST_STEP_COLUMN, QtCore.Qt.ItemDataRole.UserRole
        )
        return operation_id if isinstance(operation_id, str) else None

    def selected_ids(self) -> frozenset[str]:
        return frozenset(
            operation_id
            for item in self.operation_list.selectedItems()
            if (operation_id := self._operation_id_for_item(item)) is not None
        )

    def current_id(self) -> str | None:
        return self._operation_id_for_item(self.operation_list.currentItem())

    def current_index(self) -> int:
        item = self.operation_list.currentItem()
        return -1 if item is None else self.operation_list.indexOfTopLevelItem(item)

    def select_row(self, index: int) -> None:
        """Select a row and notify consumers through the normal selection signals."""
        item = self.operation_list.topLevelItem(index)
        if item is None:
            self.operation_list.setCurrentIndex(QtCore.QModelIndex())
        else:
            self.operation_list.setCurrentItem(item)

    def set_current_row(self, index: int, *, preserve_selection: bool = True) -> None:
        selected_ids = self.selected_ids() if preserve_selection else frozenset()
        was_blocked = self.operation_list.blockSignals(True)
        try:
            if not preserve_selection:
                self.operation_list.clearSelection()
            item = self.operation_list.topLevelItem(index)
            if item is None:
                self.operation_list.setCurrentIndex(QtCore.QModelIndex())
            else:
                self.operation_list.setCurrentItem(item)
            if preserve_selection and selected_ids:
                self._apply_selected_ids(selected_ids)
        finally:
            self.operation_list.blockSignals(was_blocked)
        self._synchronize_selection_cache()

    def install_target_delegate(self, color_source: QtWidgets.QWidget) -> None:
        """Install the operation-target preview delegate owned by this panel."""
        delegate = _AxesTargetItemDelegate(
            int(_OPERATION_LIST_TARGET_ROLE), color_source, self.operation_list
        )
        self.operation_list.setItemDelegateForColumn(
            _OPERATION_LIST_TARGET_COLUMN, delegate
        )
        self._target_delegate = delegate

    def set_selected_ids(self, operation_ids: Collection[str]) -> None:
        selected = set(operation_ids)
        was_blocked = self.operation_list.blockSignals(True)
        try:
            self._apply_selected_ids(selected)
        finally:
            self.operation_list.blockSignals(was_blocked)
        self._synchronize_selection_cache()

    def _apply_selected_ids(self, operation_ids: Collection[str]) -> None:
        """Apply selection while the caller owns signal blocking and cache sync."""
        selected = set(operation_ids)
        for row in range(self.operation_list.topLevelItemCount()):
            item = self.operation_list.topLevelItem(row)
            if item is not None:
                item.setSelected(self._operation_id_for_item(item) in selected)

    def set_rows(
        self,
        rows: Sequence[FigureOperationRow],
        *,
        selected_ids: Collection[str],
        current_id: str | None,
    ) -> None:
        """Render operation rows while preserving stable-ID selection."""
        operation_ids = tuple(row.operation_id for row in rows)
        reuse_items = self.operation_list._operation_ids() == operation_ids
        was_blocked = self.operation_list.blockSignals(True)
        try:
            if not reuse_items:
                self.operation_list.clear()
                self.operation_list.addTopLevelItems(
                    [QtWidgets.QTreeWidgetItem() for _row in rows]
                )
            for index, row in enumerate(rows):
                item = typing.cast(
                    "QtWidgets.QTreeWidgetItem",
                    self.operation_list.topLevelItem(index),
                )
                self._set_item(item, row)
            if not reuse_items:
                self._apply_selected_ids(selected_ids)
                if current_id is None:
                    self.operation_list.setCurrentIndex(QtCore.QModelIndex())
                else:
                    for index, row in enumerate(rows):
                        if row.operation_id == current_id:
                            self.operation_list.setCurrentItem(
                                self.operation_list.topLevelItem(index)
                            )
                            break
        finally:
            self.operation_list.blockSignals(was_blocked)
        self._synchronize_selection_cache()

    @staticmethod
    def _set_item(item: QtWidgets.QTreeWidgetItem, row: FigureOperationRow) -> None:
        item.setText(_OPERATION_LIST_STEP_COLUMN, row.display)
        item.setText(_OPERATION_LIST_STATUS_COLUMN, row.status)
        item.setFlags(
            (
                item.flags()
                | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                | QtCore.Qt.ItemFlag.ItemIsDragEnabled
            )
            & ~QtCore.Qt.ItemFlag.ItemIsDropEnabled
        )
        item.setCheckState(
            _OPERATION_LIST_STEP_COLUMN,
            QtCore.Qt.CheckState.Checked
            if row.enabled
            else QtCore.Qt.CheckState.Unchecked,
        )
        item.setData(
            _OPERATION_LIST_STEP_COLUMN,
            QtCore.Qt.ItemDataRole.UserRole,
            row.operation_id,
        )
        item.setData(
            _OPERATION_LIST_TARGET_COLUMN,
            _OPERATION_LIST_TARGET_ROLE,
            row.target_descriptor,
        )
        item.setData(
            _OPERATION_LIST_STATUS_COLUMN,
            _OPERATION_LIST_STATUS_ROLE,
            row.status_codes,
        )
        item.setSizeHint(_OPERATION_LIST_STEP_COLUMN, QtCore.QSize(0, 22))
        item.setToolTip(_OPERATION_LIST_STEP_COLUMN, row.tooltip)
        item.setData(
            _OPERATION_LIST_STEP_COLUMN,
            QtCore.Qt.ItemDataRole.AccessibleDescriptionRole,
            row.tooltip,
        )
        item.setToolTip(_OPERATION_LIST_TARGET_COLUMN, row.target_description)
        item.setData(
            _OPERATION_LIST_TARGET_COLUMN,
            QtCore.Qt.ItemDataRole.AccessibleDescriptionRole,
            row.target_description,
        )
        item.setToolTip(_OPERATION_LIST_STATUS_COLUMN, row.status_tooltip)
        item.setData(
            _OPERATION_LIST_STATUS_COLUMN,
            QtCore.Qt.ItemDataRole.AccessibleDescriptionRole,
            row.status_tooltip,
        )
        item.setForeground(
            _OPERATION_LIST_STATUS_COLUMN,
            QtGui.QBrush(QtGui.QColor("darkRed"))
            if row.status_codes
            else QtGui.QBrush(),
        )

    def set_action_availability(
        self,
        *,
        selection: bool,
        paste: bool,
        duplicate: bool,
        move_up: bool,
        move_down: bool,
    ) -> None:
        self.delete_button.setEnabled(selection)
        self.copy_button.setEnabled(selection)
        self.cut_button.setEnabled(selection)
        self.paste_button.setEnabled(paste)
        self._can_duplicate = duplicate
        self._can_move_up = move_up
        self._can_move_down = move_down

    @QtCore.Slot()
    def _show_add_menu(self) -> None:
        self.add_step_menu.popup(
            self.add_step_button.mapToGlobal(
                QtCore.QPoint(0, self.add_step_button.height())
            )
        )

    @QtCore.Slot(QtCore.QPoint)
    def _show_context_menu(self, position: QtCore.QPoint) -> None:
        menu = QtWidgets.QMenu("Recipe Steps", self.operation_list)
        self._context_menu = menu

        def clear_menu(_destroyed: object | None = None) -> None:
            if self._context_menu is menu:
                self._context_menu = None

        menu.destroyed.connect(clear_menu)
        menu.aboutToHide.connect(menu.deleteLater)
        for text, object_name, enabled, signal in (
            (
                "Copy",
                "figureComposerContextCopyStepsAction",
                self.copy_button.isEnabled(),
                self.copy_requested,
            ),
            (
                "Cut",
                "figureComposerContextCutStepsAction",
                self.cut_button.isEnabled(),
                self.cut_requested,
            ),
            (
                "Paste",
                "figureComposerContextPasteStepsAction",
                self.paste_button.isEnabled(),
                self.paste_requested,
            ),
        ):
            action = QtGui.QAction(text, menu)
            action.setObjectName(object_name)
            action.setEnabled(enabled)
            action.triggered.connect(signal)
            menu.addAction(action)
        menu.addSeparator()
        duplicate_action = QtGui.QAction("Duplicate", menu)
        duplicate_action.setObjectName("figureComposerContextDuplicateStepAction")
        duplicate_action.setEnabled(self._can_duplicate)
        duplicate_action.triggered.connect(self.duplicate_requested)
        menu.addAction(duplicate_action)
        for text, object_name, offset, enabled in (
            (
                "Move Up",
                "figureComposerContextMoveStepUpAction",
                -1,
                self._can_move_up,
            ),
            (
                "Move Down",
                "figureComposerContextMoveStepDownAction",
                1,
                self._can_move_down,
            ),
        ):
            action = QtGui.QAction(text, menu)
            action.setObjectName(object_name)
            action.setEnabled(enabled)
            action.triggered.connect(
                lambda _checked=False, direction=offset: self.move_requested.emit(
                    direction
                )
            )
            menu.addAction(action)
        delete_action = QtGui.QAction("Delete", menu)
        delete_action.setObjectName("figureComposerContextDeleteStepAction")
        delete_action.setEnabled(self.delete_button.isEnabled())
        delete_action.triggered.connect(self.delete_requested)
        menu.addAction(delete_action)
        viewport = typing.cast("QtWidgets.QWidget", self.operation_list.viewport())
        menu.popup(viewport.mapToGlobal(position))

    @QtCore.Slot(QtWidgets.QTreeWidgetItem, QtWidgets.QTreeWidgetItem)
    def _current_item_changed(
        self,
        current: QtWidgets.QTreeWidgetItem | None,
        _previous: QtWidgets.QTreeWidgetItem | None,
    ) -> None:
        if current is not None and not self._multi_select_event:
            operation_id = self._operation_id_for_item(current)
            if operation_id is not None:
                was_blocked = self.operation_list.blockSignals(True)
                try:
                    self._apply_selected_ids((operation_id,))
                finally:
                    self.operation_list.blockSignals(was_blocked)
        self._selection_did_change()

    @QtCore.Slot()
    def _selection_did_change(self) -> None:
        if self._closing:
            return
        if self._selection_input_event:
            if not self._selection_notification_pending:
                self._selection_notification_pending = True
                erlab.interactive.utils.single_shot(
                    self, 0, self._emit_selection_change
                )
            return
        self._emit_selection_change()

    def _emit_selection_change(self) -> None:
        self._selection_notification_pending = False
        if self._closing or not erlab.interactive.utils.qt_is_valid(self):
            return
        state = (self.current_id(), self.selected_ids())
        if state == self._emitted_selection_state:
            return
        self._emitted_selection_state = state
        self.selection_changed.emit(*state)

    def _synchronize_selection_cache(self) -> None:
        """Record a signal-blocked selection without notifying consumers."""
        self._emitted_selection_state = (self.current_id(), self.selected_ids())

    @QtCore.Slot(QtWidgets.QTreeWidgetItem, int)
    def _item_changed(self, item: QtWidgets.QTreeWidgetItem, column: int) -> None:
        if column != _OPERATION_LIST_STEP_COLUMN:
            return
        operation_id = self._operation_id_for_item(item)
        if operation_id is not None:
            self.enabled_requested.emit(
                operation_id,
                item.checkState(_OPERATION_LIST_STEP_COLUMN)
                == QtCore.Qt.CheckState.Checked,
            )

    @staticmethod
    def _modifiers_enable_multi_selection(
        modifiers: QtCore.Qt.KeyboardModifier,
    ) -> bool:
        multi_modifiers = (
            QtCore.Qt.KeyboardModifier.ShiftModifier
            | QtCore.Qt.KeyboardModifier.ControlModifier
            | QtCore.Qt.KeyboardModifier.MetaModifier
        )
        return bool(modifiers & multi_modifiers)

    def _clear_selection_input_state(self) -> None:
        if erlab.interactive.utils.qt_is_valid(self):
            self._multi_select_event = False
            self._selection_input_event = False

    def eventFilter(
        self, watched: QtCore.QObject | None, event: QtCore.QEvent | None
    ) -> bool:
        self._handle_combo_interaction_event(watched, event)
        if (
            self._operation_viewport is not None
            and watched is self._operation_viewport
            and event is not None
            and event.type()
            in {
                QtCore.QEvent.Type.MouseButtonPress,
                QtCore.QEvent.Type.MouseButtonDblClick,
                QtCore.QEvent.Type.KeyPress,
            }
        ):
            input_event = typing.cast("QtGui.QInputEvent", event)
            self._selection_input_event = True
            self._multi_select_event = self._modifiers_enable_multi_selection(
                input_event.modifiers()
            )
            erlab.interactive.utils.single_shot(
                self, 0, self._clear_selection_input_state
            )
        return super().eventFilter(watched, event)

    def _handle_combo_interaction_event(
        self, watched: QtCore.QObject | None, event: QtCore.QEvent | None
    ) -> None:
        if event is None or watched is None:
            return
        event_type = event.type()
        if isinstance(watched, QtWidgets.QComboBox):
            if event_type in {
                QtCore.QEvent.Type.MouseButtonPress,
                QtCore.QEvent.Type.KeyPress,
                QtCore.QEvent.Type.Wheel,
            }:
                self._begin_transient_combo_popup_guard()
            return
        if not isinstance(watched, QtWidgets.QAbstractItemView):
            return
        if self._combo_for_popup_view(watched) is None:
            return
        if event_type in {QtCore.QEvent.Type.Show, QtCore.QEvent.Type.ShowToParent}:
            self._begin_combo_popup_view_guard(watched)
        elif event_type in {
            QtCore.QEvent.Type.Hide,
            QtCore.QEvent.Type.HideToParent,
            QtCore.QEvent.Type.Close,
        }:
            self._end_combo_popup_view_guard(watched)

    def release(self) -> None:
        """Detach event filters before tool teardown."""
        self._closing = True
        viewport = self._operation_viewport
        self._operation_viewport = None
        self._multi_select_event = False
        self._selection_input_event = False
        self._selection_notification_pending = False
        if viewport is not None and erlab.interactive.utils.qt_is_valid(self, viewport):
            viewport.removeEventFilter(self)
        for combo in self._live_tracked_combos():
            combo.removeEventFilter(self)
            combo_view = combo.view()
            if combo_view is not None and erlab.interactive.utils.qt_is_valid(
                combo_view
            ):
                combo_view.removeEventFilter(self)
        self._tracked_combo_refs.clear()
        self._combo_popup_guard_tokens.clear()

    def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:
        self.release()
        if event is not None:
            super().closeEvent(event)
