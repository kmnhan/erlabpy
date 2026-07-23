"""Operation-editor presentation and Qt lifetime ownership."""

from __future__ import annotations

import dataclasses
import textwrap
import typing
import weakref

from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive._figurecomposer._exceptions import FigureComposerInputError
from erlab.interactive._figurecomposer._ui._editor_controls import (
    MIXED_VALUE,
    MIXED_VALUES_TEXT,
    CheckBoxControlAdapter,
    ComboBoxControlAdapter,
    ComboBoxDataControlAdapter,
    LineEditControlAdapter,
    PlainTextControlAdapter,
    SignalValueControlAdapter,
)
from erlab.interactive._widgets import _Separator

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from matplotlib.axes import Axes

    from erlab.interactive._figurecomposer._model._document import FigureRecipeContext
    from erlab.interactive._figurecomposer._model._state import (
        FigureAxesSelectionState,
        FigureOperationState,
    )


COMMON_SOURCE_SECTION_TOOLTIP = "Choose which data this step uses."
COMMON_AXES_SECTION_TOOLTIP = "Choose which subplot axes this step draws into."

_RETIRED_EDITOR_DRAIN_DELAY_MS = 100
_COMBO_POPUP_REBUILD_GRACE_MS = 150
_COMBO_INTERACTION_REBUILD_GRACE_MS = 250
_COMBO_TRACKED_PROPERTY = "figure_composer_combo_tracked"
_COMBO_POPUP_GUARD_ID_PROPERTY = "figure_composer_combo_popup_guard_id"


@dataclasses.dataclass(frozen=True)
class StepSection:
    """One semantic section of the operation editor."""

    key: str
    title: str
    page: QtWidgets.QWidget
    tooltip: str


@dataclasses.dataclass(frozen=True)
class OperationEditRequest:
    """Stable-ID operation mutation requested by an editor control."""

    operation_ids: tuple[str, ...]
    updater: Callable[[int, FigureOperationState], FigureOperationState]
    render: bool = True
    defer_render: bool = False
    rebuild_editor: bool = False
    defer_editor_rebuild: bool = False
    sync_axes: bool = True


@dataclasses.dataclass(frozen=True)
class OperationRecipeEditRequest:
    """Stable-ID whole-recipe mutation requested by an editor control."""

    operation_ids: tuple[str, ...]
    updater: Callable[
        [Sequence[FigureOperationState], tuple[str, ...]],
        Sequence[FigureOperationState],
    ]
    render: bool = True
    defer_render: bool = False
    rebuild_editor: bool = False
    defer_editor_rebuild: bool = False
    sync_axes: bool = True


@dataclasses.dataclass(frozen=True)
class OperationEditorBinding:
    """Read-only state and semantic queries supplied by the composition root."""

    context: FigureRecipeContext
    current_operation_id: Callable[[], str | None]
    editable_operation_ids: Callable[[], tuple[str, ...]]
    updates_allowed: Callable[[], bool]
    selected_axes_state: Callable[[], FigureAxesSelectionState]
    source_display_name: Callable[[str], str]
    source_tooltip: Callable[[str], str]
    first_live_axis: Callable[[FigureAxesSelectionState], Axes | None]
    subplot_parameter_default: Callable[[str], float]
    rendered_value: Callable[
        [FigureOperationState, Callable[[Sequence[Axes]], typing.Any]], typing.Any
    ]
    styled_rcparams_value: Callable[[str], typing.Any]


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


class FigureOperationEditor(QtWidgets.QWidget):
    """Own operation-editor pages, navigation, and deferred Qt cleanup."""

    edit_requested = QtCore.Signal(object)
    validation_changed = QtCore.Signal()

    def __init__(
        self,
        editor_tabs: QtWidgets.QTabWidget,
        tab_order_prefix: Sequence[QtWidgets.QWidget],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("figureComposerStepInspector")
        self.setAutoFillBackground(False)
        self._editor_tabs = editor_tabs
        self._tab_order_prefix = tuple(tab_order_prefix)
        self._sections: tuple[StepSection, ...] = ()
        self._section_buttons: dict[str, QtWidgets.QToolButton] = {}
        self._current_section_key = "sources"
        self._section_tab_stop_refs: dict[
            str, weakref.ReferenceType[QtWidgets.QWidget]
        ] = {}
        self._tab_order_update_pending = False
        self._transient_pages: list[QtWidgets.QWidget] = []
        self._retired_widgets: list[QtWidgets.QWidget] = []
        self._retired_drain_pending = False
        self._combo_popup_guard_tokens: set[int] = set()
        self._next_combo_popup_guard_token = 0
        self._tracked_combo_refs: list[weakref.ReferenceType[QtWidgets.QComboBox]] = []
        self._binding: OperationEditorBinding | None = None
        self._generation = 0
        self._active_signal_widget: QtWidgets.QWidget | None = None
        self._input_errors: dict[str, dict[str, str]] = {}
        self._closing = False
        self._build_ui()

    def bind(self, binding: OperationEditorBinding) -> None:
        """Attach the read-only document and semantic callbacks used by controls."""
        self._binding = binding

    @property
    def binding(self) -> OperationEditorBinding:
        if self._binding is None:
            raise RuntimeError("Figure operation editor is not bound")
        return self._binding

    @property
    def context(self) -> FigureRecipeContext:
        return self.binding.context

    @property
    def active_signal_widget(self) -> QtWidgets.QWidget | None:
        return self._active_signal_widget

    def _build_ui(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.navigator = QtWidgets.QWidget(self)
        self.navigator.setObjectName("figureComposerStepNavigator")
        self.navigator.setFixedWidth(150)
        self.navigator_layout = QtWidgets.QVBoxLayout(self.navigator)
        self.navigator_layout.setContentsMargins(0, 0, 0, 0)
        self.navigator_layout.setSpacing(3)
        self.navigator_layout.addStretch(1)
        layout.addWidget(self.navigator)

        self.scroll_area = _FigureComposerStepEditorScroll(self)
        self.scroll_area.setObjectName("figureComposerStepEditorScroll")
        self.scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.scroll_area.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.scroll_area.setAutoFillBackground(False)
        viewport = typing.cast("QtWidgets.QWidget", self.scroll_area.viewport())
        viewport.setObjectName("figureComposerStepEditorViewport")
        viewport.setAutoFillBackground(False)
        layout.addWidget(self.scroll_area, 1)

        self.stack = QtWidgets.QStackedWidget()
        self.stack.setObjectName("figureComposerStepSectionStack")
        self.stack.setAutoFillBackground(False)
        self.scroll_area.setWidget(self.stack)

        self._source_page = self.create_page("figureComposerStepSourcesPage")
        source_page_layout = QtWidgets.QVBoxLayout(self._source_page)
        source_page_layout.setContentsMargins(6, 6, 6, 6)
        source_page_layout.setSpacing(4)
        self._source_controls = QtWidgets.QWidget(self._source_page)
        self._source_controls_layout = QtWidgets.QFormLayout(self._source_controls)
        self._source_controls_layout.setContentsMargins(0, 0, 0, 0)
        self._source_controls_layout.setSpacing(4)
        self._source_controls_layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        source_page_layout.addWidget(self._source_controls)
        self._source_status_label = QtWidgets.QLabel(self._source_page)
        self._source_status_label.setObjectName("figureComposerStepSourceStatus")
        self._source_status_label.setWordWrap(True)
        self._source_status_label.setVisible(False)
        source_page_layout.addWidget(self._source_status_label)

    @property
    def source_page(self) -> QtWidgets.QWidget:
        """Persistent editor page for the selected operation's data sources."""
        return self._source_page

    @property
    def source_controls(self) -> QtWidgets.QWidget:
        """Parent widget for controls contributed by an operation source editor."""
        return self._source_controls

    def clear_source_controls(self) -> None:
        """Retire controls contributed by the previously selected operation."""
        self.retire_form_controls(self._source_controls_layout)

    def add_source_row(
        self,
        label: str,
        widget: QtWidgets.QWidget,
        tooltip: str,
    ) -> None:
        """Add one operation-owned control to the persistent source form."""
        self.add_form_row(self._source_controls_layout, label, widget, tooltip)

    def set_source_status(self, text: str | None) -> None:
        """Show a source validation or availability message when needed."""
        self._source_status_label.setText("" if text is None else text)
        self._source_status_label.setVisible(bool(text))

    def create_page(
        self, object_name: str, *, transient: bool = False
    ) -> QtWidgets.QWidget:
        """Create a page whose lifetime is owned by this editor."""
        page = _FigureComposerStepEditorPage(self._editor_tabs, self.stack)
        page.setObjectName(object_name)
        if transient:
            self._transient_pages.append(page)
        return page

    @property
    def section_keys(self) -> tuple[str, ...]:
        """Keys of the currently mounted editor sections."""
        return tuple(section.key for section in self._sections)

    def _current_page(self) -> QtWidgets.QWidget | None:
        return self.stack.currentWidget()

    def refresh_current_background(self) -> None:
        """Refresh the current page after a palette or window change."""
        current_page = self._current_page()
        if isinstance(current_page, _FigureComposerStepEditorPage):
            current_page.refresh_background()

    def prepare_rebuild(self) -> None:
        """Unmount sections and safely retire transient pages."""
        self.flush_pending_commits()
        self._generation += 1
        self._section_tab_stop_refs.clear()
        while self.stack.count():
            page = typing.cast("QtWidgets.QWidget", self.stack.widget(0))
            page.hide()
            self.stack.removeWidget(page)
        pages = self._transient_pages
        self._transient_pages = []
        for page in pages:
            self._retire_widget(page)

    def flush_pending_commits(self, *, render: bool = False) -> None:
        """Commit debounced editor content before pages are replaced or saved."""
        for widget in self.findChildren(QtWidgets.QWidget):
            flush = getattr(
                widget,
                "_figure_composer_custom_code_flush_pending_commit",
                None,
            )
            if callable(flush):
                try:
                    flush(render=render)
                except RuntimeError:
                    continue

    def replace_sections(
        self,
        sections: Sequence[StepSection],
        *,
        summaries: Mapping[str, str],
    ) -> None:
        """Mount a complete editor-section presentation atomically."""
        for button in self._section_buttons.values():
            button.setObjectName("")
            self._retire_widget(button)
        self._section_buttons.clear()
        self._sections = tuple(sections)

        for index, section in enumerate(self._sections):
            self.stack.addWidget(section.page)
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
        self.scroll_area.updateGeometry()
        self.updateGeometry()

    def select_section(self, key: str) -> None:
        """Show one mounted editor section by semantic key."""
        keys = self.section_keys
        if key not in keys:
            return
        self._current_section_key = key
        self.stack.setCurrentIndex(keys.index(key))
        for button_key, button in self._section_buttons.items():
            selected = button_key == key
            font = button.font()
            if font.bold() != selected:
                font.setBold(selected)
                button.setFont(font)
        self.refresh_current_background()
        self._queue_tab_order_refresh()

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

    def _retire_widget(self, widget: QtWidgets.QWidget) -> None:
        """Disable and defer deletion of a replaced editor widget safely."""
        if not erlab.interactive.utils.qt_is_valid(widget):
            return
        self._block_signals_recursive(widget)
        widget.setEnabled(False)
        widget.hide()
        if widget not in self._retired_widgets:
            self._retired_widgets.append(widget)
        self._queue_retired_drain()

    def retire_form_controls(self, layout: QtWidgets.QFormLayout) -> None:
        """Retire every widget currently owned by a dynamic form layout."""
        while layout.count():
            item = layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                self._retire_widget(widget)

    def _queue_retired_drain(self) -> None:
        if self._retired_drain_pending or self._closing:
            return
        self._retired_drain_pending = True
        erlab.interactive.utils.single_shot(
            self,
            _RETIRED_EDITOR_DRAIN_DELAY_MS,
            self._drain_retired_widgets,
        )

    def _drain_retired_widgets(self) -> None:
        self._retired_drain_pending = False
        if not erlab.interactive.utils.qt_is_valid(self) or self._closing:
            return
        if (
            not self._retired_widgets
            or self.rebuild_must_wait()
            or self._retired_widget_has_focus()
        ):
            if self._retired_widgets:
                self._queue_retired_drain()
            return

        retired_widgets = self._retired_widgets
        self._retired_widgets = []
        for widget in retired_widgets:
            if not erlab.interactive.utils.qt_is_valid(widget):
                continue
            self._remove_posted_events_recursive(widget)
            widget.setParent(None)
            widget.deleteLater()

    def _retired_widget_has_focus(self) -> bool:
        focus_widget = QtWidgets.QApplication.focusWidget()
        return (
            focus_widget is not None
            and erlab.interactive.utils.qt_is_valid(focus_widget)
            and any(
                widget is focus_widget or widget.isAncestorOf(focus_widget)
                for widget in self._retired_widgets
                if erlab.interactive.utils.qt_is_valid(widget)
            )
        )

    def track_controls(self, widget: QtWidgets.QWidget) -> None:
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

    def rebuild_must_wait(self) -> bool:
        """Return whether a live popup or interaction makes rebuilding unsafe."""
        if self._combo_popup_guard_tokens or self._tracked_combo_popup_is_visible():
            return True
        popup = QtWidgets.QApplication.activePopupWidget()
        return popup is not None and erlab.interactive.utils.qt_is_valid(popup)

    def contains_widget(self, widget: QtWidgets.QWidget) -> bool:
        """Return whether a widget belongs to this operation editor."""
        return self is widget or self.isAncestorOf(widget)

    def current_operation(self) -> tuple[int, FigureOperationState] | None:
        operation_id = self.binding.current_operation_id()
        if operation_id is None:
            return None
        index = self.context.operation_index(operation_id)
        if index is None:
            return None
        operation = self.context.operation_by_id(operation_id)
        return None if operation is None else (index, operation)

    def editable_operations(self) -> tuple[tuple[int, FigureOperationState], ...]:
        editable: list[tuple[int, FigureOperationState]] = []
        for operation_id in self.binding.editable_operation_ids():
            index = self.context.operation_index(operation_id)
            operation = self.context.operation_by_id(operation_id)
            if index is not None and operation is not None:
                editable.append((index, operation))
        return tuple(editable)

    def batch_value(
        self,
        operation: FigureOperationState,
        getter: Callable[[FigureOperationState], typing.Any],
    ) -> typing.Any:
        editable = self.editable_operations()
        if len(editable) <= 1:
            return getter(operation)
        values = [getter(target) for _index, target in editable]
        first = values[0]
        if all(value == first for value in values[1:]):
            return first
        return MIXED_VALUE

    def batch_is_mixed(
        self,
        operation: FigureOperationState,
        getter: Callable[[FigureOperationState], typing.Any],
    ) -> bool:
        return self.batch_value(operation, getter) is MIXED_VALUE

    def batch_text(
        self,
        operation: FigureOperationState,
        getter: Callable[[FigureOperationState], typing.Any],
        formatter: Callable[[typing.Any], str],
    ) -> tuple[str, bool]:
        value = self.batch_value(operation, getter)
        if value is MIXED_VALUE:
            return "", True
        return formatter(value), False

    def batch_combo_text(
        self,
        operation: FigureOperationState,
        getter: Callable[[FigureOperationState], typing.Any],
        formatter: Callable[[typing.Any], str] = str,
    ) -> str | None:
        value = self.batch_value(operation, getter)
        if value is MIXED_VALUE:
            return None
        return formatter(value)

    def batch_options_match(
        self,
        operation: FigureOperationState,
        options_getter: Callable[[FigureOperationState], Sequence[typing.Any]],
    ) -> bool:
        editable = self.editable_operations()
        if len(editable) <= 1:
            return True
        expected = tuple(options_getter(operation))
        return all(
            tuple(options_getter(target)) == expected for _index, target in editable
        )

    @staticmethod
    def line_edit_batch_unchanged(edit: QtWidgets.QLineEdit) -> bool:
        return LineEditControlAdapter(edit).unchanged_mixed()

    @staticmethod
    def apply_mixed_line_edit(edit: QtWidgets.QLineEdit, mixed: bool) -> None:
        LineEditControlAdapter(edit).set_mixed(mixed)

    @staticmethod
    def apply_mixed_plain_text_edit(
        edit: QtWidgets.QPlainTextEdit, mixed: bool
    ) -> None:
        PlainTextControlAdapter(edit).set_mixed(mixed)

    @staticmethod
    def set_combo_mixed_placeholder(combo: QtWidgets.QComboBox) -> None:
        ComboBoxControlAdapter(combo).set_mixed(True)

    @staticmethod
    def mixed_combo_text(text: str) -> bool:
        return text == MIXED_VALUES_TEXT

    def request_transform(
        self,
        updater: Callable[[int, FigureOperationState], FigureOperationState],
        *,
        render: bool = True,
        defer_render: bool = False,
        rebuild_editor: bool = False,
        defer_editor_rebuild: bool = False,
        sync_axes: bool = True,
    ) -> None:
        self.request_transform_by_ids(
            self.binding.editable_operation_ids(),
            updater,
            render=render,
            defer_render=defer_render,
            rebuild_editor=rebuild_editor,
            defer_editor_rebuild=defer_editor_rebuild,
            sync_axes=sync_axes,
        )

    def request_transform_by_ids(
        self,
        operation_ids: Iterable[str],
        updater: Callable[[int, FigureOperationState], FigureOperationState],
        *,
        render: bool = True,
        defer_render: bool = False,
        rebuild_editor: bool = False,
        defer_editor_rebuild: bool = False,
        sync_axes: bool = True,
    ) -> None:
        if (
            self._closing
            or not erlab.interactive.utils.qt_is_valid(self)
            or not self.binding.updates_allowed()
        ):
            return
        stable_ids = tuple(dict.fromkeys(operation_ids))
        if not stable_ids:
            return
        self.edit_requested.emit(
            OperationEditRequest(
                operation_ids=stable_ids,
                updater=updater,
                render=render,
                defer_render=defer_render,
                rebuild_editor=rebuild_editor,
                defer_editor_rebuild=defer_editor_rebuild,
                sync_axes=sync_axes,
            )
        )

    def request_update(self, **updates: typing.Any) -> None:
        self.request_transform(
            lambda _index, operation: operation.model_copy(update=updates)
        )

    def request_update_rebuild(self, **updates: typing.Any) -> None:
        self.request_transform(
            lambda _index, operation: operation.model_copy(update=updates),
            rebuild_editor=True,
            defer_editor_rebuild=True,
        )

    def request_recipe_transform(
        self,
        updater: Callable[
            [Sequence[FigureOperationState], tuple[str, ...]],
            Sequence[FigureOperationState],
        ],
        *,
        render: bool = True,
        defer_render: bool = False,
        rebuild_editor: bool = False,
        defer_editor_rebuild: bool = False,
        sync_axes: bool = True,
    ) -> None:
        """Request an atomic edit that may insert or remove recipe operations."""
        if (
            self._closing
            or not erlab.interactive.utils.qt_is_valid(self)
            or not self.binding.updates_allowed()
        ):
            return
        operation_ids = tuple(dict.fromkeys(self.binding.editable_operation_ids()))
        if not operation_ids:
            return
        self.edit_requested.emit(
            OperationRecipeEditRequest(
                operation_ids=operation_ids,
                updater=updater,
                render=render,
                defer_render=defer_render,
                rebuild_editor=rebuild_editor,
                defer_editor_rebuild=defer_editor_rebuild,
                sync_axes=sync_axes,
            )
        )

    def selected_axes_state(self) -> FigureAxesSelectionState:
        return self.binding.selected_axes_state()

    def source_display_name(self, name: str) -> str:
        return self.binding.source_display_name(name)

    def source_tooltip(self, name: str) -> str:
        return self.binding.source_tooltip(name)

    def first_live_axis(self, selection: FigureAxesSelectionState) -> Axes | None:
        return self.binding.first_live_axis(selection)

    def subplot_parameter_default(self, key: str) -> float:
        return self.binding.subplot_parameter_default(key)

    def rendered_value(
        self,
        operation: FigureOperationState,
        reader: Callable[[Sequence[Axes]], typing.Any],
    ) -> typing.Any:
        return self.binding.rendered_value(operation, reader)

    def styled_rcparams_value(self, key: str) -> typing.Any:
        """Return an rcParam under the composition root's effective options."""
        return self.binding.styled_rcparams_value(key)

    def new_form_page(
        self, object_name: str
    ) -> tuple[QtWidgets.QWidget, QtWidgets.QFormLayout]:
        page = self.create_page(object_name, transient=True)
        layout = QtWidgets.QFormLayout(page)
        return page, layout

    def mark_control(self, widget: QtWidgets.QWidget) -> None:
        widget.setProperty("figure_composer_editor_generation", self._generation)
        self.track_controls(widget)

    def control_signal_allowed(self, widget: QtWidgets.QWidget) -> bool:
        return (
            not self._closing
            and erlab.interactive.utils.qt_is_valid(self, widget)
            and widget.property("figure_composer_editor_generation") == self._generation
        )

    def connect_signal(
        self,
        widget: QtWidgets.QWidget,
        signal: typing.Any,
        callback: Callable[..., None],
    ) -> None:
        """Connect a control through generation, lifetime, and input guards."""
        self.mark_control(widget)
        widget_ref = weakref.ref(widget)

        def guarded_callback(*args: typing.Any) -> None:
            guarded_widget = widget_ref()
            if guarded_widget is None or not self.control_signal_allowed(
                guarded_widget
            ):
                return
            operation_ids = self._operation_ids_for_error()
            input_error_key = self._input_error_key(guarded_widget)
            previous_widget = self._active_signal_widget
            self._active_signal_widget = guarded_widget
            try:
                callback(*args)
            except FigureComposerInputError as exc:
                self._record_input_error_for_key(input_error_key, operation_ids, exc)
            else:
                self._clear_input_error_for_key(input_error_key, operation_ids)
            finally:
                self._active_signal_widget = previous_widget

        signal.connect(guarded_callback)

    def connect_line_edit_finished(
        self,
        edit: QtWidgets.QLineEdit,
        callback: Callable[[str], None],
    ) -> None:
        LineEditControlAdapter(edit).connect_commit(self.connect_signal, callback)

    def connect_plain_text_changed(
        self,
        edit: QtWidgets.QPlainTextEdit,
        callback: Callable[[str], None],
    ) -> None:
        PlainTextControlAdapter(edit).connect_commit(self.connect_signal, callback)

    def connect_value_signal(
        self,
        widget: QtWidgets.QWidget,
        signal: typing.Any,
        value_getter: Callable[..., typing.Any],
        callback: Callable[[typing.Any], None],
        *,
        unchanged_mixed: Callable[[], bool] | None = None,
    ) -> None:
        SignalValueControlAdapter(
            widget,
            signal,
            value_getter,
            unchanged_mixed=unchanged_mixed,
        ).connect_commit(self.connect_signal, callback)

    def line_edit(
        self,
        text: str = "",
        *,
        parent: QtWidgets.QWidget | None = None,
    ) -> QtWidgets.QLineEdit:
        edit = QtWidgets.QLineEdit(parent or self._current_page() or self)
        self.mark_control(edit)
        edit.setText(text)
        return edit

    def mixed_value_widget(
        self,
        widget: QtWidgets.QWidget,
        *,
        mixed: bool,
        parent: QtWidgets.QWidget | None = None,
    ) -> QtWidgets.QWidget:
        if not mixed:
            return widget
        container = QtWidgets.QWidget(parent or widget.parentWidget())
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        marker = QtWidgets.QLabel(MIXED_VALUES_TEXT, container)
        marker.setObjectName("figureComposerMixedValueMarker")
        marker.setEnabled(False)
        layout.addWidget(widget, 1)
        layout.addWidget(marker)
        return container

    def combo(
        self,
        values: Sequence[str],
        current: str | None,
        changed: Callable[[str], None],
        *,
        parent: QtWidgets.QWidget | None = None,
        mixed: bool = False,
        enabled: bool = True,
    ) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox(parent or self._current_page() or self)
        self.mark_control(combo)
        combo.addItems(list(values))
        adapter = ComboBoxControlAdapter(combo)
        if mixed:
            adapter.set_mixed(True)
        elif current is not None:
            self._set_combo_value(combo, current)
        combo.setEnabled(enabled)
        adapter.connect_commit(self.connect_signal, changed)
        return combo

    def source_combo(
        self,
        values: Sequence[str],
        current: str | None,
        changed: Callable[[str | None], None],
        *,
        parent: QtWidgets.QWidget | None = None,
        mixed: bool = False,
        enabled: bool = True,
    ) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox(parent or self._current_page() or self)
        self.mark_control(combo)
        adapter = ComboBoxDataControlAdapter(combo)
        if mixed:
            combo.addItem(MIXED_VALUES_TEXT, MIXED_VALUE)
        for value in values:
            combo.addItem(self.source_display_name(value), value)
            combo.setItemData(
                combo.count() - 1,
                self.source_tooltip(value),
                QtCore.Qt.ItemDataRole.ToolTipRole,
            )
        if current is not None and current not in values and not mixed:
            combo.addItem(self.source_display_name(current), current)
            combo.setItemData(
                combo.count() - 1,
                self.source_tooltip(current),
                QtCore.Qt.ItemDataRole.ToolTipRole,
            )
        if mixed:
            item = typing.cast("typing.Any", combo.model()).item(0)
            if item is not None:
                item.setEnabled(False)
            combo.setCurrentIndex(0)
        elif current is not None:
            for index in range(combo.count()):
                if combo.itemData(index) == current:
                    combo.setCurrentIndex(index)
                    break
        combo.setEnabled(enabled)
        adapter.connect_commit(
            self.connect_signal,
            lambda value: changed(typing.cast("str | None", value)),
        )
        return combo

    def optional_name_combo(
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
        combo = QtWidgets.QComboBox(parent or self._current_page() or self)
        self.mark_control(combo)
        adapter = ComboBoxDataControlAdapter(combo)
        if mixed:
            combo.addItem(MIXED_VALUES_TEXT, MIXED_VALUE)
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
        adapter.connect_commit(
            self.connect_signal,
            lambda value: changed(typing.cast("str | None", value)),
        )
        return combo

    def check_box(
        self,
        checked: bool,
        changed: Callable[[bool], None],
        *,
        parent: QtWidgets.QWidget | None = None,
        mixed: bool = False,
    ) -> QtWidgets.QCheckBox:
        check = QtWidgets.QCheckBox(parent or self._current_page() or self)
        self.mark_control(check)
        adapter = CheckBoxControlAdapter(check)
        if mixed:
            adapter.set_mixed(True)
        else:
            check.setChecked(checked)
        adapter.connect_commit(self.connect_signal, changed)
        return check

    @staticmethod
    def _set_combo_value(combo: QtWidgets.QComboBox, value: str) -> None:
        index = combo.findText(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    @staticmethod
    def _wrapped_tooltip(tooltip: str) -> str:
        if "\n" in tooltip:
            return tooltip
        return "\n".join(textwrap.wrap(tooltip, width=58, break_long_words=False))

    @staticmethod
    def add_form_section(
        layout: QtWidgets.QFormLayout,
        title: str,
        *,
        object_name: str | None = None,
    ) -> QtWidgets.QWidget:
        section = QtWidgets.QWidget(layout.parentWidget())
        if object_name:
            section.setObjectName(object_name)
        section.setProperty("figureComposerSectionHeader", True)
        section_layout = QtWidgets.QHBoxLayout(section)
        top_margin = 8 if layout.rowCount() else 0
        section_layout.setContentsMargins(0, top_margin, 0, 2)
        section_layout.setSpacing(6)

        label = QtWidgets.QLabel(title, section)
        label.setProperty("figureComposerSectionHeaderLabel", True)
        label_font = label.font()
        label_font.setBold(True)
        label.setFont(label_font)
        line = _Separator(parent=section)
        if object_name:
            line.setObjectName(f"{object_name}Line")

        section_layout.addWidget(label)
        section_layout.addWidget(line, 1)
        layout.addRow(section)
        return section

    @staticmethod
    def add_form_row(
        layout: QtWidgets.QFormLayout,
        label: str,
        widget: QtWidgets.QWidget,
        tooltip: str,
    ) -> None:
        tooltip = FigureOperationEditor._wrapped_tooltip(tooltip)
        widget.setToolTip(tooltip)
        layout.addRow(label, widget)
        label_widget = layout.labelForField(widget)
        if label_widget is not None:
            label_widget.setToolTip(tooltip)

    @staticmethod
    def add_compound_form_row(
        layout: QtWidgets.QFormLayout,
        label: str,
        controls: Sequence[tuple[str, QtWidgets.QWidget, str]],
        tooltip: str,
    ) -> QtWidgets.QWidget:
        row_widget = QtWidgets.QWidget(layout.parentWidget())
        row_layout = QtWidgets.QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        for control_label, widget, control_tooltip in controls:
            control_tooltip = FigureOperationEditor._wrapped_tooltip(control_tooltip)
            label_widget = QtWidgets.QLabel(control_label, row_widget)
            label_widget.setBuddy(widget)
            label_widget.setToolTip(control_tooltip)
            widget.setToolTip(control_tooltip)
            row_layout.addWidget(label_widget)
            row_layout.addWidget(widget, 1)
        row_tooltip = FigureOperationEditor._wrapped_tooltip(tooltip)
        row_widget.setToolTip(row_tooltip)
        layout.addRow(label, row_widget)
        label_widget = layout.labelForField(row_widget)
        if label_widget is not None:
            label_widget.setToolTip(row_tooltip)
        return row_widget

    @staticmethod
    def _input_error_key(widget: QtWidgets.QWidget) -> str:
        if not erlab.interactive.utils.qt_is_valid(widget):
            return f"anonymous:{id(widget)}"
        object_name = widget.objectName()
        if object_name:
            return object_name
        return f"anonymous:{id(widget)}"

    def _operation_ids_for_error(self) -> tuple[str, ...]:
        editable_ids = self.binding.editable_operation_ids()
        if editable_ids:
            return editable_ids
        operation_id = self.binding.current_operation_id()
        return () if operation_id is None else (operation_id,)

    def set_input_errors(self, errors: Mapping[str, Mapping[str, str]]) -> None:
        input_errors = {
            operation_id: dict(operation_errors)
            for operation_id, operation_errors in errors.items()
            if operation_errors
        }
        if input_errors == self._input_errors:
            return
        self._input_errors = input_errors
        self.validation_changed.emit()

    def has_input_error(self, operation: FigureOperationState) -> bool:
        return operation.operation_id in self._input_errors

    def input_error_text(self, operation: FigureOperationState) -> str | None:
        operation_errors = self._input_errors.get(operation.operation_id)
        if not operation_errors:
            return None
        messages = tuple(dict.fromkeys(operation_errors.values()))
        if len(messages) == 1:
            return messages[0]
        return f"{len(messages)} invalid inputs: " + "; ".join(messages)

    def record_input_error(
        self, widget: QtWidgets.QWidget, error: FigureComposerInputError
    ) -> None:
        self._record_input_error_for_key(
            self._input_error_key(widget), self._operation_ids_for_error(), error
        )

    def _record_input_error_for_key(
        self,
        key: str,
        operation_ids: Sequence[str],
        error: FigureComposerInputError,
    ) -> None:
        if not operation_ids:
            return
        errors = {
            operation_id: dict(operation_errors)
            for operation_id, operation_errors in self._input_errors.items()
        }
        for operation_id in operation_ids:
            errors.setdefault(operation_id, {})[key] = str(error)
        self.set_input_errors(errors)

    def clear_input_error(self, widget: QtWidgets.QWidget) -> None:
        self._clear_input_error_for_key(
            self._input_error_key(widget), self._operation_ids_for_error()
        )

    def _clear_input_error_for_key(
        self, key: str, operation_ids: Sequence[str]
    ) -> None:
        if not self._input_errors or not operation_ids:
            return
        errors = {
            operation_id: dict(operation_errors)
            for operation_id, operation_errors in self._input_errors.items()
        }
        changed = False
        for operation_id in operation_ids:
            operation_errors = errors.get(operation_id)
            if operation_errors is None or key not in operation_errors:
                continue
            del operation_errors[key]
            changed = True
        if changed:
            self.set_input_errors(errors)

    def clear_input_errors(self, operation_ids: Iterable[str]) -> None:
        operation_id_set = set(operation_ids)
        if not operation_id_set or not self._input_errors:
            return
        self.set_input_errors(
            {
                operation_id: errors
                for operation_id, errors in self._input_errors.items()
                if operation_id not in operation_id_set
            }
        )

    @staticmethod
    def _accepts_tab_focus(widget: QtWidgets.QWidget) -> bool:
        return bool(widget.focusPolicy() & QtCore.Qt.FocusPolicy.TabFocus)

    def _first_tab_stop(self) -> QtWidgets.QWidget | None:
        current_page = self._current_page()
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

    def _refresh_tab_order(self) -> None:
        buttons = list(self._section_buttons.values())
        if not buttons:
            return
        tab_chain = [*self._tab_order_prefix, *buttons]
        next_widget = self._first_tab_stop()
        if next_widget is not None:
            tab_chain.append(next_widget)
        for index, widget in enumerate(tab_chain[:-1]):
            QtWidgets.QWidget.setTabOrder(widget, tab_chain[index + 1])

    def _queue_tab_order_refresh(self) -> None:
        if self._tab_order_update_pending or self._closing:
            return
        self._tab_order_update_pending = True
        erlab.interactive.utils.single_shot(self, 0, self._run_tab_order_refresh)

    def _run_tab_order_refresh(self) -> None:
        if not erlab.interactive.utils.qt_is_valid(self):
            return
        self._tab_order_update_pending = False
        self._refresh_tab_order()

    def eventFilter(
        self, watched: QtCore.QObject | None, event: QtCore.QEvent | None
    ) -> bool:
        if event is not None and watched is not None:
            event_type = event.type()
            if isinstance(watched, QtWidgets.QComboBox):
                if event_type in {
                    QtCore.QEvent.Type.MouseButtonPress,
                    QtCore.QEvent.Type.KeyPress,
                    QtCore.QEvent.Type.Wheel,
                }:
                    self._begin_transient_combo_popup_guard()
            elif isinstance(watched, QtWidgets.QAbstractItemView) and (
                self._combo_for_popup_view(watched) is not None
            ):
                if event_type in {
                    QtCore.QEvent.Type.Show,
                    QtCore.QEvent.Type.ShowToParent,
                }:
                    self._begin_combo_popup_view_guard(watched)
                elif event_type in {
                    QtCore.QEvent.Type.Hide,
                    QtCore.QEvent.Type.HideToParent,
                    QtCore.QEvent.Type.Close,
                }:
                    self._end_combo_popup_view_guard(watched)
        return super().eventFilter(watched, event)

    def release(self) -> None:
        """Detach tracked editor event filters before tool teardown."""
        self.flush_pending_commits()
        self._closing = True
        self._generation += 1
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
