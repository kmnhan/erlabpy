"""Figure Composer canvas, navigation toolbar, and display window."""

from __future__ import annotations

import contextlib
import functools
import math
import typing

# Matplotlib's Qt backend should see the qtpy-selected binding first.
# isort: off
from qtpy import QtCore, QtGui, QtWidgets

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
# isort: on

import erlab.interactive.utils
from erlab.interactive._figurecomposer._defaults import (
    _apply_figure_dpi,
    _figure_draw_context,
    _figure_style_context,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from matplotlib.backend_bases import Event

    from erlab.interactive._figurecomposer._model._state import FigureSubplotsState

_MANAGER_WORKSPACE_SAVE_SHORTCUT_OBJECT_NAME = "managerWorkspaceSaveShortcut"


class _StyledFigureCanvas(FigureCanvas):
    def draw(self, *args, **kwargs):
        with _figure_draw_context():
            return super().draw(*args, **kwargs)

    def print_figure(self, *args, **kwargs):
        with _figure_draw_context():
            return super().print_figure(*args, **kwargs)


_TOOLBAR_ICON_NAMES = {
    "figure_composer": "mdi6.application-edit-outline",
    "back": "mdi6.undo",
    "forward": "mdi6.redo",
    "move": "mdi6.arrow-all",
    "zoom_to_rect": "mdi6.magnify-plus",
    "subplots": "mdi6.view-grid-outline",
    "qt4_editor_options": "mdi6.tune",
    "copy_figure_to_clipboard": "mdi6.content-copy",
    "filesave": "mdi6.export",
}

_SHOW_COMPOSER_TOOLITEM = (
    "Composer",
    "Show the corresponding Figure Composer window",
    "figure_composer",
    "show_composer",
)
_TOOLBAR_ICON_REFRESH_EVENTS = {
    QtCore.QEvent.Type.ActivationChange,
    QtCore.QEvent.Type.ApplicationPaletteChange,
    QtCore.QEvent.Type.EnabledChange,
    QtCore.QEvent.Type.PaletteChange,
    QtCore.QEvent.Type.StyleChange,
}


def _noop_toolbar_callback() -> None:
    return


def _noop_navigation_callback(_changes: Mapping[object, tuple[bool, bool]]) -> None:
    return


def _noop_colorbar_callback(_changes: Mapping[object, tuple[float, float]]) -> None:
    return


def _false_toolbar_state() -> bool:
    return False


def _false_mime_state(_mime: QtCore.QMimeData) -> bool:
    return False


def _axis_limit_pair(axis: object, getter_name: str) -> tuple[float, float] | None:
    getter = getattr(axis, getter_name, None)
    if getter is None:
        return None
    try:
        first, second = getter()
    except (TypeError, ValueError):
        return None
    return float(first), float(second)


def _axis_view(axis: object) -> tuple[tuple[float, float], tuple[float, float]] | None:
    xlim = _axis_limit_pair(axis, "get_xlim")
    ylim = _axis_limit_pair(axis, "get_ylim")
    if xlim is None or ylim is None:
        return None
    return xlim, ylim


def _same_axis_limits(first: tuple[float, float], second: tuple[float, float]) -> bool:
    return math.isclose(first[0], second[0]) and math.isclose(first[1], second[1])


def _is_colorbar_axis(axis: object) -> bool:
    return hasattr(axis, "_colorbar")


def _colorbar_mappable(axis: object) -> object | None:
    colorbar = getattr(axis, "_colorbar", None)
    if colorbar is None:
        return None
    return getattr(colorbar, "mappable", None)


def _mappable_clim(mappable: object) -> tuple[float, float] | None:
    get_clim = getattr(mappable, "get_clim", None)
    if get_clim is None:
        return None
    try:
        first, second = get_clim()
    except (TypeError, ValueError):
        return None
    if first is None or second is None:
        return None
    try:
        return float(first), float(second)
    except (TypeError, ValueError):
        return None


class _FigureComposerNavigationToolbar(NavigationToolbar):
    """Navigation toolbar that routes composer actions through recipe edits."""

    toolitems = [  # noqa: RUF012 - Matplotlib expects toolbar items on the class.
        _SHOW_COMPOSER_TOOLITEM if item[3] == "home" else item
        for item in NavigationToolbar.toolitems
    ]

    def __init__(
        self,
        canvas: FigureCanvas,
        parent: QtWidgets.QWidget,
        *,
        export_callback: Callable[[], None],
        subplot_adjust_callback: Callable[[], None],
        axes_customize_callback: Callable[[], None],
        show_composer_callback: Callable[[], None] = _noop_toolbar_callback,
        navigation_callback: Callable[
            [Mapping[object, tuple[bool, bool]]], None
        ] = _noop_navigation_callback,
        colorbar_callback: Callable[
            [Mapping[object, tuple[float, float]]], None
        ] = _noop_colorbar_callback,
        undo_callback: Callable[[], None] = _noop_toolbar_callback,
        redo_callback: Callable[[], None] = _noop_toolbar_callback,
        undoable_callback: Callable[[], bool] = _false_toolbar_state,
        redoable_callback: Callable[[], bool] = _false_toolbar_state,
    ) -> None:
        self._export_callback = export_callback
        self._subplot_adjust_callback = subplot_adjust_callback
        self._axes_customize_callback = axes_customize_callback
        self._show_composer_callback = show_composer_callback
        self._navigation_callback = navigation_callback
        self._colorbar_callback = colorbar_callback
        self._undo_callback = undo_callback
        self._redo_callback = redo_callback
        self._undoable_callback = undoable_callback
        self._redoable_callback = redoable_callback
        self._navigation_press_views: dict[
            object, tuple[tuple[float, float], tuple[float, float]]
        ] = {}
        self._colorbar_press_clims: dict[object, tuple[float, float]] = {}
        super().__init__(canvas, parent)
        self.setObjectName("figureComposerNavigationToolbar")
        for action_id, action in self._actions.items():
            action.setObjectName(f"figureComposerToolbar_{action_id}")
        tooltips = {
            "show_composer": "Show the corresponding Figure Composer window.",
            "back": "Undo the last Figure Composer recipe change.",
            "forward": "Redo the last undone Figure Composer recipe change.",
            "save_figure": "Export the composer figure using recipe export settings.",
            "copy_figure_to_clipboard": "Copy the current figure image.",
            "configure_subplots": (
                "Add or edit recipe steps for subplot spacing and layout engine."
            ),
            "edit_parameters": "Add or edit recipe steps for selected axes.",
        }
        self._add_copy_action()
        for action_id, tooltip in tooltips.items():
            if action := self._actions.get(action_id):
                action.setToolTip(tooltip)
        if action := self._actions.get("back"):
            action.setText("Undo")
            action.setShortcut(QtGui.QKeySequence.StandardKey.Undo)
            action.setShortcutContext(QtCore.Qt.ShortcutContext.WindowShortcut)
        if action := self._actions.get("forward"):
            action.setText("Redo")
            action.setShortcut(QtGui.QKeySequence.StandardKey.Redo)
            action.setShortcutContext(QtCore.Qt.ShortcutContext.WindowShortcut)
        self.set_history_buttons()

    def _add_copy_action(self) -> None:
        action_id = "copy_figure_to_clipboard"
        action = QtGui.QAction(self._icon(action_id), "Copy", self)
        action.setObjectName(f"figureComposerToolbar_{action_id}")
        action.triggered.connect(self.copy_figure_to_clipboard)
        if save_action := self._actions.get("save_figure"):
            self.insertAction(save_action, action)
        else:
            self.addAction(action)
        self._actions[action_id] = action

    def _icon(self, name: str) -> QtGui.QIcon:
        icon_key = name.removesuffix(".png").removesuffix("_large")
        icon_name = _TOOLBAR_ICON_NAMES.get(icon_key)
        if icon_name is None:
            return super()._icon(name)
        color = self.palette().color(QtGui.QPalette.ColorRole.ButtonText)
        return erlab.interactive.utils.qtawesome.icon(icon_name, color=color)

    def _refresh_icons(self) -> None:
        erlab.interactive.utils.qtawesome.reset_cache()
        for callback_name, action in self._actions.items():
            icon_name = {
                "zoom": "zoom_to_rect",
                "pan": "move",
                "show_composer": "figure_composer",
                "configure_subplots": "subplots",
                "edit_parameters": "qt4_editor_options",
                "save_figure": "filesave",
            }.get(callback_name, callback_name)
            action.setIcon(self._icon(icon_name))

    def configure_subplots(self, *args: typing.Any) -> typing.Any:
        self._subplot_adjust_callback()

    def edit_parameters(self) -> None:
        self._axes_customize_callback()

    def show_composer(self) -> None:
        self._show_composer_callback()

    def save_figure(self, *args: object) -> None:
        self._export_callback()

    def copy_figure_to_clipboard(self, *args: object) -> None:
        if not erlab.interactive.utils.qt_is_valid(self, self.canvas):
            return
        canvas = typing.cast("FigureCanvas", self.canvas)
        pixmap = canvas.grab()
        if pixmap.isNull():
            return
        application = typing.cast(
            "QtWidgets.QApplication | None", QtWidgets.QApplication.instance()
        )
        if (
            application is not None
            and (clipboard := application.clipboard()) is not None
        ):
            clipboard.setPixmap(pixmap)

    def set_history_buttons(self) -> None:
        if "back" in self._actions:
            self._actions["back"].setEnabled(self._undoable_callback())
        if "forward" in self._actions:
            self._actions["forward"].setEnabled(self._redoable_callback())

    def back(self, *args: object) -> None:
        self._undo_callback()
        self.set_history_buttons()

    def forward(self, *args: object) -> None:
        self._redo_callback()
        self.set_history_buttons()

    def _capture_navigation_views(
        self, axes: Iterable[object]
    ) -> dict[object, tuple[tuple[float, float], tuple[float, float]]]:
        views: dict[object, tuple[tuple[float, float], tuple[float, float]]] = {}
        for axis in axes:
            if _is_colorbar_axis(axis):
                continue
            view = _axis_view(axis)
            if view is not None:
                views[axis] = view
        return views

    def _capture_colorbar_clims(
        self, axes: Iterable[object]
    ) -> dict[object, tuple[float, float]]:
        clims: dict[object, tuple[float, float]] = {}
        for axis in axes:
            mappable = _colorbar_mappable(axis)
            if mappable is None:
                continue
            clim = _mappable_clim(mappable)
            if clim is not None:
                clims[mappable] = clim
        return clims

    def _commit_navigation_views(
        self,
        before: Mapping[object, tuple[tuple[float, float], tuple[float, float]]],
    ) -> None:
        changes: dict[object, tuple[bool, bool]] = {}
        for axis, previous in before.items():
            current = _axis_view(axis)
            if current is None:
                continue
            x_changed = not _same_axis_limits(previous[0], current[0])
            y_changed = not _same_axis_limits(previous[1], current[1])
            if x_changed or y_changed:
                changes[axis] = (x_changed, y_changed)
        if changes:
            self._navigation_callback(changes)
        self.set_history_buttons()

    def _commit_colorbar_clims(
        self,
        before: Mapping[object, tuple[float, float]],
    ) -> None:
        if not before:
            return
        changes: dict[object, tuple[float, float]] = {}
        for mappable, previous in before.items():
            current = _mappable_clim(mappable)
            if current is not None and not _same_axis_limits(previous, current):
                changes[mappable] = current
        if changes:
            self._colorbar_callback(changes)
        self.set_history_buttons()

    def press_pan(self, event: Event) -> None:
        super().press_pan(event)
        pan_info = getattr(self, "_pan_info", None)
        self._navigation_press_views = (
            {} if pan_info is None else self._capture_navigation_views(pan_info.axes)
        )
        self._colorbar_press_clims = (
            {} if pan_info is None else self._capture_colorbar_clims(pan_info.axes)
        )

    def release_pan(self, event: Event) -> None:
        before = dict(self._navigation_press_views)
        colorbar_before = dict(self._colorbar_press_clims)
        self._navigation_press_views = {}
        self._colorbar_press_clims = {}
        super().release_pan(event)
        self._commit_navigation_views(before)
        self._commit_colorbar_clims(colorbar_before)

    def press_zoom(self, event: Event) -> None:
        super().press_zoom(event)
        zoom_info = getattr(self, "_zoom_info", None)
        self._navigation_press_views = (
            {} if zoom_info is None else self._capture_navigation_views(zoom_info.axes)
        )
        self._colorbar_press_clims = (
            {} if zoom_info is None else self._capture_colorbar_clims(zoom_info.axes)
        )

    def release_zoom(self, event: Event) -> None:
        before = dict(self._navigation_press_views)
        colorbar_before = dict(self._colorbar_press_clims)
        self._navigation_press_views = {}
        self._colorbar_press_clims = {}
        super().release_zoom(event)
        self._commit_navigation_views(before)
        self._commit_colorbar_clims(colorbar_before)

    def changeEvent(self, event: QtCore.QEvent | None) -> None:
        if event is not None and event.type() in _TOOLBAR_ICON_REFRESH_EVENTS:
            self._refresh_icons()
        super().changeEvent(event)


class _FigureComposerDisplayWindow(QtWidgets.QMainWindow):
    """Top-level Matplotlib display owned by a figure composer."""

    sigCanvasSizeChanged = QtCore.Signal(float, float)

    def __init__(
        self,
        setup: FigureSubplotsState,
        *,
        export_callback: Callable[[], None] = _noop_toolbar_callback,
        subplot_adjust_callback: Callable[[], None] = _noop_toolbar_callback,
        axes_customize_callback: Callable[[], None] = _noop_toolbar_callback,
        show_composer_callback: Callable[[], None] = _noop_toolbar_callback,
        navigation_callback: Callable[
            [Mapping[object, tuple[bool, bool]]], None
        ] = _noop_navigation_callback,
        colorbar_callback: Callable[
            [Mapping[object, tuple[float, float]]], None
        ] = _noop_colorbar_callback,
        undo_callback: Callable[[], None] = _noop_toolbar_callback,
        redo_callback: Callable[[], None] = _noop_toolbar_callback,
        undoable_callback: Callable[[], bool] = _false_toolbar_state,
        redoable_callback: Callable[[], bool] = _false_toolbar_state,
        source_drop_available_callback: Callable[
            [QtCore.QMimeData], bool
        ] = _false_mime_state,
        source_drop_callback: Callable[[QtCore.QMimeData], bool] = _false_mime_state,
    ) -> None:
        super().__init__(None)
        erlab.interactive.utils.patch_macos_matplotlib_qt_cursor()
        self._closing_from_owner = False
        self._suppress_resize_signal = False
        self._resize_signal_pending = False
        self._resize_signal_generation = 0
        self._source_drop_available_callback = source_drop_available_callback
        self._source_drop_callback = source_drop_callback
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, False)

        with _figure_style_context():
            self.figure = Figure(
                figsize=setup.figsize,
                dpi=setup.dpi,
                layout=typing.cast("typing.Any", setup.layout),
            )
        self.canvas = _StyledFigureCanvas(self.figure)
        self.toolbar = _FigureComposerNavigationToolbar(
            self.canvas,
            self,
            export_callback=export_callback,
            subplot_adjust_callback=subplot_adjust_callback,
            axes_customize_callback=axes_customize_callback,
            show_composer_callback=show_composer_callback,
            navigation_callback=navigation_callback,
            colorbar_callback=colorbar_callback,
            undo_callback=undo_callback,
            redo_callback=redo_callback,
            undoable_callback=undoable_callback,
            redoable_callback=redoable_callback,
        )

        root = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)
        self.setCentralWidget(root)
        self.setWindowTitle("Figure")
        for widget in (self, root, self.toolbar, self.canvas):
            widget.setAcceptDrops(True)
            widget.installEventFilter(self)
        self._close_shortcut = erlab.interactive.utils._install_close_shortcut(
            self, self.hide
        )

    def _event_filter_targets(self) -> tuple[QtCore.QObject, ...]:
        return tuple(
            target
            for target in (self, self.centralWidget(), self.toolbar, self.canvas)
            if target is not None
        )

    def _remove_event_filters(self) -> None:
        for target in self._event_filter_targets():
            if not erlab.interactive.utils.qt_is_valid(target):
                continue
            with contextlib.suppress(RuntimeError):
                target.removeEventFilter(self)

    def set_source_drop_callbacks(
        self,
        *,
        can_drop: Callable[[QtCore.QMimeData], bool] = _false_mime_state,
        drop: Callable[[QtCore.QMimeData], bool] = _false_mime_state,
    ) -> None:
        self._source_drop_available_callback = can_drop
        self._source_drop_callback = drop

    def _handle_source_drag_event(self, event: QtCore.QEvent | None) -> bool:
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
        if mime is None:
            return False
        if not self._source_drop_available_callback(mime):
            return False
        if event.type() == QtCore.QEvent.Type.Drop:
            accepted = self._source_drop_callback(mime)
            if not accepted:
                return False
        event.setDropAction(QtCore.Qt.DropAction.CopyAction)
        event.accept()
        return True

    def _cancel_resize_callbacks(self, *, suppress: bool) -> None:
        self._suppress_resize_signal = suppress
        self._resize_signal_pending = False
        self._resize_signal_generation += 1

    def eventFilter(
        self,
        watched: QtCore.QObject | None,
        event: QtCore.QEvent | None,
    ) -> bool:
        if self._handle_source_drag_event(event):
            return True
        if self._is_workspace_save_shortcut_event(event) and (
            shortcut := self._workspace_save_shortcut()
        ):
            shortcut.activated.emit()
            if event is not None:
                event.accept()
            return True
        if self._is_close_shortcut_event(event):
            self.hide()
            if event is not None:
                event.accept()
            return True
        return super().eventFilter(watched, event)

    def changeEvent(self, event: QtCore.QEvent | None) -> None:
        if (
            event is not None
            and event.type() in _TOOLBAR_ICON_REFRESH_EVENTS
            and erlab.interactive.utils.qt_is_valid(self.toolbar)
        ):
            self.toolbar._refresh_icons()
        super().changeEvent(event)

    def _workspace_save_shortcut(self) -> QtWidgets.QShortcut | None:
        for shortcut in self.findChildren(QtWidgets.QShortcut):
            if shortcut.objectName() == _MANAGER_WORKSPACE_SAVE_SHORTCUT_OBJECT_NAME:
                return shortcut
        return None

    @staticmethod
    def _is_workspace_save_shortcut_event(event: QtCore.QEvent | None) -> bool:
        return (
            event is not None
            and event.type() == QtCore.QEvent.Type.ShortcutOverride
            and isinstance(event, QtGui.QKeyEvent)
            and event.matches(QtGui.QKeySequence.StandardKey.Save)
        )

    @staticmethod
    def _is_close_shortcut_event(event: QtCore.QEvent | None) -> bool:
        if (
            event is None
            or event.type()
            not in {
                QtCore.QEvent.Type.ShortcutOverride,
                QtCore.QEvent.Type.KeyPress,
            }
            or not isinstance(event, QtGui.QKeyEvent)
        ):
            return False
        relevant_modifiers = (
            QtCore.Qt.KeyboardModifier.ControlModifier
            | QtCore.Qt.KeyboardModifier.ShiftModifier
            | QtCore.Qt.KeyboardModifier.AltModifier
            | QtCore.Qt.KeyboardModifier.MetaModifier
        )
        modifiers = event.modifiers() & relevant_modifiers
        return event.matches(QtGui.QKeySequence.StandardKey.Close) or (
            event.key() == QtCore.Qt.Key.Key_W
            and modifiers
            in {
                QtCore.Qt.KeyboardModifier.ControlModifier,
                QtCore.Qt.KeyboardModifier.MetaModifier,
            }
        )

    def resize_to_setup(self, setup: FigureSubplotsState) -> None:
        canvas_width = max(1, round(setup.figsize[0] * setup.dpi))
        canvas_height = max(1, round(setup.figsize[1] * setup.dpi))
        target_canvas_size = QtCore.QSize(canvas_width, canvas_height)
        self.figure.set_size_inches(setup.figsize, forward=False)
        _apply_figure_dpi(self.figure, setup.dpi)
        if (
            self.isVisible()
            and self.canvas.size().isValid()
            and not self.canvas.size().isEmpty()
        ):
            size_delta = self.size() - self.canvas.size()
            target_size = QtCore.QSize(
                target_canvas_size.width() + size_delta.width(),
                target_canvas_size.height() + size_delta.height(),
            )
        else:
            target_size = QtCore.QSize(
                target_canvas_size.width(),
                target_canvas_size.height() + self.toolbar.sizeHint().height(),
            )
        self._suppress_resize_signal = True
        self.resize(target_size)
        self._resize_signal_generation += 1
        generation = self._resize_signal_generation
        erlab.interactive.utils.single_shot(
            self,
            0,
            functools.partial(self._allow_resize_signal, generation),
        )

    @QtCore.Slot()
    def _allow_resize_signal(self, generation: int | None = None) -> None:
        if (
            generation is not None and generation != self._resize_signal_generation
        ) or self._closing_from_owner:
            return
        self._suppress_resize_signal = False

    def _ensure_recallable_geometry(self) -> None:
        frame = self.frameGeometry()
        if frame.isEmpty():
            return
        screen_geometries = tuple(
            geometry
            for screen in QtGui.QGuiApplication.screens()
            for geometry in (screen.availableGeometry(),)
            if not geometry.isEmpty()
        )
        if not screen_geometries or any(
            geometry.intersects(frame) for geometry in screen_geometries
        ):
            return
        target_screen = self.screen() or QtGui.QGuiApplication.primaryScreen()
        if target_screen is None:
            target_geometry = screen_geometries[0]
        else:
            target_geometry = target_screen.availableGeometry()
            if target_geometry.isEmpty():
                target_geometry = screen_geometries[0]
        frame.moveCenter(target_geometry.center())
        self.move(frame.topLeft())

    def show_for_setup(
        self, setup: FigureSubplotsState, title: str, *, activate: bool
    ) -> None:
        self.setWindowTitle(title)
        self.resize_to_setup(setup)
        self.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating, not activate
        )
        if self.isMinimized():
            self.showNormal()
        elif not self.isVisible():
            self.show()
        self._ensure_recallable_geometry()
        if activate:
            self.raise_()
            self.activateWindow()

    def close_from_owner(self) -> None:
        self._closing_from_owner = True
        self._cancel_resize_callbacks(suppress=True)
        self._remove_event_filters()
        self.close()
        self.deleteLater()

    def resizeEvent(self, event: QtGui.QResizeEvent | None) -> None:
        if event is not None:
            super().resizeEvent(event)
        if self._suppress_resize_signal or self._resize_signal_pending:
            return
        self._resize_signal_pending = True
        self._resize_signal_generation += 1
        generation = self._resize_signal_generation
        canvas = self.canvas
        erlab.interactive.utils.single_shot(
            self,
            0,
            functools.partial(self._emit_canvas_size_changed, generation, canvas),
            canvas,
        )

    @QtCore.Slot()
    def _emit_canvas_size_changed(
        self, generation: int | None = None, canvas: FigureCanvas | None = None
    ) -> None:
        if generation is not None and generation != self._resize_signal_generation:
            self._resize_signal_pending = False
            return
        self._resize_signal_pending = False
        if self._suppress_resize_signal or self._closing_from_owner:
            return
        if canvas is None:
            canvas = self.canvas
        if not erlab.interactive.utils.qt_is_valid(self, canvas):
            return
        canvas_size = canvas.size()
        if canvas_size.isEmpty():
            return
        dpi = float(typing.cast("typing.Any", canvas.figure)._original_dpi)
        if dpi <= 0.0:
            return
        self.sigCanvasSizeChanged.emit(
            canvas_size.width() / dpi,
            canvas_size.height() / dpi,
        )

    def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:
        if (
            self._closing_from_owner
            or erlab.interactive.utils._application_quit_requested()
        ):
            self._cancel_resize_callbacks(suppress=True)
            self._remove_event_filters()
            if event is not None:
                super().closeEvent(event)
            return
        if event is not None:
            event.ignore()
        self._cancel_resize_callbacks(suppress=False)
        self.hide()
