from __future__ import annotations

import contextlib
import dataclasses
import re
import time
import typing
import weakref

from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive._widgets import _CenteredIconToolButton

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    Signal = typing.Any


_INPUT_EVENTS: dict[QtCore.QEvent.Type, str] = {
    QtCore.QEvent.Type.MouseButtonPress: "mouse",
    QtCore.QEvent.Type.MouseButtonRelease: "mouse",
    QtCore.QEvent.Type.MouseButtonDblClick: "mouse",
    QtCore.QEvent.Type.MouseMove: "mouse",
    QtCore.QEvent.Type.Wheel: "wheel",
    QtCore.QEvent.Type.TouchBegin: "touch",
    QtCore.QEvent.Type.TouchUpdate: "touch",
    QtCore.QEvent.Type.TouchEnd: "touch",
    QtCore.QEvent.Type.TouchCancel: "touch",
    QtCore.QEvent.Type.DragEnter: "drag",
    QtCore.QEvent.Type.DragMove: "drag",
    QtCore.QEvent.Type.DragLeave: "drag",
    QtCore.QEvent.Type.Drop: "drop",
    QtCore.QEvent.Type.KeyPress: "key",
    QtCore.QEvent.Type.KeyRelease: "key",
    QtCore.QEvent.Type.ShortcutOverride: "shortcut",
    QtCore.QEvent.Type.Shortcut: "shortcut",
    QtCore.QEvent.Type.ContextMenu: "context_menu",
}
_REPOSITION_EVENTS = {
    QtCore.QEvent.Type.Show,
    QtCore.QEvent.Type.Hide,
    QtCore.QEvent.Type.Close,
    QtCore.QEvent.Type.Move,
    QtCore.QEvent.Type.Resize,
    QtCore.QEvent.Type.LayoutRequest,
    QtCore.QEvent.Type.Scroll,
    QtCore.QEvent.Type.ContentsRectChange,
    QtCore.QEvent.Type.PaletteChange,
    QtCore.QEvent.Type.StyleChange,
}
_ALL_INPUTS = frozenset(_INPUT_EVENTS.values())
_CardDirection = typing.Literal["left", "right", "top", "bottom"]
_UI_OBJECT_ID_PATTERN = r"[A-Za-z][A-Za-z0-9_.-]*"
_TUTORIAL_TEXT_TOKEN_PATTERN = re.compile(
    rf"\[\[(ui):({_UI_OBJECT_ID_PATTERN})\]\]"
    rf"|\[\[(menu):({_UI_OBJECT_ID_PATTERN}(?:\|{_UI_OBJECT_ID_PATTERN})+)\]\]"
)

_CARD_BACKGROUND_ROLE = QtGui.QPalette.ColorRole.Base
_CARD_BORDER_ROLE = QtGui.QPalette.ColorRole.Mid
_TEXT_ROLE = QtGui.QPalette.ColorRole.Text
_MUTED_TEXT_ROLE = QtGui.QPalette.ColorRole.PlaceholderText
_UI_BACKGROUND_ROLE = QtGui.QPalette.ColorRole.AlternateBase
_UI_BORDER_ROLE = QtGui.QPalette.ColorRole.Mid
_UI_TEXT_ROLE = QtGui.QPalette.ColorRole.Text
_MENU_BACKGROUND_ROLE = QtGui.QPalette.ColorRole.Button
_MENU_BORDER_ROLE = QtGui.QPalette.ColorRole.Mid
_MENU_ACTION_BORDER_ROLE = QtGui.QPalette.ColorRole.Highlight
_MENU_TEXT_ROLE = QtGui.QPalette.ColorRole.ButtonText
_BUTTON_TEXT_ROLE = QtGui.QPalette.ColorRole.ButtonText
_OVERLAY_ROLE = QtGui.QPalette.ColorRole.Shadow
_SPOTLIGHT_BORDER_ROLE = QtGui.QPalette.ColorRole.Highlight


@dataclasses.dataclass(frozen=True)
class _TutorialTextSpan:
    text: str
    kind: typing.Literal["plain", "ui", "menu", "menu_action", "menu_separator"] = (
        "plain"
    )


@dataclasses.dataclass(frozen=True)
class _TutorialText:
    spans: tuple[_TutorialTextSpan, ...]

    @property
    def plain_text(self) -> str:
        return "".join(span.text for span in self.spans)

    @classmethod
    def from_plain_text(cls, text: str) -> _TutorialText:
        return cls((_TutorialTextSpan(text),))


_TutorialTextKind: typing.TypeAlias = typing.Literal[
    "plain", "space", "ui", "menu", "menu_action", "menu_separator"
]


class TutorialStepUnavailableError(RuntimeError):
    """Error raised when a tutorial step cannot resolve its interface target."""


class TutorialDebugActionError(RuntimeError):
    """Error raised when tutorial debug automation cannot complete a step."""


def _weak_ref(obj: object) -> Callable[[], object | None]:
    try:
        return weakref.ref(obj)
    except TypeError:
        return lambda: obj


class ActionTarget:
    """A menu action and its optional owning menu."""

    def __init__(
        self, action: QtGui.QAction, menu: QtWidgets.QMenu | None = None
    ) -> None:
        self._action = _weak_ref(action)
        self._menu = _weak_ref(menu) if menu is not None else lambda: None

    @property
    def action(self) -> QtGui.QAction | None:
        action = self._action()
        return action if isinstance(action, QtGui.QAction) else None

    @property
    def menu(self) -> QtWidgets.QMenu | None:
        menu = self._menu()
        return menu if isinstance(menu, QtWidgets.QMenu) else None


class ModelIndexTarget:
    """A model index displayed by a specific item view."""

    def __init__(
        self,
        view: QtWidgets.QAbstractItemView,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ) -> None:
        self._view = _weak_ref(view)
        self.index = QtCore.QPersistentModelIndex(index)

    @property
    def view(self) -> QtWidgets.QAbstractItemView | None:
        view = self._view()
        return view if isinstance(view, QtWidgets.QAbstractItemView) else None


class ModelRowTarget:
    """A row displayed by a specific item view."""

    def __init__(
        self,
        view: QtWidgets.QAbstractItemView,
        row: int,
        column: int = 0,
        parent: QtCore.QModelIndex | None = None,
    ) -> None:
        self._view = _weak_ref(view)
        self.row = row
        self.column = column
        self.parent = QtCore.QPersistentModelIndex(
            QtCore.QModelIndex() if parent is None else parent
        )

    @property
    def view(self) -> QtWidgets.QAbstractItemView | None:
        view = self._view()
        return view if isinstance(view, QtWidgets.QAbstractItemView) else None


class GraphicsItemTarget:
    """A graphics item displayed by an optional graphics view."""

    def __init__(
        self,
        item: QtWidgets.QGraphicsItem,
        view: QtWidgets.QGraphicsView | None = None,
    ) -> None:
        self._item = _weak_ref(item)
        self._view = _weak_ref(view) if view is not None else lambda: None

    @property
    def item(self) -> QtWidgets.QGraphicsItem | None:
        item = self._item()
        return item if isinstance(item, QtWidgets.QGraphicsItem) else None

    @property
    def view(self) -> QtWidgets.QGraphicsView | None:
        view = self._view()
        return view if isinstance(view, QtWidgets.QGraphicsView) else None


class RectTarget:
    """A rectangle in global coordinates or in a widget's coordinates."""

    def __init__(
        self, rect: QtCore.QRect, widget: QtWidgets.QWidget | None = None
    ) -> None:
        self.rect = QtCore.QRect(rect)
        self._widget = widget

    @property
    def widget(self) -> QtWidgets.QWidget | None:
        return (
            self._widget
            if isinstance(self._widget, QtWidgets.QWidget)
            and erlab.interactive.utils.qt_is_valid(self._widget)
            else None
        )


class CompositeTarget:
    """Multiple targets represented by their rectangular union."""

    def __init__(self, *targets: object) -> None:
        if not targets:
            raise ValueError("A composite target must contain at least one target.")
        self.targets = tuple(
            QtCore.QRect(target) if isinstance(target, QtCore.QRect) else target
            for target in targets
        )


if typing.TYPE_CHECKING:
    Target: typing.TypeAlias = (
        QtWidgets.QWidget
        | QtGui.QAction
        | QtCore.QModelIndex
        | QtCore.QPersistentModelIndex
        | QtWidgets.QGraphicsItem
        | QtCore.QRect
        | ActionTarget
        | ModelIndexTarget
        | ModelRowTarget
        | GraphicsItemTarget
        | RectTarget
        | CompositeTarget
    )
    TargetResolver: typing.TypeAlias = Target | Callable[[], Target | None] | None


@dataclasses.dataclass(frozen=True)
class TourStep:
    """Declarative state for one tutorial step."""

    id: str
    title: str
    body: str
    mode: typing.Literal["information", "action"] = "information"
    target: TargetResolver = None
    target_required: bool = True
    reveal: Callable[[], None] | None = None
    allowed_inputs: frozenset[str] = _ALL_INPUTS
    allowed_objects: tuple[
        QtCore.QObject | Callable[[], QtCore.QObject | None], ...
    ] = ()
    event_predicate: Callable[[QtCore.QObject | None, QtCore.QEvent], bool] | None = (
        None
    )
    subscriptions: tuple[Callable[[], Signal], ...] = ()
    completion: Callable[[], bool] | None = None
    ready: Callable[[], bool] | None = None
    debug_action: Callable[[], None] | None = None
    continue_label: str = "Next"
    hint: str = ""
    recovery_label: str = ""
    recovery_hint: str = ""
    recovery_action: Callable[[], None] | None = None
    recovery_available: Callable[[], bool] | None = None
    card_position: typing.Literal[
        "target", "center", "left", "right", "top", "bottom"
    ] = "target"
    auto_advance: bool = True
    timeout_ms: int = 2500
    retry_interval_ms: int = 150

    def __post_init__(self) -> None:
        if not self.id or not self.title or not self.body:
            raise ValueError("Tour step id, title, and body must not be empty.")
        if self.mode not in {"information", "action"}:
            raise ValueError("Tour step mode must be 'information' or 'action'.")
        if not self.allowed_inputs <= _ALL_INPUTS:
            raise ValueError("Tour step allowed_inputs contains an unknown input type.")
        if self.card_position not in {
            "target",
            "center",
            "left",
            "right",
            "top",
            "bottom",
        }:
            raise ValueError("Tour step card position is not valid.")
        if bool(self.recovery_label) != (self.recovery_action is not None):
            raise ValueError(
                "Tour step recovery label and action must be specified together."
            )
        if self.recovery_action is None and (
            self.recovery_hint or self.recovery_available is not None
        ):
            raise ValueError("Tour step recovery options require a recovery action.")
        if self.timeout_ms < 0 or self.retry_interval_ms <= 0:
            raise ValueError("Tour step timeout and retry interval must be positive.")


@dataclasses.dataclass(frozen=True)
class TargetGeometry:
    """Resolved target geometry in global coordinates."""

    rect: QtCore.QRect
    window: QtWidgets.QWidget | None
    receivers: tuple[object, ...] = ()


def _valid_widget(widget: QtWidgets.QWidget | None) -> bool:
    return (
        widget is not None
        and erlab.interactive.utils.qt_is_valid(widget)
        and widget.isVisible()
    )


def _global_widget_rect(widget: QtWidgets.QWidget) -> QtCore.QRect:
    return QtCore.QRect(widget.mapToGlobal(QtCore.QPoint()), widget.size())


def _window_for_global_rect(rect: QtCore.QRect) -> QtWidgets.QWidget | None:
    application = QtWidgets.QApplication.instance()
    if not isinstance(application, QtWidgets.QApplication):
        return None
    windows = [
        widget
        for widget in application.topLevelWidgets()
        if _valid_widget(widget)
        and not isinstance(widget, _TourOverlay)
        and not isinstance(widget, QtWidgets.QMenu)
    ]
    center = rect.center()
    return next(
        (
            window
            for window in reversed(windows)
            if _global_widget_rect(window).contains(center)
        ),
        windows[-1] if windows else None,
    )


def _action_menu(action: QtGui.QAction) -> QtWidgets.QMenu | None:
    parent = action.parent()
    if isinstance(parent, QtWidgets.QMenu) and _valid_widget(parent):
        return parent
    associated = getattr(action, "associatedObjects", None)
    if callable(associated):
        return next(
            (
                obj
                for obj in associated()
                if isinstance(obj, QtWidgets.QMenu) and _valid_widget(obj)
            ),
            None,
        )
    return None


def _view_for_index(
    index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
) -> QtWidgets.QAbstractItemView | None:
    application = QtWidgets.QApplication.instance()
    if not isinstance(application, QtWidgets.QApplication):
        return None
    for widget in application.allWidgets():
        if (
            isinstance(widget, QtWidgets.QAbstractItemView)
            and _valid_widget(widget)
            and widget.model() is index.model()
        ):
            return widget
    return None


def _view_for_graphics_item(
    item: QtWidgets.QGraphicsItem,
) -> QtWidgets.QGraphicsView | None:
    scene = item.scene()
    if scene is None:
        return None
    return next((view for view in scene.views() if _valid_widget(view)), None)


def target_geometry(target: Target | None) -> TargetGeometry | None:
    """Resolve a supported tour target to a global rectangle."""
    if target is None:
        return None
    if isinstance(target, CompositeTarget):
        parts: list[TargetGeometry] = []
        for member in target.targets:
            resolved = member() if callable(member) else member
            geometry = target_geometry(typing.cast("Target", resolved))
            if geometry is None:
                return None
            parts.append(geometry)
        rect = QtCore.QRect(parts[0].rect)
        composite_receivers: list[object] = []
        for part in parts:
            rect = rect.united(part.rect)
            composite_receivers.extend(part.receivers)
        return TargetGeometry(rect, parts[0].window, tuple(composite_receivers))
    if isinstance(target, QtWidgets.QWidget):
        if not _valid_widget(target):
            return None
        return TargetGeometry(_global_widget_rect(target), target.window(), (target,))
    if isinstance(target, ActionTarget):
        action = target.action
        menu = target.menu
        if action is None:
            return None
        target = action
    elif isinstance(target, QtGui.QAction):
        action = target
        menu = None
    else:
        action = None
        menu = None
    if action is not None:
        menu = menu if _valid_widget(menu) else _action_menu(action)
        if menu is None or not action.isVisible():
            return None
        rect = menu.actionGeometry(action)
        if rect.isEmpty():
            return None
        rect.moveTopLeft(menu.mapToGlobal(rect.topLeft()))
        return TargetGeometry(rect, menu.window(), (menu,))
    if isinstance(target, ModelIndexTarget):
        index = target.index
        view = target.view
    elif isinstance(target, ModelRowTarget):
        view = target.view
        model = view.model() if view is not None else None
        index = (
            model.index(target.row, target.column, QtCore.QModelIndex(target.parent))
            if model is not None
            else QtCore.QModelIndex()
        )
    elif isinstance(target, (QtCore.QModelIndex, QtCore.QPersistentModelIndex)):
        index = target
        view = _view_for_index(index)
    else:
        index = None
        view = None
    if index is not None:
        if view is None or not index.isValid() or not _valid_widget(view):
            return None
        rect = view.visualRect(QtCore.QModelIndex(index))
        if rect.isEmpty():
            return None
        viewport = view.viewport()
        if viewport is None:
            return None
        rect.moveTopLeft(viewport.mapToGlobal(rect.topLeft()))
        return TargetGeometry(rect, view.window(), (view, viewport))
    if isinstance(target, GraphicsItemTarget):
        item = target.item
        view = target.view
    elif isinstance(target, QtWidgets.QGraphicsItem):
        item = target
        view = _view_for_graphics_item(item)
    else:
        item = None
        view = None
    if item is not None:
        view = view if _valid_widget(view) else _view_for_graphics_item(item)
        if view is None or item.scene() is None or not item.isVisible():
            return None
        rect = view.mapFromScene(item.sceneBoundingRect()).boundingRect()
        viewport = view.viewport()
        if viewport is None:
            return None
        rect.moveTopLeft(viewport.mapToGlobal(rect.topLeft()))
        return TargetGeometry(rect, view.window(), (view, viewport))
    if isinstance(target, RectTarget):
        rect = QtCore.QRect(target.rect)
        widget = target.widget
        if widget is not None:
            if not _valid_widget(widget):
                return None
            rect.moveTopLeft(widget.mapToGlobal(rect.topLeft()))
            window = widget.window()
            receivers: tuple[object, ...] = (widget,)
        else:
            window = _window_for_global_rect(rect)
            receivers = ()
    elif isinstance(target, QtCore.QRect):
        rect = QtCore.QRect(target)
        window = _window_for_global_rect(rect)
        receivers = ()
    else:
        raise TypeError(f"Unsupported tour target: {type(target).__name__}")
    if rect.isEmpty():
        return None
    return TargetGeometry(rect, window, receivers)


class _CardHeader(QtWidgets.QWidget):
    drag_started = QtCore.Signal(QtCore.QPoint)
    drag_moved = QtCore.Signal(QtCore.QPoint)

    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("tutorialCardHeader")
        self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        self.setToolTip("Drag to move")
        self._dragging = False

    def mousePressEvent(self, event: QtGui.QMouseEvent | None) -> None:
        if event is not None and event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._dragging = True
            self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
            self.drag_started.emit(event.globalPosition().toPoint())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent | None) -> None:
        if event is not None and self._dragging:
            self.drag_moved.emit(event.globalPosition().toPoint())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent | None) -> None:
        if (
            event is not None
            and self._dragging
            and event.button() == QtCore.Qt.MouseButton.LeftButton
        ):
            self._dragging = False
            self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)


@dataclasses.dataclass(frozen=True)
class _TutorialTextItem:
    text: str
    kind: _TutorialTextKind
    rect: QtCore.QRectF
    baseline: float


class _TutorialTextWidget(QtWidgets.QWidget):
    """Render tutorial prose with palette-aware inline UI labels."""

    _horizontal_padding = 7.0
    _vertical_padding = 2.0
    _corner_radius = 5.0

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        text_role: QtGui.QPalette.ColorRole = _TEXT_ROLE,
    ) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Minimum,
        )
        self._text = _TutorialText.from_plain_text("")
        self._plain_text_role = text_role
        self._layout_width = -1
        self._layout_items: tuple[_TutorialTextItem, ...] = ()
        self._layout_height = 0.0

    def text(self) -> str:
        """Return the accessible plain-text representation."""
        return self._text.plain_text

    def setText(self, text: str) -> None:
        """Set plain text with the same interface as QLabel."""
        self.set_tutorial_text(_TutorialText.from_plain_text(text))

    def set_tutorial_text(self, text: _TutorialText) -> None:
        if text == self._text:
            return
        self._text = text
        self.setAccessibleName(text.plain_text)
        self._invalidate_layout()

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        self._ensure_layout(max(1, width))
        return max(1, int(self._layout_height + 0.999))

    def sizeHint(self) -> QtCore.QSize:
        width = 404
        return QtCore.QSize(width, self.heightForWidth(width))

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(80, int(QtGui.QFontMetricsF(self.font()).height() + 8))

    def event(self, event: QtCore.QEvent | None) -> bool:
        if event is not None and event.type() in {
            QtCore.QEvent.Type.FontChange,
            QtCore.QEvent.Type.PaletteChange,
            QtCore.QEvent.Type.StyleChange,
        }:
            self._invalidate_layout()
        return super().event(event)

    def resizeEvent(self, event: QtGui.QResizeEvent | None) -> None:
        self._ensure_layout(max(1, self.width()))
        super().resizeEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent | None) -> None:
        del event
        self._ensure_layout(max(1, self.width()))
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        palette = self.palette()
        font = self.font()
        painter.setFont(font)
        font_metrics = QtGui.QFontMetricsF(font)
        for item in self._layout_items:
            if item.kind == "space":
                continue
            if item.kind == "plain":
                painter.setPen(palette.color(self._plain_text_role))
                painter.drawText(
                    QtCore.QPointF(item.rect.left(), item.baseline), item.text
                )
                continue
            if item.kind == "menu_separator":
                painter.setPen(palette.color(_MUTED_TEXT_ROLE))
                painter.drawText(
                    QtCore.QPointF(item.rect.left(), item.baseline), item.text
                )
                continue
            background_role = (
                _UI_BACKGROUND_ROLE if item.kind == "ui" else _MENU_BACKGROUND_ROLE
            )
            border_role = (
                _UI_BORDER_ROLE
                if item.kind == "ui"
                else _MENU_ACTION_BORDER_ROLE
                if item.kind == "menu_action"
                else _MENU_BORDER_ROLE
            )
            text_role = _UI_TEXT_ROLE if item.kind == "ui" else _MENU_TEXT_ROLE
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(palette.color(background_role))
            painter.drawRoundedRect(item.rect, self._corner_radius, self._corner_radius)
            pen = QtGui.QPen(palette.color(border_role))
            pen.setWidthF(1.5 if item.kind == "menu_action" else 1.0)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(
                item.rect.adjusted(0.5, 0.5, -0.5, -0.5),
                self._corner_radius,
                self._corner_radius,
            )
            painter.setPen(palette.color(text_role))
            text_x = item.rect.left() + self._horizontal_padding
            text_y = (
                item.rect.center().y()
                + (font_metrics.ascent() - font_metrics.descent()) / 2
            )
            painter.drawText(QtCore.QPointF(text_x, text_y), item.text)

    def _invalidate_layout(self) -> None:
        self._layout_width = -1
        self._layout_items = ()
        self._layout_height = 0.0
        self.updateGeometry()
        self.update()

    def _ensure_layout(self, width: int) -> None:
        if width == self._layout_width:
            return
        self._layout_width = width
        self._layout_items, self._layout_height = self._make_layout(width)

    def _make_layout(self, width: int) -> tuple[tuple[_TutorialTextItem, ...], float]:
        font_metrics = QtGui.QFontMetricsF(self.font())
        text_height = font_metrics.height()
        chip_height = text_height + 2 * self._vertical_padding
        line_height = max(text_height, chip_height) + 1.0
        items: list[_TutorialTextItem] = []
        x = 0.0
        y = 0.0

        def new_line() -> None:
            nonlocal x, y
            x = 0.0
            y += line_height

        def add_item(text: str, kind: _TutorialTextKind) -> None:
            nonlocal x
            if kind == "space":
                item_width = font_metrics.horizontalAdvance(text)
                if x == 0.0:
                    return
            elif kind in {"ui", "menu", "menu_action"}:
                item_width = (
                    font_metrics.horizontalAdvance(text) + 2 * self._horizontal_padding
                )
            else:
                item_width = font_metrics.horizontalAdvance(text)
            if x > 0.0 and x + item_width > width and kind != "space":
                new_line()
            if kind == "space" and x + item_width > width:
                new_line()
                return
            item_height = (
                chip_height if kind in {"ui", "menu", "menu_action"} else text_height
            )
            top = y + (line_height - item_height) / 2
            rect = QtCore.QRectF(x, top, item_width, item_height)
            baseline = top + font_metrics.ascent()
            items.append(_TutorialTextItem(text, kind, rect, baseline))
            x += item_width

        spans = self._text.spans
        index = 0
        while index < len(spans):
            span = spans[index]
            if span.kind == "plain":
                for part in re.findall(r"\n|[^\S\n]+|[^\s]+", span.text):
                    if part == "\n":
                        new_line()
                    elif part.isspace():
                        add_item(part, "space")
                    else:
                        add_item(part, "plain")
                index += 1
                continue
            if span.kind == "menu_separator" and index + 1 < len(spans):
                next_span = spans[index + 1]
                separator_width = font_metrics.horizontalAdvance(span.text)
                next_width = (
                    font_metrics.horizontalAdvance(next_span.text)
                    + 2 * self._horizontal_padding
                )
                if x > 0.0 and x + separator_width + next_width > width:
                    new_line()
            add_item(span.text, span.kind)
            index += 1
        return tuple(items), y + line_height


class _TutorialExitButton(_CenteredIconToolButton):
    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self._refresh_icon()

    def changeEvent(self, event: QtCore.QEvent | None) -> None:
        if event is not None and event.type() == QtCore.QEvent.Type.PaletteChange:
            self._refresh_icon()
        super().changeEvent(event)

    def _refresh_icon(self) -> None:
        self.setIcon(
            erlab.interactive.utils.qtawesome.icon(
                "ph.x", color=self.palette().color(_BUTTON_TEXT_ROLE)
            )
        )


class _TutorialAdvanceButton(QtWidgets.QPushButton):
    def keyPressEvent(self, event: QtGui.QKeyEvent | None) -> None:
        if event is not None and event.key() in {
            QtCore.Qt.Key.Key_Enter,
            QtCore.Qt.Key.Key_Return,
        }:
            event.accept()
            if self.isEnabled():
                self.click()
            return
        super().keyPressEvent(event)


class _InstructionCard(QtWidgets.QFrame):
    continue_requested = QtCore.Signal()
    skip_requested = QtCore.Signal()
    recovery_requested = QtCore.Signal()
    exit_requested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget, *, debug: bool = False) -> None:
        super().__init__(parent)
        self.setObjectName("tutorialInstructionCard")
        self.setAutoFillBackground(False)
        self.setBackgroundRole(_CARD_BACKGROUND_ROLE)
        self.setForegroundRole(_TEXT_ROLE)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setMinimumWidth(0)
        self.setMaximumWidth(440)
        self._drag_offset = QtCore.QPoint()
        self._manual_position: QtCore.QPoint | None = None
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 14)
        layout.setSpacing(8)
        self.header = _CardHeader(self)
        header_layout = QtWidgets.QHBoxLayout(self.header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)
        header_text_layout = QtWidgets.QVBoxLayout()
        header_text_layout.setContentsMargins(0, 0, 0, 0)
        header_text_layout.setSpacing(3)
        self.progress = QtWidgets.QLabel(self.header)
        self.progress.setObjectName("tutorialProgress")
        self.progress.setForegroundRole(_MUTED_TEXT_ROLE)
        self.title = QtWidgets.QLabel(self.header)
        self.title.setObjectName("tutorialTitle")
        self.title.setForegroundRole(_TEXT_ROLE)
        self.title.setWordWrap(True)
        title_font = self.title.font()
        title_font.setBold(True)
        title_font.setPointSizeF(title_font.pointSizeF() + 2)
        self.title.setFont(title_font)
        self.progress.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )
        self.title.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        header_text_layout.addWidget(self.progress)
        header_text_layout.addWidget(self.title)
        header_layout.addLayout(header_text_layout, 1)
        self.exit_button = _TutorialExitButton(self.header)
        self.exit_button.setObjectName("tutorialExitButton")
        self.exit_button.setAutoRaise(True)
        self.exit_button.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        self.exit_button.setToolTip("Exit Tutorial")
        self.exit_button.setAccessibleName("Exit Tutorial")
        self.exit_button.setIconSize(QtCore.QSize(14, 14))
        self.exit_button.setFixedSize(24, 24)
        header_layout.addWidget(
            self.exit_button,
            0,
            QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignRight,
        )
        self.body = _TutorialTextWidget(self)
        self.body.setObjectName("tutorialBody")
        self.hint = _TutorialTextWidget(self, _MUTED_TEXT_ROLE)
        self.hint.setObjectName("tutorialHint")
        buttons = QtWidgets.QHBoxLayout()
        self.recovery_button = _TutorialAdvanceButton("Back", self)
        self.recovery_button.setObjectName("tutorialRecoveryButton")
        self.recovery_button.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.recovery_button.hide()
        buttons.addWidget(self.recovery_button)
        buttons.addStretch(1)
        self.skip_button = _TutorialAdvanceButton("Skip", self)
        self.skip_button.setObjectName("tutorialSkipButton")
        self.skip_button.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.skip_button.setVisible(debug)
        buttons.addWidget(self.skip_button)
        self.continue_button = _TutorialAdvanceButton("Next", self)
        self.continue_button.setObjectName("tutorialContinueButton")
        self.continue_button.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        buttons.addWidget(self.continue_button)
        layout.addWidget(self.header)
        layout.addWidget(self.body)
        layout.addWidget(self.hint)
        layout.addLayout(buttons)
        self.skip_button.clicked.connect(self.skip_requested)
        self.continue_button.clicked.connect(self.continue_requested)
        self.recovery_button.clicked.connect(self.recovery_requested)
        self.exit_button.clicked.connect(self.exit_requested)
        self.header.drag_started.connect(self._start_drag)
        self.header.drag_moved.connect(self._move_drag)

    def paintEvent(self, event: QtGui.QPaintEvent | None) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        card_rect = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.setPen(QtGui.QPen(self.palette().color(_CARD_BORDER_ROLE), 1.0))
        painter.setBrush(self.palette().color(_CARD_BACKGROUND_ROLE))
        painter.drawRoundedRect(card_rect, 10.0, 10.0)

    def reset_manual_position(self) -> None:
        self._manual_position = None

    def _start_drag(self, global_position: QtCore.QPoint) -> None:
        self._drag_offset = global_position - self.mapToGlobal(QtCore.QPoint())

    def _move_drag(self, global_position: QtCore.QPoint) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        point = parent.mapFromGlobal(global_position - self._drag_offset)
        self._manual_position = self._bounded_position(point, parent.rect())
        self.move(self._manual_position)

    def _bounded_position(
        self, point: QtCore.QPoint, bounds: QtCore.QRect
    ) -> QtCore.QPoint:
        margin = 8
        min_x = bounds.left() + margin
        min_y = bounds.top() + margin
        max_x = max(min_x, bounds.right() - self.width() - margin)
        max_y = max(min_y, bounds.bottom() - self.height() - margin)
        return QtCore.QPoint(
            max(min_x, min(point.x(), max_x)),
            max(min_y, min(point.y(), max_y)),
        )

    def place(
        self,
        target: QtCore.QRect,
        bounds: QtCore.QRect,
        *,
        centered: bool = False,
        preferred_direction: _CardDirection | None = None,
    ) -> None:
        self.adjustSize()
        available = bounds.size() - QtCore.QSize(16, 16)
        available.setWidth(max(1, available.width()))
        available.setHeight(max(1, available.height()))
        preferred = self.sizeHint()
        width = min(preferred.width(), available.width())
        height = preferred.height()
        layout = self.layout()
        if layout is not None and layout.hasHeightForWidth():
            height = max(height, layout.heightForWidth(width))
        size = QtCore.QSize(width, height).boundedTo(available)
        self.resize(size)
        if layout is not None:
            layout.activate()
        size = self.size()
        if self._manual_position is not None:
            self._manual_position = self._bounded_position(
                self._manual_position, bounds
            )
            self.move(self._manual_position)
            return
        if centered:
            self.move(
                bounds.center() - QtCore.QPoint(size.width() // 2, size.height() // 2)
            )
            return
        gap = 12
        spaces: dict[_CardDirection, int] = {
            "right": bounds.right() - target.right(),
            "left": target.left() - bounds.left(),
            "bottom": bounds.bottom() - target.bottom(),
            "top": target.top() - bounds.top(),
        }
        direction = (
            preferred_direction
            if preferred_direction is not None
            else max(spaces, key=spaces.__getitem__)
        )
        if direction == "left":
            point = QtCore.QPoint(
                target.left() - size.width() - gap,
                target.center().y() - size.height() // 2,
            )
        elif direction == "bottom":
            point = QtCore.QPoint(
                target.center().x() - size.width() // 2,
                target.bottom() + gap,
            )
        elif direction == "top":
            point = QtCore.QPoint(
                target.center().x() - size.width() // 2,
                target.top() - size.height() - gap,
            )
        else:
            point = QtCore.QPoint(
                target.right() + gap,
                target.center().y() - size.height() // 2,
            )
        self.move(self._bounded_position(point, bounds))


class _TourOverlay(QtWidgets.QWidget):
    def __init__(self, window: QtWidgets.QWidget) -> None:
        super().__init__(window)
        self.setObjectName("tutorialOverlay")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self._spotlight: QtCore.QRect | None = None
        self.hide()

    def set_spotlight(
        self,
        rect: QtCore.QRect | None,
    ) -> None:
        self._spotlight = None if rect is None else QtCore.QRect(rect)
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent | None) -> None:
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        color = self.palette().color(_OVERLAY_ROLE)
        color.setAlpha(56)
        if self._spotlight is None:
            painter.fillRect(self.rect(), color)
            return
        rect = self._spotlight.adjusted(-6, -6, 6, 6)
        shade = QtGui.QPainterPath()
        shade.setFillRule(QtCore.Qt.FillRule.OddEvenFill)
        shade.addRect(QtCore.QRectF(self.rect()))
        shade.addRoundedRect(QtCore.QRectF(rect), 6.0, 6.0)
        painter.fillPath(shade, color)
        outline = QtCore.QRectF(rect).intersected(
            QtCore.QRectF(self.rect()).adjusted(1.0, 1.0, -1.0, -1.0)
        )
        if not outline.isEmpty():
            accent = self.palette().color(_SPOTLIGHT_BORDER_ROLE)
            accent.setAlpha(180)
            painter.setPen(QtGui.QPen(accent, 2.0))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(outline, 6.0, 6.0)


class TourController(QtCore.QObject):
    """Drive tour steps, overlays, and application-wide input gating."""

    exit_requested = QtCore.Signal()
    finished = QtCore.Signal()
    step_changed = QtCore.Signal(object)

    def __init__(
        self,
        steps: Sequence[TourStep],
        window: QtWidgets.QWidget | None = None,
        parent: QtCore.QObject | None = None,
        text_resolver: Callable[[str], str | None] | None = None,
        *,
        debug: bool = False,
    ) -> None:
        super().__init__(parent)
        if not steps:
            raise ValueError("A tour must contain at least one step.")
        ids = [step.id for step in steps]
        if len(ids) != len(set(ids)):
            raise ValueError("Tour step ids must be unique.")
        self._steps = tuple(steps)
        self._window = _weak_ref(window) if window is not None else lambda: None
        self._text_resolver = text_resolver
        self._debug = debug
        self._index = -1
        self._running = False
        self._overlays: list[
            tuple[weakref.ReferenceType[QtWidgets.QWidget], _TourOverlay]
        ] = []
        self._card: _InstructionCard | None = None
        self._target_receivers: tuple[Callable[[], object | None], ...] = ()
        self._connections: list[tuple[Signal, Callable[..., None]]] = []
        self._title_override: str | None = None
        self._body_override: str | None = None
        self._target_started = 0.0
        self._refresh_pending = False
        self._fatal_error: RuntimeError | None = None
        self._retry_timer = QtCore.QTimer(self)
        self._retry_timer.setSingleShot(True)
        self._retry_timer.timeout.connect(self._retry_target)
        self._debug_timer = QtCore.QTimer(self)
        self._debug_timer.setInterval(50)
        self._debug_timer.timeout.connect(self._poll_debug_action)
        self._debug_step_index: int | None = None
        self._debug_deadline = 0.0
        self._step_activation = 0
        self._focused_button_key: tuple[int, str] | None = None

    @property
    def current_step(self) -> TourStep | None:
        if 0 <= self._index < len(self._steps):
            return self._steps[self._index]
        return None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def debug(self) -> bool:
        """Whether tutorial debug controls are enabled."""
        return self._debug

    def start(self) -> None:
        if self._running:
            return
        application = QtWidgets.QApplication.instance()
        if application is None:
            raise RuntimeError("QApplication is not available.")
        owner = self._window()
        if not isinstance(owner, QtWidgets.QWidget):
            raise TypeError("The tutorial window is not available.")
        self._card = _InstructionCard(owner, debug=self._debug)
        self._card.continue_requested.connect(self._card_continue)
        self._card.skip_requested.connect(self._card_skip)
        self._card.recovery_requested.connect(self._card_recovery)
        self._card.exit_requested.connect(self.request_exit)
        self._fatal_error = None
        self._focused_button_key = None
        self._running = True
        self._index = 0
        application.installEventFilter(self)
        self._activate_step()

    def close(self) -> None:
        if not self._running:
            return
        self._running = False
        self._retry_timer.stop()
        self._debug_timer.stop()
        self._debug_step_index = None
        self._disconnect_step_signals()
        application = QtWidgets.QApplication.instance()
        if application is not None:
            application.removeEventFilter(self)
        for _, overlay in self._overlays:
            if erlab.interactive.utils.qt_is_valid(overlay):
                self._dispose_overlay(overlay)
        self._overlays.clear()
        card = self._card
        self._card = None
        if card is not None and erlab.interactive.utils.qt_is_valid(card):
            with contextlib.suppress(RuntimeError, TypeError):
                card.continue_requested.disconnect(self._card_continue)
            with contextlib.suppress(RuntimeError, TypeError):
                card.skip_requested.disconnect(self._card_skip)
            with contextlib.suppress(RuntimeError, TypeError):
                card.recovery_requested.disconnect(self._card_recovery)
            with contextlib.suppress(RuntimeError, TypeError):
                card.exit_requested.disconnect(self.request_exit)
            card.hide()
            card.setParent(None)
            card.deleteLater()
        self._target_receivers = ()
        self._index = -1

    def continue_step(self) -> None:
        if self._fatal_error is not None:
            return
        step = self.current_step
        if step is None or self._step_text(step)[-1]:
            return
        if step.mode == "information" and not self._is_ready(step):
            return
        if step.mode == "action" and not self._is_complete(step):
            return
        self._advance()

    def notify_state_changed(self) -> None:
        if self._fatal_error is not None:
            return
        step = self.current_step
        if (
            step is not None
            and step.mode == "action"
            and step.auto_advance
            and self._is_complete(step)
            and not self._step_text(step)[-1]
        ):
            self._advance()
            return
        self._refresh()

    def update_current(
        self, *, title: str | None = None, body: str | None = None
    ) -> None:
        if title is not None:
            self._title_override = title
        if body is not None:
            self._body_override = body
        self._refresh()

    def request_exit(self) -> None:
        if not self._running:
            return
        self.exit_requested.emit()

    def eventFilter(
        self, watched: QtCore.QObject | None, event: QtCore.QEvent | None
    ) -> bool:
        if not self._running or event is None:
            return False
        try:
            event_type = event.type()
        except RuntimeError:
            return False
        if event_type in _REPOSITION_EVENTS:
            self._retain_card_before_parent_closes(watched, event_type)
            self._schedule_refresh()
        input_type = _INPUT_EVENTS.get(event_type)
        if input_type is None:
            return False
        # Native input first reaches the top-level QWindow. Let Qt translate and
        # dispatch it to the child widget, where the tutorial can validate the real
        # receiver. Blocking the QWindow event prevents every control from receiving
        # mouse and key input.
        if isinstance(watched, QtGui.QWindow):
            return False
        if (
            event_type == QtCore.QEvent.Type.KeyPress
            and isinstance(event, QtGui.QKeyEvent)
            and event.key() == QtCore.Qt.Key.Key_Escape
        ):
            event.accept()
            self.request_exit()
            return True
        step = self.current_step
        if step is None:
            return False
        if self._event_is_allowed(step, watched, event, input_type):
            return False
        event.accept()
        return True

    def _retain_card_before_parent_closes(
        self, watched: QtCore.QObject | None, event_type: QtCore.QEvent.Type
    ) -> None:
        if event_type not in {QtCore.QEvent.Type.Close, QtCore.QEvent.Type.Hide}:
            return
        card = self._card
        owner = self._window()
        if (
            card is None
            or not erlab.interactive.utils.qt_is_valid(card)
            or watched is not card.parentWidget()
            or not isinstance(owner, QtWidgets.QWidget)
            or watched is owner
            or not erlab.interactive.utils.qt_is_valid(owner)
        ):
            return
        card.hide()
        card.setParent(owner)
        card.reset_manual_position()

    def _activate_step(self) -> None:
        step = self.current_step
        if step is None:
            return
        self._step_activation += 1
        activation = self._step_activation
        self._retry_timer.stop()
        if self._debug_step_index != self._index:
            self._debug_timer.stop()
            self._debug_step_index = None
        self._disconnect_step_signals()
        self._title_override = None
        self._body_override = None
        self._target_started = time.monotonic()
        card = self._card
        if card is not None and erlab.interactive.utils.qt_is_valid(card):
            card.reset_manual_position()
        if step.reveal is not None:
            with contextlib.suppress(RuntimeError):
                step.reveal()
        for resolver in step.subscriptions:
            try:
                signal = resolver()
                slot = self._on_observed
                signal.connect(slot)
            except (AttributeError, RuntimeError, TypeError):
                continue
            self._connections.append((signal, slot))
        self.step_changed.emit(step)
        if not self._running or activation != self._step_activation:
            return
        if (
            step.mode == "action"
            and step.auto_advance
            and self._is_complete(step)
            and not self._step_text(step)[-1]
        ):
            erlab.interactive.utils.single_shot(
                self,
                0,
                lambda: self._advance_if_active(activation),
            )
            return
        self._refresh()

    def _advance_if_active(self, activation: int) -> None:
        if self._running and activation == self._step_activation:
            self._advance()

    def _advance(self) -> None:
        if not self._running:
            return
        if self._index + 1 >= len(self._steps):
            self.close()
            self.finished.emit()
            return
        self._index += 1
        self._activate_step()

    def _is_ready(self, step: TourStep) -> bool:
        if step.ready is None:
            return True
        with contextlib.suppress(RuntimeError):
            return bool(step.ready())
        return False

    def _is_complete(self, step: TourStep) -> bool:
        if step.completion is None:
            return False
        with contextlib.suppress(RuntimeError):
            return bool(step.completion())
        return False

    def _recovery_is_available(self, step: TourStep) -> bool:
        if step.recovery_action is None:
            return False
        if step.mode == "action" and self._is_complete(step):
            return False
        if step.recovery_available is None:
            return True
        with contextlib.suppress(RuntimeError):
            return bool(step.recovery_available())
        return False

    def _on_observed(self, *_args: object) -> None:
        self.notify_state_changed()

    def _disconnect_step_signals(self) -> None:
        for signal, slot in self._connections:
            with contextlib.suppress(RuntimeError, TypeError):
                signal.disconnect(slot)
        self._connections.clear()

    def _resolve_target(self, step: TourStep) -> tuple[TargetGeometry | None, bool]:
        resolver = step.target
        if resolver is None:
            return None, False
        try:
            target = resolver() if callable(resolver) else resolver
            geometry = target_geometry(target)
        except (RuntimeError, TypeError):
            geometry = None
        return geometry, geometry is None and step.target_required

    def _resolve_tutorial_text(self, template: str) -> tuple[_TutorialText, bool]:
        spans: list[_TutorialTextSpan] = []
        missing = False
        offset = 0
        resolver = self._text_resolver
        for match in _TUTORIAL_TEXT_TOKEN_PATTERN.finditer(template):
            if match.start() > offset:
                spans.append(_TutorialTextSpan(template[offset : match.start()]))
            object_ids = (
                (match.group(2),)
                if match.group(1) == "ui"
                else tuple(typing.cast("str", match.group(4)).split("|"))
            )
            values: list[str] = []
            for object_id in object_ids:
                if resolver is None:
                    value = None
                else:
                    try:
                        value = resolver(object_id)
                    except (RuntimeError, TypeError):
                        value = None
                if not value:
                    missing = True
                    value = ""
                values.append(value)
            if match.group(1) == "ui":
                spans.append(_TutorialTextSpan(values[0], "ui"))
            else:
                for index, value in enumerate(values):
                    if index:
                        spans.append(
                            _TutorialTextSpan(
                                " \N{SINGLE RIGHT-POINTING ANGLE QUOTATION MARK} ",
                                "menu_separator",
                            )
                        )
                    kind: typing.Literal["menu", "menu_action"] = (
                        "menu_action" if index == len(values) - 1 else "menu"
                    )
                    spans.append(_TutorialTextSpan(value, kind))
            offset = match.end()
        if offset < len(template):
            spans.append(_TutorialTextSpan(template[offset:]))
        unmatched = _TUTORIAL_TEXT_TOKEN_PATTERN.sub("", template)
        if "[[ui:" in unmatched or "[[menu:" in unmatched:
            missing = True
        return _TutorialText(tuple(spans)), missing

    def _render_text(self, template: str) -> tuple[str, bool]:
        rendered, missing = self._resolve_tutorial_text(template)
        return rendered.plain_text, missing

    def _step_content(
        self, step: TourStep
    ) -> tuple[_TutorialText, _TutorialText, _TutorialText, _TutorialText, bool]:
        templates = (
            self._title_override or step.title,
            self._body_override or step.body,
            step.hint,
            step.continue_label,
        )
        rendered: list[_TutorialText] = []
        missing = False
        for template in templates:
            value, value_missing = self._resolve_tutorial_text(template)
            rendered.append(value)
            missing = missing or value_missing
        return rendered[0], rendered[1], rendered[2], rendered[3], missing

    def _step_text(self, step: TourStep) -> tuple[str, str, str, str, bool]:
        title, body, hint, continue_label, missing = self._step_content(step)
        return (
            title.plain_text,
            body.plain_text,
            hint.plain_text,
            continue_label.plain_text,
            missing,
        )

    def _refresh(self) -> None:
        self._refresh_pending = False
        if not self._running or self._fatal_error is not None:
            return
        step = self.current_step
        if step is None:
            return
        geometry, target_missing = self._resolve_target(step)
        *_, text_missing = self._step_text(step)
        action_complete = step.mode == "action" and self._is_complete(step)
        if action_complete:
            target_missing = False
        missing = target_missing or text_missing
        if missing:
            elapsed_ms = (time.monotonic() - self._target_started) * 1000
            if elapsed_ms >= step.timeout_ms:
                self._raise_unavailable_step(
                    step,
                    target_missing=target_missing,
                    text_missing=text_missing,
                )
            if not self._retry_timer.isActive():
                self._retry_timer.start(step.retry_interval_ms)
        self._target_receivers = (
            tuple(_weak_ref(receiver) for receiver in geometry.receivers)
            if geometry is not None
            else ()
        )
        windows = self._visible_windows()
        display_geometry = geometry
        target_window = geometry.window if geometry is not None else None
        active_window = QtWidgets.QApplication.activeWindow()
        if (
            action_complete
            and active_window in windows
            and (geometry is None or geometry.window is not active_window)
        ):
            display_geometry = None
            target_window = active_window
        primary = self._primary_window(target_window, windows)
        self._sync_overlays(windows)
        for window_ref, overlay in self._overlays:
            window = window_ref()
            if window is None or not erlab.interactive.utils.qt_is_valid(
                window, overlay
            ):
                continue
            overlay.setGeometry(window.rect())
            has_spotlight = window is target_window
            spotlight = None
            if has_spotlight and display_geometry is not None:
                spotlight = QtCore.QRect(display_geometry.rect)
                spotlight.moveTopLeft(
                    overlay.mapFromGlobal(display_geometry.rect.topLeft())
                )
            overlay.set_spotlight(spotlight)
            overlay.raise_()
            overlay.show()
        card = self._card
        if (
            card is not None
            and erlab.interactive.utils.qt_is_valid(card)
            and primary is not None
        ):
            if card.parentWidget() is not primary:
                card.hide()
                card.setParent(primary)
                card.reset_manual_position()
            self._update_card(card, step, missing, text_missing)
            target = QtCore.QRect(primary.rect().center(), QtCore.QSize(1, 1))
            if display_geometry is not None:
                target = QtCore.QRect(display_geometry.rect)
                target.moveTopLeft(
                    primary.mapFromGlobal(display_geometry.rect.topLeft())
                )
            card.show()
            card.place(
                target,
                primary.rect(),
                centered=(step.card_position == "center" or display_geometry is None),
                preferred_direction=(
                    step.card_position
                    if step.card_position in {"left", "right", "top", "bottom"}
                    else None
                ),
            )
            card.raise_()
            self._focus_step_button(card, step)

    def _retry_target(self) -> None:
        if self._running and self._fatal_error is None:
            self._refresh()

    def _schedule_refresh(self) -> None:
        if self._refresh_pending or self._fatal_error is not None:
            return
        self._refresh_pending = True
        QtCore.QTimer.singleShot(0, self._refresh)

    def _visible_windows(self) -> list[QtWidgets.QWidget]:
        application = QtWidgets.QApplication.instance()
        if not isinstance(application, QtWidgets.QApplication):
            return []
        return [
            window
            for window in application.topLevelWidgets()
            if _valid_widget(window)
            and not isinstance(window, _TourOverlay)
            and not isinstance(window, QtWidgets.QMenu)
            and window.windowType()
            not in {QtCore.Qt.WindowType.Popup, QtCore.Qt.WindowType.ToolTip}
        ]

    def _primary_window(
        self,
        target_window: QtWidgets.QWidget | None,
        windows: list[QtWidgets.QWidget],
    ) -> QtWidgets.QWidget | None:
        if isinstance(target_window, QtWidgets.QMenu):
            parent = target_window.parentWidget()
            target_window = None if parent is None else parent.window()
        if target_window in windows:
            return target_window
        owner = self._window()
        if isinstance(owner, QtWidgets.QWidget) and owner in windows:
            return owner
        active = QtWidgets.QApplication.activeWindow()
        if active in windows:
            return active
        return windows[-1] if windows else None

    def _sync_overlays(self, windows: list[QtWidgets.QWidget]) -> None:
        kept: list[tuple[weakref.ReferenceType[QtWidgets.QWidget], _TourOverlay]] = []
        existing: dict[int, _TourOverlay] = {}
        for window_ref, overlay in self._overlays:
            window = window_ref()
            if (
                window is not None
                and window in windows
                and erlab.interactive.utils.qt_is_valid(window, overlay)
            ):
                existing[id(window)] = overlay
                kept.append((window_ref, overlay))
            elif erlab.interactive.utils.qt_is_valid(overlay):
                self._dispose_overlay(overlay)
        self._overlays = kept
        for window in windows:
            if id(window) in existing:
                continue
            overlay = _TourOverlay(window)
            self._overlays.append((weakref.ref(window), overlay))

    def _dispose_overlay(self, overlay: _TourOverlay) -> None:
        overlay.hide()
        overlay.deleteLater()

    def _update_card(
        self,
        card: _InstructionCard,
        step: TourStep,
        missing: bool,
        text_missing: bool,
    ) -> None:
        card.progress.setText(f"{self._index + 1} / {len(self._steps)}")
        title, body, hint, continue_label, _ = self._step_content(step)
        recovery_available = self._recovery_is_available(step)
        if text_missing:
            card.title.setText("Tutorial step unavailable")
            card.body.setText("A required interface label is not available.")
        else:
            card.title.setText(title.plain_text)
            card.body.set_tutorial_text(body)
        if missing:
            if text_missing:
                card.title.setText("Preparing tutorial step")
                card.body.setText("Wait a moment.")
            if recovery_available:
                card.hint.setText(step.recovery_hint)
            else:
                card.hint.set_tutorial_text(hint)
            card.continue_button.setText(
                "Next" if text_missing else continue_label.plain_text
            )
            card.continue_button.setEnabled(False)
            card.continue_button.show()
            card.skip_button.setEnabled(
                self._debug
                and not recovery_available
                and step.mode == "action"
                and step.debug_action is not None
                and self._debug_step_index is None
            )
        else:
            if recovery_available:
                card.hint.setText(step.recovery_hint)
            else:
                card.hint.set_tutorial_text(hint)
            card.continue_button.setText(continue_label.plain_text)
            can_continue = (
                self._is_ready(step)
                if step.mode == "information"
                else self._is_complete(step)
            )
            card.continue_button.setEnabled(can_continue)
            card.continue_button.show()
            card.skip_button.setEnabled(
                self._debug
                and not recovery_available
                and self._debug_step_index is None
                and (
                    (step.mode == "information" and can_continue)
                    or (step.mode == "action" and step.debug_action is not None)
                )
            )
        card.recovery_button.setText(step.recovery_label or "Back")
        card.recovery_button.setVisible(recovery_available)
        card.recovery_button.setEnabled(recovery_available)
        card.hint.setVisible(bool(card.hint.text()))
        card.adjustSize()

    def _focus_step_button(self, card: _InstructionCard, step: TourStep) -> None:
        button: QtWidgets.QPushButton | None = None
        if card.recovery_button.isVisible() and card.recovery_button.isEnabled():
            button = card.recovery_button
        elif self._debug and step.mode == "action" and card.skip_button.isEnabled():
            button = card.skip_button
        elif card.continue_button.isEnabled():
            button = card.continue_button
        if button is None:
            return
        key = (self._index, button.objectName())
        if self._focused_button_key == key:
            return
        button.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
        self._focused_button_key = key

    def _card_recovery(self) -> None:
        if self._fatal_error is not None:
            return
        step = self.current_step
        if step is None or not self._recovery_is_available(step):
            return
        action = step.recovery_action
        if action is None:  # pragma: no cover - validated by TourStep
            return
        index = self._index
        action()
        if self._running and self._index == index:
            self.notify_state_changed()

    def _revisit_step(self, step_id: str) -> None:
        if not self._running:
            return
        try:
            index = next(
                index for index, step in enumerate(self._steps) if step.id == step_id
            )
        except StopIteration as exc:
            raise ValueError(f"Unknown tutorial step {step_id!r}.") from exc
        if index >= self._index:
            raise ValueError("A recovery step must precede the current tutorial step.")
        self._index = index
        self._activate_step()

    def _card_skip(self) -> None:
        if not self._debug or self._fatal_error is not None:
            return
        step = self.current_step
        if step is None:
            return
        if step.mode == "information":
            self._card_continue()
            return
        if step.debug_action is None:
            self._raise_debug_error(step, "no debug action is defined")
        self._debug_step_index = self._index
        self._debug_deadline = time.monotonic() + 60.0
        self._debug_timer.start()
        self._refresh()
        QtCore.QTimer.singleShot(
            0,
            lambda index=self._index: self._run_debug_action(index),
        )

    def _run_debug_action(self, index: int) -> None:
        if index != self._index or self._debug_step_index != index:
            return
        step = self.current_step
        if step is None or step.debug_action is None:
            return
        try:
            step.debug_action()
        except Exception as error:
            self._raise_debug_error(step, "the debug action failed", error)
        self._poll_debug_action()

    def _poll_debug_action(self) -> None:
        index = self._debug_step_index
        if index is None:
            self._debug_timer.stop()
            return
        if index != self._index:
            self._debug_timer.stop()
            self._debug_step_index = None
            return
        step = self.current_step
        if step is None:
            self._debug_timer.stop()
            self._debug_step_index = None
            return
        if self._is_complete(step):
            self._debug_timer.stop()
            self._debug_step_index = None
            self._advance()
            return
        if time.monotonic() >= self._debug_deadline:
            self._raise_debug_error(step, "the completion condition timed out")

    def _raise_debug_error(
        self,
        step: TourStep,
        reason: str,
        cause: Exception | None = None,
    ) -> typing.Never:
        error = TutorialDebugActionError(
            f"Tutorial debug action for step {step.id!r} ({step.title!r}) "
            f"failed: {reason}."
        )
        self._fatal_error = error
        self._retry_timer.stop()
        self._debug_timer.stop()
        self._debug_step_index = None
        self._handle_fatal_error(error)
        if cause is None:
            raise error
        raise error from cause

    def _card_continue(self) -> None:
        step = self.current_step
        if step is None:
            return
        geometry, target_missing = self._resolve_target(step)
        *_, text_missing = self._step_text(step)
        if step.mode == "action" and self._is_complete(step):
            target_missing = False
        missing = target_missing or text_missing
        if missing:
            self._target_started = time.monotonic()
            if step.reveal is not None:
                with contextlib.suppress(RuntimeError):
                    step.reveal()
            self._refresh()
            return
        del geometry
        self.continue_step()

    def _raise_unavailable_step(
        self,
        step: TourStep,
        *,
        target_missing: bool,
        text_missing: bool,
    ) -> typing.Never:
        reasons: list[str] = []
        if target_missing:
            reasons.append("the target is not visible")
        if text_missing:
            reasons.append("required interface text could not be resolved")
        reason = " and ".join(reasons) or "the step is not available"
        error = TutorialStepUnavailableError(
            f"Tutorial step {step.id!r} ({step.title!r}) is unavailable: {reason}."
        )
        self._fatal_error = error
        self._retry_timer.stop()
        self._handle_fatal_error(error)
        raise error

    def _handle_fatal_error(self, error: RuntimeError) -> None:
        """Remove tutorial input gating after a fatal error."""
        del error
        self.close()

    def _event_is_allowed(
        self,
        step: TourStep,
        watched: QtCore.QObject | None,
        event: QtCore.QEvent,
        input_type: str,
    ) -> bool:
        if self._is_message_details_toggle(watched):
            return True
        if self._is_overlay_object(watched):
            return True
        if step.event_predicate is not None:
            with contextlib.suppress(RuntimeError):
                if step.event_predicate(watched, event):
                    return True
        if input_type not in step.allowed_inputs:
            return False
        allowed: list[object] = []
        for resolver in step.allowed_objects:
            with contextlib.suppress(RuntimeError):
                obj = resolver() if callable(resolver) else resolver
                if obj is not None:
                    allowed.append(obj)
        if step.mode == "action":
            allowed.extend(
                receiver
                for receiver_ref in self._target_receivers
                if (receiver := receiver_ref()) is not None
            )
        return any(self._object_contains(obj, watched) for obj in allowed)

    @staticmethod
    def _is_message_details_toggle(watched: QtCore.QObject | None) -> bool:
        current = watched
        with contextlib.suppress(RuntimeError):
            while current is not None:
                if current.objectName() == "messageDialogDetailsToggle":
                    return True
                current = current.parent()
        return False

    def _is_overlay_object(self, watched: QtCore.QObject | None) -> bool:
        card = self._card
        if (
            card is not None
            and erlab.interactive.utils.qt_is_valid(card)
            and self._object_contains(card, watched)
        ):
            return True
        return any(
            watched is overlay or self._object_contains(overlay, watched)
            for _, overlay in self._overlays
            if erlab.interactive.utils.qt_is_valid(overlay)
        )

    @staticmethod
    def _object_contains(parent: object, watched: QtCore.QObject | None) -> bool:
        if watched is parent:
            return True
        if not isinstance(parent, QtCore.QObject) or watched is None:
            return False
        current: QtCore.QObject | None = watched
        with contextlib.suppress(RuntimeError):
            while current is not None:
                if current is parent:
                    return True
                current = current.parent()
        return False
