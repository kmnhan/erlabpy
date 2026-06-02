"""Qt widgets used by the private Figure Composer framework."""

from __future__ import annotations

import typing

# Matplotlib's Qt backend should see the qtpy-selected binding first.
# isort: off
from qtpy import QtCore, QtGui, QtWidgets

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
# isort: on

import erlab.interactive.utils
from erlab.interactive._figurecomposer._axes import _all_axes_for_shape
from erlab.interactive._figurecomposer._defaults import (
    _figure_draw_context,
    _figure_style_context,
)

if typing.TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from erlab.interactive._figurecomposer._state import FigureSubplotsState


def _step_toolbar_button(
    parent: QtWidgets.QWidget,
    object_name: str,
    text: str,
    tooltip: str,
) -> QtWidgets.QToolButton:
    button = QtWidgets.QToolButton(parent)
    button.setObjectName(object_name)
    button.setText(text)
    button.setToolTip(tooltip)
    button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
    button.setSizePolicy(
        QtWidgets.QSizePolicy.Policy.Minimum,
        QtWidgets.QSizePolicy.Policy.Fixed,
    )
    return button


class _StyledFigureCanvas(FigureCanvas):
    def draw(self, *args, **kwargs):
        with _figure_draw_context():
            return super().draw(*args, **kwargs)

    def print_figure(self, *args, **kwargs):
        with _figure_draw_context():
            return super().print_figure(*args, **kwargs)


class _FigureComposerDisplayWindow(QtWidgets.QMainWindow):
    """Top-level Matplotlib display owned by a figure composer."""

    sigCanvasSizeChanged = QtCore.Signal(float, float)

    def __init__(self, setup: FigureSubplotsState) -> None:
        super().__init__(None)
        self._closing_from_owner = False
        self._suppress_resize_signal = False
        self._resize_signal_pending = False
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, False)

        with _figure_style_context():
            self.figure = Figure(
                figsize=setup.figsize,
                dpi=setup.dpi,
                layout=setup.layout,
            )
        self.canvas = _StyledFigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        root = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)
        self.setCentralWidget(root)
        self.setWindowTitle("Figure")
        for widget in (self, root, self.toolbar, self.canvas):
            widget.installEventFilter(self)
        self._close_shortcut = erlab.interactive.utils._install_close_shortcut(
            self, self.hide
        )

    def eventFilter(
        self,
        watched: QtCore.QObject | None,
        event: QtCore.QEvent | None,
    ) -> bool:
        if self._is_close_shortcut_event(event):
            self.hide()
            if event is not None:
                event.accept()
            return True
        return super().eventFilter(watched, event)

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
        QtCore.QTimer.singleShot(0, self._allow_resize_signal)

    @QtCore.Slot()
    def _allow_resize_signal(self) -> None:
        self._suppress_resize_signal = False

    def show_for_setup(
        self, setup: FigureSubplotsState, title: str, *, activate: bool
    ) -> None:
        self.setWindowTitle(title)
        self.resize_to_setup(setup)
        self.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating, not activate
        )
        if not self.isVisible():
            self.show()
        if activate:
            self.activateWindow()
            self.raise_()

    def close_from_owner(self) -> None:
        self._closing_from_owner = True
        self.close()

    def resizeEvent(self, event: QtGui.QResizeEvent | None) -> None:
        if event is not None:
            super().resizeEvent(event)
        if self._suppress_resize_signal or self._resize_signal_pending:
            return
        self._resize_signal_pending = True
        QtCore.QTimer.singleShot(0, self._emit_canvas_size_changed)

    @QtCore.Slot()
    def _emit_canvas_size_changed(self) -> None:
        self._resize_signal_pending = False
        if self._suppress_resize_signal:
            return
        canvas_size = self.canvas.size()
        if canvas_size.isEmpty():
            return
        dpi = float(typing.cast("typing.Any", self.figure)._original_dpi)
        if dpi <= 0.0:
            return
        self.sigCanvasSizeChanged.emit(
            canvas_size.width() / dpi,
            canvas_size.height() / dpi,
        )

    def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:
        if self._closing_from_owner:
            if event is not None:
                super().closeEvent(event)
            return
        if event is not None:
            event.ignore()
        self.hide()


class _AxesSelectorWidget(QtWidgets.QWidget):
    """Grid selector for target axes in the current figure layout."""

    sigSelectionChanged = QtCore.Signal(object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._nrows = 1
        self._ncols = 1
        self._cell_labels: dict[tuple[int, int], str] = {}
        self._selected_axes: frozenset[tuple[int, int]] = frozenset({(0, 0)})
        self._anchor_axis: tuple[int, int] | None = (0, 0)
        self._drag_origin: tuple[int, int] | None = None
        self._drag_base: frozenset[tuple[int, int]] = frozenset()
        self._drag_additive = False
        self._hovered_axis: tuple[int, int] | None = None
        self.setObjectName("figureComposerAxesSelector")
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.setMouseTracking(True)
        self.setMinimumHeight(118)
        self.setToolTip(
            "Click an axis to target it. Shift-click selects a range; "
            "Ctrl/Cmd-click toggles one axis; drag selects a rectangular range."
        )

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(280, 150)

    def set_grid(
        self,
        nrows: int,
        ncols: int,
        labels: Mapping[tuple[int, int], str] | None = None,
    ) -> None:
        self._nrows = max(1, nrows)
        self._ncols = max(1, ncols)
        self._cell_labels = dict(labels or {})
        self.set_selected_axes(
            tuple(axis for axis in self._selected_axes if self._axis_in_grid(axis)),
            emit=False,
        )
        self.updateGeometry()
        self.update()

    def set_selected_axes(
        self, axes: Sequence[tuple[int, int]], *, emit: bool = False
    ) -> None:
        selected = frozenset(axis for axis in axes if self._axis_in_grid(axis))
        if selected:
            self._anchor_axis = next(iter(sorted(selected)))
        elif self._anchor_axis is not None and not self._axis_in_grid(
            self._anchor_axis
        ):
            self._anchor_axis = None
        if selected == self._selected_axes:
            return
        self._selected_axes = selected
        self.update()
        if emit:
            self.sigSelectionChanged.emit(self.selected_axes())

    def selected_axes(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            axis
            for axis in _all_axes_for_shape(self._nrows, self._ncols)
            if axis in self._selected_axes
        )

    def cell_rect(self, axis: tuple[int, int]) -> QtCore.QRect:
        if not self._axis_in_grid(axis):
            return QtCore.QRect()
        grid_rect = self._grid_rect()
        gap = self._cell_gap()
        cell_width = (grid_rect.width() - gap * max(self._ncols - 1, 0)) / self._ncols
        cell_height = (grid_rect.height() - gap * max(self._nrows - 1, 0)) / self._nrows
        row, col = axis
        return QtCore.QRect(
            round(grid_rect.left() + col * (cell_width + gap)),
            round(grid_rect.top() + row * (cell_height + gap)),
            round(cell_width),
            round(cell_height),
        )

    def paintEvent(self, event: QtGui.QPaintEvent | None) -> None:
        if event is not None:
            super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        palette = self.palette()
        background = palette.color(QtGui.QPalette.ColorRole.Base)
        border = palette.color(QtGui.QPalette.ColorRole.Mid)
        text_color = palette.color(QtGui.QPalette.ColorRole.Text)
        selected_fill = palette.color(QtGui.QPalette.ColorRole.Highlight)
        selected_text = palette.color(QtGui.QPalette.ColorRole.HighlightedText)
        hover_fill = palette.color(QtGui.QPalette.ColorRole.AlternateBase)
        for axis in _all_axes_for_shape(self._nrows, self._ncols):
            rect = self.cell_rect(axis)
            selected = axis in self._selected_axes
            hovered = axis == self._hovered_axis
            painter.setPen(QtGui.QPen(selected_fill if selected else border, 1.4))
            painter.setBrush(
                QtGui.QBrush(
                    selected_fill if selected else hover_fill if hovered else background
                )
            )
            painter.drawRoundedRect(rect, 6, 6)
            painter.setPen(selected_text if selected else text_color)
            painter.drawText(
                rect,
                QtCore.Qt.AlignmentFlag.AlignCenter,
                self._axis_label(axis),
            )

    def mousePressEvent(self, event: QtGui.QMouseEvent | None) -> None:
        if event is None or event.button() != QtCore.Qt.MouseButton.LeftButton:
            if event is not None:
                super().mousePressEvent(event)
            return
        axis = self._axis_at(event.position().toPoint())
        if axis is None:
            return
        modifiers = event.modifiers()
        additive = self._has_toggle_modifier(modifiers)
        shift = bool(modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier)
        self._drag_additive = additive
        self._drag_base = self._selected_axes if additive else frozenset()
        if shift and self._anchor_axis is not None:
            origin = self._anchor_axis
            selected = self._rect_axes(origin, axis)
            self._drag_origin = origin
        elif additive:
            selected = set(self._selected_axes)
            if axis in selected and len(selected) > 1:
                selected.remove(axis)
            else:
                selected.add(axis)
            self._drag_origin = axis
        else:
            selected = {axis}
            self._drag_origin = axis
        if not shift:
            self._anchor_axis = axis
        self.set_selected_axes(tuple(selected), emit=True)
        event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent | None) -> None:
        if event is None:
            return
        axis = self._axis_at(event.position().toPoint())
        if axis != self._hovered_axis:
            self._hovered_axis = axis
            self.update()
        if (
            self._drag_origin is not None
            and event.buttons() & QtCore.Qt.MouseButton.LeftButton
            and axis is not None
        ):
            axes = set(self._rect_axes(self._drag_origin, axis))
            if self._drag_additive:
                axes.update(self._drag_base)
            self.set_selected_axes(tuple(axes), emit=True)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent | None) -> None:
        self._drag_origin = None
        self._drag_base = frozenset()
        self._drag_additive = False
        if event is not None:
            event.accept()

    def leaveEvent(self, event: QtCore.QEvent | None) -> None:
        self._hovered_axis = None
        self.update()
        if event is not None:
            super().leaveEvent(event)

    def _grid_rect(self) -> QtCore.QRect:
        margin = 8
        rect = self.rect()
        if rect.isEmpty():
            rect = QtCore.QRect(QtCore.QPoint(0, 0), self.sizeHint())
        return rect.adjusted(margin, margin, -margin, -margin)

    def _cell_gap(self) -> int:
        return 5

    def _axis_in_grid(self, axis: tuple[int, int]) -> bool:
        row, col = axis
        return 0 <= row < self._nrows and 0 <= col < self._ncols

    def _axis_label(self, axis: tuple[int, int]) -> str:
        return self._cell_labels.get(axis, f"axs[{axis[0]}, {axis[1]}]")

    def _axis_at(self, pos: QtCore.QPoint) -> tuple[int, int] | None:
        for axis in _all_axes_for_shape(self._nrows, self._ncols):
            if self.cell_rect(axis).contains(pos):
                return axis
        return None

    def _rect_axes(
        self, start: tuple[int, int], end: tuple[int, int]
    ) -> tuple[tuple[int, int], ...]:
        row0, col0 = start
        row1, col1 = end
        rows = range(min(row0, row1), max(row0, row1) + 1)
        cols = range(min(col0, col1), max(col0, col1) + 1)
        return tuple((row, col) for row in rows for col in cols)

    @staticmethod
    def _has_toggle_modifier(modifiers: QtCore.Qt.KeyboardModifier) -> bool:
        return bool(
            modifiers
            & (
                QtCore.Qt.KeyboardModifier.ControlModifier
                | QtCore.Qt.KeyboardModifier.MetaModifier
            )
        )
