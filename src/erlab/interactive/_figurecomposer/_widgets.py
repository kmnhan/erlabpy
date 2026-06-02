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
from erlab.interactive._figurecomposer._state import (
    FigureGridSpecGridState,
    FigureGridSpecSpanState,
)

if typing.TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from erlab.interactive._figurecomposer._state import FigureSubplotsState


class _GridSpecRegionInfo(typing.NamedTuple):
    region_id: str
    kind: typing.Literal["axes", "grid"]
    span: FigureGridSpecSpanState
    label: str
    valid: bool = True


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
        erlab.interactive.utils.single_shot(self, 0, self._allow_resize_signal)

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
        erlab.interactive.utils.single_shot(
            self, 0, self._emit_canvas_size_changed, self.canvas
        )

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
    _CELL_WIDTH = 46
    _CELL_HEIGHT = 26
    _CELL_GAP = 3
    _GRID_MARGIN = 4
    _MIN_WIDTH = 68
    _MIN_HEIGHT = 34

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
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Maximum,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.setToolTip(
            "Click an axis to target it. Shift-click selects a range; "
            "Ctrl/Cmd-click toggles one axis; drag selects a rectangular range."
        )

    def sizeHint(self) -> QtCore.QSize:
        margin = 2 * self._GRID_MARGIN
        width = (
            margin
            + self._ncols * self._CELL_WIDTH
            + max(self._ncols - 1, 0) * self._CELL_GAP
        )
        height = (
            margin
            + self._nrows * self._CELL_HEIGHT
            + max(self._nrows - 1, 0) * self._CELL_GAP
        )
        return QtCore.QSize(
            max(self._MIN_WIDTH, width),
            max(self._MIN_HEIGHT, height),
        )

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(self._MIN_WIDTH, self._MIN_HEIGHT)

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
            painter.setPen(QtGui.QPen(selected_fill if selected else border, 1.0))
            painter.setBrush(
                QtGui.QBrush(
                    selected_fill if selected else hover_fill if hovered else background
                )
            )
            painter.drawRoundedRect(rect, 4, 4)
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
        rect = self.rect()
        if rect.isEmpty():
            rect = QtCore.QRect(QtCore.QPoint(0, 0), self.sizeHint())
        return rect.adjusted(
            self._GRID_MARGIN,
            self._GRID_MARGIN,
            -self._GRID_MARGIN,
            -self._GRID_MARGIN,
        )

    def _cell_gap(self) -> int:
        return self._CELL_GAP

    def _axis_in_grid(self, axis: tuple[int, int]) -> bool:
        row, col = axis
        return 0 <= row < self._nrows and 0 <= col < self._ncols

    def _axis_label(self, axis: tuple[int, int]) -> str:
        return self._cell_labels.get(axis, f"{axis[0]}, {axis[1]}")

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


class _GridSpecAxesSelectorWidget(QtWidgets.QWidget):
    """Read-only nested GridSpec selector for target axes."""

    sigSelectionChanged = QtCore.Signal(object)

    _CELL_WIDTH = 46
    _CELL_HEIGHT = 28
    _CELL_GAP = 3
    _GRID_MARGIN = 4
    _NESTED_INSET = 4
    _MIN_WIDTH = 100
    _MIN_HEIGHT = 44

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._root_grid = FigureGridSpecGridState(grid_id="root", label="Root")
        self._labels: dict[str, str] = {}
        self._selected_axes: frozenset[str] = frozenset()
        self._anchor_axis_id: str | None = None
        self._drag_origin: QtCore.QPoint | None = None
        self._drag_base: frozenset[str] = frozenset()
        self._drag_additive = False
        self._hovered_axis_id: str | None = None
        self.setObjectName("figureComposerGridSpecAxesSelector")
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.setMouseTracking(True)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Maximum,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.setToolTip(
            "Click an axis to target it.\n"
            "Shift-click selects a contiguous display range.\n"
            "Ctrl/Cmd-click toggles one axis; drag selects visible axes."
        )

    def sizeHint(self) -> QtCore.QSize:
        root = self._root_grid
        margin = 2 * self._GRID_MARGIN
        width = (
            margin
            + root.ncols * self._CELL_WIDTH
            + max(root.ncols - 1, 0) * self._CELL_GAP
        )
        height = (
            margin
            + root.nrows * self._CELL_HEIGHT
            + max(root.nrows - 1, 0) * self._CELL_GAP
        )
        return QtCore.QSize(
            max(self._MIN_WIDTH, width),
            max(self._MIN_HEIGHT, height),
        )

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(self._MIN_WIDTH, self._MIN_HEIGHT)

    def set_layout(
        self,
        root_grid: FigureGridSpecGridState,
        labels: Mapping[str, str] | None = None,
    ) -> None:
        self._root_grid = root_grid
        self._labels = dict(labels or {})
        valid_axes = set(self.axes_ids())
        self._selected_axes = frozenset(
            axes_id for axes_id in self._selected_axes if axes_id in valid_axes
        )
        if self._anchor_axis_id not in valid_axes:
            self._anchor_axis_id = next(iter(self._selected_axes), None)
        self.updateGeometry()
        self.update()

    def set_selected_axes_ids(
        self, axes_ids: Sequence[str], *, emit: bool = False
    ) -> None:
        valid_axes = set(self.axes_ids())
        selected = frozenset(
            axes_id for axes_id in dict.fromkeys(axes_ids) if axes_id in valid_axes
        )
        if selected:
            self._anchor_axis_id = next(iter(self._ordered_selected_ids(selected)))
        elif self._anchor_axis_id not in valid_axes:
            self._anchor_axis_id = None
        if selected == self._selected_axes:
            return
        self._selected_axes = selected
        self.update()
        if emit:
            self.sigSelectionChanged.emit(self.selected_axes_ids())

    def selected_axes_ids(self) -> tuple[str, ...]:
        return self._ordered_selected_ids(self._selected_axes)

    def axes_ids(self) -> tuple[str, ...]:
        return tuple(axes_id for axes_id, _rect in self._axis_rect_items())

    def axis_rect(self, axes_id: str) -> QtCore.QRect:
        return self._axis_rects().get(axes_id, QtCore.QRect())

    def paintEvent(self, event: QtGui.QPaintEvent | None) -> None:
        if event is not None:
            super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self._draw_grid(painter, self._root_grid, self._grid_rect())

    def mousePressEvent(self, event: QtGui.QMouseEvent | None) -> None:
        if event is None or event.button() != QtCore.Qt.MouseButton.LeftButton:
            if event is not None:
                super().mousePressEvent(event)
            return
        axis_id = self._axis_at(event.position().toPoint())
        if axis_id is None:
            return
        modifiers = event.modifiers()
        additive = self._has_toggle_modifier(modifiers)
        shift = bool(modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier)
        self._drag_origin = event.position().toPoint()
        self._drag_additive = additive
        self._drag_base = self._selected_axes if additive else frozenset()
        if shift and self._anchor_axis_id is not None:
            selected = set(self._range_axes_ids(self._anchor_axis_id, axis_id))
        elif additive:
            selected = set(self._selected_axes)
            if axis_id in selected:
                selected.remove(axis_id)
            else:
                selected.add(axis_id)
            self._anchor_axis_id = axis_id
        else:
            selected = {axis_id}
            self._anchor_axis_id = axis_id
        self.set_selected_axes_ids(tuple(selected), emit=True)
        event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent | None) -> None:
        if event is None:
            return
        axis_id = self._axis_at(event.position().toPoint())
        if axis_id != self._hovered_axis_id:
            self._hovered_axis_id = axis_id
            self.update()
        if (
            self._drag_origin is not None
            and event.buttons() & QtCore.Qt.MouseButton.LeftButton
        ):
            drag_rect = QtCore.QRect(
                self._drag_origin, event.position().toPoint()
            ).normalized()
            selected = set(self._axes_intersecting(drag_rect))
            if self._drag_additive:
                selected.update(self._drag_base)
            if selected:
                self.set_selected_axes_ids(tuple(selected), emit=True)
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
        self._hovered_axis_id = None
        self.update()
        if event is not None:
            super().leaveEvent(event)

    def _grid_rect(self) -> QtCore.QRect:
        rect = self.rect()
        if rect.isEmpty():
            rect = QtCore.QRect(QtCore.QPoint(0, 0), self.sizeHint())
        return rect.adjusted(
            self._GRID_MARGIN,
            self._GRID_MARGIN,
            -self._GRID_MARGIN,
            -self._GRID_MARGIN,
        )

    def _axis_rects(self) -> dict[str, QtCore.QRect]:
        return dict(self._axis_rect_items())

    def _axis_rect_items(self) -> tuple[tuple[str, QtCore.QRect], ...]:
        items: list[tuple[str, QtCore.QRect]] = []
        self._collect_axis_rects(self._root_grid, self._grid_rect(), items)
        return tuple(items)

    def _collect_axis_rects(
        self,
        grid: FigureGridSpecGridState,
        grid_rect: QtCore.QRect,
        items: list[tuple[str, QtCore.QRect]],
    ) -> None:
        items.extend(
            (axis.axes_id, self._span_rect(grid, grid_rect, axis.span))
            for axis in grid.axes
            if self._span_within_grid(grid, axis.span)
        )
        for child in grid.child_grids:
            if child.span is None or not self._span_within_grid(grid, child.span):
                continue
            child_rect = self._span_rect(grid, grid_rect, child.span).adjusted(
                self._NESTED_INSET,
                self._NESTED_INSET,
                -self._NESTED_INSET,
                -self._NESTED_INSET,
            )
            self._collect_axis_rects(child, child_rect, items)

    def _draw_grid(
        self,
        painter: QtGui.QPainter,
        grid: FigureGridSpecGridState,
        grid_rect: QtCore.QRect,
    ) -> None:
        palette = self.palette()
        background = palette.color(QtGui.QPalette.ColorRole.Base)
        border = palette.color(QtGui.QPalette.ColorRole.Mid)
        nested_fill = palette.color(QtGui.QPalette.ColorRole.AlternateBase)
        text_color = palette.color(QtGui.QPalette.ColorRole.Text)
        selected_fill = palette.color(QtGui.QPalette.ColorRole.Highlight)
        selected_text = palette.color(QtGui.QPalette.ColorRole.HighlightedText)
        hover_fill = palette.color(QtGui.QPalette.ColorRole.AlternateBase)

        painter.setPen(QtGui.QPen(border, 1.0))
        painter.setBrush(QtGui.QBrush(background))
        painter.drawRoundedRect(grid_rect.adjusted(0, 0, -1, -1), 3, 3)
        self._draw_grid_lines(painter, grid, grid_rect, border)

        for child in grid.child_grids:
            if child.span is None or not self._span_within_grid(grid, child.span):
                continue
            rect = self._span_rect(grid, grid_rect, child.span)
            painter.setPen(QtGui.QPen(border, 1.0))
            painter.setBrush(QtGui.QBrush(nested_fill))
            painter.drawRoundedRect(rect, 4, 4)
            child_rect = rect.adjusted(
                self._NESTED_INSET,
                self._NESTED_INSET,
                -self._NESTED_INSET,
                -self._NESTED_INSET,
            )
            self._draw_grid(painter, child, child_rect)

        for axis in grid.axes:
            if not self._span_within_grid(grid, axis.span):
                continue
            rect = self._span_rect(grid, grid_rect, axis.span)
            selected = axis.axes_id in self._selected_axes
            hovered = axis.axes_id == self._hovered_axis_id
            painter.setPen(QtGui.QPen(selected_fill if selected else border, 1.0))
            painter.setBrush(
                QtGui.QBrush(
                    selected_fill if selected else hover_fill if hovered else background
                )
            )
            painter.drawRoundedRect(rect, 4, 4)
            painter.setPen(selected_text if selected else text_color)
            text = painter.fontMetrics().elidedText(
                self._axis_label(axis.axes_id),
                QtCore.Qt.TextElideMode.ElideRight,
                max(0, rect.width() - 6),
            )
            painter.drawText(
                rect.adjusted(3, 1, -3, -1),
                QtCore.Qt.AlignmentFlag.AlignCenter,
                text,
            )

    def _draw_grid_lines(
        self,
        painter: QtGui.QPainter,
        grid: FigureGridSpecGridState,
        grid_rect: QtCore.QRect,
        color: QtGui.QColor,
    ) -> None:
        painter.setPen(QtGui.QPen(color, 0.8))
        x_edges = self._axis_edges(
            grid_rect.left(), grid_rect.width(), grid.ncols, grid.width_ratios
        )
        y_edges = self._axis_edges(
            grid_rect.top(), grid_rect.height(), grid.nrows, grid.height_ratios
        )
        for x in x_edges[1:-1]:
            painter.drawLine(round(x), grid_rect.top(), round(x), grid_rect.bottom())
        for y in y_edges[1:-1]:
            painter.drawLine(grid_rect.left(), round(y), grid_rect.right(), round(y))

    def _span_rect(
        self,
        grid: FigureGridSpecGridState,
        grid_rect: QtCore.QRect,
        span: FigureGridSpecSpanState,
    ) -> QtCore.QRect:
        x_edges = self._axis_edges(
            grid_rect.left(), grid_rect.width(), grid.ncols, grid.width_ratios
        )
        y_edges = self._axis_edges(
            grid_rect.top(), grid_rect.height(), grid.nrows, grid.height_ratios
        )
        gap = self._CELL_GAP
        left = x_edges[span.col_start] + (gap / 2 if span.col_start else 0)
        right = x_edges[span.col_stop] - (gap / 2 if span.col_stop < grid.ncols else 0)
        top = y_edges[span.row_start] + (gap / 2 if span.row_start else 0)
        bottom = y_edges[span.row_stop] - (gap / 2 if span.row_stop < grid.nrows else 0)
        return QtCore.QRect(
            round(left),
            round(top),
            max(1, round(right - left)),
            max(1, round(bottom - top)),
        ).adjusted(1, 1, -1, -1)

    def _axis_edges(
        self, start: int, size: int, count: int, ratios: Sequence[float]
    ) -> tuple[float, ...]:
        if count <= 0:
            return (float(start),)
        if len(ratios) != count or sum(ratios) <= 0:
            ratios = tuple(1.0 for _index in range(count))
        total = float(sum(ratios))
        edges = [float(start)]
        current = float(start)
        for ratio in ratios:
            current += size * float(ratio) / total
            edges.append(current)
        return tuple(edges)

    def _axis_at(self, pos: QtCore.QPoint) -> str | None:
        for axes_id, rect in reversed(self._axis_rect_items()):
            if rect.contains(pos):
                return axes_id
        return None

    def _axes_intersecting(self, rect: QtCore.QRect) -> tuple[str, ...]:
        return tuple(
            axes_id
            for axes_id, axis_rect in self._axis_rect_items()
            if axis_rect.intersects(rect)
        )

    def _ordered_selected_ids(self, selected: frozenset[str]) -> tuple[str, ...]:
        return tuple(axes_id for axes_id in self.axes_ids() if axes_id in selected)

    def _range_axes_ids(self, start: str, end: str) -> tuple[str, ...]:
        axes_ids = self.axes_ids()
        if start not in axes_ids or end not in axes_ids:
            return (end,)
        start_index = axes_ids.index(start)
        end_index = axes_ids.index(end)
        return axes_ids[min(start_index, end_index) : max(start_index, end_index) + 1]

    def _axis_label(self, axes_id: str) -> str:
        return self._labels.get(axes_id, axes_id)

    @staticmethod
    def _span_within_grid(
        grid: FigureGridSpecGridState, span: FigureGridSpecSpanState
    ) -> bool:
        return (
            0 <= span.row_start < span.row_stop <= grid.nrows
            and 0 <= span.col_start < span.col_stop <= grid.ncols
        )

    @staticmethod
    def _has_toggle_modifier(modifiers: QtCore.Qt.KeyboardModifier) -> bool:
        return bool(
            modifiers
            & (
                QtCore.Qt.KeyboardModifier.ControlModifier
                | QtCore.Qt.KeyboardModifier.MetaModifier
            )
        )


class _GridSpecLayoutWidget(QtWidgets.QWidget):
    """GridSpec region editor for the active Figure Composer grid."""

    sigRegionCreated = QtCore.Signal(object, str)
    sigRegionChanged = QtCore.Signal(str, object)
    sigRegionSelected = QtCore.Signal(str, str)
    sigNestedGridActivated = QtCore.Signal(str)

    _CELL_WIDTH = 36
    _CELL_HEIGHT = 24
    _CELL_GAP = 3
    _GRID_MARGIN = 4
    _MIN_WIDTH = 100
    _MIN_HEIGHT = 48
    _HANDLE_SIZE = 7
    _HANDLE_HIT_SIZE = 13

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._nrows = 1
        self._ncols = 1
        self._regions: tuple[_GridSpecRegionInfo, ...] = ()
        self._selected_region_id = ""
        self._drag_mode: typing.Literal["none", "create", "resize"] = "none"
        self._drag_origin: tuple[int, int] | None = None
        self._drag_current: tuple[int, int] | None = None
        self._drag_region_id = ""
        self._drag_moved = False
        self._resize_handle: typing.Literal["nw", "ne", "sw", "se"] | None = None
        self._resize_original_span: FigureGridSpecSpanState | None = None
        self._resize_preview_span: FigureGridSpecSpanState | None = None
        self._creation_kind: typing.Literal["axes", "grid"] = "axes"
        self._drag_creation_kind: typing.Literal["axes", "grid"] = "axes"
        self.setObjectName("figureComposerGridSpecLayoutWidget")
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Maximum,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.setToolTip(
            "Drag empty cells to create a region.\n"
            "Select a region, then drag a corner handle to resize it.\n"
            "Double-click a nested grid region to edit that grid."
        )

    def sizeHint(self) -> QtCore.QSize:
        margin = 2 * self._GRID_MARGIN
        width = (
            margin
            + self._ncols * self._CELL_WIDTH
            + max(self._ncols - 1, 0) * self._CELL_GAP
        )
        height = (
            margin
            + self._nrows * self._CELL_HEIGHT
            + max(self._nrows - 1, 0) * self._CELL_GAP
        )
        return QtCore.QSize(
            max(self._MIN_WIDTH, width),
            max(self._MIN_HEIGHT, height),
        )

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(self._MIN_WIDTH, self._MIN_HEIGHT)

    def set_grid(
        self,
        nrows: int,
        ncols: int,
        regions: Sequence[_GridSpecRegionInfo],
    ) -> None:
        self._nrows = max(1, nrows)
        self._ncols = max(1, ncols)
        self._regions = tuple(regions)
        if self._selected_region_id and not any(
            region.region_id == self._selected_region_id for region in self._regions
        ):
            self._selected_region_id = ""
        self.updateGeometry()
        self.update()

    def set_creation_kind(self, kind: typing.Literal["axes", "grid"]) -> None:
        self._creation_kind = kind

    def selected_region_id(self) -> str:
        return self._selected_region_id

    def set_selected_region(self, region_id: str) -> None:
        if region_id and not any(
            region.region_id == region_id for region in self._regions
        ):
            self._selected_region_id = ""
        else:
            self._selected_region_id = region_id
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent | None) -> None:
        if event is not None:
            super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        palette = self.palette()
        cell_border = palette.color(QtGui.QPalette.ColorRole.Mid)
        cell_fill = palette.color(QtGui.QPalette.ColorRole.Base)
        axes_fill = palette.color(QtGui.QPalette.ColorRole.Highlight).lighter(170)
        grid_fill = palette.color(QtGui.QPalette.ColorRole.AlternateBase)
        selected = palette.color(QtGui.QPalette.ColorRole.Highlight)
        text_color = palette.color(QtGui.QPalette.ColorRole.Text)
        invalid_pen = QtGui.QColor(170, 40, 40)
        invalid_fill = QtGui.QColor(170, 40, 40, 45)

        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QBrush(cell_fill))
        for row in range(self._nrows):
            for col in range(self._ncols):
                painter.drawRect(self.cell_rect((row, col)))
        painter.setPen(QtGui.QPen(cell_border, 1.0))
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(self._grid_rect().adjusted(0, 0, -1, -1), 3, 3)

        for region in self._regions:
            rect = self.span_rect(region.span)
            is_selected = region.region_id == self._selected_region_id
            is_valid = region.valid and self._span_within_grid(region.span)
            pen_color = (
                selected
                if is_selected
                else invalid_pen
                if not is_valid
                else cell_border
            )
            painter.setPen(
                QtGui.QPen(
                    pen_color,
                    1.6 if is_selected else 1.2 if not is_valid else 1.0,
                )
            )
            if is_valid:
                brush = axes_fill if region.kind == "axes" else grid_fill
            else:
                brush = invalid_fill
            painter.setBrush(QtGui.QBrush(brush))
            painter.drawRoundedRect(rect, 5, 5)
            painter.setPen(invalid_pen if not is_valid else text_color)
            text = region.label or ("Axes" if region.kind == "axes" else "Grid")
            if not is_valid:
                text = f"{text} (outside)"
            text = painter.fontMetrics().elidedText(
                text,
                QtCore.Qt.TextElideMode.ElideRight,
                max(0, rect.width() - 6),
            )
            painter.drawText(
                rect.adjusted(3, 1, -3, -1),
                QtCore.Qt.AlignmentFlag.AlignCenter,
                text,
            )
            if is_selected and is_valid:
                self._draw_resize_handles(painter, rect, selected)

        preview_span = self._active_preview_span()
        if preview_span is not None:
            span = preview_span
            rect = self.span_rect(span)
            preview_color = selected
            preview_fill = QtGui.QColor(preview_color)
            preview_fill.setAlpha(35)
            pen = QtGui.QPen(preview_color, 1.4)
            pen.setStyle(QtCore.Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(QtGui.QBrush(preview_fill))
            painter.drawRoundedRect(rect, 5, 5)

    def mousePressEvent(self, event: QtGui.QMouseEvent | None) -> None:
        if event is None or event.button() != QtCore.Qt.MouseButton.LeftButton:
            if event is not None:
                super().mousePressEvent(event)
            return
        pos = event.position().toPoint()
        handle = self._handle_at(pos)
        if handle is not None:
            region = self._selected_region()
            if region is None:
                return
            self._drag_mode = "resize"
            self._drag_region_id = region.region_id
            self._resize_handle = handle
            self._resize_original_span = region.span
            self._resize_preview_span = region.span
            self._drag_moved = False
            event.accept()
            return
        region = self._region_at(pos)
        if region is not None:
            self._reset_drag()
            self._selected_region_id = region.region_id
            self.sigRegionSelected.emit(region.region_id, region.kind)
            self.update()
            event.accept()
            return
        cell = self._cell_at(pos)
        if cell is None:
            return
        self._drag_mode = "create"
        self._drag_origin = cell
        self._drag_current = cell
        self._drag_moved = False
        self._drag_region_id = ""
        self._drag_creation_kind = self._creation_kind
        event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent | None) -> None:
        if (
            event is not None
            and self._drag_mode == "resize"
            and event.buttons() & QtCore.Qt.MouseButton.LeftButton
        ):
            current = self._cell_at(event.position().toPoint(), clamp_to_grid=True)
            if current is not None and self._resize_original_span is not None:
                span = self._span_from_resize(
                    self._resize_original_span, self._resize_handle, current
                )
                self._resize_preview_span = span
                self._drag_moved = span != self._resize_original_span
                self.update()
            event.accept()
            return
        if (
            event is not None
            and self._drag_mode == "create"
            and self._drag_origin is not None
            and event.buttons() & QtCore.Qt.MouseButton.LeftButton
        ):
            current = self._cell_at(event.position().toPoint(), clamp_to_grid=True)
            self._drag_current = current
            self._drag_moved = current is not None and current != self._drag_origin
            self.update()
            event.accept()
            return
        if event is not None:
            self._update_hover_cursor(event.position().toPoint())
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent | None) -> None:
        if event is None or self._drag_mode == "none":
            return
        mode = self._drag_mode
        end_cell = self._cell_at(event.position().toPoint(), clamp_to_grid=True)
        origin = self._drag_origin
        region_id = self._drag_region_id
        moved = self._drag_moved
        span = self._resize_preview_span
        creation_kind = self._drag_creation_kind
        self._reset_drag()
        self.update()
        if mode == "resize":
            if moved and region_id and span is not None:
                self.sigRegionChanged.emit(region_id, span)
            event.accept()
            return
        if end_cell is not None and origin is not None:
            span = self._span_from_cells(origin, end_cell)
            self.sigRegionCreated.emit(span, creation_kind)
        event.accept()

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent | None) -> None:
        if event is None:
            return
        self._reset_drag()
        region = self._region_at(event.position().toPoint())
        if region is not None and region.kind == "grid":
            self._selected_region_id = region.region_id
            self.sigRegionSelected.emit(region.region_id, region.kind)
            self.sigNestedGridActivated.emit(region.region_id)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def cell_rect(self, axis: tuple[int, int]) -> QtCore.QRect:
        row, col = axis
        grid_rect = self._grid_rect()
        gap = self._CELL_GAP
        cell_width = (grid_rect.width() - gap * max(self._ncols - 1, 0)) / self._ncols
        cell_height = (grid_rect.height() - gap * max(self._nrows - 1, 0)) / self._nrows
        return QtCore.QRect(
            round(grid_rect.left() + col * (cell_width + gap)),
            round(grid_rect.top() + row * (cell_height + gap)),
            round(cell_width),
            round(cell_height),
        )

    def span_rect(self, span: FigureGridSpecSpanState) -> QtCore.QRect:
        row_start = min(max(span.row_start, 0), self._nrows - 1)
        row_stop = min(max(span.row_stop, row_start + 1), self._nrows)
        col_start = min(max(span.col_start, 0), self._ncols - 1)
        col_stop = min(max(span.col_stop, col_start + 1), self._ncols)
        start = self.cell_rect((row_start, col_start))
        end = self.cell_rect((row_stop - 1, col_stop - 1))
        return QtCore.QRect(start.topLeft(), end.bottomRight()).adjusted(1, 1, -1, -1)

    def _grid_rect(self) -> QtCore.QRect:
        rect = self.rect()
        if rect.isEmpty():
            rect = QtCore.QRect(QtCore.QPoint(0, 0), self.sizeHint())
        return rect.adjusted(
            self._GRID_MARGIN,
            self._GRID_MARGIN,
            -self._GRID_MARGIN,
            -self._GRID_MARGIN,
        )

    def _cell_at(
        self, pos: QtCore.QPoint, *, clamp_to_grid: bool = False
    ) -> tuple[int, int] | None:
        if clamp_to_grid:
            grid_rect = self._grid_rect()
            if grid_rect.isEmpty():
                return None
            x = min(max(pos.x(), grid_rect.left()), grid_rect.right())
            y = min(max(pos.y(), grid_rect.top()), grid_rect.bottom())
            gap = self._CELL_GAP
            cell_width = (
                grid_rect.width() - gap * max(self._ncols - 1, 0)
            ) / self._ncols
            cell_height = (
                grid_rect.height() - gap * max(self._nrows - 1, 0)
            ) / self._nrows
            col = int((x - grid_rect.left()) / (cell_width + gap))
            row = int((y - grid_rect.top()) / (cell_height + gap))
            return (
                min(max(row, 0), self._nrows - 1),
                min(max(col, 0), self._ncols - 1),
            )
        for row in range(self._nrows):
            for col in range(self._ncols):
                if self.cell_rect((row, col)).contains(pos):
                    return row, col
        return None

    def _region_at(self, pos: QtCore.QPoint) -> _GridSpecRegionInfo | None:
        for region in reversed(self._regions):
            if self.span_rect(region.span).contains(pos):
                return region
        return None

    def _selected_region(self) -> _GridSpecRegionInfo | None:
        for region in self._regions:
            if region.region_id == self._selected_region_id:
                return region
        return None

    def _handle_at(
        self, pos: QtCore.QPoint
    ) -> typing.Literal["nw", "ne", "sw", "se"] | None:
        region = self._selected_region()
        if (
            region is None
            or not region.valid
            or not self._span_within_grid(region.span)
        ):
            return None
        for handle, rect in self._handle_rects(self.span_rect(region.span), hit=True):
            if rect.contains(pos):
                return handle
        return None

    def _handle_rects(
        self, rect: QtCore.QRect, *, hit: bool = False
    ) -> tuple[tuple[typing.Literal["nw", "ne", "sw", "se"], QtCore.QRect], ...]:
        size = self._HANDLE_HIT_SIZE if hit else self._HANDLE_SIZE
        half = size // 2

        def handle_rect(point: QtCore.QPoint) -> QtCore.QRect:
            return QtCore.QRect(
                point.x() - half,
                point.y() - half,
                size,
                size,
            )

        return (
            ("nw", handle_rect(rect.topLeft())),
            ("ne", handle_rect(rect.topRight())),
            ("sw", handle_rect(rect.bottomLeft())),
            ("se", handle_rect(rect.bottomRight())),
        )

    def _draw_resize_handles(
        self,
        painter: QtGui.QPainter,
        rect: QtCore.QRect,
        color: QtGui.QColor,
    ) -> None:
        handle_fill = self.palette().color(QtGui.QPalette.ColorRole.Base)
        painter.setPen(QtGui.QPen(color, 1.3))
        painter.setBrush(QtGui.QBrush(handle_fill))
        for _handle, handle_rect in self._handle_rects(rect):
            painter.drawRect(handle_rect)

    def _update_hover_cursor(self, pos: QtCore.QPoint) -> None:
        handle = self._handle_at(pos)
        if handle in {"nw", "se"}:
            self.setCursor(QtCore.Qt.CursorShape.SizeFDiagCursor)
        elif handle in {"ne", "sw"}:
            self.setCursor(QtCore.Qt.CursorShape.SizeBDiagCursor)
        else:
            self.unsetCursor()

    def _active_preview_span(self) -> FigureGridSpecSpanState | None:
        if self._drag_mode == "resize":
            return self._resize_preview_span
        if self._drag_mode == "create" and self._drag_origin is not None:
            current = self._drag_current
            if current is not None:
                return self._span_from_cells(self._drag_origin, current)
        return None

    def _reset_drag(self) -> None:
        self._drag_mode = "none"
        self._drag_origin = None
        self._drag_current = None
        self._drag_region_id = ""
        self._drag_moved = False
        self._resize_handle = None
        self._resize_original_span = None
        self._resize_preview_span = None
        self._drag_creation_kind = self._creation_kind

    @staticmethod
    def _span_from_cells(
        start: tuple[int, int], end: tuple[int, int]
    ) -> FigureGridSpecSpanState:
        row0, col0 = start
        row1, col1 = end
        return FigureGridSpecSpanState(
            row_start=min(row0, row1),
            row_stop=max(row0, row1) + 1,
            col_start=min(col0, col1),
            col_stop=max(col0, col1) + 1,
        )

    def _span_from_resize(
        self,
        original: FigureGridSpecSpanState,
        handle: typing.Literal["nw", "ne", "sw", "se"] | None,
        cell: tuple[int, int],
    ) -> FigureGridSpecSpanState:
        row, col = cell
        row_start = original.row_start
        row_stop = original.row_stop
        col_start = original.col_start
        col_stop = original.col_stop
        if handle is None:
            return original
        if "n" in handle:
            row_start = min(max(row, 0), row_stop - 1)
        elif "s" in handle:
            row_stop = max(min(row + 1, self._nrows), row_start + 1)
        if "w" in handle:
            col_start = min(max(col, 0), col_stop - 1)
        elif "e" in handle:
            col_stop = max(min(col + 1, self._ncols), col_start + 1)
        return FigureGridSpecSpanState(
            row_start=row_start,
            row_stop=row_stop,
            col_start=col_start,
            col_stop=col_stop,
        )

    def _span_within_grid(self, span: FigureGridSpecSpanState) -> bool:
        return (
            0 <= span.row_start < span.row_stop <= self._nrows
            and 0 <= span.col_start < span.col_stop <= self._ncols
        )
