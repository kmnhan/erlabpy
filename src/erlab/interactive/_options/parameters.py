"""New pyqtgraph parameter types for user customization options.

This module defines custom parameter types for use in the options dialog of the
ImageTool. It includes a color list parameter and a custom colormap parameter.
"""

import pyqtgraph as pg
import pyqtgraph.parametertree
from qtpy import QtCore, QtGui, QtWidgets

import erlab


class _SimpleColorButton(QtWidgets.QPushButton):
    """A simple color button for the color list widget."""

    def __init__(self, parent=None, color=(128, 128, 128), padding=0):
        super().__init__(parent)
        self._color = pg.mkColor(color)
        self.padding = (
            (padding, padding, -padding, -padding)
            if isinstance(padding, int | float)
            else padding
        )
        self.setFixedSize(18, 18)
        self._menu = QtWidgets.QMenu(self)
        self._remove_action = self._menu.addAction("Remove")
        self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_menu)
        self.setFlat(True)

    def paintEvent(self, ev):
        super().paintEvent(ev)
        p = QtGui.QPainter(self)
        rect = self.rect().adjusted(*self.padding)
        p.setBrush(QtGui.QBrush(QtGui.QColor("white")))
        p.drawRect(rect)
        p.setBrush(QtGui.QBrush(QtCore.Qt.BrushStyle.DiagCrossPattern))
        p.drawRect(rect)
        p.setBrush(QtGui.QBrush(self._color))
        p.drawRect(rect)
        p.end()

    @QtCore.Slot(QtCore.QPoint)
    def _show_menu(self, position: QtCore.QPoint) -> None:
        self._menu.popup(self.mapToGlobal(position))


class ColorListWidget(QtWidgets.QWidget):
    """Widget for editing a list of colors.

    Displays a horizontal list of color buttons with an "Add" button at the end. Each
    color button can be clicked to edit the color, and right-clicked to remove it from
    the list. The widget emits a signal when the list of colors changes.
    """

    sigColorChanged = QtCore.Signal(list)

    def __init__(
        self,
        colors: list[str | QtGui.QColor] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        if colors is None:
            colors = []
        self.colors: list[QtGui.QColor] = [pg.mkColor(c) for c in colors]
        self.color_buttons: list[_SimpleColorButton] = []

        self.layout_ = QtWidgets.QHBoxLayout(self)
        self.layout_.setContentsMargins(0, 0, 0, 0)
        self.layout_.setSpacing(2)

        self.layout_.addStretch()
        self.add_button = erlab.interactive.utils.IconButton(
            QtGui.QIcon.fromTheme("list-add"), on_fallback="ph.plus"
        )
        self.add_button.setFlat(True)
        self.add_button.setFixedSize(20, 20)
        self.add_button.setToolTip("Add Color")
        self.add_button.clicked.connect(self.add_color)
        self.layout_.addWidget(self.add_button)

        self._refresh()

    def _update_buttons(self) -> None:
        """Update the color buttons in the layout to match the current colors."""
        # Remove old buttons
        for btn in self.color_buttons:
            self.layout_.removeWidget(btn)
            btn.deleteLater()
        self.color_buttons = []
        # Add new buttons
        for idx, color in enumerate(self.colors):
            btn = _SimpleColorButton(self, color=color)
            btn.clicked.connect(lambda _, i=idx: self.edit_color(i))
            btn._remove_action.triggered.connect(lambda _, i=idx: self.remove_color(i))
            self.layout_.insertWidget(idx, btn)
            self.color_buttons.append(btn)

    def _refresh(self) -> None:
        self._update_buttons()
        self.sigColorChanged.emit(self.colors)

    def add_color(self) -> None:
        """Open a color dialog to add a new color to the list."""
        col = QtWidgets.QColorDialog.getColor()
        if col.isValid():
            self.colors.append(col)
            self._refresh()

    def edit_color(self, idx: int) -> None:
        """Open a color dialog to edit the color at the given index."""
        col = QtWidgets.QColorDialog.getColor(self.colors[idx])
        if col.isValid():
            self.colors[idx] = col
            self._refresh()

    def remove_color(self, idx: int) -> None:
        """Remove the color at the given index from the list."""
        self.colors.pop(idx)
        self._refresh()

    def set_colors(self, colors: list[str | QtGui.QColor]) -> None:
        """Set a new list of colors."""
        self.colors = [pg.mkColor(c) for c in colors]
        self._refresh()

    def get_colors(self) -> list[str]:
        """Get the current list of colors as strings."""
        return [c.name() for c in self.colors]


class ColorListParameterItem(
    pyqtgraph.parametertree.parameterTypes.WidgetParameterItem
):
    def makeWidget(self) -> ColorListWidget:
        w = ColorListWidget()
        w.sigChanged = w.sigColorChanged  # type: ignore[attr-defined]
        w.value = w.get_colors  # type: ignore[attr-defined]
        w.setValue = w.set_colors  # type: ignore[attr-defined]
        self.hideWidget = False
        return w


class ColorListParameter(pyqtgraph.parametertree.parameterTypes.SimpleParameter):
    itemClass = ColorListParameterItem

    def __init__(self, **opts):
        opts.setdefault("type", "colorlist")
        super().__init__(**opts)

    def saveState(self, filter=None):  # noqa: A002
        state = super().saveState(filter)
        state["value"] = [c.getRgb() for c in self.value()]
        return state


class _CustomColorMapParameterItem(
    pyqtgraph.parametertree.parameterTypes.WidgetParameterItem
):
    """Parameter type that displays a ColorMapComboBox."""

    def __init__(self, param, depth) -> None:
        self.targetValue = None
        super().__init__(param, depth)

    def makeWidget(self) -> erlab.interactive.colors.ColorMapComboBox:
        # Adapted from ListParameterItem
        w = erlab.interactive.colors.ColorMapComboBox()
        w.setMaximumHeight(20)
        w.sigChanged = w.currentIndexChanged  # type: ignore[attr-defined]
        w.value = self.value  # type: ignore[attr-defined]
        w.setValue = self.setValue  # type: ignore[attr-defined]
        self.widget = w
        return w

    def value(self) -> str:
        return self.widget.currentText()

    def setValue(self, val) -> None:
        self.widget.setCurrentText(val)


class _CustomColorMapParameter(pyqtgraph.parametertree.parameterTypes.SimpleParameter):
    itemClass = _CustomColorMapParameterItem
