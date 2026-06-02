"""New pyqtgraph parameter types for user customization options.

This module defines custom parameter types for use in the options dialog of the
ImageTool. It includes list-style parameters and a custom colormap parameter.
"""

import pyqtgraph as pg
import pyqtgraph.parametertree
from qtpy import QtCore, QtGui, QtWidgets

import erlab
import erlab.interactive._stylesheets

_STYLESHEET_AVAILABLE_ROLE = QtCore.Qt.ItemDataRole.UserRole + 1
_STYLESHEET_NAME_ROLE = QtCore.Qt.ItemDataRole.UserRole + 2


def _stylesheet_names(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, list | tuple):
        value = [value]
    seen: set[str] = set()
    names: list[str] = []
    for item in value:
        name = str(item).strip()
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


class _StylesheetComboBox(QtWidgets.QComboBox):
    sigPopupAboutToShow = QtCore.Signal()

    def showPopup(self) -> None:
        self.sigPopupAboutToShow.emit()
        super().showPopup()


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


class StylesheetListWidget(QtWidgets.QWidget):
    """Widget for editing an ordered list of Matplotlib stylesheets."""

    sigStylesheetsChanged = QtCore.Signal(list)

    def __init__(
        self,
        stylesheets: list[str] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.stylesheets = _stylesheet_names(stylesheets)

        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(4)
        layout.setVerticalSpacing(3)

        self.add_combo = _StylesheetComboBox(self)
        self.add_combo.setObjectName("matplotlibStylesheetCombo")
        self.add_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.add_combo.setMinimumContentsLength(18)
        self.add_combo.setToolTip("Available Matplotlib stylesheet to add.")
        self.add_combo.sigPopupAboutToShow.connect(self._load_available_stylesheets)
        layout.addWidget(self.add_combo, 0, 0)

        self.add_button = QtWidgets.QToolButton(self)
        self.add_button.setObjectName("matplotlibStylesheetAddButton")
        self.add_button.setText("Add")
        self.add_button.setToolTip("Add the selected stylesheet.")
        self.add_button.clicked.connect(self.add_stylesheet)
        layout.addWidget(self.add_button, 0, 1)

        self.list_widget = QtWidgets.QListWidget(self)
        self.list_widget.setObjectName("matplotlibStylesheetList")
        self.list_widget.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.list_widget.setFixedHeight(88)
        self.list_widget.currentRowChanged.connect(self._update_button_state)
        layout.addWidget(self.list_widget, 1, 0, 1, 2)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(3)
        self.remove_button = QtWidgets.QToolButton(self)
        self.remove_button.setObjectName("matplotlibStylesheetRemoveButton")
        self.remove_button.setText("Remove")
        self.remove_button.setToolTip("Remove the selected stylesheet.")
        self.remove_button.clicked.connect(self.remove_selected_stylesheet)
        button_layout.addWidget(self.remove_button)

        self.up_button = QtWidgets.QToolButton(self)
        self.up_button.setObjectName("matplotlibStylesheetMoveUpButton")
        self.up_button.setText("Up")
        self.up_button.setToolTip("Move the selected stylesheet earlier.")
        self.up_button.clicked.connect(lambda: self.move_selected_stylesheet(-1))
        button_layout.addWidget(self.up_button)

        self.down_button = QtWidgets.QToolButton(self)
        self.down_button.setObjectName("matplotlibStylesheetMoveDownButton")
        self.down_button.setText("Down")
        self.down_button.setToolTip("Move the selected stylesheet later.")
        self.down_button.clicked.connect(lambda: self.move_selected_stylesheet(1))
        button_layout.addWidget(self.down_button)
        button_layout.addStretch(1)
        layout.addLayout(button_layout, 2, 0, 1, 2)

        self._refresh()

    def _refresh(self) -> None:
        current_name = self.current_stylesheet()
        self._refresh_list()
        self._refresh_add_combo()
        if current_name in self.stylesheets:
            self.list_widget.setCurrentRow(self.stylesheets.index(current_name))
        elif self.stylesheets:
            self.list_widget.setCurrentRow(0)
        self._update_button_state()
        self.sigStylesheetsChanged.emit(self.stylesheets)

    def _refresh_list(self) -> None:
        available = erlab.interactive._stylesheets.available_stylesheets(
            self.stylesheets
        )
        self.list_widget.clear()
        for name in self.stylesheets:
            is_available = name in available
            text = name if is_available else f"{name} (unavailable)"
            item = QtWidgets.QListWidgetItem(text)
            item.setData(_STYLESHEET_NAME_ROLE, name)
            item.setData(_STYLESHEET_AVAILABLE_ROLE, is_available)
            if is_available:
                item.setToolTip("This stylesheet is available in this environment.")
            else:
                item.setForeground(QtGui.QBrush(QtGui.QColor("#a65f00")))
                item.setToolTip(
                    "This saved stylesheet is unavailable here and will be skipped."
                )
            self.list_widget.addItem(item)

    def _refresh_add_combo(self) -> None:
        current = self.add_combo.currentText()
        selected = set(self.stylesheets)
        choices = [
            name
            for name in erlab.interactive._stylesheets.sorted_available_stylesheets()
            if name not in selected
        ]
        self.add_combo.blockSignals(True)
        try:
            self.add_combo.clear()
            self.add_combo.addItems(choices)
            if current in choices:
                self.add_combo.setCurrentText(current)
        finally:
            self.add_combo.blockSignals(False)
        self.add_button.setEnabled(bool(choices))

    @QtCore.Slot()
    def _load_available_stylesheets(self) -> None:
        current = self.add_combo.currentText()
        erlab.interactive._stylesheets.load_erlab_plotting_stylesheets()
        self._refresh_list()
        self._refresh_add_combo()
        if current:
            self.add_combo.setCurrentText(current)

    def _update_button_state(self, *_args) -> None:
        row = self.list_widget.currentRow()
        has_selection = 0 <= row < len(self.stylesheets)
        self.remove_button.setEnabled(has_selection)
        self.up_button.setEnabled(has_selection and row > 0)
        self.down_button.setEnabled(has_selection and row < len(self.stylesheets) - 1)

    def current_stylesheet(self) -> str | None:
        item = self.list_widget.currentItem()
        if item is None:
            return None
        return item.data(_STYLESHEET_NAME_ROLE)

    def add_stylesheet(self) -> None:
        name = self.add_combo.currentText().strip()
        if not name or name in self.stylesheets:
            return
        self.stylesheets.append(name)
        self._refresh()
        self.list_widget.setCurrentRow(len(self.stylesheets) - 1)

    def remove_selected_stylesheet(self) -> None:
        row = self.list_widget.currentRow()
        if not 0 <= row < len(self.stylesheets):
            return
        self.stylesheets.pop(row)
        self._refresh()
        if self.stylesheets:
            self.list_widget.setCurrentRow(min(row, len(self.stylesheets) - 1))

    def move_selected_stylesheet(self, offset: int) -> None:
        row = self.list_widget.currentRow()
        new_row = row + offset
        if not 0 <= row < len(self.stylesheets) or not 0 <= new_row < len(
            self.stylesheets
        ):
            return
        self.stylesheets.insert(new_row, self.stylesheets.pop(row))
        self._refresh()
        self.list_widget.setCurrentRow(new_row)

    def set_stylesheets(self, stylesheets: list[str]) -> None:
        self.stylesheets = _stylesheet_names(stylesheets)
        self._refresh()

    def get_stylesheets(self) -> list[str]:
        return list(self.stylesheets)


class StylesheetListParameterItem(
    pyqtgraph.parametertree.parameterTypes.WidgetParameterItem
):
    def makeWidget(self) -> StylesheetListWidget:
        w = StylesheetListWidget()
        w.sigChanged = w.sigStylesheetsChanged  # type: ignore[attr-defined]
        w.value = w.get_stylesheets  # type: ignore[attr-defined]
        w.setValue = w.set_stylesheets  # type: ignore[attr-defined]
        self.hideWidget = False
        return w


class StylesheetListParameter(pyqtgraph.parametertree.parameterTypes.SimpleParameter):
    itemClass = StylesheetListParameterItem

    def __init__(self, **opts):
        opts.setdefault("type", "matplotlib_stylesheets")
        super().__init__(**opts)


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
