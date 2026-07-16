"""Matplotlib-aware Qt color editors used by Figure Composer."""

from __future__ import annotations

import ast
import typing

import matplotlib.colors
from qtpy import QtCore, QtGui, QtWidgets

import erlab.interactive.utils
from erlab.interactive._figurecomposer._defaults import _figure_style_context


def _qcolor_to_mpl_color_text(color: QtGui.QColor) -> str:
    if color.alpha() == 255:
        return color.name(QtGui.QColor.NameFormat.HexRgb)
    return f"#{color.red():02x}{color.green():02x}{color.blue():02x}{color.alpha():02x}"


def _qcolor_from_mpl_color_value(color_value: object) -> QtGui.QColor | None:
    try:
        with _figure_style_context():
            red, green, blue, alpha = matplotlib.colors.to_rgba(
                typing.cast("typing.Any", color_value)
            )
    except (TypeError, ValueError):
        return None
    color = QtGui.QColor.fromRgbF(
        float(red),
        float(green),
        float(blue),
        float(alpha),
    )
    return color if color.isValid() else None


def _qcolor_from_mpl_color_text(text: str) -> QtGui.QColor | None:
    stripped = text.strip()
    if not stripped:
        return None
    color_value: object = stripped
    if stripped[0] in "[(":
        try:
            color_value = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            color_value = stripped
        if isinstance(color_value, list):
            color_value = tuple(color_value)
    return _qcolor_from_mpl_color_value(color_value)


def _default_mpl_line_color(index: int = 0) -> QtGui.QColor:
    with _figure_style_context():
        cycle_colors = matplotlib.rcParams["axes.prop_cycle"].by_key().get("color", ())
        color_value = (
            cycle_colors[index % len(cycle_colors)]
            if cycle_colors
            else matplotlib.rcParams["lines.color"]
        )
    color = _qcolor_from_mpl_color_value(color_value)
    if color is not None:
        return color
    return QtGui.QColor.fromRgbF(0.0, 0.0, 0.0, 1.0)


def _inherited_qcolor(
    color: object | None,
    *,
    default_index: int = 0,
) -> QtGui.QColor:
    parsed = None if color is None else _qcolor_from_mpl_color_value(color)
    return _default_mpl_line_color(default_index) if parsed is None else parsed


def _top_level_comma_parts(text: str) -> tuple[str, ...]:
    parts: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    escaped = False
    for index, char in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(depth - 1, 0)
        elif char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return tuple(part for part in parts if part)


def _color_tuple_from_text(text: str) -> tuple[str, ...]:
    stripped = text.strip()
    if not stripped:
        return ()
    if stripped[0] == "[":
        try:
            value = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            pass
        else:
            if isinstance(value, list | tuple):
                return tuple(str(item) for item in value)
    return _top_level_comma_parts(stripped)


def _format_color_tuple(colors: tuple[str, ...]) -> str:
    return ", ".join(colors)


class _ColorLineEditWidget(QtWidgets.QWidget):
    """Color text input with a lifetime-safe color picker."""

    editingFinished = QtCore.Signal()

    def __init__(
        self,
        text: str = "",
        parent: QtWidgets.QWidget | None = None,
        *,
        inherited_color: object | None = None,
        inherited_color_index: int = 0,
    ) -> None:
        super().__init__(parent)
        self._syncing = False
        self._inherited_color = _inherited_qcolor(
            inherited_color,
            default_index=inherited_color_index,
        )
        self.line_edit = QtWidgets.QLineEdit(text, self)
        self.color_button = _ColorPickerButton(self, color=self._button_color())
        self.color_button.setToolTip("Choose a color.")
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.line_edit, 1)
        layout.addWidget(self.color_button)
        self.line_edit.editingFinished.connect(self._text_finished)
        self.color_button.colorSelected.connect(self._button_color_changed)
        self._update_button_from_text()

    def text(self) -> str:
        return self.line_edit.text()

    def setText(self, text: str) -> None:
        with QtCore.QSignalBlocker(self.line_edit):
            self.line_edit.setText(text)
            self.line_edit.setModified(False)
        self._update_button_from_text()

    def setInheritedColor(
        self, color: object | None, *, default_index: int = 0
    ) -> None:
        self._inherited_color = _inherited_qcolor(color, default_index=default_index)
        if not self.line_edit.text().strip():
            self._update_button_from_text()

    def setPlaceholderText(self, text: str) -> None:
        self.line_edit.setPlaceholderText(text)

    def setToolTip(self, text: str | None) -> None:
        super().setToolTip(text)
        self.line_edit.setToolTip(text)
        tooltip = "" if text is None else text
        self.color_button.setToolTip(f"{tooltip}\nChoose a color.")

    def setLineEditObjectName(self, object_name: str) -> None:
        self.line_edit.setObjectName(object_name)

    def setColorButtonObjectName(self, object_name: str) -> None:
        self.color_button.setObjectName(object_name)

    def setModified(self, modified: bool) -> None:
        self.line_edit.setModified(modified)

    def isModified(self) -> bool:
        return self.line_edit.isModified()

    def _text_finished(self) -> None:
        if not self._syncing:
            self._update_button_from_text()
        self.editingFinished.emit()

    def _button_color_changed(self, color: object) -> None:
        if self._syncing:
            return
        if not isinstance(color, QtGui.QColor) or not color.isValid():
            return
        self._syncing = True
        try:
            self.line_edit.setText(_qcolor_to_mpl_color_text(color))
            self.line_edit.setModified(True)
        finally:
            self._syncing = False
        self.editingFinished.emit()

    def _update_button_from_text(self) -> None:
        color = self._button_color()
        if color is None:
            return
        with QtCore.QSignalBlocker(self.color_button):
            self.color_button.setColor(color)

    def _button_color(self) -> QtGui.QColor | None:
        if not self.line_edit.text().strip():
            return QtGui.QColor(self._inherited_color)
        return _qcolor_from_mpl_color_text(self.line_edit.text())


class _ColorPickerButton(QtWidgets.QPushButton):
    """Push button that opens a color dialog without persistent dialog ownership."""

    colorSelected = QtCore.Signal(QtGui.QColor)

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        color: QtGui.QColor | None = None,
        show_alpha_channel: bool = True,
    ) -> None:
        super().__init__(parent)
        self._color = (
            QtGui.QColor(color)
            if color is not None and color.isValid()
            else _default_mpl_line_color()
        )
        self._show_alpha_channel = show_alpha_channel
        self._dialog_open = False
        self.setMinimumSize(22, 18)
        self.clicked.connect(self._choose_color)

    def color(self) -> QtGui.QColor:
        return QtGui.QColor(self._color)

    def setColor(self, color: object) -> None:
        if not isinstance(color, QtGui.QColor) or not color.isValid():
            return
        self._color = QtGui.QColor(color)
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent | None) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        try:
            rect = self.rect().adjusted(5, 4, -5, -4)
            painter.fillRect(rect, QtCore.Qt.GlobalColor.white)
            painter.fillRect(
                rect,
                QtGui.QBrush(QtCore.Qt.BrushStyle.DiagCrossPattern),
            )
            painter.fillRect(rect, self._color)
            painter.setPen(self.palette().color(QtGui.QPalette.ColorRole.Mid))
            painter.drawRect(rect)
        finally:
            painter.end()

    def _choose_color(self) -> None:
        if self._dialog_open or not erlab.interactive.utils.qt_is_valid(self):
            return
        self._dialog_open = True
        try:
            parent = self.window()
            if not erlab.interactive.utils.qt_is_valid(parent):
                parent = None
            if self._show_alpha_channel:
                color = QtWidgets.QColorDialog.getColor(
                    self._color,
                    parent,
                    "Choose Color",
                    QtWidgets.QColorDialog.ColorDialogOption.ShowAlphaChannel,
                )
            else:
                color = QtWidgets.QColorDialog.getColor(
                    self._color,
                    parent,
                    "Choose Color",
                )
        finally:
            if erlab.interactive.utils.qt_is_valid(self):
                self._dialog_open = False
        if not erlab.interactive.utils.qt_is_valid(self):
            return
        if not color.isValid():
            return
        self.setColor(color)
        self.colorSelected.emit(QtGui.QColor(color))


class _ColorListEditorWidget(QtWidgets.QWidget):
    """Synchronized comma-separated color text and per-color editors."""

    colorsChanged = QtCore.Signal(object)

    def __init__(
        self,
        colors: tuple[str, ...] = (),
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._syncing = False
        self._pending_row_rebuild_colors: tuple[str, ...] | None = None
        self._row_rebuild_pending = False
        self.main_edit = QtWidgets.QLineEdit(self)
        self._rows_widget = QtWidgets.QWidget(self)
        self._rows_layout = QtWidgets.QVBoxLayout(self._rows_widget)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(4)
        self._add_button = QtWidgets.QToolButton(self)
        self._add_button.setText("Add")
        self._add_button.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly
        )
        self._add_button.setObjectName("figureComposerColorListAddButton")
        self._add_button.setToolTip("Add another color entry.")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.main_edit)
        layout.addWidget(self._rows_widget)
        layout.addWidget(self._add_button, 0, QtCore.Qt.AlignmentFlag.AlignLeft)

        self.main_edit.editingFinished.connect(self._main_text_finished)
        self._add_button.clicked.connect(self._add_color)
        self.setColors(colors)

    def colors(self) -> tuple[str, ...]:
        return _color_tuple_from_text(self.main_edit.text())

    def setColors(self, colors: tuple[str, ...]) -> None:
        self._syncing = True
        try:
            with QtCore.QSignalBlocker(self.main_edit):
                self.main_edit.setText(_format_color_tuple(colors))
                self.main_edit.setModified(False)
            self._rebuild_rows(colors)
        finally:
            self._syncing = False

    def setMainEditObjectName(self, object_name: str) -> None:
        self.main_edit.setObjectName(object_name)

    def setToolTip(self, text: str | None) -> None:
        super().setToolTip(text)
        self.main_edit.setToolTip(text)
        self._rows_widget.setToolTip(text)
        for edit in self._row_editors():
            edit.setToolTip(text)

    def setMixedPlaceholder(self, text: str) -> None:
        self.main_edit.setPlaceholderText(text)
        self.main_edit.setProperty("batch_mixed", True)
        self.main_edit.setModified(False)
        self.setColors(())

    def batchUnchanged(self) -> bool:
        return bool(self.main_edit.property("batch_mixed")) and not (
            self.main_edit.isModified()
        )

    def _main_text_finished(self) -> None:
        if self._syncing:
            return
        colors = _color_tuple_from_text(self.main_edit.text())
        self._syncing = True
        try:
            self._set_row_colors_from_text(colors)
        finally:
            self._syncing = False
        self.colorsChanged.emit(colors)

    def _add_color(self) -> None:
        colors = (*self.colors(), "")
        self._set_colors_from_structure_change(colors)

    def _remove_color(self, index: int) -> None:
        colors = tuple(color for i, color in enumerate(self.colors()) if i != index)
        self._set_colors_from_structure_change(colors)

    def _row_color_changed(self) -> None:
        colors = tuple(edit.text().strip() for edit in self._row_editors())
        self._set_colors_from_rows(colors)

    def _set_colors_from_rows(self, colors: tuple[str, ...]) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            with QtCore.QSignalBlocker(self.main_edit):
                self.main_edit.setText(_format_color_tuple(colors))
                self.main_edit.setModified(True)
        finally:
            self._syncing = False
        self.colorsChanged.emit(colors)

    def _set_colors_from_structure_change(self, colors: tuple[str, ...]) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            with QtCore.QSignalBlocker(self.main_edit):
                self.main_edit.setText(_format_color_tuple(colors))
                self.main_edit.setModified(True)
            self._rebuild_rows_when_safe(colors)
        finally:
            self._syncing = False
        self.colorsChanged.emit(colors)

    def _set_row_colors_from_text(self, colors: tuple[str, ...]) -> None:
        editors = self._row_editors()
        if len(editors) != len(colors):
            self._rebuild_rows_when_safe(colors)
            return
        for edit, color in zip(editors, colors, strict=True):
            edit.setText(color)

    def _rebuild_rows_when_safe(self, colors: tuple[str, ...]) -> None:
        if self._rows_have_focus():
            self._queue_rebuild_rows(colors)
            return
        self._pending_row_rebuild_colors = None
        self._rebuild_rows(colors)

    def _rows_have_focus(self) -> bool:
        focus_widget = QtWidgets.QApplication.focusWidget()
        return (
            focus_widget is not None
            and erlab.interactive.utils.qt_is_valid(focus_widget)
            and (
                focus_widget is self._rows_widget
                or self._rows_widget.isAncestorOf(focus_widget)
            )
        )

    def _queue_rebuild_rows(
        self, colors: tuple[str, ...], *, delay_ms: int = 0
    ) -> None:
        self._pending_row_rebuild_colors = colors
        if self._row_rebuild_pending:
            return
        self._row_rebuild_pending = True
        erlab.interactive.utils.single_shot(
            self, delay_ms, self._run_pending_row_rebuild
        )

    def _run_pending_row_rebuild(self) -> None:
        self._row_rebuild_pending = False
        colors = self._pending_row_rebuild_colors
        self._pending_row_rebuild_colors = None
        if colors is None:
            return
        if self._rows_have_focus():
            self._queue_rebuild_rows(colors, delay_ms=50)
            return
        self._rebuild_rows(colors)

    def _rebuild_rows(self, colors: tuple[str, ...]) -> None:
        while self._rows_layout.count():
            item = self._rows_layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for index, color in enumerate(colors):
            self._rows_layout.addWidget(self._make_row(index, color))

    def _row_editors(self) -> tuple[_ColorLineEditWidget, ...]:
        editors: list[_ColorLineEditWidget] = []
        for index in range(self._rows_layout.count()):
            item = self._rows_layout.itemAt(index)
            if item is None:
                continue
            row = item.widget()
            if row is None:
                continue
            edit = row.findChild(_ColorLineEditWidget)
            if edit is not None:
                editors.append(edit)
        return tuple(editors)

    def _make_row(self, index: int, color: str) -> QtWidgets.QWidget:
        row = QtWidgets.QWidget(self._rows_widget)
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        edit = _ColorLineEditWidget(
            color,
            row,
            inherited_color_index=index,
        )
        edit.setLineEditObjectName(f"figureComposerLineColorItemEdit_{index}")
        edit.setColorButtonObjectName(f"figureComposerLineColorButton_{index}")
        edit.setToolTip(self.toolTip())
        edit.editingFinished.connect(self._row_color_changed)
        remove_button = QtWidgets.QToolButton(row)
        remove_button.setText("Remove")
        remove_button.setObjectName(f"figureComposerLineColorRemoveButton_{index}")
        remove_button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
        remove_button.setToolTip("Remove this color entry.")
        remove_button.clicked.connect(
            lambda _checked=False, i=index: self._remove_color(i)
        )
        layout.addWidget(edit, 1)
        layout.addWidget(remove_button)
        return row
