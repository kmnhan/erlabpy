"""Qt widgets used by the private Figure Composer framework."""

from __future__ import annotations

import ast
import contextlib
import math
import typing
import weakref

# Matplotlib's Qt backend should see the qtpy-selected binding first.
# isort: off
from qtpy import QtCore, QtGui, QtWidgets

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.colors
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
    from collections.abc import Callable, Mapping, Sequence

    from erlab.interactive._figurecomposer._state import FigureSubplotsState


class _GridSpecRegionInfo(typing.NamedTuple):
    region_id: str
    kind: typing.Literal["axes", "grid"]
    span: FigureGridSpecSpanState
    label: str
    valid: bool = True


class _SelectorColors(typing.NamedTuple):
    face: QtGui.QColor
    panel: QtGui.QColor
    nested_face: QtGui.QColor
    hover_face: QtGui.QColor
    selection_face: QtGui.QColor
    border: QtGui.QColor
    selection: QtGui.QColor
    text: QtGui.QColor
    muted_text: QtGui.QColor
    selected_text: QtGui.QColor


class _GridSpecOutsideClickFilter(QtCore.QObject):
    """Hide edit affordances when the active GridSpec view is clicked away from."""

    def __init__(self, widget: QtWidgets.QWidget) -> None:
        super().__init__(widget)
        self._widget_ref = weakref.ref(widget)

    def eventFilter(
        self, watched: QtCore.QObject | None, event: QtCore.QEvent | None
    ) -> bool:
        widget = self._widget_ref()
        if widget is not None and erlab.interactive.utils.qt_is_valid(widget):
            typing.cast("_GridSpecViewWidget", widget)._handle_application_event(event)
        return super().eventFilter(watched, event)


_SELECTOR_CORNER_RADIUS = 6.0
_SELECTOR_BORDER_WIDTH = 1.0
_SELECTOR_SELECTED_BORDER_WIDTH = 1.6
_SELECTOR_MUTED_STATUS_COLOR = QtGui.QColor("#59636e")


def _blend_qcolors(
    base: QtGui.QColor,
    overlay: QtGui.QColor,
    ratio: float,
) -> QtGui.QColor:
    ratio = min(max(ratio, 0.0), 1.0)
    inverse = 1.0 - ratio
    return QtGui.QColor(
        int(base.red() * inverse + overlay.red() * ratio),
        int(base.green() * inverse + overlay.green() * ratio),
        int(base.blue() * inverse + overlay.blue() * ratio),
        int(base.alpha() * inverse + overlay.alpha() * ratio),
    )


def _selector_colors(palette: QtGui.QPalette) -> _SelectorColors:
    face = palette.color(QtGui.QPalette.ColorRole.Base)
    panel = palette.color(QtGui.QPalette.ColorRole.Window)
    alternate = palette.color(QtGui.QPalette.ColorRole.AlternateBase)
    border = palette.color(QtGui.QPalette.ColorRole.Mid)
    selection = palette.color(QtGui.QPalette.ColorRole.Highlight)
    text = palette.color(QtGui.QPalette.ColorRole.Text)
    nested_face = _blend_qcolors(face, alternate, 0.42)
    hover_face = _blend_qcolors(face, selection, 0.12)
    selection_face = _blend_qcolors(face, selection, 0.18)
    muted_text = _blend_qcolors(
        text,
        _SELECTOR_MUTED_STATUS_COLOR,
        0.55 if text.lightness() > panel.lightness() else 0.72,
    )
    return _SelectorColors(
        face=face,
        panel=panel,
        nested_face=nested_face,
        hover_face=hover_face,
        selection_face=selection_face,
        border=border,
        selection=selection,
        text=text,
        muted_text=muted_text,
        selected_text=selection,
    )


def _draw_selector_rect(
    painter: QtGui.QPainter,
    rect: QtCore.QRect,
    *,
    facecolor: QtGui.QColor,
    edgecolor: QtGui.QColor,
    linewidth: float = _SELECTOR_BORDER_WIDTH,
    radius: float = _SELECTOR_CORNER_RADIUS,
    draw_edge: bool = True,
) -> None:
    painter.save()
    rectf = QtCore.QRectF(rect)
    edge_width = linewidth if draw_edge else 0.0
    rectf.adjust(edge_width / 2, edge_width / 2, -edge_width / 2, -edge_width / 2)
    path = QtGui.QPainterPath()
    path.addRoundedRect(rectf, radius, radius)
    painter.fillPath(path, QtGui.QBrush(facecolor))
    if draw_edge:
        painter.setPen(QtGui.QPen(edgecolor, linewidth))
        painter.drawPath(path)
    painter.restore()


def _centered_rect(
    container: QtCore.QRect,
    size: QtCore.QSize,
    margin: int,
) -> QtCore.QRect:
    available = container.adjusted(margin, margin, -margin, -margin)
    if available.width() <= 0 or available.height() <= 0:
        return QtCore.QRect()
    width = min(size.width(), available.width())
    height = min(size.height(), available.height())
    rect = QtCore.QRect(0, 0, width, height)
    rect.moveCenter(available.center())
    return rect


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
            if isinstance(value, str):
                return (value,)
            if isinstance(value, list | tuple):
                return tuple(str(item) for item in value)
            raise TypeError("Expected a color string, list, or tuple literal")
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
    ) -> None:
        super().__init__(parent)
        self._color = (
            QtGui.QColor(color)
            if color is not None and color.isValid()
            else _default_mpl_line_color()
        )
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
            color = QtWidgets.QColorDialog.getColor(
                self._color,
                parent,
                "Choose Color",
                QtWidgets.QColorDialog.ColorDialogOption.ShowAlphaChannel,
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


class _StyledFigureCanvas(FigureCanvas):
    def draw(self, *args, **kwargs):
        with _figure_draw_context():
            return super().draw(*args, **kwargs)

    def print_figure(self, *args, **kwargs):
        with _figure_draw_context():
            return super().print_figure(*args, **kwargs)


_TOOLBAR_ICON_NAMES = {
    "home": "mdi6.home",
    "back": "mdi6.arrow-left",
    "forward": "mdi6.arrow-right",
    "move": "mdi6.arrow-all",
    "zoom_to_rect": "mdi6.magnify-plus",
    "subplots": "mdi6.view-grid-outline",
    "qt4_editor_options": "mdi6.tune",
    "filesave": "mdi6.export",
}


def _noop_toolbar_callback() -> None:
    return


class _FigureComposerNavigationToolbar(NavigationToolbar):
    """Navigation toolbar that routes composer actions through recipe edits."""

    def __init__(
        self,
        canvas: FigureCanvas,
        parent: QtWidgets.QWidget,
        *,
        export_callback: Callable[[], None],
        subplot_adjust_callback: Callable[[], None],
        axes_customize_callback: Callable[[], None],
    ) -> None:
        self._export_callback = export_callback
        self._subplot_adjust_callback = subplot_adjust_callback
        self._axes_customize_callback = axes_customize_callback
        super().__init__(canvas, parent)
        self.setObjectName("figureComposerNavigationToolbar")
        for action_id, action in self._actions.items():
            action.setObjectName(f"figureComposerToolbar_{action_id}")
        tooltips = {
            "save_figure": "Export the composer figure using recipe export settings.",
            "configure_subplots": (
                "Add or edit recipe steps for subplot spacing and layout engine."
            ),
            "edit_parameters": "Add or edit recipe steps for selected axes.",
        }
        for action_id, tooltip in tooltips.items():
            if action := self._actions.get(action_id):
                action.setToolTip(tooltip)

    def _icon(self, name: str) -> QtGui.QIcon:
        icon_key = name.removesuffix(".png").removesuffix("_large")
        icon_name = _TOOLBAR_ICON_NAMES.get(icon_key)
        if icon_name is None:
            return super()._icon(name)
        color = self.palette().color(QtGui.QPalette.ColorRole.ButtonText)
        return erlab.interactive.utils.qtawesome.icon(icon_name, color=color)

    def configure_subplots(self) -> None:
        self._subplot_adjust_callback()

    def edit_parameters(self) -> None:
        self._axes_customize_callback()

    def save_figure(self, *args: object) -> None:
        self._export_callback()

    def changeEvent(self, event: QtCore.QEvent | None) -> None:
        if event is not None and event.type() == QtCore.QEvent.Type.PaletteChange:
            erlab.interactive.utils.qtawesome.reset_cache()
            for callback_name, action in self._actions.items():
                icon_name = {
                    "zoom": "zoom_to_rect",
                    "pan": "move",
                    "configure_subplots": "subplots",
                    "edit_parameters": "qt4_editor_options",
                    "save_figure": "filesave",
                }.get(callback_name, callback_name)
                action.setIcon(self._icon(icon_name))
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
    ) -> None:
        super().__init__(None)
        self._closing_from_owner = False
        self._suppress_resize_signal = False
        self._resize_signal_pending = False
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
        self._suppress_resize_signal = True
        self._resize_signal_pending = False
        self._remove_event_filters()
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
        size = self._grid_content_size()
        return QtCore.QSize(
            max(self._MIN_WIDTH, size.width() + margin),
            max(self._MIN_HEIGHT, size.height() + margin),
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
        colors = _selector_colors(self.palette())
        for axis in _all_axes_for_shape(self._nrows, self._ncols):
            rect = self.cell_rect(axis)
            selected = axis in self._selected_axes
            hovered = axis == self._hovered_axis
            face = (
                colors.selection_face
                if selected
                else colors.hover_face
                if hovered
                else colors.face
            )
            edge = colors.selection if selected else colors.border
            _draw_selector_rect(
                painter,
                rect,
                facecolor=face,
                edgecolor=edge,
                linewidth=(
                    _SELECTOR_SELECTED_BORDER_WIDTH
                    if selected
                    else _SELECTOR_BORDER_WIDTH
                ),
            )
            painter.setPen(colors.selected_text if selected else colors.text)
            text = painter.fontMetrics().elidedText(
                self._axis_label(axis),
                QtCore.Qt.TextElideMode.ElideRight,
                max(0, rect.width() - 6),
            )
            painter.drawText(
                rect.adjusted(3, 1, -3, -1),
                QtCore.Qt.AlignmentFlag.AlignCenter,
                text,
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
        return _centered_rect(rect, self._grid_content_size(), self._GRID_MARGIN)

    def _grid_content_size(self) -> QtCore.QSize:
        return QtCore.QSize(
            self._ncols * self._CELL_WIDTH + max(self._ncols - 1, 0) * self._CELL_GAP,
            self._nrows * self._CELL_HEIGHT + max(self._nrows - 1, 0) * self._CELL_GAP,
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


class _GridSpecViewWidget(QtWidgets.QWidget):
    """Nested GridSpec view used for both editing and target-axis selection."""

    sigSelectionChanged = QtCore.Signal(object)
    sigRegionCreated = QtCore.Signal(object, str)
    sigRegionChanged = QtCore.Signal(str, object)
    sigRegionSelected = QtCore.Signal(str, str)
    sigNestedGridActivated = QtCore.Signal(str)

    _CELL_WIDTH = 46
    _CELL_HEIGHT = 28
    _CELL_GAP = 3
    _GRID_MARGIN = 4
    _NESTED_INSET = 4
    _NESTED_LABEL_HEIGHT = 16
    _MIN_WIDTH = 100
    _MIN_HEIGHT = 44
    _HANDLE_SIZE = 7
    _HANDLE_HIT_SIZE = 13

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        mode: typing.Literal["edit", "select"],
    ) -> None:
        super().__init__(parent)
        self._mode = mode
        self._root_grid = FigureGridSpecGridState(grid_id="root", label="Root")
        self._display_grid = self._root_grid
        self._regions: tuple[_GridSpecRegionInfo, ...] = ()
        self._labels: dict[str, str] = {}
        self._selected_region_id = ""
        self._selected_axes: frozenset[str] = frozenset()
        self._anchor_axis_id: str | None = None
        self._drag_mode: typing.Literal[
            "none", "create", "move", "resize", "select"
        ] = "none"
        self._drag_origin: QtCore.QPoint | None = None
        self._drag_origin_cell: tuple[int, int] | None = None
        self._drag_current_cell: tuple[int, int] | None = None
        self._drag_base: frozenset[str] = frozenset()
        self._drag_additive = False
        self._drag_region_id = ""
        self._drag_moved = False
        self._resize_handle: typing.Literal["nw", "ne", "sw", "se"] | None = None
        self._resize_original_span: FigureGridSpecSpanState | None = None
        self._resize_preview_span: FigureGridSpecSpanState | None = None
        self._creation_kind: typing.Literal["axes", "grid"] = "axes"
        self._drag_creation_kind: typing.Literal["axes", "grid"] = "axes"
        self._hovered_axis_id: str | None = None
        self._region_handles_visible = False
        self._application: QtWidgets.QApplication | None = None
        self._application_event_filter_installed = False
        self._outside_click_filter = (
            _GridSpecOutsideClickFilter(self) if mode == "edit" else None
        )
        self.setObjectName(
            "figureComposerGridSpecLayoutWidget"
            if mode == "edit"
            else "figureComposerGridSpecAxesSelector"
        )
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.setMouseTracking(True)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Maximum,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        if mode == "edit":
            self.setToolTip(
                "Drag empty cells to create a region.\n"
                "Drag a region to move it.\n"
                "Select a region, then drag a corner handle to resize it.\n"
                "Double-click a nested grid region to edit that grid."
            )
        else:
            self.setToolTip(
                "Click an axis to target it.\n"
                "Shift-click selects a contiguous display range.\n"
                "Ctrl/Cmd-click toggles one axis; drag selects visible axes."
            )

    def sizeHint(self) -> QtCore.QSize:
        grid = self._display_grid if self._mode == "edit" else self._root_grid
        size = self._grid_size_hint(grid)
        return QtCore.QSize(
            max(self._MIN_WIDTH, size.width() + 2 * self._GRID_MARGIN),
            max(self._MIN_HEIGHT, size.height() + 2 * self._GRID_MARGIN),
        )

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(self._MIN_WIDTH, self._MIN_HEIGHT)

    def set_edit_grid(
        self,
        grid: FigureGridSpecGridState,
        regions: Sequence[_GridSpecRegionInfo],
        labels: Mapping[str, str] | None = None,
    ) -> None:
        self._root_grid = grid
        self._display_grid = grid
        self._regions = tuple(regions)
        self._labels = dict(labels or {})
        if self._selected_region_id and not any(
            region.region_id == self._selected_region_id for region in self._regions
        ):
            self._selected_region_id = ""
        self.updateGeometry()
        self.update()

    def set_layout(
        self,
        root_grid: FigureGridSpecGridState,
        labels: Mapping[str, str] | None = None,
    ) -> None:
        self._root_grid = root_grid
        self._display_grid = root_grid
        self._labels = dict(labels or {})
        valid_axes = set(self.axes_ids())
        self._selected_axes = frozenset(
            axes_id for axes_id in self._selected_axes if axes_id in valid_axes
        )
        if self._anchor_axis_id not in valid_axes:
            self._anchor_axis_id = next(iter(self._selected_axes), None)
        self.updateGeometry()
        self.update()

    def showEvent(self, event: QtGui.QShowEvent | None) -> None:
        super().showEvent(event)
        self._install_application_event_filter()

    def hideEvent(self, event: QtGui.QHideEvent | None) -> None:
        self._remove_application_event_filter()
        self._set_region_handles_visible(False)
        erlab.interactive.utils.set_widget_cursor(self, None)
        super().hideEvent(event)

    def set_creation_kind(self, kind: typing.Literal["axes", "grid"]) -> None:
        self._creation_kind = kind

    def selected_region_id(self) -> str:
        return self._selected_region_id

    def set_selected_region(self, region_id: str) -> None:
        previous_region_id = self._selected_region_id
        if region_id and not any(
            region.region_id == region_id for region in self._regions
        ):
            self._selected_region_id = ""
        else:
            self._selected_region_id = region_id
        if not self._selected_region_id:
            self._region_handles_visible = False
        elif self._selected_region_id != previous_region_id:
            self._region_handles_visible = True
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
        grid = self._display_grid if self._mode == "edit" else self._root_grid
        self._draw_grid(painter, grid, self._grid_rect(), depth=0)
        preview_span = self._active_preview_span()
        if self._mode == "edit" and preview_span is not None:
            self._draw_preview_span(painter, preview_span)

    def mousePressEvent(self, event: QtGui.QMouseEvent | None) -> None:
        if event is None or event.button() != QtCore.Qt.MouseButton.LeftButton:
            if event is not None:
                super().mousePressEvent(event)
            return
        if self._mode == "edit":
            self._edit_mouse_press(event)
            return
        self._select_mouse_press(event)

    def _select_mouse_press(self, event: QtGui.QMouseEvent) -> None:
        axis_id = self._axis_at(event.position().toPoint())
        if axis_id is None:
            return
        modifiers = event.modifiers()
        additive = self._has_toggle_modifier(modifiers)
        shift = bool(modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier)
        self._drag_mode = "select"
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

    def _edit_mouse_press(self, event: QtGui.QMouseEvent) -> None:
        pos = event.position().toPoint()
        handle = self._handle_at(pos)
        if handle is not None:
            region = self._selected_region()
            if region is None:
                return
            self._set_region_handles_visible(True)
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
            self._reset_edit_drag()
            self._selected_region_id = region.region_id
            self._set_region_handles_visible(True)
            self.sigRegionSelected.emit(region.region_id, region.kind)
            cell = self._cell_at(pos)
            if cell is not None:
                self._drag_mode = "move"
                self._drag_origin_cell = cell
                self._drag_current_cell = cell
                self._drag_region_id = region.region_id
                self._resize_original_span = region.span
                self._resize_preview_span = region.span
                self._drag_moved = False
            self._update_hover_cursor(pos)
            event.accept()
            return
        cell = self._cell_at(pos)
        if cell is None:
            return
        self._set_region_handles_visible(False)
        self._drag_mode = "create"
        self._drag_origin_cell = cell
        self._drag_current_cell = cell
        self._drag_moved = False
        self._drag_region_id = ""
        self._drag_creation_kind = self._creation_kind
        event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent | None) -> None:
        if event is None:
            return
        if self._mode == "edit":
            self._edit_mouse_move(event)
            return
        self._select_mouse_move(event)

    def _select_mouse_move(self, event: QtGui.QMouseEvent) -> None:
        axis_id = self._axis_at(event.position().toPoint())
        if axis_id != self._hovered_axis_id:
            self._hovered_axis_id = axis_id
            self.update()
        if (
            self._drag_mode == "select"
            and self._drag_origin is not None
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

    def _edit_mouse_move(self, event: QtGui.QMouseEvent) -> None:
        if self._drag_mode == "move" and (
            event.buttons() & QtCore.Qt.MouseButton.LeftButton
        ):
            current = self._cell_at(event.position().toPoint(), clamp_to_grid=True)
            if (
                current is not None
                and self._drag_origin_cell is not None
                and self._resize_original_span is not None
            ):
                span = self._span_from_move(
                    self._resize_original_span,
                    origin=self._drag_origin_cell,
                    current=current,
                )
                self._resize_preview_span = span
                self._drag_moved = span != self._resize_original_span
                self.update()
            event.accept()
            return
        if self._drag_mode == "resize" and (
            event.buttons() & QtCore.Qt.MouseButton.LeftButton
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
            self._drag_mode == "create"
            and self._drag_origin_cell is not None
            and event.buttons() & QtCore.Qt.MouseButton.LeftButton
        ):
            current = self._cell_at(event.position().toPoint(), clamp_to_grid=True)
            self._drag_current_cell = current
            self._drag_moved = current is not None and current != self._drag_origin_cell
            self.update()
            event.accept()
            return
        self._update_hover_cursor(event.position().toPoint())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent | None) -> None:
        if self._mode == "edit":
            self._edit_mouse_release(event)
            return
        self._drag_mode = "none"
        self._drag_origin = None
        self._drag_base = frozenset()
        self._drag_additive = False
        if event is not None:
            event.accept()

    def _edit_mouse_release(self, event: QtGui.QMouseEvent | None) -> None:
        if event is None or self._drag_mode == "none":
            return
        mode = self._drag_mode
        end_cell = self._cell_at(event.position().toPoint(), clamp_to_grid=True)
        origin = self._drag_origin_cell
        region_id = self._drag_region_id
        moved = self._drag_moved
        span = self._resize_preview_span
        creation_kind = self._drag_creation_kind
        self._reset_edit_drag()
        self.update()
        if mode in {"move", "resize"}:
            if moved and region_id and span is not None:
                self.sigRegionChanged.emit(region_id, span)
            event.accept()
            return
        if end_cell is not None and origin is not None:
            span = self._span_from_cells(origin, end_cell)
            self.sigRegionCreated.emit(span, creation_kind)
        event.accept()

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent | None) -> None:
        if event is None or self._mode != "edit":
            if event is not None:
                super().mouseDoubleClickEvent(event)
            return
        self._reset_edit_drag()
        region = self._region_at(event.position().toPoint())
        if region is not None and region.kind == "grid":
            self._selected_region_id = region.region_id
            self._set_region_handles_visible(True)
            self.sigRegionSelected.emit(region.region_id, region.kind)
            self.sigNestedGridActivated.emit(region.region_id)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def leaveEvent(self, event: QtCore.QEvent | None) -> None:
        self._hovered_axis_id = None
        if self._mode == "edit":
            erlab.interactive.utils.set_widget_cursor(self, None)
        self.update()
        if event is not None:
            super().leaveEvent(event)

    def _grid_rect(self) -> QtCore.QRect:
        rect = self.rect()
        if rect.isEmpty():
            rect = QtCore.QRect(QtCore.QPoint(0, 0), self.sizeHint())
        grid = self._display_grid if self._mode == "edit" else self._root_grid
        return _centered_rect(rect, self._grid_size_hint(grid), self._GRID_MARGIN)

    def _axis_rects(self) -> dict[str, QtCore.QRect]:
        return dict(self._axis_rect_items())

    def _axis_rect_items(self) -> tuple[tuple[str, QtCore.QRect], ...]:
        items: list[tuple[str, QtCore.QRect]] = []
        grid = self._display_grid if self._mode == "edit" else self._root_grid
        self._collect_axis_rects(grid, self._grid_rect(), items, depth=0)
        return tuple(items)

    def cell_rect(self, axis: tuple[int, int]) -> QtCore.QRect:
        row, col = axis
        if not (
            0 <= row < self._display_grid.nrows and 0 <= col < self._display_grid.ncols
        ):
            return QtCore.QRect()
        return self._span_rect(
            self._display_grid,
            self._grid_rect(),
            FigureGridSpecSpanState(
                row_start=row,
                row_stop=row + 1,
                col_start=col,
                col_stop=col + 1,
            ),
        )

    def span_rect(self, span: FigureGridSpecSpanState) -> QtCore.QRect:
        if not self._span_within_grid(self._display_grid, span):
            return QtCore.QRect()
        return self._span_rect(self._display_grid, self._grid_rect(), span)

    def _collect_axis_rects(
        self,
        grid: FigureGridSpecGridState,
        grid_rect: QtCore.QRect,
        items: list[tuple[str, QtCore.QRect]],
        *,
        depth: int,
    ) -> None:
        items.extend(
            (axis.axes_id, self._span_rect(grid, grid_rect, axis.span))
            for axis in grid.axes
            if self._span_within_grid(grid, axis.span)
        )
        for child in grid.child_grids:
            if child.span is None or not self._span_within_grid(grid, child.span):
                continue
            child_rect = self._nested_content_rect(
                self._span_rect(grid, grid_rect, child.span),
                depth=depth,
            )
            self._collect_axis_rects(child, child_rect, items, depth=depth + 1)

    def _draw_grid(
        self,
        painter: QtGui.QPainter,
        grid: FigureGridSpecGridState,
        grid_rect: QtCore.QRect,
        *,
        depth: int,
    ) -> None:
        colors = _selector_colors(self.palette())

        _draw_selector_rect(
            painter,
            grid_rect,
            facecolor=colors.panel,
            edgecolor=colors.border,
            draw_edge=False,
        )
        self._draw_empty_cells(painter, grid, grid_rect, colors)

        for child in grid.child_grids:
            if child.span is None or not self._span_within_grid(grid, child.span):
                continue
            rect = self._span_rect(grid, grid_rect, child.span)
            selected_child = (
                self._mode == "edit"
                and depth == 0
                and child.grid_id == self._selected_region_id
                and self._region_handles_visible
            )
            _draw_selector_rect(
                painter,
                rect,
                facecolor=(
                    colors.selection_face if selected_child else colors.nested_face
                ),
                edgecolor=colors.selection if selected_child else colors.border,
                linewidth=(
                    _SELECTOR_SELECTED_BORDER_WIDTH
                    if selected_child
                    else _SELECTOR_BORDER_WIDTH
                ),
            )
            if selected_child:
                self._draw_resize_handles(painter, rect, colors.selection)
            child_rect = self._nested_content_rect(rect, depth=depth)
            self._draw_grid(painter, child, child_rect, depth=depth + 1)
            if self._mode == "edit" and depth == 0:
                painter.setPen(colors.muted_text)
                text = painter.fontMetrics().elidedText(
                    self._region_label(child.grid_id),
                    QtCore.Qt.TextElideMode.ElideRight,
                    max(0, rect.width() - 6),
                )
                painter.drawText(
                    rect.adjusted(self._HANDLE_SIZE + 3, 1, -3, -1),
                    QtCore.Qt.AlignmentFlag.AlignLeft
                    | QtCore.Qt.AlignmentFlag.AlignTop,
                    text,
                )

        for axis in grid.axes:
            if not self._span_within_grid(grid, axis.span):
                continue
            rect = self._span_rect(grid, grid_rect, axis.span)
            selected = (
                axis.axes_id == self._selected_region_id
                if self._mode == "edit" and depth == 0 and self._region_handles_visible
                else axis.axes_id in self._selected_axes
            )
            hovered = axis.axes_id == self._hovered_axis_id
            face = (
                colors.selection_face
                if selected
                else colors.hover_face
                if hovered
                else colors.face
            )
            _draw_selector_rect(
                painter,
                rect,
                facecolor=face,
                edgecolor=colors.selection if selected else colors.border,
                linewidth=(
                    _SELECTOR_SELECTED_BORDER_WIDTH
                    if selected
                    else _SELECTOR_BORDER_WIDTH
                ),
            )
            painter.setPen(colors.selected_text if selected else colors.text)
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
            if self._mode == "edit" and depth == 0 and selected:
                self._draw_resize_handles(painter, rect, colors.selection)

    def _draw_empty_cells(
        self,
        painter: QtGui.QPainter,
        grid: FigureGridSpecGridState,
        grid_rect: QtCore.QRect,
        colors: _SelectorColors,
    ) -> None:
        occupied_cells = self._occupied_grid_cells(grid)
        empty_edge = QtGui.QColor(colors.border)
        empty_edge.setAlpha(96)
        for row in range(grid.nrows):
            for col in range(grid.ncols):
                if (row, col) in occupied_cells:
                    continue
                _draw_selector_rect(
                    painter,
                    self._span_rect(
                        grid,
                        grid_rect,
                        FigureGridSpecSpanState(
                            row_start=row,
                            row_stop=row + 1,
                            col_start=col,
                            col_stop=col + 1,
                        ),
                    ),
                    facecolor=colors.face,
                    edgecolor=empty_edge,
                    linewidth=0.8,
                )

    def _occupied_grid_cells(
        self, grid: FigureGridSpecGridState
    ) -> set[tuple[int, int]]:
        occupied: set[tuple[int, int]] = set()

        def add_cells(span: FigureGridSpecSpanState) -> None:
            if not self._span_within_grid(grid, span):
                return
            for row in range(span.row_start, span.row_stop):
                for col in range(span.col_start, span.col_stop):
                    occupied.add((row, col))

        for child in grid.child_grids:
            if child.span is not None:
                add_cells(child.span)
        for axis in grid.axes:
            add_cells(axis.span)
        return occupied

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
        )

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

    def _install_application_event_filter(self) -> None:
        if (
            self._mode != "edit"
            or self._application_event_filter_installed
            or self._outside_click_filter is None
        ):
            return
        application = QtWidgets.QApplication.instance()
        if isinstance(application, QtWidgets.QApplication):
            application.installEventFilter(self._outside_click_filter)
            self._application = application
            self._application_event_filter_installed = True

    def _remove_application_event_filter(self) -> None:
        if not self._application_event_filter_installed:
            return
        application = self._application
        event_filter = self._outside_click_filter
        self._application = None
        self._application_event_filter_installed = False
        if (
            application is not None
            and event_filter is not None
            and erlab.interactive.utils.qt_is_valid(application, event_filter)
        ):
            application.removeEventFilter(event_filter)

    def _handle_application_event(self, event: QtCore.QEvent | None) -> None:
        if (
            self._mode != "edit"
            or not self._region_handles_visible
            or event is None
            or not self.isVisible()
        ):
            return
        if event.type() == QtCore.QEvent.Type.WindowDeactivate:
            self._set_region_handles_visible(False)
            return
        if event.type() != QtCore.QEvent.Type.MouseButtonPress:
            return
        mouse_event = event if isinstance(event, QtGui.QMouseEvent) else None
        if mouse_event is None:
            return
        editor_rect = QtCore.QRect(
            self.mapToGlobal(QtCore.QPoint(0, 0)),
            self.size(),
        )
        if not editor_rect.contains(mouse_event.globalPosition().toPoint()):
            self._set_region_handles_visible(False)

    def _set_region_handles_visible(self, visible: bool) -> None:
        if self._region_handles_visible == visible:
            return
        self._region_handles_visible = visible
        self.update()

    def _region_label(self, region_id: str) -> str:
        for region in self._regions:
            if region.region_id == region_id:
                return region.label
        return self._labels.get(region_id, region_id)

    def _grid_size_hint(
        self, grid: FigureGridSpecGridState, *, depth: int = 0
    ) -> QtCore.QSize:
        cell_width = self._CELL_WIDTH
        cell_height = self._CELL_HEIGHT
        for child in grid.child_grids:
            if child.span is None or not self._span_within_grid(grid, child.span):
                continue
            child_size = self._grid_size_hint(child, depth=depth + 1)
            span_cols = child.span.col_stop - child.span.col_start
            span_rows = child.span.row_stop - child.span.row_start
            if span_cols:
                cell_width = max(
                    cell_width,
                    math.ceil(
                        (child_size.width() + 2 * self._NESTED_INSET) / span_cols
                    ),
                )
            if span_rows:
                cell_height = max(
                    cell_height,
                    math.ceil(
                        (
                            child_size.height()
                            + self._nested_top_inset(depth)
                            + self._NESTED_INSET
                        )
                        / span_rows
                    ),
                )
        width = grid.ncols * cell_width + max(grid.ncols - 1, 0) * self._CELL_GAP
        height = grid.nrows * cell_height + max(grid.nrows - 1, 0) * self._CELL_GAP
        return QtCore.QSize(width, height)

    def _nested_content_rect(self, rect: QtCore.QRect, *, depth: int) -> QtCore.QRect:
        return rect.adjusted(
            self._NESTED_INSET,
            self._nested_top_inset(depth),
            -self._NESTED_INSET,
            -self._NESTED_INSET,
        )

    def _nested_top_inset(self, depth: int) -> int:
        if self._mode == "edit" and depth == 0:
            return self._NESTED_INSET + self._NESTED_LABEL_HEIGHT
        return self._NESTED_INSET

    def _cell_at(
        self, pos: QtCore.QPoint, *, clamp_to_grid: bool = False
    ) -> tuple[int, int] | None:
        grid_rect = self._grid_rect()
        if clamp_to_grid:
            if grid_rect.isEmpty():
                return None
            x = min(max(pos.x(), grid_rect.left()), grid_rect.right())
            y = min(max(pos.y(), grid_rect.top()), grid_rect.bottom())
            pos = QtCore.QPoint(x, y)
        for row in range(self._display_grid.nrows):
            for col in range(self._display_grid.ncols):
                if self.cell_rect((row, col)).contains(pos):
                    return row, col
        if clamp_to_grid:
            cells = [
                (row, col)
                for row in range(self._display_grid.nrows)
                for col in range(self._display_grid.ncols)
            ]
            if cells:
                return min(
                    cells,
                    key=lambda cell: (
                        self.cell_rect(cell).center() - pos
                    ).manhattanLength(),
                )
        return None

    def _region_at(self, pos: QtCore.QPoint) -> _GridSpecRegionInfo | None:
        for region in reversed(self._regions):
            if not region.valid or not self._span_within_grid(
                self._display_grid, region.span
            ):
                continue
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
            or not self._region_handles_visible
            or not region.valid
            or not self._span_within_grid(self._display_grid, region.span)
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
            return QtCore.QRect(point.x() - half, point.y() - half, size, size)

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
            painter.drawRoundedRect(handle_rect, 2, 2)

    def _draw_preview_span(
        self, painter: QtGui.QPainter, span: FigureGridSpecSpanState
    ) -> None:
        preview_color = self.palette().color(QtGui.QPalette.ColorRole.Highlight)
        preview_fill = QtGui.QColor(preview_color)
        preview_fill.setAlpha(35)
        pen = QtGui.QPen(preview_color, 1.4)
        pen.setStyle(QtCore.Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(QtGui.QBrush(preview_fill))
        painter.drawRoundedRect(
            self.span_rect(span),
            _SELECTOR_CORNER_RADIUS,
            _SELECTOR_CORNER_RADIUS,
        )

    def _update_hover_cursor(self, pos: QtCore.QPoint) -> None:
        handle = self._handle_at(pos)
        if handle in {"nw", "se"}:
            cursor_shape = QtCore.Qt.CursorShape.SizeFDiagCursor
        elif handle in {"ne", "sw"}:
            cursor_shape = QtCore.Qt.CursorShape.SizeBDiagCursor
        elif self._region_at(pos) is not None:
            cursor_shape = QtCore.Qt.CursorShape.SizeAllCursor
        else:
            cursor_shape = None
        erlab.interactive.utils.set_widget_cursor(self, cursor_shape)

    def _active_preview_span(self) -> FigureGridSpecSpanState | None:
        if self._drag_mode in {"move", "resize"}:
            return self._resize_preview_span
        if self._drag_mode == "create" and self._drag_origin_cell is not None:
            current = self._drag_current_cell
            if current is not None:
                return self._span_from_cells(self._drag_origin_cell, current)
        return None

    def _reset_edit_drag(self) -> None:
        self._drag_mode = "none"
        self._drag_origin = None
        self._drag_origin_cell = None
        self._drag_current_cell = None
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
            row_stop = max(min(row + 1, self._display_grid.nrows), row_start + 1)
        if "w" in handle:
            col_start = min(max(col, 0), col_stop - 1)
        elif "e" in handle:
            col_stop = max(min(col + 1, self._display_grid.ncols), col_start + 1)
        return FigureGridSpecSpanState(
            row_start=row_start,
            row_stop=row_stop,
            col_start=col_start,
            col_stop=col_stop,
        )

    def _span_from_move(
        self,
        original: FigureGridSpecSpanState,
        *,
        origin: tuple[int, int],
        current: tuple[int, int],
    ) -> FigureGridSpecSpanState:
        row_delta = current[0] - origin[0]
        col_delta = current[1] - origin[1]
        row_height = original.row_stop - original.row_start
        col_width = original.col_stop - original.col_start
        row_start = min(
            max(original.row_start + row_delta, 0),
            max(self._display_grid.nrows - row_height, 0),
        )
        col_start = min(
            max(original.col_start + col_delta, 0),
            max(self._display_grid.ncols - col_width, 0),
        )
        return FigureGridSpecSpanState(
            row_start=row_start,
            row_stop=row_start + row_height,
            col_start=col_start,
            col_stop=col_start + col_width,
        )

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
