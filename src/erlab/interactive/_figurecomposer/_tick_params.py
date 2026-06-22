"""Compact editor for Matplotlib ``Axes.tick_params`` keyword arguments."""

from __future__ import annotations

import typing

from qtpy import QtCore, QtWidgets

from erlab.interactive._figurecomposer._line_style import (
    LINE_STYLE_OPTIONS,
    color_kw_value_from_text,
    configure_style_combo,
    style_combo_value,
)
from erlab.interactive._figurecomposer._text import _code_args, _literal_from_text
from erlab.interactive._figurecomposer._widgets import _ColorLineEditWidget

if typing.TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

TICK_PARAMS_DEFAULT_KWARGS: dict[str, typing.Any] = {
    "axis": "both",
    "which": "major",
}
TICK_PARAMS_CONTROLLED_KWARGS = frozenset(
    (
        "axis",
        "which",
        "direction",
        "reset",
        "bottom",
        "top",
        "left",
        "right",
        "labelbottom",
        "labeltop",
        "labelleft",
        "labelright",
        "length",
        "width",
        "pad",
        "labelrotation",
        "labelsize",
        "labelfontfamily",
        "colors",
        "color",
        "labelcolor",
        "zorder",
        "grid_color",
        "grid_alpha",
        "grid_linewidth",
        "grid_linestyle",
    )
)

_SIDE_ROWS = (
    ("Bottom", "bottom", "labelbottom"),
    ("Top", "top", "labeltop"),
    ("Left", "left", "labelleft"),
    ("Right", "right", "labelright"),
)


class TickParamsEditorWidget(QtWidgets.QWidget):
    """Compact editor that emits complete controlled ``tick_params`` kwargs."""

    sigTickParamsChanged = QtCore.Signal(object)

    def __init__(
        self,
        tick_params: Mapping[str, typing.Any] | None = None,
        parent: QtWidgets.QWidget | None = None,
        *,
        object_prefix: str = "figureComposerAxesMethodTickParams",
    ) -> None:
        super().__init__(parent)
        self._object_prefix = object_prefix
        self._kwargs: dict[str, typing.Any] = {}
        self._updating = False
        self.setObjectName(f"{object_prefix}Editor")
        self.setProperty("figureComposerTickParamsEditor", True)

        self.axis_control = _SegmentedChoiceWidget(
            (("both", "both"), ("x", "x"), ("y", "y")),
            self,
        )
        self.axis_control.setObjectName(f"{object_prefix}AxisCombo")
        self.which_control = _SegmentedChoiceWidget(
            (("major", "major"), ("minor", "minor"), ("both", "both")),
            self,
        )
        self.which_control.setObjectName(f"{object_prefix}WhichCombo")
        self.direction_control = _SegmentedChoiceWidget(
            (("Default", None), ("in", "in"), ("out", "out"), ("inout", "inout")),
            self,
        )
        self.direction_control.setObjectName(f"{object_prefix}DirectionCombo")
        self.reset_button = _TriStateButton("Reset", self)
        self.reset_button.setObjectName(f"{object_prefix}ResetCombo")

        self.side_buttons: dict[str, _TriStateButton] = {}
        for _label, tick_key, label_key in _SIDE_ROWS:
            tick_button = _TriStateButton(f"{tick_key} ticks", self)
            tick_button.setObjectName(f"{object_prefix}{tick_key.title()}Combo")
            label_button = _TriStateButton(f"{tick_key} labels", self)
            label_button.setObjectName(f"{object_prefix}Label{tick_key.title()}Combo")
            self.side_buttons[tick_key] = tick_button
            self.side_buttons[label_key] = label_button

        self.length_edit = _line_edit(f"{object_prefix}LengthEdit", self)
        self.width_edit = _line_edit(f"{object_prefix}WidthEdit", self)
        self.pad_edit = _line_edit(f"{object_prefix}PadEdit", self)
        self.label_rotation_edit = _line_edit(f"{object_prefix}LabelRotationEdit", self)
        self.label_size_edit = _line_edit(f"{object_prefix}LabelSizeEdit", self)
        self.label_font_edit = _line_edit(f"{object_prefix}LabelFontEdit", self)
        self.zorder_edit = _line_edit(f"{object_prefix}ZOrderEdit", self)
        self.grid_alpha_edit = _line_edit(f"{object_prefix}GridAlphaEdit", self)
        self.grid_width_edit = _line_edit(f"{object_prefix}GridLineWidthEdit", self)

        self.colors_edit = _ColorLineEditWidget("", self)
        self.colors_edit.setLineEditObjectName(f"{object_prefix}ColorsEdit")
        self.colors_edit.setColorButtonObjectName(f"{object_prefix}ColorsEditButton")
        self.tick_color_edit = _ColorLineEditWidget("", self)
        self.tick_color_edit.setLineEditObjectName(f"{object_prefix}TickColorEdit")
        self.tick_color_edit.setColorButtonObjectName(
            f"{object_prefix}TickColorEditButton"
        )
        self.label_color_edit = _ColorLineEditWidget("", self)
        self.label_color_edit.setLineEditObjectName(f"{object_prefix}LabelColorEdit")
        self.label_color_edit.setColorButtonObjectName(
            f"{object_prefix}LabelColorEditButton"
        )
        self.grid_color_edit = _ColorLineEditWidget("", self)
        self.grid_color_edit.setLineEditObjectName(f"{object_prefix}GridColorEdit")
        self.grid_color_edit.setColorButtonObjectName(
            f"{object_prefix}GridColorEditButton"
        )

        self.grid_style_combo = QtWidgets.QComboBox(self)
        self.grid_style_combo.setObjectName(f"{object_prefix}GridLineStyleCombo")
        configure_style_combo(self.grid_style_combo, LINE_STYLE_OPTIONS, None)

        self._build_layout()
        self._connect_controls()
        self.set_tick_params(tick_params or {})

    def tick_params(self) -> dict[str, typing.Any]:
        """Return explicit controlled keyword arguments."""
        return dict(self._kwargs)

    def set_tick_params(self, tick_params: Mapping[str, typing.Any]) -> None:
        self._updating = True
        try:
            self._kwargs = {
                key: value
                for key, value in tick_params.items()
                if key in TICK_PARAMS_CONTROLLED_KWARGS and value is not None
            }
            self.axis_control.set_value(
                self._kwargs.get("axis", TICK_PARAMS_DEFAULT_KWARGS["axis"])
            )
            self.which_control.set_value(
                self._kwargs.get("which", TICK_PARAMS_DEFAULT_KWARGS["which"])
            )
            self.direction_control.set_value(self._kwargs.get("direction"))
            self.reset_button.set_value(self._kwargs.get("reset"))
            for key, button in self.side_buttons.items():
                button.set_value(self._kwargs.get(key))
            self._set_line_edit(
                self.length_edit,
                _format_float(self._kwargs.get("length")),
            )
            self._set_line_edit(
                self.width_edit,
                _format_float(self._kwargs.get("width")),
            )
            self._set_line_edit(self.pad_edit, _format_float(self._kwargs.get("pad")))
            self._set_line_edit(
                self.label_rotation_edit,
                _format_float(self._kwargs.get("labelrotation")),
            )
            self._set_line_edit(
                self.label_size_edit, _format_literal(self._kwargs.get("labelsize"))
            )
            self._set_line_edit(
                self.label_font_edit, str(self._kwargs.get("labelfontfamily") or "")
            )
            self.colors_edit.setText(str(self._kwargs.get("colors") or ""))
            self.tick_color_edit.setText(str(self._kwargs.get("color") or ""))
            self.label_color_edit.setText(str(self._kwargs.get("labelcolor") or ""))
            self._set_line_edit(
                self.zorder_edit,
                _format_float(self._kwargs.get("zorder")),
            )
            self.grid_color_edit.setText(str(self._kwargs.get("grid_color") or ""))
            self._set_line_edit(
                self.grid_alpha_edit, _format_float(self._kwargs.get("grid_alpha"))
            )
            self._set_line_edit(
                self.grid_width_edit,
                _format_float(self._kwargs.get("grid_linewidth")),
            )
            self._set_combo_value(
                self.grid_style_combo,
                self._kwargs.get("grid_linestyle"),
            )
            self.grid_disclosure.setExpanded(
                any(key.startswith("grid_") for key in self._kwargs)
            )
            self.advanced_disclosure.setExpanded("zorder" in self._kwargs)
        finally:
            self._updating = False

    def _build_layout(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        scope = QtWidgets.QWidget(self)
        scope_layout = QtWidgets.QGridLayout(scope)
        scope_layout.setContentsMargins(0, 0, 0, 0)
        scope_layout.setHorizontalSpacing(6)
        scope_layout.setVerticalSpacing(2)
        _add_labeled_widget(scope_layout, 0, 0, "Axis", self.axis_control)
        _add_labeled_widget(scope_layout, 0, 2, "Ticks", self.which_control)
        _add_labeled_widget(scope_layout, 1, 0, "Direction", self.direction_control)
        _add_labeled_widget(scope_layout, 1, 2, "Reset", self.reset_button)
        scope_layout.setColumnStretch(1, 1)
        scope_layout.setColumnStretch(3, 1)
        layout.addWidget(scope)

        sides = QtWidgets.QWidget(self)
        sides.setObjectName(f"{self._object_prefix}SidesMatrix")
        sides_layout = QtWidgets.QGridLayout(sides)
        sides_layout.setContentsMargins(0, 0, 0, 0)
        sides_layout.setHorizontalSpacing(6)
        sides_layout.setVerticalSpacing(2)
        sides_layout.addWidget(QtWidgets.QLabel("Side", sides), 0, 0)
        sides_layout.addWidget(QtWidgets.QLabel("Tick", sides), 0, 1)
        sides_layout.addWidget(QtWidgets.QLabel("Label", sides), 0, 2)
        for row, (label, tick_key, label_key) in enumerate(_SIDE_ROWS, start=1):
            sides_layout.addWidget(QtWidgets.QLabel(label, sides), row, 0)
            sides_layout.addWidget(self.side_buttons[tick_key], row, 1)
            sides_layout.addWidget(self.side_buttons[label_key], row, 2)
        sides_layout.setColumnStretch(3, 1)
        layout.addWidget(sides)

        appearance = QtWidgets.QWidget(self)
        appearance_layout = QtWidgets.QGridLayout(appearance)
        appearance_layout.setContentsMargins(0, 0, 0, 0)
        appearance_layout.setHorizontalSpacing(6)
        appearance_layout.setVerticalSpacing(4)
        _add_labeled_widget(appearance_layout, 0, 0, "Length", self.length_edit)
        _add_labeled_widget(appearance_layout, 0, 2, "Width", self.width_edit)
        _add_labeled_widget(appearance_layout, 0, 4, "Pad", self.pad_edit)
        _add_labeled_widget(appearance_layout, 1, 0, "Both color", self.colors_edit)
        _add_labeled_widget(appearance_layout, 1, 2, "Tick color", self.tick_color_edit)
        _add_labeled_widget(
            appearance_layout, 1, 4, "Label color", self.label_color_edit
        )
        _add_labeled_widget(appearance_layout, 2, 0, "Size", self.label_size_edit)
        _add_labeled_widget(
            appearance_layout, 2, 2, "Rotation", self.label_rotation_edit
        )
        _add_labeled_widget(appearance_layout, 2, 4, "Font", self.label_font_edit)
        for column in (1, 3, 5):
            appearance_layout.setColumnStretch(column, 1)
        layout.addWidget(appearance)

        grid_content = QtWidgets.QWidget(self)
        grid_layout = QtWidgets.QGridLayout(grid_content)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setHorizontalSpacing(6)
        grid_layout.setVerticalSpacing(4)
        _add_labeled_widget(grid_layout, 0, 0, "Color", self.grid_color_edit)
        _add_labeled_widget(grid_layout, 0, 2, "Alpha", self.grid_alpha_edit)
        _add_labeled_widget(grid_layout, 0, 4, "Width", self.grid_width_edit)
        _add_labeled_widget(grid_layout, 1, 0, "Style", self.grid_style_combo)
        for column in (1, 3, 5):
            grid_layout.setColumnStretch(column, 1)
        self.grid_disclosure = _DisclosureWidget("Grid", grid_content, self)
        self.grid_disclosure.setObjectName(f"{self._object_prefix}GridDisclosure")
        self.grid_disclosure.setExpanded(
            any(key.startswith("grid_") for key in self._kwargs)
        )
        layout.addWidget(self.grid_disclosure)

        advanced_content = QtWidgets.QWidget(self)
        advanced_layout = QtWidgets.QGridLayout(advanced_content)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_layout.setHorizontalSpacing(6)
        _add_labeled_widget(advanced_layout, 0, 0, "Z order", self.zorder_edit)
        advanced_layout.setColumnStretch(1, 1)
        self.advanced_disclosure = _DisclosureWidget("Advanced", advanced_content, self)
        self.advanced_disclosure.setObjectName(
            f"{self._object_prefix}AdvancedDisclosure"
        )
        self.advanced_disclosure.setExpanded("zorder" in self._kwargs)
        layout.addWidget(self.advanced_disclosure)

    def _connect_controls(self) -> None:
        self.axis_control.valueChanged.connect(
            lambda value: self._set_kwarg("axis", value)
        )
        self.which_control.valueChanged.connect(
            lambda value: self._set_kwarg("which", value)
        )
        self.direction_control.valueChanged.connect(
            lambda value: self._set_kwarg("direction", value)
        )
        self.reset_button.valueChanged.connect(
            lambda value: self._set_kwarg("reset", value)
        )
        for key, button in self.side_buttons.items():
            button.valueChanged.connect(
                lambda value, key=key: self._set_kwarg(key, value)
            )

        self.length_edit.editingFinished.connect(
            lambda: self._commit_float(self.length_edit, "length", minimum=0.0)
        )
        self.width_edit.editingFinished.connect(
            lambda: self._commit_float(self.width_edit, "width", minimum=0.0)
        )
        self.pad_edit.editingFinished.connect(
            lambda: self._commit_float(self.pad_edit, "pad", minimum=0.0)
        )
        self.label_rotation_edit.editingFinished.connect(
            lambda: self._commit_float(self.label_rotation_edit, "labelrotation")
        )
        self.label_size_edit.editingFinished.connect(
            lambda: self._commit_literal(self.label_size_edit, "labelsize")
        )
        self.label_font_edit.editingFinished.connect(
            lambda: self._commit_text(self.label_font_edit, "labelfontfamily")
        )
        self.colors_edit.editingFinished.connect(
            lambda: self._commit_color(self.colors_edit, "colors")
        )
        self.tick_color_edit.editingFinished.connect(
            lambda: self._commit_color(self.tick_color_edit, "color")
        )
        self.label_color_edit.editingFinished.connect(
            lambda: self._commit_color(self.label_color_edit, "labelcolor")
        )
        self.zorder_edit.editingFinished.connect(
            lambda: self._commit_float(self.zorder_edit, "zorder")
        )
        self.grid_color_edit.editingFinished.connect(
            lambda: self._commit_color(self.grid_color_edit, "grid_color")
        )
        self.grid_alpha_edit.editingFinished.connect(
            lambda: self._commit_float(
                self.grid_alpha_edit, "grid_alpha", minimum=0.0, maximum=1.0
            )
        )
        self.grid_width_edit.editingFinished.connect(
            lambda: self._commit_float(
                self.grid_width_edit, "grid_linewidth", minimum=0.0
            )
        )
        self.grid_style_combo.activated.connect(
            lambda _index: self._set_kwarg(
                "grid_linestyle", style_combo_value(self.grid_style_combo)
            )
        )

    def _set_kwarg(self, key: str, value: typing.Any) -> None:
        if self._updating:
            return
        if value is None:
            self._kwargs.pop(key, None)
        else:
            self._kwargs[key] = value
        self.sigTickParamsChanged.emit(self.tick_params())

    def _commit_float(
        self,
        edit: QtWidgets.QLineEdit,
        key: str,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> None:
        text = edit.text().strip()
        if not text:
            self._set_kwarg(key, None)
            return
        try:
            value = float(text)
        except ValueError:
            return
        if minimum is not None and value < minimum:
            return
        if maximum is not None and value > maximum:
            return
        self._set_line_edit(edit, _format_float(value))
        self._set_kwarg(key, value)

    def _commit_literal(self, edit: QtWidgets.QLineEdit, key: str) -> None:
        text = edit.text().strip()
        if not text:
            self._set_kwarg(key, None)
            return
        try:
            value = _literal_from_text(text)
        except ValueError:
            return
        self._set_kwarg(key, value)

    def _commit_text(self, edit: QtWidgets.QLineEdit, key: str) -> None:
        self._set_kwarg(key, edit.text().strip() or None)

    def _commit_color(self, edit: _ColorLineEditWidget, key: str) -> None:
        text = edit.text().strip()
        self._set_kwarg(key, None if not text else color_kw_value_from_text(text))

    @staticmethod
    def _set_line_edit(edit: QtWidgets.QLineEdit, text: str) -> None:
        with QtCore.QSignalBlocker(edit):
            edit.setText(text)
            edit.setModified(False)

    @staticmethod
    def _set_combo_value(combo: QtWidgets.QComboBox, value: typing.Any) -> None:
        normalized = None if value in {None, ""} else str(value)
        with QtCore.QSignalBlocker(combo):
            for index in range(combo.count()):
                if combo.itemData(index) == normalized:
                    combo.setCurrentIndex(index)
                    return
            combo.setCurrentIndex(0)


class _SegmentedChoiceWidget(QtWidgets.QWidget):
    valueChanged = QtCore.Signal(object)

    def __init__(
        self,
        options: Sequence[tuple[str, typing.Any]],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._values = [value for _label, value in options]
        self._buttons: list[QtWidgets.QToolButton] = []
        self._group = QtWidgets.QButtonGroup(self)
        self._group.setExclusive(True)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        for index, (label, value) in enumerate(options):
            button = QtWidgets.QToolButton(self)
            button.setText(label)
            button.setCheckable(True)
            button.setAutoRaise(True)
            button.setProperty("tick_params_value", value)
            button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
            button.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            self._group.addButton(button, index)
            self._buttons.append(button)
            layout.addWidget(button)
        self._group.idClicked.connect(self._id_clicked)

    def value(self) -> typing.Any:
        checked = self._group.checkedId()
        if checked < 0:
            return None
        return self._values[checked]

    def set_value(self, value: typing.Any) -> None:
        with QtCore.QSignalBlocker(self._group):
            for index, candidate in enumerate(self._values):
                if candidate == value:
                    self._buttons[index].setChecked(True)
                    return
            self._buttons[0].setChecked(True)

    def _id_clicked(self, index: int) -> None:
        if 0 <= index < len(self._values):
            self.valueChanged.emit(self._values[index])


class _TriStateButton(QtWidgets.QToolButton):
    valueChanged = QtCore.Signal(object)

    def __init__(self, label: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._value: bool | None = None
        self._label = label
        self.setAutoRaise(True)
        self.setCheckable(True)
        self.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.clicked.connect(self._cycle)
        self.set_value(None)

    def value(self) -> bool | None:
        return self._value

    def set_value(self, value: bool | None) -> None:
        if value not in {None, True, False}:
            value = None
        self._value = value
        self.setProperty("tick_params_value", value)
        if value is True:
            self.setText("On")
            self.setChecked(True)
        elif value is False:
            self.setText("Off")
            self.setChecked(True)
        else:
            self.setText("Auto")
            self.setChecked(False)
        self.setToolTip(f"{self._label}: {self.text()}")

    def _cycle(self) -> None:
        match self._value:
            case None:
                value = True
            case True:
                value = False
            case False:
                value = None
        self.set_value(value)
        self.valueChanged.emit(value)


class _DisclosureWidget(QtWidgets.QWidget):
    def __init__(
        self,
        title: str,
        content: QtWidgets.QWidget,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._content = content
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self._button = QtWidgets.QToolButton(self)
        self._button.setText(title)
        self._button.setCheckable(True)
        self._button.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self._button.clicked.connect(lambda checked: self.setExpanded(checked))
        layout.addWidget(self._button)
        layout.addWidget(content)
        self.setExpanded(False)

    def setExpanded(self, expanded: bool) -> None:
        self._button.setChecked(expanded)
        self._button.setArrowType(
            QtCore.Qt.ArrowType.DownArrow
            if expanded
            else QtCore.Qt.ArrowType.RightArrow
        )
        self._content.setVisible(expanded)


def _line_edit(object_name: str, parent: QtWidgets.QWidget) -> QtWidgets.QLineEdit:
    edit = QtWidgets.QLineEdit(parent)
    edit.setObjectName(object_name)
    edit.setPlaceholderText("Auto")
    edit.setMaximumWidth(110)
    return edit


def _add_labeled_widget(
    layout: QtWidgets.QGridLayout,
    row: int,
    column: int,
    label: str,
    widget: QtWidgets.QWidget,
) -> None:
    label_widget = QtWidgets.QLabel(label, layout.parentWidget())
    label_widget.setBuddy(widget)
    layout.addWidget(label_widget, row, column)
    layout.addWidget(widget, row, column + 1)


def _format_float(value: typing.Any) -> str:
    return "" if value is None else f"{float(value):g}"


def _format_literal(value: typing.Any) -> str:
    if value is None:
        return ""
    return _code_args((value,))
