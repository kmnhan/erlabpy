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
from erlab.interactive._figurecomposer._operations._method._catalog import (
    TICK_PARAMS_CONTROLLED_KWARGS,
    TICK_PARAMS_DEFAULT_KWARGS,
)
from erlab.interactive._figurecomposer._text import _code_args, _literal_from_text
from erlab.interactive._figurecomposer._ui._widgets import _ColorLineEditWidget

if typing.TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

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

        self.axis_control = _ChoiceComboBox(
            (("Both", "both"), ("X", "x"), ("Y", "y")),
            self,
        )
        self.axis_control.setObjectName(f"{object_prefix}AxisCombo")
        self.which_control = _ChoiceComboBox(
            (("Major", "major"), ("Minor", "minor"), ("Both", "both")),
            self,
        )
        self.which_control.setObjectName(f"{object_prefix}WhichCombo")
        self.direction_control = _ChoiceComboBox(
            (("Default", None), ("in", "in"), ("out", "out"), ("inout", "inout")),
            self,
        )
        self.direction_control.setObjectName(f"{object_prefix}DirectionCombo")
        self.reset_button = _TriStateCheckBox("Reset", self, show_text=True)
        self.reset_button.setObjectName(f"{object_prefix}ResetCombo")

        self.side_buttons: dict[str, _TriStateCheckBox] = {}
        for _label, tick_key, label_key in _SIDE_ROWS:
            tick_button = _TriStateCheckBox(f"{tick_key} ticks", self)
            tick_button.setObjectName(f"{object_prefix}{tick_key.title()}Combo")
            label_button = _TriStateCheckBox(f"{tick_key} labels", self)
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

        self.colors_edit = _color_edit(self)
        self.colors_edit.setLineEditObjectName(f"{object_prefix}ColorsEdit")
        self.colors_edit.setColorButtonObjectName(f"{object_prefix}ColorsEditButton")
        self.tick_color_edit = _color_edit(self)
        self.tick_color_edit.setLineEditObjectName(f"{object_prefix}TickColorEdit")
        self.tick_color_edit.setColorButtonObjectName(
            f"{object_prefix}TickColorEditButton"
        )
        self.label_color_edit = _color_edit(self)
        self.label_color_edit.setLineEditObjectName(f"{object_prefix}LabelColorEdit")
        self.label_color_edit.setColorButtonObjectName(
            f"{object_prefix}LabelColorEditButton"
        )
        self.grid_color_edit = _color_edit(self)
        self.grid_color_edit.setLineEditObjectName(f"{object_prefix}GridColorEdit")
        self.grid_color_edit.setColorButtonObjectName(
            f"{object_prefix}GridColorEditButton"
        )

        self.grid_style_combo = QtWidgets.QComboBox(self)
        self.grid_style_combo.setObjectName(f"{object_prefix}GridLineStyleCombo")
        configure_style_combo(self.grid_style_combo, LINE_STYLE_OPTIONS, None)
        self.grid_style_combo.setMaximumWidth(120)

        self._configure_tooltips()
        self._build_layout()
        self._connect_controls()
        self.set_tick_params(tick_params or {})

    def setToolTip(self, _text: str | None) -> None:
        """Keep generic form-row tooltips off the compound widget itself."""
        super().setToolTip("")

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

    def _configure_tooltips(self) -> None:
        self.axis_control.setToolTip("Choose x ticks, y ticks, or both.")
        self.which_control.setToolTip("Choose major ticks, minor ticks, or both.")
        self.direction_control.setToolTip(
            "Choose the direction tick marks point.\n"
            "Default leaves tick direction unchanged."
        )
        self.reset_button.setToolTip(
            "Auto leaves reset unspecified.\n"
            "On resets ticks before applying these changes.\n"
            "Off passes reset=False."
        )
        side_tooltips = {
            "bottom": "Bottom tick marks.",
            "top": "Top tick marks.",
            "left": "Left tick marks.",
            "right": "Right tick marks.",
            "labelbottom": "Bottom tick labels.",
            "labeltop": "Top tick labels.",
            "labelleft": "Left tick labels.",
            "labelright": "Right tick labels.",
        }
        for key, button in self.side_buttons.items():
            button.setToolTip(
                f"{side_tooltips[key]}\n"
                "Auto leaves this setting unchanged.\n"
                "Checked passes True; unchecked passes False."
            )
        self.length_edit.setToolTip("Tick mark length in points.")
        self.width_edit.setToolTip("Tick mark width in points.")
        self.pad_edit.setToolTip("Distance between tick marks and labels in points.")
        self.label_rotation_edit.setToolTip("Tick label rotation in degrees.")
        self.label_size_edit.setToolTip(
            "Tick label font size, such as 8, 'small', or 'large'."
        )
        self.label_font_edit.setToolTip("Tick label font family.")
        self.colors_edit.setToolTip("Color applied to both tick marks and labels.")
        self.tick_color_edit.setToolTip("Tick mark color.")
        self.label_color_edit.setToolTip("Tick label color.")
        self.zorder_edit.setToolTip("Drawing order for ticks and labels.")
        self.grid_color_edit.setToolTip("Grid line color for the selected ticks.")
        self.grid_alpha_edit.setToolTip("Grid line opacity between 0 and 1.")
        self.grid_width_edit.setToolTip("Grid line width.")
        self.grid_style_combo.setToolTip("Grid line style.")

    def _build_layout(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        scope = QtWidgets.QGroupBox("Scope", self)
        scope_layout = QtWidgets.QGridLayout(scope)
        scope_layout.setContentsMargins(8, 4, 8, 8)
        scope_layout.setHorizontalSpacing(6)
        scope_layout.setVerticalSpacing(4)
        _add_labeled_widget(scope_layout, 0, 0, "Axis", self.axis_control)
        _add_labeled_widget(scope_layout, 0, 2, "Set", self.which_control)
        _add_labeled_widget(scope_layout, 1, 0, "Direction", self.direction_control)
        _add_labeled_widget(scope_layout, 1, 2, "Reset", self.reset_button)
        for column in (1, 3):
            scope_layout.setColumnStretch(column, 1)
        layout.addWidget(scope)

        sides = QtWidgets.QGroupBox("Sides", self)
        sides.setObjectName(f"{self._object_prefix}SidesMatrix")
        sides_layout = QtWidgets.QGridLayout(sides)
        sides_layout.setContentsMargins(8, 4, 8, 8)
        sides_layout.setHorizontalSpacing(8)
        sides_layout.setVerticalSpacing(3)
        for column in (0, 3):
            sides_layout.addWidget(QtWidgets.QLabel("Side", sides), 0, column)
            sides_layout.addWidget(QtWidgets.QLabel("Ticks", sides), 0, column + 1)
            sides_layout.addWidget(QtWidgets.QLabel("Labels", sides), 0, column + 2)
        for index, (label, tick_key, label_key) in enumerate(_SIDE_ROWS):
            row = index // 2 + 1
            column = (index % 2) * 3
            sides_layout.addWidget(QtWidgets.QLabel(label, sides), row, column)
            sides_layout.addWidget(
                self.side_buttons[tick_key],
                row,
                column + 1,
                alignment=QtCore.Qt.AlignmentFlag.AlignCenter,
            )
            sides_layout.addWidget(
                self.side_buttons[label_key],
                row,
                column + 2,
                alignment=QtCore.Qt.AlignmentFlag.AlignCenter,
            )
        layout.addWidget(sides)

        appearance = QtWidgets.QGroupBox("Appearance", self)
        appearance_layout = QtWidgets.QGridLayout(appearance)
        appearance_layout.setContentsMargins(8, 4, 8, 8)
        appearance_layout.setHorizontalSpacing(6)
        appearance_layout.setVerticalSpacing(4)
        _add_labeled_widget(appearance_layout, 0, 0, "Length", self.length_edit)
        _add_labeled_widget(appearance_layout, 0, 2, "Width", self.width_edit)
        _add_labeled_widget(appearance_layout, 1, 0, "Pad", self.pad_edit)
        _add_labeled_widget(
            appearance_layout, 1, 2, "Rotation", self.label_rotation_edit
        )
        _add_labeled_widget(appearance_layout, 2, 0, "Size", self.label_size_edit)
        _add_labeled_widget(appearance_layout, 2, 2, "Font", self.label_font_edit)
        for column in (1, 3):
            appearance_layout.setColumnStretch(column, 1)
        layout.addWidget(appearance)

        colors = QtWidgets.QGroupBox("Color", self)
        colors_layout = QtWidgets.QGridLayout(colors)
        colors_layout.setContentsMargins(8, 4, 8, 8)
        colors_layout.setHorizontalSpacing(6)
        colors_layout.setVerticalSpacing(4)
        _add_labeled_widget(colors_layout, 0, 0, "All", self.colors_edit)
        _add_labeled_widget(colors_layout, 0, 2, "Ticks", self.tick_color_edit)
        _add_labeled_widget(colors_layout, 1, 0, "Labels", self.label_color_edit)
        for column in (1, 3):
            colors_layout.setColumnStretch(column, 1)
        layout.addWidget(colors)

        grid_content = QtWidgets.QWidget(self)
        grid_layout = QtWidgets.QGridLayout(grid_content)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setHorizontalSpacing(6)
        grid_layout.setVerticalSpacing(4)
        _add_labeled_widget(grid_layout, 0, 0, "Color", self.grid_color_edit)
        _add_labeled_widget(grid_layout, 0, 2, "Alpha", self.grid_alpha_edit)
        _add_labeled_widget(grid_layout, 1, 0, "Width", self.grid_width_edit)
        _add_labeled_widget(grid_layout, 1, 2, "Style", self.grid_style_combo)
        for column in (1, 3):
            grid_layout.setColumnStretch(column, 1)
        self.grid_disclosure = _DisclosureWidget("Grid", grid_content, self)
        self.grid_disclosure.setObjectName(f"{self._object_prefix}GridDisclosure")
        self.grid_disclosure.setToolTip("Show grid-line tick parameter controls.")
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
        self.advanced_disclosure.setToolTip("Show less common tick parameter controls.")
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


class _ChoiceComboBox(QtWidgets.QComboBox):
    valueChanged = QtCore.Signal(object)

    def __init__(
        self,
        options: Sequence[tuple[str, typing.Any]],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        for label, value in options:
            self.addItem(label, value)
        self.setMaximumWidth(115)
        self.activated.connect(self._index_activated)

    def value(self) -> typing.Any:
        return self.currentData()

    def set_value(self, value: typing.Any) -> None:
        with QtCore.QSignalBlocker(self):
            for index in range(self.count()):
                if self.itemData(index) == value:
                    self.setCurrentIndex(index)
                    return
            self.setCurrentIndex(0)

    def _index_activated(self, _index: int) -> None:
        self.valueChanged.emit(self.currentData())


class _TriStateCheckBox(QtWidgets.QCheckBox):
    valueChanged = QtCore.Signal(object)

    def __init__(
        self,
        label: str,
        parent: QtWidgets.QWidget | None = None,
        *,
        show_text: bool = False,
    ) -> None:
        super().__init__(parent)
        self._value: bool | None = None
        self._label = label
        self._show_text = show_text
        self._base_tooltip = ""
        self.setTristate(True)
        self.stateChanged.connect(self._state_changed)
        self.set_value(None)

    def value(self) -> bool | None:
        return self._value

    def set_value(self, value: bool | None) -> None:
        if value not in {None, True, False}:
            value = None
        self._value = value
        self.setProperty("tick_params_value", value)
        if value is True:
            self.setCheckState(QtCore.Qt.CheckState.Checked)
        elif value is False:
            self.setCheckState(QtCore.Qt.CheckState.Unchecked)
        else:
            self.setCheckState(QtCore.Qt.CheckState.PartiallyChecked)
        if self._show_text:
            self.setText(_tri_state_text(value))
        self._update_tooltip()

    def setToolTip(self, text: str | None) -> None:
        self._base_tooltip = "" if text is None else text
        self._update_tooltip()

    def _state_changed(self, state: int) -> None:
        match QtCore.Qt.CheckState(state):
            case QtCore.Qt.CheckState.Checked:
                value = True
            case QtCore.Qt.CheckState.Unchecked:
                value = False
            case _:
                value = None
        self.set_value(value)
        self.valueChanged.emit(value)

    def _update_tooltip(self) -> None:
        current = f"Current: {_tri_state_text(self._value)}."
        if self._base_tooltip:
            super().setToolTip(f"{self._base_tooltip}\n{current}")
        else:
            super().setToolTip(f"{self._label}: {current}")


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

    def setToolTip(self, text: str | None) -> None:
        super().setToolTip(text)
        self._button.setToolTip(text)


def _line_edit(object_name: str, parent: QtWidgets.QWidget) -> QtWidgets.QLineEdit:
    edit = QtWidgets.QLineEdit(parent)
    edit.setObjectName(object_name)
    edit.setPlaceholderText("Auto")
    edit.setMaximumWidth(80)
    return edit


def _color_edit(parent: QtWidgets.QWidget) -> _ColorLineEditWidget:
    edit = _ColorLineEditWidget("", parent)
    edit.line_edit.setPlaceholderText("Auto")
    edit.line_edit.setMaximumWidth(82)
    edit.color_button.setFixedWidth(34)
    edit.setMaximumWidth(124)
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


def _tri_state_text(value: bool | None) -> str:
    if value is True:
        return "On"
    if value is False:
        return "Off"
    return "Auto"
