from __future__ import annotations

import contextlib
import typing

import pydantic
from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive._options.core import (
    model_with_workspace_overrides,
    normalize_workspace_option_overrides,
    option_model_with_value,
    option_paths,
    option_value,
    workspace_overridable_option_paths,
)
from erlab.interactive._options.parameters import (
    ColorListWidget,
    FigureDpiOverrideWidget,
    StylesheetListWidget,
)
from erlab.interactive._options.schema import AppOptions

_Scope = typing.Literal["user", "workspace"]
_CATEGORY_ORDER = ("colors", "io", "ktool", "figure")


def _object_path(path: str) -> str:
    return path.replace("/", "__")


def _field_constraints(field: pydantic.fields.FieldInfo) -> dict[str, typing.Any]:
    constraints: dict[str, typing.Any] = {}
    for item in field.metadata:
        for name in ("ge", "gt", "le", "lt"):
            value = getattr(item, name, None)
            if value is not None:
                constraints[name] = value
    return constraints


def _model_field(path: str) -> pydantic.fields.FieldInfo:
    model_type: type[pydantic.BaseModel] = AppOptions
    field: pydantic.fields.FieldInfo | None = None
    for key in path.split("/"):
        field = model_type.model_fields[key]
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, pydantic.BaseModel):
            model_type = annotation
    if field is None:  # pragma: no cover
        raise KeyError(path)
    return field


def _path_title(path: str) -> str:
    model_type: type[pydantic.BaseModel] = AppOptions
    titles: list[str] = []
    for index, key in enumerate(path.split("/")):
        field = model_type.model_fields[key]
        if index > 0:
            titles.append(str(field.title or key.replace("_", " ").title()))
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, pydantic.BaseModel):
            model_type = annotation
    if len(titles) <= 1:
        return titles[0] if titles else path
    return " / ".join(titles)


def _path_description(path: str) -> str:
    description = _model_field(path).description
    return "" if description is None else str(description)


def _field_extra(path: str) -> dict[str, typing.Any]:
    extra = _model_field(path).json_schema_extra or {}
    return dict(extra) if isinstance(extra, dict) else {}


def _category_title(key: str) -> str:
    field = AppOptions.model_fields[key]
    return str(field.title or key.replace("_", " ").title())


def _leaf_paths_for_category(category: str, *, workspace_only: bool) -> tuple[str, ...]:
    prefix = f"{category}/"
    allowed = set(workspace_overridable_option_paths()) if workspace_only else None
    return tuple(
        path
        for path in option_paths()
        if path.startswith(prefix) and (allowed is None or path in allowed)
    )


def _is_bool_path(path: str) -> bool:
    value = option_value(AppOptions(), path)
    return isinstance(value, bool)


def _is_int_path(path: str) -> bool:
    value = option_value(AppOptions(), path)
    return isinstance(value, int) and not isinstance(value, bool)


def _is_float_path(path: str) -> bool:
    value = option_value(AppOptions(), path)
    return isinstance(value, float)


def _stylesheet_names(value: typing.Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, list | tuple):
        value = [value]
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        name = str(item).strip()
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out


class _ChoiceSliderLabelRow(QtWidgets.QWidget):
    def __init__(
        self,
        slider: QtWidgets.QSlider,
        labels: typing.Sequence[str],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._slider = slider
        self._labels: tuple[QtWidgets.QLabel, ...] = tuple(
            self._make_label(index, label) for index, label in enumerate(labels)
        )
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )

    def _make_label(self, index: int, text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text, self)
        label.setObjectName(f"choiceSliderLabel_{index}")
        label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        return label

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(
            sum(label.sizeHint().width() for label in self._labels),
            self._label_height(),
        )

    def minimumSizeHint(self) -> QtCore.QSize:
        return self.sizeHint()

    def edge_margins(self) -> tuple[int, int]:
        if not self._labels:
            return 0, 0
        left_inset = self._slider_handle_center(0).x()
        right_inset = (
            self._slider.width()
            - 1
            - self._slider_handle_center(len(self._labels) - 1).x()
        )
        return (
            max(0, self._half_label_width(self._labels[0]) - left_inset),
            max(0, self._half_label_width(self._labels[-1]) - right_inset),
        )

    def update_label_positions(self) -> None:
        row_height = self.height()
        for index, label in enumerate(self._labels):
            label_size = label.sizeHint()
            tick_x = self._tick_x(index)
            label.setGeometry(
                tick_x - label_size.width() // 2,
                0,
                label_size.width(),
                row_height,
            )

    def resizeEvent(self, event: QtGui.QResizeEvent | None) -> None:
        super().resizeEvent(event)
        self.update_label_positions()

    @staticmethod
    def _half_label_width(label: QtWidgets.QLabel) -> int:
        return (label.sizeHint().width() + 1) // 2

    def _label_height(self) -> int:
        return max((label.sizeHint().height() for label in self._labels), default=0)

    def _slider_handle_center(self, index: int) -> QtCore.QPoint:
        option = QtWidgets.QStyleOptionSlider()
        self._slider.initStyleOption(option)
        option.sliderPosition = index
        option.sliderValue = index
        handle_rect = self._slider.style().subControlRect(
            QtWidgets.QStyle.ComplexControl.CC_Slider,
            option,
            QtWidgets.QStyle.SubControl.SC_SliderHandle,
            self._slider,
        )
        return handle_rect.center()

    def _tick_x(self, index: int) -> int:
        return self.mapFromGlobal(
            self._slider.mapToGlobal(self._slider_handle_center(index))
        ).x()


class _ChoiceSlider(QtWidgets.QWidget):
    sigValueChanged = QtCore.Signal(object)

    def __init__(
        self,
        choices: typing.Sequence[typing.Mapping[str, typing.Any]],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if not choices:
            raise ValueError("Choice slider requires at least one choice")
        self._choices = tuple(
            (str(choice.get("label", choice.get("value"))), choice.get("value"))
            for choice in choices
        )

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self)
        self.slider.setRange(0, len(self._choices) - 1)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(1)
        self.slider.setTickInterval(1)
        self.slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.slider.valueChanged.connect(self._slider_value_changed)
        self.setFocusProxy(self.slider)

        slider_row = QtWidgets.QWidget(self)
        self._slider_layout = QtWidgets.QHBoxLayout(slider_row)
        self._slider_layout.setContentsMargins(0, 0, 0, 0)
        self._slider_layout.setSpacing(0)
        self._slider_layout.addWidget(self.slider)

        self._label_row = _ChoiceSliderLabelRow(
            self.slider, [label for label, _value in self._choices], self
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(slider_row)
        layout.addWidget(self._label_row)
        self._sync_label_geometry()

    def count(self) -> int:
        return len(self._choices)

    def itemText(self, index: int) -> str:
        return self._choices[index][0]

    def itemData(self, index: int) -> typing.Any:
        return self._choices[index][1]

    def findData(self, value: typing.Any) -> int:
        for index, (_label, choice_value) in enumerate(self._choices):
            if choice_value == value:
                return index
        return -1

    def currentData(self) -> typing.Any:
        return self.itemData(self.slider.value())

    def setCurrentData(self, value: typing.Any) -> None:
        index = self.findData(value)
        if index < 0:
            raise ValueError(f"Unknown choice slider value: {value!r}")
        self.slider.setValue(index)

    def resizeEvent(self, event: QtGui.QResizeEvent | None) -> None:
        super().resizeEvent(event)
        self._sync_label_geometry()

    def showEvent(self, event: QtGui.QShowEvent | None) -> None:
        super().showEvent(event)
        self._sync_label_geometry()

    def event(self, event: QtCore.QEvent | None) -> bool:
        handled = super().event(event)
        if (
            event is not None
            and hasattr(self, "_label_row")
            and event.type()
            in {
                QtCore.QEvent.Type.FontChange,
                QtCore.QEvent.Type.StyleChange,
            }
        ):
            self._sync_label_geometry()
        return handled

    def _sync_label_geometry(self) -> None:
        left_margin, right_margin = self._label_row.edge_margins()
        margins = self._slider_layout.contentsMargins()
        if margins.left() != left_margin or margins.right() != right_margin:
            self._slider_layout.setContentsMargins(left_margin, 0, right_margin, 0)
            layout = self.layout()
            if layout is not None:
                layout.activate()
        self._label_row.update_label_positions()

    def _slider_value_changed(self, _value: int) -> None:
        self.sigValueChanged.emit(self.currentData())


class _SettingsRow(QtWidgets.QFrame):
    def __init__(
        self,
        *,
        scope: _Scope,
        path: str,
        control: QtWidgets.QWidget,
        parent: QtWidgets.QWidget,
    ) -> None:
        super().__init__(parent)
        self.scope = scope
        self.path = path
        self.control = control
        object_path = _object_path(path)
        self.setObjectName(f"settingsRow_{scope}_{object_path}")
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        multiline_control = isinstance(control, StylesheetListWidget)

        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(3)
        layout.setColumnStretch(1, 1)

        label = QtWidgets.QLabel(_path_title(path), self)
        label.setObjectName(f"settingsLabel_{scope}_{object_path}")
        label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        layout.addWidget(label, 0, 0)

        description = _path_description(path)
        description_label = QtWidgets.QLabel(description, self)
        description_label.setObjectName(f"settingsDescription_{scope}_{object_path}")
        description_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        description_label.setWordWrap(True)
        description_label.setEnabled(False)
        layout.addWidget(description_label, 1, 0, 1, 4 if multiline_control else 2)

        control.setObjectName(f"settingsControl_{scope}_{object_path}")
        control.setToolTip(description)
        label.setBuddy(control)
        if multiline_control:
            layout.addWidget(control, 2, 0, 1, 4)
        else:
            layout.addWidget(control, 0, 1)

        self.override_check: QtWidgets.QCheckBox | None = None
        if scope == "workspace":
            self.override_check = QtWidgets.QCheckBox(self)
            self.override_check.setObjectName(f"settingsOverride_{object_path}")
            self.override_check.setToolTip("Store this value as a workspace override.")
            layout.addWidget(self.override_check, 0, 2)

        self.action_button = QtWidgets.QToolButton(self)
        self.action_button.setObjectName(f"settingsReset_{scope}_{object_path}")
        self.action_button.setAutoRaise(True)
        self.action_button.setToolTip(
            "Reset this setting to its default value."
            if scope == "user"
            else "Use the user setting for this workspace."
        )
        icon_name = "edit-undo" if scope == "user" else "edit-clear"
        self.action_button.setIcon(QtGui.QIcon.fromTheme(icon_name))
        self.action_button.setText("Reset" if scope == "user" else "Inherit")
        layout.addWidget(self.action_button, 0, 3)


class OptionDialog(QtWidgets.QDialog):
    """Native settings dialog with immediate save and session revert."""

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("settingsDialog")
        self.setWindowTitle("Settings")
        self.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        self.setSizeGripEnabled(True)

        self._workspace_manager = self._find_workspace_manager(parent)
        self._baseline_user = erlab.interactive.options.model
        self._baseline_workspace = self._workspace_overrides()
        self._rows: dict[tuple[_Scope, str], _SettingsRow] = {}
        self._page_indexes: dict[tuple[_Scope, str], int] = {}
        self._updating = False
        self._current_scope: _Scope = "user"

        self._setup_ui()
        self._refresh_all()

    def showEvent(self, event: QtGui.QShowEvent | None) -> None:
        self.reset_session_baseline()
        super().showEvent(event)

    @staticmethod
    def _find_workspace_manager(parent: QtWidgets.QWidget | None) -> typing.Any | None:
        current: QtCore.QObject | None = parent
        required = (
            "workspace_option_overrides",
            "_set_workspace_option_overrides",
            "set_workspace_option_override",
            "clear_workspace_option_override",
            "effective_interactive_options",
        )
        while current is not None:
            if all(hasattr(current, name) for name in required):
                return current
            current = current.parent()
        return None

    def _setup_ui(self) -> None:
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(8)

        self.scope_tabs = QtWidgets.QTabBar(self)
        self.scope_tabs.setObjectName("settingsScopeTabs")
        self.scope_tabs.setExpanding(False)
        self.scope_tabs.addTab("User")
        if self._workspace_manager is not None:
            self.scope_tabs.addTab("Workspace")
        self.scope_tabs.currentChanged.connect(self._scope_changed)
        main_layout.addWidget(self.scope_tabs)

        content = QtWidgets.QWidget(self)
        content_layout = QtWidgets.QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(12)
        main_layout.addWidget(content, 1)

        self.category_list = QtWidgets.QListWidget(content)
        self.category_list.setObjectName("settingsCategoryList")
        self.category_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.category_list.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.category_list.setFixedWidth(170)
        self.category_list.currentRowChanged.connect(self._category_changed)
        content_layout.addWidget(self.category_list)

        self.page_stack = QtWidgets.QStackedWidget(content)
        self.page_stack.setObjectName("settingsPageStack")
        content_layout.addWidget(self.page_stack, 1)

        self._build_pages()
        self.category_list.setCurrentRow(0)

        bottom_bar = QtWidgets.QWidget(self)
        bottom_layout = QtWidgets.QHBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(8)
        main_layout.addWidget(bottom_bar)

        self.status_label = QtWidgets.QLabel("Saved", bottom_bar)
        self.status_label.setObjectName("settingsStatusLabel")
        self.status_label.setEnabled(False)
        bottom_layout.addWidget(self.status_label)
        bottom_layout.addStretch(1)

        self.reset_scope_button = QtWidgets.QPushButton(bottom_bar)
        self.reset_scope_button.setObjectName("settingsResetScopeButton")
        self.reset_scope_button.clicked.connect(self._reset_current_scope)
        bottom_layout.addWidget(self.reset_scope_button)

        self.revert_button = QtWidgets.QPushButton("Revert Changes", bottom_bar)
        self.revert_button.setObjectName("settingsRevertButton")
        self.revert_button.clicked.connect(self.revert_changes)
        bottom_layout.addWidget(self.revert_button)

        self.close_button = QtWidgets.QPushButton("Close", bottom_bar)
        self.close_button.setObjectName("settingsCloseButton")
        self.close_button.clicked.connect(self.close)
        bottom_layout.addWidget(self.close_button)

        close_shortcut = QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Close, self)
        close_shortcut.activated.connect(self.close)
        escape_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key.Key_Escape), self
        )
        escape_shortcut.activated.connect(self.close)

        self.resize(780, 560)
        self.setMinimumSize(640, 420)

    def _build_pages(self) -> None:
        self.category_list.clear()
        while self.page_stack.count():
            widget = self.page_stack.widget(0)
            self.page_stack.removeWidget(widget)
            if widget is not None:
                widget.deleteLater()
        self._page_indexes.clear()
        self._rows.clear()

        for category in _CATEGORY_ORDER:
            item = QtWidgets.QListWidgetItem(_category_title(category))
            item.setData(QtCore.Qt.ItemDataRole.UserRole, category)
            self.category_list.addItem(item)

        for scope in self._scopes():
            for category in _CATEGORY_ORDER:
                page = self._make_category_page(scope, category)
                self._page_indexes[(scope, category)] = self.page_stack.addWidget(page)

    def _scopes(self) -> tuple[_Scope, ...]:
        if self._workspace_manager is None:
            return ("user",)
        return ("user", "workspace")

    def _make_category_page(self, scope: _Scope, category: str) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget(self.page_stack)
        container.setObjectName(f"settingsPageContainer_{scope}_{category}")
        container_layout = QtWidgets.QVBoxLayout(container)
        style = container.style()
        right_margin = (
            style.pixelMetric(
                QtWidgets.QStyle.PixelMetric.PM_DefaultFrameWidth,
                None,
                container,
            )
            if style is not None
            else 0
        )
        container_layout.setContentsMargins(0, 0, right_margin, 0)

        scroll = QtWidgets.QScrollArea(container)
        scroll.setObjectName(f"settingsPage_{scope}_{category}")
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        container_layout.addWidget(scroll)

        page = QtWidgets.QWidget(scroll)
        layout = QtWidgets.QVBoxLayout(page)
        layout.setSpacing(0)

        paths = _leaf_paths_for_category(category, workspace_only=scope == "workspace")
        if not paths:
            empty = QtWidgets.QLabel(page)
            empty.setObjectName(f"settingsEmpty_{scope}_{category}")
            empty.setWordWrap(True)
            empty.setEnabled(False)
            empty.setText(
                "Workspace overrides are saved with ImageTool workspace files."
            )
            layout.addWidget(empty)
            layout.addStretch(1)
            scroll.setWidget(page)
            return container

        for path in paths:
            control = self._make_control(path)
            row = _SettingsRow(scope=scope, path=path, control=control, parent=page)
            self._rows[(scope, path)] = row
            self._connect_control(row)
            row.action_button.clicked.connect(
                lambda _checked=False, row=row: self._row_action(row)
            )
            if row.override_check is not None:
                row.override_check.stateChanged.connect(
                    lambda _state, row=row: self._override_changed(row)
                )
            layout.addWidget(row)
            if path != paths[-1]:
                line = QtWidgets.QFrame(page)
                line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
                line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
                layout.addWidget(line)
        layout.addStretch(1)
        scroll.setWidget(page)
        return container

    def _make_control(self, path: str) -> QtWidgets.QWidget:
        extra = _field_extra(path)
        ui_type = extra.get("ui_type")
        if ui_type == "erlabpy_colormap":
            widget = erlab.interactive.colors.ColorMapComboBox(self)
            widget.ensure_populated()
            return widget
        if ui_type == "colorlist":
            return ColorListWidget(parent=self)
        if ui_type == "matplotlib_stylesheets":
            return StylesheetListWidget(parent=self)
        if ui_type == "figure_dpi_override":
            return FigureDpiOverrideWidget(parent=self)
        if ui_type == "choice_slider":
            return _ChoiceSlider(extra.get("ui_choices", ()), self)
        if ui_type == "list" or "ui_limits" in extra:
            combo = QtWidgets.QComboBox(self)
            for choice in extra.get("ui_limits", ()):
                combo.addItem(str(choice), choice)
            return combo
        if _is_bool_path(path):
            return QtWidgets.QCheckBox(self)
        if _is_int_path(path):
            spin = QtWidgets.QSpinBox(self)
            self._configure_spinbox(spin, path)
            return spin
        if _is_float_path(path):
            spin = QtWidgets.QDoubleSpinBox(self)
            self._configure_spinbox(spin, path)
            return spin
        return QtWidgets.QLineEdit(self)

    def _configure_spinbox(
        self, spin: QtWidgets.QSpinBox | QtWidgets.QDoubleSpinBox, path: str
    ) -> None:
        def set_minimum(value: float) -> None:
            if isinstance(spin, QtWidgets.QDoubleSpinBox):
                spin.setMinimum(float(value))
            else:
                spin.setMinimum(int(value))

        def set_maximum(value: float) -> None:
            if isinstance(spin, QtWidgets.QDoubleSpinBox):
                spin.setMaximum(float(value))
            else:
                spin.setMaximum(int(value))

        constraints = _field_constraints(_model_field(path))
        if "ge" in constraints:
            set_minimum(constraints["ge"])
        elif "gt" in constraints:
            set_minimum(constraints["gt"])
        else:
            minimum = -2147483648 if isinstance(spin, QtWidgets.QSpinBox) else -1e12
            set_minimum(minimum)
        if "le" in constraints:
            set_maximum(constraints["le"])
        elif "lt" in constraints:
            set_maximum(constraints["lt"])
        else:
            maximum = 2147483647 if isinstance(spin, QtWidgets.QSpinBox) else 1e12
            set_maximum(maximum)
        extra = _field_extra(path)
        if "ui_step" in extra:
            if isinstance(spin, QtWidgets.QDoubleSpinBox):
                spin.setSingleStep(float(extra["ui_step"]))
            else:
                spin.setSingleStep(int(extra["ui_step"]))
        if isinstance(spin, QtWidgets.QDoubleSpinBox):
            spin.setDecimals(6)
        suffix = extra.get("ui_suffix")
        if suffix:
            spin.setSuffix(f" {suffix}")

    def _connect_control(self, row: _SettingsRow) -> None:
        control = row.control
        if isinstance(control, QtWidgets.QCheckBox):
            control.stateChanged.connect(
                lambda _state, row=row: self._control_changed(row)
            )
        elif isinstance(control, QtWidgets.QSpinBox | QtWidgets.QDoubleSpinBox):
            control.valueChanged.connect(
                lambda _value, row=row: self._control_changed(row)
            )
        elif isinstance(control, FigureDpiOverrideWidget):
            control.sigDpiChanged.connect(
                lambda _value, row=row: self._control_changed(row)
            )
        elif isinstance(control, _ChoiceSlider):
            control.sigValueChanged.connect(
                lambda _value, row=row: self._control_changed(row)
            )
        elif isinstance(control, QtWidgets.QComboBox):
            control.currentIndexChanged.connect(
                lambda _index, row=row: self._control_changed(row)
            )
        elif isinstance(control, ColorListWidget):
            control.sigColorChanged.connect(
                lambda _colors, row=row: self._control_changed(row)
            )
        elif isinstance(control, StylesheetListWidget):
            control.sigStylesheetsChanged.connect(
                lambda _stylesheets, row=row: self._control_changed(row)
            )
        elif isinstance(control, QtWidgets.QLineEdit):
            control.editingFinished.connect(lambda row=row: self._control_changed(row))

    def _scope_changed(self, index: int) -> None:
        self._current_scope = "workspace" if index == 1 else "user"
        self._update_visible_page()
        self._refresh_all()

    def _category_changed(self, _row: int) -> None:
        self._update_visible_page()

    def _update_visible_page(self) -> None:
        item = self.category_list.currentItem()
        if item is None:
            return
        category = item.data(QtCore.Qt.ItemDataRole.UserRole)
        index = self._page_indexes.get((self._current_scope, str(category)))
        if index is not None:
            self.page_stack.setCurrentIndex(index)
        if not hasattr(self, "reset_scope_button"):
            return
        button_text = (
            "Reset User Settings…"
            if self._current_scope == "user"
            else "Clear Workspace Overrides…"
        )
        self.reset_scope_button.setText(button_text)
        self.reset_scope_button.setToolTip(
            "Reset all user settings to their defaults."
            if self._current_scope == "user"
            else "Remove every workspace-specific setting override."
        )

    def _workspace_overrides(self) -> dict[str, typing.Any]:
        if self._workspace_manager is None:
            return {}
        overrides = self._workspace_manager.workspace_option_overrides()
        return normalize_workspace_option_overrides(overrides)

    def _effective_options(self) -> AppOptions:
        if self._workspace_manager is None:
            return erlab.interactive.options.model
        return model_with_workspace_overrides(
            erlab.interactive.options.model,
            self._workspace_overrides(),
        )

    @staticmethod
    def _keeps_raw_workspace_value(control: QtWidgets.QWidget) -> bool:
        if isinstance(control, StylesheetListWidget):
            return True
        if isinstance(control, FigureDpiOverrideWidget):
            return False
        if isinstance(control, _ChoiceSlider):
            return False
        return isinstance(control, QtWidgets.QComboBox) and not isinstance(
            control, erlab.interactive.colors.ColorMapComboBox
        )

    def _value_for_row(self, row: _SettingsRow) -> typing.Any:
        if row.scope == "user":
            return option_value(erlab.interactive.options.model, row.path)
        overrides = self._workspace_overrides()
        if row.path in overrides:
            if self._keeps_raw_workspace_value(row.control):
                return overrides[row.path]
            return option_value(self._effective_options(), row.path)
        return option_value(erlab.interactive.options.model, row.path)

    def _fallback_value_for_row(self, row: _SettingsRow) -> typing.Any:
        for model in (erlab.interactive.options.model, AppOptions()):
            with contextlib.suppress(Exception):
                return option_value(model, row.path)
        return None  # pragma: no cover

    def _control_value(self, control: QtWidgets.QWidget, path: str) -> typing.Any:
        if isinstance(control, QtWidgets.QCheckBox):
            return control.isChecked()
        if isinstance(control, QtWidgets.QSpinBox | QtWidgets.QDoubleSpinBox):
            return control.value()
        if isinstance(control, erlab.interactive.colors.ColorMapComboBox):
            return control.currentText()
        if isinstance(control, _ChoiceSlider):
            return control.currentData()
        if isinstance(control, QtWidgets.QComboBox):
            value = control.currentData()
            return control.currentText() if value is None else value
        if isinstance(control, ColorListWidget):
            return control.get_colors()
        if isinstance(control, StylesheetListWidget):
            return control.get_stylesheets()
        if isinstance(control, FigureDpiOverrideWidget):
            return control.get_dpi()
        if isinstance(control, QtWidgets.QLineEdit):
            text = control.text()
            default_value = option_value(AppOptions(), path)
            if isinstance(default_value, list):
                return [item.strip() for item in text.split(",") if item.strip()]
            return text
        return None

    def _set_control_value(
        self, control: QtWidgets.QWidget, path: str, value: typing.Any
    ) -> None:
        if isinstance(control, QtWidgets.QCheckBox):
            control.setChecked(bool(value))
            return
        if isinstance(control, QtWidgets.QSpinBox):
            control.setValue(int(value))
            return
        if isinstance(control, QtWidgets.QDoubleSpinBox):
            control.setValue(float(value))
            return
        if isinstance(control, erlab.interactive.colors.ColorMapComboBox):
            control.ensure_populated()
            control.setCurrentText(str(value))
            return
        if isinstance(control, _ChoiceSlider):
            control.setCurrentData(value)
            return
        if isinstance(control, QtWidgets.QComboBox):
            index = control.findData(value)
            if index < 0:
                index = control.findText(str(value))
            if index < 0:
                control.addItem(f"{value} (unavailable)", value)
                index = control.count() - 1
            control.setCurrentIndex(index)
            return
        if isinstance(control, ColorListWidget):
            control.set_colors(list(value or []))
            return
        if isinstance(control, StylesheetListWidget):
            control.set_stylesheets(_stylesheet_names(value))
            return
        if isinstance(control, FigureDpiOverrideWidget):
            control.set_dpi(value)
            return
        if isinstance(control, QtWidgets.QLineEdit):
            if isinstance(value, list):
                control.setText(", ".join(str(item) for item in value))
            else:
                control.setText(str(value))

    def _refresh_all(self) -> None:
        self._updating = True
        try:
            overrides = self._workspace_overrides()
            for row in self._rows.values():
                value = self._value_for_row(row)
                try:
                    self._set_control_value(row.control, row.path, value)
                except (TypeError, ValueError):
                    if row.scope != "workspace" or row.path not in overrides:
                        raise
                    self._set_control_value(
                        row.control,
                        row.path,
                        self._fallback_value_for_row(row),
                    )
                if row.scope == "workspace":
                    active = row.path in overrides
                    if row.override_check is not None:
                        row.override_check.setChecked(active)
                    row.control.setEnabled(active)
                    row.action_button.setEnabled(active)
                else:
                    default_value = option_value(AppOptions(), row.path)
                    row.action_button.setEnabled(value != default_value)
        finally:
            self._updating = False
        self._update_status()
        self._update_visible_page()

    def _control_changed(self, row: _SettingsRow) -> None:
        if self._updating:
            return
        value = self._control_value(row.control, row.path)
        if row.scope == "workspace":
            if row.override_check is None or not row.override_check.isChecked():
                return
            self._set_workspace_override(row.path, value)
        else:
            self._set_user_option(row.path, value)
        self._refresh_all()

    def _set_user_option(self, path: str, value: typing.Any) -> None:
        with contextlib.suppress(pydantic.ValidationError, ValueError, TypeError):
            erlab.interactive.options.model = option_model_with_value(
                erlab.interactive.options.model, path, value
            )

    def _set_workspace_override(self, path: str, value: typing.Any) -> None:
        if self._workspace_manager is None:
            return
        self._workspace_manager.set_workspace_option_override(path, value)

    def _clear_workspace_override(self, path: str) -> None:
        if self._workspace_manager is not None:
            self._workspace_manager.clear_workspace_option_override(path)

    def _override_changed(self, row: _SettingsRow) -> None:
        if self._updating or row.override_check is None:
            return
        if row.override_check.isChecked():
            value = self._control_value(row.control, row.path)
            self._set_workspace_override(row.path, value)
        else:
            self._clear_workspace_override(row.path)
        self._refresh_all()

    def _row_action(self, row: _SettingsRow) -> None:
        if row.scope == "workspace":
            self._clear_workspace_override(row.path)
        else:
            self._set_user_option(row.path, option_value(AppOptions(), row.path))
        self._refresh_all()

    def _reset_current_scope(self) -> None:
        if self._current_scope == "workspace":
            if self._workspace_manager is None or not self._workspace_overrides():
                return
            reply = QtWidgets.QMessageBox.question(
                self,
                "Clear Workspace Overrides",
                "Clear all workspace-specific settings?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.Cancel,
            )
            if reply == QtWidgets.QMessageBox.StandardButton.Yes:
                self._workspace_manager._set_workspace_option_overrides({})
                self._refresh_all()
            return

        reply = QtWidgets.QMessageBox.question(
            self,
            "Reset User Settings",
            "Reset all user settings to their defaults?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            erlab.interactive.options.restore()
            self._refresh_all()

    def _change_count(self) -> int:
        count = 0
        current_user = erlab.interactive.options.model
        for path in option_paths():
            if option_value(current_user, path) != option_value(
                self._baseline_user, path
            ):
                count += 1
        current_workspace = self._workspace_overrides()
        keys = set(current_workspace) | set(self._baseline_workspace)
        count += sum(
            current_workspace.get(key) != self._baseline_workspace.get(key)
            for key in keys
        )
        return count

    def _update_status(self) -> None:
        count = self._change_count()
        if count == 0:
            self.status_label.setText("Saved")
            self.revert_button.setEnabled(False)
        elif count == 1:
            self.status_label.setText("1 change saved")
            self.revert_button.setEnabled(True)
        else:
            self.status_label.setText(f"{count} changes saved")
            self.revert_button.setEnabled(True)

    @property
    def current_options(self) -> AppOptions:
        """Return the currently saved user settings."""
        return erlab.interactive.options.model

    @property
    def modified(self) -> bool:
        """Whether settings changed since this window opened."""
        return self._change_count() > 0

    @property
    def is_default(self) -> bool:
        """Whether current user settings equal built-in defaults."""
        return self.current_options.model_dump() == AppOptions().model_dump()

    @QtCore.Slot()
    def apply(self) -> None:
        """Compatibility no-op; edits are saved immediately."""
        self._refresh_all()

    @QtCore.Slot()
    def update(self) -> None:
        """Refresh displayed settings from the saved models."""
        self._refresh_all()

    @QtCore.Slot()
    def restore(self) -> None:
        """Reset user settings to defaults immediately."""
        erlab.interactive.options.restore()
        self._refresh_all()

    @QtCore.Slot()
    def reset_session_baseline(self) -> None:
        """Start a new revert session from the currently saved settings."""
        self._baseline_user = erlab.interactive.options.model
        self._baseline_workspace = self._workspace_overrides()
        self._refresh_all()

    @QtCore.Slot()
    def revert_changes(self) -> None:
        erlab.interactive.options.model = self._baseline_user
        if self._workspace_manager is not None:
            self._workspace_manager._set_workspace_option_overrides(
                dict(self._baseline_workspace)
            )
        self._refresh_all()

    def reject(self) -> None:
        super().reject()

    def accept(self) -> None:
        super().accept()
