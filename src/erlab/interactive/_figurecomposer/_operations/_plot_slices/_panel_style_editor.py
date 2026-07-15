"""Qt editors for per-panel plot-slices image and line styles."""

from __future__ import annotations

import typing

from qtpy import QtCore, QtWidgets

import erlab
from erlab.interactive._figurecomposer._line_style import (
    CONTROLLED_LINE_KW_KEYS,
    LINE_MARKER_OPTIONS,
    LINE_STYLE_OPTIONS,
    color_kw_value_from_text,
    line_kw_style_value,
    line_kw_text,
)
from erlab.interactive._figurecomposer._model._state import (
    _POWER_NORM_NAME,
    FigureOperationState,
    FigurePlotSlicesPanelStyleState,
)
from erlab.interactive._figurecomposer._norms import (
    _NORM_CHOICES,
    _cmap_base_and_reverse,
    _cmap_with_reverse,
    _effective_norm_name,
    _norm_kwarg_fields,
)
from erlab.interactive._figurecomposer._operations._plot_slices._model import (
    _MISSING,
    _effective_panel_cmap,
    _norm_clip_from_text,
    _operation_with_panel_norm_style,
    _panel_style_from_map,
    _panel_style_has_cmap_override,
    _panel_style_has_norm_override,
    _panel_style_has_overrides,
    _panel_style_map,
    _PlotSlicesPanelKey,
)
from erlab.interactive._figurecomposer._text import _dict_from_text, _format_dict
from erlab.interactive._figurecomposer._ui._color_widgets import _ColorLineEditWidget
from erlab.interactive._figurecomposer._ui._line_style import (
    configure_style_combo,
    set_style_combo_value,
    style_combo_value,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable


class _PanelStyleEditorWidget(QtWidgets.QWidget):
    """Editor for optional per-panel image style overrides."""

    sigPanelStylesChanged = QtCore.Signal(object)

    def __init__(
        self,
        operation: FigureOperationState,
        panel_keys: tuple[_PlotSlicesPanelKey, ...],
        connect_signal: Callable[
            [QtWidgets.QWidget, typing.Any, Callable[..., None]], None
        ],
        default_cmap: str,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._operation = operation
        self._panel_keys = panel_keys
        self._styles = _panel_style_map(operation)
        self._default_cmap = default_cmap
        self._updating = False

        self.panel_list = QtWidgets.QListWidget(self)
        self.panel_list.setObjectName("figureComposerPlotSlicesPanelStyleList")
        self.panel_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.panel_list.setMaximumHeight(96)
        self.panel_list.setToolTip("Select one or more panels to override.")
        for key in panel_keys:
            item = QtWidgets.QListWidgetItem(self._panel_row_text(key))
            item.setData(
                QtCore.Qt.ItemDataRole.UserRole,
                (key.map_index, key.slice_index),
            )
            self.panel_list.addItem(item)
        if self.panel_list.count():
            self.panel_list.setCurrentRow(0)

        self.cmap_override_check = QtWidgets.QCheckBox("Override colormap", self)
        self.cmap_override_check.setObjectName("figureComposerPanelCmapOverrideCheck")
        self.cmap_override_check.setToolTip(
            "Store a colormap override for the selected panels."
        )
        self.cmap_combo = erlab.interactive.colors.ColorMapComboBox(self)
        self.cmap_combo.setObjectName("figureComposerPanelCmapCombo")
        self.cmap_combo.setToolTip("Per-panel colormap override.")
        self.cmap_combo.ensure_populated()
        self.cmap_reverse_check = QtWidgets.QCheckBox("Reverse", self)
        self.cmap_reverse_check.setObjectName("figureComposerPanelCmapReverseCheck")
        self.cmap_reverse_check.setToolTip("Append _r to the per-panel colormap.")

        self.norm_override_check = QtWidgets.QCheckBox("Override norm", self)
        self.norm_override_check.setObjectName("figureComposerPanelNormOverrideCheck")
        self.norm_override_check.setToolTip(
            "Store normalization overrides for the selected panels."
        )
        self.norm_combo = QtWidgets.QComboBox(self)
        self.norm_combo.setObjectName("figureComposerPanelNormCombo")
        self.norm_combo.addItems(list(_NORM_CHOICES))
        self.norm_combo.setToolTip("Per-panel normalization class.")

        self.gamma_edit = self._number_edit("figureComposerPanelGammaEdit")
        self.vmin_edit = self._number_edit("figureComposerPanelVminEdit")
        self.vmax_edit = self._number_edit("figureComposerPanelVmaxEdit")
        self.vcenter_edit = self._number_edit("figureComposerPanelVcenterEdit")
        self.halfrange_edit = self._number_edit("figureComposerPanelHalfrangeEdit")
        self.clip_combo = QtWidgets.QComboBox(self)
        self.clip_combo.setObjectName("figureComposerPanelClipCombo")
        self.clip_combo.addItems(["inherit", "False", "True"])
        self.clip_combo.setToolTip("Per-panel norm clip argument.")
        self.norm_kwargs_edit = QtWidgets.QLineEdit(self)
        self.norm_kwargs_edit.setObjectName("figureComposerPanelNormKwargsEdit")
        self.norm_kwargs_edit.setToolTip(
            "Extra keyword arguments for selected panel norm constructors."
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.panel_list)

        cmap_row = QtWidgets.QHBoxLayout()
        cmap_row.setContentsMargins(0, 0, 0, 0)
        cmap_row.addWidget(self.cmap_override_check)
        cmap_row.addWidget(self.cmap_combo, 1)
        cmap_row.addWidget(self.cmap_reverse_check)
        layout.addLayout(cmap_row)

        norm_row = QtWidgets.QHBoxLayout()
        norm_row.setContentsMargins(0, 0, 0, 0)
        norm_row.addWidget(self.norm_override_check)
        norm_row.addWidget(self.norm_combo, 1)
        layout.addLayout(norm_row)

        numbers = QtWidgets.QGridLayout()
        numbers.setContentsMargins(0, 0, 0, 0)
        numbers.setHorizontalSpacing(6)
        numbers.setVerticalSpacing(4)
        for row, (label, widget) in enumerate(
            (
                ("Gamma", self.gamma_edit),
                ("vmin", self.vmin_edit),
                ("vmax", self.vmax_edit),
                ("vcenter", self.vcenter_edit),
                ("halfrange", self.halfrange_edit),
                ("Clip", self.clip_combo),
                ("Norm kwargs", self.norm_kwargs_edit),
            )
        ):
            numbers.addWidget(QtWidgets.QLabel(label, self), row, 0)
            numbers.addWidget(widget, row, 1)
        layout.addLayout(numbers)

        connect_signal(self, self.panel_list.itemSelectionChanged, self._sync_controls)
        connect_signal(
            self, self.cmap_override_check.stateChanged, self._cmap_override_changed
        )
        connect_signal(self, self.cmap_combo.activated, self._cmap_changed)
        connect_signal(
            self, self.cmap_reverse_check.stateChanged, self._cmap_reverse_changed
        )
        connect_signal(
            self, self.norm_override_check.stateChanged, self._norm_override_changed
        )
        connect_signal(self, self.norm_combo.activated, self._norm_changed)
        for attr, edit in (
            ("norm_gamma", self.gamma_edit),
            ("vmin", self.vmin_edit),
            ("vmax", self.vmax_edit),
            ("vcenter", self.vcenter_edit),
            ("halfrange", self.halfrange_edit),
        ):
            connect_signal(
                self,
                edit.editingFinished,
                lambda attr=attr, edit=edit: self._number_changed(attr, edit),
            )
        connect_signal(self, self.clip_combo.activated, self._clip_changed)
        connect_signal(
            self, self.norm_kwargs_edit.editingFinished, self._norm_kwargs_changed
        )
        self._sync_controls()

    def styles(self) -> tuple[FigurePlotSlicesPanelStyleState, ...]:
        valid_keys = {(key.map_index, key.slice_index) for key in self._panel_keys}
        return tuple(
            self._styles[key]
            for key in sorted(self._styles)
            if key in valid_keys and _panel_style_has_overrides(self._styles[key])
        )

    @staticmethod
    def _number_edit(object_name: str) -> QtWidgets.QLineEdit:
        edit = QtWidgets.QLineEdit()
        edit.setObjectName(object_name)
        edit.setToolTip("Leave blank to inherit the global value.")
        return edit

    def _panel_row_text(self, key: _PlotSlicesPanelKey) -> str:
        style = _panel_style_from_map(self._styles, key)
        parts = [key.label]
        if _panel_style_has_cmap_override(style):
            parts.append(
                _effective_panel_cmap(
                    self._operation,
                    style,
                    default_cmap=self._default_cmap,
                )
            )
        if _panel_style_has_norm_override(style):
            norm_operation = _operation_with_panel_norm_style(self._operation, style)
            parts.append(_effective_norm_name(norm_operation.norm_name))
        return " | ".join(parts)

    def _selected_keys(self) -> tuple[_PlotSlicesPanelKey, ...]:
        by_index = {(key.map_index, key.slice_index): key for key in self._panel_keys}
        keys: list[_PlotSlicesPanelKey] = []
        for item in self.panel_list.selectedItems():
            raw_key = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if raw_key in by_index:
                keys.append(by_index[raw_key])
        if keys:
            return tuple(keys)
        if self._panel_keys:
            return (self._panel_keys[0],)
        return ()

    def _selected_styles(self) -> tuple[FigurePlotSlicesPanelStyleState, ...]:
        return tuple(
            _panel_style_from_map(self._styles, key) for key in self._selected_keys()
        )

    @staticmethod
    def _common_value(values: tuple[typing.Any, ...]) -> typing.Any:
        if not values:
            return None
        first = values[0]
        if all(value == first for value in values[1:]):
            return first
        return _MISSING

    def _sync_controls(self) -> None:
        if self._updating:
            return
        self._updating = True
        try:
            styles = self._selected_styles()
            cmap_override = self._common_value(
                tuple(_panel_style_has_cmap_override(style) for style in styles)
            )
            self._set_check_state(self.cmap_override_check, cmap_override)
            cmap_enabled = cmap_override is True
            self.cmap_combo.setEnabled(cmap_enabled)
            self.cmap_reverse_check.setEnabled(cmap_enabled)
            self._set_cmap_value(styles)

            norm_override = self._common_value(
                tuple(_panel_style_has_norm_override(style) for style in styles)
            )
            self._set_check_state(self.norm_override_check, norm_override)
            norm_enabled = norm_override is True
            self.norm_combo.setEnabled(norm_enabled)
            norm_name = self._set_norm_value(styles)
            self._sync_norm_fields(styles, norm_enabled, norm_name)
        finally:
            self._updating = False

    @staticmethod
    def _set_check_state(check: QtWidgets.QCheckBox, value: object) -> None:
        with QtCore.QSignalBlocker(check):
            check.setTristate(value is _MISSING)
            if value is _MISSING:
                check.setCheckState(QtCore.Qt.CheckState.PartiallyChecked)
            else:
                check.setCheckState(
                    QtCore.Qt.CheckState.Checked
                    if value is True
                    else QtCore.Qt.CheckState.Unchecked
                )

    def _set_cmap_value(
        self, styles: tuple[FigurePlotSlicesPanelStyleState, ...]
    ) -> None:
        values = tuple(
            _cmap_base_and_reverse(style.cmap)[0]
            if style.cmap is not None
            else _effective_panel_cmap(
                self._operation,
                style,
                default_cmap=self._default_cmap,
            )
            for style in styles
        )
        reversed_values = tuple(
            _cmap_base_and_reverse(style.cmap)[1] if style.cmap is not None else False
            for style in styles
        )
        value = self._common_value(values)
        reversed_value = self._common_value(reversed_values)
        with QtCore.QSignalBlocker(self.cmap_combo):
            if value is _MISSING:
                self._set_combo_mixed(self.cmap_combo)
            else:
                self._remove_combo_mixed(self.cmap_combo)
                self.cmap_combo.setCurrentText(str(value))
        self._set_check_state(self.cmap_reverse_check, reversed_value)

    def _set_norm_value(
        self, styles: tuple[FigurePlotSlicesPanelStyleState, ...]
    ) -> str | None:
        values = tuple(
            _effective_norm_name(
                _operation_with_panel_norm_style(self._operation, style).norm_name
            )
            for style in styles
        )
        value = self._common_value(values)
        with QtCore.QSignalBlocker(self.norm_combo):
            if value is _MISSING:
                self._set_combo_mixed(self.norm_combo)
                return None
            self._remove_combo_mixed(self.norm_combo)
            self.norm_combo.setCurrentText(str(value))
            return str(value)

    def _sync_norm_fields(
        self,
        styles: tuple[FigurePlotSlicesPanelStyleState, ...],
        enabled: bool,
        norm_name: str | None,
    ) -> None:
        norm_fields = set(_norm_kwarg_fields(norm_name)) if norm_name else set()
        for attr, edit in (
            ("norm_gamma", self.gamma_edit),
            ("vmin", self.vmin_edit),
            ("vmax", self.vmax_edit),
            ("vcenter", self.vcenter_edit),
            ("halfrange", self.halfrange_edit),
        ):
            field_name = "gamma" if attr == "norm_gamma" else attr
            field_enabled = enabled and field_name in norm_fields
            edit.setEnabled(field_enabled)
            values = tuple(getattr(style, attr) for style in styles)
            value = self._common_value(values)
            with QtCore.QSignalBlocker(edit):
                edit.setText("" if value in (None, _MISSING) else f"{value:g}")
                edit.setPlaceholderText(
                    "(multiple values)" if value is _MISSING else "inherit"
                )
                edit.setModified(False)
        self.clip_combo.setEnabled(enabled and "clip" in norm_fields)
        clip_values = tuple(style.norm_clip for style in styles)
        clip_value = self._common_value(clip_values)
        with QtCore.QSignalBlocker(self.clip_combo):
            if clip_value is _MISSING:
                self._set_combo_mixed(self.clip_combo)
            else:
                self._remove_combo_mixed(self.clip_combo)
                self.clip_combo.setCurrentText(
                    "inherit" if clip_value is None else str(clip_value)
                )
        self.norm_kwargs_edit.setEnabled(enabled)
        kwargs_values = tuple(style.norm_kwargs for style in styles)
        kwargs_value = self._common_value(kwargs_values)
        with QtCore.QSignalBlocker(self.norm_kwargs_edit):
            self.norm_kwargs_edit.setText(
                "" if kwargs_value in (None, _MISSING) else _format_dict(kwargs_value)
            )
            self.norm_kwargs_edit.setPlaceholderText(
                "(multiple values)" if kwargs_value is _MISSING else "optional"
            )
            self.norm_kwargs_edit.setModified(False)

    @staticmethod
    def _set_combo_mixed(combo: QtWidgets.QComboBox) -> None:
        if combo.findData(_MISSING) < 0:
            combo.insertItem(0, "(multiple values)", _MISSING)
            item = typing.cast("typing.Any", combo.model()).item(0)
            if item is not None:
                item.setEnabled(False)
        combo.setCurrentIndex(0)

    @staticmethod
    def _remove_combo_mixed(combo: QtWidgets.QComboBox) -> None:
        index = combo.findData(_MISSING)
        if index >= 0:
            combo.removeItem(index)

    def _cmap_override_changed(self, state: int) -> None:
        check_state = QtCore.Qt.CheckState(state)
        if self._updating or check_state == QtCore.Qt.CheckState.PartiallyChecked:
            return
        if check_state == QtCore.Qt.CheckState.Checked:
            cmap = self._operation.cmap or self._default_cmap
            self._update_selected_styles({"cmap": cmap})
        else:
            self._update_selected_styles({"cmap": None})

    def _cmap_override_active(self) -> bool:
        return self.cmap_override_check.checkState() == QtCore.Qt.CheckState.Checked

    def _cmap_changed(self, _index: int) -> None:
        if (
            self._updating
            or not self._cmap_override_active()
            or self.cmap_combo.currentData() is _MISSING
        ):
            return
        base = self.cmap_combo.current_matplotlib_name()
        reverse = self.cmap_reverse_check.checkState() == QtCore.Qt.CheckState.Checked
        self._update_selected_styles({"cmap": _cmap_with_reverse(base, reverse)})

    def _cmap_reverse_changed(self, state: int) -> None:
        check_state = QtCore.Qt.CheckState(state)
        if (
            self._updating
            or not self._cmap_override_active()
            or check_state == QtCore.Qt.CheckState.PartiallyChecked
        ):
            return
        base = self.cmap_combo.current_matplotlib_name()
        if self.cmap_combo.currentData() is _MISSING:
            base = self._operation.cmap or self._default_cmap
        reverse = check_state == QtCore.Qt.CheckState.Checked
        self._update_selected_styles({"cmap": _cmap_with_reverse(base, reverse)})

    def _norm_override_changed(self, state: int) -> None:
        check_state = QtCore.Qt.CheckState(state)
        if self._updating or check_state == QtCore.Qt.CheckState.PartiallyChecked:
            return
        if check_state == QtCore.Qt.CheckState.Checked:
            self._update_selected_styles(
                {"norm_name": self._operation.norm_name or _POWER_NORM_NAME}
            )
        else:
            self._update_selected_styles(
                {
                    "norm_name": None,
                    "norm_gamma": None,
                    "norm_clip": None,
                    "norm_kwargs": {},
                    "vmin": None,
                    "vmax": None,
                    "vcenter": None,
                    "halfrange": None,
                }
            )

    def _norm_override_active(self) -> bool:
        return self.norm_override_check.checkState() == QtCore.Qt.CheckState.Checked

    def _norm_changed(self, _index: int) -> None:
        if (
            self._updating
            or not self._norm_override_active()
            or self.norm_combo.currentData() is _MISSING
        ):
            return
        self._update_selected_styles({"norm_name": self.norm_combo.currentText()})

    def _number_changed(self, attr: str, edit: QtWidgets.QLineEdit) -> None:
        if (
            self._updating
            or not self._norm_override_active()
            or (edit.placeholderText() == "(multiple values)" and not edit.isModified())
        ):
            return
        text = edit.text().strip()
        self._update_selected_styles({attr: float(text) if text else None})

    def _clip_changed(self, _index: int) -> None:
        if (
            self._updating
            or not self._norm_override_active()
            or self.clip_combo.currentData() is _MISSING
        ):
            return
        text = self.clip_combo.currentText()
        self._update_selected_styles({"norm_clip": _norm_clip_from_text(text)})

    def _norm_kwargs_changed(self) -> None:
        if (
            self._updating
            or not self._norm_override_active()
            or (
                self.norm_kwargs_edit.placeholderText() == "(multiple values)"
                and not self.norm_kwargs_edit.isModified()
            )
        ):
            return
        self._update_selected_styles(
            {"norm_kwargs": _dict_from_text(self.norm_kwargs_edit.text())}
        )

    def _update_selected_styles(self, updates: dict[str, typing.Any]) -> None:
        for key in self._selected_keys():
            style_key = (key.map_index, key.slice_index)
            style = self._styles.get(
                style_key,
                FigurePlotSlicesPanelStyleState(
                    map_index=key.map_index,
                    slice_index=key.slice_index,
                ),
            )
            next_style = style.model_copy(update=updates)
            if _panel_style_has_overrides(next_style):
                self._styles[style_key] = next_style
            else:
                self._styles.pop(style_key, None)
        styles = self.styles()
        self._operation = self._operation.model_copy(
            update={
                "panel_styles_enabled": bool(styles),
                "panel_styles": styles,
            }
        )
        self.sigPanelStylesChanged.emit(styles)
        self._sync_rows()
        self._sync_controls()

    def _sync_rows(self) -> None:
        for row, key in enumerate(self._panel_keys):
            item = self.panel_list.item(row)
            if item is not None:
                item.setText(self._panel_row_text(key))


class _PanelLineStyleEditorWidget(QtWidgets.QWidget):
    """Editor for optional per-panel 1D line style overrides."""

    sigPanelStylesChanged = QtCore.Signal(object)

    def __init__(
        self,
        operation: FigureOperationState,
        panel_keys: tuple[_PlotSlicesPanelKey, ...],
        connect_signal: Callable[
            [QtWidgets.QWidget, typing.Any, Callable[..., None]], None
        ],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._operation = operation
        self._panel_keys = panel_keys
        self._styles = _panel_style_map(operation)
        self._updating = False

        self.panel_list = QtWidgets.QListWidget(self)
        self.panel_list.setObjectName("figureComposerPlotSlicesPanelLineStyleList")
        self.panel_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.panel_list.setMaximumHeight(96)
        for key in panel_keys:
            item = QtWidgets.QListWidgetItem(self._panel_row_text(key))
            item.setData(
                QtCore.Qt.ItemDataRole.UserRole,
                (key.map_index, key.slice_index),
            )
            self.panel_list.addItem(item)
        if self.panel_list.count():
            self.panel_list.setCurrentRow(0)

        self.color_edit = _ColorLineEditWidget(parent=self)
        self.color_edit.setLineEditObjectName("figureComposerPanelLineColorEdit")
        self.color_edit.setColorButtonObjectName("figureComposerPanelLineColorButton")
        self.style_combo = QtWidgets.QComboBox(self)
        self.style_combo.setObjectName("figureComposerPanelLineStyleCombo")
        configure_style_combo(self.style_combo, LINE_STYLE_OPTIONS, None)
        self.width_edit = self._line_edit("figureComposerPanelLineWidthEdit")
        self.marker_combo = QtWidgets.QComboBox(self)
        self.marker_combo.setObjectName("figureComposerPanelLineMarkerCombo")
        configure_style_combo(self.marker_combo, LINE_MARKER_OPTIONS, None)
        self.marker_size_edit = self._line_edit("figureComposerPanelLineMarkerSizeEdit")
        self.marker_face_edit = _ColorLineEditWidget(parent=self)
        self.marker_face_edit.setLineEditObjectName(
            "figureComposerPanelLineMarkerFaceEdit"
        )
        self.marker_face_edit.setColorButtonObjectName(
            "figureComposerPanelLineMarkerFaceButton"
        )
        self.marker_edge_edit = _ColorLineEditWidget(parent=self)
        self.marker_edge_edit.setLineEditObjectName(
            "figureComposerPanelLineMarkerEdgeEdit"
        )
        self.marker_edge_edit.setColorButtonObjectName(
            "figureComposerPanelLineMarkerEdgeButton"
        )
        self.line_kwargs_edit = self._line_edit("figureComposerPanelLineKwEdit")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.panel_list)
        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )
        form.addRow("Color", self.color_edit)
        form.addRow("Line style", self.style_combo)
        form.addRow("Line width", self.width_edit)
        form.addRow("Marker", self.marker_combo)
        form.addRow("Marker size", self.marker_size_edit)
        form.addRow("Marker face", self.marker_face_edit)
        form.addRow("Marker edge", self.marker_edge_edit)
        form.addRow("Line kwargs", self.line_kwargs_edit)
        layout.addLayout(form)

        connect_signal(self, self.panel_list.itemSelectionChanged, self._sync_controls)
        connect_signal(
            self,
            self.color_edit.editingFinished,
            lambda: self._line_kw_changed(
                "color",
                color_kw_value_from_text(self.color_edit.text()),
                aliases=("c",),
            ),
        )
        connect_signal(
            self,
            self.style_combo.activated,
            lambda _index: self._line_kw_changed(
                "linestyle",
                style_combo_value(self.style_combo),
                aliases=("ls",),
            ),
        )
        connect_signal(
            self,
            self.width_edit.editingFinished,
            lambda: self._line_kw_changed(
                "linewidth",
                self._optional_float(self.width_edit.text()),
                aliases=("lw",),
            ),
        )
        connect_signal(
            self,
            self.marker_combo.activated,
            lambda _index: self._line_kw_changed(
                "marker", style_combo_value(self.marker_combo)
            ),
        )
        connect_signal(
            self,
            self.marker_size_edit.editingFinished,
            lambda: self._line_kw_changed(
                "markersize",
                self._optional_float(self.marker_size_edit.text()),
                aliases=("ms",),
            ),
        )
        connect_signal(
            self,
            self.marker_face_edit.editingFinished,
            lambda: self._line_kw_changed(
                "markerfacecolor",
                color_kw_value_from_text(self.marker_face_edit.text()),
                aliases=("mfc",),
            ),
        )
        connect_signal(
            self,
            self.marker_edge_edit.editingFinished,
            lambda: self._line_kw_changed(
                "markeredgecolor",
                color_kw_value_from_text(self.marker_edge_edit.text()),
                aliases=("mec",),
            ),
        )
        connect_signal(
            self, self.line_kwargs_edit.editingFinished, self._extra_line_kw_changed
        )
        self._sync_controls()

    def styles(self) -> tuple[FigurePlotSlicesPanelStyleState, ...]:
        valid_keys = {(key.map_index, key.slice_index) for key in self._panel_keys}
        return tuple(
            self._styles[key]
            for key in sorted(self._styles)
            if key in valid_keys and _panel_style_has_overrides(self._styles[key])
        )

    @staticmethod
    def _line_edit(object_name: str) -> QtWidgets.QLineEdit:
        edit = QtWidgets.QLineEdit()
        edit.setObjectName(object_name)
        edit.setPlaceholderText("inherit")
        return edit

    @staticmethod
    def _optional_float(text: str) -> float | None:
        stripped = text.strip()
        return None if not stripped else float(stripped)

    def _panel_row_text(self, key: _PlotSlicesPanelKey) -> str:
        style = _panel_style_from_map(self._styles, key)
        color = line_kw_text(
            self._operation.model_copy(update={"line_kw": style.line_kw}),
            "color",
            "c",
        )
        return key.label if not color else f"{key.label} | {color}"

    def _selected_keys(self) -> tuple[_PlotSlicesPanelKey, ...]:
        by_index = {(key.map_index, key.slice_index): key for key in self._panel_keys}
        keys: list[_PlotSlicesPanelKey] = []
        for item in self.panel_list.selectedItems():
            raw_key = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if raw_key in by_index:
                keys.append(by_index[raw_key])
        if keys:
            return tuple(keys)
        if self._panel_keys:
            return (self._panel_keys[0],)
        return ()

    def _selected_styles(self) -> tuple[FigurePlotSlicesPanelStyleState, ...]:
        return tuple(
            _panel_style_from_map(self._styles, key) for key in self._selected_keys()
        )

    @staticmethod
    def _common_value(values: tuple[typing.Any, ...]) -> typing.Any:
        if not values:
            return None
        first = values[0]
        if all(value == first for value in values[1:]):
            return first
        return _MISSING

    def _line_value(self, style: FigurePlotSlicesPanelStyleState, *keys: str) -> str:
        operation = self._operation.model_copy(update={"line_kw": style.line_kw})
        return line_kw_text(operation, *keys)

    def _line_style_value(
        self, style: FigurePlotSlicesPanelStyleState, *keys: str
    ) -> str | None:
        operation = self._operation.model_copy(update={"line_kw": style.line_kw})
        return line_kw_style_value(operation, *keys)

    def _sync_controls(self) -> None:
        if self._updating:
            return
        self._updating = True
        try:
            styles = self._selected_styles()
            self._set_color_widget(
                self.color_edit,
                self._common_value(
                    tuple(self._line_value(style, "color", "c") for style in styles)
                ),
            )
            self._set_combo(
                self.style_combo,
                self._common_value(
                    tuple(
                        self._line_style_value(style, "linestyle", "ls")
                        for style in styles
                    )
                ),
            )
            self._set_line_edit(
                self.width_edit,
                self._common_value(
                    tuple(
                        self._line_value(style, "linewidth", "lw") for style in styles
                    )
                ),
            )
            self._set_combo(
                self.marker_combo,
                self._common_value(
                    tuple(self._line_style_value(style, "marker") for style in styles)
                ),
            )
            self._set_line_edit(
                self.marker_size_edit,
                self._common_value(
                    tuple(
                        self._line_value(style, "markersize", "ms") for style in styles
                    )
                ),
            )
            self._set_color_widget(
                self.marker_face_edit,
                self._common_value(
                    tuple(
                        self._line_value(style, "markerfacecolor", "mfc")
                        for style in styles
                    )
                ),
            )
            self._set_color_widget(
                self.marker_edge_edit,
                self._common_value(
                    tuple(
                        self._line_value(style, "markeredgecolor", "mec")
                        for style in styles
                    )
                ),
            )
            self._set_line_edit(
                self.line_kwargs_edit,
                self._common_value(
                    tuple(
                        _format_dict(self._extra_line_kw(style.line_kw))
                        for style in styles
                    )
                ),
            )
        finally:
            self._updating = False

    @staticmethod
    def _set_line_edit(edit: QtWidgets.QLineEdit, value: typing.Any) -> None:
        with QtCore.QSignalBlocker(edit):
            edit.setText("" if value in (None, _MISSING) else str(value))
            edit.setPlaceholderText(
                "(multiple values)" if value is _MISSING else "inherit"
            )
            edit.setModified(False)

    def _set_color_widget(
        self, widget: _ColorLineEditWidget, value: typing.Any
    ) -> None:
        self._set_line_edit(widget.line_edit, value)
        widget.setText("" if value in (None, _MISSING) else str(value))

    @staticmethod
    def _set_combo(combo: QtWidgets.QComboBox, value: typing.Any) -> None:
        with QtCore.QSignalBlocker(combo):
            if value is _MISSING:
                _PanelStyleEditorWidget._set_combo_mixed(combo)
            else:
                _PanelStyleEditorWidget._remove_combo_mixed(combo)
                set_style_combo_value(combo, None if value is None else str(value))

    @classmethod
    def _extra_line_kw(cls, line_kw: dict[str, typing.Any]) -> dict[str, typing.Any]:
        return {
            key: value
            for key, value in line_kw.items()
            if key not in CONTROLLED_LINE_KW_KEYS
        }

    def _line_kw_changed(
        self,
        key: str,
        value: typing.Any,
        *,
        aliases: tuple[str, ...] = (),
    ) -> None:
        if self._updating:
            return
        self._update_selected_line_kw({key: value}, clear_keys=(key, *aliases))

    def _extra_line_kw_changed(self) -> None:
        if self._updating or (
            self.line_kwargs_edit.placeholderText() == "(multiple values)"
            and not self.line_kwargs_edit.isModified()
        ):
            return
        extra_kwargs = _dict_from_text(self.line_kwargs_edit.text())
        for key in CONTROLLED_LINE_KW_KEYS:
            extra_kwargs.pop(key, None)
        self._update_selected_extra_line_kw(extra_kwargs)

    def _update_selected_line_kw(
        self,
        updates: dict[str, typing.Any],
        *,
        clear_keys: tuple[str, ...],
    ) -> None:
        for panel_key in self._selected_keys():
            style_key = (panel_key.map_index, panel_key.slice_index)
            style = self._styles.get(
                style_key,
                FigurePlotSlicesPanelStyleState(
                    map_index=panel_key.map_index,
                    slice_index=panel_key.slice_index,
                ),
            )
            line_kw = dict(style.line_kw)
            for clear_key in clear_keys:
                line_kw.pop(clear_key, None)
            line_kw.update(
                {
                    key: value
                    for key, value in updates.items()
                    if value not in (None, "")
                }
            )
            self._replace_style(
                style_key,
                style.model_copy(update={"line_kw": line_kw}),
            )
        self._emit_styles_changed()

    def _update_selected_extra_line_kw(
        self, extra_kwargs: dict[str, typing.Any]
    ) -> None:
        for panel_key in self._selected_keys():
            style_key = (panel_key.map_index, panel_key.slice_index)
            style = self._styles.get(
                style_key,
                FigurePlotSlicesPanelStyleState(
                    map_index=panel_key.map_index,
                    slice_index=panel_key.slice_index,
                ),
            )
            line_kw = {
                key: value
                for key, value in style.line_kw.items()
                if key in CONTROLLED_LINE_KW_KEYS
            }
            line_kw.update(extra_kwargs)
            self._replace_style(
                style_key,
                style.model_copy(update={"line_kw": line_kw}),
            )
        self._emit_styles_changed()

    def _replace_style(
        self,
        style_key: tuple[int, int],
        style: FigurePlotSlicesPanelStyleState,
    ) -> None:
        if _panel_style_has_overrides(style):
            self._styles[style_key] = style
        else:
            self._styles.pop(style_key, None)

    def _emit_styles_changed(self) -> None:
        styles = self.styles()
        self._operation = self._operation.model_copy(
            update={
                "panel_styles_enabled": bool(styles),
                "panel_styles": styles,
            }
        )
        self.sigPanelStylesChanged.emit(styles)
        self._sync_rows()
        self._sync_controls()

    def _sync_rows(self) -> None:
        for row, key in enumerate(self._panel_keys):
            item = self.panel_list.item(row)
            if item is not None:
                item.setText(self._panel_row_text(key))
