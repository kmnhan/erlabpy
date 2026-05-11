"""Reusable widgets used by ImageTool dialogs."""

from __future__ import annotations

import typing

import numpy as np
import numpy.typing as npt
from qtpy import QtCore, QtWidgets

import erlab

__all__ = ["CoordinateEditorWidget", "CoordinateGridWidget"]


class CoordinateGridWidget(QtWidgets.QWidget):
    """Edit coordinate values using start/end-or-delta controls and a table.

    Parameters
    ----------
    values
        Initial reference coordinate values. These values seed the spin boxes and table
        when the widget is reset.
    editable_count
        If `True`, show a ``Count`` spin box and let the generated table length differ
        from the reference coordinate length. This is used for interpolation targets.
    preserve_shape
        If `True`, generated and edited values keep the shape of ``values``. If
        `False`, values are flattened to one dimension.
    require_complete
        If `True`, every table row must contain a numeric value. If `False`, blank
        table cells keep the corresponding reference value.
    numeric_reference
        If `True`, coerce reference values to ``float64``. Use this for tools that only
        accept numeric target coordinates.
    disable_singleton_controls
        If `True`, disable the start/end-or-delta controls when the reference
        coordinate has a single value.
    reset_table_to_reference
        If `True`, reset fills the table with the reference coordinate values. If
        `False`, reset fills the table from the current start/end-or-delta controls.
    update_table_on_mode_changed
        If `True`, changing between ``End`` and ``Delta`` immediately regenerates the
        table. If `False`, the table is left unchanged until another update trigger.
    auto_minimum_width
        If `True`, adjust the widget minimum width to fit the coordinate table.
    connect_reset_button
        If `True`, the reset button is connected directly to :meth:`reset`. Set this to
        `False` when a containing widget needs to run additional reset logic.
    """

    def __init__(
        self,
        values: npt.ArrayLike,
        *,
        editable_count: bool = False,
        preserve_shape: bool = True,
        require_complete: bool = False,
        numeric_reference: bool = False,
        disable_singleton_controls: bool = True,
        reset_table_to_reference: bool = True,
        update_table_on_mode_changed: bool = False,
        auto_minimum_width: bool = False,
        connect_reset_button: bool = True,
    ) -> None:
        super().__init__()
        self._editable_count = editable_count
        self._preserve_shape = preserve_shape
        self._require_complete = require_complete
        self._numeric_reference = numeric_reference
        self._disable_singleton_controls = disable_singleton_controls
        self._reset_table_to_reference = reset_table_to_reference
        self._update_table_on_mode_changed = update_table_on_mode_changed
        self._auto_minimum_width = auto_minimum_width
        self._reference_coord = self._coerce_reference(values)
        self.init_ui(connect_reset_button=connect_reset_button)
        self.reset()

    def init_ui(self, *, connect_reset_button: bool) -> None:
        layout = QtWidgets.QFormLayout(self)

        self.spin0 = erlab.interactive.utils.BetterSpinBox(compact=False, trim="0")
        self.spin0.valueChanged.connect(self.update_table)
        layout.addRow("Start", self.spin0)

        self.spin1 = erlab.interactive.utils.BetterSpinBox(compact=False, trim="0")
        self.spin1.valueChanged.connect(self.update_table)

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["End", "Delta"])
        self.mode_combo.setCurrentIndex(0)
        self.mode_combo.currentTextChanged.connect(self.mode_changed)
        layout.addRow(self.mode_combo, self.spin1)

        self.count_spin = QtWidgets.QSpinBox()
        self.count_spin.setRange(1, 2_147_483_647)
        self.count_spin.valueChanged.connect(self.update_table)
        if self._editable_count:
            layout.addRow("Count", self.count_spin)

        self.reset_btn = QtWidgets.QPushButton("Reset")
        if connect_reset_button:
            self.reset_btn.clicked.connect(self.reset)
        layout.addRow(self.reset_btn)

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(1)
        typing.cast("QtWidgets.QHeaderView", self.table.horizontalHeader()).setVisible(
            False
        )
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setSortingEnabled(False)
        self.table.setAlternatingRowColors(True)
        layout.addRow(self.table)

    @property
    def reference_coord(self) -> npt.NDArray:
        return self._reference_coord

    @property
    def _reference_values(self) -> npt.NDArray:
        return np.atleast_1d(self._reference_coord)

    @property
    def _value_count(self) -> int:
        if self._editable_count:
            return self.count_spin.value()
        return np.atleast_1d(self._reference_coord).size

    def _coerce_reference(self, values: npt.ArrayLike) -> npt.NDArray:
        if self._numeric_reference:
            values = np.asarray(values, dtype=np.float64)
        else:
            values = np.asarray(values).copy()
        if self._preserve_shape:
            return values
        return values.reshape(-1)

    def _current_values(self, mode: str | None = None) -> npt.NDArray:
        count = self._value_count
        start = self.spin0.value()
        stop_or_step = self.spin1.value()
        if (mode or self.mode_combo.currentText()) == "Delta":
            stop_or_step = start + stop_or_step * (count - 1)
        values = np.linspace(start, stop_or_step, count)
        if self._preserve_shape:
            return values.reshape(self._reference_values.shape)
        return values

    @property
    def _current_values_end(self) -> npt.NDArray:
        return self._current_values("End")

    @property
    def _current_values_delta(self) -> npt.NDArray:
        return self._current_values("Delta")

    @property
    def new_coord(self) -> npt.NDArray:
        if self._require_complete:
            values = np.empty(self.table.rowCount(), dtype=np.float64)
            flat_values = values
        else:
            values = self._reference_coord.copy()
            flat_values = values.reshape(-1)

        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is None or not item.text().strip():
                if self._require_complete:
                    raise ValueError(f"Missing value in row {row}.")
                continue
            try:
                flat_values[row] = float(item.text())
            except Exception as exc:
                raise ValueError(f"Invalid value in row {row}: {item.text()}") from exc
        return values

    def set_reference_coord(self, values: npt.ArrayLike) -> None:
        self._reference_coord = self._coerce_reference(values)
        self.reset()

    def set_old_coord(self, values: npt.ArrayLike) -> None:
        self.set_reference_coord(values)

    @QtCore.Slot()
    def mode_changed(self) -> None:
        if self._value_count < 2:
            return
        with QtCore.QSignalBlocker(self.spin1):
            match self.mode_combo.currentText():
                case "End":
                    self.spin1.setValue(
                        float(self._current_values_delta.reshape(-1)[-1])
                    )
                case _:
                    arr = self._current_values_end.reshape(-1)
                    self.spin1.setValue(float(arr[1] - arr[0]))
        if self._update_table_on_mode_changed:
            self.update_table()

    @QtCore.Slot()
    def reset(self) -> None:
        coord = self._reference_values
        count = max(int(coord.size), 1)
        is_scalar = self._disable_singleton_controls and count == 1
        self.spin0.setDisabled(is_scalar)
        self.spin1.setDisabled(is_scalar)
        self.mode_combo.setDisabled(is_scalar)

        with (
            QtCore.QSignalBlocker(self.spin0),
            QtCore.QSignalBlocker(self.spin1),
            QtCore.QSignalBlocker(self.count_spin),
        ):
            self.count_spin.setValue(count)
            if not is_scalar:
                decimals = erlab.utils.array.unique_decimals(coord)
                self.spin0.setDecimals(decimals)
                self.spin1.setDecimals(decimals)
                use_reference_range = erlab.utils.array.is_uniform_spaced(coord)
                if self._reset_table_to_reference and not use_reference_range:
                    self.spin0.setValue(0.0)
                    self.spin1.setValue(0.0)
                else:
                    flat_coord = coord.reshape(-1)
                    self.spin0.setValue(float(flat_coord[0]))
                    if self.mode_combo.currentText() == "Delta":
                        if count > 1:
                            self.spin1.setValue(float(flat_coord[1] - flat_coord[0]))
                        else:
                            self.spin1.setValue(1.0)
                    else:
                        self.spin1.setValue(float(flat_coord[-1]))

        if self._reset_table_to_reference:
            self._set_table_values(coord)
        else:
            self.update_table()

    @QtCore.Slot()
    def update_table(self) -> None:
        self._set_table_values(self._current_values())

    def _set_table_values(self, values: npt.NDArray) -> None:
        flat_values = np.ravel(values)
        self.table.setRowCount(len(flat_values))
        self.table.setVerticalHeaderLabels([str(i) for i in range(len(flat_values))])
        for row, val in enumerate(flat_values):
            item = QtWidgets.QTableWidgetItem(np.format_float_positional(val, trim="0"))
            item.setTextAlignment(
                QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
            self.table.setItem(row, 0, item)
        self.table.resizeColumnsToContents()
        if self._auto_minimum_width:
            horizontal_header = typing.cast(
                "QtWidgets.QHeaderView", self.table.horizontalHeader()
            )
            vertical_header = typing.cast(
                "QtWidgets.QHeaderView", self.table.verticalHeader()
            )
            self.setMinimumWidth(horizontal_header.length() + vertical_header.width())


class CoordinateEditorWidget(QtWidgets.QWidget):
    """Coordinate editor with direct values and affine scale/offset modes.

    Parameters
    ----------
    values
        Initial reference coordinate values for both the editable value grid and affine
        preview table.
    """

    def __init__(self, values: npt.ArrayLike) -> None:
        super().__init__()
        self.init_ui(values)
        self.set_old_coord(values)

    def init_ui(self, values: npt.ArrayLike) -> None:
        container_layout = QtWidgets.QVBoxLayout(self)
        container_layout.setContentsMargins(0, 0, 0, 0)

        self.edit_mode_tabs = QtWidgets.QTabWidget()
        self.edit_mode_tabs.currentChanged.connect(self._edit_mode_changed)
        container_layout.addWidget(self.edit_mode_tabs)

        self.grid = CoordinateGridWidget(
            values,
            connect_reset_button=False,
            auto_minimum_width=True,
        )
        self.grid.reset_btn.clicked.connect(self.reset)
        self.edit_mode_tabs.addTab(self.grid, "Values")

        self.spin0 = self.grid.spin0
        self.spin1 = self.grid.spin1
        self.mode_combo = self.grid.mode_combo
        self.count_spin = self.grid.count_spin
        self.reset_btn = self.grid.reset_btn
        self.table = self.grid.table

        affine_widget = QtWidgets.QWidget()
        affine_layout = QtWidgets.QFormLayout(affine_widget)

        self.scale_spin = erlab.interactive.utils.BetterSpinBox(
            compact=False, trim="0", value=1.0
        )
        self.scale_spin.valueChanged.connect(self.update_affine_preview)
        affine_layout.addRow("Scale", self.scale_spin)

        self.offset_spin = erlab.interactive.utils.BetterSpinBox(
            compact=False, trim="0"
        )
        self.offset_spin.valueChanged.connect(self.update_affine_preview)
        affine_layout.addRow("Offset", self.offset_spin)

        self.affine_table = QtWidgets.QTableWidget()
        self.affine_table.setColumnCount(2)
        self.affine_table.setHorizontalHeaderLabels(["Current", "Transformed"])
        self.affine_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.affine_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.affine_table.setSortingEnabled(False)
        self.affine_table.setAlternatingRowColors(True)
        affine_layout.addRow(self.affine_table)

        self.edit_mode_tabs.addTab(affine_widget, "Scale/Offset")

    @property
    def _old_coord(self) -> npt.NDArray:
        return self.grid.reference_coord

    @property
    def use_affine_transform(self) -> bool:
        """Whether the widget is currently editing by scale and offset."""
        return self.edit_mode_tabs.currentIndex() == 1

    @property
    def affine_scale(self) -> float:
        """Get the scale for affine coordinate editing."""
        return float(self.scale_spin.value())

    @property
    def affine_offset(self) -> float:
        """Get the offset for affine coordinate editing."""
        return float(self.offset_spin.value())

    @property
    def affine_coord(self) -> npt.NDArray:
        """Get the affine-transformed coordinates as a numpy array."""
        return self.affine_scale * self._old_coord + self.affine_offset

    def _affine_supported(self) -> bool:
        values = np.asarray(self._old_coord)
        return (
            values.ndim <= 1
            and np.issubdtype(values.dtype, np.number)
            and not np.issubdtype(values.dtype, np.complexfloating)
        )

    def _sync_affine_state(self) -> None:
        affine_supported = self._affine_supported()
        affine_index = 1
        self.edit_mode_tabs.setTabEnabled(affine_index, affine_supported)
        if not affine_supported and self.edit_mode_tabs.currentIndex() == affine_index:
            self.edit_mode_tabs.setCurrentIndex(0)
        with (
            QtCore.QSignalBlocker(self.scale_spin),
            QtCore.QSignalBlocker(self.offset_spin),
        ):
            self.scale_spin.setValue(1.0)
            self.offset_spin.setValue(0.0)
        self.update_affine_preview()

    @QtCore.Slot(int)
    def _edit_mode_changed(self, _index: int) -> None:
        if self.use_affine_transform:
            self.update_affine_preview()

    @QtCore.Slot()
    def mode_changed(self) -> None:
        """Handle the change of the mode combo box."""
        self.grid.mode_changed()

    @QtCore.Slot()
    def reset(self) -> None:
        """Reset the spin boxes to the original values."""
        self.grid.reset()
        self._sync_affine_state()

    def set_old_coord(self, values: npt.ArrayLike) -> None:
        """Set the old coordinates to the given values."""
        self.grid.set_old_coord(values)
        self._sync_affine_state()

    @property
    def _current_values_end(self) -> npt.NDArray:
        """Get the current values assuming spin1 value is the end."""
        return self.grid._current_values_end

    @property
    def _current_values_delta(self) -> npt.NDArray:
        """Get the current values assuming spin1 value is the step size."""
        return self.grid._current_values_delta

    @property
    def new_coord(self) -> npt.NDArray:
        """Get the edited coordinates as a numpy array."""
        return self.grid.new_coord

    @QtCore.Slot()
    def update_table(self) -> None:
        """Update the table with the current values from the spin boxes."""
        self.grid.update_table()

    @QtCore.Slot()
    def update_affine_preview(self) -> None:
        """Update the preview table for affine coordinate editing."""
        if not self._affine_supported():
            self.affine_table.setRowCount(0)
            return
        self._set_affine_table_values(self.affine_coord)

    def _set_table_values(self, values: npt.NDArray) -> None:
        """Set the table contents to the given numpy array."""
        self.grid._set_table_values(values)

    def _set_affine_table_values(self, values: npt.NDArray) -> None:
        """Set the affine preview table contents."""
        old_values = np.ravel(np.atleast_1d(self._old_coord))
        new_values = np.ravel(np.atleast_1d(values))
        self.affine_table.setRowCount(len(old_values))
        self.affine_table.setVerticalHeaderLabels(
            [str(i) for i in range(len(old_values))]
        )
        for row, (old, new) in enumerate(zip(old_values, new_values, strict=True)):
            for col, val in enumerate((old, new)):
                item = QtWidgets.QTableWidgetItem(
                    np.format_float_positional(val, trim="0")
                )
                item.setTextAlignment(
                    QtCore.Qt.AlignmentFlag.AlignLeft
                    | QtCore.Qt.AlignmentFlag.AlignVCenter
                )
                self.affine_table.setItem(row, col, item)
        self.affine_table.resizeColumnsToContents()
