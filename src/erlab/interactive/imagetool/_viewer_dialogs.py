"""Dialog helpers used by :mod:`erlab.interactive.imagetool.viewer`."""

from __future__ import annotations

import typing
import weakref

from qtpy import QtCore, QtWidgets

import erlab

if typing.TYPE_CHECKING:
    from collections.abc import Hashable

    from erlab.interactive.imagetool.viewer import ImageSlicerArea


class _AssociatedCoordsDialog(QtWidgets.QDialog):
    def __init__(self, slicer_area: ImageSlicerArea) -> None:
        super().__init__(slicer_area)
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)

        self._slicer_area = weakref.ref(slicer_area)

        self._layout = QtWidgets.QFormLayout()
        self.setLayout(self._layout)

        self._checks: dict[Hashable, QtWidgets.QCheckBox] = {}
        for dim, coords in slicer_area.array_slicer.associated_coords.items():
            for k in coords:
                self._checks[k] = QtWidgets.QCheckBox(str(k))
                self._checks[k].setChecked(
                    k in slicer_area.array_slicer.twin_coord_names
                )
                self._layout.addRow(self._checks[k], QtWidgets.QLabel(f"({dim})"))

        self._button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self._button_box.accepted.connect(self.accept)
        self._button_box.rejected.connect(self.reject)
        self._layout.addRow(self._button_box)

    def exec(self) -> int:
        if len(self._checks) == 0:
            QtWidgets.QMessageBox.warning(
                self,
                "No Associated Coordinates",
                "No 1D non-dimension coordinates were found in the data.",
            )
            return QtWidgets.QDialog.DialogCode.Rejected
        return super().exec()

    @QtCore.Slot()
    def accept(self) -> None:
        slicer_area = self._slicer_area()
        if slicer_area:  # pragma: no branch
            slicer_area.array_slicer.twin_coord_names = {
                coord for coord, check in self._checks.items() if check.isChecked()
            }
        super().accept()


class _CursorColorCoordDialog(QtWidgets.QDialog):
    def __init__(self, slicer_area: ImageSlicerArea) -> None:
        super().__init__(slicer_area)
        self._slicer_area = weakref.ref(slicer_area)
        self.setup_ui()

    def update_params(self) -> None:
        slicer_area = self._slicer_area()
        if slicer_area:  # pragma: no branch
            cursor_color_params = slicer_area.array_slicer._cursor_color_params
            if cursor_color_params is not None:
                _, coord_name, cmap, reverse, vmin, vmax = cursor_color_params
                self.choose_coord(coord_name)
                self.cmap_combo.setCurrentText(cmap)
                self.reverse_check.setChecked(reverse)
                self.start_spin.setValue(vmin)
                self.stop_spin.setValue(vmax)
            else:
                self.main_group.setChecked(False)

    def choose_coord(self, coord_name: Hashable) -> None:
        self.coord_combo.setCurrentText(str(coord_name))

    def get_checked_coord_name(self) -> tuple[Hashable, Hashable] | None:
        if not self.main_group.isChecked():
            return None
        slicer_area = self._slicer_area()
        if not slicer_area:  # pragma: no cover
            return None
        coord_name = self.coord_combo.currentText()
        for dim_name in slicer_area.data.dims:
            if coord_name == str(dim_name):
                return dim_name, dim_name
        for dim, coords in slicer_area.array_slicer.associated_coords.items():
            for k in coords:
                if coord_name == str(k):
                    return dim, k
        return None

    def setup_ui(self):
        slicer_area = self._slicer_area()
        if not slicer_area:  # pragma: no cover
            return

        self.layout_ = QtWidgets.QVBoxLayout(self)
        self.setLayout(self.layout_)

        self.main_group = QtWidgets.QGroupBox("Set cursor color by coordinate", self)
        self.layout_.addWidget(self.main_group)
        main_layout = QtWidgets.QVBoxLayout()
        self.main_group.setLayout(main_layout)
        self.main_group.setCheckable(True)

        # Coord selection
        coord_layout = QtWidgets.QFormLayout()
        main_layout.addLayout(coord_layout)

        self.coord_combo = QtWidgets.QComboBox()
        coord_layout.addRow("Coordinate", self.coord_combo)

        for name in slicer_area.data.dims:
            self.coord_combo.addItem(str(name))

        for coords in slicer_area.array_slicer.associated_coords.values():
            for k in coords:
                self.coord_combo.addItem(str(k))

        # Colormap selection
        cmap_group = QtWidgets.QGroupBox("Colormap parameters", self)
        main_layout.addWidget(cmap_group)
        cmap_layout = QtWidgets.QHBoxLayout()
        cmap_group.setLayout(cmap_layout)

        self.cmap_combo = erlab.interactive.colors.ColorMapComboBox(self)
        self.cmap_combo.setToolTip("Select a colormap to sample colors from")
        self.cmap_combo.setDefaultCmap("coolwarm")
        cmap_layout.addWidget(self.cmap_combo)

        self.reverse_check = QtWidgets.QCheckBox("Reverse", self)
        self.reverse_check.setToolTip("Reverse the colormap")
        cmap_layout.addWidget(self.reverse_check)

        cmap_layout.addStretch()
        cmap_layout.addWidget(QtWidgets.QLabel("Range:"))
        self.start_spin = QtWidgets.QDoubleSpinBox(self)
        self.start_spin.setRange(0.0, 1.0)
        self.start_spin.setDecimals(2)
        self.start_spin.setSingleStep(0.1)
        self.start_spin.setValue(0.1)
        self.start_spin.setToolTip("Start of the colormap")
        cmap_layout.addWidget(self.start_spin)

        self.stop_spin = QtWidgets.QDoubleSpinBox(self)
        self.stop_spin.setRange(0.0, 1.0)
        self.stop_spin.setDecimals(2)
        self.stop_spin.setSingleStep(0.1)
        self.stop_spin.setValue(0.9)
        self.stop_spin.setToolTip("End of the colormap")
        cmap_layout.addWidget(self.stop_spin)
        cmap_layout.addStretch()

        # Bottom layout
        bottom_layout = QtWidgets.QHBoxLayout()
        self.layout_.addLayout(bottom_layout)

        # Dialog buttons
        self.button_box = QtWidgets.QDialogButtonBox()
        btn_ok = self.button_box.addButton(
            QtWidgets.QDialogButtonBox.StandardButton.Ok,
        )
        btn_cancel = self.button_box.addButton(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel,
        )
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)

        self.layout_.addWidget(self.button_box)
        self.update_params()

    def accept(self) -> None:
        slicer_area = self._slicer_area()
        if slicer_area:  # pragma: no branch
            dim_and_coord_names = self.get_checked_coord_name()
            if dim_and_coord_names is None:
                slicer_area.array_slicer._cursor_color_params = None
            else:
                dim_name, coord_name = dim_and_coord_names
                slicer_area.array_slicer._cursor_color_params = (
                    dim_name,
                    coord_name,
                    self.cmap_combo.currentText(),
                    self.reverse_check.isChecked(),
                    self.start_spin.value(),
                    self.stop_spin.value(),
                )
            slicer_area._refresh_cursor_colors(
                tuple(i for i in range(slicer_area.n_cursors)), None
            )
        super().accept()
