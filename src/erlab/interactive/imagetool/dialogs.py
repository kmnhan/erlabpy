"""Dialogs for data manipulation found in the menu bar."""

from __future__ import annotations

__all__ = ["RotationDialog", "CropDialog", "NormalizeDialog"]

import weakref
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

from erlab.interactive.utils import (
    ExclusiveComboGroup,
    copy_to_clipboard,
    generate_code,
)

if TYPE_CHECKING:
    from collections.abc import Hashable

    from erlab.interactive.imagetool.core import ImageSlicerArea
    from erlab.interactive.imagetool.slicer import ArraySlicer


class _DataManipulationDialog(QtWidgets.QDialog):
    title: str | None = None
    show_copy_button: bool = False

    def __init__(self, slicer_area: ImageSlicerArea) -> None:
        super().__init__()
        if self.title is not None:
            self.setWindowTitle(self.title)

        self.slicer_area = slicer_area

        self._layout = QtWidgets.QFormLayout()
        self.setLayout(self._layout)

        self.setup_widgets()

        self.buttonBox = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        if self.show_copy_button:
            self.copy_button = QtWidgets.QPushButton("Copy Code")
            self.copy_button.clicked.connect(
                lambda: copy_to_clipboard(self.make_code())
            )
            self.buttonBox.addButton(
                self.copy_button, QtWidgets.QDialogButtonBox.ButtonRole.ActionRole
            )

        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.layout_.addRow(self.buttonBox)

    @property
    def layout_(self) -> QtWidgets.QFormLayout:
        return self._layout

    @property
    def slicer_area(self) -> ImageSlicerArea:
        _slicer_area = self._slicer_area()
        if _slicer_area:
            return _slicer_area
        raise LookupError("Parent was destroyed")

    @slicer_area.setter
    def slicer_area(self, value: ImageSlicerArea) -> None:
        self._slicer_area = weakref.ref(value)

    @property
    def array_slicer(self) -> ArraySlicer:
        return self.slicer_area.array_slicer

    def setup_widgets(self) -> None:
        # Overridden by subclasses
        pass

    def process_data(self) -> xr.DataArray:
        # Overridden by subclasses
        return self.slicer_area.data

    def make_code(self) -> str:
        # Overridden by subclasses
        return ""


class _DataTransformDialog(_DataManipulationDialog):
    """
    Parent class for implementing data changes that affect both shape and values.

    These changes are destructive and cannot be undone.
    """

    prefix: str = ""
    suffix: str = ""

    def __init__(self, slicer_area: ImageSlicerArea) -> None:
        super().__init__(slicer_area)
        self.new_window_check = QtWidgets.QCheckBox("Open in New Window")
        self.new_window_check.setChecked(True)
        self.layout_.insertRow(-1, self.new_window_check)

    @QtCore.Slot()
    def accept(self) -> None:
        if self.slicer_area.data.name is not None:
            new_name = f"{self.prefix}{self.slicer_area.data.name}{self.suffix}"
        else:
            new_name = None

        try:
            if self.new_window_check.isChecked():
                from erlab.interactive.imagetool import itool

                itool(self.process_data().rename(new_name), execute=False)
            else:
                self.slicer_area.set_data(self.process_data().rename(new_name))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"An error occurred: {e}")
            return

        super().accept()


class _DataFilterDialog(_DataManipulationDialog):
    """Parent class for implementing data changes that affect only the values."""

    enable_preview: bool = True

    def __init__(self, slicer_area: ImageSlicerArea) -> None:
        super().__init__(slicer_area)
        self._previewed: bool = False

        if self.enable_preview:
            self.preview_button = QtWidgets.QPushButton("Preview")
            self.preview_button.clicked.connect(self._preview)
            self.buttonBox.addButton(
                self.preview_button, QtWidgets.QDialogButtonBox.ButtonRole.ActionRole
            )

    @QtCore.Slot()
    def _preview(self):
        self._previewed = True
        self.slicer_area.apply_func(self.func)

    @QtCore.Slot()
    def reject(self) -> None:
        if self._previewed:
            self.slicer_area.apply_func(None)
        super().reject()

    @QtCore.Slot()
    def accept(self) -> None:
        try:
            self.slicer_area.apply_func(self.func)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"An error occurred: {e}")
            return
        super().accept()

    def func(self, data: xr.DataArray) -> xr.DataArray:
        # Implement this method in subclasses
        raise NotImplementedError


class RotationDialog(_DataTransformDialog):
    suffix = " Rotated"
    show_copy_button = True

    @property
    def _rotate_params(self) -> dict[str, Any]:
        return {
            "angle": self.angle_spin.value(),
            "axes": cast(tuple[str, str], tuple(self.slicer_area.main_image.axis_dims)),
            "center": cast(
                tuple[float, float], tuple(spin.value() for spin in self.center_spins)
            ),
            "reshape": self.reshape_check.isChecked(),
            "order": self.order_spin.value(),
        }

    def setup_widgets(self) -> None:
        main_image = self.slicer_area.main_image
        self.angle_spin = QtWidgets.QDoubleSpinBox()
        self.angle_spin.setRange(-360, 360)
        self.angle_spin.setSingleStep(1)
        self.angle_spin.setValue(0)
        self.angle_spin.setDecimals(2)
        self.angle_spin.setSuffix("°")
        self.layout_.addRow("Angle", self.angle_spin)

        self.center_spins = (QtWidgets.QDoubleSpinBox(), QtWidgets.QDoubleSpinBox())
        for i in range(2):
            axis: int = main_image.display_axis[i]
            dim: str = str(main_image.axis_dims[i])

            self.center_spins[i].setRange(*map(float, self.array_slicer.lims[i]))
            self.center_spins[i].setSingleStep(float(self.array_slicer.incs[i]))
            self.center_spins[i].setValue(0.0)
            self.center_spins[i].setDecimals(self.array_slicer.get_significant(axis))

            self.layout_.addRow(f"Center {dim}", self.center_spins[i])

        self.order_spin: QtWidgets.QSpinBox = QtWidgets.QSpinBox()
        self.order_spin.setRange(0, 5)
        self.order_spin.setValue(1)
        self.layout_.addRow("Spline Order", self.order_spin)

        self.reshape_check = QtWidgets.QCheckBox("Reshape")
        self.reshape_check.setChecked(True)
        self.layout_.addRow(self.reshape_check)

        if main_image.is_guidelines_visible:
            # Fill values from guideline
            self.angle_spin.setValue(-main_image._guideline_angle)
            for spin, val in zip(
                self.center_spins, main_image._guideline_offset, strict=True
            ):
                spin.setValue(val)

    def process_data(self) -> xr.DataArray:
        from erlab.analysis.transform import rotate

        return rotate(self.slicer_area.data, **self._rotate_params)

    def make_code(self) -> str:
        from erlab.analysis.transform import rotate

        placeholder = " "
        params = dict(self._rotate_params)

        for k, v in params.items():
            if isinstance(v, tuple):
                params[k] = f"({', '.join(map(str, v))})"
            else:
                params[k] = str(v)

        return generate_code(
            rotate, [f"|{placeholder}|"], self._rotate_params, module="era.transform"
        )


class CropDialog(_DataTransformDialog):
    suffix = " Cropped"
    show_copy_button = True

    @property
    def _enabled_dims(self) -> list[Hashable]:
        return [k for k, v in self.dim_checks.items() if v.isChecked()]

    @property
    def _cursor_indices(self) -> tuple[int, int]:
        return cast(
            tuple[int, int], tuple(combo.currentIndex() for combo in self.cursor_combos)
        )

    @property
    def _slice_kwargs(self) -> dict[Hashable, slice]:
        c0, c1 = self._cursor_indices

        vals = self.array_slicer._values

        slice_dict = {}

        for k in self._enabled_dims:
            ax_idx = self.slicer_area.data.dims.index(k)
            sig_digits = self.array_slicer.get_significant(ax_idx)

            v0, v1 = vals[c0][ax_idx], vals[c1][ax_idx]
            if v0 > v1:
                v0, v1 = v1, v0

            slice_dict[k] = slice(
                float(np.round(v0, sig_digits)), float(np.round(v1, sig_digits))
            )
        return slice_dict

    def exec(self):
        if self.slicer_area.n_cursors == 1:
            QtWidgets.QMessageBox.warning(
                self,
                "Only 1 Cursor",
                "You need at least 2 cursors to crop the data.",
            )
            return
        super().exec()

    @QtCore.Slot()
    def accept(self) -> None:
        if self._slice_kwargs == {}:
            QtWidgets.QMessageBox.warning(
                self,
                "No Dimensions Selected",
                "You need to select at least one dimension.",
            )
            return
        super().accept()

    def setup_widgets(self) -> None:
        if self.slicer_area.n_cursors == 1:
            return

        self._cursors_group = ExclusiveComboGroup(self)

        self.cursor_combos: list[QtWidgets.QComboBox] = []

        cursor_group = QtWidgets.QGroupBox("Between")
        cursor_layout = QtWidgets.QHBoxLayout()
        cursor_group.setLayout(cursor_layout)

        for i in range(2):
            combo = QtWidgets.QComboBox()
            for cursor_idx in range(self.slicer_area.n_cursors):
                combo.addItem(
                    self.slicer_area._cursor_icon(cursor_idx),
                    self.slicer_area._cursor_name(cursor_idx),
                )
            combo.setCurrentIndex(i)
            combo.setMaximumHeight(QtGui.QFontMetrics(combo.font()).height() + 3)
            combo.setIconSize(QtCore.QSize(10, 10))
            cursor_layout.addWidget(combo)
            self._cursors_group.addCombo(combo)
            self.cursor_combos.append(combo)

        self.layout_.addRow(cursor_group)

        dim_group = QtWidgets.QGroupBox("Dimensions")
        dim_layout = QtWidgets.QVBoxLayout()
        dim_group.setLayout(dim_layout)

        self.dim_checks: dict[Hashable, QtWidgets.QCheckBox] = {}

        for d in self.slicer_area.data.dims:
            self.dim_checks[d] = QtWidgets.QCheckBox(str(d))
            dim_layout.addWidget(self.dim_checks[d])

        self.layout_.addRow(dim_group)

    def process_data(self) -> xr.DataArray:
        return self.slicer_area.data.sel(self._slice_kwargs)

    def make_code(self) -> str:
        kwargs: dict[Hashable, slice] = self._slice_kwargs
        if all(isinstance(k, str) and str(k).isidentifier() for k in kwargs):
            out = generate_code(
                xr.DataArray.sel, [], kwargs=cast(dict[str, slice], kwargs), module="."
            )
        else:
            out = f".sel({self._slice_kwargs})"

        return out.replace(", None)", ")")


class NormalizeDialog(_DataFilterDialog):
    title = "Normalize"
    show_copy_button = False

    def setup_widgets(self) -> None:
        dim_group = QtWidgets.QGroupBox("Dimensions")
        dim_layout = QtWidgets.QVBoxLayout()
        dim_group.setLayout(dim_layout)

        self.dim_checks: dict[Hashable, QtWidgets.QCheckBox] = {}

        for d in self.slicer_area.data.dims:
            self.dim_checks[d] = QtWidgets.QCheckBox(str(d))
            dim_layout.addWidget(self.dim_checks[d])

        option_group = QtWidgets.QGroupBox("Options")
        option_layout = QtWidgets.QVBoxLayout()
        option_group.setLayout(option_layout)

        self.opts: list[QtWidgets.QRadioButton] = []
        self.opts.append(QtWidgets.QRadioButton("Data/Area"))
        self.opts.append(QtWidgets.QRadioButton("(Data−m)/(M−m)"))
        self.opts.append(QtWidgets.QRadioButton("Data−m"))
        self.opts.append(QtWidgets.QRadioButton("(Data−m)/Area"))

        self.opts[0].setChecked(True)
        for opt in self.opts:
            option_layout.addWidget(opt)

        self.layout_.addRow(dim_group)
        self.layout_.addRow(option_group)

    def func(self, data: xr.DataArray) -> xr.DataArray:
        norm_dims = tuple(k for k, v in self.dim_checks.items() if v.isChecked())
        if len(norm_dims) == 0:
            return data

        calc_area: bool = self.opts[0].isChecked() or self.opts[3].isChecked()
        calc_minimum: bool = not self.opts[0].isChecked()
        calc_maximum: bool = self.opts[1].isChecked()

        if calc_area:
            area = data.mean(norm_dims)

        if calc_minimum:
            minimum = data.min(norm_dims)

        if calc_maximum:
            maximum = data.max(norm_dims)

        if self.opts[0].isChecked():
            return data / area

        if self.opts[1].isChecked():
            return (data - minimum) / (maximum - minimum)

        if self.opts[2].isChecked():
            return data - minimum

        return (data - minimum) / area
