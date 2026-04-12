"""Dialogs for data manipulation found in the menu bar."""

from __future__ import annotations

import math
import typing
import weakref

import numpy as np
import numpy.typing as npt
import pyqtgraph as pg
from qtpy import QtCore, QtGui, QtWidgets

import erlab

if typing.TYPE_CHECKING:
    from collections.abc import Hashable

    import xarray as xr

    from erlab.interactive.imagetool.plot_items import ItoolPolyLineROI
    from erlab.interactive.imagetool.slicer import ArraySlicer
    from erlab.interactive.imagetool.viewer import ColorMapState, ImageSlicerArea

_GAUSSIAN_FWHM_FACTOR: float = 2 * math.sqrt(2 * math.log(2))


class _DataManipulationDialog(QtWidgets.QDialog):
    """Parent class for a dialog that manipulates data.

    In practice, use child classes `DataTransformDialog` and `DataFilterDialog`.
    """

    whatsthis: str | None = None
    """The whatsthis text for the dialog window."""

    title: str | None = None
    """The title of the dialog window."""

    enable_copy: bool = False
    """Whether to show a button to copy the code to the clipboard.

    If True, the button will be shown in the dialog box. The `make_code` method must be
    overridden to provide the code to be copied.
    """

    _sigCodeCopied = QtCore.Signal(str)

    def __init__(self, slicer_area: ImageSlicerArea) -> None:
        super().__init__(slicer_area)
        if self.title is not None:
            self.setWindowTitle(self.title)

        self.slicer_area = slicer_area

        self._layout = QtWidgets.QFormLayout()
        self.setLayout(self._layout)

        if self.whatsthis is not None:
            self.setWhatsThis(self.whatsthis)
            self.setWindowFlags(
                self.windowFlags() | QtCore.Qt.WindowType.WindowContextHelpButtonHint
            )

        self.buttonBox = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        if self.enable_copy:
            self.copy_button = QtWidgets.QPushButton("Copy Code")
            self.copy_button.clicked.connect(self._copy)
            self.buttonBox.addButton(
                self.copy_button, QtWidgets.QDialogButtonBox.ButtonRole.ActionRole
            )

        self.setup_widgets()

        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.layout_.addRow(self.buttonBox)

    @property
    def layout_(self) -> QtWidgets.QFormLayout:
        return self._layout

    @property
    def slicer_area(self) -> ImageSlicerArea:
        slicer_area = self._slicer_area()
        if slicer_area:
            return slicer_area
        raise LookupError("Parent was destroyed")

    @slicer_area.setter
    def slicer_area(self, value: ImageSlicerArea) -> None:
        self._slicer_area = weakref.ref(value)

    @property
    def array_slicer(self) -> ArraySlicer:
        return self.slicer_area.array_slicer

    @QtCore.Slot()
    def _copy(self) -> None:
        code = self.make_code()
        if code:
            self._sigCodeCopied.emit(erlab.interactive.utils.copy_to_clipboard(code))
        else:
            QtWidgets.QMessageBox.warning(
                self, "Nothing to Copy", "Generated code is empty."
            )

    def _validate(self) -> QtWidgets.QDialog.DialogCode:
        """Run checks before opening the dialog.

        Reimplement this method in subclasses to perform checks before showing the
        dialog.

        Returns
        -------
        QDialog.DialogCode
            Anything other than `QDialog.DialogCode.Accepted` will prevent the dialog
            from showing.
        """
        return QtWidgets.QDialog.DialogCode.Accepted

    def show(self) -> None:
        """Run checks and (asynchronously) open the dialog if they pass.

        If checks fail, the dialog is never shown; reject() is scheduled so that
        finished(Rejected) still fires (allowing uniform handling).
        """
        code = self._validate()
        if code != QtWidgets.QDialog.DialogCode.Accepted:
            erlab.interactive.utils.single_shot(self, 0, self.reject)
            return
        super().show()

    def setup_widgets(self) -> None:
        # Overridden by subclasses
        pass

    def process_data(self, data: xr.DataArray) -> xr.DataArray:
        # Overridden by subclasses
        return data

    def make_code(self) -> str:
        # Overridden by subclasses
        return ""


class DataTransformDialog(_DataManipulationDialog):
    """Parent class for implementing data changes that affect both shape and values.

    These changes are destructive and cannot be undone. The user can choose to open the
    transformed data in a new window or replace the current data.

    - Override method `setup_widgets` to add widgets to the dialog.

    - Override method `process_data` to implement the data transformation.

    - Override method `make_code` to generate code that can be copied to the clipboard.

    - Override attribute `title` to set the title of the dialog window.

    - Override attribute `enable_copy` to show or hide the copy button.

    - Override attributes `prefix` and `suffix` to set the prefix and suffix of the new
      data name.

    - Override attribute `keep_colors` and `keep_color_limits` to control which
      color-related settings are migrated when opening in a new window.

    """

    prefix: str = ""
    suffix: str = ""

    keep_colors: bool = True
    """Whether to keep the color settings when opening in a new window.

    If True, the same colormap and normalization is used in the new window.
    """

    keep_color_limits: bool = True
    """Whether to also keep manual color limits when opening in a new window."""

    apply_on_nonuniform_data: bool = False
    """Whether to apply the transform on data with non-uniform dimensions.

    Set to `True` for transforms that can handle coordinates that are not evenly spaced.
    """

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
            new_name = self.suffix.lstrip("_")

        try:
            applied_func = None
            if self.slicer_area._applied_func is not None:
                # Transform must be done on unfiltered data
                applied_func = self.slicer_area._applied_func
                self.slicer_area.apply_func(None)

            if self.apply_on_nonuniform_data:
                processed = self.process_data(
                    erlab.interactive.imagetool.slicer.restore_nonuniform_dims(
                        self.slicer_area.data
                    )
                )
            else:
                processed = erlab.interactive.imagetool.slicer.restore_nonuniform_dims(
                    self.process_data(self.slicer_area.data)
                )

            processed = processed.rename(new_name)

            if self.new_window_check.isChecked():
                itool_kw: dict[str, typing.Any] = {
                    "data": processed,
                    "execute": False,
                }

                if self.keep_colors:
                    color_props: ColorMapState = self.slicer_area.colormap_properties

                    itool_kw["cmap"] = color_props["cmap"]
                    itool_kw["gamma"] = color_props["gamma"]
                    itool_kw["high_contrast"] = color_props["high_contrast"]
                    itool_kw["zero_centered"] = color_props["zero_centered"]

                    if color_props["levels_locked"] and self.keep_color_limits:
                        itool_kw["vmin"], itool_kw["vmax"] = color_props["levels"]

                erlab.interactive.itool(**itool_kw)
            else:
                self.slicer_area.replace_source_data(processed, emit_edited=True)

            del processed

            if applied_func is not None:
                self.slicer_area.apply_func(applied_func)

        except Exception:
            erlab.interactive.utils.MessageDialog.critical(
                self, "Error", "An error occurred while processing data."
            )
            return

        super().accept()


class DataFilterDialog(_DataManipulationDialog):
    """Parent class for implementing data changes that only affects the appearance.

    These changes are not destructive and can be undone from the menu bar. Only one kind
    of filter can be applied at a time, and applying a new kind of filter will replace
    the previous one.

    - Override method `setup_widgets` to add widgets to the dialog.

    - Override method `process_data` to implement the filter. The output must be a new
      DataArray with the same shape as the input.

    - Override method `make_code` to generate code that can be copied to the clipboard.

    - Override attribute `title` to set the title of the dialog window.

    - Override attribute `enable_copy` to show or hide the copy button.

    - Override attributes `enable_preview` to show or hide the preview button.

    """

    enable_preview: bool = True
    """Whether to show a preview button."""

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
        self.slicer_area.apply_func(self.process_data)

    @QtCore.Slot()
    def reject(self) -> None:
        if self._previewed:
            self.slicer_area.apply_func(None)
        super().reject()

    @QtCore.Slot()
    def accept(self) -> None:
        try:
            self.slicer_area.apply_func(self.process_data)
        except Exception:
            erlab.interactive.utils.MessageDialog.critical(
                self, "Error", "An error occurred while processing data."
            )
            return
        super().accept()


class RotationDialog(DataTransformDialog):
    enable_copy = True

    @property
    def suffix(self) -> str:
        angle_str = str(self._rotate_params["angle"])
        return f"_rot{angle_str}"

    @suffix.setter
    def suffix(self, value: str) -> None:
        # To satisfy mypy
        pass

    @property
    def _rotate_params(self) -> dict[str, typing.Any]:
        return {
            "angle": float(
                np.round(self.angle_spin.value(), self.angle_spin.decimals())
            ),
            "axes": typing.cast(
                "tuple[str, str]", tuple(self.slicer_area.main_image.axis_dims_uniform)
            ),
            "center": typing.cast(
                "tuple[float, float]", tuple(spin.value() for spin in self.center_spins)
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

        self.center_spins = (
            pg.SpinBox(dec=True, compactHeight=False),
            pg.SpinBox(dec=True, compactHeight=False),
        )
        for i in range(2):
            axis: int = main_image.display_axis[i]
            dim: str = str(main_image.axis_dims_uniform[i])
            self.center_spins[i].setDecimals(
                self.array_slicer.get_significant(axis, uniform=True)
            )
            self.center_spins[i].setSingleStep(float(self.array_slicer.incs_uniform[i]))
            self.center_spins[i].setValue(0.0)

            self.layout_.addRow(f"Center {dim}", self.center_spins[i])

        self.order_spin: QtWidgets.QSpinBox = QtWidgets.QSpinBox()
        self.order_spin.setRange(0, 5)
        self.order_spin.setValue(1)
        self.layout_.addRow("Spline Order", self.order_spin)

        self.reshape_check: QtWidgets.QCheckBox = QtWidgets.QCheckBox("Reshape")
        self.reshape_check.setChecked(True)
        self.layout_.addRow(self.reshape_check)

        if main_image.is_guidelines_visible:
            # Fill values from guideline
            self.angle_spin.setValue(-main_image._guideline_angle)
            for spin, val in zip(
                self.center_spins, main_image._guideline_offset, strict=True
            ):
                spin.setValue(val)

    def process_data(self, data: xr.DataArray) -> xr.DataArray:
        return erlab.analysis.transform.rotate(data, **self._rotate_params)

    def make_code(self) -> str:
        placeholder = self.slicer_area.watched_data_name or " "
        return erlab.interactive.utils.generate_code(
            erlab.analysis.transform.rotate,
            [f"|{placeholder}|"],
            self._rotate_params,
            module="era.transform",
        )


class AverageDialog(DataTransformDialog):
    title = "Average Over Dimensions"
    suffix = "_avg"
    enable_copy = True

    def setup_widgets(self) -> None:
        dim_group = QtWidgets.QGroupBox("Dimensions")
        dim_layout = QtWidgets.QVBoxLayout()
        dim_group.setLayout(dim_layout)

        self.dim_checks: dict[Hashable, QtWidgets.QCheckBox] = {}

        for d in self.slicer_area.data.dims:
            self.dim_checks[d] = QtWidgets.QCheckBox(str(d))
            dim_layout.addWidget(self.dim_checks[d])

        self.layout_.addRow(dim_group)

    @property
    def _target_dims(self) -> tuple[Hashable, ...]:
        return tuple(k for k, v in self.dim_checks.items() if v.isChecked())

    def process_data(self, data: xr.DataArray) -> xr.DataArray:
        return data.qsel.average(self._target_dims)

    @QtCore.Slot()
    def accept(self) -> None:
        if self._target_dims == {}:
            QtWidgets.QMessageBox.warning(
                self,
                "No Dimensions Selected",
                "You need to select at least one dimension.",
            )
            return

        if (self.slicer_area.data.ndim - len(self._target_dims)) < 1:
            QtWidgets.QMessageBox.warning(
                self,
                "Not Enough Dimensions Left",
                "Data must have at least 1 dimension after averaging to be displayed.",
            )
            return

        super().accept()

    def make_code(self) -> str:
        placeholder = self.slicer_area.watched_data_name or ""
        arg = (
            str(self._target_dims)
            if len(self._target_dims) > 1
            else f'"{self._target_dims[0]}"'
        )
        return f"{placeholder}.qsel.average({arg})"


class CoarsenDialog(DataTransformDialog):
    title = "Coarsen"
    enable_copy = True
    apply_on_nonuniform_data = True

    _REDUCERS: tuple[str, ...] = (
        "all",
        "any",
        "count",
        "max",
        "mean",
        "median",
        "min",
        "prod",
        "std",
        "sum",
        "var",
    )

    @property
    def suffix(self) -> str:
        return f"_coarsen_{self._reducer}"

    @suffix.setter
    def suffix(self, value: str) -> None:
        # To satisfy mypy
        pass

    def setup_widgets(self) -> None:
        self._source_data = erlab.interactive.imagetool.slicer.restore_nonuniform_dims(
            self.slicer_area.data
        )

        dim_group = QtWidgets.QGroupBox("Dimensions")
        dim_layout = QtWidgets.QGridLayout()
        dim_group.setLayout(dim_layout)

        dim_layout.addWidget(QtWidgets.QLabel("Dimension"), 0, 0)
        dim_layout.addWidget(QtWidgets.QLabel("Window Size"), 0, 1)

        self.dim_checks: dict[Hashable, QtWidgets.QCheckBox] = {}
        self.window_spins: dict[Hashable, QtWidgets.QSpinBox] = {}

        for row, dim in enumerate(self._source_data.dims, start=1):
            check = QtWidgets.QCheckBox(str(dim))
            spin = QtWidgets.QSpinBox()
            spin.setRange(1, 2_147_483_647)
            spin.setValue(3)
            spin.setEnabled(False)
            check.toggled.connect(spin.setEnabled)

            self.dim_checks[dim] = check
            self.window_spins[dim] = spin

            dim_layout.addWidget(check, row, 0)
            dim_layout.addWidget(spin, row, 1)

        options_group = QtWidgets.QGroupBox("Options")
        options_layout = QtWidgets.QFormLayout()
        options_group.setLayout(options_layout)

        self.boundary_combo = QtWidgets.QComboBox()
        self.boundary_combo.addItems(["exact", "trim", "pad"])
        self.boundary_combo.setCurrentText("trim")
        options_layout.addRow("Boundary", self.boundary_combo)

        self.side_combo = QtWidgets.QComboBox()
        self.side_combo.addItems(["left", "right"])
        options_layout.addRow("Side", self.side_combo)

        self.coord_func_combo = QtWidgets.QComboBox()
        self.coord_func_combo.addItems(
            ["mean", "median", "min", "max", "first", "last"]
        )
        self.coord_func_combo.setCurrentText("mean")
        options_layout.addRow("Coordinate Function", self.coord_func_combo)

        self.reducer_combo = QtWidgets.QComboBox()
        self.reducer_combo.addItems(list(self._REDUCERS))
        self.reducer_combo.setCurrentText("mean")
        options_layout.addRow("Reducer", self.reducer_combo)

        self.layout_.addRow(dim_group)
        self.layout_.addRow(options_group)

    @property
    def _selected_windows(self) -> dict[Hashable, int]:
        return {
            dim: self.window_spins[dim].value()
            for dim, check in self.dim_checks.items()
            if check.isChecked()
        }

    @property
    def _coord_func(self) -> str:
        return self.coord_func_combo.currentText().strip()

    @property
    def _reducer(self) -> str:
        return self.reducer_combo.currentText()

    @property
    def _coarsen_kwargs(self) -> dict[str, typing.Any]:
        kwargs: dict[str, typing.Any] = {}
        kwargs["dim"] = self._selected_windows
        kwargs["boundary"] = self.boundary_combo.currentText()
        kwargs["side"] = self.side_combo.currentText()
        kwargs["coord_func"] = self._coord_func
        return kwargs

    def process_data(self, data: xr.DataArray) -> xr.DataArray:
        coarsened = data.coarsen(**self._coarsen_kwargs)
        return typing.cast("xr.DataArray", getattr(coarsened, self._reducer)())

    @QtCore.Slot()
    def accept(self) -> None:
        if not self._selected_windows:
            QtWidgets.QMessageBox.warning(
                self,
                "No Dimensions Selected",
                "You need to select at least one dimension.",
            )
            return

        if self.boundary_combo.currentText() == "exact":
            invalid_dims = [
                str(dim)
                for dim, window in self._selected_windows.items()
                if self._source_data.sizes[dim] % window != 0
            ]
            if invalid_dims:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Incompatible Window Size",
                    "Window sizes must evenly divide the selected dimensions when "
                    "boundary is exact: "
                    f"{', '.join(invalid_dims)}. Try trim or pad instead.",
                )
                return

        if self.boundary_combo.currentText() == "trim":
            empty_dims = [
                str(dim)
                for dim, window in self._selected_windows.items()
                if self._source_data.sizes[dim] // window == 0
            ]
            if empty_dims:
                QtWidgets.QMessageBox.warning(
                    self,
                    "No Output Blocks",
                    "Trim boundary would remove all data along: "
                    f"{', '.join(empty_dims)}.",
                )
                return

        super().accept()

    def make_code(self) -> str:
        if not self._selected_windows:
            return ""
        placeholder = self.slicer_area.watched_data_name or ""

        kwargs = self._coarsen_kwargs.copy()
        if all(
            isinstance(k, str) for k in self._coarsen_kwargs["dim"]
        ):  # pragma: no branch
            window_kwargs = kwargs.pop("dim")
            kwargs = dict(**window_kwargs, **kwargs)

        return (
            erlab.interactive.utils.generate_code(
                type(self._source_data).coarsen,
                args=[],
                kwargs=kwargs,
                module=placeholder,
            )
            + f".{self._reducer}()"
        )


class ThinDialog(DataTransformDialog):
    title = "Thin Data"
    suffix = "_thin"
    enable_copy = True
    apply_on_nonuniform_data = True

    def setup_widgets(self) -> None:
        self._source_data = erlab.interactive.imagetool.slicer.restore_nonuniform_dims(
            self.slicer_area.data
        )

        mode_group = QtWidgets.QGroupBox("Mode")
        mode_layout = QtWidgets.QVBoxLayout()
        mode_group.setLayout(mode_layout)

        self.per_dim_radio = QtWidgets.QRadioButton("Thin selected dimensions")
        self.global_radio = QtWidgets.QRadioButton("Thin all dimensions equally")
        self.per_dim_radio.setChecked(True)

        mode_layout.addWidget(self.per_dim_radio)
        mode_layout.addWidget(self.global_radio)

        self._mode_group = QtWidgets.QButtonGroup(self)
        self._mode_group.addButton(self.per_dim_radio)
        self._mode_group.addButton(self.global_radio)

        dim_group = QtWidgets.QGroupBox("Dimensions")
        dim_layout = QtWidgets.QGridLayout()
        dim_group.setLayout(dim_layout)

        dim_layout.addWidget(QtWidgets.QLabel("Dimension"), 0, 0)
        dim_layout.addWidget(QtWidgets.QLabel("Factor"), 0, 1)

        self.dim_group = dim_group
        self.dim_checks: dict[Hashable, QtWidgets.QCheckBox] = {}
        self.factor_spins: dict[Hashable, QtWidgets.QSpinBox] = {}

        for row, dim in enumerate(self._source_data.dims, start=1):
            check = QtWidgets.QCheckBox(str(dim))
            spin = QtWidgets.QSpinBox()
            spin.setRange(1, 2_147_483_647)
            spin.setValue(2)
            spin.setEnabled(False)
            check.toggled.connect(spin.setEnabled)

            self.dim_checks[dim] = check
            self.factor_spins[dim] = spin

            dim_layout.addWidget(check, row, 0)
            dim_layout.addWidget(spin, row, 1)

        global_group = QtWidgets.QGroupBox("Global Factor")
        global_layout = QtWidgets.QFormLayout()
        global_group.setLayout(global_layout)

        self.global_group = global_group
        self.global_spin = QtWidgets.QSpinBox()
        self.global_spin.setRange(1, 2_147_483_647)
        self.global_spin.setValue(2)
        global_layout.addRow("Factor", self.global_spin)

        self.per_dim_radio.toggled.connect(self._update_mode)
        self.global_radio.toggled.connect(self._update_mode)

        self.layout_.addRow(mode_group)
        self.layout_.addRow(dim_group)
        self.layout_.addRow(global_group)
        self._update_mode()

    @property
    def _use_global_mode(self) -> bool:
        return self.global_radio.isChecked()

    @property
    def _selected_factors(self) -> dict[Hashable, int]:
        return {
            dim: self.factor_spins[dim].value()
            for dim, check in self.dim_checks.items()
            if check.isChecked()
        }

    @property
    def _effective_factors(self) -> dict[Hashable, int]:
        if self._use_global_mode:
            return {
                dim: self.global_spin.value()
                for dim in self._source_data.dims
                if self.global_spin.value() > 1
            }
        return {
            dim: factor for dim, factor in self._selected_factors.items() if factor > 1
        }

    @QtCore.Slot()
    def _update_mode(self) -> None:
        self.dim_group.setEnabled(not self._use_global_mode)
        self.global_group.setEnabled(self._use_global_mode)

    def process_data(self, data: xr.DataArray) -> xr.DataArray:
        if self._use_global_mode:
            return data.thin(self.global_spin.value())
        return data.thin(self._effective_factors)

    @QtCore.Slot()
    def accept(self) -> None:
        if not self._use_global_mode and not self._selected_factors:
            QtWidgets.QMessageBox.warning(
                self,
                "No Dimensions Selected",
                "You need to select at least one dimension.",
            )
            return

        if not self._effective_factors:
            QtWidgets.QMessageBox.warning(
                self,
                "No Thinning Requested",
                "Choose at least one thinning factor greater than 1.",
            )
            return

        super().accept()

    def make_code(self) -> str:
        placeholder = self.slicer_area.watched_data_name or ""
        if self._use_global_mode:
            if self.global_spin.value() <= 1:
                return ""
            return f"{placeholder}.thin({self.global_spin.value()})"

        if not self._effective_factors:
            return ""
        kwargs = erlab.interactive.utils.format_kwargs(self._effective_factors)
        return f"{placeholder}.thin({kwargs})"


class SymmetrizeDialog(DataTransformDialog):
    title = "Symmetrize"
    enable_copy = True

    @property
    def suffix(self) -> str:
        return f"_sym_{self._params['dim']}"

    @suffix.setter
    def suffix(self, value: str) -> None:
        # To satisfy mypy
        pass

    def setup_widgets(self) -> None:
        dim_group = QtWidgets.QGroupBox("Mirror plane")
        dim_layout = QtWidgets.QHBoxLayout()
        dim_group.setLayout(dim_layout)

        self._dim_combo = QtWidgets.QComboBox()
        self._dim_combo.addItems([str(d) for d in self.slicer_area.data.dims])
        dim_layout.addWidget(self._dim_combo)

        self._center_spin = QtWidgets.QDoubleSpinBox()
        dim_layout.addWidget(self._center_spin)
        self._dim_combo.currentIndexChanged.connect(self._update_spin)
        self._update_spin()

        option_group = QtWidgets.QGroupBox("Options")
        option_group_layout = QtWidgets.QVBoxLayout()
        option_group.setLayout(option_group_layout)

        self.subtract_check = QtWidgets.QCheckBox("Subtract")
        self.subtract_check.setChecked(False)
        self.subtract_check.setToolTip(
            "Subtract the reflected part from the data instead of adding it."
        )
        option_group_layout.addWidget(self.subtract_check)

        option_mode = QtWidgets.QWidget()
        option_group_layout.addWidget(option_mode)
        option_mode_layout = QtWidgets.QHBoxLayout()
        option_mode_layout.setContentsMargins(0, 0, 0, 0)
        option_mode.setLayout(option_mode_layout)
        option_mode_layout.addWidget(QtWidgets.QLabel("Mode:"))
        self.opt_mode: list[QtWidgets.QRadioButton] = []
        self.opt_mode.append(QtWidgets.QRadioButton("full"))
        self.opt_mode.append(QtWidgets.QRadioButton("valid"))
        self.opt_mode[0].setChecked(True)
        for opt in self.opt_mode:
            option_mode_layout.addWidget(opt)

        option_part = QtWidgets.QWidget()
        option_group_layout.addWidget(option_part)
        option_part_layout = QtWidgets.QHBoxLayout()
        option_part_layout.setContentsMargins(0, 0, 0, 0)
        option_part.setLayout(option_part_layout)
        option_part_layout.addWidget(QtWidgets.QLabel("Part to Keep:"))
        self.opt_part: list[QtWidgets.QRadioButton] = []
        self.opt_part.append(QtWidgets.QRadioButton("both"))
        self.opt_part.append(QtWidgets.QRadioButton("below"))
        self.opt_part.append(QtWidgets.QRadioButton("above"))
        self.opt_part[0].setChecked(True)
        for opt in self.opt_part:
            option_part_layout.addWidget(opt)

        self.layout_.addRow(dim_group)
        self.layout_.addRow(option_group)

    @QtCore.Slot()
    def _update_spin(self) -> None:
        axis = self._dim_combo.currentIndex()
        self._center_spin.setRange(*map(float, self.array_slicer.lims_uniform[axis]))
        self._center_spin.setSingleStep(float(self.array_slicer.incs_uniform[axis]))
        self._center_spin.setDecimals(
            self.array_slicer.get_significant(axis, uniform=True)
        )
        self._center_spin.setValue(
            self.slicer_area.current_values_uniform[self._dim_combo.currentIndex()]
        )

    @property
    def _params(self) -> dict[str, typing.Any]:
        return {
            "dim": self._dim_combo.currentText(),
            "center": float(
                np.round(self._center_spin.value(), self._center_spin.decimals())
            ),
            "subtract": self.subtract_check.isChecked(),
            "mode": ("full", "valid")[
                next(i for i, opt in enumerate(self.opt_mode) if opt.isChecked())
            ],
            "part": ("both", "below", "above")[
                next(i for i, opt in enumerate(self.opt_part) if opt.isChecked())
            ],
        }

    def process_data(self, data: xr.DataArray) -> xr.DataArray:
        return erlab.analysis.transform.symmetrize(data, **self._params)

    def make_code(self) -> str:
        placeholder = self.slicer_area.watched_data_name or " "
        return erlab.interactive.utils.generate_code(
            erlab.analysis.transform.symmetrize,
            [f"|{placeholder}|"],
            self._params,
            module="era.transform",
        )


class SymmetrizeNfoldDialog(DataTransformDialog):
    title = "Rotational Symmetrize"
    enable_copy = True

    @property
    def suffix(self) -> str:
        return f"_symC{self._params['fold']}"

    @suffix.setter
    def suffix(self, value: str) -> None:
        # To satisfy mypy
        pass

    @property
    def _params(self) -> dict[str, typing.Any]:
        return {
            "fold": self.fold_spin.value(),
            "axes": self._axes,
            "center": {
                dim: float(np.round(spin.value(), spin.decimals()))
                for dim, spin in zip(self._axes, self.center_spins, strict=True)
            },
            "reshape": self.reshape_check.isChecked(),
            "order": self.order_spin.value(),
        }

    def setup_widgets(self) -> None:
        main_image = self.slicer_area.main_image

        self._axes: tuple[str, str] = (
            str(main_image.axis_dims_uniform[0]),
            str(main_image.axis_dims_uniform[1]),
        )

        plane_group = QtWidgets.QGroupBox("Rotation plane")
        plane_layout = QtWidgets.QHBoxLayout()
        plane_group.setLayout(plane_layout)
        plane_dims = ", ".join(self._axes)
        plane_label = QtWidgets.QLabel(plane_dims)
        plane_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        plane_layout.addWidget(plane_label)
        self.layout_.addRow(plane_group)

        self.fold_spin = QtWidgets.QSpinBox()
        self.fold_spin.setRange(2, 99)
        self.fold_spin.setValue(4)
        self.layout_.addRow("Fold", self.fold_spin)

        self.center_spins = (
            erlab.interactive.utils.BetterSpinBox(compact=False),
            erlab.interactive.utils.BetterSpinBox(compact=False),
        )
        for i in range(2):
            axis = main_image.display_axis[i]
            dim = str(main_image.axis_dims_uniform[i])
            self.center_spins[i].setDecimals(
                self.array_slicer.get_significant(axis, uniform=True)
            )
            self.center_spins[i].setSingleStep(float(self.array_slicer.incs_uniform[i]))
            self.center_spins[i].setValue(self.slicer_area.current_values_uniform[axis])
            self.layout_.addRow(f"Center {dim}", self.center_spins[i])

        self.order_spin = QtWidgets.QSpinBox()
        self.order_spin.setRange(0, 5)
        self.order_spin.setValue(1)
        self.layout_.addRow("Spline Order", self.order_spin)

        self.reshape_check = QtWidgets.QCheckBox("Reshape")
        self.reshape_check.setChecked(True)
        self.layout_.addRow(self.reshape_check)

        if main_image.is_guidelines_visible:
            for spin, val in zip(
                self.center_spins, main_image._guideline_offset, strict=True
            ):
                spin.setValue(val)
            self.fold_spin.setValue(2 * (len(main_image._guidelines_items) - 1))

    def process_data(self, data: xr.DataArray) -> xr.DataArray:
        return erlab.analysis.transform.symmetrize_nfold(data, **self._params)

    def make_code(self) -> str:
        placeholder = self.slicer_area.watched_data_name or " "
        return erlab.interactive.utils.generate_code(
            erlab.analysis.transform.symmetrize_nfold,
            [f"|{placeholder}|"],
            self._params,
            module="era.transform",
        )


class EdgeCorrectionDialog(DataTransformDialog):
    title = "Edge Correction"
    suffix = "_corr"
    enable_copy = False

    def setup_widgets(self) -> None:
        self.shift_coord_check = QtWidgets.QCheckBox("Shift Coordinates")
        self.shift_coord_check.setChecked(True)

        self.layout_.addRow(self.shift_coord_check)

    def process_data(self, data: xr.DataArray) -> xr.DataArray:
        return erlab.analysis.gold.correct_with_edge(
            data,
            self._edge_fit,
            shift_coords=self.shift_coord_check.isChecked(),
        )

    def _validate(self) -> QtWidgets.QDialog.DialogCode:
        if "eV" not in self.slicer_area.data.dims:
            QtWidgets.QMessageBox.warning(
                self,
                "No Energy Dimension",
                "Edge correction requires an energy dimension (eV) in the data.",
            )
            return QtWidgets.QDialog.DialogCode.Rejected

        self._edge_fit: xr.Dataset | None = erlab.interactive.utils.load_fit_ui()
        if self._edge_fit is None:
            # User canceled the fit dialog
            return QtWidgets.QDialog.DialogCode.Rejected
        return super()._validate()


class _BaseCropDialog(DataTransformDialog):
    suffix = "_crop"
    enable_copy = True

    @property
    def _slice_kwargs(self) -> dict[Hashable, slice]:
        raise NotImplementedError

    def process_data(self, data: xr.DataArray) -> xr.DataArray:
        return data.sel(self._slice_kwargs)

    def make_code(self) -> str:
        sel_kwargs: dict[Hashable, slice] = dict(self._slice_kwargs)
        isel_kwargs: dict[Hashable, slice] = {}

        for k in list(sel_kwargs.keys()):
            if str(k).endswith("_idx"):
                isel_kwargs[str(k).removesuffix("_idx")] = sel_kwargs.pop(k)

        out: str = self.slicer_area.watched_data_name or ""
        if sel_kwargs:
            out += f".sel({erlab.interactive.utils.format_kwargs(sel_kwargs)})"
        if isel_kwargs:
            out += f".isel({erlab.interactive.utils.format_kwargs(isel_kwargs)})"

        return out


class CropToViewDialog(_BaseCropDialog):
    title = "Crop to View"
    whatsthis = "Crop the data to the currently visible area."

    def setup_widgets(self) -> None:
        dim_group = QtWidgets.QGroupBox("Dimensions")
        dim_layout = QtWidgets.QVBoxLayout()
        dim_group.setLayout(dim_layout)

        self.dim_checks: dict[Hashable, QtWidgets.QCheckBox] = {}

        for i, d in enumerate(self.slicer_area.data.dims):
            self.dim_checks[d] = QtWidgets.QCheckBox(str(d))
            dim_layout.addWidget(self.dim_checks[d])
            if i < 2:
                # Enable first 2 dimensions by default
                self.dim_checks[d].setChecked(True)

            if d not in self.slicer_area.manual_limits:
                # Disable dimensions without manual limits
                self.dim_checks[d].setChecked(False)
                self.dim_checks[d].setDisabled(True)

        self.layout_.addRow(dim_group)

    @property
    def _slice_kwargs(self) -> dict[Hashable, slice]:
        return {
            k: v
            for k, v in self.slicer_area.make_slice_dict().items()
            if self.dim_checks[k].isChecked()
        }

    @QtCore.Slot()
    def accept(self) -> None:
        if self._slice_kwargs == {}:
            QtWidgets.QMessageBox.warning(
                self,
                "No Dimensions Selected",
                "You need to select at least one dimension with manual limits.",
            )
            return
        super().accept()

    def _validate(self) -> QtWidgets.QDialog.DialogCode:
        if len(self.slicer_area.manual_limits) == 0:
            QtWidgets.QMessageBox.warning(
                self, "Nothing to Crop", "Manually zoom in to define the crop area."
            )
            return QtWidgets.QDialog.DialogCode.Rejected
        return super()._validate()


class CropDialog(_BaseCropDialog):
    title = "Crop Between Cursors"

    @property
    def _enabled_dims(self) -> list[Hashable]:
        return [k for k, v in self.dim_checks.items() if v.isChecked()]

    @property
    def _cursor_indices(self) -> tuple[int, int]:
        return typing.cast(
            "tuple[int, int]",
            tuple(combo.currentIndex() for combo in self.cursor_combos),
        )

    @property
    def _slice_kwargs(self) -> dict[Hashable, slice]:
        c0, c1 = self._cursor_indices

        slice_dict: dict[Hashable, slice] = {}

        for k in self._enabled_dims:
            ax_idx = self.slicer_area.data.dims.index(k)
            sig_digits = self.array_slicer.get_significant(ax_idx, uniform=True)

            start = self.array_slicer.get_value(cursor=c0, axis=ax_idx, uniform=True)
            end = self.array_slicer.get_value(cursor=c1, axis=ax_idx, uniform=True)

            if start > end:
                start, end = end, start

            start = float(np.round(start, sig_digits))
            end = float(np.round(end, sig_digits))

            if sig_digits == 0:
                start, end = int(start), int(end)

            slice_dict[k] = slice(start, end)

        return slice_dict

    def _validate(self) -> QtWidgets.QDialog.DialogCode:
        if self.slicer_area.n_cursors == 1:
            QtWidgets.QMessageBox.warning(
                self, "Only 1 Cursor", "You need at least 2 cursors to crop the data."
            )
            return QtWidgets.QDialog.DialogCode.Rejected
        return super()._validate()

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

        self._cursors_group = erlab.interactive.utils.ExclusiveComboGroup(self)

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


class NormalizeDialog(DataFilterDialog):
    title = "Normalize"
    enable_copy = False

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

    def process_data(self, data: xr.DataArray) -> xr.DataArray:
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


class GaussianFilterDialog(DataFilterDialog):
    title = "Gaussian Filter"
    enable_copy = True

    def setup_widgets(self) -> None:
        self._source_data = self.slicer_area._data
        self.dim_checks: dict[Hashable, QtWidgets.QCheckBox] = {}
        self.sigma_spins: dict[Hashable, erlab.interactive.utils.BetterSpinBox] = {}
        self.fwhm_spins: dict[Hashable, erlab.interactive.utils.BetterSpinBox] = {}

        dim_group = QtWidgets.QGroupBox("Dimensions")
        dim_layout = QtWidgets.QGridLayout()
        dim_group.setLayout(dim_layout)

        dim_layout.addWidget(QtWidgets.QLabel("Dimension"), 0, 0)
        dim_layout.addWidget(QtWidgets.QLabel("Sigma"), 0, 1)
        dim_layout.addWidget(QtWidgets.QLabel("FWHM"), 0, 2)

        for row, dim in enumerate(self._source_data.dims, start=1):
            check = QtWidgets.QCheckBox(str(dim))
            sigma_spin = erlab.interactive.utils.BetterSpinBox(compact=False)
            fwhm_spin = erlab.interactive.utils.BetterSpinBox(compact=False)

            sigma_spin.setMinimum(0.0)
            fwhm_spin.setMinimum(0.0)

            support_reason = self._unsupported_reason(dim)
            if support_reason is None:
                step = self._coord_step(dim)
                sigma_spin.setSingleStep(step)
                fwhm_spin.setSingleStep(step * _GAUSSIAN_FWHM_FACTOR)

                sigma_spin.setDecimals(self._sigma_decimals(dim))
                fwhm_spin.setDecimals(self._fwhm_decimals(dim, step))

                sigma_spin.setValue(step)
                fwhm_spin.setValue(step * _GAUSSIAN_FWHM_FACTOR)

                sigma_spin.setEnabled(False)
                fwhm_spin.setEnabled(False)

                check.toggled.connect(sigma_spin.setEnabled)
                check.toggled.connect(fwhm_spin.setEnabled)
                sigma_spin.valueChanged.connect(
                    lambda _value, dim=dim: self._sync_from_sigma(dim)
                )
                fwhm_spin.valueChanged.connect(
                    lambda _value, dim=dim: self._sync_from_fwhm(dim)
                )
            else:
                for widget in (check, sigma_spin, fwhm_spin):
                    widget.setToolTip(support_reason)
                    widget.setEnabled(False)

            self.dim_checks[dim] = check
            self.sigma_spins[dim] = sigma_spin
            self.fwhm_spins[dim] = fwhm_spin

            dim_layout.addWidget(check, row, 0)
            dim_layout.addWidget(sigma_spin, row, 1)
            dim_layout.addWidget(fwhm_spin, row, 2)

        self.layout_.addRow(dim_group)

    def _unsupported_reason(self, dim: Hashable) -> str | None:
        coord = np.asarray(self._source_data[dim].values, dtype=np.float64)
        if coord.size < 2:
            return "Gaussian filtering requires at least two coordinate values."
        if np.allclose(np.diff(coord), 0.0):
            return "Gaussian filtering does not support constant coordinates."
        if not erlab.utils.array.is_uniform_spaced(coord):
            return "Gaussian filtering requires uniformly spaced coordinates."
        return None

    def _coord_step(self, dim: Hashable) -> float:
        coord = np.asarray(self._source_data[dim].values, dtype=np.float64)
        return float(np.abs(coord[1] - coord[0]))

    def _sigma_decimals(self, dim: Hashable) -> int:
        return erlab.utils.array.effective_decimals(
            np.asarray(self._source_data[dim].values, dtype=np.float64)
        )

    def _fwhm_decimals(self, dim: Hashable, sigma: float) -> int:
        sigma_decimals = self._sigma_decimals(dim)
        fwhm_decimals = max(
            sigma_decimals,
            erlab.utils.array.effective_decimals(sigma * _GAUSSIAN_FWHM_FACTOR),
        )

        sigma_literal = self._format_literal(sigma, sigma_decimals)
        while (
            fwhm_decimals < 15
            and self._roundtrip_sigma_literal(
                sigma_literal, sigma_decimals, fwhm_decimals
            )
            != sigma_literal
        ):
            fwhm_decimals += 1
        return fwhm_decimals

    def _format_literal(self, value: float, decimals: int) -> str:
        return np.format_float_positional(
            value, precision=decimals, unique=False, fractional=True, trim="k"
        )

    def _roundtrip_sigma_literal(
        self, sigma_literal: str, sigma_decimals: int, fwhm_decimals: int
    ) -> str:
        fwhm_literal = self._format_literal(
            float(sigma_literal) * _GAUSSIAN_FWHM_FACTOR, fwhm_decimals
        )
        return self._format_literal(
            float(fwhm_literal) / _GAUSSIAN_FWHM_FACTOR, sigma_decimals
        )

    def _spin_literal(self, spin: erlab.interactive.utils.BetterSpinBox) -> str:
        return spin.text().removeprefix(spin.prefix())

    def _spin_value(self, spin: erlab.interactive.utils.BetterSpinBox) -> float:
        return float(self._spin_literal(spin))

    def _commit_numeric_inputs(self) -> None:
        for spin in (*self.sigma_spins.values(), *self.fwhm_spins.values()):
            line = spin.lineEdit()
            if line is None or not line.isModified():
                continue
            spin.editingFinishedEvent()
            line.setModified(False)

    @QtCore.Slot()
    def _sync_from_sigma(self, dim: Hashable) -> None:
        value = self._spin_value(self.sigma_spins[dim]) * _GAUSSIAN_FWHM_FACTOR
        with QtCore.QSignalBlocker(self.fwhm_spins[dim]):
            self.fwhm_spins[dim].setValue(value)

    @QtCore.Slot()
    def _sync_from_fwhm(self, dim: Hashable) -> None:
        value = self._spin_value(self.fwhm_spins[dim]) / _GAUSSIAN_FWHM_FACTOR
        with QtCore.QSignalBlocker(self.sigma_spins[dim]):
            self.sigma_spins[dim].setValue(value)

    def _sigma_values(self) -> tuple[dict[Hashable, float], dict[Hashable, str]]:
        self._commit_numeric_inputs()

        sigma_values: dict[Hashable, float] = {}
        sigma_literals: dict[Hashable, str] = {}
        for dim in self._source_data.dims:
            if not self.dim_checks[dim].isChecked():
                continue
            sigma_literal = self._spin_literal(self.sigma_spins[dim])
            sigma_literals[dim] = sigma_literal
            sigma_values[dim] = float(sigma_literal)
        return sigma_values, sigma_literals

    def process_data(self, data: xr.DataArray) -> xr.DataArray:
        sigma_values, _ = self._sigma_values()
        if not sigma_values:
            return data
        return erlab.analysis.image.gaussian_filter(data, sigma=sigma_values)

    def make_code(self) -> str:
        _, sigma_literals = self._sigma_values()
        if not sigma_literals:
            return ""

        placeholder = self.slicer_area.watched_data_name or " "
        return erlab.interactive.utils.generate_code(
            erlab.analysis.image.gaussian_filter,
            [f"|{placeholder}|"],
            {"sigma": {dim: f"|{literal}|" for dim, literal in sigma_literals.items()}},
            module="era.image",
        )


class SwapDimsDialog(DataTransformDialog):
    title = "Swap Dimensions"
    enable_copy = True
    apply_on_nonuniform_data = True

    def setup_widgets(self) -> None:
        self._source_data = erlab.interactive.imagetool.slicer.restore_nonuniform_dims(
            self.slicer_area.data
        )
        public_dims = tuple(
            dim.removesuffix("_idx")
            if isinstance(dim, str) and dim.endswith("_idx")
            else dim
            for dim in self.slicer_area.data.dims
        )
        if tuple(self._source_data.dims) != public_dims and set(
            self._source_data.dims
        ) == set(public_dims):
            self._source_data = self._source_data.transpose(*public_dims)

        dim_group = QtWidgets.QGroupBox("Dimensions")
        dim_layout = QtWidgets.QGridLayout()
        dim_group.setLayout(dim_layout)

        dim_layout.addWidget(QtWidgets.QLabel("Dimension"), 0, 0)
        dim_layout.addWidget(QtWidgets.QLabel("Target"), 0, 1)

        self.source_labels: dict[Hashable, QtWidgets.QLabel] = {}
        self.target_combos: dict[Hashable, QtWidgets.QComboBox] = {}
        self.target_options: dict[Hashable, tuple[Hashable, ...]] = {}

        for row, dim in enumerate(self._source_data.dims, start=1):
            label = QtWidgets.QLabel(str(dim))
            combo = QtWidgets.QComboBox()

            targets = [dim]
            targets.extend(
                name
                for name, coord in self._source_data.coords.items()
                if name != dim and coord.ndim == 1 and coord.dims == (dim,)
            )
            self.target_options[dim] = tuple(targets)

            for target in self.target_options[dim]:
                combo.addItem(str(target), userData=target)

            if len(self.target_options[dim]) == 1:
                tooltip = (
                    "No compatible 1D coordinates are available for this dimension."
                )
                label.setToolTip(tooltip)
                combo.setToolTip(tooltip)
                combo.setEnabled(False)

            self.source_labels[dim] = label
            self.target_combos[dim] = combo

            dim_layout.addWidget(label, row, 0)
            dim_layout.addWidget(combo, row, 1)

        self.layout_.addRow(dim_group)

    @property
    def _swap_mapping(self) -> dict[Hashable, Hashable]:
        mapping: dict[Hashable, Hashable] = {}
        for dim, combo in self.target_combos.items():
            if not combo.isEnabled():
                continue
            target = typing.cast(
                "Hashable | None",
                combo.currentData(QtCore.Qt.ItemDataRole.UserRole),
            )
            if target is not None and target != dim:
                mapping[dim] = target
        return mapping

    def _validate(self) -> QtWidgets.QDialog.DialogCode:
        if not any(combo.isEnabled() for combo in self.target_combos.values()):
            QtWidgets.QMessageBox.warning(
                self,
                "Nothing to Swap",
                "No compatible 1D coordinates are available for any dimension.",
            )
            return QtWidgets.QDialog.DialogCode.Rejected
        return super()._validate()

    @QtCore.Slot()
    def accept(self) -> None:
        if not self._swap_mapping:
            QtWidgets.QMessageBox.warning(
                self,
                "No Dimensions Changed",
                "Choose at least one dimension to swap.",
            )
            return
        super().accept()

    def process_data(self, data: xr.DataArray) -> xr.DataArray:
        return data.swap_dims(self._swap_mapping)

    def make_code(self) -> str:
        if not self._swap_mapping:
            return ""

        placeholder = self.slicer_area.watched_data_name or ""
        kwargs = erlab.interactive.utils.format_kwargs(self._swap_mapping)
        return f"{placeholder}.swap_dims({kwargs})"


class _CoordinateWidget(QtWidgets.QWidget):
    def __init__(self, values: npt.NDArray) -> None:
        super().__init__()
        self.init_ui()
        self.set_old_coord(values)

    def init_ui(self):
        container_layout = QtWidgets.QHBoxLayout(self)
        container_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(container_layout)

        left_widget = QtWidgets.QWidget()
        container_layout.addWidget(left_widget)
        left_layout = QtWidgets.QFormLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_widget.setLayout(left_layout)

        self.spin0 = erlab.interactive.utils.BetterSpinBox(compact=False, trim="0")
        self.spin0.valueChanged.connect(self.update_table)
        left_layout.addRow("Start", self.spin0)

        self.spin1 = erlab.interactive.utils.BetterSpinBox(compact=False, trim="0")
        self.spin1.valueChanged.connect(self.update_table)

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["End", "Delta"])
        self.mode_combo.setCurrentIndex(0)
        self.mode_combo.currentTextChanged.connect(self.mode_changed)
        left_layout.addRow(self.mode_combo, self.spin1)

        self.reset_btn = QtWidgets.QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset)
        left_layout.addRow(self.reset_btn)

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(1)
        self.table.horizontalHeader().setVisible(False)
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setSortingEnabled(False)
        self.table.setAlternatingRowColors(True)
        left_layout.addRow(self.table)

    @QtCore.Slot()
    def mode_changed(self) -> None:
        """Handle the change of the mode combo box."""
        new_mode = self.mode_combo.currentText()
        self.spin1.blockSignals(True)
        match new_mode:
            case "End":
                self.spin1.setValue(self._current_values_delta[-1])
            case "Delta":
                arr = self._current_values_end
                self.spin1.setValue(arr[1] - arr[0])
        self.spin1.blockSignals(False)

    @QtCore.Slot()
    def reset(self) -> None:
        """Reset the spin boxes to the original values."""
        is_scalar: bool = np.atleast_1d(self._old_coord).size == 1
        self.spin0.setDisabled(is_scalar)
        self.spin1.setDisabled(is_scalar)
        self.mode_combo.setDisabled(is_scalar)

        if not is_scalar:
            with QtCore.QSignalBlocker(self.spin0), QtCore.QSignalBlocker(self.spin1):
                decimals = erlab.utils.array.unique_decimals(self._old_coord)
                self.spin0.setDecimals(decimals)
                self.spin1.setDecimals(decimals)

                if erlab.utils.array.is_uniform_spaced(self._old_coord):
                    self.spin0.setValue(float(self._old_coord[0]))
                    if self.mode_combo.currentText() == "End":
                        self.spin1.setValue(float(self._old_coord[-1]))
                    else:
                        self.spin1.setValue(self._old_coord[1] - self._old_coord[0])
                else:
                    self.spin0.setValue(0.0)
                    self.spin1.setValue(0.0)

        self._set_table_values(np.atleast_1d(self._old_coord))

    def set_old_coord(self, values: npt.NDArray) -> None:
        """Set the old coordinates to the given values."""
        self._old_coord = values.copy()
        self.reset()

    @property
    def _current_values_end(self) -> npt.NDArray:
        """Get the current values assuming spin1 value is the end."""
        return np.linspace(self.spin0.value(), self.spin1.value(), len(self._old_coord))

    @property
    def _current_values_delta(self) -> npt.NDArray:
        """Get the current values assuming spin1 value is the step size."""
        sz: int = len(self._old_coord)
        return np.linspace(
            self.spin0.value(),
            self.spin0.value() + self.spin1.value() * (sz - 1),
            sz,
        )

    @property
    def new_coord(self) -> npt.NDArray:
        """Get the edited coordinates as a numpy array."""
        values = self._old_coord.copy()
        for i in range(self.table.rowCount()):
            item = self.table.item(i, 0)
            if item is not None and item.text():
                try:
                    values[i] = float(item.text())
                except Exception as e:
                    raise ValueError(f"Invalid value in row {i}: {item.text()}") from e
        return values

    @QtCore.Slot()
    def update_table(self) -> None:
        """Update the table with the current values from the spin boxes."""
        match self.mode_combo.currentText():
            case "End":
                vals = self._current_values_end
            case _:
                vals = self._current_values_delta
        self._set_table_values(vals)

    def _set_table_values(self, values: npt.NDArray) -> None:
        """Set the table contents to the given numpy array."""
        self.table.setRowCount(len(values))
        # Make zero-based
        self.table.setVerticalHeaderLabels([str(i) for i in range(len(values))])
        for i, val in enumerate(values):
            item = QtWidgets.QTableWidgetItem(np.format_float_positional(val, trim="0"))
            item.setTextAlignment(
                QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
            self.table.setItem(i, 0, item)
        self.table.resizeColumnsToContents()
        self.setMinimumWidth(
            self.table.horizontalHeader().length() + self.table.verticalHeader().width()
        )


class AssignCoordsDialog(DataTransformDialog):
    title = "Coordinate Editor"

    def setup_widgets(self) -> None:
        existing_widget = QtWidgets.QWidget(self)
        existing_layout = QtWidgets.QVBoxLayout(existing_widget)
        existing_layout.setContentsMargins(0, 0, 0, 0)
        self.layout_.addRow(existing_widget)

        existing_coord_names: list[str] = [str(d) for d in self.slicer_area.data.dims]
        existing_coord_names.extend(
            str(k)
            for k in self.slicer_area.data.coords
            if k not in self.slicer_area.data.dims
        )
        self._coord_combo = QtWidgets.QComboBox()
        self._coord_combo.addItems(existing_coord_names)
        existing_layout.addWidget(self._coord_combo)

        self._coord_combo.currentTextChanged.connect(self._coord_selection_changed)

        self.coord_widget = _CoordinateWidget(np.array([0, 1]))
        self._coord_selection_changed()
        existing_layout.addWidget(self.coord_widget)

    @property
    def current_coord_name(self) -> str:
        """Get the name of the currently selected coordinate."""
        return self._coord_combo.currentText()

    @QtCore.Slot()
    def _coord_selection_changed(self) -> None:
        self.coord_widget.set_old_coord(
            self.slicer_area.data[self.current_coord_name].values
        )

    def process_data(self, data: xr.DataArray) -> xr.DataArray:
        return erlab.utils.array.sort_coord_order(
            data.assign_coords(
                {
                    self.current_coord_name: data[self.current_coord_name].copy(
                        data=self.coord_widget.new_coord
                    )
                }
            ),
            keys=data.coords.keys(),
            dims_first=False,
        )


class ROIPathDialog(DataTransformDialog):
    title = "Slice Along ROI Path"
    enable_copy = True
    apply_on_nonuniform_data = True

    @property
    def suffix(self) -> str:
        return "_path"

    @suffix.setter
    def suffix(self, value: str) -> None:
        # To satisfy mypy
        pass

    def __init__(self, roi: ItoolPolyLineROI) -> None:
        self.roi = roi
        super().__init__(self.roi.plot_item.slicer_area)

    def setup_widgets(self) -> None:
        group = QtWidgets.QGroupBox()
        layout = QtWidgets.QFormLayout()
        group.setLayout(layout)

        decimals = max(
            self.array_slicer.get_significant(ax, uniform=False)
            for ax in self.roi.plot_item.display_axis
        )
        default_step = min(
            self.roi.slicer_area.array_slicer.incs[ax]
            for ax in self.roi.plot_item.display_axis
        )  # Reasonable default step size

        # TODO: add vertice customization

        self._step_spin = pg.SpinBox()
        self._step_spin.setDecimals(decimals)
        self._step_spin.setSingleStep(10 ** (-decimals))
        self._step_spin.setMinimum(10 ** (-decimals - 1))
        self._step_spin.setOpts(compactHeight=False)
        self._step_spin.setValue(round(default_step, decimals))

        layout.addRow("Step Size", self._step_spin)

        self._dim_name_line = QtWidgets.QLineEdit()
        self._dim_name_line.setText("path")
        layout.addRow("New Dim Name", self._dim_name_line)

        self.layout_.addRow(group)

    @property
    def _params(self) -> dict[str, typing.Any]:
        vert_dict = self.roi._get_vertices()

        if self.roi.closed:
            for k, v in dict(vert_dict).items():
                vert_dict[k] = [*v, v[0]]  # Close the path

        return {
            "vertices": vert_dict,
            "step_size": self._step_spin.value(),
            "dim_name": self._dim_name_line.text().strip(),
        }

    def process_data(self, data: xr.DataArray) -> xr.DataArray:
        return erlab.analysis.interpolate.slice_along_path(data, **self._params)

    def make_code(self) -> str:
        placeholder = self.slicer_area.watched_data_name or " "
        return erlab.interactive.utils.generate_code(
            erlab.analysis.interpolate.slice_along_path,
            [f"|{placeholder}|"],
            self._params,
            module="era.interpolate",
        )


class ROIMaskDialog(DataTransformDialog):
    title = "Mask with ROI"
    enable_copy = True
    apply_on_nonuniform_data = True

    @property
    def suffix(self) -> str:
        return "_masked"

    @suffix.setter
    def suffix(self, value: str) -> None:
        # To satisfy mypy
        pass

    def __init__(self, roi: ItoolPolyLineROI) -> None:
        self.roi = roi
        super().__init__(self.roi.plot_item.slicer_area)

    def setup_widgets(self) -> None:
        group = QtWidgets.QGroupBox()
        layout = QtWidgets.QFormLayout()
        group.setLayout(layout)

        self._invert_check = QtWidgets.QCheckBox("Invert Mask")
        layout.addRow(self._invert_check)

        self._drop_check = QtWidgets.QCheckBox("Drop Masked Values")
        layout.addRow(self._drop_check)

        self.layout_.addRow(group)

    @property
    def _params(self) -> dict[str, typing.Any]:
        vert_dict = self.roi._get_vertices()
        return {
            "vertices": np.column_stack(tuple(vert_dict.values())),
            "dims": tuple(vert_dict.keys()),
            "invert": self._invert_check.isChecked(),
            "drop": self._drop_check.isChecked(),
        }

    def process_data(self, data: xr.DataArray) -> xr.DataArray:
        return erlab.analysis.mask.mask_with_polygon(data, **self._params)

    def make_code(self) -> str:
        placeholder = self.slicer_area.watched_data_name or " "
        return erlab.interactive.utils.generate_code(
            erlab.analysis.mask.mask_with_polygon,
            [f"|{placeholder}|"],
            self._params,
            module="era.mask",
        )
