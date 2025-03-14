"""Interactive momentum conversion tool.

.. image:: ../images/ktool_1_light.png
    :align: center
    :alt: Momentum conversion tool screenshot
    :class: only-light

.. only:: format_html

    .. image:: ../images/ktool_1_dark.png
        :align: center
        :alt: Momentum conversion tool screenshot
        :class: only-dark
"""

from __future__ import annotations

__all__ = ["ktool"]

import functools
import os
import typing
import warnings

import numpy as np
import numpy.typing as npt
import pyqtgraph as pg
from qtpy import QtCore, QtGui, QtWidgets, uic

import erlab
from erlab.accessors.kspace import MomentumAccessor

if typing.TYPE_CHECKING:
    import matplotlib
    import varname
    import xarray as xr
else:
    import lazy_loader as _lazy

    matplotlib = _lazy.load("matplotlib")
    varname = _lazy.load("varname")


class _CircleROIControlWidget(QtWidgets.QWidget):
    def __init__(self, roi: _MovableCircleROI) -> None:
        super().__init__()
        self.setGeometry(QtCore.QRect(0, 640, 242, 182))
        self._roi = roi

        layout = QtWidgets.QFormLayout(self)
        self.setLayout(layout)

        self.x_spin = pg.SpinBox(dec=True, compactHeight=False)
        self.y_spin = pg.SpinBox(dec=True, compactHeight=False)
        self.r_spin = pg.SpinBox(dec=True, compactHeight=False)
        self.x_spin.sigValueChanged.connect(self.update_roi)
        self.y_spin.sigValueChanged.connect(self.update_roi)
        self.r_spin.sigValueChanged.connect(self.update_roi)

        layout.addRow("X", self.x_spin)
        layout.addRow("Y", self.y_spin)
        layout.addRow("Radius", self.r_spin)

        self._roi.sigRegionChanged.connect(self.update_spins)

    @QtCore.Slot()
    def update_roi(self) -> None:
        self._roi.blockSignals(True)
        self._roi.set_position(
            (self.x_spin.value(), self.y_spin.value()), self.r_spin.value()
        )
        self._roi.blockSignals(False)

    @QtCore.Slot()
    def update_spins(self) -> None:
        x, y, r = self._roi.get_position()
        self.x_spin.blockSignals(True)
        self.y_spin.blockSignals(True)
        self.r_spin.blockSignals(True)
        self.x_spin.setValue(x)
        self.y_spin.setValue(y)
        self.r_spin.setValue(r)
        self.x_spin.blockSignals(False)
        self.y_spin.blockSignals(False)
        self.r_spin.blockSignals(False)

    def setVisible(self, visible: bool) -> None:
        super().setVisible(visible)
        if visible:
            self.update_spins()


class _MovableCircleROI(pg.CircleROI):
    """Circle ROI with a menu to control position and radius."""

    def __init__(self, pos, size=None, radius=None, **args):
        args.setdefault("removable", True)
        super().__init__(pos, size, radius, **args)

    def getMenu(self):
        if self.menu is None:
            self.menu = QtWidgets.QMenu()
            self.menu.setTitle("ROI")
            if self.removable:
                remAct = QtGui.QAction("Remove Circle", self.menu)
                remAct.triggered.connect(self.removeClicked)
                self.menu.addAction(remAct)
                self.menu.remAct = remAct
            self._pos_menu = self.menu.addMenu("Edit Circle")
            ctrlAct = QtWidgets.QWidgetAction(self._pos_menu)
            ctrlAct.setDefaultWidget(_CircleROIControlWidget(self))
            self._pos_menu.addAction(ctrlAct)

        return self.menu

    def radius(self) -> float:
        """Radius of the circle."""
        return float(self.size()[0] / 2)

    def center(self) -> tuple[float, float]:
        """Center of the circle."""
        x, y = self.pos()
        r = self.radius()
        return x + r, y + r

    def get_position(self) -> tuple[float, float, float]:
        """Return the center and radius of the circle."""
        return (*self.center(), self.radius())

    def set_position(self, center, radius: float | None = None) -> None:
        """Set the center and radius of the circle."""
        if radius is None:
            radius = self.radius()
        else:
            diameter = 2 * radius
            self.setSize((diameter, diameter), update=False)
        self.setPos(center[0] - radius, center[1] - radius)


class KspaceToolGUI(
    *uic.loadUiType(os.path.join(os.path.dirname(__file__), "ktool.ui"))  # type: ignore[misc]
):
    def __init__(
        self,
        avec: npt.NDArray | None = None,
        rotate_bz: float = 0.0,
        cmap: str | None = None,
        gamma: float = 0.5,
    ) -> None:
        # Initialize UI
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Momentum Conversion")

        self.plotitems: tuple[pg.PlotItem, pg.PlotItem] = (pg.PlotItem(), pg.PlotItem())
        self.images: tuple[
            erlab.interactive.utils.xImageItem, erlab.interactive.utils.xImageItem
        ] = (
            erlab.interactive.utils.xImageItem(axisOrder="row-major"),
            erlab.interactive.utils.xImageItem(axisOrder="row-major"),
        )

        for i, plot in enumerate(self.plotitems):
            self.graphics_layout.addItem(plot, i, 0)
            plot.addItem(self.images[i])
            plot.showGrid(x=True, y=True, alpha=0.5)

        if cmap is None:
            cmap = matplotlib.rcParams["image.cmap"]

        if cmap.endswith("_r"):
            cmap = cmap[:-2]
            self.invert_check.setChecked(True)

        # Set up colormap controls
        self.cmap_combo.setDefaultCmap(cmap)
        self.cmap_combo.textActivated.connect(self.update_cmap)
        self.gamma_widget.setValue(gamma)
        self.gamma_widget.valueChanged.connect(self.update_cmap)
        self.invert_check.stateChanged.connect(self.update_cmap)
        self.contrast_check.stateChanged.connect(self.update_cmap)
        self.update_cmap()

        # Set up BZ controls
        self.bz_group.toggled.connect(self.update_bz)
        self.a_spin.valueChanged.connect(self.update_bz)
        self.b_spin.valueChanged.connect(self.update_bz)
        self.ang_spin.valueChanged.connect(self.update_bz)
        self.rot_spin.valueChanged.connect(self.update_bz)
        self.c_spin.valueChanged.connect(self.update_bz)
        self.ab_spin.valueChanged.connect(self.update_bz)
        self.n1_spin.valueChanged.connect(self.update_bz)
        self.n2_spin.valueChanged.connect(self.update_bz)
        self.reciprocal_check.stateChanged.connect(self.update_bz)
        self.points_check.stateChanged.connect(self.update_bz)

        self.plotitems[0].setVisible(False)
        self.angle_plot_check.stateChanged.connect(
            lambda: self.plotitems[0].setVisible(self.angle_plot_check.isChecked())
        )

        self._roi_list: list[_MovableCircleROI] = []
        self.add_circle_btn.clicked.connect(self._add_circle)

        if avec is not None:
            self._populate_bz(avec)
        self.rot_spin.setValue(rotate_bz)

    def _populate_bz(self, avec) -> None:
        if avec.shape == (2, 2):
            avec = np.pad(avec, ((0, 1), (0, 1)))
            avec[2, 2] = 1.0e-15
        a, b, c, _, _, gamma = erlab.lattice.avec2abc(avec)
        self.a_spin.setValue(a)
        self.b_spin.setValue(b)
        self.c_spin.setValue(c)
        self.ab_spin.setValue(a)
        self.ang_spin.setValue(gamma)

    @QtCore.Slot()
    def _add_circle(self) -> None:
        roi = _MovableCircleROI(
            [-0.3, -0.3], radius=0.3, removable=True, pen=pg.mkPen("m", width=2)
        )
        self.plotitems[1].addItem(roi)
        self._roi_list.append(roi)

        def _remove_roi():
            self.plotitems[1].removeItem(roi)
            self._roi_list.remove(roi)

        roi.sigRemoveRequested.connect(_remove_roi)

    def update_cmap(self) -> None:
        name = self.cmap_combo.currentText()
        for im in self.images:
            im.set_colormap(
                name,
                self.gamma_widget.value(),
                reverse=self.invert_check.isChecked(),
                high_contrast=self.contrast_check.isChecked(),
                update=True,
            )

    def get_bz_lines(self):
        raise NotImplementedError

    def update_bz(self) -> None:
        self.plotitems[1].clearPlots()
        if not self.bz_group.isChecked():
            return
        if self.reciprocal_check.isChecked():
            self.a_spin.setSuffix(" Å⁻¹")
            self.b_spin.setSuffix(" Å⁻¹")
            self.c_spin.setSuffix(" Å⁻¹")
            self.ab_spin.setSuffix(" Å⁻¹")
        else:
            self.a_spin.setSuffix(" Å")
            self.b_spin.setSuffix(" Å")
            self.c_spin.setSuffix(" Å")
            self.ab_spin.setSuffix(" Å")

        lines, vertices = self.get_bz_lines()
        for line in lines:
            self.plotitems[1].plot(line[:, 0], line[:, 1], pen=pg.mkPen("m", width=2))
            vertices = np.vstack((vertices, np.mean(line, axis=0)))

        if self.points_check.isChecked():
            self.plotitems[1].plot(
                vertices[:, 0],
                vertices[:, 1],
                symbol="o",
                pen=pg.mkColor(255, 255, 255, 0),
                symbolPen=pg.mkColor(255, 255, 255, 0),
                symbolBrush=pg.mkColor("m"),
                symbolSize=6,
            )


class KspaceTool(KspaceToolGUI):
    def __init__(
        self,
        data: xr.DataArray,
        avec: npt.NDArray | None = None,
        rotate_bz: float = 0.0,
        cmap: str | None = None,
        gamma: float = 0.5,
        *,
        data_name: str | None = None,
    ) -> None:
        super().__init__(avec=avec, rotate_bz=rotate_bz, cmap=cmap, gamma=gamma)

        self._argnames = {}

        self._itool: QtWidgets.QWidget | None = None

        if data_name is None:
            try:
                self._argnames["data"] = varname.argname(
                    "data",
                    func=self.__init__,  # type: ignore[misc]
                    vars_only=False,
                )
            except varname.VarnameRetrievingError:
                self._argnames["data"] = "data"
        else:
            self._argnames["data"] = data_name

        self.data: xr.DataArray = data.copy(deep=True)

        if self.data.kspace._has_eV:
            self.center_spin.setRange(self.data.eV[0], self.data.eV[-1])
            eV_step = self.data.eV.values[1] - self.data.eV.values[0]
            self.center_spin.setDecimals(erlab.utils.array.effective_decimals(eV_step))
            self.center_spin.setSingleStep(eV_step)
            self.width_spin.setRange(1, len(self.data.eV))
            self.center_spin.valueChanged.connect(self.update)
            self.width_spin.valueChanged.connect(self.update)
        else:
            if "eV" in self.data.coords and self.data["eV"].size == 1:
                fixed_energy = float(self.data.eV)
                # Although spinbox will be disabled, setting range and value for
                # displaying the fixed energy value.
                self.center_spin.setRange(fixed_energy - 0.1, fixed_energy + 0.1)
                self.center_spin.setValue(fixed_energy)

            self.energy_group.setDisabled(True)

        self.bounds_supergroup.toggled.connect(self.update)
        self.resolution_supergroup.toggled.connect(self.update)

        self._offset_spins: dict[str, QtWidgets.QDoubleSpinBox] = {}
        offset_labels = {"delta": "𝛿", "chi": "𝜒₀", "xi": "𝜉₀", "beta": "𝛽₀"}
        for k in self.data.kspace._valid_offset_keys:
            self._offset_spins[k] = QtWidgets.QDoubleSpinBox()
            self._offset_spins[k].setRange(-180, 180)
            self._offset_spins[k].setSingleStep(0.01)
            self._offset_spins[k].setDecimals(3)
            self._offset_spins[k].setValue(self.data.kspace.offsets[k])
            self._offset_spins[k].valueChanged.connect(self.update)
            self._offset_spins[k].setSuffix("°")
            self.offsets_group.layout().addRow(offset_labels[k], self._offset_spins[k])

        if self.data.kspace._has_hv:
            self._offset_spins["V0"] = QtWidgets.QDoubleSpinBox()
            self._offset_spins["V0"].setRange(0, 100)
            self._offset_spins["V0"].setSingleStep(1)
            self._offset_spins["V0"].setDecimals(1)
            self._offset_spins["V0"].setSuffix(" eV")
            self._offset_spins["V0"].setToolTip("Inner potential of the sample.")
            with warnings.catch_warnings(action="ignore", category=UserWarning):
                self._offset_spins["V0"].setValue(self.data.kspace.inner_potential)
            self._offset_spins["V0"].valueChanged.connect(self.update)
            self.offsets_group.layout().addRow("V₀", self._offset_spins["V0"])

            for i in range(8):
                self.bz_form.setRowVisible(i, i not in (0, 1, 2))
        else:
            for i in range(8):
                self.bz_form.setRowVisible(i, i not in (3, 4))

        # Work function spinbox
        self._offset_spins["wf"] = QtWidgets.QDoubleSpinBox()
        self._offset_spins["wf"].setRange(0.0, 9.999)
        self._offset_spins["wf"].setSingleStep(0.01)
        self._offset_spins["wf"].setDecimals(4)
        self._offset_spins["wf"].setSuffix(" eV")
        self._offset_spins["wf"].setToolTip("Work function of the system.")
        with warnings.catch_warnings(action="ignore", category=UserWarning):
            self._offset_spins["wf"].setValue(self.data.kspace.work_function)
        self._offset_spins["wf"].valueChanged.connect(self.update)

        self.offsets_group.layout().addRow("𝜙", self._offset_spins["wf"])

        self._bound_spins: dict[str, QtWidgets.QDoubleSpinBox] = {}
        self._resolution_spins: dict[str, QtWidgets.QDoubleSpinBox] = {}
        bounds = self.data.kspace.estimate_bounds()
        for k in self.data.kspace.momentum_axes:
            for j in range(2):
                name = f"{k}{j}"
                self._bound_spins[name] = QtWidgets.QDoubleSpinBox()
                if k == "kz":
                    self._bound_spins[name].setRange(0, 100)
                else:
                    self._bound_spins[name].setRange(-10, 10)
                self._bound_spins[name].setSingleStep(0.01)
                self._bound_spins[name].setDecimals(3)
                self._bound_spins[name].setValue(bounds[k][j])
                self._bound_spins[name].valueChanged.connect(self.update)
                self._bound_spins[name].setSuffix(" Å⁻¹")
                self.bounds_group.layout().addRow(name, self._bound_spins[name])

            self._resolution_spins[k] = QtWidgets.QDoubleSpinBox()
            self._resolution_spins[k].setRange(0.001, 10)
            self._resolution_spins[k].setSingleStep(0.001)
            self._resolution_spins[k].setDecimals(5)
            self._resolution_spins[k].setValue(self.data.kspace.estimate_resolution(k))
            self._resolution_spins[k].valueChanged.connect(self.update)
            self._resolution_spins[k].setSuffix(" Å⁻¹")
            self.resolution_group.layout().addRow(k, self._resolution_spins[k])

        # Temporary customization for beta scaling
        # self._beta_scale_spin = QtWidgets.QDoubleSpinBox()
        # self._beta_scale_spin.setValue(1.0)
        # self._beta_scale_spin.setDecimals(2)
        # self._beta_scale_spin.setSingleStep(0.01)
        # self._beta_scale_spin.setRange(0.01, 10)
        # self.offsets_group.layout().addRow("scale", self._beta_scale_spin)
        # self._beta_scale_spin.valueChanged.connect(self.update)

        self.bounds_btn.clicked.connect(self.calculate_bounds)

        self.res_btn.clicked.connect(self.calculate_resolution)
        self.res_npts_check.toggled.connect(self.calculate_resolution)

        for pi in self.plotitems:
            if self.data.kspace._has_beta and not self.data.kspace._has_hv:
                pi.vb.setAspectLocked(lock=True, ratio=1)
        self.open_btn.clicked.connect(self.show_converted)
        self.copy_btn.clicked.connect(self.copy_code)
        self.update()
        if avec is not None:
            self.bz_group.setChecked(True)

    @functools.cached_property
    def _binding_energy(self) -> npt.NDArray:
        return self.data.kspace._binding_energy.values

    @QtCore.Slot()
    def calculate_bounds(self) -> None:
        data = self.data.copy()
        data.kspace.offsets = self.offset_dict
        bounds = data.kspace.estimate_bounds()
        for k in data.kspace.momentum_axes:
            for j in range(2):
                name = f"{k}{j}"
                self._bound_spins[name].blockSignals(True)
                self._bound_spins[name].setValue(bounds[k][j])
                self._bound_spins[name].blockSignals(False)
        self.update()

    @QtCore.Slot()
    def calculate_resolution(self) -> None:
        for k, spin in self._resolution_spins.items():
            spin.setValue(
                self.data.kspace.estimate_resolution(
                    k, from_numpoints=self.res_npts_check.isChecked()
                )
            )

    @property
    def _work_function(self) -> float:
        return float(
            np.round(
                self._offset_spins["wf"].value(), self._offset_spins["wf"].decimals()
            )
        )

    @property
    def _inner_potential(self) -> float:
        return float(
            np.round(
                self._offset_spins["V0"].value(), self._offset_spins["V0"].decimals()
            )
        )

    @QtCore.Slot()
    def show_converted(self) -> None:
        with erlab.interactive.utils.wait_dialog(self, "Converting..."):
            data_kconv = self._assign_params(self.data.copy()).kspace.convert(
                bounds=self.bounds, resolution=self.resolution
            )

        tool = erlab.interactive.itool(data_kconv, execute=False)
        if isinstance(tool, QtWidgets.QWidget):
            if self._itool is not None:
                self._itool.close()

            tool.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
            tool.destroyed.connect(lambda: setattr(self, "_itool", None))
            self._itool = tool
            self._itool.show()

    @QtCore.Slot()
    def copy_code(self) -> str:
        arg_dict: dict[str, typing.Any] = {}
        if self.bounds is not None:
            arg_dict["bounds"] = self.bounds
        if self.resolution is not None:
            arg_dict["resolution"] = self.resolution

        # Detected input name must be single identifier.
        # Otherwise the generated code will not apply offsets correctly.
        input_name: str = str(self._argnames["data"])
        if not input_name.isidentifier():
            input_name = "data"

        out_lines: list[str] = []

        if self.data.kspace._has_hv:
            v0: float = self._inner_potential
            with warnings.catch_warnings(action="ignore", category=UserWarning):
                if not np.isclose(v0, self.data.kspace.inner_potential):
                    out_lines.append(f"{input_name}.kspace.inner_potential = {v0}")

        wf: float = self._work_function
        with warnings.catch_warnings(action="ignore", category=UserWarning):
            if not np.isclose(wf, self.data.kspace.work_function):
                out_lines.append(f"{input_name}.kspace.work_function = {wf}")

        offset_dict_repr = str(self.offset_dict).replace("'", '"')

        out_lines.extend(
            (
                f"{input_name}.kspace.offsets = {offset_dict_repr}",
                erlab.interactive.utils.generate_code(
                    MomentumAccessor.convert,
                    [],
                    arg_dict,
                    module=f"{input_name}.kspace",
                    assign=f"{input_name}_kconv",
                ),
            )
        )

        return erlab.interactive.utils.copy_to_clipboard(out_lines)

    @property
    def bounds(self) -> dict[str, tuple[float, float]] | None:
        if self.bounds_supergroup.isChecked():
            return {
                k: (
                    float(np.round(self._bound_spins[f"{k}0"].value(), 5)),
                    float(np.round(self._bound_spins[f"{k}1"].value(), 5)),
                )
                for k in self.data.kspace.momentum_axes
            }
        return None

    @property
    def resolution(self) -> dict[str, float] | None:
        if self.resolution_supergroup.isChecked():
            return {
                k: float(np.round(self._resolution_spins[k].value(), 5))
                for k in self.data.kspace.momentum_axes
            }
        return None

    @property
    def offset_dict(self) -> dict[str, float]:
        return {
            k: float(np.round(self._offset_spins[k].value(), 5))
            for k in self.data.kspace._valid_offset_keys
        }

    def _angle_data(self) -> xr.DataArray:
        if self.data.kspace._has_eV:
            data_binding = self.data.copy().assign_coords(eV=self._binding_energy)

            center, width = self.center_spin.value(), self.width_spin.value()
            if width == 1:
                return data_binding.isel(
                    eV=np.argmin(np.abs(self.data.eV.values - center))
                )

            arr = self.data.eV.values
            idx = np.searchsorted((arr[:-1] + arr[1:]) / 2, center)
            return (
                data_binding.isel(
                    eV=slice(idx - width // 2, idx + (width - 1) // 2 + 1)
                )
                .mean("eV", skipna=True, keep_attrs=True)
                .assign_coords(eV=center)
            )
        return self.data.copy()

    def _assign_params(self, data: xr.DataArray) -> xr.DataArray:
        data.kspace.offsets = self.offset_dict

        if self.data.kspace._has_hv:
            v0: float = self._inner_potential
            with warnings.catch_warnings(action="ignore", category=UserWarning):
                if not np.isclose(v0, self.data.kspace.inner_potential):
                    data.kspace.inner_potential = v0

        wf: float = self._work_function
        with warnings.catch_warnings(action="ignore", category=UserWarning):
            if not np.isclose(wf, self.data.kspace.work_function):
                data.kspace.work_function = wf
        return data

    def get_data(self) -> tuple[xr.DataArray, xr.DataArray]:
        # Set angle offsets
        data_ang = self._assign_params(self._angle_data())
        # if "beta" in data_ang.dims:
        #     data_ang = data_ang.assign_coords(
        #         beta=data_ang.beta * self._beta_scale_spin.value()
        #     )

        # Convert to kspace
        data_k = data_ang.kspace.convert(
            bounds=self.bounds, resolution=self.resolution, silent=True
        )
        return data_ang, data_k

    @QtCore.Slot()
    def update(self) -> None:
        ang, k = self.get_data()
        self.images[0].setDataArray(ang.T)
        self.images[1].setDataArray(k.T)

    def get_bz_lines(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        from erlab.plotting.bz import get_bz_edge

        if self.data.kspace._has_hv:
            # Out-of-plane BZ
            a, c = self.ab_spin.value(), self.c_spin.value()
            rot = np.deg2rad(self.rot_spin.value())

            basis = np.array([[a, 0], [0, c]])
            if not self.reciprocal_check.isChecked():
                basis = 2 * np.pi * np.linalg.inv(basis).T

            if rot != 0.0:
                basis[0, :] *= np.cos(rot)

            lines, vertices = get_bz_edge(
                basis,
                reciprocal=True,
                extend=(self.n1_spin.value(), self.n2_spin.value()),
            )

        else:
            # In-plane BZ
            a, b = self.a_spin.value(), self.b_spin.value()
            ang = np.deg2rad(self.ang_spin.value())
            rot = np.deg2rad(self.rot_spin.value())

            avec = np.array([[a, 0], [b * np.cos(ang), b * np.sin(ang)]])

            lines, vertices = get_bz_edge(
                avec,
                reciprocal=self.reciprocal_check.isChecked(),
                extend=(self.n1_spin.value(), self.n2_spin.value()),
            )

            if rot != 0.0:
                rotmat = np.array(
                    [[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]]
                )
                lines = (rotmat @ lines.transpose(1, 2, 0)).transpose(2, 0, 1)
                vertices = (rotmat @ vertices.T).T

        return lines, vertices


def ktool(
    data: xr.DataArray,
    avec: npt.NDArray | None = None,
    rotate_bz: float = 0.0,
    cmap: str | None = None,
    gamma: float = 0.5,
    *,
    data_name: str | None = None,
    execute: bool | None = None,
) -> KspaceTool:
    """Interactive momentum conversion tool.

    This tool can also be accessed with :meth:`DataArray.kspace.interactive`, or from
    the ``View`` menu of an ImageTool window.

    Parameters
    ----------
    data
        Data to convert. Currently supports constant energy slices (2D data with alpha
        and beta dimensions) and all 3D data that has eV and alpha dimensions, including
        maps and photon energy dependent data.
    avec : array-like, optional
        Real-space lattice vectors as a 2x2 or 3x3 numpy array. If provided, the
        Brillouin zone boundary overlay will be calculated based on these vectors. If
        given as a 2x2 array, the third row and column will be assumed to be all 0. You
        can use utilities from :mod:`erlab.lattice` to construct these vectors.
    rotate_bz
        Rotation angle for the Brillouin zone boundary overlay.
    cmap : str, optional
        Name of the colormap to use.
    gamma
        Initial gamma value for the colormap.
    data_name
        Name of the data variable in the generated code. If not provided, the name is
        automatically determined.

    """
    if data_name is None:
        try:
            data_name = str(varname.argname("data", func=ktool, vars_only=False))
        except varname.VarnameRetrievingError:
            data_name = "data"

    with erlab.interactive.utils.setup_qapp(execute):
        win = KspaceTool(
            data,
            avec=avec,
            rotate_bz=rotate_bz,
            cmap=cmap,
            gamma=gamma,
            data_name=data_name,
        )
        win.show()
        win.raise_()
        win.activateWindow()

    return win
