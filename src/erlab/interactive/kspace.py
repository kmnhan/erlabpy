"""Interactive momentum conversion tool."""

__all__ = ["ktool"]

import os
import sys

import numpy as np
import numpy.typing as npt
import pyqtgraph as pg
import varname
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets, uic

import erlab.analysis
from erlab.interactive.colors import BetterColorBarItem  # noqa: F401
from erlab.interactive.colors import BetterImageItem
from erlab.interactive.colors import ColorMapComboBox  # noqa: F401
from erlab.interactive.colors import ColorMapGammaWidget  # noqa: F401
from erlab.interactive.imagetool import ImageTool
from erlab.interactive.utilities import array_rect, copy_to_clipboard, gen_function_code
from erlab.plotting.bz import get_bz_edge


class KspaceToolImageItem(BetterImageItem):
    def setDataArray(self, data=None, **kargs):
        rect = array_rect(data)
        if self.axisOrder == "row-major":
            img = np.ascontiguousarray(data.values)
        else:
            img = np.asfortranarray(data.values.T)
        pi = self.getPlotItem()
        if pi is not None:
            pi.setLabel("left", data.dims[0])
            pi.setLabel("bottom", data.dims[1])
        self.setImage(img, rect=rect, **kargs)

    def getPlotItem(self) -> pg.PlotItem | None:
        p = self
        while True:
            try:
                p = p.parentItem()
            except RuntimeError:
                return None
            if p is None:
                return None
            if isinstance(p, pg.PlotItem):
                return p


class KspaceToolGUI(
    *uic.loadUiType(os.path.join(os.path.dirname(__file__), "ktool.ui"))
):
    def __init__(self):
        # Start the QApplication if it doesn't exist
        self.qapp = QtCore.QCoreApplication.instance()
        if not self.qapp:
            self.qapp = QtWidgets.QApplication(sys.argv)
        self.qapp.setStyle("Fusion")

        # Initialize UI
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Momentum Conversion")

        self.plotitems: tuple[pg.PlotItem, pg.PlotItem] = (pg.PlotItem(), pg.PlotItem())
        self.images: tuple[KspaceToolImageItem] = (
            KspaceToolImageItem(axisOrder="row-major"),
            KspaceToolImageItem(axisOrder="row-major"),
        )

        for i, plot in enumerate(self.plotitems):
            self.graphics_layout.addItem(plot, i, 0)
            plot.addItem(self.images[i])
            plot.showGrid(x=True, y=True, alpha=0.5)

        # Set up colormap controls
        self.cmap_combo.setDefaultCmap("terrain")
        self.cmap_combo.textActivated.connect(self.update_cmap)
        self.gamma_widget.setValue(0.5)
        self.gamma_widget.valueChanged.connect(self.update_cmap)
        self.invert_check.stateChanged.connect(self.update_cmap)
        self.invert_check.setChecked(True)
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

    def update_cmap(self):
        name = self.cmap_combo.currentText()
        if name == self.cmap_combo.LOAD_ALL_TEXT:
            self.cmap_combo.load_all()
            return

        for im in self.images:
            im.set_colormap(
                name,
                self.gamma_widget.value(),
                reverse=self.invert_check.isChecked(),
                highContrast=self.contrast_check.isChecked(),
                update=True,
            )

    def get_bz_lines(self):
        raise NotImplementedError

    def update_bz(self):
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

    def __post_init__(self, execute=None):
        self.show()
        self.raise_()
        self.activateWindow()

        if execute is None:
            execute = True
            try:
                shell = get_ipython().__class__.__name__  # type: ignore
                if shell in ["ZMQInteractiveShell", "TerminalInteractiveShell"]:
                    execute = False
                    from IPython.lib.guisupport import start_event_loop_qt4

                    start_event_loop_qt4(self.qapp)
            except NameError:
                pass
        if execute:
            self.qapp.exec()


class KspaceTool(KspaceToolGUI):
    def __init__(self, data: xr.DataArray, *, data_name: str | None = None):
        super().__init__()

        self._argnames = {}

        if data_name is None:
            try:
                self._argnames["data"] = varname.argname(
                    "data", func=self.__init__, vars_only=False
                )
            except varname.VarnameRetrievingError:
                self._argnames["data"] = "data"
        else:
            self._argnames["data"] = data_name

        self.data: xr.DataArray = data.copy(deep=True)

        if self.data.kspace.has_eV:
            self.center_spin.setRange(self.data.eV[0], self.data.eV[-1])
            self.width_spin.setRange(1, len(self.data.eV))
            self.center_spin.valueChanged.connect(self.update)
            self.width_spin.valueChanged.connect(self.update)
        else:
            self.energy_group.setDisabled(True)

        self.bounds_group.toggled.connect(self.update)
        self.resolution_supergroup.toggled.connect(self.update)

        self._offset_spins: dict[str, QtWidgets.QDoubleSpinBox] = {}
        offset_labels = {"delta": "𝛿", "chi": "𝜒₀", "xi": "𝜉₀", "beta": "𝛽₀"}
        for k in self.data.kspace.valid_offset_keys:
            self._offset_spins[k] = QtWidgets.QDoubleSpinBox()
            self._offset_spins[k].setRange(-180, 180)
            self._offset_spins[k].setSingleStep(0.01)
            self._offset_spins[k].setDecimals(3)
            self._offset_spins[k].setValue(self.data.kspace.offsets[k])
            self._offset_spins[k].valueChanged.connect(self.update)
            self._offset_spins[k].setSuffix("°")
            self.offsets_group.layout().addRow(offset_labels[k], self._offset_spins[k])

        if self.data.kspace.has_hv:
            self._offset_spins["V0"] = QtWidgets.QDoubleSpinBox()
            self._offset_spins["V0"].setRange(0, 100)
            self._offset_spins["V0"].setSingleStep(1)
            self._offset_spins["V0"].setDecimals(1)
            self._offset_spins["V0"].setValue(self.data.kspace.inner_potential)
            self._offset_spins["V0"].valueChanged.connect(self.update)
            self._offset_spins["V0"].setSuffix(" eV")
            self.offsets_group.layout().addRow("V₀", self._offset_spins["V0"])

            for i in range(8):
                self.bz_form.setRowVisible(i, i not in (0, 1, 2))
        else:
            for i in range(8):
                self.bz_form.setRowVisible(i, i not in (3, 4))

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
                self._bound_spins[name].setDecimals(2)
                self._bound_spins[name].setValue(bounds[k][j])
                self._bound_spins[name].valueChanged.connect(self.update)
                self._bound_spins[name].setSuffix(" Å⁻¹")
                self.bounds_group.layout().addRow(name, self._bound_spins[name])

            self._resolution_spins[k] = QtWidgets.QDoubleSpinBox()
            self._resolution_spins[k].setRange(0.001, 10)
            self._resolution_spins[k].setSingleStep(0.001)
            self._resolution_spins[k].setDecimals(4)
            self._resolution_spins[k].setValue(self.data.kspace.estimate_resolution(k))
            self._resolution_spins[k].valueChanged.connect(self.update)
            self._resolution_spins[k].setSuffix(" Å⁻¹")
            self.resolution_group.layout().addRow(k, self._resolution_spins[k])

        self.res_btn.clicked.connect(self.calculate_resolution)
        self.res_npts_check.toggled.connect(self.calculate_resolution)

        for pi in self.plotitems:
            if self.data.kspace.has_beta and not self.data.kspace.has_hv:
                pi.vb.setAspectLocked(lock=True, ratio=1)
        self.open_btn.clicked.connect(self.show_converted)
        self.copy_btn.clicked.connect(self.copy_code)
        self.update()

        self.__post_init__()

    def calculate_resolution(self):
        for k, spin in self._resolution_spins.items():
            spin.setValue(
                self.data.kspace.estimate_resolution(
                    k, from_numpoints=self.res_npts_check.isChecked()
                )
            )

    def show_converted(self):
        self.data.kspace.offsets = self.offset_dict

        if self.data.kspace.has_hv:
            self.data.kspace.inner_potential = self._offset_spins["V0"].value()

        wait_dialog = QtWidgets.QDialog(self)
        wait_dialog.setLayout(QtWidgets.QVBoxLayout())
        wait_dialog.layout().addWidget(QtWidgets.QLabel("Converting..."))
        wait_dialog.open()
        itool = ImageTool(
            self.data.kspace.convert(bounds=self.bounds, resolution=self.resolution)
        )
        wait_dialog.close()
        itool.show()

    def copy_code(self):
        arg_dict = {}
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

        if self.data.kspace.has_hv:
            out_lines.append(
                f"{input_name}.kspace.inner_potential"
                f" = {self._offset_spins['V0'].value()}"
            )
        out_lines.append(f"{input_name}.kspace.offsets = {self.offset_dict}")
        out_lines.append(
            gen_function_code(
                copy=False,
                **{f"{input_name}_kconv = {input_name}.kspace.convert": [arg_dict]},
            )
        )

        copy_to_clipboard(out_lines)

    @property
    def bounds(self) -> dict[str, tuple[float, float]] | None:
        if self.bounds_group.isChecked():
            return {
                k: tuple(self._bound_spins[f"{k}{j}"].value() for j in range(2))
                for k in self.data.kspace.momentum_axes
            }
        else:
            return None

    @property
    def resolution(self) -> dict[str, float] | None:
        if self.resolution_supergroup.isChecked():
            return {
                k: self._resolution_spins[k].value()
                for k in self.data.kspace.momentum_axes
            }
        else:
            return None

    @property
    def offset_dict(self) -> dict[str, float]:
        return {
            k: np.round(self._offset_spins[k].value(), 5)
            for k in self.data.kspace.valid_offset_keys
        }

    def _angle_data(self) -> xr.DataArray:
        if self.data.kspace.has_eV:
            center, width = self.center_spin.value(), self.width_spin.value()
            if width == 0:
                return self.data.sel(eV=center, method="nearest")
            else:
                arr = self.data.eV.values
                idx = np.searchsorted((arr[:-1] + arr[1:]) / 2, center)
                return (
                    self.data.isel(
                        eV=slice(idx - width // 2, idx + (width - 1) // 2 + 1)
                    )
                    .mean("eV", skipna=True, keep_attrs=True)
                    .assign_coords(eV=center)
                )
        else:
            return self.data

    def get_data(self) -> tuple[xr.DataArray, xr.DataArray]:
        # Set angle offsets
        data_ang = self._angle_data()
        data_ang.kspace.offsets = self.offset_dict

        if self.data.kspace.has_hv:
            data_ang.kspace.inner_potential = self._offset_spins["V0"].value()

        # Convert to kspace
        data_k = data_ang.kspace.convert(
            bounds=self.bounds, resolution=self.resolution, silent=True
        )
        return data_ang, data_k

    def update(self):
        ang, k = self.get_data()
        self.images[0].setDataArray(ang.T)
        self.images[1].setDataArray(k.T)

    def get_bz_lines(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        if self.data.kspace.has_hv:
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

    def closeEvent(self, event: QtGui.QCloseEvent):
        del self.data
        super().closeEvent(event)


def ktool(data: xr.DataArray, *, data_name: str | None = None) -> KspaceTool:
    """Interactive momentum conversion tool."""
    if data_name is None:
        try:
            data_name = varname.argname("data", func=ktool, vars_only=False)
        except varname.VarnameRetrievingError:
            data_name = "data"
    return KspaceTool(data, data_name=data_name)


if __name__ == "__main__":
    dat = erlab.io.load_hdf5("/Users/khan/2210_ALS_f0008.h5")
    win = ktool(dat)
