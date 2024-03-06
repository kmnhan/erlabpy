"""Interactive momentum conversion tool."""

import os
import sys

import numpy as np
import pyqtgraph as pg
import varname
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets, uic

import erlab.accessors
import erlab.analysis
from erlab.interactive.colors import (
    BetterColorBarItem,
    BetterImageItem,
    ColorMapComboBox,
    ColorMapGammaWidget,
)
from erlab.interactive.imagetool import ImageTool
from erlab.interactive.utilities import array_rect, copy_to_clipboard, gen_function_code
from erlab.plotting.bz import get_bz_edge

__all__ = ["ktool"]


class KtoolImageItem(BetterImageItem):

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


class ktoolGUI(*uic.loadUiType(os.path.join(os.path.dirname(__file__), "ktool.ui"))):
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
        self.images: tuple[KtoolImageItem] = (
            KtoolImageItem(axisOrder="row-major"),
            KtoolImageItem(axisOrder="row-major"),
        )

        for i, plot in enumerate(self.plotitems):
            self.graphics_layout.addItem(plot, i, 0)
            plot.addItem(self.images[i])
            plot.vb.setAspectLocked(lock=True, ratio=1)
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

    def update_bz(self):
        self.plotitems[1].clearPlots()
        if not self.bz_group.isChecked():
            return
        if self.reciprocal_check.isChecked():
            self.a_spin.setSuffix(" Å⁻¹")
            self.b_spin.setSuffix(" Å⁻¹")
        else:
            self.a_spin.setSuffix(" Å")
            self.b_spin.setSuffix(" Å")

        a, b = self.a_spin.value(), self.b_spin.value()
        ang = np.deg2rad(self.ang_spin.value())
        rot = np.deg2rad(self.rot_spin.value())

        avec = np.array([[a, 0], [b * np.cos(ang), b * np.sin(ang)]])

        lines, vertices = get_bz_edge(
            avec, reciprocal=self.reciprocal_check.isChecked()
        )

        rotmat = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])

        lines = [(rotmat @ line.T).T for line in lines]
        vertices = (rotmat @ vertices.T).T

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


class ktool(ktoolGUI):

    def __init__(self, data: xr.DataArray, **kwargs):
        super().__init__()

        self._argnames = dict()
        try:
            self._argnames["data"] = varname.argname(
                "data", func=self.__init__, vars_only=False
            )
        except varname.VarnameRetrievingError:
            self._argnames["data"] = "data"

        self.data: xr.DataArray = data.copy(deep=True)

        if self.data.kspace.has_eV:
            self.center_spin.setRange(self.data.eV[0], self.data.eV[-1])
            self.width_spin.setRange(0, self.data.eV[-1] - self.data.eV[0])
            self.center_spin.valueChanged.connect(self.update)
            self.width_spin.valueChanged.connect(self.update)
        else:
            self.energy_group.setDisabled(True)

        self.bounds_group.toggled.connect(self.update)
        self.resolution_group.toggled.connect(self.update)

        self._offset_spins: dict[str, QtWidgets.QDoubleSpinBox] = {}
        for k in self.data.kspace._valid_offset_keys:
            self._offset_spins[k] = QtWidgets.QDoubleSpinBox()
            self._offset_spins[k].setRange(-180, 180)
            self._offset_spins[k].setSingleStep(0.01)
            self._offset_spins[k].setDecimals(3)
            self._offset_spins[k].setValue(self.data.kspace.get_offset(k))
            self._offset_spins[k].valueChanged.connect(self.update)
            self._offset_spins[k].setSuffix("°")
            self.offsets_group.layout().addRow(k, self._offset_spins[k])

        self._bound_spins: dict[str, QtWidgets.QDoubleSpinBox] = {}
        self._resolution_spins: dict[str, QtWidgets.QDoubleSpinBox] = {}
        bounds = self.data.kspace.get_bounds()
        for i, k in enumerate(self.data.kspace.momentum_axes):
            for j in range(2):
                name = f"{k}{j}"
                self._bound_spins[name] = QtWidgets.QDoubleSpinBox()
                self._bound_spins[name].setRange(-10, 10)
                self._bound_spins[name].setSingleStep(0.01)
                self._bound_spins[name].setDecimals(2)
                self._bound_spins[name].setValue(bounds[i][j])
                self._bound_spins[name].valueChanged.connect(self.update)
                self._bound_spins[name].setSuffix(" Å⁻¹")
                self.bounds_group.layout().addRow(name, self._bound_spins[name])

            self._resolution_spins[k] = QtWidgets.QDoubleSpinBox()
            self._resolution_spins[k].setRange(0.001, 10)
            self._resolution_spins[k].setSingleStep(0.001)
            self._resolution_spins[k].setDecimals(4)
            self._resolution_spins[k].setValue(self.data.kspace.minimum_k_resolution)
            self._resolution_spins[k].valueChanged.connect(self.update)
            self._resolution_spins[k].setSuffix(" Å⁻¹")
            self.resolution_group.layout().addRow(k, self._resolution_spins[k])

        self.open_btn.clicked.connect(self.show_converted)
        self.copy_btn.clicked.connect(self.copy_code)
        self.update()

        self.__post_init__()

    def show_converted(self):
        self.data.kspace.offsets = self.offset_dict
        itool = ImageTool(
            self.data.kspace.convert(bounds=self.bounds, resolution=self.resolution)
        )
        itool.show()

    def copy_code(self):
        arg_dict = dict()
        if self.bounds is not None:
            arg_dict["bounds"] = self.bounds
        if self.resolution is not None:
            arg_dict["resolution"] = self.resolution

        # Detected input name must be single identifier.
        # Otherwise the generated code will not apply offsets correctly.
        input_name: str = str(self._argnames["data"])
        if not input_name.isidentifier():
            input_name = "data"

        attr_code = f"{input_name}.kspace.offsets = {self.offset_dict}"
        conv_code = gen_function_code(
            copy=False,
            **{f"{input_name}_kconv = {input_name}.kspace.convert": [arg_dict]},
        )

        copy_to_clipboard([attr_code, conv_code])

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
        if self.resolution_group.isChecked():
            return {
                k: self._resolution_spins[k].value()
                for k in self.data.kspace.momentum_axes
            }
        else:
            return None

    @property
    def offset_dict(self) -> dict[str, float]:
        return {
            k: self._offset_spins[k].value()
            for k in self.data.kspace._valid_offset_keys
        }

    def _angle_data(self) -> xr.DataArray:
        if self.data.kspace.has_eV:
            center, width = self.center_spin.value(), self.width_spin.value()
            if width == 0:
                return self.data.sel(eV=center, method="nearest")
            else:
                return (
                    self.data.sel(eV=slice(center - width / 2, center + width / 2))
                    .mean("eV", keep_attrs=True)
                    .assign_coords(eV=center)
                )
        else:
            return self.data

    def get_data(self) -> tuple[xr.DataArray, xr.DataArray]:
        # Set angle offsets
        data_ang = self._angle_data()
        data_ang.kspace.offsets = self.offset_dict
        # Convert to kspace
        data_k = data_ang.kspace.convert(bounds=self.bounds, resolution=self.resolution)
        return data_ang, data_k

    def update(self):
        ang, k = self.get_data()
        self.images[0].setDataArray(ang)
        self.images[1].setDataArray(k)

    def closeEvent(self, event: QtGui.QCloseEvent):
        del self.data
        super().closeEvent(event)


if __name__ == "__main__":
    dat = erlab.io.load_hdf5("/Users/khan/2210_ALS_f0008.h5")

    win = ktool(dat)
