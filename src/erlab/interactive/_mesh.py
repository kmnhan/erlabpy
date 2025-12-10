__all__ = ["meshtool"]

import importlib.resources
import os
import typing

import numpy as np
import numpy.typing as npt
import pydantic
import pyqtgraph as pg
import scipy.fft
import varname
import xarray as xr
from qtpy import QtCore, QtWidgets, uic

import erlab


class MeshTool(erlab.interactive.utils.ToolWindow):
    tool_name = "meshtool"

    @property
    def preview_imageitem(self) -> pg.ImageItem:
        return self.main_image

    class StateModel(pydantic.BaseModel):
        first_order_peaks: list[list[int]]
        order: int
        n_pad: int
        roi_hw: int
        k: float
        feather: float
        undo_edge_correction: bool
        method: str

    @property
    def tool_status(self) -> StateModel:
        return self.StateModel(
            first_order_peaks=[
                [self._data.alpha.size // 2, self._data.eV.size // 2],
                [self.p0_spin0.value(), self.p0_spin1.value()],
                [self.p1_spin0.value(), self.p1_spin1.value()],
            ],
            order=self.order_spin.value(),
            n_pad=self.n_pad_spin.value(),
            roi_hw=self.roi_hw_spin.value(),
            k=self.k_spin.value(),
            feather=self.feather_spin.value(),
            undo_edge_correction=self.undo_edge_correction_check.isChecked(),
            method=self.method_combo.currentText(),
        )

    @tool_status.setter
    def tool_status(self, status: StateModel) -> None:
        self.p0_spin0.setValue(int(status.first_order_peaks[1][0]))
        self.p0_spin1.setValue(int(status.first_order_peaks[1][1]))
        self.p1_spin0.setValue(int(status.first_order_peaks[2][0]))
        self.p1_spin1.setValue(int(status.first_order_peaks[2][1]))

        self.order_spin.setValue(status.order)
        self.n_pad_spin.setValue(status.n_pad)
        self.roi_hw_spin.setValue(status.roi_hw)
        self.k_spin.setValue(status.k)
        self.feather_spin.setValue(status.feather)
        self.undo_edge_correction_check.setChecked(status.undo_edge_correction)
        self.method_combo.setCurrentText(status.method)

    @property
    def tool_data(self) -> xr.DataArray:
        return self._data

    def __init__(self, data: xr.DataArray, *, data_name: str | None = None) -> None:
        if data_name is None:
            try:
                data_name = typing.cast(
                    "str",
                    varname.argname("data", func=self.__init__, vars_only=False),  # type: ignore[misc]
                )
            except varname.VarnameRetrievingError:
                data_name = "data"

        self.data_name = data_name

        # Initialize UI
        super().__init__()
        uic.loadUi(
            str(importlib.resources.files(erlab.interactive).joinpath("meshtool.ui")),
            self,
        )

        self._data = data
        self._corrected: xr.DataArray | None = None
        self._mesh: xr.DataArray | None = None
        self._itool_corr: QtWidgets.QWidget | None = None
        self._itool_mesh: QtWidgets.QWidget | None = None

        graphics_layout = typing.cast("pg.GraphicsLayoutWidget", self.graphics_layout)
        graphics_layout.setContentsMargins(0, 0, 0, 0)
        graphics_layout.ci.setSpacing(0)

        col0 = graphics_layout.addLayout(0, 0, 6, 1)
        col0.layout.setVerticalSpacing(0)

        col1 = graphics_layout.addLayout(0, 1, 6, 1)
        col1.layout.setVerticalSpacing(0)

        self.image_plots: list[pg.PlotItem] = [
            col0.addPlot(0, 0),
            col0.addPlot(1, 0),
            col0.addPlot(2, 0),
        ]
        self.add_label(self.image_plots[0], "Original Data")
        self.add_label(self.image_plots[1], "Mesh Removed Data")
        self.add_label(self.image_plots[2], "Extracted Mesh")

        self.fft_plots: list[pg.PlotItem] = [
            col1.addPlot(0, 0),
            col1.addPlot(1, 0),
            col1.addPlot(2, 0),
        ]
        self.add_label(self.fft_plots[0], "FFT of Original Data")
        self.add_label(self.fft_plots[1], "FFT of Mesh Removed Data")
        self.add_label(self.fft_plots[2], "FFT Mask")

        self.main_image = erlab.interactive.utils.xImageItem(axisOrder="row-major")
        self.corr_image = erlab.interactive.utils.xImageItem(axisOrder="row-major")
        self.mesh_image = erlab.interactive.utils.xImageItem(axisOrder="row-major")
        self.image_plots[0].addItem(self.main_image)
        self.image_plots[1].addItem(self.corr_image)
        self.image_plots[2].addItem(self.mesh_image)

        self.main_fft_image = erlab.interactive.utils.xImageItem(axisOrder="row-major")
        self.corr_fft_image = erlab.interactive.utils.xImageItem(axisOrder="row-major")
        self.mask_fft_image = erlab.interactive.utils.xImageItem(axisOrder="row-major")
        self.fft_plots[0].addItem(self.main_fft_image)
        self.fft_plots[1].addItem(self.corr_fft_image)
        self.fft_plots[2].addItem(self.mask_fft_image)

        self.p0_target = pg.TargetItem(pen=(255, 255, 0, 200), brush=(0, 0, 0, 0))
        self.p1_target = pg.TargetItem(pen=(255, 255, 0, 200), brush=(0, 0, 0, 0))
        self.fft_plots[0].addItem(self.p0_target)
        self.fft_plots[0].addItem(self.p1_target)

        self.higher_order_targets: list[pg.TargetItem] = []

        self.set_data_beforecalc(initial=True)

        opts = erlab.interactive.options.model
        for img in (self.main_image, self.corr_image):
            img.set_colormap(
                opts.colors.cmap.name, gamma=0.5, reverse=opts.colors.cmap.reverse
            )
        self.mesh_image.set_colormap("bwr", gamma=1.0, reverse=False)

        for img in (self.main_fft_image, self.corr_fft_image):
            img.set_colormap(opts.colors.cmap.name, gamma=0.5, reverse=False)

        self.mask_fft_image.set_colormap("gray", gamma=1.0, reverse=True)

        self.cbar_data = erlab.interactive.colors.BetterColorBarItem(
            image=(self.main_image, self.corr_image)
        )
        col0.addItem(self.cbar_data, 0, 1, 2, 1)

        self.cbar_mesh = erlab.interactive.colors.BetterColorBarItem(
            image=self.mesh_image
        )
        self.cbar_mesh.setSpanRegion((0.5, 1.5))
        col0.addItem(self.cbar_mesh, 2, 1)

        self.cbar_fft = erlab.interactive.colors.BetterColorBarItem(
            image=(self.main_fft_image, self.corr_fft_image)
        )
        col1.addItem(self.cbar_fft, 0, 1, 2, 1)

        self.cbar_mask = erlab.interactive.colors.BetterColorBarItem(
            image=self.mask_fft_image
        )
        col1.addItem(self.cbar_mask, 2, 1)

        for cbar in (self.cbar_data, self.cbar_mesh, self.cbar_fft, self.cbar_mask):
            cbar.set_dimensions(horiz_pad=30, vert_pad=30)
            cbar.setPreferredWidth(60)

        for i in range(1, len(self.image_plots)):
            self.image_plots[i].setXLink(self.image_plots[0])
            self.image_plots[i].setYLink(self.image_plots[0])

        for i in range(1, len(self.fft_plots)):
            self.fft_plots[i].setXLink(self.fft_plots[0])
            self.fft_plots[i].setYLink(self.fft_plots[0])

        self.p0_spin0.setMaximum(self._data.alpha.size - 1)
        self.p0_spin1.setMaximum(self._data.eV.size - 1)
        self.p1_spin0.setMaximum(self._data.alpha.size - 1)
        self.p1_spin1.setMaximum(self._data.eV.size - 1)

        # Set plot visibility
        for plot in self.image_plots:
            plot.vb.setDefaultPadding(0)
            if plot != self.image_plots[-1]:
                plot.hideAxis("bottom")
            plot.vb.autoRange()

        for plot in self.fft_plots:
            plot.vb.setDefaultPadding(0)
            if plot != self.fft_plots[-1]:
                plot.hideAxis("bottom")
            plot.vb.autoRange()
        self._connect_signals()

    def _connect_signals(self) -> None:
        self.auto_btn.clicked.connect(self.auto_find_peaks)
        self.go_btn.clicked.connect(self.update)

        self.order_spin.valueChanged.connect(self._update_higher_order_targets)

        self.p0_spin0.valueChanged.connect(self._update_target_pos)
        self.p0_spin1.valueChanged.connect(self._update_target_pos)
        self.p1_spin0.valueChanged.connect(self._update_target_pos)
        self.p1_spin1.valueChanged.connect(self._update_target_pos)

        self.p0_target.sigPositionChanged.connect(self._target_moved)
        self.p1_target.sigPositionChanged.connect(self._target_moved)

        self.undo_edge_correction_check.toggled.connect(self.set_data_beforecalc)

        self.itool_corr_btn.clicked.connect(self._corr_itool)
        self.itool_mesh_btn.clicked.connect(self._mesh_itool)

        self.save_mesh_btn.clicked.connect(self.save_mesh)
        self.copy_btn.clicked.connect(self.copy_code)

    @staticmethod
    def add_label(plot: pg.PlotItem, text: str) -> None:
        label = pg.LabelItem(text, size="11pt")
        label.setParentItem(plot.getViewBox())
        label.anchor((0, 0), (0, 0), offset=(5, 0))

    def get_reduced(self) -> tuple[xr.DataArray, npt.NDArray]:
        if not hasattr(self, "_data_averaged"):
            self._data_averaged = self.reduce_to_cut(self._data).compute()
        reduced = self._data_averaged
        if self.undo_edge_correction_check.isChecked():
            _, reduced = erlab.analysis.mesh.auto_correct_curvature(reduced)

        image = reduced.fillna(0).values
        image = erlab.analysis.mesh.pad_and_taper(image, self.tool_status.n_pad)

        log_magnitude = np.log(
            np.abs(scipy.fft.fftshift(scipy.fft.fft2(image))).clip(min=1e-15)
        )
        return reduced, log_magnitude

    @QtCore.Slot()
    def set_data_beforecalc(self, initial: bool = False) -> None:
        reduced, log_magnitude = self.get_reduced()
        n_pad: int = self.tool_status.n_pad

        log_magnitude_unpadded = erlab.analysis.mesh.unpad(log_magnitude, n_pad)

        self.main_image.setDataArray(reduced.T, update_labels=initial)
        self.main_fft_image.setImage(log_magnitude_unpadded)
        if initial:
            self.corr_image.setDataArray(reduced.T)

            placeholder = xr.ones_like(reduced.T)
            placeholder = placeholder + np.cos(np.deg2rad(placeholder.alpha * 24))
            self.mesh_image.setDataArray(placeholder)
            self.corr_fft_image.setImage(log_magnitude_unpadded)
            self.mask_fft_image.setImage(placeholder.values.T)

    @QtCore.Slot()
    def _update_target_pos(self) -> None:
        with (
            QtCore.QSignalBlocker(self.p0_target),
            QtCore.QSignalBlocker(self.p1_target),
        ):
            self.p0_target.setPos(self.p0_spin1.value(), self.p0_spin0.value())
            self.p1_target.setPos(self.p1_spin1.value(), self.p1_spin0.value())
        self._update_higher_order_targets()

    @QtCore.Slot()
    def _target_moved(self) -> None:
        with (
            QtCore.QSignalBlocker(self.p0_spin0),
            QtCore.QSignalBlocker(self.p0_spin1),
            QtCore.QSignalBlocker(self.p1_spin0),
            QtCore.QSignalBlocker(self.p1_spin1),
        ):
            p0_pos = self.p0_target.pos()
            p1_pos = self.p1_target.pos()
            self.p0_spin0.setValue(int(p0_pos.y()))
            self.p0_spin1.setValue(int(p0_pos.x()))
            self.p1_spin0.setValue(int(p1_pos.y()))
            self.p1_spin1.setValue(int(p1_pos.x()))
        self._update_higher_order_targets()

    @QtCore.Slot()
    def _update_higher_order_targets(self) -> None:
        status = self.tool_status
        order: int = status.order
        first_order: list[list[int]] = status.first_order_peaks

        higher_order = erlab.analysis.mesh.higher_order_peaks(
            first_order,
            order=order,
            shape=self.main_fft_image.image.shape,
            include_center=False,
            only_upper=False,
        )[2:]

        while len(self.higher_order_targets) < len(higher_order):
            target = pg.TargetItem(
                movable=False, symbol="o", pen="cyan", brush=(0, 0, 0, 0)
            )
            self.fft_plots[0].addItem(target)
            self.higher_order_targets.append(target)

        while len(self.higher_order_targets) > len(higher_order):
            target = self.higher_order_targets.pop()
            self.fft_plots[0].removeItem(target)
            target.deleteLater()

        for i, target in enumerate(self.higher_order_targets):
            pos = higher_order[i]
            target.setPos(pos[1], pos[0])

    def get_params_dict(self) -> dict[str, typing.Any]:
        return self.tool_status.model_dump()

    @staticmethod
    def reduce_to_cut(darr: xr.DataArray) -> xr.DataArray:
        core_dims = ("alpha", "eV")
        other_dims = tuple(dim for dim in darr.dims if dim not in core_dims)
        return darr.mean(other_dims).transpose(*core_dims)

    @QtCore.Slot()
    def auto_find_peaks(self) -> None:
        peaks = (
            erlab.analysis.mesh.find_peaks(
                self.get_reduced()[1],
                bins=self.bins_spin.value(),
                n_peaks=2,
                plot=False,
            )
            - self.tool_status.n_pad
        )
        self.p0_spin0.setValue(int(peaks[1, 0]))
        self.p0_spin1.setValue(int(peaks[1, 1]))
        self.p1_spin0.setValue(int(peaks[2, 0]))
        self.p1_spin1.setValue(int(peaks[2, 1]))

    @QtCore.Slot()
    def update(self) -> None:
        with erlab.interactive.utils.wait_dialog(self, "Removing mesh..."):
            (
                self._corrected,
                self._mesh,
                shift,
                log_magnitude,
                log_magnitude_corr,
                _,
                mask,
            ) = erlab.analysis.mesh.remove_mesh(
                self._data, **self.get_params_dict(), full_output=True
            )  # type: ignore[misc]
            self.main_fft_image.setImage(log_magnitude)
            self.mask_fft_image.setImage(mask)

            # Plot corrected and mesh
            corrected = self.reduce_to_cut(self._corrected).compute()
            mesh = self._mesh.copy()
            if self.undo_edge_correction_check.isChecked():
                # Corrected and mesh plots should display shifted data to match FFTs
                corrected = erlab.analysis.transform.shift(
                    corrected,
                    typing.cast("xr.DataArray", shift),
                    "eV",
                    shift_coords=False,
                    order=1,
                    cval=0.0,
                )
                mesh = erlab.analysis.transform.shift(
                    mesh,
                    typing.cast("xr.DataArray", shift),
                    "eV",
                    shift_coords=False,
                    order=1,
                    cval=0.0,
                )
            self.corr_image.setDataArray(corrected.T, update_labels=False)
            self.mesh_image.setDataArray(mesh.T, update_labels=False)
            self.corr_fft_image.setImage(log_magnitude_corr)

    @QtCore.Slot()
    def _corr_itool(self) -> None:
        if self._corrected is not None:  # pragma: no branch
            tool = erlab.interactive.itool(self._corrected, execute=False)
            if isinstance(tool, QtWidgets.QWidget):
                if self._itool_corr is not None:
                    self._itool_corr.close()
                    self._itool_corr.deleteLater()
                self._itool_corr = tool
                self._itool_corr.show()

    @QtCore.Slot()
    def _mesh_itool(self) -> None:
        if self._mesh is not None:  # pragma: no branch
            tool = erlab.interactive.itool(self._mesh, execute=False)
            if isinstance(tool, QtWidgets.QWidget):
                if self._itool_mesh is not None:
                    self._itool_mesh.close()
                    self._itool_mesh.deleteLater()
                self._itool_mesh = tool
                self._itool_mesh.show()

    @QtCore.Slot()
    def save_mesh(self) -> None:
        if self._mesh is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No mesh data",
                "No mesh data to save. Please run the mesh removal first.",
            )
            return

        dialog = QtWidgets.QFileDialog()
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.AnyFile)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        dialog.setNameFilters(["xarray HDF5 Files (*.h5)", "All files (*)"])

        last_dir = pg.PlotItem.lastFileDir
        if not last_dir:
            last_dir = erlab.interactive.imagetool.manager._get_recent_directory()
        if not last_dir:
            last_dir = os.getcwd()

        dialog.setDirectory(os.path.join(last_dir, "mesh.h5"))

        if dialog.exec():
            filename = dialog.selectedFiles()[0]
            self._mesh.to_netcdf(filename, engine="h5netcdf", invalid_netcdf=True)
            pg.PlotItem.lastFileDir = os.path.dirname(filename)

    @QtCore.Slot()
    def copy_code(self) -> str:
        return erlab.interactive.utils.generate_code(
            erlab.analysis.mesh.remove_mesh,
            args=[f"|{self.data_name}|"],
            kwargs=self.get_params_dict(),
            module="era.mesh",
            assign=("corrected", "mesh"),
            remove_defaults=False,
            copy=True,
        )


def meshtool(
    data: xr.DataArray, data_name: str | None = None, *, execute: bool | None = None
) -> MeshTool:
    """Interactive tool for mesh removal.

    This tool can also be accessed from the menu

    Parameters
    ----------
    data
        Data to extract the mesh from. The data dimensions must include 'alpha' and
        'eV'. All other dimensions are averaged over to extract the mesh pattern.
    data_name
        Name of the data variable in the generated code. If not provided, the name is
        automatically determined.
    """
    if data_name is None:
        try:
            data_name = str(varname.argname("data", func=meshtool, vars_only=False))
        except varname.VarnameRetrievingError:
            data_name = "data"

    if not all(dim in data.dims for dim in {"alpha", "eV"}):
        raise ValueError("Input DataArray must have 'alpha' and 'eV' dimensions.")

    with erlab.interactive.utils.setup_qapp(execute):
        win = MeshTool(data, data_name=data_name)
        win.show()
        win.raise_()
        win.activateWindow()

    return win
