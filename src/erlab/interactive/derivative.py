"""Interactive tool for visualizing dispersive data."""

__all__ = ["dtool"]

import functools
import os
import sys
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pyqtgraph as pg
import varname
import xarray as xr
from qtpy import QtCore, QtWidgets, uic

from erlab.analysis.image import (
    curvature,
    gaussian_filter,
    minimum_gradient,
    scaled_laplace,
)
from erlab.interactive.utils import (
    copy_to_clipboard,
    gen_function_code,
    parse_data,
    xImageItem,
)

if TYPE_CHECKING:
    from collections.abc import Hashable


class DerivativeTool(
    *uic.loadUiType(os.path.join(os.path.dirname(__file__), "dtool.ui"))  # type: ignore[misc]
):
    def __init__(self, data: xr.DataArray, *, data_name: str | None = None):
        if data_name is None:
            try:
                data_name = cast(
                    str,
                    varname.argname("data", func=self.__init__, vars_only=False),  # type: ignore[misc]
                )
            except varname.VarnameRetrievingError:
                data_name = "data"

        self.data_name: str = data_name

        # Initialize UI
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("")

        if data.ndim != 2:
            raise ValueError("Input DataArray must be 2D")

        self.data: xr.DataArray = parse_data(data)
        self._result: xr.DataArray = self.data.copy()

        self.xdim: Hashable = self.data.dims[1]
        self.ydim: Hashable = self.data.dims[0]

        self.xinc: float = abs(float(self.data[self.xdim][1] - self.data[self.xdim][0]))
        self.yinc: float = abs(float(self.data[self.ydim][1] - self.data[self.ydim][0]))

        self.sx_spin.setRange(0.1 * self.xinc, 50 * self.xinc)
        self.sy_spin.setRange(0.1 * self.yinc, 50 * self.yinc)

        self.sx_spin.setSingleStep(self.xinc / 2)
        self.sy_spin.setSingleStep(self.yinc / 2)

        self.reset_smooth()
        self.reset_interp()

        self.interp_group.setChecked(False)

        self.images: tuple[xImageItem, xImageItem] = (
            xImageItem(axisOrder="row-major"),
            xImageItem(axisOrder="row-major"),
        )
        self.hists: tuple[pg.HistogramLUTItem, pg.HistogramLUTItem] = (
            pg.HistogramLUTItem(),
            pg.HistogramLUTItem(),
        )
        self.plots: list[pg.PlotItem] = []

        for i in range(2):
            plot = self.graphics_layout.addPlot(i, 0, 1, 1)
            self.plots.append(plot)
            self.graphics_layout.addItem(self.hists[i], i, 1)
            plot.addItem(self.images[i])
            self.hists[i].setImageItem(self.images[i])
            # plot.showGrid(x=True, y=True, alpha=0.5)

        self.plots[0].setXLink(self.plots[1])
        self.plots[0].setYLink(self.plots[1])

        self.interp_rst_btn.clicked.connect(self.reset_interp)
        self.smooth_rst_btn.clicked.connect(self.reset_smooth)

        self.interp_group.toggled.connect(self.update_preprocess)
        self.smooth_group.toggled.connect(self.update_preprocess)
        self.nx_spin.valueChanged.connect(self.update_preprocess)
        self.ny_spin.valueChanged.connect(self.update_preprocess)
        self.sx_spin.valueChanged.connect(self.update_preprocess)
        self.sy_spin.valueChanged.connect(self.update_preprocess)
        self.sn_spin.valueChanged.connect(self.update_preprocess)

        self.hi_spin.valueChanged.connect(self.update_image)
        self.lo_spin.valueChanged.connect(self.update_image)

        self.tab_widget.currentChanged.connect(self.update_result)
        self.x_radio.clicked.connect(self.update_result)
        self.y_radio.clicked.connect(self.update_result)
        self.lapl_factor_spin.valueChanged.connect(self.update_result)
        self.curv_a0_spin.valueChanged.connect(self.update_result)
        self.curv_factor_spin.valueChanged.connect(self.update_result)

        self.copy_btn.clicked.connect(self.copy_code)

        self.update_result()

    @QtCore.Slot()
    def reset_smooth(self):
        self.sx_spin.setValue(self.xinc)
        self.sy_spin.setValue(self.yinc)

    @QtCore.Slot()
    def reset_interp(self):
        self.nx_spin.setValue(self.data.sizes[self.xdim])
        self.ny_spin.setValue(self.data.sizes[self.ydim])

    @property
    def result(self) -> xr.DataArray:
        return self._result

    @result.setter
    def result(self, value: xr.DataArray):
        self._result = value
        self.update_image()

    @functools.cached_property
    def processed_data(self) -> xr.DataArray:
        out = self.data
        if self.interp_group.isChecked():
            out = self.data.interp(
                {
                    self.xdim: np.linspace(  # type: ignore[call-overload]
                        *self.data[self.xdim].values[[0, -1]], self.nx_spin.value()
                    ),
                    self.ydim: np.linspace(  # type: ignore[call-overload]
                        *self.data[self.ydim].values[[0, -1]], self.ny_spin.value()
                    ),
                }
            )
        if self.smooth_group.isChecked():
            for _ in range(self.sn_spin.value()):
                out = gaussian_filter(
                    out,
                    sigma={
                        self.xdim: np.round(
                            self.sx_spin.value(), self.sx_spin.decimals()
                        ),
                        self.ydim: np.round(
                            self.sy_spin.value(), self.sy_spin.decimals()
                        ),
                    },
                )
        self.images[0].setDataArray(out)
        return out

    @QtCore.Slot()
    def update_image(self):
        self.images[1].setDataArray(
            self.result, levels=self.get_levels(self.result.values)
        )

    def get_levels(self, data, cutoff=None) -> tuple[float, float]:
        if cutoff is None:
            cutoff = (self.lo_spin.value(), self.hi_spin.value())
        else:
            try:
                cutoff = list(cutoff.__iter__)
            except AttributeError:
                cutoff = [cutoff] * 2

        pu, pl = np.percentile(data, [100 - cutoff[1], cutoff[0]])
        return max(pl, data.min()), min(pu, data.max())

    @QtCore.Slot()
    def update_preprocess(self):
        self.__dict__.pop("processed_data", None)
        self.update_result()

    @QtCore.Slot()
    def update_result(self):
        match self.tab_widget.currentIndex():
            case 0:
                dim = self.xdim if self.x_radio.isChecked() else self.ydim
                self.result = self.processed_data.differentiate(dim).differentiate(dim)
            case 1:
                self.result = scaled_laplace(
                    self.processed_data,
                    factor=np.round(
                        self.lapl_factor_spin.value(), self.lapl_factor_spin.decimals()
                    ),
                )
            case 2:
                self.result = curvature(
                    self.processed_data,
                    a0=np.round(
                        self.curv_a0_spin.value(), self.curv_a0_spin.decimals()
                    ),
                    factor=np.round(
                        self.curv_factor_spin.value(), self.curv_factor_spin.decimals()
                    ),
                )
            case 3:
                self.result = minimum_gradient(self.processed_data)

    def copy_code(self) -> str:
        lines: list[str] = []

        data_name = (
            self.data_name
        )  # "".join([s.strip() for s in self.data_name.split("\n")])
        if self.interp_group.isChecked():
            arg_dict: dict[str, Any] = {
                str(dim): f"|np.linspace(*{data_name}['{dim}'][[0, -1]], {n})|"
                for dim, n in zip(
                    [self.xdim, self.ydim],
                    [self.nx_spin.value(), self.ny_spin.value()],
                    strict=True,
                )
            }
            lines.append(
                gen_function_code(
                    copy=False, **{f"_processed = {data_name}.interp": [arg_dict]}
                )
            )
            data_name = "_processed"

        if self.smooth_group.isChecked():
            arg_dict = {
                "sigma": dict(
                    zip(
                        (self.xdim, self.ydim),
                        [
                            np.round(s.value(), s.decimals())
                            for s in (self.sx_spin, self.sy_spin)
                        ],
                        strict=True,
                    )
                )
            }
            lines.append(f"_processed = {data_name}.copy()")
            data_name = "_processed"
            lines.extend(
                (
                    f"for _ in range({self.sn_spin.value()}):",
                    "\t"
                    + gen_function_code(
                        copy=False,
                        **{
                            "_processed = era.image.gaussian_filter": [
                                f"|{data_name}|",
                                arg_dict,
                            ]
                        },
                    ),
                )
            )
            data_name = "_processed"

        if self.tab_widget.currentIndex() == 0:
            dim = self.xdim if self.x_radio.isChecked() else self.ydim
            lines.append(
                f"result = {data_name}.differentiate('{dim}').differentiate('{dim}')"
            )
        else:
            match self.tab_widget.currentIndex():
                case 1:
                    fname = "era.image.scaled_laplace"
                    arg_dict = {
                        "factor": np.round(
                            self.lapl_factor_spin.value(),
                            self.lapl_factor_spin.decimals(),
                        )
                    }
                case 2:
                    fname = "era.image.curvature"
                    arg_dict = {
                        "a0": np.round(
                            self.curv_a0_spin.value(), self.curv_a0_spin.decimals()
                        ),
                        "factor": np.round(
                            self.curv_factor_spin.value(),
                            self.curv_factor_spin.decimals(),
                        ),
                    }
                case 3:
                    fname = "era.image.minimum_gradient"
                    arg_dict = {}

            lines.append(
                gen_function_code(
                    copy=False, **{f"result = {fname}": [f"|{data_name}|", arg_dict]}
                )
            )

        return copy_to_clipboard(lines)


def dtool(data, data_name: str | None = None, *, execute: bool | None = None):
    if data_name is None:
        try:
            data_name = varname.argname("data", func=dtool, vars_only=False)  # type: ignore[assignment]
        except varname.VarnameRetrievingError:
            data_name = "data"

    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)

    cast(QtWidgets.QApplication, qapp).setStyle("Fusion")

    win = DerivativeTool(data, data_name=data_name)
    win.show()
    win.raise_()
    win.activateWindow()

    if execute is None:
        execute = True
        try:
            shell = get_ipython().__class__.__name__  # type: ignore
            if shell in ["ZMQInteractiveShell", "TerminalInteractiveShell"]:
                execute = False
                from IPython.lib.guisupport import start_event_loop_qt4

                start_event_loop_qt4(qapp)
        except NameError:
            pass
    if execute:
        qapp.exec()
    if not execute:
        return win
