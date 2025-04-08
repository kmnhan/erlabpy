"""Interactive tool for visualizing dispersive data.

.. image:: ../images/dtool_light.png
    :align: center
    :alt: Derivative tool screenshot
    :class: only-light

.. only:: format_html

    .. image:: ../images/dtool_dark.png
        :align: center
        :alt: Derivative tool screenshot
        :class: only-dark
"""

__all__ = ["dtool"]

import functools
import os
import typing
from collections.abc import Callable, Hashable

import numpy as np
import pyqtgraph as pg
import xarray as xr
from qtpy import QtCore, QtWidgets, uic

import erlab

if typing.TYPE_CHECKING:
    import varname
else:
    import lazy_loader as _lazy

    varname = _lazy.load("varname")


class DerivativeTool(
    *uic.loadUiType(os.path.join(os.path.dirname(__file__), "dtool.ui"))  # type: ignore[misc]
):
    def __init__(self, data: xr.DataArray, *, data_name: str | None = None) -> None:
        if data_name is None:
            try:
                data_name = typing.cast(
                    "str",
                    varname.argname("data", func=self.__init__, vars_only=False),  # type: ignore[misc]
                )
            except varname.VarnameRetrievingError:
                data_name = "data"

        self.data_name: str = data_name

        # Initialize UI
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("")

        if data.isnull().any():
            self.show()
            QtWidgets.QMessageBox.warning(
                self,
                "Warning",
                "Input DataArray contains NaN values. These will be filled with zeros.",
            )
            data = data.fillna(0.0)

        if data.ndim != 2:
            raise ValueError("Input DataArray must be 2D")

        self.data: xr.DataArray = erlab.interactive.utils.parse_data(data)
        self._result: xr.DataArray = self.data.copy()

        self.xdim: Hashable = self.data.dims[1]
        self.ydim: Hashable = self.data.dims[0]

        self.reset_smooth()
        self.reset_interp()

        self.interp_group.setChecked(False)

        self.images: tuple[
            erlab.interactive.utils.xImageItem, erlab.interactive.utils.xImageItem
        ] = (
            erlab.interactive.utils.xImageItem(axisOrder="row-major"),
            erlab.interactive.utils.xImageItem(axisOrder="row-major"),
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
        self.smooth_combo.currentIndexChanged.connect(self.refresh_smooth_mode)

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
        self.curv1d_a0_spin.valueChanged.connect(self.update_result)
        self.x_radio_curv1d.clicked.connect(self.update_result)
        self.y_radio_curv1d.clicked.connect(self.update_result)
        self.curv_a0_spin.valueChanged.connect(self.update_result)
        self.curv_factor_spin.valueChanged.connect(self.update_result)

        self.copy_btn.clicked.connect(self.copy_code)

        self.update_result()

    @QtCore.Slot()
    def reset_smooth(self) -> None:
        self.sx_spin.blockSignals(True)
        self.sy_spin.blockSignals(True)
        self.sx_spin.setValue(1.0)
        self.sy_spin.setValue(1.0)
        self.sx_spin.blockSignals(False)
        self.sy_spin.blockSignals(False)
        self.sn_spin.setValue(1)

    @QtCore.Slot()
    def reset_interp(self) -> None:
        self.nx_spin.setValue(self.data.sizes[self.xdim])
        self.ny_spin.setValue(self.data.sizes[self.ydim])

    @QtCore.Slot()
    def refresh_smooth_mode(self) -> None:
        match self.smooth_combo.currentIndex():
            case 0:
                self.sx_spin.setDecimals(4)
                self.sy_spin.setDecimals(4)
            case 1:
                self.sx_spin.setDecimals(0)
                self.sy_spin.setDecimals(0)
        self.update_preprocess()

    @property
    def smooth_args(self) -> dict[Hashable, float]:
        sx_value = np.round(self.sx_spin.value(), self.sx_spin.decimals())
        sy_value = np.round(self.sy_spin.value(), self.sy_spin.decimals())
        match self.smooth_combo.currentIndex():
            case 0:
                xcoords, ycoords = (
                    self.data[self.xdim].values,
                    self.data[self.ydim].values,
                )
                xinc, yinc = (
                    np.abs(xcoords[1] - xcoords[0]),
                    np.abs(ycoords[1] - ycoords[0]),
                )

                sx_value, sy_value = (
                    round(sx_value * xinc, erlab.utils.array.effective_decimals(xinc)),
                    round(sy_value * yinc, erlab.utils.array.effective_decimals(yinc)),
                )
            case _:
                sx_value, sy_value = int(sx_value), int(sy_value)

        out = {}
        if sx_value > 0:
            out[self.xdim] = sx_value
        if sy_value > 0:
            out[self.ydim] = sy_value
        return out

    @property
    def result(self) -> xr.DataArray:
        return self._result

    @result.setter
    def result(self, value: xr.DataArray) -> None:
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
            match self.smooth_combo.currentIndex():
                case 0:
                    for _ in range(self.sn_spin.value()):
                        out = erlab.analysis.image.gaussian_filter(
                            out, sigma=self.smooth_args
                        )
                case 1:
                    for _ in range(self.sn_spin.value()):
                        out = erlab.analysis.image.boxcar_filter(
                            out,
                            size=typing.cast("dict[Hashable, int]", self.smooth_args),
                        )

        self.images[0].setDataArray(out)
        return out

    @QtCore.Slot()
    def update_image(self) -> None:
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
    def update_preprocess(self) -> None:
        self.__dict__.pop("processed_data", None)
        self.update_result()

    @property
    def process_func(self) -> Callable:
        match self.tab_widget.currentIndex():
            case 0:
                return erlab.analysis.image.diffn
            case 1:
                return erlab.analysis.image.scaled_laplace
            case 2:
                return erlab.analysis.image.curvature1d
            case 3:
                return erlab.analysis.image.curvature
            case _:
                return erlab.analysis.image.minimum_gradient

    @property
    def process_kwargs(self) -> dict[str, typing.Any]:
        match self.tab_widget.currentIndex():
            case 0:
                return {
                    "coord": self.xdim if self.x_radio.isChecked() else self.ydim,
                    "order": 2,
                }
            case 1:
                return {
                    "factor": np.round(
                        self.lapl_factor_spin.value(), self.lapl_factor_spin.decimals()
                    )
                }
            case 2:
                return {
                    "along": self.xdim
                    if self.x_radio_curv1d.isChecked()
                    else self.ydim,
                    "a0": np.round(
                        self.curv1d_a0_spin.value(), self.curv1d_a0_spin.decimals()
                    ),
                }
            case 3:
                return {
                    "a0": np.round(
                        self.curv_a0_spin.value(), self.curv_a0_spin.decimals()
                    ),
                    "factor": np.round(
                        self.curv_factor_spin.value(), self.curv_factor_spin.decimals()
                    ),
                }
            case _:
                return {}

    @QtCore.Slot()
    def update_result(self) -> None:
        self.result = self.process_func(self.processed_data, **self.process_kwargs)

    def copy_code(self) -> str:
        lines: list[str] = []

        data_name = (
            self.data_name
        )  # "".join([s.strip() for s in self.data_name.split("\n")])
        if self.interp_group.isChecked():
            arg_dict: dict[str, typing.Any] = {
                str(dim): f"|np.linspace(*{data_name}['{dim}'].values[[0, -1]], {n})|"
                for dim, n in zip(
                    [self.xdim, self.ydim],
                    [self.nx_spin.value(), self.ny_spin.value()],
                    strict=True,
                )
            }
            lines.append(
                erlab.interactive.utils.generate_code(
                    xr.DataArray.interp,
                    args=[],
                    kwargs=arg_dict,
                    module=data_name,
                    assign="_processed",
                )
            )

            data_name = "_processed"

        if self.smooth_group.isChecked():
            match self.smooth_combo.currentIndex():
                case 0:
                    smooth_func: Callable = erlab.analysis.image.gaussian_filter
                    smooth_kwargs: dict[str, typing.Any] = {"sigma": self.smooth_args}

                case _:
                    smooth_func = erlab.analysis.image.boxcar_filter
                    smooth_kwargs = {"size": self.smooth_args}

            n_repeat = self.sn_spin.value()

            if n_repeat > 1:
                lines.append(f"_processed = {data_name}.copy()")
                data_name = "_processed"

            smooth_func_code: str = erlab.interactive.utils.generate_code(
                smooth_func,
                [f"|{data_name}|"],
                smooth_kwargs,
                module="era.image",
                assign=data_name if (n_repeat > 1) else None,
            )
            if n_repeat == 1:
                data_name = smooth_func_code.replace(" = ", "=")
            else:
                lines.append(f"for _ in range({self.sn_spin.value()}):")
                lines.append("\t" + smooth_func_code)

        lines.append(
            erlab.interactive.utils.generate_code(
                self.process_func,
                [f"|{data_name}|"],
                self.process_kwargs,
                module="era.image",
                assign="result",
                remove_defaults=False,
            )
        )

        return erlab.interactive.utils.copy_to_clipboard(lines)


def dtool(
    data: xr.DataArray, data_name: str | None = None, *, execute: bool | None = None
) -> DerivativeTool:
    """Interactive tool for visualizing dispersive data.

    This tool can also be accessed from the right-click context menu of an image plot in
    an ImageTool window.

    Parameters
    ----------
    data
        Data to visualize. Must be a 2D DataArray with no NaN values.
    data_name
        Name of the data variable in the generated code. If not provided, the name is
        automatically determined.
    """
    if data_name is None:
        try:
            data_name = str(varname.argname("data", func=dtool, vars_only=False))
        except varname.VarnameRetrievingError:
            data_name = "data"

    with erlab.interactive.utils.setup_qapp(execute):
        win = DerivativeTool(data, data_name=data_name)
        win.show()
        win.raise_()
        win.activateWindow()

    return win
