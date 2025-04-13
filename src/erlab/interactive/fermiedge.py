__all__ = ["goldtool", "restool"]

import concurrent.futures
import os
import time
import typing

import numpy as np
import pyqtgraph as pg
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets, uic

import erlab

if typing.TYPE_CHECKING:
    from collections.abc import Callable

    import joblib
    import lmfit
    import scipy.interpolate
    import varname
else:
    import lazy_loader as _lazy

    joblib = _lazy.load("joblib")
    varname = _lazy.load("varname")


LMFIT_METHODS = [
    "leastsq",
    "least_squares",
    "differential_evolution",
    # "brute",
    # "basinhopping",
    # "ampgo",
    # "nelder",
    # "lbfgsb",
    # "powell",
    "cg",
    # "newton",
    # "cobyla",
    # "bfgs",
    # "tnc",
    # "trust-ncg",
    # "trust-exact",
    # "trust-krylov",
    # "trust-constr",
    # "dogleg",
    # "slsqp",
    "emcee",
    # "shgo",
    # "dual_annealing",
]


class EdgeFitter(QtCore.QThread):
    sigIterated = QtCore.Signal(int)
    sigFinished = QtCore.Signal()

    def set_params(self, data, x0, y0, x1, y1, params) -> None:
        self.data = data
        self.x_range: tuple[float, float] = (x0, x1)
        self.y_range: tuple[float, float] = (y0, y1)
        self.params = params
        self.parallel_obj = joblib.Parallel(
            n_jobs=self.params["# CPU"],
            max_nbytes=None,
            return_as="list",
            pre_dispatch="n_jobs",
        )
        self.edge_center: xr.DataArray | None = None
        self.edge_stderr: xr.DataArray | None = None

    @QtCore.Slot()
    def abort_fit(self) -> None:
        self.parallel_obj._aborting = True
        self.parallel_obj._exception = True

    def run(self) -> None:
        self.sigIterated.emit(0)
        with erlab.utils.parallel.joblib_progress_qt(self.sigIterated) as _:
            self.edge_center, self.edge_stderr = typing.cast(
                "tuple[xr.DataArray, xr.DataArray]",
                erlab.analysis.gold.edge(
                    gold=self.data,
                    angle_range=self.x_range,
                    eV_range=self.y_range,
                    bin_size=(self.params["Bin x"], self.params["Bin y"]),
                    temp=self.params["T (K)"],
                    vary_temp=not self.params["Fix T"],
                    bkg_slope=self.params["Linear"],
                    resolution=self.params["Resolution"],
                    fast=self.params["Fast"],
                    method=self.params["Method"],
                    scale_covar=self.params["Scale cov"],
                    progress=False,
                    parallel_obj=self.parallel_obj,
                ),
            )
        self.sigFinished.emit()


class GoldTool(erlab.interactive.utils.AnalysisWindow):
    """Interactive gold edge fitting.

    Parameters
    ----------
    data
        The data to perform Fermi edge fitting on.
    data_corr
        The data to correct with the edge. Defaults to `data`.
    data_name
        Name of the data used in generating the code snipped copied to the clipboard.
        Overrides automatic detection.
    execute
        Whether to execute the tool immediately.
    **kwargs
        Arguments passed onto `erlab.interactive.utils.AnalysisWindow`.

    Signals
    -------
    sigProgressUpdated(int)
        Signal used to update the progress bar.
    sigAbortFitting()
        Signal used to abort the fitting, emitted when the cancel button is clicked.

    """

    sigProgressUpdated = QtCore.Signal(int)  #: :meta private:
    sigAbortFitting = QtCore.Signal()  #: :meta private:

    def __init__(
        self,
        data: xr.DataArray,
        data_corr: xr.DataArray | None = None,
        *,
        data_name: str | None = None,
        execute: bool = True,
        **kwargs,
    ) -> None:
        if set(data.dims) != {"alpha", "eV"}:
            raise ValueError(
                "`data` must be a DataArray with dimensions `alpha` and `eV`"
            )
        if data.dims[0] != "eV":
            data = data.copy().T

        super().__init__(
            data,
            link="x",
            layout="horizontal",
            orientation="vertical",
            num_ax=3,
            **kwargs,
        )

        self._argnames = {}
        if data_name is None:
            try:
                self._argnames["data"] = varname.argname(
                    "data",
                    func=self.__init__,  # type: ignore[misc]
                    vars_only=False,
                )
            except varname.VarnameRetrievingError:
                self._argnames["data"] = "gold"
        else:
            self._argnames["data"] = data_name

        if data_corr is not None:
            try:
                self._argnames["data_corr"] = varname.argname(
                    "data_corr",
                    func=self.__init__,  # type: ignore[misc]
                    vars_only=False,
                )
            except varname.VarnameRetrievingError:
                self._argnames["data_corr"] = "data_corr"

        self.data_corr = data_corr
        self.hists: pg.HistogramLUTItem
        self.axes: list[pg.PlotItem]
        self.images: list[erlab.interactive.utils.xImageItem]

        self.axes[1].setVisible(False)
        self.hists[1].setVisible(False)
        self.axes[2].setVisible(False)
        self.hists[2].setVisible(False)

        temp = self.data.qinfo.get_value("sample_temp")
        if temp is None:
            temp = 30.0
        temp = float(temp)

        self.params_roi = erlab.interactive.utils.ROIControls(self.aw.add_roi(0))
        self.params_edge = erlab.interactive.utils.ParameterGroup(
            {
                "T (K)": {"qwtype": "dblspin", "value": temp, "range": (0.0, 400.0)},
                "Fix T": {"qwtype": "chkbox", "checked": True},
                "Bin x": {"qwtype": "spin", "value": 1, "minimum": 1},
                "Bin y": {"qwtype": "spin", "value": 1, "minimum": 1},
                "Resolution": {
                    "qwtype": "dblspin",
                    "value": 0.02,
                    "range": (0.0, 99.9),
                    "singleStep": 0.001,
                    "decimals": 5,
                },
                "Fast": {
                    "qwtype": "chkbox",
                    "checked": False,
                    "toolTip": "If checked, fit with a broadened step function "
                    "instead of Fermi-Dirac",
                },
                "Linear": {
                    "qwtype": "chkbox",
                    "checked": True,
                    "toolTip": "If unchecked, fixes the slope of the background "
                    "above the Fermi level to zero.",
                },
                "Method": {"qwtype": "combobox", "items": LMFIT_METHODS},
                "Scale cov": {"qwtype": "chkbox", "checked": True},
                "# CPU": {
                    "qwtype": "spin",
                    "value": os.cpu_count(),
                    "minimum": 1,
                    "maximum": os.cpu_count(),
                },
                "go": {
                    "qwtype": "pushbtn",
                    "showlabel": False,
                    "text": "Go",
                    "clicked": self.perform_edge_fit,
                },
            }
        )

        typing.cast(
            "QtWidgets.QComboBox", self.params_edge.widgets["Method"]
        ).setCurrentIndex(1)

        typing.cast(
            "QtWidgets.QCheckBox", self.params_edge.widgets["Fast"]
        ).stateChanged.connect(self._toggle_fast)

        self.params_poly = erlab.interactive.utils.ParameterGroup(
            {
                "Degree": {"qwtype": "spin", "value": 4, "range": (1, 20)},
                "Scale cov": {"qwtype": "chkbox", "checked": True},
                "Residuals": {"qwtype": "chkbox", "checked": False},
                "Corrected": {"qwtype": "chkbox", "checked": False},
                "Shift coords": {"qwtype": "chkbox", "checked": True},
                "itool": {
                    "qwtype": "pushbtn",
                    "notrack": True,
                    "showlabel": False,
                    "text": "Open in ImageTool",
                    "clicked": self.open_itool,
                },
                "copy": {
                    "qwtype": "pushbtn",
                    "notrack": True,
                    "showlabel": False,
                    "text": "Copy to clipboard",
                    "clicked": lambda: self.gen_code("poly"),
                },
            }
        )

        self.params_spl = erlab.interactive.utils.ParameterGroup(
            {
                "Auto": {"qwtype": "chkbox", "checked": True},
                "lambda": {
                    "qwtype": "dblspin",
                    "minimum": 0,
                    "maximum": 10000,
                    "singleStep": 0.001,
                    "decimals": 4,
                },
                "Residuals": {"qwtype": "chkbox", "checked": False},
                "Corrected": {"qwtype": "chkbox", "checked": False},
                "Shift coords": {"qwtype": "chkbox", "checked": True},
                "itool": {
                    "qwtype": "pushbtn",
                    "notrack": True,
                    "showlabel": False,
                    "text": "Open in ImageTool",
                    "clicked": self.open_itool,
                },
                "copy": {
                    "qwtype": "pushbtn",
                    "notrack": True,
                    "showlabel": False,
                    "text": "Copy to clipboard",
                    "clicked": lambda: self.gen_code("spl"),
                },
            }
        )
        _auto_check = typing.cast(
            "QtWidgets.QCheckBox", self.params_spl.widgets["Auto"]
        )
        self.params_spl.widgets["lambda"].setDisabled(
            _auto_check.checkState() == QtCore.Qt.CheckState.Checked
        )
        _auto_check.toggled.connect(
            lambda _: self.params_spl.widgets["lambda"].setDisabled(
                _auto_check.checkState() == QtCore.Qt.CheckState.Checked
            )
        )

        self.controls.addWidget(self.params_roi)
        self.controls.addWidget(self.params_edge)

        self.params_tab = QtWidgets.QTabWidget()
        self.params_tab.addTab(self.params_poly, "Polynomial")
        self.params_tab.addTab(self.params_spl, "Spline")
        self.params_tab.currentChanged.connect(
            lambda i: self.perform_fit(("poly", "spl")[i])
        )
        self.controls.addWidget(self.params_tab)

        self.params_poly.setDisabled(True)
        self.params_spl.setDisabled(True)
        self.params_tab.setDisabled(True)

        self.controlgroup.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Preferred
        )

        self.scatterplots = [
            pg.ScatterPlotItem(
                size=3,
                pen=pg.mkPen(None),
                brush=pg.mkBrush(255, 0, 0, 200),
                pxMode=True,
            )
            for _ in range(2)
        ]

        self.errorbars = [pg.ErrorBarItem(pen=pg.mkPen(color="red")) for _ in range(2)]
        self.polycurves = [pg.PlotDataItem() for _ in range(2)]
        for i in range(2):
            self.axes[i].addItem(self.scatterplots[i])
            self.axes[i].addItem(self.errorbars[i])
            self.axes[i].addItem(self.polycurves[i])
        self.params_poly.sigParameterChanged.connect(lambda: self.perform_fit("poly"))
        self.params_spl.sigParameterChanged.connect(lambda: self.perform_fit("spl"))

        self.axes[0].disableAutoRange()

        # Setup time calculation
        self.start_time: float
        self.step_times: list[float]

        # Setup progress bar
        self.progress: QtWidgets.QProgressDialog = QtWidgets.QProgressDialog(self)
        self.pbar: QtWidgets.QProgressBar = QtWidgets.QProgressBar()

        self.progress.setLabelText("Fitting...")
        self.progress.setCancelButtonText("Abort!")
        self.progress.setRange(0, 100)
        self.progress.setMinimumDuration(0)
        self.progress.setBar(self.pbar)
        self.progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.progress.setFixedSize(self.progress.size())
        self.progress.canceled.disconnect(self.progress.cancel)  # don't auto close
        self.progress.canceled.connect(self.abort_fit)
        self.progress.setAutoReset(False)
        self.progress.cancel()

        # Setup fitter thread
        # This allows the GUI to remain responsive during fitting so it can be aborted
        self.fitter = EdgeFitter()
        self.fitter.sigIterated.connect(self.iterated)
        self.fitter.sigFinished.connect(self.post_fit)
        self.sigAbortFitting.connect(self.fitter.abort_fit)

        # Resize roi to data bounds
        eV_span = self.data.eV.values[-1] - self.data.eV.values[0]
        ang_span = self.data.alpha.values[-1] - self.data.alpha.values[0]
        x1 = self.data.alpha.values.mean() + ang_span * 0.45
        x0 = self.data.alpha.values.mean() - ang_span * 0.45
        y1 = self.data.eV.values[-1] - eV_span * 0.015
        y0 = y1 - eV_span * 0.3
        self.params_roi.modify_roi(x0, y0, x1, y1)

        # Initialize imagetool variable
        self._itool: QtWidgets.QWidget | None = None

        # Initialize fit result
        self.result: scipy.interpolate.BSpline | lmfit.model.ModelResult | None = None

        self.__post_init__(execute=execute)

    def _toggle_fast(self) -> None:
        self.params_edge.widgets["T (K)"].setDisabled(
            bool(self.params_edge.values["Fast"])
        )
        self.params_edge.widgets["Fix T"].setDisabled(
            bool(self.params_edge.values["Fast"])
        )

    @QtCore.Slot(int)
    def iterated(self, n: int) -> None:
        if n == 0:
            self.progress.setLabelText("")
        elif n == 1:
            self.step_times = []
            self.start_time = time.perf_counter()
        elif n > 1:
            now = time.perf_counter()
            self.step_times.append(now - self.start_time)
            self.start_time = now
            timeleft = (self.progress.maximum() - (n - 1)) * np.mean(self.step_times)
            self.progress.setLabelText(f"{round(timeleft)} seconds left...")

        self.progress.setValue(n)
        self.pbar.setFormat(f"{n}/{self.progress.maximum()} finished")

    @QtCore.Slot()
    def perform_edge_fit(self) -> None:
        self.progress.setVisible(True)
        self.params_roi.draw_button.setChecked(False)
        x0, y0, x1, y1 = (float(np.round(x, 3)) for x in self.params_roi.roi_limits)
        params = self.params_edge.values
        n_total = len(
            self.data.alpha.coarsen(alpha=params["Bin x"], boundary="trim")
            .mean()
            .sel(alpha=slice(x0, x1))
        )
        self.progress.setMaximum(n_total)
        self.fitter.set_params(self.data, x0, y0, x1, y1, params)
        self.fitter.start()

    @QtCore.Slot()
    def abort_fit(self) -> None:
        self.sigAbortFitting.emit()

    @QtCore.Slot()
    def post_fit(self) -> None:
        self.progress.reset()
        self.edge_center, self.edge_stderr = (
            typing.cast("xr.DataArray", self.fitter.edge_center),
            typing.cast("xr.DataArray", self.fitter.edge_stderr),
        )

        xval = self.edge_center.alpha.values
        yval = self.edge_center.values
        for i in range(2):
            self.scatterplots[i].setData(x=xval, y=yval)
            self.errorbars[i].setData(x=xval, y=yval, height=self.edge_stderr.values)

        self.params_poly.setDisabled(False)
        self.params_spl.setDisabled(False)
        self.params_tab.setDisabled(False)
        self.perform_fit("poly")

    def perform_fit(self, mode="poly") -> None:
        match mode:
            case "poly":
                edgefunc = self._perform_poly_fit()
                params = self.params_poly.values
            case "spl":
                edgefunc = self._perform_spline_fit()
                params = self.params_spl.values
        for i in range(2):
            xval = self.data.alpha.values
            if i == 1 and params["Residuals"]:
                yval = np.zeros_like(xval)
            else:
                yval = edgefunc(xval)
            self.polycurves[i].setData(x=xval, y=yval)

        xval = self.edge_center.alpha.values
        if params["Residuals"]:
            yval = edgefunc(xval) - self.edge_center.values
        else:
            yval = self.edge_center.values
        self.errorbars[1].setData(x=xval, y=yval)
        self.scatterplots[1].setData(x=xval, y=yval, height=self.edge_stderr)

        self.aw.axes[1].setVisible(True)
        self.aw.images[-1].setDataArray(self.corrected)
        self.aw.axes[2].setVisible(params["Corrected"])
        self.aw.hists[2].setVisible(params["Corrected"])

    def _perform_poly_fit(self):
        params = self.params_poly.values
        self.result = erlab.analysis.gold.poly_from_edge(
            center=self.edge_center,
            weights=1 / self.edge_stderr,
            degree=params["Degree"],
            method=self.params_edge.values["Method"],
            scale_covar=params["Scale cov"],
        ).modelfit_results.values.item()
        target = self.data if self.data_corr is None else self.data_corr
        self.corrected = erlab.analysis.gold.correct_with_edge(
            target, self.result, plot=False, shift_coords=params["Shift coords"]
        )
        return lambda x: self.result.eval(self.result.params, x=x)

    def _perform_spline_fit(self):
        params = self.params_spl.values
        if params["Auto"]:
            params["lambda"] = None
        self.result = erlab.analysis.gold.spline_from_edge(
            center=self.edge_center,
            weights=np.asarray(1 / self.edge_stderr),
            lam=params["lambda"],
        )

        target = self.data if self.data_corr is None else self.data_corr
        self.corrected = erlab.analysis.gold.correct_with_edge(
            target, self.result, plot=False, shift_coords=params["Shift coords"]
        )
        return self.result

    @QtCore.Slot()
    def open_itool(self) -> None:
        tool = erlab.interactive.itool(self.corrected, execute=False)
        if isinstance(tool, QtWidgets.QWidget):
            if self._itool is not None:
                self._itool.close()

            tool.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
            tool.destroyed.connect(lambda: setattr(self, "_itool", None))
            self._itool = tool
            self._itool.show()

    def gen_code(self, mode: str) -> None:
        p0 = self.params_edge.values
        match mode:
            case "poly":
                p1 = self.params_poly.values
            case "spl":
                p1 = self.params_spl.values
        x0, y0, x1, y1 = (float(np.round(x, 3)) for x in self.params_roi.roi_limits)

        arg_dict: dict[str, typing.Any] = {
            "angle_range": (x0, x1),
            "eV_range": (y0, y1),
            "bin_size": (p0["Bin x"], p0["Bin y"]),
            "temp": p0["T (K)"],
            "vary_temp": not p0["Fix T"],
            "bkg_slope": p0["Linear"],
            "resolution": p0["Resolution"],
            "fast": p0["Fast"],
            "method": p0["Method"],
            "scale_covar_edge": p0["Scale cov"],
        }

        match mode:
            case "poly":
                func: Callable = erlab.analysis.gold.poly
                arg_dict["degree"] = p1["Degree"]
            case "spl":
                func = erlab.analysis.gold.spline
                if p1["Auto"]:
                    arg_dict["lam"] = None
                else:
                    arg_dict["lam"] = p1["lambda"]

        if p0["Fast"]:
            del arg_dict["temp"]

        if mode == "poly" and not p1["Scale cov"]:
            arg_dict["scale_covar"] = False

        code_str = erlab.interactive.utils.generate_code(
            func,
            [f"|{self._argnames['data']}|"],
            arg_dict,
            module="era.gold",
            assign="modelresult",
        )
        if self.data_corr is not None:
            code_str += "\n" + erlab.interactive.utils.generate_code(
                erlab.analysis.gold.correct_with_edge,
                [f"|{self._argnames['data_corr']}|", "|modelresult|"],
                {"shift_coords": p1["Shift coords"]},
                module="era.gold",
                assign="corrected",
            )
        erlab.interactive.utils.copy_to_clipboard(code_str)


class ResolutionTool(
    *uic.loadUiType(os.path.join(os.path.dirname(__file__), "restool.ui"))  # type: ignore[misc]
):
    _sigTriggerFit = QtCore.Signal()

    def __init__(self, data: xr.DataArray, *, data_name: str | None = None) -> None:
        if (data.ndim != 2) or ("eV" not in data.dims):
            raise ValueError("Data must be 2D and have an 'eV' dimension.")
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("")

        if data.dims.index("eV") != 1:
            data = data.T
        self.data = data

        self.y_dim: str = str(data.dims[0])
        self._x_range = data["eV"].values[[0, -1]]
        self._y_range = data[self.y_dim].values[[0, -1]]

        self._x_decimals = erlab.utils.array.effective_decimals(data["eV"].values)
        self._y_decimals = erlab.utils.array.effective_decimals(data[self.y_dim].values)

        self.x0_spin.setRange(*self._x_range)
        self.x1_spin.setRange(*self._x_range)
        self.y0_spin.setRange(*self._y_range)
        self.y1_spin.setRange(*self._y_range)
        self.x0_spin.setDecimals(self._x_decimals)
        self.x1_spin.setDecimals(self._x_decimals)
        self.y0_spin.setDecimals(self._y_decimals)
        self.y1_spin.setDecimals(self._y_decimals)
        self.x0_spin.setSingleStep(10 ** -(self._x_decimals - 1))
        self.x1_spin.setSingleStep(10 ** -(self._x_decimals - 1))
        self.y0_spin.setSingleStep(10 ** -(self._y_decimals - 1))
        self.y1_spin.setSingleStep(10 ** -(self._y_decimals - 1))

        self.res_spin.setDecimals(self._x_decimals + 1)
        self.res_spin.setSingleStep(10 ** -(self._x_decimals - 1))
        self.res_spin.setValue(0.002)
        self.center_spin.setRange(*self._x_range)
        self.center_spin.setDecimals(self._x_decimals + 1)
        self.center_spin.setSingleStep(10 ** -(self._x_decimals - 1))

        if data_name is None:
            try:
                data_name = typing.cast(
                    "str",
                    varname.argname("data", func=self.__init__, vars_only=False),  # type: ignore[misc]
                )
            except varname.VarnameRetrievingError:
                data_name = "data"

        self.data_name: str = data_name

        self.plot0 = self.graphics_layout.addPlot(row=0, col=0)
        self.plot1 = self.graphics_layout.addPlot(row=1, col=0)
        self.plot1.setXLink(self.plot0)

        self.image = erlab.interactive.utils.xImageItem(axisOrder="row-major")
        self.image.setDataArray(self.data)
        self.plot0.addItem(self.image)

        self.edc_curve = pg.ScatterPlotItem(
            size=3,
            pen=pg.mkPen(None),
            brush=pg.mkBrush(255, 255, 255, 200),
            pxMode=True,
        )
        self.plot1.addItem(self.edc_curve)

        self.edc_fit = pg.PlotDataItem(pen=pg.mkPen("r"))
        self.plot1.addItem(self.edc_fit)

        y_offset = self._y_range.mean()
        initial_y_range = (self._y_range - y_offset) * 0.75 + y_offset
        self.y_region = pg.LinearRegionItem(
            values=initial_y_range,
            orientation="horizontal",
            swapMode="block",
            bounds=self._y_range,
        )
        self.y_region.sigRegionChanged.connect(self._y_region_changed)
        self.plot0.addItem(self.y_region)

        x_width = self._x_range[-1] - self._x_range[0]
        initial_x_range = (self._x_range.mean(), self._x_range[-1] - x_width * 0.04)
        self.x_region = pg.LinearRegionItem(
            values=initial_x_range,
            orientation="vertical",
            swapMode="block",
            bounds=self._x_range,
        )
        self.x_region.sigRegionChanged.connect(self._x_region_changed)
        self.plot1.addItem(self.x_region)

        self.connect_signals()
        self.graphics_layout.setFocus()

        self.resize(800, 600)

        self._result_ds: xr.Dataset | None = None

        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    @property
    def x_range(self) -> tuple[float, float]:
        """Currently selected x range (eV) for the fit."""
        x0 = round(self.x0_spin.value(), self._x_decimals)
        x1 = round(self.x1_spin.value(), self._x_decimals)
        return x0, x1

    @property
    def y_range(self) -> tuple[float, float]:
        """Currently selected y range to average EDCs."""
        y0 = round(self.y0_spin.value(), self._y_decimals)
        y1 = round(self.y1_spin.value(), self._y_decimals)
        return y0, y1

    def _update_edc(self) -> None:
        """Calculate averaged EDC and update the plot."""
        with xr.set_options(keep_attrs=True):
            self.averaged_edc = self.data.sel({self.y_dim: slice(*self.y_range)}).mean(
                self.y_dim
            )
        self.edc_curve.setData(
            x=self.averaged_edc["eV"].values, y=self.averaged_edc.values
        )
        self.edc_fit.setData(x=[], y=[])

    @property
    def _guessed_params(self) -> "lmfit.Parameters":
        with xr.set_options(keep_attrs=True):
            target = self.averaged_edc.sel(eV=slice(*self.x_range))
        return erlab.analysis.fit.models.FermiEdgeModel().guess(
            target, target.eV.values
        )

    @QtCore.Slot()
    def _guess_temp(self) -> None:
        self.temp_spin.setValue(self._guessed_params["temp"].value)

    @QtCore.Slot()
    def _guess_center(self) -> None:
        self.center_spin.setValue(self._guessed_params["center"].value)

    @property
    def fit_params(self) -> dict[str, typing.Any]:
        """Current arguments for :func:`erlab.analysis.gold.quick_fit`."""
        return {
            "eV_range": self.x_range,
            "method": self.method_combo.currentText(),
            "temp": self.temp_spin.value(),
            "resolution": self.res_spin.value(),
            "center": self.center_spin.value(),
            "fix_temp": self.fix_temp_check.isChecked(),
            "fix_center": self.fix_center_check.isChecked(),
            "fix_resolution": self.fix_res_check.isChecked(),
            "bkg_slope": self.slope_check.isChecked(),
        }

    def _fit_failed(self, info_text: str) -> None:
        self.overview_label.setText(
            f'<span style="font-weight:600; color:#ff5555;">{info_text}</span>'
        )

    @QtCore.Slot()
    def do_fit(self) -> None:
        """Perform a fit on the averaged EDC and update results."""
        t0 = time.perf_counter()

        # Execute in threadpool
        future = self._executor.submit(
            erlab.analysis.gold.quick_fit,
            self.averaged_edc,
            **(self.fit_params | {"max_nfev": self.nfev_spin.value()}),
        )
        try:
            self._result_ds = future.result(timeout=self.timeout_spin.value())
        except TimeoutError:
            future.cancel()
            self.live_check.setChecked(False)
            self._fit_failed(f"Fit timed out in {time.perf_counter() - t0:.2f} seconds")
            self.edc_fit.setData(x=[], y=[])
            return
        fit_time: float = time.perf_counter() - t0

        # Update plot
        self.edc_fit.setData(
            x=self._result_ds["eV"].values, y=self._result_ds.modelfit_best_fit.values
        )

        # Update overview
        modelresult = self._result_ds.modelfit_results.item()
        if not modelresult.success:
            self._fit_failed(
                f"Fit failed in {fit_time:.2f} s (nfev = {modelresult.nfev})"
            )
        else:
            self.overview_label.setText(
                '<span style="font-weight:600; color:#32cd32;">'
                f"Fit converged (nfev = {modelresult.nfev})"
                "</span>"
            )

        def _get_param_text(param: str) -> str:
            """Get the string representation of a parameter value."""
            factor = (
                1e3 if (param == "resolution" and self.mev_check.isChecked()) else 1
            )
            if hasattr(modelresult, "uvars") and modelresult.uvars is not None:
                return f"{modelresult.uvars[param] * factor:P}"
            return f"{modelresult.params[param].value * factor:.4f}"

        for param, textedit in zip(
            ("temp", "center", "resolution"),
            (self.temp_val, self.center_val, self.res_val),
            strict=True,
        ):
            textedit.setText(_get_param_text(param))

        self.redchi_val.setText(f"{modelresult.redchi:.4f}")

    def connect_signals(self) -> None:
        self.x0_spin.valueChanged.connect(self._update_region)
        self.x1_spin.valueChanged.connect(self._update_region)
        self.y0_spin.valueChanged.connect(self._update_region)
        self.y1_spin.valueChanged.connect(self._update_region)
        self._x_region_changed()
        self._y_region_changed()  # Set spinbox initial values

        self.go_btn.clicked.connect(self.do_fit)
        self._sigTriggerFit.connect(self.do_fit)
        self.guess_temp_btn.clicked.connect(self._guess_temp)
        self.guess_center_btn.clicked.connect(self._guess_center)
        self.copy_btn.clicked.connect(self.copy_code)

    @QtCore.Slot()
    def copy_code(self) -> str:
        """Copy the code for the current fit to the clipboard."""
        data_name = erlab.interactive.utils.generate_code(
            xr.DataArray.sel,
            args=[],
            kwargs={self.y_dim: slice(*self.y_range)},
            module=self.data_name,
        )

        # Check if any parameters are the same as the automatically guessed ones
        params = dict(self.fit_params)
        params_guessed = self._guessed_params
        for k in ("temp", "center", "resolution"):
            if params[k] == params_guessed[k].value:
                del params[k]

        params["plot"] = True

        return erlab.interactive.utils.generate_code(
            erlab.analysis.gold.quick_fit,
            args=[f"|{data_name}|"],
            kwargs=params,
            module="era.gold",
            copy=True,
        )

    @QtCore.Slot()
    def _update_region(self) -> None:
        """Update the region items when the spinboxes are changed."""
        if self.x_range != self.x_region.getRegion():
            self.x_region.setRegion(self.x_range)
        if self.y_range != self.y_region.getRegion():
            self.y_region.setRegion(self.y_range)

    @QtCore.Slot()
    def _x_region_changed(self) -> None:
        """Update the x0 and x1 spinboxes when the region is changed.

        If live fitting is enabled, trigger a fit.
        """
        self.x0_spin.blockSignals(True)
        self.x1_spin.blockSignals(True)

        x0, x1 = self.x_region.getRegion()
        self.x0_spin.setValue(x0)
        self.x1_spin.setValue(x1)

        self.x0_spin.blockSignals(False)
        self.x1_spin.blockSignals(False)

        self.x0_spin.setMaximum(self.x1_spin.value())
        self.x1_spin.setMinimum(self.x0_spin.value())

        self.edc_fit.setData(x=[], y=[])
        if self.live_check.isChecked():
            self._sigTriggerFit.emit()

    @QtCore.Slot()
    def _y_region_changed(self) -> None:
        """Update the y0 and y1 spinboxes when the region is changed.

        Also updates the averaged EDC plot.
        If live fitting is enabled, trigger a fit.
        """
        self.y0_spin.blockSignals(True)
        self.y1_spin.blockSignals(True)

        y0, y1 = self.y_region.getRegion()
        self.y0_spin.setValue(y0)
        self.y1_spin.setValue(y1)

        self.y0_spin.blockSignals(False)
        self.y1_spin.blockSignals(False)

        self.y0_spin.setMaximum(self.y1_spin.value())
        self.y1_spin.setMinimum(self.y0_spin.value())

        self._update_edc()
        if self.live_check.isChecked():
            self._sigTriggerFit.emit()

    def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        return super().closeEvent(event)


def goldtool(
    data: xr.DataArray,
    data_corr: xr.DataArray | None = None,
    *,
    data_name: str | None = None,
    **kwargs,
) -> GoldTool:
    """Interactive tool for correcting curved Fermi edges.

    This tool can also be accessed from the right-click context menu of an image plot in
    an ImageTool window.

    Parameters
    ----------
    data
        The data to perform Fermi edge fitting on.
    data_corr
        The data to correct with the edge. Defaults to `data`.
    data_name
        Name of the data used in generating the code snipped copied to the clipboard.
        Overrides automatic detection.
    **kwargs
        Arguments passed onto `erlab.interactive.utils.AnalysisWindow`.
    """
    if data_name is None:
        try:
            data_name = str(varname.argname("data", func=goldtool, vars_only=False))
        except varname.VarnameRetrievingError:
            data_name = "data"
    return GoldTool(data, data_corr, data_name=data_name, **kwargs)


def restool(
    data: xr.DataArray, *, data_name: str | None = None, execute: bool | None = None
) -> ResolutionTool:
    """Interactive tool for precise resolution fitting of EDCs.

    This tool can also be accessed from the right-click context menu of an image plot in
    an ImageTool window.

    Parameters
    ----------
    data
        Data to visualize. Must be a 2D DataArray with an 'eV' dimension.
    data_name
        Name of the data variable in the generated code. If not provided, the name is
        automatically determined.
    """
    if data_name is None:
        try:
            data_name = str(varname.argname("data", func=restool, vars_only=False))
        except varname.VarnameRetrievingError:
            data_name = "data"

    with erlab.interactive.utils.setup_qapp(execute):
        win = ResolutionTool(data, data_name=data_name)
        win.show()
        win.raise_()
        win.activateWindow()

    return win
