__all__ = ["goldtool"]

import os
import time

import joblib
import numpy as np
import pyqtgraph as pg
import uncertainties
import varname
import xarray as xr
from qtpy import QtCore, QtWidgets

import arpes.xarray_extensions
import erlab.analysis
from erlab.interactive.imagetool import ImageTool
from erlab.interactive.utilities import (
    AnalysisWindow,
    ParameterGroup,
    ROIControls,
    gen_function_code,
)
from erlab.parallel import joblib_progress_qt

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

    def set_params(self, data, x0, y0, x1, y1, params):
        self.data = data
        self.x_range: tuple[float, float] = (x0, x1)
        self.y_range: tuple[float, float] = (y0, y1)
        self.params = params
        self.parallel_obj = joblib.Parallel(n_jobs=self.params["# CPU"])

    @QtCore.Slot()
    def abort_fit(self):
        self.parallel_obj._aborting = True
        self.parallel_obj._exception = True

    def run(self):
        self.sigIterated.emit(0)
        with joblib_progress_qt(self.sigIterated) as _:
            self.edge_center, self.edge_stderr = erlab.analysis.gold.edge(
                gold=self.data,
                angle_range=self.x_range,
                eV_range=self.y_range,
                bin_size=(self.params["Bin x"], self.params["Bin y"]),
                temp=self.params["T (K)"],
                vary_temp=not self.params["Fix T"],
                fast=self.params["Fast"],
                method=self.params["Method"],
                scale_covar=self.params["Scale cov"],
                progress=False,
                parallel_obj=self.parallel_obj,
            )
        self.sigFinished.emit()


class goldtool(AnalysisWindow):
    """Interactive gold edge fitting.

    Parameters
    ----------
    data
        The data to perform Fermi edge fitting on.
    data_corr
        The data to correct with the edge. Defaults to `data`.
    **kwargs
        Arguments passed onto `erlab.interactive.utilities.AnalysisWindow`.

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
        self, data: xr.DataArray, data_corr: xr.DataArray | None = None, **kwargs: dict
    ):
        super().__init__(
            data,
            link="x",
            layout="horizontal",
            orientation="vertical",
            num_ax=3,
            **kwargs,
        )
        self._argnames = dict()
        try:
            self._argnames["data"] = varname.argname(
                "data", func=self.__init__, vars_only=False
            )
        except varname.VarnameRetrievingError:
            self._argnames["data"] = "gold"

        if data_corr is not None:
            try:
                self._argnames["data_corr"] = varname.argname(
                    "data_corr", func=self.__init__, vars_only=False
                )
            except varname.VarnameRetrievingError:
                self._argnames["data_corr"] = "data_corr"

        self.data_corr = data_corr

        self.axes[1].setVisible(False)
        self.hists[1].setVisible(False)
        self.axes[2].setVisible(False)
        self.hists[2].setVisible(False)

        try:
            temp = data.S.temp
        except AttributeError:
            temp = 30

        self.params_roi = ROIControls(self.add_roi(0))
        self.params_edge = ParameterGroup(
            **{
                "T (K)": dict(qwtype="dblspin", value=temp, range=(0.0, 400.0)),
                "Fix T": dict(qwtype="chkbox", checked=True),
                "Bin x": dict(qwtype="spin", value=1, minimum=1),
                "Bin y": dict(qwtype="spin", value=1, minimum=1),
                "Fast": dict(qwtype="chkbox", checked=False),
                "Method": dict(qwtype="combobox", items=LMFIT_METHODS),
                "Scale cov": dict(qwtype="chkbox", checked=True),
                "# CPU": dict(
                    qwtype="spin",
                    value=os.cpu_count(),
                    minimum=1,
                    maximum=os.cpu_count(),
                ),
                "go": dict(
                    qwtype="pushbtn",
                    showlabel=False,
                    text="Go",
                    clicked=self.perform_edge_fit,
                ),
            }
        )

        self.params_edge.widgets["Fast"].stateChanged.connect(self._toggle_fast)

        self.params_poly = ParameterGroup(
            **{
                "Degree": dict(qwtype="spin", value=4, range=(1, 20)),
                "Method": dict(qwtype="combobox", items=LMFIT_METHODS),
                "Scale cov": dict(qwtype="chkbox", checked=True),
                "Residuals": dict(qwtype="chkbox", checked=False),
                "Corrected": dict(qwtype="chkbox", checked=False),
                "Shift coords": dict(qwtype="chkbox", checked=True),
                "itool": dict(
                    qwtype="pushbtn",
                    notrack=True,
                    showlabel=False,
                    text="Open in ImageTool",
                    clicked=self.open_itool,
                ),
                "copy": dict(
                    qwtype="pushbtn",
                    notrack=True,
                    showlabel=False,
                    text="Copy to clipboard",
                    clicked=lambda: self.gen_code("poly"),
                ),
            }
        )

        self.params_spl = ParameterGroup(
            **{
                "Auto": dict(qwtype="chkbox", checked=True),
                "lambda": dict(
                    qwtype="dblspin", minimum=0, singleStep=0.001, decimals=4
                ),
                "Residuals": dict(qwtype="chkbox", checked=False),
                "Corrected": dict(qwtype="chkbox", checked=False),
                "Shift coords": dict(qwtype="chkbox", checked=True),
                "itool": dict(
                    qwtype="pushbtn",
                    notrack=True,
                    showlabel=False,
                    text="Open in ImageTool",
                    clicked=self.open_itool,
                ),
                "copy": dict(
                    qwtype="pushbtn",
                    notrack=True,
                    showlabel=False,
                    text="Copy to clipboard",
                    clicked=lambda: self.gen_code("spl"),
                ),
            }
        )
        self.params_spl.widgets["lambda"].setDisabled(
            self.params_spl.widgets["Auto"].checkState() == QtCore.Qt.CheckState.Checked
        )
        self.params_spl.widgets["Auto"].toggled.connect(
            lambda _: self.params_spl.widgets["lambda"].setDisabled(
                self.params_spl.widgets["Auto"].checkState()
                == QtCore.Qt.CheckState.Checked
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

        # setup time calculation
        self.start_time: float | None = None
        self.step_times: list[float] = []

        # setup progress bar
        self.progress = QtWidgets.QProgressDialog(
            labelText="Fitting...",
            minimum=0,
            parent=self,
            minimumDuration=0,
            windowModality=QtCore.Qt.WindowModal,
        )
        self.pbar = QtWidgets.QProgressBar()
        self.progress.setBar(self.pbar)
        self.progress.setFixedSize(self.progress.size())
        self.progress.setCancelButtonText("Abort!")
        self.progress.canceled.disconnect(self.progress.cancel)  # don't auto close
        self.progress.canceled.connect(self.abort_fit)
        self.progress.setAutoReset(False)
        self.progress.cancel()

        # setup fitter
        self.fitter = EdgeFitter()
        self.fitter.sigIterated.connect(self.iterated)
        self.fitter.sigFinished.connect(self.post_fit)
        self.sigAbortFitting.connect(self.fitter.abort_fit)

        # resize roi to data bounds
        eV_span = self.data.eV.values[-1] - self.data.eV.values[0]
        ang_span = self.data.alpha.values[-1] - self.data.alpha.values[0]
        x1 = self.data.alpha.values.mean() + ang_span * 0.45
        x0 = self.data.alpha.values.mean() - ang_span * 0.45
        y1 = self.data.eV.values[-1] - eV_span * 0.015
        y0 = y1 - eV_span * 0.3
        self.params_roi.modify_roi(x0, y0, x1, y1)

        self.__post_init__(execute=True)

    def _toggle_fast(self):
        self.params_edge.widgets["T (K)"].setDisabled(self.params_edge.values["Fast"])
        self.params_edge.widgets["Fix T"].setDisabled(self.params_edge.values["Fast"])

    def iterated(self, n: int):
        self.step_times.append(time.perf_counter() - self.start_time)
        self.progress.setValue(n)

        deltas = np.diff(self.step_times)
        # avg_time = uncertainties.ufloat(np.mean(deltas), np.std(deltas))
        timeleft = (self.progress.maximum() - (n - 1)) * np.mean(deltas)

        # timeleft: str = humanize.precisedelta(datetime.timedelta(seconds=timeleft))
        # steptime: str = humanize.precisedelta(datetime.timedelta(seconds=steptime))

        self.progress.setLabelText(f"{timeleft:.1f} seconds left...")
        self.pbar.setFormat(f"{n}/{self.progress.maximum()} finished")

    @QtCore.Slot()
    def perform_edge_fit(self):
        self.start_time = time.perf_counter()
        self.step_times: list[float] = [0.0]

        self.progress.setVisible(True)
        self.params_roi.draw_button.setChecked(False)
        x0, y0, x1, y1 = self.params_roi.roi_limits
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
    def abort_fit(self):
        self.sigAbortFitting.emit()

    def closeEvent(self, event):
        super().closeEvent(event)

    @QtCore.Slot()
    def post_fit(self):
        self.progress.reset()
        self.edge_center, self.edge_stderr = (
            self.fitter.edge_center,
            self.fitter.edge_stderr,
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

    def perform_fit(self, mode="poly"):
        match mode:
            case "poly":
                modelresult = self._perform_poly_fit()
                params = self.params_poly.values
            case "spl":
                modelresult = self._perform_spline_fit()
                params = self.params_spl.values
        for i in range(2):
            xval = self.data.alpha.values
            if i == 1 and params["Residuals"]:
                yval = np.zeros_like(xval)
            else:
                yval = modelresult(xval)
            self.polycurves[i].setData(x=xval, y=yval)

        xval = self.edge_center.alpha.values
        if params["Residuals"]:
            yval = modelresult(xval) - self.edge_center.values
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
        modelresult = erlab.analysis.gold.poly_from_edge(
            center=self.edge_center,
            weights=1 / self.edge_stderr,
            degree=params["Degree"],
            method=params["Method"],
            scale_covar=params["Scale cov"],
        )
        if self.data_corr is None:
            target = self.data
        else:
            target = self.data_corr
        self.corrected = erlab.analysis.correct_with_edge(
            target, modelresult, plot=False, shift_coords=params["Shift coords"]
        )
        return lambda x: modelresult.eval(modelresult.params, x=x)

    def _perform_spline_fit(self):
        params = self.params_spl.values
        if params["Auto"]:
            params["lambda"] = None
        modelresult = erlab.analysis.gold.spline_from_edge(
            center=self.edge_center,
            weights=np.asarray(1 / self.edge_stderr),
            lam=params["lambda"],
        )

        if self.data_corr is None:
            target = self.data
        else:
            target = self.data_corr
        self.corrected = erlab.analysis.correct_with_edge(
            target, modelresult, plot=False, shift_coords=params["Shift coords"]
        )
        return modelresult

    def open_itool(self):
        self.itool = ImageTool(self.corrected)
        self.itool.show()

    def gen_code(self, mode: str):
        p0 = self.params_edge.values
        match mode:
            case "poly":
                p1 = self.params_poly.values
            case "spl":
                p1 = self.params_spl.values
        x0, y0, x1, y1 = self.params_roi.roi_limits

        arg_dict = dict(
            angle_range=(x0, x1),
            eV_range=(y0, y1),
            bin_size=(p0["Bin x"], p0["Bin y"]),
            temp=p0["T (K)"],
            method=p0["Method"],
        )

        match mode:
            case "poly":
                fname = "poly"
                arg_dict["degree"] = p1["Degree"]
            case "spl":
                fname = "spline"
                if p1["Auto"]:
                    arg_dict["lam"] = None
                else:
                    arg_dict["lam"] = p1["lambda"]

        if p0["Fast"]:
            arg_dict["fast"] = True
            del arg_dict["temp"]
        else:
            if not p0["Fix T"]:
                arg_dict["vary_temp"] = True

        if not p0["Scale cov"]:
            arg_dict["scale_covar_edge"] = False

        if mode == "poly":
            if not p1["Scale cov"]:
                arg_dict["scale_covar"] = False

        if self.data_corr is None:
            gen_function_code(
                copy=True,
                **{
                    f"modelresult = era.gold.{fname}": [
                        f"|{self._argnames['data']}|",
                        arg_dict,
                    ]
                },
            )
        else:
            arg_dict["correct"] = True
            gen_function_code(
                copy=True,
                **{
                    f"modelresult = era.gold.{fname}": [
                        f"|{self._argnames['data']}|",
                        arg_dict,
                    ],
                    "corrected = era.correct_with_edge": [
                        f"|{self._argnames['data_corr']}|",
                        "|modelresult|",
                        dict(shift_coords=p1["Shift coords"]),
                    ],
                },
            )


if __name__ == "__main__":
    import erlab.io

    dt = goldtool(
        erlab.io.load_als_bl4(
            "/Users/khan/Documents/ERLab/TiSe2/220630_ALS_BL4/data/csvsb2_gold.pxt"
        )
    )