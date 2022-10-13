import sys

import arpes.xarray_extensions
import numpy as np
import os
import varname
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

# if __name__ != "__main__":
#     from ..analysis.gold import gold_edge, gold_poly_from_edge
#     from ..analysis.utilities import correct_with_edge
#     from ..parallel import joblib_qt, joblib_pg
#     from .imagetool import ImageTool
#     from .interactive.utilities import (
#         AnalysisWindow,
#         ParameterGroup,
#         ROIControls,
#         gen_function_code,
#     )
# else:
import erlab.analysis
from erlab.parallel import joblib_qt, joblib_pg
from erlab.plotting.interactive.imagetool import ImageTool
from erlab.plotting.interactive.utilities import (
    AnalysisWindow,
    ParameterGroup,
    ROIControls,
    gen_function_code,
)


__all__ = ["goldtool"]

LMFIT_METHODS = [
    "leastsq",
    "least_squares",
    "differential_evolution",
    "brute",
    "basinhopping",
    "ampgo",
    "nelder",
    "lbfgsb",
    "powell",
    "cg",
    "newton",
    "cobyla",
    "bfgs",
    "tnc",
    "trust-ncg",
    "trust-exact",
    "trust-krylov",
    "trust-constr",
    "dogleg",
    "slsqp",
    "emcee",
    "shgo",
    "dual_annealing",
]


class goldtool(AnalysisWindow):

    sigProgressUpdated = QtCore.Signal(int)

    def __init__(self, data, data_corr=None, *args, **kwargs):
        super().__init__(
            data,
            *args,
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
                "T (K)": dict(qwtype="dblspin", value=temp, range=(0, 400)),
                "Fix T": dict(qwtype="chkbox", checked=True),
                "Bin x": dict(qwtype="spin", value=1, minimum=1),
                "Bin y": dict(qwtype="spin", value=1, minimum=1),
                "Method": dict(qwtype="combobox", items=LMFIT_METHODS),
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

        self.params_poly = ParameterGroup(
            **{
                "Degree": dict(qwtype="spin", value=4, range=(1, 9)),
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
                    clicked=self.gen_code,
                ),
            }
        )

        self.controls.addWidget(self.params_roi)
        self.controls.addWidget(self.params_edge)
        self.controls.addWidget(self.params_poly)

        self.params_poly.setDisabled(True)

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
        self.params_poly.sigParameterChanged.connect(self.perform_function_fit)

        # self.progress = QtWidgets.QProgressDialog("Performing fits", None, 0, 10, parent=self)
        # self.progress.setWindowModality(QtCore.Qt.WindowModal)
        # self.sigProgressUpdated.connect(self.progress.setValue)
        # self.progress.setAutoClose(True)
        # self.progress.reset()
        # self.progress.setValue(0)
        # self.axes[0].disableAutoRange(axis=self.axes[0].vb.YAxis)
        self.axes[0].disableAutoRange()
        self.__post_init__(execute=True)

    def perform_edge_fit(self):
        self.params_roi.draw_button.setChecked(False)
        # self.proxy = pg.SignalProxy(
        #     self.sigProgressUpdated,
        #     rateLimit=60,
        #     slot=lambda x: self.progress.setValue(x[0]),
        # )
        x0, y0, x1, y1 = self.params_roi.roi_limits
        # self.progress.setMaximum(len(self.data.phi.sel(phi=slice(x0, x1))))

        params = self.params_edge.values

        # self.progress = joblib_pg("Fitting...", 0, len(self.data.phi.coarsen(phi=params["Bin x"],boundary="trim").mean().sel(phi=slice(x0, x1))), None)
        # def pg_print_progress(self):
        #     if self.n_completed_tasks > self.progress.value():
        #         print(self.n_completed_tasks)
        #         self.progress.sigProgressUpdated.emit(self.n_completed_tasks)
        # with joblib_pg(
        #     "Fitting...",
        #     0,
        #     len(
        #         self.data.phi.coarsen(phi=params["Bin x"], boundary="trim")
        #         .mean()
        #         .sel(phi=slice(x0, x1))
        #     ),
        #     None,
        # ) as self.asdfasfd:
        self.edge_center, self.edge_stderr = erlab.analysis.gold_edge(
            gold=self.data,
            phi_range=(x0, x1),
            eV_range=(y0, y1),
            bin_size=(params["Bin x"], params["Bin y"]),
            method=params["Method"],
            progress=True,
            parallel_kw=dict(n_jobs=params["# CPU"]),
        )

        xval = self.edge_center.phi.values
        yval = self.edge_center.values
        for i in range(2):
            self.scatterplots[i].setData(x=xval, y=yval)
            self.errorbars[i].setData(x=xval, y=yval, height=self.edge_stderr.values)

        self.params_poly.setDisabled(False)
        self.perform_function_fit()

    def perform_function_fit(self):
        params = self.params_poly.values
        modelresult = erlab.analysis.gold_poly_from_edge(
            center=self.edge_center,
            weights=1 / self.edge_stderr,
            degree=params["Degree"],
            method=self.params_edge.values["Method"],
        )
        if self.data_corr is None:
            target = self.data
        else:
            target = self.data_corr
        self.corrected = erlab.analysis.correct_with_edge(
            target, modelresult, plot=False, shift_coords=params["Shift coords"]
        )

        for i in range(2):
            xval = self.data.phi.values
            if i == 1 and params["Residuals"]:
                yval = np.zeros_like(xval)
            else:
                yval = modelresult.eval(modelresult.params, x=xval)
            self.polycurves[i].setData(x=xval, y=yval)

        xval = self.edge_center.phi.values
        if params["Residuals"]:
            yval = modelresult.eval() - modelresult.data
        else:
            yval = self.edge_center.values
        self.errorbars[1].setData(x=xval, y=yval)
        self.scatterplots[1].setData(x=xval, y=yval, height=self.edge_stderr)

        self.aw.axes[1].setVisible(True)
        self.aw.images[-1].setDataArray(self.corrected)
        self.aw.axes[2].setVisible(params["Corrected"])
        self.aw.hists[2].setVisible(params["Corrected"])

    def open_itool(self):
        self.itool = ImageTool(self.corrected)
        self.itool.show()

    def gen_code(self):
        p0 = self.params_edge.values
        p1 = self.params_poly.values
        x0, y0, x1, y1 = self.params_roi.roi_limits

        if self.data_corr is None:
            gen_function_code(
                copy=True,
                **{
                    "modelresult = era.gold_poly": [
                        f"|{self._argnames['data']}|",
                        dict(
                            phi_range=(x0, x1),
                            eV_range=(y0, y1),
                            bin_size=(p0["Bin x"], p0["Bin y"]),
                            degree=p1["Degree"],
                            method=p0["Method"],
                        ),
                    ]
                },
            )
        else:
            gen_function_code(
                copy=True,
                **{
                    "modelresult = era.gold_poly": [
                        f"|{self._argnames['data']}|",
                        dict(
                            phi_range=(x0, x1),
                            eV_range=(y0, y1),
                            bin_size=(p0["Bin x"], p0["Bin y"]),
                            degree=p1["Degree"],
                            method=p0["Method"],
                            correct=False,
                        ),
                    ],
                    "corrected = era.correct_with_edge": [
                        f"|{self._argnames['data_corr']}|",
                        "|modelresult|",
                        dict(shift_coords=p1["Shift coords"]),
                    ],
                },
            )


if __name__ == "__main__":
    import arpes.io

    # qapp = QtWidgets.QApplication.instance()
    # if not qapp:
    # qapp = QtWidgets.QApplication(sys.argv)
    # qapp.setStyle("Fusion")
    dat = arpes.io.load_data(
        "/Users/khan/Documents/ERLab/TiSe2/220630_ALS_BL4/data/csvsb2_gold.pxt",
        location="BL4",
    )
    dt = goldtool(dat, dat)
    # dt.show()
    # dt.activateWindow()
    # dt.raise_()
    # qapp.exec()
