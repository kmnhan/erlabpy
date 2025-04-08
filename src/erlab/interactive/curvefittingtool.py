import copy
import sys
import typing

import lmfit
import pyqtgraph as pg
from qtpy import QtCore, QtWidgets

from erlab.analysis.fit.models import MultiPeakModel
from erlab.interactive.utils import FittingParameterWidget, ParameterGroup

# EDCmultiFitting Igor procedure - 2D wave EDC fit along momentum range

# Momentum range to fit
# Number of bands
# Initial parameters
#   const_bkg
#   shirley_bkg
#   total resolution (fixed?)
#   Temperature (fixed?)
#   Fermi level (fixed?)
#
#   For each band
#       Lor or Gauss
#       Peak intensity
#       Peak position
#       Peak width

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


class SinglePeakWidget(ParameterGroup):
    VALID_LINESHAPE: tuple[str, ...] = ("lorentzian", "gaussian")

    def __init__(self, peak_index) -> None:
        self.peak_index = peak_index
        super().__init__(
            **{
                "Peak Shape": {"qwtype": "combobox", "items": self.VALID_LINESHAPE},
                "height": {
                    "qwtype": "fitparam",
                    "showlabel": "Intensity",
                    "name": "height",
                    "spin_kw": {"value": 0.001, "minimumWidth": 100},
                },
                "center": {
                    "qwtype": "fitparam",
                    "showlabel": "Position",
                    "name": "center",
                    "spin_kw": {"value": 0.001, "minimumWidth": 100},
                },
                "width": {
                    "qwtype": "fitparam",
                    "showlabel": "FWHM",
                    "name": "width",
                    "spin_kw": {"value": 0.03, "minimumWidth": 100},
                },
            },
            groupbox_kw={"flat": True, "objectName": f"PeakWidget{self.peak_index}"},
        )

        self.setStyleSheet("QGroupBox#" + self.objectName() + " {border:0;}")
        for w in self.widgets_of_type(FittingParameterWidget):
            w.set_prefix(f"p{int(self.peak_index)}_")

    @property
    def param_dict(self):
        pd = {}
        for w in self.widgets_of_type(FittingParameterWidget):
            pd = pd | w.param_dict
        return pd

    @property
    def peak_shape(self) -> str:
        return str(self.values["Peak Shape"])


class PlotPeakItem(pg.PlotCurveItem):
    def __init__(self, param_widget: SinglePeakWidget, *args, **kargs) -> None:
        self.param_widget = param_widget
        super().__init__(*args, **kargs)
        self._pen_color = self.opts["pen"].color()
        self._pen_width = self.opts["pen"].width()

    def setTempPen(self, *args, **kargs) -> None:
        self.opts["pen"] = pg.mkPen(*args, **kargs)
        self.invalidateBounds()
        self.update()

    def setPen(self, *args, **kargs) -> None:
        super().setPen(*args, **kargs)
        self._pen_color = self.opts["pen"].color()
        self._pen_width = self.opts["pen"].width()

    def viewRangeChanged(self) -> None:
        super().viewRangeChanged()
        self._mouseShape = None

    def setMouseHover(self, hover) -> None:
        # Inform the item that the mouse is (not) hovering over it
        # if self.mouseHovering == hover:
        # return
        # self.mouseHovering = hover
        if hover:
            self.setTempPen(self._pen_color, width=2 * self._pen_width)
        else:
            self.setTempPen(self._pen_color, width=self._pen_width)

        self.update()


class PlotPeakPosition(pg.InfiniteLine):
    def __init__(
        self, param_widget: SinglePeakWidget, curve: PlotPeakItem, *args, **kargs
    ) -> None:
        self.param_widget = param_widget
        self.curve = curve
        super().__init__(*args, movable=True, **kargs)

        self.addMarker("o", -0.5)

    def boundingRect(self):
        return super().boundingRect()

    def mouseDragEvent(self, ev) -> None:
        if not self.movable:
            ev.ignore()
            return
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            if ev.isStart():
                self.moving = True
                self.cursorOffset = self.pos() - self.mapToParent(ev.buttonDownPos())
                self.startPosition = self.pos()
                self._start_height = self.param_widget.widgets["height"].value()

            ev.accept()

            if not self.moving:
                return

            pos = self.cursorOffset + self.mapToParent(ev.pos())

            self.param_widget.widgets["center"].setValue(pos.x())
            self.param_widget.widgets["height"].setValue(
                self._start_height + self.mapToParent(ev.pos() - ev.buttonDownPos()).y()
            )

        elif QtCore.Qt.MouseButton.RightButton in ev.buttons():
            if ev.isStart():
                self._start_width = self.param_widget.widgets["width"].value()
            ev.accept()

            val = self.mapToParent(ev.buttonDownPos() - ev.pos()).y()
            y0, _, y1, _ = self.boundingRect().getCoords()
            amount = val / abs(y1 - y0)
            self.param_widget.widgets["width"].setValue(self._start_width + amount)
        if ev.isFinish():
            self.moving = False
            self.sigPositionChangeFinished.emit(self)

        self.setMouseHover(self.moving)

    def refresh_pos(self) -> None:
        self.setPos(self.param_widget.widgets["center"].value())

    def setMouseHover(self, hover):
        self.param_widget.parent().parent().setCurrentWidget(self.param_widget)
        self.curve.setMouseHover(hover)
        return super().setMouseHover(hover)


class edctool(QtWidgets.QMainWindow):
    def __init__(
        self, data, n_bands: int = 1, parameters=None, execute: bool = True
    ) -> None:
        self.data = data

        self.qapp = QtWidgets.QApplication.instance()
        if not self.qapp:
            self.qapp = QtWidgets.QApplication(sys.argv)
        typing.cast("QtWidgets.QApplication", self.qapp).setStyle("Fusion")
        super().__init__()
        self.resize(720, 360)

        self._dock0 = QtWidgets.QDockWidget("Parameters", self)
        self._options = QtWidgets.QWidget(self)
        self._options_layout = QtWidgets.QVBoxLayout(self._options)
        self._params_init = ParameterGroup(
            n_bands={
                "showlabel": "# Bands",
                "qwtype": "btspin",
                "integer": True,
                "value": n_bands,
                "minimum": 1,
                "fixedWidth": 60,
                "notrack": True,
                "valueChanged": self.refresh_n_peaks,
            },
            lin_bkg={
                "qwtype": "fitparam",
                "showlabel": "Linear Background",
                "name": "lin_bkg",
                "spin_kw": {"value": 0.0, "minimumWidth": 200},
            },
            const_bkg={
                "qwtype": "fitparam",
                "showlabel": "Constant Background",
                "name": "const_bkg",
                "spin_kw": {"value": 0.0, "minimumWidth": 200},
            },
            offset={
                "qwtype": "fitparam",
                "showlabel": "Offset",
                "name": "offset",
                "spin_kw": {"value": 0.0, "minimumWidth": 200},
            },
            resolution={
                "qwtype": "fitparam",
                "showlabel": "Total Resolution",
                "name": "resolution",
                "spin_kw": {
                    "value": 0.01,
                    "singleStep": 0.0001,
                    "decimals": 4,
                    "minimumWidth": 200,
                },
            },
            temp={
                "qwtype": "fitparam",
                "showlabel": "Temperature",
                "name": "temp",
                "fixed": True,
                "spin_kw": {"value": 30, "minimum": 0, "minimumWidth": 200},
            },
            efermi={
                "qwtype": "fitparam",
                "showlabel": "Fermi Level",
                "name": "efermi",
                "fixed": True,
                "spin_kw": {"value": 0, "minimumWidth": 200},
            },
            Method={"qwtype": "combobox", "items": LMFIT_METHODS},
            go={
                "qwtype": "pushbtn",
                "showlabel": False,
                "text": "Go",
                "clicked": self.do_fit,
            },
        )

        # label_width = 0
        # for w in self._params_init.widgets_of_type(FittingParameterWidget):
        #     label_width = max(
        #         label_width,
        #         w.label.fontMetrics().boundingRect(w.label.text()).width() + 5,
        #     )
        # for w in self._params_init.widgets_of_type(FittingParameterWidget):
        #     w.label.setFixedWidth(label_width)

        self._options_layout.addWidget(self._params_init)

        self._params_peak = QtWidgets.QTabWidget()

        self._options_layout.addWidget(self._params_peak)

        self._dock0.setWidget(self._options)
        self._dock0.setFloating(False)

        self.plotwidget = pg.PlotWidget(self)
        self.rawplot = self.plotwidget.plot()
        self.rawplot.setData(x=self.xdata, y=self.ydata)

        self.modelplot = self.plotwidget.plot()
        self.modelplot.setPen(pg.mkPen("y"))

        self.fitplot = self.plotwidget.plot()
        self.fitplot.setPen(pg.mkPen("c"))

        self.peakcurves: list[PlotPeakItem] = []
        self.peaklines: list[PlotPeakPosition] = []

        self.refresh_n_peaks()

        self._params_init.sigParameterChanged.connect(self._refresh_plot_peaks)

        self.setCentralWidget(self.plotwidget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self._dock0)
        self.setWindowTitle("EDC Fitting")

        if parameters is not None:
            self.set_params(parameters)
        self.__post_init__(execute=execute)

    @property
    def xdata(self):
        return self.data[self.data.dims[0]].values

    @property
    def ydata(self):
        return self.data.values

    @property
    def n_bands(self):
        return self._params_init.values["n_bands"]

    @property
    def params_dict(self):
        out = {}
        for k in ("efermi", "temp", "lin_bkg", "const_bkg", "offset", "resolution"):
            out = out | self._params_init.widgets[k].param_dict
        for i in range(self.n_bands):
            peak_widget = self._params_peak.widget(i)
            out = out | peak_widget.param_dict
        return out

    @property
    def params(self):
        return {k: v["value"] for k, v in self.params_dict.items()}

    @property
    def model(self):
        return MultiPeakModel(
            self.n_bands,
            peak_shapes=[
                self._params_peak.widget(i).peak_shape for i in range(self.n_bands)
            ],
        )

    def refresh_n_peaks(self) -> None:
        if not hasattr(self, "_params_peak"):
            return
        current = int(self._params_peak.count())
        if self.n_bands > current:
            while self.n_bands > self._params_peak.count():
                self._params_peak.addTab(SinglePeakWidget(current), f"Peak {current}")
                self._params_peak.widget(current).sigParameterChanged.connect(
                    self._refresh_plot_peaks
                )
                current += 1
        elif self.n_bands == current:
            return
        else:
            while self.n_bands < self._params_peak.count():
                current -= 1
                self._params_peak.removeTab(current)

        self._refresh_plot_peaks()

    def _refresh_plot_peaks(self) -> None:
        model = self.model
        params = self.params
        for i in range(self.n_bands):
            peak_widget = self._params_peak.widget(i)
            try:
                curve = self.peakcurves[i]
                line = self.peaklines[i]

            except IndexError:
                curve = PlotPeakItem(peak_widget)
                self.plotwidget.addItem(curve)
                curve.setPen(pg.mkPen("r"))
                self.peakcurves.append(curve)

                line = PlotPeakPosition(peak_widget, curve)
                self.plotwidget.addItem(line)
                line.setPen(pg.mkPen("r"))
                line.setHoverPen(pg.mkPen("r", width=2))
                self.peaklines.append(line)

            curve.setData(x=self.xdata, y=model.func.eval_peak(i, self.xdata, **params))
            line.refresh_pos()

        self.modelplot.setData(x=self.xdata, y=model.eval(x=self.xdata, **params))

    def do_fit(self) -> None:
        params = lmfit.create_params(**self.params_dict)
        model = self.model
        params = model.guess(self.data, self.data[self.data.dims[0]]).update(params)
        res = self.model.fit(
            self.ydata,
            x=self.xdata,
            params=params,
            method=self._params_init.values["Method"],
        )
        print(res.best_values)
        self.fitplot.setData(x=self.xdata, y=res.best_fit)
        self.set_params(res.best_values)

        self.result = res

    def set_params(self, params: dict) -> None:
        params = copy.deepcopy(params)
        self._params_init.set_values(
            **{
                k: params[k]
                for k in (
                    "efermi",
                    "temp",
                    "lin_bkg",
                    "const_bkg",
                    "offset",
                    "resolution",
                )
            }
        )
        for i in range(self.n_bands):
            self._params_peak.widget(i).set_values(  # type: ignore[union-attr]
                **{k[3:]: v for k, v in params.items() if k.startswith(f"p{i}")}
            )

    def __post_init__(self, execute=None):
        self.show()
        self.activateWindow()
        # self.raise_()

        if execute is None:
            execute = True
            try:
                shell = get_ipython().__class__.__name__  # type: ignore
                if shell in ["ZMQInteractiveShell", "TerminalInteractiveShell"]:
                    execute = False
            except NameError:
                pass
        if execute:
            self.qapp.exec()


class mdctool(QtWidgets.QMainWindow):
    def __init__(
        self, data, n_bands: int = 1, parameters=None, execute: bool = True
    ) -> None:
        self.data = data

        self.qapp = QtWidgets.QApplication.instance()
        if not self.qapp:
            self.qapp = QtWidgets.QApplication(sys.argv)
        typing.cast("QtWidgets.QApplication", self.qapp).setStyle("Fusion")
        super().__init__()
        self.resize(720, 360)

        self._dock0 = QtWidgets.QDockWidget("Parameters", self)
        self._options = QtWidgets.QWidget(self)
        self._options_layout = QtWidgets.QVBoxLayout(self._options)
        self._params_init = ParameterGroup(
            n_bands={
                "showlabel": "# Bands",
                "qwtype": "btspin",
                "integer": True,
                "value": n_bands,
                "minimum": 1,
                "fixedWidth": 60,
                "notrack": True,
                "valueChanged": self.refresh_n_peaks,
            },
            lin_bkg={
                "qwtype": "fitparam",
                "showlabel": "Linear Background",
                "name": "lin_bkg",
                "spin_kw": {"value": 0.0, "minimumWidth": 200},
            },
            const_bkg={
                "qwtype": "fitparam",
                "showlabel": "Constant Background",
                "name": "const_bkg",
                "spin_kw": {"value": 0.0, "minimumWidth": 200},
            },
            resolution={
                "qwtype": "fitparam",
                "showlabel": "Total Resolution",
                "name": "resolution",
                "spin_kw": {
                    "value": 0.01,
                    "singleStep": 0.0001,
                    "decimals": 4,
                    "minimumWidth": 200,
                },
            },
            Method={"qwtype": "combobox", "items": LMFIT_METHODS},
            go={
                "qwtype": "pushbtn",
                "showlabel": False,
                "text": "Go",
                "clicked": self.do_fit,
            },
        )

        # label_width = 0
        # for w in self._params_init.widgets_of_type(FittingParameterWidget):
        #     label_width = max(
        #         label_width,
        #         w.label.fontMetrics().boundingRect(w.label.text()).width() + 5,
        #     )
        # for w in self._params_init.widgets_of_type(FittingParameterWidget):
        #     w.label.setFixedWidth(label_width)

        self._options_layout.addWidget(self._params_init)

        self._params_peak = QtWidgets.QTabWidget()

        self._options_layout.addWidget(self._params_peak)

        self._dock0.setWidget(self._options)
        self._dock0.setFloating(False)

        self.plotwidget = pg.PlotWidget(self)
        self.rawplot = self.plotwidget.plot()
        self.rawplot.setData(x=self.xdata, y=self.ydata)

        self.modelplot = self.plotwidget.plot()
        self.modelplot.setPen(pg.mkPen("y"))

        self.fitplot = self.plotwidget.plot()
        self.fitplot.setPen(pg.mkPen("c"))

        self.peakcurves: list[PlotPeakItem] = []
        self.peaklines: list[PlotPeakPosition] = []

        self.refresh_n_peaks()

        self._params_init.sigParameterChanged.connect(self._refresh_plot_peaks)

        self.setCentralWidget(self.plotwidget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self._dock0)
        self.setWindowTitle("MDC Fitting")

        if parameters is not None:
            self.set_params(parameters)
        self.__post_init__(execute=execute)

    @property
    def xdata(self):
        return self.data[self.data.dims[0]].values

    @property
    def ydata(self):
        return self.data.values

    @property
    def n_bands(self):
        return self._params_init.values["n_bands"]

    @property
    def params_dict(self):
        out = {}
        for k in ("lin_bkg", "const_bkg", "resolution"):
            out = out | self._params_init.widgets[k].param_dict
        for i in range(self.n_bands):
            peak_widget = self._params_peak.widget(i)
            out = out | peak_widget.param_dict
        return out

    @property
    def params(self):
        return {k: v["value"] for k, v in self.params_dict.items()}

    @property
    def model(self):
        return MultiPeakModel(
            self.n_bands,
            peak_shapes=[
                self._params_peak.widget(i).peak_shape for i in range(self.n_bands)
            ],
            fd=False,
        )

    def refresh_n_peaks(self) -> None:
        if not hasattr(self, "_params_peak"):
            return
        current = int(self._params_peak.count())
        if self.n_bands > current:
            while self.n_bands > self._params_peak.count():
                self._params_peak.addTab(SinglePeakWidget(current), f"Peak {current}")
                self._params_peak.widget(current).sigParameterChanged.connect(
                    self._refresh_plot_peaks
                )
                current += 1
        elif self.n_bands == current:
            return
        else:
            while self.n_bands < self._params_peak.count():
                current -= 1
                self._params_peak.removeTab(current)

        self._refresh_plot_peaks()

    def _refresh_plot_peaks(self) -> None:
        model = self.model
        params = self.params
        for i in range(self.n_bands):
            peak_widget = self._params_peak.widget(i)
            try:
                curve = self.peakcurves[i]
                line = self.peaklines[i]

            except IndexError:
                curve = PlotPeakItem(peak_widget)
                self.plotwidget.addItem(curve)
                curve.setPen(pg.mkPen("r"))
                self.peakcurves.append(curve)

                line = PlotPeakPosition(peak_widget, curve)
                self.plotwidget.addItem(line)
                line.setPen(pg.mkPen("r"))
                line.setHoverPen(pg.mkPen("r", width=2))
                self.peaklines.append(line)

            curve.setData(x=self.xdata, y=model.func.eval_peak(i, self.xdata, **params))
            line.refresh_pos()

        self.modelplot.setData(x=self.xdata, y=model.eval(x=self.xdata, **params))

    def do_fit(self) -> None:
        params = lmfit.create_params(**self.params_dict)
        model = self.model
        params = model.guess(self.data, self.data[self.data.dims[0]]).update(params)
        res = self.model.fit(
            self.ydata,
            x=self.xdata,
            params=params,
            method=self._params_init.values["Method"],
        )
        print(res.best_values)
        self.fitplot.setData(x=self.xdata, y=res.best_fit)
        self.set_params(res.best_values)

        self.result = res

    def set_params(self, params: dict) -> None:
        params = copy.deepcopy(params)
        self._params_init.set_values(
            **{k: params[k] for k in ("lin_bkg", "const_bkg", "resolution")}
        )
        for i in range(self.n_bands):
            self._params_peak.widget(i).set_values(  # type: ignore[union-attr]
                **{k[3:]: v for k, v in params.items() if k.startswith(f"p{i}")}
            )

    def __post_init__(self, execute=None):
        self.show()
        self.activateWindow()
        # self.raise_()

        if execute is None:
            execute = True
            try:
                shell = get_ipython().__class__.__name__  # type: ignore
                if shell in ["ZMQInteractiveShell", "TerminalInteractiveShell"]:
                    execute = False
            except NameError:
                pass
        if execute:
            self.qapp.exec()
