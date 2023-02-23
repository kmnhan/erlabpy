import sys

import numba
import numpy as np
import pyqtgraph as pg
import xarray as xr
from PySide6 import QtCore, QtGui, QtWidgets

from erlab.plotting.interactive.utilities import ParameterGroup, FittingParameterWidget

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


# def gaussian(x, center=0, sigma=1, amplitude=1):
#     """Some constants are absorbed here into the amplitude factor."""
#     return amplitude * np.exp(-((x - center) ** 2) / (2 * sigma**2))


# def g(x, mu=0, sigma=0.1):
#     """TODO, unify this with the standard Gaussian definition because it's gross."""
#     return (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(
#         -(1 / 2) * ((x - mu) / sigma) ** 2
#     )


@numba.njit(cache=True)
def gaussian_wh(x, center, width, height):
    """Gaussian parametrized with FWHM and peak height"""
    # sigma = width / (2 * np.sqrt(2 * np.log(2)))
    return height * np.exp(-16 * np.log(2) * ((x - center) / width) ** 2)


@numba.njit(cache=True)
def lorentzian_wh(x, center, width, height):
    """Lorentzian parametrized with FWHM and peak height"""
    # sigma = width / 2
    return height / (1 + 4 * ((x - center) / width) ** 2)


@numba.njit(cache=True)
def gen_kernel(x, resolution, pad=5):
    delta_x = x[1] - x[0]
    n_pad = int(resolution * pad / delta_x)
    x_pad = n_pad * delta_x
    sigma = resolution / np.sqrt(8 * np.log(2))  # resolution given in FWHM

    extended = np.linspace(x[0] - x_pad, x[-1] + x_pad, int(2 * n_pad + len(x)))
    gauss = (
        delta_x
        * np.exp(-(np.linspace(-x_pad, x_pad, 2 * n_pad + 1) ** 2) / (2 * sigma**2))
        / np.sqrt(2 * np.pi * sigma**2)
    )

    return extended, gauss


def do_convolve(x, fcn, resolution):
    """Generates gaussian convolution of given function in given domain."""
    x, g = gen_kernel(x, resolution)
    return np.convolve(fcn(x), g, mode="valid")


# @numba.njit(cache=True)
def lorentzian(x, gamma, center, amplitude):
    """A straightforward Lorentzian."""
    return (
        amplitude * (1 / (2 * np.pi)) * gamma / ((x - center) ** 2 + (0.5 * gamma) ** 2)
    )


# @numba.njit(cache=True)
def fermi_dirac(x, center=0, width=0.05, scale=1):
    """Fermi edge, with somewhat arbitrary normalization."""
    return scale / (np.exp((x - center) / width) + 1)

class SinglePeakWidget(ParameterGroup):
    
    VALID_LINESHAPE = ["gaussian", "lorentzian"]
     
    def __init__(self, num_peak):
        super().__init__(**{
            "Peak Shape": dict(qwtype="combobox", items=self.VALID_LINESHAPE),
            "height": dict(
                qwtype="fitparam",
                showlabel="Intensity",
                name="height",
                spin_kw=dict(value=0.001, minimumWidth=100),
            ),
            "center": dict(
                qwtype="fitparam",
                showlabel="Position",
                name="position",
                spin_kw=dict(value=0.001, minimumWidth=100),
            ),
            "width": dict(
                qwtype="fitparam",
                showlabel="FWHM",
                name="width",
                spin_kw=dict(value=0.001, minimumWidth=100),
            ),
        }, groupbox_kw=dict(flat=True, objectName=f"PeakWidget{num_peak}"))
        
        self.setStyleSheet("QGroupBox#" + self.objectName() + " {border:0;}")
        for w in self.widgets_of_type(FittingParameterWidget):
            w.set_prefix(f"p{int(num_peak)}_")
    
    def peak_func(self):
        peak_args = self.values
        shape = peak_args.pop("Peak Shape")
        match shape:
            case "gaussian":
                return lambda x: gaussian_wh(x, **peak_args)
            case "lorentzian":
                return lambda x: lorentzian_wh(x, **peak_args)

class PlotPeakItem(pg.PlotCurveItem):
    def __init__(self, param_widget:SinglePeakWidget, *args, **kargs):
        self.param_widget = param_widget
        super().__init__(*args, **kargs)
        self.setClickable(True, 20)
        self._pen_color = self.opts['pen'].color()
        self._pen_width = self.opts['pen'].width()
        self.moving=False
        
    def setTempPen(self, *args, **kargs):
        self.opts['pen'] = pg.mkPen(*args, **kargs)
        self.invalidateBounds()
        self.update()
        
    def setPen(self, *args, **kargs):
        super().setPen(*args, **kargs)
        self._pen_color = self.opts['pen'].color()
        self._pen_width = self.opts['pen'].width()
    
    def mouseDragEvent(self, ev):

        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            if ev.isStart():
                self.moving=True
                # self.cursorOffset = self.pos() - self.mapToParent(ev.buttonDownPos())
                # self.startPosition = self.pos()
            ev.accept()
            if self.moving:
                self.setClickable(True, 100)
            else:
                self.setClickable(True, 20)
            if not self.mouseShape().contains(ev.pos()):
                ev.ignore()
            
            pos = self.mapToParent(ev.pos())
            # pos = self.cursorOffset + self.mapToParent(ev.pos())
            self.param_widget.widgets["center"].setValue(pos.x())
            self.param_widget.widgets["height"].setValue(pos.y())
            
            if ev.isFinish():
                self.moving = False
        
        elif ev.button() == QtCore.Qt.MouseButton.RightButton:
            if ev.isStart():
                self._start_width = self.param_widget.widgets["width"].value()
            ev.accept()
            x0, x1 = self.dataBounds(0)
            y0, y1 = self.dataBounds(1)
            val = self.mapToParent(ev.buttonDownPos() - ev.pos()).y()
            amount = 1.0 * (x1-x0) * val/(y1 - y0)
            # amount
            
            self.param_widget.widgets["width"].setValue(self._start_width + amount)
    
    def hoverEvent(self, ev):
        self.setMouseHover(False)
        if (not ev.isExit()) and ev.acceptDrags(QtCore.Qt.MouseButton.LeftButton):
            if not self.mouseShape().contains(ev.pos()):
                return
            self.setMouseHover(True)
        else:
            self.setMouseHover(False)
            
    def setMouseHover(self, hover):
        ## Inform the item that the mouse is (not) hovering over it
        # if self.mouseHovering == hover:
            # return
        # self.mouseHovering = hover
        if hover:
            self.setTempPen(self._pen_color, width=2*self._pen_width)
        else:
            self.setTempPen(self._pen_color, width=self._pen_width)

        self.update()

class edctool(QtWidgets.QMainWindow):
    def __init__(self, data, *args, **kwargs):
        self.data = data

        self.qapp = QtCore.QCoreApplication.instance()
        if not self.qapp:
            self.qapp = QtWidgets.QApplication(sys.argv)
        self.qapp.setStyle("Fusion")
        super().__init__()

        self._dock0 = QtWidgets.QDockWidget("Parameters", self)
        self._options = QtWidgets.QWidget(self)
        self._options_layout = QtWidgets.QVBoxLayout(self._options)
        self._params_init = ParameterGroup(
            **{
                "n_bands": dict(
                    showlabel="# Bands",
                    qwtype="btspin",
                    integer=True,
                    minimum=1,
                    fixedWidth=60,
                    notrack=True,
                    valueChanged=self.refresh_n_peaks,
                ),
                "const_bkg": dict(
                    qwtype="fitparam",
                    showlabel="Constant Background",
                    name="const_bkg",
                    spin_kw=dict(value=0.001, minimumWidth=100),
                ),
                "tot_res": dict(
                    qwtype="fitparam",
                    showlabel="Total Resolution",
                    name="tot_res",
                    spin_kw=dict(value=0.001, singleStep=0.0001, decimals=4, minimumWidth=100),
                ),
                "temp": dict(
                    qwtype="fitparam",
                    showlabel="Temperature",
                    name="temp",
                    fixed=True,
                    spin_kw=dict(value=30, minimum=0, minimumWidth=100),
                ),
                "efermi": dict(
                    qwtype="fitparam",
                    showlabel="Fermi Level",
                    name="efermi",
                    fixed=True,
                    spin_kw=dict(value=0, minimum=0, minimumWidth=100),
                ),
                # "Fix T": dict(qwtype="chkbox", checked=True),
                # "Bin x": dict(qwtype="spin", value=1, minimum=1),
                # "Bin y": dict(qwtype="spin", value=1, minimum=1),
                # "Method": dict(qwtype="combobox"),
                # "# CPU": dict(
                #     qwtype="spin",
                #     value=os.cpu_count(),
                #     minimum=1,
                #     maximum=os.cpu_count(),
                # ),
                # "go": dict(
                #     qwtype="pushbtn",
                #     showlabel=False,
                #     text="Go",
                #     clicked=self.perform_edge_fit,
                # ),
            }
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
        
        # self._params_peak.addTab(SinglePeakWidget(0), "Peak 0")
        
        self._options_layout.addWidget(self._params_peak)
        

        self._dock0.setWidget(self._options)
        self._dock0.setFloating(False)

        self.plotwidget = pg.PlotWidget(self)
        self.rawplot = self.plotwidget.plot()
        self.rawplot.setData(x=self.xdata, y=self.ydata)
        
        self.modelplot = self.plotwidget.plot()
        
        self.peakplots = []


        self.refresh_n_peaks()
        
        self.setCentralWidget(self.plotwidget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self._dock0)
        self.setWindowTitle("EDC Fitting")
        self.__post_init__()
    
    @property    
    def xdata(self):
        return self.data[self.data.dims[0]].values
    
    @property
    def ydata(self):
        return self.data.values
    
    @property
    def n_bands(self):
        return self._params_init.values["n_bands"]
    
    def refresh_n_peaks(self):
        current = int(self._params_peak.count())
        if self.n_bands > current:
            while self.n_bands > self._params_peak.count():
                self._params_peak.addTab(SinglePeakWidget(current), f"Peak {current}")
                self._params_peak.widget(current).sigParameterChanged.connect(self._refresh_plot_peaks)
                current += 1
        elif self.n_bands == current:
            return
        else:
            while self.n_bands < self._params_peak.count():
                current -= 1
                self._params_peak.removeTab(current)
        
        self._refresh_plot_peaks()
                
    def _refresh_plot_peaks(self):
        
        sum_peaks = np.zeros_like(self.xdata)
        
        for i in range(self.n_bands):
            param_widget = self._params_peak.widget(i)
            try:
                peak = self.peakplots[i]
            except IndexError:
                peak = PlotPeakItem(param_widget)
                self.plotwidget.addItem(peak)
                peak.setPen(pg.mkPen("r"))
                self.peakplots.append(peak)
            
            func = param_widget.peak_func()(self.xdata)
            peak.setData(x=self.xdata, y=func)
            sum_peaks += func
            
        self.modelplot.setData(x=self.xdata, y=sum_peaks)

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


if __name__ == "__main__":
    data = xr.open_dataarray(
        "~/Documents/ERLab/TiSe2/220922_ALS_BL4/TS2_testedc_2209ALS.nc"
    )
    edctool(data)
