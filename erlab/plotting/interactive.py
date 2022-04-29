import numpy as np
import sys
import pyqtgraph as pg
import xarray as xr
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from superqt import QDoubleSlider

__all__ = ["parse_data", "xImageItem", "ParameterGroup", "AnalysisWidgetBase"]


def parse_data(data):
    if isinstance(data, xr.Dataset):
        try:
            data = data.spectrum
        except:
            raise TypeError(
                "input argument data must be a xarray.DataArray or a "
                "numpy.ndarray. Create an xarray.DataArray "
                "first, either with indexing on the Dataset or by "
                "invoking the `to_array()` method."
            ) from None
    elif isinstance(data, np.ndarray):
        data = xr.DataArray(data)
    return data


def array_rect(data):
    data_coords = tuple(data[dim].values for dim in data.dims)
    data_incs = tuple(coord[1] - coord[0] for coord in data_coords)
    data_lims = tuple((coord[0], coord[-1]) for coord in data_coords)
    y, x = data_lims[0][0] - data_incs[0], data_lims[1][0] - data_incs[1]
    h, w = data_lims[0][-1] - y, data_lims[1][-1] - x
    y += 0.5 * data_incs[0]
    x += 0.5 * data_incs[1]
    return QtCore.QRectF(x, y, w, h)


class xImageItem(pg.ImageItem):

    sigToleranceChanged = QtCore.Signal(float, float)

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.cut_tolerance = (30, 30)
        self.data_array = None

    def set_cut_tolerance(self, cut_tolerance):
        try:
            self.cut_tolerance = list(cut_tolerance.__iter__)
        except AttributeError:
            self.cut_tolerance = [cut_tolerance] * 2
        self.setImage(levels=self.data_cut_levels())

    def data_cut_levels(self, data=None):
        """Return appropriate levels estimated from the data."""
        if data is None:
            data = self.image
            if data is None:
                return (np.nan, np.nan)
        data = np.nan_to_num(data)
        q3, q1, pu, pl = np.percentile(
            data, [75, 25, 100 - self.cut_tolerance[0], self.cut_tolerance[1]]
        )
        ql, qu = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        self.sigToleranceChanged.emit(
            100 * (data > qu).mean(), 100 * (data < ql).mean()
        )
        mn, mx = max(min(pl, ql), data.min()), min(max(pu, qu), data.max())
        return (mn, mx)

    def setImage(self, image=None, autoLevels=None, cut_to_data=False, **kargs):
        if cut_to_data:
            kargs["levels"] = self.data_cut_levels(data=image)
        super().setImage(image=image, autoLevels=autoLevels, **kargs)

    def setDataArray(self, data=None, **kargs):
        self.data_array = parse_data(data)
        rect = array_rect(self.data_array)
        if self.axisOrder == "row-major":
            img = np.ascontiguousarray(self.data_array.values)
        else:
            img = np.asfortranarray(self.data_array.values.T)
        self.setImage(img, rect=rect, **kargs)


valid_qwtype = {
    "spin": QtWidgets.QSpinBox,
    "dblspin": QtWidgets.QDoubleSpinBox,
    "slider": QtWidgets.QSlider,
    "dblslider": QDoubleSlider,
    "chkbox": QtWidgets.QCheckBox,
    "pushbtn": QtWidgets.QPushButton,
    "chkpushbtn": QtWidgets.QPushButton,
    "combobox": QtWidgets.QComboBox,
}


def getParameterWidget(qwtype, **kwargs):
    """

    Parameters
    ----------
    qwtype : {'spin', 'dblspin', 'slider', 'dblslider', 'chkbox', 'pushbtn', 'chkpushbtn', 'combobox'}

    """

    if qwtype not in valid_qwtype.keys():
        raise ValueError(f"qwtype must be one of {list(valid_qwtype.keys())}")

    widget_class = valid_qwtype[qwtype]
    if qwtype == "combobox":
        items = kwargs.pop("items", None)
    elif qwtype == "chkpushbtn":
        kwargs["checkable"] = True
        toggled = kwargs.pop("toggled", None)

    widget = widget_class(**kwargs)

    if qwtype == "combobox":
        widget.addItems(items)
    elif qwtype == "chkpushbtn":
        if toggled is not None:
            widget.toggled.connect(toggled)

    return widget


class ParameterGroup(QtWidgets.QGroupBox):
    def __init__(self, groupbox_kw=dict(), **kwargs):
        """Easy creation of groupboxes with multiple varying parameters.

        Parameters
        ----------
        params : dict
            see Examples

        Examples
        --------

        >>> ParameterGroup(
            {
                "a": QtWidgets.QDoubleSpinBox(range=(0, 1), singleStep=0.01, value=0.2),
                "b": dict(qwtype="dblspin", range=(0, 2), singleStep=0.04),
                "c": QtWidgets.QSlider(range=(0, 10000))
            }
        )

        """
        super(ParameterGroup, self).__init__(**groupbox_kw)
        self.layout = QtWidgets.QGridLayout(self)

        self.widget_list = []
        self.label_list = []
        for i, (k, v) in enumerate(kwargs.items()):
            if isinstance(v, dict):
                self.widget_list.append(getParameterWidget(**v))
            elif isinstance(v, QtWidgets.QWidget):
                self.widget_list.append(v)
            else:
                raise ValueError(
                    "Each value must be a QtWidgets.QWidget instance"
                    "or a dictionary of keyword arguments to getParameterWidget."
                )
            self.label_list.append(QtWidgets.QLabel(k))
            self.label_list[i].setBuddy(self.widget_list[i])
            self.layout.addWidget(self.label_list[i], i, 0)
            self.layout.addWidget(self.widget_list[i], i, 1)


class AnalysisWidgetBase(pg.GraphicsLayoutWidget):
    def __init__(
        self, orientation="vertical", link="both", cut_to_data="out", **kwargs
    ):
        """__init__ summ

        Parameters
        ----------
        orientation : {"vertical", "horizontal"}, optional
            Sets the orientation of the plots, by default "vertical"
        link : {"x", "y", "both", "none"}, optional
            Link axes, by default "both"
        cut_to_data : {"in", "out", "both", "none"}, optional
            Whether to remove outliers by adjusting color levels, by default "out"

        Raises
        ------
        ValueError
            _description_
        """

        super().__init__(**kwargs)
        if orientation == "horizontal":
            self.is_vertical = False
        elif orientation == "vertical":
            self.is_vertical = True
        else:
            raise ValueError("Orientation must be 'vertical' or 'horizontal'.")
        self.cut_to_data = cut_to_data

        self.prefunc = lambda x: x
        self.mainfunc = lambda x: x
        self.prefunc_only_values = False
        self.mainfunc_only_values = False
        self.prefunc_args = []
        self.prefunc_kwargs = dict()
        self.mainfunc_args = []
        self.mainfunc_kwargs = dict()

        self.qapp = QtCore.QCoreApplication.instance()
        self.initialize_layout()
        if link in ("x", "both"):
            self.axes[1].setXLink(self.axes[0])
        if link in ("y", "both"):
            self.axes[1].setYLink(self.axes[0])

    def initialize_layout(self):
        self.hists = [pg.HistogramLUTItem() for _ in range(2)]
        self.axes = [pg.PlotItem() for _ in range(2)]
        self.images = [xImageItem(axisOrder="row-major") for _ in range(2)]
        for i in range(2):
            if self.is_vertical:
                self.addItem(self.axes[i], i, 0, 1, 1)
                self.addItem(self.hists[i], i, 1, 1, 1)
            else:
                self.addItem(self.axes[i], 0, 2 * i, 1, 1)
                self.addItem(self.hists[i], 0, 2 * i + 1, 1, 1)
            self.axes[i].addItem(self.images[i])
            self.hists[i].setImageItem(self.images[i])
    
    def set_input(self, data=None):
        if data is not None:
            self.input = parse_data(data)
        if self.prefunc_only_values:
            self.input_ = self.prefunc(
                np.asarray(self.input), *self.prefunc_args, **self.prefunc_kwargs
            )
            self.images[0].setImage(
                np.ascontiguousarray(self.input_),
                rect=array_rect(self.input),
                cut_to_data=self.cut_to_data in ("in", "both"),
            )
        else:
            self.input_ = self.prefunc(
                self.input, *self.prefunc_args, **self.prefunc_kwargs
            )
            self.images[0].setDataArray(
                self.input_,
                cut_to_data=self.cut_to_data in ("in", "both"),
            )

    def set_pre_function(self, func, only_values=False, **kwargs):
        self.prefunc_only_values = only_values
        self.prefunc = func
        self.set_pre_function_args(**kwargs)
        self.set_input()

    def set_pre_function_args(self, *args, **kwargs):
        self.prefunc_args = args
        self.prefunc_kwargs = kwargs

    def set_main_function(self, func, only_values=False, **kwargs):
        self.mainfunc_only_values = only_values
        self.mainfunc = func
        self.set_main_function_args(**kwargs)
        self.refresh_output()

    def set_main_function_args(self, *args, **kwargs):
        self.mainfunc_args = args
        self.mainfunc_kwargs = kwargs

    def refresh_output(self):

        if self.mainfunc_only_values:
            self.output = self.mainfunc(
                np.asarray(self.input_), *self.mainfunc_args, **self.mainfunc_kwargs
            )
            self.images[1].setImage(
                np.ascontiguousarray(self.output),
                rect=array_rect(self.input),
                cut_to_data=self.cut_to_data in ("out", "both"),
            )
        else:
            self.output = self.mainfunc(
                self.input_, *self.mainfunc_args, **self.mainfunc_kwargs
            )
            self.images[1].setDataArray(
                self.output,
                cut_to_data=self.cut_to_data in ("out", "both"),
            )

    def refresh_all(self):
        self.set_input()
        self.refresh_output()
        
    
class AnalysisWindow(QtWidgets.QMainWindow):
    def __init__(self, data=None, title=None, *args, **kwargs):
        super().__init__()
        self._main = QtWidgets.QWidget(self)
        self.data = parse_data(data)
        if title is None:
            title = self.data.name
        self.setWindowTitle(title)
        self.setCentralWidget(self._main)
        self.layout = QtWidgets.QVBoxLayout(self._main)
    
    
    # self.ci = GraphicsLayout(**kargs)
    # for n in ['nextRow', 'nextCol', 'nextColumn', 'addPlot', 'addViewBox', 'addItem', 'getItem', 'addLayout', 'addLabel', 'removeItem', 'itemIndex', 'clear']:
    #     setattr(self, n, getattr(self.ci, n))


from scipy.ndimage import gaussian_filter, uniform_filter


if __name__ == "__main__":
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)
    qapp.setStyle("Fusion")

    wdgt = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(wdgt)

    win = AnalysisWidgetBase(orientation="vertical")

    dat = xr.open_dataarray(
        "/Users/khan/Documents/ERLab/CsV3Sb5/2021_Dec_ALS_CV3Sb5/Data/cvs_kxy_small.nc"
    ).sel(eV=-0.15, method="nearest")
    win.set_input(dat)

    win.set_pre_function(gaussian_filter, sigma=[1, 1], only_values=True)
    # win.set_pre_function_args()

    layout.addWidget(win)
    layout.addWidget(
        ParameterGroup(
            a=dict(qwtype="dblspin"),
            b=dict(qwtype="combobox", items=["item1", "item2", "item3"]),
        )
    )

    wdgt.show()
    wdgt.activateWindow()
    wdgt.raise_()
    qapp.exec()
