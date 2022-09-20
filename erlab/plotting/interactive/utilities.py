"""Various helper functions and extensions to pyqtgraph."""

import numpy as np
import sys
import pyqtgraph as pg
import varname
import xarray as xr
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from superqt import QDoubleSlider

__all__ = [
    "parse_data",
    "copy_to_clipboard",
    "gen_single_function_code",
    "gen_function_code",
    "xImageItem",
    "ParameterGroup",
    "AnalysisWidgetBase",
    "AnalysisWindow",
]


def parse_data(data):
    if isinstance(data, xr.Dataset):
        try:
            data = data.spectrum
        except AttributeError:
            raise TypeError(
                "input argument data must be a xarray.DataArray or a "
                "numpy.ndarray. Create an xarray.DataArray "
                "first, either with indexing on the Dataset or by "
                "invoking the `to_array()` method."
            ) from None
    elif isinstance(data, np.ndarray):
        data = xr.DataArray(data)
    return data  # .astype(float, order="C")


def array_rect(data):
    data_coords = tuple(data[dim].values for dim in data.dims)
    data_incs = tuple(coord[1] - coord[0] for coord in data_coords)
    data_lims = tuple((coord[0], coord[-1]) for coord in data_coords)
    y, x = data_lims[0][0] - data_incs[0], data_lims[1][0] - data_incs[1]
    h, w = data_lims[0][-1] - y, data_lims[1][-1] - x
    y += 0.5 * data_incs[0]
    x += 0.5 * data_incs[1]
    return QtCore.QRectF(x, y, w, h)


def copy_to_clipboard(content: str):
    if isinstance(content, list):
        content = "\n".join(content)

    cb = QtWidgets.QApplication.instance().clipboard()
    cb.clear(mode=cb.Clipboard)
    cb.setText(content, mode=cb.Clipboard)
    return content


def process_arg(arg):
    if isinstance(arg, str):
        if arg.startswith("|") and arg.endswith("|"):
            arg = arg[1:-1]
        else:
            arg = f'"{arg}"'
    return arg


def gen_single_function_code(funcname: str, *args, **kwargs):
    """gen_single_function_code generates the string for a Python function call.

    The first argument is the name of the function, and subsequent arguments are
    passed as positional arguments. Keyword arguments are also supported.

    Parameters
    ----------
    funcname : str
        Name of the function.
    *args : list
        Mandatory arguments passed onto the function.
    **kwargs : dict
        Keyword arguments passed onto the function.

    Returns
    -------
    code : str
        generated code.
    """
    tab = "    "
    code = f"{funcname}(\n"
    for v in args:
        if isinstance(v, str):
            if v.startswith("|") and v.endswith("|"):
                v = v[1:-1]
            else:
                v = f'"{v}"'
        code += f"{tab}{v},\n"
    for k, v in kwargs.items():
        if isinstance(v, str):
            v = f'"{v}"'
        code += f"{tab}{k}={v},\n"
    code += ")"
    return code


def gen_function_code(copy=True, **kwargs):
    r"""Copies the Python code for function calls to the clipboard.

    The result is
    copied to your clipboard in a form that can be pasted into an interactive Python
    session or Jupyter notebook cell.

    Parameters
    ----------
    copy : bool, default=True

    **kwargs : dict
        Dictionary where the keys are the string of the function call and the values are
        a list of function arguments. The last item, if a dictionary, is interpreted as
        keyword arguments.

    """
    code_list = []
    for k, v in kwargs.items():
        if not isinstance(v[-1], dict):
            v.append(dict())
        code_list.append(gen_single_function_code(k, *v[:-1], **v[-1]))

    code_str = "\n".join(code_list)

    if copy:
        return copy_to_clipboard(code_str)
    else:
        return code_str


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


class ParameterGroup(QtWidgets.QGroupBox):

    VALID_QWTYPE = {
        "spin": QtWidgets.QSpinBox,
        "dblspin": QtWidgets.QDoubleSpinBox,
        "slider": QtWidgets.QSlider,
        "dblslider": QDoubleSlider,
        "chkbox": QtWidgets.QCheckBox,
        "pushbtn": QtWidgets.QPushButton,
        "chkpushbtn": QtWidgets.QPushButton,
        "combobox": QtWidgets.QComboBox,
    }
    VALID_QLTYPE = {
        "hbox": QtWidgets.QHBoxLayout,
        "vbox": QtWidgets.QVBoxLayout,
    }
    # !TODO: reimplement everything, add label by keyword argument. Apply more flexible
    # placements by adding layout support

    sigParameterChanged = QtCore.Signal(list)

    @staticmethod
    def getParameterWidget(qwtype, **kwargs):
        """

        Parameters
        ----------
        qwtype : {'spin', 'dblspin', 'slider', 'dblslider', 'chkbox', 'pushbtn', 'chkpushbtn', 'combobox'}

        """

        if qwtype not in ParameterGroup.VALID_QWTYPE.keys():
            raise ValueError(
                f"qwtype must be one of {list(ParameterGroup.VALID_QWTYPE.keys())}"
            )

        widget_class = ParameterGroup.VALID_QWTYPE[qwtype]

        if qwtype == "combobox":
            items = kwargs.pop("items", None)
            currtxt = kwargs.pop("currentText", None)
            curridx = kwargs.pop("currentIndex", None)
                
        elif qwtype.endswith("pushbtn"):
            pressed = kwargs.pop("pressed", None)
            released = kwargs.pop("released", None)
            if qwtype == "chkpushbtn":
                kwargs["checkable"] = True
                toggled = kwargs.pop("toggled", None)
            else:
                clicked = kwargs.pop("clicked", None)
        newrange = kwargs.pop("range", None)

        valueChanged = kwargs.pop("valueChanged", None)
        textChanged = kwargs.pop("textChanged", None)

        widget = widget_class(**kwargs)

        if qwtype == "combobox":
            widget.addItems(items)
            if currtxt is not None:
                widget.setCurrentText(currtxt)
            if curridx is not None:
                widget.setCurrentIndex(curridx)
        elif qwtype.endswith("pushbtn"):
            if pressed is not None:
                widget.pressed.connect(pressed)
            if released is not None:
                widget.released.connect(released)
            if qwtype == "chkpushbtn":
                if toggled is not None:
                    widget.toggled.connect(toggled)
            else:
                if clicked is not None:
                    widget.clicked.connect(clicked)

        if newrange is not None:
            widget.setRange(*newrange)

        if valueChanged is not None:
            widget.valueChanged.connect(valueChanged)
        if textChanged is not None:
            widget.textChanged.connect(textChanged)

        return widget

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

        self.labels = []
        self.untracked = []
        self.widgets = dict()
        for i, (k, v) in enumerate(kwargs.items()):
            if isinstance(v, dict):
                showlabel = v.pop("showlabel", True)
                if v.pop("notrack", False):
                    self.untracked.append(k)
                self.widgets[k] = self.getParameterWidget(**v)
            elif isinstance(v, QtWidgets.QWidget):
                self.widgets[k] = v
            else:
                raise ValueError(
                    "Each value must be a QtWidgets.QWidget instance"
                    "or a dictionary of keyword arguments to getParameterWidget."
                )
            self.labels.append(QtWidgets.QLabel(k))
            self.labels[i].setBuddy(self.widgets[k])
            if showlabel:
                self.layout.addWidget(self.labels[i], i, 0)
                self.layout.addWidget(self.widgets[k], i, 1)
            else:
                self.layout.addWidget(self.widgets[k], i, 0, 1, 2)

        self.global_connect()

    def widget_value(self, widget):
        if isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
            return widget.value()
        elif isinstance(widget, QtWidgets.QAbstractSpinBox):
            return widget.text()
        elif isinstance(widget, QtWidgets.QAbstractSlider):
            return widget.value()
        elif isinstance(widget, QtWidgets.QCheckBox):
            if widget.isTristate():
                return widget.checkState()
            else:
                return widget.isChecked()
        elif isinstance(widget, QtWidgets.QAbstractButton):
            if widget.isCheckable():
                return widget.isChecked()
            else:
                return widget.isDown()
        elif isinstance(widget, QtWidgets.QComboBox):
            return widget.currentText()

    def widget_change_signal(self, widget):
        if isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
            return widget.valueChanged
        elif isinstance(widget, QtWidgets.QAbstractSpinBox):
            return widget.editingFinished
        elif isinstance(widget, QtWidgets.QAbstractSlider):
            return widget.valueChanged
        elif isinstance(widget, QtWidgets.QCheckBox):
            return widget.stateChanged
        elif isinstance(widget, QtWidgets.QAbstractButton):
            if widget.isCheckable():
                return widget.clicked
            else:
                return widget.toggled
        elif isinstance(widget, QtWidgets.QComboBox):
            return widget.currentTextChanged

    def global_connect(self):
        for k, v in self.widgets.items():
            if k not in self.untracked:
                self.widget_change_signal(v).connect(
                    lambda x: self.sigParameterChanged.emit([k, x])
                )

    @property
    def values(self):
        return {k: self.widget_value(v) for k, v in self.widgets.items()}

    # "spin": QtWidgets.QSpinBox,
    # "dblspin": QtWidgets.QDoubleSpinBox,
    # "slider": QtWidgets.QSlider,
    # "dblslider": QDoubleSlider,
    # "chkbox": QtWidgets.QCheckBox,
    # "pushbtn": QtWidgets.QPushButton,
    # "chkpushbtn": QtWidgets.QPushButton,
    # "combobox": QtWidgets.QComboBox,


class ROIControls(ParameterGroup):
    def __init__(self, roi: pg.ROI, spinbox_kw=dict(), **kwargs):
        self.roi = roi
        x0, y0, x1, y1 = self.roi_limits
        xm, ym, xM, yM = self.roi.maxBounds.getCoords()

        default_properties = dict(decimals=3, singleStep=0.002, keyboardTracking=False)
        for k, v in default_properties.items():
            spinbox_kw.setdefault(k, v)
        print(self.roi.parentItem())

        super(ROIControls, self).__init__(
            x0=dict(
                qwtype="dblspin",
                value=x0,
                valueChanged=lambda x: self.modify_roi(x0=x),
                minimum=xm,
                maximum=xM,
                **spinbox_kw,
            ),
            x1=dict(
                qwtype="dblspin",
                value=x1,
                valueChanged=lambda x: self.modify_roi(x1=x),
                minimum=xm,
                maximum=xM,
                **spinbox_kw,
            ),
            y0=dict(
                qwtype="dblspin",
                value=y0,
                valueChanged=lambda x: self.modify_roi(y0=x),
                minimum=ym,
                maximum=yM,
                **spinbox_kw,
            ),
            y1=dict(
                qwtype="dblspin",
                value=y1,
                valueChanged=lambda x: self.modify_roi(y1=x),
                minimum=ym,
                maximum=yM,
                **spinbox_kw,
            ),
            drawbtn=dict(
                qwtype="chkpushbtn",
                toggled=self.draw_mode,
                showlabel=False,
                text="Draw",
            ),
            **kwargs,
        )
        self.draw_button = self.widgets["drawbtn"]
        self.roi_spin = [self.widgets[i] for i in ["x0", "y0", "x1", "y1"]]
        self.roi.sigRegionChanged.connect(self.update_pos)

    @property
    def roi_limits(self):
        x0, y0 = self.roi.state["pos"]
        w, h = self.roi.state["size"]
        x1, y1 = x0 + w, y0 + h
        return x0, y0, x1, y1

    def update_pos(self):
        self.widgets["x0"].setMaximum(self.widgets["x1"].value())
        self.widgets["y0"].setMaximum(self.widgets["y1"].value())
        self.widgets["x1"].setMinimum(self.widgets["x0"].value())
        self.widgets["y1"].setMinimum(self.widgets["y0"].value())
        for pos, spin in zip(self.roi_limits, self.roi_spin):
            spin.blockSignals(True)
            spin.setValue(pos)
            spin.blockSignals(False)

    def modify_roi(self, x0=None, y0=None, x1=None, y1=None, update=True):
        lim_new = (x0, y0, x1, y1)
        lim_old = self.roi_limits
        x0, y0, x1, y1 = ((f if f is not None else i) for i, f in zip(lim_old, lim_new))
        xm, ym, xM, yM = self.roi.maxBounds.getCoords()
        x0, y0, x1, y1 = max(x0, xm), max(y0, ym), min(x1, xM), min(y1, yM)
        self.roi.setPos((x0, y0), update=False)
        self.roi.setSize((x1 - x0, y1 - y0), update=update)

    def draw_mode(self, toggle):

        vb = self.roi.parentItem().getViewBox()

        if not toggle:
            vb.mouseDragEvent = self._drag_evt_old
            vb.setMouseMode(self._state_old)
            vb.rbScaleBox.setPen(pg.mkPen((255, 255, 100), width=1))
            vb.rbScaleBox.setBrush(pg.mkBrush(255, 255, 0, 100))
            vb.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
            return

        vb.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self._state_old = vb.state["mouseMode"]
        self._drag_evt_old = vb.mouseDragEvent

        vb.setMouseMode(pg.ViewBox.RectMode)
        vb.rbScaleBox.setPen(pg.mkPen((255, 255, 255), width=1))
        vb.rbScaleBox.setBrush(pg.mkBrush(255, 255, 255, 100))

        def mouseDragEventCustom(ev, axis=None):
            ev.accept()
            pos = ev.scenePos()
            dif = pos - ev.lastScenePos()
            dif = dif * -1

            mouseEnabled = np.array(vb.state["mouseEnabled"], dtype=np.float64)
            mask = mouseEnabled.copy()
            if axis is not None:
                mask[1 - axis] = 0.0

            if ev.button() & QtCore.Qt.MouseButton.LeftButton:
                if vb.state["mouseMode"] == pg.ViewBox.RectMode and axis is None:
                    if ev.isFinish():
                        vb.rbScaleBox.hide()
                        ax = QtCore.QRectF(
                            pg.Point(ev.buttonDownScenePos(ev.button())), pg.Point(pos)
                        )
                        ax = vb.childGroup.mapRectFromScene(ax)
                        self.modify_roi(*ax.getCoords())
                    else:
                        vb.updateScaleBox(ev.buttonDownScenePos(), ev.scenePos())
            elif ev.button() & QtCore.Qt.MouseButton.MiddleButton:
                tr = vb.childGroup.transform()
                tr = pg.invertQTransform(tr)
                tr = tr.map(dif * mask) - tr.map(pg.Point(0, 0))

                x = tr.x() if mask[0] == 1 else None
                y = tr.y() if mask[1] == 1 else None

                vb._resetTarget()
                if x is not None or y is not None:
                    vb.translateBy(x=x, y=y)
                vb.sigRangeChangedManually.emit(vb.state["mouseEnabled"])
            elif ev.button() & QtCore.Qt.MouseButton.RightButton:
                if vb.state["aspectLocked"] is not False:
                    mask[0] = 0

                dif = ev.screenPos() - ev.lastScreenPos()
                dif = np.array([dif.x(), dif.y()])
                dif[0] *= -1
                s = ((mask * 0.02) + 1) ** dif

                tr = pg.invertQTransform(vb.childGroup.transform())
                x = s[0] if mouseEnabled[0] == 1 else None
                y = s[1] if mouseEnabled[1] == 1 else None

                center = pg.Point(
                    tr.map(ev.buttonDownPos(QtCore.Qt.MouseButton.RightButton))
                )
                vb._resetTarget()
                vb.scaleBy(x=x, y=y, center=center)
                vb.sigRangeChangedManually.emit(vb.state["mouseEnabled"])

        vb.mouseDragEvent = mouseDragEventCustom  # set to modified mouseDragEvent

class PostInitCaller(type(QtWidgets.QMainWindow)):
    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.__post_init__()
        return obj

class AnalysisWindow(QtWidgets.QMainWindow):
    def __init__(self, data, title=None, layout="horizontal", data_is_input=True, *args, **kwargs):
        self.qapp = QtCore.QCoreApplication.instance()
        if not self.qapp:
            self.qapp = QtWidgets.QApplication(sys.argv)
        self.qapp.setStyle("Fusion")
        
        super().__init__()
        
        self.data = parse_data(data)
        if title is None:
            title = self.data.name
        self.setWindowTitle(title)

        self._main = QtWidgets.QWidget(self)
        self.setCentralWidget(self._main)

        self.controlgroup = QtWidgets.QWidget()
        if layout == "vertical":
            self.layout = QtWidgets.QVBoxLayout(self._main)
            self.controls = QtWidgets.QHBoxLayout(self.controlgroup)
        elif layout == "horizontal":
            self.layout = QtWidgets.QHBoxLayout(self._main)
            self.controls = QtWidgets.QVBoxLayout(self.controlgroup)
        else:
            raise ValueError("Layout must be 'vertical' or 'horizontal'.")

        self.aw = AnalysisWidgetBase(*args, **kwargs)
        for n in [
            "set_input",
            "add_roi",
            "axes",
            "hists",
            "images",
            # "set_pre_function",
            # "set_pre_function_args",
            # "set_main_function",
            # "set_main_function_args",
            # "refresh_output",
            # "refresh_all",
        ]:
            setattr(self, n, getattr(self.aw, n))
        self.layout.addWidget(self.aw)
        self.layout.addWidget(self.controlgroup)
        
        self.layout.setContentsMargins(QtCore.QMargins(0, 0, 0, 0))
        self.layout.setSpacing(0)
        self.controlgroup.setContentsMargins(QtCore.QMargins(0, 0, 0, 0))
        if data_is_input:
            self.set_input(data)
            
    def __post_init__(self, execute=None):
        self.show()
        self.activateWindow()
        self.raise_()
        
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

    def addParameterGroup(self, *args, **kwargs):
        group = ParameterGroup(*args, **kwargs)
        self.controls.addWidget(group)
        return group


class AnalysisWidgetBase(pg.GraphicsLayoutWidget):
    def __init__(
        self,
        orientation="vertical",
        num_ax=2,
        link="both",
        cut_to_data="none",
        **kwargs,
    ):
        """__init__ summ

        Parameters
        ----------
        orientation : {"vertical", "horizontal"}, optional
            Sets the orientation of the plots, by default "vertical"
        link : {"x", "y", "both", "none"}, optional
            Link axes, by default "both"
        cut_to_data : {"in", "out", "both", "none"}, optional
            Whether to remove outliers by adjusting color levels, by default "none"

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
        
        self.input = None

        self.initialize_layout(num_ax)

        for i in range(1, num_ax):
            if link in ("x", "both"):
                self.axes[i].setXLink(self.axes[0])
            if link in ("y", "both"):
                self.axes[i].setYLink(self.axes[0])

    def initialize_layout(self, nax):
        self.hists = [pg.HistogramLUTItem() for _ in range(nax)]
        self.axes = [pg.PlotItem() for _ in range(nax)]
        self.images = [xImageItem(axisOrder="row-major") for _ in range(nax)]
        for i in range(nax):
            self.addItem(self.axes[i], *self.get_axis_pos(i))
            self.addItem(self.hists[i], *self.get_hist_pos(i))
            self.axes[i].addItem(self.images[i])
            self.hists[i].setImageItem(self.images[i])
        self.roi = [None for _ in range(nax)]

    def get_axis_pos(self, ax):
        if self.is_vertical:
            return ax, 0, 1, 1
        else:
            return 0, 2 * ax, 1, 1

    def get_hist_pos(self, ax):
        if self.is_vertical:
            return ax, 1, 1, 1
        else:
            return 0, 2 * ax + 1, 1, 1

    def setStretchFactors(self, factors):
        for i, f in enumerate(factors):
            self.setStretchFactor(i, f)

    def setStretchFactor(self, i, factor):
        if self.is_vertical:
            self.ci.layout.setRowStretchFactor(i, factor)
        else:
            self.ci.layout.setColumnStretchFactor(i, factor)

    def set_input(self, data=None):
        if data is not None:
            self.input = parse_data(data)
        self.images[0].setDataArray(
            self.input,
            cut_to_data=self.cut_to_data in ("in", "both"),
        )

    def add_roi(self, i):
        self.roi[i] = pg.ROI(
            [-0.1, -0.5],
            [0.3, 0.5],
            parent=self.images[i],
            rotatable=False,
            resizable=True,
            maxBounds=self.axes[i].getViewBox().itemBoundingRect(self.images[i]),
        )
        self.roi[i].addScaleHandle([0.5, 1], [0.5, 0])
        self.roi[i].addScaleHandle([0.5, 0], [0.5, 1])
        self.roi[i].addScaleHandle([1, 0.5], [0, 0.5])
        self.roi[i].addScaleHandle([0, 0.5], [1, 0.5])
        self.axes[i].addItem(self.roi[i])
        self.roi[i].setZValue(10)
        return self.roi[i]


class ComparisonWidget(AnalysisWidgetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, nax=2, **kwargs)
        self.prefunc = lambda x: x
        self.mainfunc = lambda x: x
        self.prefunc_only_values = False
        self.mainfunc_only_values = False
        self.prefunc_kwargs = dict()
        self.mainfunc_kwargs = dict()

    def call_prefunc(self, x):
        if self.prefunc_only_values:
            x = np.asarray(x)
        return self.prefunc(x, **self.prefunc_kwargs)

    def set_input(self, data=None):
        if data is not None:
            self.input_ = parse_data(data)

        self.input = self.call_prefunc(self.input_)
        if self.prefunc_only_values:
            self.images[0].setImage(
                np.ascontiguousarray(self.input),
                rect=array_rect(self.input_),
                cut_to_data=self.cut_to_data in ("in", "both"),
            )
        else:
            self.images[0].setDataArray(
                self.input,
                cut_to_data=self.cut_to_data in ("in", "both"),
            )

    def set_pre_function(self, func, only_values=False, **kwargs):
        self.prefunc_only_values = only_values
        self.prefunc = func
        self.set_pre_function_args(**kwargs)
        self.set_input()

    def set_main_function(self, func, only_values=False, **kwargs):
        self.mainfunc_only_values = only_values
        self.mainfunc = func
        self.set_main_function_args(**kwargs)
        self.refresh_output()

    def set_pre_function_args(self, **kwargs):
        for k, v in kwargs.items():
            self.prefunc_kwargs[k] = v
        self.refresh_output()

    def set_main_function_args(self, **kwargs):
        for k, v in kwargs.items():
            self.mainfunc_kwargs[k] = v
        self.refresh_output()

    def refresh_output(self):
        if self.mainfunc_only_values:
            self.output = self.mainfunc(np.asarray(self.input_), **self.mainfunc_kwargs)
            self.images[1].setImage(
                np.ascontiguousarray(self.output),
                rect=array_rect(self.input),
                cut_to_data=self.cut_to_data in ("out", "both"),
            )
        else:
            self.output = self.mainfunc(self.input_, **self.mainfunc_kwargs)
            self.images[1].setDataArray(
                self.output,
                cut_to_data=self.cut_to_data in ("out", "both"),
            )

    def refresh_all(self):
        self.set_input()
        self.refresh_output()


if __name__ == "__main__":
    from scipy.ndimage import gaussian_filter, uniform_filter

    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)
    qapp.setStyle("Fusion")

    wdgt = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(wdgt)

    dat = (
        xr.open_dataarray(
            "/Users/khan/Documents/ERLab/CsV3Sb5/2021_Dec_ALS_CV3Sb5/Data/cvs_kxy_small.nc"
        )
        .sel(eV=-0.15, method="nearest")
        .fillna(0)
    )
    win = AnalysisWindow(dat, orientation="vertical")

    # win.set_pre_function(gaussian_filter, sigma=[1, 1], only_values=True)
    # win.set_pre_function(gaussian_filter, sigma=(0.1, 0.1))

    gaussfilt_2d = lambda dat, sx, sy: gaussian_filter(dat, sigma=(sx, sy))
    # win.set_main_function(gaussfilt_2d, sx=0.1, sy=1)

    layout.addWidget(win)
    layout.addWidget(
        ParameterGroup(
            sigma_x=dict(
                qwtype="dblspin",
                minimum=0,
                maximum=10,
                valueChanged=lambda x: win.set_main_function_args(sx=x),
            ),
            sigma_y=dict(
                qwtype="dblspin",
                minimum=0,
                maximum=10,
                valueChanged=lambda x: win.set_main_function_args(sy=x),
            ),
            b=dict(qwtype="combobox", items=["item1", "item2", "item3"]),
        )
    )
    new_roi = win.add_roi(0)

    layout.addWidget(ROIControls(new_roi))

    wdgt.show()
    wdgt.activateWindow()
    wdgt.raise_()
    qapp.exec()
