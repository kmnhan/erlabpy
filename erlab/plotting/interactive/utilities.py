"""Various helper functions and extensions to pyqtgraph."""

import sys
import re
import weakref
from collections.abc import Iterable, Sequence

import numpy as np
import pyclip
import pyqtgraph as pg
import xarray as xr
from PySide6 import QtCore, QtGui, QtWidgets
from superqt import QDoubleSlider
from erlab.plotting.interactive.colors import pg_colormap_powernorm

__all__ = [
    "parse_data",
    "copy_to_clipboard",
    "gen_single_function_code",
    "gen_function_code",
    "BetterSpinBox",
    "BetterImageItem",
    "BetterAxisItem",
    "BetterColorBarItem",
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
    cb.clear(mode=cb.Mode.Clipboard)
    cb.setText(content, mode=cb.Mode.Clipboard)
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


class BetterSpinBox(QtWidgets.QAbstractSpinBox):
    valueChanged = QtCore.Signal(object)
    textChanged = QtCore.Signal(object)

    def __init__(
        self,
        *args,
        integer=False,
        compact=True,
        discrete=False,
        decimals=3,
        significant=False,
        scientific=False,
        value=0.0,
        **kwargs,
    ):
        """
        Parameters
        ----------
        integer : boolean, optional
            If `True`, the spinbox will only display integer values.
        compact : boolean, optional
            Whether to reduce the height of the spinbox.
        discrete : boolean, optional
            If `True` the spinbox will only step to pre-determined discrete values.
            If `False`, the spinbox will just add or subtract the predetermined
            increment when increasing or decreasing the step.
        scientific : boolean, optional
            Whether to print in scientific notation.
        decimals : int, optional
            The precision of the spinbox. See the `significant` argument for the
            meaning. When `int` is `True`, this argument is ignored.
        significant : boolean, optional
            If `True`, `decimals` will specify the total number of significant digits,
            before or after the decimal point, ignoring leading zeros.
            If `False`, `decimals` will specify the total number of digits after the
            decimal point, including leading zeros.
            When `int` or `scientific` is `True`, this argument is ignored.
        value : float, optional
            Initial value of the spinbox.
        """

        self._only_int = integer
        self._is_compact = compact
        self._is_discrete = discrete
        self._is_scientific = scientific
        self._decimal_significant = significant
        self.setDecimals(decimals)

        self._value = value
        self._min = -np.inf
        self._max = np.inf
        self._step = 1 if self._only_int else 0.01
        super().__init__(*args, **kwargs)
        # self.editingFinished.disconnect()
        self.setSizePolicy(
            # QtWidgets.QSizePolicy.Policy.Expanding,
            # QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Minimum,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.editingFinished.connect(self.editingFinishedEvent)
        self._updateHeight()
        self._updateWidth()

        if self.isReadOnly():
            self.lineEdit().setReadOnly(True)
            self.setButtonSymbols(self.ButtonSymbols.NoButtons)
        self.setValue(self.value())

    def setDecimals(self, decimals):
        self._decimals = decimals

    def decimals(self):
        return self._decimals

    def setRange(self, mn, mx):
        self.setMinimum(min(mn, mx))
        self.setMaximum(max(mn, mx))

    def widthFromText(self, text):
        return QtGui.QFontMetrics(self.font()).boundingRect(text).width()

    def widthFromValue(self, value):
        return self.widthFromText(self.textFromValue(value))

    # def sizeHint(self):
    #     return QtCore.QSize(
    #         max(
    #             self.widthFromValue(self.maximum()), self.widthFromValue(self.minimum())
    #         ),
    #         0,
    #     )

    def setMaximum(self, mx):
        if self._only_int and np.isfinite(mx):
            mx = round(mx)
        elif np.isnan(mx):
            mx = np.inf
        self._max = mx
        self._updateWidth()

    def setMinimum(self, mn):
        if self._only_int and np.isfinite(mn):
            mn = round(mn)
        elif np.isnan(mn):
            mn = -np.inf
        self._min = mn
        self._updateWidth()

    def setSingleStep(self, step):
        if self._only_int:
            step = round(step)
        self._step = abs(step)

    def singleStep(self):
        return self._step

    def maximum(self):
        return self._max

    def minimum(self):
        return self._min

    def value(self):
        if self._only_int:
            return int(self._value)
        else:
            return self._value

    def text(self):
        return self.textFromValue(self.value())

    def textFromValue(self, value):
        if (not self._only_int) or (not np.isfinite(value)):
            if self._is_scientific:
                return np.format_float_scientific(
                    value,
                    precision=self.decimals(),
                    unique=False,
                    trim="k",
                    exp_digits=1,
                )
            else:
                return np.format_float_positional(
                    value,
                    precision=self.decimals(),
                    unique=False,
                    fractional=not self._decimal_significant,
                    trim="k",
                )
        else:
            return str(int(value))

    def valueFromText(self, text):
        if text == "":
            return np.nan
        if self._only_int:
            return int(text)
        else:
            return float(text)

    def stepBy(self, steps):
        inc = self.singleStep()
        if (
            all(np.isfinite([self.maximum(), self.minimum(), self.value()]))
            and self._is_discrete
        ):
            self.setValue(
                self.minimum()
                + inc
                * max(
                    min(
                        round((self.value() + steps * inc - self.minimum()) / inc),
                        int((self.maximum() - self.minimum()) / inc),
                    ),
                    0,
                )
            )
        else:
            if steps > 0:
                self.setValue(min(inc * steps + self.value(), self.maximum()))
            else:
                self.setValue(max(inc * steps + self.value(), self.minimum()))

    def stepEnabled(self):
        if self.isReadOnly():
            return self.StepEnabledFlag.StepNone
        if self.wrapping():
            return (
                self.StepEnabledFlag.StepDownEnabled
                | self.StepEnabledFlag.StepUpEnabled
            )
        if self.value() < self.maximum():
            if self.value() > self.minimum():
                return (
                    self.StepEnabledFlag.StepDownEnabled
                    | self.StepEnabledFlag.StepUpEnabled
                )
            else:
                return self.StepEnabledFlag.StepUpEnabled
        elif self.value() > self.minimum():
            return self.StepEnabledFlag.StepDownEnabled
        else:
            return self.StepEnabledFlag.StepNone

    def setValue(self, val):
        if np.isnan(val):
            self._value = np.nan
        else:
            self._value = max(self.minimum(), min(val, self.maximum()))

        if self._only_int and np.isfinite(self._value):
            self._value = round(self._value)

        self.valueChanged.emit(self.value())
        self.lineEdit().setText(self.text())
        self.textChanged.emit(self.text())

    # def fixup(self, input):
    #     # fixup is called when the spinbox loses focus with an invalid or intermediate string
    #     self.lineEdit().setText(self.text())

    #     # support both PyQt APIs (for Python 2 and 3 respectively)
    #     # http://pyqt.sourceforge.net/Docs/PyQt4/python_v3.html#qvalidator

    #     print(input)
    #     try:
    #         input.clear()
    #         input.append(self.lineEdit().text())
    #     except AttributeError:
    #         return self.lineEdit().text()

    # # def hasAcceptableInput(self) -> bool:
    # #     return True

    def validate(self, strn, pos):
        # if self.skipValidate:
        if False:
            ret = QtGui.QValidator.State.Acceptable
        else:
            ret = QtGui.QValidator.State.Intermediate
            try:
                val = float(self.value())
                if val < self.maximum() and val > self.minimum():
                    ret = QtGui.QValidator.State.Acceptable
            except ValueError:
                pass

        ## note: if text is invalid, we don't change the textValid flag
        ## since the text will be forced to its previous state anyway
        self.update()

        ## support 2 different pyqt APIs. Bleh.

        # print(strn, pos, ret)
        if hasattr(QtCore, "QString"):
            return (ret, pos)
        else:
            return (ret, strn, pos)

    def editingFinishedEvent(self):
        self.setValue(self.valueFromText(self.lineEdit().text()))

    # def keyPressEvent(self, evt):
    #     if evt.key() == QtGui.QKeySequence("Escape") or evt.key() == QtGui.QKeySequence("Return"):
    #         self.focusOutEvent(QtGui.QFocusEvent(QtCore.QEvent.FocusIn, QtCore.Qt.MouseFocusReason))
    #         self.editingFinishedEvent()
    #         print("hey")
    #     else:
    #         super().keyPressEvent(evt)

    def _updateHeight(self):
        if self._is_compact:
            self.setFixedHeight(QtGui.QFontMetrics(self.font()).height() + 3)

    def _get_offset(self):
        spin = QtWidgets.QDoubleSpinBox(self, minimum=0, maximum=0)
        w = (
            spin.minimumSizeHint().width()
            - QtGui.QFontMetrics(spin.font())
            .boundingRect(spin.textFromValue(0.0))
            .width()
        )
        spin.setDisabled(True)
        spin.setVisible(False)
        del spin
        return w + 10

    def _updateWidth(self):
        self.setMinimumWidth(
            max(
                self.widthFromValue(self.maximum()), self.widthFromValue(self.minimum())
            )
            + self._get_offset()
        )


class BetterImageItem(pg.ImageItem):
    sigColorChanged = QtCore.Signal()
    sigLimitChanged = QtCore.Signal(float, float)

    def __init__(self, image=None, **kargs):
        self.auto_levels: bool = True
        super().__init__(image, **kargs)

    def setImage(self, image=None, autoLevels: bool | None = None, **kargs):
        if autoLevels is None:
            if "levels" in kargs:
                self.setAutoLevels(False)
        else:
            self.setAutoLevels(autoLevels)

        super().setImage(image=image, autoLevels=self.auto_levels, **kargs)

        if image is not None:
            if self.auto_levels:
                self.sigLimitChanged.emit(*self.levels)

    def setAutoLevels(self, autoLevels: bool):
        self.auto_levels = autoLevels

    def updateImage(self, *args, **kargs):
        kargs.setdefault("autoLevels", self.auto_levels)
        return self.setImage(*args, **kargs)

    def setLevels(self, levels, update: bool = True):
        super().setLevels(levels, update)
        self.sigLimitChanged.emit(*self.levels)

    def set_colormap(
        self,
        cmap: pg.ColorMap | str,
        gamma: float,
        reverse: bool = False,
        highContrast: bool = False,
        zeroCentered: bool = False,
        update: bool = True,
    ):
        cmap = pg_colormap_powernorm(
            cmap,
            gamma,
            reverse,
            highContrast=highContrast,
            zeroCentered=zeroCentered,
        )
        self.set_pg_colormap(cmap, update=update)

    def set_pg_colormap(self, cmap: pg.ColorMap, update: bool = True):
        self._colorMap = cmap
        self.setLookupTable(cmap.getStops()[1], update=update)
        self.sigColorChanged.emit()


class BetterAxisItem(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def updateAutoSIPrefix(self):
        if self.label.isVisible():
            if self.logMode:
                _range = 10 ** np.array(self.range)
            else:
                _range = self.range
            (scale, prefix) = pg.siScale(
                max(abs(_range[0] * self.scale), abs(_range[1] * self.scale))
            )
            # if self.labelUnits == '' and prefix in ['k', 'm']:  ## If we are not showing units, wait until 1e6 before scaling.
            #     scale = 1.0
            #     prefix = ''
            self.autoSIPrefixScale = scale
            self.labelUnitPrefix = prefix
        else:
            self.autoSIPrefixScale = 1.0

        self._updateLabel()

    def tickStrings(self, values, scale, spacing):
        if self.logMode:
            return self.logTickStrings(values, scale, spacing)

        places = max(0, np.ceil(-np.log10(spacing * scale)))
        strings = []
        for v in values:
            vs = v * scale
            if abs(vs) < 0.001 or abs(vs) >= 10000:
                vstr = "%g" % vs
            else:
                vstr = ("%%0.%df" % places) % vs
            strings.append(vstr.replace("-", "−"))
        return strings

    def labelString(self):
        if self.labelUnits == "":
            if not self.autoSIPrefix or self.autoSIPrefixScale == 1.0:
                units = ""
            else:
                # units = re.sub(
                #     r"1E\+?(\-?)0?(\d?\d)",
                #     r"10<sup>\1\2</sup>",
                #     f"(×{1.0 / self.autoSIPrefixScale:.3G})",
                # )#.replace("-", "−")

                try:
                    units = "".join(
                        re.search(
                            r"1E\+?(\-?)0?(\d?\d)",
                            f"{1.0 / self.autoSIPrefixScale:.3G}",
                        ).groups()
                    )
                    for k, v in zip(
                        ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-"),
                        ("⁰", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸", "⁹", "⁻"),
                    ):
                        units = units.replace(k, v)
                    units = f"10{units}"
                except AttributeError:
                    units = f"{1.0 / self.autoSIPrefixScale:.3G}"
                units = f"(×{units})"

        else:
            units = f"({self.labelUnitPrefix}{self.labelUnits})"

        if self.labelText == "":
            s = units
        else:
            s = f"{self.labelText} {units}"

        style = ";".join([f"{k}: {v}" for k, v in self.labelStyle.items()])

        return f"<span style='{style}'>{s}</span>"

    def setLabel(self, text=None, units=None, unitPrefix=None, **args):
        # `None` input is kept for backward compatibility!
        self.labelText = text or ""
        self.labelUnits = units or ""
        self.labelUnitPrefix = unitPrefix or ""
        if len(args) > 0:
            self.labelStyle = args
        # Account empty string and `None` for units and text
        visible = True if (text or units) else False
        if text == units == "":
            visible = True
        self.showLabel(visible)
        self._updateLabel()


class BetterColorBarItem(pg.PlotItem):
    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        image: Sequence[BetterImageItem] | BetterImageItem | None = None,
        limits: tuple[float, float] | None = None,
        pen: QtGui.QPen | str = "c",
        hoverPen: QtGui.QPen | str = "m",
        hoverBrush: QtGui.QBrush | str = "#FFFFFF33",
        **kargs,
    ):
        super().__init__(
            parent,
            axisItems={
                a: BetterAxisItem(a) for a in ("left", "right", "top", "bottom")
            },
            **kargs,
        )

        self.setDefaultPadding(0)
        # self.hideButtons()
        self.setMenuEnabled(False)
        self.vb.setMouseEnabled(x=False, y=True)

        self._colorbar = pg.ImageItem(
            np.linspace(0, 1, 4096).reshape((-1, 1)), axisOrder="row-major"
        )
        self.addItem(self._colorbar)

        self._span = pg.LinearRegionItem(
            (0, 1),
            "horizontal",
            swapMode="block",
            pen=pen,
            brush=pg.mkBrush(None),
            hoverPen=hoverPen,
            hoverBrush=hoverBrush,
        )
        self._span.setZValue(1000)
        self._span.lines[0].addMarker("<|>", size=6)
        self._span.lines[1].addMarker("<|>", size=6)

        self.addItem(self._span)

        self._fixedlimits: tuple[float, float] | None = None

        self._images: set[weakref.ref[BetterImageItem]] = set()
        self._primary_image: weakref.ref[BetterImageItem] | None = None
        if image is not None:
            self.setImageItem(image)
        if limits is not None:
            self.setLimits(limits)
        self.setLabels(right=("", ""))
        self.set_dimensions()

    def set_dimensions(
        self,
        horiz_pad: int | None = None,
        vert_pad: int | None = None,
        font_size: float = 11.0,
    ):
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Expanding
        )
        self.showAxes((True, True, True, True), showValues=(False, False, True, False))
        self.getAxis("top").setHeight(vert_pad)
        self.getAxis("bottom").setHeight(vert_pad)
        self.getAxis("right").setWidth(horiz_pad)
        self.getAxis("left").setWidth(None)

        font = QtGui.QFont()
        font.setPointSizeF(float(font_size))
        for axis in ("left", "bottom", "right", "top"):
            self.getAxis(axis).setTickFont(font)

    def _level_change(self):
        if not self.isVisible():
            return
        for img_ref in self._images:
            if not img_ref().auto_levels:
                img_ref().setLevels(self._span.getRegion())
        self.limit_changed()

    def _level_change_fin(self):
        pass

    @property
    def levels(self) -> Sequence[float]:
        return self._primary_image().getLevels()

    @property
    def limits(self) -> tuple[float, float]:
        if self._fixedlimits is not None:
            return self._fixedlimits
        else:
            return self._primary_image().quickMinMax(targetSize=2**16)

    def setLimits(self, limits: tuple[float, float] | None):
        self._fixedlimits = limits
        self.limit_changed()
    
    def addImage(self, image:Sequence[BetterImageItem]| BetterImageItem):
        if isinstance(image, BetterImageItem):
            self._images.add(weakref.ref(image))
        else:
            for img in image:
                self._images.add(weakref.ref(img))
    
    def removeImage(self, image:Sequence[BetterImageItem] | BetterImageItem):
        if isinstance(image, BetterImageItem):
            self._images.remove(weakref.ref(image))
        else:
            for img in image:
                self._images.remove(weakref.ref(img))
        

    def setImageItem(
        self,
        image: Sequence[BetterImageItem] | BetterImageItem,
        insert_in: pg.PlotItem | None = None,
    ):  
        self.addImage(image)
        for img_ref in self._images:
            img = img_ref()
            if img is not None:
                if img.getColorMap() is not None:
                    self._primary_image = img_ref
                    break

        if self._primary_image is None:
            raise ValueError("ImageItem with a colormap was not found")
        self._primary_image().sigLimitChanged.connect(self.limit_changed)
        # self._primary_image().sigImageChanged.connect(self.limit_changed)
        # self._primary_image().sigColorChanged.connect(self.color_changed)
        self._primary_image().sigColorChanged.connect(self.limit_changed)
        # else:
        # print("hello")

        if insert_in is not None:
            insert_in.layout.addItem(self, 2, 5)
            insert_in.layout.setColumnFixedWidth(4, 5)

        self._span.blockSignals(True)
        self._span.setRegion(self.limits)
        self._span.blockSignals(False)
        self._span.sigRegionChanged.connect(self._level_change)
        self._span.sigRegionChangeFinished.connect(self._level_change_fin)
        self.color_changed()
        self.limit_changed()
        # self.isocurve.setParentItem(image)

    # def hideEvent(self, event: QtGui.QHideEvent):
    #     super().hideEvent(event)
    #     print("hide")

    # def showEvent(self, event: QtGui.QShowEvent):
    #     super().showEvent(event)
    #     # self._level_change()
    #     print("show")
    #     self.color_changed()
    #     self.limit_changed()

    # def setVisible(self, visible:bool, *args, **kwargs):
    # super().setVisible(visible, *args, **kwargs)
    # if visible:
    # self._level_change()
    # print('e')
    # self.isocurve.setVisible(visible, *args, **kwargs)

    def color_changed(self):
        if not self.isVisible():
            return
        cmap = self._primary_image()._colorMap
        lut = cmap.getStops()[1]
        if not self._colorbar.image.shape[0] == lut.shape[0]:
            self._colorbar.setImage(cmap.pos.reshape((-1, 1)))
        self._colorbar._colorMap = cmap
        self._colorbar.setLookupTable(lut, update=True)

    # def limit_changed(self, mn: float | None = None, mx: float | None = None):
    def limit_changed(self):
        if not self.isVisible():
            return
        self.color_changed()
        # if (self._fixedlimits is not None) or (mn is None):
        mn, mx = self.limits
        self._colorbar.setRect(0.0, mn, 1.0, mx - mn)
        if self.levels is not None:
            self._colorbar.setLevels((self.levels - mn) / (mx - mn))
        self._span.setBounds((mn, mx))

    # def cmap_changed(self):
    #     cmap = self.imageItem()._colorMap
    #     # lut = self.imageItem().lut
    #     # lut = cmap.getLookupTable(nPts=4096)
    #     # lut = self._eff_lut_for_image(self.imageItem())
    #     # if lut is None:
    #         # lut = self.imageItem()._effectiveLut
    #     # if lut is not None:
    #         # print(lut)
    #     # if lut is None:
    #     lut = cmap.getStops()[1]
    #     # if not self.npts == lut.shape[0]:
    #     # self.npts = lut.shape[0]
    #     if not self._colorbar.image.shape[0] == lut.shape[0]:
    #         self._colorbar.setImage(cmap.pos.reshape((-1, 1)))
    #     self._colorbar._colorMap = cmap
    #     self._colorbar.setLookupTable(lut)
    #     # self._colorbar.setColorMap(cmap)

    def mouseDragEvent(self, ev):
        ev.ignore()


class FittingParameterWidget(QtWidgets.QWidget):
    sigParamChanged = QtCore.Signal(dict)

    def __init__(
        self,
        name: str,
        spin_kw=dict(),
        checkable: bool = True,
        fixed: bool = False,
        label: str | None = None,
        show_label: bool = True,
    ):
        super().__init__()
        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.param_name = name
        self._prefix = ""
        if label is None:
            label = self.param_name
        self.label = QtWidgets.QLabel(label)
        self.spin_value = BetterSpinBox(**spin_kw)
        self.spin_lb = BetterSpinBox(
            value=-np.inf, minimumWidth=60, toolTip="Lower Bound"
        )
        self.spin_ub = BetterSpinBox(
            value=np.inf, minimumWidth=60, toolTip="Upper Bound"
        )
        self.check = QtWidgets.QCheckBox(toolTip="Fixed")

        if show_label:
            self.layout.addWidget(self.label)
        self.layout.addWidget(self.spin_value)
        self.layout.addWidget(self.spin_lb)
        self.layout.addWidget(self.spin_ub)
        self.layout.addWidget(self.check)

        for spin in (self.spin_value, self.spin_lb, self.spin_ub):
            spin.valueChanged.connect(
                lambda: self.sigParamChanged.emit(self.param_dict())
            )
        self.spin_lb.valueChanged.connect(self._refresh_bounds)
        self.spin_ub.valueChanged.connect(self._refresh_bounds)
        self.check.stateChanged.connect(self.setFixed)

        self.setCheckable(checkable)
        self.setFixed(fixed)

    def _refresh_bounds(self):
        self.spin_value.setRange(self.spin_lb.value(), self.spin_ub.value())

    def setValue(self, value):
        self.spin_value.setValue(value)

    def checkable(self):
        return self._checkable

    def setCheckable(self, value: bool):
        self._checkable = value
        self.check.setVisible(value)

    def fixed(self):
        if self.checkable():
            return self.check.isChecked()
        else:
            return False

    def setFixed(self, value: bool):
        if isinstance(value, QtCore.Qt.CheckState):
            if value == QtCore.Qt.CheckState.Unchecked:
                value = False
            elif value == QtCore.Qt.CheckState.Checked:
                value = True
        else:
            self.check.setChecked(value)
        if value:
            self.spin_lb.setDisabled(True)
            self.spin_ub.setDisabled(True)
        else:
            self.spin_lb.setEnabled(True)
            self.spin_ub.setEnabled(True)

    def value(self):
        return self.spin_value.value()

    def prefix(self):
        return self._prefix

    def set_prefix(self, prefix):
        self._prefix = prefix

    def minimum(self):
        return self.spin_lb.value()

    def maximum(self):
        return self.spin_ub.value()

    def param_dict(self):
        param_info = dict(value=self.value())
        if self.checkable():
            param_info["vary"] = not self.fixed()
        if np.isfinite(self.minimum()):
            param_info["min"] = float(self.minimum())
        if np.isfinite(self.maximum()):
            param_info["max"] = float(self.maximum())

        return {self.prefix() + self.param_name: param_info}


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
        "btspin": BetterSpinBox,
        "slider": QtWidgets.QSlider,
        "dblslider": QDoubleSlider,
        "chkbox": QtWidgets.QCheckBox,
        "pushbtn": QtWidgets.QPushButton,
        "chkpushbtn": QtWidgets.QPushButton,
        "combobox": QtWidgets.QComboBox,
        "fitparam": FittingParameterWidget,
    }
    VALID_QLTYPE = {
        "hbox": QtWidgets.QHBoxLayout,
        "vbox": QtWidgets.QVBoxLayout,
    }
    # !TODO: reimplement everything, add label by keyword argument. Apply more flexible
    # placements by adding layout support

    sigParameterChanged = QtCore.Signal(dict)

    @staticmethod
    def getParameterWidget(qwtype=None, **kwargs):
        """

        Parameters
        ----------
        qwtype : {'spin', 'dblspin', 'btspin', 'slider', 'dblslider', 'chkbox',
        'pushbtn', 'chkpushbtn', 'combobox', 'fitparam'}

        """
        if qwtype is None:
            widget = kwargs.pop("widget")
            if not isinstance(widget, QtWidgets.QWidget):
                raise ValueError("widget is not a valid QWidget")
            return widget
        elif qwtype not in ParameterGroup.VALID_QWTYPE.keys():
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

        if qwtype == "fitparam":
            show_param_label = kwargs.pop("show_param_label", False)
            kwargs["show_label"] = show_param_label

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

    def __init__(self, ncols: int = 1, groupbox_kw=dict(), **kwargs):
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
        self.setLayout(QtWidgets.QGridLayout(self))

        self.labels = []
        self.untracked = []
        self.widgets: dict[str, QtWidgets.QWidget] = dict()
        j = 0
        for i, (k, v) in enumerate(kwargs.items()):
            if isinstance(v, dict):
                showlabel = v.pop("showlabel", k)
                ind_eff = v.pop("colspan", 1)
                if ind_eff == "ncols":
                    ind_eff = ncols
                if v.pop("notrack", False):
                    self.untracked.append(k)
                self.widgets[k] = self.getParameterWidget(**v)

            elif isinstance(v, QtWidgets.QWidget):
                showlabel = k
                ind_eff = 1
                self.widgets[k] = v
            else:
                raise ValueError(
                    "Each value must be a QtWidgets.QWidget instance"
                    "or a dictionary of keyword arguments to getParameterWidget."
                )

            self.labels.append(QtWidgets.QLabel(str(showlabel)))
            self.labels[i].setBuddy(self.widgets[k])
            if showlabel:
                self.layout().addWidget(self.labels[i], j // ncols, 2 * (j % ncols))
                self.layout().addWidget(
                    self.widgets[k], j // ncols, 2 * (j % ncols) + 1, 1, 2 * ind_eff - 1
                )
            else:
                self.layout().addWidget(
                    self.widgets[k], j // ncols, 2 * (j % ncols), 1, 2 * ind_eff
                )
            j += ind_eff

        self.global_connect()

    def set_values(self, **kwargs):
        for k, v in kwargs.items():
            self.widgets[k].blockSignals(True)
            self.widgets[k].setValue(v)
            self.widgets[k].blockSignals(False)
        self.sigParameterChanged.emit(kwargs)

    def widget_value(self, widget: str | QtWidgets.QWidget):
        if isinstance(widget, str):
            widget = self.widgets[widget]
        if isinstance(
            widget,
            (
                QtWidgets.QSpinBox,
                QtWidgets.QDoubleSpinBox,
                BetterSpinBox,
                FittingParameterWidget,
            ),
        ):
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
        if isinstance(
            widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox, BetterSpinBox)
        ):
            return widget.valueChanged
        elif isinstance(widget, FittingParameterWidget):
            return widget.sigParamChanged
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
                    lambda x: self.sigParameterChanged.emit({k: x})
                )

    def widgets_of_type(self, widgetclass):
        if isinstance(widgetclass, str):
            widgetclass = self.VALID_QWTYPE[widgetclass]
        return [w for w in self.widgets.values() if isinstance(w, widgetclass)]

    @property
    def values(self) -> dict[str, float | int | bool]:
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
                qwtype="btspin",
                value=x0,
                valueChanged=lambda x: self.modify_roi(x0=x),
                minimum=xm,
                maximum=xM,
                **spinbox_kw,
            ),
            x1=dict(
                qwtype="btspin",
                value=x1,
                valueChanged=lambda x: self.modify_roi(x1=x),
                minimum=xm,
                maximum=xM,
                **spinbox_kw,
            ),
            y0=dict(
                qwtype="btspin",
                value=y0,
                valueChanged=lambda x: self.modify_roi(y0=x),
                minimum=ym,
                maximum=yM,
                **spinbox_kw,
            ),
            y1=dict(
                qwtype="btspin",
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
            pos = ev.pos()
            lastPos = ev.lastPos()
            dif = pos - lastPos

            mouseEnabled = np.array(vb.state["mouseEnabled"], dtype=np.float64)
            mask = mouseEnabled.copy()
            if axis is not None:
                mask[1 - axis] = 0.0

            if ev.button() & QtCore.Qt.MouseButton.LeftButton:
                if vb.state["mouseMode"] == pg.ViewBox.RectMode and axis is None:
                    if ev.isFinish():
                        vb.rbScaleBox.hide()
                        ax = QtCore.QRectF(
                            pg.Point(ev.buttonDownPos(ev.button())), pg.Point(pos)
                        )
                        ax = vb.childGroup.mapRectFromParent(ax)
                        self.modify_roi(*ax.getCoords())
                    else:
                        vb.updateScaleBox(ev.buttonDownPos(), ev.pos())
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
    def __init__(
        self,
        data,
        title=None,
        layout="horizontal",
        data_is_input=True,
        analysisWidget=None,
        *args,
        **kwargs,
    ):
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

        if analysisWidget is None:
            self.aw = AnalysisWidgetBase(*args, **kwargs)
        elif isinstance(analysisWidget, type):
            self.aw = analysisWidget(*args, **kwargs)
        else:
            self.aw = analysisWidget

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

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        cb = QtWidgets.QApplication.instance().clipboard()
        if cb.text(cb.Mode.Clipboard) != "":
            pyclip.copy(cb.text(cb.Mode.Clipboard))
        return super().closeEvent(event)


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
        super().__init__(*args, num_ax=2, **kwargs)
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
    win = AnalysisWindow(dat, analysisWidget=ComparisonWidget, orientation="vertical")

    # win.set_pre_function(gaussian_filter, sigma=[1, 1], only_values=True)
    # win.set_pre_function(gaussian_filter, sigma=(0.1, 0.1))

    gaussfilt_2d = lambda dat, sx, sy: gaussian_filter(dat, sigma=(sx, sy))
    win.aw.set_main_function(gaussfilt_2d, sx=0.1, sy=1, only_values=True)

    layout.addWidget(win)
    layout.addWidget(
        ParameterGroup(
            sigma_x=dict(
                qwtype="btspin",
                minimum=0,
                maximum=10,
                valueChanged=lambda x: win.aw.set_main_function_args(sx=x),
            ),
            sigma_y=dict(
                qwtype="btspin",
                minimum=0,
                maximum=10,
                valueChanged=lambda x: win.aw.set_main_function_args(sy=x),
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
