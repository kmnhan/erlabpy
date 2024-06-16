"""
.. deprecated:: 0.1.

    This module is deprecated, and is only kept for reference purposes.
    Use `erlab.interactive.imagetool` instead.

"""

import colorsys
import enum
import importlib
import sys
import weakref
from itertools import chain, compress
from time import perf_counter

import matplotlib.mathtext
import numba
import numpy as np
import pyqtgraph as pg
import qtawesome as qta
from matplotlib import colors as mcolors
from matplotlib import figure, rc_context
from matplotlib.backends import backend_agg, backend_svg
from matplotlib.font_manager import FontProperties
from pyqtgraph.dockarea.Dock import Dock, DockLabel
from pyqtgraph.dockarea.DockArea import DockArea
from qtpy import QtCore, QtGui, QtSvg, QtSvgWidgets, QtWidgets

from erlab.interactive.colors import (
    pg_colormap_names,
    pg_colormap_powernorm,
    pg_colormap_to_QPixmap,
)
from erlab.interactive.utils import parse_data, xImageItem

# pg.setConfigOption('useNumba', True)
# pg.setConfigOption('background', 'w')
# pg.setConfigOption('foreground', 'k')

if importlib.util.find_spec("numbagg"):
    import numbagg

    _general_nanmean_func = numbagg.nanmean
else:
    _general_nanmean_func = np.nanmean

__all__ = ["itool_", "pg_itool"]

suppressnanwarning = np.testing.suppress_warnings()
suppressnanwarning.filter(RuntimeWarning, r"All-NaN (slice|axis) encountered")


ICON_NAME = {
    "invert": "mdi6.invert-colors",
    "invert_off": "mdi6.invert-colors-off",
    "contrast": "mdi6.contrast-box",
    "lock": "mdi6.lock",
    "unlock": "mdi6.lock-open-variant",
    "colorbar": "mdi6.gradient-vertical",
    "transpose_0": "mdi6.arrow-left-right",
    "transpose_1": "mdi6.arrow-top-left-bottom-right",
    "transpose_2": "mdi6.arrow-up-down",
    "transpose_3": "mdi6.arrow-up-down",
    "snap": "mdi6.grid",
    "snap_off": "mdi6.grid-off",
    "palette": "mdi6.palette-advanced",
    "styles": "mdi6.palette-swatch",
    "layout": "mdi6.page-layout-body",
    "zero_center": "mdi6.format-vertical-align-center",
    "table_eye": "mdi6.table-eye",
}


class IconButton(QtWidgets.QPushButton):
    def __init__(self, *args, on: str | None = None, off: str | None = None, **kwargs):
        self.icon_key_on = None
        self.icon_key_off = None
        if on is not None:
            self.icon_key_on = on
            kwargs["icon"] = qta.icon(ICON_NAME[self.icon_key_on])
        if off is not None:
            if on is None and kwargs["icon"] is None:
                raise ValueError("Icon for `on` state was not supplied.")
            self.icon_key_off = off
            kwargs.setdefault("checkable", True)
        super().__init__(*args, **kwargs)
        if self.isCheckable() and off is not None:
            self.toggled.connect(self.refresh_icons)

    def refresh_icons(self):
        if self.icon_key_off is not None:
            if self.isChecked():
                self.setIcon(qta.icon(ICON_NAME[self.icon_key_off]))
                return
        self.setIcon(qta.icon(ICON_NAME[self.icon_key_on]))

    def changeEvent(self, evt):
        if evt.type() == QtCore.QEvent.PaletteChange:
            qta.reset_cache()
            self.refresh_icons()
        super().changeEvent(evt)


# class FlowLayout(QtGui.QVBoxLayout):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)


class FlowLayout(QtWidgets.QLayout):
    def __init__(self, parent=None, margin=0):
        super().__init__(parent)

        # if parent is not None:
        # self.setContentsMargins(QtCore.QMargins(margin, margin, margin, margin))

        self._item_list = []
        self.setHorizontalSpacing(self.spacing())
        self.setVerticalSpacing(self.spacing())
        # self.setVerticalSpacing(0)

    def __del__(self):
        item = self.takeAt(0)
        while item:
            item = self.takeAt(0)

    def horizontalSpacing(self):
        return self._spacing_horizontal

    def verticalSpacing(self):
        return self._spacing_vertical

    def setHorizontalSpacing(self, spacing: int):
        self._spacing_horizontal = spacing

    def setVerticalSpacing(self, spacing: int):
        self._spacing_vertical = spacing

    def setSpacing(self, spacing: int):
        super().setSpacing(spacing)
        self.setHorizontalSpacing(spacing)
        self.setVerticalSpacing(spacing)

    def addItem(self, item):
        self._item_list.append(item)

    def count(self):
        return len(self._item_list)

    def itemAt(self, index):
        if 0 <= index < len(self._item_list):
            return self._item_list[index]

        return None

    def takeAt(self, index):
        if 0 <= index < len(self._item_list):
            return self._item_list.pop(index)

        return None

    def expandingDirections(self):
        return QtCore.Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        height = self._do_layout(QtCore.QRect(0, 0, width, 0), True)
        return height

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QtCore.QSize()

        for item in self._item_list:
            size = size.expandedTo(item.minimumSize())

        size += QtCore.QSize(
            2 * self.contentsMargins().top(), 2 * self.contentsMargins().top()
        )
        return size

    def _do_layout(self, rect, test_only):
        x = rect.x()
        y = rect.y()
        line_height = 0

        for item in self._item_list:
            style = item.widget().style()
            layout_spacing_x = style.layoutSpacing(
                QtWidgets.QSizePolicy.PushButton,
                QtWidgets.QSizePolicy.PushButton,
                QtCore.Qt.Horizontal,
            )
            layout_spacing_y = style.layoutSpacing(
                QtWidgets.QSizePolicy.PushButton,
                QtWidgets.QSizePolicy.PushButton,
                QtCore.Qt.Vertical,
            )
            space_x = self.horizontalSpacing() + layout_spacing_x
            space_y = self.verticalSpacing() + layout_spacing_y
            next_x = x + item.sizeHint().width() + space_x
            if next_x - space_x > rect.right() and line_height > 0:
                x = rect.x()
                y = y + line_height + space_y
                next_x = x + item.sizeHint().width() + space_x
                line_height = 0

            if not test_only:
                item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), item.sizeHint()))

            x = next_x
            line_height = max(line_height, item.sizeHint().height())

        return y + line_height - rect.y()


class InnerQHBoxLayout(QtWidgets.QHBoxLayout):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setContentsMargins(0, 0, 0, 0)


class InnerQGridLayout(QtWidgets.QGridLayout):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setContentsMargins(0, 0, 0, 0)
        # self.setSpacing(1)


class BorderlessGroupBox(QtWidgets.QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        name = self.objectName()
        if name == "":
            raise ValueError(
                "GroupBox has no objectName. Supply an "
                "`objectName` argument to set a new name."
            )
        # self.setStyleSheet("QGroupBox#" + name + " {border:0;}")
        self.setContentsMargins(0, 0, 0, 0)


def qt_style_names():
    """Return a list of styles, default platform style first."""
    default_style_name = QtWidgets.QApplication.style().objectName().lower()
    result = []
    for style in QtWidgets.QStyleFactory.keys():
        if style.lower() == default_style_name:
            result.insert(0, style)
        else:
            result.append(style)
    return result


def change_style(style_name):
    QtWidgets.QApplication.setStyle(QtWidgets.QStyleFactory.create(style_name))


# @numba.njit(nogil=True, cache=True, fastmath={'nnan','ninf', 'nsz', 'contract', 'reassoc', 'afn', 'arcp'})
@numba.njit(
    nogil=True,
    cache=True,
    fastmath={"ninf", "nsz", "contract", "reassoc", "afn", "arcp"},
)
# @numba.njit(nogil=True, cache=True)
def move_mean1d(a, window, min_count):
    out = np.empty_like(a)
    asum = 0.0
    count = 0

    for i in range(min_count - 1):
        ai = a[i]
        if not np.isnan(ai):
            asum += ai
            count += 1
        out[i] = np.nan

    for i in range(min_count - 1, window):
        ai = a[i]
        if not np.isnan(ai):
            asum += ai
            count += 1
        out[i] = asum / count if count >= min_count else np.nan

    count_inv = 1 / count if count >= min_count else np.nan
    for i in range(window, len(a)):
        ai = a[i]
        aold = a[i - window]

        ai_valid = not np.isnan(ai)
        aold_valid = not np.isnan(aold)

        if ai_valid and aold_valid:
            asum += ai - aold
        elif ai_valid:
            asum += ai
            count += 1
            count_inv = 1 / count if count >= min_count else np.nan
        elif aold_valid:
            asum -= aold
            count -= 1
            count_inv = 1 / count if count >= min_count else np.nan

        out[i] = asum * count_inv
    return out


@numba.njit(nogil=True, parallel=True, cache=True)
def move_mean2d(a, window_list, min_count_list):
    ii, jj = a.shape
    for i in numba.prange(ii):
        a[i, :] = move_mean1d(a[i, :], window_list[0], min_count_list[0])
    for j in numba.prange(jj):
        a[:, j] = move_mean1d(a[:, j], window_list[1], min_count_list[1])
    return a


@numba.njit(nogil=True, parallel=True, cache=True)
def move_mean3d(a, window_list, min_count_list):
    ii, jj, kk = a.shape
    for i in numba.prange(ii):
        for k in numba.prange(kk):
            a[i, :, k] = move_mean1d(a[i, :, k], window_list[1], min_count_list[1])
    for i in numba.prange(ii):
        for j in numba.prange(jj):
            a[i, j, :] = move_mean1d(a[i, j, :], window_list[2], min_count_list[2])
    for j in numba.prange(jj):
        for k in numba.prange(kk):
            a[:, j, k] = move_mean1d(a[:, j, k], window_list[0], min_count_list[0])
    return a


def get_true_indices(a):
    return list(compress(range(len(a)), a))


# https://www.pythonguis.com/widgets/qcolorbutton-a-color-selector-tool-for-pyqt/
class ColorButton(QtWidgets.QPushButton):
    """
    Custom Qt Widget to show a chosen color.

    Left-clicking the button shows the color-chooser, while
    right-clicking resets the color to None (no-color).
    """

    colorChanged = QtCore.Signal(object)

    def __init__(self, *args, color=None, **kwargs):
        super().__init__(*args, **kwargs)

        self._color = None
        self._default = color
        self.pressed.connect(self.onColorPicker)

        # Set the initial/default state.
        self.setColor(self._default)

    def setColor(self, color):
        self._color = color
        self.colorChanged.emit(color.getRgbF())
        if self._color:
            self.setStyleSheet(
                f"QWidget {{ background-color: {self._color.name(QtGui.QColor.HexArgb)}; border: 0; }}"
            )
        else:
            self.setStyleSheet("")

    def color(self):
        return self._color

    def onColorPicker(self):
        """
        Show color-picker dialog to select color.

        Qt will use the native dialog by default.

        """
        dlg = QtWidgets.QColorDialog(self)
        if self._color:
            dlg.setCurrentColor(QtGui.QColor(self._color))
        if dlg.exec_():
            self.setColor(dlg.currentColor())

    def mousePressEvent(self, e):
        if e.button() == QtCore.Qt.RightButton:
            self.setColor(self._default)
        return super().mousePressEvent(e)


class ItoolImageItem(xImageItem):
    def __init__(self, itool, *args, **kargs):
        self.itool = itool
        super().__init__(*args, **kargs)


class ItoolDockLabel(DockLabel):
    def __init__(self, *args, color="#591e71", **kwargs):
        self.bg_color = mcolors.to_hex(color)
        super().__init__(*args, **kwargs)

    def dim_color(self, color, l_factor=1.0, s_factor=1.0):
        h, l, s = colorsys.rgb_to_hls(*mcolors.to_rgb(color))  # noqa: E741
        return QtGui.QColor.fromRgbF(
            *colorsys.hls_to_rgb(h, min(1, l * l_factor), min(1, s * s_factor))
        ).name()

    def set_fg_color(self):
        rgb = list(mcolors.to_rgb(self.bg_color))
        # for i in range(3):
        #     if rgb[i] <= 0.04045:
        #         rgb[i] /= 12.92
        #     else:
        #         rgb[i] = ((rgb[i] + 0.055) / 1.055) ** 2.4
        # L = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
        L = rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114
        # if L > 0.179:
        if L > 0.729:
            self.fg_color = "#000000"
        else:
            self.fg_color = "#ffffff"

    def updateStyle(self):
        r = "3px"
        self.set_fg_color()
        if self.dim:
            self.dim_l_factor = 0.8
            fg = self.dim_color(self.fg_color, self.dim_l_factor, 0.8)
            bg = self.dim_color(self.bg_color, self.dim_l_factor, 0.8)
        else:
            fg = self.fg_color
            bg = self.bg_color
        border = self.dim_color(bg, 0.9)

        if self.orientation == "vertical":
            self.vStyle = f"""DockLabel {{
                background-color : {bg};
                color : {fg};
                border-top-right-radius: 0px;
                border-top-left-radius: {r};
                border-bottom-right-radius: 0px;
                border-bottom-left-radius: {r};
                border-width: 0px;
                border-right: 2px solid {border};
                padding-top: 3px;
                padding-bottom: 3px;
                font-size: {self.fontSize};
            }}"""
            self.setStyleSheet(self.vStyle)
        else:
            self.hStyle = f"""DockLabel {{
                background-color : {bg};
                color : {fg};
                border-top-right-radius: {r};
                border-top-left-radius: {r};
                border-bottom-right-radius: 0px;
                border-bottom-left-radius: 0px;
                border-width: 0px;
                border-bottom: 2px solid {border};
                padding-left: 3px;
                padding-right: 3px;
                font-size: {self.fontSize};
            }}"""
            self.setStyleSheet(self.hStyle)


class ItoolDock(Dock):
    def __init__(
        self,
        name,
        area=None,
        size=(10, 10),
        widget=None,
        hideTitle=False,
        autoOrientation=True,
        closable=False,
        fontSize="13px",
        color="#591e71",
    ):
        super().__init__(
            name,
            area=area,
            size=size,
            widget=widget,
            hideTitle=hideTitle,
            autoOrientation=autoOrientation,
            closable=closable,
            fontSize=fontSize,
            label=ItoolDockLabel(name, closable, fontSize, color=color),
        )
        # self.label.setVisible(False)
        # self.label.dock = self
        # if closable:
        # self.label.sigCloseClicked.connect(self.close)
        # self.topLayout.addWidget(self.label, 0, 1)
        self.topLayout.setContentsMargins(0, 0, 0, 0)

    def changeEvent(self, evt):
        if evt.type() == QtCore.QEvent.PaletteChange:
            self.label.updateStyle()
        super().changeEvent(evt)


def get_pixmap_label(s: str, prop=None, dpi=300, **text_kw):
    """Create a QtGui.QPixmap from a mathtext string.

    Parameters
    ----------
    s : str
        Mathtext string to be rendered.
    prop : matplotlib.font_manager.FontProperties
        Font properties.
    dpi : float, optional (default=300)
        Dots per inch of the created pixmap.
    **text_kw : dict, optional
        Extra arguments to `matplotlib.figure.Figure.text`: refer to the
        `matplotlib` documentation for a list of all possible arguments.

    Returns
    -------
    A QtGui.QPixmap object.

    """
    parser = matplotlib.mathtext.MathTextParser("path")
    if prop is None:
        prop = FontProperties(size=9)
    width, height, depth, _, _ = parser.parse(s, dpi=72, prop=prop)
    fig = figure.Figure(figsize=(width / 72.0, height / 72.0), dpi=dpi)
    fig.patch.set_facecolor("none")
    text_kw["fontproperties"] = prop
    fig.text(0, depth / height, s, **text_kw)

    backend_agg.FigureCanvasAgg(fig)
    buf, size = fig.canvas.print_to_buffer()
    img = QtGui.QImage(buf, size[0], size[1], QtGui.QImage.Format_ARGB32)
    img.setDevicePixelRatio(fig._dpi / 100.0)
    pixmap = QtGui.QPixmap(img.rgbSwapped())
    return pixmap


def get_svg_label(
    s: str, outfile: QtCore.QTemporaryFile, prop=None, dpi=300, **text_kw
):
    """Create an SVG image from a mathtext string.

    Parameters
    ----------
    s : str
        Mathtext string to be rendered.
    outfile : QtCore.QTemporaryFile
        Output temp file to store the SVG.
    prop : matplotlib.font_manager.FontProperties
        Font properties.
    dpi = 300
        dpi : float, optional (default=300)
    **text_kw : dict, optional
        Extra arguments to `matplotlib.figure.Figure.text`: refer to the
        `matplotlib` documentation for a list of all possible arguments.

    Returns
    -------
    filename : str
        Name of the output file containing the rendered SVG.

    """
    parser = matplotlib.mathtext.MathTextParser("path")
    if prop is None:
        prop = FontProperties(size=12)
    width, height, depth, _, _ = parser.parse(s, dpi=1000, prop=prop)
    fig = figure.Figure(figsize=(width / 1000.0, height / 1000.0), dpi=dpi)
    fig.patch.set_facecolor("none")
    text_kw["fontproperties"] = prop
    fig.text(0, depth / height, s, **text_kw)

    backend_svg.FigureCanvasSVG(fig)
    if outfile.open():
        fig.canvas.print_svg(outfile.fileName())
    return outfile.fileName()


class ItoolAxisItem(pg.AxisItem):
    class LabelType(enum.Flag):
        TextLabel = 0
        SvgLabel = 1
        PixmapLabel = 2

    def __init__(self, *args, **kwargs):
        self.label_mode = self.LabelType.TextLabel
        super().__init__(*args, **kwargs)
        self.set_label_mode(self.LabelType.TextLabel)

    def set_label_mode(self, labelmode: LabelType):
        self.label_mode = labelmode
        match self.label_mode:
            case self.LabelType.TextLabel:
                if not isinstance(self.label, QtWidgets.QGraphicsTextItem):
                    self.label = QtWidgets.QGraphicsTextItem(self)
            case self.LabelType.SvgLabel:
                if not isinstance(self.label, QtSvgWidgets.QGraphicsSvgItem):
                    self.label = QtSvgWidgets.QGraphicsSvgItem(self)
            case self.LabelType.PixmapLabel:
                if not isinstance(self.label, QtWidgets.QGraphicsPixmapItem):
                    self.label = QtWidgets.QGraphicsPixmapItem(self)
                    self.label.setTransformationMode(QtCore.Qt.SmoothTransformation)

        if self.orientation in ["left", "right"]:
            self.label.setRotation(-90)

    def mathtextLabelPixmap(self, file=None):
        if self.labelUnits == "":
            if not self.autoSIPrefix or self.autoSIPrefixScale == 1.0:
                units = ""
            else:
                units = "(x%g)" % (1.0 / self.autoSIPrefixScale)
        else:
            units = f"({self.labelUnitPrefix}{self.labelUnits})"

        s = f"{self.labelText} {units}"

        match self.label_mode:
            case self.LabelType.SvgLabel:
                return get_svg_label(s, file, **self.labelStyle)
            case self.LabelType.PixmapLabel:
                return get_pixmap_label(s, **self.labelStyle)

    def _updateLabel(self):
        match self.label_mode:
            case self.LabelType.TextLabel:
                self.label.setHtml(self.labelString())
            case self.LabelType.SvgLabel:
                file = QtCore.QTemporaryFile()
                self.svg_renderer = QtSvg.QSvgRenderer(self.mathtextLabelPixmap(file))
                self.label.setSharedRenderer(self.svg_renderer)
            case self.LabelType.PixmapLabel:
                self.label.setPixmap(self.mathtextLabelPixmap())

        self._adjustSize()
        self.picture = None
        self.update()

    def resizeEvent(self, ev=None):
        # s = self.size()

        # Set the position of the label
        # match self.label_mode:
        #     case self.LabelType.TextLabel:
        #         nudge = 5
        #     case self.LabelType.SvgLabel:
        #         nudge = -3
        #     case self.LabelType.PixmapLabel:
        #         nudge = 0
        if self.label_mode == self.LabelType.TextLabel:
            nudge = 5
        elif self.label_mode == self.LabelType.SvgLabel:
            nudge = -3
        if self.label_mode == self.LabelType.PixmapLabel:
            nudge = 0

        if (
            self.label is None
        ):  # self.label is set to None on close, but resize events can still occur.
            self.picture = None
            return

        br = self.label.boundingRect()
        p = QtCore.QPointF(0, 0)
        if self.orientation == "left":
            p.setY(int(self.size().height() / 2 + br.width() / 2))
            p.setX(-nudge)
        elif self.orientation == "right":
            p.setY(int(self.size().height() / 2 + br.width() / 2))
            p.setX(int(self.size().width() - br.height() + nudge))
        elif self.orientation == "top":
            p.setY(-nudge)
            p.setX(int(self.size().width() / 2.0 - br.width() / 2.0))
        elif self.orientation == "bottom":
            p.setX(int(self.size().width() / 2.0 - br.width() / 2.0))
            p.setY(int(self.size().height() - br.height() + nudge))
        self.label.setPos(p)
        self.picture = None


class ItoolPlotItem(pg.PlotItem):
    def __init__(self, itool, *args, **kargs):
        self.itool = itool
        super().__init__(*args, **kargs)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        # self.ctrl.transformGroup.setVisible(False)
        for action in self.ctrlMenu.actions():
            if action.text() in [
                "Transforms",
                "Downsample",
                "Average",
                "Alpha",
                "Points",
            ]:
                action.setVisible(False)

        for i in (0, 1):
            self.vb.menu.ctrl[i].linkCombo.setVisible(False)
            self.vb.menu.ctrl[i].label.setVisible(False)
        # self.ctrlMenu.menuAction().setVisible(False)
        # self.setMenuEnabled(False, enableViewBoxMenu=None)

    def mouseDragEvent(self, ev):
        if (
            self.itool.qapp.queryKeyboardModifiers() == QtCore.Qt.ControlModifier
            and ev.button() == QtCore.Qt.MouseButton.LeftButton
        ):
            ev.accept()
            self.itool.onMouseDrag(ev)
        else:
            ev.ignore()

    def setLabels(
        self, mode: ItoolAxisItem.LabelType = ItoolAxisItem.LabelType.TextLabel, **kwds
    ):
        for k in kwds.keys():
            if k != "title":
                self.getAxis(k).set_label_mode(mode)
        super().setLabels(**kwds)

    def setAxisItems(self, axisItems=None):
        if axisItems is None:
            axisItems = {}

        visibleAxes = ["left", "bottom"]
        visibleAxes.extend(axisItems.keys())

        for k, pos in (
            ("top", (1, 1)),
            ("bottom", (3, 1)),
            ("left", (2, 0)),
            ("right", (2, 2)),
        ):
            if k in self.axes:
                if k not in axisItems:
                    continue

                oldAxis = self.axes[k]["item"]
                self.layout.removeItem(oldAxis)
                oldAxis.scene().removeItem(oldAxis)
                oldAxis.unlinkFromView()

            if k in axisItems:
                axis = axisItems[k]
                if axis.scene() is not None:
                    if k not in self.axes or axis != self.axes[k]["item"]:
                        raise RuntimeError(
                            "Can't add an axis to multiple plots. Shared axes"
                            " can be achieved with multiple AxisItem instances"
                            " and set[X/Y]Link."
                        )
            else:
                axis = ItoolAxisItem(orientation=k, parent=self)

            axis.linkToView(self.vb)
            self.axes[k] = {"item": axis, "pos": pos}
            self.layout.addItem(axis, *pos)

            axis.setZValue(0.5)
            axis.setFlag(axis.GraphicsItemFlag.ItemNegativeZStacksBehindParent)
            axisVisible = k in visibleAxes
            self.showAxis(k, axisVisible)


class ItoolCursorLine(pg.InfiniteLine):
    def __init__(self, itool, *args, **kargs):
        self.itool = itool
        super().__init__(*args, **kargs)

    def mouseDragEvent(self, ev):
        if self.itool.qapp.queryKeyboardModifiers() != QtCore.Qt.ControlModifier:
            self.setMovable(True)
            super().mouseDragEvent(ev)
        else:
            self.setMovable(False)
            self.setMouseHover(False)
            ev.ignore()

    def mouseClickEvent(self, ev):
        if self.itool.qapp.queryKeyboardModifiers() != QtCore.Qt.ControlModifier:
            self.setMovable(True)
            super().mouseClickEvent(ev)
        else:
            self.setMovable(False)
            self.setMouseHover(False)
            ev.ignore()

    def hoverEvent(self, ev):
        if self.itool.qapp.queryKeyboardModifiers() != QtCore.Qt.ControlModifier:
            self.setMovable(True)
            super().hoverEvent(ev)
        else:
            self.setMovable(False)
            self.setMouseHover(False)


class pg_itool(pg.GraphicsLayoutWidget):
    """A interactive tool based on :obj:`pyqtgraph` for exploring 3D data.

    For the tool to remain responsive you must
    keep a reference to it.

    Parameters
    ----------
    data : `xarray.DataArray`
        The data to explore. Must have three coordinate axes.

    snap :  bool, default: True
        Wheter to snap the cursor to data pixels.

    gamma : float, default: 0.5
        Colormap default gamma.
    cmap : str or `pyqtgraph.ColorMap`, default: 'magma'
        Default colormap.

    bench : bool, default: False
        Whether to print frames per second.

    plot_kw : dict, optional
        Extra arguments to `matplotlib.pyplot.plot`: refer to the
        `matplotlib` documentation for a list of all possible arguments.
    cursor_kw : dict, optional
        Extra arguments to `pyqtgraph.InfiniteLine`: refer to the
        `pyqtgraph` documentation for a list of all possible arguments.
    image_kw : dict, optional
        Extra arguments to `pyqtgraph.ImageItem`: refer to the
        `pyqtgraph` documentation for a list of all possible arguments.
    profile_kw : dict, optional
        Extra arguments to `PlotDataItem.__init__`: refer to the
        `pyqtgraph` documentation for a list of all possible arguments.
    span_kw : dict, optional
        Extra arguments to `pyqtgraph.LinearRegionItem`: refer to the
        `pyqtgraph` documentation for a list of all possible arguments.
    fermi_kw : dict, optional
        Extra arguments to `pyqtgraph.InfiniteLine`: refer to the
        `pyqtgraph` documentation for a list of all possible arguments.

    Note
    ----
    Axes indices for 2D data:

    .. code-block:: text

        ┌───┬───┐
        │ 1 │   │
        │───┼───│
        │ 0 │ 2 │
        └───┴───┘

    Axes indices for 3D data:

    .. code-block:: text

        ┌───┬─────┐
        │ 1 │     │
        │───┤  3  │
        │ 4 │     │
        │───┼───┬─│
        │ 0 │ 5 │2│
        └───┴───┴─┘

    Axes indices for 4D data:

    .. code-block:: text

        ┌───┬─────┐
        │ 1 │  6  │
        │───┼─────│
        │ 4 │  3  │
        │───┼───┬─│
        │ 0 │ 5 │2│
        └───┴───┴─┘

    Signals
    -------
    sigDataChanged()
    sigIndexChanged(indices, values)

    """

    sigDataChanged = QtCore.Signal(object)  #: :meta private:
    sigIndexChanged = QtCore.Signal(list, list)  #: :meta private:

    _only_axis = ("x", "y", "z", "t")
    _only_maps = "maps"

    def _get_middle_index(self, x):
        return len(x) // 2 - (1 if len(x) % 2 == 0 else 0)

    def __init__(
        self,
        data,
        snap=False,
        gamma=1.0,
        cmap="magma",
        reverse=False,
        bench=False,
        plot_kw=None,
        cursor_kw=None,
        image_kw=None,
        profile_kw=None,
        span_kw=None,
        fermi_kw=None,
        zero_centered=False,
        rad2deg=False,
        **kwargs,
    ):
        if fermi_kw is None:
            fermi_kw = {}
        if span_kw is None:
            span_kw = {}
        if profile_kw is None:
            profile_kw = {}
        if image_kw is None:
            image_kw = {}
        if cursor_kw is None:
            cursor_kw = {}
        if plot_kw is None:
            plot_kw = {}
        super().__init__(show=True, **kwargs)
        self.qapp = QtCore.QCoreApplication.instance()
        self.screen = self.qapp.primaryScreen()
        self.snap = snap
        self.gamma = gamma
        self.cmap = cmap
        self.reverse = reverse
        self.bench = bench
        if self.bench:
            self._fpsLastUpdate = perf_counter()
            self._avg_fps = 0.0
        self.colorbar = None
        self.plot_kw = plot_kw
        self.cursor_kw = cursor_kw
        self.image_kw = image_kw
        self.profile_kw = profile_kw
        self.span_kw = span_kw
        self.fermi_kw = fermi_kw
        self.zero_centered = zero_centered
        if self.zero_centered:
            self.gamma = 1.0
            self.cmap = "bwr"
        # cursor_c = pg.mkColor(0.5)
        cursor_c, cursor_c_hover, span_c, span_c_edge = (
            pg.mkColor("gray") for _ in range(4)
        )
        cursor_c.setAlphaF(0.75)
        cursor_c_hover.setAlphaF(0.95)
        span_c.setAlphaF(0.15)
        span_c_edge.setAlphaF(0.35)
        # span_c_hover = pg.mkColor(0.75)
        # span_c_hover.setAlphaF(0.5)

        self.cursor_kw.update(
            {
                "pen": pg.mkPen(cursor_c, width=2.25),
                "hoverPen": pg.mkPen(cursor_c_hover, width=2.5),
            }
        )
        self.plot_kw.update({"defaultPadding": 0.0, "clipToView": False})
        # self.profile_kw.update(dict(
        #     linestyle='-', linewidth=.8,
        #     color=colors.to_rgba(plt.rcParams.get('axes.edgecolor'),
        #                          alpha=1),
        #     animated=self.useblit, visible=True,
        # ))
        # self.fermi_kw.update(dict(
        #     linestyle='--', linewidth=.8,
        #     color=colors.to_rgba(plt.rcParams.get('axes.edgecolor'),
        #                          alpha=1),
        #     animated=False,
        # ))
        self.image_kw.update(
            {
                "autoDownsample": False,
                "axisOrder": "row-major",
            }
        )
        self.span_kw.update(
            {
                "movable": False,
                "pen": pg.mkPen(span_c_edge, width=1),
                "brush": pg.mkBrush(span_c),
            }
        )
        # self.cursor_pos = None
        self.data = None
        if rad2deg is not False:
            if np.iterable(rad2deg):
                conv_dims = rad2deg
            else:
                conv_dims = [
                    d
                    for d in ["phi", "theta", "beta", "alpha", "chi"]
                    if d in data.dims
                ]
            self.set_data(
                data.assign_coords({d: np.rad2deg(data[d]) for d in conv_dims}),
                update_all=True,
                reset_cursor=True,
            )
        else:
            self.set_data(data, update_all=True, reset_cursor=True)
        self.set_cmap()
        self.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.setFocus()
        self.connect_signals()

    def _update_stretch(self, row=None, col=None):
        if row is None:
            if self.data_ndim == 2:
                row_factor = (250000, 750000)
            elif self.data_ndim >= 3:
                row_factor = (100000, 150000, 300000)
        else:
            row_factor = row
        if col is None:
            col_factor = row_factor
        else:
            col_factor = col

        self._stretch_factors = (row_factor, col_factor)

        # for i in range(self.ci.layout.rowCount()):
        #     for j in range(self.ci.layout.columnCount()):
        #         item = self.ci.getItem(i, j)
        #         if item is not None:
        #             item.setSizePolicy(
        #                 QtWidgets.QSizePolicy(
        #                     QtWidgets.QSizePolicy.Policy.Expanding,
        #                     QtWidgets.QSizePolicy.Policy.Expanding,
        #                 )
        #             )

        # for i, p in enumerate(self.axes):
        # if i in
        # self.axes[1].setSizePolicy(
        #     QtWidgets.QSizePolicy.Policy.Ignored,
        #     QtWidgets.QSizePolicy.Policy.Ignored,
        # )
        # self.axes[2].setSizePolicy(
        #     QtWidgets.QSizePolicy.Policy.Preferred,
        #     QtWidgets.QSizePolicy.Policy.Preferred,
        # )
        #     for axis in ["left", "bottom", "right", "top"]:
        #             ax = p.getAxis(axis)
        #     p.setMinimumSize(1e-4, 1e-4)
        #     p.setMaximumSize(self.ci.width(), self.ci.height())
        # p.setSizePolicy(
        #     QtWidgets.QSizePolicy(
        #         QtWidgets.QSizePolicy.Policy.Ignored,
        #         QtWidgets.QSizePolicy.Policy.Ignored,
        #     )
        # )
        for i in range(self.ci.layout.rowCount()):
            self.ci.layout.setRowPreferredHeight(
                i,
                self.ci.height() * row_factor[i] / np.sum(row_factor),
            )
            self.ci.layout.setRowStretchFactor(i, row_factor[i])
        for j in range(self.ci.layout.columnCount()):
            col_index = (-j - 1) % len(col_factor)
            self.ci.layout.setColumnPreferredWidth(
                j,
                self.ci.width() * col_factor[col_index] / np.sum(col_factor),
            )
            self.ci.layout.setColumnStretchFactor(j, col_factor[col_index])
        #         self.ci.layout.setColumnMinimumWidth(j, 0)
        #         self.ci.layout.setColumnMaximumWidth(j, self.ci.width())

        #     # item.setMinimumSize(1, 1)
        #     item.setMaximumSize(self.ci.width(), self.ci.height())

        # elif factor == 0:
        # for i in range(self.ci.layout.columnCount()):
        # self.ci.layout.setColumnStretchFactor(i, 0)
        # self.ci.layout.setColumnAlignment(i, QtCore.Qt.AlignCenter)
        # for i in range(self.ci.layout.rowCount()):
        # self.ci.layout.setRowAlignment(i, QtCore.Qt.AlignCenter)
        # self.ci.layout.setRowStretchFactor(i, 0)
        #     return
        # self.ci.setSizePolicy(
        #     QtWidgets.QSizePolicy(
        #         QtWidgets.QSizePolicy.Policy.Ignored,
        #         QtWidgets.QSizePolicy.Policy.Ignored,
        #     )
        # )
        # for i in range(self.ci.layout.columnCount()):

        # self.ci.layout.setColumnMaximumWidth(i, self.ci.width())
        # self.ci.layout.setColumnPreferredWidth(i, -10)
        # for i in range(self.ci.layout.rowCount()):
        # self.ci.layout.setRowMaximumHeight(i, self.ci.height())
        # self.ci.layout.setRowPreferredHeight(i, -10)

    def _initialize_layout(
        self, horiz_pad=45, vert_pad=30, inner_pad=15, font_size=11.0
    ):
        font = QtGui.QFont()
        font.setPointSizeF(float(font_size))
        self.ci.layout.setSpacing(inner_pad)
        self.ci.layout.setContentsMargins(0, 0, 0, 0)
        if self.data_ndim == 2:
            self.axes = [ItoolPlotItem(self, **self.plot_kw) for _ in range(3)]
            self.addItem(self.axes[0], 1, 0, 1, 1)
            self.addItem(self.axes[1], 0, 0, 1, 1)
            self.addItem(self.axes[2], 1, 1, 1, 1)
            valid_selection = ((1, 0, 0, 1), (1, 1, 0, 0), (0, 0, 1, 1))
        elif self.data_ndim == 3:
            self.axes = [ItoolPlotItem(self, **self.plot_kw) for _ in range(6)]
            self.addItem(self.axes[0], 2, 0, 1, 1)
            self.addItem(self.axes[1], 0, 0, 1, 1)
            self.addItem(self.axes[2], 2, 2, 1, 1)
            self.addItem(self.axes[3], 0, 1, 2, 2)
            self.addItem(self.axes[4], 1, 0, 1, 1)
            self.addItem(self.axes[5], 2, 1, 1, 1)
            valid_selection = (
                (1, 0, 0, 1),
                (1, 1, 0, 0),
                (0, 0, 1, 1),
                (0, 1, 1, 0),
                (1, 0, 0, 0),
                (0, 0, 0, 1),
            )
        elif self.data_ndim == 4:
            self.axes = [ItoolPlotItem(self, **self.plot_kw) for _ in range(7)]
            self.addItem(self.axes[0], 2, 0, 1, 1)
            self.addItem(self.axes[1], 0, 0, 1, 1)
            self.addItem(self.axes[2], 2, 2, 1, 1)
            self.addItem(self.axes[3], 1, 1, 1, 2)
            self.addItem(self.axes[4], 1, 0, 1, 1)
            self.addItem(self.axes[5], 2, 1, 1, 1)
            self.addItem(self.axes[6], 0, 1, 1, 2)
            valid_selection = (
                (1, 0, 0, 1),
                (1, 1, 0, 0),
                (0, 0, 1, 1),
                (0, 0, 1, 0),
                (1, 0, 0, 0),
                (0, 0, 0, 1),
                (0, 1, 1, 0),
            )
        else:
            raise NotImplementedError("Only supports 2D, 3D, and 4D arrays.")
        for i, (p, sel) in enumerate(zip(self.axes, valid_selection, strict=True)):
            p.setDefaultPadding(0)
            for axis in ["left", "bottom", "right", "top"]:
                p.getAxis(axis).setTickFont(font)
                p.getAxis(axis).setStyle(
                    autoExpandTextSpace=True, autoReduceTextSpace=True
                )
            p.showAxes(sel, showValues=sel, size=(horiz_pad, vert_pad))
            # p.showAxes(sel, showValues=(0, 0, 0, 0), size=(0,0))
            if i in [1, 4]:
                p.setXLink(self.axes[0])
            elif i in [2, 5]:
                p.setYLink(self.axes[0])
        self._update_stretch()

    def _lims_to_rect(self, i, j):
        x = self.data_lims[i][0] - self.data_incs[i]
        y = self.data_lims[j][0] - self.data_incs[j]
        w = self.data_lims[i][-1] - x
        h = self.data_lims[j][-1] - y
        x += 0.5 * self.data_incs[i]
        y += 0.5 * self.data_incs[j]
        return QtCore.QRectF(x, y, w, h)

    def _initialize_plots(self):
        if self.data_ndim == 2:
            self.maps = (ItoolImageItem(self, name="Main Image", **self.image_kw),)
            self.hists = (
                self.axes[1].plot(name="X Profile", **self.profile_kw),
                self.axes[2].plot(name="Y Profile", **self.profile_kw),
            )
            self.cursors = (
                (
                    ItoolCursorLine(
                        self, angle=90, movable=True, name="X Cursor", **self.cursor_kw
                    ),
                    ItoolCursorLine(
                        self, angle=90, movable=True, name="X Cursor", **self.cursor_kw
                    ),
                ),
                (
                    ItoolCursorLine(
                        self, angle=0, movable=True, name="Y Cursor", **self.cursor_kw
                    ),
                    ItoolCursorLine(
                        self, angle=0, movable=True, name="Y Cursor", **self.cursor_kw
                    ),
                ),
            )
            self.spans = (
                (
                    pg.LinearRegionItem(orientation="vertical", **self.span_kw),
                    pg.LinearRegionItem(orientation="vertical", **self.span_kw),
                ),
                (
                    pg.LinearRegionItem(orientation="horizontal", **self.span_kw),
                    pg.LinearRegionItem(orientation="horizontal", **self.span_kw),
                ),
            )
            self.axes[0].addItem(self.maps[0])
            self.axes[0].addItem(self.cursors[0][0])
            self.axes[1].addItem(self.cursors[0][1])
            self.axes[0].addItem(self.cursors[1][0])
            self.axes[2].addItem(self.cursors[1][1])
            self.axes[0].addItem(self.spans[0][0])
            self.axes[1].addItem(self.spans[0][1])
            self.axes[0].addItem(self.spans[1][0])
            self.axes[2].addItem(self.spans[1][1])
        elif self.data_ndim >= 3:
            self.maps = (
                ItoolImageItem(self, name="Main Image", **self.image_kw),
                ItoolImageItem(self, name="Horiz Slice", **self.image_kw),
                ItoolImageItem(self, name="Vert Slice", **self.image_kw),
            )
            self.hists = (
                self.axes[1].plot(name="X Profile", **self.profile_kw),
                self.axes[2].plot(name="Y Profile", **self.profile_kw),
                self.axes[3].plot(name="Z Profile", **self.profile_kw),
            )
            self.cursors = (
                (
                    ItoolCursorLine(
                        self, angle=90, movable=True, name="X Cursor", **self.cursor_kw
                    ),
                    ItoolCursorLine(
                        self, angle=90, movable=True, name="X Cursor", **self.cursor_kw
                    ),
                    ItoolCursorLine(
                        self, angle=90, movable=True, name="X Cursor", **self.cursor_kw
                    ),
                ),
                (
                    ItoolCursorLine(
                        self, angle=0, movable=True, name="Y Cursor", **self.cursor_kw
                    ),
                    ItoolCursorLine(
                        self, angle=0, movable=True, name="Y Cursor", **self.cursor_kw
                    ),
                    ItoolCursorLine(
                        self, angle=0, movable=True, name="Y Cursor", **self.cursor_kw
                    ),
                ),
                (
                    ItoolCursorLine(
                        self, angle=90, movable=True, name="Z Cursor", **self.cursor_kw
                    ),
                    ItoolCursorLine(
                        self, angle=90, movable=True, name="Z Cursor", **self.cursor_kw
                    ),
                    ItoolCursorLine(
                        self, angle=0, movable=True, name="Z Cursor", **self.cursor_kw
                    ),
                ),
            )
            self.spans = (
                (
                    pg.LinearRegionItem(orientation="vertical", **self.span_kw),
                    pg.LinearRegionItem(orientation="vertical", **self.span_kw),
                    pg.LinearRegionItem(orientation="vertical", **self.span_kw),
                ),
                (
                    pg.LinearRegionItem(orientation="horizontal", **self.span_kw),
                    pg.LinearRegionItem(orientation="horizontal", **self.span_kw),
                    pg.LinearRegionItem(orientation="horizontal", **self.span_kw),
                ),
                (
                    pg.LinearRegionItem(orientation="vertical", **self.span_kw),
                    pg.LinearRegionItem(orientation="vertical", **self.span_kw),
                    pg.LinearRegionItem(orientation="horizontal", **self.span_kw),
                ),
            )
            self.axes[0].addItem(self.maps[0])
            self.axes[4].addItem(self.maps[1])
            self.axes[5].addItem(self.maps[2])
            self.axes[0].addItem(self.cursors[0][0])
            self.axes[1].addItem(self.cursors[0][1])
            self.axes[4].addItem(self.cursors[0][2])
            self.axes[0].addItem(self.cursors[1][0])
            self.axes[2].addItem(self.cursors[1][1])
            self.axes[5].addItem(self.cursors[1][2])
            self.axes[3].addItem(self.cursors[2][0])
            self.axes[5].addItem(self.cursors[2][1])
            self.axes[4].addItem(self.cursors[2][2])
            self.axes[0].addItem(self.spans[0][0])
            self.axes[1].addItem(self.spans[0][1])
            self.axes[4].addItem(self.spans[0][2])
            self.axes[0].addItem(self.spans[1][0])
            self.axes[2].addItem(self.spans[1][1])
            self.axes[5].addItem(self.spans[1][2])
            self.axes[3].addItem(self.spans[2][0])
            self.axes[5].addItem(self.spans[2][1])
            self.axes[4].addItem(self.spans[2][2])
            # if self.data_lims[-1][-1] * self.data_lims[-1][0] < 0:
            # self.axes[3].axvline(0., label='Fermi Level', **self.fermi_kw)
            # for m in self.maps:
            # m.sigImageChanged.connect(self._refresh_histograms)
            if self.data_ndim == 4:
                self.hists += (self.axes[6].plot(name="T Profile", **self.profile_kw),)
                self.cursors += (
                    (
                        ItoolCursorLine(
                            self,
                            angle=90,
                            movable=True,
                            name="T Cursor",
                            **self.cursor_kw,
                        ),
                    ),
                )
                self.spans += (
                    (pg.LinearRegionItem(orientation="vertical", **self.span_kw),),
                )

                self.axes[6].addItem(self.cursors[3][0])
                self.axes[6].addItem(self.spans[3][0])

        for s in chain.from_iterable(self.spans):
            s.setVisible(False)

    @property
    def all(self):
        return self.maps + self.hists + self.cursors

    def autoRange(self, padding=None):
        for ax in self.axes:
            if not all(ax.autoRangeEnabled()):
                ax.autoRange(padding=padding)

    def toggle_axes(self, axis):
        target = self.axes[axis]
        toggle = False if target in self.ci.items.keys() else True

        if self.data_ndim == 2:
            ref_dims = ((1, 0, 1, 1), (0, 0, 1, 1), (1, 1, 1, 1))
            top_left = (1,)
            bottom_right = (2,)
            top_right = ()
        elif self.data_ndim == 3:
            ref_dims = (
                (2, 0, 1, 1),
                (0, 0, 1, 1),
                (2, 2, 1, 1),
                (0, 1, 2, 2),
                (1, 0, 1, 1),
                (2, 1, 1, 1),
            )
            top_left = (1, 4)
            bottom_right = (5, 2)
            top_right = (3,)
        elif self.data_ndim == 4:
            ref_dims = (
                (2, 0, 1, 1),
                (0, 0, 1, 1),
                (2, 2, 1, 1),
                (0, 1, 1, 2),
                (1, 0, 1, 1),
                (2, 1, 1, 1),
                (1, 1, 1, 2),
            )
            top_left = (1, 4)
            bottom_right = (5, 2)
            top_right = (3, 6)

        if axis in top_left:
            group = top_left
            totalspan = (2, 1)
        elif axis in bottom_right:
            group = bottom_right
            totalspan = (1, 2)
        elif axis in top_right:
            group = top_right
            totalspan = (2, 2)
        elif not toggle:
            self.removeItem(target)
            return
        else:
            self.addItem(target, *ref_dims[axis])
            return

        anchors = tuple(ref_dims[i][:2] for i in group)
        other_index = [
            x for x in group if x != axis and self.axes[x] in self.ci.items.keys()
        ]
        other = [self.axes[i] for i in other_index]
        unique = True if len(other) == 0 else False
        if not toggle:
            self.removeItem(target)
            if not unique:
                for o in other:
                    self.removeItem(o)
                    self.addItem(o, *anchors[0], *totalspan)
        elif unique:
            self.addItem(target, *anchors[0], *totalspan)
        else:
            # for i, o, oi in enumerate(zip(other, other_index)):
            # self.removeItem(o)
            # for o, oi in zip(other, other_index):
            #     self.removeItem(o)
            # self.addItem(self.axes[group[0]], *ref_dims[axis])
            # for o, oi in zip(other, other_index):
            #     self.addItem(o, *ref_dims[oi])
            # for a in group:
            #     ax = self.axes[a]
            #     if ax in self.ci.items.keys():
            #         self.removeItem(ax)
            #     self.addItem(ax, *ref_dims[a])
            self.addItem(self.axes[group[0]], *anchors[0], *ref_dims[axis][2:])
            self.addItem(self.axes[group[1]], *anchors[1], *ref_dims[axis][2:])

    def set_labels(self, labels=None):
        """labels: list or tuple of str."""
        if labels is None:
            labels = self.data_dims
        # 0: default, 1: svg, 2: pixmap
        labelmode = ItoolAxisItem.LabelType.TextLabel
        if labelmode is ItoolAxisItem.LabelType.TextLabel:
            labels_ = labels
        else:
            with rc_context({"text.usetex": True}):
                labels_ = [self.labelify(lab) for lab in labels]

        self.axes[0].setLabels(left=labels_[1], bottom=labels_[0], mode=labelmode)
        self.axes[1].setLabels(top=labels_[0], mode=labelmode)
        self.axes[2].setLabels(right=labels_[1], mode=labelmode)
        if self.data_ndim >= 3:
            self.axes[3].setLabels(top=labels_[2], mode=labelmode)
            self.axes[4].setLabels(left=labels_[2], mode=labelmode)
            self.axes[5].setLabels(bottom=labels_[2], mode=labelmode)
            if self.data_ndim == 4:
                self.axes[6].setLabels(top=labels_[3], mode=labelmode)

    @property
    def data_dims(self):
        return self.data.dims

    @property
    def data_shape(self):
        return self.data.shape

    @property
    def data_ndim(self):
        return self.data.ndim

    @property
    def data_coords(self):
        return tuple(self.data[dim].values for dim in self.data_dims)

    @property
    def data_incs(self):
        return tuple(coord[1] - coord[0] for coord in self.data_coords)

    @property
    def data_lims(self):
        return tuple((coord[0], coord[-1]) for coord in self.data_coords)

    @property
    def data_vals_T(self):
        if self.data_ndim == 2:
            return self.data.values.T
        elif self.data_ndim == 3:
            return self.data.values.transpose(1, 2, 0)
        elif self.data_ndim == 4:
            return self.data.values.transpose(1, 2, 3, 0)

    def _assign_vals_T(self):
        return

    #     if self.data_ndim == 2:
    #         self.data_vals_T = np.ascontiguousarray(self.data.values.T)
    #     elif self.data_ndim == 3:
    #         self.data_vals_T = np.ascontiguousarray(
    #             np.transpose(self.data.values, axes=(1, 2, 0))
    #         )
    #     elif self.data_ndim == 4:
    #         self.data_vals_T = np.ascontiguousarray(
    #             np.transpose(self.data.values, axes=(1, 2, 3, 0))
    #         )
    #     else:
    #         raise NotImplementedError("Wrong data dimensions")

    def set_data(self, data, update_all=False, reset_cursor=True):
        # Data properties
        if self.data is not None:
            ndim_old = self.data_ndim
            self.data = parse_data(data)
            # self.data_ndim = self.data.ndim
            if self.data_ndim != ndim_old:
                update_all = True
        self.data = parse_data(data)
        self._assign_vals_T()
        if update_all:
            self.clear()
            self._initialize_layout()
            self._initialize_plots()
            self.clim_locked = False
            self.clim_list = [()] * self.data_ndim
            self.avg_win = [
                1,
            ] * self.data_ndim
            self.averaged = [
                False,
            ] * self.data_ndim
            self.axis_locked = [
                False,
            ] * self.data_ndim

        # Imagetool properties
        if reset_cursor:
            self.cursor_pos = [
                None,
            ] * self.data_ndim
            self._last_ind = [
                None,
            ] * self.data_ndim
            self.reset_cursor()
        self.set_labels()
        self._apply_change()
        # self.autoRange()
        # for ax in self.axes:
        #     ax.enableAutoRange(enable=True)

        self.sigDataChanged.emit(self)

    def toggle_colorbar(self, val):
        if self.colorbar is None:
            self.colorbar = ItoolColorBar(self, width=20)
            self.addItem(self.colorbar, None, None, self.ci.layout.rowCount(), 1)
        self.colorbar.setVisible(val)

    def reset_cursor(self, update=False):
        """Return the cursor to the center of the image."""
        for axis, coord in enumerate(self.data_coords):
            self.set_index(axis, self._get_middle_index(coord), update=update)

    def _cursor_drag(self, axis, line):
        self.set_value(axis, line.value())

    def connect_signals(self):
        """Connect events."""
        for axis, cursors in enumerate(self.cursors):
            for c in cursors:
                c.sigDragged.connect(lambda v, i=axis: self.set_value(i, v.value()))
        # self.proxy = pg.SignalProxy(
        #     self.scene().sigMouseMoved,
        #     rateLimit=self.screen.refreshRate(),
        #     slot=self.onMouseDrag,
        # )
        self.ci.geometryChanged.connect(
            lambda: self._update_stretch(*self._stretch_factors)
        )
        self.scene().sigMouseClicked.connect(self.onMouseDrag)

    def _get_curr_axes_index(self, pos):
        for i, ax in enumerate(self.axes):
            if ax.vb.sceneBoundingRect().contains(pos):
                return i, self._get_mouse_datapos(ax, pos)
        if self.colorbar is not None:
            if self.colorbar.sceneBoundingRect().contains(pos):
                return -1, self._get_mouse_datapos(self.colorbar, pos)
        return None, None

    def _measure_fps(self):
        now = perf_counter()
        fps = 1.0 / (now - self._fpsLastUpdate)
        self._fpsLastUpdate = now
        w = 0.8
        self._avg_fps = self._avg_fps * (1 - w) + fps * w
        self.axes[1].setTitle(f"{self._avg_fps:0.2f} fps")

    def labelify(self, text):
        """Prettify some frequently used axis labels."""
        labelformats = {
            "kx": "$k_x$",
            "ky": "$k_y$",
            "kz": "$k_z$",
            "alpha": "$\\alpha$",
            "beta": "$\\beta$",
            "theta": "$\\theta$",
            "phi": "$\\phi$",
            "chi": "$\\chi$",
            "eV": "$E$",
        }
        try:
            return labelformats[text]
        except KeyError:
            return text

    def set_cmap(
        self,
        cmap=None,
        gamma=None,
        reverse=None,
        high_contrast=False,
        zero_centered=None,
    ):
        if cmap is not None:
            self.cmap = cmap
        if gamma is not None:
            self.gamma = gamma
        if reverse is not None:
            self.reverse = reverse
        if zero_centered is None:
            zero_centered = self.zero_centered
        else:
            self.zero_centered = zero_centered
        self.norm_cmap = pg_colormap_powernorm(
            self.cmap,
            self.gamma,
            reverse=self.reverse,
            high_contrast=high_contrast,
            zero_centered=zero_centered,
        )
        for im in self.maps:
            im._colorMap = self.norm_cmap
            im.setLookupTable(self.norm_cmap.getStops()[1], update=False)
        self._apply_change(self._only_maps)

    def set_clim_lock(self, lock):
        if self.colorbar is not None:
            self.colorbar.autolevels = ~lock
        if lock:
            self.clim_locked = True
            for i, m in enumerate(self.maps):
                self.clim_list[i] = m.getLevels()
        else:
            self.clim_locked = False

    def set_index(self, axis, index, update=True):
        self._last_ind[axis] = index
        self.cursor_pos[axis] = self.data_coords[axis][index]
        if update is True:
            self._apply_change(self._only_axis[axis])

    def set_value(self, axis, val, update=True):
        self._last_ind[axis] = self.get_index_of_value(axis, val)
        self.cursor_pos[axis] = val
        if update is True:
            self._apply_change(self._only_axis[axis])

    def set_cursor_color(self, c):
        for cursor in self.cursors:
            cursor.setPen(pg.mkPen(c))
        self._apply_change()

    def set_line_color(self, c):
        for line in self.hists:
            line.setPen(pg.mkPen(c))
        self._apply_change()

    def set_navg(self, axis, n, update=True):
        self.avg_win[axis] = n
        if n == 1:
            self.averaged[axis] = False
        else:
            self.averaged[axis] = True
        if update:
            self._refresh_navg(reset=False)

    def _refresh_navg(self, reset=False):
        if reset:
            for axis in range(self.data_ndim):
                self.averaged[axis] = False
                self.avg_win[axis] = 1
        for axis in range(self.data_ndim):
            for s in self.spans[axis]:
                s.setVisible(self.averaged[axis])
        self._apply_change()

    def _get_bin_slice(self, axis):
        if self.averaged[axis]:
            center = self._last_ind[axis]
            window = self.avg_win[axis]
            return slice(center - window // 2, center + (window - 1) // 2 + 1)
        else:
            return slice(self._last_ind[axis], self._last_ind[axis] + 1)

    def _get_binned_data(self, axis):
        axis -= 1
        if not self.averaged[axis + 1]:
            return self.data_vals_T[
                (slice(None),) * (axis % self.data_ndim)
                + (self._get_bin_slice(axis + 1),)
            ].squeeze(axis=axis)
        else:
            return _general_nanmean_func(
                self.data_vals_T[
                    (slice(None),) * (axis % self.data_ndim)
                    + (self._get_bin_slice(axis + 1),)
                ],
                axis=axis,
            )

    def _binned_profile(self, avg_axis):
        if not any(self.averaged):
            return self._block_slicer(avg_axis, [self._last_ind[i] for i in avg_axis])
        slices = tuple(self._get_bin_slice(ax) for ax in avg_axis)
        return self._block_slice_avg(avg_axis, slices)
        # self._slice_block = self._block_slicer(avg_axis, slices)
        # return _general_nanmean_func(self._slice_block, axis=[(ax - 1) for ax in avg_axis])

    def _block_slice_avg(self, axis=None, slices=None):
        axis = [(ax - 1) % self.data_ndim for ax in axis]
        return _general_nanmean_func(
            self.data_vals_T[
                tuple(
                    slices[axis.index(d)] if d in axis else slice(None)
                    for d in range(self.data_ndim)
                )
            ],
            axis=axis,
        )

    def _block_slicer(self, axis=None, slices=None):
        axis = [(ax - 1) % self.data_ndim for ax in axis]
        return self.data_vals_T[
            tuple(
                slices[axis.index(d)] if d in axis else slice(None)
                for d in range(self.data_ndim)
            )
        ]

    def update_spans(self, axis):
        slc = self._get_bin_slice(axis)
        lb = max(0, slc.start)
        ub = min(self.data_shape[axis] - 1, slc.stop - 1)
        region = self.data_coords[axis][[lb, ub]]
        for span in self.spans[axis]:
            span.setRegion(region)

    def get_index_of_value(self, axis, val):
        ind = min(
            round((val - self.data_lims[axis][0]) / self.data_incs[axis]),
            self.data_shape[axis] - 1,
        )
        if ind < 0:
            return 0
        return ind

    def set_axis_lock(self, axis, lock):
        self.axis_locked[axis] = lock

    def transpose_axes(self, axis1, axis2):
        dims_new = list(self.data_dims)
        dims_new[axis1], dims_new[axis2] = self.data_dims[axis2], self.data_dims[axis1]
        data_new = self.data.transpose(*dims_new)
        self._last_ind[axis1], self._last_ind[axis2] = (
            self._last_ind[axis2],
            self._last_ind[axis1],
        )
        self.cursor_pos[axis1], self.cursor_pos[axis2] = (
            self.cursor_pos[axis2],
            self.cursor_pos[axis1],
        )
        self.clim_list[axis1], self.clim_list[axis2] = (
            self.clim_list[axis2],
            self.clim_list[axis1],
        )
        self.avg_win[axis1], self.avg_win[axis2] = (
            self.avg_win[axis2],
            self.avg_win[axis1],
        )
        self.averaged[axis1], self.averaged[axis2] = (
            self.averaged[axis2],
            self.averaged[axis1],
        )
        self.axis_locked[axis1], self.axis_locked[axis2] = (
            self.axis_locked[axis2],
            self.axis_locked[axis1],
        )
        self.set_data(data_new, update_all=False, reset_cursor=False)

    def get_key_modifiers(self):
        Qmods = self.qapp.queryKeyboardModifiers()
        mods = []
        if (Qmods & QtCore.Qt.ShiftModifier) == QtCore.Qt.ShiftModifier:
            mods.append("shift")
        if (Qmods & QtCore.Qt.ControlModifier) == QtCore.Qt.ControlModifier:
            mods.append("control")
        if (Qmods & QtCore.Qt.AltModifier) == QtCore.Qt.AltModifier:
            mods.append("alt")
        return mods

    def _get_mouse_datapos(self, plot, pos):
        """Return mouse position in data coords."""
        mouse_point = plot.vb.mapSceneToView(pos)
        return mouse_point.x(), mouse_point.y()

    def onMouseDrag(self, evt):
        try:
            axis_ind, datapos = self._get_curr_axes_index(evt.scenePos())
        except AttributeError:
            try:
                axis_ind, datapos = self._get_curr_axes_index(evt.scenePosition())
            except AttributeError:
                axis_ind, datapos = self._get_curr_axes_index(evt[0])

        if hasattr(evt, "_buttonDownScenePos"):
            axis_start, _ = self._get_curr_axes_index(evt.buttonDownScenePos())
            if axis_ind != axis_start:
                return
        elif self.qapp.queryKeyboardModifiers() != QtCore.Qt.ControlModifier:
            evt.ignore()
            return

        if axis_ind is None:
            return

        V = [None] * self.data_ndim
        if axis_ind == 0:
            V[0:2] = datapos
        elif axis_ind == 1:
            V[0] = datapos[0]
        elif axis_ind == 2:
            V[1] = datapos[1]
        elif axis_ind == 3:
            V[2] = datapos[0]
        elif axis_ind == 4:
            V[0], V[2] = datapos
        elif axis_ind == 5:
            V[2], V[1] = datapos
        elif axis_ind == 6:
            V[3] = datapos[0]
        elif axis_ind == -1:
            self.colorbar.isoline.setPos(datapos[1])
            return
        D = [v is not None for v in V]
        if not any(D):
            return
        for i in range(self.data_ndim):
            if D[i]:
                ind = self.get_index_of_value(i, V[i])
                if (self.snap and (ind == self._last_ind[i])) or self.axis_locked[i]:
                    D[i] = False
                else:
                    self._last_ind[i] = ind
            if not D[i]:
                V[i] = self.cursor_pos[i]
        if not any(D):
            return
        if not self.snap:
            self.cursor_pos = V
        self._apply_change(D)

    def _apply_change(self, cond=None):
        if cond is None:
            update = (True,) * len(self.all)
        elif isinstance(cond, str):
            if cond == self._only_maps:
                if self.data_ndim == 2:
                    update = (True,) + (False,) * 4
                elif self.data_ndim == 3:
                    update = (True,) * 3 + (False,) * 6
                elif self.data_ndim == 4:
                    update = (True,) * 3 + (False,) * 8
            elif cond in self._only_axis:
                update = [False] * self.data_ndim
                update[self._only_axis.index(cond)] = True
            else:
                raise ValueError
        else:
            update = cond
        if len(update) == self.data_ndim:
            if self.data_ndim == 2:
                update = (False, update[1], update[0], update[0], update[1])
            elif self.data_ndim == 3:
                update = (
                    update[2],
                    update[1],
                    update[0],
                    update[1] or update[2],
                    update[0] or update[2],
                    update[0] or update[1],
                    update[0],
                    update[1],
                    update[2],
                )
            elif self.data_ndim == 4:
                update = (
                    update[2] or update[3],
                    update[1] or update[3],
                    update[0] or update[3],
                    update[1] or update[2] or update[3],
                    update[0] or update[2] or update[3],
                    update[0] or update[1] or update[3],
                    update[0] or update[1] or update[2],
                    update[0],
                    update[1],
                    update[2],
                    update[3],
                )
        elif len(update) != len(self.all):
            raise ValueError
        for i in get_true_indices(update):
            self._refresh_data(i)
        if self.bench:
            self._measure_fps()

    @suppressnanwarning
    def _refresh_data(self, i):
        if self.snap:
            self.cursor_pos = [
                self.data_coords[i][self._last_ind[i]] for i in range(self.data_ndim)
            ]
        if self.data_ndim == 2:
            self._refresh_data_2d(i)
        elif self.data_ndim == 3:
            self._refresh_data_3d(i)
        elif self.data_ndim == 4:
            self._refresh_data_4d(i)
        self.sigIndexChanged.emit(self._last_ind, self.cursor_pos)

    # def get_levels(self, i):
    #     if self.clim_locked:
    #         if self.zero_centered:

    #         else:
    #             return self.clim_list[i]
    #     else:
    #         self.all[i].getLevels()
    # def _refresh_histograms(self):
    #     self.hists[0].setData(self.data_coords[0], self.maps[0].image[self._last_ind[1], :])
    #     self.hists[1].setData(self.maps[0].image[:, self._last_ind[0]], self.data_coords[1])
    #     if self.data_ndim == 3:
    #         self.hists[2].setData(self.data_coords[2], self.maps[1].image[:, self._last_ind[0]])

    def _refresh_data_2d(self, i):
        if i == 0:
            if self.clim_locked:
                self.all[i].setImage(
                    self.data_vals_T,
                    levels=self.clim_list[0],
                    rect=self._lims_to_rect(0, 1),
                )
            else:
                self.all[i].setImage(self.data_vals_T, rect=self._lims_to_rect(0, 1))
                if self.zero_centered:
                    lim = np.amax(np.abs(np.asarray(self.all[i].getLevels())))
                    self.all[i].setLevels([-lim, lim])
        elif i == 1:
            self.all[i].setData(
                self.data_coords[0],
                self._binned_profile([1]),
            )
        elif i == 2:
            self.all[i].setData(
                self._binned_profile([0]),
                self.data_coords[1],
            )
        elif i in [3, 4]:
            for cursor in self.all[i]:
                cursor.maxRange = self.data_lims[i - 3]
                cursor.setPos(self.cursor_pos[i - 3])
                if self.averaged[i - 3]:
                    self.update_spans(i - 3)

    def _refresh_data_3d(self, i):
        if i == 0:
            if self.clim_locked:
                self.all[i].setImage(
                    self._get_binned_data(2),
                    levels=self.clim_list[i],
                    rect=self._lims_to_rect(0, 1),
                )
            else:
                self.all[i].setImage(
                    self._get_binned_data(2),
                    rect=self._lims_to_rect(0, 1),
                )
                if self.zero_centered:
                    lim = np.amax(np.abs(np.asarray(self.all[i].getLevels())))
                    self.all[i].setLevels([-lim, lim])
        elif i == 1:
            if self.clim_locked:
                self.all[i].setImage(
                    self._get_binned_data(1),
                    levels=self.clim_list[i],
                    rect=self._lims_to_rect(0, 2),
                )
            else:
                self.all[i].setImage(
                    # self.data_vals_T[self._last_ind[1], :, :],
                    self._get_binned_data(1),
                    rect=self._lims_to_rect(0, 2),
                )
                if self.zero_centered:
                    lim = np.amax(np.abs(np.asarray(self.all[i].getLevels())))
                    self.all[i].setLevels([-lim, lim])
        elif i == 2:
            if self.clim_locked:
                self.all[i].setImage(
                    self._get_binned_data(0),
                    levels=self.clim_list[i],
                    rect=self._lims_to_rect(2, 1),
                )
            else:
                self.all[i].setImage(
                    self._get_binned_data(0),
                    rect=self._lims_to_rect(2, 1),
                )
                if self.zero_centered:
                    lim = np.amax(np.abs(np.asarray(self.all[i].getLevels())))
                    self.all[i].setLevels([-lim, lim])
        elif i == 3:
            self.hists[0].setData(
                self.data_coords[0],
                self._binned_profile((1, 2)),
            )
        elif i == 4:
            self.hists[1].setData(
                self._binned_profile((0, 2)),
                self.data_coords[1],
            )
        elif i == 5:
            self.hists[2].setData(
                self.data_coords[2],
                self._binned_profile((0, 1)),
            )
        elif i in [6, 7, 8]:
            for cursor in self.all[i]:
                cursor.maxRange = self.data_lims[i - 6]
                cursor.setPos(self.cursor_pos[i - 6])
                if self.averaged[i - 6]:
                    self.update_spans(i - 6)

    def _refresh_data_4d(self, i):
        if i == 0:
            if self.clim_locked:
                self.all[i].setImage(
                    self._binned_profile((2, 3)),
                    levels=self.clim_list[i],
                    rect=self._lims_to_rect(0, 1),
                )
            else:
                self.all[i].setImage(
                    self._binned_profile((2, 3)),
                    rect=self._lims_to_rect(0, 1),
                )
                if self.zero_centered:
                    lim = np.amax(np.abs(np.asarray(self.all[i].getLevels())))
                    self.all[i].setLevels([-lim, lim])
        elif i == 1:
            if self.clim_locked:
                self.all[i].setImage(
                    self._binned_profile((1, 3)),
                    levels=self.clim_list[i],
                    rect=self._lims_to_rect(0, 2),
                )
            else:
                self.all[i].setImage(
                    self._binned_profile((1, 3)),
                    rect=self._lims_to_rect(0, 2),
                )
                if self.zero_centered:
                    lim = np.amax(np.abs(np.asarray(self.all[i].getLevels())))
                    self.all[i].setLevels([-lim, lim])
        elif i == 2:
            if self.clim_locked:
                self.all[i].setImage(
                    self._binned_profile((0, 3)),
                    levels=self.clim_list[i],
                    rect=self._lims_to_rect(2, 1),
                )
            else:
                self.all[i].setImage(
                    self._binned_profile((0, 3)),
                    rect=self._lims_to_rect(2, 1),
                )
                if self.zero_centered:
                    lim = np.amax(np.abs(np.asarray(self.all[i].getLevels())))
                    self.all[i].setLevels([-lim, lim])
        elif i == 3:
            self.hists[0].setData(
                self.data_coords[0],
                self._binned_profile((1, 2, 3)),
            )
        elif i == 4:
            self.hists[1].setData(
                self._binned_profile((0, 2, 3)),
                self.data_coords[1],
            )
        elif i == 5:
            self.hists[2].setData(
                self.data_coords[2],
                self._binned_profile((0, 1, 3)),
            )
        elif i == 6:
            self.hists[3].setData(
                self.data_coords[3],
                self._binned_profile((0, 1, 2)),
            )
        elif i in [7, 8, 9, 10]:
            for cursor in self.all[i]:
                cursor.maxRange = self.data_lims[i - 7]
                cursor.setPos(self.cursor_pos[i - 7])
                if self.averaged[i - 7]:
                    self.update_spans(i - 7)

    def changeEvent(self, evt):
        if evt.type() == QtCore.QEvent.PaletteChange:
            self.update()
        super().changeEvent(evt)

    def _drawpath(self):
        # ld = LineDrawer(self.canvas, self.axes[0])
        # points = ld.draw_line()
        # print(points)
        pass

    def _onselectpath(self, verts):
        print(verts)


class ImageToolColors(QtWidgets.QDialog):
    def __init__(self, parent):
        self.parent = parent
        super().__init__(self.parent)
        self.setWindowTitle("Colors")
        raise NotImplementedError

    #     self.cursor_default = color_to_QColor(self.parent.itool.cursor_kw["color"])
    #     self.line_default = color_to_QColor(self.parent.itool.profile_kw["color"])
    #     self.cursor_current = color_to_QColor(self.parent.itool.cursors[0].get_color())
    #     self.line_current = color_to_QColor(self.parent.itool.hists[0].get_color())

    #     if (self.cursor_default.getRgbF() == self.cursor_current.getRgbF()) & (
    #         self.line_default.getRgbF() == self.line_current.getRgbF()
    #     ):
    #         buttons = QtWidgets.QDialogButtonBox(
    #             QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
    #         )
    #     else:
    #         buttons = QtWidgets.QDialogButtonBox(
    #             QtWidgets.QDialogButtonBox.RestoreDefaults
    #             | QtWidgets.QDialogButtonBox.Ok
    #             | QtWidgets.QDialogButtonBox.Cancel
    #         )
    #         buttons.button(QtWidgets.QDialogButtonBox.RestoreDefaults).clicked.connect(
    #             self.reset_colors
    #         )
    #     buttons.rejected.connect(self.reject)
    #     buttons.accepted.connect(self.accept)

    #     cursorlabel = QtWidgets.QLabel("Cursors:")
    #     linelabel = QtWidgets.QLabel("Lines:")
    #     self.cursorpicker = ColorButton(color=self.cursor_current)
    #     self.cursorpicker.colorChanged.connect(self.parent.itool.set_cursor_color)
    #     self.linepicker = ColorButton(color=self.line_current)
    #     self.linepicker.colorChanged.connect(self.parent.itool.set_line_color)

    #     layout = QtWidgets.QGridLayout()
    #     layout.addWidget(cursorlabel, 0, 0)
    #     layout.addWidget(self.cursorpicker, 0, 1)
    #     layout.addWidget(linelabel, 1, 0)
    #     layout.addWidget(self.linepicker, 1, 1)
    #     layout.addWidget(buttons)
    #     self.setLayout(layout)

    # def reject(self):
    #     self.cursorpicker.setColor(self.cursor_current)
    #     self.linepicker.setColor(self.line_current)
    #     super().reject()

    # def reset_colors(self):
    #     self.cursorpicker.setColor(self.cursor_default)
    #     self.linepicker.setColor(self.line_default)


@numba.njit(nogil=True, fastmath=True)
def fast_isocurve_extend(data):
    d2 = np.empty((data.shape[0] + 2, data.shape[1] + 2), dtype=data.dtype)
    d2[1:-1, 1:-1] = data
    d2[0, 1:-1] = data[0]
    d2[-1, 1:-1] = data[-1]
    d2[1:-1, 0] = data[:, 0]
    d2[1:-1, -1] = data[:, -1]
    d2[0, 0] = d2[0, 1]
    d2[0, -1] = d2[1, -1]
    d2[-1, 0] = d2[-1, 1]
    d2[-1, -1] = d2[-1, -2]
    return d2


@numba.njit(nogil=True, fastmath=True)
def fast_isocurve_lines(data, level, index, extendToEdge=False):
    sideTable = (
        [np.int64(x) for x in range(0)],
        [0, 1],
        [1, 2],
        [0, 2],
        [0, 3],
        [1, 3],
        [0, 1, 2, 3],
        [2, 3],
        [2, 3],
        [0, 1, 2, 3],
        [1, 3],
        [0, 3],
        [0, 2],
        [1, 2],
        [0, 1],
        [np.int64(x) for x in range(0)],
    )
    edgeKey = [[(0, 1), (0, 0)], [(0, 0), (1, 0)], [(1, 0), (1, 1)], [(1, 1), (0, 1)]]
    lines = []
    for i in range(index.shape[0]):
        for j in range(index.shape[1]):
            sides = sideTable[index[i, j]]
            for k in range(0, len(sides), 2):
                edges = sides[k : k + 2]
                pts = []
                for m in range(2):
                    p1, p2 = edgeKey[edges[m]][0], edgeKey[edges[m]][1]
                    v1, v2 = data[i + p1[0], j + p1[1]], data[i + p2[0], j + p2[1]]
                    f = (level - v1) / (v2 - v1)
                    fi = 1.0 - f
                    p = (
                        p1[0] * fi + p2[0] * f + i + 0.5,
                        p1[1] * fi + p2[1] * f + j + 0.5,
                    )
                    if extendToEdge:
                        p = (
                            min(data.shape[0] - 2, max(0, p[0] - 1)),
                            min(data.shape[1] - 2, max(0, p[1] - 1)),
                        )
                    pts.append(p)
                lines.append(pts)
    return lines


@numba.njit(nogil=True, fastmath=True)
def fast_isocurve_lines_connected(data, level, index, extendToEdge=False):
    sideTable = (
        [np.int64(x) for x in range(0)],
        [0, 1],
        [1, 2],
        [0, 2],
        [0, 3],
        [1, 3],
        [0, 1, 2, 3],
        [2, 3],
        [2, 3],
        [0, 1, 2, 3],
        [1, 3],
        [0, 3],
        [0, 2],
        [1, 2],
        [0, 1],
        [np.int64(x) for x in range(0)],
    )
    edgeKey = [[(0, 1), (0, 0)], [(0, 0), (1, 0)], [(1, 0), (1, 1)], [(1, 1), (0, 1)]]
    lines = []
    for i in range(index.shape[0]):
        for j in range(index.shape[1]):
            sides = sideTable[index[i, j]]
            for k in range(0, len(sides), 2):
                edges = sides[k : k + 2]
                pts = []
                for m in range(2):
                    p1, p2 = edgeKey[edges[m]][0], edgeKey[edges[m]][1]
                    v1, v2 = data[i + p1[0], j + p1[1]], data[i + p2[0], j + p2[1]]
                    f = (level - v1) / (v2 - v1)
                    fi = 1.0 - f
                    p = (
                        p1[0] * fi + p2[0] * f + i + 0.5,
                        p1[1] * fi + p2[1] * f + j + 0.5,
                    )
                    if extendToEdge:
                        p = (
                            min(data.shape[0] - 2, max(0, p[0] - 1)),
                            min(data.shape[1] - 2, max(0, p[1] - 1)),
                        )
                    gridKey = (
                        i + (1 if edges[m] == 2 else 0),
                        j + (1 if edges[m] == 3 else 0),
                        edges[m] % 2,
                    )
                    pts.append((p, gridKey))
                lines.append(pts)
    return lines


def fast_isocurve(data, level, connected=False, extendToEdge=False, path=False):
    """
    Generate isocurve from 2D data using marching squares algorithm.

    ============== =========================================================
    **Arguments:**
    data           2D numpy array of scalar values
    level          The level at which to generate an isosurface
    connected      If False, return a single long list of point pairs
                   If True, return multiple long lists of connected point
                   locations. (This is slower but better for drawing
                   continuous lines)
    extendToEdge   If True, extend the curves to reach the exact edges of
                   the data.
    path           if True, return a QPainterPath rather than a list of
                   vertex coordinates.
    ============== =========================================================
    """
    if path is True:
        connected = True
    np.nan_to_num(data, copy=False)
    if extendToEdge:
        data = fast_isocurve_extend(data)

    # mark everything below the isosurface level
    mask = data < level
    index = np.zeros([x - 1 for x in mask.shape], dtype=np.int64)
    fields = np.empty((2, 2), dtype=object)
    slices = [slice(0, -1), slice(1, None)]
    for i in range(2):
        for j in range(2):
            fields[i, j] = mask[slices[i], slices[j]]
            vertIndex = i + 2 * j
            index += fields[i, j] * 2**vertIndex

    # make four sub-fields and compute indexes for grid cells
    if connected:
        lines = fast_isocurve_lines_connected(data, level, index, extendToEdge)
        points = {}
        for a, b in lines:
            if a[1] not in points:
                points[a[1]] = [[a, b]]
            else:
                points[a[1]].append([a, b])
            if b[1] not in points:
                points[b[1]] = [[b, a]]
            else:
                points[b[1]].append([b, a])
        lines = fast_isocurve_chain(points)
    else:
        lines = fast_isocurve_lines(data, level, index, extendToEdge)

    if not path:
        return lines  # a list of pairs of points

    path = QtGui.QPainterPath()
    for line in lines:
        path.moveTo(*line[0])
        for p in line[1:]:
            path.lineTo(*p)

    return path


def fast_isocurve_chain(points):
    for k in list(points.keys()):
        try:
            chains = points[k]
        except KeyError:
            continue
        for ch in chains:
            x = None
            while True:
                if x == ch[-1][1]:
                    break
                x = ch[-1][1]
                if x == k:
                    break
                y = ch[-2][1]
                connects = points[x]
                for conn in connects[:]:
                    if conn[1][1] != y:
                        ch.extend(conn[1:])
                del points[x]
            if ch[0][1] == ch[-1][1]:
                chains.pop()
                break
    lines_linked = [np.float64(x) for x in range(0)]
    for ch in points.values():
        if len(ch) == 2:
            ch = ch[1][1:][::-1] + ch[0]  # join together ends of chain
        else:
            ch = ch[0]
        lines_linked.append([p[0] for p in ch])
    return lines_linked


class betterIsocurve(pg.IsocurveItem):
    def __init__(
        self,
        data=None,
        level=0,
        pen="cyan",
        axisOrder=None,
        connected=False,
        extendToEdge=False,
    ):
        super().__init__(data, level, pen, axisOrder)
        self.connected = connected
        self.extendToEdge = extendToEdge

    def generatePath(self):
        if self.data is None:
            self.path = None
            return

        if self.axisOrder == "row-major":
            data = self.data.T
        else:
            data = self.data

        lines = fast_isocurve(data, self.level, self.connected, self.extendToEdge)
        # lines = pg.functions.isocurve(data, self.level, connected=True, extendToEdge=True)
        self.path = QtGui.QPainterPath()
        for line in lines:
            self.path.moveTo(*line[0])
            for p in line[1:]:
                self.path.lineTo(*p)

    def setData(self, data, level=None):
        if self.parentItem() is not None:
            self.axisOrder = self.parentItem().axisOrder
        super().setData(data, level)


class ItoolColorBar(ItoolPlotItem):
    def __init__(
        self,
        itool,
        width=25,
        horiz_pad=45,
        vert_pad=30,
        inner_pad=5,
        font_size=10,
        curve_kw=None,
        line_kw=None,
        *args,
        **kwargs,
    ):
        if line_kw is None:
            line_kw = {"pen": "cyan"}
        if curve_kw is None:
            curve_kw = {}
        super().__init__(itool, *args, **kwargs)
        self.setDefaultPadding(0)
        self.cbar = ItoolImageItem(self.itool, axisOrder="row-major")
        self.npts = 4096
        self.autolevels = True

        self.cbar.setImage(np.linspace(0, 1, self.npts).reshape((-1, 1)))
        self.addItem(self.cbar)

        self.isocurve = betterIsocurve(**curve_kw)
        self.isocurve.setZValue(5)

        self.isoline = ItoolCursorLine(self.itool, angle=0, movable=True, **line_kw)
        self.addItem(self.isoline)
        self.isoline.setZValue(1000)
        self.isoline.sigPositionChanged.connect(self.update_level)

        self.setImageItem(self.itool.maps[0])
        self.setMouseEnabled(x=False, y=True)
        self.setMenuEnabled(True)

        font = QtGui.QFont()
        font.setPointSizeF(float(font_size))
        for axis in ["left", "bottom", "right", "top"]:
            self.getAxis(axis).setTickFont(font)
        self.layout.setColumnFixedWidth(1, width)
        self.layout.setSpacing(inner_pad)
        self.layout.setContentsMargins(0.0, 0.0, 0.0, 0.0)
        self.showAxes(
            (True, True, True, True),
            showValues=(False, False, True, False),
            size=(horiz_pad, 0.0),
        )
        # self.getAxis('right').setStyle(
        # showValues=True, tickTextWidth=horiz_pad,
        # autoExpandTextSpace=False, autoReduceTextSpace=False)

        self.getAxis("top").setHeight(vert_pad)
        self.getAxis("bottom").setHeight(vert_pad)

        # self.getAxis('left').setWidth(inner_pad)

    def setImageItem(self, img):
        self.imageItem = weakref.ref(img)
        self.isocurve.setParentItem(img)
        img.sigImageChanged.connect(self.image_changed)
        self.image_changed()

    def image_changed(self):
        self.cmap_changed()
        levels = self.imageItem().getLevels()
        if self.autolevels:
            self.cbar.setRect(0.0, levels[0], 1.0, levels[1] - levels[0])
        else:
            mn, mx = self.imageItem().quickMinMax(targetSize=2**16)
            self.cbar.setRect(0.0, mn, 1.0, mx - mn)
            self.cbar.setLevels(levels / (mx - mn) - mn)
        self.isoline.setBounds(levels)
        self.update_isodata()

    def cmap_changed(self):
        self.cmap = self.imageItem()._colorMap
        self.lut = self.imageItem().lut
        # self.lut = self.cmap.getStops()[1]
        if not self.npts == self.lut.shape[0]:
            self.npts = self.lut.shape[0]
            self.cbar.setImage(self.cmap.pos.reshape((-1, 1)))
        self.cbar._colorMap = self.cmap
        self.cbar.setLookupTable(self.lut)
        # self.cbar.setColorMap(self.cmap)
        # pg.ImageItem

    def update_isodata(self):
        self.isocurve.setData(self.imageItem().image)

    def update_level(self, line):
        self.isocurve.setLevel(line.value())

    def setVisible(self, visible, *args, **kwargs):
        super().setVisible(visible, *args, **kwargs)
        self.isocurve.setVisible(visible, *args, **kwargs)
        # self.showAxes((False, False, True, False),
        #               showValues=(False, False, True, False),
        #               size=(45, 30))


class itoolJoystick(pg.JoystickButton):
    sigJoystickHeld = QtCore.Signal(object, object)
    sigJoystickReset = QtCore.Signal(object)

    def __init__(self, parent=None):
        QtWidgets.QPushButton.__init__(self, parent)
        self.radius = 100
        self.marker_r = 3
        self.setCheckable(True)
        self.state = None
        self.setState(0, 0)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(lambda: self.sigJoystickHeld.emit(self, self.state))
        # self.setFixedWidth(50)
        # self.setFixedHeight(50)

    def mousePressEvent(self, ev):
        super().mousePressEvent(ev)
        self.timer.start(1000 / 30)

    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        self.timer.stop()

    def mouseDoubleClickEvent(self, ev):
        self.sigJoystickReset.emit(self)
        ev.accept()

    def setState(self, *xy):
        xy = list(xy)
        d = np.sqrt(xy[0] ** 2 + xy[1] ** 2)  # length
        nxy = [0, 0]
        for i in [0, 1]:
            if xy[i] == 0:
                nxy[i] = 0
            else:
                nxy[i] = xy[i] / d
        if d > self.radius:
            d = self.radius
        d = (d / self.radius) ** 2
        xy = [nxy[0] * d, nxy[1] * d]
        w2 = self.width() / 2
        h2 = self.height() / 2
        self.spotPos = QtCore.QPoint(int(w2 * (1 + xy[0])), int(h2 * (1 - xy[1])))
        self.update()
        if self.state == xy:
            return
        self.state = xy
        self.sigStateChanged.emit(self, self.state)


class itoolCursorControls(QtWidgets.QWidget):
    def __init__(self, itool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.itool = itool
        self.ndim = self.itool.data_ndim
        self.layout = FlowLayout(self)

        self.initialize_widgets()
        self.update_content()
        self.itool.sigIndexChanged.connect(self.update_spin)
        self.itool.sigDataChanged.connect(self.update_content)

    def initialize_widgets(self):
        self._cursor_group = BorderlessGroupBox(self, objectName="CursorGroup")
        self._transpose_group = BorderlessGroupBox(self, objectName="TransposeGroup")
        cursor_layout = InnerQHBoxLayout(self._cursor_group)
        transpose_layout = InnerQHBoxLayout(self._transpose_group)

        self._spingroups = tuple(
            BorderlessGroupBox(self, objectName=f"SpinGroup_{i}")
            for i in range(self.ndim)
        )
        self._spingrouplayouts = tuple(InnerQHBoxLayout(sg) for sg in self._spingroups)
        self._spinlabels = tuple(
            QtWidgets.QPushButton(self._spingroups[i], checkable=True)
            for i in range(self.ndim)
        )
        self._spin = tuple(
            QtWidgets.QSpinBox(
                self._spingroups[i],
                singleStep=1,
                wrapping=False,
                minimumWidth=60,
                keyboardTracking=False,
            )
            for i in range(self.ndim)
        )
        self._dblspin = tuple(
            QtWidgets.QDoubleSpinBox(
                self._spingroups[i],
                decimals=3,
                wrapping=False,
                minimumWidth=70,
                # correctionMode=QtWidgets.QAbstractSpinBox.CorrectToNearestValue,
                keyboardTracking=False,
            )
            for i in range(self.ndim)
        )
        self._transpose_button = tuple(
            IconButton(self, on=f"transpose_{i}") for i in range(self.ndim)
        )
        # if self.ndim == 2:
        #     self._hide_button = (QtWidgets.QPushButton(self),
        #                          QtWidgets.QPushButton(self))
        #     self._hide_button[0].toggled.connect(
        #         lambda val, i=1: self.toggle_axes(val, i))
        #     self._hide_button[1].toggled.connect(
        #         lambda val, i=2: self.toggle_axes(val, i))
        # elif self.ndim == 3:

        self._snap_button = IconButton(
            self, on="snap", off="snap_off", toolTip="Snap cursor to data"
        )
        self._snap_button.toggled.connect(self._assign_snap)

        self._joystick = itoolJoystick(self)
        self._joystick.sigJoystickHeld.connect(self._joystick_held)
        self._joystick.sigJoystickReset.connect(self._joystick_reset)

        # col_default = self.itool._stretch_factors[1]
        # self._hslider = QtWidgets.QSlider(
        #     QtCore.Qt.Horizontal,
        #     self,
        #     minimum=0,
        #     maximum=col_default[2] + col_default[1],
        #     value=col_default[2],
        # )
        # self._hslider.valueChanged.connect(lambda v: self._assign_stretch(col=v))
        # self._hslider.mouseDoubleClickEvent = lambda _: self._hslider.setValue(300000)
        # # self._hslider.setFixedHeight(30)
        # self._hslider.setFixedWidth(100)

        # row_default = self.itool._stretch_factors[0]
        # self._vslider = QtWidgets.QSlider(
        #     QtCore.Qt.Horizontal,
        #     self,
        #     minimum=0,
        #     maximum=row_default[2] + row_default[1],
        #     value=row_default[2],
        # )
        # self._vslider.valueChanged.connect(lambda v: self._assign_stretch(row=v))
        # self._vslider.mouseDoubleClickEvent = lambda _: self._vslider.setValue(300000)
        # # self._vslider.setFixedHeight(30)
        # self._vslider.setFixedWidth(100)

        # cursor_layout.addWidget(self._hslider)
        # cursor_layout.addWidget(self._vslider)

        for i in range(self.ndim):
            self._spinlabels[i].toggled.connect(
                lambda v, axis=i: self.itool.set_axis_lock(axis, v)
            )
            self._spin[i].valueChanged.connect(
                lambda v, axis=i: self._index_changed(axis, v)
            )
            self._dblspin[i].valueChanged.connect(
                lambda v, axis=i: self._value_changed(axis, v)
            )
            self._transpose_button[i].clicked.connect(
                lambda axis1=i, axis2=i - 1: self.itool.transpose_axes(axis1, axis2)
            )
            self._spingrouplayouts[i].addWidget(self._spinlabels[i])
            self._spingrouplayouts[i].addWidget(self._spin[i])
            self._spingrouplayouts[i].addWidget(self._dblspin[i])
            self.layout.addWidget(self._spingroups[i])
            # cursor_layout.addWidget(self._spinlabels[i], 0, 3 * i)
            # cursor_layout.addWidget(self._spin[i], 0, 3 * i + 1)
            # cursor_layout.addWidget(self._dblspin[i], 0, 3 * i + 2)
            # cursor_layout.addSpacing(5)
        cursor_layout.addWidget(self._snap_button)  # , 0, 3 * self.ndim)
        cursor_layout.addWidget(self._joystick)  # , 0, 3 * self.ndim + 1)

        self.layout.addWidget(self._cursor_group)
        self.layout.addWidget(self._transpose_group)

        for tb in self._transpose_button:
            transpose_layout.addWidget(tb)

        # self.layout.addStretch()

    def _joystick_reset(self, _):
        if self.itool.qapp.queryKeyboardModifiers() == QtCore.Qt.ControlModifier:
            self.itool._update_stretch()
        else:
            self.itool.reset_cursor(update=True)

    def _joystick_held(self, _, state):
        if self.itool.qapp.queryKeyboardModifiers() == QtCore.Qt.ControlModifier:
            self._assign_stretch(row=20000 * state[1], col=20000 * state[0])
        else:
            linearity = 1
            factor = 0.05
            for i in range(2):
                if not self._spinlabels[i].isChecked():
                    if self.itool.snap:
                        self._spin[i].setValue(
                            self._spin[i].value()
                            + np.sign(state[i])
                            * (self.itool.data_shape[i] - 1)
                            * factor
                            * np.float_power(np.abs(state[i]), linearity)
                        )
                    else:
                        lims = self.itool.data_lims[i]
                        self._dblspin[i].setValue(
                            self._dblspin[i].value()
                            + np.sign(state[i])
                            * np.abs(lims[1] - lims[0])
                            * factor
                            * np.float_power(np.abs(state[i]), linearity)
                        )

    def update_content(self):
        ndim = self.itool.data_ndim
        if ndim != self.ndim:
            self.layout.clear()
            self.ndim = ndim
            self.initialize_widgets()
        # self._snap_button.blockSignals(True)
        self._snap_button.setChecked(self.itool.snap)
        # self._snap_button.blockSignals(False)

        width_spinlabel = []
        width_dblspin = []
        width_spin = []
        for i in range(self.ndim):
            self._spingroups[i].blockSignals(True)
            self._spin[i].blockSignals(True)
            self._dblspin[i].blockSignals(True)

            self._spinlabels[i].setText(self.itool.data_dims[i])
            self._spinlabels[i].setChecked(self.itool.axis_locked[i])

            width_spinlabel.append(
                self._spinlabels[i]
                .fontMetrics()
                .boundingRect(self._spinlabels[i].text())
                .width()
                + 15
            )

            self._spin[i].setRange(0, self.itool.data_shape[i] - 1)
            self._spin[i].setValue(self.itool._last_ind[i])

            self._dblspin[i].setRange(*self.itool.data_lims[i])
            self._dblspin[i].setSingleStep(self.itool.data_incs[i])
            self._dblspin[i].setValue(
                self.itool.data_coords[i][self.itool._last_ind[i]]
            )

            width_dblspin.append(self._dblspin[i].width())
            width_spin.append(self._spin[i].width())

            self._spinlabels[i].blockSignals(False)
            self._spin[i].blockSignals(False)
            self._dblspin[i].blockSignals(False)

        for i in range(self.ndim):
            self._spinlabels[i].setMaximumWidth(max(width_spinlabel))
            # self._spingrouplayouts[i].setColumnMinimumWidth(1, max(width_spin))
            # self._spingrouplayouts[i].setColumnMinimumWidth(2, max(width_dblspin))
            # self._spin[i].setMaximumWidth(max(width_spin))
            # self._dblspin[i].setMaximumWidth(max(width_dblspin))

    def _assign_stretch(self, row=None, col=None):
        if row is None:
            row = 0
        if col is None:
            col = 0
        if self.ndim == 2:
            r0, r1 = self.itool._stretch_factors[0]
            c0, c1 = self.itool._stretch_factors[1]
            row_factor = (r0 - row, r1 + row)
            col_factor = (c0 - col, c1 + col)
        elif self.ndim >= 3:
            r0, r1, r2 = self.itool._stretch_factors[0]
            c0, c1, c2 = self.itool._stretch_factors[1]
            row_factor = (r0, r1 - row, r2 + row)
            col_factor = (c0, c1 - col, c2 + col)
        self.itool._update_stretch(row=row_factor, col=col_factor)

    def _assign_snap(self, value):
        self.itool.snap = value

    def _index_changed(self, axis, index):
        self._dblspin[axis].blockSignals(True)
        self.itool.set_index(axis, index)
        self._dblspin[axis].setValue(self.itool.data_coords[axis][index])
        self._dblspin[axis].blockSignals(False)

    def _value_changed(self, axis, value):
        self._spin[axis].blockSignals(True)
        self.itool.set_value(axis, value)
        self._spin[axis].setValue(self.itool._last_ind[axis])
        self._spin[axis].blockSignals(False)

    def update_spin(self, index, value):
        for i in range(self.ndim):
            self._spin[i].blockSignals(True)
            self._dblspin[i].blockSignals(True)
            self._spin[i].setValue(index[i])
            self._dblspin[i].setValue(value[i])
            self._spin[i].blockSignals(False)
            self._dblspin[i].blockSignals(False)


class itoolColorControls(QtWidgets.QWidget):
    def __init__(self, itool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.itool = itool
        self.layout = FlowLayout(self)
        self._cmap_group = BorderlessGroupBox(self, objectName="CmapGroup")
        # self._button_group = BorderlessGroupBox(self, objectName="ClrCntrls")
        self.initialize_widgets()

    def initialize_widgets(self):
        cmap_layout = InnerQHBoxLayout(self._cmap_group)
        # button_layout = InnerQHBoxLayout(self._button_group)
        # cmap_layout.setContentsMargins(0, 0, 0, 0)

        self._gamma_spin = QtWidgets.QDoubleSpinBox(
            toolTip="Colormap gamma",
            singleStep=0.01,
            value=self.itool.gamma,
            stepType=QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType,
        )

        self._gamma_spin.setRange(0.01, 100.0)
        self._gamma_spin.valueChanged.connect(self.set_cmap)
        gamma_label = QtWidgets.QLabel("γ")
        gamma_label.setBuddy(self._gamma_spin)

        self.gamma_scale = lambda y: 1000 * np.log10(y)
        self.gamma_scale_inv = lambda x: np.power(10, x / 1000)
        self._gamma_slider = QtWidgets.QSlider(
            toolTip="Colormap gamma",
            value=self.gamma_scale(self.itool.gamma),
            orientation=QtCore.Qt.Horizontal,
        )
        self._gamma_slider.setRange(
            int(self.gamma_scale(self._gamma_spin.minimum())),
            int(self.gamma_scale(self._gamma_spin.maximum())),
        )
        self._gamma_slider.valueChanged.connect(
            lambda x: self._gamma_spin.setValue(self.gamma_scale_inv(x))
        )

        self._cmap_combo = ColorMapComboBox(self, maximumWidth=175)
        if isinstance(self.itool.cmap, str):
            self._cmap_combo.setDefaultCmap(self.itool.cmap)

        self._cmap_combo.textActivated.connect(self._cmap_combo_changed)

        self._cmap_r_button = IconButton(
            self,
            on="invert",
            off="invert_off",
            checkable=True,
            toolTip="Invert colormap",
        )
        self._cmap_r_button.setChecked(self.itool.reverse)
        self._cmap_r_button.toggled.connect(self.set_cmap)

        self._cmap_mode_button = IconButton(
            self, on="contrast", checkable=True, toolTip="High contrast mode"
        )
        self._cmap_mode_button.toggled.connect(self.set_cmap)

        self._cmap_lock_button = IconButton(
            self, on="unlock", off="lock", checkable=True, toolTip="Lock colors"
        )
        self._cmap_lock_button.toggled.connect(self.itool.set_clim_lock)

        self._cbar_show_button = IconButton(
            self, on="colorbar", checkable=True, toolTip="Show colorbar"
        )
        self._cbar_show_button.toggled.connect(self.itool.toggle_colorbar)

        self._zero_center_button = IconButton(
            self,
            on="zero_center",
            checkable=True,
            checked=self.itool.zero_centered,
            toolTip="Center colormap at zero",
        )
        self._zero_center_button.toggled.connect(
            lambda z: self.itool.set_cmap(zero_centered=z)
        )

        axes_names = [
            "Main Image",
            "X Profile",
            "Y Profile",
            "Z Profile",
            "Horiz Slice",
            "Vert Slice",
            "T Profile",
        ]
        self._hide_button = tuple(
            QtWidgets.QPushButton(
                self, text=axes_names[i], clicked=lambda i=i: self.itool.toggle_axes(i)
            )
            for i in range(len(self.itool.axes))
        )
        self._axes_visibility_button = IconButton(self, on="table_eye", checkable=True)
        self._axes_visibility_button.toggled.connect(
            lambda v: [hb.setVisible(v) for hb in self._hide_button]
        )

        colors_button = IconButton(
            self, on="palette", clicked=self._color_button_clicked
        )
        # style_combo = QtWidgets.QComboBox(toolTip="Qt style")
        # style_combo.addItems(qt_style_names())
        # style_combo.textActivated.connect(QtWidgets.QApplication.setStyle)
        # style_combo.setCurrentText("Fusion")

        cmap_layout.addWidget(gamma_label)
        cmap_layout.addWidget(self._gamma_spin)
        cmap_layout.addWidget(self._gamma_slider)

        self.layout.addWidget(self._cmap_combo)
        self.layout.addWidget(self._cmap_group)

        self.layout.addWidget(self._cmap_r_button)
        self.layout.addWidget(self._cmap_lock_button)
        self.layout.addWidget(self._cmap_mode_button)
        self.layout.addWidget(self._cbar_show_button)
        self.layout.addWidget(self._zero_center_button)
        self.layout.addWidget(self._axes_visibility_button)
        self.layout.addWidget(colors_button)
        # self.layout.addWidget(style_combo)

        for hb in self._hide_button:
            hb.setMaximumWidth(hb.fontMetrics().boundingRect(hb.text()).width() + 15)
        for hb in self._hide_button:
            self.layout.addWidget(hb)
            hb.setVisible(False)

        # self._cmap_group.setSizePolicy(QtWidgets.QSizePolicy.Minimum,
        #    QtWidgets.QSizePolicy.Minimum)

        # self.layout.addWidget(self._button_group)

    def _cmap_combo_changed(self, text=None):
        if text == "Load all...":
            self._cmap_combo.load_all()
        else:
            self.set_cmap(name=text)

    def set_cmap(self, name=None):
        reverse = self._cmap_r_button.isChecked()
        gamma = self._gamma_spin.value()
        self._gamma_slider.blockSignals(True)
        self._gamma_slider.setValue(self.gamma_scale(gamma))
        self._gamma_slider.blockSignals(False)
        if isinstance(name, str):
            cmap = name
        else:
            cmap = self._cmap_combo.currentText()
        mode = self._cmap_mode_button.isChecked()
        self.itool.set_cmap(cmap, gamma=gamma, reverse=reverse, high_contrast=mode)

    def _color_button_clicked(self, s):
        # print("click", s)
        dialog = ImageToolColors(self)
        if dialog.exec():
            # print("Success!")
            pass
        else:
            pass
            # print("Cancel!")


class ColorMapComboBox(QtWidgets.QComboBox):
    LOAD_ALL_TEXT = "Load all..."

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setPlaceholderText("Select colormap...")
        self.setToolTip("Colormap")
        w, h = 64, 16
        self.setIconSize(QtCore.QSize(w, h))
        for name in pg_colormap_names("mpl"):
            # for name in pg_colormap_names("local"):
            self.addItem(name)
        self.insertItem(0, self.LOAD_ALL_TEXT)
        self.thumbnails_loaded = False
        self.currentIndexChanged.connect(self.load_thumbnail)
        self.default_cmap = None

    def load_thumbnail(self, index):
        if not self.thumbnails_loaded:
            text = self.itemText(index)
            try:
                self.setItemIcon(index, QtGui.QIcon(pg_colormap_to_QPixmap(text)))
            except KeyError:
                pass

    def load_all(self):
        self.clear()
        for name in pg_colormap_names("all"):
            self.addItem(QtGui.QIcon(pg_colormap_to_QPixmap(name)), name)

    # https://forum.qt.io/topic/105012/qcombobox-specify-width-less-than-content/11
    def showPopup(self):
        maxWidth = self.maximumWidth()
        if maxWidth and maxWidth < 16777215:
            self.setPopupMinimumWidthForItems()
        if not self.thumbnails_loaded:
            for i in range(self.count()):
                self.load_thumbnail(i)
            self.thumbnails_loaded = True
        super().showPopup()

    def setPopupMinimumWidthForItems(self):
        view = self.view()
        fm = self.fontMetrics()
        maxWidth = max(fm.width(self.itemText(i)) for i in range(self.count()))
        if maxWidth:
            view.setMinimumWidth(maxWidth)

    def hidePopup(self):
        self.activated.emit(self.currentIndex())
        self.textActivated.emit(self.currentText())
        self.currentIndexChanged.emit(self.currentIndex())
        self.currentTextChanged.emit(self.currentText())
        super().hidePopup()

    def setDefaultCmap(self, cmap: str):
        self.default_cmap = cmap
        self.setCurrentText(cmap)

    def resetCmap(self):
        if self.default_cmap is not None:
            self.setCurrentText(self.default_cmap)


class itoolBinningControls(QtWidgets.QWidget):
    def __init__(self, itool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.itool = itool
        self.ndim = self.itool.data_ndim
        self.layout = FlowLayout(self)
        # self._bin_group = BorderlessGroupBox(self, objectName="BinGroup")
        self.initialize_widgets()
        self.update_content()
        self.itool.sigDataChanged.connect(self.update_content)

    def initialize_widgets(self):
        # bin_layout = InnerQHBoxLayout(self._bin_group)
        self._spinlabels = tuple(QtWidgets.QLabel(self) for _ in range(self.ndim))
        self._spin = tuple(QtWidgets.QSpinBox(self) for _ in range(self.ndim))
        self._reset = QtWidgets.QPushButton("Reset")
        self._reset.clicked.connect(self._navg_reset)
        for i in range(self.ndim):
            self._spin[i].setSingleStep(2)
            self._spin[i].setValue(1)
            self._spin[i].setWrapping(False)
            self._spin[i].valueChanged.connect(
                lambda n, axis=i: self.itool.set_navg(axis, n)
            )
        for i in range(self.ndim):
            self.layout.addWidget(self._spinlabels[i])
            self.layout.addWidget(self._spin[i])
            # bin_layout.addSpacing( , 20)
        self.layout.addWidget(self._reset)
        # bin_layout.addStretch()
        # self.layout.addWidget(self._bin_group)

    def _navg_reset(self):
        for i in range(self.ndim):
            self._spin[i].blockSignals(True)
            self._spin[i].setValue(1)
            self._spin[i].blockSignals(False)
        self.itool._refresh_navg(reset=True)

    def update_content(self):
        ndim = self.itool.data_ndim
        if ndim != self.ndim:
            self.layout.clear()
            self.ndim = ndim
            self.initialize_widgets()
        for i in range(self.ndim):
            self._spin[i].blockSignals(True)
            self._spinlabels[i].setText(self.itool.data_dims[i])
            self._spin[i].setRange(1, self.itool.data_shape[i] - 1)
            self._spin[i].setValue(self.itool.avg_win[i])
            self._spin[i].blockSignals(False)
        self.itool._refresh_navg()


# from qtpy import QtWidgets, QtCore
class ImageTool(QtWidgets.QMainWindow):
    def __init__(self, data, title=None, *args, **kwargs):
        super().__init__()
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.qapp = QtCore.QCoreApplication.instance()
        self._main = QtWidgets.QWidget(self)
        self.data = parse_data(data)
        if title is None:
            title = self.data.name
        self.setWindowTitle(title)
        self.setCentralWidget(self._main)
        self.layout = QtWidgets.QVBoxLayout(self._main)
        self.layout.setContentsMargins(QtCore.QMargins(0, 0, 0, 0))
        self.layout.setSpacing(0)
        self.data_ndim = self.data.ndim

        self.itool = pg_itool(self.data, *args, **kwargs)

        self.tab1 = itoolCursorControls(self.itool)
        self.tab2 = itoolColorControls(self.itool)
        self.tab3 = itoolBinningControls(self.itool)

        self.dockarea = DockArea()

        self.dock1 = ItoolDock(
            "Cursor", widget=self.tab1, size=(1, 5), autoOrientation=False
        )
        self.dock2 = ItoolDock(
            "Appearance", widget=self.tab2, size=(1, 5), autoOrientation=False
        )
        self.dock3 = ItoolDock(
            "Binning", widget=self.tab3, size=(1, 5), autoOrientation=False
        )

        self.dock1.layout.setContentsMargins(5, 5, 5, 5)
        self.dock2.layout.setContentsMargins(5, 5, 5, 5)
        self.dock3.layout.setContentsMargins(5, 5, 5, 5)
        self.dockarea.addDock(self.dock3)
        self.dockarea.addDock(self.dock2, "above", self.dock3)
        self.dockarea.addDock(self.dock1, "above", self.dock2)
        self.dockarea.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum
        )

        self.layout.addWidget(self.dockarea)
        self.layout.addWidget(self.itool)
        self.resize(700, 700)

        # Shortcut: (Description, Action)
        self.keyboard_shortcuts = {
            "R": ("Reverse colormap", self.tab2._cmap_r_button.click),
            "L": ("Lock color levels", self.tab2._cmap_lock_button.click),
            "S": ("Toggle cursor snap", self.tab1._snap_button.click),
            "T": ("Transpose main image", self.tab1._transpose_button[1].click),
            "Ctrl+A": ("View All", lambda: self.itool.autoRange()),
        }
        for k, v in self.keyboard_shortcuts.items():
            sc = QtGui.QShortcut(QtGui.QKeySequence(k), self)
            sc.activated.connect(v[-1])

        self.itool.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.itool.setFocus()

    def changeEvent(self, evt):
        if evt.type() == QtCore.QEvent.PaletteChange:
            self.qapp.setStyle(self.qapp.style().name())
        super().changeEvent(evt)

    def tab_changed(self, i):
        pass
        # if i == self.tabwidget.indexOf(self.tab3):
        # lazy loading
        # self.tab3.initialize_functions()


def itool_(data, execute=None, *args, **kwargs):
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)
    qapp.setStyle("Fusion")

    if isinstance(data, list | tuple):
        win = ()
        for d in data:
            win += (ImageTool(d, *args, **kwargs),)
        for w in win:
            w.show()
        win[-1].activateWindow()
        win[-1].raise_()
    else:
        win = ImageTool(data, *args, **kwargs)
        win.show()
        win.activateWindow()
        win.raise_()
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
    return win
