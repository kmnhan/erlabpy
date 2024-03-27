import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.qt_compat import QtCore, QtWidgets
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Ellipse

from erlab.interactive.utilities import parse_data
from erlab.plotting.bz import plot_hex_bz
from erlab.plotting.colors import CenteredInversePowerNorm


def plot_bz_tise2(ax=None, a=3.54, pockets=False, aspect=1, rotate=0, **kwargs):
    """
    Plots a TiSe2 BZ on top of specified axis.
    """
    kwargs.setdefault("alpha", 1)
    kwargs.setdefault("color", "w")
    kwargs.setdefault("linestyle", "-")
    kwargs.setdefault("linewidth", 0.75)
    kwargs.setdefault("zorder", 5)
    # kwargs.setdefault('ax',plt.gca())

    # warnings.warn('Deprecated plot_bz_tise2')
    if ax is None:
        ax = plt.gca()
    plot_hex_bz(a=a, rotate=rotate, ax=ax)
    if pockets is True:
        color = kwargs.pop("color", None)
        kwargs["edgecolor"] = color
        kwargs["facecolor"] = "none"

        ln = 2 * np.pi / (a * 3)
        width = 0.25
        center = (0, np.sqrt(3) * ln)
        offset = (0, 0)
        for i, ang in enumerate(
            [np.deg2rad(rotate - 30 + a) for a in [0, 60, 120, 180, 240, 300]]
        ):
            x = np.cos(ang) * center[0] - np.sin(ang) * center[1]
            y = np.sin(ang) * center[0] + np.cos(ang) * center[1]
            if aspect == 1:
                p = Circle((x + offset[0], y + offset[1]), radius=width / 2, **kwargs)
            else:
                p = Ellipse(
                    (x + offset[0], y + offset[1]),
                    width=width,
                    height=width * aspect,
                    angle=np.rad2deg(ang),
                    **kwargs,
                )
            ax.add_patch(p)

        # p1 = Ellipse(
        #     (0, 2 * np.pi / (np.sqrt(3) * a)),
        #     width=0.2,
        #     height=0.2 * pocket_length,
        #     **kwargs
        # )
        # p2 = copy.deepcopy(p1)
        # p3 = copy.deepcopy(p1)
        # p4 = copy.deepcopy(p1)
        # p5 = copy.deepcopy(p1)
        # p6 = copy.deepcopy(p1)
        # p1.set_transform(Affine2D().rotate_deg(rotate - 30 + 0) + ax.transData)
        # p2.set_transform(Affine2D().rotate_deg(rotate - 30 + 60) + ax.transData)
        # p3.set_transform(Affine2D().rotate_deg(rotate - 30 + 120) + ax.transData)
        # p4.set_transform(Affine2D().rotate_deg(rotate - 30 + 180) + ax.transData)
        # p5.set_transform(Affine2D().rotate_deg(rotate - 30 + 240) + ax.transData)
        # p6.set_transform(Affine2D().rotate_deg(rotate - 30 + 300) + ax.transData)
        # ax.add_patch(p1)
        # ax.add_patch(p2)
        # ax.add_patch(p3)
        # ax.add_patch(p4)
        # ax.add_patch(p5)
        # ax.add_patch(p6)


class kTool(QtWidgets.QMainWindow):
    def __init__(
        self,
        data,
        bounds: dict | None = None,
        resolution: dict | None = None,
        gamma=0.5,
        cmap="twilight",
        plot_hex_bz=True,
        diff_data=None,
        a=3.54,
        rotate=90,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if bounds is None:
            bounds = {"kx": [-1.5, 1.5], "ky": [-1.5, 1.5], "kp": [-1.5, 1.5]}
        if resolution is None:
            resolution = {"kx": 0.02, "ky": 0.02}
        self.diff_data = diff_data
        self.data = parse_data(data)

        def _get_middle_index(x):
            return len(x) // 2 - (1 if len(x) % 2 == 0 else 0)

        self.has_eV = "eV" in self.data.dims
        self.show_eV = self.has_eV and self.data.ndim == 3
        if self.show_eV:
            self.data_all = self.data.copy(deep=True)
            self.coord_z = self.data_all["eV"].values
            self.ind_z = _get_middle_index(self.coord_z)
            self.data = self.data_all.isel(eV=self.ind_z)
            self.lims_z = (self.coord_z[0], self.coord_z[-1])
            self.inc_z = self.coord_z[1] - self.coord_z[0]
        self.dim_x, self.dim_y = self.data.dims
        self.coord_x, self.coord_y = (
            self.data[self.dim_x].values,
            self.data[self.dim_y].values,
        )
        self.lims_x, self.lims_y = (
            (self.coord_x[0], self.coord_x[-1]),
            (self.coord_y[0], self.coord_y[-1]),
        )
        self.vals = self.data.values
        self.bounds = bounds
        self.resolution = resolution
        self.kxy = self.data.kspace.convert(
            bounds=self.bounds, resolution=self.resolution
        )
        self.gamma = gamma
        self.cmap = cmap
        self.visible = True
        self.background = None
        improps = dict(
            animated=True,
            visible=True,
            interpolation="none",
            aspect="auto",
            origin="lower",
            norm=colors.PowerNorm(self.gamma),
            cmap=self.cmap,
        )

        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        self.layout = QtWidgets.QVBoxLayout(self._main)

        self.new_offsets_rad = {}
        self.new_offsets_deg = {}

        self.canvas = FigureCanvas(
            Figure(figsize=(8, 16), dpi=100, layout="constrained")
        )
        self.addToolBar(
            QtCore.Qt.BottomToolBarArea, NavigationToolbar(self.canvas, self)
        )

        gs = self.canvas.figure.add_gridspec(1, 2)
        self.ax0 = self.canvas.figure.add_subplot(gs[0])
        self.ax1 = self.canvas.figure.add_subplot(gs[1])
        self.ax0.grid()
        self.ax1.grid()
        self.ax0.set_aspect("equal")
        self.ax1.set_aspect("equal")

        self.im_r = self.ax0.imshow(
            self.vals, extent=(*self.lims_x, *self.lims_y), **improps
        )

        self.dispmode = QtWidgets.QComboBox()
        self.dispmode.addItems(["diff", "converted", "reference"])
        self.dispmode.setCurrentIndex(0)
        self.dispmode.currentIndexChanged.connect(self._update_all)

        self.im_k = self.ax1.imshow(self.disp_values, extent=self.extent, **improps)
        self.ax0.set_xlabel(self.dim_x)
        self.ax0.set_ylabel(self.dim_y)
        self.ax1.set_xlabel(self.dim_kx)
        self.ax1.set_ylabel(self.dim_ky)
        self.ax1.axline(
            (0, 0), (np.pi / 3.54, -np.pi / np.sqrt(3) / 3.54), ls="--", lw=0.75, c="w"
        )
        self.ax1.axline(
            (0, 0), (-np.pi / 3.54, -np.pi / np.sqrt(3) / 3.54), ls="--", lw=0.75, c="w"
        )

        self.offsetpanel = QtWidgets.QWidget()
        offsetpanelcontent = QtWidgets.QHBoxLayout()

        self.offsetcoords = ["theta", "phi", "beta", "psi", "chi"]

        spin0label = QtWidgets.QLabel(self.offsetcoords[0])
        spin1label = QtWidgets.QLabel(self.offsetcoords[1])
        spin2label = QtWidgets.QLabel(self.offsetcoords[2])
        spin3label = QtWidgets.QLabel(self.offsetcoords[3])
        spin4label = QtWidgets.QLabel(self.offsetcoords[4])

        self.spin0 = QtWidgets.QDoubleSpinBox()
        self.spin1 = QtWidgets.QDoubleSpinBox()
        self.spin2 = QtWidgets.QDoubleSpinBox()
        self.spin3 = QtWidgets.QDoubleSpinBox()
        self.spin4 = QtWidgets.QDoubleSpinBox()
        self.spin0.setRange(-180, 180)
        self.spin1.setRange(-180, 180)
        self.spin2.setRange(-180, 180)
        self.spin3.setRange(-180, 180)
        self.spin4.setRange(-180, 180)
        self.spin0.setSingleStep(0.01)
        self.spin1.setSingleStep(0.01)
        self.spin2.setSingleStep(0.01)
        self.spin3.setSingleStep(0.01)
        self.spin4.setSingleStep(0.01)
        self.spin0.setDecimals(3)
        self.spin1.setDecimals(3)
        self.spin2.setDecimals(3)
        self.spin3.setDecimals(3)
        self.spin4.setDecimals(3)
        self.spin0.setValue(0.0)
        self.spin1.setValue(0.0)
        self.spin2.setValue(0.0)
        self.spin3.setValue(0.0)
        self.spin4.setValue(0.0)
        self.spin0.valueChanged.connect(lambda v: self._spinchanged(0, v))
        self.spin1.valueChanged.connect(lambda v: self._spinchanged(1, v))
        self.spin2.valueChanged.connect(lambda v: self._spinchanged(2, v))
        self.spin3.valueChanged.connect(lambda v: self._spinchanged(3, v))
        self.spin4.valueChanged.connect(lambda v: self._spinchanged(4, v))

        offsetpanelcontent.addWidget(spin0label)
        offsetpanelcontent.addWidget(self.spin0)
        offsetpanelcontent.addWidget(spin1label)
        offsetpanelcontent.addWidget(self.spin1)
        offsetpanelcontent.addWidget(spin2label)
        offsetpanelcontent.addWidget(self.spin2)
        offsetpanelcontent.addWidget(spin3label)
        offsetpanelcontent.addWidget(self.spin3)
        offsetpanelcontent.addWidget(spin4label)
        offsetpanelcontent.addWidget(self.spin4)
        offsetpanelcontent.addStretch()
        self.offsetpanel.setLayout(offsetpanelcontent)

        self.colorstab = QtWidgets.QWidget()
        colorstabcontent = QtWidgets.QHBoxLayout()
        gammalabel = QtWidgets.QLabel("g")
        gammaspin = QtWidgets.QDoubleSpinBox()
        gammaspin.setToolTip("Colormap Gamma")
        gammaspin.setSingleStep(0.05)
        gammaspin.setValue(0.5)
        gammaspin.setMinimum(0.00001)
        gammaspin.valueChanged.connect(self._set_gamma)
        colormaps = QtWidgets.QComboBox()
        colormaps.setToolTip("Colormap")
        colormaps.addItems(plt.colormaps())
        colormaps.setCurrentIndex(colormaps.findText(self.cmap))
        colormaps.currentTextChanged.connect(self._set_cmap)
        colorstabcontent.addWidget(gammalabel)
        colorstabcontent.addWidget(gammaspin)
        colorstabcontent.addWidget(colormaps)
        colorstabcontent.addStretch()
        self.colorstab.setLayout(colorstabcontent)

        self.boundstab = QtWidgets.QWidget()
        boundstabcontent = QtWidgets.QHBoxLayout(self.boundstab)
        self.boundsgroup = QtWidgets.QGroupBox("Bounds")
        self.boundsgrid = QtWidgets.QGridLayout(self.boundsgroup)
        self.kxminlabel = QtWidgets.QLabel(self.boundsgroup)
        self.boundsgrid.addWidget(self.kxminlabel, 0, 0, 1, 1)
        self.kxminspin = QtWidgets.QDoubleSpinBox(self.boundsgroup)
        self.boundsgrid.addWidget(self.kxminspin, 0, 1, 1, 1)
        self.kxmaxlabel = QtWidgets.QLabel(self.boundsgroup)
        self.boundsgrid.addWidget(self.kxmaxlabel, 0, 2, 1, 1, QtCore.Qt.AlignHCenter)
        self.kxmaxspin = QtWidgets.QDoubleSpinBox(self.boundsgroup)
        self.boundsgrid.addWidget(self.kxmaxspin, 0, 3, 1, 1)
        self.kyminlabel = QtWidgets.QLabel(self.boundsgroup)
        self.boundsgrid.addWidget(self.kyminlabel, 1, 0, 1, 1)
        self.kyminspin = QtWidgets.QDoubleSpinBox(self.boundsgroup)
        self.boundsgrid.addWidget(self.kyminspin, 1, 1, 1, 1)
        self.kymaxlabel = QtWidgets.QLabel(self.boundsgroup)
        self.boundsgrid.addWidget(self.kymaxlabel, 1, 2, 1, 1, QtCore.Qt.AlignHCenter)
        self.kymaxspin = QtWidgets.QDoubleSpinBox(self.boundsgroup)
        self.boundsgrid.addWidget(self.kymaxspin, 1, 3, 1, 1)
        boundstabcontent.addWidget(self.boundsgroup)
        self.resolutiongroup = QtWidgets.QGroupBox("Resolution")
        self.resgrid = QtWidgets.QGridLayout(self.resolutiongroup)
        self.kxreslabel = QtWidgets.QLabel(self.resolutiongroup)
        self.resgrid.addWidget(self.kxreslabel, 0, 0, 1, 1)
        self.kxresspin = QtWidgets.QDoubleSpinBox(self.resolutiongroup)
        self.resgrid.addWidget(self.kxresspin, 0, 1, 1, 1)
        self.kyreslabel = QtWidgets.QLabel(self.resolutiongroup)
        self.resgrid.addWidget(self.kyreslabel, 1, 0, 1, 1)
        self.kyresspin = QtWidgets.QDoubleSpinBox(self.resolutiongroup)
        self.resgrid.addWidget(self.kyresspin, 1, 1, 1, 1)
        boundstabcontent.addWidget(self.resolutiongroup)
        self.kxminlabel.setText("kx")
        self.kxmaxlabel.setText("to")
        self.kyminlabel.setText("ky")
        self.kymaxlabel.setText("to")
        self.kxreslabel.setText("kx")
        self.kyreslabel.setText("ky")
        self.kxminspin.setSingleStep(0.05)
        self.kxmaxspin.setSingleStep(0.05)
        self.kyminspin.setSingleStep(0.05)
        self.kymaxspin.setSingleStep(0.05)
        self.kxminspin.setRange(-50, 50)
        self.kxmaxspin.setRange(-50, 50)
        self.kyminspin.setRange(-50, 50)
        self.kymaxspin.setRange(-50, 50)
        self.kxminspin.setValue(self.bounds["kx"][0])
        self.kxmaxspin.setValue(self.bounds["kx"][1])
        self.kyminspin.setValue(self.bounds["ky"][0])
        self.kymaxspin.setValue(self.bounds["ky"][1])
        self.kxminspin.valueChanged.connect(lambda v: self._set_bounds("kx", 0, v))
        self.kxmaxspin.valueChanged.connect(lambda v: self._set_bounds("kx", 1, v))
        self.kyminspin.valueChanged.connect(lambda v: self._set_bounds("ky", 0, v))
        self.kymaxspin.valueChanged.connect(lambda v: self._set_bounds("ky", 1, v))
        self.kxresspin.setValue(self.resolution["kx"])
        self.kyresspin.setValue(self.resolution["ky"])
        self.kxresspin.setSingleStep(0.001)
        self.kyresspin.setSingleStep(0.001)
        self.kxresspin.setMinimum(0.001)
        self.kyresspin.setMinimum(0.001)
        self.kxresspin.valueChanged.connect(lambda v: self._set_resolution("kx", v))
        self.kyresspin.valueChanged.connect(lambda v: self._set_resolution("ky", v))
        self.kxresspin.setDecimals(3)
        self.kyresspin.setDecimals(3)

        if self.diff_data is not None:
            boundstabcontent.addWidget(self.dispmode)

        self.tabwidget = QtWidgets.QTabWidget()
        self.tabwidget.addTab(self.offsetpanel, "Offsets")
        self.tabwidget.addTab(self.boundstab, "Bounds")
        self.tabwidget.addTab(self.colorstab, "Colors")
        self.tabwidget.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum
        )
        self.canvas.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Expanding
        )

        self.layout.addWidget(self.tabwidget)
        if self.show_eV:
            zvaluepanel = QtWidgets.QWidget()
            zvaluecontent = QtWidgets.QHBoxLayout(zvaluepanel)
            self.zspin = QtWidgets.QDoubleSpinBox()
            self.zspin.setSingleStep(self.inc_z)
            self.zspin.setRange(*self.lims_z)
            self.zspin.setValue(self.coord_z[self.ind_z])
            self.zspin.valueChanged.connect(self._zspinchanged)
            self.zslider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.zslider.setSingleStep(1)
            self.zslider.setPageStep(10)
            self.zslider.setMinimum(0)
            self.zslider.setMaximum(len(self.coord_z) - 1)
            self.zslider.setValue(self.ind_z)
            self.zslider.valueChanged.connect(self._zsliderchanged)
            zvaluecontent.addWidget(self.zspin)
            zvaluecontent.addWidget(self.zslider)
            self.layout.addWidget(zvaluepanel)

        self.layout.addWidget(self.canvas)
        # self.canvas.mpl_connect('draw_event', self.clear)
        if plot_hex_bz:
            plot_bz_tise2(a=a, ax=self.ax1, rotate=rotate, pockets=True)
        self.canvas.draw()
        self.spin4.setValue(np.rad2deg(data.chi))
        # self.im_r.set_visible(self.visible)
        # self.im_k.set_visible(self.visible)

    def clear(self, event):
        self.im_r.set_visible(False)
        self.im_k.set_visible(False)
        # self.background = self.canvas.copy_from_bbox(self.canvas.figure.bbox)

    def closeEvent(self, event):
        # do stuff
        print("ktool closed with offsets")
        print("in radians:")
        print(self.new_offsets_rad)
        print("in degrees:")
        print(self.new_offsets_deg)
        event.accept()

    def _update_plots(self):
        self.im_r.set_visible(self.visible)
        self.im_k.set_visible(self.visible)
        # if self.background is not None:
        # self.canvas.restore_region(self.background)
        self.ax0.draw_artist(self.im_r)
        self.ax1.draw_artist(self.im_k)
        # self.canvas.blit()
        self.canvas.draw()

    def _set_resolution(self, ax, value):
        self.resolution[ax] = value
        self._update_kxy()
        self._update_plots()

    def _update_all(self):
        self._update_disp()
        self._update_plots()

    def _set_bounds(self, ax, ind, value):
        self.bounds[ax][ind] = value
        self.bounds["kp"][ind] = value
        self._update_kxy()
        self._update_plots()
        if (ax == "kx") & (ind == 0):
            self.kxmaxspin.setMinimum(value)
        elif (ax == "kx") & (ind == 1):
            self.kxminspin.setMaximum(value)
        elif (ax == "ky") & (ind == 0):
            self.kymaxspin.setMinimum(value)
        elif (ax == "ky") & (ind == 1):
            self.kyminspin.setMaximum(value)

    def _zspinchanged(self, value):
        self.ind_z = np.rint((value - self.lims_z[0]) / self.inc_z).astype(int)
        self.zslider.blockSignals(True)
        self.zslider.setValue(self.ind_z)
        self.zslider.blockSignals(False)
        self._update_data()
        self._update_kxy()
        self.im_r.set_norm(colors.PowerNorm(self.gamma))
        self._update_plots()

    def _zsliderchanged(self, value):
        self.ind_z = value
        self.zspin.blockSignals(True)
        self.zspin.setValue(self.coord_z[self.ind_z])
        self.zspin.blockSignals(False)
        self._update_data()
        self._update_kxy()
        self._update_plots()

    def _update_data(self):
        self.data = self.data_all.isel(eV=self.ind_z)
        self.vals = self.data.values
        self.im_r.set_data(self.vals)
        self.im_r.set_norm(colors.PowerNorm(self.gamma))

    def _set_gamma(self, gamma):
        self.gamma = gamma
        self.im_r.set_norm(colors.PowerNorm(self.gamma))
        if self.diff_data is not None:
            self.im_k.set_norm(CenteredInversePowerNorm(self.gamma))
        else:
            self.im_k.set_norm(colors.PowerNorm(self.gamma))
        self._update_plots()

    def _set_cmap(self, cmap):
        self.cmap = cmap
        self.im_r.set_cmap(self.cmap)
        self.im_k.set_cmap(self.cmap)
        self._update_plots()

    def _spinchanged(self, n, value):
        # self.data.S.apply_offsets({self.offsetcoords[n]:value*np.pi/180})
        self.new_offsets_rad[self.offsetcoords[n]] = np.deg2rad(value)
        self.new_offsets_deg[self.offsetcoords[n]] = np.around(value, 3)
        self.data.S.apply_offsets(self.new_offsets_rad)
        if self.show_eV:
            self.data_all.S.apply_offsets(self.new_offsets_rad)
        self._update_kxy()
        self._update_plots()

    @property
    def disp_values(self):
        self.dim_ky, self.dim_kx = self.kxy.dims
        self.coord_kx, self.coord_ky = (
            self.kxy[self.dim_kx].values,
            self.kxy[self.dim_ky].values,
        )
        self.lims_kx, self.lims_ky = (
            (self.coord_kx[0], self.coord_kx[-1]),
            (self.coord_ky[0], self.coord_ky[-1]),
        )
        if self.diff_data is not None:
            try:
                ref = (
                    self.diff_data.sel(
                        eV=self.data_all.eV[self.ind_z], method="nearest"
                    )
                    .sel(kx=slice(*self.lims_kx), ky=slice(*self.lims_ky))
                    .values
                )
                match self.dispmode.currentText():
                    case "diff":
                        return self.kxy.values - ref
                    case "converted":
                        return self.kxy.values
                    case _:
                        return ref
            except ValueError:
                pass
        return self.kxy.values

    @property
    def extent(self):
        return (*self.lims_kx, *self.lims_ky)

    def _update_disp(self):
        self.im_k.set_data(self.disp_values)
        self.im_k.set_extent(self.extent)
        if self.diff_data is not None:
            self.im_k.set_norm(CenteredInversePowerNorm(self.gamma))
        else:
            self.im_k.set_norm(colors.PowerNorm(self.gamma))

    def _update_kxy(self):
        self.kxy = self.data.kspace.convert(
            bounds=self.bounds, resolution=self.resolution
        )
        self._update_disp()


def ktool(data, execute=True, *args, **kwargs):
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)
    with plt.rc_context(
        {
            "text.usetex": False,
            #     #  'mathtext.fontset':'stixsans',
            "font.size": 7,
            "font.family": "sans",
        }
    ):
        win = kTool(data, *args, **kwargs)
        win.show()
        win.activateWindow()
        win.raise_()
    if execute is None:
        execute = True
        try:
            shell = get_ipython().__class__.__name__  # type: ignore
            if shell == "ZMQInteractiveShell":
                execute = False
            elif shell == "TerminalInteractiveShell":
                execute = False
        except NameError:
            pass
    if execute:
        qapp.exec()
    qapp.setStyle("Fusion")
    return win
