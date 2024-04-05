__all__ = ["dtool"]

import sys

import numpy as np
import pyqtgraph as pg
from qtpy import QtCore, QtWidgets
from scipy.ndimage import gaussian_filter, uniform_filter

from erlab.analysis.image import curvature, minimum_gradient
from erlab.interactive.utilities import parse_data


class DerivativeTool(QtWidgets.QMainWindow):
    def __init__(self, data=None, *args, **kwargs):
        super().__init__()

        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        self.layout = QtWidgets.QGridLayout(self._main)
        plotwidget = pg.GraphicsLayoutWidget(show=True)

        self.hists = [
            pg.HistogramLUTItem(),
            pg.HistogramLUTItem(),
        ]
        self.ax1 = plotwidget.addPlot(0, 0, 1, 1)
        plotwidget.addItem(self.hists[0])
        plotwidget.nextRow()
        self.ax2 = plotwidget.addPlot(1, 0, 1, 1)
        plotwidget.addItem(self.hists[1])
        self.axes = [self.ax1, self.ax2]
        self.ax2.setYLink(self.ax1)
        self.ax2.setXLink(self.ax1)

        self.images = [
            pg.ImageItem(axisOrder="row-major"),
            pg.ImageItem(axisOrder="row-major"),
        ]
        for i, img in enumerate(self.images):
            self.axes[i].addItem(img)
            self.hists[i].setImageItem(img)
            # img.sigImageChanged.connect()

        self.use_gaussfilt = False

        self.win_list = [1, 1]
        self.smooth_num = 1
        self.smooth2_num = 1

        self._smooth_group = QtWidgets.QGroupBox("Smoothing")
        smooth_layout = QtWidgets.QGridLayout(self._smooth_group)
        self._smooth_mode_check = QtWidgets.QCheckBox(self._smooth_group)
        self._smooth_mode_check.setChecked(self.use_gaussfilt)
        smooth_mode_label = QtWidgets.QLabel("Gaussian filter")
        smooth_mode_label.setBuddy(self._smooth_mode_check)
        self._smooth_x_spin = QtWidgets.QSpinBox(self._smooth_group)
        self._smooth_y_spin = QtWidgets.QSpinBox(self._smooth_group)
        self._smooth_n_spin = QtWidgets.QSpinBox(self._smooth_group)
        self._smooth_n_spin.setRange(1, 100)
        smooth_x_spin_label = QtWidgets.QLabel("x")
        smooth_y_spin_label = QtWidgets.QLabel("y")
        smooth_n_spin_label = QtWidgets.QLabel("n")
        smooth_layout.addWidget(smooth_mode_label, 0, 0)
        smooth_layout.addWidget(self._smooth_mode_check, 0, 1)
        smooth_layout.addWidget(smooth_x_spin_label, 1, 0)
        smooth_layout.addWidget(self._smooth_x_spin, 1, 1)
        smooth_layout.addWidget(smooth_y_spin_label, 2, 0)
        smooth_layout.addWidget(self._smooth_y_spin, 2, 1)
        smooth_layout.addWidget(smooth_n_spin_label, 3, 0)
        smooth_layout.addWidget(self._smooth_n_spin, 3, 1)

        self._color_group = QtWidgets.QGroupBox("Color")
        color_layout = QtWidgets.QGridLayout(self._color_group)
        self._color_range_spin = (
            QtWidgets.QDoubleSpinBox(self._smooth_group),
            QtWidgets.QDoubleSpinBox(self._smooth_group),
        )
        for i, s in enumerate(self._color_range_spin):
            s.setRange(0, 100)
            s.setValue(10)
            s.setSingleStep(0.1)
            s.setSuffix(" %")
            s.setDecimals(1)
            color_layout.addWidget(s, i, 0)
        self._color_range_spin[1].setValue(1)

        self._smooth2_group = QtWidgets.QGroupBox("2nd Smoothing")
        smooth2_layout = QtWidgets.QGridLayout(self._smooth2_group)
        self._smooth2_x_spin = QtWidgets.QSpinBox(self._smooth2_group)
        self._smooth2_y_spin = QtWidgets.QSpinBox(self._smooth2_group)
        self._smooth2_n_spin = QtWidgets.QSpinBox(self._smooth2_group)
        self._smooth2_x_spin_label = QtWidgets.QLabel("x")
        self._smooth2_y_spin_label = QtWidgets.QLabel("y")
        self._smooth2_n_spin_label = QtWidgets.QLabel("n")
        self._smooth2_n_spin.setRange(1, 100)
        smooth2_layout.addWidget(self._smooth2_x_spin_label, 1, 0)
        smooth2_layout.addWidget(self._smooth2_x_spin, 1, 1)
        smooth2_layout.addWidget(self._smooth2_y_spin_label, 2, 0)
        smooth2_layout.addWidget(self._smooth2_y_spin, 2, 1)
        smooth2_layout.addWidget(self._smooth2_n_spin_label, 3, 0)
        smooth2_layout.addWidget(self._smooth2_n_spin, 3, 1)

        self._deriv_group = QtWidgets.QGroupBox("Derivative")
        deriv_layout = QtWidgets.QGridLayout(self._deriv_group)
        self._deriv_axis_combo = QtWidgets.QComboBox(self._deriv_group)
        self._deriv_axis_combo.addItems(["x", "y"])
        deriv_layout.addWidget(self._deriv_axis_combo, 0, 0)

        self._curv_group = QtWidgets.QGroupBox("Curvature")
        curv_layout = QtWidgets.QGridLayout(self._curv_group)
        self._curv_alpha_spin = QtWidgets.QDoubleSpinBox(self._curv_group)
        self._curv_alpha_spin.setRange(0, 100)
        self._curv_alpha_spin.setDecimals(4)
        self._curv_alpha_spin.setValue(1.0)
        self._curv_alpha_spin.setSingleStep(0.001)
        self._curv_factor_spin = QtWidgets.QDoubleSpinBox(self._curv_group)
        self._curv_factor_spin.setRange(-1, 1)
        self._curv_factor_spin.setDecimals(4)
        self._curv_factor_spin.setValue(1.0)
        self._curv_factor_spin.setSingleStep(0.0001)
        curv_layout.addWidget(self._curv_alpha_spin, 0, 0)
        curv_layout.addWidget(self._curv_factor_spin, 1, 0)

        self._mingrad_group = QtWidgets.QGroupBox("Minimum Gradient")
        mingrad_layout = QtWidgets.QGridLayout(self._mingrad_group)
        self._mingrad_btn = QtWidgets.QPushButton("Calculate")
        # self._mingrad_delta_spin = QtWidgets.QSpinBox(self._mingrad_group)
        # self._mingrad_delta_spin.setRange(1, 100)
        # self._mingrad_delta_spin.setValue(1)
        # mingrad_layout.addWidget(self._mingrad_delta_spin, 0, 0)
        mingrad_layout.addWidget(self._mingrad_btn, 0, 0)

        self._deriv_tab = QtWidgets.QWidget()
        deriv_content = QtWidgets.QGridLayout(self._deriv_tab)
        deriv_content.addWidget(self._smooth2_group, 0, 0)
        deriv_content.addWidget(self._deriv_group, 0, 1)

        self._curv_tab = QtWidgets.QWidget()
        curv_content = QtWidgets.QGridLayout(self._curv_tab)
        curv_content.addWidget(self._curv_group, 0, 0)

        self._mingrad_tab = QtWidgets.QWidget()
        mingrad_content = QtWidgets.QGridLayout(self._mingrad_tab)
        mingrad_content.addWidget(self._mingrad_group, 0, 0)

        self.tabwidget = QtWidgets.QTabWidget()
        self.tabwidget.addTab(self._deriv_tab, "Derivative")
        self.tabwidget.addTab(self._curv_tab, "Curvature")
        self.tabwidget.addTab(self._mingrad_tab, "Minimum Gradient")
        self.tabwidget.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.Maximum,
        )

        self.layout.addWidget(self._smooth_group, 0, 0, 1, 1)
        self.layout.addWidget(self._color_group, 0, 1, 1, 1)
        self.layout.addWidget(self.tabwidget, 0, 2, 1, 1)
        self.layout.addWidget(plotwidget, 1, 0, 2, 3)

        self.use_curv = False
        self.use_mingrad = False
        self.use_deriv = True

        if data is not None:
            self.set_data(data)

        self._smooth_mode_check.stateChanged.connect(self.set_gauss_smooth)
        self._smooth_x_spin.valueChanged.connect(
            lambda val, ax=1: self.smooth_data(ax, val)
        )
        self._smooth_y_spin.valueChanged.connect(
            lambda val, ax=0: self.smooth_data(ax, val)
        )
        self._smooth_n_spin.valueChanged.connect(self.set_smooth_num)
        self._smooth2_x_spin.valueChanged.connect(lambda: self.deriv_data())
        self._smooth2_y_spin.valueChanged.connect(lambda: self.deriv_data())
        self._smooth2_n_spin.valueChanged.connect(self.set_smooth2_num)
        self._deriv_axis_combo.currentTextChanged.connect(self.deriv_data)
        self._curv_alpha_spin.valueChanged.connect(self.curv_data)
        self._curv_factor_spin.valueChanged.connect(self.curv_data)
        self._mingrad_btn.clicked.connect(lambda: self.mingrad_data())
        # self._mingrad_delta_spin.valueChanged.connect(
        #     lambda delta: self.mingrad_data(delta=delta)
        # )
        for s in self._color_range_spin:
            s.valueChanged.connect(self._result_update_lims)

    def get_cutoff(self, data, cutoff=None):
        q3, q1 = np.percentile(data, [75, 25])
        ql = q1 - 1.5 * (q3 - q1)
        qu = q3 + 1.5 * (q3 - q1)
        i_qu = 100 * (data > qu).mean()
        i_ql = 100 * (data < ql).mean()
        for s in self._color_range_spin:
            s.blockSignals(True)
        self._color_range_spin[0].setMaximum(i_qu)
        self._color_range_spin[1].setMaximum(i_ql)
        for s in self._color_range_spin:
            s.blockSignals(False)
        if cutoff is None:
            # cutoff = [30, 30]
            cutoff = [s.value() for s in self._color_range_spin]
        else:
            try:
                cutoff = list(cutoff.__iter__)
            except AttributeError:
                cutoff = [cutoff] * 2

        pu, pl = np.percentile(data, [100 - cutoff[0], cutoff[1]])
        mn = max(min(pl, ql), data.min())
        mx = min(max(pu, qu), data.max())
        return [mn, mx]

    def set_gauss_smooth(self, value):
        self.use_gaussfilt = value
        self.smooth_data()

    def _result_update_lims(self):
        self.images[1].setImage(
            self.data_s_d.values,
            rect=self._lims_to_rect(1, 0),
            levels=self.get_cutoff(self.data_s_d.values),
        )

    def set_smooth_num(self, n):
        self.smooth_num = n
        self.smooth_data()

    def set_smooth2_num(self, n):
        self.smooth2_num = n
        self.deriv_data()

    # def mingrad_data(self, delta=1):
    def mingrad_data(self):
        self.use_curv = False
        self.use_mingrad = True
        self.use_deriv = False
        # self.data_s_d = self.data_s.copy(deep=True)
        # self.data_s_d = -minimum_gradient(self.data_s, delta=delta)
        self.data_s_d = -minimum_gradient(self.data_s)
        self._result_update_lims()

    def curv_data(self):
        self.use_curv = True
        self.use_deriv = False
        self.use_mingrad = False

        # self.data_s_d = self.data_s.copy(deep=True)
        self.data_s_d = curvature(
            # self.data_s / self.data_s.max(), alpha=alpha, beta=beta, values=True
            self.data_s,
            a0=self._curv_alpha_spin.value(),
            factor=self._curv_factor_spin.value(),
        )
        # m = self.data_s_d.max()
        # self.data_s_d /= m
        self._result_update_lims()

    def deriv_data(self, axis=None):
        self.use_curv = False
        self.use_mingrad = False
        self.use_deriv = True
        if axis is None:
            axis = self._deriv_axis_combo.currentText()
        if axis == "x":
            axis = 0
        elif axis == "y":
            axis = 1
        # axis = self.data_dims[axis]
        self.data_s_d = self.data_s.copy(deep=True)
        self.data_s_d.values = np.gradient(self.data_s.values, axis=axis)
        amount = [self._smooth2_y_spin.value(), self._smooth2_x_spin.value()]
        self.use_gaussfilt = self._smooth_mode_check.isChecked()
        self.data_s_d.values = self.smoothfunc(
            self.data_s_d.values, amount, self.smooth2_num
        )
        self.data_s_d.values = np.gradient(self.data_s_d.values, axis=axis)
        # self.data_s_d = d1_along_axis(self.data_s_d, axis=axis)
        self._result_update_lims()

    def _get_middle_index(self, x):
        return len(x) // 2 - (1 if len(x) % 2 == 0 else 0)

    def smoothfunc(self, data, amount, num):
        if not self.use_gaussfilt:
            for _ in range(num):
                data = uniform_filter(data, amount, mode="constant", origin=(0, 0))
        else:
            sigma = [(amount[0] - 1) / 3, (amount[1] - 1) / 3]
            for _ in range(num):
                data = gaussian_filter(data, sigma=sigma, truncate=5)
        return data

    def smooth_data(self, axis=None, amount=None):
        if axis is not None:
            self.win_list[axis] = amount
        else:
            self.win_list = [self._smooth_y_spin.value(), self._smooth_x_spin.value()]
        self.use_gaussfilt = self._smooth_mode_check.isChecked()

        self.data_s = self.data.copy(deep=True)
        vals = self.data_s.values

        self.data_s.values = self.smoothfunc(vals, self.win_list, self.smooth_num)

        self.images[0].setImage(self.data_s.values, rect=self._lims_to_rect(1, 0))
        if self.use_deriv:
            self.deriv_data()
        elif self.use_curv:
            self.curv_data()
        elif self.use_mingrad:
            self.mingrad_data()

    def _lims_to_rect(self, i, j):
        x = self.data_lims[i][0] - self.data_incs[i]
        y = self.data_lims[j][0] - self.data_incs[j]
        w = self.data_lims[i][-1] - x
        h = self.data_lims[j][-1] - y
        x += 0.5 * self.data_incs[i]
        y += 0.5 * self.data_incs[j]
        return QtCore.QRectF(x, y, w, h)

    def set_data(self, data):
        self.data = parse_data(data).fillna(0)
        self.data_s = self.data.copy(deep=True)
        self.vals = self.data.values
        self.data_dims = self.data.dims
        self.data_coords = tuple(self.data[dim].values for dim in self.data_dims)
        self.data_incs = tuple(c[1] - c[0] for c in self.data_coords)
        self.data_lims = tuple((c[0], c[-1]) for c in self.data_coords)
        self.data_shape = self.data.shape
        self.data_origin = tuple(self._get_middle_index(c) for c in self.data_coords)
        self._smooth_x_spin.setRange(1, self.data_shape[0] - 1)
        self._smooth_y_spin.setRange(1, self.data_shape[1] - 1)
        self._smooth2_x_spin.setRange(1, self.data_shape[0] - 1)
        self._smooth2_y_spin.setRange(1, self.data_shape[1] - 1)
        self.smooth_data()
        # self.images[0].setImage(self.vals,
        # rect=self._lims_to_rect(1, 0))


def dtool(data, *args, **kwargs):
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)
    qapp.setApplicationName("Dispersive features analysis")
    app = DerivativeTool(data, *args, **kwargs)
    qapp.setStyle("Fusion")
    app.show()
    app.activateWindow()
    app.raise_()
    qapp.exec()


if __name__ == "__main__":
    import erlab.io

    gkmk_cvs = erlab.io.merlin.load(
        "/Users/khan/Documents/ERLab/CsV3Sb5/2021_Dec_ALS_CV3Sb5/211217 ALS BL4/csvtisb1/f_003.pxt"
    )
    alpha_new = np.linspace(gkmk_cvs.alpha[0], gkmk_cvs.alpha[-1], 1000)
    eV_new = np.linspace(gkmk_cvs.eV[0], gkmk_cvs.eV[-1], 2000)
    gkmk_cvs = gkmk_cvs.interp(alpha=alpha_new, eV=eV_new)
    gkmk_cvs = gkmk_cvs.sel(
        alpha=slice(*np.rad2deg((-0.25, 0.25))), eV=slice(-1.25, 0.15)
    )
    dtool(gkmk_cvs, bench=False)
