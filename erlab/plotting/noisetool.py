import sys
import weakref
from itertools import compress

import numpy as np
import numba
import bottleneck as bn
# import numbagg
import xarray as xr
import darkdetect
import qtawesome as qta
from time import perf_counter
from matplotlib import colors
from scipy.ndimage import uniform_filter, gaussian_filter
import pyqtgraph as pg
pg.setConfigOption('imageAxisOrder', 'row-major')

from arpes.analysis.derivative import (
    d1_along_axis, curvature, minimum_gradient
)

from erlab.plotting.imagetool import (
    change_style, parse_data, move_mean_centered_multiaxis
)
# pg.setConfigOption('useNumba', True)
# pg.setConfigOption('background', 'w')
# pg.setConfigOption('foreground', 'k')

# from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

__all__ = ['noisetool']

def boxcarfilt2(array, window_list, min_count_list=None):
    return move_mean_centered_multiaxis(array, window_list, min_count_list)

from PySide6 import QtCore, QtGui, QtWidgets
QtWidgets.QGraphicsScene
class NoiseTool(QtWidgets.QMainWindow):
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
            pg.ImageItem(),
            pg.ImageItem(),
        ]
        for i, img in enumerate(self.images):
            self.axes[i].addItem(img)
            self.hists[i].setImageItem(img)
            # img.sigImageChanged.connect()
        
        self.use_gaussfilt = False

        self._smooth_group = QtWidgets.QGroupBox('Smoothing')
        smooth_layout = QtWidgets.QGridLayout(self._smooth_group)
        smooth_mode_check = QtWidgets.QCheckBox(self._smooth_group)
        smooth_mode_check.setChecked(self.use_gaussfilt)
        smooth_mode_check.stateChanged.connect(self.set_gauss_smooth)
        smooth_mode_label = QtWidgets.QLabel('Gaussian filter')
        smooth_mode_label.setBuddy(smooth_mode_check)
        self._smooth_x_spin = QtWidgets.QSpinBox(self._smooth_group)
        self._smooth_x_spin.valueChanged.connect(
            lambda val, ax=1: self.smooth_data(ax, val))
        self._smooth_y_spin = QtWidgets.QSpinBox(self._smooth_group)
        self._smooth_y_spin.valueChanged.connect(
            lambda val, ax=0: self.smooth_data(ax, val))
        self._smooth_n_spin = QtWidgets.QSpinBox(self._smooth_group)
        self._smooth_x_spin_label = QtWidgets.QLabel('x')
        self._smooth_y_spin_label = QtWidgets.QLabel('y')
        self._smooth_n_spin_label = QtWidgets.QLabel('n')
        self._smooth_n_spin.setRange(1, 100)
        self._smooth_n_spin.valueChanged.connect(self.set_smooth_num)
        smooth_layout.addWidget(smooth_mode_label, 0, 0)
        smooth_layout.addWidget(smooth_mode_check, 0, 1)
        smooth_layout.addWidget(self._smooth_x_spin_label, 1, 0)
        smooth_layout.addWidget(self._smooth_x_spin, 1, 1)
        smooth_layout.addWidget(self._smooth_y_spin_label, 2, 0)
        smooth_layout.addWidget(self._smooth_y_spin, 2, 1)
        smooth_layout.addWidget(self._smooth_n_spin_label, 3, 0)
        smooth_layout.addWidget(self._smooth_n_spin, 3, 1)


        self._smooth2_group = QtWidgets.QGroupBox('2nd Smoothing')
        smooth2_layout = QtWidgets.QGridLayout(self._smooth2_group)
        self._smooth2_x_spin = QtWidgets.QSpinBox(self._smooth2_group)
        self._smooth2_x_spin.valueChanged.connect(
            lambda: self.deriv_data())
        self._smooth2_y_spin = QtWidgets.QSpinBox(self._smooth2_group)
        self._smooth2_y_spin.valueChanged.connect(
            lambda: self.deriv_data())
        self._smooth2_n_spin = QtWidgets.QSpinBox(self._smooth2_group)
        self._smooth2_x_spin_label = QtWidgets.QLabel('x')
        self._smooth2_y_spin_label = QtWidgets.QLabel('y')
        self._smooth2_n_spin_label = QtWidgets.QLabel('n')
        self._smooth2_n_spin.setRange(1, 100)
        self._smooth2_n_spin.valueChanged.connect(self.set_smooth2_num)
        smooth2_layout.addWidget(self._smooth2_x_spin_label, 1, 0)
        smooth2_layout.addWidget(self._smooth2_x_spin, 1, 1)
        smooth2_layout.addWidget(self._smooth2_y_spin_label, 2, 0)
        smooth2_layout.addWidget(self._smooth2_y_spin, 2, 1)
        smooth2_layout.addWidget(self._smooth2_n_spin_label, 3, 0)
        smooth2_layout.addWidget(self._smooth2_n_spin, 3, 1)

        self._deriv_group = QtWidgets.QGroupBox('Derivative')
        deriv_layout = QtWidgets.QGridLayout(self._deriv_group)
        self._deriv_axis_combo = QtWidgets.QComboBox(self._deriv_group)
        self._deriv_axis_combo.currentTextChanged.connect(self.deriv_data)
        self._deriv_axis_combo.addItems(['x', 'y'])
        deriv_layout.addWidget(self._deriv_axis_combo, 0, 0)

        self._curv_group = QtWidgets.QGroupBox('Curvature')
        curv_layout = QtWidgets.QGridLayout(self._curv_group)
        self._curv_alpha_spin = QtWidgets.QDoubleSpinBox(self._curv_group)
        self._curv_alpha_spin.setRange(-30, 30)
        self._curv_alpha_spin.setValue(0)
        self._curv_alpha_spin.setSingleStep(0.01)
        self._curv_alpha_spin.valueChanged.connect(
            lambda beta: self.curv_data(beta=beta))
        curv_layout.addWidget(self._curv_alpha_spin, 0, 0)

        self._mingrad_group = QtWidgets.QGroupBox('Minimum Gradient')
        mingrad_layout = QtWidgets.QGridLayout(self._mingrad_group)
        self._mingrad_delta_spin = QtWidgets.QSpinBox(self._mingrad_group)
        self._mingrad_delta_spin.setRange(1,100)
        self._mingrad_delta_spin.setValue(1)
        self._mingrad_delta_spin.valueChanged.connect(
            lambda delta: self.mingrad_data(delta=delta))
        mingrad_layout.addWidget(self._mingrad_delta_spin, 0, 0)

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
        self.tabwidget.addTab(self._deriv_tab, 'Derivative')
        self.tabwidget.addTab(self._curv_tab, 'Curvature')
        self.tabwidget.addTab(self._mingrad_tab, 'Minimum Gradient')
        self.tabwidget.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.Maximum,
        )
        
        self.layout.addWidget(self._smooth_group, 0, 0, 1, 1)
        self.layout.addWidget(self.tabwidget, 0, 1, 1, 1)
        self.layout.addWidget(plotwidget, 1, 0, 1, 2)

        self.win_list = [1, 1]
        self.smooth_num = 1
        self.smooth2_num = 1

        if data is not None:
            self.set_data(data)
        

    def get_cutoff(self, data, cutoff=None):
        if cutoff is None:
            cutoff = [3, 3]
        else:
            try:
                cutoff = list(cutoff.__iter__)
            except AttributeError:
                cutoff = [cutoff] * 2
        pl, q3, q1, pu = np.percentile(data,
                                       [100-cutoff[0], 65, 20, cutoff[1]])
        ql = q1 - 1.5 * (q3 - q1)
        qu = q3 + 1.5 * (q3 - q1)
        mn = min(pl, ql)
        mx = max(pu, qu)
        return [mn, mx]

    def set_gauss_smooth(self, value):
        self.use_gaussfilt = value
        self.smooth_data()


    def _result_update_lims(self):
        self.images[1].setImage(self.data_s_d.values,
                                rect=self._lims_to_rect(1, 0),
                                levels=self.get_cutoff(self.data_s_d.values))

    def set_smooth_num(self, n):
        self.smooth_num = n
        self.smooth_data()
    
    def set_smooth2_num(self, n):
        self.smooth2_num = n
        self.deriv_data()
    
    def mingrad_data(self, delta=1):
        self.use_curv = False
        self.use_mingrad = True
        self.use_deriv = False
        # self.data_s_d = self.data_s.copy(deep=True)
        self.data_s_d = minimum_gradient(self.data_s, delta=delta)
        self._result_update_lims()

    def curv_data(self, alpha=1, beta=None):
        self.use_curv = True
        self.use_deriv = False
        self.use_mingrad = False
        if beta is None:
            beta = self._curv_alpha_spin.value()
        # self.data_s_d = self.data_s.copy(deep=True)
        self.data_s_d = curvature(self.data_s/self.data_s.max(), alpha=alpha, beta=beta)
        # m = self.data_s_d.max()
        # self.data_s_d /= m
        self._result_update_lims()

    def deriv_data(self, axis=None):
        self.use_curv = False
        self.use_mingrad = False
        self.use_deriv = True
        if axis is None:
            axis = self._deriv_axis_combo.currentText()
        if axis == 'x':
            axis = 0
        elif axis == 'y':
            axis = 1
        axis = self.data_dims[axis]
        self.data_s_d = d1_along_axis(self.data_s, axis=axis)
        amount = [self._smooth2_y_spin.value(),
                  self._smooth2_x_spin.value()]
        self.data_s_d.values = self.smoothfunc(self.data_s_d.values,
                                               amount,
                                               self.smooth2_num)
        self.data_s_d = d1_along_axis(self.data_s_d, axis=axis)
        self._result_update_lims()

    
    def _get_middle_index(self, x):
        return len(x)//2 - (1 if len(x) % 2 == 0 else 0)
    def smoothfunc(self, data, amount, num):
        
        if not self.use_gaussfilt:
            for _ in range(num):
                data = uniform_filter(data, amount, mode='constant', origin=(0, 0))
        else:
            sigma = [(amount[0]-1)/3, (amount[1]-1)/3]
            for _ in range(num):
                data = gaussian_filter(data, sigma=sigma)
        return data
        

    def smooth_data(self, axis=None, amount=None):
        if axis is not None:
            self.win_list[axis] = amount
        self.data_s = self.data.copy(deep=True)
        vals = self.data_s.values

        # if not self.use_gaussfilt:
            # for i in range(self.smooth_num):
                # vals = uniform_filter(vals, size=self.win_list, mode='mirror')
        # else:
            # self.sigma_list = [(self.win_list[0]-1)/3,
                            #    (self.win_list[1]-1)/3]
            # for i in range(self.smooth_num):
                # vals = gaussian_filter(vals, sigma=self.sigma_list, mode='mirror')
        # self.data_s.values = vals
        self.data_s.values = self.smoothfunc(vals,
                                             self.win_list,
                                             self.smooth_num)

        self.images[0].setImage(self.data_s.values,
                                rect=self._lims_to_rect(1, 0))
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
        self.data = parse_data(data)
        self.vals = self.data.values
        self.data_dims = self.data.dims
        self.data_coords = tuple(self.data[dim].values
                                 for dim in self.data_dims)
        self.data_incs = tuple(c[1] - c[0] for c in self.data_coords)
        self.data_lims = tuple((c[0], c[-1]) for c in self.data_coords)
        self.data_shape = self.data.shape
        self.data_origin = tuple(self._get_middle_index(c)
                                 for c in self.data_coords)
        self._smooth_x_spin.setRange(1, self.data_shape[0] - 1)
        self._smooth_y_spin.setRange(1, self.data_shape[1] - 1)
        self._smooth2_x_spin.setRange(1, self.data_shape[0] - 1)
        self._smooth2_y_spin.setRange(1, self.data_shape[1] - 1)
        self.images[0].setImage(self.vals,
                                rect=self._lims_to_rect(1, 0))

def noisetool(data, *args, **kwargs):
    # TODO: implement multiple windows, add transpose, equal aspect settings
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)
    qapp.setApplicationName('Dispersive features analysis')
    # if darkdetect.isDark():
        # pass
    app = NoiseTool(data, *args, **kwargs)
    change_style('Fusion')
    app.show()
    app.activateWindow()
    app.raise_()
    qapp.exec()

if __name__ == "__main__":
    from arpes.io import load_data
    gkmk_cvs = load_data('/Users/khan/Documents/ERLab/CsV3Sb5/2021_Dec_ALS_CV3Sb5/211217 ALS BL4/csvtisb1/f_003.pxt',location="BL4").spectrum
    phi_new = np.linspace(gkmk_cvs.phi[0], gkmk_cvs.phi[-1], 1000)
    eV_new = np.linspace(gkmk_cvs.eV[0], gkmk_cvs.eV[-1], 2000)
    gkmk_cvs = gkmk_cvs.interp(phi=phi_new, eV=eV_new)
    gkmk_cvs = gkmk_cvs.sel(phi=slice(-0.25,0.25),eV=slice(-1.25,0.15))
    noisetool(gkmk_cvs, bench=False)

    dose_kmk_2 = load_data('/Users/khan/Documents/ERLab/CsV3Sb5/2021_Dec_ALS_CV3Sb5/Data/f_003_R1_S001.pxt',location="BL4").spectrum
    gkmk_cs1 = dose_kmk_2[0,:,:]
    phi_new = np.linspace(gkmk_cs1.phi[0], gkmk_cs1.phi[-1], 1000)
    eV_new = np.linspace(gkmk_cs1.eV[0], gkmk_cs1.eV[-1], 2000)
    gkmk_cs1 = gkmk_cs1.interp(phi=phi_new, eV=eV_new)
    gkmk_cs1 = gkmk_cs1.sel(phi=slice(-0.25,0.25),eV=slice(-1.25,0.15))
    # noisetool(gkmk_cs1, bench=False)
    # dat = xr.open_dataarray('/Users/khan/Documents/ERLab/CsV3Sb5/2021_Dec_ALS_CV3Sb5/Data/cvs_kxy_small.nc')
    # noisetool(dat.sel(eV=0,method='nearest'), bench=False)
    
    # noisetool(gkmk_cs1, bench=False)