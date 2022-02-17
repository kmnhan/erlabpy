import sys
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.figure import Figure
from matplotlib.backends.qt_compat import QtWidgets, QtCore
from matplotlib.backends.backend_qtagg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)

from arpes.utilities.conversion import convert_to_kspace
from .bz import plot_hex_bz

__all__ = ['ktool']

class kTool(QtWidgets.QMainWindow):
    def __init__(self, data, bounds, resolution={'kx':0.02, 'ky':0.02}, gamma=0.5, cmap='terrain', *args, **kwargs):
        super().__init__()
        self.data = data
        self.vals = data.values
        self.bounds = bounds
        self.resolution = resolution
        self.kxy = convert_to_kspace(self.data, bounds=self.bounds, resolution=self.resolution)
        self.gamma = gamma
        self.cmap = cmap
        
        improps = dict(
            animated=True, visible=True,
            interpolation='none', aspect='auto', origin='lower',
            norm=colors.PowerNorm(self.gamma), 
            cmap=self.cmap
        )

        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        self.layout = QtWidgets.QVBoxLayout(self._main)

        self.main_canvas = FigureCanvas(Figure(figsize=(8,16),dpi=100))
        self.addToolBar(QtCore.Qt.BottomToolBarArea,
                        NavigationToolbar(self.main_canvas, self))

        gs = self.main_canvas.figure.add_gridspec(1, 2)
        self.ax0 = self.main_canvas.figure.add_subplot(gs[0])
        self.ax1 = self.main_canvas.figure.add_subplot(gs[1])
        self.ax0.grid()
        self.ax1.grid()
        self.ax0.set_aspect('equal')
        self.ax1.set_aspect('equal')
        self.main_canvas.figure.tight_layout()
        plot_hex_bz(ax=self.ax1)

        self.dim_x, self.dim_y = self.data.dims
        self.coord_x, self.coord_y = (
            self.data[self.dim_x].values,
            self.data[self.dim_y].values,
        )
        self.lims_x, self.lims_y = (
            (self.coord_x[0], self.coord_x[-1]),
            (self.coord_y[0], self.coord_y[-1]),
        )
        self.im_r = self.ax0.imshow(self.vals, extent=(*self.lims_x, *self.lims_y), **improps)

        self._update_extent()
        self.im_k = self.ax1.imshow(self.kxy.values, extent=self.extent, **improps)

        self.offsetpanel = QtWidgets.QWidget()
        offsetpanelcontent = QtWidgets.QHBoxLayout()
        
        self.offsetcoords = ['theta','phi','beta','psi','chi']

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
        self.spin0.setSingleStep(0.1)
        self.spin1.setSingleStep(0.1)
        self.spin2.setSingleStep(0.1)
        self.spin3.setSingleStep(0.1)
        self.spin4.setSingleStep(0.1)
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
        gammalabel = QtWidgets.QLabel('g')
        gammaspin = QtWidgets.QDoubleSpinBox()
        gammaspin.setToolTip("Colormap Gamma")
        gammaspin.setSingleStep(0.05)
        gammaspin.setValue(0.5)
        gammaspin.valueChanged.connect(self._set_gamma)
        colormaps = QtWidgets.QComboBox()
        colormaps.setToolTip("Colormap")
        colormaps.addItems(plt.colormaps())
        colormaps.setCurrentIndex(colormaps.findText(self.cmap))
        self.main_canvas.draw()
        colormaps.currentTextChanged.connect(self._set_cmap)
        colorstabcontent.addWidget(gammalabel)
        colorstabcontent.addWidget(gammaspin)
        colorstabcontent.addWidget(colormaps)
        colorstabcontent.addStretch()
        self.colorstab.setLayout(colorstabcontent)

        self.tabwidget = QtWidgets.QTabWidget()
        self.tabwidget.addTab(self.offsetpanel, "Offsets")
        self.tabwidget.addTab(self.colorstab, "Colors")
        # self.tabwidget.addTab(self.pathtab, "Path")
        self.layout.addWidget(self.tabwidget)
        
        self.layout.addWidget(self.main_canvas)

    def _set_gamma(self, gamma):
        self.gamma = gamma
        self.im_r.set_norm(colors.PowerNorm(self.gamma))
        self.im_k.set_norm(colors.PowerNorm(self.gamma))
        self.main_canvas.draw()

    def _set_cmap(self, cmap):
        self.cmap = cmap
        self.im_k.set_cmap(self.cmap)
        self.main_canvas.draw()

    def _spinchanged(self, n, value):
        self.data.S.apply_offsets({self.offsetcoords[n]:value*np.pi/180})
        self._update_kxy()
        self.main_canvas.draw()

    def _update_kxy(self):
        self.kxy = convert_to_kspace(self.data, bounds=self.bounds, resolution=self.resolution)
        self.im_k.set_data(self.kxy.values)
        self._update_extent()
        self.im_k.set_extent(self.extent)

    def _update_extent(self):
        self.dim_kx, self.dim_ky = self.kxy.dims
        self.coord_kx, self.coord_ky = (
            self.kxy[self.dim_kx].values,
            self.kxy[self.dim_ky].values,
        )
        self.lims_kx, self.lims_ky = (
            (self.coord_kx[0], self.coord_kx[-1]),
            (self.coord_ky[0], self.coord_ky[-1]),
        )
        self.extent = (*self.lims_kx, *self.lims_ky)
    

def ktool(data, *args, **kwargs):
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)
    with plt.rc_context({
        'text.usetex':False,
    #     #  'mathtext.fontset':'stixsans',
        'font.size':7,
        # 'font.family':'sans'
    }):
        app = kTool(data, *args, **kwargs)
    # qapp.setStyle('Fusion')
    app.show()
    app.activateWindow()
    app.raise_()
    qapp.exec()