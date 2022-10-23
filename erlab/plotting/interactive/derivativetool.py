import sys

import numpy as np
import pyqtgraph as pg
import xarray as xr
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

if __name__ != "__main__":
    from .imagetool import move_mean_centered_multiaxis
    from ..interactive.utilities import (AnalysisWidgetBase, ParameterGroup, parse_data,
                              xImageItem, AnalysisWindow)
else:
    from erlab.plotting.interactive.imagetool import move_mean_centered_multiaxis
    from erlab.plotting.interactive.utilities import (AnalysisWidgetBase, ParameterGroup,
                                            parse_data, xImageItem, AnalysisWindow)


class pg_dtool(AnalysisWindow):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        
        # self._main = QtWidgets.QWidget(self)
        # self.setCentralWidget(self._main)
        # self.layout = QtWidgets.QVBoxLayout(self._main)
        # self.layout.setContentsMargins(QtCore.QMargins(0, 0, 0, 0))
        # self.layout.setSpacing(0)
        # self.aw = AnalysisWidgetBase(orientation="vertical")
        # self.layout.addWidget(self.aw)
        # for n in ['set_input', 'set_pre_function', 'set_pre_function_args', 'set_main_function', 'set_main_function_args', 'refresh_output', 'refresh_all']:
        #     setattr(self, n, getattr(self.aw, n))
        # self.set_input(data)
        
        # ParameterGroup(
        #     a=dict(qwtype="dblspin"),
        #     b=dict(qwtype="combobox", items=["item1", "item2", "item3"]),
        # )
        
        


if __name__ == "__main__":
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)
    qapp.setStyle("Fusion")
    dat = xr.open_dataarray(
        "/Users/khan/Documents/ERLab/CsV3Sb5/2021_Dec_ALS_CV3Sb5/Data/cvs_kxy_small.nc"
    ).sel(eV=-0.15, method="nearest")
    dt = pg_dtool(dat)
    dt.show()
    dt.activateWindow()
    dt.raise_()
    qapp.exec()

    