import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

from erlab.interactive.utils import AnalysisWindow, ParameterGroup

__all__ = ["masktool"]


class PolyLineROIControls(ParameterGroup):
    def __init__(self, roi: pg.PolyLineROI, spinbox_kw=None, **kwargs):
        pass


class masktool(AnalysisWindow):
    # sigProgressUpdated = QtCore.Signal(int)

    def __init__(self, data, *args, **kwargs):
        super().__init__(
            data,
            *args,
            link="x",
            layout="horizontal",
            orientation="vertical",
            num_ax=2,
            data_is_input=False,
            **kwargs,
        )
        self._argnames = {}

        self.cursor = self.addParameterGroup(
            **{
                "Z dim": {
                    "qwtype": "combobox",
                    "items": self.data.dims,
                    "currentText": self.data.dims[-1],
                },
                "slider": {
                    "qwtype": "slider",
                    "orientation": QtCore.Qt.Horizontal,
                    "showlabel": False,
                    "value": 0,
                    "minimum": 0,
                    "maximum": self.data.shape[-1] - 1,
                },
                "Transpose": {"qwtype": "chkbox"},
            }
        )
        self.images[0].setDataArray(self.data[:, :, 0])
        self.cursor.sigParameterChanged.connect(self.update_cursor)

        self.roi = pg.PolyLineROI(
            [[0, 0]],
            parent=self.images[0],
            maxBounds=self.axes[0].getViewBox().itemBoundingRect(self.images[0]),
        )
        self.axes[0].addItem(self.roi)
        # self.roi.getArrayRegion(data, self.images[0], axes=(1,2))
        # self.roi.sigRegionChanged.connect()
        self.images[0]

    def update_cursor(self, change):
        # dim_z = change[-1]
        # update_only_values = True

        cursor_params = self.cursor.values

        dim_z = cursor_params["Z dim"]
        if "Z dim" in change:
            self.cursor.blockSignals(True)
            self.cursor.widgets["slider"].setMaximum(len(self.data[dim_z]) - 1)
            self.cursor.widgets["slider"].setValue(0)
            self.cursor.blockSignals(False)
            # update_only_values = False

        new_arr = self.data.isel({dim_z: self.cursor.widgets["slider"].value()})
        # if update_only_values:
        if cursor_params["Transpose"]:
            self.images[0].setDataArray(new_arr.T)
        else:
            self.images[0].setDataArray(new_arr)
        # else:
        # self.images[0].setDataArray(new_arr)
        # self.images[0].setImage(self.data.isel({dim_z:self.cursor.widgets["slider"].value()}).values)

        # self.cursor.values["slider"]
