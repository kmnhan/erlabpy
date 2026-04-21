__all__ = ["goldtool", "restool"]

import contextlib
import dataclasses
import enum
import importlib.resources
import logging
import os
import threading
import time
import traceback
import typing
from collections.abc import Callable

import numpy as np
import pydantic
import pyqtgraph as pg
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

import erlab

if typing.TYPE_CHECKING:
    import joblib
    import lmfit
    import scipy.interpolate
    import varname
else:
    import lazy_loader as _lazy

    joblib = _lazy.load("joblib")
    varname = _lazy.load("varname")


LMFIT_METHODS = [
    "leastsq",
    "least_squares",
    "differential_evolution",
    # "brute",
    # "basinhopping",
    # "ampgo",
    # "nelder",
    # "lbfgsb",
    # "powell",
    "cg",
    # "newton",
    # "cobyla",
    # "bfgs",
    # "tnc",
    # "trust-ncg",
    # "trust-exact",
    # "trust-krylov",
    # "trust-constr",
    # "dogleg",
    # "slsqp",
    "emcee",
    # "shgo",
    # "dual_annealing",
]

logger = logging.getLogger(__name__)


class EdgeFitSignals(QtCore.QObject):
    sigIterated = QtCore.Signal(int)
    sigFinished = QtCore.Signal(object, object)
    sigFailed = QtCore.Signal(str)


class EdgeFitTask(QtCore.QRunnable):
    def __init__(
        self,
        data: xr.DataArray,
        along: str,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        params,
    ) -> None:
        super().__init__()
        self.data = data.copy()
        self.along = along
        self.x_range: tuple[float, float] = (x0, x1)
        self.y_range: tuple[float, float] = (y0, y1)
        self.params = params
        self.parallel_obj: joblib.Parallel | None = None
        self.signals = EdgeFitSignals()
        self._mutex = QtCore.QMutex()

    @QtCore.Slot()
    def abort_fit(self) -> None:
        if self.parallel_obj is None:
            return
        self._mutex.lock()
        self.parallel_obj._aborting = True
        self.parallel_obj._exception = True
        self._mutex.unlock()

    def run(self) -> None:
        try:
            # https://github.com/joblib/joblib/issues/1002
            backend = (
                "threading"
                if erlab.utils.misc._IS_PACKAGED
                else joblib.parallel.DEFAULT_BACKEND
            )
            self.parallel_obj = joblib.Parallel(
                n_jobs=self.params["# CPU"],
                max_nbytes=None,
                return_as="generator",
                pre_dispatch="n_jobs",
                backend=backend,
            )
            self.signals.sigIterated.emit(0)
            with erlab.utils.parallel.joblib_progress_qt(self.signals.sigIterated) as _:
                edge_center, edge_stderr = typing.cast(
                    "tuple[xr.DataArray, xr.DataArray]",
                    erlab.analysis.gold.edge(
                        gold=self.data,
                        along=self.along,
                        angle_range=self.x_range,
                        eV_range=self.y_range,
                        bin_size=(self.params["Bin x"], self.params["Bin y"]),
                        temp=self.params["T (K)"],
                        vary_temp=not self.params["Fix T"],
                        bkg_slope=self.params["Linear"],
                        resolution=self.params["Resolution"],
                        fast=self.params["Fast"],
                        method=self.params["Method"],
                        scale_covar=self.params["Scale cov"],
                        progress=False,
                        parallel_obj=self.parallel_obj,
                        drop_nans=True,
                    ),
                )
            self.signals.sigFinished.emit(edge_center, edge_stderr)
        except Exception:  # pragma: no cover - defensive for worker thread
            self.signals.sigFailed.emit(traceback.format_exc())


@dataclasses.dataclass(slots=True)
class _GoldUpdateRequest:
    data: xr.DataArray
    had_fit: bool
    roi_limits: tuple[float, float, float, float]
    tab_index: int
    edge_values: dict[str, typing.Any]
    poly_values: dict[str, typing.Any]
    spline_values: dict[str, typing.Any]
    refit: bool


class _GoldFitSnapshot(pydantic.BaseModel):
    along_coords: list[float]
    edge_center: list[float]
    edge_stderr: list[float]


class GoldTool(erlab.interactive.utils.AnalysisWindow):
    """Interactive gold edge fitting.

    Parameters
    ----------
    data
        The data to perform Fermi edge fitting on.
    data_corr
        The data to correct with the edge. Defaults to `data`.
    data_name
        Name of the data used in generating the code snipped copied to the clipboard.
        Overrides automatic detection.
    execute
        Whether to execute the tool immediately.
    **kwargs
        Arguments passed onto `erlab.interactive.utils.AnalysisWindow`.

    Signals
    -------
    sigProgressUpdated(int)
        Signal used to update the progress bar.
    sigAbortFitting()
        Signal used to abort the fitting, emitted when the cancel button is clicked.
    sigUpdated()
        Signal emitted when all fitting steps are finished and plots are updated.
    """

    tool_name = "goldtool"
    COPY_PROVENANCE: typing.ClassVar = (
        erlab.interactive.utils.ToolScriptProvenanceDefinition(
            start_label="Start from current goldtool input data",
            label_method="_current_mode_copy_label",
            expression_method="_current_mode_copy_expression",
            assign="modelresult",
        )
    )

    class Output(enum.StrEnum):
        CORRECTED = "goldtool.corrected"

    IMAGE_TOOL_OUTPUTS: typing.ClassVar = {
        Output.CORRECTED: erlab.interactive.utils.ToolImageOutputDefinition(
            data_method="_corrected_output",
            provenance=erlab.interactive.utils.ToolScriptProvenanceDefinition(
                start_label="Start from current goldtool input data",
                label_method="_current_mode_corrected_label",
                prelude_method="_current_mode_corrected_prelude",
                expression_method="_current_mode_corrected_expression",
                assign="corrected",
            ),
        )
    }

    class StateModel(pydantic.BaseModel):
        data_name: str
        roi_limits: tuple[float, float, float, float]
        edge_values: dict[str, typing.Any]
        poly_values: dict[str, typing.Any]
        spline_values: dict[str, typing.Any]
        tab_index: typing.Literal[0, 1]
        refit_on_source_update: bool = False
        fit_snapshot: _GoldFitSnapshot | None = None

    sigProgressUpdated = QtCore.Signal(int)  #: :meta private:
    sigAbortFitting = QtCore.Signal()  #: :meta private:
    sigUpdated = QtCore.Signal()  #: :meta private:

    def __init__(
        self,
        data: xr.DataArray,
        data_corr: xr.DataArray | None = None,
        *,
        data_name: str | None = None,
        **kwargs,
    ) -> None:
        if data.ndim != 2 or "eV" not in data.dims:
            raise ValueError("`data` must be a 2D DataArray with an `eV` dimension")
        if data.dims[0] != "eV":
            data = data.copy().T

        self._along_dim: str = str(data.dims[1])

        super().__init__(
            data,
            link="x",
            layout="horizontal",
            orientation="vertical",
            num_ax=3,
            **kwargs,
        )

        self._argnames = {}
        if data_name is None:
            try:
                self._argnames["data"] = varname.argname(
                    "data",
                    func=self.__init__,  # type: ignore[misc]
                    vars_only=False,
                )
            except varname.VarnameRetrievingError:
                self._argnames["data"] = "gold"
        else:
            self._argnames["data"] = data_name

        if data_corr is not None:
            try:
                self._argnames["data_corr"] = varname.argname(
                    "data_corr",
                    func=self.__init__,  # type: ignore[misc]
                    vars_only=False,
                )
            except varname.VarnameRetrievingError:
                self._argnames["data_corr"] = "data_corr"

        self.data_corr = data_corr
        self.hists: pg.HistogramLUTItem
        self.axes: list[pg.PlotItem]
        self.images: list[erlab.interactive.utils.xImageItem]

        self.axes[1].setVisible(False)
        self.hists[1].setVisible(False)
        self.axes[2].setVisible(False)
        self.hists[2].setVisible(False)

        temp = self.data.qinfo.get_value("sample_temp")
        if temp is None:
            temp = 30.0
        temp = float(temp)

        x_decimals = erlab.utils.array.effective_decimals(
            self.data[self._along_dim].values
        )
        y_decimals = erlab.utils.array.effective_decimals(self.data.eV.values)

        self.params_roi = erlab.interactive.utils.ROIControls(
            self.aw.add_roi(0),
            x_decimals=x_decimals,
            y_decimals=y_decimals,
        )
        self.params_edge = erlab.interactive.utils.ParameterGroup(
            {
                "T (K)": {"qwtype": "dblspin", "value": temp, "range": (0.0, 400.0)},
                "Fix T": {"qwtype": "chkbox", "checked": True},
                "Bin x": {"qwtype": "spin", "value": 1, "minimum": 1},
                "Bin y": {"qwtype": "spin", "value": 1, "minimum": 1},
                "Resolution": {
                    "qwtype": "dblspin",
                    "value": 0.02,
                    "range": (0.0, 99.9),
                    "singleStep": 0.001,
                    "decimals": 5,
                },
                "Fast": {
                    "qwtype": "chkbox",
                    "checked": False,
                    "toolTip": "If checked, fit with a broadened step function "
                    "instead of Fermi-Dirac",
                },
                "Linear": {
                    "qwtype": "chkbox",
                    "checked": True,
                    "toolTip": "If unchecked, fixes the slope of the background "
                    "above the Fermi level to zero.",
                },
                "Method": {"qwtype": "combobox", "items": LMFIT_METHODS},
                "Scale cov": {"qwtype": "chkbox", "checked": True},
                "# CPU": {
                    "qwtype": "spin",
                    "value": os.cpu_count(),
                    "minimum": 1,
                    "maximum": os.cpu_count(),
                },
                "go": {
                    "qwtype": "pushbtn",
                    "showlabel": False,
                    "text": "Go",
                    "clicked": self.perform_edge_fit,
                },
            }
        )

        typing.cast(
            "QtWidgets.QComboBox", self.params_edge.widgets["Method"]
        ).setCurrentIndex(1)

        typing.cast(
            "QtWidgets.QCheckBox", self.params_edge.widgets["Fast"]
        ).stateChanged.connect(self._toggle_fast)

        self.params_poly = erlab.interactive.utils.ParameterGroup(
            {
                "Degree": {"qwtype": "spin", "value": 4, "range": (0, 20)},
                "Scale cov": {"qwtype": "chkbox", "checked": True},
                "Residuals": {"qwtype": "chkbox", "checked": False},
                "Corrected": {"qwtype": "chkbox", "checked": False},
                "Shift coords": {"qwtype": "chkbox", "checked": True},
                "itool": {
                    "qwtype": "pushbtn",
                    "notrack": True,
                    "showlabel": False,
                    "text": "Open corrected in ImageTool",
                    "clicked": self.open_itool,
                },
                "copy": {
                    "qwtype": "pushbtn",
                    "notrack": True,
                    "showlabel": False,
                    "text": "Copy code to clipboard",
                    "clicked": self.copy_code,
                },
                "save": {
                    "qwtype": "pushbtn",
                    "notrack": True,
                    "showlabel": False,
                    "text": "Save polynomial fit to file",
                    "clicked": self._save_poly_fit,
                },
            }
        )

        self.params_spl = erlab.interactive.utils.ParameterGroup(
            {
                "Auto": {"qwtype": "chkbox", "checked": True},
                "lambda": {
                    "qwtype": "dblspin",
                    "minimum": 0,
                    "maximum": 10000,
                    "singleStep": 0.001,
                    "decimals": 4,
                },
                "Residuals": {"qwtype": "chkbox", "checked": False},
                "Corrected": {"qwtype": "chkbox", "checked": False},
                "Shift coords": {"qwtype": "chkbox", "checked": True},
                "itool": {
                    "qwtype": "pushbtn",
                    "notrack": True,
                    "showlabel": False,
                    "text": "Open corrected in ImageTool",
                    "clicked": self.open_itool,
                },
                "copy": {
                    "qwtype": "pushbtn",
                    "notrack": True,
                    "showlabel": False,
                    "text": "Copy code to clipboard",
                    "clicked": self.copy_code,
                },
            }
        )
        auto_check = typing.cast("QtWidgets.QCheckBox", self.params_spl.widgets["Auto"])
        self._sync_spline_lambda_enabled()
        auto_check.toggled.connect(lambda _: self._sync_spline_lambda_enabled())

        self.controls.addWidget(self.params_roi)
        self.controls.addWidget(self.params_edge)
        self.refit_on_source_update_check = QtWidgets.QCheckBox(
            "Refit on source update"
        )
        self.refit_on_source_update_check.setToolTip(
            "If checked, rerun the edge fit when this tool refreshes from ImageTool."
        )
        self.controls.addWidget(self.refit_on_source_update_check)

        self.params_tab = QtWidgets.QTabWidget()
        self.params_tab.addTab(self.params_poly, "Polynomial")
        self.params_tab.addTab(self.params_spl, "Spline")
        self.params_tab.currentChanged.connect(self.perform_fit)
        self.controls.addWidget(self.params_tab)

        self.params_poly.setDisabled(True)
        self.params_spl.setDisabled(True)
        self.params_tab.setDisabled(True)

        self.controlgroup.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Preferred
        )

        self.scatterplots = [
            pg.ScatterPlotItem(
                size=3,
                pen=pg.mkPen(None),
                brush=pg.mkBrush(255, 0, 0, 200),
                pxMode=True,
            )
            for _ in range(2)
        ]

        self.errorbars = [pg.ErrorBarItem(pen=pg.mkPen(color="red")) for _ in range(2)]
        self.polycurves = [pg.PlotDataItem() for _ in range(2)]
        for i in range(2):
            self.axes[i].addItem(self.scatterplots[i])
            self.axes[i].addItem(self.errorbars[i])
            self.axes[i].addItem(self.polycurves[i])
        self.params_poly.sigParameterChanged.connect(self.perform_fit)
        self.params_spl.sigParameterChanged.connect(self.perform_fit)

        self.axes[0].disableAutoRange()

        # Setup time calculation
        self.start_time: float
        self.step_times: list[float]

        # Setup progress bar
        self.progress: QtWidgets.QProgressDialog = QtWidgets.QProgressDialog(self)
        self.pbar: QtWidgets.QProgressBar = QtWidgets.QProgressBar()

        self.progress.setLabelText("Fitting...")
        self.progress.setCancelButtonText("Abort!")
        self.progress.setRange(0, 100)
        self.progress.setMinimumDuration(0)
        self.progress.setBar(self.pbar)
        self.progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.progress.setFixedSize(self.progress.size())
        self.progress.canceled.disconnect(self.progress.cancel)  # don't auto close
        self.progress.canceled.connect(self.abort_fit)
        self.progress.setAutoReset(False)
        self.progress.cancel()

        # Setup fitter worker
        # This allows the GUI to remain responsive during fitting so it can be aborted
        self._threadpool = QtCore.QThreadPool(self)
        self._fit_task: EdgeFitTask | None = None
        self.sigAbortFitting.connect(self._abort_fit_task)
        self._pending_update_request: _GoldUpdateRequest | None = None
        self._pending_update_timer = QtCore.QTimer(self)
        self._pending_update_timer.setSingleShot(True)
        self._pending_update_timer.timeout.connect(self._flush_pending_update)

        # Resize roi to data bounds
        eV_span = self.data.eV.values[-1] - self.data.eV.values[0]
        ang_span = (
            self.data[self._along_dim].values[-1] - self.data[self._along_dim].values[0]
        )
        x1 = self.data[self._along_dim].values.mean() + ang_span * 0.45
        x0 = self.data[self._along_dim].values.mean() - ang_span * 0.45
        y1 = self.data.eV.values[-1] - eV_span * 0.015
        y0 = y1 - eV_span * 0.3
        self.params_roi.modify_roi(x0, y0, x1, y1)

        # Initialize imagetool variable
        self._itool: QtWidgets.QWidget | None = None

        # Initialize fit result
        self.result: scipy.interpolate.BSpline | xr.Dataset | None = None

    @property
    def data_name(self) -> str:
        return typing.cast("str", self._argnames["data"])

    @data_name.setter
    def data_name(self, value: str) -> None:
        self._argnames["data"] = value

    @property
    def tool_status(self) -> StateModel:
        fit_snapshot: _GoldFitSnapshot | None = None
        if hasattr(self, "edge_center") and hasattr(self, "edge_stderr"):
            fit_snapshot = _GoldFitSnapshot(
                along_coords=np.asarray(
                    self.edge_center[self._along_dim].values, dtype=float
                ).tolist(),
                edge_center=np.asarray(self.edge_center.values, dtype=float).tolist(),
                edge_stderr=np.asarray(self.edge_stderr.values, dtype=float).tolist(),
            )
        return self.StateModel(
            data_name=self.data_name,
            roi_limits=self.params_roi.roi_limits,
            edge_values=dict(self.params_edge.values),
            poly_values=dict(self.params_poly.values),
            spline_values=dict(self.params_spl.values),
            tab_index=typing.cast(
                "typing.Literal[0, 1]", self.params_tab.currentIndex()
            ),
            refit_on_source_update=self.refit_on_source_update_check.isChecked(),
            fit_snapshot=fit_snapshot,
        )

    @tool_status.setter
    def tool_status(self, status: StateModel) -> None:
        self.data_name = status.data_name
        self._clear_edge_fit_results()

        with (
            QtCore.QSignalBlocker(self.params_edge),
            QtCore.QSignalBlocker(self.params_poly),
            QtCore.QSignalBlocker(self.params_spl),
            QtCore.QSignalBlocker(self.params_tab),
            QtCore.QSignalBlocker(self.refit_on_source_update_check),
        ):
            self._restore_parameter_group_values(self.params_edge, status.edge_values)
            self._restore_parameter_group_values(self.params_poly, status.poly_values)
            self._restore_parameter_group_values(self.params_spl, status.spline_values)
            self.params_tab.setCurrentIndex(status.tab_index)
            self.refit_on_source_update_check.setChecked(status.refit_on_source_update)

        self.params_roi.modify_roi(*self._clamp_roi_limits_to_bounds(status.roi_limits))
        self.params_roi.update_pos()
        self._toggle_fast()
        self._sync_spline_lambda_enabled()

        if status.fit_snapshot is not None:
            self._restore_fit_snapshot(status.fit_snapshot)

    @property
    def tool_data(self) -> xr.DataArray:
        return self.data

    def _clear_edge_fit_results(self) -> None:
        self.result = None
        self._fit_task = None
        self.progress.reset()
        self.params_poly.setDisabled(True)
        self.params_spl.setDisabled(True)
        self.params_tab.setDisabled(True)
        self.aw.axes[1].setVisible(False)
        self.aw.hists[1].setVisible(False)
        self.aw.axes[2].setVisible(False)
        self.aw.hists[2].setVisible(False)
        for scatter, errorbar, curve in zip(
            self.scatterplots, self.errorbars, self.polycurves, strict=True
        ):
            scatter.setData(x=np.array([]), y=np.array([]))
            errorbar.setData(
                x=np.array([]), y=np.array([]), height=np.array([], dtype=float)
            )
            curve.setData(x=np.array([]), y=np.array([]))
        with contextlib.suppress(AttributeError):
            del self.edge_center
        with contextlib.suppress(AttributeError):
            del self.edge_stderr

    def _sync_spline_lambda_enabled(self) -> None:
        auto_check = typing.cast("QtWidgets.QCheckBox", self.params_spl.widgets["Auto"])
        self.params_spl.widgets["lambda"].setDisabled(
            auto_check.checkState() == QtCore.Qt.CheckState.Checked
        )

    @staticmethod
    def _restore_parameter_group_values(
        group: erlab.interactive.utils.ParameterGroup,
        values: dict[str, typing.Any],
    ) -> None:
        for name, value in values.items():
            widget = group.widgets[name]
            widget.blockSignals(True)
            try:
                if hasattr(widget, "setValue"):
                    widget.setValue(value)
                elif isinstance(widget, QtWidgets.QAbstractButton):
                    if widget.isCheckable():
                        widget.setChecked(bool(value))
                elif isinstance(widget, QtWidgets.QComboBox):
                    index = widget.findText(str(value))
                    if index < 0:
                        raise ValueError(
                            f"Unknown saved value {value!r} for {name!r} in goldtool"
                        )
                    widget.setCurrentIndex(index)
            finally:
                widget.blockSignals(False)

    def _restore_fit_snapshot(self, snapshot: _GoldFitSnapshot) -> None:
        if not (
            len(snapshot.along_coords)
            == len(snapshot.edge_center)
            == len(snapshot.edge_stderr)
        ):
            raise ValueError("Saved goldtool fit snapshot has mismatched array lengths")

        coords = {self._along_dim: np.asarray(snapshot.along_coords, dtype=float)}
        self.post_fit(
            xr.DataArray(
                np.asarray(snapshot.edge_center, dtype=float),
                coords=coords,
                dims=(self._along_dim,),
            ),
            xr.DataArray(
                np.asarray(snapshot.edge_stderr, dtype=float),
                coords=coords,
                dims=(self._along_dim,),
            ),
        )

    def _ensure_serializable_state(self) -> None:
        if self.data_corr is not None:
            raise ValueError(
                "goldtool save/load/duplication is unsupported when `data_corr` "
                "is provided separately"
            )

    @staticmethod
    def _clamp_roi_interval(
        start: float, stop: float, lower: float, upper: float
    ) -> tuple[float, float]:
        lower, upper = min(lower, upper), max(lower, upper)
        available = upper - lower
        lo, hi = min(start, stop), max(start, stop)
        width = hi - lo

        if available <= 0:
            return lower, upper
        if width <= 0 or width >= available:
            return lower, upper

        lo = min(max(lo, lower), upper - width)
        return lo, lo + width

    def _clamp_roi_limits_to_bounds(
        self, roi_limits: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        x0, x1 = self._clamp_roi_interval(
            roi_limits[0], roi_limits[2], *self.params_roi.max_bounds[::2]
        )
        y0, y1 = self._clamp_roi_interval(
            roi_limits[1], roi_limits[3], *self.params_roi.max_bounds[1::2]
        )
        return x0, y0, x1, y1

    def _make_update_request(self, data: xr.DataArray) -> _GoldUpdateRequest:
        return _GoldUpdateRequest(
            data=data,
            had_fit=hasattr(self, "edge_center"),
            roi_limits=self.params_roi.roi_limits,
            tab_index=self.params_tab.currentIndex(),
            edge_values=dict(self.params_edge.values),
            poly_values=dict(self.params_poly.values),
            spline_values=dict(self.params_spl.values),
            refit=self.refit_on_source_update_check.isChecked(),
        )

    def _apply_update_request(self, request: _GoldUpdateRequest) -> None:
        self._clear_edge_fit_results()
        self.data = request.data
        self._along_dim = str(self.data.dims[1])
        self.aw.set_input(self.data)
        self.aw.axes[0].autoRange()

        roi_rect = self.aw.axes[0].getViewBox().itemBoundingRect(self.aw.images[0])
        self.params_roi.roi.maxBounds = roi_rect
        self.params_roi._x_decimals = erlab.utils.array.effective_decimals(
            self.data[self._along_dim].values
        )
        self.params_roi._y_decimals = erlab.utils.array.effective_decimals(
            self.data.eV.values
        )
        xm, ym, xM, yM = self.params_roi.max_bounds
        for name in ("x0", "x1"):
            spin = typing.cast(
                "QtWidgets.QDoubleSpinBox", self.params_roi.widgets[name]
            )
            spin.setRange(xm, xM)
            spin.setDecimals(self.params_roi._x_decimals)
        for name in ("y0", "y1"):
            spin = typing.cast(
                "QtWidgets.QDoubleSpinBox", self.params_roi.widgets[name]
            )
            spin.setRange(ym, yM)
            spin.setDecimals(self.params_roi._y_decimals)
        self.params_roi.modify_roi(
            *self._clamp_roi_limits_to_bounds(request.roi_limits)
        )
        self.params_roi.update_pos()

        with (
            QtCore.QSignalBlocker(self.params_edge),
            QtCore.QSignalBlocker(self.params_poly),
            QtCore.QSignalBlocker(self.params_spl),
            QtCore.QSignalBlocker(self.params_tab),
        ):
            self._restore_parameter_group_values(self.params_edge, request.edge_values)
            self._restore_parameter_group_values(self.params_poly, request.poly_values)
            self._restore_parameter_group_values(self.params_spl, request.spline_values)
            self.params_tab.setCurrentIndex(request.tab_index)
        self._toggle_fast()
        self._sync_spline_lambda_enabled()

        if request.had_fit and request.refit:
            self.perform_edge_fit()

    @QtCore.Slot()
    def _flush_pending_update(self) -> None:
        request = self._pending_update_request
        if request is None:
            return
        if self._threadpool.activeThreadCount():
            self._pending_update_timer.start(50)
            return
        self._pending_update_request = None
        self._apply_update_request(request)
        self.finalize_source_refresh()

    def update_data(self, new_data: xr.DataArray) -> bool:
        data = self.validate_update_data(new_data)
        self._pending_update_request = self._make_update_request(data)
        self._abort_fit_task()
        if self._threadpool.activeThreadCount():
            self._pending_update_timer.start(50)
            return False
        self._flush_pending_update()
        return True

    def validate_update_data(self, new_data: xr.DataArray) -> xr.DataArray:
        data = erlab.interactive.utils.parse_data(new_data)
        if data.ndim != 2 or "eV" not in data.dims:
            raise ValueError("`data` must be a 2D DataArray with an `eV` dimension")
        if data.dims[0] != "eV":
            data = data.copy().T
        return data

    def _toggle_fast(self) -> None:
        self.params_edge.widgets["T (K)"].setDisabled(
            bool(self.params_edge.values["Fast"])
        )
        self.params_edge.widgets["Fix T"].setDisabled(
            bool(self.params_edge.values["Fast"])
        )

    def iterated(self, n: int, *, task: EdgeFitTask | None = None) -> None:
        if task is not None and task is not self._fit_task:
            return
        if n == 0:
            self.progress.setLabelText("")
        elif n == 2:
            self.start_time = time.perf_counter()
        elif n > 2:
            step_time_avg = (time.perf_counter() - self.start_time) / (n - 2)
            n_left = self.progress.maximum() - n
            time_left = n_left * step_time_avg
            self.progress.setLabelText(f"{round(time_left)} seconds left...")

        self.progress.setValue(n)
        self.pbar.setFormat(f"{n}/{self.progress.maximum()} finished")

    @property
    def roi_limits_ordered(self) -> tuple[float, float, float, float]:
        """Returns the ordered ROI limits respecting data coordinate directions."""
        x0, y0, x1, y1 = self.params_roi.roi_limits
        if self.data[self._along_dim][-1] < self.data[self._along_dim][0]:
            x0, x1 = x1, x0
        if self.data.eV[-1] < self.data.eV[0]:
            y0, y1 = y1, y0
        return x0, y0, x1, y1

    @QtCore.Slot()
    def perform_edge_fit(self) -> None:
        if self._pending_update_request is not None:
            return
        self.progress.setVisible(True)
        self.params_roi.draw_button.setChecked(False)
        x0, y0, x1, y1 = self.roi_limits_ordered
        params = self.params_edge.values
        n_total: int = len(
            self.data[self._along_dim]
            .coarsen({self._along_dim: int(params["Bin x"])}, boundary="trim")
            .mean()
            .sel({self._along_dim: slice(x0, x1)})
        )
        self.progress.setMaximum(n_total)
        self._abort_fit_task()
        task = EdgeFitTask(self.data, self._along_dim, x0, y0, x1, y1, params)
        self._fit_task = task
        task.signals.sigIterated.connect(
            lambda n, task=task: self.iterated(n, task=task)
        )
        task.signals.sigFinished.connect(
            lambda edge_center, edge_stderr, task=task: self.post_fit(
                edge_center, edge_stderr, task=task
            )
        )
        task.signals.sigFailed.connect(
            lambda message, task=task: self._handle_fit_failed(message, task=task)
        )
        self._threadpool.start(task)

    @QtCore.Slot()
    def abort_fit(self) -> None:
        self.sigAbortFitting.emit()

    def post_fit(
        self,
        edge_center: xr.DataArray,
        edge_stderr: xr.DataArray,
        *,
        task: EdgeFitTask | None = None,
    ) -> None:
        if task is not None and task is not self._fit_task:
            return
        self.progress.reset()
        self.edge_center, self.edge_stderr = edge_center, edge_stderr
        self._fit_task = None

        xval = self.edge_center[self._along_dim].values
        yval = self.edge_center.values
        for i in range(2):
            self.scatterplots[i].setData(x=xval, y=yval)
            self.errorbars[i].setData(x=xval, y=yval, height=self.edge_stderr.values)

        self.params_poly.setDisabled(False)
        self.params_spl.setDisabled(False)
        self.params_tab.setDisabled(False)
        self.perform_fit()

    @QtCore.Slot()
    def _abort_fit_task(self) -> None:
        task = self._fit_task
        if task is None:
            return
        self._fit_task = None
        task.abort_fit()
        self.progress.reset()

    def _handle_fit_failed(
        self, message: str, *, task: EdgeFitTask | None = None
    ) -> None:
        if task is not None and task is not self._fit_task:
            return
        self.progress.reset()
        self._fit_task = None
        erlab.interactive.utils.MessageDialog.critical(
            self,
            "Fermi Edge Fitting Failed",
            "An error occurred during fitting. See details below.",
            detailed_text=erlab.interactive.utils._format_traceback(message),
        )

    @property
    def edge_func(self) -> "Callable":
        """Returns the edge function."""
        if self.params_tab.currentIndex() == 0:
            return self._perform_poly_fit()
        return self._perform_spline_fit()

    @property
    def edge_params(self) -> dict[str, typing.Any]:
        """Returns the edge parameters."""
        if self.params_tab.currentIndex() == 0:
            return self.params_poly.values
        return self.params_spl.values

    def perform_fit(self) -> None:
        edgefunc = self.edge_func
        params = self.edge_params

        for i in range(2):
            xval = self.data[self._along_dim].values
            if i == 1 and params["Residuals"]:
                yval = np.zeros_like(xval)
            else:
                yval = edgefunc(xval)
            self.polycurves[i].setData(x=xval, y=yval)

        xval = self.edge_center[self._along_dim].values
        if params["Residuals"]:
            yval = edgefunc(xval) - self.edge_center.values
        else:
            yval = self.edge_center.values
        self.errorbars[1].setData(x=xval, y=yval)
        self.scatterplots[1].setData(x=xval, y=yval, height=self.edge_stderr)

        self.aw.axes[1].setVisible(True)

        if params["Corrected"]:
            self.aw.images[-1].setDataArray(self.corrected)
        self.aw.axes[2].setVisible(params["Corrected"])
        self.aw.hists[2].setVisible(params["Corrected"])
        self.sigUpdated.emit()
        self._notify_data_changed()

    def _perform_poly_fit(self):
        params = self.params_poly.values
        self.result = erlab.analysis.gold.poly_from_edge(
            center=self.edge_center,
            weights=1 / self.edge_stderr,
            degree=params["Degree"],
            method=self.params_edge.values["Method"],
            scale_covar=params["Scale cov"],
            along=self._along_dim,
        )

        return lambda x: self.result.modelfit_results.values.item().eval(x=x)

    def _perform_spline_fit(self):
        params = self.params_spl.values
        if params["Auto"]:
            params["lambda"] = None
        self.result = erlab.analysis.gold.spline_from_edge(
            center=self.edge_center,
            weights=np.asarray(1 / self.edge_stderr),
            lam=params["lambda"],
            along=self._along_dim,
        )
        return self.result

    @property
    def corrected(self) -> xr.DataArray:
        target = self.data if self.data_corr is None else self.data_corr
        return erlab.analysis.gold.correct_with_edge(
            target,
            self.result,
            along=self._along_dim,
            plot=False,
            shift_coords=self.edge_params["Shift coords"],
        )

    @QtCore.Slot()
    def _save_poly_fit(self) -> None:
        """Save the polynomial fit to a file."""
        if self.result is None:
            raise ValueError("No fit result available. Please perform a fit first.")

        erlab.interactive.utils.save_fit_ui(self.result, parent=self)

    @QtCore.Slot()
    def open_itool(self) -> None:
        tool = self._launch_output_imagetool(
            self.corrected,
            output_id=self.Output.CORRECTED,
        )
        if tool is not None:
            self._itool = tool

    def _current_fit_mode(self) -> typing.Literal["poly", "spl"]:
        return "poly" if self.params_tab.currentIndex() == 0 else "spl"

    def _fit_expression(
        self,
        mode: typing.Literal["poly", "spl"] | None = None,
        *,
        input_name: str | None = None,
    ) -> str:
        if mode is None:
            mode = self._current_fit_mode()
        p0 = self.params_edge.values
        match mode:
            case "poly":
                p1 = self.params_poly.values
            case "spl":
                p1 = self.params_spl.values
        x0, y0, x1, y1 = self.roi_limits_ordered

        arg_dict: dict[str, typing.Any] = {
            "along": self._along_dim,
            "angle_range": (x0, x1),
            "eV_range": (y0, y1),
            "bin_size": (p0["Bin x"], p0["Bin y"]),
            "temp": p0["T (K)"],
            "vary_temp": not p0["Fix T"],
            "bkg_slope": p0["Linear"],
            "resolution": p0["Resolution"],
            "fast": p0["Fast"],
            "method": p0["Method"],
            "scale_covar_edge": p0["Scale cov"],
        }

        match mode:
            case "poly":
                func: Callable = erlab.analysis.gold.poly
                arg_dict["degree"] = p1["Degree"]
            case "spl":
                func = erlab.analysis.gold.spline
                if p1["Auto"]:
                    arg_dict["lam"] = None
                else:
                    arg_dict["lam"] = p1["lambda"]

        if p0["Fast"]:
            del arg_dict["temp"]

        if mode == "poly" and not p1["Scale cov"]:
            arg_dict["scale_covar"] = False

        source_name = input_name or str(self._argnames["data"])
        return erlab.interactive.utils.generate_code(
            func,
            [f"|{source_name}|"],
            arg_dict,
            module="era.gold",
        )

    def _corrected_expression(
        self,
        mode: typing.Literal["poly", "spl"] | None = None,
        *,
        input_name: str | None = None,
    ) -> str:
        if mode is None:
            mode = self._current_fit_mode()
        match mode:
            case "poly":
                p1 = self.params_poly.values
            case "spl":
                p1 = self.params_spl.values
        corrected_target = input_name or self._argnames.get(
            "data_corr", self._argnames["data"]
        )
        return erlab.interactive.utils.generate_code(
            erlab.analysis.gold.correct_with_edge,
            args=[f"|{corrected_target}|", "|modelresult|"],
            kwargs={"along": self._along_dim, "shift_coords": p1["Shift coords"]},
            module="era.gold",
        )

    def _copy_label(
        self,
        mode: typing.Literal["poly", "spl"],
        *,
        include_corrected: bool | None = None,
    ) -> str:
        return (
            "Fit and correct current data with the polynomial edge model"
            if mode == "poly" and include_corrected
            else "Fit current data with the polynomial edge model"
            if mode == "poly"
            else "Fit and correct current data with the spline edge model"
            if include_corrected
            else "Fit current data with the spline edge model"
        )

    def _current_mode_copy_label(
        self,
        *,
        input_name: str | None = None,
        data: xr.DataArray | None = None,
    ) -> str:
        return self._copy_label(self._current_fit_mode())

    def _current_mode_corrected_label(
        self,
        *,
        input_name: str | None = None,
        data: xr.DataArray | None = None,
    ) -> str:
        return self._copy_label(self._current_fit_mode(), include_corrected=True)

    def _current_mode_copy_expression(
        self,
        *,
        input_name: str | None = None,
        data: xr.DataArray | None = None,
    ) -> str:
        return self._fit_expression(
            self._current_fit_mode(),
            input_name=input_name,
        )

    def _current_mode_corrected_prelude(
        self,
        *,
        input_name: str | None = None,
        data: xr.DataArray | None = None,
    ) -> str:
        return "modelresult = " + self._fit_expression(
            self._current_fit_mode(),
            input_name=input_name,
        )

    def _current_mode_corrected_expression(
        self,
        *,
        input_name: str | None = None,
        data: xr.DataArray | None = None,
    ) -> str:
        return self._corrected_expression(
            self._current_fit_mode(),
            input_name=input_name,
        )

    def _corrected_output(self) -> xr.DataArray:
        return self.corrected

    def to_dataset(self) -> xr.Dataset:
        self._ensure_serializable_state()
        return super().to_dataset()

    def duplicate(self, **kwargs) -> typing.Self:
        self._ensure_serializable_state()
        return super().duplicate(**kwargs)

    def _stop_server(self) -> bool:
        """Stop the fitter thread properly."""
        self._pending_update_request = None
        self._pending_update_timer.stop()
        self._abort_fit_task()
        return self._wait_for_threadpool(
            self._threadpool,
            timeout_ms=self.BACKGROUND_TASK_TIMEOUT_MS,
        )

    def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:
        """Overridden close event to ensure proper cleanup."""
        if not self._stop_server():
            logger.warning(
                "Gold fit worker did not stop within timeout; aborting window close"
            )
            if event is not None:
                event.ignore()
            return
        super().closeEvent(event)


class ResolutionFitThread(QtCore.QThread):
    sigFinished = QtCore.Signal(object, float)
    sigTimedOut = QtCore.Signal(float)
    sigErrored = QtCore.Signal(str)
    sigCancelled = QtCore.Signal()

    def __init__(
        self,
        fit_data: xr.DataArray,
        fit_params: dict[str, typing.Any],
        *,
        timeout: float,
    ) -> None:
        super().__init__()
        self._fit_data = fit_data.copy()
        self._fit_params = dict(fit_params)
        self._timeout = timeout
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    @QtCore.Slot()
    def run(self) -> None:
        t0 = time.perf_counter()
        timed_out = False
        cancelled = False
        self._cancel.clear()

        def _callback(*args, **kwargs) -> bool | None:
            nonlocal timed_out, cancelled

            interruption_requested = False
            try:
                interruption_requested = self.isInterruptionRequested()
            except RuntimeError:
                # The thread wrapper can be deleted during shutdown; treat as cancel.
                interruption_requested = True

            if self._cancel.is_set() or interruption_requested:
                cancelled = True
                return True

            if self._timeout > 0 and (time.perf_counter() - t0) >= self._timeout:
                timed_out = True
                return True
            return None

        try:
            result_ds = erlab.analysis.gold.quick_fit(
                self._fit_data,
                **self._fit_params,
                iter_cb=_callback,
            )
            # Materialize lazy fit outputs while this thread object is still alive.
            result_ds = result_ds.load()
        except Exception:
            if timed_out:
                self.sigTimedOut.emit(time.perf_counter() - t0)
            elif cancelled:
                self.sigCancelled.emit()
            else:
                self.sigErrored.emit(traceback.format_exc())
            return

        if cancelled:
            self.sigCancelled.emit()
        elif timed_out:
            self.sigTimedOut.emit(time.perf_counter() - t0)
        else:
            self.sigFinished.emit(result_ds, time.perf_counter() - t0)


class ResolutionTool(erlab.interactive.utils.ToolWindow):
    tool_name = "restool"
    COPY_PROVENANCE: typing.ClassVar = (
        erlab.interactive.utils.ToolScriptProvenanceDefinition(
            start_label="Start from current restool input data",
            label="Fit the current averaged edge distribution",
            expression_method="_copy_expression",
            assign="result",
        )
    )

    class StateModel(pydantic.BaseModel):
        data_name: str
        x0: float
        x1: float
        y0: float
        y1: float
        live_fit: bool
        refit_on_source_update: bool = False
        temp: float
        fix_temp: bool
        center: float
        fix_center: bool
        resolution: float
        fix_resolution: bool
        bkg_slope: bool
        method: str
        timeout: float
        max_nfev: int
        use_mev: bool
        results: tuple[str, str, str, str, str]

    @property
    def info_text(self) -> str:
        from erlab.utils.formatting import (
            format_darr_shape_html,
            format_html_accent,
            format_html_table,
        )

        status = self.tool_status
        info: str = f"<b>{self.tool_name}</b>" + format_darr_shape_html(self.tool_data)

        info += "<b>Initial Parameters</b>"

        param_dict: dict[str, str] = {
            "eV range": f"{status.x0} to {status.x1}",
            f"{self.y_dim} range": f"{status.y0} to {status.y1}",
            "<i>T</i>": f"{status.temp} K"
            + f" ({'fixed' if status.fix_temp else 'varied'})",
            "<i>E</i><sub>F</sub>": f"{status.center} eV"
            + f" ({'fixed' if status.fix_center else 'varied'})",
            "Δ<i>E</i>": f"{status.resolution} eV"
            + f" ({'fixed' if status.fix_resolution else 'varied'})",
            "Background above <i>E</i><sub>F</sub>": "linear"
            if status.bkg_slope
            else "constant",
            "Fitting method": status.method,
        }

        info += format_html_table(
            [[format_html_accent(k), v] for k, v in param_dict.items()]
        )

        info += "<br><b>Fit Result</b>"
        info += f"<br>{status.results[0]}"
        result_dict: dict[str, str] = {
            "Temperature": status.results[1],
            "Edge center": status.results[2],
            "Resolution": status.results[3],
            "Reduced chi-squared": status.results[4],
        }
        info += format_html_table(
            [[format_html_accent(k), v] for k, v in result_dict.items()]
        )
        return info

    @property
    def tool_status(self) -> StateModel:
        return self.StateModel(
            data_name=self.data_name,
            x0=self.x0_spin.value(),
            x1=self.x1_spin.value(),
            y0=self.y0_spin.value(),
            y1=self.y1_spin.value(),
            live_fit=self.live_check.isChecked(),
            refit_on_source_update=self.refit_on_source_update_check.isChecked(),
            temp=self.temp_spin.value(),
            fix_temp=self.fix_temp_check.isChecked(),
            center=self.center_spin.value(),
            fix_center=self.fix_center_check.isChecked(),
            resolution=self.res_spin.value(),
            fix_resolution=self.fix_res_check.isChecked(),
            bkg_slope=self.slope_check.isChecked(),
            method=self.method_combo.currentText(),
            timeout=self.timeout_spin.value(),
            max_nfev=self.nfev_spin.value(),
            use_mev=self.mev_check.isChecked(),
            results=(
                self.overview_label.text(),
                self.temp_val.text(),
                self.center_val.text(),
                self.res_val.text(),
                self.redchi_val.text(),
            ),
        )

    @tool_status.setter
    def tool_status(self, status: StateModel) -> None:
        self.data_name: str = status.data_name
        self.live_check.setChecked(False)

        self.x0_spin.setValue(status.x0)
        self.x1_spin.setValue(status.x1)
        self.y0_spin.setValue(status.y0)
        self.y1_spin.setValue(status.y1)
        self.live_check.setChecked(status.live_fit)
        self.refit_on_source_update_check.setChecked(status.refit_on_source_update)

        self.temp_spin.setValue(status.temp)
        self.fix_temp_check.setChecked(status.fix_temp)
        self.center_spin.setValue(status.center)
        self.fix_center_check.setChecked(status.fix_center)
        self.res_spin.setValue(status.resolution)
        self.fix_res_check.setChecked(status.fix_resolution)
        self.slope_check.setChecked(status.bkg_slope)
        method_index = LMFIT_METHODS.index(status.method)
        self.method_combo.setCurrentIndex(method_index)
        self.timeout_spin.setValue(status.timeout)
        self.nfev_spin.setValue(status.max_nfev)
        self.mev_check.setChecked(status.use_mev)

        self.overview_label.setText(status.results[0])
        self.temp_val.setText(status.results[1])
        self.center_val.setText(status.results[2])
        self.res_val.setText(status.results[3])
        self.redchi_val.setText(status.results[4])

    @property
    def tool_data(self) -> xr.DataArray:
        return self.data

    _sigTriggerFit = QtCore.Signal()

    def __init__(self, data: xr.DataArray, *, data_name: str | None = None) -> None:
        if (data.ndim != 2) or ("eV" not in data.dims):
            raise ValueError("Data must be 2D and have an 'eV' dimension.")
        super().__init__()
        erlab.interactive.utils.load_ui(
            str(importlib.resources.files(erlab.interactive).joinpath("restool.ui")),
            self,
        )
        self.setWindowTitle("")

        self.refit_on_source_update_check = QtWidgets.QCheckBox(
            "Refit on source update", self.groupBox_2
        )
        self.refit_on_source_update_check.setToolTip(
            "If checked, rerun the resolution fit when this tool refreshes from "
            "ImageTool."
        )
        typing.cast("QtWidgets.QGridLayout", self.groupBox_2.layout()).addWidget(
            self.refit_on_source_update_check, 10, 0, 1, 4
        )

        if data.dims.index("eV") != 1:
            data = data.T
        self.data = data

        self.y_dim: str = str(data.dims[0])

        x_coords = data["eV"].values
        y_coords = data[self.y_dim].values

        self._x_range = x_coords.min(), x_coords.max()
        self._y_range = y_coords.min(), y_coords.max()

        self._x_decimals = erlab.utils.array.effective_decimals(x_coords)
        self._y_decimals = erlab.utils.array.effective_decimals(y_coords)

        self.x0_spin.setRange(*self._x_range)
        self.x1_spin.setRange(*self._x_range)
        self.y0_spin.setRange(*self._y_range)
        self.y1_spin.setRange(*self._y_range)
        self.x0_spin.setDecimals(self._x_decimals)
        self.x1_spin.setDecimals(self._x_decimals)
        self.y0_spin.setDecimals(self._y_decimals)
        self.y1_spin.setDecimals(self._y_decimals)
        self.x0_spin.setSingleStep(10 ** -(self._x_decimals - 1))
        self.x1_spin.setSingleStep(10 ** -(self._x_decimals - 1))
        self.y0_spin.setSingleStep(10 ** -(self._y_decimals - 1))
        self.y1_spin.setSingleStep(10 ** -(self._y_decimals - 1))

        self.res_spin.setDecimals(self._x_decimals + 1)
        self.res_spin.setSingleStep(10 ** -(self._x_decimals - 1))
        self.res_spin.setValue(0.002)
        self.center_spin.setRange(*self._x_range)
        self.center_spin.setDecimals(self._x_decimals + 1)
        self.center_spin.setSingleStep(10 ** -(self._x_decimals - 1))

        if data_name is None:
            try:
                data_name = typing.cast(
                    "str",
                    varname.argname("data", func=self.__init__, vars_only=False),  # type: ignore[misc]
                )
            except varname.VarnameRetrievingError:
                data_name = "data"

        self.data_name = data_name

        self.plot0 = self.graphics_layout.addPlot(row=0, col=0)
        self.plot1 = self.graphics_layout.addPlot(row=1, col=0)
        self.plot1.setXLink(self.plot0)

        self.image = erlab.interactive.utils.xImageItem(axisOrder="row-major")
        self.image.setDataArray(self.data)
        self.plot0.addItem(self.image)

        self.edc_curve = pg.ScatterPlotItem(
            size=3,
            pen=pg.mkPen(None),
            brush=pg.mkBrush(255, 255, 255, 200),
            pxMode=True,
        )
        self.plot1.addItem(self.edc_curve)

        self.edc_fit = pg.PlotDataItem(pen=pg.mkPen("r"))
        self.plot1.addItem(self.edc_fit)

        y_offset = np.mean(self._y_range)
        initial_y_range = (self._y_range - y_offset) * 0.75 + y_offset
        self.y_region = pg.LinearRegionItem(
            values=initial_y_range,
            orientation="horizontal",
            swapMode="block",
            bounds=self._y_range,
        )
        self.y_region.sigRegionChanged.connect(self._y_region_changed)
        self.plot0.addItem(self.y_region)

        x_width = self._x_range[-1] - self._x_range[0]
        initial_x_range = (np.mean(self._x_range), self._x_range[-1] - x_width * 0.04)
        self.x_region = pg.LinearRegionItem(
            values=initial_x_range,
            orientation="vertical",
            swapMode="block",
            bounds=self._x_range,
        )
        self.x_region.sigRegionChanged.connect(self._x_region_changed)
        self.plot1.addItem(self.x_region)

        self.connect_signals()
        self.graphics_layout.setFocus()

        self.resize(800, 600)

        self._result_ds: xr.Dataset | None = None
        self._fit_thread: ResolutionFitThread | None = None
        self._fit_cancel_requested: bool = False
        self._fit_queued: bool = False
        self._pending_fit_action: Callable[[], None] | None = None
        self._fit_signature_current: tuple[typing.Any, ...] | None = None
        self._fit_signature_displayed: tuple[typing.Any, ...] | None = None

    def _configure_data(self, data: xr.DataArray) -> None:
        if (data.ndim != 2) or ("eV" not in data.dims):
            raise ValueError("Data must be 2D and have an 'eV' dimension.")
        if data.dims.index("eV") != 1:
            data = data.T

        self.data = data
        self.y_dim = str(data.dims[0])

        x_coords = data["eV"].values
        y_coords = data[self.y_dim].values

        self._x_range = x_coords.min(), x_coords.max()
        self._y_range = y_coords.min(), y_coords.max()

        self._x_decimals = erlab.utils.array.effective_decimals(x_coords)
        self._y_decimals = erlab.utils.array.effective_decimals(y_coords)

        self.x0_spin.setRange(*self._x_range)
        self.x1_spin.setRange(*self._x_range)
        self.y0_spin.setRange(*self._y_range)
        self.y1_spin.setRange(*self._y_range)
        self.x0_spin.setDecimals(self._x_decimals)
        self.x1_spin.setDecimals(self._x_decimals)
        self.y0_spin.setDecimals(self._y_decimals)
        self.y1_spin.setDecimals(self._y_decimals)
        self.x0_spin.setSingleStep(10 ** -(self._x_decimals - 1))
        self.x1_spin.setSingleStep(10 ** -(self._x_decimals - 1))
        self.y0_spin.setSingleStep(10 ** -(self._y_decimals - 1))
        self.y1_spin.setSingleStep(10 ** -(self._y_decimals - 1))

        self.res_spin.setDecimals(self._x_decimals + 1)
        self.res_spin.setSingleStep(10 ** -(self._x_decimals - 1))
        self.center_spin.setRange(*self._x_range)
        self.center_spin.setDecimals(self._x_decimals + 1)
        self.center_spin.setSingleStep(10 ** -(self._x_decimals - 1))

        self.image.setDataArray(self.data, update_labels=True)
        self.plot0.autoRange()
        self.x_region.setBounds(self._x_range)
        self.y_region.setBounds(self._y_range)

    def _clear_fit_outputs(self) -> None:
        self._result_ds = None
        self._fit_signature_current = None
        self._fit_signature_displayed = None
        self.edc_fit.setData(x=[], y=[])
        self.overview_label.setText("No fit results")
        self.temp_val.setText("—")
        self.center_val.setText("—")
        self.res_val.setText("—")
        self.redchi_val.setText("—")

    @property
    def _x_range_ui(self) -> tuple[float, float]:
        x0 = round(self.x0_spin.value(), self._x_decimals)
        x1 = round(self.x1_spin.value(), self._x_decimals)
        return x0, x1

    @property
    def _y_range_ui(self) -> tuple[float, float]:
        y0 = round(self.y0_spin.value(), self._y_decimals)
        y1 = round(self.y1_spin.value(), self._y_decimals)
        return y0, y1

    @property
    def x_range(self) -> tuple[float, float]:
        """Currently selected x range (eV) for the fit."""
        x0, x1 = self._x_range_ui
        if self.data.eV[-1] < self.data.eV[0]:
            x0, x1 = x1, x0
        return x0, x1

    @property
    def y_range(self) -> tuple[float, float]:
        """Currently selected y range to average EDCs."""
        y0, y1 = self._y_range_ui
        if self.data[self.y_dim][-1] < self.data[self.y_dim][0]:
            y0, y1 = y1, y0
        return y0, y1

    def _fit_running(self) -> bool:
        # A non-None fit thread is considered in-flight, even if Qt has not yet
        # flipped `isRunning()` to True. This avoids replacing/deallocating a
        # just-starting thread.
        return self._fit_thread is not None

    def _cancel_fit(self, *, wait: bool = False, timeout_ms: int | None = 5000) -> bool:
        thread = self._fit_thread
        if thread is not None:
            thread.cancel()
            thread.requestInterruption()
        self._fit_cancel_requested = True
        self._fit_queued = False
        self._pending_fit_action = None
        if wait and thread is not None:
            finished = thread.wait() if timeout_ms is None else thread.wait(timeout_ms)
            if finished:
                self._finalize_fit_thread(thread)
            return finished
        return True

    def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:
        if not self._cancel_fit(wait=True):
            logger.warning(
                "Resolution fit worker did not stop within timeout; "
                "aborting window close"
            )
            if event is not None:
                event.ignore()
            return
        super().closeEvent(event)

    def _cancel_background_work(self, *, timeout_ms: int) -> bool:
        return self._cancel_fit(wait=True, timeout_ms=timeout_ms)

    def update_data(self, new_data: xr.DataArray) -> bool:
        had_fit = self._result_ds is not None
        status = self.tool_status.model_copy(
            update={"results": ("No fit results", "—", "—", "—", "—")}
        )

        def _apply_update(validated: xr.DataArray) -> None:
            self._configure_data(validated)
            self._clear_fit_outputs()
            self.tool_status = status
            self._update_edc()
            self._notify_data_changed()

            if had_fit and self.refit_on_source_update_check.isChecked():
                self.do_fit()

        return self._perform_source_update(new_data, apply_update=_apply_update)

    def validate_update_data(self, new_data: xr.DataArray) -> xr.DataArray:
        data = erlab.interactive.utils.parse_data(new_data)
        if (data.ndim != 2) or ("eV" not in data.dims):
            raise ValueError("Data must be 2D and have an 'eV' dimension.")
        if data.dims.index("eV") != 1:
            data = data.T
        return data

    def _update_edc(self) -> None:
        """Calculate averaged EDC and update the plot."""
        with xr.set_options(keep_attrs=True):
            self.averaged_edc = self.data.sel({self.y_dim: slice(*self.y_range)}).mean(
                self.y_dim
            )
        self.edc_curve.setData(
            x=self.averaged_edc["eV"].values, y=self.averaged_edc.values
        )
        self._clear_fit_preview()

    def _clear_fit_preview(self) -> None:
        # Keep the previous fit line visible during live refits to avoid flicker.
        if self.live_check.isChecked():
            return
        self._fit_signature_displayed = None
        self.edc_fit.setData(x=[], y=[])

    @property
    def _guessed_params(self) -> "lmfit.Parameters":
        with xr.set_options(keep_attrs=True):
            target = self.averaged_edc.sel(eV=slice(*self.x_range))
        return erlab.analysis.fit.models.FermiEdgeModel().guess(
            target, target.eV.values
        )

    @QtCore.Slot()
    def _guess_temp(self) -> None:
        self.temp_spin.setValue(self._guessed_params["temp"].value)

    @QtCore.Slot()
    def _guess_center(self) -> None:
        self.center_spin.setValue(self._guessed_params["center"].value)

    @property
    def fit_params(self) -> dict[str, typing.Any]:
        """Current arguments for :func:`erlab.analysis.gold.quick_fit`."""
        return {
            "eV_range": self.x_range,
            "method": self.method_combo.currentText(),
            "temp": self.temp_spin.value(),
            "resolution": self.res_spin.value(),
            "center": self.center_spin.value(),
            "fix_temp": self.fix_temp_check.isChecked(),
            "fix_center": self.fix_center_check.isChecked(),
            "fix_resolution": self.fix_res_check.isChecked(),
            "bkg_slope": self.slope_check.isChecked(),
        }

    def _fit_failed(self, info_text: str) -> None:
        self.overview_label.setText(
            f'<span style="font-weight:600; color:#ff5555;">{info_text}</span>'
        )

    def _current_fit_signature(self) -> tuple[typing.Any, ...]:
        params = self.fit_params
        return (
            self.y_range,
            params["eV_range"],
            params["method"],
            params["temp"],
            params["resolution"],
            params["center"],
            params["fix_temp"],
            params["fix_center"],
            params["fix_resolution"],
            params["bkg_slope"],
            self.nfev_spin.value(),
        )

    def _invalidate_displayed_fit_if_stale(self) -> tuple[typing.Any, ...]:
        signature = self._current_fit_signature()
        self._fit_signature_current = signature
        if (
            self._fit_signature_displayed is not None
            and self._fit_signature_displayed != signature
        ):
            self._fit_signature_displayed = None
            self.edc_fit.setData(x=[], y=[])
        return signature

    def _queue_fit_action(
        self,
        thread: ResolutionFitThread,
        action: Callable[[], None],
        *,
        allow_when_cancelled: bool = False,
    ) -> None:
        if self._fit_thread is not thread:
            return
        if self._fit_cancel_requested and not allow_when_cancelled:
            return
        self._pending_fit_action = action
        if self._fit_thread is None:
            self._pending_fit_action = None
            action()

    def _finalize_fit_thread(self, thread: ResolutionFitThread) -> None:
        if self._fit_thread is not thread:
            return
        action = self._pending_fit_action
        self._pending_fit_action = None
        self._fit_thread = None
        was_cancelled = self._fit_cancel_requested
        self._fit_cancel_requested = False
        if action is not None:
            action()
        if self._fit_queued and not was_cancelled:
            self._fit_queued = False
            self._start_fit_worker()
        thread.deleteLater()

    def _start_fit_worker(self) -> bool:
        if self._fit_thread is not None:
            return False

        fit_signature = self._fit_signature_current
        if fit_signature is None:
            fit_signature = self._current_fit_signature()

        thread = ResolutionFitThread(
            self.averaged_edc,
            self.fit_params | {"max_nfev": self.nfev_spin.value()},
            timeout=self.timeout_spin.value(),
        )
        if hasattr(thread, "setServiceLevel"):
            thread.setServiceLevel(QtCore.QThread.QualityOfService.High)
        thread.finished.connect(lambda thread=thread: self._finalize_fit_thread(thread))

        thread.sigFinished.connect(
            lambda result_ds, fit_time, thread=thread: self._queue_fit_action(
                thread,
                lambda: self._handle_fit_success(result_ds, fit_time, fit_signature),
            )
        )
        thread.sigTimedOut.connect(
            lambda elapsed, thread=thread: self._queue_fit_action(
                thread, lambda: self._handle_fit_timeout(elapsed, fit_signature)
            )
        )
        thread.sigErrored.connect(
            lambda message, thread=thread: self._queue_fit_action(
                thread, lambda: self._handle_fit_error(message, fit_signature)
            )
        )
        thread.sigCancelled.connect(
            lambda thread=thread: self._queue_fit_action(
                thread, self._handle_fit_cancelled, allow_when_cancelled=True
            )
        )

        self._fit_thread = thread
        self._fit_cancel_requested = False
        thread.start()
        return True

    def _handle_fit_cancelled(self) -> None:
        return

    def _handle_fit_timeout(
        self, elapsed: float, fit_signature: tuple[typing.Any, ...]
    ) -> None:
        if fit_signature != self._fit_signature_current:
            return
        self._result_ds = None
        self._fit_signature_displayed = None
        self.live_check.setChecked(False)
        self._fit_failed(f"Fit timed out in {elapsed:.2f} seconds")
        self.edc_fit.setData(x=[], y=[])

    def _handle_fit_error(
        self, message: str, fit_signature: tuple[typing.Any, ...]
    ) -> None:
        if fit_signature != self._fit_signature_current:
            return
        self._result_ds = None
        self._fit_signature_displayed = None
        self.live_check.setChecked(False)
        self.edc_fit.setData(x=[], y=[])
        self._fit_failed("Fit failed")
        logger.error("Error while fitting resolution tool data:\n%s", message)

    def _handle_fit_success(
        self,
        result_ds: xr.Dataset,
        fit_time: float,
        fit_signature: tuple[typing.Any, ...],
    ) -> None:
        if fit_signature != self._fit_signature_current:
            return
        self._result_ds = result_ds
        self._fit_signature_displayed = fit_signature

        # Update plot
        self.edc_fit.setData(
            x=self._result_ds["eV"].values, y=self._result_ds.modelfit_best_fit.values
        )

        # Update overview
        modelresult = self._result_ds.modelfit_results.item()
        if not modelresult.success:
            self._fit_failed(
                f"Fit failed in {fit_time:.2f} s (nfev = {modelresult.nfev})"
            )
        else:
            self.overview_label.setText(
                '<span style="font-weight:600; color:#32cd32;">'
                f"Fit converged (nfev = {modelresult.nfev})"
                "</span>"
            )

        def _get_param_text(param: str) -> str:
            """Get the string representation of a parameter value."""
            factor = 1e3 if (param != "temp" and self.mev_check.isChecked()) else 1
            unit: str = (
                "K"
                if param == "temp"
                else ("meV" if self.mev_check.isChecked() else "eV")
            )
            if hasattr(modelresult, "uvars") and modelresult.uvars is not None:
                return f"{modelresult.uvars[param] * factor:P} {unit}"
            return f"{modelresult.params[param].value * factor:.4f} {unit}"

        for param, textedit in zip(
            ("temp", "center", "resolution"),
            (self.temp_val, self.center_val, self.res_val),
            strict=True,
        ):
            textedit.setText(_get_param_text(param))
        redchi: float | None = modelresult.redchi
        self.redchi_val.setText(f"{redchi:.4f}" if redchi is not None else "—")

    @QtCore.Slot()
    def do_fit(self) -> None:
        """Perform a fit on the averaged EDC and update results."""
        self._invalidate_displayed_fit_if_stale()
        if self._fit_running():
            self._fit_queued = True
            return
        self._fit_queued = False
        self._start_fit_worker()

    def connect_signals(self) -> None:
        self.x0_spin.valueChanged.connect(self._update_region)
        self.x1_spin.valueChanged.connect(self._update_region)
        self.y0_spin.valueChanged.connect(self._update_region)
        self.y1_spin.valueChanged.connect(self._update_region)
        self._x_region_changed()
        self._y_region_changed()  # Set spinbox initial values

        self.go_btn.clicked.connect(self.do_fit)
        self._sigTriggerFit.connect(self.do_fit)
        self.guess_temp_btn.clicked.connect(self._guess_temp)
        self.guess_center_btn.clicked.connect(self._guess_center)
        self.copy_btn.clicked.connect(self.copy_code)

    def _copy_expression(
        self,
        *,
        input_name: str | None = None,
        data: xr.DataArray | None = None,
    ) -> str:
        data_name = erlab.interactive.utils.generate_code(
            xr.DataArray.sel,
            args=[],
            kwargs={self.y_dim: slice(*self.y_range)},
            module=input_name or self.data_name,
        )

        # Check if any parameters are the same as the automatically guessed ones
        params = dict(self.fit_params)
        params_guessed = self._guessed_params
        for k in ("temp", "center", "resolution"):
            if params[k] == params_guessed[k].value:
                del params[k]

        params["plot"] = True

        return erlab.interactive.utils.generate_code(
            erlab.analysis.gold.quick_fit,
            args=[f"|{data_name}|"],
            kwargs=params,
            module="era.gold",
            copy=False,
        )

    @QtCore.Slot()
    def _update_region(self) -> None:
        """Update the region items when the spinboxes are changed."""
        if self._x_range_ui != self.x_region.getRegion():
            self.x_region.setRegion(self._x_range_ui)
        if self._y_range_ui != self.y_region.getRegion():
            self.y_region.setRegion(self._y_range_ui)

    @QtCore.Slot()
    def _x_region_changed(self) -> None:
        """Update the x0 and x1 spinboxes when the region is changed.

        If live fitting is enabled, trigger a fit.
        """
        self.x0_spin.blockSignals(True)
        self.x1_spin.blockSignals(True)

        x0, x1 = self.x_region.getRegion()
        self.x0_spin.setValue(x0)
        self.x1_spin.setValue(x1)

        self.x0_spin.blockSignals(False)
        self.x1_spin.blockSignals(False)

        self.x0_spin.setMaximum(self.x1_spin.value())
        self.x1_spin.setMinimum(self.x0_spin.value())

        self._clear_fit_preview()
        if self.live_check.isChecked():
            self._sigTriggerFit.emit()

    @QtCore.Slot()
    def _y_region_changed(self) -> None:
        """Update the y0 and y1 spinboxes when the region is changed.

        Also updates the averaged EDC plot.
        If live fitting is enabled, trigger a fit.
        """
        self.y0_spin.blockSignals(True)
        self.y1_spin.blockSignals(True)

        y0, y1 = self.y_region.getRegion()
        self.y0_spin.setValue(y0)
        self.y1_spin.setValue(y1)

        self.y0_spin.blockSignals(False)
        self.y1_spin.blockSignals(False)

        self.y0_spin.setMaximum(self.y1_spin.value())
        self.y1_spin.setMinimum(self.y0_spin.value())

        self._update_edc()
        if self.live_check.isChecked():
            self._sigTriggerFit.emit()


def goldtool(
    data: xr.DataArray,
    data_corr: xr.DataArray | None = None,
    *,
    data_name: str | None = None,
    execute: bool | None = None,
    **kwargs,
) -> GoldTool:
    """Interactive tool for correcting curved Fermi edges.

    This tool can also be accessed from the right-click context menu of an image plot in
    an ImageTool window.

    Parameters
    ----------
    data
        The data to perform Fermi edge fitting on. Must be a 2D DataArray with an 'eV'
        dimension.
    data_corr
        The data to correct with the edge. Defaults to ``data``.
    data_name
        Name of the data used in generating the code snipped copied to the clipboard.
        Overrides automatic detection.
    **kwargs
        Arguments passed onto `erlab.interactive.utils.AnalysisWindow`.

    """
    if data_name is None:
        try:
            data_name = str(varname.argname("data", func=goldtool, vars_only=False))
        except varname.VarnameRetrievingError:
            data_name = "data"

    with erlab.interactive.utils.setup_qapp(execute):
        win = GoldTool(data, data_corr, data_name=data_name, **kwargs)
        win.show()
        win.raise_()
        win.activateWindow()

    return win


def restool(
    data: xr.DataArray, *, data_name: str | None = None, execute: bool | None = None
) -> ResolutionTool:
    """Interactive tool for precise resolution fitting of EDCs.

    This tool can also be accessed from the right-click context menu of an image plot in
    an ImageTool window.

    Parameters
    ----------
    data
        Data to visualize. Must be a 2D DataArray with an 'eV' dimension.
    data_name
        Name of the data variable in the generated code. If not provided, the name is
        automatically determined.
    """
    if data_name is None:
        try:
            data_name = str(varname.argname("data", func=restool, vars_only=False))
        except varname.VarnameRetrievingError:
            data_name = "data"

    with erlab.interactive.utils.setup_qapp(execute):
        win = ResolutionTool(data, data_name=data_name)
        win.show()
        win.raise_()
        win.activateWindow()

    return win
