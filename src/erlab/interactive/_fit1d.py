"""Interactive tool for setting initial conditions for curve fitting."""

from __future__ import annotations

import concurrent.futures
import time
import typing

import numpy as np
import pydantic
import pyqtgraph as pg
import xarray_lmfit  # noqa: F401
from qtpy import QtCore, QtGui, QtWidgets

import erlab.interactive.utils

if typing.TYPE_CHECKING:
    from collections.abc import Mapping

    import lmfit
    import varname
    import xarray as xr
else:
    import lazy_loader as _lazy

    varname = _lazy.load("varname")
    lmfit = _lazy.load("lmfit")


class _ParameterEditDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, model: _ParameterTableModel, parent=None) -> None:
        super().__init__(parent)
        self._model = model

    def createEditor(self, parent, option, index):
        return QtWidgets.QLineEdit(parent)

    def setEditorData(self, editor, index):
        if isinstance(editor, QtWidgets.QLineEdit):
            text = self._model.edit_value_string(index.row(), index.column())
            editor.setText(text)
            return
        super().setEditorData(editor, index)

    def setModelData(self, editor, model, index):
        if isinstance(editor, QtWidgets.QLineEdit):
            model.setData(index, editor.text(), QtCore.Qt.ItemDataRole.EditRole)
            return
        super().setModelData(editor, model, index)


class _ParameterTableModel(QtCore.QAbstractTableModel):
    sigParamsChanged = QtCore.Signal()

    _COLUMN_NAMES = ("Parameter", "Value", "StdErr", "Min", "Max", "Vary")

    def __init__(self, params: lmfit.Parameters | None, parent=None) -> None:
        super().__init__(parent)
        self._params = params
        self._param_names = list(params.keys()) if params is not None else []

    @property
    def params(self) -> lmfit.Parameters | None:
        return self._params

    def set_params(self, params: lmfit.Parameters | None) -> None:
        self.beginResetModel()
        self._params = params
        self._param_names = list(params.keys()) if params is not None else []
        self.endResetModel()
        self.sigParamsChanged.emit()

    def rowCount(self, parent: QtCore.QModelIndex | None = None) -> int:
        return len(self._param_names)

    def columnCount(self, parent: QtCore.QModelIndex | None = None) -> int:
        return len(self._COLUMN_NAMES)

    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> typing.Any:
        if (
            role == QtCore.Qt.ItemDataRole.DisplayRole
            and orientation == QtCore.Qt.Orientation.Horizontal
        ):
            return self._COLUMN_NAMES[section]
        return None

    def data(
        self,
        index: QtCore.QModelIndex,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> typing.Any:
        if not index.isValid():
            return None
        if self._params is None:
            return None
        param = self._params[self._param_names[index.row()]]
        col = index.column()

        if role in (
            QtCore.Qt.ItemDataRole.DisplayRole,
            QtCore.Qt.ItemDataRole.EditRole,
        ):
            if col == 0:
                return param.name
            if col == 1:
                return self._format_value(param.value)
            if col == 2:
                if param.stderr is None:
                    return ""
                return self._format_scientific(param.stderr)
            if col == 3:
                return "-inf" if not np.isfinite(param.min) else float(param.min)
            if col == 4:
                return "inf" if not np.isfinite(param.max) else float(param.max)
            if col == 5:
                return None

        if role == QtCore.Qt.ItemDataRole.ToolTipRole:
            return self._param_tooltip(param)

        if role == QtCore.Qt.ItemDataRole.CheckStateRole and col == 5:
            if self._is_expr_param(param):
                return None
            return (
                QtCore.Qt.CheckState.Checked
                if param.vary
                else QtCore.Qt.CheckState.Unchecked
            )
        return None

    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlag:
        if not index.isValid():
            return QtCore.Qt.ItemFlag.NoItemFlags
        if self._params is None:
            return QtCore.Qt.ItemFlag.NoItemFlags
        param = self._params[self._param_names[index.row()]]
        flags = QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable
        if index.column() in (1, 3, 4) and not self._is_expr_param(param):
            flags |= QtCore.Qt.ItemFlag.ItemIsEditable
        if index.column() == 5 and not self._is_expr_param(param):
            flags |= QtCore.Qt.ItemFlag.ItemIsUserCheckable
        if index.column() == 5 and self._is_expr_param(param):
            flags &= ~QtCore.Qt.ItemFlag.ItemIsEnabled
        return flags

    def setData(
        self,
        index: QtCore.QModelIndex,
        value: typing.Any,
        role: int = QtCore.Qt.ItemDataRole.EditRole,
    ) -> bool:
        if not index.isValid():
            return False
        if self._params is None:
            return False

        param = self._params[self._param_names[index.row()]]
        if self._is_expr_param(param):
            return False
        col = index.column()

        if col == 5 and role in (
            QtCore.Qt.ItemDataRole.CheckStateRole,
            QtCore.Qt.ItemDataRole.EditRole,
        ):
            try:
                check_state = QtCore.Qt.CheckState(value)
            except Exception:
                check_state = (
                    QtCore.Qt.CheckState.Checked
                    if bool(value)
                    else QtCore.Qt.CheckState.Unchecked
                )
            param.vary = check_state == QtCore.Qt.CheckState.Checked
            self.dataChanged.emit(index, index, [QtCore.Qt.ItemDataRole.CheckStateRole])
            self.sigParamsChanged.emit()
            return True

        if role != QtCore.Qt.ItemDataRole.EditRole:
            return False

        try:
            if col == 1:
                param.value = float(value)
            elif col == 3:
                param.min = self._parse_bound(value, default=-np.inf)
            elif col == 4:
                param.max = self._parse_bound(value, default=np.inf)
            else:
                return False
        except (TypeError, ValueError):
            return False

        if param.min > param.max:
            param.max = param.min
        if param.value < param.min:
            param.value = param.min
        if param.value > param.max:
            param.value = param.max

        top_left = self.index(index.row(), 1)
        bottom_right = self.index(index.row(), 4)
        self.dataChanged.emit(
            top_left,
            bottom_right,
            [QtCore.Qt.ItemDataRole.DisplayRole, QtCore.Qt.ItemDataRole.EditRole],
        )
        self.sigParamsChanged.emit()
        return True

    def edit_value_string(self, row: int, column: int) -> str:
        if self._params is None:
            return ""
        param = self._params[self._param_names[row]]
        if column == 1:
            return self._format_value(param.value)
        if column == 2:
            if param.stderr is None:
                return ""
            return self._format_scientific(param.stderr)
        if column == 3:
            return self._format_bound(param.min, default="-inf")
        if column == 4:
            return self._format_bound(param.max, default="inf")
        return ""

    def param_name(self, row: int) -> str:
        return self._param_names[row]

    def param_at(self, row: int) -> lmfit.Parameter:
        if self._params is None:
            raise IndexError("Parameters are not initialized.")
        return self._params[self._param_names[row]]

    @staticmethod
    def _is_expr_param(param: lmfit.Parameter) -> bool:
        return bool(param.expr)

    @staticmethod
    def _param_tooltip(param: lmfit.Parameter) -> str:
        expr = f"expr: {param.expr}\n" if param.expr else ""
        value_line = f"value: {param.value}"
        if param.stderr is not None and np.isfinite(param.stderr):
            uncertainty = _ParameterTableModel._format_uncertainty(
                param.value, param.stderr
            )
            value_line = f"value: {uncertainty}"
        return (
            f"{expr}{value_line}\n"
            f"min: {param.min}\n"
            f"max: {param.max}\n"
            f"vary: {param.vary}"
        )

    @staticmethod
    def _parse_bound(value: typing.Any, default: float) -> float:
        if value is None:
            return default
        if isinstance(value, str):
            stripped = value.strip()
            if stripped == "":
                return default
            if stripped.lower() in {"inf", "+inf", "infinity", "+infinity"}:
                return np.inf
            if stripped.lower() in {"-inf", "-infinity"}:
                return -np.inf
        return float(value)

    @staticmethod
    def _format_float(value: float) -> str:
        return f"{value:.16g}"

    @staticmethod
    def _format_value(value: float) -> str:
        if np.isnan(value):
            return "nan"
        if np.isposinf(value):
            return "inf"
        if np.isneginf(value):
            return "-inf"
        return f"{value:.16g}"

    @staticmethod
    def _format_scientific(value: float) -> str:
        if np.isnan(value):
            return "nan"
        if np.isposinf(value):
            return "inf"
        if np.isneginf(value):
            return "-inf"
        return f"{value:.3e}"

    @staticmethod
    def _format_uncertainty(value: float, stderr: float) -> str:
        import uncertainties

        return f"{uncertainties.ufloat(value, stderr):P}"

    @staticmethod
    def _format_bound(value: float, default: str) -> str:
        if not np.isfinite(value):
            return default
        return f"{value:.16g}"


class _MultiPeakState(pydantic.BaseModel):
    npeaks: int
    peak_shape: str
    fd: bool
    background: str
    degree: int
    convolve: bool


class Fit1DTool(erlab.interactive.utils.ToolWindow):
    """GUI for selecting initial parameters and bounds for 1D curve fitting.

    Parameters
    ----------
    model
        The lmfit model to fit. If `None`, uses :class:`MultiPeakModel
        <erlab.analysis.fit.models.MultiPeakModel>`, and displays options for model
        initialization.
    data
        The 1D data to fit.
    data_name
        Optional display name for the dataset.

    Signals
    -------
    sigFitExecuted(lmfit.Parameters)
        Emitted after a successful fit with the latest parameters.
    """

    tool_name = "fit1d"

    class StateModel(pydantic.BaseModel):
        data_name: str
        model_name: str
        model_is_default: bool
        multipeak: _MultiPeakState | None = None
        domain: tuple[float, float] | None = None
        show_components: bool
        timeout: float
        max_nfev: int
        method: str
        slider_widths: dict[str, float]
        params: dict[str, dict[str, float | bool]]

    @property
    def tool_data(self) -> xr.DataArray:
        return self._data

    PLOT_RESAMPLE: int = 10
    FIT_COLOR: str = "c"

    sigFitExecuted = QtCore.Signal(object)

    def __init__(
        self,
        data: xr.DataArray,
        model: lmfit.Model | None = None,
        params: lmfit.Parameters | Mapping[str, typing.Any] = None,
        *,
        data_name: str | None = None,
        model_name: str | None = None,
    ) -> None:
        super().__init__()
        data = erlab.interactive.utils.parse_data(data)
        if data.ndim != 1:
            raise ValueError("`data` must be a 1D DataArray")

        self._data = data
        self._coord_name = str(data.dims[0])
        if model is None:
            self._model = erlab.analysis.fit.models.MultiPeakModel(
                npeaks=1,
                peak_shapes="lorentzian",
                fd=False,
                background="none",
                convolve=True,
                segmented=self._auto_segmented(convolve=True),
            )
            self._model_is_default = True
        else:
            self._model = model
            self._model_is_default = False
        if data_name is None:
            data_name = "data"
        if model_name is None:
            model_name = "model"
        self._data_name = data_name
        self._argnames = {"data": data_name, "model": model_name}

        if params is None:
            params = self._model.make_params()
        elif not isinstance(params, lmfit.Parameters):
            params = self._model.make_params(**params)

        self._params = params
        self._initial_params = self._params.copy()
        self._current_row: int | None = None
        self._slider_steps: int = 10000
        self._slider_updating: bool = False
        self._slider_dragging: bool = False
        self._slider_widths: dict[str, float] = {}
        self._last_fit_y: np.ndarray | None = None
        self._last_residual: np.ndarray | None = None
        self._last_result: lmfit.model.ModelResult | None = None
        self._last_result_ds: xr.Dataset | None = None
        self._slider_drag_range: tuple[float, float] | None = None
        self._fit_is_current = False
        self._table_widths_initialized = False
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._fit_future: concurrent.futures.Future | None = None

        self._build_ui()
        self._update_fit_curve()

    def _build_ui(self) -> None:
        self.setWindowTitle("Fit Initial Conditions")
        self.resize(733, 453)

        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        right_container = QtWidgets.QWidget(central)
        right_layout = QtWidgets.QVBoxLayout(right_container)
        right_container.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        layout.addWidget(right_container)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical, central)
        layout.addWidget(splitter, stretch=1)

        plot_container = QtWidgets.QWidget(splitter)
        plot_layout = QtWidgets.QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        self.plot_widget = pg.GraphicsLayoutWidget()
        plot_layout.addWidget(self.plot_widget)
        self.residual_plot = self.plot_widget.addPlot(row=0, col=0)
        self.main_plot = self.plot_widget.addPlot(row=1, col=0)
        self.residual_plot.setXLink(self.main_plot)
        self.residual_plot.hideAxis("bottom")
        self.plot_widget.ci.layout.setRowStretchFactor(0, 1)
        self.plot_widget.ci.layout.setRowStretchFactor(1, 3)
        self.plot_widget.ci.layout.setRowSpacing(0, 20)
        self.main_plot.getAxis("bottom").autoSIPrefix = False

        self.main_plot.getAxis("left").setWidth(50)
        self.residual_plot.getAxis("left").setWidth(50)

        for plot_item in (self.main_plot, self.residual_plot):
            plot_item.setDefaultPadding(0)
            font = QtGui.QFont()
            font.setPointSizeF(11.0)
            for ax in ("left", "top", "right", "bottom"):
                plot_item.getAxis(ax).setTickFont(font)
                plot_item.getAxis(ax).setStyle(
                    autoExpandTextSpace=True, autoReduceTextSpace=True
                )

        self.legend = self.main_plot.addLegend(offset=(5, 5))

        self.data_curve = self.main_plot.plot(
            pen=None,
            symbol="o",
            symbolSize=6,
            symbolBrush=pg.mkBrush(30, 30, 30),
            symbolPen=pg.mkPen(120, 120, 120),
            name="Data",
        )
        self.fit_curve = self.main_plot.plot(
            pen=pg.mkPen(self.FIT_COLOR, width=2), name="Fit"
        )
        self.residual_curve = self.residual_plot.plot(
            pen=None,
            symbol="o",
            symbolSize=6,
            symbolBrush=pg.mkBrush(30, 30, 30),
            symbolPen=pg.mkPen(120, 120, 120),
        )
        self.residual_zero = self.residual_plot.addLine(
            y=0.0, pen=pg.mkPen(self.FIT_COLOR, width=2)
        )
        self.component_curves: dict[str, pg.PlotCurveItem] = {}
        self.main_plot.showGrid(x=True, y=True, alpha=0.2)
        self.residual_plot.showGrid(x=True, y=True, alpha=0.2)
        self.main_plot.setLabel("bottom", self._coord_name)
        self.main_plot.setLabel("left", self._data.name)
        self.residual_plot.setLabel("left", "Residual")

        xvals = self._x_values()
        x_min = float(np.nanmin(xvals))
        x_max = float(np.nanmax(xvals))
        self.fit_line_min = pg.InfiniteLine(
            pos=x_min,
            angle=90,
            movable=True,
            bounds=(x_min, x_max),
            pen=pg.mkPen(self.FIT_COLOR, width=1),
        )
        self.fit_line_max = pg.InfiniteLine(
            pos=x_max,
            angle=90,
            movable=True,
            bounds=(x_min, x_max),
            pen=pg.mkPen(self.FIT_COLOR, width=1),
        )
        self.main_plot.addItem(self.fit_line_min)
        self.main_plot.addItem(self.fit_line_max)
        self.fit_line_min.sigPositionChanged.connect(self._sync_domain_from_lines)
        self.fit_line_max.sigPositionChanged.connect(self._sync_domain_from_lines)

        components_container = QtWidgets.QWidget(splitter)
        components_layout = QtWidgets.QHBoxLayout(components_container)
        components_layout.setContentsMargins(0, 0, 0, 0)
        self.components_check = QtWidgets.QCheckBox("Show components")
        self.components_check.setChecked(False)
        self.components_check.toggled.connect(self._update_fit_curve)
        components_layout.addWidget(self.components_check)
        components_layout.addStretch(1)
        components_container.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        components_container.setMaximumHeight(components_container.sizeHint().height())

        table_container = QtWidgets.QWidget(splitter)
        table_layout = QtWidgets.QVBoxLayout(table_container)
        table_layout.setContentsMargins(0, 0, 0, 0)

        self.param_model: _ParameterTableModel = _ParameterTableModel(
            self._params, self
        )
        self.param_model.sigParamsChanged.connect(self._update_fit_curve)
        self.param_model.sigParamsChanged.connect(self._refresh_slider_from_model)
        self.param_model.sigParamsChanged.connect(self._mark_fit_stale)

        self.param_view = QtWidgets.QTableView()
        self.param_view.setModel(self.param_model)
        self.param_view.setCornerButtonEnabled(False)
        vert_header = typing.cast(
            "QtWidgets.QHeaderView", self.param_view.verticalHeader()
        )
        horiz_header = typing.cast(
            "QtWidgets.QHeaderView", self.param_view.horizontalHeader()
        )
        vert_header.setVisible(False)
        self.param_view.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.param_view.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.param_view.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
            | QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed
            | QtWidgets.QAbstractItemView.EditTrigger.SelectedClicked
        )
        horiz_header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        self._schedule_table_width_init()
        value_delegate = _ParameterEditDelegate(self.param_model, self.param_view)
        for col in (1, 3, 4):
            self.param_view.setItemDelegateForColumn(col, value_delegate)
        self.param_view.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.param_view.resizeColumnsToContents()
        table_layout.addWidget(self.param_view, stretch=1)

        self.guess_button = QtWidgets.QPushButton("Guess")
        self.guess_button.clicked.connect(self._guess_params)
        self.fit_button = QtWidgets.QPushButton("Fit")
        self.fit_button.clicked.connect(self._run_fit)
        self.fit_multi_button = QtWidgets.QPushButton("Fit ×20")
        self.fit_multi_button.clicked.connect(lambda: self._run_fit_multiple(20))
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.clicked.connect(self._reset_params)
        self.copy_button = QtWidgets.QPushButton("Copy code")
        self.copy_button.clicked.connect(self.copy_code)
        self.copy_button.setEnabled(False)
        self.save_button = QtWidgets.QPushButton("Save fit")
        self.save_button.clicked.connect(self._save_fit)
        self.save_button.setEnabled(False)
        if self._model_is_default:
            self.multipeak_group = QtWidgets.QGroupBox("MultiPeakModel options")
            multipeak_layout = QtWidgets.QGridLayout(self.multipeak_group)

            self.npeaks_label = QtWidgets.QLabel("Peaks")
            self.npeaks_spin = QtWidgets.QSpinBox()
            self.npeaks_spin.setRange(1, 20)
            self.npeaks_spin.setValue(self._model.func.npeaks)

            self.peak_shape_label = QtWidgets.QLabel("Peak shape")
            self.peak_shape_combo = QtWidgets.QComboBox()
            self.peak_shape_combo.addItems(["lorentzian", "gaussian"])
            self.peak_shape_combo.setCurrentText(self._model.func._peak_shapes[0])

            self.fd_check = QtWidgets.QCheckBox("Fermi-Dirac")
            self.fd_check.setChecked(self._model.func.fd)

            self.background_label = QtWidgets.QLabel("Background")
            self.background_combo = QtWidgets.QComboBox()
            self.background_combo.addItems(["none", "constant", "linear", "polynomial"])
            self.background_combo.setCurrentText(self._model.func.background)

            self.degree_label = QtWidgets.QLabel("Degree")
            self.degree_spin = QtWidgets.QSpinBox()
            self.degree_spin.setRange(0, 10)
            degree = (
                self._model.func.bkg_degree
                if self._model.func.background == "polynomial"
                else 2
            )
            self.degree_spin.setValue(degree)
            self._set_degree_visibility(
                self.background_combo.currentText() == "polynomial"
            )

            self.convolve_check = QtWidgets.QCheckBox("Convolve")
            self.convolve_check.setChecked(self._model.func.convolve)

            multipeak_layout.addWidget(self.npeaks_label, 0, 0)
            multipeak_layout.addWidget(self.npeaks_spin, 0, 1)
            multipeak_layout.addWidget(self.peak_shape_label, 1, 0)
            multipeak_layout.addWidget(self.peak_shape_combo, 1, 1)
            multipeak_layout.addWidget(self.background_label, 2, 0)
            multipeak_layout.addWidget(self.background_combo, 2, 1)
            multipeak_layout.addWidget(self.degree_label, 3, 0)
            multipeak_layout.addWidget(self.degree_spin, 3, 1)
            multipeak_layout.addWidget(self.fd_check, 4, 0)
            multipeak_layout.addWidget(self.convolve_check, 4, 1)

            self.npeaks_spin.valueChanged.connect(self._refresh_multipeak_model)
            self.peak_shape_combo.currentTextChanged.connect(
                self._refresh_multipeak_model
            )
            self.fd_check.toggled.connect(self._refresh_multipeak_model)
            self.background_combo.currentTextChanged.connect(self._on_bkg_changed)
            self.degree_spin.valueChanged.connect(self._refresh_multipeak_model)
            self.convolve_check.toggled.connect(self._refresh_multipeak_model)

            right_layout.addWidget(self.multipeak_group)

        x_vals = self._x_values()
        x_min, x_max = float(np.nanmin(x_vals)), float(np.nanmax(x_vals))
        x_decimals = erlab.utils.array.effective_decimals(x_vals)
        self.domain_group = QtWidgets.QGroupBox("Fit domain")
        domain_layout = QtWidgets.QHBoxLayout(self.domain_group)
        self.domain_min_label = QtWidgets.QLabel("Min")
        self.domain_max_label = QtWidgets.QLabel("Max")
        self.domain_min_spin: QtWidgets.QDoubleSpinBox = QtWidgets.QDoubleSpinBox()
        self.domain_max_spin: QtWidgets.QDoubleSpinBox = QtWidgets.QDoubleSpinBox()
        for spin in (self.domain_min_spin, self.domain_max_spin):
            spin.setRange(x_min, x_max)
            spin.setDecimals(x_decimals)
            spin.setSingleStep(10**-x_decimals if x_decimals > 0 else 1.0)
        self.domain_min_spin.setValue(x_min)
        self.domain_max_spin.setValue(x_max)
        self.domain_min_spin.valueChanged.connect(self._sync_lines_from_domain)
        self.domain_max_spin.valueChanged.connect(self._sync_lines_from_domain)
        domain_layout.addWidget(self.domain_min_label)
        domain_layout.addWidget(self.domain_min_spin)
        domain_layout.addStretch(1)
        domain_layout.addWidget(self.domain_max_label)
        domain_layout.addWidget(self.domain_max_spin)
        right_layout.addWidget(self.domain_group)

        self.slider_group = QtWidgets.QGroupBox("Parameter")
        slider_layout = QtWidgets.QGridLayout(self.slider_group)
        slider_layout.setContentsMargins(8, 8, 8, 8)

        self.slider_expr_label = QtWidgets.QLabel("")
        self.slider_expr_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )

        self.slider_width_label = QtWidgets.QLabel("Range")
        self.slider_width_spin = pg.SpinBox(dec=True, compactHeight=False, finite=False)
        self.slider_width_spin.setOpts(decimals=6, step=0.1)

        self.slider_value_label = QtWidgets.QLabel("Value")
        self.slider_value_spin = pg.SpinBox(dec=True, compactHeight=False, finite=False)
        self.slider_value_spin.setOpts(decimals=6, step=0.1)

        self.value_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.value_slider.setRange(0, self._slider_steps)
        self.slider_width_label.setToolTip("The range of the slider.")
        self.slider_width_spin.setToolTip("The range of the slider.")
        self.slider_value_label.setToolTip("Parameter value used for the fit.")
        self.slider_value_spin.setToolTip("Parameter value used for the fit.")
        self.value_slider.setToolTip("Drag to adjust the parameter value.")

        slider_layout.addWidget(self.slider_expr_label, 0, 0, 1, 2)
        slider_layout.addWidget(self.slider_value_label, 1, 0)
        slider_layout.addWidget(self.slider_value_spin, 1, 1)
        slider_layout.addWidget(self.value_slider, 2, 0, 1, 2)
        slider_layout.addWidget(self.slider_width_label, 3, 0)
        slider_layout.addWidget(self.slider_width_spin, 3, 1)

        fit_group = QtWidgets.QGroupBox("Fit options")
        fit_layout = QtWidgets.QFormLayout(fit_group)

        self.timeout_spin = QtWidgets.QDoubleSpinBox()
        self.timeout_spin.setRange(0.1, 1e6)
        self.timeout_spin.setDecimals(2)
        self.timeout_spin.setValue(2.0)
        self.timeout_spin.setSingleStep(1.0)
        self.timeout_spin.setToolTip("Timeout for the fit evaluation.")

        self.nfev_spin: QtWidgets.QSpinBox = QtWidgets.QSpinBox()
        self.nfev_spin.setRange(1, 10_000_000)
        self.nfev_spin.setValue(100)
        self.nfev_spin.setSingleStep(100)
        self.nfev_spin.setToolTip("Maximum number of function evaluations.")

        self.method_combo: QtWidgets.QComboBox = QtWidgets.QComboBox()
        self.method_combo.addItems(
            [
                "leastsq",
                "least_squares",
                "nelder",
                "powell",
                "cobyla",
                "trust-constr",
            ]
        )
        self.method_combo.setCurrentText("least_squares")

        self.fit_buttons = QtWidgets.QGridLayout()
        self.fit_buttons.setContentsMargins(0, 0, 0, 0)
        self.fit_buttons.addWidget(self.guess_button, 0, 0)
        self.fit_buttons.addWidget(self.reset_button, 0, 1)
        self.fit_buttons.addWidget(self.fit_button, 1, 0)
        self.fit_buttons.addWidget(self.fit_multi_button, 1, 1)

        fit_layout.addRow("Timeout (s)", self.timeout_spin)
        fit_layout.addRow("Max nfev", self.nfev_spin)
        fit_layout.addRow("Method", self.method_combo)
        fit_layout.addRow(self.fit_buttons)
        right_layout.addWidget(fit_group)

        stats_group = QtWidgets.QGroupBox("Fit stats")
        stats_layout = QtWidgets.QGridLayout(stats_group)

        self.elapsed_label = QtWidgets.QLabel("Elapsed")
        self.elapsed_value = QtWidgets.QLabel("—")
        self.nfev_out_label = QtWidgets.QLabel("nfev")
        self.nfev_out_value = QtWidgets.QLabel("—")
        self.redchi_label = QtWidgets.QLabel("redchi")
        self.redchi_value = QtWidgets.QLabel("—")
        self.rsq_label = QtWidgets.QLabel("R²")
        self.rsq_value = QtWidgets.QLabel("—")
        self.aic_label = QtWidgets.QLabel("AIC")
        self.aic_value = QtWidgets.QLabel("—")
        self.bic_label = QtWidgets.QLabel("BIC")
        self.bic_value = QtWidgets.QLabel("—")

        self.elapsed_label.setToolTip("Wall-clock time for the last fit run.")
        self.elapsed_value.setToolTip("Wall-clock time for the last fit run.")
        self.nfev_out_label.setToolTip("Number of function evaluations used.")
        self.nfev_out_value.setToolTip("Number of function evaluations used.")
        self.redchi_label.setToolTip(
            "Reduced chi-square (chi-square per degree of freedom)."
        )
        self.redchi_value.setToolTip(
            "Reduced chi-square (chi-square per degree of freedom)."
        )
        self.rsq_label.setToolTip("Coefficient of determination (R²).")
        self.rsq_value.setToolTip("Coefficient of determination (R²).")
        self.aic_label.setToolTip("Akaike Information Criterion; lower is better.")
        self.aic_value.setToolTip("Akaike Information Criterion; lower is better.")
        self.bic_label.setToolTip("Bayesian Information Criterion; lower is better.")
        self.bic_value.setToolTip("Bayesian Information Criterion; lower is better.")

        stats_layout.addWidget(self.elapsed_label, 0, 0)
        stats_layout.addWidget(self.elapsed_value, 0, 1)
        stats_layout.addWidget(self.nfev_out_label, 0, 2)
        stats_layout.addWidget(self.nfev_out_value, 0, 3)
        stats_layout.addWidget(self.redchi_label, 1, 0)
        stats_layout.addWidget(self.redchi_value, 1, 1)
        stats_layout.addWidget(self.rsq_label, 1, 2)
        stats_layout.addWidget(self.rsq_value, 1, 3)
        stats_layout.addWidget(self.aic_label, 2, 0)
        stats_layout.addWidget(self.aic_value, 2, 1)
        stats_layout.addWidget(self.bic_label, 2, 2)
        stats_layout.addWidget(self.bic_value, 2, 3)
        right_layout.addWidget(stats_group)
        right_layout.addWidget(self.slider_group)

        copy_row = QtWidgets.QHBoxLayout()
        copy_row.addWidget(self.copy_button)
        copy_row.addWidget(self.save_button)
        right_layout.addStretch(1)
        right_layout.addLayout(copy_row)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        selection_model = typing.cast(
            "QtCore.QItemSelectionModel", self.param_view.selectionModel()
        )
        selection_model.currentChanged.connect(self._on_param_selected)

        self.slider_width_spin.valueChanged.connect(self._on_slider_width_changed)
        self.slider_value_spin.valueChanged.connect(self._on_slider_value_changed)
        self.value_slider.valueChanged.connect(self._on_slider_moved)
        self.value_slider.sliderPressed.connect(self._on_slider_pressed)
        self.value_slider.sliderReleased.connect(self._on_slider_released)

        self._populate_data_curve()
        if self.param_model.rowCount() > 0:
            self.param_view.selectRow(0)

    @property
    def tool_status(self) -> StateModel:
        return self.StateModel(
            data_name=self._data_name,
            model_name=str(self._argnames["model"]),
            model_is_default=self._model_is_default,
            multipeak=(
                _MultiPeakState(
                    npeaks=self.npeaks_spin.value(),
                    peak_shape=self.peak_shape_combo.currentText(),
                    fd=self.fd_check.isChecked(),
                    background=self.background_combo.currentText(),
                    degree=self.degree_spin.value(),
                    convolve=self.convolve_check.isChecked(),
                )
                if self._model_is_default
                else None
            ),
            domain=self._fit_domain(),
            show_components=self.components_check.isChecked(),
            timeout=self.timeout_spin.value(),
            max_nfev=self.nfev_spin.value(),
            method=self.method_combo.currentText(),
            slider_widths=dict(self._slider_widths),
            params=self._parameter_values(),
        )

    @tool_status.setter
    def tool_status(self, status: StateModel) -> None:
        self._data_name = status.data_name
        self._argnames["data"] = status.data_name
        self._argnames["model"] = status.model_name

        if self._model_is_default and status.model_is_default and status.multipeak:
            with (
                QtCore.QSignalBlocker(self.npeaks_spin),
                QtCore.QSignalBlocker(self.peak_shape_combo),
                QtCore.QSignalBlocker(self.fd_check),
                QtCore.QSignalBlocker(self.background_combo),
                QtCore.QSignalBlocker(self.degree_spin),
                QtCore.QSignalBlocker(self.convolve_check),
            ):
                self.npeaks_spin.setValue(status.multipeak.npeaks)
                self.peak_shape_combo.setCurrentText(status.multipeak.peak_shape)
                self.fd_check.setChecked(status.multipeak.fd)
                self.background_combo.setCurrentText(status.multipeak.background)
                self.degree_spin.setValue(status.multipeak.degree)
                self.convolve_check.setChecked(status.multipeak.convolve)
            self._set_degree_visibility(
                self.background_combo.currentText() == "polynomial"
            )
            self._refresh_multipeak_model()

        self.components_check.setChecked(status.show_components)
        self.timeout_spin.setValue(status.timeout)
        self.nfev_spin.setValue(status.max_nfev)
        self.method_combo.setCurrentText(status.method)

        self._slider_widths = dict(status.slider_widths)

        if status.params:
            params = self._model.make_params()
            for name, entry in status.params.items():
                if name not in params:
                    continue
                param = params[name]
                value = entry.get("value", param.value)
                value = param.value if not value else float(value)
                mn = entry.get("min", param.min)
                mn = param.min if mn is None else float(mn)
                mx = entry.get("max", param.max)
                mx = param.max if mx is None else float(mx)
                param.set(
                    value=value,
                    min=mn,
                    max=mx,
                    vary=bool(entry.get("vary", param.vary)),
                )
            self._params = params
            self._initial_params = params.copy()
            self.param_model.set_params(self._params)
            self._refresh_slider_from_model()
            self._mark_fit_stale()

        if status.domain is not None:
            with (
                QtCore.QSignalBlocker(self.domain_min_spin),
                QtCore.QSignalBlocker(self.domain_max_spin),
            ):
                self.domain_min_spin.setValue(status.domain[0])
                self.domain_max_spin.setValue(status.domain[1])
            self._sync_lines_from_domain()
        else:
            self._sync_lines_from_domain()

    def _schedule_table_width_init(self) -> None:
        if self._table_widths_initialized:
            return
        QtCore.QTimer.singleShot(0, self._init_table_column_widths)

    def _init_table_column_widths(self) -> None:
        if self._table_widths_initialized:
            return
        self.param_view.resizeColumnsToContents()
        viewport = self.param_view.viewport()
        if viewport:  # pragma: no branch
            viewport_width = viewport.width()
            if viewport_width <= 0:
                self._schedule_table_width_init()
                return
        total = 0
        for col in range(self.param_model.columnCount()):
            total += self.param_view.columnWidth(col)
        stderr_col = 2
        stderr_min = max(self.param_view.sizeHintForColumn(stderr_col), 90)
        if self.param_view.columnWidth(stderr_col) < stderr_min:
            total += stderr_min - self.param_view.columnWidth(stderr_col)
            self.param_view.setColumnWidth(stderr_col, stderr_min)
        value_col = 1
        delta = viewport_width - total
        if delta != 0:
            min_width = self.param_view.sizeHintForColumn(value_col)

            target = max(self.param_view.columnWidth(value_col) + delta, min_width)

            style = self.param_view.style()
            if style:  # pragma: no branch
                target = target - style.pixelMetric(
                    QtWidgets.QStyle.PixelMetric.PM_ScrollBarExtent
                )
            self.param_view.setColumnWidth(value_col, target)
        self._table_widths_initialized = True

    def _domain_brushes(self, xvals: np.ndarray) -> list[QtGui.QBrush] | None:
        if xvals.size == 0:
            return None
        domain = self._fit_domain()
        if domain is None:
            mask = np.isfinite(xvals)
        else:
            lo, hi = domain
            mask = np.isfinite(xvals) & (xvals >= lo) & (xvals <= hi)
        base = pg.mkBrush(30, 30, 30)
        tint_color = pg.mkColor(self.FIT_COLOR)
        tint_color.setAlpha(50)
        tint = pg.mkBrush(tint_color)
        return [tint if inside else base for inside in mask]

    def _update_domain_brushes(self) -> None:
        xvals = self._x_values()
        brushes = self._domain_brushes(xvals)
        if brushes is None:
            return
        self.data_curve.setData(xvals, self._data.values, symbolBrush=brushes)
        if self._last_residual is not None:
            self.residual_curve.setData(xvals, self._last_residual, symbolBrush=brushes)

    def _populate_data_curve(self) -> None:
        self._update_domain_brushes()
        self.main_plot.enableAutoRange()

    def _x_values(self) -> np.ndarray:
        if self._coord_name in self._data.coords:
            coords = self._data.coords[self._coord_name]
            return np.asarray(coords.values)
        return np.arange(self._data.size, dtype=float)

    def _sync_domain_from_lines(self) -> None:
        if not hasattr(self, "fit_line_min"):
            return
        x0 = float(self.fit_line_min.value())
        x1 = float(self.fit_line_max.value())
        if not (np.isfinite(x0) and np.isfinite(x1)):
            return
        x_vals = self._x_values()
        x_min, x_max = float(np.nanmin(x_vals)), float(np.nanmax(x_vals))
        lo, hi = sorted((x0, x1))
        lo = float(np.clip(lo, x_min, x_max))
        hi = float(np.clip(hi, x_min, x_max))
        with (
            QtCore.QSignalBlocker(self.fit_line_min),
            QtCore.QSignalBlocker(self.fit_line_max),
            QtCore.QSignalBlocker(self.domain_min_spin),
            QtCore.QSignalBlocker(self.domain_max_spin),
        ):
            if not np.isclose(self.fit_line_min.value(), lo):
                self.fit_line_min.setValue(lo)
            if not np.isclose(self.fit_line_max.value(), hi):
                self.fit_line_max.setValue(hi)
            self.domain_min_spin.setValue(lo)
            self.domain_max_spin.setValue(hi)
        self._update_domain_brushes()

    def _sync_lines_from_domain(self) -> None:
        if not hasattr(self, "fit_line_min"):
            return
        x0 = float(self.domain_min_spin.value())
        x1 = float(self.domain_max_spin.value())
        if not (np.isfinite(x0) and np.isfinite(x1)):
            return
        x_vals = self._x_values()
        x_min, x_max = float(np.nanmin(x_vals)), float(np.nanmax(x_vals))
        lo, hi = sorted((x0, x1))
        lo = float(np.clip(lo, x_min, x_max))
        hi = float(np.clip(hi, x_min, x_max))
        with (
            QtCore.QSignalBlocker(self.fit_line_min),
            QtCore.QSignalBlocker(self.fit_line_max),
            QtCore.QSignalBlocker(self.domain_min_spin),
            QtCore.QSignalBlocker(self.domain_max_spin),
        ):
            if not np.isclose(self.domain_min_spin.value(), lo):
                self.domain_min_spin.setValue(lo)
            if not np.isclose(self.domain_max_spin.value(), hi):
                self.domain_max_spin.setValue(hi)
            self.fit_line_min.setValue(lo)
            self.fit_line_max.setValue(hi)
        self._update_domain_brushes()

    def _fit_domain(self) -> tuple[float, float] | None:
        if not hasattr(self, "fit_line_min"):
            return None
        x0 = float(self.fit_line_min.value())
        x1 = float(self.fit_line_max.value())
        x_vals = self._x_values()
        x_min, x_max = float(np.nanmin(x_vals)), float(np.nanmax(x_vals))
        x0, x1 = sorted((float(x0), float(x1)))
        if np.isclose(x0, x_min) and np.isclose(x1, x_max):
            return None
        x_decimals = erlab.utils.array.effective_decimals(x_vals)
        return round(x0, x_decimals), round(x1, x_decimals)

    def _fit_data(self) -> xr.DataArray:
        domain = self._fit_domain()
        if domain is None:
            return self._data
        try:
            return self._data.sel({self._coord_name: slice(*domain)})
        except Exception:
            coords = np.asarray(self._data[self._coord_name].values)
            mask = (coords >= domain[0]) & (coords <= domain[1])
            return self._data.isel({self._coord_name: mask})

    def _guess_params(self) -> None:
        if not hasattr(self._model, "guess"):
            self._show_warning("Guess not supported", "Model does not support guess().")
            return
        try:
            fit_data = self._fit_data()
            if self._coord_name in fit_data.coords:
                x_vals = np.asarray(fit_data.coords[self._coord_name].values)
            else:
                x_vals = np.arange(fit_data.size, dtype=float)
            params = self._model.guess(
                fit_data.values, **{self._independent_var(): x_vals}
            )
        except Exception:  # pragma: no cover - GUI feedback
            self._show_error("Guess failed", "Failed to estimate initial parameters.")
            return
        self._params = params
        self._initial_params = self._params.copy()
        self.param_model.set_params(self._params)
        self._mark_fit_stale()

    def _reset_params(self) -> None:
        if self._initial_params is None:
            self._params = None
            self.param_model.set_params(None)
        else:
            self._params = self._initial_params.copy()
            self.param_model.set_params(self._params)
        self._set_fit_stats(None)
        self._mark_fit_stale()

    def _independent_var(self) -> str:
        if self._model.independent_vars:
            return str(self._model.independent_vars[0])
        return self._coord_name

    def _update_fit_curve(self) -> None:
        xvals = self._x_values()
        if self._params is None:
            self.fit_curve.setData([], [])
            self.residual_curve.setData([], [])
            self._last_fit_y = None
            self._last_residual = None
            self._update_component_curves(np.array([]))
            return
        if self._has_non_finite_params():
            self.fit_curve.setData([], [])
            self.residual_curve.setData([], [])
            self._last_fit_y = None
            self._last_residual = None
            self._update_component_curves(np.array([]))
            return

        x_min = float(np.nanmin(xvals)) if xvals.size else 0.0
        x_max = float(np.nanmax(xvals)) if xvals.size else 0.0
        fine_count = max(int(xvals.size) * self.PLOT_RESAMPLE, 2)
        x_fine = np.linspace(x_min, x_max, fine_count, dtype=float)
        if "y" in self._model.independent_vars:
            x_fine = xvals
        try:
            indep_var_kwargs = {self._independent_var(): x_fine}
            if "y" in self._model.independent_vars:
                indep_var_kwargs["y"] = self._data.values
            y_fine = self._model.eval(params=self._params, **indep_var_kwargs)
        except Exception:  # pragma: no cover - GUI feedback
            self._show_error("Evaluation failed", "Failed to evaluate model.")
            return
        self.fit_curve.setData(x_fine, y_fine)
        residuals = self._residuals_from_result(xvals)
        brushes = self._domain_brushes(xvals)
        if brushes is None:
            self.residual_curve.setData(xvals, residuals)
        else:
            self.residual_curve.setData(xvals, residuals, symbolBrush=brushes)
        self._last_fit_y = y_fine
        self._last_residual = residuals
        self._update_component_curves(x_fine)

    def _on_bkg_changed(self, value: str) -> None:
        self._set_degree_visibility(value == "polynomial")
        self._refresh_multipeak_model()

    def _refresh_multipeak_model(self) -> None:
        if not self._model_is_default:
            return
        prev_params = self._params
        prev_widths = dict(self._slider_widths)
        self._model = erlab.analysis.fit.models.MultiPeakModel(
            npeaks=self.npeaks_spin.value(),
            peak_shapes=typing.cast(
                "typing.Literal['lorentzian', 'gaussian']",
                self.peak_shape_combo.currentText(),
            ),
            fd=self.fd_check.isChecked(),
            background=typing.cast(
                "typing.Literal['none', 'constant', 'linear', 'polynomial']",
                self.background_combo.currentText(),
            ),
            degree=self.degree_spin.value(),
            convolve=self.convolve_check.isChecked(),
            segmented=self._auto_segmented(self.convolve_check.isChecked()),
        )
        self._params = self._model.make_params()
        if prev_params is not None:
            self._merge_params(prev_params, self._params)
        self._slider_widths = {
            name: width for name, width in prev_widths.items() if name in self._params
        }
        self._initial_params = self._params.copy()
        self.param_model.set_params(self._params)
        self._update_fit_curve()
        self._mark_fit_stale()

    @staticmethod
    def _merge_params(
        old_params: lmfit.Parameters, new_params: lmfit.Parameters
    ) -> None:
        for name in new_params:
            if name not in old_params:
                continue
            old_param = old_params[name]
            new_param = new_params[name]
            if new_param.expr:
                continue
            new_param.set(
                value=old_param.value,
                min=old_param.min,
                max=old_param.max,
                vary=old_param.vary,
            )

    def _auto_segmented(self, convolve: bool) -> bool:
        if not convolve:
            return False
        xvals = self._x_values()
        if xvals.size < 3:
            return False
        diffs = np.diff(xvals)
        return not np.allclose(diffs, diffs[0])

    def _set_degree_visibility(self, visible: bool) -> None:
        self.degree_label.setVisible(visible)
        self.degree_spin.setVisible(visible)

    def _update_component_curves(self, xvals: np.ndarray) -> None:
        if not self.components_check.isChecked():
            for curve in self.component_curves.values():
                curve.hide()
            self._remove_component_legend(self.component_curves.keys())
            return
        if self._params is None or xvals.size == 0:
            for curve in self.component_curves.values():
                curve.hide()
            self._remove_component_legend(self.component_curves.keys())
            return
        if not hasattr(self._model, "eval_components"):
            for curve in self.component_curves.values():
                curve.hide()
            self._remove_component_legend(self.component_curves.keys())
            return
        try:
            comps = self._model.eval_components(
                params=self._params, **{self._independent_var(): xvals}
            )
        except Exception:  # pragma: no cover - GUI feedback
            return
        pens = [
            pg.intColor(i, hues=max(len(comps), 1), sat=128) for i in range(len(comps))
        ]
        for idx, (name, values) in enumerate(comps.items()):
            if name not in self.component_curves:
                curve = self.main_plot.plot(pen=pg.mkPen(pens[idx], width=1), name=name)
                self.component_curves[name] = curve
            curve = self.component_curves[name]
            curve.setData(xvals, values)
            curve.show()
            if not self._legend_has_name(name):
                self.legend.addItem(curve, name)
        for name, curve in list(self.component_curves.items()):
            if name not in comps:
                curve.hide()
        self._remove_component_legend(
            name for name in self.component_curves if name not in comps
        )

    def _residuals_from_result(self, xvals: np.ndarray) -> np.ndarray:
        if (
            self._fit_is_current
            and self._last_result is not None
            and self._params is not None
            and self._params_match_result(self._last_result, self._params)
        ):
            best_fit = getattr(self._last_result, "best_fit", None)
            if best_fit is not None and best_fit.size == self._data.size:
                return self._data.values - best_fit
        indep_var_kwargs = {self._independent_var(): xvals}
        if "y" in self._model.independent_vars:
            indep_var_kwargs["y"] = self._data.values
        yvals = self._model.eval(params=self._params, **indep_var_kwargs)
        return self._data.values - yvals

    @staticmethod
    def _params_match_result(
        result: lmfit.model.ModelResult, params: lmfit.Parameters
    ) -> bool:
        for name, param in params.items():
            if name not in result.params:
                return False
            if not np.isclose(param.value, result.params[name].value, rtol=0, atol=0):
                return False
        return True

    def _legend_has_name(self, name: str) -> bool:
        return any(item[1].text == name for item in self.legend.items)

    def _remove_component_legend(self, names: typing.Iterable[str]) -> None:
        for name in names:
            if self._legend_has_name(name):
                self.legend.removeItem(name)

    def _set_fit_ds(self, result_ds: xr.Dataset, t0: float) -> lmfit.Parameters:
        self._last_result_ds = result_ds.copy()
        result = self._last_result_ds.modelfit_results.compute().item()
        self._last_result = result
        self._params = self._last_result.params.copy()
        self.param_model.set_params(self._params)
        self._update_fit_curve()
        self._refresh_slider_from_model()
        elapsed = time.perf_counter() - t0
        self._set_fit_stats(result, elapsed=elapsed)
        self._mark_fit_fresh()
        self.sigFitExecuted.emit(self._params.copy())

        viewport = self.param_view.viewport()
        if viewport:  # pragma: no branch
            viewport.update()
        return self._params

    def _run_fit(self) -> None:
        if self._fit_future is not None and not self._fit_future.done():
            self._show_warning("Fit running", "A fit is already running.")
            return
        if self._params is None:
            self._show_warning("No parameters", "Parameters are not initialized.")
            return
        self._set_fit_running(True, multi=False)
        self._set_fit_stats(None)
        t0 = time.perf_counter()
        try:
            fit_data = self._fit_data()
            self._fit_future = self._executor.submit(
                fit_data.xlm.modelfit,
                self._coord_name,
                model=self._model,
                params=self._params,
                max_nfev=self.nfev_spin.value(),
                method=self.method_combo.currentText(),
            )
            result_ds = self._fit_future.result(timeout=self.timeout_spin.value())
        except TimeoutError:
            if self._fit_future is not None:
                self._fit_future.cancel()
            elapsed = time.perf_counter() - t0
            self._show_error(
                "Fit timed out",
                f"Fit timed out in {elapsed:.2f} s.",
            )
            self._set_fit_stats(None)
            self._set_elapsed_status(elapsed, timed_out=True)
            self._mark_fit_stale()
            self._set_fit_running(False, multi=False)
            return
        except Exception:  # pragma: no cover - GUI feedback
            self._show_error("Fit failed", "Fit failed to complete.")
            self._set_fit_stats(None)
            self._mark_fit_stale()
            self._set_fit_running(False, multi=False)
            return

        self._set_fit_ds(result_ds, t0)
        self._set_fit_running(False, multi=False)

    def _run_fit_multiple(self, count: int) -> None:
        if self._fit_future is not None and not self._fit_future.done():
            self._show_warning("Fit running", "A fit is already running.")
            return
        if self._params is None:
            self._show_warning("No parameters", "Parameters are not initialized.")
            return
        self._set_fit_running(True, multi=True, step=0, total=count)
        self._set_fit_stats(None)
        fit_data = self._fit_data()
        params = self._params
        result_ds: xr.Dataset | None = None
        try:
            for idx in range(count):
                self._set_fit_running(True, multi=True, step=idx + 1, total=count)
                step_t0 = time.perf_counter()
                try:
                    self._fit_future = self._executor.submit(
                        fit_data.xlm.modelfit,
                        self._coord_name,
                        model=self._model,
                        params=params,
                        max_nfev=self.nfev_spin.value(),
                        method=self.method_combo.currentText(),
                    )
                    result_ds = self._fit_future.result(
                        timeout=self.timeout_spin.value()
                    )
                except TimeoutError:
                    if self._fit_future is not None:
                        self._fit_future.cancel()
                    elapsed = time.perf_counter() - step_t0
                    self._show_error(
                        "Fit timed out",
                        f"Fit timed out in {elapsed:.2f} s.",
                    )
                    self._set_fit_stats(None)
                    self._set_elapsed_status(elapsed, timed_out=True)
                    self._mark_fit_stale()
                    return
                except Exception:  # pragma: no cover - GUI feedback
                    self._show_error("Fit failed", "Fit failed to complete.")
                    self._set_fit_stats(None)
                    self._mark_fit_stale()
                    return

                params = self._set_fit_ds(result_ds, step_t0)
                QtWidgets.QApplication.processEvents()
        finally:
            self._set_fit_running(False, multi=True)

    def _set_fit_running(
        self, running: bool, *, multi: bool, step: int = 0, total: int = 0
    ) -> None:
        if running:
            self.fit_button.setEnabled(False)
            self.fit_multi_button.setEnabled(False)
            self.fit_button.setText("Fitting...")
            if multi and total > 0:
                self.fit_multi_button.setText(f"Fit {step}/{total}")
            else:
                self.fit_multi_button.setText("Fit ×20")
        else:
            self.fit_button.setEnabled(True)
            self.fit_multi_button.setEnabled(True)
            self.fit_button.setText("Fit")
            self.fit_multi_button.setText("Fit ×20")

    def _parameter_values(self) -> dict[str, dict[str, float | bool]]:
        """Return current parameter values and bounds for serialization."""
        values: dict[str, dict[str, float | bool]] = {}
        if self._params is None:
            return values
        for name in self._params:
            param = self._params[name]
            values[name] = {
                "value": float(param.value),
                "min": float(param.min),
                "max": float(param.max),
                "vary": bool(param.vary),
            }
        return values

    @QtCore.Slot()
    def copy_code(self) -> str:
        data_name = str(self._argnames["data"])
        if not data_name.isidentifier():
            data_name = "data"
        model_name = str(self._argnames["model"])
        if not model_name.isidentifier():
            model_name = "model"

        lines: list[str] = []
        if self._model_is_default:
            lines.append(
                erlab.interactive.utils.generate_code(
                    erlab.analysis.fit.models.MultiPeakModel,
                    args=[],
                    kwargs={
                        "npeaks": self.npeaks_spin.value(),
                        "peak_shapes": self.peak_shape_combo.currentText(),
                        "fd": self.fd_check.isChecked(),
                        "background": self.background_combo.currentText(),
                        "degree": self.degree_spin.value(),
                        "convolve": self.convolve_check.isChecked(),
                        "segmented": self._auto_segmented(
                            self.convolve_check.isChecked()
                        ),
                    },
                    module="era.fit.models",
                    assign=model_name,
                )
            )

        fit_domain = self._fit_domain()
        data_var = data_name
        if fit_domain is not None:
            sel_kw = erlab.interactive.utils.format_kwargs(
                {self._coord_name: slice(*fit_domain)}
            )
            lines.append(f"{data_name}_fit = {data_name}.sel({sel_kw})")
            data_var = f"{data_name}_fit"

        if self._params is None:
            lines.append(f"params = {model_name}.make_params()")
            lines.append(
                f'result = {data_var}.xlm.modelfit("{self._coord_name}", '
                f"model={model_name}, params=params, "
                f"max_nfev={self.nfev_spin.value()}, "
                f'method="{self.method_combo.currentText()}")'
            )
            return erlab.interactive.utils.copy_to_clipboard(lines)

        param_entries: list[str] = []
        param_kwargs: dict[str, typing.Any] = {}
        needs_dict = False
        for name in self._params:
            param = self._params[name]
            entry_kwargs: dict[str, typing.Any] = {}
            if param.expr:
                continue
            entry_kwargs["value"] = param.value
            if np.isfinite(param.min):
                entry_kwargs["min"] = param.min
            if np.isfinite(param.max):
                entry_kwargs["max"] = param.max
            if not param.vary:
                entry_kwargs["vary"] = False

            entry_value: typing.Any = entry_kwargs if entry_kwargs else param.value
            param_kwargs[name] = entry_value
            if not name.isidentifier() or needs_dict:
                needs_dict = True
                continue
            if entry_kwargs:
                param_entries.append(f"{name}={entry_kwargs!r}")
            else:
                param_entries.append(f"{name}={param.value!r}")

        if needs_dict:
            lines.append(f"params = {model_name}.make_params(**{param_kwargs!r})")
        else:
            joined = ",\n    ".join(param_entries)
            if joined:
                lines.append(f"params = {model_name}.make_params(\n    {joined},\n)")
            else:
                lines.append(f"params = {model_name}.make_params()")
        lines.append(
            erlab.interactive.utils.generate_code(
                self._data.xlm.modelfit,
                args=[self._coord_name],
                kwargs={
                    "model": f"|{model_name}|",
                    "params": "|params|",
                    "max_nfev": self.nfev_spin.value(),
                    "method": self.method_combo.currentText(),
                },
                name="modelfit",
                module=f"{data_var}.xlm",
                assign="result",
            )
        )
        return erlab.interactive.utils.copy_to_clipboard(lines)

    @QtCore.Slot()
    def _save_fit(self) -> None:
        if self._last_result_ds is None:
            self._show_warning("No fit result", "There is no fit result to save.")
            return
        erlab.interactive.utils.save_fit_ui(self._last_result_ds, parent=self)

    def _on_param_selected(
        self, current: QtCore.QModelIndex, previous: QtCore.QModelIndex
    ) -> None:
        if not current.isValid() or self._params is None:
            self._current_row = None
            self.slider_group.setTitle("Parameter")
            self._show_slider_message("")
            return
        self._current_row = current.row()
        self._refresh_slider_from_model()

    def _set_slider_enabled(self, enabled: bool) -> None:
        for widget in (
            self.slider_width_spin,
            self.slider_value_spin,
            self.value_slider,
        ):
            widget.setEnabled(enabled)

    def _set_slider_widget_visibility(self, visible: bool) -> None:
        self.slider_width_label.setVisible(visible)
        self.slider_width_spin.setVisible(visible)
        self.slider_value_label.setVisible(visible)
        self.slider_value_spin.setVisible(visible)
        self.value_slider.setVisible(visible)
        self.slider_expr_label.setVisible(not visible)

    def _show_slider_message(self, message: str) -> None:
        self.slider_expr_label.setText(message)
        self._set_slider_enabled(False)
        self._set_slider_widget_visibility(False)

    def _refresh_slider_from_model(self) -> None:
        if self._current_row is None or self._params is None:
            self.slider_group.setTitle("Parameter")
            self._show_slider_message("")
            return
        param = self.param_model.param_at(self._current_row)
        if self._slider_dragging and not param.expr:
            self._slider_updating = True
            self.slider_value_spin.blockSignals(True)
            self.value_slider.blockSignals(True)
            self.slider_value_spin.setValue(param.value)
            if self._slider_drag_range is None:
                slider_min, slider_max, _ = self._slider_range(param.value, param)
                self._slider_drag_range = (slider_min, slider_max)
            else:
                slider_min, slider_max = self._slider_drag_range
            self._set_slider_position(param.value, slider_min, slider_max)
            self.value_slider.blockSignals(False)
            self.slider_value_spin.blockSignals(False)
            self._slider_updating = False
            return

        self.slider_group.setTitle(f"Parameter: {param.name}")

        if param.expr:
            self._show_slider_message(f"expr: {param.expr}")
            self._set_slider_values(param.value, None, None, None)
            return
        if not np.isfinite(param.value):
            self._show_slider_message(
                f"value: {_ParameterTableModel._format_value(param.value)}"
            )
            return

        self.slider_expr_label.setText("")
        self._set_slider_enabled(True)
        self._set_slider_widget_visibility(True)

        slider_min, slider_max, width = self._slider_range(param.value, param)
        self._set_slider_values(param.value, width, slider_min, slider_max)

    def _slider_range(
        self, value: float, param: lmfit.Parameter
    ) -> tuple[float, float, float]:
        width = self._slider_widths.get(param.name)
        if width is None or not np.isfinite(width) or width <= 0:
            if np.isfinite(param.min) and np.isfinite(param.max):
                width = float(param.max - param.min)
            else:
                width = float(self._default_slider_width(value))
            self._slider_widths[param.name] = width
        half_span = width / 2.0
        vmin = float(value - half_span)
        vmax = float(value + half_span)
        if np.isfinite(param.min):
            vmin = max(vmin, float(param.min))
        if np.isfinite(param.max):
            vmax = min(vmax, float(param.max))
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        return vmin, vmax, width

    def _default_slider_width(self, value: float) -> float:
        y_std = float(np.nanstd(self._data.values))
        x_std = float(np.nanstd(self._x_values()))
        base = max(y_std, x_std, 1.0)
        if np.isfinite(value):
            base = max(base, abs(value) * 0.5, 1.0)
        return base * 0.05

    def _set_slider_values(
        self,
        value: float,
        width: float | None,
        slider_min: float | None,
        slider_max: float | None,
    ) -> None:
        if not np.isfinite(value):
            return
        self._slider_updating = True
        try:
            if width is not None and np.isfinite(width):
                self.slider_width_spin.setValue(width)
            if slider_min is not None and slider_max is not None:
                if slider_min > slider_max:
                    slider_min, slider_max = slider_max, slider_min
                self.slider_value_spin.setRange(slider_min, slider_max)
                self._set_slider_position(value, slider_min, slider_max)
            self.slider_value_spin.setValue(value)
        finally:
            self._slider_updating = False

    def _set_slider_position(self, value: float, vmin: float, vmax: float) -> None:
        if not np.isfinite(vmin) or not np.isfinite(vmax) or not np.isfinite(value):
            self.value_slider.setEnabled(False)
            return
        if vmax <= vmin:
            self.value_slider.setEnabled(False)
            return
        self.value_slider.setEnabled(True)
        clamped = min(max(value, vmin), vmax)
        ratio = (clamped - vmin) / (vmax - vmin)
        self.value_slider.setValue(round(ratio * self._slider_steps))

    def _on_slider_moved(self, slider_value: int) -> None:
        if self._slider_updating or self._current_row is None:
            return
        param = self.param_model.param_at(self._current_row)
        if self._slider_dragging and self._slider_drag_range is not None:
            vmin, vmax = self._slider_drag_range
        else:
            vmin, vmax, _ = self._slider_range(self.slider_value_spin.value(), param)
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return
        if vmax <= vmin:
            return
        value = vmin + (vmax - vmin) * (slider_value / self._slider_steps)
        self._slider_updating = True
        self.slider_value_spin.blockSignals(True)
        self.slider_value_spin.setValue(value)
        self.slider_value_spin.blockSignals(False)
        self._slider_updating = False
        self._update_param_value(value)

    def _on_slider_value_changed(self, value: float) -> None:
        if self._slider_updating or self._current_row is None:
            return
        if not np.isfinite(value):
            self.value_slider.setEnabled(False)
            return
        param = self.param_model.param_at(self._current_row)
        self._update_param_value(value)
        vmin, vmax, _ = self._slider_range(value, param)
        self._slider_updating = True
        self.slider_value_spin.blockSignals(True)
        self.value_slider.blockSignals(True)
        self.slider_value_spin.setRange(vmin, vmax)
        self._set_slider_position(value, vmin, vmax)
        self.value_slider.blockSignals(False)
        self.slider_value_spin.blockSignals(False)
        self._slider_updating = False

    def _on_slider_width_changed(self) -> None:
        if self._slider_updating or self._current_row is None:
            return
        param = self.param_model.param_at(self._current_row)
        width = self.slider_width_spin.value()
        if not np.isfinite(width) or width <= 0:
            return
        self._slider_widths[param.name] = width
        vmin, vmax, _ = self._slider_range(self.slider_value_spin.value(), param)
        self._slider_updating = True
        self.slider_value_spin.blockSignals(True)
        self.value_slider.blockSignals(True)
        self.slider_value_spin.setRange(vmin, vmax)
        self._set_slider_position(self.slider_value_spin.value(), vmin, vmax)
        self.value_slider.blockSignals(False)
        self.slider_value_spin.blockSignals(False)
        self._slider_updating = False

    def _update_param_value(self, value: float) -> None:
        if self._current_row is None:
            return
        index = self.param_model.index(self._current_row, 1)
        self.param_model.setData(index, value, QtCore.Qt.ItemDataRole.EditRole)

    def _on_slider_pressed(self) -> None:
        self._slider_dragging = True
        if self._current_row is None or self._params is None:
            self._slider_drag_range = None
            return
        param = self.param_model.param_at(self._current_row)
        vmin, vmax, _ = self._slider_range(self.slider_value_spin.value(), param)
        self._slider_drag_range = (vmin, vmax)

    def _on_slider_released(self) -> None:
        self._slider_dragging = False
        self._slider_drag_range = None
        self._refresh_slider_from_model()

    def _set_fit_stats(
        self, result: lmfit.model.ModelResult | None, *, elapsed: float | None = None
    ) -> None:
        if result is None:
            self.elapsed_value.setText("—")
            self._set_elapsed_status(None, timed_out=False)
            self.nfev_out_value.setText("—")
            self.redchi_value.setText("—")
            self.rsq_value.setText("—")
            self.aic_value.setText("—")
            self.bic_value.setText("—")
            return
        if elapsed is None:
            elapsed = float("nan")
        self._set_elapsed_status(elapsed, timed_out=False)
        nfev_text = str(result.nfev)
        if result.nfev >= self.nfev_spin.value():
            nfev_text = (
                f'<span style="color:#d62728; font-weight:600;">{nfev_text}</span>'
            )
            self.nfev_out_label.setText(
                '<span style="color:#d62728; font-weight:600;">nfev</span>'
            )
        else:
            self.nfev_out_label.setText("nfev")
        self.nfev_out_value.setText(nfev_text)
        self.redchi_value.setText(f"{result.redchi:.4g}")
        r_squared = result.rsquared if result.rsquared is not None else float("nan")
        self.rsq_value.setText(f"{r_squared:.4g}")
        self.aic_value.setText(f"{result.aic:.4g}")
        self.bic_value.setText(f"{result.bic:.4g}")

    def _set_elapsed_status(self, elapsed: float | None, timed_out: bool) -> None:
        if elapsed is None or not np.isfinite(elapsed):
            self.elapsed_value.setText("—")
        else:
            self.elapsed_value.setText(f"{elapsed:.2f} s")
        if timed_out:
            self.elapsed_label.setText(
                '<span style="color:#d62728; font-weight:600;">Elapsed</span>'
            )
            self.elapsed_value.setText(
                f'<span style="color:#d62728; font-weight:600;">{elapsed:.2f} s</span>'
            )
        else:
            self.elapsed_label.setText("Elapsed")

    def _has_non_finite_params(self) -> bool:
        if self._params is None:
            return False
        return any(not np.isfinite(param.value) for param in self._params.values())

    def _mark_fit_stale(self) -> None:
        self._fit_is_current = False
        self.save_button.setEnabled(False)
        self.copy_button.setEnabled(False)

    def _mark_fit_fresh(self) -> None:
        self._fit_is_current = True
        self.save_button.setEnabled(True)
        self.copy_button.setEnabled(True)

    def _show_warning(self, title: str, text: str) -> None:
        QtWidgets.QMessageBox.warning(self, title, text)

    def _show_error(self, title: str, text: str) -> None:
        erlab.interactive.utils.MessageDialog.critical(self, title, text)

    def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:
        if self._fit_future is not None and not self._fit_future.done():
            self._fit_future.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)
        super().closeEvent(event)


def fit1d(
    data: xr.DataArray,
    model: lmfit.Model | None = None,
    params=None,
    *,
    data_name: str | None = None,
    model_name: str | None = None,
    execute: bool | None = None,
) -> Fit1DTool:
    """Launch an interactive 1D fitting tool.

    Parameters
    ----------
    data : xarray.DataArray
        The 1D data to fit.
    model : lmfit.Model, optional
        The model to fit to the data. If `None`, a default multi-peak model
        will be used.
    params : lmfit.Parameters, optional
        Initial parameters for the fit. If `None`, parameters will be
        initialized from the model.
    data_name : str, optional
        The name of the data variable, used in code generation.
        If `None`, an attempt will be made to infer the name from the
        calling context.
    model_name : str, optional
        The name of the model variable, used in code generation.
        If `None`, an attempt will be made to infer the name from the
        calling context.
    """
    if data_name is None:
        try:
            data_name = str(varname.argname("data", func=fit1d))
        except Exception:
            data_name = "data"
    if model_name is None:
        try:
            model_name = str(varname.argname("model", func=fit1d))
        except Exception:
            model_name = "model"
    with erlab.interactive.utils.setup_qapp(execute):
        win = Fit1DTool(data, model, params, data_name=data_name, model_name=model_name)
        win.show()
        win.raise_()
        win.activateWindow()
    return win
