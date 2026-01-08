"""Interactive tool for fitting 1D curves to images."""

from __future__ import annotations

import typing

import numpy as np
import pyqtgraph as pg
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets
from xarray_lmfit._io import _patch_encode4js
from xarray_lmfit.modelfit import _ParametersWrapper, _parse_params

import erlab.interactive.utils
from erlab.interactive._fit1d import Fit1DTool, _SnapCursorLine, _State2D

if typing.TYPE_CHECKING:
    from collections.abc import Hashable, Mapping

    import lmfit
    import varname

else:
    import lazy_loader as _lazy

    varname = _lazy.load("varname")
    lmfit = _lazy.load("lmfit")


class Fit2DTool(Fit1DTool):
    """Interactive tool for fitting 1D curves to images."""

    tool_name = "ftool_2d"

    @property
    def tool_data(self) -> xr.DataArray:
        return self._data_full

    def __init__(
        self,
        data: xr.DataArray,
        model: lmfit.Model | None = None,
        params: lmfit.Parameters | Mapping[str, typing.Any] | None = None,
        *,
        data_name: str | None = None,
        model_name: str | None = None,
    ) -> None:
        if data.ndim != 2:
            raise ValueError("`data` must be a 2D DataArray")

        if data_name is None:
            data_name = "data"
        self._init_full_data_state(data, data_name=data_name)

        if params is not None:
            parsed_params: xr.DataArray | _ParametersWrapper = _parse_params(params)
            if isinstance(parsed_params, _ParametersWrapper):
                params = parsed_params.params
            else:
                if parsed_params.dims[0] != self._y_dim_name:
                    raise ValueError(
                        f"Some parameters are dependent on dimension "
                        f"`{parsed_params.dims[0]}`, which does not match the "
                        f"independent dimension of the data (`{self._y_dim_name}`)."
                    )
                if (
                    parsed_params.sizes[self._y_dim_name]
                    != data.sizes[self._y_dim_name]
                ):
                    raise ValueError(
                        "The number of parameter sets does not match the size of "
                        "the independent dimension of the data."
                    )
                self._params_full: list[lmfit.Parameters | None] = (
                    parsed_params.values.tolist()
                )
                self._initial_params_full: list[lmfit.Parameters] | None = [
                    p.copy() for p in parsed_params.values
                ]
                params = self._params_full[0]

        super().__init__(
            self._data_full.isel({self._y_dim_name: self._current_idx}),
            model,
            params,
            model_name=model_name,
        )

        self.param_model.sigParamsChanged.connect(self._update_params_full)
        self.sigFitFinished.connect(self._update_params_full)
        self.sigFitFinished.connect(self._update_ds_full)
        self._refresh_contents_from_index()
        self._reset_history_stack()

    def _init_full_data_state(self, data: xr.DataArray, *, data_name: str) -> None:
        self._data_full: xr.DataArray = data
        self._y_dim_name: Hashable = data.dims[0]
        y_size = int(self._data_full.sizes[self._y_dim_name])
        self._current_idx: int = y_size // 2
        self._params_full = [None] * y_size
        self._initial_params_full = None
        self._params_from_coord_full: list[dict[str, str]] = [{} for _ in range(y_size)]
        self._result_ds_full: list[xr.Dataset | None] = [None] * y_size
        self._data_name_full: str = data_name
        self._fit_2d_indices: list[int] = []
        self._fit_2d_total: int = 0
        self._fit_2d_direction: typing.Literal["down", "up"] | None = None
        self._fit_2d_start_idx: int = 0

    def _update_params_full(self) -> None:
        self._params_full[self._current_idx] = self._params
        self._params_from_coord_full[self._current_idx] = self._params_from_coord
        self._update_param_plot()

    def _update_ds_full(self) -> None:
        self._result_ds_full[self._current_idx] = self._last_result_ds

    def _build_ui(self) -> None:
        super()._build_ui()
        y_vals = self._y_values()

        self.transpose_button = QtWidgets.QPushButton("Transpose")
        self.transpose_button.setToolTip("Transpose the 2D data (swap axes)")
        self.transpose_button.clicked.connect(self._transpose)
        self._setup_tab_layout.insertWidget(0, self.transpose_button)

        self.image_plot = self.plot_widget.addPlot(row=0, col=1, rowspan=2)
        self.image_plot.setDefaultPadding(0)
        font = QtGui.QFont()
        font.setPointSizeF(11.0)
        for ax in ("left", "top", "right", "bottom"):
            axis = self.image_plot.getAxis(ax)
            axis.setTickFont(font)
            axis.setStyle(autoExpandTextSpace=True, autoReduceTextSpace=True)
            axis.autoSIPrefix = False

        self.image = erlab.interactive.utils.xImageItem(axisOrder="row-major")
        self.image_plot.addItem(self.image)
        self._refresh_main_image()
        opts = erlab.interactive.options.model
        self.image.set_colormap(
            opts.colors.cmap.name, gamma=0.5, reverse=opts.colors.cmap.reverse
        )

        self.cbar = erlab.interactive.colors.BetterColorBarItem(image=self.image)
        self.cbar.set_dimensions(vert_pad=40)
        self.cbar.setPreferredWidth(60)
        self.plot_widget.addItem(self.cbar, 0, 2, 2, 1)

        self.param_plot_container = QtWidgets.QWidget()
        param_plot_layout = QtWidgets.QVBoxLayout(self.param_plot_container)
        param_plot_layout.setContentsMargins(0, 0, 0, 0)

        self.param_plot_combo = QtWidgets.QComboBox()
        self.param_plot_combo.currentIndexChanged.connect(self._update_param_plot)
        param_plot_layout.addWidget(self.param_plot_combo)

        self.param_plot_widget = pg.GraphicsLayoutWidget()
        param_plot_layout.addWidget(self.param_plot_widget)
        self.param_plot = self.param_plot_widget.addPlot()
        self.param_plot.setDefaultPadding(0)
        self.param_plot.showGrid(x=True, y=True)
        self.param_plot.setLabel("left", self._y_dim_name)
        self.param_plot_errbar = pg.ErrorBarItem(pen=pg.mkPen(color="red"))
        self.param_plot_scatter = pg.ScatterPlotItem(
            size=3,
            pen=pg.mkPen(None),
            brush=pg.mkBrush(255, 0, 0, 200),
            pxMode=True,
        )
        self.param_plot_index_line = _SnapCursorLine(
            pos=y_vals[self._current_idx], angle=0, movable=True
        )
        self.param_plot_index_line.sigDragged.connect(self._on_index_line_dragged)
        self.param_plot.addItem(self.param_plot_index_line)
        self.param_plot.addItem(self.param_plot_errbar)
        self.param_plot.addItem(self.param_plot_scatter)
        font = QtGui.QFont()
        font.setPointSizeF(11.0)
        for ax in ("left", "top", "right", "bottom"):
            axis = self.param_plot.getAxis(ax)
            axis.setTickFont(font)
            axis.setStyle(autoExpandTextSpace=True, autoReduceTextSpace=True)
            axis.autoSIPrefix = False
        self.table_splitter.addWidget(self.param_plot_container)
        self.table_splitter.setStretchFactor(0, 3)
        self.table_splitter.setStretchFactor(1, 1)

        self.index_group = QtWidgets.QGroupBox()
        index_layout = QtWidgets.QVBoxLayout(self.index_group)

        y_min_max_layout = QtWidgets.QHBoxLayout()
        index_layout.addLayout(y_min_max_layout)

        max_idx = self._data_full.sizes[self._y_dim_name] - 1
        y_decimals = erlab.utils.array.effective_decimals(y_vals)

        self.y_range_label = QtWidgets.QLabel("Y range")
        self.y_min_spin = QtWidgets.QSpinBox()
        self.y_max_spin = QtWidgets.QSpinBox()
        self.y_min_spin.setMinimum(0)
        self.y_max_spin.setMinimum(0)
        self.y_min_spin.setMaximum(max_idx)
        self.y_max_spin.setMaximum(max_idx)
        self.y_min_spin.setValue(0)
        self.y_max_spin.setValue(max_idx)
        self.y_min_spin.setKeyboardTracking(False)
        self.y_max_spin.setKeyboardTracking(False)
        y_min_max_layout.addWidget(self.y_range_label)
        y_min_max_layout.addWidget(self.y_min_spin)
        y_min_max_layout.addWidget(self.y_max_spin)

        index_layout_line = QtWidgets.QHBoxLayout()
        index_layout.addLayout(index_layout_line)

        self.y_next_button = erlab.interactive.utils.IconButton("mdi6.triangle")
        self.y_prev_button = erlab.interactive.utils.IconButton("mdi6.triangle-down")
        self.y_index_spin = QtWidgets.QSpinBox()
        self.y_index_spin.setMinimum(0)
        self.y_index_spin.setMaximum(max_idx)
        self.y_index_spin.setValue(self._current_idx)
        self.y_value_spin = QtWidgets.QDoubleSpinBox()
        self.y_value_spin.setDecimals(y_decimals)
        self.y_value_spin.setRange(float(y_vals[0]), float(y_vals[-1]))
        self.y_value_spin.setReadOnly(True)
        self.y_value_spin.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons
        )
        self.y_value_spin.setValue(y_vals[self._current_idx])
        self.y_next_button.clicked.connect(self._next_index)
        self.y_prev_button.clicked.connect(self._prev_index)
        self.y_index_spin.valueChanged.connect(self._y_index_changed)
        self.y_min_spin.valueChanged.connect(self._y_minmax_changed)
        self.y_max_spin.valueChanged.connect(self._y_minmax_changed)
        index_layout_line.addWidget(QtWidgets.QLabel("Index"))
        index_layout_line.addWidget(self.y_index_spin)
        index_layout_line.addWidget(self.y_value_spin)

        index_button_line = QtWidgets.QHBoxLayout()
        index_layout.addLayout(index_button_line)
        index_button_line.addWidget(QtWidgets.QLabel("Navigate"))
        index_button_line.addWidget(self.y_prev_button)
        index_button_line.addWidget(self.y_next_button)

        self.fill_from_prev_button = erlab.interactive.utils.IconButton(
            "mdi6.arrow-left-top"
        )
        self.fill_from_prev_button.setToolTip("Fill parameters from previous index")
        self.fill_from_next_button = erlab.interactive.utils.IconButton(
            "mdi6.arrow-left-bottom"
        )
        self.fill_from_next_button.setToolTip("Fill parameters from next index")
        self.fill_from_prev_button.clicked.connect(self._fill_params_from_prev)
        self.fill_from_next_button.clicked.connect(self._fill_params_from_next)

        self.fill_from_idx_button = erlab.interactive.utils.IconButton("mdi6.numeric")
        self.fill_from_idx_button.setToolTip("Fill parameters from specified index")
        self.fill_from_idx_button.clicked.connect(self._fill_params_from_index)
        self.fill_params_layout = QtWidgets.QHBoxLayout()
        self.fill_params_layout.addWidget(QtWidgets.QLabel("Fill params"))
        self.fill_params_layout.addWidget(self.fill_from_prev_button)
        self.fill_params_layout.addWidget(self.fill_from_next_button)
        self.fill_params_layout.addWidget(self.fill_from_idx_button)
        self.parameters_layout.addLayout(self.fill_params_layout)
        self.reset_all_button = QtWidgets.QPushButton("Reset All")
        self.reset_all_button.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.reset_button.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.guess_button.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.reset_all_button.clicked.connect(self._reset_params_all)
        self.parameters_button_layout.addWidget(self.reset_all_button)

        fill_mode_layout = QtWidgets.QHBoxLayout()
        self.fill_mode_label = QtWidgets.QLabel("Fill mode")
        self.fill_mode_combo = QtWidgets.QComboBox()
        self.fill_mode_combo.addItems(["Previous", "Extrapolate", "None"])
        self.fill_mode_combo.setToolTip(
            "Select how to fill parameters when filling from adjacent indices:\n\n"
            "Previous: fill from previous index.\n"
            "Extrapolate: linearly extrapolate from previous two indices.\n"
            "None: do not fill parameters.\n\n"
            "These are used when:\n"
            "1) filling parameters using the fill from previous/next buttons, and\n"
            "2) when fitting to the upper/lower Y bounds using the fit up/down buttons."
        )
        fill_mode_layout.addWidget(self.fill_mode_label)
        fill_mode_layout.addWidget(self.fill_mode_combo)
        self.parameters_layout.addLayout(fill_mode_layout)

        self._right_layout.insertWidget(0, self.index_group)

        self.x_min_line = _SnapCursorLine(
            pos=self.domain_min_line.x(),
            angle=90,
            movable=True,
            bounds=self.domain_min_line.bounds(),
            pen=pg.mkPen(self.BOUNDS_COLOR, width=2, style=QtCore.Qt.PenStyle.DashLine),
        )
        self.x_min_line.sigDragged.connect(self._domain_min_line_dragged)
        self.x_max_line = _SnapCursorLine(
            pos=self.domain_max_line.x(),
            angle=90,
            movable=True,
            bounds=self.domain_max_line.bounds(),
            pen=pg.mkPen(self.BOUNDS_COLOR, width=2, style=QtCore.Qt.PenStyle.DashLine),
        )
        self.x_max_line.sigDragged.connect(self._domain_max_line_dragged)

        self.index_line = _SnapCursorLine(
            pos=y_vals[self._current_idx], angle=0, movable=True
        )
        self.index_line.sigDragged.connect(self._on_index_line_dragged)
        self.y_min_line = _SnapCursorLine(
            pos=y_vals[0],
            angle=0,
            movable=True,
            pen=pg.mkPen(self.BOUNDS_COLOR, width=2, style=QtCore.Qt.PenStyle.DashLine),
        )
        self.y_min_line.sigDragged.connect(self._on_min_line_dragged)
        self.y_max_line = _SnapCursorLine(
            pos=y_vals[max_idx],
            angle=0,
            movable=True,
            pen=pg.mkPen(self.BOUNDS_COLOR, width=2, style=QtCore.Qt.PenStyle.DashLine),
        )
        self.y_max_line.sigDragged.connect(self._on_max_line_dragged)
        self.image_plot.addItem(self.x_min_line)
        self.image_plot.addItem(self.x_max_line)
        self.image_plot.addItem(self.y_min_line)
        self.image_plot.addItem(self.y_max_line)
        self.image_plot.addItem(self.index_line)

        self.fit_down_button = QtWidgets.QPushButton("Fit ⤓")
        self.fit_down_button.setToolTip(
            "Fit to the lower bound of the Y range, filling in parameters "
            "from previous indices."
        )
        self.fit_up_button = QtWidgets.QPushButton("Fit ⤒")
        self.fit_up_button.setToolTip(
            "Fit to the upper bound of the Y range, filling in parameters "
            "from previous indices."
        )
        self.fit_down_button.clicked.connect(lambda: self._run_fit_2d(direction="down"))
        self.fit_up_button.clicked.connect(lambda: self._run_fit_2d(direction="up"))
        self.fit_buttons.addWidget(self.fit_down_button, 1, 0)
        self.fit_buttons.addWidget(self.fit_up_button, 1, 1)

        self.copy_button.setText("Copy 1D code")
        self.save_button.setText("Save 1D fit")

        self.copy_full_button = QtWidgets.QPushButton("Copy code")
        self.copy_full_button.clicked.connect(self._copy_code_full)
        self.save_full_button = QtWidgets.QPushButton("Save fit")
        self.save_full_button.clicked.connect(self._save_fit_full)

        self.copy_layout.addWidget(self.copy_full_button, 1, 0)
        self.copy_layout.addWidget(self.save_full_button, 1, 1)

        self.main_splitter.setStretchFactor(0, 3)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 2)

        self.resize(1024, 610)
        self._update_param_plot_options()

    @property
    def tool_status(self) -> Fit1DTool.StateModel:
        state_dict = super().tool_status.model_dump()
        state_dict["state2d"] = _State2D(
            current_idx=int(self._current_idx),
            data_name_full=str(self._data_name_full),
            params_full=[
                self._serialize_params(params) if params is not None else None
                for params in self._params_full
            ],
            params_from_coord_full=self._params_from_coord_full.copy(),
            fill_mode=typing.cast(
                'typing.Literal["previous", "extrapolate", "none"]',
                self.fill_mode_combo.currentText().lower(),
            ),
        )
        return self.StateModel(**state_dict)

    @tool_status.setter
    def tool_status(self, status: Fit1DTool.StateModel) -> None:
        state2d = status.state2d
        if state2d is not None:  # pragma: no branch
            self._current_idx = state2d.current_idx
            self._data_name_full = state2d.data_name_full
            self._params_from_coord_full = state2d.params_from_coord_full.copy()
            self._params_full = [
                self._deserialize_params(params) for params in state2d.params_full
            ]
            self.fill_mode_combo.setCurrentText(state2d.fill_mode.capitalize())
        super(Fit2DTool, self.__class__).tool_status.__set__(  # type: ignore[attr-defined]
            self, status
        )
        self.y_index_spin.setValue(self._current_idx)

    def _transpose(self) -> None:
        # Reset everything, transpose data, and rebuild UI.
        old_geom = self.saveGeometry() if hasattr(self, "saveGeometry") else None

        self._cancel_fit()

        # Remove the existing UI to avoid duplicate widgets/signals.
        if hasattr(self, "centralWidget"):
            old_cw = self.centralWidget()
            if old_cw is not None:
                old_cw.setParent(None)
                old_cw.deleteLater()

        # Transpose the full 2D data (swap axes).
        self._init_full_data_state(
            erlab.interactive.utils.parse_data(self._data_full.transpose()),
            data_name=self._data_name_full,
        )

        # Reset current slice + fit state.
        model = self._model
        self._reset_fit_state(
            self._data_full.isel({self._y_dim_name: self._current_idx}),
            model,
            None,
            data_name=self._data_name,
            model_name=self._model_name,
        )

        # Rebuild UI and refresh views.
        self._build_ui()
        self.param_model.sigParamsChanged.connect(self._update_params_full)
        self.sigFitFinished.connect(self._update_params_full)
        self.sigFitFinished.connect(self._update_ds_full)
        self._update_fit_curve()
        self._write_history = True
        self._write_state()
        self._refresh_contents_from_index()

        if old_geom is not None and hasattr(self, "restoreGeometry"):
            self.restoreGeometry(old_geom)

    @QtCore.Slot()
    def _domain_changed(self) -> None:
        super()._domain_changed()
        self.x_min_line.setPos(self.domain_min_line.x())
        self.x_max_line.setPos(self.domain_max_line.x())

    @QtCore.Slot(object)
    def _on_index_line_dragged(self, line: _SnapCursorLine) -> None:
        y_vals = self._y_values()
        pos = line.temp_value
        idx = (np.abs(y_vals - pos)).argmin()
        self._set_current_index(idx)
        self._current_idx = self.y_index_spin.value()

    @QtCore.Slot(object)
    def _on_min_line_dragged(self, line: _SnapCursorLine) -> None:
        y_vals = self._y_values()
        pos = line.temp_value
        idx = (np.abs(y_vals - pos)).argmin()
        self.y_min_spin.setValue(int(idx))

    @QtCore.Slot(object)
    def _on_max_line_dragged(self, line: _SnapCursorLine) -> None:
        y_vals = self._y_values()
        pos = line.temp_value
        idx = (np.abs(y_vals - pos)).argmin()
        self.y_max_spin.setValue(int(idx))

    @QtCore.Slot()
    def _y_index_changed(self) -> None:
        self._current_idx = self.y_index_spin.value()
        y_vals = self._y_values()
        curr_val = y_vals[self._current_idx]
        self.index_line.setPos(curr_val)
        self.param_plot_index_line.setPos(curr_val)
        self.y_value_spin.setValue(curr_val)
        self._refresh_contents_from_index()

    @QtCore.Slot()
    def _refresh_multipeak_model(self) -> None:
        super()._refresh_multipeak_model()
        for idx in range(self._data_full.sizes[self._y_dim_name]):
            if idx != self._current_idx:
                prev_params = self._params_full[idx]
                if prev_params is not None:
                    new_params = self._model.make_params()
                    self._merge_params(prev_params, new_params)
                    self._params_full[idx] = new_params
                    prev_params_from_coord = self._params_from_coord_full[idx]
                    for k in list(prev_params_from_coord.keys()):
                        if k not in new_params:
                            del prev_params_from_coord[k]
                            self._params_from_coord_full[idx] = prev_params_from_coord

    @QtCore.Slot()
    def _update_param_plot_options(self) -> None:
        with QtCore.QSignalBlocker(self.param_plot_combo):
            prev_param = self.param_plot_combo.currentText()
            self.param_plot_combo.clear()
            updated_names = self._model.param_names
            self.param_plot_combo.addItems(updated_names)
            if prev_param in updated_names:
                self.param_plot_combo.setCurrentText(prev_param)
            else:
                self._update_param_plot()

    def set_model(self, *args, **kwargs) -> None:
        super().set_model(*args, **kwargs)
        self._update_param_plot_options()

    @QtCore.Slot()
    def _update_param_plot(self) -> None:
        param_name = self.param_plot_combo.currentText()
        self.param_plot.setLabel("bottom", param_name)
        y_range_slice = self._y_range_slice()

        plot_y = []
        param_values = []
        param_errors = []
        for y, params, result_ds in zip(
            self._y_values()[y_range_slice],
            self._params_full[y_range_slice],
            self._result_ds_full[y_range_slice],
            strict=True,
        ):
            if (result_ds is None) or (params is None) or (param_name not in params):
                continue
            param = params[param_name]
            plot_y.append(y)
            param_values.append(param.value)
            param_errors.append(param.stderr if param.stderr is not None else 0.0)

        plot_y = np.array(plot_y)
        param_values = np.array(param_values)
        self.param_plot_errbar.setData(
            x=param_values,
            y=plot_y,
            width=np.array(param_errors),
        )
        self.param_plot_scatter.setData(x=param_values, y=plot_y)

    def _refresh_contents_from_index(self) -> None:
        self._data = self._data_full.isel({self._y_dim_name: self._current_idx})

        params = self._params_full[self._current_idx]
        params_from_coord = self._params_from_coord_full[self._current_idx]
        last_ds = self._result_ds_full[self._current_idx]
        if self._initial_params_full is not None:
            self._initial_params = self._initial_params_full[self._current_idx]
        if params is None:
            params = self._initial_params.copy()
        for param_name, param_coord in params_from_coord.items():
            if param_name in params and param_coord in self._data.coords:
                params[param_name].value = float(self._data[param_coord].values)

        self._params = params
        self._params_from_coord = params_from_coord
        self.param_model.set_params(self._params, self._params_from_coord)
        self._populate_data_curve()

        self._data_name = (
            self._data_name_full
            + ".isel("
            + erlab.interactive.utils.format_kwargs(
                {self._y_dim_name: self._current_idx}
            )
            + ")"
        )

        if last_ds is not None:
            result = last_ds.modelfit_results.compute().item()
            self._set_fit_stats(result)
        else:
            self._set_fit_stats(None)
        self._mark_fit_stale()

    @QtCore.Slot()
    def _y_minmax_changed(self) -> None:
        y_vals = self._y_values()
        min_idx = self.y_min_spin.value()
        max_idx = self.y_max_spin.value()
        self.y_min_spin.setMaximum(max_idx)
        self.y_max_spin.setMinimum(min_idx)
        self.y_index_spin.setRange(min_idx, max_idx)
        min_val, max_val = (y_vals[min_idx], y_vals[max_idx])
        self.y_min_line.setBounds((y_vals[0], max_val))
        self.y_max_line.setBounds((min_val, y_vals[-1]))
        self.index_line.setBounds((min_val, max_val))
        self.y_min_line.setPos(min_val)
        self.y_max_line.setPos(max_val)
        self._update_full_fit_saveable()

    def _y_range_slice(self) -> slice:
        return slice(self.y_min_spin.value(), self.y_max_spin.value() + 1)

    @QtCore.Slot()
    def _next_index(self) -> None:
        if self._current_idx < self.y_max_spin.value():
            self._set_current_index(self._current_idx + 1)

    @QtCore.Slot()
    def _prev_index(self) -> None:
        if self._current_idx > self.y_min_spin.value():
            self._set_current_index(self._current_idx - 1)

    def _set_current_index(self, index: int) -> None:
        self.y_index_spin.setValue(int(index))

    def _fill_params_from(self, target_index: int, mode=None) -> None:
        if (
            target_index < self.y_min_spin.value()
            or target_index > self.y_max_spin.value()
        ):
            self._show_warning(
                "Invalid Index",
                "Cannot fill parameters because the given index is out of bounds.",
            )
            return
        if mode is None:
            mode = self.fill_mode_combo.currentText().lower()

        self._params_from_coord_full[self._current_idx] = self._params_from_coord_full[
            target_index
        ].copy()

        if mode == "none":
            return
        if mode == "extrapolate":
            i2 = target_index
            i1 = i2 + (target_index - self._current_idx)

            params1 = self._params_full[i1]
            params2 = self._params_full[i2]
            if (
                i1 < self.y_min_spin.value()
                or i1 > self.y_max_spin.value()
                or params1 is None
                or params2 is None
            ):
                self._fill_params_from(target_index, mode="previous")
                return

            params = params2.copy()
            for name in params:
                if (
                    name in params1
                    and params1[name].expr is None
                    and params2[name].expr is None
                ):
                    params[name].set(
                        value=2 * params2[name].value - params1[name].value
                    )

        elif mode == "previous":
            params = self._params_full[target_index]
            if params is None:
                params = self._initial_params.copy()
                self._params_full[target_index] = params.copy()

        self._params_full[self._current_idx] = params.copy()

        self._refresh_contents_from_index()

    def _fill_params_from_prev(self) -> None:
        self._fill_params_from(self._current_idx - 1)

    def _fill_params_from_next(self) -> None:
        self._fill_params_from(self._current_idx + 1)

    def _fill_params_from_index(self) -> None:
        idx, ok = QtWidgets.QInputDialog.getInt(
            self,
            "Fill Parameters From Index",
            f"Enter index ({self.y_min_spin.value()} - {self.y_max_spin.value()}):",
            value=self._current_idx,
            min=self.y_min_spin.value(),
            max=self.y_max_spin.value(),
        )
        if ok:
            self._fill_params_from(idx)

    def _reset_params_all(self) -> None:
        match QtWidgets.QMessageBox.question(
            self,
            "Reset All",
            "All parameters for every index will be reset to their default values. "
            "Continue?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        ):
            case QtWidgets.QMessageBox.StandardButton.No:
                return
            case _:
                pass

        if self._initial_params_full is not None:
            self._params_full = [params.copy() for params in self._initial_params_full]
        else:
            self._params_full = [
                self._initial_params.copy() for _ in range(len(self._params_full))
            ]
        self._params_from_coord_full = [{} for _ in range(len(self._params_full))]
        self._result_ds_full = [None] * len(self._params_full)
        self._refresh_contents_from_index()
        self._update_param_plot()
        self._mark_fit_stale()

    def _set_fit_running(
        self, running: bool, *, multi: bool, step: int = 0, total: int = 0
    ) -> None:
        super()._set_fit_running(running, multi=multi, step=step, total=total)
        self.fit_down_button.setDisabled(running)
        self.fit_up_button.setDisabled(running)

    def _fit_cancelled(self) -> None:
        if self._fit_2d_total > 0 or self._fit_2d_indices:
            self._finish_fit_2d_sequence()
        else:
            super()._fit_cancelled()

    def _run_fit_2d(self, direction: typing.Literal["down", "up"]) -> None:
        if self._fit_running():
            self._show_warning("Fit running", "A fit is already running.")
            return

        start_idx = int(self._current_idx)
        if direction == "up":
            end_idx = self.y_max_spin.value()
        else:
            end_idx = self.y_min_spin.value()

        indices = (
            range(start_idx, end_idx - 1, -1)
            if end_idx < start_idx
            else range(start_idx, end_idx + 1, 1)
        )
        self._fit_2d_indices = list(indices)
        self._fit_2d_total = len(self._fit_2d_indices)
        self._fit_2d_direction = direction
        self._fit_2d_start_idx = start_idx
        if self._fit_2d_indices:
            self._start_next_fit_2d()

    def _start_next_fit_2d(self) -> None:
        if not self._fit_2d_indices:
            self._finish_fit_2d_sequence()
            return

        idx = self._fit_2d_indices.pop(0)
        self._set_current_index(idx)
        if idx != self._fit_2d_start_idx:
            direction = self._fit_2d_direction or "up"
            self._fill_params_from(idx - 1 if direction == "up" else idx + 1)

        step = self._fit_2d_total - len(self._fit_2d_indices)

        def _on_success(result_ds: xr.Dataset) -> None:
            if self._fit_start_time is None:
                return
            self._set_fit_ds(result_ds, self._fit_start_time)
            max_nfev = self.nfev_spin.value()
            result = result_ds.modelfit_results.compute().item()
            if max_nfev > 0 and result.nfev >= max_nfev:
                self._show_warning(
                    "Fit Stopped",
                    f"Fit stopped at index {idx} because the maximum number of "
                    "function evaluations was reached.",
                )
                self._finish_fit_2d_sequence()
                return
            self._start_next_fit_2d()

        def _on_timeout() -> None:
            if self._fit_start_time is None:
                return
            self._fit_timed_out(self._fit_start_time)
            self._finish_fit_2d_sequence()

        def _on_error(message) -> None:
            self._fit_errored(message)
            self._finish_fit_2d_sequence()

        started = self._start_fit_worker(
            self._fit_data(),
            self._params,
            multi=True,
            step=step,
            total=self._fit_2d_total,
            on_success=_on_success,
            on_timeout=_on_timeout,
            on_error=_on_error,
        )
        if not started:
            self._finish_fit_2d_sequence()

    def _finish_fit_2d_sequence(self) -> None:
        self._set_fit_running(False, multi=True)
        self._fit_running_multi = False
        self._fit_2d_indices = []
        self._fit_2d_total = 0
        self._fit_2d_direction = None
        self._update_full_fit_saveable()
        self._update_param_plot()

    def _y_values(self) -> np.ndarray:
        if self._y_dim_name in self._data_full.coords:
            coords = self._data_full.coords[self._y_dim_name]
            return np.asarray(coords.values)
        return np.arange(self._data_full.sizes[self._y_dim_name], dtype=float)

    def _refresh_main_image(self) -> None:
        self.image.setDataArray(self._data_full)

    def _mark_fit_stale(self) -> None:
        super()._mark_fit_stale()
        self.save_full_button.setDisabled(True)
        self.copy_full_button.setDisabled(True)

    def _mark_fit_fresh(self) -> None:
        super()._mark_fit_fresh()
        self._update_full_fit_saveable()

    def _update_full_fit_saveable(self) -> None:
        can_save: bool = self._fit_is_current and not any(
            ds is None for ds in self._result_ds_full[self._y_range_slice()]
        )
        self.save_full_button.setEnabled(can_save)
        self.copy_full_button.setEnabled(can_save)

    @QtCore.Slot()
    def _save_fit_full(self) -> None:
        results = []
        for i, ds in enumerate(self._result_ds_full[self._y_range_slice()]):
            if ds is None:
                real_index = i + self.y_min_spin.value()
                self._show_warning(
                    "Missing Fit",
                    f"No fit result for index {real_index}. Please fit all indices "
                    "in the range before saving the full fit.",
                )
                return
            results.append(ds)
        with erlab.interactive.utils.wait_dialog(self, "Combining fit results..."):
            full_ds = xr.concat(
                results,
                dim=self._y_dim_name,
                data_vars="all",
                coords="minimal",
                compat="override",
                join="override",
                combine_attrs="override",
            )
        erlab.interactive.utils.save_fit_ui(full_ds, parent=self)

    @QtCore.Slot()
    def _copy_code_full(self) -> str:
        data_name, model_name, lines = self._make_model_code(self._data_name_full)

        isel_kw = erlab.interactive.utils.format_kwargs(
            {self._y_dim_name: self._y_range_slice()}
        )

        fit_domain = self._fit_domain()
        if fit_domain is not None:
            sel_kw = erlab.interactive.utils.format_kwargs(
                {self._coord_name: slice(*fit_domain)}
            )
            lines.append(
                f"{data_name}_crop = {data_name}.sel({sel_kw}).isel({isel_kw})"
            )
        else:
            lines.append(f"{data_name}_crop = {data_name}.isel({isel_kw})")
        data_name = f"{data_name}_crop"

        if self.normalize_check.isChecked():
            lines.append(
                f"{data_name}_norm = "
                f'{data_name} / {data_name}.mean("{self._coord_name}")'
            )
            data_name = f"{data_name}_norm"

        param_names: list[str] = []
        param_names_all: list[str] = []
        params_expr: dict[str, str] = {}
        for k in self._params:
            param = self._params[k]
            if param.expr is not None:
                if self._param_expr_from_hint(param):
                    continue
                params_expr[k] = param.expr
            else:
                param_names.append(k)
            param_names_all.append(k)

        # param_names_all = list(self._model.param_names)
        params_value: dict[str, list[float]] = {name: [] for name in param_names}
        params_min: dict[str, list[float]] = {name: [] for name in param_names}
        params_max: dict[str, list[float]] = {name: [] for name in param_names}
        params_vary: dict[str, bool] = {}

        y_range_slice = self._y_range_slice()

        for i, (params, ds) in enumerate(
            zip(
                self._params_full[y_range_slice],
                self._result_ds_full[y_range_slice],
                strict=True,
            )
        ):
            if ds is None or params is None:
                real_index = i + self.y_min_spin.value()
                self._show_warning(
                    "Missing Fit",
                    f"No fit result for index {real_index}. Please fit all indices "
                    "in the range before saving the full fit.",
                )
                return ""
            for expr_param in params_expr:
                if expr_param in params:
                    this_expr = params[expr_param].expr
                    if this_expr != params_expr[expr_param]:
                        self._show_warning(
                            "Inconsistent Parameters",
                            f"Parameter {expr_param!r} has differing expressions "
                            "between fits. Cannot generate combined fit code.",
                        )
                        return ""
                    continue
                self._show_warning(
                    "Inconsistent Parameters",
                    f"Parameter {expr_param!r} not found in fit at index "
                    f"{real_index}. Cannot generate combined fit code.",
                )
                return ""

            valid_params = [k for k in params if params[k].expr is None]
            if valid_params != param_names:
                self._show_warning(
                    "Inconsistent Parameters",
                    "Parameter names or counts differ between fits. "
                    "Cannot generate combined fit code.",
                )
                return ""
            for name in param_names:
                param = params[name]
                params_value[name].append(param.value)
                params_min[name].append(param.min if param.min is not None else -np.inf)
                params_max[name].append(param.max if param.max is not None else np.inf)
                if name not in params_vary:
                    params_vary[name] = param.vary
                else:
                    if params_vary[name] != param.vary:
                        self._show_warning(
                            "Inconsistent Parameters",
                            "Parameter vary flags differ between fits. "
                            "Cannot generate combined fit code.",
                        )
                        return ""

        param_entries: list[str] = []

        for name in param_names_all:
            if name in params_expr:
                param_entries.append(f'"{name}": dict(expr="{params_expr[name]}"),')
                continue
            values = params_value[name]
            mins = params_min[name]
            maxs = params_max[name]
            vary: bool = params_vary[name]

            single_value: bool = np.allclose(values, values[0])
            single_min: bool = np.allclose(mins, mins[0])
            single_max: bool = np.allclose(maxs, maxs[0])

            entry_kwargs_lines: list[str] = []

            if single_value:
                if (
                    single_min
                    and single_max
                    and not np.isfinite(mins[0])
                    and not np.isfinite(maxs[0])
                    and vary
                ):
                    param_entries.append(f'"{name}": {values[0]!r},')
                    continue
                entry_kwargs_lines.append(f"value={values[0]!r}")
            else:
                darr_line = (
                    f"xr.DataArray({values!r}, coords=[{data_name}.{self._y_dim_name}])"
                )
                if (
                    single_min
                    and single_max
                    and not np.isfinite(mins[0])
                    and not np.isfinite(maxs[0])
                    and vary
                ):
                    param_entries.append(f'"{name}": {darr_line},')
                    continue
                entry_kwargs_lines.append(f"value={darr_line}")

            if single_min:
                if np.isfinite(mins[0]):
                    entry_kwargs_lines.append(f"min={mins[0]!r}")
            else:
                entry_kwargs_lines.append(
                    f"xr.DataArray({mins!r}, coords=[{data_name}.{self._y_dim_name}])"
                )

            if single_max:
                if np.isfinite(maxs[0]):
                    entry_kwargs_lines.append(f"max={maxs[0]!r}")
            else:
                entry_kwargs_lines.append(
                    f"xr.DataArray({maxs!r}, coords=[{data_name}.{self._y_dim_name}])"
                )

            if not vary:
                entry_kwargs_lines.append("vary=False")

            param_entries.append(f'"{name}": dict({", ".join(entry_kwargs_lines)}),')

        lines.extend(["params = {", "\n    ".join(param_entries), "}\n"])

        lines.append(
            erlab.interactive.utils.generate_code(
                self._data_full.xlm.modelfit,
                args=[self._coord_name],
                kwargs={
                    "model": f"|{model_name}|",
                    "params": "|params|",
                    "method": self.method_combo.currentText(),
                },
                name="modelfit",
                module=f"{data_name}.xlm",
                assign="result",
            )
        )
        return erlab.interactive.utils.copy_to_clipboard(lines)


def ftool(
    data: xr.DataArray,
    model: lmfit.Model | None = None,
    params: lmfit.Parameters | dict[str, typing.Any] | None = None,
    *,
    data_name: str | None = None,
    model_name: str | None = None,
    execute: bool | None = None,
) -> Fit1DTool | Fit2DTool:
    """Launch an interactive fitting tool.

    The tool provides an interactive GUI for fitting 1D models to 1D or 2D data.

    See the :ref:`user guide <guide-ftool>` for more information on using the tool.

    Parameters
    ----------
    data
        The 1D or 2D data to fit.
    model
        The model to fit to the data. If `None`, :class:`MultiPeakModel
        <erlab.analysis.fit.models.MultiPeakModel>` will be used.
    params
        Initial parameters for the fit. If `None`, parameters will be initialized from
        the model. If `data` is 2D, `params` can be a dictionary that is interpreted
        like the ``params`` argument of :meth:`xarray.DataArray.xlm.modelfit`.
    data_name : str, optional
        The name of the data variable, used in code generation. If `None`, an attempt
        will be made to infer the name from the calling context.
    model_name : str, optional
        The name of the model variable, used in code generation. If `None`, an attempt
        will be made to infer the name from the calling context.
    """
    if data_name is None:
        try:
            data_name = str(varname.argname("data", func=ftool, vars_only=False))
        except Exception:
            data_name = "data"
    if model_name is None:
        try:
            model_name = str(varname.argname("model", func=ftool, vars_only=False))
        except Exception:
            model_name = "model"
    match data.ndim:
        case 1:
            tool_cls = Fit1DTool
        case 2:
            tool_cls = Fit2DTool
        case _:
            raise ValueError("`data` must be a 1D or 2D DataArray")
    with _patch_encode4js(), erlab.interactive.utils.setup_qapp(execute):
        win = tool_cls(data, model, params, data_name=data_name, model_name=model_name)
        win.show()
        win.raise_()
        win.activateWindow()
    return win
