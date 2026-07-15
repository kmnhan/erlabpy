"""Interactive tool for fitting 1D curves to images."""

from __future__ import annotations

import contextlib
import dataclasses
import enum
import functools
import logging
import os
import time
import typing
import urllib.parse

import numpy as np
import pyqtgraph as pg
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets
from xarray_lmfit._io import _patch_encode4js
from xarray_lmfit.modelfit import (
    _materialize_broadcast_params,
    _ParametersWrapper,
    _parse_params,
)

import erlab.interactive.utils
from erlab.interactive._fit1d import (
    Fit1DTool,
    _FitRestoreState,
    _load_lmfit_for_ftool_restore,
    _SnapCursorLine,
    _State2D,
)
from erlab.interactive.imagetool._provenance._model import (
    ToolProvenanceOperation,
    ToolProvenanceSpec,
    compose_full_provenance,
    direct_replay_input_name,
    replay_input_name,
    script,
    to_replay_provenance_spec,
)
from erlab.interactive.imagetool._provenance._operations import (
    IselOperation,
    ModelFitOperation,
    ScriptCodeOperation,
    SelOperation,
    _model_fit_parameters_code,
    _ModelFitParameterSpec,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Mapping

    import lmfit
    import varname

    from erlab.interactive._figurecomposer import FigureOperationState
    from erlab.interactive.imagetool.manager import ImageToolManager

else:
    import lazy_loader as _lazy

    varname = _lazy.load("varname")
    lmfit = _lazy.load("lmfit")

logger = logging.getLogger(__name__)

_P = typing.ParamSpec("_P")
_FIT2D_SEQUENCE_LIVE_REFRESH_INTERVAL_S = 0.10
_R = typing.TypeVar("_R")


def _rebuild_ui(
    *,
    mark_fresh: bool = True,
) -> Callable[
    [Callable[typing.Concatenate[Fit2DTool, _P], _R]],
    Callable[typing.Concatenate[Fit2DTool, _P], _R],
]:
    """Decorate a method to rebuild the UI after changing the data/model."""

    def decorator(
        func: Callable[typing.Concatenate[Fit2DTool, _P], _R],
    ) -> Callable[typing.Concatenate[Fit2DTool, _P], _R]:
        @functools.wraps(func)
        def wrapper(self: Fit2DTool, *args: _P.args, **kwargs: _P.kwargs) -> _R:
            old_geom = self.saveGeometry() if hasattr(self, "saveGeometry") else None
            self._cancel_fit()

            # Remove the existing UI to avoid duplicate widgets/signals.
            if hasattr(self, "centralWidget"):
                old_cw = self.centralWidget()
                if old_cw is not None:
                    old_cw.setParent(None)
                    old_cw.deleteLater()

            try:
                return func(self, *args, **kwargs)
            finally:
                # Reset current slice + fit state.
                self._reset_fit_state(
                    self._data_full.isel({self._y_dim_name: self._current_idx}),
                    self._model,
                    None,
                    data_name=self._data_name,
                    model_name=self._model_name,
                )

                # Rebuild UI and refresh views.
                self._build_ui()
                self.param_model.sigParamsChanged.connect(self._update_params_full)
                self._update_fit_curve()
                self._write_history = True
                self._write_state()
                self._refresh_contents_from_index(mark_fit_stale=not mark_fresh)

                if old_geom is not None and hasattr(self, "restoreGeometry"):
                    self.restoreGeometry(old_geom)

        return wrapper

    return decorator


class _Fit2DParameterPlotItem(pg.PlotItem):
    """Parameter plot item with context menu actions for exporting data."""

    def __init__(self, tool: Fit2DTool) -> None:
        super().__init__()
        self._tool = tool
        self._setup_actions()

    def _setup_actions(self) -> None:
        self.vb.menu.addSeparator()

        save_values_action = self.vb.menu.addAction("Save parameter values as HDF5…")
        save_values_action.triggered.connect(self._save_parameter_values)

        show_values_action = self.vb.menu.addAction(
            "Show parameter values in ImageTool"
        )
        show_values_action.triggered.connect(self._show_parameter_values)

        self.vb.menu.addSeparator()

        save_stderr_action = self.vb.menu.addAction(
            "Save parameter standard error as HDF5"
        )
        save_stderr_action.triggered.connect(self._save_parameter_stderr)

        show_stderr_action = self.vb.menu.addAction(
            "Show parameter standard error in ImageTool"
        )
        show_stderr_action.triggered.connect(self._show_parameter_stderr)

        self.vb.menu.addSeparator()

        add_to_figure_action = self.vb.menu.addAction("Add parameter plot to Figure…")
        add_to_figure_action.setObjectName("fit2dParamPlotAddToFigureAction")
        add_to_figure_action.triggered.connect(self._add_parameter_plot_to_figure)

    def _current_param_dataarray(
        self, *, stderr: bool
    ) -> tuple[str, xr.DataArray] | None:
        param_name = self._tool.param_plot_combo.currentText().strip()
        if not param_name:
            self._tool._show_warning(
                "No parameter selected",
                "Select a parameter in the parameter plot first.",
            )
            return None

        da = self._tool._param_plot_dataarray(param_name, stderr=stderr)
        if da.size == 0:
            kind = "standard errors" if stderr else "parameter values"
            self._tool._show_warning(
                "No data available",
                f"No {kind} are available for the selected parameter in the "
                "current y-range.",
            )
            return None
        return param_name, da

    def _save_dataarray_as_hdf5(self, data: xr.DataArray) -> None:
        if isinstance(data.name, str):
            default_name = data.name or "data"
        elif data.name is None:
            default_name = "data"
        else:
            default_name = str(data.name)

        dialog = QtWidgets.QFileDialog(self._tool)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.AnyFile)
        dialog.setNameFilters(["xarray HDF5 Files (*.h5)", "All files (*)"])
        dialog.setDefaultSuffix("h5")

        if os.environ.get("PYTEST_VERSION") is not None:
            dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)

        last_dir = pg.PlotItem.lastFileDir
        if not last_dir:
            last_dir = erlab.interactive.imagetool.manager._get_recent_directory()
        if not last_dir:
            last_dir = os.getcwd()
        dialog.setDirectory(os.path.join(last_dir, f"{default_name}.h5"))

        if dialog.exec():
            filename = dialog.selectedFiles()[0]
            try:
                data.to_netcdf(filename, engine="h5netcdf", invalid_netcdf=True)
            except Exception:
                logger.exception(
                    "Error while saving Fit2D parameter data",
                    extra={"suppress_ui_alert": True},
                )
                erlab.interactive.utils.MessageDialog.critical(
                    self._tool,
                    "Error",
                    "An error occurred while saving the parameter data.",
                )
                return
            pg.PlotItem.lastFileDir = os.path.dirname(filename)

    @QtCore.Slot()
    def _save_parameter_values(self) -> None:
        current = self._current_param_dataarray(stderr=False)
        if current is not None:
            _, da = current
            self._save_dataarray_as_hdf5(da)

    @QtCore.Slot()
    def _show_parameter_values(self) -> None:
        current = self._current_param_dataarray(stderr=False)
        if current is not None:
            param_name, da = current
            self._tool._show_dataarray_in_itool(
                da,
                output_id=self._tool._parameter_output_id(
                    Fit2DTool.Output.PARAMETER_VALUES, param_name
                ),
            )

    @QtCore.Slot()
    def _save_parameter_stderr(self) -> None:
        current = self._current_param_dataarray(stderr=True)
        if current is not None:
            _, da = current
            self._save_dataarray_as_hdf5(da)

    @QtCore.Slot()
    def _show_parameter_stderr(self) -> None:
        current = self._current_param_dataarray(stderr=True)
        if current is not None:
            param_name, da = current
            self._tool._show_dataarray_in_itool(
                da,
                output_id=self._tool._parameter_output_id(
                    Fit2DTool.Output.PARAMETER_STDERR, param_name
                ),
            )

    @QtCore.Slot()
    def _add_parameter_plot_to_figure(self) -> None:
        self._tool._add_parameter_plot_to_figure()


class Fit2DTool(Fit1DTool):
    """Interactive tool for fitting 1D curves to images."""

    tool_name = "ftool_2d"
    COPY_PROVENANCE: typing.ClassVar = (
        erlab.interactive.utils.ToolScriptProvenanceDefinition(
            start_label="Start from current ftool input data",
            label="Fit current 2D data with the current model",
            prelude_method="_checked_full_copy_prelude",
            expression_method="_full_fit_expression",
            assign="result",
        )
    )
    _DETACHED_COPY_PROVENANCE: typing.ClassVar = (
        erlab.interactive.utils.ToolScriptProvenanceDefinition(
            start_label="Start from current ftool input data",
            label="Fit current 2D data with the current model",
            prelude_method="_detached_full_copy_prelude",
            expression_method="_full_fit_expression",
            assign="result",
        )
    )
    _PERSISTED_FIT_RESULT_VAR: typing.ClassVar[str] = "__ftool_fit_results__"
    _PERSISTED_FIT_RESULT_DIM: typing.ClassVar[str] = "__ftool_fit_results_bytes__"
    _PERSISTED_FIT_INDEX_DIM: typing.ClassVar[str] = "__ftool_fit_result_index__"
    _PERSISTED_FIT_CURRENT_ATTR: typing.ClassVar[str] = "__ftool_fit_is_current__"
    _PARAMETER_OUTPUT_SEPARATOR: typing.ClassVar[str] = ":"

    class Output(enum.StrEnum):
        PARAMETER_VALUES = "fit2d.param_plot.values"
        PARAMETER_STDERR = "fit2d.param_plot.stderr"

    IMAGE_TOOL_OUTPUTS: typing.ClassVar = {
        Output.PARAMETER_VALUES: erlab.interactive.utils.ToolImageOutputDefinition(
            data_method="_parameter_values_output_data",
        ),
        Output.PARAMETER_STDERR: erlab.interactive.utils.ToolImageOutputDefinition(
            data_method="_parameter_stderr_output_data",
        ),
    }

    @classmethod
    def _parameter_output_id(cls, output: Output, param_name: str) -> str:
        if output not in (cls.Output.PARAMETER_VALUES, cls.Output.PARAMETER_STDERR):
            raise ValueError("output must be a Fit2DTool parameter output")
        return (
            f"{output.value}{cls._PARAMETER_OUTPUT_SEPARATOR}"
            f"{urllib.parse.quote(param_name, safe='')}"
        )

    @classmethod
    def _parameter_output_parts(
        cls, output_id: str | enum.Enum
    ) -> tuple[Output, str | None] | None:
        normalized = erlab.interactive.utils._normalize_tool_output_id(output_id)
        for output in (cls.Output.PARAMETER_VALUES, cls.Output.PARAMETER_STDERR):
            if normalized == output.value:
                return output, None
            prefix = f"{output.value}{cls._PARAMETER_OUTPUT_SEPARATOR}"
            if normalized.startswith(prefix):
                return output, urllib.parse.unquote(normalized.removeprefix(prefix))
        return None

    def _image_output_definition(
        self, output_id: str | enum.Enum
    ) -> tuple[str, erlab.interactive.utils.ToolImageOutputDefinition]:
        normalized = erlab.interactive.utils._normalize_tool_output_id(output_id)
        parts = self._parameter_output_parts(normalized)
        if parts is None:
            return super()._image_output_definition(normalized)
        output, _ = parts
        _, definition = super()._image_output_definition(output)
        return normalized, definition

    @property
    def tool_data(self) -> xr.DataArray:
        return self._data_full

    @property
    def preview_imageitem(self) -> pg.ImageItem:
        return self.image

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
            parameter_inputs = _parse_params(params)
            if parameter_inputs.plan.mode == "static":
                if parameter_inputs.plan.template is None:  # pragma: no cover
                    # Guaranteed by xarray-lmfit for static parameter inputs.
                    raise RuntimeError("Static parameter template was not initialized.")
                params = parameter_inputs.plan.template.params
            elif parameter_inputs.plan.mode == "broadcast":
                template = lmfit.create_params(
                    **{
                        name: dict(specs)
                        for name, specs in parameter_inputs.plan.template_specs
                    }
                )
                parameter_plan = dataclasses.replace(
                    parameter_inputs.plan,
                    template=_ParametersWrapper(template),
                )
                parameter_sets = xr.apply_ufunc(
                    lambda *values: (
                        _materialize_broadcast_params(
                            parameter_plan, values, guess=False
                        ).params
                    ),
                    *(array.compute() for array in parameter_inputs.arrays),
                    vectorize=True,
                    output_dtypes=[object],
                )
                if parameter_sets.dims[0] != self._y_dim_name:
                    raise ValueError(
                        f"Some parameters are dependent on dimension "
                        f"`{parameter_sets.dims[0]}`, which does not match the "
                        f"independent dimension of the data (`{self._y_dim_name}`)."
                    )
                if (
                    parameter_sets.sizes[self._y_dim_name]
                    != data.sizes[self._y_dim_name]
                ):
                    raise ValueError(
                        "The number of parameter sets does not match the size of "
                        "the independent dimension of the data."
                    )
                self._params_full: list[lmfit.Parameters | None] = (
                    parameter_sets.values.tolist()
                )
                self._initial_params_full: list[lmfit.Parameters] | None = [
                    p.copy() for p in parameter_sets.values
                ]
                params = self._params_full[0]
            else:  # pragma: no cover - _parse_params only returns these two modes
                raise RuntimeError(
                    f"Unsupported parameter input mode: {parameter_inputs.plan.mode}"
                )

        super().__init__(
            self._data_full.isel({self._y_dim_name: self._current_idx}),
            model,
            params,
            model_name=model_name,
        )

        self.param_model.sigParamsChanged.connect(self._update_params_full)
        self._refresh_contents_from_index()
        self._reset_history_stack()

    def _summary_2d_rows(self) -> list[tuple[str, str]]:
        y_values = self._y_values()
        min_idx = self.y_min_spin.value()
        max_idx = self.y_max_spin.value()
        total_slices = max_idx - min_idx + 1
        fitted_slices = sum(
            ds is not None for ds in self._result_ds_full[self._y_range_slice()]
        )
        param_name = self.param_plot_combo.currentText().strip()

        return [
            (
                "Current slice",
                f"Index {self._current_idx} at {y_values[self._current_idx]}",
            ),
            ("Fit range", f"Indices {min_idx} to {max_idx}"),
            ("Y range", f"{y_values[min_idx]} to {y_values[max_idx]}"),
            ("Coverage", f"{fitted_slices} / {total_slices} slices fit"),
            ("Parameter plot", param_name or "None"),
        ]

    @property
    def info_text(self) -> str:
        from erlab.utils.formatting import format_darr_shape_html

        info = f"<b>{self.tool_name}</b>" + format_darr_shape_html(self.tool_data)
        info += self._summary_section("Setup", self._summary_setup_rows())
        info += self._summary_section("2D Context", self._summary_2d_rows())
        info += self._summary_section("Current Slice Fit", self._summary_fit_rows())
        info += self._summary_section("Current Slice Stats", self._fit_stats_rows())
        return info

    def _init_full_data_state(self, data: xr.DataArray, *, data_name: str) -> None:
        self._data_full: xr.DataArray = data
        self._y_dim_name: Hashable = data.dims[0]
        self._y_values_cache: np.ndarray | None = None
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
        self._fit_2d_initial_range: tuple[int, int] | None = None
        self._fit_2d_last_live_refresh: float = 0.0
        self._fit_2d_live_refresh_pending: bool = False
        self._fit_2d_param_plot_refresh_pending: bool = False
        self._fit_2d_last_completed_idx: int | None = None
        self._fit_2d_last_completed_elapsed: float | None = None
        self._fit_2d_sequence_write_history: bool | None = None

    @staticmethod
    def _data_with_saved_dims(
        data: xr.DataArray, state2d: _State2D | None
    ) -> xr.DataArray:
        if state2d is None or state2d.data_dims_full is None:
            return data
        saved_dims = tuple(state2d.data_dims_full)
        if data.dims == saved_dims:
            return data
        if len(saved_dims) == data.ndim and set(saved_dims) == set(data.dims):
            return data.transpose(*saved_dims)
        return data

    def _rebuild_ui_for_full_data(
        self, data: xr.DataArray, params: lmfit.Parameters | None
    ) -> None:
        old_cw = self.centralWidget()
        if old_cw is not None:
            old_cw.setParent(None)
            old_cw.deleteLater()

        self._init_full_data_state(data, data_name=self._data_name_full)
        self._reset_fit_state(
            self._data_full.isel({self._y_dim_name: self._current_idx}),
            self._model,
            params,
            data_name=self._data_name,
            model_name=self._model_name,
        )
        self._build_ui()
        self.param_model.sigParamsChanged.connect(self._update_params_full)

    def _update_params_full(self) -> None:
        self._params_full[self._current_idx] = self._params
        self._params_from_coord_full[self._current_idx] = self._params_from_coord
        if self._fit_2d_sequence_active():
            self._fit_2d_param_plot_refresh_pending = True
            return
        self._update_param_plot()

    def _sync_fit_result_state(self, *, notify: bool = True) -> None:
        self._params_full[self._current_idx] = self._params
        self._params_from_coord_full[self._current_idx] = self._params_from_coord
        self._result_ds_full[self._current_idx] = self._last_result_ds
        if self._fit_2d_sequence_active():
            self._fit_2d_param_plot_refresh_pending = True
            return
        self._update_param_plot_options()
        self._update_param_plot(notify=notify)

    def _fit_2d_sequence_active(self) -> bool:
        return self._fit_2d_total > 0

    def _fit_2d_live_refresh_due(self) -> bool:
        now = time.monotonic()
        if (
            self._fit_2d_last_live_refresh <= 0.0
            or now - self._fit_2d_last_live_refresh
            >= _FIT2D_SEQUENCE_LIVE_REFRESH_INTERVAL_S
        ):
            self._fit_2d_last_live_refresh = now
            return True
        return False

    def _begin_fit_2d_sequence_history(self) -> None:
        if self._fit_2d_sequence_write_history is not None:
            return
        self._fit_2d_sequence_write_history = self._write_history
        self._write_history = False

    def _finish_fit_2d_sequence_history(self) -> None:
        write_history = self._fit_2d_sequence_write_history
        self._fit_2d_sequence_write_history = None
        if write_history is None:
            return
        self._write_history = write_history
        if write_history:
            self._replace_last_state()

    def _sync_fit_2d_sequence_view(
        self, index: int, *, mark_fit_stale: bool = True, full: bool = True
    ) -> None:
        y_vals = self._y_values()
        curr_val = y_vals[index]
        self._current_idx = int(index)
        with QtCore.QSignalBlocker(self.y_index_spin):
            self.y_index_spin.setValue(self._current_idx)
        self.index_line.setPos(curr_val)
        self.param_plot_index_line.setPos(curr_val)
        self.y_value_spin.setValue(curr_val)
        elapsed = (
            self._fit_2d_last_completed_elapsed
            if index == self._fit_2d_last_completed_idx
            else None
        )
        if not full:
            if not self._fit_2d_live_refresh_pending:
                return
            self._fit_2d_live_refresh_pending = False
            self._refresh_contents_from_index(
                mark_fit_stale=mark_fit_stale,
                elapsed=elapsed,
                emit_info=False,
                emit_param_changed=False,
            )
            self._flush_fit_2d_sequence_param_plot(notify=False)
            return

        self._fit_2d_live_refresh_pending = False
        self._refresh_contents_from_index(
            mark_fit_stale=mark_fit_stale,
            elapsed=elapsed,
        )

    def _flush_fit_2d_sequence_param_plot(
        self, *, force: bool = False, notify: bool = True
    ) -> None:
        if not (force or self._fit_2d_param_plot_refresh_pending):
            return
        self._fit_2d_param_plot_refresh_pending = False
        self._update_param_plot_options()
        self._update_param_plot(notify=notify)

    def _defer_next_fit_step(self, callback: Callable[[], None]) -> None:
        if not self._fit_2d_sequence_active():
            super()._defer_next_fit_step(callback)
            return
        if self._fit_2d_live_refresh_due():
            if self._fit_2d_last_completed_idx is not None:
                self._sync_fit_2d_sequence_view(
                    self._fit_2d_last_completed_idx,
                    mark_fit_stale=False,
                    full=False,
                )
            self._request_fit_step_paint()
        erlab.interactive.utils.single_shot(self, 0, callback)

    def _build_ui(self) -> None:
        super()._build_ui()
        y_vals = self._y_values()

        if not hasattr(self, "_param_plot_overlay_states"):
            self._param_plot_overlay_states: dict[str, bool] = {}
        self._param_plot_overlay_items: dict[
            str, tuple[pg.ErrorBarItem, pg.ScatterPlotItem]
        ] = {}

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
        self.image_plot_legend: pg.LegendItem = self.image_plot.addLegend(offset=(5, 5))
        self.image_plot_legend.setVisible(False)
        if hasattr(self.image_plot_legend, "sigSampleClicked"):  # pragma: no branch
            self.image_plot_legend.sigSampleClicked.connect(
                self._on_image_legend_sample_clicked
            )

        self.param_plot_container = QtWidgets.QWidget()
        param_plot_layout = QtWidgets.QVBoxLayout(self.param_plot_container)
        param_plot_layout.setContentsMargins(0, 0, 0, 0)

        param_plot_controls = QtWidgets.QHBoxLayout()
        param_plot_controls.addWidget(QtWidgets.QLabel("Parameter"))
        self.param_plot_combo = QtWidgets.QComboBox()
        self.param_plot_combo.currentIndexChanged.connect(self._update_param_plot)
        self.param_plot_combo.currentTextChanged.connect(self._emit_info_changed)
        param_plot_controls.addWidget(self.param_plot_combo)

        self.param_plot_widget = pg.GraphicsLayoutWidget()
        self.param_plot_overlay_check = QtWidgets.QCheckBox("Overlay")
        self.param_plot_overlay_check.setToolTip(
            "Overlay the current parameter plot on the image plot."
        )
        self.param_plot_overlay_check.toggled.connect(self._toggle_param_plot_overlay)
        param_plot_controls.addWidget(self.param_plot_overlay_check)
        param_plot_controls.addStretch()
        param_plot_layout.addLayout(param_plot_controls)
        param_plot_layout.addWidget(self.param_plot_widget)
        self.param_plot = _Fit2DParameterPlotItem(self)
        self.param_plot_widget.addItem(self.param_plot, row=0, col=0)
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
        with contextlib.suppress(TypeError):
            self.copy_button.clicked.disconnect(self.copy_code)
        self.copy_button.clicked.connect(self.copy_code_1d)

        self.copy_full_button = QtWidgets.QPushButton("Copy code")
        self.copy_full_button.clicked.connect(self.copy_code)
        self.save_full_button = QtWidgets.QPushButton("Save fit")
        self.save_full_button.clicked.connect(self._save_fit_full)

        self.copy_layout.addWidget(self.copy_full_button, 1, 0)
        self.copy_layout.addWidget(self.save_full_button, 1, 1)

        self.main_splitter.setStretchFactor(0, 3)
        self.main_splitter.setStretchFactor(1, 2)

        self.resize(1024, 610)
        self._update_param_plot_options()
        self._update_param_plot_overlays()

    @property
    def tool_status(self) -> Fit1DTool.StateModel:
        state_dict = super().tool_status.model_dump()
        param_plot_names = set(self._param_plot_names())
        state_dict["state2d"] = _State2D(
            current_idx=int(self._current_idx),
            data_name_full=str(self._data_name_full),
            data_dims_full=tuple(self._data_full.dims),
            params_full=[
                self._serialize_params(params) if params is not None else None
                for params in self._params_full
            ],
            initial_params_full=(
                [self._serialize_params(params) for params in self._initial_params_full]
                if self._initial_params_full is not None
                else None
            ),
            params_from_coord_full=self._params_from_coord_full.copy(),
            fill_mode=typing.cast(
                'typing.Literal["previous", "extrapolate", "none"]',
                self.fill_mode_combo.currentText().lower(),
            ),
            y_limits=(self.y_min_spin.value(), self.y_max_spin.value()),
            param_plot_overlay_states={
                name: checked
                for name, checked in self._param_plot_overlay_states.items()
                if name in param_plot_names
            },
        )
        return self.StateModel(**state_dict)

    @tool_status.setter
    def tool_status(self, status: Fit1DTool.StateModel) -> None:
        state2d = status.state2d
        restored_data = self._data_with_saved_dims(self._data_full, state2d)
        if restored_data.dims != self._data_full.dims:
            with self._history_suppressed():
                self._rebuild_ui_for_full_data(restored_data, self._params.copy())

        repaired_bounds: list[str] = []
        previous_repaired_bounds = self._param_bounds_repair_names
        self._param_bounds_repair_names = repaired_bounds
        try:
            super(Fit2DTool, self.__class__).tool_status.__set__(  # type: ignore[attr-defined]
                self, status
            )
            self._update_param_plot_options()
            if state2d is not None:  # pragma: no branch
                y_size = int(self._data_full.sizes[self._y_dim_name])
                self._data_name_full = state2d.data_name_full
                restored_params_full = [
                    self._deserialize_params(params, repaired_bounds=repaired_bounds)
                    for params in state2d.params_full
                ]
                self._params_full = [None] * y_size
                for i, params in enumerate(restored_params_full[:y_size]):
                    self._params_full[i] = params

                self._initial_params_full = None
                if state2d.initial_params_full is not None:
                    self._initial_params_full = [
                        self._initial_params.copy() for _ in range(y_size)
                    ]
                    for i, params in enumerate(state2d.initial_params_full[:y_size]):
                        restored = self._deserialize_params(
                            params, repaired_bounds=repaired_bounds
                        )
                        if restored is not None:
                            self._initial_params_full[i] = restored

                self._params_from_coord_full = [{} for _ in range(y_size)]
                for i, mapping in enumerate(state2d.params_from_coord_full[:y_size]):
                    self._params_from_coord_full[i] = mapping.copy()

                self.fill_mode_combo.setCurrentText(state2d.fill_mode.capitalize())
                self._apply_param_plot_overlay_states(state2d.param_plot_overlay_states)
                if state2d.y_limits is not None:  # pragma: no branch
                    max_idx = y_size - 1
                    y_min = min(max(state2d.y_limits[0], 0), max_idx)
                    y_max = min(max(state2d.y_limits[1], 0), max_idx)
                    with (
                        QtCore.QSignalBlocker(self.y_min_spin),
                        QtCore.QSignalBlocker(self.y_max_spin),
                    ):
                        self.y_min_spin.setValue(min(y_min, y_max))
                        self.y_max_spin.setValue(max(y_min, y_max))
                    self._y_minmax_changed()
                self._current_idx = min(max(state2d.current_idx, 0), y_size - 1)
        finally:
            self._param_bounds_repair_names = previous_repaired_bounds
        self._show_repaired_parameter_bounds_warning(repaired_bounds)
        self.y_index_spin.setValue(self._current_idx)
        self._update_param_plot_options()
        self._update_param_plot()

    @QtCore.Slot()
    def _transpose(self) -> None:
        """Wrap around _do_transpose with a confirmation dialog."""
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setText("Transpose Data?")
        msg_box.setInformativeText("All fit results and parameters will be lost.")
        msg_box.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.Cancel
        )
        msg_box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Yes)

        if (
            msg_box.exec() != QtWidgets.QMessageBox.StandardButton.Yes
        ):  # pragma: no cover
            return

        self._do_transpose()

    @_rebuild_ui(mark_fresh=False)
    def _do_transpose(self) -> None:
        """Transpose the full 2D data (swap axes)."""
        self._init_full_data_state(
            self._data_full.transpose(), data_name=self._data_name_full
        )

    @QtCore.Slot()
    def _domain_changed(self) -> None:
        super()._domain_changed()
        self.x_min_line.setPos(self.domain_min_line.x())
        self.x_max_line.setPos(self.domain_max_line.x())

    @QtCore.Slot(object)
    def _on_index_line_dragged(self, line: _SnapCursorLine) -> None:
        pos = line.temp_value
        idx = self._nearest_y_index(pos)
        self._set_current_index(idx)
        self._current_idx = self.y_index_spin.value()

    @QtCore.Slot(object)
    def _on_min_line_dragged(self, line: _SnapCursorLine) -> None:
        pos = line.temp_value
        idx = self._nearest_y_index(pos)
        idx = min(int(idx), self.y_max_spin.value())
        self.y_min_spin.setValue(idx)

    @QtCore.Slot(object)
    def _on_max_line_dragged(self, line: _SnapCursorLine) -> None:
        pos = line.temp_value
        idx = self._nearest_y_index(pos)
        idx = max(int(idx), self.y_min_spin.value())
        self.y_max_spin.setValue(idx)

    def _nearest_y_index(self, value: float) -> int:
        y_vals = self._y_values()
        return int(np.abs(y_vals - value).argmin())

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
        old_model = self._model
        super()._refresh_multipeak_model()
        new_model = self._model
        for idx in range(self._data_full.sizes[self._y_dim_name]):
            if idx != self._current_idx:
                prev_params = self._params_full[idx]
                if prev_params is not None:
                    new_params = new_model.make_params()
                    self._merge_params(prev_params, new_params, old_model, new_model)
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
            updated_names = self._param_plot_names()
            self.param_plot_combo.addItems(updated_names)
            for name in updated_names:
                self._param_plot_overlay_states.setdefault(name, False)
            self._remove_unavailable_param_plot_overlay_items(updated_names)
            if prev_param in updated_names:
                self.param_plot_combo.setCurrentText(prev_param)
            self.param_plot_combo.setEnabled(bool(updated_names))
        self._sync_param_plot_overlay_check()
        self._update_param_plot_overlays()

    def _apply_param_plot_overlay_states(self, states: dict[str, bool]) -> None:
        """Apply saved overlay states and refresh overlay UI."""
        self._param_plot_overlay_states = dict(states)
        for name in self._param_plot_names():
            self._param_plot_overlay_states.setdefault(name, False)
        self._sync_param_plot_overlay_check()
        self._update_param_plot_overlays()

    def _remove_unavailable_param_plot_overlay_items(
        self, param_names: list[str]
    ) -> None:
        """Remove visible overlays for parameters without fit-result data."""
        available = set(param_names)
        for name in list(self._param_plot_overlay_items):
            if name in available:
                continue
            errbar, scatter = self._param_plot_overlay_items.pop(name)
            self.image_plot.removeItem(errbar)
            self.image_plot.removeItem(scatter)
            if self.image_plot_legend is not None:
                self.image_plot_legend.removeItem(name)

    def _toggle_param_plot_overlay(self, checked: bool) -> None:
        """Toggle overlay visibility for the currently selected parameter."""
        param_name = self.param_plot_combo.currentText()
        if not param_name:  # pragma: no cover
            return
        self._param_plot_overlay_states[param_name] = checked
        self._update_param_plot_overlays()
        self._write_state()

    def _image_legend_has_name(self, name: str) -> bool:
        if self.image_plot_legend is None:  # pragma: no cover
            return False
        return any(item[1].text == name for item in self.image_plot_legend.items)

    def set_model(
        self,
        model: lmfit.Model,
        *,
        model_load_path: str | None = None,
        merge_params: bool = False,
        reset_params_from_coord: bool = False,
    ) -> None:
        old_model = self._model
        for i in range(self._data_full.sizes[self._y_dim_name]):
            if i == self._current_idx:
                continue

            prev_params: lmfit.Parameters | None = self._params_full[i]
            new_params: lmfit.Parameters | None = None
            if prev_params is not None:
                prev_params = prev_params.copy()
                new_params = model.make_params()
                if merge_params and old_model is not None:
                    self._merge_params(prev_params, new_params, old_model, model)
                self._params_full[i] = new_params

            if reset_params_from_coord:
                self._params_from_coord_full[i] = {}

            for k in list(self._params_from_coord_full[i].keys()):
                if new_params is not None and k not in new_params:
                    self._params_from_coord_full[i].pop(k)

            if self._initial_params_full is not None and new_params is not None:
                self._initial_params_full[i] = new_params.copy()

        super().set_model(
            model,
            model_load_path=model_load_path,
            merge_params=merge_params,
            reset_params_from_coord=reset_params_from_coord,
        )
        self._update_param_plot_options()
        self._update_param_plot()

    @QtCore.Slot()
    @QtCore.Slot(int)
    def _update_param_plot(
        self, _index: int | None = None, *, notify: bool = True
    ) -> None:
        del _index
        param_name = self.param_plot_combo.currentText()
        self.param_plot.setLabel("bottom", param_name)
        plot_y, param_values, param_errors = self._param_plot_data(param_name)
        self.param_plot_errbar.setData(x=param_values, y=plot_y, width=param_errors)
        self.param_plot_scatter.setData(x=param_values, y=plot_y)
        self._sync_param_plot_overlay_check()
        self._update_param_plot_overlays()
        if notify:
            self._notify_data_changed()

    def _on_image_legend_sample_clicked(self, sample, event=None) -> None:
        """Mirror legend-driven visibility changes to error bars and state."""
        item = getattr(sample, "item", None)
        if item is None:
            return
        for name, (errbar, scatter) in self._param_plot_overlay_items.items():
            if scatter is not item:
                continue
            QtCore.QTimer.singleShot(
                0,
                functools.partial(self._sync_overlay_visibility, name, scatter, errbar),
            )
            break

    def _sync_overlay_visibility(
        self,
        name: str,
        scatter: pg.ScatterPlotItem,
        errbar: pg.ErrorBarItem,
    ) -> None:
        """Sync overlay visibility/state after legend toggles."""
        visible = scatter.isVisible()
        errbar.setVisible(visible)
        self._param_plot_overlay_states[name] = visible
        if self.param_plot_combo.currentText() == name:
            self._sync_param_plot_overlay_check(checked=visible)
        self._write_state()

    def _sync_param_plot_overlay_check(self, *, checked: bool | None = None) -> None:
        """Update the overlay checkbox for the current parameter."""
        param_name = self.param_plot_combo.currentText()
        if not param_name:  # pragma: no cover
            with QtCore.QSignalBlocker(self.param_plot_overlay_check):
                self.param_plot_overlay_check.setChecked(False)
                self.param_plot_overlay_check.setEnabled(False)
            return
        if checked is None:  # pragma: no branch
            checked = self._param_plot_overlay_states.get(param_name, False)
        with QtCore.QSignalBlocker(self.param_plot_overlay_check):
            self.param_plot_overlay_check.setChecked(checked)
            self.param_plot_overlay_check.setEnabled(True)

    def _clear_image_plot_legend(self) -> None:
        """Remove all entries from the image plot legend."""
        if self.image_plot_legend is None:  # pragma: no cover
            return
        for item in list(self.image_plot_legend.items):  # pragma: no branch
            self.image_plot_legend.removeItem(item[1].text)
        self.image_plot_legend.setVisible(False)

    def _param_plot_data(
        self,
        param_name: str,
        *,
        y_vals: np.ndarray | None = None,
        params_list: list[lmfit.Parameters | None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not param_name:  # pragma: no cover
            return np.array([]), np.array([]), np.array([])
        if y_vals is None or params_list is None:
            y_range_slice = self._y_range_slice()
            y_vals = self._y_values()[y_range_slice]
            params_list = self._fit_result_params_list(
                self._result_ds_full[y_range_slice]
            )

        plot_y = []
        param_values = []
        param_errors = []
        for y, params in zip(y_vals, params_list, strict=True):
            if (params is None) or (param_name not in params):
                continue
            param = params[param_name]
            plot_y.append(y)
            param_values.append(param.value)
            param_errors.append(param.stderr if param.stderr is not None else 0.0)

        return np.array(plot_y), np.array(param_values), np.array(param_errors)

    @staticmethod
    def _fit_result_params(result_ds: xr.Dataset | None) -> lmfit.Parameters | None:
        if result_ds is None or "modelfit_results" not in result_ds.data_vars:
            return None
        try:
            result = result_ds.modelfit_results.compute().item()
        except Exception:
            logger.warning(
                "Ignoring invalid Fit2D result dataset for parameter plot",
                exc_info=True,
                extra={"suppress_ui_alert": True},
            )
            return None
        params = getattr(result, "params", None)
        if not isinstance(result, lmfit.model.ModelResult) or params is None:
            logger.warning(
                "Ignoring non-ModelResult Fit2D result dataset for parameter plot",
                extra={"suppress_ui_alert": True},
            )
            return None
        try:
            nfev = int(getattr(result, "nfev", 0) or 0)
        except (TypeError, ValueError, OverflowError):
            nfev = 0
        if nfev <= 0:
            logger.debug(
                "Ignoring unfitted Fit2D result dataset for parameter plot",
                extra={"suppress_ui_alert": True},
            )
            return None
        return params

    @classmethod
    def _fit_result_params_list(
        cls, result_datasets: list[xr.Dataset | None]
    ) -> list[lmfit.Parameters | None]:
        return [cls._fit_result_params(result_ds) for result_ds in result_datasets]

    def _param_plot_names(self) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
        for params in self._fit_result_params_list(
            self._result_ds_full[self._y_range_slice()]
        ):
            if params is None:
                continue
            for name in params:
                if name in seen:
                    continue
                seen.add(name)
                names.append(name)
        return names

    def _param_plot_dataarray(
        self, param_name: str, *, stderr: bool = False
    ) -> xr.DataArray:
        plot_y, param_values, param_errors = self._param_plot_data(param_name)
        values = param_errors if stderr else param_values
        kind = "stderr" if stderr else "values"
        return xr.DataArray(
            values,
            coords={self._y_dim_name: plot_y},
            dims=(self._y_dim_name,),
            name=f"{param_name}_{kind}",
        )

    def _show_dataarray_in_itool(
        self,
        data: xr.DataArray,
        *,
        output_id: str | enum.Enum | None = None,
    ) -> None:
        if output_id is None:
            tool = self._launch_detached_output_imagetool(
                data,
                provenance_spec=self.detached_output_imagetool_provenance(data),
            )
        else:
            tool = self._launch_output_imagetool(data, output_id=output_id)
        if tool is not None:
            self._itool = tool

    def _parameter_figure_output_target(
        self,
        param_name: str,
        output: Output,
        data: xr.DataArray,
    ) -> str | None:
        output_id = self._parameter_output_id(output, param_name)
        tool = self._open_output_imagetool(
            data,
            output_id=output_id,
            provenance_spec=self.output_imagetool_provenance(output_id, data),
            prompt_on_reuse=False,
        )
        if tool is None:
            return None
        target = self._output_imagetool_target(output_id)
        return target if isinstance(target, str) else None

    def _parameter_figure_operation(
        self,
        manager: ImageToolManager,
        *,
        param_name: str,
        values_target: str,
        stderr_target: str,
    ) -> FigureOperationState:
        from erlab.interactive._figurecomposer import (
            FigureMethodFamily,
            FigureMethodPlotValueState,
            FigureOperationState,
        )

        values_source = manager._script_input_name_for_node(
            manager._node_for_target(values_target)
        )
        stderr_source = manager._script_input_name_for_node(
            manager._node_for_target(stderr_target)
        )
        return FigureOperationState.method(
            family=FigureMethodFamily.AXES,
            name="errorbar",
            label=param_name,
        ).model_copy(
            update={
                "method_plot_data_mode": "from_data",
                "method_plot_x": FigureMethodPlotValueState(
                    source=values_source,
                    kind="coord",
                    name=str(self._y_dim_name),
                ),
                "method_plot_y": FigureMethodPlotValueState(
                    source=values_source, kind="data"
                ),
                "method_plot_yerr": FigureMethodPlotValueState(
                    source=stderr_source, kind="data"
                ),
                "method_kwargs": {
                    "label": param_name,
                    "linestyle": "none",
                    "marker": "o",
                },
            }
        )

    def _parameter_figure_prompt_operation(
        self, param_name: str
    ) -> FigureOperationState:
        from erlab.interactive._figurecomposer import (
            FigureMethodFamily,
            FigureOperationState,
        )

        return FigureOperationState.method(
            family=FigureMethodFamily.AXES,
            name="errorbar",
            label=param_name,
        )

    def _add_parameter_plot_to_figure(self) -> None:
        current = self.param_plot._current_param_dataarray(stderr=False)
        if current is None:
            return

        manager, parent_uid = self._managed_output_imagetool_parent()
        if manager is None or parent_uid is None:
            self._show_warning(
                "ImageTool Manager required",
                "Open ftool in ImageTool Manager to add parameter plots to Figure "
                "Composer.",
            )
            return

        param_name, values = current
        target_figure = None
        if manager._figure_uids():
            target_figure = manager._prompt_append_figure_target(
                self._parameter_figure_prompt_operation(param_name)
            )
            if target_figure is None:
                return

        stderr = self._param_plot_dataarray(param_name, stderr=True)
        if values.size == 0 or stderr.size == 0:
            self._show_warning(
                "No data available",
                "No parameter values or standard errors are available for the "
                "selected parameter in the current y-range.",
            )
            return

        values_target = self._parameter_figure_output_target(
            param_name, self.Output.PARAMETER_VALUES, values
        )
        stderr_target = self._parameter_figure_output_target(
            param_name, self.Output.PARAMETER_STDERR, stderr
        )
        if values_target is None or stderr_target is None:
            self._show_warning(
                "Could not open parameter data",
                "Could not open the selected parameter values and standard errors "
                "in ImageTool Manager.",
            )
            return

        operation = self._parameter_figure_operation(
            manager,
            param_name=param_name,
            values_target=values_target,
            stderr_target=stderr_target,
        )
        targets = (values_target, stderr_target)
        if target_figure is None:
            manager.create_figure_from_targets(
                targets,
                operation=operation,
                title=f"{param_name} parameter plot",
            )
            return
        figure_uid, axes_selection = target_figure
        if not manager.append_figure_from_targets(
            targets,
            figure_uid=figure_uid,
            axes_selection=axes_selection,
            operation=operation,
        ):
            self._show_warning(
                "Could not add to Figure",
                "Could not add the selected parameter plot to Figure Composer.",
            )

    def _update_param_plot_overlays(self) -> None:
        """Update overlay items and legend for active parameters."""
        if not any(self._param_plot_overlay_states.values()):
            if self._param_plot_overlay_items:
                for errbar, scatter in self._param_plot_overlay_items.values():
                    self.image_plot.removeItem(errbar)
                    self.image_plot.removeItem(scatter)
                self._param_plot_overlay_items = {}
            self._clear_image_plot_legend()
            return

        y_range_slice = self._y_range_slice()
        y_vals = self._y_values()[y_range_slice]
        params_list = self._fit_result_params_list(self._result_ds_full[y_range_slice])
        available_names = set(self._param_plot_names())
        enabled = []
        for name, checked in self._param_plot_overlay_states.items():
            if not checked or name not in available_names:
                items = self._param_plot_overlay_items.pop(name, None)
                if items is not None:
                    errbar, scatter = items
                    self.image_plot.removeItem(errbar)
                    self.image_plot.removeItem(scatter)
                continue
            enabled.append(name)
            self._update_param_plot_overlay_data(
                name, y_vals=y_vals, params_list=params_list
            )

        if self.image_plot_legend is None:
            return
        if not enabled:
            self._clear_image_plot_legend()
            return
        for name in enabled:
            if not self._image_legend_has_name(name):
                scatter = self._param_plot_overlay_items.get(name, (None, None))[1]
                if scatter is not None:
                    self.image_plot_legend.addItem(scatter, name)
        for item in list(self.image_plot_legend.items):
            if item[1].text not in enabled:
                self.image_plot_legend.removeItem(item[1].text)
        self.image_plot_legend.setVisible(True)

    def _update_param_plot_overlay_data(
        self,
        param_name: str,
        *,
        y_vals: np.ndarray | None = None,
        params_list: list[lmfit.Parameters | None] | None = None,
    ) -> None:
        """Create/update overlay items for a single parameter."""
        items = self._param_plot_overlay_items.get(param_name)
        if items is None:
            names = list(self._model.param_names)
            idx = names.index(param_name) if param_name in names else 0
            color = pg.intColor(idx, hues=max(len(names), 1), sat=128)
            errbar = pg.ErrorBarItem(pen=pg.mkPen(color=color, width=1))
            scatter = pg.ScatterPlotItem(
                size=2,
                pen=pg.mkPen(color=color),
                brush=pg.mkBrush(color),
                pxMode=True,
            )
            items = (errbar, scatter)
            self._param_plot_overlay_items[param_name] = items
            self.image_plot.addItem(errbar)
            self.image_plot.addItem(scatter)
        plot_y, param_values, param_errors = self._param_plot_data(
            param_name, y_vals=y_vals, params_list=params_list
        )
        errbar, scatter = items
        errbar.setData(x=param_values, y=plot_y, width=param_errors)
        scatter.setData(x=param_values, y=plot_y)

    def _load_contents_from_index(self) -> None:
        self._data = self._data_full.isel({self._y_dim_name: self._current_idx})

        params = self._params_full[self._current_idx]
        params_from_coord = self._params_from_coord_full[self._current_idx]
        last_ds = self._result_ds_full[self._current_idx]
        self._last_result_ds = last_ds
        if self._initial_params_full is not None:
            self._initial_params = self._initial_params_full[self._current_idx]
        if params is None:
            params = self._initial_params.copy()
        for param_name, param_coord in dict(params_from_coord).items():
            if param_name in params:
                if param_coord in self._data.coords:
                    params[param_name].value = float(self._data[param_coord].values)
            else:
                del params_from_coord[param_name]

        self._params = params
        self._params_from_coord = params_from_coord

    def _refresh_contents_from_index(
        self,
        *,
        mark_fit_stale: bool = True,
        update_widgets: bool = True,
        elapsed: float | None = None,
        emit_info: bool = True,
        emit_param_changed: bool = True,
    ) -> None:
        self._load_contents_from_index()
        if not update_widgets:
            return

        self.param_model.set_params(
            self._params,
            self._params_from_coord,
            emit_changed=emit_param_changed,
        )
        if not emit_param_changed:
            self._update_fit_curve()
            self._refresh_slider_from_model()
        self._populate_data_curve()

        self._data_name = (
            self._data_name_full
            + ".isel("
            + erlab.interactive.utils.format_kwargs(
                {self._y_dim_name: self._current_idx}
            )
            + ")"
        )

        if self._last_result_ds is not None:
            result = self._last_result_ds.modelfit_results.compute().item()
            self._set_fit_stats(result, elapsed=elapsed, emit_info=emit_info)
            if mark_fit_stale:
                self._mark_fit_stale(emit_info=emit_info)
            else:
                self._mark_fit_fresh(emit_info=emit_info)
        else:
            self._set_fit_stats(None, emit_info=emit_info)
            self._mark_fit_stale(emit_info=emit_info)
        if emit_info:
            self._emit_info_changed()

    @_rebuild_ui(mark_fresh=True)
    def _restore_from_fit_dataset(
        self, fit_ds: xr.Dataset, *, model: lmfit.Model | None = None
    ) -> None:
        fit_data = self._extract_fit_data(fit_ds)
        if fit_data.ndim != 2:
            raise ValueError("Fit dataset does not contain 2D data.")

        if "modelfit_results" not in fit_ds.data_vars:
            raise ValueError(
                "Fit dataset is missing the 'modelfit_results' variable containing "
                "the serialized fit results."
            )

        y_dim = fit_data.dims[0]
        results_da = fit_ds["modelfit_results"]
        if results_da.ndim != 1 or results_da.dims[0] != y_dim:
            raise ValueError(
                "Fit dataset result dimensions do not match the data slices."
            )
        if results_da.sizes[y_dim] != fit_data.sizes[y_dim]:
            raise ValueError(
                "Fit dataset contains a different number of fit results than data "
                "slices."
            )

        slice_states: list[tuple[_FitRestoreState, xr.Dataset]] = []
        for idx in range(results_da.sizes[y_dim]):
            slice_ds = fit_ds.isel({y_dim: idx}).copy()
            restore = self._parse_fit_dataset_for_restore(slice_ds, model=model)
            if restore.data.ndim != 1:
                raise ValueError(
                    "Fit dataset slice does not contain a 1D DataArray for fitting."
                )
            slice_states.append((restore, slice_ds))

        base_model = slice_states[0][0].model
        base_param_names = list(base_model.param_names)
        base_indep = list(base_model.independent_vars)
        for restore, _ in slice_states:
            result_model = restore.result.model or restore.model
            if (
                type(result_model) is not type(base_model)
                or list(result_model.param_names) != base_param_names
                or list(result_model.independent_vars) != base_indep
            ):
                raise ValueError(
                    "Fit dataset contains mixed model definitions across slices."
                )

        self._model = base_model
        self._init_full_data_state(fit_data, data_name=self._data_name_full)
        self._current_idx = min(
            self._current_idx, self._data_full.sizes[self._y_dim_name] - 1
        )
        y_size = fit_data.sizes[y_dim]
        self._params_full = [restore.params.copy() for restore, _ in slice_states]
        self._initial_params_full = [
            restore.params.copy() for restore, _ in slice_states
        ]
        self._params_from_coord_full = [{} for _ in range(y_size)]
        self._result_ds_full = [slice_ds for _, slice_ds in slice_states]
        self._fit_is_current = True
        self._update_param_plot_options()
        self._update_param_plot()

    def _append_persistence_payload(self, ds: xr.Dataset) -> xr.Dataset:
        if self._pending_persisted_fit_result_blob is not None:
            ds = ds.copy()
            ds[self._PERSISTED_FIT_RESULT_VAR] = xr.DataArray(
                np.array(self._pending_persisted_fit_result_blob, copy=True),
                dims=(self._PERSISTED_FIT_RESULT_DIM,),
            )
            ds.attrs[self._PERSISTED_FIT_CURRENT_ATTR] = bool(
                self._pending_persisted_fit_is_current
            )
            return ds
        saved_results = [
            result_ds.expand_dims({self._PERSISTED_FIT_INDEX_DIM: [idx]})
            for idx, result_ds in enumerate(self._result_ds_full)
            if result_ds is not None
        ]
        if not saved_results:
            return ds
        sparse = xr.concat(
            saved_results,
            dim=self._PERSISTED_FIT_INDEX_DIM,
            data_vars="all",
            coords="all",
            compat="override",
            join="override",
            combine_attrs="override",
        )
        ds = ds.copy()
        ds[self._PERSISTED_FIT_RESULT_VAR] = xr.DataArray(
            erlab.interactive.utils._serialize_fit_dataset_blob(sparse),
            dims=(self._PERSISTED_FIT_RESULT_DIM,),
        )
        ds.attrs[self._PERSISTED_FIT_CURRENT_ATTR] = bool(self._fit_is_current)
        return ds

    def _restore_persisted_fit_result_blob(
        self, blob: np.ndarray, *, fit_is_current: bool
    ) -> None:
        sparse = _load_lmfit_for_ftool_restore(
            lambda: erlab.interactive.utils._deserialize_fit_dataset_blob(blob)
        )
        y_size = int(self._data_full.sizes[self._y_dim_name])
        self._result_ds_full = [None] * y_size
        for i, index in enumerate(sparse[self._PERSISTED_FIT_INDEX_DIM].values):
            idx = int(index)
            if 0 <= idx < y_size:
                result_ds = sparse.isel(
                    {self._PERSISTED_FIT_INDEX_DIM: i}, drop=True
                ).copy()
                if self._y_dim_name in self._data_full.coords:
                    result_ds = result_ds.assign_coords(
                        {
                            self._y_dim_name: self._data_full.coords[
                                self._y_dim_name
                            ].isel({self._y_dim_name: idx})
                        }
                    )
                self._result_ds_full[idx] = result_ds
        self._refresh_contents_from_index(mark_fit_stale=not fit_is_current)
        self._update_param_plot_options()
        self._update_param_plot()
        self._pending_persisted_fit_result_blob = None
        self._pending_persisted_fit_is_current = False

    def _restore_persistence_payload(self, ds: xr.Dataset) -> None:
        if self._PERSISTED_FIT_RESULT_VAR not in ds:
            return
        blob = np.array(
            ds[self._PERSISTED_FIT_RESULT_VAR].values,
            copy=True,
        )
        fit_is_current = bool(ds.attrs.get(self._PERSISTED_FIT_CURRENT_ATTR, False))
        self._pending_persisted_fit_result_blob = blob
        self._pending_persisted_fit_is_current = fit_is_current
        self._run_or_defer_restore_work(
            lambda: self._restore_persisted_fit_result_blob(
                blob,
                fit_is_current=fit_is_current,
            ),
            key=self._PERSISTED_FIT_RESULT_VAR,
            run_on_show=True,
        )

    @QtCore.Slot()
    def _y_minmax_changed(self) -> None:
        y_vals = self._y_values()
        min_idx = self.y_min_spin.value()
        max_idx = self.y_max_spin.value()
        self.y_min_spin.setMaximum(max_idx)
        self.y_max_spin.setMinimum(min_idx)
        self.y_index_spin.setRange(min_idx, max_idx)
        min_val, max_val = y_vals[min_idx], y_vals[max_idx]
        y_min, y_max = float(np.min(y_vals)), float(np.max(y_vals))
        self.y_min_line.setBounds((y_min, y_max))
        self.y_max_line.setBounds((y_min, y_max))
        low, high = sorted((min_val, max_val))
        self.index_line.setBounds((low, high))
        self.y_min_line.setPos(min_val)
        self.y_max_line.setPos(max_val)
        self._update_full_fit_saveable()
        self._update_param_plot_options()
        self._update_param_plot()
        self._write_state()
        self._emit_info_changed()

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

    def _fill_params_from(
        self, target_index: int, mode=None, *, update_widgets: bool = True
    ) -> None:
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
                self._fill_params_from(
                    target_index, mode="previous", update_widgets=update_widgets
                )
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

        self._refresh_contents_from_index(update_widgets=update_widgets)

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
        self._update_param_plot_options()
        self._update_param_plot()
        self._mark_fit_stale()

    def _set_fit_running(
        self,
        running: bool,
        *,
        multi: bool,
        step: int = 0,
        total: int = 0,
        emit_info: bool = True,
    ) -> None:
        super()._set_fit_running(
            running,
            multi=multi,
            step=step,
            total=total,
            emit_info=emit_info,
        )
        self.fit_down_button.setDisabled(running)
        self.fit_up_button.setDisabled(running)

    def _fit_step_paint_widgets(self) -> tuple[QtWidgets.QWidget, ...]:
        widgets = list(super()._fit_step_paint_widgets())
        for widget in (
            self.param_plot_widget.viewport(),
            self.y_index_spin,
            self.y_value_spin,
            self.fit_down_button,
            self.fit_up_button,
        ):
            if (
                isinstance(widget, QtWidgets.QWidget)
                and erlab.interactive.utils.qt_is_valid(widget)
                and not any(widget is existing for existing in widgets)
            ):
                widgets.append(widget)
        return tuple(widgets)

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
        self._fit_2d_initial_range = (
            self.y_min_spin.value(),
            self.y_max_spin.value(),
        )

        indices = (
            range(start_idx, end_idx - 1, -1)
            if end_idx < start_idx
            else range(start_idx, end_idx + 1, 1)
        )
        self._fit_2d_indices = list(indices)
        self._fit_2d_total = len(self._fit_2d_indices)
        self._fit_2d_direction = direction
        self._fit_2d_start_idx = start_idx
        self._fit_2d_last_live_refresh = time.monotonic()
        self._fit_2d_live_refresh_pending = False
        self._fit_2d_param_plot_refresh_pending = False
        self._fit_2d_last_completed_idx = None
        self._begin_fit_2d_sequence_history()
        if self._fit_2d_indices:
            self._start_next_fit_2d()

    def _prepare_fit_2d_sequence_index(self, idx: int) -> None:
        self._current_idx = int(idx)
        self._refresh_contents_from_index(update_widgets=False)
        if idx != self._fit_2d_start_idx:
            direction = self._fit_2d_direction or "up"
            self._fill_params_from(
                idx - 1 if direction == "up" else idx + 1,
                update_widgets=False,
            )

    def _store_fit_2d_sequence_result(
        self, idx: int, result_ds: xr.Dataset, t0: float
    ) -> lmfit.model.ModelResult:
        self._last_result_ds = result_ds.copy()
        result = self._last_result_ds.modelfit_results.compute().item()
        self._params = result.params.copy()
        self._result_ds_full[idx] = self._last_result_ds
        self._params_full[idx] = self._params.copy()
        self._fit_is_current = True
        self._fit_2d_last_completed_idx = idx
        self._fit_2d_last_completed_elapsed = time.perf_counter() - t0
        self._fit_2d_live_refresh_pending = True
        self._fit_2d_param_plot_refresh_pending = True
        self.sigFitFinished.emit(self._params.copy())
        return result

    def _start_next_fit_2d(self) -> None:
        if self._fit_cancel_requested:
            self._finish_fit_2d_sequence()
            return
        if not self._fit_2d_indices:
            self._finish_fit_2d_sequence()
            return
        if self._fit_2d_initial_range is not None:
            current_range = (
                self.y_min_spin.value(),
                self.y_max_spin.value(),
            )
            if (
                current_range[0] < self._fit_2d_initial_range[0]
                or current_range[1] > self._fit_2d_initial_range[1]
            ):
                self._finish_fit_2d_sequence()
                return

        idx = self._fit_2d_indices.pop(0)
        if idx < self.y_min_spin.value() or idx > self.y_max_spin.value():
            self._finish_fit_2d_sequence()
            return
        self._prepare_fit_2d_sequence_index(idx)

        step = self._fit_2d_total - len(self._fit_2d_indices)

        def _on_success(result_ds: xr.Dataset) -> None:
            if self._fit_start_time is None:
                return
            result = self._store_fit_2d_sequence_result(
                idx, result_ds, self._fit_start_time
            )
            max_nfev = self.nfev_spin.value()
            if max_nfev > 0 and result.nfev >= max_nfev:
                self._show_warning(
                    "Fit Stopped",
                    f"Fit stopped at index {idx} because the maximum number of "
                    "function evaluations was reached.",
                )
                self._finish_fit_2d_sequence()
                return
            self._defer_next_fit_step(self._start_next_fit_2d)

        def _on_timeout() -> None:
            if self._fit_start_time is None:
                return
            self._fit_timed_out(self._fit_start_time)
            self._finish_fit_2d_sequence()

        def _on_error(message) -> None:
            self._fit_errored(message)
            self._finish_fit_2d_sequence()

        try:
            fit_data = self._fit_data()
        except Exception:
            self._fit_start_errored(multi=True)
            self._finish_fit_2d_sequence()
            return

        started = self._start_fit_worker(
            fit_data,
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
        final_idx = self._fit_2d_last_completed_idx
        if final_idx is None and self._data_full.sizes[self._y_dim_name] > 0:
            final_idx = self._current_idx
        if final_idx is not None:
            self._sync_fit_2d_sequence_view(
                final_idx,
                mark_fit_stale=self._result_ds_full[final_idx] is None,
            )
        self._set_fit_running(False, multi=True)
        self._fit_running_multi = False
        self._fit_2d_indices = []
        self._fit_2d_total = 0
        self._fit_2d_direction = None
        self._fit_2d_initial_range = None
        self._fit_2d_last_completed_idx = None
        self._fit_2d_last_completed_elapsed = None
        self._fit_2d_live_refresh_pending = False
        self._update_full_fit_saveable()
        self._flush_fit_2d_sequence_param_plot(force=True)
        self._finish_fit_2d_sequence_history()

    def _y_values(self) -> np.ndarray:
        if self._y_values_cache is not None:
            return self._y_values_cache
        if self._y_dim_name in self._data_full.coords:
            coords = self._data_full.coords[self._y_dim_name]
            self._y_values_cache = np.asarray(coords.values)
        else:
            self._y_values_cache = np.arange(
                self._data_full.sizes[self._y_dim_name], dtype=float
            )
        return self._y_values_cache

    def _refresh_main_image(self) -> None:
        self.image.setDataArray(self._data_full)

    def _mark_fit_stale(self, *, emit_info: bool = True) -> None:
        super()._mark_fit_stale(emit_info=emit_info)
        self.save_full_button.setDisabled(True)
        self.copy_full_button.setDisabled(True)

    def _mark_fit_fresh(self, *, emit_info: bool = True) -> None:
        super()._mark_fit_fresh(emit_info=emit_info)
        self._update_full_fit_saveable()

    def validate_update_data(self, new_data: xr.DataArray) -> xr.DataArray:
        data = erlab.interactive.utils.parse_data(new_data)
        if data.ndim != 2:
            raise ValueError("`data` must be a 2D DataArray")
        return data

    def update_data(self, new_data: xr.DataArray) -> bool:
        had_fit = self._last_result_ds is not None
        status = self.tool_status
        old_geom = self.saveGeometry()

        def _apply_update(validated: xr.DataArray) -> bool:
            validated = self._data_with_saved_dims(validated, status.state2d)
            self._rebuild_ui_for_full_data(validated, self._params.copy())
            with self._history_suppressed():
                self.tool_status = status
            self._refresh_main_image()
            self._refresh_contents_from_index()
            self._update_param_plot()
            self._write_history = True
            self._reset_history_stack()
            self._mark_fit_stale()
            self.restoreGeometry(old_geom)
            self._notify_data_changed()

            if had_fit and self.refit_on_source_update_check.isChecked():
                self._source_refresh_deferred = self.has_source_binding
                self._run_fit()
                return False
            return True

        return self._perform_source_update(new_data, apply_update=_apply_update)

    def _update_full_fit_saveable(self) -> None:
        can_save: bool = self._fit_is_current and not any(
            ds is None for ds in self._result_ds_full[self._y_range_slice()]
        )
        self.save_full_button.setEnabled(can_save)
        self.copy_full_button.setEnabled(can_save)

    @QtCore.Slot()
    def _save_fit_full(self) -> None:
        self._flush_restore_work()
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

    def _full_fit_parameter_specs(
        self, *, warn: bool
    ) -> dict[str, _ModelFitParameterSpec] | None:
        """Collect one consistent parameter mapping across the selected fit range."""

        def _warn_user(title: str, text: str) -> None:
            if warn:
                self._show_warning(title, text)

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
                _warn_user(
                    "Missing Fit",
                    f"No fit result for index {real_index}. Please fit all indices "
                    "in the range before saving the full fit.",
                )
                return None
            for expr_param in params_expr:
                if expr_param in params:
                    this_expr = params[expr_param].expr
                    if this_expr != params_expr[expr_param]:
                        _warn_user(
                            "Inconsistent Parameters",
                            f"Parameter {expr_param!r} has differing expressions "
                            "between fits. Cannot generate combined fit code.",
                        )
                        return None
                    continue
                _warn_user(
                    "Inconsistent Parameters",
                    f"Parameter {expr_param!r} not found in fit at index "
                    f"{real_index}. Cannot generate combined fit code.",
                )
                return None

            valid_params = [k for k in params if params[k].expr is None]
            if valid_params != param_names:
                _warn_user(
                    "Inconsistent Parameters",
                    "Parameter names or counts differ between fits. "
                    "Cannot generate combined fit code.",
                )
                return None
            for name in param_names:
                param = params[name]
                params_value[name].append(param.value)
                params_min[name].append(param.min if param.min is not None else -np.inf)
                params_max[name].append(param.max if param.max is not None else np.inf)
                if name not in params_vary:
                    params_vary[name] = param.vary
                else:
                    if params_vary[name] != param.vary:
                        _warn_user(
                            "Inconsistent Parameters",
                            "Parameter vary flags differ between fits. "
                            "Cannot generate combined fit code.",
                        )
                        return None

        parameter_specs: dict[str, _ModelFitParameterSpec] = {}
        for name in param_names_all:
            if name in params_expr:
                parameter_specs[name] = _ModelFitParameterSpec(expr=params_expr[name])
                continue
            values = params_value[name]
            mins = params_min[name]
            maxs = params_max[name]
            vary: bool = params_vary[name]

            single_value: bool = np.allclose(values, values[0])
            single_min: bool = np.allclose(mins, mins[0])
            single_max: bool = np.allclose(maxs, maxs[0])

            minimum: float | tuple[float, ...] | None
            maximum: float | tuple[float, ...] | None
            if single_min:
                minimum = mins[0] if np.isfinite(mins[0]) else None
            else:
                minimum = tuple(mins)
            if single_max:
                maximum = maxs[0] if np.isfinite(maxs[0]) else None
            else:
                maximum = tuple(maxs)
            parameter_specs[name] = _ModelFitParameterSpec(
                value=values[0] if single_value else tuple(values),
                minimum=minimum,
                maximum=maximum,
                vary=vary,
            )
        return parameter_specs

    def _build_full_copy_prelude(
        self, *, warn: bool = True, input_name: str | None = None
    ) -> str:
        data_name, _model_name, lines = self._make_model_code(
            input_name or self._data_name_full
        )
        parameters = self._full_fit_parameter_specs(warn=warn)
        if parameters is None:
            return ""

        isel_kw = erlab.interactive.utils.format_kwargs(
            {self._y_dim_name: self._y_range_slice()}
        )
        data_name = self._full_copy_fit_data_name(
            data_name,
            isel_kw=isel_kw,
            lines=lines,
        )
        parameters_code = _model_fit_parameters_code(
            parameters,
            input_name=data_name,
            broadcast_dim=self._y_dim_name,
        )
        lines.append(f"params = {parameters_code}")
        return "\n".join(lines)

    def _full_copy_fit_data_name(
        self,
        data_name: str,
        *,
        isel_kw: str | None = None,
        lines: list[str] | None = None,
    ) -> str:
        if isel_kw is None:
            isel_kw = erlab.interactive.utils.format_kwargs(
                {self._y_dim_name: self._y_range_slice()}
            )

        fit_domain = self._fit_domain()
        if fit_domain is not None:
            sel_kw = erlab.interactive.utils.format_kwargs(
                {self._coord_name: slice(*fit_domain)}
            )
            if lines is not None:
                lines.append(
                    f"{data_name}_crop = {data_name}.sel({sel_kw}).isel({isel_kw})"
                )
        else:
            if lines is not None:
                lines.append(f"{data_name}_crop = {data_name}.isel({isel_kw})")
        data_name = f"{data_name}_crop"

        if self.normalize_check.isChecked():
            if lines is not None:
                lines.append(
                    f"{data_name}_norm = "
                    f'{data_name} / {data_name}.mean("{self._coord_name}")'
                )
            data_name = f"{data_name}_norm"

        return data_name

    def _full_fit_expression(
        self,
        *,
        input_name: str | None = None,
        data: xr.DataArray | None = None,
    ) -> str:
        data_name, model_name, _ = self._make_model_code(
            input_name or self._data_name_full
        )
        data_name = self._full_copy_fit_data_name(data_name)
        return erlab.interactive.utils.generate_code(
            self._data_full.xlm.modelfit,
            args=[self._coord_name],
            kwargs={
                "model": f"|{model_name}|",
                "params": "|params|",
                "method": self.method_combo.currentText(),
            },
            name="modelfit",
            module=f"{data_name}.xlm",
        )

    def _checked_full_copy_prelude(
        self,
        *,
        input_name: str | None = None,
        data: xr.DataArray | None = None,
    ) -> str | None:
        prelude = self._build_full_copy_prelude(input_name=input_name, warn=True)
        return prelude or None

    def _detached_full_copy_prelude(
        self,
        *,
        input_name: str | None = None,
        data: xr.DataArray | None = None,
    ) -> str | None:
        prelude = self._build_full_copy_prelude(input_name=input_name, warn=False)
        return prelude or None

    def current_provenance_spec(
        self, *, flush_deferred_restore: bool = True
    ) -> ToolProvenanceSpec | None:
        # Manager metadata and other passive provenance consumers should not trigger
        # interactive warnings for incomplete fit ranges.
        return self._resolve_script_provenance(
            self._DETACHED_COPY_PROVENANCE,
            flush_deferred_restore=flush_deferred_restore,
        )

    @QtCore.Slot()
    def copy_code(self) -> str:
        return self._copy_provenance_code(
            self._resolve_script_provenance(
                self.COPY_PROVENANCE,
                include_parent_provenance=False,
            )
        )

    @QtCore.Slot()
    def _copy_code_full(self) -> str:
        return self.copy_code()

    @QtCore.Slot()
    def copy_code_1d(self) -> str:
        return self._copy_provenance_code(
            self._resolve_script_provenance(
                Fit1DTool.COPY_PROVENANCE,
                include_parent_provenance=False,
            )
        )

    def detached_output_imagetool_provenance(
        self,
        data: xr.DataArray,
        *,
        source: QtCore.QObject | None = None,
    ) -> ToolProvenanceSpec | None:
        if source is None:
            return self._resolve_script_provenance(self._DETACHED_COPY_PROVENANCE)
        return super().detached_output_imagetool_provenance(
            data,
            source=source,
        )

    def _current_param_output(self, *, stderr: bool) -> tuple[str, xr.DataArray] | None:
        param_name = self.param_plot_combo.currentText().strip()
        if not param_name or param_name not in self._param_plot_names():
            return None
        return param_name, self._param_plot_dataarray(param_name, stderr=stderr)

    def _resolve_parameter_output(
        self, output: Output, param_name: str
    ) -> tuple[str, bool] | None:
        stderr = output == self.Output.PARAMETER_STDERR
        if not param_name or param_name not in self._param_plot_names():
            return None
        return param_name, stderr

    def output_imagetool_data(self, output_id: str | enum.Enum) -> xr.DataArray | None:
        self._flush_restore_work()
        parts = self._parameter_output_parts(output_id)
        if parts is None:
            return super().output_imagetool_data(output_id)
        output, param_name = parts
        if param_name is None:
            return super().output_imagetool_data(output)

        request = self._resolve_parameter_output(output, param_name)
        if request is None:
            return None
        param_name, stderr = request
        return self._param_plot_dataarray(param_name, stderr=stderr)

    def output_imagetool_provenance(
        self, output_id: str | enum.Enum, data: xr.DataArray
    ) -> ToolProvenanceSpec | None:
        self._flush_restore_work()
        parts = self._parameter_output_parts(output_id)
        if parts is None:
            return super().output_imagetool_provenance(output_id, data)
        output, param_name = parts
        if param_name is None:
            current = self._current_param_output(
                stderr=output == self.Output.PARAMETER_STDERR
            )
            if current is None:
                return None
            param_name = current[0]

        request = self._resolve_parameter_output(output, param_name)
        if request is None:
            return None
        param_name, stderr = request
        return self._parameter_output_provenance(param_name, stderr=stderr, data=data)

    def _parameter_model_fit_operation(
        self, param_name: str, *, stderr: bool
    ) -> ModelFitOperation | None:
        model_choice = self._infer_model_choice(self._model)
        if model_choice not in ModelFitOperation.supported_models:
            return None
        parameters = self._full_fit_parameter_specs(warn=False)
        if parameters is None:
            return None
        registry = self._model_option_registry()
        model_kwargs = (
            registry[model_choice]["kwargs"]() if model_choice in registry else {}
        )
        return ModelFitOperation(
            fit_dim=self._coord_name,
            model=model_choice,
            model_kwargs=model_kwargs,
            parameters=parameters,
            method=self.method_combo.currentText(),
            parameter=param_name,
            output="stderr" if stderr else "value",
            broadcast_dim=self._y_dim_name,
            normalize=self.normalize_check.isChecked(),
        )

    def _parameter_output_script_operation(
        self,
        param_name: str,
        *,
        stderr: bool,
        input_name: str | None,
    ) -> ScriptCodeOperation | None:
        """Build public replay code for models outside the structured catalog."""
        prelude = self._build_full_copy_prelude(
            warn=False,
            input_name=input_name,
        )
        fit_expression = self._full_fit_expression(input_name=input_name)
        if not prelude or not fit_expression:
            return None
        if "lmfit." in prelude:
            prelude = f"import lmfit\n\n{prelude}"

        assign = "parameter_stderr" if stderr else "parameter_values"
        result_variable = "modelfit_stderr" if stderr else "modelfit_coefficients"
        output_name = f"{param_name}_{'stderr' if stderr else 'values'}"
        output_expression = (
            f"({fit_expression}).{result_variable}.sel(param={param_name!r}, drop=True)"
        )
        if stderr:
            output_expression += ".fillna(0.0)"
        output_expression += f".rename({output_name!r})"
        output_label = "standard errors" if stderr else "values"
        return ScriptCodeOperation(
            label=f"Fit model and extract {param_name!r} parameter {output_label}",
            code=f"{prelude}\n{assign} = {output_expression}",
        )

    def _parameter_output_provenance(
        self,
        param_name: str,
        *,
        stderr: bool,
        data: xr.DataArray,
    ) -> ToolProvenanceSpec | None:
        del data
        input_provenance = self._effective_input_provenance_spec()
        model_fit = self._parameter_model_fit_operation(param_name, stderr=stderr)
        assign = "parameter_stderr" if stderr else "parameter_values"
        if model_fit is not None:
            operations: list[ToolProvenanceOperation] = []
            fit_domain = self._fit_domain()
            if fit_domain is not None:
                operations.append(
                    SelOperation(kwargs={self._coord_name: slice(*fit_domain)})
                )
            operations.extend(
                (
                    IselOperation(kwargs={self._y_dim_name: self._y_range_slice()}),
                    model_fit,
                )
            )
            local_spec = script(
                *operations,
                start_label="Start from current fit-tool input data",
                seed_code=(
                    "derived = data"
                    if input_provenance is not None
                    else f"derived = {self._data_name_full}"
                ),
                active_name=assign,
            )
            return compose_full_provenance(input_provenance, local_spec)

        input_name = replay_input_name(input_provenance) or self._data_name_full
        fallback = self._parameter_output_script_operation(
            param_name,
            stderr=stderr,
            input_name=input_name,
        )
        if fallback is None:
            return None
        local_spec = script(
            fallback,
            start_label="Start from current fit-tool input data",
            active_name=assign,
        )
        if (
            input_provenance is not None
            and direct_replay_input_name(input_provenance) is not None
        ):
            replay_spec = to_replay_provenance_spec(local_spec)
            if replay_spec is None:
                raise RuntimeError("Could not convert local provenance to replay spec.")
            return replay_spec.model_copy(
                update={"start_label": typing.cast("str", input_provenance.start_label)}
            )
        return compose_full_provenance(input_provenance, local_spec)

    def _parameter_values_output_data(self) -> xr.DataArray | None:
        current = self._current_param_output(stderr=False)
        if current is None:
            return None
        return current[1]

    def _parameter_stderr_output_data(self) -> xr.DataArray | None:
        current = self._current_param_output(stderr=True)
        if current is None:
            return None
        return current[1]


def ftool(
    data: xr.DataArray | xr.Dataset,
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
        The 1D or 2D data to fit. Also accepted is a fit result :class:`xarray.Dataset`,
        from which the data to fit will be extracted. In this case, the tool will
        attempt to restore the fit state from the dataset.
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
    fit_ds: xr.Dataset | None = None
    if isinstance(data, xr.Dataset):
        fit_ds = data
        data = Fit1DTool._extract_fit_data(fit_ds)
        if data_name is None:
            data_name = "data"
        params = None
    elif data_name is None:
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
        if fit_ds is not None:
            try:
                win._restore_from_fit_dataset(fit_ds, model=model)
            except Exception:
                win.close()
                raise
        win.show()
        win.raise_()
        win.activateWindow()
    return win
