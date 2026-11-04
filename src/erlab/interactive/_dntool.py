"""Interactive denoising of angle-energy data."""

from __future__ import annotations

import enum
import typing
import weakref
from pathlib import Path

import pydantic
import pyqtgraph as pg
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive import _persistence_constants
from erlab.interactive.imagetool._provenance._operations import (
    DenoiseOperation,
    TransposeOperation,
)

if typing.TYPE_CHECKING:
    from collections.abc import Mapping

    import varname

    from erlab.interactive.imagetool._provenance._model import (
        ToolProvenanceOperation,
        ToolProvenanceSpec,
    )
else:
    import lazy_loader as _lazy

    varname = _lazy.load("varname")

__all__ = ["dntool"]


class _Parameters(pydantic.BaseModel):
    resize: tuple[int, int] = (300, 300)
    model_path: str | None = None
    tta: bool = True
    restore_scale: bool = True
    restore_size: bool = True
    num_threads: int = 0


class _DenoiseImageItem(erlab.interactive.utils.xImageItem):
    def __init__(self) -> None:
        super().__init__(axisOrder="row-major")
        self._source_data: xr.DataArray | None = None

    def set_source_data(self, data: xr.DataArray) -> None:
        self._source_data = data
        self.setDataArray(data.transpose("eV", ...))

    def _imagetool_data(self) -> xr.DataArray | None:
        return None if self._source_data is None else self._source_data.T

    def clear(self) -> None:
        self._source_data = None
        super().clear()


class _DenoiseSignals(QtCore.QObject):
    finished = QtCore.Signal(object, object, object)


class _DenoiseTask(QtCore.QRunnable):
    def __init__(
        self,
        tool: DenoiseTool,
        data: xr.DataArray,
        operation: DenoiseOperation,
        parameters: _Parameters,
        *,
        update_input: bool = False,
        restore_history: bool = False,
    ) -> None:
        super().__init__()
        self.setAutoDelete(False)
        self.tool_ref = weakref.ref(tool)
        self.data = data
        self.operation = operation
        self.parameters = parameters
        self.update_input = update_input
        self.restore_history = restore_history
        self.signals = _DenoiseSignals()

    @QtCore.Slot()
    def run(self) -> None:
        result: xr.DataArray | None = None
        error: Exception | None = None
        try:
            result = self.operation.apply(self.data)
            if result.chunks is not None:
                result = result.compute()
        except Exception as exc:
            error = exc
        self.signals.finished.emit(self, result, error)


class _DenoiseCoordinator(QtCore.QObject):
    """Keep workers alive after their tool window closes."""

    def __init__(self) -> None:
        super().__init__()
        self._pool = QtCore.QThreadPool()
        self._pool.setStackSize(erlab.interactive.utils._python_thread_stack_size())
        self._tasks: set[_DenoiseTask] = set()

    def start(self, task: _DenoiseTask) -> None:
        task.signals.finished.connect(
            self._finish, QtCore.Qt.ConnectionType.QueuedConnection
        )
        self._tasks.add(task)
        try:
            self._pool.start(task)
        except Exception:
            self._tasks.discard(task)
            raise

    @QtCore.Slot(object, object, object)
    def _finish(
        self,
        task: _DenoiseTask,
        result: xr.DataArray | None,
        error: Exception | None,
    ) -> None:
        self._tasks.discard(task)
        tool = task.tool_ref()
        if tool is not None and erlab.interactive.utils.qt_is_valid(tool):
            tool._finish_denoise(task, result, error)


_DENOISE_COORDINATOR: _DenoiseCoordinator | None = None


def _denoise_coordinator() -> _DenoiseCoordinator:
    global _DENOISE_COORDINATOR
    if _DENOISE_COORDINATOR is None or not erlab.interactive.utils.qt_is_valid(
        _DENOISE_COORDINATOR
    ):
        _DENOISE_COORDINATOR = _DenoiseCoordinator()
    return _DENOISE_COORDINATOR


class DenoiseTool(erlab.interactive.utils.ToolWindow):
    """Denoise a two-dimensional angle-energy image."""

    tool_name = "dntool"
    COPY_PROVENANCE: typing.ClassVar = (
        erlab.interactive.utils.ToolScriptProvenanceDefinition(
            start_label="Start from current dntool input data",
            operations_method="_result_provenance",
            active_name="result",
            seed_code_method="_provenance_seed_code",
        )
    )

    class Output(enum.StrEnum):
        RESULT = "dntool.result"

    IMAGE_TOOL_OUTPUTS: typing.ClassVar = {
        Output.RESULT: erlab.interactive.utils.ToolImageOutputDefinition(
            data_method="_result_output_data",
            provenance=erlab.interactive.utils.ToolScriptProvenanceDefinition(
                start_label="Start from current dntool input data",
                operations_method="_result_provenance",
                active_name="result",
                seed_code_method="_provenance_seed_code",
            ),
        )
    }

    class StateModel(pydantic.BaseModel):
        data_name: str = "data"
        parameters: _Parameters = pydantic.Field(default_factory=_Parameters)
        applied: _Parameters | None = None

    def __init__(self, data: xr.DataArray, *, data_name: str | None = None) -> None:
        super().__init__()
        self.data_name = erlab.interactive.utils._tool_window_argname(
            data_name, "data", func=DenoiseTool.__init__, fallback="data"
        )
        self._data = self._validate_data(data)
        self._result: xr.DataArray | None = None
        self._applied: _Parameters | None = None
        self._model_path: str | None = None
        self._busy = False
        self._failed = False
        self._denoise_task: _DenoiseTask | None = None

        root = QtWidgets.QWidget(self)
        layout = QtWidgets.QHBoxLayout(root)
        self.setCentralWidget(root)
        control_panel = QtWidgets.QWidget()
        control_panel.setFixedWidth(290)
        controls = QtWidgets.QVBoxLayout(control_panel)
        controls.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(control_panel)

        grid_group = QtWidgets.QGroupBox("Processing grid")
        grid_layout = QtWidgets.QVBoxLayout(grid_group)
        grid_form = QtWidgets.QFormLayout()
        grid_layout.addLayout(grid_form)
        self.energy_spin = QtWidgets.QSpinBox()
        self.angle_spin = QtWidgets.QSpinBox()
        for spin in (self.energy_spin, self.angle_spin):
            spin.setRange(1, 10000)
            spin.setValue(300)
        grid_form.addRow("Energy pixels", self.energy_spin)
        grid_form.addRow("Angle pixels", self.angle_spin)
        grid_footer = QtWidgets.QHBoxLayout()
        self.input_size_label = QtWidgets.QLabel()
        self.reset_grid_btn = QtWidgets.QPushButton("Reset to 300 × 300")
        grid_footer.addWidget(self.input_size_label)
        grid_footer.addWidget(self.reset_grid_btn)
        grid_layout.addLayout(grid_footer)
        controls.addWidget(grid_group)

        options_group = QtWidgets.QGroupBox("Options")
        options_layout = QtWidgets.QVBoxLayout(options_group)
        self.tta_check = QtWidgets.QCheckBox("Average over 8 orientations")
        self.tta_check.setToolTip(
            "Average predictions over eight flips and transposes. "
            "This can reduce artifacts but takes about eight times as long."
        )
        self.tta_check.setChecked(True)
        self.scale_check = QtWidgets.QCheckBox("Restore intensity scale")
        self.scale_check.setToolTip(
            "Undo the normalization to a total intensity of 10,000 for each slice. "
            "This does not force the output total to equal the input total."
        )
        self.scale_check.setChecked(True)
        self.size_check = QtWidgets.QCheckBox("Restore original grid")
        self.size_check.setToolTip(
            "Rebin the result to the input grid and keep the original coordinates. "
            "When disabled, return the result on the processing grid."
        )
        self.size_check.setChecked(True)
        for check in (self.tta_check, self.scale_check, self.size_check):
            options_layout.addWidget(check)
        thread_form = QtWidgets.QFormLayout()
        options_layout.addLayout(thread_form)
        self.threads_spin = QtWidgets.QSpinBox()
        self.threads_spin.setRange(0, 1024)
        self.threads_spin.setSpecialValueText("Automatic")
        thread_form.addRow("# of threads", self.threads_spin)
        controls.addWidget(options_group)

        model_group = QtWidgets.QGroupBox("Model")
        model_layout = QtWidgets.QVBoxLayout(model_group)
        self.model_summary = QtWidgets.QLabel()
        self.model_summary.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.model_summary.setAccessibleName("Selected model")
        model_layout.addWidget(self.model_summary)
        model_buttons = QtWidgets.QHBoxLayout()
        self.choose_model_btn = QtWidgets.QPushButton("Choose model…")
        self.use_bundled_model_btn = QtWidgets.QPushButton("Use bundled model")
        model_buttons.addWidget(self.choose_model_btn)
        model_buttons.addWidget(self.use_bundled_model_btn)
        model_layout.addLayout(model_buttons)
        controls.addWidget(model_group)
        self._update_model_display()

        controls.addStretch(1)
        self.status_label = QtWidgets.QLabel()
        self.status_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        controls.addWidget(self.status_label)
        self.run_btn = QtWidgets.QPushButton("Run denoise")
        self.open_btn = QtWidgets.QPushButton("Open in ImageTool")
        self.copy_btn = QtWidgets.QPushButton("Copy Code")
        controls.addWidget(self.run_btn)
        output_buttons = QtWidgets.QHBoxLayout()
        output_buttons.addWidget(self.open_btn)
        output_buttons.addWidget(self.copy_btn)
        controls.addLayout(output_buttons)

        graphics = pg.GraphicsLayoutWidget()
        layout.addWidget(graphics, 1)
        self.images = tuple(_DenoiseImageItem() for _ in range(2))
        self.plots: list[pg.PlotItem] = []
        self.colorbars: list[erlab.interactive.colors.BetterColorBarItem] = []
        opts = erlab.interactive.options.model
        for row, image in enumerate(self.images):
            plot = graphics.addPlot(row=row, col=0)
            plot.addItem(image)
            if row == 0:
                image.set_source_data(self._data)
            image.set_colormap(
                opts.colors.cmap.name, gamma=0.5, reverse=opts.colors.cmap.reverse
            )
            colorbar = erlab.interactive.colors.BetterColorBarItem(image=image)
            colorbar.setPreferredWidth(60)
            graphics.addItem(colorbar, row=row, col=1)
            self._register_plot_appearance(("input", "result")[row], colorbar)
            self.plots.append(plot)
            self.colorbars.append(colorbar)
        self.plots[0].setXLink(self.plots[1])
        self.plots[0].setYLink(self.plots[1])

        self.open_btn.setEnabled(False)
        self.copy_btn.setEnabled(False)
        self.reset_grid_btn.clicked.connect(self._reset_processing_grid)
        self.run_btn.clicked.connect(self.denoise)
        self.open_btn.clicked.connect(self.open_itool)
        self.copy_btn.clicked.connect(self.copy_code)
        for spin in (self.energy_spin, self.angle_spin, self.threads_spin):
            spin.valueChanged.connect(self._parameters_changed)
        for check in (self.tta_check, self.scale_check, self.size_check):
            check.toggled.connect(self._parameters_changed)
        self.choose_model_btn.clicked.connect(self._choose_model)
        self.use_bundled_model_btn.clicked.connect(self._use_bundled_model)
        self.plots[0].autoRange()
        self._update_input_size()
        self._update_status()
        self._reset_history_stack()

    def sizeHint(self) -> QtCore.QSize:
        return super().sizeHint().expandedTo(QtCore.QSize(0, 800))

    def _update_input_size(self) -> None:
        along = next(dim for dim in self._data.dims if dim != "eV")
        self.input_size_label.setText(
            f"Input {self._data.sizes['eV']} × {self._data.sizes[along]}"
        )

    def _update_status(self, *_args: object) -> None:
        if self._busy:
            color, message = "#65a9e9", "Denoising…"
        elif self._failed:
            color, message = "#e47878", "Denoising failed"
        elif self._applied is None:
            color, message = "#a8a8a8", "Ready to denoise"
        elif self._result is None:
            color, message = "#a8a8a8", "Result pending"
        elif self.parameters != self._applied:
            color, message = "#e6ac56", "Settings changed · run denoise"
        else:
            passes = "8 forward passes" if self._applied.tta else "1 forward pass"
            color, message = "#75c79a", f"Up to date · {passes}"
        self.status_label.setText(f'<span style="color: {color}">●</span> {message}')

    def _parameters_changed(self, *_args: object) -> None:
        self._failed = False
        self._update_status()
        self._write_state()

    def _update_model_display(self) -> None:
        if self._model_path is None:
            name = "Bundled model"
            description = "Use the model included with ERLabPy."
        else:
            name = Path(self._model_path).name
            description = QtCore.QDir.toNativeSeparators(self._model_path)
        self.model_summary.setText(
            self.model_summary.fontMetrics().elidedText(
                name, QtCore.Qt.TextElideMode.ElideMiddle, 240
            )
        )
        self.model_summary.setToolTip(description)
        self.model_summary.setAccessibleDescription(description)
        self.use_bundled_model_btn.setVisible(self._model_path is not None)

    @QtCore.Slot()
    def _choose_model(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose ONNX model",
            self._model_path or "",
            "ONNX models (*.onnx);;All files (*)",
        )
        if not path or path == self._model_path:
            return
        self._model_path = path
        self._update_model_display()
        self._parameters_changed()

    @QtCore.Slot()
    def _use_bundled_model(self) -> None:
        if self._model_path is None:
            return
        self._model_path = None
        self._update_model_display()
        self._parameters_changed()

    @QtCore.Slot()
    def _reset_processing_grid(self) -> None:
        if self.energy_spin.value() == self.angle_spin.value() == 300:
            return
        with (
            QtCore.QSignalBlocker(self.energy_spin),
            QtCore.QSignalBlocker(self.angle_spin),
        ):
            self.energy_spin.setValue(300)
            self.angle_spin.setValue(300)
        self._parameters_changed()

    @staticmethod
    def _validate_data(data: object) -> xr.DataArray:
        if (
            not isinstance(data, xr.DataArray)
            or data.ndim != 2
            or data.dims.count("eV") != 1
        ):
            raise ValueError("Input must be a 2D DataArray with an 'eV' dimension")
        if not isinstance(next(dim for dim in data.dims if dim != "eV"), str):
            raise TypeError("The other dimension must have a string name")
        return data

    @property
    def tool_data(self) -> xr.DataArray:
        return self._data

    @property
    def preview_imageitem(self) -> pg.ImageItem:
        return self.images[1] if self._result is not None else self.images[0]

    @property
    def parameters(self) -> _Parameters:
        return _Parameters(
            resize=(self.energy_spin.value(), self.angle_spin.value()),
            model_path=self._model_path,
            tta=self.tta_check.isChecked(),
            restore_scale=self.scale_check.isChecked(),
            restore_size=self.size_check.isChecked(),
            num_threads=self.threads_spin.value(),
        )

    @property
    def tool_status(self) -> StateModel:
        return self.StateModel(
            data_name=self.data_name, parameters=self.parameters, applied=self._applied
        )

    @tool_status.setter
    def tool_status(self, status: StateModel) -> None:
        reuse_result = (
            status.applied is not None
            and self._result is not None
            and self._applied == status.applied
        )
        self._invalidate_denoise()
        self._discard_restore_work(self._recompute_applied)
        self.data_name = status.data_name
        widgets = (
            self.energy_spin,
            self.angle_spin,
            self.tta_check,
            self.scale_check,
            self.size_check,
            self.threads_spin,
        )
        blockers = [QtCore.QSignalBlocker(widget) for widget in widgets]
        params = status.parameters
        self.energy_spin.setValue(params.resize[0])
        self.angle_spin.setValue(params.resize[1])
        self._model_path = params.model_path or None
        self._update_model_display()
        self.tta_check.setChecked(params.tta)
        self.scale_check.setChecked(params.restore_scale)
        self.size_check.setChecked(params.restore_size)
        self.threads_spin.setValue(params.num_threads)
        del blockers
        self._failed = False
        if self._restoring_from_dataset:
            # The persistence hook will restore the completed result or schedule the
            # legacy fallback once the saved data items are available.
            self._applied = status.applied
            self._clear_result()
        elif reuse_result:
            self._update_status()
        else:
            self._applied = None
            self._clear_result()
            if status.applied is not None:
                try:
                    self._start_denoise(
                        status.applied, self._data, restore_history=True
                    )
                except Exception as exc:
                    self._show_denoise_error(exc)

    def _operation(
        self, params: _Parameters, data: xr.DataArray | None = None
    ) -> DenoiseOperation:
        if data is None:
            data = self._data
        along = typing.cast("str", next(dim for dim in data.dims if dim != "eV"))
        return DenoiseOperation(along=along, **params.model_dump())

    def _calculate(
        self, params: _Parameters, data: xr.DataArray | None = None
    ) -> xr.DataArray:
        if data is None:
            data = self._data
        result = self._operation(params, data).apply(data)
        return result.compute() if result.chunks is not None else result

    def _show_result(self, result: xr.DataArray, *, notify: bool = True) -> None:
        self._result = result
        self.images[1].set_source_data(result)
        self.colorbars[1].reset_levels()
        self.images[1].setLevels(self.colorbars[1].limits)
        self.open_btn.setEnabled(True)
        self.copy_btn.setEnabled(True)
        self._failed = False
        self._update_status()
        if notify:
            self._notify_data_changed()

    def _clear_result(self, *, notify: bool = True) -> None:
        self._result = None
        self.images[1].clear()
        self.open_btn.setEnabled(False)
        self.copy_btn.setEnabled(False)
        self._update_status()
        if notify:
            self._notify_data_changed()

    def _recompute_applied(self) -> None:
        if self._applied is not None:
            self._show_result(self._calculate(self._applied))

    def _invalidate_denoise(self, *, abort_refresh: bool = True) -> None:
        if self._denoise_task is None:
            return
        updating_input = self._denoise_task.update_input
        self._denoise_task = None
        self._busy = False
        self.run_btn.setEnabled(True)
        self._update_status()
        if abort_refresh and updating_input and self._source_refresh_deferred:
            # State replacement must finish before any newer refresh can start.
            self.abort_source_refresh(resume_pending=False)

    def _start_denoise(
        self,
        params: _Parameters,
        data: xr.DataArray,
        *,
        update_input: bool = False,
        restore_history: bool = False,
    ) -> None:
        task = _DenoiseTask(
            self,
            data,
            self._operation(params, data),
            params,
            update_input=update_input,
            restore_history=restore_history,
        )
        self._denoise_task = task
        self._busy = True
        self._failed = False
        self.run_btn.setEnabled(False)
        self._update_status()
        try:
            _denoise_coordinator().start(task)
        except Exception:
            self._invalidate_denoise(abort_refresh=False)
            raise

    def _show_denoise_error(self, error: Exception) -> None:
        self._failed = True
        self._update_status()
        QtWidgets.QMessageBox.critical(self, "Denoising failed", str(error))

    @QtCore.Slot()
    def denoise(self) -> None:
        if self._denoise_task is not None:
            return
        try:
            self._start_denoise(self.parameters, self._data)
        except Exception as exc:
            self._show_denoise_error(exc)

    def _finish_denoise(
        self,
        task: _DenoiseTask,
        result: xr.DataArray | None,
        error: Exception | None,
    ) -> None:
        if task is not self._denoise_task:
            return
        deferred_refresh = task.update_input and self._source_refresh_deferred
        self._invalidate_denoise(abort_refresh=False)
        if error is not None or result is None:
            if deferred_refresh:
                self.abort_source_refresh()
            if task.restore_history:
                self._replace_last_state()
            self._show_denoise_error(
                error
                if error is not None
                else RuntimeError("Denoising returned no result")
            )
            return
        if task.update_input:
            self._set_input_data(task.data)
        self._applied = task.parameters
        self._show_result(result, notify=not deferred_refresh)
        if task.update_input:
            self._reset_history_stack()
        elif task.restore_history:
            self._replace_last_state()
        else:
            self._write_state()
        if deferred_refresh:
            self.finalize_source_refresh()

    def _persistence_data_items(self) -> Mapping[str, xr.DataArray]:
        items = dict(super()._persistence_data_items())
        if self._applied is not None and self._result is not None:
            items["denoised"] = self._result
        return items

    def _restore_persistence_data_items(
        self, data_items: Mapping[str, xr.DataArray], ds: xr.Dataset
    ) -> None:
        del ds
        self._invalidate_denoise()
        self._discard_restore_work(self._recompute_applied)
        self._set_input_data(
            self._validate_data(
                data_items[_persistence_constants.SAVED_TOOL_DATA_NAME].rename(
                    self._data.name
                )
            )
        )
        if self._applied is None:
            self._clear_result(notify=False)
        elif "denoised" in data_items:
            self._show_result(self._validate_data(data_items["denoised"]), notify=False)
        else:
            self._run_or_defer_restore_work(self._recompute_applied, run_on_show=True)

    @property
    def result(self) -> xr.DataArray:
        self._flush_restore_work(self._recompute_applied)
        if self._result is None:
            raise ValueError("Denoise the data first")
        return self._result

    def _result_output_data(self) -> xr.DataArray | None:
        self._flush_restore_work(self._recompute_applied)
        return None if self._applied is None else self._result

    @QtCore.Slot()
    def open_itool(self) -> None:
        self._launch_output_imagetool(self.result, output_id=self.Output.RESULT)

    def _result_provenance(
        self, *, primary_input: str | None = None, data: xr.DataArray | None = None
    ) -> tuple[ToolProvenanceOperation, ...]:
        del primary_input, data
        if self._applied is None:
            return ()
        return (self._operation(self._applied),)

    def _provenance_seed_code(
        self, *, primary_input: str | None = None, data: xr.DataArray | None = None
    ) -> str:
        del data
        return f"result = {primary_input or self.data_name}"

    def _detached_raw_operations(
        self, *, primary_input: str | None = None, data: xr.DataArray | None = None
    ) -> tuple[ToolProvenanceOperation, ...]:
        del primary_input, data
        return (TransposeOperation(),)

    def _detached_result_operations(
        self, *, primary_input: str | None = None, data: xr.DataArray | None = None
    ) -> tuple[ToolProvenanceOperation, ...]:
        return (
            *self._result_provenance(primary_input=primary_input, data=data),
            *self._detached_raw_operations(),
        )

    def detached_output_imagetool_provenance(
        self, data: xr.DataArray, *, source: QtCore.QObject | None = None
    ) -> ToolProvenanceSpec | None:
        if source is self.images[0]:
            method = "_detached_raw_operations"
        elif source is self.images[1]:
            method = "_detached_result_operations"
        else:
            return super().detached_output_imagetool_provenance(data, source=source)
        return self._resolve_script_provenance(
            erlab.interactive.utils.ToolScriptProvenanceDefinition(
                start_label="Start from current dntool input data",
                operations_method=method,
                active_name="result",
                seed_code_method="_provenance_seed_code",
            ),
            data=data,
        )

    def validate_update_inputs(
        self, inputs: Mapping[str, xr.DataArray]
    ) -> Mapping[str, xr.DataArray]:
        validated = dict(super().validate_update_inputs(inputs))
        validated["data"] = self._validate_data(validated["data"])
        return validated

    def update_inputs(self, inputs: Mapping[str, xr.DataArray]) -> bool:
        self._invalidate_denoise()
        data = inputs["data"]
        if self._applied is not None:
            self._start_denoise(self._applied, data, update_input=True)
            if self._source_refreshing:
                self._defer_source_refresh()
            return False
        self._set_input_data(data)
        self._update_status()
        self._notify_data_changed()
        self._reset_history_stack()
        return True

    def _set_input_data(self, data: xr.DataArray) -> None:
        self._data = data
        self.images[0].set_source_data(self._data)
        self.colorbars[0].reset_levels()
        self.images[0].setLevels(self.colorbars[0].limits)
        self._update_input_size()

    def _cancel_background_work(self, *, timeout_ms: int) -> bool:
        del timeout_ms
        # The worker owns only a snapshot. Discard its result before replacing inputs.
        self._invalidate_denoise()
        return True

    def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:
        self._invalidate_denoise()
        super().closeEvent(event)


def dntool(
    data: xr.DataArray, data_name: str | None = None, *, execute: bool | None = None
) -> DenoiseTool:
    """Open the interactive denoising tool for a 2D DataArray containing ``eV``."""
    if data_name is None:
        try:
            data_name = str(varname.argname("data", func=dntool, vars_only=False))
        except varname.VarnameRetrievingError:
            data_name = "data"
    with erlab.interactive.utils.setup_qapp(execute):
        window = DenoiseTool(data, data_name=data_name)
        window.show()
        window.raise_()
        window.activateWindow()
    return window
