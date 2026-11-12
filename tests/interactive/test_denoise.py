import threading

import numpy as np
import pytest
import xarray as xr
from qtpy import QtCore, QtWidgets

import erlab
from erlab.interactive._dntool import DenoiseTool, _denoise_coordinator
from erlab.interactive.colors import BetterColorBarItem
from erlab.interactive.imagetool._provenance._operations import DenoiseOperation


def test_dntool_controls_and_processing_status(qtbot):
    data = xr.DataArray(np.arange(20).reshape(4, 5), dims=("eV", "beta"))
    tool = DenoiseTool(data)
    qtbot.addWidget(tool)

    layout = tool.centralWidget().layout()
    controls = layout.itemAt(0).widget()
    graphics = layout.itemAt(1).widget()
    assert controls is not None
    assert graphics is not None
    assert controls.isAncestorOf(tool.energy_spin)
    assert not graphics.isAncestorOf(tool.energy_spin)
    groups = {group.title() for group in controls.findChildren(QtWidgets.QGroupBox)}
    assert {"Processing grid", "Options", "Model"} <= groups
    assert tool.input_size_label.text() == "Input 4 × 5"
    assert tool.plots[0].getAxis("left").labelText == "eV"
    assert tool.plots[0].getAxis("bottom").labelText == "beta"
    assert tool.threads_spin.value() == 0
    assert "Ready to denoise" in tool.status_label.text()
    assert "eight" in tool.tta_check.toolTip()
    assert "10,000" in tool.scale_check.toolTip()
    assert "original coordinates" in tool.size_check.toolTip()
    assert len(tool.colorbars) == len(tool.images) == 2
    for colorbar, image in zip(tool.colorbars, tool.images, strict=True):
        assert isinstance(colorbar, BetterColorBarItem)
        assert colorbar.primary_image() is image
    input_levels = tool.images[0].getLevels()
    assert input_levels is not None
    assert input_levels[0] < input_levels[1]
    tool._show_result(data * 2)
    result_levels = tool.images[1].getLevels()
    assert result_levels is not None
    assert result_levels[0] < result_levels[1]
    tool._show_result(data + 200)
    assert tool.images[1].getLevels() == pytest.approx(tool.colorbars[1].limits)
    assert tool.images[1].getLevels()[0] >= 200
    tool.update_inputs({"data": data + 100})
    assert tool.images[0].getLevels() == pytest.approx(tool.colorbars[0].limits)
    assert tool.images[0].getLevels()[0] >= 100
    tool.show()
    tool.colorbars[0].setSpanRegion((103.0, 110.0))
    tool.update_inputs({"data": data + 200})
    qtbot.wait_until(
        lambda: tool.colorbars[0].spanRegion() == pytest.approx((103.0, 110.0))
    )
    assert tool.images[0].getLevels() == pytest.approx((103.0, 110.0))

    tool.energy_spin.setValue(200)
    tool.angle_spin.setValue(400)
    tool.reset_grid_btn.click()
    assert tool.parameters.resize == (300, 300)
    tool._clear_result()
    assert tool.images[1]._imagetool_data() is None


def test_dntool_uses_automatic_provider(qtbot, monkeypatch):
    data = xr.DataArray(np.ones((4, 5)), dims=("eV", "beta"))
    calls = []

    def denoise(data, *, provider="auto", **kwargs):
        calls.append((provider, kwargs))
        return data * 2

    monkeypatch.setattr(erlab.analysis.image.ml, "denoise", denoise)
    tool = DenoiseTool(data, data_name="data")
    qtbot.addWidget(tool)
    tool.denoise()
    qtbot.waitUntil(tool.open_btn.isEnabled)
    operation = tool._result_provenance()[0]
    assert operation.provider == "auto"
    assert calls == [("auto", operation.kwargs)]
    namespace = {"era": erlab.analysis, "data": data}
    exec(tool.copy_code(), namespace, namespace)  # noqa: S102
    xr.testing.assert_identical(namespace["result"], tool.result)
    assert calls == [("auto", operation.kwargs), ("auto", operation.kwargs)]

    tool.energy_spin.setValue(320)
    assert tool._result_provenance()[0] == operation
    restored = erlab.interactive.utils.ToolWindow.from_dataset(tool.to_dataset())
    qtbot.addWidget(restored)
    assert restored.tool_status == tool.tool_status
    assert restored._result_provenance()[0] == operation
    xr.testing.assert_identical(restored.result, tool.result)
    assert len(calls) == 2


def test_dntool_model_picker_cancel_select_restore_and_reset(
    qtbot, monkeypatch, tmp_path
):
    data = xr.DataArray(np.ones((4, 5)), dims=("eV", "beta"))
    tool = DenoiseTool(data)
    qtbot.addWidget(tool)
    model_file = tmp_path / "custom.onnx"
    model_file.write_bytes(b"model")
    responses = iter((("", ""), (str(model_file), ""), ("", "")))
    calls = []

    def choose_file(parent, title, directory, name_filter):
        calls.append((parent, title, directory, name_filter))
        return next(responses)

    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", choose_file)
    initial_status = tool.tool_status
    assert tool.use_bundled_model_btn.isHidden()
    tool.choose_model_btn.click()
    assert tool.tool_status == initial_status
    assert tool.use_bundled_model_btn.isHidden()

    tool.choose_model_btn.click()
    assert tool.parameters.model_path == str(model_file)
    assert tool.model_summary.text() == model_file.name
    assert tool.model_summary.toolTip() == QtCore.QDir.toNativeSeparators(
        str(model_file)
    )
    assert not tool.use_bundled_model_btn.isHidden()
    assert calls[1][0] is tool
    assert "*.onnx" in calls[1][3]

    selected_status = tool.tool_status
    tool.choose_model_btn.click()
    assert calls[2][2] == str(model_file)
    assert tool.tool_status == selected_status

    restored = erlab.interactive.utils.ToolWindow.from_dataset(tool.to_dataset())
    qtbot.addWidget(restored)
    assert isinstance(restored, DenoiseTool)
    assert restored.parameters.model_path == str(model_file)
    assert restored.model_summary.toolTip() == tool.model_summary.toolTip()
    restored.use_bundled_model_btn.click()
    assert restored.parameters.model_path is None
    assert restored.use_bundled_model_btn.isHidden()


@pytest.mark.parametrize("dims", [("eV", "beta"), ("beta", "eV")])
def test_dntool_result_and_provenance(qtbot, dims):
    pytest.importorskip("onnxruntime")
    data = xr.DataArray(np.arange(1.0, 21.0).reshape(4, 5), dims=dims, name="spectrum")
    data = data.assign_coords(
        eV=np.linspace(-0.2, 0.1, data.sizes["eV"]),
        beta=np.linspace(-15.0, 15.0, data.sizes["beta"]),
    )
    tool = erlab.interactive.dntool(data, execute=False)
    qtbot.addWidget(tool)
    assert not tool.open_btn.isEnabled()
    assert not tool.copy_btn.isEnabled()
    xr.testing.assert_identical(tool.tool_data, data)
    xr.testing.assert_identical(tool.images[0].data_array, data.transpose("eV", "beta"))
    assert tool.plots[0].getAxis("left").labelText == "eV"
    assert tool.plots[0].getAxis("bottom").labelText == "beta"
    tool.energy_spin.setValue(8)
    tool.angle_spin.setValue(10)
    tool.tta_check.setChecked(False)
    tool.denoise()
    qtbot.waitUntil(tool.open_btn.isEnabled)

    expected = erlab.analysis.image.ml.denoise(
        data, resize=(8, 10), along="beta", tta=False
    )
    xr.testing.assert_identical(tool.result, expected)
    xr.testing.assert_identical(
        tool.images[1].data_array, expected.transpose("eV", "beta")
    )
    assert tool.plots[1].getAxis("left").labelText == "eV"
    assert tool.plots[1].getAxis("bottom").labelText == "beta"
    operation = tool._result_provenance()[0]
    assert isinstance(operation, DenoiseOperation)
    assert operation.along == "beta"
    assert operation.resize == (8, 10)
    assert tool.open_btn.isEnabled()
    assert tool.copy_btn.isEnabled()
    assert "Up to date" in tool.status_label.text()

    expected_appearance = []
    for colorbar, cmap in zip(tool.colorbars, ("plasma", "viridis"), strict=True):
        colorbar.set_colormap(cmap, 0.6)
        lower, upper = colorbar.limits
        levels = (
            float(lower + 0.2 * (upper - lower)),
            float(lower + 0.8 * (upper - lower)),
        )
        colorbar.setSpanRegion(levels)
        expected_appearance.append((colorbar.colormap_properties, levels))

    # Pending edits do not change the last output or its provenance.
    tool.energy_spin.setValue(12)
    assert tool._result_provenance()[0] == operation
    assert "Settings changed" in tool.status_label.text()
    namespace = {"era": erlab.analysis, "data": data}
    exec(tool.copy_code(), namespace, namespace)  # noqa: S102
    xr.testing.assert_identical(namespace["result"], expected)

    restored = erlab.interactive.utils.ToolWindow.from_dataset(tool.to_dataset())
    qtbot.addWidget(restored)
    assert isinstance(restored, DenoiseTool)
    assert restored.tool_status == tool.tool_status
    xr.testing.assert_identical(restored.tool_data, data)
    xr.testing.assert_identical(restored.result, expected)
    xr.testing.assert_identical(
        restored.images[0].data_array, data.transpose("eV", "beta")
    )
    xr.testing.assert_identical(
        restored.images[1].data_array, expected.transpose("eV", "beta")
    )
    xr.testing.assert_identical(restored.images[0]._imagetool_data(), data.T)
    xr.testing.assert_identical(restored.images[1]._imagetool_data(), expected.T)
    for colorbar, (properties, levels) in zip(
        restored.colorbars, expected_appearance, strict=True
    ):
        assert colorbar.colormap_properties == properties
        assert colorbar.spanRegion() == pytest.approx(levels)


@pytest.mark.parametrize(
    "data",
    [
        xr.DataArray(np.ones(4), dims=("eV",)),
        xr.DataArray(np.ones((2, 2)), dims=("x", "y")),
        xr.DataArray(np.ones((2, 2, 2)), dims=("eV", "x", "y")),
    ],
)
def test_dntool_requires_two_dimensions_with_energy(qtbot, data):
    with pytest.raises(ValueError, match=r"2D.*eV"):
        DenoiseTool(data)


def test_dntool_rejects_duplicate_energy_dimensions(qtbot):
    with pytest.warns(UserWarning, match="Duplicate dimension"):
        data = xr.DataArray(np.ones((2, 2)), dims=("eV", "eV"))
    with pytest.raises(ValueError, match=r"2D.*eV"):
        DenoiseTool(data)


def test_dntool_source_update_and_failed_run(qtbot, monkeypatch, tmp_path):
    pytest.importorskip("onnxruntime")
    data = xr.DataArray(np.ones((4, 5)), dims=("eV", "alpha"))
    tool = DenoiseTool(data)
    qtbot.addWidget(tool)
    tool.energy_spin.setValue(8)
    tool.angle_spin.setValue(10)
    tool.tta_check.setChecked(False)
    tool.denoise()
    qtbot.waitUntil(tool.open_btn.isEnabled)

    updated = xr.DataArray(np.full((6, 7), 2.0), dims=("beta", "eV"))
    assert tool.update_inputs(tool.validate_update_inputs({"data": updated})) is False
    qtbot.waitUntil(lambda: not tool._busy)
    expected = erlab.analysis.image.ml.denoise(
        updated, resize=(8, 10), along="beta", tta=False
    )
    xr.testing.assert_identical(tool.result, expected)
    xr.testing.assert_identical(tool.tool_data, updated)
    xr.testing.assert_identical(
        tool.images[0].data_array, updated.transpose("eV", "beta")
    )
    xr.testing.assert_identical(
        tool.images[1].data_array, expected.transpose("eV", "beta")
    )
    assert tool._result_provenance()[0].along == "beta"
    assert tool.input_size_label.text() == "Input 7 × 6"

    tool.show()
    exported = []
    monkeypatch.setattr(
        tool,
        "_launch_detached_output_imagetool",
        lambda data, *, provenance_spec: exported.append(data),
    )
    for image in tool.images:
        image.open_itool()
    xr.testing.assert_identical(exported[0], updated.T)
    xr.testing.assert_identical(exported[1], expected.T)

    messages = []
    monkeypatch.setattr(
        "erlab.interactive._dntool.QtWidgets.QMessageBox.critical",
        lambda *args: messages.append(args),
    )
    invalid_model = tmp_path / "invalid.onnx"
    invalid_model.write_bytes(b"invalid model")
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *_args: (str(invalid_model), "ONNX models (*.onnx)"),
    )
    tool.choose_model_btn.click()
    assert tool.parameters.model_path == str(invalid_model)
    assert "Settings changed" in tool.status_label.text()
    tool.denoise()
    qtbot.waitUntil(lambda: len(messages) == 1)
    assert len(messages) == 1
    xr.testing.assert_identical(tool.result, expected)
    assert tool._result_provenance()[0].model_path is None
    assert "Denoising failed" in tool.status_label.text()


def test_dntool_copy_code_uses_caller_input(qtbot):
    pytest.importorskip("onnxruntime")
    spectrum = xr.DataArray(np.ones((4, 5)), dims=("eV", "alpha"))
    tool = erlab.interactive.dntool(spectrum, execute=False)
    qtbot.addWidget(tool)
    tool.energy_spin.setValue(8)
    tool.angle_spin.setValue(10)
    tool.tta_check.setChecked(False)
    tool.denoise()
    qtbot.waitUntil(tool.open_btn.isEnabled)

    namespace = {"era": erlab.analysis, "spectrum": spectrum}
    exec(tool.copy_code(), namespace, namespace)  # noqa: S102
    xr.testing.assert_identical(namespace["result"], tool.result)


@pytest.mark.parametrize("model_change", ["removed", "replaced"])
def test_dntool_saved_result_does_not_require_original_model(
    qtbot, monkeypatch, tmp_path, model_change
):
    data = xr.DataArray(
        np.arange(20.0).reshape(4, 5),
        dims=("beta", "eV"),
        coords={"beta": np.arange(4.0), "eV": np.linspace(-0.2, 0.2, 5)},
        name="spectrum",
        attrs={"sample": "test"},
    )
    model_file = tmp_path / "custom.onnx"
    model_file.write_text("2")
    calls = []

    def apply(operation, input_data):
        calls.append(operation)
        scale = float(model_file.read_text())
        return input_data.isel(eV=slice(None, None, 2)) * scale

    monkeypatch.setattr(DenoiseOperation, "apply", apply)
    tool = DenoiseTool(data)
    qtbot.addWidget(tool)
    tool._model_path = str(model_file)
    tool.size_check.setChecked(False)
    tool.denoise()
    qtbot.waitUntil(tool.open_btn.isEnabled)
    expected = tool.result.copy(deep=True)
    saved_path = tmp_path / "denoised.h5"
    tool.to_file(saved_path)

    if model_change == "removed":
        model_file.unlink()
    else:
        model_file.write_text("7")

    restored = erlab.interactive.utils.ToolWindow.from_file(saved_path)
    qtbot.addWidget(restored)
    assert isinstance(restored, DenoiseTool)
    assert len(calls) == 1
    xr.testing.assert_identical(restored.tool_data, data)
    xr.testing.assert_identical(restored.result, expected)
    assert restored._result_provenance() == tool._result_provenance()
    assert restored.open_btn.isEnabled()
    assert restored.copy_btn.isEnabled()


@pytest.mark.parametrize("restore_saved", [False, True], ids=["live", "restored"])
@pytest.mark.parametrize("model_change", ["removed", "replaced"])
def test_dntool_control_undo_redo_reuses_completed_result(
    qtbot, monkeypatch, tmp_path, restore_saved, model_change
):
    data = xr.DataArray(np.arange(20.0).reshape(4, 5), dims=("eV", "beta"))
    model_file = tmp_path / "custom.onnx"
    model_file.write_text("2")
    calls = []

    def apply(_operation, input_data):
        calls.append(input_data)
        return input_data * float(model_file.read_text())

    monkeypatch.setattr(DenoiseOperation, "apply", apply)
    tool = DenoiseTool(data)
    qtbot.addWidget(tool)
    tool._model_path = str(model_file)
    tool.energy_spin.setValue(8)
    tool._flush_pending_history_write()
    tool.denoise()
    qtbot.waitUntil(tool.open_btn.isEnabled)
    tool._flush_pending_history_write()
    if restore_saved:
        restored = erlab.interactive.utils.ToolWindow.from_dataset(tool.to_dataset())
        qtbot.addWidget(restored)
        assert isinstance(restored, DenoiseTool)
        tool = restored
    previous = tool.result.copy(deep=True)
    applied = tool._result_provenance()
    if model_change == "removed":
        model_file.unlink()
    else:
        model_file.write_text("7")

    tool.energy_spin.setValue(12)
    tool._flush_pending_history_write()
    tool.undo()
    assert tool.energy_spin.value() == 8
    assert tool.redoable
    assert not tool._busy
    assert len(calls) == 1
    xr.testing.assert_identical(tool.result, previous)
    assert tool._result_provenance() == applied

    tool.redo()
    assert tool.energy_spin.value() == 12
    assert not tool.redoable
    assert not tool._busy
    assert len(calls) == 1
    xr.testing.assert_identical(tool.result, previous)
    assert tool._result_provenance() == applied
    assert tool.open_btn.isEnabled()
    assert tool.copy_btn.isEnabled()


@pytest.mark.parametrize("succeeds", [True, False], ids=["success", "failure"])
def test_dntool_applied_undo_recomputes_in_worker_without_changing_history(
    qtbot, monkeypatch, succeeds
):
    data = xr.DataArray(np.arange(20.0).reshape(4, 5), dims=("eV", "beta"))
    tool = DenoiseTool(data)
    qtbot.addWidget(tool)
    entered = threading.Event()
    release = threading.Event()
    calls = []
    errors = []
    gui_thread = threading.get_ident()

    def apply(operation, input_data):
        calls.append((operation.resize, threading.get_ident()))
        if len(calls) == 3:
            entered.set()
            if not release.wait(10):
                raise TimeoutError("Denoise test worker was not released")
            if not succeeds:
                raise FileNotFoundError("The previous custom model is unavailable")
        return input_data + operation.resize[0]

    monkeypatch.setattr(DenoiseOperation, "apply", apply)
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "critical",
        lambda *args: errors.append((args, threading.get_ident())),
    )
    for size in (8, 12):
        tool.energy_spin.setValue(size)
        tool._flush_pending_history_write()
        tool.denoise()
        qtbot.waitUntil(lambda: not tool._busy)
        tool._flush_pending_history_write()
    before_undo = len(tool._prev_states)

    tool.undo()
    try:
        qtbot.waitUntil(entered.is_set)
        assert calls[-1][1] != gui_thread
        assert not tool.run_btn.isEnabled()
        assert tool.redoable
        assert tool._result_output_data() is None
        assert not tool.open_btn.isEnabled()
        assert not tool.copy_btn.isEnabled()
        with pytest.raises(ValueError, match="Denoise the data first"):
            _ = tool.result
        processed = []
        QtCore.QTimer.singleShot(0, lambda: processed.append(threading.get_ident()))
        qtbot.waitUntil(lambda: bool(processed))
        assert processed == [gui_thread]
    finally:
        release.set()

    qtbot.waitUntil(lambda: not tool._busy)
    tool._flush_pending_history_write()
    assert len(tool._prev_states) == before_undo - 1
    assert len(tool._next_states) == 1
    assert tool.redoable
    if succeeds:
        xr.testing.assert_identical(tool.result, data + 8)
        assert tool._result_provenance()[0].resize[0] == 8
        assert errors == []
    else:
        qtbot.waitUntil(lambda: bool(errors))
        assert errors[0][1] == gui_thread
        assert "unavailable" in errors[0][0][-1]
        assert tool._result_output_data() is None
        assert tool._result_provenance() == ()

    tool.redo()
    qtbot.waitUntil(lambda: not tool._busy)
    tool._flush_pending_history_write()
    assert len(tool._prev_states) == before_undo
    assert not tool.redoable
    xr.testing.assert_identical(tool.result, data + 12)
    assert tool._result_provenance()[0].resize[0] == 12


def test_dntool_redo_discards_pending_undo_result(qtbot, monkeypatch):
    data = xr.DataArray(np.arange(20.0).reshape(4, 5), dims=("eV", "beta"))
    tool = DenoiseTool(data)
    qtbot.addWidget(tool)
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def apply(operation, input_data):
        calls.append(operation.resize)
        if len(calls) == 3:
            entered.set()
            if not release.wait(10):
                raise TimeoutError("Denoise test worker was not released")
        return input_data + operation.resize[0]

    monkeypatch.setattr(DenoiseOperation, "apply", apply)
    for size in (8, 12):
        tool.energy_spin.setValue(size)
        tool._flush_pending_history_write()
        tool.denoise()
        qtbot.waitUntil(lambda: not tool._busy)
        tool._flush_pending_history_write()
    history_size = len(tool._prev_states)
    tool.undo()
    task = tool._denoise_task
    assert task is not None
    try:
        qtbot.waitUntil(entered.is_set)
        tool.redo()
        qtbot.waitUntil(lambda: not tool._busy)
        xr.testing.assert_identical(tool.result, data + 12)
    finally:
        release.set()

    coordinator = _denoise_coordinator()
    qtbot.waitUntil(lambda: task not in coordinator._tasks)
    tool._flush_pending_history_write()
    assert len(tool._prev_states) == history_size
    assert not tool.redoable
    xr.testing.assert_identical(tool.result, data + 12)
    assert tool._result_provenance()[0].resize[0] == 12


@pytest.mark.parametrize("defer_restore", [False, True])
def test_dntool_legacy_saved_result_is_recomputed(qtbot, monkeypatch, defer_restore):
    data = xr.DataArray(np.arange(20.0).reshape(4, 5), dims=("eV", "beta"))
    calls = []

    def apply(_operation, input_data):
        calls.append(input_data)
        return input_data + len(calls)

    monkeypatch.setattr(DenoiseOperation, "apply", apply)
    tool = DenoiseTool(data)
    qtbot.addWidget(tool)
    tool.denoise()
    qtbot.waitUntil(tool.open_btn.isEnabled)
    legacy = tool.to_dataset().drop_vars("denoised")

    restored = erlab.interactive.utils.ToolWindow.from_dataset(
        legacy, _defer_restore_work=defer_restore
    )
    qtbot.addWidget(restored)
    assert isinstance(restored, DenoiseTool)
    assert len(calls) == (1 if defer_restore else 2)
    xr.testing.assert_identical(restored.result, data + 2)
    assert len(calls) == 2
    assert restored._result_provenance() == tool._result_provenance()


@pytest.mark.parametrize("image_index", [0, 1])
@pytest.mark.parametrize("dims", [("eV", "beta"), ("beta", "eV")])
def test_dntool_plot_output_provenance(qtbot, monkeypatch, image_index, dims):
    pytest.importorskip("onnxruntime")
    data = xr.DataArray(np.arange(1.0, 21.0).reshape(4, 5), dims=dims)
    tool = DenoiseTool(data, data_name="data")
    qtbot.addWidget(tool)
    tool.show()
    tool.energy_spin.setValue(8)
    tool.angle_spin.setValue(10)
    tool.tta_check.setChecked(False)
    tool.denoise()
    qtbot.waitUntil(tool.open_btn.isEnabled)

    captured = []
    monkeypatch.setattr(
        tool,
        "_launch_detached_output_imagetool",
        lambda data, *, provenance_spec: captured.append((data, provenance_spec)),
    )
    tool.images[image_index].open_itool()
    output, spec = captured[0]
    assert spec is not None
    namespace = {"era": erlab.analysis, "data": data}
    exec(spec.display_code(), namespace, namespace)  # noqa: S102
    xr.testing.assert_identical(namespace["result"], output)
    source = data if image_index == 0 else tool.result
    xr.testing.assert_identical(
        tool.images[image_index].data_array, source.transpose("eV", "beta")
    )
    xr.testing.assert_identical(output, source.T)


def test_dntool_run_keeps_qt_responsive_and_discards_stale_work(qtbot, monkeypatch):
    data = xr.DataArray(np.arange(20).reshape(4, 5), dims=("eV", "beta"))
    tool = DenoiseTool(data)
    qtbot.addWidget(tool)
    entered = threading.Event()
    release = threading.Event()
    worker_threads = []
    gui_thread = threading.get_ident()
    completed = []
    finish_denoise = tool._finish_denoise

    def record_completion(*args):
        finish_denoise(*args)
        completed.append(args[0])

    def blocked_apply(_operation, input_data):
        worker_threads.append(threading.get_ident())
        entered.set()
        if not release.wait(10):
            raise TimeoutError("Denoise test worker was not released")
        return input_data * 2

    monkeypatch.setattr(DenoiseOperation, "apply", blocked_apply)
    monkeypatch.setattr(tool, "_finish_denoise", record_completion)
    tool.denoise()
    try:
        qtbot.waitUntil(entered.is_set)
        assert not tool.run_btn.isEnabled()
        assert not tool.open_btn.isEnabled()
        assert len(worker_threads) == 1
        assert worker_threads[0] != gui_thread

        processed = []
        QtCore.QTimer.singleShot(0, lambda: processed.append(threading.get_ident()))
        qtbot.waitUntil(lambda: bool(processed))
        assert processed == [gui_thread]

        tool.denoise()
        assert len(worker_threads) == 1
        updated = data + 10
        assert tool.update_inputs({"data": updated}) is True
        xr.testing.assert_identical(tool.tool_data, updated)
    finally:
        release.set()

    qtbot.waitUntil(lambda: len(completed) == 1)
    assert not tool._busy
    assert tool.run_btn.isEnabled()
    assert not tool.open_btn.isEnabled()
    with pytest.raises(ValueError, match="Denoise the data first"):
        _ = tool.result
    assert len(worker_threads) == 1

    tool.denoise()
    qtbot.waitUntil(tool.open_btn.isEnabled)
    xr.testing.assert_identical(tool.result, updated * 2)
    assert len(worker_threads) == 2


def test_dntool_worker_error_preserves_last_result(qtbot, monkeypatch):
    data = xr.DataArray(np.arange(20).reshape(4, 5), dims=("eV", "beta"))
    tool = DenoiseTool(data)
    qtbot.addWidget(tool)
    entered = threading.Event()
    release = threading.Event()
    calls = []
    messages = []
    gui_thread = threading.get_ident()

    def apply(_operation, input_data):
        calls.append(threading.get_ident())
        if len(calls) == 1:
            return input_data * 2
        entered.set()
        if not release.wait(10):
            raise TimeoutError("Denoise test worker was not released")
        raise ValueError("Invalid test model")

    def show_error(*args):
        messages.append((args, threading.get_ident()))

    monkeypatch.setattr(DenoiseOperation, "apply", apply)
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", show_error)
    tool.denoise()
    qtbot.waitUntil(tool.open_btn.isEnabled)
    previous = tool.result
    previous_operation = tool._result_provenance()

    tool.energy_spin.setValue(8)
    tool.denoise()
    try:
        qtbot.waitUntil(entered.is_set)
        assert not tool.run_btn.isEnabled()
        xr.testing.assert_identical(tool.result, previous)
    finally:
        release.set()

    qtbot.waitUntil(lambda: len(messages) == 1)
    assert messages[0][1] == gui_thread
    assert "Invalid test model" in messages[0][0][-1]
    assert tool.run_btn.isEnabled()
    xr.testing.assert_identical(tool.result, previous)
    assert tool._result_provenance() == previous_operation
    assert len(calls) == 2


def test_dntool_input_update_keeps_qt_responsive_and_publishes_atomically(
    qtbot, monkeypatch
):
    data = xr.DataArray(np.arange(20.0).reshape(4, 5), dims=("eV", "beta"))
    updated = data + 10
    tool = DenoiseTool(data)
    qtbot.addWidget(tool)
    entered = threading.Event()
    release = threading.Event()
    worker_threads = []
    gui_thread = threading.get_ident()

    def apply(_operation, input_data):
        worker_threads.append(threading.get_ident())
        if len(worker_threads) > 1:
            entered.set()
            if not release.wait(10):
                raise TimeoutError("Denoise test worker was not released")
        return input_data * 2

    monkeypatch.setattr(DenoiseOperation, "apply", apply)
    tool.denoise()
    qtbot.waitUntil(tool.open_btn.isEnabled)
    previous = tool.result.copy(deep=True)
    applied = tool._result_provenance()
    assert tool.update_inputs({"data": updated}) is False
    try:
        qtbot.waitUntil(entered.is_set)
        assert worker_threads[-1] != gui_thread
        assert not tool.run_btn.isEnabled()
        xr.testing.assert_identical(tool.tool_data, data)
        xr.testing.assert_identical(tool.result, previous)
        assert tool._result_provenance() == applied

        processed = []
        QtCore.QTimer.singleShot(0, lambda: processed.append(threading.get_ident()))
        qtbot.waitUntil(lambda: bool(processed))
        assert processed == [gui_thread]
        tool.energy_spin.setValue(12)
    finally:
        release.set()

    qtbot.waitUntil(lambda: not tool._busy)
    xr.testing.assert_identical(tool.tool_data, updated)
    xr.testing.assert_identical(tool.result, updated * 2)
    assert tool._result_provenance() == applied
    assert tool.parameters.resize[0] == 12
    assert tool.run_btn.isEnabled()


def test_dntool_input_update_discards_superseded_completion(qtbot, monkeypatch):
    data = xr.DataArray(np.arange(20.0).reshape(4, 5), dims=("eV", "beta"))
    tool = DenoiseTool(data)
    qtbot.addWidget(tool)
    entered = threading.Event()
    release = threading.Event()

    def apply(_operation, input_data):
        if input_data.identical(first):
            entered.set()
            if not release.wait(10):
                raise TimeoutError("Denoise test worker was not released")
        return input_data * 2

    first = data + 10
    latest = data + 20
    monkeypatch.setattr(DenoiseOperation, "apply", apply)
    tool.denoise()
    qtbot.waitUntil(tool.open_btn.isEnabled)
    assert tool.update_inputs({"data": first}) is False
    task = tool._denoise_task
    assert task is not None
    try:
        qtbot.waitUntil(entered.is_set)
        assert tool.update_inputs({"data": latest}) is False
        qtbot.waitUntil(lambda: not tool._busy)
        xr.testing.assert_identical(tool.tool_data, latest)
        xr.testing.assert_identical(tool.result, latest * 2)
    finally:
        release.set()

    coordinator = _denoise_coordinator()
    qtbot.waitUntil(lambda: task not in coordinator._tasks)
    xr.testing.assert_identical(tool.tool_data, latest)
    xr.testing.assert_identical(tool.result, latest * 2)


def test_dntool_failed_input_update_preserves_current_input_and_result(
    qtbot, monkeypatch
):
    data = xr.DataArray(np.arange(20.0).reshape(4, 5), dims=("eV", "beta"))
    tool = DenoiseTool(data)
    qtbot.addWidget(tool)
    calls = []
    messages = []

    def apply(_operation, input_data):
        calls.append(input_data)
        if len(calls) > 1:
            raise ValueError("Invalid refreshed input")
        return input_data * 2

    monkeypatch.setattr(DenoiseOperation, "apply", apply)
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "critical", lambda *args: messages.append(args)
    )
    tool.denoise()
    qtbot.waitUntil(tool.open_btn.isEnabled)
    previous = tool.result.copy(deep=True)
    applied = tool._result_provenance()
    assert tool.update_inputs({"data": data + 10}) is False
    qtbot.waitUntil(lambda: len(messages) == 1)

    assert not tool._busy
    assert tool.run_btn.isEnabled()
    xr.testing.assert_identical(tool.tool_data, data)
    xr.testing.assert_identical(tool.result, previous)
    assert tool._result_provenance() == applied


def test_dntool_close_during_worker_is_safe(qtbot, monkeypatch):
    data = xr.DataArray(np.ones((4, 5)), dims=("eV", "beta"))
    tool = DenoiseTool(data)
    tool.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
    tool.show()
    entered = threading.Event()
    release = threading.Event()

    def blocked_apply(_operation, input_data):
        entered.set()
        if not release.wait(10):
            raise TimeoutError("Denoise test worker was not released")
        return input_data * 2

    monkeypatch.setattr(DenoiseOperation, "apply", blocked_apply)
    tool.denoise()
    coordinator = _denoise_coordinator()
    task = tool._denoise_task
    assert task is not None
    try:
        qtbot.waitUntil(entered.is_set)
        tool.close()
        qtbot.waitUntil(lambda: not erlab.interactive.utils.qt_is_valid(tool))
    finally:
        release.set()

    qtbot.waitUntil(lambda: task not in coordinator._tasks)
