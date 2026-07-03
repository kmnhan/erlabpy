import json
import tempfile
import time
import typing

import numpy as np
import pytest
import scipy
import xarray as xr
import xarray_lmfit as xlm
from qtpy import QtCore, QtGui, QtWidgets

import erlab
import erlab.interactive._fit1d as fit1d_module
from erlab.interactive._fit1d import Fit1DTool
from erlab.interactive.fermiedge import (
    EdgeFitSignals,
    EdgeFitTask,
    GoldTool,
    ResolutionFitThread,
    ResolutionTool,
    goldtool,
    restool,
)
from erlab.io.exampledata import generate_gold_edge


def _configure_goldtool_state(
    win: GoldTool, *, fitted: bool = False, spline: bool = False
) -> None:
    with (
        QtCore.QSignalBlocker(win.params_edge),
        QtCore.QSignalBlocker(win.params_poly),
        QtCore.QSignalBlocker(win.params_spl),
        QtCore.QSignalBlocker(win.params_tab),
    ):
        win._restore_parameter_group_values(
            win.params_edge,
            {
                "T (K)": 45.0,
                "Fix T": False,
                "Bin x": 2,
                "Bin y": 3,
                "Resolution": 0.015,
                "Fast": False,
                "Linear": False,
                "Method": "cg",
                "Scale cov": False,
                "# CPU": 1,
            },
        )
        win._restore_parameter_group_values(
            win.params_poly,
            {
                "Degree": 3,
                "Scale cov": False,
                "Residuals": True,
                "Corrected": not spline,
                "Shift coords": False,
            },
        )
        win._restore_parameter_group_values(
            win.params_spl,
            {
                "Auto": False,
                "lambda": 2.5,
                "Residuals": True,
                "Corrected": spline,
                "Shift coords": False,
            },
        )
        win.params_tab.setCurrentIndex(1 if spline else 0)

    win.params_roi.modify_roi(x0=-8.0, x1=8.0, y0=-0.2, y1=0.1)
    win.refit_on_source_update_check.setChecked(True)
    win._toggle_fast()
    win._sync_spline_lambda_enabled()

    if fitted:
        edge_center = win.data.mean("eV")
        edge_stderr = xr.ones_like(edge_center)
        win.post_fit(edge_center, edge_stderr)


def _spy_goldtool_post_fit(monkeypatch: pytest.MonkeyPatch) -> list[GoldTool]:
    calls: list[GoldTool] = []
    original_post_fit = GoldTool.post_fit

    def tracked_post_fit(
        self: GoldTool,
        edge_center: xr.DataArray,
        edge_stderr: xr.DataArray,
        *,
        task: EdgeFitTask | None = None,
    ) -> None:
        calls.append(self)
        original_post_fit(self, edge_center, edge_stderr, task=task)

    monkeypatch.setattr(GoldTool, "post_fit", tracked_post_fit)
    return calls


def _drop_goldtool_fit_payload(ds: xr.Dataset) -> xr.Dataset:
    return ds.drop_vars(
        [
            GoldTool._PERSISTED_FIT_ALONG_VAR,
            GoldTool._PERSISTED_EDGE_CENTER_VAR,
            GoldTool._PERSISTED_EDGE_STDERR_VAR,
        ]
    )


def test_goldtool_can_save_and_load() -> None:
    assert GoldTool.can_save_and_load() is True


def _seed_goldtool_poly_result(win: GoldTool, result: xr.Dataset) -> None:
    win.edge_center = result.modelfit_data.copy()
    win.edge_stderr = xr.ones_like(win.edge_center)
    win.result = result
    win.params_poly.setDisabled(False)
    win.params_spl.setDisabled(False)
    win.params_tab.setDisabled(False)


@pytest.mark.parametrize("fast", [True, False], ids=["StepB", "FD"])
def test_goldtool(
    qtbot, gold, fast, gold_fit_res, gold_fit_res_fd, accept_dialog
) -> None:
    win: GoldTool = goldtool(gold, execute=False)
    qtbot.addWidget(win)
    win.params_edge.widgets["Fast"].setChecked(fast)
    win.params_edge.widgets["# CPU"].setValue(1)

    expected = gold_fit_res if fast else gold_fit_res_fd
    _seed_goldtool_poly_result(win, expected)

    def check_generated_code(w: GoldTool) -> None:
        namespace = {"era": erlab.analysis, "gold": gold}

        exec(  # noqa: S102
            w.current_provenance_spec().display_code(), {"__builtins__": {}}, namespace
        )

        xr.testing.assert_identical(
            w.result.drop_vars("modelfit_results"),
            namespace["model_result"].drop_vars("modelfit_results"),
        )
        xr.testing.assert_identical(
            w.result.drop_vars("modelfit_results"),
            expected.drop_vars("modelfit_results"),
        )

    check_generated_code(win)

    # Test save fit
    tmp_dir = tempfile.TemporaryDirectory()
    filename = f"{tmp_dir.name}/fit_save.h5"

    def _go_to_file(dialog: QtWidgets.QFileDialog):
        dialog.setDirectory(tmp_dir.name)
        dialog.selectFile(filename)
        focused = dialog.focusWidget()
        if isinstance(focused, QtWidgets.QLineEdit):
            focused.setText("fit_save.h5")

    # Save fit
    accept_dialog(win._save_poly_fit, pre_call=_go_to_file)

    # Check saved file
    xr.testing.assert_identical(
        expected.drop_vars("modelfit_results"),
        xlm.load_fit(filename).drop_vars("modelfit_results"),
    )

    # Move to spline tab
    win.params_tab.setCurrentIndex(1)

    assert isinstance(win.result, scipy.interpolate.BSpline)

    tmp_dir.cleanup()


def test_goldtool_roundtrip_unfitted(qtbot, gold) -> None:
    win: GoldTool = goldtool(gold, execute=False, data_name="gold_input")
    qtbot.addWidget(win)
    _configure_goldtool_state(win, fitted=False, spline=True)

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        filename = f"{tmp_dir_name}/goldtool_unfitted.h5"
        win.to_file(filename)

        win_restored = erlab.interactive.utils.ToolWindow.from_file(filename)
        qtbot.addWidget(win_restored)
        assert isinstance(win_restored, GoldTool)

        assert win.tool_status == win_restored.tool_status
        assert win_restored.result is None
        assert not hasattr(win_restored, "edge_center")
        assert not hasattr(win_restored, "edge_stderr")


def test_fit1d_persistence_helper_edges(qtbot, monkeypatch, gold) -> None:
    win = Fit1DTool(gold.mean("alpha"), data_name="gold_input")
    qtbot.addWidget(win)

    empty = xr.Dataset()
    win._restore_persistence_payload(empty)
    assert win._append_persistence_payload(empty) is empty

    blob = np.array([1, 2, 3], dtype=np.uint8)
    win._pending_persisted_fit_result_blob = blob
    win._pending_persisted_fit_is_current = True
    appended = win._append_persistence_payload(empty)
    np.testing.assert_array_equal(
        appended[win._PERSISTED_FIT_RESULT_VAR].values,
        blob,
    )
    assert appended.attrs[win._PERSISTED_FIT_CURRENT_ATTR] is True

    calls: list[str] = []
    monkeypatch.setattr(
        win, "_flush_restore_work", lambda *_, **__: calls.append("flush")
    )
    monkeypatch.setattr(
        win,
        "_show_warning",
        lambda title, message: calls.append(f"{title}: {message}"),
    )

    win._last_result_ds = None
    win._save_fit()

    assert calls == ["flush", "No fit result: There is no fit result to save."]

    class _FakeFitResults:
        def compute(self):
            return self

        def item(self):
            return object()

    fake_result_ds = typing.cast(
        "xr.Dataset",
        typing.cast("object", type("FakeResultDataset", (), {})()),
    )
    fake_result_ds.modelfit_results = _FakeFitResults()
    monkeypatch.setattr(
        fit1d_module,
        "_load_lmfit_for_ftool_restore",
        lambda load, **_kwargs: fake_result_ds,
    )
    monkeypatch.setattr(win, "_set_fit_stats", lambda _result: calls.append("stats"))
    monkeypatch.setattr(win, "_update_fit_curve", lambda: calls.append("curve"))
    monkeypatch.setattr(win, "_mark_fit_fresh", lambda: calls.append("fresh"))
    monkeypatch.setattr(win, "_mark_fit_stale", lambda: calls.append("stale"))

    win._restore_persisted_fit_result_blob(blob, fit_is_current=True)
    win._restore_persisted_fit_result_blob(blob, fit_is_current=False)

    assert calls[-6:] == ["stats", "curve", "fresh", "stats", "curve", "stale"]


def test_goldtool_roundtrip_fitted(qtbot, gold) -> None:
    win: GoldTool = goldtool(gold, execute=False, data_name="gold_input")
    qtbot.addWidget(win)
    _configure_goldtool_state(win, fitted=True, spline=True)

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        filename = f"{tmp_dir_name}/goldtool_fitted.h5"
        win.to_file(filename)

        with xr.load_dataset(filename, engine="h5netcdf") as saved:
            saved_state = json.loads(saved.attrs["tool_state"])
            assert saved_state["schema_version"] == 2
            assert "fit_snapshot" not in saved_state
            assert "go" not in saved_state["edge_values"]
            assert "copy" not in saved_state["poly_values"]
            assert "itool" not in saved_state["poly_values"]
            assert "save" not in saved_state["poly_values"]
            assert GoldTool._PERSISTED_FIT_ALONG_VAR in saved
            assert GoldTool._PERSISTED_EDGE_CENTER_VAR in saved
            assert GoldTool._PERSISTED_EDGE_STDERR_VAR in saved

        win_restored = erlab.interactive.utils.ToolWindow.from_file(filename)
        qtbot.addWidget(win_restored)
        assert isinstance(win_restored, GoldTool)

        assert win.tool_status == win_restored.tool_status
        assert win_restored.params_tab.currentIndex() == 1
        assert isinstance(win_restored.result, scipy.interpolate.BSpline)
        xr.testing.assert_identical(win.corrected, win_restored.corrected)
        assert str(win_restored.info_text) == str(win.info_text)


def test_goldtool_loads_legacy_state_with_raw_dicts_and_fit_snapshot(
    qtbot, gold
) -> None:
    win: GoldTool = goldtool(gold, execute=False, data_name="gold_input")
    qtbot.addWidget(win)
    _configure_goldtool_state(win, fitted=True, spline=True)
    snapshot = win.tool_status.fit_snapshot
    assert snapshot is not None

    saved = _drop_goldtool_fit_payload(win.to_dataset())
    legacy_state = json.loads(saved.attrs["tool_state"])
    legacy_state.pop("schema_version", None)
    legacy_state["edge_values"] = {
        **dict(win.params_edge.values),
        "go": True,
        "unknown": "ignored",
        "Method": "unsupported-method",
        "# CPU": 0,
    }
    legacy_state["poly_values"] = {
        **dict(win.params_poly.values),
        "itool": True,
        "copy": True,
        "save": True,
        "unknown": "ignored",
    }
    legacy_state["spline_values"] = {
        **dict(win.params_spl.values),
        "itool": True,
        "copy": True,
        "unknown": "ignored",
    }
    legacy_state["tab_index"] = "invalid"
    legacy_state["fit_snapshot"] = snapshot.model_dump()
    saved.attrs["tool_state"] = json.dumps(legacy_state)

    restored = erlab.interactive.utils.ToolWindow.from_dataset(saved)
    qtbot.addWidget(restored)
    assert isinstance(restored, GoldTool)

    assert restored.params_tab.currentIndex() == 0
    assert restored.params_edge.values["Method"] == "least_squares"
    assert restored.params_edge.values["# CPU"] == 1
    assert "go" not in restored.tool_status.edge_values.model_dump(by_alias=True)
    assert "copy" not in restored.tool_status.poly_values.model_dump(by_alias=True)
    xr.testing.assert_equal(restored.edge_center, win.edge_center)
    xr.testing.assert_equal(restored.edge_stderr, win.edge_stderr)


def test_goldtool_deferred_restore_fit_payload(qtbot, gold, monkeypatch) -> None:
    win: GoldTool = goldtool(gold, execute=False, data_name="gold_input")
    qtbot.addWidget(win)
    _configure_goldtool_state(win, fitted=True, spline=True)
    expected_corrected = win.corrected
    saved = win.to_dataset()

    post_fit_calls = _spy_goldtool_post_fit(monkeypatch)
    monkeypatch.setattr(
        erlab.interactive.utils.varname,
        "argname",
        lambda *_args, **_kwargs: pytest.fail(
            "deferred goldtool restore should not inspect the caller frame"
        ),
    )
    restored = erlab.interactive.utils.ToolWindow.from_dataset(
        saved, _defer_restore_work=True
    )
    qtbot.addWidget(restored)
    assert isinstance(restored, GoldTool)

    assert post_fit_calls == []
    assert restored.data_name == "gold_input"
    assert not hasattr(restored, "edge_center")
    assert restored.result is None
    assert restored._pending_persisted_fit_snapshot is not None

    restored._flush_restore_work()

    assert post_fit_calls == [restored]
    assert restored._pending_persisted_fit_snapshot is None
    assert isinstance(restored.result, scipy.interpolate.BSpline)
    xr.testing.assert_identical(restored.corrected, expected_corrected)


def test_goldtool_deferred_restore_legacy_fit_snapshot(
    qtbot, gold, monkeypatch
) -> None:
    win: GoldTool = goldtool(gold, execute=False, data_name="gold_input")
    qtbot.addWidget(win)
    _configure_goldtool_state(win, fitted=True, spline=True)
    snapshot = win.tool_status.fit_snapshot
    assert snapshot is not None

    saved = _drop_goldtool_fit_payload(win.to_dataset())
    legacy_state = json.loads(saved.attrs["tool_state"])
    legacy_state["fit_snapshot"] = snapshot.model_dump()
    saved.attrs["tool_state"] = json.dumps(legacy_state)

    post_fit_calls = _spy_goldtool_post_fit(monkeypatch)
    restored = erlab.interactive.utils.ToolWindow.from_dataset(
        saved, _defer_restore_work=True
    )
    qtbot.addWidget(restored)
    assert isinstance(restored, GoldTool)

    assert post_fit_calls == []
    assert not hasattr(restored, "edge_center")
    assert restored._pending_persisted_fit_snapshot is not None

    restored._flush_restore_work()

    assert post_fit_calls == [restored]
    assert restored._pending_persisted_fit_snapshot is None
    xr.testing.assert_equal(restored.edge_center, win.edge_center)
    xr.testing.assert_equal(restored.edge_stderr, win.edge_stderr)


def test_goldtool_deferred_restore_resaves_fit_payload_before_show(
    qtbot, gold, monkeypatch
) -> None:
    win: GoldTool = goldtool(gold, execute=False, data_name="gold_input")
    qtbot.addWidget(win)
    _configure_goldtool_state(win, fitted=True, spline=True)
    expected_corrected = win.corrected
    saved = win.to_dataset()

    post_fit_calls = _spy_goldtool_post_fit(monkeypatch)
    restored = erlab.interactive.utils.ToolWindow.from_dataset(
        saved, _defer_restore_work=True
    )
    qtbot.addWidget(restored)
    assert isinstance(restored, GoldTool)

    resaved = restored.to_dataset()

    assert post_fit_calls == []
    assert restored.result is None
    assert restored._pending_persisted_fit_snapshot is not None
    assert GoldTool._PERSISTED_FIT_ALONG_VAR in resaved
    assert GoldTool._PERSISTED_EDGE_CENTER_VAR in resaved
    assert GoldTool._PERSISTED_EDGE_STDERR_VAR in resaved

    eager_restored = erlab.interactive.utils.ToolWindow.from_dataset(resaved)
    qtbot.addWidget(eager_restored)
    assert isinstance(eager_restored, GoldTool)

    assert post_fit_calls == [eager_restored]
    assert isinstance(eager_restored.result, scipy.interpolate.BSpline)
    xr.testing.assert_identical(eager_restored.corrected, expected_corrected)


def test_goldtool_deferred_restore_update_data_refits_pending_fit(
    qtbot, gold, monkeypatch
) -> None:
    win: GoldTool = goldtool(gold, execute=False, data_name="gold_input")
    qtbot.addWidget(win)
    _configure_goldtool_state(win, fitted=True, spline=True)
    saved = win.to_dataset()

    restored = erlab.interactive.utils.ToolWindow.from_dataset(
        saved, _defer_restore_work=True
    )
    qtbot.addWidget(restored)
    assert isinstance(restored, GoldTool)
    assert not hasattr(restored, "edge_center")
    assert restored._pending_persisted_fit_snapshot is not None

    called: list[bool] = []
    monkeypatch.setattr(restored, "perform_edge_fit", lambda: called.append(True))
    new_gold = gold.copy(deep=True)
    new_gold.data = np.asarray(new_gold.data) * 1.02

    assert restored.update_data(new_gold) is False

    assert called == [True]
    assert restored._pending_persisted_fit_snapshot is None
    assert not hasattr(restored, "edge_center")
    xr.testing.assert_identical(restored.data, new_gold)


@pytest.mark.parametrize("defer_restore_work", [False, True])
def test_goldtool_legacy_bad_fit_snapshot_lengths_raise(
    qtbot, gold, defer_restore_work, monkeypatch
) -> None:
    win: GoldTool = goldtool(gold, execute=False, data_name="gold_input")
    qtbot.addWidget(win)

    saved = win.to_dataset()
    legacy_state = json.loads(saved.attrs["tool_state"])
    legacy_state["fit_snapshot"] = {
        "along_coords": [0.0, 1.0],
        "edge_center": [0.0],
        "edge_stderr": [0.0, 0.0],
    }
    saved.attrs["tool_state"] = json.dumps(legacy_state)

    deleted_tools: list[GoldTool] = []
    original_delete_later = GoldTool.deleteLater

    def tracked_delete_later(self: GoldTool) -> None:
        deleted_tools.append(self)
        original_delete_later(self)

    monkeypatch.setattr(GoldTool, "deleteLater", tracked_delete_later)

    with pytest.raises(ValueError, match="mismatched array lengths"):
        erlab.interactive.utils.ToolWindow.from_dataset(
            saved, _defer_restore_work=defer_restore_work
        )
    assert len(deleted_tools) == 1


def test_goldtool_handles_missing_cpu_count(qtbot, gold, monkeypatch) -> None:
    monkeypatch.setattr("erlab.interactive.fermiedge.os.cpu_count", lambda: None)

    win: GoldTool = goldtool(gold, execute=False)
    qtbot.addWidget(win)

    cpu_widget = typing.cast("QtWidgets.QSpinBox", win.params_edge.widgets["# CPU"])
    assert cpu_widget.value() == 1
    assert cpu_widget.maximum() == 1


def test_goldtool_duplicate_roundtrip(qtbot, gold) -> None:
    win: GoldTool = goldtool(gold, execute=False, data_name="gold_input")
    qtbot.addWidget(win)
    _configure_goldtool_state(win, fitted=True, spline=True)

    duplicated = win.duplicate()
    qtbot.addWidget(duplicated)

    assert isinstance(duplicated, GoldTool)
    assert duplicated.tool_status == win.tool_status
    assert isinstance(duplicated.result, scipy.interpolate.BSpline)
    xr.testing.assert_identical(win.corrected, duplicated.corrected)


@pytest.mark.parametrize("operation", ["to_file", "duplicate"])
def test_goldtool_rejects_serialization_with_separate_data_corr(
    qtbot, gold, operation, tmp_path
) -> None:
    corrected = gold.copy(deep=True)
    corrected.data = np.asarray(corrected.data) * 1.02

    win: GoldTool = goldtool(gold, data_corr=corrected, execute=False)
    qtbot.addWidget(win)

    if operation == "to_file":
        with pytest.raises(
            ValueError,
            match="unsupported when `data_corr` is provided separately",
        ):
            win.to_file(tmp_path / "goldtool.h5")
    else:
        with pytest.raises(
            ValueError,
            match="unsupported when `data_corr` is provided separately",
        ):
            win.duplicate()


def test_goldtool_copy_code_includes_separate_data_corr(qtbot, gold) -> None:
    corrected = gold.copy(deep=True)
    corrected.data = np.asarray(corrected.data) * 1.02

    win: GoldTool = goldtool(gold, data_corr=corrected, execute=False)
    qtbot.addWidget(win)
    win._argnames["data"] = "gold_data"
    win._argnames["data_corr"] = "corr_data"

    code = win.current_provenance_spec().display_code()
    assert code is not None
    assert "model_result = era.gold.poly(" in code
    assert "corrected = era.gold.correct_with_edge(corr_data, model_result)" in code
    assert "modelresult" not in code


def test_goldtool_roi_limits_descending_coords(qtbot, gold) -> None:
    gold_desc = gold.isel(alpha=slice(None, None, -1), eV=slice(None, None, -1))
    win: GoldTool = goldtool(gold_desc, execute=False)
    qtbot.addWidget(win)

    win.params_roi.modify_roi(x0=-10.0, x1=10.0, y0=-0.5, y1=0.2)
    x0, y0, x1, y1 = win.roi_limits_ordered
    assert x0 == pytest.approx(10.0)
    assert x1 == pytest.approx(-10.0)
    assert y0 == pytest.approx(0.2)
    assert y1 == pytest.approx(-0.5)


def test_goldtool_update_data_invalidates_fit_and_can_refit(
    qtbot, gold, monkeypatch
) -> None:
    win: GoldTool = goldtool(gold, execute=False)
    qtbot.addWidget(win)

    win.params_edge.widgets["T (K)"].setValue(45.0)
    degree_widget = typing.cast("QtWidgets.QSpinBox", win.params_poly.widgets["Degree"])
    with QtCore.QSignalBlocker(win.params_poly):
        degree_widget.setValue(3)
    win.params_roi.modify_roi(x0=-8.0, x1=8.0, y0=-0.2, y1=0.1)
    win.params_poly.setDisabled(False)
    win.params_spl.setDisabled(False)
    win.params_tab.setDisabled(False)
    win.edge_center = gold.mean("eV")
    win.edge_stderr = xr.ones_like(win.edge_center)
    win.result = xr.Dataset()

    called: list[bool] = []
    monkeypatch.setattr(win, "perform_edge_fit", lambda: called.append(True))

    new_gold = gold.copy(deep=True)
    new_gold.data = np.asarray(new_gold.data) * 1.05
    win.update_data(new_gold)

    assert win.params_edge.values["T (K)"] == pytest.approx(45.0)
    assert win.params_poly.values["Degree"] == 3
    assert win.result is None
    assert win.params_tab.isEnabled() is False
    assert not called

    win.edge_center = new_gold.mean("eV")
    win.edge_stderr = xr.ones_like(win.edge_center)
    win.result = xr.Dataset()
    win.refit_on_source_update_check.setChecked(True)

    newer_gold = new_gold.copy(deep=True)
    newer_gold.data = np.asarray(newer_gold.data) * 1.02
    win.update_data(newer_gold)

    assert called == [True]


def test_goldtool_update_data_ignores_late_results_from_aborted_task(
    qtbot, gold, monkeypatch
) -> None:
    win: GoldTool = goldtool(gold, execute=False)
    qtbot.addWidget(win)

    class _DummyTask:
        def __init__(self) -> None:
            self.aborted = False
            self.signals = EdgeFitSignals()

        def abort_fit(self) -> None:
            self.aborted = True

    stale_task = _DummyTask()
    win._fit_task = stale_task

    perform_called: list[bool] = []
    monkeypatch.setattr(win, "perform_fit", lambda: perform_called.append(True))

    new_gold = gold.copy(deep=True)
    new_gold.data = np.asarray(new_gold.data) * 1.01
    win.update_data(new_gold)

    assert stale_task.aborted is True
    assert win._fit_task is None
    assert not hasattr(win, "edge_center")
    assert not perform_called

    stale_center = gold.mean("eV")
    stale_stderr = xr.ones_like(stale_center)
    win.post_fit(stale_center, stale_stderr, task=stale_task)

    assert not hasattr(win, "edge_center")
    assert not perform_called

    win.progress.setMaximum(5)
    win.progress.setValue(0)
    win.pbar.setFormat("unchanged")
    win.iterated(3, task=typing.cast("typing.Any", stale_task))

    assert win.progress.value() == 0
    assert win.pbar.format() == "unchanged"

    win.iterated(0)

    assert win.progress.value() == 0
    assert win.pbar.format() == f"0/{win.progress.maximum()} finished"


def test_goldtool_abort_fit_task_disconnects_late_worker_signals(qtbot, gold) -> None:
    win: GoldTool = goldtool(gold, execute=False)
    qtbot.addWidget(win)
    task = EdgeFitTask(
        win.data,
        win._along_dim,
        *win.roi_limits_ordered,
        dict(win.params_edge.values),
    )
    win._fit_task = task

    late_calls: list[str] = []
    task.signals.sigIterated.connect(lambda _n: late_calls.append("iterated"))
    task.signals.sigFinished.connect(
        lambda _center, _stderr: late_calls.append("finished")
    )
    task.signals.sigFailed.connect(lambda _message: late_calls.append("failed"))

    win._abort_fit_task()
    task.signals.sigIterated.emit(1)
    edge_center = gold.mean("eV")
    edge_stderr = xr.ones_like(edge_center)
    task.signals.sigFinished.emit(edge_center, edge_stderr)
    task.signals.sigFailed.emit("late failure")

    assert win._fit_task is None
    assert late_calls == []


def test_edge_fit_task_aborted_before_run_skips_fit(gold, monkeypatch) -> None:
    task = EdgeFitTask(
        gold,
        "alpha",
        float(gold.alpha.min()),
        float(gold.eV.min()),
        float(gold.alpha.max()),
        float(gold.eV.max()),
        {
            "# CPU": 1,
            "Bin x": 1,
            "Bin y": 1,
            "T (K)": 30.0,
            "Fix T": False,
            "Linear": True,
            "Resolution": 0.01,
            "Fast": True,
            "Method": "leastsq",
            "Scale cov": True,
        },
    )
    events: list[str] = []
    task.signals.sigIterated.connect(lambda _n: events.append("iterated"))
    task.signals.sigFinished.connect(lambda _center, _stderr: events.append("finished"))
    task.signals.sigFailed.connect(lambda _message: events.append("failed"))

    def fail_edge(*_args, **_kwargs):
        raise AssertionError("aborted task should not start fitting")

    monkeypatch.setattr(erlab.analysis.gold, "edge", fail_edge)

    task.abort_fit()
    task.run()

    assert events == []


def test_edge_fit_task_aborted_after_progress_skips_fit(gold, monkeypatch) -> None:
    task = EdgeFitTask(
        gold,
        "alpha",
        float(gold.alpha.min()),
        float(gold.eV.min()),
        float(gold.alpha.max()),
        float(gold.eV.max()),
        {
            "# CPU": 1,
            "Bin x": 1,
            "Bin y": 1,
            "T (K)": 30.0,
            "Fix T": False,
            "Linear": True,
            "Resolution": 0.01,
            "Fast": True,
            "Method": "leastsq",
            "Scale cov": True,
        },
    )
    events: list[str] = []

    def abort_after_progress(_n: int) -> None:
        events.append("iterated")
        task.abort_fit()

    task.signals.sigIterated.connect(abort_after_progress)
    task.signals.sigFinished.connect(lambda _center, _stderr: events.append("finished"))
    task.signals.sigFailed.connect(lambda _message: events.append("failed"))

    def fail_edge(*_args, **_kwargs):
        raise AssertionError("aborted task should not start fitting")

    monkeypatch.setattr(erlab.analysis.gold, "edge", fail_edge)

    task.run()

    assert events == ["iterated"]


def test_edge_fit_task_aborted_during_parallel_setup_skips_fit(
    gold, monkeypatch
) -> None:
    task = EdgeFitTask(
        gold,
        "alpha",
        float(gold.alpha.min()),
        float(gold.eV.min()),
        float(gold.alpha.max()),
        float(gold.eV.max()),
        {
            "# CPU": 1,
            "Bin x": 1,
            "Bin y": 1,
            "T (K)": 30.0,
            "Fix T": False,
            "Linear": True,
            "Resolution": 0.01,
            "Fast": True,
            "Method": "leastsq",
            "Scale cov": True,
        },
    )
    events: list[str] = []
    task.signals.sigIterated.connect(lambda _n: events.append("iterated"))
    task.signals.sigFinished.connect(lambda _center, _stderr: events.append("finished"))
    task.signals.sigFailed.connect(lambda _message: events.append("failed"))

    class _FakeParallel:
        def __init__(self) -> None:
            self._aborting = False
            self._exception = False

    fake_parallel = _FakeParallel()

    def make_parallel(*_args, **_kwargs):
        task.abort_fit()
        return fake_parallel

    monkeypatch.setattr("erlab.interactive.fermiedge.joblib.Parallel", make_parallel)

    task.run()

    assert fake_parallel._aborting
    assert fake_parallel._exception
    assert events == []


def test_edge_fit_task_aborted_after_fit_skips_finished_signal(
    gold, monkeypatch
) -> None:
    task = EdgeFitTask(
        gold,
        "alpha",
        float(gold.alpha.min()),
        float(gold.eV.min()),
        float(gold.alpha.max()),
        float(gold.eV.max()),
        {
            "# CPU": 1,
            "Bin x": 1,
            "Bin y": 1,
            "T (K)": 30.0,
            "Fix T": False,
            "Linear": True,
            "Resolution": 0.01,
            "Fast": True,
            "Method": "leastsq",
            "Scale cov": True,
        },
    )
    events: list[str] = []
    task.signals.sigIterated.connect(lambda _n: events.append("iterated"))
    task.signals.sigFinished.connect(lambda _center, _stderr: events.append("finished"))
    task.signals.sigFailed.connect(lambda _message: events.append("failed"))

    edge_center = gold.mean("eV")
    edge_stderr = xr.ones_like(edge_center)

    def aborting_edge(*_args, **_kwargs):
        task.abort_fit()
        return edge_center, edge_stderr

    monkeypatch.setattr(erlab.analysis.gold, "edge", aborting_edge)

    task.run()

    assert events == ["iterated"]


def test_goldtool_late_task_results_do_not_update_while_closing(qtbot, gold) -> None:
    win: GoldTool = goldtool(gold, execute=False)
    qtbot.addWidget(win)
    task = EdgeFitTask(
        win.data,
        win._along_dim,
        *win.roi_limits_ordered,
        dict(win.params_edge.values),
    )
    win._fit_task = task
    win._fit_closing = True

    edge_center = gold.mean("eV")
    edge_stderr = xr.ones_like(edge_center)
    win.post_fit(edge_center, edge_stderr, task=task)

    assert win._fit_task is task
    assert not hasattr(win, "edge_center")


def test_goldtool_perform_edge_fit_ignores_pending_or_closing_updates(
    qtbot, gold
) -> None:
    win: GoldTool = goldtool(gold, execute=False)
    qtbot.addWidget(win)

    win._pending_update_request = object()  # type: ignore[assignment]
    win.perform_edge_fit()

    assert win._fit_task is None

    class _ThreadPoolDouble:
        def __init__(self) -> None:
            self.started_tasks: list[EdgeFitTask] = []

        def activeThreadCount(self) -> int:
            return 0

        def start(self, task: EdgeFitTask) -> None:
            self.started_tasks.append(task)

    threadpool = _ThreadPoolDouble()
    win._pending_update_request = None
    win._fit_closing = False
    win._threadpool = threadpool  # type: ignore[assignment]
    win.perform_edge_fit()

    assert threadpool.started_tasks == [win._fit_task]

    win._abort_fit_task()
    win._fit_closing = True
    win.perform_edge_fit()

    assert win._fit_task is None


@pytest.mark.parametrize(
    "empty_case", ["empty_roi", "overlarge_bin_x", "overlarge_bin_y"]
)
def test_goldtool_perform_edge_fit_skips_empty_selection(
    qtbot, gold, monkeypatch, empty_case
) -> None:
    win: GoldTool = goldtool(gold, execute=False)
    qtbot.addWidget(win)

    if empty_case == "empty_roi":
        between = float((gold.alpha.values[0] + gold.alpha.values[1]) / 2)
        win.params_roi._x_decimals = 12
        win.params_roi.modify_roi(x0=between, x1=between, y0=-0.2, y1=0.1)
    elif empty_case == "overlarge_bin_x":
        typing.cast("QtWidgets.QSpinBox", win.params_edge.widgets["Bin x"]).setValue(
            gold.sizes["alpha"] + 1
        )
    else:
        typing.cast("QtWidgets.QSpinBox", win.params_edge.widgets["Bin y"]).setValue(
            gold.sizes["eV"] + 1
        )

    class _ThreadPoolDouble:
        def __init__(self) -> None:
            self.started_tasks: list[EdgeFitTask] = []

        def activeThreadCount(self) -> int:
            return 0

        def start(self, task: EdgeFitTask) -> None:
            self.started_tasks.append(task)

    warnings: list[bool] = []
    threadpool = _ThreadPoolDouble()
    monkeypatch.setattr(
        win, "_show_empty_fit_selection_warning", lambda: warnings.append(True)
    )
    win._threadpool = threadpool  # type: ignore[assignment]

    win.perform_edge_fit()

    assert warnings == [True]
    assert win._fit_task is None
    assert threadpool.started_tasks == []


def test_goldtool_post_fit_accepts_current_task_and_disconnects(
    qtbot, gold, monkeypatch
) -> None:
    win: GoldTool = goldtool(gold, execute=False)
    qtbot.addWidget(win)
    task = EdgeFitTask(
        win.data,
        win._along_dim,
        *win.roi_limits_ordered,
        dict(win.params_edge.values),
    )
    win._fit_task = task

    disconnected: list[EdgeFitTask] = []
    perform_called: list[bool] = []
    replace_called: list[bool] = []
    monkeypatch.setattr(
        GoldTool,
        "_disconnect_fit_task_signals",
        staticmethod(lambda fit_task: disconnected.append(fit_task)),
    )
    monkeypatch.setattr(win, "perform_fit", lambda: perform_called.append(True))
    monkeypatch.setattr(win, "_replace_last_state", lambda: replace_called.append(True))

    edge_center = gold.mean("eV")
    edge_stderr = xr.ones_like(edge_center)
    win.post_fit(edge_center, edge_stderr, task=task)

    assert disconnected == [task]
    assert win._fit_task is None
    assert perform_called == [True]
    assert replace_called == [True]
    xr.testing.assert_identical(win.edge_center, edge_center)
    xr.testing.assert_identical(win.edge_stderr, edge_stderr)

    win._fit_closing = True
    rejected_center = edge_center + 1.0
    win.post_fit(rejected_center, edge_stderr)

    assert perform_called == [True]
    assert replace_called == [True]
    xr.testing.assert_identical(win.edge_center, edge_center)


def test_goldtool_handle_fit_failed_ignores_stale_and_closing_tasks(
    qtbot, gold, monkeypatch
) -> None:
    win: GoldTool = goldtool(gold, execute=False)
    qtbot.addWidget(win)
    current_task = EdgeFitTask(
        win.data,
        win._along_dim,
        *win.roi_limits_ordered,
        dict(win.params_edge.values),
    )
    stale_task = EdgeFitTask(
        win.data,
        win._along_dim,
        *win.roi_limits_ordered,
        dict(win.params_edge.values),
    )
    win._fit_task = current_task

    critical_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    disconnected: list[EdgeFitTask] = []
    monkeypatch.setattr(
        GoldTool,
        "_disconnect_fit_task_signals",
        staticmethod(lambda fit_task: disconnected.append(fit_task)),
    )
    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        lambda *args, **kwargs: critical_calls.append((args, kwargs)),
    )

    win._handle_fit_failed("stale failure", task=stale_task)

    assert win._fit_task is current_task
    assert disconnected == []
    assert critical_calls == []

    win._fit_closing = True
    win._handle_fit_failed("closing failure")

    assert win._fit_task is current_task
    assert disconnected == []
    assert critical_calls == []

    win._fit_closing = False
    win._handle_fit_failed("current failure", task=current_task)

    assert win._fit_task is None
    assert disconnected == [current_task]
    assert len(critical_calls) == 1

    win._handle_fit_failed("taskless failure")

    assert len(critical_calls) == 2


def test_goldtool_update_data_defers_until_fit_worker_drains(
    qtbot, gold, monkeypatch
) -> None:
    win: GoldTool = goldtool(gold, execute=False)
    qtbot.addWidget(win)

    class _DummyTask:
        def __init__(self) -> None:
            self.aborted = False
            self.signals = EdgeFitSignals()

        def abort_fit(self) -> None:
            self.aborted = True

    stale_task = _DummyTask()
    win._fit_task = stale_task

    active_counts = iter((1, 0))
    monkeypatch.setattr(
        win._threadpool,
        "activeThreadCount",
        lambda: next(active_counts, 0),
    )

    new_gold = gold.copy(deep=True)
    new_gold.data = np.asarray(new_gold.data) * 1.02
    win.update_data(new_gold)

    assert stale_task.aborted is True
    assert win._pending_update_request is not None
    xr.testing.assert_identical(win.data, gold)

    win._pending_update_timer.stop()
    win._flush_pending_update()

    assert win._pending_update_request is None
    xr.testing.assert_identical(win.data, new_gold)


def test_goldtool_auto_source_update_stays_stale_until_deferred_refresh_applies(
    qtbot, gold, monkeypatch
) -> None:
    win: GoldTool = goldtool(gold, execute=False)
    qtbot.addWidget(win)
    win.set_source_binding(
        erlab.interactive.imagetool.provenance.full_data(),
        auto_update=True,
    )

    class _DummyTask:
        def __init__(self) -> None:
            self.aborted = False
            self.signals = EdgeFitSignals()

        def abort_fit(self) -> None:
            self.aborted = True

    stale_task = _DummyTask()
    win._fit_task = stale_task

    active_counts = iter((1, 0))
    monkeypatch.setattr(
        win._threadpool,
        "activeThreadCount",
        lambda: next(active_counts, 0),
    )

    new_gold = gold.copy(deep=True)
    new_gold.data = np.asarray(new_gold.data) * 1.02
    win.handle_parent_source_replaced(new_gold)

    assert stale_task.aborted is True
    assert win.source_state == "stale"
    xr.testing.assert_identical(win.data, gold)

    win._pending_update_timer.stop()
    win._flush_pending_update()

    assert win.source_state == "fresh"
    xr.testing.assert_identical(win.data, new_gold)


def test_goldtool_auto_source_update_emits_one_data_change(qtbot, gold) -> None:
    win: GoldTool = goldtool(gold, execute=False)
    qtbot.addWidget(win)
    win.set_source_binding(
        erlab.interactive.imagetool.provenance.full_data(),
        auto_update=True,
    )

    emissions: list[bool] = []
    win.sigDataChanged.connect(lambda: emissions.append(True))

    new_gold = gold.copy(deep=True)
    new_gold.data = np.asarray(new_gold.data) * 1.02
    win.handle_parent_source_replaced(new_gold)

    assert win.source_state == "fresh"
    assert emissions == [True]
    xr.testing.assert_identical(win.data, new_gold)


def test_goldtool_auto_source_update_with_refit_stays_stale_until_fit_finishes(
    qtbot, gold, monkeypatch
) -> None:
    win: GoldTool = goldtool(gold, execute=False)
    qtbot.addWidget(win)
    _configure_goldtool_state(win, fitted=True)
    win.set_source_binding(
        erlab.interactive.imagetool.provenance.full_data(),
        auto_update=True,
    )

    fit_started: list[bool] = []
    monkeypatch.setattr(win, "perform_edge_fit", lambda: fit_started.append(True))

    new_gold = gold.copy(deep=True)
    new_gold.data = np.asarray(new_gold.data) * 1.02
    win.handle_parent_source_replaced(new_gold)

    assert fit_started == [True]
    assert win.source_state == "stale"
    assert win.result is None

    edge_center = win.data.mean("eV")
    edge_stderr = xr.ones_like(edge_center)
    win.post_fit(edge_center, edge_stderr)

    assert win.source_state == "fresh"


def test_goldtool_close_event_ignored_if_threadpool_does_not_quiesce(
    qtbot, gold, monkeypatch
) -> None:
    win: GoldTool = goldtool(gold, execute=False)
    qtbot.addWidget(win)
    threadpool_ready = False

    def _wait_for_threadpool(*args, **kwargs) -> bool:
        return threadpool_ready

    monkeypatch.setattr(
        type(win),
        "_wait_for_threadpool",
        staticmethod(_wait_for_threadpool),
    )

    event = QtGui.QCloseEvent()
    assert event.isAccepted()
    win.closeEvent(event)
    assert not event.isAccepted()
    assert not win._fit_closing
    threadpool_ready = True
    win.close()


def test_goldtool_update_data_clamps_roi_to_non_empty_bounds(qtbot, gold) -> None:
    win: GoldTool = goldtool(gold, execute=False)
    qtbot.addWidget(win)

    win.params_roi.modify_roi(x0=-12.0, x1=-8.0, y0=-0.45, y1=-0.3)
    narrowed = gold.sel(alpha=slice(-2.5, 2.5), eV=slice(-0.1, 0.05))

    win.update_data(narrowed)

    x0, y0, x1, y1 = win.params_roi.roi_limits
    xmin, ymin, xmax, ymax = win.params_roi.max_bounds
    assert xmin <= x0 < x1 <= xmax
    assert y0 < y1 <= ymax
    assert y0 >= ymin - 1e-3

    sel_x0, sel_y0, sel_x1, sel_y1 = win.roi_limits_ordered
    selected = win.data.sel(
        {win._along_dim: slice(sel_x0, sel_x1), "eV": slice(sel_y0, sel_y1)}
    )
    assert selected.sizes[win._along_dim] > 0
    assert selected.sizes["eV"] > 0


def test_goldtool_validate_update_data_transposes_and_rejects_invalid(
    qtbot, gold
) -> None:
    win: GoldTool = goldtool(gold, execute=False)
    qtbot.addWidget(win)

    transposed = win.validate_update_data(gold.transpose(*reversed(gold.dims)))
    assert transposed.dims[0] == "eV"

    with pytest.raises(ValueError, match="2D DataArray with an `eV` dimension"):
        win.validate_update_data(xr.DataArray(np.arange(5), dims=("alpha",)))


def test_goldtool_undo_redo_state_change(qtbot, gold) -> None:
    win: GoldTool = goldtool(gold, execute=False)
    qtbot.addWidget(win)

    initial = win.tool_status
    win.refit_on_source_update_check.setChecked(not initial.refit_on_source_update)

    assert win._flush_pending_history_write()
    assert win.undoable is True
    assert win.tool_status.refit_on_source_update is not initial.refit_on_source_update

    win.undo()

    assert win.tool_status == initial
    assert win.redoable is True

    win.redo()

    assert win.tool_status.refit_on_source_update is not initial.refit_on_source_update


def test_restool(qtbot) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)
    win.timeout_spin.setValue(5.0)

    win._guess_temp()
    qtbot.wait_until(lambda: win.fit_params["temp"] == 100.0, timeout=1000)

    win._guess_center()
    win.res_spin.setValue(0.02)
    win.live_check.setChecked(True)
    win.y0_spin.setValue(-12.0)
    win.x0_spin.setValue(-0.3)
    win.x1_spin.setValue(0.3)

    qtbot.wait_until(lambda: isinstance(win._result_ds, xr.Dataset), timeout=10000)

    for k, v in {
        "eV_range": (-0.3, 0.3),
        "temp": 100.0,
        "resolution": 0.02,
        "center": -0.01037,
        "bkg_slope": False,
    }.items():
        assert win.fit_params[k] == v

    def check_generated_code(w: ResolutionTool) -> None:
        namespace = {"era": erlab.analysis, "gold": gold, "data": gold, "result": None}
        code = w.copy_code().replace("quick_resolution", "quick_fit")
        exec(code, {"__builtins__": {"slice": slice}}, namespace)  # noqa: S102

        xr.testing.assert_identical(
            w._result_ds.drop_vars("modelfit_results"),
            namespace["result"].drop_vars("modelfit_results"),
        )

    check_generated_code(win)

    # Test tool save & restore
    tmp_dir = tempfile.TemporaryDirectory()
    filename = f"{tmp_dir.name}/tool_save.h5"
    win.to_file(filename)

    win_restored = erlab.interactive.utils.ToolWindow.from_file(filename)
    qtbot.addWidget(win_restored)
    assert isinstance(win_restored, ResolutionTool)

    assert win.tool_status == win_restored.tool_status
    assert str(win_restored.info_text) == str(win.info_text)

    tmp_dir.cleanup()


def test_restool_deferred_restore_live_fit_does_not_trigger_fit(qtbot, monkeypatch):
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, data_name="resolution_input", execute=False)
    qtbot.addWidget(win)
    win.live_check.setChecked(True)
    saved = win.to_dataset()
    calls: list[ResolutionTool] = []

    def _tracked_start_fit_worker(self: ResolutionTool) -> bool:
        calls.append(self)
        return True

    monkeypatch.setattr(
        ResolutionTool,
        "_start_fit_worker",
        _tracked_start_fit_worker,
    )
    monkeypatch.setattr(
        erlab.interactive.utils.varname,
        "argname",
        lambda *_args, **_kwargs: pytest.fail(
            "deferred restool restore should not inspect the caller frame"
        ),
    )

    restored = erlab.interactive.utils.ToolWindow.from_dataset(
        saved,
        _defer_restore_work=True,
    )
    qtbot.addWidget(restored)
    assert isinstance(restored, ResolutionTool)
    assert restored.data_name == "resolution_input"
    assert restored.live_check.isChecked()
    assert calls == []

    restored.do_fit()

    assert calls == [restored]


def test_restool_undo_redo_state_change(qtbot) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)

    initial = win.tool_status
    win.temp_spin.setValue(initial.temp + 1.0)

    assert win._flush_pending_history_write()
    assert win.undoable is True
    assert win.tool_status.temp == initial.temp + 1.0

    win.undo()

    assert win.tool_status == initial
    assert win.redoable is True

    win.redo()

    assert win.tool_status.temp == initial.temp + 1.0


def test_restool_timeout_cleans_up_worker(qtbot, monkeypatch) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)

    def _slow_quick_fit(*args, **kwargs) -> xr.Dataset:
        iter_cb = kwargs["iter_cb"]
        while True:
            if iter_cb():
                raise RuntimeError("timed out")
            time.sleep(0.01)

    monkeypatch.setattr(erlab.analysis.gold, "quick_fit", _slow_quick_fit)

    win.live_check.setChecked(True)
    win.timeout_spin.setValue(0.1)
    win.do_fit()

    qtbot.wait_until(lambda: "Fit timed out" in win.overview_label.text(), timeout=2000)

    assert not win.live_check.isChecked()
    assert win._fit_thread is None
    assert win._result_ds is None


def test_restool_loads_legacy_saved_state_without_source_update_flag(qtbot) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        filename = f"{tmp_dir_name}/tool_save.h5"
        win.to_file(filename)

        with xr.load_dataset(filename, engine="h5netcdf") as ds:
            legacy_ds = ds.load()

        legacy_state = json.loads(legacy_ds.attrs["tool_state"])
        legacy_state.pop("refit_on_source_update", None)
        legacy_ds.attrs["tool_state"] = json.dumps(legacy_state)
        legacy_ds.to_netcdf(filename, engine="h5netcdf", invalid_netcdf=True)

        win_restored = erlab.interactive.utils.ToolWindow.from_file(filename)
        qtbot.addWidget(win_restored)
        assert isinstance(win_restored, ResolutionTool)
        assert win_restored.tool_status.refit_on_source_update is False
        assert win_restored.refit_on_source_update_check.isChecked() is False


def test_restool_update_data_invalidates_fit_and_can_refit(qtbot, monkeypatch) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)

    win.x0_spin.setValue(-0.2)
    win.x1_spin.setValue(0.2)
    win.y0_spin.setValue(-10.0)
    win.y1_spin.setValue(10.0)
    win.live_check.setChecked(True)
    win.temp_spin.setValue(90.0)
    win.fix_temp_check.setChecked(False)
    win.center_spin.setValue(-0.01)
    win.fix_center_check.setChecked(True)
    win.res_spin.setValue(0.015)
    win.fix_res_check.setChecked(True)
    win.slope_check.setChecked(False)
    win.timeout_spin.setValue(2.5)
    win.nfev_spin.setValue(250)
    win.refit_on_source_update_check.setChecked(False)
    win._result_ds = xr.Dataset()

    fit_inputs: list[xr.DataArray] = []
    monkeypatch.setattr(
        win, "do_fit", lambda: fit_inputs.append(win.averaged_edc.copy(deep=True))
    )

    status = win.tool_status
    new_gold = gold.copy(deep=True)
    new_gold.data = np.asarray(new_gold.data) * 1.03
    win.update_data(new_gold)

    expected = status.model_copy(
        update={"results": ("No fit results", "—", "—", "—", "—")}
    )
    assert win.tool_status == expected
    assert win._result_ds is None
    xr.testing.assert_identical(
        win.averaged_edc,
        new_gold.sel({win.y_dim: slice(*win.y_range)}).mean(win.y_dim),
    )
    assert not fit_inputs

    win._result_ds = xr.Dataset()
    win.refit_on_source_update_check.setChecked(True)
    newer_gold = new_gold.copy(deep=True)
    newer_gold.data = np.asarray(newer_gold.data) * 1.01
    win.update_data(newer_gold)

    xr.testing.assert_identical(
        fit_inputs[0], newer_gold.sel({win.y_dim: slice(*win.y_range)}).mean(win.y_dim)
    )


def test_restool_update_data_returns_false_if_fit_thread_stays_alive(qtbot) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)
    original = win.tool_data.copy(deep=True)

    class _StuckThread:
        def __init__(self) -> None:
            self.cancel_called = False
            self.interrupted = False
            self.wait_timeout_ms: int | None = None

        def cancel(self) -> None:
            self.cancel_called = True

        def isRunning(self) -> bool:
            return True

        def requestInterruption(self) -> None:
            self.interrupted = True

        def wait(self, timeout_ms: int) -> bool:
            self.wait_timeout_ms = timeout_ms
            return False

    stuck_thread = _StuckThread()
    win._fit_thread = stuck_thread  # type: ignore[assignment]

    updated = gold.copy(deep=True)
    updated.data = np.asarray(updated.data) * 1.01

    assert win.update_data(updated) is False
    assert stuck_thread.cancel_called
    assert stuck_thread.interrupted
    assert stuck_thread.wait_timeout_ms == win.BACKGROUND_TASK_TIMEOUT_MS
    xr.testing.assert_identical(win.tool_data, original)
    win._fit_thread = None


def test_restool_update_data_auto_refit_after_waiting_cancelled_thread(
    qtbot, monkeypatch
) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)

    class _FinishedThread:
        def __init__(self) -> None:
            self.cancel_called = False
            self.interrupted = False
            self.wait_timeout_ms: int | None = None
            self.deleted = False

        def cancel(self) -> None:
            self.cancel_called = True

        def requestInterruption(self) -> None:
            self.interrupted = True

        def wait(self, timeout_ms: int) -> bool:
            self.wait_timeout_ms = timeout_ms
            return True

        def deleteLater(self) -> None:
            self.deleted = True

    old_thread = _FinishedThread()
    win._fit_thread = old_thread  # type: ignore[assignment]
    win._result_ds = xr.Dataset()
    win.refit_on_source_update_check.setChecked(True)

    started: list[bool] = []

    def _start_fit_worker() -> bool:
        started.append(True)
        assert win._fit_thread is None
        return True

    monkeypatch.setattr(win, "_start_fit_worker", _start_fit_worker)

    updated = gold.copy(deep=True)
    updated.data = np.asarray(updated.data) * 1.01

    assert win.update_data(updated) is False
    assert started == [True]
    assert old_thread.cancel_called
    assert old_thread.interrupted
    assert old_thread.wait_timeout_ms == win.BACKGROUND_TASK_TIMEOUT_MS
    assert old_thread.deleted is True


def test_restool_queue_fit_action_ignores_stale_thread(qtbot) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)

    class _ThreadPlaceholder:
        def cancel(self) -> None:
            return

        def requestInterruption(self) -> None:
            return

        def wait(self, timeout_ms: int) -> bool:
            return True

        def deleteLater(self) -> None:
            return

    stale_thread = _ThreadPlaceholder()
    current_thread = _ThreadPlaceholder()
    win._fit_thread = typing.cast("typing.Any", current_thread)

    called: list[str] = []
    win._queue_fit_action(
        typing.cast("typing.Any", stale_thread), lambda: called.append("stale")
    )
    assert called == []
    assert win._pending_fit_action is None


def test_restool_validate_update_data_transposes_and_rejects_invalid(qtbot) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)

    transposed = win.validate_update_data(gold.transpose(*reversed(gold.dims)))
    assert transposed.dims[-1] == "eV"

    with pytest.raises(ValueError, match="2D and have an 'eV' dimension"):
        win.validate_update_data(xr.DataArray(np.arange(5), dims=("alpha",)))


def test_restool_fit_thread_loads_result_before_emit(monkeypatch) -> None:
    eV = np.linspace(-0.2, 0.2, 31)
    data = xr.DataArray(np.exp(-(eV**2)), dims=("eV",), coords={"eV": eV})

    thread = ResolutionFitThread(
        data,
        {
            "eV_range": (-0.1, 0.1),
            "method": "least_squares",
            "temp": 100.0,
            "resolution": 0.02,
            "center": 0.0,
            "fix_temp": True,
            "fix_center": False,
            "fix_resolution": False,
            "bkg_slope": True,
            "max_nfev": 5,
        },
        timeout=1.0,
    )

    class _DummyResult:
        def __init__(self) -> None:
            self.loaded = False

        def load(self):
            self.loaded = True
            return self

    dummy = _DummyResult()

    def _quick_fit(*_args, **_kwargs):
        return dummy

    captured: dict[str, object | None] = {"result": None, "elapsed": None}

    def _on_finished(result, elapsed: float) -> None:
        captured["result"] = result
        captured["elapsed"] = elapsed

    thread.sigFinished.connect(_on_finished)
    monkeypatch.setattr(erlab.analysis.gold, "quick_fit", _quick_fit)

    thread.run()

    assert dummy.loaded
    assert captured["result"] is dummy
    assert isinstance(captured["elapsed"], float)


def test_restool_fit_thread_cancel_sets_event() -> None:
    eV = np.linspace(-0.2, 0.2, 31)
    data = xr.DataArray(np.exp(-(eV**2)), dims=("eV",), coords={"eV": eV})
    thread = ResolutionFitThread(
        data,
        {
            "eV_range": (-0.1, 0.1),
            "method": "least_squares",
            "temp": 100.0,
            "resolution": 0.02,
            "center": 0.0,
            "fix_temp": True,
            "fix_center": False,
            "fix_resolution": False,
            "bkg_slope": True,
            "max_nfev": 5,
        },
        timeout=1.0,
    )
    thread.cancel()
    assert thread._cancel.is_set()


def test_restool_fit_thread_runtimeerror_interruption_treated_as_cancelled(
    monkeypatch,
) -> None:
    eV = np.linspace(-0.2, 0.2, 31)
    data = xr.DataArray(np.exp(-(eV**2)), dims=("eV",), coords={"eV": eV})

    class _ThreadWithInterruptedWrapper(ResolutionFitThread):
        def isInterruptionRequested(self) -> bool:
            raise RuntimeError("thread wrapper deleted")

    thread = _ThreadWithInterruptedWrapper(
        data,
        {
            "eV_range": (-0.1, 0.1),
            "method": "least_squares",
            "temp": 100.0,
            "resolution": 0.02,
            "center": 0.0,
            "fix_temp": True,
            "fix_center": False,
            "fix_resolution": False,
            "bkg_slope": True,
            "max_nfev": 5,
        },
        timeout=1.0,
    )

    events = {"cancelled": False}

    def _quick_fit(*_args, **kwargs):
        kwargs["iter_cb"]()
        raise RuntimeError("cancel")

    thread.sigCancelled.connect(lambda: events.__setitem__("cancelled", True))
    monkeypatch.setattr(erlab.analysis.gold, "quick_fit", _quick_fit)

    thread.run()
    assert events["cancelled"]


def test_restool_fit_thread_error_emits_errored(monkeypatch) -> None:
    eV = np.linspace(-0.2, 0.2, 31)
    data = xr.DataArray(np.exp(-(eV**2)), dims=("eV",), coords={"eV": eV})
    thread = ResolutionFitThread(
        data,
        {
            "eV_range": (-0.1, 0.1),
            "method": "least_squares",
            "temp": 100.0,
            "resolution": 0.02,
            "center": 0.0,
            "fix_temp": True,
            "fix_center": False,
            "fix_resolution": False,
            "bkg_slope": True,
            "max_nfev": 5,
        },
        timeout=1.0,
    )
    events = {"errored": False}

    def _quick_fit(*_args, **_kwargs):
        raise RuntimeError("boom")

    thread.sigErrored.connect(lambda _msg: events.__setitem__("errored", True))
    monkeypatch.setattr(erlab.analysis.gold, "quick_fit", _quick_fit)
    thread.run()
    assert events["errored"]


def test_restool_fit_thread_cancelled_after_success_emits_cancelled(
    monkeypatch,
) -> None:
    eV = np.linspace(-0.2, 0.2, 31)
    data = xr.DataArray(np.exp(-(eV**2)), dims=("eV",), coords={"eV": eV})
    thread = ResolutionFitThread(
        data,
        {
            "eV_range": (-0.1, 0.1),
            "method": "least_squares",
            "temp": 100.0,
            "resolution": 0.02,
            "center": 0.0,
            "fix_temp": True,
            "fix_center": False,
            "fix_resolution": False,
            "bkg_slope": True,
            "max_nfev": 5,
        },
        timeout=1.0,
    )

    class _DummyResult:
        def load(self):
            return self

    events = {"cancelled": False}

    def _quick_fit(*_args, **kwargs):
        thread._cancel.set()
        kwargs["iter_cb"]()
        return _DummyResult()

    thread.sigCancelled.connect(lambda: events.__setitem__("cancelled", True))
    monkeypatch.setattr(erlab.analysis.gold, "quick_fit", _quick_fit)
    thread.run()
    assert events["cancelled"]


def test_restool_fit_thread_timed_out_after_success_emits_timeout(
    monkeypatch,
) -> None:
    eV = np.linspace(-0.2, 0.2, 31)
    data = xr.DataArray(np.exp(-(eV**2)), dims=("eV",), coords={"eV": eV})
    thread = ResolutionFitThread(
        data,
        {
            "eV_range": (-0.1, 0.1),
            "method": "least_squares",
            "temp": 100.0,
            "resolution": 0.02,
            "center": 0.0,
            "fix_temp": True,
            "fix_center": False,
            "fix_resolution": False,
            "bkg_slope": True,
            "max_nfev": 5,
        },
        timeout=1.0,
    )

    class _DummyResult:
        def load(self):
            return self

    events = {"timed_out": False}

    def _quick_fit(*_args, **kwargs):
        kwargs["iter_cb"]()
        return _DummyResult()

    timer = iter([0.0, 2.0, 3.0])
    monkeypatch.setattr(
        "erlab.interactive.fermiedge.time.perf_counter", lambda: next(timer)
    )
    thread.sigTimedOut.connect(lambda _elapsed: events.__setitem__("timed_out", True))
    monkeypatch.setattr(erlab.analysis.gold, "quick_fit", _quick_fit)
    thread.run()
    assert events["timed_out"]


def test_restool_close_event_ignored_if_fit_thread_stuck(qtbot) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)

    class _StuckThread:
        def __init__(self) -> None:
            self.cancel_called = False
            self.interrupted = False
            self.running = True
            self.wait_timeout_ms: int | None = None

        def cancel(self) -> None:
            self.cancel_called = True

        def isRunning(self) -> bool:
            return self.running

        def requestInterruption(self) -> None:
            self.interrupted = True

        def wait(self, timeout_ms: int) -> bool:
            self.wait_timeout_ms = timeout_ms
            return False

    stuck_thread = _StuckThread()
    win._fit_thread = stuck_thread  # type: ignore[assignment]

    event = QtGui.QCloseEvent()
    assert event.isAccepted()
    win.closeEvent(event)
    assert not event.isAccepted()

    assert stuck_thread.cancel_called
    assert stuck_thread.interrupted
    assert stuck_thread.wait_timeout_ms == 5000
    stuck_thread.running = False
    win.close()


def test_restool_cancel_fit_waits_without_timeout(qtbot) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)

    class _DummyThread:
        def __init__(self) -> None:
            self.cancel_called = False
            self.interrupted = False
            self.wait_args: tuple[object, ...] | None = None
            self.deleted = False

        def cancel(self) -> None:
            self.cancel_called = True

        def isRunning(self) -> bool:
            return True

        def requestInterruption(self) -> None:
            self.interrupted = True

        def wait(self, *args) -> bool:
            self.wait_args = args
            return True

        def deleteLater(self) -> None:
            self.deleted = True

    dummy_thread = _DummyThread()
    win._fit_thread = dummy_thread  # type: ignore[assignment]

    assert win._cancel_fit(wait=True, timeout_ms=None)
    assert dummy_thread.cancel_called
    assert dummy_thread.interrupted
    assert dummy_thread.wait_args == ()
    assert dummy_thread.deleted


def test_restool_clear_fit_preview_keeps_line_when_live(qtbot) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)

    win.edc_fit.setData(x=[0.0, 1.0], y=[1.0, 2.0])
    win.live_check.setChecked(True)
    win._clear_fit_preview()

    x, y = win.edc_fit.getData()
    assert x is not None
    assert y is not None
    assert len(x) == 2
    assert len(y) == 2


def test_restool_clear_fit_preview_clears_line_when_not_live(qtbot) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)

    win.edc_fit.setData(x=[0.0, 1.0], y=[1.0, 2.0])
    win.live_check.setChecked(False)
    win._clear_fit_preview()

    x, y = win.edc_fit.getData()
    assert x is None
    assert y is None


def test_restool_invalidate_displayed_fit_clears_stale_overlay(
    qtbot, monkeypatch
) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)

    win._fit_signature_displayed = ("old",)
    win.edc_fit.setData(x=[0.0, 1.0], y=[1.0, 2.0])
    monkeypatch.setattr(win, "_current_fit_signature", lambda: ("new",))

    signature = win._invalidate_displayed_fit_if_stale()

    assert signature == ("new",)
    assert win._fit_signature_current == ("new",)
    assert win._fit_signature_displayed is None
    x, y = win.edc_fit.getData()
    assert x is None
    assert y is None


def test_restool_handle_fit_success_ignores_stale_result(qtbot) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)

    win._result_ds = None
    win._fit_signature_current = ("current",)
    win._handle_fit_success(xr.Dataset(), 0.1, ("stale",))
    assert win._result_ds is None


def test_restool_do_fit_queues_when_thread_object_exists(qtbot, monkeypatch) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)

    class _ThreadPlaceholder:
        def cancel(self) -> None:
            return

        def requestInterruption(self) -> None:
            return

        def wait(self, timeout_ms: int) -> bool:
            return True

        def deleteLater(self) -> None:
            return

    called: list[bool] = []

    def _start_fit_worker() -> bool:
        called.append(True)
        return True

    monkeypatch.setattr(win, "_start_fit_worker", _start_fit_worker)
    win._fit_thread = _ThreadPlaceholder()  # type: ignore[assignment]
    win._fit_queued = False

    win.do_fit()

    assert win._fit_queued
    assert not called


def test_restool_queue_fit_action_drops_when_cancel_requested(qtbot) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)

    called = {"value": False}
    win._fit_cancel_requested = True
    win._pending_fit_action = None
    win._queue_fit_action(None, lambda: called.__setitem__("value", True))  # type: ignore[arg-type]

    assert not called["value"]
    assert win._pending_fit_action is None


def test_restool_queue_fit_action_drops_while_closing(qtbot) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)

    class _ThreadPlaceholder:
        def cancel(self) -> None:
            return

        def requestInterruption(self) -> None:
            return

        def wait(self, timeout_ms: int) -> bool:
            return True

        def deleteLater(self) -> None:
            return

    thread = _ThreadPlaceholder()
    called = {"value": False}
    win._fit_thread = thread  # type: ignore[assignment]
    win._fit_closing = True
    win._pending_fit_action = None

    win._queue_fit_action(thread, lambda: called.__setitem__("value", True))  # type: ignore[arg-type]

    assert not called["value"]
    assert win._pending_fit_action is None


def test_restool_queue_fit_action_runs_immediately_when_no_thread(qtbot) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)

    called = {"value": False}
    win._fit_cancel_requested = False
    win._fit_thread = None
    win._pending_fit_action = None
    win._queue_fit_action(None, lambda: called.__setitem__("value", True))  # type: ignore[arg-type]

    assert called["value"]
    assert win._pending_fit_action is None


def test_restool_finalize_fit_thread_drops_actions_while_closing(qtbot) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)

    class _ThreadPlaceholder:
        def __init__(self) -> None:
            self.deleted = False

        def deleteLater(self) -> None:
            self.deleted = True

    thread = _ThreadPlaceholder()
    called: list[str] = []
    win._fit_thread = thread  # type: ignore[assignment]
    win._pending_fit_action = lambda: called.append("action")
    win._fit_queued = True
    win._fit_closing = True

    win._finalize_fit_thread(thread)  # type: ignore[arg-type]

    assert called == []
    assert win._fit_thread is None
    assert not win._fit_queued
    assert thread.deleted


def test_restool_start_fit_worker_returns_false_when_thread_exists(qtbot) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)

    class _ThreadPlaceholder:
        def cancel(self) -> None:
            return

        def requestInterruption(self) -> None:
            return

        def wait(self, timeout_ms: int) -> bool:
            return True

        def deleteLater(self) -> None:
            return

    win._fit_thread = _ThreadPlaceholder()  # type: ignore[assignment]
    assert not win._start_fit_worker()
    win._fit_thread = None


def test_restool_start_fit_worker_sets_signature_when_missing(
    qtbot, monkeypatch
) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)

    class _DummySignal:
        def connect(self, *_args, **_kwargs) -> None:
            return

    class _DummyThread:
        def __init__(self, *_args, **_kwargs) -> None:
            self.finished = _DummySignal()
            self.sigFinished = _DummySignal()
            self.sigTimedOut = _DummySignal()
            self.sigErrored = _DummySignal()
            self.sigCancelled = _DummySignal()
            self.started = False

        def cancel(self) -> None:
            return

        def requestInterruption(self) -> None:
            return

        def wait(self, timeout_ms: int) -> bool:
            return True

        def start(self) -> None:
            self.started = True

    monkeypatch.setattr("erlab.interactive.fermiedge.ResolutionFitThread", _DummyThread)
    monkeypatch.setattr(win, "_current_fit_signature", lambda: ("sig",))
    win._fit_thread = None
    win._fit_signature_current = None
    win._fit_cancel_requested = True

    assert win._start_fit_worker()
    assert isinstance(win._fit_thread, _DummyThread)
    assert win._fit_cancel_requested is False
    assert win._fit_thread.started
    win._fit_thread = None


def test_restool_handle_fit_cancelled_returns_none(qtbot) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)
    assert win._handle_fit_cancelled() is None


def test_restool_handle_fit_timeout_ignores_stale_signature(qtbot) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)
    win._result_ds = xr.Dataset({"x": xr.DataArray([1.0])})
    win._fit_signature_displayed = ("current",)
    win.live_check.setChecked(True)

    win._fit_signature_current = ("current",)
    win._handle_fit_timeout(0.1, ("stale",))

    assert win._result_ds is not None
    assert win._fit_signature_displayed == ("current",)
    assert win.live_check.isChecked()


def test_restool_handle_fit_error_updates_ui_for_current_signature(
    qtbot, monkeypatch
) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)

    win._fit_signature_current = ("current",)
    win._fit_signature_displayed = ("current",)
    win._result_ds = xr.Dataset({"x": xr.DataArray([1.0])})
    win.live_check.setChecked(True)
    win.edc_fit.setData(x=[0.0, 1.0], y=[1.0, 2.0])

    failed = {"text": None}
    monkeypatch.setattr(
        win, "_fit_failed", lambda text: failed.__setitem__("text", text)
    )

    win._handle_fit_error("boom", ("current",))

    assert win._result_ds is None
    assert win._fit_signature_displayed is None
    assert not win.live_check.isChecked()
    x, y = win.edc_fit.getData()
    assert x is None
    assert y is None
    assert failed["text"] == "Fit failed"


def test_restool_ranges_descending_coords(qtbot) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    gold_desc = gold.isel(alpha=slice(None, None, -1), eV=slice(None, None, -1))
    win = restool(gold_desc, execute=False)
    qtbot.addWidget(win)

    win.x0_spin.setValue(-0.3)
    win.x1_spin.setValue(0.3)
    win.y0_spin.setValue(-12.0)
    win.y1_spin.setValue(12.0)

    assert win.x_range == pytest.approx((0.3, -0.3))
    assert win.y_range == pytest.approx((12.0, -12.0))
