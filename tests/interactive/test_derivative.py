import tempfile

import numpy as np
import pytest
import xarray as xr

import erlab
from erlab.interactive.derivative import DerivativeTool, dtool


@pytest.mark.parametrize("method_idx", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("interpmode", ["interp", "nointerp"])
@pytest.mark.parametrize(
    ("smoothmode", "nsmooth"),
    [
        ("none", 1),
        ("gaussian", 1),
        ("gaussian", 3),
        ("boxcar", 1),
        ("boxcar", 3),
    ],
)
def test_dtool(qtbot, interpmode, smoothmode, nsmooth, method_idx) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)), dims=["x", "y"], name="data"
    ).astype(np.float64)
    win: DerivativeTool = dtool(data, execute=False)
    qtbot.addWidget(win)

    def check_generated_code(w: DerivativeTool) -> None:
        namespace = {"era": erlab.analysis, "data": data, "np": np, "result": None}
        exec(w.copy_code(), {"__builtins__": {"range": range}}, namespace)  # noqa: S102
        xr.testing.assert_identical(w.result, namespace["result"])

    win.interp_group.setChecked(interpmode == "interp")
    win.smooth_group.setChecked(smoothmode != "none")
    win.sn_spin.setValue(nsmooth)

    match smoothmode:
        case "gaussian":
            win.smooth_combo.setCurrentIndex(0)
        case "boxcar":
            win.smooth_combo.setCurrentIndex(1)

    win.tab_widget.setCurrentIndex(method_idx)
    if method_idx == 1:
        win.lapl_factor_spin.setValue(40)
    elif method_idx == 3:
        win.curv_factor_spin.setValue(40)

    check_generated_code(win)
    win.set_source_binding(
        erlab.interactive.utils.make_tool_source_spec("full_data"),
        auto_update=True,
        state="stale",
    )

    # Test save & restore
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        filename = f"{tmp_dir_name}/tool_save.h5"
        win.to_file(filename)

        win_restored = erlab.interactive.utils.ToolWindow.from_file(filename)
        qtbot.addWidget(win_restored)
        assert isinstance(win_restored, DerivativeTool)

        assert win.tool_status == win_restored.tool_status
        assert str(win_restored.info_text) == str(win.info_text)
        assert win_restored.source_spec == win.source_spec
        assert win_restored.source_auto_update is True
        assert win_restored.source_state == "stale"
        check_generated_code(win_restored)


def test_dtool_update_data_preserves_state(qtbot) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)), dims=["x", "y"], name="data"
    ).astype(np.float64)
    new_data = xr.DataArray(
        np.arange(25, 50).reshape((5, 5)), dims=["x", "y"], name="data"
    ).astype(np.float64)
    win: DerivativeTool = dtool(data, execute=False)
    qtbot.addWidget(win)

    win.interp_group.setChecked(True)
    win.nx_spin.setValue(7)
    win.ny_spin.setValue(9)
    win.smooth_group.setChecked(True)
    win.smooth_combo.setCurrentIndex(1)
    win.sx_spin.setValue(2)
    win.sy_spin.setValue(3)
    win.sn_spin.setValue(2)
    win.tab_widget.setCurrentIndex(3)
    win.curv_a0_spin.setValue(1.5)
    win.curv_factor_spin.setValue(12.0)

    status = win.tool_status
    win.update_data(new_data)

    assert win.tool_status == status
    xr.testing.assert_identical(win.tool_data, new_data)
    assert win.result.shape == win.processed_data.shape


def test_dtool_source_update_marks_unavailable_for_incompatible_data(qtbot) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)), dims=["x", "y"], name="data"
    ).astype(np.float64)
    win: DerivativeTool = dtool(data, execute=False)
    qtbot.addWidget(win)

    win.set_source_binding(
        erlab.interactive.utils.make_tool_source_spec(
            "selection", operations=[{"op": "transpose"}]
        ),
        auto_update=True,
    )

    parent_data = xr.DataArray(
        np.arange(125).reshape((5, 5, 5)),
        dims=("x", "y", "z"),
        coords={"x": np.arange(5), "y": np.arange(5), "z": np.arange(5)},
        name="data",
    )
    win.handle_parent_source_replaced(parent_data)

    assert win.source_state == "unavailable"
    xr.testing.assert_identical(win.tool_data, data)


def test_dtool_full_data_source_update_marks_unavailable_for_incompatible_data(
    qtbot,
) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)), dims=["x", "y"], name="data"
    ).astype(np.float64)
    win: DerivativeTool = dtool(data, execute=False)
    qtbot.addWidget(win)

    win.set_source_binding(
        erlab.interactive.utils.make_tool_source_spec("full_data"),
        auto_update=False,
    )

    parent_data = xr.DataArray(np.arange(5), dims=("x",), name="data")
    win.handle_parent_source_replaced(parent_data)

    assert win.source_state == "unavailable"
    xr.testing.assert_identical(win.tool_data, data)


def test_dtool_restored_source_binding_without_parent_stays_stale(qtbot) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)), dims=["x", "y"], name="data"
    ).astype(np.float64)
    win: DerivativeTool = dtool(data, execute=False)
    qtbot.addWidget(win)

    win.set_source_binding(
        erlab.interactive.utils.make_tool_source_spec("full_data"),
        auto_update=True,
        state="stale",
    )

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        filename = f"{tmp_dir_name}/tool_save.h5"
        win.to_file(filename)

        win_restored = erlab.interactive.utils.ToolWindow.from_file(filename)
        qtbot.addWidget(win_restored)
        assert isinstance(win_restored, DerivativeTool)
        assert win_restored.source_state == "stale"

        assert win_restored._update_from_parent_source() is False
        assert win_restored.source_state == "stale"


def test_dtool_source_update_with_temporarily_missing_parent_stays_stale(qtbot) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)), dims=["x", "y"], name="data"
    ).astype(np.float64)
    updated = xr.DataArray(
        np.arange(25, 50).reshape((5, 5)), dims=["x", "y"], name="data"
    ).astype(np.float64)
    win: DerivativeTool = dtool(data, execute=False)
    qtbot.addWidget(win)

    win.set_source_binding(
        erlab.interactive.utils.make_tool_source_spec("full_data"),
        auto_update=True,
        state="stale",
    )

    available = False

    def fetcher() -> xr.DataArray:
        if not available:
            raise LookupError("Parent tool is temporarily unavailable")
        return updated

    win.set_source_parent_fetcher(fetcher)

    assert win._update_from_parent_source() is False
    assert win.source_state == "stale"

    available = True

    assert win._update_from_parent_source() is True
    assert win.source_state == "fresh"
    xr.testing.assert_identical(win.tool_data, updated)


def test_source_update_dialog_disables_auto_update_without_update_action(qtbot) -> None:
    dialog = erlab.interactive.utils._ToolSourceUpdateDialog(
        None, state="unavailable", auto_update=True
    )
    qtbot.addWidget(dialog)

    assert dialog.update_button.isEnabled() is False
    assert dialog.auto_update_check.isEnabled() is False
