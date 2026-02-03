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

    # Test save & restore
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        filename = f"{tmp_dir_name}/tool_save.h5"
        win.to_file(filename)

        win_restored = erlab.interactive.utils.ToolWindow.from_file(filename)
        qtbot.addWidget(win_restored)
        assert isinstance(win_restored, DerivativeTool)

        assert win.tool_status == win_restored.tool_status
        assert str(win_restored.info_text) == str(win.info_text)
        check_generated_code(win_restored)


def test_smooth_args_handles_singleton_coords(qtbot) -> None:
    data = xr.DataArray(
        np.arange(5).reshape((1, 5)).astype(np.float64),
        dims=["x", "y"],
        coords={"x": np.array([0.0]), "y": np.arange(5, dtype=float)},
        name="data",
    )
    win: DerivativeTool = dtool(data, execute=False)
    qtbot.addWidget(win)

    win.smooth_combo.setCurrentIndex(0)
    args = win.smooth_args

    assert isinstance(args, dict)
