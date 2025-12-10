import numpy as np
import pytest
import xarray as xr
from qtpy import QtWidgets

import erlab
from erlab.interactive._mesh import MeshTool, meshtool


@pytest.fixture
def meshy_data() -> xr.DataArray:
    height = width = 32
    alpha = np.arange(height)
    ev = np.arange(width)

    yy, xx = np.meshgrid(alpha, ev, indexing="ij")
    base = np.exp(-((yy - height / 2) ** 2 + (xx - width / 2) ** 2) / 200)
    mesh_pattern = 1 + 0.1 * np.cos(2 * np.pi * ev / 8)
    mesh_pattern = np.tile(mesh_pattern, (height, 1))

    data = np.stack([base * mesh_pattern, base * mesh_pattern * 1.05], axis=0)
    return xr.DataArray(
        data,
        dims=("rep", "alpha", "eV"),
        coords={"rep": [0, 1], "alpha": alpha, "eV": ev},
        name="mesh_data",
    )


def test_meshtool_update_and_copy_code(qtbot, meshy_data) -> None:
    first_order = [[16, 16], [16, 20], [16, 12]]
    win: MeshTool = meshtool(meshy_data, data_name="mesh_data", execute=False)
    qtbot.addWidget(win)

    win.order_spin.setValue(1)
    win.n_pad_spin.setValue(0)
    win.roi_hw_spin.setValue(8)
    win.k_spin.setValue(0.1)
    win.feather_spin.setValue(1.0)
    win.method_combo.setCurrentText("constant")

    win.p0_spin0.setValue(first_order[1][0])
    win.p0_spin1.setValue(first_order[1][1])
    win.p1_spin0.setValue(first_order[2][0])
    win.p1_spin1.setValue(first_order[2][1])

    win.set_data_beforecalc()
    win.update()

    qtbot.wait_until(lambda: win._corrected is not None and win._mesh is not None)
    assert win._corrected is not None
    assert win._mesh is not None

    (
        corrected,
        mesh_da,
        _,
        log_mag,
        log_mag_corr,
        _,
        mask,
    ) = erlab.analysis.mesh.remove_mesh(
        meshy_data,
        first_order_peaks=first_order,
        order=1,
        n_pad=0,
        roi_hw=8,
        k=0.1,
        feather=1.0,
        full_output=True,
    )

    xr.testing.assert_identical(win._corrected, corrected)
    xr.testing.assert_identical(win._mesh, mesh_da)

    np.testing.assert_allclose(win.main_fft_image.image, log_mag)
    np.testing.assert_allclose(win.corr_fft_image.image, log_mag_corr)
    np.testing.assert_allclose(win.mask_fft_image.image, mask)

    code = win.copy_code()
    namespace: dict[str, object] = {
        "era": erlab.analysis,
        "mesh_data": meshy_data,
        "corrected": None,
        "mesh": None,
    }
    exec(code, {"__builtins__": {}}, namespace)  # noqa: S102

    xr.testing.assert_identical(win._corrected, namespace["corrected"])
    xr.testing.assert_identical(win._mesh, namespace["mesh"])


def test_meshtool_autofind_and_persistence(
    qtbot, meshy_data, tmp_path, monkeypatch
) -> None:
    win: MeshTool = meshtool(meshy_data, execute=False)
    qtbot.addWidget(win)

    win.n_pad_spin.setValue(0)
    win.set_data_beforecalc()

    expected_peaks = (
        erlab.analysis.mesh.find_peaks(
            win.get_reduced()[1], bins=win.bins_spin.value(), n_peaks=2, plot=False
        )
        - win.n_pad_spin.value()
    )
    win.auto_find_peaks()

    assert win.p0_spin0.value() == expected_peaks[1, 0]
    assert win.p0_spin1.value() == expected_peaks[1, 1]
    assert win.p1_spin0.value() == expected_peaks[2, 0]
    assert win.p1_spin1.value() == expected_peaks[2, 1]

    win.order_spin.setValue(2)
    win._update_higher_order_targets()
    expected_higher = erlab.analysis.mesh.higher_order_peaks(
        win.tool_status.first_order_peaks,
        order=2,
        shape=win.main_fft_image.image.shape,
        include_center=False,
        only_upper=False,
    )[2:]

    assert len(win.higher_order_targets) == len(expected_higher)
    for target, expected in zip(win.higher_order_targets, expected_higher, strict=True):
        pos = target.pos()
        assert (pos.y(), pos.x()) == pytest.approx(expected)

    warnings: list[str] = []

    def _fake_warning(parent, title, text):
        warnings.append(text)
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", staticmethod(_fake_warning))
    win.save_mesh()
    assert warnings
    assert "No mesh data" in warnings[-1]

    state_file = tmp_path / "meshtool_state.h5"
    win.to_file(state_file)

    restored = erlab.interactive.utils.ToolWindow.from_file(state_file)
    qtbot.addWidget(restored)
    assert isinstance(restored, MeshTool)

    assert restored.tool_status == win.tool_status
    xr.testing.assert_identical(restored.tool_data, win.tool_data)
