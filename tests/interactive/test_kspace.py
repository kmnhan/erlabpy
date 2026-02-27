import tempfile

import numpy as np
import pytest
import xarray as xr

import erlab
from erlab.interactive.kspace import KspaceTool, ktool


def test_ktool_compatible(anglemap) -> None:
    cut = anglemap.qsel(beta=-8.3)
    data_4d = anglemap.expand_dims("x", 2)
    data_3d_without_alpha = data_4d.qsel(alpha=-8.3)

    for data in (cut, data_4d, data_3d_without_alpha):
        with pytest.raises(
            ValueError, match=r"Data is not compatible with the interactive tool."
        ):
            data.kspace.interactive()


@pytest.mark.parametrize("wf", ["wf_auto", "wf_manual"])
@pytest.mark.parametrize("kind", ["map", "const_energy"])
@pytest.mark.parametrize("assignment", ["before", "after"])
def test_ktool(qtbot, anglemap, wf, kind, assignment) -> None:
    offset_dict = {"delta": 30.0, "xi": 20.0, "beta": 10.0}

    anglemap = anglemap.copy()

    def assign_attrs():
        anglemap.kspace.offsets = offset_dict
        if wf != "wf_auto":
            anglemap.kspace.work_function = 4.0

    if kind != "map":
        anglemap = anglemap.qsel(eV=-0.1)

    if assignment == "before":
        assign_attrs()

    win = ktool(
        anglemap,
        avec=erlab.lattice.abc2avec(6.97, 6.97, 8.685, 90, 90, 120),
        rotate_bz=30.0,
        cmap="terrain_r",
        execute=False,
    )
    qtbot.addWidget(win)
    win.centering_combo.setCurrentText("I")
    if not win.data.kspace._has_hv:
        win.kz_spin.setValue(0.5)

    for k, v in offset_dict.items():
        if assignment == "before":
            assert np.isclose(win._offset_spins[k].value(), v)
        else:
            win._offset_spins[k].setValue(v)

    if wf != "wf_auto":
        if assignment == "before":
            assert np.isclose(win._offset_spins["wf"].value(), 4.0)
        else:
            win._offset_spins["wf"].setValue(4.0)

    if assignment != "before":
        assign_attrs()
    anglemap_kconv = anglemap.kspace.convert()

    def _check_code_kconv(w: KspaceTool):
        namespace = {"anglemap": anglemap}
        exec(w.copy_code(), {"__builtins__": {}}, namespace)  # noqa: S102
        xr.testing.assert_identical(anglemap_kconv, namespace["anglemap_kconv"])

    _check_code_kconv(win)

    # Test ROI
    win.add_circle_btn.click()
    roi = win._roi_list[0]
    roi.getMenu()
    roi.set_position((0.1, 0.1), 0.2)
    assert roi.get_position() == (0.1, 0.1, 0.2)

    roi_control_widget = roi._pos_menu.actions()[0].defaultWidget()
    roi_control_widget.x_spin.setValue(0.0)
    roi_control_widget.y_spin.setValue(0.2)
    roi_control_widget.r_spin.setValue(0.3)
    assert roi.get_position() == (0.0, 0.2, 0.3)

    # Show imagetool
    win.show_converted()
    win._itool.show()
    xr.testing.assert_identical(win._itool.slicer_area.data, anglemap_kconv)
    win._itool.close()

    # Test save & restore
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        filename = f"{tmp_dir_name}/tool_save.h5"
        win.to_file(filename)

        win_restored = erlab.interactive.utils.ToolWindow.from_file(filename)
        qtbot.addWidget(win_restored)
        assert isinstance(win_restored, KspaceTool)

        assert win.tool_status == win_restored.tool_status
        assert str(win_restored.info_text) == str(win.info_text)


def test_ktool_update_rate_limited(qtbot, anglemap, monkeypatch) -> None:
    win = ktool(anglemap, execute=False)
    qtbot.addWidget(win)

    # Allow any startup-triggered delayed update to finish before counting.
    wait_ms = int(1000 / win._UPDATE_LIMIT_HZ) + 50
    qtbot.wait(wait_ms)

    call_count = 0
    original_get_data = win.get_data

    def _counting_get_data():
        nonlocal call_count
        call_count += 1
        return original_get_data()

    monkeypatch.setattr(win, "get_data", _counting_get_data)

    spin = win._offset_spins["delta"]
    base = spin.value()
    spin.setValue(base + 0.01)
    spin.setValue(base + 0.02)
    spin.setValue(base + 0.03)

    qtbot.wait_until(lambda: call_count == 1, timeout=2000)
    qtbot.wait(wait_ms)
    assert call_count == 1

    win.update()
    assert call_count == 2


def test_ktool_kinetic_energy_axis_preview(qtbot, anglemap) -> None:
    data = anglemap.copy().assign_coords(hv=6.2)
    data = data.assign_coords(eV=data.hv - data.kspace.work_function + data.eV)

    win = ktool(data, execute=False)
    qtbot.addWidget(win)

    min0, max0 = win.center_spin.minimum(), win.center_spin.maximum()
    win._offset_spins["wf"].setValue(win._offset_spins["wf"].value() - 0.2)
    assert np.isclose(win.center_spin.minimum(), min0 - 0.2)
    assert np.isclose(win.center_spin.maximum(), max0 - 0.2)

    win.width_spin.setValue(5)
    energy_axis = data.eV.values - float(data.hv.values) + win._work_function
    center = float(energy_axis[len(energy_axis) // 2])
    win.center_spin.setValue(center)
    center = win.center_spin.value()

    ang, kpreview = win.get_data()

    idx = np.searchsorted((energy_axis[:-1] + energy_axis[1:]) / 2, center)
    start = max(0, idx - win.width_spin.value() // 2)
    stop = min(
        energy_axis.size,
        idx + (win.width_spin.value() - 1) // 2 + 1,
    )

    expected_ang = (
        data.copy()
        .assign_coords(eV=energy_axis)
        .isel(eV=slice(start, stop))
        .mean("eV", skipna=True, keep_attrs=True)
        .assign_coords(eV=center)
    )
    expected_ang = win._assign_params(expected_ang)
    expected_k = expected_ang.kspace.convert(
        bounds=win.bounds, resolution=win.resolution, silent=True
    )

    xr.testing.assert_allclose(ang, expected_ang)
    xr.testing.assert_allclose(kpreview, expected_k)
