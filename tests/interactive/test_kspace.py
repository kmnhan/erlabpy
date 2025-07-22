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
            ValueError, match="Data is not compatible with the interactive tool."
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
    xr.testing.assert_identical(win._itool.slicer_area.data, anglemap_kconv)
    win._itool.close()
    win.close()
