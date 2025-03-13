import tempfile
import weakref

import numpy as np
import pyperclip
import pytest
import xarray as xr
import xarray.testing
from numpy.testing import assert_almost_equal
from qtpy import QtWidgets

import erlab
from erlab.interactive.derivative import DerivativeTool
from erlab.interactive.fermiedge import GoldTool, ResolutionTool
from erlab.interactive.imagetool import ImageTool, itool
from erlab.interactive.imagetool.controls import ItoolColormapControls
from erlab.interactive.imagetool.core import _AssociatedCoordsDialog, _parse_input
from erlab.interactive.imagetool.dialogs import (
    AverageDialog,
    CropDialog,
    CropToViewDialog,
    NormalizeDialog,
    RotationDialog,
    SymmetrizeDialog,
)

_TEST_DATA: dict[str, xr.DataArray] = {
    "2D": xr.DataArray(
        np.arange(25).reshape((5, 5)),
        dims=["alpha", "eV"],
        coords={"alpha": np.arange(5), "eV": np.arange(5)},
    ),
    "3D": xr.DataArray(
        np.arange(125).reshape((5, 5, 5)),
        dims=["alpha", "eV", "beta"],
        coords={"alpha": np.arange(5), "eV": np.arange(5), "beta": np.arange(5)},
    ),
    "3D_nonuniform": xr.DataArray(
        np.arange(125).reshape((5, 5, 5)),
        dims=["alpha", "eV", "beta"],
        coords={
            "alpha": np.array([0.1, 0.4, 0.5, 0.55, 0.8]),
            "eV": np.arange(5),
            "beta": np.arange(5),
        },
    ),
    "3D_const_nonuniform": xr.DataArray(
        np.arange(125).reshape((5, 5, 5)),
        dims=["x", "eV", "beta"],
        coords={
            "x": np.array([0.1, 0.1, 0.1, 0.1, 0.1]),
            "eV": np.arange(5),
            "beta": np.arange(5),
        },
    ),
}


@pytest.mark.parametrize("val_dtype", [np.float32, np.float64, np.int32, np.int64])
@pytest.mark.parametrize("coord_dtype", [np.float32, np.float64, np.int32, np.int64])
def test_itool_dtypes(qtbot, move_and_compare_values, val_dtype, coord_dtype) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(val_dtype),
        dims=["x", "y"],
        coords={
            "x": np.arange(5, dtype=coord_dtype),
            "y": np.array([1, 3, 2, 7, 8], dtype=coord_dtype),  # non-uniform
        },
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    move_and_compare_values(qtbot, win, [12.0, 7.0, 6.0, 11.0])
    win.close()


def test_itool_load(qtbot, move_and_compare_values, accept_dialog) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)),
        dims=["x", "y"],
        coords={"x": np.arange(5), "y": np.arange(5)},
    )

    win = itool(np.zeros((2, 2)), execute=False)
    qtbot.addWidget(win)

    tmp_dir = tempfile.TemporaryDirectory()
    filename = f"{tmp_dir.name}/data.h5"
    data.to_netcdf(filename, engine="h5netcdf")

    def _go_to_file(dialog: QtWidgets.QFileDialog):
        dialog.setDirectory(tmp_dir.name)
        dialog.selectFile(filename)
        focused = dialog.focusWidget()
        if isinstance(focused, QtWidgets.QLineEdit):
            focused.setText("data.h5")

    _handler = accept_dialog(lambda: win._open_file(native=False), pre_call=_go_to_file)
    move_and_compare_values(qtbot, win, [12.0, 7.0, 6.0, 11.0])

    win.close()

    tmp_dir.cleanup()


def test_itool_save(qtbot, accept_dialog) -> None:
    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    tmp_dir = tempfile.TemporaryDirectory()
    filename = f"{tmp_dir.name}/data.h5"

    def _go_to_file(dialog: QtWidgets.QFileDialog):
        dialog.setDirectory(tmp_dir.name)
        dialog.selectFile(filename)
        focused = dialog.focusWidget()
        if isinstance(focused, QtWidgets.QLineEdit):
            focused.setText("data.h5")

    _handler = accept_dialog(
        lambda: win._export_file(native=False), pre_call=_go_to_file
    )

    win.close()

    xr.testing.assert_equal(data, xr.load_dataarray(filename, engine="h5netcdf"))
    tmp_dir.cleanup()


def test_itool_general(qtbot, move_and_compare_values) -> None:
    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])
    win = itool(data, execute=False, cmap="terrain_r")
    qtbot.addWidget(win)

    # Copy cursor values
    win.mnb._copy_cursor_val()
    assert pyperclip.paste() == "[[2, 2]]"
    win.mnb._copy_cursor_idx()
    assert pyperclip.paste() == "[[2, 2]]"

    move_and_compare_values(qtbot, win, [12.0, 7.0, 6.0, 11.0])

    # Snap
    win.array_slicer.snap_act.setChecked(True)
    assert win.array_slicer.snap_to_data

    # Transpose
    win.slicer_area.transpose_act.trigger()
    assert win.slicer_area.data.dims == ("y", "x")
    move_and_compare_values(qtbot, win, [12.0, 11.0, 6.0, 7.0])

    # Set bin
    win.array_slicer.set_bin(0, 0, 2, update=False)
    win.array_slicer.set_bin(0, 1, 2, update=True)
    move_and_compare_values(qtbot, win, [9.0, 8.0, 3.0, 4.0])

    # Test code generation
    assert win.array_slicer.qsel_code(0, (0,)) == ".qsel(x=1.5, x_width=2.0)"

    # Set colormap and gamma
    win.slicer_area.set_colormap(
        "BuWh", gamma=1.5, reverse=True, high_contrast=True, zero_centered=True
    )

    # Lock levels
    win.slicer_area.lock_levels(True)
    # qtbot.wait_until(lambda: win.slicer_area.levels_locked, timeout=1000)
    win.slicer_area.levels = (1.0, 23.0)
    assert win.slicer_area._colorbar.cb._copy_limits() == str((1.0, 23.0))

    # Test color limits editor
    clw = win.slicer_area._colorbar.cb._clim_menu.actions()[0].defaultWidget()
    assert clw.min_spin.value() == win.slicer_area.levels[0]
    assert clw.max_spin.value() == win.slicer_area.levels[1]
    clw.min_spin.setValue(1.0)
    assert clw.min_spin.value() == 1.0
    clw.max_spin.setValue(2.0)
    assert clw.max_spin.value() == 2.0
    clw.rst_btn.click()
    assert win.slicer_area.levels == (0.0, 24.0)
    clw.center_zero()
    win.slicer_area.levels = (1.0, 23.0)
    win.slicer_area.lock_levels(False)

    # Undo and redo
    win.slicer_area.undo()
    win.slicer_area.redo()

    # Check restoring the state works
    old_state = dict(win.slicer_area.state)
    win.slicer_area.state = old_state

    # Add and remove cursor
    win.slicer_area.add_cursor()
    expected_state = {
        "color": {
            "cmap": "BuWh",
            "gamma": 1.5,
            "reverse": True,
            "high_contrast": True,
            "zero_centered": True,
            "levels_locked": False,
        },
        "slice": {
            "dims": ("y", "x"),
            "bins": [[2, 2], [2, 2]],
            "indices": [[2, 2], [2, 2]],
            "values": [[2, 2], [2, 2]],
            "snap_to_data": True,
            "twin_coord_names": (),
        },
        "current_cursor": 1,
        "manual_limits": {},
        "splitter_sizes": list(old_state["splitter_sizes"]),
        "file_path": None,
        "cursor_colors": ["#cccccc", "#ffff00"],
        "plotitem_states": [
            {"vb_aspect_locked": False, "vb_x_inverted": False, "vb_y_inverted": False},
            {"vb_aspect_locked": False, "vb_x_inverted": False, "vb_y_inverted": False},
            {"vb_aspect_locked": False, "vb_x_inverted": False, "vb_y_inverted": False},
        ],
    }
    assert win.slicer_area.state == expected_state
    win.slicer_area.remove_current_cursor()
    assert win.slicer_area.state == old_state

    # See if restoring the state works for the second cursor
    win.slicer_area.state = expected_state
    assert win.slicer_area.state == expected_state

    # Setting data
    win.slicer_area.set_data(data.rename("new_data"))
    assert win.windowTitle() == "new_data"

    # Colormap combobox
    cmap_ctrl = win.docks[1].widget().layout().itemAt(0).widget()
    assert isinstance(cmap_ctrl, ItoolColormapControls)
    cmap_ctrl.cb_colormap.load_all()
    cmap_ctrl.cb_colormap.showPopup()

    win.close()


@pytest.mark.parametrize(
    "test_data_type", ["2D", "3D", "3D_nonuniform", "3D_const_nonuniform"]
)
@pytest.mark.parametrize("condition", ["unbinned", "binned"])
def test_itool_tools(qtbot, test_data_type, condition) -> None:
    data = _TEST_DATA[test_data_type].copy()
    win = ImageTool(data)
    qtbot.addWidget(win)

    main_image = win.slicer_area.images[0]

    # Test code generation
    if data.ndim == 2:
        assert main_image.selection_code == ""
    else:
        assert main_image.selection_code == ".qsel(beta=2.0)"

    if condition == "binned":
        win.array_slicer.set_bin(0, axis=0, value=3, update=False)
        win.array_slicer.set_bin(0, axis=1, value=2, update=True)
        if data.ndim == 3:
            win.array_slicer.set_bin(0, axis=2, value=3, update=True)

    # Open goldtool from main image
    if not test_data_type.endswith("nonuniform"):
        main_image.open_in_goldtool()
        assert isinstance(
            next(iter(win.slicer_area._associated_tools.values())), GoldTool
        )

        # Close associated windows
        win.slicer_area.close_associated_windows()
        qtbot.wait_until(
            lambda w=win: len(w.slicer_area._associated_tools) == 0, timeout=1000
        )

        main_image.open_in_restool()
        assert isinstance(
            next(iter(win.slicer_area._associated_tools.values())), ResolutionTool
        )

        # Close associated windows
        win.slicer_area.close_associated_windows()
        qtbot.wait_until(
            lambda w=win: len(w.slicer_area._associated_tools) == 0, timeout=1000
        )

        # Open dtool from main image
        main_image.open_in_dtool()
        assert isinstance(
            next(iter(win.slicer_area._associated_tools.values())), DerivativeTool
        )

    # Open main image in new window
    main_image.open_in_new_window()
    assert isinstance(list(win.slicer_area._associated_tools.values())[-1], ImageTool)

    win.slicer_area.close_associated_windows()

    win.close()


def test_itool_load_compat(qtbot) -> None:
    original = xr.DataArray(
        np.arange(25).reshape((5, 5)),
        dims=["x", "y"],
        coords={"x": np.arange(5), "y": np.arange(5)},
    )

    win = itool(original.expand_dims(z=2, axis=-1).T, execute=False)
    qtbot.addWidget(win)

    win.slicer_area.add_cursor()
    win.slicer_area.add_cursor()

    # Check if setting compatible data does not change cursor count
    win.slicer_area.set_data(original.expand_dims(z=5, axis=-1))

    assert win.slicer_area.n_cursors == 3

    win.close()


def test_parse_input() -> None:
    # If no 2D to 4D data is present in given Dataset, ValueError is raised
    with pytest.raises(
        ValueError, match="No valid data for ImageTool found in Dataset"
    ):
        _parse_input(
            xr.Dataset({"data1d": xr.DataArray(np.arange(5), dims=["x"]), "data0d": 1})
        )


def test_itool_ds(qtbot) -> None:
    data = xr.Dataset(
        {
            "data1d": xr.DataArray(np.arange(5), dims=["x"]),
            "a": xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"]),
            "b": xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"]),
        }
    )
    wins = itool(data, execute=False, link=True)
    assert isinstance(wins, list)
    assert len(wins) == 2

    qtbot.addWidget(wins[0])
    qtbot.addWidget(wins[1])

    with qtbot.waitExposed(wins[0]):
        wins[0].show()
    with qtbot.waitExposed(wins[1]):
        wins[1].show()

    assert wins[0].windowTitle() == "a"
    assert wins[1].windowTitle() == "b"

    # Check if properly linked
    assert wins[0].slicer_area._linking_proxy == wins[1].slicer_area._linking_proxy
    assert wins[0].slicer_area.linked_slicers == weakref.WeakSet([wins[1].slicer_area])

    wins[0].slicer_area.unlink()
    wins[1].slicer_area.unlink()
    wins[0].close()
    wins[1].close()


def test_itool_multidimensional(qtbot, move_and_compare_values) -> None:
    win = ImageTool(xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"]))
    qtbot.addWidget(win)

    win.slicer_area.set_data(
        xr.DataArray(np.arange(125).reshape((5, 5, 5)), dims=["x", "y", "z"])
    )
    move_and_compare_values(qtbot, win, [62.0, 37.0, 32.0, 57.0])

    win.slicer_area.set_data(
        xr.DataArray(np.arange(625).reshape((5, 5, 5, 5)), dims=["x", "y", "z", "t"])
    )
    move_and_compare_values(qtbot, win, [312.0, 187.0, 162.0, 287.0])
    # Test aspect ratio lock
    for img in win.slicer_area.images:
        img.toggle_aspect_equal()
    for img in win.slicer_area.images:
        img.toggle_aspect_equal()

    win.close()


def test_value_update(qtbot) -> None:
    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])
    new_vals = -data.values.astype(np.float64)

    win = ImageTool(data)
    qtbot.addWidget(win)
    with qtbot.waitExposed(win):
        win.show()
        win.activateWindow()

    win.slicer_area.update_values(new_vals)
    assert_almost_equal(win.array_slicer.point_value(0), -12.0)
    win.close()


def test_value_update_errors(qtbot) -> None:
    win = ImageTool(xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"]))
    qtbot.addWidget(win)
    with qtbot.waitExposed(win):
        win.show()
        win.activateWindow()

    with pytest.raises(ValueError, match="DataArray dimensions do not match"):
        win.slicer_area.update_values(
            xr.DataArray(np.arange(24).reshape((2, 2, 6)), dims=["x", "y", "z"])
        )
    with pytest.raises(ValueError, match="DataArray dimensions do not match"):
        win.slicer_area.update_values(
            xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "z"])
        )
    with pytest.raises(ValueError, match="DataArray shape does not match"):
        win.slicer_area.update_values(
            xr.DataArray(np.arange(24).reshape((4, 6)), dims=["x", "y"])
        )
    with pytest.raises(ValueError, match="^Data shape does not match.*"):
        win.slicer_area.update_values(np.arange(24).reshape((4, 6)))

    win.close()


def test_itool_rotate(qtbot, accept_dialog) -> None:
    data = xr.DataArray(np.arange(25).reshape((5, 5)).astype(float), dims=["x", "y"])
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    # Test dialog
    def _set_dialog_params(dialog: RotationDialog) -> None:
        dialog.angle_spin.setValue(60.0)
        dialog.reshape_check.setChecked(True)
        dialog.new_window_check.setChecked(False)

    _handler = accept_dialog(win.mnb._rotate, pre_call=_set_dialog_params)

    # Check if the data is rotated
    xarray.testing.assert_allclose(
        win.slicer_area._data,
        erlab.analysis.transform.rotate(data, angle=60.0, reshape=True),
    )

    # Test guidelines
    win.slicer_area.set_data(data)
    win.slicer_area.main_image.set_guidelines(3)
    assert win.slicer_area.main_image.is_guidelines_visible

    win.slicer_area.main_image._guidelines_items[0].setAngle(90.0 - 30.0)
    win.slicer_area.main_image._guidelines_items[-1].setPos((3.0, 3.1))

    def _set_dialog_params(dialog: RotationDialog) -> None:
        assert dialog.angle_spin.value() == 30.0
        assert dialog.center_spins[0].value() == 3.0
        assert dialog.center_spins[1].value() == 3.1
        dialog.copy_button.click()
        qtbot.wait_signal(dialog._sigCodeCopied)
        dialog.reshape_check.setChecked(True)
        dialog.new_window_check.setChecked(False)

    _handler = accept_dialog(win.mnb._rotate, pre_call=_set_dialog_params)

    # Check if the data is rotated
    xarray.testing.assert_allclose(
        win.slicer_area._data,
        erlab.analysis.transform.rotate(
            data, angle=30.0, center=(3.0, 3.1), reshape=True
        ),
    )

    # Test copy button
    assert pyperclip.paste().startswith("era.transform.rotate")

    # Transpose should remove guidelines
    win.slicer_area.swap_axes(0, 1)
    qtbot.wait_until(
        lambda: not win.slicer_area.main_image.is_guidelines_visible, timeout=1000
    )

    win.close()


def test_itool_crop_view(qtbot, accept_dialog) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["x", "y"],
        coords={"x": np.arange(5), "y": np.arange(5)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    # Change limits
    win.slicer_area.main_image.getViewBox().setRange(xRange=[1, 4], yRange=[0, 3])
    # Trigger manual range propagation
    win.slicer_area.main_image.getViewBox().sigRangeChangedManually.emit(
        win.slicer_area.main_image.getViewBox().state["mouseEnabled"][:]
    )

    # Test 2D crop
    def _set_dialog_params(dialog: CropToViewDialog) -> None:
        dialog.dim_checks["x"].setChecked(True)
        dialog.dim_checks["y"].setChecked(True)
        dialog.copy_button.click()
        qtbot.wait_signal(dialog._sigCodeCopied)
        dialog.new_window_check.setChecked(False)

    _handler = accept_dialog(win.mnb._crop_to_view, pre_call=_set_dialog_params)
    xarray.testing.assert_allclose(
        win.slicer_area._data, data.sel(x=slice(1.0, 4.0), y=slice(0.0, 3.0))
    )
    assert pyperclip.paste() == ".sel(x=slice(1.0, 4.0), y=slice(0.0, 3.0))"

    win.close()


def test_itool_crop(qtbot, accept_dialog) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["x", "y"],
        coords={"x": np.arange(5), "y": np.arange(5)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    win.slicer_area.add_cursor()
    win.slicer_area.add_cursor()

    # Move cursors to define 2D crop region
    win.slicer_area.set_value(axis=0, value=1.0, cursor=0)
    win.slicer_area.set_value(axis=1, value=0.0, cursor=0)
    win.slicer_area.set_value(axis=0, value=3.0, cursor=1)
    win.slicer_area.set_value(axis=1, value=2.0, cursor=1)
    win.slicer_area.set_value(axis=0, value=4.0, cursor=2)
    win.slicer_area.set_value(axis=1, value=3.0, cursor=2)

    # Test 1D plot normalization
    for profile_axis in win.slicer_area.profiles:
        profile_axis.set_normalize(True)
        for data_item in profile_axis.slicer_data_items:
            yvals = (
                data_item.getData()[0]
                if data_item.is_vertical
                else data_item.getData()[1]
            )
            assert_almost_equal(np.nanmean(yvals), 1.0)
        profile_axis.set_normalize(False)

    # Test 2D crop
    def _set_dialog_params(dialog: CropDialog) -> None:
        # activate combo to increase ExclusiveComboGroup coverage
        dialog.cursor_combos[0].activated.emit(0)
        dialog.cursor_combos[0].setCurrentIndex(0)
        dialog.cursor_combos[0].activated.emit(2)
        dialog.cursor_combos[1].setCurrentIndex(2)
        dialog.dim_checks["x"].setChecked(True)
        dialog.dim_checks["y"].setChecked(True)
        dialog.copy_button.click()
        qtbot.wait_signal(dialog._sigCodeCopied)
        dialog.new_window_check.setChecked(False)

    _h0 = accept_dialog(win.mnb._crop, pre_call=_set_dialog_params)
    xarray.testing.assert_allclose(
        win.slicer_area._data, data.sel(x=slice(1.0, 4.0), y=slice(0.0, 3.0))
    )
    assert pyperclip.paste() == ".sel(x=slice(1.0, 4.0), y=slice(0.0, 3.0))"

    # 1D crop
    win.slicer_area.set_value(axis=0, value=4.0, cursor=1)
    win.slicer_area.set_value(axis=1, value=3.0, cursor=1)

    def _set_dialog_params(dialog: CropDialog) -> None:
        dialog.cursor_combos[0].activated.emit(1)
        dialog.cursor_combos[0].setCurrentIndex(1)
        dialog.cursor_combos[0].activated.emit(2)
        dialog.cursor_combos[1].setCurrentIndex(2)
        dialog.dim_checks["x"].setChecked(True)
        dialog.dim_checks["y"].setChecked(False)
        dialog.copy_button.click()
        qtbot.wait_signal(dialog._sigCodeCopied)
        dialog.new_window_check.setChecked(False)

    _h1 = accept_dialog(win.mnb._crop, pre_call=_set_dialog_params)
    xarray.testing.assert_allclose(
        win.slicer_area._data, data.sel(x=slice(2.0, 4.0), y=slice(0.0, 3.0))
    )
    assert pyperclip.paste() == ".sel(x=slice(2.0, 4.0))"

    win.close()


def test_itool_average(qtbot, accept_dialog) -> None:
    data = xr.DataArray(
        np.arange(60).reshape((3, 4, 5)).astype(float),
        dims=["x", "y", "z"],
        coords={
            "x": np.arange(3),
            "y": np.arange(4),
            "z": np.arange(5),
            "t": ("x", np.arange(3)),
        },
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    # Test dialog
    def _set_dialog_params(dialog: AverageDialog) -> None:
        dialog.dim_checks["x"].setChecked(True)
        dialog.copy_button.click()
        qtbot.wait_signal(dialog._sigCodeCopied)
        dialog.new_window_check.setChecked(False)

    _handler = accept_dialog(win.mnb._average, pre_call=_set_dialog_params)
    xarray.testing.assert_identical(win.slicer_area._data, data.qsel.average("x"))

    assert pyperclip.paste() == '.qsel.average("x")'
    win.close()


def test_itool_symmetrize(qtbot, accept_dialog) -> None:
    data = xr.DataArray(
        np.arange(60).reshape((3, 4, 5)).astype(float),
        dims=["x", "y", "z"],
        coords={
            "x": np.arange(3),
            "y": np.arange(4),
            "z": np.arange(5),
            "t": ("x", np.arange(3)),
        },
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    # Test dialog
    def _set_dialog_params(dialog: SymmetrizeDialog) -> None:
        dialog._dim_combo.setCurrentIndex(0)
        dialog.copy_button.click()
        qtbot.wait_signal(dialog._sigCodeCopied)
        dialog.new_window_check.setChecked(False)

    _handler = accept_dialog(win.mnb._symmetrize, pre_call=_set_dialog_params)
    xarray.testing.assert_identical(
        win.slicer_area._data, erlab.analysis.transform.symmetrize(data, "x", center=1)
    )

    assert pyperclip.paste() == 'era.transform.symmetrize(, dim="x", center=1.0)'
    win.close()


def test_itool_assoc_coords(qtbot, accept_dialog) -> None:
    data = data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["x", "y"],
        coords={
            "x": np.arange(5),
            "y": np.arange(5),
            "z": ("x", [1, 3, 2, 4, 5]),
            "u": ("x", np.arange(5)),
            "t": ("y", np.arange(5)),
            "v": ("y", np.arange(5)),
        },
    )
    win = itool(data, execute=False, cmap="terrain_r")
    qtbot.addWidget(win)

    def _set_dialog_params(dialog: _AssociatedCoordsDialog) -> None:
        for check in dialog._checks.values():
            check.setChecked(True)

    _handler = accept_dialog(
        win.slicer_area._choose_associated_coords, pre_call=_set_dialog_params
    )

    # Change limits
    win.slicer_area.main_image.getViewBox().setRange(xRange=[1, 4], yRange=[0, 3])
    # Trigger manual range propagation
    win.slicer_area.main_image.getViewBox().sigRangeChangedManually.emit(
        win.slicer_area.main_image.getViewBox().state["mouseEnabled"][:]
    )

    win.slicer_area.transpose_act.trigger()

    win.close()


def normalize(data, norm_dims, option):
    area = data.mean(norm_dims)
    minimum = data.min(norm_dims)
    maximum = data.max(norm_dims)

    match option:
        case 0:
            return data / area
        case 1:
            return (data - minimum) / (maximum - minimum)
        case 2:
            return data - minimum
        case _:
            return (data - minimum) / area


@pytest.mark.parametrize("option", [0, 1, 2, 3])
def test_itool_normalize(qtbot, accept_dialog, option) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["x", "y"],
        coords={"x": np.arange(5), "y": np.arange(5)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    # Test dialog
    def _set_dialog_params(dialog: NormalizeDialog) -> None:
        dialog.dim_checks["x"].setChecked(True)
        dialog.opts[option].setChecked(True)

        # Preview
        dialog._preview()

    _handler = accept_dialog(win.mnb._normalize, pre_call=_set_dialog_params)

    # Check if the data is normalized
    xarray.testing.assert_identical(
        win.slicer_area.data, normalize(data, ("x",), option)
    )

    # Reset normalization
    win.mnb._reset_filters()
    xarray.testing.assert_identical(win.slicer_area.data, data)

    # Check if canceling the dialog does not change the data
    _handler = accept_dialog(
        win.mnb._normalize,
        pre_call=_set_dialog_params,
        accept_call=lambda d: d.reject(),
    )
    xarray.testing.assert_identical(win.slicer_area.data, data)

    win.close()
