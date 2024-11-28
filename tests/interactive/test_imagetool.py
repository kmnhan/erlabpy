import tempfile
import time
import weakref

import numpy as np
import pyperclip
import pytest
import xarray as xr
import xarray.testing
from numpy.testing import assert_almost_equal
from qtpy import QtCore, QtWidgets

import erlab.analysis.transform
from erlab.interactive.derivative import DerivativeTool
from erlab.interactive.fermiedge import GoldTool
from erlab.interactive.imagetool import ImageTool, _parse_input, itool
from erlab.interactive.imagetool.dialogs import (
    CropDialog,
    NormalizeDialog,
    RotationDialog,
)


@pytest.mark.parametrize("val_dtype", [np.float32, np.float64, np.int32, np.int64])
@pytest.mark.parametrize("coord_dtype", [np.float32, np.float64, np.int32, np.int64])
def test_itool_dtypes(qtbot, move_and_compare_values, val_dtype, coord_dtype):
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(val_dtype),
        dims=["x", "y"],
        coords={
            "x": np.arange(5, dtype=coord_dtype),
            "y": np.arange(5, dtype=coord_dtype),
        },
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    with qtbot.waitExposed(win):
        win.show()
        win.activateWindow()
    time.sleep(0.5)

    move_and_compare_values(qtbot, win, [12.0, 7.0, 6.0, 11.0])
    win.close()
    del win


def test_itool_load(qtbot, move_and_compare_values, accept_dialog):
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)),
        dims=["x", "y"],
        coords={"x": np.arange(5), "y": np.arange(5)},
    )

    win = itool(np.zeros((2, 2)), execute=False)
    qtbot.addWidget(win)
    with qtbot.waitExposed(win):
        win.show()
        win.activateWindow()

    tmp_dir = tempfile.TemporaryDirectory()
    filename = f"{tmp_dir.name}/data.h5"
    data.to_netcdf(filename, engine="h5netcdf")

    def _go_to_file(dialog: QtWidgets.QFileDialog):
        dialog.setDirectory(tmp_dir.name)
        dialog.selectFile(filename)
        focused = dialog.focusWidget()
        if isinstance(focused, QtWidgets.QLineEdit):
            focused.setText("data.h5")

    accept_dialog(lambda: win._open_file(native=False), pre_call=_go_to_file)
    move_and_compare_values(qtbot, win, [12.0, 7.0, 6.0, 11.0])

    win.close()

    tmp_dir.cleanup()


def test_itool_save(qtbot, accept_dialog):
    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])
    win = itool(data, execute=False)

    qtbot.addWidget(win)
    with qtbot.waitExposed(win):
        win.show()
        win.activateWindow()

    tmp_dir = tempfile.TemporaryDirectory()
    filename = f"{tmp_dir.name}/data.h5"

    def _go_to_file(dialog: QtWidgets.QFileDialog):
        dialog.setDirectory(tmp_dir.name)
        dialog.selectFile(filename)
        focused = dialog.focusWidget()
        if isinstance(focused, QtWidgets.QLineEdit):
            focused.setText("data.h5")

    accept_dialog(lambda: win._export_file(native=False), pre_call=_go_to_file)

    win.close()

    xr.testing.assert_equal(data, xr.load_dataarray(filename, engine="h5netcdf"))
    tmp_dir.cleanup()


def test_itool(qtbot, move_and_compare_values):
    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])
    win = itool(data, execute=False, cmap="terrain_r")
    qtbot.addWidget(win)

    with qtbot.waitExposed(win):
        win.show()
        win.activateWindow()

    # Copy cursor values
    win.mnb._copy_cursor_val()
    assert pyperclip.paste() == "[[2, 2]]"
    win.mnb._copy_cursor_idx()
    assert pyperclip.paste() == "[[2, 2]]"

    move_and_compare_values(qtbot, win, [12.0, 7.0, 6.0, 11.0])

    # Snap
    qtbot.keyClick(win, QtCore.Qt.Key.Key_S)
    assert win.array_slicer.snap_to_data

    # Transpose
    qtbot.keyClick(win, QtCore.Qt.Key.Key_T)
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
        "ColdWarm", gamma=1.5, reverse=True, high_contrast=True, zero_centered=True
    )

    # Lock levels
    win.slicer_area.lock_levels(True)
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
    clw.zero_btn.click()
    assert win.slicer_area.levels == (0.0, 24.0)
    win.slicer_area.levels = (1.0, 23.0)
    win.slicer_area.lock_levels(False)

    # Undo and redo
    win.slicer_area.undo()
    qtbot.keyClick(win, QtCore.Qt.Key.Key_Z, QtCore.Qt.KeyboardModifier.ControlModifier)
    qtbot.keyClick(
        win,
        QtCore.Qt.Key.Key_Z,
        QtCore.Qt.KeyboardModifier.ControlModifier
        | QtCore.Qt.KeyboardModifier.ShiftModifier,
    )
    win.slicer_area.redo()

    # Check restoring the state works
    old_state = dict(win.slicer_area.state)
    win.slicer_area.state = old_state

    # Add and remove cursor
    win.slicer_area.add_cursor()
    expected_state = {
        "color": {
            "cmap": "ColdWarm",
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
            "values": [[2.0, 2.0], [2.0, 2.0]],
            "snap_to_data": True,
        },
        "current_cursor": 1,
        "manual_limits": {"x": [-0.5, 4.5], "y": [-0.5, 4.5]},
        "splitter_sizes": list(old_state["splitter_sizes"]),
        "cursor_colors": ["#cccccc", "#ffff00"],
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

    win.close()

    del win


def test_itool_tools(qtbot, gold):
    win = itool(gold, execute=False)
    qtbot.addWidget(win)

    with qtbot.waitExposed(win):
        win.show()
        win.activateWindow()

    main_image = win.slicer_area.images[0]
    # Test code generation
    assert main_image.selection_code == ""

    # Open goldtool from main image
    main_image.open_in_goldtool()

    assert isinstance(next(iter(win.slicer_area._associated_tools.values())), GoldTool)

    # Close associated windows
    win.slicer_area.close_associated_windows()
    qtbot.waitUntil(
        lambda w=win: len(w.slicer_area._associated_tools) == 0, timeout=1000
    )

    # Open dtool from main image
    main_image.open_in_dtool()
    assert isinstance(
        next(iter(win.slicer_area._associated_tools.values())), DerivativeTool
    )

    # Open main image in new window
    main_image.open_in_new_window()
    assert isinstance(list(win.slicer_area._associated_tools.values())[1], ImageTool)

    win.slicer_area.close_associated_windows()

    win.close()
    del win


def test_parse_input():
    # If no 2D to 4D data is present in given Dataset, ValueError is raised
    with pytest.raises(
        ValueError, match="No valid data for ImageTool found in the Dataset"
    ):
        _parse_input(
            xr.Dataset(
                {
                    "data1d": xr.DataArray(np.arange(5), dims=["x"]),
                    "data0d": 1,
                }
            )
        )


def test_itool_ds(qtbot):
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

    del wins


def test_itool_multidimensional(qtbot, move_and_compare_values):
    win = itool(
        xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"]), execute=False
    )
    qtbot.addWidget(win)

    with qtbot.waitExposed(win):
        win.show()
        win.activateWindow()

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


def test_value_update(qtbot):
    win = itool(
        xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"]), execute=False
    )
    qtbot.addWidget(win)

    with qtbot.waitExposed(win):
        win.show()
        win.activateWindow()

    new_vals = -np.arange(25).reshape((5, 5)).astype(float)
    win.slicer_area.update_values(new_vals)
    assert_almost_equal(win.array_slicer.point_value(0), -12.0)

    win.close()


def test_value_update_errors(qtbot):
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


def test_itool_rotate(qtbot, accept_dialog):
    data = xr.DataArray(np.arange(25).reshape((5, 5)).astype(float), dims=["x", "y"])
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    # Test dialog
    def _set_dialog_params(dialog: RotationDialog) -> None:
        dialog.angle_spin.setValue(60.0)
        dialog.reshape_check.setChecked(True)
        dialog.new_window_check.setChecked(False)

    accept_dialog(win.mnb._rotate, pre_call=_set_dialog_params)

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
        dialog.reshape_check.setChecked(True)
        dialog.new_window_check.setChecked(False)

    accept_dialog(win.mnb._rotate, pre_call=_set_dialog_params)

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
    qtbot.waitUntil(
        lambda: not win.slicer_area.main_image.is_guidelines_visible, timeout=1000
    )

    win.close()


def test_itool_crop(qtbot, accept_dialog):
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["x", "y"],
        coords={"x": np.arange(5), "y": np.arange(5)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    win.slicer_area.add_cursor()
    win.slicer_area.add_cursor()

    # 2D crop
    win.slicer_area.set_value(axis=0, value=1.0, cursor=0)
    win.slicer_area.set_value(axis=1, value=0.0, cursor=0)
    win.slicer_area.set_value(axis=0, value=3.0, cursor=1)
    win.slicer_area.set_value(axis=1, value=2.0, cursor=1)
    win.slicer_area.set_value(axis=0, value=4.0, cursor=2)
    win.slicer_area.set_value(axis=1, value=3.0, cursor=2)

    def _set_dialog_params(dialog: CropDialog) -> None:
        # activate combo to increase ExclusiveComboGroup coverage
        dialog.cursor_combos[0].activated.emit(0)
        dialog.cursor_combos[0].setCurrentIndex(0)
        dialog.cursor_combos[0].activated.emit(2)
        dialog.cursor_combos[1].setCurrentIndex(2)
        dialog.dim_checks["x"].setChecked(True)
        dialog.dim_checks["y"].setChecked(True)
        dialog.copy_button.click()
        dialog.new_window_check.setChecked(False)

    accept_dialog(win.mnb._crop, pre_call=_set_dialog_params)
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
        dialog.new_window_check.setChecked(False)

    accept_dialog(win.mnb._crop, pre_call=_set_dialog_params)
    xarray.testing.assert_allclose(
        win.slicer_area._data, data.sel(x=slice(2.0, 4.0), y=slice(0.0, 3.0))
    )
    assert pyperclip.paste() == ".sel(x=slice(2.0, 4.0))"

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
def test_itool_normalize(qtbot, accept_dialog, option):
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
        dialog.preview_button.click()

    accept_dialog(win.mnb._normalize, pre_call=_set_dialog_params)

    # Check if the data is normalized
    xarray.testing.assert_identical(
        win.slicer_area.data, normalize(data, ("x",), option)
    )

    # Reset normalization
    win.mnb._reset_filters()
    xarray.testing.assert_identical(win.slicer_area.data, data)

    # Check if canceling the dialog does not change the data
    accept_dialog(
        win.mnb._normalize,
        pre_call=_set_dialog_params,
        accept_call=lambda d: d.reject(),
    )
    xarray.testing.assert_identical(win.slicer_area.data, data)

    win.close()
