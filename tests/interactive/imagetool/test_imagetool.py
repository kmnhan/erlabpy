import copy
import json
import logging
import pathlib
import tempfile
import types
import typing
import warnings
import weakref

import numpy as np
import pyperclip
import pyqtgraph as pg
import pytest
import xarray as xr
import xarray.testing
import xarray_lmfit
from numpy.testing import assert_almost_equal
from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive.derivative import DerivativeTool, dtool
from erlab.interactive.fermiedge import GoldTool, ResolutionTool
from erlab.interactive.imagetool import ImageTool, itool
from erlab.interactive.imagetool.controls import (
    ItoolColormapControls,
    ItoolCrosshairControls,
)
from erlab.interactive.imagetool.dialogs import (
    AssignAttrsDialog,
    AssignCoordsDialog,
    AverageDialog,
    CoarsenDialog,
    CropDialog,
    CropToViewDialog,
    DivideByCoordDialog,
    EdgeCorrectionDialog,
    GaussianFilterDialog,
    InterpolationDialog,
    NormalizeDialog,
    RenameDimsCoordsDialog,
    ROIMaskDialog,
    ROIPathDialog,
    RotationDialog,
    SelectionDialog,
    SwapDimsDialog,
    SymmetrizeDialog,
    SymmetrizeNfoldDialog,
    ThinDialog,
)
from erlab.interactive.imagetool.plot_items import _PolyROIEditDialog
from erlab.interactive.imagetool.viewer import (
    ImageSlicerArea,
    _AssociatedCoordsDialog,
    _CursorColorCoordDialog,
    _parse_input,
)

logger = logging.getLogger(__name__)

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


def _press_alt(monkeypatch):
    """Pretend that the Alt/Option key is currently pressed."""
    monkeypatch.setattr(
        QtWidgets.QApplication,
        "queryKeyboardModifiers",
        lambda: QtCore.Qt.KeyboardModifier.AltModifier,
    )


def _exec_generated_code(
    code: str, namespace: dict[str, typing.Any]
) -> dict[str, typing.Any]:
    locals_ns = dict(namespace)
    exec(  # noqa: S102
        code,
        {
            "np": np,
            "xr": xr,
            "erlab": erlab,
            "era": erlab.analysis,
        },
        locals_ns,
    )
    return locals_ns


def _exec_data_fragment(
    data: xr.DataArray,
    code: str,
    *,
    data_name: str = "data",
) -> xr.DataArray:
    statement = (
        f"result = {data_name}{code}" if code.startswith(".") else f"result = {code}"
    )
    namespace = _exec_generated_code(statement, {data_name: data.copy(deep=True)})
    result = namespace["result"]
    assert isinstance(result, xr.DataArray)
    return result


def _menu_action_by_data(menu: QtWidgets.QMenu, data: object) -> QtGui.QAction:
    matches = [action for action in menu.actions() if action.data() == data]
    assert len(matches) == 1
    return matches[0]


def _set_combo_data(combo: QtWidgets.QComboBox, data: object) -> None:
    index = combo.findData(data, QtCore.Qt.ItemDataRole.UserRole)
    assert index != -1
    combo.setCurrentIndex(index)


def _clear_selection_dialog(dialog: SelectionDialog) -> None:
    for row in dialog.rows:
        row.use_check.setChecked(False)
        row.width_check.setChecked(False)


def _assert_guideline_state(
    plot_item,
    *,
    count: int,
    angle: float,
    offset: tuple[float, float],
    follow_cursor: bool = True,
) -> None:
    assert plot_item.is_guidelines_visible
    assert len(plot_item._guidelines_items) == count + 1
    assert plot_item._guideline_angle == angle
    assert tuple(plot_item._guideline_offset) == offset
    assert plot_item._serializable_state["guideline_state"] == {
        "count": count,
        "angle": angle,
        "offset": offset,
        "follow_cursor": follow_cursor,
    }


@pytest.mark.parametrize(
    "test_data_type", ["2D", "3D", "3D_nonuniform", "3D_const_nonuniform"]
)
@pytest.mark.parametrize("condition", ["unbinned", "binned"])
@pytest.mark.parametrize("use_dask", [True, False], ids=["dask", "no_dask"])
def test_itool_tools(qtbot, test_data_type, condition, use_dask) -> None:
    data = _TEST_DATA[test_data_type].copy()
    if use_dask:
        data = data.chunk()

        old_threshold = erlab.interactive.options["io/dask/compute_threshold"]
        # force compute for dask
        erlab.interactive.options["io/dask/compute_threshold"] = 0

    try:
        win = itool(data, execute=False)
        qtbot.addWidget(win)
        win.show()

        main_image = win.slicer_area.images[0]

        logger.info("Test code generation")
        selection_code = main_image.get_selection_code(placeholder="")
        if data.ndim == 2:
            assert not selection_code
        else:
            xarray.testing.assert_identical(
                _exec_data_fragment(win.slicer_area.data, selection_code),
                main_image.current_data,
            )

        if condition == "binned":
            logger.info("Set bins")
            win.array_slicer.set_bin(0, axis=0, value=3, update=False)
            win.array_slicer.set_bin(0, axis=1, value=2, update=data.ndim != 3)
            if data.ndim == 3:
                win.array_slicer.set_bin(0, axis=2, value=3, update=True)

        logger.info("Test alt key menu")
        main_image.vb.menu.popup(QtCore.QPoint(0, 0))
        main_image.vb.menu.eventFilter(
            main_image.vb.menu,
            QtGui.QKeyEvent(
                QtCore.QEvent.KeyPress, QtCore.Qt.Key_Alt, QtCore.Qt.AltModifier
            ),
        )

        logger.info("Check access to cropped data")
        assert isinstance(main_image._current_data_cropped, xr.DataArray)

        if not test_data_type.endswith("nonuniform"):
            logger.info("Opening goldtool from main image")
            main_image.open_in_goldtool()
            qtbot.wait_until(
                lambda: isinstance(
                    win.slicer_area._associated_tools_list[-1], GoldTool
                ),
                timeout=2000,
            )

            logger.info("Opening restool from main image")
            main_image.open_in_restool()
            qtbot.wait_until(
                lambda: isinstance(
                    win.slicer_area._associated_tools_list[-1], ResolutionTool
                ),
                timeout=2000,
            )

            logger.info("Opening dtool from main image")
            main_image.open_in_dtool()
            qtbot.wait_until(
                lambda: isinstance(
                    win.slicer_area._associated_tools_list[-1], DerivativeTool
                ),
                timeout=2000,
            )

        logger.info("Open main image in new window")
        main_image.open_in_new_window()
        qtbot.wait_until(
            lambda: isinstance(win.slicer_area._associated_tools_list[-1], ImageTool),
            timeout=2000,
        )
    finally:
        for img in win.slicer_area.images:
            # Prevent segfault before shutdown
            img.disconnect_signals()
            img.deleteLater()
        if use_dask:
            erlab.interactive.options["io/dask/compute_threshold"] = old_threshold

    logger.info("Closing ImageTool")
    win.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
    win.close()


def test_itool_file_menu_visibility_updates_compute_action(qtbot) -> None:
    data = _TEST_DATA["2D"].copy().chunk()
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    win.slicer_area._in_manager = True
    win.slicer_area.compute_act.setEnabled(False)
    win.mnb._file_menu_visibility()

    assert win.slicer_area.compute_act.isEnabled() == win.slicer_area.data_loadable


def test_copy_selection_code_includes_crop_with_alt(qtbot, monkeypatch) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)),
        dims=["x", "y"],
        coords={"x": np.arange(5), "y": np.arange(5)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    main_image = win.slicer_area.images[0]
    win.slicer_area.set_manual_limits({"x": [1.0, 3.0], "y": [0.0, 2.0]})

    _press_alt(monkeypatch)

    copied: list[str] = []

    def fake_copy(content: str | list[str]) -> str:
        copied.append(content if isinstance(content, str) else "\n".join(content))
        return copied[-1]

    monkeypatch.setattr(erlab.interactive.utils, "copy_to_clipboard", fake_copy)

    main_image.copy_selection_code()
    assert len(copied) == 1
    xr.testing.assert_identical(
        _exec_data_fragment(data, copied[0]),
        data.sel(x=slice(1.0, 3.0), y=slice(0.0, 2.0)),
    )
    win.close()


def test_copy_selection_code_descending_coords(qtbot, monkeypatch) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)),
        dims=["x", "y"],
        coords={"x": np.arange(4, -1, -1), "y": np.arange(4, -1, -1)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    main_image = win.slicer_area.images[0]
    win.slicer_area.set_manual_limits({"x": [1.0, 3.0], "y": [0.0, 2.0]})

    _press_alt(monkeypatch)

    copied: list[str] = []

    def fake_copy(content: str | list[str]) -> str:
        copied.append(content if isinstance(content, str) else "\n".join(content))
        return copied[-1]

    monkeypatch.setattr(erlab.interactive.utils, "copy_to_clipboard", fake_copy)

    main_image.copy_selection_code()
    assert len(copied) == 1
    xr.testing.assert_identical(
        _exec_data_fragment(data, copied[0]),
        data.sel(x=slice(3.0, 1.0), y=slice(2.0, 0.0)),
    )
    win.close()


def test_selection_code_merges_cursor_and_crop_on_alt(qtbot, monkeypatch) -> None:
    data = _TEST_DATA["3D_nonuniform"].copy()
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    main_image = win.slicer_area.images[0]
    alpha_dim = next(str(d) for d in win.slicer_area.data.dims if "alpha" in str(d))
    e_dim = next(str(d) for d in win.slicer_area.data.dims if "eV" in str(d))
    win.slicer_area.set_manual_limits({alpha_dim: [0.0, 2.0], e_dim: [1.0, 3.0]})

    _press_alt(monkeypatch)

    sel_code = main_image.selection_code_for_cursor(
        main_image.slicer_area.current_cursor
    )
    assert sel_code == ".qsel(beta=2.0, eV=slice(1.0, 3.0)).isel(alpha=slice(0, 2))"
    win.close()


def test_qsel_kwargs_multicursor_with_varying_dim(qtbot) -> None:
    data = _TEST_DATA["3D"].copy()
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    main_image = win.slicer_area.images[0]

    win.slicer_area.add_cursor()
    win.slicer_area.set_value(axis=2, value=1.0, cursor=0)
    win.slicer_area.set_value(axis=2, value=3.0, cursor=1)

    kwargs, variable = main_image._uniform_qsel_kwargs_multicursor()
    assert kwargs == {"beta": [1.0, 3.0]}
    assert variable == "beta"

    win.close()


def test_qsel_kwargs_multicursor_width_only_error(qtbot) -> None:
    data = _TEST_DATA["3D"].copy()
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    main_image = win.slicer_area.images[0]

    win.slicer_area.add_cursor()
    win.slicer_area.set_value(axis=2, value=2.0, cursor=0)
    win.slicer_area.set_value(axis=2, value=2.0, cursor=1)
    win.array_slicer.set_bin(0, axis=2, value=3, update=False)
    win.array_slicer.set_bin(1, axis=2, value=1, update=True)

    with pytest.raises(ValueError, match="Cannot plot when"):
        main_image._uniform_qsel_kwargs_multicursor()

    win.close()


def test_qsel_kwargs_multicursor_rejects_nonuniform_axes(qtbot) -> None:
    data = _TEST_DATA["3D_nonuniform"].copy()
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    main_image = win.slicer_area.images[0]
    main_image.display_axis = (1, 2)

    with pytest.raises(ValueError, match="indexing along non-uniform axes"):
        main_image._uniform_qsel_kwargs_multicursor()

    win.close()


def test_qsel_kwargs_multicursor_with_width_and_value_changes(qtbot) -> None:
    data = _TEST_DATA["3D"].copy()
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    main_image = win.slicer_area.images[0]

    win.slicer_area.add_cursor()
    win.slicer_area.set_value(axis=2, value=1.0, cursor=0)
    win.slicer_area.set_value(axis=2, value=3.0, cursor=1)
    win.array_slicer.set_bin(0, axis=2, value=3, update=False)
    win.array_slicer.set_bin(1, axis=2, value=1, update=True)

    kwargs, variable = main_image._uniform_qsel_kwargs_multicursor()
    assert list(kwargs.keys())[:2] == ["beta", "beta_width"]
    assert kwargs["beta"] == [1.0, 3.0]
    assert kwargs["beta_width"] == [3.0, 0.0]
    assert variable == "beta"

    win.close()


def test_qsel_kwargs_multicursor_rejects_multiple_varying_dims(qtbot) -> None:
    data = xr.DataArray(
        np.arange(120).reshape((2, 3, 4, 5)),
        dims=["a", "b", "c", "d"],
        coords={
            "a": np.arange(2),
            "b": np.arange(3),
            "c": np.arange(4),
            "d": np.arange(5),
        },
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    main_image = win.slicer_area.images[0]

    win.slicer_area.add_cursor()
    win.slicer_area.set_value(axis=2, value=1.0, cursor=0)
    win.slicer_area.set_value(axis=3, value=0.0, cursor=0)
    win.slicer_area.set_value(axis=2, value=2.0, cursor=1)
    win.slicer_area.set_value(axis=3, value=1.0, cursor=1)

    with pytest.raises(
        ValueError,
        match="Cannot plot when more than one dimension has differing values"
        " across cursors",
    ):
        main_image._uniform_qsel_kwargs_multicursor()

    win.close()


def test_multicursor_variable_key_with_width_first(qtbot) -> None:
    data = _TEST_DATA["3D"].copy()
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    main_image = win.slicer_area.images[0]

    assert main_image._multicursor_variable_key(["beta_width", "beta"]) == "beta"

    win.close()


def test_make_cursors_single_color_does_not_recreate_cursor(qtbot, monkeypatch) -> None:
    data = _TEST_DATA["2D"].copy()
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    color = win.slicer_area.cursor_colors[0].name()

    def _unexpected_add_cursor(*args, **kwargs):
        raise AssertionError("add_cursor should not be called for one cursor")

    monkeypatch.setattr(win.slicer_area, "add_cursor", _unexpected_add_cursor)

    win.slicer_area.make_cursors([color], update=False)
    win.slicer_area.make_cursors([color], update=True)
    assert win.slicer_area.n_cursors == 1
    assert win.slicer_area.cursor_colors[0].name() == color

    win.close()


def test_plot_code_multicursor_line_includes_limits_and_colors(qtbot) -> None:
    data = _TEST_DATA["2D"].copy()
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    line_plot = win.slicer_area.get_axes(1)

    win.slicer_area.add_cursor()
    win.slicer_area.set_value(axis=1, value=1.0, cursor=0)
    win.slicer_area.set_value(axis=1, value=3.0, cursor=1)
    win.slicer_area.cursor_colors = [QtGui.QColor("#123456"), QtGui.QColor("#654321")]
    line_plot.set_normalize(True)
    win.slicer_area.set_manual_limits({"alpha": [1.0, 3.0]})

    code = line_plot._plot_code_multicursor()
    assert "line_colors" in code
    assert "line / line.mean()" in code
    assert "xlim=(1.0, 3.0)" in code
    assert "beta" not in code

    win.close()


def test_plot_code_multicursor_line_uniform_default_colors(qtbot) -> None:
    data = _TEST_DATA["2D"].copy()
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    line_plot = win.slicer_area.get_axes(1)

    win.slicer_area.add_cursor()
    win.slicer_area.set_value(axis=1, value=1.0, cursor=0)
    win.slicer_area.set_value(axis=1, value=3.0, cursor=1)

    code = line_plot._plot_code_multicursor()
    assert "for line in " in code
    assert 'transpose("eV", ...)' in code
    assert "enumerate(" not in code

    win.close()


def test_plot_code_multicursor_line_without_variation(qtbot) -> None:
    data = _TEST_DATA["2D"].copy()
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    line_plot = win.slicer_area.get_axes(1)

    code = line_plot._plot_code_multicursor()
    assert code.startswith("fig, ax = plt.subplots()")
    assert ".plot(ax=ax)" in code
    assert "for" not in code

    win.close()


def test_plot_code_multicursor_image_includes_norm_settings(qtbot) -> None:
    data = _TEST_DATA["3D"].copy()
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    main_image = win.slicer_area.images[0]

    win.slicer_area.add_cursor()
    win.slicer_area.set_value(axis=2, value=1.0, cursor=0)
    win.slicer_area.set_value(axis=2, value=2.0, cursor=1)
    win.slicer_area.set_colormap(
        cmap="magma",
        gamma=1.5,
        reverse=True,
        high_contrast=True,
        zero_centered=True,
    )
    win.slicer_area.levels = (1.0, 4.0)
    win.slicer_area.lock_levels(True)
    main_image.getViewBox().setAspectLocked(True)

    code = main_image._plot_code_multicursor()
    assert "transpose=True" in code
    assert "same_limits=True" in code
    assert 'axis="image"' in code
    assert 'cmap="magma_r"' in code
    assert "CenteredInversePowerNorm" in code

    win.close()


def test_slicer_area_colormap_lut_matches_dense_powernorm(qtbot) -> None:
    kwargs = {
        "cmap": "magma",
        "gamma": 0.01,
        "reverse": True,
        "high_contrast": True,
        "zero_centered": True,
    }
    win = itool(_TEST_DATA["2D"].copy(), execute=False)
    qtbot.addWidget(win)

    win.slicer_area.set_colormap(**kwargs)
    dense = erlab.interactive.colors.pg_colormap_powernorm(**kwargs)

    assert np.array_equal(win.slicer_area._imageitems[0].lut, dense.getStops()[1])

    win.close()


def test_plot_code_multicursor_image_without_cursor_variation_nonuniform(qtbot) -> None:
    data = _TEST_DATA["3D_nonuniform"].copy()
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    image_plot = win.slicer_area.get_axes(5)  # display_axis=(2, 1), non-display alpha

    code = image_plot._plot_code_multicursor()
    assert "selected = [" not in code
    assert "selected = data." in code

    win.close()


@pytest.mark.parametrize("bin_value", [1, 3])
def test_plot_code_multicursor_image_supports_nonuniform_hidden_axis(
    qtbot, bin_value
) -> None:
    data = _TEST_DATA["3D_nonuniform"].copy()
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    image_plot = win.slicer_area.get_axes(5)  # display_axis=(2, 1), hidden alpha

    win.slicer_area.add_cursor()
    win.slicer_area.set_value(axis=0, value=0.1, cursor=0)
    win.slicer_area.set_value(axis=0, value=0.8, cursor=1)
    win.array_slicer.set_bin(0, axis=0, value=bin_value, update=False)
    win.array_slicer.set_bin(1, axis=0, value=bin_value, update=True)

    code = image_plot._plot_code_multicursor()
    assert "selected = [" in code
    assert "plot_slices" in code
    assert ".isel(alpha=" in code
    if bin_value == 1:
        assert ".qsel.average(" not in code
    else:
        assert ".qsel.average(" in code
        assert '.qsel.average("alpha")' in code

    win.close()


def test_plot_code_multicursor_line_without_cursor_variation_nonuniform(qtbot) -> None:
    data = _TEST_DATA["3D_nonuniform"].copy()
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    line_plot = win.slicer_area.get_axes(3)  # display_axis=(2,), non-display alpha/eV

    win.slicer_area.add_cursor()
    win.slicer_area.set_value(axis=0, value=0.4, cursor=0)
    win.slicer_area.set_value(axis=0, value=0.4, cursor=1)
    win.array_slicer.set_bin(0, axis=0, value=3, update=False)
    win.array_slicer.set_bin(1, axis=0, value=3, update=True)

    code = line_plot._plot_code_multicursor()
    assert "for line in" not in code
    assert ".plot(ax=ax)" in code

    win.close()


def test_plot_code_multicursor_line_nonuniform_custom_colors_and_widths(qtbot) -> None:
    data = _TEST_DATA["3D_nonuniform"].copy()
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    line_plot = win.slicer_area.get_axes(3)  # display_axis=(2,), non-display alpha/eV

    win.slicer_area.add_cursor()
    win.slicer_area.set_value(axis=0, value=0.1, cursor=0)
    win.slicer_area.set_value(axis=0, value=0.8, cursor=1)
    win.array_slicer.set_bin(0, axis=0, value=3, update=False)
    win.array_slicer.set_bin(1, axis=0, value=1, update=True)
    win.slicer_area.cursor_colors = [QtGui.QColor("#123456"), QtGui.QColor("#654321")]

    code = line_plot._plot_code_multicursor()
    assert "for i, line in enumerate([" in code
    assert "line_colors" in code

    win.close()


def test_selection_expr_for_cursor_multiple_average_dims_with_quotes(qtbot) -> None:
    data = xr.DataArray(
        np.arange(625).reshape((5, 5, 5, 5)),
        dims=['a"b', "eV", "c", "beta"],
        coords={
            'a"b': np.array([0.1, 0.4, 0.5, 0.55, 0.8]),
            "eV": np.arange(5),
            "c": np.array([1.0, 1.2, 1.7, 2.5, 3.0]),
            "beta": np.arange(5),
        },
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    image_plot = win.slicer_area.get_axes(7)  # display_axis=(3, 2)

    win.array_slicer.set_bin(0, axis=0, value=3, update=False)
    win.array_slicer.set_bin(0, axis=2, value=3, update=True)

    expr = image_plot._selection_expr_for_cursor("data", 0, (0, 2))
    assert ".qsel.average((" in expr
    assert "'a\"b'" in expr
    assert '"c"' in expr

    win.close()


def test_selection_expr_for_cursor_uniform_axis_only(qtbot) -> None:
    data = _TEST_DATA["3D_nonuniform"].copy()
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    image_plot = win.slicer_area.get_axes(5)  # display_axis=(2, 1), non-display alpha

    expr = image_plot._selection_expr_for_cursor("data", 0, (1,))
    assert ".isel(" not in expr
    assert ".qsel(" in expr

    win.close()


def test_selection_expr_for_cursor_preserves_nonstring_qsel_dim(qtbot) -> None:
    data = xr.DataArray(
        np.arange(24).reshape((2, 3, 4)),
        dims=["k-space", "y", "z"],
        coords={
            "k-space": np.arange(2, dtype=float),
            "y": np.arange(3),
            "z": np.arange(4),
        },
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    expr = win.slicer_area.main_image._selection_expr_for_cursor("data", 0, (0,))
    assert expr == 'data.qsel({"k-space": 0.0})'

    win.close()


@pytest.mark.parametrize("bin_value", [1, 3])
def test_plot_code_multicursor_line_supports_nonuniform_hidden_axis(
    qtbot, bin_value
) -> None:
    data = _TEST_DATA["3D_nonuniform"].copy()
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    line_plot = win.slicer_area.get_axes(3)  # display_axis=(2,), hidden alpha/eV

    win.slicer_area.add_cursor()
    win.slicer_area.set_value(axis=0, value=0.1, cursor=0)
    win.slicer_area.set_value(axis=0, value=0.8, cursor=1)
    win.array_slicer.set_bin(0, axis=0, value=bin_value, update=False)
    win.array_slicer.set_bin(1, axis=0, value=bin_value, update=True)

    code = line_plot._plot_code_multicursor()
    assert "for line in [" in code
    assert ".isel(alpha=" in code
    assert ".qsel(" in code
    if bin_value == 1:
        assert ".qsel.average(" not in code
    else:
        assert ".qsel.average(" in code
        assert '.qsel.average("alpha")' in code

    win.close()


def test_plot_code_multicursor_image_with_non_identifier_dim_name(qtbot) -> None:
    data = xr.DataArray(
        np.arange(125).reshape((5, 5, 5)),
        dims=["alpha", "eV", "k-space"],
        coords={
            "alpha": np.arange(5),
            "eV": np.arange(5),
            "k-space": np.arange(5),
        },
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    main_image = win.slicer_area.images[0]

    code = main_image._plot_code_multicursor()
    assert 'selected = data.qsel({"k-space": 2.0})' in code

    win.close()


def test_plot_code_multicursor_image_includes_both_crop_limits(qtbot) -> None:
    data = _TEST_DATA["2D"].copy()
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    main_image = win.slicer_area.images[0]

    win.slicer_area.set_manual_limits({"alpha": [1.0, 3.0], "eV": [0.0, 2.0]})
    code = main_image._plot_code_multicursor()
    assert "xlim=(1.0, 3.0)" in code
    assert "ylim=(0.0, 2.0)" in code

    win.close()


def test_plot_with_matplotlib_executes_in_manager(qtbot, monkeypatch) -> None:
    data = _TEST_DATA["3D"].copy()
    win = itool(data, execute=False)
    win.slicer_area._in_manager = True
    qtbot.addWidget(win)
    main_image = win.slicer_area.images[0]

    class _Console:
        def __init__(self) -> None:
            self.executed: list[str] = []

        def initialize_kernel(self) -> None:
            self.initialized = True

        def execute(self, code: str) -> None:
            self.executed.append(code)

    console = _Console()

    class _Manager:
        def __init__(self) -> None:
            self.console = types.SimpleNamespace(_console_widget=console)

        def ensure_console_initialized(self) -> None:
            self.initialized = True

        def index_from_slicer_area(self, slicer_area):
            assert slicer_area is win.slicer_area
            return 0

    monkeypatch.setattr(
        erlab.interactive.imagetool.manager, "_manager_instance", _Manager()
    )

    main_image.plot_with_matplotlib()
    assert console.executed
    assert "tools[0].data" in console.executed[0]
    assert console.executed[0].strip().endswith("fig.show()")

    win.close()


def test_copy_matplotlib_code_uses_generated_output(qtbot, monkeypatch) -> None:
    import matplotlib.pyplot as plt

    data = _TEST_DATA["3D"].copy()
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    main_image = win.slicer_area.images[0]

    win.slicer_area.add_cursor()
    win.slicer_area.set_value(axis=2, value=1.0, cursor=0)
    win.slicer_area.set_value(axis=2, value=3.0, cursor=1)

    copied: dict[str, str] = {}

    def _copy(arg: str) -> None:
        copied["text"] = arg

    monkeypatch.setattr(erlab.interactive.utils, "copy_to_clipboard", _copy)

    main_image.copy_matplotlib_code()
    namespace = _exec_generated_code(
        copied["text"],
        {"data": data.copy(deep=True), "plt": plt, "eplt": erlab.plotting},
    )
    assert "fig" in namespace
    assert len(np.asarray(namespace["axs"]).ravel()) == 2
    plt.close(namespace["fig"])

    win.close()


def test_cursor_colors_follow_coordinate(qtbot) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)),
        dims=["x", "y"],
        coords={"x": np.arange(5), "y": np.arange(5)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    slicer = win.slicer_area
    slicer.array_slicer._cursor_color_params = (
        ("x",),
        "x",
        "coolwarm",
        False,
        0.0,
        1.0,
    )
    slicer._refresh_cursor_colors(tuple(range(slicer.n_cursors)), None)

    cmap = erlab.interactive.colors.pg_colormap_from_name("coolwarm")
    coords = data.coords["x"].values
    mn, mx = np.min(coords), np.max(coords)
    scale = (1.0 - 0.0) / (mx - mn)
    idx = slicer.array_slicer.get_index(0, 0)
    expected = cmap.map((coords[idx] - mn) * scale, mode=cmap.QCOLOR).name()
    assert slicer.cursor_colors[0].name() == expected

    slicer.set_value(axis=0, value=4.0, cursor=0)
    idx = slicer.array_slicer.get_index(0, 0)
    expected = cmap.map((coords[idx] - mn) * scale, mode=cmap.QCOLOR).name()
    assert slicer.cursor_colors[0].name() == expected

    slicer.set_value(axis=1, value=4.0, cursor=0)
    assert slicer.cursor_colors[0].name() == expected

    for ax in slicer.axes:
        display_ax = ax.display_axis[0]
        assert (
            ax.cursor_lines[0][display_ax].pen.color().name()
            == slicer.cursor_colors[0].name()
        )

    win.close()


def test_cursor_colors_from_associated_coord(qtbot) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)),
        dims=["x", "y"],
        coords={
            "x": np.arange(5),
            "y": np.arange(5),
            "temp": ("x", np.linspace(-1, 1, 5)),
        },
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    slicer = win.slicer_area
    slicer.array_slicer._cursor_color_params = (
        ("x",),
        "temp",
        "viridis",
        True,
        0.2,
        0.8,
    )
    slicer._refresh_cursor_colors(tuple(range(slicer.n_cursors)), None)

    cmap = erlab.interactive.colors.pg_colormap_from_name("viridis")
    coords = data.coords["temp"].values
    mn, mx = np.min(coords), np.max(coords)
    scale = (0.8 - 0.2) / (mx - mn)

    idx = slicer.array_slicer.get_index(0, 0)
    raw = (coords[idx] - mn) * scale + 0.2
    expected = cmap.map(1 - raw, mode=cmap.QCOLOR).name()
    assert slicer.cursor_colors[0].name() == expected

    slicer.set_value(axis=0, value=1.0, cursor=0)
    idx = slicer.array_slicer.get_index(0, 0)
    raw = (coords[idx] - mn) * scale + 0.2
    expected = cmap.map(1 - raw, mode=cmap.QCOLOR).name()
    assert slicer.cursor_colors[0].name() == expected

    win.close()


def test_associated_coord_profile_nd_cursor_and_bins(qtbot) -> None:
    x = np.arange(3, dtype=float)
    y = np.arange(4, dtype=float)
    z = np.arange(5, dtype=float)
    plane = x[:, None] * 10.0 + y[None, :] ** 2
    full = x[:, None, None] * 100.0 + y[None, :, None] * 10.0 + z[None, None, :]
    data = xr.DataArray(
        np.zeros((3, 4, 5), dtype=float),
        dims=["x", "y", "z"],
        coords={
            "x": x,
            "y": y,
            "z": z,
            "temp": ("x", x + 300.0),
            "plane": (("x", "y"), plane),
            "full": (("x", "y", "z"), full),
            "label": ("x", ["a", "b", "c"]),
            "complex": ("x", np.arange(3, dtype=complex)),
            "scalar": 1.0,
        },
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    slicer = win.slicer_area.array_slicer

    assert slicer.associated_coord_dims == {
        "temp": ("x",),
        "plane": ("x", "y"),
        "full": ("x", "y", "z"),
    }

    dialog = _AssociatedCoordsDialog(win.slicer_area)
    qtbot.addWidget(dialog)
    assert set(dialog._checks) == {"temp", "plane", "full"}

    x_profile, full_profile = slicer.associated_coord_profile("full", 0, (0,))
    np.testing.assert_allclose(x_profile, x)
    np.testing.assert_allclose(full_profile, x * 100.0 + 12.0)

    slicer.set_index(0, 1, 2, update=False)
    slicer.set_index(0, 2, 3, update=False)
    _, full_profile = slicer.associated_coord_profile("full", 0, (0,))
    np.testing.assert_allclose(full_profile, x * 100.0 + 23.0)

    slicer.set_index(0, 1, 1, update=False)
    slicer.set_bin(0, 1, 3, update=False)
    _, plane_profile = slicer.associated_coord_profile("plane", 0, (0,))
    np.testing.assert_allclose(plane_profile, x * 10.0 + np.mean([0.0, 1.0, 4.0]))

    assert slicer.associated_coord_profile("plane", 0, (2,)) is None
    assert slicer.associated_coord_profile("plane", 0, (0, 1)) is None
    assert slicer.associated_coord_profile("missing", 0, (0,)) is None
    assert slicer.cursor_color_coord(0, ("y", "x"), "plane") is None
    win.close()


def test_point_value_context_menu_selects_associated_coord(qtbot) -> None:
    temp = np.arange(25, dtype=float).reshape(5, 5) + 100.0
    data = xr.DataArray(
        np.zeros((5, 5), dtype=float),
        dims=["x", "y"],
        coords={
            "x": np.arange(5, dtype=float),
            "y": np.arange(5, dtype=float),
            "temp": (("x", "y"), temp),
        },
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    control = win.docks[0].widget().findChild(ItoolCrosshairControls)
    assert control is not None
    control.update_content()

    win.slicer_area.array_slicer.twin_coord_names = {"temp"}
    menu = control._build_point_value_context_menu()
    action_data = [
        action.data() for action in menu.actions() if not action.isSeparator()
    ]

    assert control._readout_source_indicator.isHidden()
    header_action = _menu_action_by_data(menu, ("display_header", None))
    assert header_action.isSeparator()
    assert action_data == [
        ("copy", None),
        ("select_all", None),
        ("data", None),
        ("coord", "temp"),
    ]
    assert _menu_action_by_data(menu, ("data", None)).isChecked()
    temp_action = _menu_action_by_data(menu, ("coord", "temp"))
    assert not temp_action.icon().isNull()

    temp_action.trigger()
    assert control._readout_source == "temp"
    assert not control._readout_source_indicator.isHidden()
    assert control.spin_dat.value() == temp[2, 2]

    win.slicer_area.set_index(0, 4)
    assert control.spin_dat.value() == temp[4, 2]

    control._set_computed_point_value(-1.0)
    assert control.spin_dat.value() == temp[4, 2]

    win.slicer_area.array_slicer.twin_coord_names = set()
    assert control._readout_source is None
    assert control._readout_source_indicator.isHidden()
    assert control.spin_dat.value() == 0.0
    win.close()


def test_profile_menu_opens_associated_coord_targets(
    qtbot, monkeypatch: pytest.MonkeyPatch
) -> None:
    x = np.arange(3, dtype=float)
    y = np.arange(4, dtype=float)
    plane = x[:, None] * 10.0 + y[None, :]
    data = xr.DataArray(
        np.zeros((3, 4, 2), dtype=float),
        dims=["x", "y", "z"],
        coords={
            "x": x,
            "y": y,
            "z": np.arange(2, dtype=float),
            "plane": (("x", "y"), plane),
            "temp": ("x", x + 100.0),
        },
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    profile = win.slicer_area.profiles[0]
    win.slicer_area.array_slicer.twin_coord_names = {"plane", "temp"}

    captured: list[
        tuple[
            xr.DataArray,
            erlab.interactive.imagetool.provenance.ToolProvenanceSpec,
            erlab.interactive.imagetool.provenance.ImageToolSelectionSourceBinding
            | None,
            bool,
        ]
    ] = []

    def _capture_open(data, source_spec, *, source_binding=None, use_parent_colormap):
        captured.append((data, source_spec, source_binding, use_parent_colormap))

    monkeypatch.setattr(profile, "_open_data_in_new_window", _capture_open)

    profile._refresh_associated_coord_menu()
    assert profile._associated_coord_menu is not None
    assert profile._associated_coord_menu.menuAction().isVisible()
    plot_actions = profile.vb.menu.actions()
    menu_index = plot_actions.index(profile._associated_coord_menu.menuAction())
    assert plot_actions[menu_index - 1].isSeparator()
    assert plot_actions[menu_index - 1].isVisible()
    assert plot_actions[menu_index + 1].isSeparator()
    assert plot_actions[menu_index + 1].isVisible()

    temp_action = _menu_action_by_data(
        profile._associated_coord_menu, ("associated_coord_open", "temp")
    )
    assert temp_action.menu() is None
    temp_action.trigger()
    temp_data, temp_spec, temp_binding, use_parent_colormap = captured[-1]
    xr.testing.assert_identical(temp_data, data.coords["temp"])
    assert temp_spec.kind == "public_data"
    assert temp_spec.operations[-1].op == "select_coord"
    assert temp_binding is None
    assert use_parent_colormap is False

    coord_menu = _menu_action_by_data(
        profile._associated_coord_menu, ("associated_coord", "plane")
    ).menu()
    assert coord_menu is not None

    _menu_action_by_data(coord_menu, ("associated_coord_full", "plane")).trigger()
    full_data, full_spec, full_binding, use_parent_colormap = captured[-1]
    xr.testing.assert_identical(full_data, data.coords["plane"])
    assert full_spec.kind == "public_data"
    assert full_spec.operations[-1].op == "select_coord"
    assert full_binding is None
    assert use_parent_colormap is False

    _menu_action_by_data(coord_menu, ("associated_coord_profile", "plane")).trigger()
    profile_data, profile_spec, profile_binding, use_parent_colormap = captured[-1]
    xr.testing.assert_identical(profile_data, data.isel(y=1, z=0).coords["plane"])
    assert profile_spec.kind == "selection"
    assert profile_spec.operations[-1].op == "select_coord"
    assert profile_binding is None
    assert use_parent_colormap is False
    win.close()


def test_profile_associated_coord_menu_does_not_compute_dask_coord(qtbot) -> None:
    da = pytest.importorskip("dask.array")
    from dask.callbacks import Callback

    x = np.arange(3, dtype=float)
    y = np.arange(4, dtype=float)
    plane = da.from_array(x[:, None] * 10.0 + y[None, :], chunks=(1, 4))
    data = xr.DataArray(
        np.zeros((3, 4, 2), dtype=float),
        dims=["x", "y", "z"],
        coords={
            "x": x,
            "y": y,
            "z": np.arange(2, dtype=float),
            "plane": (("x", "y"), plane),
        },
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    profile = win.slicer_area.profiles[0]
    win.slicer_area.array_slicer.twin_coord_names = {"plane"}

    computed_keys: list[object] = []
    with Callback(pretask=lambda key, _dsk, _state: computed_keys.append(key)):
        profile._refresh_associated_coord_menu()

    assert computed_keys == []
    assert profile._associated_coord_menu is not None
    assert profile._associated_coord_menu.menuAction().isVisible()
    associated_coord_action = _menu_action_by_data(
        profile._associated_coord_menu, ("associated_coord", "plane")
    )
    assert associated_coord_action.menu() is not None
    win.close()


def test_associated_coord_dialog_empty_warning(qtbot, monkeypatch) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)),
        dims=["x", "y"],
        coords={"x": np.arange(5), "y": np.arange(5)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    dialog = _AssociatedCoordsDialog(win.slicer_area)
    qtbot.addWidget(dialog)
    warnings: list[tuple[str, str]] = []

    def _record_warning(_parent, title, message, *args, **kwargs):
        warnings.append((title, message))
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _record_warning)

    assert dialog.exec() == QtWidgets.QDialog.DialogCode.Rejected
    assert warnings == [
        (
            "No Associated Coordinates",
            "No numeric non-dimension coordinates were found in the data.",
        )
    ]
    win.close()


def test_cursor_colors_from_nd_associated_coord(qtbot) -> None:
    temp = np.arange(25, dtype=float).reshape(5, 5)
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)),
        dims=["x", "y"],
        coords={
            "x": np.arange(5),
            "y": np.arange(5),
            "temp": (("x", "y"), temp),
        },
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    slicer = win.slicer_area

    dialog = _CursorColorCoordDialog(slicer)
    qtbot.addWidget(dialog)
    assert "temp" in [
        dialog.coord_combo.itemText(i) for i in range(dialog.coord_combo.count())
    ]
    dialog.main_group.setChecked(True)
    dialog.coord_combo.setCurrentText("temp")
    assert dialog.get_checked_coord_name() == (("x", "y"), "temp")
    dialog.coord_combo.setCurrentText("x")
    assert dialog.get_checked_coord_name() == (("x",), "x")
    dialog.main_group.setChecked(False)
    assert dialog.get_checked_coord_name() is None
    dialog.main_group.setChecked(True)
    dialog.coord_combo.setEditable(True)
    dialog.coord_combo.setCurrentText("missing")
    assert dialog.get_checked_coord_name() is None
    dialog.coord_combo.setCurrentText("temp")

    slicer.array_slicer._cursor_color_params = (
        ("x", "y"),
        "temp",
        "viridis",
        False,
        0.2,
        0.8,
    )
    slicer._refresh_cursor_colors(tuple(range(slicer.n_cursors)), None)

    cmap = erlab.interactive.colors.pg_colormap_from_name("viridis")
    mn, mx = np.min(temp), np.max(temp)
    scale = (0.8 - 0.2) / (mx - mn)
    idx_x, idx_y = slicer.array_slicer.get_indices(0)
    raw = (temp[idx_x, idx_y] - mn) * scale + 0.2
    expected = cmap.map(raw, mode=cmap.QCOLOR).name()
    assert slicer.cursor_colors[0].name() == expected

    slicer.set_value(axis=1, value=4.0, cursor=0)
    idx_x, idx_y = slicer.array_slicer.get_indices(0)
    raw = (temp[idx_x, idx_y] - mn) * scale + 0.2
    expected = cmap.map(raw, mode=cmap.QCOLOR).name()
    assert slicer.cursor_colors[0].name() == expected

    win.close()


def test_cursor_color_invalid_state_is_cleared(qtbot) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)),
        dims=["x", "y"],
        coords={"x": np.arange(5), "y": np.arange(5)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    slicer = win.slicer_area
    slicer.array_slicer._cursor_color_params = (
        ("x",),
        "missing",
        "viridis",
        False,
        0.0,
        1.0,
    )

    slicer._refresh_cursor_colors(tuple(range(slicer.n_cursors)), None)

    assert slicer.array_slicer._cursor_color_params is None
    win.close()


def test_cursor_color_dialog_accept_disabled_clears_params(qtbot) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)),
        dims=["x", "y"],
        coords={
            "x": np.arange(5),
            "y": np.arange(5),
            "temp": ("x", np.linspace(-1, 1, 5)),
        },
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    slicer = win.slicer_area
    slicer.array_slicer._cursor_color_params = (
        ("x",),
        "temp",
        "viridis",
        False,
        0.0,
        1.0,
    )
    dialog = _CursorColorCoordDialog(slicer)
    qtbot.addWidget(dialog)
    dialog.main_group.setChecked(False)

    dialog.accept()

    assert slicer.array_slicer._cursor_color_params is None
    win.close()


def test_cursor_color_state_restores_legacy_dim(qtbot) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)),
        dims=["x", "y"],
        coords={
            "x": np.arange(5),
            "y": np.arange(5),
            "temp": ("x", np.linspace(-1, 1, 5)),
        },
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    state = win.slicer_area.array_slicer.state
    state["cursor_color_params"] = ("x", "temp", "magma", True, 0.1, 0.9)

    win.slicer_area.array_slicer.state = state

    assert win.slicer_area.array_slicer._cursor_color_params == (
        ("x",),
        "temp",
        "magma",
        True,
        0.1,
        0.9,
    )

    state = win.slicer_area.array_slicer.state
    win.slicer_area.array_slicer.state = state
    assert win.slicer_area.array_slicer._cursor_color_params == (
        ("x",),
        "temp",
        "magma",
        True,
        0.1,
        0.9,
    )
    win.close()


def test_manual_cursor_colors_disable_coord_updates(qtbot) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)),
        dims=["x", "y"],
        coords={"x": np.arange(5), "y": np.arange(5)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    slicer = win.slicer_area
    slicer.array_slicer._cursor_color_params = (
        ("x",),
        "x",
        "plasma",
        False,
        0.0,
        1.0,
    )
    slicer._refresh_cursor_colors(tuple(range(slicer.n_cursors)), None)

    slicer.set_value(axis=0, value=4.0, cursor=0)
    dynamic_color = slicer.cursor_colors[0].name()

    slicer.set_cursor_colors(["#123456"])
    assert slicer.array_slicer._cursor_color_params is None

    slicer.set_value(axis=0, value=0.0, cursor=0)
    assert slicer.cursor_colors[0].name() == "#123456"
    assert slicer.cursor_colors[0].name() != dynamic_color

    win.close()


def test_cursor_color_coord_dialog_updates_params(
    qtbot, accept_dialog, monkeypatch
) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)),
        dims=["x", "y"],
        coords={
            "x": np.arange(5),
            "y": np.arange(5),
            "temp": ("x", np.linspace(-1, 1, 5)),
        },
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    slicer = win.slicer_area
    slicer.array_slicer._cursor_color_params = (
        ("x",),
        "temp",
        "magma",
        True,
        0.1,
        0.9,
    )

    called: dict[str, object] = {}

    def _refresh(cursor, axes):
        called["cursor"] = cursor
        called["axes"] = axes

    monkeypatch.setattr(slicer, "_refresh_cursor_colors", _refresh)

    def prepare(dialog: _CursorColorCoordDialog) -> None:
        assert dialog.coord_combo.currentText() == "temp"
        assert dialog.cmap_combo.currentText() == "magma"
        assert dialog.reverse_check.isChecked()
        assert dialog.start_spin.value() == 0.1
        assert dialog.stop_spin.value() == 0.9

        dialog.main_group.setChecked(True)
        dialog.cmap_combo.setCurrentText("viridis")
        dialog.reverse_check.setChecked(False)
        dialog.start_spin.setValue(0.0)
        dialog.stop_spin.setValue(0.5)
        dialog.coord_combo.setCurrentText("temp")

    accept_dialog(slicer._set_cursor_colors_by_coord, pre_call=prepare)

    assert slicer.array_slicer._cursor_color_params == (
        ("x",),
        "temp",
        "viridis",
        False,
        0.0,
        0.5,
    )
    assert called == {"cursor": tuple(range(slicer.n_cursors)), "axes": None}

    win.close()


def test_itool_edit_cursor_colors(qtbot, accept_dialog) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)),
        dims=["x", "y"],
        coords={"x": np.arange(5), "y": np.arange(5)},
    )

    win = itool(data, execute=False)
    qtbot.addWidget(win)

    win.slicer_area.add_cursor()
    win.slicer_area.add_cursor()
    assert win.slicer_area.n_cursors == 3

    def parse_dialog(dialog: erlab.interactive.colors.ColorCycleDialog):
        dialog.set_from_cmap()

    accept_dialog(win.slicer_area.edit_cursor_colors, pre_call=parse_dialog)

    assert [c.name() for c in win.slicer_area.cursor_colors] == [
        "#5978e3",
        "#dddddd",
        "#d75344",
    ]

    for plot_item in win.slicer_area.profiles:
        for cursor, plot_data_item in enumerate(plot_item.slicer_data_items):
            assert (
                plot_data_item.opts["pen"].color().name()
                == win.slicer_area.cursor_colors[cursor].name()
            )

    win.close()


@pytest.mark.parametrize("coord_dtype", [np.int32, np.int64, np.float32, np.float64])
@pytest.mark.parametrize("val_dtype", [np.int32, np.int64, np.float32, np.float64])
@pytest.mark.parametrize("use_dask", [True, False], ids=["dask", "no_dask"])
def test_itool_dtypes(
    qtbot, move_and_compare_values, val_dtype, coord_dtype, use_dask
) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(val_dtype),
        dims=["x", "y"],
        coords={
            "x": np.arange(5, dtype=coord_dtype),
            "y": np.array([1, 3, 2, 7, 8], dtype=coord_dtype),  # non-uniform
        },
    )
    if use_dask:
        data = data.chunk()

        old_threshold = erlab.interactive.options["io/dask/compute_threshold"]
        # force compute for dask
        erlab.interactive.options["io/dask/compute_threshold"] = 0

    win: ImageTool | None = None
    try:
        win = itool(data, execute=False)
        qtbot.addWidget(win)

        move_and_compare_values(qtbot, win, [12.0, 7.0, 6.0, 11.0])

        if use_dask:
            win.slicer_area.compute_act.trigger()
            qtbot.wait_until(lambda: not win.slicer_area.data_chunked, timeout=2000)

    finally:
        if use_dask:
            erlab.interactive.options["io/dask/compute_threshold"] = old_threshold
        if win is not None:
            for img in win.slicer_area.images:
                # Prevent segfault before shutdown
                img.disconnect_signals()
                img.deleteLater()
            win.close()
            QtWidgets.QApplication.sendPostedEvents(None, 0)
            QtWidgets.QApplication.processEvents()


def test_parse_data() -> None:
    with pytest.raises(
        TypeError,
        match=r"Unsupported input type str. Expected DataArray, Dataset, DataTree, "
        r"numpy array, or a list of DataArray or numpy arrays.",
    ):
        erlab.interactive.imagetool.viewer._parse_input("string")


def test_itool_load(qtbot, monkeypatch, move_and_compare_values, accept_dialog) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)),
        dims=["x", "y"],
        coords={"x": np.arange(5), "y": np.arange(5)},
    )

    win = itool(np.zeros((2, 2)), execute=False)
    qtbot.addWidget(win)

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        filename = f"{tmp_dir_name}/data.h5"
        data.to_netcdf(filename, engine="h5netcdf")

        def _go_to_file(dialog: QtWidgets.QFileDialog):
            dialog.setDirectory(tmp_dir_name)
            dialog.selectFile(filename)
            focused = dialog.focusWidget()
            if isinstance(focused, QtWidgets.QLineEdit):
                focused.setText("data.h5")

        accept_dialog(lambda: win._open_file(native=False), pre_call=_go_to_file)
        move_and_compare_values(qtbot, win, [12.0, 7.0, 6.0, 11.0])

        def _replace_average(dialog: AverageDialog) -> None:
            dialog.dim_checks["x"].setChecked(True)
            dialog.launch_mode_combo.setCurrentText("Replace Current")

        accept_dialog(win.mnb._average, pre_call=_replace_average)

        assert win.provenance_spec is not None
        entries = win.provenance_spec.display_entries()
        assert entries[0].label == "Load data from file 'data.h5'"
        assert any("Average" in entry.label for entry in entries)

        display_code = win.provenance_spec.display_code()
        assert display_code is not None
        assert "data =" not in display_code
        namespace = _exec_generated_code(display_code, {})
        derived = namespace["derived"]
        assert isinstance(derived, xr.DataArray)
        xarray.testing.assert_identical(
            derived.rename(None),
            data.astype(np.float64).qsel.average("x").rename(None),
        )

        assert win.slicer_area._file_path is None
        assert win.slicer_area.reloadable
        updated = data + 100
        updated.to_netcdf(filename, engine="h5netcdf")

        def _fail_derivation_code(self) -> str | None:
            raise AssertionError("provenance reload must not execute generated code")

        monkeypatch.setattr(
            type(win.provenance_spec),
            "derivation_code",
            _fail_derivation_code,
        )

        with qtbot.wait_signal(win.slicer_area.sigDataChanged):
            win.slicer_area.reload()

        assert win.slicer_area._file_path is None
        xarray.testing.assert_identical(
            win.slicer_area._data.rename(None),
            updated.astype(np.float64).qsel.average("x").rename(None),
        )

    win.close()


def test_itool_provenance_reload_rejects_incomplete_or_invalid_replay(
    qtbot,
    tmp_path: pathlib.Path,
) -> None:
    win = itool(xr.DataArray(np.arange(4.0), dims=("x",)), execute=False)
    qtbot.addWidget(win)
    prov = erlab.interactive.imagetool.provenance

    with pytest.raises(RuntimeError, match="cannot be reloaded"):
        win.slicer_area._fetch_for_provenance_reload()

    def _file_source(path: pathlib.Path) -> prov.FileLoadSource:
        return prov.FileLoadSource(
            path=path,
            loader_label="Loader",
            loader_text="xarray.load_dataarray",
            kwargs_text="(none)",
            replay_call=prov.FileReplayCall(
                kind="callable",
                target="xarray.load_dataarray",
                kwargs={},
                selected_index=0,
            ),
            load_code=None,
        )

    missing_file = tmp_path / "missing.h5"
    win.set_provenance_spec(
        prov.file_load(
            start_label="Load missing file",
            seed_code="derived = xr.DataArray([1.0])",
            file_load_source=_file_source(missing_file),
        )
    )
    assert not win.slicer_area.reloadable
    with pytest.raises(FileNotFoundError):
        win.slicer_area._fetch_for_provenance_reload()

    source_file = tmp_path / "source.h5"
    xr.DataArray(np.arange(3.0), dims=("x",)).to_netcdf(
        source_file,
        engine="h5netcdf",
    )
    win.set_provenance_spec(
        prov.script(
            start_label="Needs external data",
            seed_code="derived = data",
            active_name="derived",
            file_load_source=_file_source(source_file),
        )
    )
    assert not win.slicer_area.reloadable
    with pytest.raises(RuntimeError, match="provenance"):
        win.slicer_area._fetch_for_provenance_reload()

    win.set_provenance_spec(
        prov.file_load(
            start_label="Bad selected index",
            seed_code="derived = xr.load_dataarray(source_file)",
            file_load_source=_file_source(source_file).model_copy(
                update={
                    "replay_call": prov.FileReplayCall(
                        kind="callable",
                        target="xarray.load_dataarray",
                        kwargs={},
                        selected_index=1,
                    )
                }
            ),
        ).append_replay_stage(prov.full_data())
    )
    assert win.slicer_area.reloadable
    with pytest.raises(IndexError, match="out of range"):
        win.slicer_area._fetch_for_provenance_reload()

    with pytest.raises(TypeError, match="script-only operations"):
        prov.file_load(
            start_label="Bad replay operation",
            seed_code="derived = xr.load_dataarray(source_file)",
            file_load_source=_file_source(source_file),
            replay_stages=[
                prov.ReplayStage(
                    source_kind="full_data",
                    operations=[
                        prov.ScriptCodeOperation(
                            label="Generated code",
                            code="derived = derived + 1",
                        )
                    ],
                )
            ],
        )

    win.set_provenance_spec(None)
    with pytest.raises(RuntimeError, match="cannot be reloaded"):
        win.slicer_area._fetch_reload_data()

    win.close()


def test_itool_reload_reports_failure_and_nonreloadable_noop(qtbot, monkeypatch):
    win = itool(xr.DataArray(np.arange(4.0), dims=("x",)), execute=False)
    qtbot.addWidget(win)

    assert not win.slicer_area.reloadable
    assert not win.slicer_area._reload()

    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        lambda _parent, title, text: errors.append((title, text)),
    )
    monkeypatch.setattr(
        type(win.slicer_area), "reloadable", property(lambda _area: True)
    )
    monkeypatch.setattr(
        win.slicer_area,
        "_fetch_reload_data",
        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    assert not win.slicer_area._reload()
    assert errors == [("Error", "An error occurred while reloading data.")]
    win.close()


def test_itool_save(qtbot, accept_dialog) -> None:
    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        filename = f"{tmp_dir_name}/data.h5"

        def _go_to_file(dialog: QtWidgets.QFileDialog):
            dialog.setDirectory(tmp_dir_name)
            dialog.selectFile(filename)
            focused = dialog.focusWidget()
            if isinstance(focused, QtWidgets.QLineEdit):
                focused.setText("data.h5")

        accept_dialog(lambda: win._export_file(native=False), pre_call=_go_to_file)
        xr.testing.assert_equal(data, xr.load_dataarray(filename, engine="h5netcdf"))

    win.close()


def test_itool_save_igor_wave(qtbot, accept_dialog) -> None:
    data = xr.DataArray(
        np.arange(25, dtype=np.float32).reshape((5, 5)), dims=["x", "y"], name="wave0"
    )
    expected = data.assign_coords(
        {dim: np.arange(data.sizes[dim], dtype=np.float64) for dim in data.dims}
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        filename = f"{tmp_dir_name}/data.ibw"

        def _go_to_file(dialog: QtWidgets.QFileDialog):
            dialog.selectNameFilter("Igor Binary Waves (*.ibw)")
            dialog.setDirectory(tmp_dir_name)
            dialog.selectFile(filename)
            focused = dialog.focusWidget()
            if isinstance(focused, QtWidgets.QLineEdit):
                focused.setText("data.ibw")

        accept_dialog(lambda: win._export_file(native=False), pre_call=_go_to_file)
        loaded = xr.load_dataarray(filename, engine="erlab-igor")
        xr.testing.assert_allclose(loaded, expected, atol=1e-6)

    win.close()


@pytest.mark.parametrize("use_dask", [True, False], ids=["dask", "no_dask"])
def test_itool_general(qtbot, move_and_compare_values, use_dask) -> None:
    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])
    if use_dask:
        data = data.chunk()

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
            "cursor_color_params": None,
        },
        "current_cursor": 1,
        "manual_limits": {},
        "splitter_sizes": list(old_state["splitter_sizes"]),
        "file_path": None,
        "load_func": None,
        "cursor_colors": ["#cccccc", "#ffff00"],
        "controls_visible": True,
        "plotitem_states": [
            {
                "roi_states": [],
                "vb_aspect_locked": False,
                "vb_autorange": (True, True),
                "vb_x_inverted": False,
                "vb_y_inverted": False,
            },
            {
                "roi_states": [],
                "vb_aspect_locked": False,
                "vb_autorange": (True, True),
                "vb_x_inverted": False,
                "vb_y_inverted": False,
            },
            {
                "roi_states": [],
                "vb_aspect_locked": False,
                "vb_autorange": (True, True),
                "vb_x_inverted": False,
                "vb_y_inverted": False,
            },
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

    # Toggle cursor visibility off
    win.slicer_area.toggle_cursor_act.setChecked(False)
    win.slicer_area.toggle_cursor_visibility()

    for plot_item in win.slicer_area.axes:
        for ax in plot_item.display_axis:
            for line_dict in plot_item.cursor_lines:
                assert not line_dict[ax].isVisible()
            for span_dict in plot_item.cursor_spans:
                assert not span_dict[ax].isVisible()

    # Try setting bins
    win.slicer_area.set_bin_all(axis=0, value=1, update=True)

    # Check again if still hidden
    for plot_item in win.slicer_area.axes:
        for ax in plot_item.display_axis:
            for line_dict in plot_item.cursor_lines:
                assert not line_dict[ax].isVisible()
            for span_dict in plot_item.cursor_spans:
                assert not span_dict[ax].isVisible()

    # Toggle cursor visibility on
    win.slicer_area.toggle_cursor_act.setChecked(True)
    win.slicer_area.toggle_cursor_visibility()

    # Check if cursors are visible again
    for plot_item in win.slicer_area.axes:
        for ax in plot_item.display_axis:
            for line_dict in plot_item.cursor_lines:
                assert line_dict[ax].isVisible()
            for cursor, span_dict in enumerate(plot_item.cursor_spans):
                span_region = win.array_slicer.span_bounds(cursor, ax)
                if span_region[0] == span_region[1]:
                    assert not span_dict[ax].isVisible()
                else:
                    assert span_dict[ax].isVisible()

    win.close()


def test_image_slicer_area_history_and_manual_limits(qtbot):
    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])
    win = ImageTool(data)
    qtbot.addWidget(win)
    area = win.slicer_area

    # undoable/redoable
    assert not area.undoable
    assert not area.redoable

    # history_suppressed context
    with area.history_suppressed():
        area._write_history = True

    # set_manual_limits/propagate_limit_change
    area.set_manual_limits({"x": [0, 4]})
    area.propagate_limit_change(area.main_image)
    # make_cursors with custom colors and error
    area.make_cursors(["#ff0000", "#00ff00"])

    # create colors with more cursors than COLORS
    for _ in range(10):
        area.color_for_cursor(_)
    # apply_func with None and with a function
    area.apply_func(None)
    area.apply_func(lambda d: d + 1)

    # reloadable/reload (simulate missing file/loader)
    area._file_path = None
    assert not area.reloadable

    # add_tool_window (no manager)
    w = QtWidgets.QWidget()
    area.add_tool_window(w)
    w.close()
    qtbot.wait_until(lambda: len(area._associated_tools) == 0)

    # state setter with partial state
    s = dict(area.state)
    s.pop("splitter_sizes", None)
    area.state = s
    win.close()


def test_itool_keeps_child_tool_registered_when_close_is_ignored(
    qtbot, monkeypatch
) -> None:
    win = itool(_TEST_DATA["2D"].copy(), execute=False)
    qtbot.addWidget(win)

    child = erlab.interactive.goldtool(_TEST_DATA["2D"].copy(), execute=False)
    monkeypatch.setattr(child, "_stop_server", lambda: False)

    win.slicer_area.add_tool_window(child)
    qtbot.wait_until(lambda: len(win.slicer_area._associated_tools) == 1, timeout=5000)

    assert child.close() is False
    assert len(win.slicer_area._associated_tools) == 1
    assert next(iter(win.slicer_area._associated_tools.values())) is child
    assert child.isVisible()

    monkeypatch.setattr(child, "_stop_server", lambda: True)
    assert child.close() is True
    qtbot.wait_until(lambda: len(win.slicer_area._associated_tools) == 0, timeout=5000)
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


def test_itool_child_tool_source_specs_and_non_source_updates(qtbot) -> None:
    data = _TEST_DATA["2D"]
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    selection_spec = win.slicer_area.images[0].make_tool_source_spec(
        transpose=True, squeeze=True
    )
    assert selection_spec.kind == "selection"
    assert [op.op for op in selection_spec.operations] == [
        "sort_coord_order",
        "transpose",
    ]

    win.slicer_area.open_in_meshtool()
    qtbot.wait_until(lambda: len(win.slicer_area._associated_tools) == 1, timeout=5000)
    child = next(iter(win.slicer_area._associated_tools.values()))
    assert child.source_spec == erlab.interactive.imagetool.provenance.full_data()
    assert child.source_state == "fresh"

    new_data = data.copy(deep=True)
    new_data.data = np.asarray(new_data.data) * 2
    win.slicer_area.set_data(new_data)
    assert child.source_state == "fresh"


def test_child_tool_copy_code_streamlines_noop_source_steps(qtbot) -> None:
    prov = erlab.interactive.imagetool.provenance
    win = itool(_TEST_DATA["2D"].copy(), execute=False)
    qtbot.addWidget(win)

    image = win.slicer_area.images[0]

    derivative = dtool(
        image.current_data.T,
        data_name=image.get_selection_code(),
        execute=False,
    )
    qtbot.addWidget(derivative)
    derivative.set_source_binding(image.make_tool_source_spec(transpose=True))

    derivative_code = derivative.copy_code()
    assert ".isel()" not in derivative_code
    assert "sort_coord_order" not in derivative_code
    assert ".transpose(" in derivative_code

    squeezed_child = dtool(
        _TEST_DATA["2D"].copy(),
        data_name="data",
        execute=False,
    )
    qtbot.addWidget(squeezed_child)
    squeezed_child.set_source_parent_fetcher(lambda: _TEST_DATA["2D"].copy())
    squeezed_child.set_source_binding(prov.selection(prov.SqueezeOperation()))

    squeezed_code = squeezed_child.copy_code()
    assert ".isel()" not in squeezed_code
    assert "sort_coord_order" not in squeezed_code
    assert ".squeeze()" not in squeezed_code

    squeezed_child.close()
    derivative.close()
    win.close()


def test_child_tool_copy_code_keeps_meaningful_parent_selection(qtbot) -> None:
    win = itool(_TEST_DATA["3D"].copy(), execute=False)
    qtbot.addWidget(win)

    win.slicer_area.set_value(1, 3.0)
    image = win.slicer_area.images[0]

    child = dtool(
        image.current_data.T,
        data_name=image.get_selection_code(),
        execute=False,
    )
    qtbot.addWidget(child)
    child.set_source_binding(image.make_tool_source_spec(transpose=True))

    code = child.copy_code()
    assert "sort_coord_order" not in code
    assert ".isel()" not in code
    assert ".qsel(" in code
    assert ".transpose(" in code

    child.close()
    win.close()


def test_itool_make_tool_source_spec_includes_alt_crop_indexers(
    qtbot, monkeypatch
) -> None:
    win = itool(_TEST_DATA["2D"].copy(), execute=False)
    qtbot.addWidget(win)

    image = win.slicer_area.images[0]
    monkeypatch.setattr(
        type(image),
        "_crop_indexers",
        property(lambda self: {"alpha": slice(1, 4)}),
    )

    monkeypatch.setattr(
        QtWidgets.QApplication,
        "queryKeyboardModifiers",
        staticmethod(lambda: QtCore.Qt.KeyboardModifier.AltModifier),
    )

    binding = image.make_tool_source_binding()
    spec = image.make_tool_source_spec()
    sel_kwargs = next(op.decoded_kwargs for op in spec.operations if op.op == "sel")

    assert binding.crop_sel_indexers == {"alpha": slice(1, 5)}
    assert sel_kwargs == {"alpha": slice(1, 4)}

    win.close()


def test_itool_make_tool_source_binding_uses_index_crop_for_nonuniform_dim(
    qtbot, monkeypatch
) -> None:
    data = xr.DataArray(
        np.zeros((4, 5), dtype=float),
        dims=("x", "y"),
        coords={"x": [0.0, 0.5, 2.0, 3.0], "y": np.arange(5.0)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    image = win.slicer_area.images[0]
    monkeypatch.setattr(
        type(image),
        "_crop_indexers",
        property(lambda self: {"x_idx": slice(None, None)}),
    )
    monkeypatch.setattr(
        QtWidgets.QApplication,
        "queryKeyboardModifiers",
        staticmethod(lambda: QtCore.Qt.KeyboardModifier.AltModifier),
    )

    binding = image.make_tool_source_binding()

    assert binding.crop_isel_indexers == {"x": slice(None, None)}

    win.close()


def test_itool_make_tool_source_binding_falls_back_to_dim_lookup(
    qtbot, monkeypatch
) -> None:
    win = itool(_TEST_DATA["3D"].copy(), execute=False)
    qtbot.addWidget(win)

    image = win.slicer_area.images[0]
    image.array_slicer._dim_indices = {}
    monkeypatch.setattr(
        type(image),
        "_crop_indexers",
        property(lambda self: {"alpha": slice(1, 4)}),
    )
    monkeypatch.setattr(
        QtWidgets.QApplication,
        "queryKeyboardModifiers",
        staticmethod(lambda: QtCore.Qt.KeyboardModifier.AltModifier),
    )

    binding = image.make_tool_source_binding()

    assert binding.selection_indexers["beta"] == 2
    assert binding.crop_sel_indexers == {"alpha": slice(1, 5)}

    win.close()


def test_parse_input() -> None:
    data_1d = xr.DataArray(np.arange(5), dims=["x"])
    parsed = _parse_input(xr.Dataset({"data1d": data_1d, "data0d": 1}))
    assert len(parsed) == 1
    xr.testing.assert_identical(parsed[0], data_1d.rename("data1d"))

    parsed_tree = _parse_input(
        xr.DataTree.from_dict({"dummy": xr.Dataset({"a": data_1d})})
    )
    assert len(parsed_tree) == 1
    xr.testing.assert_identical(parsed_tree[0], data_1d.rename("a"))

    with pytest.raises(ValueError, match="No valid data for ImageTool found"):
        _parse_input([])

    with pytest.raises(ValueError, match="No valid data for ImageTool found"):
        _parse_input(xr.Dataset({"scalar": xr.DataArray(1)}))

    with pytest.raises(ValueError, match="No valid data for ImageTool found"):
        _parse_input(
            xr.DataTree.from_dict({"dummy": xr.Dataset({"scalar": xr.DataArray(1)})})
        )


def test_itool_promotes_1d_input(qtbot) -> None:
    data = xr.DataArray(np.arange(5), dims=["x"], coords={"x": np.arange(5)})
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    assert win.slicer_area.data.dims == ("x", "stack_dim")
    assert win.slicer_area.data.shape == (5, 1)
    xr.testing.assert_identical(
        win.slicer_area.data.squeeze("stack_dim", drop=True), data
    )

    win.close()


def test_profile_open_in_new_window_from_1d_plot(qtbot) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)),
        dims=["x", "y"],
        coords={"x": np.arange(5), "y": np.arange(5)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    profile = win.slicer_area.profiles[0]
    initial_count = len(win.slicer_area._associated_tools_list)
    profile.open_in_new_window()

    qtbot.wait_until(
        lambda: len(win.slicer_area._associated_tools_list) > initial_count,
        timeout=2000,
    )
    new_tool = win.slicer_area._associated_tools_list[-1]
    assert isinstance(new_tool, ImageTool)
    xr.testing.assert_identical(
        new_tool.slicer_area.data.squeeze(drop=True),
        profile.current_data.squeeze(drop=True),
    )

    new_tool.close()
    win.close()


def test_itool_singleton_dimension(qtbot) -> None:
    data = xr.DataArray(
        np.arange(5, dtype=float).reshape((1, 5)),
        dims=["x", "y"],
        coords={"x": [0.0], "y": np.arange(5, dtype=float)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    assert win.slicer_area.data.shape == (1, 5)
    assert win.slicer_area.data.dims == ("x", "y")
    assert win.array_slicer._obj.shape == (1, 5)
    np.testing.assert_array_equal(win.array_slicer.incs, (1.0, 1.0))
    xr.testing.assert_identical(win.slicer_area.data, data)

    win.close()


def test_itool_squeezes_high_dim_input(qtbot) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 1, 5, 1, 1)),
        dims=["a", "b", "c", "d", "e"],
        coords={
            "a": np.arange(5),
            "b": [0],
            "c": np.arange(5),
            "d": [0],
            "e": [0],
        },
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    assert win.slicer_area.data.shape == (5, 5)
    assert win.slicer_area.data.dims == ("a", "c")
    np.testing.assert_array_equal(win.slicer_area.data.values, data.squeeze().values)

    win.close()


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
    assert len(wins) == 3

    for win in wins:
        qtbot.addWidget(win)

    with qtbot.waitExposed(wins[0]):
        wins[0].show()
    with qtbot.waitExposed(wins[1]):
        wins[1].show()
    with qtbot.waitExposed(wins[2]):
        wins[2].show()

    assert [w.windowTitle() for w in wins] == ["data1d", "a", "b"]

    data1d_out = wins[0].slicer_area.data.squeeze("stack_dim", drop=True)
    assert data1d_out.name == "data1d"
    assert data1d_out.dims == ("x",)
    np.testing.assert_array_equal(data1d_out.values, data.data1d.values)

    # Check if properly linked
    assert (
        wins[0].slicer_area._linking_proxy
        == wins[1].slicer_area._linking_proxy
        == wins[2].slicer_area._linking_proxy
    )
    assert wins[0].slicer_area.linked_slicers == weakref.WeakSet(
        [wins[1].slicer_area, wins[2].slicer_area]
    )

    for win in wins:
        win.slicer_area.unlink()
        win.close()


def _linked_pair(qtbot):
    data = xr.DataArray(np.arange(25).reshape((5, 5)).astype(float), dims=["x", "y"])
    wins = itool([data, data], execute=False, link=True)
    assert isinstance(wins, list)
    assert len(wins) == 2
    for win in wins:
        qtbot.addWidget(win)
    return typing.cast("list[ImageTool]", wins)


class _SceneDragEvent:
    def __init__(self, scene_pos: QtCore.QPointF) -> None:
        self._scene_pos = scene_pos

    def scenePos(self) -> QtCore.QPointF:
        return self._scene_pos


def _cursor_line_values(image, cursor: int) -> tuple[float, ...]:
    return tuple(line.value() for line in image.cursor_lines[cursor].values())


def test_linked_command_option_image_drag_refreshes_linked_cursors(qtbot) -> None:
    win0, win1 = _linked_pair(qtbot)
    win0.slicer_area.add_cursor()
    assert win0.slicer_area.n_cursors == 2
    assert win1.slicer_area.n_cursors == 2

    with qtbot.waitExposed(win0):
        win0.show()
    with qtbot.waitExposed(win1):
        win1.show()

    source_image = win0.slicer_area.main_image
    target_image = win1.slicer_area.main_image
    assert _cursor_line_values(target_image, 0) == (2.0, 2.0)
    assert _cursor_line_values(target_image, 1) == (2.0, 2.0)

    scene_pos = source_image.getViewBox().mapViewToScene(QtCore.QPointF(3.0, 1.0))
    source_image.process_drag(
        (
            _SceneDragEvent(scene_pos),
            QtCore.Qt.KeyboardModifier.ControlModifier
            | QtCore.Qt.KeyboardModifier.AltModifier,
        )
    )

    for cursor in range(2):
        qtbot.waitUntil(
            lambda cursor=cursor: np.allclose(
                _cursor_line_values(target_image, cursor), (3.0, 1.0)
            )
        )
        assert win0.array_slicer.get_indices(cursor) == [3, 1]
        assert win1.array_slicer.get_indices(cursor) == [3, 1]

    win0.slicer_area.unlink()
    win0.close()
    win1.close()


def test_linked_option_cursor_line_drag_refreshes_linked_cursors(
    qtbot, monkeypatch
) -> None:
    win0, win1 = _linked_pair(qtbot)
    win0.slicer_area.add_cursor()

    with qtbot.waitExposed(win0):
        win0.show()
    with qtbot.waitExposed(win1):
        win1.show()

    monkeypatch.setattr(
        QtWidgets.QApplication,
        "keyboardModifiers",
        staticmethod(lambda: QtCore.Qt.KeyboardModifier.AltModifier),
    )

    source_image = win0.slicer_area.main_image
    target_image = win1.slicer_area.main_image
    axis = source_image.display_axis[0]
    line = source_image.cursor_lines[0][axis]

    source_image.line_drag(line, 4.0, axis)

    for cursor in range(2):
        qtbot.waitUntil(
            lambda cursor=cursor: np.allclose(
                _cursor_line_values(target_image, cursor), (4.0, 2.0)
            )
        )
        assert win0.array_slicer.get_indices(cursor) == [4, 2]
        assert win1.array_slicer.get_indices(cursor) == [4, 2]

    win0.slicer_area.unlink()
    win0.close()
    win1.close()


def test_linked_cursor_undo_redo_propagates(qtbot) -> None:
    win0, win1 = _linked_pair(qtbot)

    win0.slicer_area.set_index(0, 4)
    assert win0.array_slicer.get_indices(0) == [4, 2]
    assert win1.array_slicer.get_indices(0) == [4, 2]

    win0.slicer_area.undo()
    assert win0.array_slicer.get_indices(0) == [2, 2]
    assert win1.array_slicer.get_indices(0) == [2, 2]

    win0.slicer_area.redo()
    assert win0.array_slicer.get_indices(0) == [4, 2]
    assert win1.array_slicer.get_indices(0) == [4, 2]

    win0.slicer_area.unlink()
    win0.close()
    win1.close()


def test_linked_cursor_entry_closes_before_unrecorded_linked_state(qtbot) -> None:
    win0, win1 = _linked_pair(qtbot)
    for win in (win0, win1):
        win.slicer_area.set_manual_limits({"x": [1.0, 3.0]})
        win.slicer_area.flush_history()

    win0.slicer_area.set_index(0, 4)
    assert win0.slicer_area._pending_history_entry is None
    assert win1.slicer_area._pending_history_entry is None

    win0.slicer_area.view_all()
    assert win0.slicer_area.manual_limits == {}
    assert win1.slicer_area.manual_limits == {}

    win0.slicer_area.undo()
    assert win0.array_slicer.get_indices(0) == [2, 2]
    assert win1.array_slicer.get_indices(0) == [2, 2]
    assert win0.slicer_area.manual_limits == {}
    assert win1.slicer_area.manual_limits == {}

    win0.slicer_area.unlink()
    win0.close()
    win1.close()


def test_linked_grouped_cursor_undo_propagates(qtbot) -> None:
    win0, win1 = _linked_pair(qtbot)
    control = win0.docks[0].widget().findChild(ItoolCrosshairControls)
    assert control is not None
    control.update_content()

    control.spin_idx[0].stepBy(1)
    control.spin_idx[0].stepBy(1)
    assert win0.array_slicer.get_indices(0) == [4, 2]
    assert win1.array_slicer.get_indices(0) == [4, 2]

    win0.slicer_area.undo()
    assert win0.array_slicer.get_indices(0) == [2, 2]
    assert win1.array_slicer.get_indices(0) == [2, 2]

    win0.slicer_area.unlink()
    win0.close()
    win1.close()


def test_linked_remove_current_cursor_undo_propagates(qtbot) -> None:
    win0, win1 = _linked_pair(qtbot)
    win0.slicer_area.add_cursor()
    win0.slicer_area.flush_history()
    win1.slicer_area.flush_history()

    win0.slicer_area.remove_current_cursor()
    assert win0.slicer_area.n_cursors == 1
    assert win1.slicer_area.n_cursors == 1
    assert (
        win0.slicer_area._prev_states[-1].transaction_id
        == win1.slicer_area._prev_states[-1].transaction_id
    )

    win0.slicer_area.undo()
    assert win0.slicer_area.n_cursors == 2
    assert win1.slicer_area.n_cursors == 2

    win0.slicer_area.unlink()
    win0.close()
    win1.close()


def test_linked_gamma_undo_redo_propagates(qtbot) -> None:
    win0, win1 = _linked_pair(qtbot)

    win0.slicer_area.set_colormap(gamma=1.4)
    assert win0.slicer_area.colormap_properties["gamma"] == pytest.approx(1.4)
    assert win1.slicer_area.colormap_properties["gamma"] == pytest.approx(1.4)

    win0.slicer_area.undo()
    assert win0.slicer_area.colormap_properties["gamma"] == pytest.approx(0.5)
    assert win1.slicer_area.colormap_properties["gamma"] == pytest.approx(0.5)

    win0.slicer_area.redo()
    assert win0.slicer_area.colormap_properties["gamma"] == pytest.approx(1.4)
    assert win1.slicer_area.colormap_properties["gamma"] == pytest.approx(1.4)

    win0.slicer_area.unlink()
    win0.close()
    win1.close()


def test_linked_grouped_gamma_undo_propagates(qtbot) -> None:
    win0, win1 = _linked_pair(qtbot)
    control = win0.docks[1].widget().findChild(ItoolColormapControls)
    assert control is not None

    control.gamma_widget.slider.sliderPressed.emit()
    control.gamma_widget.setValue(1.2)
    control.gamma_widget.setValue(1.4)
    assert win0.slicer_area.colormap_properties["gamma"] == pytest.approx(1.4)
    assert win1.slicer_area.colormap_properties["gamma"] == pytest.approx(1.4)

    win0.slicer_area.undo()
    assert win0.slicer_area.colormap_properties["gamma"] == pytest.approx(0.5)
    assert win1.slicer_area.colormap_properties["gamma"] == pytest.approx(0.5)

    win0.slicer_area.unlink()
    win0.close()
    win1.close()


def test_linked_compound_colormap_undo_propagates(qtbot) -> None:
    win0, win1 = _linked_pair(qtbot)

    win0.slicer_area.set_colormap(gamma=1.2, reverse=True)
    for win in (win0, win1):
        assert win.slicer_area.colormap_properties["gamma"] == pytest.approx(1.2)
        assert win.slicer_area.colormap_properties["reverse"] is True
        assert win.slicer_area.reverse_act.isChecked()
        assert len(win.slicer_area._prev_states) == 1
        assert win.slicer_area._prev_states[-1].changed_paths == frozenset(
            {("color", "gamma"), ("color", "reverse")}
        )

    assert (
        win0.slicer_area._prev_states[-1].transaction_id
        == win1.slicer_area._prev_states[-1].transaction_id
    )

    win0.slicer_area.undo()
    for win in (win0, win1):
        assert win.slicer_area.colormap_properties["gamma"] == pytest.approx(0.5)
        assert win.slicer_area.colormap_properties["reverse"] is False
        assert not win.slicer_area.reverse_act.isChecked()

    win0.slicer_area.unlink()
    win0.close()
    win1.close()


def test_linked_colormap_action_undo_propagates(qtbot) -> None:
    win0, win1 = _linked_pair(qtbot)

    win0.slicer_area.reverse_act.trigger()
    assert win0.slicer_area.colormap_properties["reverse"] is True
    assert win1.slicer_area.colormap_properties["reverse"] is True
    assert win1.slicer_area.reverse_act.isChecked()

    win0.slicer_area.undo()
    assert win0.slicer_area.colormap_properties["reverse"] is False
    assert win1.slicer_area.colormap_properties["reverse"] is False
    assert not win0.slicer_area.reverse_act.isChecked()
    assert not win1.slicer_area.reverse_act.isChecked()

    win0.slicer_area.unlink()
    win0.close()
    win1.close()


def test_linked_controls_visibility_undo_redo_stays_local(qtbot) -> None:
    win0, win1 = _linked_pair(qtbot)

    win0.mnb.action_dict["toggleControlsAct"].trigger()
    assert not win0.controls_visible
    assert win1.controls_visible

    win0.slicer_area.undo()
    assert win0.controls_visible
    assert win1.controls_visible
    assert not win1.slicer_area.redoable

    win0.slicer_area.redo()
    assert not win0.controls_visible
    assert win1.controls_visible
    assert not win1.slicer_area.redoable

    win0.slicer_area.unlink()
    win0.close()
    win1.close()


def test_linked_undo_handles_local_entry_before_linked_entry(qtbot) -> None:
    win0, win1 = _linked_pair(qtbot)

    win0.slicer_area.set_index(0, 4)
    win0.mnb.action_dict["toggleControlsAct"].trigger()

    win0.slicer_area.undo()
    assert win0.controls_visible
    assert win1.controls_visible
    assert win0.array_slicer.get_indices(0) == [4, 2]
    assert win1.array_slicer.get_indices(0) == [4, 2]

    win0.slicer_area.undo()
    assert win0.array_slicer.get_indices(0) == [2, 2]
    assert win1.array_slicer.get_indices(0) == [2, 2]

    win0.slicer_area.unlink()
    win0.close()
    win1.close()


def test_linked_history_group_splits_linked_then_local_action(qtbot) -> None:
    win0, win1 = _linked_pair(qtbot)

    win0.slicer_area.begin_history_group()
    win0.slicer_area.set_index(0, 4)
    win0.mnb.action_dict["toggleControlsAct"].trigger()
    win0.slicer_area.end_history_group()

    assert [entry.transaction_id for entry in win0.slicer_area._prev_states] == [
        win1.slicer_area._prev_states[-1].transaction_id,
        None,
    ]

    win0.slicer_area.undo()
    assert win0.controls_visible
    assert win0.array_slicer.get_indices(0) == [4, 2]
    assert win1.array_slicer.get_indices(0) == [4, 2]

    win0.slicer_area.undo()
    assert win0.array_slicer.get_indices(0) == [2, 2]
    assert win1.array_slicer.get_indices(0) == [2, 2]

    win0.slicer_area.unlink()
    win0.close()
    win1.close()


def test_linked_history_group_splits_local_then_linked_action(qtbot) -> None:
    win0, win1 = _linked_pair(qtbot)

    win0.slicer_area.begin_history_group()
    win0.mnb.action_dict["toggleControlsAct"].trigger()
    win0.slicer_area.set_index(0, 4)
    win0.slicer_area.end_history_group()

    assert [entry.transaction_id for entry in win0.slicer_area._prev_states] == [
        None,
        win1.slicer_area._prev_states[-1].transaction_id,
    ]

    win0.slicer_area.undo()
    assert not win0.controls_visible
    assert win0.array_slicer.get_indices(0) == [2, 2]
    assert win1.array_slicer.get_indices(0) == [2, 2]

    win0.slicer_area.undo()
    assert win0.controls_visible
    assert win1.controls_visible

    win0.slicer_area.unlink()
    win0.close()
    win1.close()


def test_linked_history_group_resets_transaction_after_local_split(qtbot) -> None:
    win0, win1 = _linked_pair(qtbot)

    win0.slicer_area.begin_history_group()
    win0.slicer_area.set_index(0, 4)
    win0.mnb.action_dict["toggleControlsAct"].trigger()
    win0.slicer_area.set_index(1, 4)
    win0.slicer_area.end_history_group()

    source_transaction_ids = [
        entry.transaction_id for entry in win0.slicer_area._prev_states
    ]
    assert source_transaction_ids[0] is not None
    assert source_transaction_ids[1] is None
    assert source_transaction_ids[2] is not None
    assert source_transaction_ids[2] != source_transaction_ids[0]
    assert [entry.transaction_id for entry in win1.slicer_area._prev_states] == [
        source_transaction_ids[0],
        source_transaction_ids[2],
    ]

    win0.slicer_area.undo()
    assert win0.array_slicer.get_indices(0) == [4, 2]
    assert win1.array_slicer.get_indices(0) == [4, 2]
    assert not win0.controls_visible

    win0.slicer_area.undo()
    assert win0.array_slicer.get_indices(0) == [4, 2]
    assert win1.array_slicer.get_indices(0) == [4, 2]
    assert win0.controls_visible

    win0.slicer_area.undo()
    assert win0.array_slicer.get_indices(0) == [2, 2]
    assert win1.array_slicer.get_indices(0) == [2, 2]

    win0.slicer_area.unlink()
    win0.close()
    win1.close()


def test_linked_undo_preserves_non_conflicting_peer_local_change(qtbot) -> None:
    win0, win1 = _linked_pair(qtbot)

    win0.slicer_area.set_index(0, 4)
    win1.mnb.action_dict["toggleControlsAct"].trigger()

    win0.slicer_area.undo()
    assert win0.array_slicer.get_indices(0) == [2, 2]
    assert win1.array_slicer.get_indices(0) == [2, 2]
    assert win1.controls_visible is False

    win0.slicer_area.redo()
    assert win0.array_slicer.get_indices(0) == [4, 2]
    assert win1.array_slicer.get_indices(0) == [4, 2]
    assert win1.controls_visible is False

    win0.slicer_area.unlink()
    win0.close()
    win1.close()


def test_linked_undo_preserves_non_conflicting_peer_cursor_change(qtbot) -> None:
    win0, win1 = _linked_pair(qtbot)
    win0.slicer_area.add_cursor()
    win0.slicer_area.flush_history()
    win1.slicer_area.flush_history()

    win0.slicer_area.set_index(0, 4, cursor=0)
    with win1.slicer_area.link_sync_suppressed():
        win1.slicer_area.set_index(1, 4, cursor=1)

    win0.slicer_area.undo()
    assert win0.array_slicer.get_indices(0) == [2, 2]
    assert win1.array_slicer.get_indices(0) == [2, 2]
    assert win1.array_slicer.get_indices(1) == [2, 4]

    win1.slicer_area.undo()
    assert win1.array_slicer.get_indices(0) == [2, 2]
    assert win1.array_slicer.get_indices(1) == [2, 2]

    win0.slicer_area.unlink()
    win0.close()
    win1.close()


def test_linked_undo_skips_peer_with_conflicting_local_change(qtbot) -> None:
    win0, win1 = _linked_pair(qtbot)

    win0.slicer_area.set_index(0, 4)
    with win1.slicer_area.link_sync_suppressed():
        win1.slicer_area.set_index(0, 3)

    win0.slicer_area.undo()
    assert win0.array_slicer.get_indices(0) == [2, 2]
    assert win1.array_slicer.get_indices(0) == [3, 2]

    win1.slicer_area.undo()
    assert win1.array_slicer.get_indices(0) == [4, 2]

    win0.slicer_area.unlink()
    win0.close()
    win1.close()


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


def test_apply_func_preserves_source_data(qtbot) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)),
        dims=["x", "y"],
        coords={"x": np.arange(5), "y": np.arange(5)},
    )

    win = ImageTool(data)
    qtbot.addWidget(win)
    with qtbot.waitExposed(win):
        win.show()
        win.activateWindow()

    win.slicer_area.apply_func(lambda darr: darr + 1)
    xarray.testing.assert_identical(win.slicer_area._data, data)
    xarray.testing.assert_identical(win.slicer_area.data, data + 1)

    win.slicer_area.apply_func(None)
    xarray.testing.assert_identical(win.slicer_area.data, data)
    win.close()


def test_apply_func_preserves_dask_backed_preview(qtbot) -> None:
    da = pytest.importorskip("dask.array")
    from dask.callbacks import Callback

    values = np.arange(25, dtype=np.float32).reshape((5, 5))
    data = xr.DataArray(
        da.from_array(values, chunks=(2, 5)),
        dims=["x", "y"],
        coords={"x": np.arange(5), "y": np.arange(5)},
    )

    win = ImageTool(data, auto_compute=False)
    qtbot.addWidget(win)

    computed_keys: list[object] = []
    with Callback(pretask=lambda key, _dsk, _state: computed_keys.append(key)):
        win.slicer_area.apply_func(lambda darr: darr + 1, update=False)

    assert computed_keys == []
    assert win.slicer_area.data_chunked
    assert win.slicer_area.data.chunks is not None
    assert win.slicer_area._obj_shares_data_values is False
    xarray.testing.assert_identical(
        win.slicer_area.data.compute(), (data + 1).compute()
    )

    win.slicer_area.apply_func(None, update=False)
    assert win.slicer_area.data_chunked
    assert win.slicer_area.data.chunks == data.chunks
    assert win.slicer_area._obj_shares_data_values is True
    win.close()


def test_apply_func_accepts_transposed_dask_preview_and_updates(qtbot) -> None:
    da = pytest.importorskip("dask.array")

    values = np.arange(20, dtype=np.float32).reshape((4, 5))
    data = xr.DataArray(
        da.from_array(values, chunks=(2, 5)),
        dims=["x", "y"],
        coords={"x": np.arange(4), "y": np.arange(5)},
    )
    win = ImageTool(data, auto_compute=False)
    qtbot.addWidget(win)

    win.slicer_area.apply_func(lambda darr: (darr + 1).transpose("y", "x"))

    assert win.slicer_area.data.dims == data.dims
    assert win.slicer_area.data.chunks is not None
    xarray.testing.assert_identical(
        win.slicer_area.data.compute(), (data + 1).compute()
    )
    win.close()


def test_apply_func_rejects_mismatched_dask_preview(qtbot) -> None:
    da = pytest.importorskip("dask.array")

    values = np.arange(20, dtype=np.float32).reshape((4, 5))
    data = xr.DataArray(
        da.from_array(values, chunks=(2, 5)),
        dims=["x", "y"],
        coords={"x": np.arange(4), "y": np.arange(5)},
    )
    win = ImageTool(data, auto_compute=False)
    qtbot.addWidget(win)

    with pytest.raises(ValueError, match="dimensions do not match"):
        win.slicer_area.apply_func(lambda darr: darr.expand_dims("z"), update=False)
    with pytest.raises(ValueError, match="dimensions do not match"):
        win.slicer_area.apply_func(lambda darr: darr.rename({"x": "z"}), update=False)
    with pytest.raises(ValueError, match="shape does not match"):
        win.slicer_area.apply_func(
            lambda darr: darr.isel(x=slice(1, None)), update=False
        )
    win.close()


def test_eager_preview_from_dask_source_updates_readout(qtbot) -> None:
    da = pytest.importorskip("dask.array")

    source_values = np.arange(20, dtype=np.float32).reshape((4, 5))
    preview_values = source_values + 100.0
    data = xr.DataArray(
        da.from_array(source_values, chunks=(2, 5)),
        dims=["x", "y"],
        coords={"x": np.arange(4), "y": np.arange(5)},
    )
    preview = xr.DataArray(preview_values, dims=data.dims, coords=data.coords)

    win = ImageTool(data, auto_compute=False)
    qtbot.addWidget(win)
    control = win.docks[0].widget().findChild(ItoolCrosshairControls)
    assert control is not None

    win.slicer_area.apply_func(lambda _darr: preview)
    assert win.slicer_area.data_chunked
    assert win.slicer_area.data.chunks is None

    control.update_content()
    win.slicer_area.set_index(0, 3)
    assert control.spin_dat.value() == preview_values[3, 2]
    win.close()


def test_set_source_item_restores_preview_before_source_write(qtbot) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(np.float32),
        dims=["x", "y"],
        coords={"x": np.arange(5), "y": np.arange(5)},
    )

    win = ImageTool(data)
    qtbot.addWidget(win)
    with qtbot.waitExposed(win):
        win.show()
        win.activateWindow()

    win.slicer_area.apply_func(lambda darr: darr + 10)
    win.slicer_area._set_source_item((0, 0), -5.0)

    assert float(data.values[0, 0]) == 0.0
    assert float(win.slicer_area._data.values[0, 0]) == -5.0
    assert float(win.slicer_area.data.values[0, 0]) == -5.0
    assert win.slicer_area._applied_func is None
    win.close()


def test_owned_values_copy_handles_dask_and_array_fallback() -> None:
    da = pytest.importorskip("dask.array")

    darr = xr.DataArray(
        da.from_array(np.arange(6, dtype=np.float32).reshape(2, 3), chunks=(1, 3)),
        dims=("x", "y"),
    )
    copied = ImageSlicerArea._owned_values_copy(darr)
    np.testing.assert_array_equal(copied, np.arange(6, dtype=np.float32).reshape(2, 3))

    fake = types.SimpleNamespace(data=(1, 2, 3))
    fallback = ImageSlicerArea._owned_values_copy(fake)
    np.testing.assert_array_equal(fallback, np.array([1, 2, 3]))


def test_set_data_rad2deg_converts_angle_coords(qtbot) -> None:
    data = xr.DataArray(
        np.arange(15, dtype=np.float32).reshape((3, 5)),
        dims=("phi", "eV"),
        coords={
            "phi": np.deg2rad(np.array([0.0, 30.0, 60.0], dtype=np.float32)),
            "eV": np.arange(5, dtype=np.float32),
        },
    )

    win = ImageTool(data)
    qtbot.addWidget(win)
    with qtbot.waitExposed(win):
        win.show()
        win.activateWindow()

    win.slicer_area.set_data(data, rad2deg=True)

    np.testing.assert_allclose(win.slicer_area._data["phi"].values, [0.0, 30.0, 60.0])
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
    with pytest.raises(ValueError, match=r"^Data shape does not match.*"):
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
        dialog.launch_mode_combo.setCurrentText("Replace Current")

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
        with qtbot.wait_signal(dialog._sigCodeCopied):
            dialog.copy_button.click()
        dialog.reshape_check.setChecked(True)
        dialog.launch_mode_combo.setCurrentText("Replace Current")

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
    qtbot.wait_until(
        lambda: not win.slicer_area.main_image.is_guidelines_visible, timeout=1000
    )

    win.close()


def test_itool_guidelines_start_at_active_cursor(qtbot) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["x", "y"],
        coords={"x": np.arange(5, dtype=float), "y": np.arange(5, dtype=float)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    area = win.slicer_area
    plot_item = area.main_image

    area.add_cursor()
    area.set_value(0, 1.0)
    area.set_value(1, 3.0)
    plot_item.set_guidelines(3)

    _assert_guideline_state(plot_item, count=3, angle=0.0, offset=(1.0, 3.0))

    win.close()


def test_itool_guidelines_follow_active_cursor(qtbot) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["x", "y"],
        coords={"x": np.arange(5, dtype=float), "y": np.arange(5, dtype=float)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    area = win.slicer_area
    plot_item = area.main_image
    plot_item.set_guidelines(3)
    plot_item.set_guidelines_follow_cursor(True)

    target = plot_item._guidelines_items[-1]
    assert not target.isVisible()
    assert not target.movable
    _assert_guideline_state(
        plot_item, count=3, angle=0.0, offset=(2.0, 2.0), follow_cursor=True
    )
    assert all(item.pos() == target.pos() for item in plot_item._guidelines_items[:-1])

    area.set_value(0, 4.0)
    area.set_value(1, 1.0)
    _assert_guideline_state(
        plot_item, count=3, angle=0.0, offset=(4.0, 1.0), follow_cursor=True
    )
    assert all(item.pos() == target.pos() for item in plot_item._guidelines_items[:-1])

    area.add_cursor()
    area.set_value(0, 0.0)
    area.set_value(1, 3.0)
    _assert_guideline_state(
        plot_item, count=3, angle=0.0, offset=(0.0, 3.0), follow_cursor=True
    )
    assert all(item.pos() == target.pos() for item in plot_item._guidelines_items[:-1])

    plot_item.set_guidelines_follow_cursor(False)
    assert target.isVisible()
    assert target.movable
    _assert_guideline_state(
        plot_item, count=3, angle=0.0, offset=(0.0, 3.0), follow_cursor=False
    )

    area.set_value(0, 2.0)
    area.set_value(1, 4.0)
    _assert_guideline_state(
        plot_item, count=3, angle=0.0, offset=(0.0, 3.0), follow_cursor=False
    )

    win.close()


def test_itool_guideline_state_roundtrip(qtbot) -> None:
    data = xr.DataArray(np.arange(25).reshape((5, 5)).astype(float), dims=["x", "y"])
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    plot_item = win.slicer_area.main_image
    state_without_guidelines = copy.deepcopy(plot_item._serializable_state)

    plot_item.set_guidelines(3)
    plot_item._guidelines_items[0].setAngle(60.0)

    state_with_guidelines = copy.deepcopy(plot_item._serializable_state)
    _assert_guideline_state(
        plot_item, count=3, angle=-30.0, offset=(2.0, 2.0), follow_cursor=True
    )

    plot_item._serializable_state = state_without_guidelines
    assert not plot_item.is_guidelines_visible

    plot_item._serializable_state = state_with_guidelines
    _assert_guideline_state(
        plot_item, count=3, angle=-30.0, offset=(2.0, 2.0), follow_cursor=True
    )

    win.close()


def test_itool_guideline_state_dataset_roundtrip(qtbot) -> None:
    data = xr.DataArray(np.arange(25).reshape((5, 5)).astype(float), dims=["x", "y"])
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    plot_item = win.slicer_area.main_image
    plot_item.set_guidelines(3)
    plot_item._guidelines_items[0].setAngle(60.0)

    restored = ImageTool.from_dataset(win.to_dataset())
    qtbot.addWidget(restored)

    _assert_guideline_state(
        restored.slicer_area.main_image,
        count=3,
        angle=-30.0,
        offset=(2.0, 2.0),
        follow_cursor=True,
    )

    restored.close()
    win.close()


def test_itool_controls_visibility_menu_history_and_dataset_roundtrip(qtbot) -> None:
    data = xr.DataArray(np.arange(25).reshape((5, 5)).astype(float), dims=["x", "y"])
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    action = win.mnb.action_dict["toggleControlsAct"]
    assert action.isCheckable()
    assert action.isChecked()
    assert win.controls_visible
    assert win.slicer_area.state["controls_visible"] is True

    action.trigger()
    assert not action.isChecked()
    assert not win.controls_visible
    assert win.slicer_area.state["controls_visible"] is False
    assert win.slicer_area.undoable

    win.slicer_area.undo()
    assert action.isChecked()
    assert win.controls_visible
    assert win.slicer_area.state["controls_visible"] is True

    win.slicer_area.redo()
    assert not action.isChecked()
    assert not win.controls_visible
    assert win.slicer_area.state["controls_visible"] is False

    restored = ImageTool.from_dataset(win.to_dataset())
    qtbot.addWidget(restored)

    assert not restored.controls_visible
    assert not restored.mnb.action_dict["toggleControlsAct"].isChecked()
    assert restored.slicer_area.state["controls_visible"] is False

    restored.close()
    win.close()


def test_itool_controls_visibility_legacy_state_defaults_visible(qtbot) -> None:
    data = xr.DataArray(np.arange(25).reshape((5, 5)).astype(float), dims=["x", "y"])
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    win.controls_visible = False

    ds = win.to_dataset()
    state = json.loads(ds.attrs["itool_state"])
    state.pop("controls_visible")
    ds.attrs["itool_state"] = json.dumps(state)

    restored = ImageTool.from_dataset(ds)
    qtbot.addWidget(restored)

    assert restored.controls_visible
    assert restored.mnb.action_dict["toggleControlsAct"].isChecked()
    assert restored.slicer_area.state["controls_visible"] is True

    restored.close()
    win.close()


@pytest.mark.parametrize(
    ("coord_dims", "coord_values"),
    [
        (("x",), np.linspace(10.0, 20.0, 5)),
        (("x", "y"), np.arange(25.0).reshape((5, 5))),
    ],
)
def test_itool_file_roundtrip_preserves_spaced_associated_coord(
    qtbot,
    tmp_path,
    coord_dims: tuple[str, ...],
    coord_values: np.ndarray,
) -> None:
    data = xr.DataArray(
        np.arange(25.0).reshape((5, 5)),
        dims=["x", "y"],
        coords={
            "x": np.arange(5.0),
            "y": np.arange(5.0),
            "Fake Motor": (coord_dims, coord_values),
        },
    )
    win = ImageTool(data)
    qtbot.addWidget(win)

    fname = tmp_path / "spaced-coord.h5"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        win.to_file(fname)

    assert not any("space in its name" in str(item.message) for item in caught)
    restored = ImageTool.from_file(fname)
    qtbot.addWidget(restored)

    assert "Fake Motor" in restored.slicer_area._data.coords
    xarray.testing.assert_equal(
        restored.slicer_area._data.coords["Fake Motor"],
        data.coords["Fake Motor"],
    )

    restored.close()
    win.close()


def test_itool_from_file_recovers_legacy_spaced_associated_coord(
    qtbot, tmp_path
) -> None:
    data = xr.DataArray(
        np.arange(25.0).reshape((5, 5)),
        dims=["x", "y"],
        coords={
            "x": np.arange(5.0),
            "y": np.arange(5.0),
            "Fake Motor": ("x", np.linspace(10.0, 20.0, 5)),
        },
    )
    win = ImageTool(data)
    qtbot.addWidget(win)
    legacy = win.to_dataset().reset_coords("Fake Motor")

    fname = tmp_path / "legacy-spaced-coord.h5"
    legacy.to_netcdf(fname, engine="h5netcdf", invalid_netcdf=True)
    restored = ImageTool.from_file(fname)
    qtbot.addWidget(restored)

    assert "Fake Motor" in restored.slicer_area._data.coords
    xarray.testing.assert_equal(
        restored.slicer_area._data.coords["Fake Motor"],
        data.coords["Fake Motor"],
    )

    restored.close()
    win.close()


def test_itool_open_in_ktool_uses_active_cursor_seed(qtbot, monkeypatch) -> None:
    win = itool(_TEST_DATA["3D"].copy(), execute=False)
    qtbot.addWidget(win)

    captured: dict[str, object] = {}
    tool = QtWidgets.QWidget()

    def fake_ktool(data, **kwargs):
        captured["data"] = data
        captured["kwargs"] = kwargs
        return tool

    monkeypatch.setattr(erlab.interactive, "ktool", fake_ktool)

    win.slicer_area.set_value(0, 1.0)
    win.slicer_area.set_value(2, 4.0)
    win.slicer_area.open_in_ktool()

    assert captured["data"] is win.slicer_area.data
    assert captured["kwargs"]["initial_normal_emission"] == (1.0, 4.0)
    assert captured["kwargs"]["initial_delta"] is None

    win.close()


def test_itool_open_in_ktool_uses_scalar_beta_for_angle_energy_cut(
    qtbot, monkeypatch
) -> None:
    win = itool(_TEST_DATA["3D"].isel(beta=3).copy(), execute=False)
    qtbot.addWidget(win)

    captured: dict[str, object] = {}
    tool = QtWidgets.QWidget()

    def fake_ktool(data, **kwargs):
        captured["data"] = data
        captured["kwargs"] = kwargs
        return tool

    monkeypatch.setattr(erlab.interactive, "ktool", fake_ktool)

    assert win.slicer_area.ktool_act.isEnabled()
    win.slicer_area.set_value(0, 2.0)
    win.slicer_area.open_in_ktool()

    assert captured["data"] is win.slicer_area.data
    assert captured["kwargs"]["initial_normal_emission"] == (2.0, 3.0)
    assert captured["kwargs"]["initial_delta"] is None

    win.close()


def test_itool_open_in_ktool_ignores_nonscalar_beta_coord(qtbot, monkeypatch) -> None:
    data = _TEST_DATA["3D"].isel(beta=0).drop_vars("beta").copy()
    data = data.assign_coords(beta=("alpha", np.arange(data.sizes["alpha"])))
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    captured: dict[str, object] = {}
    tool = QtWidgets.QWidget()

    def fake_ktool(data, **kwargs):
        captured["data"] = data
        captured["kwargs"] = kwargs
        return tool

    monkeypatch.setattr(erlab.interactive, "ktool", fake_ktool)

    win.slicer_area.set_value(0, 2.0)
    win.slicer_area.open_in_ktool()

    assert captured["data"] is win.slicer_area.data
    assert captured["kwargs"]["initial_normal_emission"] is None
    assert captured["kwargs"]["initial_delta"] is None

    win.close()


def test_itool_open_in_ktool_sets_full_data_source_binding(qtbot, monkeypatch) -> None:
    win = itool(_TEST_DATA["3D"].copy(), execute=False)
    qtbot.addWidget(win)

    child = dtool(_TEST_DATA["2D"].copy(), execute=False)
    monkeypatch.setattr(erlab.interactive, "ktool", lambda *args, **kwargs: child)

    win.slicer_area.open_in_ktool()

    assert child.source_spec == erlab.interactive.imagetool.provenance.full_data()
    assert child.source_state == "fresh"

    win.close()


def test_itool_open_in_ktool_uses_guideline_seed_on_alpha_beta_plane(
    qtbot, monkeypatch
) -> None:
    win = itool(_TEST_DATA["3D"].qsel(eV=2.0).copy(), execute=False)
    qtbot.addWidget(win)

    captured: dict[str, object] = {}
    tool = QtWidgets.QWidget()

    def fake_ktool(data, **kwargs):
        captured["data"] = data
        captured["kwargs"] = kwargs
        return tool

    monkeypatch.setattr(erlab.interactive, "ktool", fake_ktool)

    plot_item = win.slicer_area.main_image
    plot_item.set_guidelines(3)
    plot_item._guidelines_items[0].setAngle(60.0)
    plot_item._guidelines_items[-1].setPos((1.0, 4.0))
    win.slicer_area.open_in_ktool()

    assert captured["kwargs"]["initial_normal_emission"] == (1.0, 4.0)
    assert captured["kwargs"]["initial_delta"] == pytest.approx(30.0)

    win.close()


def test_itool_open_in_ktool_ignores_non_alpha_beta_guidelines(
    qtbot, monkeypatch
) -> None:
    win = itool(_TEST_DATA["3D"].copy(), execute=False)
    qtbot.addWidget(win)

    captured: dict[str, object] = {}
    tool = QtWidgets.QWidget()

    def fake_ktool(data, **kwargs):
        captured["data"] = data
        captured["kwargs"] = kwargs
        return tool

    monkeypatch.setattr(erlab.interactive, "ktool", fake_ktool)

    win.slicer_area.set_value(0, 4.0)
    win.slicer_area.set_value(2, 1.0)

    plot_item = win.slicer_area.main_image
    plot_item.set_guidelines(3)
    plot_item._guidelines_items[0].setAngle(60.0)
    plot_item._guidelines_items[-1].setPos((0.0, 3.0))

    win.slicer_area.open_in_ktool()

    assert captured["kwargs"]["initial_normal_emission"] == (4.0, 1.0)
    assert captured["kwargs"]["initial_delta"] is None

    win.close()


def test_itool_open_in_ftool_sets_squeezed_source_binding(qtbot, monkeypatch) -> None:
    win = itool(_TEST_DATA["2D"].copy(), execute=False)
    qtbot.addWidget(win)

    child = dtool(_TEST_DATA["2D"].copy(), execute=False)
    monkeypatch.setattr(erlab.interactive, "ftool", lambda *args, **kwargs: child)

    image = win.slicer_area.images[0]
    image.open_in_ftool()

    assert child.source_spec == image.make_tool_source_spec(squeeze=True)
    assert child.source_binding == image.make_tool_source_binding(squeeze=True)
    assert child.source_state == "fresh"

    win.close()


def test_profile_open_in_ftool_omits_noop_squeeze_source_binding(
    qtbot, monkeypatch
) -> None:
    prov = erlab.interactive.imagetool.provenance
    win = itool(_TEST_DATA["2D"].copy(), execute=False)
    qtbot.addWidget(win)

    child = dtool(_TEST_DATA["2D"].copy(), execute=False)
    monkeypatch.setattr(erlab.interactive, "ftool", lambda *args, **kwargs: child)

    profile = win.slicer_area.profiles[0]
    profile.open_in_ftool()

    assert child.source_spec is not None
    assert not any(
        isinstance(operation, prov.SqueezeOperation)
        for operation in child.source_spec.operations
    )

    child.close()
    win.close()


def test_itool_guideline_undo_redo(qtbot) -> None:
    data = xr.DataArray(np.arange(25).reshape((5, 5)).astype(float), dims=["x", "y"])
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    area = win.slicer_area
    plot_item = area.main_image

    plot_item.set_guidelines(3)
    plot_item.set_guidelines_follow_cursor(False)
    assert area.undoable

    area.undo()
    _assert_guideline_state(plot_item, count=3, angle=0.0, offset=(2.0, 2.0))

    area.redo()
    initial_offset = tuple(plot_item._guideline_offset)
    _assert_guideline_state(
        plot_item, count=3, angle=0.0, offset=initial_offset, follow_cursor=False
    )

    plot_item._guidelines_items[0]._sigAngleDragStarted.emit()
    plot_item._guidelines_items[0].setAngle(60.0)
    _assert_guideline_state(
        plot_item,
        count=3,
        angle=-30.0,
        offset=initial_offset,
        follow_cursor=False,
    )

    area.undo()
    _assert_guideline_state(
        plot_item, count=3, angle=0.0, offset=initial_offset, follow_cursor=False
    )

    area.redo()
    _assert_guideline_state(
        plot_item,
        count=3,
        angle=-30.0,
        offset=initial_offset,
        follow_cursor=False,
    )

    plot_item._guidelines_items[-1].sigPositionDragStarted.emit()
    plot_item._guidelines_items[-1].setPos((3.0, 3.1))
    _assert_guideline_state(
        plot_item,
        count=3,
        angle=-30.0,
        offset=(3.0, 3.1),
        follow_cursor=False,
    )

    area.undo()
    _assert_guideline_state(
        plot_item,
        count=3,
        angle=-30.0,
        offset=initial_offset,
        follow_cursor=False,
    )

    area.redo()
    _assert_guideline_state(
        plot_item,
        count=3,
        angle=-30.0,
        offset=(3.0, 3.1),
        follow_cursor=False,
    )

    win.close()


def test_itool_guideline_transpose_undo_redo(qtbot) -> None:
    data = xr.DataArray(np.arange(25).reshape((5, 5)).astype(float), dims=["x", "y"])
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    plot_item = win.slicer_area.main_image
    plot_item.set_guidelines(3)
    plot_item.set_guidelines_follow_cursor(False)
    plot_item._guidelines_items[0].setAngle(60.0)
    plot_item._guidelines_items[-1].setPos((3.0, 3.1))

    win.slicer_area.swap_axes(0, 1)
    qtbot.wait_until(lambda: not plot_item.is_guidelines_visible, timeout=1000)

    win.slicer_area.undo()
    qtbot.wait_until(lambda: plot_item.is_guidelines_visible, timeout=1000)
    _assert_guideline_state(
        plot_item,
        count=3,
        angle=-30.0,
        offset=(3.0, 3.1),
        follow_cursor=False,
    )

    win.slicer_area.redo()
    qtbot.wait_until(lambda: not plot_item.is_guidelines_visible, timeout=1000)

    win.close()


def test_itool_rotate_center_accepts_out_of_bounds_values(qtbot, accept_dialog) -> None:
    data = xr.DataArray(np.arange(25).reshape((5, 5)).astype(float), dims=["x", "y"])
    center = (-1.5, 9.25)

    win = itool(data, execute=False)
    qtbot.addWidget(win)

    def _set_dialog_params(dialog: RotationDialog) -> None:
        dialog.angle_spin.setValue(30.0)
        dialog.center_spins[0].setValue(center[0])
        dialog.center_spins[1].setValue(center[1])
        assert dialog.center_spins[0].value() == center[0]
        assert dialog.center_spins[1].value() == center[1]
        dialog.reshape_check.setChecked(True)
        dialog.launch_mode_combo.setCurrentText("Replace Current")

    accept_dialog(win.mnb._rotate, pre_call=_set_dialog_params)

    xarray.testing.assert_allclose(
        win.slicer_area._data,
        erlab.analysis.transform.rotate(data, angle=30.0, center=center, reshape=True),
    )
    win.close()


def set_vb_range(
    vb: pg.ViewBox, x_range: tuple[float, float], y_range: tuple[float, float]
) -> None:
    vb.setRange(xRange=x_range, yRange=y_range)
    vb.sigRangeChangedManually.emit(vb.state["mouseEnabled"][:])


def test_itool_normalize_to_view(qtbot) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["x", "y"],
        coords={"x": np.arange(5), "y": np.arange(5)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    with qtbot.wait_exposed(win):
        win.show()
        win.activateWindow()

    # Change limits
    set_vb_range(
        win.slicer_area.main_image.getViewBox(), x_range=(1, 4), y_range=(0, 3)
    )
    assert win.slicer_area.manual_limits == {"x": [1.0, 4.0], "y": [0.0, 3.0]}
    slice_dict = {"x": slice(1.0, 4.0), "y": slice(0.0, 3.0)}
    assert win.slicer_area.make_slice_dict() == slice_dict

    # Adjust colors
    win.slicer_area.main_image.normalize_to_current_view()

    cropped = data.sel(**slice_dict)
    xr.testing.assert_identical(
        win.slicer_area.main_image._current_data_cropped, cropped
    )
    assert erlab.utils.array.minmax_darr(cropped) == (5.0, 23.0)

    mn, mx = win.slicer_area.levels
    np.testing.assert_allclose(mn, 5.0)
    np.testing.assert_allclose(mx, 23.0)

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
        with qtbot.wait_signal(dialog._sigCodeCopied):
            dialog.copy_button.click()
        dialog.launch_mode_combo.setCurrentText("Replace Current")

    accept_dialog(win.mnb._crop_to_view, pre_call=_set_dialog_params)
    xarray.testing.assert_allclose(
        win.slicer_area._data, data.sel(x=slice(1.0, 4.0), y=slice(0.0, 3.0))
    )
    xarray.testing.assert_allclose(
        _exec_data_fragment(data, pyperclip.paste()),
        data.sel(x=slice(1.0, 4.0), y=slice(0.0, 3.0)),
    )

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
        with qtbot.wait_signal(dialog._sigCodeCopied):
            dialog.copy_button.click()
        dialog.launch_mode_combo.setCurrentText("Replace Current")

    accept_dialog(win.mnb._crop, pre_call=_set_dialog_params)
    xarray.testing.assert_allclose(
        win.slicer_area._data, data.sel(x=slice(1.0, 4.0), y=slice(0.0, 3.0))
    )
    xarray.testing.assert_allclose(
        _exec_data_fragment(data, pyperclip.paste()),
        data.sel(x=slice(1.0, 4.0), y=slice(0.0, 3.0)),
    )

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
        with qtbot.wait_signal(dialog._sigCodeCopied):
            dialog.copy_button.click()
        dialog.launch_mode_combo.setCurrentText("Replace Current")

    accept_dialog(win.mnb._crop, pre_call=_set_dialog_params)
    xarray.testing.assert_allclose(
        win.slicer_area._data, data.sel(x=slice(2.0, 4.0), y=slice(0.0, 3.0))
    )
    xarray.testing.assert_allclose(
        _exec_data_fragment(data, pyperclip.paste()),
        data.sel(x=slice(2.0, 4.0)),
    )

    win.close()


def test_itool_roi_lifecycle(qtbot) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["x", "y"],
        coords={"x": np.arange(5), "y": np.arange(5)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    plot_item = win.slicer_area.main_image
    add_roi_action = next(
        act
        for act in plot_item.vb.menu.actions()
        if act.objectName() == "itool_add_polygon_roi_action"
    )
    add_roi_action.trigger()

    assert len(plot_item._roi_list) == 1
    roi = plot_item._roi_list[0]

    roi_menu = roi.getMenu()
    assert {
        "itool_edit_roi_action",
        "itool_slice_along_roi_path_action",
        "itool_mask_data_with_roi_action",
    }.issubset({act.objectName() for act in roi_menu.actions()})

    state = copy.deepcopy(plot_item._serializable_state)
    assert "roi_states" in state
    saved_points = [
        (float(pt[0]), float(pt[1]))
        for pt in plot_item._roi_list[0].getState()["points"]
    ]
    assert state["roi_states"][0]["points"] == saved_points

    plot_item.clear_rois()
    assert not plot_item._roi_list

    plot_item._serializable_state = state
    assert len(plot_item._roi_list) == 1
    restored_points = [
        (float(pt[0]), float(pt[1]))
        for pt in plot_item._roi_list[0].getState()["points"]
    ]
    assert restored_points == state["roi_states"][0]["points"]

    plot_item.add_roi()
    assert len(plot_item._roi_list) == 2

    win.slicer_area.sigShapeChanged.emit()
    assert not plot_item._roi_list

    plot_item.add_roi()
    assert plot_item._roi_list
    win.slicer_area.adjust_layout()
    assert not plot_item._roi_list

    win.close()


def test_itool_roi_dialogs(qtbot, monkeypatch) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["x", "y"],
        coords={
            "x": np.array([0.1, 0.4, 0.55, 0.65, 0.95]),
            "y": np.linspace(-1.0, 1.0, 5),
        },
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    plot_item = win.slicer_area.main_image
    plot_item.add_roi()
    roi = plot_item._roi_list[0]

    array_slicer = win.slicer_area.array_slicer
    x_axis, y_axis = plot_item.display_axis
    roi_points = [
        (
            float(array_slicer.coords_uniform[x_axis][idx]),
            float(array_slicer.coords_uniform[y_axis][idy]),
        )
        for idx, idy in [(0, 0), (2, 2), (4, 3)]
    ]
    roi.setPoints(roi_points, closed=True)
    assert roi.closed

    vertices = roi._get_vertices()

    path_dialog = ROIPathDialog(roi)
    qtbot.addWidget(path_dialog)
    path_dialog._dim_name_line.setText("trace_dim")

    path_calls: dict[str, object] = {}

    def fake_slice(data_arg, **kwargs):
        path_calls["data"] = data_arg
        path_calls["params"] = kwargs
        return data_arg

    fake_slice.__name__ = "slice_along_path"

    monkeypatch.setattr(
        erlab.analysis.interpolate, "slice_along_path", fake_slice, raising=False
    )

    result = path_dialog.process_data(win.slicer_area.data)
    assert result is win.slicer_area.data

    expected_vertices = {
        dim: [*list(values), values[0]] for dim, values in vertices.items()
    }

    assert path_calls["data"] is win.slicer_area.data
    assert path_calls["params"]["dim_name"] == "trace_dim"
    assert np.isclose(path_calls["params"]["step_size"], path_dialog._step_spin.value())
    assert path_calls["params"]["vertices"] == expected_vertices
    for dim in expected_vertices:
        assert path_calls["params"]["vertices"][dim][-1] == expected_vertices[dim][0]

    path_code = path_dialog.make_code()
    assert "era.interpolate.slice_along_path" in path_code
    assert 'dim_name="trace_dim"' in path_code

    mask_dialog = ROIMaskDialog(roi)
    qtbot.addWidget(mask_dialog)
    mask_dialog._invert_check.setChecked(True)
    mask_dialog._drop_check.setChecked(True)

    mask_calls: dict[str, object] = {}

    def fake_mask(data_arg, **kwargs):
        mask_calls["data"] = data_arg
        mask_calls["params"] = kwargs
        return data_arg

    fake_mask.__name__ = "mask_with_polygon"

    monkeypatch.setattr(
        erlab.analysis.mask, "mask_with_polygon", fake_mask, raising=False
    )

    mask_dialog.process_data(win.slicer_area.data)

    expected_vertices_array = np.column_stack(tuple(vertices.values()))
    np.testing.assert_allclose(
        mask_calls["params"]["vertices"], expected_vertices_array
    )
    assert mask_calls["params"]["dims"] == tuple(vertices.keys())
    assert mask_calls["params"]["invert"] is True
    assert mask_calls["params"]["drop"] is True

    mask_code = mask_dialog.make_code()
    assert "era.mask.mask_with_polygon" in mask_code
    assert "vertices=np.array(" in mask_code
    assert "invert=True" in mask_code
    assert "drop=True" in mask_code

    win.close()


def test_itool_roi_edit_dialog_updates_points(qtbot) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["x", "y"],
        coords={"x": np.arange(5), "y": np.arange(5)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    plot_item = win.slicer_area.main_image
    plot_item.add_roi()
    roi = plot_item._roi_list[0]
    roi.setPoints([(0.0, 0.0), (2.0, 1.0)], closed=False)

    dialog = _PolyROIEditDialog(roi)
    qtbot.addWidget(dialog)
    dialog.show()

    assert dialog.table.rowCount() == 2

    dialog._add_row()
    assert dialog.table.rowCount() == 3

    dialog.table.setCurrentCell(0, 0)
    dialog._delete_row()
    assert dialog.table.rowCount() == 2
    assert not dialog.del_row_btn.isEnabled()

    dialog._add_row()
    dialog._add_row()
    assert dialog.table.rowCount() == 4

    new_points = [(0.5, 1.5), (1.5, 0.5), (2.5, 1.0), (3.5, 1.5)]
    for row, (x_val, y_val) in enumerate(new_points):
        x_item = dialog.table.item(row, 0)
        y_item = dialog.table.item(row, 1)
        assert x_item is not None
        assert y_item is not None
        x_item.setText(np.format_float_positional(x_val))
        y_item.setText(np.format_float_positional(y_val))

    dialog.closed_check.setChecked(True)
    dialog.accept()

    assert roi.closed
    assert [
        (float(pt[0]), float(pt[1])) for pt in roi.getState()["points"]
    ] == new_points

    win.close()


def test_itool_roi_edit_dialog_invalid_values(qtbot, monkeypatch) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["x", "y"],
        coords={"x": np.arange(5), "y": np.arange(5)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    plot_item = win.slicer_area.main_image
    plot_item.add_roi()
    roi = plot_item._roi_list[0]
    original_points = [(0.0, 0.0), (1.0, 1.0)]
    roi.setPoints(original_points, closed=False)

    dialog = _PolyROIEditDialog(roi)
    qtbot.addWidget(dialog)
    dialog.show()

    dialog.table.item(0, 0).setText("invalid")

    message_calls: dict[str, str] = {}

    def fake_critical(*args, **kwargs):
        message_calls["title"] = args[1] if len(args) > 1 else ""
        message_calls["text"] = args[2] if len(args) > 2 else ""
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", fake_critical)

    dialog.accept()
    assert message_calls
    assert [
        (float(pt[0]), float(pt[1])) for pt in roi.getState()["points"]
    ] == original_points

    dialog.table.item(0, 0).setText("3.0")
    dialog.accept()
    assert [(float(pt[0]), float(pt[1])) for pt in roi.getState()["points"]] == [
        (3.0, 0.0),
        (1.0, 1.0),
    ]

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
        with qtbot.wait_signal(dialog._sigCodeCopied):
            dialog.copy_button.click()
        dialog.launch_mode_combo.setCurrentText("Replace Current")

    accept_dialog(win.mnb._average, pre_call=_set_dialog_params)
    xarray.testing.assert_identical(
        win.slicer_area._data.rename(None), data.qsel.average("x")
    )

    xarray.testing.assert_identical(
        _exec_data_fragment(data, pyperclip.paste()),
        data.qsel.average("x"),
    )
    assert win.provenance_spec is not None
    display_code = win.provenance_spec.display_code()
    assert display_code is not None
    display_namespace = _exec_generated_code(
        display_code,
        {"data": data.copy(deep=True)},
    )
    derived = display_namespace["derived"]
    assert isinstance(derived, xr.DataArray)
    xarray.testing.assert_identical(derived, data.qsel.average("x"))
    win.close()


def test_average_source_spec_restores_nonuniform_dims_after_refresh(qtbot) -> None:
    data = xr.DataArray(
        np.arange(20).reshape((5, 4)).astype(float),
        dims=["x", "y"],
        coords={"x": [0.0, 0.2, 0.8, 1.4, 2.0], "y": np.arange(4)},
        name="scan",
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    assert win.slicer_area.data.dims == ("x_idx", "y")
    dialog = AverageDialog(win.slicer_area)
    qtbot.addWidget(dialog)
    dialog.dim_checks["y"].setChecked(True)

    spec = dialog.source_spec("scan_avg")
    expected = erlab.interactive.imagetool.slicer.restore_nonuniform_dims(
        dialog.process_data(win.slicer_area.data)
    ).rename("scan_avg")
    refreshed = spec.apply(win.slicer_area.data)

    assert spec.kind == "full_data"
    assert [op.op for op in spec.operations] == [
        "average",
        "restore_nonuniform_dims",
        "rename",
    ]
    assert refreshed.dims == ("x",)
    xarray.testing.assert_identical(refreshed, expected)

    display_code = spec.display_code(parent_data=win.slicer_area.data)
    assert display_code is not None
    assert "restore_nonuniform_dims" in display_code
    display_namespace = _exec_generated_code(
        display_code,
        {"data": win.slicer_area.data.copy(deep=True)},
    )
    derived = display_namespace["derived"]
    assert isinstance(derived, xr.DataArray)
    xarray.testing.assert_identical(derived.rename(None), expected.rename(None))

    dialog.close()
    win.close()


def test_itool_average_marks_incompatible_child_tools_unavailable(
    qtbot, accept_dialog
) -> None:
    data = _TEST_DATA["2D"].copy()
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    win.slicer_area.open_in_meshtool()
    qtbot.wait_until(lambda: len(win.slicer_area._associated_tools) == 1, timeout=5000)
    child = next(iter(win.slicer_area._associated_tools.values()))
    assert child.source_state == "fresh"

    def _set_dialog_params(dialog: AverageDialog) -> None:
        dialog.dim_checks["alpha"].setChecked(True)
        dialog.launch_mode_combo.setCurrentText("Replace Current")

    with qtbot.wait_signal(win.slicer_area.sigSourceDataReplaced):
        accept_dialog(win.mnb._average, pre_call=_set_dialog_params)

    xarray.testing.assert_identical(
        win.slicer_area._data.rename(None), data.qsel.average("alpha")
    )
    qtbot.wait_until(lambda: child.source_state == "unavailable", timeout=5000)

    win.close()


def test_average_dialog_make_code_preserves_nonstring_dim(qtbot) -> None:
    data = xr.DataArray(
        np.arange(6).reshape((2, 3)).astype(float),
        dims=["k-space", "y"],
        coords={"k-space": np.arange(2), "y": np.arange(3)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    dialog = AverageDialog(win.slicer_area)
    qtbot.addWidget(dialog)
    dialog.dim_checks["k-space"].setChecked(True)

    xarray.testing.assert_identical(
        _exec_data_fragment(data, dialog.make_code()),
        data.qsel.average("k-space"),
    )

    dialog.close()
    win.close()


def _selection_4d_data() -> xr.DataArray:
    return xr.DataArray(
        np.arange(3 * 4 * 5 * 6).reshape((3, 4, 5, 6)).astype(float),
        dims=["alpha", "eV", "beta", "hv"],
        coords={
            "alpha": np.arange(3, dtype=float),
            "eV": np.arange(4, dtype=float),
            "beta": np.arange(5, dtype=float),
            "hv": np.linspace(20.0, 70.0, 6),
        },
        name="scan",
    )


def test_selection_dialog_seeds_4d_cursor_slice(qtbot) -> None:
    data = _selection_4d_data()
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    win.slicer_area.set_index(3, 4)

    dialog = SelectionDialog(win.slicer_area)

    assert [row.use_check.isChecked() for row in dialog.rows] == [
        False,
        False,
        False,
        True,
    ]
    expected = data.qsel(hv=60.0)
    xarray.testing.assert_identical(dialog.process_data(dialog.public_data), expected)
    xarray.testing.assert_identical(
        _exec_data_fragment(data, dialog.make_code()), expected
    )
    assert dialog.buttonBox.button(
        QtWidgets.QDialogButtonBox.StandardButton.Ok
    ).isEnabled()

    dialog.close()
    win.close()


def test_selection_dialog_dimension_checkbox_label_toggles_row(qtbot) -> None:
    data = _selection_4d_data()
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    dialog = SelectionDialog(win.slicer_area)
    _clear_selection_dialog(dialog)
    row = dialog.rows[0]
    ok_button = dialog.buttonBox.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)

    assert not row.use_check.isChecked()
    assert dialog.source_operations() == []
    assert not ok_button.isEnabled()

    dialog.show()
    qtbot.waitExposed(dialog)
    option = QtWidgets.QStyleOptionButton()
    row.use_check.initStyleOption(option)
    indicator_rect = row.use_check.style().subElementRect(
        QtWidgets.QStyle.SubElement.SE_CheckBoxIndicator,
        option,
        row.use_check,
    )
    contents_rect = row.use_check.style().subElementRect(
        QtWidgets.QStyle.SubElement.SE_CheckBoxContents,
        option,
        row.use_check,
    )
    click_pos = contents_rect.center()
    assert contents_rect.isValid()
    assert not indicator_rect.contains(click_pos)

    qtbot.mouseClick(
        row.use_check,
        QtCore.Qt.MouseButton.LeftButton,
        pos=click_pos,
    )

    assert row.use_check.isChecked()
    assert dialog.source_operations()
    assert ok_button.isEnabled()

    win.close()


def test_selection_dialog_accept_replaces_current_data(qtbot) -> None:
    data = _selection_4d_data()
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    win.slicer_area.set_index(3, 2)

    dialog = SelectionDialog(win.slicer_area)
    dialog.launch_mode_combo.setCurrentText("Replace Current")

    dialog.accept()

    xarray.testing.assert_identical(
        win.slicer_area._data.rename(None), data.qsel(hv=40.0).rename(None)
    )
    assert win.provenance_spec is not None
    assert [op.op for op in win.provenance_spec.operations] == ["qsel", "rename"]

    win.close()


def test_selection_dialog_isel_range_executes_code(qtbot) -> None:
    data = _selection_4d_data()
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    dialog = SelectionDialog(win.slicer_area)
    _clear_selection_dialog(dialog)

    row = dialog.rows[0]
    row.use_check.setChecked(True)
    _set_combo_data(row.method_combo, "isel")
    _set_combo_data(row.kind_combo, "range")
    row.index_start_spin.setValue(1)
    row.index_stop_spin.setValue(3)

    expected = data.isel(alpha=slice(1, 3))
    xarray.testing.assert_identical(dialog.process_data(dialog.public_data), expected)
    xarray.testing.assert_identical(
        _exec_data_fragment(data, dialog.make_code()), expected
    )

    dialog.close()
    win.close()


def test_selection_dialog_formats_non_identifier_dim_as_mapping(qtbot) -> None:
    data = xr.DataArray(
        np.arange(3 * 4 * 5).reshape((3, 4, 5)).astype(float),
        dims=["Fake Motor", "eV", "beta"],
        coords={
            "Fake Motor": np.arange(3, dtype=float),
            "eV": np.arange(4, dtype=float),
            "beta": np.arange(5, dtype=float),
        },
        name="scan",
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    dialog = SelectionDialog(win.slicer_area)
    _clear_selection_dialog(dialog)

    row = dialog.rows[0]
    row.use_check.setChecked(True)
    _set_combo_data(row.method_combo, "isel")
    row.index_start_spin.setValue(1)

    expected = data.isel({"Fake Motor": 1})
    assert dialog.make_code() == '.isel({"Fake Motor": 1})'
    xarray.testing.assert_identical(
        _exec_data_fragment(data, dialog.make_code()), expected
    )

    dialog.close()
    win.close()


def test_selection_dialog_sel_range_executes_code(qtbot) -> None:
    data = _selection_4d_data()
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    dialog = SelectionDialog(win.slicer_area)
    _clear_selection_dialog(dialog)

    row = dialog.rows[1]
    row.use_check.setChecked(True)
    _set_combo_data(row.method_combo, "sel")
    _set_combo_data(row.kind_combo, "range")
    row.value_start_spin.setValue(1.0)
    row.value_stop_spin.setValue(3.0)

    expected = data.sel(eV=slice(1.0, 3.0))
    xarray.testing.assert_identical(dialog.process_data(dialog.public_data), expected)
    xarray.testing.assert_identical(
        _exec_data_fragment(data, dialog.make_code()), expected
    )

    dialog.close()
    win.close()


def test_selection_dialog_qsel_range_ignores_cursor_width(qtbot) -> None:
    data = _selection_4d_data()
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    win.array_slicer.set_bin(0, axis=2, value=3, update=True)
    dialog = SelectionDialog(win.slicer_area)
    for dialog_row in dialog.rows:
        dialog_row.use_check.setChecked(False)

    row = dialog.rows[2]
    row.use_check.setChecked(True)
    _set_combo_data(row.method_combo, "qsel")
    _set_combo_data(row.kind_combo, "range")
    row.value_start_spin.setValue(1.0)
    row.value_stop_spin.setValue(3.0)

    expected = data.qsel(beta=slice(1.0, 3.0))
    assert row.width_check.isChecked()
    assert not row.width_widget.isEnabled()
    assert row.qsel_width_indexer() is None
    xarray.testing.assert_identical(dialog.process_data(dialog.public_data), expected)
    xarray.testing.assert_identical(
        _exec_data_fragment(data, dialog.make_code()), expected
    )
    assert dialog.buttonBox.button(
        QtWidgets.QDialogButtonBox.StandardButton.Ok
    ).isEnabled()

    dialog.close()
    win.close()


def test_selection_dialog_qsel_width_executes_code(qtbot) -> None:
    data = _selection_4d_data()
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    dialog = SelectionDialog(win.slicer_area)
    _clear_selection_dialog(dialog)

    row = dialog.rows[2]
    row.use_check.setChecked(True)
    _set_combo_data(row.method_combo, "qsel")
    _set_combo_data(row.kind_combo, "point")
    row.value_start_spin.setValue(2.0)
    row.width_check.setChecked(True)
    row.width_spin.setValue(2.0)

    expected = data.qsel(beta=2.0, beta_width=2.0)
    xarray.testing.assert_identical(dialog.process_data(dialog.public_data), expected)
    xarray.testing.assert_identical(
        _exec_data_fragment(data, dialog.make_code()), expected
    )

    dialog.close()
    win.close()


def test_selection_dialog_uses_public_nonuniform_dimensions(qtbot) -> None:
    data = xr.DataArray(
        np.arange(3 * 4 * 5).reshape((3, 4, 5)).astype(float),
        dims=["alpha", "eV", "beta"],
        coords={
            "alpha": np.array([0.1, 0.4, 0.8]),
            "eV": np.arange(4, dtype=float),
            "beta": np.arange(5, dtype=float),
        },
        name="scan",
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    dialog = SelectionDialog(win.slicer_area)
    _clear_selection_dialog(dialog)

    assert win.slicer_area.data.dims == ("alpha_idx", "eV", "beta")
    row = dialog.rows[0]
    row.use_check.setChecked(True)
    _set_combo_data(row.method_combo, "qsel")
    row.value_start_spin.setValue(0.4)

    expected = data.qsel(alpha=0.4)
    selected = dialog.process_data(dialog.public_data)
    assert selected.dims == ("eV", "beta")
    xarray.testing.assert_identical(selected, expected)
    xarray.testing.assert_identical(
        _exec_data_fragment(data, dialog.make_code()), expected
    )
    spec = dialog.source_spec("scan_sel")
    assert spec.kind == "public_data"
    xarray.testing.assert_identical(
        spec.apply(win.slicer_area.data), expected.rename("scan_sel")
    )

    dialog.close()
    win.close()


def test_selection_dialog_menu_action_key(qtbot) -> None:
    win = itool(_TEST_DATA["2D"].copy(), execute=False)
    qtbot.addWidget(win)

    action = win.mnb.action_dict["selectDataAct"]
    assert action.isEnabled()

    win.close()


def test_selection_dialog_rejects_empty_selection(qtbot) -> None:
    win = itool(_TEST_DATA["2D"].copy(), execute=False)
    qtbot.addWidget(win)
    dialog = SelectionDialog(win.slicer_area)
    _clear_selection_dialog(dialog)

    assert dialog.source_operations() == []
    assert not dialog.buttonBox.button(
        QtWidgets.QDialogButtonBox.StandardButton.Ok
    ).isEnabled()

    dialog.close()
    win.close()


def test_average_dialog_rejects_empty_dimension_selection(qtbot, monkeypatch) -> None:
    data = xr.DataArray(
        np.arange(6).reshape((2, 3)).astype(float),
        dims=["x", "y"],
        coords={"x": np.arange(2), "y": np.arange(3)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda _parent, title, text: warnings.append((title, text)),
    )

    dialog = AverageDialog(win.slicer_area)
    dialog.accept()

    assert warnings == [
        ("No Dimensions Selected", "You need to select at least one dimension.")
    ]
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Rejected
    xarray.testing.assert_identical(win.slicer_area._data, data)

    dialog.close()
    win.close()


def test_transform_dialog_restores_filter_after_processing_error(
    qtbot, monkeypatch
) -> None:
    data = xr.DataArray(
        np.arange(6).reshape((2, 3)).astype(float),
        dims=["x", "y"],
        coords={"x": np.arange(2), "y": np.arange(3)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    def _filter(darr: xr.DataArray) -> xr.DataArray:
        return darr + 1

    win.slicer_area.apply_func(_filter)

    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        lambda _parent, title, text, **_kwargs: errors.append((title, text)),
    )

    dialog = AverageDialog(win.slicer_area)
    dialog.dim_checks["x"].setChecked(True)
    monkeypatch.setattr(
        dialog,
        "process_data",
        lambda _data: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    dialog.accept()

    assert errors == [("Error", "An error occurred while processing data.")]
    assert win.slicer_area._applied_func is _filter

    dialog.close()
    win.close()


def test_transform_replace_composes_after_script_active_name(qtbot) -> None:
    source = xr.DataArray(
        np.arange(6).reshape((2, 3)).astype(float),
        dims=["x", "y"],
        coords={"x": np.arange(2), "y": np.arange(3)},
        name="source",
    )
    displayed = (source + 1).rename("result")
    win = itool(displayed, execute=False)
    qtbot.addWidget(win)

    win.set_provenance_spec(
        erlab.interactive.imagetool.provenance.script(
            erlab.interactive.imagetool.provenance.ScriptCodeOperation(
                label="Compute intermediate result",
                code="result = data + 1",
            ),
            start_label="Start from current tool input data",
            active_name="result",
        )
    )

    dialog = AverageDialog(win.slicer_area)
    dialog.dim_checks["x"].setChecked(True)
    dialog.launch_mode_combo.setCurrentText("Replace Current")
    dialog.accept()

    assert win.provenance_spec is not None
    code = win.provenance_spec.derivation_code()
    assert code == (
        'result = data + 1\nderived = result\nderived = derived.qsel.average("x")'
    )
    namespace = _exec_generated_code(code, {"data": source.copy(deep=True)})
    derived = namespace["derived"]
    assert isinstance(derived, xr.DataArray)
    xarray.testing.assert_identical(
        derived.rename(None), displayed.qsel.average("x").rename(None)
    )

    dialog.close()
    win.close()


def test_average_dialog_launch_modes_for_standalone(qtbot, monkeypatch) -> None:
    data = xr.DataArray(
        np.arange(60).reshape((3, 4, 5)).astype(float),
        dims=["x", "y", "z"],
        coords={"x": np.arange(3), "y": np.arange(4), "z": np.arange(5)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    dialog = AverageDialog(win.slicer_area)
    qtbot.addWidget(dialog)
    mode_labels = [
        dialog.launch_mode_combo.itemText(i)
        for i in range(dialog.launch_mode_combo.count())
    ]
    assert mode_labels == ["Replace Current", "Open Top-Level Window"]
    assert dialog.launch_mode == "detach"
    launched: list[xr.DataArray] = []
    monkeypatch.setattr(
        erlab.interactive,
        "itool",
        lambda *args, **kwargs: launched.append(kwargs["data"]) or None,
    )
    dialog.dim_checks["x"].setChecked(True)
    dialog.accept()
    xarray.testing.assert_identical(launched[0].rename(None), data.qsel.average("x"))

    dialog_replace = AverageDialog(win.slicer_area)
    qtbot.addWidget(dialog_replace)
    dialog_replace.dim_checks["y"].setChecked(True)
    dialog_replace.launch_mode_combo.setCurrentText("Replace Current")
    dialog_replace.accept()
    xarray.testing.assert_identical(
        win.slicer_area._data.rename(None), data.qsel.average("y")
    )

    win.close()


def test_itool_interpolate(qtbot, accept_dialog) -> None:
    data = xr.DataArray(
        np.arange(6).reshape((3, 2)).astype(float),
        dims=["x", "y"],
        coords={"x": np.arange(3, dtype=float), "y": [10.0, 20.0]},
        name="scan",
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    assert "interpolateAct" in win.mnb.action_dict

    target = np.linspace(0.0, 2.0, 5)

    def _set_dialog_params(dialog: InterpolationDialog) -> None:
        dialog.dim_combo.setCurrentText("x")
        dialog.coord_widget.count_spin.setValue(target.size)
        with qtbot.wait_signal(dialog._sigCodeCopied):
            dialog.copy_button.click()
        dialog.launch_mode_combo.setCurrentText("Replace Current")

    accept_dialog(win.mnb._interpolate, pre_call=_set_dialog_params)

    expected = data.interp({"x": target}, method="linear")
    xarray.testing.assert_identical(
        win.slicer_area._data.rename(None), expected.rename(None)
    )
    assert win.slicer_area._data.name == "scan_interp_x"

    xarray.testing.assert_identical(
        _exec_data_fragment(data, pyperclip.paste()), expected
    )
    assert win.provenance_spec is not None
    display_code = win.provenance_spec.display_code()
    assert display_code is not None
    namespace = _exec_generated_code(display_code, {"data": data.copy(deep=True)})
    derived = namespace["derived"]
    assert isinstance(derived, xr.DataArray)
    xarray.testing.assert_identical(derived.rename(None), expected.rename(None))

    win.close()


def test_itool_interpolate_nearest(qtbot, accept_dialog) -> None:
    data = xr.DataArray(
        np.arange(6).reshape((3, 2)).astype(float),
        dims=["x", "y"],
        coords={"x": np.arange(3, dtype=float), "y": [10.0, 20.0]},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    target = np.array([0.2, 1.8])

    def _set_dialog_params(dialog: InterpolationDialog) -> None:
        dialog.method_combo.setCurrentText("nearest")
        dialog.coord_widget.count_spin.setValue(target.size)
        for row, value in enumerate(target):
            dialog.coord_widget.table.item(row, 0).setText(str(value))
        dialog.launch_mode_combo.setCurrentText("Replace Current")

    accept_dialog(win.mnb._interpolate, pre_call=_set_dialog_params)
    xarray.testing.assert_identical(
        win.slicer_area._data.rename(None),
        data.interp({"x": target}, method="nearest").rename(None),
    )

    win.close()


def test_itool_interpolate_nonuniform_public_dims(qtbot, accept_dialog) -> None:
    data = xr.DataArray(
        np.arange(6).reshape((3, 2)).astype(float),
        dims=["x", "y"],
        coords={"x": np.array([0.0, 0.4, 1.0]), "y": [10.0, 20.0]},
        name="scan",
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    assert win.slicer_area.data.dims == ("x_idx", "y")
    target = np.linspace(0.0, 1.0, 4)

    def _set_dialog_params(dialog: InterpolationDialog) -> None:
        assert dialog.dim_combo.currentText() == "x"
        dialog.coord_widget.count_spin.setValue(target.size)
        dialog.launch_mode_combo.setCurrentText("Replace Current")

    accept_dialog(win.mnb._interpolate, pre_call=_set_dialog_params)

    expected = data.interp({"x": target}, method="linear")
    xarray.testing.assert_identical(
        win.slicer_area._data.rename(None), expected.rename(None)
    )
    assert win.provenance_spec is not None
    display_code = win.provenance_spec.display_code()
    assert display_code is not None
    assert ".interp(" in display_code
    assert "x_idx" not in display_code
    namespace = _exec_generated_code(display_code, {"data": data.copy(deep=True)})
    derived = namespace["derived"]
    assert isinstance(derived, xr.DataArray)
    xarray.testing.assert_identical(derived.rename(None), expected.rename(None))

    win.close()


def test_interpolation_dialog_rejects_invalid_target_values(qtbot, monkeypatch) -> None:
    data = xr.DataArray(
        np.arange(6).reshape((3, 2)).astype(float),
        dims=["x", "y"],
        coords={"x": np.arange(3, dtype=float), "y": [10.0, 20.0]},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    dialog = InterpolationDialog(win.slicer_area)
    qtbot.addWidget(dialog)
    dialog.launch_mode_combo.setCurrentText("Replace Current")
    dialog.coord_widget.table.item(0, 0).setText("bad")

    warnings: list[tuple[str, str]] = []

    def _record_warning(_parent, title, message, *args, **kwargs):
        warnings.append((title, message))
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _record_warning)
    dialog.accept()

    assert warnings == [("Invalid Target Coordinates", "Invalid value in row 0: bad")]
    xarray.testing.assert_identical(win.slicer_area._data.rename(None), data)

    dialog.close()
    win.close()


def test_interpolation_dialog_edge_validation_paths(qtbot, monkeypatch) -> None:
    data = xr.DataArray(
        np.arange(6).reshape((3, 2)).astype(float),
        dims=["x", "y"],
        coords={"x": np.array([0.0, 0.0, 1.0]), "y": [10.0, 20.0]},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    dialog = InterpolationDialog(win.slicer_area)
    qtbot.addWidget(dialog)
    dialog.launch_mode_combo.setCurrentText("Replace Current")

    warnings: list[tuple[str, str]] = []

    def _record_warning(_parent, title, message, *args, **kwargs):
        warnings.append((title, message))
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _record_warning)

    dialog.dim_combo.setCurrentIndex(-1)
    assert dialog.make_code() == ""
    with pytest.raises(ValueError, match="No dimension selected"):
        dialog.source_transform_operation()
    dialog.accept()

    dialog.dim_combo.setCurrentText("x")
    assert dialog.make_code() == ""
    dialog.accept()

    assert warnings == [
        ("No Dimension Selected", "Choose a dimension to interpolate."),
        (
            "Invalid Source Coordinate",
            "The selected dimension coordinate values must be unique.",
        ),
    ]
    xarray.testing.assert_identical(
        erlab.interactive.imagetool.slicer.restore_nonuniform_dims(
            win.slicer_area._data
        ).rename(None),
        data,
    )

    dialog.close()
    win.close()


def test_itool_full_data_child_updates_follow_transposed_view(qtbot) -> None:
    data = _TEST_DATA["2D"].copy(deep=True)
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    win.slicer_area.transpose_main_image()
    assert win.slicer_area.data.dims == ("eV", "alpha")

    win.slicer_area.open_in_meshtool()
    qtbot.wait_until(lambda: len(win.slicer_area._associated_tools) == 1, timeout=5000)
    child = next(iter(win.slicer_area._associated_tools.values()))
    xarray.testing.assert_identical(child.tool_data, win.slicer_area.data)

    replaced = data.copy(deep=True)
    replaced.data = np.asarray(replaced.data) * 2

    with qtbot.wait_signal(win.slicer_area.sigSourceDataReplaced):
        win.slicer_area.replace_source_data(replaced)

    qtbot.wait_until(lambda: child.source_state == "stale", timeout=5000)
    assert child._update_from_parent_source() is True
    xarray.testing.assert_identical(child.tool_data, win.slicer_area.data)

    child.set_source_binding(child.source_spec, auto_update=True, state="fresh")
    replaced2 = replaced.copy(deep=True)
    replaced2.data = np.asarray(replaced2.data) + 5

    with qtbot.wait_signal(win.slicer_area.sigSourceDataReplaced):
        win.slicer_area.replace_source_data(replaced2)

    qtbot.wait_until(lambda: child.source_state == "fresh", timeout=5000)
    xarray.testing.assert_identical(child.tool_data, win.slicer_area.data)


def test_itool_coarsen(qtbot, accept_dialog) -> None:
    data = xr.DataArray(
        np.arange(120).reshape((5, 6, 4)).astype(float),
        dims=["x", "y", "z"],
        coords={"x": np.arange(5), "y": np.arange(6), "z": np.arange(4)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    def _set_dialog_params(dialog: CoarsenDialog) -> None:
        dialog.dim_checks["x"].setChecked(True)
        dialog.dim_checks["y"].setChecked(True)
        dialog.window_spins["x"].setValue(2)
        dialog.window_spins["y"].setValue(4)
        dialog.boundary_combo.setCurrentText("trim")
        dialog.side_combo.setCurrentText("right")
        dialog.coord_func_combo.setCurrentText("min")
        dialog.reducer_combo.setCurrentText("sum")
        with qtbot.wait_signal(dialog._sigCodeCopied):
            dialog.copy_button.click()
        dialog.launch_mode_combo.setCurrentText("Replace Current")

    accept_dialog(win.mnb._coarsen, pre_call=_set_dialog_params)
    xarray.testing.assert_identical(
        win.slicer_area._data.rename(None),
        data.coarsen(x=2, y=4, boundary="trim", side="right", coord_func="min").sum(),
    )

    assert (
        pyperclip.paste()
        == '.coarsen(x=2, y=4, boundary="trim", side="right", coord_func="min").sum()'
    )
    win.close()


def test_itool_coarsen_nonuniform_public_dims(qtbot, accept_dialog) -> None:
    data = xr.DataArray(
        np.arange(30).reshape((5, 6)).astype(float),
        dims=["x", "y"],
        coords={"x": np.array([0.0, 0.3, 0.9, 1.4, 2.2]), "y": np.arange(6)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    assert win.slicer_area.data.dims == ("x_idx", "y")

    def _set_dialog_params(dialog: CoarsenDialog) -> None:
        assert "x" in dialog.dim_checks
        assert "x_idx" not in dialog.dim_checks
        dialog.dim_checks["x"].setChecked(True)
        dialog.window_spins["x"].setValue(2)
        assert dialog.boundary_combo.currentText() == "trim"
        assert dialog.coord_func_combo.currentText() == "mean"
        dialog.launch_mode_combo.setCurrentText("Replace Current")

    accept_dialog(win.mnb._coarsen, pre_call=_set_dialog_params)
    xarray.testing.assert_identical(
        win.slicer_area._data.rename(None),
        data.coarsen(x=2, boundary="trim", side="left", coord_func="mean").mean(),
    )
    assert win.provenance_spec is not None
    display_code = win.provenance_spec.display_code()
    assert display_code is not None
    assert "coarsen(x=2" in display_code
    assert "x_idx" not in display_code

    win.close()


def test_coarsen_dialog_make_code_uses_watched_data_name(qtbot, monkeypatch) -> None:
    data = xr.DataArray(np.arange(12).reshape((3, 4)).astype(float), dims=["x", "y"])
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    dialog = CoarsenDialog(win.slicer_area)
    qtbot.addWidget(dialog)
    dialog.dim_checks["x"].setChecked(True)
    dialog.window_spins["x"].setValue(2)

    monkeypatch.setattr(
        type(win.slicer_area),
        "watched_data_name",
        property(lambda _self: "my_data"),
    )

    namespace = _exec_generated_code(
        f"result = {dialog.make_code()}",
        {"my_data": data.copy(deep=True)},
    )
    result = namespace["result"]
    assert isinstance(result, xr.DataArray)
    xarray.testing.assert_identical(
        result,
        data.coarsen(x=2, boundary="trim").mean(),
    )

    dialog.close()
    win.close()


def test_itool_thin(qtbot, accept_dialog) -> None:
    data = xr.DataArray(
        np.arange(120).reshape((5, 6, 4)).astype(float),
        dims=["x", "y", "z"],
        coords={"x": np.arange(5), "y": np.arange(6), "z": np.arange(4)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    def _set_dialog_params(dialog: ThinDialog) -> None:
        dialog.dim_checks["x"].setChecked(True)
        dialog.factor_spins["x"].setValue(1)
        dialog.dim_checks["y"].setChecked(True)
        dialog.factor_spins["y"].setValue(3)
        with qtbot.wait_signal(dialog._sigCodeCopied):
            dialog.copy_button.click()
        dialog.launch_mode_combo.setCurrentText("Replace Current")

    accept_dialog(win.mnb._thin, pre_call=_set_dialog_params)
    xarray.testing.assert_identical(
        win.slicer_area._data.rename(None),
        data.thin(y=3),
    )

    xarray.testing.assert_identical(
        _exec_data_fragment(data, pyperclip.paste()),
        data.thin(y=3),
    )
    win.close()


def test_itool_thin_global_factor(qtbot, accept_dialog) -> None:
    data = xr.DataArray(
        np.arange(120).reshape((5, 6, 4)).astype(float),
        dims=["x", "y", "z"],
        coords={"x": np.arange(5), "y": np.arange(6), "z": np.arange(4)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    def _set_dialog_params(dialog: ThinDialog) -> None:
        dialog.global_radio.setChecked(True)
        dialog.global_spin.setValue(2)
        with qtbot.wait_signal(dialog._sigCodeCopied):
            dialog.copy_button.click()
        dialog.launch_mode_combo.setCurrentText("Replace Current")

    accept_dialog(win.mnb._thin, pre_call=_set_dialog_params)
    xarray.testing.assert_identical(
        win.slicer_area._data.rename(None),
        data.thin(2),
    )

    xarray.testing.assert_identical(
        _exec_data_fragment(data, pyperclip.paste()),
        data.thin(2),
    )
    win.close()


def test_itool_thin_nonuniform_public_dims(qtbot, accept_dialog) -> None:
    data = xr.DataArray(
        np.arange(30).reshape((5, 6)).astype(float),
        dims=["x", "y"],
        coords={"x": np.array([0.0, 0.3, 0.9, 1.4, 2.2]), "y": np.arange(6)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    assert win.slicer_area.data.dims == ("x_idx", "y")

    def _set_dialog_params(dialog: ThinDialog) -> None:
        assert "x" in dialog.dim_checks
        assert "x_idx" not in dialog.dim_checks
        dialog.dim_checks["x"].setChecked(True)
        dialog.factor_spins["x"].setValue(2)
        dialog.launch_mode_combo.setCurrentText("Replace Current")

    accept_dialog(win.mnb._thin, pre_call=_set_dialog_params)
    xarray.testing.assert_identical(
        win.slicer_area._data.rename(None),
        data.thin(x=2),
    )
    assert win.provenance_spec is not None
    display_code = win.provenance_spec.display_code()
    assert display_code is not None
    assert "thin(x=2)" in display_code
    assert "x_idx" not in display_code

    win.close()


def test_itool_symmetrize(qtbot, accept_dialog, monkeypatch) -> None:
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
    monkeypatch.setattr(
        type(win.slicer_area),
        "watched_data_name",
        property(lambda _self: "data"),
    )

    # Test dialog
    def _set_dialog_params(dialog: SymmetrizeDialog) -> None:
        dialog._dim_combo.setCurrentIndex(2)
        dialog._center_spin.setValue(2.0)
        with qtbot.wait_signal(dialog._sigCodeCopied):
            dialog.copy_button.click()
        dialog.launch_mode_combo.setCurrentText("Replace Current")

    accept_dialog(win.mnb._symmetrize, pre_call=_set_dialog_params)
    xarray.testing.assert_identical(
        win.slicer_area._data.rename(None),
        erlab.analysis.transform.symmetrize(data, "z", center=2),
    )

    namespace = _exec_generated_code(
        f"result = {pyperclip.paste()}",
        {"data": data.copy(deep=True)},
    )
    result = namespace["result"]
    assert isinstance(result, xr.DataArray)
    xarray.testing.assert_identical(
        result,
        erlab.analysis.transform.symmetrize(data, "z", center=2),
    )
    win.close()


def test_itool_symmetrize_nfold(qtbot, accept_dialog, monkeypatch) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["y", "x"],
        coords={
            "y": np.arange(-2.0, 3.0, dtype=float),
            "x": np.arange(-2.0, 3.0, dtype=float),
        },
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    monkeypatch.setattr(
        type(win.slicer_area),
        "watched_data_name",
        property(lambda _self: "data"),
    )

    def _set_dialog_params(dialog: SymmetrizeNfoldDialog) -> None:
        assert dialog._axes == ("y", "x")
        assert dialog.fold_spin.value() == 4
        assert dialog.center_spins[0].value() == 0.0
        assert dialog.center_spins[1].value() == 0.0
        dialog.fold_spin.setValue(6)
        dialog.center_spins[0].setValue(1.0)
        dialog.center_spins[1].setValue(-1.0)
        dialog.reshape_check.setChecked(False)
        dialog.order_spin.setValue(3)
        with qtbot.wait_signal(dialog._sigCodeCopied):
            dialog.copy_button.click()
        dialog.launch_mode_combo.setCurrentText("Replace Current")

    accept_dialog(win.mnb._symmetrize_nfold, pre_call=_set_dialog_params)
    xarray.testing.assert_allclose(
        win.slicer_area._data.rename(None),
        erlab.analysis.transform.symmetrize_nfold(
            data,
            6,
            axes=("y", "x"),
            center={"y": 1.0, "x": -1.0},
            reshape=False,
            order=3,
        ),
    )

    namespace = _exec_generated_code(
        f"result = {pyperclip.paste()}",
        {"data": data.copy(deep=True)},
    )
    result = namespace["result"]
    assert isinstance(result, xr.DataArray)
    xarray.testing.assert_allclose(
        result,
        erlab.analysis.transform.symmetrize_nfold(
            data,
            6,
            axes=("y", "x"),
            center={"y": 1.0, "x": -1.0},
            reshape=False,
            order=3,
        ),
    )
    win.close()


def test_itool_symmetrize_nfold_guideline_prefill_defaults(qtbot) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["y", "x"],
        coords={"y": np.arange(5, dtype=float), "x": np.arange(5, dtype=float)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    dialog = SymmetrizeNfoldDialog(win.slicer_area)
    qtbot.addWidget(dialog)

    assert dialog._axes == ("y", "x")
    assert dialog.fold_spin.value() == 4
    assert dialog.center_spins[0].value() == 2.0
    assert dialog.center_spins[1].value() == 2.0
    assert dialog.reshape_check.isChecked()
    assert dialog.order_spin.value() == 1

    dialog.close()
    win.close()


@pytest.mark.parametrize(
    ("guideline_count", "expected_fold"),
    [(1, 2), (2, 4), (3, 6)],
    ids=["C2", "C4", "C6"],
)
def test_itool_symmetrize_nfold_guideline_prefill(
    qtbot, guideline_count, expected_fold
) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["y", "x"],
        coords={"y": np.arange(5, dtype=float), "x": np.arange(5, dtype=float)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    plot_item = win.slicer_area.main_image
    plot_item.set_guidelines(guideline_count)
    plot_item._guidelines_items[-1].setPos((1.0, 3.0))

    dialog = SymmetrizeNfoldDialog(win.slicer_area)
    qtbot.addWidget(dialog)

    assert dialog.fold_spin.value() == expected_fold
    assert dialog.center_spins[0].value() == 1.0
    assert dialog.center_spins[1].value() == 3.0

    dialog.close()
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

    accept_dialog(
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


@pytest.mark.parametrize("shift_coords", [True, False], ids=["shift", "no_shift"])
def test_itool_edgecorr(qtbot, accept_dialog, gold, gold_fit_res, shift_coords) -> None:
    win = itool(gold, execute=False)
    qtbot.addWidget(win)

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        filename = f"{tmp_dir_name}/fit_res.nc"

        xarray_lmfit.save_fit(gold_fit_res, filename)

        # Test dialog
        def _set_dialog_params(dialog: EdgeCorrectionDialog) -> None:
            dialog.shift_coord_check.setChecked(shift_coords)
            dialog.launch_mode_combo.setCurrentText("Replace Current")

        def _go_to_file(dialog: QtWidgets.QFileDialog):
            dialog.setDirectory(tmp_dir_name)
            dialog.selectFile(filename)
            focused = dialog.focusWidget()
            if isinstance(focused, QtWidgets.QLineEdit):
                focused.setText("fit_res.nc")

        accept_dialog(
            win.mnb._correct_with_edge,
            pre_call=[_go_to_file, _set_dialog_params],
            chained_dialogs=2,
        )
        xarray.testing.assert_identical(
            win.slicer_area._data.rename(None),
            erlab.analysis.gold.correct_with_edge(
                gold, gold_fit_res, shift_coords=shift_coords
            ),
        )


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


def _set_spinbox_text(spin: erlab.interactive.utils.BetterSpinBox, text: str) -> None:
    line = spin.lineEdit()
    assert line is not None
    line.setText(text)
    spin.editingFinishedEvent()


def test_itool_assign_coords(qtbot, accept_dialog) -> None:
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
    def _set_dialog_params(dialog: AssignCoordsDialog) -> None:
        dialog._coord_combo.setCurrentText("t")
        dialog.coord_widget.mode_combo.setCurrentIndex(1)  # Set to 'Delta'
        dialog.coord_widget.spin0.setValue(1)
        dialog.launch_mode_combo.setCurrentText("Replace Current")

    accept_dialog(win.mnb._assign_coords, pre_call=_set_dialog_params, timeout=10.0)
    np.testing.assert_allclose(win.slicer_area._data.t.values, np.arange(3) + 1.0)


def test_itool_assign_coords_affine(qtbot, accept_dialog) -> None:
    data = xr.DataArray(
        np.arange(12).reshape((3, 4)).astype(float),
        dims=["x", "y"],
        coords={"x": np.arange(3), "y": np.arange(4)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    def _set_dialog_params(dialog: AssignCoordsDialog) -> None:
        dialog._coord_combo.setCurrentText("y")
        dialog.coord_widget.edit_mode_tabs.setCurrentIndex(1)
        dialog.coord_widget.scale_spin.setValue(2.0)
        dialog.coord_widget.offset_spin.setValue(0.5)
        dialog.launch_mode_combo.setCurrentText("Replace Current")

    accept_dialog(win.mnb._assign_coords, pre_call=_set_dialog_params, timeout=10.0)
    np.testing.assert_allclose(win.slicer_area._data.y.values, 2.0 * np.arange(4) + 0.5)


def _combo_index_for_data(combo: QtWidgets.QComboBox, data: object) -> int:
    index = combo.findData(data, QtCore.Qt.ItemDataRole.UserRole)
    assert index >= 0
    return index


def test_itool_assign_coords_add_scalar(qtbot, accept_dialog) -> None:
    data = xr.DataArray(
        np.arange(6).reshape((2, 3)).astype(float),
        dims=["x", "y"],
        coords={"x": np.arange(2), "y": np.arange(3)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    def _set_dialog_params(dialog: AssignCoordsDialog) -> None:
        dialog._mode_tabs.setCurrentIndex(1)
        dialog._add_name_edit.setText("temperature")
        dialog._add_kind_combo.setCurrentText("Scalar")
        dialog._add_literal_edit.setText("21.5")
        dialog.launch_mode_combo.setCurrentText("Replace Current")

    accept_dialog(win.mnb._assign_coords, pre_call=_set_dialog_params, timeout=10.0)
    assert win.slicer_area._data.coords["temperature"].dims == ()
    assert float(win.slicer_area._data.temperature) == pytest.approx(21.5)


def test_itool_assign_coords_add_1d_numeric(qtbot, accept_dialog) -> None:
    data = xr.DataArray(
        np.arange(12).reshape((3, 4)).astype(float),
        dims=["x", "y"],
        coords={"x": np.arange(3), "y": np.arange(4)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    def _set_dialog_params(dialog: AssignCoordsDialog) -> None:
        dialog._mode_tabs.setCurrentIndex(1)
        dialog._add_name_edit.setText("delay")
        dialog._add_kind_combo.setCurrentText("1D Along Coordinate")
        dialog._add_ref_combo.setCurrentIndex(
            _combo_index_for_data(dialog._add_ref_combo, "x")
        )
        dialog._add_value_mode_combo.setCurrentText("Numeric Values")
        dialog._add_coord_widget.mode_combo.setCurrentText("Delta")
        dialog._add_coord_widget.spin0.setValue(10.0)
        dialog._add_coord_widget.spin1.setValue(2.0)
        dialog.launch_mode_combo.setCurrentText("Replace Current")

    accept_dialog(win.mnb._assign_coords, pre_call=_set_dialog_params, timeout=10.0)
    assert win.slicer_area._data.delay.dims == ("x",)
    np.testing.assert_allclose(win.slicer_area._data.delay.values, [10.0, 12.0, 14.0])


def test_itool_assign_coords_add_1d_literal(qtbot, accept_dialog) -> None:
    data = xr.DataArray(
        np.arange(12).reshape((3, 4)).astype(float),
        dims=["x", "y"],
        coords={"x": np.arange(3), "y": np.arange(4)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    def _set_dialog_params(dialog: AssignCoordsDialog) -> None:
        dialog._mode_tabs.setCurrentIndex(1)
        dialog._add_name_edit.setText("label")
        dialog._add_kind_combo.setCurrentText("1D Along Coordinate")
        dialog._add_ref_combo.setCurrentIndex(
            _combo_index_for_data(dialog._add_ref_combo, "x")
        )
        dialog._add_value_mode_combo.setCurrentText("Python Literal")
        dialog._add_literal_edit.setText("['low', 'mid', 'high']")
        dialog.launch_mode_combo.setCurrentText("Replace Current")

    accept_dialog(win.mnb._assign_coords, pre_call=_set_dialog_params, timeout=10.0)
    assert win.slicer_area._data.label.dims == ("x",)
    assert win.slicer_area._data.label.values.tolist() == ["low", "mid", "high"]


def test_assign_coords_add_dialog_accept_validates_inputs(qtbot, monkeypatch) -> None:
    data = xr.DataArray(
        np.arange(6).reshape((2, 3)).astype(float),
        dims=["x", "y"],
        coords={"x": np.arange(2), "y": np.arange(3)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    dialog = AssignCoordsDialog(win.slicer_area)
    qtbot.addWidget(dialog)
    dialog._mode_tabs.setCurrentIndex(1)

    warnings: list[tuple[str, str]] = []

    def _record_warning(_parent, title, message, *args, **kwargs):
        warnings.append((title, message))
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _record_warning)

    dialog._add_name_edit.setText("x")
    dialog.accept()
    dialog._add_name_edit.setText("new_coord")
    dialog._add_kind_combo.setCurrentText("Scalar")
    dialog._add_literal_edit.setText("{bad")
    dialog.accept()
    dialog._add_kind_combo.setCurrentText("1D Along Coordinate")
    dialog._add_ref_combo.setCurrentIndex(
        _combo_index_for_data(dialog._add_ref_combo, "x")
    )
    dialog._add_value_mode_combo.setCurrentText("Python Literal")
    dialog._add_literal_edit.setText("[1.0]")
    dialog.accept()

    assert warnings[0] == (
        "Duplicate Name",
        "A coordinate or dimension named 'x' already exists.",
    )
    assert [title for title, _message in warnings[1:]] == [
        "Invalid Coordinate Value",
        "Invalid Coordinate Value",
    ]
    xarray.testing.assert_identical(win.slicer_area._data.rename(None), data)

    dialog.close()
    win.close()


def _attr_row(dialog: AssignAttrsDialog, key: object) -> int:
    for row in range(dialog.table.rowCount()):
        if dialog._row_key(row) == key:
            return row
    raise AssertionError(f"Attribute row {key!r} was not found")


def _attr_type_combo(dialog: AssignAttrsDialog, row: int) -> QtWidgets.QComboBox:
    return typing.cast("QtWidgets.QComboBox", dialog.table.cellWidget(row, 1))


def test_itool_assign_attrs_typed_values(qtbot, accept_dialog) -> None:
    data = xr.DataArray(
        np.arange(6).reshape((2, 3)).astype(float),
        dims=["x", "y"],
        coords={"x": np.arange(2), "y": np.arange(3)},
        attrs={"source": "old", "count": 1},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    def _set_dialog_params(dialog: AssignAttrsDialog) -> None:
        source_row = _attr_row(dialog, "source")
        source_type_combo = _attr_type_combo(dialog, source_row)
        type_names = [
            source_type_combo.itemText(index)
            for index in range(source_type_combo.count())
        ]
        assert "Bool" in type_names
        assert "None" not in type_names
        dialog.table.item(source_row, 2).setText("new")
        dialog._add_empty_row()
        new_row = dialog.table.rowCount() - 1
        dialog.table.item(new_row, 0).setText("temperature")
        _attr_type_combo(dialog, new_row).setCurrentText("Float")
        dialog.table.item(new_row, 2).setText("21.5")
        dialog.launch_mode_combo.setCurrentText("Replace Current")

    accept_dialog(win.mnb._assign_attrs, pre_call=_set_dialog_params, timeout=10.0)
    assert win.slicer_area._data.attrs == {
        "source": "new",
        "count": 1,
        "temperature": 21.5,
    }


def test_itool_assign_attrs_blank_string_does_not_delete(qtbot, accept_dialog) -> None:
    data = xr.DataArray(
        np.arange(6).reshape((2, 3)).astype(float),
        dims=["x", "y"],
        coords={"x": np.arange(2), "y": np.arange(3)},
        attrs={"note": "keep"},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    def _set_dialog_params(dialog: AssignAttrsDialog) -> None:
        note_row = _attr_row(dialog, "note")
        dialog.table.item(note_row, 2).setText("")
        dialog.launch_mode_combo.setCurrentText("Replace Current")

    accept_dialog(win.mnb._assign_attrs, pre_call=_set_dialog_params, timeout=10.0)
    assert "note" in win.slicer_area._data.attrs
    assert win.slicer_area._data.attrs["note"] == ""


def test_itool_assign_attrs_python_literal_none(qtbot, accept_dialog) -> None:
    data = xr.DataArray(
        np.arange(6).reshape((2, 3)).astype(float),
        dims=["x", "y"],
        coords={"x": np.arange(2), "y": np.arange(3)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    def _set_dialog_params(dialog: AssignAttrsDialog) -> None:
        dialog._add_empty_row()
        new_row = dialog.table.rowCount() - 1
        dialog.table.item(new_row, 0).setText("optional_note")
        _attr_type_combo(dialog, new_row).setCurrentText("Python literal")
        dialog.table.item(new_row, 2).setText("None")
        dialog.launch_mode_combo.setCurrentText("Replace Current")

    accept_dialog(win.mnb._assign_attrs, pre_call=_set_dialog_params, timeout=10.0)
    assert win.slicer_area._data.attrs == {"optional_note": None}


def test_assign_attrs_dialog_accept_validates_inputs(qtbot, monkeypatch) -> None:
    data = xr.DataArray(
        np.arange(6).reshape((2, 3)).astype(float),
        dims=["x", "y"],
        coords={"x": np.arange(2), "y": np.arange(3)},
        attrs={"count": 1},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    dialog = AssignAttrsDialog(win.slicer_area)
    qtbot.addWidget(dialog)

    warnings: list[tuple[str, str]] = []

    def _record_warning(_parent, title, message, *args, **kwargs):
        warnings.append((title, message))
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _record_warning)

    dialog.accept()
    dialog._add_empty_row()
    dialog.accept()
    new_row = dialog.table.rowCount() - 1
    dialog.table.item(new_row, 0).setText("count")
    dialog.accept()
    dialog.table.item(new_row, 0).setText("bad")
    _attr_type_combo(dialog, new_row).setCurrentText("Python literal")
    dialog.table.item(new_row, 2).setText("{bad")
    dialog.accept()

    assert [title for title, _message in warnings] == [
        "No Attributes Changed",
        "Invalid Attribute Value",
        "Duplicate Names",
        "Invalid Attribute Value",
    ]
    xarray.testing.assert_identical(win.slicer_area._data.rename(None), data)

    dialog.close()
    win.close()


def _set_rename_dialog_name(
    dialog: RenameDimsCoordsDialog, old_name: object, new_name: str
) -> None:
    row = dialog._rename_sources.index(old_name)
    item = dialog.table.item(row, 1)
    assert item is not None
    item.setText(new_name)


def test_itool_rename_dims_coords(qtbot, accept_dialog) -> None:
    data = xr.DataArray(
        np.arange(6).reshape((2, 3)).astype(float),
        dims=["x", "y"],
        coords={
            "x": [0.0, 1.0],
            "y": [10.0, 11.0, 12.0],
            "temp": ("x", [100.0, 101.0]),
        },
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    def _set_dialog_params(dialog: RenameDimsCoordsDialog) -> None:
        _set_rename_dialog_name(dialog, "x", "kx")
        _set_rename_dialog_name(dialog, "temp", "temperature")
        with qtbot.wait_signal(dialog._sigCodeCopied):
            dialog.copy_button.click()
        dialog.launch_mode_combo.setCurrentText("Replace Current")

    accept_dialog(win.mnb._rename_dims_coords, pre_call=_set_dialog_params)

    expected = data.rename({"x": "kx", "temp": "temperature"})
    xarray.testing.assert_identical(win.slicer_area._data.rename(None), expected)

    result = _exec_data_fragment(data, pyperclip.paste())
    xarray.testing.assert_identical(result, expected)

    win.close()


def test_itool_rename_dims_coords_nonuniform_public_dims(qtbot, accept_dialog) -> None:
    data = xr.DataArray(
        np.arange(15).reshape((3, 5)).astype(float),
        dims=["x", "y"],
        coords={
            "x": np.array([0.0, 0.4, 1.0]),
            "y": np.arange(5),
            "temperature": ("x", [100.0, 101.0, 102.0]),
        },
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    assert win.slicer_area.data.dims == ("x_idx", "y")

    def _set_dialog_params(dialog: RenameDimsCoordsDialog) -> None:
        assert "x" in dialog._rename_sources
        assert "x_idx" not in dialog._rename_sources
        _set_rename_dialog_name(dialog, "x", "kx")
        dialog.launch_mode_combo.setCurrentText("Replace Current")

    accept_dialog(win.mnb._rename_dims_coords, pre_call=_set_dialog_params)
    xarray.testing.assert_identical(
        win.slicer_area._data.rename(None), data.rename({"x": "kx"})
    )
    assert win.provenance_spec is not None
    display_code = win.provenance_spec.display_code()
    assert display_code is not None
    assert ".rename(" in display_code
    assert "x_idx" not in display_code

    win.close()


def test_rename_dims_coords_dialog_accept_validates_inputs(qtbot, monkeypatch) -> None:
    data = xr.DataArray(
        np.arange(6).reshape((2, 3)).astype(float),
        dims=["x", "y"],
        coords={"x": [0.0, 1.0], "y": [10.0, 11.0, 12.0]},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    dialog = RenameDimsCoordsDialog(win.slicer_area)
    qtbot.addWidget(dialog)

    warnings: list[tuple[str, str]] = []

    def _record_warning(_parent, title, message, *args, **kwargs):
        warnings.append((title, message))
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _record_warning)

    dialog.accept()
    _set_rename_dialog_name(dialog, "x", " ")
    dialog.accept()
    _set_rename_dialog_name(dialog, "x", "y")
    dialog.accept()

    assert warnings == [
        ("No Names Changed", "Edit at least one coordinate or dimension name."),
        ("Empty Name", "Names cannot be empty."),
        ("Duplicate Names", "Names must be unique after renaming: y."),
    ]
    xarray.testing.assert_identical(win.slicer_area._data.rename(None), data)

    dialog.close()
    win.close()


def test_itool_swap_dims(qtbot, accept_dialog) -> None:
    data = xr.DataArray(
        np.arange(24).reshape((2, 3, 4)).astype(float),
        dims=["x", "y", "z"],
        coords={
            "x": np.arange(2),
            "y": np.arange(3),
            "z": np.arange(4),
            "u": ("x", [5.0, 6.0]),
            "v": ("y", [10.0, 11.0, 12.0]),
            "w": ("z", [20.0, 21.0, 22.0, 23.0]),
        },
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    def _set_dialog_params(dialog: SwapDimsDialog) -> None:
        dialog.target_combos["x"].setCurrentText("u")
        dialog.launch_mode_combo.setCurrentText("Replace Current")

    accept_dialog(win.mnb._swap_dims, pre_call=_set_dialog_params)
    xarray.testing.assert_identical(
        win.slicer_area._data.rename(None), data.swap_dims({"x": "u"})
    )

    win.close()


def test_itool_swap_dims_multiple_and_code(qtbot, accept_dialog) -> None:
    data = xr.DataArray(
        np.arange(24).reshape((2, 3, 4)).astype(float),
        dims=["x", "y", "z"],
        coords={
            "x": np.arange(2),
            "y": np.arange(3),
            "z": np.arange(4),
            "u": ("x", [5.0, 6.0]),
            "v": ("y", [10.0, 11.0, 12.0]),
            "w": ("z", [20.0, 21.0, 22.0, 23.0]),
        },
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    def _set_dialog_params(dialog: SwapDimsDialog) -> None:
        dialog.target_combos["x"].setCurrentText("u")
        dialog.target_combos["y"].setCurrentText("v")
        with qtbot.wait_signal(dialog._sigCodeCopied):
            dialog.copy_button.click()
        dialog.launch_mode_combo.setCurrentText("Replace Current")

    accept_dialog(win.mnb._swap_dims, pre_call=_set_dialog_params)
    xarray.testing.assert_identical(
        win.slicer_area._data.rename(None), data.swap_dims({"x": "u", "y": "v"})
    )

    code = pyperclip.paste()
    assert code == '.swap_dims(x="u", y="v")'
    assert "z=" not in code
    assert '"z"' not in code

    win.close()


def test_itool_swap_dims_nonuniform_public_dims(qtbot, accept_dialog) -> None:
    data = xr.DataArray(
        np.arange(15).reshape((3, 5)).astype(float),
        dims=["x", "y"],
        coords={
            "x": np.array([0.0, 0.4, 1.0]),
            "y": np.arange(5),
            "temperature": ("x", [100.0, 101.0, 102.0]),
        },
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    assert win.slicer_area.data.dims == ("x_idx", "y")

    def _set_dialog_params(dialog: SwapDimsDialog) -> None:
        assert "x" in dialog.target_combos
        assert "x_idx" not in dialog.target_combos
        dialog.target_combos["x"].setCurrentText("temperature")
        dialog.launch_mode_combo.setCurrentText("Replace Current")

    accept_dialog(win.mnb._swap_dims, pre_call=_set_dialog_params)
    xarray.testing.assert_identical(
        win.slicer_area._data.rename(None), data.swap_dims({"x": "temperature"})
    )
    assert win.provenance_spec is not None
    display_code = win.provenance_spec.display_code()
    assert display_code is not None
    assert "swap_dims(x='temperature')" in display_code
    assert "x_idx" not in display_code

    win.close()


def test_swap_dims_dialog_rejects_unswappable_data(qtbot, monkeypatch) -> None:
    data = xr.DataArray(
        np.arange(12).reshape((3, 4)).astype(float),
        dims=["x", "y"],
        coords={"x": np.arange(3), "y": np.arange(4)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    dialog = SwapDimsDialog(win.slicer_area)
    qtbot.addWidget(dialog)

    warnings: list[tuple[str, str]] = []

    def _record_warning(_parent, title, message, *args, **kwargs):
        warnings.append((title, message))
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _record_warning)

    assert dialog._validate() == QtWidgets.QDialog.DialogCode.Rejected
    assert warnings == [
        (
            "Nothing to Swap",
            "No compatible 1D coordinates are available for any dimension.",
        )
    ]

    dialog.close()
    win.close()


def test_swap_dims_dialog_accept_requires_changes(qtbot, monkeypatch) -> None:
    data = xr.DataArray(
        np.arange(12).reshape((3, 4)).astype(float),
        dims=["x", "y"],
        coords={"x": np.arange(3), "y": np.arange(4), "u": ("x", [5.0, 6.0, 7.0])},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    dialog = SwapDimsDialog(win.slicer_area)
    qtbot.addWidget(dialog)

    warnings: list[tuple[str, str]] = []

    def _record_warning(_parent, title, message, *args, **kwargs):
        warnings.append((title, message))
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _record_warning)

    dialog.accept()

    assert warnings == [
        ("No Dimensions Changed", "Choose at least one dimension to swap.")
    ]
    xarray.testing.assert_identical(win.slicer_area._data.rename(None), data)

    dialog.close()
    win.close()


def test_coarsen_dialog_requires_selected_dimension(qtbot, monkeypatch) -> None:
    data = xr.DataArray(np.arange(12).reshape((3, 4)).astype(float), dims=["x", "y"])
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    dialog = CoarsenDialog(win.slicer_area)
    qtbot.addWidget(dialog)

    warnings: list[tuple[str, str]] = []

    def _record_warning(_parent, title, message, *args, **kwargs):
        warnings.append((title, message))
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _record_warning)

    dialog.accept()

    assert warnings == [
        ("No Dimensions Selected", "You need to select at least one dimension.")
    ]
    xarray.testing.assert_identical(win.slicer_area._data.rename(None), data)

    dialog.close()
    win.close()


def test_coarsen_dialog_rejects_exact_incompatible_windows(qtbot, monkeypatch) -> None:
    data = xr.DataArray(np.arange(10).reshape((5, 2)).astype(float), dims=["x", "y"])
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    dialog = CoarsenDialog(win.slicer_area)
    qtbot.addWidget(dialog)
    dialog.dim_checks["x"].setChecked(True)
    dialog.window_spins["x"].setValue(3)
    dialog.boundary_combo.setCurrentText("exact")

    warnings: list[tuple[str, str]] = []

    def _record_warning(_parent, title, message, *args, **kwargs):
        warnings.append((title, message))
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _record_warning)

    dialog.accept()

    assert warnings == [
        (
            "Incompatible Window Size",
            "Window sizes must evenly divide the selected dimensions when boundary "
            "is exact: x. Try trim or pad instead.",
        )
    ]
    xarray.testing.assert_identical(win.slicer_area._data.rename(None), data)

    dialog.close()
    win.close()


def test_coarsen_dialog_rejects_trim_without_output_blocks(qtbot, monkeypatch) -> None:
    data = xr.DataArray(np.arange(4).reshape((2, 2)).astype(float), dims=["x", "y"])
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    dialog = CoarsenDialog(win.slicer_area)
    qtbot.addWidget(dialog)
    dialog.dim_checks["x"].setChecked(True)
    dialog.window_spins["x"].setValue(3)
    dialog.boundary_combo.setCurrentText("trim")

    warnings: list[tuple[str, str]] = []

    def _record_warning(_parent, title, message, *args, **kwargs):
        warnings.append((title, message))
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _record_warning)

    dialog.accept()

    assert warnings == [
        ("No Output Blocks", "Trim boundary would remove all data along: x.")
    ]
    xarray.testing.assert_identical(win.slicer_area._data.rename(None), data)

    dialog.close()
    win.close()


def test_thin_dialog_requires_selected_dimension(qtbot, monkeypatch) -> None:
    data = xr.DataArray(np.arange(12).reshape((3, 4)).astype(float), dims=["x", "y"])
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    dialog = ThinDialog(win.slicer_area)
    qtbot.addWidget(dialog)

    warnings: list[tuple[str, str]] = []

    def _record_warning(_parent, title, message, *args, **kwargs):
        warnings.append((title, message))
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _record_warning)

    dialog.accept()

    assert warnings == [
        ("No Dimensions Selected", "You need to select at least one dimension.")
    ]
    xarray.testing.assert_identical(win.slicer_area._data.rename(None), data)

    dialog.close()
    win.close()


@pytest.mark.parametrize("global_mode", [False, True], ids=["per_dim", "global"])
def test_thin_dialog_rejects_noop_factors(qtbot, monkeypatch, global_mode) -> None:
    data = xr.DataArray(np.arange(12).reshape((3, 4)).astype(float), dims=["x", "y"])
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    dialog = ThinDialog(win.slicer_area)
    qtbot.addWidget(dialog)

    if global_mode:
        dialog.global_radio.setChecked(True)
        dialog.global_spin.setValue(1)
    else:
        dialog.dim_checks["x"].setChecked(True)
        dialog.factor_spins["x"].setValue(1)

    warnings: list[tuple[str, str]] = []

    def _record_warning(_parent, title, message, *args, **kwargs):
        warnings.append((title, message))
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _record_warning)

    dialog.accept()

    assert warnings == [
        (
            "No Thinning Requested",
            "Choose at least one thinning factor greater than 1.",
        )
    ]
    xarray.testing.assert_identical(win.slicer_area._data.rename(None), data)

    dialog.close()
    win.close()


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


@pytest.mark.parametrize(
    ("option", "expected_valid"),
    [
        (0, [1.0, 1.0]),
        (3, [0.0, 0.0]),
    ],
)
def test_itool_normalize_masks_unsafe_area_denominators(
    qtbot, option, expected_valid
) -> None:
    data = xr.DataArray(
        np.array(
            [
                [1.0, 1.0, np.inf, np.nan, 1e15],
                [-1.0, -1.0 + 1e-14, np.inf, np.nan, 1e15],
            ]
        ),
        dims=["x", "y"],
        coords={"x": np.arange(2), "y": np.arange(5)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    dialog = NormalizeDialog(win.slicer_area)
    qtbot.addWidget(dialog)
    dialog.dim_checks["x"].setChecked(True)
    dialog.opts[option].setChecked(True)

    result = dialog.process_data(data)

    assert np.isnan(result.isel(y=slice(0, 4)).values).all()
    np.testing.assert_allclose(result.isel(y=4).values, expected_valid)

    dialog.close()
    win.close()


def test_itool_normalize_masks_unsafe_range_denominators(qtbot) -> None:
    data = xr.DataArray(
        np.array(
            [
                [2.0, 1.0, np.inf, np.nan, 0.0],
                [2.0, 1.0 + 1e-14, np.inf, np.nan, 1e15],
            ]
        ),
        dims=["x", "y"],
        coords={"x": np.arange(2), "y": np.arange(5)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    dialog = NormalizeDialog(win.slicer_area)
    qtbot.addWidget(dialog)
    dialog.dim_checks["x"].setChecked(True)
    dialog.opts[1].setChecked(True)

    result = dialog.process_data(data)

    assert np.isnan(result.isel(y=slice(0, 4)).values).all()
    np.testing.assert_allclose(result.isel(y=4).values, [0.0, 1.0])

    dialog.close()
    win.close()


def test_itool_masks_unsafe_values_for_display_only(qtbot) -> None:
    data = xr.DataArray(
        np.array([[0.0, 1e15], [np.inf, 1e300]]),
        dims=["x", "y"],
        coords={"x": np.arange(2), "y": np.arange(2)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    displayed = win.slicer_area._imageitems[0].image

    assert displayed is not None
    assert np.nanmax(displayed) == 1e15
    assert np.isnan(displayed).sum() == 2
    assert np.isinf(win.slicer_area.data.values[1, 0])
    assert win.slicer_area.data.values[1, 1] == 1e300

    control = win.docks[0].widget().findChild(ItoolCrosshairControls)
    assert control is not None
    assert np.isnan(control._readout_value_to_float(np.array([np.inf, 1e300])))

    win.slicer_area.lock_levels(True)
    np.testing.assert_allclose(win.slicer_area.levels, (0.0, 1e15))

    win.close()


def test_itool_normalize_to_view_ignores_unsafe_display_values(qtbot) -> None:
    values = np.arange(25, dtype=float).reshape((5, 5))
    values[2, 1] = 1e300
    values[2, 2] = np.inf
    values[3, 1] = 1e15
    data = xr.DataArray(
        values,
        dims=["x", "y"],
        coords={"x": np.arange(5), "y": np.arange(5)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    with qtbot.wait_exposed(win):
        win.show()
        win.activateWindow()

    set_vb_range(
        win.slicer_area.main_image.getViewBox(), x_range=(1, 4), y_range=(0, 3)
    )
    win.slicer_area.main_image.normalize_to_current_view()

    np.testing.assert_allclose(win.slicer_area.levels, (5.0, 1e15))
    assert win.slicer_area.data.values[2, 1] == 1e300
    assert np.isinf(win.slicer_area.data.values[2, 2])

    win.close()


def test_itool_skips_display_mask_when_global_limits_are_safe(qtbot, monkeypatch):
    data = xr.DataArray(
        np.arange(27, dtype=float).reshape((3, 3, 3)),
        dims=["x", "y", "z"],
        coords={"x": np.arange(3), "y": np.arange(3), "z": np.arange(3)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    win.slicer_area.lock_levels(True)
    assert win.slicer_area.array_slicer.display_values_known_safe

    calls = []

    def record_display_mask(values, limit=None):
        calls.append(values)
        return values

    monkeypatch.setattr(
        erlab.interactive.imagetool.slicer,
        "_display_safe_values",
        record_display_mask,
    )

    win.slicer_area.refresh_all(only_plots=True)

    assert calls == []

    win.close()


def test_itool_divide_by_coord(qtbot, accept_dialog) -> None:
    data = xr.DataArray(
        np.arange(12, dtype=float).reshape((3, 4)) + 1.0,
        dims=["x", "y"],
        coords={
            "x": np.arange(3),
            "y": np.arange(4),
            "mesh_current": ("x", [1.0, 2.0, 4.0]),
        },
        name="scan",
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    def _set_dialog_params(dialog: DivideByCoordDialog) -> None:
        dialog.coord_combo.setCurrentText("mesh_current")
        with qtbot.wait_signal(dialog._sigCodeCopied):
            dialog.copy_button.click()
        dialog.launch_mode_combo.setCurrentText("Replace Current")

    accept_dialog(win.mnb._divide_by_coord, pre_call=_set_dialog_params)

    expected = (data / data.mesh_current).rename("scan_div_mesh_current")
    xarray.testing.assert_identical(win.slicer_area._data, expected)
    copied_code = pyperclip.paste()
    assert "data.mesh_current" in copied_code
    namespace: dict[str, typing.Any] = {"data": data.copy(deep=True)}
    exec(f"result = {copied_code}", {}, namespace)  # noqa: S102
    xarray.testing.assert_identical(
        namespace["result"].rename(None), expected.rename(None)
    )

    win.close()


def test_itool_divide_by_coord_rejects_zero_values(qtbot, monkeypatch) -> None:
    data = xr.DataArray(
        np.arange(12, dtype=float).reshape((3, 4)) + 1.0,
        dims=["x", "y"],
        coords={
            "x": np.arange(3),
            "y": np.arange(4),
            "mesh_current": ("x", [1.0, 0.0, 4.0]),
        },
        name="scan",
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    dialog = DivideByCoordDialog(win.slicer_area)
    qtbot.addWidget(dialog)
    dialog.coord_combo.setCurrentText("mesh_current")
    dialog.launch_mode_combo.setCurrentText("Replace Current")

    warnings: list[tuple[str, str]] = []

    def _record_warning(_parent, title, message, *args, **kwargs):
        warnings.append((title, message))
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _record_warning)

    dialog.accept()

    assert warnings == [
        (
            "Zero Coordinate Values",
            "The selected coordinate contains zero values and cannot be used as a "
            "divisor.",
        )
    ]
    xarray.testing.assert_identical(win.slicer_area._data, data)

    dialog.close()
    win.close()


def test_divide_by_coord_dialog_edge_paths(qtbot, monkeypatch) -> None:
    data = xr.DataArray(
        np.arange(12, dtype=float).reshape((3, 4)) + 1.0,
        dims=["x", "y"],
        coords={
            "x": np.arange(3),
            "y": np.arange(4),
            "2 current": ("x", [1.0, 2.0, 4.0]),
            "scalar_current": 2.0,
            "label": ("x", ["a", "b", "c"]),
            "complex_current": ("x", [1.0 + 0.0j, 2.0 + 0.0j, 4.0 + 0.0j]),
        },
        name="scan",
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    dialog = DivideByCoordDialog(win.slicer_area)
    qtbot.addWidget(dialog)

    coord_names = {
        dialog.coord_combo.itemData(i, QtCore.Qt.ItemDataRole.UserRole)
        for i in range(dialog.coord_combo.count())
    }
    assert "label" not in coord_names
    assert "complex_current" not in coord_names

    dialog.coord_combo.setCurrentText("scalar_current")
    assert dialog.coord_dims_label.text() == "scalar"
    assert dialog.suffix == "_div_scalar_current"
    dialog.suffix = ""

    dialog.coord_combo.setCurrentText("2 current")
    assert dialog.suffix == "_div_coord_2_current"

    warnings: list[tuple[str, str]] = []

    def _record_warning(_parent, title, message, *args, **kwargs):
        warnings.append((title, message))
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _record_warning)

    dialog.coord_combo.setCurrentIndex(-1)
    dialog._update_coord_dims_label()
    assert dialog.coord_dims_label.text() == ""
    assert dialog.suffix == "_div_coord"
    assert dialog.make_code() == ""
    with pytest.raises(ValueError, match="No coordinate selected"):
        dialog.source_transform_operation()

    dialog.accept()
    assert warnings == [("No Coordinate Selected", "Choose a coordinate to divide by.")]

    warnings.clear()
    dialog.coord_combo.clear()
    assert dialog._validate() == QtWidgets.QDialog.DialogCode.Rejected
    assert warnings == [
        (
            "No Coordinates",
            "No numeric coordinates that can be broadcast to the data were found.",
        )
    ]

    dialog.close()
    win.close()


def test_itool_divide_by_coord_nonuniform_generated_code(qtbot, accept_dialog) -> None:
    data = xr.DataArray(
        np.arange(12, dtype=float).reshape((3, 4)) + 1.0,
        dims=["x", "y"],
        coords={
            "x": np.array([0.0, 0.4, 1.0]),
            "y": np.arange(4),
            "mesh_current": ("x", [1.0, 2.0, 4.0]),
        },
        name="scan",
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    def _set_dialog_params(dialog: DivideByCoordDialog) -> None:
        assert "x_idx" not in dialog.coord_dims_label.text()
        dialog.coord_combo.setCurrentText("mesh_current")
        dialog.launch_mode_combo.setCurrentText("Replace Current")

    accept_dialog(win.mnb._divide_by_coord, pre_call=_set_dialog_params)

    restored = erlab.interactive.imagetool.slicer.restore_nonuniform_dims(
        win.slicer_area._data.rename(None)
    )
    xarray.testing.assert_identical(restored, (data / data.mesh_current).rename(None))
    assert win.provenance_spec is not None
    display_code = win.provenance_spec.display_code()
    assert display_code is not None
    assert "mesh_current" in display_code
    assert "x_idx" not in display_code
    namespace = {"data": data.copy(deep=True)}
    exec(display_code, {}, namespace)  # noqa: S102
    xarray.testing.assert_identical(
        namespace["derived"].rename(None), (data / data.mesh_current).rename(None)
    )

    win.close()


def test_itool_gaussian_filter_sigma_path(qtbot, accept_dialog) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["x", "y"],
        coords={"x": np.linspace(0.0, 0.04, 5), "y": np.arange(5, dtype=float)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    def _set_normalize_params(dialog: NormalizeDialog) -> None:
        dialog.dim_checks["x"].setChecked(True)

    accept_dialog(win.mnb._normalize, pre_call=_set_normalize_params)
    xarray.testing.assert_identical(win.slicer_area.data, normalize(data, ("x",), 0))

    sigma_literal = "0.015"

    def _set_gaussian_params(dialog: GaussianFilterDialog) -> None:
        dialog.dim_checks["x"].setChecked(True)
        _set_spinbox_text(dialog.sigma_spins["x"], sigma_literal)
        assert dialog.sigma_spins["x"].text() == sigma_literal
        assert dialog.fwhm_spins["x"].text() == "0.035"
        dialog._preview()

    accept_dialog(win.mnb._gaussian_filter, pre_call=_set_gaussian_params)

    xarray.testing.assert_identical(
        win.slicer_area.data,
        erlab.analysis.image.gaussian_filter(data, sigma={"x": float(sigma_literal)}),
    )

    win.mnb._reset_filters()
    xarray.testing.assert_identical(win.slicer_area.data, data)

    accept_dialog(
        win.mnb._gaussian_filter,
        pre_call=_set_gaussian_params,
        accept_call=lambda d: d.reject(),
    )
    xarray.testing.assert_identical(win.slicer_area.data, data)

    win.close()


def test_itool_gaussian_filter_fwhm_path_and_code(qtbot) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["x", "y"],
        coords={"x": np.linspace(0.0, 0.04, 5), "y": np.arange(5, dtype=float)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    dialog = GaussianFilterDialog(win.slicer_area)
    qtbot.addWidget(dialog)

    dialog.dim_checks["x"].setChecked(True)
    _set_spinbox_text(dialog.fwhm_spins["x"], "0.035")

    sigma_literal = dialog.sigma_spins["x"].text()
    assert sigma_literal == "0.015"
    assert dialog.fwhm_spins["x"].text() == "0.035"

    code = dialog.make_code()
    assert 'sigma={"x": 0.015}' in code

    xarray.testing.assert_identical(
        dialog.process_data(data),
        erlab.analysis.image.gaussian_filter(data, sigma={"x": float(sigma_literal)}),
    )

    dialog.close()
    win.close()


def test_itool_gaussian_filter_disables_unsupported_dimensions(qtbot) -> None:
    data = xr.DataArray(
        np.arange(125).reshape((1, 5, 5, 5)).astype(float),
        dims=["x", "y", "z", "t"],
        coords={
            "x": [0.0],
            "y": np.full(5, 0.5),
            "z": np.array([0.0, 0.2, 0.5, 0.7, 1.0]),
            "t": np.arange(5, dtype=float),
        },
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    dialog = GaussianFilterDialog(win.slicer_area)
    qtbot.addWidget(dialog)

    assert not dialog.dim_checks["x"].isEnabled()
    assert "at least two coordinate values" in dialog.dim_checks["x"].toolTip()
    assert not dialog.dim_checks["y"].isEnabled()
    assert "constant coordinates" in dialog.dim_checks["y"].toolTip()
    assert not dialog.dim_checks["z"].isEnabled()
    assert "uniformly spaced coordinates" in dialog.dim_checks["z"].toolTip()

    assert dialog.dim_checks["t"].isEnabled()
    assert not dialog.sigma_spins["t"].isEnabled()
    assert not dialog.fwhm_spins["t"].isEnabled()

    dialog.close()
    win.close()


def test_itool_auto_chunk(qtbot) -> None:
    data = xr.DataArray(
        np.arange(100).reshape((10, 10)).astype(float),
        dims=["x", "y"],
        coords={"x": np.arange(10), "y": np.arange(10)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    # Auto chunk
    win.slicer_area._auto_chunk()
    assert win.slicer_area._data.chunks is not None


def test_itool_chunk(qtbot, accept_dialog) -> None:
    data = xr.DataArray(
        np.arange(100).reshape((10, 10)).astype(float),
        dims=["x", "y"],
        coords={"x": np.arange(10), "y": np.arange(10)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    # Test chunk dialog
    def _set_dialog_params(dialog: erlab.interactive.utils.ChunkEditDialog) -> None:
        dialog.table.item(0, 2).setText("4")
        dialog.table.item(1, 2).setText("5")

    with qtbot.wait_signal(win.slicer_area.sigDataChanged):
        accept_dialog(win.slicer_area._edit_chunks, pre_call=_set_dialog_params)

    # Check if the data is chunked
    assert win.slicer_area._data.chunks is not None
    assert win.slicer_area._data.chunks == ((4, 4, 2), (5, 5))

    win.close()
