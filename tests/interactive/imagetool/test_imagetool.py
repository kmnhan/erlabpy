import copy
import logging
import tempfile
import types
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
from erlab.interactive.derivative import DerivativeTool
from erlab.interactive.fermiedge import GoldTool, ResolutionTool
from erlab.interactive.imagetool import ImageTool, itool
from erlab.interactive.imagetool.controls import ItoolColormapControls
from erlab.interactive.imagetool.core import (
    _AssociatedCoordsDialog,
    _CursorColorCoordDialog,
    _parse_input,
    _PolyROIEditDialog,
)
from erlab.interactive.imagetool.dialogs import (
    AssignCoordsDialog,
    AverageDialog,
    CropDialog,
    CropToViewDialog,
    EdgeCorrectionDialog,
    NormalizeDialog,
    ROIMaskDialog,
    ROIPathDialog,
    RotationDialog,
    SymmetrizeDialog,
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
        if data.ndim == 2:
            assert main_image.get_selection_code(placeholder="") == ""
        else:
            assert main_image.get_selection_code(placeholder="") == ".qsel(beta=2.0)"

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
        for action in main_image.vb.menu.actions():
            if action.text().startswith("goldtool"):
                action.text().endswith("(Crop)")

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
    assert copied == [".sel(x=slice(1.0, 3.0), y=slice(0.0, 2.0))"]
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
    assert copied == [".sel(x=slice(3.0, 1.0), y=slice(2.0, 0.0))"]
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
    assert '**{"k-space": 2.0}' in code

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
    data = _TEST_DATA["3D"].copy()
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    main_image = win.slicer_area.images[0]

    win.slicer_area.add_cursor()
    win.slicer_area.set_value(axis=2, value=1.0, cursor=0)
    win.slicer_area.set_value(axis=2, value=3.0, cursor=1)

    expected = main_image._plot_code_multicursor()
    copied: dict[str, str] = {}

    def _copy(arg: str) -> None:
        copied["text"] = arg

    monkeypatch.setattr(erlab.interactive.utils, "copy_to_clipboard", _copy)

    main_image.copy_matplotlib_code()
    assert copied["text"] == expected
    assert "beta=[1.0, 3.0]" in expected

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
    slicer.array_slicer._cursor_color_params = ("x", "x", "coolwarm", False, 0.0, 1.0)
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
    slicer.array_slicer._cursor_color_params = ("x", "temp", "viridis", True, 0.2, 0.8)
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


def test_manual_cursor_colors_disable_coord_updates(qtbot) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)),
        dims=["x", "y"],
        coords={"x": np.arange(5), "y": np.arange(5)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    slicer = win.slicer_area
    slicer.array_slicer._cursor_color_params = ("x", "x", "plasma", False, 0.0, 1.0)
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
    slicer.array_slicer._cursor_color_params = ("x", "temp", "magma", True, 0.1, 0.9)

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
        "x",
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

    win.close()


def test_parse_data() -> None:
    with pytest.raises(
        TypeError,
        match=r"Unsupported input type str. Expected DataArray, Dataset, DataTree, "
        r"numpy array, or a list of DataArray or numpy arrays.",
    ):
        erlab.interactive.imagetool.core._parse_input("string")


def test_itool_load(qtbot, move_and_compare_values, accept_dialog) -> None:
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
        with qtbot.wait_signal(dialog._sigCodeCopied):
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
    qtbot.wait_until(
        lambda: not win.slicer_area.main_image.is_guidelines_visible, timeout=1000
    )

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
        dialog.new_window_check.setChecked(False)

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
        dialog.new_window_check.setChecked(False)

    accept_dialog(win.mnb._crop_to_view, pre_call=_set_dialog_params)
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
        with qtbot.wait_signal(dialog._sigCodeCopied):
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
        with qtbot.wait_signal(dialog._sigCodeCopied):
            dialog.copy_button.click()
        dialog.new_window_check.setChecked(False)

    accept_dialog(win.mnb._crop, pre_call=_set_dialog_params)
    xarray.testing.assert_allclose(
        win.slicer_area._data, data.sel(x=slice(2.0, 4.0), y=slice(0.0, 3.0))
    )
    assert pyperclip.paste() == ".sel(x=slice(2.0, 4.0))"

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
        act for act in plot_item.vb.menu.actions() if act.text() == "Add Polygon ROI"
    )
    add_roi_action.trigger()

    assert len(plot_item._roi_list) == 1
    roi = plot_item._roi_list[0]

    roi_menu = roi.getMenu()
    assert {
        "Edit ROI...",
        "Slice Along ROI Path",
        "Mask Data with ROI",
    }.issubset({act.text() for act in roi_menu.actions()})

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
        dialog.new_window_check.setChecked(False)

    accept_dialog(win.mnb._average, pre_call=_set_dialog_params)
    xarray.testing.assert_identical(
        win.slicer_area._data.rename(None), data.qsel.average("x")
    )

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
        dialog._dim_combo.setCurrentIndex(2)
        dialog._center_spin.setValue(2.0)
        with qtbot.wait_signal(dialog._sigCodeCopied):
            dialog.copy_button.click()
        dialog.new_window_check.setChecked(False)

    accept_dialog(win.mnb._symmetrize, pre_call=_set_dialog_params)
    xarray.testing.assert_identical(
        win.slicer_area._data.rename(None),
        erlab.analysis.transform.symmetrize(data, "z", center=2),
    )

    assert pyperclip.paste() == 'era.transform.symmetrize(, dim="z", center=2.0)'
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
            dialog.new_window_check.setChecked(False)

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
        dialog.new_window_check.setChecked(False)

    accept_dialog(win.mnb._assign_coords, pre_call=_set_dialog_params, timeout=10.0)
    np.testing.assert_allclose(win.slicer_area._data.t.values, np.arange(3) + 1.0)


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
