import collections.abc
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
import erlab.interactive.imagetool._highdim as imagetool_highdim
import erlab.interactive.imagetool._itool as itool_mod
import erlab.interactive.imagetool._mainwindow as imagetool_mainwindow
import erlab.interactive.imagetool.dialogs as imagetool_dialogs
import erlab.interactive.imagetool.manager._server as imagetool_manager_server
import erlab.interactive.imagetool.viewer_state as imagetool_viewer_state
from erlab.interactive._figurecomposer import FigureOperationKind, FigureOperationState
from erlab.interactive._figurecomposer._exceptions import (
    FigureComposerPlotSlicesSelectionError,
)
from erlab.interactive.derivative import DerivativeTool, dtool
from erlab.interactive.fermiedge import GoldTool, ResolutionTool
from erlab.interactive.imagetool import ImageTool, itool, provenance
from erlab.interactive.imagetool._viewer_dialogs import (
    _AssociatedCoordsDialog,
    _CursorColorCoordDialog,
)
from erlab.interactive.imagetool.controls import (
    ItoolBinningControls,
    ItoolColormapControls,
    ItoolCrosshairControls,
)
from erlab.interactive.imagetool.dialogs import (
    AggregateDialog,
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
    LeadingEdgeDialog,
    NormalizeDialog,
    RenameDimsCoordsDialog,
    ROIMaskDialog,
    ROIPathDialog,
    RotationDialog,
    SelectionDialog,
    SortByDialog,
    SwapDimsDialog,
    SymmetrizeDialog,
    SymmetrizeNfoldDialog,
    ThinDialog,
)
from erlab.interactive.imagetool.plot_items import ItoolPlotItem, _PolyROIEditDialog
from erlab.interactive.imagetool.slicer import ArraySlicerState
from erlab.interactive.imagetool.viewer import ImageSlicerArea
from erlab.interactive.imagetool.viewer_state import (
    _parse_input,
    _SelectDataArraysDialog,
)
from erlab.io.dataloader import LoaderBase

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def isolate_pyperclip(monkeypatch) -> None:
    clipboard_text = ""

    def copy_to_memory(content: object) -> None:
        nonlocal clipboard_text
        clipboard_text = str(content)

    def paste_from_memory() -> str:
        return clipboard_text

    monkeypatch.setattr(pyperclip, "copy", copy_to_memory)
    monkeypatch.setattr(pyperclip, "paste", paste_from_memory)


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


def _high_dimensional_data(
    dtype: type[np.generic] | type[float] = float,
) -> xr.DataArray:
    shape = (2, 3, 4, 5, 6)
    return xr.DataArray(
        np.arange(np.prod(shape), dtype=dtype).reshape(shape),
        dims=("scan", "pol", "z", "y", "x"),
        coords={
            dim: np.arange(size, dtype=float)
            for dim, size in zip(("scan", "pol", "z", "y", "x"), shape, strict=True)
        },
        name="high_dimensional",
    )


def _press_alt(monkeypatch):
    """Pretend that the Alt/Option key is currently pressed."""
    monkeypatch.setattr(
        QtWidgets.QApplication,
        "queryKeyboardModifiers",
        lambda *_args: QtCore.Qt.KeyboardModifier.AltModifier,
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
        row.step_check.setChecked(False)
        row.width_check.setChecked(False)


def test_operation_backed_transform_dialogs_use_base_make_code() -> None:
    custom_make_code_dialogs = [
        name
        for name, value in vars(imagetool_dialogs).items()
        if isinstance(value, type)
        and issubclass(value, imagetool_dialogs.DataTransformDialog)
        and value is not imagetool_dialogs.DataTransformDialog
        and "make_code" in value.__dict__
    ]

    assert custom_make_code_dialogs == []
    assert "make_code" not in GaussianFilterDialog.__dict__
    assert "make_code" not in NormalizeDialog.__dict__


def test_operation_backed_dialog_empty_operation_edges(qtbot, monkeypatch) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["x", "y"],
        coords={"x": np.arange(5, dtype=float), "y": np.arange(5, dtype=float)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    filter_dialog = imagetool_dialogs.DataFilterDialog(win.slicer_area)
    qtbot.addWidget(filter_dialog)
    assert filter_dialog.filter_operation() is None
    assert filter_dialog.filter_operations() == []

    def _raise_expression_code(*_args, **_kwargs) -> str:
        raise RuntimeError("cannot emit")

    monkeypatch.setattr(
        erlab.interactive.imagetool.provenance,
        "operations_expression_code",
        _raise_expression_code,
    )
    assert filter_dialog.make_code() == ""

    aggregate_dialog = AggregateDialog(win.slicer_area)
    qtbot.addWidget(aggregate_dialog)
    for check in aggregate_dialog.dim_checks.values():
        check.setChecked(False)
    with pytest.raises(ValueError, match="No dimensions selected"):
        aggregate_dialog.source_transform_operation()

    coarsen_dialog = CoarsenDialog(win.slicer_area)
    qtbot.addWidget(coarsen_dialog)
    with pytest.raises(ValueError, match="No dimensions selected"):
        coarsen_dialog.source_transform_operation()

    thin_dialog = ThinDialog(win.slicer_area)
    qtbot.addWidget(thin_dialog)
    thin_dialog.global_radio.setChecked(True)
    thin_dialog.global_spin.setValue(1)
    with pytest.raises(ValueError, match="No thinning requested"):
        thin_dialog.source_transform_operation()
    thin_dialog.per_dim_radio.setChecked(True)
    for spin in thin_dialog.factor_spins.values():
        spin.setValue(1)
    with pytest.raises(ValueError, match="No thinning requested"):
        thin_dialog.source_transform_operation()

    normalize_dialog = NormalizeDialog(win.slicer_area)
    qtbot.addWidget(normalize_dialog)
    assert normalize_dialog.filter_operation() is None
    xr.testing.assert_identical(normalize_dialog.process_data(data), data)

    gaussian_dialog = GaussianFilterDialog(win.slicer_area)
    qtbot.addWidget(gaussian_dialog)
    assert gaussian_dialog.filter_operation() is None

    swap_dialog = SwapDimsDialog(win.slicer_area)
    qtbot.addWidget(swap_dialog)
    with pytest.raises(ValueError, match="No dimensions changed"):
        swap_dialog.source_transform_operation()

    rename_dialog = RenameDimsCoordsDialog(win.slicer_area)
    qtbot.addWidget(rename_dialog)
    with pytest.raises(ValueError, match="No names changed"):
        rename_dialog.source_transform_operation()

    win.close()


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


def test_itool_all_nan_image_updates_do_not_warn(qtbot) -> None:
    data = xr.DataArray(np.full((4, 4), np.nan), dims=["x", "y"])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        win = itool(data, execute=False)
        qtbot.addWidget(win)

        image = win.slicer_area.main_image.slicer_data_items[0]
        image.quickMinMax()
        image.getHistogram()
        image.updateImage()

        controls = ItoolCrosshairControls(win.slicer_area)
        qtbot.addWidget(controls)
        controls.update_content()

        win.slicer_area.refresh_all()

        image_item = erlab.interactive.utils.xImageItem()
        image_item.setDataArray(data)
        image_item.quickMinMax()
        image_item.getHistogram()

    assert [
        warning
        for warning in caught
        if issubclass(warning.category, RuntimeWarning)
        and "All-NaN" in str(warning.message)
    ] == []


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


def test_multicursor_restore_updates_cursor_combo(qtbot) -> None:
    win = itool(_TEST_DATA["2D"], execute=False)
    qtbot.addWidget(win)
    win.slicer_area.add_cursor()
    win.slicer_area.add_cursor()
    win.slicer_area.set_bin(0, 3)
    win.slicer_area.set_bin(1, 5)

    restored = ImageTool.from_dataset(win.to_dataset())
    qtbot.addWidget(restored)
    cursor_ctrl = restored.docks[0].widget().layout().itemAt(0).widget()
    bin_ctrl = restored.docks[2].widget().layout().itemAt(0).widget()
    assert isinstance(cursor_ctrl, ItoolCrosshairControls)
    assert isinstance(bin_ctrl, ItoolBinningControls)

    assert restored.slicer_area.n_cursors == 3
    assert cursor_ctrl.cb_cursors.count() == restored.slicer_area.n_cursors
    assert cursor_ctrl.cb_cursors.currentIndex() == restored.slicer_area.current_cursor
    assert [spin.value() for spin in bin_ctrl.spins] == [3, 5]

    restored.close()
    win.close()


def test_cursor_combo_update_colors_resyncs_stale_count(qtbot) -> None:
    win = itool(_TEST_DATA["2D"], execute=False)
    qtbot.addWidget(win)
    cursor_ctrl = win.docks[0].widget().layout().itemAt(0).widget()
    assert isinstance(cursor_ctrl, ItoolCrosshairControls)

    win.slicer_area.add_cursor()
    win.slicer_area.add_cursor()
    with QtCore.QSignalBlocker(cursor_ctrl.cb_cursors):
        cursor_ctrl.cb_cursors.clear()
        cursor_ctrl.cb_cursors.addItem("stale")

    cursor_ctrl.update_colors()
    assert cursor_ctrl.cb_cursors.count() == 3
    assert cursor_ctrl.cb_cursors.currentIndex() == win.slicer_area.current_cursor
    assert cursor_ctrl.cb_cursors.isEnabled()
    assert cursor_ctrl.btn_rem.isEnabled()

    win.slicer_area.state = {
        **win.slicer_area.state,
        "cursor_colors": ["#cccccc"],
        "current_cursor": 0,
        "slice": {
            **win.slicer_area.array_slicer.state,
            "bins": [[1, 1]],
            "indices": [[2, 2]],
            "values": [[2, 2]],
        },
    }
    with QtCore.QSignalBlocker(cursor_ctrl.cb_cursors):
        cursor_ctrl.cb_cursors.addItem("stale")

    cursor_ctrl.update_colors()
    assert cursor_ctrl.cb_cursors.count() == 1
    assert cursor_ctrl.cb_cursors.currentIndex() == 0
    assert not cursor_ctrl.cb_cursors.isEnabled()
    assert not cursor_ctrl.btn_rem.isEnabled()

    cursor_ctrl.addCursor()
    assert cursor_ctrl.cb_cursors.count() == 2
    assert cursor_ctrl.cb_cursors.isEnabled()
    assert cursor_ctrl.btn_rem.isEnabled()

    cursor_ctrl.remCursor()
    assert cursor_ctrl.cb_cursors.count() == 1
    assert not cursor_ctrl.cb_cursors.isEnabled()
    assert not cursor_ctrl.btn_rem.isEnabled()

    win.close()


def test_itool_dataset_metadata_fields_roundtrip(qtbot, tmp_path: pathlib.Path) -> None:
    file_path = tmp_path / "scan.h5"
    data = xr.DataArray(
        np.arange(25, dtype=np.float32).reshape((5, 5)),
        dims=["alpha", "eV"],
        coords={
            "alpha": np.arange(5, dtype=float),
            "eV": np.arange(5, dtype=float),
            "temperature": ("alpha", np.linspace(10.0, 50.0, 5)),
        },
        name="scan",
    )
    data.to_netcdf(file_path, engine="h5netcdf")

    operation = provenance.NormalizeOperation(dims=("alpha",), mode="min")
    expected_display = operation.apply(data, parent_data=data)
    provenance_spec = provenance.full_data()
    win = ImageTool(
        data,
        file_path=file_path,
        load_func=(xr.load_dataarray, {"engine": "h5netcdf"}, 0),
    )
    qtbot.addWidget(win)
    win.setWindowTitle("saved scan")
    win.set_provenance_spec(provenance_spec)

    area = win.slicer_area
    area.add_cursor()
    area.add_cursor()
    area.set_current_cursor(2)
    area.set_index(0, 4, cursor=2)
    area.set_index(1, 3, cursor=2)
    area.set_bin(0, 3, cursor=2)
    area.set_bin(1, 5, cursor=2)
    area.array_slicer.snap_to_data = True
    area.array_slicer.twin_coord_names = {"temperature"}
    area.array_slicer._cursor_color_params = (
        ("alpha",),
        "temperature",
        "viridis",
        True,
        0.0,
        1.0,
    )
    area._refresh_cursor_colors(tuple(range(area.n_cursors)), None)
    area.set_manual_limits({"alpha": [1.0, 4.0], "eV": [0.0, 3.0]})
    area.set_axis_inverted("alpha", True)
    area.set_colormap(
        "viridis",
        gamma=1.7,
        reverse=True,
        high_contrast=True,
        zero_centered=True,
        levels_locked=True,
    )
    area.main_image.set_guidelines(3)
    area.main_image._guidelines_items[0].setAngle(60.0)
    area.main_image.add_roi()
    area.apply_filter_operation(operation)

    ds = win.to_dataset()
    saved_state = json.loads(ds.attrs["itool_state"])
    assert set(ds.attrs) == {
        "itool_state",
        "itool_title",
        "itool_name",
        "itool_window_state",
        "erlab_version",
        "itool_provenance_spec",
    }
    assert ds.attrs["itool_title"] == "saved scan"
    assert ds.attrs["itool_name"] == "scan"
    assert ds.attrs["erlab_version"] == erlab.__version__
    assert json.loads(ds.attrs["itool_window_state"]).keys() >= {
        "geometry",
        "rect",
        "visible",
    }
    assert json.loads(ds.attrs["itool_provenance_spec"]) == (
        provenance_spec.model_dump(mode="json")
    )
    assert set(saved_state) == {
        "color",
        "slice",
        "current_cursor",
        "manual_limits",
        "axis_inversions",
        "splitter_sizes",
        "file_path",
        "load_func",
        "cursor_colors",
        "controls_visible",
        "plotitem_states",
        "filter_operation",
    }
    assert set(saved_state["color"]) == {
        "cmap",
        "gamma",
        "reverse",
        "high_contrast",
        "zero_centered",
        "levels_locked",
        "levels",
    }
    assert set(saved_state["slice"]) == {
        "dims",
        "bins",
        "indices",
        "values",
        "snap_to_data",
        "twin_coord_names",
        "cursor_color_params",
    }

    saved_color = saved_state["color"]
    assert saved_color["cmap"] == "viridis"
    assert saved_color["gamma"] == 1.7
    assert saved_color["reverse"] is True
    assert saved_color["high_contrast"] is True
    assert saved_color["zero_centered"] is True
    assert saved_color["levels_locked"] is True
    assert all(isinstance(level, float) for level in saved_color["levels"])
    assert saved_state["slice"] == {
        "dims": ["alpha", "eV"],
        "bins": [[1, 1], [1, 1], [3, 5]],
        "indices": [[2, 2], [2, 2], [4, 3]],
        "values": [[2.0, 2.0], [2.0, 2.0], [4.0, 3.0]],
        "snap_to_data": True,
        "twin_coord_names": ["temperature"],
        "cursor_color_params": [
            ["alpha"],
            "temperature",
            "viridis",
            True,
            0.0,
            1.0,
        ],
    }
    assert saved_state["current_cursor"] == 2
    assert saved_state["manual_limits"] == {"alpha": [1.0, 4.0], "eV": [0.0, 3.0]}
    assert saved_state["axis_inversions"] == {"alpha": True}
    assert saved_state["splitter_sizes"] == area.splitter_sizes
    assert saved_state["file_path"] == str(file_path)
    assert saved_state["load_func"][0].endswith(":load_dataarray")
    assert saved_state["load_func"][1:] == [{"engine": "h5netcdf"}, 0]
    assert len(saved_state["cursor_colors"]) == 3
    assert len(set(saved_state["cursor_colors"])) > 1
    assert saved_state["controls_visible"] is True
    assert saved_state["filter_operation"] == operation.model_dump(mode="json")
    assert len(saved_state["plotitem_states"]) == len(area.axes)
    for plotitem_state in saved_state["plotitem_states"]:
        assert plotitem_state.keys() >= {
            "vb_aspect_locked",
            "vb_autorange",
            "roi_states",
        }
    assert saved_state["plotitem_states"][0]["guideline_state"] == {
        "count": 3,
        "angle": -30.0,
        "offset": [4.0, 3.0],
        "follow_cursor": True,
    }
    assert saved_state["plotitem_states"][0]["roi_states"]

    restored = ImageTool.from_dataset(ds)
    qtbot.addWidget(restored)
    restored_area = restored.slicer_area
    cursor_ctrl = restored.docks[0].widget().layout().itemAt(0).widget()
    color_ctrl = restored.docks[1].widget().layout().itemAt(0).widget()
    bin_ctrl = restored.docks[2].widget().layout().itemAt(0).widget()
    assert isinstance(cursor_ctrl, ItoolCrosshairControls)
    assert isinstance(color_ctrl, ItoolColormapControls)
    assert isinstance(bin_ctrl, ItoolBinningControls)
    color_ctrl.cb_colormap.load_all()

    assert restored.windowTitle() == "saved scan"
    assert restored.provenance_spec is not None
    assert restored.provenance_spec.model_dump(mode="json") == (
        provenance_spec.model_dump(mode="json")
    )
    xarray.testing.assert_identical(restored_area._data, data)
    xarray.testing.assert_identical(restored_area.data, expected_display)
    assert restored_area.n_cursors == 3
    assert restored_area.current_cursor == 2
    assert [color.name() for color in restored_area.cursor_colors] == (
        saved_state["cursor_colors"]
    )
    assert cursor_ctrl.cb_cursors.count() == 3
    assert cursor_ctrl.cb_cursors.currentIndex() == 2
    assert [spin.value() for spin in bin_ctrl.spins] == [3, 5]
    restored_slice = restored_area.array_slicer.state
    assert restored_slice["dims"] == ("alpha", "eV")
    assert restored_slice["bins"] == [[1, 1], [1, 1], [3, 5]]
    assert restored_slice["indices"] == [[2, 2], [2, 2], [4, 3]]
    assert restored_slice["values"] == [[2.0, 2.0], [2.0, 2.0], [4.0, 3.0]]
    assert restored_slice["snap_to_data"] is True
    assert restored_slice["twin_coord_names"] == ("temperature",)
    assert restored_slice["cursor_color_params"] == (
        ("alpha",),
        "temperature",
        "viridis",
        True,
        0.0,
        1.0,
    )
    assert restored_area.manual_limits == {"alpha": [1.0, 4.0], "eV": [0.0, 3.0]}
    _assert_manual_limits_view_ranges(restored_area, restored_area.manual_limits)
    assert restored_area.axis_inversions == {"alpha": True}
    _assert_dimension_inverted(restored_area, "alpha", True)
    assert restored_area.colormap_properties["cmap"] == "viridis"
    assert restored_area.colormap_properties["gamma"] == pytest.approx(1.7)
    assert restored_area.colormap_properties["reverse"] is True
    assert restored_area.colormap_properties["high_contrast"] is True
    assert restored_area.colormap_properties["zero_centered"] is True
    assert restored_area.levels_locked
    assert restored_area.levels == pytest.approx(saved_state["color"]["levels"])
    assert color_ctrl.cb_colormap.currentText() == "viridis"
    assert color_ctrl.gamma_widget.value() == pytest.approx(1.7)
    assert restored_area.reverse_act.isChecked()
    assert restored_area.high_contrast_act.isChecked()
    assert restored_area.zero_centered_act.isChecked()
    assert restored_area.lock_levels_act.isChecked()
    assert not restored_area._colorbar.isHidden()
    assert restored_area.controls_visible is True
    restored_splitter_sizes = restored_area.splitter_sizes
    assert len(restored_splitter_sizes) == len(saved_state["splitter_sizes"])
    assert [len(sizes) for sizes in restored_splitter_sizes] == [
        len(sizes) for sizes in saved_state["splitter_sizes"]
    ]
    assert restored_area._file_path == file_path
    assert restored_area._load_func is not None
    loader, kwargs, selection = restored_area._load_func
    assert loader is xr.load_dataarray
    assert kwargs == {"engine": "h5netcdf"}
    assert selection == 0
    assert restored_area.reloadable
    assert restored_area.state["filter_operation"] == operation.model_dump(mode="json")
    _assert_guideline_state(
        restored_area.main_image,
        count=3,
        angle=-30.0,
        offset=(4.0, 3.0),
    )
    assert len(restored_area.main_image._roi_list) == 1
    saved_roi_points = [
        tuple(point)
        for point in saved_state["plotitem_states"][0]["roi_states"][0]["points"]
    ]
    restored_roi_points = [
        tuple(point)
        for point in restored_area.main_image._serializable_state["roi_states"][0][
            "points"
        ]
    ]
    assert restored_roi_points == saved_roi_points

    restored.close()
    win.close()


def test_itool_state_schema_guard() -> None:
    assert set(imagetool_viewer_state.ImageSlicerState.__annotations__) == {
        "color",
        "slice",
        "current_cursor",
        "manual_limits",
        "axis_inversions",
        "filter_operation",
        "cursor_colors",
        "controls_visible",
        "file_path",
        "load_func",
        "splitter_sizes",
        "plotitem_states",
    }
    assert set(imagetool_viewer_state.ColorMapState.__annotations__) == {
        "cmap",
        "gamma",
        "reverse",
        "high_contrast",
        "zero_centered",
        "levels_locked",
        "levels",
    }
    assert set(ArraySlicerState.__annotations__) == {
        "dims",
        "bins",
        "indices",
        "values",
        "snap_to_data",
        "twin_coord_names",
        "cursor_color_params",
    }
    assert set(imagetool_viewer_state.PlotItemState.__annotations__) == {
        "vb_aspect_locked",
        "vb_x_inverted",
        "vb_y_inverted",
        "vb_autorange",
        "roi_states",
        "guideline_state",
    }


def test_itool_plotitem_viewbox_metadata_roundtrip(qtbot) -> None:
    data = xr.DataArray(
        np.arange(25, dtype=float).reshape((5, 5)),
        dims=["x", "y"],
        coords={"x": np.arange(5, dtype=float), "y": np.arange(5, dtype=float)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    viewbox = win.slicer_area.main_image.getViewBox()
    viewbox.setAspectLocked(True, ratio=1.5)
    viewbox.enableAutoRange(x=False, y=False)

    ds = win.to_dataset()
    saved_state = json.loads(ds.attrs["itool_state"])
    assert saved_state["plotitem_states"][0]["vb_aspect_locked"] == 1.5
    assert saved_state["plotitem_states"][0]["vb_autorange"] == [False, False]

    restored = ImageTool.from_dataset(ds)
    qtbot.addWidget(restored)
    restored_state = restored.slicer_area.main_image._serializable_state
    assert restored_state["vb_aspect_locked"] == 1.5
    assert restored_state["vb_autorange"] == (False, False)

    restored.close()
    win.close()


def test_itool_state_loader_string_selection_roundtrip_and_reload(
    qtbot,
    monkeypatch,
    tmp_path: pathlib.Path,
) -> None:
    loader_name = "_metadata_test_loader"

    class _MetadataTestLoader(LoaderBase):
        name = loader_name
        description = "Metadata test loader"

        def load(self, identifier, **kwargs):
            return xr.load_dataset(identifier, **kwargs)

    file_path = tmp_path / "scan.h5"
    data = xr.DataArray(
        np.arange(25, dtype=float).reshape((5, 5)),
        dims=["x", "y"],
        coords={"x": np.arange(5, dtype=float), "y": np.arange(5, dtype=float)},
        name="signal",
    )
    data.to_dataset().to_netcdf(file_path, engine="h5netcdf")
    selection = provenance.FileDataSelection(
        kind="dataset_variable",
        value="signal",
    )
    monkeypatch.setitem(erlab.io.loaders._loaders, loader_name, _MetadataTestLoader())
    monkeypatch.setitem(erlab.io.loaders._alias_mapping, loader_name, loader_name)

    win = ImageTool(
        data,
        file_path=file_path,
        load_func=(loader_name, {"engine": "h5netcdf"}, selection),
    )
    qtbot.addWidget(win)
    ds = win.to_dataset()
    saved_state = json.loads(ds.attrs["itool_state"])
    assert saved_state["load_func"] == [
        loader_name,
        {"engine": "h5netcdf"},
        {"kind": "dataset_variable", "value": "signal"},
    ]

    restored = ImageTool.from_dataset(ds)
    qtbot.addWidget(restored)
    assert restored.slicer_area._load_func == (
        loader_name,
        {"engine": "h5netcdf"},
        {"kind": "dataset_variable", "value": "signal"},
    )
    assert restored.slicer_area.reloadable

    updated = (data + 100.0).rename("signal")
    updated.to_dataset().to_netcdf(file_path, engine="h5netcdf")
    with qtbot.wait_signal(restored.slicer_area.sigDataChanged):
        restored.slicer_area.reload()
    xarray.testing.assert_identical(restored.slicer_area.data, updated)

    restored.close()
    win.close()


def test_itool_state_optional_metadata_fields_restore_defaults(qtbot) -> None:
    win = itool(_TEST_DATA["2D"], execute=False)
    qtbot.addWidget(win)
    ds = win.to_dataset()
    state = json.loads(ds.attrs["itool_state"])
    for key in (
        "axis_inversions",
        "controls_visible",
        "splitter_sizes",
        "file_path",
        "load_func",
        "plotitem_states",
        "filter_operation",
    ):
        state.pop(key, None)
    ds.attrs["itool_state"] = json.dumps(state)

    restored = ImageTool.from_dataset(ds)
    qtbot.addWidget(restored)
    assert restored.slicer_area.controls_visible is True
    assert restored.slicer_area.axis_inversions == {}
    assert restored.slicer_area._file_path is None
    assert restored.slicer_area._load_func is None
    assert restored.slicer_area.n_cursors == 1
    assert restored.slicer_area.current_cursor == 0
    assert restored.slicer_area.state["plotitem_states"]

    restored.close()
    win.close()


def test_locked_levels_state_uses_json_scalars(qtbot) -> None:
    data = _TEST_DATA["2D"].astype(np.float32)
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    win.slicer_area.lock_levels(True)
    restored = ImageTool.from_dataset(win.to_dataset())
    qtbot.addWidget(restored)

    assert restored.slicer_area.levels_locked
    assert restored.slicer_area.levels == pytest.approx(win.slicer_area.levels)

    restored.close()
    win.close()


def test_levels_preserve_requested_span_with_nonfinite_cached_limits(qtbot) -> None:
    win = itool(_TEST_DATA["2D"], execute=False)
    qtbot.addWidget(win)
    win.slicer_area.array_slicer.__dict__["limits"] = (np.nan, np.nan)

    win.slicer_area.levels = (-1.0, 1.0)

    assert not win.slicer_area.levels_locked
    assert win.slicer_area.levels == pytest.approx((-1.0, 1.0))

    win.close()


def test_figure_composer_multicursor_line_seeds_normalization_and_colors(
    qtbot,
) -> None:
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

    operation = line_plot.figure_composer_operation(source_name="data")
    assert operation.kind == FigureOperationKind.LINE
    assert operation.line_x == "alpha"
    assert operation.line_selection == {"eV": [1.0, 3.0]}
    assert operation.line_iter_dim == "eV"
    assert operation.line_normalize == "mean"
    assert operation.line_colors == ("#123456", "#654321")
    assert operation.xlim == (1.0, 3.0)

    win.close()


def test_figure_composer_multicursor_line_skips_default_colors(qtbot) -> None:
    data = _TEST_DATA["2D"].copy()
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    line_plot = win.slicer_area.get_axes(1)

    win.slicer_area.add_cursor()
    win.slicer_area.set_value(axis=1, value=1.0, cursor=0)
    win.slicer_area.set_value(axis=1, value=3.0, cursor=1)

    operation = line_plot.figure_composer_operation(source_name="data")
    assert operation.kind == FigureOperationKind.LINE
    assert operation.line_x == "alpha"
    assert operation.line_selection == {"eV": [1.0, 3.0]}
    assert operation.line_iter_dim == "eV"
    assert operation.line_normalize == "none"
    assert operation.line_colors == ()

    win.close()


def test_figure_composer_line_without_multicursor_variation(qtbot) -> None:
    data = _TEST_DATA["2D"].copy()
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    line_plot = win.slicer_area.get_axes(1)

    operation = line_plot.figure_composer_operation(source_name="data")
    assert operation.kind == FigureOperationKind.LINE
    assert operation.line_x == "alpha"
    assert operation.line_selection == {"eV": 2.0}
    assert operation.line_iter_dim is None

    win.close()


def test_figure_composer_multicursor_image_seeds_norm_settings(qtbot) -> None:
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

    operation = main_image.figure_composer_operation(source_name="data")
    assert operation.kind == FigureOperationKind.PLOT_SLICES
    assert operation.transpose is True
    assert operation.same_limits is True
    assert operation.axis == "image"
    assert operation.cmap == "magma_r"
    assert operation.norm_name == "CenteredInversePowerNorm"
    assert operation.norm_gamma == pytest.approx(1.5)
    assert operation.vcenter == pytest.approx(62.0)
    assert operation.halfrange == pytest.approx(62.0)
    assert operation.slice_dim == "beta"
    assert operation.slice_values == (1.0, 2.0)

    win.close()


def test_figure_composer_single_cursor_image_seeds_cut_and_width(qtbot) -> None:
    data = _TEST_DATA["3D"].copy()
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    main_image = win.slicer_area.images[0]

    win.slicer_area.array_slicer.set_bin(0, 2, 3)

    operation = main_image.figure_composer_operation(source_name="data")
    assert operation.kind == FigureOperationKind.PLOT_SLICES
    assert operation.slice_dim == "beta"
    assert operation.slice_values == (2.0,)
    assert operation.slice_width == pytest.approx(3.0)
    assert operation.slice_kwargs == {}
    assert operation.extra_kwargs == {}

    win.close()


def test_figure_composer_operation_updates_keep_independent_state() -> None:
    updates = ItoolPlotItem._figure_composer_operation_updates(
        {
            "xlim": (1.0, 3.0),
            "ylim": (0.5, 2.5),
            "cmap": "magma",
            "gamma": 0.3,
            "norm": "|custom.Norm(dynamic_value)|",
            "unsupported": "|not_literal|",
        }
    )

    assert updates["xlim"] == (1.0, 3.0)
    assert updates["ylim"] == (0.5, 2.5)
    assert updates["cmap"] == "magma"
    assert updates["norm_name"] == "PowerNorm"
    assert updates["norm_gamma"] == pytest.approx(0.3)
    assert updates["extra_kwargs"] == {}


def test_figure_composer_seed_helpers_promote_editable_selection_kwargs(
    qtbot,
) -> None:
    data = _TEST_DATA["3D"].copy()
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    main_image = win.slicer_area.images[0]

    converted = ItoolPlotItem._figure_composer_plain_value(
        {
            "flag": np.bool_(True),
            "plain_int": 3,
            "count": np.int64(2),
            "scale": np.float64(0.25),
            "items": (np.int64(1), np.float64(2.5)),
        }
    )
    assert converted == {
        "flag": True,
        "plain_int": 3,
        "count": 2,
        "scale": 0.25,
        "items": (1, 2.5),
    }

    assert ItoolPlotItem._figure_composer_operation_updates({}) is None
    assert ItoolPlotItem._figure_composer_norm_updates("not a call") is None
    assert ItoolPlotItem._figure_composer_norm_updates("CenteredPowerNorm(1)") is None
    assert ItoolPlotItem._figure_composer_norm_updates("other.Norm(1)") is None
    assert ItoolPlotItem._figure_composer_norm_updates("eplt.PowerNorm(1)") is None
    assert (
        ItoolPlotItem._figure_composer_norm_updates("eplt.CenteredPowerNorm(**{})")
        is None
    )
    assert (
        ItoolPlotItem._figure_composer_norm_updates("eplt.CenteredPowerNorm(foo=bar)")
        is None
    )
    norm_updates = ItoolPlotItem._figure_composer_norm_updates(
        "eplt.CenteredPowerNorm(0.5, halfrange=1.0)"
    )
    assert norm_updates is not None
    assert norm_updates["norm_name"] == "CenteredPowerNorm"
    assert norm_updates["norm_gamma"] == pytest.approx(0.5)
    assert norm_updates["halfrange"] == pytest.approx(1.0)

    operation = main_image._figure_composer_plot_slices_operation(
        source_name="data",
        variable_dim=None,
        dim_order_plot=["alpha", "eV"],
        qsel_kwargs={
            "beta": [1.0, 2.0],
            "beta_width": [0.25, 0.25],
            "temperature": "base",
        },
    )
    assert operation.slice_dim == "beta"
    assert operation.slice_values == (1.0, 2.0)
    assert operation.slice_width == pytest.approx(0.25)
    assert operation.slice_kwargs == {"temperature": "base"}

    varying_width_operation = main_image._figure_composer_plot_slices_operation(
        source_name="data",
        variable_dim="beta",
        dim_order_plot=["alpha", "eV"],
        qsel_kwargs={"beta": [1.0, 2.0], "beta_width": [0.25, 0.5]},
    )
    assert varying_width_operation.slice_dim == "beta"
    assert varying_width_operation.slice_width is None
    assert varying_width_operation.slice_kwargs == {"beta_width": [0.25, 0.5]}

    scalar_width_operation = main_image._figure_composer_plot_slices_operation(
        source_name="data",
        variable_dim="beta",
        dim_order_plot=["alpha", "eV"],
        qsel_kwargs={"beta": 1.0, "beta_width": 0.25},
    )
    assert scalar_width_operation.slice_width == pytest.approx(0.25)
    assert scalar_width_operation.slice_kwargs == {"beta": 1.0}

    unparsed_width_operation = main_image._figure_composer_plot_slices_operation(
        source_name="data",
        variable_dim=None,
        dim_order_plot=["alpha", "eV"],
        qsel_kwargs={"beta": [1.0, 2.0], "beta_width": ["wide"]},
    )
    assert unparsed_width_operation.slice_dim == "beta"
    assert unparsed_width_operation.slice_width is None
    assert unparsed_width_operation.slice_kwargs == {"beta_width": ["wide"]}

    win.slicer_area.add_cursor()
    win.slicer_area.set_value(axis=2, value=1.0, cursor=0)
    win.slicer_area.set_value(axis=2, value=2.0, cursor=1)
    map_selections = main_image._figure_composer_map_selections(
        source_name="data",
        non_display_axes=(2,),
        variable_dim="beta",
    )
    assert len(map_selections) == 2
    assert [selection.qsel["beta"] for selection in map_selections] == [1.0, 2.0]

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
    assert ".qsel.mean((" in expr
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


def test_plot_with_matplotlib_executes_in_manager(qtbot, monkeypatch) -> None:
    data = _TEST_DATA["3D"].copy()
    win = itool(data, execute=False)
    win.slicer_area._in_manager = True
    qtbot.addWidget(win)
    main_image = win.slicer_area.images[0]
    created: list[dict[str, object]] = []
    x_range = (1.0, 3.0)
    y_range = (0.5, 2.5)
    win.slicer_area.set_colormap(cmap="magma", gamma=1.5, reverse=True)
    win.slicer_area.levels = (10.0, 20.0)
    win.slicer_area.lock_levels(True)
    main_image.getViewBox().setRange(xRange=x_range, yRange=y_range, padding=0.0)
    win.slicer_area.manual_limits.clear()

    class _Manager:
        def target_from_slicer_area(self, slicer_area):
            assert slicer_area is win.slicer_area
            return 0

        def _node_for_target(self, target):
            assert target == 0
            return types.SimpleNamespace(uid="n0")

        def _script_input_name_for_node(self, node):
            assert node.uid == "n0"
            return "data_0"

        def create_figure_from_slicer_area(self, slicer_area, **kwargs):
            assert slicer_area is win.slicer_area
            created.append(kwargs)
            return "figure"

    monkeypatch.setattr(
        erlab.interactive.imagetool.manager, "_manager_instance", _Manager()
    )

    main_image.plot_with_matplotlib()
    assert created
    operation = created[0]["operation"]
    from erlab.interactive._figurecomposer import (
        FigureComposerTool,
        FigureOperationKind,
        FigureSourceState,
    )

    assert operation.kind == FigureOperationKind.PLOT_SLICES
    assert operation.sources == ("data_0",)
    assert isinstance(operation.transpose, bool)
    assert isinstance(operation.crop, bool)
    assert operation.xlim == x_range
    assert operation.ylim == y_range
    assert operation.cmap == "magma_r"
    assert operation.norm_name == "PowerNorm"
    assert operation.norm_gamma == pytest.approx(1.5)
    assert operation.vmin == pytest.approx(0.0)
    assert operation.vmax == pytest.approx(124.0)
    assert operation.same_limits is True
    assert "custom_code" not in created[0]

    composer = FigureComposerTool.from_sources(
        {"data_0": data},
        sources=(FigureSourceState(name="data_0", label="data"),),
        operations=(operation,),
        primary_source="data_0",
    )
    qtbot.addWidget(composer)

    win.close()


def test_plot_with_matplotlib_accepts_spaced_selection_dim(qtbot, monkeypatch) -> None:
    data = xr.DataArray(
        np.arange(125).reshape((5, 5, 5)),
        dims=("alpha", "eV", "Track Shift"),
        coords={
            "alpha": np.arange(5),
            "eV": np.arange(5),
            "Track Shift": np.arange(5),
        },
    )
    win = itool(data, execute=False)
    win.slicer_area._in_manager = True
    qtbot.addWidget(win)
    main_image = win.slicer_area.images[0]
    created: list[dict[str, object]] = []
    warnings_shown: list[tuple[QtWidgets.QWidget | None, str, str]] = []

    class _Manager:
        def target_from_slicer_area(self, slicer_area):
            assert slicer_area is win.slicer_area
            return 0

        def _node_for_target(self, target):
            assert target == 0
            return types.SimpleNamespace(uid="n0")

        def _script_input_name_for_node(self, node):
            assert node.uid == "n0"
            return "data_0"

        def create_figure_from_slicer_area(self, slicer_area, **kwargs):
            assert slicer_area is win.slicer_area
            created.append(kwargs)
            return "figure"

    def record_warning(
        parent: QtWidgets.QWidget | None,
        title: str,
        text: str,
    ) -> QtWidgets.QMessageBox.StandardButton:
        warnings_shown.append((parent, title, text))
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(
        erlab.interactive.imagetool.manager, "_manager_instance", _Manager()
    )
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", record_warning)

    main_image.plot_with_matplotlib()

    assert len(created) == 1
    assert warnings_shown == []
    operation = created[0]["operation"]
    assert isinstance(operation, FigureOperationState)
    assert operation.kind == FigureOperationKind.PLOT_SLICES
    assert operation.slice_dim == "Track Shift"
    assert operation.slice_values == (2.0,)
    assert operation.map_selections == ()

    win.close()


def test_figure_composer_operation_reports_uneditable_plot_slices_details(
    qtbot, monkeypatch
) -> None:
    win = itool(_TEST_DATA["3D"], execute=False)
    qtbot.addWidget(win)
    main_image = win.slicer_area.images[0]

    cases = (
        (
            ({("bad", "key"): 0.0}, None, None, set()),
            "Unsupported qsel selection keys: ('bad', 'key')",
        ),
        (
            ({"beta": 0.0}, ["data.qsel(beta=0.0)"], None, {"beta"}),
            "Selection requires per-cursor expressions",
        ),
        (
            (None, None, None, set()),
            "Selection did not produce qsel coordinates",
        ),
    )
    for plan, detail in cases:
        monkeypatch.setattr(
            main_image,
            "_multicursor_selection_plan",
            lambda _plan=plan, **_kwargs: _plan,
        )
        with pytest.raises(FigureComposerPlotSlicesSelectionError) as exc_info:
            main_image.figure_composer_operation(source_name="data")
        assert detail in str(exc_info.value)

    win.close()


def test_plot_with_matplotlib_preserves_state_with_editable_selection_dim(
    qtbot, monkeypatch
) -> None:
    data = xr.DataArray(
        np.arange(125).reshape((5, 5, 5)),
        dims=("alpha", "eV", "Track_Shift"),
        coords={
            "alpha": np.arange(5),
            "eV": np.arange(5),
            "Track_Shift": np.arange(5),
        },
    )
    win = itool(data, execute=False)
    win.slicer_area._in_manager = True
    qtbot.addWidget(win)
    main_image = win.slicer_area.images[0]
    created: list[dict[str, object]] = []
    win.slicer_area.set_colormap(cmap="magma", gamma=0.3)
    win.slicer_area.set_manual_limits({"alpha": [1.0, 3.0], "eV": [0.5, 2.5]})

    class _Manager:
        def target_from_slicer_area(self, slicer_area):
            assert slicer_area is win.slicer_area
            return 0

        def _node_for_target(self, target):
            assert target == 0
            return types.SimpleNamespace(uid="n0")

        def _script_input_name_for_node(self, node):
            assert node.uid == "n0"
            return "data_0"

        def create_figure_from_slicer_area(self, slicer_area, **kwargs):
            assert slicer_area is win.slicer_area
            created.append(kwargs)
            return "figure"

    monkeypatch.setattr(
        erlab.interactive.imagetool.manager, "_manager_instance", _Manager()
    )

    main_image.plot_with_matplotlib()
    operation = created[0]["operation"]

    assert operation.xlim == (1.0, 3.0)
    assert operation.ylim == (0.5, 2.5)
    assert operation.cmap == "magma"
    assert operation.norm_gamma == pytest.approx(0.3)
    assert operation.map_selections == ()
    assert operation.slice_dim == "Track_Shift"
    assert operation.slice_values == (2.0,)
    assert operation.slice_kwargs == {}

    from erlab.interactive._figurecomposer import FigureComposerTool, FigureSourceState

    composer = FigureComposerTool.from_sources(
        {"data_0": data},
        sources=(FigureSourceState(name="data_0", label="data"),),
        operations=(operation,),
        primary_source="data_0",
    )
    qtbot.addWidget(composer)

    dim_combo = composer.findChild(
        QtWidgets.QComboBox, "figureComposerPlotSlicesDimensionCombo"
    )
    values_edit = composer.findChild(
        QtWidgets.QLineEdit, "figureComposerPlotSlicesValuesEdit"
    )
    slice_kwargs_edit = composer.findChild(
        QtWidgets.QLineEdit, "figureComposerPlotSlicesSliceKwargsEdit"
    )
    assert dim_combo is not None
    assert values_edit is not None
    assert slice_kwargs_edit is not None
    assert dim_combo.currentText() == "Track_Shift"
    assert values_edit.text() == "2"
    assert slice_kwargs_edit.text() == ""

    import matplotlib.pyplot as plt

    namespace: dict[str, typing.Any] = {"data_0": data}
    exec(composer.generated_code(), namespace)  # noqa: S102
    assert namespace["fig"].axes
    plt.close(namespace["fig"])

    composer.close()
    win.close()


def test_figure_composer_operation_uses_transposed_source_axis_order(qtbot) -> None:
    data = _TEST_DATA["2D"].copy()
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    main_image = win.slicer_area.images[0]

    win.slicer_area.transpose_main_image()
    win.slicer_area.set_manual_limits({"alpha": [1.0, 3.0], "eV": [0.0, 2.0]})

    operation = main_image.figure_composer_operation(source_name="data")

    assert win.slicer_area._tool_source_parent_data().dims == ("eV", "alpha")
    assert operation.transpose is True
    assert operation.xlim == (0.0, 2.0)
    assert operation.ylim == (1.0, 3.0)

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
            bool,
        ]
    ] = []

    def _capture_open(data, source_spec, *, use_parent_colormap):
        captured.append((data, source_spec, use_parent_colormap))

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
    temp_data, temp_spec, use_parent_colormap = captured[-1]
    xr.testing.assert_identical(temp_data, data.coords["temp"])
    assert temp_spec.kind == "public_data"
    assert temp_spec.operations[-1].op == "select_coord"
    assert use_parent_colormap is False

    coord_menu = _menu_action_by_data(
        profile._associated_coord_menu, ("associated_coord", "plane")
    ).menu()
    assert coord_menu is not None

    _menu_action_by_data(coord_menu, ("associated_coord_full", "plane")).trigger()
    full_data, full_spec, use_parent_colormap = captured[-1]
    xr.testing.assert_identical(full_data, data.coords["plane"])
    assert full_spec.kind == "public_data"
    assert full_spec.operations[-1].op == "select_coord"
    assert use_parent_colormap is False

    _menu_action_by_data(coord_menu, ("associated_coord_profile", "plane")).trigger()
    profile_data, profile_spec, use_parent_colormap = captured[-1]
    xr.testing.assert_identical(profile_data, data.isel(y=1, z=0).coords["plane"])
    assert profile_spec.kind == "selection"
    assert profile_spec.operations[-1].op == "select_coord"
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
        erlab.interactive.imagetool.viewer_state._parse_input("string")


def test_prepare_high_dimensional_data_requires_dialog_when_unavailable() -> None:
    data = _high_dimensional_data()

    with pytest.raises(ValueError, match="Reduce it to four or fewer dimensions"):
        imagetool_viewer_state._prepare_input_data(data, allow_dialog=False)


@pytest.mark.parametrize("dialog_result", [True, False])
def test_prepare_high_dimensional_data_dialog_branches(
    qtbot,
    monkeypatch,
    dialog_result: bool,
) -> None:
    data = _high_dimensional_data()
    operation = provenance.IselOperation(kwargs={"x": 2})
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)

    class _Dialog:
        def __init__(self, dialog_parent, dialog_data) -> None:
            assert dialog_parent is parent
            assert dialog_data is data

        def exec(self) -> bool:
            return dialog_result

        @property
        def result_data(self) -> xr.DataArray:
            return operation.apply(data, parent_data=data)

        def source_operations(self) -> list[provenance.ToolProvenanceOperation]:
            return [operation]

    monkeypatch.setattr(
        imagetool_highdim,
        "_HighDimensionalReductionDialog",
        _Dialog,
    )

    prepared = imagetool_viewer_state._prepare_input_data(data, parent)

    if not dialog_result:
        assert prepared is None
        return

    assert prepared is not None
    assert len(prepared) == 1
    assert prepared[0].source_ndim == data.ndim
    assert prepared[0].source_dtype == np.dtype(data.dtype)
    assert prepared[0].operations == (operation,)
    xarray.testing.assert_identical(prepared[0].data, data.isel(x=2))


@pytest.mark.parametrize("dialog_result", [True, False])
def test_itool_high_dimensional_data_dialog_branches(
    qtbot,
    monkeypatch,
    dialog_result: bool,
) -> None:
    data = _high_dimensional_data()
    operation = provenance.IselOperation(kwargs={"x": 2})

    class _Dialog:
        def __init__(self, _parent, dialog_data) -> None:
            assert dialog_data is data

        def exec(self) -> bool:
            return dialog_result

        @property
        def result_data(self) -> xr.DataArray:
            return operation.apply(data, parent_data=data)

        def source_operations(self) -> list[provenance.ToolProvenanceOperation]:
            return [operation]

    monkeypatch.setattr(
        imagetool_highdim,
        "_HighDimensionalReductionDialog",
        _Dialog,
    )

    result = itool(data, execute=False)

    if not dialog_result:
        assert result is None
        return

    assert isinstance(result, ImageTool)
    qtbot.addWidget(result)
    xarray.testing.assert_identical(result.slicer_area.data, data.isel(x=2))
    assert result.slicer_area._load_preparation_operations == (operation,)
    result.close()


@pytest.mark.parametrize("dialog_result", [True, False])
def test_show_in_manager_high_dimensional_data_dialog_branches(
    qtbot,
    monkeypatch,
    dialog_result: bool,
) -> None:
    data = _high_dimensional_data()
    operation = provenance.IselOperation(kwargs={"x": 2})
    received: list[tuple[list[xr.DataArray], dict[str, typing.Any]]] = []

    class _Manager(QtWidgets.QWidget):
        def _data_recv(self, input_data, kwargs) -> None:
            received.append((input_data, kwargs))

    manager = _Manager()
    qtbot.addWidget(manager)

    class _Dialog:
        def __init__(self, dialog_parent, dialog_data) -> None:
            assert dialog_parent is manager
            assert dialog_data is data

        def exec(self) -> bool:
            return dialog_result

        @property
        def result_data(self) -> xr.DataArray:
            return operation.apply(data, parent_data=data)

        def source_operations(self) -> list[provenance.ToolProvenanceOperation]:
            return [operation]

    monkeypatch.setattr(
        imagetool_highdim,
        "_HighDimensionalReductionDialog",
        _Dialog,
    )
    monkeypatch.setattr(
        imagetool_manager_server,
        "_direct_manager_for_target",
        lambda _target: manager,
    )

    response = erlab.interactive.imagetool.manager.show_in_manager(data)

    assert response is None
    if not dialog_result:
        assert received == []
        return

    assert len(received) == 1
    input_data, kwargs = received[0]
    xarray.testing.assert_identical(input_data[0], data.isel(x=2))
    assert kwargs["source_input_ndims"] == (data.ndim,)
    assert kwargs["source_input_dtypes"] == (np.dtype(data.dtype),)
    assert kwargs["preparation_operations"] == ((operation,),)


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
        assert win.provenance_spec.replay_stages is not None
        assert any(
            op.op == "qsel_aggregate"
            for stage in win.provenance_spec.replay_stages
            for op in stage.operations
        )

        display_code = win.provenance_spec.display_code()
        assert display_code is not None
        assert "data =" not in display_code
        namespace = _exec_generated_code(display_code, {})
        derived = namespace["derived"]
        assert isinstance(derived, xr.DataArray)
        xarray.testing.assert_identical(
            derived.rename(None),
            data.astype(np.float64).qsel.mean("x").rename(None),
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
            updated.astype(np.float64).qsel.mean("x").rename(None),
        )

    win.close()


def test_itool_file_open_uses_selected_dataset_variable(
    qtbot,
    monkeypatch,
    accept_dialog,
    tmp_path: pathlib.Path,
) -> None:
    first = xr.DataArray(np.zeros((2, 3)), dims=("x", "y"), name="first")
    second = xr.DataArray(
        np.ones((4, 5)),
        dims=("u", "v"),
        coords={"u": np.arange(4), "v": np.arange(5)},
        name="second",
    )
    updated_second = second + 2.0
    selection = erlab.interactive.imagetool.provenance.FileDataSelection(
        kind="dataset_variable",
        value="second",
    )
    file_path = tmp_path / "multi.h5"
    file_path.touch()
    datasets = {
        "current": xr.Dataset({"first": first, "second": second}),
    }

    def _load_multi(_path: str) -> xr.Dataset:
        return datasets["current"]

    def _select_second(data, parent=None):
        assert parent is not None
        return (
            imagetool_viewer_state._PreparedInputData(
                data=data["second"],
                selection=selection,
                source_ndim=data["second"].ndim,
                source_dtype=np.dtype(data["second"].dtype),
            ),
        )

    monkeypatch.setattr(
        erlab.interactive.utils,
        "file_loaders",
        lambda *_args: {"Test xarray (*.h5)": (_load_multi, {})},
    )
    monkeypatch.setattr(
        imagetool_mainwindow, "_select_input_dataarrays", _select_second
    )

    win = itool(np.zeros((2, 2)), execute=False)
    qtbot.addWidget(win)

    def _go_to_file(dialog: QtWidgets.QFileDialog) -> None:
        dialog.setDirectory(str(tmp_path))
        dialog.selectFile(str(file_path))
        dialog.selectNameFilter("Test xarray (*.h5)")
        focused = dialog.focusWidget()
        if isinstance(focused, QtWidgets.QLineEdit):
            focused.setText(file_path.name)

    accept_dialog(lambda: win._open_file(native=False), pre_call=_go_to_file)

    assert isinstance(win, ImageTool)
    xr.testing.assert_identical(win.slicer_area.data, second)
    assert win.slicer_area._load_func is not None
    assert win.slicer_area._load_func[2] == selection

    datasets["current"] = xr.Dataset(
        {
            "inserted": xr.DataArray(np.full((2, 3), 5.0), dims=("x", "y")),
            "second": updated_second,
            "first": first,
        }
    )
    with qtbot.wait_signal(win.slicer_area.sigDataChanged):
        win.slicer_area.reload()
    xr.testing.assert_identical(win.slicer_area.data, updated_second)

    win.close()


def test_itool_file_open_reduces_high_dimensional_data_with_provenance(
    qtbot,
    monkeypatch,
    accept_dialog,
    tmp_path: pathlib.Path,
) -> None:
    data = _high_dimensional_data(np.int64)
    operation = provenance.IselOperation(kwargs={"x": 2})
    expected = data.astype(np.float64).isel(x=2)
    file_path = tmp_path / "high_dimensional.h5"
    data.to_netcdf(file_path, engine="h5netcdf")

    class _ReductionDialog:
        def __init__(self, parent, dialog_data) -> None:
            assert parent is win
            xarray.testing.assert_identical(dialog_data, data)
            self._data = dialog_data

        def exec(self) -> bool:
            return True

        @property
        def result_data(self) -> xr.DataArray:
            return operation.apply(self._data, parent_data=self._data)

        def source_operations(self) -> list[provenance.ToolProvenanceOperation]:
            return [operation]

    monkeypatch.setattr(
        imagetool_highdim,
        "_HighDimensionalReductionDialog",
        _ReductionDialog,
    )
    monkeypatch.setattr(
        erlab.interactive.utils,
        "file_loaders",
        lambda *_args: {
            "Test xarray (*.h5)": (xr.load_dataarray, {"engine": "h5netcdf"})
        },
    )

    win = itool(np.zeros((2, 2)), execute=False)
    qtbot.addWidget(win)

    def _go_to_file(dialog: QtWidgets.QFileDialog) -> None:
        dialog.setDirectory(str(tmp_path))
        dialog.selectFile(str(file_path))
        dialog.selectNameFilter("Test xarray (*.h5)")
        focused = dialog.focusWidget()
        if isinstance(focused, QtWidgets.QLineEdit):
            focused.setText(file_path.name)

    accept_dialog(lambda: win._open_file(native=False), pre_call=_go_to_file)

    xr.testing.assert_identical(win.slicer_area.data, expected)
    assert win.provenance_spec is not None
    assert [
        replay_operation
        for stage in win.provenance_spec.replay_stages
        for replay_operation in stage.operations
    ] == [operation]

    replayed = provenance.replay_file_provenance(win.provenance_spec)
    xarray.testing.assert_identical(replayed, expected)
    display_code = win.provenance_spec.display_code()
    assert display_code is not None
    namespace = _exec_generated_code(display_code, {})
    xarray.testing.assert_identical(namespace["derived"], expected)

    win.close()


@pytest.mark.parametrize("selection", ["all", "cancel"])
def test_itool_file_open_selection_branches(
    qtbot,
    monkeypatch,
    accept_dialog,
    tmp_path: pathlib.Path,
    selection: str,
) -> None:
    first = xr.DataArray(
        np.zeros((2, 3)),
        dims=("x", "y"),
        coords={"x": np.arange(2), "y": np.arange(3)},
        name="first",
    )
    second = xr.DataArray(
        np.ones((4, 5)),
        dims=("u", "v"),
        coords={"u": np.arange(4), "v": np.arange(5)},
        name="second",
    )
    file_path = tmp_path / "multi.h5"
    file_path.touch()

    def _load_multi(_path: str) -> xr.Dataset:
        return xr.Dataset({"first": first, "second": second})

    def _select_data(data, parent=None):
        assert parent is not None
        if selection == "cancel":
            return None
        return (
            imagetool_viewer_state._PreparedInputData(
                data=data["first"],
                selection=0,
                source_ndim=data["first"].ndim,
                source_dtype=np.dtype(data["first"].dtype),
            ),
            imagetool_viewer_state._PreparedInputData(
                data=data["second"],
                selection=1,
                source_ndim=data["second"].ndim,
                source_dtype=np.dtype(data["second"].dtype),
            ),
        )

    monkeypatch.setattr(
        erlab.interactive.utils,
        "file_loaders",
        lambda *_args: {"Test xarray (*.h5)": (_load_multi, {})},
    )
    monkeypatch.setattr(imagetool_mainwindow, "_select_input_dataarrays", _select_data)

    win = itool(np.full((2, 2), -1.0), execute=False)
    qtbot.addWidget(win)

    def _go_to_file(dialog: QtWidgets.QFileDialog) -> None:
        dialog.setDirectory(str(tmp_path))
        dialog.selectFile(str(file_path))
        dialog.selectNameFilter("Test xarray (*.h5)")
        focused = dialog.focusWidget()
        if isinstance(focused, QtWidgets.QLineEdit):
            focused.setText(file_path.name)

    accept_dialog(lambda: win._open_file(native=False), pre_call=_go_to_file)

    if selection == "cancel":
        np.testing.assert_array_equal(win.slicer_area.data.values, -1.0)
    else:
        xr.testing.assert_identical(win.slicer_area.data, first)
        qtbot.wait_until(
            lambda: len(win.slicer_area._associated_tools) == 1, timeout=5000
        )
        child = next(iter(win.slicer_area._associated_tools.values()))
        assert isinstance(child, ImageTool)
        xr.testing.assert_identical(child.slicer_area.data, second)
        child.close()

    win.close()


def test_itool_provenance_reload_rejects_incomplete_or_invalid_replay(
    qtbot,
    tmp_path: pathlib.Path,
) -> None:
    win = itool(xr.DataArray(np.arange(4.0), dims=("x",)), execute=False)
    qtbot.addWidget(win)

    with pytest.raises(RuntimeError, match="cannot be reloaded"):
        win.slicer_area._fetch_for_provenance_reload()

    def _file_source(path: pathlib.Path) -> provenance.FileLoadSource:
        return provenance.FileLoadSource(
            path=path,
            loader_label="Loader",
            loader_text="xarray.load_dataarray",
            kwargs_text="(none)",
            replay_call=provenance.FileReplayCall(
                kind="callable",
                target="xarray.load_dataarray",
                kwargs={},
                selected_index=0,
            ),
            load_code=None,
        )

    missing_file = tmp_path / "missing.h5"
    win.set_provenance_spec(
        provenance.file_load(
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
        provenance.script(
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
        provenance.file_load(
            start_label="Bad selected index",
            seed_code="derived = xr.load_dataarray(source_file)",
            file_load_source=_file_source(source_file).model_copy(
                update={
                    "replay_call": provenance.FileReplayCall(
                        kind="callable",
                        target="xarray.load_dataarray",
                        kwargs={},
                        selected_index=1,
                    )
                }
            ),
        ).append_replay_stage(provenance.full_data())
    )
    assert win.slicer_area.reloadable
    with pytest.raises(IndexError, match="out of range"):
        win.slicer_area._fetch_for_provenance_reload()

    with pytest.raises(TypeError, match="script-only operations"):
        provenance.file_load(
            start_label="Bad replay operation",
            seed_code="derived = xr.load_dataarray(source_file)",
            file_load_source=_file_source(source_file),
            replay_stages=[
                provenance.ReplayStage(
                    source_kind="full_data",
                    operations=[
                        provenance.ScriptCodeOperation(
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


def test_itool_save_preserves_filter_state_and_exports_displayed_data(
    qtbot, accept_dialog
) -> None:
    data = xr.DataArray(
        np.arange(25, dtype=float).reshape((5, 5)),
        dims=["x", "y"],
        coords={"x": np.arange(5), "y": np.arange(5)},
        name="scan",
    )
    operation = erlab.interactive.imagetool.provenance.NormalizeOperation(
        dims=("x",),
        mode="min",
    )
    expected = operation.apply(data, parent_data=data)
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    win.slicer_area.apply_filter_operation(operation)

    ds = win.to_dataset()
    state = json.loads(ds.attrs["itool_state"])
    assert state["filter_operation"] == operation.model_dump(mode="json")
    xarray.testing.assert_identical(
        ds[imagetool_mainwindow._ITOOL_DATA_NAME],
        data.rename(imagetool_mainwindow._ITOOL_DATA_NAME),
    )
    restored = ImageTool.from_dataset(ds)
    qtbot.addWidget(restored)
    xarray.testing.assert_identical(restored.slicer_area.data, expected)
    xarray.testing.assert_identical(restored.slicer_area._data, data)
    display_spec = restored.slicer_area.displayed_provenance_spec(
        erlab.interactive.imagetool.provenance.full_data()
    )
    assert display_spec is not None
    code = display_spec.display_code()
    assert code is not None
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    xarray.testing.assert_identical(namespace["derived"], expected)

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        filename = f"{tmp_dir_name}/filtered.h5"

        def _go_to_file(dialog: QtWidgets.QFileDialog):
            dialog.setDirectory(tmp_dir_name)
            dialog.selectFile(filename)
            focused = dialog.focusWidget()
            if isinstance(focused, QtWidgets.QLineEdit):
                focused.setText("filtered.h5")

        accept_dialog(lambda: win._export_file(native=False), pre_call=_go_to_file)
        xr.testing.assert_equal(
            xr.load_dataarray(filename, engine="h5netcdf"),
            expected,
        )

    restored.close()
    win.close()


def test_saved_filtered_file_data_reloads_by_reapplying_filter(
    qtbot, tmp_path: pathlib.Path
) -> None:
    file_path = tmp_path / "scan.h5"
    data = xr.DataArray(
        np.arange(25, dtype=float).reshape((5, 5)),
        dims=["x", "y"],
        coords={"x": np.arange(5, dtype=float), "y": np.arange(5, dtype=float)},
        name="scan",
    )
    updated = data + 100.0
    data.to_netcdf(file_path, engine="h5netcdf")
    operation = erlab.interactive.imagetool.provenance.GaussianFilterOperation(
        sigma={"x": 1.0}
    )

    win = ImageTool(
        data,
        file_path=file_path,
        load_func=(xr.load_dataarray, {"engine": "h5netcdf"}, 0),
    )
    qtbot.addWidget(win)
    win.slicer_area.apply_filter_operation(operation)
    restored = ImageTool.from_dataset(win.to_dataset())
    qtbot.addWidget(restored)
    assert restored.slicer_area.reloadable

    updated.to_netcdf(file_path, engine="h5netcdf")
    with qtbot.wait_signal(restored.slicer_area.sigDataChanged):
        restored.slicer_area.reload()

    expected = operation.apply(updated, parent_data=updated)
    xarray.testing.assert_identical(restored.slicer_area.data, expected)

    restored.close()
    win.close()


def test_filter_state_restore_does_not_emit_edit_signals(qtbot) -> None:
    data = xr.DataArray(
        np.arange(25, dtype=float).reshape((5, 5)),
        dims=["x", "y"],
        coords={"x": np.arange(5, dtype=float), "y": np.arange(5, dtype=float)},
    )
    operation = erlab.interactive.imagetool.provenance.GaussianFilterOperation(
        sigma={"x": 1.0}
    )
    expected = operation.apply(data, parent_data=data)
    source = itool(data, execute=False)
    target = itool(data.copy(deep=True), execute=False)
    qtbot.addWidget(source)
    qtbot.addWidget(target)
    source.slicer_area.apply_filter_operation(operation)
    state = copy.deepcopy(source.slicer_area.state)
    source_replaced = []
    data_edited = []
    target.slicer_area.sigSourceDataReplaced.connect(source_replaced.append)
    target.slicer_area.sigDataEdited.connect(lambda: data_edited.append(True))

    target.slicer_area.state = state

    assert source_replaced == []
    assert data_edited == []
    xarray.testing.assert_identical(target.slicer_area.data, expected)
    assert target.slicer_area._accepted_filter_provenance_operation == operation

    target.close()
    source.close()


def test_process_only_filter_dialog_is_rejected(qtbot) -> None:
    class OffsetFilterDialog(imagetool_dialogs.DataFilterDialog):
        def process_data(self, data: xr.DataArray) -> xr.DataArray:
            return data + 1

    data = xr.DataArray(
        np.arange(9, dtype=float).reshape((3, 3)),
        dims=("x", "y"),
        coords={"x": np.arange(3), "y": np.arange(3)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    dialog = OffsetFilterDialog(win.slicer_area)
    qtbot.addWidget(dialog)

    with pytest.raises(NotImplementedError, match="filter_operation"):
        dialog._apply_current_filter()
    xarray.testing.assert_identical(win.slicer_area.data, data)
    assert not win.slicer_area.has_active_filter

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
        "axis_inversions": {},
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
            },
            {
                "roi_states": [],
                "vb_aspect_locked": False,
                "vb_autorange": (True, True),
            },
            {
                "roi_states": [],
                "vb_aspect_locked": False,
                "vb_autorange": (True, True),
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
    area.apply_func(lambda d: d + 1, preview=True)
    area.apply_func(None)

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


def test_child_tool_from_gaussian_filtered_itool_keeps_display_provenance(
    qtbot,
) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["alpha", "eV"],
        coords={"alpha": np.arange(5, dtype=float), "eV": np.arange(5, dtype=float)},
    )
    operation = erlab.interactive.imagetool.provenance.GaussianFilterOperation(
        sigma={"alpha": 1.0}
    )
    expected = operation.apply(data, parent_data=data)

    win = itool(data, execute=False)
    qtbot.addWidget(win)
    win.slicer_area.apply_filter_operation(operation)

    win.slicer_area.open_in_meshtool()
    qtbot.wait_until(lambda: len(win.slicer_area._associated_tools) == 1, timeout=5000)
    child = next(iter(win.slicer_area._associated_tools.values()))

    xarray.testing.assert_identical(child.tool_data, expected)
    assert child.input_provenance_spec is not None
    display_code = child.input_provenance_spec.display_code()
    assert display_code is not None
    assert "gaussian_filter" in display_code
    namespace = _exec_generated_code(display_code, {"data": data.copy(deep=True)})
    xarray.testing.assert_identical(namespace["derived"], expected)

    win.close()


def test_image_child_from_gaussian_filtered_itool_keeps_display_provenance(
    qtbot,
) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["alpha", "eV"],
        coords={"alpha": np.arange(5, dtype=float), "eV": np.arange(5, dtype=float)},
    )
    operation = erlab.interactive.imagetool.provenance.GaussianFilterOperation(
        sigma={"alpha": 1.0}
    )
    expected = operation.apply(data, parent_data=data)

    win = itool(data, execute=False)
    qtbot.addWidget(win)
    win.slicer_area.apply_filter_operation(operation)

    win.slicer_area.images[0].open_in_new_window()
    qtbot.wait_until(
        lambda: isinstance(win.slicer_area._associated_tools_list[-1], ImageTool),
        timeout=5000,
    )
    child = typing.cast("ImageTool", win.slicer_area._associated_tools_list[-1])

    xarray.testing.assert_identical(child.slicer_area.data, expected)
    assert child.provenance_spec is not None
    display_code = child.provenance_spec.display_code()
    assert display_code is not None
    namespace = _exec_generated_code(display_code, {"data": data.copy(deep=True)})
    xarray.testing.assert_identical(namespace["derived"], expected)

    child.close()
    win.close()


def test_image_open_in_new_window_preserves_spaced_qsel_dimension(qtbot) -> None:
    data = xr.DataArray(
        np.arange(12.0).reshape(3, 2, 2),
        dims=("Track Shift", "kx", "ky"),
        coords={
            "Track Shift": [0.0, 1.0, 2.0],
            "kx": [0.0, 1.0],
            "ky": [0.0, 1.0],
        },
        name="map",
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    image = win.slicer_area.images[2]
    win.slicer_area.set_value(axis=0, value=1.0, cursor=0)
    expected = image.current_data
    source_spec = image.make_tool_source_spec()
    xarray.testing.assert_identical(
        source_spec.apply(data).rename(None),
        expected.rename(None),
    )

    image.open_in_new_window()
    qtbot.wait_until(
        lambda: isinstance(win.slicer_area._associated_tools_list[-1], ImageTool),
        timeout=5000,
    )
    child = typing.cast("ImageTool", win.slicer_area._associated_tools_list[-1])

    xarray.testing.assert_identical(
        child.slicer_area.data.rename(None),
        expected.rename(None),
    )
    assert child.provenance_spec is not None
    display_code = child.provenance_spec.display_code()
    assert display_code is not None
    assert "qsel({" in display_code
    assert "Track Shift" in display_code
    namespace = _exec_generated_code(display_code, {"data": data.copy(deep=True)})
    xarray.testing.assert_identical(
        namespace["derived"].rename(None),
        expected.rename(None),
    )

    child.close()
    win.close()


def test_child_tool_copy_code_streamlines_noop_source_steps(qtbot) -> None:
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
    squeezed_child.set_source_binding(
        provenance.selection(provenance.SqueezeOperation())
    )

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

    spec = image.make_tool_source_spec()
    sel_kwargs = next(op.decoded_kwargs for op in spec.operations if op.op == "sel")

    assert sel_kwargs == {"alpha": slice(1, 4)}

    win.close()


def test_itool_make_tool_source_spec_uses_index_crop_for_nonuniform_dim(
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

    spec = image.make_tool_source_spec()
    crop_isel_kwargs = [
        op.decoded_kwargs
        for op in spec.operations
        if op.op == "isel" and op.decoded_kwargs == {"x": slice(None, None)}
    ]

    assert crop_isel_kwargs == [{"x": slice(None, None)}]

    win.close()


def test_itool_make_tool_source_spec_falls_back_to_dim_lookup(
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

    spec = image.make_tool_source_spec()
    selection_kwargs = next(
        op.decoded_kwargs
        for op in spec.operations
        if op.op in {"qsel", "isel"} and "beta" in op.decoded_kwargs
    )
    crop_kwargs = next(op.decoded_kwargs for op in spec.operations if op.op == "sel")

    assert selection_kwargs["beta"] == _TEST_DATA["3D"].coords["beta"][2].item()
    assert crop_kwargs == {"alpha": slice(1, 4)}

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


def test_select_dataarrays_dialog_preserves_tree_source_paths(qtbot) -> None:
    first = xr.DataArray(
        np.zeros((2, 3), dtype=np.float32),
        dims=("alpha", "eV"),
        attrs={"long_name": "first map", "units": "counts"},
    )
    second = xr.DataArray(
        np.ones((4, 5), dtype=np.float64),
        dims=("alpha", "eV"),
        attrs={"description": "second branch"},
    )
    tree = xr.DataTree.from_dict(
        {
            "branch_a": xr.Dataset({"signal": first}),
            "branch_b": xr.Dataset({"signal": second, "scalar": xr.DataArray(1)}),
        }
    )

    dialog = _SelectDataArraysDialog(None, tree)
    qtbot.addWidget(dialog)
    selected_data = dialog.selected_dataarrays()

    assert [prepared.selection for prepared in selected_data] == [
        provenance.FileDataSelection(kind="datatree_path", value="/branch_a/signal"),
        provenance.FileDataSelection(kind="datatree_path", value="/branch_b/signal"),
    ]
    assert dialog._tree_widget.topLevelItem(0).text(1) == "branch_a"
    assert dialog._tree_widget.topLevelItem(0).text(2) == "signal"
    assert dialog._tree_widget.topLevelItem(1).text(1) == "branch_b"
    assert dialog._tree_widget.topLevelItem(1).text(2) == "signal"
    xr.testing.assert_identical(
        selected_data[0].data, tree["branch_a"].dataset["signal"]
    )
    xr.testing.assert_identical(
        selected_data[1].data, tree["branch_b"].dataset["signal"]
    )


def test_select_dataarrays_dialog_includes_high_dimensional_variables(qtbot) -> None:
    high = _high_dimensional_data()
    ds = xr.Dataset({"high": high, "scalar": xr.DataArray(1)})

    dialog = _SelectDataArraysDialog(None, ds)
    qtbot.addWidget(dialog)

    assert dialog._tree_widget.topLevelItemCount() == 1
    item = dialog._tree_widget.topLevelItem(0)
    assert item is not None
    assert item.text(1) == "high"
    assert item.text(2) == f"{high.ndim} -> reduce"
    selected_data = dialog.selected_dataarrays()
    assert len(selected_data) == 1
    xr.testing.assert_identical(selected_data[0].data, high.rename("high"))
    assert selected_data[0].selection == provenance.FileDataSelection(
        kind="dataset_variable",
        value="high",
    )


def test_select_dataarrays_dialog_formats_selected_dataarray(
    qtbot, monkeypatch
) -> None:
    ds = xr.Dataset(
        {
            "first": xr.DataArray(
                np.zeros((2, 3)),
                dims=("x", "y"),
                attrs={"long_name": "first image"},
            ),
            "second": xr.DataArray(
                np.ones((4, 5)),
                dims=("u", "v"),
                coords={"u": np.arange(4), "v": np.arange(5)},
                attrs={"units": "arb."},
            ),
        }
    )
    formatted: list[tuple[xr.DataArray, bool, bool, tuple[str, ...]]] = []

    def format_darr_html(
        darr: xr.DataArray,
        *,
        show_size: bool = True,
        show_summary: bool = True,
        additional_info: collections.abc.Iterable[str] | None = None,
    ) -> str:
        formatted.append((darr, show_size, show_summary, tuple(additional_info or ())))
        return f"<p>{darr.name}</p>"

    monkeypatch.setattr(erlab.utils.formatting, "format_darr_html", format_darr_html)

    dialog = _SelectDataArraysDialog(None, ds)
    qtbot.addWidget(dialog)

    first_item = dialog._tree_widget.topLevelItem(0)
    second_item = dialog._tree_widget.topLevelItem(1)
    assert first_item is not None
    assert second_item is not None
    assert not dialog._tree_widget.rootIsDecorated()
    assert dialog._tree_widget.indentation() == 0
    assert dialog._path_tree is None
    assert isinstance(dialog._item_checkbox(second_item), QtWidgets.QCheckBox)
    assert dialog._item_checkbox(second_item).isChecked()
    assert second_item.text(1) == "second"
    assert second_item.text(2) == "2"
    second_label = dialog._tree_widget.itemWidget(second_item, 3)
    assert isinstance(second_label, QtWidgets.QLabel)
    assert "second" not in second_label.text()
    assert "<b>u</b>: 4" in second_label.text()
    assert "<b>v</b>: 5" in second_label.text()
    assert "Size " not in second_label.text()
    assert second_item.text(4) == "160 Bytes"

    dialog._item_checkbox(first_item).setChecked(False)
    dialog._tree_widget.setCurrentItem(second_item)

    selected_data = dialog.selected_dataarrays()
    assert [prepared.selection for prepared in selected_data] == [
        erlab.interactive.imagetool.provenance.FileDataSelection(
            kind="dataset_variable",
            value="second",
        )
    ]
    xr.testing.assert_identical(selected_data[0].data, ds["second"])
    xr.testing.assert_identical(formatted[-1][0], ds["second"])
    assert formatted[-1][1] is False
    assert formatted[-1][2] is False
    assert formatted[-1][3] == ()

    accepted_dialog = _SelectDataArraysDialog(None, ds)
    qtbot.addWidget(accepted_dialog)
    accepted_dialog.accept()
    assert accepted_dialog.result() == QtWidgets.QDialog.DialogCode.Accepted

    dialog._update_details(None)
    assert not dialog._details.toPlainText()
    dialog._update_details(QtWidgets.QTreeWidgetItem(["orphan"]))
    assert not dialog._details.toPlainText()

    dialog._item_checkbox(second_item).setChecked(False)
    dialog.accept()

    assert dialog.result() == QtWidgets.QDialog.DialogCode.Rejected

    tree_dialog = _SelectDataArraysDialog(
        None,
        xr.DataTree.from_dict({"branch": ds}),
    )
    qtbot.addWidget(tree_dialog)

    assert tree_dialog._path_tree is not None
    tree_item = tree_dialog._path_tree.topLevelItem(0)
    assert tree_item is not None
    assert tree_item.data(0, QtCore.Qt.ItemDataRole.UserRole) == "/branch"
    assert tree_item.childCount() == 0
    assert tree_dialog._tree_widget.topLevelItemCount() == 2
    first_tree_child = tree_dialog._tree_widget.topLevelItem(0)
    assert first_tree_child is not None
    assert first_tree_child.text(1) == "branch"
    assert first_tree_child.text(2) == "first"
    tree_label = tree_dialog._tree_widget.itemWidget(first_tree_child, 4)
    assert isinstance(tree_label, QtWidgets.QLabel)
    tree_dialog._tree_widget.setCurrentItem(first_tree_child)
    assert formatted[-1][3] == ()


def test_select_dataarrays_dialog_nests_datatree_paths(qtbot) -> None:
    tree = xr.DataTree.from_dict(
        {
            "/branch_a/sweep_0": xr.Dataset(
                {
                    "signal": xr.DataArray(
                        np.zeros((2, 3)),
                        dims=("alpha", "eV"),
                    )
                }
            ),
            "/branch_a/sweep_1": xr.Dataset(
                {
                    "signal": xr.DataArray(
                        np.ones((4, 5)),
                        dims=("alpha", "eV"),
                    )
                }
            ),
        }
    )

    dialog = _SelectDataArraysDialog(None, tree)
    qtbot.addWidget(dialog)

    assert dialog._path_tree is not None
    branch_item = dialog._path_tree.topLevelItem(0)
    assert branch_item is not None
    assert all(not item.isHidden() for item in dialog._items())
    assert dialog._clear_path_filter_button is not None
    assert not dialog._clear_path_filter_button.isEnabled()
    assert branch_item.text(0) == "branch_a"
    assert branch_item.childCount() == 2
    assert branch_item.data(0, QtCore.Qt.ItemDataRole.UserRole) == "/branch_a"

    sweep_item = branch_item.child(1)
    assert sweep_item is not None
    assert sweep_item.text(0) == "sweep_1"
    assert sweep_item.childCount() == 0
    assert sweep_item.data(0, QtCore.Qt.ItemDataRole.UserRole) == "/branch_a/sweep_1"

    dialog._path_tree.setCurrentItem(sweep_item)
    signal_item = next(item for item in dialog._items() if not item.isHidden())
    assert signal_item is not None
    assert signal_item.text(1) == "branch_a/sweep_1"
    assert signal_item.text(2) == "signal"
    assert dialog._item_checkbox(signal_item).isChecked()
    assert dialog._clear_path_filter_button.isEnabled()

    dialog._uncheck_all()
    assert not dialog._item_checkbox(signal_item).isChecked()
    dialog._check_all()
    assert dialog._item_checkbox(signal_item).isChecked()

    dialog._item_checkbox(signal_item).setChecked(False)

    assert [prepared.selection for prepared in dialog.selected_dataarrays()] == [
        erlab.interactive.imagetool.provenance.FileDataSelection(
            kind="datatree_path",
            value="/branch_a/sweep_0/signal",
        )
    ]

    empty_item = QtWidgets.QTreeWidgetItem(["missing"])
    empty_item.setData(0, QtCore.Qt.ItemDataRole.UserRole, "/missing")
    dialog._filter_for_path(empty_item)
    assert all(item.isHidden() for item in dialog._items())
    assert not dialog._details.toPlainText()

    dialog._clear_path_filter()
    assert all(not item.isHidden() for item in dialog._items())
    assert not dialog._clear_path_filter_button.isEnabled()


def test_select_dataarrays_dialog_collapses_single_child_datatree_paths(qtbot) -> None:
    tree = xr.DataTree.from_dict(
        {
            "/cut_a": xr.Dataset(
                {"spectrum": xr.DataArray(np.zeros((2, 3)), dims=("alpha", "eV"))}
            ),
            "/nested/cut_b": xr.Dataset(
                {"spectrum": xr.DataArray(np.ones((4, 5)), dims=("beta", "eV"))}
            ),
        }
    )

    dialog = _SelectDataArraysDialog(None, tree)
    qtbot.addWidget(dialog)

    assert dialog._path_tree is not None
    path_items = [
        dialog._path_tree.topLevelItem(row)
        for row in range(dialog._path_tree.topLevelItemCount())
    ]
    nested_item = next(item for item in path_items if item.text(0) == "nested/cut_b")

    assert nested_item.childCount() == 0
    assert nested_item.toolTip(0) == "/nested/cut_b"
    dialog._path_tree.setCurrentItem(nested_item)
    visible_items = [item for item in dialog._items() if not item.isHidden()]
    assert len(visible_items) == 1
    assert visible_items[0].text(1) == "nested/cut_b"
    assert visible_items[0].text(2) == "spectrum"
    assert visible_items[0].toolTip(0) == "/nested/cut_b"


@pytest.mark.parametrize(
    ("dialog_result", "selected", "expected"),
    [
        (False, ((xr.DataArray(np.ones((2, 2)), dims=("x", "y")), 0),), None),
        (True, (), None),
        (
            True,
            ((xr.DataArray(np.ones((3, 4)), dims=("u", "v")), 1),),
            (1,),
        ),
    ],
)
def test_select_input_dataarrays_dialog_branches(
    monkeypatch,
    dialog_result: bool,
    selected: tuple[tuple[xr.DataArray, int], ...],
    expected: tuple[int, ...] | None,
) -> None:
    ds = xr.Dataset(
        {
            "first": xr.DataArray(np.zeros((2, 3)), dims=("x", "y")),
            "second": xr.DataArray(np.ones((4, 5)), dims=("u", "v")),
        }
    )

    class _Dialog:
        def __init__(self, parent, data) -> None:
            assert parent is None
            assert data is ds

        def exec(self) -> bool:
            return dialog_result

        def selected_dataarrays(
            self,
        ) -> tuple[imagetool_viewer_state._PreparedInputData, ...]:
            return tuple(
                imagetool_viewer_state._PreparedInputData(
                    data=darr,
                    selection=source_index,
                    source_ndim=darr.ndim,
                    source_dtype=np.dtype(darr.dtype),
                )
                for darr, source_index in selected
            )

    monkeypatch.setattr(imagetool_viewer_state, "_SelectDataArraysDialog", _Dialog)

    result = imagetool_viewer_state._select_input_dataarrays(ds)

    if expected is None:
        assert result is None
    else:
        assert result is not None
        assert [prepared.selection for prepared in result] == list(expected)


def test_parse_input_data_records_dataset_and_datatree_selectors() -> None:
    image = xr.DataArray(np.ones((2, 3)), dims=("x", "y"), name="image")
    ds = xr.Dataset({"image": image})
    tree = xr.DataTree.from_dict({"diag": xr.Dataset({"image": image})})

    dataset_parsed = imagetool_viewer_state._parse_input_data(ds)
    datatree_parsed = imagetool_viewer_state._parse_input_data(tree)

    assert dataset_parsed[0].selection == provenance.FileDataSelection(
        kind="dataset_variable", value="image"
    )
    assert datatree_parsed[0].selection == provenance.FileDataSelection(
        kind="datatree_path", value="/diag/image"
    )


def test_itool_dataset_selection_returns_selected_variable(qtbot, monkeypatch) -> None:
    ds = xr.Dataset(
        {
            "first": xr.DataArray(np.zeros((2, 3)), dims=("x", "y")),
            "second": xr.DataArray(
                np.ones((4, 5)),
                dims=("u", "v"),
                coords={"u": np.arange(4), "v": np.arange(5)},
            ),
        }
    )
    monkeypatch.setattr(
        itool_mod,
        "_select_input_dataarrays",
        lambda _data: (
            imagetool_viewer_state._PreparedInputData(
                data=ds["second"],
                selection=1,
                source_ndim=ds["second"].ndim,
                source_dtype=np.dtype(ds["second"].dtype),
            ),
        ),
    )

    win = itool(ds, execute=False)
    qtbot.addWidget(win)

    assert isinstance(win, ImageTool)
    xr.testing.assert_identical(win.slicer_area.data, ds["second"])
    win.close()


def test_itool_dataset_selection_cancel(monkeypatch) -> None:
    ds = xr.Dataset(
        {
            "first": xr.DataArray(np.zeros((2, 3)), dims=("x", "y")),
            "second": xr.DataArray(
                np.ones((4, 5)),
                dims=("u", "v"),
                coords={"u": np.arange(4), "v": np.arange(5)},
            ),
        }
    )
    monkeypatch.setattr(itool_mod, "_select_input_dataarrays", lambda _data: None)

    assert itool(ds, execute=False) is None


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


def test_itool_ds(qtbot, monkeypatch) -> None:
    data = xr.Dataset(
        {
            "data1d": xr.DataArray(np.arange(5), dims=["x"]),
            "a": xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"]),
            "b": xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"]),
        }
    )
    monkeypatch.setattr(
        itool_mod,
        "_select_input_dataarrays",
        lambda _data: tuple(imagetool_viewer_state._parse_input_data(_data)),
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


def test_linked_swap_axes_skips_targets_without_swapped_dimensions(qtbot) -> None:
    data2d = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["x", "y"],
        coords={"x": np.arange(5, dtype=float), "y": np.arange(5, dtype=float)},
    )
    data1d = data2d.isel(y=0, drop=True)
    wins = itool([data2d, data1d], execute=False, link=True)
    assert isinstance(wins, list)
    assert len(wins) == 2
    for win in wins:
        qtbot.addWidget(win)
    target_dims = wins[1].slicer_area.data.dims
    target_indices = list(wins[1].slicer_area.array_slicer.get_indices(0))
    refreshed_axes: list[tuple[int, ...] | None] = []
    wins[1].slicer_area.sigIndexChanged.connect(
        lambda _cursor, axes: refreshed_axes.append(axes)
    )

    wins[0].slicer_area.set_index(1, 2)

    assert wins[1].slicer_area.array_slicer.get_indices(0) == target_indices
    wins[0].slicer_area.refresh_current((1,))
    assert refreshed_axes == []
    wins[0].slicer_area.refresh_current((0, 1))
    assert refreshed_axes == [(0,)]

    wins[0].slicer_area.manual_limits = {"x": [0.0, 2.0], "y": [1.0, 3.0]}
    wins[0].slicer_area.propagate_limit_change(wins[0].slicer_area.main_image)

    assert wins[1].slicer_area.manual_limits == {"x": [0.0, 2.0]}
    target_slice = wins[1].slicer_area.make_slice_dict()
    assert set(target_slice) == {"x"}
    assert target_slice["x"].start == 0
    assert target_slice["x"].stop == 2

    wins[0].slicer_area.swap_axes(0, 1)

    assert wins[0].slicer_area.data.dims == ("y", "x")
    assert wins[1].slicer_area.data.dims == target_dims
    assert len(wins[1].slicer_area.array_slicer.get_bins(0)) == len(target_dims)

    for win in wins:
        win.close()


def test_linked_axis_arguments_follow_dimension_names(qtbot) -> None:
    data = xr.DataArray(
        np.arange(60).reshape((3, 4, 5)).astype(float),
        dims=["x", "y", "z"],
        coords={
            "x": np.arange(3, dtype=float),
            "y": np.arange(4, dtype=float),
            "z": np.arange(5, dtype=float),
        },
    )
    wins = itool([data, data.transpose("z", "y", "x")], execute=False, link=True)
    assert isinstance(wins, list)
    assert len(wins) == 2
    for win in wins:
        qtbot.addWidget(win)

    wins[0].slicer_area.set_index(0, 2)

    target_area = wins[1].slicer_area
    target_x_axis = target_area.data.dims.index("x")
    assert target_area.array_slicer.get_indices(0)[target_x_axis] == 2

    wins[0].slicer_area.swap_axes(0, 1)

    assert wins[0].slicer_area.data.dims == ("y", "x", "z")
    assert target_area.data.dims == ("z", "x", "y")

    for win in wins:
        win.close()


def test_manual_limits_ignore_dimensions_missing_from_data(qtbot) -> None:
    data = xr.DataArray(
        np.arange(5).astype(float),
        dims=["x"],
        coords={"x": np.arange(5, dtype=float)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    area = win.slicer_area

    area.set_manual_limits({"x": [0.0, 2.0], "missing": [1.0, 3.0]})

    assert area.manual_limits == {"x": [0.0, 2.0]}

    area.manual_limits["missing"] = [1.0, 3.0]
    slice_dict = area.make_slice_dict()

    assert set(slice_dict) == {"x"}

    win.close()


class _SceneDragEvent:
    def __init__(self, scene_pos: QtCore.QPointF) -> None:
        self._scene_pos = scene_pos

    def scenePos(self) -> QtCore.QPointF:
        return self._scene_pos


def _cursor_line_values(image, cursor: int) -> tuple[float, ...]:
    return tuple(line.value() for line in image.cursor_lines[cursor].values())


def _plot_axis_inverted(plot_item, axis: int) -> bool:
    key = "xInverted" if axis == 0 else "yInverted"
    return bool(plot_item.getViewBox().state[key])


def _assert_dimension_inverted(area: ImageSlicerArea, dim: str, inverted: bool) -> None:
    matched = False
    for plot_item in area.axes:
        for axis, axis_dim in enumerate(plot_item.axis_dims_uniform):
            if axis_dim == dim:
                matched = True
                assert _plot_axis_inverted(plot_item, axis) is inverted
    assert matched


def _assert_manual_limits_view_ranges(
    area: ImageSlicerArea, expected_limits: dict[str, list[float]]
) -> None:
    matched_dims: set[str] = set()
    for plot_item in area.axes:
        view_ranges = plot_item.getViewBox().viewRange()
        for axis, axis_dim in enumerate(plot_item.axis_dims_uniform):
            if axis_dim in expected_limits:
                matched_dims.add(axis_dim)
                np.testing.assert_allclose(view_ranges[axis], expected_limits[axis_dim])
    assert matched_dims == set(expected_limits)


def test_axis_inversion_viewbox_shared_undo_redo_and_roundtrip(qtbot) -> None:
    data = xr.DataArray(np.arange(25).reshape((5, 5)).astype(float), dims=["x", "y"])
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    area = win.slicer_area

    area.main_image.getViewBox().invertX(True)

    assert area.axis_inversions == {"x": True}
    _assert_dimension_inverted(area, "x", True)
    _assert_dimension_inverted(area, "y", False)
    assert area.undoable

    area.undo()
    assert area.axis_inversions == {}
    _assert_dimension_inverted(area, "x", False)

    area.redo()
    assert area.axis_inversions == {"x": True}
    _assert_dimension_inverted(area, "x", True)

    restored = ImageTool.from_dataset(win.to_dataset())
    qtbot.addWidget(restored)
    assert restored.slicer_area.axis_inversions == {"x": True}
    _assert_dimension_inverted(restored.slicer_area, "x", True)

    restored.close()
    win.close()


def test_axis_inversion_transposed_manual_limits_roundtrip(qtbot) -> None:
    data = xr.DataArray(
        np.arange(125).reshape((5, 5, 5)).astype(float),
        dims=["x", "y", "z"],
        coords={"x": np.arange(5.0), "y": np.arange(5.0), "z": np.arange(5.0)},
    )
    expected_limits = {"x": [1.0, 3.0], "y": [0.0, 2.0], "z": [2.0, 4.0]}
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    win.slicer_area.set_manual_limits(expected_limits)
    win.slicer_area.set_axis_inverted("x", True)
    win.slicer_area.transpose_main_image()
    assert win.slicer_area.data.dims == ("y", "x", "z")

    ds = win.to_dataset()
    saved_state = json.loads(ds.attrs["itool_state"])
    assert saved_state["manual_limits"] == expected_limits

    restored = ImageTool.from_dataset(ds)
    qtbot.addWidget(restored)

    assert restored.slicer_area.data.dims == ("y", "x", "z")
    assert restored.slicer_area.manual_limits == expected_limits
    _assert_manual_limits_view_ranges(restored.slicer_area, expected_limits)
    _assert_dimension_inverted(restored.slicer_area, "x", True)

    restored.close()
    win.close()


def test_axis_inversion_view_menu_toggles_shared_state(qtbot) -> None:
    data = xr.DataArray(np.arange(25).reshape((5, 5)).astype(float), dims=["x", "y"])
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    menu = win.mnb.menu_dict["invertAxisMenu"]
    win.mnb._populate_invert_axis_menu()
    actions = {action.data(): action for action in menu.actions()}

    y_action = actions["y"]
    assert y_action.objectName() == "itool_invert_axis_1_action"
    assert y_action.isCheckable()

    y_action.trigger()
    assert win.slicer_area.axis_inversions == {"y": True}
    _assert_dimension_inverted(win.slicer_area, "y", True)

    y_action.trigger()
    assert win.slicer_area.axis_inversions == {}
    _assert_dimension_inverted(win.slicer_area, "y", False)

    win.close()


def test_axis_inversion_legacy_plotitem_state_migrates(qtbot) -> None:
    data = xr.DataArray(np.arange(25).reshape((5, 5)).astype(float), dims=["x", "y"])
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    ds = win.to_dataset()
    state = json.loads(ds.attrs["itool_state"])
    state.pop("axis_inversions")
    state["plotitem_states"][0]["vb_x_inverted"] = True
    state["plotitem_states"][2]["vb_y_inverted"] = True
    ds.attrs["itool_state"] = json.dumps(state)

    restored = ImageTool.from_dataset(ds)
    qtbot.addWidget(restored)

    assert restored.slicer_area.axis_inversions == {"x": True, "y": True}
    _assert_dimension_inverted(restored.slicer_area, "x", True)
    _assert_dimension_inverted(restored.slicer_area, "y", True)

    restored.close()
    win.close()


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


def test_linked_axis_inversion_undo_redo_propagates(qtbot) -> None:
    win0, win1 = _linked_pair(qtbot)

    win0.slicer_area.set_axis_inverted("x", True)
    assert win0.slicer_area.axis_inversions == {"x": True}
    assert win1.slicer_area.axis_inversions == {"x": True}
    _assert_dimension_inverted(win0.slicer_area, "x", True)
    _assert_dimension_inverted(win1.slicer_area, "x", True)

    win0.slicer_area.undo()
    assert win0.slicer_area.axis_inversions == {}
    assert win1.slicer_area.axis_inversions == {}
    _assert_dimension_inverted(win0.slicer_area, "x", False)
    _assert_dimension_inverted(win1.slicer_area, "x", False)

    win0.slicer_area.redo()
    assert win0.slicer_area.axis_inversions == {"x": True}
    assert win1.slicer_area.axis_inversions == {"x": True}
    _assert_dimension_inverted(win0.slicer_area, "x", True)
    _assert_dimension_inverted(win1.slicer_area, "x", True)

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

    win.slicer_area.apply_func(lambda darr: darr + 1, preview=True)
    xarray.testing.assert_identical(win.slicer_area._data, data)
    xarray.testing.assert_identical(win.slicer_area.data, data + 1)

    win.slicer_area.apply_func(None)
    xarray.testing.assert_identical(win.slicer_area.data, data)
    win.close()


def test_apply_func_rejects_operation_backed_filters(qtbot) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["x", "y"],
        coords={"x": np.arange(5, dtype=float), "y": np.arange(5, dtype=float)},
    )
    operation = erlab.interactive.imagetool.provenance.GaussianFilterOperation(
        sigma={"x": 1.0}
    )

    win = ImageTool(data)
    qtbot.addWidget(win)

    win.slicer_area.apply_filter_operation(operation)
    expected = win.slicer_area.data.copy(deep=True)
    before = win.slicer_area.displayed_provenance_spec()
    assert before is not None

    with pytest.raises(ValueError, match="apply_filter_operation"):
        win.slicer_area.apply_func(
            lambda darr: darr + 1,
            operation=erlab.interactive.imagetool.provenance.NormalizeOperation(
                dims=("x",),
                mode="area",
            ),
        )
    with pytest.raises(ValueError, match="preview-only"):
        win.slicer_area.apply_func(lambda darr: darr + 1, preview=False)

    xarray.testing.assert_identical(win.slicer_area.data, expected)
    after = win.slicer_area.displayed_provenance_spec()
    assert after is not None
    assert after.display_code() == before.display_code()

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
        win.slicer_area.apply_func(lambda darr: darr + 1, update=False, preview=True)

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

    win.slicer_area.apply_func(
        lambda darr: (darr + 1).transpose("y", "x"),
        preview=True,
    )

    assert win.slicer_area.data.dims == data.dims
    assert win.slicer_area.data.chunks is not None
    xarray.testing.assert_identical(
        win.slicer_area.data.compute(), (data + 1).compute()
    )
    xarray.testing.assert_identical(
        win.slicer_area.displayed_data.compute(), data.compute()
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
        win.slicer_area.apply_func(
            lambda darr: darr.expand_dims("z"), update=False, preview=True
        )
    with pytest.raises(ValueError, match="dimensions do not match"):
        win.slicer_area.apply_func(
            lambda darr: darr.rename({"x": "z"}), update=False, preview=True
        )
    with pytest.raises(ValueError, match="shape does not match"):
        win.slicer_area.apply_func(
            lambda darr: darr.isel(x=slice(1, None)), update=False, preview=True
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

    win.slicer_area.apply_func(lambda _darr: preview, preview=True)
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

    win.slicer_area.apply_func(lambda darr: darr + 10, preview=True)
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
    assert pyperclip.paste().startswith("era.transform.rotate(data")

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


def test_itool_guidelines_draw_above_all_cursors(qtbot) -> None:
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
    plot_item.set_guidelines(2)
    area.add_cursor()

    cursor_z_values = [
        item.zValue()
        for item_dict in (*plot_item.cursor_lines, *plot_item.cursor_spans)
        for item in item_dict.values()
    ]

    assert cursor_z_values
    assert all(
        guideline_item.zValue() > max(cursor_z_values)
        for guideline_item in plot_item._guidelines_items
    )

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


def test_itool_open_in_ftool_sets_squeezed_source_spec(qtbot, monkeypatch) -> None:
    win = itool(_TEST_DATA["2D"].copy(), execute=False)
    qtbot.addWidget(win)

    child = dtool(_TEST_DATA["2D"].copy(), execute=False)
    monkeypatch.setattr(erlab.interactive, "ftool", lambda *args, **kwargs: child)

    image = win.slicer_area.images[0]
    image.open_in_ftool()

    assert child.source_spec == image.make_tool_source_spec(squeeze=True)
    assert child.source_binding is None
    assert child.source_state == "fresh"

    win.close()


def test_profile_open_in_ftool_omits_noop_squeeze_source_binding(
    qtbot, monkeypatch
) -> None:
    win = itool(_TEST_DATA["2D"].copy(), execute=False)
    qtbot.addWidget(win)

    child = dtool(_TEST_DATA["2D"].copy(), execute=False)
    monkeypatch.setattr(erlab.interactive, "ftool", lambda *args, **kwargs: child)

    profile = win.slicer_area.profiles[0]
    profile.open_in_ftool()

    assert child.source_spec is not None
    assert not any(
        isinstance(operation, provenance.SqueezeOperation)
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


def test_crop_to_view_nonuniform_source_spec_uses_public_indices(qtbot) -> None:
    data = xr.DataArray(
        np.arange(25, dtype=float).reshape((5, 5)),
        dims=["x", "y"],
        coords={"x": [0.0, 0.2, 0.8, 1.4, 2.0], "y": np.arange(5.0)},
        name="scan",
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    win.slicer_area.set_manual_limits({"x_idx": [1.0, 3.0], "y": [0.0, 2.0]})

    dialog = CropToViewDialog(win.slicer_area)
    qtbot.addWidget(dialog)
    dialog.dim_checks["x_idx"].setChecked(True)
    dialog.dim_checks["y"].setChecked(True)

    expected = data.isel(x=slice(1, 4)).sel(y=slice(0.0, 2.0))
    public_data = erlab.interactive.imagetool.slicer.restore_nonuniform_dims(
        win.slicer_area.data
    )
    source_spec = dialog.source_spec("ignored")
    code = dialog.make_code()

    assert source_spec.kind == "public_data"
    assert CropToViewDialog._nonuniform_isel_slice(slice(3, 1, -1)) == slice(3, 0, -1)
    assert "x_idx" not in code
    xarray.testing.assert_identical(
        dialog.process_data(public_data).rename(None), expected.rename(None)
    )
    xarray.testing.assert_identical(
        source_spec.apply(win.slicer_area.data).rename(None), expected.rename(None)
    )
    xarray.testing.assert_identical(
        _exec_data_fragment(data, code).rename(None), expected.rename(None)
    )

    dialog.close()
    win.close()


def test_crop_between_cursors_nonuniform_source_spec_uses_public_indices(qtbot) -> None:
    data = xr.DataArray(
        np.arange(25, dtype=float).reshape((5, 5)),
        dims=["x", "y"],
        coords={"x": [0.0, 0.2, 0.8, 1.4, 2.0], "y": np.arange(5.0)},
        name="scan",
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    win.slicer_area.add_cursor()
    win.slicer_area.set_index(axis=0, value=1, cursor=0)
    win.slicer_area.set_index(axis=1, value=0, cursor=0)
    win.slicer_area.set_index(axis=0, value=3, cursor=1)
    win.slicer_area.set_index(axis=1, value=2, cursor=1)

    dialog = CropDialog(win.slicer_area)
    qtbot.addWidget(dialog)
    dialog.cursor_combos[0].setCurrentIndex(0)
    dialog.cursor_combos[1].setCurrentIndex(1)
    dialog.dim_checks["x_idx"].setChecked(True)
    dialog.dim_checks["y"].setChecked(True)

    expected = data.isel(x=slice(1, 4)).sel(y=slice(0.0, 2.0))
    public_data = erlab.interactive.imagetool.slicer.restore_nonuniform_dims(
        win.slicer_area.data
    )
    source_spec = dialog.source_spec("ignored")
    code = dialog.make_code()

    assert source_spec.kind == "public_data"
    assert "x_idx" not in code
    xarray.testing.assert_identical(
        dialog.process_data(public_data).rename(None), expected.rename(None)
    )
    xarray.testing.assert_identical(
        source_spec.apply(win.slicer_area.data).rename(None), expected.rename(None)
    )
    xarray.testing.assert_identical(
        _exec_data_fragment(data, code).rename(None), expected.rename(None)
    )

    dialog.close()
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
        win.slicer_area._data.rename(None), data.qsel.mean("x")
    )

    xarray.testing.assert_identical(
        _exec_data_fragment(data, pyperclip.paste()),
        data.qsel.mean("x"),
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
    assert ".rename(" not in display_code
    xarray.testing.assert_identical(
        derived.rename(None), data.qsel.mean("x").rename(None)
    )
    win.close()


def test_itool_transform_after_filter_uses_displayed_data_and_provenance(
    qtbot, accept_dialog
) -> None:
    data = xr.DataArray(
        np.arange(60).reshape((3, 4, 5)).astype(float),
        dims=["x", "y", "z"],
        coords={"x": np.arange(3), "y": np.arange(4), "z": np.arange(5)},
        name="scan",
    )
    filter_operation = erlab.interactive.imagetool.provenance.NormalizeOperation(
        dims=("x",),
        mode="min",
    )
    filtered = filter_operation.apply(data, parent_data=data)
    expected = filtered.qsel.mean("y")
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    win.slicer_area.apply_filter_operation(filter_operation)

    def _set_dialog_params(dialog: AverageDialog) -> None:
        dialog.dim_checks["y"].setChecked(True)
        dialog.launch_mode_combo.setCurrentText("Replace Current")

    accept_dialog(win.mnb._average, pre_call=_set_dialog_params)

    assert win.slicer_area._applied_func is None
    xarray.testing.assert_identical(win.slicer_area._data, expected)
    assert win.provenance_spec is not None
    code = win.provenance_spec.display_code()
    assert code is not None
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    xarray.testing.assert_identical(
        namespace["derived"].rename(None), expected.rename(None)
    )

    win.close()


def test_itool_aggregate_sum(qtbot, accept_dialog) -> None:
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

    def _set_dialog_params(dialog: AggregateDialog) -> None:
        dialog.dim_checks["x"].setChecked(True)
        _set_combo_data(dialog.reducer_combo, "sum")
        with qtbot.wait_signal(dialog._sigCodeCopied):
            dialog.copy_button.click()
        dialog.launch_mode_combo.setCurrentText("Replace Current")

    accept_dialog(win.mnb._aggregate, pre_call=_set_dialog_params)

    expected = data.qsel.sum("x")
    xarray.testing.assert_identical(win.slicer_area._data.rename(None), expected)
    xarray.testing.assert_identical(
        _exec_data_fragment(data, pyperclip.paste()), expected
    )

    assert win.provenance_spec is not None
    assert [op.op for op in win.provenance_spec.operations] == [
        "qsel_aggregate",
    ]
    aggregate_op = win.provenance_spec.operations[0]
    assert isinstance(
        aggregate_op,
        erlab.interactive.imagetool.provenance.QSelAggregationOperation,
    )
    assert aggregate_op.func == "sum"

    display_code = win.provenance_spec.display_code()
    assert display_code is not None
    display_namespace = _exec_generated_code(
        display_code,
        {"data": data.copy(deep=True)},
    )
    derived = display_namespace["derived"]
    assert isinstance(derived, xr.DataArray)
    assert ".rename(" not in display_code
    xarray.testing.assert_identical(derived.rename(None), expected.rename(None))

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

    spec = dialog.source_spec("ignored")
    expected = erlab.interactive.imagetool.slicer.restore_nonuniform_dims(
        dialog.process_data(win.slicer_area.data)
    )
    refreshed = spec.apply(win.slicer_area.data)

    assert spec.kind == "full_data"
    assert [op.op for op in spec.operations] == [
        "qsel_aggregate",
        "restore_nonuniform_dims",
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


def test_aggregate_source_spec_restores_nonuniform_dims_after_refresh(qtbot) -> None:
    data = xr.DataArray(
        np.arange(20).reshape((5, 4)).astype(float),
        dims=["x", "y"],
        coords={"x": [0.0, 0.2, 0.8, 1.4, 2.0], "y": np.arange(4)},
        name="scan",
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    assert win.slicer_area.data.dims == ("x_idx", "y")
    dialog = AggregateDialog(win.slicer_area)
    qtbot.addWidget(dialog)
    dialog.dim_checks["y"].setChecked(True)
    _set_combo_data(dialog.reducer_combo, "sum")

    spec = dialog.source_spec("ignored")
    expected = erlab.interactive.imagetool.slicer.restore_nonuniform_dims(
        dialog.process_data(win.slicer_area.data)
    )
    refreshed = spec.apply(win.slicer_area.data)

    assert spec.kind == "full_data"
    assert [op.op for op in spec.operations] == [
        "qsel_aggregate",
        "restore_nonuniform_dims",
    ]
    assert refreshed.dims == ("x",)
    xarray.testing.assert_identical(refreshed, expected)

    display_code = spec.display_code(parent_data=win.slicer_area.data)
    assert display_code is not None
    namespace = _exec_generated_code(
        display_code,
        {"data": win.slicer_area.data.copy(deep=True)},
    )
    derived = namespace["derived"]
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
        win.slicer_area._data.rename(None), data.qsel.mean("alpha")
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
        data.qsel.mean("k-space"),
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


def test_high_dimensional_reduction_dialog_selects_scalar(qtbot) -> None:
    data = _high_dimensional_data()
    dialog = imagetool_highdim._HighDimensionalReductionDialog(None, data)
    qtbot.addWidget(dialog)
    open_button = dialog.button_box.button(
        QtWidgets.QDialogButtonBox.StandardButton.Open
    )

    assert not open_button.isEnabled()

    row = dialog.rows[-1]
    _set_combo_data(row.action_combo, "select")
    row.scalar_controls.index_spin.setValue(2)

    expected = data.isel(x=2)
    assert open_button.isEnabled()
    assert dialog.source_operations() == [provenance.IselOperation(kwargs={"x": 2})]
    with pytest.raises(RuntimeError, match="No reduced data"):
        _ = dialog.result_data
    xarray.testing.assert_identical(
        _exec_data_fragment(data, dialog.make_code()), expected
    )

    dialog.copy_button.click()
    assert pyperclip.paste() == dialog.make_code()
    dialog.accept()
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted
    xarray.testing.assert_identical(dialog.result_data, expected)


def test_high_dimensional_reduction_dialog_aggregates_dimension(qtbot) -> None:
    data = _high_dimensional_data()
    dialog = imagetool_highdim._HighDimensionalReductionDialog(None, data)
    qtbot.addWidget(dialog)
    open_button = dialog.button_box.button(
        QtWidgets.QDialogButtonBox.StandardButton.Open
    )

    row = dialog.rows[-1]
    _set_combo_data(row.action_combo, "aggregate")
    _set_combo_data(row.reducer_combo, "sum")

    operation = provenance.QSelAggregationOperation(dims=("x",), func="sum")
    expected = data.qsel.sum(("x",))
    assert open_button.isEnabled()
    assert dialog.source_operations() == [operation]
    with pytest.raises(RuntimeError, match="No reduced data"):
        _ = dialog.result_data
    xarray.testing.assert_identical(
        _exec_data_fragment(data, dialog.make_code()), expected
    )
    dialog.accept()
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted
    xarray.testing.assert_identical(dialog.result_data, expected)


def test_high_dimensional_reduction_dialog_scalar_methods_and_parent(qtbot) -> None:
    data = _high_dimensional_data()
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    dialog = imagetool_highdim._HighDimensionalReductionDialog(parent, data)
    qtbot.addWidget(dialog)

    assert dialog.windowModality() == QtCore.Qt.WindowModality.WindowModal

    scan_row = dialog.rows[0]
    _set_combo_data(scan_row.action_combo, "select")
    _set_combo_data(scan_row.scalar_controls.method_combo, "qsel")
    scan_row.scalar_controls.value_spin.setValue(1.0)

    x_row = dialog.rows[-1]
    _set_combo_data(x_row.action_combo, "select")
    _set_combo_data(x_row.scalar_controls.method_combo, "sel")
    x_row.scalar_controls.value_spin.setValue(2.0)

    expected = data.sel(x=2.0).qsel(scan=1.0)
    assert dialog.source_operations() == [
        provenance.SelOperation(kwargs={"x": 2.0}),
        provenance.QSelOperation(kwargs={"scan": 1.0}),
    ]
    xarray.testing.assert_identical(
        _exec_data_fragment(data, dialog.make_code()), expected
    )

    dialog.accept()

    assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted
    xarray.testing.assert_identical(dialog.result_data, expected)


def test_high_dimensional_reduction_dialog_preview_does_not_apply_operations(
    qtbot,
    monkeypatch,
) -> None:
    data = _high_dimensional_data()
    dialog = imagetool_highdim._HighDimensionalReductionDialog(None, data)
    qtbot.addWidget(dialog)
    open_button = dialog.button_box.button(
        QtWidgets.QDialogButtonBox.StandardButton.Open
    )
    calls: list[provenance.QSelAggregationOperation] = []

    def _fail_apply(
        self: provenance.QSelAggregationOperation,
        _data: xr.DataArray,
        *,
        parent_data: xr.DataArray,
    ) -> xr.DataArray:
        calls.append(self)
        raise AssertionError("preview must not apply aggregation")

    monkeypatch.setattr(provenance.QSelAggregationOperation, "apply", _fail_apply)

    row = dialog.rows[-1]
    _set_combo_data(row.action_combo, "aggregate")
    _set_combo_data(row.reducer_combo, "sum")
    _set_combo_data(row.reducer_combo, "mean")

    assert open_button.isEnabled()
    assert calls == []
    with pytest.raises(RuntimeError, match="No reduced data"):
        _ = dialog.result_data


def test_high_dimensional_reduction_dialog_accept_applies_once(
    qtbot,
    monkeypatch,
) -> None:
    data = _high_dimensional_data()
    dialog = imagetool_highdim._HighDimensionalReductionDialog(None, data)
    qtbot.addWidget(dialog)
    calls: list[provenance.QSelAggregationOperation] = []
    original_apply = provenance.QSelAggregationOperation.apply

    def _count_apply(
        self: provenance.QSelAggregationOperation,
        data_array: xr.DataArray,
        *,
        parent_data: xr.DataArray,
    ) -> xr.DataArray:
        calls.append(self)
        return original_apply(self, data_array, parent_data=parent_data)

    monkeypatch.setattr(provenance.QSelAggregationOperation, "apply", _count_apply)

    row = dialog.rows[-1]
    _set_combo_data(row.action_combo, "aggregate")
    _set_combo_data(row.reducer_combo, "sum")

    dialog.accept()

    assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted
    assert len(calls) == 1
    assert calls[0] == provenance.QSelAggregationOperation(dims=("x",), func="sum")
    xarray.testing.assert_identical(dialog.result_data, data.qsel.sum(("x",)))


def test_high_dimensional_reduction_dialog_accept_keeps_dialog_open_on_error(
    qtbot,
    monkeypatch,
) -> None:
    data = _high_dimensional_data()
    dialog = imagetool_highdim._HighDimensionalReductionDialog(None, data)
    qtbot.addWidget(dialog)
    open_button = dialog.button_box.button(
        QtWidgets.QDialogButtonBox.StandardButton.Open
    )

    row = dialog.rows[-1]
    _set_combo_data(row.action_combo, "aggregate")
    assert open_button.isEnabled()

    def _raise_process(_data: xr.DataArray) -> xr.DataArray:
        raise ValueError("cannot aggregate")

    monkeypatch.setattr(dialog, "process_data", _raise_process)

    dialog.accept()

    assert dialog.result() != QtWidgets.QDialog.DialogCode.Accepted
    assert not open_button.isEnabled()
    with pytest.raises(RuntimeError, match="No reduced data"):
        _ = dialog.result_data


def test_high_dimensional_reduction_dialog_metadata_and_warning_paths(
    qtbot,
    monkeypatch,
) -> None:
    data = _high_dimensional_data()
    dialog = imagetool_highdim._HighDimensionalReductionDialog(None, data)
    qtbot.addWidget(dialog)
    warnings: list[tuple[object, ...]] = []

    def _record_warning(*args: object) -> QtWidgets.QMessageBox.StandardButton:
        warnings.append(args)
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _record_warning)

    assert dialog._processed_ndim_from_shape((5,)) == 2
    assert dialog._processed_ndim_from_shape((2, 1, 3, 1, 4)) == 3
    assert dialog._set_preview_from_metadata(("profile",), (5,))

    row = dialog.rows[-1]
    dialog.accept()
    assert len(warnings) == 1
    assert dialog.result() != QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(
        provenance,
        "operations_expression_code",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("no code")),
    )
    assert dialog.make_code() == ""
    dialog.copy_button.click()
    assert len(warnings) == 2

    _set_combo_data(row.action_combo, "aggregate")
    monkeypatch.setattr(dialog, "process_data", lambda _data: data)
    dialog.accept()
    assert len(warnings) == 3
    assert dialog.result() != QtWidgets.QDialog.DialogCode.Accepted
    with pytest.raises(RuntimeError, match="No reduced data"):
        _ = dialog.result_data


def test_high_dimensional_reduction_dialog_rejects_empty_result(qtbot) -> None:
    data = xr.DataArray(
        np.empty((0, 2, 3, 4, 5)),
        dims=("empty", "scan", "pol", "y", "x"),
        coords={
            "empty": np.array([], dtype=float),
            "scan": np.arange(2, dtype=float),
            "pol": np.arange(3, dtype=float),
            "y": np.arange(4, dtype=float),
            "x": np.arange(5, dtype=float),
        },
    )
    dialog = imagetool_highdim._HighDimensionalReductionDialog(None, data)
    qtbot.addWidget(dialog)
    open_button = dialog.button_box.button(
        QtWidgets.QDialogButtonBox.StandardButton.Open
    )

    _set_combo_data(dialog.rows[-1].action_combo, "aggregate")

    assert not open_button.isEnabled()
    with pytest.raises(RuntimeError, match="No reduced data"):
        _ = dialog.result_data


def test_scalar_selection_controls_non_numeric_and_width_branches(qtbot) -> None:
    string_coord_data = xr.DataArray(
        np.arange(3),
        dims=("label",),
        coords={"label": np.array(["a", "b", "c"], dtype=object)},
    )
    string_controls = imagetool_dialogs._ScalarSelectionControls(
        string_coord_data,
        "label",
        0,
        object_name_prefix="test_scalar",
        include_width=False,
    )
    qtbot.addWidget(string_controls.method_combo)
    qtbot.addWidget(string_controls.stack)
    qtbot.addWidget(string_controls.width_widget)

    assert string_controls.method == "isel"
    assert string_controls.indexer() == ("label", 1)
    assert string_controls.qsel_width_indexer() is None

    numeric_data = xr.DataArray(
        np.arange(3),
        dims=("x",),
        coords={"x": np.arange(3, dtype=float)},
    )
    numeric_controls = imagetool_dialogs._ScalarSelectionControls(
        numeric_data,
        "x",
        0,
        object_name_prefix="test_numeric_scalar",
        current_index=None,
    )
    qtbot.addWidget(numeric_controls.method_combo)
    qtbot.addWidget(numeric_controls.stack)
    qtbot.addWidget(numeric_controls.width_widget)

    assert numeric_controls.indexer() == ("x", 1.0)
    numeric_controls.width_check.setChecked(True)
    numeric_controls.width_spin.setValue(0.0)
    assert numeric_controls.qsel_width_indexer() is None


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
    assert [op.op for op in win.provenance_spec.operations] == ["qsel"]

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


def test_selection_dialog_isel_range_step_executes_code(qtbot) -> None:
    data = _selection_4d_data()
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    dialog = SelectionDialog(win.slicer_area)
    _clear_selection_dialog(dialog)

    row = dialog.rows[0]
    row.use_check.setChecked(True)
    _set_combo_data(row.method_combo, "isel")
    _set_combo_data(row.kind_combo, "range")
    row.index_start_spin.setValue(0)
    row.index_stop_spin.setValue(3)
    row.step_check.setChecked(True)
    row.step_spin.setValue(2)

    expected = data.isel(alpha=slice(0, 3, 2))
    assert dialog.source_operations() == [
        provenance.IselOperation(kwargs={"alpha": slice(0, 3, 2)})
    ]
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
    assert dialog.make_code() == 'data.isel({"Fake Motor": 1})'
    xarray.testing.assert_identical(
        _exec_data_fragment(data, dialog.make_code()), expected
    )

    dialog.close()
    win.close()


@pytest.mark.parametrize("method", ["sel", "qsel"])
def test_selection_dialog_label_range_step_executes_code(qtbot, method: str) -> None:
    data = _selection_4d_data()
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    dialog = SelectionDialog(win.slicer_area)
    _clear_selection_dialog(dialog)

    row = dialog.rows[1]
    row.use_check.setChecked(True)
    _set_combo_data(row.method_combo, method)
    _set_combo_data(row.kind_combo, "range")
    row.value_start_spin.setValue(0.0)
    row.value_stop_spin.setValue(3.0)
    row.step_check.setChecked(True)
    row.step_spin.setValue(2)

    expected = getattr(data, method)(eV=slice(0.0, 3.0, 2))
    xarray.testing.assert_identical(dialog.process_data(dialog.public_data), expected)
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


def test_selection_dialog_restore_stepped_selection_operation(qtbot) -> None:
    data = _selection_4d_data()
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    dialog = SelectionDialog(win.slicer_area)
    _clear_selection_dialog(dialog)

    operation = provenance.IselOperation(kwargs={"alpha": slice(0, 3, 2)})
    dialog.restore_transform_operation(operation)

    row = dialog.rows[0]
    assert row.use_check.isChecked()
    assert row.step_check.isChecked()
    assert row.step_spin.value() == 2
    assert dialog.source_operations() == [operation]
    expected = data.isel(alpha=slice(0, 3, 2))
    xarray.testing.assert_identical(dialog.process_data(dialog.public_data), expected)

    dialog.close()
    win.close()


def test_selection_dialog_restore_selection_variants_and_rejects_invalid_steps(
    qtbot,
) -> None:
    data = _selection_4d_data()
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    dialog = SelectionDialog(win.slicer_area)
    _clear_selection_dialog(dialog)

    sel_operation = provenance.SelOperation(kwargs={"eV": slice(1.0, 3.0)})
    dialog.restore_transform_operation(sel_operation)
    assert dialog.source_operations() == [sel_operation]

    qsel_operation = provenance.QSelOperation(kwargs={"beta": 2.0, "beta_width": 1.5})
    dialog.restore_transform_operation(qsel_operation)
    beta_row = next(row for row in dialog.rows if row.dim == "beta")
    assert beta_row.width_check.isChecked()
    assert beta_row.width_spin.value() == 1.5
    assert dialog.source_operations() == [qsel_operation]

    with pytest.raises(ValueError, match="Open-ended selections"):
        dialog.restore_transform_operation(
            provenance.IselOperation(kwargs={"alpha": slice(None, 3)})
        )
    with pytest.raises(ValueError, match="integer strides"):
        dialog.restore_transform_operation(
            provenance.IselOperation(kwargs={"alpha": slice(0, 3, 1.5)})
        )
    with pytest.raises(ValueError, match="Reverse or zero-step"):
        dialog.restore_transform_operation(
            provenance.IselOperation(kwargs={"alpha": slice(0, 3, 0)})
        )
    with pytest.raises(ValueError, match="not available"):
        dialog.restore_transform_operation(
            provenance.SelOperation(kwargs={"missing": 1.0})
        )

    dialog.close()
    win.close()


def test_selection_dialog_descending_range_step_restore(qtbot) -> None:
    data = xr.DataArray(
        np.arange(5, dtype=float),
        dims=("x",),
        coords={"x": np.arange(5.0, 0.0, -1.0)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    dialog = SelectionDialog(win.slicer_area)
    _clear_selection_dialog(dialog)
    row = dialog.rows[0]

    row.use_check.setChecked(True)
    _set_combo_data(row.method_combo, "sel")
    _set_combo_data(row.kind_combo, "range")
    row.value_start_spin.setValue(2.0)
    row.value_stop_spin.setValue(4.0)
    row.step_check.setChecked(True)
    row.step_spin.setValue(2)

    assert row.indexer() == ("x", slice(4.0, 2.0, 2))

    dialog.close()
    win.close()


def _restore_dialog_data() -> xr.DataArray:
    return xr.DataArray(
        np.arange(20, dtype=float).reshape((4, 5)),
        dims=("x", "y"),
        coords={
            "x": np.arange(4.0),
            "y": np.arange(5.0),
            "kx": ("x", np.linspace(-0.2, 0.2, 4)),
            "ky": ("y", np.linspace(-0.3, 0.3, 5)),
            "temperature": ("x", [20.0, 10.0, 30.0, 15.0]),
        },
        attrs={"note": "old"},
        name="scan",
    )


@pytest.mark.parametrize(
    ("dialog_cls", "operation", "expected"),
    [
        (
            AggregateDialog,
            provenance.AverageOperation(dims=("x",)),
            provenance.QSelAggregationOperation(dims=("x",), func="mean"),
        ),
        (
            AggregateDialog,
            provenance.QSelAggregationOperation(dims=("y",), func="sum"),
            provenance.QSelAggregationOperation(dims=("y",), func="sum"),
        ),
        (
            InterpolationDialog,
            provenance.InterpolationOperation(
                dim="x",
                values=[0.0, 1.5, 3.0],
                method="linear",
            ),
            provenance.InterpolationOperation(
                dim="x",
                values=[0.0, 1.5, 3.0],
                method="linear",
            ),
        ),
        (
            LeadingEdgeDialog,
            provenance.LeadingEdgeOperation(
                dim="x",
                fraction=0.4,
                direction="positive",
            ),
            provenance.LeadingEdgeOperation(
                dim="x",
                fraction=0.4,
                direction="positive",
            ),
        ),
        (
            CoarsenDialog,
            provenance.CoarsenOperation(
                dim={"x": 2},
                boundary="trim",
                side="left",
                coord_func="mean",
                reducer="sum",
            ),
            provenance.CoarsenOperation(
                dim={"x": 2},
                boundary="trim",
                side="left",
                coord_func="mean",
                reducer="sum",
            ),
        ),
        (
            ThinDialog,
            provenance.ThinOperation(mode="global", factor=2),
            provenance.ThinOperation(mode="global", factor=2),
        ),
        (
            ThinDialog,
            provenance.ThinOperation(mode="per_dim", factors={"x": 2}),
            provenance.ThinOperation(mode="per_dim", factors={"x": 2}),
        ),
        (
            SymmetrizeDialog,
            provenance.SymmetrizeOperation(
                dim="x",
                center=1.0,
                part="below",
                mode="full",
                subtract=True,
            ),
            provenance.SymmetrizeOperation(
                dim="x",
                center=1.0,
                part="below",
                mode="full",
                subtract=True,
            ),
        ),
        (
            SymmetrizeNfoldDialog,
            provenance.SymmetrizeNfoldOperation(
                fold=4,
                axes=("x", "y"),
                center={"x": 1.0, "y": 2.0},
                reshape=False,
                order=2,
            ),
            provenance.SymmetrizeNfoldOperation(
                fold=4,
                axes=("x", "y"),
                center={"x": 1.0, "y": 2.0},
                reshape=False,
                order=2,
            ),
        ),
        (
            DivideByCoordDialog,
            provenance.DivideByCoordOperation(coord_name="temperature"),
            provenance.DivideByCoordOperation(coord_name="temperature"),
        ),
        (
            SwapDimsDialog,
            provenance.SwapDimsOperation(mapping={"x": "kx"}),
            provenance.SwapDimsOperation(mapping={"x": "kx"}),
        ),
        (
            RenameDimsCoordsDialog,
            provenance.RenameDimsCoordsOperation(mapping={"x": "x_new"}),
            provenance.RenameDimsCoordsOperation(mapping={"x": "x_new"}),
        ),
        (
            AssignCoordsDialog,
            provenance.AffineCoordOperation(coord_name="x", scale=2.0, offset=1.0),
            provenance.AffineCoordOperation(coord_name="x", scale=2.0, offset=1.0),
        ),
        (
            AssignCoordsDialog,
            provenance.AssignCoordsOperation(
                coord_name="x", values=[0.0, 1.0, 2.0, 3.0]
            ),
            provenance.AssignCoordsOperation(
                coord_name="x", values=[0.0, 1.0, 2.0, 3.0]
            ),
        ),
        (
            AssignCoordsDialog,
            provenance.AssignScalarCoordOperation(coord_name="sample_temp", value=20.0),
            provenance.AssignScalarCoordOperation(coord_name="sample_temp", value=20.0),
        ),
        (
            AssignCoordsDialog,
            provenance.AssignCoord1DOperation(
                coord_name="kx_new",
                dim="x",
                values=[0.0, 0.1, 0.2, 0.3],
            ),
            provenance.AssignCoord1DOperation(
                coord_name="kx_new",
                dim="x",
                values=[0.0, 0.1, 0.2, 0.3],
            ),
        ),
        (
            AssignAttrsDialog,
            provenance.AssignAttrsOperation(
                attrs={"note": "edited", "temperature": 20.0}
            ),
            provenance.AssignAttrsOperation(
                attrs={"note": "edited", "temperature": 20.0}
            ),
        ),
    ],
)
def test_transform_dialog_restore_operation_roundtrip(
    qtbot,
    dialog_cls: type[imagetool_dialogs.DataTransformDialog],
    operation: provenance.ToolProvenanceOperation,
    expected: provenance.ToolProvenanceOperation,
) -> None:
    win = itool(_restore_dialog_data(), execute=False)
    qtbot.addWidget(win)
    dialog = dialog_cls(win.slicer_area)
    qtbot.addWidget(dialog)

    dialog.restore_transform_operation(operation)

    assert dialog.source_transform_operation() == expected

    dialog.close()
    win.close()


def test_rotation_dialog_restore_operation_roundtrip_and_rejects_wrong_axes(
    qtbot,
) -> None:
    win = itool(_restore_dialog_data(), execute=False)
    qtbot.addWidget(win)
    dialog = RotationDialog(win.slicer_area)
    qtbot.addWidget(dialog)
    axes = tuple(win.slicer_area.main_image.axis_dims_uniform)
    operation = provenance.RotateOperation(
        angle=12.5,
        axes=axes,
        center=(1.0, 2.0),
        reshape=False,
        order=2,
    )

    dialog.restore_transform_operation(operation)

    assert dialog.source_transform_operation() == operation
    with pytest.raises(ValueError, match="not currently visible"):
        dialog.restore_transform_operation(
            provenance.RotateOperation(angle=0.0, axes=("x", "z"), center=(0.0, 0.0))
        )

    dialog.close()
    win.close()


def test_symmetrize_nfold_restore_accepts_mapping_center(qtbot) -> None:
    win = itool(_restore_dialog_data(), execute=False)
    qtbot.addWidget(win)
    dialog = SymmetrizeNfoldDialog(win.slicer_area)
    qtbot.addWidget(dialog)

    dialog.restore_transform_operation(
        provenance.SymmetrizeNfoldOperation(
            fold=3,
            axes=("x", "y"),
            center={"x": 1.0, "y": 2.0},
        )
    )

    assert [spin.value() for spin in dialog.center_spins] == [1.0, 2.0]
    with pytest.raises(ValueError, match="not currently visible"):
        dialog.restore_transform_operation(
            provenance.SymmetrizeNfoldOperation(fold=3, axes=("x", "z"))
        )

    dialog.close()
    win.close()


def test_restore_transform_operation_ignores_unrelated_operations(qtbot) -> None:
    win = itool(_restore_dialog_data(), execute=False)
    qtbot.addWidget(win)
    dialogs = [
        imagetool_dialogs.DataTransformDialog(win.slicer_area),
        RotationDialog(win.slicer_area),
        AggregateDialog(win.slicer_area),
        InterpolationDialog(win.slicer_area),
        SortByDialog(win.slicer_area),
        LeadingEdgeDialog(win.slicer_area),
        CoarsenDialog(win.slicer_area),
        ThinDialog(win.slicer_area),
        SymmetrizeDialog(win.slicer_area),
        SymmetrizeNfoldDialog(win.slicer_area),
        DivideByCoordDialog(win.slicer_area),
        SwapDimsDialog(win.slicer_area),
        RenameDimsCoordsDialog(win.slicer_area),
        AssignCoordsDialog(win.slicer_area),
        AssignAttrsDialog(win.slicer_area),
    ]
    for dialog in dialogs:
        qtbot.addWidget(dialog)

    unrelated = provenance.ScriptCodeOperation(label="script", code="derived = data")
    for dialog in dialogs:
        dialog.restore_transform_operation(unrelated)

    for dialog in dialogs:
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


def test_selection_dialog_qsel_exact_value_and_width(qtbot) -> None:
    data = _selection_4d_data()
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    dialog = SelectionDialog(win.slicer_area)
    _clear_selection_dialog(dialog)

    value_literal = "2.000000000000001"
    width_literal = "2.000000000000001"
    row = dialog.rows[2]
    row.use_check.setChecked(True)
    _set_combo_data(row.method_combo, "qsel")
    _set_combo_data(row.kind_combo, "point")
    _set_spinbox_text(row.value_start_spin, value_literal)
    row.width_check.setChecked(True)
    _set_spinbox_text(row.width_spin, width_literal)

    assert row.value_start_spin.text() == value_literal
    assert row.width_spin.text() == width_literal
    _, _, qsel_kwargs = dialog._selection_kwargs()
    assert qsel_kwargs["beta"] == float(value_literal)
    assert qsel_kwargs["beta_width"] == float(width_literal)

    expected = data.qsel(beta=float(value_literal), beta_width=float(width_literal))
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
    spec = dialog.source_spec("ignored")
    assert spec.kind == "public_data"
    xarray.testing.assert_identical(spec.apply(win.slicer_area.data), expected)

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

    filter_operation = erlab.interactive.imagetool.provenance.GaussianFilterOperation(
        sigma={"x": 1.0}
    )
    win.slicer_area.apply_filter_operation(filter_operation)

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
    assert win.slicer_area._accepted_filter_provenance_operation == filter_operation

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
    assert (
        code == 'result = data + 1\nderived = result\nderived = derived.qsel.mean("x")'
    )
    namespace = _exec_generated_code(code, {"data": source.copy(deep=True)})
    derived = namespace["derived"]
    assert isinstance(derived, xr.DataArray)
    xarray.testing.assert_identical(
        derived.rename(None), displayed.qsel.mean("x").rename(None)
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
    xarray.testing.assert_identical(launched[0].rename(None), data.qsel.mean("x"))

    dialog_replace = AverageDialog(win.slicer_area)
    qtbot.addWidget(dialog_replace)
    dialog_replace.dim_checks["y"].setChecked(True)
    dialog_replace.launch_mode_combo.setCurrentText("Replace Current")
    dialog_replace.accept()
    xarray.testing.assert_identical(
        win.slicer_area._data.rename(None), data.qsel.mean("y")
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
    assert win.slicer_area._data.name == "scan"

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


def test_itool_interpolate_exact_generated_coordinates(qtbot, accept_dialog) -> None:
    data = xr.DataArray(
        np.arange(6).reshape((3, 2)).astype(float),
        dims=["x", "y"],
        coords={"x": np.arange(3, dtype=float), "y": [10.0, 20.0]},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    start_literal = "0.123456789012345"
    step_literal = "0.123456789012345"
    target = np.linspace(
        float(start_literal),
        float(start_literal) + 2 * float(step_literal),
        3,
    )

    def _set_dialog_params(dialog: InterpolationDialog) -> None:
        dialog.coord_widget.mode_combo.setCurrentText("Delta")
        _set_spinbox_text(dialog.coord_widget.spin0, start_literal)
        _set_spinbox_text(dialog.coord_widget.spin1, step_literal)
        assert dialog.coord_widget.spin0.text() == start_literal
        assert dialog.coord_widget.spin1.text() == step_literal
        dialog.launch_mode_combo.setCurrentText("Replace Current")

    accept_dialog(win.mnb._interpolate, pre_call=_set_dialog_params)
    xarray.testing.assert_identical(
        win.slicer_area._data.rename(None),
        data.interp({"x": target}, method="linear").rename(None),
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


def test_itool_sortby(qtbot, accept_dialog) -> None:
    data = xr.DataArray(
        np.arange(12).reshape((4, 3)).astype(float),
        dims=("x", "y"),
        coords={
            "x": [2.0, 0.0, 1.0, 3.0],
            "y": [0.0, 1.0, 2.0],
            "temperature": ("x", [20.0, 10.0, 15.0, 5.0]),
        },
        name="scan",
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    def _check_key(dialog: SortByDialog, key: str) -> None:
        for row in range(dialog.key_table.rowCount()):
            item = dialog.key_table.item(row, 0)
            if item is not None and item.data(QtCore.Qt.ItemDataRole.UserRole) == key:
                item.setCheckState(QtCore.Qt.CheckState.Checked)
                dialog.key_table.selectRow(row)
                return
        raise AssertionError(f"Missing sort key {key!r}")

    def _set_dialog_params(dialog: SortByDialog) -> None:
        _check_key(dialog, "temperature")
        dialog.ascending_combo.setCurrentIndex(
            dialog.ascending_combo.findData(False, QtCore.Qt.ItemDataRole.UserRole)
        )
        with qtbot.wait_signal(dialog._sigCodeCopied):
            dialog.copy_button.click()
        dialog.launch_mode_combo.setCurrentText("Replace Current")

    accept_dialog(win.mnb._sort_by, pre_call=_set_dialog_params)

    expected = data.sortby("temperature", ascending=False)
    xarray.testing.assert_identical(
        win.slicer_area._data.rename(None), expected.rename(None)
    )
    assert win.slicer_area._data.name == "scan"
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


def test_itool_sortby_nonuniform_public_dims(qtbot, accept_dialog) -> None:
    data = xr.DataArray(
        np.arange(12).reshape((4, 3)).astype(float),
        dims=("x", "y"),
        coords={
            "x": np.array([2.0, 0.0, 1.0, 3.0]),
            "y": np.arange(3),
        },
        name="scan",
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    assert win.slicer_area.data.dims == ("x_idx", "y")

    def _set_dialog_params(dialog: SortByDialog) -> None:
        keys = [
            dialog.key_table.item(row, 0).data(QtCore.Qt.ItemDataRole.UserRole)
            for row in range(dialog.key_table.rowCount())
            if dialog.key_table.item(row, 0) is not None
        ]
        assert "x" in keys
        assert "x_idx" not in keys
        for row, key in enumerate(keys):
            if key == "x":
                item = dialog.key_table.item(row, 0)
                assert item is not None
                item.setCheckState(QtCore.Qt.CheckState.Checked)
        dialog.launch_mode_combo.setCurrentText("Replace Current")

    accept_dialog(win.mnb._sort_by, pre_call=_set_dialog_params)

    expected = data.sortby("x")
    xarray.testing.assert_identical(
        win.slicer_area._data.rename(None), expected.rename(None)
    )
    assert win.provenance_spec is not None
    display_code = win.provenance_spec.display_code()
    assert display_code is not None
    assert ".sortby(" in display_code
    assert "x_idx" not in display_code
    namespace = _exec_generated_code(display_code, {"data": data.copy(deep=True)})
    derived = namespace["derived"]
    assert isinstance(derived, xr.DataArray)
    xarray.testing.assert_identical(derived.rename(None), expected.rename(None))

    win.close()


def test_sortby_dialog_public_dim_order_and_coord_filters(qtbot, monkeypatch) -> None:
    data = xr.DataArray(
        np.arange(12).reshape((4, 3)).astype(float),
        dims=("x", "y"),
        coords={
            "x": np.array([2.0, 0.0, 1.0, 3.0]),
            "y": np.arange(3),
            "temperature": ("x", [20.0, 10.0, 15.0, 5.0]),
            "sample_id": 1,
            "mesh": (("x", "y"), np.ones((4, 3))),
        },
        name="scan",
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    restore_nonuniform_dims = erlab.interactive.imagetool.slicer.restore_nonuniform_dims

    def restore_transposed(source: xr.DataArray) -> xr.DataArray:
        return restore_nonuniform_dims(source).transpose("y", "x")

    monkeypatch.setattr(
        erlab.interactive.imagetool.slicer,
        "restore_nonuniform_dims",
        restore_transposed,
    )

    dialog = SortByDialog(win.slicer_area)
    qtbot.addWidget(dialog)
    keys = [
        item.data(QtCore.Qt.ItemDataRole.UserRole)
        for row in range(dialog.key_table.rowCount())
        if (item := dialog.key_table.item(row, 0)) is not None
    ]

    assert keys[:2] == ["x", "y"]
    assert "temperature" in keys
    assert "sample_id" not in keys
    assert "mesh" not in keys

    dialog.close()
    win.close()


def test_sortby_dialog_key_table_order_and_empty_items(qtbot) -> None:
    data = xr.DataArray(
        np.arange(12).reshape((4, 3)).astype(float),
        dims=("x", "y"),
        coords={
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [0.0, 1.0, 2.0],
            "temperature": ("x", [20.0, 10.0, 15.0, 5.0]),
            "pressure": ("x", [3.0, 1.0, 2.0, 4.0]),
        },
        name="scan",
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    dialog = SortByDialog(win.slicer_area)
    qtbot.addWidget(dialog)

    def table_keys() -> list[collections.abc.Hashable]:
        return [
            item.data(QtCore.Qt.ItemDataRole.UserRole)
            for row in range(dialog.key_table.rowCount())
            if (item := dialog.key_table.item(row, 0)) is not None
        ]

    initial_keys = table_keys()
    assert initial_keys[:4] == ["x", "y", "temperature", "pressure"]
    with pytest.raises(ValueError, match="No sort keys selected"):
        dialog.source_transform_operation()

    removed_item = dialog.key_table.takeItem(0, 0)
    assert removed_item is not None
    assert dialog._sort_keys == ()
    dialog.key_table.setItem(0, 0, removed_item)

    dialog.key_table.selectRow(0)
    dialog._move_selected_row(-1)
    assert table_keys() == initial_keys

    dialog.key_table.selectRow(initial_keys.index("temperature"))
    dialog._move_selected_row(-1)
    assert table_keys()[:4] == ["x", "temperature", "y", "pressure"]
    dialog._move_selected_row(1)
    assert table_keys() == initial_keys

    dialog.key_table.selectRow(dialog.key_table.rowCount() - 1)
    dialog._move_selected_row(1)
    assert table_keys() == initial_keys

    for key in ("y", "temperature"):
        item = dialog.key_table.item(table_keys().index(key), 0)
        assert item is not None
        item.setCheckState(QtCore.Qt.CheckState.Checked)
    dialog.key_table.selectRow(table_keys().index("temperature"))
    dialog._move_selected_row(-1)
    operation = dialog.source_transform_operation()
    assert operation.variables == ("temperature", "y")

    restored_operation = provenance.SortByOperation(
        variables=("pressure", "temperature"),
        ascending=False,
    )
    dialog.restore_transform_operation(restored_operation)
    assert table_keys()[:4] == ["pressure", "temperature", "x", "y"]
    assert dialog.source_transform_operation() == restored_operation

    empty_item_dialog = SortByDialog(win.slicer_area)
    qtbot.addWidget(empty_item_dialog)
    empty_item_dialog.key_table.selectRow(1)
    assert empty_item_dialog.key_table.takeItem(1, 0) is not None
    empty_item_dialog._move_selected_row(-1)

    empty_item_dialog.close()
    dialog.close()
    win.close()


def test_sortby_dialog_accept_requires_key(qtbot, monkeypatch) -> None:
    data = xr.DataArray(
        np.arange(6).reshape((2, 3)).astype(float),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0, 2.0]},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    dialog = SortByDialog(win.slicer_area)
    warnings_shown: list[tuple[QtWidgets.QWidget | None, str, str]] = []

    def record_warning(
        parent: QtWidgets.QWidget | None,
        title: str,
        text: str,
    ) -> QtWidgets.QMessageBox.StandardButton:
        warnings_shown.append((parent, title, text))
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", record_warning)

    with pytest.raises(ValueError, match="No sort keys selected"):
        dialog.source_transform_operation()
    dialog.accept()

    assert warnings_shown
    assert dialog.result() != QtWidgets.QDialog.DialogCode.Accepted

    dialog.close()
    win.close()


def test_itool_leading_edge(qtbot, accept_dialog) -> None:
    ev = np.linspace(0.0, 4.0, 5)
    data = xr.DataArray(
        np.vstack([4.0 - ev, 8.0 - 2.0 * ev, 2.0 - 0.5 * ev]),
        dims=["x", "eV"],
        coords={"x": np.arange(3), "eV": ev},
        name="scan",
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    assert "leadingEdgeAct" in win.mnb.action_dict

    def _set_dialog_params(dialog: LeadingEdgeDialog) -> None:
        assert dialog.dim_combo.currentData(QtCore.Qt.ItemDataRole.UserRole) == "eV"
        dialog.fraction_spin.setValue(0.5)
        dialog.direction_combo.setCurrentIndex(
            dialog.direction_combo.findData("positive", QtCore.Qt.ItemDataRole.UserRole)
        )
        with qtbot.wait_signal(dialog._sigCodeCopied):
            dialog.copy_button.click()
        dialog.launch_mode_combo.setCurrentText("Replace Current")

    accept_dialog(win.mnb._leading_edge, pre_call=_set_dialog_params)

    expected = erlab.analysis.interpolate.leading_edge(data)
    xarray.testing.assert_identical(
        win.slicer_area._data.rename(None), expected.rename(None)
    )
    assert win.slicer_area._data.name == "scan"
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


def test_leading_edge_dialog_rejects_no_dimension(qtbot, monkeypatch) -> None:
    ev = np.linspace(0.0, 4.0, 5)
    data = xr.DataArray(
        np.vstack([4.0 - ev, 8.0 - 2.0 * ev]),
        dims=["x", "eV"],
        coords={"x": np.arange(2), "eV": ev},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    dialog = LeadingEdgeDialog(win.slicer_area)
    dialog.launch_mode_combo.setCurrentText("Replace Current")
    dialog.dim_combo.setCurrentIndex(-1)

    warning_calls: list[tuple[object, ...]] = []

    def _record_warning(*args, **kwargs):
        warning_calls.append(args)
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _record_warning)
    assert dialog.make_code() == ""
    with pytest.raises(ValueError, match="No dimension selected"):
        dialog.source_transform_operation()
    dialog.accept()

    assert len(warning_calls) == 1
    xarray.testing.assert_identical(win.slicer_area._data.rename(None), data)

    dialog.close()
    win.close()


def test_leading_edge_dialog_defaults_to_first_dimension(qtbot) -> None:
    data = xr.DataArray(
        np.arange(6, dtype=float).reshape((2, 3)),
        dims=["1 axis", "energy"],
        coords={"1 axis": np.arange(2), "energy": np.arange(3)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    dialog = LeadingEdgeDialog(win.slicer_area)

    assert dialog.dim_combo.currentData(QtCore.Qt.ItemDataRole.UserRole) == "1 axis"

    dialog.close()
    win.close()


@pytest.mark.parametrize(
    "coord",
    [
        np.zeros((2, 2)),
        np.array(["a", "b"]),
        np.array([0.0, np.nan]),
        np.array([0.0, 0.0]),
    ],
)
def test_leading_edge_dialog_source_coord_validation(qtbot, coord) -> None:
    data = xr.DataArray(
        np.arange(6, dtype=float).reshape((2, 3)),
        dims=["x", "eV"],
        coords={"x": np.arange(2), "eV": np.arange(3)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    dialog = LeadingEdgeDialog(win.slicer_area)
    dialog._source_data = {"eV": types.SimpleNamespace(values=coord)}

    assert dialog._source_coord_error("eV") is not None

    dialog.close()
    win.close()


def test_leading_edge_dialog_rejects_invalid_source_coord(qtbot, monkeypatch) -> None:
    data = xr.DataArray(
        np.arange(10, dtype=float).reshape((2, 5)),
        dims=["x", "eV"],
        coords={"x": np.arange(2), "eV": np.array([0.0, 1.0, 1.0, 2.0, 3.0])},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    dialog = LeadingEdgeDialog(win.slicer_area)
    dialog.launch_mode_combo.setCurrentText("Replace Current")

    warning_calls: list[tuple[object, ...]] = []

    def _record_warning(*args, **kwargs):
        warning_calls.append(args)
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _record_warning)
    assert dialog.make_code() == ""
    dialog.accept()

    assert len(warning_calls) == 1
    xarray.testing.assert_identical(win.slicer_area._data.rename(None), data)

    dialog.close()
    win.close()


def test_leading_edge_dialog_make_code_suppresses_operation_errors(
    qtbot, monkeypatch
) -> None:
    data = xr.DataArray(
        np.arange(10, dtype=float).reshape((2, 5)),
        dims=["x", "eV"],
        coords={"x": np.arange(2), "eV": np.arange(5)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    dialog = LeadingEdgeDialog(win.slicer_area)

    def _raise_operation_error():
        raise RuntimeError

    monkeypatch.setattr(dialog, "source_transform_operation", _raise_operation_error)
    assert dialog.make_code() == ""

    dialog.close()
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
        pyperclip.paste() == "data.coarsen("
        'x=2, y=4, boundary="trim", side="right", coord_func="min"'
        ").sum()"
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
        _set_spinbox_text(dialog.coord_widget.scale_spin, "1.0")
        _set_spinbox_text(dialog.coord_widget.offset_spin, "-3.7")
        dialog.launch_mode_combo.setCurrentText("Replace Current")

    accept_dialog(win.mnb._assign_coords, pre_call=_set_dialog_params, timeout=10.0)
    np.testing.assert_allclose(
        win.slicer_area._data.y.values,
        np.arange(4) - 3.7,
        rtol=0,
        atol=0,
    )
    assert win.provenance_spec is not None
    display_code = win.provenance_spec.display_code(parent_data=data)
    assert display_code is not None
    assert "['y'].values - 3.7" in display_code
    assert "1.0 *" not in display_code
    assert "+ -3.7" not in display_code


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
    assert code == 'data.swap_dims(x="u", y="v")'
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

    expected = (data / data.mesh_current).rename("scan")
    xarray.testing.assert_identical(win.slicer_area._data, expected)
    copied_code = pyperclip.paste()
    assert "data.mesh_current" in copied_code
    assert ".rename(" not in copied_code
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
    dialog.coord_combo.setCurrentText("2 current")

    warnings: list[tuple[str, str]] = []

    def _record_warning(_parent, title, message, *args, **kwargs):
        warnings.append((title, message))
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _record_warning)

    dialog.coord_combo.setCurrentIndex(-1)
    dialog._update_coord_dims_label()
    assert dialog.coord_dims_label.text() == ""
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
    assert ".rename(" not in display_code
    assert "x_idx" not in display_code
    namespace = {"data": data.copy(deep=True)}
    exec(display_code, {}, namespace)  # noqa: S102
    xarray.testing.assert_identical(
        namespace["derived"].rename(None), (data / data.mesh_current).rename(None)
    )

    win.close()


def test_itool_gaussian_filter_sigma_path(qtbot, accept_dialog, monkeypatch) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["x", "y"],
        coords={"x": np.linspace(0.0, 0.04, 5), "y": np.arange(5, dtype=float)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    monkeypatch.setattr(
        type(win.slicer_area),
        "watched_data_name",
        property(lambda _self: "data"),
    )

    def _set_normalize_params(dialog: NormalizeDialog) -> None:
        dialog.dim_checks["x"].setChecked(True)

    accept_dialog(win.mnb._normalize, pre_call=_set_normalize_params)
    xarray.testing.assert_identical(win.slicer_area.data, normalize(data, ("x",), 0))

    sigma_literal = "0.0151234567890123"

    def _set_gaussian_params(dialog: GaussianFilterDialog) -> None:
        dialog.dim_checks["x"].setChecked(True)
        _set_spinbox_text(dialog.sigma_spins["x"], sigma_literal)
        assert dialog.sigma_spins["x"].text() == sigma_literal
        code = dialog.make_code()
        assert f'sigma={{"x": {sigma_literal}}}' in code
        namespace = _exec_generated_code(
            f"result = {code}",
            {"data": data.copy(deep=True)},
        )
        result = namespace["result"]
        assert isinstance(result, xr.DataArray)
        xarray.testing.assert_identical(
            result,
            erlab.analysis.image.gaussian_filter(
                data, sigma={"x": float(sigma_literal)}
            ),
        )
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


def test_itool_filter_dialog_reopens_with_current_settings(
    qtbot, accept_dialog
) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["x", "y"],
        coords={"x": np.linspace(0.0, 0.04, 5), "y": np.arange(5, dtype=float)},
    )
    first_sigma = 0.015
    second_sigma = 0.02
    first_operation = erlab.interactive.imagetool.provenance.GaussianFilterOperation(
        sigma={"x": first_sigma}
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    win.slicer_area.apply_filter_operation(first_operation)
    win.mnb._view_menu_visibility()
    assert win.mnb.action_dict["resetAct"].isEnabled()

    def _set_gaussian_params(dialog: GaussianFilterDialog) -> None:
        assert dialog.dim_checks["x"].isChecked()
        np.testing.assert_allclose(dialog.sigma_spins["x"].value(), first_sigma)
        assert not dialog.dim_checks["y"].isChecked()
        _set_spinbox_text(dialog.sigma_spins["x"], str(second_sigma))

    accept_dialog(win.mnb._gaussian_filter, pre_call=_set_gaussian_params)

    expected = erlab.analysis.image.gaussian_filter(data, sigma={"x": second_sigma})
    xarray.testing.assert_identical(win.slicer_area.data, expected)

    win.close()


def test_normalize_filter_dialog_reopens_with_current_settings(qtbot) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["x", "y"],
        coords={"x": np.arange(5, dtype=float), "y": np.arange(5, dtype=float)},
    )
    operation = erlab.interactive.imagetool.provenance.NormalizeOperation(
        dims=("x",),
        mode="min_area",
        denominator_rtol=1e-9,
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    win.slicer_area.apply_filter_operation(operation)

    base_dialog = imagetool_dialogs.DataFilterDialog(win.slicer_area)
    qtbot.addWidget(base_dialog)

    dialog = NormalizeDialog(win.slicer_area)
    qtbot.addWidget(dialog)

    assert dialog.dim_checks["x"].isChecked()
    assert not dialog.dim_checks["y"].isChecked()
    assert dialog.opts[3].isChecked()
    assert dialog.denominator_rtol == pytest.approx(1e-9)

    dialog.restore_filter_operation(
        erlab.interactive.imagetool.provenance.GaussianFilterOperation(sigma={"x": 1.0})
    )
    assert dialog.dim_checks["x"].isChecked()
    assert dialog.opts[3].isChecked()

    base_dialog.close()
    dialog.close()
    win.close()


def test_gaussian_filter_restore_skips_unknown_dimensions(qtbot) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["x", "y"],
        coords={"x": np.arange(5, dtype=float), "y": np.arange(5, dtype=float)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    dialog = GaussianFilterDialog(win.slicer_area)
    qtbot.addWidget(dialog)

    dialog.restore_filter_operation(
        erlab.interactive.imagetool.provenance.GaussianFilterOperation(
            sigma={"missing": 1.0, "x": 2.0}
        )
    )

    assert dialog.dim_checks["x"].isChecked()
    assert not dialog.dim_checks["y"].isChecked()
    assert dialog.sigma_spins["x"].value() == pytest.approx(2.0)

    dialog.close()
    win.close()


def test_itool_filter_preview_reject_restores_active_filter(qtbot) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["x", "y"],
        coords={"x": np.linspace(0.0, 0.04, 5), "y": np.arange(5, dtype=float)},
    )
    first_operation = erlab.interactive.imagetool.provenance.GaussianFilterOperation(
        sigma={"x": 0.015}
    )
    second_operation = erlab.interactive.imagetool.provenance.GaussianFilterOperation(
        sigma={"x": 0.02}
    )
    first_expected = first_operation.apply(data, parent_data=data)
    second_expected = second_operation.apply(data, parent_data=data)
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    win.slicer_area.apply_filter_operation(first_operation)

    dialog = GaussianFilterDialog(win.slicer_area)
    qtbot.addWidget(dialog)
    _set_spinbox_text(dialog.sigma_spins["x"], "0.02")
    dialog._preview()
    xarray.testing.assert_identical(win.slicer_area.data, second_expected)
    xarray.testing.assert_identical(win.slicer_area.displayed_data, first_expected)
    display_spec = win.slicer_area.displayed_provenance_spec()
    assert display_spec is not None
    code = display_spec.display_code()
    assert code is not None
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    xarray.testing.assert_identical(namespace["derived"], first_expected)

    dialog.reject()

    xarray.testing.assert_identical(win.slicer_area.data, first_expected)
    xarray.testing.assert_identical(win.slicer_area.displayed_data, first_expected)
    assert win.slicer_area._applied_provenance_operation == first_operation
    win.close()


def test_accepted_filter_displayed_data_uses_materialized_filter(qtbot) -> None:
    data = xr.DataArray(
        np.arange(9, dtype=float).reshape((3, 3)),
        dims=["x", "y"],
        coords={"x": np.arange(3, dtype=float), "y": np.arange(3, dtype=float)},
    )
    operation = erlab.interactive.imagetool.provenance.NormalizeOperation(
        dims=("x",),
        mode="min",
    )
    win = ImageTool(data)
    qtbot.addWidget(win)
    win.slicer_area.apply_filter_operation(operation)
    accepted = win.slicer_area.data.copy(deep=True)

    data.values[:] *= 2.0

    recomputed = operation.apply(data, parent_data=data)
    assert float(recomputed.values[1, 0]) != float(accepted.values[1, 0])
    xarray.testing.assert_identical(win.slicer_area.data, accepted)
    xarray.testing.assert_identical(win.slicer_area.displayed_data, accepted)
    xarray.testing.assert_identical(
        win.slicer_area._tool_source_parent_data(), accepted
    )

    win.slicer_area._accepted_filter_data = None
    with pytest.raises(RuntimeError, match="Accepted filter data is missing"):
        _ = win.slicer_area.displayed_data
    win.close()


def test_filter_helpers_reject_invalid_normalized_results(qtbot, monkeypatch) -> None:
    data = xr.DataArray(
        np.arange(9, dtype=float).reshape((3, 3)),
        dims=["x", "y"],
        coords={"x": np.arange(3, dtype=float), "y": np.arange(3, dtype=float)},
    )
    operation = erlab.interactive.imagetool.provenance.NormalizeOperation(
        dims=("x",),
        mode="min",
    )
    win = ImageTool(data)
    qtbot.addWidget(win)

    func = win.slicer_area._filter_func_from_operation(operation)
    xarray.testing.assert_identical(func(data), operation.apply(data, parent_data=data))

    monkeypatch.setattr(
        win.slicer_area,
        "_expected_layout_shape",
        lambda _source_data, _dims: (99, 99),
    )
    with pytest.raises(ValueError, match="shape does not match"):
        win.slicer_area._normalize_filter_result_for_source_dims(
            data,
            data.copy(deep=True),
            tuple(data.dims),
        )
    win.close()


def test_apply_filter_operation_caches_dask_accepted_filter_lazily(qtbot) -> None:
    da = pytest.importorskip("dask.array")
    from dask.callbacks import Callback

    data = xr.DataArray(
        da.from_array(np.arange(12, dtype=float).reshape((3, 4)), chunks=(2, 2)),
        dims=["x", "y"],
        coords={"x": np.arange(3, dtype=float), "y": np.arange(4, dtype=float)},
    )
    operation = erlab.interactive.imagetool.provenance.NormalizeOperation(
        dims=("x",),
        mode="min",
    )
    win = ImageTool(data, auto_compute=False)
    qtbot.addWidget(win)
    source_replaced: list[xr.DataArray] = []
    data_edited: list[bool] = []
    win.slicer_area.sigSourceDataReplaced.connect(source_replaced.append)
    win.slicer_area.sigDataEdited.connect(lambda: data_edited.append(True))

    computed_keys: list[object] = []
    with Callback(pretask=lambda key, _dsk, _state: computed_keys.append(key)):
        win.slicer_area.apply_filter_operation(
            operation,
            update=False,
            emit_edited=True,
        )
        cached = win.slicer_area.displayed_data

    assert len(source_replaced) == 1
    assert data_edited == [True]
    assert computed_keys == []
    assert win.slicer_area.data.chunks is not None
    assert cached.chunks is not None
    xarray.testing.assert_identical(
        cached.compute(),
        operation.apply(data, parent_data=data).compute(),
    )
    win.close()


def test_compute_chunked_preserves_accepted_filter(qtbot) -> None:
    da = pytest.importorskip("dask.array")

    data = xr.DataArray(
        da.from_array(np.arange(12, dtype=float).reshape((3, 4)), chunks=(2, 2)),
        dims=["x", "y"],
        coords={"x": np.arange(3, dtype=float), "y": np.arange(4, dtype=float)},
    )
    operation = erlab.interactive.imagetool.provenance.NormalizeOperation(
        dims=("x",),
        mode="min",
    )
    win = ImageTool(data, auto_compute=False)
    qtbot.addWidget(win)
    win.slicer_area.apply_filter_operation(operation, update=False)

    win.slicer_area._compute_chunked()

    loaded = data.compute()
    expected = operation.apply(loaded, parent_data=loaded)
    assert win.slicer_area._data.chunks is None
    assert win.slicer_area._accepted_filter_provenance_operation == operation
    xarray.testing.assert_identical(win.slicer_area.data, expected)
    xarray.testing.assert_identical(win.slicer_area.displayed_data, expected)
    win.close()


def test_itool_reload_reapplies_accepted_filter(qtbot, tmp_path: pathlib.Path) -> None:
    data = xr.DataArray(
        np.arange(25, dtype=float).reshape((5, 5)),
        dims=["x", "y"],
        coords={"x": np.linspace(0.0, 0.04, 5), "y": np.arange(5, dtype=float)},
        name="scan",
    )
    updated = data + 100.0
    current = {"data": data}
    file_path = tmp_path / "scan.h5"
    file_path.touch()

    def _load_current(_path: str) -> xr.DataArray:
        return current["data"]

    operation = erlab.interactive.imagetool.provenance.GaussianFilterOperation(
        sigma={"x": 0.015}
    )
    win = ImageTool(
        data,
        file_path=file_path,
        load_func=(_load_current, {}, 0),
    )
    qtbot.addWidget(win)
    win.slicer_area.apply_filter_operation(operation)

    expected = operation.apply(data, parent_data=data)
    xarray.testing.assert_identical(win.slicer_area.data, expected)
    assert win.slicer_area.reloadable

    current["data"] = updated
    with qtbot.wait_signal(win.slicer_area.sigDataChanged):
        win.slicer_area.reload()

    updated_expected = operation.apply(updated, parent_data=updated)
    xarray.testing.assert_identical(win.slicer_area._data, updated)
    xarray.testing.assert_identical(win.slicer_area.data, updated_expected)
    xarray.testing.assert_identical(win.slicer_area.displayed_data, updated_expected)
    assert win.slicer_area._applied_provenance_operation == operation

    win.close()


def test_itool_reload_filter_failure_keeps_existing_filtered_data(
    qtbot, tmp_path: pathlib.Path, monkeypatch
) -> None:
    data = xr.DataArray(
        np.arange(25, dtype=float).reshape((5, 5)),
        dims=["x", "y"],
        coords={"x": np.linspace(0.0, 0.04, 5), "y": np.arange(5, dtype=float)},
        name="scan",
    )
    bad_reload = xr.DataArray(
        np.arange(25, dtype=float).reshape((5, 5)),
        dims=["u", "y"],
        coords={"u": np.arange(5, dtype=float), "y": np.arange(5, dtype=float)},
        name="scan",
    )
    current = {"data": data}
    file_path = tmp_path / "scan.h5"
    file_path.touch()

    def _load_current(_path: str) -> xr.DataArray:
        return current["data"]

    operation = erlab.interactive.imagetool.provenance.GaussianFilterOperation(
        sigma={"x": 0.015}
    )
    win = ImageTool(
        data,
        file_path=file_path,
        load_func=(_load_current, {}, 0),
    )
    qtbot.addWidget(win)
    win.slicer_area.apply_filter_operation(operation)
    expected_source = win.slicer_area._data.copy(deep=True)
    expected_display = win.slicer_area.data.copy(deep=True)
    errors: list[tuple[str, str]] = []

    def _critical(parent, title, text, informative_text="", detailed_text=None):
        del parent, informative_text, detailed_text
        errors.append((title, text))

    monkeypatch.setattr(erlab.interactive.utils.MessageDialog, "critical", _critical)

    current["data"] = bad_reload
    assert not win.slicer_area._reload()

    assert errors == [("Error", "An error occurred while reloading data.")]
    xarray.testing.assert_identical(win.slicer_area._data, expected_source)
    xarray.testing.assert_identical(win.slicer_area.data, expected_display)
    assert win.slicer_area._accepted_filter_provenance_operation == operation

    win.close()


def test_itool_empty_filter_accept_clears_active_filter(qtbot) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["x", "y"],
        coords={"x": np.linspace(0.0, 0.04, 5), "y": np.arange(5, dtype=float)},
    )
    operation = erlab.interactive.imagetool.provenance.GaussianFilterOperation(
        sigma={"x": 0.015}
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    win.slicer_area.apply_filter_operation(operation)

    dialog = GaussianFilterDialog(win.slicer_area)
    qtbot.addWidget(dialog)
    dialog.dim_checks["x"].setChecked(False)
    dialog.accept()

    assert win.slicer_area._applied_func is None
    assert win.slicer_area._applied_provenance_operation is None
    xarray.testing.assert_identical(win.slicer_area.data, data)
    win.close()


def test_itool_filter_accept_and_reset_are_undoable(qtbot, accept_dialog) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["x", "y"],
        coords={"x": np.linspace(0.0, 0.04, 5), "y": np.arange(5, dtype=float)},
    )
    sigma = 0.015
    operation = erlab.interactive.imagetool.provenance.GaussianFilterOperation(
        sigma={"x": sigma}
    )
    expected = operation.apply(data, parent_data=data)
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    def _set_filter(dialog: GaussianFilterDialog) -> None:
        dialog.dim_checks["x"].setChecked(True)
        _set_spinbox_text(dialog.sigma_spins["x"], str(sigma))

    accept_dialog(win.mnb._gaussian_filter, pre_call=_set_filter)

    xarray.testing.assert_identical(win.slicer_area.data, expected)
    assert win.slicer_area.undoable

    with qtbot.wait_signal(win.slicer_area.sigSourceDataReplaced):
        win.slicer_area.undo()
    xarray.testing.assert_identical(win.slicer_area.data, data)
    assert win.slicer_area._accepted_filter_provenance_operation is None

    with qtbot.wait_signal(win.slicer_area.sigSourceDataReplaced):
        win.slicer_area.redo()
    xarray.testing.assert_identical(win.slicer_area.data, expected)
    assert win.slicer_area._accepted_filter_provenance_operation == operation

    win.mnb._reset_filters()
    xarray.testing.assert_identical(win.slicer_area.data, data)

    with qtbot.wait_signal(win.slicer_area.sigSourceDataReplaced):
        win.slicer_area.undo()
    xarray.testing.assert_identical(win.slicer_area.data, expected)
    assert win.slicer_area._accepted_filter_provenance_operation == operation
    win.close()


def test_itool_normalize_filter_copies_code_and_records_display_provenance(
    qtbot, accept_dialog, monkeypatch
) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["x", "y"],
        coords={"x": np.arange(5, dtype=float), "y": np.arange(5, dtype=float)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    monkeypatch.setattr(
        type(win.slicer_area),
        "watched_data_name",
        property(lambda _self: "data"),
    )

    def _set_normalize_params(dialog: NormalizeDialog) -> None:
        dialog.dim_checks["x"].setChecked(True)
        dialog.opts[1].setChecked(True)
        with qtbot.wait_signal(dialog._sigCodeCopied):
            dialog.copy_button.click()

    accept_dialog(win.mnb._normalize, pre_call=_set_normalize_params)

    expected = normalize(data, ("x",), 1)
    xarray.testing.assert_identical(win.slicer_area.data, expected)
    assert ".min(" in pyperclip.paste()
    assert ".max(" in pyperclip.paste()
    xarray.testing.assert_identical(
        _exec_data_fragment(data, pyperclip.paste()),
        expected,
    )

    display_code = win.slicer_area.displayed_provenance_spec()
    assert display_code is not None
    code = display_code.display_code()
    assert code is not None
    assert ".min(" in code
    assert ".max(" in code
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    xarray.testing.assert_identical(namespace["derived"], expected)

    win.close()


def test_itool_gaussian_filter_fwhm_path_and_code(qtbot, monkeypatch) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["x", "y"],
        coords={"x": np.linspace(0.0, 0.04, 5), "y": np.arange(5, dtype=float)},
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)
    monkeypatch.setattr(
        type(win.slicer_area),
        "watched_data_name",
        property(lambda _self: "data"),
    )

    dialog = GaussianFilterDialog(win.slicer_area)
    qtbot.addWidget(dialog)

    fwhm_literal = "0.0351234567890123"
    expected_sigma = float(fwhm_literal) / (2 * np.sqrt(2 * np.log(2)))
    dialog.dim_checks["x"].setChecked(True)
    _set_spinbox_text(dialog.fwhm_spins["x"], fwhm_literal)

    sigma_literal = dialog.sigma_spins["x"].text()
    assert sigma_literal == str(expected_sigma)
    assert dialog.fwhm_spins["x"].text() == fwhm_literal

    code = dialog.make_code()
    namespace = _exec_generated_code(
        f"result = {code}",
        {"data": data.copy(deep=True)},
    )
    result = namespace["result"]
    assert isinstance(result, xr.DataArray)
    xarray.testing.assert_identical(
        result,
        erlab.analysis.image.gaussian_filter(data, sigma={"x": expected_sigma}),
    )

    xarray.testing.assert_identical(
        dialog.process_data(data),
        erlab.analysis.image.gaussian_filter(data, sigma={"x": expected_sigma}),
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
