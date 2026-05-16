import ast
import concurrent.futures
import contextlib
import dataclasses
import enum
import errno
import gc
import io
import json
import logging
import os
import pathlib
import pickle
import subprocess
import sys
import tempfile
import time
import types
import typing
import webbrowser
from collections.abc import Callable, Iterable, Mapping

import numpy as np
import pydantic
import pytest
import xarray as xr
import xarray.testing
import zmq
from IPython.core.interactiveshell import InteractiveShell
from qtpy import QtCore, QtGui, QtWidgets

import erlab
import erlab.interactive.imagetool._itool as itool_mod
import erlab.interactive.imagetool.manager._dialogs as manager_dialogs
import erlab.interactive.imagetool.manager._mainwindow as manager_mainwindow
import erlab.interactive.imagetool.manager._registry as manager_registry
import erlab.interactive.imagetool.manager._server as manager_server
import erlab.interactive.imagetool.manager._workspace as manager_workspace
import erlab.interactive.imagetool.manager._xarray as manager_xarray
from erlab.interactive._fit1d import Fit1DTool
from erlab.interactive._fit2d import Fit2DTool
from erlab.interactive._mesh import MeshTool
from erlab.interactive.derivative import DerivativeTool
from erlab.interactive.explorer._tabbed_explorer import _TabbedExplorer
from erlab.interactive.fermiedge import GoldTool, ResolutionTool
from erlab.interactive.imagetool import itool
from erlab.interactive.imagetool._load_source import (
    _load_code_from_file_details,
    _load_provenance_from_file_details,
    _load_source_label_and_text,
    _loader_callable_text,
    _LoadSourceDetails,
    _resolve_identified_path,
    _scan_number_load_call_args,
)
from erlab.interactive.imagetool._magic import _normalize_manager_target_args
from erlab.interactive.imagetool.manager import (
    ImageToolManager,
    fetch,
    load_in_manager,
    replace_data,
)
from erlab.interactive.imagetool.manager._console import (
    ToolNamespace,
    _ConsoleDataAssignmentTransformer,
    _rewrite_console_source,
)
from erlab.interactive.imagetool.manager._dialogs import (
    _ChooseFromDataTreeDialog,
    _ConcatDialog,
    _CoordinateAttrsPickerDialog,
    _NameFilterDialog,
    _NameMapEditorDialog,
    _RenameDialog,
    _text_to_loader_extension_value,
)
from erlab.interactive.imagetool.manager._mainwindow import _LoadSourceDetailsDialog
from erlab.interactive.imagetool.manager._modelview import (
    _MIME,
    _NODE_UID_ROLE,
    _TOOL_TYPE_ROLE,
    _ImageToolWrapperItemDelegate,
    _ImageToolWrapperItemModel,
    _RowBadge,
)
from erlab.interactive.imagetool.manager._server import (
    AddDataPacket,
    Response,
    _recv_multipart,
    _remove_idx,
    _show_idx,
    _WatcherServer,
)
from erlab.interactive.imagetool.manager._wrapper import (
    _format_chunk_summary,
    _preview_from_imagetool,
)
from erlab.interactive.ptable import PeriodicTableWindow

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def test_data():
    return xr.DataArray(
        np.arange(25).reshape((5, 5)),
        dims=["alpha", "eV"],
        coords={"alpha": np.arange(5), "eV": np.arange(5)},
    )


def select_tools(
    manager: ImageToolManager, indices: list[int], deselect: bool = False
) -> None:
    selection_model = manager.tree_view.selectionModel()

    for index in indices:
        qmodelindex = manager.tree_view._model._row_index(index)
        selection_model.select(
            QtCore.QItemSelection(qmodelindex, qmodelindex),
            QtCore.QItemSelectionModel.SelectionFlag.Deselect
            if deselect
            else QtCore.QItemSelectionModel.SelectionFlag.Select,
        )


def select_child_tool(
    manager: ImageToolManager, uid: str, deselect: bool = False
) -> None:
    selection_model = manager.tree_view.selectionModel()

    qmodelindex = manager.tree_view._model._row_index(uid)
    selection_model.select(
        QtCore.QItemSelection(qmodelindex, qmodelindex),
        QtCore.QItemSelectionModel.SelectionFlag.Deselect
        if deselect
        else QtCore.QItemSelectionModel.SelectionFlag.Select,
    )


def click_tree_view_pos(
    view: QtWidgets.QTreeView,
    pos: QtCore.QPoint,
) -> None:
    global_pos = view.viewport().mapToGlobal(pos)
    view.mouseReleaseEvent(
        QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseButtonRelease,
            QtCore.QPointF(pos),
            QtCore.QPointF(global_pos),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
    )


def assert_nonempty_tooltip(text: str | None) -> None:
    assert isinstance(text, str)
    assert text.strip()


def _use_isolated_manager_registry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
):
    registry_path = tmp_path / "managers.json"
    monkeypatch.setattr(manager_registry, "_REGISTRY_PATH", registry_path)
    monkeypatch.setattr(
        manager_registry,
        "_LOCK_PATH",
        registry_path.with_suffix(registry_path.suffix + ".lock"),
    )
    manager_registry.clear_default_manager()
    return manager_registry


def child_status_badge(
    manager: ImageToolManager, uid: str
) -> tuple[QtCore.QRect, str | None, QtCore.QModelIndex]:
    model = typing.cast("_ImageToolWrapperItemModel", manager.tree_view.model())
    delegate = typing.cast(
        "_ImageToolWrapperItemDelegate", manager.tree_view.itemDelegate()
    )

    ancestors: list[str] = []
    node = manager._child_node(uid)
    while node.parent_uid is not None:
        ancestors.append(node.parent_uid)
        parent = manager._all_nodes[node.parent_uid]
        if not hasattr(parent, "parent_uid"):
            break
        node = typing.cast("typing.Any", parent)
    for ancestor_uid in reversed(ancestors):
        manager.tree_view.expand(model._row_index(ancestor_uid))

    index = model._row_index(uid)
    manager.tree_view.scrollTo(index)
    option = QtWidgets.QStyleOptionViewItem()
    option.rect = manager.tree_view.visualRect(index)
    option.font = manager.tree_view.font()
    badge_rect, badge_text, _ = delegate._compute_child_status_info(
        option, manager._child_node(uid)
    )
    assert badge_rect is not None
    return badge_rect, badge_text, index


def click_child_status_badge(
    manager: ImageToolManager,
    uid: str,
    accept_dialog,
    *,
    pre_call: Callable[[QtWidgets.QDialog], None] | None = None,
    accept_call: Callable[[QtWidgets.QDialog], None] | None = None,
) -> None:
    badge_rect, _, _ = child_status_badge(manager, uid)
    click_pos = badge_rect.center()
    global_click_pos = manager.tree_view.viewport().mapToGlobal(click_pos)
    accept_dialog(
        lambda: manager.tree_view.mouseReleaseEvent(
            QtGui.QMouseEvent(
                QtCore.QEvent.Type.MouseButtonRelease,
                QtCore.QPointF(click_pos),
                QtCore.QPointF(global_click_pos),
                QtCore.Qt.MouseButton.LeftButton,
                QtCore.Qt.MouseButton.LeftButton,
                QtCore.Qt.KeyboardModifier.NoModifier,
            )
        ),
        pre_call=pre_call,
        accept_call=accept_call,
    )


def configure_goldtool_child(
    tool: GoldTool, *, fitted: bool = False, spline: bool = False
) -> None:
    with (
        QtCore.QSignalBlocker(tool.params_edge),
        QtCore.QSignalBlocker(tool.params_poly),
        QtCore.QSignalBlocker(tool.params_spl),
        QtCore.QSignalBlocker(tool.params_tab),
    ):
        tool._restore_parameter_group_values(
            tool.params_edge,
            {
                "T (K)": 45.0,
                "Fix T": False,
                "Bin x": 2,
                "Bin y": 3,
                "Resolution": 0.015,
                "Fast": False,
                "Linear": False,
                "Method": "cg",
                "Scale cov": False,
                "# CPU": 1,
            },
        )
        tool._restore_parameter_group_values(
            tool.params_poly,
            {
                "Degree": 3,
                "Scale cov": False,
                "Residuals": True,
                "Corrected": not spline,
                "Shift coords": False,
            },
        )
        tool._restore_parameter_group_values(
            tool.params_spl,
            {
                "Auto": False,
                "lambda": 2.5,
                "Residuals": True,
                "Corrected": spline,
                "Shift coords": False,
            },
        )
        tool.params_tab.setCurrentIndex(1 if spline else 0)

    tool.params_roi.modify_roi(x0=-8.0, x1=8.0, y0=-0.2, y1=0.1)
    tool.refit_on_source_update_check.setChecked(True)
    tool._toggle_fast()
    tool._sync_spline_lambda_enabled()

    if fitted:
        edge_center = tool.data.mean("eV")
        edge_stderr = xr.ones_like(edge_center)
        tool.post_fit(edge_center, edge_stderr)


class _UnserializableChildState(pydantic.BaseModel):
    value: int = 0


class _UnserializableChildTool(
    erlab.interactive.utils.ToolWindow[_UnserializableChildState]
):
    StateModel = _UnserializableChildState
    tool_name = "unserializable-child"

    def __init__(self, data: xr.DataArray) -> None:
        super().__init__()
        self._data = data
        self._status = _UnserializableChildState()

    @property
    def tool_data(self) -> xr.DataArray:
        return self._data

    @property
    def tool_status(self) -> _UnserializableChildState:
        return self._status

    @tool_status.setter
    def tool_status(self, status: _UnserializableChildState) -> None:
        self._status = status

    def _raise_serialization_error(self) -> typing.NoReturn:
        raise ValueError(
            "goldtool save/load/duplication is unsupported when `data_corr` "
            "is provided separately"
        )

    def to_dataset(self) -> xr.Dataset:
        self._raise_serialization_error()

    def duplicate(self, **kwargs) -> typing.Self:
        self._raise_serialization_error()


def make_fit2d_child(
    manager: ImageToolManager, parent: int | str, exp_decay_model
) -> tuple[str, Fit2DTool]:
    t = np.linspace(0.0, 4.0, 25)
    y = np.arange(3)
    data = xr.DataArray(
        np.stack([((1.0 + 0.5 * idx) * np.exp(-t / 2.0)) for idx in y], axis=0),
        dims=("y", "t"),
        coords={"y": y, "t": t},
        name="decay2d",
    )
    params = exp_decay_model.make_params(n0=1.0, tau=1.0)
    tool = erlab.interactive.ftool(
        data, model=exp_decay_model, params=params, execute=False
    )
    assert isinstance(tool, Fit2DTool)
    child_uid = manager.add_childtool(tool, parent, show=False)
    return child_uid, tool


def make_fit1d_child(
    manager: ImageToolManager, parent: int | str, exp_decay_model
) -> tuple[str, Fit1DTool]:
    t = np.linspace(0.0, 4.0, 25)
    data = xr.DataArray(
        3.0 * np.exp(-t / 2.0),
        dims=("t",),
        coords={"t": t},
        name="decay",
    )
    params = exp_decay_model.make_params(n0=2.0, tau=1.0)
    tool = erlab.interactive.ftool(
        data, model=exp_decay_model, params=params, execute=False
    )
    assert isinstance(tool, Fit1DTool)
    child_uid = manager.add_childtool(tool, parent, show=False)
    return child_uid, tool


def manager_preview_pixmap(manager: ImageToolManager) -> QtGui.QPixmap:
    return manager.preview_widget._pixmapitem.pixmap()


def metadata_derivation_texts(manager: ImageToolManager) -> list[str]:
    return [
        manager.metadata_derivation_list.item(row).text()
        for row in range(manager.metadata_derivation_list.count())
    ]


def metadata_detail_map(manager: ImageToolManager) -> dict[str, str]:
    details: dict[str, str] = {}
    for key, label in manager._metadata_detail_labels.items():
        details[key] = getattr(label, "full_text", label.text())
    return details


def select_metadata_rows(
    manager: ImageToolManager, rows: list[int], clear: bool = True
) -> None:
    if clear:
        manager.metadata_derivation_list.clearSelection()
    for row in rows:
        item = manager.metadata_derivation_list.item(row)
        if item is None:
            continue
        item.setSelected(True)
        manager.metadata_derivation_list.setCurrentItem(item)


def set_transform_launch_mode(
    dialog: QtWidgets.QDialog, mode: typing.Literal["replace", "detach", "nest"]
) -> None:
    labels = {
        "replace": "Replace Current",
        "detach": "Open Top-Level Window",
        "nest": "Open Child Window",
    }
    dialog.launch_mode_combo.setCurrentText(labels[mode])  # type: ignore[attr-defined]
    assert dialog.launch_mode == mode  # type: ignore[attr-defined]


def make_drop_event(filename: str) -> QtGui.QDropEvent:
    mime_data = QtCore.QMimeData()
    mime_data.setUrls([QtCore.QUrl.fromLocalFile(filename)])
    return QtGui.QDropEvent(
        QtCore.QPointF(0.0, 0.0),
        QtCore.Qt.DropAction.CopyAction,
        mime_data,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )


def bring_manager_to_top(bot, manager):
    with bot.waitExposed(manager):
        manager.hide_all()  # Prevent windows from obstructing the manager
        manager.activateWindow()
        manager.raise_()


def action_map(menu: QtWidgets.QMenu) -> dict[str, QtWidgets.QAction]:
    return {
        action.text().replace("&", ""): action
        for action in menu.actions()
        if not action.isSeparator()
    }


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


def _assert_modelfit_code_replays_source(
    code: str, source_name: str, source: xr.DataArray
) -> None:
    namespace = _exec_generated_code(code, {source_name: source.copy(deep=True)})
    result = namespace["result"]
    assert isinstance(result, xr.Dataset)
    assert "derived" not in namespace

    expected = source
    if expected.dtype not in (np.float32, np.float64):
        expected = expected.astype(np.float64)
    xr.testing.assert_identical(
        result["modelfit_data"].rename(expected.name),
        expected,
    )


def assert_fit_result_dataset_equivalent(
    actual: xr.Dataset, expected: xr.Dataset
) -> None:
    xr.testing.assert_identical(
        actual.drop_vars("modelfit_results"),
        expected.drop_vars("modelfit_results"),
    )
    actual_result = actual.modelfit_results.compute().item()
    expected_result = expected.modelfit_results.compute().item()
    assert type(actual_result.model) is type(expected_result.model)
    assert list(actual_result.params.keys()) == list(expected_result.params.keys())
    for name, expected_param in expected_result.params.items():
        actual_param = actual_result.params[name]
        assert actual_param.value == pytest.approx(expected_param.value)
        if expected_param.stderr is None:
            assert actual_param.stderr is None
        else:
            assert actual_param.stderr == pytest.approx(expected_param.stderr)
        assert actual_param.expr == expected_param.expr
        assert actual_param.vary == expected_param.vary


def assert_fit_result_list_equivalent(
    actual: list[xr.Dataset | None], expected: list[xr.Dataset | None]
) -> None:
    assert len(actual) == len(expected)
    for actual_ds, expected_ds in zip(actual, expected, strict=True):
        if expected_ds is None:
            assert actual_ds is None
            continue
        assert actual_ds is not None
        assert_fit_result_dataset_equivalent(actual_ds, expected_ds)


def copy_full_code_for_uid(
    monkeypatch,
    manager: ImageToolManager,
    uid: str,
    *,
    source_name: str = "data",
) -> str:
    copied: list[str] = []
    monkeypatch.setattr(
        erlab.interactive.utils,
        "copy_to_clipboard",
        lambda text: copied.append(text) or text,
    )
    monkeypatch.setattr(
        manager,
        "_prompt_replay_input_name",
        lambda _node: source_name,
    )
    manager.tree_view.clearSelection()
    select_child_tool(manager, uid)
    manager._update_info(uid=uid)
    menu = manager._build_metadata_derivation_menu()
    assert menu is not None
    action_map(menu)["Copy Full Code"].trigger()
    assert copied
    return copied[-1]


def test_load_source_details_dialog_kwargs_editor_wraps_and_highlights(
    qtbot, tmp_path
) -> None:
    kwargs_text = (
        'engine="h5netcdf", '
        "very_long_keyword_argument_name=123, "
        'another_long_keyword_argument_name="abcdef"'
    )
    dialog = _LoadSourceDetailsDialog(
        _LoadSourceDetails(
            path=tmp_path / "scan.nc",
            loader_label="Loader",
            loader_text="xarray.load_dataarray",
            kwargs_text=kwargs_text,
            load_code=None,
        )
    )
    qtbot.addWidget(dialog)
    dialog.show()

    qtbot.wait_until(lambda: dialog.kwargs_edit._visual_row_count() > 1, timeout=2000)

    expected_rows = min(
        dialog.kwargs_edit._MAX_VISIBLE_ROWS,
        dialog.kwargs_edit._visual_row_count(),
    )
    assert dialog.kwargs_edit.height() == (
        expected_rows * dialog.kwargs_edit.fontMetrics().lineSpacing()
        + dialog.kwargs_edit._VERTICAL_PADDING
    )
    assert isinstance(
        dialog.kwargs_highlighter, erlab.interactive.utils.PythonHighlighter
    )
    dialog.kwargs_edit.setPlainText(None)
    assert dialog.kwargs_edit.toPlainText() == ""


@pytest.mark.parametrize("use_socket", [False, True], ids=["no_socket", "socket"])
def test_manager(
    qtbot,
    accept_dialog,
    test_data,
    use_socket,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context(use_socket=use_socket) as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        logger.info("Manager is running, adding test data")
        test_data.qshow(manager=True)

        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        logger.info("Confirmed tool is added, checking data")
        assert manager.get_imagetool(0).array_slicer.point_value(0) == 12.0

        logger.info("Checking data retrieval via fetch")

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(fetch, 0)

            # Let the GUI thread keep processing events while waiting
            qtbot.waitUntil(lambda: fut.done(), timeout=10000)

            result = fut.result()

        xr.testing.assert_identical(result, test_data)

        logger.info("Confirmed fetch works, adding more tools")
        # Add two tools
        for tool in itool(
            [test_data, test_data], link=False, execute=False, manager=False
        ):
            tool.move_to_manager()

        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)

        # Linking
        select_tools(manager, [1, 2])
        manager.link_selected()
        manager.tree_view.refresh(None)

        # Unlinking one unlinks both
        select_tools(manager, [1])
        manager.unlink_selected()
        manager.tree_view.refresh(None)
        assert not manager.get_imagetool(1).slicer_area.is_linked
        assert not manager.get_imagetool(2).slicer_area.is_linked

        # Linking again
        select_tools(manager, [1, 2])
        manager.link_selected()
        manager.tree_view.refresh(None)
        assert manager.get_imagetool(1).slicer_area.is_linked
        assert manager.get_imagetool(2).slicer_area.is_linked

        # Archiving and unarchiving
        manager._imagetool_wrappers[1].archive()
        manager._imagetool_wrappers[1].touch_archive()
        assert manager._imagetool_wrappers[1].archived
        manager._imagetool_wrappers[1].unarchive()
        assert not manager._imagetool_wrappers[1].archived

        # Toggle visibility
        geometry = manager.get_imagetool(1).geometry()
        manager._imagetool_wrappers[1].hide()
        assert not manager.get_imagetool(1).isVisible()
        manager._imagetool_wrappers[1].show()
        assert manager.get_imagetool(1).geometry() == geometry

        # Removing archived tool
        manager._imagetool_wrappers[0].archive()
        manager.remove_imagetool(0)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        # Batch renaming
        select_tools(manager, [1, 2])

        def _handle_renaming(dialog: _RenameDialog):
            dialog._new_name_lines[1].setText("new_name_1")
            dialog._new_name_lines[2].setText("new_name_2")

        accept_dialog(manager.rename_action.trigger, pre_call=_handle_renaming)
        assert manager._imagetool_wrappers[1].name == "new_name_1"
        assert manager._imagetool_wrappers[2].name == "new_name_2"

        # Rename single
        select_tools(manager, [2], deselect=True)
        select_tools(manager, [1])
        manager.rename_action.trigger()

        qtbot.wait_until(
            lambda: (
                manager.tree_view.state()
                == QtWidgets.QAbstractItemView.State.EditingState
            ),
            timeout=5000,
        )
        delegate = manager.tree_view.itemDelegate()
        assert isinstance(delegate, _ImageToolWrapperItemDelegate)
        assert isinstance(delegate._current_editor, QtWidgets.QLineEdit)
        delegate._current_editor.setText("new_name_1_single")
        qtbot.keyClick(delegate._current_editor, QtCore.Qt.Key.Key_Return)
        qtbot.wait_until(
            lambda: manager._imagetool_wrappers[1].name == "new_name_1_single",
            timeout=5000,
        )

        # Select single tool
        select_tools(manager, [1])

        # Update info panel
        bring_manager_to_top(qtbot, manager)
        manager._update_info()

        # Archive & unarchive single
        manager.archive_action.trigger()
        qtbot.wait_until(lambda: manager._imagetool_wrappers[1].archived, timeout=5000)

        # Update info panel
        bring_manager_to_top(qtbot, manager)
        manager._update_info()

        manager._imagetool_wrappers[1].unarchive()

        # Batch archiving with show/hide
        select_tools(manager, [1])
        manager.archive_action.trigger()

        select_tools(manager, [1, 2])

        # Hide non-archived window, does nothing to archived window
        manager.hide_action.trigger()
        # Unarchive the archived one and show both
        manager.show_action.trigger()

        assert not manager._imagetool_wrappers[1].archived
        assert not manager._imagetool_wrappers[2].archived
        assert manager.get_imagetool(1).isVisible()
        assert manager.get_imagetool(2).isVisible()

        # Add third tool
        xr.concat(
            [
                manager.get_imagetool(1).slicer_area._data,
                manager.get_imagetool(2).slicer_area._data,
            ],
            "concat_dim",
        ).qshow(manager=True)
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)

        # Update info panel
        bring_manager_to_top(qtbot, manager)
        manager.tree_view.clearSelection()
        select_tools(manager, [1, 2, 3])
        manager._update_info()

        # Show goldtool
        logger.info("Opening goldtool")
        manager.get_imagetool(3).slicer_area.images[2].open_in_goldtool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[3]._childtools) == 1, timeout=5000
        )
        assert isinstance(
            next(iter(manager._imagetool_wrappers[3]._childtools.values())), GoldTool
        )
        logger.info("Confirmed goldtool is added")

        # Trigger paint event
        manager.tree_view.expandAll()

        # Test rename goldtool
        goldtool_uid: str = manager._imagetool_wrappers[3]._childtool_indices[0]

        # Bring manager to top
        manager.tree_view.clearSelection()
        bring_manager_to_top(qtbot, manager)
        select_child_tool(manager, goldtool_uid)

        manager._update_actions()
        assert manager.rename_action.isEnabled()
        manager.rename_action.trigger()
        qtbot.wait_until(
            lambda: (
                manager.tree_view.state()
                == QtWidgets.QAbstractItemView.State.EditingState
            ),
            timeout=5000,
        )
        delegate = manager.tree_view.itemDelegate()
        assert isinstance(delegate, _ImageToolWrapperItemDelegate)
        assert isinstance(delegate._current_editor, QtWidgets.QLineEdit)
        delegate._current_editor.setText("new_goldtool_name")
        qtbot.keyClick(delegate._current_editor, QtCore.Qt.Key.Key_Return)
        qtbot.wait_until(
            lambda: (
                next(
                    iter(manager._imagetool_wrappers[3]._childtools.values())
                )._tool_display_name
                == "new_goldtool_name"
            ),
            timeout=5000,
        )

        # Close goldtool
        logger.info("Closing goldtool")
        manager._remove_childtool(
            next(iter(manager._imagetool_wrappers[3]._childtools.keys()))
        )

        # Show dtool
        logger.info("Opening dtool")
        manager.get_imagetool(3).slicer_area.images[2].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[3]._childtools) == 1, timeout=5000
        )
        assert isinstance(
            next(iter(manager._imagetool_wrappers[3]._childtools.values())),
            DerivativeTool,
        )
        logger.info("Confirmed dtool is added")
        manager.tree_view.expandAll()
        tool_uid: str = manager._imagetool_wrappers[3]._childtool_indices[0]

        # Show dtool
        manager.show_childtool(tool_uid)

        # Tool and parent
        logger.info("Checking parent and childtool retrieval")
        tool, idx = manager._get_childtool_and_parent(tool_uid)
        assert isinstance(tool, DerivativeTool)
        assert idx == 3

        # Check dtool info printing
        bring_manager_to_top(qtbot, manager)
        manager.tree_view.clearSelection()
        select_child_tool(manager, tool_uid)
        manager._update_info(uid=tool_uid)

        # Duplicate dtool
        logger.info("Duplicating dtool")
        bring_manager_to_top(qtbot, manager)
        manager.tree_view.clearSelection()
        select_child_tool(manager, tool_uid)
        manager.duplicate_selected()
        manager.tree_view.refresh(None)

        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[3]._childtools) == 2, timeout=5000
        )

        # Check calling invalid indices
        logger.info("Checking invalid index handling")
        parent_qindex = manager.tree_view._model._row_index(3)
        assert manager.tree_view._model.index(1, 0, parent_qindex).isValid()
        assert not manager.tree_view._model.index(4, 0, parent_qindex).isValid()

        valid_but_wrong_pointer_type = manager.tree_view._model.createIndex(
            parent_qindex.row(), parent_qindex.column(), "invalid data"
        )
        assert not manager.tree_view._model.index(
            1, 0, valid_but_wrong_pointer_type
        ).isValid()

        # Close dtools
        logger.info("Closing dtools")
        for uid in list(manager._imagetool_wrappers[3]._childtools.keys()):
            manager._remove_childtool(uid)

        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[3]._childtools) == 0, timeout=5000
        )
        logger.info("Confirmed dtools are removed")

        # Bring manager to top
        logger.info("Testing mouse events")
        with qtbot.waitExposed(manager):
            manager.hide_all()  # Prevent windows from obstructing the manager
            manager.activateWindow()
            manager.raise_()
            manager.preview_action.setChecked(True)

        # Test mouse hover over list view
        # This may not work on all systems due to the way the mouse events are generated
        delegate._force_hover = True

        first_index = manager.tree_view.model().index(0, 0)
        first_rect_center = manager.tree_view.visualRect(first_index).center()
        qtbot.mouseMove(manager.tree_view.viewport())
        qtbot.mouseMove(manager.tree_view.viewport(), first_rect_center)
        qtbot.mouseMove(
            manager.tree_view.viewport(), first_rect_center - QtCore.QPoint(10, 10)
        )
        qtbot.mouseMove(manager.tree_view.viewport())  # move to blank should hide popup

        # Remove third tool
        select_tools(manager, [3])
        accept_dialog(manager.remove_action.trigger)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        # Test concatenate
        concat_data = xr.concat(
            [
                manager.get_imagetool(1).slicer_area._data,
                manager.get_imagetool(2).slicer_area._data,
            ],
            "concat_dim",
        )
        select_tools(manager, [1, 2])
        accept_dialog(manager.concat_action.trigger)
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)
        xr.testing.assert_identical(
            manager.get_imagetool(3).slicer_area._data, concat_data
        )

        # Test concatenate (remove originals)
        select_tools(manager, [1, 2])

        def _handle_concat(dialog: _ConcatDialog):
            dialog._remove_original_check.setChecked(True)

        accept_dialog(manager.concat_action.trigger, pre_call=_handle_concat)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        xr.testing.assert_identical(
            manager.get_imagetool(4).slicer_area._data, concat_data
        )

        # Remove all selected
        select_tools(manager, [3, 4])
        accept_dialog(manager.remove_action.trigger)
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        # Show about dialog
        accept_dialog(manager.about)


def test_manager_archived_cache_cleanup(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        test_data.qshow(manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        wrapper = manager._imagetool_wrappers[0]

        wrapper.archive()
        qtbot.wait_until(lambda: wrapper.archived, timeout=5000)
        archived_path = pathlib.Path(typing.cast("str", wrapper._archived_fname))
        assert archived_path.exists()

        wrapper.unarchive()
        qtbot.wait_until(lambda: not wrapper.archived, timeout=5000)
        assert wrapper._archived_fname is None
        assert not archived_path.exists()

        wrapper.archive()
        qtbot.wait_until(lambda: wrapper.archived, timeout=5000)
        archived_path = pathlib.Path(typing.cast("str", wrapper._archived_fname))
        assert archived_path.exists()

        manager.remove_imagetool(0)
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)
        assert not archived_path.exists()


def test_remove_from_window_shortcut(
    qtbot,
    accept_dialog,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        test_data.qshow(manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        tool = manager.get_imagetool(0)

        with qtbot.waitExposed(tool):
            tool.activateWindow()
            tool.raise_()
            tool.setFocus()

        assert tool.remove_act.isVisible()

        accept_dialog(lambda: qtbot.keyClick(tool, QtCore.Qt.Key.Key_Delete))
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)


def test_remove_childtool_delete_shortcut(
    qtbot,
    accept_dialog,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        test_data.qshow(manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_dtool()

        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 1, timeout=5000
        )
        wrapper = manager._imagetool_wrappers[0]
        uid, child = next(iter(wrapper._childtools.items()))

        with qtbot.waitExposed(child):
            child.activateWindow()
            child.raise_()
            child.setFocus()

        accept_dialog(lambda: qtbot.keyClick(child, QtCore.Qt.Key.Key_Delete))
        qtbot.wait_until(lambda: uid not in wrapper._childtools, timeout=5000)


def test_manager_childtool_type_badge_only_for_tool_windows(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        named_data = test_data.rename("source")
        itool(named_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        model = typing.cast("_ImageToolWrapperItemModel", manager.tree_view.model())
        delegate = typing.cast(
            "_ImageToolWrapperItemDelegate", manager.tree_view.itemDelegate()
        )
        parent = manager._imagetool_wrappers[0]
        parent_tool = manager.get_imagetool(0)
        root_index = model._row_index(0)
        assert root_index.data(_TOOL_TYPE_ROLE) is None

        parent_tool.slicer_area.images[0].open_in_new_window()
        qtbot.wait_until(lambda: len(parent._childtool_indices) == 1, timeout=5000)
        image_uid = parent._childtool_indices[0]
        image_node = manager._child_node(image_uid)
        image_index = model._row_index(image_uid)
        assert image_node.is_imagetool
        assert image_index.data(_TOOL_TYPE_ROLE) is None

        option = QtWidgets.QStyleOptionViewItem()
        option.rect = QtCore.QRect(0, 0, 360, 25)
        option.font = manager.tree_view.font()
        option.palette = manager.tree_view.palette()
        assert delegate._compute_tool_type_info(option, image_node) == (
            None,
            None,
            None,
        )

        parent_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(lambda: len(parent._childtool_indices) == 2, timeout=5000)
        tool_uid = next(uid for uid in parent._childtool_indices if uid != image_uid)
        tool_node = manager._child_node(tool_uid)
        tool = manager.get_childtool(tool_uid)
        tool_index = model._row_index(tool_uid)
        assert isinstance(tool, DerivativeTool)
        assert tool_index.data(_TOOL_TYPE_ROLE) == tool.tool_name
        assert tool_index.data(QtCore.Qt.ItemDataRole.DisplayRole) == (
            tool._tool_display_name
        )
        assert tool_index.data(QtCore.Qt.ItemDataRole.EditRole) == (
            tool._tool_display_name
        )

        type_rect, type_text, _ = delegate._compute_tool_type_info(option, tool_node)
        assert type_rect is not None
        assert type_text == tool.tool_name

        tooltip_text = None

        def _show_tooltip(*args, **kwargs) -> None:
            nonlocal tooltip_text
            tooltip_text = args[1]

        monkeypatch.setattr(QtWidgets.QToolTip, "showText", _show_tooltip)
        help_event = QtGui.QHelpEvent(
            QtCore.QEvent.Type.ToolTip,
            type_rect.center(),
            manager.tree_view.viewport().mapToGlobal(type_rect.center()),
        )
        assert delegate.helpEvent(help_event, manager.tree_view, option, tool_index)
        assert_nonempty_tooltip(tooltip_text)

        manager.tree_view.expand(root_index)
        actual_option = QtWidgets.QStyleOptionViewItem()
        delegate.initStyleOption(actual_option, tool_index)
        actual_option.rect = manager.tree_view.visualRect(tool_index)
        actual_type_rect, _, _ = delegate._compute_tool_type_info(
            actual_option, tool_node
        )
        assert actual_type_rect is not None
        show_calls: list[str] = []
        monkeypatch.setattr(
            manager, "show_childtool", lambda uid: show_calls.append(uid)
        )
        click_tree_view_pos(manager.tree_view, actual_type_rect.center())
        assert show_calls == [tool_uid]

        editor = QtWidgets.QLineEdit(manager.tree_view.viewport())
        delegate.updateEditorGeometry(editor, option, tool_index)
        assert editor.geometry().left() > type_rect.right()
        editor.deleteLater()

        pixmap = QtGui.QPixmap(option.rect.size())
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(pixmap)
        delegate.paint(painter, option, tool_index)
        painter.end()

        assert model.setData(
            tool_index,
            "renamed_dtool",
            QtCore.Qt.ItemDataRole.EditRole,
        )
        assert tool._tool_display_name == "renamed_dtool"
        assert tool_index.data(_TOOL_TYPE_ROLE) == tool.tool_name
        assert tool_index.data(QtCore.Qt.ItemDataRole.DisplayRole) == "renamed_dtool"
        assert tool_index.data(QtCore.Qt.ItemDataRole.EditRole) == "renamed_dtool"


def test_manager_multi_data_not_shown(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool([test_data, test_data], manager=True)

        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        assert not manager.get_imagetool(0).isVisible()
        assert not manager.get_imagetool(1).isVisible()


def test_manager_replace(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        # Open a tool with the manager
        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        assert manager.get_imagetool(0).array_slicer.point_value(0) == 12.0

        # Replace data in the tool
        with qtbot.wait_signal(manager._sigDataReplaced):
            itool(test_data**2, manager=True, replace=0)

        assert manager.get_imagetool(0).array_slicer.point_value(0) == 144.0

        # Replacing 1 should create a new tool
        itool(test_data**2, manager=True, replace=1)
        qtbot.wait_until(lambda: manager.ntools == 2)
        assert manager.get_imagetool(1).array_slicer.point_value(0) == 144.0

        # Negative indexing
        with qtbot.wait_signal(manager._sigDataReplaced):
            itool(test_data, manager=True, replace=-1)

        assert manager.get_imagetool(1).array_slicer.point_value(0) == 12.0


def test_manager_childtool_source_updates(
    qtbot,
    accept_dialog,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 1, timeout=5000
        )

        wrapper = manager._imagetool_wrappers[0]
        uid, child = next(iter(wrapper._childtools.items()))
        assert isinstance(child, DerivativeTool)
        assert child.source_spec is not None

        initial = test_data.transpose("eV", "alpha")
        xr.testing.assert_identical(child.tool_data, initial)

        replaced = test_data.copy(deep=True)
        replaced.data = np.asarray(replaced.data) * 3

        with qtbot.wait_signal(manager._sigDataReplaced):
            itool(replaced, manager=True, replace=0)

        qtbot.wait_until(lambda: child.source_state == "stale", timeout=5000)
        xr.testing.assert_identical(child.tool_data, initial)

        badge_rect, badge_text, index = child_status_badge(manager, uid)
        assert badge_text == "Stale"
        delegate = typing.cast(
            "_ImageToolWrapperItemDelegate", manager.tree_view.itemDelegate()
        )
        option = QtWidgets.QStyleOptionViewItem()
        option.rect = manager.tree_view.visualRect(index)
        option.font = manager.tree_view.font()
        tooltip_text = None

        def _show_tooltip(*args, **kwargs) -> None:
            nonlocal tooltip_text
            tooltip_text = args[1]

        monkeypatch.setattr(QtWidgets.QToolTip, "showText", _show_tooltip)
        help_event = QtGui.QHelpEvent(
            QtCore.QEvent.Type.ToolTip,
            badge_rect.center(),
            manager.tree_view.viewport().mapToGlobal(badge_rect.center()),
        )
        assert delegate.helpEvent(help_event, manager.tree_view, option, index)
        assert_nonempty_tooltip(tooltip_text)
        assert index == manager.tree_view.indexAt(badge_rect.center())

        def _enable_auto_update(dialog: QtWidgets.QDialog) -> None:
            dialog.auto_update_check.setChecked(True)  # type: ignore[attr-defined]

        def _update_now(dialog: QtWidgets.QDialog) -> None:
            dialog.update_button.click()  # type: ignore[attr-defined]

        refresh_calls: list[str] = []
        original_refresh_chain = manager._refresh_source_chain_to_uid

        def _track_refresh_chain(refresh_uid: str) -> bool:
            refresh_calls.append(refresh_uid)
            return original_refresh_chain(refresh_uid)

        monkeypatch.setattr(
            manager, "_refresh_source_chain_to_uid", _track_refresh_chain
        )

        click_child_status_badge(
            manager,
            uid,
            accept_dialog,
            pre_call=_enable_auto_update,
            accept_call=_update_now,
        )

        assert refresh_calls == [uid]
        assert child.source_state == "fresh"
        assert child.source_auto_update is True
        xr.testing.assert_identical(child.tool_data, replaced.transpose("eV", "alpha"))
        auto_badge_rect, auto_badge_text, _ = child_status_badge(manager, uid)
        assert auto_badge_text == "Auto"
        assert not child._source_status_bar.isHidden()
        assert child.source_status_text == "Automatic Updates Enabled"

        tooltip_text = None
        auto_help_event = QtGui.QHelpEvent(
            QtCore.QEvent.Type.ToolTip,
            auto_badge_rect.center(),
            manager.tree_view.viewport().mapToGlobal(auto_badge_rect.center()),
        )
        assert delegate.helpEvent(auto_help_event, manager.tree_view, option, index)
        assert_nonempty_tooltip(tooltip_text)

        def _disable_auto_update(dialog: QtWidgets.QDialog) -> None:
            dialog.auto_update_check.setChecked(False)  # type: ignore[attr-defined]

        click_child_status_badge(
            manager,
            uid,
            accept_dialog,
            pre_call=_disable_auto_update,
        )

        assert child.source_state == "fresh"
        assert child.source_auto_update is False
        assert child._source_status_bar.isHidden()

        replaced2 = replaced.copy(deep=True)
        replaced2.data = np.asarray(replaced2.data) + 5

        with qtbot.wait_signal(manager._sigDataReplaced):
            itool(replaced2, manager=True, replace=0)

        qtbot.wait_until(lambda: child.source_state == "stale", timeout=5000)
        xr.testing.assert_identical(child.tool_data, replaced.transpose("eV", "alpha"))

        accept_dialog(
            lambda: child._source_status_button.click(), accept_call=_update_now
        )

        assert refresh_calls == [uid, uid]
        assert child.source_state == "fresh"
        xr.testing.assert_identical(child.tool_data, replaced2.transpose("eV", "alpha"))

        child._set_source_state("unavailable")
        qtbot.wait_until(lambda: child.source_state == "unavailable", timeout=5000)
        unavailable_badge_rect, unavailable_badge_text, _ = child_status_badge(
            manager, uid
        )
        assert isinstance(unavailable_badge_text, str)
        assert unavailable_badge_text.strip()

        tooltip_text = None
        unavailable_help_event = QtGui.QHelpEvent(
            QtCore.QEvent.Type.ToolTip,
            unavailable_badge_rect.center(),
            manager.tree_view.viewport().mapToGlobal(unavailable_badge_rect.center()),
        )
        assert delegate.helpEvent(
            unavailable_help_event, manager.tree_view, option, index
        )
        assert_nonempty_tooltip(tooltip_text)


def test_manager_full_data_childtool_updates_follow_transposed_view(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.transpose_main_image()
        assert parent_tool.slicer_area.data.dims == ("eV", "alpha")

        parent_tool.slicer_area.open_in_meshtool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 1, timeout=5000
        )

        child = next(iter(manager._imagetool_wrappers[0]._childtools.values()))
        xarray.testing.assert_identical(child.tool_data, parent_tool.slicer_area.data)

        replaced = test_data.copy(deep=True)
        replaced.data = np.asarray(replaced.data) * 2

        with qtbot.wait_signal(manager._sigDataReplaced):
            itool(replaced, manager=True, replace=0)

        qtbot.wait_until(lambda: child.source_state == "stale", timeout=5000)
        assert child._update_from_parent_source() is True
        xarray.testing.assert_identical(child.tool_data, parent_tool.slicer_area.data)

        child.set_source_binding(child.source_spec, auto_update=True, state="fresh")
        replaced2 = replaced.copy(deep=True)
        replaced2.data = np.asarray(replaced2.data) + 5

        with qtbot.wait_signal(manager._sigDataReplaced):
            itool(replaced2, manager=True, replace=0)

        qtbot.wait_until(lambda: child.source_state == "fresh", timeout=5000)
        xarray.testing.assert_identical(child.tool_data, parent_tool.slicer_area.data)


def test_manager_goldtool_output_itool_nests_under_tool(
    qtbot,
    monkeypatch,
    gold,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(gold, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child_uid = manager.add_childtool(
            GoldTool(gold.copy(deep=True), data_name="gold_input"),
            0,
            show=False,
        )
        child = manager.get_childtool(child_uid)
        assert isinstance(child, GoldTool)
        configure_goldtool_child(child, fitted=True, spline=False)

        child.open_itool()

        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)

        output_uid = child_node._childtool_indices[0]
        output_node = manager._child_node(output_uid)
        assert manager.ntools == 1
        assert output_node.is_imagetool
        assert output_node.parent_uid == child_uid
        assert output_node.output_id == "goldtool.corrected"
        assert output_node.source_spec is None
        assert output_node.provenance_spec is not None
        xr.testing.assert_identical(fetch(output_uid), child.corrected)
        manager.tree_view.clearSelection()
        select_child_tool(manager, output_uid)
        manager._update_info(uid=output_uid)
        assert metadata_derivation_texts(manager) == [
            "Start from current goldtool input data",
            "Fit and correct current data with the polynomial edge model",
        ]
        copied = copy_full_code_for_uid(monkeypatch, manager, output_uid)
        assert "era.gold.poly(" in copied
        assert "corrected = era.gold.correct_with_edge(" in copied


def test_manager_dtool_output_itool_nests_under_tool(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 1, timeout=5000
        )

        child_uid = manager._imagetool_wrappers[0]._childtool_indices[0]
        child = manager.get_childtool(child_uid)
        assert isinstance(child, DerivativeTool)

        child.open_itool()

        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)

        output_uid = child_node._childtool_indices[0]
        output_node = manager._child_node(output_uid)
        assert manager.ntools == 1
        assert output_node.is_imagetool
        assert output_node.parent_uid == child_uid
        assert output_node.output_id == "dtool.result"
        assert output_node.source_spec is None
        assert output_node.provenance_spec is not None
        xr.testing.assert_identical(fetch(output_uid), child.result.T)
        manager.tree_view.clearSelection()
        select_child_tool(manager, output_uid)
        manager._update_info(uid=output_uid)
        derivation = metadata_derivation_texts(manager)
        assert derivation[0] == "Start from selected parent ImageTool data"
        assert derivation[-2:] == [
            "Compute derivative output",
            "Transpose derivative output for ImageTool display",
        ]
        copied = copy_full_code_for_uid(monkeypatch, manager, output_uid)
        namespace = _exec_generated_code(
            copied,
            {"data": parent_tool.slicer_area.data.copy(deep=True)},
        )
        result = namespace["result"]
        assert isinstance(result, xr.DataArray)
        xr.testing.assert_identical(result, child.result.T)


def test_manager_ktool_output_itool_nests_under_tool(
    qtbot,
    monkeypatch,
    anglemap,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(anglemap, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.open_in_ktool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 1, timeout=5000
        )

        child_uid = manager._imagetool_wrappers[0]._childtool_indices[0]
        child = typing.cast("typing.Any", manager.get_childtool(child_uid))
        child.show_converted()

        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)

        output_uid = child_node._childtool_indices[0]
        output_node = manager._child_node(output_uid)
        assert manager.ntools == 1
        assert output_node.parent_uid == child_uid
        assert output_node.output_id == "ktool.converted_output"
        assert output_node.source_spec is None
        assert output_node.provenance_spec is not None
        xr.testing.assert_identical(fetch(output_uid), child._itool.slicer_area.data)
        copied = copy_full_code_for_uid(monkeypatch, manager, output_uid)
        assert ".kspace.set_normal(" in copied
        assert "_kconv = " in copied

        replaced = anglemap.copy(deep=True)
        replaced.data = np.asarray(replaced.data) + 1.0
        with qtbot.wait_signal(manager._sigDataReplaced):
            itool(replaced, link=False, manager=True, replace=0)

        qtbot.wait_until(lambda: child.source_state == "stale", timeout=5000)
        qtbot.wait_until(lambda: output_node.source_state == "stale", timeout=5000)

        child.set_source_binding(child.source_spec, auto_update=True, state="fresh")
        output_node.set_output_binding(
            typing.cast("str", output_node.output_id),
            provenance_spec=output_node.provenance_spec,
            auto_update=True,
            state="fresh",
        )

        replaced2 = replaced.copy(deep=True)
        replaced2.data = np.asarray(replaced2.data) + 2.0
        with qtbot.wait_signal(manager._sigDataReplaced):
            itool(replaced2, link=False, manager=True, replace=0)

        qtbot.wait_until(lambda: child.source_state == "fresh", timeout=5000)
        qtbot.wait_until(lambda: output_node.source_state == "fresh", timeout=5000)
        xr.testing.assert_identical(fetch(output_uid), child._converted_output())


def test_manager_ktool_output_itool_marks_stale_without_recomputing(
    qtbot,
    monkeypatch,
    anglemap,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(anglemap, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.open_in_ktool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 1, timeout=5000
        )

        child_uid = manager._imagetool_wrappers[0]._childtool_indices[0]
        child = typing.cast("typing.Any", manager.get_childtool(child_uid))
        child.show_converted()

        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)

        output_uid = child_node._childtool_indices[0]
        output_node = manager._child_node(output_uid)
        assert output_node.source_auto_update is False

        before = fetch(output_uid).copy(deep=True)
        wait_ms = int(1000 / child._UPDATE_LIMIT_HZ) + 50
        qtbot.wait(wait_ms)

        call_count = 0
        original_converted_output = child._converted_output

        def _counting_converted_output():
            nonlocal call_count
            call_count += 1
            return original_converted_output()

        monkeypatch.setattr(child, "_converted_output", _counting_converted_output)

        delta_spin = child._offset_spins["delta"]
        delta_spin.setValue(delta_spin.value() + 0.01)

        qtbot.wait_until(lambda: output_node.source_state == "stale", timeout=5000)
        qtbot.wait(wait_ms)

        assert call_count == 0
        xr.testing.assert_identical(fetch(output_uid), before)


def test_manager_reused_output_child_keeps_stale_state(
    qtbot,
    monkeypatch,
    anglemap,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(anglemap, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.open_in_ktool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 1, timeout=5000
        )

        child_uid = manager._imagetool_wrappers[0]._childtool_indices[0]
        child = typing.cast("typing.Any", manager.get_childtool(child_uid))
        child.show_converted()

        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)

        output_uid = child_node._childtool_indices[0]
        output_node = manager._child_node(output_uid)

        child.set_source_binding(child.source_spec, auto_update=False, state="stale")
        output_node.set_output_binding(
            typing.cast("str", output_node.output_id),
            provenance_spec=output_node.provenance_spec,
            auto_update=False,
            state="stale",
        )

        monkeypatch.setattr(
            child, "_prompt_existing_output_imagetool", lambda: "update"
        )
        child.show_converted()

        qtbot.wait_until(lambda: output_node.source_state == "stale", timeout=5000)
        xr.testing.assert_identical(fetch(output_uid), child._converted_output())


def test_manager_dtool_output_itool_refreshes_with_parent_updates(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 1, timeout=5000
        )

        child_uid = manager._imagetool_wrappers[0]._childtool_indices[0]
        child = manager.get_childtool(child_uid)
        assert isinstance(child, DerivativeTool)

        child.open_itool()

        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)

        output_uid = child_node._childtool_indices[0]
        output_node = manager._child_node(output_uid)

        replaced = test_data.copy(deep=True)
        replaced.data = np.asarray(replaced.data) * 2
        with qtbot.wait_signal(manager._sigDataReplaced):
            itool(replaced, link=False, manager=True, replace=0)

        qtbot.wait_until(lambda: child.source_state == "stale", timeout=5000)
        qtbot.wait_until(lambda: output_node.source_state == "stale", timeout=5000)

        child.set_source_binding(child.source_spec, auto_update=True, state="fresh")
        output_node.set_output_binding(
            typing.cast("str", output_node.output_id),
            provenance_spec=output_node.provenance_spec,
            auto_update=True,
            state="fresh",
        )

        replaced2 = replaced.copy(deep=True)
        replaced2.data = np.asarray(replaced2.data) + 5.0
        with qtbot.wait_signal(manager._sigDataReplaced):
            itool(replaced2, link=False, manager=True, replace=0)

        qtbot.wait_until(lambda: child.source_state == "fresh", timeout=5000)
        qtbot.wait_until(lambda: output_node.source_state == "fresh", timeout=5000)
        xr.testing.assert_identical(fetch(output_uid), child.result.T)


def test_manager_output_itool_auto_update_can_be_disabled_from_auto_badge(
    qtbot,
    accept_dialog,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 1, timeout=5000
        )

        child_uid = manager._imagetool_wrappers[0]._childtool_indices[0]
        child = manager.get_childtool(child_uid)
        assert isinstance(child, DerivativeTool)
        child.open_itool()

        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)
        output_uid = child_node._childtool_indices[0]
        output_node = manager._child_node(output_uid)

        child.set_source_binding(child.source_spec, auto_update=True, state="fresh")

        replaced = test_data.copy(deep=True)
        replaced.data = np.asarray(replaced.data) * 2
        with qtbot.wait_signal(manager._sigDataReplaced):
            itool(replaced, link=False, manager=True, replace=0)

        qtbot.wait_until(lambda: child.source_state == "fresh", timeout=5000)
        qtbot.wait_until(lambda: output_node.source_state == "stale", timeout=5000)

        def _enable_auto_update(dialog: QtWidgets.QDialog) -> None:
            dialog.auto_update_check.setChecked(True)  # type: ignore[attr-defined]

        def _update_now(dialog: QtWidgets.QDialog) -> None:
            dialog.update_button.click()  # type: ignore[attr-defined]

        click_child_status_badge(
            manager,
            output_uid,
            accept_dialog,
            pre_call=_enable_auto_update,
            accept_call=_update_now,
        )

        qtbot.wait_until(lambda: output_node.source_state == "fresh", timeout=5000)
        assert output_node.source_auto_update is True
        xr.testing.assert_identical(fetch(output_uid), child.result.T)
        refreshed_output = fetch(output_uid).copy(deep=True)
        _, badge_text, _ = child_status_badge(manager, output_uid)
        assert badge_text == "Auto"

        def _disable_auto_update(dialog: QtWidgets.QDialog) -> None:
            dialog.auto_update_check.setChecked(False)  # type: ignore[attr-defined]

        click_child_status_badge(
            manager,
            output_uid,
            accept_dialog,
            pre_call=_disable_auto_update,
        )
        assert output_node.source_auto_update is False

        replaced2 = replaced.copy(deep=True)
        replaced2.data = np.asarray(replaced2.data) + 5.0
        with qtbot.wait_signal(manager._sigDataReplaced):
            itool(replaced2, link=False, manager=True, replace=0)

        qtbot.wait_until(lambda: child.source_state == "fresh", timeout=5000)
        qtbot.wait_until(lambda: output_node.source_state == "stale", timeout=5000)
        xr.testing.assert_identical(fetch(output_uid), refreshed_output)


def test_load_code_from_file_details_uses_erlab_io_loader_syntax(
    tmp_path: pathlib.Path,
    example_loader,
) -> None:
    file_path = tmp_path / "example.pxt"
    code = _load_code_from_file_details(
        file_path,
        ("merlin", {"bad-key": 1, "single": True}, 0),
    )

    expected = (
        "import erlab\n\n"
        "erlab.io.set_loader('merlin')\n"
        f"data = erlab.io.load({str(file_path)!r}, "
        '**{"bad-key": 1, "single": True})'
    )
    assert code == expected
    assigned_code = _load_code_from_file_details(
        file_path,
        ("merlin", {"bad-key": 1, "single": True}, 0),
        assign="source_scan",
    )
    assert assigned_code == expected.replace("data = ", "source_scan = ")
    with pytest.raises(ValueError, match="assign"):
        _load_code_from_file_details(file_path, ("merlin", {}, 0), assign="bad-name")

    extension_code = _load_code_from_file_details(
        file_path,
        (
            "example",
            {"loader_extensions": {"additional_coords": {"gui_extra": 7.0}}},
            0,
        ),
    )
    assert extension_code == (
        "import erlab\n\n"
        "erlab.io.set_loader('example')\n"
        f"data = erlab.io.load({str(file_path)!r}, "
        'loader_extensions={"additional_coords": {"gui_extra": 7.0}})'
    )


def test_load_code_from_file_details_prefers_scan_number_for_erlab_loader(
    example_loader,
    example_data_dir: pathlib.Path,
) -> None:
    file_path = example_data_dir / "data_002.h5"
    code = _load_code_from_file_details(file_path, ("example", {}, 0))
    assert code == (
        "import erlab\n\n"
        "erlab.io.set_loader('example')\n"
        f"data = erlab.io.load(2, data_dir={str(example_data_dir)!r})"
    )

    multi_file_code = _load_code_from_file_details(
        example_data_dir / "data_001_S001.h5", ("example", {}, 0)
    )
    assert multi_file_code == (
        "import erlab\n\n"
        "erlab.io.set_loader('example')\n"
        f"data = erlab.io.load(1, data_dir={str(example_data_dir)!r})"
    )

    single_file_code = _load_code_from_file_details(
        file_path, ("example", {"single": True}, 0)
    )
    assert single_file_code == (
        "import erlab\n\n"
        "erlab.io.set_loader('example')\n"
        f"data = erlab.io.load({str(file_path)!r}, single=True)"
    )

    extension_code = _load_code_from_file_details(
        file_path,
        (
            "example",
            {"loader_extensions": {"additional_coords": {"gui_extra": 7.0}}},
            0,
        ),
    )
    assert extension_code == (
        "import erlab\n\n"
        "erlab.io.set_loader('example')\n"
        f"data = erlab.io.load(2, data_dir={str(example_data_dir)!r}, "
        'loader_extensions={"additional_coords": {"gui_extra": 7.0}})'
    )

    del example_loader
    bound_loader_code = _load_code_from_file_details(
        file_path, (erlab.io.loaders["example"].load, {}, 0)
    )
    assert bound_loader_code == code


def test_scan_number_load_call_args_rejects_ambiguous_loader_matches(
    monkeypatch,
    tmp_path: pathlib.Path,
) -> None:
    file_path = tmp_path / "scan_007.h5"
    file_path.touch()
    assert _resolve_identified_path("scan_007.h5", tmp_path) == file_path.resolve()
    assert _scan_number_load_call_args(file_path, "missing_loader", {}) is None
    assert _scan_number_load_call_args(file_path, "coverage_loader", {1: "bad"}) is None

    class _FakeLoader:
        infer_result: typing.Any = (7, {})
        identify_result: typing.Any = ([str(file_path)],)
        infer_error = False
        identify_error = False

        def infer_index(self, stem: str) -> typing.Any:
            assert stem == file_path.stem
            if self.infer_error:
                raise RuntimeError("infer failed")
            return self.infer_result

        def identify(
            self, scan_num: int, data_dir: pathlib.Path, **kwargs
        ) -> typing.Any:
            assert scan_num == 7
            assert data_dir == tmp_path
            if self.identify_error:
                raise RuntimeError("identify failed")
            return self.identify_result

    loader = _FakeLoader()
    monkeypatch.setattr(erlab.io, "loaders", {"coverage_loader": loader})

    loader.infer_error = True
    assert _scan_number_load_call_args(file_path, "coverage_loader", {}) is None

    loader.infer_error = False
    loader.infer_result = (7, None)
    assert _scan_number_load_call_args(file_path, "coverage_loader", {}) == [
        "7",
        f"data_dir={str(tmp_path)!r}",
    ]

    loader.infer_result = (7, ["not", "mapping"])
    assert _scan_number_load_call_args(file_path, "coverage_loader", {}) is None

    loader.infer_result = (7, {"data_dir": tmp_path})
    assert _scan_number_load_call_args(file_path, "coverage_loader", {}) is None

    loader.infer_result = (7, {})
    loader.identify_error = True
    assert _scan_number_load_call_args(file_path, "coverage_loader", {}) is None

    loader.identify_error = False
    loader.identify_result = None
    assert _scan_number_load_call_args(file_path, "coverage_loader", {}) is None

    loader.identify_result = ([str(tmp_path / "other.h5")],)
    assert _scan_number_load_call_args(file_path, "coverage_loader", {}) is None


def test_loader_extension_literal_parser() -> None:
    assert _text_to_loader_extension_value("coordinate_attrs", "['theta', 'phi']") == {
        "coordinate_attrs": ["theta", "phi"]
    }
    assert _text_to_loader_extension_value("additional_coords", "{'scan': 1}") == {
        "additional_coords": {"scan": 1}
    }
    assert _text_to_loader_extension_value(
        "name_map", "{'theta': ['Theta', 'Angle']}"
    ) == {"name_map": {"theta": ["Theta", "Angle"]}}

    with pytest.raises(ValueError, match="not a valid literal"):
        _text_to_loader_extension_value("additional_coords", "dict(scan=1)")


def test_name_filter_dialog_loader_extensions_toggle_resizes(
    qtbot, example_loader
) -> None:
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    dialog = _NameFilterDialog(
        parent,
        {"Example Raw Data (*.h5)": (erlab.io.loaders["example"].load, {})},
    )
    qtbot.addWidget(dialog)
    dialog.check_filter("Example Raw Data (*.h5)")
    dialog.show()
    QtWidgets.QApplication.processEvents()

    collapsed_height = dialog.height()
    assert dialog.extensions_toggle.isVisible()
    assert "extend_loader" in dialog.extensions_toggle.toolTip()
    assert "<tt>" in dialog.extensions_toggle.toolTip()
    assert not dialog.extensions_group.isVisible()

    dialog.extensions_toggle.setChecked(True)
    QtWidgets.QApplication.processEvents()
    expanded_height = dialog.height()
    assert expanded_height > collapsed_height
    extensions_layout = typing.cast(
        "QtWidgets.QFormLayout", dialog.extensions_group.layout()
    )
    for field in dialog.loader_extension_fields.values():
        label = extensions_layout.labelForField(field)
        assert label is not None
        assert field.toolTip()
        assert "<tt>" in field.toolTip()
        assert label.toolTip() == field.toolTip()
    assert dialog.name_map_editor_button is not None
    assert (
        dialog.loader_extension_lines["name_map"].sizePolicy().horizontalPolicy()
        == QtWidgets.QSizePolicy.Policy.Ignored
    )
    assert dialog.coordinate_attrs_picker_button is not None
    assert (
        dialog.loader_extension_lines["coordinate_attrs"]
        .sizePolicy()
        .horizontalPolicy()
        == QtWidgets.QSizePolicy.Policy.Ignored
    )

    dialog.extensions_toggle.setChecked(False)
    QtWidgets.QApplication.processEvents()
    assert not dialog.extensions_group.isVisible()
    assert dialog.height() < expanded_height

    dialog.loader_extension_lines["additional_coords"].setText("{'scan': 1}")
    filter_name, _func, kwargs = dialog.checked_filter()
    assert filter_name == "Example Raw Data (*.h5)"
    assert kwargs["loader_extensions"] == {"additional_coords": {"scan": 1}}


def _tree_item_by_text(
    tree: QtWidgets.QTreeWidget, column: int, text: str
) -> QtWidgets.QTreeWidgetItem:
    for i in range(tree.topLevelItemCount()):
        item = tree.topLevelItem(i)
        if item is not None and item.text(column) == text:
            return item
    raise AssertionError(f"Could not find tree item {text!r}")


def _table_row_by_text(table: QtWidgets.QTableWidget, column: int, text: str) -> int:
    for row in range(table.rowCount()):
        item = table.item(row, column)
        if item is not None and item.text() == text:
            return row
    raise AssertionError(f"Could not find table row {text!r}")


def _dialog_label_text(dialog: QtWidgets.QDialog) -> str:
    return "\n".join(label.text() for label in dialog.findChildren(QtWidgets.QLabel))


def test_loader_extension_literal_helpers_handle_edge_cases() -> None:
    assert manager_dialogs._string_tuple_from_literal("coordinate_attrs", "") == ()
    with pytest.raises(TypeError, match="not a string"):
        manager_dialogs._string_tuple_from_literal("coordinate_attrs", "'theta'")
    with pytest.raises(TypeError, match=r"coordinate_attrs must be an iterable$"):
        manager_dialogs._string_tuple_from_literal("coordinate_attrs", "1")

    assert manager_dialogs._name_map_from_literal("") == {}
    with pytest.raises(TypeError, match="name_map must be a dict"):
        manager_dialogs._name_map_from_literal("['theta']")

    assert list(
        manager_dialogs._iter_name_map_pairs({"theta": ["Theta", "Angle"]})
    ) == [("theta", "Theta"), ("theta", "Angle")]
    assert manager_dialogs._name_map_from_pairs(
        [("theta", "Theta"), ("theta", "Theta"), ("theta", "Angle")]
    ) == {"theta": ["Theta", "Angle"]}
    assert manager_dialogs._name_map_literal({}) == ""
    assert manager_dialogs._coordinate_attrs_literal(()) == ""


def test_coordinate_attrs_sample_attrs_handles_loader_variants() -> None:
    class _PlainLoader:
        def __init__(self) -> None:
            self.paths: list[pathlib.Path] = []

        def load_single(self, path: pathlib.Path) -> types.SimpleNamespace:
            self.paths.append(path)
            return types.SimpleNamespace(attrs={"LensMode": "Angular"})

    plain_loader = _PlainLoader()
    plain_path = pathlib.Path("plain.h5")
    assert manager_dialogs._coordinate_attrs_sample_attrs(
        typing.cast("erlab.io.dataloader.LoaderBase", plain_loader),
        plain_path,
    ) == {"LensMode": "Angular"}
    assert plain_loader.paths == [plain_path]

    class _RetryLoader:
        def __init__(self) -> None:
            self.calls: list[bool] = []

        def load_single(
            self, _path: pathlib.Path, *, without_values: bool = False
        ) -> object:
            self.calls.append(without_values)
            if without_values:
                raise TypeError("without_values is unavailable")
            return object()

    retry_loader = _RetryLoader()
    assert (
        manager_dialogs._coordinate_attrs_sample_attrs(
            typing.cast("erlab.io.dataloader.LoaderBase", retry_loader),
            pathlib.Path("retry.h5"),
        )
        == {}
    )
    assert retry_loader.calls == [True, False]

    class _FailingLoader:
        def load_single(self, _path: pathlib.Path) -> object:
            raise TypeError("load failed")

    with pytest.raises(TypeError, match="load failed"):
        manager_dialogs._coordinate_attrs_sample_attrs(
            typing.cast("erlab.io.dataloader.LoaderBase", _FailingLoader()),
            pathlib.Path("failed.h5"),
        )


def test_name_map_editor_emits_custom_mapping_and_omits_blank_rows(
    qtbot,
    example_loader,
    example_data_dir: pathlib.Path,
) -> None:
    editor = _NameMapEditorDialog(
        None,
        erlab.io.loaders["example"],
        example_data_dir / "data_002.h5",
        {},
    )
    qtbot.addWidget(editor)

    table = editor.findChild(QtWidgets.QTableWidget)
    assert table is not None

    lens_mode_row = _table_row_by_text(table, 0, "LensMode")
    lens_mode_target = table.item(lens_mode_row, 1)
    assert lens_mode_target is not None
    assert lens_mode_target.text() == ""
    assert lens_mode_target.flags() & QtCore.Qt.ItemFlag.ItemIsEditable
    lens_mode_target.setText("lens_mode")

    temp_row = _table_row_by_text(table, 0, "TB")
    temp_target = table.item(temp_row, 1)
    assert temp_target is not None
    assert temp_target.text() == "sample_temp"
    assert not (temp_target.flags() & QtCore.Qt.ItemFlag.ItemIsEnabled)
    assert not (temp_target.flags() & QtCore.Qt.ItemFlag.ItemIsEditable)

    editor.accept()
    assert editor.selected_name_map() == {"lens_mode": "LensMode"}


def test_name_map_editor_prefills_and_preserves_unmatched_mappings(
    qtbot,
    example_loader,
    example_data_dir: pathlib.Path,
) -> None:
    editor = _NameMapEditorDialog(
        None,
        erlab.io.loaders["example"],
        example_data_dir / "data_002.h5",
        {"lens_mode": "LensMode", "legacy": "MissingRaw"},
    )
    qtbot.addWidget(editor)

    table = editor.findChild(QtWidgets.QTableWidget)
    assert table is not None
    lens_mode_row = _table_row_by_text(table, 0, "LensMode")
    lens_mode_target = table.item(lens_mode_row, 1)
    assert lens_mode_target is not None
    assert lens_mode_target.text() == "lens_mode"

    editor.accept()
    assert editor.selected_name_map() == {
        "legacy": "MissingRaw",
        "lens_mode": "LensMode",
    }


def test_name_map_editor_disabled_sample_states(
    qtbot,
    monkeypatch,
    example_loader,
) -> None:
    no_sample = _NameMapEditorDialog(
        None,
        erlab.io.loaders["example"],
        None,
        {"legacy": "MissingRaw"},
    )
    qtbot.addWidget(no_sample)
    assert no_sample.findChild(QtWidgets.QTableWidget) is None
    assert "No sample file is available." in _dialog_label_text(no_sample)
    assert no_sample.selected_name_map() == {"legacy": "MissingRaw"}

    def _raise_sample_attrs(*_args: object) -> dict[str, typing.Any]:
        raise RuntimeError("sample failed")

    monkeypatch.setattr(
        manager_dialogs,
        "_coordinate_attrs_sample_attrs",
        _raise_sample_attrs,
    )
    failed_sample = _NameMapEditorDialog(
        None,
        erlab.io.loaders["example"],
        pathlib.Path("bad.h5"),
        {"legacy": "MissingRaw"},
    )
    qtbot.addWidget(failed_sample)
    assert failed_sample.findChild(QtWidgets.QTableWidget) is None
    failed_text = _dialog_label_text(failed_sample)
    assert "Could not inspect the selected file." in failed_text
    assert "RuntimeError: sample failed" in failed_text

    monkeypatch.setattr(
        manager_dialogs,
        "_coordinate_attrs_sample_attrs",
        lambda *_args: {},
    )
    empty_sample = _NameMapEditorDialog(
        None,
        erlab.io.loaders["example"],
        pathlib.Path("empty.h5"),
        {"legacy": "MissingRaw"},
    )
    qtbot.addWidget(empty_sample)
    assert empty_sample.findChild(QtWidgets.QTableWidget) is None
    assert "No attributes were found in the sample." in _dialog_label_text(empty_sample)


def test_name_filter_dialog_name_map_editor_updates_literal(
    qtbot,
    monkeypatch,
    example_loader,
    example_data_dir: pathlib.Path,
) -> None:
    file_path = example_data_dir / "data_002.h5"
    editor_calls: list[tuple[typing.Any, ...]] = []

    class _FakeNameMapEditorDialog:
        def __init__(
            self,
            parent,
            loader,
            sample_path,
            current_name_map,
        ) -> None:
            editor_calls.append((parent, loader, sample_path, current_name_map))

        def exec(self) -> bool:
            return True

        def selected_name_map(self) -> dict[str, str]:
            return {"lens_mode": "LensMode"}

    monkeypatch.setattr(
        manager_dialogs,
        "_NameMapEditorDialog",
        _FakeNameMapEditorDialog,
    )
    dialog = _NameFilterDialog(
        None,
        {"Example Raw Data (*.h5)": (erlab.io.loaders["example"].load, {})},
        sample_paths=[file_path],
    )
    qtbot.addWidget(dialog)
    dialog.check_filter("Example Raw Data (*.h5)")
    dialog.loader_extension_lines["name_map"].setText("{'old_name': 'Old Raw'}")

    dialog._open_name_map_editor()

    assert editor_calls == [
        (
            dialog,
            erlab.io.loaders["example"],
            file_path,
            {"old_name": "Old Raw"},
        )
    ]
    assert dialog.loader_extension_lines["name_map"].text() == (
        "{'lens_mode': 'LensMode'}"
    )


def test_name_filter_dialog_invalid_name_map_editor_literal_shows_error(
    qtbot,
    monkeypatch,
    example_loader,
) -> None:
    critical_calls: list[tuple[typing.Any, ...]] = []
    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        staticmethod(lambda *args: critical_calls.append(args) or 0),
    )
    dialog = _NameFilterDialog(
        None,
        {"Example Raw Data (*.h5)": (erlab.io.loaders["example"].load, {})},
    )
    qtbot.addWidget(dialog)
    dialog.check_filter("Example Raw Data (*.h5)")
    dialog.loader_extension_lines["name_map"].setText("dict(scan=1)")

    dialog._open_name_map_editor()

    assert critical_calls
    assert critical_calls[0][1:4] == (
        "Error",
        "Invalid loader arguments.",
        "Value for 'name_map' is not a valid literal",
    )


def test_name_filter_dialog_editor_cancel_leaves_literals(
    qtbot,
    monkeypatch,
    example_loader,
) -> None:
    class _CancelNameMapEditorDialog:
        def __init__(self, *_args: object) -> None:
            pass

        def exec(self) -> bool:
            return False

        def selected_name_map(self) -> dict[str, str]:
            raise AssertionError("selected_name_map must not be called")

    class _CancelCoordinateAttrsPickerDialog:
        def __init__(self, *_args: object) -> None:
            pass

        def exec(self) -> bool:
            return False

        def selected_coordinate_attrs(self) -> tuple[str, ...]:
            raise AssertionError("selected_coordinate_attrs must not be called")

    monkeypatch.setattr(
        manager_dialogs,
        "_NameMapEditorDialog",
        _CancelNameMapEditorDialog,
    )
    monkeypatch.setattr(
        manager_dialogs,
        "_CoordinateAttrsPickerDialog",
        _CancelCoordinateAttrsPickerDialog,
    )
    dialog = _NameFilterDialog(
        None,
        {"Example Raw Data (*.h5)": (erlab.io.loaders["example"].load, {})},
        sample_paths=[pathlib.Path("sample.h5")],
    )
    qtbot.addWidget(dialog)
    dialog.check_filter("Example Raw Data (*.h5)")
    dialog.loader_extension_lines["name_map"].setText("{'old_name': 'Old Raw'}")
    dialog.loader_extension_lines["coordinate_attrs"].setText("['old_coord']")

    dialog._open_name_map_editor()
    dialog._open_coordinate_attrs_picker()

    assert dialog.loader_extension_lines["name_map"].text() == "{'old_name': 'Old Raw'}"
    assert dialog.loader_extension_lines["coordinate_attrs"].text() == "['old_coord']"


def test_name_filter_dialog_editor_helpers_ignore_non_loader_functions(qtbot) -> None:
    def non_loader(*_args: object, **_kwargs: object) -> None:
        return None

    dialog = _NameFilterDialog(
        None,
        {"Plain Files (*.txt)": (non_loader, {})},
        sample_paths=[pathlib.Path("plain.txt")],
    )
    qtbot.addWidget(dialog)
    dialog.check_filter("Plain Files (*.txt)")
    dialog.loader_extension_lines["name_map"].setText("{'old_name': 'Old Raw'}")
    dialog.loader_extension_lines["coordinate_attrs"].setText("['old_coord']")

    dialog._open_name_map_editor()
    dialog._open_coordinate_attrs_picker()

    assert dialog.loader_extension_lines["name_map"].text() == "{'old_name': 'Old Raw'}"
    assert dialog.loader_extension_lines["coordinate_attrs"].text() == "['old_coord']"


def test_name_filter_dialog_invalid_coordinate_attrs_picker_literal_shows_error(
    qtbot,
    monkeypatch,
    example_loader,
) -> None:
    critical_calls: list[tuple[typing.Any, ...]] = []
    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        staticmethod(lambda *args: critical_calls.append(args) or 0),
    )
    dialog = _NameFilterDialog(
        None,
        {"Example Raw Data (*.h5)": (erlab.io.loaders["example"].load, {})},
    )
    qtbot.addWidget(dialog)
    dialog.check_filter("Example Raw Data (*.h5)")
    dialog.loader_extension_lines["coordinate_attrs"].setText("'LensMode'")

    dialog._open_coordinate_attrs_picker()

    assert critical_calls
    assert critical_calls[0][1:4] == (
        "Error",
        "Invalid loader arguments.",
        "coordinate_attrs must be an iterable, not a string",
    )


def test_coordinate_attrs_picker_shows_mapped_and_builtin_attrs(
    qtbot,
    example_loader,
    example_data_dir: pathlib.Path,
) -> None:
    picker = _CoordinateAttrsPickerDialog(
        None,
        erlab.io.loaders["example"],
        example_data_dir / "data_002.h5",
        ("LensMode",),
        {},
    )
    qtbot.addWidget(picker)

    tree = picker.findChild(QtWidgets.QTreeWidget)
    assert tree is not None

    lens_mode_item = _tree_item_by_text(tree, 0, "LensMode")
    assert lens_mode_item.text(1) == ""
    assert lens_mode_item.checkState(0) == QtCore.Qt.CheckState.Checked
    assert lens_mode_item.flags() & QtCore.Qt.ItemFlag.ItemIsUserCheckable

    temp_item = _tree_item_by_text(tree, 0, "TB")
    assert temp_item.text(1) == "sample_temp"
    assert temp_item.checkState(0) == QtCore.Qt.CheckState.Checked
    assert not (temp_item.flags() & QtCore.Qt.ItemFlag.ItemIsEnabled)
    assert not (temp_item.flags() & QtCore.Qt.ItemFlag.ItemIsUserCheckable)

    picker.accept()
    assert picker.selected_coordinate_attrs() == ("LensMode",)


def test_coordinate_attrs_picker_uses_literal_name_map(
    qtbot,
    example_loader,
    example_data_dir: pathlib.Path,
) -> None:
    picker = _CoordinateAttrsPickerDialog(
        None,
        erlab.io.loaders["example"],
        example_data_dir / "data_002.h5",
        (),
        {"lens_mode": "LensMode"},
    )
    qtbot.addWidget(picker)

    tree = picker.findChild(QtWidgets.QTreeWidget)
    assert tree is not None
    lens_mode_item = _tree_item_by_text(tree, 0, "LensMode")
    assert lens_mode_item.text(1) == "lens_mode"

    lens_mode_item.setCheckState(0, QtCore.Qt.CheckState.Checked)
    picker.accept()
    assert picker.selected_coordinate_attrs() == ("lens_mode",)


def test_name_filter_dialog_coordinate_attrs_picker_updates_literal(
    qtbot,
    monkeypatch,
    example_loader,
    example_data_dir: pathlib.Path,
) -> None:
    file_path = example_data_dir / "data_002.h5"
    picker_calls: list[tuple[typing.Any, ...]] = []

    class _FakeCoordinateAttrsPickerDialog:
        def __init__(
            self,
            parent,
            loader,
            sample_path,
            current_values,
            name_map,
        ) -> None:
            picker_calls.append((parent, loader, sample_path, current_values, name_map))

        def exec(self) -> bool:
            return True

        def selected_coordinate_attrs(self) -> tuple[str, ...]:
            return ("LensMode", "extra_coord")

    monkeypatch.setattr(
        manager_dialogs,
        "_CoordinateAttrsPickerDialog",
        _FakeCoordinateAttrsPickerDialog,
    )
    dialog = _NameFilterDialog(
        None,
        {"Example Raw Data (*.h5)": (erlab.io.loaders["example"].load, {})},
        sample_paths=[file_path],
    )
    qtbot.addWidget(dialog)
    dialog.check_filter("Example Raw Data (*.h5)")
    dialog.loader_extension_lines["coordinate_attrs"].setText("['old_coord']")
    dialog.loader_extension_lines["name_map"].setText("{'extra_coord': 'Extra Raw'}")

    dialog._open_coordinate_attrs_picker()

    assert picker_calls == [
        (
            dialog,
            erlab.io.loaders["example"],
            file_path,
            ("old_coord",),
            {"extra_coord": "Extra Raw"},
        )
    ]
    assert (
        dialog.loader_extension_lines["coordinate_attrs"].text()
        == "['LensMode', 'extra_coord']"
    )


def test_coordinate_attrs_picker_failure_leaves_literal_editor(
    qtbot,
    example_loader,
) -> None:
    dialog = _NameFilterDialog(
        None,
        {"Example Raw Data (*.h5)": (erlab.io.loaders["example"].load, {})},
        sample_paths=[pathlib.Path("missing.h5")],
    )
    qtbot.addWidget(dialog)
    dialog.check_filter("Example Raw Data (*.h5)")
    dialog.loader_extension_lines["coordinate_attrs"].setText("['LensMode']")

    picker = _CoordinateAttrsPickerDialog(
        dialog,
        erlab.io.loaders["example"],
        pathlib.Path("missing.h5"),
        ("LensMode",),
        {},
    )
    qtbot.addWidget(picker)

    assert picker.findChild(QtWidgets.QTreeWidget) is None
    assert picker.selected_coordinate_attrs() == ("LensMode",)
    assert dialog.loader_extension_lines["coordinate_attrs"].text() == "['LensMode']"


def test_coordinate_attrs_picker_disabled_sample_states(
    qtbot,
    monkeypatch,
    example_loader,
) -> None:
    no_sample = _CoordinateAttrsPickerDialog(
        None,
        erlab.io.loaders["example"],
        None,
        ("legacy_coord",),
        {},
    )
    qtbot.addWidget(no_sample)
    assert no_sample.findChild(QtWidgets.QTreeWidget) is None
    assert "No sample file is available." in _dialog_label_text(no_sample)
    assert no_sample.selected_coordinate_attrs() == ("legacy_coord",)

    monkeypatch.setattr(
        manager_dialogs,
        "_coordinate_attrs_sample_attrs",
        lambda *_args: {},
    )
    empty_sample = _CoordinateAttrsPickerDialog(
        None,
        erlab.io.loaders["example"],
        pathlib.Path("empty.h5"),
        ("legacy_coord",),
        {},
    )
    qtbot.addWidget(empty_sample)
    assert empty_sample.findChild(QtWidgets.QTreeWidget) is None
    assert "No attributes were found in the sample." in _dialog_label_text(empty_sample)
    assert empty_sample.selected_coordinate_attrs() == ("legacy_coord",)


def test_name_filter_dialog_without_loader_extensions_returns_kwargs(
    qtbot,
    example_loader,
) -> None:
    loader_dialog = _NameFilterDialog(
        None,
        {"Example Raw Data (*.h5)": (erlab.io.loaders["example"].load, {})},
    )
    qtbot.addWidget(loader_dialog)
    loader_dialog.check_filter("Example Raw Data (*.h5)")
    assert loader_dialog.checked_filter() == (
        "Example Raw Data (*.h5)",
        erlab.io.loaders["example"].load,
        {},
    )

    def non_loader(*_args, **_kwargs) -> None:
        return None

    non_loader_dialog = _NameFilterDialog(
        None,
        {"Plain Files (*.txt)": (non_loader, {"plain": True})},
    )
    qtbot.addWidget(non_loader_dialog)
    non_loader_dialog.check_filter("Plain Files (*.txt)")
    assert not non_loader_dialog.extensions_toggle.isVisible()
    assert non_loader_dialog.checked_filter() == (
        "Plain Files (*.txt)",
        non_loader,
        {"plain": True},
    )


def test_name_filter_dialog_invalid_loader_extensions_shows_error(
    qtbot,
    monkeypatch,
    example_loader,
) -> None:
    critical_calls: list[tuple[typing.Any, ...]] = []
    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        staticmethod(lambda *args: critical_calls.append(args) or 0),
    )

    dialog = _NameFilterDialog(
        None,
        {"Example Raw Data (*.h5)": (erlab.io.loaders["example"].load, {})},
    )
    qtbot.addWidget(dialog)
    dialog.check_filter("Example Raw Data (*.h5)")
    dialog.loader_extension_lines["additional_coords"].setText("dict(scan=1)")

    dialog.accept()

    assert critical_calls
    assert critical_calls[0][1:4] == (
        "Error",
        "Invalid loader arguments.",
        "Value for 'additional_coords' is not a valid literal",
    )
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Rejected


def test_wrapper_preview_fallback_branches(monkeypatch) -> None:
    fallback = QtGui.QPixmap(3, 2)
    fallback.fill(QtGui.QColor("red"))
    rendered = QtGui.QPixmap(4, 6)
    rendered.fill(QtGui.QColor("blue"))
    invalid = object()

    class _FakeImageItem:
        def __init__(
            self, pixmap: QtGui.QPixmap | None = None, *, raise_pixmap: bool = False
        ) -> None:
            self.pixmap = pixmap if pixmap is not None else rendered
            self.raise_pixmap = raise_pixmap

        def getPixmap(self) -> QtGui.QPixmap:
            if self.raise_pixmap:
                raise RuntimeError("pixmap unavailable")
            return self.pixmap

    class _FakeViewBox:
        def __init__(self, width: float, height: float) -> None:
            self._rect = QtCore.QRectF(0.0, 0.0, width, height)

        def rect(self) -> QtCore.QRectF:
            return self._rect

    class _FakeMainImage:
        def __init__(
            self,
            *,
            view_box: object = None,
            items: list[object] | None = None,
        ) -> None:
            self._view_box = (
                view_box if view_box is not None else _FakeViewBox(2.0, 8.0)
            )
            self.slicer_data_items = [_FakeImageItem()] if items is None else items

        def getViewBox(self) -> object:
            return self._view_box

    class _FakeSlicerArea:
        def __init__(
            self, main_image: object = None, *, raise_main: bool = False
        ) -> None:
            self._main_image = _FakeMainImage() if main_image is None else main_image
            self._raise_main = raise_main

        def _update_if_delayed(self) -> None:
            return

        @property
        def main_image(self) -> object:
            if self._raise_main:
                raise RuntimeError("main image unavailable")
            return self._main_image

    class _FakeImageTool:
        def __init__(self, slicer_area: _FakeSlicerArea) -> None:
            self.slicer_area = slicer_area

    def _preview(slicer_area: _FakeSlicerArea) -> tuple[float, QtGui.QPixmap]:
        return _preview_from_imagetool(
            typing.cast(
                "erlab.interactive.imagetool.ImageTool",
                _FakeImageTool(slicer_area),
            ),
            1.5,
            fallback,
        )

    monkeypatch.setattr(
        erlab.interactive.utils,
        "qt_is_valid",
        lambda obj, *_args: obj is not invalid,
    )

    assert _preview_from_imagetool(None, 1.5, fallback) == (1.5, fallback)
    assert _preview(_FakeSlicerArea(raise_main=True)) == (1.5, fallback)
    assert _preview(_FakeSlicerArea(invalid)) == (1.5, fallback)
    assert _preview(_FakeSlicerArea(_FakeMainImage(view_box=invalid))) == (
        1.5,
        fallback,
    )
    assert _preview(
        _FakeSlicerArea(_FakeMainImage(view_box=_FakeViewBox(0.0, 2.0)))
    ) == (
        1.5,
        fallback,
    )
    assert _preview(_FakeSlicerArea(_FakeMainImage(items=[]))) == (1.5, fallback)
    assert _preview(_FakeSlicerArea(_FakeMainImage(items=[invalid]))) == (1.5, fallback)
    assert _preview(
        _FakeSlicerArea(_FakeMainImage(items=[_FakeImageItem(raise_pixmap=True)]))
    ) == (1.5, fallback)
    assert _preview(
        _FakeSlicerArea(_FakeMainImage(items=[_FakeImageItem(QtGui.QPixmap())]))
    ) == (1.5, fallback)

    ratio, pixmap = _preview(_FakeSlicerArea(_FakeMainImage()))
    assert ratio == 4.0
    assert not pixmap.isNull()


def test_wrapper_loader_code_and_metadata_helper_branches(
    tmp_path: pathlib.Path,
) -> None:
    import math

    file_path = tmp_path / "data.nc"

    def _local_loader() -> None:
        return None

    def _missing_module_loader() -> None:
        return None

    _missing_module_loader.__module__ = "missing_erlab_loader_module"
    _missing_module_loader.__qualname__ = "load"

    def _missing_attr_loader() -> None:
        return None

    _missing_attr_loader.__module__ = "math"
    _missing_attr_loader.__qualname__ = "missing_loader"

    assert _load_code_from_file_details(file_path, None) is None
    assert _load_provenance_from_file_details(file_path, None) is None
    provenance = _load_provenance_from_file_details(
        file_path,
        (xr.load_dataarray, {"engine": "h5netcdf"}, 0),
    )
    assert provenance is not None
    assert provenance.kind == "file"
    assert provenance.file_load_source is not None
    assert provenance.file_load_source.replay_call is not None
    assert provenance.file_load_source.replay_call.target == "xarray.load_dataarray"
    selected_code = _load_code_from_file_details(file_path, ("example", {}, 1))
    assert selected_code == (
        "import erlab\n\n"
        "erlab.io.set_loader('example')\n"
        "data = erlab.interactive.imagetool.viewer._parse_input("
        f"erlab.io.load({str(file_path)!r}))[1]"
    )
    assert _load_code_from_file_details(file_path, (_local_loader, {}, 0)) is None
    assert _load_provenance_from_file_details(file_path, (_local_loader, {}, 0)) is None
    assert _loader_callable_text(_local_loader) is None
    assert _load_source_label_and_text(None) == ("Loader", "(unavailable)")
    assert _load_source_label_and_text(("example", {}, 0)) == ("Loader", "example")
    assert _load_source_label_and_text((_local_loader, {}, 0)) == (
        "Load Function",
        repr(_local_loader),
    )
    assert _loader_callable_text(_missing_module_loader) == (
        "missing_erlab_loader_module.load"
    )
    assert _loader_callable_text(_missing_attr_loader) == "math.missing_loader"

    math_code = _load_code_from_file_details(
        file_path,
        (math.sqrt, {"bad-key": 1}, 0),
    )
    assert math_code == (
        f'import math\n\ndata = math.sqrt({str(file_path)!r}, **{{"bad-key": 1}})'
    )

    missing_module_code = _load_code_from_file_details(
        file_path,
        (_missing_module_loader, {}, 0),
    )
    assert missing_module_code == (
        "import missing_erlab_loader_module\n\n"
        f"data = missing_erlab_loader_module.load({str(file_path)!r})"
    )

    chunked = xr.DataArray(
        np.zeros((5, 4)),
        dims=("x", "y"),
    ).chunk({"x": (2, 3), "y": 2})
    assert _format_chunk_summary(xr.DataArray(np.zeros(2), dims=("x",))) == "In memory"
    assert _format_chunk_summary(chunked) == "x=2, 3; y=2"


def test_choose_from_datatree_dialog_tree_helper_branches(qtbot) -> None:
    class _FakeManager(QtWidgets.QWidget):
        next_idx = 7

    manager = _FakeManager()
    qtbot.addWidget(manager)
    tree = xr.DataTree.from_dict(
        {
            "root/imagetool": xr.Dataset(attrs={"itool_title": "Root"}),
            "root/childtools/child/tool": xr.Dataset(attrs={"tool_title": "Child"}),
        }
    )

    dialog = _ChooseFromDataTreeDialog(
        typing.cast("ImageToolManager", manager),
        tree,
        "load",
    )
    qtbot.addWidget(dialog)

    root_item = dialog._tree_widget.topLevelItem(0)
    assert root_item is not None
    assert root_item.text(0) == "7: Root"
    child_item = root_item.child(0)
    assert child_item is not None
    assert child_item.text(0) == "Child"

    dialog._on_item_changed(root_item, 1)
    dialog._uncheck_children()
    assert child_item.checkState(0) == QtCore.Qt.CheckState.Unchecked

    dialog._check_all()
    assert root_item.checkState(0) == QtCore.Qt.CheckState.Checked
    assert child_item.checkState(0) == QtCore.Qt.CheckState.Checked

    child_item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
    dialog._on_item_changed(child_item, 0)
    assert root_item.checkState(0) == QtCore.Qt.CheckState.Unchecked

    dialog._populate_tree(typing.cast("xr.DataTree", {"bad": object()}))
    with pytest.raises(ValueError, match="supported payload"):
        dialog._node_payload(xr.DataTree())


def test_manager_goldtool_output_itool_stales_when_fit_results_change(
    qtbot,
    gold,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(gold, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child_uid = manager.add_childtool(
            GoldTool(gold.copy(deep=True), data_name="gold_input"),
            0,
            show=False,
        )
        child = manager.get_childtool(child_uid)
        assert isinstance(child, GoldTool)
        configure_goldtool_child(child, fitted=True, spline=False)
        child.open_itool()

        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)

        output_uid = child_node._childtool_indices[0]
        output_node = manager._child_node(output_uid)
        before = fetch(output_uid).copy(deep=True)

        child.post_fit(child.edge_center + 1, child.edge_stderr)

        qtbot.wait_until(lambda: output_node.source_state == "stale", timeout=5000)
        xr.testing.assert_identical(fetch(output_uid), before)


def test_manager_ximageitem_open_itool_creates_independent_top_level_window(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.open_in_meshtool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 1, timeout=5000
        )

        child_uid = manager._imagetool_wrappers[0]._childtool_indices[0]
        child = typing.cast("typing.Any", manager.get_childtool(child_uid))
        assert child.main_image.data_array is not None

        child.main_image.open_itool()

        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        child_node = manager._child_node(child_uid)
        assert child_node._childtool_indices == []
        output_node = manager._imagetool_wrappers[1]
        assert output_node.parent_uid is None
        assert output_node.output_id is None
        assert output_node.source_spec is None
        assert output_node.provenance_spec is not None
        assert output_node.provenance_spec.display_code() is not None
        xr.testing.assert_identical(fetch(1), child.main_image.data_array.T)

        monkeypatch.setattr(
            child,
            "_prompt_existing_output_imagetool",
            lambda: pytest.fail("unbound xImageItem opens should not prompt"),
        )
        updated = (child.main_image.data_array * 2).rename(
            child.main_image.data_array.name
        )
        child.main_image.setDataArray(updated)
        child.main_image.open_itool()

        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)
        assert child_node._childtool_indices == []
        second_output_node = manager._imagetool_wrappers[2]
        assert second_output_node.parent_uid is None
        assert second_output_node.output_id is None
        assert second_output_node.source_spec is None
        assert second_output_node.provenance_spec is not None
        assert second_output_node.provenance_spec.display_code() is not None
        xr.testing.assert_identical(fetch(2), updated.T)


def test_manager_workspace_roundtrip_independent_unbound_imagetool(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.open_in_meshtool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 1, timeout=5000
        )

        child_uid = manager._imagetool_wrappers[0]._childtool_indices[0]
        child = typing.cast("typing.Any", manager.get_childtool(child_uid))
        expected = child.main_image.data_array.T.copy(deep=True)

        child.main_image.open_itool()
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        tree = manager._to_datatree()

        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        for node in tree.values():
            manager._load_workspace_node(typing.cast("xr.DataTree", node))

        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        matching_roots = [
            wrapper
            for index, wrapper in manager._imagetool_wrappers.items()
            if wrapper.parent_uid is None
            and wrapper.source_spec is None
            and wrapper.provenance_spec is not None
            and wrapper.output_id is None
            and wrapper._childtool_indices == []
            and fetch(index).identical(expected)
        ]
        assert len(matching_roots) == 1
        assert matching_roots[0].provenance_spec is not None
        assert matching_roots[0].provenance_spec.display_code() is not None


def test_manager_metadata_uses_streamlined_child_derivation(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        data = xr.DataArray(
            np.arange(25).reshape((5, 5)),
            dims=["alpha", "eV"],
            coords={"alpha": np.arange(5), "eV": np.arange(5)},
        )
        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtool_indices) == 1,
            timeout=5000,
        )

        child_uid = manager._imagetool_wrappers[0]._childtool_indices[0]
        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)

        derivation = metadata_derivation_texts(manager)
        assert derivation[0] == "Start from selected parent ImageTool data"
        assert not any(line == "isel()" for line in derivation)
        assert not any("Sort coordinates" in line for line in derivation)
        assert any(line.startswith("transpose(") for line in derivation)

        copied = copy_full_code_for_uid(monkeypatch, manager, child_uid)
        namespace = _exec_generated_code(
            copied,
            {"data": parent_tool.slicer_area.data.copy(deep=True)},
        )
        result = namespace["result"]
        assert isinstance(result, xr.DataArray)
        xr.testing.assert_identical(result, manager.get_childtool(child_uid).result)
        assert ".isel()" not in copied
        assert "sort_coord_order" not in copied
        assert ".transpose(" in copied


def test_manager_archived_output_child_reopens_existing_slot(
    qtbot,
    monkeypatch,
    gold,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()

        itool(gold, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child = GoldTool(gold.copy(deep=True), data_name="gold_input")
        child_uid = manager.add_childtool(child, 0, show=False)
        configure_goldtool_child(child, fitted=True, spline=True)
        child.open_itool()

        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)

        output_uid = child_node._childtool_indices[0]
        output_node = manager._child_node(output_uid)
        output_node.archive()
        assert output_node.archived

        monkeypatch.setattr(
            child, "_prompt_existing_output_imagetool", lambda: "update"
        )
        child.open_itool()

        assert child_node._childtool_indices == [output_uid]
        assert not output_node.archived
        xr.testing.assert_identical(fetch(output_uid), child.corrected)


def test_manager_archived_auto_updating_child_refreshes_on_unarchive(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    prov = erlab.interactive.imagetool.provenance
    base = xr.DataArray(
        np.arange(16, dtype=float).reshape((4, 4)),
        dims=["x", "y"],
        coords={"x": np.arange(4), "y": np.arange(4)},
        name="scan",
    )
    updated = base.isel(x=slice(1, 3))

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root_tool = itool(base.isel(x=slice(0, 2)), manager=False, execute=False)
        assert isinstance(root_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(
            root_tool,
            show=False,
            provenance_spec=prov.selection(
                prov.IselOperation(kwargs={"x": slice(0, 2)})
            ),
        )

        child_tool = itool(base.isel(x=slice(0, 2)), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=prov.full_data(),
            source_auto_update=True,
        )

        child_node = manager._child_node(child_uid)
        child_node.archive()
        assert child_node.archived

        manager._imagetool_wrappers[0].set_detached_provenance(
            prov.selection(prov.IselOperation(kwargs={"x": slice(1, 3)}))
        )
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated)

        qtbot.wait_until(lambda: child_node.source_state == "stale", timeout=5000)
        assert child_node.archived

        child_node.unarchive()
        qtbot.wait_until(lambda: not child_node.archived, timeout=5000)
        qtbot.wait_until(lambda: child_node.source_state == "fresh", timeout=5000)
        xr.testing.assert_identical(fetch(child_uid), updated)


def test_manager_nested_imagetool_refresh_updates_descendant_lineage(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    prov = erlab.interactive.imagetool.provenance
    base = xr.DataArray(
        np.arange(16, dtype=float).reshape((4, 4)),
        dims=["x", "y"],
        coords={"x": np.arange(4), "y": np.arange(4)},
        name="scan",
    )
    initial_root_spec = prov.selection(prov.IselOperation(kwargs={"x": slice(0, 2)}))
    updated_root_spec = prov.selection(prov.IselOperation(kwargs={"x": slice(1, 3)}))

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root_data = base.isel(x=slice(0, 2))
        root_tool = itool(root_data, manager=False, execute=False)
        assert isinstance(root_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root_tool, show=False, provenance_spec=initial_root_spec)

        child_tool = itool(root_data.copy(deep=False), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=prov.full_data(),
            source_auto_update=True,
        )

        grandchild_data = root_data.isel(y=slice(0, 2))
        grandchild_tool = itool(grandchild_data, manager=False, execute=False)
        assert isinstance(grandchild_tool, erlab.interactive.imagetool.ImageTool)
        grandchild_uid = manager.add_imagetool_child(
            grandchild_tool,
            child_uid,
            show=False,
            source_spec=prov.selection(prov.IselOperation(kwargs={"y": slice(0, 2)})),
            source_auto_update=True,
        )

        root_node = manager._imagetool_wrappers[0]
        grandchild_node = manager._child_node(grandchild_uid)
        assert grandchild_node.provenance_spec is not None
        assert "slice(0, 2)" in typing.cast(
            "str", grandchild_node.provenance_spec.derivation_code()
        )

        root_node.set_detached_provenance(updated_root_spec)
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, base.isel(x=slice(1, 3)))

        qtbot.wait_until(
            lambda: (
                grandchild_node.provenance_spec is not None
                and grandchild_node.provenance_spec.derivation_code() is not None
                and "slice(1, 3)"
                in typing.cast("str", grandchild_node.provenance_spec.derivation_code())
            ),
            timeout=5000,
        )
        code = typing.cast("str", grandchild_node.provenance_spec.derivation_code())
        assert "derived = derived.isel(x=slice(1, 3))" in code
        assert "derived = derived.isel(x=slice(0, 2))" not in code


def test_manager_nested_imagetool_auto_update_can_be_disabled_from_auto_badge(
    qtbot,
    accept_dialog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    prov = erlab.interactive.imagetool.provenance
    base = xr.DataArray(
        np.arange(24, dtype=float).reshape((6, 4)),
        dims=["x", "y"],
        coords={"x": np.arange(6), "y": np.arange(4)},
        name="scan",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root_data = base.isel(x=slice(0, 2))
        root_tool = itool(root_data, manager=False, execute=False)
        assert isinstance(root_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(
            root_tool,
            show=False,
            provenance_spec=prov.selection(
                prov.IselOperation(kwargs={"x": slice(0, 2)})
            ),
        )

        child_tool = itool(root_data.copy(deep=False), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=prov.full_data(),
            source_auto_update=False,
        )
        child_node = manager._child_node(child_uid)

        updated = base.isel(x=slice(2, 4))
        manager._imagetool_wrappers[0].set_detached_provenance(
            prov.selection(prov.IselOperation(kwargs={"x": slice(2, 4)}))
        )
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated)

        qtbot.wait_until(lambda: child_node.source_state == "stale", timeout=5000)
        xr.testing.assert_identical(fetch(child_uid), root_data)

        def _enable_auto_update(dialog: QtWidgets.QDialog) -> None:
            dialog.auto_update_check.setChecked(True)  # type: ignore[attr-defined]

        def _update_now(dialog: QtWidgets.QDialog) -> None:
            dialog.update_button.click()  # type: ignore[attr-defined]

        click_child_status_badge(
            manager,
            child_uid,
            accept_dialog,
            pre_call=_enable_auto_update,
            accept_call=_update_now,
        )

        qtbot.wait_until(lambda: child_node.source_state == "fresh", timeout=5000)
        assert child_node.source_auto_update is True
        xr.testing.assert_identical(fetch(child_uid), updated)
        _, badge_text, _ = child_status_badge(manager, child_uid)
        assert badge_text == "Auto"

        def _disable_auto_update(dialog: QtWidgets.QDialog) -> None:
            dialog.auto_update_check.setChecked(False)  # type: ignore[attr-defined]

        click_child_status_badge(
            manager,
            child_uid,
            accept_dialog,
            pre_call=_disable_auto_update,
        )
        assert child_node.source_auto_update is False

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_actions()
        assert manager.source_update_action.isVisible()
        assert manager.source_update_action.isEnabled()

        unbound_tool = itool(updated.copy(deep=False), manager=False, execute=False)
        assert isinstance(unbound_tool, erlab.interactive.imagetool.ImageTool)
        unbound_uid = manager.add_imagetool_child(unbound_tool, 0, show=False)
        manager.tree_view.clearSelection()
        select_child_tool(manager, unbound_uid)
        manager._update_actions()
        assert not manager.source_update_action.isVisible()

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        select_child_tool(manager, unbound_uid)
        manager._update_actions()
        assert not manager.source_update_action.isVisible()

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_actions()
        assert manager.source_update_action.isVisible()
        select_tools(manager, [0])
        manager._update_actions()
        assert not manager.source_update_action.isVisible()

        updated2 = base.isel(x=slice(4, 6))
        manager._imagetool_wrappers[0].set_detached_provenance(
            prov.selection(prov.IselOperation(kwargs={"x": slice(4, 6)}))
        )
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated2)

        qtbot.wait_until(lambda: child_node.source_state == "stale", timeout=5000)
        xr.testing.assert_identical(fetch(child_uid), updated)


def test_manager_nested_stale_imagetool_marks_grandchildren_stale(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    prov = erlab.interactive.imagetool.provenance
    base = xr.DataArray(
        np.arange(16, dtype=float).reshape((4, 4)),
        dims=["x", "y"],
        coords={"x": np.arange(4), "y": np.arange(4)},
        name="scan",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root_data = base.isel(x=slice(0, 2))
        root_tool = itool(root_data, manager=False, execute=False)
        assert isinstance(root_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(
            root_tool,
            show=False,
            provenance_spec=prov.selection(
                prov.IselOperation(kwargs={"x": slice(0, 2)})
            ),
        )

        child_tool = itool(root_data.copy(deep=False), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=prov.full_data(),
            source_auto_update=False,
        )

        grandchild_tool = itool(
            root_data.isel(y=slice(0, 2)), manager=False, execute=False
        )
        assert isinstance(grandchild_tool, erlab.interactive.imagetool.ImageTool)
        grandchild_uid = manager.add_imagetool_child(
            grandchild_tool,
            child_uid,
            show=False,
            source_spec=prov.selection(prov.IselOperation(kwargs={"y": slice(0, 2)})),
            source_auto_update=True,
        )

        root_node = manager._imagetool_wrappers[0]
        child_node = manager._child_node(child_uid)
        grandchild_node = manager._child_node(grandchild_uid)

        root_node.set_detached_provenance(
            prov.selection(prov.IselOperation(kwargs={"x": slice(1, 3)}))
        )
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, base.isel(x=slice(1, 3)))

        qtbot.wait_until(lambda: child_node.source_state == "stale", timeout=5000)
        qtbot.wait_until(lambda: grandchild_node.source_state == "stale", timeout=5000)


def test_manager_manual_nested_refresh_updates_stale_ancestors(
    qtbot,
    accept_dialog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    prov = erlab.interactive.imagetool.provenance
    base = xr.DataArray(
        np.arange(24, dtype=float).reshape((6, 4)),
        dims=["x", "y"],
        coords={"x": np.arange(6), "y": np.arange(4)},
        name="scan",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root_data = base.isel(x=slice(0, 2))
        root_tool = itool(root_data, manager=False, execute=False)
        assert isinstance(root_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(
            root_tool,
            show=False,
            provenance_spec=prov.selection(
                prov.IselOperation(kwargs={"x": slice(0, 2)})
            ),
        )

        child_tool = itool(root_data.copy(deep=False), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=prov.full_data(),
            source_auto_update=False,
        )

        grandchild_tool = itool(
            root_data.isel(y=slice(0, 2)), manager=False, execute=False
        )
        assert isinstance(grandchild_tool, erlab.interactive.imagetool.ImageTool)
        grandchild_uid = manager.add_imagetool_child(
            grandchild_tool,
            child_uid,
            show=False,
            source_spec=prov.selection(prov.IselOperation(kwargs={"y": slice(0, 2)})),
            source_auto_update=False,
        )

        child_node = manager._child_node(child_uid)
        grandchild_node = manager._child_node(grandchild_uid)
        updated_root = base.isel(x=slice(2, 4))

        manager._imagetool_wrappers[0].set_detached_provenance(
            prov.selection(prov.IselOperation(kwargs={"x": slice(2, 4)}))
        )
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated_root)

        qtbot.wait_until(lambda: child_node.source_state == "stale", timeout=5000)
        qtbot.wait_until(lambda: grandchild_node.source_state == "stale", timeout=5000)

        def _update_now(dialog: QtWidgets.QDialog) -> None:
            dialog.update_button.click()  # type: ignore[attr-defined]

        click_child_status_badge(
            manager,
            grandchild_uid,
            accept_dialog,
            accept_call=_update_now,
        )

        qtbot.wait_until(lambda: child_node.source_state == "fresh", timeout=5000)
        qtbot.wait_until(lambda: grandchild_node.source_state == "fresh", timeout=5000)
        assert child_node.source_auto_update is False
        assert grandchild_node.source_auto_update is False
        xr.testing.assert_identical(fetch(child_uid), updated_root)
        xr.testing.assert_identical(
            fetch(grandchild_uid), updated_root.isel(y=slice(0, 2))
        )


def test_manager_manual_nested_refresh_resumes_after_deferred_parent(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    prov = erlab.interactive.imagetool.provenance

    class _DeferredToolState(pydantic.BaseModel):
        value: int = 0

    class _DeferredTool(erlab.interactive.utils.ToolWindow[_DeferredToolState]):
        StateModel = _DeferredToolState
        tool_name = "deferred-dummy"

        def __init__(self, data: xr.DataArray) -> None:
            super().__init__()
            self._data = data
            self._status = _DeferredToolState()
            self.pending_data: xr.DataArray | None = None

        @property
        def tool_status(self) -> _DeferredToolState:
            return self._status

        @tool_status.setter
        def tool_status(self, status: _DeferredToolState) -> None:
            self._status = status

        @property
        def tool_data(self) -> xr.DataArray:
            return self._data

        def update_data(self, new_data: xr.DataArray) -> bool:
            self.pending_data = new_data
            self._source_refresh_deferred = self.has_source_binding
            return False

        def finish_deferred_update(self) -> None:
            if self.pending_data is None:
                raise RuntimeError("No deferred data is pending")
            self._data = self.pending_data
            self.pending_data = None
            self.finalize_source_refresh()

    base = xr.DataArray(
        np.arange(24, dtype=float).reshape((6, 4)),
        dims=["x", "y"],
        coords={"x": np.arange(6), "y": np.arange(4)},
        name="scan",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root_data = base.isel(x=slice(0, 2))
        root_tool = itool(root_data, manager=False, execute=False)
        assert isinstance(root_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root_tool, show=False)

        parent_tool = _DeferredTool(root_data)
        parent_uid = manager.add_childtool(parent_tool, 0, show=False)
        parent_tool.set_source_binding(prov.full_data(), auto_update=False)

        leaf_tool = itool(root_data.isel(y=slice(0, 2)), manager=False, execute=False)
        assert isinstance(leaf_tool, erlab.interactive.imagetool.ImageTool)
        leaf_uid = manager.add_imagetool_child(
            leaf_tool,
            parent_uid,
            show=False,
            source_spec=prov.selection(prov.IselOperation(kwargs={"y": slice(0, 2)})),
            source_auto_update=False,
        )

        parent_node = manager._child_node(parent_uid)
        leaf_node = manager._child_node(leaf_uid)
        updated_root = base.isel(x=slice(2, 4))

        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated_root)

        qtbot.wait_until(lambda: parent_node.source_state == "stale", timeout=5000)
        qtbot.wait_until(lambda: leaf_node.source_state == "stale", timeout=5000)

        assert manager._refresh_source_chain_to_uid(leaf_uid) is False
        assert parent_tool.pending_data is not None
        xr.testing.assert_identical(fetch(leaf_uid), root_data.isel(y=slice(0, 2)))

        parent_tool.finish_deferred_update()

        qtbot.wait_until(lambda: parent_node.source_state == "fresh", timeout=5000)
        qtbot.wait_until(lambda: leaf_node.source_state == "fresh", timeout=5000)
        xr.testing.assert_identical(parent_tool.tool_data, updated_root)
        xr.testing.assert_identical(fetch(leaf_uid), updated_root.isel(y=slice(0, 2)))
        assert manager._pending_source_refresh_targets == {}


def test_manager_archived_nested_imagetool_refreshes_descendants_on_unarchive(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    prov = erlab.interactive.imagetool.provenance
    base = xr.DataArray(
        np.arange(16, dtype=float).reshape((4, 4)),
        dims=["x", "y"],
        coords={"x": np.arange(4), "y": np.arange(4)},
        name="scan",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root_data = base.isel(x=slice(0, 2))
        updated_root = base.isel(x=slice(1, 3))

        root_tool = itool(root_data, manager=False, execute=False)
        assert isinstance(root_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(
            root_tool,
            show=False,
            provenance_spec=prov.selection(
                prov.IselOperation(kwargs={"x": slice(0, 2)})
            ),
        )

        child_tool = itool(root_data.copy(deep=False), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=prov.full_data(),
            source_auto_update=True,
        )

        grandchild_tool = itool(
            root_data.isel(y=slice(0, 2)), manager=False, execute=False
        )
        assert isinstance(grandchild_tool, erlab.interactive.imagetool.ImageTool)
        grandchild_uid = manager.add_imagetool_child(
            grandchild_tool,
            child_uid,
            show=False,
            source_spec=prov.selection(prov.IselOperation(kwargs={"y": slice(0, 2)})),
            source_auto_update=True,
        )

        child_node = manager._child_node(child_uid)
        grandchild_node = manager._child_node(grandchild_uid)

        child_node.archive()
        assert child_node.archived
        qtbot.wait_until(
            lambda: grandchild_node.source_state == "unavailable", timeout=5000
        )

        manager._imagetool_wrappers[0].set_detached_provenance(
            prov.selection(prov.IselOperation(kwargs={"x": slice(1, 3)}))
        )
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated_root)

        qtbot.wait_until(lambda: child_node.source_state == "stale", timeout=5000)
        assert grandchild_node.source_state == "unavailable"

        child_node.unarchive()
        qtbot.wait_until(lambda: not child_node.archived, timeout=5000)
        qtbot.wait_until(lambda: child_node.source_state == "fresh", timeout=5000)
        qtbot.wait_until(lambda: grandchild_node.source_state == "fresh", timeout=5000)

        xr.testing.assert_identical(fetch(child_uid), updated_root)
        xr.testing.assert_identical(
            fetch(grandchild_uid), updated_root.isel(y=slice(0, 2))
        )


def test_manager_meshtool_output_itools_use_distinct_output_ids(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.open_in_meshtool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 1, timeout=5000
        )

        child_uid = manager._imagetool_wrappers[0]._childtool_indices[0]
        child = typing.cast("typing.Any", manager.get_childtool(child_uid))
        monkeypatch.setattr(
            child,
            "_prompt_existing_output_imagetool",
            lambda: (_ for _ in ()).throw(AssertionError("prompt should not open")),
        )

        child._corrected = child.tool_data.copy(deep=True) + 1
        child._mesh = child.tool_data.copy(deep=True) - 1

        child._corr_itool()
        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)

        child._mesh_itool()
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 2, timeout=5000)

        corr_uid, mesh_uid = child_node._childtool_indices
        corr_node = manager._child_node(corr_uid)
        mesh_node = manager._child_node(mesh_uid)
        assert manager.ntools == 1
        assert corr_node.parent_uid == child_uid
        assert mesh_node.parent_uid == child_uid
        assert corr_node.output_id == "meshtool.corrected_output"
        assert mesh_node.output_id == "meshtool.mesh_output"
        assert corr_node.source_spec is None
        assert corr_node.provenance_spec is not None
        assert mesh_node.source_spec is None
        assert mesh_node.provenance_spec is not None
        xr.testing.assert_identical(fetch(corr_uid), child._corrected)
        xr.testing.assert_identical(fetch(mesh_uid), child._mesh)


@pytest.mark.parametrize(
    ("output_id", "expected_name"),
    [
        ("meshtool.corrected_output", "corrected"),
        ("meshtool.mesh_output", "mesh"),
    ],
)
def test_manager_meshtool_output_child_qsel_copy_code_tracks_selected_output_id(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
    output_id: str,
    expected_name: str,
) -> None:
    prov = erlab.interactive.imagetool.provenance

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.open_in_meshtool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 1, timeout=5000
        )

        child_uid = manager._imagetool_wrappers[0]._childtool_indices[0]
        child = typing.cast("typing.Any", manager.get_childtool(child_uid))
        child._corrected = child.tool_data.copy(deep=True) + 1
        child._mesh = child.tool_data.copy(deep=True) - 1

        if output_id == "meshtool.corrected_output":
            child._corr_itool()
        else:
            child._mesh_itool()

        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)

        output_uid = child_node._childtool_indices[0]
        output_data = fetch(output_uid)
        nested_tool = itool(
            output_data.qsel(alpha=1, alpha_width=1), manager=False, execute=False
        )
        nested_uid = manager.add_imagetool_child(
            nested_tool,
            output_uid,
            show=False,
            source_spec=prov.selection(
                prov.QSelOperation(kwargs={"alpha": 1, "alpha_width": 1})
            ),
            source_auto_update=True,
        )

        copied = copy_full_code_for_uid(monkeypatch, manager, nested_uid)
        assert "corrected, mesh =" in copied
        assert "era.mesh.remove_mesh(" in copied
        assert f"derived = {expected_name}" in copied
        assert ")[0]" not in copied
        assert ")[1]" not in copied
        assert "derived = derived.qsel(alpha=1, alpha_width=1)" in copied


def test_manager_fit2d_output_itools_use_distinct_output_ids(
    qtbot,
    monkeypatch,
    exp_decay_model,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child_uid, child = make_fit2d_child(manager, 0, exp_decay_model)
        monkeypatch.setattr(
            child,
            "_prompt_existing_output_imagetool",
            lambda: (_ for _ in ()).throw(AssertionError("prompt should not open")),
        )
        child.timeout_spin.setValue(30.0)
        child.nfev_spin.setValue(0)
        child.y_index_spin.setValue(child.y_min_spin.value())
        child._run_fit_2d("up")
        qtbot.wait_until(
            lambda: all(ds is not None for ds in child._result_ds_full),
            timeout=10000,
        )

        child.param_plot_combo.setCurrentIndex(0)
        param_name = child.param_plot_combo.currentText()
        assert param_name

        child.param_plot._show_parameter_values()
        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)

        child.param_plot._show_parameter_stderr()
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 2, timeout=5000)

        values_uid, stderr_uid = child_node._childtool_indices
        values_node = manager._child_node(values_uid)
        stderr_node = manager._child_node(stderr_uid)
        assert manager.ntools == 1
        assert values_node.parent_uid == child_uid
        assert stderr_node.parent_uid == child_uid
        assert values_node.output_id == "fit2d.param_plot.values"
        assert stderr_node.output_id == "fit2d.param_plot.stderr"
        assert values_node.source_spec is None
        assert values_node.provenance_spec is not None
        assert stderr_node.source_spec is None
        assert stderr_node.provenance_spec is not None
        xr.testing.assert_identical(
            fetch(values_uid), child._param_plot_dataarray(param_name, stderr=False)
        )
        xr.testing.assert_identical(
            fetch(stderr_uid), child._param_plot_dataarray(param_name, stderr=True)
        )
        values_code = copy_full_code_for_uid(monkeypatch, manager, values_uid)
        stderr_code = copy_full_code_for_uid(monkeypatch, manager, stderr_uid)
        assert ".modelfit_coefficients.sel(param=" in values_code
        assert ".modelfit_stderr.sel(param=" in stderr_code


def test_manager_output_refresh_updates_stale_parent_source(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    prov = erlab.interactive.imagetool.provenance

    class _OutputToolState(pydantic.BaseModel):
        value: int = 0

    class _OutputTool(erlab.interactive.utils.ToolWindow[_OutputToolState]):
        StateModel = _OutputToolState
        tool_name = "output-dummy"

        def __init__(self, data: xr.DataArray) -> None:
            super().__init__()
            self._data = data
            self._status = _OutputToolState()
            self.refreshed_inputs: list[xr.DataArray] = []

        @property
        def tool_status(self) -> _OutputToolState:
            return self._status

        @tool_status.setter
        def tool_status(self, status: _OutputToolState) -> None:
            self._status = status

        @property
        def tool_data(self) -> xr.DataArray:
            return self._data

        def update_data(self, new_data: xr.DataArray) -> bool:
            self.refreshed_inputs.append(new_data)
            self._data = new_data
            return True

        def output_imagetool_data(
            self, output_id: str | enum.Enum
        ) -> xr.DataArray | None:
            assert output_id == "out"
            return self._data + 10.0

        def output_imagetool_provenance(
            self, output_id: str | enum.Enum, data: xr.DataArray
        ) -> erlab.interactive.imagetool.provenance.ToolProvenanceSpec | None:
            assert output_id == "out"
            return prov.script(
                prov.ScriptCodeOperation(label="Use output", code="result = data + 10"),
                start_label="Start from parent",
                active_name="result",
            )

    data = xr.DataArray(
        np.arange(12, dtype=float).reshape((3, 4)),
        dims=["x", "y"],
        coords={"x": np.arange(3), "y": np.arange(4)},
        name="scan",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root_tool = itool(data, manager=False, execute=False)
        assert isinstance(root_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root_tool, show=False)

        child = _OutputTool(data)
        child_uid = manager.add_childtool(child, 0, show=False)
        child.set_source_binding(prov.full_data(), auto_update=False)

        initial_output = typing.cast("xr.DataArray", child.output_imagetool_data("out"))
        output_tool = itool(initial_output, manager=False, execute=False)
        assert isinstance(output_tool, erlab.interactive.imagetool.ImageTool)
        output_uid = manager.add_imagetool_child(
            output_tool,
            child_uid,
            show=False,
            provenance_spec=child.output_imagetool_provenance("out", initial_output),
            source_state="fresh",
            output_id="out",
        )

        child_node = manager._child_node(child_uid)
        output_node = manager._child_node(output_uid)
        updated = data * 2.0

        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated)

        qtbot.wait_until(lambda: child_node.source_state == "stale", timeout=5000)
        qtbot.wait_until(lambda: output_node.source_state == "stale", timeout=5000)
        xr.testing.assert_identical(fetch(output_uid), initial_output)

        assert manager._refresh_source_chain_to_uid(output_uid) is True
        assert child.refreshed_inputs
        assert child.source_state == "fresh"
        assert output_node.source_state == "fresh"
        xr.testing.assert_identical(fetch(output_uid), updated + 10.0)


def test_manager_fit2d_unbound_output_itool_creates_independent_top_level_windows(
    qtbot,
    monkeypatch,
    exp_decay_model,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child_uid, child = make_fit2d_child(manager, 0, exp_decay_model)
        initial = xr.DataArray(
            np.arange(4.0), dims=("x",), coords={"x": np.arange(4)}, name="initial"
        )
        updated = xr.DataArray(
            np.arange(4.0) + 10,
            dims=("x",),
            coords={"x": np.arange(4)},
            name="updated",
        )

        child._show_dataarray_in_itool(initial)
        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        assert child_node._childtool_indices == []
        first_output_node = manager._imagetool_wrappers[1]
        assert first_output_node.parent_uid is None
        assert first_output_node.output_id is None
        assert first_output_node.source_spec is None
        assert first_output_node.provenance_spec is None
        assert not first_output_node.reloadable
        xr.testing.assert_identical(fetch(1), initial)
        monkeypatch.setattr(
            child,
            "_prompt_existing_output_imagetool",
            lambda: pytest.fail("unbound fit2d opens should not prompt"),
        )

        child._show_dataarray_in_itool(updated)

        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)
        assert child_node._childtool_indices == []
        second_output_node = manager._imagetool_wrappers[2]
        assert second_output_node.parent_uid is None
        assert second_output_node.output_id is None
        assert second_output_node.source_spec is None
        assert second_output_node.provenance_spec is None
        assert not second_output_node.reloadable
        xr.testing.assert_identical(fetch(2), updated)


def test_manager_open_in_new_window_nests_imagetool_children(
    qtbot,
    monkeypatch,
    accept_dialog,
    tmp_path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        file_dir = tmp_path / ("very_long_directory_name_" * 4)
        file_dir.mkdir(parents=True)
        file_path = file_dir / "scan_with_a_long_name.h5"
        test_data.to_netcdf(file_path, engine="h5netcdf")

        itool(
            test_data,
            manager=True,
            file_path=file_path,
            load_func=(xr.load_dataarray, {"engine": "h5netcdf"}, 0),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent = manager._imagetool_wrappers[0]
        manager.tree_view.clearSelection()
        select_tools(manager, [0])
        manager._update_info()
        root_index = manager.tree_view._model._row_index(0)
        assert root_index.data(_NODE_UID_ROLE) == parent.uid
        right_splitter = typing.cast(
            "QtWidgets.QSplitter", manager.text_box.parentWidget()
        )
        assert right_splitter.indexOf(manager.preview_widget) < right_splitter.indexOf(
            manager.metadata_group
        )
        details = metadata_detail_map(manager)
        assert details["Kind"] == "ImageTool"
        assert details["File"] == str(file_path)
        assert "Chunks" not in details
        assert "Added" in details
        assert metadata_derivation_texts(manager) == [
            "Load data from file 'scan_with_a_long_name.h5'"
        ]
        assert manager._build_metadata_derivation_menu() is not None

        copied: list[str] = []
        monkeypatch.setattr(
            erlab.interactive.utils,
            "copy_to_clipboard",
            lambda text: copied.append(text) or text,
        )
        file_label = manager._metadata_detail_labels["File"]
        assert file_label.toolTip() == str(file_path)
        file_label.setFixedWidth(84)
        qtbot.waitUntil(
            lambda: (
                getattr(file_label, "full_text", file_label.text()) != file_label.text()
            ),
            timeout=2000,
        )

        def _inspect_source_dialog(dialog: QtWidgets.QDialog) -> None:
            assert dialog.path_edit.text() == str(file_path)  # type: ignore[attr-defined]
            assert (
                dialog.loader_edit.text().endswith("xarray.load_dataarray")  # type: ignore[attr-defined]
            )
            assert (
                dialog.kwargs_edit.toPlainText() == 'engine="h5netcdf"'  # type: ignore[attr-defined]
            )
            dialog.copy_code_button.click()  # type: ignore[attr-defined]

        accept_dialog(
            lambda: qtbot.mouseClick(file_label, QtCore.Qt.MouseButton.LeftButton),
            pre_call=_inspect_source_dialog,
        )
        assert copied
        load_namespace = _exec_generated_code(copied[-1], {})
        xr.testing.assert_identical(
            load_namespace["data"],
            xr.load_dataarray(file_path, engine="h5netcdf"),
        )

        manager.get_imagetool(0).slicer_area.images[0].open_in_new_window()
        qtbot.wait_until(lambda: len(parent._childtool_indices) == 1, timeout=5000)

        child_uid = parent._childtool_indices[0]
        child_node = manager._child_node(child_uid)
        child_tool = manager.get_imagetool(child_uid)

        assert child_node.is_imagetool
        assert child_node.parent_uid == parent.uid
        assert child_node.source_spec is not None
        xr.testing.assert_identical(fetch(child_uid), child_tool.slicer_area._data)

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)
        child_index = manager.tree_view._model._row_index(child_uid)
        assert child_index.data(_NODE_UID_ROLE) == child_uid
        child_details = metadata_detail_map(manager)
        assert child_details["Kind"] == "ImageTool"
        assert "Added" in child_details
        assert child_details["File"] == str(file_path)
        assert "Chunks" not in child_details
        assert metadata_derivation_texts(manager)

        child_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)

        nested_uid = child_node._childtool_indices[0]
        nested_tool = manager.get_childtool(nested_uid)
        assert isinstance(nested_tool, DerivativeTool)

        manager.tree_view.clearSelection()
        select_child_tool(manager, nested_uid)
        manager._update_info(uid=nested_uid)
        nested_details = metadata_detail_map(manager)
        assert nested_details["Kind"] == nested_tool.tool_name
        assert "Added" in nested_details
        assert metadata_derivation_texts(manager)
        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        monkeypatch.setattr(
            manager,
            "_prompt_replay_input_name",
            lambda _node: pytest.fail("file-backed replay should not prompt"),
        )
        action_map(menu)["Copy Full Code"].trigger()
        assert copied
        assert not erlab.interactive.imagetool.provenance.uses_default_replay_input(
            copied[-1]
        )
        namespace = _exec_generated_code(copied[-1], {})
        result = namespace["result"]
        assert isinstance(result, xr.DataArray)
        xr.testing.assert_identical(result, nested_tool.result)


def test_manager_promote_action_enablement_and_menus(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent = manager._imagetool_wrappers[0]
        manager.get_imagetool(0).slicer_area.images[0].open_in_new_window()
        qtbot.wait_until(lambda: len(parent._childtool_indices) == 1, timeout=5000)

        child_uid = parent._childtool_indices[0]
        child_tool = manager.get_imagetool(child_uid)
        child_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._child_node(child_uid)._childtool_indices) == 1,
            timeout=5000,
        )
        nested_uid = manager._child_node(child_uid)._childtool_indices[0]

        menu_actions = {
            action.text().replace("&", ""): action
            for action in manager.menu_bar.actions()
            if action.menu() is not None
        }
        assert "Promote Window" in action_map(
            typing.cast("QtWidgets.QMenu", menu_actions["Edit"].menu())
        )
        assert (
            action_map(manager.tree_view._menu)["Promote Window"]
            is manager.promote_action
        )

        manager.tree_view.clearSelection()
        manager._update_actions()
        assert not manager.promote_action.isEnabled()

        select_tools(manager, [0])
        manager._update_actions()
        assert not manager.promote_action.isEnabled()

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_actions()
        assert manager.promote_action.isEnabled()

        select_tools(manager, [0])
        manager._update_actions()
        assert not manager.promote_action.isEnabled()

        manager.tree_view.clearSelection()
        select_child_tool(manager, nested_uid)
        manager._update_actions()
        assert not manager.promote_action.isEnabled()

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._child_node(child_uid).archive()
        manager._update_actions()
        assert not manager.promote_action.isEnabled()


def test_manager_rename_action_enablement_for_child_selection(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent = manager._imagetool_wrappers[0]
        manager.get_imagetool(0).slicer_area.images[0].open_in_new_window()
        qtbot.wait_until(lambda: len(parent._childtool_indices) == 1, timeout=5000)

        child_uid = parent._childtool_indices[0]
        child_tool = manager.get_imagetool(child_uid)
        child_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._child_node(child_uid)._childtool_indices) == 1,
            timeout=5000,
        )
        nested_uid = manager._child_node(child_uid)._childtool_indices[0]

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_actions()
        assert manager.rename_action.isEnabled()

        select_tools(manager, [0])
        manager._update_actions()
        assert not manager.rename_action.isEnabled()

        manager.tree_view.clearSelection()
        select_child_tool(manager, nested_uid)
        manager._update_actions()
        assert manager.rename_action.isEnabled()

        manager.rename_action.trigger()
        qtbot.wait_until(
            lambda: (
                manager.tree_view.state()
                == QtWidgets.QAbstractItemView.State.EditingState
            ),
            timeout=5000,
        )
        delegate = manager.tree_view.itemDelegate()
        assert isinstance(delegate, _ImageToolWrapperItemDelegate)
        assert isinstance(delegate._current_editor, QtWidgets.QLineEdit)
        delegate._current_editor.setText("renamed_child_tool")
        qtbot.keyClick(delegate._current_editor, QtCore.Qt.Key.Key_Return)
        qtbot.wait_until(
            lambda: (
                manager.get_childtool(nested_uid)._tool_display_name
                == "renamed_child_tool"
            ),
            timeout=5000,
        )

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        select_child_tool(manager, nested_uid)
        manager._update_actions()
        assert not manager.rename_action.isEnabled()


def test_manager_promote_selected_cancel_keeps_nested_imagetool(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent = manager._imagetool_wrappers[0]
        manager.get_imagetool(0).slicer_area.images[0].open_in_new_window()
        qtbot.wait_until(lambda: len(parent._childtool_indices) == 1, timeout=5000)

        child_uid = parent._childtool_indices[0]
        select_child_tool(manager, child_uid)

        captured: dict[str, str] = {}

        def _cancel_prompt(
            dialog: QtWidgets.QMessageBox,
        ) -> QtWidgets.QMessageBox.StandardButton:
            captured["text"] = dialog.text()
            captured["info"] = dialog.informativeText()
            return QtWidgets.QMessageBox.StandardButton.Cancel

        monkeypatch.setattr(QtWidgets.QMessageBox, "exec", _cancel_prompt)

        manager.promote_action.trigger()

        assert captured["text"] == "Promote selected ImageTool to a top-level window?"
        assert "live update linkage" in captured["info"].lower()
        assert "detached history" in captured["info"].lower()
        assert manager.ntools == 1
        assert parent._childtool_indices == [child_uid]
        assert manager._child_node(child_uid).parent_uid == parent.uid


def test_manager_promote_child_imagetool_rehomes_subtree_and_detaches_provenance(
    qtbot,
    monkeypatch,
    accept_dialog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(60).reshape((3, 4, 5)).astype(float),
        dims=["x", "y", "z"],
        coords={"x": np.arange(3), "y": np.arange(4), "z": np.arange(5)},
        name="scan",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)

        def _nest_average(dialog) -> None:
            dialog.dim_checks["x"].setChecked(True)
            set_transform_launch_mode(dialog, "nest")

        accept_dialog(parent_tool.mnb._average, pre_call=_nest_average)

        parent = manager._imagetool_wrappers[0]
        qtbot.wait_until(lambda: len(parent._childtool_indices) == 1, timeout=5000)

        child_uid = parent._childtool_indices[0]
        child_node = manager._child_node(child_uid)
        child_tool = manager.get_imagetool(child_uid)
        child_node.name = "averaged child"
        child_before = fetch(child_uid).copy(deep=True)

        child_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)
        nested_uid = child_node._childtool_indices[0]

        select_child_tool(manager, child_uid)
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "exec",
            lambda _: QtWidgets.QMessageBox.StandardButton.Yes,
        )

        manager.promote_action.trigger()

        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        promoted_index = 1
        promoted = manager._imagetool_wrappers[promoted_index]
        assert promoted.uid == child_uid
        assert child_uid not in parent._childtool_indices
        assert promoted.parent_uid is None
        assert promoted.source_spec is None
        assert promoted.provenance_spec is not None
        assert promoted._childtool_indices == [nested_uid]
        assert manager._child_node(nested_uid).parent_uid == child_uid
        assert manager.tree_view.selected_imagetool_indices == [promoted_index]
        assert manager.tree_view.selected_childtool_uids == []
        assert manager._root_wrapper_for_uid(nested_uid).index == promoted_index
        assert (
            manager.get_imagetool(promoted_index)
            .windowTitle()
            .startswith(f"{promoted_index}: averaged child")
        )
        xr.testing.assert_identical(fetch(child_uid), child_before)
        xr.testing.assert_identical(
            manager._parent_source_data_for_uid(nested_uid),
            manager.get_imagetool(promoted_index).slicer_area._data,
        )

        manager._update_info()
        derivation = metadata_derivation_texts(manager)
        assert any("Average" in line for line in derivation)

        updated = data.copy(deep=True)
        updated.data = np.asarray(updated.data) + 10
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated)

        assert promoted.source_state == "fresh"
        xr.testing.assert_identical(fetch(child_uid), child_before)


def test_manager_replace_current_sets_provenance_on_provenance_free_root(
    qtbot,
    accept_dialog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(60).reshape((3, 4, 5)).astype(float),
        dims=["x", "y", "z"],
        coords={"x": np.arange(3), "y": np.arange(4), "z": np.arange(5)},
        name="scan",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        root = manager._imagetool_wrappers[0]
        root_tool = manager.get_imagetool(0)
        assert root.provenance_spec is None

        def _replace_average(dialog) -> None:
            dialog.dim_checks["x"].setChecked(True)
            set_transform_launch_mode(dialog, "replace")

        accept_dialog(root_tool.mnb._average, pre_call=_replace_average)

        assert root.source_spec is None
        assert root.provenance_spec is not None
        assert root.provenance_spec.derivation_code() == (
            'derived = data\nderived = derived.qsel.average("x")'
        )
        xr.testing.assert_identical(
            root_tool.slicer_area._data.rename(None),
            data.qsel.average("x").rename(None),
        )

        manager.tree_view.clearSelection()
        select_tools(manager, [0])
        manager._update_info()
        derivation = metadata_derivation_texts(manager)
        assert derivation == [
            "Start from current parent ImageTool data",
            'Average(dims=("x",))',
        ]


def test_manager_file_backed_replace_current_keeps_file_provenance(
    qtbot,
    monkeypatch,
    accept_dialog,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    file_path = tmp_path / "scan.h5"
    test_data.to_netcdf(file_path, engine="h5netcdf")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(
            test_data,
            manager=True,
            file_path=file_path,
            load_func=(xr.load_dataarray, {"engine": "h5netcdf"}, 0),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        root = manager._imagetool_wrappers[0]
        root_tool = manager.get_imagetool(0)
        assert root.provenance_spec is not None
        assert root.provenance_spec.display_entries()[0].label == (
            "Load data from file 'scan.h5'"
        )

        def _replace_average(dialog) -> None:
            dialog.dim_checks["alpha"].setChecked(True)
            set_transform_launch_mode(dialog, "replace")

        accept_dialog(root_tool.mnb._average, pre_call=_replace_average)

        assert root.provenance_spec is not None
        assert root.provenance_spec.kind == "file"
        assert len(root.provenance_spec.replay_stages) == 1
        assert root.provenance_spec.replay_stages[0].source_kind == "full_data"
        assert [op.op for op in root.provenance_spec.replay_stages[0].operations] == [
            "average",
            "rename",
        ]
        entries = root.provenance_spec.display_entries()
        assert entries[0].label == "Load data from file 'scan.h5'"
        assert any("Average" in entry.label for entry in entries)

        manager.tree_view.clearSelection()
        select_tools(manager, [0])
        manager._update_info()
        assert metadata_derivation_texts(manager)[0] == "Load data from file 'scan.h5'"

        copied: list[str] = []
        monkeypatch.setattr(
            erlab.interactive.utils,
            "copy_to_clipboard",
            lambda text: copied.append(text) or text,
        )
        monkeypatch.setattr(
            manager,
            "_prompt_replay_input_name",
            lambda _node: pytest.fail("file-backed replay should not prompt"),
        )
        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        action_map(menu)["Copy Full Code"].trigger()
        assert copied
        assert "scan.h5" in copied[-1]

        namespace = _exec_generated_code(copied[-1], {})
        derived = namespace["derived"]
        assert isinstance(derived, xr.DataArray)
        xr.testing.assert_identical(
            derived.rename(None),
            xr.load_dataarray(file_path, engine="h5netcdf")
            .astype(np.float64)
            .qsel.average("alpha")
            .rename(None),
        )


def test_manager_detached_file_provenance_metadata_and_reload_roundtrip(
    qtbot,
    accept_dialog,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    file_path = tmp_path / "scan.h5"
    test_data.to_netcdf(file_path, engine="h5netcdf")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(
            test_data,
            manager=True,
            file_path=file_path,
            load_func=(xr.load_dataarray, {"engine": "h5netcdf"}, 0),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        root_tool = manager.get_imagetool(0)

        def _detach_average(dialog) -> None:
            dialog.dim_checks["alpha"].setChecked(True)
            set_transform_launch_mode(dialog, "detach")

        accept_dialog(root_tool.mnb._average, pre_call=_detach_average)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        tree = manager._to_datatree()
        provenance_payload = json.loads(
            tree["1/imagetool"].attrs["manager_node_provenance_spec"]
        )
        assert provenance_payload["schema_version"] == 2
        assert provenance_payload["kind"] == "file"
        assert provenance_payload["operations"] == []
        assert len(provenance_payload["replay_stages"]) == 1
        assert provenance_payload["replay_stages"][0]["source_kind"] == "full_data"
        assert [
            operation["op"]
            for operation in provenance_payload["replay_stages"][0]["operations"]
        ] == ["average", "rename"]
        assert (
            provenance_payload["file_load_source"]["replay_call"]["target"]
            == "xarray.load_dataarray"
        )
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        for node in tree.values():
            manager._load_workspace_node(typing.cast("xr.DataTree", node))

        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        detached = manager._imagetool_wrappers[1]
        detached_tool = manager.get_imagetool(1)
        assert detached.parent_uid is None
        assert detached.output_id is None
        assert detached.source_spec is None
        assert detached.provenance_spec is not None
        assert detached.provenance_spec.file_load_source is not None
        assert detached.reloadable
        assert detached_tool.slicer_area._file_path is None

        manager.tree_view.clearSelection()
        select_tools(manager, [1])
        manager._update_actions()
        manager._update_info()
        assert metadata_detail_map(manager)["File"] == str(file_path)
        assert metadata_derivation_texts(manager)[0] == "Load data from file 'scan.h5'"
        assert manager.reload_action.isVisible()

        updated = test_data + 100
        updated.to_netcdf(file_path, engine="h5netcdf")

        with qtbot.wait_signal(detached_tool.slicer_area.sigDataChanged):
            manager.reload_selected()

        assert detached.parent_uid is None
        assert detached.output_id is None
        assert detached.source_spec is None
        assert detached.provenance_spec is not None
        assert detached_tool.slicer_area._file_path is None
        xr.testing.assert_identical(
            fetch(1).rename(None),
            updated.astype(np.float64).qsel.average("alpha").rename(None),
        )

        file_path.unlink()
        manager._update_actions()
        assert not detached.reloadable
        assert not manager.reload_action.isVisible()


def test_manager_workspace_loads_legacy_321_provenance_payload(
    qtbot,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    file_path = tmp_path / "scan.h5"
    test_data.to_netcdf(file_path, engine="h5netcdf")
    legacy_payload = {
        "schema_version": 1,
        "kind": "script",
        "start_label": "Load data from file 'scan.h5'",
        "seed_code": (
            "import xarray\n\n"
            f"derived = xarray.load_dataarray({str(file_path)!r}, "
            'engine="h5netcdf").astype("float64")'
        ),
        "active_name": "derived",
        "file_load_source": {
            "path": str(file_path),
            "loader_label": "Load Function",
            "loader_text": "xarray.load_dataarray",
            "kwargs_text": 'engine="h5netcdf"',
            "load_code": (
                "import xarray\n\n"
                f"data = xarray.load_dataarray({str(file_path)!r}, "
                'engine="h5netcdf").astype("float64")'
            ),
        },
        "operations": [
            {
                "op": "script_code",
                "label": 'Average(dims=("alpha",))',
                "code": 'derived = derived.qsel.average("alpha")',
                "copyable": True,
            }
        ],
    }

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        tree = manager._to_datatree()
        tree["0/imagetool"].attrs["manager_node_provenance_spec"] = json.dumps(
            legacy_payload
        )
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        for node in tree.values():
            manager._load_workspace_node(typing.cast("xr.DataTree", node))

        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        loaded = manager._imagetool_wrappers[0]
        assert loaded.provenance_spec is not None
        assert loaded.provenance_spec.schema_version == 2
        assert loaded.provenance_spec.kind == "script"
        assert loaded.provenance_spec.file_load_source is not None
        assert loaded.provenance_spec.file_load_source.replay_call is None
        assert not loaded.reloadable

        manager.tree_view.clearSelection()
        select_tools(manager, [0])
        manager._update_info()
        assert metadata_detail_map(manager)["File"] == str(file_path)
        assert metadata_derivation_texts(manager) == [
            "Load data from file 'scan.h5'",
            'Average(dims=("alpha",))',
        ]


def test_manager_prompt_replay_input_name_accept_cancel_and_invalid(
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    class _Node:
        def __init__(self, name: str | None) -> None:
            self.data = xr.DataArray(np.arange(2), dims=("x",), name=name)

        def _metadata_data(self) -> xr.DataArray:
            return self.data

    class _FakeLineEdit:
        def __init__(self) -> None:
            self.validator_set = False
            self.selected = False

        def setValidator(self, _validator) -> None:
            self.validator_set = True

        def selectAll(self) -> None:
            self.selected = True

    class _FakeInputDialog:
        InputMode = QtWidgets.QInputDialog.InputMode
        responses: typing.ClassVar[list[tuple[int, str]]] = []
        instances: typing.ClassVar[list[typing.Any]] = []

        def __init__(self, _parent) -> None:
            self.line_edit = _FakeLineEdit()
            self.initial_text = ""
            self._result, self._text = self.responses.pop(0)
            self.instances.append(self)

        def setWindowTitle(self, _title: str) -> None:
            pass

        def setLabelText(self, _text: str) -> None:
            pass

        def setTextValue(self, text: str) -> None:
            self.initial_text = text

        def setInputMode(self, _mode) -> None:
            pass

        def findChild(self, _cls):
            return self.line_edit

        def exec(self) -> int:
            return self._result

        def textValue(self) -> str:
            return self._text

    accepted = int(QtWidgets.QDialog.DialogCode.Accepted)
    rejected = int(QtWidgets.QDialog.DialogCode.Rejected)
    _FakeInputDialog.responses = [
        (rejected, ""),
        (accepted, "bad-name"),
        (accepted, " custom_source "),
    ]
    monkeypatch.setattr(QtWidgets, "QInputDialog", _FakeInputDialog)

    with manager_context() as manager:
        assert manager._prompt_replay_input_name(_Node("data")) is None
        assert _FakeInputDialog.instances[0].initial_text == "source_data"
        assert _FakeInputDialog.instances[0].line_edit.validator_set
        assert _FakeInputDialog.instances[0].line_edit.selected

        assert manager._prompt_replay_input_name(_Node("valid_name")) is None
        assert _FakeInputDialog.instances[1].initial_text == "valid_name"

        assert manager._prompt_replay_input_name(_Node(None)) == "custom_source"
        assert _FakeInputDialog.instances[2].initial_text == "source_data"


def test_manager_nonuniform_transform_children_refresh_from_public_data(
    qtbot,
    accept_dialog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(20).reshape((5, 4)).astype(float),
        dims=["x", "y"],
        coords={"x": [0.0, 0.2, 0.8, 1.4, 2.0], "y": np.arange(4)},
        name="scan",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        assert parent_tool.slicer_area.data.dims == ("x_idx", "y")

        def _nest_coarsen(dialog) -> None:
            assert "x_idx" not in dialog.dim_checks
            dialog.dim_checks["x"].setChecked(True)
            dialog.window_spins["x"].setValue(2)
            dialog.boundary_combo.setCurrentText("trim")
            dialog.side_combo.setCurrentText("left")
            dialog.coord_func_combo.setCurrentText("mean")
            dialog.reducer_combo.setCurrentText("mean")
            set_transform_launch_mode(dialog, "nest")

        accept_dialog(parent_tool.mnb._coarsen, pre_call=_nest_coarsen)

        parent = manager._imagetool_wrappers[0]
        qtbot.wait_until(lambda: len(parent._childtool_indices) == 1, timeout=5000)

        child_uid = parent._childtool_indices[0]
        child_node = manager._child_node(child_uid)
        child_tool = manager.get_imagetool(child_uid)

        assert child_node.source_spec is not None
        assert child_node.source_spec.kind == "public_data"
        xr.testing.assert_identical(
            child_tool.slicer_area._data.rename(None),
            data.coarsen(x=2, boundary="trim", side="left", coord_func="mean")
            .mean()
            .rename(None),
        )

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)
        derivation = metadata_derivation_texts(manager)
        assert derivation[0] == "Start from current parent ImageTool data"
        assert len(derivation) == 2
        assert "Coarsen" in derivation[1]

        updated = data.copy(deep=True)
        updated.data = np.asarray(updated.data) * 2

        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated)

        qtbot.wait_until(lambda: child_node.source_state == "stale", timeout=5000)
        assert child_node._update_from_parent_source() is True
        xr.testing.assert_identical(
            child_tool.slicer_area._data.rename(None),
            updated.coarsen(x=2, boundary="trim", side="left", coord_func="mean")
            .mean()
            .rename(None),
        )


def test_manager_transform_launch_modes_refresh_nested_and_detached(
    qtbot,
    monkeypatch,
    accept_dialog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(60).reshape((3, 4, 5)).astype(float),
        dims=["x", "y", "z"],
        coords={"x": np.arange(3), "y": np.arange(4), "z": np.arange(5)},
        name="scan",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)

        def _nest_average(dialog) -> None:
            assert dialog.launch_mode == "nest"
            assert dialog.launch_mode_combo.toolTip()
            dialog.dim_checks["x"].setChecked(True)

        accept_dialog(parent_tool.mnb._average, pre_call=_nest_average)

        parent = manager._imagetool_wrappers[0]
        qtbot.wait_until(lambda: len(parent._childtool_indices) == 1, timeout=5000)

        child_uid = parent._childtool_indices[0]
        child_node = manager._child_node(child_uid)
        child_tool = manager.get_imagetool(child_uid)
        xr.testing.assert_identical(
            child_tool.slicer_area._data.rename(None),
            data.qsel.average("x").rename(None),
        )

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)
        details = metadata_detail_map(manager)
        assert details["Kind"] == "ImageTool"
        assert "Added" in details
        derivation = metadata_derivation_texts(manager)
        assert any("Average" in line for line in derivation)
        assert all("rename(" not in line for line in derivation)

        copied: list[str] = []
        monkeypatch.setattr(
            erlab.interactive.utils,
            "copy_to_clipboard",
            lambda text: copied.append(text) or text,
        )

        def _replace_average(dialog) -> None:
            dialog.dim_checks["y"].setChecked(True)
            set_transform_launch_mode(dialog, "replace")

        accept_dialog(child_tool.mnb._average, pre_call=_replace_average)

        transforms = [
            op
            for op in typing.cast(
                "erlab.interactive.imagetool.provenance.ToolProvenanceSpec",
                child_node.source_spec,
            ).operations
            if op.op == "average"
        ]
        assert [op.op for op in transforms] == ["average", "average"]
        xr.testing.assert_identical(
            child_tool.slicer_area._data.rename(None),
            data.qsel.average("x").qsel.average("y").rename(None),
        )

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)
        derivation = metadata_derivation_texts(manager)
        assert derivation[0] == "Start from current parent ImageTool data"
        assert len(derivation) == 3
        assert "Average" in derivation[1]
        assert "dims=" in derivation[1]
        assert "Average" in derivation[2]
        assert "dims=" in derivation[2]
        manager.metadata_derivation_list.setFocus()
        select_metadata_rows(manager, [0])
        qtbot.keyClick(
            manager.metadata_derivation_list,
            QtCore.Qt.Key.Key_C,
            QtCore.Qt.KeyboardModifier.ControlModifier,
        )
        assert copied == []

        select_metadata_rows(manager, [1, 2])
        qtbot.keyClick(
            manager.metadata_derivation_list,
            QtCore.Qt.Key.Key_C,
            QtCore.Qt.KeyboardModifier.ControlModifier,
        )
        selected_namespace = _exec_generated_code(
            copied[-1],
            {"derived": data.copy(deep=True)},
        )
        selected_result = selected_namespace["derived"]
        assert isinstance(selected_result, xr.DataArray)
        xr.testing.assert_identical(
            selected_result.rename(None),
            data.qsel.average("x").qsel.average("y").rename(None),
        )

        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        action_map(menu)["Copy Selected Code"].trigger()
        selected_namespace = _exec_generated_code(
            copied[-1],
            {"derived": data.copy(deep=True)},
        )
        selected_result = selected_namespace["derived"]
        assert isinstance(selected_result, xr.DataArray)
        xr.testing.assert_identical(
            selected_result.rename(None),
            data.qsel.average("x").qsel.average("y").rename(None),
        )

        monkeypatch.setattr(
            manager,
            "_prompt_replay_input_name",
            lambda _node: "source_data",
        )
        action_map(menu)["Copy Full Code"].trigger()
        full_namespace = _exec_generated_code(
            copied[-1],
            {"source_data": data.copy(deep=True)},
        )
        assert not erlab.interactive.imagetool.provenance.uses_default_replay_input(
            copied[-1]
        )
        full_result = full_namespace["derived"]
        assert isinstance(full_result, xr.DataArray)
        xr.testing.assert_identical(
            full_result.rename(None),
            data.qsel.average("x").qsel.average("y").rename(None),
        )
        assert ".rename(" not in copied[-1]

        manual = xr.DataArray(
            np.arange(5, dtype=float) + 100.0,
            dims=["z"],
            coords={"z": data["z"].values},
            name=child_tool.slicer_area._data.name,
        )
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(child_uid, manual)
        xr.testing.assert_identical(fetch(child_uid), manual)

        updated = data.copy(deep=True)
        updated.data = np.asarray(updated.data) * 2

        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated)

        qtbot.wait_until(lambda: child_node.source_state == "stale", timeout=5000)
        assert child_node._update_from_parent_source() is True
        xr.testing.assert_identical(
            child_tool.slicer_area._data.rename(None),
            updated.qsel.average("x").qsel.average("y").rename(None),
        )

        def _detach_average(dialog) -> None:
            dialog.dim_checks["x"].setChecked(True)
            set_transform_launch_mode(dialog, "detach")

        accept_dialog(parent_tool.mnb._average, pre_call=_detach_average)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        detached = manager._imagetool_wrappers[1]
        assert detached.source_spec is None
        assert detached.provenance_spec is not None
        detached_tool = manager.get_imagetool(1)
        detached_derivation_before = detached.provenance_spec.derivation_code()

        def _replace_detached_average(dialog) -> None:
            dialog.dim_checks["y"].setChecked(True)
            set_transform_launch_mode(dialog, "replace")

        accept_dialog(detached_tool.mnb._average, pre_call=_replace_detached_average)
        assert detached.source_spec is None
        assert detached.provenance_spec is not None
        detached_transforms = [
            op for op in detached.provenance_spec.operations if op.op == "average"
        ]
        assert [op.op for op in detached_transforms] == ["average", "average"]
        assert detached.provenance_spec.derivation_code() == (
            "derived = data\n"
            'derived = derived.qsel.average("x")\n'
            'derived = derived.qsel.average("y")'
        )
        assert detached.provenance_spec.derivation_code() != detached_derivation_before
        xr.testing.assert_identical(
            detached_tool.slicer_area._data.rename(None),
            updated.qsel.average("x").qsel.average("y").rename(None),
        )
        detached_before = detached_tool.slicer_area._data.copy(deep=True)

        manager.tree_view.clearSelection()
        select_tools(manager, [1])
        manager._update_info()
        detached_derivation = metadata_derivation_texts(manager)
        assert detached_derivation[0] == "Start from current parent ImageTool data"
        assert len(detached_derivation) == 3
        assert "Average" in detached_derivation[1]
        assert "Average" in detached_derivation[2]

        duplicated_detached_index = typing.cast("int", manager.duplicate_imagetool(1))
        duplicated_detached = manager._imagetool_wrappers[duplicated_detached_index]
        assert duplicated_detached.source_spec is None
        assert duplicated_detached.provenance_spec == detached.provenance_spec
        xr.testing.assert_identical(
            manager.get_imagetool(duplicated_detached_index).slicer_area._data.rename(
                None
            ),
            detached_tool.slicer_area._data.rename(None),
        )

        updated2 = data.copy(deep=True)
        updated2.data = np.asarray(updated2.data) * 3
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated2)

        qtbot.wait_until(lambda: child_node.source_state == "stale", timeout=5000)
        xr.testing.assert_identical(detached_tool.slicer_area._data, detached_before)

        manager.tree_view.clearSelection()
        manager._update_info()
        assert metadata_detail_map(manager) == {}
        assert metadata_derivation_texts(manager) == []
        assert manager._build_metadata_derivation_menu() is None

        select_tools(manager, [0])
        select_child_tool(manager, child_uid)
        manager._update_info()
        assert metadata_detail_map(manager) == {}
        assert metadata_derivation_texts(manager) == []
        assert manager._build_metadata_derivation_menu() is None


def test_manager_divide_by_coord_child_refresh_and_code(
    qtbot,
    accept_dialog,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
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

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)

        def _nest_divide(dialog) -> None:
            assert dialog.launch_mode == "nest"
            dialog.coord_combo.setCurrentText("mesh_current")

        accept_dialog(parent_tool.mnb._divide_by_coord, pre_call=_nest_divide)

        parent = manager._imagetool_wrappers[0]
        qtbot.wait_until(lambda: len(parent._childtool_indices) == 1, timeout=5000)
        child_uid = parent._childtool_indices[0]
        child_node = manager._child_node(child_uid)
        child_tool = manager.get_imagetool(child_uid)

        expected = (data / data.mesh_current).rename("scan_div_mesh_current")
        xr.testing.assert_identical(child_tool.slicer_area._data, expected)
        assert child_node.source_spec is not None
        operations = [
            op for op in child_node.source_spec.operations if op.op == "divide_by_coord"
        ]
        assert len(operations) == 1
        assert operations[0].coord_name == "mesh_current"

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)
        derivation = metadata_derivation_texts(manager)
        assert any("Divide by Coordinate" in line for line in derivation)

        copied: list[str] = []
        monkeypatch.setattr(
            erlab.interactive.utils,
            "copy_to_clipboard",
            lambda text: copied.append(text) or text,
        )
        monkeypatch.setattr(
            manager,
            "_prompt_replay_input_name",
            lambda _node: "source_data",
        )
        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        action_map(menu)["Copy Full Code"].trigger()
        assert not erlab.interactive.imagetool.provenance.uses_default_replay_input(
            copied[-1]
        )

        namespace = _exec_generated_code(
            copied[-1], {"source_data": data.copy(deep=True)}
        )
        xr.testing.assert_identical(
            namespace["derived"].rename(None), expected.rename(None)
        )

        updated = data.copy(deep=True)
        updated.data = np.asarray(updated.data) * 2
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated)

        qtbot.wait_until(lambda: child_node.source_state == "stale", timeout=5000)
        assert child_node._update_from_parent_source() is True
        xr.testing.assert_identical(
            child_tool.slicer_area._data.rename(None),
            (updated / updated.mesh_current).rename(None),
        )


def test_manager_affine_coord_child_refreshes_from_formula(
    qtbot,
    accept_dialog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(12, dtype=float).reshape((3, 4)),
        dims=["x", "y"],
        coords={"x": np.arange(3), "y": np.arange(4)},
        name="scan",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)

        def _nest_affine(dialog) -> None:
            assert dialog.launch_mode == "nest"
            dialog._coord_combo.setCurrentText("y")
            dialog.coord_widget.edit_mode_tabs.setCurrentIndex(1)
            dialog.coord_widget.scale_spin.setValue(2.0)
            dialog.coord_widget.offset_spin.setValue(0.5)

        accept_dialog(parent_tool.mnb._assign_coords, pre_call=_nest_affine)

        parent = manager._imagetool_wrappers[0]
        qtbot.wait_until(lambda: len(parent._childtool_indices) == 1, timeout=5000)
        child_uid = parent._childtool_indices[0]
        child_node = manager._child_node(child_uid)
        child_tool = manager.get_imagetool(child_uid)

        operation = erlab.interactive.imagetool.provenance.AffineCoordOperation(
            coord_name="y",
            scale=2.0,
            offset=0.5,
        )
        expected = operation.apply(data, parent_data=data).rename("scan")
        xr.testing.assert_identical(child_tool.slicer_area._data, expected)

        assert child_node.source_spec is not None
        operations = [
            op for op in child_node.source_spec.operations if op.op == "affine_coord"
        ]
        assert len(operations) == 1
        assert operations[0] == operation

        updated = data.assign_coords(y=np.arange(4, dtype=float) + 10.0)
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated)

        qtbot.wait_until(lambda: child_node.source_state == "stale", timeout=5000)
        assert child_node._update_from_parent_source() is True
        xr.testing.assert_identical(
            child_tool.slicer_area._data.rename(None),
            operation.apply(updated, parent_data=updated).rename(None),
        )


def test_manager_assign_attrs_child_refreshes_from_operation(
    qtbot,
    accept_dialog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(12, dtype=float).reshape((3, 4)),
        dims=["x", "y"],
        coords={"x": np.arange(3), "y": np.arange(4)},
        attrs={"source": "old"},
        name="scan",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)

        def _nest_attrs(dialog) -> None:
            assert dialog.launch_mode == "nest"
            source_row = next(
                row
                for row in range(dialog.table.rowCount())
                if dialog._row_key(row) == "source"
            )
            dialog.table.item(source_row, 2).setText("new")
            dialog._add_empty_row()
            flag_row = dialog.table.rowCount() - 1
            dialog.table.item(flag_row, 0).setText("flag")
            typing.cast(
                "QtWidgets.QComboBox", dialog.table.cellWidget(flag_row, 1)
            ).setCurrentText("Bool")
            dialog.table.item(flag_row, 2).setText("True")

        accept_dialog(parent_tool.mnb._assign_attrs, pre_call=_nest_attrs)

        parent = manager._imagetool_wrappers[0]
        qtbot.wait_until(lambda: len(parent._childtool_indices) == 1, timeout=5000)
        child_uid = parent._childtool_indices[0]
        child_node = manager._child_node(child_uid)
        child_tool = manager.get_imagetool(child_uid)

        operation = erlab.interactive.imagetool.provenance.AssignAttrsOperation(
            attrs={"source": "new", "flag": True}
        )
        expected = operation.apply(data, parent_data=data).rename("scan")
        xr.testing.assert_identical(child_tool.slicer_area._data, expected)

        assert child_node.source_spec is not None
        operations = [
            op for op in child_node.source_spec.operations if op.op == "assign_attrs"
        ]
        assert operations == [operation]

        updated = data.assign_attrs(source="updated", count=2)
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated)

        qtbot.wait_until(lambda: child_node.source_state == "stale", timeout=5000)
        assert child_node._update_from_parent_source() is True
        xr.testing.assert_identical(
            child_tool.slicer_area._data.rename(None),
            operation.apply(updated, parent_data=updated).rename(None),
        )


def test_wrapper_source_data_replaced_uses_parent_fallback_and_skips_missing_child(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 1, timeout=5000
        )

        wrapper = manager._imagetool_wrappers[0]
        _, child = next(iter(wrapper._childtools.items()))
        updated = test_data.copy(deep=True)
        updated.data = np.asarray(updated.data) * 7
        handled: list[xr.DataArray] = []

        monkeypatch.setattr(
            wrapper.slicer_area, "_tool_source_parent_data", lambda: updated
        )
        monkeypatch.setattr(
            child, "handle_parent_source_replaced", lambda data: handled.append(data)
        )
        wrapper._childtool_indices.append("missing")

        wrapper._handle_source_data_replaced(object())

        assert handled == [updated]


def test_manager_reindex(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        # Open a tool with the manager
        itool([test_data, test_data, test_data], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 3)

        assert manager._displayed_indices == [0, 1, 2]

        # Remove tool at index 1
        manager.remove_imagetool(1)
        qtbot.wait_until(lambda: manager.ntools == 2)

        assert manager._displayed_indices == [0, 2]

        # Reindex
        manager.reindex_action.trigger()
        qtbot.wait_until(lambda: manager._displayed_indices == [0, 1], timeout=5000)


def test_manager_server_show_remove(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        # Open a tool with the manager
        itool([test_data, test_data], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        # Show tool at index 0
        _show_idx(0)

        # Remove tool at index 0
        _remove_idx(0)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)


def test_manager_data_watched_update_replaces_existing_tool_source_data(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        manager._data_recv([test_data], {}, watched_var=("data", "kernel-0"))
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        tool = manager.get_imagetool(0)
        updated = test_data.copy(deep=True)
        updated.data = np.asarray(updated.data) * 11

        with qtbot.wait_signal(tool.slicer_area.sigSourceDataReplaced):
            manager._data_watched_update("data", "kernel-0", updated)

        xr.testing.assert_identical(tool.slicer_area.data, updated)


def test_manager_workspace_roundtrip_preserves_watched_binding(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        manager._data_recv(
            [test_data],
            {},
            watched_var=("data", "watch:stable-data"),
            watched_metadata={
                "workspace_link_id": manager._workspace_link_id,
                "source_label": "notebook-a",
                "connected": True,
            },
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        workspace_link_id = manager._workspace_link_id
        tree = manager._to_datatree()
        tree.attrs.update(manager._workspace_root_attrs_payload(delta_save_count=0))
        manifest = json.loads(tree.attrs["imagetool_workspace_manifest"])
        assert manifest["workspace_link_id"] == workspace_link_id
        attrs = tree["0/imagetool"].attrs
        assert attrs["manager_node_watched_varname"] == "data"
        assert attrs["manager_node_watched_uid"] == "watch:stable-data"
        assert attrs["manager_node_watched_workspace_link_id"] == workspace_link_id
        assert attrs["manager_node_watched_source_label"] == "notebook-a"

        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)
        manager._workspace_link_id = "different-workspace-link"

        manager._load_workspace_node(typing.cast("xr.DataTree", tree["0"]))
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        wrapper = manager._imagetool_wrappers[0]
        assert wrapper.watched
        assert wrapper._watched_varname == "data"
        assert wrapper._watched_uid == "watch:stable-data"
        assert wrapper._watched_workspace_link_id == workspace_link_id
        assert wrapper._watched_source_label == "notebook-a"
        assert wrapper._watched_connected is False

        with qtbot.wait_signal(manager._sigReplyData) as blocker:
            manager._send_watch_info()
        assert blocker.args[0]["workspace_link_id"] == "different-workspace-link"
        assert blocker.args[0]["watched"][0]["workspace_link_id"] == workspace_link_id
        assert blocker.args[0]["watched"][0]["source_label"] == "notebook-a"

        manager._from_datatree(tree, replace=True, select=False)
        assert manager._workspace_link_id == workspace_link_id


def test_manager_workspace_watched_attrs_skip_missing_workspace_link(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(test_data, _in_manager=True),
            watched_var=("data", "watch:stable-data"),
            watched_workspace_link_id=None,
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        attrs = manager._to_datatree()["0/imagetool"].attrs
        assert attrs["manager_node_watched_varname"] == "data"
        assert attrs["manager_node_watched_uid"] == "watch:stable-data"
        assert "manager_node_watched_workspace_link_id" not in attrs


def test_manager_watched_root_provenance_uses_variable_name(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        manager._data_recv([test_data], {}, watched_var=("my_data", "kernel-0"))
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        node = manager._imagetool_wrappers[0]
        provenance = node.provenance_spec
        assert provenance is not None
        code = provenance.display_code()
        assert code is not None
        namespace = _exec_generated_code(code, {"my_data": test_data.copy(deep=True)})
        derived = namespace["derived"]
        assert isinstance(derived, xr.DataArray)
        xr.testing.assert_identical(derived, manager.get_imagetool(0).slicer_area.data)
        assert provenance.display_entries()[0].label == (
            "Start from watched variable 'my_data'"
        )

        copied: list[str] = []
        monkeypatch.setattr(
            erlab.interactive.utils,
            "copy_to_clipboard",
            lambda text: copied.append(text) or text,
        )
        monkeypatch.setattr(
            manager,
            "_prompt_replay_input_name",
            lambda _node: pytest.fail("watched roots should not prompt"),
        )
        manager.tree_view.clearSelection()
        select_tools(manager, [0])
        manager._update_info(uid=node.uid)
        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        action_map(menu)["Copy Full Code"].trigger()
        namespace = _exec_generated_code(
            copied[-1],
            {"my_data": test_data.copy(deep=True)},
        )
        derived = namespace["derived"]
        assert isinstance(derived, xr.DataArray)
        xr.testing.assert_identical(derived, manager.get_imagetool(0).slicer_area.data)


def test_manager_non_watched_full_code_prompts_for_source_variable(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        manager._data_recv([test_data], {})
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        node = manager._imagetool_wrappers[0]
        node.set_detached_provenance(
            erlab.interactive.imagetool.provenance.full_data(
                erlab.interactive.imagetool.provenance.AverageOperation(dims=("alpha",))
            )
        )

        copied: list[str] = []
        prompted: list[str] = []
        monkeypatch.setattr(
            erlab.interactive.utils,
            "copy_to_clipboard",
            lambda text: copied.append(text) or text,
        )
        monkeypatch.setattr(
            manager,
            "_prompt_replay_input_name",
            lambda prompt_node: prompted.append(prompt_node.uid) or "source_data",
        )
        manager.tree_view.clearSelection()
        select_tools(manager, [0])
        manager._update_info(uid=node.uid)
        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        action_map(menu)["Copy Full Code"].trigger()

        assert prompted == [node.uid]
        assert copied
        assert not erlab.interactive.imagetool.provenance.uses_default_replay_input(
            copied[-1]
        )
        namespace = _exec_generated_code(
            copied[-1], {"source_data": test_data.copy(deep=True)}
        )
        xr.testing.assert_identical(
            namespace["derived"], test_data.qsel.average("alpha")
        )


def test_manager_non_watched_full_code_prompt_cancel_does_not_copy(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        manager._data_recv([test_data], {})
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        node = manager._imagetool_wrappers[0]
        node.set_detached_provenance(
            erlab.interactive.imagetool.provenance.full_data(
                erlab.interactive.imagetool.provenance.AverageOperation(dims=("alpha",))
            )
        )

        copied: list[str] = []
        monkeypatch.setattr(
            erlab.interactive.utils,
            "copy_to_clipboard",
            lambda text: copied.append(text) or text,
        )
        monkeypatch.setattr(manager, "_prompt_replay_input_name", lambda _node: None)
        manager.tree_view.clearSelection()
        select_tools(manager, [0])
        manager._update_info(uid=node.uid)
        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        action_map(menu)["Copy Full Code"].trigger()

        assert copied == []


def test_manager_file_backed_full_code_uses_load_code(
    qtbot,
    monkeypatch,
    tmp_path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    file_path = tmp_path / "scan.h5"
    test_data.to_netcdf(file_path, engine="h5netcdf")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(
            test_data,
            manager=True,
            file_path=file_path,
            load_func=(xr.load_dataarray, {"engine": "h5netcdf"}, 0),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        node = manager._imagetool_wrappers[0]
        node.set_detached_provenance(
            erlab.interactive.imagetool.provenance.full_data(
                erlab.interactive.imagetool.provenance.AverageOperation(dims=("alpha",))
            )
        )

        copied: list[str] = []
        monkeypatch.setattr(
            erlab.interactive.utils,
            "copy_to_clipboard",
            lambda text: copied.append(text) or text,
        )
        monkeypatch.setattr(
            manager,
            "_prompt_replay_input_name",
            lambda _node: pytest.fail("file-backed replay should not prompt"),
        )
        manager.tree_view.clearSelection()
        select_tools(manager, [0])
        manager._update_info(uid=node.uid)
        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        action_map(menu)["Copy Full Code"].trigger()

        assert copied
        assert not erlab.interactive.imagetool.provenance.uses_default_replay_input(
            copied[-1]
        )
        namespace = _exec_generated_code(copied[-1], {})
        xr.testing.assert_identical(
            namespace["derived"], test_data.qsel.average("alpha")
        )


def test_manager_file_backed_full_code_prefers_scan_number_loader(
    qtbot,
    monkeypatch,
    example_loader,
    example_data_dir: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    file_path = example_data_dir / "data_002.h5"
    data = erlab.io.loaders["example"].load(file_path)

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(
            data,
            manager=True,
            file_path=file_path,
            load_func=("example", {}, 0),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        node = manager._imagetool_wrappers[0]
        node.set_detached_provenance(
            erlab.interactive.imagetool.provenance.full_data(
                erlab.interactive.imagetool.provenance.AverageOperation(dims=("alpha",))
            )
        )

        copied: list[str] = []
        monkeypatch.setattr(
            erlab.interactive.utils,
            "copy_to_clipboard",
            lambda text: copied.append(text) or text,
        )
        monkeypatch.setattr(
            manager,
            "_prompt_replay_input_name",
            lambda _node: pytest.fail("file-backed replay should not prompt"),
        )
        manager.tree_view.clearSelection()
        select_tools(manager, [0])
        manager._update_info(uid=node.uid)
        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        action_map(menu)["Copy Full Code"].trigger()

        assert copied
        assert f"erlab.io.load(2, data_dir={str(example_data_dir)!r})" in copied[-1]
        namespace = _exec_generated_code(copied[-1], {})
        xr.testing.assert_identical(namespace["derived"], data.qsel.average("alpha"))


def test_manager_watched_root_child_tool_copy_code_uses_variable_name(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        manager._data_recv([test_data], {}, watched_var=("my_data", "kernel-0"))
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtool_indices) == 1,
            timeout=5000,
        )

        child_uid = manager._imagetool_wrappers[0]._childtool_indices[0]
        copied = copy_full_code_for_uid(monkeypatch, manager, child_uid)
        namespace = _exec_generated_code(
            copied,
            {"my_data": test_data.copy(deep=True)},
        )
        result = namespace["result"]
        assert isinstance(result, xr.DataArray)
        child_tool = manager.get_childtool(child_uid)
        assert isinstance(child_tool, DerivativeTool)
        xr.testing.assert_identical(result, child_tool.result)


def test_manager_watched_root_child_imagetool_copy_code_uses_variable_name(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        manager._data_recv([test_data], {}, watched_var=("my_data", "kernel-0"))
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_new_window()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtool_indices) == 1,
            timeout=5000,
        )

        child_uid = manager._imagetool_wrappers[0]._childtool_indices[0]
        copied = copy_full_code_for_uid(monkeypatch, manager, child_uid)
        namespace = _exec_generated_code(
            copied,
            {"my_data": test_data.copy(deep=True)},
        )
        derived = namespace["derived"]
        assert isinstance(derived, xr.DataArray)
        xr.testing.assert_identical(derived, fetch(child_uid))


def test_manager_watched_root_ftool_copy_code_1d_omits_duplicate_seed_and_noop_squeeze(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        manager._data_recv([test_data], {}, watched_var=("my_data", "kernel-0"))
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_ftool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtool_indices) == 1,
            timeout=5000,
        )

        child_uid = manager._imagetool_wrappers[0]._childtool_indices[0]
        child_tool = manager.get_childtool(child_uid)
        assert isinstance(child_tool, Fit2DTool)

        copied: list[str] = []
        monkeypatch.setattr(
            erlab.interactive.utils,
            "copy_to_clipboard",
            lambda text: copied.append(text) or text,
        )
        child_tool.copy_code_1d()

        assert copied
        _assert_modelfit_code_replays_source(copied[-1], "my_data", test_data)


def test_manager_selecting_unfit_ftool_child_does_not_warn(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        manager._data_recv([test_data], {}, watched_var=("my_data", "kernel-0"))
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_ftool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtool_indices) == 1,
            timeout=5000,
        )

        child_uid = manager._imagetool_wrappers[0]._childtool_indices[0]
        child_tool = manager.get_childtool(child_uid)
        assert isinstance(child_tool, Fit2DTool)

        warnings: list[tuple[str, str]] = []
        monkeypatch.setattr(
            child_tool,
            "_show_warning",
            lambda title, text: warnings.append((title, text)),
        )

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)

        qtbot.wait_until(
            lambda: "ftool_2d" in manager.text_box.toHtml().lower(), timeout=5000
        )
        assert "ftool_2d" in manager.text_box.toPlainText().lower()
        assert manager.preview_widget.isVisible()
        assert not manager_preview_pixmap(manager).isNull()
        assert metadata_detail_map(manager)["Kind"] == "ftool_2d"
        assert "modelfit" not in (manager._metadata_full_code or "")
        assert not warnings


def test_manager_fit1d_child_side_panel(
    qtbot,
    exp_decay_model,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child_uid, _ = make_fit1d_child(manager, 0, exp_decay_model)

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)

        qtbot.wait_until(
            lambda: "ftool_1d" in manager.text_box.toHtml().lower(), timeout=5000
        )
        assert "ftool_1d" in manager.text_box.toPlainText().lower()
        assert "not fit yet" in manager.text_box.toPlainText().lower()
        assert not manager.preview_widget.isVisible()
        assert metadata_detail_map(manager)["Kind"] == "ftool_1d"


def test_manager_fit2d_child_side_panel_live_refresh(
    qtbot,
    exp_decay_model,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child_uid, child = make_fit2d_child(manager, 0, exp_decay_model)

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)

        qtbot.wait_until(
            lambda: "ftool_2d" in manager.text_box.toHtml().lower(), timeout=5000
        )
        old_html = manager.text_box.toHtml()
        new_index = (
            child.y_min_spin.value()
            if child._current_idx != child.y_min_spin.value()
            else child.y_max_spin.value()
        )
        child.y_index_spin.setValue(new_index)

        qtbot.wait_until(lambda: manager.text_box.toHtml() != old_html, timeout=5000)
        assert f"index {new_index}" in manager.text_box.toPlainText().lower()


def test_manager_goldtool_child_side_panel(
    qtbot,
    gold,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(gold, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child = GoldTool(gold.copy(deep=True), data_name="gold_input")
        child_uid = manager.add_childtool(child, 0, show=False)
        configure_goldtool_child(child, fitted=True, spline=True)

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)

        qtbot.wait_until(
            lambda: "goldtool" in manager.text_box.toHtml().lower(), timeout=5000
        )
        assert "goldtool" in manager.text_box.toPlainText().lower()
        assert manager.preview_widget.isVisible()
        assert not manager_preview_pixmap(manager).isNull()
        assert metadata_detail_map(manager)["Kind"] == "goldtool"


def test_manager_goldtool_child_side_panel_live_refresh(
    qtbot,
    gold,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(gold, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child = GoldTool(gold.copy(deep=True), data_name="gold_input")
        child_uid = manager.add_childtool(child, 0, show=False)
        configure_goldtool_child(child, fitted=True, spline=False)

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)

        qtbot.wait_until(
            lambda: "goldtool" in manager.text_box.toHtml().lower(), timeout=5000
        )
        old_html = manager.text_box.toHtml()
        child.params_tab.setCurrentIndex(1)

        qtbot.wait_until(lambda: manager.text_box.toHtml() != old_html, timeout=5000)
        assert "spline" in manager.text_box.toPlainText().lower()


def test_manager_restool_child_side_panel(
    qtbot,
    gold,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(gold, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child_uid = manager.add_childtool(
            ResolutionTool(gold.copy(deep=True), data_name="gold_input"),
            0,
            show=False,
        )

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)

        qtbot.wait_until(
            lambda: "restool" in manager.text_box.toHtml().lower(), timeout=5000
        )
        assert "restool" in manager.text_box.toPlainText().lower()
        assert manager.preview_widget.isVisible()
        assert not manager_preview_pixmap(manager).isNull()
        assert metadata_detail_map(manager)["Kind"] == "restool"


def test_manager_restool_child_side_panel_live_refresh(
    qtbot,
    gold,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(gold, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child_uid = manager.add_childtool(
            ResolutionTool(gold.copy(deep=True), data_name="gold_input"),
            0,
            show=False,
        )
        child = manager.get_childtool(child_uid)
        assert isinstance(child, ResolutionTool)

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)

        qtbot.wait_until(
            lambda: "restool" in manager.text_box.toHtml().lower(), timeout=5000
        )
        old_html = manager.text_box.toHtml()
        step = max(child.x0_spin.singleStep(), 10**-child._x_decimals)
        new_value = min(child.x0_spin.value() + step, child.x1_spin.value())
        if new_value == child.x0_spin.value():
            new_value = max(child._x_range[0], child.x0_spin.value() - step)
        child.x0_spin.setValue(new_value)

        qtbot.wait_until(lambda: manager.text_box.toHtml() != old_html, timeout=5000)


def test_manager_meshtool_child_side_panel(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child_uid = manager.add_childtool(
            MeshTool(test_data.copy(deep=True), data_name="mesh_input"),
            0,
            show=False,
        )

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)

        qtbot.wait_until(
            lambda: "meshtool" in manager.text_box.toHtml().lower(), timeout=5000
        )
        assert "meshtool" in manager.text_box.toPlainText().lower()
        assert manager.preview_widget.isVisible()
        assert not manager_preview_pixmap(manager).isNull()
        assert metadata_detail_map(manager)["Kind"] == "meshtool"


def test_manager_meshtool_child_side_panel_live_refresh(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child_uid = manager.add_childtool(
            MeshTool(test_data.copy(deep=True), data_name="mesh_input"),
            0,
            show=False,
        )
        child = manager.get_childtool(child_uid)
        assert isinstance(child, MeshTool)

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)

        qtbot.wait_until(
            lambda: "meshtool" in manager.text_box.toHtml().lower(), timeout=5000
        )
        old_html = manager.text_box.toHtml()
        child.order_spin.setValue(child.order_spin.value() + 1)

        qtbot.wait_until(lambda: manager.text_box.toHtml() != old_html, timeout=5000)


def test_manager_watched_1d_root_ftool_copy_code_omits_synthetic_squeeze(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(np.arange(5), dims=("x",), coords={"x": np.arange(5)})

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        manager._data_recv([data], {}, watched_var=("my_1d", "kernel-0"))
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_ftool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtool_indices) == 1,
            timeout=5000,
        )

        child_uid = manager._imagetool_wrappers[0]._childtool_indices[0]
        child_tool = manager.get_childtool(child_uid)

        copied: list[str] = []
        monkeypatch.setattr(
            erlab.interactive.utils,
            "copy_to_clipboard",
            lambda text: copied.append(text) or text,
        )
        child_tool.copy_code()

        assert copied
        _assert_modelfit_code_replays_source(copied[-1], "my_1d", data)


def test_manager_watched_update_to_1d_refreshes_copy_code_cleanup(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    updated = xr.DataArray(np.arange(5), dims=("x",), coords={"x": np.arange(5)})

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        manager._data_recv([test_data], {}, watched_var=("my_data", "kernel-0"))
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        with qtbot.wait_signal(parent_tool.slicer_area.sigSourceDataReplaced):
            manager._data_watched_update("my_data", "kernel-0", updated)

        parent_tool.slicer_area.images[0].open_in_ftool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtool_indices) == 1,
            timeout=5000,
        )

        child_uid = manager._imagetool_wrappers[0]._childtool_indices[0]
        child_tool = manager.get_childtool(child_uid)

        copied: list[str] = []
        monkeypatch.setattr(
            erlab.interactive.utils,
            "copy_to_clipboard",
            lambda text: copied.append(text) or text,
        )
        child_tool.copy_code()

        assert copied
        _assert_modelfit_code_replays_source(copied[-1], "my_data", updated)


def test_manager_duplicate_watched_1d_root_preserves_copy_code_cleanup(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(np.arange(5), dims=("x",), coords={"x": np.arange(5)})

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        manager._data_recv([data], {}, watched_var=("my_1d", "kernel-0"))
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        duplicated = manager.duplicate_imagetool(0)
        assert isinstance(duplicated, int)

        parent_tool = manager.get_imagetool(duplicated)
        parent_tool.slicer_area.images[0].open_in_ftool()
        qtbot.wait_until(
            lambda: (
                len(manager._imagetool_wrappers[duplicated]._childtool_indices) == 1
            ),
            timeout=5000,
        )

        child_uid = manager._imagetool_wrappers[duplicated]._childtool_indices[0]
        child_tool = manager.get_childtool(child_uid)

        copied: list[str] = []
        monkeypatch.setattr(
            erlab.interactive.utils,
            "copy_to_clipboard",
            lambda text: copied.append(text) or text,
        )
        child_tool.copy_code()

        assert copied
        _assert_modelfit_code_replays_source(copied[-1], "my_1d", data)


def test_manager_workspace_roundtrip_watched_1d_root_preserves_copy_code_cleanup(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(np.arange(5), dims=("x",), coords={"x": np.arange(5)})

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        manager._data_recv([data], {}, watched_var=("my_1d", "kernel-0"))
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        tree = manager._to_datatree()
        assert tree["0/imagetool"].attrs["manager_node_source_input_ndim"] == 1

        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        for node in tree.values():
            manager._load_workspace_node(typing.cast("xr.DataTree", node))

        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_ftool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtool_indices) == 1,
            timeout=5000,
        )

        child_uid = manager._imagetool_wrappers[0]._childtool_indices[0]
        child_tool = manager.get_childtool(child_uid)

        copied: list[str] = []
        monkeypatch.setattr(
            erlab.interactive.utils,
            "copy_to_clipboard",
            lambda text: copied.append(text) or text,
        )
        child_tool.copy_code()

        assert copied
        _assert_modelfit_code_replays_source(copied[-1], "my_1d", data)


def test_manager_duplicate(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        # Open a tool with the manager
        itool([test_data, test_data], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        select_tools(manager, [0, 1])
        manager.duplicate_selected()
        qtbot.wait_until(lambda: manager.ntools == 4, timeout=5000)

        # Check if the duplicated tools have the same data
        for i in range(2):
            original_tool = manager.get_imagetool(i)
            duplicated_tool = manager.get_imagetool(i + 2)

            assert original_tool.slicer_area._data.equals(
                duplicated_tool.slicer_area._data
            )
            assert (
                manager._imagetool_wrappers[i].name
                == manager._imagetool_wrappers[i + 2].name
            )


def test_manager_duplicate_goldtool_child(
    qtbot,
    monkeypatch,
    gold,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()

        itool(gold, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child = GoldTool(gold.copy(deep=True), data_name="gold_input")
        child_uid = manager.add_childtool(child, 0, show=False)
        configure_goldtool_child(child, fitted=True, spline=True)
        child.open_itool()

        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)

        select_child_tool(manager, child_uid)
        manager.duplicate_selected()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 2,
            timeout=5000,
        )

        duplicate_uid = next(
            uid
            for uid in manager._imagetool_wrappers[0]._childtool_indices
            if uid != child_uid
        )
        duplicated = manager.get_childtool(duplicate_uid)

        assert isinstance(duplicated, GoldTool)
        assert duplicated is not child
        assert duplicated.tool_status == child.tool_status
        xr.testing.assert_identical(duplicated.corrected, child.corrected)

        duplicate_node = manager._child_node(duplicate_uid)
        qtbot.wait_until(
            lambda: len(duplicate_node._childtool_indices) == 1, timeout=5000
        )
        duplicate_output_uid = duplicate_node._childtool_indices[0]
        duplicate_output_node = manager._child_node(duplicate_output_uid)
        assert duplicate_output_node.output_id == "goldtool.corrected"
        assert duplicate_output_node.source_spec is None
        assert duplicate_output_node.provenance_spec is not None
        xr.testing.assert_identical(fetch(duplicate_output_uid), duplicated.corrected)

        monkeypatch.setattr(
            duplicated, "_prompt_existing_output_imagetool", lambda: "update"
        )
        duplicated.open_itool()
        assert duplicate_node._childtool_indices == [duplicate_output_uid]


def test_manager_sync(
    qtbot,
    move_and_compare_values,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool([test_data, test_data], link=True, link_colors=True, manager=True)

        qtbot.wait_until(lambda: manager.ntools == 2)

        win0, win1 = manager.get_imagetool(0), manager.get_imagetool(1)
        win0.show()
        win1.show()

        win1.slicer_area.set_colormap("RdYlBu", gamma=1.5)
        assert (
            win0.slicer_area._colormap_properties
            == win1.slicer_area._colormap_properties
        )

        move_and_compare_values(qtbot, win0, [12.0, 7.0, 6.0, 11.0], target_win=win1)

        # Transpose
        win0.slicer_area.transpose_main_image()
        move_and_compare_values(qtbot, win0, [12.0, 11.0, 6.0, 7.0], target_win=win1)

        # Set bin
        win1.slicer_area.set_bin(0, 2, update=False)
        win1.slicer_area.set_bin(1, 2, update=True)

        # Set all bins, same effect as above since we only have 1 cursor
        win1.slicer_area.set_bin_all(1, 2, update=True)

        move_and_compare_values(qtbot, win0, [9.0, 8.0, 3.0, 4.0], target_win=win1)

        # Change limits
        win0.slicer_area.main_image.getViewBox().setRange(xRange=[2, 3], yRange=[1, 2])
        # Trigger manual range propagation
        win0.slicer_area.main_image.getViewBox().sigRangeChangedManually.emit(
            win0.slicer_area.main_image.getViewBox().state["mouseEnabled"][:]
        )
        assert win1.slicer_area.main_image.getViewBox().viewRange() == [[2, 3], [1, 2]]


def test_manager_workspace_io(
    qtbot,
    accept_dialog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()

        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        # Add two tools
        itool([data, data], link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        # Open dtool for first tool
        manager.get_imagetool(0).slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 1, timeout=5000
        )

        # Save and load workspace
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            filename = f"{tmp_dir_name}/workspace.itws"

            def _go_to_file(dialog: QtWidgets.QFileDialog):
                dialog.setDirectory(tmp_dir_name)
                dialog.selectFile(filename)
                focused = dialog.focusWidget()
                if isinstance(focused, QtWidgets.QLineEdit):
                    focused.setText("workspace.itws")

            # Save workspace
            accept_dialog(
                lambda: manager.save(native=False),
                pre_call=_go_to_file,
            )
            assert manager.workspace_path == str(pathlib.Path(filename).resolve())
            assert not manager.is_workspace_modified

            # Load workspace
            accept_dialog(lambda: manager.load(native=False), pre_call=_go_to_file)

            # Check if the data is loaded
            assert manager.ntools == 2

            # Check if the child dtool is also loaded
            assert len(manager._imagetool_wrappers[0]._childtools) == 1

            select_tools(manager, list(manager._imagetool_wrappers.keys()))
            accept_dialog(manager.remove_action.trigger)
            qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)


def test_manager_workspace_save_selection_cancel_does_not_write(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    class _RejectedChooseDialog:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def exec(self) -> QtWidgets.QDialog.DialogCode:
            return QtWidgets.QDialog.DialogCode.Rejected

    monkeypatch.setattr(
        erlab.interactive.imagetool.manager._mainwindow,
        "_ChooseFromDataTreeDialog",
        _RejectedChooseDialog,
    )
    closed_trees: list[xr.DataTree] = []
    original_close = xr.DataTree.close

    def _close_spy(tree: xr.DataTree) -> None:
        closed_trees.append(tree)
        original_close(tree)

    monkeypatch.setattr(xr.DataTree, "close", _close_spy)

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])
        itool(data, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            filename = pathlib.Path(tmp_dir_name) / "workspace.itws"
            manager._save_to_file(str(filename))

            assert not filename.exists()
            assert len(closed_trees) == 1


def test_manager_workspace_load_selection_skips_unchecked_children(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    class _SelectedChooseDialog(
        erlab.interactive.imagetool.manager._mainwindow._ChooseFromDataTreeDialog
    ):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            root_item = self._tree_widget.topLevelItem(0)
            assert root_item is not None
            unchecked_child = root_item.child(1)
            assert unchecked_child is not None
            unchecked_child.setCheckState(0, QtCore.Qt.CheckState.Unchecked)

        def exec(self) -> QtWidgets.QDialog.DialogCode:
            return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(
        erlab.interactive.imagetool.manager._mainwindow,
        "_ChooseFromDataTreeDialog",
        _SelectedChooseDialog,
    )

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root_tool = itool(data, manager=False, execute=False)
        assert isinstance(root_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root_tool, show=False)

        child_uids: list[str] = []
        for offset in (1.0, 2.0):
            child_tool = itool(data + offset, manager=False, execute=False)
            assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
            child_uids.append(manager.add_imagetool_child(child_tool, 0, show=False))

        tree = manager._to_datatree()
        try:
            manager.remove_all_tools()
            qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

            manager._from_datatree(tree)
        finally:
            tree.close()

        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        assert manager._imagetool_wrappers[0]._childtool_indices == [child_uids[0]]
        assert child_uids[1] not in manager._all_nodes


def _open_external_hdf5_imagetool_data(
    fname: pathlib.Path, *, chunks: str | None = None
) -> xr.DataArray:
    open_kwargs: dict[str, typing.Any] = {
        "engine": "h5netcdf",
        "phony_dims": "sort",
    }
    if chunks is not None:
        open_kwargs["chunks"] = chunks
    tree = xr.open_datatree(fname, **open_kwargs)
    try:
        ds = typing.cast("xr.DataTree", tree["0/imagetool"]).to_dataset(inherit=False)
        return ds[next(iter(ds.data_vars))]
    finally:
        tree.close()


def _open_external_lazy_hdf5_imagetool_data(fname: pathlib.Path) -> xr.DataArray:
    return _open_external_hdf5_imagetool_data(fname, chunks="auto")


def _open_external_file_backed_hdf5_imagetool_data(
    fname: pathlib.Path,
) -> xr.DataArray:
    return _open_external_hdf5_imagetool_data(fname)


def _compute_first_value(darr: xr.DataArray) -> object:
    return darr.isel(dict.fromkeys(darr.dims, 0)).compute().item()


def _hdf5_filter_ids(dataset) -> list[int]:
    create_plist = dataset.id.get_create_plist()
    return [create_plist.get_filter(i)[0] for i in range(create_plist.get_nfilters())]


def _transaction_test_root_attrs(delta_save_count: int = 0) -> dict[str, object]:
    manifest: dict[str, object] = {
        "schema_version": 4,
        "root_order": [0],
        "nodes": [],
    }
    if delta_save_count > 0:
        manifest["transaction_protocol"] = (
            manager_workspace._WORKSPACE_TRANSACTION_PROTOCOL
        )
        manifest["delta_save_count"] = delta_save_count
    return {
        "imagetool_workspace_schema_version": 4,
        manager_workspace._WORKSPACE_MANIFEST_ATTR: json.dumps(manifest),
    }


def _transaction_test_dataset(value: float, *, title: str) -> xr.Dataset:
    ds = xr.Dataset({"data": ("x", np.array([value], dtype=np.float64))})
    ds.attrs["itool_title"] = title
    return ds


def _write_transaction_test_workspace(fname: pathlib.Path, value: float = 1.0) -> None:
    tree = xr.DataTree.from_dict(
        {"0/imagetool": _transaction_test_dataset(value, title="old")}
    )
    try:
        manager_workspace._write_full_workspace_tree_file(
            fname, tree, _transaction_test_root_attrs()
        )
    finally:
        tree.close()


def _read_transaction_test_value(fname: pathlib.Path) -> float:
    opened = manager_xarray.open_workspace_datatree(fname, chunks=None)
    try:
        ds = typing.cast("xr.DataTree", opened["/0/imagetool"]).to_dataset(
            inherit=False
        )
        return float(ds["data"].item())
    finally:
        opened.close()


def _assert_no_workspace_internal_groups(fname: pathlib.Path) -> None:
    import h5py

    with h5py.File(fname, "r") as h5_file:
        assert not any(
            name.startswith(manager_workspace._WORKSPACE_INTERNAL_GROUP_PREFIXES)
            for name in h5_file
        )


def test_workspace_dataset_encoding_compresses_only_large_numeric_payloads() -> None:
    import hdf5plugin

    ds = xr.Dataset(
        {
            "large": (
                ("x", "y"),
                np.arange(512 * 512, dtype=np.float64).reshape(512, 512),
            ),
            "small": ("x", np.arange(512, dtype=np.float64)),
            "metadata": ("label", np.array(["a", "b"], dtype=object)),
        },
        coords={
            "x": np.linspace(-1.0, 1.0, 512),
            "y": np.linspace(-2.0, 2.0, 512),
            "label": ["a", "b"],
        },
    )

    encoding = manager_xarray.workspace_dataset_encoding(ds)

    assert set(encoding) == {"large"}
    assert encoding["large"] == dict(
        hdf5plugin.Blosc2(
            cname="blosclz",
            clevel=3,
            filters=hdf5plugin.Blosc2.SHUFFLE,
        )
    )


def test_workspace_dataset_encoding_respects_compression_preference() -> None:
    ds = xr.Dataset(
        {
            "large": (
                ("x", "y"),
                np.arange(512 * 512, dtype=np.float64).reshape(512, 512),
            )
        }
    )
    old_value = erlab.interactive.options["io/workspace/compress"]
    try:
        erlab.interactive.options["io/workspace/compress"] = False
        assert manager_xarray.workspace_dataset_encoding(ds) == {}

        erlab.interactive.options["io/workspace/compress"] = True
        assert set(manager_xarray.workspace_dataset_encoding(ds)) == {"large"}
    finally:
        erlab.interactive.options["io/workspace/compress"] = old_value


def test_workspace_datatree_encoding_uses_group_paths() -> None:
    large_ds = xr.Dataset(
        {
            "data": (
                ("x", "y"),
                np.arange(512 * 512, dtype=np.float64).reshape(512, 512),
            )
        },
        coords={"x": np.arange(512, dtype=np.float64), "y": np.arange(512)},
    )
    small_ds = xr.Dataset({"data": ("x", np.arange(4, dtype=np.float64))})
    tree = xr.DataTree.from_dict({"0/imagetool": large_ds, "1/imagetool": small_ds})
    try:
        encoding = manager_xarray.workspace_datatree_encoding(tree)
    finally:
        tree.close()

    assert set(encoding) == {"/0/imagetool"}
    assert set(encoding["/0/imagetool"]) == {"data"}


def test_workspace_datatree_encoding_can_be_disabled() -> None:
    tree = xr.DataTree.from_dict(
        {
            "0/imagetool": xr.Dataset(
                {
                    "data": (
                        ("x", "y"),
                        np.arange(512 * 512, dtype=np.float64).reshape(512, 512),
                    )
                }
            )
        }
    )
    try:
        assert manager_xarray.workspace_datatree_encoding(tree, compress=False) == {}
    finally:
        tree.close()


def test_workspace_xarray_path_helpers_cover_fallbacks(monkeypatch, tmp_path) -> None:
    class _BadPath(os.PathLike):
        def __fspath__(self) -> str:
            raise TypeError

    assert manager_xarray._normalized_file_path(object()) is None
    assert manager_xarray._normalized_file_path(_BadPath()) is None
    assert manager_xarray._normalized_file_path("") is None

    def _raise_oserror(_path: pathlib.Path) -> pathlib.Path:
        raise OSError("resolve failed")

    monkeypatch.setattr(pathlib.Path, "resolve", _raise_oserror)
    assert manager_xarray._normalized_file_path(tmp_path / "workspace.itws") == str(
        tmp_path / "workspace.itws"
    )

    monkeypatch.setattr(manager_xarray, "_normalized_file_path", lambda _path: None)
    lock = manager_xarray._workspace_file_lock("fallback.itws")
    assert lock is manager_xarray._workspace_file_lock("fallback.itws")

    def _raise_stat_oserror(_path: str):
        raise OSError

    monkeypatch.setattr(manager_xarray.os, "stat", _raise_stat_oserror)
    assert manager_xarray._workspace_file_identity("missing.itws") == (
        "missing.itws",
        0,
        0,
        0,
    )


def test_workspace_file_manager_uses_fsdecode_fallback(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_init(self, opener, *args, **kwargs):
        captured["opener"] = opener
        captured["args"] = args
        captured["kwargs"] = kwargs
        self._key = "fake-key"
        self._ref_counter = types.SimpleNamespace(decrement=lambda _key: None)
        self._cache = {}

    monkeypatch.setattr(
        manager_xarray, "ensure_workspace_hdf5_filters_registered", lambda: None
    )
    monkeypatch.setattr(manager_xarray, "_normalized_file_path", lambda _path: None)
    monkeypatch.setattr(
        manager_xarray, "_workspace_file_identity", lambda path: (path, 0, 0, 0)
    )
    monkeypatch.setattr(manager_xarray.CachingFileManager, "__init__", _fake_init)

    file_manager = manager_xarray.WorkspaceFileManager("fallback.itws")

    assert file_manager.workspace_path == "fallback.itws"
    assert captured["args"][0] == "fallback.itws"


def test_open_workspace_dataset_uses_fsdecode_fallback(monkeypatch) -> None:
    calls: list[tuple[object, str, str | None]] = []

    class _FakeFileManager:
        def __init__(self, path: str) -> None:
            self.workspace_path = path

    def _fake_open(file_manager, group: str, *, chunks: str | None):
        calls.append((file_manager, group, chunks))
        return "dataset"

    monkeypatch.setattr(manager_xarray, "_normalized_file_path", lambda _path: None)
    monkeypatch.setattr(manager_xarray, "WorkspaceFileManager", _FakeFileManager)
    monkeypatch.setattr(
        manager_xarray, "_open_workspace_dataset_from_manager", _fake_open
    )

    assert (
        manager_xarray.open_workspace_dataset("fallback.itws", "/0", chunks=None)
        == "dataset"
    )
    file_manager, group, chunks = calls[0]
    assert isinstance(file_manager, _FakeFileManager)
    assert file_manager.workspace_path == "fallback.itws"
    assert group == "/0"
    assert chunks is None


def test_open_workspace_datatree_closes_partial_groups_on_error(monkeypatch) -> None:
    closed: list[str] = []

    class _FakeDataset:
        def __init__(self, group_path: str) -> None:
            self.group_path = group_path

        def close(self) -> None:
            closed.append(self.group_path)

    class _FakeFileManager:
        workspace_path = "fallback.itws"

        def __init__(self, _path: str) -> None:
            pass

        def acquire_context(self):
            return contextlib.nullcontext(object())

    def _fake_open(_file_manager, group_path: str, *, chunks: str | None):
        if group_path == "/broken":
            raise RuntimeError("broken group")
        return _FakeDataset(group_path)

    monkeypatch.setattr(manager_xarray, "_normalized_file_path", lambda _path: None)
    monkeypatch.setattr(manager_xarray, "WorkspaceFileManager", _FakeFileManager)
    monkeypatch.setattr(
        manager_xarray, "_iter_h5netcdf_group_paths", lambda _h5_file: ("/", "/broken")
    )
    monkeypatch.setattr(
        manager_xarray, "_open_workspace_dataset_from_manager", _fake_open
    )

    with pytest.raises(RuntimeError, match="broken group"):
        manager_xarray.open_workspace_datatree("fallback.itws", chunks="auto")

    assert closed == ["/"]


def test_write_full_workspace_tree_file_compresses_payload_not_coords(
    tmp_path,
) -> None:
    import h5py
    import hdf5plugin

    ds = xr.Dataset(
        {
            "data": (
                ("x", "y"),
                np.arange(512 * 512, dtype=np.float64).reshape(512, 512),
            ),
            "small": ("x", np.arange(512, dtype=np.int64)),
        },
        coords={
            "x": np.linspace(-1.0, 1.0, 512),
            "y": np.linspace(-2.0, 2.0, 512),
        },
    )
    tree = xr.DataTree.from_dict({"0/imagetool": ds})
    fname = tmp_path / "compressed.itws"
    try:
        manager_workspace._write_full_workspace_tree_file(
            fname, tree, {"imagetool_workspace_schema_version": 4}
        )
    finally:
        tree.close()

    with h5py.File(fname, "r") as h5_file:
        assert hdf5plugin.Blosc2.filter_id in _hdf5_filter_ids(
            h5_file["0/imagetool/data"]
        )
        assert _hdf5_filter_ids(h5_file["0/imagetool/x"]) == []
        assert _hdf5_filter_ids(h5_file["0/imagetool/y"]) == []
        assert _hdf5_filter_ids(h5_file["0/imagetool/small"]) == []

    opened = manager_xarray.open_workspace_datatree(fname, chunks=None)
    try:
        loaded = typing.cast("xr.DataTree", opened["/0/imagetool"]).to_dataset(
            inherit=False
        )
        xarray.testing.assert_equal(loaded["data"], ds["data"])
        xarray.testing.assert_equal(loaded["x"], ds["x"])
        xarray.testing.assert_equal(loaded["y"], ds["y"])
    finally:
        opened.close()


def test_open_workspace_datatree_reads_uncompressed_workspace(tmp_path) -> None:
    ds = xr.Dataset(
        {"data": (("x", "y"), np.arange(12, dtype=np.float64).reshape(3, 4))},
        coords={"x": np.arange(3), "y": np.arange(4)},
    )
    tree = xr.DataTree.from_dict({"0/imagetool": ds})
    fname = tmp_path / "uncompressed.itws"
    try:
        tree.to_netcdf(fname, engine="h5netcdf", invalid_netcdf=True)
    finally:
        tree.close()

    opened = manager_xarray.open_workspace_datatree(fname, chunks=None)
    try:
        loaded = typing.cast("xr.DataTree", opened["/0/imagetool"]).to_dataset(
            inherit=False
        )
        xarray.testing.assert_equal(loaded["data"], ds["data"])
    finally:
        opened.close()


def test_write_full_workspace_tree_file_replaces_stale_root_attrs(tmp_path) -> None:
    import h5py

    fname = tmp_path / "root-attrs.itws"
    tree = xr.DataTree.from_dict(
        {"0/imagetool": _transaction_test_dataset(1.0, title="old")}
    )
    tree.attrs["stale_workspace_attr"] = "remove me"
    try:
        manager_workspace._write_full_workspace_tree_file(
            fname, tree, _transaction_test_root_attrs()
        )
    finally:
        tree.close()

    with h5py.File(fname, "r") as h5_file:
        assert "stale_workspace_attr" not in h5_file.attrs
        manifest = manager_workspace._workspace_manifest_from_attrs(h5_file.attrs)
        assert manifest == {"schema_version": 4, "root_order": [0], "nodes": []}


def test_workspace_recovery_discards_pending_only_transaction(tmp_path) -> None:
    fname = tmp_path / "pending-only.itws"
    _write_transaction_test_workspace(fname)
    rewrite = manager_workspace._WorkspaceRewriteGroup(
        "0", {"0/imagetool": _transaction_test_dataset(2.0, title="new")}
    )
    rewrite_map = {"0": rewrite}
    txn_id = "pendingonly"
    txn_path = f"{manager_workspace._WORKSPACE_TRANSACTION_GROUP_PREFIX}{txn_id}"
    pending_root = f"{manager_workspace._WORKSPACE_PENDING_GROUP_PREFIX}{txn_id}"
    backup_root = f"{manager_workspace._WORKSPACE_BACKUP_GROUP_PREFIX}{txn_id}"

    manager_workspace._prepare_workspace_transaction(
        fname,
        txn_path,
        pending_root,
        backup_root,
        rewrite_map,
        (),
        _transaction_test_root_attrs(delta_save_count=1),
    )
    manager_workspace._write_workspace_transaction_pending_groups(
        fname, rewrite_map, pending_root
    )

    manager_workspace._recover_workspace_transactions(fname)

    assert _read_transaction_test_value(fname) == 1.0
    _assert_no_workspace_internal_groups(fname)


def test_workspace_recovery_restores_backup_before_pending_move(tmp_path) -> None:
    import h5py

    fname = tmp_path / "backup-before-pending.itws"
    _write_transaction_test_workspace(fname)
    rewrite = manager_workspace._WorkspaceRewriteGroup(
        "0", {"0/imagetool": _transaction_test_dataset(2.0, title="new")}
    )
    rewrite_map = {"0": rewrite}
    txn_id = "backuponly"
    txn_path = f"{manager_workspace._WORKSPACE_TRANSACTION_GROUP_PREFIX}{txn_id}"
    pending_root = f"{manager_workspace._WORKSPACE_PENDING_GROUP_PREFIX}{txn_id}"
    backup_root = f"{manager_workspace._WORKSPACE_BACKUP_GROUP_PREFIX}{txn_id}"
    group_operations, _ = manager_workspace._prepare_workspace_transaction(
        fname,
        txn_path,
        pending_root,
        backup_root,
        rewrite_map,
        (),
        _transaction_test_root_attrs(delta_save_count=1),
    )
    manager_workspace._write_workspace_transaction_pending_groups(
        fname, rewrite_map, pending_root
    )

    with h5py.File(fname, "a") as h5_file:
        manager_workspace._set_workspace_transaction_status(
            h5_file,
            txn_path,
            "committing",
        )
        operation = group_operations[0]
        manager_workspace._move_h5_path(
            h5_file,
            typing.cast("str", operation["group_path"]),
            typing.cast("str", operation["backup_path"]),
        )

    manager_workspace._recover_workspace_transactions(fname)

    assert _read_transaction_test_value(fname) == 1.0
    _assert_no_workspace_internal_groups(fname)


def test_workspace_recovery_rolls_back_active_moved_before_commit(tmp_path) -> None:
    import h5py

    fname = tmp_path / "active-before-commit.itws"
    _write_transaction_test_workspace(fname)
    rewrite = manager_workspace._WorkspaceRewriteGroup(
        "0", {"0/imagetool": _transaction_test_dataset(2.0, title="new")}
    )
    rewrite_map = {"0": rewrite}
    txn_id = "activemoved"
    txn_path = f"{manager_workspace._WORKSPACE_TRANSACTION_GROUP_PREFIX}{txn_id}"
    pending_root = f"{manager_workspace._WORKSPACE_PENDING_GROUP_PREFIX}{txn_id}"
    backup_root = f"{manager_workspace._WORKSPACE_BACKUP_GROUP_PREFIX}{txn_id}"
    group_operations, _ = manager_workspace._prepare_workspace_transaction(
        fname,
        txn_path,
        pending_root,
        backup_root,
        rewrite_map,
        (),
        _transaction_test_root_attrs(delta_save_count=1),
    )
    manager_workspace._write_workspace_transaction_pending_groups(
        fname, rewrite_map, pending_root
    )

    with h5py.File(fname, "a") as h5_file:
        manager_workspace._set_workspace_transaction_status(
            h5_file,
            txn_path,
            "committing",
        )
        operation = group_operations[0]
        manager_workspace._move_h5_path(
            h5_file,
            typing.cast("str", operation["group_path"]),
            typing.cast("str", operation["backup_path"]),
        )
        manager_workspace._move_h5_path(
            h5_file,
            typing.cast("str", operation["pending_path"]),
            typing.cast("str", operation["group_path"]),
        )

    manager_workspace._recover_workspace_transactions(fname)

    assert _read_transaction_test_value(fname) == 1.0
    _assert_no_workspace_internal_groups(fname)


def test_workspace_recovery_accepts_committed_before_cleanup(tmp_path) -> None:
    fname = tmp_path / "committed-before-cleanup.itws"
    _write_transaction_test_workspace(fname)
    rewrite = manager_workspace._WorkspaceRewriteGroup(
        "0", {"0/imagetool": _transaction_test_dataset(2.0, title="new")}
    )
    rewrite_map = {"0": rewrite}
    txn_id = "committed"
    txn_path = f"{manager_workspace._WORKSPACE_TRANSACTION_GROUP_PREFIX}{txn_id}"
    pending_root = f"{manager_workspace._WORKSPACE_PENDING_GROUP_PREFIX}{txn_id}"
    backup_root = f"{manager_workspace._WORKSPACE_BACKUP_GROUP_PREFIX}{txn_id}"
    root_attrs = _transaction_test_root_attrs(delta_save_count=1)
    group_operations, attr_updates = manager_workspace._prepare_workspace_transaction(
        fname,
        txn_path,
        pending_root,
        backup_root,
        rewrite_map,
        (),
        root_attrs,
    )
    manager_workspace._write_workspace_transaction_pending_groups(
        fname, rewrite_map, pending_root
    )
    manager_workspace._commit_workspace_transaction(
        fname, txn_path, group_operations, attr_updates, root_attrs
    )

    manager_workspace._recover_workspace_transactions(fname)

    assert _read_transaction_test_value(fname) == 2.0
    _assert_no_workspace_internal_groups(fname)


def test_workspace_recovery_rolls_back_attr_only_transaction(tmp_path) -> None:
    import h5py

    fname = tmp_path / "attrs-before-commit.itws"
    _write_transaction_test_workspace(fname)
    fallback = manager_workspace._WorkspaceRewriteGroup(
        "0", {"0/imagetool": _transaction_test_dataset(2.0, title="fallback")}
    )
    attr_update = manager_workspace._WorkspaceAttrUpdate(
        payload_path="0/imagetool",
        attrs={"itool_title": "new"},
        fallback=fallback,
    )
    txn_id = "attrrollback"
    txn_path = f"{manager_workspace._WORKSPACE_TRANSACTION_GROUP_PREFIX}{txn_id}"
    pending_root = f"{manager_workspace._WORKSPACE_PENDING_GROUP_PREFIX}{txn_id}"
    backup_root = f"{manager_workspace._WORKSPACE_BACKUP_GROUP_PREFIX}{txn_id}"
    root_attrs = _transaction_test_root_attrs(delta_save_count=1)
    _, attr_updates = manager_workspace._prepare_workspace_transaction(
        fname, txn_path, pending_root, backup_root, {}, (attr_update,), root_attrs
    )

    with h5py.File(fname, "a") as h5_file:
        manager_workspace._set_workspace_transaction_status(
            h5_file,
            txn_path,
            "committing",
        )
        manager_workspace._replace_h5_attrs(
            h5_file["0/imagetool"].attrs, attr_updates[0].attrs
        )
        manager_workspace._write_root_attrs_to_open_workspace_file(h5_file, root_attrs)
        h5_file.flush()

    manager_workspace._recover_workspace_transactions(fname)

    with h5py.File(fname, "r") as h5_file:
        assert h5_file["0/imagetool"].attrs["itool_title"] == "old"
        assert (
            manager_workspace._workspace_delta_save_count_from_attrs(h5_file.attrs) == 0
        )
    _assert_no_workspace_internal_groups(fname)


def test_workspace_recovery_cleans_orphan_internal_groups(tmp_path) -> None:
    import h5py

    fname = tmp_path / "orphan-internal.itws"
    _write_transaction_test_workspace(fname)
    with h5py.File(fname, "a") as h5_file:
        h5_file.create_group(
            f"{manager_workspace._WORKSPACE_PENDING_GROUP_PREFIX}orphan"
        )
        h5_file.create_group(
            f"{manager_workspace._WORKSPACE_BACKUP_GROUP_PREFIX}orphan"
        )

    manager_workspace._recover_workspace_transactions(fname)

    _assert_no_workspace_internal_groups(fname)


def test_workspace_lock_path_uses_hidden_sidecar(tmp_path) -> None:
    fname = tmp_path / "example.itws"

    assert manager_workspace._workspace_lock_path(fname) == str(
        (tmp_path / ".example.itws.lock").resolve()
    )


def test_workspace_lock_conflict_is_reported(tmp_path) -> None:
    fname = tmp_path / "locked.itws"
    _write_transaction_test_workspace(fname)
    hidden_lock_path = pathlib.Path(manager_workspace._workspace_lock_path(fname))
    visible_lock_path = pathlib.Path(f"{fname.resolve()}.lock")
    lock = manager_workspace._acquire_workspace_document_lock(fname)
    try:
        assert lock.staleLockTime() == 0
        assert hidden_lock_path.exists()
        assert not visible_lock_path.exists()
        with pytest.raises(BlockingIOError):
            manager_workspace._acquire_workspace_document_lock(fname)
    finally:
        lock.unlock()


def test_hide_workspace_lock_file_sets_macos_hidden_flag(monkeypatch) -> None:
    calls: list[tuple[str, int]] = []
    lock_path = "/workspace/.workspace.itws.lock"
    regular_stat = types.SimpleNamespace(st_mode=0o100600)

    monkeypatch.setattr(manager_workspace.sys, "platform", "darwin")
    monkeypatch.setattr(manager_workspace.os, "lstat", lambda _path: regular_stat)
    monkeypatch.setattr(
        manager_workspace.os,
        "chflags",
        lambda path, flags: calls.append((path, flags)),
        raising=False,
    )

    manager_workspace._hide_workspace_lock_file(lock_path)

    assert calls == [(lock_path, 0x8000)]


def test_hide_workspace_lock_file_skips_macos_symlink(monkeypatch) -> None:
    calls: list[tuple[str, int]] = []
    symlink_stat = types.SimpleNamespace(st_mode=0o120777)

    monkeypatch.setattr(manager_workspace.sys, "platform", "darwin")
    monkeypatch.setattr(manager_workspace.os, "lstat", lambda _path: symlink_stat)
    monkeypatch.setattr(
        manager_workspace.os,
        "chflags",
        lambda path, flags: calls.append((path, flags)),
        raising=False,
    )

    manager_workspace._hide_workspace_lock_file("/workspace/.workspace.itws.lock")

    assert calls == []


def test_workspace_lock_error_message_names_owner(monkeypatch, tmp_path) -> None:
    fname = tmp_path / "busy-message.itws"
    _write_transaction_test_workspace(fname)
    lock = manager_workspace._acquire_workspace_document_lock(fname)
    lock_info = manager_workspace._workspace_document_lock_info(fname)
    calls: list[dict[str, object]] = []

    def _critical(*args, **kwargs) -> int:
        calls.append({"args": args, "kwargs": kwargs})
        return int(QtWidgets.QDialog.DialogCode.Accepted)

    monkeypatch.setattr(erlab.interactive.utils.MessageDialog, "critical", _critical)
    try:
        manager_mainwindow._show_workspace_file_lock_error(None, fname)
    finally:
        lock.unlock()

    assert len(calls) == 1
    args = calls[0]["args"]
    assert isinstance(args, tuple)
    assert args[1] == "Workspace Already Open"
    assert args[2] == "This workspace is already open somewhere else."
    informative_text = args[3]
    assert isinstance(informative_text, str)
    assert fname.name in informative_text
    assert "lock" not in informative_text.casefold()
    if lock_info.owner:
        assert lock_info.owner in informative_text
    if lock_info.hostname:
        assert lock_info.hostname in informative_text
    detailed_text = calls[0]["kwargs"]["detailed_text"]
    assert isinstance(detailed_text, str)
    assert "Temporary workspace ownership marker:" in detailed_text
    assert lock_info.path in detailed_text


def test_workspace_lock_text_variants(tmp_path) -> None:
    app_only = manager_workspace._WorkspaceDocumentLockInfo(
        path="marker",
        owner="user",
        hostname="",
        appname="ImageTool",
        pid=None,
    )
    pid_only = manager_workspace._WorkspaceDocumentLockInfo(
        path="marker",
        owner="",
        hostname="",
        appname="",
        pid=123,
    )
    full_info = manager_workspace._WorkspaceDocumentLockInfo(
        path="marker",
        owner="user",
        hostname="workstation",
        appname="ImageTool",
        pid=123,
    )

    assert manager_mainwindow._workspace_lock_owner_text(app_only) == (
        "user using ImageTool"
    )
    assert manager_mainwindow._workspace_lock_owner_text(pid_only) == (
        "using process 123"
    )
    assert manager_mainwindow._workspace_lock_owner_text(full_info) == (
        "user on workstation using ImageTool (process 123)"
    )

    def _raise_owner_details_failed() -> None:
        raise RuntimeError("owner details failed")

    def _details_from_active_exception() -> str:
        try:
            _raise_owner_details_failed()
        except RuntimeError:
            return manager_mainwindow._workspace_lock_details_text(
                tmp_path / "workspace.itws", full_info
            )

    details = _details_from_active_exception()

    assert "owner details failed" in details
    assert "Temporary workspace ownership marker: marker" in details


def test_workspace_window_title_placeholder_non_macos(monkeypatch) -> None:
    monkeypatch.setattr(manager_mainwindow.sys, "platform", "linux")

    assert manager_mainwindow._strip_workspace_modified_placeholder("Name[*]") == "Name"
    assert manager_mainwindow._window_title_with_modified_placeholder("Name[*]") == (
        "Name[*]"
    )


def test_workspace_lock_error_message_without_owner(monkeypatch, tmp_path) -> None:
    fname = tmp_path / "busy-message.itws"
    calls: list[dict[str, object]] = []
    lock_info = manager_workspace._WorkspaceDocumentLockInfo(
        path=str(tmp_path / ".busy-message.itws.lock"),
        owner="",
        hostname="",
        appname="",
        pid=None,
    )

    def _critical(*args, **kwargs) -> int:
        calls.append({"args": args, "kwargs": kwargs})
        return int(QtWidgets.QDialog.DialogCode.Accepted)

    monkeypatch.setattr(
        manager_workspace, "_workspace_document_lock_info", lambda _fname: lock_info
    )
    monkeypatch.setattr(erlab.interactive.utils.MessageDialog, "critical", _critical)

    manager_mainwindow._show_workspace_file_lock_error(None, fname)

    args = calls[0]["args"]
    assert isinstance(args, tuple)
    informative_text = args[3]
    assert isinstance(informative_text, str)
    assert informative_text == (
        "Close the other ImageTool Manager that has busy-message.itws open, "
        "then try again."
    )


def test_application_quit_filter_routes_quit_events(qtbot) -> None:
    manager = QtWidgets.QWidget()
    qtbot.addWidget(manager)
    calls: list[str] = []

    def _handle_application_quit_request() -> bool:
        calls.append("quit")
        return True

    manager._handle_application_quit_request = _handle_application_quit_request
    event_filter = manager_mainwindow._ApplicationQuitFilter(
        typing.cast("ImageToolManager", manager)
    )

    assert not event_filter.eventFilter(None, None)
    assert event_filter.eventFilter(None, QtCore.QEvent(QtCore.QEvent.Type.Quit))

    class _QuitKeyEvent(QtGui.QKeyEvent):
        def matches(self, key: QtGui.QKeySequence.StandardKey) -> bool:
            return key == QtGui.QKeySequence.StandardKey.Quit

    quit_event = _QuitKeyEvent(
        QtCore.QEvent.Type.KeyPress,
        QtCore.Qt.Key.Key_Q,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )

    assert event_filter.eventFilter(None, quit_event)
    assert quit_event.isAccepted()
    assert calls == ["quit", "quit"]


def test_workspace_document_access_releases_lock(tmp_path) -> None:
    class _FakeLock:
        def __init__(self) -> None:
            self.unlock_count = 0

        def unlock(self) -> None:
            self.unlock_count += 1

    lock = _FakeLock()
    access = manager_mainwindow._WorkspaceDocumentAccess(
        tmp_path / "workspace.itws", lock
    )

    assert access.take_lock() is lock
    access.release()
    assert lock.unlock_count == 0

    access = manager_mainwindow._WorkspaceDocumentAccess(
        tmp_path / "workspace.itws", lock
    )
    access.release()
    access.release()
    assert lock.unlock_count == 1


def test_choose_from_datatree_dialog_root_keys_skip_missing(qtbot) -> None:
    manager = QtWidgets.QWidget()
    manager.next_idx = 7
    qtbot.addWidget(manager)
    tree = xr.DataTree.from_dict(
        {
            "0/imagetool": xr.Dataset(
                attrs={"itool_title": "Loaded"},
            )
        }
    )
    try:
        dialog = _ChooseFromDataTreeDialog(
            typing.cast("ImageToolManager", manager),
            tree,
            mode="load",
            root_keys=("missing", "0"),
        )
        qtbot.addWidget(dialog)

        assert dialog._tree_widget.topLevelItemCount() == 1
        assert dialog._tree_widget.topLevelItem(0).text(0) == "7: Loaded"
    finally:
        tree.close()


def test_manager_workspace_save_as_locked_target_does_not_write(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    fname = tmp_path / "locked-save-as.itws"
    _write_transaction_test_workspace(fname)
    lock = manager_workspace._acquire_workspace_document_lock(fname)
    operation_errors: list[tuple[typing.Any, ...]] = []
    try:
        with manager_context() as manager:
            monkeypatch.setattr(
                manager, "_workspace_save_dialog", lambda *args, **kwargs: str(fname)
            )
            monkeypatch.setattr(
                manager,
                "_save_workspace_document",
                lambda *args, **kwargs: pytest.fail(
                    "Save As should lock the target before writing"
                ),
            )
            monkeypatch.setattr(
                manager,
                "_show_operation_error",
                lambda *args, **kwargs: operation_errors.append(args),
            )

            assert not manager.save_as(native=False)
    finally:
        lock.unlock()

    assert operation_errors


def test_manager_workspace_load_locks_before_recovery(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    fname = tmp_path / "locked-load.itws"
    _write_transaction_test_workspace(fname)
    lock = manager_workspace._acquire_workspace_document_lock(fname)
    recovery_calls: list[pathlib.Path] = []
    try:
        monkeypatch.setattr(
            manager_workspace,
            "_recover_workspace_transactions",
            lambda path: (
                recovery_calls.append(pathlib.Path(path))
                or pytest.fail("Load should lock the workspace before recovery")
            ),
        )
        with manager_context() as manager, pytest.raises(BlockingIOError):
            manager._load_workspace_file(
                fname,
                replace=True,
                associate=True,
                mark_dirty=False,
                select=False,
            )
    finally:
        lock.unlock()

    assert recovery_calls == []


def test_manager_workspace_path_lock_contract(
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    class _FakeLock:
        def __init__(self) -> None:
            self.unlock_count = 0

        def unlock(self) -> None:
            self.unlock_count += 1

    with manager_context() as manager:
        current = (tmp_path / "current.itws").resolve()
        manager._workspace_path = current
        lock = _FakeLock()

        manager._set_workspace_path(current, workspace_lock=lock)

        assert lock.unlock_count == 1
        with pytest.raises(RuntimeError, match="pre-acquired document lock"):
            manager._set_workspace_path(tmp_path / "other.itws")


def test_manager_application_quit_request_resets_when_close_fails(
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        original_close = manager.close
        monkeypatch.setattr(manager, "close", lambda: False)

        assert manager._handle_application_quit_request()
        assert not manager._application_quit_requested

        monkeypatch.setattr(manager, "close", lambda: True)
        assert manager._handle_application_quit_request()
        assert manager._application_quit_requested
        manager._application_quit_requested = False
        monkeypatch.setattr(manager, "close", original_close)


def test_manager_active_window_and_focus_restore_guards(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        active = QtWidgets.QWidget()
        origin = QtWidgets.QWidget()
        other = QtWidgets.QWidget()
        qtbot.addWidget(active)
        qtbot.addWidget(origin)
        qtbot.addWidget(other)
        origin.show()
        other.show()

        monkeypatch.setattr(
            QtWidgets.QApplication, "activeWindow", staticmethod(lambda: active)
        )
        monkeypatch.setattr(manager, "_node_uid_from_window", lambda _window: "uid")
        monkeypatch.setattr(erlab.interactive.utils, "qt_is_valid", lambda _obj: False)
        assert manager._active_managed_window() is None

        monkeypatch.setattr(
            QtWidgets.QApplication, "activeWindow", staticmethod(lambda: other)
        )
        monkeypatch.setattr(erlab.interactive.utils, "qt_is_valid", lambda _obj: True)
        manager._restore_focus_after_workspace_save(origin)


def test_manager_compact_workspace_edge_paths(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        monkeypatch.setattr(manager, "save_as", lambda: True)
        assert manager.compact_workspace()

        manager._workspace_path = tmp_path / "workspace.itws"
        manager._workspace_save_in_progress = True
        assert not manager.compact_workspace()
        manager._workspace_save_in_progress = False

        operation_errors: list[tuple[typing.Any, ...]] = []
        focus_restores: list[QtWidgets.QWidget | None] = []
        monkeypatch.setattr(
            erlab.interactive.utils,
            "wait_dialog",
            lambda *args, **kwargs: contextlib.nullcontext(),
        )
        monkeypatch.setattr(
            manager,
            "_save_workspace_document",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("compact failed")
            ),
        )
        monkeypatch.setattr(
            manager,
            "_show_operation_error",
            lambda *args: operation_errors.append(args),
        )
        monkeypatch.setattr(
            manager,
            "_restore_focus_after_workspace_save",
            lambda origin: focus_restores.append(origin),
        )

        assert not manager.compact_workspace()
        assert operation_errors == [
            (
                "Error while compacting workspace",
                "An error occurred while compacting the workspace file.",
            )
        ]
        assert focus_restores == [None]


def test_manager_shutdown_compaction_logs_failure(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager._workspace_path = tmp_path / "workspace.itws"
        manager._workspace_delta_save_count = 1
        monkeypatch.setattr(
            manager,
            "_save_workspace_document",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("compact failed")
            ),
        )

        manager._compact_workspace_before_shutdown()


def test_manager_workspace_save_dialog_paths(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    calls: list[tuple[str, object]] = []
    original_file_dialog = QtWidgets.QFileDialog

    class _FakeFileDialog:
        AcceptMode = original_file_dialog.AcceptMode
        FileMode = original_file_dialog.FileMode
        Option = original_file_dialog.Option
        exec_result = 0

        def __init__(self, _parent, caption: str) -> None:
            calls.append(("caption", caption))

        def setAcceptMode(self, mode) -> None:
            calls.append(("accept", mode))

        def setFileMode(self, mode) -> None:
            calls.append(("file_mode", mode))

        def setNameFilter(self, name_filter: str) -> None:
            calls.append(("filter", name_filter))

        def setDefaultSuffix(self, suffix: str) -> None:
            calls.append(("suffix", suffix))

        def selectFile(self, fname: str) -> None:
            calls.append(("select", fname))

        def setDirectory(self, directory: str) -> None:
            calls.append(("directory", directory))

        def setOption(self, option) -> None:
            calls.append(("option", option))

        def exec(self) -> int:
            return self.exec_result

        def selectedFiles(self) -> list[str]:
            return [str(tmp_path / "selected.itws")]

    monkeypatch.setattr(QtWidgets, "QFileDialog", _FakeFileDialog)
    with manager_context() as manager:
        assert (
            manager._workspace_save_dialog(
                native=False, selected_file=tmp_path / "explicit.itws"
            )
            is None
        )
        assert ("select", str(tmp_path / "explicit.itws")) in calls

        _FakeFileDialog.exec_result = 1
        manager._workspace_path = tmp_path / "bound.itws"
        assert manager._workspace_save_dialog(native=True) == str(
            tmp_path / "selected.itws"
        )
        assert ("select", str(tmp_path / "bound.itws")) in calls

        manager._workspace_path = None
        manager._recent_directory = str(tmp_path)
        assert manager._workspace_save_dialog(native=True) == str(
            tmp_path / "selected.itws"
        )
        assert ("directory", str(tmp_path)) in calls


def test_manager_confirm_save_dirty_workspace_save_branch(
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager._workspace_structure_modified = True
        monkeypatch.setattr(manager, "save", lambda: True)
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "exec",
            lambda _msg_box: QtWidgets.QMessageBox.StandardButton.Save,
        )

        assert manager._confirm_save_dirty_workspace("Save before continuing.")


def test_manager_legacy_workspace_save_helpers(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "exec",
            lambda _msg_box: QtWidgets.QMessageBox.StandardButton.Ok,
        )
        manager._show_legacy_workspace_upgrade_message(tmp_path / "legacy.itws")

        monkeypatch.setattr(manager, "_workspace_save_dialog", lambda **_kwargs: None)
        assert manager._save_legacy_workspace_as_v4(tmp_path / "legacy.itws") is None

        dirty_reasons: list[str] = []
        monkeypatch.setattr(
            manager,
            "_save_legacy_workspace_as_v4",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            manager, "_mark_workspace_structure_dirty", dirty_reasons.append
        )
        manager._associate_loaded_workspace_file(
            tmp_path / "legacy.itws",
            manager_workspace._WORKSPACE_LEGACY_SCHEMA_VERSION - 1,
        )

        assert manager._workspace_path is None
        assert manager._workspace_needs_full_save
        assert dirty_reasons == ["Legacy workspace needs conversion"]


def test_manager_save_and_wait_dialog_error_paths(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    critical_calls: list[tuple[typing.Any, ...]] = []

    def _critical(*args, **kwargs) -> int:
        critical_calls.append(args)
        return int(QtWidgets.QDialog.DialogCode.Accepted)

    monkeypatch.setattr(erlab.interactive.utils.MessageDialog, "critical", _critical)
    with manager_context() as manager:
        wait_dialog = manager._open_workspace_save_wait_dialog(manager)
        qtbot.addWidget(wait_dialog)
        wait_dialog.close()

        manager._show_workspace_save_worker_error("Traceback text")
        assert critical_calls[-1][2] == (
            "An error occurred while saving the workspace file."
        )

        manager._workspace_path = tmp_path / "workspace.itws"
        manager._workspace_save_in_progress = True
        assert not manager.save()

        manager._workspace_save_in_progress = False
        monkeypatch.setattr(
            manager,
            "_workspace_save_snapshot",
            lambda _path: (_ for _ in ()).throw(RuntimeError("snapshot failed")),
        )
        monkeypatch.setattr(
            manager, "_restore_focus_after_workspace_save", lambda _origin: None
        )
        assert not manager.save()

        monkeypatch.setattr(manager, "_workspace_save_dialog", lambda **_kwargs: None)
        assert not manager.save_as()


def test_open_multiple_files_workspace_locks_before_recovery(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    fname = tmp_path / "locked-dropped.itws"
    _write_transaction_test_workspace(fname)
    lock = manager_workspace._acquire_workspace_document_lock(fname)
    lock_calls: list[pathlib.Path] = []
    recovery_calls: list[pathlib.Path] = []
    try:
        monkeypatch.setattr(
            manager_workspace,
            "_recover_workspace_transactions",
            lambda path: (
                recovery_calls.append(pathlib.Path(path))
                or pytest.fail("Dropped workspace should lock before recovery")
            ),
        )
        monkeypatch.setattr(
            manager_mainwindow,
            "_show_workspace_file_lock_error",
            lambda _parent, locked_fname: lock_calls.append(pathlib.Path(locked_fname)),
        )
        monkeypatch.setattr(
            erlab.interactive.utils,
            "file_loaders",
            lambda *args, **kwargs: pytest.fail(
                "locked workspace should not fall through to loaders"
            ),
        )

        with manager_context() as manager:
            manager.open_multiple_files([fname], try_workspace=True)
    finally:
        lock.unlock()

    assert recovery_calls == []
    assert lock_calls == [fname]


def test_workspace_high_risk_path_detection() -> None:
    assert manager_workspace._workspace_path_is_high_risk(
        pathlib.Path.home() / "OneDrive" / "workspace.itws"
    )
    assert manager_workspace._workspace_path_is_high_risk(
        pathlib.Path.home()
        / "Library"
        / "Mobile Documents"
        / "com~apple~CloudDocs"
        / "workspace.itws"
    )
    assert manager_workspace._workspace_path_is_high_risk(
        pathlib.Path("//server/share/workspace.itws")
    )


def test_workspace_lock_error_detection_message_variants() -> None:
    transient = OSError(errno.EACCES, "resource temporarily unavailable")
    assert manager_workspace._is_workspace_file_lock_error(transient)
    assert manager_workspace._is_workspace_file_lock_error(
        RuntimeError("file is already open by another process")
    )
    assert manager_workspace._is_workspace_file_lock_error(
        RuntimeError("unable to lock file")
    )
    assert not manager_workspace._is_workspace_file_lock_error(
        OSError(errno.EINVAL, "resource temporarily unavailable")
    )


def test_hide_workspace_lock_file_windows_paths(monkeypatch) -> None:
    import ctypes

    calls: list[tuple[str, int]] = []

    class _Kernel32:
        @staticmethod
        def SetFileAttributesW(path: str, attrs: int) -> None:
            calls.append((path, attrs))

    monkeypatch.setattr(manager_workspace.sys, "platform", "win32")
    monkeypatch.setattr(manager_workspace.os, "name", "nt")
    monkeypatch.setattr(ctypes, "windll", None, raising=False)
    manager_workspace._hide_workspace_lock_file("missing-windll.itws.lock")
    assert calls == []

    monkeypatch.setattr(
        ctypes, "windll", types.SimpleNamespace(kernel32=_Kernel32()), raising=False
    )
    manager_workspace._hide_workspace_lock_file("hidden.itws.lock")
    assert calls == [("hidden.itws.lock", 0x2)]


def test_workspace_document_lock_info_without_lock(tmp_path) -> None:
    info = manager_workspace._workspace_document_lock_info(tmp_path / "free.itws")

    assert info.pid is None
    assert info.hostname == ""
    assert info.appname == ""


def test_workspace_metadata_helpers_cover_invalid_payloads() -> None:
    manifest_attrs = manager_workspace._workspace_root_attrs_payload(
        root_order=["1"],
        nodes=[{"path": "1"}],
        delta_save_count=2,
        erlab_version="test",
    )
    raw_manifest = manifest_attrs[manager_workspace._WORKSPACE_MANIFEST_ATTR]

    assert (
        manager_workspace._workspace_manifest_from_attrs(
            {manager_workspace._WORKSPACE_MANIFEST_ATTR: raw_manifest.encode()}
        )["delta_save_count"]
        == 2
    )
    assert (
        manager_workspace._workspace_manifest_from_attrs(
            {manager_workspace._WORKSPACE_MANIFEST_ATTR: "{not-json"}
        )
        == {}
    )
    assert (
        manager_workspace._workspace_delta_save_count_from_attrs(
            {
                manager_workspace._WORKSPACE_MANIFEST_ATTR: (
                    '{"delta_save_count": "not-an-int"}'
                )
            }
        )
        == 0
    )


def test_workspace_path_risk_detection_fallbacks(monkeypatch, tmp_path) -> None:
    def _raise_oserror(_path: pathlib.Path) -> pathlib.Path:
        raise OSError("resolve failed")

    monkeypatch.setattr(pathlib.Path, "resolve", _raise_oserror)
    assert manager_workspace._workspace_path_is_likely_cloud_path(
        tmp_path / "Dropbox" / "workspace.itws"
    )
    assert manager_workspace._workspace_path_is_likely_network_path(
        pathlib.Path("/net/server/workspace.itws")
    )

    monkeypatch.setattr(manager_workspace.sys, "platform", "darwin")
    assert manager_workspace._workspace_path_is_likely_network_path(
        pathlib.Path("/Volumes/share/workspace.itws")
    )


def test_workspace_requires_full_save_reasons(tmp_path) -> None:
    options = erlab.interactive.options
    old_incremental = options["io/workspace/use_incremental"]
    old_remote = options["io/workspace/incremental_save_on_remote"]
    existing = tmp_path / "existing.itws"
    existing.touch()
    try:
        options["io/workspace/use_incremental"] = False
        assert manager_workspace._workspace_requires_full_save(
            existing,
            needs_full_save=False,
            schema_version=manager_workspace._current_workspace_schema_version(),
            structure_modified=False,
            has_dirty_added=False,
            has_dirty_removed=False,
        )

        options["io/workspace/use_incremental"] = True
        options["io/workspace/incremental_save_on_remote"] = True
        assert manager_workspace._workspace_requires_full_save(
            tmp_path / "missing.itws",
            needs_full_save=False,
            schema_version=manager_workspace._current_workspace_schema_version(),
            structure_modified=False,
            has_dirty_added=False,
            has_dirty_removed=False,
        )
        for kwargs in (
            {"needs_full_save": True},
            {
                "schema_version": (
                    manager_workspace._current_workspace_schema_version() - 1
                )
            },
            {"structure_modified": True},
            {"has_dirty_added": True},
            {"has_dirty_removed": True},
        ):
            call_kwargs = {
                "needs_full_save": False,
                "schema_version": manager_workspace._current_workspace_schema_version(),
                "structure_modified": False,
                "has_dirty_added": False,
                "has_dirty_removed": False,
            }
            call_kwargs.update(kwargs)
            assert manager_workspace._workspace_requires_full_save(
                existing, **call_kwargs
            )
    finally:
        options["io/workspace/use_incremental"] = old_incremental
        options["io/workspace/incremental_save_on_remote"] = old_remote


def test_workspace_h5_transaction_helper_edge_cases(tmp_path) -> None:
    import h5py

    fname = tmp_path / "transaction-helpers.itws"
    with h5py.File(fname, "w") as h5_file:
        h5_file.attrs["imagetool_workspace_schema_version"] = (
            manager_workspace._current_workspace_schema_version()
        )
        assert manager_workspace._workspace_txn_attr_target(h5_file, "/missing") is None

        txn = h5_file.create_group(
            f"{manager_workspace._WORKSPACE_TRANSACTION_GROUP_PREFIX}x"
        )
        txn_name = txn.name.strip("/")
        manager_workspace._restore_workspace_attr_backups(h5_file, txn)

        txn.attrs["operations"] = b'{"group_replacements": []}'
        assert manager_workspace._workspace_transaction_operations(txn) == {
            "group_replacements": []
        }
        txn.attrs["operations"] = "{not-json"
        assert manager_workspace._workspace_transaction_operations(txn) == {}

        txn.attrs["pending_root"] = b"__itws_pending_x"
        txn.attrs["backup_root"] = b"__itws_backup_x"
        assert manager_workspace._workspace_transaction_roots(txn) == (
            "__itws_pending_x",
            "__itws_backup_x",
        )

        manager_workspace._rollback_workspace_group_operations(
            h5_file, {"group_replacements": "not-a-list"}
        )
        manager_workspace._rollback_workspace_group_operations(
            h5_file,
            {"group_replacements": [None, {"group_path": 1, "backup_path": "x"}]},
        )

        target = h5_file.create_group("target")
        target.attrs["value"] = "old"
        txn.attrs["status"] = b"committing"
        txn.attrs["operations"] = json.dumps(
            {
                "group_replacements": [
                    {
                        "group_path": "target",
                        "backup_path": "missing-backup",
                        "old_exists": False,
                    }
                ]
            }
        )
        pending = h5_file.create_group("__itws_pending_x")
        pending.attrs["unused"] = True
        backup = h5_file.create_group("__itws_backup_x")
        backup.attrs["unused"] = True

        manager_workspace._recover_open_workspace_transaction(h5_file, txn.name)

        assert "target" not in h5_file
        assert "__itws_pending_x" not in h5_file
        assert "__itws_backup_x" not in h5_file
        assert txn_name not in h5_file


def test_recover_workspace_transactions_ignores_non_workspace_file(tmp_path) -> None:
    import h5py

    fname = tmp_path / "plain.h5"
    with h5py.File(fname, "w") as h5_file:
        h5_file.create_group(
            f"{manager_workspace._WORKSPACE_TRANSACTION_GROUP_PREFIX}x"
        )

    manager_workspace._recover_workspace_transactions(fname)

    with h5py.File(fname, "r") as h5_file:
        assert f"{manager_workspace._WORKSPACE_TRANSACTION_GROUP_PREFIX}x" in h5_file


def test_validate_workspace_h5_file_rejects_non_workspace(tmp_path) -> None:
    import h5py

    fname = tmp_path / "invalid.h5"
    with h5py.File(fname, "w"):
        pass

    with pytest.raises(ValueError, match="not valid"):
        manager_workspace._validate_workspace_h5_file(fname)


def test_fsync_parent_directory_skips_non_posix(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(manager_workspace.os, "name", "nt")
    monkeypatch.setattr(
        manager_workspace.os,
        "open",
        lambda *args, **kwargs: pytest.fail("non-posix platforms should not fsync"),
    )

    manager_workspace._fsync_parent_directory(tmp_path / "workspace.itws")


def test_manager_workspace_v4_save_open_replaces_and_binds_path(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        child = itool(data + 1, manager=False, execute=False)
        assert isinstance(child, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(child, 0, show=False)

        fname = tmp_path / "bound.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        assert manager.workspace_path == str(fname.resolve())
        assert not manager.is_workspace_modified

        with h5py.File(fname, "r") as h5_file:
            assert h5_file.attrs["imagetool_workspace_schema_version"] == 4
            manifest = json.loads(h5_file.attrs["imagetool_workspace_manifest"])
        assert manifest["schema_version"] == 4
        assert {node["uid"] for node in manifest["nodes"]} >= {
            manager._imagetool_wrappers[0].uid,
            child_uid,
        }

        extra = itool(data + 2, manager=False, execute=False)
        assert isinstance(extra, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(extra, show=False)
        assert manager.ntools == 2

        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        assert manager.workspace_path == str(fname.resolve())
        assert not manager.is_workspace_modified
        assert manager._imagetool_wrappers[0]._childtool_indices == [child_uid]
        assert manager.get_imagetool(0).slicer_area._data.chunks is None
        assert _compute_first_value(manager.get_imagetool(0).slicer_area._data) == 0


def test_manager_workspace_import_appends_without_reassociation(
    qtbot,
    monkeypatch,
    accept_dialog,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    choose_dialog_calls = {"count": 0}

    class _SelectSecondDialog(_ChooseFromDataTreeDialog):
        def __init__(self, *args, **kwargs) -> None:
            choose_dialog_calls["count"] += 1
            super().__init__(*args, **kwargs)
            first_item = self._tree_widget.topLevelItem(0)
            assert first_item is not None
            first_item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)

        def exec(self) -> QtWidgets.QDialog.DialogCode:
            return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(
        erlab.interactive.imagetool.manager._mainwindow,
        "_ChooseFromDataTreeDialog",
        _SelectSecondDialog,
    )

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        base_tool = itool(data, manager=False, execute=False)
        assert isinstance(base_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(base_tool, show=False)

        current_fname = tmp_path / "current.itws"
        manager._save_workspace_document(current_fname, force_full=True)
        manager._adopt_workspace_path(current_fname)
        manager._mark_workspace_clean()

        import_tool = itool(data + 1, manager=False, execute=False)
        assert isinstance(import_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(import_tool, show=False)
        import_fname = tmp_path / "import.itws"
        manager._save_workspace_document(import_fname, force_full=True)

        manager.remove_imagetool(1)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        manager._mark_workspace_clean()

        def _go_to_file(dialog: QtWidgets.QFileDialog):
            dialog.setDirectory(str(tmp_path))
            dialog.selectFile(str(import_fname))
            focused = dialog.focusWidget()
            if isinstance(focused, QtWidgets.QLineEdit):
                focused.setText(import_fname.name)

        accept_dialog(
            lambda: manager.import_workspace(native=False),
            pre_call=_go_to_file,
        )

        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        assert choose_dialog_calls["count"] == 1
        assert manager.workspace_path == str(current_fname.resolve())
        assert manager.is_workspace_modified


def test_manager_workspace_save_as_preserves_live_in_memory_windows(
    qtbot,
    accept_dialog,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    dask_options = erlab.interactive.options.model.io.dask
    old_threshold = dask_options.compute_threshold
    object.__setattr__(dask_options, "compute_threshold", 0)
    try:
        with manager_context() as manager:
            qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
            data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])

            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)
            child = itool(data + 1.0, manager=False, execute=False)
            assert isinstance(child, erlab.interactive.imagetool.ImageTool)
            child_uid = manager.add_imagetool_child(child, 0, show=False)

            def _load_workspace_file_should_not_run(*args, **kwargs):
                raise AssertionError("Save As should not reload the saved workspace")

            monkeypatch.setattr(
                manager, "_load_workspace_file", _load_workspace_file_should_not_run
            )

            new_fname = tmp_path / "new.itws"

            def _go_to_file(dialog: QtWidgets.QFileDialog):
                dialog.setDirectory(str(tmp_path))
                dialog.selectFile(str(new_fname))
                focused = dialog.focusWidget()
                if isinstance(focused, QtWidgets.QLineEdit):
                    focused.setText(new_fname.name)

            accept_dialog(lambda: manager.save_as(native=False), pre_call=_go_to_file)

            assert manager.workspace_path == str(new_fname.resolve())
            assert not manager.is_workspace_modified
            assert manager.get_imagetool(0) is root
            assert manager._child_node(child_uid).imagetool is child
            assert manager._imagetool_wrappers[0]._childtool_indices == [child_uid]
            assert root.slicer_area._data.chunks is None
            assert child.slicer_area._data.chunks is None
    finally:
        object.__setattr__(dask_options, "compute_threshold", old_threshold)


def test_manager_workspace_full_save_preserves_non_dask_data(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    dask_options = erlab.interactive.options.model.io.dask
    old_threshold = dask_options.compute_threshold
    object.__setattr__(dask_options, "compute_threshold", 0)
    try:
        with manager_context() as manager:
            qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
            data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])

            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)
            assert root.slicer_area._data.chunks is None

            fname = tmp_path / "full-save.itws"
            manager._save_workspace_document(fname, force_full=True)
            manager._adopt_workspace_path(fname)
            manager._mark_workspace_clean()
            manager._workspace_needs_full_save = True

            assert manager.save()
            assert root.slicer_area._data.chunks is None
            assert _compute_first_value(root.slicer_area._data) == 0.0
    finally:
        object.__setattr__(dask_options, "compute_threshold", old_threshold)


def test_manager_workspace_save_as_rebinds_non_dask_file_backed_data(
    qtbot,
    accept_dialog,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])

        old_fname = tmp_path / "old.h5"
        new_fname = tmp_path / "new.itws"
        xr.DataTree.from_dict({"0/imagetool": data.to_dataset(name="data")}).to_netcdf(
            old_fname, engine="h5netcdf", invalid_netcdf=True
        )
        source = _open_external_file_backed_hdf5_imagetool_data(old_fname)
        assert source.chunks is None

        root = itool(source, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        live_data = manager.get_imagetool(0).slicer_area._data
        old_source = str(old_fname.resolve())
        assert (
            manager_xarray._normalized_file_path(live_data.encoding.get("source"))
            == old_source
        )

        def _go_to_file(dialog: QtWidgets.QFileDialog):
            dialog.setDirectory(str(tmp_path))
            dialog.selectFile(str(new_fname))
            focused = dialog.focusWidget()
            if isinstance(focused, QtWidgets.QLineEdit):
                focused.setText(new_fname.name)

        accept_dialog(lambda: manager.save_as(native=False), pre_call=_go_to_file)

        rebound = manager.get_imagetool(0).slicer_area._data
        new_source = str(new_fname.resolve())
        rebound_source = manager_xarray._normalized_file_path(
            rebound.encoding.get("source")
        )
        assert rebound_source == new_source
        assert rebound_source != old_source
        assert rebound.chunks is None

        old_fname.unlink()
        assert _compute_first_value(rebound) == 0.0


def test_manager_workspace_save_as_preserves_manually_chunked_file_backed_data(
    qtbot,
    accept_dialog,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])

        old_fname = tmp_path / "old.h5"
        new_fname = tmp_path / "new.itws"
        xr.DataTree.from_dict({"0/imagetool": data.to_dataset(name="data")}).to_netcdf(
            old_fname, engine="h5netcdf", invalid_netcdf=True
        )
        source = _open_external_file_backed_hdf5_imagetool_data(old_fname)
        assert source.chunks is None

        root = itool(source, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root.slicer_area.replace_source_data(
            root.slicer_area._data.chunk({"x": 2, "y": 2}),
            auto_compute=False,
        )
        assert root.slicer_area._data.chunks is not None

        def _go_to_file(dialog: QtWidgets.QFileDialog):
            dialog.setDirectory(str(tmp_path))
            dialog.selectFile(str(new_fname))
            focused = dialog.focusWidget()
            if isinstance(focused, QtWidgets.QLineEdit):
                focused.setText(new_fname.name)

        accept_dialog(lambda: manager.save_as(native=False), pre_call=_go_to_file)

        rebound = manager.get_imagetool(0).slicer_area._data
        assert rebound.chunks is not None
        assert manager_xarray._normalized_file_path(
            rebound.encoding.get("source")
        ) == str(new_fname.resolve())

        old_fname.unlink()
        assert _compute_first_value(rebound) == 0.0


def test_manager_workspace_save_as_rebinds_lazy_data_to_new_document(
    qtbot,
    accept_dialog,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    dask_options = erlab.interactive.options.model.io.dask
    old_threshold = dask_options.compute_threshold
    object.__setattr__(dask_options, "compute_threshold", 0)
    try:
        with manager_context() as manager:
            qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
            data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)

            old_fname = tmp_path / "old.h5"
            new_fname = tmp_path / "new.itws"
            xr.DataTree.from_dict(
                {"0/imagetool": data.to_dataset(name="data")}
            ).to_netcdf(old_fname, engine="h5netcdf", invalid_netcdf=True)
            old_lazy = _open_external_lazy_hdf5_imagetool_data(old_fname)
            root.slicer_area.replace_source_data(old_lazy + 0, auto_compute=False)
            assert _compute_first_value(old_lazy) == 0

            def _load_workspace_file_should_not_run(*args, **kwargs):
                raise AssertionError("Save As should not reload the saved workspace")

            monkeypatch.setattr(
                manager, "_load_workspace_file", _load_workspace_file_should_not_run
            )

            def _go_to_file(dialog: QtWidgets.QFileDialog):
                dialog.setDirectory(str(tmp_path))
                dialog.selectFile(str(new_fname))
                focused = dialog.focusWidget()
                if isinstance(focused, QtWidgets.QLineEdit):
                    focused.setText(new_fname.name)

            accept_dialog(lambda: manager.save_as(native=False), pre_call=_go_to_file)

            assert manager.workspace_path == str(new_fname.resolve())
            rebound = manager.get_imagetool(0).slicer_area._data
            assert rebound.chunks is not None
            assert manager_xarray._normalized_file_path(
                rebound.encoding.get("source")
            ) == str(new_fname.resolve())
            old_fname.unlink()
            assert _compute_first_value(rebound) == 0
    finally:
        object.__setattr__(dask_options, "compute_threshold", old_threshold)


def test_manager_workspace_dirty_markers_are_node_scoped(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        child = itool(data + 1, manager=False, execute=False)
        assert isinstance(child, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(child, 0, show=False)

        fname = tmp_path / "dirty.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        assert not manager.is_workspace_modified
        assert not root.isWindowModified()
        assert not child.isWindowModified()

        manager._child_node(child_uid).name = "renamed child"
        assert manager.is_workspace_modified
        assert manager.isWindowModified()
        assert not root.isWindowModified()
        assert child.isWindowModified()
        details = manager._dirty_details_text()
        assert "State modified:\n- renamed child" in details
        assert "Data modified:" not in details


def test_manager_workspace_save_clears_deferred_dirty_events(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._imagetool_wrappers[0].uid

        fname = tmp_path / "deferred-dirty.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()

        QtCore.QTimer.singleShot(0, lambda: manager._mark_node_state_dirty(uid))
        manager._mark_node_state_dirty(uid)
        assert manager.is_workspace_modified
        assert root.isWindowModified()

        focus_restored: list[QtWidgets.QWidget | None] = []
        monkeypatch.setattr(manager, "_active_managed_window", lambda: root)
        monkeypatch.setattr(
            manager,
            "_restore_focus_after_workspace_save",
            lambda origin: focus_restored.append(origin),
        )
        monkeypatch.setattr(
            manager,
            "_open_workspace_save_wait_dialog",
            lambda *args, **kwargs: pytest.fail("regular Save should not be modal"),
        )

        assert manager.save()
        manager._drain_workspace_deferred_events()
        assert not manager.is_workspace_modified
        assert not root.isWindowModified()
        assert focus_restored == [root]


def test_manager_workspace_state_save_updates_attrs_without_full_rewrite(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._imagetool_wrappers[0].uid

        fname = tmp_path / "state-delta.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()
        manager._mark_node_state_dirty(uid)

        original_transaction_write = manager_workspace._write_workspace_transaction_file
        attr_write_calls: list[str] = []

        def _record_transaction_write(
            _fname: str | os.PathLike[str],
            rewrite_groups: Iterable[manager_workspace._WorkspaceRewriteGroup],
            attr_updates: Iterable[manager_workspace._WorkspaceAttrUpdate],
            root_attrs: Mapping[str, typing.Any],
        ) -> None:
            rewrite_groups = tuple(rewrite_groups)
            updates = tuple(attr_updates)
            assert rewrite_groups == ()
            attr_write_calls.extend(update.payload_path for update in updates)
            original_transaction_write(_fname, rewrite_groups, updates, root_attrs)

        monkeypatch.setattr(
            manager_workspace,
            "_write_full_workspace_tree_file",
            lambda *args, **kwargs: pytest.fail(
                "state-only Save should not rewrite the full workspace"
            ),
        )
        monkeypatch.setattr(
            manager_workspace,
            "_write_workspace_root_attrs_to_file",
            lambda *args, **kwargs: pytest.fail(
                "state-only Save should batch root attrs with node attrs"
            ),
        )
        monkeypatch.setattr(
            manager_workspace,
            "_write_workspace_transaction_file",
            _record_transaction_write,
        )
        monkeypatch.setattr(
            manager,
            "_open_workspace_save_wait_dialog",
            lambda *args, **kwargs: pytest.fail("state-only Save should be fast"),
        )

        assert manager.save()
        assert attr_write_calls == ["0/imagetool"]
        assert not manager.is_workspace_modified
        assert not root.isWindowModified()


def test_manager_workspace_save_does_not_close_live_workspace_handles(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._imagetool_wrappers[0].uid

        fname = tmp_path / "live-handles.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()
        manager._mark_node_state_dirty(uid)

        assert manager.save()

        manager._mark_node_data_dirty(uid)
        assert manager.save()


def test_manager_workspace_save_preserves_live_lazy_readers_during_write(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "live-lazy.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        root = manager.get_imagetool(0)
        uid = manager._imagetool_wrappers[0].uid
        root.slicer_area.replace_source_data(
            manager._workspace_rebind_data_for_uid(fname, uid, chunks="auto"),
            auto_compute=False,
        )
        live_data = root.slicer_area._data
        assert live_data.chunks is not None
        assert _compute_first_value(live_data) == 0.0

        manager._mark_node_state_dirty(uid)
        original_write = manager_workspace._write_workspace_transaction_file
        computed_values: list[object] = []

        def _slow_write_workspace_transaction_file(*args, **kwargs):
            time.sleep(0.05)
            return original_write(*args, **kwargs)

        def _compute_live_data() -> None:
            computed_values.append(live_data.isel({"x": 1, "y": 1}).compute().item())

        monkeypatch.setattr(
            manager_workspace,
            "_write_workspace_transaction_file",
            _slow_write_workspace_transaction_file,
        )
        QtCore.QTimer.singleShot(10, _compute_live_data)

        assert manager.save()
        assert computed_values == [6.0]


def test_manager_workspace_save_shows_wait_dialog_when_actual_save_is_slow(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "slow-save.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()
        manager._mark_node_state_dirty(manager._imagetool_wrappers[0].uid)

        original_write = manager_workspace._write_workspace_transaction_file

        def _slow_write_workspace_transaction_file(*args, **kwargs):
            time.sleep(0.05)
            return original_write(*args, **kwargs)

        wait_calls: list[QtWidgets.QWidget] = []

        def _fake_open_wait_dialog(parent: QtWidgets.QWidget) -> QtWidgets.QDialog:
            wait_calls.append(parent)
            return QtWidgets.QDialog(parent)

        focus_restored: list[QtWidgets.QWidget | None] = []
        monkeypatch.setattr(
            manager_mainwindow,
            "_WORKSPACE_SAVE_WAIT_DIALOG_THRESHOLD_SECONDS",
            0.01,
        )
        monkeypatch.setattr(
            manager_workspace,
            "_write_workspace_transaction_file",
            _slow_write_workspace_transaction_file,
        )
        monkeypatch.setattr(manager, "_active_managed_window", lambda: root)
        monkeypatch.setattr(
            manager,
            "_restore_focus_after_workspace_save",
            lambda origin: focus_restored.append(origin),
        )
        monkeypatch.setattr(
            manager,
            "_open_workspace_save_wait_dialog",
            _fake_open_wait_dialog,
        )

        assert manager.save()
        assert wait_calls == [root]
        assert focus_restored == [root]


def test_manager_workspace_save_keeps_post_command_changes_dirty(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._imagetool_wrappers[0].uid

        fname = tmp_path / "post-command-dirty.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()
        manager._mark_node_data_dirty(uid)

        def _fake_run_workspace_save_worker(
            _fname: str | os.PathLike[str],
            snapshot: manager_workspace._WorkspaceSaveSnapshot,
            _origin: QtWidgets.QWidget | None,
        ) -> tuple[bool, float, str]:
            snapshot.close()
            manager._mark_node_state_dirty(uid)
            return True, 0.0, ""

        monkeypatch.setattr(
            manager,
            "_run_workspace_save_worker",
            _fake_run_workspace_save_worker,
        )

        assert manager.save()
        assert manager.is_workspace_modified
        assert root.isWindowModified()
        details = manager._dirty_details_text()
        assert "State modified:" in details
        assert "Data modified:" not in details


def test_manager_workspace_compact_resets_delta_count_and_cleans_internal_groups(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._imagetool_wrappers[0].uid

        fname = tmp_path / "compact.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()
        manager._mark_node_state_dirty(uid)
        assert manager.save()
        assert manager._workspace_delta_save_count == 1

        monkeypatch.setattr(
            erlab.interactive.utils,
            "wait_dialog",
            lambda *args, **kwargs: contextlib.nullcontext(),
        )

        assert manager.compact_workspace()
        assert manager._workspace_delta_save_count == 0
        _assert_no_workspace_internal_groups(fname)
        with h5py.File(fname, "r") as h5_file:
            assert (
                manager_workspace._workspace_delta_save_count_from_attrs(h5_file.attrs)
                == 0
            )
            manifest = manager_workspace._workspace_manifest_from_attrs(h5_file.attrs)
            assert "delta_save_count" not in manifest
            assert "transaction_protocol" not in manifest


def test_manager_workspace_shutdown_compacts_clean_delta_workspace(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._imagetool_wrappers[0].uid

        fname = tmp_path / "shutdown-compact.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()
        manager._mark_node_state_dirty(uid)
        assert manager.save()
        assert manager._workspace_delta_save_count == 1

        manager._compact_workspace_before_shutdown()

        assert manager._workspace_delta_save_count == 0
        with h5py.File(fname, "r") as h5_file:
            assert (
                manager_workspace._workspace_delta_save_count_from_attrs(h5_file.attrs)
                == 0
            )
            manifest = manager_workspace._workspace_manifest_from_attrs(h5_file.attrs)
            assert "delta_save_count" not in manifest
            assert "transaction_protocol" not in manifest


def test_manager_workspace_shutdown_compact_skips_dirty_workspace(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._imagetool_wrappers[0].uid

        fname = tmp_path / "dirty-shutdown-compact.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()
        manager._workspace_delta_save_count = 1
        manager._mark_node_state_dirty(uid)

        monkeypatch.setattr(
            manager,
            "_save_workspace_document",
            lambda *args, **kwargs: pytest.fail(
                "Dirty shutdown compaction should not write discarded changes"
            ),
        )

        manager._compact_workspace_before_shutdown()


def test_manager_workspace_high_risk_path_forces_full_save_snapshot(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._imagetool_wrappers[0].uid

        fname = tmp_path / "high-risk.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()
        manager._mark_node_state_dirty(uid)
        monkeypatch.setattr(
            manager_workspace, "_workspace_path_is_high_risk", lambda *_args: True
        )

        snapshot = manager._workspace_save_snapshot(fname)
        try:
            assert snapshot.full_tree is not None
            assert snapshot.delta_save_count == 0
        finally:
            snapshot.close()


def test_workspace_remote_incremental_option_allows_delta_save(
    monkeypatch,
    tmp_path,
) -> None:
    options = erlab.interactive.options
    incremental_name = "io/workspace/incremental_save_on_remote"
    use_incremental_name = "io/workspace/use_incremental"
    old_remote_value = options[incremental_name]
    old_incremental_value = options[use_incremental_name]
    fname = tmp_path / "remote-incremental.itws"
    fname.touch()
    monkeypatch.setattr(
        manager_workspace, "_workspace_path_is_high_risk", lambda *_args: True
    )
    options[incremental_name] = True
    options[use_incremental_name] = True
    try:
        assert not manager_workspace._workspace_requires_full_save(
            fname,
            needs_full_save=False,
            schema_version=manager_workspace._current_workspace_schema_version(),
            structure_modified=False,
            has_dirty_added=False,
            has_dirty_removed=False,
        )
    finally:
        options[incremental_name] = old_remote_value
        options[use_incremental_name] = old_incremental_value


def test_manager_application_quit_suppresses_child_visibility_dirty(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=True)
        qtbot.wait_until(root.isVisible)

        fname = tmp_path / "quit-clean.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._drain_workspace_deferred_events()
        manager._mark_workspace_clean()
        assert not manager.is_workspace_modified

        manager._application_quit_requested = True
        root.hide()
        manager._drain_workspace_deferred_events()

        assert not manager.is_workspace_modified
        assert not root.isWindowModified()


def test_manager_application_quit_filter_routes_quit_to_manager(
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        calls: list[str] = []
        monkeypatch.setattr(
            manager,
            "_handle_application_quit_request",
            lambda: calls.append("quit") or True,
        )

        event = QtCore.QEvent(QtCore.QEvent.Type.Quit)
        assert manager._application_quit_filter is not None
        assert manager._application_quit_filter.eventFilter(None, event)
        assert calls == ["quit"]


def test_manager_workspace_dirty_marker_not_saved_in_titles(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25).reshape((5, 5)), dims=["x", "y"], name="source"
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root_uid = manager._imagetool_wrappers[0].uid
        tool = DerivativeTool(data)
        tool_uid = manager.add_childtool(tool, 0, show=False)

        expect_title_placeholder = sys.platform != "darwin"
        assert ("[*]" in root.windowTitle()) is expect_title_placeholder
        assert ("[*]" in tool.windowTitle()) is expect_title_placeholder

        root.setWindowTitle("stale root title[*]")
        manager._imagetool_wrappers[0].update_title()
        assert "stale root title" in manager._imagetool_wrappers[0].label_text
        assert "[*]" not in manager._imagetool_wrappers[0].label_text

        root.setWindowTitle("stale root title[*]")
        tool.setWindowTitle("stale tool title[*]")
        manager._set_node_window_modified(root_uid, True)
        manager._set_node_window_modified(tool_uid, True)

        assert (
            root.windowTitle()
            == manager_mainwindow._window_title_with_modified_placeholder(
                manager._imagetool_wrappers[0].label_text
            )
        )
        assert (
            tool.windowTitle()
            == manager_mainwindow._window_title_with_modified_placeholder(
                f"{tool.tool_name}: {tool._tool_display_name}"
            )
        )

        fname = tmp_path / "titles.itws"
        manager._save_workspace_document(fname, force_full=True)

        import h5py

        with h5py.File(fname, "r") as h5_file:
            root_title = h5_file["0/imagetool"].attrs["itool_title"]
            tool_title = h5_file[f"0/childtools/{tool_uid}/tool"].attrs["tool_title"]

        assert "[*]" not in root_title
        assert "[*]" not in tool_title


def test_manager_workspace_save_preserves_reordered_roots(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        for value in range(3):
            data = xr.DataArray(
                np.full((5, 5), value), dims=["x", "y"], name=f"data_{value}"
            )
            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)

        fname = tmp_path / "ordered.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()

        model = manager.tree_view._model
        assert model.dropMimeData(
            model.mimeData([model.index(0, 0)]),
            QtCore.Qt.DropAction.MoveAction,
            model.rowCount(),
            0,
            QtCore.QModelIndex(),
        )
        assert manager._displayed_indices == [1, 2, 0]
        assert manager.is_workspace_modified
        assert manager.save()

        with h5py.File(fname, "r") as h5_file:
            manifest = json.loads(h5_file.attrs["imagetool_workspace_manifest"])
        assert manifest["root_order"] == [1, 2, 0]

        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)
        loaded_order = [
            int(manager.get_imagetool(index).slicer_area._data.values[0, 0])
            for index in manager._displayed_indices
        ]
        assert loaded_order == [1, 2, 0]


def test_manager_workspace_child_save_shortcuts_call_manager_save(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()
        calls: list[bool] = []

        def _fake_save(*, native: bool = True) -> bool:
            calls.append(native)
            return True

        monkeypatch.setattr(manager, "save", _fake_save)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=True)
        root_shortcuts = root.findChildren(QtWidgets.QShortcut)
        root_save = [
            shortcut
            for shortcut in root_shortcuts
            if shortcut.objectName() == "managerWorkspaceSaveShortcut"
        ]
        assert len(root_save) == 1
        root_save[0].activated.emit()

        child = itool(data + 1, manager=False, execute=False)
        assert isinstance(child, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool_child(child, 0, show=False)
        child_save = [
            shortcut
            for shortcut in child.findChildren(QtWidgets.QShortcut)
            if shortcut.objectName() == "managerWorkspaceSaveShortcut"
        ]
        assert len(child_save) == 1
        child_save[0].activated.emit()

        tool = DerivativeTool(data)
        manager.add_childtool(tool, 0, show=False)
        tool_save = [
            shortcut
            for shortcut in tool.findChildren(QtWidgets.QShortcut)
            if shortcut.objectName() == "managerWorkspaceSaveShortcut"
        ]
        assert len(tool_save) == 1
        tool_save[0].activated.emit()

        assert calls == [True, True, True]


def test_manager_workspace_delta_save_splits_state_and_data_writes(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        fname = tmp_path / "delta.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)

        dataset_writes: list[str | None] = []
        original_to_netcdf = xr.Dataset.to_netcdf

        def _to_netcdf_spy(self, *args, **kwargs):
            dataset_writes.append(kwargs.get("group"))
            return original_to_netcdf(self, *args, **kwargs)

        monkeypatch.setattr(xr.Dataset, "to_netcdf", _to_netcdf_spy)

        manager.rename_imagetool(0, "state only")
        assert manager.save()
        assert dataset_writes == []

        replacement = data.copy(deep=True)
        replacement.data = np.asarray(replacement.data) + 10
        root.slicer_area.replace_source_data(replacement)
        assert manager.save()
        assert any(
            group is not None
            and group.startswith(
                f"/{manager_workspace._WORKSPACE_PENDING_GROUP_PREFIX}"
            )
            and group.endswith("/imagetool")
            for group in dataset_writes
        )


def test_manager_workspace_rejects_external_xarray_reader_for_active_workspace(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "lazy-state.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        lazy = _open_external_lazy_hdf5_imagetool_data(fname)
        manager.get_imagetool(0).slicer_area.set_data(lazy, auto_compute=False)
        assert _compute_first_value(manager.get_imagetool(0).slicer_area._data) == 0
        manager._mark_workspace_clean()

        errors: list[str] = []
        monkeypatch.setattr(manager, "_show_workspace_save_worker_error", errors.append)

        manager.rename_imagetool(0, "lazy state")
        assert not manager.save()
        assert errors
        with contextlib.suppress(Exception):
            lazy.close()


def test_manager_workspace_lazy_data_delta_save_uses_pending_group_before_replacing(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "lazy-data.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        replacement = data.copy(deep=True)
        replacement.data = np.asarray(replacement.data) + 10
        manager.get_imagetool(0).slicer_area.replace_source_data(
            replacement, auto_compute=False
        )
        assert manager.save()
        assert list(tmp_path.glob("lazy-data.itws.delta-*")) == []

        import h5py

        with h5py.File(fname, "r") as h5_file:
            saved = h5_file["0/imagetool"][manager_mainwindow._ITOOL_DATA_NAME]
            assert saved[0, 0] == 10


def test_manager_workspace_lazy_data_delta_pending_failure_preserves_old_group(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "lazy-failure.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()

        replacement = data.copy(deep=True)
        replacement.data = np.asarray(replacement.data) + 10
        root.slicer_area.replace_source_data(replacement, auto_compute=False)

        def _write_partial_pending_then_raise(
            fname: str | os.PathLike[str],
            _constructor: Mapping[str, xr.Dataset],
            _group_path: str,
            pending_path: str,
        ) -> None:
            with manager_workspace._open_workspace_h5_file_for_update(fname) as h5_file:
                h5_file.create_group(pending_path)
            raise RuntimeError("pending write failed")

        monkeypatch.setattr(
            manager_workspace,
            "_write_workspace_constructor_groups_to_pending",
            _write_partial_pending_then_raise,
        )
        monkeypatch.setattr(
            manager, "_show_workspace_save_worker_error", lambda *args: None
        )

        assert not manager.save()
        import h5py

        with h5py.File(fname, "r") as h5_file:
            saved = h5_file["0/imagetool"][manager_mainwindow._ITOOL_DATA_NAME]
            assert saved[0, 0] == 0
            assert not any(
                name.startswith(manager_workspace._WORKSPACE_INTERNAL_GROUP_PREFIXES)
                for name in h5_file
            )


def test_manager_workspace_stale_pending_groups_do_not_poison_open_or_save(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "stale-pending.itws"
        manager._save_workspace_document(fname, force_full=True)
        with h5py.File(fname, "a") as h5_file:
            h5_file.create_group(
                f"{manager_workspace._WORKSPACE_PENDING_GROUP_PREFIX}stale"
            )
            h5_file.create_group(
                f"{manager_workspace._WORKSPACE_BACKUP_GROUP_PREFIX}stale"
            )

        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        manager.rename_imagetool(0, "cleaned")
        assert manager.save()
        with h5py.File(fname, "r") as h5_file:
            assert not any(
                name.startswith(manager_workspace._WORKSPACE_INTERNAL_GROUP_PREFIXES)
                for name in h5_file
            )


def test_manager_workspace_load_dialog_skips_stale_internal_groups(
    qtbot,
    accept_dialog,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "stale-dialog.itws"
        manager._save_workspace_document(fname, force_full=True)
        with h5py.File(fname, "a") as h5_file:
            h5_file.create_group(
                f"{manager_workspace._WORKSPACE_PENDING_GROUP_PREFIX}stale"
            )
            h5_file.create_group(
                f"{manager_workspace._WORKSPACE_BACKUP_GROUP_PREFIX}stale"
            )

        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        tree = manager_xarray.open_workspace_datatree(fname, chunks="auto")
        accept_dialog(
            lambda: manager._from_datatree(
                tree,
                replace=True,
                mark_dirty=False,
                select=True,
            )
        )

        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)


def test_manager_workspace_replace_load_failure_restores_previous_workspace(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        current_fname = tmp_path / "current.itws"
        manager._save_workspace_document(current_fname, force_full=True)
        manager._adopt_workspace_path(current_fname)
        manager._mark_workspace_clean()

        broken_fname = tmp_path / "broken.itws"
        with h5py.File(broken_fname, "w") as h5_file:
            h5_file.attrs["imagetool_workspace_schema_version"] = 4
            h5_file.create_group("0")

        with pytest.raises(ValueError, match="Workspace node"):
            manager._load_workspace_file(
                broken_fname,
                replace=True,
                associate=True,
                mark_dirty=False,
                select=False,
            )

        assert manager.workspace_path == str(current_fname.resolve())
        assert manager.ntools == 1
        xarray.testing.assert_equal(manager.get_imagetool(0).slicer_area._data, data)
        assert not manager.is_workspace_modified


def test_manager_workspace_load_visible_windows_stays_clean_after_events(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=True)
        qtbot.wait_until(root.isVisible)

        fname = tmp_path / "visible.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()

        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        for _ in range(3):
            QtWidgets.QApplication.sendPostedEvents(None, 0)
            QtWidgets.QApplication.processEvents()

        assert not manager.is_workspace_modified


def test_manager_workspace_delta_save_persists_geometry_changes(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "geometry.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()

        root.setGeometry(12, 34, 321, 234)
        qtbot.wait_until(lambda: manager.is_workspace_modified, timeout=5000)
        expected_rect = tuple(root.geometry().getRect())

        assert manager.save()
        with h5py.File(fname, "r") as h5_file:
            saved_rect = tuple(
                int(value) for value in h5_file["0/imagetool"].attrs["itool_rect"]
            )
        assert saved_rect == expected_rect


def test_manager_workspace_roundtrip_goldtool_child(
    qtbot,
    monkeypatch,
    gold,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()

        itool(gold, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child = GoldTool(gold.copy(deep=True), data_name="gold_input")
        child.set_source_binding(erlab.interactive.imagetool.provenance.full_data())
        child_uid = manager.add_childtool(child, 0, show=False)
        configure_goldtool_child(child, fitted=True, spline=True)

        expected_status = child.tool_status.model_copy(deep=True)
        expected_corrected = child.corrected.copy(deep=True)
        expected_source_spec = child.source_spec
        child.open_itool()
        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)
        output_uid = child_node._childtool_indices[0]

        tree = manager._to_datatree()
        assert (
            tree[f"0/childtools/{child_uid}/tool"].attrs["manager_node_uid"]
            == child_uid
        )

        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        for node in tree.values():
            manager._load_workspace_node(typing.cast("xr.DataTree", node))

        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        loaded_root = manager._imagetool_wrappers[0]
        assert loaded_root._childtool_indices == [child_uid]

        loaded_child = manager.get_childtool(child_uid)
        assert isinstance(loaded_child, GoldTool)
        assert loaded_child.source_spec == expected_source_spec
        assert loaded_child.tool_status == expected_status
        xr.testing.assert_identical(loaded_child.corrected, expected_corrected)
        loaded_child_node = manager._child_node(child_uid)
        assert loaded_child_node._childtool_indices == [output_uid]
        loaded_output_node = manager._child_node(output_uid)
        assert loaded_output_node.output_id == "goldtool.corrected"
        assert loaded_output_node.source_spec is None
        assert loaded_output_node.provenance_spec is not None
        assert loaded_output_node.provenance_spec.active_name == "corrected"
        xr.testing.assert_identical(fetch(output_uid), expected_corrected)

        monkeypatch.setattr(
            loaded_child, "_prompt_existing_output_imagetool", lambda: "update"
        )
        loaded_child.open_itool()
        assert loaded_child_node._childtool_indices == [output_uid]


def test_manager_workspace_roundtrip_dtool_child(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()

        itool(test_data, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 1, timeout=5000
        )

        child_uid = manager._imagetool_wrappers[0]._childtool_indices[0]
        child = manager.get_childtool(child_uid)
        assert isinstance(child, DerivativeTool)

        expected_status = child.tool_status.model_copy(deep=True)
        expected_result = child.result.T.copy(deep=True)
        expected_source_spec = child.source_spec
        child.open_itool()
        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)
        output_uid = child_node._childtool_indices[0]

        tree = manager._to_datatree()
        assert (
            tree[f"0/childtools/{child_uid}/tool"].attrs["manager_node_uid"]
            == child_uid
        )

        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        for node in tree.values():
            manager._load_workspace_node(typing.cast("xr.DataTree", node))

        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        loaded_root = manager._imagetool_wrappers[0]
        assert loaded_root._childtool_indices == [child_uid]

        loaded_child = manager.get_childtool(child_uid)
        assert isinstance(loaded_child, DerivativeTool)
        assert loaded_child.source_spec == expected_source_spec
        assert loaded_child.tool_status == expected_status
        xr.testing.assert_identical(loaded_child.result.T, expected_result)
        loaded_child_node = manager._child_node(child_uid)
        assert loaded_child_node._childtool_indices == [output_uid]
        loaded_output_node = manager._child_node(output_uid)
        assert loaded_output_node.output_id == "dtool.result"
        assert loaded_output_node.source_spec is None
        assert loaded_output_node.provenance_spec is not None
        assert loaded_output_node.provenance_spec.active_name == "result"
        xr.testing.assert_identical(fetch(output_uid), expected_result)

        monkeypatch.setattr(
            loaded_child, "_prompt_existing_output_imagetool", lambda: "update"
        )
        loaded_child.open_itool()
        assert loaded_child_node._childtool_indices == [output_uid]


def test_manager_workspace_roundtrip_fit1d_child(
    qtbot,
    monkeypatch,
    exp_decay_model,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()

        itool(test_data, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        t = np.linspace(0.0, 4.0, 25)
        data = xr.DataArray(
            3.0 * np.exp(-t / 2.0), dims=("t",), coords={"t": t}, name="decay"
        )
        params = exp_decay_model.make_params(n0=2.0, tau=1.0)
        child = erlab.interactive.ftool(
            data, model=exp_decay_model, params=params, execute=False
        )
        assert isinstance(child, Fit1DTool)
        child_uid = manager.add_childtool(child, 0, show=False)

        assert child._run_fit()
        qtbot.wait_until(lambda: child._last_result_ds is not None, timeout=10000)
        assert child._last_result_ds is not None
        expected_fit_ds = child._last_result_ds.copy(deep=True)
        expected_status = child.tool_status.model_dump()

        tree = manager._to_datatree()
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        for node in tree.values():
            manager._load_workspace_node(typing.cast("xr.DataTree", node))

        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        loaded_child = manager.get_childtool(child_uid)
        assert isinstance(loaded_child, Fit1DTool)
        assert loaded_child._last_result_ds is not None
        assert_fit_result_dataset_equivalent(
            loaded_child._last_result_ds, expected_fit_ds
        )
        assert loaded_child.tool_status.model_dump() == expected_status
        assert loaded_child._fit_is_current
        assert loaded_child.save_button.isEnabled()
        assert loaded_child.copy_button.isEnabled()

        warnings: list[tuple[str, str]] = []
        monkeypatch.setattr(
            loaded_child,
            "_show_warning",
            lambda title, text: warnings.append((title, text)),
        )
        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)
        copied = copy_full_code_for_uid(monkeypatch, manager, child_uid)
        assert "modelfit" in copied
        assert not warnings


def test_manager_workspace_roundtrip_fit2d_child(
    qtbot,
    monkeypatch,
    exp_decay_model,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()

        itool(test_data, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child_uid, child = make_fit2d_child(manager, 0, exp_decay_model)
        child.timeout_spin.setValue(30.0)
        child.nfev_spin.setValue(0)
        child.y_index_spin.setValue(child.y_min_spin.value())
        child._run_fit_2d("up")
        qtbot.wait_until(
            lambda: all(ds is not None for ds in child._result_ds_full),
            timeout=10000,
        )
        expected_results = [
            None if ds is None else ds.copy(deep=True) for ds in child._result_ds_full
        ]
        expected_status = child.tool_status.model_dump()

        tree = manager._to_datatree()
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        for node in tree.values():
            manager._load_workspace_node(typing.cast("xr.DataTree", node))

        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        loaded_child = manager.get_childtool(child_uid)
        assert isinstance(loaded_child, Fit2DTool)
        assert all(ds is not None for ds in loaded_child._result_ds_full)
        assert_fit_result_list_equivalent(
            loaded_child._result_ds_full, expected_results
        )
        assert loaded_child.tool_status.model_dump() == expected_status
        assert loaded_child._fit_is_current
        assert loaded_child.copy_full_button.isEnabled()
        assert loaded_child.save_full_button.isEnabled()

        warnings: list[tuple[str, str]] = []
        monkeypatch.setattr(
            loaded_child,
            "_show_warning",
            lambda title, text: warnings.append((title, text)),
        )
        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)
        copied = copy_full_code_for_uid(monkeypatch, manager, child_uid)
        assert "modelfit" in copied
        assert not warnings


def test_manager_workspace_roundtrip_recursive_nested_imagetools(
    qtbot,
    accept_dialog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(60).reshape((3, 4, 5)).astype(float),
        dims=["x", "y", "z"],
        coords={"x": np.arange(3), "y": np.arange(4), "z": np.arange(5)},
        name="scan",
    )

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()

        itool(data, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        def _nest_average(dialog) -> None:
            dialog.dim_checks["x"].setChecked(True)
            set_transform_launch_mode(dialog, "nest")

        accept_dialog(manager.get_imagetool(0).mnb._average, pre_call=_nest_average)

        root_wrapper = manager._imagetool_wrappers[0]
        qtbot.wait_until(
            lambda: len(root_wrapper._childtool_indices) == 1, timeout=5000
        )
        child_uid = root_wrapper._childtool_indices[0]
        child_node = manager._child_node(child_uid)
        child_spec = child_node.source_spec

        child_tool = manager.get_imagetool(child_uid)
        child_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)
        tool_uid = child_node._childtool_indices[0]

        tree = manager._to_datatree()
        assert tree.attrs["imagetool_workspace_schema_version"] == 3
        assert (
            tree[f"0/childtools/{child_uid}/imagetool"].attrs["manager_node_uid"]
            == child_uid
        )
        assert (
            tree[f"0/childtools/{child_uid}/childtools/{tool_uid}/tool"].attrs[
                "manager_node_uid"
            ]
            == tool_uid
        )

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            filename = pathlib.Path(tmp_dir_name) / "workspace.itws"
            tree.to_netcdf(filename, engine="h5netcdf", invalid_netcdf=True)

            manager.remove_all_tools()
            qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

            loaded = manager_xarray.open_workspace_datatree(filename, chunks="auto")
            try:
                assert manager._is_datatree_workspace(loaded)
                assert loaded.attrs["imagetool_workspace_schema_version"] == 3
                for node in loaded.values():
                    manager._load_workspace_node(typing.cast("xr.DataTree", node))
            finally:
                loaded.close()

        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        loaded_root = manager._imagetool_wrappers[0]
        assert loaded_root._childtool_indices == [child_uid]

        loaded_child = manager._child_node(child_uid)
        assert loaded_child.source_spec == child_spec
        assert loaded_child._childtool_indices == [tool_uid]

        updated = data.copy(deep=True)
        updated.data = np.asarray(updated.data) * 4

        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated)

        qtbot.wait_until(lambda: loaded_child.source_state == "stale", timeout=5000)
        assert loaded_child._update_from_parent_source() is True
        xr.testing.assert_identical(
            manager.get_imagetool(child_uid).slicer_area._data.rename(None),
            updated.qsel.average("x").rename(None),
        )


def test_manager_workspace_load_legacy(
    qtbot,
    accept_dialog,
    datadir,
    monkeypatch,
    tmp_path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    legacy_workspace = tmp_path / "manager_workspace_legacy.h5"
    legacy_workspace.write_bytes((datadir / "manager_workspace_legacy.h5").read_bytes())

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()

        def _go_to_file(dialog: QtWidgets.QFileDialog):
            dialog.setDirectory(str(tmp_path))
            dialog.selectFile(str(legacy_workspace))
            focused = dialog.focusWidget()
            if isinstance(focused, QtWidgets.QLineEdit):
                focused.setText("manager_workspace_legacy.h5")

        legacy_notices: list[pathlib.Path] = []
        save_dialog_calls: list[tuple[bool, str, pathlib.Path | None]] = []

        def _record_legacy_notice(fname: str | os.PathLike[str]) -> None:
            legacy_notices.append(pathlib.Path(fname))

        def _record_save_dialog(
            *,
            native: bool = True,
            caption: str = "Save Workspace",
            selected_file: str | os.PathLike[str] | None = None,
        ) -> str:
            selected_path = (
                None if selected_file is None else pathlib.Path(selected_file)
            )
            save_dialog_calls.append((native, caption, selected_path))
            return str(legacy_workspace)

        monkeypatch.setattr(
            manager, "_show_legacy_workspace_upgrade_message", _record_legacy_notice
        )
        monkeypatch.setattr(manager, "_workspace_save_dialog", _record_save_dialog)

        # Load workspace
        accept_dialog(lambda: manager.load(native=False), pre_call=_go_to_file)

        assert legacy_notices == [legacy_workspace]
        assert save_dialog_calls == [
            (False, "Save Converted Workspace", legacy_workspace)
        ]

        # Check if the data is loaded
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        # Check data
        xr.testing.assert_identical(
            manager.get_imagetool(0).slicer_area._data,
            test_data,
        )

        assert not manager.is_workspace_modified
        assert manager.save()
        import h5py

        with h5py.File(legacy_workspace, "r") as h5_file:
            assert h5_file.attrs["imagetool_workspace_schema_version"] == 4

        select_tools(manager, list(manager._imagetool_wrappers.keys()))
        accept_dialog(manager.remove_action.trigger)
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        assert manager._load_workspace_file(
            legacy_workspace,
            replace=True,
            associate=True,
            mark_dirty=False,
            select=False,
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)


def test_drop_mimedata(
    qtbot,
    accept_dialog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        # Add three tools
        logger.info("Adding three tools")
        itool([data, data, data], link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)

        # Add three childtools to the first tool
        logger.info("Adding three childtools to the first tool")
        manager.get_imagetool(0).slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 1, timeout=5000
        )
        logger.info("First childtool added")
        manager.get_imagetool(0).slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 2, timeout=5000
        )
        logger.info("Second childtool added")
        manager.get_imagetool(0).slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 3, timeout=5000
        )
        logger.info("Third childtool added")
        manager.hide_all()  # Prevent windows from obstructing the manager

        # Check mimedata
        model = typing.cast("_ImageToolWrapperItemModel", manager.tree_view.model())

        # Drop None
        assert not model.canDropMimeData(None, 0, 0, 0, QtCore.QModelIndex())

        # Drop invalid mime type
        mime = QtCore.QMimeData()
        mime.setData("wrong/type", QtCore.QByteArray(b"{}"))
        assert not model.dropMimeData(
            mime, QtCore.Qt.DropAction.MoveAction, 0, 0, QtCore.QModelIndex()
        )

        # Drop invalid payload (not json)
        mime = QtCore.QMimeData()
        mime.setData(_MIME, QtCore.QByteArray(b"not json"))
        assert not model.dropMimeData(
            mime, QtCore.Qt.DropAction.MoveAction, 0, 0, QtCore.QModelIndex()
        )

        mime = QtCore.QMimeData()

        # Drop invalid payload (missing keys)
        mime.setData(
            _MIME,
            QtCore.QByteArray(
                json.dumps({"invalid_key": "invalid_value"}).encode("utf-8")
            ),
        )
        assert not model.dropMimeData(
            mime, QtCore.Qt.DropAction.MoveAction, 0, 0, QtCore.QModelIndex()
        )

        # Drop with invalid action
        assert not model.dropMimeData(
            model.mimeData([model.index(0, 0)]),  # valid mimedata
            QtCore.Qt.DropAction.CopyAction,  # invalid action
            0,
            0,
            QtCore.QModelIndex(),
        )

        # Test single selection, top-level drops
        mime_single = model.mimeData([model.index(0, 0)])
        assert model.canDropMimeData(
            mime_single, QtCore.Qt.DropAction.MoveAction, 0, 0, QtCore.QModelIndex()
        )
        # No-op drop
        assert not model.dropMimeData(
            mime_single, QtCore.Qt.DropAction.MoveAction, 0, 0, QtCore.QModelIndex()
        )

        # Drop at the end
        assert model.dropMimeData(
            mime_single,
            QtCore.Qt.DropAction.MoveAction,
            model.rowCount(),
            0,
            QtCore.QModelIndex(),
        )

        # Check new order
        assert manager._displayed_indices == [1, 2, 0]

        # No-op drop (drop on itself)
        assert not model.dropMimeData(
            model.mimeData([model.index(0, 0)]),
            QtCore.Qt.DropAction.MoveAction,
            0,
            0,
            model.index(0, 0),
        )

        # Check unchanged
        assert manager._displayed_indices == [1, 2, 0]

        # Test move child tool
        parent_wrapper = model.manager._imagetool_wrappers[0]
        child_uid: str = parent_wrapper._childtool_indices[0]
        old_order = list(parent_wrapper._childtool_indices)
        parent_index: QtCore.QModelIndex = model._row_index(0)
        child_index: QtCore.QModelIndex = model._row_index(child_uid)

        # Drop to different parent
        logger.info("Testing drop to different parent")
        assert not model.dropMimeData(
            model.mimeData([child_index]),
            QtCore.Qt.DropAction.MoveAction,
            0,
            0,
            model._row_index(1),  # Different parent
        )

        # Drop to different position in the same parent
        logger.info("Testing drop to different position in the same parent")
        assert model.dropMimeData(
            model.mimeData([child_index]),
            QtCore.Qt.DropAction.MoveAction,
            2,
            0,
            parent_index,
        )

        assert list(parent_wrapper._childtool_indices) == [
            old_order[1],
            old_order[0],
            old_order[2],
        ]

        # Test multiple selection
        logger.info("Testing multiple selection drop")
        mime_multiple = model.mimeData([model.index(0, 0), model.index(0, 1)])
        assert model.canDropMimeData(
            mime_multiple, QtCore.Qt.DropAction.MoveAction, 0, 0, QtCore.QModelIndex()
        )
        model.dropMimeData(
            mime_multiple, QtCore.Qt.DropAction.MoveAction, 0, 0, QtCore.QModelIndex()
        )

        # Test mixed top-level and childtool selection (should be rejected)
        logger.info("Testing mixed top-level and childtool selection drop")
        parent_wrapper = model.manager._imagetool_wrappers[0]
        child_uid: str = parent_wrapper._childtool_indices[0]
        mime_mixed = model.mimeData([model._row_index(0), model._row_index(child_uid)])
        assert not model.dropMimeData(
            mime_mixed, QtCore.Qt.DropAction.MoveAction, 0, 0, QtCore.QModelIndex()
        )

        # Test invalid mimedata
        logger.info("Testing invalid mimedata decoding")
        invalid_mime = QtCore.QMimeData()
        invalid_mime.setData(
            _MIME,
            QtCore.QByteArray(json.dumps({"invalid": "dictionary"}).encode("utf-8")),
        )
        assert model._decode_mime(invalid_mime) is None

        logger.info("Testing invalid mimedata decoding with non-dict payload")
        invalid_mime = QtCore.QMimeData()
        invalid_mime.setData(
            _MIME,
            QtCore.QByteArray(json.dumps("not a dict").encode("utf-8")),
        )
        assert model._decode_mime(invalid_mime) is None


def test_treeview(qtbot, accept_dialog, test_data) -> None:
    manager = ImageToolManager()

    def _cleanup_manager(widget: ImageToolManager) -> None:
        widget._workspace_loading_depth += 1
        try:
            widget.remove_all_tools()
            widget._mark_workspace_clean()
        finally:
            widget._workspace_loading_depth -= 1

    qtbot.addWidget(manager, before_close_func=_cleanup_manager)

    with qtbot.waitExposed(manager):
        manager.show()
        manager.activateWindow()

    qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

    test_data.qshow(manager=True)
    test_data.qshow(manager=True)
    qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

    manager.raise_()
    manager.activateWindow()

    model = manager.tree_view._model
    assert model.supportedDropActions() == QtCore.Qt.DropAction.MoveAction
    first_row_rect = manager.tree_view.visualRect(model.index(0, 0))

    # Click on first row
    qtbot.mouseMove(manager.tree_view.viewport(), first_row_rect.center())
    qtbot.mousePress(
        manager.tree_view.viewport(),
        QtCore.Qt.MouseButton.LeftButton,
        pos=first_row_rect.center(),
    )
    assert manager.tree_view.selected_imagetool_indices == [0]

    # Show context menu
    manager.tree_view._show_menu(first_row_rect.center())
    menu = None
    for tl in QtWidgets.QApplication.topLevelWidgets():
        if isinstance(tl, QtWidgets.QMenu):
            menu = tl
            break
    assert isinstance(menu, QtWidgets.QMenu)

    def _discard_changes(dialog: QtWidgets.QMessageBox) -> None:
        dialog.button(QtWidgets.QMessageBox.StandardButton.Discard).click()

    accept_dialog(manager.close, accept_call=_discard_changes)
    qtbot.wait_until(lambda: not erlab.interactive.imagetool.manager.is_running())


def test_childtool_remove_after_tree_clear(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        test_data.qshow(manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        uid = manager.add_childtool(erlab.interactive.utils.ToolWindow(), 0, show=False)
        qtbot.wait_until(
            lambda: uid in manager._imagetool_wrappers[0]._childtool_indices,
            timeout=5000,
        )

        manager.tree_view.clear_imagetools()
        assert manager._displayed_indices == []

        # Child destruction callbacks can arrive after top-level rows are reset.
        manager._remove_childtool(uid)
        qtbot.wait_until(
            lambda: uid not in manager._imagetool_wrappers[0]._childtools, timeout=5000
        )


def test_remove_imagetool_removes_childtools() -> None:
    uid = "child-uid-0"
    removed_uids: list[str] = []
    removed_rows: list[int] = []

    class _DummyWrapper:
        def __init__(self):
            self.uid = "root-uid-0"
            self._childtool_indices = [uid]
            self.archived = True
            self.disposed = False
            self.deleted = False

        def dispose(self):
            self.disposed = True

        def deleteLater(self):
            self.deleted = True

    wrapper = _DummyWrapper()
    manager = types.SimpleNamespace(
        _imagetool_wrappers={0: wrapper},
        _all_nodes={wrapper.uid: wrapper},
        _mark_removed_subtree_dirty=lambda _uid: None,
        _remove_uid_target=lambda child_uid: removed_uids.append(child_uid),
        tree_view=types.SimpleNamespace(
            imagetool_removed=lambda index: removed_rows.append(index)
        ),
    )

    ImageToolManager.remove_imagetool(manager, 0)
    assert removed_uids == [uid]
    assert removed_rows == [0]
    assert wrapper.disposed
    assert wrapper.deleted
    assert manager._imagetool_wrappers == {}
    assert manager._all_nodes == {}


def test_remove_imagetools_deduplicates_explicit_child_uids() -> None:
    uid0 = "child-uid-0"
    uid1 = "child-uid-1"

    manager = types.SimpleNamespace(
        _imagetool_wrappers={
            0: types.SimpleNamespace(_childtool_indices=[uid0]),
            1: types.SimpleNamespace(_childtool_indices=[uid1]),
        },
        removed_indices=[],
        removed_uids=[],
    )
    manager._bulk_remove_context = contextlib.nullcontext
    manager.remove_imagetool = lambda index, *, update_view=True: (
        manager.removed_indices.append((index, update_view))
    )
    manager._remove_childtool = lambda uid: manager.removed_uids.append(uid)

    ImageToolManager._remove_imagetools(manager, [0], child_uids=[uid0, uid1, uid1])
    assert manager.removed_indices == [(0, True)]
    assert manager.removed_uids == [uid1]


def test_remove_selected_calls_batch_remove(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        test_data.qshow(manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        uid = manager.add_childtool(erlab.interactive.utils.ToolWindow(), 0, show=False)
        qtbot.wait_until(
            lambda: uid in manager._imagetool_wrappers[0]._childtool_indices,
            timeout=5000,
        )

        manager.tree_view.expandAll()
        select_tools(manager, [0])
        select_child_tool(manager, uid)

        called: list[tuple[list[int], list[str] | None, bool]] = []

        def _remove_imagetools_spy(
            indices: list[int],
            *,
            child_uids: list[str] | None = None,
            clear_view: bool = False,
        ) -> None:
            called.append((indices, child_uids, clear_view))

        original_remove_imagetools = manager._remove_imagetools
        manager._remove_imagetools = _remove_imagetools_spy
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "exec",
            lambda _: QtWidgets.QMessageBox.StandardButton.Yes,
        )

        manager.remove_selected()
        assert called == [([0], [uid], False)]
        manager._remove_imagetools = original_remove_imagetools


def test_select_loader_options_cancel_keeps_recent_filter(
    monkeypatch,
    example_loader,
) -> None:
    class _CancelNameFilterDialog:
        def __init__(
            self, parent, valid_loaders, *, loader_extensions=None, sample_paths=None
        ) -> None:
            assert valid_loaders == {
                "Example Raw Data (*.h5)": (erlab.io.loaders["example"].load, {})
            }
            assert loader_extensions == {"Example Raw Data (*.h5)": {}}
            assert sample_paths is None
            self.checked_name = None

        def check_filter(self, name_filter: str | None) -> None:
            self.checked_name = name_filter

        def exec(self) -> bool:
            assert self.checked_name == "Example Raw Data (*.h5)"
            return False

    monkeypatch.setattr(
        manager_mainwindow, "_NameFilterDialog", _CancelNameFilterDialog
    )
    manager = types.SimpleNamespace(
        _recent_loader_extensions_by_filter={"Example Raw Data (*.h5)": {}},
        _recent_name_filter="Previous",
    )

    selected = ImageToolManager._select_loader_options(
        manager,
        {"Example Raw Data (*.h5)": (erlab.io.loaders["example"].load, {})},
        "Example Raw Data (*.h5)",
    )

    assert selected is None
    assert manager._recent_name_filter == "Previous"


def _set_default_loader_option(monkeypatch, loader_name: str) -> None:
    options = erlab.interactive.options.model
    monkeypatch.setattr(
        erlab.interactive.options,
        "model",
        options.model_copy(
            update={"io": options.io.model_copy(update={"default_loader": loader_name})}
        ),
    )


@pytest.mark.parametrize(
    ("default_loader", "recent_filter", "expected_filter"),
    [
        ("example", "xarray HDF5 Files (*.h5)", "xarray HDF5 Files (*.h5)"),
        ("example", None, "Example Raw Data (*.h5)"),
        ("example", "Missing (*.missing)", "Example Raw Data (*.h5)"),
        ("None", None, None),
    ],
)
def test_preferred_name_filter_precedence(
    monkeypatch,
    example_loader,
    default_loader: str,
    recent_filter: str | None,
    expected_filter: str | None,
) -> None:
    _set_default_loader_option(monkeypatch, default_loader)
    example_filter = "Example Raw Data (*.h5)"
    xarray_filter = "xarray HDF5 Files (*.h5)"
    valid_loaders = {
        xarray_filter: (xr.load_dataarray, {"engine": "h5netcdf"}),
        example_filter: erlab.io.loaders["example"].file_dialog_methods[example_filter],
    }
    manager = types.SimpleNamespace(_recent_name_filter=recent_filter)

    assert (
        ImageToolManager._preferred_name_filter(manager, valid_loaders)
        == expected_filter
    )


def test_preferred_name_filter_uses_default_loader_method_order(monkeypatch) -> None:
    _set_default_loader_option(monkeypatch, "merlin")
    loader_methods = erlab.io.loaders["merlin"].file_dialog_methods
    first_filter = "ALS BL4.0.3 Data (*.pxt *.ibw)"
    second_filter = "ALS BL4.0.3 Single File (*.pxt)"
    valid_loaders = {
        second_filter: loader_methods[second_filter],
        first_filter: loader_methods[first_filter],
    }
    manager = types.SimpleNamespace(_recent_name_filter=None)

    assert (
        ImageToolManager._preferred_name_filter(manager, valid_loaders) == first_filter
    )


def test_open_multiple_files_preselects_default_loader_filter(
    monkeypatch,
    tmp_path: pathlib.Path,
    example_loader,
) -> None:
    _set_default_loader_option(monkeypatch, "example")
    file_path = tmp_path / "data_002.h5"
    example_filter = "Example Raw Data (*.h5)"
    valid_loaders = {
        "xarray HDF5 Files (*.h5)": (xr.load_dataarray, {"engine": "h5netcdf"}),
        example_filter: erlab.io.loaders["example"].file_dialog_methods[example_filter],
    }
    dialogs = []

    class _CancelNameFilterDialog:
        def __init__(
            self, parent, valid_loaders, *, loader_extensions=None, sample_paths=None
        ) -> None:
            self.checked_name = None
            assert list(sample_paths or ()) == [file_path]
            dialogs.append(self)

        def check_filter(self, name_filter: str | None) -> None:
            self.checked_name = name_filter

        def exec(self) -> bool:
            return False

    manager = types.SimpleNamespace(
        _recent_loader_extensions_by_filter={},
        _recent_name_filter=None,
        _add_from_multiple_files=lambda *_args, **_kwargs: None,
        open_multiple_files=lambda *_args, **_kwargs: None,
    )
    manager._preferred_name_filter = types.MethodType(
        ImageToolManager._preferred_name_filter, manager
    )
    manager._select_loader_options = types.MethodType(
        ImageToolManager._select_loader_options, manager
    )
    monkeypatch.setattr(
        manager_mainwindow, "_NameFilterDialog", _CancelNameFilterDialog
    )
    monkeypatch.setattr(
        erlab.interactive.utils,
        "file_loaders",
        lambda *_args: valid_loaders,
    )

    ImageToolManager.open_multiple_files(manager, [file_path])

    assert dialogs[-1].checked_name == example_filter


def test_manager_open_preselects_default_loader_filter(
    monkeypatch,
    example_loader,
) -> None:
    _set_default_loader_option(monkeypatch, "example")
    example_filter = "Example Raw Data (*.h5)"
    selected_filters: list[str] = []
    real_file_dialog = QtWidgets.QFileDialog

    class _FakeFileDialog:
        AcceptMode = real_file_dialog.AcceptMode
        FileMode = real_file_dialog.FileMode
        Option = real_file_dialog.Option

        def __init__(self, parent) -> None:
            pass

        def setAcceptMode(self, mode) -> None:
            pass

        def setFileMode(self, mode) -> None:
            pass

        def setNameFilters(self, filters) -> None:
            pass

        def setOption(self, option) -> None:
            pass

        def selectNameFilter(self, selected_filter: str) -> None:
            selected_filters.append(selected_filter)

        def setDirectory(self, directory: str) -> None:
            pass

        def exec(self) -> bool:
            return False

    manager = types.SimpleNamespace(_recent_name_filter=None, _recent_directory=None)
    manager._preferred_name_filter = types.MethodType(
        ImageToolManager._preferred_name_filter, manager
    )
    monkeypatch.setattr(QtWidgets, "QFileDialog", _FakeFileDialog)
    monkeypatch.setattr(
        erlab.interactive.utils,
        "file_loaders",
        lambda *_args: {
            "xarray HDF5 Files (*.h5)": (
                xr.load_dataarray,
                {"engine": "h5netcdf"},
            ),
            example_filter: erlab.io.loaders["example"].file_dialog_methods[
                example_filter
            ],
        },
    )

    ImageToolManager.open(manager, native=False)

    assert selected_filters == [example_filter]


@pytest.mark.parametrize("mode", ["dragdrop", "ask"])
def test_manager_open_files(
    qtbot,
    accept_dialog,
    test_data,
    mode,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager, tempfile.TemporaryDirectory() as tmp_dir_name:
        filename = f"{tmp_dir_name}/data.h5"
        test_data.to_netcdf(filename, engine="h5netcdf")

        with qtbot.waitExposed(manager):
            manager.show()
            manager.activateWindow()

        if mode == "dragdrop":
            mime_data = QtCore.QMimeData()
            mime_data.setUrls([QtCore.QUrl.fromLocalFile(filename)])
            evt = QtGui.QDropEvent(
                QtCore.QPointF(0.0, 0.0),
                QtCore.Qt.DropAction.CopyAction,
                mime_data,
                QtCore.Qt.MouseButton.LeftButton,
                QtCore.Qt.KeyboardModifier.NoModifier,
            )

            # Simulate drag and drop
            def trigger_drop():
                return manager.dropEvent(evt)
        else:

            def trigger_drop():
                return load_in_manager([filename], loader_name=None)

        accept_dialog(trigger_drop)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        xarray.testing.assert_identical(
            manager.get_imagetool(0).slicer_area.data, test_data
        )

        # Try reload
        with qtbot.wait_signal(manager.get_imagetool(0).slicer_area.sigDataChanged):
            manager.get_imagetool(0).slicer_area.reload()

        # Try archive
        manager._imagetool_wrappers[0].archive()
        qtbot.wait_until(lambda: manager._imagetool_wrappers[0].archived, timeout=5000)

        # Unarchive
        manager._imagetool_wrappers[0].unarchive()
        qtbot.wait_until(
            lambda: not manager._imagetool_wrappers[0].archived, timeout=5000
        )

        # Simulate drag and drop with wrong filter, retry with correct filter
        # Dialogs created are:
        # select loader → failed alert → retry → select loader
        def _choose_wrong_filter(dialog: _NameFilterDialog):
            assert (
                next(iter(dialog._valid_loaders.keys())) == "xarray HDF5 Files (*.h5)"
            )
            dialog._button_group.buttons()[-1].setChecked(True)

        def _choose_correct_filter(dialog: _NameFilterDialog):
            dialog._button_group.buttons()[0].setChecked(True)

        accept_dialog(
            trigger_drop,
            pre_call=[_choose_wrong_filter, None, None, _choose_correct_filter],
            chained_dialogs=4,
        )
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        xarray.testing.assert_identical(
            manager.get_imagetool(1).slicer_area.data, test_data
        )


@pytest.mark.parametrize("case", ["loader_cancel", "non_loader"])
def test_manager_open_loader_selection_branches(
    monkeypatch,
    tmp_path: pathlib.Path,
    example_loader,
    case: str,
) -> None:
    file_path = tmp_path / "data_002.h5"
    name_filter = "Example Raw Data (*.h5)"

    class _FakeFileDialog:
        AcceptMode = QtWidgets.QFileDialog.AcceptMode
        FileMode = QtWidgets.QFileDialog.FileMode
        Option = QtWidgets.QFileDialog.Option

        def __init__(self, parent) -> None:
            pass

        def setAcceptMode(self, mode) -> None:
            pass

        def setFileMode(self, mode) -> None:
            pass

        def setNameFilters(self, filters) -> None:
            pass

        def setOption(self, option) -> None:
            pass

        def selectNameFilter(self, selected_filter: str) -> None:
            pass

        def setDirectory(self, directory: str) -> None:
            pass

        def exec(self) -> bool:
            return True

        def selectedFiles(self) -> list[str]:
            return [str(file_path)]

        def selectedNameFilter(self) -> str:
            return name_filter

    def non_loader(*_args, **_kwargs) -> None:
        return None

    add_calls: list[tuple[tuple[typing.Any, ...], dict[str, typing.Any]]] = []
    select_calls: list[tuple[typing.Any, ...]] = []

    def _select_loader_options(*args, **kwargs):
        select_calls.append((*args, kwargs))

    manager = types.SimpleNamespace(
        _recent_name_filter=None,
        _recent_directory=None,
        _select_loader_options=_select_loader_options,
        _add_from_multiple_files=lambda *args, **kwargs: add_calls.append(
            (args, kwargs)
        ),
    )
    manager._preferred_name_filter = types.MethodType(
        ImageToolManager._preferred_name_filter, manager
    )
    monkeypatch.setattr(QtWidgets, "QFileDialog", _FakeFileDialog)
    monkeypatch.setattr(
        erlab.interactive.utils,
        "file_loaders",
        lambda *_args: {
            name_filter: (
                erlab.io.loaders["example"].load
                if case == "loader_cancel"
                else non_loader,
                {},
            )
        },
    )

    ImageToolManager.open(manager, native=False)

    if case == "loader_cancel":
        assert len(select_calls) == 1
        assert list(select_calls[0][-1]["sample_paths"]) == [str(file_path)]
        assert add_calls == []
    else:
        assert select_calls == []
        assert len(add_calls) == 1


@pytest.mark.parametrize(
    "case",
    ["single_non_loader", "single_loader_cancel", "multiple_cancel", "multiple_accept"],
)
def test_open_multiple_files_loader_selection_branches(
    monkeypatch,
    tmp_path: pathlib.Path,
    example_loader,
    case: str,
) -> None:
    file_path = tmp_path / "data_002.h5"

    def non_loader(*_args, **_kwargs) -> None:
        return None

    loader_func = erlab.io.loaders["example"].load
    valid_loaders = {
        "single_non_loader": {"Plain Files (*.txt)": (non_loader, {"plain": True})},
        "single_loader_cancel": {"Example Raw Data (*.h5)": (loader_func, {})},
        "multiple_cancel": {
            "Example Raw Data (*.h5)": (loader_func, {}),
            "Plain Files (*.txt)": (non_loader, {}),
        },
        "multiple_accept": {
            "Example Raw Data (*.h5)": (loader_func, {}),
            "Plain Files (*.txt)": (non_loader, {"plain": True}),
        },
    }[case]
    select_result = {
        "single_non_loader": None,
        "single_loader_cancel": None,
        "multiple_cancel": None,
        "multiple_accept": ("Plain Files (*.txt)", non_loader, {"plain": True}),
    }[case]

    add_calls: list[
        tuple[
            list[pathlib.Path],
            list[pathlib.Path],
            list[pathlib.Path],
            Callable,
            dict[str, typing.Any],
        ]
    ] = []
    select_calls: list[tuple[list[str], str | None, list[pathlib.Path]]] = []

    def _select_loader_options(loaders, name_filter=None, *, sample_paths=None):
        select_calls.append((list(loaders), name_filter, list(sample_paths or ())))
        return select_result

    def _retry_open_multiple_files(*_args, **_kwargs) -> None:
        return None

    manager = types.SimpleNamespace(
        _recent_name_filter=None,
        _select_loader_options=_select_loader_options,
        open_multiple_files=_retry_open_multiple_files,
        _add_from_multiple_files=lambda loaded, queued, failed, func, kwargs, _: (
            add_calls.append((loaded, queued, failed, func, kwargs))
        ),
    )
    monkeypatch.setattr(
        erlab.interactive.utils,
        "file_loaders",
        lambda *_args: valid_loaders,
    )

    ImageToolManager.open_multiple_files(manager, [file_path])

    if case == "single_non_loader":
        assert select_calls == []
        assert manager._recent_name_filter == "Plain Files (*.txt)"
        assert add_calls == [
            ([], [file_path], [], non_loader, {"plain": True}),
        ]
    elif case == "single_loader_cancel":
        assert select_calls == [
            (["Example Raw Data (*.h5)"], "Example Raw Data (*.h5)", [file_path])
        ]
        assert add_calls == []
    elif case == "multiple_cancel":
        assert select_calls == [
            (["Example Raw Data (*.h5)", "Plain Files (*.txt)"], None, [file_path])
        ]
        assert add_calls == []
    else:
        assert select_calls == [
            (["Example Raw Data (*.h5)", "Plain Files (*.txt)"], None, [file_path])
        ]
        assert manager._recent_name_filter == "Plain Files (*.txt)"
        assert add_calls == [
            ([], [file_path], [], non_loader, {"plain": True}),
        ]


@pytest.mark.parametrize("entry_point", ["open", "drop"])
def test_manager_file_loads_with_loader_extensions(
    qtbot,
    accept_dialog,
    monkeypatch,
    example_loader,
    example_data_dir: pathlib.Path,
    entry_point: str,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    file_path = example_data_dir / "data_002.h5"
    name_filter = "Example Raw Data (*.h5)"

    def _file_loaders(*_args):
        return {name_filter: (erlab.io.loaders["example"].load, {})}

    def _set_loader_extensions(dialog: _NameFilterDialog) -> None:
        assert dialog.extensions_toggle.isVisible()
        assert not dialog.extensions_group.isVisible()
        dialog.extensions_toggle.setChecked(True)
        assert dialog.extensions_group.isVisible()
        dialog.loader_extension_lines["additional_coords"].setText("{'gui_extra': 7.0}")

    monkeypatch.setattr(erlab.interactive.utils, "file_loaders", _file_loaders)

    if entry_point == "open":
        monkeypatch.setattr(QtWidgets.QFileDialog, "exec", lambda self: True)
        monkeypatch.setattr(
            QtWidgets.QFileDialog, "selectedFiles", lambda self: [str(file_path)]
        )
        monkeypatch.setattr(
            QtWidgets.QFileDialog, "selectedNameFilter", lambda self: name_filter
        )

    with manager_context() as manager:
        if entry_point == "open":

            def _trigger_load():
                return ImageToolManager.open(manager, native=False)

        else:

            def _trigger_load():
                return manager.open_multiple_files([file_path])

        accept_dialog(_trigger_load, pre_call=_set_loader_extensions)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=10000)

        slicer_area = manager.get_imagetool(0).slicer_area
        assert float(slicer_area._data["gui_extra"]) == 7.0
        assert slicer_area._load_func is not None
        assert slicer_area._load_func[1]["loader_extensions"] == {
            "additional_coords": {"gui_extra": 7.0}
        }

        with qtbot.wait_signal(slicer_area.sigDataChanged, timeout=10000):
            slicer_area.reload()
        assert float(slicer_area._data["gui_extra"]) == 7.0

        tree = manager._to_datatree()
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)
        accept_dialog(lambda: manager._from_datatree(tree))
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        restored_area = manager.get_imagetool(0).slicer_area
        assert restored_area._load_func is not None
        assert restored_area._load_func[1]["loader_extensions"] == {
            "additional_coords": {"gui_extra": 7.0}
        }
        with qtbot.wait_signal(restored_area.sigDataChanged, timeout=10000):
            restored_area.reload()
        assert float(restored_area._data["gui_extra"]) == 7.0


def test_manager_console(
    qtbot,
    accept_dialog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        data = xr.DataArray(
            np.arange(25).reshape((5, 5)),
            dims=["x", "y"],
            coords={"x": np.arange(5), "y": np.arange(5)},
        )

        manager.show()
        manager.activateWindow()

        manager._data_recv([data, data], kwargs={"link": True, "link_colors": True})
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        # Open console
        manager.toggle_console()
        qtbot.wait_until(manager.console.isVisible, timeout=5000)

        def _get_last_output_contents():
            return manager.console._console_widget.kernel_manager.kernel.shell.user_ns[
                "_"
            ]

        # Test delayed import
        manager.console._console_widget.execute("era")
        assert _get_last_output_contents() == erlab.analysis

        # Loader state should persist across console cells.
        manager.console._console_widget.execute("erlab.io.set_loader('merlin')")
        manager.console._console_widget.execute("erlab.io.loaders.current_loader.name")
        assert _get_last_output_contents() == "merlin"

        # Test repr
        manager.console._console_widget.execute("tools")
        assert str(_get_last_output_contents()) == "0: \n1: "
        manager.console._console_widget.execute("tools[0]")

        # Select all
        select_tools(manager, list(manager._imagetool_wrappers.keys()))
        manager.console._console_widget.execute("tools.selected_data")
        assert _get_last_output_contents() == [
            wrapper.imagetool.slicer_area._data
            for wrapper in manager._imagetool_wrappers.values()
        ]

        # Test storing with ipython
        accept_dialog(manager.store_action.trigger)
        manager.console._console_widget.execute(r"%store -d data_0 data_1")

        # Test calling wrapped methods
        manager.console._console_widget.execute("tools[0].archive()")
        qtbot.wait_until(lambda: manager._imagetool_wrappers[0].archived, timeout=5000)

        # Test setting data
        manager.console._console_widget.execute(
            "tools[1].data = xr.DataArray("
            "np.arange(25).reshape((5, 5)) * 2, "
            "dims=['x', 'y'], "
            "coords={'x': np.arange(5), 'y': np.arange(5)}"
            ")",
        )
        xr.testing.assert_identical(manager.get_imagetool(1).slicer_area.data, data * 2)

        manager.console._console_widget.execute("tools[1].data += 1")
        xr.testing.assert_identical(
            manager.get_imagetool(1).slicer_area.data, data * 2 + 1
        )

        manager.console._console_widget.execute("tools[1].data[0, 0] = -5.0")
        assert float(manager.get_imagetool(1).slicer_area._data.values[0, 0]) == -5.0
        assert float(manager.get_imagetool(1).slicer_area.data.values[0, 0]) == -5.0
        xr.testing.assert_identical(manager.get_imagetool(0).slicer_area.data, data)

        # Remove all tools
        select_tools(manager, list(manager._imagetool_wrappers.keys()))
        accept_dialog(manager.remove_action.trigger)
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        # Test repr
        manager.console._console_widget.execute("tools")
        assert str(_get_last_output_contents()) == "No tools"

        # Test magic command: itool
        manager.console._console_widget.kernel_manager.kernel.shell.user_ns[
            "example_data"
        ] = xr.DataArray(
            np.arange(25).reshape((5, 5)),
            dims=["alpha", "eV"],
            coords={"alpha": np.arange(5), "eV": np.arange(5)},
        )
        manager.console._console_widget.execute(r"%itool example_data --cmap viridis")
        qtbot.wait_until(lambda: manager.ntools == 1)
        assert manager.get_imagetool(0).array_slicer.point_value(0) == 12.0

        # Destroy console
        manager.console._console_widget.shutdown_kernel()
        InteractiveShell.clear_instance()


def test_tool_namespace_get_data_item(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        manager.activateWindow()

        itool([test_data], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        namespace = ToolNamespace(manager._imagetool_wrappers[0])
        xr.testing.assert_identical(
            namespace._get_data_item((slice(None), 0)), test_data[:, 0]
        )


def test_tool_namespace_set_data_replaces_source(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        manager.activateWindow()

        itool([test_data], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 1, timeout=5000
        )

        namespace = ToolNamespace(manager._imagetool_wrappers[0])
        child = next(iter(manager._imagetool_wrappers[0]._childtools.values()))
        updated = (test_data * 2).rename(test_data.name)

        namespace.data = updated

        xr.testing.assert_identical(parent_tool.slicer_area.data, updated)
        qtbot.wait_until(lambda: child.source_state == "stale", timeout=5000)


def test_tool_namespace_set_data_item_marks_child_tools_stale(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        manager.activateWindow()

        itool([test_data], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 1, timeout=5000
        )

        namespace = ToolNamespace(manager._imagetool_wrappers[0])
        child = next(iter(manager._imagetool_wrappers[0]._childtools.values()))

        namespace._set_data_item((0, 0), -5.0)

        assert float(parent_tool.slicer_area._data.values[0, 0]) == -5.0
        qtbot.wait_until(lambda: child.source_state == "stale", timeout=5000)


def test_tool_namespace_set_data_item_failure_keeps_child_tools_fresh(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        manager.activateWindow()

        itool([test_data], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 1, timeout=5000
        )

        namespace = ToolNamespace(manager._imagetool_wrappers[0])
        child = next(iter(manager._imagetool_wrappers[0]._childtools.values()))

        with pytest.raises(IndexError, match="too many indices"):
            namespace._set_data_item((0, 0, 0), -5.0)

        assert child.source_state == "fresh"
        assert float(parent_tool.slicer_area._data.values[0, 0]) == 0.0


def test_console_assignment_transformer_match_helpers() -> None:
    plain_name = typing.cast("ast.Expr", ast.parse("value").body[0]).value
    attr_without_subscript = typing.cast(
        "ast.Expr", ast.parse("tool.data").body[0]
    ).value
    other_tool = typing.cast("ast.Expr", ast.parse("other[0].data").body[0]).value
    subscript_target = typing.cast(
        "ast.Expr", ast.parse("other[0].data[1]").body[0]
    ).value

    assert _ConsoleDataAssignmentTransformer._match_tool_data(plain_name) is None
    assert (
        _ConsoleDataAssignmentTransformer._match_tool_data(attr_without_subscript)
        is None
    )
    assert _ConsoleDataAssignmentTransformer._match_tool_data(other_tool) is None
    assert _ConsoleDataAssignmentTransformer._match_target(subscript_target) is None


def test_rewrite_console_source_handles_augassign_and_passthrough() -> None:
    rewritten = _rewrite_console_source("tools[1].data[0, 0] += 1")

    assert "_get_data_item" in rewritten
    assert "_set_data_item" in rewritten

    assert _rewrite_console_source("left = right = 1") == "left = right = 1"
    assert _rewrite_console_source("value += 1") == "value += 1"


def test_manager_hover_tooltip(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
    monkeypatch,
) -> None:
    with manager_context() as manager:
        manager.show()
        manager.activateWindow()

        itool([test_data, test_data, test_data], link=True, manager=True)

        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)

        manager.get_imagetool(0).slicer_area._auto_chunk()
        manager.get_imagetool(1).slicer_area._auto_chunk()
        manager.get_imagetool(2).slicer_area._auto_chunk()
        select_tools(manager, [0])
        manager._update_info()
        assert "Chunks" in metadata_detail_map(manager)

        view = manager.tree_view

        model = view._model
        delegate = view._delegate

        index = model.index(0, 0)  # first tool
        option = QtWidgets.QStyleOptionViewItem()
        delegate.initStyleOption(option, index)
        option.rect = view.visualRect(index)
        _, dask_rect, link_rect, _ = delegate._compute_icons_info(
            option, index.internalPointer()
        )

        text = None

        def fake_show_text(pos, s, *args, **kwargs):
            nonlocal text
            text = s

        monkeypatch.setattr(QtWidgets.QToolTip, "showText", fake_show_text)

        # Hover over dask icon
        pos = dask_rect.center()
        event = QtGui.QHelpEvent(
            QtCore.QEvent.Type.ToolTip, pos, view.viewport().mapToGlobal(pos)
        )
        handled = delegate.helpEvent(event, view, option, index)

        assert handled
        assert_nonempty_tooltip(text)

        popup_positions: list[QtCore.QPoint] = []
        dask_menu = manager.get_imagetool(0)._dask_menu
        monkeypatch.setattr(dask_menu, "popup", popup_positions.append)
        click_tree_view_pos(view, dask_rect.center())
        assert popup_positions == [view.viewport().mapToGlobal(dask_rect.bottomLeft())]
        assert manager.get_imagetool(0).slicer_area.data_chunked

        # Hover over link icon
        text = None
        pos = link_rect.center()
        event = QtGui.QHelpEvent(
            QtCore.QEvent.Type.ToolTip, pos, view.viewport().mapToGlobal(pos)
        )
        handled = delegate.helpEvent(event, view, option, index)

        assert handled
        assert_nonempty_tooltip(text)

        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "question",
            lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        click_tree_view_pos(view, link_rect.center())
        assert manager.get_imagetool(0).slicer_area.is_linked
        assert manager.get_imagetool(1).slicer_area.is_linked
        assert manager.get_imagetool(2).slicer_area.is_linked

        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "question",
            lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Yes,
        )
        click_tree_view_pos(view, link_rect.center())
        assert not manager.get_imagetool(0).slicer_area.is_linked
        assert manager.get_imagetool(1).slicer_area.is_linked
        assert manager.get_imagetool(2).slicer_area.is_linked

        wrapper = manager._imagetool_wrappers[0]
        wrapper.set_watched_binding("sample", "sample kernel", connected=False)
        option = QtWidgets.QStyleOptionViewItem()
        delegate.initStyleOption(option, index)
        option.rect = view.visualRect(index)
        _, _, _, watched_rect = delegate._compute_icons_info(option, wrapper)
        assert watched_rect is not None

        text = None
        pos = watched_rect.center()
        event = QtGui.QHelpEvent(
            QtCore.QEvent.Type.ToolTip, pos, view.viewport().mapToGlobal(pos)
        )
        handled = delegate.helpEvent(event, view, option, index)
        assert handled
        assert_nonempty_tooltip(text)

        click_tree_view_pos(view, watched_rect.center())
        assert view._badge_menu is not None
        refresh_action, stop_action = view._badge_menu.actions()
        assert not refresh_action.isEnabled()

        wrapper.set_watched_binding("sample", "sample kernel", connected=True)
        click_tree_view_pos(view, watched_rect.center())
        assert view._badge_menu is not None
        refresh_action, stop_action = view._badge_menu.actions()
        assert refresh_action.isEnabled()
        with qtbot.wait_signal(manager._sigWatchedDataEdited) as blocker:
            refresh_action.trigger()
        assert blocker.args == ["sample", "sample kernel", "updated"]

        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "question",
            lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        stop_action.trigger()
        assert wrapper.watched
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "question",
            lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Yes,
        )
        with qtbot.wait_signal(manager._sigWatchedDataEdited) as blocker:
            stop_action.trigger()
        assert blocker.args == ["sample", "sample kernel", "removed"]
        assert not wrapper.watched
        assert wrapper.watched_metadata() == {}

        # Hover outside icons
        text = None
        pos = dask_rect.topRight() + QtCore.QPoint(2, 0)
        event = QtGui.QHelpEvent(
            QtCore.QEvent.Type.ToolTip, pos, view.viewport().mapToGlobal(pos)
        )
        handled = delegate.helpEvent(event, view, option, index)
        assert not handled
        assert text is None


def test_manager_badge_hit_testing_edge_paths(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        manager.activateWindow()

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        view = manager.tree_view
        model = view._model
        delegate = view._delegate
        index = model.index(0, 0)
        wrapper = manager._imagetool_wrappers[0]

        manager.get_imagetool(0).slicer_area._auto_chunk()
        view.refresh(0)
        option = delegate._option_for_index(view, index)
        _, dask_rect, _, _ = delegate._compute_icons_info(option, wrapper)
        assert dask_rect is not None

        assert delegate._badge_at(option, QtCore.QModelIndex(), QtCore.QPoint()) is None
        assert (
            delegate._badge_at(
                option, model.createIndex(0, 0, object()), option.rect.center()
            )
            is None
        )
        missing_child_index = model.createIndex(0, 0, "missing-child")
        assert (
            delegate._badge_at(option, missing_child_index, option.rect.center())
            is None
        )
        view._handle_badge_click(
            missing_child_index, _RowBadge("source_status", QtCore.QRect(), "")
        )
        for kind in ("dask", "link", "watched"):
            view._handle_badge_click(
                missing_child_index, _RowBadge(kind, QtCore.QRect(), "")
            )
        for kind in ("tool_type", "source_status"):
            view._handle_badge_click(index, _RowBadge(kind, QtCore.QRect(), ""))

        def _left_release(pos: QtCore.QPoint) -> QtGui.QMouseEvent:
            global_pos = view.viewport().mapToGlobal(pos)
            return QtGui.QMouseEvent(
                QtCore.QEvent.Type.MouseButtonRelease,
                QtCore.QPointF(pos),
                QtCore.QPointF(global_pos),
                QtCore.Qt.MouseButton.LeftButton,
                QtCore.Qt.MouseButton.LeftButton,
                QtCore.Qt.KeyboardModifier.NoModifier,
            )

        view.mouseReleaseEvent(_left_release(QtCore.QPoint(-10, -10)))
        view.mouseReleaseEvent(_left_release(option.rect.center()))

        def _mouse_move(pos: QtCore.QPoint) -> QtGui.QMouseEvent:
            global_pos = view.viewport().mapToGlobal(pos)
            return QtGui.QMouseEvent(
                QtCore.QEvent.Type.MouseMove,
                QtCore.QPointF(pos),
                QtCore.QPointF(global_pos),
                QtCore.Qt.MouseButton.NoButton,
                QtCore.Qt.MouseButton.NoButton,
                QtCore.Qt.KeyboardModifier.NoModifier,
            )

        delegate.eventFilter(view.viewport(), _mouse_move(dask_rect.center()))
        assert (
            view.viewport().cursor().shape() == QtCore.Qt.CursorShape.PointingHandCursor
        )

        delegate.eventFilter(view.viewport(), _mouse_move(option.rect.center()))
        assert (
            view.viewport().cursor().shape() != QtCore.Qt.CursorShape.PointingHandCursor
        )

        delegate.eventFilter(view.viewport(), _mouse_move(QtCore.QPoint(-10, -10)))
        assert (
            view.viewport().cursor().shape() != QtCore.Qt.CursorShape.PointingHandCursor
        )

        delegate.eventFilter(view.viewport(), QtCore.QEvent(QtCore.QEvent.Type.Leave))
        assert (
            view.viewport().cursor().shape() != QtCore.Qt.CursorShape.PointingHandCursor
        )
        delegate.eventFilter(None, QtCore.QEvent(QtCore.QEvent.Type.Leave))
        delegate.eventFilter(None, _mouse_move(dask_rect.center()))

        fake_link_rect = QtCore.QRect(
            option.rect.left() + 4, option.rect.top() + 4, 16, 16
        )
        with monkeypatch.context() as patch:
            patch.setattr(
                delegate,
                "_compute_icons_info",
                lambda option_arg, wrapper_arg: (16, None, fake_link_rect, None),
            )
            patch.setattr(wrapper.slicer_area, "_linking_proxy", None)
            assert delegate._badge_at(option, index, fake_link_rect.center()) is None

        view._show_dask_badge_menu(
            types.SimpleNamespace(imagetool=None),
            QtCore.QRect(),
        )
        view._stop_watching_badge_target(wrapper)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(lambda: len(wrapper._childtool_indices) == 1, timeout=5000)
        child_uid = wrapper._childtool_indices[0]
        child_index = model._row_index(child_uid)
        child_node = manager._child_node(child_uid)
        child_option = delegate._option_for_index(view, child_index)
        assert (
            delegate._badge_at(child_option, child_index, child_option.rect.center())
            is None
        )

        source_dialog_parents: list[ImageToolManager] = []
        monkeypatch.setattr(
            child_node,
            "show_source_update_dialog",
            lambda *, parent: source_dialog_parents.append(parent),
        )
        view._handle_badge_click(
            child_index, _RowBadge("source_status", QtCore.QRect(), "")
        )
        assert source_dialog_parents == [manager]


def test_warning_alert(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        warning_logger = logging.getLogger("test.warning.history")
        warning_logger.warning("First warning")
        warning_logger.warning("Second warning")

        qtbot.wait_until(lambda: len(manager._alert_dialogs) == 2)
        qtbot.wait_until(
            lambda: all(warning.isVisible() for warning in manager._alert_dialogs)
        )

        texts = [notification.text() for notification in manager._alert_dialogs]
        assert any("First warning" in text for text in texts)
        assert any("Second warning" in text for text in texts)

        clear_all_button = manager._alert_dialogs[-1].findChild(
            QtWidgets.QPushButton, "warningDismissAllButton"
        )
        assert clear_all_button is not None

        qtbot.mouseClick(clear_all_button, QtCore.Qt.MouseButton.LeftButton, delay=10)
        qtbot.wait_until(lambda: len(manager._alert_dialogs) == 0)


def test_warning_alert_suppressed_by_log_flag(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        suppressed = logging.LogRecord(
            name="test.warning.suppressed",
            level=logging.WARNING,
            pathname=__file__,
            lineno=0,
            msg="suppressed warning",
            args=(),
            exc_info=None,
        )
        suppressed.suppress_ui_alert = True
        manager._warning_handler.emit(suppressed)
        QtWidgets.QApplication.processEvents()
        assert manager._alert_dialogs == []

        regular = logging.LogRecord(
            name="test.warning.regular",
            level=logging.WARNING,
            pathname=__file__,
            lineno=0,
            msg="regular warning",
            args=(),
            exc_info=None,
        )
        manager._warning_handler.emit(regular)
        qtbot.wait_until(lambda: len(manager._alert_dialogs) == 1)

        manager._clear_all_alerts()
        QtWidgets.QApplication.processEvents()


def test_error_creating_imagetool_does_not_duplicate_alert_dialog(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    critical_calls: list[tuple[typing.Any, ...]] = []

    def _fake_critical(*args, **kwargs):
        critical_calls.append((args, kwargs))
        return int(QtWidgets.QDialog.DialogCode.Accepted)

    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        staticmethod(_fake_critical),
    )

    with manager_context() as manager:
        try:
            raise RuntimeError("boom")  # noqa: TRY301
        except RuntimeError:
            manager._error_creating_imagetool()

        QtWidgets.QApplication.processEvents()

        assert len(critical_calls) == 1
        assert manager._alert_dialogs == []


def test_manager_duplicate_unserializable_child_shows_error(
    qtbot,
    gold,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    critical_calls: list[tuple[tuple[typing.Any, ...], dict[str, typing.Any]]] = []

    def _fake_critical(*args, **kwargs):
        critical_calls.append((args, kwargs))
        return int(QtWidgets.QDialog.DialogCode.Accepted)

    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        staticmethod(_fake_critical),
    )

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()

        itool(gold, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child_uid = manager.add_childtool(
            _UnserializableChildTool(gold.copy(deep=True)),
            0,
            show=False,
        )

        select_child_tool(manager, child_uid)
        manager.duplicate_selected()

        assert len(manager._imagetool_wrappers[0]._childtools) == 1
        assert len(critical_calls) == 1
        assert critical_calls[0][0][2] == (
            "An error occurred while duplicating the selected window."
        )
        assert "data_corr" in critical_calls[0][1]["detailed_text"]


def test_manager_save_unserializable_child_shows_error(
    qtbot,
    accept_dialog,
    gold,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    critical_calls: list[tuple[tuple[typing.Any, ...], dict[str, typing.Any]]] = []

    def _fake_critical(*args, **kwargs):
        critical_calls.append((args, kwargs))
        return int(QtWidgets.QDialog.DialogCode.Accepted)

    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        staticmethod(_fake_critical),
    )

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()

        itool(gold, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        manager.add_childtool(
            _UnserializableChildTool(gold.copy(deep=True)),
            0,
            show=False,
        )

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            filename = f"{tmp_dir_name}/workspace.itws"

            def _go_to_file(dialog: QtWidgets.QFileDialog):
                dialog.setDirectory(tmp_dir_name)
                dialog.selectFile(filename)
                focused = dialog.focusWidget()
                if isinstance(focused, QtWidgets.QLineEdit):
                    focused.setText("workspace.itws")

            accept_dialog(lambda: manager.save(native=False), pre_call=_go_to_file)

            assert len(critical_calls) == 1
            assert critical_calls[0][0][2] == (
                "An error occurred while saving the workspace file."
            )
            assert "data_corr" in critical_calls[0][1]["detailed_text"]
            assert not pathlib.Path(filename).exists()


def test_data_recv_dataset_creation_error_no_duplicate_alert(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    critical_calls: list[tuple[typing.Any, ...]] = []

    def _fake_critical(*args, **kwargs):
        critical_calls.append((args, kwargs))
        return int(QtWidgets.QDialog.DialogCode.Accepted)

    def _raise_from_dataset(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        staticmethod(_fake_critical),
    )
    monkeypatch.setattr(
        erlab.interactive.imagetool.manager._mainwindow.ImageTool,
        "from_dataset",
        staticmethod(_raise_from_dataset),
    )

    with manager_context() as manager:
        ds = xr.Dataset({"v": xr.DataArray(np.ones((2, 2)), dims=("x", "y"))})
        flags = manager._data_recv([ds], {})

        QtWidgets.QApplication.processEvents()

        assert flags == [False]
        assert len(critical_calls) == 1
        assert manager._alert_dialogs == []


def test_data_recv_dataarray_creation_error_no_duplicate_alert(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    critical_calls: list[tuple[typing.Any, ...]] = []

    def _fake_critical(*args, **kwargs):
        critical_calls.append((args, kwargs))
        return int(QtWidgets.QDialog.DialogCode.Accepted)

    def _raise_imagetool(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        staticmethod(_fake_critical),
    )
    monkeypatch.setattr(
        erlab.interactive.imagetool.manager._mainwindow,
        "ImageTool",
        _raise_imagetool,
    )

    with manager_context() as manager:
        flags = manager._data_recv([xr.DataArray(np.ones((2, 2)), dims=("x", "y"))], {})

        QtWidgets.QApplication.processEvents()

        assert flags == [False]
        assert len(critical_calls) == 1
        assert manager._alert_dialogs == []


def test_load_workspace_error_no_duplicate_alert(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    critical_calls: list[tuple[typing.Any, ...]] = []
    exec_calls = {"count": 0}
    native_opt_calls = {"count": 0}

    def _fake_critical(*args, **kwargs):
        critical_calls.append((args, kwargs))
        return int(QtWidgets.QDialog.DialogCode.Accepted)

    def _fake_exec(self):
        exec_calls["count"] += 1
        return exec_calls["count"] == 1

    def _raise_open_datatree(*args, **kwargs):
        raise RuntimeError("broken workspace")

    def _track_set_option(self, option, on=True):
        if option == QtWidgets.QFileDialog.Option.DontUseNativeDialog and on:
            native_opt_calls["count"] += 1
        return original_set_option(self, option, on)

    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        staticmethod(_fake_critical),
    )
    original_set_option = QtWidgets.QFileDialog.setOption
    monkeypatch.setattr(QtWidgets.QFileDialog, "setOption", _track_set_option)
    monkeypatch.setattr(QtWidgets.QFileDialog, "exec", _fake_exec)
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "selectedFiles",
        lambda self: ["broken_workspace.itws"],
    )
    monkeypatch.setattr(manager_xarray, "open_workspace_datatree", _raise_open_datatree)

    with manager_context() as manager:
        ImageToolManager.load(manager, native=False)

        QtWidgets.QApplication.processEvents()

        assert exec_calls["count"] == 1
        assert native_opt_calls["count"] == 1
        assert len(critical_calls) == 1
        assert manager._alert_dialogs == []


def test_workspace_file_lock_error_detects_nested_blocking_io() -> None:
    err = RuntimeError("open failed")
    err.__cause__ = BlockingIOError(35, "unable to lock file")

    assert manager_workspace._is_workspace_file_lock_error(err)
    assert not manager_workspace._is_workspace_file_lock_error(
        RuntimeError("broken workspace")
    )


def test_import_workspace_locked_file_shows_specific_error(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    lock_calls: list[pathlib.Path] = []
    generic_calls: list[tuple[typing.Any, ...]] = []
    fname = tmp_path / "locked.itws"

    def _raise_locked(*args, **kwargs):
        raise BlockingIOError(35, "unable to lock file")

    monkeypatch.setattr(QtWidgets.QFileDialog, "exec", lambda self: True)
    monkeypatch.setattr(QtWidgets.QFileDialog, "selectedFiles", lambda self: [fname])
    monkeypatch.setattr(manager_xarray, "open_workspace_datatree", _raise_locked)
    monkeypatch.setattr(
        manager_mainwindow,
        "_show_workspace_file_lock_error",
        lambda _parent, locked_fname: lock_calls.append(pathlib.Path(locked_fname)),
    )
    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        staticmethod(lambda *args, **kwargs: generic_calls.append(args)),
    )

    with manager_context() as manager:
        assert not manager.import_workspace(native=False)

    assert lock_calls == [fname]
    assert generic_calls == []


def test_open_multiple_files_locked_workspace_does_not_fall_through_to_loaders(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    lock_calls: list[pathlib.Path] = []
    fname = tmp_path / "locked.itws"

    def _raise_locked(*args, **kwargs):
        raise BlockingIOError(35, "unable to lock file")

    def _file_loaders_should_not_run(*args, **kwargs):
        raise AssertionError("locked workspace should not fall through to loaders")

    monkeypatch.setattr(manager_xarray, "open_workspace_datatree", _raise_locked)
    monkeypatch.setattr(
        manager_mainwindow,
        "_show_workspace_file_lock_error",
        lambda _parent, locked_fname: lock_calls.append(pathlib.Path(locked_fname)),
    )
    monkeypatch.setattr(
        erlab.interactive.utils, "file_loaders", _file_loaders_should_not_run
    )

    with manager_context() as manager:
        manager.open_multiple_files([fname], try_workspace=True)

    assert lock_calls == [fname]


def test_open_retry_preserves_non_native_dialog(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    exec_calls = {"count": 0}
    native_opt_calls = {"count": 0}
    add_calls = {"count": 0}

    def _fake_exec(self):
        exec_calls["count"] += 1
        return exec_calls["count"] == 1

    def _track_set_option(self, option, on=True):
        if option == QtWidgets.QFileDialog.Option.DontUseNativeDialog and on:
            native_opt_calls["count"] += 1
        return original_set_option(self, option, on)

    original_set_option = QtWidgets.QFileDialog.setOption
    monkeypatch.setattr(QtWidgets.QFileDialog, "setOption", _track_set_option)
    monkeypatch.setattr(QtWidgets.QFileDialog, "exec", _fake_exec)
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "selectedFiles",
        lambda self: ["fake_data.h5"],
    )
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "selectedNameFilter",
        lambda self: "Fake Loader (*.h5)",
    )
    monkeypatch.setattr(
        erlab.interactive.utils,
        "file_loaders",
        lambda: {"Fake Loader (*.h5)": (lambda *_a, **_k: None, {})},
    )

    with manager_context() as manager:

        def _fake_add_from_multiple_files(
            *,
            loaded,
            queued,
            failed,
            func,
            kwargs,
            retry_callback,
        ):
            add_calls["count"] += 1
            if add_calls["count"] == 1:
                retry_callback(None)

        monkeypatch.setattr(
            manager, "_add_from_multiple_files", _fake_add_from_multiple_files
        )
        ImageToolManager.open(manager, native=False)

    assert exec_calls["count"] >= 2
    assert native_opt_calls["count"] >= 2


def test_open_multiple_files_workspace_error_no_duplicate_alert(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    critical_calls: list[tuple[typing.Any, ...]] = []

    def _fake_critical(*args, **kwargs):
        critical_calls.append((args, kwargs))
        return int(QtWidgets.QDialog.DialogCode.Accepted)

    class _FakeDataTree:
        def __init__(self) -> None:
            self.attrs = {"imagetool_workspace_schema_version": 4}

        def close(self) -> None:
            pass

    def _fake_open_datatree(*args, **kwargs):
        return _FakeDataTree()

    def _raise_from_datatree(*args, **kwargs):
        raise RuntimeError("cannot restore workspace")

    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        staticmethod(_fake_critical),
    )
    monkeypatch.setattr(manager_xarray, "open_workspace_datatree", _fake_open_datatree)

    with manager_context() as manager:
        monkeypatch.setattr(manager, "_is_datatree_workspace", lambda *args: True)
        monkeypatch.setattr(manager, "_from_datatree", _raise_from_datatree)

        with tempfile.TemporaryDirectory() as tmp_dir:
            p = pathlib.Path(tmp_dir) / "workspace.itws"
            p.write_text("placeholder", encoding="utf-8")
            manager.open_multiple_files([p], try_workspace=True)

        QtWidgets.QApplication.processEvents()

        assert len(critical_calls) == 1
        assert manager._alert_dialogs == []


def test_manager_context_starts_cleanly_back_to_back(
    qtbot,
    caplog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with caplog.at_level(logging.ERROR):
        for _ in range(2):
            with manager_context() as manager:
                manager.show()
                qtbot.wait_until(
                    lambda: (
                        manager.server.isRunning()
                        and manager.watcher_server.isRunning()
                    ),
                    timeout=5000,
                )
                qtbot.wait(100)
                QtWidgets.QApplication.processEvents()
                assert manager._alert_dialogs == []

    assert "Address already in use" not in caplog.text
    assert not any(
        record.name == "erlab.interactive.imagetool.manager._server"
        and record.levelno >= logging.ERROR
        for record in caplog.records
    )


def test_manager_selection_info_single_manager(
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    manager_mod = erlab.interactive.imagetool.manager
    with manager_context() as manager:
        info = manager_mod.manager_selection_info()

        assert info["reason"] == "single"
        assert info["resolved_index"] == manager.manager_index
        assert info["needs_selection"] is False
        assert info["default_index"] is None
        assert info["managers"][0]["index"] == manager.manager_index
        assert info["managers"][0]["workspace_path"] is None

        workspace_path = tmp_path / "selected.itws"
        manager._adopt_workspace_path(workspace_path)
        info = manager_mod.manager_selection_info()
        assert info["managers"][0]["workspace_path"] == str(workspace_path.resolve())

        manager_mod.set_default_manager(manager.manager_index)
        info = manager_mod.manager_selection_info()

        assert info["reason"] == "default"
        assert info["default_index"] == manager.manager_index
        assert info["managers"][0]["is_default"] is True


def test_manager_registry_object_repr_and_mapping(monkeypatch, tmp_path) -> None:
    registry = _use_isolated_manager_registry(monkeypatch, tmp_path)
    monkeypatch.setattr(registry, "_pid_exists", lambda _pid: True)
    monkeypatch.setattr(registry, "_is_tcp_port_open", lambda *_args: True)
    manager_mod = erlab.interactive.imagetool.manager

    assert not manager_mod.managers
    assert repr(manager_mod.managers) == "No live ImageTool managers."
    assert "No live ImageTool managers" in manager_mod.managers._repr_html_()

    records = [
        registry.activate_manager_record(
            registry.reserve_manager_record(host="localhost").internal_id,
            port=45555 + idx * 2,
            watch_port=45556 + idx * 2,
        )
        for idx in range(2)
    ]
    workspace_path = str(tmp_path / "workspace.itws")
    registry.refresh_manager_record(
        records[0].internal_id, workspace_path=workspace_path
    )
    manager_mod.set_default_manager(records[1].index)

    handles = tuple(manager_mod.managers)
    assert len(manager_mod.managers) == 2
    assert [manager.index for manager in handles] == [0, 1]
    assert handles[0].workspace_path == workspace_path
    assert handles[1].workspace_path is None
    assert manager_mod.managers.keys() == (0, 1)
    assert [manager.index for manager in manager_mod.managers.values()] == [0, 1]
    assert [
        (index, manager.index) for index, manager in manager_mod.managers.items()
    ] == [(0, 0), (1, 1)]
    assert manager_mod.managers[1].is_default is True

    with pytest.raises(KeyError):
        manager_mod.managers[2]
    with pytest.raises(TypeError, match="Manager index must be an integer"):
        manager_mod.managers[True]

    text = repr(manager_mod.managers)
    assert "Index" in text
    assert "Default" in text
    assert "Workspace" in text
    assert "#0" in text
    assert "#1" in text
    assert "localhost:45555" in text
    assert workspace_path in text
    assert "yes" in text
    assert "ManagerInfo(" not in text
    assert "internal_id" not in text

    html = manager_mod.managers._repr_html_()
    assert "<table" in html
    assert "#0" in html
    assert workspace_path in html
    assert "ManagerInfo(" not in html
    assert "internal_id" not in html
    assert f"workspace={workspace_path!r}" in repr(manager_mod.managers[0])


def test_manager_handle_forwards_to_public_api(
    monkeypatch, tmp_path, test_data
) -> None:
    registry = _use_isolated_manager_registry(monkeypatch, tmp_path)
    monkeypatch.setattr(registry, "_pid_exists", lambda _pid: True)
    monkeypatch.setattr(registry, "_is_tcp_port_open", lambda *_args: True)
    record = registry.activate_manager_record(
        registry.reserve_manager_record(host="localhost").internal_id,
        port=45555,
        watch_port=45556,
    )
    manager_mod = erlab.interactive.imagetool.manager
    handle = manager_mod.managers[record.index]
    calls: list[tuple[str, tuple[typing.Any, ...], dict[str, typing.Any]]] = []

    def record_call(name: str, result: typing.Any):
        def inner(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            calls.append((name, args, kwargs))
            return result

        return inner

    monkeypatch.setattr(manager_mod, "show_in_manager", record_call("show", "shown"))
    monkeypatch.setattr(manager_mod, "load_in_manager", record_call("load", "loaded"))
    monkeypatch.setattr(manager_mod, "replace_data", record_call("replace", "done"))
    monkeypatch.setattr(manager_mod, "fetch", record_call("fetch", test_data))
    monkeypatch.setattr(manager_mod, "watch", record_call("watch", ("data",)))
    monkeypatch.setattr(
        manager_mod, "set_default_manager", record_call("use", record.index)
    )

    assert handle.show(test_data, link=True) == "shown"
    assert handle.load(("path.dat",), loader_name="example") == "loaded"
    assert handle.replace(0, test_data) == "done"
    xr.testing.assert_identical(handle.fetch(0), test_data)
    assert handle.watch("data") == ("data",)
    assert handle.use() == record.index

    assert calls[0][0] == "show"
    assert calls[0][1][0] is test_data
    assert calls[0][2] == {"target": record.index, "link": True}
    assert calls[1] == (
        "load",
        (("path.dat",),),
        {"loader_name": "example", "target": record.index},
    )
    assert calls[2][0] == "replace"
    assert calls[2][1][0] == 0
    assert calls[2][1][1] is test_data
    assert calls[2][2] == {"target": record.index}
    assert calls[3] == ("fetch", (0,), {"target": record.index})
    assert calls[4] == ("watch", ("data",), {"target": record.index})
    assert calls[5] == ("use", (record.index,), {})


def test_manager_registry_hides_starting_records(monkeypatch, tmp_path) -> None:
    registry = _use_isolated_manager_registry(monkeypatch, tmp_path)
    monkeypatch.setattr(registry, "_pid_exists", lambda _pid: True)
    monkeypatch.setattr(registry, "_is_tcp_port_open", lambda *_args: True)

    record = registry.reserve_manager_record(host="localhost")

    assert record.state == "starting"
    assert registry.live_manager_records() == ()
    assert registry.manager_selection_info()["reason"] == "none"

    ready_record = registry.activate_manager_record(
        record.internal_id, port=45555, watch_port=45556
    )

    assert ready_record.state == "ready"
    assert [item.index for item in registry.live_manager_records()] == [record.index]
    assert registry.manager_selection_info()["reason"] == "single"


def test_manager_registry_handles_invalid_records_and_paths(
    monkeypatch, tmp_path
) -> None:
    registry = _use_isolated_manager_registry(monkeypatch, tmp_path)
    monkeypatch.setenv("ITOOL_MANAGER_REGISTRY", str(tmp_path / "custom.json"))

    assert registry._default_registry_path() == tmp_path / "custom.json"
    assert registry._ManagerRecord.from_dict({"state": "unknown"}) is None
    assert registry._ManagerRecord.from_dict({"state": "ready"}) is None

    registry._REGISTRY_PATH.write_text("{", encoding="utf-8")
    assert registry._read_records_unlocked() == []

    registry._REGISTRY_PATH.write_text('{"records": []}', encoding="utf-8")
    assert registry._read_records_unlocked() == []

    record = registry._ManagerRecord(
        internal_id="abc",
        index=0,
        pid=123,
        host="localhost",
        port=45555,
        watch_port=45556,
        started="2026-05-08T10:00:00",
        version="3.22.0",
        heartbeat=100.0,
    )
    registry._REGISTRY_PATH.write_text(
        json.dumps([1, {"state": "unknown"}, dataclasses.asdict(record)]),
        encoding="utf-8",
    )

    assert registry._read_records_unlocked() == [record]


def test_manager_registry_write_open_and_commit_failures(monkeypatch, tmp_path) -> None:
    registry = _use_isolated_manager_registry(monkeypatch, tmp_path)

    class _OpenFailSaveFile:
        def __init__(self, _path: str) -> None:
            return None

        def open(self, _mode) -> bool:
            return False

        def errorString(self) -> str:
            return "open failed"

    monkeypatch.setattr(registry.QtCore, "QSaveFile", _OpenFailSaveFile)
    with pytest.raises(registry.ImageToolManagerRegistryError, match="open failed"):
        registry._write_records_unlocked([])

    class _CommitFailSaveFile:
        def __init__(self, _path: str) -> None:
            return None

        def open(self, _mode) -> bool:
            return True

        def write(self, payload: bytes) -> int:
            return len(payload)

        def commit(self) -> bool:
            return False

        def errorString(self) -> str:
            return "commit failed"

    monkeypatch.setattr(registry.QtCore, "QSaveFile", _CommitFailSaveFile)
    with pytest.raises(registry.ImageToolManagerRegistryError, match="commit failed"):
        registry._write_records_unlocked([])


def test_manager_registry_process_and_activity_helpers(monkeypatch, tmp_path) -> None:
    registry = _use_isolated_manager_registry(monkeypatch, tmp_path)
    record = registry._ManagerRecord(
        internal_id="abc",
        index=0,
        pid=999999,
        host="localhost",
        port=45555,
        watch_port=45556,
        started="2026-05-08T10:00:00",
        version="3.22.0",
        heartbeat=100.0,
    )

    assert registry._pid_exists(0) is False
    with pytest.raises(ValueError, match="Manager index must be >= 0"):
        registry._normalize_manager_index(-1, label="index")

    monkeypatch.setattr(registry.os, "kill", lambda *_args: None)
    assert registry._pid_exists(999999) is True
    monkeypatch.setattr(
        registry.os,
        "kill",
        lambda *_args: (_ for _ in ()).throw(ProcessLookupError),
    )
    assert registry._pid_exists(999999) is False
    monkeypatch.setattr(
        registry.os,
        "kill",
        lambda *_args: (_ for _ in ()).throw(PermissionError),
    )
    assert registry._pid_exists(999999) is True
    monkeypatch.setattr(
        registry.os, "kill", lambda *_args: (_ for _ in ()).throw(OSError)
    )
    assert registry._pid_exists(999999) is False

    monkeypatch.setattr(registry, "_pid_exists", lambda _pid: False)
    assert registry._record_is_active(record, now=101.0) is False

    monkeypatch.setattr(registry, "_pid_exists", lambda _pid: True)
    monkeypatch.setattr(registry, "_is_tcp_port_open", lambda *_args: False)
    assert registry._record_is_active(record, now=101.0) is True
    assert registry._record_is_active(record, now=200.0) is False


def test_manager_registry_does_not_reuse_holes(monkeypatch, tmp_path) -> None:
    registry = _use_isolated_manager_registry(monkeypatch, tmp_path)
    monkeypatch.setattr(registry, "_pid_exists", lambda _pid: True)
    monkeypatch.setattr(registry, "_is_tcp_port_open", lambda *_args: True)

    records = [
        registry.activate_manager_record(
            registry.reserve_manager_record(host="localhost").internal_id,
            port=45555 + idx * 2,
            watch_port=45556 + idx * 2,
        )
        for idx in range(3)
    ]

    registry.unregister_manager_record(records[1].internal_id)
    new_record = registry.reserve_manager_record(host="localhost")

    assert [record.index for record in records] == [0, 1, 2]
    assert new_record.index == 3

    for record in (*records[::2], new_record):
        registry.unregister_manager_record(record.internal_id)

    assert registry.reserve_manager_record(host="localhost").index == 0


def test_manager_registry_removes_stale_starting_records(monkeypatch, tmp_path) -> None:
    registry = _use_isolated_manager_registry(monkeypatch, tmp_path)
    monkeypatch.setattr(registry, "_pid_exists", lambda _pid: True)
    current_time = 1000.0
    monkeypatch.setattr(registry.time, "time", lambda: current_time)

    registry.reserve_manager_record(host="localhost")
    current_time += registry._STARTUP_GRACE_S + 1.0

    assert registry.live_manager_records() == ()
    with registry._registry_lock():
        assert registry._read_records_unlocked() == []


def test_manager_registry_target_resolution_edges(monkeypatch, tmp_path) -> None:
    registry = _use_isolated_manager_registry(monkeypatch, tmp_path)
    monkeypatch.setattr(registry, "_pid_exists", lambda _pid: True)
    monkeypatch.setattr(registry, "_is_tcp_port_open", lambda *_args: True)

    with pytest.raises(ValueError, match="No live ImageTool manager"):
        registry.set_default_manager(0)
    with pytest.raises(registry.ImageToolManagerNotFoundError, match="not running"):
        registry.resolve_manager_record()

    records = [
        registry.activate_manager_record(
            registry.reserve_manager_record(host="localhost").internal_id,
            port=45555 + idx * 2,
            watch_port=45556 + idx * 2,
        )
        for idx in range(2)
    ]

    with pytest.raises(registry.ImageToolManagerNotFoundError, match="index 99"):
        registry.resolve_manager_record(99)
    with pytest.raises(registry.ImageToolManagerAmbiguousError, match="Multiple"):
        registry.resolve_manager_record()

    registry._default_manager_index = 99
    assert registry.get_default_manager() is None
    registry._default_manager_index = 99
    info = registry.manager_selection_info()
    assert info["reason"] == "multiple"
    assert info["default_index"] is None

    registry.set_default_manager(records[0].index)
    assert registry.resolve_manager_record() == records[0]
    with registry.use_manager(records[1].index):
        assert registry.get_default_manager() == records[1].index
    assert registry.get_default_manager() == records[0].index

    with (
        pytest.raises(RuntimeError, match="boom"),
        registry.use_manager(records[1].index),
    ):
        raise RuntimeError("boom")
    assert registry.get_default_manager() == records[0].index


def test_manager_registry_write_failure_preserves_existing_file(
    monkeypatch, tmp_path
) -> None:
    registry = _use_isolated_manager_registry(monkeypatch, tmp_path)
    registry._REGISTRY_PATH.write_text("[]\n", encoding="utf-8")
    cancelled: list[bool] = []

    class _FailingSaveFile:
        def __init__(self, _path: str) -> None:
            return None

        def open(self, _mode) -> bool:
            return True

        def write(self, payload: bytes) -> int:
            return len(payload) - 1

        def cancelWriting(self) -> None:
            cancelled.append(True)

        def errorString(self) -> str:
            return "partial write"

    monkeypatch.setattr(registry.QtCore, "QSaveFile", _FailingSaveFile)

    with pytest.raises(registry.ImageToolManagerRegistryError, match="partial write"):
        registry._write_records_unlocked([])

    assert cancelled == [True]
    assert registry._REGISTRY_PATH.read_text(encoding="utf-8") == "[]\n"


def test_manager_registry_lock_assigns_unique_concurrent_indexes(tmp_path) -> None:
    registry_path = tmp_path / "concurrent-managers.json"
    worker_code = """
import pathlib
import sys
import time

import erlab.interactive.imagetool.manager._registry as registry

registry._REGISTRY_PATH = pathlib.Path(sys.argv[1])
registry._LOCK_PATH = registry._REGISTRY_PATH.with_suffix(
    registry._REGISTRY_PATH.suffix + ".lock"
)
record = registry.reserve_manager_record(host="localhost")
print(record.index, flush=True)
time.sleep(1.0)
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", worker_code, str(registry_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(3)
    ]

    outputs = [process.communicate(timeout=15) for process in processes]
    indexes = sorted(int(stdout.strip()) for stdout, _stderr in outputs)

    assert indexes == [0, 1, 2]
    for process, (_stdout, stderr) in zip(processes, outputs, strict=True):
        assert process.returncode == 0, stderr


def test_itool_magic_manager_target_normalization() -> None:
    assert _normalize_manager_target_args("-m data") == "-m data"
    assert _normalize_manager_target_args("-m 1 data") == "-m --manager-index 1 data"
    assert (
        _normalize_manager_target_args("--manager=2 data")
        == "--manager --manager-index 2 data"
    )


def test_itool_manager_accepts_index_like_values(monkeypatch, test_data) -> None:
    calls: list[dict[str, typing.Any]] = []

    monkeypatch.setattr(
        erlab.interactive.imagetool.manager,
        "is_running",
        lambda target=None: True,
    )
    monkeypatch.setattr(
        erlab.interactive.imagetool.manager,
        "show_in_manager",
        lambda _data, **kwargs: calls.append(kwargs),
    )

    itool(test_data, manager=np.int64(2))
    itool(test_data, manager=True)

    assert calls[0]["target"] == 2
    assert calls[1]["target"] is None


def test_itool_manager_invalid_index_object_and_unavailable_fallback(
    monkeypatch, test_data
) -> None:
    warnings: list[str] = []
    running_targets: list[int | None] = []

    class _DummyImageTool:
        def __init__(self, data, **_kwargs) -> None:
            self.data = data

        def show(self) -> None:
            return None

        def activateWindow(self) -> None:
            return None

        def raise_(self) -> None:
            return None

    monkeypatch.setattr(itool_mod, "_parse_input", lambda _data: [test_data])
    monkeypatch.setattr(erlab.interactive.imagetool, "ImageTool", _DummyImageTool)
    monkeypatch.setattr(
        erlab.interactive.imagetool.manager,
        "is_running",
        lambda target=None: running_targets.append(target) or False,
    )
    monkeypatch.setattr(
        erlab.utils.misc,
        "emit_user_level_warning",
        lambda message: warnings.append(message),
    )

    direct_tool = itool(test_data, manager=object(), execute=False)
    fallback_tool = itool(test_data, manager=3, execute=False)

    assert isinstance(direct_tool, _DummyImageTool)
    assert isinstance(fallback_tool, _DummyImageTool)
    assert running_targets == [3]
    assert warnings == [
        "The manager is not running. Opening the ImageTool window(s) directly."
    ]


def test_manager_target_validation_rejects_bool(monkeypatch, tmp_path) -> None:
    _use_isolated_manager_registry(monkeypatch, tmp_path)

    assert erlab.interactive.imagetool.manager.is_running(np.int64(0)) is False
    with pytest.raises(TypeError, match="Manager target must be an integer"):
        erlab.interactive.imagetool.manager.is_running(True)


def test_multi_manager_integer_targets(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    manager_mod = erlab.interactive.imagetool.manager
    with manager_context() as manager0:
        manager1 = ImageToolManager()
        qtbot.addWidget(manager1, before_close_func=lambda w: w.remove_all_tools())
        try:
            qtbot.wait_until(
                lambda: (
                    manager0.server.isRunning()
                    and manager0.watcher_server.isRunning()
                    and manager1.server.isRunning()
                    and manager1.watcher_server.isRunning()
                ),
                timeout=5000,
            )

            info = manager_mod.manager_selection_info()
            assert info["reason"] == "multiple"
            assert info["needs_selection"] is True
            assert [item["index"] for item in info["managers"]] == [
                manager0.manager_index,
                manager1.manager_index,
            ]

            test_data.qshow(manager=manager0.manager_index)
            qtbot.wait_until(lambda: manager0.ntools == 1, timeout=5000)
            assert manager1.ntools == 0

            (test_data + 1).qshow(manager=manager1.manager_index)
            qtbot.wait_until(lambda: manager1.ntools == 1, timeout=5000)
            assert manager0.ntools == 1
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(fetch, 0, target=manager1.manager_index)
                qtbot.waitUntil(lambda: fut.done(), timeout=10000)
                fetched = fut.result()
            xr.testing.assert_identical(fetched, test_data + 1)
        finally:
            manager1.remove_all_tools()
            manager1.close()
            qtbot.wait_until(
                lambda: (
                    not manager1.server.isRunning()
                    and not manager1.watcher_server.isRunning()
                ),
                timeout=5000,
            )
            manager1.deleteLater()


def test_watcher_server_run_stops_cleanly_without_pending_payload(monkeypatch) -> None:
    class _DummySocket:
        def __init__(self) -> None:
            self.bound: list[str] = []
            self.closed = False
            self.sent: list[dict[str, str]] = []

        def setsockopt(self, *args) -> None:
            return None

        def bind(self, address: str) -> None:
            self.bound.append(address)

        def send_json(self, payload: dict[str, str]) -> None:
            self.sent.append(payload)

        def close(self) -> None:
            self.closed = True

    class _DummyContext:
        def __init__(self, socket: _DummySocket) -> None:
            self._socket = socket

        def socket(self, *_args) -> _DummySocket:
            return self._socket

    class _DummyWaitCondition:
        def __init__(self, server: _WatcherServer) -> None:
            self.server = server
            self.calls = 0

        def wait(self, *_args) -> bool:
            self.calls += 1
            self.server.stopped.set()
            return True

        def wakeAll(self) -> None:
            return None

    socket = _DummySocket()
    monkeypatch.setattr(
        zmq.Context, "instance", staticmethod(lambda: _DummyContext(socket))
    )

    server = _WatcherServer()
    wait_condition = _DummyWaitCondition(server)
    server._cv = wait_condition

    server.run()

    assert wait_condition.calls == 1
    assert socket.bound == [f"tcp://*:{erlab.interactive.imagetool.manager.PORT_WATCH}"]
    assert socket.sent == []
    assert socket.closed


def test_manager_server_wait_until_bound_errors() -> None:
    watcher_server = _WatcherServer()
    with pytest.raises(TimeoutError, match="watcher server"):
        watcher_server.wait_until_bound(timeout_ms=1)
    watcher_error = RuntimeError("watcher bind failed")
    watcher_server._bind_error = watcher_error
    watcher_server._bound_event.set()
    with pytest.raises(RuntimeError, match="Watcher server failed") as exc_info:
        watcher_server.wait_until_bound(timeout_ms=1)
    assert exc_info.value.__cause__ is watcher_error

    manager = manager_server._ManagerServer()
    with pytest.raises(TimeoutError, match="manager server"):
        manager.wait_until_bound(timeout_ms=1)
    manager_error = RuntimeError("manager bind failed")
    manager._bind_error = manager_error
    manager._bound_event.set()
    with pytest.raises(RuntimeError, match="Manager server failed") as exc_info:
        manager.wait_until_bound(timeout_ms=1)
    assert exc_info.value.__cause__ is manager_error


def test_manager_server_bind_failures_close_socket(monkeypatch) -> None:
    class _FailingSocket:
        def __init__(self) -> None:
            self.closed = False

        def setsockopt(self, *args) -> None:
            return None

        def bind(self, _address: str) -> None:
            raise RuntimeError("bind failed")

        def close(self) -> None:
            self.closed = True

    class _DummyContext:
        def __init__(self, socket: _FailingSocket) -> None:
            self._socket = socket

        def socket(self, *_args) -> _FailingSocket:
            return self._socket

    watcher_socket = _FailingSocket()
    monkeypatch.setattr(
        zmq.Context,
        "instance",
        staticmethod(lambda: _DummyContext(watcher_socket)),
    )
    watcher_server = _WatcherServer(port=45556)
    watcher_server.run()
    assert watcher_server._bound_event.is_set()
    assert isinstance(watcher_server._bind_error, RuntimeError)
    assert watcher_socket.closed

    manager_socket = _FailingSocket()
    monkeypatch.setattr(
        zmq.Context,
        "instance",
        staticmethod(lambda: _DummyContext(manager_socket)),
    )
    manager = manager_server._ManagerServer(port=45555)
    manager.run()
    assert manager._bound_event.is_set()
    assert isinstance(manager._bind_error, RuntimeError)
    assert manager_socket.closed


def test_manager_server_wait_for_return_value_stops_cleanly() -> None:
    manager = manager_server._ManagerServer()
    manager.stopped.set()

    assert manager._wait_for_return_value() is manager_server._UNSET


def test_manager_server_run_returns_watch_info(monkeypatch) -> None:
    sent: list[dict[str, typing.Any]] = []

    class _DummySocket:
        def __init__(self) -> None:
            self.closed = False
            self.bound: list[str] = []

        def setsockopt(self, *args) -> None:
            return None

        def bind(self, address: str) -> None:
            self.bound.append(address)

        def close(self) -> None:
            self.closed = True

    class _DummyContext:
        def __init__(self, socket: _DummySocket) -> None:
            self._socket = socket

        def socket(self, *_args) -> _DummySocket:
            return self._socket

    socket = _DummySocket()
    manager = manager_server._ManagerServer(port=45555)
    watch_info = {"workspace_link_id": "workspace-1", "watched": []}

    monkeypatch.setattr(
        zmq.Context, "instance", staticmethod(lambda: _DummyContext(socket))
    )
    monkeypatch.setattr(
        manager_server,
        "_recv_multipart",
        lambda _socket: {"packet_type": "command", "command": "watch-info"},
    )
    monkeypatch.setattr(
        manager_server,
        "_send_multipart",
        lambda _socket, payload, **_kwargs: sent.append(payload),
    )
    manager.sigWatchInfoRequested.connect(
        lambda: (manager.set_return_value(watch_info), manager.stopped.set())
    )

    manager.run()

    assert sent == [{"status": "ok", "watch_info": watch_info}]
    assert socket.bound == ["tcp://*:45555"]
    assert socket.closed


def test_manager_server_run_errors_when_data_request_stops(monkeypatch) -> None:
    sent: list[dict[str, typing.Any]] = []

    class _DummySocket:
        def __init__(self) -> None:
            self.closed = False

        def setsockopt(self, *args) -> None:
            return None

        def bind(self, _address: str) -> None:
            return None

        def close(self) -> None:
            self.closed = True

    class _DummyContext:
        def __init__(self, socket: _DummySocket) -> None:
            self._socket = socket

        def socket(self, *_args) -> _DummySocket:
            return self._socket

    socket = _DummySocket()
    manager = manager_server._ManagerServer(port=45555)

    monkeypatch.setattr(
        zmq.Context, "instance", staticmethod(lambda: _DummyContext(socket))
    )
    monkeypatch.setattr(
        manager_server,
        "_recv_multipart",
        lambda _socket: {
            "packet_type": "command",
            "command": "get-data",
            "command_arg": 0,
        },
    )
    monkeypatch.setattr(
        manager_server,
        "_send_multipart",
        lambda _socket, payload, **_kwargs: sent.append(payload),
    )
    manager.sigDataRequested.connect(lambda _arg: manager.stopped.set())

    manager.run()

    assert sent == [{"status": "error"}]
    assert socket.closed


def test_manager_server_run_errors_when_watch_info_request_stops(monkeypatch) -> None:
    sent: list[dict[str, typing.Any]] = []

    class _DummySocket:
        def __init__(self) -> None:
            self.closed = False

        def setsockopt(self, *args) -> None:
            return None

        def bind(self, _address: str) -> None:
            return None

        def close(self) -> None:
            self.closed = True

    class _DummyContext:
        def __init__(self, socket: _DummySocket) -> None:
            self._socket = socket

        def socket(self, *_args) -> _DummySocket:
            return self._socket

    socket = _DummySocket()
    manager = manager_server._ManagerServer(port=45555)

    monkeypatch.setattr(
        zmq.Context, "instance", staticmethod(lambda: _DummyContext(socket))
    )
    monkeypatch.setattr(
        manager_server,
        "_recv_multipart",
        lambda _socket: {"packet_type": "command", "command": "watch-info"},
    )
    monkeypatch.setattr(
        manager_server,
        "_send_multipart",
        lambda _socket, payload, **_kwargs: sent.append(payload),
    )
    manager.sigWatchInfoRequested.connect(lambda: manager.stopped.set())

    manager.run()

    assert sent == [{"status": "error"}]
    assert socket.closed


def test_manager_server_client_helpers_target_socket_branches(
    monkeypatch, tmp_path, test_data
) -> None:
    calls: list[tuple[typing.Any, dict[str, typing.Any]]] = []
    record = manager_registry._ManagerRecord(
        internal_id="abc",
        index=2,
        pid=123,
        host="localhost",
        port=45555,
        watch_port=45556,
        started="2026-05-08T10:00:00",
        version="3.22.0",
        heartbeat=100.0,
    )

    monkeypatch.setattr(erlab.interactive.imagetool.manager, "_manager_instance", None)
    monkeypatch.setattr(
        manager_server, "resolve_manager_record", lambda target=None: record
    )

    def _query_zmq(payload, **kwargs):
        calls.append((payload, kwargs))
        if (
            isinstance(payload, manager_server.CommandPacket)
            and payload.command == "watch-info"
        ):
            return Response(status="ok", watch_info={"watched": []})
        return Response(status="ok", data=test_data)

    monkeypatch.setattr(manager_server, "_query_zmq", _query_zmq)

    path = tmp_path / "data.dat"
    path.write_text("data", encoding="utf-8")
    manager_server.load_in_manager((path,), target=2)
    manager_server.show_in_manager(None, target=2)
    manager_server.replace_data(0, test_data, target=2)
    manager_server._watch_data("data", "uid", test_data, show=True, target=2)
    assert manager_server._unwatch_data("uid", target=2).status == "ok"
    assert manager_server._unwatch_data("uid", remove=True, target=2).status == "ok"
    xr.testing.assert_identical(manager_server.fetch("uid", target=2), test_data)
    assert manager_server.watch_info(target=2) == {"watched": []}
    with pytest.warns(DeprecationWarning, match="watch_data"):
        manager_server.watch_data("data", "uid", test_data)
    with pytest.warns(DeprecationWarning, match="unwatch_data"):
        assert manager_server.unwatch_data("uid").status == "ok"

    with pytest.raises(FileNotFoundError):
        manager_server.load_in_manager((tmp_path / "missing.dat",), target=2)
    with pytest.raises(ValueError, match="Mismatch"):
        manager_server.replace_data([0, 1], test_data, target=2)

    packet_types = [payload.packet_type for payload, _kwargs in calls]
    assert packet_types == [
        "open",
        "add",
        "replace",
        "watch",
        "command",
        "command",
        "command",
        "command",
        "command",
        "watch",
        "command",
    ]
    assert all(kwargs["record"] == record for _payload, kwargs in calls)


def test_manager_handle_watch_info_delegates_to_target(monkeypatch) -> None:
    handle = manager_server.ImageToolManagerHandle(
        index=7,
        pid=123,
        host="localhost",
        port=45555,
        watch_port=45556,
        started="2026-05-08T10:00:00",
        version="3.22.0",
    )

    monkeypatch.setattr(
        erlab.interactive.imagetool.manager,
        "watch_info",
        lambda *, target=None: {"target": target},
    )

    assert handle.watch_info() == {"target": 7}


def test_manager_progressbar_alert(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        message = "Load data:   8%|8         | 1/12 [00:00<00:07,  1.54it/s]"
        manager._show_alert("INFO", logging.INFO, message, "")

        assert 12 in manager._progress_bars
        pbar = manager._progress_bars[12]
        assert pbar.labelText() == "Load data"
        assert pbar.value() == 1
        assert manager._alert_dialogs == []

        manager._show_alert(
            "INFO",
            logging.INFO,
            "Load data:  50%|##        | 6/12 [00:00<00:07,  1.54it/s]",
            "",
        )
        assert manager._progress_bars[12] is pbar
        assert pbar.value() == 6

        pbar.close()


def test_manager_alert_icons(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    recorded_icons: list[QtWidgets.QStyle.StandardPixmap | None] = []

    class _RecordingMessageDialog(erlab.interactive.utils.MessageDialog):
        def __init__(self, *args, icon_pixmap=None, **kwargs):
            recorded_icons.append(icon_pixmap)
            super().__init__(*args, icon_pixmap=icon_pixmap, **kwargs)

    monkeypatch.setattr(
        erlab.interactive.utils, "MessageDialog", _RecordingMessageDialog
    )

    with manager_context() as manager:
        manager._show_alert("INFO", logging.INFO, "info", "")
        manager._show_alert("WARNING", logging.WARNING, "warning", "")
        manager._show_alert("ERROR", logging.ERROR, "error", "")

        assert recorded_icons == [
            QtWidgets.QStyle.StandardPixmap.SP_MessageBoxInformation,
            QtWidgets.QStyle.StandardPixmap.SP_MessageBoxWarning,
            QtWidgets.QStyle.StandardPixmap.SP_MessageBoxCritical,
        ]

        manager._clear_all_alerts()
        QtWidgets.QApplication.processEvents()


def test_uncaught_exception_alert(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        assert sys.excepthook == manager._handle_uncaught_exception
        manager._previous_excepthook = lambda *exc: None

        try:
            raise RuntimeError("boom")  # noqa: TRY301
        except RuntimeError:
            exc_info = sys.exc_info()

        sys.excepthook(*exc_info)

        qtbot.wait_until(lambda: len(manager._alert_dialogs) == 1)
        qtbot.wait_until(manager._alert_dialogs[0].isVisible)

        text = manager._alert_dialogs[0].text()
        detailed_text = manager._alert_dialogs[0].detailedText()

        assert "ERROR" in manager._alert_dialogs[0].windowTitle()
        assert "An unexpected error occurred" in text
        assert detailed_text is not None
        assert "boom" in detailed_text
        assert "RuntimeError" in detailed_text


def test_manager_reload(
    qtbot,
    example_loader,
    example_data_dir,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        manager.activateWindow()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        gc_enabled = gc.isenabled()
        gc.disable()
        try:
            load_in_manager(
                [example_data_dir / "data_006.h5"], loader_name=example_loader.name
            )
            qtbot.wait_until(
                lambda: manager.ntools == 1 and len(manager._file_handlers) == 0,
                timeout=5000,
            )
        finally:
            if gc_enabled:
                gc.enable()

        # Try reload
        with qtbot.wait_signal(manager.get_imagetool(0).slicer_area.sigDataChanged):
            manager.get_imagetool(0).slicer_area.reload()

        # Try archive
        manager._imagetool_wrappers[0].archive()
        qtbot.wait_until(lambda: manager._imagetool_wrappers[0].archived, timeout=5000)

        # Unarchive
        manager._imagetool_wrappers[0].unarchive()
        qtbot.wait_until(
            lambda: not manager._imagetool_wrappers[0].archived, timeout=5000
        )

        # Try reload again
        with qtbot.wait_signal(manager.get_imagetool(0).slicer_area.sigDataChanged):
            manager.get_imagetool(0).slicer_area.reload()


def make_dataarray_unpicklable(darr):
    mod = types.ModuleType("temp_mod_for_apply_ufunc")
    exec(  # noqa: S102
        "def myfunc(dat):\n    return dat * 2\n",
        mod.__dict__,
    )
    sys.modules[mod.__name__] = mod
    from temp_mod_for_apply_ufunc import myfunc

    darr = xr.apply_ufunc(myfunc, darr.chunk(), vectorize=True, dask="parallelized")
    return darr, mod.__name__


def _create_frames(
    obj: dict[str, typing.Any], pickler_cls: type[pickle.Pickler]
) -> list[memoryview]:
    buffers: list[pickle.PickleBuffer] = []  # out-of-band frames will be appended here
    bio = io.BytesIO()

    p = pickler_cls(bio, protocol=5, buffer_callback=buffers.append)
    p.dump(obj)
    header = memoryview(bio.getbuffer())
    return [header] + [memoryview(b) for b in buffers]


def test_manager_cloudpickle(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context(use_socket=True) as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        # Test with unpicklable data
        darr, modname = make_dataarray_unpicklable(test_data)
        del sys.modules[modname]
        erlab.interactive.imagetool.manager.show_in_manager(darr)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        logger.info("Confirmed tool is added, checking data")
        assert manager.get_imagetool(0).array_slicer.point_value(0) == 24.0

        # Pickle data first and remove module before trying unpickle
        darr, modname = make_dataarray_unpicklable(test_data)
        content = AddDataPacket(
            packet_type="add", data_list=darr, arguments={}
        ).model_dump(exclude_unset=True)
        ctx = zmq.Context.instance()
        sock: zmq.Socket = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.SNDHWM, 0)
        sock.setsockopt(zmq.RCVHWM, 0)
        try:
            sock.connect(
                "tcp://"
                f"{erlab.interactive.imagetool.manager.HOST_IP}:"
                f"{erlab.interactive.imagetool.manager.PORT}"
            )
            frames = _create_frames(content, pickler_cls=pickle.Pickler)
            del sys.modules[modname]
            sock.send_multipart(frames, copy=False)

            timeout_seconds = 5.0
            start_time = time.time()
            while True:
                try:
                    response = Response(**_recv_multipart(sock, flags=zmq.NOBLOCK))
                except zmq.Again as e:
                    if time.time() - start_time > timeout_seconds:
                        raise TimeoutError(
                            "Timed out waiting for response from ZeroMQ socket."
                        ) from e
                else:
                    break
            assert response.status == "unpickle-failed"
        finally:
            sock.close()


@pytest.mark.parametrize(
    (
        "old_version",
        "new_version",
        "button_text",
        "expected_url",
        "expected_title",
        "expected_info",
    ),
    [
        (
            "",
            "1.2.3",
            "Open Release Notes",
            "https://github.com/kmnhan/erlabpy/releases",
            "ImageTool Manager Installed",
            "Welcome to ImageTool Manager! You are using version 1.2.3.",
        ),
        (
            "1.0.0",
            "1.1.0",
            "Open Documentation",
            "https://erlabpy.readthedocs.io/en/stable/user-guide/interactive/imagetool.html",
            "ImageTool Manager Updated",
            "ImageTool Manager has been successfully updated from version 1.0.0 to "
            "1.1.0.",
        ),
    ],
)
def test_manager_updated_opens_links(
    qtbot,
    accept_dialog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
    monkeypatch,
    old_version,
    new_version,
    button_text,
    expected_url,
    expected_title,
    expected_info,
) -> None:
    opened: list[str] = []

    def _open(url: str) -> bool:
        opened.append(url)
        return True

    def _accept_call(dialog: QtWidgets.QMessageBox) -> None:
        assert dialog.text() == expected_title
        assert dialog.informativeText() == expected_info
        for button in dialog.buttons():
            if button.text() == button_text:
                button.click()
                return
        pytest.fail(f"Button {button_text!r} not found.")

    monkeypatch.setattr(webbrowser, "open", _open)

    with manager_context() as manager:
        manager.show()
        accept_dialog(
            lambda: manager.updated(old_version=old_version, new_version=new_version),
            accept_call=_accept_call,
        )

    assert opened == [expected_url]


def test_manager_standalone_app_menus(
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    launched: list[bool] = []
    monkeypatch.setattr(
        manager_mainwindow,
        "_launch_new_manager_instance",
        lambda: launched.append(True),
    )

    with manager_context() as manager:
        menu_actions = {
            action.text().replace("&", ""): action
            for action in manager.menu_bar.actions()
            if action.menu() is not None
        }

        assert "File" in menu_actions
        assert "Apps" in menu_actions
        file_actions = action_map(
            typing.cast("QtWidgets.QMenu", menu_actions["File"].menu())
        )
        assert "Data Explorer" in file_actions
        assert "New Manager Instance" in file_actions
        assert file_actions["New Manager Instance"].shortcut().isEmpty()
        file_actions["New Manager Instance"].trigger()
        assert launched == [True]

        apps_actions = action_map(
            typing.cast("QtWidgets.QMenu", menu_actions["Apps"].menu())
        )
        assert "Periodic Table" in apps_actions
        assert (
            apps_actions["Periodic Table"]
            .shortcut()
            .toString(QtGui.QKeySequence.SequenceFormat.PortableText)
            == "Ctrl+Shift+P"
        )


def test_launch_new_manager_instance_uses_detached_source_process(monkeypatch) -> None:
    calls: list[tuple[list[str], dict[str, typing.Any]]] = []

    monkeypatch.setattr(erlab.utils.misc, "_IS_PACKAGED", False)
    monkeypatch.setattr(manager_mainwindow.sys, "platform", "linux")
    monkeypatch.setattr(manager_mainwindow.sys, "executable", "/env/bin/python")
    monkeypatch.setattr(
        manager_mainwindow.subprocess,
        "Popen",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    manager_mainwindow._launch_new_manager_instance()

    assert calls == [
        (
            ["/env/bin/python", "-m", "erlab.interactive.imagetool.manager"],
            {
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
                "close_fds": True,
                "start_new_session": True,
            },
        )
    ]


def test_launch_new_manager_instance_uses_macos_app_bundle(
    monkeypatch, tmp_path
) -> None:
    calls: list[tuple[list[str], dict[str, typing.Any]]] = []
    app_bundle = tmp_path / "ImageTool Manager.app"
    executable = app_bundle / "Contents" / "MacOS" / "ImageTool Manager"
    executable.parent.mkdir(parents=True)
    executable.touch()

    monkeypatch.setattr(erlab.utils.misc, "_IS_PACKAGED", True)
    monkeypatch.setattr(manager_mainwindow.sys, "platform", "darwin")
    monkeypatch.setattr(manager_mainwindow.sys, "executable", str(executable))
    monkeypatch.setattr(
        manager_mainwindow.subprocess,
        "Popen",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    manager_mainwindow._launch_new_manager_instance()

    assert calls == [
        (
            ["/usr/bin/open", "-n", str(app_bundle.resolve())],
            {
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
                "close_fds": True,
                "start_new_session": True,
            },
        )
    ]


def test_launch_new_manager_instance_uses_windows_detached_flags(monkeypatch) -> None:
    calls: list[tuple[list[str], dict[str, typing.Any]]] = []

    monkeypatch.setattr(erlab.utils.misc, "_IS_PACKAGED", False)
    monkeypatch.setattr(manager_mainwindow.sys, "platform", "win32")
    monkeypatch.setattr(manager_mainwindow.sys, "executable", r"C:\env\python.exe")
    monkeypatch.setattr(
        manager_mainwindow.subprocess, "DETACHED_PROCESS", 8, raising=False
    )
    monkeypatch.setattr(
        manager_mainwindow.subprocess,
        "CREATE_NEW_PROCESS_GROUP",
        512,
        raising=False,
    )
    monkeypatch.setattr(
        manager_mainwindow.subprocess,
        "Popen",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    manager_mainwindow._launch_new_manager_instance()

    assert calls == [
        (
            [r"C:\env\python.exe", "-m", "erlab.interactive.imagetool.manager"],
            {
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
                "close_fds": True,
                "creationflags": 520,
            },
        )
    ]


def test_open_new_manager_instance_shows_error_dialog(monkeypatch) -> None:
    dialogs: list[tuple[object, str, str]] = []

    monkeypatch.setattr(
        manager_mainwindow,
        "_launch_new_manager_instance",
        lambda: (_ for _ in ()).throw(RuntimeError("launch failed")),
    )
    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        lambda parent, *, title, text: dialogs.append((parent, title, text)),
    )

    parent = object()
    manager_mainwindow.ImageToolManager.open_new_manager_instance(parent)

    assert dialogs == [
        (
            parent,
            "New Manager Instance",
            "Could not open another ImageTool Manager instance.",
        )
    ]


def test_manager_explorer_launcher_reuses_instance_and_opens_directory_tabs(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    class _TrackingTabbedExplorer(_TabbedExplorer):
        def __init__(self, *args, **kwargs) -> None:
            self.close_event_count = 0
            super().__init__(*args, **kwargs)

        def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:
            self.close_event_count += 1
            super().closeEvent(event)

    with (
        manager_context() as manager,
        tempfile.TemporaryDirectory() as recent_dir,
        tempfile.TemporaryDirectory() as dropped_dir,
    ):
        manager._recent_directory = recent_dir
        spec = manager._standalone_app_specs["explorer"]
        manager._standalone_app_specs["explorer"] = dataclasses.replace(
            spec,
            factory=lambda: _TrackingTabbedExplorer(
                root_path=manager._recent_directory,
                loader_name=manager._recent_loader_name,
            ),
        )

        manager.ensure_explorer_initialized()
        explorer = manager.explorer

        assert isinstance(explorer, _TrackingTabbedExplorer)
        assert hasattr(manager, "explorer")
        assert explorer.tab_widget.count() == 1

        explorer.close_tab(0)
        qtbot.wait_until(lambda: not explorer.isVisible())
        assert explorer.close_event_count == 0
        manager.show_explorer()
        qtbot.wait_until(explorer.isVisible)
        assert manager.explorer is explorer

        explorer.hide()
        manager.show_explorer()
        qtbot.wait_until(explorer.isVisible)
        assert manager.explorer is explorer

        explorer.close()
        qtbot.wait_until(lambda: not explorer.isVisible())
        manager.show_explorer()
        qtbot.wait_until(explorer.isVisible)
        assert manager.explorer is explorer

        manager.open_multiple_files([pathlib.Path(dropped_dir)], try_workspace=True)

        qtbot.wait_until(lambda: explorer.tab_widget.count() == 2)
        assert explorer.current_explorer is not None
        assert explorer.current_explorer.current_directory == pathlib.Path(dropped_dir)


def test_manager_ptable_launcher_reuses_instance_without_affecting_tree(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    class _TrackingPeriodicTableWindow(PeriodicTableWindow):
        def __init__(self) -> None:
            self.close_event_count = 0
            super().__init__()

        def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:
            self.close_event_count += 1
            super().closeEvent(event)

    with manager_context() as manager:
        initial_ntools = manager.ntools
        initial_rows = manager.tree_view.model().rowCount(QtCore.QModelIndex())
        spec = manager._standalone_app_specs["ptable"]
        manager._standalone_app_specs["ptable"] = dataclasses.replace(
            spec,
            factory=_TrackingPeriodicTableWindow,
        )

        manager.show_ptable()
        ptable = manager.ptable_window

        qtbot.wait_until(ptable.isVisible)
        assert isinstance(ptable, _TrackingPeriodicTableWindow)
        assert manager.ntools == initial_ntools
        assert manager.tree_view.model().rowCount(QtCore.QModelIndex()) == initial_rows

        ptable.search_edit.setFocus(QtCore.Qt.FocusReason.ShortcutFocusReason)
        qtbot.keyClick(
            ptable.search_edit,
            QtCore.Qt.Key.Key_W,
            QtCore.Qt.KeyboardModifier.ControlModifier,
        )
        qtbot.wait_until(lambda: not ptable.isVisible())
        manager.show_ptable()
        qtbot.wait_until(ptable.isVisible)
        assert manager.ptable_window is ptable
        assert ptable.close_event_count == 0

        ptable.hide()
        manager.show_ptable()
        qtbot.wait_until(ptable.isVisible)
        assert manager.ptable_window is ptable

        ptable.close()
        qtbot.wait_until(lambda: not ptable.isVisible())
        manager.show_ptable()
        qtbot.wait_until(ptable.isVisible)
        assert manager.ptable_window is ptable
        assert manager.ntools == initial_ntools


def test_manager_close_event_closes_standalone_apps(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show_explorer()
        manager.show_ptable()

        explorer = manager.explorer
        ptable = manager.ptable_window

        qtbot.wait_until(explorer.isVisible)
        qtbot.wait_until(ptable.isVisible)

        manager.close()
        QtWidgets.QApplication.sendPostedEvents(None, 0)
        QtWidgets.QApplication.processEvents()

        assert manager._standalone_app_windows == {}
        assert not erlab.interactive.utils.qt_is_valid(explorer, ptable)
