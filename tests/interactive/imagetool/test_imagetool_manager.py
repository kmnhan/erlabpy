import ast
import concurrent.futures
import contextlib
import gc
import io
import json
import logging
import pathlib
import pickle
import sys
import tempfile
import time
import types
import typing
import webbrowser
from collections.abc import Callable

import numpy as np
import pytest
import xarray as xr
import xarray.testing
import zmq
from IPython.core.interactiveshell import InteractiveShell
from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive._fit1d import Fit1DTool
from erlab.interactive._fit2d import Fit2DTool
from erlab.interactive._mesh import MeshTool
from erlab.interactive.derivative import DerivativeTool
from erlab.interactive.explorer._tabbed_explorer import _TabbedExplorer
from erlab.interactive.fermiedge import GoldTool, ResolutionTool
from erlab.interactive.imagetool import itool
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
    _ConcatDialog,
    _NameFilterDialog,
    _RenameDialog,
)
from erlab.interactive.imagetool.manager._modelview import (
    _MIME,
    _NODE_UID_ROLE,
    _ImageToolWrapperItemDelegate,
    _ImageToolWrapperItemModel,
)
from erlab.interactive.imagetool.manager._server import (
    AddDataPacket,
    Response,
    _recv_multipart,
    _remove_idx,
    _show_idx,
    _WatcherServer,
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


def copy_full_code_for_uid(
    monkeypatch,
    manager: ImageToolManager,
    uid: str,
) -> str:
    copied: list[str] = []
    monkeypatch.setattr(
        erlab.interactive.utils,
        "copy_to_clipboard",
        lambda text: copied.append(text) or text,
    )
    manager.tree_view.clearSelection()
    select_child_tool(manager, uid)
    manager._update_info(uid=uid)
    menu = manager._build_metadata_derivation_menu()
    assert menu is not None
    action_map(menu)["Copy Full Code"].trigger()
    assert copied
    return copied[-1]


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

        manager.tree_view.edit(manager.tree_view._model._row_index(goldtool_uid))
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

        delegate = typing.cast(
            "_ImageToolWrapperItemDelegate", manager.tree_view.itemDelegate()
        )
        model = typing.cast("_ImageToolWrapperItemModel", manager.tree_view.model())
        manager.tree_view.expand(model._row_index(0))
        index = model._row_index(uid)
        manager.tree_view.scrollTo(index)
        option = QtWidgets.QStyleOptionViewItem()
        option.rect = manager.tree_view.visualRect(index)
        option.font = manager.tree_view.font()
        badge_rect, badge_text, _ = delegate._compute_child_status_info(option, child)
        assert badge_rect is not None
        assert badge_text == "Stale"
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
        assert (
            tooltip_text == "Click to update this tool from the latest ImageTool data."
        )
        click_pos = badge_rect.center()
        assert model._row_index(uid) == manager.tree_view.indexAt(click_pos)
        global_click_pos = manager.tree_view.viewport().mapToGlobal(click_pos)

        def _enable_auto_update(dialog: QtWidgets.QDialog) -> None:
            dialog.auto_update_check.setChecked(True)  # type: ignore[attr-defined]

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
            pre_call=_enable_auto_update,
        )

        assert child.source_state == "fresh"
        assert child.source_auto_update is True
        xr.testing.assert_identical(child.tool_data, replaced.transpose("eV", "alpha"))

        replaced2 = replaced.copy(deep=True)
        replaced2.data = np.asarray(replaced2.data) + 5

        with qtbot.wait_signal(manager._sigDataReplaced):
            itool(replaced2, manager=True, replace=0)

        qtbot.wait_until(lambda: child.source_state == "fresh", timeout=5000)
        xr.testing.assert_identical(child.tool_data, replaced2.transpose("eV", "alpha"))


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
        assert "era.image.diffn(" in copied
        assert copied.endswith(".transpose()")


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
        assert output_node.provenance_spec is None
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
        assert second_output_node.provenance_spec is None
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
            and wrapper.provenance_spec is None
            and wrapper.output_id is None
            and wrapper._childtool_indices == []
            and fetch(index).identical(expected)
        ]
        assert len(matching_roots) == 1


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
        assert copied.startswith("result = era.image.diffn(")
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
        assert metadata_derivation_texts(manager) == []
        assert manager._build_metadata_derivation_menu() is None

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
        assert "load_dataarray(" in copied[-1]
        assert str(file_path) in copied[-1]
        assert "_parse_input" not in copied[-1]
        assert "data = " in copied[-1]

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
        action_map(menu)["Copy Full Code"].trigger()
        assert copied
        assert copied[-1].startswith("result = era.image.diffn(")


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
        assert "live source linkage" in captured["info"].lower()
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
        assert copied[-1] == (
            'derived = derived.qsel.average("x")\nderived = derived.qsel.average("y")'
        )

        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        action_map(menu)["Copy Selected Code"].trigger()
        assert copied[-1] == (
            'derived = derived.qsel.average("x")\nderived = derived.qsel.average("y")'
        )

        action_map(menu)["Copy Full Code"].trigger()
        assert copied[-1] == "derived = data.qsel.average('x').qsel.average('y')"
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
        assert provenance.display_code() == "derived = my_data"
        assert provenance.display_entries()[0].label == (
            "Start from watched variable 'my_data'"
        )

        copied: list[str] = []
        monkeypatch.setattr(
            erlab.interactive.utils,
            "copy_to_clipboard",
            lambda text: copied.append(text) or text,
        )
        manager.tree_view.clearSelection()
        select_tools(manager, [0])
        manager._update_info(uid=node.uid)
        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        action_map(menu)["Copy Full Code"].trigger()
        assert copied[-1] == "derived = my_data"


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
        assert "my_data" in copied
        assert "derived = data" not in copied


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
        assert copied.startswith("derived = my_data")
        assert "derived = data" not in copied


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
        assert "derived = my_data" not in copied[-1]
        assert "derived = data" not in copied[-1]
        assert "derived = derived.squeeze()" not in copied[-1]
        assert "result = my_data.xlm.modelfit(" in copied[-1]


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
        assert "derived = my_1d" not in copied[-1]
        assert "derived = data" not in copied[-1]
        assert ".squeeze()" not in copied[-1]
        assert "result = my_1d.xlm.modelfit(" in copied[-1]


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
        assert "derived = my_data" not in copied[-1]
        assert "derived = data" not in copied[-1]
        assert ".squeeze()" not in copied[-1]
        assert "result = my_data.xlm.modelfit(" in copied[-1]


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
                chained_dialogs=2,
            )

            # Load workspace
            accept_dialog(
                lambda: manager.load(native=False),
                pre_call=_go_to_file,
                chained_dialogs=2,
            )

            # Check if the data is loaded
            assert manager.ntools == 4

            # Check if the child dtool is also loaded
            assert len(manager._imagetool_wrappers[2]._childtools) == 1

            select_tools(manager, list(manager._imagetool_wrappers.keys()))
            accept_dialog(manager.remove_action.trigger)
            qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)


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

        tree = manager._to_datatree()
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        for node in tree.values():
            manager._load_workspace_node(typing.cast("xr.DataTree", node))

        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        loaded_child = manager.get_childtool(child_uid)
        assert isinstance(loaded_child, Fit1DTool)
        assert loaded_child._last_result_ds is not None
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

        tree = manager._to_datatree()
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        for node in tree.values():
            manager._load_workspace_node(typing.cast("xr.DataTree", node))

        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        loaded_child = manager.get_childtool(child_uid)
        assert isinstance(loaded_child, Fit2DTool)
        assert all(ds is not None for ds in loaded_child._result_ds_full)
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

            loaded = xr.open_datatree(
                filename, engine="h5netcdf", chunks="auto", phony_dims="sort"
            )
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
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()

        def _go_to_file(dialog: QtWidgets.QFileDialog):
            dialog.setDirectory(str(datadir))
            dialog.selectFile(str(datadir / "manager_workspace_legacy.h5"))
            focused = dialog.focusWidget()
            if isinstance(focused, QtWidgets.QLineEdit):
                focused.setText("manager_workspace_legacy.h5")

        # Load workspace
        accept_dialog(
            lambda: manager.load(native=False), pre_call=_go_to_file, chained_dialogs=2
        )

        # Check if the data is loaded
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        # Check data
        xr.testing.assert_identical(
            manager.get_imagetool(0).slicer_area._data,
            test_data,
        )

        select_tools(manager, list(manager._imagetool_wrappers.keys()))
        accept_dialog(manager.remove_action.trigger)
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)


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
    qtbot.addWidget(manager, before_close_func=lambda w: w.remove_all_tools())

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

    accept_dialog(manager.close)
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
        _remove_childtool=lambda child_uid: removed_uids.append(child_uid),
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

        itool([test_data, test_data], link=True, manager=True)

        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        manager.get_imagetool(0).slicer_area._auto_chunk()
        manager.get_imagetool(1).slicer_area._auto_chunk()
        select_tools(manager, [0])
        manager._update_info()
        assert "Chunks" in metadata_detail_map(manager)

        view = manager.tree_view

        model = view._model
        delegate = view._delegate

        index = model.index(0, 0)  # first tool
        option = QtWidgets.QStyleOptionViewItem()
        delegate.initStyleOption(option, index)
        icons_width, dask_rect, link_rect, _ = delegate._compute_icons_info(
            option, index.internalPointer()
        )
        uid_rect = delegate._compute_uid_badge_rect(
            option,
            index.data(_NODE_UID_ROLE),
            right_reserved=icons_width,
        )
        assert uid_rect is not None

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
        assert text == "Dask-backed data (chunked array)"

        # Hover over link icon
        text = None
        pos = link_rect.center()
        event = QtGui.QHelpEvent(
            QtCore.QEvent.Type.ToolTip, pos, view.viewport().mapToGlobal(pos)
        )
        handled = delegate.helpEvent(event, view, option, index)

        assert handled
        assert text == "Linked (#0)"

        # Hover outside icons
        text = None
        pos = dask_rect.topRight() + QtCore.QPoint(2, 0)
        event = QtGui.QHelpEvent(
            QtCore.QEvent.Type.ToolTip, pos, view.viewport().mapToGlobal(pos)
        )
        handled = delegate.helpEvent(event, view, option, index)
        assert not handled
        assert text is None

        # Hover over node id badge
        text = None
        pos = uid_rect.center()
        event = QtGui.QHelpEvent(
            QtCore.QEvent.Type.ToolTip, pos, view.viewport().mapToGlobal(pos)
        )
        handled = delegate.helpEvent(event, view, option, index)
        assert handled
        assert text == f"Node ID: {index.data(_NODE_UID_ROLE)}"


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


def test_manager_duplicate_goldtool_child_with_data_corr_shows_error(
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

        corrected = gold.copy(deep=True)
        corrected.data = np.asarray(corrected.data) * 1.01
        child_uid = manager.add_childtool(
            GoldTool(gold.copy(deep=True), data_corr=corrected, data_name="gold_input"),
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


def test_manager_save_goldtool_child_with_data_corr_shows_error(
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

        corrected = gold.copy(deep=True)
        corrected.data = np.asarray(corrected.data) * 1.01
        manager.add_childtool(
            GoldTool(gold.copy(deep=True), data_corr=corrected, data_name="gold_input"),
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
    monkeypatch.setattr(xr, "open_datatree", _raise_open_datatree)

    with manager_context() as manager:
        ImageToolManager.load(manager, native=False)

        QtWidgets.QApplication.processEvents()

        assert exec_calls["count"] >= 2  # One retry after the failure path.
        assert native_opt_calls["count"] >= 2  # Retry should preserve `native=False`.
        assert len(critical_calls) == 1
        assert manager._alert_dialogs == []


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

    def _fake_open_datatree(*args, **kwargs):
        return object()

    def _raise_from_datatree(*args, **kwargs):
        raise RuntimeError("cannot restore workspace")

    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        staticmethod(_fake_critical),
    )
    monkeypatch.setattr(xr, "open_datatree", _fake_open_datatree)

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
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        menu_actions = {
            action.text().replace("&", ""): action
            for action in manager.menu_bar.actions()
            if action.menu() is not None
        }

        assert "File" in menu_actions
        assert "Apps" in menu_actions
        assert "Data Explorer" in action_map(
            typing.cast("QtWidgets.QMenu", menu_actions["File"].menu())
        )

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


def test_manager_explorer_launcher_reuses_instance_and_opens_directory_tabs(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with (
        manager_context() as manager,
        tempfile.TemporaryDirectory() as recent_dir,
        tempfile.TemporaryDirectory() as dropped_dir,
    ):
        manager._recent_directory = recent_dir

        manager.ensure_explorer_initialized()
        explorer = manager.explorer

        assert isinstance(explorer, _TabbedExplorer)
        assert hasattr(manager, "explorer")
        assert explorer.tab_widget.count() == 1

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
    with manager_context() as manager:
        initial_ntools = manager.ntools
        initial_rows = manager.tree_view.model().rowCount(QtCore.QModelIndex())

        manager.show_ptable()
        ptable = manager.ptable_window

        qtbot.wait_until(ptable.isVisible)
        assert isinstance(ptable, PeriodicTableWindow)
        assert manager.ntools == initial_ntools
        assert manager.tree_view.model().rowCount(QtCore.QModelIndex()) == initial_rows

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
