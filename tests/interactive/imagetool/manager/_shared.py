# ruff: noqa: F401,TC001
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
import warnings
import weakref
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
import erlab.interactive.imagetool._serialization as imagetool_serialization
import erlab.interactive.imagetool.manager as manager_module
import erlab.interactive.imagetool.manager._actions as manager_actions
import erlab.interactive.imagetool.manager._console as manager_console
import erlab.interactive.imagetool.manager._desktop as manager_desktop
import erlab.interactive.imagetool.manager._dialogs as manager_dialogs
import erlab.interactive.imagetool.manager._io as manager_io
import erlab.interactive.imagetool.manager._mainwindow as manager_mainwindow
import erlab.interactive.imagetool.manager._registry as manager_registry
import erlab.interactive.imagetool.manager._server as manager_server
import erlab.interactive.imagetool.manager._widgets as manager_widgets
import erlab.interactive.imagetool.manager._workspace as manager_workspace
import erlab.interactive.imagetool.manager._workspace_io as manager_workspace_io
import erlab.interactive.imagetool.manager._xarray as manager_xarray
import erlab.interactive.imagetool.viewer as imagetool_viewer
import erlab.interactive.imagetool.viewer_state as imagetool_viewer_state
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
from erlab.interactive.imagetool.controls import ItoolColormapControls
from erlab.interactive.imagetool.dialogs import SelectionDialog
from erlab.interactive.imagetool.manager import (
    ImageToolManager,
    fetch,
    load_in_manager,
    replace_data,
)
from erlab.interactive.imagetool.manager._console import ToolNamespace
from erlab.interactive.imagetool.manager._dialogs import (
    _ChooseFromDataTreeDialog,
    _ConcatDialog,
    _CoordinateAttrsPickerDialog,
    _NameFilterDialog,
    _NameMapEditorDialog,
    _RenameDialog,
    _text_to_loader_extension_value,
)
from erlab.interactive.imagetool.manager._mainwindow import (
    _LoadSourceDetailsDialog,
    _WorkspacePropertiesDialog,
    _WorkspacePropertiesState,
)
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


CONSOLE_HELPER_OFFSET = 2.5


def console_helper_dependency(value):
    return value + CONSOLE_HELPER_OFFSET


def console_helper(value):
    return console_helper_dependency(value)


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


def dependency_status_badge(
    manager: ImageToolManager, target: int | str
) -> tuple[QtCore.QRect, QtCore.QModelIndex]:
    model = typing.cast("_ImageToolWrapperItemModel", manager.tree_view.model())
    delegate = typing.cast(
        "_ImageToolWrapperItemDelegate", manager.tree_view.itemDelegate()
    )

    index = model._row_index(target)
    manager.tree_view.scrollTo(index)
    option = delegate._option_for_index(manager.tree_view, index)
    badge_rect, badge_text, _ = delegate._compute_dependency_status_info(
        option, manager._node_for_target(target)
    )
    assert badge_rect is not None
    assert badge_text is not None
    return badge_rect, index


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


def metadata_detail_labels(manager: ImageToolManager) -> list[str]:
    labels: list[str] = []
    for row in range(manager.metadata_details_layout.rowCount()):
        item = manager.metadata_details_layout.itemAtPosition(row, 0)
        widget = None if item is None else item.widget()
        if isinstance(widget, QtWidgets.QLabel):
            labels.append(widget.text())
    return labels


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


def action_map_by_object_name(menu: QtWidgets.QMenu) -> dict[str, QtWidgets.QAction]:
    return {
        action.objectName(): action
        for action in menu.actions()
        if not action.isSeparator() and action.objectName()
    }


def trigger_menu_action(menu: QtWidgets.QMenu, action: QtGui.QAction) -> None:
    assert action in menu.actions()
    action.trigger()


def menu_map_by_object_name(
    menu_bar: QtWidgets.QMenuBar,
) -> dict[str, QtWidgets.QMenu]:
    menus: dict[str, QtWidgets.QMenu] = {}
    for action in menu_bar.actions():
        menu = action.menu()
        if menu is not None and menu.objectName():
            menus[menu.objectName()] = menu
    return menus


@pytest.fixture(autouse=True)
def isolated_recent_workspace_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> pathlib.Path:
    settings_path = tmp_path / "recent-workspaces.ini"

    def _settings() -> QtCore.QSettings:
        return QtCore.QSettings(str(settings_path), QtCore.QSettings.Format.IniFormat)

    _settings().clear()
    monkeypatch.setattr(manager_mainwindow, "_manager_settings", _settings)
    monkeypatch.setattr(manager_workspace_io, "_manager_settings", _settings)
    return settings_path


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
    trigger_menu_action(menu, manager._metadata_copy_full_action)
    assert copied
    return copied[-1]


__all__ = [name for name in globals() if not name.startswith("__")]
