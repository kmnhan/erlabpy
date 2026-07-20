import contextlib
import json
import logging
import pathlib
import tempfile
import types
import typing
from collections.abc import Callable

import numpy as np
import pydantic
import pytest
import xarray
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

import erlab
import erlab.interactive._options.core
import erlab.interactive.imagetool.manager._base as manager_base
import erlab.interactive.imagetool.manager._io as manager_io
import erlab.interactive.imagetool.viewer_state as imagetool_viewer_state
from erlab.interactive.imagetool import itool
from erlab.interactive.imagetool._provenance._model import (
    FileDataSelection,
    ScriptInput,
    ToolProvenanceSpec,
    full_data,
    script,
)
from erlab.interactive.imagetool.manager import ImageToolManager, load_in_manager
from erlab.interactive.imagetool.manager._dependency import _ManagerDependencyTracker
from erlab.interactive.imagetool.manager._dialogs import _NameFilterDialog
from erlab.interactive.imagetool.manager._modelview import (
    _FIGURE_SOURCE_MIME,
    _MIME,
    _ImageToolWrapperItemDelegate,
    _ImageToolWrapperItemModel,
    _RowBadge,
)
from erlab.interactive.imagetool.manager._tool_graph import _ManagerToolGraph
from erlab.interactive.imagetool.manager._wrapper import _ImageToolWrapper

from .helpers import (
    assert_nonempty_tooltip,
    click_tree_view_pos,
    metadata_detail_map,
    select_child_tool,
    select_tools,
)

logger = logging.getLogger(__name__)


def test_dependency_tracker_uses_passive_tool_provenance() -> None:
    source = script(
        start_label="Source",
        seed_code="source = data",
        active_name="source",
    )
    dependent = script(
        start_label="Dependent",
        seed_code="derived = source",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="source",
                label="Source",
                node_uid="source-uid",
                provenance_spec=source,
            ),
        ),
    )

    class _PassiveTool:
        def current_provenance_spec(
            self, *, flush_deferred_restore: bool = True
        ) -> ToolProvenanceSpec:
            assert flush_deferred_restore is False
            return dependent

    class _PassiveNode:
        @property
        def tool_window(self):
            return _PassiveTool()

        @property
        def provenance_spec(self):
            pytest.fail("dependency tracking must not request flushing provenance")

    graph = types.SimpleNamespace(nodes={"dependent-uid": _PassiveNode()})
    tracker = _ManagerDependencyTracker(typing.cast("_ManagerToolGraph", graph))

    refs = tracker.refs_for_uid("dependent-uid")

    assert len(refs) == 1
    assert refs[0].node_uid == "source-uid"


class _InfoRefreshToolState(pydantic.BaseModel):
    value: int = 0


class _InfoRefreshTool(erlab.interactive.utils.ToolWindow[_InfoRefreshToolState]):
    StateModel = _InfoRefreshToolState
    tool_name = "info-refresh"

    def __init__(self, data: xr.DataArray) -> None:
        super().__init__()
        self._data = data
        self._status = _InfoRefreshToolState()
        self._info_text = "initial child info"

    @property
    def tool_data(self) -> xr.DataArray:
        return self._data

    @property
    def tool_status(self) -> _InfoRefreshToolState:
        return self._status

    @tool_status.setter
    def tool_status(self, status: _InfoRefreshToolState) -> None:
        self._status = status

    @property
    def info_text(self) -> str:
        return self._info_text

    def emit_info_text(self, text: str) -> None:
        self._info_text = text
        self.sigInfoChanged.emit()


def test_childtool_hover_preview_hides_missing_imageitem_pixmap(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeIndex:
        def internalPointer(self) -> str:
            return "child-1"

        def data(self, *, role: QtCore.Qt.ItemDataRole) -> str:
            assert role == QtCore.Qt.ItemDataRole.DisplayRole
            return "child"

    class _FakePreviewAction:
        def isChecked(self) -> bool:
            return True

    class _FakeImageItem:
        def getViewBox(self) -> object:
            return types.SimpleNamespace(rect=lambda: QtCore.QRectF(0.0, 0.0, 2.0, 4.0))

        def getPixmap(self) -> None:
            return None

    child_node = types.SimpleNamespace(
        uid="child-1",
        display_text="child",
        type_badge_text="",
        source_state="fresh",
        source_auto_update=False,
        imagetool=None,
        pending_workspace_tool_payload=None,
        tool_window=types.SimpleNamespace(
            preview_pixmap=None,
            preview_imageitem=_FakeImageItem(),
        ),
    )

    class _FakeManager:
        preview_action = _FakePreviewAction()

        def _child_node(self, uid: str) -> object:
            assert uid == "child-1"
            return child_node

        def dependency_status_for_uid(self, uid: str) -> None:
            assert uid == "child-1"
            return

    monkeypatch.setattr(
        erlab.interactive.utils,
        "qt_is_valid",
        lambda *objects: all(obj is not None for obj in objects),
    )
    view = QtWidgets.QTreeView()
    qtbot.addWidget(view)
    manager = _FakeManager()
    delegate = _ImageToolWrapperItemDelegate(
        typing.cast("ImageToolManager", manager),
        typing.cast("typing.Any", view),
    )
    option = QtWidgets.QStyleOptionViewItem()
    option.rect = QtCore.QRect(0, 0, 160, 28)
    option.widget = view
    option.state = QtWidgets.QStyle.StateFlag.State_MouseOver
    canvas = QtGui.QPixmap(160, 28)
    canvas.fill(QtGui.QColor("white"))
    painter = QtGui.QPainter(canvas)
    try:
        delegate._paint_childtool(
            painter,
            option,
            typing.cast("QtCore.QModelIndex", _FakeIndex()),
        )
    finally:
        painter.end()

    assert not delegate.preview_popup.isVisible()


def test_link_badge_falls_back_to_live_linker(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        for offset in (0.0, 10.0):
            tool = itool(
                xr.DataArray(
                    np.arange(offset, offset + 16).reshape(4, 4),
                    dims=("x", "y"),
                ),
                manager=False,
                execute=False,
            )
            assert isinstance(tool, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(tool, show=False)
        manager.link_imagetools(0, 1, link_colors=False)

        wrapper = manager._tool_graph.root_wrappers[0]
        proxy = wrapper.slicer_area._linking_proxy
        assert proxy is not None
        link_key = wrapper.workspace_link_key
        assert link_key is not None
        linker_calls: list[object] = []
        original_color_for_linker = manager.color_for_linker

        def _record_linker(candidate):
            linker_calls.append(candidate)
            return original_color_for_linker(candidate)

        monkeypatch.setattr(manager, "color_for_linker", _record_linker)
        monkeypatch.setattr(
            _ImageToolWrapper,
            "workspace_linked",
            property(lambda _wrapper: True),
        )
        index = manager.tree_view._model.index(0, 0)

        def _paint() -> None:
            option = manager.tree_view._delegate._option_for_index(
                manager.tree_view, index
            )
            canvas = QtGui.QPixmap(200, 32)
            canvas.fill(QtGui.QColor("white"))
            painter = QtGui.QPainter(canvas)
            try:
                manager.tree_view._delegate.paint(painter, option, index)
            finally:
                painter.end()

        wrapper._workspace_link_key = None
        try:
            _paint()
            assert linker_calls == [proxy]

            wrapper.slicer_area._linking_proxy = None
            try:
                _paint()
            finally:
                wrapper.slicer_area._linking_proxy = proxy
        finally:
            wrapper._workspace_link_key = link_key

        assert linker_calls == [proxy]


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
            lambda: len(manager._tool_graph.root_wrappers[0]._childtools) == 1,
            timeout=5000,
        )
        logger.info("First childtool added")
        manager.get_imagetool(0).slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[0]._childtools) == 2,
            timeout=5000,
        )
        logger.info("Second childtool added")
        manager.get_imagetool(0).slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[0]._childtools) == 3,
            timeout=5000,
        )
        logger.info("Third childtool added")
        manager.hide_all()  # Prevent windows from obstructing the manager

        # Check mimedata
        model = typing.cast("_ImageToolWrapperItemModel", manager.tree_view.model())
        assert model.mimeTypes() == [_MIME, _FIGURE_SOURCE_MIME, "text/plain"]
        assert (
            model.supportedDragActions()
            == QtCore.Qt.DropAction.MoveAction | QtCore.Qt.DropAction.CopyAction
        )

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
        assert mime_single.text() == "tools[0]"
        assert manager.tree_view.figure_source_uids_from_mime(mime_single) == (
            model.index(0, 0).internalPointer().uid,
        )
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
        assert manager._tool_graph.displayed_indices == [1, 2, 0]
        assert model.mimeData([model.index(0, 0)]).text() == "tools[1]"

        # No-op drop (drop on itself)
        assert not model.dropMimeData(
            model.mimeData([model.index(0, 0)]),
            QtCore.Qt.DropAction.MoveAction,
            0,
            0,
            model.index(0, 0),
        )

        # Check unchanged
        assert manager._tool_graph.displayed_indices == [1, 2, 0]

        # Test move child tool
        parent_wrapper = model.manager._tool_graph.root_wrappers[0]
        child_uid: str = parent_wrapper._childtool_indices[0]
        old_order = list(parent_wrapper._childtool_indices)
        parent_index: QtCore.QModelIndex = model._row_index(0)
        child_index: QtCore.QModelIndex = model._row_index(child_uid)
        assert model.mimeData([child_index]).text() == "tools[0].children[0]"
        assert (
            manager.tree_view.figure_source_uids_from_mime(
                model.mimeData([child_index])
            )
            == ()
        )

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
        assert (
            model.mimeData([model._row_index(child_uid)]).text()
            == "tools[0].children[1]"
        )

        # Test multiple selection
        logger.info("Testing multiple selection drop")
        mime_multiple = model.mimeData([model.index(0, 0), model.index(1, 0)])
        assert not mime_multiple.hasText()
        assert model.canDropMimeData(
            mime_multiple, QtCore.Qt.DropAction.MoveAction, 0, 0, QtCore.QModelIndex()
        )
        model.dropMimeData(
            mime_multiple, QtCore.Qt.DropAction.MoveAction, 0, 0, QtCore.QModelIndex()
        )

        # Test mixed top-level and childtool selection (should be rejected)
        logger.info("Testing mixed top-level and childtool selection drop")
        parent_wrapper = model.manager._tool_graph.root_wrappers[0]
        child_uid: str = parent_wrapper._childtool_indices[0]
        mime_mixed = model.mimeData([model._row_index(0), model._row_index(child_uid)])
        assert not mime_mixed.hasText()
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

        logger.info("Testing invalid Figure Composer source mimedata decoding")
        assert model.decode_figure_source_mime(None) == ()
        invalid_source_mime = QtCore.QMimeData()
        invalid_source_mime.setData(_FIGURE_SOURCE_MIME, QtCore.QByteArray(b"not json"))
        assert model.decode_figure_source_mime(invalid_source_mime) == ()
        invalid_source_mime.setData(
            _FIGURE_SOURCE_MIME,
            QtCore.QByteArray(json.dumps("not a dict").encode("utf-8")),
        )
        assert model.decode_figure_source_mime(invalid_source_mime) == ()
        invalid_source_mime.setData(
            _FIGURE_SOURCE_MIME,
            QtCore.QByteArray(json.dumps({"uids": "not a list"}).encode("utf-8")),
        )
        assert model.decode_figure_source_mime(invalid_source_mime) == ()
        missing_child_index = model.createIndex(0, 0, "missing-child")
        missing_child_mime = model.mimeData([missing_child_index])
        assert _MIME not in missing_child_mime.formats()
        assert not missing_child_mime.hasText()


def test_figure_source_mime_filters_duplicates_and_malformed_rows(
    qtbot, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FakeManager(QtWidgets.QWidget):
        pass

    manager = _FakeManager()
    qtbot.addWidget(manager)
    wrapper = _ImageToolWrapper(
        typing.cast("ImageToolManager", manager),
        index=0,
        uid="root-source",
        tool=None,
    )
    manager._tool_graph = _ManagerToolGraph()
    manager._tool_graph.displayed_indices = [0]
    manager._tool_graph.root_wrappers[0] = wrapper
    manager._tool_graph.nodes["root-source"] = wrapper
    model = _ImageToolWrapperItemModel(
        typing.cast("ImageToolManager", manager), manager
    )
    root_index = model.index(0, 0)
    root_uid = root_index.internalPointer().uid

    duplicate_source_mime = model.mimeData([root_index, root_index])
    assert model.decode_figure_source_mime(duplicate_source_mime) == (root_uid,)
    assert duplicate_source_mime.text() == "tools[0]"

    mixed_source_mime = QtCore.QMimeData()
    mixed_source_mime.setData(
        _FIGURE_SOURCE_MIME,
        QtCore.QByteArray(
            json.dumps({"uids": ["first", 1, "first", None, "second"]}).encode("utf-8")
        ),
    )
    assert model.decode_figure_source_mime(mixed_source_mime) == (
        "first",
        "second",
    )

    missing_child_index = model.createIndex(0, 0, "missing-child")
    malformed_pointer = object()
    malformed_parent = model.createIndex(0, 0, malformed_pointer)
    with monkeypatch.context() as context:
        context.setattr(
            type(model),
            "parent",
            lambda _self, _index: malformed_parent,
        )
        malformed_child_mime = model.mimeData([missing_child_index])
    assert malformed_child_mime.formats() == []


def test_treeview(qtbot, accept_dialog, test_data) -> None:
    manager = ImageToolManager()

    def _cleanup_manager(widget: ImageToolManager) -> None:
        widget._workspace_state.loading_depth += 1
        try:
            widget.remove_all_tools()
            widget._workspace_controller._mark_workspace_clean()
        finally:
            widget._workspace_state.loading_depth -= 1

    qtbot.addWidget(manager, before_close_func=_cleanup_manager)

    with qtbot.waitExposed(manager):
        manager.show()
        manager.activateWindow()

    qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

    test_data.qshow(manager=manager.manager_index)
    test_data.qshow(manager=manager.manager_index)
    qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

    manager.raise_()
    manager.activateWindow()

    model = manager.tree_view._model
    assert model.supportedDropActions() == QtCore.Qt.DropAction.MoveAction
    first_row_rect = manager.tree_view.visualRect(model.index(0, 0))

    # Click on first row
    qtbot.mousePress(
        manager.tree_view.viewport(),
        QtCore.Qt.MouseButton.LeftButton,
        pos=first_row_rect.center(),
    )
    assert manager.tree_view.selected_imagetool_indices == [0]

    # Show context menu
    manager.tree_view._show_menu(first_row_rect.center())
    menu = manager.tree_view._menu
    qtbot.wait_until(menu.isVisible)
    menu.close()
    QtWidgets.QApplication.processEvents()

    def _discard_changes(dialog: QtWidgets.QMessageBox) -> None:
        dialog.button(QtWidgets.QMessageBox.StandardButton.Discard).click()

    accept_dialog(manager.close, accept_call=_discard_changes)
    qtbot.wait_until(
        lambda: (
            not erlab.interactive.imagetool.manager.is_running(manager.manager_index)
        )
    )


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
            lambda: uid in manager._tool_graph.root_wrappers[0]._childtool_indices,
            timeout=5000,
        )

        manager.tree_view.clear_imagetools()
        assert manager._tool_graph.displayed_indices == []

        # Child destruction callbacks can arrive after top-level rows are reset.
        manager._remove_childtool(uid)
        qtbot.wait_until(
            lambda: uid not in manager._tool_graph.root_wrappers[0]._childtools,
            timeout=5000,
        )


def test_childtool_info_changed_debounces_manager_details_refresh(
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

        tool = _InfoRefreshTool(test_data)
        uid = manager.add_childtool(tool, 0, show=False)
        qtbot.wait_until(
            lambda: uid in manager._tool_graph.root_wrappers[0]._childtool_indices,
            timeout=5000,
        )
        selection_model = manager.tree_view.selectionModel()
        selection_model.clearSelection()
        select_child_tool(manager, uid)
        qtbot.wait_until(
            lambda: manager.tree_view.selected_childtool_uids == [uid],
            timeout=5000,
        )

        metadata_updates: list[str] = []
        original_set_metadata_node = manager._set_metadata_node

        def _record_metadata_rebuild(node) -> None:
            metadata_updates.append(node.uid)
            original_set_metadata_node(node)

        monkeypatch.setattr(manager, "_set_metadata_node", _record_metadata_rebuild)
        child_node = manager._child_node(uid)
        tool._info_text = "updated child info"
        child_node._handle_tool_info_changed()
        tool._info_text = "updated child info again"
        child_node._handle_tool_info_changed()
        tool._info_text = "updated child info final"
        child_node._handle_tool_info_changed()

        assert "updated child info" not in manager.text_box.toPlainText()
        assert metadata_updates == []
        assert (
            manager._interaction_gate.pending_keys.count(("tool-info-refresh", uid))
            == 1
        )
        manager._flush_idle_work(force=True)
        assert metadata_updates == [uid]
        assert "updated child info final" in manager.text_box.toPlainText()


def test_childtool_state_changed_marks_dirty_without_details_refresh(
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

        tool = _InfoRefreshTool(test_data)
        uid = manager.add_childtool(tool, 0, show=False)
        qtbot.wait_until(
            lambda: uid in manager._tool_graph.root_wrappers[0]._childtool_indices,
            timeout=5000,
        )
        select_child_tool(manager, uid)
        manager._update_info()

        metadata_updates: list[str] = []
        original_set_metadata_node = manager._set_metadata_node

        def _record_metadata_rebuild(node) -> None:
            metadata_updates.append(node.uid)
            original_set_metadata_node(node)

        monkeypatch.setattr(manager, "_set_metadata_node", _record_metadata_rebuild)
        manager._workspace_controller._mark_workspace_clean()

        tool.sigStateChanged.emit()

        assert uid in manager._workspace_state.dirty_state
        assert metadata_updates == []
        assert ("tool-info-refresh", uid) not in manager._interaction_gate.pending_keys

        child_node = manager._child_node(uid)
        manager._workspace_controller._mark_workspace_clean()
        original_node = manager._tool_graph.nodes[uid]
        try:
            manager._tool_graph.nodes[uid] = object()
            child_node._handle_tool_state_changed()
        finally:
            manager._tool_graph.nodes[uid] = original_node
        assert uid not in manager._workspace_state.dirty_state

        with monkeypatch.context() as patch:
            patch.setattr(erlab.interactive.utils, "qt_is_valid", lambda _obj: False)
            child_node._handle_tool_state_changed()
        assert uid not in manager._workspace_state.dirty_state


def test_manager_idle_queue_deduplicates_and_waits_for_idle(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager._interaction_gate.set_quiet_interval(1)
        calls: list[str] = []

        manager._note_interaction_activity()
        manager._queue_idle_work(("test", "work"), lambda: calls.append("old"))
        manager._queue_idle_work(("test", "work"), lambda: calls.append("new"))

        assert calls == []
        assert manager._interaction_gate.pending_keys == (("test", "work"),)
        qtbot.wait_until(lambda: calls == ["new"], timeout=1000)
        assert manager._interaction_gate.pending_keys == ()


def test_manager_idle_queue_stops_when_activity_resumes(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        calls: list[str] = []

        def first_callback() -> None:
            calls.append("first")
            manager._note_interaction_activity()

        manager._queue_idle_work(("test", "first"), first_callback)
        manager._queue_idle_work(("test", "second"), lambda: calls.append("second"))

        qtbot.wait_until(lambda: calls == ["first"], timeout=1000)
        assert manager._interaction_gate.pending_keys == (("test", "second"),)

        manager._flush_idle_work(force=True)

        assert calls == ["first", "second"]
        assert manager._interaction_gate.pending_keys == ()


def test_manager_idle_queue_perf_timing_logs(
    monkeypatch,
    caplog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    monkeypatch.setenv("ERLAB_MANAGER_PERF_TIMING", "1")
    caplog.set_level(
        logging.DEBUG,
        logger="erlab.interactive.imagetool.manager._interaction",
    )
    with manager_context() as manager:
        calls: list[str] = []
        manager._queue_idle_work(("test", "work"), lambda: calls.append("work"))
        manager._flush_idle_work(force=True)

        assert calls == ["work"]
        assert any(
            "Manager forced work flush ran 1 callbacks" in record.message
            for record in caplog.records
        )


def test_childtool_data_changed_deduplicates_descendant_refresh(
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

        tool = _InfoRefreshTool(test_data)
        uid = manager.add_childtool(tool, 0, show=False)
        qtbot.wait_until(
            lambda: uid in manager._tool_graph.root_wrappers[0]._childtool_indices,
            timeout=5000,
        )
        manager._workspace_controller._mark_workspace_clean()

        propagated_uids: list[str] = []
        refreshed_uids: list[str | None] = []
        monkeypatch.setattr(
            manager,
            "_propagate_source_change_from_uid",
            lambda changed_uid: propagated_uids.append(changed_uid),
        )
        monkeypatch.setattr(
            manager.tree_view,
            "refresh",
            lambda target_uid=None: refreshed_uids.append(target_uid),
        )

        child_node = manager._child_node(uid)
        child_node._handle_tool_data_changed()
        child_node._handle_tool_data_changed()
        child_node._handle_tool_data_changed()

        assert manager._workspace_state.dirty_data == {uid}
        assert propagated_uids == []
        assert refreshed_uids == []
        assert (
            manager._interaction_gate.pending_keys.count(
                ("snapshot-token-refresh", uid)
            )
            == 1
        )
        assert (
            manager._interaction_gate.pending_keys.count(("tool-data-refresh", uid))
            == 1
        )

        manager._flush_idle_work(force=True)

        assert propagated_uids == [uid]
        assert refreshed_uids == [uid, uid]


def test_manager_interaction_gate_tracks_key_and_editor_focus_events(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager._interaction_gate.set_quiet_interval(1)
        key_event = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_A,
            QtCore.Qt.KeyboardModifier.NoModifier,
            "a",
        )
        QtWidgets.QApplication.sendEvent(manager, key_event)

        assert manager._interaction_active
        qtbot.wait_until(lambda: not manager._interaction_active, timeout=1000)

        edit = QtWidgets.QLineEdit(manager)
        qtbot.addWidget(edit)
        focus_event = QtGui.QFocusEvent(QtCore.QEvent.Type.FocusIn)
        QtWidgets.QApplication.sendEvent(edit, focus_event)

        assert manager._interaction_active


def test_childtool_info_changed_for_unselected_node_keeps_visible_details(
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

        tool = _InfoRefreshTool(test_data)
        uid = manager.add_childtool(tool, 0, show=False)
        qtbot.wait_until(
            lambda: uid in manager._tool_graph.root_wrappers[0]._childtool_indices,
            timeout=5000,
        )
        selection_model = manager.tree_view.selectionModel()
        selection_model.clearSelection()
        select_tools(manager, [0])
        manager._update_info()
        visible_html = manager.text_box.toHtml()
        manager._tool_metadata_queue.set_interval(1)

        tool.emit_info_text("updated child info")

        assert manager._tool_metadata_queue.pending_uids == frozenset()
        assert not manager._tool_metadata_queue.is_active()
        assert manager.text_box.toHtml() == visible_html
        assert "updated child info" not in manager.text_box.toPlainText()


def test_childtool_repeated_info_changes_mark_state_dirty_once(
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

        tool = _InfoRefreshTool(test_data)
        uid = manager.add_childtool(tool, 0, show=False)
        qtbot.wait_until(
            lambda: uid in manager._tool_graph.root_wrappers[0]._childtool_indices,
            timeout=5000,
        )
        manager._workspace_controller._mark_workspace_clean()

        tool.emit_info_text("first")
        tool.emit_info_text("second")
        tool.emit_info_text("third")

        assert manager._workspace_state.dirty_state == {uid}
        assert [
            event for event in manager._workspace_state.dirty_events if event.uid == uid
        ] == [manager._workspace_state.dirty_events[-1]]


def test_root_imagetool_repeated_history_changes_mark_state_dirty_once(
    qtbot,
    monkeypatch,
    tmp_path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        test_data.qshow(manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        workspace = tmp_path / "normal.itws"
        manager._workspace_state.path = workspace
        manager._workspace_controller._mark_workspace_clean()
        file_path_calls: list[str] = []
        modified_calls: list[bool] = []
        monkeypatch.setattr(
            ImageToolManager,
            "setWindowFilePath",
            lambda _manager, path: file_path_calls.append(path),
        )
        monkeypatch.setattr(
            ImageToolManager,
            "setWindowModified",
            lambda _manager, modified: modified_calls.append(modified),
        )

        root = manager._tool_graph.root_wrappers[0]
        root._handle_imagetool_state_changed()
        root._handle_imagetool_state_changed()
        root._handle_imagetool_state_changed()

        assert manager._workspace_state.dirty_state == {root.uid}
        assert [
            event
            for event in manager._workspace_state.dirty_events
            if event.uid == root.uid
        ] == [manager._workspace_state.dirty_events[-1]]
        assert file_path_calls == []
        assert modified_calls == []

        manager._flush_idle_work(force=True)

        assert file_path_calls == []
        assert modified_calls == [True]


def test_remove_imagetool_removes_childtools() -> None:
    uid = "child-uid-0"
    removed_uids: list[str] = []
    removed_rows: list[int] = []
    refresh_calls: list[None] = []

    class _DummyWrapper:
        def __init__(self):
            self.uid = "root-uid-0"
            self._childtool_indices = [uid]
            self.workspace_link_key = None
            self.disposed = False
            self.deleted = False

        def dispose(self):
            self.disposed = True

        def deleteLater(self):
            self.deleted = True

    wrapper = _DummyWrapper()
    tool_graph = _ManagerToolGraph()
    tool_graph.root_wrappers[0] = wrapper
    tool_graph.nodes[wrapper.uid] = wrapper
    manager = types.SimpleNamespace(
        _tool_graph=tool_graph,
        _workspace_link_keys_for_subtree=lambda _uid: set(),
        _mark_removed_subtree_dirty=lambda _uid: None,
        _mark_singleton_workspace_link_groups_dirty=lambda _link_keys: None,
        _remove_uid_target=lambda child_uid: removed_uids.append(child_uid),
        _refresh_dependency_dependents=lambda _uid: None,
        _figure_workflows=types.SimpleNamespace(
            _refresh_figure_source_controls=lambda: refresh_calls.append(None)
        ),
        _workspace_state=types.SimpleNamespace(closing_document=False),
        tree_view=types.SimpleNamespace(
            imagetool_removed=lambda index: removed_rows.append(index)
        ),
    )

    ImageToolManager.remove_imagetool(manager, 0)
    assert removed_uids == [uid]
    assert removed_rows == [0]
    assert refresh_calls == [None]
    assert wrapper.disposed
    assert wrapper.deleted
    assert manager._tool_graph.root_wrappers == {}
    assert manager._tool_graph.nodes == {}


def test_remove_imagetools_deduplicates_explicit_child_uids() -> None:
    uid0 = "child-uid-0"
    uid1 = "child-uid-1"

    manager = types.SimpleNamespace(
        _tool_graph=_ManagerToolGraph(),
        removed_indices=[],
        removed_uids=[],
    )
    manager._tool_graph.root_wrappers.update(
        {
            0: types.SimpleNamespace(_childtool_indices=[uid0]),
            1: types.SimpleNamespace(_childtool_indices=[uid1]),
        }
    )
    manager._bulk_remove_context = contextlib.nullcontext
    manager.remove_imagetool = lambda index, *, update_view=True: (
        manager.removed_indices.append((index, update_view))
    )
    manager._remove_childtool = lambda uid: manager.removed_uids.append(uid)
    manager._iter_descendant_uids = lambda uid: []

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
            lambda: uid in manager._tool_graph.root_wrappers[0]._childtool_indices,
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

    monkeypatch.setattr(manager_base, "_NameFilterDialog", _CancelNameFilterDialog)
    manager = types.SimpleNamespace(
        _recent_loader_kwargs_by_filter={},
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


def _manager_for_loader_preference(recent_filter: str | None = None):
    return types.SimpleNamespace(
        _recent_name_filter=recent_filter,
        effective_interactive_options=erlab.interactive.options.model,
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
    manager = _manager_for_loader_preference(recent_filter)

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
    manager = _manager_for_loader_preference()

    assert (
        ImageToolManager._preferred_name_filter(manager, valid_loaders) == first_filter
    )


def test_preferred_name_filter_uses_workspace_default_loader(example_loader) -> None:
    example_filter = "Example Raw Data (*.h5)"
    valid_loaders = {
        example_filter: erlab.io.loaders["example"].file_dialog_methods[example_filter],
    }
    manager = types.SimpleNamespace(
        _recent_name_filter=None,
        effective_interactive_options=(
            erlab.interactive._options.core.model_with_workspace_overrides(
                erlab.interactive.options.model,
                {"io/default_loader": "example"},
            )
        ),
    )

    assert (
        ImageToolManager._preferred_name_filter(manager, valid_loaders)
        == example_filter
    )


def test_manager_new_imagetool_uses_workspace_options(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager._set_workspace_option_overrides(
            {
                "colors/cmap/name": "viridis",
                "colors/max_rendered_abs_value": 12.0,
            }
        )
        data = xr.DataArray(np.arange(4.0).reshape(2, 2), dims=("x", "y"))

        assert manager._data_ingress.receive_data([data], {}, show=False) == [True]

        tool = manager.get_imagetool(0)
        assert tool.slicer_area._options_model.colors.cmap.name == "viridis"
        assert tool.slicer_area.colormap_properties["cmap"] == "viridis"
        assert tool.array_slicer.display_value_abs_limit == 12.0


def test_manager_figure_generated_code_uses_workspace_stylesheets(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager._set_workspace_option_overrides(
            {"figure/stylesheets": ["classic", "missing-style"]}
        )
        data = xr.DataArray(
            np.arange(4.0).reshape(2, 2),
            dims=("x", "y"),
            name="data",
        )
        assert manager._data_ingress.receive_data([data], {}, show=False) == [True]

        uid = manager.create_figure_from_targets((0,), show=False)
        if uid is None:
            raise AssertionError("Figure Composer tool was not created")
        tool = manager._child_node(uid).tool_window
        if tool is None:
            raise AssertionError("Figure Composer tool window is missing")

        code = tool.generated_code()

        assert "plt.style.use(['classic'])" in code
        assert "# Skipped unavailable stylesheets: 'missing-style'" in code


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
        _recent_loader_kwargs_by_filter={},
        _recent_loader_extensions_by_filter={},
        _recent_name_filter=None,
        effective_interactive_options=erlab.interactive.options.model,
    )
    manager._preferred_name_filter = types.MethodType(
        ImageToolManager._preferred_name_filter, manager
    )
    manager._select_loader_options = types.MethodType(
        ImageToolManager._select_loader_options, manager
    )
    monkeypatch.setattr(manager_base, "_NameFilterDialog", _CancelNameFilterDialog)
    monkeypatch.setattr(
        erlab.interactive.utils,
        "file_loaders",
        lambda *_args: valid_loaders,
    )

    manager_io._DataIngressController(manager).open_multiple_files([file_path])

    assert dialogs[-1].checked_name == example_filter


def test_manager_open_preselects_default_loader_filter(
    monkeypatch,
    example_loader,
) -> None:
    _set_default_loader_option(monkeypatch, "example")
    example_filter = "Example Raw Data (*.h5)"
    default_directory = "/example/default"
    directories: list[str] = []
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
            directories.append(directory)

        def exec(self) -> bool:
            return False

    class _FakeManager(QtCore.QObject):
        def __init__(self) -> None:
            super().__init__()
            self._recent_name_filter = None
            self._recent_directory = None
            self.effective_interactive_options = erlab.interactive.options.model

        def _recent_or_default_directory(self) -> str | None:
            return self._recent_directory or default_directory

    manager = _FakeManager()
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

    manager_io._DataIngressController(manager).open(native=False)

    assert selected_filters == [example_filter]
    assert directories == [default_directory]


def test_managed_imagetool_open_uses_default_directory_before_recent(
    qtbot,
    monkeypatch,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    default_dir = tmp_path / "default"
    default_dir.mkdir()
    options = erlab.interactive.options.model
    monkeypatch.setattr(
        erlab.interactive.options,
        "model",
        options.model_copy(
            update={
                "io": options.io.model_copy(
                    update={"default_directory": str(default_dir)}
                )
            }
        ),
    )
    directories: list[str] = []
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
            pass

        def setDirectory(self, directory: str) -> None:
            directories.append(directory)

        def exec(self) -> bool:
            return False

    monkeypatch.setattr(QtWidgets, "QFileDialog", _FakeFileDialog)
    monkeypatch.setattr(
        erlab.interactive.utils,
        "file_loaders",
        lambda *_args: {"xarray HDF5 Files (*.h5)": (xr.load_dataarray, {})},
    )

    with manager_context() as manager:
        itool(xr.DataArray(np.arange(4).reshape((2, 2)), dims=["x", "y"]), manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        manager.get_imagetool(0)._open_file(native=False)

    assert directories == [str(default_dir)]


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


def test_manager_file_open_uses_selected_dataset_variable(
    qtbot,
    monkeypatch,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    file_path = tmp_path / "multi.h5"
    file_path.touch()
    initial_second = xr.DataArray(
        np.ones((4, 5)),
        dims=("u", "v"),
        coords={"u": np.arange(4), "v": np.arange(5)},
        name="second",
    )
    updated_second = xr.DataArray(
        np.full((4, 5), 2.0),
        dims=("u", "v"),
        coords={"u": np.arange(4), "v": np.arange(5)},
        name="second",
    )
    selection = FileDataSelection(
        kind="dataset_variable",
        value="second",
    )
    datasets = {
        "current": xr.Dataset(
            {
                "first": xr.DataArray(np.zeros((2, 3)), dims=("x", "y")),
                "second": initial_second,
            }
        )
    }

    def _load_multi(_path: pathlib.Path) -> xr.Dataset:
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
        imagetool_viewer_state,
        "_select_input_dataarrays",
        _select_second,
    )

    with manager_context() as manager:
        manager.show()
        manager._data_ingress.add_from_multiple_files(
            [],
            [file_path],
            [],
            _load_multi,
            {},
            lambda _failed: None,
        )

        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        tool = manager.get_imagetool(0)

        xr.testing.assert_identical(tool.slicer_area.data, initial_second)
        assert tool.slicer_area._load_func is not None
        assert tool.slicer_area._load_func[2] == selection

        datasets["current"] = xr.Dataset(
            {
                "inserted": xr.DataArray(np.full((2, 3), 5.0), dims=("x", "y")),
                "second": updated_second,
                "first": xr.DataArray(np.zeros((2, 3)), dims=("x", "y")),
            }
        )
        with qtbot.wait_signal(tool.slicer_area.sigDataChanged):
            tool.slicer_area.reload()

        xr.testing.assert_identical(tool.slicer_area.data, updated_second)


@pytest.mark.parametrize("case", ["cancel", "selection_error", "failed_reject"])
def test_manager_multifile_handler_selection_failure_branches(
    monkeypatch,
    tmp_path: pathlib.Path,
    case: str,
) -> None:
    file_path = tmp_path / "multi.h5"
    queued_path = tmp_path / "queued.h5"
    data = xr.Dataset({"signal": xr.DataArray(np.ones((2, 3)), dims=("x", "y"))})

    class _StatusBar:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def showMessage(self, message: str) -> None:
            self.messages.append(message)

    class _Manager(QtCore.QObject):
        def __init__(self) -> None:
            super().__init__()
            self._status_bar = _StatusBar()
            self._recent_directory: str | None = None
            self.received: list[
                tuple[tuple[typing.Any, ...], dict[str, typing.Any]]
            ] = []
            self._data_ingress = types.SimpleNamespace(receive_data=self._receive_data)

        def _receive_data(self, *args, **kwargs) -> None:
            self.received.append((args, kwargs))

    class _MessageDialog:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

        def setDetailedText(self, text: str) -> None:
            self.detail = text

        def adjustSize(self) -> None:
            self.adjusted = True

        def exec(self) -> QtWidgets.QDialog.DialogCode:
            if case == "failed_reject":
                return QtWidgets.QDialog.DialogCode.Rejected
            return QtWidgets.QDialog.DialogCode.Accepted

    manager = _Manager()
    handler = manager_io._MultiFileHandler(
        manager,
        [queued_path] if case == "failed_reject" else [],
        lambda _path: data,
        {},
    )
    finished: list[
        tuple[list[pathlib.Path], list[pathlib.Path], list[pathlib.Path]]
    ] = []
    handler.sigFinished.connect(
        lambda loaded, aborted, failed: finished.append((loaded, aborted, failed))
    )
    monkeypatch.setattr(erlab.interactive.utils, "MessageDialog", _MessageDialog)
    monkeypatch.setattr(
        erlab.interactive.utils, "single_shot", lambda _obj, _ms, func: func()
    )

    if case == "cancel":
        monkeypatch.setattr(
            imagetool_viewer_state,
            "_select_input_dataarrays",
            lambda _data, _parent: None,
        )
        handler._on_loaded(file_path, data)

        assert handler.canceled == [file_path]
        assert finished == [([], [file_path], [])]
    elif case == "selection_error":

        def _raise_selection_error(_data, _parent):
            raise ValueError("bad selection")

        monkeypatch.setattr(
            imagetool_viewer_state,
            "_select_input_dataarrays",
            _raise_selection_error,
        )
        handler._on_loaded(file_path, data)

        assert handler.failed == [file_path]
        assert finished == [([], [], [file_path])]
    else:
        handler._on_failed(file_path, "Traceback\nboom")

        assert handler.failed == [file_path]
        assert finished == [([], [queued_path], [file_path])]


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

    class _FakeManager(QtCore.QObject):
        def __init__(self) -> None:
            super().__init__()
            self._recent_name_filter = None
            self._recent_directory = None
            self.effective_interactive_options = erlab.interactive.options.model
            self._select_loader_options = _select_loader_options

        def _recent_or_default_directory(self) -> str | None:
            return self._recent_directory

    manager = _FakeManager()
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

    ingress = manager_io._DataIngressController(manager)
    monkeypatch.setattr(
        ingress,
        "add_from_multiple_files",
        lambda *args, **kwargs: add_calls.append((args, kwargs)),
    )
    ingress.open(native=False)

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

    manager = types.SimpleNamespace(
        _recent_name_filter=None,
        _select_loader_options=_select_loader_options,
    )
    ingress = manager_io._DataIngressController(manager)
    monkeypatch.setattr(
        ingress,
        "add_from_multiple_files",
        lambda loaded, queued, failed, func, kwargs, _: add_calls.append(
            (loaded, queued, failed, func, kwargs)
        ),
    )
    monkeypatch.setattr(
        erlab.interactive.utils,
        "file_loaders",
        lambda *_args: valid_loaders,
    )

    ingress.open_multiple_files([file_path])

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
                return manager.open(native=False)

        else:

            def _trigger_load():
                return manager._data_ingress.open_multiple_files([file_path])

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

        tree = manager._workspace_controller.saving._to_datatree()
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

        wrapper = manager._tool_graph.root_wrappers[0]
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


def test_manager_child_imagetool_dask_badge(
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

        parent_tool = itool(test_data, manager=False, execute=False)
        assert isinstance(parent_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(parent_tool, show=False)

        child_tool = itool(test_data.copy(deep=False), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=full_data(),
            source_auto_update=True,
        )
        child_tool.slicer_area._auto_chunk()
        assert child_tool.slicer_area.data_chunked

        view = manager.tree_view
        model = view._model
        delegate = view._delegate

        parent_index = model.index(0, 0)
        view.expand(parent_index)
        QtWidgets.QApplication.processEvents()

        child_index = model._row_index(child_uid)
        child_node = manager._child_node(child_uid)
        child_option = delegate._option_for_index(view, child_index)
        assert not child_option.rect.isEmpty()

        _, dask_rect, _, _ = delegate._compute_icons_info(child_option, child_node)
        assert dask_rect is not None
        status_rect, _, _ = delegate._compute_child_status_info(
            child_option, child_node, right_edge=dask_rect.left()
        )
        assert status_rect is not None
        assert not dask_rect.intersects(status_rect)

        editor = QtWidgets.QLineEdit(view.viewport())
        delegate.updateEditorGeometry(editor, child_option, child_index)
        assert editor.geometry().right() < status_rect.left()
        editor.deleteLater()

        dask_badge = delegate._badge_at(child_option, child_index, dask_rect.center())
        assert dask_badge is not None
        assert dask_badge.kind == "dask"

        status_badge = delegate._badge_at(
            child_option, child_index, status_rect.center()
        )
        assert status_badge is not None
        assert status_badge.kind == "source_status"

        text = None

        def fake_show_text(pos, s, *args, **kwargs):
            nonlocal text
            text = s

        monkeypatch.setattr(QtWidgets.QToolTip, "showText", fake_show_text)

        event = QtGui.QHelpEvent(
            QtCore.QEvent.Type.ToolTip,
            dask_rect.center(),
            view.viewport().mapToGlobal(dask_rect.center()),
        )
        handled = delegate.helpEvent(event, view, child_option, child_index)
        assert handled
        assert_nonempty_tooltip(text)

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

        requested_cursors: list[QtCore.Qt.CursorShape | None] = []
        monkeypatch.setattr(
            erlab.interactive.utils,
            "set_widget_cursor",
            lambda widget, shape: requested_cursors.append(shape),
        )
        delegate.eventFilter(view.viewport(), _mouse_move(dask_rect.center()))
        assert requested_cursors == [QtCore.Qt.CursorShape.PointingHandCursor]

        popup_positions: list[QtCore.QPoint] = []
        monkeypatch.setattr(child_tool._dask_menu, "popup", popup_positions.append)
        click_tree_view_pos(view, dask_rect.center())
        assert popup_positions == [view.viewport().mapToGlobal(dask_rect.bottomLeft())]

        source_dialog_parents: list[ImageToolManager] = []
        monkeypatch.setattr(
            child_node,
            "show_source_update_dialog",
            lambda *, parent: source_dialog_parents.append(parent),
        )
        click_tree_view_pos(view, status_rect.center())
        assert source_dialog_parents == [manager]


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
        wrapper = manager._tool_graph.root_wrappers[0]

        manager.get_imagetool(0).slicer_area._auto_chunk()
        view.refresh(0)
        option = delegate._option_for_index(view, index)
        _, dask_rect, _, _ = delegate._compute_icons_info(option, wrapper)
        assert dask_rect is not None

        assert delegate._badge_at(option, QtCore.QModelIndex(), QtCore.QPoint()) is None
        malformed_pointer = object()
        assert (
            delegate._badge_at(
                option,
                model.createIndex(0, 0, malformed_pointer),
                option.rect.center(),
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

        requested_cursors: list[QtCore.Qt.CursorShape | None] = []
        monkeypatch.setattr(
            erlab.interactive.utils,
            "set_widget_cursor",
            lambda widget, shape: requested_cursors.append(shape),
        )
        delegate.eventFilter(view.viewport(), _mouse_move(dask_rect.center()))
        assert requested_cursors[-1] == QtCore.Qt.CursorShape.PointingHandCursor

        delegate.eventFilter(view.viewport(), _mouse_move(option.rect.center()))
        assert requested_cursors[-1] is None

        delegate.eventFilter(view.viewport(), _mouse_move(QtCore.QPoint(-10, -10)))
        assert requested_cursors[-1] is None

        delegate.eventFilter(view.viewport(), QtCore.QEvent(QtCore.QEvent.Type.Leave))
        assert requested_cursors[-1] is None
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
