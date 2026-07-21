import json
import pathlib
import types
import typing
from collections.abc import Callable, Sequence

import numpy as np
import pytest
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

import erlab
import erlab.interactive.imagetool.manager._details_panel as manager_details_panel
import erlab.interactive.imagetool.manager._widgets as manager_widgets
import erlab.interactive.imagetool.manager._wrapper as manager_wrapper
from erlab.interactive.imagetool import _kspace_conversion, itool
from erlab.interactive.imagetool._provenance._code import uses_default_replay_input
from erlab.interactive.imagetool._provenance._execution import (
    replay_file_provenance,
    replay_script_provenance,
)
from erlab.interactive.imagetool._provenance._model import (
    DerivationEntry,
    ScriptInput,
    ToolProvenanceOperation,
    ToolProvenanceSpec,
    _ProvenanceDisplayRow,
    _ProvenanceStepRef,
    full_data,
    operation_group_range,
    script,
    stamp_operation_group,
    strip_operation_groups,
)
from erlab.interactive.imagetool._provenance._operations import (
    AffineCoordOperation,
    AssignAttrsOperation,
    AverageOperation,
    DivideByCoordOperation,
    GaussianFilterOperation,
    IselOperation,
    KspaceConvertOperation,
    KspaceSetNormalOperation,
    KspaceWorkFunctionOperation,
    ScriptCodeOperation,
    SortByOperation,
    SwapDimsOperation,
    TransposeOperation,
)
from erlab.interactive.imagetool.manager import replace_data
from erlab.interactive.imagetool.manager._provenance_edit import (
    _controller as provenance_edit_controller,
)
from tests.interactive.imagetool.manager.helpers import (
    _exec_generated_code,
    metadata_derivation_texts,
    select_child_tool,
    select_metadata_rows,
    select_tools,
    set_transform_launch_mode,
    trigger_menu_action,
)

from ._common import (
    _add_file_replay_tool,
    _fake_edit_controller,
    _fake_edit_node,
    _manager_replay_file_spec,
    _provenance_paste_test_data,
    _set_provenance_steps_clipboard,
)


def test_manager_provenance_step_clipboard_payload_validation() -> None:
    operation_payload = AverageOperation(dims=("z",)).model_dump(mode="json")
    valid_payload = {
        "type": manager_details_panel._PROVENANCE_STEPS_CLIPBOARD_PAYLOAD_TYPE,
        "version": manager_details_panel._PROVENANCE_STEPS_CLIPBOARD_PAYLOAD_VERSION,
        "operations": [operation_payload],
    }

    mime_data = QtCore.QMimeData()
    mime_data.setData(
        manager_details_panel._PROVENANCE_STEPS_CLIPBOARD_MIME,
        json.dumps(valid_payload).encode("utf-8"),
    )
    payload = manager_details_panel._provenance_step_clipboard_payload(mime_data)
    assert payload is not None
    assert payload[0] == (AverageOperation(dims=("z",)),)
    assert payload[1] == "derived"
    assert not payload[2]

    text_mime_data = QtCore.QMimeData()
    text_mime_data.setText(json.dumps({**valid_payload, "active_name": "result"}))
    payload = manager_details_panel._provenance_step_clipboard_payload(text_mime_data)
    assert payload is not None
    assert payload[1] == "result"

    plain_text_mime_data = QtCore.QMimeData()
    plain_text_mime_data.setText("derived = data")
    assert (
        manager_details_panel._provenance_step_clipboard_payload(plain_text_mime_data)
        is None
    )
    assert (
        manager_details_panel._provenance_step_clipboard_payload(QtCore.QMimeData())
        is None
    )

    invalid_utf8_mime_data = QtCore.QMimeData()
    invalid_utf8_mime_data.setData(
        manager_details_panel._PROVENANCE_STEPS_CLIPBOARD_MIME,
        b"\xff",
    )
    assert (
        manager_details_panel._provenance_step_clipboard_payload(invalid_utf8_mime_data)
        is None
    )

    for malformed_payload in (
        [],
        {**valid_payload, "type": "other"},
        {**valid_payload, "version": -1},
        {**valid_payload, "operations": {}},
        {**valid_payload, "operations": [{"op": "unknown"}]},
    ):
        malformed_mime_data = QtCore.QMimeData()
        malformed_mime_data.setData(
            manager_details_panel._PROVENANCE_STEPS_CLIPBOARD_MIME,
            json.dumps(malformed_payload).encode("utf-8"),
        )
        assert (
            manager_details_panel._provenance_step_clipboard_payload(
                malformed_mime_data
            )
            is None
        )


def test_manager_selected_derivation_step_payload_filters_rows() -> None:
    def item(
        row: _ProvenanceDisplayRow | str,
        *,
        copyable: bool = True,
        code: str | None = "derived = data",
    ) -> QtWidgets.QListWidgetItem:
        list_item = QtWidgets.QListWidgetItem()
        list_item.setData(manager_details_panel._METADATA_DERIVATION_ROW_ROLE, row)
        list_item.setData(
            manager_details_panel._METADATA_DERIVATION_COPYABLE_ROLE,
            copyable,
        )
        list_item.setData(manager_details_panel._METADATA_DERIVATION_CODE_ROLE, code)
        return list_item

    operation_ref = _ProvenanceStepRef("operation", operation_index=0)
    script_row_a = _ProvenanceDisplayRow(
        DerivationEntry("script a", None),
        replay_ref=operation_ref,
    )
    script_row_b = _ProvenanceDisplayRow(
        DerivationEntry("script b", None),
        replay_ref=operation_ref,
    )
    missing_operation_row = _ProvenanceDisplayRow(
        DerivationEntry("missing op", None),
        replay_ref=_ProvenanceStepRef(
            "operation",
            operation_index=10,
        ),
    )
    non_copyable_script_row = _ProvenanceDisplayRow(
        DerivationEntry("non-copyable script", None),
        replay_ref=operation_ref,
    )
    non_live_row = _ProvenanceDisplayRow(
        DerivationEntry("non-live", None),
        replay_ref=operation_ref,
    )
    row_to_spec = {
        script_row_a: script(
            ScriptCodeOperation(label="script a", code="a = data"),
            start_label="Start script a",
            active_name="a",
        ),
        script_row_b: script(
            ScriptCodeOperation(label="script b", code="b = data"),
            start_label="Start script b",
            active_name="b",
        ),
        missing_operation_row: full_data(AverageOperation(dims=("x",))),
        non_copyable_script_row: types.SimpleNamespace(
            _operation_for_ref=lambda _ref: ScriptCodeOperation(
                label="hidden script",
                code="derived = data",
                copyable=False,
            )
        ),
        non_live_row: types.SimpleNamespace(
            _operation_for_ref=lambda _ref: types.SimpleNamespace(live_applicable=False)
        ),
    }
    edit_controller = types.SimpleNamespace(
        _display_spec_for_row=lambda _node, row: row_to_spec.get(row)
    )
    selected_items: list[QtWidgets.QListWidgetItem] = []
    manager = types.SimpleNamespace(
        _metadata_node_uid="node",
        _tool_graph=types.SimpleNamespace(nodes={"node": object()}),
        _selected_derivation_items=lambda: selected_items,
        _provenance_edit_controller=edit_controller,
    )
    controller = manager_details_panel._DetailsPanelController(
        typing.cast("typing.Any", manager)
    )

    manager._metadata_node_uid = None
    assert controller._selected_derivation_step_payload() is None

    manager._metadata_node_uid = "missing"
    assert controller._selected_derivation_step_payload() is None

    manager._metadata_node_uid = "node"
    selected_items = [
        item("not a row"),
        item(_ProvenanceDisplayRow(DerivationEntry("start", None))),
        item(
            _ProvenanceDisplayRow(
                DerivationEntry("file", None),
                replay_ref=_ProvenanceStepRef("file_load"),
            )
        ),
        item(script_row_a, copyable=False),
        item(script_row_a, code=None),
        item(
            _ProvenanceDisplayRow(
                DerivationEntry("no spec", None),
                replay_ref=operation_ref,
            )
        ),
        item(missing_operation_row),
        item(non_copyable_script_row),
        item(non_live_row),
    ]
    assert controller._selected_derivation_step_payload() is None

    selected_items = [item(script_row_a), item(script_row_b)]
    assert controller._selected_derivation_step_payload() is None


def test_manager_metadata_derivation_rows_render_as_tree(qtbot) -> None:
    child_row = _ProvenanceDisplayRow(
        DerivationEntry("Offset parent", "data_0 = data_0 + 1", True),
        edit_ref=_ProvenanceStepRef("operation", operation_index=0),
        replay_ref=_ProvenanceStepRef("operation", operation_index=0),
        script_input_path=(0,),
    )
    parent_row = _ProvenanceDisplayRow(
        DerivationEntry("Use data_0 from Parent", None, False),
        replay_ref=_ProvenanceStepRef(
            "script_input",
            script_input_index=0,
        ),
        children=(child_row,),
    )
    sibling_row = _ProvenanceDisplayRow(
        DerivationEntry("Use derived data", None, False),
        replay_ref=_ProvenanceStepRef("operation", operation_index=0),
    )
    derivation_list = manager_widgets._MetadataDerivationListWidget()
    qtbot.addWidget(derivation_list)
    manager = types.SimpleNamespace(
        metadata_derivation_list=derivation_list,
        _provenance_edit_controller=types.SimpleNamespace(
            can_edit_row=lambda row: (row is child_row, "")
        ),
        _set_metadata_fields=lambda _fields: None,
        _update_metadata_pane=lambda: None,
    )
    controller = manager_details_panel._DetailsPanelController(
        typing.cast("typing.Any", manager)
    )
    node = types.SimpleNamespace(
        uid="node",
        displayed_provenance_spec=full_data(),
        metadata_fields=[],
        derivation_display_rows=[parent_row, sibling_row],
    )

    controller._set_metadata_node(node)

    parent_item = derivation_list.topLevelItem(0)
    assert parent_item is not None
    assert parent_item.text() == "Use data_0 from Parent"
    parent_tree_item = typing.cast(
        "manager_widgets._MetadataDerivationTreeItem",
        parent_item,
    )
    parent_tree_item.setText("Use data_0 from Renamed Parent")
    assert parent_tree_item.text() == "Use data_0 from Renamed Parent"
    parent_tree_item.setText(0, "Use data_0 from Parent")
    assert parent_tree_item.text(0) == "Use data_0 from Parent"
    parent_tree_item.setData(QtCore.Qt.ItemDataRole.UserRole + 20, "one-arg")
    assert parent_tree_item.data(QtCore.Qt.ItemDataRole.UserRole + 20) == "one-arg"
    parent_tree_item.setData(0, QtCore.Qt.ItemDataRole.UserRole + 21, "two-arg")
    assert parent_tree_item.data(0, QtCore.Qt.ItemDataRole.UserRole + 21) == "two-arg"
    parent_tree_item.setToolTip("one-arg tooltip")
    assert parent_tree_item.toolTip() == "one-arg tooltip"
    parent_tree_item.setToolTip(0, "two-arg tooltip")
    assert parent_tree_item.toolTip(0) == "two-arg tooltip"
    parent_tree_item.setForeground(QtGui.QColor("red"))
    assert parent_tree_item.foreground().color() == QtGui.QColor("red")
    parent_tree_item.setForeground(0, QtGui.QColor("blue"))
    assert parent_tree_item.foreground(0).color() == QtGui.QColor("blue")
    with pytest.raises(TypeError):
        parent_tree_item.setText()
    with pytest.raises(TypeError):
        parent_tree_item.data()
    with pytest.raises(TypeError):
        parent_tree_item.setData(QtCore.Qt.ItemDataRole.UserRole)
    with pytest.raises(TypeError):
        parent_tree_item.setToolTip()
    with pytest.raises(TypeError):
        parent_tree_item.setForeground()

    assert parent_item.childCount() == 1
    child_item = parent_item.child(0)
    assert child_item is not None
    assert child_item.text() == "Offset parent"
    assert (
        child_item.data(manager_details_panel._METADATA_DERIVATION_ROW_ROLE)
        is child_row
    )
    sibling_item = derivation_list.topLevelItem(1)
    assert sibling_item is not None
    assert (
        sibling_item.data(manager_details_panel._METADATA_DERIVATION_ROW_ROLE)
        is sibling_row
    )
    assert derivation_list.count() == 3
    assert derivation_list.item(0) is parent_item
    assert derivation_list.item(1) is child_item
    assert derivation_list.item(2) is sibling_item
    assert derivation_list.item(3) is None
    assert derivation_list.row(parent_item) == 0
    assert derivation_list.row(child_item) == 1
    assert derivation_list.row(sibling_item) == 2
    assert derivation_list.display_order(child_item) == 1
    assert (
        derivation_list.display_order(
            manager_widgets._MetadataDerivationTreeItem("orphan")
        )
        == -1
    )
    derivation_list.setUniformItemSizes(True)
    assert derivation_list.uniformRowHeights()


def test_manager_metadata_script_input_labels_use_current_nodes(qtbot) -> None:
    source_spec = script(
        start_label="Build source",
        seed_code="derived = xr.DataArray([1.0], dims=('x',))",
        active_name="derived",
    )
    spec = script(
        ScriptCodeOperation(label="Copy", code="derived = data_10"),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="data_10",
                label="ImageTool 10: stale",
                node_uid="n16",
                provenance_spec=source_spec,
            ),
        ),
    )
    derivation_list = manager_widgets._MetadataDerivationListWidget()
    qtbot.addWidget(derivation_list)
    source_node = types.SimpleNamespace(
        uid="n16",
        parent_uid=None,
        is_imagetool=True,
        name="3 V",
        type_badge_text=None,
        _childtool_indices=[],
    )
    manager = types.SimpleNamespace(
        metadata_derivation_list=derivation_list,
        _tool_graph=types.SimpleNamespace(
            nodes={"n16": source_node},
            root_wrappers={4: source_node},
        ),
        _provenance_edit_controller=types.SimpleNamespace(
            can_edit_row=lambda _row: (False, "")
        ),
        _set_metadata_fields=lambda _fields: None,
        _update_metadata_pane=lambda: None,
    )
    controller = manager_details_panel._DetailsPanelController(
        typing.cast("typing.Any", manager)
    )
    node = types.SimpleNamespace(
        uid="node",
        displayed_provenance_spec=spec,
        metadata_fields=[],
        derivation_display_rows=spec.display_rows(),
    )

    controller._set_metadata_node(node)

    input_item = derivation_list.topLevelItem(1)
    assert input_item is not None
    assert input_item.text() == "Use data_10 from ImageTool 4: 3 V"
    assert "ImageTool 10" not in input_item.text()


def test_manager_metadata_missing_script_input_uses_neutral_label(qtbot) -> None:
    spec = script(
        ScriptCodeOperation(label="Copy", code="derived = data_10"),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="data_10",
                label="ImageTool 10: stale",
                node_uid="missing",
            ),
        ),
    )
    derivation_list = manager_widgets._MetadataDerivationListWidget()
    qtbot.addWidget(derivation_list)
    manager = types.SimpleNamespace(
        metadata_derivation_list=derivation_list,
        _tool_graph=types.SimpleNamespace(nodes={}, root_wrappers={}),
        _provenance_edit_controller=types.SimpleNamespace(
            can_edit_row=lambda _row: (False, "")
        ),
        _set_metadata_fields=lambda _fields: None,
        _update_metadata_pane=lambda: None,
    )
    controller = manager_details_panel._DetailsPanelController(
        typing.cast("typing.Any", manager)
    )
    node = types.SimpleNamespace(
        uid="node",
        displayed_provenance_spec=spec,
        metadata_fields=[],
        derivation_display_rows=spec.display_rows(),
    )

    controller._set_metadata_node(node)
    details = controller._unavailable_replay_code_details(
        typing.cast("typing.Any", node)
    )

    input_item = derivation_list.topLevelItem(1)
    assert input_item is not None
    assert input_item.text() == "Missing source for data_10"
    assert "ImageTool 10" not in input_item.text()
    assert "Missing source for data_10 (recorded as ImageTool 10: stale)" in details


def test_manager_copy_selected_derivation_code_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = types.SimpleNamespace(_selected_derivation_code=lambda: "derived = data")
    controller = manager_details_panel._DetailsPanelController(
        typing.cast("typing.Any", manager)
    )

    copied: list[str] = []
    monkeypatch.setattr(
        erlab.interactive.utils,
        "copy_to_clipboard",
        lambda text: copied.append(text) or text,
    )
    monkeypatch.setattr(
        controller,
        "_selected_derivation_step_payload",
        lambda: None,
    )
    controller._copy_selected_derivation_code()
    assert copied == ["derived = data"]

    monkeypatch.setattr(
        controller,
        "_selected_derivation_step_payload",
        lambda: ((AverageOperation(dims=("x",)),), "derived", False),
    )
    monkeypatch.setattr(QtWidgets.QApplication, "clipboard", lambda: None)
    controller._copy_selected_derivation_code()


def test_manager_unavailable_replay_details_skip_replayable_script_inputs() -> None:
    source_spec = script(
        start_label="Build source",
        seed_code="derived = xr.DataArray([1.0], dims=('x',))",
        active_name="derived",
    )
    spec = script(
        ScriptCodeOperation(
            label="Run opaque code",
            code=None,
            copyable=False,
        ),
        start_label="Build figure",
        active_name="fig",
        script_inputs=(
            ScriptInput(
                name="data_3",
                label="ImageTool 3: D10cu",
                provenance_spec=source_spec,
            ),
        ),
    )
    node = types.SimpleNamespace(
        displayed_provenance_spec=spec,
        derivation_entries=spec.derivation_entries(),
        derivation_display_rows=spec.display_rows(),
    )
    controller = manager_details_panel._DetailsPanelController(
        typing.cast("typing.Any", types.SimpleNamespace())
    )

    details = controller._unavailable_replay_code_details(
        typing.cast("typing.Any", node)
    )

    assert "Run opaque code" in details
    assert "Use data_3 from ImageTool 3: D10cu" not in details


def test_manager_derivation_context_menu_ignores_empty_space(
    qtbot,
) -> None:
    list_widget = QtWidgets.QListWidget()
    qtbot.addWidget(list_widget)
    calls: list[bool] = []
    exec_positions: list[QtCore.QPoint] = []

    class _Menu:
        def exec(self, pos: QtCore.QPoint) -> None:
            exec_positions.append(pos)

    def build_menu(*, include_row_actions: bool) -> QtWidgets.QMenu:
        calls.append(include_row_actions)
        return typing.cast("QtWidgets.QMenu", _Menu())

    manager = types.SimpleNamespace(
        metadata_derivation_list=list_widget,
        _build_metadata_derivation_menu=build_menu,
    )
    controller = manager_details_panel._DetailsPanelController(
        typing.cast("typing.Any", manager)
    )

    controller._show_metadata_derivation_menu(QtCore.QPoint(10, 10))

    assert calls == [False]
    assert len(exec_positions) == 1


def test_manager_paste_steps_validation_error_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _fake_edit_controller(None, metadata_uid=None)
    unavailable: list[str] = []
    monkeypatch.setattr(controller, "_show_unavailable", unavailable.append)
    controller.paste_steps(
        (AverageOperation(dims=("x",)),),
        active_name="derived",
        contains_script=False,
    )
    assert unavailable

    node = _fake_edit_node(
        full_data(),
        parent_uid="parent",
        source_display_spec=full_data(),
    )
    with pytest.raises(TypeError, match="Only live provenance operations"):
        controller._paste_structured_steps(
            node,
            (ScriptCodeOperation(label="script", code="derived = data"),),
        )

    monkeypatch.setattr(
        controller,
        "_replay_candidate_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad replay")),
    )
    with pytest.raises(provenance_edit_controller._ProvenanceReplayFailure) as exc_info:
        controller._paste_structured_steps(
            node,
            (AverageOperation(dims=("x",)),),
        )
    assert "pasted provenance steps" in exc_info.value.where


def test_manager_paste_steps_targets_editable_imagetools_only() -> None:
    editable = _fake_edit_node(full_data(), uid="editable")
    pending = _fake_edit_node(full_data(), uid="pending")
    pending.imagetool = None
    pending.pending_workspace_memory_payload = (
        pathlib.Path("workspace.itws"),
        "0/imagetool",
    )
    unavailable = _fake_edit_node(full_data(), uid="unavailable")
    unavailable.imagetool = None
    non_imagetool = _fake_edit_node(full_data(), uid="non-imagetool")
    non_imagetool.is_imagetool = False
    graph_nodes = {
        editable.uid: editable,
        pending.uid: pending,
        unavailable.uid: unavailable,
        non_imagetool.uid: non_imagetool,
    }
    controller = _fake_edit_controller(
        editable,
        nodes=graph_nodes,
        metadata_uid=None,
    )
    controller._manager._selected_imagetool_targets = lambda: tuple(graph_nodes)

    assert controller._paste_target_nodes() == [editable, pending]
    assert controller.can_paste_steps(
        (AssignAttrsOperation(attrs={"copied": "yes"}),)
    ) == (True, "")


def test_manager_paste_detached_steps_uses_replay_spec_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _provenance_paste_test_data()
    replay_source = data.copy(deep=True)
    local = full_data(AverageOperation(dims=("z",)))
    controller = _fake_edit_controller(None, metadata_uid=None)
    replaced: list[
        tuple[xr.DataArray, ToolProvenanceSpec, bool, xr.DataArray | None]
    ] = []

    def _replace_with_detached_data(
        result: xr.DataArray,
        spec: ToolProvenanceSpec,
        *,
        preserve_filter: bool,
        replay_source_data: xr.DataArray | None,
    ) -> None:
        replaced.append((result, spec, preserve_filter, replay_source_data))

    node = types.SimpleNamespace(
        uid="node",
        displayed_provenance_spec=full_data(),
        current_public_data=lambda: data,
        resolved_replay_source_data=lambda: replay_source,
        replace_with_detached_data=_replace_with_detached_data,
    )
    monkeypatch.setattr(
        provenance_edit_controller,
        "compose_full_provenance",
        lambda _parent, _local: None,
    )

    controller._paste_detached_steps(
        typing.cast("typing.Any", node),
        local,
        where="testing fallback",
    )

    assert len(replaced) == 1
    xr.testing.assert_identical(replaced[0][0], local.apply(data))
    assert replaced[0][1] == local.to_replay_spec()
    assert replaced[0][2] is False
    assert replaced[0][3] is replay_source


def test_manager_can_delete_row_reports_unavailable_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation_ref = _ProvenanceStepRef(
        "operation",
        operation_index=0,
    )
    row = _ProvenanceDisplayRow(
        DerivationEntry("Average", None),
        replay_ref=operation_ref,
    )

    controller = _fake_edit_controller(None, metadata_uid=None)
    deletable, reason = controller.can_delete_row(row)
    assert not deletable
    assert reason

    source_child = _fake_edit_node(
        full_data(AverageOperation(dims=("x",))),
        parent_uid="parent",
        source_spec=full_data(),
    )
    controller = _fake_edit_controller(source_child)
    deletable, reason = controller.can_delete_row(row)
    assert not deletable
    assert reason

    controller = _fake_edit_controller(_fake_edit_node(None))
    deletable, reason = controller.can_delete_row(row)
    assert not deletable
    assert reason

    missing_ref_row = _ProvenanceDisplayRow(
        DerivationEntry("Missing", None),
        replay_ref=_ProvenanceStepRef(
            "operation",
            operation_index=99,
        ),
    )
    controller = _fake_edit_controller(
        _fake_edit_node(full_data(AverageOperation(dims=("x",))))
    )
    deletable, reason = controller.can_delete_row(missing_ref_row)
    assert not deletable
    assert reason

    broken_script_spec = types.SimpleNamespace(
        kind="script",
        operations=(
            ScriptCodeOperation(
                label="Broken",
                code="derived = data",
            ),
        ),
        _operation_for_ref=lambda _ref: ScriptCodeOperation(
            label="Broken",
            code="derived = data",
        ),
        _replace_operation_ref=lambda *_args: (_ for _ in ()).throw(
            ValueError("bad ref")
        ),
    )
    controller = _fake_edit_controller(
        _fake_edit_node(typing.cast("typing.Any", broken_script_spec))
    )
    deletable, reason = controller.can_delete_row(row)
    assert not deletable
    assert reason

    valid_script_spec = script(
        ScriptCodeOperation(
            label="Offset",
            code="derived = derived + 1",
        ),
        ScriptCodeOperation(
            label="Scale",
            code="derived = derived * 2",
        ),
        start_label="Start from data",
        seed_code="derived = data_0",
        active_name="derived",
        script_inputs=(ScriptInput(name="data_0", label="Input"),),
    )
    controller = _fake_edit_controller(_fake_edit_node(valid_script_spec))
    deletable, reason = controller.can_delete_row(valid_script_spec.display_rows()[2])
    assert deletable
    assert reason == ""

    source_live_spec = full_data(AverageOperation(dims=("x",)))
    live_row = source_live_spec.display_rows(scope="source")[1]
    source_bound = _fake_edit_node(
        full_data(),
        parent_uid="parent",
        source_display_spec=source_live_spec,
    )
    controller = _fake_edit_controller(source_bound)
    deletable, reason = controller.can_delete_row(live_row)
    assert deletable
    assert reason == ""

    controller = _fake_edit_controller(_fake_edit_node(source_live_spec))
    deletable, reason = controller.can_delete_row(source_live_spec.display_rows()[1])
    assert not deletable
    assert reason


def test_manager_delete_row_error_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _ProvenanceDisplayRow(
        DerivationEntry("Average", None),
        replay_ref=_ProvenanceStepRef(
            "operation",
            operation_index=0,
        ),
    )
    node = _fake_edit_node(full_data(AverageOperation(dims=("x",))))
    controller = _fake_edit_controller(node)
    monkeypatch.setattr(controller, "can_delete_row", lambda _row: (True, ""))

    failures: list[tuple[str, Exception]] = []
    monkeypatch.setattr(
        controller,
        "_show_failed",
        lambda title, exc: failures.append((title, exc)),
    )
    monkeypatch.setattr(controller, "_display_spec_for_row", lambda *_args: None)
    controller.delete_row(row)
    assert failures

    failures.clear()
    monkeypatch.setattr(
        controller,
        "_display_spec_for_row",
        lambda *_args: full_data(AverageOperation(dims=("x",))),
    )
    monkeypatch.setattr(
        controller,
        "_validate_and_replace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad replay")),
    )
    monkeypatch.setattr(
        controller,
        "_handle_missing_source_file",
        lambda *_args, **_kwargs: False,
    )
    controller.delete_row(row)
    assert failures

    failures.clear()
    monkeypatch.setattr(
        controller,
        "_handle_missing_source_file",
        lambda *_args, **_kwargs: True,
    )
    controller.delete_row(row)
    assert failures == []

    live_row = full_data(AverageOperation(dims=("x",))).display_rows()[1]
    replaced: list[
        tuple[
            object,
            typing.Literal["display", "source"],
            ToolProvenanceSpec,
        ]
    ] = []
    monkeypatch.setattr(controller, "can_delete_row", lambda _row: (True, ""))
    monkeypatch.setattr(
        controller,
        "_display_spec_for_row",
        lambda *_args: full_data(AverageOperation(dims=("x",))),
    )
    monkeypatch.setattr(
        controller,
        "_validate_and_replace",
        lambda edit_node, scope, candidate, **_kwargs: replaced.append(
            (edit_node, scope, candidate)
        ),
    )
    controller.delete_row(live_row)
    assert len(replaced) == 1
    assert replaced[0][2].operations == ()


def test_manager_delete_provenance_step_preserves_later_operations(
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = _provenance_paste_test_data("scan")
    file_path = tmp_path / "scan.h5"
    data.to_netcdf(file_path, engine="h5netcdf")
    operations = (
        AverageOperation(dims=("z",)),
        IselOperation(kwargs={"y": 1}),
    )
    spec = _manager_replay_file_spec(file_path, *operations)
    displayed = replay_file_provenance(spec)

    with manager_context() as manager:
        tool = _add_file_replay_tool(manager, displayed, spec)
        index = 0

        select_tools(manager, [index])
        manager._update_info()
        select_metadata_rows(manager, [1])
        row = manager._selected_derivation_row()
        deletable, reason = manager._provenance_edit_controller.can_delete_row(row)
        assert deletable, reason
        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        trigger_menu_action(menu, manager._metadata_delete_step_action)

        expected_spec = spec._replace_operation_ref(
            _ProvenanceStepRef(
                "operation",
                operation_index=0,
            ),
            (),
        )
        expected = replay_file_provenance(expected_spec)
        xr.testing.assert_identical(tool.slicer_area._data, expected)
        root = manager._tool_graph.root_wrappers[index]
        assert root.provenance_spec == expected_spec
        assert metadata_derivation_texts(manager) == [
            f"Load data from file {file_path.name!r}",
            operations[1].derivation_entry().label,
        ]


def test_manager_delete_invalid_script_step_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = _provenance_paste_test_data("scan")
    operations = (
        ScriptCodeOperation(
            label="Create result",
            code="result = derived + 1.0",
        ),
        ScriptCodeOperation(
            label="Use result",
            code="result = result * 2.0",
        ),
    )
    spec = script(
        *operations,
        start_label="Start from data",
        seed_code="derived = data",
        active_name="result",
    )
    displayed = replay_script_provenance(spec, {"data": data})

    with manager_context() as manager:
        tool = itool(displayed, manager=False, execute=False)
        assert isinstance(tool, erlab.interactive.imagetool.ImageTool)
        index = manager.add_imagetool(tool, show=False, provenance_spec=spec)

        select_tools(manager, [index])
        manager._update_info()
        select_metadata_rows(manager, [1])
        row = manager._selected_derivation_row()
        deletable, reason = manager._provenance_edit_controller.can_delete_row(row)
        assert not deletable
        assert reason

        unavailable: list[str] = []
        monkeypatch.setattr(
            manager._provenance_edit_controller,
            "_show_unavailable",
            unavailable.append,
        )
        before = tool.slicer_area._data.copy(deep=True)
        before_spec = manager._tool_graph.root_wrappers[index].provenance_spec
        manager._delete_selected_derivation_step()

        assert unavailable
        xr.testing.assert_identical(tool.slicer_area._data, before)
        assert manager._tool_graph.root_wrappers[index].provenance_spec == before_spec


def test_manager_copy_paste_structured_provenance_steps(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source_base = _provenance_paste_test_data("source")
    source_operations = (
        AssignAttrsOperation(attrs={"copied": "yes"}),
        AverageOperation(dims=("z",)),
    )
    source_spec = full_data(*source_operations)
    source_data = source_spec.apply(source_base)

    dest_base = _provenance_paste_test_data("dest") + 100.0
    dest_seed_op = ScriptCodeOperation(
        label="Keep existing destination provenance",
        code="derived = derived.assign_attrs({'existing': 'yes'})",
    )
    dest_spec = script(
        dest_seed_op,
        start_label="Start from destination data",
        seed_code="derived = data",
        active_name="derived",
    )
    dest_data = replay_script_provenance(dest_spec, {"data": dest_base})

    with manager_context() as manager:
        source_tool = itool(source_data, manager=False, execute=False)
        assert isinstance(source_tool, erlab.interactive.imagetool.ImageTool)
        source_index = manager.add_imagetool(
            source_tool,
            show=False,
            provenance_spec=source_spec,
        )
        dest_tool = itool(dest_data, manager=False, execute=False)
        assert isinstance(dest_tool, erlab.interactive.imagetool.ImageTool)
        dest_index = manager.add_imagetool(
            dest_tool,
            show=False,
            provenance_spec=dest_spec,
        )

        clipboard = QtWidgets.QApplication.clipboard()
        clipboard.clear()
        manager.tree_view.clearSelection()
        select_tools(manager, [source_index])
        manager._update_info()
        select_metadata_rows(manager, [1, 2])
        qtbot.keyClick(
            manager.metadata_derivation_list,
            QtCore.Qt.Key.Key_C,
            QtCore.Qt.KeyboardModifier.ControlModifier,
        )

        payload = manager_details_panel._provenance_step_clipboard_payload(
            clipboard.mimeData()
        )
        assert payload is not None
        payload_operations, _active_name, contains_script = payload
        assert payload_operations == source_operations
        assert not contains_script
        assert clipboard.mimeData().hasFormat(
            manager_details_panel._PROVENANCE_STEPS_CLIPBOARD_MIME
        )
        copied_namespace = _exec_generated_code(
            clipboard.text(),
            {"derived": dest_data.copy(deep=True)},
        )
        copied_result = copied_namespace["derived"]
        assert isinstance(copied_result, xr.DataArray)
        expected = source_spec.apply(dest_data)
        xr.testing.assert_identical(copied_result, expected)

        manager.tree_view.clearSelection()
        select_tools(manager, [dest_index])
        manager._update_info()
        select_metadata_rows(manager, [0])
        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        assert manager._metadata_paste_steps_action in menu.actions()
        assert manager._metadata_paste_steps_action.isEnabled()

        manager.metadata_derivation_list.setFocus()
        qtbot.keyClick(
            manager.metadata_derivation_list,
            QtCore.Qt.Key.Key_V,
            QtCore.Qt.KeyboardModifier.ControlModifier,
        )
        xr.testing.assert_identical(dest_tool.slicer_area._data, expected)

        dest_node = manager._tool_graph.root_wrappers[dest_index]
        assert dest_node.displayed_provenance_spec is not None
        assert dest_node.displayed_provenance_spec.script_context_bindings == ()
        replay_namespace = _exec_generated_code(
            dest_node.displayed_provenance_spec.derivation_code(),
            {"data": dest_base.copy(deep=True)},
        )
        replayed = replay_namespace["derived"]
        assert isinstance(replayed, xr.DataArray)
        xr.testing.assert_identical(replayed, expected)

        bad_mime = QtCore.QMimeData()
        bad_mime.setData(
            manager_details_panel._PROVENANCE_STEPS_CLIPBOARD_MIME,
            b"{not-json",
        )
        clipboard.setMimeData(bad_mime)
        failures: list[str] = []
        monkeypatch.setattr(
            manager._provenance_edit_controller,
            "_show_unavailable",
            lambda reason: failures.append(reason),
        )
        before = dest_tool.slicer_area._data.copy(deep=True)
        manager._paste_provenance_steps_from_clipboard()
        assert failures
        xr.testing.assert_identical(dest_tool.slicer_area._data, before)


def test_manager_paste_structured_provenance_steps_into_selected_imagetools(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    operations = (
        AssignAttrsOperation(attrs={"copied": "yes"}),
        AverageOperation(dims=("z",)),
    )
    source_spec = full_data(*operations)
    dest_data_a = _provenance_paste_test_data("dest_a") + 100.0
    dest_data_b = _provenance_paste_test_data("dest_b") + 200.0
    expected_a = source_spec.apply(dest_data_a)
    expected_b = source_spec.apply(dest_data_b)

    with manager_context() as manager:
        tool_a = itool(dest_data_a.copy(deep=True), manager=False, execute=False)
        tool_b = itool(dest_data_b.copy(deep=True), manager=False, execute=False)
        assert isinstance(tool_a, erlab.interactive.imagetool.ImageTool)
        assert isinstance(tool_b, erlab.interactive.imagetool.ImageTool)
        index_a = manager.add_imagetool(
            tool_a,
            show=False,
            provenance_spec=full_data(),
        )
        index_b = manager.add_imagetool(
            tool_b,
            show=False,
            provenance_spec=full_data(),
        )
        _set_provenance_steps_clipboard(operations)

        manager.tree_view.clearSelection()
        select_tools(manager, [index_a, index_b])
        manager._paste_provenance_steps_from_clipboard()

        xr.testing.assert_identical(tool_a.slicer_area._data, expected_a)
        xr.testing.assert_identical(tool_b.slicer_area._data, expected_b)
        assert manager._tool_graph.root_wrappers[index_a].displayed_provenance_spec
        assert manager._tool_graph.root_wrappers[index_b].displayed_provenance_spec


@pytest.mark.parametrize("target_kind", ["root", "source_child"])
def test_manager_paste_steps_preserves_nonuniform_public_dimension(
    target_kind: str,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(12, dtype=float).reshape(3, 4) + 1.0,
        dims=("Track Shift", "eV"),
        coords={
            "Track Shift": [0.0, 1.0, 2.0],
            "eV": [-0.2, -0.1, 0.0, 0.1],
            "sample_temp": ("Track Shift", [20.0, 22.0, 25.0]),
            "mesh_current": ("Track Shift", [1.0, 2.0, 4.0]),
        },
        name="scan",
    )
    operations = (
        SwapDimsOperation(mapping={"Track Shift": "sample_temp"}),
        DivideByCoordOperation(coord_name="mesh_current"),
    )
    expected = full_data(*operations).apply(data)

    with manager_context() as manager:
        if target_kind == "root":
            tool = itool(data.copy(deep=True), manager=False, execute=False)
            assert isinstance(tool, erlab.interactive.imagetool.ImageTool)
            index = manager.add_imagetool(
                tool,
                show=False,
                provenance_spec=full_data(),
            )
            node = manager._tool_graph.root_wrappers[index]
            select_tools(manager, [index])
        else:
            parent_tool = itool(data, manager=False, execute=False)
            assert isinstance(parent_tool, erlab.interactive.imagetool.ImageTool)
            parent_index = manager.add_imagetool(parent_tool, show=False)
            tool = itool(data.copy(deep=True), manager=False, execute=False)
            assert isinstance(tool, erlab.interactive.imagetool.ImageTool)
            child_uid = manager.add_imagetool_child(
                tool,
                parent_index,
                show=False,
                source_spec=full_data(),
                source_auto_update=True,
            )
            node = manager._child_node(child_uid)
            select_child_tool(manager, child_uid)

        _set_provenance_steps_clipboard(operations)
        manager._paste_provenance_steps_from_clipboard()

        xr.testing.assert_identical(tool.slicer_area._data, expected)
        assert tuple(tool.slicer_area.data.dims) == ("sample_temp_idx", "eV")
        xr.testing.assert_identical(tool.slicer_area.displayed_data, expected)
        assert isinstance(node.info_text, str)


@pytest.mark.parametrize(
    "case",
    [
        "nonuniform_attr",
        "nonuniform_transpose",
        "nonuniform_sort",
        "nonuniform_script",
        "promoted_1d",
        "squeezed_5d",
    ],
)
def test_manager_paste_steps_starts_from_public_target_data(
    case: str,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(12, dtype=float).reshape(3, 4) + 1.0,
        dims=("sample_temp", "eV"),
        coords={
            "sample_temp": [20.0, 22.0, 25.0],
            "eV": [-0.2, -0.1, 0.0, 0.1],
        },
        name="scan",
    )
    operations: tuple[ToolProvenanceOperation, ...]
    contains_script = False
    displayed_expected: xr.DataArray
    if case == "nonuniform_attr":
        operations = (AssignAttrsOperation(attrs={"sample": "reference"}),)
        expected = operations[0].apply(data, parent_data=data)
    elif case == "nonuniform_transpose":
        operations = (TransposeOperation(dims=("eV", "sample_temp")),)
        expected = operations[0].apply(data, parent_data=data)
    elif case == "nonuniform_sort":
        data = data.assign_coords(sample_temp=[25.0, 20.0, 22.0])
        operations = (SortByOperation(variables=("sample_temp",)),)
        expected = operations[0].apply(data, parent_data=data)
    elif case == "nonuniform_script":
        operations = (
            ScriptCodeOperation(label="Offset data", code="derived = derived + 1.0"),
        )
        contains_script = True
        expected = data + 1.0
    elif case == "promoted_1d":
        data = data.isel(eV=0, drop=True)
        operations = (AssignAttrsOperation(attrs={"sample": "reference"}),)
        expected = operations[0].apply(data, parent_data=data)
    else:
        data = data.expand_dims(a=[1.0], b=[2.0], c=[3.0])
        operations = (AssignAttrsOperation(attrs={"sample": "reference"}),)
        expected = operations[0].apply(data, parent_data=data)
    displayed_expected = expected.transpose(*data.dims, transpose_coords=True)

    with manager_context() as manager:
        tool = itool(data.copy(deep=True), manager=False, execute=False)
        assert isinstance(tool, erlab.interactive.imagetool.ImageTool)
        index = manager.add_imagetool(
            tool,
            show=False,
            provenance_spec=full_data(),
        )
        node = manager._tool_graph.root_wrappers[index]

        manager._provenance_edit_controller._paste_steps_into_node(
            node,
            operations,
            active_name="derived",
            contains_script=contains_script,
        )

        xr.testing.assert_identical(tool.slicer_area._data, expected)
        xr.testing.assert_identical(tool.slicer_area.displayed_data, displayed_expected)
        xr.testing.assert_identical(node.current_public_data(), expected)
        assert isinstance(node.info_text, str)


def test_manager_paste_structured_provenance_steps_into_pending_memory_imagetool(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = _provenance_paste_test_data("pending_dest").assign_coords(z=np.arange(5))
    operations = (AssignAttrsOperation(attrs={"copied": "yes"}),)
    expected = full_data(*operations).apply(data)

    with manager_context() as manager:
        tool = itool(data.copy(deep=True), manager=False, execute=False)
        assert isinstance(tool, erlab.interactive.imagetool.ImageTool)
        index = manager.add_imagetool(
            tool,
            show=False,
            provenance_spec=full_data(),
        )
        tool.hide()

        workspace_path = tmp_path / "pending-provenance-paste.itws"
        manager._workspace_controller.saving._save_workspace_document(
            workspace_path, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
            workspace_path, replace=True, associate=True, mark_dirty=False, select=False
        )
        node = manager._tool_graph.root_wrappers[index]
        assert node.pending_workspace_memory_payload is not None
        assert node.imagetool is None

        materialize_calls = 0
        materialize = manager._materialize_pending_workspace_payload

        def _record_materialize(
            target: manager_wrapper._ImageToolWrapper
            | manager_wrapper._ManagedWindowNode,
        ) -> bool:
            nonlocal materialize_calls
            materialize_calls += 1
            return materialize(target)

        monkeypatch.setattr(
            manager,
            "_materialize_pending_workspace_payload",
            _record_materialize,
        )

        _set_provenance_steps_clipboard(operations)
        manager.tree_view.clearSelection()
        select_tools(manager, [index])
        manager._update_info()
        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        assert manager._metadata_paste_steps_action in menu.actions()
        assert manager._metadata_paste_steps_action.isEnabled()
        assert materialize_calls == 0
        assert node.pending_workspace_memory_payload is not None

        manager._paste_provenance_steps_from_clipboard()

        assert materialize_calls == 1
        assert node.pending_workspace_memory_payload is None
        assert node.imagetool is not None
        xr.testing.assert_identical(node.slicer_area._data, expected)
        assert node.displayed_provenance_spec is not None


def test_manager_paste_structured_provenance_steps_reports_partial_failures(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    operations = (
        AssignAttrsOperation(attrs={"copied": "yes"}),
        AverageOperation(dims=("z",)),
    )
    source_spec = full_data(*operations)
    good_data = _provenance_paste_test_data("good") + 100.0
    bad_data = xr.DataArray(
        np.arange(12, dtype=float).reshape(3, 4),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0, 3.0]},
        name="bad",
    )
    expected_good = source_spec.apply(good_data)

    with manager_context() as manager:
        good_tool = itool(good_data.copy(deep=True), manager=False, execute=False)
        bad_tool = itool(bad_data.copy(deep=True), manager=False, execute=False)
        assert isinstance(good_tool, erlab.interactive.imagetool.ImageTool)
        assert isinstance(bad_tool, erlab.interactive.imagetool.ImageTool)
        good_index = manager.add_imagetool(
            good_tool,
            show=False,
            provenance_spec=full_data(),
        )
        bad_index = manager.add_imagetool(
            bad_tool,
            show=False,
            provenance_spec=full_data(),
        )
        reports: list[
            tuple[
                int,
                tuple[tuple[str, Exception], ...],
            ]
        ] = []

        def _record_partial(
            pasted_count: int,
            failures: Sequence[tuple[manager_wrapper._ImageToolWrapper, Exception]],
        ) -> None:
            reports.append(
                (
                    pasted_count,
                    tuple((node.uid, exc) for node, exc in failures),
                )
            )

        monkeypatch.setattr(
            manager._provenance_edit_controller,
            "_show_partial_paste_failures",
            _record_partial,
        )
        _set_provenance_steps_clipboard(operations)

        manager.tree_view.clearSelection()
        select_tools(manager, [good_index, bad_index])
        manager._paste_provenance_steps_from_clipboard()

        xr.testing.assert_identical(good_tool.slicer_area._data, expected_good)
        xr.testing.assert_identical(bad_tool.slicer_area._data, bad_data)
        assert len(reports) == 1
        assert reports[0][0] == 1
        assert len(reports[0][1]) == 1
        assert reports[0][1][0][0] == manager._tool_graph.root_wrappers[bad_index].uid


def test_manager_copy_paste_kspace_conversion_steps_remain_group_editable(
    qtbot,
    anglemap,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source_base = anglemap.qsel(eV=-0.1).copy(deep=True)
    source_operations = stamp_operation_group(
        (
            KspaceWorkFunctionOperation(work_function=4.2),
            KspaceSetNormalOperation(alpha=1.0, beta=2.0, delta=3.0),
            KspaceConvertOperation(bounds=None, resolution=None),
        ),
        kind=_kspace_conversion.KSPACE_CONVERSION_GROUP_KIND,
    )
    source_spec = full_data(*source_operations)
    source_data = source_spec.apply(source_base)
    dest_base = source_base.copy(deep=True)
    dest_base.values = dest_base.values + 100.0
    expected = source_spec.apply(dest_base)
    child_base_spec = full_data()

    with manager_context() as manager:
        source_tool = itool(source_data, manager=False, execute=False)
        assert isinstance(source_tool, erlab.interactive.imagetool.ImageTool)
        source_index = manager.add_imagetool(
            source_tool,
            show=False,
            provenance_spec=source_spec,
        )
        parent_tool = itool(dest_base, manager=False, execute=False)
        assert isinstance(parent_tool, erlab.interactive.imagetool.ImageTool)
        parent_index = manager.add_imagetool(parent_tool, show=False)
        child_tool = itool(dest_base, manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            parent_index,
            show=False,
            source_spec=child_base_spec,
        )
        partial_child_tool = itool(dest_base, manager=False, execute=False)
        assert isinstance(partial_child_tool, erlab.interactive.imagetool.ImageTool)
        partial_child_uid = manager.add_imagetool_child(
            partial_child_tool,
            parent_index,
            show=False,
            source_spec=child_base_spec,
        )

        clipboard = QtWidgets.QApplication.clipboard()
        clipboard.clear()
        manager.tree_view.clearSelection()
        select_tools(manager, [source_index])
        manager._update_info()

        select_metadata_rows(manager, [1, 2])
        qtbot.keyClick(
            manager.metadata_derivation_list,
            QtCore.Qt.Key.Key_C,
            QtCore.Qt.KeyboardModifier.ControlModifier,
        )
        partial_payload = manager_details_panel._provenance_step_clipboard_payload(
            clipboard.mimeData()
        )
        assert partial_payload is not None
        partial_operations, _active_name, contains_script = partial_payload
        assert [operation.op for operation in partial_operations] == [
            "kspace_work_function",
            "kspace_set_normal",
        ]
        assert all(operation.group is None for operation in partial_operations)
        assert not contains_script

        manager.tree_view.clearSelection()
        select_child_tool(manager, partial_child_uid)
        manager._update_info(uid=partial_child_uid)
        manager._paste_provenance_steps_from_clipboard()
        partial_node = manager._child_node(partial_child_uid)
        assert partial_node.displayed_source_spec is not None
        assert partial_node.displayed_source_spec.operations == partial_operations
        manager._update_info(uid=partial_child_uid)
        for row_index in (1, 2):
            row_item = manager.metadata_derivation_list.item(row_index)
            assert row_item is not None
            row = row_item.data(manager_details_panel._METADATA_DERIVATION_ROW_ROLE)
            assert isinstance(row, _ProvenanceDisplayRow)
            editable, reason = manager._provenance_edit_controller.can_edit_row(row)
            assert not editable
            assert "complete editable momentum-conversion group" in reason
            assert "`Set normal emission`" in reason
            assert "`Convert to momentum`" in reason

        manager.tree_view.clearSelection()
        select_tools(manager, [source_index])
        manager._update_info()
        select_metadata_rows(manager, [1, 2, 3])
        qtbot.keyClick(
            manager.metadata_derivation_list,
            QtCore.Qt.Key.Key_C,
            QtCore.Qt.KeyboardModifier.ControlModifier,
        )

        payload = manager_details_panel._provenance_step_clipboard_payload(
            clipboard.mimeData()
        )
        assert payload is not None
        payload_operations, _active_name, contains_script = payload
        assert payload_operations == source_operations
        assert not contains_script

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)
        manager._paste_provenance_steps_from_clipboard()

        xr.testing.assert_allclose(child_tool.slicer_area._data, expected)
        child_node = manager._child_node(child_uid)
        assert child_node.displayed_source_spec is not None
        pasted_operations = child_node.displayed_source_spec.operations
        assert strip_operation_groups(pasted_operations) == strip_operation_groups(
            source_operations
        )
        assert operation_group_range(
            pasted_operations,
            0,
            kind=_kspace_conversion.KSPACE_CONVERSION_GROUP_KIND,
        ) == (0, len(pasted_operations))
        source_group_ids = {
            operation.group.id for operation in source_operations if operation.group
        }
        pasted_group_ids = {
            operation.group.id for operation in pasted_operations if operation.group
        }
        assert len(pasted_group_ids) == 1
        assert pasted_group_ids.isdisjoint(source_group_ids)

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)
        for row_index in (1, 2, 3):
            row_item = manager.metadata_derivation_list.item(row_index)
            assert row_item is not None
            row = row_item.data(manager_details_panel._METADATA_DERIVATION_ROW_ROLE)
            assert isinstance(row, _ProvenanceDisplayRow)
            editable, reason = manager._provenance_edit_controller.can_edit_row(row)
            assert editable, reason


def test_manager_paste_structured_steps_preserves_source_binding(
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    parent_data = _provenance_paste_test_data("parent")
    copied_operation = AverageOperation(dims=("z",))
    source_spec = full_data(copied_operation)
    source_data = source_spec.apply(parent_data)
    child_base_spec = full_data(AssignAttrsOperation(attrs={"child": "bound"}))
    child_data = child_base_spec.apply(parent_data)

    with manager_context() as manager:
        parent_tool = itool(parent_data, manager=False, execute=False)
        assert isinstance(parent_tool, erlab.interactive.imagetool.ImageTool)
        parent_index = manager.add_imagetool(parent_tool, show=False)
        child_tool = itool(child_data, manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            parent_index,
            show=False,
            source_spec=child_base_spec,
            source_auto_update=True,
        )
        child_node = manager._child_node(child_uid)
        filter_operation = GaussianFilterOperation(sigma={"z": 1.0})
        child_tool.slicer_area.apply_filter_operation(filter_operation)
        displayed_child_source = child_node.displayed_source_spec
        assert displayed_child_source is not None
        source_tool = itool(source_data, manager=False, execute=False)
        assert isinstance(source_tool, erlab.interactive.imagetool.ImageTool)
        source_index = manager.add_imagetool(
            source_tool,
            show=False,
            provenance_spec=source_spec,
        )

        manager.tree_view.clearSelection()
        select_tools(manager, [source_index])
        manager._update_info()
        select_metadata_rows(manager, [1])
        manager._copy_selected_derivation_code()

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info()
        manager._paste_provenance_steps_from_clipboard()

        child_node = manager._child_node(child_uid)
        assert (
            child_node.parent_uid == manager._tool_graph.root_wrappers[parent_index].uid
        )
        assert child_node.source_auto_update is True
        expected_source_spec = displayed_child_source.append_replacement_operations(
            copied_operation
        )
        assert child_node.source_spec == expected_source_spec
        assert child_tool.slicer_area._accepted_filter_provenance_operation is None
        expected = child_node.source_spec.apply(parent_data)
        xr.testing.assert_identical(child_tool.slicer_area._data, expected)


def test_manager_copy_paste_script_provenance_steps_detaches_and_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = _provenance_paste_test_data("scan")
    child_source_spec = full_data(
        AssignAttrsOperation(attrs={"before": "script paste"})
    )
    child_data = child_source_spec.apply(data)
    script_operations = (
        ScriptCodeOperation(
            label="Offset data",
            code="result = derived + 2.0",
        ),
        AverageOperation(dims=("z",)),
    )
    script_spec = script(
        *script_operations,
        start_label="Run copied script",
        seed_code="derived = data",
        active_name="result",
    )
    script_data = replay_script_provenance(script_spec, {"data": data})

    with manager_context() as manager:
        parent_tool = itool(data, manager=False, execute=False)
        assert isinstance(parent_tool, erlab.interactive.imagetool.ImageTool)
        parent_index = manager.add_imagetool(parent_tool, show=False)
        child_tool = itool(child_data, manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            parent_index,
            show=False,
            source_spec=child_source_spec,
            source_auto_update=True,
        )
        source_tool = itool(script_data, manager=False, execute=False)
        assert isinstance(source_tool, erlab.interactive.imagetool.ImageTool)
        source_index = manager.add_imagetool(
            source_tool,
            show=False,
            provenance_spec=script_spec,
        )

        manager.tree_view.clearSelection()
        select_tools(manager, [source_index])
        manager._update_info()
        select_metadata_rows(manager, [1, 2])
        manager._copy_selected_derivation_code()
        payload = manager_details_panel._provenance_step_clipboard_payload(
            QtWidgets.QApplication.clipboard().mimeData()
        )
        assert payload is not None
        payload_operations, _active_name, contains_script = payload
        assert payload_operations == script_operations
        assert contains_script

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info()
        manager._paste_provenance_steps_from_clipboard()

        child_node = manager._child_node(child_uid)
        expected = replay_script_provenance(
            script_spec,
            {"data": child_data},
        )
        xr.testing.assert_identical(child_tool.slicer_area._data, expected)
        assert child_node.source_spec is None
        assert child_node.source_auto_update is False
        pasted_code = child_node.displayed_provenance_spec.derivation_code()
        assert pasted_code is not None
        replay_namespace = _exec_generated_code(
            pasted_code,
            {"data": data.copy(deep=True)},
        )
        replayed = replay_namespace["result"]
        assert isinstance(replayed, xr.DataArray)
        xr.testing.assert_identical(replayed, expected)
        derivation = metadata_derivation_texts(manager)
        assert script_operations[0].derivation_entry().label in derivation
        assert script_operations[1].derivation_entry().label in derivation
        assert not any("Paste provenance steps" in label for label in derivation)
        assert not any("derived = data" in label for label in derivation)

        bad_operation = ScriptCodeOperation(
            label="Use unavailable scratch value",
            code="result = scratch + 1.0",
        )
        bad_spec = script(
            bad_operation,
            start_label="Run invalid copied script",
            seed_code="derived = data",
            active_name="result",
        )
        bad_tool = itool(data, manager=False, execute=False)
        assert isinstance(bad_tool, erlab.interactive.imagetool.ImageTool)
        bad_index = manager.add_imagetool(
            bad_tool,
            show=False,
            provenance_spec=bad_spec,
        )

        manager.tree_view.clearSelection()
        select_tools(manager, [bad_index])
        manager._update_info()
        select_metadata_rows(manager, [1])
        manager._copy_selected_derivation_code()

        failures: list[tuple[str, Exception, dict[str, str]]] = []
        monkeypatch.setattr(
            manager._provenance_edit_controller,
            "_show_failed",
            lambda title, exc, **kwargs: failures.append((title, exc, kwargs)),
        )
        before = child_tool.slicer_area._data.copy(deep=True)
        before_spec = child_node.provenance_spec
        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info()
        manager._paste_provenance_steps_from_clipboard()

        assert failures
        assert failures[0][0] == "Could Not Paste Provenance Steps"
        assert failures[0][2]["text"] == (
            "The copied provenance steps could not be applied."
        )
        assert (
            "dimensions, coordinates, and inputs" in failures[0][2]["unchanged_reason"]
        )
        assert "Revert to This Step" not in failures[0][2]["unchanged_reason"]
        xr.testing.assert_identical(child_tool.slicer_area._data, before)
        assert child_node.provenance_spec == before_spec
        assert child_node.source_spec is None


def test_manager_paste_script_steps_replays_from_current_output_name(
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = _provenance_paste_test_data("scan")
    dest_base_spec = script(
        ScriptCodeOperation(
            label="Compute destination result",
            code="result = derived + 1.0",
        ),
        start_label="Start from destination data",
        seed_code="derived = data",
        active_name="result",
    )
    dest_data = replay_script_provenance(dest_base_spec, {"data": data})
    copied_operation = ScriptCodeOperation(
        label="Offset copied result",
        code="result = derived + 2.0",
    )
    source_spec = script(
        copied_operation,
        start_label="Run copied script",
        seed_code="derived = data",
        active_name="result",
    )
    source_data = replay_script_provenance(source_spec, {"data": data})

    with manager_context() as manager:
        dest_tool = itool(dest_data, manager=False, execute=False)
        assert isinstance(dest_tool, erlab.interactive.imagetool.ImageTool)
        dest_index = manager.add_imagetool(
            dest_tool,
            show=False,
            provenance_spec=dest_base_spec,
        )
        source_tool = itool(source_data, manager=False, execute=False)
        assert isinstance(source_tool, erlab.interactive.imagetool.ImageTool)
        source_index = manager.add_imagetool(
            source_tool,
            show=False,
            provenance_spec=source_spec,
        )

        manager.tree_view.clearSelection()
        select_tools(manager, [source_index])
        manager._update_info()
        select_metadata_rows(manager, [1])
        manager._copy_selected_derivation_code()

        manager.tree_view.clearSelection()
        select_tools(manager, [dest_index])
        manager._update_info()
        manager._paste_provenance_steps_from_clipboard()

        expected = dest_data + 2.0
        renderer_expected = (
            erlab.interactive.imagetool.slicer.ArraySlicer.validate_array(
                expected,
                copy_values=False,
            )
        )
        xr.testing.assert_identical(dest_tool.slicer_area._data, expected)
        xr.testing.assert_identical(dest_tool.slicer_area.data, renderer_expected)
        dest_node = manager._tool_graph.root_wrappers[dest_index]
        assert dest_node.displayed_provenance_spec is not None
        assert [
            operation.derivation_entry().label
            for operation in dest_node.displayed_provenance_spec.operations
        ] == [
            "Compute destination result",
            "Offset copied result",
        ]
        assert [
            binding.model_dump(mode="json")
            for binding in dest_node.displayed_provenance_spec.script_context_bindings
        ] == [{"operation_index": 1, "names": ["data", "derived"]}]

        code = dest_node.displayed_provenance_spec.derivation_code()
        assert code is not None
        assert "Start from current ImageTool data" not in code
        assert "derived = derived" not in code
        replay_namespace = _exec_generated_code(
            code,
            {"data": data.copy(deep=True)},
        )
        replayed = replay_namespace["result"]
        assert isinstance(replayed, xr.DataArray)
        xr.testing.assert_identical(replayed, expected)


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
            set_transform_launch_mode(dialog, "nest")
            dialog.coord_combo.setCurrentText("mesh_current")

        accept_dialog(parent_tool.mnb._divide_by_coord, pre_call=_nest_divide)

        parent = manager._tool_graph.root_wrappers[0]
        qtbot.wait_until(lambda: len(parent._childtool_indices) == 1, timeout=5000)
        child_uid = parent._childtool_indices[0]
        child_node = manager._child_node(child_uid)
        child_tool = manager.get_imagetool(child_uid)

        expected = (data / data.mesh_current).rename("scan")
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
        trigger_menu_action(menu, manager._metadata_copy_full_action)
        assert not uses_default_replay_input(copied[-1])

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
            set_transform_launch_mode(dialog, "nest")
            dialog._coord_combo.setCurrentText("y")
            dialog.coord_widget.edit_mode_tabs.setCurrentIndex(1)
            dialog.coord_widget.scale_spin.setValue(2.0)
            dialog.coord_widget.offset_spin.setValue(0.5)

        accept_dialog(parent_tool.mnb._assign_coords, pre_call=_nest_affine)

        parent = manager._tool_graph.root_wrappers[0]
        qtbot.wait_until(lambda: len(parent._childtool_indices) == 1, timeout=5000)
        child_uid = parent._childtool_indices[0]
        child_node = manager._child_node(child_uid)
        child_tool = manager.get_imagetool(child_uid)

        operation = AffineCoordOperation(
            coord_name="y",
            scale=2.0,
            offset=0.5,
        )
        expected = operation.apply(data).rename("scan")
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
            operation.apply(updated).rename(None),
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
            set_transform_launch_mode(dialog, "nest")
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

        parent = manager._tool_graph.root_wrappers[0]
        qtbot.wait_until(lambda: len(parent._childtool_indices) == 1, timeout=5000)
        child_uid = parent._childtool_indices[0]
        child_node = manager._child_node(child_uid)
        child_tool = manager.get_imagetool(child_uid)

        operation = AssignAttrsOperation(attrs={"source": "new", "flag": True})
        expected = operation.apply(data).rename("scan")
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
            operation.apply(updated).rename(None),
        )
