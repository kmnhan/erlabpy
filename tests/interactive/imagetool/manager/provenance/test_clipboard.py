import ast
import json
import pathlib
import types
import typing
import uuid
from collections.abc import Callable, Sequence

import numpy as np
import pytest
import xarray as xr
from qtpy import QtCore, QtWidgets

import erlab
import erlab.interactive.imagetool.manager._details_panel as manager_details_panel
import erlab.interactive.imagetool.manager._widgets as manager_widgets
import erlab.interactive.imagetool.manager._wrapper as manager_wrapper
from erlab.interactive.imagetool import _kspace_conversion, itool
from erlab.interactive.imagetool._provenance._execution import (
    replay_file_provenance,
    replay_script_provenance,
)
from erlab.interactive.imagetool._provenance._graph import ReplayGraphError
from erlab.interactive.imagetool._provenance._model import (
    DerivationEntry,
    ReplayStep,
    ScriptInput,
    ToolProvenanceOperation,
    ToolProvenanceSpec,
    _ProvenanceDisplayRow,
    _ProvenanceOperationBlockRef,
    _ProvenanceStepRef,
    full_data,
    operation_group_range,
    script,
    script_input_dependency_refs,
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
    ShiftFromInputOperation,
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
    _authorize_execution,
    _authorize_local_edit,
    _fake_edit_controller,
    _fake_edit_node,
    _manager_replay_file_spec,
    _provenance_paste_test_data,
    _set_provenance_steps_clipboard,
)


@pytest.fixture
def derivation_panel(qtbot, monkeypatch):
    def make_panel(
        *,
        can_edit_row: Callable[[_ProvenanceDisplayRow], tuple[bool, str]] | None = None,
        graph: types.SimpleNamespace | None = None,
        spec: ToolProvenanceSpec | None = None,
    ):
        tree = manager_widgets._MetadataDerivationListWidget()
        qtbot.addWidget(tree)
        manager = types.SimpleNamespace(
            metadata_derivation_list=tree,
            _metadata_node_uid=None,
            _tool_graph=(
                types.SimpleNamespace(
                    nodes={}, root_wrappers={}, structure_generation=0
                )
                if graph is None
                else graph
            ),
            _provenance_edit_controller=types.SimpleNamespace(
                can_edit_row=can_edit_row or (lambda _row: (False, ""))
            ),
            _set_metadata_fields=lambda _fields: None,
            _update_metadata_pane=lambda: None,
            _lineage_controller=types.SimpleNamespace(
                _dependency_refs_for_uid=lambda _uid: (
                    () if spec is None else script_input_dependency_refs(spec)
                ),
                _has_recorded_script_input=lambda _input: False,
            ),
        )
        controller = manager_details_panel._DetailsPanelController(
            typing.cast("typing.Any", manager)
        )
        # Isolate derivation rendering from the separate notes editor.
        monkeypatch.setattr(controller, "_set_notes_node", lambda _node: None)
        return controller, tree

    return make_panel


def test_released_clipboard_keeps_resolved_input_fallback() -> None:
    complete = script(
        start_label="Correction",
        active_name="correction",
        seed_code="correction = xr.DataArray([1.0, 2.0, 3.0], dims='x')",
    )
    source = full_data(IselOperation(kwargs={"x": slice(1, 4)}))
    step = ReplayStep(
        operation=ScriptCodeOperation(
            label="Add correction", code="derived = data + correction"
        )
    )
    mime = QtCore.QMimeData()
    mime.setData(
        manager_details_panel._PROVENANCE_STEPS_CLIPBOARD_MIME,
        json.dumps(
            {
                "type": manager_details_panel._PROVENANCE_STEPS_CLIPBOARD_PAYLOAD_TYPE,
                "version": 2,
                "active_name": "derived",
                "primary_input": "data",
                "steps": [step.model_dump(mode="json")],
                "script_inputs": [
                    {
                        "name": "correction",
                        "source_spec": source.model_dump(mode="json"),
                        "provenance_spec": complete.model_dump(mode="json"),
                    }
                ],
            }
        ).encode(),
    )
    payload = manager_details_panel._provenance_step_clipboard_payload(mime)
    assert payload is not None
    data = xr.DataArray([10.0, 20.0, 30.0], dims="x")
    result = replay_script_provenance(
        payload.spec, {"data": data}, authorize=_authorize_execution
    )
    xr.testing.assert_identical(result, xr.DataArray([11.0, 22.0, 33.0], dims="x"))


@pytest.mark.parametrize("version", [1, 2])
def test_released_clipboard_normalizes_whole_display_operation_sequence(
    version,
) -> None:
    data = xr.DataArray(
        np.arange(12.0).reshape(4, 3),
        dims=("energy", "y"),
        coords={"energy": [0.0, 0.2, 0.8, 1.4], "y": np.arange(3.0)},
    )
    operations = [
        AverageOperation(dims=("energy_idx",)).model_dump(mode="json"),
        {
            "op": "restore_nonuniform_dims",
            "dimension_mapping": {"energy_idx": "energy"},
        },
    ]
    serialized = {
        "type": manager_details_panel._PROVENANCE_STEPS_CLIPBOARD_PAYLOAD_TYPE,
        "version": version,
        "active_name": "derived",
    }
    if version == 1:
        serialized["operations"] = operations
    else:
        serialized.update(
            primary_input="data",
            script_inputs=[],
            steps=[
                {"operation": operation, "input_bindings": {}}
                for operation in operations
            ],
        )
    mime = QtCore.QMimeData()
    mime.setData(
        manager_details_panel._PROVENANCE_STEPS_CLIPBOARD_MIME,
        json.dumps(serialized).encode(),
    )
    payload = manager_details_panel._provenance_step_clipboard_payload(mime)
    assert payload is not None
    assert payload.operations == (AverageOperation(dims=("energy",)),)
    pasted = full_data(*payload.operations)
    code = pasted.display_code(parent_data=data)
    assert code is not None
    assert "energy_idx" not in code
    assert "restore_nonuniform" not in code
    xr.testing.assert_identical(
        _exec_generated_code(code, {"data": data})["derived"], data.qsel.mean("energy")
    )


def test_manager_provenance_step_clipboard_payload_validation() -> None:
    operation_payload = AverageOperation(dims=("z",)).model_dump(mode="json")
    valid_payload = {
        "type": manager_details_panel._PROVENANCE_STEPS_CLIPBOARD_PAYLOAD_TYPE,
        "version": 2,
        "active_name": "derived",
        "primary_input": None,
        "steps": [{"operation": operation_payload, "input_bindings": {}}],
        "script_inputs": [],
        "source_manager_id": None,
    }

    mime_data = QtCore.QMimeData()
    mime_data.setData(
        manager_details_panel._PROVENANCE_STEPS_CLIPBOARD_MIME,
        json.dumps(valid_payload).encode("utf-8"),
    )
    payload = manager_details_panel._provenance_step_clipboard_payload(mime_data)
    assert payload is not None
    assert payload.steps == (ReplayStep(operation=AverageOperation(dims=("z",))),)
    assert payload.active_name == "derived"
    assert payload.primary_input is None
    assert payload.script_inputs == ()
    assert payload.source_manager_id is None

    canonical_payload = {
        "type": manager_details_panel._PROVENANCE_STEPS_CLIPBOARD_PAYLOAD_TYPE,
        "version": manager_details_panel._PROVENANCE_STEPS_CLIPBOARD_PAYLOAD_VERSION,
        "provenance_spec": payload.spec.model_dump(mode="json"),
        "source_manager_id": "source-manager",
    }
    canonical_mime_data = QtCore.QMimeData()
    canonical_mime_data.setData(
        manager_details_panel._PROVENANCE_STEPS_CLIPBOARD_MIME,
        json.dumps(canonical_payload).encode("utf-8"),
    )
    canonical = manager_details_panel._provenance_step_clipboard_payload(
        canonical_mime_data
    )
    assert canonical is not None
    assert canonical.spec == payload.spec
    assert canonical.source_manager_id == "source-manager"

    v1_mime_data = QtCore.QMimeData()
    v1_mime_data.setData(
        manager_details_panel._PROVENANCE_STEPS_CLIPBOARD_MIME,
        json.dumps(
            {
                "type": manager_details_panel._PROVENANCE_STEPS_CLIPBOARD_PAYLOAD_TYPE,
                "version": 1,
                "operations": [operation_payload],
            }
        ).encode("utf-8"),
    )
    v1_payload = manager_details_panel._provenance_step_clipboard_payload(v1_mime_data)
    assert v1_payload is not None
    assert v1_payload.steps == (ReplayStep(operation=AverageOperation(dims=("z",))),)
    assert v1_payload.primary_input is None
    assert v1_payload.script_inputs == ()
    assert v1_payload.source_manager_id is None

    text_mime_data = QtCore.QMimeData()
    text_mime_data.setText(json.dumps({**valid_payload, "active_name": "result"}))
    payload = manager_details_panel._provenance_step_clipboard_payload(text_mime_data)
    assert payload is not None
    assert payload.active_name == "result"

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

    bound_step = ReplayStep(
        operation=ShiftFromInputOperation(along="x"),
        input_bindings={"shift": "shift"},
    ).model_dump(mode="json")
    shift_input = ScriptInput(name="shift").model_dump(mode="json")
    for malformed_payload in (
        [],
        {**valid_payload, "type": "other"},
        {**valid_payload, "version": -1},
        {
            "type": manager_details_panel._PROVENANCE_STEPS_CLIPBOARD_PAYLOAD_TYPE,
            "version": 1,
            "operations": {},
        },
        {**valid_payload, "steps": []},
        {**valid_payload, "steps": {}},
        {**valid_payload, "steps": [{}]},
        {**valid_payload, "steps": [None]},
        {**valid_payload, "steps": [{"operation": {"op": "unknown"}}]},
        {**valid_payload, "script_inputs": {}},
        {**valid_payload, "primary_input": 1},
        {**valid_payload, "source_manager_id": 1},
        {**valid_payload, "active_name": "not a name"},
        {**canonical_payload, "provenance_spec": None},
        {**canonical_payload, "provenance_spec": {}},
        {**canonical_payload, "provenance_spec": {"kind": "unknown"}},
        {
            **canonical_payload,
            "provenance_spec": script(
                start_label="Empty selection", active_name="derived"
            ).model_dump(mode="json"),
        },
        {
            **canonical_payload,
            "provenance_spec": full_data(AverageOperation(dims=("z",))).model_dump(
                mode="json"
            ),
        },
        {**canonical_payload, "source_manager_id": 1},
        {
            **valid_payload,
            "primary_input": "data",
            "steps": [bound_step],
        },
        {
            **valid_payload,
            "primary_input": "shift",
            "steps": [bound_step],
            "script_inputs": [shift_input],
        },
        {
            **valid_payload,
            "primary_input": "data",
            "steps": [bound_step],
            "script_inputs": [shift_input, shift_input],
        },
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
    class _NonLiveOperation(ToolProvenanceOperation):
        live_applicable: typing.ClassVar[bool] = False

    def item(
        row: _ProvenanceDisplayRow,
        *,
        copyable: bool = True,
        code: str | None = "derived = data",
    ) -> QtWidgets.QTreeWidgetItem:
        tree_item = QtWidgets.QTreeWidgetItem()
        tree_item.setData(0, manager_details_panel._METADATA_DERIVATION_ROW_ROLE, row)
        tree_item.setData(
            0,
            manager_details_panel._METADATA_DERIVATION_COPYABLE_ROLE,
            copyable,
        )
        tree_item.setData(0, manager_details_panel._METADATA_DERIVATION_CODE_ROLE, code)
        tree_item.setData(
            0, manager_details_panel._METADATA_DERIVATION_TARGET_ROWS_ROLE, ()
        )
        return tree_item

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
    bound_row_a = _ProvenanceDisplayRow(
        DerivationEntry("bound a", None),
        replay_ref=operation_ref,
    )
    bound_row_b = _ProvenanceDisplayRow(
        DerivationEntry("bound b", None),
        replay_ref=operation_ref,
    )
    bound_step = ReplayStep(
        operation=ShiftFromInputOperation(along="x"),
        input_bindings={"shift": "shift"},
    )

    def bound_spec(shift_uid: str) -> ToolProvenanceSpec:
        return script(
            start_label="Shift data",
            active_name="shifted",
            steps=(bound_step,),
            script_inputs=(
                ScriptInput(name="data"),
                ScriptInput(name="shift", node_uid=shift_uid),
            ),
            primary_input="data",
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
            kind="full",
            _operation_for_ref=lambda _ref: ScriptCodeOperation(
                label="hidden script",
                code="derived = data",
                copyable=False,
            ),
        ),
        non_live_row: types.SimpleNamespace(
            kind="full", _operation_for_ref=lambda _ref: _NonLiveOperation()
        ),
        bound_row_a: bound_spec("shift-a"),
        bound_row_b: bound_spec("shift-b"),
    }
    edit_controller = types.SimpleNamespace(
        _display_spec_for_row=lambda _node, row: row_to_spec.get(row)
    )
    selected_items: list[QtWidgets.QTreeWidgetItem] = []
    manager = types.SimpleNamespace(
        _metadata_node_uid="node",
        _tool_graph=types.SimpleNamespace(nodes={"node": object()}),
        _selected_derivation_items=lambda: selected_items,
        _provenance_edit_controller=edit_controller,
        _manager_record=types.SimpleNamespace(internal_id="manager"),
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

    selected_items = [item(bound_row_a), item(bound_row_b)]
    assert controller._selected_derivation_step_payload() is None

    common_item = item(bound_row_a)
    common_item.setData(
        0,
        manager_details_panel._METADATA_DERIVATION_TARGET_ROWS_ROLE,
        (("node", bound_row_a),),
    )
    selected_items = [common_item]
    manager._metadata_node_uid = None
    payload = controller._selected_derivation_step_payload()
    assert payload is not None
    assert payload.steps == (bound_step,)
    assert payload.primary_input == "data"


def test_manager_metadata_derivation_rows_render_as_tree(derivation_panel) -> None:
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
    controller, derivation_list = derivation_panel(
        can_edit_row=lambda row: (row is child_row, "")
    )
    node = types.SimpleNamespace(
        uid="node",
        tool_window=None,
        displayed_provenance_spec=full_data(),
        passive_displayed_provenance_spec=full_data(),
        metadata_fields=[],
        derivation_display_rows=[parent_row, sibling_row],
        derivation_display_rows_cache_key=("node", None, None),
    )

    controller._set_metadata_node(node)

    parent_item = derivation_list.topLevelItem(0)
    assert parent_item is not None
    assert type(parent_item) is QtWidgets.QTreeWidgetItem
    assert parent_item.text(0) == "Use data_0 from Parent"

    assert parent_item.childCount() == 0
    assert parent_item.childIndicatorPolicy() == (
        QtWidgets.QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator
    )
    derivation_list.expandItem(parent_item)
    assert parent_item.childCount() == 1
    child_item = parent_item.child(0)
    assert child_item is not None
    assert child_item.text(0) == "Offset parent"
    assert (
        child_item.data(0, manager_details_panel._METADATA_DERIVATION_ROW_ROLE)
        is child_row
    )
    sibling_item = derivation_list.topLevelItem(1)
    assert sibling_item is not None
    assert (
        sibling_item.data(0, manager_details_panel._METADATA_DERIVATION_ROW_ROLE)
        is sibling_row
    )
    assert derivation_list.count() == 3
    assert derivation_list.conceptual_count() == 3
    assert derivation_list.item(0) is parent_item
    assert derivation_list.item(1) is child_item
    assert derivation_list.item(2) is sibling_item
    assert derivation_list.item(3) is None
    assert derivation_list.conceptual_item(0) is parent_item
    assert derivation_list.conceptual_item(1) is child_item
    assert derivation_list.conceptual_item(2) is sibling_item
    assert derivation_list.conceptual_item(3) is None
    assert derivation_list.row(parent_item) == 0
    assert derivation_list.row(child_item) == 1
    assert derivation_list.row(sibling_item) == 2
    assert derivation_list.conceptual_row(parent_item) == 0
    assert derivation_list.conceptual_row(child_item) == 1
    assert derivation_list.conceptual_row(sibling_item) == 2
    assert derivation_list.display_order(child_item) == 1
    assert derivation_list.display_order(QtWidgets.QTreeWidgetItem(["orphan"])) == -1
    derivation_list.setUniformItemSizes(True)
    assert derivation_list.uniformRowHeights()


def test_manager_metadata_derivation_tree_populates_only_expanded_branches(
    derivation_panel,
) -> None:
    row = _ProvenanceDisplayRow(DerivationEntry("Leaf", None, False))
    for depth in range(100):
        row = _ProvenanceDisplayRow(
            DerivationEntry(f"Level {depth}", None, False),
            children=(row,),
        )

    editability_checks: list[_ProvenanceDisplayRow] = []

    def _record_editability_check(
        checked_row: _ProvenanceDisplayRow,
    ) -> tuple[bool, str]:
        editability_checks.append(checked_row)
        return False, ""

    controller, derivation_list = derivation_panel(
        can_edit_row=_record_editability_check
    )
    node = types.SimpleNamespace(
        uid="node",
        tool_window=None,
        displayed_provenance_spec=full_data(),
        passive_displayed_provenance_spec=full_data(),
        metadata_fields=[],
        derivation_display_rows=[row],
        derivation_display_rows_cache_key=("node", None, None),
    )

    controller._set_metadata_node(node)

    root_item = derivation_list.topLevelItem(0)
    assert root_item is not None
    assert derivation_list.count() == 1
    assert derivation_list.conceptual_count() == 101
    assert root_item.childCount() == 0
    assert editability_checks == [row]

    derivation_list.expandItem(root_item)

    child_item = root_item.child(0)
    assert child_item is not None
    assert child_item.childCount() == 0
    assert editability_checks == [row, row.children[0]]

    child_item.setSelected(True)
    selected_rows, expanded_rows = controller._metadata_derivation_view_state()
    assert selected_rows == frozenset({row.children[0]})
    assert expanded_rows == frozenset({row})

    controller._set_metadata_node(node)

    restored_root = derivation_list.topLevelItem(0)
    assert restored_root is not None
    assert restored_root.isExpanded()
    restored_child = restored_root.child(0)
    assert restored_child is not None
    assert restored_child.isSelected()


def test_manager_metadata_derivation_state_restore_uses_indexed_lookups(
    derivation_panel,
    monkeypatch,
) -> None:
    rows = tuple(
        _ProvenanceDisplayRow(
            DerivationEntry(f"Parent {index}", None, False),
            replay_ref=_ProvenanceStepRef("operation", operation_index=index),
            children=(
                _ProvenanceDisplayRow(
                    DerivationEntry(f"Child {index}", None, False),
                ),
            ),
        )
        for index in range(96)
    )
    controller, derivation_list = derivation_panel()
    spec = full_data()
    node = types.SimpleNamespace(
        uid="node",
        tool_window=None,
        displayed_provenance_spec=spec,
        passive_displayed_provenance_spec=spec,
        metadata_fields=[],
        derivation_display_rows=rows,
        derivation_display_rows_cache_key=("node", None, None),
    )

    controller._set_metadata_node(node)
    for index in range(derivation_list.topLevelItemCount()):
        item = derivation_list.topLevelItem(index)
        assert item is not None
        derivation_list.expandItem(item)

    comparison_count = 0
    original_eq = _ProvenanceDisplayRow.__eq__

    def _count_row_comparisons(
        left: _ProvenanceDisplayRow,
        right: object,
    ) -> bool:
        nonlocal comparison_count
        comparison_count += 1
        return original_eq(left, right)

    monkeypatch.setattr(_ProvenanceDisplayRow, "__eq__", _count_row_comparisons)

    controller._set_metadata_node(node)

    assert comparison_count < len(rows) * 2
    for index in range(derivation_list.topLevelItemCount()):
        item = derivation_list.topLevelItem(index)
        assert item is not None
        assert item.isExpanded()


def test_manager_metadata_derivation_refresh_uses_stable_cache_key(
    derivation_panel,
) -> None:
    row = _ProvenanceDisplayRow(DerivationEntry("Source", None, False))
    controller, derivation_list = derivation_panel()
    spec = full_data()
    node = types.SimpleNamespace(
        uid="node",
        tool_window=None,
        displayed_provenance_spec=spec,
        passive_displayed_provenance_spec=spec,
        metadata_fields=[],
        derivation_display_rows=(row,),
        derivation_display_rows_cache_key=("snapshot", None, None),
    )

    controller._set_metadata_node(node)
    original_item = derivation_list.topLevelItem(0)
    assert original_item is not None

    equivalent_spec = full_data()
    assert equivalent_spec == spec
    assert equivalent_spec is not spec
    node.passive_displayed_provenance_spec = equivalent_spec
    controller._set_metadata_node(node)

    assert derivation_list.topLevelItem(0) is original_item
    assert controller._metadata_derivation_spec is equivalent_spec

    node.derivation_display_rows_cache_key = ("next-snapshot", None, None)
    controller._set_metadata_node(node)

    assert derivation_list.topLevelItem(0) is not original_item


def test_manager_metadata_row_mapping_keeps_collapsed_rows_consistent(
    derivation_panel,
) -> None:
    materialization_calls = 0

    def _load_children() -> tuple[_ProvenanceDisplayRow, ...]:
        nonlocal materialization_calls
        materialization_calls += 1
        return (_ProvenanceDisplayRow(DerivationEntry("Hidden child", None, False)),)

    parent_row = _ProvenanceDisplayRow(
        DerivationEntry("Parent", None, False),
        _children_factory=_load_children,
    )
    sibling_row = _ProvenanceDisplayRow(DerivationEntry("Sibling", None, False))
    controller, derivation_list = derivation_panel()
    spec = full_data()
    node = types.SimpleNamespace(
        uid="node",
        tool_window=None,
        displayed_provenance_spec=spec,
        passive_displayed_provenance_spec=spec,
        metadata_fields=[],
        derivation_display_rows=[parent_row, sibling_row],
        derivation_display_rows_cache_key=("node", None, None),
    )

    controller._set_metadata_node(node)

    sibling_item = derivation_list.topLevelItem(1)
    assert sibling_item is not None
    assert derivation_list.display_order(sibling_item) == 1
    assert materialization_calls == 0
    sibling_row_index = derivation_list.row(sibling_item)
    assert sibling_row_index == 1
    assert derivation_list.item(sibling_row_index) is sibling_item
    assert materialization_calls == 0

    sibling_conceptual_row = derivation_list.conceptual_row(sibling_item)
    assert sibling_conceptual_row == 2
    assert materialization_calls == 1
    assert parent_row._materialized_children_cache is not None
    parent_item = derivation_list.topLevelItem(0)
    assert parent_item is not None
    assert parent_item.childCount() == 0
    assert derivation_list.conceptual_item(sibling_conceptual_row) is sibling_item
    assert materialization_calls == 1


def test_metadata_derivation_tree_repopulates_with_plain_qt_items(qtbot) -> None:
    derivation_list = manager_widgets._MetadataDerivationListWidget()
    qtbot.addWidget(derivation_list)
    manager = types.SimpleNamespace(
        metadata_derivation_list=derivation_list,
        _provenance_edit_controller=types.SimpleNamespace(
            can_edit_row=lambda _row: (True, "")
        ),
    )
    controller = manager_details_panel._DetailsPanelController(
        typing.cast("typing.Any", manager)
    )
    child_row = _ProvenanceDisplayRow(DerivationEntry("Child", None))
    parent_row = _ProvenanceDisplayRow(
        DerivationEntry("Parent", None), children=(child_row,)
    )
    retained_items: list[QtWidgets.QTreeWidgetItem] = []

    for _generation in range(500):
        derivation_list._clear_contents_safely()
        parent_item = controller._metadata_derivation_item(parent_row)
        child_item = controller._metadata_derivation_item(child_row)
        assert type(parent_item) is QtWidgets.QTreeWidgetItem
        assert type(child_item) is QtWidgets.QTreeWidgetItem
        parent_item.addChild(child_item)
        derivation_list.addItem(parent_item)
        derivation_list.setCurrentItem(child_item)
        child_item.setSelected(True)
        parent_item.setExpanded(True)
        retained_items.extend((parent_item, child_item))

    derivation_list._clear_contents_safely()
    assert derivation_list.topLevelItemCount() == 0
    assert derivation_list.currentItem() is None
    assert derivation_list.selectedItems() == []
    assert len(retained_items) == 1000
    assert all(erlab.interactive.utils.qt_is_valid(item) for item in retained_items)
    assert all(item.parent() is None for item in retained_items)
    assert all(item.treeWidget() is None for item in retained_items)


def test_manager_metadata_script_input_labels_use_current_nodes(
    derivation_panel,
) -> None:
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
    current_path = [(4,)]

    class LabelSource(manager_wrapper._ImageToolWrapper):
        def __init__(self) -> None:
            QtCore.QObject.__init__(self)
            self.uid = "n16"
            self.parent_uid = None
            self._childtool_indices = []

        @property
        def is_imagetool(self) -> bool:
            return True

        @property
        def name(self) -> str:
            return "3 V"

        @property
        def index(self) -> int:
            return current_path[0][0]

    source_node = LabelSource()
    graph = types.SimpleNamespace(
        nodes={"n16": source_node},
        structure_generation=0,
        node_path=lambda _node: current_path[0],
    )
    controller, derivation_list = derivation_panel(graph=graph, spec=spec)
    node = types.SimpleNamespace(
        uid="node",
        tool_window=None,
        displayed_provenance_spec=spec,
        passive_displayed_provenance_spec=spec,
        metadata_fields=[],
        derivation_display_rows=spec.display_rows(),
        derivation_display_rows_cache_key=("node", None, None),
    )

    controller._set_metadata_node(node)

    input_item = derivation_list.topLevelItem(1)
    assert input_item is not None
    assert input_item.text(0) == "Use data_10 from ImageTool 4"
    assert "ImageTool 10" not in input_item.text(0)

    graph.structure_generation += 1
    controller._set_metadata_node(node)
    assert derivation_list.topLevelItem(1) is input_item

    current_path[0] = (7, 1)
    controller._set_metadata_node(node)
    updated_input_item = derivation_list.topLevelItem(1)
    assert updated_input_item is not None
    assert updated_input_item is not input_item
    assert updated_input_item.text(0) == "Use data_10 from ImageTool 7"


def test_manager_metadata_missing_script_input_uses_neutral_label(
    derivation_panel,
) -> None:
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
    controller, derivation_list = derivation_panel(spec=spec)
    node = types.SimpleNamespace(
        uid="node",
        tool_window=None,
        displayed_provenance_spec=spec,
        passive_displayed_provenance_spec=spec,
        metadata_fields=[],
        derivation_display_rows=spec.display_rows(),
        derivation_display_rows_cache_key=("node", None, None),
    )

    controller._set_metadata_node(node)

    input_item = derivation_list.topLevelItem(1)
    assert input_item is not None
    assert input_item.text(0) == "Missing"
    assert "ImageTool 10" not in input_item.text(0)


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
        lambda: manager_details_panel._ProvenanceStepsClipboardPayload(
            spec=script(
                AverageOperation(dims=("x",)),
                start_label="Average data",
                seed_code="derived = data",
                active_name="derived",
            ),
        ),
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
        passive_displayed_provenance_spec=spec,
        derivation_display_rows=spec.display_rows(),
    )
    controller = manager_details_panel._DetailsPanelController(
        typing.cast(
            "typing.Any",
            types.SimpleNamespace(
                _tool_graph=types.SimpleNamespace(nodes={}),
                _lineage_controller=types.SimpleNamespace(
                    _has_recorded_script_input=lambda _input: False
                ),
            ),
        )
    )

    details = controller._unavailable_replay_code_details(
        typing.cast("typing.Any", node)
    )

    assert "Run opaque code" in details
    assert "Missing" not in details


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
        script(
            AverageOperation(dims=("x",)),
            start_label="Copied average",
            active_name="derived",
        ),
    )
    assert unavailable

    node = _fake_edit_node(
        full_data(),
        parent_uid="parent",
        source_display_spec=full_data(),
    )
    parent = _fake_edit_node(full_data(), uid="parent")
    parent.current_source_data = lambda: xr.DataArray([1.0], dims=("x",))
    controller = _fake_edit_controller(node, parent=parent)
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
        script(
            AssignAttrsOperation(attrs={"copied": "yes"}),
            start_label="Copied attribute",
        )
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
        on_commit: Callable[[], None] | None = None,
    ) -> None:
        replaced.append((result, spec, preserve_filter, replay_source_data))
        if on_commit is not None:
            on_commit()

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
        _operation_block_ref=lambda _ref: _ProvenanceOperationBlockRef(0, 1),
        _replace_operation_range_ref=lambda *_args: (_ for _ in ()).throw(
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
    displayed = replay_script_provenance(
        spec, {"data": data}, authorize=_authorize_execution
    )

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
    dest_data = replay_script_provenance(
        dest_spec, {"data": dest_base}, authorize=_authorize_execution
    )

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
        assert payload.operations == source_operations
        assert not payload.contains_script
        assert clipboard.mimeData().hasFormat(
            manager_details_panel._PROVENANCE_STEPS_CLIPBOARD_MIME
        )
        copied_namespace = _exec_generated_code(
            clipboard.text(),
            {"data": dest_data.copy(deep=True)},
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


def test_manager_copied_nonuniform_steps_use_physical_inputs(
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = _provenance_paste_test_data().assign_coords(x=[0.0, 0.2, 1.0])
    operations = (AverageOperation(dims=("z",)),)
    spec = script(
        *operations,
        start_label="Start from notebook data",
        seed_code="derived = data",
        active_name="derived",
    )
    expected = operations[0].apply(data)

    with manager_context() as manager:
        tool = itool(expected, manager=False, execute=False)
        assert isinstance(tool, erlab.interactive.imagetool.ImageTool)
        index = manager.add_imagetool(tool, show=False, provenance_spec=spec)
        select_tools(manager, [index])
        manager._update_info()
        select_metadata_rows(manager, [1])
        manager._copy_selected_derivation_code()

        clipboard = QtWidgets.QApplication.clipboard()
        code = clipboard.text()
        copied = _exec_generated_code(code, {"data": data})
        xr.testing.assert_identical(copied["derived"], expected)
        assert "restore_nonuniform" not in code
        assert "_idx" not in code
        assert ".copy(" not in code
        payload = manager_details_panel._provenance_step_clipboard_payload(
            clipboard.mimeData()
        )
        assert payload is not None
        assert payload.operations == operations
        replay_spec = script(
            start_label="Replay copied steps",
            seed_code="derived = data",
            active_name=payload.active_name,
            steps=payload.steps,
        )
        xr.testing.assert_identical(
            replay_script_provenance(
                replay_spec, {"data": data}, authorize=_authorize_execution
            ),
            expected,
        )


@pytest.mark.parametrize("selected_rows", [(1,), (1, 2)])
def test_manager_copied_partial_script_steps_keep_selected_output(
    selected_rows: tuple[int, ...],
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = _provenance_paste_test_data() - 12.0
    spec = script(
        ScriptCodeOperation(
            label="Offset magnitude", code="derived = np.abs(derived) + 1"
        ),
        AverageOperation(dims=("z",)),
        ScriptCodeOperation(label="Scale result", code="result = derived * 2"),
        start_label="Use notebook data",
        seed_code="derived = data",
        active_name="result",
    )
    with manager_context() as manager:
        tool = itool(data, manager=False, execute=False)
        assert isinstance(tool, erlab.interactive.imagetool.ImageTool)
        index = manager.add_imagetool(tool, show=False, provenance_spec=spec)
        select_tools(manager, [index])
        manager._update_info()
        select_metadata_rows(manager, selected_rows)
        code = manager._selected_derivation_code()
        assert code is not None
        copied = _exec_generated_code(code, {"data": data, "np": np})
        expected = np.abs(data) + 1
        if len(selected_rows) > 1:
            expected = expected.qsel.mean("z")
        xr.testing.assert_identical(copied["derived"], expected)
        assert "result" not in copied
        assert ".copy(" not in code
        assert len(ast.parse(code).body) == 1


@pytest.mark.parametrize("shift_name", ["shift", "offsets", "derived"])
def test_manager_copy_paste_bound_shift_steps_keeps_live_dependency(
    qtbot,
    shift_name: str,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source_data = xr.DataArray(
        np.arange(15.0).reshape(3, 5),
        dims=("x", "eV"),
        coords={"x": np.arange(3), "eV": np.linspace(-0.2, 0.2, 5)},
        name="source",
    )
    shift_data = xr.DataArray(
        [0.0, 0.1, -0.1],
        dims="x",
        coords={"x": np.arange(3)},
        name="shift",
    )
    target_data = source_data + 100.0
    operation = ShiftFromInputOperation(negate=True, along="eV")

    with manager_context() as manager:
        source_tool = itool(source_data, manager=False, execute=False)
        shift_tool = itool(shift_data, manager=False, execute=False)
        assert isinstance(source_tool, erlab.interactive.imagetool.ImageTool)
        assert isinstance(shift_tool, erlab.interactive.imagetool.ImageTool)
        source_index = manager.add_imagetool(source_tool, show=False)
        shift_index = manager.add_imagetool(shift_tool, show=False)
        shift_node = manager._node_for_target(shift_index)
        bound_spec = manager._multi_input_operation_provenance(
            (source_index, shift_index),
            operation=operation,
            active_name="shifted",
            start_label="Shift ImageTool data",
            data_role="source",
            input_names=("data", shift_name),
            primary_input="data",
            input_bindings={"shift": shift_name},
        )
        source_expected = erlab.analysis.transform.shift(
            source_data, -shift_data, along="eV"
        )
        source_child = itool(source_expected, manager=False, execute=False)
        assert isinstance(source_child, erlab.interactive.imagetool.ImageTool)
        source_child_uid = manager.add_imagetool_child(
            source_child,
            source_index,
            show=False,
            provenance_spec=bound_spec,
            source_state="fresh",
        )
        target_tool = itool(target_data, manager=False, execute=False)
        assert isinstance(target_tool, erlab.interactive.imagetool.ImageTool)
        target_index = manager.add_imagetool(target_tool, show=False)
        target_node = manager._node_for_target(target_index)

        manager.tree_view.clearSelection()
        select_child_tool(manager, source_child_uid)
        manager._update_info(uid=source_child_uid)
        operation_rows = [
            index
            for index in range(manager.metadata_derivation_list.conceptual_count())
            if (
                (item := manager.metadata_derivation_list.conceptual_item(index))
                is not None
                and isinstance(
                    row := item.data(
                        0, manager_details_panel._METADATA_DERIVATION_ROW_ROLE
                    ),
                    _ProvenanceDisplayRow,
                )
                and row.replay_ref is not None
                and row.replay_ref.operation_index == 0
            )
        ]
        assert len(operation_rows) == 1
        select_metadata_rows(manager, operation_rows)
        qtbot.keyClick(
            manager.metadata_derivation_list,
            QtCore.Qt.Key.Key_C,
            QtCore.Qt.KeyboardModifier.ControlModifier,
        )
        payload = manager_details_panel._provenance_step_clipboard_payload(
            QtWidgets.QApplication.clipboard().mimeData()
        )
        assert payload is not None
        assert len(payload.steps) == 1
        assert isinstance(payload.steps[0].operation, ShiftFromInputOperation)
        assert payload.steps[0].operation.negate is True
        assert payload.steps[0].operation.along == "eV"
        assert payload.steps[0].input_bindings == {"shift": shift_name}
        assert payload.primary_input == "data"
        assert [item.name for item in payload.script_inputs] == [shift_name]
        assert payload.source_manager_id == manager._manager_record.internal_id
        assert not payload.contains_script
        assert not any(
            isinstance(item, ScriptCodeOperation) for item in payload.operations
        )

        copied = _exec_generated_code(
            QtWidgets.QApplication.clipboard().text(),
            {
                "data": target_data.copy(deep=True),
                shift_name: shift_data.copy(deep=True),
            },
        )
        target_expected = erlab.analysis.transform.shift(
            target_data, -shift_data, along="eV"
        )
        xr.testing.assert_identical(copied["shifted"], target_expected)

        manager.tree_view.clearSelection()
        select_tools(manager, [target_index])
        manager._update_info()
        manager._paste_provenance_steps_from_clipboard()
        xr.testing.assert_identical(target_tool.slicer_area._data, target_expected)
        target_spec = target_node.displayed_provenance_spec
        assert target_spec is not None
        assert target_spec.primary_input == "data"
        assert len(target_spec.steps) == 1
        assert isinstance(target_spec.steps[0].operation, ShiftFromInputOperation)
        assert target_spec.steps[0].operation.negate is True
        assert target_spec.steps[0].operation.along == "eV"
        assert target_spec.steps[0].input_bindings == {"shift": shift_name}
        assert [item.node_uid for item in target_spec.script_inputs] == [
            None,
            shift_node.uid,
        ]
        assert not any(
            isinstance(item, ScriptCodeOperation) for item in target_spec.operations
        )
        assert manager.dependency_status_for_uid(target_node.uid) == "current"

        updated_shift = shift_data + 0.05
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(shift_index, updated_shift)
        qtbot.wait_until(
            lambda: manager.dependency_status_for_uid(target_node.uid) == "changed",
            timeout=5000,
        )
        xr.testing.assert_identical(target_tool.slicer_area._data, target_expected)
        assert target_node.reload_source_data()
        xr.testing.assert_identical(
            target_tool.slicer_area._data,
            erlab.analysis.transform.shift(target_data, -updated_shift, along="eV"),
        )
        assert manager.dependency_status_for_uid(target_node.uid) == "current"


def test_manager_nested_copied_steps_keep_pinned_shift_after_source_closes(
    monkeypatch: pytest.MonkeyPatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = (
        _provenance_paste_test_data("sample")
        .rename(y="eV")
        .assign_coords(x=[0.0, 0.2, 1.0])
    )
    offsets = xr.DataArray(
        [0.0, 0.05, -0.05], dims="x", coords={"x": data.x}, name="offsets"
    )
    primary_spec = script(
        AverageOperation(dims=("z",)),
        start_label="Use sample data",
        seed_code="prepared = sample",
        active_name="prepared",
    )
    auxiliary_spec = script(
        ScriptCodeOperation(
            label="Scale correction", code="correction = correction * 2"
        ),
        start_label="Use measured offsets",
        seed_code="correction = offsets",
        active_name="correction",
    )
    prepared = data.qsel.mean("z")
    correction = offsets * 2

    def fail_paste(_title, error, **_kwargs):
        raise error

    with manager_context() as manager:
        manager._workspace_controller.local_code_edit = _authorize_local_edit
        monkeypatch.setattr(
            manager._provenance_edit_controller,
            "_show_failed",
            fail_paste,
        )
        primary = manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(prepared, _in_manager=True),
            show=False,
            provenance_spec=primary_spec,
        )
        auxiliary = manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(correction, _in_manager=True),
            show=False,
            provenance_spec=auxiliary_spec,
        )
        auxiliary_node = manager._node_for_target(auxiliary)
        bound_spec = manager._multi_input_operation_provenance(
            (primary, auxiliary),
            operation=ShiftFromInputOperation(along="eV", negate=False),
            active_name="shifted",
            start_label="Shift sample data",
            data_role="source",
            input_names=("prepared", "correction"),
            primary_input="prepared",
            input_bindings={"shift": "correction"},
        )
        captured_revision = auxiliary_node.current_revision_id
        assert captured_revision is not None
        bound_spec = bound_spec.model_copy(
            update={
                "script_inputs": tuple(
                    item.model_copy(update={"follow_source": False})
                    if item.name == "correction"
                    else item
                    for item in bound_spec.script_inputs
                )
            }
        )
        expected = erlab.analysis.transform.shift(prepared, correction, along="eV")
        child_uid = manager.add_imagetool_child(
            erlab.interactive.imagetool.ImageTool(expected, _in_manager=True),
            primary,
            show=False,
            provenance_spec=bound_spec,
            source_state="fresh",
        )
        auxiliary_node.replace_with_detached_data(
            correction + 0.5, full_data(), replay_source_data=correction
        )
        assert auxiliary_node.current_revision_id != captured_revision
        manager._remove_imagetool(auxiliary)
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)

        clipboard = QtWidgets.QApplication.clipboard()
        separate_branch_rows = []
        for path, namespace, output_name, result in (
            ((0,), {"sample": data}, "prepared", prepared),
            ((1,), {"offsets": offsets}, "correction", correction),
            ((), {"prepared": prepared, "correction": correction}, "shifted", expected),
        ):
            rows = []
            for index in range(manager.metadata_derivation_list.conceptual_count()):
                item = manager.metadata_derivation_list.conceptual_item(index)
                assert item is not None
                row = item.data(0, manager_details_panel._METADATA_DERIVATION_ROW_ROLE)
                if (
                    isinstance(row, _ProvenanceDisplayRow)
                    and row.script_input_path == path
                    and row.replay_ref is not None
                    and row.replay_ref.kind == "operation"
                ):
                    rows.append(index)
            assert rows
            if path:
                separate_branch_rows.extend(rows)
            select_metadata_rows(manager, rows)
            manager._copy_selected_derivation_code()
            code = clipboard.text()
            copied = _exec_generated_code(code, namespace)
            xr.testing.assert_identical(copied[output_name], result)
            assert "restore_image_tool_dimensions" not in code
            assert ".copy(" not in code
            assert not any(
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and isinstance(statement.value, ast.Name)
                and statement.targets[0].id == statement.value.id
                for statement in ast.walk(ast.parse(code))
            )

        payload = manager_details_panel._provenance_step_clipboard_payload(
            clipboard.mimeData()
        )
        assert payload is not None
        assert payload.primary_input == "prepared"
        assert payload.steps[0].input_bindings == {"shift": "correction"}
        assert payload.script_inputs[0].revision_id == captured_revision
        assert not payload.script_inputs[0].follow_source

        select_metadata_rows(manager, separate_branch_rows)
        separate_code = manager._selected_derivation_code()
        assert separate_code is not None
        separate_results = _exec_generated_code(
            separate_code, {"sample": data, "offsets": offsets}
        )
        xr.testing.assert_identical(separate_results["prepared"], prepared)
        xr.testing.assert_identical(separate_results["correction"], correction)
        assert manager._details_panel._selected_derivation_step_payload() is None

        target_data = prepared + 100.0
        target = manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(target_data, _in_manager=True),
            show=False,
        )
        manager.tree_view.clearSelection()
        select_tools(manager, [target])
        manager._update_info()
        manager._paste_provenance_steps_from_clipboard()
        target_node = manager._node_for_target(target)
        xr.testing.assert_identical(
            target_node.data_for_role("source"),
            erlab.analysis.transform.shift(target_data, correction, along="eV"),
        )
        target_spec = target_node.provenance_spec
        assert target_spec is not None
        target_correction = next(
            item for item in target_spec.script_inputs if item.name == "correction"
        )
        assert target_correction.revision_id == captured_revision
        assert not target_correction.follow_source


@pytest.mark.parametrize(
    ("depth", "active_name", "shift_name", "script_steps", "alias_seed"),
    [
        (1, "shifted", "correction", False, False),
        (2, "derived", "correction", False, False),
        (2, "shifted", "derived", False, False),
        (2, "derived", "correction", True, False),
        (1, "shifted", "correction", False, True),
    ],
)
def test_manager_copied_primary_chain_connects_nested_outputs(
    depth: int,
    active_name: str,
    shift_name: str,
    script_steps: bool,
    alias_seed: bool,
    monkeypatch: pytest.MonkeyPatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = _provenance_paste_test_data("sample").rename(y="eV")
    correction = xr.DataArray(
        [0.0, 0.25, -0.25], dims="x", coords={"x": data.x}, name="offsets"
    )
    average = AverageOperation(dims=("z",))
    average_step: ToolProvenanceOperation = average
    if script_steps:
        average_step = ScriptCodeOperation(
            label="Average sample", code="derived = data.qsel.mean('z')"
        )
    shift = ShiftFromInputOperation(along="eV", negate=False)

    def fail_paste(_title, error, **_kwargs):
        raise error

    with manager_context() as manager:
        manager._workspace_controller.local_code_edit = _authorize_local_edit
        monkeypatch.setattr(
            manager._provenance_edit_controller, "_show_failed", fail_paste
        )
        monkeypatch.setattr(
            manager._provenance_edit_controller,
            "_show_unavailable",
            lambda reason: pytest.fail(reason),
        )
        seed_revision = manager._provenance_revisions.add(data)
        inner = script(
            average_step,
            start_label="Average sample",
            active_name="derived",
            primary_input="data",
            script_inputs=(
                ScriptInput(
                    name="data", revision_id=seed_revision, follow_source=False
                ),
            ),
        )
        prepared = average.apply(data)
        operations: tuple[ToolProvenanceOperation, ...] = (average_step,)
        if depth == 2:
            attributes = AssignAttrsOperation(attrs={"prepared": "yes"})
            attribute_step: ToolProvenanceOperation = attributes
            if script_steps:
                attribute_step = ScriptCodeOperation(
                    label="Mark prepared data",
                    code="derived = data.assign_attrs({'prepared': 'yes'})",
                )
            inner = script(
                attribute_step,
                start_label="Mark prepared data",
                active_name="derived",
                primary_input="data",
                script_inputs=(
                    ScriptInput(
                        name="data",
                        revision_id=manager._provenance_revisions.add(prepared),
                        follow_source=False,
                        provenance_spec=inner.model_dump(mode="json"),
                    ),
                ),
            )
            prepared = attributes.apply(prepared)
            operations += (attribute_step,)
        correction_revision = manager._provenance_revisions.add(correction)
        primary = ScriptInput(
            name="data",
            revision_id=manager._provenance_revisions.add(prepared),
            follow_source=False,
            provenance_spec=inner.model_dump(mode="json"),
        )
        outer_operation: ToolProvenanceOperation = shift
        if alias_seed:
            outer_operation = AssignAttrsOperation(attrs={"shifted": "yes"})
            spec = script(
                outer_operation,
                start_label="Mark prepared data",
                seed_code=f"{active_name} = data",
                active_name=active_name,
                script_inputs=(primary,),
            )
            assert spec.primary_input is None
            expected = outer_operation.apply(prepared)
        else:
            spec = script(
                start_label="Shift prepared data",
                active_name=active_name,
                primary_input="data",
                steps=(
                    ReplayStep(operation=shift, input_bindings={"shift": shift_name}),
                ),
                script_inputs=(
                    primary,
                    ScriptInput(
                        name=shift_name,
                        revision_id=correction_revision,
                        follow_source=False,
                    ),
                ),
            )
            expected = erlab.analysis.transform.shift(prepared, correction, along="eV")
        source = manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(expected, _in_manager=True),
            show=False,
            provenance_spec=spec,
        )
        select_tools(manager, [source])
        manager._update_info()
        rows = []
        for index in range(manager.metadata_derivation_list.conceptual_count()):
            item = manager.metadata_derivation_list.conceptual_item(index)
            assert item is not None
            row = item.data(0, manager_details_panel._METADATA_DERIVATION_ROW_ROLE)
            if (
                isinstance(row, _ProvenanceDisplayRow)
                and row.replay_ref is not None
                and row.replay_ref.kind == "operation"
                and (row.on_primary_chain or alias_seed)
            ):
                rows.append(index)
        assert len(rows) == depth + 1
        select_metadata_rows(manager, rows)
        code = manager._selected_derivation_code()
        assert code is not None
        copied = _exec_generated_code(code, {"data": data, shift_name: correction})
        xr.testing.assert_identical(copied[active_name], expected)
        assert copied[active_name].dims == ("x", "eV")
        manager._copy_selected_derivation_code()
        clipboard = QtWidgets.QApplication.clipboard()
        assert clipboard.text() == code
        payload = manager_details_panel._provenance_step_clipboard_payload(
            clipboard.mimeData()
        )
        assert payload is not None
        assert [operation.model_dump_json() for operation in payload.operations] == [
            operation.model_dump_json() for operation in (*operations, outer_operation)
        ]
        assert payload.active_name == active_name
        selected_spec = payload.spec
        assert [
            operation.model_dump_json() for operation in selected_spec.operations
        ] == [outer_operation.model_dump_json()]
        for operation in reversed(operations):
            primary_input = next(
                item
                for item in selected_spec.script_inputs
                if item.name == selected_spec.primary_input
            )
            assert primary_input.provenance_spec is not None
            selected_spec = ToolProvenanceSpec.model_validate(
                primary_input.provenance_spec
            )
            assert selected_spec.operations == (operation,)
        if alias_seed:
            assert payload.script_inputs == ()
        else:
            bound_name = payload.steps[-1].input_bindings["shift"]
            assert len(payload.script_inputs) == 1
            captured_input = payload.script_inputs[0]
            assert captured_input.name == bound_name
            assert captured_input.revision_id == correction_revision
            assert not captured_input.follow_source

        target_data = data + 100.0
        target = manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(target_data, _in_manager=True),
            show=False,
        )
        manager.tree_view.clearSelection()
        select_tools(manager, [target])
        manager._update_info()
        manager._paste_provenance_steps_from_clipboard()
        target_expected = average.apply(target_data)
        if depth == 2:
            target_expected = attributes.apply(target_expected)
        if alias_seed:
            target_expected = outer_operation.apply(target_expected)
        else:
            target_expected = erlab.analysis.transform.shift(
                target_expected, correction, along="eV"
            )
        xr.testing.assert_identical(
            manager._node_for_target(target).data_for_role("source"), target_expected
        )


@pytest.mark.parametrize("function_scope", [None, "default", "global"])
def test_manager_copied_branches_keep_display_order(
    function_scope: str | None,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = _provenance_paste_test_data("sample").rename(y="eV").qsel.mean("z")
    offsets = xr.DataArray(
        [0.0, 0.25, -0.25], dims="x", coords={"x": data.x}, name="offsets"
    )
    first_code = "events.append('A')\nderived = data + 1"
    last_code = "events.append('B')\nderived = derived * 2 + correction"
    input_name = "derived" if function_scope == "global" else "data"
    expected = (data + (2 if function_scope == "global" else 1)) * 2 + offsets * 2
    if function_scope == "default":
        first_code = (
            "events.append('A')\n"
            "def normalize(values=data):\n"
            "    return values + 1\n"
            "derived = normalize()"
        )
        last_code = "events.append('B')\nderived = normalize() * 2 + correction"
    elif function_scope == "global":
        first_code = (
            "events.append('A')\n"
            "def normalize():\n"
            "    return derived + 1\n"
            "derived = normalize()"
        )
        last_code = "events.append('B')\nderived = normalize() * 2 + correction"
    auxiliary_spec = script(
        ScriptCodeOperation(
            label="Prepare correction",
            code="events.append('X')\ncorrection = correction * 2",
        ),
        start_label="Use measured offsets",
        seed_code="correction = offsets",
        active_name="correction",
    )
    spec = script(
        start_label="Prepare and shift sample",
        active_name="derived",
        primary_input=input_name,
        script_inputs=(
            ScriptInput(name=input_name),
            ScriptInput(
                name="correction",
                provenance_spec=auxiliary_spec.model_dump(mode="json"),
            ),
        ),
        steps=(
            ReplayStep(
                operation=ScriptCodeOperation(
                    label="Offset sample",
                    code=first_code,
                )
            ),
            ReplayStep(
                operation=ShiftFromInputOperation(along="eV", negate=False),
                input_bindings={"shift": "correction"},
            ),
            ReplayStep(
                operation=ScriptCodeOperation(
                    label="Scale sample",
                    code=last_code,
                )
            ),
        ),
    )
    displayed = erlab.analysis.transform.shift(data + 1, offsets * 2, along="eV") * 2
    with manager_context() as manager:
        source = manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(displayed, _in_manager=True),
            show=False,
            provenance_spec=spec,
        )
        select_tools(manager, [source])
        manager._update_info()
        rows = []
        for index in range(manager.metadata_derivation_list.conceptual_count()):
            item = manager.metadata_derivation_list.conceptual_item(index)
            assert item is not None
            row = item.data(0, manager_details_panel._METADATA_DERIVATION_ROW_ROLE)
            if (
                isinstance(row, _ProvenanceDisplayRow)
                and row.replay_ref is not None
                and row.replay_ref.kind == "operation"
                and (
                    row.script_input_path == (1,)
                    or row.replay_ref.operation_index in (0, 2)
                )
            ):
                rows.append(index)
        select_metadata_rows(manager, rows)
        assert len(manager._details_panel._selected_derivation_steps()) == 3
        code = manager._selected_derivation_code()
        assert code is not None
        events: list[str] = []
        copied = _exec_generated_code(
            code, {input_name: data, "offsets": offsets, "events": events}
        )
        assert events == ["A", "X", "B"]
        xr.testing.assert_identical(copied["derived"], expected)
        xr.testing.assert_identical(copied["correction"], offsets * 2)
        assert manager._details_panel._selected_derivation_step_payload() is None


def test_manager_copied_nested_scopes_preserve_outputs_across_generated_aliases(
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source = xr.DataArray(np.full((3, 2), 10.0), dims=("x", "eV"))
    offsets = xr.DataArray([3.0, 4.0, 5.0], dims="x")
    inner = script(
        ScriptCodeOperation(label="Prepare offsets", code="derived = offsets * 2"),
        start_label="Use offsets",
        active_name="derived",
        primary_input="offsets",
        script_inputs=(ScriptInput(name="offsets"),),
    )
    auxiliary = script(
        ScriptCodeOperation(
            label="Prepare correction",
            code="def f():\n    return data * 10\ncorrection = f()",
        ),
        start_label="Use prepared offsets",
        active_name="correction",
        primary_input="data",
        script_inputs=(
            ScriptInput(name="data", provenance_spec=inner.model_dump(mode="json")),
        ),
    )
    spec = script(
        start_label="Prepare and shift sample",
        active_name="data",
        primary_input="source",
        script_inputs=(
            ScriptInput(name="source"),
            ScriptInput(
                name="correction", provenance_spec=auxiliary.model_dump(mode="json")
            ),
        ),
        steps=(
            ReplayStep(
                operation=ScriptCodeOperation(
                    label="Offset sample", code="data = source + 1"
                )
            ),
            ReplayStep(
                operation=ShiftFromInputOperation(along="eV"),
                input_bindings={"shift": "correction"},
            ),
            ReplayStep(
                operation=ScriptCodeOperation(
                    label="Add correction", code="data = data + correction"
                )
            ),
        ),
    )
    with manager_context() as manager:
        source_index = manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(source, _in_manager=True),
            show=False,
            provenance_spec=spec,
        )
        select_tools(manager, [source_index])
        manager._update_info()
        rows = []
        for index in range(manager.metadata_derivation_list.conceptual_count()):
            item = manager.metadata_derivation_list.conceptual_item(index)
            assert item is not None
            row = item.data(0, manager_details_panel._METADATA_DERIVATION_ROW_ROLE)
            if (
                isinstance(row, _ProvenanceDisplayRow)
                and row.replay_ref is not None
                and row.replay_ref.kind == "operation"
                and not (
                    row.script_input_path == () and row.replay_ref.operation_index == 1
                )
            ):
                rows.append(index)
        select_metadata_rows(manager, rows)
        assert len(manager._details_panel._selected_derivation_steps()) == 4
        code = manager._selected_derivation_code()
        assert code is not None
        namespace = _exec_generated_code(code, {"source": source, "offsets": offsets})

        xr.testing.assert_identical(namespace["data"], source + 1 + offsets * 20)
        xr.testing.assert_identical(namespace["correction"], offsets * 20)
        assert ".copy(" not in code
        assert "_itool_replay_" not in code


def test_manager_copied_nested_scopes_publish_swapped_inputs_together(
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source = xr.DataArray(np.full((3, 2), 10.0), dims=("x", "eV"))
    offsets = xr.DataArray(np.full((3, 2), 3.0), dims=("x", "eV"))
    sample = script(
        ScriptCodeOperation(label="Prepare sample", code="correction = source + 1"),
        start_label="Use sample",
        active_name="correction",
        primary_input="source",
        script_inputs=(ScriptInput(name="source"),),
    )
    correction = script(
        ScriptCodeOperation(label="Prepare correction", code="data = offsets * 2"),
        start_label="Use offsets",
        active_name="data",
        primary_input="offsets",
        script_inputs=(ScriptInput(name="offsets"),),
    )
    spec = script(
        ScriptCodeOperation(
            label="Subtract correction",
            code="def subtract():\n    return data - correction\nresult = subtract()",
        ),
        start_label="Correct sample",
        active_name="result",
        primary_input="data",
        script_inputs=(
            ScriptInput(name="data", provenance_spec=sample.model_dump(mode="json")),
            ScriptInput(
                name="correction", provenance_spec=correction.model_dump(mode="json")
            ),
        ),
    )
    with manager_context() as manager:
        index = manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(source, _in_manager=True),
            show=False,
            provenance_spec=spec,
        )
        select_tools(manager, [index])
        manager._update_info()
        rows = []
        for row_index in range(manager.metadata_derivation_list.conceptual_count()):
            item = manager.metadata_derivation_list.conceptual_item(row_index)
            assert item is not None
            row = item.data(0, manager_details_panel._METADATA_DERIVATION_ROW_ROLE)
            if (
                isinstance(row, _ProvenanceDisplayRow)
                and row.replay_ref is not None
                and row.replay_ref.kind == "operation"
            ):
                rows.append(row_index)
        assert len(rows) == 3
        select_metadata_rows(manager, rows)
        code = manager._selected_derivation_code()
        assert code is not None
        namespace = _exec_generated_code(code, {"source": source, "offsets": offsets})

        xr.testing.assert_identical(namespace["result"], source + 1 - offsets * 2)
        assert ".copy(" not in code
        assert "_itool_replay_" not in code


def test_manager_copied_bound_operation_runs_selected_input_history_first(
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source = xr.DataArray(
        np.arange(15.0).reshape(3, 5),
        dims=("x", "eV"),
        coords={"x": np.arange(3), "eV": np.linspace(-0.2, 0.2, 5)},
    )
    offsets = xr.DataArray([0.0, 0.02, -0.02], dims="x", coords={"x": source.x})
    correction = script(
        ScriptCodeOperation(label="Prepare offsets", code="correction = offsets * 2"),
        start_label="Use offsets",
        active_name="correction",
        primary_input="offsets",
        script_inputs=(ScriptInput(name="offsets"),),
    )
    spec = script(
        start_label="Shift sample",
        active_name="shifted",
        primary_input="data",
        script_inputs=(
            ScriptInput(name="data"),
            ScriptInput(
                name="correction", provenance_spec=correction.model_dump(mode="json")
            ),
        ),
        steps=(
            ReplayStep(
                operation=ShiftFromInputOperation(along="eV"),
                input_bindings={"shift": "correction"},
            ),
        ),
    )
    with manager_context() as manager:
        index = manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(source, _in_manager=True),
            show=False,
            provenance_spec=spec,
        )
        select_tools(manager, [index])
        manager._update_info()
        rows = []
        for row_index in range(manager.metadata_derivation_list.conceptual_count()):
            item = manager.metadata_derivation_list.conceptual_item(row_index)
            assert item is not None
            row = item.data(0, manager_details_panel._METADATA_DERIVATION_ROW_ROLE)
            if (
                isinstance(row, _ProvenanceDisplayRow)
                and row.replay_ref is not None
                and row.replay_ref.kind == "operation"
            ):
                rows.append(row_index)
        assert len(rows) == 2
        select_metadata_rows(manager, rows)
        code = manager._selected_derivation_code()
        assert code is not None
        namespace = _exec_generated_code(code, {"data": source, "offsets": offsets})

        xr.testing.assert_identical(
            namespace["shifted"],
            erlab.analysis.transform.shift(source, offsets * 2, along="eV"),
        )
        assert ".copy(" not in code


def test_manager_copied_auxiliary_precedes_reused_bound_input(
    monkeypatch: pytest.MonkeyPatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = _provenance_paste_test_data("sample").rename(y="eV").qsel.mean("z")
    offsets = xr.DataArray(
        [0.0, 0.25, -0.25], dims="x", coords={"x": data.x}, name="offsets"
    )
    auxiliary = script(
        ScriptCodeOperation(
            label="Prepare correction",
            code="events.append('X')\ncorrection = correction * 2",
        ),
        start_label="Use measured offsets",
        seed_code="correction = offsets",
        active_name="correction",
    )
    shift = ReplayStep(
        operation=ShiftFromInputOperation(along="eV", negate=False),
        input_bindings={"shift": "correction"},
    )
    spec = script(
        start_label="Prepare and shift sample",
        active_name="derived",
        primary_input="data",
        script_inputs=(
            ScriptInput(name="data"),
            ScriptInput(
                name="correction", provenance_spec=auxiliary.model_dump(mode="json")
            ),
        ),
        steps=(
            ReplayStep(
                operation=ScriptCodeOperation(
                    label="Offset sample", code="events.append('A')\nderived = data + 1"
                )
            ),
            shift,
            shift,
        ),
    )
    expected = erlab.analysis.transform.shift(data + 1, offsets * 2, along="eV")
    with manager_context() as manager:
        source = manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(expected, _in_manager=True),
            show=False,
            provenance_spec=spec,
        )
        select_tools(manager, [source])
        manager._update_info()
        rows = []
        auxiliary_selected = False
        for index in range(manager.metadata_derivation_list.conceptual_count()):
            item = manager.metadata_derivation_list.conceptual_item(index)
            assert item is not None
            row = item.data(0, manager_details_panel._METADATA_DERIVATION_ROW_ROLE)
            if not isinstance(row, _ProvenanceDisplayRow) or row.replay_ref is None:
                continue
            if row.replay_ref.kind != "operation":
                continue
            if not row.script_input_path and row.replay_ref.operation_index in (0, 2):
                rows.append(index)
            elif row.script_input_path == (1,) and not auxiliary_selected:
                rows.append(index)
                auxiliary_selected = True
        assert len(rows) == 3
        select_metadata_rows(manager, rows)
        code = manager._selected_derivation_code()
        assert code is not None
        events: list[str] = []
        original_shift = erlab.analysis.transform.shift

        def observed_shift(*args, **kwargs):
            events.append("B")
            return original_shift(*args, **kwargs)

        monkeypatch.setattr(erlab.analysis.transform, "shift", observed_shift)
        copied = _exec_generated_code(
            code, {"data": data, "offsets": offsets, "events": events}
        )

        xr.testing.assert_identical(copied["derived"], expected)
        assert events == ["A", "X", "B"]
        assert ".copy(" not in code
        assert manager._details_panel._selected_derivation_step_payload() is None


def test_manager_paste_foreign_bound_shift_uses_recorded_fallback(
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(15.0).reshape(3, 5),
        dims=("x", "eV"),
        coords={"x": np.arange(3), "eV": np.linspace(-0.2, 0.2, 5)},
    )
    shift_data = xr.DataArray([0.0, 0.1, -0.1], dims="x", coords={"x": np.arange(3)})
    shift_path = tmp_path / "shift.nc"
    shift_data.to_netcdf(shift_path, engine="h5netcdf")
    step = ReplayStep(
        operation=ShiftFromInputOperation(negate=True, along="eV"),
        input_bindings={"shift": "shift"},
    )
    portable_shift = ScriptInput(
        name="shift",
        label="Shift ImageTool",
        node_uid="foreign-shift",
        node_snapshot_token=uuid.uuid4().hex,
        provenance_spec=_manager_replay_file_spec(shift_path),
    )

    with manager_context() as manager:
        manager._workspace_controller.local_code_edit = _authorize_local_edit
        unrelated_revision = manager._provenance_revisions.add(shift_data + 100)
        portable_shift = portable_shift.model_copy(
            update={"revision_id": unrelated_revision}
        )
        tool = itool(data, manager=False, execute=False)
        assert isinstance(tool, erlab.interactive.imagetool.ImageTool)
        index = manager.add_imagetool(tool, show=False)
        node = manager._node_for_target(index)

        manager._provenance_edit_controller._paste_steps_into_node(
            node,
            script(
                start_label="Copied shift",
                steps=(step,),
                active_name="shifted",
                primary_input="data",
                script_inputs=(ScriptInput(name="data"), portable_shift),
            ),
            source_manager_id="another-manager",
        )

        expected = erlab.analysis.transform.shift(data, -shift_data, along="eV")
        xr.testing.assert_identical(tool.slicer_area._data, expected)
        spec = node.provenance_spec
        assert spec is not None
        assert isinstance(spec.steps[0].operation, ShiftFromInputOperation)
        assert not any(
            isinstance(operation, ScriptCodeOperation) for operation in spec.operations
        )
        shift_input = next(item for item in spec.script_inputs if item.name == "shift")
        assert shift_input.node_uid is None
        assert shift_input.node_snapshot_token is None
        assert shift_input.parsed_provenance_spec() == _manager_replay_file_spec(
            shift_path
        )

        before_data = tool.slicer_area._data.copy(deep=True)
        before_spec = node.provenance_spec
        with pytest.raises(ReplayGraphError, match="no portable recorded provenance"):
            manager._provenance_edit_controller._paste_steps_into_node(
                node,
                script(
                    start_label="Copied shift",
                    steps=(step,),
                    active_name="shifted",
                    primary_input="data",
                    script_inputs=(
                        ScriptInput(name="data"),
                        ScriptInput(
                            name="shift",
                            label="Unavailable shift",
                            node_uid="foreign-shift",
                        ),
                    ),
                ),
                source_manager_id="another-manager",
            )
        assert node.provenance_spec is before_spec
        xr.testing.assert_identical(tool.slicer_area._data, before_data)


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


def test_manager_pasted_scalar_coordinate_offset_uses_each_destination_value(
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    operation = AffineCoordOperation(
        coord_name="eV",
        scale=1.0,
        offset=0.0,
        offset_coord="hv",
        offset_coord_sign=-1,
    )
    kinetic_energy = np.linspace(20.0, 23.0, 4)
    data_a = xr.DataArray(
        np.arange(4, dtype=float),
        dims="eV",
        coords={"eV": kinetic_energy, "hv": 21.2},
    )
    data_b = data_a.assign_coords(hv=40.0)

    with manager_context() as manager:
        tool_a = itool(data_a.copy(deep=True), manager=False, execute=False)
        tool_b = itool(data_b.copy(deep=True), manager=False, execute=False)
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
        _set_provenance_steps_clipboard((operation,))

        select_tools(manager, [index_a, index_b])
        manager._paste_provenance_steps_from_clipboard()

        np.testing.assert_allclose(
            tool_a.slicer_area._data.eV.values,
            kinetic_energy - data_a.hv.item(),
        )
        np.testing.assert_allclose(
            tool_b.slicer_area._data.eV.values,
            kinetic_energy - data_b.hv.item(),
        )
        assert (
            manager._tool_graph.root_wrappers[
                index_a
            ].displayed_provenance_spec.operations[-1]
            == operation
        )
        assert (
            manager._tool_graph.root_wrappers[
                index_b
            ].displayed_provenance_spec.operations[-1]
            == operation
        )


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
    displayed_expected: xr.DataArray
    if case == "nonuniform_attr":
        operations = (AssignAttrsOperation(attrs={"sample": "reference"}),)
        expected = operations[0].apply(data)
    elif case == "nonuniform_transpose":
        operations = (TransposeOperation(dims=("eV", "sample_temp")),)
        expected = operations[0].apply(data)
    elif case == "nonuniform_sort":
        data = data.assign_coords(sample_temp=[25.0, 20.0, 22.0])
        operations = (SortByOperation(variables=("sample_temp",)),)
        expected = operations[0].apply(data)
    elif case == "nonuniform_script":
        operations = (
            ScriptCodeOperation(label="Offset data", code="derived = derived + 1.0"),
        )
        expected = data + 1.0
    elif case == "promoted_1d":
        data = data.isel(eV=0, drop=True)
        operations = (AssignAttrsOperation(attrs={"sample": "reference"}),)
        expected = operations[0].apply(data)
    else:
        data = data.expand_dims(a=[1.0], b=[2.0], c=[3.0])
        operations = (AssignAttrsOperation(attrs={"sample": "reference"}),)
        expected = operations[0].apply(data)
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
            script(
                *operations,
                start_label="Copied operations",
                active_name="derived",
            ),
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
        manager._workspace_controller.saving._save_workspace_document(workspace_path)
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
        partial_operations = partial_payload.operations
        assert [operation.op for operation in partial_operations] == [
            "kspace_work_function",
            "kspace_set_normal",
        ]
        assert all(operation.group is None for operation in partial_operations)
        assert not partial_payload.contains_script

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
            row = row_item.data(0, manager_details_panel._METADATA_DERIVATION_ROW_ROLE)
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
        assert payload.operations == source_operations
        assert not payload.contains_script

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
            row = row_item.data(0, manager_details_panel._METADATA_DERIVATION_ROW_ROLE)
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
    script_data = replay_script_provenance(
        script_spec, {"data": data}, authorize=_authorize_execution
    )

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
        assert payload.operations == script_operations
        assert payload.contains_script
        copied = _exec_generated_code(
            QtWidgets.QApplication.clipboard().text(), {"data": data}
        )
        xr.testing.assert_identical(copied["result"], script_data)

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info()
        manager._paste_provenance_steps_from_clipboard()

        child_node = manager._child_node(child_uid)
        expected = replay_script_provenance(
            script_spec,
            {"data": child_data},
            authorize=_authorize_execution,
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
    monkeypatch: pytest.MonkeyPatch,
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
    dest_data = replay_script_provenance(
        dest_base_spec, {"data": data}, authorize=_authorize_execution
    )
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
    source_data = replay_script_provenance(
        source_spec, {"data": data}, authorize=_authorize_execution
    )

    with manager_context() as manager:

        def fail_paste(_title, error, **_kwargs):
            raise error

        monkeypatch.setattr(
            manager._provenance_edit_controller, "_show_failed", fail_paste
        )
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
        pasted_spec = dest_node.displayed_provenance_spec
        assert pasted_spec.operations == (copied_operation,)
        primary = next(
            item
            for item in pasted_spec.script_inputs
            if item.name == pasted_spec.primary_input
        )
        assert primary.provenance_spec is not None
        destination_spec = ToolProvenanceSpec.model_validate(primary.provenance_spec)
        assert destination_spec.operations == dest_base_spec.operations
        assert [
            binding.model_dump(mode="json")
            for binding in pasted_spec.script_context_bindings
        ] == [{"operation_index": 0, "names": ["derived"]}]

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
