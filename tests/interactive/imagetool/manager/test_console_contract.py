"""Observable console requirements from the provenance redesign behavior matrix."""

import functools
import gc
import inspect
import types
import weakref

import numpy as np
import pytest
import xarray as xr

import erlab
import erlab.interactive.imagetool.manager._console as manager_console
from erlab.interactive.imagetool import itool
from erlab.interactive.imagetool._provenance._model import ScriptInput
from erlab.interactive.imagetool._provenance._program import RecordedProgram
from erlab.interactive.imagetool.manager import fetch
from erlab.interactive.imagetool.manager._data_graph import (
    _CalculatedResult,
    _OperationRequest,
)

from .helpers import (
    _exec_generated_code,
    select_child_tool,
    select_metadata_items,
    select_tools,
)


@pytest.fixture
def console(manager_context):
    data = xr.DataArray(
        np.arange(12.0).reshape(3, 4),
        dims=("x", "y"),
        coords={"x": np.arange(3.0), "y": np.arange(4.0), "gain": ("x", np.ones(3))},
        attrs={"sample": {"name": "original"}},
    )
    with manager_context() as manager:
        manager.add_imagetool(itool(data, manager=False, execute=False), show=False)
        yield manager, manager_console.ToolsNamespace(manager), data


def _assert_replay(manager, tools, index, expected, monkeypatch, *, output=None):
    namespace = {}
    copied = []

    def checkpoint_name(key, _suggestion, _used_names):
        name = f"input_{len(namespace)}"
        namespace[name] = tools.checkpoint(key)
        return name

    monkeypatch.setattr(manager._details_panel, "_checkpoint_name", checkpoint_name)
    monkeypatch.setattr(erlab.interactive.utils, "copy_to_clipboard", copied.append)
    node = manager._node_for_target(index)
    manager._set_metadata_node(node)
    manager._copy_full_derivation_code()
    assert copied
    result = _exec_generated_code(copied[-1], namespace)
    if output is None and node.provenance_spec is not None:
        output = node.provenance_spec.active_name
    xr.testing.assert_identical(result[output or "derived"], expected)


@pytest.mark.parametrize("derived", [False, True])
@pytest.mark.parametrize(
    "expression",
    [
        "h.coords['gain'].values[:] = 7",
        "h.attrs['sample']['name'] = 'edited'",
        "h.encoding['edited'] = True",
        "h.values[:] = 7",
        "h.variable[0, 0] = 7",
        "h.loc[dict(x=0)] = 7",
        "h.attrs = {}",
        "h.values = np.zeros((3, 4))",
        "np.asarray(h)",
        "h.to_numpy()",
    ],
)
def test_console_contract_rejects_implicit_mutable_exports(
    console, derived, expression
):
    manager, tools, data = console
    handle = tools[0] * 2 if derived else tools[0]
    expected = handle.data
    graph = manager._data_graph.graph
    revision = graph.current_revision(tools[0].uid)
    with pytest.raises((TypeError, AttributeError), match=r"\.data"):
        _exec_generated_code(expression, {"h": handle})
    xr.testing.assert_identical(handle.data, expected)
    xr.testing.assert_identical(fetch(0), data)
    assert graph.current_revision(tools[0].uid) == revision


def test_console_library_aliases_preserve_installed_modules(console):
    _manager, tools, _data = console
    original_concat = xr.concat
    original_rotate = erlab.analysis.transform.rotate
    resolved = manager_console._resolve_console_namespace(
        {
            "xr": "xarray",
            "np": np,
            "era": "erlab.analysis",
            "erlab": erlab,
            "tools": tools,
        }
    )
    assert isinstance(resolved["xr"], types.ModuleType)
    assert resolved["xr"].DataArray is xr.DataArray
    assert resolved["xr"].set_options is xr.set_options
    assert inspect.unwrap(resolved["xr"].concat) is original_concat
    assert xr.concat is original_concat
    assert resolved["np"] is np
    assert inspect.unwrap(resolved["era"].transform.rotate) is original_rotate
    assert erlab.analysis.transform.rotate is original_rotate
    assert resolved["erlab"].constants is erlab.constants
    assert resolved["era"].transform.rotate is resolved["era"].transform.rotate
    assert inspect.signature(resolved["xr"].concat) == inspect.signature(xr.concat)
    assert "concat" in dir(resolved["xr"])
    assert resolved["tools"] is tools


def test_console_library_aliases_keep_plain_calls_and_explicit_apply(console):
    _manager, tools, data = console
    namespace = manager_console._resolve_console_namespace({"xr": xr})
    concat = namespace["xr"].concat
    plain = concat([data, data * 2], dim="scan")
    assert isinstance(plain, xr.DataArray)
    expected = xr.concat([data, data * 2], dim="scan")
    xr.testing.assert_identical(plain, expected)
    recorded = tools.apply(concat, [tools[0], tools[0] * 2], dim="scan")
    xr.testing.assert_identical(recorded.data, expected)
    detached = manager_console._DerivedDataNamespace(
        None, data, "data", (), copyable=False
    )
    with pytest.raises(TypeError, match=r"\.data"):
        concat([detached, detached], dim="scan")


def test_console_library_aliases_reject_unsupported_tracked_calls(console, monkeypatch):
    _manager, tools, data = console
    calls = []

    def unrecorded(array):
        calls.append(array)
        array[:] = 0
        return array

    unrecorded.__module__ = erlab.analysis.transform.__name__
    monkeypatch.setattr(
        erlab.analysis.transform, "unrecorded", unrecorded, raising=False
    )
    namespace = manager_console._resolve_console_namespace({"era": erlab.analysis})
    with pytest.raises(TypeError, match=r"\.data"):
        namespace["era"].transform.unrecorded(tools[0])
    assert not calls
    xr.testing.assert_identical(fetch(0), data)
    snapshot = tools[0].data
    namespace["era"].transform.unrecorded(snapshot)
    assert calls == [snapshot]
    xr.testing.assert_identical(snapshot, xr.zeros_like(data))
    xr.testing.assert_identical(fetch(0), data)


@pytest.mark.parametrize("derived_input", [False, True])
def test_concat_slots_and_editors_use_the_recorded_application(
    console, monkeypatch, derived_input
):
    from erlab.interactive.imagetool._provenance._model import full_data
    from erlab.interactive.imagetool._provenance._operations import ConcatOperation

    manager, tools, a = console
    b = a + 3
    manager.add_imagetool(itool(b, manager=False, execute=False), show=False)
    source = tools[0] * 2 if derived_input else tools[0]
    alias = manager_console._resolve_console_namespace({"xr": xr})["xr"]
    alias.concat([source, tools[1], source], dim="scan").qshow(manager=True)
    result = manager._node_for_target(2)
    expected = xr.concat(
        [a * 2 if derived_input else a, b, a * 2 if derived_input else a], dim="scan"
    )
    graph = manager._data_graph.graph
    revision = graph.current_revision(result.uid)
    applied = graph.revision(revision).producer
    assert isinstance(applied.operation, ConcatOperation)
    assert [slot for slot, _ in applied.inputs] == ["data", "other_0", "other_1"]
    assert applied.inputs[0][1] == applied.inputs[2][1]
    assert applied.inputs[1][1] == graph.current_revision(tools[1].uid)
    for index in (0, 2):
        xr.testing.assert_identical(
            graph.data(applied.inputs[index][1]), expected.isel(scan=index, drop=True)
        )
    child_uid = manager.import_imagetool_child(
        itool(expected, manager=False, execute=False),
        result.uid,
        source_spec=full_data(),
        show=False,
    )
    # A stale interchange payload must have no influence on a live result.
    result._provenance_spec = None
    result._source_spec = None
    result.imagetool.set_provenance_spec(None)
    inputs, *rows = result.derivation_display_rows
    assert [row.revision_id for row in inputs.children] == [
        key for _, key in applied.inputs
    ]
    assert rows[-1].graph_revision == revision
    child = manager._child_node(child_uid)
    historical = child.derivation_display_rows[0].children[0].materialized_children()[0]
    assert historical.graph_revision == revision
    controller = manager._provenance_edit_controller
    assert controller._can_edit_row(result, rows[-1])[0]
    assert controller._can_edit_row(child, historical)[0]
    _assert_replay(manager, tools, 2, expected, monkeypatch)
    tools[1].data = b + 7
    assert graph.is_stale(result.uid)
    _assert_replay(manager, tools, 2, expected, monkeypatch)
    assert manager._data_graph.run(result.uid, "refresh")
    xr.testing.assert_identical(fetch(2).isel(scan=1, drop=True), b + 7)


@pytest.mark.parametrize("response", ["defaults", "collision", "cancel"])
def test_copy_code_two_revisions_of_one_source(
    console, monkeypatch, accept_dialog, response
):
    from qtpy import QtWidgets

    manager, tools, data = console
    original = data.rename("sample")
    tools[0].data = original
    graph = manager._data_graph.graph
    old_revision = graph.current_revision(tools[0].uid)
    old_result = tools[0] * 2
    tools[0].data = original + 10
    new_revision = graph.current_revision(tools[0].uid)
    alias = manager_console._resolve_console_namespace({"xr": xr})["xr"]
    alias.concat([old_result, tools[0]], dim="scan").qshow(manager=True)
    node = manager._node_for_target(1)
    prompts = []
    namespace = {}

    def answer(dialog):
        (revision,) = (
            key for key in (old_revision, new_revision) if key in dialog.labelText()
        )
        suggested = dialog.textValue()
        prompts.append(revision)
        if len(prompts) == 2 and response != "defaults":
            # Python treats the full-width spelling as the same notebook name.
            dialog.setTextValue("ｓａｍｐｌｅ")  # noqa: RUF001
        else:
            assert suggested not in namespace
            namespace[suggested] = tools.checkpoint(revision)

    copied = []
    monkeypatch.setattr(erlab.interactive.utils, "copy_to_clipboard", copied.append)
    manager._set_metadata_node(node)
    count = 2 if response == "defaults" else 3
    accept_dialog(
        manager._copy_full_derivation_code,
        pre_call=[answer] * count,
        accept_call=[None] * (count - 1)
        + [QtWidgets.QDialog.reject if response == "cancel" else None],
        chained_dialogs=count,
    )
    assert set(prompts) == {old_revision, new_revision}
    if response != "defaults":
        assert len(prompts) == 3
        assert prompts[1] == prompts[2]
    if response == "cancel":
        assert not copied
    else:
        assert "sample" in namespace
        (code,) = copied
        assert node.derivation_code == code
        xr.testing.assert_identical(
            _exec_generated_code(code, namespace)["derived"], tools[1].data
        )


def test_copy_imported_step_with_an_unavailable_input(console, monkeypatch):
    from qtpy import QtWidgets

    from erlab.interactive.imagetool._provenance._operations import ScriptCodeOperation
    from erlab.interactive.imagetool.provenance import ScriptInput, script

    manager, tools, data = console
    clipboard = QtWidgets.QApplication.clipboard()
    monkeypatch.setattr(erlab.interactive.utils, "copy_to_clipboard", clipboard.setText)
    program = script(
        ScriptCodeOperation(label="Add one", code="derived = original + 1"),
        start_label="Imported history",
        active_name="derived",
        primary_input="original",
        script_inputs=(ScriptInput(name="original", node_uid="missing"),),
    )
    index = manager.import_imagetool(
        itool(data + 1, manager=False, execute=False),
        show=False,
        provenance_spec=program,
    )
    (tools[index] * 2).qshow(manager=True)
    manager._set_metadata_node(manager._node_for_target(2))
    tree = manager.metadata_derivation_list
    inputs = tree.topLevelItem(0)
    inputs.setExpanded(True)
    source = inputs.child(0)
    source.setExpanded(True)
    imported_step = source.child(0)
    for items, expected in (
        ([imported_step], data + 1),
        ([imported_step, tree.topLevelItem(1)], (data + 1) * 2),
    ):
        select_metadata_items(manager, items)
        assert manager._details_panel._build_metadata_derivation_menu() is not None
        assert manager._metadata_copy_selected_action.isEnabled()
        manager._metadata_copy_selected_action.trigger()
        code = clipboard.text()
        xr.testing.assert_identical(
            _exec_generated_code(code, {"original": data})["derived"], expected
        )


@pytest.mark.parametrize("action", ["copy", "delete", "revert"])
def test_graph_program_actions_address_the_selected_member(
    console, monkeypatch, action
):
    from erlab.interactive.imagetool._provenance._graph import selected_provenance_code
    from erlab.interactive.imagetool._provenance._model import script
    from erlab.interactive.imagetool._provenance._operations import ScriptCodeOperation
    from erlab.interactive.imagetool._provenance._trust import (
        provenance_code_trust_entries,
    )

    manager, _tools, data = console
    input_node = manager._node_for_target(0)
    origin = manager._lineage_controller._script_input_for_node(
        input_node, name="source"
    )
    program = script(
        ScriptCodeOperation(label="Add one", code="value = value + 1"),
        ScriptCodeOperation(label="Double", code="value = value * 2"),
        start_label="Start from source",
        seed_code="value = source",
        active_name="value",
        script_inputs=(origin,),
    )
    controller = manager._workspace_controller
    with controller.local_provenance_change(
        edited_entries=provenance_code_trust_entries(
            program, location_prefix="workspace/provenance"
        )
    ):
        index = manager.add_imagetool(
            manager._data_graph.evaluate_request(
                _OperationRequest(
                    (
                        RecordedProgram(
                            spec=program.with_replay_inputs(
                                (ScriptInput(name="source"),)
                            )
                        ),
                    ),
                    (manager._data_graph.read_input(0).revision,),
                ),
            ),
            show=False,
        )
    node = manager._node_for_target(index)
    select_tools(manager, [index])
    manager._update_info()
    first, second = [
        row for row in node.derivation_display_rows if row.program_index is not None
    ]
    assert first.graph_revision == second.graph_revision
    adapter = manager._data_graph
    if action == "copy":
        scopes = adapter.selected_scopes(
            node.uid, {second.graph_revision: {second.program_index}}
        )
        ns = _exec_generated_code(selected_provenance_code(scopes), {"value": data})
        xr.testing.assert_identical(ns["value"], data * 2)
    elif action == "delete":
        manager._provenance_edit_controller.delete_row(first)
        xr.testing.assert_identical(node.slicer_area._data, data * 2)
    else:
        monkeypatch.setattr(
            manager._provenance_edit_controller, "_confirm_revert", lambda: True
        )
        manager._provenance_edit_controller.revert_row(first)
        xr.testing.assert_identical(node.slicer_area._data, data + 1)
    assert adapter.graph.follows(node.uid) == {input_node.uid}


def test_metadata_edit_reuses_assignment_inside_recorded_program(console, monkeypatch):
    from erlab.interactive.imagetool._provenance._model import script
    from erlab.interactive.imagetool._provenance._operations import (
        AssignAttrsOperation,
        ScriptCodeOperation,
    )
    from erlab.interactive.imagetool._provenance._trust import (
        provenance_code_trust_entries,
    )
    from erlab.interactive.imagetool.manager._metadata_editor import (
        MetadataCellEdit,
        MetadataField,
    )

    manager, tools, data = console
    source = manager._node_for_target(0)
    program = script(
        ScriptCodeOperation(label="Add one", code="value = value + 1"),
        AssignAttrsOperation(attrs={"run": 1, "station": "A"}),
        start_label="Start from source",
        seed_code="value = source",
        active_name="value",
        script_inputs=(
            manager._lineage_controller._script_input_for_node(source, name="source"),
        ),
    )
    with manager._workspace_controller.local_provenance_change(
        edited_entries=provenance_code_trust_entries(
            program, location_prefix="workspace/provenance"
        )
    ):
        index = manager.add_imagetool(
            manager._data_graph.evaluate_request(
                _OperationRequest(
                    (
                        RecordedProgram(
                            spec=program.with_replay_inputs(
                                (ScriptInput(name="source"),)
                            )
                        ),
                    ),
                    (manager._data_graph.read_input(0).revision,),
                ),
            ),
            show=False,
        )
    node = manager._node_for_target(index)
    source.slicer_area.replace_source_data(data + 10)
    field = MetadataField(kind="attribute", name="run")
    editor = manager._metadata_editor
    graph = manager._data_graph.graph
    assert editor.apply_edits({index: (MetadataCellEdit(field, value=2),)})
    record = graph.revision(graph.current_revision(node.uid))
    assert record.producer.operation.spec.steps[-1].operation.attrs == {
        "run": 2,
        "station": "A",
    }
    assert editor.apply_edits({index: (MetadataCellEdit(field, revert=True),)})
    expected = (data + 1).assign_attrs(station="A")
    xr.testing.assert_identical(node.current_public_data(), expected)
    _assert_replay(manager, tools, index, expected, monkeypatch)
    assert graph.is_stale(node.uid)
    graph.refresh(node.uid)
    xr.testing.assert_identical(
        node.current_public_data(), (data + 11).assign_attrs(station="A")
    )


def test_plain_replacement_remains_a_graph_checkpoint(console):
    from erlab.interactive.imagetool._provenance._model import full_data
    from erlab.interactive.imagetool._provenance._operations import IselOperation

    manager, tools, data = console
    tools[0].isel(x=slice(1, 3)).qshow(manager=True)
    tools[1].data = data
    node = manager._node_for_target(1)
    node._provenance_spec = full_data(IselOperation(kwargs={"x": 1}))
    assert manager._data_graph.owns_result(node.uid)
    assert node.provenance_spec is None
    assert all(row.edit_ref is None for row in node.derivation_display_rows)
    assert manager._data_graph.graph.follows(node.uid) == set()
    assert not node.slicer_area.reloadable


def test_delete_matching_graph_steps_uses_each_selected_result(console, monkeypatch):
    manager, tools, data = console
    tools.itool(data + 10)
    for index in (0, 1):
        tools[index].isel(x=slice(1, 3)).qshow(manager=True)
    nodes = [manager._node_for_target(index) for index in (2, 3)]
    for node in nodes:
        node._provenance_spec = None
        node.imagetool.set_provenance_spec(None)
    select_tools(manager, [2, 3])
    manager._update_info()
    controller = manager._provenance_edit_controller
    rows = [(node.uid, node.derivation_display_rows[-1]) for node in nodes]
    assert controller.can_delete_rows(rows)[0]
    monkeypatch.setattr(controller, "_confirm_delete_multiple", lambda: True)
    controller.delete_rows(rows)
    graph = manager._data_graph.graph
    for index, node in enumerate(nodes):
        xr.testing.assert_identical(
            node.slicer_area._data, data + 10 if index else data
        )
        assert graph.follows(node.uid) == {tools[index].uid}


def test_graph_filter_editor_updates_view_and_deletion_restores_source(
    console, accept_dialog
):
    from erlab.interactive.imagetool._provenance._operations import NormalizeOperation

    manager, tools, data = console
    tools[0].isel(x=slice(None)).qshow(manager=True)
    node = manager._node_for_target(1)
    area = node.slicer_area
    area.apply_filter_operation(
        NormalizeOperation(dims=("x",), mode="area"), emit_edited=True
    )
    select_tools(manager, [1])
    manager._update_info()
    controller = manager._provenance_edit_controller

    def change(dialog):
        dialog.dim_checks["x"].setChecked(False)
        dialog.dim_checks["y"].setChecked(True)
        dialog.opts[2].setChecked(True)

    accept_dialog(
        lambda: controller.edit_row(node.derivation_display_rows[-1]), pre_call=change
    )
    expected = NormalizeOperation(dims=("y",), mode="min")
    assert (
        area._accepted_filter_provenance_operation
        == expected.capture_execution_options()
    )
    xr.testing.assert_identical(area.displayed_data, expected.apply(data))
    xr.testing.assert_identical(area._data, data)
    controller.delete_row(node.derivation_display_rows[-1])
    assert area._accepted_filter_provenance_operation is None
    xr.testing.assert_identical(area.displayed_data, data)
    assert manager._data_graph.graph.entry(node.uid).source_revision is None


def test_unchanged_recorded_selection_preserves_live_parent(console, accept_dialog):
    from erlab.interactive.imagetool._provenance._model import full_data

    manager, tools, data = console
    result = tools[0].isel(y=slice(1, 4), x=slice(0, 2))
    result.qshow(manager=True)
    parent = manager._node_for_target(1)
    child_uid = manager.import_imagetool_child(
        itool(result.data, manager=False, execute=False),
        parent.uid,
        source_spec=full_data(),
        show=False,
    )
    child = manager._child_node(child_uid)
    manager.tree_view.clearSelection()
    select_child_tool(manager, child.uid)
    manager._update_info(uid=child.uid)
    row = child.derivation_display_rows[0].children[0].materialized_children()[0]
    graph = manager._data_graph.graph
    previous = graph.current_revision(child.uid)
    accept_dialog(lambda: manager._provenance_edit_controller.edit_row(row))
    assert graph.current_revision(child.uid) == previous
    assert graph.follows(child.uid) == {parent.uid}
    tools[1].data = data + 5
    assert graph.is_stale(child.uid)
    assert manager._data_graph.run(child.uid, "refresh")
    xr.testing.assert_identical(child.slicer_area._data, data + 5)


def test_edit_historical_bound_shift_uses_recorded_inputs(
    console, accept_dialog, monkeypatch
):
    from erlab.interactive.imagetool._provenance._model import full_data

    manager, tools, data = console

    def fail(_title, error, **_kwargs):
        raise error

    monkeypatch.setattr(manager._provenance_edit_controller, "_show_failed", fail)
    monkeypatch.setattr(
        manager._provenance_edit_controller, "_show_unavailable", pytest.fail
    )
    shift = xr.DataArray([0.0, 0.25, -0.25], dims="x", coords={"x": data.x})
    for array in (shift, shift * 2):
        manager.add_imagetool(itool(array, manager=False, execute=False), show=False)
    tools.apply(erlab.analysis.transform.shift, tools[0], tools[1], "y").qshow(
        manager=True
    )
    parent = manager._node_for_target(3)
    child_uid = manager.import_imagetool_child(
        itool(parent.slicer_area._data, manager=False, execute=False),
        parent.uid,
        source_spec=full_data(),
        show=False,
    )
    child = manager._child_node(child_uid)
    manager.tree_view.clearSelection()
    select_child_tool(manager, child.uid)
    manager._update_info(uid=child.uid)
    row = child.derivation_display_rows[0].children[0].materialized_children()[0]
    graph = manager._data_graph.graph
    revision = graph.current_revision(child.uid)
    assert row.graph_revision == graph.current_revision(parent.uid)

    def inspect(dialog):
        assert dialog.shift_tool_selector.current_uid == tools[1].uid

    accept_dialog(
        lambda: manager._provenance_edit_controller.edit_row(row), pre_call=inspect
    )
    assert graph.current_revision(child.uid) == revision

    def rebind(dialog):
        inspect(dialog)
        dialog.shift_tool_selector.set_current_uid(tools[2].uid)

    accept_dialog(
        lambda: manager._provenance_edit_controller.edit_row(row), pre_call=rebind
    )
    expected = erlab.analysis.transform.shift(data, shift * 2, "y")
    xr.testing.assert_identical(child.slicer_area._data, expected)
    assert graph.follows(child.uid) == {tools[0].uid, tools[2].uid}
    assert graph.current_revision(parent.uid) == row.graph_revision
    _assert_replay(manager, tools, child.uid, expected, monkeypatch)
    tools[2].data = shift * 3
    assert graph.is_stale(child.uid)
    assert manager._data_graph.run(child.uid, "refresh")
    xr.testing.assert_identical(
        child.slicer_area._data, erlab.analysis.transform.shift(data, shift * 3, "y")
    )


@pytest.mark.parametrize("call", ["concat", "rotate", "rotate_keyword", "erlab_rotate"])
def test_console_direct_library_calls_keep_recorded_inputs(console, monkeypatch, call):
    manager, tools, data = console
    namespace = manager_console._resolve_console_namespace(
        {"tools": tools, "xr": xr, "era": erlab.analysis, "erlab": erlab}
    )
    expressions = {
        "concat": 'xr.concat([tools[0], tools[0] * 2], dim="scan")',
        "rotate": 'era.transform.rotate(tools[0], 5.0, axes=("x", "y"), reshape=False)',
        "rotate_keyword": (
            "era.transform.rotate(darr=tools[0], angle=5.0, "
            'axes=("x", "y"), reshape=False)'
        ),
        "erlab_rotate": (
            "erlab.analysis.transform.rotate(tools[0], 5.0, "
            'axes=("x", "y"), reshape=False)'
        ),
    }
    result = _exec_generated_code(f"result = {expressions[call]}", namespace)["result"]
    expected = (
        xr.concat([data, data * 2], dim="scan")
        if call == "concat"
        else erlab.analysis.transform.rotate(data, 5.0, axes=("x", "y"), reshape=False)
    )
    tools[0].data = data + 100
    result.qshow(manager=True)
    xr.testing.assert_identical(fetch(1), expected)
    _assert_replay(manager, tools, 1, expected, monkeypatch)
    manager._data_graph.graph.refresh(tools[1].uid)
    expected_next = (
        xr.concat([data + 100, (data + 100) * 2], dim="scan")
        if call == "concat"
        else erlab.analysis.transform.rotate(
            data + 100, 5.0, axes=("x", "y"), reshape=False
        )
    )
    xr.testing.assert_identical(fetch(1), expected_next)
    _assert_replay(manager, tools, 1, expected_next, monkeypatch)


@pytest.mark.parametrize(
    "expression",
    [
        "h.pipe(callback)",
        "h.assign_coords(gain=callback)",
        "h.where(callback)",
        "h.coarsen(y=2, coord_func={'gain': callback}).mean()",
        "h.coarsen(y=2).reduce(callback)",
        "h.kspace.offsets.update({'delta': 7})",
    ],
)
def test_console_contract_rejects_opaque_callbacks(console, expression):
    _manager, tools, data = console
    calls = []

    def callback(value, *args, **kwargs):
        calls.append(value)
        return value

    with pytest.raises((TypeError, AttributeError), match=r"\.data"):
        _exec_generated_code(expression, {"h": tools[0], "callback": callback})
    assert not calls
    xr.testing.assert_identical(fetch(0), data)


@pytest.mark.parametrize("method", ["out", "at", "reduce", "where"])
def test_console_contract_rejects_ufunc_mutation(console, method):
    _manager, tools, data = console
    handle = tools[0]
    out = np.full(data.shape, -1.0)

    def call():
        if method == "out":
            np.add(handle, 1, out=out)
        elif method == "at":
            np.add.at(handle, (0, 0), 1)
        elif method == "reduce":
            np.add.reduce(handle)
        else:
            np.add(handle, 1, where=False)

    with pytest.raises(TypeError, match=r"\.data"):
        call()
    np.testing.assert_array_equal(out, -1)
    xr.testing.assert_identical(fetch(0), data)


def test_console_contract_explicit_registered_calls(console, monkeypatch):
    manager, tools, data = console
    result = tools.apply(xr.concat, [tools[0], tools[0] * 2], dim="state")
    result.qshow(manager=True)
    expected = xr.concat([data, data * 2], dim="state")
    xr.testing.assert_identical(fetch(1), expected)
    _assert_replay(manager, tools, 1, expected, monkeypatch)
    calls = []

    def unknown(value):
        calls.append(value)
        return value

    with pytest.raises(TypeError, match=r"\.data"):
        tools.apply(unknown, tools[0])
    assert not calls
    xr.testing.assert_identical(fetch(0), data)


@pytest.mark.parametrize("derived", [False, True])
@pytest.mark.parametrize("entrypoint", ["qshow", "itool"])
@pytest.mark.parametrize("target", [1, -1, [1]])
def test_console_contract_publication_replaces_target(
    console, monkeypatch, derived, entrypoint, target
):
    manager, tools, data = console
    manager.add_imagetool(itool(data + 50, manager=False, execute=False), show=False)
    uid = tools[1].uid
    window = tools[1].tool
    tools._show_handle(tools[1] * 3, active_name="derived", label="Earlier result")
    source = tools[0] * 2 if derived else tools[0]
    expected = data * 2 if derived else data
    # Resolve the negative index before adding the follower above.
    replacement = -2 if target == -1 else target
    if entrypoint == "qshow":
        source.qshow(manager=True, replace=replacement, execute=False)
    else:
        tools.itool(source, manager=True, replace=replacement, execute=False)
    assert manager.ntools == 3
    assert tools[1].uid == uid
    assert tools[1].tool is window
    xr.testing.assert_identical(fetch(1), expected)
    assert manager._node_for_target(2).source_state == "stale"
    _assert_replay(manager, tools, 1, expected, monkeypatch)
    _assert_replay(manager, tools, 2, (data + 50) * 3, monkeypatch)


def test_console_contract_augmented_assignment(console, monkeypatch):
    manager, tools, data = console
    handle = tools[0]
    tools._show_handle(handle * 3, active_name="derived", label="Earlier result")
    handle.data += 10
    xr.testing.assert_identical(fetch(0), data + 10)
    assert not manager._data_graph.graph.has_recipe(handle.uid)
    assert manager._node_for_target(1).source_state == "stale"
    _assert_replay(manager, tools, 1, data * 3, monkeypatch)
    handle += 2
    assert isinstance(handle, manager_console._DerivedDataNamespace)
    xr.testing.assert_identical(handle.data, data + 12)
    xr.testing.assert_identical(fetch(0), data + 10)


@pytest.mark.parametrize("failure", ["prepare", "install"])
@pytest.mark.parametrize("edit", ["snapshot", "item", "replacement"])
def test_console_contract_failed_edit_rolls_back(console, monkeypatch, failure, edit):
    manager, tools, data = console
    tools._show_handle(tools[0] * 3, active_name="derived", label="Follower")
    handle = tools[0]
    area = handle.tool.slicer_area
    graph = manager._data_graph.graph
    revision = graph.current_revision(handle.uid)
    state = area.state
    if failure == "install":
        method = "_install_edited_values" if edit == "item" else "_set_data"
        original = getattr(area, method)
        failed = False

        def fail_once(*args, **kwargs):
            nonlocal failed
            result = original(*args, **kwargs)
            if not failed:
                failed = True
                raise RuntimeError("installation failed")
            return result

        monkeypatch.setattr(area, method, fail_once)

    def change():
        if edit == "snapshot":
            handle.data = xr.DataArray([]) if failure == "prepare" else data + 100
        elif edit == "item":
            handle[100 if failure == "prepare" else 0, 0] = 100
        else:
            result = handle.isel(x=slice(0)) if failure == "prepare" else handle + 100
            result.qshow(manager=True, replace=0)

    with pytest.raises((RuntimeError, IndexError, ValueError)):
        change()
    xr.testing.assert_identical(fetch(0), data)
    assert area.state == state
    assert graph.current_revision(handle.uid) == revision
    assert manager._node_for_target(1).source_state == "fresh"


@pytest.mark.parametrize("auto", [False, True])
def test_console_contract_refresh_keeps_chosen_python_branch(
    console, monkeypatch, qtbot, auto
):
    manager, tools, data = console
    tools[0].data = -data - 1

    def scale(handle):
        factor = 2 if handle.mean().item() > 0 else 3
        return handle * factor

    result = scale(tools[0])
    result.qshow(manager=True)
    old = (-data - 1) * 3
    xr.testing.assert_identical(fetch(1), old)
    graph = manager._data_graph.graph
    graph.set_auto(tools[1].uid, enabled=auto)
    tools[0].data = data + 1
    xr.testing.assert_identical(fetch(1), old)
    assert manager._node_for_target(1).source_state == "stale"
    _assert_replay(manager, tools, 1, old, monkeypatch)
    if auto:
        qtbot.wait_until(lambda: not graph.is_stale(tools[1].uid))
    else:
        graph.refresh(tools[1].uid)
    xr.testing.assert_identical(fetch(1), (data + 1) * 3)
    _assert_replay(manager, tools, 1, (data + 1) * 3, monkeypatch)


def test_console_contract_named_results_retain_inputs_without_original(
    console, monkeypatch
):
    manager, tools, data = console
    first = tools[0] * 2
    first._set_console_name("first")
    first_ref = weakref.ref(first)
    revision = first._calculation_revision
    result = first + 1
    del first
    gc.collect()
    assert first_ref() is None
    assert manager._data_graph.graph.has_revision(revision)
    tools[0].data = data + 100
    result.qshow(manager=True)
    xr.testing.assert_identical(fetch(1), data * 2 + 1)
    _assert_replay(manager, tools, 1, data * 2 + 1, monkeypatch)


def test_console_contract_retained_result_does_not_keep_manager(manager_context):
    data = xr.DataArray(np.arange(12.0).reshape(3, 4), dims=("x", "y"))
    with manager_context() as manager:
        manager.add_imagetool(itool(data, manager=False, execute=False), show=False)
        tools = manager_console.ToolsNamespace(manager)
        result = tools[0] + 1
        result._set_console_name("first")
        result = result * 2
        manager_ref = weakref.ref(manager)
    del manager, tools
    gc.collect()
    assert manager_ref() is None
    xr.testing.assert_identical(result.data, (data + 1) * 2)


@pytest.mark.parametrize("direct", [True, False])
def test_console_handle_retains_graph_dispatch_without_namespace(
    console, monkeypatch, direct
):
    manager, tools, data = console
    handle = (
        manager_console.ToolNamespace(manager._node_for_target(0))
        if direct
        else manager_console.ToolsNamespace(manager)[0]
    )
    result = (handle + 1).isel(x=slice(1, None))
    gc.collect()
    (result * 2).qshow(manager=True)
    expected = (data + 1).isel(x=slice(1, None)) * 2
    _assert_replay(manager, tools, 1, expected, monkeypatch)
    tools[0].data = data + 10
    manager._data_graph.graph.refresh(tools[1].uid)
    xr.testing.assert_identical(tools[1].data, (data + 11).isel(x=slice(1, None)) * 2)


def test_console_contract_closed_handle_is_unavailable(console):
    manager, tools, data = console
    handle = tools[0]
    selected = handle.isel
    result = handle + 1
    manager.remove_imagetool(0)
    with pytest.raises(LookupError, match=r"closed|available|destroyed"):
        selected(x=0)
    xr.testing.assert_identical(result.data, data + 1)


def test_console_contract_imported_helper_keeps_argument_identity(console):
    _manager, tools, data = console
    module = types.ModuleType("console_contract_helper")
    module.__dict__.update(
        _exec_generated_code(
            "def edit(first, *, second):\n"
            "    assert first is second\n"
            "    first[0, 0] = 100\n"
            "    return second + 1\n",
            module.__dict__,
        )
    )
    handle = tools[0]
    result = module.edit(handle, second=handle)
    expected = data.copy(deep=True)
    expected[0, 0] = 100
    xr.testing.assert_identical(fetch(0), expected)
    xr.testing.assert_identical(result.data, expected + 1)


def test_console_contract_result_mutation_requires_snapshot(console, monkeypatch):
    manager, tools, data = console
    result = tools[0] + 1
    with pytest.raises(TypeError, match=r"\.data"):
        result[0, 0] = 10
    snapshot = result.data
    snapshot[0, 0] = 10
    result.qshow(manager=True)
    xr.testing.assert_identical(fetch(1), data + 1)
    _assert_replay(manager, tools, 1, data + 1, monkeypatch)


def test_console_contract_replacement_accepts_next_index_and_child_uid(
    console, monkeypatch
):
    manager, tools, data = console
    result = tools[0] + 1
    result.qshow(manager=True, replace=manager.next_idx)
    assert manager.ntools == 2
    _assert_replay(manager, tools, 1, data + 1, monkeypatch)
    child = manager.add_imagetool_child(
        itool(data + 10, manager=False, execute=False), 0, show=False
    )
    window = manager.get_imagetool(child)
    result.qshow(manager=True, replace=child)
    assert manager.ntools == 2
    assert manager.get_imagetool(child) is window
    xr.testing.assert_identical(window.slicer_area.displayed_data, data + 1)
    _assert_replay(manager, tools, child, data + 1, monkeypatch)


@pytest.mark.parametrize(
    "options", [{"manager": False}, {"replace": [0, 1]}, {"replace": 99}]
)
def test_console_contract_invalid_publication_preserves_windows(console, options):
    manager, tools, data = console
    result = tools[0] + 1
    graph = manager._data_graph.graph
    revision = graph.current_revision(tools[0].uid)
    with pytest.raises((TypeError, ValueError, KeyError, IndexError)):
        result.qshow(**options)
    assert manager.ntools == 1
    assert graph.current_revision(tools[0].uid) == revision
    xr.testing.assert_identical(fetch(0), data)


def test_console_contract_temporary_results_release_input_captures(console):
    manager, tools, _data = console
    graph = manager._data_graph.graph
    result = tools[0] + 1
    calculation = result._calculation_revision
    item = result._script_inputs[0]
    item_ref = weakref.ref(item)
    snapshot = result.data
    snapshot_ref = weakref.ref(snapshot)
    alias = item.revision_id
    assert graph.captured_input(alias) is not None
    del result, item, snapshot
    gc.collect()
    assert item_ref() is None
    assert snapshot_ref() is None
    assert not graph.has_revision(calculation)


@pytest.mark.parametrize("close_source", [False, True])
@pytest.mark.parametrize("file_change", ["unchanged", "replaced", "removed"])
def test_console_contract_retained_lazy_result_keeps_file_guard(
    manager_context, tmp_path, close_source, file_change, monkeypatch
):
    from erlab.interactive.imagetool._provenance._model import FileDataSelection

    path = tmp_path / "source.nc"
    values = xr.DataArray(np.arange(120.0).reshape(4, 5, 6), dims=("scan", "x", "y"))
    values.to_netcdf(path, engine="h5netcdf")
    with manager_context() as manager:
        source = xr.open_dataarray(path, engine="h5netcdf", chunks={"scan": 1})
        manager.add_imagetool(
            itool(
                source,
                manager=False,
                execute=False,
                auto_compute=False,
                file_path=path,
                load_func=(
                    xr.open_dataarray,
                    {"engine": "h5netcdf", "chunks": {"scan": 1}},
                    FileDataSelection(kind="dataarray"),
                ),
            ),
            show=False,
        )
        tools = manager_console.ToolsNamespace(manager)
        result = tools[0] * 2
        if close_source:
            manager.remove_imagetool(0)
        else:
            tools[0].data = values + 100
        target = manager.next_idx
        result.qshow(manager=True, auto_compute=False)
        published = manager.get_imagetool(target).slicer_area.displayed_data
        assert published.chunks is not None
        xr.testing.assert_identical(published, values * 2)
        _assert_replay(manager, tools, target, values * 2, monkeypatch)
        if file_change == "replaced":
            replacement = tmp_path / "replacement.nc"
            (values + 10).to_netcdf(replacement, engine="h5netcdf")
            replacement.replace(path)
        elif file_change == "removed":
            path.unlink()
        if file_change == "unchanged":
            xr.testing.assert_identical(result.data.compute(), values * 2)
        else:
            with pytest.raises(OSError, match="unavailable or has changed"):
                published.isel(scan=-1).compute()
            with pytest.raises(OSError, match="unavailable or has changed"):
                result.data.isel(scan=-1).compute()


def test_console_contract_lazy_inspection_does_not_compute(manager_context):
    from dask.callbacks import Callback

    data = xr.DataArray(np.ones((4, 5, 6)), dims=("scan", "x", "y")).chunk({"scan": 1})
    with manager_context() as manager:
        manager.add_imagetool(
            itool(data, manager=False, execute=False, auto_compute=False), show=False
        )
        tools = manager_console.ToolsNamespace(manager)
        computed = []
        with Callback(posttask=lambda *args: computed.append(args[0])):
            snapshot = tools[0].data
            for _ in range(3):
                assert tools[0].shape == (4, 5, 6)
                plane = tools[0].isel(scan=0)
                assert plane.data.chunks is not None
            assert snapshot.chunks is not None
        assert not computed


@pytest.mark.parametrize("derived", [False, True])
@pytest.mark.parametrize("auto", [False, True])
def test_console_contract_child_publication_uses_consumed_inputs(
    console, monkeypatch, qtbot, derived, auto
):
    manager, tools, data = console
    handle = tools[0] * 2 if derived else tools[0]
    operand = handle._console_operand()
    publication = (
        _CalculatedResult(handle._captured_input._capture, "derived")
        if derived
        else handle._view_publication("derived", operand)
    )
    child = manager.add_imagetool_child(
        publication,
        0,
        show=False,
        source_auto_update=auto,
    )
    expected = data * 2 if derived else data
    tools[0].data = data + 100
    xr.testing.assert_identical(manager.get_imagetool(child).slicer_area.data, expected)
    _assert_replay(manager, tools, child, expected, monkeypatch)
    graph = manager._data_graph.graph
    assert graph.auto_enabled(child) == auto
    assert graph.is_stale(child)
    if auto:
        qtbot.wait_until(lambda: not graph.is_stale(child))
    else:
        graph.refresh(child)
    expected_next = (data + 100) * 2 if derived else data + 100
    xr.testing.assert_identical(
        manager.get_imagetool(child).slicer_area.data, expected_next
    )
    _assert_replay(manager, tools, child, expected_next, monkeypatch)


@pytest.mark.parametrize(
    ("op", "expression"),
    [
        ("qsel", "source.qsel(x=1.0, x_width=2.0, func='sum')"),
        (
            "gaussian_filter",
            "era.image.gaussian_filter(source, 1.1, (0, 1), ('reflect', 'constant'), "
            "cval=-6, truncate=2.5, radius=[1, 2])",
        ),
        (
            "gaussian_filter",
            "era.image.gaussian_filter(source, (0.5, 1.5), "
            "mode={'x':'nearest','y':'wrap'})",
        ),
        ("boxcar_filter", "era.image.boxcar_filter(source, 2, mode='grid-wrap')"),
        (
            "boxcar_filter",
            "era.image.boxcar_filter(source, [2, 4], mode='constant', cval=np.inf)",
        ),
        ("image_derivative", "era.image.diffn(source, 'y', acc=2)"),
        (
            "image_derivative",
            "era.image.minimum_gradient(source, mode='constant', cval=11)",
        ),
        ("sel", "source.sel(x=[0.3, 1.7], method='nearest', tolerance=0.4, drop=True)"),
        (
            "isel",
            "source.isel(y=slice(1, 3), missing=0, drop=True, missing_dims='ignore')",
        ),
        (
            "rotate",
            "era.transform.rotate(source, 27, (1, 0), {'y': 2, 'x': 0.5}, "
            "reshape=False, order=3, mode='grid-constant', cval=-43, prefilter=False)",
        ),
        (
            "symmetrize_nfold",
            "era.transform.symmetrize_nfold(source, 3, (0, 1), {'y': 2, 'x': 0.5}, "
            "reshape=False, order=3, mode='nearest', cval=17, prefilter=False)",
        ),
        (
            "symmetrize",
            "era.transform.symmetrize(source, 'y', center=1.25, "
            "interp_kw={'method': 'nearest', "
            "'kwargs': {'fill_value': np.array(-22.0)}})",
        ),
        (
            "slice_along_path",
            "era.interpolate.slice_along_path(source, {'x': [0, 2], 'y': [0, 3]}, "
            "interp_kwargs={'method': 'nearest'})",
        ),
        (
            "slice_along_path",
            "era.interpolate.slice_along_path(source, None, x=np.array([0, 2]), "
            "y=np.array([0, 3]), interp_kwargs={'method': 'nearest'})",
        ),
        (
            "mask_with_polygon",
            "era.mask.mask_with_polygon(source, "
            "np.array([[0., 0.], [2., 0.], [2., 3.]]))",
        ),
        (
            "leading_edge",
            "era.interpolate.leading_edge(source, fraction=0.75, dim='y', "
            "direction='negative', on_failure='raise')",
        ),
        (
            "correct_with_edge",
            "era.gold.correct_with_edge(source, (0.12, -0.03), along='x', "
            "shift_coords=False, order=3, mode='nearest', prefilter=False)",
        ),
        (
            "correct_with_edge",
            "era.gold.correct_with_edge(source, np.array([0.1, -0.2, 0.3], "
            "dtype=np.float32), along='x', shift_coords=False, cval=-25)",
        ),
        (
            "correct_with_edge",
            "era.gold.correct_with_edge(source, xr.Dataset({'modelfit_coefficients': "
            "('param', [0.2])}, coords={'param': ['c0']}), "
            "along='x', shift_coords=False)",
        ),
    ],
)
def test_console_library_parameters_survive_refresh(
    console, monkeypatch, op, expression
):
    manager, tools, data = console
    if op == "leading_edge":
        data = data.copy(data=np.tile([0.0, 10.0, 5.0, 0.0], (3, 1)))
        tools[0].data = data
    elif op == "correct_with_edge":
        data = data.rename(y="eV")
        tools[0].data = data
    ns = manager_console._resolve_console_namespace({"era": erlab.analysis})
    code = f"result = {expression}"
    result = _exec_generated_code(code, {**ns, "source": tools[0]})["result"]
    expected = _exec_generated_code(code, {"source": data})["result"]
    xr.testing.assert_identical(result.data, expected)
    (operation,) = result._operations
    assert operation.op == op
    restored = type(operation).model_validate_json(operation.model_dump_json())
    xr.testing.assert_identical(restored.apply(data), expected)
    result.qshow(manager=True)
    tools[0].data = 2 * data
    _assert_replay(manager, tools, 1, expected, monkeypatch)
    manager._data_graph.graph.refresh(tools[1].uid)
    updated = _exec_generated_code(code, {"source": 2 * data})["result"]
    xr.testing.assert_identical(tools[1].data, updated)
    _assert_replay(manager, tools, 1, updated, monkeypatch)


def test_live_handle_inspector_opens_graph_input_with_conflicting_spec(
    console, monkeypatch
):
    from erlab.interactive.imagetool._provenance._operations import AssignAttrsOperation

    manager, tools, data = console
    manager.add_imagetool(itool(data + 10, manager=False, execute=False), show=False)
    tools[0].qshow(manager=True)
    result = manager._node_for_target(2)
    adapter = manager._data_graph
    graph = adapter.graph
    wrong = tools[1]._bound_data_recipe("derived", tools[1]._script_input())
    wrong = wrong.append_operations(AssignAttrsOperation(attrs={"station": "other"}))
    result._provenance_spec = wrong
    result.imagetool.set_provenance_spec(wrong)
    result._invalidate_provenance_caches(refresh_display=True)
    select_tools(manager, [2])
    manager._set_metadata_node(result)
    manager._update_info()

    rows = result.derivation_display_rows
    controller = manager._provenance_edit_controller
    # A bare handle has no metadata assignment for the inspector to delete.
    assert not any(controller.can_delete_row(row)[0] for row in rows)
    used = rows[0].children[0]
    assert used.revision_id == graph.current_revision(tools[0].uid)
    controller.open_input_row(used)
    xr.testing.assert_identical(fetch(3), data)
    _assert_replay(manager, tools, 2, data, monkeypatch)
    tools[0].data = data + 2
    assert graph.is_stale(result.uid)
    graph.refresh(result.uid)
    xr.testing.assert_identical(fetch(2), data + 2)
    xr.testing.assert_identical(fetch(3), data)


@pytest.mark.parametrize("named", [False, True])
def test_console_consumers_share_calculated_revision_and_alignment(
    console, monkeypatch, named
):
    manager, tools, left = console
    right = (left + 10).assign_coords(x=left.x + 1)
    manager.add_imagetool(itool(right, manager=False, execute=False), show=False)
    graph = manager._data_graph.graph
    with xr.set_options(arithmetic_join="inner"):
        expected = left + right
        partial = tools[0] + tools[1]
        if named:
            partial._set_console_name("partial")
        first, second = partial * 2, partial * 3
    tools[0].data = left + 4
    with xr.set_options(arithmetic_join="outer"):
        first.qshow(manager=True)
        second.qshow(manager=True)
        revisions = []
        for index, factor in [(2, 2), (3, 3)]:
            uid = tools[index].uid
            (edge,) = graph.revision(graph.current_revision(uid)).producer.inputs
            revisions.append(edge[1])
            xr.testing.assert_identical(graph.data(edge[1]), expected)
            xr.testing.assert_identical(fetch(index), expected * factor)
            _assert_replay(manager, tools, index, expected * factor, monkeypatch)
            assert graph.is_stale(uid)
            graph.refresh(uid)
            xr.testing.assert_identical(fetch(index), (expected + 4) * factor)
        assert revisions[0] == revisions[1]


@pytest.mark.parametrize("close_source", [False, True])
def test_consuming_named_calculation_after_closure_or_self_replacement(
    console, monkeypatch, close_source
):
    manager, tools, data = console
    manager.add_imagetool(itool(data + 10, manager=False, execute=False), show=False)
    partial = tools[0] + tools[1]
    partial._set_console_name("partial")
    result = partial * 2
    if close_source:
        manager._remove_imagetool(0, preflight=False)
        result.qshow(manager=True)
        index = manager._tool_graph.displayed_indices[-1]
    else:
        tools[0].data = result
        index = 0
    graph = manager._data_graph.graph
    target = manager._node_for_target(index)
    assert graph.follows(target.uid) == {tools[1].uid}
    assert not graph.is_stale(target.uid)
    expected = (data + (data + 10)) * 2
    _assert_replay(manager, tools, index, expected, monkeypatch)
    tools[1].data = data + 11
    graph.refresh(target.uid)
    xr.testing.assert_identical(fetch(index), expected + 2)


def test_typed_console_consumers_retain_the_same_selected_revision(
    console, monkeypatch
):
    manager, tools, data = console
    selected = tools[0].isel(x=slice(1, None))
    selected_revision = selected._calculation_revision
    first = selected.transpose("y", "x")
    second = selected.isel(y=slice(1, None))
    del selected
    gc.collect()
    graph = manager._data_graph.graph
    for result, expected in [
        (first, data.isel(x=slice(1, None)).transpose("y", "x")),
        (second, data.isel(x=slice(1, None), y=slice(1, None))),
    ]:
        result.qshow(manager=True)
        index = manager._tool_graph.displayed_indices[-1]
        revision = graph.current_revision(tools[index].uid)
        assert graph.revision(revision).producer.inputs == (
            ("data", selected_revision),
        )
        xr.testing.assert_identical(fetch(index), expected)
        _assert_replay(manager, tools, index, expected, monkeypatch)


def test_console_accessor_uses_calculated_receiver_and_indexer(console, monkeypatch):
    manager, tools, data = console
    selected = tools[0].isel(y=slice(2))
    indexer = tools[0].x.isel(x=slice(1, None))
    selected.qsel(x=indexer).qshow(manager=True)
    graph = manager._data_graph.graph
    producer = graph.revision(graph.current_revision(tools[1].uid)).producer
    assert [key for _, key in producer.inputs] == [
        selected._calculation_revision,
        indexer._calculation_revision,
    ]
    expected = data.isel(y=slice(2)).qsel(x=data.x.isel(x=slice(1, None)))
    xr.testing.assert_identical(fetch(1), expected)
    _assert_replay(manager, tools, 1, expected, monkeypatch)


def test_console_conversion_groups_the_exact_setup_revision(
    console, monkeypatch, accept_dialog
):
    manager, tools, _ = console
    data = xr.DataArray(
        np.arange(27.0).reshape(3, 3, 3),
        dims=("alpha", "beta", "eV"),
        coords={
            "alpha": [-1.0, 0.0, 1.0],
            "beta": [-1.0, 0.0, 1.0],
            "eV": [-0.2, 0.0, 0.2],
            "xi": 0.0,
            "hv": 21.2,
        },
        attrs={
            "configuration": int(erlab.constants.AxesConfiguration.Type1),
            "sample_workfunction": 4.5,
        },
    )
    tools[0].data = data
    setup = tools[0].kspace.set_normal(1.5, -0.5, delta=2.0)
    converted = setup.kspace.convert()
    converted.qshow(manager=True)
    node = manager._node_for_target(1)
    adapter = manager._data_graph
    graph = adapter.graph
    revision = graph.current_revision(node.uid)
    applied = graph.revision(revision).producer
    assert applied.inputs == (("data", setup._calculation_revision),)
    members = (setup._calculation_revision, revision)
    rows = node.derivation_display_rows
    setup_row = rows[0].children[0].materialized_children()[0]
    conversion_row = next(row for row in rows if row.graph_revision == revision)
    assert setup_row.graph_revision == setup._calculation_revision
    for row in (setup_row, conversion_row):
        assert row.graph_group == members
        assert manager._provenance_edit_controller._can_edit_row(node, row)[0]
    scopes = adapter.selected_scopes(node.uid, {key: {None} for key in members})
    markers = [scope.spec.operations[-1].group for scope in scopes]
    assert [marker.index for marker in markers] == [0, 1]
    assert markers[0].id == markers[1].id
    assert [op.group for op in node.provenance_spec.operations[-2:]] == markers
    assert graph.revision(setup._calculation_revision).producer.operation.group is None
    _assert_replay(manager, tools, 1, converted.data, monkeypatch)
    select_tools(manager, [1])
    manager._set_metadata_node(node)
    accept_dialog(lambda: manager._provenance_edit_controller.edit_row(setup_row))
    assert graph.current_revision(node.uid) == revision


@pytest.mark.parametrize("read_first", [False, True])
def test_first_notebook_publication_owns_origin_before_rename(
    manager_context, monkeypatch, read_first
):
    data = xr.DataArray(np.arange(6.0).reshape(2, 3), dims=("x", "y"))
    with manager_context() as manager:
        itool([data, data + 10], manager=True)
        tools = manager_console.ToolsNamespace(manager)
        node = manager._node_for_target(0)
        graph = manager._data_graph.graph
        original = graph.current_revision(node.uid)
        wrong = tools[1]._bound_data_recipe("derived", tools[1]._script_input())
        node._provenance_spec = wrong
        node.imagetool.set_provenance_spec(wrong)
        if read_first:
            xr.testing.assert_identical(tools[0].data, data)
        manager.rename_imagetool(0, "scan")
        (tools[0] * 2).qshow(manager=True)
        applied = graph.revision(graph.current_revision(node.uid)).producer
        assert applied.operation.op == "rename"
        assert applied.inputs == (("data", original),)
        _assert_replay(manager, tools, 2, (data * 2).rename("scan"), monkeypatch)


@pytest.mark.parametrize("calculated", [False, True])
def test_window_publication_records_angle_coordinates(console, monkeypatch, calculated):
    manager, tools, data = console
    source = tools[0] * 2 if calculated else tools[0]
    source.qshow(manager=True, rad2deg=("x",))
    expected = (data * 2 if calculated else data).assign_coords(x=np.rad2deg(data.x))
    xr.testing.assert_identical(fetch(1), expected)
    _assert_replay(manager, tools, 1, expected, monkeypatch)
    tools[0].data = data + 10
    manager._data_graph.graph.refresh(tools[1].uid)
    updated = expected + (20 if calculated else 10)
    xr.testing.assert_identical(fetch(1), updated)


@pytest.mark.parametrize("placement", ["new", "replace", "snapshot"])
def test_window_publication_records_selected_high_dimensional_slice(
    console, accept_dialog, monkeypatch, placement
):
    manager, tools, data = console
    extra = {"u": [0, 1], "v": [0, 1], "w": [0, 1]}
    expanded = tools[0].expand_dims(extra)

    def choose(dialog):
        row = dialog.rows[-1]
        row.action_combo.setCurrentIndex(row.action_combo.findData("select"))
        row.scalar_controls.index_spin.setValue(1)

    if placement != "new":
        manager.add_imagetool(itool(data, manager=False, execute=False), show=False)

    def show():
        if placement == "snapshot":
            tools[1].data = expanded.data
        elif placement == "replace":
            tools[1].data = expanded
        else:
            expanded.qshow(manager=True)

    accept_dialog(show, pre_call=choose)
    expected = data.expand_dims(extra).isel(y=1)
    xr.testing.assert_identical(fetch(1), expected)
    _assert_replay(manager, tools, 1, expected, monkeypatch)
    if placement == "snapshot":
        assert not manager._data_graph.graph.follows(tools[1].uid)
        return
    tools[0].data = data + 3
    manager._data_graph.graph.refresh(tools[1].uid)
    xr.testing.assert_identical(fetch(1), expected + 3)


@pytest.mark.parametrize("operand", ["window", "calculation", "both"])
def test_repeated_operand_slots_survive_show_and_close(console, monkeypatch, operand):
    manager, tools, data = console
    graph = manager._data_graph.graph
    partial = tools[0] * 2
    partial._set_console_name("partial")
    partial.qshow(manager=True)
    partial.qshow(manager=True)
    revision = partial._calculation_revision
    assert (
        graph.current_revision(tools[1].uid)
        == graph.current_revision(tools[2].uid)
        == revision
    )
    manager.remove_imagetool(1)
    if operand == "window":
        result = tools[2] + tools[2]
    elif operand == "calculation":
        result = partial + partial
    else:
        result = partial + tools[2]
    result.qshow(manager=True)
    node = manager._node_for_target(3)
    applied = graph.revision(graph.current_revision(node.uid)).producer
    assert [key for _, key in applied.inputs] == [revision, revision]
    assert len(node.derivation_display_rows[0].children) == 2
    _assert_replay(manager, tools, 3, data * 4, monkeypatch)
    tools[0].data = data + 5
    graph.refresh(node.uid)
    xr.testing.assert_identical(fetch(3), (data + 5) * 4)


@pytest.mark.parametrize(
    "action", ["concat", "shift", "open", "shift-child", "shift-replace"]
)
def test_gui_publication_uses_calculation_inputs_with_conflicting_import(
    manager_context, accept_dialog, monkeypatch, action
):
    from erlab.interactive.imagetool import dialogs

    data = xr.DataArray(
        np.arange(15.0).reshape(3, 5),
        dims=("x", "eV"),
        coords={"x": [0.0, 1.0, 2.0], "eV": np.linspace(-0.2, 0.2, 5)},
    )
    second = (
        xr.DataArray([0.0, 0.05, -0.05], dims="x", coords={"x": data.x})
        if action.startswith("shift")
        else data + 10
    )
    with manager_context() as manager:
        itool([data, second, second + 10], manager=True)
        tools = manager_console.ToolsNamespace(manager)
        graph = manager._data_graph.graph
        revisions = [graph.current_revision(tools[i].uid) for i in (0, 1)]
        wrong = tools[2]._bound_data_recipe("wrong", tools[2]._script_input())
        make_window = manager._window_for_input

        def substitute(*args, **kwargs):
            tool, origin = make_window(*args, **kwargs)
            tool.set_provenance_spec(wrong)
            return tool, origin

        monkeypatch.setattr(manager, "_window_for_input", substitute)
        if action == "shift-replace":
            source = manager._node_for_target(0)
            replace = source.publish_replacement

            def substitute_replacement(result):
                source.imagetool.set_provenance_spec(wrong)
                return replace(result)

            monkeypatch.setattr(source, "publish_replacement", substitute_replacement)
        if action == "concat":
            select_tools(manager, [0, 1])
            accept_dialog(manager.concat_action.trigger)
        elif action.startswith("shift"):
            dialog = dialogs.ShiftDialog(manager.get_imagetool(0).slicer_area)
            dialog.shift_source_combo.setCurrentIndex(
                dialog.shift_source_combo.findData(dialog._SOURCE_MANAGER)
            )
            dialog.shift_tool_selector.set_current_uid(tools[1].uid)
            dialog.launch_mode_combo.setCurrentIndex(
                dialog.launch_mode_combo.findData(
                    {"shift-child": "nest", "shift-replace": "replace"}.get(
                        action, "detach"
                    )
                )
            )
            dialog.accept()
        else:
            manager._data_graph.open_recorded_input(revisions[0])
        target = (
            manager._node_for_target(0)._childtool_indices[0]
            if action == "shift-child"
            else 0
            if action == "shift-replace"
            else 3
        )
        node = manager._node_for_target(target)
        applied = graph.revision(graph.current_revision(node.uid)).producer
        assert [key for _, key in applied.inputs] == (
            revisions[:1] if action == "open" else revisions
        )
        _assert_replay(manager, tools, target, node.current_public_data(), monkeypatch)
        tools[2].data = second + 20
        assert not graph.is_stale(node.uid)
        tools[1].data = second + 1
        assert graph.is_stale(node.uid) == (action != "open")


@pytest.mark.parametrize("typed", [False, True])
def test_console_calculation_uses_captured_operands_without_export(
    console, monkeypatch, typed
):
    manager, tools, data = console
    graph = manager._data_graph.graph
    selected = graph.current_revision(tools[0].uid)

    def forbid_export(*args, **kwargs):
        pytest.fail("Capturing a live application must not depend on a portable recipe")

    monkeypatch.setattr(
        manager_console._DerivedDataNamespace, "_console_provenance_spec", forbid_export
    )
    result = tools[0].rename("sample") if typed else tools[0] * 2
    result.qshow(manager=True)
    node = manager._node_for_target(1)
    assert graph.revision(graph.current_revision(node.uid)).producer.inputs == (
        ("data", selected),
    )
    assert graph.follows(node.uid) == {tools[0].uid}
    expected = data.rename("sample") if typed else data * 2
    _assert_replay(manager, tools, 1, expected, monkeypatch)


@pytest.mark.parametrize("action", ["arithmetic", "typed", "item"])
def test_console_executes_once_from_recorded_revisions(console, monkeypatch, action):
    from dataclasses import replace

    from erlab.interactive.imagetool._provenance._operations import RenameOperation

    manager, tools, data = console
    handle = tools[0]
    read = handle._console_operand

    def read_with_conflicting_value():
        operand = read()
        return replace(operand, value=xr.full_like(operand.value, -100))

    monkeypatch.setattr(handle, "_console_operand", read_with_conflicting_value)
    calls = []
    method_owner, method_name = {
        "arithmetic": (xr.DataArray, "__add__"),
        "typed": (RenameOperation, "apply"),
        "item": (xr.DataArray, "__setitem__"),
    }[action]
    run = getattr(method_owner, method_name)

    def count(*args, **kwargs):
        calls.append(None)
        return run(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(method_owner, method_name, count)
        if action == "arithmetic":
            result = handle + 1
        elif action == "typed":
            result = handle.rename("renamed")
        else:
            handle[0, 1] = 7.0
    assert len(calls) == 1
    target = 0 if action == "item" else 1
    if action == "item":
        expected = data.copy(deep=True)
        expected[0, 1] = 7.0
    else:
        expected = data + 1 if action == "arithmetic" else data.rename("renamed")
        xr.testing.assert_identical(result.data, expected)
        result.qshow(manager=True)
    xr.testing.assert_identical(
        manager.get_imagetool(target).slicer_area.displayed_data, expected
    )
    _assert_replay(manager, tools, target, expected, monkeypatch)


def test_nested_notebook_input_owns_its_origin_at_ingress(console, monkeypatch):
    manager, tools, data = console
    child = manager.add_imagetool_child(
        itool(data + 3, manager=False, execute=False), 0, show=False
    )
    node = manager._node_for_target(child)
    graph = manager._data_graph.graph
    origin = graph.current_revision(child)
    wrong = tools[0]._bound_data_recipe("wrong", tools[0]._script_input())
    node._provenance_spec = wrong
    node.imagetool.set_provenance_spec(wrong)
    (tools[0].children[0] * 2).qshow(manager=True)
    result = manager._node_for_target(1)
    assert graph.revision(graph.current_revision(result.uid)).producer.inputs == (
        ("data", origin),
    )
    assert graph.follows(result.uid) == {child}
    _assert_replay(manager, tools, 1, (data + 3) * 2, monkeypatch)


@pytest.mark.parametrize("reduction", ["none", "sender", "manager"])
def test_notebook_registration_ignores_window_history(console, monkeypatch, reduction):
    from erlab.interactive.imagetool import _highdim
    from erlab.interactive.imagetool._provenance._operations import IselOperation

    manager, tools, data = console
    expected = (data + 3).rename("new_input")
    other = tools[0]._bound_data_recipe("other", tools[0]._script_input())
    register = manager._register_imagetool

    def substitute(tool, origin, **options):
        tool.set_provenance_spec(other)
        return register(tool, origin, **options)

    monkeypatch.setattr(manager, "_register_imagetool", substitute)
    incoming = expected
    if reduction != "none":
        incoming = incoming.expand_dims({"a": [0, 1], "b": [0, 1], "c": [0, 1]})
        operation = IselOperation(kwargs={"a": 0, "b": 0, "c": 0})
        expected = operation.apply(incoming)

        class Reduce:
            def __init__(self, _parent, value):
                self.result_data = operation.apply(value)

            def exec(self):
                return True

            accepted_operations = (operation,)

        monkeypatch.setattr(_highdim, "_HighDimensionalReductionDialog", Reduce)

    if reduction == "manager":
        # The sender already selected a higher dimensional input. Only its sent
        # array and the additional selection made locally can be checkpoints.
        manager._data_ingress.receive_data(
            [incoming],
            {
                "preparation_operations": ((IselOperation(kwargs={"run": 0}),),),
            },
        )
    else:
        itool(incoming, manager=True)
    uid = manager._node_for_target(1).uid
    graph = manager._data_graph.graph
    revision = graph.current_revision(uid)
    applied = graph.revision(revision).producer
    if reduction == "manager":
        assert applied.operation == operation
        xr.testing.assert_identical(graph.data(applied.inputs[0][1]), incoming)
    else:
        assert applied is None
    xr.testing.assert_identical(graph.data(revision), expected)
    (tools[1] * 2).qshow(manager=True)
    _assert_replay(manager, tools, 2, expected * 2, monkeypatch)
    tools[0].data = data + 19
    assert graph.current_revision(uid) == revision
    assert not graph.follows(uid)
    assert not graph.is_stale(tools[2].uid)
    area = manager.get_imagetool(1).slicer_area
    assert area.reloadable == (reduction == "manager")
    assert area._reload() == (reduction == "manager")
    xr.testing.assert_identical(fetch(1), expected)


@pytest.mark.parametrize("child", [False, True])
def test_registration_callbacks_wait_for_the_published_input(
    console, monkeypatch, child
):
    manager, _tools, data = console
    graph = manager._data_graph
    method = "_register_child_node" if child else "_register_root_wrapper"
    register = getattr(manager, method)
    seen = []

    def inspect_during_registration(node):
        register(node)
        with pytest.raises(RuntimeError, match="being updated"):
            fetch(node.uid)
        with pytest.raises(RuntimeError, match="being updated"):
            graph.read_input(node.uid)
        with pytest.raises(RuntimeError, match="being updated"):
            node.slicer_area.set_data(data + 71)
        seen.append(node.uid)

    monkeypatch.setattr(manager, method, inspect_during_registration)
    incoming = data + 3
    tool = itool(incoming, manager=False, execute=False)
    if child:
        target = manager.add_imagetool_child(tool, 0)
    else:
        target = manager.add_imagetool(tool)
    node = manager._node_for_target(target)
    assert seen == [node.uid]
    assert graph.owns_result(node.uid)
    xr.testing.assert_identical(fetch(node.uid), incoming)


@pytest.mark.parametrize("child", [False, True])
def test_live_registration_rejects_imported_recipe(console, qtbot, child):
    manager, tools, data = console
    tool = itool(data + 3, manager=False, execute=False)
    qtbot.addWidget(tool)
    wrong = tools[0]._bound_data_recipe("other", tools[0]._script_input())
    register = manager.add_imagetool_child if child else manager.add_imagetool
    args = (tool, 0) if child else (tool,)
    with pytest.raises(TypeError, match="provenance_spec"):
        register(*args, provenance_spec=wrong)
    assert manager.ntools == 1
    assert not manager._node_for_target(0)._childtool_indices


@pytest.mark.parametrize("missing_spec", [False, True])
def test_file_registration_uses_the_loaded_file(
    console, tmp_path, monkeypatch, missing_spec
):
    from erlab._provenance import FileLoad
    from erlab.interactive.imagetool._provenance._model import FileDataSelection

    manager, tools, _ = console
    data = xr.DataArray(np.arange(6.0).reshape(2, 3), dims=("x", "y"), name="scan")
    selected, other = tmp_path / "selected.h5", tmp_path / "other.h5"
    load_func = (
        xr.load_dataarray,
        {"engine": "h5netcdf"},
        FileDataSelection(kind="dataarray"),
    )
    for path, array in ((other, data + 19), (selected, data)):
        array.to_netcdf(path, engine="h5netcdf")
    manager._data_ingress.receive_data(
        [xr.load_dataarray(other)], {"file_path": other, "load_func": load_func}
    )
    spec = None if missing_spec else manager._node_for_target(1).provenance_spec
    register = manager._register_imagetool

    def substitute(tool, origin, **options):
        tool.set_provenance_spec(spec)
        return register(tool, origin, **options)

    monkeypatch.setattr(manager, "_register_imagetool", substitute)
    assert manager._data_ingress.receive_data(
        [xr.load_dataarray(selected)], {"file_path": selected, "load_func": load_func}
    ) == [True]
    node = manager._node_for_target(2)
    graph = manager._data_graph.graph
    root = graph.revision(graph.current_revision(node.uid)).producer
    assert isinstance(root, FileLoad)
    assert root.source.path == str(selected)
    _assert_replay(manager, tools, 2, data, monkeypatch)
    (data + 7).to_netcdf(selected, engine="h5netcdf")
    assert node.slicer_area._reload()
    xr.testing.assert_identical(fetch(2), data + 7)


@pytest.mark.parametrize("prepared", [False, True])
def test_file_registration_keeps_session_loader(console, tmp_path, prepared):
    from erlab._provenance import FileLoad
    from erlab.interactive.imagetool._provenance._model import FileDataSelection
    from erlab.interactive.imagetool._provenance._operations import TransposeOperation

    manager, tools, _ = console
    data = xr.DataArray(np.arange(6).reshape(2, 3), dims=("x", "y"), name="scan")
    path = tmp_path / "session.h5"
    data.to_netcdf(path, engine="h5netcdf")
    reads = []

    def load(path, *, offset):
        reads.append(path)
        return (xr.load_dataarray(path, engine="h5netcdf") + offset).to_dataset()

    selection = FileDataSelection(kind="dataset_variable", value="scan")
    incoming = load(path, offset=7)["scan"]
    options = {"file_path": path, "load_func": (load, {"offset": 7}, selection)}
    if prepared:
        options["preparation_operations"] = [(TransposeOperation(dims=("y", "x")),)]
        incoming = incoming.transpose()
    assert manager._data_ingress.receive_data([incoming], options) == [True]
    node = manager._node_for_target(1)
    graph = manager._data_graph.graph
    origin = graph.recorded_lineage(graph.current_revision(node.uid))[0].producer
    assert isinstance(origin, FileLoad)
    assert origin.source.path == str(path)
    assert origin.call.selection == selection
    inputs = {}

    def name_checkpoint(key, _suggestion, _used_names):
        inputs["selected"] = tools.checkpoint(key)
        return "selected"

    code = manager._data_graph.copy_code(node, name_checkpoint)
    xr.testing.assert_identical(
        _exec_generated_code(code, inputs)["derived"], incoming.astype(float)
    )
    assert reads == [path]

    (data + 20).to_netcdf(path, engine="h5netcdf")
    assert node.slicer_area._reload()
    assert reads == [path, path]
    xr.testing.assert_identical(fetch(1), (incoming + 20).astype(float))


def test_file_reload_keeps_each_wrapped_loader(console, tmp_path, monkeypatch):
    from erlab.interactive.imagetool._provenance._model import FileDataSelection

    manager, tools, _ = console
    data = xr.DataArray(np.arange(6.0).reshape(2, 3), dims=("x", "y"), name="scan")
    path = tmp_path / "wrapped.h5"
    data.to_netcdf(path, engine="h5netcdf")

    def shifted_loader(offset):
        @functools.wraps(xr.load_dataarray)
        def load(path, **kwargs):
            return xr.load_dataarray(path, **kwargs) + offset

        return load

    selection = FileDataSelection(kind="dataarray")
    for offset in (10, 20):
        load = shifted_loader(offset)
        kwargs = {"engine": "h5netcdf"}
        assert manager._data_ingress.receive_data(
            [load(path, **kwargs)],
            {"file_path": path, "load_func": (load, kwargs, selection)},
        ) == [True]
    for target, offset in ((1, 10), (2, 20)):
        _assert_replay(
            manager, tools, target, data + offset, monkeypatch, output="derived"
        )
        assert manager.get_imagetool(target).slicer_area._reload()
        xr.testing.assert_identical(fetch(target), data + offset)


@pytest.mark.parametrize("degrees", [False, True])
def test_file_reload_after_external_replacement(console, tmp_path, degrees):
    from erlab.interactive.imagetool._provenance._model import FileDataSelection
    from erlab.interactive.imagetool.manager import replace_data

    manager, _, data = console
    source = data.rename("scan")
    source.attrs = {}
    path = tmp_path / "angles.h5"
    source.to_netcdf(path, engine="h5netcdf")
    assert manager._data_ingress.receive_data(
        [xr.load_dataarray(path)],
        {
            "file_path": path,
            "load_func": (xr.load_dataarray, {}, FileDataSelection(kind="dataarray")),
            "rad2deg": ("x",) if degrees else False,
        },
    ) == [True]
    area = manager.get_imagetool(1).slicer_area
    replacement = fetch(1) + 100
    replace_data(1, replacement)
    xr.testing.assert_identical(fetch(1), replacement)
    expected = source + 7
    expected.to_netcdf(path, engine="h5netcdf")
    assert area.reloadable
    assert area._reload()
    if degrees:
        expected = expected.assign_coords(x=np.rad2deg(expected.x))
    xr.testing.assert_identical(fetch(1), expected)


@pytest.mark.parametrize("destination", ["new", "existing", "derived"])
def test_file_open_replaces_managed_origin(
    console, qtbot, tmp_path, monkeypatch, accept_dialog, destination
):
    from erlab._provenance import FileLoad
    from erlab.interactive.imagetool.manager import replace_data
    from erlab.interactive.imagetool.manager._acquisition_context import (
        AcquisitionContextField,
        AcquisitionContextState,
    )

    manager, tools, data = console
    if destination == "derived":
        (tools[0] * 2).qshow(manager=True)
    target = 0 if destination == "existing" else 1
    file_data = (data + 17).rename("scan")
    file_data.attrs = {"sample": "file"}
    path = tmp_path / "opened.h5"
    file_data.to_netcdf(path, engine="h5netcdf")
    monkeypatch.setattr(
        erlab.interactive.utils,
        "file_loaders",
        lambda paths=None: {
            "NetCDF (*.h5)": (xr.load_dataarray, {"engine": "h5netcdf"})
        },
    )
    manager._acquisition_context.set_state(
        AcquisitionContextState(
            enabled=True,
            fields=(
                AcquisitionContextField.from_value(
                    kind="coordinate", name="photon_energy", value=21.2
                ),
            ),
        )
    )
    open_file = (
        manager.open
        if destination == "new"
        else manager.get_imagetool(target)._open_file
    )
    accept_dialog(
        lambda: open_file(native=False),
        pre_call=lambda dialog: dialog.selectFile(str(path)),
    )
    qtbot.wait_until(lambda: manager.ntools > target and not manager._file_handlers)
    node = manager._node_for_target(target)
    uid = node.uid

    graph = manager._data_graph.graph
    revision = graph.current_revision(uid)
    root = graph.recorded_lineage(revision)[0].producer
    assert isinstance(root, FileLoad)
    assert root.source.path == str(path)
    assert not graph.follows(uid)
    if destination == "derived":
        tools[0].data = data + 11
        assert not graph.is_stale(uid)
    expected = file_data.assign_coords(photon_energy=21.2)
    xr.testing.assert_identical(fetch(target), expected)
    _assert_replay(manager, tools, target, expected, monkeypatch)
    manager._acquisition_context.set_state(AcquisitionContextState())
    (file_data + 1).to_netcdf(path, engine="h5netcdf")
    assert node.slicer_area._reload()
    xr.testing.assert_identical(fetch(target), expected + 1)
    replace_data(target, file_data + 20)
    xr.testing.assert_identical(fetch(target), expected + 20)
    assert node.slicer_area._reload()
    xr.testing.assert_identical(fetch(target), expected + 1)


@pytest.mark.parametrize("failure", ["registration", "file_guard", "view"])
def test_failed_file_open_preserves_window_and_followed_input(
    console, tmp_path, monkeypatch, failure
):
    from erlab.interactive.imagetool.viewer_state import _prepare_input_data

    manager, tools, data = console
    (tools[0] * 2).qshow(manager=True)
    (tools[1] + 7).qshow(manager=True)
    follower = tools[2].uid
    node = manager._node_for_target(1)
    adapter = manager._data_graph
    graph = adapter.graph
    revision = graph.current_revision(node.uid)
    follower_revision = graph.current_revision(follower)
    path = tmp_path / "new.h5"
    incoming = (data + 17).rename("file")
    incoming.attrs = {}
    incoming.to_netcdf(path, engine="h5netcdf")
    (prepared,) = _prepare_input_data(xr.load_dataarray(path))

    def reject(*args, **kwargs):
        raise ValueError("File registration failed")

    with monkeypatch.context() as patch:
        if failure == "registration":
            patch.setattr(adapter, "prepare_input", reject)
        elif failure == "file_guard":
            patch.setattr(node.slicer_area, "_guard_lazy_data", reject)
        else:
            prepare_view = adapter._prepare_view

            def reject_prepared_view(*args):
                prepare_view(*args)
                reject()

            patch.setattr(adapter, "_prepare_view", reject_prepared_view)
        with pytest.raises(ValueError, match="File registration failed"):
            manager._data_ingress.receive_file(
                path, xr.load_dataarray, {}, (prepared,), incoming, target=node
            )
    assert graph.current_revision(node.uid) == revision
    assert graph.follows(node.uid) == {tools[0].uid}
    assert graph.current_revision(follower) == follower_revision
    assert not graph.is_stale(follower)
    assert node.slicer_area._file_path is None
    xr.testing.assert_identical(fetch(1), data * 2)
    tools[0].data = data + 5
    assert node._update_from_parent_source()
    xr.testing.assert_identical(fetch(1), (data + 5) * 2)


@pytest.mark.parametrize("write", ["set_data", "replace_source_data", "replace_data"])
def test_source_write_rolls_back_and_blocks_reload_until_published(
    console, monkeypatch, write
):
    from erlab.interactive.imagetool.manager import replace_data

    manager, tools, data = console
    (tools[0] * 2).qshow(manager=True)
    (tools[1] + 1).qshow(manager=True)
    node = manager._node_for_target(1)
    area = node.slicer_area
    adapter = manager._data_graph
    graph = adapter.graph
    revision = graph.current_revision(node.uid)
    observed = []

    def reject(*_args):
        observed.append(
            (
                area.reloadable,
                manager._reload_unavailable_reason_for_target(1),
                area._reload(),
            )
        )
        raise ValueError("Rejected view")

    def update():
        if write == "replace_data":
            manager._data_replace([data + 50], [1])
        else:
            getattr(area, write)(data + 50)

    with monkeypatch.context() as patch:
        patch.setattr(adapter, "_prepare_view", reject)
        with pytest.raises(ValueError, match="Rejected view"):
            update()
    assert len(observed) == 1
    reloadable, reason, reloaded = observed[0]
    assert not reloadable
    assert reason
    assert not reloaded
    assert graph.current_revision(node.uid) == revision
    assert not graph.is_stale(tools[2].uid)
    xr.testing.assert_identical(fetch(1), data * 2)
    replace_data(0, data + 3)
    assert node._update_from_parent_source()
    xr.testing.assert_identical(fetch(1), (data + 3) * 2)


@pytest.mark.parametrize("write", ["replace_source_data", "set_data"])
def test_source_write_with_file_reloads_its_selected_input(console, tmp_path, write):
    from erlab.interactive.imagetool._provenance._model import FileDataSelection

    manager, tools, data = console
    (tools[0] * 2).qshow(manager=True)
    node = manager._node_for_target(1)
    original = (data + 17).rename("scan")
    original.attrs = {}
    path = tmp_path / "scan.h5"
    original.to_netcdf(path, engine="h5netcdf")
    getattr(node.slicer_area, write)(
        original,
        file_path=path,
        load_func=(
            xr.load_dataarray,
            {"engine": "h5netcdf"},
            FileDataSelection(kind="dataarray"),
        ),
        rad2deg=("x",),
    )
    tools[0].data = data + 13
    graph = manager._data_graph.graph
    assert not graph.follows(node.uid)
    assert not graph.is_stale(node.uid)
    (original + 2).to_netcdf(path, engine="h5netcdf")
    assert node.slicer_area._reload()
    xr.testing.assert_identical(
        fetch(1), (original + 2).assign_coords(x=np.rad2deg(original.x))
    )


def test_manual_write_to_watched_window_copies_the_supplied_input(console, monkeypatch):
    from erlab.interactive.imagetool.manager._metadata_editor import (
        MetadataCellEdit,
        MetadataField,
    )

    manager, tools, data = console
    manager._data_watched_update("scan", "watch-id", data)
    node = manager._node_for_target(1)
    edit = MetadataCellEdit(
        MetadataField(kind="attribute", name="sample"), value="edited"
    )
    assert manager._metadata_editor.apply_edits({1: (edit,)})
    manual = (data + 37).astype("int16").rename("manual")
    node.slicer_area.replace_source_data(manual)
    expected = manual.astype("float64").assign_attrs(sample="edited")
    _assert_replay(manager, tools, 1, expected, monkeypatch)
    watched = (data + 83).astype("int32").rename("scan")
    manager._data_watched_update("scan", "watch-id", watched)
    provenance = node.displayed_provenance_spec
    expected = (
        watched.astype("float64")
        .rename(node.name or None)
        .assign_attrs(sample="edited")
    )
    xr.testing.assert_identical(node.current_public_data(), expected)
    replayed = _exec_generated_code(provenance.display_code(), {"scan": watched})
    xr.testing.assert_identical(replayed[provenance.active_name], expected)


def test_raw_replacement_keeps_pending_metadata_edit(console):
    from erlab._provenance import Follow, Recipe
    from erlab.interactive.imagetool._provenance._operations import AssignAttrsOperation

    manager, tools, data = console
    node = manager._node_for_target(0)
    (tools[0] * 2).qshow(manager=True)
    target = manager._node_for_target(1)
    graph = manager._data_graph.graph
    # An editor changed the recipe while Auto was off.
    graph.set_recipe(
        target.uid,
        Recipe.bind(
            AssignAttrsOperation(attrs={"sample": "displayed"}), Follow(node.uid)
        ),
    )
    graph.refresh(target.uid)
    graph.set_recipe(
        target.uid,
        Recipe.bind(
            AssignAttrsOperation(attrs={"sample": "pending"}), Follow(node.uid)
        ),
    )
    incoming = data + 42
    target.slicer_area.set_data(incoming)
    xr.testing.assert_identical(
        target.current_public_data(), incoming.assign_attrs(sample="displayed")
    )
    assert graph.is_stale(target.uid)
    graph.refresh(target.uid)
    xr.testing.assert_identical(
        target.current_public_data(), data.assign_attrs(sample="pending")
    )


def test_callback_raw_replacement_uses_published_metadata(console, monkeypatch):
    from erlab.interactive.imagetool.manager._metadata_editor import (
        MetadataCellEdit,
        MetadataField,
    )

    manager, tools, data = console
    node = manager._node_for_target(0)
    edit = MetadataCellEdit(
        MetadataField(kind="attribute", name="sample"), value="old edit"
    )
    assert manager._metadata_editor.apply_edits({0: (edit,)})
    incoming = (data + 11).astype("int16")

    def replace_again(_data):
        node.slicer_area.sigSourceDataReplaced.disconnect(replace_again)
        node.slicer_area.set_data(incoming, rad2deg=("x",))

    node.slicer_area.sigSourceDataReplaced.connect(replace_again)
    with pytest.raises(RuntimeError, match="changed during installation"):
        tools[0].data = data + 30
    expected = (
        incoming.astype("float64")
        .assign_coords(x=np.rad2deg(incoming.x))
        .assign_attrs(sample="old edit")
    )
    xr.testing.assert_identical(fetch(0), expected)
    graph = manager._data_graph.graph
    revision = graph.current_revision(node.uid)
    xr.testing.assert_identical(graph.data(revision), expected)
    origin = graph.recorded_lineage(revision)[0]
    xr.testing.assert_identical(graph.data(origin.revision_id), incoming)
    _assert_replay(manager, tools, 0, expected, monkeypatch)


def test_file_write_during_replacement_keeps_newer_origin(console, tmp_path):
    from erlab.interactive.imagetool._provenance._model import FileDataSelection

    manager, tools, data = console
    (tools[0] * 2).qshow(manager=True)
    node = manager._node_for_target(1)
    area = node.slicer_area
    newer = (data + 11).rename("newer")
    newer.attrs = {}
    path = tmp_path / "newer.h5"
    newer.to_netcdf(path, engine="h5netcdf")

    def install_newer(_value):
        area.sigSourceDataReplaced.disconnect(install_newer)
        area.set_data(
            newer,
            file_path=path,
            load_func=(xr.load_dataarray, {}, FileDataSelection(kind="dataarray")),
        )

    area.sigSourceDataReplaced.connect(install_newer)
    with pytest.raises(RuntimeError, match="changed during installation"):
        node.publish_replacement(data + 100)
    xr.testing.assert_identical(fetch(1), newer)
    (newer + 1).to_netcdf(path, engine="h5netcdf")
    assert area._reload()
    xr.testing.assert_identical(fetch(1), newer + 1)


def test_notebook_checkpoint_precedes_window_conversion(console, monkeypatch):
    from erlab.interactive.imagetool._provenance._operations._coordinates import (
        ImageToolInputOperation,
    )

    manager, tools, data = console
    incoming = data.astype("int16")
    itool(incoming, manager=True, rad2deg=("x",))
    graph = manager._data_graph.graph
    original, displayed = graph.recorded_lineage(graph.current_revision(tools[1].uid))
    assert original.producer is None
    xr.testing.assert_identical(graph.data(original.revision_id), incoming)
    assert (
        displayed.producer.operation
        == ImageToolInputOperation(
            cast_float64=True, degree_coords=("x",)
        ).capture_execution_options()
    )
    expected = incoming.astype("float64").assign_coords(x=np.rad2deg(incoming.x))
    _assert_replay(manager, tools, 1, expected, monkeypatch)


def test_rename_below_layout_replays_after_transpose_back(console, monkeypatch):
    manager, tools, data = console
    node = manager._node_for_target(0)
    node.slicer_area.swap_axes(0, 1)
    manager.rename_imagetool(0, "renamed")
    node.slicer_area.swap_axes(0, 1)
    expected = data.rename("renamed")
    xr.testing.assert_identical(fetch(0), expected)
    _assert_replay(manager, tools, 0, expected, monkeypatch)


@pytest.mark.parametrize("restore", [False, True])
@pytest.mark.parametrize("already_named", [False, True])
def test_rejected_watch_retires_notebook_name(
    console, monkeypatch, tmp_path, restore, already_named
):
    manager, tools, data = console
    if already_named:
        data = data.rename("scan")
    manager._data_watched_update("scan", "watch-id", data)
    node = manager._node_for_target(1)
    (tools[1] * 3).qshow(manager=True)
    if restore:
        workspace = manager._workspace_controller
        path = tmp_path / "watched.itws"
        workspace.saving._save_workspace_document(path)
        assert workspace.loading._load_workspace_file(
            path, replace=True, associate=True, mark_dirty=False, select=False
        )
        node = manager._node_for_target(1)
    old = None if restore else manager._data_graph.displayed_edit_revision(node)
    invalid = xr.DataArray(np.zeros((2,) * 5), dims=list("abcde"))
    messages = []
    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        lambda *args: messages.append(args),
    )
    manager._data_watched_update("scan", "watch-id", invalid)
    assert messages
    if old is not None:
        assert manager._data_graph.graph.current_revision(node.uid) == old
    _assert_replay(manager, tools, 1, data.rename("scan"), monkeypatch)
    _assert_replay(manager, tools, 2, data.rename("scan") * 3, monkeypatch)
    updated = data + 9
    manager._data_watched_update("scan", "watch-id", updated)
    code = manager._data_graph.copy_code(
        node, lambda *_: pytest.fail("missing watch binding")
    )
    xr.testing.assert_identical(
        _exec_generated_code(code, {"scan": updated})["derived"], fetch(1)
    )

    # The old calculation retains its input even after the new value acquires
    # the notebook name. Refresh then follows the accepted watch revision.
    _assert_replay(manager, tools, 2, data.rename("scan") * 3, monkeypatch)
    follower = manager._node_for_target(2)
    manager._data_graph.graph.refresh(follower.uid)
    xr.testing.assert_identical(fetch(2), updated.rename("scan") * 3)


@pytest.mark.parametrize("reduce", [False, True])
def test_raw_replacement_records_received_array_and_preparation(
    console, monkeypatch, reduce
):
    from erlab.interactive.imagetool import _highdim

    manager, tools, data = console
    incoming = (data + 23).astype("int16")
    expected = incoming
    if reduce:
        incoming = incoming.expand_dims(a=[0, 1], b=[0, 1], c=[0, 1])

        def accept(dialog):
            row = dialog.rows[0]
            row.action_combo.setCurrentIndex(row.action_combo.findData("select"))
            row.scalar_controls.index_spin.setValue(1)
            dialog.accept()
            return dialog.result()

        monkeypatch.setattr(_highdim._HighDimensionalReductionDialog, "exec", accept)
        expected = incoming.isel(a=1)
    node = manager._node_for_target(0)
    node.slicer_area.set_data(incoming, rad2deg=("x",))
    expected = expected.astype("float64").assign_coords(x=np.rad2deg(expected.x))
    graph = manager._data_graph.graph
    records = graph.recorded_lineage(graph.current_revision(node.uid))
    xr.testing.assert_identical(graph.data(records[0].revision_id), incoming)
    assert [r.producer.operation.op for r in records[1:]] == (
        ["isel", "imagetool_input"] if reduce else ["imagetool_input"]
    )
    _assert_replay(manager, tools, 0, expected, monkeypatch)


@pytest.mark.parametrize("placement", ["replace", "detach", "nest"])
def test_squeeze_preflight_uses_captured_settings(console, monkeypatch, placement):
    from erlab.interactive.imagetool.dialogs import SqueezeDialog

    manager, tools, data = console
    source = data.expand_dims(z=[17.0])
    tools[0].data = source
    node = manager._node_for_target(0)
    dialog = SqueezeDialog(node.slicer_area)
    dialog.launch_mode_combo.setCurrentIndex(
        dialog.launch_mode_combo.findData(placement)
    )
    check = dialog.preflight_data

    def change_control(value, spec):
        check(value, spec)
        dialog.drop_check.setChecked(True)

    monkeypatch.setattr(dialog, "preflight_data", change_control)
    dialog.accept()
    assert dialog.result()
    target = {"replace": 0, "detach": 1}.get(placement)
    if target is None:
        target = node._childtool_indices[0]
    expected = source.squeeze("z", drop=False)
    xr.testing.assert_identical(fetch(target), expected)
    _assert_replay(manager, tools, target, expected, monkeypatch)
    if placement != "replace":
        tools[0].data = source + 9
        manager._data_graph.graph.refresh(manager._node_for_target(target).uid)
        xr.testing.assert_identical(fetch(target), expected + 9)


def test_rename_install_failure_restores_window_and_follower(console, monkeypatch):
    manager, tools, _ = console
    tools.itool(tools[0] + 1)
    source = manager._node_for_target(0)
    follower = manager._node_for_target(1)
    graph = manager._data_graph.graph
    source.slicer_area.swap_axes(0, 1)
    before = source.slicer_area.displayed_data.copy(deep=True)
    entries = [graph.entry(node.uid) for node in (source, follower)]
    stale = graph.is_stale(follower.uid)

    def fail_after_install(*_args):
        assert source.slicer_area._data.name == "interrupted"
        raise RuntimeError("Rename installation interrupted")

    monkeypatch.setattr(graph, "_installed_step", fail_after_install)
    with pytest.raises(RuntimeError, match="Rename installation interrupted"):
        manager.rename_imagetool(0, "interrupted")
    xr.testing.assert_identical(source.slicer_area.displayed_data, before)
    assert [graph.entry(node.uid) for node in (source, follower)] == entries
    assert graph.is_stale(follower.uid) == stale


@pytest.mark.parametrize("duplicate", [False, True])
@pytest.mark.parametrize("dialog_filter", [False, True], ids=["positional", "dialog"])
def test_transposed_filter_preserves_source_and_display(
    console, monkeypatch, qtbot, tmp_path, dialog_filter, duplicate
):
    from erlab._provenance import Follow
    from erlab.interactive.imagetool._provenance._operations import (
        GaussianFilterOperation,
    )
    from erlab.interactive.imagetool.dialogs import GaussianFilterDialog

    manager, tools, base = console
    data = base.copy(data=np.random.default_rng(3456).normal(size=base.shape))
    tools[0].data = data
    tool = manager.get_imagetool(0)
    area = tool.slicer_area
    area.swap_axes(0, 1)
    index = manager.duplicate_imagetool(0) if duplicate else 0
    tool = manager.get_imagetool(index)
    area = tool.slicer_area
    graph = manager._data_graph.graph
    source = Follow(tools[index].uid, source=True)
    source_revision = graph.followed_revision(source)
    if dialog_filter:
        dialog = GaussianFilterDialog(area)
        for dim, sigma in (("x", 0.3), ("y", 1.4)):
            dialog.dim_checks[dim].setChecked(True)
            dialog.sigma_spins[dim].setValue(sigma)
        operation = dialog.filter_operation()
        dialog.accept()
        assert dialog.result()
    else:
        operation = GaussianFilterOperation(sigma=(0.3, 1.4))
        area.apply_filter_operation(operation, emit_edited=True)

    expected = operation.apply(data).T
    xr.testing.assert_identical(area.displayed_data, expected)
    assert graph.followed_revision(source) == source_revision
    xr.testing.assert_identical(
        manager._data_graph.read_input(index, data_role="source").data, data.T
    )
    _assert_replay(manager, tools, index, expected, monkeypatch)
    dataset = tool.to_dataset()
    restored = type(tool).from_dataset(dataset)
    qtbot.addWidget(restored)
    xr.testing.assert_identical(restored.slicer_area.displayed_data, expected)
    pending = manager._workspace_controller.loading.pending
    xr.testing.assert_identical(
        pending._workspace_imagetool_source_data_from_payload(
            dataset, attrs=dataset.attrs, data_role="displayed", name=data.name
        ),
        expected,
    )
    graph.reload(tools[index].uid)
    xr.testing.assert_identical(area.displayed_data, expected)
    _assert_replay(manager, tools, index, expected, monkeypatch)

    workspace = manager._workspace_controller
    path = tmp_path / "filter.itws"
    workspace.saving._save_workspace_document(path)
    assert workspace.loading._load_workspace_file(
        path, replace=True, associate=True, mark_dirty=False, select=False
    )
    _assert_replay(manager, tools, index, expected, monkeypatch)


@pytest.mark.parametrize(
    "expression",
    [
        "h + 1",
        "1 - h",
        "-h",
        "abs(h)",
        "np.add(h, h)",
        "h.mean(dim='x')",
        "h.coarsen(y=2).mean()",
    ],
)
def test_console_arithmetic_captures_xarray_options(console, monkeypatch, expression):
    manager, tools, data = console
    with xr.set_options(keep_attrs=False):
        result = eval(expression, {"h": tools[0], "np": np})  # noqa: S307
        expected = eval(expression, {"h": data, "np": np})  # noqa: S307
    with xr.set_options(keep_attrs=True):
        result.qshow(manager=True)
        _assert_replay(manager, tools, 1, expected, monkeypatch)
        tools[0].data = data + 10
        manager._data_graph.graph.refresh(tools[1].uid)
    with xr.set_options(keep_attrs=False):
        refreshed = eval(expression, {"h": tools[0].data, "np": np})  # noqa: S307
    xr.testing.assert_identical(
        manager.get_imagetool(1).slicer_area.displayed_data, refreshed
    )


def test_batch_squeeze_captures_settings_once(console, monkeypatch):
    from erlab.interactive.imagetool.dialogs import SqueezeDialog

    manager, tools, base = console
    data = base.expand_dims(z=[17.0])
    tools[0].data = data
    manager.add_imagetool(itool(data, manager=False, execute=False), show=False)
    select_tools(manager, [0, 1])
    dialog = SqueezeDialog(manager.get_imagetool(0).slicer_area)
    check = dialog.preflight_data
    dialog.drop_check.setChecked(False)

    def change(value, spec):
        check(value, spec)
        dialog.drop_check.setChecked(True)

    monkeypatch.setattr(dialog, "preflight_data", change)
    assert manager.apply_batch_transform_dialog(dialog, "replace")
    for index in (0, 1):
        expected = data.squeeze(drop=False)
        xr.testing.assert_identical(
            manager.get_imagetool(index).slicer_area.displayed_data, expected
        )
        _assert_replay(manager, tools, index, expected, monkeypatch)


def test_batch_filter_reads_each_source_once(console, monkeypatch):
    from erlab.interactive.imagetool._provenance._operations import (
        GaussianFilterOperation,
    )
    from erlab.interactive.imagetool.dialogs import GaussianFilterDialog

    manager, tools, data = console
    second = data + 10
    tools.itool(second)
    select_tools(manager, [0, 1])
    dialog = GaussianFilterDialog(manager.get_imagetool(0).slicer_area)
    dialog.dim_checks["x"].setChecked(True)
    dialog.sigma_spins["x"].setValue(0.7)
    accepted = dialog.filter_operation()
    apply = GaussianFilterOperation.apply
    inputs = []

    def record(operation, source):
        inputs.append(source)
        dialog.sigma_spins["x"].setValue(1.4)
        return apply(operation, source)

    monkeypatch.setattr(GaussianFilterOperation, "apply", record)
    assert manager.apply_batch_filter_dialog(dialog)
    assert len(inputs) == 2
    for index, (used, source) in enumerate(zip(inputs, (data, second), strict=True)):
        xr.testing.assert_identical(used, source)
        expected = apply(accepted, source)
        xr.testing.assert_identical(
            manager.get_imagetool(index).slicer_area.displayed_data, expected
        )
        _assert_replay(manager, tools, index, expected, monkeypatch)


def test_filter_install_detects_newer_source(console, monkeypatch):
    from erlab.interactive.imagetool._provenance._operations import (
        GaussianFilterOperation,
    )

    manager, tools, data = console
    area = manager.get_imagetool(0).slicer_area
    updated = data + 10
    install = area._install_filter

    def replace_source(*args, **kwargs):
        install(*args, **kwargs)
        area.set_data(updated)

    monkeypatch.setattr(area, "_install_filter", replace_source)
    with pytest.raises(RuntimeError, match="changed during installation"):
        area.apply_filter_operation(GaussianFilterOperation(sigma=1.0))
    xr.testing.assert_identical(area.displayed_data, updated)
    xr.testing.assert_identical(
        manager._data_graph.read_input(tools[0].uid).data, updated
    )


def test_publication_uses_revision_when_gui_operand_disagrees(console, monkeypatch):
    import dataclasses

    from erlab.interactive.imagetool._provenance._operations import RenameOperation

    manager, tools, data = console
    source = manager._data_graph.read_input(0)
    source = dataclasses.replace(source, data=xr.full_like(data, -100))
    result = manager._data_graph.calculate((RenameOperation(name="scan"),), (source,))
    index = manager.add_imagetool(result, show=False)
    expected = data.rename("scan")
    xr.testing.assert_identical(tools[index].data, expected)
    _assert_replay(manager, tools, index, expected, monkeypatch)


def test_bound_shift_uses_captured_attribute_policy(console, monkeypatch):
    manager, tools, data = console
    shifts = data.isel(y=0, drop=True) * 0 + 1
    tools.itool(shifts)
    with xr.set_options(keep_attrs=False):
        expected = erlab.analysis.transform.shift(data, shifts, along="y")
        update = data + 10
        refreshed = erlab.analysis.transform.shift(update, shifts, along="y")
        result = tools.apply(
            erlab.analysis.transform.shift, tools[0], tools[1], along="y"
        )
    with xr.set_options(keep_attrs=True):
        result.qshow(manager=True)
        xr.testing.assert_identical(tools[2].data, expected)
        _assert_replay(manager, tools, 2, expected, monkeypatch)
        tools[0].data = update
        manager._data_graph.graph.refresh(tools[2].uid)
        xr.testing.assert_identical(tools[2].data, refreshed)


def test_float32_reduction_uses_captured_backend(console, monkeypatch):
    manager, tools, _ = console
    data = xr.DataArray(
        np.random.default_rng(393).normal(size=(60, 2, 81)).astype("float32"),
        dims=("x", "z", "y"),
    )
    data.values[0, 0, 0] = np.nan
    tools[0].data = data
    with xr.set_options(use_numbagg=False):
        result = tools[0].mean("y")
        expected = data.mean("y")
    with xr.set_options(use_numbagg=True):
        result.qshow(manager=True)
        xr.testing.assert_identical(tools[1].data, expected)
        _assert_replay(manager, tools, 1, expected, monkeypatch)


def test_layout_reuses_accepted_filter(console, monkeypatch):
    from erlab.interactive.imagetool._provenance._operations import (
        GaussianFilterOperation,
    )

    manager, tools, data = console
    area = manager.get_imagetool(0).slicer_area
    operation = GaussianFilterOperation(sigma={"x": 1.0})
    expected = operation.apply(data)
    count = 0
    apply = GaussianFilterOperation.apply

    def record(op, value):
        nonlocal count
        count += 1
        return apply(op, value)

    monkeypatch.setattr(GaussianFilterOperation, "apply", record)
    area.apply_filter_operation(operation)
    area.swap_axes(0, 1)
    xr.testing.assert_identical(tools[0].data, expected.T)
    area.swap_axes(0, 1)
    xr.testing.assert_identical(tools[0].data, expected)
    assert count == 1
    area.apply_filter_operation(None)
    xr.testing.assert_identical(tools[0].data, data)


def test_raw_replacement_installs_evaluated_assignment(console, monkeypatch):
    from erlab.interactive.imagetool._provenance._operations import AssignAttrsOperation
    from erlab.interactive.imagetool.manager._metadata_editor import (
        MetadataCellEdit,
        MetadataField,
    )

    manager, tools, source = console
    edit = MetadataCellEdit(
        MetadataField(kind="attribute", name="sample"), value="fresh"
    )
    assert manager._metadata_editor.apply_edits({0: (edit,)})
    calls = []
    apply = AssignAttrsOperation.apply

    def record(op, value):
        calls.append(value)
        return apply(op, value)

    monkeypatch.setattr(AssignAttrsOperation, "apply", record)
    data = source + 7
    manager._data_replace([data], [0])
    assert len(calls) == 1
    xr.testing.assert_identical(tools[0].data, data.assign_attrs(sample="fresh"))
    _assert_replay(manager, tools, 0, tools[0].data, monkeypatch)


def test_gui_filter_replays_attribute_policy(console, monkeypatch):
    from erlab.interactive.imagetool._provenance._operations import (
        DivideByCoordOperation,
    )

    manager, tools, source = console
    source = source.rename("scan")
    tools[0].data = source
    area = manager.get_imagetool(0).slicer_area
    op = DivideByCoordOperation(coord_name="gain")
    with xr.set_options(keep_attrs=False):
        expected = op.apply(source)
        area.apply_filter_operation(op)
    with xr.set_options(keep_attrs=True):
        xr.testing.assert_identical(tools[0].data, expected)
        _assert_replay(manager, tools, 0, expected, monkeypatch)
        manager._data_graph.graph.refresh(tools[0].uid)
        xr.testing.assert_identical(tools[0].data, expected)


@pytest.mark.parametrize("operation", ["divide", "normalize"])
def test_broadcast_policy_is_used_for_copied_code_and_refresh(
    console, monkeypatch, operation
):
    from erlab.interactive.imagetool._provenance._operations import (
        DivideByCoordOperation,
        NormalizeOperation,
    )

    manager, tools, data = console
    area = manager.get_imagetool(0).slicer_area
    op = (
        DivideByCoordOperation(coord_name="gain")
        if operation == "divide"
        else NormalizeOperation(dims=("y",))
    )
    with xr.set_options(arithmetic_broadcast=True):
        expected = op.apply(data)
        area.apply_filter_operation(op)
    with xr.set_options(arithmetic_broadcast=False):
        xr.testing.assert_identical(area.displayed_data, expected)
        _assert_replay(manager, tools, 0, expected, monkeypatch)
        manager._data_graph.graph.refresh(tools[0].uid)
        xr.testing.assert_identical(area.displayed_data, expected)


@pytest.mark.parametrize("batch", [False, True])
def test_transform_captures_policy_before_preflight(console, monkeypatch, batch):
    from erlab.interactive.imagetool.dialogs import DivideByCoordDialog

    manager, tools, data = console
    indices = (0, 1) if batch else (0,)
    if batch:
        tools.itool(data)
        select_tools(manager, list(indices))
    dialog = DivideByCoordDialog(manager.get_imagetool(0).slicer_area)
    dialog.coord_combo.setCurrentText("gain")
    dialog.launch_mode_combo.setCurrentIndex(
        dialog.launch_mode_combo.findData("replace")
    )
    check = dialog.preflight_data

    def change(value, spec):
        check(value, spec)
        xr.set_options(keep_attrs=False)

    monkeypatch.setattr(dialog, "preflight_data", change)
    with xr.set_options(keep_attrs=True):
        expected = (data / data.gain).rename(data.name)
        if batch:
            assert manager.apply_batch_transform_dialog(dialog, "replace")
        else:
            dialog.accept()
            assert dialog.result()
        for index in indices:
            xr.testing.assert_identical(tools[index].data, expected)
            _assert_replay(manager, tools, index, expected, monkeypatch)
