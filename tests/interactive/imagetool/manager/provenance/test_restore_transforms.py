import json
import pathlib
import typing
from collections.abc import Callable

import numpy as np
import pytest
import xarray as xr
from qtpy import QtCore, QtWidgets

import erlab
import erlab.interactive.imagetool.manager._details_panel as manager_details_panel
from erlab.interactive.imagetool import itool
from erlab.interactive.imagetool._provenance._model import (
    FileDataSelection,
    ReplayStep,
    ScriptInput,
    ToolProvenanceSpec,
    compose_full_provenance,
    full_data,
    public_data,
    recipe,
    script,
    selection,
)
from erlab.interactive.imagetool._provenance._operations import (
    AssignAttrsOperation,
    IselOperation,
    NormalizeOperation,
    RenameOperation,
    ScriptCodeOperation,
    ShiftFromInputOperation,
    ShiftOperation,
    TransposeOperation,
)
from erlab.interactive.imagetool.manager import fetch, replace_data
from tests.interactive.imagetool.manager.helpers import (
    _exec_generated_code,
    metadata_derivation_texts,
    metadata_detail_map,
    select_child_tool,
    select_metadata_rows,
    select_tools,
    set_transform_launch_mode,
    trigger_menu_action,
)
from tests.interactive.imagetool.manager.provenance._common import (
    _authorize_local_edit,
    _manager_replay_file_spec,
)


@pytest.mark.parametrize(
    ("nested", "multiple", "missing"),
    [
        (False, False, False),
        (True, False, True),
        (True, True, False),
        (False, True, True),
    ],
)
@pytest.mark.parametrize("fallback_role", ["raw", "resolved"])
def test_revert_selected_input_uses_saved_raw_prefix_and_reloads_current_file(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context,
    nested,
    multiple,
    missing,
    fallback_role,
) -> None:
    data = xr.DataArray(
        np.arange(1.0, 25.0).reshape(4, 6),
        dims=("x", "y"),
        coords={"x": np.arange(4), "y": np.arange(6)},
        name="raw",
    )
    path = tmp_path / "raw.nc"
    data.to_netcdf(path, engine="h5netcdf")
    raw_operations = (
        NormalizeOperation(dims=("x",)),
        RenameOperation(name="prepared"),
    )

    def prepare(value):
        for operation in raw_operations:
            value = operation.apply(value)
        return value

    source_spec = selection(
        IselOperation(kwargs={"x": slice(1, 3)}),
        TransposeOperation(dims=("y", "x")),
    )
    prepared = prepare(data)
    selected = source_spec.apply(prepared)
    expected = prepared.isel(x=slice(1, 3))
    workspace = tmp_path / "selected-input.itws"
    with manager_context() as manager:
        raw_index = manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(prepared, _in_manager=True),
            provenance_spec=_manager_replay_file_spec(path, *raw_operations),
            show=False,
        )
        raw = manager._node_for_target(raw_index)
        captured = manager._lineage_controller._script_input_for_node(
            raw, name="data", source_spec=source_spec, data_role="source"
        )
        if fallback_role == "resolved":
            complete = captured.resolved_provenance_spec()
            assert complete is not None
            captured = captured.model_copy(
                update={
                    "fallback_role": "resolved",
                    "provenance_spec": complete.model_dump(mode="json"),
                }
            )
        spec = recipe(
            ReplayStep(operation=AssignAttrsOperation(attrs={"later": True})),
            start_label="Use selected input",
            active_name="derived",
            primary_input="data",
            script_inputs=(captured,),
        )
        output = selected.assign_attrs(later=True)
        intermediates = []
        if nested:
            intermediate = manager.add_imagetool(
                erlab.interactive.imagetool.ImageTool(output, _in_manager=True),
                provenance_spec=spec,
                show=False,
            )
            intermediates.append(intermediate)
            captured = manager._lineage_controller._script_input_for_node(
                manager._node_for_target(intermediate), name="data", data_role="source"
            )
            spec = recipe(
                ReplayStep(operation=AssignAttrsOperation(attrs={"outer": True})),
                start_label="Use nested input",
                active_name="derived",
                primary_input="data",
                script_inputs=(captured,),
            )
            output = output.assign_attrs(outer=True)
        for _ in range(2 if multiple else 1):
            manager.add_imagetool(
                erlab.interactive.imagetool.ImageTool(output, _in_manager=True),
                provenance_spec=spec,
                show=False,
            )
        for index in intermediates:
            manager.remove_imagetool(index)
        manager.remove_imagetool(raw_index)
        if missing:
            path.unlink()
        else:
            (data + 100).to_netcdf(path, engine="h5netcdf", mode="w")
        manager._workspace_controller.saving._save_workspace_document(workspace)

    with manager_context() as manager:
        assert manager._workspace_controller.loading._load_workspace_file(
            workspace, replace=True, associate=True, mark_dirty=False, select=False
        )
        manager._workspace_controller.local_code_edit = _authorize_local_edit
        controller = manager._provenance_edit_controller
        monkeypatch.setattr(controller, "_confirm_revert", lambda: True)
        monkeypatch.setattr(controller, "_show_unavailable", pytest.fail)
        monkeypatch.setattr(
            controller,
            "_show_failed",
            lambda title, error: pytest.fail(f"{title}: {error}"),
        )
        nodes = list(manager._tool_graph.root_wrappers.values())
        targets = []
        for node in nodes:
            row = next(
                row
                for row in node.derivation_display_rows
                if row.replay_ref is not None
                and isinstance(
                    controller._display_spec_for_row(node, row)._operation_for_ref(
                        row.replay_ref
                    ),
                    IselOperation,
                )
            )
            assert len(row.script_input_path) == (2 if nested else 1)
            targets.append((node.uid, row))
        select_tools(manager, [node.index for node in nodes])
        manager._update_info()
        assert all(
            controller._can_revert_row(node, row)[0]
            for node, (_, row) in zip(nodes, targets, strict=True)
        )
        if multiple:
            controller.revert_rows(targets)
        else:
            controller.revert_row(targets[0][1])
        for node in nodes:
            xr.testing.assert_identical(node.slicer_area._data, expected)
            assert node.slicer_area.data.dims == selected.dims
            restored = node.provenance_spec
            assert restored is not None
            assert restored.root_input is not None
            assert restored.root_input.name == "data"
            assert restored.root_input.source_spec is None
            assert restored.root_input.follow_source
        restored_workspace = tmp_path / "restored-input.itws"
        manager._workspace_controller.saving._save_workspace_document(
            restored_workspace
        )

    (data + 200).to_netcdf(path, engine="h5netcdf", mode="w")
    with manager_context() as manager:
        assert manager._workspace_controller.loading._load_workspace_file(
            restored_workspace,
            replace=True,
            associate=True,
            mark_dirty=False,
            select=False,
        )
        for node in list(manager._tool_graph.root_wrappers.values()):
            assert node.materialize_pending_workspace_payload()
            # Workspace data follows the saved slicer layout. The recorded
            # selection and its next replay retain the source dimension order.
            xr.testing.assert_identical(
                node.slicer_area._data, expected.transpose(*selected.dims)
            )
            assert manager._lineage_controller._reload_script_derived_target(node.index)
            xr.testing.assert_identical(
                node.slicer_area._data, prepare(data + 200).isel(x=slice(1, 3))
            )


@pytest.mark.parametrize(
    ("missing", "pin_at"),
    [(False, None), (True, None), (False, "outer"), (True, "inner")],
)
def test_revert_selection_after_closed_source_reloads_uses_latest_raw_checkpoint(
    qtbot, monkeypatch, tmp_path, manager_context, missing, pin_at
) -> None:
    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        lambda _parent, _title, text, **kwargs: pytest.fail(f"{text}: {kwargs}"),
    )
    data = xr.DataArray(np.arange(24.0).reshape(4, 6), dims=("x", "y"))
    path = tmp_path / "source.nc"
    data.to_netcdf(path, engine="h5netcdf")
    source_spec = selection(
        IselOperation(kwargs={"x": slice(1, 3)}),
        IselOperation(kwargs={"y": slice(1, 5)}),
    )
    workspace = tmp_path / "reloaded-input.itws"
    with manager_context() as manager:
        raw_index = manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(data, _in_manager=True),
            provenance_spec=_manager_replay_file_spec(path),
            show=False,
        )
        captured = manager._lineage_controller._script_input_for_node(
            manager._node_for_target(raw_index),
            name="data",
            source_spec=source_spec,
            data_role="source",
        )
        provenance = recipe(
            ReplayStep(operation=AssignAttrsOperation(attrs={"later": True})),
            start_label="Use selected source",
            active_name="derived",
            primary_input="data",
            script_inputs=(captured,),
        )
        target = manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(
                source_spec.apply(data).assign_attrs(later=True), _in_manager=True
            ),
            provenance_spec=provenance,
            show=False,
        )
        manager.remove_imagetool(raw_index)
        node = manager._node_for_target(target)
        for offset in (100, 200):
            (data + offset).to_netcdf(path, engine="h5netcdf", mode="w")
            assert manager._lineage_controller._reload_script_derived_target(target)
            xr.testing.assert_identical(
                node.slicer_area._data,
                source_spec.apply(data + offset).assign_attrs(later=True),
            )
        if pin_at is not None:
            current = node.provenance_spec
            source = current.script_inputs[0]
            if pin_at == "outer":
                source = source.model_copy(update={"follow_source": False})
            else:
                pinned = source.model_copy(
                    update={
                        "follow_source": False,
                        "source_spec": None,
                        "resolved_revision_id": None,
                    }
                )
                fallback = script(
                    start_label="Use saved raw input",
                    active_name="data",
                    root_input=pinned,
                )
                store = manager._provenance_revisions
                source = source.model_copy(
                    update={
                        "revision_id": store.add(
                            store.resolve_data(source.revision_id), fallback
                        ),
                        "provenance_spec": fallback.model_dump(mode="json"),
                    }
                )
            node.set_displayed_provenance(
                current.model_copy(update={"script_inputs": (source,)})
            )
            (data + 250).to_netcdf(path, engine="h5netcdf", mode="w")
            assert manager._lineage_controller._reload_script_derived_target(target)
            xr.testing.assert_identical(
                node.slicer_area._data,
                source_spec.apply(data + 200).assign_attrs(later=True),
            )
        manager._workspace_controller.saving._save_workspace_document(workspace)

    if missing:
        path.unlink()
    else:
        (data + 300).to_netcdf(path, engine="h5netcdf", mode="w")
    with manager_context() as manager:
        assert manager._workspace_controller.loading._load_workspace_file(
            workspace, replace=True, associate=True, mark_dirty=False, select=False
        )
        node = next(iter(manager._tool_graph.root_wrappers.values()))
        controller = manager._provenance_edit_controller
        monkeypatch.setattr(controller, "_confirm_revert", lambda: True)
        monkeypatch.setattr(controller, "_show_unavailable", pytest.fail)
        monkeypatch.setattr(
            controller,
            "_show_failed",
            lambda title, error: pytest.fail(f"{title}: {error}"),
        )
        manager._workspace_controller.local_code_edit = _authorize_local_edit
        select_tools(manager, [node.index])
        manager._update_info()
        row = next(
            row
            for row in node.derivation_display_rows
            if row.replay_ref is not None
            and isinstance(
                controller._display_spec_for_row(node, row)._operation_for_ref(
                    row.replay_ref
                ),
                IselOperation,
            )
        )
        assert controller.can_revert_row(row)[0]
        controller.revert_row(row)
        xr.testing.assert_identical(
            node.slicer_area._data, (data + 200).isel(x=slice(1, 3))
        )
        if pin_at is not None:
            restored_input = node.provenance_spec.root_input
            assert restored_input is not None
            if restored_input.follow_source:
                fallback = restored_input.parsed_provenance_spec()
                assert fallback.root_input is not None
                assert not fallback.root_input.follow_source
        (data + 400).to_netcdf(path, engine="h5netcdf", mode="w")
        assert manager._lineage_controller._reload_target_with_continuations(
            node.index, ()
        )
        xr.testing.assert_identical(
            node.slicer_area._data,
            (data + (400 if pin_at is None else 200)).isel(x=slice(1, 3)),
        )


@pytest.mark.parametrize(
    "source_spec",
    [
        full_data(),
        full_data(RenameOperation(name="raw")),
        public_data(),
        selection(),
        selection(IselOperation(kwargs={"x": slice(1, 3)})),
        selection(RenameOperation(name="raw")),
    ],
)
def test_closed_source_reloads_keep_source_recipe_depth_and_transform_once(
    qtbot, tmp_path, manager_context, source_spec
) -> None:
    data = xr.DataArray(np.arange(24.0).reshape(4, 6), dims=("x", "y"), name="raw")
    path = tmp_path / "source.nc"
    data.to_netcdf(path, engine="h5netcdf")

    def input_depth(spec):
        nested = [
            item.parsed_provenance_spec()
            for item in (() if spec is None else spec.replay_inputs)
        ]
        return 1 + max((input_depth(value) for value in nested), default=0)

    with manager_context() as manager:
        raw = manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(data, _in_manager=True),
            provenance_spec=_manager_replay_file_spec(path),
            show=False,
        )
        source = manager._lineage_controller._script_input_for_node(
            manager._node_for_target(raw), name="data", source_spec=source_spec
        )
        provenance = recipe(
            ReplayStep(operation=AssignAttrsOperation(attrs={"result": True})),
            start_label="Use source",
            active_name="derived",
            primary_input="data",
            script_inputs=(source,),
        )
        target = manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(
                source_spec.apply(data).assign_attrs(result=True), _in_manager=True
            ),
            provenance_spec=provenance,
            show=False,
        )
        manager.remove_imagetool(raw)
        node = manager._node_for_target(target)
        depth = None
        for offset in (100, 200, 300):
            (data + offset).to_netcdf(path, engine="h5netcdf", mode="w")
            assert manager._lineage_controller._reload_script_derived_target(target)
            xr.testing.assert_identical(
                node.slicer_area._data,
                source_spec.apply(data + offset).assign_attrs(result=True),
            )
            if depth is None:
                depth = input_depth(node.provenance_spec)
            assert input_depth(node.provenance_spec) == depth


@pytest.mark.parametrize("approved", [False, True])
def test_selected_input_revert_preserves_exact_raw_revision_code_approval(
    qtbot, monkeypatch, tmp_path, manager_context, approved
) -> None:
    from erlab.interactive._code_trust import untrusted_document_trust
    from erlab.interactive.imagetool._provenance._trust import (
        provenance_code_trust_entries,
    )
    from erlab.interactive.imagetool.manager._workspace._trust import (
        current_workspace_code_trust_manifest,
    )

    data = xr.DataArray(np.arange(24.0).reshape(4, 6), dims=("x", "y"))
    path = tmp_path / "source.nc"
    data.to_netcdf(path, engine="h5netcdf")
    source_spec = selection(
        IselOperation(kwargs={"x": slice(1, 3)}),
        IselOperation(kwargs={"y": slice(1, 5)}),
    )
    raw_provenance = script(
        ScriptCodeOperation(label="Add offset", code="derived = data + 7"),
        start_label="Prepare raw input",
        active_name="derived",
        primary_input="data",
        script_inputs=(
            ScriptInput(name="data", provenance_spec=_manager_replay_file_spec(path)),
        ),
    )
    expected = (data + 7).isel(x=slice(1, 3))
    with manager_context() as manager:
        raw = manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(data + 7, _in_manager=True),
            provenance_spec=raw_provenance,
            show=False,
        )
        captured = manager._lineage_controller._script_input_for_node(
            manager._node_for_target(raw), name="data", source_spec=source_spec
        )
        provenance = recipe(
            ReplayStep(operation=AssignAttrsOperation(attrs={"later": True})),
            start_label="Use selected input",
            active_name="derived",
            primary_input="data",
            script_inputs=(captured,),
        )
        original = source_spec.apply(data + 7).assign_attrs(later=True)
        target = manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(original, _in_manager=True),
            provenance_spec=provenance,
            show=False,
        )
        external = manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(data + 7, _in_manager=True),
            provenance_spec=raw_provenance,
            show=False,
        )
        manager.remove_imagetool(raw)
        workspace = manager._workspace_controller
        workspace._set_workspace_code_trust(
            untrusted_document_trust(current_workspace_code_trust_manifest(manager))
        )
        if approved:
            entries = provenance_code_trust_entries(
                raw_provenance,
                location_prefix=f"revisions/{captured.revision_id}/provenance",
            )
            with workspace.local_code_edit(
                entries, edited_entries=entries
            ) as capability:
                assert capability is not None
        path.unlink()
        node = manager._node_for_target(target)
        controller = manager._provenance_edit_controller
        select_tools(manager, [target])
        manager._update_info()
        row = next(
            row
            for row in node.derivation_display_rows
            if row.replay_ref is not None
            and isinstance(
                controller._display_spec_for_row(node, row)._operation_for_ref(
                    row.replay_ref
                ),
                IselOperation,
            )
        )
        monkeypatch.setattr(controller, "_confirm_revert", lambda: True)
        monkeypatch.setattr(controller, "_show_unavailable", pytest.fail)
        monkeypatch.setattr(
            controller,
            "_show_failed",
            lambda title, error: pytest.fail(f"{title}: {error}"),
        )
        controller.revert_row(row)
        xr.testing.assert_identical(node.slicer_area._data, expected)
        stored_entries = provenance_code_trust_entries(
            node.provenance_spec, location_prefix="runtime"
        )
        assert (
            node.slicer_area._stored_code_authorizer(stored_entries) is not None
        ) is approved
        external_node = manager._node_for_target(external)
        external_entries = provenance_code_trust_entries(
            external_node.provenance_spec, location_prefix="runtime"
        )
        assert (
            external_node.slicer_area._stored_code_authorizer(external_entries) is None
        )
        (data + 100).to_netcdf(path, engine="h5netcdf", mode="w")
        assert (
            manager._lineage_controller._reload_script_derived_target(target)
            is approved
        )
        xr.testing.assert_identical(
            node.slicer_area._data,
            (data + 107).isel(x=slice(1, 3)) if approved else expected,
        )


@pytest.mark.parametrize("approved", [False, True])
@pytest.mark.parametrize("file_backed", [False, True])
def test_relative_restore_keeps_only_its_own_parameter_expression_approval(
    qtbot, tmp_path, manager_context, approved, file_backed
) -> None:
    from erlab.interactive._code_trust import (
        execution_capability_allows,
        untrusted_document_trust,
    )
    from erlab.interactive.imagetool._provenance._operations import (
        ModelFitOperation,
        _ModelFitParameterSpec,
    )
    from erlab.interactive.imagetool._provenance._trust import (
        provenance_code_trust_entries,
    )
    from erlab.interactive.imagetool.manager._workspace._trust import (
        current_workspace_code_trust_manifest,
    )

    data = xr.DataArray(
        np.arange(24.0).reshape(4, 6),
        dims=("y", "x"),
        coords={"x": np.arange(6), "y": np.arange(4)},
    )
    path = tmp_path / "source.nc"
    data.to_netcdf(path, engine="h5netcdf")
    operation = ModelFitOperation(
        fit_dim="x",
        model="PolynomialModel",
        model_kwargs={"degree": 1},
        parameters={
            "c0": _ModelFitParameterSpec(value=0.0),
            "c1": _ModelFitParameterSpec(expr="2 * c0"),
        },
        method="leastsq",
        parameter="c0",
    )
    with manager_context() as manager:
        raw = manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(data, _in_manager=True),
            provenance_spec=_manager_replay_file_spec(path) if file_backed else None,
            show=False,
        )
        source = manager._lineage_controller._script_input_for_node(
            manager._node_for_target(raw),
            name="data",
            source_spec=selection(operation),
        )
        provenance = recipe(
            ReplayStep(operation=AssignAttrsOperation(attrs={"later": True})),
            start_label="Use fit parameter",
            active_name="derived",
            primary_input="data",
            script_inputs=(source,),
        )
        targets = [
            manager.add_imagetool(
                erlab.interactive.imagetool.ImageTool(data, _in_manager=True),
                provenance_spec=provenance,
                show=False,
            )
            for _ in range(2)
        ]
        manager.remove_imagetool(raw)
        controller = manager._provenance_edit_controller
        workspace = manager._workspace_controller
        workspace._set_workspace_code_trust(
            untrusted_document_trust(current_workspace_code_trust_manifest(manager))
        )
        node = manager._node_for_target(targets[0])
        if approved:
            prior = controller._provenance_code_entries(
                node.provenance_spec, node=node, scope="display"
            )
            with workspace.local_code_edit(prior, edited_entries=prior) as capability:
                assert capability is not None
        row = next(
            row
            for row in node.derivation_display_rows
            if row.replay_ref is not None
            and isinstance(
                controller._display_spec_for_row(node, row)._operation_for_ref(
                    row.replay_ref
                ),
                ModelFitOperation,
            )
        )
        prefix = controller._display_spec_for_row(node, row)._prefix_through_ref(
            row.replay_ref
        )
        candidate = controller._revert_candidate(node, row, prefix)
        assert candidate.restored_input_revision is not None
        runtime = controller._provenance_code_entries(
            candidate.spec,
            node=node,
            scope="display",
            resolve_inputs=True,
            prefer_recorded_revisions=True,
        )
        assert runtime
        relocated = controller._approved_recorded_revision_entries((candidate,))
        with workspace.local_code_edit(
            (*runtime, *relocated), edited_entries=relocated
        ) as capability:
            assert execution_capability_allows(capability, runtime) is approved
        other = manager._node_for_target(targets[1])
        other_entries = provenance_code_trust_entries(
            other.provenance_spec, location_prefix="runtime"
        )
        assert other.slicer_area._stored_code_authorizer(other_entries) is None


@pytest.mark.parametrize("approved", [False, True])
@pytest.mark.parametrize("multiple", [False, True])
def test_exact_selected_checkpoint_revert_keeps_pin_without_replaying_fit(
    qtbot, monkeypatch, tmp_path, manager_context, approved, multiple
) -> None:
    from erlab.interactive._code_trust import untrusted_document_trust
    from erlab.interactive.imagetool._provenance._operations import (
        ModelFitOperation,
        _ModelFitParameterSpec,
    )
    from erlab.interactive.imagetool._provenance._trust import (
        provenance_code_trust_entries,
    )
    from erlab.interactive.imagetool.manager._workspace._trust import (
        current_workspace_code_trust_manifest,
    )
    from tests.interactive.imagetool.manager.workspace._support import (
        _current_workspace_manifest,
    )

    data = xr.DataArray(np.arange(48.0).reshape(2, 4, 6), dims=("z", "y", "x"))
    selected = xr.DataArray(np.arange(8.0).reshape(2, 4), dims=("z", "y"))
    path = tmp_path / "source.nc"
    data.to_netcdf(path, engine="h5netcdf")
    operation = ModelFitOperation(
        fit_dim="x",
        model="PolynomialModel",
        model_kwargs={"degree": 1},
        parameters={
            "c0": _ModelFitParameterSpec(value=0.0),
            "c1": _ModelFitParameterSpec(expr="2 * c0"),
        },
        method="leastsq",
        parameter="c0",
    )
    source = selection(operation)
    with manager_context() as manager:
        store = manager._provenance_revisions
        raw_spec = _manager_replay_file_spec(path)
        raw = ScriptInput(
            name="data", revision_id=store.add(data, raw_spec), provenance_spec=raw_spec
        )
        complete = compose_full_provenance(raw_spec, source)
        assert complete is not None
        checkpoint = store.add(selected, complete)
        pinned = ScriptInput(
            name="data",
            revision_id=raw.revision_id,
            resolved_revision_id=checkpoint,
            follow_source=False,
            source_spec=source,
            provenance_spec=raw_spec,
        )
        provenance = recipe(
            ReplayStep(operation=AssignAttrsOperation(attrs={"later": True})),
            start_label="Use saved fit",
            active_name="derived",
            primary_input="data",
            script_inputs=(pinned,),
        )
        targets = [
            manager.add_imagetool(
                erlab.interactive.imagetool.ImageTool(
                    selected.assign_attrs(later=True), _in_manager=True
                ),
                provenance_spec=provenance,
                show=False,
            )
            for _ in range(2 if multiple else 1)
        ]
        workspace = manager._workspace_controller
        workspace._set_workspace_code_trust(
            untrusted_document_trust(current_workspace_code_trust_manifest(manager))
        )
        if approved:
            entries = provenance_code_trust_entries(
                complete, location_prefix=f"revisions/{checkpoint}/provenance"
            )
            with workspace.local_code_edit(
                entries, edited_entries=entries
            ) as capability:
                assert capability is not None
        controller = manager._provenance_edit_controller
        monkeypatch.setattr(controller, "_confirm_revert", lambda: True)
        monkeypatch.setattr(controller, "_show_unavailable", pytest.fail)
        monkeypatch.setattr(
            controller,
            "_show_failed",
            lambda title, error: pytest.fail(f"{title}: {error}"),
        )
        monkeypatch.setattr(
            ModelFitOperation,
            "apply",
            lambda *_args, **_kwargs: pytest.fail(
                "A checkpoint restore replayed the fit"
            ),
        )
        select_tools(manager, targets)
        manager._update_info()
        nodes = [manager._node_for_target(target) for target in targets]
        rows = []
        for node in nodes:
            row = next(
                row
                for row in node.derivation_display_rows
                if row.replay_ref is not None
                and isinstance(
                    controller._display_spec_for_row(node, row)._operation_for_ref(
                        row.replay_ref
                    ),
                    ModelFitOperation,
                )
            )
            assert (
                controller._recorded_revision_for_row(node, row).revision_id
                == checkpoint
            )
            rows.append((node.uid, row))
        if multiple:
            controller.revert_rows(rows)
        else:
            controller.revert_row(rows[0][1])
        for node in nodes:
            xr.testing.assert_identical(node.slicer_area._data, selected)
            assert node.provenance_spec.root_input is not None
            assert not node.provenance_spec.root_input.follow_source
            assert node.provenance_spec.root_input.source_spec is None
            assert node.current_revision_id != checkpoint
            store.bind(node.current_revision_id, node.provenance_spec)
            current_row = next(
                row
                for row in node.derivation_display_rows
                if row.replay_ref is not None
                and isinstance(
                    controller._display_spec_for_row(node, row)._operation_for_ref(
                        row.replay_ref
                    ),
                    ModelFitOperation,
                )
            )
            assert not controller._can_revert_row(node, current_row)[0]
            entries = provenance_code_trust_entries(
                node.provenance_spec, location_prefix="runtime"
            )
            assert (
                node.slicer_area._stored_code_authorizer(entries) is not None
            ) is approved
        later_rows = []
        for node in nodes:
            node.replace_with_detached_data(
                selected.assign_attrs(later=True),
                node.provenance_spec.append_operations(
                    AssignAttrsOperation(attrs={"later": True})
                ),
                replay_source_data=None,
            )
            row = next(
                row
                for row in node.derivation_display_rows
                if row.replay_ref is not None
                and isinstance(
                    controller._display_spec_for_row(node, row)._operation_for_ref(
                        row.replay_ref
                    ),
                    ModelFitOperation,
                )
            )
            assert controller._can_revert_row(node, row) == (True, "")
            later_rows.append((node.uid, row))
        if multiple:
            controller.revert_rows(later_rows)
        else:
            controller.revert_row(later_rows[0][1])
        for node in nodes:
            xr.testing.assert_identical(node.slicer_area._data, selected)
            saved_input = node.provenance_spec.root_input
            assert saved_input is not None
            assert not saved_input.follow_source
            assert saved_input.revision_id == checkpoint
            assert saved_input.source_spec is None
            store.bind(node.current_revision_id, node.provenance_spec)
            entries = provenance_code_trust_entries(
                node.provenance_spec, location_prefix="runtime"
            )
            assert (
                node.slicer_area._stored_code_authorizer(entries) is not None
            ) is approved
        saved = tmp_path / "pinned-fit.itws"
        workspace.saving._save_workspace_document(saved)
        assert checkpoint in {
            entry["revision_id"]
            for entry in _current_workspace_manifest(saved)["provenance_revisions"]
        }
        (data + 100).to_netcdf(path, engine="h5netcdf", mode="w")
        for node in nodes:
            assert manager._lineage_controller._reload_script_derived_target(node.index)
            xr.testing.assert_identical(node.slicer_area._data, selected)


def test_recorded_resolved_input_with_nan_parameters_applies_transform_once(
    manager_context,
) -> None:
    data = xr.DataArray(np.arange(5.0), dims="x", coords={"x": np.arange(5.0)})
    source_spec = full_data(ShiftOperation(shift=1.0, along="x"))
    resolved = source_spec.apply(data)
    with manager_context() as manager:
        store = manager._provenance_revisions
        raw_id = store.add(data)
        recipe = store.normalize_provenance(
            source_spec, replay_source_revision_id=raw_id
        )
        revision_id = store.add(resolved, recipe)
        source = ScriptInput(
            name="data",
            revision_id=raw_id,
            resolved_revision_id=revision_id,
            source_spec=source_spec,
        )
        recorded, _source = manager._lineage_controller._resolve_recorded_script_input(
            source
        )
        xr.testing.assert_identical(recorded, resolved)


def test_manager_mixed_revert_restores_checkpoint_and_rebuilds_legacy_recipe(
    qtbot, monkeypatch, tmp_path, manager_context
) -> None:
    data = xr.DataArray(
        np.arange(1.0, 16.0).reshape(3, 5),
        dims=("x", "eV"),
        coords={"x": np.arange(3), "eV": np.linspace(-0.2, 0.2, 5)},
    )
    operation = NormalizeOperation(dims=("x",))
    paths = [tmp_path / "recorded.nc", tmp_path / "legacy.nc"]
    for path in paths:
        data.to_netcdf(path, engine="h5netcdf")
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager._workspace_controller.local_code_edit = _authorize_local_edit
        recorded = manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(data, _in_manager=True),
            provenance_spec=_manager_replay_file_spec(paths[0]),
            show=False,
        )
        node = manager._node_for_target(recorded)
        node.replace_with_detached_data(
            operation.apply(data),
            _manager_replay_file_spec(paths[0], operation),
            replay_source_data=None,
        )
        legacy = manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(
                operation.apply(data), _in_manager=True
            ),
            provenance_spec=_manager_replay_file_spec(paths[1], operation),
            show=False,
        )
        for path in paths:
            (data + 100).to_netcdf(path, engine="h5netcdf", mode="w")
        legacy_node = manager._node_for_target(legacy)
        controller = manager._provenance_edit_controller
        monkeypatch.setattr(controller, "_confirm_revert", lambda: True)
        monkeypatch.setattr(
            controller,
            "_show_failed",
            lambda _title, error: (_ for _ in ()).throw(error),
        )
        select_tools(manager, [recorded, legacy])
        controller._apply_multiple_rows(
            [
                (node.uid, node.derivation_display_rows[0]),
                (legacy_node.uid, legacy_node.derivation_display_rows[0]),
            ],
            action="revert",
        )
        xr.testing.assert_identical(node.data_for_role("source"), data)
        xr.testing.assert_identical(legacy_node.data_for_role("source"), data + 100)


def test_manager_checkpoint_batch_materializes_all_targets_before_restoring(
    qtbot, monkeypatch, tmp_path, manager_context
) -> None:
    data = xr.DataArray(np.arange(6.0).reshape(2, 3), dims=("x", "y"))
    path = tmp_path / "source.nc"
    data.to_netcdf(path, engine="h5netcdf")
    with manager_context() as manager:
        nodes = []
        target_rows = []
        for _ in range(2):
            index = manager.add_imagetool(
                erlab.interactive.imagetool.ImageTool(data, _in_manager=True),
                provenance_spec=_manager_replay_file_spec(path),
                show=False,
            )
            node = manager._node_for_target(index)
            normalize = NormalizeOperation(dims=("x",))
            node.replace_with_detached_data(
                normalize.apply(data),
                _manager_replay_file_spec(path, normalize),
                replay_source_data=None,
            )
            node.revision_id_for_role("source")
            node.replace_with_detached_data(
                data + 10,
                _manager_replay_file_spec(
                    path, normalize, AssignAttrsOperation(attrs={"current": True})
                ),
                replay_source_data=None,
            )
            nodes.append(node)
            row = next(
                row
                for row in node.derivation_display_rows
                if row.replay_ref is not None
                and row.replay_ref.kind == "operation"
                and row.replay_ref.operation_index == 0
            )
            target_rows.append((node.uid, row))
        editor = manager._provenance_edit_controller
        monkeypatch.setattr(editor, "_confirm_revert", lambda: True)
        failures = []
        monkeypatch.setattr(editor, "_show_failed", lambda *args: failures.append(args))
        monkeypatch.setattr(editor, "_show_unavailable", pytest.fail)
        monkeypatch.setattr(
            nodes[1], "materialize_pending_workspace_payload", lambda: False
        )
        select_tools(manager, [node.index for node in nodes])
        manager._update_info()

        editor.revert_rows(target_rows)

        assert len(failures) == 1
        for node in nodes:
            xr.testing.assert_identical(node.current_source_data(), data + 10)


@pytest.mark.parametrize("lifecycle", ["original", "duplicate", "promote"])
def test_manager_repeated_shift_restores_primary_checkpoints_after_sources_close(
    qtbot, monkeypatch, manager_context, lifecycle
) -> None:
    data = xr.DataArray(
        np.arange(1.0, 16.0).reshape(3, 5),
        dims=("x", "eV"),
        coords={"x": np.arange(3), "eV": np.linspace(-0.2, 0.2, 5)},
        name="sample",
    )
    correction = xr.DataArray([0.0, 0.05, -0.05], dims="x", coords={"x": np.arange(3)})
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        target = manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(data, _in_manager=True), show=False
        )
        if lifecycle == "promote":
            target = manager.add_imagetool_child(
                erlab.interactive.imagetool.ImageTool(data, _in_manager=True),
                target,
                show=False,
            )
        auxiliary = manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(correction, _in_manager=True),
            show=False,
        )
        node = manager._node_for_target(target)
        operation = NormalizeOperation(dims=("x",))
        normalized = operation.apply(data)
        node.replace_with_detached_data(
            normalized, full_data(operation), replay_source_data=data
        )
        normalized_revision = node.current_revision_id
        normalized_spec = node.provenance_spec
        for _ in range(2):
            current = node.data_for_role("source")
            spec = manager._multi_input_operation_provenance(
                (target, auxiliary),
                operation=ShiftFromInputOperation(along="eV", negate=False),
                active_name="shifted",
                start_label="Shift ImageTool data",
                detached_input_uid=node.uid,
                data_role="source",
                input_names=("data", "shift"),
                primary_input="data",
                input_bindings={"shift": "shift"},
            )
            shifted = erlab.analysis.transform.shift(current, correction, along="eV")
            node.replace_with_detached_data(shifted, spec, replay_source_data=current)
        manager._remove_imagetool(auxiliary)
        if lifecycle == "duplicate":
            original = target
            target = manager.duplicate_imagetool(target)
            manager._remove_imagetool(original)
            node = manager._node_for_target(target)
        elif lifecycle == "promote":
            target = manager.promote_child_imagetool(node.uid)
            node = manager._node_for_target(target)
        manager._metadata_node_uid = node.uid
        controller = manager._provenance_edit_controller
        monkeypatch.setattr(controller, "_confirm_revert", lambda: True)
        monkeypatch.setattr(
            manager._lineage_controller,
            "_authorize_provenance_execution",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("exact restoration must not execute provenance")
            ),
        )
        row = next(
            row
            for row in node.derivation_display_rows
            if row.revision_id == normalized_revision
        )
        assert controller.can_revert_row(row) == (True, "")
        controller.revert_row(row)
        xr.testing.assert_identical(node.data_for_role("source"), normalized)
        store = manager._provenance_revisions
        assert node.current_revision_id == normalized_revision
        assert node.provenance_spec == normalized_spec
        store.bind(node.current_revision_id, node.provenance_spec)
        saved_input = node.provenance_spec.root_input
        assert saved_input is not None
        assert not saved_input.follow_source
        assert saved_input.source_spec is None
        xr.testing.assert_identical(store.resolve_data(normalized_revision), normalized)
        saved_spec = saved_input.parsed_provenance_spec()
        store.bind(saved_input.revision_id, saved_spec)
        assert any(
            isinstance(step.operation, NormalizeOperation)
            for step in node.provenance_spec.steps
        )
        assert all(
            not isinstance(step.operation, ShiftFromInputOperation)
            for step in node.provenance_spec.steps
        )
        for index in tuple(manager._tool_graph.root_wrappers):
            manager._remove_imagetool(index)
        assert manager._provenance_revisions.revision_ids == frozenset()


def test_manager_recorded_input_edit_uses_captured_data_and_open_input_is_independent(
    qtbot, manager_context
) -> None:
    data = xr.DataArray(
        np.arange(15.0).reshape(3, 5),
        dims=("x", "eV"),
        coords={"x": np.arange(3), "eV": np.linspace(-0.2, 0.2, 5)},
    )
    correction = xr.DataArray([0.0, 0.05, -0.05], dims="x", coords={"x": np.arange(3)})
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        target = manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(data, _in_manager=True), show=False
        )
        auxiliary = manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(correction, _in_manager=True),
            show=False,
        )
        node = manager._node_for_target(target)
        spec = manager._multi_input_operation_provenance(
            (target, auxiliary),
            operation=ShiftFromInputOperation(along="eV", negate=False),
            active_name="shifted",
            start_label="Shift ImageTool data",
            detached_input_uid=node.uid,
            data_role="source",
            input_names=("data", "shift"),
            primary_input="data",
            input_bindings={"shift": "shift"},
        )
        expected = erlab.analysis.transform.shift(data, correction, along="eV")
        node.replace_with_detached_data(expected, spec, replay_source_data=data)
        parent = manager._node_for_target(auxiliary)
        parent.replace_with_detached_data(
            correction + 0.1, None, replay_source_data=None
        )
        parent._set_source_state("stale")
        pinned_input = spec.script_inputs[1].model_copy(update={"follow_source": False})
        plan = manager._lineage_controller._input_resolution_plan(
            (pinned_input,), target_node_uid=node.uid, force_live_reload=True
        )
        assert plan.unavailable_reason is None
        assert plan.actions == ()
        controller = manager._provenance_edit_controller
        replayed = controller._replay_candidate_result(node, "display", spec).data
        xr.testing.assert_identical(replayed, expected)
        manager._metadata_node_uid = node.uid
        row = node.derivation_display_rows[-1].materialized_children()[0]
        assert not row.on_primary_chain
        assert not controller.can_revert_row(row)[0]
        assert controller.can_open_input_row(row) == (True, "")
        previous_count = manager.ntools
        controller.open_input_row(row)
        assert manager.ntools == previous_count + 1
        opened = manager._node_for_target(manager._tool_graph.next_index - 1)
        xr.testing.assert_identical(opened.current_public_data(), correction)
        opened.slicer_area.data.values[:] = 100.0
        xr.testing.assert_identical(
            manager._provenance_revisions.resolve_data(
                spec.script_inputs[1].revision_id
            ),
            correction,
        )


@pytest.mark.parametrize("nonuniform", [False, True])
def test_manager_detached_file_provenance_metadata_and_reload_roundtrip(
    qtbot,
    accept_dialog,
    tmp_path: pathlib.Path,
    test_data,
    nonuniform: bool,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    if nonuniform:
        test_data = test_data.assign_coords(alpha=[0.0, 0.25, 1.1, 3.0, 10.0])
    file_path = tmp_path / "scan.h5"
    test_data.to_netcdf(file_path, engine="h5netcdf")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(
            test_data,
            manager=True,
            file_path=file_path,
            load_func=(
                xr.load_dataarray,
                {"engine": "h5netcdf"},
                FileDataSelection(kind="dataarray"),
            ),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        root_tool = manager.get_imagetool(0)

        def _detach_average(dialog) -> None:
            dialog.dim_checks["alpha"].setChecked(True)
            set_transform_launch_mode(dialog, "detach")

        accept_dialog(root_tool.mnb._average, pre_call=_detach_average)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        tree = manager._workspace_controller.saving._to_datatree()
        provenance_payload = json.loads(
            tree["1/imagetool"].attrs["manager_node_provenance_spec"]
        )
        assert provenance_payload["schema_version"] == 3
        assert provenance_payload["kind"] == "file"
        assert provenance_payload["operations"] == []
        assert len(provenance_payload["steps"]) == 1
        assert provenance_payload["steps"][0]["input_policy"] == "restored"
        assert provenance_payload["steps"][0]["operation"]["op"] == "qsel_aggregate"
        assert (
            provenance_payload["file_load_source"]["replay_call"]["target"]
            == "xarray.load_dataarray"
        )
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        for node in tree.values():
            manager._workspace_controller.loading._load_workspace_node(
                typing.cast("xr.DataTree", node)
            )

        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        detached = manager._tool_graph.root_wrappers[1]
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
            updated.astype(np.float64).qsel.mean("alpha").rename(None),
        )

        file_path.unlink()
        manager._update_actions()
        assert not detached.reloadable
        assert manager.reload_action.isVisible()
        assert manager.reload_action.isEnabled()
        assert str(file_path) in manager.reload_action.toolTip()


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

        tree = manager._workspace_controller.saving._to_datatree()
        tree["0/imagetool"].attrs["manager_node_provenance_spec"] = json.dumps(
            legacy_payload
        )
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        for node in tree.values():
            manager._workspace_controller.loading._load_workspace_node(
                typing.cast("xr.DataTree", node)
            )

        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        loaded = manager._tool_graph.root_wrappers[0]
        assert loaded.provenance_spec is not None
        assert loaded.provenance_spec.schema_version == 3
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

        parent = manager._tool_graph.root_wrappers[0]
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
            set_transform_launch_mode(dialog, "nest")
            assert dialog.launch_mode_combo.toolTip()
            dialog.dim_checks["x"].setChecked(True)

        accept_dialog(parent_tool.mnb._average, pre_call=_nest_average)

        parent = manager._tool_graph.root_wrappers[0]
        qtbot.wait_until(lambda: len(parent._childtool_indices) == 1, timeout=5000)

        child_uid = parent._childtool_indices[0]
        child_node = manager._child_node(child_uid)
        child_tool = manager.get_imagetool(child_uid)
        xr.testing.assert_identical(
            child_tool.slicer_area._data.rename(None),
            data.qsel.mean("x").rename(None),
        )

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)
        details = metadata_detail_map(manager)
        assert details["Kind"] == "ImageTool"
        assert "Added" in details
        derivation = metadata_derivation_texts(manager)
        assert any("Aggregate" in line for line in derivation)
        assert not any("rename(" in line for line in derivation)

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
                "ToolProvenanceSpec",
                child_node.source_spec,
            ).operations
            if op.op == "qsel_aggregate"
        ]
        assert [op.op for op in transforms] == ["qsel_aggregate", "qsel_aggregate"]
        xr.testing.assert_identical(
            child_tool.slicer_area._data.rename(None),
            data.qsel.mean("x").qsel.mean("y").rename(None),
        )

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)
        derivation = metadata_derivation_texts(manager)
        assert derivation[0] == "Start from current parent ImageTool data"
        assert len(derivation) == 3
        assert "Aggregate" in derivation[1]
        assert "dims=" in derivation[1]
        assert "Aggregate" in derivation[2]
        assert "dims=" in derivation[2]
        manager.metadata_derivation_list.setFocus()
        clipboard = QtWidgets.QApplication.clipboard()
        select_metadata_rows(manager, [0])
        clipboard.clear()
        qtbot.keyClick(
            manager.metadata_derivation_list,
            QtCore.Qt.Key.Key_C,
            QtCore.Qt.KeyboardModifier.ControlModifier,
        )
        assert copied == []
        assert clipboard.text() == ""

        select_metadata_rows(manager, [1, 2])
        qtbot.keyClick(
            manager.metadata_derivation_list,
            QtCore.Qt.Key.Key_C,
            QtCore.Qt.KeyboardModifier.ControlModifier,
        )
        selected_namespace = _exec_generated_code(
            clipboard.text(),
            {"data": data.copy(deep=True)},
        )
        assert clipboard.mimeData().hasFormat(
            manager_details_panel._PROVENANCE_STEPS_CLIPBOARD_MIME
        )
        selected_result = selected_namespace["derived"]
        assert isinstance(selected_result, xr.DataArray)
        xr.testing.assert_identical(
            selected_result.rename(None),
            data.qsel.mean("x").qsel.mean("y").rename(None),
        )

        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        trigger_menu_action(menu, manager._metadata_copy_selected_action)
        selected_namespace = _exec_generated_code(
            clipboard.text(),
            {"data": data.copy(deep=True)},
        )
        selected_result = selected_namespace["derived"]
        assert isinstance(selected_result, xr.DataArray)
        xr.testing.assert_identical(
            selected_result.rename(None),
            data.qsel.mean("x").qsel.mean("y").rename(None),
        )

        monkeypatch.setattr(
            manager,
            "_prompt_replay_input_name",
            lambda _node: "source_data",
        )
        trigger_menu_action(menu, manager._metadata_copy_full_action)
        full_namespace = _exec_generated_code(
            copied[-1],
            {"source_data": data.copy(deep=True)},
        )
        full_result = full_namespace["derived"]
        assert isinstance(full_result, xr.DataArray)
        xr.testing.assert_identical(
            full_result.rename(None),
            data.qsel.mean("x").qsel.mean("y").rename(None),
        )

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
            updated.qsel.mean("x").qsel.mean("y").rename(None),
        )

        def _detach_average(dialog) -> None:
            dialog.dim_checks["x"].setChecked(True)
            set_transform_launch_mode(dialog, "detach")

        accept_dialog(parent_tool.mnb._average, pre_call=_detach_average)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        detached = manager._tool_graph.root_wrappers[1]
        assert detached.source_spec is None
        assert detached.provenance_spec is not None
        assert detached.replay_source_data is None
        assert detached.provenance_spec.root_input is not None
        seed_revision = detached.provenance_spec.root_input.revision_id
        assert manager._provenance_revisions.has_data(seed_revision)
        detached_tool = manager.get_imagetool(1)
        detached_derivation_before = detached.provenance_spec.derivation_code()

        def _replace_detached_average(dialog) -> None:
            dialog.dim_checks["y"].setChecked(True)
            set_transform_launch_mode(dialog, "replace")

        accept_dialog(detached_tool.mnb._average, pre_call=_replace_detached_average)
        assert detached.source_spec is None
        assert detached.provenance_spec is not None
        detached_transforms = [
            op
            for op in detached.provenance_spec.operations
            if op.op == "qsel_aggregate"
        ]
        assert [op.op for op in detached_transforms] == [
            "qsel_aggregate",
            "qsel_aggregate",
        ]
        detached_derivation = detached.provenance_spec.derivation_code()
        assert detached_derivation.count("derived =") == 1
        detached_namespace = _exec_generated_code(
            detached_derivation,
            {"data": updated.copy(deep=True)},
        )
        detached_result = detached_namespace["derived"]
        assert isinstance(detached_result, xr.DataArray)
        xr.testing.assert_identical(
            detached_result.rename(None),
            updated.qsel.mean("x").qsel.mean("y").rename(None),
        )
        assert detached.provenance_spec.derivation_code() != detached_derivation_before
        xr.testing.assert_identical(
            detached_tool.slicer_area._data.rename(None),
            updated.qsel.mean("x").qsel.mean("y").rename(None),
        )
        detached_before = detached_tool.slicer_area._data.copy(deep=True)

        manager.tree_view.clearSelection()
        select_tools(manager, [1])
        manager._update_info()
        detached_derivation = metadata_derivation_texts(manager)
        assert detached_derivation[0] == "Use data from Saved"
        assert len(detached_derivation) == 3
        assert "Aggregate" in detached_derivation[1]
        assert "Aggregate" in detached_derivation[2]

        duplicated_detached_index = typing.cast("int", manager.duplicate_imagetool(1))
        duplicated_detached = manager._tool_graph.root_wrappers[
            duplicated_detached_index
        ]
        assert duplicated_detached.source_spec is None
        assert duplicated_detached.provenance_spec == detached.provenance_spec
        assert duplicated_detached.replay_source_data is None
        assert duplicated_detached.current_revision_id == detached.current_revision_id
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
        assert manager.metadata_derivation_list.topLevelItemCount() == 1
        assert manager.metadata_derivation_list.topLevelItem(0).isDisabled()
        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        assert not manager._metadata_edit_step_action.isEnabled()
        assert not manager._metadata_revert_step_action.isEnabled()
        assert not manager._metadata_delete_step_action.isEnabled()


def test_reload_output_failure_precedes_node_and_capture_adoption(
    qtbot, manager_context
) -> None:
    import dask.array as da
    from dask import delayed

    from erlab.interactive.imagetool.manager._widgets import _ProvenanceReplayResult

    data = xr.DataArray(np.arange(6.0).reshape(2, 3), dims=("x", "y"))

    @delayed
    def fail():
        raise OSError("output source unavailable")

    output = data.copy(data=da.from_delayed(fail(), shape=(2, 3), dtype=float))
    with manager_context() as manager:
        index = manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(data), show=False
        )
        node = manager._node_for_target(index)
        original_spec = node.provenance_spec
        original_snapshot = node.snapshot_token
        store = manager._provenance_revisions
        revisions = store.stage()
        revisions.add(data)
        result = _ProvenanceReplayResult(output, full_data(), revisions)
        initial = store.revision_ids
        with pytest.raises(OSError, match="output source unavailable"):
            manager._lineage_controller._replace_script_reload_target(node, result)
        assert store.revision_ids == initial
        assert node.provenance_spec == original_spec
        assert node.snapshot_token == original_snapshot
        xr.testing.assert_identical(node.current_source_data(), data)
        assert isinstance(output.data, da.Array)


@pytest.mark.parametrize("choice", ["cancel", "new", "replace"])
def test_reloaded_fallback_captures_exist_before_publication(
    qtbot, monkeypatch, manager_context, choice: str
) -> None:
    current = xr.DataArray(
        np.zeros((2, 2)), dims=("x", "y"), coords={"x": [0.0, 1.0], "y": [0.0, 1.0]}
    )
    expected = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("x", "y"),
        coords={"x": [10.0, 11.0], "y": [10.0, 11.0, 12.0]},
    )
    with manager_context() as manager:
        store = manager._provenance_revisions
        raw = store.add(expected)
        fallback = script(
            start_label="Recorded input",
            primary_input="raw",
            active_name="raw",
            script_inputs=(
                ScriptInput(name="raw", revision_id=raw, follow_source=False),
            ),
        )
        spec = script(
            start_label="Recorded result",
            primary_input="data",
            active_name="data",
            script_inputs=(ScriptInput(name="data", provenance_spec=fallback),),
        )
        index = manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(current),
            provenance_spec=spec,
            show=False,
        )
        node = manager._node_for_target(index)
        monkeypatch.setattr(
            manager._lineage_controller,
            "_prompt_incompatible_reload_commit",
            lambda _details: choice,
        )
        observed: list[str] = []

        def check_published_inputs():
            for published in manager._tool_graph.nodes.values():
                published_spec = published.provenance_spec
                if published_spec is None:
                    continue
                for item in published_spec.replay_inputs:
                    for revision_id in (item.revision_id, item.resolved_revision_id):
                        if revision_id is not None:
                            assert store.has_data(revision_id)
                            observed.append(revision_id)

        manager._sigReloadLinkers.connect(check_published_inputs)
        initial = store.revision_ids
        assert manager._lineage_controller._reload_script_derived_target(index) is (
            choice != "cancel"
        )
        if choice == "cancel":
            assert store.revision_ids == initial
            xr.testing.assert_identical(node.current_source_data(), current)
        else:
            target = manager._node_for_target(1) if choice == "new" else node
            captured = target.provenance_spec.script_inputs[0].revision_id
            assert captured is not None
            assert store.has_data(captured)
            xr.testing.assert_identical(store.resolve_data(captured), expected)
            xr.testing.assert_identical(target.current_source_data(), expected)
            if choice == "new":
                assert captured in observed
                xr.testing.assert_identical(node.current_source_data(), current)


@pytest.mark.parametrize("valid", [True, False])
def test_provenance_edit_captures_follow_validation_and_adoption(
    qtbot, manager_context, valid: bool
) -> None:
    from erlab.interactive.imagetool.manager._provenance_edit._controller import (
        _ProvenanceReplayFailure,
    )

    data = xr.DataArray(np.arange(6.0).reshape(2, 3), dims=("x", "y"))
    with manager_context() as manager:
        index = manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(data), show=False
        )
        node = manager._node_for_target(index)
        store = manager._provenance_revisions
        raw = store.add(data + 1 if valid else xr.DataArray(1.0))
        fallback = script(
            start_label="Recorded input",
            primary_input="raw",
            active_name="raw",
            script_inputs=(
                ScriptInput(name="raw", revision_id=raw, follow_source=False),
            ),
        )
        candidate = script(
            start_label="Recorded result",
            primary_input="data",
            active_name="data",
            script_inputs=(ScriptInput(name="data", provenance_spec=fallback),),
        )
        controller = manager._provenance_edit_controller
        initial = store.revision_ids
        preview = controller._replay_candidate_result(node, "display", candidate)
        assert preview.revisions is not None
        assert store.revision_ids == initial
        if not valid:
            with pytest.raises(_ProvenanceReplayFailure):
                controller._validated_edit(
                    node, "display", candidate, where="validating the edit"
                )
            assert store.revision_ids == initial
            xr.testing.assert_identical(node.current_source_data(), data)
            return
        edit = controller._validated_edit(
            node, "display", candidate, where="validating the edit"
        )
        assert store.revision_ids == initial
        controller._apply_validated_edit(edit)
        captured = node.provenance_spec.script_inputs[0].revision_id
        assert captured is not None
        assert store.has_data(captured)
        xr.testing.assert_identical(store.resolve_data(captured), data + 1)
        xr.testing.assert_identical(node.current_source_data(), data + 1)
