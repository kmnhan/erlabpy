import contextlib
import datetime
import errno
import hashlib
import json
import os
import pathlib
import sys
import time
import types
import typing
import warnings
from collections.abc import Callable

import h5py
import numpy as np
import pytest
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

import erlab
import erlab.interactive.imagetool.manager._widgets as manager_widgets
import erlab.interactive.imagetool.manager._workspace._arrays as workspace_arrays
import erlab.interactive.imagetool.manager._workspace._format as workspace_format
import erlab.interactive.imagetool.manager._workspace._saving as workspace_saving
import erlab.interactive.imagetool.manager._workspace._state as workspace_state
import erlab.interactive.imagetool.manager._workspace._storage as workspace_storage
import erlab.interactive.imagetool.manager._workspace._store as workspace_store
import erlab.interactive.imagetool.viewer as imagetool_viewer
from erlab.interactive._code_trust import new_document_trust
from erlab.interactive._options.schema import AppOptions
from erlab.interactive.derivative import DerivativeTool
from erlab.interactive.imagetool import itool
from erlab.interactive.imagetool._mainwindow import _ITOOL_DATA_NAME
from erlab.interactive.imagetool._provenance._model import ScriptInput, full_data
from erlab.interactive.imagetool._provenance._operations import (
    ImageToolSelectionSourceBinding,
    IselOperation,
)
from erlab.interactive.imagetool.manager import ImageToolManager
from erlab.interactive.imagetool.manager._extensions._models import (
    _WorkspaceScriptRequirement,
)
from erlab.interactive.imagetool.manager._workspace import (
    _controller as workspace_controller,
)
from erlab.interactive.kspace import KspaceTool
from tests.interactive.imagetool.manager.helpers import (
    action_map_by_object_name,
    adopt_workspace_path,
    select_child_tool,
    select_tools,
)
from tests.interactive.imagetool.manager.workspace._support import (
    _AddedTimeChildTool,
    _assert_no_workspace_internal_groups,
    _assert_rich_workspace_attr,
    _compute_first_value,
    _current_workspace_manifest,
    _current_workspace_payload_attrs,
    _current_workspace_payload_path,
    _open_external_file_backed_hdf5_imagetool_data,
    _open_external_lazy_hdf5_imagetool_data,
    _request_workspace_save_and_wait,
    _request_workspace_save_as_and_wait,
    _rich_workspace_attr_value,
    _write_transaction_test_workspace,
    add_source_childtool,
)


def test_manager_workspace_saves_added_time_for_all_node_kinds(
    qtbot,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    root_added = datetime.datetime(
        2024, 1, 2, 3, 4, 5, tzinfo=datetime.timezone(datetime.timedelta(hours=9))
    )
    child_added = datetime.datetime(
        2024, 1, 3, 4, 5, 6, tzinfo=datetime.timezone(datetime.timedelta(hours=-5))
    )
    tool_added = datetime.datetime(2024, 1, 4, 5, 6, 7, tzinfo=datetime.UTC)

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        root_index = manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(test_data, _in_manager=True),
            show=False,
            created_time=root_added,
        )
        child_uid = manager.add_imagetool_child(
            erlab.interactive.imagetool.ImageTool(test_data + 1, _in_manager=True),
            root_index,
            show=False,
            created_time=child_added,
        )
        tool_uid = add_source_childtool(
            manager,
            _AddedTimeChildTool(test_data),
            root_index,
            show=False,
            created_time=tool_added,
        )

        fname = tmp_path / "added-time.itws"
        manager._workspace_controller.saving._save_workspace_document(fname)

    attrs = workspace_arrays._read_workspace_root_attrs_h5py(fname)
    manifest = workspace_format._workspace_manifest_from_attrs(attrs)
    attrs_by_uid = {
        str(entry["uid"]): workspace_format._restore_workspace_manifest_attrs(
            entry["payload_attrs"]
        )
        for entry in workspace_format._iter_workspace_manifest_node_entries(manifest)
    }
    root_uid = next(
        str(entry["uid"])
        for entry in workspace_format._iter_workspace_manifest_node_entries(manifest)
        if entry["path"] == "0"
    )
    assert attrs_by_uid[root_uid]["manager_node_added_at"] == root_added.isoformat(
        timespec="seconds"
    )
    assert attrs_by_uid[child_uid]["manager_node_added_at"] == child_added.isoformat(
        timespec="seconds"
    )
    assert attrs_by_uid[tool_uid]["manager_node_added_at"] == tool_added.isoformat(
        timespec="seconds"
    )


def test_manager_workspace_restores_hidden_ktool_angle_scales(
    qtbot,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    options_model = AppOptions(ktool={"show_angle_scale_controls": False})
    data = xr.DataArray(
        np.arange(25.0).reshape((5, 5)),
        dims=("alpha", "eV"),
        coords={
            "alpha": np.linspace(-2.0, 2.0, 5),
            "eV": np.linspace(-1.0, -0.1, 5),
            "beta": 0.0,
            "xi": 0.0,
            "hv": 21.2,
        },
        attrs={"configuration": int(erlab.constants.AxesConfiguration.Type1)},
    )
    data.kspace.work_function = 4.5

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(
            root,
            show=False,
            provenance_spec=full_data().to_replay_spec(),
        )

        kspace_tool = KspaceTool(data, data_name="scan", options_model=options_model)
        kspace_tool._angle_scale_spins["alpha"].setValue(1.25)
        kspace_tool._angle_scale_spins["beta"].setValue(0.75)
        expected = kspace_tool._converted_output()
        tool_uid = add_source_childtool(manager, kspace_tool, 0, show=False)

        workspace_path = tmp_path / "hidden-ktool-scales.itws"
        manager._workspace_controller.saving._save_workspace_document(workspace_path)
        assert manager._workspace_controller.loading._load_workspace_file(
            workspace_path,
            replace=True,
            associate=True,
            mark_dirty=False,
            select=False,
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        restored = manager.get_childtool(tool_uid)
        assert isinstance(restored, KspaceTool)
        layout = restored.offsets_group.layout()
        assert all(
            not layout.isRowVisible(spin)
            for spin in restored._angle_scale_spins.values()
        )
        assert layout.isRowVisible(restored._angle_scale_summary)
        assert restored.data.kspace.alpha_scale == pytest.approx(1.25)
        assert restored.data.kspace.beta_scale == pytest.approx(0.75)
        xr.testing.assert_allclose(restored._converted_output(), expected)
        code = restored.copy_code()
        assert "alpha_scale=1.25" in code
        assert "beta_scale=0.75" in code

        manager._workspace_controller.saving._save_workspace_document(workspace_path)
        assert manager._workspace_controller.loading._load_workspace_file(
            workspace_path,
            replace=True,
            associate=True,
            mark_dirty=False,
            select=False,
        )
        restored_again = manager.get_childtool(tool_uid)
        assert isinstance(restored_again, KspaceTool)
        assert restored_again.data.kspace.alpha_scale == pytest.approx(1.25)
        assert restored_again.data.kspace.beta_scale == pytest.approx(0.75)


def test_manager_workspace_layout_only_save_updates_root_manifest_only(
    qtbot,
    monkeypatch,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()
        root = erlab.interactive.imagetool.ImageTool(test_data, _in_manager=True)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "layout-only.itws"
        manager._workspace_controller.saving._save_workspace_document(fname)
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()

        manager.resize(manager.width() + 40, manager.height() + 30)
        qtbot.wait_until(lambda: manager.is_workspace_modified, timeout=5000)
        assert manager._workspace_state.layout_modified
        assert manager._workspace_controller._dirty_details_text()
        manager._mark_workspace_layout_dirty()

        def _forbid_node_serialization(*_args, **_kwargs):
            raise AssertionError("layout-only save serialized a node")

        monkeypatch.setattr(
            manager._workspace_controller.saving,
            "_serialize_workspace_node",
            _forbid_node_serialization,
        )

        assert _request_workspace_save_and_wait(qtbot, manager)
        assert not manager.is_workspace_modified

        manifest = workspace_format._workspace_manifest_from_attrs(
            workspace_arrays._read_workspace_root_attrs_h5py(fname)
        )
        assert "delta_save_count" not in manifest
        assert (
            manifest["manager_layout"]
            == manager._workspace_controller.saving._workspace_layout_snapshot()
        )


def test_manager_workspace_standalone_app_only_save_updates_root_manifest_only(
    qtbot,
    monkeypatch,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()
        root = erlab.interactive.imagetool.ImageTool(test_data, _in_manager=True)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "standalone-only.itws"
        manager._workspace_controller.saving._save_workspace_document(fname)
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()

        manager.show_ptable()
        ptable = manager.ptable_window
        ptable.hv_edit.setText("150")
        qtbot.wait_until(lambda: manager.is_workspace_modified, timeout=5000)
        assert manager._workspace_state.layout_modified

        def _forbid_node_serialization(*_args, **_kwargs):
            raise AssertionError("standalone-only save serialized a node")

        monkeypatch.setattr(
            manager._workspace_controller.saving,
            "_serialize_workspace_node",
            _forbid_node_serialization,
        )

        assert _request_workspace_save_and_wait(qtbot, manager)
        assert not manager.is_workspace_modified

        manifest = workspace_format._workspace_manifest_from_attrs(
            workspace_arrays._read_workspace_root_attrs_h5py(fname)
        )
        assert "delta_save_count" not in manifest
        assert manifest["standalone_apps"]["apps"]["ptable"]["photon_energy"] == "150"


def test_manager_workspace_unlink_removes_saved_link_group(
    qtbot,
    tmp_path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool([test_data, test_data], link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        fname = tmp_path / "unlinked.itws"
        manager.link_imagetools(0, 1)
        manager._workspace_controller.saving._save_workspace_document(fname)

        select_tools(manager, [0, 1])
        manager.unlink_selected()
        assert manager.is_workspace_modified
        manager._workspace_controller.saving._save_workspace_document(fname)

        manifest = workspace_format._workspace_manifest_from_attrs(
            workspace_arrays._read_workspace_root_attrs_h5py(fname)
        )
        assert all("link_group" not in entry for entry in manifest["nodes"])

        assert manager._workspace_controller.loading._load_workspace_file(
            fname,
            replace=True,
            associate=True,
            mark_dirty=False,
            select=False,
        )
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        assert not manager.get_imagetool(0).slicer_area.is_linked
        assert not manager.get_imagetool(1).slicer_area.is_linked
        assert not manager.is_workspace_modified


def test_manager_load_workspace_dataset_ignores_invalid_saved_metadata(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": np.arange(2.0), "y": np.arange(2.0)},
    )
    saved = itool(data, manager=False, execute=False)
    qtbot.addWidget(saved)
    assert isinstance(saved, erlab.interactive.imagetool.ImageTool)
    ds = saved.to_dataset()
    ds.attrs["manager_node_uid"] = "loaded"
    ds.attrs["manager_node_provenance_spec"] = "{not-json"
    ds.attrs["manager_node_live_source_spec"] = "{not-json"
    ds.attrs["manager_node_live_source_binding"] = "{not-json"

    with manager_context() as manager:
        target = (
            manager._workspace_controller.loading._load_workspace_imagetool_dataset(
                ds, parent_target=None, node_path="-1"
            )
        )

        assert target in manager._tool_graph.root_wrappers
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        binding = ImageToolSelectionSourceBinding(
            selection_mode="isel",
            selection_indexers={"x": 0},
        )
        bound_ds = saved.to_dataset()
        bound_ds.attrs["manager_node_uid"] = "bound"
        bound_ds.attrs.pop("manager_node_live_source_spec", None)
        bound_ds.attrs["manager_node_live_source_binding"] = json.dumps(
            binding.model_dump(mode="json")
        )
        bound_ds.attrs.pop("itool_name", None)

        bound_target = (
            manager._workspace_controller.loading._load_workspace_imagetool_dataset(
                bound_ds, parent_target=None, node_path="-2"
            )
        )

        assert manager._node_for_target(bound_target).source_binding == binding
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)


@pytest.mark.parametrize(
    ("mark_method", "event_field"),
    [
        ("mark_layout_dirty", "layout"),
        ("mark_options_dirty", "options"),
        ("mark_context_dirty", "context"),
    ],
)
def test_workspace_state_repeated_non_node_dirty_during_save(
    mark_method: str,
    event_field: str,
) -> None:
    state = workspace_state._ManagerWorkspaceState()
    mark_dirty = getattr(state, mark_method)

    assert mark_dirty()
    assert state.dirty_generation == 1
    assert len(state.dirty_events) == 1
    assert getattr(state.dirty_events[0], event_field)
    assert not mark_dirty()
    assert state.dirty_generation == 1
    assert len(state.dirty_events) == 1

    state.save_in_progress = True

    assert mark_dirty()
    assert state.dirty_generation == 2
    assert len(state.dirty_events) == 2
    assert state.dirty_events[-1].generation == 2
    assert getattr(state.dirty_events[-1], event_field)


def test_workspace_script_state_owns_verified_and_explicit_sources() -> None:
    source = b"from erlab.extensions import routine\n"
    source_hash = hashlib.sha256(source).hexdigest()
    requirement = _WorkspaceScriptRequirement(
        script_name="Gaussian_Tools.py",
        capability_id="normalize",
        capability_name="Normalize",
        capability_kind="routine",
        source_hash=source_hash,
        extension_api_version=1,
        referencing_nodes=("loaded", "failed"),
    )
    scripts = workspace_state._WorkspaceScriptState((requirement,))
    scripts.remember_verified_source(
        requirement.script_name,
        source_hash,
        source,
        explicit=True,
    )

    snapshot = scripts.copy()
    assert snapshot == scripts
    assert scripts != object()
    scripts.rebase_nodes({"loaded": "loaded-import", "failed": "failed-import"})
    scripts.remove_node_references(("loaded-import",))
    scripts.remap_script("gaussian_tools.py", source_hash, "filters.py")

    assert scripts.requirements[0].script_name == "filters.py"
    assert scripts.requirements[0].referencing_nodes == ("failed-import",)
    assert ("filters.py", source_hash) in scripts.verified_sources
    assert scripts.explicit_sources == {("filters.py", source_hash)}
    assert scripts.source_manifest_value(frozenset()) == [
        {
            "script_name": "filters.py",
            "source_hash": source_hash,
            "object_id": f"extension-source-{source_hash}",
        }
    ]
    assert snapshot.requirements == (requirement,)
    assert snapshot.explicit_sources == {("Gaussian_Tools.py", source_hash)}
    assert snapshot != scripts

    with pytest.raises(ValueError, match="does not match its hash"):
        scripts.remember_verified_source("bad.py", "0" * 64, source)


@pytest.mark.parametrize(
    ("first_name", "second_name"),
    [
        ("Gaussian.py", "gaussian.py"),
        (
            "caf\N{LATIN SMALL LETTER E WITH ACUTE}.py",
            "cafe\N{COMBINING ACUTE ACCENT}.py",
        ),
    ],
)
def test_workspace_script_state_rejects_ambiguous_filenames_atomically(
    first_name: str,
    second_name: str,
) -> None:
    first = _WorkspaceScriptRequirement(
        script_name=first_name,
        capability_id="first",
        capability_name="First",
        capability_kind="routine",
        source_hash="a" * 64,
        extension_api_version=1,
    )
    second = first.model_copy(
        update={
            "script_name": second_name,
            "capability_id": "second",
            "capability_name": "Second",
        }
    )

    with pytest.raises(ValueError, match="ambiguous filenames"):
        workspace_state._WorkspaceScriptState((first, second))

    scripts = workspace_state._WorkspaceScriptState((first,))
    with pytest.raises(ValueError, match="ambiguous filenames"):
        scripts.merge(workspace_state._WorkspaceScriptState((second,)))
    assert scripts.requirements == (first,)


def test_workspace_script_state_rejects_invalid_verified_source_merges() -> None:
    source = b"source\n"
    source_hash = hashlib.sha256(source).hexdigest()
    entry = workspace_format._WorkspaceEmbeddedScriptEntry(
        script_name="script.py",
        source_hash=source_hash,
        object_id=f"extension-source-{source_hash}",
    )

    with pytest.raises(ValueError, match="key does not match"):
        workspace_state._WorkspaceScriptState(
            verified_sources={("other.py", source_hash): (entry, source)}
        )
    with pytest.raises(ValueError, match="must have verified bytes"):
        workspace_state._WorkspaceScriptState(
            explicit_sources={("script.py", source_hash)}
        )


def test_workspace_script_state_rejects_merge_and_rename_collisions() -> None:
    source = b"source\n"
    source_hash = hashlib.sha256(source).hexdigest()
    scripts = workspace_state._WorkspaceScriptState()
    scripts.remember_verified_source("first.py", source_hash, source)
    scripts.remember_verified_source("second.py", source_hash, source)
    with pytest.raises(ValueError, match="already uses"):
        scripts.remap_script("first.py", source_hash, "second.py")


def test_workspace_script_state_manifest_composition_and_name_conflicts() -> None:
    source = b"source\n"
    source_hash = hashlib.sha256(source).hexdigest()
    requirement = _WorkspaceScriptRequirement(
        script_name="Script.py",
        capability_id="routine",
        capability_name="Routine",
        capability_kind="routine",
        source_hash=source_hash,
        extension_api_version=1,
    )
    scripts = workspace_state._WorkspaceScriptState((requirement,))

    assert scripts.remember_verified_source("script.py", source_hash, source) is None
    entry = scripts.remember_verified_source("Script.py", source_hash, source)
    assert entry is not None
    assert scripts.requirement_manifest_value((requirement,)) == [
        requirement.model_dump(mode="json"),
    ]
    assert scripts.source_manifest_value({("Script.py", source_hash)}) == [
        entry.model_dump(mode="json"),
    ]
    assert scripts.source_manifest_value(frozenset()) == []


def test_workspace_script_state_validation_and_merge_boundaries() -> None:
    source = b"source\n"
    source_hash = hashlib.sha256(source).hexdigest()
    entry = workspace_format._WorkspaceEmbeddedScriptEntry(
        script_name="script.py",
        source_hash=source_hash,
        object_id=f"extension-source-{source_hash}",
    )

    with pytest.raises(ValueError, match="does not match its hash"):
        workspace_state._WorkspaceScriptState(
            verified_sources={("script.py", source_hash): (entry, b"different")}
        )
    scripts = workspace_state._WorkspaceScriptState()
    scripts.remember_verified_source("script.py", source_hash, source)
    other_source = b"other\n"
    other_hash = hashlib.sha256(other_source).hexdigest()
    scripts.remember_verified_source("other.py", other_hash, other_source)
    scripts.remap_script("script.py", source_hash, "renamed.py")
    assert set(scripts.verified_sources) == {
        ("renamed.py", source_hash),
        ("other.py", other_hash),
    }


def test_workspace_save_snapshot_selects_and_preserves_extension_objects(
    tmp_path: pathlib.Path,
) -> None:
    source_path = tmp_path / "source.itws"
    target_path = tmp_path / "target.itws"
    included_source = b"included\n"
    included_hash = hashlib.sha256(included_source).hexdigest()
    excluded_source = b"excluded\n"
    excluded_hash = hashlib.sha256(excluded_source).hexdigest()
    carried_source = b"carried\n"
    carried_hash = hashlib.sha256(carried_source).hexdigest()
    included_key = ("included.py", included_hash)
    excluded_key = ("excluded.py", excluded_hash)
    scripts = workspace_state._WorkspaceScriptState()
    included_entry = scripts.remember_verified_source(*included_key, included_source)
    excluded_entry = scripts.remember_verified_source(*excluded_key, excluded_source)
    if included_entry is None or excluded_entry is None:
        raise RuntimeError("Expected verified source entries")
    carried_entry = workspace_format._WorkspaceEmbeddedScriptEntry(
        script_name="carried.py",
        source_hash=carried_hash,
        object_id=f"extension-source-{carried_hash}",
    )

    current_manifest = {
        "schema_version": 6,
        "nodes": [],
        "embedded_extension_sources": [carried_entry.model_dump(mode="json")],
    }
    manifest = {
        "schema_version": 6,
        "nodes": [],
        "embedded_extension_sources": [
            included_entry.model_dump(mode="json"),
            excluded_entry.model_dump(mode="json"),
            carried_entry.model_dump(mode="json"),
        ],
    }
    with workspace_store.WorkspaceStore(source_path, create=True) as source_store:
        with source_store.write_session() as h5_file:
            group = h5_file.require_group(
                source_store.object_path(carried_entry.object_id)
            )
            group.attrs["erlab_object_kind"] = "extension-python-source-v1"
            group.create_dataset(
                "source",
                data=np.frombuffer(carried_source, dtype=np.uint8),
            )
        source_store.publish(current_manifest)

        manager = types.SimpleNamespace(
            _workspace_state=types.SimpleNamespace(
                dirty_data=set(),
                dirty_added=set(),
                dirty_state=set(),
                path=source_path.resolve(),
                code_trust=new_document_trust(),
            ),
            _tool_graph=types.SimpleNamespace(nodes={}),
        )
        saver = workspace_saving._WorkspaceSaver.__new__(
            workspace_saving._WorkspaceSaver
        )
        saver._manager = manager
        saver._controller = types.SimpleNamespace(_workspace_store=source_store)
        saver._workspace_script_snapshot = lambda: (
            (),
            scripts,
            frozenset({included_key}),
        )
        saver._workspace_manifest = lambda **_kwargs: json.loads(json.dumps(manifest))
        saver._workspace_stale_reference_rewrite_uids = lambda _uids: frozenset()
        saver._workspace_compression_mode = lambda: "none"
        saver._serialized_tool_data_references = lambda _datasets: ()

        snapshot = saver._workspace_generation_save_snapshot(3, fname=target_path)

    writes = {item.object_id: item for item in snapshot.generation_plan.objects}
    assert set(writes) == {included_entry.object_id, carried_entry.object_id}
    assert writes[included_entry.object_id].blob == included_source
    assert writes[carried_entry.object_id].source_file == str(source_path)
    assert writes[carried_entry.object_id].source_path == source_store.object_path(
        carried_entry.object_id
    )
    assert snapshot.embedded_script_sources == (
        (included_entry.script_name, included_hash, included_source),
    )
    snapshot.close()


def test_committed_generation_merges_embedded_source_into_live_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedded_source = b"EMBEDDED = 1\n"
    embedded_hash = hashlib.sha256(embedded_source).hexdigest()
    later_source = b"LATER = 2\n"
    later_hash = hashlib.sha256(later_source).hexdigest()
    scripts = workspace_state._WorkspaceScriptState()
    scripts.remember_verified_source("later.py", later_hash, later_source)
    manager = types.SimpleNamespace(
        _workspace_state=types.SimpleNamespace(
            extension_scripts=scripts,
            schema_version=0,
        )
    )
    controller = workspace_controller._WorkspaceController.__new__(
        workspace_controller._WorkspaceController
    )
    controller._manager = manager
    rebound: list[dict[str, str]] = []
    controller._workspace_store = types.SimpleNamespace(
        closed=False,
        rebind_legacy_readers=lambda mappings: rebound.append(mappings),
    )
    monkeypatch.setattr(
        controller,
        "_commit_saved_tool_data_references",
        lambda _snapshot: None,
    )
    monkeypatch.setattr(
        controller,
        "_repoint_saved_pending_workspace_payloads",
        lambda *_args, **_kwargs: None,
    )
    snapshot = workspace_saving._WorkspaceSaveSnapshot(
        generation=0,
        generation_plan=workspace_storage._WorkspaceGenerationPlan(
            manifest={"schema_version": 6, "nodes": []},
            objects=(),
            legacy_reader_rebindings=(("/legacy/imagetool", "payload"),),
        ),
        compression_mode="none",
        trusted_lineage=True,
        embedded_script_sources=(("embedded.py", embedded_hash, embedded_source),),
    )

    controller._adopt_committed_workspace_generation(
        pathlib.Path("workspace.itws"),
        snapshot,
        manifest=snapshot.generation_plan.manifest,
    )

    assert set(scripts.verified_sources) == {
        ("embedded.py", embedded_hash),
        ("later.py", later_hash),
    }
    assert rebound == [{"/legacy/imagetool": "payload"}]
    assert manager._workspace_state.schema_version == 6


def test_workspace_save_and_compaction_route_save_as_only_documents() -> None:
    callbacks: list[typing.Callable[[bool], None] | None] = []
    manager = types.SimpleNamespace(
        _workspace_state=types.SimpleNamespace(
            save_as_only=True,
            save_in_progress=False,
        )
    )
    controller = workspace_controller._WorkspaceController.__new__(
        workspace_controller._WorkspaceController
    )
    controller._manager = manager
    controller._current_workspace_document_path = lambda: pathlib.Path("workspace.itws")
    controller.save_as = lambda **kwargs: (
        callbacks.append(kwargs.get("on_finished")) or True
    )

    assert controller.save(native=False)
    assert callbacks == [None]
    assert controller.compact_workspace()
    compact_callback = callbacks[-1]
    if compact_callback is None:
        raise TypeError("Expected a compaction callback")

    compacted: list[None] = []
    controller.compact_workspace = lambda: compacted.append(None) or True
    compact_callback(False)
    assert compacted == []
    manager._workspace_state.save_as_only = False
    compact_callback(True)
    assert compacted == [None]


def test_committed_save_retains_always_embedded_source_for_later_save(
    qtbot,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    script_path = tmp_path / "always.py"
    script_source = b"VALUE = 1\n"
    script_path.write_bytes(script_source)
    source_hash = hashlib.sha256(script_source).hexdigest()
    workspace_path = tmp_path / "always.itws"

    with manager_context() as manager:
        catalog, _registered_hash = manager._extensions.catalog.store.register_script(
            script_path
        )
        record = catalog.extensions[script_path.name.casefold()]
        manager._extensions.catalog.store.update_script(
            record.script_name,
            expected_record_generation=record.record_generation,
            embed_policy="always",
        )
        manager._extensions.catalog.refresh()
        tool = itool(xr.DataArray([1.0]), manager=False, execute=False)
        if not isinstance(tool, erlab.interactive.imagetool.ImageTool):
            raise TypeError("Expected an ImageTool")
        manager.add_imagetool(tool, show=False)
        monkeypatch.setattr(
            manager._workspace_controller,
            "_workspace_save_dialog",
            lambda **_kwargs: workspace_path,
        )

        assert _request_workspace_save_as_and_wait(qtbot, manager, native=False)
        assert (script_path.name, source_hash) in (
            manager._workspace_state.extension_scripts.verified_sources
        )
        script_path.unlink()
        manager._workspace_state.mark_layout_dirty()
        assert _request_workspace_save_and_wait(qtbot, manager, native=False)

    source_entries = _current_workspace_manifest(workspace_path)[
        "embedded_extension_sources"
    ]
    assert source_entries == [
        {
            "script_name": script_path.name,
            "source_hash": source_hash,
            "object_id": f"extension-source-{source_hash}",
        }
    ]
    recovered_source, kind = workspace_storage._read_workspace_blob(
        workspace_path,
        source_entries[0]["object_id"],
    )
    assert recovered_source == script_source
    assert kind == "extension-python-source-v1"


def test_manager_close_save_path_updates_file_path(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager._workspace_state.path = tmp_path / "close-save.itws"
        manager._workspace_state.structure_modified = True
        save_closing_states: list[bool] = []
        save_callbacks: list[Callable[[bool], None]] = []
        file_path_calls: list[str] = []
        close_calls: list[str] = []

        def _save(
            *, native: bool = True, on_finished: Callable[[bool], None] | None = None
        ) -> bool:
            save_closing_states.append(manager._workspace_state.closing_document)
            if on_finished is not None:
                save_callbacks.append(on_finished)
            return True

        with monkeypatch.context() as patch:
            patch.setattr(
                ImageToolManager,
                "setWindowFilePath",
                lambda _manager, path: file_path_calls.append(path),
            )
            patch.setattr(
                QtWidgets.QMessageBox,
                "exec",
                lambda _msg_box: QtWidgets.QMessageBox.StandardButton.Save,
            )
            patch.setattr(manager._workspace_controller, "save", _save)
            patch.setattr(manager, "close", lambda: close_calls.append("close") or True)
            event = QtGui.QCloseEvent()
            manager.closeEvent(event)
            assert not event.isAccepted()
            manager._workspace_controller._mark_workspace_clean()
            save_callbacks[0](True)

        assert save_closing_states == [True]
        assert file_path_calls == [str(manager._workspace_state.path)]
        assert close_calls == ["close"]
        assert not manager._workspace_state.closing_document


def test_manager_workspace_save_as_locked_target_does_not_write(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    fname = tmp_path / "locked-save-as.itws"
    _write_transaction_test_workspace(fname)
    lock = workspace_storage._acquire_workspace_document_lock(fname)
    operation_errors: list[tuple[typing.Any, ...]] = []
    try:
        with manager_context() as manager:
            monkeypatch.setattr(
                manager._workspace_controller,
                "_workspace_save_dialog",
                lambda *args, **kwargs: str(fname),
            )
            monkeypatch.setattr(
                manager._workspace_controller.saving,
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

            assert not manager._workspace_controller.save_as(native=False)
    finally:
        lock.unlock()

    assert operation_errors


def test_manager_workspace_save_as_reports_snapshot_error(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    fname = tmp_path / "write-error.itws"
    operation_errors: list[tuple[typing.Any, ...]] = []

    with manager_context() as manager:
        monkeypatch.setattr(
            manager._workspace_controller,
            "_workspace_save_dialog",
            lambda *args, **kwargs: str(fname),
        )
        monkeypatch.setattr(
            manager._workspace_controller.saving,
            "_workspace_generation_save_snapshot",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        monkeypatch.setattr(
            manager,
            "_show_operation_error",
            lambda *args, **kwargs: operation_errors.append(args),
        )

        assert not manager._workspace_controller.save_as(native=False)

    assert operation_errors == [
        (
            "Error while saving workspace",
            "An error occurred while saving the workspace file.",
        )
    ]


def test_manager_workspace_save_as_rejects_h5_target(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    fname = tmp_path / "workspace.h5"
    warnings: list[tuple[typing.Any, ...]] = []

    with manager_context() as manager:
        monkeypatch.setattr(
            manager._workspace_controller,
            "_workspace_save_dialog",
            lambda *args, **kwargs: str(fname),
        )
        monkeypatch.setattr(
            QtWidgets.QMessageBox, "warning", lambda *args: warnings.append(args)
        )

        assert not manager._workspace_controller.save_as(native=False)

    assert len(warnings) == 1
    assert warnings[0][1:] == (
        "Workspace Not Saved",
        "ImageTool Manager saves workspaces as .itws files.",
    )
    assert not fname.exists()


def test_manager_workspace_load_locks_before_recovery(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    fname = tmp_path / "locked-load.itws"
    _write_transaction_test_workspace(fname)
    lock = workspace_storage._acquire_workspace_document_lock(fname)
    recovery_calls: list[pathlib.Path] = []
    try:
        monkeypatch.setattr(
            workspace_storage,
            "_recover_workspace_transactions",
            lambda path: (
                recovery_calls.append(pathlib.Path(path))
                or pytest.fail("Load should lock the workspace before recovery")
            ),
        )
        with manager_context() as manager, pytest.raises(BlockingIOError):
            manager._workspace_controller.loading._load_workspace_file(
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
        manager._workspace_state.path = current
        lock = _FakeLock()

        manager._workspace_controller._set_workspace_path(current, workspace_lock=lock)

        assert lock.unlock_count == 1
        with pytest.raises(RuntimeError, match="pre-acquired document lock"):
            manager._workspace_controller._set_workspace_path(tmp_path / "other.itws")


def test_manager_open_recent_menu_stays_disabled_while_save_in_progress(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    workspace = tmp_path / "recent.itws"
    workspace.touch()

    with manager_context() as manager:
        manager._workspace_controller._record_recent_workspace(workspace)
        assert manager.open_recent_menu.isEnabled()

        manager._workspace_state.save_in_progress = True
        manager._workspace_controller._refresh_open_recent_menu_action()
        assert not manager.open_recent_menu.isEnabled()
        manager._workspace_controller._populate_open_recent_menu()
        assert not manager.open_recent_menu.isEnabled()
        assert "manager_recent_workspace_action_0" in action_map_by_object_name(
            manager.open_recent_menu
        )

        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "warning",
            lambda *_args, **_kwargs: pytest.fail(
                "Open Recent should not show dialogs during workspace save"
            ),
        )
        assert not manager.open_recent_workspace(tmp_path / "missing.itws")

        manager._workspace_state.save_in_progress = False
        manager._workspace_controller._refresh_open_recent_menu_action()
        assert manager.open_recent_menu.isEnabled()


def test_manager_compact_workspace_edge_paths(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        monkeypatch.setattr(manager._workspace_controller, "save_as", lambda: True)
        assert manager.compact_workspace()

        manager._workspace_state.path = tmp_path / "workspace.itws"
        manager._workspace_state.save_in_progress = True
        assert not manager.compact_workspace()
        manager._workspace_state.save_in_progress = False

        operation_errors: list[tuple[typing.Any, ...]] = []
        monkeypatch.setattr(
            manager,
            "_show_operation_error",
            lambda *args: operation_errors.append(args),
        )

        assert not manager.compact_workspace()
        assert operation_errors == [
            (
                "Error while compacting workspace",
                "The workspace file is not open. Reopen it and try again.",
            )
        ]


@pytest.mark.parametrize(
    ("button_role", "expected"),
    [
        (QtWidgets.QMessageBox.ButtonRole.DestructiveRole, True),
        (QtWidgets.QMessageBox.ButtonRole.RejectRole, False),
    ],
)
def test_workspace_compaction_dask_warning_choices(
    qtbot,
    accept_dialog,
    button_role: QtWidgets.QMessageBox.ButtonRole,
    expected: bool,
) -> None:
    controller = workspace_controller._WorkspaceController.__new__(
        workspace_controller._WorkspaceController
    )
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    results: list[bool] = []

    def _choose_button(dialog: QtWidgets.QMessageBox) -> None:
        button = next(
            button
            for button in dialog.buttons()
            if dialog.buttonRole(button) == button_role
        )
        button.click()

    accept_dialog(
        lambda: results.append(
            controller._confirm_compaction_with_exported_readers(parent)
        ),
        accept_call=_choose_button,
    )

    assert results == [expected]


def test_manager_compaction_warns_only_for_exported_readers(
    monkeypatch,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    path = tmp_path / "workspace.itws"
    compacted: list[workspace_store.WorkspaceStore] = []
    with (
        manager_context() as manager,
        workspace_store.WorkspaceStore(path, create=True) as store,
    ):
        controller = manager._workspace_controller
        manager._workspace_state.path = path.resolve()
        manager._workspace_state.schema_version = (
            workspace_format._current_workspace_schema_version()
        )
        controller._workspace_store = store
        monkeypatch.setattr(
            erlab.interactive.utils,
            "wait_dialog",
            lambda *args, **kwargs: contextlib.nullcontext(),
        )
        discarded: list[tuple[frozenset[str], frozenset[str]]] = []

        def _compact(
            current_store: workspace_store.WorkspaceStore,
            *,
            discard_serialized_reader_pins=None,
        ) -> None:
            compacted.append(current_store)
            snapshot = (
                discard_serialized_reader_pins
                or workspace_store._SerializedReaderPinSnapshot({}, {})
            )
            discarded.append(
                (
                    snapshot.object_ids,
                    snapshot.legacy_group_paths,
                )
            )
            if snapshot.object_ids:
                workspace_store.WorkspaceStore.pin_serialized_reader(
                    workspace_id=store.workspace_id,
                    path=path,
                    object_id="new-export",
                    legacy_group_path=None,
                )
                workspace_store.WorkspaceStore.pin_serialized_reader(
                    workspace_id=store.workspace_id,
                    path=path,
                    object_id="exported",
                    legacy_group_path=None,
                )

        monkeypatch.setattr(workspace_storage, "_compact_workspace_store", _compact)
        confirmations: list[QtWidgets.QWidget] = []
        monkeypatch.setattr(
            controller,
            "_confirm_compaction_with_exported_readers",
            lambda parent: confirmations.append(parent) or False,
        )

        assert manager.compact_workspace()
        assert compacted == [store]
        assert discarded == [(frozenset(), frozenset())]
        assert confirmations == []

        workspace_store.WorkspaceStore.pin_serialized_reader(
            workspace_id=store.workspace_id,
            path=path,
            object_id="exported",
            legacy_group_path=None,
        )
        assert not manager.compact_workspace()
        assert compacted == [store]
        assert len(confirmations) == 1
        assert store.has_serialized_readers

        monkeypatch.setattr(
            controller,
            "_confirm_compaction_with_exported_readers",
            lambda _parent: True,
        )
        assert manager.compact_workspace()
        assert compacted == [store, store]
        assert discarded[-1] == (frozenset({"exported"}), frozenset())
        assert store.serialized_object_ids == {"exported", "new-export"}
        store.clear_serialized_reader_pins()

        controller._workspace_store = None
        manager._workspace_state.path = None


def test_manager_compact_workspace_detaches_store_that_cannot_reopen(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    path = tmp_path / "workspace.itws"
    with (
        manager_context() as manager,
        workspace_store.WorkspaceStore(path, create=True) as store,
    ):
        manager._workspace_state.path = path.resolve()
        manager._workspace_state.schema_version = (
            workspace_format._current_workspace_schema_version()
        )
        manager._workspace_controller._workspace_store = store
        operation_errors: list[tuple[str, str]] = []
        monkeypatch.setattr(
            erlab.interactive.utils,
            "wait_dialog",
            lambda *args, **kwargs: contextlib.nullcontext(),
        )
        monkeypatch.setattr(
            manager,
            "_show_operation_error",
            lambda title, text: operation_errors.append((title, text)),
        )

        def _fail_after_replacement(
            current_store: workspace_store.WorkspaceStore,
            **_kwargs,
        ) -> None:
            current_store._close_handle()
            raise workspace_store.WorkspaceStoreReopenError(path)

        monkeypatch.setattr(
            workspace_storage, "_compact_workspace_store", _fail_after_replacement
        )
        workspace_store.WorkspaceStore.pin_serialized_reader(
            workspace_id=store.workspace_id,
            path=path,
            object_id="exported",
            legacy_group_path=None,
        )
        monkeypatch.setattr(
            manager._workspace_controller,
            "_confirm_compaction_with_exported_readers",
            lambda _parent: True,
        )

        assert not manager.compact_workspace()
        assert len(operation_errors) == 1
        assert store.serialized_object_ids == {"exported"}
        assert manager._workspace_controller._workspace_store is None
        assert workspace_store.WorkspaceStore.active(path) is None
        store.clear_serialized_reader_pins()
        manager._workspace_state.path = None


def test_manager_compact_workspace_reduces_internal_holes(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    old_compression = erlab.interactive.options["io/workspace/compression"]
    erlab.interactive.options["io/workspace/compression"] = "none"
    try:
        with manager_context() as manager:
            qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
            rng = np.random.default_rng(1234)
            data = xr.DataArray(
                rng.integers(0, 256, size=(2048, 2048), dtype=np.uint8),
                dims=("x", "y"),
            )
            updated = xr.DataArray(
                rng.integers(0, 256, size=(2048, 2048), dtype=np.uint8),
                dims=("x", "y"),
            )

            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)
            uid = manager._tool_graph.root_wrappers[0].uid

            fname = tmp_path / "hole-repack.itws"
            manager._workspace_controller.saving._save_workspace_document(fname)
            adopt_workspace_path(manager, fname)
            manager._workspace_controller._mark_workspace_clean()
            size_full = fname.stat().st_size

            manager.get_imagetool(0).slicer_area.replace_source_data(
                updated,
                auto_compute=False,
            )
            manager._workspace_controller._mark_node_data_dirty(uid)
            assert _request_workspace_save_and_wait(qtbot, manager)
            size_incremental = fname.stat().st_size

            monkeypatch.setattr(
                erlab.interactive.utils,
                "wait_dialog",
                lambda *args, **kwargs: contextlib.nullcontext(),
            )
            assert manager.compact_workspace()
            size_compact = fname.stat().st_size

            assert size_incremental > size_full + data.nbytes // 2
            assert size_compact < size_incremental - data.nbytes // 2
            _assert_no_workspace_internal_groups(fname)
    finally:
        erlab.interactive.options["io/workspace/compression"] = old_compression


def test_manager_compact_workspace_upgrades_previous_schema(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    path = tmp_path / "previous-schema.itws"
    calls: list[str] = []
    with (
        manager_context() as manager,
        workspace_store.WorkspaceStore(path, create=True) as store,
    ):
        manager._workspace_state.path = path.resolve()
        manager._workspace_state.schema_version = (
            workspace_format._current_workspace_schema_version() - 1
        )
        manager._workspace_controller._workspace_store = store
        monkeypatch.setattr(
            erlab.interactive.utils,
            "wait_dialog",
            lambda *args, **kwargs: contextlib.nullcontext(),
        )
        monkeypatch.setattr(
            manager._workspace_controller.saving,
            "_save_workspace_document",
            lambda *_args, **_kwargs: calls.append("save"),
        )
        monkeypatch.setattr(
            workspace_storage,
            "_compact_workspace_store",
            lambda current_store, **_kwargs: calls.append(
                "compact" if current_store is store else "wrong-store"
            ),
        )

        assert manager.compact_workspace()
        assert calls == ["save", "compact"]

        manager._workspace_controller._workspace_store = None
        manager._workspace_state.path = None


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
            manager._workspace_controller._workspace_save_dialog(
                native=False, selected_file=tmp_path / "explicit.itws"
            )
            is None
        )
        assert ("select", str(tmp_path / "explicit.itws")) in calls

        _FakeFileDialog.exec_result = 1
        manager._workspace_state.path = tmp_path / "bound.itws"
        assert manager._workspace_controller._workspace_save_dialog(native=True) == str(
            tmp_path / "selected.itws"
        )
        assert ("select", str(tmp_path / "bound.itws")) in calls

        manager._workspace_state.path = None
        manager._recent_directory = None
        default_dir = tmp_path / "default"
        default_dir.mkdir()
        erlab.interactive.options.model = AppOptions().model_copy(
            update={
                "io": AppOptions().io.model_copy(
                    update={"default_directory": str(default_dir)}
                )
            }
        )
        assert manager._workspace_controller._workspace_save_dialog(native=True) == str(
            tmp_path / "selected.itws"
        )
        assert ("directory", str(default_dir)) in calls

        manager._recent_directory = str(tmp_path)
        assert manager._workspace_controller._workspace_save_dialog(native=True) == str(
            tmp_path / "selected.itws"
        )
        assert ("directory", str(tmp_path)) in calls


def test_manager_dirty_workspace_save_choice_save_branch(
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager._workspace_state.path = pathlib.Path("dirty.itws")
        manager._workspace_state.structure_modified = True
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "exec",
            lambda _msg_box: QtWidgets.QMessageBox.StandardButton.Save,
        )

        assert (
            manager._workspace_controller._dirty_workspace_save_choice(
                "Save before continuing."
            )
            == "save"
        )


def test_manager_legacy_itws_schema_save_helpers(
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
        manager._workspace_controller._show_legacy_workspace_upgrade_message(
            tmp_path / "legacy-schema.itws"
        )

        monkeypatch.setattr(
            manager._workspace_controller,
            "_workspace_save_dialog",
            lambda **_kwargs: None,
        )
        assert (
            manager._workspace_controller._save_legacy_workspace_as_current(
                tmp_path / "legacy-schema.itws"
            )
            is None
        )

        dirty_reasons: list[str] = []
        monkeypatch.setattr(
            manager._workspace_controller,
            "_save_legacy_workspace_as_current",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            manager._workspace_controller,
            "_mark_workspace_structure_dirty",
            dirty_reasons.append,
        )
        manager._workspace_controller._associate_loaded_workspace_file(
            tmp_path / "legacy-schema.itws",
            workspace_format._WORKSPACE_LEGACY_SCHEMA_VERSION - 1,
        )

        assert manager._workspace_state.path is None
        assert dirty_reasons == ["Legacy workspace needs conversion"]


class _DeferredWorkspaceSaveWorker:
    def __init__(
        self,
        _fname: str | os.PathLike[str],
        snapshot: workspace_saving._WorkspaceSaveSnapshot,
        *,
        store: workspace_store.WorkspaceStore | None = None,
        reader_closers: tuple[Callable[[], None], ...] = (),
    ) -> None:
        self.signals = workspace_saving._WorkspaceSaveWorkerSignals()
        self._fname = pathlib.Path(_fname)
        self._snapshot = snapshot
        self._store = store
        self._reader_closers = reader_closers

    def finish(
        self,
        *,
        elapsed: float = 0.0,
        error: workspace_saving._WorkspaceSaveError | None = None,
    ) -> None:
        self._snapshot.close()
        self.signals.finished.emit(elapsed, error)


class _DeferredWorkspaceSaveThreadPool:
    def __init__(self) -> None:
        self.workers: list[_DeferredWorkspaceSaveWorker] = []

    def start(self, worker: _DeferredWorkspaceSaveWorker) -> None:
        self.workers.append(worker)


def _install_deferred_workspace_save_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> _DeferredWorkspaceSaveThreadPool:
    pool = _DeferredWorkspaceSaveThreadPool()
    monkeypatch.setattr(
        QtCore.QThreadPool, "globalInstance", staticmethod(lambda: pool)
    )
    monkeypatch.setattr(
        workspace_saving, "_WorkspaceSaveWorker", _DeferredWorkspaceSaveWorker
    )
    return pool


def _bind_dirty_workspace_for_save_test(
    manager: erlab.interactive.imagetool.manager.ImageToolManager,
    tmp_path: pathlib.Path,
) -> pathlib.Path:
    fname = tmp_path / "background-save.itws"
    fname.touch()
    manager._workspace_state.path = fname.resolve()
    manager._workspace_controller._mark_workspace_clean()
    manager._workspace_state.mark_layout_dirty()
    return fname


def _workspace_save_test_snapshot(
    manager: erlab.interactive.imagetool.manager.ImageToolManager,
) -> workspace_saving._WorkspaceSaveSnapshot:
    return workspace_saving._WorkspaceSaveSnapshot(
        generation=manager._workspace_state.dirty_generation,
        generation_plan=workspace_storage._WorkspaceGenerationPlan(
            manifest={"schema_version": 5, "nodes": []},
            objects=(),
        ),
        compression_mode="none",
        trusted_lineage=True,
    )


def test_workspace_save_worker_reports_missing_backing_source(tmp_path) -> None:
    missing_source = tmp_path / "deleted-source.itws"
    target = tmp_path / "target.itws"
    object_id = "missing"
    snapshot = workspace_saving._WorkspaceSaveSnapshot(
        generation=0,
        generation_plan=workspace_storage._WorkspaceGenerationPlan(
            manifest={
                "schema_version": 5,
                "nodes": [
                    {
                        "uid": "node",
                        "kind": "imagetool",
                        "path": "0",
                        "payload_object_id": object_id,
                        "payload_path": workspace_store.WorkspaceStore.object_path(
                            object_id
                        ),
                    }
                ],
            },
            objects=(
                workspace_storage._WorkspaceObjectWrite(
                    object_id,
                    source_file=str(missing_source),
                    source_path="0/imagetool",
                ),
            ),
        ),
        compression_mode="none",
        trusted_lineage=True,
    )
    worker = workspace_saving._WorkspaceSaveWorker(target, snapshot)
    results: list[tuple[float, workspace_saving._WorkspaceSaveError | None]] = []
    receiver = workspace_saving._WorkspaceSaveResultReceiver(
        callback=lambda elapsed, error: results.append((elapsed, error)),
        parent=worker.signals,
    )
    worker.signals.finished.connect(receiver.finish)

    worker.run()

    assert len(results) == 1
    _elapsed, error = results[0]
    assert isinstance(error, workspace_saving._WorkspaceSaveError)
    assert error.missing_source_path == str(missing_source)
    assert error.traceback_text


@pytest.mark.parametrize(
    ("exception_factory", "error_field"),
    [
        (
            lambda path: workspace_storage._WorkspacePublicationConflictError(path),
            "publication_conflict_path",
        ),
        (
            lambda _path: workspace_store.WorkspaceStoreConflictError("changed"),
            "publication_conflict_path",
        ),
        (
            lambda path: PermissionError(errno.EACCES, "access denied", path),
            "access_denied_path",
        ),
    ],
)
def test_workspace_save_worker_classifies_publication_errors(
    monkeypatch, tmp_path, exception_factory, error_field
) -> None:
    target = tmp_path / "target.itws"
    snapshot = workspace_saving._WorkspaceSaveSnapshot(
        generation=0,
        generation_plan=workspace_storage._WorkspaceGenerationPlan(
            manifest={"schema_version": 5, "nodes": []},
            objects=(),
        ),
        compression_mode="none",
        trusted_lineage=True,
    )
    monkeypatch.setattr(
        workspace_storage,
        "_write_workspace_generation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(exception_factory(target)),
    )
    worker = workspace_saving._WorkspaceSaveWorker(target, snapshot)
    results: list[workspace_saving._WorkspaceSaveError | None] = []
    receiver = workspace_saving._WorkspaceSaveResultReceiver(
        callback=lambda _elapsed, error: results.append(error),
        parent=worker.signals,
    )
    worker.signals.finished.connect(receiver.finish)

    worker.run()

    assert len(results) == 1
    error = results[0]
    assert isinstance(error, workspace_saving._WorkspaceSaveError)
    assert getattr(error, error_field) == str(target)


@pytest.mark.parametrize("contended", [False, True])
def test_workspace_save_worker_closes_readers_only_after_contention(
    monkeypatch,
    tmp_path,
    contended: bool,
) -> None:
    target = tmp_path / "target.itws"
    snapshot = workspace_saving._WorkspaceSaveSnapshot(
        generation=0,
        generation_plan=workspace_storage._WorkspaceGenerationPlan(
            manifest={"schema_version": 5, "nodes": []},
            objects=(),
        ),
        compression_mode="none",
        trusted_lineage=True,
    )
    close_calls: list[None] = []

    def _write(*_args, on_contention=None, **_kwargs) -> None:
        assert close_calls == []
        if contended:
            assert on_contention is not None
            on_contention()

    monkeypatch.setattr(workspace_storage, "_write_workspace_generation", _write)
    worker = workspace_saving._WorkspaceSaveWorker(
        target,
        snapshot,
        reader_closers=(lambda: close_calls.append(None),),
    )
    waiting: list[None] = []
    worker.signals.waiting.connect(lambda: waiting.append(None))

    worker.run()

    assert close_calls == ([None] if contended else [])
    assert waiting == ([None] if contended else [])


def test_workspace_gc_worker_reports_contention_and_errors() -> None:
    closed: list[None] = []

    class _Store:
        def collect_garbage(
            self,
            *,
            max_objects,
            on_contention,
        ):
            assert max_objects == 1
            on_contention()
            raise RuntimeError("cleanup failed")

    worker = workspace_saving._WorkspaceGcWorker(
        typing.cast("workspace_store.WorkspaceStore", _Store()),
        reader_closers=(lambda: closed.append(None),),
    )
    results: list[tuple[bool, str | None]] = []
    worker.signals.finished.connect(
        lambda more, error: results.append((more, typing.cast("str | None", error)))
    )

    worker.run()

    assert closed == [None]
    assert len(results) == 1
    assert not results[0][0]
    assert results[0][1] is not None
    assert "cleanup failed" in results[0][1]


def test_pending_workspace_tool_attrs_update_script_inputs() -> None:
    saver = workspace_saving._WorkspaceSaver.__new__(workspace_saving._WorkspaceSaver)
    pending_base = {
        "tool_display_name": "old",
        "tool_title": "prefix old",
        erlab.interactive.utils._TOOL_SOURCE_BINDING_ATTR: "stale",
        erlab.interactive.utils._TOOL_INPUT_PROVENANCE_SPEC_ATTR: "legacy",
        erlab.interactive.utils._TOOL_PRIMARY_INPUT_ATTR: "data",
    }
    saver._pending_workspace_node_attrs = types.MethodType(
        lambda _self, _node, _attrs, *, kind: dict(pending_base),
        saver,
    )
    script_input = ScriptInput(name="data", node_uid="source", source_spec=full_data())
    node = types.SimpleNamespace(
        pending_workspace_payload_attrs={},
        name="new",
        tool_script_inputs=(script_input,),
        tool_primary_input="data",
        source_state="valid",
        source_auto_update=True,
    )

    attrs = saver._pending_workspace_tool_attrs(node)

    assert attrs["tool_title"] == "prefix new"
    assert json.loads(attrs[erlab.interactive.utils._TOOL_SCRIPT_INPUTS_ATTR]) == [
        script_input.model_dump(mode="json")
    ]
    assert attrs[erlab.interactive.utils._TOOL_PRIMARY_INPUT_ATTR] == "data"
    assert erlab.interactive.utils._TOOL_SOURCE_SPEC_ATTR not in attrs
    assert erlab.interactive.utils._TOOL_SOURCE_BINDING_ATTR not in attrs
    assert attrs[erlab.interactive.utils._TOOL_SOURCE_STATE_ATTR] == "valid"
    assert attrs[erlab.interactive.utils._TOOL_SOURCE_AUTO_UPDATE_ATTR] is True
    assert erlab.interactive.utils._TOOL_INPUT_PROVENANCE_SPEC_ATTR not in attrs

    pending_base.clear()
    pending_base.update(
        {
            erlab.interactive.utils._TOOL_SOURCE_SPEC_ATTR: json.dumps(
                full_data().model_dump(mode="json")
            ),
            erlab.interactive.utils._TOOL_DATA_REFERENCES_ATTR: json.dumps(
                {
                    erlab.interactive.utils._SAVED_TOOL_DATA_NAME: {
                        "kind": "parent_source"
                    }
                }
            ),
        }
    )
    attrs = saver._pending_workspace_tool_attrs(node)
    assert erlab.interactive.utils._TOOL_SOURCE_SPEC_ATTR in attrs

    node.name = "plain"
    node.tool_script_inputs = ()
    pending_base.clear()
    attrs = saver._pending_workspace_tool_attrs(node)

    assert attrs["tool_title"] == "plain"
    assert erlab.interactive.utils._TOOL_SCRIPT_INPUTS_ATTR not in attrs
    assert erlab.interactive.utils._TOOL_PRIMARY_INPUT_ATTR not in attrs
    assert erlab.interactive.utils._TOOL_SOURCE_STATE_ATTR not in attrs
    assert erlab.interactive.utils._TOOL_SOURCE_AUTO_UPDATE_ATTR not in attrs


def test_workspace_reference_uid_detection_and_invalid_reader_path() -> None:
    references = {
        "parent": {"kind": "parent_source"},
        "node": {"kind": "manager_node", "node_uid": "source"},
        "other": {"kind": "external"},
    }
    controller_type = workspace_controller._WorkspaceController
    includes_uids = controller_type._workspace_tool_references_include_uids

    assert not includes_uids(references, (), parent_uid="parent")
    assert includes_uids(references, {"parent"}, parent_uid="parent")
    assert includes_uids(references, {"source"}, parent_uid=None)
    assert not includes_uids(references, {"missing"}, parent_uid=None)

    saver = workspace_saving._WorkspaceSaver.__new__(workspace_saving._WorkspaceSaver)
    assert saver._workspace_reader_closers(typing.cast("str", None)) == ()

    closed: list[str] = []
    saver._workspace_reader_closers = types.MethodType(
        lambda _self, _path: (lambda: closed.append("closed"),), saver
    )
    saver._close_workspace_idle_readers("workspace.itws")
    assert closed == ["closed"]


def test_manager_node_reference_validation_applies_source_spec(
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(np.arange(4.0), dims="x", name="source")
    with manager_context() as manager:
        manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(data, _in_manager=True),
            show=False,
        )
        node = manager._tool_graph.root_wrappers[0]
        source_spec = full_data(IselOperation(kwargs={"x": slice(1, None)}))
        owner_tool = _AddedTimeChildTool(data.isel(x=slice(1, None)))
        owner_tool.set_script_inputs(
            (
                ScriptInput(
                    name="data",
                    source_spec=source_spec.model_dump(mode="json"),
                ),
            ),
            primary_input="data",
        )
        owner_uid = manager.add_childtool(
            owner_tool,
            script_inputs={"data": 0},
            show=False,
        )
        owner_node = manager._child_node(owner_uid)
        reference = {
            "kind": "manager_node",
            "node_uid": node.uid,
            "node_snapshot_token": node.snapshot_token,
            "input_name": "data",
            "source_spec": source_spec.model_dump(mode="json"),
        }

        controller = manager._workspace_controller
        assert controller._tool_data_reference_matches_current_data(
            reference,
            data.isel(x=slice(1, None)),
            owner_node=owner_node,
        )
        assert not controller._tool_data_reference_matches_current_data(
            reference,
            data,
            owner_node=owner_node,
        )


def test_serialize_workspace_node_rejects_invalid_pending_tool() -> None:
    saver = workspace_saving._WorkspaceSaver.__new__(workspace_saving._WorkspaceSaver)
    saver._manager = types.SimpleNamespace()
    pending = types.SimpleNamespace(
        is_imagetool=False,
        pending_workspace_payload=("workspace.itws", "/tool"),
        pending_workspace_payload_attrs={
            erlab.interactive.utils._TOOL_DATA_REFERENCES_ATTR: json.dumps(
                {
                    "data": {
                        "kind": "manager_node",
                        "node_uid": "missing-source",
                    }
                }
            )
        },
        display_text="Saved figure",
        parent_uid=None,
        uid="figure-uid",
        materialize_pending_workspace_payload=lambda: False,
    )
    saver._manager = types.SimpleNamespace(_tool_graph=types.SimpleNamespace(nodes={}))
    saver._controller = types.SimpleNamespace(
        _tool_data_reference_matches_current_snapshot=lambda _reference: False
    )
    with pytest.raises(ValueError, match=r"Saved figure.*missing-source"):
        saver._serialize_workspace_node({}, pending, "0", include_children=False)

    unsavable = types.SimpleNamespace(
        is_imagetool=False,
        pending_workspace_payload=None,
        tool_window=types.SimpleNamespace(can_save_and_load=lambda: False),
    )
    constructor: dict[str, xr.Dataset] = {}
    saver._serialize_workspace_node(constructor, unsavable, "0", include_children=False)
    assert constructor == {}


@pytest.mark.parametrize(
    ("attrs", "parent_uid", "nodes", "expected"),
    [
        (None, None, {}, ()),
        (
            {erlab.interactive.utils._TOOL_DATA_REFERENCES_ATTR: b"\xff"},
            None,
            {},
            (),
        ),
        (
            {erlab.interactive.utils._TOOL_DATA_REFERENCES_ATTR: "not-json"},
            None,
            {},
            (),
        ),
        (
            {erlab.interactive.utils._TOOL_DATA_REFERENCES_ATTR: "[]"},
            None,
            {},
            (),
        ),
        (
            {
                erlab.interactive.utils._TOOL_DATA_REFERENCES_ATTR: json.dumps(
                    {
                        "invalid": None,
                        "parent": {"kind": "parent_source"},
                        "external": {"kind": "external"},
                        "empty": {"kind": "manager_node", "node_uid": ""},
                        "available": {
                            "kind": "manager_node",
                            "node_uid": "available-source",
                        },
                    }
                ).encode()
            },
            "missing-parent",
            {"available-source": object()},
            ("missing-parent",),
        ),
        (
            {
                erlab.interactive.utils._TOOL_DATA_REFERENCES_ATTR: json.dumps(
                    {"parent": {"kind": "parent_source"}}
                )
            },
            "available-parent",
            {"available-parent": object()},
            (),
        ),
    ],
)
def test_pending_workspace_unavailable_reference_uids(
    attrs,
    parent_uid,
    nodes,
    expected,
) -> None:
    saver = workspace_saving._WorkspaceSaver.__new__(workspace_saving._WorkspaceSaver)
    saver._manager = types.SimpleNamespace(
        _tool_graph=types.SimpleNamespace(nodes=nodes)
    )
    saver._controller = types.SimpleNamespace(
        _tool_data_reference_matches_current_snapshot=lambda reference: (
            reference.get("node_uid") in nodes
        )
    )
    node = types.SimpleNamespace(
        pending_workspace_payload_attrs=attrs,
        parent_uid=parent_uid,
    )

    assert saver._pending_workspace_tool_reference_status(node)[1] == expected


def test_serialize_workspace_node_error_without_optional_metadata() -> None:
    saver = workspace_saving._WorkspaceSaver.__new__(workspace_saving._WorkspaceSaver)
    saver._manager = types.SimpleNamespace(_tool_graph=types.SimpleNamespace(nodes={}))
    pending = types.SimpleNamespace(
        is_imagetool=False,
        pending_workspace_payload=("workspace.itws", "/tool"),
        pending_workspace_payload_attrs=None,
        display_text="",
        parent_uid=None,
        uid="figure-uid",
        materialize_pending_workspace_payload=lambda: False,
    )

    with pytest.raises(ValueError, match="Could not read this saved tool"):
        saver._serialize_workspace_node({}, pending, "0", include_children=False)


def test_workspace_gc_controller_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    controller = workspace_controller._WorkspaceController.__new__(
        workspace_controller._WorkspaceController
    )
    manager = QtCore.QObject()
    manager._workspace_state = types.SimpleNamespace(
        save_in_progress=False, document_id="document", path=None
    )
    controller._manager = manager
    controller._workspace_gc_requested = False
    controller._workspace_gc_worker = None
    controller._workspace_gc_receiver = None
    controller._workspace_store = None
    controller._start_workspace_gc()

    controller._workspace_gc_requested = True
    controller._start_workspace_gc()
    assert not controller._workspace_gc_requested

    path = tmp_path / "workspace.itws"
    store = types.SimpleNamespace(closed=False, path=path)
    controller._workspace_store = store
    controller._current_workspace_document_path = types.MethodType(
        lambda _self: path, controller
    )
    controller.saving = types.SimpleNamespace(
        _workspace_reader_closers=lambda _path: ()
    )
    controller._workspace_gc_requested = True
    monkeypatch.setattr(QtCore.QThreadPool, "globalInstance", lambda: None)
    controller._start_workspace_gc()
    assert controller._workspace_gc_worker is None

    class _Pool:
        def __init__(self) -> None:
            self.workers = []

        def start(self, worker) -> None:
            self.workers.append(worker)

    pool = _Pool()
    scheduled: list[tuple[int, typing.Callable[[], None]]] = []
    monkeypatch.setattr(QtCore.QThreadPool, "globalInstance", lambda: pool)
    monkeypatch.setattr(
        QtCore.QTimer,
        "singleShot",
        lambda delay, callback: scheduled.append((delay, callback)),
    )
    controller._start_workspace_gc()
    pool.workers.pop().signals.finished.emit(False, "cleanup failed")
    assert not controller._workspace_gc_requested
    assert controller._workspace_gc_worker is None

    controller._workspace_gc_requested = True
    controller._start_workspace_gc()
    pool.workers.pop().signals.finished.emit(True, None)
    assert controller._workspace_gc_requested
    assert scheduled[-1] == (0, controller._start_workspace_gc)

    class _FailingPool:
        @staticmethod
        def start(_worker) -> None:
            raise RuntimeError("cannot start")

    controller._workspace_gc_worker = None
    controller._workspace_gc_receiver = None
    controller._workspace_gc_requested = True
    monkeypatch.setattr(QtCore.QThreadPool, "globalInstance", _FailingPool)
    controller._start_workspace_gc()
    assert controller._workspace_gc_requested
    assert controller._workspace_gc_worker is None


def test_manager_async_save_request_error_paths(
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
        manager._show_workspace_save_worker_error(
            workspace_saving._WorkspaceSaveError("Traceback text")
        )
        assert critical_calls[-1][2] == (
            "An error occurred while saving the workspace file."
        )

        missing_error = workspace_saving._WorkspaceSaveError(
            traceback_text="Missing source traceback",
            missing_source_path=str(tmp_path / "deleted-source.itws"),
        )
        manager._show_workspace_save_worker_error(missing_error)
        assert len(critical_calls) == 2

        manager._show_workspace_save_worker_error(
            workspace_saving._WorkspaceSaveError(
                traceback_text="Conflict traceback",
                publication_conflict_path=str(tmp_path / "changed.itws"),
            )
        )
        manager._show_workspace_save_worker_error(
            workspace_saving._WorkspaceSaveError(
                traceback_text="Access traceback",
                access_denied_path=str(tmp_path / "denied.itws"),
            )
        )
        assert len(critical_calls) == 4

        manager._workspace_state.path = tmp_path / "workspace.itws"
        manager._workspace_state.save_in_progress = True
        assert not manager._workspace_controller.save()

        manager._workspace_state.save_in_progress = False
        operation_errors: list[tuple[str, str]] = []
        monkeypatch.setattr(
            manager,
            "_show_operation_error",
            lambda title, text: operation_errors.append((title, text)),
        )
        monkeypatch.setattr(
            manager._workspace_controller.saving,
            "_workspace_save_snapshot",
            lambda _path: (_ for _ in ()).throw(RuntimeError("snapshot failed")),
        )
        monkeypatch.setattr(
            manager._workspace_controller,
            "_restore_focus_after_workspace_save",
            lambda _origin: None,
        )
        callback_results: list[bool] = []
        assert not manager._workspace_controller.save(
            on_finished=callback_results.append
        )
        assert callback_results == [False]
        assert operation_errors == [
            (
                "Error while saving workspace",
                "An error occurred while saving the workspace file.",
            )
        ]

        monkeypatch.setattr(
            manager._workspace_controller,
            "_workspace_save_dialog",
            lambda **_kwargs: None,
        )
        assert not manager._workspace_controller.save_as()


def test_manager_save_action_runs_workspace_save_in_background(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    pool = _install_deferred_workspace_save_worker(monkeypatch)

    with manager_context() as manager:
        operation_errors: list[tuple[str, str]] = []
        monkeypatch.setattr(
            manager,
            "_show_operation_error",
            lambda title, text: operation_errors.append((title, text)),
        )
        data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=("x", "y"))
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        select_tools(manager, [0])
        manager._update_actions()
        assert manager.offload_action.isEnabled()
        fname = _bind_dirty_workspace_for_save_test(manager, tmp_path)
        assert manager.workspace_path == str(fname.resolve())
        monkeypatch.setattr(
            manager._workspace_controller.saving,
            "_workspace_save_snapshot",
            lambda _path: _workspace_save_test_snapshot(manager),
        )
        monkeypatch.setattr(
            manager._workspace_controller,
            "save_as",
            lambda **_kwargs: pytest.fail("Save action unexpectedly used Save As"),
        )

        manager.save_action.trigger()

        assert len(pool.workers) == 1
        assert manager._workspace_state.save_in_progress
        assert not manager.save_action.isEnabled()
        assert not manager.save_as_action.isEnabled()
        assert not manager.compact_workspace_action.isEnabled()
        assert not manager.import_workspace_action.isEnabled()
        assert manager.is_workspace_modified
        manager._update_actions()
        assert not manager.offload_action.isEnabled()
        manager.tree_view.deselect_all()
        manager._update_actions()
        assert not manager.offload_action.isEnabled()

        pool.workers[0].finish()
        qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)

        assert manager.save_action.isEnabled()
        assert manager.save_as_action.isEnabled()
        assert manager.compact_workspace_action.isEnabled()
        assert manager.import_workspace_action.isEnabled()
        assert not manager.offload_action.isEnabled()
        assert not manager.is_workspace_modified
        assert not operation_errors
        manager._workspace_state.path = None


def test_manager_background_workspace_save_keeps_new_changes_and_queues_followup(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    pool = _install_deferred_workspace_save_worker(monkeypatch)

    with manager_context() as manager:
        operation_errors: list[tuple[str, str]] = []
        monkeypatch.setattr(
            manager,
            "_show_operation_error",
            lambda title, text: operation_errors.append((title, text)),
        )
        _fname = _bind_dirty_workspace_for_save_test(manager, tmp_path)
        monkeypatch.setattr(
            manager._workspace_controller.saving,
            "_workspace_save_snapshot",
            lambda _path: _workspace_save_test_snapshot(manager),
        )

        assert manager._workspace_controller.save()
        manager._workspace_controller._mark_workspace_options_dirty()
        assert not manager._workspace_controller.save()

        assert len(pool.workers) == 1
        pool.workers[0].finish()
        qtbot.wait_until(lambda: len(pool.workers) == 2)
        assert manager._workspace_state.save_in_progress
        assert manager.is_workspace_modified

        pool.workers[1].finish()
        qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)
        assert not manager.is_workspace_modified
        assert not operation_errors
        manager._workspace_state.path = None


def test_manager_background_workspace_save_keeps_duplicate_layout_change_dirty(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    pool = _install_deferred_workspace_save_worker(monkeypatch)

    with manager_context() as manager:
        _fname = _bind_dirty_workspace_for_save_test(manager, tmp_path)
        monkeypatch.setattr(
            manager._workspace_controller.saving,
            "_workspace_save_snapshot",
            lambda _path: _workspace_save_test_snapshot(manager),
        )

        assert manager._workspace_controller.save()
        assert len(pool.workers) == 1
        snapshot_generation = pool.workers[0]._snapshot.generation
        assert manager._workspace_state.mark_layout_dirty()
        assert manager._workspace_state.dirty_generation > snapshot_generation

        pool.workers[0].finish()
        qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)

        assert manager.is_workspace_modified
        assert manager._workspace_state.layout_modified
        manager._workspace_state.path = None


def test_manager_background_full_save_preserves_post_snapshot_data_edit(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    pool = _install_deferred_workspace_save_worker(monkeypatch)

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        operation_errors: list[tuple[str, str]] = []
        monkeypatch.setattr(
            manager,
            "_show_operation_error",
            lambda title, text: operation_errors.append((title, text)),
        )
        data = xr.DataArray(
            np.arange(25.0).reshape((5, 5)),
            dims=("x", "y"),
            coords={"x": np.arange(5.0), "y": np.arange(5.0)},
            name="source",
        )
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "post-snapshot-edit.itws"
        manager._workspace_controller.saving._save_workspace_document(fname)
        adopt_workspace_path(manager, fname)
        manager._workspace_controller.loading._rebind_workspace_backed_imagetools(
            fname, targets=[0], chunks={}
        )
        slicer_area = root.slicer_area
        assert slicer_area.data_chunked
        manager._workspace_controller._mark_workspace_clean()

        manager._workspace_controller._mark_node_data_dirty(uid)
        assert manager._workspace_controller.save()
        assert len(pool.workers) == 1
        assert manager._workspace_state.save_in_progress
        snapshot_generation = pool.workers[0]._snapshot.generation

        replacement = xr.DataArray(
            np.full((5, 5), 42.0),
            dims=("x", "y"),
            coords={"x": np.arange(5.0), "y": np.arange(5.0)},
            name="source",
        )
        slicer_area.replace_source_data(
            replacement,
            auto_compute=False,
            emit_edited=True,
        )
        assert any(
            event.uid == uid and event.data and event.generation > snapshot_generation
            for event in manager._workspace_state.dirty_events
        )

        pool.workers[0].finish()
        qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)

        np.testing.assert_array_equal(slicer_area._data.values, replacement.values)
        assert not slicer_area.data_chunked
        assert manager.is_workspace_modified
        assert uid in manager._workspace_state.dirty_data
        assert not operation_errors
        manager._workspace_state.path = None


def test_manager_background_save_as_preserves_post_snapshot_data_edit(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    pool = _install_deferred_workspace_save_worker(monkeypatch)

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25.0).reshape((5, 5)),
            dims=("x", "y"),
            name="source",
        )
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        source_path = tmp_path / "source.itws"
        target_path = tmp_path / "target.itws"
        manager._workspace_controller.saving._save_workspace_document(source_path)
        adopt_workspace_path(manager, source_path)
        manager._workspace_controller.loading._rebind_workspace_backed_imagetools(
            source_path, chunks={}
        )
        manager._workspace_controller._mark_workspace_clean()
        monkeypatch.setattr(
            manager._workspace_controller,
            "_workspace_save_dialog",
            lambda **_kwargs: target_path,
        )

        assert manager._workspace_controller.save_as(native=False)
        assert len(pool.workers) == 1
        worker = pool.workers[0]
        snapshot_generation = worker._snapshot.generation

        replacement = xr.full_like(data, 42.0)
        root.slicer_area.replace_source_data(
            replacement,
            auto_compute=False,
            emit_edited=True,
        )
        assert any(
            event.uid is not None and event.generation > snapshot_generation
            for event in manager._workspace_state.dirty_events
        )

        with workspace_store.WorkspaceStore(worker._fname, create=True) as store:
            workspace_storage._write_workspace_generation(
                store,
                worker._snapshot.generation_plan,
                compression_mode=worker._snapshot.compression_mode,
            )
        worker.finish()
        qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)

        assert manager.workspace_path == str(target_path.resolve())
        np.testing.assert_array_equal(root.slicer_area._data.values, replacement.values)
        assert manager.is_workspace_modified


def test_manager_background_save_as_preserves_post_snapshot_non_node_changes(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    pool = _install_deferred_workspace_save_worker(monkeypatch)

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25.0).reshape((5, 5)),
            dims=("x", "y"),
            name="source",
        )
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        source_path = tmp_path / "source.itws"
        target_path = tmp_path / "target.itws"
        manager._workspace_controller.saving._save_workspace_document(source_path)
        adopt_workspace_path(manager, source_path)
        manager._workspace_controller._mark_workspace_clean()
        monkeypatch.setattr(
            manager._workspace_controller,
            "_workspace_save_dialog",
            lambda **_kwargs: target_path,
        )

        assert manager._workspace_controller.save_as(native=False)
        assert len(pool.workers) == 1
        worker = pool.workers[0]
        snapshot_generation = worker._snapshot.generation

        assert manager._workspace_state.mark_layout_dirty()
        assert manager._workspace_state.mark_options_dirty()
        assert manager._workspace_state.mark_context_dirty()
        post_save_events = [
            event
            for event in manager._workspace_state.dirty_events
            if event.generation > snapshot_generation
        ]
        assert any(event.layout for event in post_save_events)
        assert any(event.options for event in post_save_events)
        assert any(event.context for event in post_save_events)

        with workspace_store.WorkspaceStore(worker._fname, create=True) as store:
            workspace_storage._write_workspace_generation(
                store,
                worker._snapshot.generation_plan,
                compression_mode=worker._snapshot.compression_mode,
            )
        worker.finish()
        qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)

        assert manager.workspace_path == str(target_path.resolve())
        assert manager.is_workspace_modified
        assert manager._workspace_state.layout_modified
        assert manager._workspace_state.options_modified
        assert manager._workspace_state.context_modified


def test_manager_background_save_as_preserves_change_from_final_event_drain(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    pool = _install_deferred_workspace_save_worker(monkeypatch)

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        root = itool(
            xr.DataArray(np.arange(9.0).reshape((3, 3)), dims=("x", "y")),
            manager=False,
            execute=False,
        )
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        source_path = tmp_path / "source.itws"
        target_path = tmp_path / "target.itws"
        controller = manager._workspace_controller
        controller.saving._save_workspace_document(source_path)
        adopt_workspace_path(manager, source_path)
        controller._mark_workspace_clean()
        monkeypatch.setattr(
            controller,
            "_workspace_save_dialog",
            lambda **_kwargs: target_path,
        )

        assert controller.save_as(native=False)
        worker = pool.workers[0]
        original_drain = controller._drain_workspace_deferred_events
        drain_count = 0

        def _drain_and_edit_on_final_call() -> None:
            nonlocal drain_count
            original_drain()
            drain_count += 1
            if drain_count == 2:
                assert manager._workspace_state.mark_layout_dirty()

        monkeypatch.setattr(
            controller,
            "_drain_workspace_deferred_events",
            _drain_and_edit_on_final_call,
        )
        with workspace_store.WorkspaceStore(worker._fname, create=True) as store:
            workspace_storage._write_workspace_generation(
                store,
                worker._snapshot.generation_plan,
                compression_mode=worker._snapshot.compression_mode,
            )
        worker.finish()
        qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)

        assert drain_count == 2
        assert manager.workspace_path == str(target_path.resolve())
        assert manager.is_workspace_modified
        assert manager._workspace_state.layout_modified


def test_manager_background_save_as_repoints_post_snapshot_pending_edit(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    pool = _install_deferred_workspace_save_worker(monkeypatch)

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        imported_data = xr.DataArray(
            np.arange(25, dtype=np.float64).reshape((5, 5)),
            dims=("x", "y"),
            name="imported",
        )
        imported_tool = itool(imported_data, manager=False, execute=False)
        assert isinstance(imported_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(imported_tool, show=False)
        imported_tool.hide()

        source = tmp_path / "import-source.itws"
        target = tmp_path / "import-target.itws"
        manager._workspace_controller.saving._save_workspace_document(source)
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        base_data = xr.DataArray(np.arange(4.0).reshape((2, 2)), dims=("x", "y"))
        base_tool = itool(base_data, manager=False, execute=False)
        assert isinstance(base_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(base_tool, show=False)
        base = tmp_path / "base.itws"
        manager._workspace_controller.saving._save_workspace_document(base)
        adopt_workspace_path(manager, base)
        manager._workspace_controller._mark_workspace_clean()

        assert manager._workspace_controller.loading._load_workspace_file(
            source,
            replace=False,
            associate=False,
            mark_dirty=True,
            select=False,
        )
        wrapper = next(
            node
            for node in manager._tool_graph.root_wrappers.values()
            if node.pending_workspace_memory_payload is not None
        )
        payload_path = manager._workspace_controller.saving._workspace_payload_path(
            wrapper.uid
        )
        node_path = payload_path.rsplit("/", maxsplit=1)[0]
        monkeypatch.setattr(
            manager._workspace_controller,
            "_workspace_save_dialog",
            lambda **_kwargs: target,
        )

        assert manager._workspace_controller.save_as(native=False)
        assert len(pool.workers) == 1
        worker = pool.workers[0]
        snapshot_generation = worker._snapshot.generation

        pending_attrs = wrapper.pending_workspace_payload_attrs
        assert pending_attrs is not None
        pending_attrs["post_snapshot_marker"] = True
        wrapper.update_pending_workspace_payload_attrs(pending_attrs)
        assert manager._workspace_controller._mark_node_state_dirty(wrapper.uid)
        assert any(
            event.uid == wrapper.uid and event.generation > snapshot_generation
            for event in manager._workspace_state.dirty_events
        )

        with workspace_store.WorkspaceStore(worker._fname, create=True) as target_store:
            workspace_storage._write_workspace_generation(
                target_store,
                worker._snapshot.generation_plan,
                compression_mode=worker._snapshot.compression_mode,
            )
        worker.finish()
        qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)

        assert wrapper.pending_workspace_memory_payload == (
            target.resolve(),
            _current_workspace_payload_path(target, node_path).lstrip("/"),
        )
        updated_attrs = wrapper.pending_workspace_payload_attrs
        assert updated_attrs is not None
        assert updated_attrs["post_snapshot_marker"] is True
        assert manager.is_workspace_modified

        source.unlink()
        manager.show_imagetool(wrapper.index)
        qtbot.wait_until(lambda: manager.get_imagetool(wrapper.index).isVisible())
        np.testing.assert_array_equal(
            manager.get_imagetool(wrapper.index).slicer_area._data.values,
            imported_data.values,
        )


def test_manager_background_workspace_save_failure_restores_state(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    pool = _install_deferred_workspace_save_worker(monkeypatch)

    with manager_context() as manager:
        operation_errors: list[tuple[str, str]] = []
        trust_records: list[dict[str, typing.Any]] = []
        monkeypatch.setattr(
            manager,
            "_show_operation_error",
            lambda title, text: operation_errors.append((title, text)),
        )
        monkeypatch.setattr(
            manager._workspace_controller,
            "_record_saved_workspace_code_trust",
            lambda manifest, **_kwargs: trust_records.append(manifest),
        )
        _fname = _bind_dirty_workspace_for_save_test(manager, tmp_path)
        errors: list[workspace_saving._WorkspaceSaveError] = []
        monkeypatch.setattr(
            manager._workspace_controller.saving,
            "_workspace_save_snapshot",
            lambda _path: _workspace_save_test_snapshot(manager),
        )

        monkeypatch.setattr(manager, "_show_workspace_save_worker_error", errors.append)

        assert manager._workspace_controller.save()
        assert manager._workspace_state.save_in_progress

        pool.workers[0].finish(
            error=workspace_saving._WorkspaceSaveError("worker boom"),
        )
        qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)

        assert errors
        assert "worker boom" in errors[-1].traceback_text
        assert manager.save_action.isEnabled()
        assert manager.save_as_action.isEnabled()
        assert manager.compact_workspace_action.isEnabled()
        assert manager.import_workspace_action.isEnabled()
        assert manager.is_workspace_modified
        assert not operation_errors
        assert not trust_records
        manager._workspace_controller._mark_workspace_clean()
        manager._workspace_state.path = None


def test_manager_save_slot_requests_async_save(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    pool = _install_deferred_workspace_save_worker(monkeypatch)

    with manager_context() as manager:
        operation_errors: list[tuple[str, str]] = []
        monkeypatch.setattr(
            manager,
            "_show_operation_error",
            lambda title, text: operation_errors.append((title, text)),
        )
        _fname = _bind_dirty_workspace_for_save_test(manager, tmp_path)
        monkeypatch.setattr(
            manager._workspace_controller.saving,
            "_workspace_save_snapshot",
            lambda _path: _workspace_save_test_snapshot(manager),
        )

        assert manager.save() is None
        assert len(pool.workers) == 1
        assert manager._workspace_state.save_in_progress
        assert manager.is_workspace_modified

        pool.workers[0].finish()
        qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)

        assert not manager._workspace_state.save_in_progress
        assert not manager.is_workspace_modified
        assert not operation_errors
        manager._workspace_state.path = None


def test_manager_close_ignored_while_workspace_save_in_progress(
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager._workspace_state.save_in_progress = True
        event = QtGui.QCloseEvent()
        try:
            manager.closeEvent(event)
            assert not event.isAccepted()
        finally:
            manager._workspace_state.save_in_progress = False


def test_open_multiple_files_workspace_locks_before_recovery(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    fname = tmp_path / "locked-dropped.itws"
    _write_transaction_test_workspace(fname)
    lock = workspace_storage._acquire_workspace_document_lock(fname)
    lock_calls: list[pathlib.Path] = []
    recovery_calls: list[pathlib.Path] = []
    try:
        monkeypatch.setattr(
            workspace_storage,
            "_recover_workspace_transactions",
            lambda path: (
                recovery_calls.append(pathlib.Path(path))
                or pytest.fail("Dropped workspace should lock before recovery")
            ),
        )
        monkeypatch.setattr(
            workspace_controller,
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
            manager._data_ingress.open_multiple_files([fname], try_workspace=True)
    finally:
        lock.unlock()

    assert recovery_calls == []
    assert lock_calls == [fname]


def test_manager_workspace_generation_save_open_replaces_and_binds_path(
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

        fname = tmp_path / "bound.itws"
        manager._workspace_controller.saving._save_workspace_document(fname)
        adopt_workspace_path(manager, fname)
        assert manager.workspace_path == str(fname.resolve())
        assert not manager.is_workspace_modified

        attrs = workspace_arrays._read_workspace_root_attrs_h5py(fname)
        assert attrs["imagetool_workspace_schema_version"] == 6
        manifest = workspace_format._workspace_manifest_from_attrs(attrs)
        assert manifest["schema_version"] == 6
        assert {node["uid"] for node in manifest["nodes"]} >= {
            manager._tool_graph.root_wrappers[0].uid,
            child_uid,
        }

        extra = itool(data + 2, manager=False, execute=False)
        assert isinstance(extra, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(extra, show=False)
        assert manager.ntools == 2

        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        assert manager.workspace_path == str(fname.resolve())
        assert not manager.is_workspace_modified
        assert manager._tool_graph.root_wrappers[0]._childtool_indices == [child_uid]
        assert manager.get_imagetool(0).slicer_area._data.chunks is None
        assert _compute_first_value(manager.get_imagetool(0).slicer_area._data) == 0


def test_manager_opens_schema_5_immutable_generation_workspace(
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    fname = tmp_path / "schema-5.itws"
    data = xr.DataArray(np.arange(9).reshape(3, 3), dims=("x", "y"))

    with manager_context() as manager:
        tool = itool(data, manager=False, execute=False)
        assert isinstance(tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(tool, show=False)
        manager._workspace_controller.saving._save_workspace_document(fname)

        with h5py.File(fname, "r+") as h5_file:
            h5_file.attrs["imagetool_workspace_schema_version"] = 5
            generation_root = h5_file[workspace_store._WORKSPACE_GENERATIONS_GROUP]
            for generation in generation_root.values():
                manifest = workspace_store.WorkspaceStore._read_manifest(generation)
                manifest["schema_version"] = 5
                del generation[workspace_store._WORKSPACE_MANIFEST_DATASET]
                workspace_store.WorkspaceStore._write_manifest(generation, manifest)

        extra = itool(data + 1, manager=False, execute=False)
        assert isinstance(extra, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(extra, show=False)

        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        assert manager.ntools == 1
        xr.testing.assert_identical(
            manager._get_imagetool_data(0),
            data.assign_coords(x=np.arange(3), y=np.arange(3)),
        )


def test_manager_workspace_import_ignored_while_save_in_progress(
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager._workspace_state.save_in_progress = True
        monkeypatch.setattr(
            QtWidgets,
            "QFileDialog",
            lambda *_args, **_kwargs: pytest.fail(
                "Import should not open a file dialog during workspace save"
            ),
        )

        assert not manager.import_workspace(native=False)

        manager._workspace_state.save_in_progress = False


def test_manager_workspace_load_ignored_while_save_in_progress(
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager._workspace_state.save_in_progress = True
        monkeypatch.setattr(
            QtWidgets,
            "QFileDialog",
            lambda *_args, **_kwargs: pytest.fail(
                "Open should not show a file dialog during workspace save"
            ),
        )

        assert not manager.load(native=False)

        manager._workspace_state.save_in_progress = False


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
                manager._workspace_controller.loading,
                "_load_workspace_file",
                _load_workspace_file_should_not_run,
            )

            new_fname = tmp_path / "new.itws"

            def _go_to_file(dialog: QtWidgets.QFileDialog):
                dialog.setDirectory(str(tmp_path))
                dialog.selectFile(str(new_fname))
                focused = dialog.focusWidget()
                if isinstance(focused, QtWidgets.QLineEdit):
                    focused.setText(new_fname.name)

            accept_dialog(lambda: manager.save_as(native=False), pre_call=_go_to_file)
            qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)

            assert manager.workspace_path == str(new_fname.resolve())
            assert not manager.is_workspace_modified
            assert manager.get_imagetool(0) is root
            assert manager._child_node(child_uid).imagetool is child
            assert manager._tool_graph.root_wrappers[0]._childtool_indices == [
                child_uid
            ]
            assert root.slicer_area._data.chunks is None
            assert child.slicer_area._data.chunks is None
    finally:
        object.__setattr__(dask_options, "compute_threshold", old_threshold)


def test_manager_offload_to_workspace_save_as_rebinds_root_as_dask(
    qtbot,
    accept_dialog,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25.0).reshape((5, 5)), dims=["x", "y"], name="source"
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        select_tools(manager, [0])
        manager._update_actions()
        assert manager.offload_action.isEnabled()
        assert root.slicer_area._data.chunks is None

        fname = tmp_path / "offload.itws"

        def _go_to_file(dialog: QtWidgets.QFileDialog):
            dialog.setDirectory(str(tmp_path))
            dialog.selectFile(str(fname))
            focused = dialog.focusWidget()
            if isinstance(focused, QtWidgets.QLineEdit):
                focused.setText(fname.name)

        results: list[bool] = []
        accept_dialog(
            lambda: results.append(manager.offload_to_workspace([0], native=False)),
            pre_call=_go_to_file,
        )

        assert results == [True]
        qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)
        assert manager.workspace_path == str(fname.resolve())
        assert not manager.is_workspace_modified

        rebound = manager.get_imagetool(0).slicer_area._data
        assert rebound.chunks is not None
        assert workspace_arrays._normalized_file_path(
            rebound.encoding.get("source")
        ) == (str(fname.resolve()))
        assert _compute_first_value(rebound) == 0.0

        manager._update_actions()
        assert not manager.offload_action.isEnabled()


def test_manager_workspace_load_reopens_offloaded_data_as_dask(
    qtbot,
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

        fname = tmp_path / "offload-reopen.itws"
        manager._workspace_controller.saving._save_workspace_document(fname)
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()

        assert manager.offload_to_workspace([0], native=False)
        assert root.slicer_area._data.chunks is not None
        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        loaded = manager.get_imagetool(0).slicer_area._data
        assert loaded.chunks is not None
        assert workspace_arrays._normalized_file_path(
            loaded.encoding.get("source")
        ) == (str(fname.resolve()))
        assert _compute_first_value(loaded) == 0.0


def test_manager_workspace_import_reopens_offloaded_data_as_dask(
    qtbot,
    accept_dialog,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    fname = tmp_path / "offload-import.itws"
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        manager._workspace_controller.saving._save_workspace_document(fname)
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()
        assert manager.offload_to_workspace([0], native=False)
        assert root.slicer_area._data.chunks is not None

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        loaded: list[bool] = []
        accept_dialog(
            lambda: loaded.append(
                manager._workspace_controller.loading._load_workspace_file(
                    fname,
                    replace=False,
                    associate=False,
                    mark_dirty=True,
                    select=True,
                )
            )
        )

        assert loaded == [True]
        loaded_data = manager.get_imagetool(0).slicer_area._data
        assert loaded_data.chunks is not None
        assert workspace_arrays._normalized_file_path(
            loaded_data.encoding.get("source")
        ) == str(fname.resolve())
        assert _compute_first_value(loaded_data) == 0.0


def test_manager_workspace_save_rehomes_imported_dask_data(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source = tmp_path / "source.itws"
    target = tmp_path / "target.itws"
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        imported_data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])
        imported = itool(imported_data, manager=False, execute=False)
        assert isinstance(imported, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(imported, show=False)
        manager._workspace_controller.saving._save_workspace_document(source)
        adopt_workspace_path(manager, source)
        manager._workspace_controller._mark_workspace_clean()
        assert manager.offload_to_workspace([0], native=False)

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        existing = itool(
            xr.DataArray(np.ones((2, 2)), dims=["x", "y"]),
            manager=False,
            execute=False,
        )
        assert isinstance(existing, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(existing, show=False)
        manager._workspace_controller.saving._save_workspace_document(target)
        adopt_workspace_path(manager, target)
        manager._workspace_controller._mark_workspace_clean()

        assert manager._workspace_controller.loading._load_workspace_file(
            source,
            replace=False,
            associate=False,
            mark_dirty=True,
            select=False,
        )
        imported = manager.get_imagetool(1).slicer_area._data
        assert imported.chunks is not None
        assert source.resolve() in (
            manager._workspace_controller._imported_workspace_accesses
        )
        with pytest.raises(BlockingIOError):
            workspace_storage._acquire_workspace_document_lock(source)

        assert _request_workspace_save_and_wait(qtbot, manager)
        rebound = manager.get_imagetool(1).slicer_area._data
        assert rebound.chunks is not None
        assert workspace_arrays._normalized_file_path(
            rebound.encoding.get("source")
        ) == str(target.resolve())
        assert _compute_first_value(rebound) == 0.0
        assert source.resolve() not in (
            manager._workspace_controller._imported_workspace_accesses
        )
        lock = workspace_storage._acquire_workspace_document_lock(source)
        lock.unlock()


def test_manager_workspace_load_reopens_offloaded_spaced_coord_data_as_dask(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25.0).reshape((5, 5)),
            dims=["x", "y"],
            coords={
                "x": np.arange(5.0),
                "y": np.arange(5.0),
                "Fake Motor": ("x", np.linspace(10.0, 20.0, 5)),
            },
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "offload-spaced-coord.itws"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            manager._workspace_controller.saving._save_workspace_document(fname)
        assert not any("space in its name" in str(item.message) for item in caught)
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()

        assert manager.offload_to_workspace([0], native=False)
        assert root.slicer_area._data.chunks is not None
        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        loaded = manager.get_imagetool(0).slicer_area._data
        assert loaded.chunks is not None
        assert "Fake Motor" in loaded.coords
        np.testing.assert_allclose(
            np.asarray(loaded.coords["Fake Motor"]),
            np.asarray(data.coords["Fake Motor"]),
        )


def test_manager_offload_to_workspace_saves_dirty_workspace_before_rebind(
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

        fname = tmp_path / "dirty-offload.itws"
        manager._workspace_controller.saving._save_workspace_document(fname)
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()

        updated = data + 10.0
        root.slicer_area.replace_source_data(
            updated, auto_compute=False, emit_edited=True
        )
        uid = manager._tool_graph.root_wrappers[0].uid
        assert ("snapshot-token-refresh", uid) in manager._interaction_gate.pending_keys
        assert manager.is_workspace_modified

        original_save = manager._workspace_controller.save
        save_calls: list[bool] = []

        def _save(
            *,
            native: bool = True,
            on_finished: Callable[[bool], None] | None = None,
        ) -> bool:
            save_calls.append(native)
            return original_save(native=native, on_finished=on_finished)

        monkeypatch.setattr(manager._workspace_controller, "save", _save)

        assert manager.offload_to_workspace([0], native=False)
        assert save_calls == [False]
        qtbot.wait_until(
            lambda: root.slicer_area._data.chunks is not None,
            timeout=30000,
        )

        rebound = manager.get_imagetool(0).slicer_area._data
        assert rebound.chunks is not None
        assert _compute_first_value(rebound) == 10.0
        assert (
            "snapshot-token-refresh",
            uid,
        ) not in manager._interaction_gate.pending_keys
        assert not manager.is_workspace_modified

        with h5py.File(fname, "r") as h5_file:
            saved = h5_file[_current_workspace_payload_path(fname)][_ITOOL_DATA_NAME]
            assert saved[0, 0] == 10.0


def test_manager_compute_offloaded_workspace_data_marks_backing_dirty(
    qtbot,
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
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "compute-offloaded.itws"
        manager._workspace_controller.saving._save_workspace_document(fname)
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()

        assert manager.offload_to_workspace([0], native=False)
        assert root.slicer_area._data.chunks is not None
        assert not manager.is_workspace_modified

        root.slicer_area._compute_chunked()

        assert root.slicer_area._data.chunks is None
        assert uid in manager._workspace_state.dirty_data
        assert manager.is_workspace_modified

        select_tools(manager, [0])
        manager._update_actions()
        assert manager.offload_action.isEnabled()

        assert _request_workspace_save_and_wait(qtbot, manager)
        assert not manager.is_workspace_modified

        with h5py.File(fname, "r") as h5_file:
            saved = h5_file[_current_workspace_payload_path(fname)][_ITOOL_DATA_NAME]
            assert saved.chunks is None


def test_manager_offload_to_workspace_save_cancel_or_failure_noop(
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

        monkeypatch.setattr(
            manager._workspace_controller,
            "_workspace_save_dialog",
            lambda **_kwargs: None,
        )
        assert not manager.offload_to_workspace([0], native=False)
        assert manager.workspace_path is None
        assert root.slicer_area._data.chunks is None

        fname = tmp_path / "failure-offload.itws"
        manager._workspace_controller.saving._save_workspace_document(fname)
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()
        manager._workspace_controller._mark_node_data_dirty(
            manager._tool_graph.root_wrappers[0].uid
        )

        monkeypatch.setattr(
            manager._workspace_controller, "save", lambda **_kwargs: False
        )
        assert not manager.offload_to_workspace([0], native=False)
        assert root.slicer_area._data.chunks is None


def test_manager_offload_to_workspace_edge_paths(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        calls: list[list[int]] = []
        monkeypatch.setattr(manager, "_selected_imagetool_targets", lambda: [0])
        monkeypatch.setattr(
            manager,
            "offload_to_workspace",
            lambda targets: calls.append(list(targets)) or True,
        )
        manager.offload_selected_to_workspace()
        assert calls == [[0]]

    with manager_context() as manager:
        assert not manager.offload_to_workspace([])

    fake_node = types.SimpleNamespace(
        is_imagetool=True,
        imagetool=object(),
        slicer_area=types.SimpleNamespace(data_chunked=False),
        pending_workspace_memory_payload=None,
    )

    with manager_context() as manager:
        monkeypatch.setattr(manager, "_node_for_target", lambda _target: fake_node)
        monkeypatch.setattr(
            manager._workspace_controller, "save_as", lambda **_kwargs: False
        )
        assert not manager.offload_to_workspace([0], native=False)

    with manager_context() as manager:
        workspace = tmp_path / "offload-error.itws"
        manager._workspace_state.path = workspace
        monkeypatch.setattr(manager, "_node_for_target", lambda _target: fake_node)
        monkeypatch.setattr(
            manager._workspace_controller,
            "_active_managed_window",
            lambda: typing.cast("typing.Any", None),
        )
        monkeypatch.setattr(
            erlab.interactive.utils,
            "wait_dialog",
            lambda *_args, **_kwargs: contextlib.nullcontext(),
        )
        monkeypatch.setattr(
            manager._workspace_controller.loading,
            "_rebind_workspace_backed_imagetools",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        errors: list[tuple[str, str]] = []
        restored: list[object | None] = []
        monkeypatch.setattr(
            manager,
            "_show_operation_error",
            lambda title, message: errors.append((title, message)),
        )
        monkeypatch.setattr(
            manager._workspace_controller,
            "_restore_focus_after_workspace_save",
            restored.append,
        )

        assert not manager.offload_to_workspace([0], native=False)
        assert errors == [
            (
                "Error while offloading to workspace",
                "An error occurred while reconnecting data from the workspace file.",
            )
        ]
        assert restored == [None]


def test_manager_offload_to_workspace_preserves_child_source_state(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False, provenance_spec=full_data())

        child = itool(data.copy(deep=False), manager=False, execute=False)
        assert isinstance(child, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child,
            0,
            show=False,
            source_spec=full_data(),
            source_auto_update=True,
        )
        child_node = manager._child_node(child_uid)

        fname = tmp_path / "child-offload.itws"
        manager._workspace_controller.saving._save_workspace_document(fname)
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()

        assert manager.offload_to_workspace([0], native=False)
        assert manager.get_imagetool(0).slicer_area._data.chunks is not None
        assert child_node.source_state == "fresh"
        assert child.slicer_area._data.chunks is None

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_actions()
        assert manager.offload_action.isEnabled()

        assert manager.offload_to_workspace([child_uid], native=False)
        assert child.slicer_area._data.chunks is not None
        assert child_node.source_state == "fresh"
        assert _compute_first_value(child.slicer_area._data) == 0.0

        manager._update_actions()
        assert not manager.offload_action.isEnabled()


def test_manager_manual_chunk_edits_persist_on_next_workspace_save(
    qtbot,
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
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "manual-chunks.itws"
        manager._workspace_controller.saving._save_workspace_document(fname)
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()

        root.slicer_area._set_chunks({"x": 2, "y": 3})

        assert root.slicer_area._data.chunks == ((2, 2, 1), (3, 2))
        assert uid in manager._workspace_state.dirty_data
        assert manager.is_workspace_modified

        with h5py.File(fname, "r") as h5_file:
            saved = h5_file[_current_workspace_payload_path(fname)][_ITOOL_DATA_NAME]
            assert saved.chunks is None

        assert _request_workspace_save_and_wait(qtbot, manager)
        assert not manager.is_workspace_modified

        with h5py.File(fname, "r") as h5_file:
            saved = h5_file[_current_workspace_payload_path(fname)][_ITOOL_DATA_NAME]
            assert saved.chunks == (2, 3)

        opened = workspace_arrays.open_workspace_dataset(
            fname,
            _current_workspace_payload_path(fname),
            chunks={},
        )
        try:
            rebound = opened[_ITOOL_DATA_NAME]
            assert rebound.chunks == ((2, 2, 1), (3, 2))
        finally:
            opened.close()


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
            manager._workspace_controller.saving._save_workspace_document(fname)
            adopt_workspace_path(manager, fname)
            manager._workspace_controller._mark_workspace_clean()

            assert _request_workspace_save_and_wait(qtbot, manager)
            assert root.slicer_area._data.chunks is None
            assert _compute_first_value(root.slicer_area._data) == 0.0
    finally:
        object.__setattr__(dask_options, "compute_threshold", old_threshold)


def test_manager_workspace_save_as_preserves_external_non_dask_file_backed_data(
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
            workspace_arrays._normalized_file_path(live_data.encoding.get("source"))
            == old_source
        )

        def _go_to_file(dialog: QtWidgets.QFileDialog):
            dialog.setDirectory(str(tmp_path))
            dialog.selectFile(str(new_fname))
            focused = dialog.focusWidget()
            if isinstance(focused, QtWidgets.QLineEdit):
                focused.setText(new_fname.name)

        accept_dialog(lambda: manager.save_as(native=False), pre_call=_go_to_file)
        qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)

        rebound = manager.get_imagetool(0).slicer_area._data
        new_source = str(new_fname.resolve())
        rebound_source = workspace_arrays._normalized_file_path(
            rebound.encoding.get("source")
        )
        assert rebound_source == old_source
        assert rebound_source != new_source
        assert rebound.chunks is None

        assert _compute_first_value(rebound) == 0.0


def test_manager_workspace_save_as_rebinds_workspace_non_dask_file_backed_data(
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
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        old_fname = tmp_path / "old.itws"
        new_fname = tmp_path / "new.itws"
        manager._workspace_controller.saving._save_workspace_document(old_fname)
        adopt_workspace_path(manager, old_fname)
        manager._workspace_controller.loading._rebind_workspace_backed_imagetools(
            old_fname, targets=[0], chunks=None
        )

        live_data = manager.get_imagetool(0).slicer_area._data
        old_source = str(old_fname.resolve())
        assert (
            workspace_arrays._normalized_file_path(live_data.encoding.get("source"))
            == old_source
        )
        assert live_data.chunks is None

        def _go_to_file(dialog: QtWidgets.QFileDialog):
            dialog.setDirectory(str(tmp_path))
            dialog.selectFile(str(new_fname))
            focused = dialog.focusWidget()
            if isinstance(focused, QtWidgets.QLineEdit):
                focused.setText(new_fname.name)

        accept_dialog(lambda: manager.save_as(native=False), pre_call=_go_to_file)
        qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)

        rebound = manager.get_imagetool(0).slicer_area._data
        new_source = str(new_fname.resolve())
        assert (
            workspace_arrays._normalized_file_path(rebound.encoding.get("source"))
            == new_source
        )
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
        qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)

        rebound = manager.get_imagetool(0).slicer_area._data
        assert rebound.chunks is not None
        assert workspace_arrays._normalized_file_path(
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
            uid = manager._tool_graph.root_wrappers[0].uid

            def _load_workspace_file_should_not_run(*args, **kwargs):
                raise AssertionError("Save As should not reload the saved workspace")

            monkeypatch.setattr(
                manager._workspace_controller.loading,
                "_load_workspace_file",
                _load_workspace_file_should_not_run,
            )
            rebind_calls: list[str] = []
            rebind_data = (
                manager._workspace_controller.loading._workspace_rebind_data_for_uid
            )

            def _record_rebind(fname, node_uid: str, *, chunks):
                rebind_calls.append(node_uid)
                return rebind_data(fname, node_uid, chunks=chunks)

            monkeypatch.setattr(
                manager._workspace_controller.loading,
                "_workspace_rebind_data_for_uid",
                _record_rebind,
            )

            def _go_to_file(dialog: QtWidgets.QFileDialog):
                dialog.setDirectory(str(tmp_path))
                dialog.selectFile(str(new_fname))
                focused = dialog.focusWidget()
                if isinstance(focused, QtWidgets.QLineEdit):
                    focused.setText(new_fname.name)

            accept_dialog(lambda: manager.save_as(native=False), pre_call=_go_to_file)
            qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)

            assert manager.workspace_path == str(new_fname.resolve())
            rebound = manager.get_imagetool(0).slicer_area._data
            assert rebound.chunks is not None
            assert workspace_arrays._normalized_file_path(
                rebound.encoding.get("source")
            ) == str(new_fname.resolve())
            assert rebind_calls == [uid]
            old_lazy.close()
            old_fname.unlink()
            assert _compute_first_value(rebound) == 0
    finally:
        object.__setattr__(dask_options, "compute_threshold", old_threshold)


def test_manager_workspace_save_as_retargets_private_dask_without_rebind(
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

        old_fname = tmp_path / "private-reader-old.itws"
        new_fname = tmp_path / "private-reader-new.itws"
        manager._workspace_controller.saving._save_workspace_document(old_fname)
        adopt_workspace_path(manager, old_fname)
        manager._workspace_controller.loading._rebind_workspace_backed_imagetools(
            old_fname, targets=[0], chunks={}
        )
        manager._workspace_controller._mark_workspace_clean()

        live_data = root.slicer_area._data
        assert live_data.chunks is not None
        assert workspace_arrays.dataarray_source_paths(live_data) == (
            str(old_fname.resolve()),
        )

        def _fail_rebind(*_args, **_kwargs) -> None:
            raise AssertionError("private workspace readers must not be reopened")

        monkeypatch.setattr(
            manager._workspace_controller.loading,
            "_rebind_workspace_backed_imagetools",
            _fail_rebind,
        )
        monkeypatch.setattr(
            manager._workspace_controller,
            "_workspace_save_dialog",
            lambda **_kwargs: new_fname,
        )

        assert _request_workspace_save_as_and_wait(qtbot, manager, native=False)

        rebound = root.slicer_area._data
        assert rebound is live_data
        assert workspace_arrays.dataarray_source_paths(rebound) == (
            str(new_fname.resolve()),
        )
        old_fname.unlink()
        assert _compute_first_value(rebound) == 0


def test_workspace_full_save_external_dask_rebind_rolls_back_on_later_failure(
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
        external_fname = tmp_path / "rollback-external.h5"
        workspace_fname = tmp_path / "rollback.itws"
        xr.DataTree.from_dict({"0/imagetool": data.to_dataset(name="data")}).to_netcdf(
            external_fname, engine="h5netcdf", invalid_netcdf=True
        )
        external = _open_external_lazy_hdf5_imagetool_data(external_fname)

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root.slicer_area.replace_source_data(external + 0, auto_compute=False)
        original_data = root.slicer_area._data
        original_dask_name = original_data.data.name
        original_source_paths = workspace_arrays.dataarray_source_paths(original_data)
        original_name = manager._tool_graph.root_wrappers[0].name
        backing_snapshot = (
            manager._workspace_controller.loading._workspace_data_backing_snapshot()
        )
        uid = manager._tool_graph.root_wrappers[0].uid
        manager._workspace_controller.saving._save_workspace_document(workspace_fname)

        rebind_calls: list[str] = []
        rebind_data = (
            manager._workspace_controller.loading._workspace_rebind_data_for_uid
        )

        def _record_rebind(fname, node_uid: str, *, chunks):
            rebind_calls.append(node_uid)
            return rebind_data(fname, node_uid, chunks=chunks)

        monkeypatch.setattr(
            manager._workspace_controller.loading,
            "_workspace_rebind_data_for_uid",
            _record_rebind,
        )
        monkeypatch.setattr(
            manager._workspace_controller,
            "_refresh_workspace_tool_data_after_full_save",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                workspace_controller._WorkspacePostSaveBindingError("later failure")
            ),
        )

        with pytest.raises(
            workspace_controller._WorkspacePostSaveBindingError,
            match="later failure",
        ):
            manager._workspace_controller._refresh_workspace_payload_bindings_after_full_save(
                workspace_fname,
                backing_snapshot=backing_snapshot,
                old_workspace_path=None,
            )

        assert rebind_calls == [uid]
        restored_data = root.slicer_area._data
        assert restored_data.data.name == original_dask_name
        assert (
            workspace_arrays.dataarray_source_paths(restored_data)
            == original_source_paths
        )
        assert manager._tool_graph.root_wrappers[0].name == original_name
        assert _compute_first_value(restored_data) == 0
        external.close()


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
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "deferred-dirty.itws"
        manager._workspace_controller.saving._save_workspace_document(fname)
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()

        QtCore.QTimer.singleShot(
            0, lambda: manager._workspace_controller._mark_node_state_dirty(uid)
        )
        manager._workspace_controller._mark_node_state_dirty(uid)
        assert manager.is_workspace_modified
        assert not root.isWindowModified()

        manager._flush_idle_work(force=True)

        assert root.isWindowModified()

        focus_restored: list[QtWidgets.QWidget | None] = []
        monkeypatch.setattr(
            manager._workspace_controller, "_active_managed_window", lambda: root
        )
        monkeypatch.setattr(
            manager._workspace_controller,
            "_restore_focus_after_workspace_save",
            lambda origin: focus_restored.append(origin),
        )
        assert _request_workspace_save_and_wait(qtbot, manager)
        manager._workspace_controller._drain_workspace_deferred_events()
        assert not manager.is_workspace_modified
        assert not root.isWindowModified()
        assert focus_restored == [root]


def test_manager_workspace_save_during_active_interaction_uses_dirty_state(
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
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "active-save.itws"
        manager._workspace_controller.saving._save_workspace_document(fname)
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()

        manager._note_interaction_activity()
        manager._workspace_controller._mark_node_state_dirty(uid)

        assert manager.is_workspace_modified
        assert not root.isWindowModified()

        assert _request_workspace_save_and_wait(qtbot, manager)

        assert not manager.is_workspace_modified
        assert not root.isWindowModified()

        manager._flush_idle_work(force=True)

        assert not manager.is_workspace_modified
        assert not root.isWindowModified()


def test_manager_workspace_save_drain_does_not_force_deferred_delete(
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager, monkeypatch.context() as save_drain_patch:
        event_types: list[int] = []
        idle_flushes: list[dict[str, object]] = []
        save_drain_patch.setattr(
            QtWidgets.QApplication,
            "sendPostedEvents",
            lambda _receiver, event_type: event_types.append(event_type),
        )
        save_drain_patch.setattr(
            QtWidgets.QApplication, "processEvents", lambda *_args, **_kwargs: None
        )
        save_drain_patch.setattr(
            manager, "_flush_idle_work", lambda **kwargs: idle_flushes.append(kwargs)
        )

        manager._workspace_controller._drain_workspace_deferred_events()

        assert event_types == [int(QtCore.QEvent.Type.MetaCall.value)] * 6
        assert idle_flushes == [{"force": True}]


def test_manager_workspace_state_save_reuses_immutable_payload(
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
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "state-delta.itws"
        manager._workspace_controller.saving._save_workspace_document(fname)
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()
        before_manifest = _current_workspace_manifest(fname)
        before_entry = next(
            entry
            for entry in workspace_format._iter_workspace_manifest_node_entries(
                before_manifest
            )
            if entry.get("uid") == uid
        )
        manager._workspace_controller._mark_node_state_dirty(uid)

        monkeypatch.setattr(
            workspace_arrays,
            "_write_workspace_dataset_group_to_file",
            lambda *args, **kwargs: pytest.fail(
                "state-only save must not write a payload object"
            ),
        )
        assert _request_workspace_save_and_wait(qtbot, manager)
        after_manifest = _current_workspace_manifest(fname)
        after_entry = next(
            entry
            for entry in workspace_format._iter_workspace_manifest_node_entries(
                after_manifest
            )
            if entry.get("uid") == uid
        )
        assert after_entry["payload_object_id"] == before_entry["payload_object_id"]
        assert "payload_attrs" in after_entry
        assert not manager.is_workspace_modified
        assert not root.isWindowModified()


def test_workspace_save_as_snapshot_preserves_live_history(
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
        uid = manager._tool_graph.root_wrappers[0].uid

        source_path = tmp_path / "source.itws"
        manager._workspace_controller.saving._save_workspace_document(source_path)
        adopt_workspace_path(manager, source_path)
        manager._workspace_controller.loading._rebind_workspace_backed_imagetools(
            source_path, chunks={}
        )
        manager._workspace_controller._mark_workspace_clean()
        old_data = root.slicer_area._data
        old_entry = next(
            entry
            for entry in workspace_format._iter_workspace_manifest_node_entries(
                _current_workspace_manifest(source_path)
            )
            if entry.get("uid") == uid
        )
        old_object_id = typing.cast("str", old_entry["payload_object_id"])

        store = manager._workspace_controller._workspace_store
        assert store is not None
        with store.write_session() as h5_file:
            workspace_arrays._write_workspace_dataset_group_to_file(
                h5_file,
                "/legacy/imagetool",
                data.to_dataset(name="data"),
                compression_mode="none",
            )
        legacy_data = workspace_arrays.open_workspace_dataset(
            source_path, "/legacy/imagetool", chunks={}
        )
        root.slicer_area.replace_source_data(data + 100, auto_compute=False)
        manager._workspace_controller._mark_node_data_dirty(uid)

        snapshot = (
            manager._workspace_controller.saving._workspace_generation_save_snapshot(
                manager._workspace_state.dirty_generation,
                fname=tmp_path / "target.itws",
            )
        )
        try:
            object_ids = {item.object_id for item in snapshot.generation_plan.objects}
            assert old_object_id in object_ids
            assert snapshot.generation_plan.preserved_groups == (
                workspace_storage._WorkspaceGroupCopy(
                    source_file=str(source_path.resolve()),
                    source_path="/legacy/imagetool",
                    target_path="/legacy/imagetool",
                ),
            )
            assert _compute_first_value(old_data) == 0
            assert float(legacy_data["data"].sum().compute()) == 300.0
        finally:
            snapshot.close()
            legacy_data.close()


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
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "live-handles.itws"
        manager._workspace_controller.saving._save_workspace_document(fname)
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()
        manager._workspace_controller._mark_node_state_dirty(uid)

        assert _request_workspace_save_and_wait(qtbot, manager)

        manager._workspace_controller._mark_node_data_dirty(uid)
        assert _request_workspace_save_and_wait(qtbot, manager)


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
        manager._workspace_controller.saving._save_workspace_document(fname)
        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        root = manager.get_imagetool(0)
        uid = manager._tool_graph.root_wrappers[0].uid
        root.slicer_area.replace_source_data(
            manager._workspace_controller.loading._workspace_rebind_data_for_uid(
                fname, uid, chunks="auto"
            ),
            auto_compute=False,
        )
        live_data = root.slicer_area._data
        assert live_data.chunks is not None
        assert _compute_first_value(live_data) == 0.0

        manager._workspace_controller._mark_node_state_dirty(uid)
        original_write = workspace_storage._write_workspace_generation
        computed_values: list[object] = []

        def _slow_write_workspace_generation(*args, **kwargs):
            time.sleep(0.05)
            return original_write(*args, **kwargs)

        def _compute_live_data() -> None:
            computed_values.append(live_data.isel({"x": 1, "y": 1}).compute().item())

        monkeypatch.setattr(
            workspace_storage,
            "_write_workspace_generation",
            _slow_write_workspace_generation,
        )
        QtCore.QTimer.singleShot(10, _compute_live_data)

        assert _request_workspace_save_and_wait(qtbot, manager)
        assert computed_values == [6.0]


def test_manager_workspace_slow_save_reports_status_after_background_write(
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
        manager._workspace_controller.saving._save_workspace_document(fname)
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()
        manager._workspace_controller._mark_node_state_dirty(
            manager._tool_graph.root_wrappers[0].uid
        )

        original_write = workspace_storage._write_workspace_generation

        def _slow_write_workspace_generation(*args, **kwargs):
            time.sleep(0.05)
            return original_write(*args, **kwargs)

        focus_restored: list[QtWidgets.QWidget | None] = []
        monkeypatch.setattr(
            erlab.interactive.imagetool.manager._workspace._controller,
            "_WORKSPACE_SAVE_WAIT_DIALOG_THRESHOLD_SECONDS",
            0.01,
        )
        monkeypatch.setattr(
            workspace_storage,
            "_write_workspace_generation",
            _slow_write_workspace_generation,
        )
        monkeypatch.setattr(
            manager._workspace_controller, "_active_managed_window", lambda: root
        )
        monkeypatch.setattr(
            manager._workspace_controller,
            "_restore_focus_after_workspace_save",
            lambda origin: focus_restored.append(origin),
        )

        assert _request_workspace_save_and_wait(qtbot, manager)
        assert manager._status_bar.currentMessage().startswith("Workspace saved")
        assert focus_restored == [root]


def test_manager_workspace_save_keeps_post_command_changes_dirty(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    pool = _install_deferred_workspace_save_worker(monkeypatch)

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "post-command-dirty.itws"
        manager._workspace_controller.saving._save_workspace_document(fname)
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()
        manager._workspace_controller._mark_node_data_dirty(uid)

        assert manager._workspace_controller.save()
        manager._workspace_controller._mark_node_state_dirty(uid)
        pool.workers[0].finish()
        qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)
        assert manager.is_workspace_modified
        assert root.isWindowModified()
        details = manager._workspace_controller._dirty_details_text()
        assert "State modified:" in details
        assert "Data modified:" not in details


def test_manager_workspace_compact_drops_history_and_keeps_store(
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
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "compact.itws"
        manager._workspace_controller.saving._save_workspace_document(fname)
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()
        initial_entry = next(
            entry
            for entry in workspace_format._iter_workspace_manifest_node_entries(
                _current_workspace_manifest(fname)
            )
            if entry.get("uid") == uid
        )
        root.slicer_area.replace_source_data(
            data + 10,
            auto_compute=False,
            emit_edited=True,
        )
        assert _request_workspace_save_and_wait(qtbot, manager)
        current_manifest = _current_workspace_manifest(fname)
        current_entry = next(
            entry
            for entry in workspace_format._iter_workspace_manifest_node_entries(
                current_manifest
            )
            if entry.get("uid") == uid
        )
        assert current_entry["payload_object_id"] != initial_entry["payload_object_id"]
        store = manager._workspace_controller._workspace_store
        assert store is not None
        assert len(store.h5_file[workspace_store._WORKSPACE_OBJECTS_GROUP]) >= 2

        monkeypatch.setattr(
            erlab.interactive.utils,
            "wait_dialog",
            lambda *args, **kwargs: contextlib.nullcontext(),
        )

        assert manager.compact_workspace()
        assert manager._workspace_controller._workspace_store is store
        assert workspace_store.WorkspaceStore.active(fname) is store
        assert store.h5_file.id.valid
        _assert_no_workspace_internal_groups(fname)
        assert set(store.h5_file[workspace_store._WORKSPACE_OBJECTS_GROUP]) == {
            current_entry["payload_object_id"]
        }
        generations = store.generations()
        assert len(generations) == 2
        assert generations[0].manifest == generations[1].manifest == current_manifest


def test_manager_workspace_save_deduplicates_legacy_payload_in_place(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    path = tmp_path / "schema-4.itws"
    data = xr.DataArray(
        np.arange(400, dtype=np.float64).reshape((20, 20)),
        dims=("x", "y"),
    )

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        tool = itool(data, manager=False, execute=False)
        if not isinstance(tool, erlab.interactive.imagetool.ImageTool):
            raise TypeError("Expected an ImageTool")
        manager.add_imagetool(tool, show=False)

        tree = manager._workspace_controller.saving._to_datatree()
        manifest = manager._workspace_controller.saving._workspace_manifest()
        manifest["schema_version"] = 4
        tree.attrs["imagetool_workspace_schema_version"] = 4
        tree.attrs[workspace_format._WORKSPACE_MANIFEST_ATTR] = json.dumps(manifest)
        tree.to_netcdf(path, engine="h5netcdf", invalid_netcdf=True)
        tree.close()

        manager.remove_all_tools()
        assert manager._workspace_controller.loading._load_workspace_file(
            path,
            replace=True,
            associate=True,
            mark_dirty=False,
            select=False,
        )
        assert _request_workspace_save_and_wait(qtbot, manager)

        store = manager._workspace_controller._workspace_store
        if store is None:
            raise RuntimeError("Expected an associated workspace store")
        entry = next(
            workspace_format._iter_workspace_manifest_node_entries(
                store.current_generation().manifest
            )
        )
        object_path = str(entry["payload_path"])
        legacy_path = "/0/imagetool"
        assert legacy_path in store.h5_file
        assert not store.leased_legacy_group_paths
        assert (
            h5py.h5o.get_info(store.h5_file[legacy_path].id).addr
            == h5py.h5o.get_info(store.h5_file[object_path].id).addr
        )
        np.testing.assert_array_equal(manager._get_imagetool_data(0), data)


def test_manager_workspace_save_as_and_compact_deduplicates_legacy_payload(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    old_path = tmp_path / "schema-4.itws"
    new_path = tmp_path / "converted.itws"
    data = xr.DataArray(
        np.arange(400, dtype=np.float64).reshape((20, 20)),
        dims=("x", "y"),
    )

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        tool = itool(data, manager=False, execute=False)
        if not isinstance(tool, erlab.interactive.imagetool.ImageTool):
            raise TypeError("Expected an ImageTool")
        manager.add_imagetool(tool, show=False)

        tree = manager._workspace_controller.saving._to_datatree()
        manifest = manager._workspace_controller.saving._workspace_manifest()
        manifest["schema_version"] = 4
        tree.attrs["imagetool_workspace_schema_version"] = 4
        tree.attrs[workspace_format._WORKSPACE_MANIFEST_ATTR] = json.dumps(manifest)
        tree.to_netcdf(old_path, engine="h5netcdf", invalid_netcdf=True)
        tree.close()

        manager.remove_all_tools()
        assert manager._workspace_controller.loading._load_workspace_file(
            old_path,
            replace=True,
            associate=True,
            mark_dirty=False,
            select=False,
        )
        monkeypatch.setattr(
            manager._workspace_controller,
            "_workspace_save_dialog",
            lambda **_kwargs: str(new_path),
        )
        assert _request_workspace_save_as_and_wait(qtbot, manager, native=False)

        store = manager._workspace_controller._workspace_store
        if store is None:
            raise RuntimeError("Expected an associated workspace store")
        manifest = store.current_generation().manifest
        entry = next(workspace_format._iter_workspace_manifest_node_entries(manifest))
        object_path = str(entry["payload_path"])
        legacy_path = "/0/imagetool"
        assert legacy_path in store.h5_file
        assert not store.leased_legacy_group_paths
        assert (
            h5py.h5o.get_info(store.h5_file[legacy_path].id).addr
            == h5py.h5o.get_info(store.h5_file[object_path].id).addr
        )
        np.testing.assert_array_equal(manager._get_imagetool_data(0), data)

        with store.write_session() as h5_file:
            del h5_file[legacy_path]
            h5_file.copy(object_path, legacy_path)
        assert (
            h5py.h5o.get_info(store.h5_file[legacy_path].id).addr
            != h5py.h5o.get_info(store.h5_file[object_path].id).addr
        )
        stale_legacy = workspace_arrays.open_workspace_dataset(
            new_path, legacy_path, chunks=None
        )
        assert store.leased_legacy_group_paths == {legacy_path}
        assert manager._workspace_controller.loading._load_workspace_file(
            new_path,
            replace=True,
            associate=True,
            mark_dirty=False,
            select=False,
        )
        assert store.leased_legacy_group_paths == {legacy_path}

        monkeypatch.setattr(
            erlab.interactive.utils,
            "wait_dialog",
            lambda *args, **kwargs: contextlib.nullcontext(),
        )
        assert manager.compact_workspace()
        assert legacy_path not in store.h5_file
        np.testing.assert_array_equal(manager._get_imagetool_data(0), data)
        np.testing.assert_array_equal(stale_legacy[_ITOOL_DATA_NAME], data)
        stale_legacy.close()


def test_manager_workspace_save_as_preserves_legacy_dependency_for_dirty_data(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    old_path = tmp_path / "schema-4.itws"
    new_path = tmp_path / "converted.itws"
    data = xr.DataArray(
        np.arange(400, dtype=np.float64).reshape((20, 20)),
        dims=("x", "y"),
    )

    with manager_context() as manager:
        tool = itool(data, manager=False, execute=False)
        if not isinstance(tool, erlab.interactive.imagetool.ImageTool):
            raise TypeError("Expected an ImageTool")
        manager.add_imagetool(tool, show=False)
        tree = manager._workspace_controller.saving._to_datatree()
        manifest = manager._workspace_controller.saving._workspace_manifest()
        manifest["schema_version"] = 4
        manifest["nodes"][0]["data_backing"] = "dask"
        tree.attrs["imagetool_workspace_schema_version"] = 4
        tree.attrs[workspace_format._WORKSPACE_MANIFEST_ATTR] = json.dumps(manifest)
        tree.to_netcdf(old_path, engine="h5netcdf", invalid_netcdf=True)
        tree.close()

        manager.remove_all_tools()
        assert manager._workspace_controller.loading._load_workspace_file(
            old_path,
            replace=True,
            associate=True,
            mark_dirty=False,
            select=False,
        )
        dirty_data = manager._get_imagetool_data(0)
        if dirty_data is None:
            raise RuntimeError("Expected loaded ImageTool data")
        dirty_data = dirty_data + 1.0
        expected = data + 1.0
        manager.get_imagetool(0).slicer_area.replace_source_data(
            dirty_data,
            auto_compute=False,
            emit_edited=True,
        )
        monkeypatch.setattr(
            manager._workspace_controller,
            "_workspace_save_dialog",
            lambda **_kwargs: str(new_path),
        )
        assert _request_workspace_save_as_and_wait(qtbot, manager, native=False)

        store = manager._workspace_controller._workspace_store
        if store is None:
            raise RuntimeError("Expected an associated workspace store")
        entry = next(
            workspace_format._iter_workspace_manifest_node_entries(
                store.current_generation().manifest
            )
        )
        object_path = str(entry["payload_path"])
        legacy_path = "/0/imagetool"
        assert store.leased_legacy_group_paths == {legacy_path}
        assert (
            h5py.h5o.get_info(store.h5_file[legacy_path].id).addr
            != h5py.h5o.get_info(store.h5_file[object_path].id).addr
        )
        np.testing.assert_array_equal(manager._get_imagetool_data(0), expected)

        monkeypatch.setattr(
            erlab.interactive.utils,
            "wait_dialog",
            lambda *args, **kwargs: contextlib.nullcontext(),
        )
        assert manager.compact_workspace()
        assert legacy_path in store.h5_file
        np.testing.assert_array_equal(manager._get_imagetool_data(0), expected)
        np.testing.assert_array_equal(dirty_data, expected)


def test_manager_workspace_upgrade_repoints_pending_payload_before_compaction(
    qtbot,
    tmp_path,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(30).reshape((5, 6)),
            dims=["x", "y"],
            coords={"x": np.arange(5), "y": np.arange(6)},
        )
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "schema-4-pending.itws"
        tree = manager._workspace_controller.saving._to_datatree()
        manifest = manager._workspace_controller.saving._workspace_manifest()
        manifest["schema_version"] = 4
        manifest["nodes"][0]["data_backing"] = "memory"
        tree.attrs["imagetool_workspace_schema_version"] = 4
        tree.attrs[workspace_format._WORKSPACE_MANIFEST_ATTR] = json.dumps(manifest)
        payload_attrs = tree["0/imagetool"].attrs
        payload_attrs.pop("itool_window_state", None)
        payload_attrs["itool_visible"] = False
        tree.to_netcdf(fname, engine="h5netcdf", invalid_netcdf=True)
        tree.close()

        manager.remove_all_tools()
        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        wrapper = manager._tool_graph.root_wrappers[0]
        assert wrapper.pending_workspace_memory_payload == (
            fname.resolve(),
            "0/imagetool",
        )

        manager._workspace_controller._mark_node_state_dirty(wrapper.uid)
        assert _request_workspace_save_and_wait(qtbot, manager)
        payload_path = _current_workspace_payload_path(fname)
        assert wrapper.pending_workspace_memory_payload == (
            fname.resolve(),
            payload_path.lstrip("/"),
        )

        monkeypatch.setattr(
            erlab.interactive.utils,
            "wait_dialog",
            lambda *args, **kwargs: contextlib.nullcontext(),
        )
        assert manager.compact_workspace()
        operation_errors: list[tuple[object, ...]] = []
        monkeypatch.setattr(
            manager,
            "_show_operation_error",
            lambda *args: operation_errors.append(args),
        )
        pending = manager._workspace_controller.loading.pending
        assert pending._materialize_pending_workspace_payload(wrapper)
        assert not operation_errors
        np.testing.assert_array_equal(wrapper.slicer_area._data.values, data.values)


def test_manager_releases_eager_legacy_source_when_conversion_is_canceled(
    qtbot,
    tmp_path,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    fname = tmp_path / "schema-2-canceled.itws"
    data = xr.DataArray(
        np.arange(30).reshape((5, 6)),
        dims=["x", "y"],
    )
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        tree = manager._workspace_controller.saving._to_datatree()
        tree.attrs["imagetool_workspace_schema_version"] = 2
        tree.attrs.pop(workspace_format._WORKSPACE_MANIFEST_ATTR, None)
        tree.to_netcdf(fname, engine="h5netcdf", invalid_netcdf=True)
        tree.close()
        manager.remove_all_tools()

        monkeypatch.setattr(
            manager._workspace_controller,
            "_save_legacy_workspace_as_current",
            lambda *args, **kwargs: None,
        )
        assert manager._workspace_controller.loading._load_workspace_file(
            fname,
            replace=True,
            associate=True,
            mark_dirty=False,
            select=False,
        )

        loaded = manager.get_imagetool(0).slicer_area._data
        assert manager._workspace_state.path is None
        assert loaded.chunks is None
        assert _compute_first_value(loaded) == 0.0
        assert fname.resolve() not in (
            manager._workspace_controller._imported_workspace_accesses
        )
        assert workspace_store.WorkspaceStore.active(fname) is None
        lock = workspace_storage._acquire_workspace_document_lock(fname)
        lock.unlock()


def test_manager_workspace_save_collects_old_generations_in_background(
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

        fname = tmp_path / "generation-gc.itws"
        manager._workspace_controller.saving._save_workspace_document(fname)
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()
        first_object_id = next(
            iter(
                workspace_store.WorkspaceStore.manifest_object_ids(
                    _current_workspace_manifest(fname)
                )
            )
        )

        for offset in (10, 20):
            root.slicer_area.replace_source_data(
                data + offset,
                auto_compute=False,
                emit_edited=True,
            )
            assert _request_workspace_save_and_wait(qtbot, manager)
            qtbot.wait_until(
                lambda: (
                    manager._workspace_controller._workspace_gc_worker is None
                    and not manager._workspace_controller._workspace_gc_requested
                ),
                timeout=5000,
            )

        store = manager._workspace_controller._workspace_store
        assert store is not None
        assert len(store.generations()) == 2
        assert (
            first_object_id
            not in store.h5_file[workspace_store._WORKSPACE_OBJECTS_GROUP]
        )
        assert len(store.h5_file[workspace_store._WORKSPACE_OBJECTS_GROUP]) == 2


def test_manager_workspace_save_snapshot_uses_compression_override(
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
        manager._set_workspace_option_overrides(
            {"io/workspace/compression": "blosclz3"}
        )
        assert (
            manager._workspace_controller.saving._workspace_compression_mode()
            == "blosclz3"
        )
        manifest = manager._workspace_controller.saving._workspace_manifest()
        assert "delta_save_count" not in manifest
        assert "estimated_obsolete_bytes" not in manifest

        snapshot = manager._workspace_controller.saving._workspace_save_snapshot(
            tmp_path / "snapshot.itws"
        )
        try:
            assert snapshot.compression_mode == "blosclz3"
        finally:
            snapshot.close()


def test_workspace_save_worker_start_and_finish_error_branches(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    class Snapshot:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    with manager_context() as manager, monkeypatch.context() as worker_patch:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        controller = manager._workspace_controller

        start_errors: list[str] = []
        snapshot = Snapshot()
        worker_patch.setattr(
            workspace_controller.QtCore.QThreadPool,
            "globalInstance",
            staticmethod(lambda: None),
        )
        assert not controller._start_workspace_save_worker(
            tmp_path / "none.itws",
            typing.cast("workspace_saving._WorkspaceSaveSnapshot", snapshot),
            on_finished=lambda *_args: None,
            on_start_error=lambda: start_errors.append("none"),
        )
        assert snapshot.closed
        assert start_errors == ["none"]

        class RaisingPool:
            def start(self, _worker) -> None:
                raise RuntimeError("cannot start")

        snapshot = Snapshot()
        worker_patch.setattr(
            workspace_controller.QtCore.QThreadPool,
            "globalInstance",
            staticmethod(lambda: RaisingPool()),
        )
        assert not controller._start_workspace_save_worker(
            tmp_path / "raise.itws",
            typing.cast("workspace_saving._WorkspaceSaveSnapshot", snapshot),
            on_finished=lambda *_args: None,
            on_start_error=lambda: start_errors.append("raise"),
        )
        assert snapshot.closed
        assert start_errors == ["none", "raise"]
        assert not manager._workspace_state.save_in_progress
        assert controller._background_save_worker is None
        assert controller._background_save_receiver is None

        class RecordingPool:
            def __init__(self) -> None:
                self.worker = None

            def start(self, worker) -> None:
                self.worker = worker

        errors: list[tuple[str, str]] = []
        status_messages: list[tuple[typing.Any, ...]] = []
        pool = RecordingPool()
        worker_patch.setattr(
            workspace_controller.QtCore.QThreadPool,
            "globalInstance",
            staticmethod(lambda: pool),
        )
        worker_patch.setattr(
            manager,
            "_show_operation_error",
            lambda title, text: errors.append((title, text)),
        )
        worker_patch.setattr(
            manager._status_bar,
            "showMessage",
            lambda *args: status_messages.append(args),
        )
        assert controller._start_workspace_save_worker(
            tmp_path / "finish.itws",
            typing.cast("workspace_saving._WorkspaceSaveSnapshot", Snapshot()),
            on_finished=lambda *_args: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        assert pool.worker is not None
        pool.worker.signals.waiting.emit()
        assert len(status_messages) == 1
        pool.worker.signals.finished.emit(0.1, None)
        assert errors == [
            (
                "Error while saving workspace",
                "An error occurred while saving the workspace file.",
            )
        ]
        assert not manager._workspace_state.save_in_progress


def test_background_workspace_save_finish_branches(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        controller = manager._workspace_controller
        workspace_path = tmp_path / "queued.itws"
        manager._workspace_state.path = workspace_path
        manager._workspace_state.dirty_state.add("uid")
        controller._background_save_requested = True

        queued_callbacks: list[Callable[[], None]] = []
        finished: list[bool] = []
        monkeypatch.setattr(
            controller,
            "_finish_workspace_save_result",
            lambda **_kwargs: True,
        )
        monkeypatch.setattr(
            controller,
            "_current_workspace_document_path",
            lambda: workspace_path,
        )
        monkeypatch.setattr(
            workspace_controller.QtCore.QTimer,
            "singleShot",
            staticmethod(lambda _delay, callback: queued_callbacks.append(callback)),
        )
        controller._finish_background_workspace_save(
            document_id=manager._workspace_state.document_id,
            workspace_path=workspace_path,
            snapshot=typing.cast(
                "workspace_saving._WorkspaceSaveSnapshot", types.SimpleNamespace()
            ),
            worker_elapsed=0.1,
            error=None,
            origin=None,
            snapshot_elapsed=0.0,
            started_at=time.perf_counter(),
            restore_focus=False,
            on_finished=finished.append,
        )
        assert len(queued_callbacks) == 1
        assert finished == [True]
        assert not controller._background_save_requested

        errors: list[tuple[str, str]] = []
        finished.clear()
        monkeypatch.setattr(
            controller,
            "_finish_workspace_save_result",
            lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        monkeypatch.setattr(
            manager,
            "_show_operation_error",
            lambda title, text: errors.append((title, text)),
        )
        controller._finish_background_workspace_save(
            document_id=manager._workspace_state.document_id,
            workspace_path=workspace_path,
            snapshot=typing.cast(
                "workspace_saving._WorkspaceSaveSnapshot", types.SimpleNamespace()
            ),
            worker_elapsed=0.1,
            error=None,
            origin=None,
            snapshot_elapsed=0.0,
            started_at=time.perf_counter(),
            restore_focus=False,
            on_finished=finished.append,
        )
        assert finished == [False]
        assert errors == [
            (
                "Error while saving workspace",
                "An error occurred while saving the workspace file.",
            )
        ]


def test_workspace_save_and_save_as_error_continuation_branches(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    def snapshot(generation: int = 0) -> workspace_saving._WorkspaceSaveSnapshot:
        return workspace_saving._WorkspaceSaveSnapshot(
            generation=generation,
            generation_plan=workspace_storage._WorkspaceGenerationPlan(
                manifest={"schema_version": 5, "nodes": []},
                objects=(),
            ),
            compression_mode="none",
            trusted_lineage=True,
        )

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(9, dtype=float).reshape(3, 3), dims=("x", "y"))
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid
        controller = manager._workspace_controller

        finished: list[bool] = []
        manager._workspace_state.save_in_progress = True
        assert not controller.save_as(native=False, on_finished=finished.append)
        assert finished == [False]
        manager._workspace_state.save_in_progress = False

        warnings: list[bool] = []
        monkeypatch.setattr(
            manager._workspace_controller,
            "_workspace_save_dialog",
            lambda **_kwargs: tmp_path / "bad.txt",
        )
        monkeypatch.setattr(
            workspace_controller,
            "_show_itws_workspace_warning",
            lambda _parent: warnings.append(True),
        )
        finished.clear()
        assert not controller.save_as(native=False, on_finished=finished.append)
        assert warnings == [True]
        assert finished == [False]

        errors: list[tuple[str, str]] = []
        monkeypatch.setattr(
            manager,
            "_show_operation_error",
            lambda title, text: errors.append((title, text)),
        )
        monkeypatch.setattr(
            manager._workspace_controller,
            "_workspace_save_dialog",
            lambda **_kwargs: tmp_path / "save.itws",
        )
        monkeypatch.setattr(
            controller.saving,
            "_workspace_generation_save_snapshot",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        finished.clear()
        assert not controller.save_as(native=False, on_finished=finished.append)
        assert finished == [False]
        assert errors[-1] == (
            "Error while saving workspace",
            "An error occurred while saving the workspace file.",
        )

        monkeypatch.setattr(
            controller.saving,
            "_workspace_generation_save_snapshot",
            lambda generation, **_kwargs: snapshot(generation),
        )
        worker_errors: list[workspace_saving._WorkspaceSaveError] = []
        monkeypatch.setattr(
            manager, "_show_workspace_save_worker_error", worker_errors.append
        )

        def _fail_worker(_fname, _snapshot, *, on_finished, on_start_error=None):
            del on_start_error
            on_finished(
                0.1,
                workspace_saving._WorkspaceSaveError("write failed"),
            )
            return True

        monkeypatch.setattr(controller, "_start_workspace_save_worker", _fail_worker)
        finished.clear()
        assert controller.save_as(native=False, on_finished=finished.append)
        assert [error.traceback_text for error in worker_errors] == ["write failed"]
        assert finished == [False]

        manager._workspace_controller._mark_workspace_clean()

        def _dirty_during_worker(
            _fname, _snapshot, *, on_finished, on_start_error=None
        ):
            del on_start_error
            manager._workspace_controller._mark_node_state_dirty(uid)
            on_finished(0.1, None)
            return True

        monkeypatch.setattr(
            controller, "_start_workspace_save_worker", _dirty_during_worker
        )
        finished.clear()
        assert controller.save_as(native=False, on_finished=finished.append)
        assert finished == [False]

        def _start_error_worker(_fname, _snapshot, *, on_finished, on_start_error=None):
            del on_finished
            if on_start_error is not None:
                on_start_error()
            return False

        monkeypatch.setattr(
            controller, "_start_workspace_save_worker", _start_error_worker
        )
        finished.clear()
        assert not controller.save_as(native=False, on_finished=finished.append)
        assert finished == [False]

        manager._workspace_state.path = tmp_path / "current.itws"
        monkeypatch.setattr(
            controller.saving, "_workspace_save_snapshot", lambda _path: snapshot()
        )
        finished.clear()
        assert not controller.save(on_finished=finished.append, restore_focus=False)
        assert finished == [False]


def test_workspace_save_completion_ignores_inactive_document(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        controller = manager._workspace_controller
        workspace_path = tmp_path / "stale-save.itws"
        manager._workspace_state.path = workspace_path.resolve()
        manager._workspace_state.mark_layout_dirty()

        snapshot = workspace_saving._WorkspaceSaveSnapshot(
            generation=manager._workspace_state.dirty_generation,
            generation_plan=workspace_storage._WorkspaceGenerationPlan(
                manifest={"schema_version": 5, "nodes": []},
                objects=(),
            ),
            compression_mode="none",
            trusted_lineage=True,
        )
        monkeypatch.setattr(
            controller.saving, "_workspace_save_snapshot", lambda _path: snapshot
        )
        recorded_recent: list[pathlib.Path] = []
        monkeypatch.setattr(
            controller, "_record_recent_workspace", recorded_recent.append
        )

        def _finish_after_document_change(
            _fname, _snapshot, *, on_finished, on_start_error=None
        ) -> bool:
            del on_start_error
            manager._workspace_state.advance_document_identity()
            manager._workspace_state.path = tmp_path / "replacement.itws"
            on_finished(0.1, None)
            return True

        monkeypatch.setattr(
            controller, "_start_workspace_save_worker", _finish_after_document_change
        )

        finished: list[bool] = []
        assert controller.save(on_finished=finished.append, restore_focus=False)

        assert finished == [False]
        assert manager.is_workspace_modified
        assert recorded_recent == []


def test_workspace_save_as_completion_ignores_inactive_document(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        controller = manager._workspace_controller
        target = tmp_path / "stale-save-as.itws"
        data = xr.DataArray(np.arange(9, dtype=float).reshape(3, 3), dims=("x", "y"))
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        manager._workspace_state.mark_layout_dirty()

        monkeypatch.setattr(
            manager._workspace_controller,
            "_workspace_save_dialog",
            lambda **_kwargs: target,
        )
        monkeypatch.setattr(
            controller.saving,
            "_workspace_generation_save_snapshot",
            lambda generation, **_kwargs: workspace_saving._WorkspaceSaveSnapshot(
                generation=generation,
                generation_plan=workspace_storage._WorkspaceGenerationPlan(
                    manifest={"schema_version": 5, "nodes": []},
                    objects=(),
                ),
                compression_mode="none",
                trusted_lineage=True,
            ),
        )

        def _finish_after_document_change(
            _fname, _snapshot, *, on_finished, on_start_error=None
        ) -> bool:
            del on_start_error
            manager._workspace_state.advance_document_identity()
            on_finished(0.1, None)
            return True

        monkeypatch.setattr(
            controller, "_start_workspace_save_worker", _finish_after_document_change
        )

        finished: list[bool] = []
        assert controller.save_as(native=False, on_finished=finished.append)

        assert finished == [False]
        assert manager.workspace_path is None
        assert manager.is_workspace_modified


def test_manager_associate_loaded_legacy_workspace_uses_converted_store(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        converted = tmp_path / "converted.itws"
        converted_store = workspace_store.WorkspaceStore(converted, create=True)
        converted_store.close()
        manager._workspace_state.path = converted.resolve()

        monkeypatch.setattr(
            workspace_format,
            "_workspace_schema_requires_conversion",
            lambda _schema_version: True,
        )
        monkeypatch.setattr(
            manager._workspace_controller,
            "_save_legacy_workspace_as_current",
            lambda *_, **__: (str(converted), None),
        )

        manager._workspace_controller._associate_loaded_workspace_file(
            tmp_path / "legacy.itws",
            1,
            rebind_data=False,
        )

        assert manager._workspace_state.path == converted
        assert (
            manager._workspace_state.schema_version
            == workspace_format._current_workspace_schema_version()
        )


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
        root_uid = manager._tool_graph.root_wrappers[0].uid
        tool = DerivativeTool(data)
        tool_uid = add_source_childtool(manager, tool, 0, show=False)

        root.setWindowTitle("stale root title[*]")
        manager._tool_graph.root_wrappers[0].update_title()
        assert "stale root title" not in manager._tool_graph.root_wrappers[0].label_text
        assert "[*]" not in manager._tool_graph.root_wrappers[0].label_text

        root.setWindowTitle("stale root title[*]")
        tool.setWindowTitle("stale tool title[*]")
        manager._workspace_controller._set_node_window_modified(root_uid, True)
        manager._workspace_controller._set_node_window_modified(tool_uid, True)

        expect_title_placeholder = sys.platform != "darwin"
        assert ("[*]" in root.windowTitle()) is expect_title_placeholder
        assert ("[*]" in tool.windowTitle()) is expect_title_placeholder
        assert (
            root.windowTitle()
            == manager_widgets._window_title_with_modified_placeholder(
                manager._tool_graph.root_wrappers[0].label_text
            )
        )
        assert (
            tool.windowTitle()
            == manager_widgets._window_title_with_modified_placeholder(
                f"{tool.tool_name}: {tool._tool_display_name}"
            )
        )

        fname = tmp_path / "titles.itws"
        manager._workspace_controller.saving._save_workspace_document(fname)

        root_title = _current_workspace_payload_attrs(fname)["itool_title"]
        tool_title = _current_workspace_payload_attrs(
            fname, f"0/childtools/{tool_uid}"
        )["tool_title"]

        assert "[*]" not in root_title
        assert "[*]" not in tool_title
        assert root_title == "source"


def test_manager_workspace_full_save_drops_empty_attr_name(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25).reshape((5, 5)),
            dims=["x", "y"],
            attrs={"": "dropped", "note": ""},
            name="data",
        )
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "empty-attr-name.itws"
        manager._workspace_controller.saving._save_workspace_document(fname)

        assert "" in root.slicer_area._data.attrs
        with h5py.File(fname, "r") as h5_file:
            saved_attrs = h5_file[_current_workspace_payload_path(fname)][
                _ITOOL_DATA_NAME
            ].attrs
            assert "" not in list(saved_attrs)
            assert saved_attrs["note"] == ""


def test_manager_workspace_full_save_roundtrips_non_native_data_attrs(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:

    rich_attr = _rich_workspace_attr_value()
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25).reshape((5, 5)),
            dims=["x", "y"],
            coords={
                "x": xr.DataArray(
                    np.arange(5),
                    dims=("x",),
                    attrs={"axis_config": rich_attr},
                ),
                "y": np.arange(5),
            },
            attrs={"Single Motor Scan": rich_attr},
            name="data",
        )
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        live_rich_attr = root.slicer_area._data.attrs["Single Motor Scan"]
        live_axis_attr = root.slicer_area._data.coords["x"].attrs["axis_config"]

        fname = tmp_path / "rich-data-attrs.itws"
        manager._workspace_controller.saving._save_workspace_document(fname)

        assert root.slicer_area._data.attrs["Single Motor Scan"] is live_rich_attr
        assert root.slicer_area._data.coords["x"].attrs["axis_config"] is live_axis_attr
        assert (
            workspace_format._WORKSPACE_ENCODED_ATTRS_ATTR
            not in root.slicer_area._data.attrs
        )
        with h5py.File(fname, "r") as h5_file:
            saved_data = h5_file[_current_workspace_payload_path(fname)][
                _ITOOL_DATA_NAME
            ]
            assert "Single Motor Scan" not in saved_data.attrs
            assert workspace_format._WORKSPACE_ENCODED_ATTRS_ATTR in saved_data.attrs

        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)
        assert manager._workspace_controller.loading._load_workspace_file(
            fname,
            replace=True,
            associate=False,
            mark_dirty=False,
            select=False,
        )
        loaded = manager.get_imagetool(0).slicer_area._data
        _assert_rich_workspace_attr(loaded.attrs["Single Motor Scan"])
        _assert_rich_workspace_attr(loaded.coords["x"].attrs["axis_config"])

        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)
        opened = workspace_arrays.open_workspace_dataset(
            fname,
            _current_workspace_payload_path(fname),
            chunks=None,
        )
        try:
            restored = workspace_format._restore_workspace_dataset_attrs(opened)
            loaded = restored[_ITOOL_DATA_NAME]
        finally:
            opened.close()
        _assert_rich_workspace_attr(loaded.attrs["Single Motor Scan"])
        _assert_rich_workspace_attr(loaded.coords["x"].attrs["axis_config"])


def test_manager_workspace_save_preserves_reordered_roots(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:

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
        manager._workspace_controller.saving._save_workspace_document(fname)
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()

        model = manager.tree_view._model
        assert model.dropMimeData(
            model.mimeData([model.index(0, 0)]),
            QtCore.Qt.DropAction.MoveAction,
            model.rowCount(),
            0,
            QtCore.QModelIndex(),
        )
        assert manager._tool_graph.displayed_indices == [1, 2, 0]
        assert manager.is_workspace_modified
        assert _request_workspace_save_and_wait(qtbot, manager)

        manifest = _current_workspace_manifest(fname)
        assert manifest["root_order"] == [1, 2, 0]

        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)
        loaded_order = [
            int(manager.get_imagetool(index).slicer_area._data.values[0, 0])
            for index in manager._tool_graph.displayed_indices
        ]
        assert loaded_order == [1, 2, 0]


def test_manager_workspace_child_save_shortcuts_use_background_save(
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

        monkeypatch.setattr(manager._workspace_controller, "save", _fake_save)
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
        add_source_childtool(manager, tool, 0, show=False)
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
        manager._workspace_controller.saving._save_workspace_document(fname)
        adopt_workspace_path(manager, fname)

        dataset_writes: list[str | None] = []
        original_to_netcdf = xr.Dataset.to_netcdf

        def _to_netcdf_spy(self, *args, **kwargs):
            dataset_writes.append(kwargs.get("group"))
            return original_to_netcdf(self, *args, **kwargs)

        monkeypatch.setattr(xr.Dataset, "to_netcdf", _to_netcdf_spy)

        manager.rename_imagetool(0, "state only")
        assert _request_workspace_save_and_wait(qtbot, manager)
        assert dataset_writes == []

        replacement = data.copy(deep=True)
        replacement.data = np.asarray(replacement.data) + 10
        root.slicer_area.replace_source_data(replacement)
        assert _request_workspace_save_and_wait(qtbot, manager)

        import h5py

        with h5py.File(fname, "r") as h5_file:
            saved = h5_file[_current_workspace_payload_path(fname)][_ITOOL_DATA_NAME]
            assert saved[0, 0] == 10


def test_manager_workspace_full_save_keeps_full_persistence_for_serialized_nodes(
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
        fname = tmp_path / "full-persistence.itws"
        manager._workspace_controller.saving._save_workspace_document(fname)

        replacement = data.copy(deep=True)
        replacement.data = np.asarray(replacement.data) + 1
        root.slicer_area.replace_source_data(replacement)

        original = imagetool_viewer.ImageSlicerArea.persistence_data_and_state
        calls = 0

        def _persistence_data_and_state_spy(self):
            nonlocal calls
            calls += 1
            return original(self)

        monkeypatch.setattr(
            imagetool_viewer.ImageSlicerArea,
            "persistence_data_and_state",
            _persistence_data_and_state_spy,
        )

        manager._workspace_controller.saving._save_workspace_document(fname)

        assert calls >= 1


def test_manager_workspace_full_save_preserves_in_memory_backing_after_rebind(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25, dtype=np.float64).reshape((5, 5)),
            dims=["x", "y"],
            coords={"x": np.arange(5), "y": np.arange(5)},
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "memory.itws"
        backing_snapshot = (
            manager._workspace_controller.loading._workspace_data_backing_snapshot()
        )
        manager._workspace_controller.saving._save_workspace_document(fname)
        manager._workspace_controller.loading._rebind_workspace_backed_imagetools(
            fname,
            backing_snapshot=backing_snapshot,
            old_workspace_path=None,
        )

        saved_data = manager.get_imagetool(0).slicer_area._data
        assert workspace_arrays.dataarray_is_numpy_backed(saved_data)


def test_manager_workspace_load_keeps_visible_saved_data_in_memory(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(512 * 512, dtype=np.float64).reshape((512, 512)),
            dims=["x", "y"],
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=True)

        fname = tmp_path / "load-visible-memory.itws"
        manager._workspace_controller.saving._save_workspace_document(fname)
        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        loaded = manager.get_imagetool(0).slicer_area
        assert not loaded.data_chunked
        assert not loaded.data_file_backed
        assert loaded.data_loadable is False
        assert workspace_arrays.dataarray_is_numpy_backed(loaded._data)
        assert loaded._data.values.flags.writeable
        np.testing.assert_array_equal(loaded._data.values, data.values)


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
        manager._workspace_controller.saving._save_workspace_document(fname)
        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        replacement = data.copy(deep=True)
        replacement.data = np.asarray(replacement.data) + 10
        manager.get_imagetool(0).slicer_area.replace_source_data(
            replacement, auto_compute=False
        )
        assert _request_workspace_save_and_wait(qtbot, manager)
        assert list(tmp_path.glob("lazy-data.itws.delta-*")) == []

        import h5py

        with h5py.File(fname, "r") as h5_file:
            saved = h5_file[_current_workspace_payload_path(fname)][_ITOOL_DATA_NAME]
            assert saved[0, 0] == 10


def test_manager_workspace_same_file_lazy_data_delta_save_does_not_deadlock(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(512 * 512, dtype=np.float64).reshape((512, 512)),
            dims=["x", "y"],
            coords={"x": np.arange(512), "y": np.arange(512)},
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "same-file-lazy.itws"
        manager._workspace_controller.saving._save_workspace_document(fname)
        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        manager._workspace_controller.loading._rebind_workspace_backed_imagetools(
            fname, targets=[0], chunks={}
        )
        assert manager.get_imagetool(0).slicer_area.data_chunked
        manager.get_imagetool(0).slicer_area._set_chunks({"x": 128, "y": 64})

        uid = manager._tool_graph.root_wrappers[0].uid
        manager._workspace_controller._mark_node_data_dirty(uid)
        assert _request_workspace_save_and_wait(qtbot, manager)

        import h5py

        with h5py.File(fname, "r") as h5_file:
            saved = h5_file[_current_workspace_payload_path(fname)][_ITOOL_DATA_NAME]
            assert saved[0, 0] == 0
            assert saved.chunks == (128, 64)


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
        manager._workspace_controller.saving._save_workspace_document(fname)
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()
        manifest_before = _current_workspace_manifest(fname)

        replacement = data.copy(deep=True)
        replacement.data = np.asarray(replacement.data) + 10
        root.slicer_area.replace_source_data(replacement, auto_compute=False)

        def _write_partial_object_then_raise(
            h5_file,
            group_path: str,
            _dataset: xr.Dataset,
            **_kwargs,
        ) -> None:
            h5_file.require_group(group_path)
            raise RuntimeError("object write failed")

        monkeypatch.setattr(
            workspace_arrays,
            "_write_workspace_dataset_group_to_file",
            _write_partial_object_then_raise,
        )
        monkeypatch.setattr(
            manager, "_show_workspace_save_worker_error", lambda *args: None
        )

        assert not _request_workspace_save_and_wait(qtbot, manager)
        assert _current_workspace_manifest(fname) == manifest_before
        with h5py.File(fname, "r") as h5_file:
            saved = h5_file[_current_workspace_payload_path(fname)][_ITOOL_DATA_NAME]
            assert saved[0, 0] == 0
            assert set(h5_file[workspace_store._WORKSPACE_OBJECTS_GROUP]) == set(
                workspace_store.WorkspaceStore.manifest_object_ids(manifest_before)
            )


def test_manager_workspace_stale_pending_groups_do_not_poison_open_or_save(
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

        fname = tmp_path / "stale-pending.itws"
        manager._workspace_controller.saving._save_workspace_document(fname)
        with h5py.File(fname, "a") as h5_file:
            h5_file.create_group(
                f"{workspace_format._WORKSPACE_PENDING_GROUP_PREFIX}stale"
            )
            h5_file.create_group(
                f"{workspace_format._WORKSPACE_BACKUP_GROUP_PREFIX}stale"
            )

        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        manager.rename_imagetool(0, "cleaned")
        assert _request_workspace_save_and_wait(qtbot, manager)
        _assert_no_workspace_internal_groups(fname)


def test_manager_workspace_delta_save_persists_geometry_changes(
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

        fname = tmp_path / "geometry.itws"
        manager._workspace_controller.saving._save_workspace_document(fname)
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()

        root.setGeometry(12, 34, 321, 234)
        qtbot.wait_until(lambda: manager.is_workspace_modified, timeout=5000)
        expected_rect = tuple(root.geometry().getRect())

        assert _request_workspace_save_and_wait(qtbot, manager)
        entry = next(
            entry
            for entry in workspace_format._iter_workspace_manifest_node_entries(
                _current_workspace_manifest(fname)
            )
            if entry.get("path") == "0"
        )
        saved_attrs = workspace_format._restore_workspace_manifest_attrs(
            entry["payload_attrs"]
        )
        saved_state = json.loads(saved_attrs["itool_window_state"])
        saved_rect = tuple(int(value) for value in saved_state["rect"])
        assert saved_rect == expected_rect
