from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys
import threading
import types
import typing

import numpy as np
import pytest
import xarray as xr
from qtpy import QtCore, QtWidgets

import erlab
import erlab.extensions._entry_points as extension_entry_points
import erlab.interactive.imagetool.manager._base as manager_base
import erlab.interactive.imagetool.manager._extensions._catalog as extension_catalog
import erlab.interactive.imagetool.manager._extensions._dialogs as extension_dialogs
from erlab.interactive.imagetool._load_source import _resolve_load_func
from erlab.interactive.imagetool._provenance._execution import (
    file_load_source_status,
    replay_file_provenance,
    replay_script_provenance,
)
from erlab.interactive.imagetool._provenance._model import (
    FileDataSelection,
    FileLoadSource,
    FileReplayCall,
    ScriptInput,
    ToolProvenanceSpec,
    file_load,
    full_data,
)
from erlab.interactive.imagetool._provenance._operations import (
    ExtensionRoutineOperation,
)
from erlab.interactive.imagetool.manager._extensions import (
    _controller as extension_controller,
)
from erlab.interactive.imagetool.manager._extensions._catalog import (
    _ExtensionCatalog,
    _ExtensionCatalogConflictError,
    _ExtensionCatalogStore,
)
from erlab.interactive.imagetool.manager._extensions._dialogs import (
    _ExtensionParameterDialog,
)
from erlab.interactive.imagetool.manager._extensions._execution import (
    _detached_routine_output,
    _ExtensionLoaderCall,
    _ExtensionLoaderWorker,
    _ExtensionRoutineWaiter,
    _ExtensionValidationWorker,
    _readonly_array,
    _validate_extension_revision,
)
from erlab.interactive.imagetool.manager._extensions._models import (
    _ExtensionCatalogModel,
    _ExtensionMetadata,
    _ExtensionRecord,
    _ExtensionRevision,
    _ResolvedWorkspaceRequirement,
    _WorkspaceExtensionRequirement,
)
from erlab.interactive.imagetool.manager._provenance_edit._files import (
    _FileLoadEditDialog,
)
from erlab.interactive.imagetool.manager._workspace import _arrays as workspace_arrays
from erlab.interactive.imagetool.manager._workspace import _format as workspace_format
from erlab.interactive.imagetool.manager._workspace import _saving as workspace_saving
from erlab.interactive.imagetool.manager._workspace import _storage as workspace_storage
from erlab.interactive.imagetool.manager._workspace import _store as workspace_store


def _script(path: pathlib.Path, expression: str = "data * scale") -> bytes:
    source = f"""import xarray as xr
from erlab.extensions import routine

@routine(name="Scale", category="Lab")
def scale(data: xr.DataArray, scale: float = 2.0) -> xr.DataArray:
    return {expression}
""".encode()
    path.write_bytes(source)
    return source


def _validate_and_enable(
    store: _ExtensionCatalogStore,
    extension_id: str,
    *,
    expected_record_generation: int,
) -> _ExtensionCatalogModel:
    record = store.read().extensions[extension_id]
    return _validate_extension_revision(
        store,
        extension_id,
        revision_hash=record.current_revision,
        expected_record_generation=expected_record_generation,
        manager_session_id="test-manager",
        script_modules={},
    )


def test_parameter_dialog_preserves_none_and_empty_string(qtbot) -> None:
    descriptor = erlab.extensions.RoutineDescriptor(
        id="optional-values",
        name="Optional values",
        category="Lab",
        summary="",
        function_name="optional_values",
        parameters=(
            erlab.extensions.ParameterDescriptor(
                id="scale",
                kind=erlab.extensions.ParameterKind.NUMBER,
                required=False,
                optional=True,
                default=2.0,
            ),
            erlab.extensions.ParameterDescriptor(
                id="label",
                kind=erlab.extensions.ParameterKind.STRING,
                required=False,
                optional=True,
            ),
        ),
    )
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    dialog = _ExtensionParameterDialog(descriptor, parent)
    qtbot.addWidget(dialog)

    assert dialog.parameters == {"scale": 2.0, "label": None}

    scale_editor = typing.cast(
        "QtWidgets.QLineEdit",
        dialog.findChild(QtWidgets.QLineEdit, "manager_extension_parameter_scale"),
    )
    scale_editor.clear()
    assert dialog.parameters["scale"] is None

    label_none = dialog.findChild(
        QtWidgets.QCheckBox, "manager_extension_parameter_label_none"
    )
    label_none.setChecked(False)
    assert dialog.parameters["label"] == ""


def test_parameter_dialog_uses_initial_values(qtbot) -> None:
    descriptor = erlab.extensions.LoaderDescriptor(
        id="configured-loader",
        name="Configured loader",
        category="Lab",
        summary="",
        function_name="load_data",
        parameters=(
            erlab.extensions.ParameterDescriptor(
                id="scale",
                kind=erlab.extensions.ParameterKind.NUMBER,
                required=False,
                optional=True,
                default=2.0,
            ),
            erlab.extensions.ParameterDescriptor(
                id="label",
                kind=erlab.extensions.ParameterKind.STRING,
                required=False,
                optional=True,
            ),
        ),
    )
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    dialog = _ExtensionParameterDialog(
        descriptor, parent, values={"scale": 3.5, "label": ""}
    )
    qtbot.addWidget(dialog)

    assert dialog.parameters == {"scale": 3.5, "label": ""}


def test_manager_extension_loader_dialog_uses_recent_values(monkeypatch) -> None:
    descriptor = erlab.extensions.LoaderDescriptor(
        id="configured-loader",
        name="Configured loader",
        category="Lab",
        summary="",
        function_name="load_data",
        parameters=(
            erlab.extensions.ParameterDescriptor(
                id="mode",
                kind=erlab.extensions.ParameterKind.STRING,
                required=False,
                default="default",
            ),
        ),
    )

    class _ConfiguredLoader(erlab.io.dataloader.LoaderBase):
        name = "_configured_extension_test"
        description = "Test extension loader."
        extensions: typing.ClassVar[set[str]] = {".dat"}

        def load(self, _path: pathlib.Path, *, mode: str = "default") -> xr.DataArray:
            del mode
            return xr.DataArray(0)

    _ConfiguredLoader.load.descriptor = descriptor  # type: ignore[attr-defined]
    _ConfiguredLoader.load.uses_standard_loader_options = False  # type: ignore[attr-defined]
    load_data = _ConfiguredLoader().load
    shared_updates: list[tuple[str, dict[str, str], dict[str, typing.Any]]] = []

    class _AcceptParameterDialog:
        def __init__(self, dialog_descriptor, parent, values) -> None:
            assert dialog_descriptor is descriptor
            assert parent is manager
            assert values == {"mode": "recent"}

        def exec(self) -> bool:
            return True

        @property
        def parameters(self) -> dict[str, str]:
            return {"mode": "recent"}

    monkeypatch.setattr(
        extension_dialogs, "_ExtensionParameterDialog", _AcceptParameterDialog
    )
    manager = types.SimpleNamespace(
        _recent_loader_kwargs_by_filter={"Configured (*.dat)": {"mode": "recent"}},
        _recent_loader_extensions_by_filter={},
        _recent_name_filter=None,
        _shared_loader_state=lambda: ({}, {}),
        _set_shared_loader_options=lambda name, kwargs, extensions: (
            shared_updates.append((name, dict(kwargs), dict(extensions)))
        ),
        _mark_workspace_layout_dirty=lambda: None,
    )

    selected = manager_base._ImageToolManagerBase._select_loader_options(
        manager,
        {"Configured (*.dat)": (load_data, {"mode": "default"})},
    )

    assert selected == ("Configured (*.dat)", load_data, {"mode": "recent"})
    assert shared_updates == [("_configured_extension_test", {"mode": "recent"}, {})]


def test_catalog_reload_identity_metadata_and_conflict(tmp_path: pathlib.Path) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    script_path = tmp_path / "scale.py"
    source = _script(script_path)

    catalog, revision, created = store.add_script(script_path)
    assert created
    assert revision == hashlib.sha256(source).hexdigest()
    first = catalog.extensions["scale"]

    catalog, unchanged_revision, created = store.add_script(script_path)
    assert not created
    assert unchanged_revision == revision
    assert len(catalog.extensions["scale"].revisions) == 1

    metadata = _ExtensionMetadata(author="A. User", change_summary="Reviewed")
    catalog = store.update_record(
        "scale",
        expected_record_generation=first.record_generation,
        metadata=metadata,
    )
    assert catalog.extensions["scale"].metadata == metadata
    assert catalog.extensions["scale"].current_revision == revision

    with pytest.raises(_ExtensionCatalogConflictError, match="another manager"):
        store.update_record(
            "scale",
            expected_record_generation=first.record_generation,
            favorite=True,
        )


@pytest.mark.parametrize(
    "corruption", ["extension-key", "revision-key", "current", "source-type"]
)
def test_catalog_rejects_inconsistent_persisted_identity(
    tmp_path: pathlib.Path,
    corruption: str,
) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    script_path = tmp_path / "scale.py"
    _script(script_path)
    catalog, revision, _created = store.add_script(script_path)
    payload = catalog.model_dump(mode="json")
    record = payload["extensions"]["scale"]
    if corruption == "extension-key":
        record["id"] = "different"
    elif corruption == "revision-key":
        record["revisions"][revision]["source_hash"] = "0" * 64
    elif corruption == "source-type":
        record["revisions"][revision]["entry_point_group"] = "erlab.extensions"
    else:
        record["current_revision"] = "0" * 64
    store.path.write_text(json.dumps(payload))

    with pytest.raises(extension_catalog._ExtensionCatalogError):
        store.read()


def test_catalog_rejects_script_and_environment_source_type_collisions(
    tmp_path: pathlib.Path,
) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    extension_id = "shared-id"
    environment_revision = "a" * 64
    environment_record = _ExtensionRecord(
        id=extension_id,
        name="Shared",
        source_type="environment-package",
        current_revision=environment_revision,
        revisions={
            environment_revision: _ExtensionRevision(
                source_hash=environment_revision,
                object_name="lab_package:extension",
                created_at="2026-01-01T00:00:00+00:00",
                entry_point_group="erlab.extensions",
                entry_point_name="extension",
                entry_point_value="lab_package:extension",
            )
        },
    )
    catalog = store.mutate(
        None,
        lambda current: current.model_copy(
            update={"extensions": {extension_id: environment_record}}
        ),
    )
    script_path = tmp_path / "shared-id.py"
    source = _script(script_path)

    with pytest.raises(_ExtensionCatalogConflictError, match="environment package"):
        store.add_script(script_path)
    with pytest.raises(_ExtensionCatalogConflictError, match="environment package"):
        store.add_embedded_script(
            source,
            extension_id=extension_id,
            expected_revision=hashlib.sha256(source).hexdigest(),
            name="Shared",
            metadata=_ExtensionMetadata(),
        )

    assert store.read() == catalog


def test_catalog_validates_callback_output_before_commit(
    tmp_path: pathlib.Path,
) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    script_path = tmp_path / "scale.py"
    _script(script_path)
    catalog, _revision, _created = store.add_script(script_path)

    def corrupt(model):
        records = dict(model.extensions)
        records["scale"] = records["scale"].model_copy(
            update={"current_revision": "0" * 64}
        )
        return model.model_copy(update={"extensions": records})

    with pytest.raises(ValueError, match="current extension revision is missing"):
        store.mutate(
            "scale",
            corrupt,
            expected_record_generation=catalog.extensions["scale"].record_generation,
        )

    assert store.read() == catalog


def test_catalog_changed_reload_requires_approval(tmp_path: pathlib.Path) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    script_path = tmp_path / "scale.py"
    _script(script_path)
    catalog, old_revision, _created = store.add_script(script_path)
    record = catalog.extensions["scale"]
    catalog = _validate_and_enable(
        store, "scale", expected_record_generation=record.record_generation
    )
    assert catalog.extensions["scale"].enabled

    _script(script_path, "data + scale")
    catalog, new_revision, created = store.add_script(script_path)
    assert created
    assert new_revision != old_revision
    assert not catalog.extensions["scale"].enabled
    assert not catalog.extensions["scale"].revisions[new_revision].approved

    _script(script_path)
    catalog, restored_revision, changed = store.add_script(script_path)
    assert changed
    assert restored_revision == old_revision
    assert catalog.extensions["scale"].current_revision == old_revision
    assert len(catalog.extensions["scale"].revisions) == 2
    assert not catalog.extensions["scale"].enabled

    catalog = _validate_and_enable(
        store,
        "scale",
        expected_record_generation=catalog.extensions["scale"].record_generation,
    )
    catalog, unchanged_revision, changed = store.add_script(script_path)
    assert not changed
    assert unchanged_revision == old_revision
    assert catalog.extensions["scale"].enabled


def test_stale_validation_does_not_import_a_newer_revision(
    tmp_path: pathlib.Path,
) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    script_path = tmp_path / "scale.py"
    marker_path = tmp_path / "imported-new-revision"
    _script(script_path)
    catalog, reviewed_revision, _created = store.add_script(script_path)
    reviewed_generation = catalog.extensions["scale"].record_generation
    script_path.write_text(
        f"""import pathlib
import xarray as xr
from erlab.extensions import routine

pathlib.Path({str(marker_path)!r}).write_text("imported")

@routine(name="Scale")
def scale(data: xr.DataArray) -> xr.DataArray:
    return data + 1.0
"""
    )
    catalog, newer_revision, _created = store.add_script(script_path)

    with pytest.raises(_ExtensionCatalogConflictError, match="before validation"):
        _validate_extension_revision(
            store,
            "scale",
            revision_hash=reviewed_revision,
            expected_record_generation=reviewed_generation,
            manager_session_id="manager",
            script_modules={},
        )

    assert not marker_path.exists()
    current = catalog.extensions["scale"]
    assert current.current_revision == newer_revision
    assert not current.revisions[newer_revision].approved


def test_source_changed_during_review_is_not_added(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "reviewed.py"
    _script(script_path)
    shown: list[None] = []

    def change_source(_dialog) -> int:
        _script(script_path, "data + scale")
        return 1

    monkeypatch.setattr(
        extension_controller._SourceReviewDialog,
        "exec",
        change_source,
    )
    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        lambda *_args, **_kwargs: shown.append(None),
    )

    with manager_context() as manager:
        assert not manager._extensions._review_and_add(script_path)
        assert "reviewed" not in manager._extensions.catalog.store.read().extensions

    assert shown == [None]


def test_catalog_reload_rejects_a_stale_same_extension_edit(
    tmp_path: pathlib.Path,
) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    script_path = tmp_path / "scale.py"
    _script(script_path)
    catalog, _revision, _created = store.add_script(script_path)
    stale_generation = catalog.extensions["scale"].record_generation
    store.update_record(
        "scale",
        expected_record_generation=stale_generation,
        favorite=True,
    )
    _script(script_path, "data + scale")

    with pytest.raises(_ExtensionCatalogConflictError, match="another manager"):
        store.add_script(
            script_path,
            extension_id="scale",
            expected_record_generation=stale_generation,
            check_record_generation=True,
        )


def test_unchanged_reload_repairs_corrupt_stored_source(
    tmp_path: pathlib.Path,
) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    script_path = tmp_path / "scale.py"
    source = _script(script_path)
    catalog, revision, _created = store.add_script(script_path)
    object_path = store.source_path("scale", revision)
    object_path.write_bytes(b"corrupt")

    reloaded, unchanged_revision, created = store.add_script(
        script_path,
        expected_record_generation=catalog.extensions["scale"].record_generation,
    )

    assert not created
    assert unchanged_revision == revision
    assert len(reloaded.extensions["scale"].revisions) == 1
    assert object_path.read_bytes() == source


def test_unchanged_reload_updates_script_source_location(
    tmp_path: pathlib.Path,
) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    original_path = tmp_path / "original" / "scale.py"
    original_path.parent.mkdir()
    source = _script(original_path)
    catalog, revision, _created = store.add_script(original_path)
    initial_generation = catalog.extensions["scale"].record_generation
    relocated_path = tmp_path / "relocated" / "scale.py"
    relocated_path.parent.mkdir()
    relocated_path.write_bytes(source)

    reloaded, unchanged_revision, created = store.add_script(relocated_path)

    record = reloaded.extensions["scale"]
    assert not created
    assert unchanged_revision == revision
    assert len(record.revisions) == 1
    assert record.revisions[revision].source_path == os.fspath(relocated_path.resolve())
    assert record.record_generation == initial_generation + 1


def test_embedded_source_review_updates_existing_metadata(
    tmp_path: pathlib.Path,
) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    script_path = tmp_path / "scale.py"
    source = _script(script_path)
    catalog, revision, _created = store.add_script(script_path)
    metadata = _ExtensionMetadata(author="A. User", changelog="Reviewed source")

    updated = store.add_embedded_script(
        source,
        extension_id="scale",
        expected_revision=revision,
        name="Scale",
        metadata=metadata,
        expected_record_generation=catalog.extensions["scale"].record_generation,
    )

    assert updated.extensions["scale"].metadata == metadata
    assert tuple(updated.extensions["scale"].revisions) == (revision,)


def test_catalog_preserves_import_failure_diagnostics(tmp_path: pathlib.Path) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    script_path = tmp_path / "broken.py"
    script_path.write_text("raise RuntimeError('broken import')\n")
    catalog, revision, _created = store.add_script(script_path)

    with pytest.raises(erlab.extensions.ExtensionImportError, match="broken import"):
        _validate_and_enable(
            store,
            "broken",
            expected_record_generation=catalog.extensions["broken"].record_generation,
        )

    record = store.read().extensions["broken"]
    assert not record.enabled
    assert "RuntimeError: broken import" in str(record.revisions[revision].import_error)


def test_exact_revision_public_replay_survives_another_catalog_closing(
    tmp_path: pathlib.Path,
) -> None:
    first = _ExtensionCatalog(directory=tmp_path / "first")
    second = _ExtensionCatalog(directory=tmp_path / "second")
    script_path = tmp_path / "scale.py"
    _script(script_path)
    try:
        catalog, revision, _created = first.store.add_script(script_path)
        _validate_and_enable(
            first.store,
            "scale",
            expected_record_generation=catalog.extensions["scale"].record_generation,
        )
        operation = ExtensionRoutineOperation(
            extension_id="scale",
            revision_hash=revision,
            routine_id="scale",
            extension_name="Scale",
            routine_name="Scale",
            source_type="script",
            function_name="scale",
            source_path=str(first.store.source_path("scale", revision)),
            entry_point_group=None,
            entry_point_name=None,
            parameters={"scale": 3.0},
        )
        data = xr.DataArray([1.0, 2.0])

        xr.testing.assert_identical(operation.apply(data), data * 3.0)
        generated = operation.expression_code("data")
        assert "run_routine" not in generated

        second.close()
        xr.testing.assert_identical(operation.apply(data), data * 3.0)
    finally:
        first.close()
        second.close()

    namespace: dict[str, typing.Any] = {"data": data, "erlab": erlab}
    exec(f"result = {generated}", namespace)  # noqa: S102
    xr.testing.assert_identical(namespace["result"], data * 3.0)


def test_catalog_watcher_recovers_after_atomic_replace(
    tmp_path: pathlib.Path, qtbot: pytest.QtBot
) -> None:
    directory = tmp_path / "catalog"
    first = _ExtensionCatalog(directory=directory)
    second = _ExtensionCatalog(directory=directory)
    script_path = tmp_path / "scale.py"
    _script(script_path)
    try:
        with qtbot.waitSignal(second.changed, timeout=3000):
            first.store.add_script(script_path)
        assert "scale" in second.model.extensions
        assert str(second.store.path) in second._watcher.files()
    finally:
        first.close()
        second.close()


def test_catalog_close_cancels_pending_refresh(
    tmp_path: pathlib.Path, qtbot: pytest.QtBot
) -> None:
    catalog = _ExtensionCatalog(directory=tmp_path / "catalog")
    changed = False

    def mark_changed() -> None:
        nonlocal changed
        changed = True

    catalog.changed.connect(mark_changed)
    catalog._schedule_refresh()
    assert catalog._refresh_timer.isActive()

    catalog.close()
    qtbot.wait(1)

    assert not catalog._refresh_timer.isActive()
    assert not changed
    catalog.close()


def test_catalog_watcher_propagates_global_record_state(
    tmp_path: pathlib.Path, qtbot: pytest.QtBot
) -> None:
    directory = tmp_path / "catalog"
    first = _ExtensionCatalog(directory=directory)
    second = _ExtensionCatalog(directory=directory)
    script_path = tmp_path / "scale.py"
    _script(script_path)
    try:
        with qtbot.waitSignal(second.changed, timeout=3000):
            catalog, _revision, _created = first.store.add_script(script_path)
        with qtbot.waitSignal(second.changed, timeout=3000):
            catalog = _validate_and_enable(
                first.store,
                "scale",
                expected_record_generation=(
                    catalog.extensions["scale"].record_generation
                ),
            )
        assert second.model.extensions["scale"].enabled

        with qtbot.waitSignal(second.changed, timeout=3000):
            catalog = first.store.update_record(
                "scale",
                expected_record_generation=(
                    catalog.extensions["scale"].record_generation
                ),
                favorite=True,
                metadata=_ExtensionMetadata(author="Lab User"),
            )
        propagated = second.model.extensions["scale"]
        assert propagated.favorite
        assert propagated.metadata.author == "Lab User"

        with qtbot.waitSignal(second.changed, timeout=3000):
            first.store.update_record(
                "scale",
                expected_record_generation=propagated.record_generation,
                removed=True,
            )
        assert second.model.extensions["scale"].removed
        assert not second.model.extensions["scale"].enabled
    finally:
        first.close()
        second.close()


def test_environment_loader_entry_point_is_inspected_before_import(
    example_loader,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    load_calls: list[None] = []

    class _EntryPoint:
        group = "erlab.io.loaders"
        name = "example"
        value = "lab_package:ExampleLoader"
        dist = None

        @staticmethod
        def load():
            load_calls.append(None)
            return example_loader

    class _EntryPoints(tuple):
        def select(self, **parameters):
            return tuple(
                entry
                for entry in self
                if all(
                    getattr(entry, key, None) == value
                    for key, value in parameters.items()
                )
            )

    entry_points = _EntryPoints((_EntryPoint(),))
    monkeypatch.setattr(
        extension_catalog.importlib.metadata,
        "entry_points",
        lambda: entry_points,
    )
    store = _ExtensionCatalogStore(tmp_path / "catalog")

    catalog = store.refresh_environment_packages()
    assert load_calls == []
    extension_id = "environment.erlab.io.loaders.example"
    record = catalog.extensions[extension_id]
    catalog = _validate_and_enable(
        store,
        extension_id,
        expected_record_generation=record.record_generation,
    )

    record = catalog.extensions[extension_id]
    assert len(load_calls) == 1
    assert record.enabled
    revision = record.revisions[record.current_revision]
    assert revision.loaders[0].id == "example"
    assert revision.loader_always_single is False
    assert tuple(
        (method.name_filter, method.method, method.defaults)
        for method in revision.loader_dialog_methods
    ) == (("Example Raw Data (*.h5)", None, {}),)
    resolved = store.resolve_capability(
        extension_id,
        record.current_revision,
        "loader",
        "example",
    )
    assert len(load_calls) == 2
    assert getattr(resolved, "__self__", None).name == "example"


def test_environment_refresh_does_not_replace_a_script_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    class _EntryPoints(tuple):
        def select(self, **parameters):
            return tuple(
                entry
                for entry in self
                if all(
                    getattr(entry, key, None) == value
                    for key, value in parameters.items()
                )
            )

    entry_point = types.SimpleNamespace(
        group="erlab.extensions",
        name="example",
        value="lab_package:extension",
        dist=None,
    )
    monkeypatch.setattr(
        extension_catalog.importlib.metadata,
        "entry_points",
        lambda: _EntryPoints((entry_point,)),
    )
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    script_path = tmp_path / "environment.erlab.extensions.example.py"
    _script(script_path)
    before, _revision, _created = store.add_script(script_path)

    after = store.refresh_environment_packages()

    assert after == before
    assert after.extensions[script_path.stem].source_type == "script"


def test_environment_loader_preserves_file_dialog_contract(
    manager_context,
    qtbot: pytest.QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    call_threads: list[int] = []
    stale_load_calls: list[None] = []

    class _Distribution:
        metadata: typing.ClassVar[dict[str, str]] = {"Name": "lab-package"}

        def __init__(self, version: str) -> None:
            self.version = version

        @staticmethod
        def read_text(_name: str) -> None:
            return None

    class _MultiLoader(erlab.io.dataloader.LoaderBase):
        name = "multi_extension"
        extensions: typing.ClassVar[set[str]] = {".txt"}
        always_single = False
        skip_validate = True

        @property
        def file_dialog_methods(self):
            return {
                "Normal Extension Data (*.txt)": (self.load, {"single": True}),
                "Scaled Extension Data (*.txt)": (
                    self.load_scaled,
                    {"scale": 3.0},
                ),
                "Multiple Extension Data (*.txt)": (self.load_multiple, {}),
            }

        def load_single(self, file_path, without_values=False):
            del without_values
            return xr.DataArray([float(pathlib.Path(file_path).read_text())])

        def load_scaled(self, file_path, scale=1.0):
            call_threads.append(threading.get_ident())
            return xr.DataArray([float(pathlib.Path(file_path).read_text()) * scale])

        def load_multiple(self, file_path):
            value = float(pathlib.Path(file_path).read_text())
            return [xr.DataArray([value]), xr.DataArray([value + 1.0])]

    class _EntryPoint:
        group = "erlab.io.loaders"
        name = "multi_extension"
        value = "lab_package:MultiLoader"
        dist = _Distribution("2")

        @staticmethod
        def load():
            return _MultiLoader

    class _StaleEntryPoint(_EntryPoint):
        dist = _Distribution("1")

        @staticmethod
        def load():
            stale_load_calls.append(None)
            return _MultiLoader

    class _EntryPoints(tuple):
        def select(self, **parameters):
            return tuple(
                entry
                for entry in self
                if all(
                    getattr(entry, key, None) == value
                    for key, value in parameters.items()
                )
            )

    entry_points = _EntryPoints((_StaleEntryPoint(), _EntryPoint()))
    monkeypatch.setattr(
        extension_catalog.importlib.metadata,
        "entry_points",
        lambda: entry_points,
    )
    value_path = tmp_path / "value.txt"
    value_path.write_text("4")

    with manager_context() as manager:
        catalog = manager._extensions.catalog.store.refresh_environment_packages()
        extension_id = "environment.erlab.io.loaders.multi_extension"
        record = catalog.extensions[extension_id]
        _validate_and_enable(
            manager._extensions.catalog.store,
            extension_id,
            expected_record_generation=record.record_generation,
        )
        manager._extensions.catalog.refresh()

        loaders = manager._extensions.file_loaders(value_path)
        normal_func, normal_defaults = loaders["Normal Extension Data (*.txt)"]
        scaled_func, scaled_defaults = loaders["Scaled Extension Data (*.txt)"]
        multiple_func, multiple_defaults = loaders["Multiple Extension Data (*.txt)"]
        assert normal_defaults == {"single": True}
        assert scaled_defaults == {"scale": 3.0}
        assert multiple_defaults == {}
        assert normal_func.__self__.always_single is False
        assert scaled_func.__self__.always_single is False
        normal_result = normal_func(value_path, **normal_defaults)
        xr.testing.assert_equal(normal_result, xr.DataArray([4.0]))
        assert normal_result.attrs["data_loader_name"] == "multi_extension"
        xr.testing.assert_identical(
            scaled_func(value_path, **scaled_defaults), xr.DataArray([12.0])
        )
        multiple_result = multiple_func(value_path, **multiple_defaults)
        assert isinstance(multiple_result, list)
        xr.testing.assert_identical(multiple_result[0], xr.DataArray([4.0]))
        xr.testing.assert_identical(multiple_result[1], xr.DataArray([5.0]))
        assert manager._extensions.explorer_loaders[
            "multi_extension"
        ].always_single is (False)

        resolved = _resolve_load_func(
            (
                scaled_func,
                scaled_defaults,
                FileDataSelection(kind="dataarray"),
            )
        )
        assert resolved is not None
        replay_call = resolved.replay_call()
        assert replay_call.kind == "extension_loader"
        assert replay_call.loader_method == "load_scaled"
        replay_spec = file_load(
            start_label="Load extension data",
            seed_code="derived = xr.DataArray([12.0])",
            file_load_source=FileLoadSource(
                path=str(value_path),
                loader_label="Extension Loader",
                loader_text="multi_extension",
                kwargs_text="scale=3.0",
                replay_call=replay_call,
            ),
        )
        call_threads.clear()
        processed_events: list[None] = []
        QtCore.QTimer.singleShot(0, lambda: processed_events.append(None))
        xr.testing.assert_identical(
            replay_file_provenance(
                replay_spec,
                extension_loader_executor=manager._extensions.replay_loader,
            ),
            xr.DataArray([12.0]),
        )
        assert len(call_threads) == 1
        assert call_threads[0] != threading.get_ident()
        assert stale_load_calls == []
        assert processed_events == [None]
        system_calls: list[str] = []
        monkeypatch.setattr(
            os,
            "system",
            lambda command: system_calls.append(command) or 0,
        )
        malicious_call = replay_call.model_copy(update={"loader_method": "os.system"})
        malicious_source = replay_spec.file_load_source.model_copy(
            update={"replay_call": malicious_call}
        )
        with pytest.raises(
            erlab.extensions.ExtensionExecutionError, match="not approved"
        ):
            manager._extensions.replay_loader(malicious_source)
        assert system_calls == []
        code = resolved.load_code(value_path, assign="loaded")
        assert code is not None

        dialog = _FileLoadEditDialog(
            FileLoadSource(
                path=str(value_path),
                loader_label="Extension Loader",
                loader_text="multi_extension",
                kwargs_text="scale=3.0",
                replay_call=replay_call,
            ),
            manager,
            file_loaders=manager._available_file_loaders,
        )
        try:
            selected_filter = dialog._checked_filter_name()
            assert selected_filter is not None
            selected_func = dialog.loader_options._valid_loaders[selected_filter][0]
            assert (
                _resolve_load_func(
                    (
                        selected_func,
                        scaled_defaults,
                        FileDataSelection(kind="dataarray"),
                    )
                )
                .replay_call()
                .loader_method
                == "load_scaled"
            )
        finally:
            dialog.close()

        manager._data_load([str(value_path)], "multi_extension", {"single": True})
        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers) == 1,
            timeout=5000,
        )
        loaded_spec = manager._tool_graph.root_wrappers[0].provenance_spec
        assert loaded_spec is not None
        assert loaded_spec.file_load_source is not None
        loaded_call = loaded_spec.file_load_source.replay_call
        assert loaded_call is not None
        assert loaded_call.kind == "extension_loader"
        assert loaded_call.target == extension_id
        assert loaded_call.revision == record.current_revision

    namespace: dict[str, typing.Any] = {}
    exec(code, namespace)  # noqa: S102
    xr.testing.assert_identical(namespace["loaded"], xr.DataArray([12.0]))


def test_editable_source_fingerprint_tracks_custom_layout_and_sibling_modules(
    tmp_path: pathlib.Path,
) -> None:
    source = tmp_path / "python" / "packages" / "lab_package" / "plugin.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n")
    helper = tmp_path / "python" / "packages" / "shared" / "helper.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("HELPER = 1\n")

    direct_url = {
        "url": tmp_path.as_uri(),
        "dir_info": {"editable": True},
    }
    first = extension_entry_points._editable_source_fingerprint(direct_url)
    source.write_text("VALUE = 2\n")
    second = extension_entry_points._editable_source_fingerprint(direct_url)
    helper.write_text("HELPER = 2\n")
    third = extension_entry_points._editable_source_fingerprint(direct_url)

    assert second is not None
    assert first != second
    assert second != third


def test_editable_source_fingerprint_rejects_unknown_source(
    tmp_path: pathlib.Path,
) -> None:
    direct_url = {
        "url": tmp_path.as_uri(),
        "dir_info": {"editable": True},
    }

    with pytest.raises(
        extension_entry_points._EntryPointRevisionError,
        match="no fingerprintable source",
    ):
        extension_entry_points._editable_source_fingerprint(direct_url)


def test_editable_package_source_change_creates_unapproved_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    source = tmp_path / "custom" / "lab_package" / "plugin.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n")
    helper = tmp_path / "custom" / "shared" / "helper.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("HELPER = 1\n")

    @erlab.extensions.routine()
    def extension(data: xr.DataArray) -> xr.DataArray:
        return data

    class _Distribution:
        metadata: typing.ClassVar[dict[str, str]] = {"Name": "lab-package"}
        version = "1"

        @staticmethod
        def read_text(name: str) -> str | None:
            if name != "direct_url.json":
                return None
            return json.dumps(
                {
                    "url": tmp_path.as_uri(),
                    "dir_info": {"editable": True},
                }
            )

    class _EntryPoint:
        group = "erlab.extensions"
        name = "editable"
        value = "lab_package.plugin:extension"
        dist = _Distribution()

        @staticmethod
        def load():
            return extension

    class _EntryPoints(tuple):
        def select(self, *, group: str):
            return tuple(entry for entry in self if entry.group == group)

    monkeypatch.setattr(
        extension_catalog.importlib.metadata,
        "entry_points",
        lambda: _EntryPoints((_EntryPoint(),)),
    )
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    catalog = store.refresh_environment_packages()
    extension_id = "environment.erlab.extensions.editable"
    catalog = _validate_and_enable(
        store,
        extension_id,
        expected_record_generation=catalog.extensions[extension_id].record_generation,
    )
    approved_revision = catalog.extensions[extension_id].current_revision

    helper.write_text("HELPER = 2\n")
    catalog = store.refresh_environment_packages()

    record = catalog.extensions[extension_id]
    assert record.current_revision != approved_revision
    assert not record.enabled
    assert not record.revisions[record.current_revision].approved

    with pytest.raises(
        extension_entry_points._EntryPointReloadRequiredError,
        match="Restart Python",
    ):
        _validate_and_enable(
            store,
            extension_id,
            expected_record_generation=record.record_generation,
        )


def test_environment_refresh_skips_an_invalid_entry_point(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class BrokenDistribution:
        metadata: typing.ClassVar[dict[str, str]] = {"Name": "broken-package"}
        version = "1"

        @staticmethod
        def read_text(name: str) -> str | None:
            if name != "direct_url.json":
                return None
            return json.dumps(
                {
                    "url": (tmp_path / "missing").as_uri(),
                    "dir_info": {"editable": True},
                }
            )

    broken = types.SimpleNamespace(
        group="erlab.extensions",
        name="broken",
        value="broken_package",
        dist=BrokenDistribution(),
    )
    valid = types.SimpleNamespace(
        group="erlab.extensions",
        name="valid",
        value="valid_package",
        dist=None,
    )

    class EntryPoints(tuple):
        def select(self, **parameters):
            return tuple(
                entry
                for entry in self
                if all(
                    getattr(entry, key, None) == value
                    for key, value in parameters.items()
                )
            )

    monkeypatch.setattr(
        extension_catalog.importlib.metadata,
        "entry_points",
        lambda: EntryPoints((broken, valid)),
    )

    catalog = _ExtensionCatalogStore(
        tmp_path / "catalog"
    ).refresh_environment_packages()

    assert "environment.erlab.extensions.valid" in catalog.extensions
    assert "environment.erlab.extensions.broken" not in catalog.extensions
    assert "Could not inspect environment extension erlab.extensions:broken" in (
        caplog.text
    )


def test_environment_refresh_does_not_restore_a_removed_extension(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    class _EntryPoint:
        group = "erlab.extensions"
        name = "removed"
        value = "lab_package:extension"
        dist = None

    class _EntryPoints(tuple):
        def select(self, *, group: str):
            return tuple(entry for entry in self if entry.group == group)

    entry_point = _EntryPoint()
    entry_points = _EntryPoints((entry_point,))
    monkeypatch.setattr(
        extension_catalog.importlib.metadata,
        "entry_points",
        lambda: entry_points,
    )
    store = _ExtensionCatalogStore(tmp_path / "catalog")

    catalog = store.refresh_environment_packages()
    extension_id = "environment.erlab.extensions.removed"
    record = catalog.extensions[extension_id]
    catalog = store.update_record(
        extension_id,
        expected_record_generation=record.record_generation,
        removed=True,
    )
    removed_revision = catalog.extensions[extension_id].current_revision

    catalog = store.refresh_environment_packages()
    assert catalog.extensions[extension_id].removed
    assert catalog.extensions[extension_id].current_revision == removed_revision

    entry_point.value = "lab_package:changed_extension"
    catalog = store.refresh_environment_packages()
    record = catalog.extensions[extension_id]
    assert record.removed
    assert not record.enabled
    assert record.current_revision != removed_revision

    entry_point.value = "lab_package:extension"
    catalog = store.refresh_environment_packages()
    record = catalog.extensions[extension_id]
    assert record.removed
    assert not record.enabled
    assert record.current_revision == removed_revision
    assert len(record.revisions) == 2


def test_packaged_manager_does_not_hide_bundled_loader_names(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _EntryPoint:
        group = "erlab.io.loaders"
        name = "bundled_loader"
        value = "lab_package:BundledLoader"
        dist = None

    class _EntryPoints(tuple):
        def select(self, **parameters):
            return tuple(
                entry
                for entry in self
                if all(
                    getattr(entry, key, None) == value
                    for key, value in parameters.items()
                )
            )

    monkeypatch.setattr(
        extension_catalog.importlib.metadata,
        "entry_points",
        lambda: _EntryPoints((_EntryPoint(),)),
    )
    _ExtensionCatalogStore().refresh_environment_packages()
    monkeypatch.setattr(erlab.utils.misc, "_IS_PACKAGED", True)

    with manager_context() as manager:
        assert manager._extensions.environment_loader_names == set()
        assert manager._extensions.explorer_loaders == {}


def test_manager_reports_unknown_requested_loader(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    shown: list[None] = []
    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        lambda *_args, **_kwargs: shown.append(None),
    )
    data_path = tmp_path / "data.txt"
    data_path.write_text("unused")

    with manager_context() as manager:
        manager._data_load([str(data_path)], "missing-loader", {})

    assert shown == [None]


def test_workspace_extension_blob_reachability(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "extensions.itws"
    source = b"print('not imported')\n"
    object_id = f"extension-{hashlib.sha256(source).hexdigest()}"
    manifest = {
        "schema_version": 6,
        "nodes": [],
        "root_order": [],
        "extension_requirements": [
            {
                "extension_id": "lab",
                "capability_id": "routine",
                "embedded_object_id": object_id,
            }
        ],
    }
    with workspace_store.WorkspaceStore(path, create=True) as store:
        workspace_storage._write_workspace_generation(
            store,
            workspace_storage._WorkspaceGenerationPlan(
                manifest=manifest,
                objects=(
                    workspace_storage._WorkspaceObjectWrite(
                        object_id,
                        blob=source,
                        blob_kind="extension-python-source-v1",
                    ),
                ),
            ),
            compression_mode="none",
        )
        assert object_id in store.manifest_object_ids(
            store.current_generation().manifest
        )
    restored, kind = workspace_storage._read_workspace_blob(path, object_id)
    assert restored == source
    assert kind == "extension-python-source-v1"


def test_workspace_requirement_rejects_a_mismatched_embedded_object_id() -> None:
    with pytest.raises(ValueError, match="does not match its revision"):
        _WorkspaceExtensionRequirement(
            extension_id="lab",
            capability_id="routine",
            capability_kind="routine",
            revision_hash="a" * 64,
            extension_api_version=1,
            source_type="script",
            embedded_object_id="extension-node-data",
        )


def test_extension_object_write_cannot_replace_node_data(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "collision.py"
    source = _script(script_path)
    revision = hashlib.sha256(source).hexdigest()
    object_id = f"extension-{revision}"
    workspace_path = tmp_path / "collision.itws"
    requirement = {
        "extension_id": "collision",
        "capability_id": "scale",
        "capability_kind": "routine",
        "revision_hash": revision,
        "extension_api_version": 1,
        "source_type": "script",
        "embedded_object_id": object_id,
        "referencing_nodes": [],
        "file_sources": [],
        "metadata_snapshot": {},
    }

    with manager_context() as manager:
        manager._extensions.catalog.store.add_script(script_path)
        manager._extensions.set_workspace_requirements(
            (), unresolved_payloads=(requirement,)
        )
        manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(
                xr.DataArray([1.0, 2.0], dims="x"), _in_manager=True
            ),
            show=False,
        )
        monkeypatch.setattr(
            workspace_saving.uuid,
            "uuid4",
            lambda: types.SimpleNamespace(hex=object_id),
        )
        manager._workspace_controller.saving._save_workspace_document(workspace_path)

    with workspace_store.WorkspaceStore(workspace_path) as store:
        group = store.h5_file[workspace_store.WorkspaceStore.object_path(object_id)]
        assert group.attrs.get("erlab_object_kind") != "extension-python-source-v1"
        assert "source" not in group


def test_unused_script_can_be_embedded_explicitly(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "scale.py"
    source = _script(script_path)
    workspace_path = tmp_path / "explicit-extension.itws"

    with manager_context() as manager:
        catalog, revision, _created = manager._extensions.catalog.store.add_script(
            script_path
        )
        catalog = _validate_and_enable(
            manager._extensions.catalog.store,
            "scale",
            expected_record_generation=catalog.extensions["scale"].record_generation,
        )
        manager._extensions.catalog.store.update_record(
            "scale",
            expected_record_generation=catalog.extensions["scale"].record_generation,
            embed_policy="always",
        )
        manager._extensions.catalog.refresh()

        manager._workspace_controller.saving._save_workspace_document(workspace_path)

    attrs = workspace_arrays._read_workspace_root_attrs_h5py(workspace_path)
    manifest = workspace_format._workspace_manifest_from_attrs(attrs)
    requirements = manifest["extension_requirements"]
    assert len(requirements) == 1
    assert requirements[0]["revision_hash"] == revision
    object_id = requirements[0]["embedded_object_id"]
    restored, kind = workspace_storage._read_workspace_blob(workspace_path, object_id)
    assert restored == source
    assert kind == "extension-python-source-v1"


def test_verified_catalog_source_replaces_corrupt_embedded_source(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "catalog_source.py"
    source = _script(script_path)
    workspace_path = tmp_path / "verified-source.itws"

    with manager_context() as manager:
        catalog, revision, _created = manager._extensions.catalog.store.add_script(
            script_path
        )
        catalog = _validate_and_enable(
            manager._extensions.catalog.store,
            "catalog_source",
            expected_record_generation=(
                catalog.extensions["catalog_source"].record_generation
            ),
        )
        manager._extensions.catalog.store.update_record(
            "catalog_source",
            expected_record_generation=(
                catalog.extensions["catalog_source"].record_generation
            ),
            embed_policy="always",
        )
        manager._extensions.catalog.refresh()
        manager._extensions.set_workspace_requirements(
            (), embedded_sources={("catalog_source", revision): b"corrupt"}
        )

        manager._workspace_controller.saving._save_workspace_document(workspace_path)

    attrs = workspace_arrays._read_workspace_root_attrs_h5py(workspace_path)
    manifest = workspace_format._workspace_manifest_from_attrs(attrs)
    object_id = manifest["extension_requirements"][0]["embedded_object_id"]
    restored, kind = workspace_storage._read_workspace_blob(workspace_path, object_id)
    assert restored == source
    assert kind == "extension-python-source-v1"


def test_missing_catalog_source_does_not_create_dangling_embedded_object(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "missing_source.py"
    _script(script_path)
    workspace_path = tmp_path / "missing-source.itws"

    with manager_context() as manager:
        catalog, revision, _created = manager._extensions.catalog.store.add_script(
            script_path
        )
        catalog = _validate_and_enable(
            manager._extensions.catalog.store,
            "missing_source",
            expected_record_generation=(
                catalog.extensions["missing_source"].record_generation
            ),
        )
        manager._extensions.catalog.store.update_record(
            "missing_source",
            expected_record_generation=(
                catalog.extensions["missing_source"].record_generation
            ),
            embed_policy="always",
        )
        manager._extensions.catalog.store.source_path(
            "missing_source", revision
        ).unlink()
        manager._extensions.catalog.refresh()

        manager._workspace_controller.saving._save_workspace_document(workspace_path)

    attrs = workspace_arrays._read_workspace_root_attrs_h5py(workspace_path)
    manifest = workspace_format._workspace_manifest_from_attrs(attrs)
    assert len(manifest["extension_requirements"]) == 1
    assert manifest["extension_requirements"][0]["embedded_object_id"] is None


def test_unavailable_script_revision_is_omitted_from_gui_discovery(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "unavailable.py"
    script_path.write_text(
        """from pathlib import Path
import xarray as xr
from erlab.extensions import loader, routine

@routine(name="Unavailable routine")
def analyze(data: xr.DataArray) -> xr.DataArray:
    return data

@loader(name="Unavailable loader", extensions=(".txt",))
def load_missing(path: Path) -> xr.DataArray:
    return xr.DataArray([1.0])
"""
    )

    with manager_context() as manager:
        catalog, revision, _created = manager._extensions.catalog.store.add_script(
            script_path
        )
        _validate_and_enable(
            manager._extensions.catalog.store,
            "unavailable",
            expected_record_generation=(
                catalog.extensions["unavailable"].record_generation
            ),
        )
        manager._extensions.catalog.refresh()
        manager._extensions._sync_explorer_loaders()
        assert (
            manager._extensions.loader_by_name("unavailable:load_missing") is not None
        )
        assert any(
            extension_id == "unavailable"
            for extension_id, _extension_name, _descriptor in (
                manager._extensions._enabled_routines()
            )
        )

        manager._extensions.catalog.store.source_path("unavailable", revision).unlink()
        manager._extensions._sync_explorer_loaders()

        assert manager._extensions.loader_by_name("unavailable:load_missing") is None
        assert all(
            extension_id != "unavailable"
            for extension_id, _extension_name, _descriptor in (
                manager._extensions._enabled_routines()
            )
        )
        assert all(
            getattr(func, "extension_id", None) != "unavailable"
            for func, _defaults in manager._extensions.file_loaders().values()
        )


def test_workspace_requirement_states_do_not_import_embedded_code(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    source = b"raise RuntimeError('must not import during inspection')\n"
    revision = hashlib.sha256(source).hexdigest()
    requirement = _WorkspaceExtensionRequirement(
        extension_id="workspace-only",
        capability_id="routine",
        capability_kind="routine",
        revision_hash=revision,
        extension_api_version=1,
        source_type="script",
        embedded_object_id=f"extension-{revision}",
    )

    with manager_context() as manager:
        manager._extensions.set_workspace_requirements(
            (requirement,),
            embedded_sources={("workspace-only", revision): source},
        )
        assert manager._extensions.resolved_workspace_requirements()[0].state == (
            "approval-required"
        )

        manager._extensions.set_workspace_requirements(
            (requirement,),
            embedded_sources={("workspace-only", revision): b"different"},
        )
        assert manager._extensions.resolved_workspace_requirements()[0].state == (
            "hash-mismatch"
        )


def test_workspace_requirements_dialog_refreshes_after_approval(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requirement = _WorkspaceExtensionRequirement(
        extension_id="workspace-only",
        capability_id="routine",
        capability_kind="routine",
        revision_hash="a" * 64,
        extension_api_version=1,
        source_type="script",
    )
    current = [
        _ResolvedWorkspaceRequirement(
            requirement=requirement, state="approval-required"
        )
    ]
    shown_dialogs = []

    with manager_context() as manager:
        manager._extensions.set_workspace_requirements(
            (requirement,),
            embedded_sources={
                (requirement.extension_id, requirement.revision_hash): b""
            },
        )
        monkeypatch.setattr(
            manager._extensions,
            "resolved_workspace_requirements",
            lambda: tuple(current),
        )

        def approve(_extension_id: str, _revision: str) -> None:
            current[0] = current[0].model_copy(update={"state": "ready"})

        def execute(dialog) -> int:
            shown_dialogs.append(dialog)
            dialog.tree.setCurrentItem(dialog.tree.topLevelItem(0))
            dialog._approve_selected()
            return 0

        monkeypatch.setattr(manager._extensions, "_approve_embedded_script", approve)
        monkeypatch.setattr(
            extension_controller._WorkspaceRequirementsDialog, "exec", execute
        )

        manager._extensions.show_workspace_requirements()

        dialog = shown_dialogs[0]
        item = dialog.tree.topLevelItem(0)
        assert item.data(0, QtCore.Qt.ItemDataRole.UserRole + 1) == "ready"
        assert not dialog._approve_button.isEnabled()


@pytest.mark.parametrize("selected", [False, True])
def test_add_script_file_dialog_accept_and_cancel(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    selected: bool,
) -> None:
    script_path = tmp_path / "selected.py"
    reviewed: list[pathlib.Path] = []

    with manager_context() as manager:
        monkeypatch.setattr(
            extension_controller.QtWidgets.QFileDialog,
            "getOpenFileName",
            lambda *_args, **_kwargs: (
                str(script_path) if selected else "",
                "Python scripts (*.py)",
            ),
        )
        monkeypatch.setattr(
            manager._extensions,
            "_review_and_add",
            lambda path: reviewed.append(path) or True,
        )

        manager._extensions.add_script()

    assert reviewed == ([script_path] if selected else [])


def test_context_extension_menu_disconnects_exact_refresh_slot(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[None] = []

    with manager_context() as manager:
        monkeypatch.setattr(
            manager._extensions,
            "_populate_routine_menu",
            lambda *_args, **_kwargs: calls.append(None),
        )
        parent_menu = extension_controller.QtWidgets.QMenu(manager)
        menu = manager._extensions.add_context_submenu(parent_menu)
        calls.clear()

        menu.aboutToShow.emit()
        assert calls == [None]

        manager._extensions.close()
        menu.aboutToShow.emit()
        assert calls == [None]


def test_extension_actions_disconnect_when_controller_closes(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dialog_calls: list[None] = []
    monkeypatch.setattr(
        extension_controller.QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (dialog_calls.append(None) or "", ""),
    )

    with manager_context() as manager:
        manager._extensions.add_script_action.trigger()
        assert dialog_calls == [None]

        manager._extensions.close()
        manager._extensions.add_script_action.trigger()
        assert dialog_calls == [None]


def test_manage_dialog_enables_only_applicable_actions(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "script.py"
    _script(script_path)

    with manager_context() as manager:
        catalog, _revision, _created = manager._extensions.catalog.store.add_script(
            script_path
        )
        script_record = catalog.extensions["script"]
        environment_revision = "b" * 64
        environment_record = _ExtensionRecord(
            id="environment",
            name="Environment",
            source_type="environment-package",
            current_revision=environment_revision,
            revisions={
                environment_revision: _ExtensionRevision(
                    source_hash=environment_revision,
                    object_name="lab_package:extension",
                    created_at="2026-01-01T00:00:00+00:00",
                    entry_point_group="erlab.extensions",
                    entry_point_name="extension",
                    entry_point_value="lab_package:extension",
                )
            },
        )
        dialog = manager._extensions._manage_dialog
        dialog.set_catalog(
            _ExtensionCatalogModel(
                extensions={
                    script_record.id: script_record,
                    environment_record.id: environment_record,
                }
            )
        )

        def select(extension_id: str) -> None:
            for index in range(dialog.tree.topLevelItemCount()):
                item = dialog.tree.topLevelItem(index)
                if item.data(0, QtCore.Qt.ItemDataRole.UserRole) == extension_id:
                    dialog.tree.setCurrentItem(item)
                    return
            raise AssertionError(extension_id)

        select("script")
        assert dialog._buttons["reload"].isEnabled()
        assert dialog._buttons["embedding"].isEnabled()

        select("environment")
        assert not dialog._buttons["reload"].isEnabled()
        assert not dialog._buttons["embedding"].isEnabled()
        assert dialog._buttons["toggle"].isEnabled()

        removed = environment_record.model_copy(update={"removed": True})
        dialog.set_catalog(_ExtensionCatalogModel(extensions={removed.id: removed}))
        select("environment")
        assert not dialog._buttons["toggle"].isEnabled()
        assert dialog._buttons["remove"].isEnabled()


def test_logically_removed_extension_can_be_restored(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "restorable.py"
    _script(script_path)

    with manager_context() as manager:
        manager._extensions.catalog.store.add_script(script_path)
        manager._extensions.catalog.refresh()

        manager._extensions._manage_action("remove", "restorable")
        removed = manager._extensions.catalog.model.extensions["restorable"]
        assert removed.removed
        assert not removed.enabled

        manager._extensions._manage_action("remove", "restorable")
        restored = manager._extensions.catalog.model.extensions["restorable"]
        assert not restored.removed
        assert not restored.enabled


def test_file_source_status_does_not_import_extension_code(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    marker = tmp_path / "imported"
    script_path = tmp_path / "source_loader.py"
    script_path.write_text(
        f"""from pathlib import Path
import xarray as xr
from erlab.extensions import loader

Path({str(marker)!r}).write_text("imported")

@loader(extensions=(".txt",))
def source_loader(path: Path) -> xr.DataArray:
    return xr.DataArray([1.0])
"""
    )
    data_path = tmp_path / "data.txt"
    data_path.write_text("unused")

    with manager_context() as manager:
        catalog, revision, _created = manager._extensions.catalog.store.add_script(
            script_path
        )
        spec = file_load(
            start_label="Load extension data",
            seed_code="data = xr.DataArray([1.0])",
            file_load_source=FileLoadSource(
                path=str(data_path),
                loader_label="Source Loader",
                loader_text="source_loader",
                kwargs_text="",
                replay_call=FileReplayCall(
                    kind="extension_loader",
                    target="source_loader",
                    revision=revision,
                    capability_id="source_loader",
                    selection=FileDataSelection(kind="dataarray"),
                ),
            ),
        )

        assert file_load_source_status(spec) == "extension-approval-required"
        assert not marker.exists()
        catalog = _validate_and_enable(
            manager._extensions.catalog.store,
            "source_loader",
            expected_record_generation=(
                catalog.extensions["source_loader"].record_generation
            ),
        )
        manager._extensions.catalog.refresh()
        marker.unlink()

        assert file_load_source_status(spec) == "loadable"
        assert not marker.exists()

        catalog = manager._extensions.catalog.store.update_record(
            "source_loader",
            expected_record_generation=(
                catalog.extensions["source_loader"].record_generation
            ),
            enabled=False,
        )
        assert file_load_source_status(spec) == "extension-disabled"
        catalog = manager._extensions.catalog.store.update_record(
            "source_loader",
            expected_record_generation=(
                catalog.extensions["source_loader"].record_generation
            ),
            enabled=True,
        )
        assert file_load_source_status(spec) == "loadable"

        load_source = spec.file_load_source
        if load_source is None or load_source.replay_call is None:
            raise RuntimeError("The test file provenance is incomplete")
        missing_revision = spec.model_copy(
            update={
                "file_load_source": load_source.model_copy(
                    update={
                        "replay_call": load_source.replay_call.model_copy(
                            update={"revision": "b" * 64}
                        )
                    }
                )
            }
        )
        assert file_load_source_status(missing_revision) == (
            "extension-missing-revision"
        )
        missing_capability = spec.model_copy(
            update={
                "file_load_source": load_source.model_copy(
                    update={
                        "replay_call": load_source.replay_call.model_copy(
                            update={"capability_id": "missing"}
                        )
                    }
                )
            }
        )
        assert file_load_source_status(missing_capability) == (
            "extension-missing-capability"
        )

        manager._extensions.catalog.store.source_path(
            "source_loader", revision
        ).write_bytes(b"changed source")
        assert file_load_source_status(spec) == "extension-hash-mismatch"
        assert not marker.exists()


def test_file_source_status_reports_extension_import_failure(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "broken_loader.py"
    script_path.write_text("raise RuntimeError('missing dependency')\n")
    data_path = tmp_path / "data.txt"
    data_path.write_text("unused")

    with manager_context() as manager:
        catalog, revision, _created = manager._extensions.catalog.store.add_script(
            script_path
        )
        with pytest.raises(erlab.extensions.ExtensionImportError):
            _validate_and_enable(
                manager._extensions.catalog.store,
                "broken_loader",
                expected_record_generation=(
                    catalog.extensions["broken_loader"].record_generation
                ),
            )
        spec = file_load(
            start_label="Load extension data",
            seed_code="data = xr.DataArray([1.0])",
            file_load_source=FileLoadSource(
                path=str(data_path),
                loader_label="Broken Loader",
                loader_text="broken_loader",
                kwargs_text="",
                replay_call=FileReplayCall(
                    kind="extension_loader",
                    target="broken_loader",
                    revision=revision,
                    capability_id="broken_loader",
                    selection=FileDataSelection(kind="dataarray"),
                ),
            ),
        )

        assert file_load_source_status(spec) == "extension-import-failed"


def test_workspace_requirement_catalog_states(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "scale.py"
    _script(script_path)

    with manager_context() as manager:
        catalog, revision, _created = manager._extensions.catalog.store.add_script(
            script_path
        )
        requirement = _WorkspaceExtensionRequirement(
            extension_id="scale",
            capability_id="scale",
            capability_kind="routine",
            revision_hash=revision,
            extension_api_version=1,
            source_type="script",
        )
        manager._extensions.catalog.refresh()
        manager._extensions.set_workspace_requirements((requirement,))
        assert manager._extensions.resolved_workspace_requirements()[0].state == (
            "approval-required"
        )

        catalog = _validate_and_enable(
            manager._extensions.catalog.store,
            "scale",
            expected_record_generation=catalog.extensions["scale"].record_generation,
        )
        manager._extensions.catalog.refresh()
        assert manager._extensions.resolved_workspace_requirements()[0].state == (
            "ready"
        )

        manager._extensions.catalog.store.update_record(
            "scale",
            expected_record_generation=catalog.extensions["scale"].record_generation,
            enabled=False,
        )
        manager._extensions.catalog.refresh()
        assert manager._extensions.resolved_workspace_requirements()[0].state == (
            "disabled"
        )

        manager._extensions.set_workspace_requirements(
            (requirement.model_copy(update={"extension_id": "missing"}),)
        )
        assert manager._extensions.resolved_workspace_requirements()[0].state == (
            "missing"
        )
        manager._extensions.set_workspace_requirements(
            (requirement.model_copy(update={"extension_api_version": 2}),)
        )
        assert manager._extensions.resolved_workspace_requirements()[0].state == (
            "unsupported-api"
        )
        manager._extensions.set_workspace_requirements(
            (requirement.model_copy(update={"source_type": "environment-package"}),)
        )
        assert manager._extensions.resolved_workspace_requirements()[0].state == (
            "missing"
        )

        broken_path = tmp_path / "broken.py"
        broken_path.write_text("raise RuntimeError('broken import')\n")
        catalog, broken_revision, _created = (
            manager._extensions.catalog.store.add_script(broken_path)
        )
        with pytest.raises(erlab.extensions.ExtensionImportError):
            _validate_and_enable(
                manager._extensions.catalog.store,
                "broken",
                expected_record_generation=(
                    catalog.extensions["broken"].record_generation
                ),
            )
        manager._extensions.catalog.refresh()
        manager._extensions.set_workspace_requirements(
            (
                requirement.model_copy(
                    update={
                        "extension_id": "broken",
                        "revision_hash": broken_revision,
                    }
                ),
            )
        )
        assert manager._extensions.resolved_workspace_requirements()[0].state == (
            "import-failed"
        )


@pytest.mark.parametrize(
    ("stored_source", "expected_state"),
    [(None, "missing"), (b"corrupt", "hash-mismatch")],
)
def test_embedded_source_does_not_mask_unusable_catalog_revision(
    manager_context,
    tmp_path: pathlib.Path,
    stored_source: bytes | None,
    expected_state: str,
) -> None:
    script_path = tmp_path / "catalog_source.py"
    source = _script(script_path)

    with manager_context() as manager:
        catalog, revision, _created = manager._extensions.catalog.store.add_script(
            script_path
        )
        _validate_and_enable(
            manager._extensions.catalog.store,
            "catalog_source",
            expected_record_generation=(
                catalog.extensions["catalog_source"].record_generation
            ),
        )
        catalog_source = manager._extensions.catalog.store.source_path(
            "catalog_source", revision
        )
        if stored_source is None:
            catalog_source.unlink()
        else:
            catalog_source.write_bytes(stored_source)
        manager._extensions.catalog.refresh()
        manager._extensions.set_workspace_requirements(
            (
                _WorkspaceExtensionRequirement(
                    extension_id="catalog_source",
                    capability_id="scale",
                    capability_kind="routine",
                    revision_hash=revision,
                    extension_api_version=1,
                    source_type="script",
                    embedded_object_id=f"extension-{revision}",
                ),
            ),
            embedded_sources={("catalog_source", revision): source},
        )

        assert manager._extensions.resolved_workspace_requirements()[0].state == (
            expected_state
        )


def test_degraded_workspace_load_preserves_requirements_for_save_as(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    source_path = tmp_path / "missing-extension.itws"
    recovered_path = tmp_path / "recovered.itws"
    healthy_path = tmp_path / "healthy.itws"
    requirement = {
        "extension_id": "missing-lab",
        "capability_id": "normalize",
        "capability_kind": "routine",
        "revision_hash": "a" * 64,
        "extension_api_version": 1,
        "source_type": "script",
        "metadata_snapshot": {"extension_name": "Missing Lab"},
        "embedded_object_id": None,
        "referencing_nodes": [],
        "file_sources": [],
    }
    with workspace_store.WorkspaceStore(source_path, create=True) as store:
        workspace_storage._write_workspace_generation(
            store,
            workspace_storage._WorkspaceGenerationPlan(
                manifest={
                    "schema_version": 6,
                    "nodes": [],
                    "root_order": [],
                    "extension_requirements": [requirement],
                },
                objects=(),
            ),
            compression_mode="none",
        )
    with workspace_store.WorkspaceStore(healthy_path, create=True) as store:
        workspace_storage._write_workspace_generation(
            store,
            workspace_storage._WorkspaceGenerationPlan(
                manifest={"schema_version": 6, "nodes": [], "root_order": []},
                objects=(),
            ),
            compression_mode="none",
        )
    original = source_path.read_bytes()

    with manager_context() as manager:
        monkeypatch.setattr(
            manager._extensions,
            "notify_unavailable_workspace_requirements",
            lambda: None,
        )
        assert manager._workspace_controller.loading._load_workspace_file(
            source_path,
            replace=True,
            associate=True,
            mark_dirty=False,
            select=False,
        )
        assert manager._workspace_state.save_as_only
        assert manager._workspace_state.degraded_reasons == ("missing-lab: missing",)
        assert source_path.read_bytes() == original

        manager._workspace_controller.saving._save_workspace_document(recovered_path)
        assert source_path.read_bytes() == original

        assert manager._workspace_controller.loading._load_workspace_file(
            healthy_path,
            replace=True,
            associate=True,
            mark_dirty=False,
            select=False,
        )
        assert not manager._workspace_state.save_as_only
        assert manager._workspace_state.degraded_reasons == ()

    assert source_path.read_bytes() == original
    recovered_attrs = workspace_arrays._read_workspace_root_attrs_h5py(recovered_path)
    recovered_manifest = workspace_format._workspace_manifest_from_attrs(
        recovered_attrs
    )
    assert recovered_manifest["extension_requirements"] == [requirement]


def test_failed_workspace_load_restores_extension_requirement_state(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    workspace_path = tmp_path / "failed-extension-load.itws"
    incoming = _WorkspaceExtensionRequirement(
        extension_id="incoming",
        capability_id="normalize",
        capability_kind="routine",
        revision_hash="a" * 64,
        extension_api_version=1,
        source_type="script",
    )
    previous = incoming.model_copy(
        update={"extension_id": "previous", "revision_hash": "b" * 64}
    )
    previous_source = b"previous source"
    unresolved = ({"future_requirement": True},)
    with workspace_store.WorkspaceStore(workspace_path, create=True) as store:
        workspace_storage._write_workspace_generation(
            store,
            workspace_storage._WorkspaceGenerationPlan(
                manifest={
                    "schema_version": 6,
                    "nodes": [],
                    "root_order": [],
                    "extension_requirements": [incoming.model_dump(mode="json")],
                },
                objects=(),
            ),
            compression_mode="none",
        )

    with manager_context() as manager:
        manager._extensions.set_workspace_requirements(
            (previous,),
            embedded_sources={
                (previous.extension_id, previous.revision_hash): previous_source
            },
            unresolved_payloads=unresolved,
        )
        manager._workspace_state.save_as_only = True
        manager._workspace_state.degraded_reasons = ("previous: missing",)

        def fail_load(*_args, **_kwargs):
            raise RuntimeError("load failed")

        monkeypatch.setattr(
            manager._workspace_controller.loading,
            "_from_h5py_workspace_file",
            fail_load,
        )

        with pytest.raises(RuntimeError, match="load failed"):
            manager._workspace_controller.loading._load_workspace_file(
                workspace_path,
                replace=True,
                associate=False,
                mark_dirty=False,
                select=False,
            )

        restored = manager._extensions.workspace_requirement_state()
        assert restored[0] == (previous,)
        assert restored[1] == {
            (previous.extension_id, previous.revision_hash): previous_source
        }
        assert restored[2] == {}
        assert restored[3] == unresolved
        assert manager._workspace_state.save_as_only
        assert manager._workspace_state.degraded_reasons == ("previous: missing",)


def test_degraded_save_as_preserves_missing_environment_requirement(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    data_path = tmp_path / "data.txt"
    data_path.write_text("unused")
    saved_path = tmp_path / "missing-environment-extension.itws"
    extension_id = "environment.erlab.io.loaders.missing"
    revision = "b" * 64
    metadata = {"extension_name": "Missing Environment Loader"}
    requirement = _WorkspaceExtensionRequirement(
        extension_id=extension_id,
        capability_id="missing",
        capability_kind="loader",
        revision_hash=revision,
        extension_api_version=2,
        source_type="environment-package",
        metadata_snapshot=metadata,
        referencing_nodes=("old-node",),
        file_sources=(str(data_path),),
    )
    spec = file_load(
        start_label="Load missing environment data",
        seed_code="data = xr.DataArray([[1.0]])",
        file_load_source=FileLoadSource(
            path=str(data_path),
            loader_label="Missing Environment Loader",
            loader_text="missing",
            kwargs_text="",
            replay_call=FileReplayCall(
                kind="extension_loader",
                target=extension_id,
                revision=revision,
                capability_id="missing",
                selection=FileDataSelection(kind="dataarray"),
            ),
        ),
    )

    with manager_context() as manager:
        tool = erlab.interactive.imagetool.ImageTool(xr.DataArray([[1.0]]))
        manager.add_imagetool(tool, show=False, provenance_spec=spec)
        manager._extensions.set_workspace_requirements((requirement,))

        collected = manager._extensions.collect_workspace_requirements()
        assert len(collected) == 1
        assert collected[0].source_type == "environment-package"
        assert collected[0].extension_api_version == 2
        assert collected[0].metadata_snapshot == metadata
        assert collected[0].embedded_object_id is None

        manager._workspace_controller.saving._save_workspace_document(saved_path)

    attrs = workspace_arrays._read_workspace_root_attrs_h5py(saved_path)
    manifest = workspace_format._workspace_manifest_from_attrs(attrs)
    saved_requirement = manifest["extension_requirements"][0]
    assert saved_requirement["source_type"] == "environment-package"
    assert saved_requirement["extension_api_version"] == 2
    assert saved_requirement["metadata_snapshot"] == metadata
    assert saved_requirement["embedded_object_id"] is None


def test_removing_node_discards_only_its_workspace_requirements(
    manager_context,
) -> None:
    operation = ExtensionRoutineOperation(
        extension_id="missing-extension",
        revision_hash="a" * 64,
        routine_id="normalize",
        extension_name="Missing Extension",
        routine_name="Normalize",
        source_type="script",
        function_name="normalize",
        source_path="missing_extension.py",
        entry_point_group=None,
        entry_point_name=None,
        parameters={},
    )
    with manager_context() as manager:
        manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(xr.DataArray([1.0])),
            show=False,
            provenance_spec=full_data(operation),
        )
        manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(xr.DataArray([2.0])),
            show=False,
            provenance_spec=full_data(operation),
        )
        first_uid = manager._tool_graph.root_wrappers[0].uid
        second_uid = manager._tool_graph.root_wrappers[1].uid
        requirement = _WorkspaceExtensionRequirement(
            extension_id="missing-extension",
            capability_id="normalize",
            capability_kind="routine",
            revision_hash="a" * 64,
            extension_api_version=1,
            source_type="script",
            referencing_nodes=(first_uid, second_uid),
        )
        manager._extensions.set_workspace_requirements((requirement,))

        manager.remove_imagetool(0)

        collected = manager._extensions.collect_workspace_requirements()
        assert len(collected) == 1
        assert collected[0].referencing_nodes == (second_uid,)

        manager.remove_imagetool(1)
        assert manager._extensions.collect_workspace_requirements() == ()


def test_collecting_requirements_reconciles_loaded_and_unresolved_nodes(
    manager_context,
) -> None:
    revision = "c" * 64
    operation = ExtensionRoutineOperation(
        extension_id="workspace-routines",
        revision_hash=revision,
        routine_id="normalize",
        extension_name="Workspace Routines",
        routine_name="Normalize",
        source_type="script",
        function_name="normalize",
        source_path="workspace_routines.py",
        entry_point_group=None,
        entry_point_name=None,
        parameters={},
    )

    with manager_context() as manager:
        tool = erlab.interactive.imagetool.ImageTool(xr.DataArray([1.0]))
        index = manager.add_imagetool(
            tool, show=False, provenance_spec=full_data(operation)
        )
        node = manager._node_for_target(index)
        requirement = _WorkspaceExtensionRequirement(
            extension_id="workspace-routines",
            capability_id="normalize",
            capability_kind="routine",
            revision_hash=revision,
            extension_api_version=1,
            source_type="script",
            referencing_nodes=(node.uid, "unresolved-node"),
        )
        manager._extensions.set_workspace_requirements((requirement,))

        collected = manager._extensions.collect_workspace_requirements()
        assert collected[0].referencing_nodes == (node.uid, "unresolved-node")

        node.set_displayed_provenance(full_data())
        collected = manager._extensions.collect_workspace_requirements()
        assert collected[0].referencing_nodes == ("unresolved-node",)

        manager._extensions.set_workspace_requirements(
            (requirement.model_copy(update={"referencing_nodes": (node.uid,)}),)
        )
        assert manager._extensions.collect_workspace_requirements() == ()


def test_workspace_requirements_include_nested_script_inputs(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    revision = "c" * 64
    data_path = tmp_path / "nested.txt"
    data_path.write_text("unused")
    operation = ExtensionRoutineOperation(
        extension_id="nested-routines",
        revision_hash=revision,
        routine_id="normalize",
        extension_name="Nested Routines",
        routine_name="Normalize",
        source_type="script",
        function_name="normalize",
        source_path="nested_routines.py",
        entry_point_group=None,
        entry_point_name=None,
        parameters={},
    )
    nested = file_load(
        start_label="Load nested extension data",
        seed_code="nested = xr.DataArray([1.0])",
        file_load_source=FileLoadSource(
            path=str(data_path),
            loader_label="Nested Loader",
            loader_text="nested_loader",
            kwargs_text="",
            replay_call=FileReplayCall(
                kind="extension_loader",
                target="nested-loaders",
                revision=revision,
                capability_id="nested_loader",
                selection=FileDataSelection(kind="dataarray"),
            ),
        ),
    ).append_replay_stage(full_data(operation))
    root = ToolProvenanceSpec(
        kind="script",
        start_label="Combine inputs",
        seed_code="result = nested",
        active_name="result",
        script_inputs=(
            ScriptInput(
                name="nested",
                provenance_spec=nested.model_dump(mode="json"),
            ),
        ),
    )

    with manager_context() as manager:
        tool = erlab.interactive.imagetool.ImageTool(xr.DataArray([[1.0]]))
        index = manager.add_imagetool(tool, show=False, provenance_spec=root)
        node_uid = manager._node_for_target(index).uid

        requirements = manager._extensions.collect_workspace_requirements()

    assert {
        (item.extension_id, item.capability_kind, item.capability_id)
        for item in requirements
    } == {
        ("nested-loaders", "loader", "nested_loader"),
        ("nested-routines", "routine", "normalize"),
    }
    assert all(item.referencing_nodes == (node_uid,) for item in requirements)
    loader_requirement = next(
        item for item in requirements if item.capability_kind == "loader"
    )
    assert loader_requirement.file_sources == (str(data_path),)


def test_degraded_save_as_prefers_preserved_source_over_corrupt_catalog(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    source_path = tmp_path / "source.itws"
    recovered_path = tmp_path / "recovered.itws"
    script_path = tmp_path / "scale.py"
    source = _script(script_path)
    revision = hashlib.sha256(source).hexdigest()
    object_id = f"extension-{revision}"
    requirement = {
        "extension_id": "scale",
        "capability_id": "scale",
        "capability_kind": "routine",
        "revision_hash": revision,
        "extension_api_version": 1,
        "source_type": "script",
        "metadata_snapshot": {},
        "embedded_object_id": object_id,
        "referencing_nodes": [],
        "file_sources": [],
    }
    with workspace_store.WorkspaceStore(source_path, create=True) as store:
        workspace_storage._write_workspace_generation(
            store,
            workspace_storage._WorkspaceGenerationPlan(
                manifest={
                    "schema_version": 6,
                    "nodes": [],
                    "root_order": [],
                    "extension_requirements": [requirement],
                },
                objects=(
                    workspace_storage._WorkspaceObjectWrite(
                        object_id,
                        blob=source,
                        blob_kind="extension-python-source-v1",
                    ),
                ),
            ),
            compression_mode="none",
        )

    with manager_context() as manager:
        catalog, _revision, _created = manager._extensions.catalog.store.add_script(
            script_path
        )
        record = catalog.extensions["scale"]
        manager._extensions.catalog.store.source_path(
            record.id, record.current_revision
        ).write_bytes(b"corrupt catalog source")
        manager._extensions.catalog.refresh()
        monkeypatch.setattr(
            manager._extensions,
            "notify_unavailable_workspace_requirements",
            lambda: None,
        )

        assert manager._workspace_controller.loading._load_workspace_file(
            source_path,
            replace=True,
            associate=True,
            mark_dirty=False,
            select=False,
        )
        assert manager._workspace_state.save_as_only
        manager._workspace_controller.saving._save_workspace_document(recovered_path)

    restored, kind = workspace_storage._read_workspace_blob(recovered_path, object_id)
    assert restored == source
    assert kind == "extension-python-source-v1"


def test_degraded_save_as_preserves_requirement_with_missing_embedded_object(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    source_path = tmp_path / "missing-embedded-source.itws"
    recovered_path = tmp_path / "recovered.itws"
    revision = hashlib.sha256(b"missing source").hexdigest()
    object_id = f"extension-{revision}"
    requirement = _WorkspaceExtensionRequirement(
        extension_id="missing-embedded",
        capability_id="normalize",
        capability_kind="routine",
        revision_hash=revision,
        extension_api_version=1,
        source_type="script",
        embedded_object_id=object_id,
    ).model_dump(mode="json")
    with workspace_store.WorkspaceStore(source_path, create=True) as store:
        workspace_storage._write_workspace_generation(
            store,
            workspace_storage._WorkspaceGenerationPlan(
                manifest={
                    "schema_version": 6,
                    "nodes": [],
                    "root_order": [],
                    "extension_requirements": [requirement],
                },
                objects=(),
            ),
            compression_mode="none",
        )

    with manager_context() as manager:
        monkeypatch.setattr(
            manager._extensions,
            "notify_unavailable_workspace_requirements",
            lambda: None,
        )
        assert manager._workspace_controller.loading._load_workspace_file(
            source_path,
            replace=True,
            associate=True,
            mark_dirty=False,
            select=False,
        )
        assert manager._workspace_state.save_as_only
        manager._workspace_controller.saving._save_workspace_document(recovered_path)

    attrs = workspace_arrays._read_workspace_root_attrs_h5py(recovered_path)
    manifest = workspace_format._workspace_manifest_from_attrs(attrs)
    assert manifest["extension_requirements"] == [requirement]
    with pytest.raises(KeyError):
        workspace_storage._read_workspace_blob(recovered_path, object_id)
    with workspace_store.WorkspaceStore(recovered_path) as store:
        workspace_storage._compact_workspace_store(store)
        assert store.current_generation().manifest["extension_requirements"] == [
            requirement
        ]
    with pytest.raises(KeyError):
        workspace_storage._read_workspace_blob(recovered_path, object_id)


@pytest.mark.parametrize(
    "raw_requirements",
    [
        {"extension_id": "future", "future_field": {"value": 1}},
        ["future-requirement"],
    ],
)
def test_degraded_save_as_preserves_unparsed_requirement_payloads(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    raw_requirements: typing.Any,
) -> None:
    source_path = tmp_path / "unparsed-requirements.itws"
    recovered_path = tmp_path / "recovered.itws"
    with workspace_store.WorkspaceStore(source_path, create=True) as store:
        workspace_storage._write_workspace_generation(
            store,
            workspace_storage._WorkspaceGenerationPlan(
                manifest={
                    "schema_version": 6,
                    "nodes": [],
                    "root_order": [],
                    "extension_requirements": raw_requirements,
                },
                objects=(),
            ),
            compression_mode="none",
        )

    with manager_context() as manager:
        monkeypatch.setattr(
            manager._extensions,
            "notify_unavailable_workspace_requirements",
            lambda: None,
        )
        assert manager._workspace_controller.loading._load_workspace_file(
            source_path,
            replace=True,
            associate=True,
            mark_dirty=False,
            select=False,
        )
        assert manager._workspace_state.save_as_only
        manager._workspace_controller.saving._save_workspace_document(recovered_path)

    attrs = workspace_arrays._read_workspace_root_attrs_h5py(recovered_path)
    manifest = workspace_format._workspace_manifest_from_attrs(attrs)
    expected = (
        raw_requirements if isinstance(raw_requirements, list) else [raw_requirements]
    )
    assert manifest["extension_requirements"] == expected


def test_degraded_save_as_preserves_source_from_malformed_requirement_container(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    source_path = tmp_path / "malformed-requirements.itws"
    recovered_path = tmp_path / "recovered.itws"
    source = b"raise RuntimeError('must remain unresolved')\n"
    revision = hashlib.sha256(source).hexdigest()
    object_id = f"extension-{revision}"
    requirement = {
        "extension_id": "future-lab",
        "capability_id": "normalize",
        "capability_kind": "routine",
        "revision_hash": revision,
        "extension_api_version": 1,
        "source_type": "script",
        "metadata_snapshot": {},
        "embedded_object_id": object_id,
        "referencing_nodes": [],
        "file_sources": [],
    }
    with workspace_store.WorkspaceStore(source_path, create=True) as store:
        workspace_storage._write_workspace_generation(
            store,
            workspace_storage._WorkspaceGenerationPlan(
                manifest={
                    "schema_version": 6,
                    "nodes": [],
                    "root_order": [],
                    "extension_requirements": requirement,
                },
                objects=(
                    workspace_storage._WorkspaceObjectWrite(
                        object_id,
                        blob=source,
                        blob_kind="extension-python-source-v1",
                    ),
                ),
            ),
            compression_mode="none",
        )
        workspace_storage._compact_workspace_store(store)

    restored, kind = workspace_storage._read_workspace_blob(source_path, object_id)
    assert restored == source
    assert kind == "extension-python-source-v1"

    with manager_context() as manager:
        monkeypatch.setattr(
            manager._extensions,
            "notify_unavailable_workspace_requirements",
            lambda: None,
        )
        assert manager._workspace_controller.loading._load_workspace_file(
            source_path,
            replace=True,
            associate=True,
            mark_dirty=False,
            select=False,
        )
        assert manager._workspace_state.save_as_only
        manager._workspace_controller.saving._save_workspace_document(recovered_path)

    attrs = workspace_arrays._read_workspace_root_attrs_h5py(recovered_path)
    manifest = workspace_format._workspace_manifest_from_attrs(attrs)
    assert manifest["extension_requirements"] == [requirement]
    restored, kind = workspace_storage._read_workspace_blob(recovered_path, object_id)
    assert restored == source
    assert kind == "extension-python-source-v1"


def test_save_as_only_routes_offload_and_compaction_to_new_file(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    save_as_calls: list[dict[str, typing.Any]] = []

    with manager_context() as manager:
        manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(
                xr.DataArray(np.ones((2, 2)), dims=("x", "y")),
                _in_manager=True,
            ),
            show=False,
        )
        manager._workspace_state.path = tmp_path / "degraded.itws"
        manager._workspace_state.save_as_only = True
        monkeypatch.setattr(
            manager._workspace_controller,
            "save_as",
            lambda **kwargs: save_as_calls.append(kwargs) or True,
        )
        monkeypatch.setattr(
            manager._workspace_controller,
            "_offload_targets_to_current_workspace",
            lambda _targets: pytest.fail("offload wrote to the protected original"),
        )

        assert manager._workspace_controller.offload_to_workspace((0,), native=False)
        assert manager._workspace_controller.compact_workspace()

    assert len(save_as_calls) == 2
    assert save_as_calls[0]["native"] is False
    assert callable(save_as_calls[0]["on_finished"])
    assert callable(save_as_calls[1]["on_finished"])


def test_workspace_import_preserves_unavailable_embedded_source(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    source_path = tmp_path / "imported-extension.itws"
    saved_path = tmp_path / "saved-after-import.itws"
    source = b"raise RuntimeError('must remain unresolved')\n"
    revision = hashlib.sha256(source).hexdigest()
    object_id = f"extension-{revision}"
    requirement = {
        "extension_id": "imported-lab",
        "capability_id": "normalize",
        "capability_kind": "routine",
        "revision_hash": revision,
        "extension_api_version": 1,
        "source_type": "script",
        "metadata_snapshot": {},
        "embedded_object_id": object_id,
        "referencing_nodes": [],
        "file_sources": [],
    }
    with workspace_store.WorkspaceStore(source_path, create=True) as store:
        workspace_storage._write_workspace_generation(
            store,
            workspace_storage._WorkspaceGenerationPlan(
                manifest={
                    "schema_version": 6,
                    "nodes": [],
                    "root_order": [],
                    "extension_requirements": [requirement],
                },
                objects=(
                    workspace_storage._WorkspaceObjectWrite(
                        object_id,
                        blob=source,
                        blob_kind="extension-python-source-v1",
                    ),
                ),
            ),
            compression_mode="none",
        )

    with manager_context() as manager:
        monkeypatch.setattr(
            manager._extensions,
            "notify_unavailable_workspace_requirements",
            lambda: None,
        )
        assert manager._workspace_controller.loading._load_workspace_file(
            source_path,
            replace=False,
            associate=False,
            mark_dirty=True,
            select=False,
        )
        assert (
            manager._extensions.revision_source_bytes("imported-lab", revision)
            == source
        )
        manager._workspace_controller.saving._save_workspace_document(saved_path)

    restored, kind = workspace_storage._read_workspace_blob(saved_path, object_id)
    assert restored == source
    assert kind == "extension-python-source-v1"


def test_workspace_import_preserves_unparsed_embedded_source(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    source_path = tmp_path / "future-extension.itws"
    saved_path = tmp_path / "saved-after-import.itws"
    source = b"raise RuntimeError('must remain unresolved')\n"
    revision = hashlib.sha256(source).hexdigest()
    object_id = f"extension-{revision}"
    requirement = {
        "extension_id": "future-lab",
        "capability_id": "normalize",
        "capability_kind": "routine",
        "revision_hash": revision,
        "extension_api_version": 1,
        "source_type": "script",
        "metadata_snapshot": {},
        "embedded_object_id": object_id,
        "referencing_nodes": [],
        "file_sources": [],
        "future_field": {"value": 1},
    }
    with workspace_store.WorkspaceStore(source_path, create=True) as store:
        workspace_storage._write_workspace_generation(
            store,
            workspace_storage._WorkspaceGenerationPlan(
                manifest={
                    "schema_version": 6,
                    "nodes": [],
                    "root_order": [],
                    "extension_requirements": [requirement],
                },
                objects=(
                    workspace_storage._WorkspaceObjectWrite(
                        object_id,
                        blob=source,
                        blob_kind="extension-python-source-v2",
                    ),
                ),
            ),
            compression_mode="none",
        )

    with manager_context() as manager:
        monkeypatch.setattr(
            manager._extensions,
            "notify_unavailable_workspace_requirements",
            lambda: None,
        )
        assert manager._workspace_controller.loading._load_workspace_file(
            source_path,
            replace=False,
            associate=False,
            mark_dirty=True,
            select=False,
        )
        manager._workspace_controller.saving._save_workspace_document(saved_path)

    attrs = workspace_arrays._read_workspace_root_attrs_h5py(saved_path)
    manifest = workspace_format._workspace_manifest_from_attrs(attrs)
    assert manifest["extension_requirements"] == [requirement]
    restored, kind = workspace_storage._read_workspace_blob(saved_path, object_id)
    assert restored == source
    assert kind == "extension-python-source-v2"


def test_workspace_import_ignores_unselected_extension_requirements(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    requirement = _WorkspaceExtensionRequirement(
        extension_id="omitted-extension",
        capability_id="normalize",
        capability_kind="routine",
        revision_hash="a" * 64,
        extension_api_version=1,
        source_type="script",
        referencing_nodes=("omitted-node",),
    )
    manifest = {
        "schema_version": 6,
        "nodes": [
            {"path": "0", "uid": "selected-node", "kind": "imagetool"},
            {"path": "1", "uid": "omitted-node", "kind": "imagetool"},
        ],
        "root_order": ["0", "1"],
        "extension_requirements": [requirement.model_dump(mode="json")],
    }

    with manager_context() as manager:
        manager._workspace_controller.loading._prepare_extension_requirements(
            tmp_path / "selected.itws",
            manifest,
            replace=True,
            selected_paths={"0"},
        )

        assert manager._extensions.collect_workspace_requirements() == ()
        assert not manager._workspace_state.save_as_only
        assert manager._workspace_state.degraded_reasons == ()


def test_extensions_menu_is_before_dask(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    monkeypatch.setenv("ERLAB_EXTENSION_CATALOG", str(tmp_path / "catalog"))
    with manager_context() as manager:
        actions = manager.menu_bar.actions()
        assert actions.index(manager.extensions_menu.menuAction()) < actions.index(
            manager._dask_menu.menuAction()
        )
        assert manager.extensions_menu.objectName() == "manager_extensions_menu"


def test_decorated_loader_is_manager_local_and_module_isolated(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "counter_loader.py"
    script_path.write_text(
        """from pathlib import Path
import xarray as xr
from erlab.extensions import loader

counter = 0

@loader(name="Counter", extensions=("txt",))
def counter_loader(path: Path, scale: float = 1.0) -> xr.DataArray:
    global counter
    counter += 1
    return xr.DataArray([counter, float(path.read_text()) * scale])
"""
    )
    value_path = tmp_path / "value.txt"
    value_path.write_text("4")

    with manager_context() as manager:
        catalog, _revision, _created = manager._extensions.catalog.store.add_script(
            script_path
        )
        _validate_and_enable(
            manager._extensions.catalog.store,
            "counter_loader",
            expected_record_generation=(
                catalog.extensions["counter_loader"].record_generation
            ),
        )
        manager._extensions.catalog.refresh()

        loader_entries = manager._extensions.file_loaders((value_path,))
        assert len(loader_entries) == 1
        call, _defaults = next(iter(loader_entries.values()))
        first = call(value_path)
        second = call(value_path)
        xr.testing.assert_identical(first, xr.DataArray([1.0, 4.0]))
        xr.testing.assert_identical(second, xr.DataArray([2.0, 4.0]))
        assert "counter_loader:counter_loader" in (manager._extensions.explorer_loaders)
        resolved = _resolve_load_func((call, {}, FileDataSelection(kind="dataarray")))
        assert resolved is not None
        assert resolved.replay_call().kind == "extension_loader"
        code = resolved.load_code(value_path, assign="loaded")
        assert code is not None
        nonfinite = _resolve_load_func(
            (call, {"scale": np.nan}, FileDataSelection(kind="dataarray"))
        )
        assert nonfinite is not None
        nonfinite_code = nonfinite.load_code(value_path, assign="nonfinite")
        assert nonfinite_code is not None

    namespace: dict[str, typing.Any] = {}
    exec(code, namespace)  # noqa: S102
    xr.testing.assert_identical(namespace["loaded"], xr.DataArray([1.0, 4.0]))
    exec(nonfinite_code, namespace)  # noqa: S102
    assert namespace["nonfinite"][0].item() == 1.0
    assert np.isnan(namespace["nonfinite"][1].item())

    with manager_context() as second_manager:
        loader_entries = second_manager._extensions.file_loaders((value_path,))
        isolated_call, _defaults = next(iter(loader_entries.values()))
        xr.testing.assert_identical(isolated_call(value_path), xr.DataArray([1.0, 4.0]))


def test_loader_shares_routine_queue_and_rechecks_enablement(
    manager_context,
    qtbot: pytest.QtBot,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "serialized.py"
    script_path.write_text(
        """import pathlib
import time
import xarray as xr
from erlab.extensions import loader, routine

@routine(name="Slow")
def slow(data: xr.DataArray, marker: pathlib.Path) -> xr.DataArray:
    time.sleep(0.2)
    marker.write_text("routine\\n")
    return data

@loader(name="Marker", extensions=("txt",))
def marker_loader(path: pathlib.Path) -> xr.DataArray:
    with path.open("a") as stream:
        stream.write("loader\\n")
    return xr.DataArray([1.0])
"""
    )
    marker = tmp_path / "marker.txt"
    marker.touch()
    failures: list[BaseException] = []

    with manager_context() as manager:
        catalog, _revision, _created = manager._extensions.catalog.store.add_script(
            script_path
        )
        _validate_and_enable(
            manager._extensions.catalog.store,
            "serialized",
            expected_record_generation=(
                catalog.extensions["serialized"].record_generation
            ),
        )
        manager._extensions.catalog.refresh()
        manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(
                xr.DataArray([1.0], dims="x"), _in_manager=True
            ),
            show=False,
        )
        call, _defaults = next(
            iter(manager._extensions.file_loaders((marker,)).values())
        )
        manager._extensions.execution.queue_routine(
            extension_id="serialized",
            routine_id="slow",
            parameters={"marker": str(marker)},
            target=0,
        )
        qtbot.wait_until(
            lambda: manager._extensions.execution.active is not None,
            timeout=2000,
        )

        def invoke_loader() -> None:
            try:
                call(marker)
            except BaseException as error:
                failures.append(error)

        loader_thread = threading.Thread(target=invoke_loader)
        loader_thread.start()
        current = manager._extensions.catalog.store.read().extensions["serialized"]
        manager._extensions.catalog.store.update_record(
            "serialized",
            expected_record_generation=current.record_generation,
            enabled=False,
        )
        manager._extensions.catalog.refresh()
        qtbot.wait_until(lambda: not loader_thread.is_alive(), timeout=5000)
        loader_thread.join()
        qtbot.wait_until(
            lambda: manager._extensions.execution.active is None,
            timeout=5000,
        )

        assert marker.read_text() == "routine\n"
        assert len(failures) == 1
        assert isinstance(failures[0], erlab.extensions.ExtensionExecutionError)


def test_routine_rechecks_enablement_after_waiting_behind_loader(
    manager_context,
    qtbot: pytest.QtBot,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "loader_first.py"
    script_path.write_text(
        """import pathlib
import time
import xarray as xr
from erlab.extensions import loader, routine

@loader(name="Slow Loader", extensions=("txt",))
def slow_loader(path: pathlib.Path) -> xr.DataArray:
    path.write_text("started")
    time.sleep(0.25)
    return xr.DataArray([1.0])

@routine(name="Must Not Run")
def must_not_run(data: xr.DataArray) -> xr.DataArray:
    return data + 1.0
"""
    )
    marker = tmp_path / "loader.txt"
    marker.touch()
    loader_failures: list[BaseException] = []

    with manager_context() as manager:
        catalog, _revision, _created = manager._extensions.catalog.store.add_script(
            script_path
        )
        _validate_and_enable(
            manager._extensions.catalog.store,
            "loader_first",
            expected_record_generation=(
                catalog.extensions["loader_first"].record_generation
            ),
        )
        manager._extensions.catalog.refresh()
        manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(
                xr.DataArray([1.0], dims="x"), _in_manager=True
            ),
            show=False,
        )
        call, _defaults = next(
            iter(manager._extensions.file_loaders((marker,)).values())
        )

        def invoke_loader() -> None:
            try:
                call(marker)
            except BaseException as error:
                loader_failures.append(error)

        loader_thread = threading.Thread(target=invoke_loader)
        loader_thread.start()
        qtbot.wait_until(lambda: marker.read_text() == "started", timeout=2000)

        manager._extensions.execution.queue_routine(
            extension_id="loader_first",
            routine_id="must_not_run",
            parameters={},
            target=0,
        )
        assert manager._extensions.execution.active is None
        assert len(manager._extensions.execution.queued) == 1
        current = manager._extensions.catalog.store.read().extensions["loader_first"]
        manager._extensions.catalog.store.update_record(
            "loader_first",
            expected_record_generation=current.record_generation,
            enabled=False,
        )
        manager._extensions.catalog.refresh()

        qtbot.wait_until(lambda: not loader_thread.is_alive(), timeout=5000)
        loader_thread.join()
        qtbot.wait_until(
            lambda: manager._extensions.execution.active is None,
            timeout=5000,
        )
        assert loader_failures == []
        assert manager.ntools == 1


def test_dispatched_routine_can_be_removed_before_worker_start(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    class _BlockingTask(QtCore.QRunnable):
        def __init__(self) -> None:
            super().__init__()
            self.started = threading.Event()
            self.release = threading.Event()

        def run(self) -> None:
            self.started.set()
            self.release.wait()

    script_path = tmp_path / "queued.py"
    _script(script_path)

    with manager_context() as manager:
        catalog, _revision, _created = manager._extensions.catalog.store.add_script(
            script_path
        )
        _validate_and_enable(
            manager._extensions.catalog.store,
            "queued",
            expected_record_generation=catalog.extensions["queued"].record_generation,
        )
        manager._extensions.catalog.refresh()
        manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(
                xr.DataArray([1.0], dims="x"), _in_manager=True
            ),
            show=False,
        )
        blocking_task = _BlockingTask()
        manager._extensions.execution._pool.start(blocking_task)
        if not blocking_task.started.wait(timeout=2.0):
            blocking_task.release.set()
            pytest.fail("The blocking extension task did not start")
        try:
            job_id = manager._extensions.execution.queue_routine(
                extension_id="queued",
                routine_id="scale",
                parameters={"scale": 2.0},
                target=0,
            )

            assert manager._extensions.execution.active is None
            assert [job.job_id for job in manager._extensions.execution.queued] == [
                job_id
            ]

            manager._extensions.execution.remove_queued(job_id)

            assert manager._extensions.execution.queued == ()
            assert manager.ntools == 1
        finally:
            blocking_task.release.set()


def test_readonly_routine_view_does_not_lock_the_original() -> None:
    data = xr.DataArray(np.arange(3.0), dims="x", coords={"x": np.arange(3.0)})

    readonly = _readonly_array(data)

    assert data.values.flags.writeable
    assert not readonly.values.flags.writeable
    assert not readonly.coords["x"].values.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        readonly.values[0] = 10.0


def test_routine_buffer_protection_preserves_xarray_indexes() -> None:
    source = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("letter", "number"),
        coords={"letter": ["a", "b"], "number": [1, 2]},
    ).stack(sample=("letter", "number"))
    original_writeability = {
        name: coordinate.values.flags.writeable
        for name, coordinate in source.coords.items()
    }

    readonly = _readonly_array(source)

    xr.testing.assert_identical(readonly, source)
    assert type(readonly.xindexes["sample"]) is type(source.xindexes["sample"])
    xr.testing.assert_identical(readonly.sel(letter="a"), source.sel(letter="a"))
    assert all(
        not coordinate.values.flags.writeable for coordinate in readonly.coords.values()
    )
    assert {
        name: coordinate.values.flags.writeable
        for name, coordinate in source.coords.items()
    } == original_writeability

    detached = _detached_routine_output(readonly, source)

    xr.testing.assert_identical(detached, source)
    assert type(detached.xindexes["sample"]) is type(source.xindexes["sample"])
    for name in source.coords:
        assert not np.shares_memory(
            detached.coords[name].values, source.coords[name].values
        )


def test_manager_executes_routine_in_serial_extension_queue(
    manager_context,
    qtbot: pytest.QtBot,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "inspect_input.py"
    script_path.write_text(
        """import threading
import xarray as xr
from erlab.extensions import routine

@routine(name="Inspect Input")
def inspect_input(data: xr.DataArray, offset: float = 0.0) -> xr.DataArray:
    return (data + offset).assign_attrs(
        input_writeable=data.values.flags.writeable,
        execution_thread=threading.get_ident(),
    )
"""
    )
    data = xr.DataArray(np.arange(3.0), dims="x")
    manager_thread = threading.get_ident()

    with manager_context() as manager:
        catalog, revision, _created = manager._extensions.catalog.store.add_script(
            script_path
        )
        _validate_and_enable(
            manager._extensions.catalog.store,
            "inspect_input",
            expected_record_generation=(
                catalog.extensions["inspect_input"].record_generation
            ),
        )
        manager._extensions.catalog.refresh()
        manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(data, _in_manager=True),
            show=False,
        )

        manager._extensions.execution.queue_routine(
            extension_id="inspect_input",
            routine_id="inspect_input",
            parameters={"offset": 2.0},
            target=0,
        )
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        result = manager._get_imagetool_data(1)
        xr.testing.assert_equal(result, data + 2.0)
        assert not result.attrs["input_writeable"]
        assert result.attrs["execution_thread"] != manager_thread
        operation = manager._tool_graph.root_wrappers[1].provenance_spec.operations[-1]
        assert operation.op == "extension_routine"
        assert operation.revision_hash == revision

        replayed = full_data(operation).apply(
            data,
            extension_executor=manager._extensions.execution.run_operation,
        )
        xr.testing.assert_equal(replayed, data + 2.0)
        assert not replayed.attrs["input_writeable"]
        assert replayed.attrs["execution_thread"] != manager_thread

        graph_replayed = replay_script_provenance(
            manager._tool_graph.root_wrappers[1].provenance_spec,
            {"data": data},
            extension_executor=manager._extensions.execution.run_operation,
        )
        xr.testing.assert_equal(graph_replayed, data + 2.0)
        assert not graph_replayed.attrs["input_writeable"]
        assert graph_replayed.attrs["execution_thread"] != manager_thread


def test_active_routine_finishes_and_queued_routine_rechecks_enablement(
    manager_context,
    qtbot: pytest.QtBot,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "slow.py"
    script_path.write_text(
        """import time
import xarray as xr
from erlab.extensions import routine

@routine(name="Slow")
def slow(data: xr.DataArray, amount: float, delay: float = 0.0) -> xr.DataArray:
    time.sleep(delay)
    return data + amount
"""
    )
    data = xr.DataArray(np.arange(3.0), dims="x")

    with manager_context() as manager:
        catalog, _revision, _created = manager._extensions.catalog.store.add_script(
            script_path
        )
        _validate_and_enable(
            manager._extensions.catalog.store,
            "slow",
            expected_record_generation=catalog.extensions["slow"].record_generation,
        )
        manager._extensions.catalog.refresh()
        manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(data, _in_manager=True), show=False
        )
        manager._extensions.execution.queue_routine(
            extension_id="slow",
            routine_id="slow",
            parameters={"amount": 1.0, "delay": 0.2},
            target=0,
        )
        manager._extensions.execution.queue_routine(
            extension_id="slow",
            routine_id="slow",
            parameters={"amount": 2.0, "delay": 0.0},
            target=0,
        )
        qtbot.wait_until(
            lambda: manager._extensions.execution.active is not None,
            timeout=2000,
        )
        current = manager._extensions.catalog.store.read().extensions["slow"]
        manager._extensions.catalog.store.update_record(
            "slow",
            expected_record_generation=current.record_generation,
            enabled=False,
        )
        manager._extensions.catalog.refresh()

        qtbot.wait_until(
            lambda: (
                manager._extensions.execution.active is None
                and not manager._extensions.execution.queued
            ),
            timeout=5000,
        )
        assert manager.ntools == 2
        xr.testing.assert_identical(manager._get_imagetool_data(1), data + 1.0)


def test_removing_queued_replay_releases_its_waiter(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "scale.py"
    _script(script_path)

    with manager_context() as manager:
        catalog, revision, _created = manager._extensions.catalog.store.add_script(
            script_path
        )
        _validate_and_enable(
            manager._extensions.catalog.store,
            "scale",
            expected_record_generation=catalog.extensions["scale"].record_generation,
        )
        manager._extensions.catalog.refresh()
        job = manager._extensions.execution._routine_job(
            extension_id="scale",
            revision_hash=revision,
            routine_id="scale",
            parameters={"scale": 2.0},
            input_data=xr.DataArray([1.0]),
            input_uid="replay",
            input_snapshot="snapshot",
        )
        waiter = _ExtensionRoutineWaiter(QtCore.QEventLoop())
        manager._extensions.execution._routine_waiters[job.job_id] = waiter
        manager._extensions.execution._pending.append(job)

        manager._extensions.execution.remove_queued(job.job_id)

        assert waiter.result is not None
        assert waiter.result.status == "discarded"
        assert manager._extensions.execution.queued == ()


def test_canceling_pending_loader_releases_qt_waiter(
    qtbot: pytest.QtBot,
    tmp_path: pathlib.Path,
) -> None:
    call = _ExtensionLoaderCall(
        manager_session_id="manager",
        catalog_generation=0,
        extension_id="loader",
        extension_name="Loader",
        revision_hash="a" * 64,
        loader_id="load",
        descriptor=erlab.extensions.LoaderDescriptor(
            id="load",
            name="Load",
            category="Lab",
            summary="",
            function_name="load",
        ),
        source_path=tmp_path / "missing.py",
        source_type="script",
        executor=lambda *_args: xr.DataArray([1.0]),
    )
    worker = _ExtensionLoaderWorker(
        call,
        tmp_path / "data.txt",
        {},
        _ExtensionCatalogStore(tmp_path / "catalog"),
        {},
    )

    with qtbot.waitSignal(worker.signals.finished, timeout=1000):
        worker.cancel_if_pending()

    assert worker.done.is_set()
    assert isinstance(worker.error, erlab.extensions.ExtensionExecutionError)


def test_validation_runs_on_extension_thread_and_contains_system_exit(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    marker_path = tmp_path / "validation-thread.txt"
    script_path = tmp_path / "validate_thread.py"
    script_path.write_text(
        f"""import pathlib
import threading
import xarray as xr
from erlab.extensions import routine

pathlib.Path({str(marker_path)!r}).write_text(str(threading.get_ident()))

@routine(name="Validate Thread")
def validate_thread(data: xr.DataArray) -> xr.DataArray:
    return data
"""
    )
    manager_thread = threading.get_ident()

    with manager_context() as manager:
        catalog, _revision, _created = manager._extensions.catalog.store.add_script(
            script_path
        )
        result = manager._extensions.execution.validate_and_enable(
            "validate_thread",
            expected_record_generation=(
                catalog.extensions["validate_thread"].record_generation
            ),
        )

        assert result.extensions["validate_thread"].enabled
        assert int(marker_path.read_text()) != manager_thread

        failing_path = tmp_path / "stops.py"
        failing_path.write_text("raise SystemExit('extension requested exit')\n")
        catalog, _revision, _created = manager._extensions.catalog.store.add_script(
            failing_path
        )
        with pytest.raises(
            erlab.extensions.ExtensionExecutionError, match="SystemExit"
        ):
            manager._extensions.execution.validate_and_enable(
                "stops",
                expected_record_generation=catalog.extensions[
                    "stops"
                ].record_generation,
            )

        failed = manager._extensions.catalog.store.read().extensions["stops"]
        assert not failed.enabled
        assert "SystemExit: extension requested exit" in (
            failed.revisions[failed.current_revision].import_error or ""
        )


def test_manager_shutdown_releases_only_its_extension_modules(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "owned_module.py"
    script_path.write_text(_script(script_path).decode())

    with manager_context() as manager:
        catalog, _revision, _created = manager._extensions.catalog.store.add_script(
            script_path
        )
        manager._extensions.execution.validate_and_enable(
            "owned_module",
            expected_record_generation=(
                catalog.extensions["owned_module"].record_generation
            ),
        )
        session_id = manager._manager_record.internal_id
        prefix = f"_erlab_extension_{session_id}_"
        owned_names = {name for name in sys.modules if name.startswith(prefix)}
        unrelated_name = f"_erlab_extension_{session_id}0_unrelated"
        sys.modules[unrelated_name] = types.ModuleType(unrelated_name)

        assert owned_names
        manager._extensions.close()

        assert all(name not in sys.modules for name in owned_names)
        assert unrelated_name in sys.modules
        sys.modules.pop(unrelated_name)


def test_canceling_pending_validation_releases_qt_waiter(
    qtbot: pytest.QtBot,
    tmp_path: pathlib.Path,
) -> None:
    worker = _ExtensionValidationWorker(
        "extension",
        "a" * 64,
        1,
        manager_session_id="manager",
        catalog_store=_ExtensionCatalogStore(tmp_path / "catalog"),
        script_modules={},
    )

    with qtbot.waitSignal(worker.signals.finished, timeout=1000):
        worker.cancel_if_pending()

    assert worker.done.is_set()
    assert isinstance(worker.error, erlab.extensions.ExtensionExecutionError)


def test_routine_output_detaches_only_shared_numpy_buffers() -> None:
    source = xr.DataArray(
        np.arange(3.0),
        dims="x",
        coords={"x": np.array([1.0, 2.0, 3.0])},
        attrs={"source": "original"},
    )
    readonly = _readonly_array(source)
    detached = _detached_routine_output(readonly, source)

    assert detached.values.flags.writeable
    assert not np.shares_memory(detached.values, source.values)
    assert not np.shares_memory(detached.coords["x"].values, source.coords["x"].values)
    detached.values[0] = 100.0
    assert source.values[0] == 0.0

    computed = xr.DataArray(source.values + 1.0, dims="x")
    unchanged = _detached_routine_output(computed, source)
    assert unchanged is computed


def test_stale_routine_result_is_not_inserted(
    manager_context,
    qtbot: pytest.QtBot,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "slow.py"
    script_path.write_text(
        """import time
import xarray as xr
from erlab.extensions import routine

@routine(name="Slow")
def slow(data: xr.DataArray, delay: float = 0.1) -> xr.DataArray:
    time.sleep(delay)
    return data + 1.0
"""
    )
    data = xr.DataArray(np.arange(3.0), dims="x")

    with manager_context() as manager:
        catalog, _revision, _created = manager._extensions.catalog.store.add_script(
            script_path
        )
        _validate_and_enable(
            manager._extensions.catalog.store,
            "slow",
            expected_record_generation=catalog.extensions["slow"].record_generation,
        )
        manager._extensions.catalog.refresh()
        manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(data, _in_manager=True), show=False
        )
        manager._extensions.execution.queue_routine(
            extension_id="slow",
            routine_id="slow",
            parameters={"delay": 0.15},
            target=0,
        )
        qtbot.wait_until(
            lambda: manager._extensions.execution.active is not None,
            timeout=2000,
        )
        manager._tool_graph.root_wrappers[0]._advance_snapshot_token()

        qtbot.wait_until(
            lambda: manager._extensions.execution.active is None,
            timeout=5000,
        )
        assert manager.ntools == 1


def test_routine_with_unsupported_output_shape_reports_one_failure(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    qtbot: pytest.QtBot,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "invalid_shape.py"
    script_path.write_text(
        """import numpy as np
import xarray as xr
from erlab.extensions import routine

@routine(name="Invalid shape")
def invalid_shape(data: xr.DataArray) -> xr.DataArray:
    return xr.DataArray(np.zeros((2, 2, 2, 2, 2)), dims=("a", "b", "c", "d", "e"))
"""
    )
    shown: list[None] = []
    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        lambda *_args, **_kwargs: shown.append(None),
    )

    with manager_context() as manager:
        catalog, _revision, _created = manager._extensions.catalog.store.add_script(
            script_path
        )
        _validate_and_enable(
            manager._extensions.catalog.store,
            "invalid_shape",
            expected_record_generation=(
                catalog.extensions["invalid_shape"].record_generation
            ),
        )
        manager._extensions.catalog.refresh()
        manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(
                xr.DataArray(np.ones((2, 2)), dims=("x", "y")),
                _in_manager=True,
            ),
            show=False,
        )

        manager._extensions.execution.queue_routine(
            extension_id="invalid_shape",
            routine_id="invalid_shape",
            parameters={},
            target=0,
        )
        qtbot.wait_until(
            lambda: (
                manager._extensions.execution.active is None
                and not manager._extensions.execution.queued
            ),
            timeout=5000,
        )

        assert manager.ntools == 1
        assert shown == [None]
