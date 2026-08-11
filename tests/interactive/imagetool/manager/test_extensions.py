from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys
import threading
import types
import typing
import uuid

import numpy as np
import pytest
import xarray as xr
from qtpy import QtCore, QtWidgets

import erlab
import erlab.extensions._entry_points as extension_entry_points
import erlab.interactive.imagetool.manager as imagetool_manager
import erlab.interactive.imagetool.manager._base as manager_base
import erlab.interactive.imagetool.manager._extensions._catalog as extension_catalog
import erlab.interactive.imagetool.manager._extensions._dialogs as extension_dialogs
import erlab.interactive.imagetool.manager._extensions._execution as extension_execution
import erlab.interactive.imagetool.viewer as imagetool_viewer
from erlab.interactive.imagetool._load_source import _resolve_load_func
from erlab.interactive.imagetool._provenance._execution import (
    can_reload_without_trust,
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


def _loader_script(
    path: pathlib.Path,
    *,
    name: str,
    extensions: tuple[str, ...],
) -> bytes:
    source = f"""from pathlib import Path
import xarray as xr
from erlab.extensions import loader

@loader(name={name!r}, extensions={extensions!r})
def load_data(path: Path) -> xr.DataArray:
    return xr.DataArray([float(path.read_text())])
""".encode()
    path.write_bytes(source)
    return source


def _generated_external_loader(
    path: pathlib.Path, *, scale: float = 1.0
) -> xr.DataArray:
    return xr.DataArray([float(path.read_text()) * scale])


def _validate_and_enable(
    store: _ExtensionCatalogStore,
    extension_id: str,
    *,
    expected_record_generation: int,
) -> _ExtensionCatalogModel:
    record = store.read().extensions[extension_id]
    manager_session_id = f"test-manager-{uuid.uuid4().hex}"
    try:
        return _validate_extension_revision(
            store,
            extension_id,
            revision_hash=record.current_revision,
            expected_record_generation=expected_record_generation,
            manager_session_id=manager_session_id,
            script_modules={},
        )
    finally:
        extension_execution._remove_manager_modules(manager_session_id)


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


def test_parameter_dialog_accepts_a_required_empty_string(qtbot) -> None:
    descriptor = erlab.extensions.RoutineDescriptor(
        id="required-string",
        name="Required string",
        category="Lab",
        summary="",
        function_name="required_string",
        parameters=(
            erlab.extensions.ParameterDescriptor(
                id="label",
                kind=erlab.extensions.ParameterKind.STRING,
                required=True,
            ),
        ),
    )
    dialog = _ExtensionParameterDialog(descriptor, None)
    qtbot.addWidget(dialog)

    assert dialog.parameters == {"label": ""}


def test_parameter_dialog_preserves_exact_literal_default_type(qtbot) -> None:
    descriptor = erlab.extensions.RoutineDescriptor(
        id="literal-types",
        name="Literal types",
        category="Lab",
        summary="",
        function_name="literal_types",
        parameters=(
            erlab.extensions.ParameterDescriptor(
                id="choice",
                kind=erlab.extensions.ParameterKind.LITERAL,
                required=False,
                default=1,
                choices=(True, 1),
            ),
        ),
    )
    dialog = _ExtensionParameterDialog(descriptor, None)
    qtbot.addWidget(dialog)

    assert dialog.parameters == {"choice": 1}
    assert type(dialog.parameters["choice"]) is int


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


def test_parameter_dialog_preserves_python_numeric_values(qtbot) -> None:
    descriptor = erlab.extensions.RoutineDescriptor(
        id="numeric-values",
        name="Numeric values",
        category="Lab",
        summary="",
        function_name="numeric_values",
        parameters=(
            erlab.extensions.ParameterDescriptor(
                id="large_integer",
                kind=erlab.extensions.ParameterKind.INTEGER,
                required=False,
                default=10**12,
            ),
            erlab.extensions.ParameterDescriptor(
                id="small_number",
                kind=erlab.extensions.ParameterKind.NUMBER,
                required=False,
                default=1e-15,
            ),
        ),
    )
    dialog = _ExtensionParameterDialog(descriptor, None)
    qtbot.addWidget(dialog)

    assert dialog.parameters == {
        "large_integer": 10**12,
        "small_number": 1e-15,
    }

    number_editor = dialog._editors["small_number"]
    assert isinstance(number_editor, QtWidgets.QLineEdit)
    number_editor.setText("nan")
    with pytest.raises(ValueError, match="must be finite"):
        _parameters = dialog.parameters


def test_source_review_dialog_reads_source_and_metadata(
    qtbot: pytest.QtBot,
    tmp_path: pathlib.Path,
) -> None:
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    with pytest.raises(ValueError, match="path or source_text"):
        extension_dialogs._SourceReviewDialog(None, parent)

    source = tmp_path / "extension.py"
    source.write_text("VALUE = 1\n")
    dialog = extension_dialogs._SourceReviewDialog(source, parent)
    qtbot.addWidget(dialog)
    source_editor = dialog.findChild(
        QtWidgets.QPlainTextEdit, "manager_extension_source_review"
    )
    if source_editor is None:
        raise RuntimeError("Source review editor was not created")
    assert source_editor.toPlainText() == "VALUE = 1\n"
    dialog.author_edit.setText("  Lab User  ")
    dialog.contact_edit.setText(" lab@example.org ")
    dialog.project_url_edit.setText(" https://example.org/lab ")
    dialog.change_summary_edit.setText(" Initial revision ")
    dialog.changelog_edit.setPlainText(" Added routine. ")
    assert dialog.metadata == _ExtensionMetadata(
        author="Lab User",
        contact="lab@example.org",
        project_url="https://example.org/lab",
        change_summary="Initial revision",
        changelog="Added routine.",
    )
    assert dialog.remember_approval

    session_dialog = extension_dialogs._SourceReviewDialog(
        None,
        parent,
        source_text="VALUE = 2\n",
        choose_approval_scope=True,
    )
    qtbot.addWidget(session_dialog)
    assert not session_dialog.remember_approval


def test_parameter_dialog_builds_all_editor_types_and_validates_path(
    qtbot: pytest.QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = erlab.extensions.RoutineDescriptor(
        id="configure",
        name="Configure",
        category="Lab",
        summary="Configure the routine.",
        function_name="configure",
        parameters=(
            erlab.extensions.ParameterDescriptor(
                id="optional_flag",
                kind=erlab.extensions.ParameterKind.BOOLEAN,
                required=False,
                optional=True,
                default=True,
            ),
            erlab.extensions.ParameterDescriptor(
                id="flag",
                kind=erlab.extensions.ParameterKind.BOOLEAN,
                required=False,
                default=False,
            ),
            erlab.extensions.ParameterDescriptor(
                id="choice",
                kind=erlab.extensions.ParameterKind.LITERAL,
                required=False,
                optional=True,
                default=None,
                choices=("first", "second"),
            ),
            erlab.extensions.ParameterDescriptor(
                id="path",
                kind=erlab.extensions.ParameterKind.PATH,
                required=True,
            ),
        ),
    )
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    dialog = _ExtensionParameterDialog(descriptor, parent)
    qtbot.addWidget(dialog)

    assert dialog.findChild(QtWidgets.QLabel) is not None
    optional_flag = dialog._editors["optional_flag"]
    assert isinstance(optional_flag, QtWidgets.QComboBox)
    assert optional_flag.currentData() is True
    flag = dialog._editors["flag"]
    assert isinstance(flag, QtWidgets.QCheckBox)
    flag.setChecked(True)
    choice = dialog._editors["choice"]
    assert isinstance(choice, QtWidgets.QComboBox)
    assert choice.currentData() is None
    path = dialog._editors["path"]
    assert isinstance(path, QtWidgets.QLineEdit)
    assert path.placeholderText()

    warnings: list[str] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda _parent, _title, text: warnings.append(text),
    )
    dialog.accept()
    assert warnings == ["'path' requires a value"]
    assert dialog.result() != QtWidgets.QDialog.DialogCode.Accepted

    path.setText("data.txt")
    assert dialog.parameters == {
        "optional_flag": True,
        "flag": True,
        "choice": None,
        "path": "data.txt",
    }
    dialog.accept()
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted


def test_routine_selection_dialog_reports_current_selection(
    qtbot: pytest.QtBot,
) -> None:
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    empty = extension_dialogs._RoutineSelectionDialog((), parent)
    qtbot.addWidget(empty)
    assert empty.selection is None

    descriptor = erlab.extensions.RoutineDescriptor(
        id="calculate",
        name="Calculate",
        category="Lab",
        summary="",
        function_name="calculate",
    )
    dialog = extension_dialogs._RoutineSelectionDialog(
        (("lab", "Lab extension", descriptor),), parent
    )
    qtbot.addWidget(dialog)
    assert dialog.selection == ("lab", "calculate")


def test_manage_and_metadata_dialogs_preserve_selected_extension(
    qtbot: pytest.QtBot,
) -> None:
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    revision_hash = "a" * 64
    revision = _ExtensionRevision(
        source_hash=revision_hash,
        object_name=f"{revision_hash}.py",
        source_path="source.py",
        created_at="2026-01-01T00:00:00+00:00",
        approved=True,
    )
    record = _ExtensionRecord(
        id="lab",
        name="Lab",
        current_revision=revision_hash,
        revisions={revision_hash: revision},
        metadata=_ExtensionMetadata(author="Lab User"),
    )
    dialog = extension_dialogs._ManageExtensionsDialog(parent)
    qtbot.addWidget(dialog)
    dialog.set_catalog(_ExtensionCatalogModel(extensions={"lab": record}))
    top = dialog.tree.topLevelItem(0)
    revision_item = top.child(0)
    dialog.tree.setCurrentItem(revision_item)
    assert dialog.selected_extension_id == "lab"

    actions: list[tuple[str, str]] = []

    def action_slot(action: str, extension: str) -> None:
        actions.append((action, extension))

    dialog.action_requested.connect(action_slot)
    try:
        dialog._emit_action("metadata")
        assert actions == [("metadata", "lab")]
        dialog.tree.setCurrentItem(None)
        dialog._emit_action("remove")
        assert actions == [("metadata", "lab")]
    finally:
        dialog.action_requested.disconnect(action_slot)

    metadata_dialog = extension_dialogs._MetadataDialog(record.metadata, parent)
    qtbot.addWidget(metadata_dialog)
    author = metadata_dialog._edits["author"]
    if not isinstance(author, QtWidgets.QLineEdit):
        raise TypeError("Author metadata editor must be a line edit")
    author.setText(" Updated User ")
    changelog = metadata_dialog._edits["changelog"]
    if not isinstance(changelog, QtWidgets.QPlainTextEdit):
        raise TypeError("Changelog metadata editor must be a plain-text edit")
    changelog.setPlainText(" Updated notes. ")
    assert metadata_dialog.metadata.author == "Updated User"
    assert metadata_dialog.metadata.changelog == "Updated notes."


def test_workspace_requirements_dialog_approves_only_eligible_selection(
    qtbot: pytest.QtBot,
) -> None:
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    requirement = _WorkspaceExtensionRequirement(
        extension_id="lab",
        capability_id="calculate",
        capability_kind="routine",
        revision_hash="a" * 64,
        extension_api_version=1,
        source_type="script",
    )
    resolved = _ResolvedWorkspaceRequirement(
        requirement=requirement,
        state="approval-required",
    )
    environment_resolved = resolved.model_copy(
        update={
            "requirement": requirement.model_copy(
                update={"source_type": "environment-package"}
            )
        }
    )
    dialog = extension_dialogs._WorkspaceRequirementsDialog(
        (environment_resolved, resolved),
        parent,
        approvable={("lab", "a" * 64, "script")},
    )
    qtbot.addWidget(dialog)
    approvals: list[tuple[str, str]] = []

    def approval_slot(extension: str, revision: str) -> None:
        approvals.append((extension, revision))

    dialog.approve_requested.connect(approval_slot)
    try:
        dialog._approve_selected()
        assert approvals == []
        dialog.tree.setCurrentItem(dialog.tree.topLevelItem(0))
        assert not dialog._approve_button.isEnabled()
        dialog._approve_selected()
        assert approvals == []
        dialog.tree.setCurrentItem(dialog.tree.topLevelItem(1))
        assert dialog._approve_button.isEnabled()
        dialog._approve_selected()
        assert approvals == [("lab", "a" * 64)]

        for state in ("missing", "hash-mismatch", "import-failed"):
            dialog.set_requirements((resolved.model_copy(update={"state": state}),))
            assert dialog._approve_button.isEnabled()
            dialog._approve_selected()

        assert approvals == [("lab", "a" * 64)] * 4
        dialog.set_requirements((resolved.model_copy(update={"state": "ready"}),))
        assert dialog.tree.currentItem() is dialog.tree.topLevelItem(0)
        assert not dialog._approve_button.isEnabled()
        dialog._approve_selected()
        assert approvals == [("lab", "a" * 64)] * 4
    finally:
        dialog.approve_requested.disconnect(approval_slot)


def test_packaged_controller_hides_environment_capabilities(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision_hash = "a" * 64
    revision = _ExtensionRevision(
        source_hash=revision_hash,
        object_name="lab_package:extension",
        created_at="2026-01-01T00:00:00+00:00",
        approved=True,
        routines=(
            erlab.extensions.RoutineDescriptor(
                id="calculate",
                name="Calculate",
                category="Lab",
                summary="",
                function_name="calculate",
            ),
        ),
        loaders=(
            erlab.extensions.LoaderDescriptor(
                id="load_data",
                name="Load Data",
                category="Lab",
                summary="",
                function_name="load_data",
            ),
        ),
        entry_point_group="erlab.extensions",
        entry_point_name="lab",
        entry_point_value="lab_package:extension",
    )
    record = _ExtensionRecord(
        id="lab",
        name="Lab",
        source_type="environment-package",
        enabled=True,
        current_revision=revision_hash,
        revisions={revision_hash: revision},
    )

    with manager_context() as manager:
        controller = manager._extensions
        controller.catalog.model = _ExtensionCatalogModel(extensions={"lab": record})
        monkeypatch.setattr(erlab.utils.misc, "_IS_PACKAGED", True)
        assert controller._enabled_routines() == ()
        assert controller.file_loaders() == {}
        controller._sync_explorer_loaders()
        assert controller.explorer_loaders == {}


def test_controller_filters_loader_paths_and_rejects_duplicate_filters(
    manager_context,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "loader.py"
    _loader_script(source_path, name="Lab Data", extensions=(".dat",))
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    descriptor = erlab.extensions.LoaderDescriptor(
        id="load_data",
        name="Lab Data",
        category="Lab",
        summary="",
        function_name="load_data",
        extensions=(".dat",),
    )
    revision = _ExtensionRevision(
        source_hash=source_hash,
        object_name=f"{source_hash}.py",
        source_path=os.fspath(source_path),
        created_at="2026-01-01T00:00:00+00:00",
        approved=True,
        loaders=(descriptor,),
    )
    records = {
        extension_id: _ExtensionRecord(
            id=extension_id,
            name=extension_id.title(),
            enabled=True,
            current_revision=source_hash,
            revisions={source_hash: revision},
        )
        for extension_id in ("first", "second")
    }

    with manager_context() as manager:
        controller = manager._extensions
        controller.catalog.model = _ExtensionCatalogModel(
            extensions={"first": records["first"]}
        )
        monkeypatch.setattr(
            controller.catalog.store,
            "revision_available",
            lambda *_args: True,
        )
        monkeypatch.setattr(
            controller.catalog.store,
            "source_path",
            lambda *_args: source_path,
        )
        assert controller.file_loaders(tmp_path / "value.txt") == {}
        assert tuple(controller.file_loaders(tmp_path / "value.dat")) == (
            "Lab Data (*.dat)",
        )

        controller.catalog.model = _ExtensionCatalogModel(extensions=records)
        with pytest.raises(
            ValueError, match="Conflicting extension file dialog filter"
        ):
            controller.file_loaders()

        environment_revision = revision.model_copy(
            update={
                "object_name": "lab_package:Loader",
                "entry_point_group": "erlab.io.loaders",
                "entry_point_name": "lab",
                "entry_point_value": "lab_package:Loader",
                "loader_dialog_methods": (),
            }
        )
        environment_record = records["first"].model_copy(
            update={
                "source_type": "environment-package",
                "revisions": {source_hash: environment_revision},
            }
        )
        controller.catalog.model = _ExtensionCatalogModel(
            extensions={"first": environment_record}
        )
        assert controller.file_loaders() == {}


def test_controller_menu_selection_and_routine_queue_paths(
    manager_context,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = tmp_path / "scale.py"
    _script(script_path)
    information_calls: list[None] = []
    critical_calls: list[None] = []

    with manager_context() as manager:
        controller = manager._extensions
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "information",
            lambda *_args, **_kwargs: information_calls.append(None),
        )
        controller.select_routine()
        assert information_calls == [None]

        catalog, _revision_hash, _created = controller.catalog.store.add_script(
            script_path
        )
        catalog = _validate_and_enable(
            controller.catalog.store,
            "scale",
            expected_record_generation=catalog.extensions["scale"].record_generation,
        )
        catalog = controller.catalog.store.update_record(
            "scale",
            expected_record_generation=catalog.extensions["scale"].record_generation,
            favorite=True,
        )
        controller.catalog.refresh()
        controller._recent.append(("scale", "scale"))
        controller._populate_menu()
        action_data = {
            action.data()
            for action in controller.menu.actions()
            if action.data() is not None
        }
        for action in controller.menu.actions():
            submenu = action.menu()
            if submenu is not None:
                action_data.update(
                    child.data()
                    for child in submenu.actions()
                    if child.data() is not None
                )
        assert ("scale", "scale") in action_data

        selected: list[tuple[str, str]] = []
        monkeypatch.setattr(
            controller,
            "run_routine",
            lambda extension_id, routine_id: selected.append(
                (extension_id, routine_id)
            ),
        )

        class AcceptedSelectionDialog:
            selection = ("scale", "scale")

            def __init__(self, *_args, **_kwargs) -> None:
                return None

            @staticmethod
            def exec() -> int:
                return 1

        monkeypatch.setattr(
            extension_controller,
            "_RoutineSelectionDialog",
            AcceptedSelectionDialog,
        )
        controller.select_routine()
        assert selected == [("scale", "scale")]

        monkeypatch.undo()
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "information",
            lambda *_args, **_kwargs: information_calls.append(None),
        )
        monkeypatch.setattr(
            erlab.interactive.utils.MessageDialog,
            "critical",
            lambda *_args, **_kwargs: critical_calls.append(None),
        )
        monkeypatch.setattr(manager, "_selected_imagetool_targets", lambda: ())
        controller.run_routine("scale", "scale")
        assert len(information_calls) == 2

        monkeypatch.setattr(manager, "_selected_imagetool_targets", lambda: (0,))
        controller.run_routine("missing", "scale")
        controller.run_routine("scale", "missing")

        class RejectedParameterDialog:
            def __init__(self, *_args, **_kwargs) -> None:
                return None

            @staticmethod
            def exec() -> int:
                return 0

        monkeypatch.setattr(
            extension_controller,
            "_ExtensionParameterDialog",
            RejectedParameterDialog,
        )
        controller.run_routine("scale", "scale")

        class AcceptedParameterDialog(RejectedParameterDialog):
            parameters: typing.ClassVar[dict[str, float]] = {"scale": 3.0}

            @staticmethod
            def exec() -> int:
                return 1

        monkeypatch.setattr(
            extension_controller,
            "_ExtensionParameterDialog",
            AcceptedParameterDialog,
        )
        monkeypatch.setattr(
            controller.execution,
            "queue_routine",
            lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("queue failed")),
        )
        controller.run_routine("scale", "scale")
        assert critical_calls == [None]

        queued: list[dict[str, typing.Any]] = []
        monkeypatch.setattr(
            controller.execution,
            "queue_routine",
            lambda **kwargs: queued.append(kwargs) or "job",
        )
        controller.run_routine("scale", "scale")
        controller.run_routine("scale", "scale")
        assert len(queued) == 2
        assert tuple(controller._recent).count(("scale", "scale")) == 1

        assert controller.loader_by_name("missing") is None
        controller.show_manager()
        assert controller._manage_dialog.isVisible()
        controller._manage_dialog.hide()


def test_controller_replay_loader_rejects_incomplete_and_unapproved_calls(
    manager_context,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_path = tmp_path / "data.dat"
    data_path.write_text("1")

    def load_source(replay_call: FileReplayCall | None) -> FileLoadSource:
        return FileLoadSource(
            path=os.fspath(data_path),
            loader_label="Lab Data",
            loader_text="lab:load_data",
            kwargs_text="",
            replay_call=replay_call,
        )

    with manager_context() as manager:
        controller = manager._extensions
        with pytest.raises(
            erlab.extensions.ExtensionExecutionError, match="metadata is incomplete"
        ):
            controller.replay_loader(load_source(None))

        missing_call = FileReplayCall(
            kind="extension_loader",
            target="missing",
            revision="a" * 64,
            capability_id="load_data",
            extension_source_type="script",
            selection=FileDataSelection(kind="dataarray"),
        )
        with pytest.raises(
            erlab.extensions.ExtensionExecutionError, match="revision is not available"
        ):
            controller.replay_loader(load_source(missing_call))

        monkeypatch.setattr(
            controller.execution,
            "session_capability_status",
            lambda *_args: "ready",
        )
        alternate_session_call = missing_call.model_copy(
            update={"loader_method": "preview"}
        )
        with pytest.raises(
            erlab.extensions.ExtensionExecutionError,
            match="do not provide alternate methods",
        ):
            controller.replay_loader(load_source(alternate_session_call))
        monkeypatch.undo()

        script_path = tmp_path / "loader.py"
        _loader_script(script_path, name="Lab Data", extensions=(".dat",))
        catalog, revision_hash, _created = controller.catalog.store.add_script(
            script_path
        )
        catalog = _validate_and_enable(
            controller.catalog.store,
            "loader",
            expected_record_generation=catalog.extensions["loader"].record_generation,
        )
        record = catalog.extensions["loader"]
        script_call = FileReplayCall(
            kind="extension_loader",
            target="loader",
            revision=revision_hash,
            capability_id="load_data",
            extension_source_type="script",
            selection=FileDataSelection(kind="dataarray"),
        )
        with pytest.raises(
            erlab.extensions.ExtensionExecutionError,
            match="revision is not available",
        ):
            controller.replay_loader(
                load_source(
                    script_call.model_copy(
                        update={"extension_source_type": "environment-package"}
                    )
                )
            )
        missing_descriptor = record.revisions[revision_hash].model_copy(
            update={"loaders": ()}
        )
        missing_descriptor_record = record.model_copy(
            update={"revisions": {revision_hash: missing_descriptor}}
        )
        monkeypatch.setattr(
            controller.catalog.store,
            "read",
            lambda: catalog.model_copy(
                update={"extensions": {"loader": missing_descriptor_record}}
            ),
        )
        with pytest.raises(
            erlab.extensions.ExtensionExecutionError, match="is not available"
        ):
            controller.replay_loader(load_source(script_call))
        monkeypatch.undo()

        with pytest.raises(
            erlab.extensions.ExtensionExecutionError,
            match="do not provide alternate methods",
        ):
            controller.replay_loader(
                load_source(script_call.model_copy(update={"loader_method": "preview"}))
            )

        environment_revision = _ExtensionRevision(
            source_hash="b" * 64,
            object_name="lab_package:Loader",
            created_at="2026-01-01T00:00:00+00:00",
            approved=True,
            loaders=(record.revisions[revision_hash].loaders[0],),
            entry_point_group="erlab.io.loaders",
            entry_point_name="lab",
            entry_point_value="lab_package:Loader",
            loader_dialog_methods=(
                extension_catalog._EnvironmentLoaderMethod(
                    name_filter="Lab Data (*.dat)", method=None
                ),
            ),
        )
        environment_record = _ExtensionRecord(
            id="environment-loader",
            name="Environment loader",
            source_type="environment-package",
            enabled=True,
            current_revision="b" * 64,
            revisions={"b" * 64: environment_revision},
        )
        environment_catalog = _ExtensionCatalogModel(
            extensions={"environment-loader": environment_record}
        )
        monkeypatch.setattr(
            controller.catalog.store, "read", lambda: environment_catalog
        )
        monkeypatch.setattr(
            controller.catalog.store,
            "capability_status",
            lambda *_args: "ready",
        )
        environment_call = FileReplayCall(
            kind="extension_loader",
            target="environment-loader",
            revision="b" * 64,
            capability_id="load_data",
            extension_source_type="environment-package",
            loader_method="unapproved",
            selection=FileDataSelection(kind="dataarray"),
        )
        with pytest.raises(
            erlab.extensions.ExtensionExecutionError,
            match="was not approved",
        ):
            controller.replay_loader(load_source(environment_call))


def test_controller_capability_status_prefers_session_ready_state(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with manager_context() as manager:
        controller = manager._extensions
        monkeypatch.setattr(
            controller.execution,
            "session_capability_status",
            lambda *_args: "ready",
        )
        assert (
            controller.capability_status("missing", "a" * 64, "routine", "calculate")
            == "ready"
        )
        assert (
            controller.capability_status(
                "missing",
                "a" * 64,
                "routine",
                "calculate",
                "environment-package",
            )
            == "missing-revision"
        )
        monkeypatch.setattr(
            controller.execution,
            "session_capability_status",
            lambda *_args: None,
        )
        assert (
            controller.capability_status("missing", "a" * 64, "routine", "calculate")
            == "missing-revision"
        )


def test_catalog_source_states_distinguish_all_script_source_failures(
    manager_context,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    object_directory = tmp_path / "objects"
    object_directory.mkdir()
    paths: dict[str, pathlib.Path] = {}
    records: dict[str, _ExtensionRecord] = {}
    unreadable_paths: set[pathlib.Path] = set()

    def add_script_record(
        extension_id: str,
        *,
        stored: bytes | None,
        original: bytes | None,
        unreadable: typing.Literal["stored", "original"] | None = None,
    ) -> None:
        source = f"source:{extension_id}".encode()
        revision_hash = hashlib.sha256(source).hexdigest()
        object_path = object_directory / f"{revision_hash}.py"
        if stored is not None:
            object_path.write_bytes(source if stored == b"source" else stored)
        original_path = tmp_path / "original" / f"{extension_id}.py"
        if original is not None:
            original_path.parent.mkdir(exist_ok=True)
            original_path.write_bytes(source if original == b"source" else original)
        revision = _ExtensionRevision(
            source_hash=revision_hash,
            object_name=object_path.name,
            source_path=(
                None if extension_id == "embedded" else os.fspath(original_path)
            ),
            created_at="2026-01-01T00:00:00+00:00",
        )
        records[extension_id] = _ExtensionRecord(
            id=extension_id,
            name=extension_id.title(),
            current_revision=revision_hash,
            revisions={revision_hash: revision},
        )
        paths[extension_id] = object_path
        if unreadable is not None:
            unreadable_paths.add(
                object_path if unreadable == "stored" else original_path
            )

    add_script_record("missing-stored", stored=None, original=b"source")
    add_script_record(
        "unreadable-stored",
        stored=b"source",
        original=b"source",
        unreadable="stored",
    )
    add_script_record("mismatch", stored=b"different", original=b"source")
    add_script_record("embedded", stored=b"source", original=None)
    add_script_record("missing-original", stored=b"source", original=None)
    add_script_record(
        "unreadable-original",
        stored=b"source",
        original=b"source",
        unreadable="original",
    )
    add_script_record("unchanged", stored=b"source", original=b"source")
    add_script_record("changed", stored=b"source", original=b"changed")

    environment_revision = _ExtensionRevision(
        source_hash="a" * 64,
        object_name="lab_package:extension",
        created_at="2026-01-01T00:00:00+00:00",
        entry_point_group="erlab.extensions",
        entry_point_name="lab",
        entry_point_value="lab_package:extension",
        editable=True,
    )
    for extension_id in ("environment-ready", "environment-missing"):
        records[extension_id] = _ExtensionRecord(
            id=extension_id,
            name=extension_id.title(),
            source_type="environment-package",
            current_revision="a" * 64,
            revisions={"a" * 64: environment_revision},
        )

    original_read_bytes = pathlib.Path.read_bytes

    def read_bytes(path: pathlib.Path) -> bytes:
        if path in unreadable_paths:
            raise OSError("unreadable")
        return original_read_bytes(path)

    with manager_context() as manager:
        controller = manager._extensions
        controller.catalog.model = _ExtensionCatalogModel(extensions=records)

        def source_path(extension_id: str, _revision: str) -> pathlib.Path:
            path = paths.get(extension_id)
            if path is None or not path.exists():
                raise FileNotFoundError(extension_id)
            return path

        monkeypatch.setattr(controller.catalog.store, "source_path", source_path)
        monkeypatch.setattr(
            controller.catalog.store,
            "revision_available",
            lambda record, _revision: record.id == "environment-ready",
        )
        monkeypatch.setattr(pathlib.Path, "read_bytes", read_bytes)
        states = controller._catalog_source_states()

    state_by_extension = {
        extension_id: state for (extension_id, _revision), state in states.items()
    }
    assert state_by_extension == {
        "missing-stored": "Stored source missing",
        "unreadable-stored": "Stored source unreadable",
        "mismatch": "Stored source hash mismatch",
        "embedded": "Stored embedded source",
        "missing-original": "Stored source; original missing",
        "unreadable-original": "Stored source; original unreadable",
        "unchanged": "Stored source; original unchanged",
        "changed": "Stored source; original changed",
        "environment-ready": "Editable environment package",
        "environment-missing": "Environment package unavailable",
    }


def test_manage_actions_dispatch_updates_and_report_failures(
    manager_context,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = tmp_path / "scale.py"
    _script(script_path)
    updates: list[dict[str, typing.Any]] = []
    validations: list[tuple[str, int]] = []
    warnings: list[None] = []
    critical: list[None] = []

    with manager_context() as manager:
        controller = manager._extensions
        catalog, revision_hash, _created = controller.catalog.store.add_script(
            script_path
        )
        catalog = _validate_and_enable(
            controller.catalog.store,
            "scale",
            expected_record_generation=catalog.extensions["scale"].record_generation,
        )
        controller.catalog.refresh()
        enabled_record = controller.catalog.model.extensions["scale"]

        def update_record(
            extension_id: str, **values: typing.Any
        ) -> _ExtensionCatalogModel:
            updates.append({"extension_id": extension_id, **values})
            return controller.catalog.model

        monkeypatch.setattr(controller.catalog.store, "update_record", update_record)
        monkeypatch.setattr(
            controller.execution,
            "validate_and_enable",
            lambda extension_id, *, expected_record_generation: validations.append(
                (extension_id, expected_record_generation)
            ),
        )
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "warning",
            lambda *_args, **_kwargs: warnings.append(None),
        )
        monkeypatch.setattr(
            erlab.interactive.utils.MessageDialog,
            "critical",
            lambda *_args, **_kwargs: critical.append(None),
        )

        controller._manage_action("toggle", "scale")
        controller._manage_action("favorite", "scale")
        controller._manage_action("remove", "scale")
        assert updates[0]["enabled"] is False
        assert updates[1]["favorite"] is True
        assert updates[2]["removed"] is True

        disabled_record = enabled_record.model_copy(update={"enabled": False})
        controller.catalog.model = catalog.model_copy(
            update={"extensions": {"scale": disabled_record}}
        )
        controller._manage_action("toggle", "scale")
        assert validations == [("scale", disabled_record.record_generation)]

        monkeypatch.setattr(
            QtWidgets.QInputDialog,
            "getItem",
            lambda *_args, **_kwargs: ("Always include", False),
        )
        before = len(updates)
        controller._manage_action("embedding", "scale")
        assert len(updates) == before
        monkeypatch.setattr(
            QtWidgets.QInputDialog,
            "getItem",
            lambda *_args, **_kwargs: ("Always include", True),
        )
        controller._manage_action("embedding", "scale")
        assert updates[-1]["embed_policy"] == "always"

        class MetadataDialog:
            metadata = _ExtensionMetadata(author="Updated")
            accepted = False

            def __init__(self, *_args, **_kwargs) -> None:
                return None

            def exec(self) -> bool:
                return self.accepted

        monkeypatch.setattr(extension_controller, "_MetadataDialog", MetadataDialog)
        before = len(updates)
        controller._manage_action("metadata", "scale")
        assert len(updates) == before
        MetadataDialog.accepted = True
        controller._manage_action("metadata", "scale")
        assert updates[-1]["metadata"].author == "Updated"

        embedded_revision = enabled_record.revisions[revision_hash].model_copy(
            update={"source_path": None}
        )
        controller.catalog.model = catalog.model_copy(
            update={
                "extensions": {
                    "scale": enabled_record.model_copy(
                        update={"revisions": {revision_hash: embedded_revision}}
                    )
                }
            }
        )
        controller._manage_action("reload", "scale")
        assert critical == [None]

        controller.catalog.model = catalog
        monkeypatch.setattr(
            controller.catalog.store,
            "update_record",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                _ExtensionCatalogConflictError("changed")
            ),
        )
        controller._manage_action("favorite", "scale")
        assert warnings == [None]


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
        _manager_loader_name_for_callable=(
            manager_base._builtin_loader_name_for_callable
        ),
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


def test_direct_extension_loader_uses_shared_loader_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = erlab.extensions.LoaderDescriptor(
        id="load_data",
        name="Load Data",
        category="Lab",
        summary="",
        function_name="load_data",
        parameters=(
            erlab.extensions.ParameterDescriptor(
                id="scale",
                kind=erlab.extensions.ParameterKind.NUMBER,
                required=False,
                default=1.0,
            ),
        ),
    )
    call = _ExtensionLoaderCall(
        manager_session_id="manager",
        catalog_generation=1,
        extension_id="lab",
        extension_name="Lab",
        revision_hash="a" * 64,
        loader_id="load_data",
        descriptor=descriptor,
        source_path=pathlib.Path("lab.py"),
        source_type="script",
        executor=lambda *_args: xr.DataArray([1.0]),
    )
    state = types.SimpleNamespace(
        explorer_loader_kwargs_by_name={},
        explorer_loader_extensions_by_name={},
    )
    manager = types.SimpleNamespace(
        _extensions=types.SimpleNamespace(
            loader_name_for_callable=lambda func: (
                func.manager_loader_name
                if isinstance(func, _ExtensionLoaderCall)
                else None
            )
        ),
        _workspace_controller=types.SimpleNamespace(_loader_state=state),
        _available_file_loaders=lambda: {"Lab Data (*.dat)": (call, {})},
        _recent_loader_kwargs_by_filter={"Lab Data (*.dat)": {"scale": 4.0}},
        _recent_loader_extensions_by_filter={},
        _recent_name_filter="Lab Data (*.dat)",
        _set_shared_loader_options=lambda *_args: None,
        _mark_workspace_layout_dirty=lambda: None,
    )
    manager._manager_loader_name_for_callable = types.MethodType(
        manager_base._ImageToolManagerBase._manager_loader_name_for_callable,
        manager,
    )

    shared_kwargs, shared_extensions = (
        manager_base._ImageToolManagerBase._shared_loader_state(manager)
    )

    assert shared_kwargs == {"lab:load_data": {"scale": 4.0}}
    assert shared_extensions == {}

    manager_base._ImageToolManagerBase._sync_shared_loader_state(
        manager,
        {"lab:load_data": {"scale": 7.0}},
        {},
        apply_explorer=False,
    )
    assert manager._recent_loader_kwargs_by_filter == {
        "Lab Data (*.dat)": {"scale": 7.0}
    }

    class _AcceptParameterDialog:
        def __init__(self, dialog_descriptor, parent, values) -> None:
            assert dialog_descriptor is descriptor
            assert parent is manager
            assert values == {"scale": 7.0}

        def exec(self) -> bool:
            return True

        @property
        def parameters(self) -> dict[str, float]:
            return {"scale": 7.0}

    monkeypatch.setattr(
        extension_dialogs, "_ExtensionParameterDialog", _AcceptParameterDialog
    )
    manager._shared_loader_state = types.MethodType(
        manager_base._ImageToolManagerBase._shared_loader_state,
        manager,
    )
    selected = manager_base._ImageToolManagerBase._select_loader_options(
        manager,
        {"Lab Data (*.dat)": (call, {"scale": 1.0})},
    )
    assert selected == ("Lab Data (*.dat)", call, {"scale": 7.0})


def test_disabled_environment_loader_does_not_use_global_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ingress_calls: list[dict[str, typing.Any]] = []
    dialogs: list[None] = []
    builtin_loader = types.SimpleNamespace(load=lambda *_args, **_kwargs: None)
    monkeypatch.setattr(erlab.io, "loaders", {"reserved": builtin_loader})
    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        lambda *_args, **_kwargs: dialogs.append(None),
    )
    manager = types.SimpleNamespace(
        _extensions=types.SimpleNamespace(
            environment_loader_names={"reserved"},
            loader_by_name=lambda _name: None,
        ),
        _data_ingress=types.SimpleNamespace(
            add_from_multiple_files=lambda *_args, **kwargs: ingress_calls.append(
                kwargs
            )
        ),
    )

    from erlab.interactive.imagetool.manager._actions import _ActionsController

    _ActionsController(manager)._data_load(["data.txt"], "reserved", {})

    assert ingress_calls == []
    assert dialogs == [None]


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


def test_catalog_uses_override_directory_and_safe_generated_id(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "extension-catalog"
    monkeypatch.setenv("ERLAB_EXTENSION_CATALOG", os.fspath(directory))

    assert extension_catalog._default_catalog_directory() == directory.resolve()
    assert extension_catalog._safe_extension_id(" Lab analysis! ") == "Lab-analysis"
    assert extension_catalog._safe_extension_id("...").startswith("extension-")


def test_catalog_reports_lock_failure(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailedLock:
        def __init__(self, path: str) -> None:
            self.path = path

        def setStaleLockTime(self, timeout: int) -> None:
            self.timeout = timeout

        def tryLock(self, timeout: int) -> bool:
            self.lock_timeout = timeout
            return False

        def error(self) -> str:
            return "lock failed"

    monkeypatch.setattr(extension_catalog.QtCore, "QLockFile", FailedLock)
    store = _ExtensionCatalogStore(tmp_path / "catalog")

    with pytest.raises(extension_catalog._ExtensionCatalogLockError, match="lock"):
        store.mutate(None, lambda catalog: catalog)


@pytest.mark.parametrize("failure", ["open", "write", "commit"])
@pytest.mark.parametrize("target", ["catalog", "source"])
def test_catalog_reports_atomic_write_failures(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    target: str,
) -> None:
    canceled: list[None] = []

    class FailedSaveFile:
        def __init__(self, path: str) -> None:
            self.path = path

        def open(self, _mode: typing.Any) -> bool:
            return failure != "open"

        def write(self, payload: bytes) -> int:
            return len(payload) - 1 if failure == "write" else len(payload)

        def cancelWriting(self) -> None:
            canceled.append(None)

        def commit(self) -> bool:
            return failure != "commit"

        def errorString(self) -> str:
            return f"{failure} failed"

    monkeypatch.setattr(extension_catalog.QtCore, "QSaveFile", FailedSaveFile)
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    source = b"source"

    def operation() -> None:
        if target == "catalog":
            store._write_unlocked(_ExtensionCatalogModel())
        else:
            store._store_script_source(source, hashlib.sha256(source).hexdigest())

    with pytest.raises(extension_catalog._ExtensionCatalogError, match=failure):
        operation()

    assert canceled == ([None] if failure == "write" else [])


def test_catalog_source_lookup_and_integrity_failures(tmp_path: pathlib.Path) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    script_path = tmp_path / "scale.py"
    source = _script(script_path)
    catalog, revision, _created = store.add_script(script_path)

    with pytest.raises(KeyError, match="Unknown extension revision"):
        store.source_path("missing", revision)
    store.source_path("scale", revision).unlink()
    with pytest.raises(FileNotFoundError):
        store.source_path("scale", revision)
    with pytest.raises(ValueError, match="does not match its revision hash"):
        store._store_script_source(source, "0" * 64)
    with pytest.raises(ValueError, match="does not match its manifest"):
        store.add_embedded_script(
            source,
            extension_id="embedded",
            expected_revision="0" * 64,
            name="Embedded",
            metadata=_ExtensionMetadata(),
        )

    assert store.read() == catalog


@pytest.mark.parametrize(
    ("record_update", "approved", "message"),
    [
        ({"source_type": "environment-package"}, False, "requires an entry point"),
        ({"enabled": True}, False, "must be approved"),
        ({"enabled": True, "removed": True}, True, "cannot be enabled"),
    ],
)
def test_extension_record_rejects_invalid_enabled_and_package_states(
    record_update: dict[str, typing.Any],
    approved: bool,
    message: str,
) -> None:
    revision = "a" * 64
    record = _ExtensionRecord(
        id="lab",
        current_revision=revision,
        name="Lab",
        revisions={
            revision: _ExtensionRevision(
                source_hash=revision,
                object_name=f"{revision}.py",
                created_at="2026-01-01T00:00:00+00:00",
                approved=approved,
            )
        },
    )

    with pytest.raises(ValueError, match=message):
        _ExtensionRecord.model_validate(
            record.model_copy(update=record_update).model_dump(mode="python")
        )


def test_extension_models_reject_invalid_hash_and_nonfinite_loader_defaults() -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        _ExtensionRevision(
            source_hash="A" * 64,
            object_name="source.py",
            created_at="2026-01-01T00:00:00+00:00",
        )
    with pytest.raises(ValueError, match="must be finite"):
        extension_catalog._EnvironmentLoaderMethod(
            name_filter="Data (*)", defaults={"scale": float("inf")}
        )


def test_catalog_validation_rejects_a_revision_changed_during_commit(
    tmp_path: pathlib.Path,
) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    script_path = tmp_path / "scale.py"
    _script(script_path)
    catalog, old_revision, _created = store.add_script(script_path)
    old_generation = catalog.extensions["scale"].record_generation
    _script(script_path, "data + scale")
    catalog, new_revision, _created = store.add_script(script_path)

    with pytest.raises(_ExtensionCatalogConflictError, match="during validation"):
        store.record_validation_failure(
            "scale",
            revision_hash=old_revision,
            expected_record_generation=catalog.extensions["scale"].record_generation,
            import_error="invalid",
        )
    with pytest.raises(_ExtensionCatalogConflictError, match="during validation"):
        store.enable_validated_revision(
            "scale",
            revision_hash=old_revision,
            expected_record_generation=catalog.extensions["scale"].record_generation,
            routines=(),
            loaders=(),
            loader_always_single=None,
            loader_dialog_methods=(),
        )

    current = store.read().extensions["scale"]
    assert current.current_revision == new_revision
    assert current.record_generation == old_generation + 1


def test_catalog_reports_exact_script_capability_states(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    script_path = tmp_path / "scale.py"
    _script(script_path)
    catalog, revision_hash, _created = store.add_script(script_path)
    catalog = _validate_and_enable(
        store,
        "scale",
        expected_record_generation=catalog.extensions["scale"].record_generation,
    )
    ready_record = catalog.extensions["scale"]

    def status_for(record: _ExtensionRecord, capability_id: str = "scale") -> str:
        model = catalog.model_copy(update={"extensions": {"scale": record}})
        monkeypatch.setattr(store, "read", lambda: model)
        return store.capability_status("scale", revision_hash, "routine", capability_id)

    assert status_for(ready_record) == "ready"
    assert status_for(ready_record.model_copy(update={"removed": True})) == (
        "missing-revision"
    )
    assert status_for(ready_record.model_copy(update={"enabled": False})) == "disabled"

    ready_revision = ready_record.revisions[revision_hash]
    assert (
        status_for(
            ready_record.model_copy(
                update={
                    "enabled": False,
                    "revisions": {
                        revision_hash: ready_revision.model_copy(
                            update={"approved": False}
                        )
                    },
                }
            )
        )
        == "approval-required"
    )
    assert (
        status_for(
            ready_record.model_copy(
                update={
                    "revisions": {
                        revision_hash: ready_revision.model_copy(
                            update={"import_error": "broken"}
                        )
                    }
                }
            )
        )
        == "import-failed"
    )
    assert status_for(ready_record, "missing") == "missing-capability"

    unsupported_descriptor = ready_revision.routines[0].model_copy(
        update={"extension_api_version": 2}
    )
    assert (
        status_for(
            ready_record.model_copy(
                update={
                    "revisions": {
                        revision_hash: ready_revision.model_copy(
                            update={"routines": (unsupported_descriptor,)}
                        )
                    }
                }
            )
        )
        == "unsupported-api"
    )

    monkeypatch.undo()
    object_path = store.objects_directory / ready_revision.object_name
    object_path.write_bytes(b"corrupt")
    assert (
        store.capability_status("scale", revision_hash, "routine", "scale")
        == "hash-mismatch"
    )
    object_path.unlink()
    assert (
        store.capability_status("scale", revision_hash, "routine", "scale")
        == "missing-revision"
    )


def test_catalog_capability_status_reports_missing_package_revision(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision_hash = "a" * 64
    revision = _ExtensionRevision(
        source_hash=revision_hash,
        object_name="lab_package:calculate",
        created_at="2026-01-01T00:00:00+00:00",
        approved=True,
        entry_point_group="erlab.extensions",
        entry_point_name="lab",
        entry_point_value="lab_package:calculate",
        routines=(
            erlab.extensions.RoutineDescriptor(
                id="calculate",
                name="Calculate",
                category="Lab",
                summary="",
                function_name="calculate",
            ),
        ),
    )
    record = _ExtensionRecord(
        id="lab",
        name="Lab",
        source_type="environment-package",
        enabled=True,
        current_revision=revision_hash,
        revisions={revision_hash: revision},
    )
    model = _ExtensionCatalogModel(extensions={"lab": record})
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    monkeypatch.setattr(store, "read", lambda: model)
    monkeypatch.setattr(
        store,
        "_entry_point_for_revision",
        lambda _revision: (_ for _ in ()).throw(ImportError("missing")),
    )

    assert (
        store.capability_status("lab", revision_hash, "routine", "calculate")
        == "missing-revision"
    )
    assert not store.revision_available(record, revision_hash)


def test_catalog_revision_availability_rejects_unknown_and_missing_sources(
    tmp_path: pathlib.Path,
) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    script_path = tmp_path / "scale.py"
    _script(script_path)
    catalog, revision_hash, _created = store.add_script(script_path)
    record = catalog.extensions["scale"]

    assert not store.revision_available(record, "0" * 64)
    store.source_path("scale", revision_hash).unlink()
    assert not store.revision_available(record, revision_hash)


def test_catalog_resolve_script_capability_rejects_unusable_state(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    script_path = tmp_path / "scale.py"
    _script(script_path)
    catalog, revision_hash, _created = store.add_script(script_path)

    with pytest.raises(erlab.extensions.ExtensionNotFoundError, match="disabled"):
        store.resolve_capability("scale", revision_hash, "routine", "scale")
    unapproved = catalog.extensions["scale"].model_copy(update={"enabled": True})
    monkeypatch.setattr(
        store,
        "read",
        lambda: catalog.model_copy(update={"extensions": {"scale": unapproved}}),
    )
    with pytest.raises(erlab.extensions.ExtensionNotFoundError, match="not approved"):
        store.resolve_capability("scale", revision_hash, "routine", "scale")
    monkeypatch.undo()

    catalog = _validate_and_enable(
        store,
        "scale",
        expected_record_generation=catalog.extensions["scale"].record_generation,
    )
    catalog = store.update_record(
        "scale",
        expected_record_generation=catalog.extensions["scale"].record_generation,
        enabled=False,
    )
    with pytest.raises(erlab.extensions.ExtensionNotFoundError, match="disabled"):
        store.resolve_capability("scale", revision_hash, "routine", "scale")
    catalog = store.update_record(
        "scale",
        expected_record_generation=catalog.extensions["scale"].record_generation,
        enabled=True,
    )
    with pytest.raises(KeyError, match="Unknown routine capability"):
        store.resolve_capability("scale", revision_hash, "routine", "missing")


def test_catalog_watcher_removes_stale_paths_and_ignores_closed_refresh(
    tmp_path: pathlib.Path,
) -> None:
    catalog = _ExtensionCatalog(directory=tmp_path / "catalog")
    stale = tmp_path / "stale"
    stale.mkdir()
    try:
        assert catalog._watcher.addPath(os.fspath(stale))
        catalog._restore_watches()
        assert os.fspath(stale) not in catalog._watcher.directories()

        catalog.close()
        catalog.refresh()
        assert catalog._closed
    finally:
        catalog.close()


def test_catalog_finds_an_uncached_exact_environment_entry_point(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EntryPoint:
        group = "erlab.extensions"
        name = "lab"
        value = "lab_package:calculate"
        dist = None

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

    invalid = EntryPoint()
    valid = EntryPoint()
    revision_hash = extension_entry_points._entry_point_revision(valid)
    revision = _ExtensionRevision(
        source_hash=revision_hash,
        object_name=valid.value,
        created_at="2026-01-01T00:00:00+00:00",
        entry_point_group=valid.group,
        entry_point_name=valid.name,
        entry_point_value=valid.value,
    )
    calls: list[object] = []

    def inspect(entry_point: object) -> str:
        calls.append(entry_point)
        if entry_point is invalid:
            raise extension_entry_points._EntryPointRevisionError("invalid")
        return revision_hash

    monkeypatch.setattr(
        extension_catalog.importlib.metadata,
        "entry_points",
        lambda: EntryPoints((invalid, valid)),
    )
    monkeypatch.setattr(extension_catalog, "_entry_point_revision", inspect)
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    type(store)._environment_entry_points = {}

    assert store._entry_point_for_revision(revision) is valid
    assert calls == [invalid, valid]
    calls.clear()
    assert store._entry_point_for_revision(revision) is valid
    assert calls == []


def test_catalog_resolves_environment_module_and_callable_capabilities(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @erlab.extensions.routine(id="calculate")
    def calculate(data: xr.DataArray) -> xr.DataArray:
        return data + 1

    @erlab.extensions.loader(id="load_data")
    def load_data(path: pathlib.Path) -> xr.DataArray:
        return xr.DataArray([len(path.name)])

    revision_hash = "a" * 64
    revision = _ExtensionRevision(
        source_hash=revision_hash,
        object_name="lab_package:extension",
        created_at="2026-01-01T00:00:00+00:00",
        approved=True,
        entry_point_group="erlab.extensions",
        entry_point_name="lab",
        entry_point_value="lab_package:extension",
    )
    record = _ExtensionRecord(
        id="lab",
        name="Lab",
        source_type="environment-package",
        enabled=True,
        current_revision=revision_hash,
        revisions={revision_hash: revision},
    )
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    monkeypatch.setattr(
        store,
        "read",
        lambda: _ExtensionCatalogModel(extensions={"lab": record}),
    )
    entry_point = types.SimpleNamespace(group="erlab.extensions")
    monkeypatch.setattr(store, "_entry_point_for_revision", lambda _: entry_point)

    module = types.ModuleType("lab_package.extension")
    calculate.__module__ = module.__name__
    load_data.__module__ = module.__name__
    module.calculate = calculate
    module.load_data = load_data
    monkeypatch.setattr(
        extension_catalog, "_load_entry_point_value", lambda *_args: module
    )
    assert (
        store.resolve_capability("lab", revision_hash, "routine", "calculate")
        is calculate
    )
    assert (
        store.resolve_capability("lab", revision_hash, "loader", "load_data")
        is load_data
    )

    monkeypatch.setattr(
        extension_catalog, "_load_entry_point_value", lambda *_args: calculate
    )
    assert (
        store.resolve_capability("lab", revision_hash, "routine", "calculate")
        is calculate
    )
    with pytest.raises(KeyError):
        store.resolve_capability("lab", revision_hash, "routine", "missing")

    monkeypatch.setattr(
        extension_catalog, "_load_entry_point_value", lambda *_args: load_data
    )
    with pytest.raises(TypeError, match="not a routine"):
        store.resolve_capability("lab", revision_hash, "routine", "load_data")


def test_catalog_resolves_loaderbase_classes_and_instances(
    example_loader,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision_hash = "a" * 64
    revision = _ExtensionRevision(
        source_hash=revision_hash,
        object_name="lab_package:ExampleLoader",
        created_at="2026-01-01T00:00:00+00:00",
        approved=True,
        entry_point_group="erlab.io.loaders",
        entry_point_name="example",
        entry_point_value="lab_package:ExampleLoader",
        loader_dialog_methods=(
            extension_catalog._EnvironmentLoaderMethod(
                name_filter="Example preview (*.txt)", method="preview"
            ),
        ),
    )
    record = _ExtensionRecord(
        id="lab-loader",
        name="Lab loader",
        source_type="environment-package",
        enabled=True,
        current_revision=revision_hash,
        revisions={revision_hash: revision},
    )
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    monkeypatch.setattr(
        store,
        "read",
        lambda: _ExtensionCatalogModel(extensions={"lab-loader": record}),
    )
    entry_point = types.SimpleNamespace(group="erlab.io.loaders")
    monkeypatch.setattr(store, "_entry_point_for_revision", lambda _: entry_point)

    def preview(self, path: pathlib.Path) -> xr.DataArray:
        del self
        return xr.DataArray([len(path.name)])

    monkeypatch.setattr(
        example_loader,
        "preview",
        preview,
        raising=False,
    )

    for value in (example_loader, example_loader()):
        load_calls: list[str] = []

        def load_entry_point_value(*_args, value=value, calls=load_calls):
            calls.append("load")
            return value

        monkeypatch.setattr(
            extension_catalog,
            "_load_entry_point_value",
            load_entry_point_value,
        )
        resolved = store.resolve_capability(
            "lab-loader", revision_hash, "loader", "example"
        )
        assert getattr(resolved, "__self__", None).name == "example"

        preview = store.resolve_capability(
            "lab-loader", revision_hash, "loader", "example", "preview"
        )
        assert getattr(preview, "__self__", None).name == "example"
        assert getattr(preview, "__name__", None) == "preview"

        load_count = len(load_calls)
        with pytest.raises(
            erlab.extensions.ExtensionNotFoundError,
            match="not approved for this revision",
        ):
            store.resolve_capability(
                "lab-loader", revision_hash, "loader", "example", "os.remove"
            )
        assert len(load_calls) == load_count

    with pytest.raises(KeyError):
        store.resolve_capability("lab-loader", revision_hash, "routine", "example")
    with pytest.raises(KeyError):
        store.resolve_capability("lab-loader", revision_hash, "loader", "different")
    monkeypatch.setattr(
        extension_catalog, "_load_entry_point_value", lambda *_args: object()
    )
    with pytest.raises(TypeError, match="does not provide LoaderBase"):
        store.resolve_capability("lab-loader", revision_hash, "loader", "example")


def test_extension_execution_value_guards_and_log_fields(
    tmp_path: pathlib.Path,
) -> None:
    error = ValueError("invalid")
    assert extension_execution._extension_error(error, "call") is error
    converted = extension_execution._extension_error(KeyboardInterrupt(), "call")
    assert isinstance(converted, erlab.extensions.ExtensionExecutionError)
    assert "KeyboardInterrupt" in str(converted)

    descriptor = erlab.extensions.LoaderDescriptor(
        id="load_data",
        name="Load Data",
        category="Lab",
        summary="",
        function_name="load_data",
    )
    call = _ExtensionLoaderCall(
        manager_session_id="manager",
        catalog_generation=1,
        extension_id="lab",
        extension_name="Lab",
        revision_hash="a" * 64,
        loader_id="load_data",
        descriptor=descriptor,
        source_path=None,
        source_type="script",
        executor=lambda *_args: xr.DataArray([1.0]),
        loader_method="preview",
    )
    assert call.manager_loader_name == "lab:load_data"
    assert not call.uses_standard_loader_options
    assert call.__name__ == "preview"
    with pytest.raises(
        erlab.extensions.ExtensionExecutionError, match="source is missing"
    ):
        extension_execution._require_loader_source(call)
    with pytest.raises(erlab.extensions.ExtensionExecutionError, match="missing"):
        extension_execution._require_loader_entry(call, None)

    array = xr.DataArray([1.0], dims="x")
    dataset = xr.Dataset({"value": array})
    tree = xr.DataTree.from_dict({"/": dataset})
    assert extension_execution._require_loader_output(
        [array, dataset, tree], allow_multiple=True
    ) == [array, dataset, tree]
    with pytest.raises(
        erlab.extensions.ExtensionExecutionError, match="expected an xarray"
    ):
        extension_execution._require_loader_output([array], allow_multiple=False)
    with pytest.raises(
        erlab.extensions.ExtensionExecutionError,
        match="list of xarray objects",
    ):
        extension_execution._require_loader_output([object()], allow_multiple=True)

    assert extension_execution._xarray_log_fields(dataset) == {
        "type": "Dataset",
        "dimensions": ("x",),
        "shape": (1,),
        "dtype": ("float64",),
    }
    tree_fields = extension_execution._xarray_log_fields(tree)
    assert tree_fields["type"] == "DataTree"
    assert tree_fields["dimensions"] == ("x",)
    assert tree_fields["shape"] == (1,)
    list_fields = extension_execution._loader_output_log_fields([array, dataset])
    assert list_fields["type"] == "list"
    assert len(list_fields["items"]) == 2

    with pytest.raises(
        erlab.extensions.ExtensionExecutionError, match="expected DataArray"
    ):
        extension_execution._require_dataarray(dataset)
    with pytest.raises(
        erlab.extensions.ExtensionExecutionError, match="source is missing"
    ):
        extension_execution._require_script_source(
            types.SimpleNamespace(source_path=None)
        )

    source = tmp_path / "loader.py"
    source.write_text("source")
    assert (
        extension_execution._require_loader_source(
            types.SimpleNamespace(source_path=source)
        )
        == source
    )


def test_decorated_loader_adapter_preserves_loader_contract(
    tmp_path: pathlib.Path,
) -> None:
    calls: list[tuple[pathlib.Path, dict[str, typing.Any]]] = []

    def execute(
        _call: _ExtensionLoaderCall,
        path: pathlib.Path,
        parameters: dict[str, typing.Any],
    ) -> xr.DataArray:
        calls.append((path, parameters))
        return xr.DataArray([float(path.read_text())])

    descriptor = erlab.extensions.LoaderDescriptor(
        id="load_data",
        name="Load Data",
        category="Lab",
        summary="Load one data file.",
        function_name="load_data",
        extensions=(".dat",),
    )
    call = _ExtensionLoaderCall(
        manager_session_id="manager",
        catalog_generation=1,
        extension_id="lab",
        extension_name="Lab",
        revision_hash="a" * 64,
        loader_id="load_data",
        descriptor=descriptor,
        source_path=tmp_path / "loader.py",
        source_type="script",
        executor=execute,
    )
    adapter = extension_execution._DecoratedLoaderAdapter(call)
    path = tmp_path / "value.dat"
    path.write_text("3")

    assert adapter.extension_id == "lab"
    assert adapter.revision_hash == "a" * 64
    assert adapter.loader_id == "load_data"
    assert adapter.loader_method is None
    assert adapter.source_path == tmp_path / "loader.py"
    assert adapter.source_type == "script"
    assert adapter.entry_point_group is None
    assert adapter.entry_point_name is None
    assert adapter.descriptor == descriptor
    assert not adapter.uses_standard_loader_options
    assert tuple(adapter.file_dialog_methods) == ("Load Data (*.dat)",)
    loaded = adapter.load(path)
    assert loaded.item() == 3.0
    assert loaded.attrs["data_loader_name"] == "lab:load_data"
    xr.testing.assert_identical(
        adapter.load_single(path, scale=2.0), xr.DataArray([3.0])
    )
    assert calls == [(path, {}), (path, {"scale": 2.0})]

    environment_call = _ExtensionLoaderCall(
        manager_session_id="manager",
        catalog_generation=1,
        extension_id="environment.erlab.io.loaders.lab",
        extension_name="Lab",
        revision_hash="b" * 64,
        loader_id="lab",
        descriptor=descriptor,
        source_path=None,
        source_type="environment-package",
        executor=execute,
        entry_point_group="erlab.io.loaders",
        entry_point_name="lab",
    )
    environment_adapter = extension_execution._DecoratedLoaderAdapter(environment_call)
    assert environment_adapter.uses_standard_loader_options
    assert environment_adapter.name == "lab"
    with pytest.raises(TypeError, match="must use keywords"):
        environment_adapter.load(path, 2.0)
    xr.testing.assert_identical(
        environment_adapter.load(path, scale=4.0), xr.DataArray([3.0])
    )
    with pytest.raises(ValueError, match="must be finite"):
        environment_call(path, scale=float("inf"))


def test_loader_and_validation_workers_ignore_repeated_cancellation(
    tmp_path: pathlib.Path,
) -> None:
    descriptor = erlab.extensions.LoaderDescriptor(
        id="load_data",
        name="Load Data",
        category="Lab",
        summary="",
        function_name="load_data",
    )
    call = _ExtensionLoaderCall(
        manager_session_id="manager",
        catalog_generation=1,
        extension_id="lab",
        extension_name="Lab",
        revision_hash="a" * 64,
        loader_id="load_data",
        descriptor=descriptor,
        source_path=tmp_path / "loader.py",
        source_type="script",
        executor=lambda *_args: xr.DataArray([1.0]),
    )
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    loader_worker = _ExtensionLoaderWorker(call, tmp_path / "data", {}, store, {})
    loader_worker.cancel_if_pending()
    first_error = loader_worker.error
    loader_worker.cancel_if_pending()
    loader_worker.run()
    assert loader_worker.error is first_error
    assert not loader_worker._started

    validation_worker = _ExtensionValidationWorker(
        "lab",
        "a" * 64,
        1,
        manager_session_id="manager",
        catalog_store=store,
        script_modules={},
    )
    validation_worker.cancel_if_pending()
    validation_error = validation_worker.error
    validation_worker.cancel_if_pending()
    validation_worker.run()
    assert validation_worker.error is validation_error
    assert not validation_worker._started


def test_loader_worker_contains_process_control_exceptions(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = erlab.extensions.LoaderDescriptor(
        id="load_data",
        name="Load Data",
        category="Lab",
        summary="",
        function_name="load_data",
    )
    call = _ExtensionLoaderCall(
        manager_session_id="manager",
        catalog_generation=1,
        extension_id="lab",
        extension_name="Lab",
        revision_hash="a" * 64,
        loader_id="load_data",
        descriptor=descriptor,
        source_path=tmp_path / "loader.py",
        source_type="script",
        executor=lambda *_args: xr.DataArray([1.0]),
    )
    record = types.SimpleNamespace(removed=False, enabled=True)
    store = types.SimpleNamespace(
        read=lambda: types.SimpleNamespace(extensions={"lab": record})
    )
    monkeypatch.setattr(
        extension_execution._ExtensionLoaderCall,
        "_invoke",
        lambda *_args: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    worker = _ExtensionLoaderWorker(
        call,
        tmp_path / "data",
        {},
        typing.cast("_ExtensionCatalogStore", store),
        {},
    )

    worker.run()

    assert worker.done.is_set()
    assert isinstance(worker.error, erlab.extensions.ExtensionExecutionError)
    assert "KeyboardInterrupt" in str(worker.error)


def test_environment_routine_resolves_callable_and_module_entry_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @erlab.extensions.routine(id="calculate")
    def calculate(data: xr.DataArray) -> xr.DataArray:
        return data + 1

    class EntryPoint:
        group = "erlab.extensions"
        name = "lab"
        value = "lab_package:extension"

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

    entry_point = EntryPoint()
    monkeypatch.setattr(
        extension_execution.importlib.metadata,
        "entry_points",
        lambda: EntryPoints((entry_point,)),
    )
    monkeypatch.setattr(
        extension_execution, "_environment_revision_matches", lambda *_args: True
    )
    routine_descriptor = erlab.extensions.RoutineDescriptor(
        id="calculate",
        name="Calculate",
        category="Lab",
        summary="",
        function_name="calculate",
    )
    job = types.SimpleNamespace(
        entry_point_group=entry_point.group,
        entry_point_name=entry_point.name,
        entry_point_value=entry_point.value,
        revision_hash="a" * 64,
        routine=routine_descriptor,
    )

    monkeypatch.setattr(
        extension_execution, "_load_entry_point_value", lambda *_args: calculate
    )
    resolved_routine = extension_execution._environment_routine(job)
    assert resolved_routine[0].id == "calculate"
    assert resolved_routine[1] is calculate

    module = types.ModuleType("lab_package.extension")
    calculate.__module__ = module.__name__
    module.calculate = calculate
    monkeypatch.setattr(
        extension_execution, "_load_entry_point_value", lambda *_args: module
    )
    assert extension_execution._environment_routine(job)[1] is calculate

    monkeypatch.setattr(
        extension_execution, "_load_entry_point_value", lambda *_args: object()
    )
    with pytest.raises(
        erlab.extensions.ExtensionExecutionError, match="no longer available"
    ):
        extension_execution._environment_routine(job)

    monkeypatch.setattr(
        extension_execution, "_environment_revision_matches", lambda *_args: False
    )
    with pytest.raises(
        erlab.extensions.ExtensionExecutionError, match="no longer available"
    ):
        extension_execution._environment_routine(job)


def test_environment_loader_resolves_supported_entry_point_shapes(
    example_loader,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @erlab.extensions.loader(id="load_data")
    def load_data(path: pathlib.Path) -> xr.DataArray:
        return xr.DataArray([len(path.name)])

    class EntryPoint:
        group = "erlab.extensions"
        name = "lab"
        value = "lab_package:extension"

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

    entry_point = EntryPoint()
    monkeypatch.setattr(
        extension_execution.importlib.metadata,
        "entry_points",
        lambda: EntryPoints((entry_point,)),
    )
    monkeypatch.setattr(
        extension_execution, "_environment_revision_matches", lambda *_args: True
    )
    descriptor = erlab.extensions.LoaderDescriptor(
        id="load_data",
        name="Load Data",
        category="Lab",
        summary="",
        function_name="load_data",
    )
    call = types.SimpleNamespace(
        entry_point_group=entry_point.group,
        entry_point_name=entry_point.name,
        entry_point_value=entry_point.value,
        revision_hash="a" * 64,
        loader_id="load_data",
        descriptor=descriptor,
        loader_method=None,
    )

    monkeypatch.setattr(
        extension_execution, "_load_entry_point_value", lambda *_args: load_data
    )
    resolved_loader = extension_execution._environment_loader(call)
    assert resolved_loader is not None
    assert resolved_loader[0].id == "load_data"
    assert resolved_loader[1] is load_data

    module = types.ModuleType("lab_package.extension")
    load_data.__module__ = module.__name__
    module.load_data = load_data
    monkeypatch.setattr(
        extension_execution, "_load_entry_point_value", lambda *_args: module
    )
    assert extension_execution._environment_loader(call)[1] is load_data

    entry_point.group = "erlab.io.loaders"
    call.entry_point_group = entry_point.group
    call.entry_point_name = "example"
    entry_point.name = "example"
    call.loader_id = "example"
    for value in (example_loader, example_loader()):
        monkeypatch.setattr(
            extension_execution,
            "_load_entry_point_value",
            lambda *_args, value=value: value,
        )
        resolved = extension_execution._environment_loader(call)
        assert resolved is not None
        assert getattr(resolved[1], "__self__", None).name == "example"

    call.loader_id = "different"
    assert extension_execution._environment_loader(call) is None
    call.loader_id = "example"
    monkeypatch.setattr(
        extension_execution, "_load_entry_point_value", lambda *_args: object()
    )
    assert extension_execution._environment_loader(call) is None

    monkeypatch.setattr(
        extension_execution, "_environment_revision_matches", lambda *_args: False
    )
    assert extension_execution._environment_loader(call) is None


def test_environment_capability_validation_rejects_invalid_entry_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = _ExtensionRevision(
        source_hash="a" * 64,
        object_name="lab_package:extension",
        created_at="2026-01-01T00:00:00+00:00",
        entry_point_group="erlab.extensions",
        entry_point_name="lab",
        entry_point_value="lab_package:extension",
    )
    entry_point = types.SimpleNamespace(group="erlab.extensions")
    store = types.SimpleNamespace(
        _entry_point_for_revision=lambda _revision: entry_point
    )

    monkeypatch.setattr(
        extension_execution, "_load_entry_point_value", lambda *_args: object()
    )
    with pytest.raises(TypeError, match="decorated function or module"):
        extension_execution._environment_capabilities(
            typing.cast("_ExtensionCatalogStore", store), revision
        )

    module = types.ModuleType("empty_extension")
    monkeypatch.setattr(
        extension_execution, "_load_entry_point_value", lambda *_args: module
    )
    with pytest.raises(TypeError, match="has no capabilities"):
        extension_execution._environment_capabilities(
            typing.cast("_ExtensionCatalogStore", store), revision
        )

    entry_point.group = "erlab.io.loaders"
    monkeypatch.setattr(
        extension_execution, "_load_entry_point_value", lambda *_args: object()
    )
    with pytest.raises(TypeError, match="must provide LoaderBase"):
        extension_execution._environment_capabilities(
            typing.cast("_ExtensionCatalogStore", store), revision
        )


def test_extension_validation_rejects_an_unknown_record(
    tmp_path: pathlib.Path,
) -> None:
    with pytest.raises(KeyError, match="missing"):
        _validate_extension_revision(
            _ExtensionCatalogStore(tmp_path / "catalog"),
            "missing",
            revision_hash="a" * 64,
            expected_record_generation=0,
            manager_session_id="manager",
            script_modules={},
        )


def test_execution_controller_reports_admission_and_validation_failures(
    manager_context,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = tmp_path / "scale.py"
    _script(script_path)

    with manager_context() as manager:
        execution = manager._extensions.execution
        with pytest.raises(KeyError, match="missing"):
            execution.validate_and_enable("missing", expected_record_generation=0)

        catalog, _revision, _created = manager._extensions.catalog.store.add_script(
            script_path
        )
        record = catalog.extensions["scale"]
        with pytest.raises(_ExtensionCatalogConflictError, match="before validation"):
            execution.validate_and_enable(
                "scale", expected_record_generation=record.record_generation + 1
            )

        monkeypatch.setattr(
            execution, "_run_blocking_task", lambda *_args, **_kwargs: None
        )
        with pytest.raises(
            erlab.extensions.ExtensionExecutionError,
            match="validation returned no result",
        ):
            execution.validate_and_enable(
                "scale", expected_record_generation=record.record_generation
            )

        descriptor = erlab.extensions.LoaderDescriptor(
            id="load_data",
            name="Load Data",
            category="Lab",
            summary="",
            function_name="load_data",
        )
        call = _ExtensionLoaderCall(
            manager_session_id="manager",
            catalog_generation=1,
            extension_id="lab",
            extension_name="Lab",
            revision_hash="a" * 64,
            loader_id="load_data",
            descriptor=descriptor,
            source_path=tmp_path / "loader.py",
            source_type="script",
            executor=lambda *_args: xr.DataArray([1.0]),
        )
        with pytest.raises(
            erlab.extensions.ExtensionExecutionError,
            match="loader returned no result",
        ):
            execution.run_loader(call, tmp_path / "data", {})
        monkeypatch.undo()

        execution._accepting = False
        with pytest.raises(
            erlab.extensions.ExtensionExecutionError, match="shutting down"
        ):
            execution._run_blocking_task(
                _ExtensionLoaderWorker(
                    call,
                    tmp_path / "data",
                    {},
                    manager._extensions.catalog.store,
                    {},
                )
            )
        with pytest.raises(RuntimeError, match="shutting down"):
            execution.queue_routine(
                extension_id="scale", routine_id="scale", parameters={}, target=0
            )


def test_execution_controller_removes_failed_pool_admission(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    class FailedPool:
        @staticmethod
        def start(_task: typing.Any) -> typing.Never:
            raise RuntimeError("pool rejected task")

    descriptor = erlab.extensions.LoaderDescriptor(
        id="load_data",
        name="Load Data",
        category="Lab",
        summary="",
        function_name="load_data",
    )
    call = _ExtensionLoaderCall(
        manager_session_id="manager",
        catalog_generation=1,
        extension_id="lab",
        extension_name="Lab",
        revision_hash="a" * 64,
        loader_id="load_data",
        descriptor=descriptor,
        source_path=tmp_path / "loader.py",
        source_type="script",
        executor=lambda *_args: xr.DataArray([1.0]),
    )

    with manager_context() as manager:
        execution = manager._extensions.execution
        original_pool = execution._pool
        execution._pool = typing.cast("QtCore.QThreadPool", FailedPool())
        task = _ExtensionLoaderWorker(
            call,
            tmp_path / "data",
            {},
            manager._extensions.catalog.store,
            {},
        )
        try:
            with pytest.raises(RuntimeError, match="pool rejected"):
                execution._run_blocking_task(task)
            assert task not in execution._blocking_tasks
        finally:
            execution._pool = original_pool


def test_session_extension_status_and_loader_errors(
    manager_context,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "loader.py"
    source = _loader_script(source_path, name="Load Data", extensions=(".dat",))
    revision_hash = hashlib.sha256(source).hexdigest()

    with manager_context() as manager:
        execution = manager._extensions.execution
        assert execution._session_revision("missing", revision_hash) is None
        with pytest.raises(
            erlab.extensions.ExtensionExecutionError, match="not available"
        ):
            execution.session_loader_call("missing", revision_hash, "load_data")

        catalog = execution._session_catalog_store.add_embedded_script(
            source,
            extension_id="loader",
            expected_revision=revision_hash,
            name="Loader",
            metadata=_ExtensionMetadata(),
        )
        assert execution._session_revision("loader", "0" * 64) is None
        assert (
            execution.session_capability_status(
                "loader", revision_hash, "loader", "load_data"
            )
            == "approval-required"
        )
        with pytest.raises(
            erlab.extensions.ExtensionExecutionError, match="not available"
        ):
            execution.session_loader_call("loader", revision_hash, "load_data")

        record = catalog.extensions["loader"]
        revision = record.revisions[revision_hash]

        def status_for(
            updated_record: _ExtensionRecord,
            capability_id: str = "load_data",
        ) -> str | None:
            model = catalog.model_copy(
                update={"extensions": {"loader": updated_record}}
            )
            monkeypatch.setattr(execution._session_catalog_store, "read", lambda: model)
            return execution.session_capability_status(
                "loader", revision_hash, "loader", capability_id
            )

        assert (
            status_for(
                record.model_copy(
                    update={
                        "revisions": {
                            revision_hash: revision.model_copy(
                                update={"import_error": "broken"}
                            )
                        }
                    }
                )
            )
            == "import-failed"
        )
        approved_revision = revision.model_copy(
            update={"approved": True, "loaders": ()}
        )
        approved_record = record.model_copy(
            update={
                "revisions": {revision_hash: approved_revision},
                "enabled": True,
            }
        )
        assert status_for(approved_record) == "missing-capability"
        with pytest.raises(
            erlab.extensions.ExtensionExecutionError, match="is not available"
        ):
            execution.session_loader_call("loader", revision_hash, "load_data")


def test_routine_job_rejects_unavailable_catalog_state(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "scale.py"
    _script(script_path)
    data = xr.DataArray([1.0])

    with manager_context() as manager:
        execution = manager._extensions.execution
        with pytest.raises(
            erlab.extensions.ExtensionExecutionError, match="not enabled"
        ):
            execution._routine_job(
                extension_id="missing",
                revision_hash=None,
                routine_id="scale",
                parameters={},
                input_data=data,
                input_uid="uid",
                input_snapshot="snapshot",
            )

        catalog, revision_hash, _created = manager._extensions.catalog.store.add_script(
            script_path
        )
        with pytest.raises(
            erlab.extensions.ExtensionExecutionError, match="not enabled"
        ):
            execution._routine_job(
                extension_id="scale",
                revision_hash=revision_hash,
                routine_id="scale",
                parameters={},
                input_data=data,
                input_uid="uid",
                input_snapshot="snapshot",
            )
        catalog = _validate_and_enable(
            manager._extensions.catalog.store,
            "scale",
            expected_record_generation=catalog.extensions["scale"].record_generation,
        )
        with pytest.raises(
            erlab.extensions.ExtensionExecutionError,
            match="different source type",
        ):
            execution._routine_job(
                extension_id="scale",
                revision_hash=revision_hash,
                routine_id="scale",
                source_type="environment-package",
                parameters={},
                input_data=data,
                input_uid="uid",
                input_snapshot="snapshot",
            )
        with pytest.raises(
            erlab.extensions.ExtensionExecutionError, match="revision is not available"
        ):
            execution._routine_job(
                extension_id="scale",
                revision_hash="0" * 64,
                routine_id="scale",
                parameters={},
                input_data=data,
                input_uid="uid",
                input_snapshot="snapshot",
            )
        with pytest.raises(
            erlab.extensions.ExtensionExecutionError, match="Routine 'missing'"
        ):
            execution._routine_job(
                extension_id="scale",
                revision_hash=catalog.extensions["scale"].current_revision,
                routine_id="missing",
                parameters={},
                input_data=data,
                input_uid="uid",
                input_snapshot="snapshot",
            )


def test_execution_controller_ignores_unknown_queue_callbacks(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "scale.py"
    _script(script_path)

    with manager_context() as manager:
        execution = manager._extensions.execution
        catalog, revision_hash, _created = manager._extensions.catalog.store.add_script(
            script_path
        )
        catalog = _validate_and_enable(
            manager._extensions.catalog.store,
            "scale",
            expected_record_generation=catalog.extensions["scale"].record_generation,
        )
        job = execution._routine_job(
            extension_id="scale",
            revision_hash=revision_hash,
            routine_id="scale",
            parameters={"scale": 2.0},
            input_data=xr.DataArray([1.0]),
            input_uid="uid",
            input_snapshot="snapshot",
        )

        execution.remove_queued("missing")
        assert execution.queued == ()
        result = extension_execution._ExtensionRoutineResult(
            job=job,
            output=None,
            duration=0.0,
            status="discarded",
        )
        execution._finished(result)
        assert execution.active is None


def test_extension_progress_dialog_removes_only_a_selected_queued_job(
    manager_context,
    tmp_path: pathlib.Path,
    qtbot: pytest.QtBot,
) -> None:
    script_path = tmp_path / "scale.py"
    _script(script_path)

    with manager_context() as manager:
        execution = manager._extensions.execution
        catalog, revision_hash, _created = manager._extensions.catalog.store.add_script(
            script_path
        )
        _validate_and_enable(
            manager._extensions.catalog.store,
            "scale",
            expected_record_generation=catalog.extensions["scale"].record_generation,
        )
        job = execution._routine_job(
            extension_id="scale",
            revision_hash=revision_hash,
            routine_id="scale",
            parameters={"scale": 2.0},
            input_data=xr.DataArray([1.0]),
            input_uid="uid",
            input_snapshot="snapshot",
        )
        dialog = execution._progress_dialog
        removed: list[str] = []
        remove_slot = removed.append
        dialog.remove_requested.connect(remove_slot)
        try:
            dialog.set_jobs(job, (job,))
            dialog._remove_selected()
            dialog.list_widget.setCurrentRow(0)
            dialog._remove_selected()
            dialog.list_widget.setCurrentRow(1)
            with qtbot.waitSignal(dialog.remove_requested, timeout=1000):
                dialog._remove_selected()
            assert removed == [job.job_id]

            execution.show_progress()
            assert dialog.isVisible()
            dialog.hide()
        finally:
            dialog.remove_requested.disconnect(remove_slot)


def test_extension_replay_reports_all_controller_result_states(
    manager_context,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = tmp_path / "scale.py"
    _script(script_path)
    data = xr.DataArray([1.0])

    with manager_context() as manager:
        execution = manager._extensions.execution
        catalog, revision_hash, _created = manager._extensions.catalog.store.add_script(
            script_path
        )
        catalog = _validate_and_enable(
            manager._extensions.catalog.store,
            "scale",
            expected_record_generation=catalog.extensions["scale"].record_generation,
        )
        operation = ExtensionRoutineOperation(
            extension_id="scale",
            revision_hash=revision_hash,
            routine_id="scale",
            extension_name="Scale",
            routine_name="Scale",
            source_type="script",
            function_name="scale",
            source_path=os.fspath(script_path),
            entry_point_group=None,
            entry_point_name=None,
            parameters={"scale": 2.0},
        )

        with pytest.raises(TypeError, match="Expected extension routine"):
            execution.run_operation(typing.cast("typing.Any", object()), data)

        thread_errors: list[Exception] = []

        def run_from_thread() -> None:
            try:
                execution.run_operation(operation, data)
            except Exception as error:
                thread_errors.append(error)

        thread = threading.Thread(target=run_from_thread)
        thread.start()
        thread.join()
        assert len(thread_errors) == 1
        assert "manager thread" in str(thread_errors[0])

        execution._accepting = False
        with pytest.raises(
            erlab.extensions.ExtensionExecutionError, match="shutting down"
        ):
            execution.run_operation(operation, data)
        execution._accepting = True

        execution._routine_waiters["existing"] = _ExtensionRoutineWaiter(
            QtCore.QEventLoop()
        )
        with pytest.raises(
            erlab.extensions.ExtensionExecutionError, match="replay is in progress"
        ):
            execution.run_operation(operation, data)
        execution._routine_waiters.clear()

        class FakeEventLoop:
            class ProcessEventsFlag:
                ExcludeUserInputEvents = 0

            def exec(self, _flags: typing.Any) -> None:
                return None

            def quit(self) -> None:
                return None

        def set_result(
            status: typing.Literal["success", "failed", "discarded"] | None,
        ) -> None:
            job = execution._pending.pop()
            if status is None:
                return
            waiter = execution._routine_waiters[job.job_id]
            waiter.result = extension_execution._ExtensionRoutineResult(
                job=job,
                output=xr.DataArray([2.0]) if status == "success" else None,
                duration=0.0,
                status=status,
            )

        with monkeypatch.context() as patch_context:
            patch_context.setattr(
                extension_execution.QtCore, "QEventLoop", FakeEventLoop
            )
            patch_context.setattr(execution, "_start_next", lambda: set_result(None))
            with pytest.raises(
                erlab.extensions.ExtensionExecutionError, match="without a result"
            ):
                execution.run_operation(operation, data)

            patch_context.setattr(
                execution, "_start_next", lambda: set_result("discarded")
            )
            with pytest.raises(
                erlab.extensions.ExtensionExecutionError, match="not enabled"
            ):
                execution.run_operation(operation, data)

            patch_context.setattr(
                execution, "_start_next", lambda: set_result("failed")
            )
            with pytest.raises(
                erlab.extensions.ExtensionExecutionError, match="could not complete"
            ):
                execution.run_operation(operation, data)

            patch_context.setattr(
                execution, "_start_next", lambda: set_result("success")
            )
            xr.testing.assert_identical(
                execution.run_operation(operation, data), xr.DataArray([2.0])
            )


def test_execution_refresh_and_shutdown_are_safe_after_qt_teardown(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with manager_context() as manager:
        execution = manager._extensions.execution
        monkeypatch.setattr(
            erlab.interactive.utils, "qt_is_valid", lambda *_objects: False
        )
        execution._refresh_progress()
        execution.shutdown()
        execution.shutdown()
        assert execution._shutdown_complete


@pytest.mark.parametrize(
    "corruption",
    ["extension-key", "revision-key", "current", "source-type", "object-name"],
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
    elif corruption == "object-name":
        record["revisions"][revision]["object_name"] = "../outside.py"
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


def test_add_script_rejects_a_different_same_stem_source(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    first_path = tmp_path / "first" / "analysis.py"
    second_path = tmp_path / "second" / "analysis.py"
    first_path.parent.mkdir()
    second_path.parent.mkdir()
    _script(first_path)
    _script(second_path, "data + scale")
    review_calls: list[None] = []
    dialogs: list[None] = []

    with manager_context() as manager:
        before, _revision, _created = manager._extensions.catalog.store.add_script(
            first_path,
            metadata=_ExtensionMetadata(author="First Author"),
        )
        manager._extensions.catalog.refresh()
        monkeypatch.setattr(
            extension_controller._SourceReviewDialog,
            "exec",
            lambda _dialog: review_calls.append(None) or 1,
        )
        monkeypatch.setattr(
            erlab.interactive.utils.MessageDialog,
            "critical",
            lambda *_args, **_kwargs: dialogs.append(None),
        )

        assert not manager._extensions._review_and_add(second_path)
        after = manager._extensions.catalog.store.read()

    assert review_calls == []
    assert dialogs == [None]
    assert after == before


def test_unchanged_add_script_preserves_existing_metadata(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "analysis.py"
    _script(script_path)
    metadata = _ExtensionMetadata(
        author="Lab Author",
        contact="lab@example.org",
        project_url="https://example.org/lab",
        change_summary="Initial revision",
        changelog="Initial changelog",
    )

    with manager_context() as manager:
        before, revision, _created = manager._extensions.catalog.store.add_script(
            script_path,
            metadata=metadata,
        )
        manager._extensions.catalog.refresh()
        monkeypatch.setattr(
            extension_controller._SourceReviewDialog,
            "exec",
            lambda _dialog: 1,
        )

        assert manager._extensions._review_and_add(script_path)
        after = manager._extensions.catalog.store.read()

    record = after.extensions["analysis"]
    assert record.metadata == metadata
    assert tuple(record.revisions) == (revision,)
    assert record.record_generation == before.extensions["analysis"].record_generation


def test_identical_same_stem_source_can_relocate_an_extension(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    first_path = tmp_path / "first" / "analysis.py"
    second_path = tmp_path / "second" / "analysis.py"
    first_path.parent.mkdir()
    second_path.parent.mkdir()
    source = _script(first_path)
    second_path.write_bytes(source)
    metadata = _ExtensionMetadata(author="Lab Author")

    with manager_context() as manager:
        _before, revision, _created = manager._extensions.catalog.store.add_script(
            first_path,
            metadata=metadata,
        )
        manager._extensions.catalog.refresh()
        monkeypatch.setattr(
            extension_controller._SourceReviewDialog,
            "exec",
            lambda _dialog: 1,
        )

        assert manager._extensions._review_and_add(second_path)
        record = manager._extensions.catalog.store.read().extensions["analysis"]

    assert record.current_revision == revision
    assert tuple(record.revisions) == (revision,)
    assert record.revisions[revision].source_path == os.fspath(second_path.resolve())
    assert record.metadata == metadata


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


def test_restored_revision_updates_script_source_location(
    tmp_path: pathlib.Path,
) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    original_path = tmp_path / "original" / "scale.py"
    original_path.parent.mkdir()
    first_source = _script(original_path)
    _catalog, first_revision, _created = store.add_script(original_path)

    _script(original_path, "data + scale")
    store.add_script(original_path)

    relocated_path = tmp_path / "relocated" / "scale.py"
    relocated_path.parent.mkdir()
    relocated_path.write_bytes(first_source)
    restored, restored_revision, created = store.add_script(relocated_path)

    record = restored.extensions["scale"]
    assert created
    assert restored_revision == first_revision
    assert record.current_revision == first_revision
    assert record.revisions[first_revision].source_path == os.fspath(
        relocated_path.resolve()
    )


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


def test_embedded_source_preserves_workspace_modification_time(
    tmp_path: pathlib.Path,
) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    script_path = tmp_path / "scale.py"
    source = _script(script_path)
    revision = hashlib.sha256(source).hexdigest()
    source_modified_at = "2025-06-01T12:34:56+00:00"

    catalog = store.add_embedded_script(
        source,
        extension_id="scale",
        expected_revision=revision,
        name="Scale",
        metadata=_ExtensionMetadata(),
        source_modified_at=source_modified_at,
    )

    assert (
        catalog.extensions["scale"].revisions[revision].source_modified_at
        == source_modified_at
    )


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


def test_extension_routine_reloadability_requires_ready_exact_revision() -> None:
    revision = "a" * 64
    operation = ExtensionRoutineOperation(
        extension_id="lab",
        revision_hash=revision,
        routine_id="normalize",
        extension_name="Lab",
        routine_name="Normalize",
        source_type="script",
        function_name="normalize",
        source_path="extension.py",
        entry_point_group=None,
        entry_point_name=None,
        parameters={},
    )
    spec = ToolProvenanceSpec(
        kind="script",
        start_label="Create data",
        seed_code="derived = xr.DataArray([1.0])",
        active_name="derived",
        operations=(operation,),
    )
    calls: list[tuple[str, str, str, str, str | None]] = []

    def ready(
        extension_id: str,
        revision_hash: str,
        capability_kind: str,
        capability_id: str,
        source_type: str | None,
    ) -> typing.Literal["ready"]:
        calls.append(
            (
                extension_id,
                revision_hash,
                capability_kind,
                capability_id,
                source_type,
            )
        )
        return "ready"

    assert can_reload_without_trust(spec, extension_status_resolver=ready)
    assert calls == [("lab", revision, "routine", "normalize", "script")]
    assert not can_reload_without_trust(
        spec,
        extension_status_resolver=lambda *_args: "disabled",
    )


def test_managed_reload_reason_uses_the_manager_extension_state(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "a" * 64
    operation = ExtensionRoutineOperation(
        extension_id="lab",
        revision_hash=revision,
        routine_id="normalize",
        extension_name="Lab",
        routine_name="Normalize",
        source_type="script",
        function_name="normalize",
        source_path="extension.py",
        entry_point_group=None,
        entry_point_name=None,
        parameters={},
    )
    spec = ToolProvenanceSpec(
        kind="script",
        start_label="Create data",
        seed_code="derived = xr.DataArray([1.0])",
        active_name="derived",
        operations=(operation,),
    )
    monkeypatch.setattr(imagetool_viewer, "_capability_status", lambda *_args: "ready")

    with manager_context() as manager:
        monkeypatch.setattr(
            manager._extensions, "capability_status", lambda *_args: "disabled"
        )
        tool = erlab.interactive.imagetool.ImageTool(
            xr.DataArray([1.0]), _in_manager=True
        )
        manager.add_imagetool(tool, show=False, provenance_spec=spec)

        assert tool.slicer_area._provenance_reload_unavailable_reason() is not None


def test_persisted_extension_parameters_reject_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        ExtensionRoutineOperation(
            extension_id="scale",
            revision_hash="a" * 64,
            routine_id="scale",
            extension_name="Scale",
            routine_name="Scale",
            source_type="script",
            function_name="scale",
            source_path="scale.py",
            entry_point_group=None,
            entry_point_name=None,
            parameters={"scale": float("inf")},
        )

    with pytest.raises(ValueError, match="must be finite"):
        FileReplayCall(
            kind="extension_loader",
            target="scale",
            revision="a" * 64,
            capability_id="load_scale",
            extension_source_type="script",
            kwargs={"scale": float("nan")},
            selection=FileDataSelection(kind="dataarray"),
        )


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


def test_environment_loader_entry_point_name_can_differ_from_loader_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    class _AliasedLoader(erlab.io.dataloader.LoaderBase):
        name = "extension_loader_actual_name"
        extensions: typing.ClassVar[set[str]] = {".alias-test"}
        skip_validate = True

        @property
        def file_dialog_methods(self):
            return {"Aliased Extension Data (*.alias-test)": (self.load, {})}

        def load_single(self, file_path, without_values=False):
            del file_path, without_values
            return xr.DataArray([1.0])

    class _EntryPoint:
        group = "erlab.io.loaders"
        name = "package_alias"
        value = "lab_package:AliasedLoader"
        dist = None

        @staticmethod
        def load():
            return _AliasedLoader

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
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    catalog = store.refresh_environment_packages()
    extension_id = "environment.erlab.io.loaders.package_alias"

    catalog = _validate_and_enable(
        store,
        extension_id,
        expected_record_generation=catalog.extensions[extension_id].record_generation,
    )

    record = catalog.extensions[extension_id]
    assert record.enabled
    assert record.revisions[record.current_revision].loaders[0].id == (
        "extension_loader_actual_name"
    )


def test_environment_loader_name_conflict_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    class _FirstLoader(erlab.io.dataloader.LoaderBase):
        name = "shared_environment_loader"
        extensions: typing.ClassVar[set[str]] = {".first-loader-test"}
        skip_validate = True

        @property
        def file_dialog_methods(self):
            return {"First Environment Data (*.first-loader-test)": (self.load, {})}

        def load_single(self, file_path, without_values=False):
            del file_path, without_values
            return xr.DataArray([1.0])

    class _SecondLoader(erlab.io.dataloader.LoaderBase):
        name = "shared_environment_loader"
        extensions: typing.ClassVar[set[str]] = {".second-loader-test"}
        skip_validate = True

        @property
        def file_dialog_methods(self):
            return {"Second Environment Data (*.second-loader-test)": (self.load, {})}

        def load_single(self, file_path, without_values=False):
            del file_path, without_values
            return xr.DataArray([2.0])

    class _EntryPoint:
        group = "erlab.io.loaders"
        dist = None

        def __init__(self, name: str, loader_type: type) -> None:
            self.name = name
            self.value = f"lab_package:{loader_type.__name__}"
            self._loader_type = loader_type

        def load(self):
            return self._loader_type

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
        lambda: _EntryPoints(
            (
                _EntryPoint("first_package", _FirstLoader),
                _EntryPoint("second_package", _SecondLoader),
            )
        ),
    )
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    catalog = store.refresh_environment_packages()
    first_id = "environment.erlab.io.loaders.first_package"
    second_id = "environment.erlab.io.loaders.second_package"

    catalog = _validate_and_enable(
        store,
        first_id,
        expected_record_generation=catalog.extensions[first_id].record_generation,
    )
    with pytest.raises(
        _ExtensionCatalogConflictError,
        match=r"conflicts with enabled extension .* for loader names",
    ):
        _validate_and_enable(
            store,
            second_id,
            expected_record_generation=(
                catalog.extensions[second_id].record_generation
            ),
        )

    persisted = store.read()
    assert persisted.extensions[first_id].enabled
    assert not persisted.extensions[second_id].enabled


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


def test_generated_external_loader_verifies_entry_point_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    revision = "a" * 64
    descriptor = erlab.extensions.LoaderDescriptor(
        id="external",
        name="External",
        category="Environment",
        summary="",
        function_name="load",
    )
    call = _ExtensionLoaderCall(
        manager_session_id="manager",
        catalog_generation=1,
        extension_id="environment.erlab.io.loaders.external",
        extension_name="External",
        revision_hash=revision,
        loader_id="external",
        descriptor=descriptor,
        source_path=None,
        source_type="environment-package",
        executor=lambda *_args: xr.DataArray([0.0]),
        entry_point_group="erlab.io.loaders",
        entry_point_name="external",
        loader_method=f"{__name__}._generated_external_loader",
    )
    resolved = _resolve_load_func(
        (call, {"scale": 3.0}, FileDataSelection(kind="dataarray"))
    )
    if resolved is None:
        raise RuntimeError("The extension loader did not resolve")
    path = tmp_path / "value.txt"
    path.write_text("4")
    calls: list[tuple[str, str, str, str]] = []

    class Loaded:
        def __init__(self, group: str, name: str, expected_revision: str) -> None:
            self.identity = (group, name, expected_revision)

        def resolve_loader(self, method: str):
            calls.append((*self.identity, method))
            return _generated_external_loader

    monkeypatch.setattr(
        erlab.extensions,
        "load_entry_point",
        lambda group, name, *, expected_revision: Loaded(
            group, name, expected_revision
        ),
    )
    code = resolved.load_code(path, assign="loaded")
    if code is None:
        raise RuntimeError("The extension loader did not generate code")
    namespace: dict[str, typing.Any] = {}

    exec(code, namespace)  # noqa: S102

    assert calls == [
        (
            "erlab.io.loaders",
            "external",
            revision,
            f"{__name__}._generated_external_loader",
        )
    ]
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


@pytest.mark.parametrize(
    ("url", "uri_path"),
    [
        ("file:///C:/lab/project%20files", "/C:/lab/project%20files"),
        (
            "file://server/share/project%20files",
            "//server/share/project%20files",
        ),
    ],
)
def test_editable_source_fingerprint_uses_platform_file_url_conversion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    url: str,
    uri_path: str,
) -> None:
    (tmp_path / "plugin.py").write_text("VALUE = 1\n")
    converted: list[str] = []

    def url2pathname(value: str) -> str:
        converted.append(value)
        return str(tmp_path)

    monkeypatch.setattr(
        extension_entry_points.urllib.request, "url2pathname", url2pathname
    )

    fingerprint = extension_entry_points._editable_source_fingerprint({"url": url})

    assert fingerprint
    assert converted == [uri_path]


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


def test_editable_package_availability_uses_refresh_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    source = tmp_path / "src" / "lab_package" / "plugin.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n")

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
        name = "cached_editable"
        value = "lab_package.plugin:extension"
        dist = _Distribution()

    class _EntryPoints(tuple):
        def select(self, *, group: str):
            return tuple(entry for entry in self if entry.group == group)

    monkeypatch.setattr(
        extension_catalog.importlib.metadata,
        "entry_points",
        lambda: _EntryPoints((_EntryPoint(),)),
    )
    fingerprint_calls = 0
    fingerprint = extension_entry_points._editable_source_fingerprint

    def counted_fingerprint(direct_url):
        nonlocal fingerprint_calls
        fingerprint_calls += 1
        return fingerprint(direct_url)

    monkeypatch.setattr(
        extension_entry_points,
        "_editable_source_fingerprint",
        counted_fingerprint,
    )
    store = _ExtensionCatalogStore(tmp_path / "catalog")

    catalog = store.refresh_environment_packages()
    extension_id = "environment.erlab.extensions.cached_editable"
    record = catalog.extensions[extension_id]
    revision_hash = record.current_revision
    assert fingerprint_calls == 1

    for _ in range(3):
        assert store.revision_available(record, revision_hash)
        assert (
            store.capability_status(
                extension_id,
                revision_hash,
                "routine",
                "extension",
            )
            == "approval-required"
        )
    assert fingerprint_calls == 1

    source.write_text("VALUE = 2\n")
    catalog = store.refresh_environment_packages()
    record = catalog.extensions[extension_id]
    assert record.current_revision != revision_hash
    assert fingerprint_calls == 2
    assert store.revision_available(record, record.current_revision)
    assert fingerprint_calls == 2


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


def test_environment_refresh_disables_an_unavailable_package_without_removing_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    class _EntryPoint:
        group = "erlab.extensions"
        name = "temporarily_unavailable"
        value = "lab_package:extension"
        dist = None

    class _EntryPoints(tuple):
        def select(self, *, group: str):
            return tuple(entry for entry in self if entry.group == group)

    entry_point = _EntryPoint()
    available = True
    monkeypatch.setattr(
        extension_catalog.importlib.metadata,
        "entry_points",
        lambda: _EntryPoints((entry_point,)) if available else _EntryPoints(),
    )
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    catalog = store.refresh_environment_packages()
    extension_id = "environment.erlab.extensions.temporarily_unavailable"
    record = catalog.extensions[extension_id]
    revision = record.current_revision
    descriptor = erlab.extensions.RoutineDescriptor(
        id="extension",
        name="Extension",
        category="Lab",
        summary="",
        function_name="extension",
    )
    catalog = store.enable_validated_revision(
        extension_id,
        revision_hash=revision,
        expected_record_generation=record.record_generation,
        routines=(descriptor,),
        loaders=(),
        loader_always_single=None,
        loader_dialog_methods=(),
    )

    available = False
    catalog = store.refresh_environment_packages()
    record = catalog.extensions[extension_id]
    assert not record.enabled
    assert not record.removed
    assert record.current_revision == revision

    available = True
    catalog = store.refresh_environment_packages()
    record = catalog.extensions[extension_id]
    assert not record.enabled
    assert not record.removed
    assert record.current_revision == revision
    assert store.revision_available(record, revision)


def test_packaged_manager_does_not_hide_bundled_loader_names(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
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
    updater_settings = QtCore.QSettings(
        str(tmp_path / "updater.ini"), QtCore.QSettings.Format.IniFormat
    )
    updater_settings.setValue("version_before_update", erlab.__version__)
    monkeypatch.setattr(
        imagetool_manager, "_get_updater_settings", lambda: updater_settings
    )

    def unexpected_update_notice(*_args: object, **_kwargs: object) -> typing.Never:
        raise AssertionError("The isolated updater state must suppress the notice")

    monkeypatch.setattr(
        imagetool_manager.ImageToolManager, "updated", unexpected_update_notice
    )

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


def test_workspace_requirement_rejects_embedded_environment_package() -> None:
    with pytest.raises(ValueError, match="cannot embed source"):
        _WorkspaceExtensionRequirement(
            extension_id="lab",
            capability_id="routine",
            capability_kind="routine",
            revision_hash="a" * 64,
            extension_api_version=1,
            source_type="environment-package",
            embedded_object_id=f"extension-{'a' * 64}",
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
        manifest = store.current_generation().manifest
        node_object_id = manifest["nodes"][0]["payload_object_id"]
        assert node_object_id != object_id
        node_group = store.h5_file[
            workspace_store.WorkspaceStore.object_path(node_object_id)
        ]
        assert node_group.attrs.get("erlab_object_kind") != (
            "extension-python-source-v1"
        )
    restored, kind = workspace_storage._read_workspace_blob(workspace_path, object_id)
    assert restored == source
    assert kind == "extension-python-source-v1"


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
        source_modified_at = (
            manager._extensions.catalog.model.extensions["scale"]
            .revisions[revision]
            .source_modified_at
        )

        manager._workspace_controller.saving._save_workspace_document(workspace_path)

    attrs = workspace_arrays._read_workspace_root_attrs_h5py(workspace_path)
    manifest = workspace_format._workspace_manifest_from_attrs(attrs)
    requirements = manifest["extension_requirements"]
    assert len(requirements) == 1
    assert requirements[0]["revision_hash"] == revision
    assert requirements[0]["metadata_snapshot"]["source_modified_at"] == (
        source_modified_at
    )
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
        object_id = f"extension-{revision}"
        manager._extensions.set_workspace_requirements(
            (),
            unresolved_embedded_objects={
                object_id: (b"corrupt", "extension-python-source-v1")
            },
        )

        manager._workspace_controller.saving._save_workspace_document(workspace_path)

    attrs = workspace_arrays._read_workspace_root_attrs_h5py(workspace_path)
    manifest = workspace_format._workspace_manifest_from_attrs(attrs)
    assert manifest["extension_requirements"][0]["embedded_object_id"] == object_id
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


def test_save_as_preserves_an_existing_embedding_when_policy_is_never(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "preserved.py"
    source = _script(script_path)
    workspace_path = tmp_path / "preserved.itws"

    with manager_context() as manager:
        catalog, revision, _created = manager._extensions.catalog.store.add_script(
            script_path
        )
        catalog = _validate_and_enable(
            manager._extensions.catalog.store,
            "preserved",
            expected_record_generation=(
                catalog.extensions["preserved"].record_generation
            ),
        )
        catalog = manager._extensions.catalog.store.update_record(
            "preserved",
            expected_record_generation=(
                catalog.extensions["preserved"].record_generation
            ),
            embed_policy="never",
        )
        manager._extensions.catalog.refresh()
        operation = ExtensionRoutineOperation(
            extension_id="preserved",
            revision_hash=revision,
            routine_id="scale",
            extension_name="Preserved",
            routine_name="Scale",
            source_type="script",
            function_name="scale",
            source_path=str(
                manager._extensions.catalog.store.source_path("preserved", revision)
            ),
            entry_point_group=None,
            entry_point_name=None,
            parameters={},
        )
        index = manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(xr.DataArray([1.0])),
            show=False,
            provenance_spec=full_data(operation),
        )
        node = manager._node_for_target(index)
        object_id = f"extension-{revision}"
        manager._extensions.set_workspace_requirements(
            (
                _WorkspaceExtensionRequirement(
                    extension_id="preserved",
                    capability_id="scale",
                    capability_kind="routine",
                    revision_hash=revision,
                    extension_api_version=1,
                    source_type="script",
                    metadata_snapshot={
                        "source_modified_at": (
                            catalog.extensions["preserved"]
                            .revisions[revision]
                            .source_modified_at
                        )
                    },
                    embedded_object_id=object_id,
                    referencing_nodes=(node.uid,),
                ),
            ),
            embedded_sources={("preserved", revision): source},
        )
        manager._extensions.catalog.store.source_path("preserved", revision).unlink()

        manager._workspace_controller.saving._save_workspace_document(workspace_path)

    attrs = workspace_arrays._read_workspace_root_attrs_h5py(workspace_path)
    manifest = workspace_format._workspace_manifest_from_attrs(attrs)
    assert manifest["extension_requirements"][0]["embedded_object_id"] == object_id
    restored, kind = workspace_storage._read_workspace_blob(workspace_path, object_id)
    assert restored == source
    assert kind == "extension-python-source-v1"


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

        manager._extensions.set_workspace_requirements(
            (
                requirement.model_copy(
                    update={
                        "source_type": "environment-package",
                        "embedded_object_id": None,
                    }
                ),
            ),
            embedded_sources={("workspace-only", revision): source},
        )
        assert manager._extensions.resolved_workspace_requirements()[0].state == (
            "missing"
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


def test_embedded_approval_is_local_to_one_manager_session(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    source = b"""from pathlib import Path
import xarray as xr
from erlab.extensions import loader, routine

@routine(name="Scale", category="Lab")
def scale(data: xr.DataArray, scale: float = 2.0) -> xr.DataArray:
    return data * scale

@loader(name="Lab Data", extensions=(".txt",), category="Lab")
def load_data(path: Path) -> xr.DataArray:
    return xr.DataArray([float(path.read_text())])
"""
    revision = hashlib.sha256(source).hexdigest()
    requirement = _WorkspaceExtensionRequirement(
        extension_id="workspace-session",
        capability_id="scale",
        capability_kind="routine",
        revision_hash=revision,
        extension_api_version=1,
        source_type="script",
        metadata_snapshot={"extension_name": "Workspace Session"},
        embedded_object_id=f"extension-{revision}",
    )
    monkeypatch.setattr(
        extension_controller._SourceReviewDialog,
        "exec",
        lambda _dialog: QtWidgets.QDialog.DialogCode.Accepted,
    )
    data_path = tmp_path / "value.txt"
    data_path.write_text("4")

    with manager_context() as manager:
        manager._extensions.set_workspace_requirements(
            (requirement,),
            embedded_sources={(requirement.extension_id, revision): source},
        )
        manager._extensions._approve_embedded_script(requirement.extension_id, revision)

        assert requirement.extension_id not in (
            manager._extensions.catalog.store.read().extensions
        )
        assert manager._extensions.resolved_workspace_requirements()[0].state == (
            "ready"
        )
        operation = ExtensionRoutineOperation(
            extension_id=requirement.extension_id,
            revision_hash=revision,
            routine_id="scale",
            extension_name="Workspace Session",
            routine_name="Scale",
            source_type="script",
            function_name="scale",
            source_path="workspace_session.py",
            entry_point_group=None,
            entry_point_name=None,
            parameters={"scale": 3.0},
        )
        xr.testing.assert_identical(
            manager._extensions.execution.run_operation(
                operation, xr.DataArray([1.0, 2.0])
            ),
            xr.DataArray([3.0, 6.0]),
        )
        load_source = FileLoadSource(
            path=str(data_path),
            loader_label="Lab Data",
            loader_text="workspace-session:load_data",
            kwargs_text="",
            replay_call=FileReplayCall(
                kind="extension_loader",
                target=requirement.extension_id,
                revision=revision,
                capability_id="load_data",
                extension_source_type="script",
                selection=FileDataSelection(kind="dataarray"),
            ),
        )
        xr.testing.assert_identical(
            manager._extensions.replay_loader(load_source), xr.DataArray([4.0])
        )
        provenance_spec = file_load(
            start_label="Load session data",
            seed_code="derived = xr.DataArray([4.0])",
            file_load_source=load_source,
        )
        assert (
            file_load_source_status(
                provenance_spec,
                extension_status_resolver=manager._extensions.capability_status,
            )
            == "loadable"
        )
        index = manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(xr.DataArray([4.0])),
            show=False,
            provenance_spec=provenance_spec,
        )
        slicer_area = manager.get_imagetool(index).slicer_area
        data_path.write_text("5")
        assert slicer_area._provenance_reloadable()
        xr.testing.assert_identical(
            slicer_area._fetch_for_provenance_reload(), xr.DataArray([5.0])
        )
        data_path.write_text("6")
        assert slicer_area._reload()
        xr.testing.assert_identical(slicer_area._data, xr.DataArray([6.0]))
        session_directory = pathlib.Path(
            manager._extensions.execution._session_directory.name
        )

    assert not session_directory.exists()
    with manager_context() as manager:
        manager._extensions.set_workspace_requirements(
            (requirement,),
            embedded_sources={(requirement.extension_id, revision): source},
        )
        assert manager._extensions.resolved_workspace_requirements()[0].state == (
            "approval-required"
        )


def test_embedded_revision_replaces_an_unusable_global_copy(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "shared.py"
    source = b"""from pathlib import Path
import xarray as xr
from erlab.extensions import loader, routine

@routine()
def scale(data: xr.DataArray, factor: float = 2.0) -> xr.DataArray:
    return data * factor

@loader(extensions=(".txt",))
def load_data(path: Path) -> xr.DataArray:
    return xr.DataArray([float(path.read_text())])
"""
    script_path.write_bytes(source)
    revision = hashlib.sha256(source).hexdigest()
    data_path = tmp_path / "value.txt"
    data_path.write_text("4")

    with manager_context() as manager:
        catalog, stored_revision, _created = (
            manager._extensions.catalog.store.add_script(script_path)
        )
        assert stored_revision == revision
        _validate_and_enable(
            manager._extensions.catalog.store,
            "shared",
            expected_record_generation=(catalog.extensions["shared"].record_generation),
        )
        manager._extensions.catalog.refresh()
        manager._extensions.execution.approve_session_script(
            source,
            extension_id="shared",
            revision_hash=revision,
            name="Shared",
            metadata=_ExtensionMetadata(),
            source_modified_at=None,
        )
        manager._extensions.catalog.store.source_path("shared", revision).write_bytes(
            b"corrupt global source"
        )

        operation = ExtensionRoutineOperation(
            extension_id="shared",
            revision_hash=revision,
            routine_id="scale",
            extension_name="Shared",
            routine_name="Scale",
            source_type="script",
            function_name="scale",
            source_path=str(script_path),
            entry_point_group=None,
            entry_point_name=None,
            parameters={"factor": 3.0},
        )
        xr.testing.assert_identical(
            manager._extensions.execution.run_operation(
                operation, xr.DataArray([1.0, 2.0])
            ),
            xr.DataArray([3.0, 6.0]),
        )
        load_source = FileLoadSource(
            path=str(data_path),
            loader_label="Shared loader",
            loader_text="shared:load_data",
            kwargs_text="",
            replay_call=FileReplayCall(
                kind="extension_loader",
                target="shared",
                revision=revision,
                capability_id="load_data",
                extension_source_type="script",
                selection=FileDataSelection(kind="dataarray"),
            ),
        )
        xr.testing.assert_identical(
            manager._extensions.replay_loader(load_source), xr.DataArray([4.0])
        )


def test_canceling_embedded_approval_keeps_source_unapproved(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "cancelled.py"
    source = _script(script_path)
    revision = hashlib.sha256(source).hexdigest()
    requirement = _WorkspaceExtensionRequirement(
        extension_id="cancelled",
        capability_id="scale",
        capability_kind="routine",
        revision_hash=revision,
        extension_api_version=1,
        source_type="script",
        embedded_object_id=f"extension-{revision}",
    )
    monkeypatch.setattr(
        extension_controller._SourceReviewDialog,
        "exec",
        lambda _dialog: QtWidgets.QDialog.DialogCode.Rejected,
    )

    with manager_context() as manager:
        manager._extensions.set_workspace_requirements(
            (requirement,),
            embedded_sources={(requirement.extension_id, revision): source},
        )
        manager._extensions._approve_embedded_script(requirement.extension_id, revision)

        assert requirement.extension_id not in (
            manager._extensions.catalog.store.read().extensions
        )
        assert (
            manager._extensions.execution.session_capability_status(
                requirement.extension_id,
                revision,
                "routine",
                requirement.capability_id,
            )
            is None
        )
        assert manager._extensions.resolved_workspace_requirements()[0].state == (
            "approval-required"
        )


def test_embedded_approval_can_be_remembered_globally(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "remembered.py"
    source = _script(script_path)
    revision = hashlib.sha256(source).hexdigest()
    requirement = _WorkspaceExtensionRequirement(
        extension_id="remembered",
        capability_id="scale",
        capability_kind="routine",
        revision_hash=revision,
        extension_api_version=1,
        source_type="script",
        embedded_object_id=f"extension-{revision}",
    )

    def remember(dialog) -> QtWidgets.QDialog.DialogCode:
        control = dialog.findChild(
            QtWidgets.QCheckBox, "manager_extension_remember_approval"
        )
        if control is None:
            raise RuntimeError("The approval scope control is unavailable")
        control.setChecked(True)
        return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(
        extension_controller._SourceReviewDialog,
        "exec",
        remember,
    )
    with manager_context() as manager:
        manager._extensions.set_workspace_requirements(
            (requirement,),
            embedded_sources={(requirement.extension_id, revision): source},
        )
        manager._extensions._approve_embedded_script(requirement.extension_id, revision)

        record = manager._extensions.catalog.store.read().extensions["remembered"]
        assert record.enabled
        assert record.revisions[revision].approved

    with manager_context() as manager:
        manager._extensions.set_workspace_requirements(
            (requirement,),
            embedded_sources={(requirement.extension_id, revision): source},
        )
        assert manager._extensions.resolved_workspace_requirements()[0].state == (
            "ready"
        )


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
                    extension_source_type="script",
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
                    extension_source_type="script",
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
                extension_source_type="environment-package",
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


def test_collecting_requirements_merges_duplicate_loaded_capability(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    source = b"# embedded extension source\n"
    revision = hashlib.sha256(source).hexdigest()
    workspace_path = tmp_path / "merged-requirements.itws"
    operation = ExtensionRoutineOperation(
        extension_id="shared-routines",
        revision_hash=revision,
        routine_id="normalize",
        extension_name="Shared Routines",
        routine_name="Normalize",
        source_type="script",
        function_name="normalize",
        source_path="shared_routines.py",
        entry_point_group=None,
        entry_point_name=None,
        parameters={},
    )

    with manager_context() as manager:
        index = manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(xr.DataArray([1.0])),
            show=False,
            provenance_spec=full_data(operation),
        )
        loaded_uid = manager._node_for_target(index).uid
        base = _WorkspaceExtensionRequirement(
            extension_id="shared-routines",
            capability_id="normalize",
            capability_kind="routine",
            revision_hash=revision,
            extension_api_version=1,
            source_type="script",
            metadata_snapshot={"author": "Existing Author"},
            referencing_nodes=("unresolved-existing",),
        )
        incoming = base.model_copy(
            update={
                "metadata_snapshot": {"contact": "incoming@example.org"},
                "embedded_object_id": f"extension-{revision}",
                "referencing_nodes": (loaded_uid,),
            }
        )
        manager._extensions.set_workspace_requirements(
            (base, incoming),
            embedded_sources={(base.extension_id, revision): source},
        )

        collected = manager._extensions.collect_workspace_requirements()
        manager._workspace_controller.saving._save_workspace_document(workspace_path)

    assert len(collected) == 1
    assert collected[0].referencing_nodes == (loaded_uid, "unresolved-existing")
    assert collected[0].metadata_snapshot == {
        "author": "Existing Author",
        "contact": "incoming@example.org",
    }
    assert collected[0].embedded_object_id == f"extension-{revision}"
    attrs = workspace_arrays._read_workspace_root_attrs_h5py(workspace_path)
    manifest = workspace_format._workspace_manifest_from_attrs(attrs)
    assert len(manifest["extension_requirements"]) == 1
    assert manifest["extension_requirements"][0]["referencing_nodes"] == [
        loaded_uid,
        "unresolved-existing",
    ]


def test_collecting_requirements_merges_duplicate_unresolved_loaders(
    manager_context,
) -> None:
    revision = "1" * 64
    base = _WorkspaceExtensionRequirement(
        extension_id="shared-loaders",
        capability_id="load_data",
        capability_kind="loader",
        revision_hash=revision,
        extension_api_version=1,
        source_type="script",
        referencing_nodes=("unresolved-first",),
        file_sources=("first.dat",),
    )
    incoming = base.model_copy(
        update={
            "referencing_nodes": ("unresolved-second",),
            "file_sources": ("second.dat",),
        }
    )

    with manager_context() as manager:
        manager._extensions.set_workspace_requirements((base, incoming))

        collected = manager._extensions.collect_workspace_requirements()

    assert len(collected) == 1
    assert collected[0].referencing_nodes == (
        "unresolved-first",
        "unresolved-second",
    )
    assert collected[0].file_sources == ("first.dat", "second.dat")


def test_collecting_requirements_keeps_workspace_source_identity(
    manager_context,
) -> None:
    revision = "d" * 64
    source_modified_at = "2025-06-01T12:34:56+00:00"
    metadata_snapshot = {
        "extension_name": "Workspace Script",
        "routine_name": "Normalize",
        "author": "Workspace Author",
        "contact": "workspace@example.org",
        "project_url": "https://example.org/workspace",
        "change_summary": "Workspace revision",
        "changelog": "Workspace changelog",
        "source_modified_at": source_modified_at,
    }
    operation = ExtensionRoutineOperation(
        extension_id="shared-extension",
        revision_hash=revision,
        routine_id="normalize",
        extension_name="Shared Extension",
        routine_name="Normalize",
        source_type="script",
        function_name="normalize",
        source_path="shared_extension.py",
        entry_point_group=None,
        entry_point_name=None,
        parameters={},
    )
    environment_record = _ExtensionRecord(
        id="shared-extension",
        name="Local Package",
        source_type="environment-package",
        metadata=_ExtensionMetadata(
            author="Local Author",
            contact="local@example.org",
            project_url="https://example.org/local",
            change_summary="Local revision",
            changelog="Local changelog",
        ),
        current_revision=revision,
        revisions={
            revision: _ExtensionRevision(
                source_hash=revision,
                object_name="lab_package:extension",
                created_at="2026-01-01T00:00:00+00:00",
                entry_point_group="erlab.extensions",
                entry_point_name="extension",
                entry_point_value="lab_package:extension",
            )
        },
    )

    with manager_context() as manager:
        manager._extensions.catalog.store.mutate(
            None,
            lambda catalog: catalog.model_copy(
                update={"extensions": {environment_record.id: environment_record}}
            ),
        )
        manager._extensions.catalog.refresh()
        index = manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(xr.DataArray([1.0])),
            show=False,
            provenance_spec=full_data(operation),
        )
        node = manager._node_for_target(index)
        object_id = f"extension-{revision}"
        manager._extensions.set_workspace_requirements(
            (
                _WorkspaceExtensionRequirement(
                    extension_id="shared-extension",
                    capability_id="normalize",
                    capability_kind="routine",
                    revision_hash=revision,
                    extension_api_version=1,
                    source_type="script",
                    metadata_snapshot=metadata_snapshot,
                    embedded_object_id=object_id,
                    referencing_nodes=(node.uid,),
                ),
            )
        )

        collected = manager._extensions.collect_workspace_requirements()
        resolved = manager._extensions.resolved_workspace_requirements()

    assert collected[0].source_type == "script"
    assert collected[0].metadata_snapshot == metadata_snapshot
    assert collected[0].embedded_object_id == object_id
    assert resolved[0].state == "missing"
    assert resolved[0].detail == "The catalog extension uses a different source type"


def test_collecting_loader_requirements_keeps_workspace_source_identity(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    revision = "e" * 64
    extension_id = "shared-loader"
    data_path = tmp_path / "data.txt"
    data_path.write_text("unused")
    metadata_snapshot = {
        "extension_name": "Workspace Package",
        "author": "Workspace Author",
        "contact": "workspace@example.org",
        "project_url": "https://example.org/workspace",
        "change_summary": "Workspace revision",
        "changelog": "Workspace changelog",
    }
    local_script = _ExtensionRecord(
        id=extension_id,
        name="Local Script",
        source_type="script",
        current_revision=revision,
        metadata=_ExtensionMetadata(author="Local Author"),
        revisions={
            revision: _ExtensionRevision(
                source_hash=revision,
                object_name=f"{revision}.py",
                created_at="2026-01-01T00:00:00+00:00",
            )
        },
    )
    spec = file_load(
        start_label="Load workspace package data",
        seed_code="data = xr.DataArray([1.0])",
        file_load_source=FileLoadSource(
            path=str(data_path),
            loader_label="Workspace Package",
            loader_text="workspace-loader",
            kwargs_text="",
            replay_call=FileReplayCall(
                kind="extension_loader",
                target=extension_id,
                revision=revision,
                capability_id="workspace-loader",
                extension_source_type="environment-package",
                selection=FileDataSelection(kind="dataarray"),
            ),
        ),
    )

    with manager_context() as manager:
        manager._extensions.catalog.store.mutate(
            None,
            lambda catalog: catalog.model_copy(
                update={"extensions": {local_script.id: local_script}}
            ),
        )
        manager._extensions.catalog.refresh()
        index = manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(xr.DataArray([1.0])),
            show=False,
            provenance_spec=spec,
        )
        node = manager._node_for_target(index)
        manager._extensions.set_workspace_requirements(
            (
                _WorkspaceExtensionRequirement(
                    extension_id=extension_id,
                    capability_id="workspace-loader",
                    capability_kind="loader",
                    revision_hash=revision,
                    extension_api_version=1,
                    source_type="environment-package",
                    metadata_snapshot=metadata_snapshot,
                    referencing_nodes=(node.uid,),
                    file_sources=(str(data_path),),
                ),
            )
        )

        collected = manager._extensions.collect_workspace_requirements()

    assert collected[0].source_type == "environment-package"
    assert collected[0].metadata_snapshot == metadata_snapshot


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
                extension_source_type="script",
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


def test_degraded_save_as_preserves_an_invalid_embedded_object_reference(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    source_path = tmp_path / "invalid-extension-object.itws"
    recovered_path = tmp_path / "recovered.itws"
    requirement = {
        "extension_id": "future-lab",
        "capability_id": "normalize",
        "capability_kind": "routine",
        "revision_hash": "a" * 64,
        "extension_api_version": 1,
        "source_type": "script",
        "metadata_snapshot": {},
        "embedded_object_id": "../future-object",
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

    with workspace_store.WorkspaceStore(recovered_path) as store:
        assert store.current_generation().manifest["extension_requirements"] == [
            requirement
        ]
        workspace_storage._compact_workspace_store(store)


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


def test_workspace_import_keeps_valid_source_over_conflicting_object(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    imported_path = tmp_path / "conflicting-extension.itws"
    saved_path = tmp_path / "saved-after-conflict.itws"
    valid_source = b"# valid embedded source\n"
    conflicting_source = b"# source with the wrong revision\n"
    revision = hashlib.sha256(valid_source).hexdigest()
    object_id = f"extension-{revision}"
    requirement = _WorkspaceExtensionRequirement(
        extension_id="shared-lab",
        capability_id="normalize",
        capability_kind="routine",
        revision_hash=revision,
        extension_api_version=1,
        source_type="script",
        embedded_object_id=object_id,
    )
    manifest = {
        "schema_version": 6,
        "nodes": [],
        "root_order": [],
        "extension_requirements": [requirement.model_dump(mode="json")],
    }
    with workspace_store.WorkspaceStore(imported_path, create=True) as store:
        workspace_storage._write_workspace_generation(
            store,
            workspace_storage._WorkspaceGenerationPlan(
                manifest=manifest,
                objects=(
                    workspace_storage._WorkspaceObjectWrite(
                        object_id,
                        blob=conflicting_source,
                        blob_kind="extension-python-source-v1",
                    ),
                ),
            ),
            compression_mode="none",
        )

    with manager_context() as manager:
        manager._extensions.set_workspace_requirements(
            (requirement,),
            embedded_sources={("shared-lab", revision): valid_source},
        )
        manager._workspace_controller.loading._prepare_extension_requirements(
            imported_path,
            manifest,
            replace=False,
            selected_paths=None,
        )

        assert (
            manager._extensions.revision_source_bytes("shared-lab", revision)
            == valid_source
        )
        assert object_id not in (
            manager._extensions._workspace_unresolved_embedded_objects
        )
        manager._workspace_controller.saving._save_workspace_document(saved_path)

    restored, kind = workspace_storage._read_workspace_blob(saved_path, object_id)
    assert restored == valid_source
    assert kind == "extension-python-source-v1"


def test_workspace_import_rebases_only_incoming_extension_requirements(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    source_path = tmp_path / "imported.itws"
    saved_path = tmp_path / "combined.itws"

    def operation(extension_id: str, revision: str) -> ExtensionRoutineOperation:
        return ExtensionRoutineOperation(
            extension_id=extension_id,
            revision_hash=revision,
            routine_id="normalize",
            extension_name=extension_id,
            routine_name="Normalize",
            source_type="script",
            function_name="normalize",
            source_path=f"{extension_id}.py",
            entry_point_group=None,
            entry_point_name=None,
            parameters={},
        )

    imported_operation = operation("imported-extension", "a" * 64)
    existing_operation = operation("existing-extension", "b" * 64)

    with manager_context() as manager:
        index = manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(xr.DataArray([1.0])),
            show=False,
            provenance_spec=full_data(imported_operation),
        )
        existing_node = manager._node_for_target(index)
        shared_saved_uid = existing_node.uid
        manager._workspace_controller.saving._save_workspace_document(source_path)

        existing_node.set_displayed_provenance(full_data(existing_operation))
        manager._extensions.set_workspace_requirements(
            manager._extensions.collect_workspace_requirements()
        )
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
        requirements = {
            item.extension_id: item
            for item in manager._extensions.collect_workspace_requirements()
        }
        imported_uid = manager._tool_graph.root_wrappers[1].uid

        assert imported_uid != shared_saved_uid
        assert requirements["existing-extension"].referencing_nodes == (
            shared_saved_uid,
        )
        assert requirements["imported-extension"].referencing_nodes == (imported_uid,)

        manager._workspace_controller.saving._save_workspace_document(saved_path)

    attrs = workspace_arrays._read_workspace_root_attrs_h5py(saved_path)
    manifest = workspace_format._workspace_manifest_from_attrs(attrs)
    saved_requirements = {
        item["extension_id"]: item for item in manifest["extension_requirements"]
    }
    assert saved_requirements["existing-extension"]["referencing_nodes"] == [
        shared_saved_uid
    ]
    assert saved_requirements["imported-extension"]["referencing_nodes"] == [
        imported_uid
    ]


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
        name_filter, (call, _defaults) = next(iter(loader_entries.items()))
        first = call(value_path)
        second = call(value_path)
        xr.testing.assert_identical(first, xr.DataArray([1.0, 4.0]))
        xr.testing.assert_identical(second, xr.DataArray([2.0, 4.0]))
        assert "counter_loader:counter_loader" in (manager._extensions.explorer_loaders)
        manager._recent_name_filter = name_filter
        assert manager._recent_loader_name == "counter_loader:counter_loader"
        manager.ensure_explorer_initialized()
        assert (
            manager.explorer.current_explorer.loader_name
            == "counter_loader:counter_loader"
        )
        resolved = _resolve_load_func((call, {}, FileDataSelection(kind="dataarray")))
        assert resolved is not None
        replay_call = resolved.replay_call()
        assert replay_call.kind == "extension_loader"
        assert replay_call.extension_source_type == "script"
        code = resolved.load_code(value_path, assign="loaded")
        assert code is not None
        with pytest.raises(ValueError, match="must be finite"):
            call(value_path, scale=np.nan)

    namespace: dict[str, typing.Any] = {}
    exec(code, namespace)  # noqa: S102
    xr.testing.assert_identical(namespace["loaded"], xr.DataArray([1.0, 4.0]))

    with manager_context() as second_manager:
        loader_entries = second_manager._extensions.file_loaders((value_path,))
        isolated_call, _defaults = next(iter(loader_entries.values()))
        xr.testing.assert_identical(isolated_call(value_path), xr.DataArray([1.0, 4.0]))


def test_direct_extension_loader_reload_rechecks_catalog_state(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "reload_loader.py"
    _loader_script(script_path, name="Reload Data", extensions=("txt",))
    value_path = tmp_path / "value.txt"
    value_path.write_text("4")

    with manager_context() as manager:
        catalog, _revision, _created = manager._extensions.catalog.store.add_script(
            script_path
        )
        _validate_and_enable(
            manager._extensions.catalog.store,
            "reload_loader",
            expected_record_generation=(
                catalog.extensions["reload_loader"].record_generation
            ),
        )
        manager._extensions.catalog.refresh()
        call, defaults = next(
            iter(manager._extensions.file_loaders((value_path,)).values())
        )
        tool = erlab.interactive.imagetool.ImageTool(
            call(value_path, **defaults),
            _in_manager=True,
            file_path=value_path,
            load_func=(call, defaults, FileDataSelection(kind="dataarray")),
        )
        manager.add_imagetool(tool, show=False)

        source_types: list[str | None] = []
        original_status = manager._extensions.capability_status

        def capability_status(
            extension_id: str,
            revision_hash: str,
            kind: str,
            capability_id: str,
            source_type: str | None = None,
        ) -> str:
            source_types.append(source_type)
            return original_status(
                extension_id,
                revision_hash,
                kind,
                capability_id,
                source_type,
            )

        monkeypatch.setattr(
            manager._extensions,
            "capability_status",
            capability_status,
        )
        assert tool.slicer_area._direct_reloadable()
        assert source_types == ["script"]
        assert tool.slicer_area._reload_unavailable_reason() is None

        current = manager._extensions.catalog.store.read().extensions["reload_loader"]
        manager._extensions.catalog.store.update_record(
            "reload_loader",
            expected_record_generation=current.record_generation,
            enabled=False,
        )
        manager._extensions.catalog.refresh()

        assert not tool.slicer_area._direct_reloadable()
        assert tool.slicer_area._direct_extension_loader_status() == "disabled"
        assert not tool.slicer_area.reloadable
        reason = tool.slicer_area._reload_unavailable_reason()
        assert reason is not None
        with pytest.raises(RuntimeError, match="cannot be reloaded"):
            tool.slicer_area._fetch_reload_data()


def test_file_load_edit_dialog_matches_extension_loader_source_type(
    qtbot,
    tmp_path: pathlib.Path,
) -> None:
    class ExtensionCall:
        __name__ = "load"
        extension_id = "shared"
        revision_hash = "a" * 64
        loader_id = "load"
        loader_method = None

        def __init__(self, source_type: str) -> None:
            self.source_type = source_type

        def __call__(self, _path: pathlib.Path) -> xr.DataArray:
            return xr.DataArray([1.0])

    package_call = ExtensionCall("environment-package")
    script_call = ExtensionCall("script")
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    dialog = _FileLoadEditDialog(
        FileLoadSource(
            path=str(tmp_path / "data.txt"),
            loader_label="Extension Loader",
            loader_text="shared: load",
            kwargs_text="(none)",
            replay_call=FileReplayCall(
                kind="extension_loader",
                target="shared",
                revision="a" * 64,
                capability_id="load",
                extension_source_type="script",
                selection=FileDataSelection(kind="dataarray"),
            ),
        ),
        parent,
        file_loaders=lambda _path: {
            "Package": (package_call, {}),
            "Script": (script_call, {}),
        },
    )
    qtbot.addWidget(dialog)

    assert dialog._checked_filter_name() == "Script"


def test_extension_loader_filter_conflict_is_rejected(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    first_path = tmp_path / "first.py"
    second_path = tmp_path / "second.py"
    _loader_script(first_path, name="Lab Data", extensions=(".txt",))
    _loader_script(second_path, name="Lab Data", extensions=(".txt",))

    with manager_context() as manager:
        first, _revision, _created = manager._extensions.catalog.store.add_script(
            first_path
        )
        _validate_and_enable(
            manager._extensions.catalog.store,
            "first",
            expected_record_generation=first.extensions["first"].record_generation,
        )
        second, _revision, _created = manager._extensions.catalog.store.add_script(
            second_path
        )
        with pytest.raises(
            _ExtensionCatalogConflictError,
            match="conflicts with enabled extension 'first'",
        ):
            _validate_and_enable(
                manager._extensions.catalog.store,
                "second",
                expected_record_generation=(
                    second.extensions["second"].record_generation
                ),
            )
        manager._extensions.catalog.refresh()

        loaders = manager._extensions.file_loaders()
        assert tuple(loaders) == ("Lab Data (*.txt)",)
        assert not manager._extensions.catalog.model.extensions["second"].enabled


def test_builtin_and_extension_loader_filter_conflict_is_rejected(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "netcdf.py"
    _loader_script(
        script_path,
        name="NetCDF Files",
        extensions=(".nc", ".nc4", ".cdf"),
    )

    with manager_context() as manager:
        catalog, _revision, _created = manager._extensions.catalog.store.add_script(
            script_path
        )
        with pytest.raises(
            _ExtensionCatalogConflictError,
            match="conflicts with built-in file dialog filters",
        ):
            _validate_and_enable(
                manager._extensions.catalog.store,
                "netcdf",
                expected_record_generation=(
                    catalog.extensions["netcdf"].record_generation
                ),
            )
        manager._extensions.catalog.refresh()

        loaders = manager._available_file_loaders()
        assert "NetCDF Files (*.nc *.nc4 *.cdf)" in loaders
        assert not manager._extensions.catalog.model.extensions["netcdf"].enabled


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


def test_readonly_routine_view_isolates_nested_metadata() -> None:
    source = xr.DataArray(
        np.arange(3.0),
        dims="x",
        coords={"x": xr.Variable("x", np.arange(3.0), attrs={"nested": [1]})},
        attrs={"nested": {"value": 1}},
    )
    source.encoding["nested"] = {"value": 2}
    source.coords["x"].encoding["nested"] = {"value": 3}

    readonly = _readonly_array(source)
    readonly.attrs["nested"]["value"] = 10
    readonly.encoding["nested"]["value"] = 20
    readonly.coords["x"].attrs["nested"].append(2)
    readonly.coords["x"].encoding["nested"]["value"] = 30

    assert source.attrs["nested"] == {"value": 1}
    assert source.encoding["nested"] == {"value": 2}
    assert source.coords["x"].attrs["nested"] == [1]
    assert source.coords["x"].encoding["nested"] == {"value": 3}


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


def test_controller_startup_contains_environment_refresh_failure(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(erlab.utils.misc, "_IS_PACKAGED", False)

    def fail_refresh(_store) -> typing.Never:
        raise RuntimeError("unreadable package metadata")

    monkeypatch.setattr(
        _ExtensionCatalogStore,
        "refresh_environment_packages",
        fail_refresh,
    )

    with (
        caplog.at_level(
            "ERROR",
            logger="erlab.interactive.imagetool.manager._extensions._controller",
        ),
        manager_context() as manager,
    ):
        assert manager._extensions.catalog.model.extensions == {}

    assert "Could not refresh environment extensions" in caplog.text


def test_controller_identifies_extension_loader_callables(
    manager_context,
) -> None:
    descriptor = erlab.extensions.LoaderDescriptor(
        id="load_data",
        name="Lab data",
        category="Lab",
        summary="Load lab data.",
        function_name="load_data",
        extensions=(".dat",),
    )
    call = _ExtensionLoaderCall(
        manager_session_id="manager",
        catalog_generation=1,
        extension_id="lab",
        extension_name="Lab",
        revision_hash="a" * 64,
        loader_id="load_data",
        descriptor=descriptor,
        source_path=pathlib.Path("lab.py"),
        source_type="script",
        executor=lambda *_args, **_kwargs: xr.DataArray([1.0]),
    )
    adapter = extension_execution._DecoratedLoaderAdapter(call)

    with manager_context() as manager:
        assert manager._extensions.loader_name_for_callable(adapter.load) == (
            "lab:load_data"
        )
        assert manager._extensions.loader_name_for_callable(call) == "lab:load_data"
        assert manager._extensions.loader_name_for_callable(lambda: None) is None


def test_controller_populate_menu_before_menu_creation(manager_context) -> None:
    with manager_context() as manager:
        menu = manager._extensions.menu
        manager._extensions.menu = None
        try:
            manager._extensions._populate_menu()
        finally:
            manager._extensions.menu = menu


@pytest.mark.parametrize("source", [None, b"\xff"])
def test_script_review_reports_unreadable_source(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    source: bytes | None,
) -> None:
    script_path = tmp_path / "unreadable.py"
    if source is not None:
        script_path.write_bytes(source)
    shown: list[str] = []
    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        lambda _parent, _title, text, **_kwargs: shown.append(text),
    )

    with manager_context() as manager:
        assert not manager._extensions._review_and_add(script_path)

    assert shown == ["The extension source could not be read."]


def test_script_review_cancel_and_success_paths(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "reviewed.py"
    _script(script_path)

    with manager_context() as manager:
        monkeypatch.setattr(
            extension_controller._SourceReviewDialog,
            "exec",
            lambda _dialog: QtWidgets.QDialog.DialogCode.Rejected,
        )
        assert not manager._extensions._review_and_add(script_path)
        assert manager._extensions.catalog.model.extensions == {}

        monkeypatch.setattr(
            extension_controller._SourceReviewDialog,
            "exec",
            lambda _dialog: QtWidgets.QDialog.DialogCode.Accepted,
        )
        assert manager._extensions._review_and_add(script_path)
        record = manager._extensions.catalog.model.extensions["reviewed"]
        assert record.enabled
        assert record.revisions[record.current_revision].approved


def test_manage_reload_paths(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "reloadable.py"
    _script(script_path)
    reviews: list[tuple[pathlib.Path, str | None]] = []
    failures: list[str] = []

    with manager_context() as manager:
        manager._extensions._manage_action("reload", "unknown")
        catalog, revision, _created = manager._extensions.catalog.store.add_script(
            script_path
        )
        manager._extensions.catalog.refresh()
        monkeypatch.setattr(
            manager._extensions,
            "_review_and_add",
            lambda path, *, extension_id=None: (
                reviews.append((path, extension_id)) or True
            ),
        )
        manager._extensions._manage_action("reload", "reloadable")
        assert reviews == [(script_path.resolve(), "reloadable")]

        record = catalog.extensions["reloadable"]
        embedded_revision = record.revisions[revision].model_copy(
            update={"source_path": None}
        )
        manager._extensions.catalog.model = _ExtensionCatalogModel(
            extensions={
                "reloadable": record.model_copy(
                    update={"revisions": {revision: embedded_revision}}
                )
            }
        )
        monkeypatch.setattr(
            erlab.interactive.utils.MessageDialog,
            "critical",
            lambda _parent, _title, text, **_kwargs: failures.append(text),
        )
        manager._extensions._manage_action("reload", "reloadable")

    assert failures == ["The extension could not be changed."]


def test_catalog_change_refreshes_visible_extension_consumers(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class Explorer(QtWidgets.QWidget):
        def refresh_loader_choices(self) -> None:
            calls.append("explorer")

    with manager_context() as manager:
        controller = manager._extensions
        menu = controller.menu
        if menu is None:
            raise RuntimeError("The manager extension menu was not created")
        monkeypatch.setattr(menu, "isVisible", lambda: True)
        monkeypatch.setattr(controller, "_populate_menu", lambda: calls.append("menu"))
        explorer = Explorer(manager)
        manager._standalone_app_windows["explorer"] = explorer
        tool = types.SimpleNamespace(
            _refresh_reload_data_action=lambda: calls.append("tool")
        )
        manager._tool_graph.nodes["extension-test-tool"] = types.SimpleNamespace(
            tool_window=tool
        )
        monkeypatch.setattr(manager, "_update_actions", lambda: calls.append("actions"))

        try:
            controller._catalog_changed(controller.catalog.model)
        finally:
            manager._tool_graph.nodes.pop("extension-test-tool")

    assert calls == ["menu", "explorer", "actions", "tool"]


@pytest.mark.parametrize(
    ("session_status", "expected_state"),
    [
        ("missing-revision", "missing"),
        ("missing-capability", "missing"),
        ("unsupported-api", "unsupported-api"),
        ("approval-required", "approval-required"),
    ],
)
def test_workspace_resolution_uses_session_revision_state(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    session_status: str,
    expected_state: str,
) -> None:
    requirement = _WorkspaceExtensionRequirement(
        extension_id="session",
        capability_id="analyze",
        capability_kind="routine",
        revision_hash="a" * 64,
        extension_api_version=1,
        source_type="script",
    )

    with manager_context() as manager:
        monkeypatch.setattr(
            manager._extensions.execution,
            "session_capability_status",
            lambda *_args: session_status,
        )
        assert manager._extensions._resolve_requirement(requirement).state == (
            expected_state
        )


def test_session_capability_status_detects_corrupt_source(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    source = _script(tmp_path / "session.py")
    revision = hashlib.sha256(source).hexdigest()

    with manager_context() as manager:
        manager._extensions.execution.approve_session_script(
            source,
            extension_id="session",
            revision_hash=revision,
            name="Session",
            metadata=_ExtensionMetadata(),
            source_modified_at=None,
        )
        source_path = manager._extensions.execution._session_catalog_store.source_path(
            "session", revision
        )
        source_path.write_bytes(b"corrupt")

        assert (
            manager._extensions.execution.session_capability_status(
                "session", revision, "routine", "scale"
            )
            == "hash-mismatch"
        )


def test_workspace_resolution_distinguishes_missing_exact_revisions(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_hash = "b" * 64
    requested_hash = hashlib.sha256(b"requested source").hexdigest()
    current_revision = _ExtensionRevision(
        source_hash=current_hash,
        object_name=f"{current_hash}.py",
        created_at="2026-01-01T00:00:00+00:00",
        approved=True,
    )
    script_record = _ExtensionRecord(
        id="lab",
        name="Lab",
        enabled=True,
        current_revision=current_hash,
        revisions={current_hash: current_revision},
    )
    requirement = _WorkspaceExtensionRequirement(
        extension_id="lab",
        capability_id="analyze",
        capability_kind="routine",
        revision_hash=requested_hash,
        extension_api_version=1,
        source_type="script",
    )

    with manager_context() as manager:
        manager._extensions.catalog.model = _ExtensionCatalogModel(
            extensions={"lab": script_record}
        )
        manager._extensions.set_workspace_requirements(
            (requirement,),
            embedded_sources={("lab", requested_hash): b"requested source"},
        )
        assert manager._extensions._resolve_requirement(requirement).state == (
            "approval-required"
        )

        manager._extensions.set_workspace_requirements(
            (requirement,),
            embedded_sources={("lab", requested_hash): b"corrupt"},
        )
        assert manager._extensions._resolve_requirement(requirement).state == (
            "hash-mismatch"
        )

        manager._extensions.set_workspace_requirements((requirement,))
        missing = manager._extensions._resolve_requirement(requirement)
        assert missing.state == "missing"
        assert missing.detail == "The exact revision is not in the application catalog"

        environment_revision = current_revision.model_copy(
            update={
                "object_name": "lab_package:extension",
                "entry_point_group": "erlab.extensions",
                "entry_point_name": "lab",
                "entry_point_value": "lab_package:extension",
            }
        )
        environment_record = script_record.model_copy(
            update={
                "source_type": "environment-package",
                "revisions": {current_hash: environment_revision},
            }
        )
        environment_requirement = requirement.model_copy(
            update={"source_type": "environment-package"}
        )
        manager._extensions.catalog.model = _ExtensionCatalogModel(
            extensions={"lab": environment_record}
        )
        assert (
            manager._extensions._resolve_requirement(environment_requirement).state
            == "missing"
        )

        monkeypatch.setattr(erlab.utils.misc, "_IS_PACKAGED", True)
        packaged = manager._extensions._resolve_requirement(
            environment_requirement.model_copy(update={"revision_hash": current_hash})
        )
        assert packaged.state == "missing"
        assert "unavailable in this build" in packaged.detail


def test_workspace_resolution_checks_environment_entry_point_and_capability(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision_hash = "c" * 64
    revision = _ExtensionRevision(
        source_hash=revision_hash,
        object_name="lab_package:extension",
        created_at="2026-01-01T00:00:00+00:00",
        approved=True,
        entry_point_group="erlab.extensions",
        entry_point_name="lab",
        entry_point_value="lab_package:extension",
    )
    record = _ExtensionRecord(
        id="lab",
        name="Lab",
        source_type="environment-package",
        enabled=True,
        current_revision=revision_hash,
        revisions={revision_hash: revision},
    )
    requirement = _WorkspaceExtensionRequirement(
        extension_id="lab",
        capability_id="analyze",
        capability_kind="routine",
        revision_hash=revision_hash,
        extension_api_version=1,
        source_type="environment-package",
    )

    with manager_context() as manager:
        manager._extensions.catalog.model = _ExtensionCatalogModel(
            extensions={"lab": record}
        )
        monkeypatch.setattr(
            manager._extensions.catalog.store,
            "_entry_point_for_revision",
            lambda _revision: (_ for _ in ()).throw(LookupError("missing")),
        )
        result = manager._extensions._resolve_requirement(requirement)
        assert result.state == "missing"
        assert result.detail == "The environment package entry point is unavailable"

        monkeypatch.setattr(
            manager._extensions.catalog.store,
            "_entry_point_for_revision",
            lambda _revision: object(),
        )
        result = manager._extensions._resolve_requirement(requirement)
        assert result.state == "missing"
        assert result.detail == "The revision does not provide the required capability"


def test_workspace_requirement_helpers_cover_empty_and_unavailable_nodes(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requirement = _WorkspaceExtensionRequirement(
        extension_id="lab",
        capability_id="analyze",
        capability_kind="routine",
        revision_hash="a" * 64,
        extension_api_version=1,
        source_type="script",
        referencing_nodes=("node",),
    )

    with manager_context() as manager:
        manager._extensions.set_workspace_requirements((requirement,))
        before = manager._extensions.workspace_requirement_state()
        manager._extensions.rebase_workspace_requirement_nodes({})
        manager._extensions.remove_workspace_node_references(())
        assert manager._extensions.workspace_requirement_state() == before

        monkeypatch.setattr(
            manager._extensions,
            "collect_workspace_requirements",
            lambda: (requirement,),
        )
        monkeypatch.setattr(
            manager._extensions,
            "_resolve_requirement",
            lambda item: _ResolvedWorkspaceRequirement(
                requirement=item, state="missing"
            ),
        )
        assert manager._extensions.unavailable_reason_for_node("other") is None
        assert "is missing" in typing.cast(
            "str", manager._extensions.unavailable_reason_for_node("node")
        )


def test_collecting_always_embedded_referenced_script_avoids_duplicates(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "always.py"
    _script(script_path)

    with manager_context() as manager:
        catalog, revision_hash, _created = manager._extensions.catalog.store.add_script(
            script_path
        )
        catalog = _validate_and_enable(
            manager._extensions.catalog.store,
            "always",
            expected_record_generation=catalog.extensions["always"].record_generation,
        )
        manager._extensions.catalog.store.update_record(
            "always",
            expected_record_generation=catalog.extensions["always"].record_generation,
            embed_policy="always",
        )
        manager._extensions.catalog.refresh()
        operation = ExtensionRoutineOperation(
            extension_id="always",
            revision_hash=revision_hash,
            routine_id="scale",
            extension_name="Always",
            routine_name="Scale",
            source_type="script",
            function_name="scale",
            source_path=str(script_path),
            entry_point_group=None,
            entry_point_name=None,
            parameters={},
        )
        manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(xr.DataArray([1.0])),
            show=False,
            provenance_spec=full_data(operation),
        )

        requirements = manager._extensions.collect_workspace_requirements()

    assert len(requirements) == 1
    assert requirements[0].embedded_object_id == f"extension-{revision_hash}"
    assert "source_modified_at" in requirements[0].metadata_snapshot


def test_embedded_approval_selects_the_script_requirement(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    source = _script(tmp_path / "mixed.py")
    revision = hashlib.sha256(source).hexdigest()
    base = _WorkspaceExtensionRequirement(
        extension_id="mixed",
        capability_id="scale",
        capability_kind="routine",
        revision_hash=revision,
        extension_api_version=1,
        source_type="script",
        embedded_object_id=f"extension-{revision}",
    )
    monkeypatch.setattr(
        extension_controller._SourceReviewDialog,
        "exec",
        lambda _dialog: QtWidgets.QDialog.DialogCode.Accepted,
    )

    with manager_context() as manager:
        manager._extensions.set_workspace_requirements(
            (
                base.model_copy(
                    update={
                        "source_type": "environment-package",
                        "embedded_object_id": None,
                    }
                ),
                base,
            ),
            embedded_sources={("mixed", revision): source},
        )
        manager._extensions._approve_embedded_script("mixed", revision)
        requirements = manager._extensions.collect_workspace_requirements()

        assert (
            manager._extensions.execution.session_capability_status(
                "mixed", revision, "routine", "scale"
            )
            == "ready"
        )
        assert {item.source_type for item in requirements} == {
            "script",
            "environment-package",
        }
        for requirement in requirements:
            _WorkspaceExtensionRequirement.model_validate(
                requirement.model_dump(mode="json")
            )


@pytest.mark.parametrize(
    ("source", "requirements", "warning_expected"),
    [
        (None, (), True),
        (b"unused", (), False),
        (b"\xff", ("script",), True),
    ],
)
def test_embedded_approval_rejects_unusable_workspace_state(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    source: bytes | None,
    requirements: tuple[str, ...],
    warning_expected: bool,
) -> None:
    revision = hashlib.sha256(source or b"unused").hexdigest()
    requirement = _WorkspaceExtensionRequirement(
        extension_id="unusable",
        capability_id="analyze",
        capability_kind="routine",
        revision_hash=revision,
        extension_api_version=1,
        source_type="script",
    )
    warnings: list[None] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda *_args, **_kwargs: warnings.append(None),
    )

    with manager_context() as manager:
        manager._extensions.set_workspace_requirements(
            (requirement,) if requirements else (),
            embedded_sources=(
                {} if source is None else {("unusable", revision): source}
            ),
        )
        manager._extensions._approve_embedded_script("unusable", revision)

    assert bool(warnings) is warning_expected


def test_embedded_approval_reports_validation_failure(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    source = _script(tmp_path / "failure.py")
    revision = hashlib.sha256(source).hexdigest()
    requirement = _WorkspaceExtensionRequirement(
        extension_id="failure",
        capability_id="scale",
        capability_kind="routine",
        revision_hash=revision,
        extension_api_version=1,
        source_type="script",
    )
    failures: list[str] = []
    monkeypatch.setattr(
        extension_controller._SourceReviewDialog,
        "exec",
        lambda _dialog: QtWidgets.QDialog.DialogCode.Accepted,
    )
    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        lambda _parent, _title, text, **_kwargs: failures.append(text),
    )

    with manager_context() as manager:
        manager._extensions.set_workspace_requirements(
            (requirement,),
            embedded_sources={("failure", revision): source},
        )
        monkeypatch.setattr(
            manager._extensions.execution,
            "approve_session_script",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("validation failed")
            ),
        )
        manager._extensions._approve_embedded_script("failure", revision)

    assert failures == ["The embedded extension could not be enabled."]


def test_workspace_notification_and_environment_refresh_paths(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requirement = _WorkspaceExtensionRequirement(
        extension_id="missing",
        capability_id="analyze",
        capability_kind="routine",
        revision_hash="a" * 64,
        extension_api_version=1,
        source_type="script",
    )
    dialogs: list[str] = []
    refreshes: list[str] = []

    class Dialog:
        def __init__(self, *_args, **kwargs) -> None:
            dialogs.append(kwargs["title"])

        @staticmethod
        def exec() -> None:
            return None

        @staticmethod
        def critical(_parent, title: str, _text: str, **_kwargs) -> None:
            dialogs.append(title)

    with manager_context() as manager:
        monkeypatch.setattr(erlab.interactive.utils, "MessageDialog", Dialog)
        monkeypatch.setattr(
            manager._extensions,
            "resolved_workspace_requirements",
            lambda: (
                _ResolvedWorkspaceRequirement(requirement=requirement, state="ready"),
            ),
        )
        manager._extensions.notify_unavailable_workspace_requirements()
        assert dialogs == []

        monkeypatch.setattr(
            manager._extensions,
            "resolved_workspace_requirements",
            lambda: (
                _ResolvedWorkspaceRequirement(requirement=requirement, state="missing"),
            ),
        )
        manager._extensions.notify_unavailable_workspace_requirements()
        assert dialogs == ["Workspace Extensions Unavailable"]

        monkeypatch.setattr(erlab.utils.misc, "_IS_PACKAGED", True)
        monkeypatch.setattr(
            manager._extensions.catalog.store,
            "refresh_environment_packages",
            lambda: refreshes.append("store"),
        )
        manager._extensions.refresh_environment_packages()
        assert refreshes == []

        monkeypatch.setattr(erlab.utils.misc, "_IS_PACKAGED", False)
        monkeypatch.setattr(
            manager._extensions.catalog,
            "refresh",
            lambda: refreshes.append("catalog"),
        )
        manager._extensions.refresh_environment_packages()
        assert refreshes == ["store", "catalog"]

        monkeypatch.setattr(
            manager._extensions.catalog.store,
            "refresh_environment_packages",
            lambda: (_ for _ in ()).throw(RuntimeError("refresh failed")),
        )
        manager._extensions.refresh_environment_packages()
        assert dialogs[-1] == "Extension Error"


def test_loader_invocation_logs_and_reraises_user_failure(
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    script_path = tmp_path / "failing_loader.py"
    source = b"""from pathlib import Path
import xarray as xr
from erlab.extensions import loader

@loader(name="Failing loader")
def load_data(path: Path) -> xr.DataArray:
    raise RuntimeError(f"cannot load {path.name}")
"""
    script_path.write_bytes(source)
    descriptor = erlab.extensions.LoaderDescriptor(
        id="load_data",
        name="Failing loader",
        category="General",
        summary="",
        function_name="load_data",
    )
    call = _ExtensionLoaderCall(
        manager_session_id="manager",
        catalog_generation=3,
        extension_id="failing_loader",
        extension_name="Failing loader",
        revision_hash=hashlib.sha256(source).hexdigest(),
        loader_id="load_data",
        descriptor=descriptor,
        source_path=script_path,
        source_type="script",
        executor=lambda *_args, **_kwargs: xr.DataArray([1.0]),
    )

    with (
        caplog.at_level(
            "ERROR",
            logger="erlab.interactive.imagetool.manager._extensions._execution",
        ),
        pytest.raises(RuntimeError, match=r"cannot load data\.dat"),
    ):
        call._invoke(tmp_path / "data.dat", {}, {})

    assert "Extension loader failed" in caplog.text


def test_environment_routine_worker_executes_resolved_callable(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = erlab.extensions.RoutineDescriptor(
        id="calculate",
        name="Calculate",
        category="Lab",
        summary="",
        function_name="calculate",
    )
    revision_hash = "a" * 64
    revision = _ExtensionRevision(
        source_hash=revision_hash,
        object_name="lab_package:calculate",
        created_at="2026-01-01T00:00:00+00:00",
        approved=True,
        routines=(descriptor,),
        entry_point_group="erlab.extensions",
        entry_point_name="lab",
        entry_point_value="lab_package:calculate",
    )
    record = _ExtensionRecord(
        id="lab",
        name="Lab",
        source_type="environment-package",
        enabled=True,
        current_revision=revision_hash,
        revisions={revision_hash: revision},
    )

    @erlab.extensions.routine(id="calculate")
    def calculate(data: xr.DataArray) -> xr.DataArray:
        return data + 1.0

    with manager_context() as manager:
        store = manager._extensions.catalog.store
        store.mutate(
            None,
            lambda catalog: catalog.model_copy(update={"extensions": {"lab": record}}),
        )
        manager._extensions.catalog.refresh()
        job = manager._extensions.execution._routine_job(
            extension_id="lab",
            revision_hash=revision_hash,
            routine_id="calculate",
            parameters={},
            input_data=xr.DataArray([1.0]),
            input_uid="input",
            input_snapshot="snapshot",
        )
        monkeypatch.setattr(
            extension_execution,
            "_environment_routine",
            lambda _job: (descriptor, calculate),
        )
        worker = extension_execution._ExtensionRoutineWorker(
            job,
            manager_session_id="manager",
            catalog_store=store,
            script_modules={},
        )

        worker.run()

    assert worker.result is not None
    assert worker.result.status == "success"
    xr.testing.assert_identical(worker.result.output, xr.DataArray([2.0]))


def test_session_capability_status_reports_api_and_enablement(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with manager_context() as manager:
        execution = manager._extensions.execution
        statuses = iter(("unsupported-api", "disabled"))
        calls: list[tuple[str, str, str, str]] = []

        def capability_status(*args: str) -> str:
            calls.append(typing.cast("tuple[str, str, str, str]", args))
            return next(statuses)

        monkeypatch.setattr(
            execution._session_catalog_store,
            "capability_status",
            capability_status,
        )
        assert (
            execution.session_capability_status("lab", "a" * 64, "routine", "calculate")
            == "unsupported-api"
        )
        assert (
            execution.session_capability_status("lab", "a" * 64, "routine", "calculate")
            == "disabled"
        )
        assert calls == [
            ("lab", "a" * 64, "routine", "calculate"),
            ("lab", "a" * 64, "routine", "calculate"),
        ]


def test_remove_queued_without_replay_waiter_discards_job(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "queued.py"
    _script(script_path)

    with manager_context() as manager:
        execution = manager._extensions.execution
        catalog, revision, _created = execution._catalog.store.add_script(script_path)
        _validate_and_enable(
            execution._catalog.store,
            "queued",
            expected_record_generation=catalog.extensions["queued"].record_generation,
        )
        job = execution._routine_job(
            extension_id="queued",
            revision_hash=revision,
            routine_id="scale",
            parameters={},
            input_data=xr.DataArray([1.0]),
            input_uid="input",
            input_snapshot="snapshot",
        )
        execution._pending.append(job)

        execution.remove_queued(job.job_id)

        assert execution.queued == ()


def test_shutdown_resolves_active_queued_and_blocking_extension_work(
    manager_context,
    tmp_path: pathlib.Path,
    qtbot: pytest.QtBot,
) -> None:
    script_path = tmp_path / "slow.py"
    script_path.write_text(
        """import time
import xarray as xr
from erlab.extensions import routine

@routine(name="Slow")
def slow(data: xr.DataArray, delay: float = 0.4) -> xr.DataArray:
    time.sleep(delay)
    return data + 1.0
"""
    )
    loader_descriptor = erlab.extensions.LoaderDescriptor(
        id="load_data",
        name="Load data",
        category="Lab",
        summary="",
        function_name="load_data",
    )
    loader_errors: list[Exception] = []

    with manager_context() as manager:
        execution = manager._extensions.execution
        catalog, revision, _created = execution._catalog.store.add_script(script_path)
        _validate_and_enable(
            execution._catalog.store,
            "slow",
            expected_record_generation=catalog.extensions["slow"].record_generation,
        )
        execution._catalog.refresh()
        manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(
                xr.DataArray([1.0]), _in_manager=True
            ),
            show=False,
        )
        active_job_id = execution.queue_routine(
            extension_id="slow",
            routine_id="slow",
            parameters={"delay": 0.4},
            target=0,
        )
        queued_job_id = execution.queue_routine(
            extension_id="slow",
            routine_id="slow",
            parameters={"delay": 0.4},
            target=0,
        )
        active_waiter = _ExtensionRoutineWaiter(QtCore.QEventLoop())
        queued_waiter = _ExtensionRoutineWaiter(QtCore.QEventLoop())
        execution._routine_waiters.update(
            {
                active_job_id: active_waiter,
                queued_job_id: queued_waiter,
            }
        )
        qtbot.wait_until(lambda: execution.active is not None, timeout=2000)

        call = _ExtensionLoaderCall(
            manager_session_id="manager",
            catalog_generation=1,
            extension_id="slow",
            extension_name="Slow",
            revision_hash=revision,
            loader_id="load_data",
            descriptor=loader_descriptor,
            source_path=script_path,
            source_type="script",
            executor=lambda *_args, **_kwargs: xr.DataArray([1.0]),
        )

        def run_loader() -> None:
            try:
                execution.run_loader(call, tmp_path / "data.dat", {})
            except Exception as error:
                loader_errors.append(error)

        loader_thread = threading.Thread(target=run_loader)
        loader_thread.start()
        qtbot.wait_until(lambda: bool(execution._blocking_tasks), timeout=2000)

        execution.shutdown()
        loader_thread.join(timeout=2.0)

        assert not loader_thread.is_alive()
        assert len(loader_errors) == 1
        assert "canceled during manager shutdown" in str(loader_errors[0])
        assert execution.active is None
        assert execution.queued == ()
        assert execution._routine_waiters == {}
        assert active_waiter.result is not None
        assert active_waiter.result.status == "success"
        assert queued_waiter.result is not None
        assert queued_waiter.result.status == "discarded"
        assert execution._blocking_tasks == set()
        assert execution._shutdown_complete
