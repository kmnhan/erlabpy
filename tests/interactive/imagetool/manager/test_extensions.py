from __future__ import annotations

import ast
import hashlib
import itertools
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
from qtpy import QtCore, QtGui, QtWidgets

import erlab
import erlab.interactive.imagetool.manager._base as manager_base
import erlab.interactive.imagetool.manager._extensions._catalog as extension_catalog
import erlab.interactive.imagetool.manager._extensions._dialogs as extension_dialogs
import erlab.interactive.imagetool.manager._extensions._execution as extension_execution
import erlab.interactive.imagetool.manager._widgets as manager_widgets
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
    _validate_extension_source,
)
from erlab.interactive.imagetool.manager._extensions._models import (
    _ExtensionCatalogModel,
    _ExtensionRecord,
    _ExtensionSource,
    _ResolvedWorkspaceRequirement,
    _WorkspaceExtensionRequirement,
)
from erlab.interactive.imagetool.manager._provenance_edit import (
    _controller as provenance_edit_controller,
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


def generated_external_routine(
    data: xr.DataArray, *, scale: float = 1.0
) -> xr.DataArray:
    return data * scale


def _validate_and_enable(
    store: _ExtensionCatalogStore,
    extension_id: str,
    *,
    expected_record_generation: int,
) -> _ExtensionCatalogModel:
    record = store.read().extensions[extension_id]
    manager_session_id = f"test-manager-{uuid.uuid4().hex}"
    try:
        return _validate_extension_source(
            store,
            extension_id,
            source_hash=record.source.source_hash,
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


def test_source_review_dialog_reads_source(
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
        erlab.interactive.utils.PythonCodeEditor,
        "manager_extension_source_review",
    )
    if source_editor is None:
        raise RuntimeError("Source review editor was not created")
    assert source_editor.toPlainText() == "VALUE = 1\n"
    assert source_editor.isReadOnly()
    assert source_editor.lineWrapMode() is QtWidgets.QTextEdit.LineWrapMode.NoWrap
    assert isinstance(
        source_editor.highlighter, erlab.interactive.utils.PythonHighlighter
    )


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
    changes: list[tuple[str, str, bool]] = []
    dialog.favorite_requested.connect(
        lambda extension_id, routine_id, favorite: changes.append(
            (extension_id, routine_id, favorite)
        )
    )
    dialog.favorite_button.click()
    assert changes == [("lab", "calculate", True)]
    assert dialog.favorite_button.property("favoriteState") is True
    dialog.favorite_button.click()
    assert changes[-1] == ("lab", "calculate", False)


def test_manage_dialog_preserves_selected_extension(
    qtbot: pytest.QtBot,
) -> None:
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    source_hash = "a" * 64
    source = _ExtensionSource(
        source_hash=source_hash,
        object_name=f"{source_hash}.py",
        source_path="source.py",
        registered_at="2026-01-01T00:00:00+00:00",
        approved=True,
    )
    record = _ExtensionRecord(
        id="lab",
        name="Lab",
        source=source,
    )
    dialog = extension_dialogs._ManageExtensionsDialog(parent)
    qtbot.addWidget(dialog)
    dialog.set_catalog(_ExtensionCatalogModel(extensions={"lab": record}))
    top = dialog.tree.topLevelItem(0)
    assert top.childCount() == 0
    dialog.tree.setCurrentItem(top)
    assert dialog.selected_extension_id == "lab"

    actions: list[tuple[str, str]] = []

    def action_slot(action: str, extension: str) -> None:
        actions.append((action, extension))

    dialog.action_requested.connect(action_slot)
    try:
        assert "metadata" not in dialog._buttons
        dialog.tree.setCurrentItem(None)
        original_source_label = dialog._detail_labels["original_source"]
        if not isinstance(original_source_label, manager_widgets._ElidedValueLabel):
            raise TypeError("The source path must use an elided value label")
        assert original_source_label.full_text == ""
        dialog._emit_action("remove")
        assert actions == []
    finally:
        dialog.action_requested.disconnect(action_slot)


def test_workspace_requirements_dialog_registers_only_recoverable_selection(
    qtbot: pytest.QtBot,
) -> None:
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    requirement = _WorkspaceExtensionRequirement(
        extension_id="lab",
        capability_id="calculate",
        capability_kind="routine",
        source_hash="a" * 64,
        extension_api_version=1,
    )
    resolved = _ResolvedWorkspaceRequirement(
        requirement=requirement,
        state="approval-required",
    )
    dialog = extension_dialogs._WorkspaceRequirementsDialog(
        (resolved,),
        parent,
        recoverable={("lab", "a" * 64)},
    )
    qtbot.addWidget(dialog)
    registrations: list[tuple[str, str]] = []

    def registration_slot(extension: str, revision: str) -> None:
        registrations.append((extension, revision))

    dialog.register_requested.connect(registration_slot)
    try:
        dialog._register_selected()
        assert registrations == []
        dialog.tree.setCurrentItem(dialog.tree.topLevelItem(0))
        assert dialog._register_button.isEnabled()
        dialog._register_selected()
        assert registrations == [("lab", "a" * 64)]

        for state in ("missing", "hash-mismatch", "validation-failed"):
            dialog.set_requirements((resolved.model_copy(update={"state": state}),))
            assert dialog._register_button.isEnabled()
            dialog._register_selected()

        assert registrations == [("lab", "a" * 64)] * 4
        dialog.set_requirements((resolved.model_copy(update={"state": "ready"}),))
        assert dialog.tree.currentItem() is dialog.tree.topLevelItem(0)
        assert not dialog._register_button.isEnabled()
        dialog._register_selected()
        assert registrations == [("lab", "a" * 64)] * 4
    finally:
        dialog.register_requested.disconnect(registration_slot)


def test_missing_scripts_dialog_lists_scripts_and_emits_selected_actions(
    qtbot: pytest.QtBot,
) -> None:
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    records = tuple(
        _ExtensionRecord(
            id=extension_id,
            name=filename,
            enabled=True,
            source=_ExtensionSource(
                source_hash=source_hash,
                object_name=f"{source_hash}.py",
                source_path=source_path,
                registered_at="2026-01-01T00:00:00+00:00",
                approved=True,
            ),
        )
        for extension_id, filename, source_hash, source_path in (
            ("first", "first.py", "a" * 64, "/missing/first.py"),
            ("second", "second.py", "b" * 64, "/missing/second.py"),
        )
    )
    dialog = extension_dialogs._MissingScriptsDialog(records, parent)
    qtbot.addWidget(dialog)
    located: list[str] = []
    restored: list[str] = []
    dialog.locate_requested.connect(located.append)
    dialog.restore_requested.connect(restored.append)

    assert dialog.tree.topLevelItemCount() == 2
    assert dialog.tree.topLevelItem(0).childCount() == 0
    dialog.tree.setCurrentItem(dialog.tree.topLevelItem(1))
    dialog.locate_button.click()
    dialog.restore_button.click()

    assert located == ["second"]
    assert restored == ["second"]


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
    source = _ExtensionSource(
        source_hash=source_hash,
        object_name=f"{source_hash}.py",
        source_path=os.fspath(source_path),
        registered_at="2026-01-01T00:00:00+00:00",
        approved=True,
        loaders=(descriptor,),
    )
    records = {
        extension_id: _ExtensionRecord(
            id=extension_id,
            name=extension_id.title(),
            enabled=True,
            source=source,
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
            "source_available",
            lambda *_args: True,
        )
        monkeypatch.setattr(
            controller.catalog.store,
            "executable_source_path",
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

        catalog, _source_hash, _created = controller.catalog.store.add_script(
            script_path
        )
        catalog = _validate_and_enable(
            controller.catalog.store,
            "scale",
            expected_record_generation=catalog.extensions["scale"].record_generation,
        )
        catalog = controller.catalog.store.set_routine_favorite(
            "scale", "scale", favorite=True
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

        class AcceptedSelectionDialog(QtCore.QObject):
            favorite_requested = QtCore.Signal(str, str, bool)
            selection = ("scale", "scale")

            def __init__(self, *_args, **_kwargs) -> None:
                super().__init__()

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
            source_hash="a" * 64,
            capability_id="load_data",
            selection=FileDataSelection(kind="dataarray"),
        )
        with pytest.raises(
            erlab.extensions.ExtensionExecutionError,
            match="source is not available",
        ):
            controller.replay_loader(load_source(missing_call))

        script_path = tmp_path / "loader.py"
        _loader_script(script_path, name="Lab Data", extensions=(".dat",))
        catalog, source_hash, _created = controller.catalog.store.add_script(
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
            source_hash=source_hash,
            capability_id="load_data",
            selection=FileDataSelection(kind="dataarray"),
        )
        missing_descriptor = record.source.model_copy(update={"loaders": ()})
        missing_descriptor_record = record.model_copy(
            update={"source": missing_descriptor}
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


def test_controller_capability_status_uses_application_catalog(
    manager_context,
) -> None:
    with manager_context() as manager:
        controller = manager._extensions
        assert (
            controller.capability_status("missing", "a" * 64, "routine", "calculate")
            == "missing-source"
        )


def test_catalog_source_states_distinguish_all_script_source_failures(
    manager_context,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    object_directory = tmp_path / "objects"
    object_directory.mkdir()
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
        source_hash = hashlib.sha256(source).hexdigest()
        object_path = object_directory / f"{source_hash}.py"
        if stored is not None:
            object_path.write_bytes(source if stored == b"source" else stored)
        original_path = tmp_path / "original" / f"{extension_id}.py"
        if original is not None:
            original_path.parent.mkdir(exist_ok=True)
            original_path.write_bytes(source if original == b"source" else original)
        source_record = _ExtensionSource(
            source_hash=source_hash,
            object_name=object_path.name,
            source_path=(
                None if extension_id == "embedded" else os.fspath(original_path)
            ),
            registered_at="2026-01-01T00:00:00+00:00",
        )
        records[extension_id] = _ExtensionRecord(
            id=extension_id,
            name=extension_id.title(),
            source=source_record,
        )
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

    original_read_bytes = pathlib.Path.read_bytes

    def read_bytes(path: pathlib.Path) -> bytes:
        if path in unreadable_paths:
            raise OSError("unreadable")
        return original_read_bytes(path)

    with manager_context() as manager:
        controller = manager._extensions
        controller.catalog.model = _ExtensionCatalogModel(extensions=records)

        monkeypatch.setattr(pathlib.Path, "read_bytes", read_bytes)
        states = controller._catalog_source_states()

    state_by_extension = {
        extension_id: state for (extension_id, _revision), state in states.items()
    }
    assert state_by_extension == {
        "missing-stored": "Ready",
        "unreadable-stored": "Ready",
        "mismatch": "Ready",
        "embedded": "No registered source file",
        "missing-original": "Source file missing",
        "unreadable-original": "Source file unreadable",
        "unchanged": "Ready",
        "changed": "Source file changed",
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
    restores: list[str] = []

    with manager_context() as manager:
        controller = manager._extensions
        catalog, _source_hash, _created = controller.catalog.store.add_script(
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
        monkeypatch.setattr(
            controller,
            "_restore_missing_script",
            lambda extension_id: restores.append(extension_id) or True,
        )

        controller._manage_action("toggle", "scale")
        assert updates[0]["enabled"] is False

        disabled_record = enabled_record.model_copy(update={"enabled": False})
        controller.catalog.model = catalog.model_copy(
            update={"extensions": {"scale": disabled_record}}
        )
        controller._manage_action("toggle", "scale")
        assert validations == [("scale", disabled_record.record_generation)]

        before = len(updates)
        controller._manage_action("embedding:invalid", "scale")
        assert len(updates) == before
        controller._manage_action("embedding:always", "scale")
        assert updates[-1]["embed_policy"] == "always"

        embedded_source = enabled_record.source.model_copy(update={"source_path": None})
        controller.catalog.model = catalog.model_copy(
            update={
                "extensions": {
                    "scale": enabled_record.model_copy(
                        update={"source": embedded_source}
                    )
                }
            }
        )
        controller._manage_action("reload", "scale")
        assert restores == ["scale"]

        controller.catalog.model = catalog
        monkeypatch.setattr(
            controller.catalog.store,
            "update_record",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                _ExtensionCatalogConflictError("changed")
            ),
        )
        controller._manage_action("toggle", "scale")
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
        source_hash="a" * 64,
        loader_id="load_data",
        descriptor=descriptor,
        source_path=pathlib.Path("lab.py"),
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


def test_catalog_reload_preserves_source_identity(
    tmp_path: pathlib.Path,
) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    script_path = tmp_path / "scale.py"
    source = _script(script_path)

    catalog, source_hash, created = store.add_script(script_path)
    assert created
    assert catalog.schema_version == 1
    assert json.loads(store.path.read_text(encoding="utf-8"))["schema_version"] == 1
    assert source_hash == hashlib.sha256(source).hexdigest()
    catalog, unchanged_hash, created = store.add_script(script_path)
    assert not created
    assert unchanged_hash == source_hash
    assert catalog.extensions["scale"].source.source_hash == source_hash


def test_catalog_uses_override_directory_and_safe_generated_id(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "extension-catalog"
    monkeypatch.setenv("ERLAB_EXTENSION_CATALOG", os.fspath(directory))

    assert extension_catalog._default_catalog_directory() == directory.resolve()
    assert extension_catalog._safe_extension_id(" Lab analysis! ") == "Lab-analysis"
    assert extension_catalog._safe_extension_id("...").startswith("extension-")


def test_default_catalog_directory_is_independent_of_application_name(
    qapp: QtWidgets.QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ERLAB_EXTENSION_CATALOG", raising=False)
    original_name = qapp.applicationName()
    try:
        qapp.setApplicationName("ImageTool Manager")
        packaged_path = extension_catalog._default_catalog_directory()
        qapp.setApplicationName("ipykernel")
        notebook_path = extension_catalog._default_catalog_directory()
    finally:
        qapp.setApplicationName(original_name)

    data_root = pathlib.Path(
        QtCore.QStandardPaths.writableLocation(
            QtCore.QStandardPaths.StandardLocation.GenericDataLocation
        )
    )
    expected = data_root / "ERLab" / "ImageTool Manager" / "extensions"
    assert packaged_path == expected
    assert notebook_path == expected


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
    catalog, source_hash, _created = store.add_script(script_path)

    with pytest.raises(KeyError, match="Unknown extension source"):
        store.recovery_source_path("missing", source_hash)
    store.recovery_source_path("scale", source_hash).unlink()
    with pytest.raises(FileNotFoundError):
        store.recovery_source_path("scale", source_hash)
    with pytest.raises(ValueError, match="does not match its source hash"):
        store._store_script_source(source, "0" * 64)

    assert store.read() == catalog


def test_extension_models_reject_invalid_hash_and_unapproved_enablement() -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        _ExtensionSource(
            source_hash="A" * 64,
            object_name="source.py",
            registered_at="2026-01-01T00:00:00+00:00",
        )
    with pytest.raises(ValueError, match="must be approved"):
        _ExtensionRecord(
            id="source",
            name="source.py",
            enabled=True,
            source=_ExtensionSource(
                source_hash="a" * 64,
                object_name=f"{'a' * 64}.py",
                registered_at="2026-01-01T00:00:00+00:00",
            ),
        )


def test_catalog_rejects_validation_for_replaced_source(
    tmp_path: pathlib.Path,
) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    script_path = tmp_path / "scale.py"
    _script(script_path)
    _catalog, old_source_hash, _created = store.add_script(script_path)
    _script(script_path, "data + scale")
    catalog, new_source_hash, _created = store.add_script(script_path)

    with pytest.raises(
        _ExtensionCatalogConflictError, match="changed during validation"
    ):
        store.enable_validated_source(
            "scale",
            source_hash=old_source_hash,
            expected_record_generation=catalog.extensions["scale"].record_generation,
            routines=(),
            loaders=(),
            enable_extension=False,
        )
    assert store.read().extensions["scale"].source.source_hash == new_source_hash


def test_catalog_reports_exact_script_capability_states(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    script_path = tmp_path / "scale.py"
    _script(script_path)
    catalog, source_hash, _created = store.add_script(script_path)
    catalog = _validate_and_enable(
        store,
        "scale",
        expected_record_generation=catalog.extensions["scale"].record_generation,
    )
    ready_record = catalog.extensions["scale"]

    def status_for(record: _ExtensionRecord, capability_id: str = "scale") -> str:
        model = catalog.model_copy(update={"extensions": {"scale": record}})
        monkeypatch.setattr(store, "read", lambda: model)
        return store.capability_status("scale", source_hash, "routine", capability_id)

    assert status_for(ready_record) == "ready"
    assert status_for(ready_record.model_copy(update={"enabled": False})) == "disabled"

    ready_source = ready_record.source
    assert (
        status_for(
            ready_record.model_copy(
                update={
                    "enabled": False,
                    "source": ready_source.model_copy(update={"approved": False}),
                }
            )
        )
        == "approval-required"
    )
    assert status_for(ready_record, "missing") == "missing-capability"

    unsupported_descriptor = ready_source.routines[0].model_copy(
        update={"extension_api_version": 2}
    )
    assert (
        status_for(
            ready_record.model_copy(
                update={
                    "source": ready_source.model_copy(
                        update={"routines": (unsupported_descriptor,)}
                    )
                }
            )
        )
        == "unsupported-api"
    )

    monkeypatch.undo()
    script_path.write_bytes(b"corrupt")
    assert (
        store.capability_status("scale", source_hash, "routine", "scale")
        == "hash-mismatch"
    )
    script_path.unlink()
    assert (
        store.capability_status("scale", source_hash, "routine", "scale")
        == "missing-source"
    )


def test_catalog_source_availability_rejects_unknown_and_missing_sources(
    tmp_path: pathlib.Path,
) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    script_path = tmp_path / "scale.py"
    _script(script_path)
    catalog, source_hash, _created = store.add_script(script_path)
    record = catalog.extensions["scale"]

    assert not store.source_available(record, "0" * 64)
    script_path.unlink()
    assert not store.source_available(record, source_hash)


def test_catalog_resolve_script_capability_rejects_unusable_state(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    script_path = tmp_path / "scale.py"
    _script(script_path)
    catalog, source_hash, _created = store.add_script(script_path)

    with pytest.raises(erlab.extensions.ExtensionNotFoundError, match="disabled"):
        store.resolve_capability("scale", source_hash, "routine", "scale")
    unapproved = catalog.extensions["scale"].model_copy(update={"enabled": True})
    monkeypatch.setattr(
        store,
        "read",
        lambda: catalog.model_copy(update={"extensions": {"scale": unapproved}}),
    )
    with pytest.raises(erlab.extensions.ExtensionNotFoundError, match="not approved"):
        store.resolve_capability("scale", source_hash, "routine", "scale")
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
        store.resolve_capability("scale", source_hash, "routine", "scale")
    catalog = store.update_record(
        "scale",
        expected_record_generation=catalog.extensions["scale"].record_generation,
        enabled=True,
    )
    with pytest.raises(KeyError, match="Unknown routine capability"):
        store.resolve_capability("scale", source_hash, "routine", "missing")


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
    source = tmp_path / "loader.py"
    source.write_text("source")
    call = _ExtensionLoaderCall(
        manager_session_id="manager",
        catalog_generation=1,
        extension_id="lab",
        extension_name="Lab",
        source_hash="a" * 64,
        loader_id="load_data",
        descriptor=descriptor,
        source_path=source,
        executor=lambda *_args: xr.DataArray([1.0]),
    )
    assert call.manager_loader_name == "lab:load_data"
    assert call.__name__ == "load"
    with pytest.raises(erlab.extensions.ExtensionExecutionError, match="missing"):
        extension_execution._require_loader_entry(call, None)

    array = xr.DataArray([1.0], dims="x")
    dataset = xr.Dataset({"value": array})
    tree = xr.DataTree.from_dict({"/": dataset})
    assert extension_execution._require_loader_output(array) is array
    assert extension_execution._require_loader_output(dataset) is dataset
    assert extension_execution._require_loader_output(tree) is tree
    with pytest.raises(
        erlab.extensions.ExtensionExecutionError, match="expected an xarray"
    ):
        extension_execution._require_loader_output([array])

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
    assert extension_execution._loader_output_log_fields(array)["type"] == "DataArray"

    with pytest.raises(
        erlab.extensions.ExtensionExecutionError, match="expected DataArray"
    ):
        extension_execution._require_dataarray(dataset)


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
        source_hash="a" * 64,
        loader_id="load_data",
        descriptor=descriptor,
        source_path=tmp_path / "loader.py",
        executor=execute,
    )
    adapter = extension_execution._DecoratedLoaderAdapter(call)
    path = tmp_path / "value.dat"
    path.write_text("3")

    assert adapter.extension_id == "lab"
    assert adapter.source_hash == "a" * 64
    assert adapter.loader_id == "load_data"
    assert adapter.source_path == tmp_path / "loader.py"
    assert adapter.descriptor == descriptor
    assert tuple(adapter.file_dialog_methods) == ("Load Data (*.dat)",)
    loaded = adapter.load(path)
    assert loaded.item() == 3.0
    assert loaded.attrs["data_loader_name"] == "lab:load_data"
    xr.testing.assert_identical(
        adapter.load_single(path, scale=2.0), xr.DataArray([3.0])
    )
    assert calls == [(path, {}), (path, {"scale": 2.0})]
    with pytest.raises(ValueError, match="must be finite"):
        call(path, scale=float("inf"))


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
        source_hash="a" * 64,
        loader_id="load_data",
        descriptor=descriptor,
        source_path=tmp_path / "loader.py",
        executor=lambda *_args: xr.DataArray([1.0]),
    )
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    loader_worker = _ExtensionLoaderWorker(
        call,
        tmp_path / "data",
        {},
        store,
        {},
        source_is_healthy=lambda *_args: True,
    )
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
        source_hash="a" * 64,
        loader_id="load_data",
        descriptor=descriptor,
        source_path=tmp_path / "loader.py",
        executor=lambda *_args: xr.DataArray([1.0]),
    )
    record = types.SimpleNamespace(enabled=True)
    store = types.SimpleNamespace(
        read=lambda: types.SimpleNamespace(extensions={"lab": record}),
        capability_status=lambda *_args: "ready",
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
        source_is_healthy=lambda *_args: True,
    )

    worker.run()

    assert worker.done.is_set()
    assert isinstance(worker.error, erlab.extensions.ExtensionExecutionError)
    assert "KeyboardInterrupt" in str(worker.error)


def test_extension_validation_rejects_an_unknown_record(
    tmp_path: pathlib.Path,
) -> None:
    with pytest.raises(KeyError, match="missing"):
        _validate_extension_source(
            _ExtensionCatalogStore(tmp_path / "catalog"),
            "missing",
            source_hash="a" * 64,
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
            source_hash="a" * 64,
            loader_id="load_data",
            descriptor=descriptor,
            source_path=tmp_path / "loader.py",
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
                    source_is_healthy=lambda *_args: True,
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
        source_hash="a" * 64,
        loader_id="load_data",
        descriptor=descriptor,
        source_path=tmp_path / "loader.py",
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
            source_is_healthy=lambda *_args: True,
        )
        try:
            with pytest.raises(RuntimeError, match="pool rejected"):
                execution._run_blocking_task(task)
            assert task not in execution._blocking_tasks
        finally:
            execution._pool = original_pool


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
                source_hash=None,
                routine_id="scale",
                parameters={},
                input_data=data,
                input_uid="uid",
                input_snapshot="snapshot",
            )

        catalog, source_hash, _created = manager._extensions.catalog.store.add_script(
            script_path
        )
        with pytest.raises(
            erlab.extensions.ExtensionExecutionError, match="not enabled"
        ):
            execution._routine_job(
                extension_id="scale",
                source_hash=source_hash,
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
            erlab.extensions.ExtensionExecutionError, match="missing-source"
        ):
            execution._routine_job(
                extension_id="scale",
                source_hash="0" * 64,
                routine_id="scale",
                parameters={},
                input_data=data,
                input_uid="uid",
                input_snapshot="snapshot",
            )
        with pytest.raises(
            erlab.extensions.ExtensionExecutionError, match="missing-capability"
        ):
            execution._routine_job(
                extension_id="scale",
                source_hash=catalog.extensions["scale"].source.source_hash,
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
        catalog, source_hash, _created = manager._extensions.catalog.store.add_script(
            script_path
        )
        catalog = _validate_and_enable(
            manager._extensions.catalog.store,
            "scale",
            expected_record_generation=catalog.extensions["scale"].record_generation,
        )
        job = execution._routine_job(
            extension_id="scale",
            source_hash=source_hash,
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
        catalog, source_hash, _created = manager._extensions.catalog.store.add_script(
            script_path
        )
        catalog = _validate_and_enable(
            manager._extensions.catalog.store,
            "scale",
            expected_record_generation=catalog.extensions["scale"].record_generation,
        )
        operation = ExtensionRoutineOperation(
            extension_id="scale",
            source_hash=source_hash,
            routine_id="scale",
            extension_name="Scale",
            routine_name="Scale",
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


def test_extension_routine_provenance_parameters_are_editable(
    manager_context,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = tmp_path / "scale.py"
    _script(script_path)
    source_data = xr.DataArray([1.0, 2.0], dims=("x",))

    with manager_context() as manager:
        catalog, source_hash, _created = manager._extensions.catalog.store.add_script(
            script_path
        )
        _validate_and_enable(
            manager._extensions.catalog.store,
            "scale",
            expected_record_generation=catalog.extensions["scale"].record_generation,
        )
        manager._extensions.catalog.refresh()
        operation = ExtensionRoutineOperation(
            extension_id="scale",
            source_hash=source_hash,
            routine_id="scale",
            extension_name="scale.py",
            routine_name="Scale",
            parameters={"scale": 2.0},
        )
        target = manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(source_data * 2.0),
            show=False,
            provenance_spec=full_data(operation),
            replay_source_data=source_data,
        )
        node = manager._node_for_target(target)
        manager._metadata_node_uid = node.uid
        spec = node.displayed_provenance_spec
        if spec is None:
            raise RuntimeError("Expected extension provenance")
        row = spec.display_rows()[1]
        assert manager._provenance_edit_controller.can_edit_row(row) == (True, "")
        initial_values: list[dict[str, typing.Any]] = []

        class ParameterDialog:
            parameters: typing.ClassVar[dict[str, float]] = {"scale": 3.0}

            def __init__(
                self,
                descriptor: erlab.extensions.RoutineDescriptor,
                _parent: object,
                values: dict[str, typing.Any] | None = None,
            ) -> None:
                if descriptor.id != "scale":
                    raise ValueError("Expected the scale routine descriptor")
                initial_values.append(dict(values or {}))

            def exec(self) -> int:
                return int(QtWidgets.QDialog.DialogCode.Accepted)

        monkeypatch.setattr(
            provenance_edit_controller,
            "_ExtensionParameterDialog",
            ParameterDialog,
        )

        manager._provenance_edit_controller.edit_row(row)

        assert initial_values == [{"scale": 2.0}]
        updated = node.displayed_provenance_spec
        if updated is None:
            raise RuntimeError("Expected edited extension provenance")
        edited_operation = updated.operations[0]
        assert isinstance(edited_operation, ExtensionRoutineOperation)
        assert edited_operation.parameters == {"scale": 3.0}
        xr.testing.assert_identical(
            node.current_public_data(),
            source_data * 3.0,
        )


def test_execution_shutdown_is_safe_after_qt_teardown(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with manager_context() as manager:
        execution = manager._extensions.execution
        monkeypatch.setattr(
            erlab.interactive.utils, "qt_is_valid", lambda *_objects: False
        )
        execution.shutdown()
        execution.shutdown()
        assert execution._shutdown_complete


@pytest.mark.parametrize(
    "corruption",
    ["extension-key", "source-hash", "object-name"],
)
def test_catalog_rejects_inconsistent_persisted_identity(
    tmp_path: pathlib.Path,
    corruption: str,
) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    script_path = tmp_path / "scale.py"
    _script(script_path)
    catalog, _source_hash, _created = store.add_script(script_path)
    payload = catalog.model_dump(mode="json")
    record = payload["extensions"]["scale"]
    if corruption == "extension-key":
        record["id"] = "different"
    elif corruption == "source-hash":
        record["source"]["source_hash"] = "0" * 64
    else:
        record["source"]["object_name"] = "../outside.py"
    store.path.write_text(json.dumps(payload))

    with pytest.raises(extension_catalog._ExtensionCatalogError):
        store.read()


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
            update={
                "source": records["scale"].source.model_copy(
                    update={"object_name": "../outside.py"}
                )
            }
        )
        return model.model_copy(update={"extensions": records})

    with pytest.raises(ValueError, match="object name must match its source hash"):
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
    catalog, old_source_hash, _created = store.add_script(script_path)
    record = catalog.extensions["scale"]
    catalog = _validate_and_enable(
        store, "scale", expected_record_generation=record.record_generation
    )
    assert catalog.extensions["scale"].enabled

    _script(script_path, "data + scale")
    catalog, new_source_hash, created = store.add_script(script_path)
    assert created
    assert new_source_hash != old_source_hash
    assert not catalog.extensions["scale"].enabled
    assert not catalog.extensions["scale"].source.approved
    assert not store.objects_directory.joinpath(f"{old_source_hash}.py").exists()

    _script(script_path)
    catalog, restored_source_hash, changed = store.add_script(script_path)
    assert changed
    assert restored_source_hash == old_source_hash
    assert catalog.extensions["scale"].source.source_hash == old_source_hash
    assert not store.objects_directory.joinpath(f"{new_source_hash}.py").exists()
    assert not catalog.extensions["scale"].enabled

    catalog = _validate_and_enable(
        store,
        "scale",
        expected_record_generation=catalog.extensions["scale"].record_generation,
    )
    catalog, unchanged_hash, changed = store.add_script(script_path)
    assert not changed
    assert unchanged_hash == old_source_hash
    assert catalog.extensions["scale"].enabled


def test_stale_validation_does_not_import_a_newer_source(
    tmp_path: pathlib.Path,
) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    script_path = tmp_path / "scale.py"
    marker_path = tmp_path / "imported-new-source"
    _script(script_path)
    catalog, reviewed_source_hash, _created = store.add_script(script_path)
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
    catalog, newer_source_hash, _created = store.add_script(script_path)

    with pytest.raises(_ExtensionCatalogConflictError, match="before validation"):
        _validate_extension_source(
            store,
            "scale",
            source_hash=reviewed_source_hash,
            expected_record_generation=reviewed_generation,
            manager_session_id="manager",
            script_modules={},
        )

    assert not marker_path.exists()
    current = catalog.extensions["scale"]
    assert current.source.source_hash == newer_source_hash
    assert not current.source.approved


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


def test_add_script_keeps_same_filename_sources_as_distinct_extensions(
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
    with manager_context() as manager:
        _before, _revision, _created = manager._extensions.catalog.store.add_script(
            first_path,
        )
        manager._extensions.catalog.refresh()
        monkeypatch.setattr(
            extension_controller._SourceReviewDialog,
            "exec",
            lambda _dialog: review_calls.append(None) or 1,
        )
        assert manager._extensions._review_and_add(second_path)
        assert manager._extensions._review_and_add(second_path)
        after = manager._extensions.catalog.store.read()

    assert review_calls == [None, None]
    assert len(after.extensions) == 2
    assert {record.name for record in after.extensions.values()} == {"analysis.py"}
    assert {
        pathlib.Path(record.source.source_path or "")
        for record in after.extensions.values()
    } == {first_path, second_path}


def test_unchanged_add_script_enables_current_source(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "analysis.py"
    _script(script_path)
    with manager_context() as manager:
        before, source_hash, _created = manager._extensions.catalog.store.add_script(
            script_path
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
    assert record.source.source_hash == source_hash
    assert record.enabled
    assert record.source.approved
    assert (
        record.record_generation == before.extensions["analysis"].record_generation + 1
    )


def test_identical_same_filename_sources_remain_distinct_extensions(
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

    with manager_context() as manager:
        _before, revision, _created = manager._extensions.catalog.store.add_script(
            first_path,
        )
        manager._extensions.catalog.refresh()
        monkeypatch.setattr(
            extension_controller._SourceReviewDialog,
            "exec",
            lambda _dialog: 1,
        )

        assert manager._extensions._review_and_add(second_path)
        records = manager._extensions.catalog.store.read().extensions

    assert len(records) == 2
    assert {record.name for record in records.values()} == {"analysis.py"}
    assert {record.source.source_hash for record in records.values()} == {revision}
    assert {
        pathlib.Path(record.source.source_path or "") for record in records.values()
    } == {first_path, second_path}


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
        embed_policy="always",
    )
    _script(script_path, "data + scale")

    with pytest.raises(_ExtensionCatalogConflictError, match="another manager"):
        store.add_script(
            script_path,
            extension_id="scale",
            expected_record_generation=stale_generation,
            check_record_generation=True,
        )


def test_catalog_reads_unreleased_schema_as_schema_one(
    tmp_path: pathlib.Path,
) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    script_path = tmp_path / "gaussian_tools.py"
    _script(script_path)
    catalog, revision, _created = store.add_script(script_path)
    catalog = _validate_and_enable(
        store,
        "gaussian_tools",
        expected_record_generation=catalog.extensions[
            "gaussian_tools"
        ].record_generation,
    )
    payload = catalog.model_dump(mode="json")
    payload["schema_version"] = 4
    payload["routine_favorites"] = [["environment.my-lab", "normalize"]]
    payload["extensions"]["environment.my-lab"] = {"source_type": "environment-package"}
    record = payload["extensions"]["gaussian_tools"]
    source = record.pop("source")
    source["import_error"] = "old validation failure"
    record["current_revision"] = revision
    record["revisions"] = {revision: source}
    record.update(
        {
            "name": "Gaussian Tools",
            "favorite": True,
            "removed": False,
            "metadata": {
                "author": "A Lab",
                "contact": "lab@example.org",
                "project_url": "https://example.org",
                "change_summary": "Initial lab version",
                "changelog": "Unused prototype metadata",
            },
        }
    )
    store.path.write_text(json.dumps(payload), encoding="utf-8")

    migrated = store.read()

    assert migrated.schema_version == 1
    assert migrated.extensions["gaussian_tools"].name == "gaussian_tools.py"
    assert migrated.extensions["gaussian_tools"].source.source_hash == revision
    assert (
        "validation_error"
        not in migrated.extensions["gaussian_tools"].source.model_dump()
    )
    assert migrated.routine_favorites == (("gaussian_tools", "scale"),)


def test_unreadable_catalog_does_not_prevent_manager_startup(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog_directory = pathlib.Path(os.environ["ERLAB_EXTENSION_CATALOG"])
    catalog_directory.mkdir(parents=True)
    catalog_path = catalog_directory / "catalog.json"
    invalid_catalog = b'{"schema_version": 999, "extensions": {}}'
    catalog_path.write_bytes(invalid_catalog)
    errors: list[tuple[str, str, str | None]] = []
    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        lambda _parent, title, text, detailed_text=None, **_kwargs: errors.append(
            (title, text, detailed_text)
        ),
    )

    with manager_context() as manager:
        QtWidgets.QApplication.processEvents()

        assert manager._extensions.catalog.model == _ExtensionCatalogModel()
        assert manager._extensions.catalog.load_error is not None
        assert catalog_path.read_bytes() == invalid_catalog
        assert len(errors) == 1
        assert errors[0][0] == "Extension Catalog Unavailable"
        assert str(catalog_path) in (errors[0][2] or "")


def test_catalog_recovers_after_an_unreadable_file_is_repaired(
    tmp_path: pathlib.Path,
) -> None:
    directory = tmp_path / "catalog"
    directory.mkdir()
    path = directory / "catalog.json"
    path.write_text("{", encoding="utf-8")
    catalog = _ExtensionCatalog(directory=directory)
    changed: list[_ExtensionCatalogModel] = []
    failures: list[str] = []
    catalog.changed.connect(changed.append)
    catalog.read_failed.connect(failures.append)
    try:
        assert catalog.load_error is not None
        path.write_text(_ExtensionCatalogModel().model_dump_json(), encoding="utf-8")

        catalog.refresh()

        assert catalog.load_error is None
        assert changed == [_ExtensionCatalogModel()]
        assert failures == []
    finally:
        catalog.close()


def test_catalog_recovers_after_its_directory_becomes_available(
    tmp_path: pathlib.Path,
) -> None:
    directory = tmp_path / "catalog"
    directory.write_text("not a directory", encoding="utf-8")

    catalog = _ExtensionCatalog(directory=directory)
    changed: list[_ExtensionCatalogModel] = []
    catalog.changed.connect(changed.append)
    try:
        assert catalog.model == _ExtensionCatalogModel()
        assert catalog.load_error is not None

        directory.unlink()
        directory.mkdir()
        catalog.refresh()

        assert catalog.load_error is None
        assert changed == [_ExtensionCatalogModel()]
    finally:
        catalog.close()


def test_failed_script_registration_does_not_leave_an_orphaned_source(
    tmp_path: pathlib.Path,
) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    original_path = tmp_path / "original.py"
    _script(original_path)
    store.add_script(original_path, extension_id="shared")
    changed_path = tmp_path / "changed.py"
    changed_source = _script(changed_path, "data + scale")
    changed_source_hash = hashlib.sha256(changed_source).hexdigest()

    with pytest.raises(_ExtensionCatalogConflictError, match="another manager"):
        store.add_script(
            changed_path,
            extension_id="shared",
            expected_record_generation=0,
            check_record_generation=True,
        )

    assert not store.objects_directory.joinpath(f"{changed_source_hash}.py").exists()


def test_unchanged_reload_repairs_corrupt_stored_source(
    tmp_path: pathlib.Path,
) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    script_path = tmp_path / "scale.py"
    source = _script(script_path)
    catalog, revision, _created = store.add_script(script_path)
    object_path = store.recovery_source_path("scale", revision)
    object_path.write_bytes(b"corrupt")

    reloaded, unchanged_revision, created = store.add_script(
        script_path,
        expected_record_generation=catalog.extensions["scale"].record_generation,
    )

    assert not created
    assert unchanged_revision == revision
    assert reloaded.extensions["scale"].source.source_hash == revision
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
    assert record.source.source_hash == revision
    assert record.source.source_path == os.fspath(relocated_path.resolve())
    assert record.record_generation == initial_generation + 1


def test_restored_source_updates_script_source_location(
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
    assert record.source.source_hash == first_revision
    assert record.source.source_path == os.fspath(relocated_path.resolve())


def test_old_source_can_be_registered_as_a_separate_extension(
    tmp_path: pathlib.Path,
) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    script_path = tmp_path / "scale.py"
    first_source = _script(script_path)
    _catalog, first_source_hash, _created = store.add_script(script_path)
    _script(script_path, "data + scale")
    _catalog, current_source_hash, _created = store.add_script(script_path)
    workspace_path = tmp_path / "scale_workspace.py"
    workspace_path.write_bytes(first_source)

    updated, registered_source_hash, created = store.add_script(
        workspace_path,
        extension_id="scale-workspace",
        expected_source_hash=first_source_hash,
    )

    assert created
    assert registered_source_hash == first_source_hash
    assert updated.extensions["scale"].source.source_hash == current_source_hash
    assert updated.extensions["scale-workspace"].source.source_path == str(
        workspace_path.resolve()
    )


def test_failed_validation_does_not_change_the_shared_catalog(
    tmp_path: pathlib.Path,
) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    script_path = tmp_path / "broken.py"
    script_path.write_text("raise RuntimeError('broken import')\n")
    catalog, _source_hash, _created = store.add_script(script_path)
    before = store.path.read_bytes()

    with pytest.raises(erlab.extensions.ExtensionImportError, match="broken import"):
        _validate_and_enable(
            store,
            "broken",
            expected_record_generation=catalog.extensions["broken"].record_generation,
        )

    record = store.read().extensions["broken"]
    assert store.path.read_bytes() == before
    assert record == catalog.extensions["broken"]


def test_environment_validation_failure_stays_in_one_manager_session(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    dependency_name = "extension_test_session_dependency"
    dependency = types.ModuleType(dependency_name)
    sys.modules[dependency_name] = dependency
    script_path = tmp_path / "session_health.py"
    script_path.write_text(
        f"""import {dependency_name}
import xarray as xr
from erlab.extensions import routine

@routine(name="Session Health")
def session_health(data: xr.DataArray) -> xr.DataArray:
    return data
"""
    )
    try:
        with manager_context() as manager:
            store = manager._extensions.catalog.store
            catalog, source_hash, _created = store.add_script(script_path)
            catalog = _validate_and_enable(
                store,
                "session_health",
                expected_record_generation=catalog.extensions[
                    "session_health"
                ].record_generation,
            )
            manager._extensions.catalog.refresh()
            record = catalog.extensions["session_health"]
            shared_catalog = store.read()
            sys.modules.pop(dependency_name)

            with pytest.raises(erlab.extensions.ExtensionImportError):
                manager._extensions.execution.validate_source(
                    record.id,
                    source_hash,
                    expected_record_generation=record.record_generation,
                    enable_extension=False,
                    persist_result=False,
                )

            assert store.read() == shared_catalog
            assert (
                store.capability_status(
                    record.id, source_hash, "routine", "session_health"
                )
                == "ready"
            )
            assert (
                manager._extensions.capability_status(
                    record.id, source_hash, "routine", "session_health"
                )
                == "validation-failed"
            )
            dialog = manager._extensions._manage_dialog
            assert dialog._buttons["toggle"].property("extensionActionState") == "retry"
            assert dialog._buttons["error"].isVisibleTo(dialog)

            sys.modules[dependency_name] = dependency
            manager._extensions.execution.validate_source(
                record.id,
                source_hash,
                expected_record_generation=record.record_generation,
                enable_extension=False,
                persist_result=False,
            )

            assert store.read() == shared_catalog
            assert (
                manager._extensions.execution.validation_error(record.id, source_hash)
                is None
            )
            assert (
                manager._extensions.capability_status(
                    record.id, source_hash, "routine", "session_health"
                )
                == "ready"
            )
    finally:
        sys.modules.pop(dependency_name, None)


def test_retry_validation_enables_a_previously_invalid_script(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    dependency_name = "extension_test_retry_dependency"
    script_path = tmp_path / "retry_health.py"
    script_path.write_text(
        f"""import {dependency_name}
import xarray as xr
from erlab.extensions import routine

@routine(name="Retry Health")
def retry_health(data: xr.DataArray) -> xr.DataArray:
    return data
"""
    )
    try:
        with manager_context() as manager:
            catalog, source_hash, _created = (
                manager._extensions.catalog.store.add_script(script_path)
            )
            manager._extensions.catalog.refresh()
            record = catalog.extensions["retry_health"]
            with pytest.raises(erlab.extensions.ExtensionImportError):
                manager._extensions.execution.validate_and_enable(
                    record.id,
                    expected_record_generation=record.record_generation,
                )

            failed = manager._extensions.catalog.store.read().extensions[record.id]
            assert not failed.enabled
            assert not failed.source.approved
            assert (
                manager._extensions.execution.validation_error(record.id, source_hash)
                is not None
            )

            sys.modules[dependency_name] = types.ModuleType(dependency_name)
            manager._extensions._manage_action("toggle", record.id)

            enabled = manager._extensions.catalog.store.read().extensions[record.id]
            assert enabled.enabled
            assert enabled.source.approved
            assert (
                manager._extensions.execution.validation_error(record.id, source_hash)
                is None
            )
    finally:
        sys.modules.pop(dependency_name, None)


def test_script_routine_generated_code_uses_public_path_api(
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
            source_hash=revision,
            routine_id="scale",
            extension_name="Scale",
            routine_name="Scale",
            parameters={"scale": 3.0},
        )
        data = xr.DataArray([1.0, 2.0])

        xr.testing.assert_identical(operation.apply(data), data * 3.0)
        generated = operation.replay_code("data", output_name="result")
        assert "run_routine" not in generated
        assert "expected_source_hash" not in generated
        assert "manager" not in generated

        second.close()
        xr.testing.assert_identical(operation.apply(data), data * 3.0)
    finally:
        first.close()
        second.close()

    namespace: dict[str, typing.Any] = {"data": data}
    exec(generated, namespace)  # noqa: S102
    xr.testing.assert_identical(namespace["result"], data * 3.0)


def test_script_routine_generated_code_renames_a_conflicting_data_variable(
    tmp_path: pathlib.Path,
) -> None:
    catalog = _ExtensionCatalog(directory=tmp_path / "catalog")
    script_path = tmp_path / "load_script.py"
    _script(script_path)
    try:
        model, source_hash, _created = catalog.store.add_script(script_path)
        _validate_and_enable(
            catalog.store,
            "load_script",
            expected_record_generation=model.extensions[
                "load_script"
            ].record_generation,
        )
        operation = ExtensionRoutineOperation(
            extension_id="load_script",
            source_hash=source_hash,
            routine_id="scale",
            extension_name="load_script.py",
            routine_name="Scale",
            parameters={"scale": 3.0},
        )
        data = xr.DataArray([1.0, 2.0])
        code = operation.replay_code("load_script", output_name="result")
    finally:
        catalog.close()

    namespace: dict[str, typing.Any] = {"load_script": data}
    exec(code, namespace)  # noqa: S102

    imports = [
        node
        for node in ast.parse(code).body
        if isinstance(node, ast.ImportFrom) and node.module == "erlab.extensions"
    ]
    assert len(imports) == 1
    assert [(alias.name, alias.asname) for alias in imports[0].names] == [
        ("load_script", None)
    ]
    xr.testing.assert_identical(namespace["result"], data * 3.0)


def test_script_routine_copied_code_follows_current_stable_capability(
    tmp_path: pathlib.Path,
) -> None:
    catalog = _ExtensionCatalog(directory=tmp_path / "catalog")
    script_path = tmp_path / "lab_routines.py"
    script_path.write_text(
        """import xarray as xr
from erlab.extensions import routine

@routine(id="scale")
def old_scale(data: xr.DataArray, scale: float = 2.0) -> xr.DataArray:
    return data * scale
"""
    )
    try:
        first, old_revision, _created = catalog.store.add_script(script_path)
        first = _validate_and_enable(
            catalog.store,
            "lab_routines",
            expected_record_generation=(
                first.extensions["lab_routines"].record_generation
            ),
        )
        operation = ExtensionRoutineOperation(
            extension_id="lab_routines",
            source_hash=old_revision,
            routine_id="scale",
            extension_name="lab_routines.py",
            routine_name="Scale",
            parameters={"scale": 3.0},
        )
        script_path.write_text(
            """import xarray as xr
from erlab.extensions import routine

@routine(id="scale")
def scale_data(data: xr.DataArray, scale: float = 2.0) -> xr.DataArray:
    return data * scale + 1.0
"""
        )
        second, _new_revision, _created = catalog.store.add_script(
            script_path,
            extension_id="lab_routines",
            expected_record_generation=(
                first.extensions["lab_routines"].record_generation
            ),
            check_record_generation=True,
        )
        _validate_and_enable(
            catalog.store,
            "lab_routines",
            expected_record_generation=(
                second.extensions["lab_routines"].record_generation
            ),
        )
        data = xr.DataArray([1.0, 2.0])
        code = operation.replay_code("data", output_name="result")
    finally:
        catalog.close()

    namespace: dict[str, typing.Any] = {"data": data}
    exec(code, namespace)  # noqa: S102

    xr.testing.assert_identical(namespace["result"], data * 3.0 + 1.0)


def test_unregistered_script_provenance_remains_visible_without_copied_code() -> None:
    operation = ExtensionRoutineOperation(
        extension_id=f"workspace-{uuid.uuid4().hex}",
        source_hash="a" * 64,
        routine_id="scale",
        extension_name="workspace_scale.py",
        routine_name="Scale",
        parameters={"scale": 3.0},
    )

    entry = operation.derivation_entry()

    assert entry.code is None
    assert not entry.copyable


def test_changed_registered_script_provenance_has_no_copied_code(
    tmp_path: pathlib.Path,
) -> None:
    catalog = _ExtensionCatalog(directory=tmp_path / "catalog")
    script_path = tmp_path / "scale.py"
    _script(script_path)
    try:
        model, revision, _created = catalog.store.add_script(script_path)
        _validate_and_enable(
            catalog.store,
            "scale",
            expected_record_generation=model.extensions["scale"].record_generation,
        )
        operation = ExtensionRoutineOperation(
            extension_id="scale",
            source_hash=revision,
            routine_id="scale",
            extension_name="scale.py",
            routine_name="Scale",
            parameters={"scale": 3.0},
        )
        script_path.write_text("changed source")

        entry = operation.derivation_entry()
    finally:
        catalog.close()

    assert entry.code is None
    assert not entry.copyable


def test_extension_routine_reloadability_requires_ready_exact_source() -> None:
    revision = "a" * 64
    operation = ExtensionRoutineOperation(
        extension_id="lab",
        source_hash=revision,
        routine_id="normalize",
        extension_name="Lab",
        routine_name="Normalize",
        parameters={},
    )
    spec = ToolProvenanceSpec(
        kind="script",
        start_label="Create data",
        seed_code="derived = xr.DataArray([1.0])",
        active_name="derived",
        operations=(operation,),
    )
    calls: list[tuple[str, str, str, str]] = []

    def ready(
        extension_id: str,
        source_hash: str,
        capability_kind: str,
        capability_id: str,
    ) -> typing.Literal["ready"]:
        calls.append(
            (
                extension_id,
                source_hash,
                capability_kind,
                capability_id,
            )
        )
        return "ready"

    assert can_reload_without_trust(spec, extension_status_resolver=ready)
    assert calls == [("lab", revision, "routine", "normalize")]
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
        source_hash=revision,
        routine_id="normalize",
        extension_name="Lab",
        routine_name="Normalize",
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
            source_hash="a" * 64,
            routine_id="scale",
            extension_name="Scale",
            routine_name="Scale",
            parameters={"scale": float("inf")},
        )

    with pytest.raises(ValueError, match="must be finite"):
        FileReplayCall(
            kind="extension_loader",
            target="scale",
            source_hash="a" * 64,
            capability_id="load_scale",
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
                embed_policy="always",
            )
        propagated = second.model.extensions["scale"]
        assert propagated.embed_policy == "always"

        with qtbot.waitSignal(second.changed, timeout=3000):
            first.store.set_routine_favorite("scale", "scale", favorite=True)
        assert second.model.routine_favorites == (("scale", "scale"),)

        with qtbot.waitSignal(second.changed, timeout=3000):
            first.store.remove_script(
                "scale", expected_record_generation=propagated.record_generation
            )
        assert "scale" not in second.model.extensions
        assert second.model.routine_favorites == ()
    finally:
        first.close()
        second.close()


def test_registered_script_loader_code_uses_current_user_path(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "original" / "lab_loader.py"
    script_path.parent.mkdir()
    _loader_script(script_path, name="Lab data", extensions=(".txt",))
    data_path = tmp_path / "value.txt"
    data_path.write_text("4")

    with manager_context() as manager:
        catalog, _source_hash, _created = manager._extensions.catalog.store.add_script(
            script_path
        )
        catalog = _validate_and_enable(
            manager._extensions.catalog.store,
            "lab_loader",
            expected_record_generation=(
                catalog.extensions["lab_loader"].record_generation
            ),
        )
        manager._extensions.catalog.refresh()
        record = catalog.extensions["lab_loader"]
        loader_call = manager._extensions._loader_call(
            record,
            record.source,
            record.source.loaders[0],
        )
        resolved = _resolve_load_func(
            (loader_call, {}, FileDataSelection(kind="dataarray"))
        )
        if resolved is None:
            raise RuntimeError("The script loader did not resolve")
        relocated_path = tmp_path / "relocated" / "lab_loader.py"
        relocated_path.parent.mkdir()
        script_path.unlink()
        relocated_path.write_text(
            """from pathlib import Path
import xarray as xr
from erlab.extensions import loader

@loader(id="load_data", extensions=(".txt",))
def read_data(path: Path) -> xr.DataArray:
    return xr.DataArray([2.0 * float(path.read_text())])
"""
        )
        updated, _new_revision, _created = manager._extensions.catalog.store.add_script(
            relocated_path,
            extension_id="lab_loader",
            expected_record_generation=record.record_generation,
            check_record_generation=True,
        )
        _validate_and_enable(
            manager._extensions.catalog.store,
            "lab_loader",
            expected_record_generation=(
                updated.extensions["lab_loader"].record_generation
            ),
        )
        manager._extensions.catalog.refresh()
        spec = file_load(
            start_label="Load lab data",
            seed_code="raise RuntimeError('stale copied code')",
            file_load_source=FileLoadSource(
                path=str(data_path),
                loader_label="Lab data",
                loader_text="lab_loader",
                kwargs_text="",
                load_code="raise RuntimeError('stale copied code')",
                replay_call=resolved.replay_call(),
            ),
        )
        code = spec.display_code()

    if code is None:
        raise RuntimeError("The registered script loader did not generate code")
    namespace: dict[str, typing.Any] = {}
    exec(code, namespace)  # noqa: S102

    xr.testing.assert_identical(namespace["derived"], xr.DataArray([8.0]))


def test_embedded_script_loader_has_replay_metadata_but_no_copied_code(
    tmp_path: pathlib.Path,
) -> None:
    data_path = tmp_path / "value.txt"
    data_path.write_text("4")
    spec = file_load(
        start_label="Load embedded data",
        seed_code="raise RuntimeError('embedded source must not generate code')",
        file_load_source=FileLoadSource(
            path=str(data_path),
            loader_label="Embedded data",
            loader_text="workspace_loader.py",
            kwargs_text="",
            replay_call=FileReplayCall(
                kind="extension_loader",
                target=f"workspace-{uuid.uuid4().hex}",
                source_hash="a" * 64,
                capability_id="load_data",
                selection=FileDataSelection(kind="dataarray"),
            ),
        ),
    )

    assert spec.display_code() is None
    xr.testing.assert_identical(
        replay_file_provenance(
            spec,
            extension_loader_executor=lambda _source: xr.DataArray([4.0]),
        ),
        xr.DataArray([4.0]),
    )


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
    with pytest.raises(ValueError, match="does not match its source"):
        _WorkspaceExtensionRequirement(
            extension_id="lab",
            capability_id="routine",
            capability_kind="routine",
            source_hash="a" * 64,
            extension_api_version=1,
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
        "source_hash": revision,
        "extension_api_version": 1,
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
        catalog, source_hash, _created = manager._extensions.catalog.store.add_script(
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
        source_modified_at = manager._extensions.catalog.model.extensions[
            "scale"
        ].source.source_modified_at

        manager._workspace_controller.saving._save_workspace_document(workspace_path)

    attrs = workspace_arrays._read_workspace_root_attrs_h5py(workspace_path)
    manifest = workspace_format._workspace_manifest_from_attrs(attrs)
    requirements = manifest["extension_requirements"]
    assert len(requirements) == 1
    assert requirements[0]["source_hash"] == source_hash
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
        catalog, source_hash, _created = manager._extensions.catalog.store.add_script(
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
        object_id = f"extension-{source_hash}"
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
        manager._extensions.catalog.store.recovery_source_path(
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
            source_hash=revision,
            routine_id="scale",
            extension_name="Preserved",
            routine_name="Scale",
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
                    source_hash=revision,
                    extension_api_version=1,
                    metadata_snapshot={
                        "source_modified_at": (
                            catalog.extensions["preserved"].source.source_modified_at
                        )
                    },
                    embedded_object_id=object_id,
                    referencing_nodes=(node.uid,),
                ),
            ),
            embedded_sources={("preserved", revision): source},
        )
        manager._extensions.catalog.store.recovery_source_path(
            "preserved", revision
        ).unlink()

        manager._workspace_controller.saving._save_workspace_document(workspace_path)

    attrs = workspace_arrays._read_workspace_root_attrs_h5py(workspace_path)
    manifest = workspace_format._workspace_manifest_from_attrs(attrs)
    assert manifest["extension_requirements"][0]["embedded_object_id"] == object_id
    restored, kind = workspace_storage._read_workspace_blob(workspace_path, object_id)
    assert restored == source
    assert kind == "extension-python-source-v1"


def test_unavailable_script_source_is_omitted_from_gui_discovery(
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
        catalog, _revision, _created = manager._extensions.catalog.store.add_script(
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

        script_path.unlink()
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
        source_hash=revision,
        extension_api_version=1,
        embedded_object_id=f"extension-{revision}",
    )

    with manager_context() as manager:
        manager._extensions.set_workspace_requirements(
            (requirement,),
            embedded_sources={("workspace-only", revision): source},
        )
        assert manager._extensions.resolved_workspace_requirements()[0].state == (
            "missing"
        )

        manager._extensions.set_workspace_requirements(
            (requirement,),
            embedded_sources={("workspace-only", revision): b"different"},
        )
        assert manager._extensions.resolved_workspace_requirements()[0].state == (
            "hash-mismatch"
        )


def test_workspace_requirements_dialog_refreshes_after_registration(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requirement = _WorkspaceExtensionRequirement(
        extension_id="workspace-only",
        capability_id="routine",
        capability_kind="routine",
        source_hash="a" * 64,
        extension_api_version=1,
    )
    current = [_ResolvedWorkspaceRequirement(requirement=requirement, state="missing")]
    shown_dialogs = []

    with manager_context() as manager:
        manager._extensions.set_workspace_requirements(
            (requirement,),
            embedded_sources={(requirement.extension_id, requirement.source_hash): b""},
        )
        monkeypatch.setattr(
            manager._extensions,
            "resolved_workspace_requirements",
            lambda: tuple(current),
        )

        def register(_extension_id: str, _revision: str) -> None:
            current[0] = current[0].model_copy(update={"state": "ready"})

        def execute(dialog) -> int:
            shown_dialogs.append(dialog)
            dialog.tree.setCurrentItem(dialog.tree.topLevelItem(0))
            dialog._register_selected()
            return 0

        monkeypatch.setattr(
            manager._extensions,
            "_save_and_register_embedded_script",
            register,
        )
        monkeypatch.setattr(
            extension_controller._WorkspaceRequirementsDialog, "exec", execute
        )

        manager._extensions.show_workspace_requirements()

        dialog = shown_dialogs[0]
        item = dialog.tree.topLevelItem(0)
        assert item.data(0, QtCore.Qt.ItemDataRole.UserRole + 1) == "ready"
        assert not dialog._register_button.isEnabled()


def test_workspace_script_must_be_saved_before_it_can_run(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    source_path = tmp_path / "source.py"
    source = _script(source_path)
    revision = hashlib.sha256(source).hexdigest()
    requirement = _WorkspaceExtensionRequirement(
        extension_id="workspace-scale",
        capability_id="scale",
        capability_kind="routine",
        source_hash=revision,
        extension_api_version=1,
        metadata_snapshot={"extension_name": "workspace_scale.py"},
        embedded_object_id=f"extension-{revision}",
    )
    destination = tmp_path / "registered" / "workspace_scale.py"
    monkeypatch.setattr(
        extension_controller._SourceReviewDialog,
        "exec",
        lambda _dialog: QtWidgets.QDialog.DialogCode.Accepted,
    )
    monkeypatch.setattr(
        extension_controller.QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(destination), "Python scripts (*.py)"),
    )

    with manager_context() as manager:
        manager._extensions.set_workspace_requirements(
            (requirement,),
            embedded_sources={(requirement.extension_id, revision): source},
        )
        assert (
            manager._extensions.resolved_workspace_requirements()[0].state == "missing"
        )
        assert manager._extensions._save_and_register_embedded_script(
            requirement.extension_id, revision
        )
        record = manager._extensions.catalog.store.read().extensions[
            requirement.extension_id
        ]
        assert destination.read_bytes() == source
        assert record.name == destination.name
        assert record.enabled
        assert record.source.approved
        assert manager._extensions.resolved_workspace_requirements()[0].state == "ready"
        operation = ExtensionRoutineOperation(
            extension_id=requirement.extension_id,
            source_hash=revision,
            routine_id="scale",
            extension_name=record.name,
            routine_name="Scale",
            parameters={"scale": 3.0},
        )
        xr.testing.assert_identical(
            manager._extensions.execution.run_operation(
                operation, xr.DataArray([1.0, 2.0])
            ),
            xr.DataArray([3.0, 6.0]),
        )


def test_workspace_source_conflict_creates_a_separate_registration(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "shared.py"
    historical_source = _script(script_path)
    workspace_source_hash = hashlib.sha256(historical_source).hexdigest()
    destination = tmp_path / "shared_workspace.py"
    monkeypatch.setattr(
        extension_controller._SourceReviewDialog,
        "exec",
        lambda _dialog: QtWidgets.QDialog.DialogCode.Accepted,
    )
    monkeypatch.setattr(
        extension_controller.QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(destination), "Python scripts (*.py)"),
    )

    with manager_context() as manager:
        catalog, stored_source_hash, _created = (
            manager._extensions.catalog.store.add_script(script_path)
        )
        assert stored_source_hash == workspace_source_hash
        catalog = _validate_and_enable(
            manager._extensions.catalog.store,
            "shared",
            expected_record_generation=(catalog.extensions["shared"].record_generation),
        )
        _script(script_path, "data + scale")
        catalog, current_source_hash, _created = (
            manager._extensions.catalog.store.add_script(
                script_path,
                extension_id="shared",
                expected_record_generation=catalog.extensions[
                    "shared"
                ].record_generation,
                check_record_generation=True,
            )
        )
        catalog = _validate_and_enable(
            manager._extensions.catalog.store,
            "shared",
            expected_record_generation=catalog.extensions["shared"].record_generation,
        )
        manager._extensions.catalog.refresh()
        requirement = _WorkspaceExtensionRequirement(
            extension_id="shared",
            capability_id="scale",
            capability_kind="routine",
            source_hash=workspace_source_hash,
            extension_api_version=1,
            metadata_snapshot={"extension_name": "shared.py"},
            embedded_object_id=f"extension-{workspace_source_hash}",
        )
        manager._extensions.set_workspace_requirements(
            (requirement,),
            embedded_sources={("shared", workspace_source_hash): historical_source},
        )
        assert manager._extensions._save_and_register_embedded_script(
            "shared", workspace_source_hash
        )
        records = manager._extensions.catalog.store.read().extensions
        assert records["shared"].source.source_hash == current_source_hash
        workspace_records = [
            record
            for extension_id, record in records.items()
            if extension_id != "shared"
        ]
        assert len(workspace_records) == 1
        workspace_record = workspace_records[0]
        assert workspace_record.enabled
        assert workspace_record.source.source_hash == workspace_source_hash
        assert workspace_record.source.source_path == str(destination.resolve())
        resolved = manager._extensions.resolved_workspace_requirements()[0]
        assert resolved.state == "ready"
        assert resolved.requirement.extension_id == workspace_record.id


def test_canceling_workspace_script_review_does_not_register_source(
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
        source_hash=revision,
        extension_api_version=1,
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
        assert not manager._extensions._save_and_register_embedded_script(
            requirement.extension_id, revision
        )

        assert requirement.extension_id not in (
            manager._extensions.catalog.store.read().extensions
        )
        assert (
            manager._extensions.resolved_workspace_requirements()[0].state == "missing"
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


def test_missing_script_prompt_locates_an_identical_script(
    manager_context,
    qtbot: pytest.QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    original_path = tmp_path / "original" / "analysis.py"
    original_path.parent.mkdir()
    source = _script(original_path)
    relocated_path = tmp_path / "relocated" / "analysis.py"
    relocated_path.parent.mkdir()
    relocated_path.write_bytes(source)

    with manager_context() as manager:
        catalog, revision, _created = manager._extensions.catalog.store.add_script(
            original_path
        )
        _validate_and_enable(
            manager._extensions.catalog.store,
            "analysis",
            expected_record_generation=catalog.extensions["analysis"].record_generation,
        )
        manager._extensions.catalog.refresh()
        original_path.unlink()
        monkeypatch.setattr(
            extension_controller.QtWidgets.QFileDialog,
            "getOpenFileName",
            lambda *_args, **_kwargs: (
                str(relocated_path),
                "Python scripts (*.py)",
            ),
        )

        manager._extensions._prompt_for_missing_scripts()
        dialog = manager._extensions._missing_scripts_dialog
        if dialog is None:
            raise RuntimeError("The missing-script dialog was not shown")
        assert dialog.tree.topLevelItemCount() == 1
        dialog.locate_button.click()
        qtbot.wait_until(
            lambda: manager._extensions._missing_scripts_dialog is None,
            timeout=3000,
        )

        assert (
            manager._extensions.catalog.store.executable_source_path(
                "analysis", revision
            )
            == relocated_path.resolve()
        )


def test_locating_a_changed_script_uses_the_review_flow(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    original_path = tmp_path / "analysis.py"
    _script(original_path)
    changed_path = tmp_path / "changed.py"
    _script(changed_path, "data + scale")
    reviews: list[tuple[pathlib.Path, str | None]] = []

    with manager_context() as manager:
        catalog, _revision, _created = manager._extensions.catalog.store.add_script(
            original_path
        )
        _validate_and_enable(
            manager._extensions.catalog.store,
            "analysis",
            expected_record_generation=catalog.extensions["analysis"].record_generation,
        )
        manager._extensions.catalog.refresh()
        original_path.unlink()
        monkeypatch.setattr(
            extension_controller.QtWidgets.QFileDialog,
            "getOpenFileName",
            lambda *_args, **_kwargs: (str(changed_path), "Python scripts (*.py)"),
        )
        monkeypatch.setattr(
            manager._extensions,
            "_review_and_add",
            lambda path, *, extension_id=None: (
                reviews.append((path, extension_id)) or True
            ),
        )

        assert manager._extensions._locate_missing_script("analysis")

    assert reviews == [(changed_path.resolve(), "analysis")]


def test_restoring_a_missing_script_preserves_the_recovery_source(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    original_path = tmp_path / "analysis.py"
    source = _script(original_path)
    destination = tmp_path / "restored" / "analysis.py"

    with manager_context() as manager:
        catalog, revision, _created = manager._extensions.catalog.store.add_script(
            original_path
        )
        _validate_and_enable(
            manager._extensions.catalog.store,
            "analysis",
            expected_record_generation=catalog.extensions["analysis"].record_generation,
        )
        manager._extensions.catalog.refresh()
        original_path.unlink()
        monkeypatch.setattr(
            extension_controller.QtWidgets.QFileDialog,
            "getSaveFileName",
            lambda *_args, **_kwargs: (str(destination), "Python scripts (*.py)"),
        )

        assert manager._extensions._restore_missing_script("analysis")

        assert destination.read_bytes() == source
        assert (
            manager._extensions.catalog.store.recovery_source_path(
                "analysis", revision
            ).read_bytes()
            == source
        )
        assert (
            manager._extensions.catalog.store.executable_source_path(
                "analysis", revision
            )
            == destination.resolve()
        )


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
    validation_refreshes: list[None] = []
    monkeypatch.setattr(
        extension_controller.QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (dialog_calls.append(None) or "", ""),
    )

    with manager_context() as manager:
        monkeypatch.setattr(
            manager._extensions,
            "_refresh_extension_state_views",
            lambda: validation_refreshes.append(None),
        )
        manager._extensions.add_script_action.trigger()
        manager._extensions.execution.validation_changed.emit()
        assert dialog_calls == [None]
        assert validation_refreshes == [None]

        manager._extensions.close()
        manager._extensions.add_script_action.trigger()
        manager._extensions.execution.validation_changed.emit()
        assert dialog_calls == [None]
        assert validation_refreshes == [None]


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
        dialog = manager._extensions._manage_dialog
        dialog.set_catalog(
            _ExtensionCatalogModel(extensions={script_record.id: script_record}),
            {
                ("script", script_record.source.source_hash): (
                    "Stored source; original unchanged"
                )
            },
            managed_paths={
                ("script", script_record.source.source_hash): os.fspath(
                    manager._extensions.catalog.store.recovery_source_path(
                        "script", script_record.source.source_hash
                    )
                )
            },
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
        assert dialog.embedding_combo.isEnabled()
        assert dialog._buttons["view_source"].isEnabled()

        dialog.set_removal_reason("Close Manager 2 first.")
        assert not dialog._buttons["remove"].isEnabled()
        assert dialog._buttons["remove"].toolTip()
        assert not dialog.removal_reason_label.isHidden()


def test_script_removal_confirmation_is_permanent_and_preserves_original(
    manager_context,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = tmp_path / "restorable.py"
    _script(script_path)

    with manager_context() as manager:
        manager._extensions.catalog.store.add_script(script_path)
        manager._extensions.catalog.refresh()
        record = manager._extensions.catalog.model.extensions["restorable"]
        managed_path = manager._extensions.catalog.store.recovery_source_path(
            record.id, record.source.source_hash
        )
        responses = iter(
            (
                QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Yes,
            )
        )
        monkeypatch.setattr(
            manager._extensions, "_removal_blocker", lambda _extension_id: None
        )
        monkeypatch.setattr(
            QtWidgets.QMessageBox, "exec", lambda _dialog: next(responses)
        )

        manager._extensions._manage_action("remove", "restorable")
        assert "restorable" in manager._extensions.catalog.model.extensions
        assert managed_path.is_file()

        manager._extensions._manage_action("remove", "restorable")
        assert "restorable" not in manager._extensions.catalog.model.extensions
        assert not managed_path.exists()
        assert script_path.is_file()


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
                    source_hash=revision,
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
        missing_source = spec.model_copy(
            update={
                "file_load_source": load_source.model_copy(
                    update={
                        "replay_call": load_source.replay_call.model_copy(
                            update={"source_hash": "b" * 64}
                        )
                    }
                )
            }
        )
        assert file_load_source_status(missing_source) == ("extension-missing-source")
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

        script_path.write_bytes(b"changed source")
        assert file_load_source_status(spec) == "extension-hash-mismatch"
        assert not marker.exists()


def test_file_source_status_reports_extension_validation_failure(
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
            manager._extensions.execution.validate_and_enable(
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
                    source_hash=revision,
                    capability_id="broken_loader",
                    selection=FileDataSelection(kind="dataarray"),
                ),
            ),
        )

        assert (
            file_load_source_status(
                spec,
                extension_status_resolver=manager._extensions.capability_status,
            )
            == "extension-validation-failed"
        )


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
            source_hash=revision,
            extension_api_version=1,
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
        broken_path = tmp_path / "broken.py"
        broken_path.write_text("raise RuntimeError('broken import')\n")
        catalog, broken_revision, _created = (
            manager._extensions.catalog.store.add_script(broken_path)
        )
        with pytest.raises(erlab.extensions.ExtensionImportError):
            manager._extensions.execution.validate_and_enable(
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
                        "source_hash": broken_revision,
                    }
                ),
            )
        )
        assert manager._extensions.resolved_workspace_requirements()[0].state == (
            "validation-failed"
        )


@pytest.mark.parametrize(
    ("stored_source", "expected_state"),
    [(None, "missing"), (b"corrupt", "hash-mismatch")],
)
def test_embedded_source_does_not_mask_unusable_registered_source(
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
        if stored_source is None:
            script_path.unlink()
        else:
            script_path.write_bytes(stored_source)
        manager._extensions.catalog.refresh()
        manager._extensions.set_workspace_requirements(
            (
                _WorkspaceExtensionRequirement(
                    extension_id="catalog_source",
                    capability_id="scale",
                    capability_kind="routine",
                    source_hash=revision,
                    extension_api_version=1,
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
        "source_hash": "a" * 64,
        "extension_api_version": 1,
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
        source_hash="a" * 64,
        extension_api_version=1,
    )
    previous = incoming.model_copy(
        update={"extension_id": "previous", "source_hash": "b" * 64}
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
                (previous.extension_id, previous.source_hash): previous_source
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
            (previous.extension_id, previous.source_hash): previous_source
        }
        assert restored[2] == {}
        assert restored[3] == unresolved
        assert manager._workspace_state.save_as_only
        assert manager._workspace_state.degraded_reasons == ("previous: missing",)


def test_removing_node_discards_only_its_workspace_requirements(
    manager_context,
) -> None:
    operation = ExtensionRoutineOperation(
        extension_id="missing-extension",
        source_hash="a" * 64,
        routine_id="normalize",
        extension_name="Missing Extension",
        routine_name="Normalize",
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
            source_hash="a" * 64,
            extension_api_version=1,
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
        source_hash=revision,
        routine_id="normalize",
        extension_name="Workspace Routines",
        routine_name="Normalize",
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
            source_hash=revision,
            extension_api_version=1,
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
        source_hash=revision,
        routine_id="normalize",
        extension_name="Shared Routines",
        routine_name="Normalize",
        parameters={},
    )

    with manager_context() as manager:
        script_path = tmp_path / "shared_routines.py"
        script_path.write_bytes(source)
        manager._extensions.catalog.store.add_script(
            script_path,
            extension_id="shared-routines",
            expected_source_hash=revision,
        )
        manager._extensions.catalog.refresh()
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
            source_hash=revision,
            extension_api_version=1,
            metadata_snapshot={
                "author": "Existing Author",
                "change_summary": "Obsolete source note",
            },
            referencing_nodes=("unresolved-existing",),
        )
        incoming = base.model_copy(
            update={
                "metadata_snapshot": {
                    "contact": "incoming@example.org",
                    "changelog": "Obsolete source history",
                },
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
        "extension_name": "shared_routines.py",
        "routine_name": "Normalize",
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
        source_hash=revision,
        extension_api_version=1,
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


def test_workspace_requirements_include_nested_script_inputs(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    revision = "c" * 64
    data_path = tmp_path / "nested.txt"
    data_path.write_text("unused")
    operation = ExtensionRoutineOperation(
        extension_id="nested-routines",
        source_hash=revision,
        routine_id="normalize",
        extension_name="Nested Routines",
        routine_name="Normalize",
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
                source_hash=revision,
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
        "source_hash": revision,
        "extension_api_version": 1,
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
        manager._extensions.catalog.store.recovery_source_path(
            record.id, record.source.source_hash
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
        source_hash=revision,
        extension_api_version=1,
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
        "source_hash": revision,
        "extension_api_version": 1,
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
        "source_hash": "a" * 64,
        "extension_api_version": 1,
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


def test_save_as_only_rejects_the_original_workspace_path(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    workspace_path = tmp_path / "degraded.itws"
    workspace_path.write_bytes(b"original")
    errors: list[tuple[str, str]] = []
    finished: list[bool] = []

    with manager_context() as manager:
        controller = manager._workspace_controller
        manager._workspace_state.path = workspace_path.resolve()
        manager._workspace_state.save_as_only = True
        monkeypatch.setattr(
            controller,
            "_workspace_save_dialog",
            lambda **_kwargs: workspace_path,
        )
        monkeypatch.setattr(
            manager,
            "_show_operation_error",
            lambda title, text: errors.append((title, text)),
        )
        monkeypatch.setattr(
            controller,
            "save",
            lambda **_kwargs: pytest.fail(
                "a degraded workspace must not save to its original path"
            ),
        )

        assert not controller.save_as(native=False, on_finished=finished.append)

    assert len(errors) == 1
    assert finished == [False]
    assert workspace_path.read_bytes() == b"original"


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
        "source_hash": revision,
        "extension_api_version": 1,
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
        assert manager._extensions.source_bytes("imported-lab", revision) == source
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
        source_hash=revision,
        extension_api_version=1,
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

        assert manager._extensions.source_bytes("shared-lab", revision) == valid_source
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
            source_hash=revision,
            routine_id="normalize",
            extension_name=extension_id,
            routine_name="Normalize",
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
        "source_hash": revision,
        "extension_api_version": 1,
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
        source_hash="a" * 64,
        extension_api_version=1,
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

        calls: list[tuple[str, str, str, str]] = []
        original_status = manager._extensions.capability_status

        def capability_status(
            extension_id: str,
            source_hash: str,
            kind: str,
            capability_id: str,
        ) -> str:
            calls.append((extension_id, source_hash, kind, capability_id))
            return original_status(extension_id, source_hash, kind, capability_id)

        monkeypatch.setattr(
            manager._extensions,
            "capability_status",
            capability_status,
        )
        assert tool.slicer_area._direct_reloadable()
        assert calls == [("reload_loader", _revision, "loader", "load_data")]
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
        rejected = manager._extensions.catalog.model.extensions["second"]
        assert not rejected.enabled
        assert (
            rejected.record_generation == second.extensions["second"].record_generation
        )


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
        rejected = manager._extensions.catalog.model.extensions["netcdf"]
        assert not rejected.enabled
        assert (
            rejected.record_generation == catalog.extensions["netcdf"].record_generation
        )


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
        assert operation.source_hash == revision

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
            source_hash=revision,
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
        source_hash="a" * 64,
        loader_id="load",
        descriptor=erlab.extensions.LoaderDescriptor(
            id="load",
            name="Load",
            category="Lab",
            summary="",
            function_name="load",
        ),
        source_path=tmp_path / "missing.py",
        executor=lambda *_args: xr.DataArray([1.0]),
    )
    worker = _ExtensionLoaderWorker(
        call,
        tmp_path / "data.txt",
        {},
        _ExtensionCatalogStore(tmp_path / "catalog"),
        {},
        source_is_healthy=lambda *_args: True,
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
        assert failed == catalog.extensions["stops"]
        assert "SystemExit: extension requested exit" in (
            manager._extensions.execution.validation_error(
                "stops", failed.source.source_hash
            )
            or ""
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
        source_hash="a" * 64,
        loader_id="load_data",
        descriptor=descriptor,
        source_path=pathlib.Path("lab.py"),
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
        assert record.source.approved


def test_reviewing_an_unchanged_unapproved_script_enables_it(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "reviewed.py"
    _script(script_path)

    with manager_context() as manager:
        manager._extensions.catalog.store.add_script(script_path)
        manager._extensions.catalog.refresh()
        monkeypatch.setattr(
            extension_controller._SourceReviewDialog,
            "exec",
            lambda _dialog: QtWidgets.QDialog.DialogCode.Accepted,
        )

        assert manager._extensions._review_and_add(script_path)

        record = manager._extensions.catalog.model.extensions["reviewed"]
        assert record.enabled
        assert record.source.approved


def test_manage_reload_paths(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "reloadable.py"
    _script(script_path)
    reviews: list[tuple[pathlib.Path, str | None]] = []
    restores: list[str] = []

    with manager_context() as manager:
        manager._extensions._manage_action("reload", "unknown")
        catalog, _source_hash, _created = manager._extensions.catalog.store.add_script(
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
        embedded_source = record.source.model_copy(update={"source_path": None})
        manager._extensions.catalog.model = _ExtensionCatalogModel(
            extensions={
                "reloadable": record.model_copy(update={"source": embedded_source})
            }
        )
        monkeypatch.setattr(
            manager._extensions,
            "_restore_missing_script",
            lambda extension_id: restores.append(extension_id) or True,
        )
        manager._extensions._manage_action("reload", "reloadable")

    assert restores == ["reloadable"]


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
        monkeypatch.setattr(manager, "_update_info", lambda: calls.append("details"))

        try:
            controller._catalog_changed(controller.catalog.model)
        finally:
            manager._tool_graph.nodes.pop("extension-test-tool")

    assert calls == ["menu", "explorer", "actions", "details", "tool"]


def test_workspace_resolution_distinguishes_missing_exact_sources(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_hash = "b" * 64
    requested_hash = hashlib.sha256(b"requested source").hexdigest()
    current_source = _ExtensionSource(
        source_hash=current_hash,
        object_name=f"{current_hash}.py",
        registered_at="2026-01-01T00:00:00+00:00",
        approved=True,
    )
    script_record = _ExtensionRecord(
        id="lab",
        name="Lab",
        enabled=True,
        source=current_source,
    )
    requirement = _WorkspaceExtensionRequirement(
        extension_id="lab",
        capability_id="analyze",
        capability_kind="routine",
        source_hash=requested_hash,
        extension_api_version=1,
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
            "missing"
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
        assert missing.detail == "The required source is not registered"


def test_workspace_requirement_helpers_cover_empty_and_unavailable_nodes(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requirement = _WorkspaceExtensionRequirement(
        extension_id="lab",
        capability_id="analyze",
        capability_kind="routine",
        source_hash="a" * 64,
        extension_api_version=1,
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
        catalog, source_hash, _created = manager._extensions.catalog.store.add_script(
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
            source_hash=source_hash,
            routine_id="scale",
            extension_name="Always",
            routine_name="Scale",
            parameters={},
        )
        manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(xr.DataArray([1.0])),
            show=False,
            provenance_spec=full_data(operation),
        )

        requirements = manager._extensions.collect_workspace_requirements()

    assert len(requirements) == 1
    assert requirements[0].embedded_object_id == f"extension-{source_hash}"
    assert "source_modified_at" in requirements[0].metadata_snapshot


def test_workspace_registration_selects_the_script_requirement(
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
        source_hash=revision,
        extension_api_version=1,
        embedded_object_id=f"extension-{revision}",
    )
    monkeypatch.setattr(
        extension_controller._SourceReviewDialog,
        "exec",
        lambda _dialog: QtWidgets.QDialog.DialogCode.Accepted,
    )
    destination = tmp_path / "registered_mixed.py"
    monkeypatch.setattr(
        extension_controller.QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(destination), "Python scripts (*.py)"),
    )

    with manager_context() as manager:
        manager._extensions.set_workspace_requirements(
            (base,),
            embedded_sources={("mixed", revision): source},
        )
        assert manager._extensions._save_and_register_embedded_script("mixed", revision)
        requirements = manager._extensions.collect_workspace_requirements()

        assert (
            manager._extensions.capability_status("mixed", revision, "routine", "scale")
            == "ready"
        )
        assert (
            manager._extensions.catalog.model.extensions["mixed"].name
            == destination.name
        )
        assert {item.capability_kind for item in requirements} == {"routine"}
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
def test_workspace_registration_rejects_unusable_workspace_state(
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
        source_hash=revision,
        extension_api_version=1,
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
        assert not manager._extensions._save_and_register_embedded_script(
            "unusable", revision
        )

    assert bool(warnings) is warning_expected


def test_workspace_registration_reports_validation_failure(
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
        source_hash=revision,
        extension_api_version=1,
    )
    failures: list[str] = []
    monkeypatch.setattr(
        extension_controller._SourceReviewDialog,
        "exec",
        lambda _dialog: QtWidgets.QDialog.DialogCode.Accepted,
    )
    destination = tmp_path / "registered_failure.py"
    monkeypatch.setattr(
        extension_controller.QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(destination), "Python scripts (*.py)"),
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
            "validate_source",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("validation failed")
            ),
        )
        assert not manager._extensions._save_and_register_embedded_script(
            "failure", revision
        )

    assert failures == ["The saved workspace extension could not be registered."]


def test_workspace_notification_repeats_missing_script_recovery(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "analysis.py"
    _script(script_path)

    with manager_context() as manager:
        catalog, revision, _created = manager._extensions.catalog.store.add_script(
            script_path
        )
        _validate_and_enable(
            manager._extensions.catalog.store,
            "analysis",
            expected_record_generation=catalog.extensions["analysis"].record_generation,
        )
        manager._extensions.catalog.refresh()
        script_path.unlink()
        manager._extensions.set_workspace_requirements(
            (
                _WorkspaceExtensionRequirement(
                    extension_id="analysis",
                    capability_id="scale",
                    capability_kind="routine",
                    source_hash=revision,
                    extension_api_version=1,
                ),
            )
        )
        manager._extensions._missing_script_prompt_shown = True

        manager._extensions.notify_unavailable_workspace_requirements()

        dialog = manager._extensions._missing_scripts_dialog
        if dialog is None:
            raise RuntimeError("The workspace did not repeat script recovery")
        assert dialog.tree.topLevelItemCount() == 1
        dialog.close()


def test_workspace_notification_opens_embedded_script_recovery(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    source = _script(tmp_path / "workspace_scale.py")
    revision = hashlib.sha256(source).hexdigest()
    requirement = _WorkspaceExtensionRequirement(
        extension_id="workspace-scale",
        capability_id="scale",
        capability_kind="routine",
        source_hash=revision,
        extension_api_version=1,
        embedded_object_id=f"extension-{revision}",
    )
    shown: list[None] = []

    with manager_context() as manager:
        manager._extensions.set_workspace_requirements(
            (requirement,),
            embedded_sources={(requirement.extension_id, revision): source},
        )
        monkeypatch.setattr(
            manager._extensions,
            "show_workspace_requirements",
            lambda: shown.append(None),
        )

        manager._extensions.notify_unavailable_workspace_requirements()

    assert shown == [None]


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
        source_hash=hashlib.sha256(source).hexdigest(),
        loader_id="load_data",
        descriptor=descriptor,
        source_path=script_path,
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
            source_hash=revision,
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
            source_hash=revision,
            loader_id="load_data",
            descriptor=loader_descriptor,
            source_path=script_path,
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


def test_extension_menus_group_routines_and_follow_manager_selection(
    manager_context,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = tmp_path / "scale.py"
    _script(script_path)

    with manager_context() as manager:
        controller = manager._extensions
        menu = controller.menu
        if menu is None:
            raise TypeError("The extension menu must exist")
        actions = menu.actions()
        assert actions[0].objectName() == "manager_run_extension_routine_action"
        assert actions[1].isSeparator()
        assert "manager_extension_jobs_action" not in {
            action.objectName() for action in actions
        }
        assert not any(
            first.isSeparator() and second.isSeparator()
            for first, second in itertools.pairwise(actions)
        )

        catalog, _revision, _created = controller.catalog.store.add_script(script_path)
        _validate_and_enable(
            controller.catalog.store,
            "scale",
            expected_record_generation=catalog.extensions["scale"].record_generation,
        )
        controller.catalog.store.set_routine_favorite("scale", "scale", favorite=True)
        controller.catalog.refresh()
        controller._recent.append(("scale", "scale"))

        def active_actions(current_menu: QtWidgets.QMenu) -> list[QtGui.QAction]:
            values: list[QtGui.QAction] = []
            for action in current_menu.actions():
                values.append(action)
                submenu = action.menu()
                if submenu is not None:
                    values.extend(active_actions(submenu))
            return values

        context_menu = manager.tree_view._extensions_menu
        for selection, expected in zip(
            ((), (0,), (0, 1)), (False, True, False), strict=True
        ):
            monkeypatch.setattr(
                manager, "_selected_imagetool_targets", lambda value=selection: value
            )
            controller._populate_menu()
            controller._populate_routine_menu(context_menu, compact=True)
            required_actions = [
                action
                for current_menu in (menu, context_menu)
                for action in active_actions(current_menu)
                if action.property("requiresImageTool") is True
            ]
            assert required_actions
            assert all(action.isEnabled() is expected for action in required_actions)
            assert controller.add_script_action.isEnabled()
            assert controller.manage_action.isEnabled()
            assert controller.requirements_action.isEnabled()

        actions = menu.actions()
        assert actions[0].objectName() == "manager_run_extension_routine_action"
        assert actions[1].isSeparator()
        assert not any(
            first.isSeparator() and second.isSeparator()
            for first, second in itertools.pairwise(actions)
        )


def test_manage_dialog_is_flat_searchable_and_preserves_selection(
    qtbot: pytest.QtBot,
) -> None:
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    records: dict[str, _ExtensionRecord] = {}
    for index in range(24):
        source_hash = f"{index + 1:064x}"
        records[f"extension-{index:02d}"] = _ExtensionRecord(
            id=f"extension-{index:02d}",
            name=f"Extension {index:02d}",
            source=_ExtensionSource(
                source_hash=source_hash,
                object_name=f"{source_hash}.py",
                registered_at="2026-01-01T00:00:00+00:00",
            ),
        )
    dialog = extension_dialogs._ManageExtensionsDialog(parent)
    dialog.resize(900, 260)
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.set_catalog(_ExtensionCatalogModel(extensions=records))

    assert dialog.tree.columnCount() == 4
    assert dialog.tree.isSortingEnabled()
    assert all(
        dialog.tree.topLevelItem(index).childCount() == 0
        for index in range(dialog.tree.topLevelItemCount())
    )
    selected = next(
        dialog.tree.topLevelItem(index)
        for index in range(dialog.tree.topLevelItemCount())
        if dialog.tree.topLevelItem(index).data(0, QtCore.Qt.ItemDataRole.UserRole)
        == "extension-12"
    )
    dialog.tree.setCurrentItem(selected)
    scroll_bar = dialog.tree.verticalScrollBar()
    if scroll_bar is None:
        raise TypeError("The extension list must have a scroll bar")
    scroll_bar.setValue(scroll_bar.maximum())
    scroll_position = scroll_bar.value()

    dialog.set_catalog(
        _ExtensionCatalogModel(extensions=dict(reversed(records.items())))
    )

    assert dialog.selected_extension_id == "extension-12"
    assert scroll_bar.value() == scroll_position
    dialog.search_edit.setText("extension 05")
    visible = [
        dialog.tree.topLevelItem(index)
        for index in range(dialog.tree.topLevelItemCount())
        if not dialog.tree.topLevelItem(index).isHidden()
    ]
    assert len(visible) == 1
    assert visible[0].data(0, QtCore.Qt.ItemDataRole.UserRole) == "extension-05"
    assert dialog.selected_extension_id == "extension-05"


def test_extension_source_viewer_uses_python_editor(
    qtbot: pytest.QtBot,
) -> None:
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    viewer = extension_dialogs._SourceViewerDialog(
        "value = 1\n", parent, title="Source"
    )
    qtbot.addWidget(viewer)
    assert isinstance(viewer.source, erlab.interactive.utils.PythonCodeEditor)
    assert viewer.source.isReadOnly()


def test_removal_blockers_cover_managers_jobs_and_workspace_requirements(
    manager_context,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = tmp_path / "blocked.py"
    _script(script_path)
    with manager_context() as manager:
        controller = manager._extensions
        controller.catalog.store.add_script(script_path)
        controller.catalog.refresh()
        current_id = manager._manager_record.internal_id
        other = types.SimpleNamespace(
            internal_id="other",
            index=7,
            workspace_path=os.fspath(tmp_path / "other.itws"),
        )
        current = types.SimpleNamespace(
            internal_id=current_id,
            index=manager.manager_index,
            workspace_path=None,
        )
        monkeypatch.setattr(
            extension_controller, "live_manager_records", lambda: (current, other)
        )
        assert controller._removal_blocker("blocked") is not None

        monkeypatch.setattr(
            extension_controller, "live_manager_records", lambda: (current,)
        )
        monkeypatch.setattr(
            controller.execution, "uses_extension", lambda extension_id: True
        )
        assert controller._removal_blocker("blocked") is not None

        monkeypatch.setattr(
            controller.execution, "uses_extension", lambda extension_id: False
        )
        controller.set_workspace_requirements(
            (
                _WorkspaceExtensionRequirement(
                    extension_id="blocked",
                    capability_id="scale",
                    capability_kind="routine",
                    source_hash=controller.catalog.model.extensions[
                        "blocked"
                    ].source.source_hash,
                    extension_api_version=1,
                ),
            )
        )
        assert controller._removal_blocker("blocked") is not None
        controller.set_workspace_requirements(())
        assert controller._removal_blocker("blocked") is None


def test_permanent_removal_preserves_shared_objects_and_rolls_back_failed_commit(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_path = tmp_path / "first.py"
    second_path = tmp_path / "second.py"
    source = _script(first_path)
    second_path.write_bytes(source)
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    catalog, revision, _created = store.add_script(first_path, extension_id="first")
    catalog, _revision, _created = store.add_script(second_path, extension_id="second")
    object_path = store.recovery_source_path("first", revision)

    with pytest.raises(_ExtensionCatalogConflictError, match="another manager"):
        store.remove_script(
            "first",
            expected_record_generation=(
                catalog.extensions["first"].record_generation + 1
            ),
        )

    catalog, retained = store.remove_script(
        "first",
        expected_record_generation=catalog.extensions["first"].record_generation,
    )
    assert retained is None
    assert object_path.is_file()
    assert first_path.is_file()
    assert "first" not in catalog.extensions

    original_write = store._write_unlocked

    def fail_write(_catalog: _ExtensionCatalogModel) -> typing.Never:
        raise _ExtensionCatalogConflictError("commit failed")

    monkeypatch.setattr(store, "_write_unlocked", fail_write)
    with pytest.raises(_ExtensionCatalogConflictError, match="commit failed"):
        store.remove_script(
            "second",
            expected_record_generation=catalog.extensions["second"].record_generation,
        )
    assert object_path.is_file()
    assert "second" in store.read().extensions
    assert not tuple(store.directory.glob(".removal-*"))

    monkeypatch.setattr(store, "_write_unlocked", original_write)
    catalog, retained = store.remove_script(
        "second",
        expected_record_generation=catalog.extensions["second"].record_generation,
    )
    assert retained is None
    assert not object_path.exists()
    assert second_path.is_file()
    assert catalog.extensions == {}


def test_permanent_removal_reports_retained_cleanup_path(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = tmp_path / "cleanup.py"
    _script(script_path)
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    catalog, revision, _created = store.add_script(script_path)
    original_cleanup = extension_catalog.shutil.rmtree

    def fail_cleanup(_path: pathlib.Path) -> typing.Never:
        raise OSError("cleanup failed")

    monkeypatch.setattr(extension_catalog.shutil, "rmtree", fail_cleanup)
    catalog, retained = store.remove_script(
        "cleanup",
        expected_record_generation=catalog.extensions["cleanup"].record_generation,
    )

    assert retained is not None
    assert retained.is_dir()
    assert "cleanup" not in catalog.extensions
    assert not store.objects_directory.joinpath(f"{revision}.py").exists()
    original_cleanup(retained)
