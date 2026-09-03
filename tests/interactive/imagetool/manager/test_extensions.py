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
import pydantic
import pytest
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

import erlab
import erlab.extensions._api as extension_api
import erlab.interactive.imagetool.manager._base as manager_base
import erlab.interactive.imagetool.manager._extensions._catalog as extension_catalog
import erlab.interactive.imagetool.manager._extensions._dialogs as extension_dialogs
import erlab.interactive.imagetool.manager._extensions._execution as extension_execution
import erlab.interactive.imagetool.manager._io as manager_io
from erlab.extensions._models import _script_name_key
from erlab.interactive.imagetool._load_source import (
    _load_source_details_from_provenance,
    _resolve_load_func,
)
from erlab.interactive.imagetool._provenance._execution import (
    can_reload_with_trusted_code,
    can_reload_without_trust,
    file_load_source_status,
    replay_file_provenance,
    replay_script_provenance,
)
from erlab.interactive.imagetool._provenance._graph import (
    ReplayGraphError,
    compile_replay_graph,
    emit_replay_code,
)
from erlab.interactive.imagetool._provenance._model import (
    FileDataSelection,
    FileLoadSource,
    FileReplayCall,
    ScriptInput,
    ToolProvenanceSpec,
    file_load,
    full_data,
    script,
)
from erlab.interactive.imagetool._provenance._operations import (
    ExtensionRoutineOperation,
    GaussianFilterOperation,
    ScriptCodeOperation,
)
from erlab.interactive.imagetool.manager._extensions import (
    _controller as extension_controller,
)
from erlab.interactive.imagetool.manager._extensions._catalog import (
    _ExtensionCatalog,
    _ExtensionCatalogConflictError,
    _ExtensionCatalogStore,
    _PinnedScript,
)
from erlab.interactive.imagetool.manager._extensions._dialogs import (
    _ExtensionParameterDialog,
    _RoutineSelectionDialog,
)
from erlab.interactive.imagetool.manager._extensions._execution import (
    _detached_routine_output,
    _ExtensionLoaderCall,
    _ExtensionLoaderWorker,
    _ExtensionRoutineWorker,
    _ExtensionValidationWorker,
    _readonly_array,
    _validate_script_snapshot,
)
from erlab.interactive.imagetool.manager._extensions._models import (
    _ExtensionCatalogModel,
    _ResolvedWorkspaceRequirement,
    _ScriptRecord,
    _WorkspaceScriptRequirement,
)
from erlab.interactive.imagetool.manager._provenance_edit import (
    _controller as provenance_edit_controller,
)
from erlab.interactive.imagetool.manager._workspace import _arrays as workspace_arrays
from erlab.interactive.imagetool.manager._workspace import _format as workspace_format
from erlab.interactive.imagetool.manager._workspace import _state as workspace_state
from erlab.interactive.imagetool.manager._workspace import _store as workspace_store
from erlab.interactive.imagetool.manager._wrapper import _ManagedWindowNode

if typing.TYPE_CHECKING:
    from collections.abc import Callable


class _ExtensionInputToolState(pydantic.BaseModel):
    value: int = 0


class _ExtensionInputTool(erlab.interactive.utils.ToolWindow[_ExtensionInputToolState]):
    StateModel = _ExtensionInputToolState
    tool_name = "extension-input-test"

    def __init__(self, data: xr.DataArray) -> None:
        super().__init__()
        self._data = data
        self._status = _ExtensionInputToolState()

    @property
    def tool_data(self) -> xr.DataArray:
        return self._data

    @property
    def tool_status(self) -> _ExtensionInputToolState:
        return self._status

    @tool_status.setter
    def tool_status(self, status: _ExtensionInputToolState) -> None:
        self._status = status


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
    script_name: str,
    *,
    expected_record_generation: int,
) -> _ExtensionCatalogModel:
    record = store.read().extensions[_script_name_key(script_name)]
    snapshot = store.resolve_script(record.script_name, record.source_hash)
    manager_session_id = f"test-manager-{uuid.uuid4().hex}"
    try:
        return _validate_script_snapshot(
            store,
            snapshot,
            expected_record_generation=expected_record_generation,
            manager_session_id=manager_session_id,
            script_modules={},
        )
    finally:
        extension_execution._remove_manager_modules(manager_session_id)


def _pinned_script(
    path: pathlib.Path,
    source: bytes = b"",
    *,
    routines: tuple[erlab.extensions.RoutineDescriptor, ...] = (),
    loaders: tuple[erlab.extensions.LoaderDescriptor, ...] = (),
    approved: bool = True,
    enabled: bool = True,
    catalog_generation: int = 1,
) -> _PinnedScript:
    """Build one valid pinned script for execution unit tests."""
    source_hash = hashlib.sha256(source).hexdigest()
    return _PinnedScript(
        catalog_generation,
        _ScriptRecord(
            script_name=path.name,
            source_path=os.fspath(path.resolve()),
            source_hash=source_hash,
            source_modified_at="2026-01-01T00:00:00+00:00",
            registered_at="2026-01-01T00:00:00+00:00",
            approved=approved,
            enabled=enabled,
            routines=routines,
            loaders=loaders,
            record_generation=1,
        ),
        source,
    )


def _loader_call(
    path: pathlib.Path,
    descriptor: erlab.extensions.LoaderDescriptor,
    executor: Callable[
        [_ExtensionLoaderCall, pathlib.Path, dict[str, typing.Any]],
        xr.DataArray | xr.Dataset | xr.DataTree,
    ],
    *,
    source: bytes = b"",
) -> _ExtensionLoaderCall:
    """Build one loader call with explicit publication hooks."""
    return _ExtensionLoaderCall(
        manager_session_id="manager",
        snapshot=_pinned_script(path, source, loaders=(descriptor,)),
        loader_id=descriptor.id,
        descriptor=descriptor,
        executor=executor,
        publication_checker=lambda _call: None,
        publication_recorder=lambda _call: None,
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
    assert dialog.sizeHint() == QtCore.QSize(760, 600)
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
        (("lab.py", descriptor),), parent
    )
    qtbot.addWidget(dialog)
    assert dialog.selection == ("lab.py", "calculate")
    changes: list[tuple[str, str, bool]] = []
    dialog.favorite_requested.connect(
        lambda script_name, routine_id, favorite: changes.append(
            (script_name, routine_id, favorite)
        )
    )
    dialog.favorite_button.click()
    assert changes == [("lab.py", "calculate", True)]
    assert dialog.favorite_button.property("favoriteState") is True
    dialog.favorite_button.click()
    assert changes[-1] == ("lab.py", "calculate", False)


def test_manage_dialog_preserves_selected_extension(
    qtbot: pytest.QtBot,
    tmp_path: pathlib.Path,
) -> None:
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    source_path = tmp_path / "lab.py"
    source_path.write_text("VALUE = 1\n")
    source_hash = "a" * 64
    record = _ScriptRecord(
        script_name=source_path.name,
        source_path=os.fspath(source_path),
        source_hash=source_hash,
        source_modified_at="2026-01-01T00:00:00+00:00",
        registered_at="2026-01-01T00:00:00+00:00",
        approved=True,
    )
    dialog = extension_dialogs._ManageExtensionsDialog(parent)
    qtbot.addWidget(dialog)
    assert dialog.sizeHint() == QtCore.QSize(1120, 620)
    dialog.set_catalog(_ExtensionCatalogModel(extensions={"lab.py": record}))
    top = dialog.tree.topLevelItem(0)
    assert top.childCount() == 0
    dialog.tree.setCurrentItem(top)
    assert dialog.selected_script_name == "lab.py"

    actions: list[tuple[str, str]] = []

    def action_slot(action: str, script_name: str) -> None:
        actions.append((action, script_name))

    dialog.action_requested.connect(action_slot)
    try:
        assert "metadata" not in dialog._buttons
        source_label = dialog._detail_labels["source"]
        assert source_label.property("sourcePath") == os.fspath(source_path)
        dialog.tree.setCurrentItem(None)
        assert source_label.text() == ""
        dialog._emit_action("remove")
        assert actions == []
    finally:
        dialog.action_requested.disconnect(action_slot)


def test_workspace_requirements_dialog_registers_only_recoverable_selection(
    qtbot: pytest.QtBot,
) -> None:
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    requirement = _WorkspaceScriptRequirement(
        script_name="lab.py",
        capability_id="calculate",
        capability_name="Calculate",
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
        recoverable={("lab.py", "a" * 64)},
    )
    qtbot.addWidget(dialog)
    registrations: list[tuple[str, str]] = []

    def registration_slot(script_name: str, source_hash: str) -> None:
        registrations.append((script_name, source_hash))

    dialog.register_requested.connect(registration_slot)
    try:
        dialog._register_selected()
        assert registrations == []
        dialog.tree.setCurrentItem(dialog.tree.topLevelItem(0))
        assert dialog._register_button.isEnabled()
        dialog._register_selected()
        assert registrations == [("lab.py", "a" * 64)]

        for state in ("missing", "hash-mismatch", "validation-failed"):
            dialog.set_requirements((resolved.model_copy(update={"state": state}),))
            assert dialog._register_button.isEnabled()
            dialog._register_selected()

        assert registrations == [("lab.py", "a" * 64)] * 4
        dialog.set_requirements((resolved.model_copy(update={"state": "ready"}),))
        assert dialog.tree.currentItem() is dialog.tree.topLevelItem(0)
        assert not dialog._register_button.isEnabled()
        dialog._register_selected()
        assert registrations == [("lab.py", "a" * 64)] * 4
    finally:
        dialog.register_requested.disconnect(registration_slot)


def test_missing_scripts_dialog_lists_scripts_and_emits_selected_actions(
    qtbot: pytest.QtBot,
) -> None:
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    records = tuple(
        _ScriptRecord(
            script_name=filename,
            source_path=source_path,
            source_hash=source_hash,
            source_modified_at="2026-01-01T00:00:00+00:00",
            registered_at="2026-01-01T00:00:00+00:00",
            approved=True,
            enabled=True,
        )
        for filename, source_hash, source_path in (
            ("first.py", "a" * 64, "/missing/first.py"),
            ("second.py", "b" * 64, "/missing/second.py"),
        )
    )
    dialog = extension_dialogs._MissingScriptsDialog(records, parent)
    qtbot.addWidget(dialog)
    located: list[str] = []
    dialog.locate_requested.connect(located.append)

    assert dialog.tree.topLevelItemCount() == 2
    assert dialog.tree.topLevelItem(0).childCount() == 0
    dialog.tree.setCurrentItem(dialog.tree.topLevelItem(1))
    dialog.locate_button.click()

    assert located == ["second.py"]


def test_controller_filters_loader_paths_and_rejects_duplicate_filters(
    manager_context,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = erlab.extensions.LoaderDescriptor(
        id="load_data",
        name="Lab Data",
        category="Lab",
        summary="",
        function_name="load_data",
        extensions=(".dat",),
    )
    records: dict[str, _ScriptRecord] = {}
    calls: dict[str, _ExtensionLoaderCall] = {}
    for script_name in ("first.py", "second.py"):
        source_path = tmp_path / script_name
        source_bytes = _loader_script(
            source_path, name="Lab Data", extensions=(".dat",)
        )
        snapshot = _pinned_script(
            source_path,
            source_bytes,
            loaders=(descriptor,),
        )
        records[script_name] = snapshot.record
        calls[script_name] = _loader_call(
            source_path,
            descriptor,
            executor=lambda *_args: xr.DataArray([1.0]),
            source=source_bytes,
        )

    with manager_context() as manager:
        controller = manager._extensions
        controller.catalog.model = _ExtensionCatalogModel(
            extensions={"first.py": records["first.py"]}
        )
        monkeypatch.setattr(
            controller.execution,
            "ready_loader_calls",
            lambda script_name, _source_hash: (calls[script_name],),
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

        catalog, source_hash = controller.catalog.store.register_script(script_path)
        catalog = _validate_and_enable(
            controller.catalog.store,
            "scale.py",
            expected_record_generation=catalog.extensions["scale.py"].record_generation,
        )
        catalog = controller.catalog.store.set_routine_favorite(
            "scale.py", "scale", favorite=True
        )
        controller.catalog.refresh()
        controller._recent.append(("scale.py", "scale"))
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
        assert ("scale.py", "scale") in action_data

        selected: list[tuple[str, str]] = []
        monkeypatch.setattr(
            controller,
            "run_routine",
            lambda script_name, routine_id: selected.append((script_name, routine_id)),
        )

        class AcceptedSelectionDialog(QtCore.QObject):
            favorite_requested = QtCore.Signal(str, str, bool)
            selection = ("scale.py", "scale")

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
        assert selected == [("scale.py", "scale")]

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
        controller.run_routine("scale.py", "scale")
        assert len(information_calls) == 2

        monkeypatch.setattr(manager, "_selected_imagetool_targets", lambda: (0,))
        controller.run_routine("missing.py", "scale")
        controller.run_routine("scale.py", "missing")

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
        controller.run_routine("scale.py", "scale")

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
        controller.run_routine("scale.py", "scale")
        assert critical_calls == [None]

        queued: list[dict[str, typing.Any]] = []
        monkeypatch.setattr(
            controller.execution,
            "queue_routine",
            lambda **kwargs: queued.append(kwargs) or "job",
        )
        controller.run_routine("scale.py", "scale")
        controller.run_routine("scale.py", "scale")
        assert len(queued) == 2
        assert {call["source_hash"] for call in queued} == {source_hash}
        assert tuple(controller._recent).count(("scale.py", "scale")) == 1

        assert controller.loader_by_name("missing") is None
        controller.show_manager()
        assert controller._manage_dialog.isVisible()
        controller._manage_dialog.hide()


def test_routine_dialog_rejects_a_source_reloaded_while_open(
    manager_context,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = tmp_path / "scale.py"
    _script(script_path)
    critical_calls: list[None] = []

    with manager_context() as manager:
        controller = manager._extensions
        catalog, old_source_hash = controller.catalog.store.register_script(script_path)
        _validate_and_enable(
            controller.catalog.store,
            "scale.py",
            expected_record_generation=catalog.extensions["scale.py"].record_generation,
        )
        controller.catalog.refresh()
        target = manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(
                xr.DataArray([1.0]), _in_manager=True
            ),
            show=False,
        )
        monkeypatch.setattr(
            manager,
            "_selected_imagetool_targets",
            lambda: (target,) if target in manager._tool_graph.root_wrappers else (),
        )

        class ReloadingParameterDialog:
            parameters: typing.ClassVar[dict[str, float]] = {"scale": 2.0}

            def __init__(self, *_args: typing.Any, **_kwargs: typing.Any) -> None:
                pass

            @staticmethod
            def exec() -> int:
                updated_source = _script(script_path, "data * scale + 1.0")
                current = controller.catalog.store.read().extensions["scale.py"]
                updated, changed = controller.catalog.store.reload_script(
                    "scale.py",
                    expected_source_hash=hashlib.sha256(updated_source).hexdigest(),
                    expected_record_generation=current.record_generation,
                )
                assert changed
                _validate_and_enable(
                    controller.catalog.store,
                    "scale.py",
                    expected_record_generation=updated.extensions[
                        "scale.py"
                    ].record_generation,
                )
                controller.catalog.refresh()
                return int(QtWidgets.QDialog.DialogCode.Accepted)

        monkeypatch.setattr(
            extension_controller,
            "_ExtensionParameterDialog",
            ReloadingParameterDialog,
        )
        monkeypatch.setattr(
            erlab.interactive.utils.MessageDialog,
            "critical",
            lambda *_args, **_kwargs: critical_calls.append(None),
        )

        controller.run_routine("scale.py", "scale")

        assert (
            controller.catalog.model.extensions["scale.py"].source_hash
            != old_source_hash
        )
        assert controller.execution.active is None
        assert controller.execution.queued == ()
        assert critical_calls == [None]


def test_controller_replay_loader_rejects_unavailable_calls(
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
        missing_call = FileReplayCall(
            kind="extension_loader",
            target="missing.py",
            source_hash="a" * 64,
            capability_id="load_data",
            selection=FileDataSelection(kind="dataarray"),
        )
        controller.catalog.load_error = "catalog unavailable"
        with pytest.raises(
            erlab.extensions.ExtensionExecutionError,
            match="catalog is unavailable",
        ):
            controller.replay_loader(load_source(missing_call))
        controller.catalog.load_error = None
        with pytest.raises(
            erlab.extensions.ExtensionExecutionError,
            match="not available",
        ):
            controller.replay_loader(load_source(missing_call))

        script_path = tmp_path / "loader.py"
        _loader_script(script_path, name="Lab Data", extensions=(".dat",))
        catalog, source_hash = controller.catalog.store.register_script(script_path)
        catalog = _validate_and_enable(
            controller.catalog.store,
            "loader.py",
            expected_record_generation=catalog.extensions[
                "loader.py"
            ].record_generation,
        )
        record = catalog.extensions["loader.py"]
        script_call = FileReplayCall(
            kind="extension_loader",
            target="loader.py",
            source_hash=source_hash,
            capability_id="load_data",
            selection=FileDataSelection(kind="dataarray"),
        )
        controller.execution._set_validation_error(
            "loader.py", source_hash, "missing dependency"
        )
        with pytest.raises(
            erlab.extensions.ExtensionExecutionError, match="validation-failed"
        ):
            controller.replay_loader(load_source(script_call))
        controller.execution._set_validation_error("loader.py", source_hash, None)

        missing_descriptor_record = record.model_copy(update={"loaders": ()})
        monkeypatch.setattr(
            controller.catalog.store,
            "resolve_script",
            lambda *_args, **_kwargs: _PinnedScript(
                catalog.generation,
                missing_descriptor_record,
                script_path.read_bytes(),
            ),
        )
        with pytest.raises(
            erlab.extensions.ExtensionExecutionError, match="is not available"
        ):
            controller.replay_loader(load_source(script_call))
        monkeypatch.undo()

        xr.testing.assert_identical(
            controller.replay_loader(load_source(script_call)),
            xr.DataArray([1.0]),
        )


def test_controller_capability_status_uses_application_catalog(
    manager_context,
) -> None:
    with manager_context() as manager:
        controller = manager._extensions
        assert (
            controller.capability_status("missing.py", "a" * 64, "routine", "calculate")
            == "missing-source"
        )


def test_bound_loader_stages_source_until_replay_publication(
    manager_context,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = tmp_path / "loader.py"
    _loader_script(script_path, name="Lab Data", extensions=(".dat",))
    data_path = tmp_path / "data.dat"
    data_path.write_text("4")

    with manager_context() as manager:
        execution = manager._extensions.execution
        catalog, source_hash = manager._extensions.catalog.store.register_script(
            script_path
        )
        _validate_and_enable(
            manager._extensions.catalog.store,
            script_path.name,
            expected_record_generation=catalog.extensions[
                _script_name_key(script_path.name)
            ].record_generation,
        )
        call = execution.loader_call(script_path.name, source_hash, "load_data")
        recorded: list[tuple[str, str, bytes]] = []
        monkeypatch.setattr(
            manager._workspace_state.extension_scripts,
            "remember_verified_source",
            lambda script_name, snapshot_hash, source: recorded.append(
                (script_name, snapshot_hash, source)
            ),
        )

        with execution.capture_replay_sources() as publication:
            result = call(data_path)
            assert recorded == []
            publication.require_current_for_publication()
            publication.publish()

        xr.testing.assert_identical(result, xr.DataArray([4.0]))
        assert recorded == [
            (call.script_name, call.source_hash, call.snapshot.source_bytes)
        ]


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
    located: list[str] = []

    with manager_context() as manager:
        controller = manager._extensions
        catalog, _source_hash = controller.catalog.store.register_script(script_path)
        catalog = _validate_and_enable(
            controller.catalog.store,
            "scale.py",
            expected_record_generation=catalog.extensions["scale.py"].record_generation,
        )
        controller.catalog.refresh()
        enabled_record = controller.catalog.model.extensions["scale.py"]

        def update_script(
            script_name: str,
            *,
            expected_record_generation: int,
            **values: typing.Any,
        ) -> _ExtensionCatalogModel:
            updates.append(
                {
                    "script_name": script_name,
                    "expected_record_generation": expected_record_generation,
                    **values,
                }
            )
            return controller.catalog.model

        monkeypatch.setattr(controller.catalog.store, "update_script", update_script)
        monkeypatch.setattr(
            controller.execution,
            "validate_script",
            lambda script_name, _source_hash, *, expected_record_generation: (
                validations.append((script_name, expected_record_generation))
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
            "_locate_missing_script",
            lambda script_name: located.append(script_name) or True,
        )

        controller._manage_action("toggle", "scale.py")
        assert updates[0]["enabled"] is False

        disabled_record = enabled_record.model_copy(update={"enabled": False})
        controller.catalog.model = catalog.model_copy(
            update={"extensions": {"scale.py": disabled_record}}
        )
        controller._manage_action("toggle", "scale.py")
        assert validations == [("scale.py", disabled_record.record_generation)]

        before = len(updates)
        controller._manage_action("embedding:invalid", "scale.py")
        assert len(updates) == before
        controller._manage_action("embedding:always", "scale.py")
        assert updates[-1]["embed_policy"] == "always"

        script_path.unlink()
        controller.catalog.model = catalog
        controller._manage_action("reload", "scale.py")
        assert located == ["scale.py"]

        controller.catalog.model = catalog
        monkeypatch.setattr(
            controller.catalog.store,
            "update_script",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                _ExtensionCatalogConflictError("changed")
            ),
        )
        controller._manage_action("toggle", "scale.py")
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
        _manager_loader_name_for_entry=lambda _name_filter, func: (
            manager_base._builtin_loader_name_for_callable(func)
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
    call = _loader_call(
        pathlib.Path("lab.py"),
        descriptor,
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
    manager._manager_loader_name_for_entry = types.MethodType(
        manager_base._ImageToolManagerBase._manager_loader_name_for_entry,
        manager,
    )

    shared_kwargs, shared_extensions = (
        manager_base._ImageToolManagerBase._shared_loader_state(manager)
    )

    assert shared_kwargs == {"lab.py:load_data": {"scale": 4.0}}
    assert shared_extensions == {}

    manager_base._ImageToolManagerBase._sync_shared_loader_state(
        manager,
        {"lab.py:load_data": {"scale": 7.0}},
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

    catalog, source_hash = store.register_script(script_path)
    assert catalog.schema_version == 1
    assert json.loads(store.path.read_text(encoding="utf-8"))["schema_version"] == 1
    assert source_hash == hashlib.sha256(source).hexdigest()
    record = catalog.extensions["scale.py"]
    reloaded, changed = store.reload_script(
        record.script_name,
        expected_source_hash=source_hash,
        expected_record_generation=record.record_generation,
    )
    assert not changed
    assert reloaded == catalog
    assert reloaded.extensions["scale.py"].source_hash == source_hash


def test_catalog_uses_override_directory(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "extension-catalog"
    monkeypatch.setenv("ERLAB_EXTENSION_CATALOG", os.fspath(directory))

    assert extension_catalog._default_catalog_directory() == directory.resolve()


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
def test_catalog_reports_atomic_write_failures(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
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

    with pytest.raises(extension_catalog._ExtensionCatalogError, match=failure):
        store._write_unlocked(_ExtensionCatalogModel())

    assert canceled == ([None] if failure == "write" else [])


def test_catalog_source_lookup_and_integrity_failures(tmp_path: pathlib.Path) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    script_path = tmp_path / "scale.py"
    source = _script(script_path)
    catalog, source_hash = store.register_script(script_path)

    snapshot = store.resolve_script("SCALE.PY", source_hash)
    assert snapshot.catalog_generation == catalog.generation
    assert snapshot.registered_path == script_path.resolve()
    assert snapshot.source_bytes == source
    with pytest.raises(_ExtensionCatalogConflictError, match="different contents"):
        store.resolve_script("scale.py", "0" * 64)
    with pytest.raises(KeyError):
        store.resolve_script("missing.py", source_hash)

    script_path.unlink()
    with pytest.raises(FileNotFoundError):
        store.resolve_script("scale.py", source_hash)
    script_path.write_bytes(b"corrupt")
    with pytest.raises(_ExtensionCatalogConflictError, match="changed on disk"):
        store.resolve_script("scale.py", source_hash)

    assert store.read() == catalog


def test_extension_models_reject_invalid_hash_and_unapproved_enablement() -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        _ScriptRecord(
            script_name="source.py",
            source_path="/source.py",
            source_hash="A" * 64,
            source_modified_at="2026-01-01T00:00:00+00:00",
            registered_at="2026-01-01T00:00:00+00:00",
        )
    with pytest.raises(ValueError, match="must be approved"):
        _ScriptRecord(
            script_name="source.py",
            source_path="/source.py",
            source_hash="a" * 64,
            source_modified_at="2026-01-01T00:00:00+00:00",
            registered_at="2026-01-01T00:00:00+00:00",
            enabled=True,
        )


def test_catalog_rejects_validation_for_replaced_source(
    tmp_path: pathlib.Path,
) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    script_path = tmp_path / "scale.py"
    _script(script_path)
    catalog, old_source_hash = store.register_script(script_path)
    old_record = catalog.extensions["scale.py"]
    _script(script_path, "data + scale")
    new_source_hash = hashlib.sha256(script_path.read_bytes()).hexdigest()
    catalog, changed = store.reload_script(
        old_record.script_name,
        expected_source_hash=new_source_hash,
        expected_record_generation=old_record.record_generation,
    )
    assert changed

    with pytest.raises(
        _ExtensionCatalogConflictError, match="changed during validation"
    ):
        store.commit_script_validation(
            "scale.py",
            source_hash=old_source_hash,
            expected_record_generation=catalog.extensions["scale.py"].record_generation,
            routines=(),
            loaders=(),
            enable_script=False,
        )
    assert store.read().extensions["scale.py"].source_hash == new_source_hash


def test_catalog_reports_exact_script_capability_states(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    script_path = tmp_path / "scale.py"
    _script(script_path)
    catalog, source_hash = store.register_script(script_path)
    catalog = _validate_and_enable(
        store,
        "scale.py",
        expected_record_generation=catalog.extensions["scale.py"].record_generation,
    )
    ready_record = catalog.extensions["scale.py"]
    source = script_path.read_bytes()

    def status_for(
        record: _ScriptRecord | None = None, capability_id: str = "scale"
    ) -> str:
        if record is not None:
            monkeypatch.setattr(
                store,
                "resolve_script",
                lambda *_args: _PinnedScript(catalog.generation, record, source),
            )
        try:
            store.resolve_registered_capability(
                "scale.py",
                "routine",
                capability_id,
                source_hash=source_hash,
            )
        except extension_api._RegisteredScriptUnavailable as error:
            return error.status
        return "ready"

    assert status_for(ready_record) == "ready"
    assert status_for(ready_record.model_copy(update={"enabled": False})) == "disabled"

    assert (
        status_for(
            ready_record.model_copy(
                update={
                    "enabled": False,
                    "approved": False,
                }
            )
        )
        == "approval-required"
    )
    assert status_for(ready_record, "missing") == "missing-capability"

    unsupported_descriptor = ready_record.routines[0].model_copy(
        update={"extension_api_version": 2}
    )
    assert (
        status_for(
            ready_record.model_copy(
                update={
                    "routines": (unsupported_descriptor,),
                }
            )
        )
        == "unsupported-api"
    )

    monkeypatch.undo()
    script_path.write_bytes(b"corrupt")
    assert status_for() == "hash-mismatch"
    script_path.unlink()
    assert status_for() == "missing-source"


def test_execution_status_uses_the_catalog_snapshot_returned_by_pin(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    script_path = tmp_path / "scale.py"
    source = _script(script_path)
    catalog, source_hash = store.register_script(script_path)
    catalog = _validate_and_enable(
        store,
        "scale.py",
        expected_record_generation=catalog.extensions["scale.py"].record_generation,
    )
    record = catalog.extensions["scale.py"]
    disabled_record = record.model_copy(update={"enabled": False})
    monkeypatch.setattr(
        store,
        "resolve_script",
        lambda *_args: _PinnedScript(catalog.generation, disabled_record, source),
    )

    resolution = extension_execution._resolve_execution_capability(
        store,
        "scale.py",
        source_hash,
        "routine",
        "scale",
        source_is_healthy=lambda *_args: True,
    )
    assert resolution.status == "disabled"
    assert resolution.snapshot is not None
    assert not resolution.snapshot.record.enabled
    with pytest.raises(extension_api._RegisteredScriptUnavailable, match="disabled"):
        store.resolve_registered_capability(
            "scale.py", "routine", "scale", source_hash=source_hash
        )


def test_pinned_script_rejects_unknown_and_missing_local_sources(
    tmp_path: pathlib.Path,
) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    script_path = tmp_path / "scale.py"
    _script(script_path)
    catalog, source_hash = store.register_script(script_path)
    catalog = _validate_and_enable(
        store,
        "scale.py",
        expected_record_generation=catalog.extensions["scale.py"].record_generation,
    )

    with pytest.raises(_ExtensionCatalogConflictError):
        store.resolve_script("scale.py", "0" * 64)
    script_path.unlink()
    with pytest.raises(FileNotFoundError):
        store.resolve_script("scale.py", source_hash)


def test_catalog_registered_capability_rejects_unusable_state(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    script_path = tmp_path / "scale.py"
    _script(script_path)
    catalog, source_hash = store.register_script(script_path)

    with pytest.raises(
        extension_api._RegisteredScriptUnavailable, match="approval-required"
    ):
        store.resolve_registered_capability(
            "scale.py", "routine", "scale", source_hash=source_hash
        )
    unapproved = catalog.extensions["scale.py"]
    monkeypatch.setattr(
        store,
        "resolve_script",
        lambda *_args: _PinnedScript(
            catalog.generation, unapproved, script_path.read_bytes()
        ),
    )
    with pytest.raises(
        extension_api._RegisteredScriptUnavailable, match="approval-required"
    ):
        store.resolve_registered_capability(
            "scale.py", "routine", "scale", source_hash=source_hash
        )
    monkeypatch.undo()

    catalog = _validate_and_enable(
        store,
        "scale.py",
        expected_record_generation=catalog.extensions["scale.py"].record_generation,
    )
    catalog = store.update_script(
        "scale.py",
        expected_record_generation=catalog.extensions["scale.py"].record_generation,
        enabled=False,
    )
    with pytest.raises(extension_api._RegisteredScriptUnavailable, match="disabled"):
        store.resolve_registered_capability(
            "scale.py", "routine", "scale", source_hash=source_hash
        )
    catalog = store.update_script(
        "scale.py",
        expected_record_generation=catalog.extensions["scale.py"].record_generation,
        enabled=True,
    )
    with pytest.raises(
        extension_api._RegisteredScriptUnavailable, match="missing-capability"
    ):
        store.resolve_registered_capability(
            "scale.py", "routine", "missing", source_hash=source_hash
        )


@pytest.mark.parametrize("source_state", ["missing", "changed", "catalog-error"])
def test_catalog_resolvers_normalize_unavailable_sources_for_public_calls(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    source_state: str,
) -> None:
    catalog = _ExtensionCatalog(directory=tmp_path / "catalog")
    script_path = tmp_path / "scale.py"
    _script(script_path)
    try:
        model, source_hash = catalog.store.register_script(script_path)
        _validate_and_enable(
            catalog.store,
            "scale.py",
            expected_record_generation=model.extensions["scale.py"].record_generation,
        )
        if source_state == "missing":
            script_path.unlink()
        elif source_state == "changed":
            _script(script_path, "data + scale")
        else:
            monkeypatch.setattr(
                catalog.store,
                "read",
                lambda: (_ for _ in ()).throw(
                    extension_catalog._ExtensionCatalogError("unreadable catalog")
                ),
            )

        with pytest.raises(erlab.extensions.ExtensionNotFoundError):
            erlab.extensions.run_routine(
                xr.DataArray([1.0]),
                registered_script="scale.py",
                source_hash=source_hash,
                routine_id="scale",
            )
        with pytest.raises(erlab.extensions.ExtensionNotFoundError):
            extension_api._resolve_registered_script_capability(
                "scale.py", "routine", "scale"
            )
    finally:
        catalog.close()


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
    call = _loader_call(source, descriptor, lambda *_args: xr.DataArray([1.0]))
    assert call.manager_loader_name == "loader.py:load_data"
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
    assert extension_execution._xarray_log_fields(array)["type"] == "DataArray"

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
    call = _loader_call(tmp_path / "loader.py", descriptor, execute)
    adapter = extension_execution._DecoratedLoaderAdapter(call)
    path = tmp_path / "value.dat"
    path.write_text("3")

    assert adapter.script_name == "loader.py"
    assert adapter.source_hash == hashlib.sha256(b"").hexdigest()
    assert adapter.loader_id == "load_data"
    assert adapter.registered_path == (tmp_path / "loader.py").resolve()
    assert adapter.descriptor == descriptor
    assert tuple(adapter.file_dialog_methods) == ("Load Data (*.dat)",)
    loaded = adapter.load(path)
    assert loaded.item() == 3.0
    assert loaded.attrs["data_loader_name"] == "loader.py:load_data"
    xr.testing.assert_identical(
        adapter.load_single(path, scale=2.0), xr.DataArray([3.0])
    )
    assert calls == [(path, {}), (path, {"scale": 2.0})]
    with pytest.raises(ValueError, match="must be finite"):
        call(path, scale=float("inf"))


def test_decorated_loader_adapter_preserves_manager_parameter_names(
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
    )
    call = _loader_call(tmp_path / "loader.py", descriptor, execute)
    adapter = extension_execution._DecoratedLoaderAdapter(call)
    path = tmp_path / "value.unknown"
    path.write_text("3")

    result = adapter.load_for_manager(
        path,
        single=False,
        chunks=2,
        metadata="lab metadata",
    )

    xr.testing.assert_equal(result, xr.DataArray([3.0]))
    assert calls == [
        (
            path,
            {"single": False, "chunks": 2, "metadata": "lab metadata"},
        )
    ]


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
    call = _loader_call(
        tmp_path / "loader.py", descriptor, lambda *_args: xr.DataArray([1.0])
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
        "loader.py",
        call.source_hash,
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
    call = _loader_call(
        tmp_path / "loader.py", descriptor, lambda *_args: xr.DataArray([1.0])
    )
    snapshot = call.snapshot
    store = types.SimpleNamespace(
        resolve_script=lambda *_args: snapshot,
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
            execution.validate_script(
                "missing.py", "a" * 64, expected_record_generation=0
            )

        catalog, source_hash = manager._extensions.catalog.store.register_script(
            script_path
        )
        record = catalog.extensions["scale.py"]
        with pytest.raises(_ExtensionCatalogConflictError, match="before validation"):
            execution.validate_script(
                record.script_name,
                source_hash,
                expected_record_generation=record.record_generation + 1,
            )

        monkeypatch.setattr(
            execution, "_run_blocking_task", lambda *_args, **_kwargs: None
        )
        with pytest.raises(
            erlab.extensions.ExtensionExecutionError,
            match="validation returned no result",
        ):
            execution.validate_script(
                record.script_name,
                source_hash,
                expected_record_generation=record.record_generation,
            )

        descriptor = erlab.extensions.LoaderDescriptor(
            id="load_data",
            name="Load Data",
            category="Lab",
            summary="",
            function_name="load_data",
        )
        call = _loader_call(
            tmp_path / "loader.py",
            descriptor,
            lambda *_args: xr.DataArray([1.0]),
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
                script_name=record.script_name,
                source_hash=source_hash,
                routine_id="scale",
                parameters={},
                target=0,
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
    call = _loader_call(
        tmp_path / "loader.py",
        descriptor,
        lambda *_args: xr.DataArray([1.0]),
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


def test_blocking_extension_task_waits_for_worker_event(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caller_thread = threading.get_ident()
    worker_threads: list[int] = []
    wait_messages: list[str] = []

    class _Task(QtCore.QRunnable):
        def __init__(self) -> None:
            super().__init__()
            self.done = threading.Event()
            self.error: Exception | None = None

        def run(self) -> None:
            worker_threads.append(threading.get_ident())
            self.done.set()

    with manager_context() as manager:
        task = _Task()

        class _WaitContext:
            def __enter__(self) -> None:
                return None

            def __exit__(self, *_args: object) -> None:
                return None

        def record_wait_dialog(
            _parent: QtWidgets.QWidget, message: str
        ) -> _WaitContext:
            wait_messages.append(message)
            return _WaitContext()

        with monkeypatch.context() as patch_context:
            patch_context.setattr(
                extension_execution.erlab.interactive.utils,
                "wait_dialog",
                record_wait_dialog,
            )
            patch_context.setattr(
                extension_execution.QtCore,
                "QEventLoop",
                pytest.fail,
            )
            manager._extensions.execution._run_blocking_task(
                typing.cast(
                    "_ExtensionLoaderWorker | _ExtensionValidationWorker",
                    task,
                ),
                wait_message="Waiting for worker",
            )

        assert wait_messages == ["Waiting for worker"]
        assert len(worker_threads) == 1
        assert worker_threads[0] != caller_thread
        assert task.done.is_set()
        assert task not in manager._extensions.execution._blocking_tasks


def test_blocking_extension_task_does_not_dispatch_gui_events(manager_context) -> None:
    gui_callbacks: list[None] = []

    class _Task(QtCore.QRunnable):
        def __init__(self) -> None:
            super().__init__()
            self.done = threading.Event()
            self.error: Exception | None = None

        def run(self) -> None:
            threading.Event().wait(0.05)
            self.done.set()

    with manager_context() as manager:
        QtCore.QTimer.singleShot(0, lambda: gui_callbacks.append(None))
        manager._extensions.execution._run_blocking_task(
            typing.cast(
                "_ExtensionLoaderWorker | _ExtensionValidationWorker",
                _Task(),
            )
        )

        assert gui_callbacks == []
        QtCore.QCoreApplication.processEvents()
        assert gui_callbacks == [None]


def test_routine_job_rejects_unavailable_catalog_state(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "scale.py"
    _script(script_path)
    data = xr.DataArray([1.0])

    with manager_context() as manager:
        execution = manager._extensions.execution
        with pytest.raises(erlab.extensions.ExtensionExecutionError, match="available"):
            execution._routine_job(
                script_name="missing.py",
                source_hash="a" * 64,
                routine_id="scale",
                parameters={},
                input_data=data,
                input_uid="uid",
                input_snapshot="snapshot",
            )

        catalog, source_hash = manager._extensions.catalog.store.register_script(
            script_path
        )
        with pytest.raises(erlab.extensions.ExtensionExecutionError, match="available"):
            execution._routine_job(
                script_name="scale.py",
                source_hash=source_hash,
                routine_id="scale",
                parameters={},
                input_data=data,
                input_uid="uid",
                input_snapshot="snapshot",
            )
        catalog = _validate_and_enable(
            manager._extensions.catalog.store,
            "scale.py",
            expected_record_generation=catalog.extensions["scale.py"].record_generation,
        )
        with pytest.raises(
            erlab.extensions.ExtensionExecutionError, match="hash-mismatch"
        ):
            execution._routine_job(
                script_name="scale.py",
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
                script_name="scale.py",
                source_hash=catalog.extensions["scale.py"].source_hash,
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
        catalog, source_hash = manager._extensions.catalog.store.register_script(
            script_path
        )
        catalog = _validate_and_enable(
            manager._extensions.catalog.store,
            "scale.py",
            expected_record_generation=catalog.extensions["scale.py"].record_generation,
        )
        job = execution._routine_job(
            script_name="scale.py",
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
        catalog, source_hash = manager._extensions.catalog.store.register_script(
            script_path
        )
        catalog = _validate_and_enable(
            manager._extensions.catalog.store,
            "scale.py",
            expected_record_generation=catalog.extensions["scale.py"].record_generation,
        )
        operation = ExtensionRoutineOperation(
            script_name="scale.py",
            source_hash=source_hash,
            routine_id="scale",
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

        queued_job = execution._routine_job(
            script_name="scale.py",
            source_hash=source_hash,
            routine_id="scale",
            parameters={"scale": 2.0},
            input_data=data,
            input_uid="queued",
            input_snapshot="snapshot",
        )
        execution._pending.append(queued_job)
        with pytest.raises(
            erlab.extensions.ExtensionExecutionError,
            match="background extension routines to finish",
        ):
            execution.run_operation(operation, data)
        execution._pending.clear()

        active_worker = execution._routine_worker(queued_job)
        execution._active = (queued_job, active_worker)
        with pytest.raises(
            erlab.extensions.ExtensionExecutionError,
            match="background extension routines to finish",
        ):
            execution.run_operation(operation, data)
        execution._active = None

        class ImmediatePool:
            def __init__(self) -> None:
                self.status: typing.Literal["success", "failed", "discarded"] | None = (
                    None
                )

            def start(self, worker: _ExtensionRoutineWorker) -> None:
                if self.status is not None:
                    worker.result = extension_execution._ExtensionRoutineResult(
                        job=worker.job,
                        output=(
                            xr.DataArray([2.0]) if self.status == "success" else None
                        ),
                        duration=0.0,
                        status=self.status,
                    )
                worker.done.set()

        immediate_pool = ImmediatePool()
        original_pool = execution._pool

        try:
            execution._pool = typing.cast("QtCore.QThreadPool", immediate_pool)
            with monkeypatch.context() as patch_context:
                patch_context.setattr(
                    extension_execution.QtCore, "QEventLoop", pytest.fail
                )
                with pytest.raises(
                    erlab.extensions.ExtensionExecutionError,
                    match="without a result",
                ):
                    execution.run_operation(operation, data)

                immediate_pool.status = "discarded"
                with pytest.raises(
                    erlab.extensions.ExtensionExecutionError, match="not enabled"
                ):
                    execution.run_operation(operation, data)

                immediate_pool.status = "failed"
                with pytest.raises(
                    erlab.extensions.ExtensionExecutionError,
                    match="could not complete",
                ):
                    execution.run_operation(operation, data)

                immediate_pool.status = "success"
                xr.testing.assert_identical(
                    execution.run_operation(operation, data), xr.DataArray([2.0])
                )
        finally:
            execution._pool = original_pool


def test_extension_routine_provenance_parameters_are_editable(
    manager_context,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = tmp_path / "scale.py"
    _script(script_path)
    source_data = xr.DataArray([1.0, 2.0], dims=("x",))

    with manager_context() as manager:
        catalog, source_hash = manager._extensions.catalog.store.register_script(
            script_path
        )
        _validate_and_enable(
            manager._extensions.catalog.store,
            "scale.py",
            expected_record_generation=(
                catalog.extensions["scale.py"].record_generation
            ),
        )
        manager._extensions.catalog.refresh()
        operation = ExtensionRoutineOperation(
            script_name="scale.py",
            source_hash=source_hash,
            routine_id="scale",
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
    ["source-hash", "relative-source-path"],
)
def test_catalog_rejects_inconsistent_persisted_identity(
    tmp_path: pathlib.Path,
    corruption: str,
) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    script_path = tmp_path / "scale.py"
    _script(script_path)
    catalog, _source_hash = store.register_script(script_path)
    payload = catalog.model_dump(mode="json")
    record = payload["extensions"]["scale.py"]
    if corruption == "source-hash":
        record["source_hash"] = "A" * 64
    else:
        record["source_path"] = "scale.py"
    store.path.write_text(json.dumps(payload))

    with pytest.raises(extension_catalog._ExtensionCatalogError):
        store.read()


def test_catalog_validates_callback_output_before_commit(
    tmp_path: pathlib.Path,
) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    script_path = tmp_path / "scale.py"
    _script(script_path)
    catalog, _source_hash = store.register_script(script_path)

    def corrupt(model):
        records = dict(model.extensions)
        records["scale.py"] = records["scale.py"].model_copy(
            update={"script_name": "different.py"}
        )
        return model.model_copy(update={"extensions": records})

    with pytest.raises(ValueError, match="path basename"):
        store.mutate(
            "scale.py",
            corrupt,
            expected_record_generation=catalog.extensions["scale.py"].record_generation,
        )

    assert store.read() == catalog


def test_catalog_changed_reload_requires_approval(tmp_path: pathlib.Path) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    script_path = tmp_path / "scale.py"
    _script(script_path)
    catalog, old_source_hash = store.register_script(script_path)
    record = catalog.extensions["scale.py"]
    catalog = _validate_and_enable(
        store, "scale.py", expected_record_generation=record.record_generation
    )
    assert catalog.extensions["scale.py"].enabled

    _script(script_path, "data + scale")
    new_source_hash = hashlib.sha256(script_path.read_bytes()).hexdigest()
    catalog, changed = store.reload_script(
        "scale.py",
        expected_source_hash=new_source_hash,
        expected_record_generation=catalog.extensions["scale.py"].record_generation,
    )
    assert changed
    assert new_source_hash != old_source_hash
    assert not catalog.extensions["scale.py"].enabled
    assert not catalog.extensions["scale.py"].approved
    assert catalog.extensions["scale.py"].routines == ()

    catalog = _validate_and_enable(
        store,
        "scale.py",
        expected_record_generation=catalog.extensions["scale.py"].record_generation,
    )
    catalog, changed = store.reload_script(
        "scale.py",
        expected_source_hash=new_source_hash,
        expected_record_generation=catalog.extensions["scale.py"].record_generation,
    )
    assert not changed
    assert catalog.extensions["scale.py"].enabled


def test_stale_validation_does_not_import_a_newer_source(
    tmp_path: pathlib.Path,
) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    script_path = tmp_path / "scale.py"
    marker_path = tmp_path / "imported-new-source"
    _script(script_path)
    catalog, reviewed_source_hash = store.register_script(script_path)
    reviewed_record = catalog.extensions["scale.py"]
    reviewed_snapshot = store.resolve_script("scale.py", reviewed_source_hash)
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
    newer_source_hash = hashlib.sha256(script_path.read_bytes()).hexdigest()
    catalog, changed = store.reload_script(
        "scale.py",
        expected_source_hash=newer_source_hash,
        expected_record_generation=reviewed_record.record_generation,
    )
    assert changed

    with pytest.raises(_ExtensionCatalogConflictError, match="another manager"):
        _validate_script_snapshot(
            store,
            reviewed_snapshot,
            expected_record_generation=reviewed_record.record_generation,
            manager_session_id="manager",
            script_modules={},
        )

    assert not marker_path.exists()
    current = catalog.extensions["scale.py"]
    assert current.source_hash == newer_source_hash
    assert not current.approved


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
        assert not manager._extensions._review_and_register(script_path)
        assert "reviewed.py" not in manager._extensions.catalog.store.read().extensions

    assert shown == [None]


def test_catalog_rejects_a_case_insensitive_filename_collision(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    first_path = tmp_path / "first" / "Analysis.py"
    second_path = tmp_path / "second" / "analysis.PY"
    first_path.parent.mkdir()
    second_path.parent.mkdir()
    _script(first_path)
    _script(second_path, "data + scale")
    warnings: list[str] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda _parent, _title, text: warnings.append(text),
    )
    with manager_context() as manager:
        manager._extensions.catalog.store.register_script(first_path)
        manager._extensions.catalog.refresh()
        monkeypatch.setattr(
            extension_controller._SourceReviewDialog,
            "exec",
            lambda _dialog: 1,
        )
        assert not manager._extensions._review_and_register(second_path)
        after = manager._extensions.catalog.store.read()

    assert tuple(after.extensions) == ("analysis.py",)
    assert pathlib.Path(after.extensions["analysis.py"].source_path) == first_path
    assert len(warnings) == 1


def test_script_record_requires_the_exact_registered_filename(
    tmp_path: pathlib.Path,
) -> None:
    source_path = (tmp_path / "lab.py").resolve()

    with pytest.raises(ValueError, match="basename must match"):
        _ScriptRecord(
            script_name="Lab.py",
            source_path=os.fspath(source_path),
            source_hash="a" * 64,
            source_modified_at="2026-01-01T00:00:00+00:00",
            registered_at="2026-01-01T00:00:00+00:00",
        )


def test_unchanged_add_script_enables_current_source(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "analysis.py"
    _script(script_path)
    with manager_context() as manager:
        before, source_hash = manager._extensions.catalog.store.register_script(
            script_path
        )
        manager._extensions.catalog.refresh()
        monkeypatch.setattr(
            extension_controller._SourceReviewDialog,
            "exec",
            lambda _dialog: 1,
        )

        assert manager._extensions._review_and_register(script_path)
        after = manager._extensions.catalog.store.read()

    record = after.extensions["analysis.py"]
    assert record.source_hash == source_hash
    assert record.enabled
    assert record.approved
    assert (
        record.record_generation
        == before.extensions["analysis.py"].record_generation + 1
    )


def test_catalog_reload_rejects_a_stale_same_extension_edit(
    tmp_path: pathlib.Path,
) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    script_path = tmp_path / "scale.py"
    _script(script_path)
    catalog, _source_hash = store.register_script(script_path)
    stale_generation = catalog.extensions["scale.py"].record_generation
    store.update_script(
        "scale.py",
        expected_record_generation=stale_generation,
        embed_policy="always",
    )
    _script(script_path, "data + scale")
    reviewed_hash = hashlib.sha256(script_path.read_bytes()).hexdigest()

    with pytest.raises(_ExtensionCatalogConflictError, match="another manager"):
        store.reload_script(
            "scale.py",
            expected_source_hash=reviewed_hash,
            expected_record_generation=stale_generation,
        )


def test_catalog_rejects_unsupported_schema_without_rewriting(
    tmp_path: pathlib.Path,
) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    payload = {
        "schema_version": 4,
        "generation": 0,
        "routine_favorites": [],
        "extensions": {},
    }
    original = json.dumps(payload).encode()
    store.directory.mkdir(parents=True)
    store.path.write_bytes(original)

    with pytest.raises(extension_catalog._ExtensionCatalogError):
        store.read()

    assert store.path.read_bytes() == original


def test_catalog_rejects_malformed_schema_one_without_dropping_records(
    tmp_path: pathlib.Path,
) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    script_path = tmp_path / "gaussian_tools.py"
    _script(script_path)
    catalog, _source_hash = store.register_script(script_path)
    payload = catalog.model_dump(mode="json")
    del payload["extensions"]["gaussian_tools.py"]["source_path"]
    original = json.dumps(payload).encode()
    store.path.write_bytes(original)

    with pytest.raises(extension_catalog._ExtensionCatalogError):
        store.read()

    assert store.path.read_bytes() == original


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


def test_manager_disables_extensions_until_an_unreadable_catalog_recovers(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "catalog_probe.py"
    script_path.write_text(
        """from pathlib import Path
import xarray as xr
from erlab.extensions import loader, routine

@routine()
def scale(data: xr.DataArray) -> xr.DataArray:
    return data * 2.0

@loader(extensions=(".catalogprobe",))
def load_data(path: Path) -> xr.DataArray:
    return xr.DataArray([float(path.read_text())])
"""
    )
    errors: list[str] = []
    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        lambda _parent, title, *_args, **_kwargs: errors.append(title),
    )

    with manager_context() as manager:
        catalog, source_hash = manager._extensions.catalog.store.register_script(
            script_path
        )
        _validate_and_enable(
            manager._extensions.catalog.store,
            "catalog_probe.py",
            expected_record_generation=(
                catalog.extensions["catalog_probe.py"].record_generation
            ),
        )
        manager._extensions.catalog.refresh()
        valid_payload = manager._extensions.catalog.model.model_dump_json()
        assert manager._extensions._enabled_routines()
        assert manager._extensions.file_loaders()
        assert manager._extensions.explorer_loaders

        manager._extensions.catalog.store.path.write_text("{", encoding="utf-8")
        manager._extensions.catalog.refresh()

        assert manager._extensions.catalog.load_error is not None
        assert manager._extensions._enabled_routines() == ()
        assert manager._extensions.file_loaders() == {}
        assert manager._extensions.explorer_loaders == {}
        assert (
            manager._extensions.capability_status(
                "catalog_probe.py", source_hash, "routine", "scale"
            )
            == "missing-source"
        )
        assert manager._extensions._manage_dialog.tree.topLevelItemCount() == 0
        assert errors == ["Extension Catalog Unavailable"]

        manager._extensions.catalog.store.path.write_text(
            valid_payload, encoding="utf-8"
        )
        manager._extensions.catalog.refresh()

        assert manager._extensions.catalog.load_error is None
        assert manager._extensions._enabled_routines()
        assert manager._extensions.file_loaders()
        assert manager._extensions.explorer_loaders

        manager._extensions.catalog.store.path.write_text("{", encoding="utf-8")
        manager._extensions.catalog.refresh()

        assert manager._extensions.catalog.load_error is not None
        assert manager._extensions._enabled_routines() == ()
        assert manager._extensions.file_loaders() == {}
        assert manager._extensions.explorer_loaders == {}
        assert manager._extensions._manage_dialog.tree.topLevelItemCount() == 0
        assert errors == ["Extension Catalog Unavailable"]


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


def test_unchanged_reload_updates_script_source_location(
    tmp_path: pathlib.Path,
) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    original_path = tmp_path / "original" / "scale.py"
    original_path.parent.mkdir()
    source = _script(original_path)
    catalog, source_hash = store.register_script(original_path)
    initial_generation = catalog.extensions["scale.py"].record_generation
    relocated_path = tmp_path / "relocated" / "scale.py"
    relocated_path.parent.mkdir()
    relocated_path.write_bytes(source)

    reloaded = store.relocate_script(
        "scale.py",
        relocated_path,
        expected_record_generation=initial_generation,
    )

    record = reloaded.extensions["scale.py"]
    assert record.source_hash == source_hash
    assert record.source_path == os.fspath(relocated_path.resolve())
    assert record.record_generation == initial_generation + 1


def test_relocating_a_script_requires_the_exact_filename(
    tmp_path: pathlib.Path,
) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    original_path = tmp_path / "original" / "Lab.py"
    original_path.parent.mkdir()
    source = _script(original_path)
    catalog, _source_hash = store.register_script(original_path)
    relocated_path = tmp_path / "relocated" / "lab.py"
    relocated_path.parent.mkdir()
    relocated_path.write_bytes(source)

    with pytest.raises(_ExtensionCatalogConflictError, match="different script name"):
        store.relocate_script(
            "Lab.py",
            relocated_path,
            expected_record_generation=catalog.extensions[
                _script_name_key("Lab.py")
            ].record_generation,
        )


def test_old_source_can_be_registered_as_a_separate_extension(
    tmp_path: pathlib.Path,
) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    script_path = tmp_path / "scale.py"
    first_source = _script(script_path)
    catalog, first_source_hash = store.register_script(script_path)
    _script(script_path, "data + scale")
    current_source_hash = hashlib.sha256(script_path.read_bytes()).hexdigest()
    catalog, changed = store.reload_script(
        "scale.py",
        expected_source_hash=current_source_hash,
        expected_record_generation=catalog.extensions["scale.py"].record_generation,
    )
    assert changed
    workspace_path = tmp_path / "scale_workspace.py"
    workspace_path.write_bytes(first_source)

    updated, registered_source_hash = store.register_script(
        workspace_path,
        expected_source_hash=first_source_hash,
    )

    assert registered_source_hash == first_source_hash
    assert updated.extensions["scale.py"].source_hash == current_source_hash
    assert updated.extensions["scale_workspace.py"].source_path == str(
        workspace_path.resolve()
    )


def test_failed_validation_does_not_change_the_shared_catalog(
    tmp_path: pathlib.Path,
) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    script_path = tmp_path / "broken.py"
    script_path.write_text("raise RuntimeError('broken import')\n")
    catalog, _source_hash = store.register_script(script_path)
    before = store.path.read_bytes()

    with pytest.raises(erlab.extensions.ExtensionImportError, match="broken import"):
        _validate_and_enable(
            store,
            "broken.py",
            expected_record_generation=catalog.extensions[
                "broken.py"
            ].record_generation,
        )

    record = store.read().extensions["broken.py"]
    assert store.path.read_bytes() == before
    assert record == catalog.extensions["broken.py"]


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
            catalog, source_hash = store.register_script(script_path)
            catalog = _validate_and_enable(
                store,
                "session_health.py",
                expected_record_generation=catalog.extensions[
                    "session_health.py"
                ].record_generation,
            )
            manager._extensions.catalog.refresh()
            record = catalog.extensions["session_health.py"]
            shared_catalog = store.read()
            sys.modules.pop(dependency_name)

            with pytest.raises(erlab.extensions.ExtensionImportError):
                manager._extensions.execution.validate_script(
                    record.script_name,
                    source_hash,
                    expected_record_generation=record.record_generation,
                    enable_script=False,
                    persist_result=False,
                )

            assert store.read() == shared_catalog
            assert (
                store.resolve_registered_capability(
                    record.script_name,
                    "routine",
                    "session_health",
                    source_hash=source_hash,
                ).descriptor.id
                == "session_health"
            )
            assert (
                manager._extensions.capability_status(
                    record.script_name, source_hash, "routine", "session_health"
                )
                == "validation-failed"
            )
            dialog = manager._extensions._manage_dialog
            assert dialog._buttons["toggle"].property("extensionActionState") == "retry"
            assert dialog._buttons["error"].isVisibleTo(dialog)

            sys.modules[dependency_name] = dependency
            manager._extensions.execution.validate_script(
                record.script_name,
                source_hash,
                expected_record_generation=record.record_generation,
                enable_script=False,
                persist_result=False,
            )

            assert store.read() == shared_catalog
            assert (
                manager._extensions.execution.validation_error(
                    record.script_name, source_hash
                )
                is None
            )
            assert (
                manager._extensions.capability_status(
                    record.script_name, source_hash, "routine", "session_health"
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
            catalog, source_hash = manager._extensions.catalog.store.register_script(
                script_path
            )
            manager._extensions.catalog.refresh()
            record = catalog.extensions["retry_health.py"]
            with pytest.raises(erlab.extensions.ExtensionImportError):
                manager._extensions.execution.validate_script(
                    record.script_name,
                    source_hash,
                    expected_record_generation=record.record_generation,
                )

            failed = manager._extensions.catalog.store.read().extensions[
                "retry_health.py"
            ]
            assert not failed.enabled
            assert not failed.approved
            assert (
                manager._extensions.execution.validation_error(
                    record.script_name, source_hash
                )
                is not None
            )

            sys.modules[dependency_name] = types.ModuleType(dependency_name)
            manager._extensions._manage_action("toggle", record.script_name)

            enabled = manager._extensions.catalog.store.read().extensions[
                "retry_health.py"
            ]
            assert enabled.enabled
            assert enabled.approved
            assert (
                manager._extensions.execution.validation_error(
                    record.script_name, source_hash
                )
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
        catalog, source_hash = first.store.register_script(script_path)
        _validate_and_enable(
            first.store,
            "scale.py",
            expected_record_generation=catalog.extensions["scale.py"].record_generation,
        )
        operation = ExtensionRoutineOperation(
            script_name="scale.py",
            source_hash=source_hash,
            routine_id="scale",
            routine_name="Scale",
            parameters={"scale": 3.0},
        )
        data = xr.DataArray([1.0, 2.0])

        xr.testing.assert_identical(operation.apply(data), data * 3.0)
        spec = ToolProvenanceSpec(
            kind="script",
            start_label="Start from data",
            seed_code="result = data",
            active_name="result",
        ).append_replay_stage(full_data(operation))
        generated = spec.display_code()
        if generated is None:
            raise RuntimeError("The registered routine did not generate code")
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


@pytest.mark.parametrize("display", [False, True])
def test_script_routine_workflow_code_reuses_one_loaded_module(
    tmp_path: pathlib.Path, display: bool
) -> None:
    script_path = tmp_path / "stateful.py"
    script_path.write_text(
        """import xarray as xr
from erlab.extensions import routine

counter = 0

@routine(name="Bump")
def bump(data: xr.DataArray) -> xr.DataArray:
    global counter
    counter += 1
    return data + counter
"""
    )
    catalog = _ExtensionCatalog(directory=tmp_path / "catalog")
    try:
        model, source_hash = catalog.store.register_script(script_path)
        _validate_and_enable(
            catalog.store,
            "stateful.py",
            expected_record_generation=model.extensions[
                "stateful.py"
            ].record_generation,
        )
        operation = ExtensionRoutineOperation(
            script_name="stateful.py",
            source_hash=source_hash,
            routine_id="bump",
            routine_name="Bump",
            parameters={},
        )
        spec = ToolProvenanceSpec(
            kind="script",
            start_label="Create data",
            seed_code="import xarray as xr\nload_script = xr.DataArray([0])",
            active_name="load_script",
        ).append_replay_stage(full_data(operation, operation))
        code = emit_replay_code(
            compile_replay_graph(spec, display=display), output_name="result"
        )
    finally:
        catalog.close()

    loaded = erlab.extensions.load_script(script_path)
    expected = loaded.erlab.routines["bump"][1](xr.DataArray([0]))
    expected = loaded.erlab.routines["bump"][1](expected)
    namespace: dict[str, typing.Any] = {}
    exec(code, namespace)  # noqa: S102

    load_calls = [
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_script"
    ]
    assert len(load_calls) == 1
    xr.testing.assert_identical(namespace["result"], expected)


def test_nested_extension_loader_and_routine_share_one_loaded_module(
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "stateful_io.py"
    script_path.write_text(
        """from pathlib import Path
import xarray as xr
from erlab.extensions import loader, routine

counter = 0

@loader(extensions=(".txt",))
def load_data(path: Path) -> xr.DataArray:
    global counter
    counter += 1
    return xr.DataArray([float(path.read_text()) + counter])

@routine()
def bump(data: xr.DataArray) -> xr.DataArray:
    global counter
    counter += 1
    return data + counter
"""
    )
    data_path = tmp_path / "value.txt"
    data_path.write_text("1")
    catalog = _ExtensionCatalog(directory=tmp_path / "catalog")
    try:
        model, source_hash = catalog.store.register_script(script_path)
        _validate_and_enable(
            catalog.store,
            "stateful_io.py",
            expected_record_generation=model.extensions[
                "stateful_io.py"
            ].record_generation,
        )
        file_spec = file_load(
            start_label="Load source data",
            seed_code="raise RuntimeError('recorded code must not run')",
            file_load_source=FileLoadSource(
                path=str(data_path),
                loader_label="Stateful data",
                loader_text="stateful_io: load_data",
                kwargs_text="",
                replay_call=FileReplayCall(
                    kind="extension_loader",
                    target="stateful_io.py",
                    source_hash=source_hash,
                    capability_id="load_data",
                    selection=FileDataSelection(kind="dataarray"),
                ),
                load_code="raise RuntimeError('recorded code must not run')",
            ),
        )
        operation = ExtensionRoutineOperation(
            script_name="stateful_io.py",
            source_hash=source_hash,
            routine_id="bump",
            routine_name="Bump",
            parameters={},
        )
        spec = ToolProvenanceSpec(
            kind="script",
            start_label="Use loaded data",
            seed_code="derived = load_script",
            active_name="derived",
            script_inputs=(
                ScriptInput(
                    name="load_script",
                    label="Loaded source",
                    provenance_spec=file_spec,
                ),
            ),
        ).append_replay_stage(full_data(operation))
        code = spec.display_code()
    finally:
        catalog.close()

    if code is None:
        raise RuntimeError("The extension workflow did not generate code")
    namespace: dict[str, typing.Any] = {}
    exec(code, namespace)  # noqa: S102

    module = ast.parse(code)
    load_calls = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_script"
    ]
    imports = [
        node
        for node in module.body
        if isinstance(node, ast.ImportFrom) and node.module == "erlab.extensions"
    ]
    assert len(load_calls) == 1
    assert len(imports) == 1
    assert [(alias.name, alias.asname) for alias in imports[0].names] == [
        ("load_script", None)
    ]
    xr.testing.assert_identical(namespace["derived"], xr.DataArray([4.0]))


def test_extension_loader_script_name_does_not_shadow_support_imports(
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "pathlib.py"
    _loader_script(script_path, name="Path data", extensions=(".txt",))
    data_path = tmp_path / "value.txt"
    data_path.write_text("5")
    catalog = _ExtensionCatalog(directory=tmp_path / "catalog")
    try:
        model, source_hash = catalog.store.register_script(script_path)
        _validate_and_enable(
            catalog.store,
            "pathlib.py",
            expected_record_generation=model.extensions["pathlib.py"].record_generation,
        )
        spec = file_load(
            start_label="Load path data",
            seed_code="raise RuntimeError('recorded code must not run')",
            file_load_source=FileLoadSource(
                path=str(data_path),
                loader_label="Path data",
                loader_text="pathlib: load_data",
                kwargs_text="",
                replay_call=FileReplayCall(
                    kind="extension_loader",
                    target="pathlib.py",
                    source_hash=source_hash,
                    capability_id="load_data",
                    selection=FileDataSelection(kind="dataarray"),
                ),
            ),
        )
        code = spec.display_code()
    finally:
        catalog.close()

    if code is None:
        raise RuntimeError("The registered loader did not generate code")
    namespace: dict[str, typing.Any] = {}
    exec(code, namespace)  # noqa: S102

    xr.testing.assert_identical(namespace["derived"], xr.DataArray([5.0]))


def test_extension_script_name_does_not_shadow_framework_imports(
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "era.py"
    _script(script_path)
    catalog = _ExtensionCatalog(directory=tmp_path / "catalog")
    try:
        model, source_hash = catalog.store.register_script(script_path)
        _validate_and_enable(
            catalog.store,
            script_path.name,
            expected_record_generation=model.extensions["era.py"].record_generation,
        )
        extension_operation = ExtensionRoutineOperation(
            script_name="era.py",
            source_hash=source_hash,
            routine_id="scale",
            routine_name="Scale",
            parameters={"scale": 2.0},
        )
        gaussian_operation = GaussianFilterOperation(sigma={"dim_0": 1.0})
        spec = ToolProvenanceSpec(
            kind="script",
            start_label="Create data",
            seed_code="import xarray as xr\ndata = xr.DataArray([1.0, 2.0])",
            active_name="data",
        ).append_replay_stage(full_data(extension_operation, gaussian_operation))
        code = spec.display_code()
    finally:
        catalog.close()

    if code is None:
        raise RuntimeError("The extension workflow did not generate code")
    namespace: dict[str, typing.Any] = {}
    exec(code, namespace)  # noqa: S102

    expected = gaussian_operation.apply(xr.DataArray([2.0, 4.0]))
    xr.testing.assert_identical(namespace["data"], expected)


@pytest.mark.parametrize(
    ("return_annotation", "return_expression", "selection", "cast_float64"),
    [
        (
            "xr.Dataset",
            'xr.Dataset({"signal": xr.DataArray([float(path.read_text())])})',
            FileDataSelection(kind="dataset_variable", value="signal"),
            False,
        ),
        (
            "xr.DataTree",
            "xr.DataTree.from_dict("
            '{"/group": xr.Dataset({"signal": '
            "xr.DataArray([float(path.read_text())])})})",
            FileDataSelection(kind="datatree_variable", value=("/group", "signal")),
            False,
        ),
        (
            "xr.DataArray",
            "xr.DataArray([int(path.read_text())])",
            FileDataSelection(kind="dataarray"),
            True,
        ),
    ],
)
def test_extension_loader_generated_code_preserves_output_selection(
    tmp_path: pathlib.Path,
    return_annotation: str,
    return_expression: str,
    selection: FileDataSelection,
    cast_float64: bool,
) -> None:
    script_path = tmp_path / "structured_loader.py"
    script_path.write_text(
        f"""from pathlib import Path
import xarray as xr
from erlab.extensions import loader

@loader(extensions=(".txt",))
def load_data(path: Path) -> {return_annotation}:
    return {return_expression}
"""
    )
    data_path = tmp_path / "value.txt"
    data_path.write_text("7")
    catalog = _ExtensionCatalog(directory=tmp_path / "catalog")
    try:
        model, source_hash = catalog.store.register_script(script_path)
        _validate_and_enable(
            catalog.store,
            "structured_loader.py",
            expected_record_generation=model.extensions[
                "structured_loader.py"
            ].record_generation,
        )
        spec = file_load(
            start_label="Load structured data",
            seed_code="raise RuntimeError('recorded code must not run')",
            file_load_source=FileLoadSource(
                path=str(data_path),
                loader_label="Structured data",
                loader_text="structured_loader: load_data",
                kwargs_text="",
                replay_call=FileReplayCall(
                    kind="extension_loader",
                    target="structured_loader.py",
                    source_hash=source_hash,
                    capability_id="load_data",
                    selection=selection,
                    cast_float64=cast_float64,
                ),
            ),
        )
        code = spec.display_code()
    finally:
        catalog.close()

    if code is None:
        raise RuntimeError("The registered loader did not generate code")
    namespace: dict[str, typing.Any] = {}
    exec(code, namespace)  # noqa: S102

    expected = xr.DataArray(
        [7.0],
        name=(
            "signal"
            if selection.kind in {"dataset_variable", "datatree_variable"}
            else None
        ),
    )
    xr.testing.assert_identical(namespace["derived"], expected)


def test_extension_generated_code_coerces_enum_and_path_parameters(
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "typed_extension.py"
    script_path.write_text(
        """from enum import Enum
from pathlib import Path
import xarray as xr
from erlab.extensions import loader, routine

class Mode(Enum):
    ADD = "add"

@loader(extensions=(".txt",))
def load_data(path: Path, extra_path: Path, mode: Mode) -> xr.DataArray:
    if mode is not Mode.ADD:
        raise ValueError("unexpected mode")
    return xr.DataArray([float(path.read_text()) + float(extra_path.read_text())])

@routine()
def scale(data: xr.DataArray, factor_path: Path, mode: Mode) -> xr.DataArray:
    if mode is not Mode.ADD:
        raise ValueError("unexpected mode")
    return data * float(factor_path.read_text())
"""
    )
    data_path = tmp_path / "value.txt"
    extra_path = tmp_path / "extra.txt"
    factor_path = tmp_path / "factor.txt"
    data_path.write_text("2")
    extra_path.write_text("3")
    factor_path.write_text("4")
    catalog = _ExtensionCatalog(directory=tmp_path / "catalog")
    try:
        model, source_hash = catalog.store.register_script(script_path)
        _validate_and_enable(
            catalog.store,
            "typed_extension.py",
            expected_record_generation=model.extensions[
                "typed_extension.py"
            ].record_generation,
        )
        operation = ExtensionRoutineOperation(
            script_name="typed_extension.py",
            source_hash=source_hash,
            routine_id="scale",
            routine_name="Scale",
            parameters={"factor_path": str(factor_path), "mode": "add"},
        )
        spec = file_load(
            start_label="Load typed data",
            seed_code="raise RuntimeError('recorded code must not run')",
            file_load_source=FileLoadSource(
                path=str(data_path),
                loader_label="Typed data",
                loader_text="typed_extension: load_data",
                kwargs_text="",
                replay_call=FileReplayCall(
                    kind="extension_loader",
                    target="typed_extension.py",
                    source_hash=source_hash,
                    capability_id="load_data",
                    kwargs={"extra_path": str(extra_path), "mode": "add"},
                    selection=FileDataSelection(kind="dataarray"),
                ),
            ),
        ).append_replay_stage(full_data(operation))
        code = spec.display_code()
    finally:
        catalog.close()

    if code is None:
        raise RuntimeError("The typed extension did not generate code")
    namespace: dict[str, typing.Any] = {}
    exec(code, namespace)  # noqa: S102

    xr.testing.assert_identical(namespace["derived"], xr.DataArray([20.0]))


def test_extension_modules_load_before_recorded_code_rebinds_public_import(
    tmp_path: pathlib.Path,
) -> None:
    first_path = tmp_path / "first.py"
    second_path = tmp_path / "second.py"
    _script(first_path, expression="data + 1.0")
    _script(second_path, expression="data * 2.0")
    catalog = _ExtensionCatalog(directory=tmp_path / "catalog")
    operations: list[ExtensionRoutineOperation] = []
    try:
        for path in (first_path, second_path):
            script_name = path.name
            model, source_hash = catalog.store.register_script(path)
            _validate_and_enable(
                catalog.store,
                script_name,
                expected_record_generation=model.extensions[
                    _script_name_key(script_name)
                ].record_generation,
            )
            operations.append(
                ExtensionRoutineOperation(
                    script_name=script_name,
                    source_hash=source_hash,
                    routine_id="scale",
                    routine_name="Scale",
                    parameters={"scale": 1.0},
                )
            )
        spec = script(
            operations[0],
            ScriptCodeOperation(
                label="Use a local load_script variable",
                code="load_script = data\ndata = load_script + 1.0",
            ),
            operations[1],
            start_label="Create data",
            seed_code="import xarray as xr\ndata = xr.DataArray([1.0])",
            active_name="data",
        )
        code = spec.display_code()
    finally:
        catalog.close()

    if code is None:
        raise RuntimeError("The extension workflow did not generate code")
    namespace: dict[str, typing.Any] = {}
    exec(code, namespace)  # noqa: S102

    xr.testing.assert_identical(namespace["data"], xr.DataArray([6.0]))


def test_script_routine_workflow_code_separates_sanitized_name_collisions(
    tmp_path: pathlib.Path,
) -> None:
    first_path = tmp_path / "lab-a.py"
    second_path = tmp_path / "lab_a.py"
    first_path.write_text(
        """import xarray as xr
from erlab.extensions import routine

@routine()
def transform(data: xr.DataArray) -> xr.DataArray:
    return data + 1.0
"""
    )
    second_path.write_text(
        """import xarray as xr
from erlab.extensions import routine

@routine()
def transform(data: xr.DataArray) -> xr.DataArray:
    return data * 2.0
"""
    )
    catalog = _ExtensionCatalog(directory=tmp_path / "catalog")
    operations: list[ExtensionRoutineOperation] = []
    try:
        for path in (first_path, second_path):
            model, source_hash = catalog.store.register_script(path)
            script_name = path.name
            _validate_and_enable(
                catalog.store,
                script_name,
                expected_record_generation=model.extensions[
                    _script_name_key(script_name)
                ].record_generation,
            )
            operations.append(
                ExtensionRoutineOperation(
                    script_name=script_name,
                    source_hash=source_hash,
                    routine_id="transform",
                    routine_name="Transform",
                    parameters={},
                )
            )
        spec = ToolProvenanceSpec(
            kind="script",
            start_label="Create data",
            seed_code="import xarray as xr\ndata = xr.DataArray([1.0, 2.0])",
            active_name="data",
        ).append_replay_stage(full_data(*operations))
        code = emit_replay_code(
            compile_replay_graph(spec, display=True), output_name="result"
        )
    finally:
        catalog.close()

    namespace: dict[str, typing.Any] = {}
    exec(code, namespace)  # noqa: S102
    module = ast.parse(code)
    imports = [
        node
        for node in module.body
        if isinstance(node, ast.ImportFrom) and node.module == "erlab.extensions"
    ]
    load_calls = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_script"
    ]
    assert len(imports) == 1
    assert len(load_calls) == 2
    xr.testing.assert_identical(namespace["result"], xr.DataArray([4.0, 6.0]))


def test_unregistered_script_routine_workflow_has_no_copied_code() -> None:
    operation = ExtensionRoutineOperation(
        script_name=f"missing-{uuid.uuid4().hex}.py",
        source_hash="a" * 64,
        routine_id="missing",
        routine_name="Missing",
        parameters={},
    )
    spec = ToolProvenanceSpec(
        kind="script",
        start_label="Create data",
        seed_code="import xarray as xr\ndata = xr.DataArray([1.0])",
        active_name="data",
    ).append_replay_stage(full_data(operation))

    with pytest.raises(ReplayGraphError, match="registered local script"):
        emit_replay_code(
            compile_replay_graph(spec, display=False), output_name="result"
        )


def test_script_routine_generated_code_renames_a_conflicting_data_variable(
    tmp_path: pathlib.Path,
) -> None:
    catalog = _ExtensionCatalog(directory=tmp_path / "catalog")
    script_path = tmp_path / "load_script.py"
    _script(script_path)
    try:
        model, source_hash = catalog.store.register_script(script_path)
        _validate_and_enable(
            catalog.store,
            "load_script.py",
            expected_record_generation=model.extensions[
                "load_script.py"
            ].record_generation,
        )
        operation = ExtensionRoutineOperation(
            script_name="load_script.py",
            source_hash=source_hash,
            routine_id="scale",
            routine_name="Scale",
            parameters={"scale": 3.0},
        )
        data = xr.DataArray([1.0, 2.0])
        spec = ToolProvenanceSpec(
            kind="script",
            start_label="Start from data",
            seed_code="result = load_script",
            active_name="result",
        ).append_replay_stage(full_data(operation))
        code = spec.display_code()
        if code is None:
            raise RuntimeError("The registered routine did not generate code")
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
        first, old_source_hash = catalog.store.register_script(script_path)
        first = _validate_and_enable(
            catalog.store,
            "lab_routines.py",
            expected_record_generation=(
                first.extensions["lab_routines.py"].record_generation
            ),
        )
        operation = ExtensionRoutineOperation(
            script_name="lab_routines.py",
            source_hash=old_source_hash,
            routine_id="scale",
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
        new_source_hash = hashlib.sha256(script_path.read_bytes()).hexdigest()
        second, changed = catalog.store.reload_script(
            "lab_routines.py",
            expected_source_hash=new_source_hash,
            expected_record_generation=first.extensions[
                "lab_routines.py"
            ].record_generation,
        )
        assert changed
        _validate_and_enable(
            catalog.store,
            "lab_routines.py",
            expected_record_generation=(
                second.extensions["lab_routines.py"].record_generation
            ),
        )
        data = xr.DataArray([1.0, 2.0])
        spec = ToolProvenanceSpec(
            kind="script",
            start_label="Start from data",
            seed_code="result = data",
            active_name="result",
        ).append_replay_stage(full_data(operation))
        code = spec.display_code()
        if code is None:
            raise RuntimeError("The registered routine did not generate code")
    finally:
        catalog.close()

    namespace: dict[str, typing.Any] = {"data": data}
    exec(code, namespace)  # noqa: S102

    xr.testing.assert_identical(namespace["result"], data * 3.0 + 1.0)


def test_unregistered_script_provenance_remains_visible_without_copied_code() -> None:
    operation = ExtensionRoutineOperation(
        script_name=f"workspace-{uuid.uuid4().hex}.py",
        source_hash="a" * 64,
        routine_id="scale",
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
        model, source_hash = catalog.store.register_script(script_path)
        _validate_and_enable(
            catalog.store,
            "scale.py",
            expected_record_generation=model.extensions["scale.py"].record_generation,
        )
        operation = ExtensionRoutineOperation(
            script_name="scale.py",
            source_hash=source_hash,
            routine_id="scale",
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
    source_hash = "a" * 64
    operation = ExtensionRoutineOperation(
        script_name="lab.py",
        source_hash=source_hash,
        routine_id="normalize",
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
        script_name: str,
        requested_source_hash: str,
        capability_kind: str,
        capability_id: str,
    ) -> typing.Literal["ready"]:
        calls.append(
            (
                script_name,
                requested_source_hash,
                capability_kind,
                capability_id,
            )
        )
        return "ready"

    assert not can_reload_without_trust(spec, extension_status_resolver=ready)
    assert calls == []
    assert can_reload_with_trusted_code(spec, extension_status_resolver=ready)
    assert calls == [("lab.py", source_hash, "routine", "normalize")]
    assert not can_reload_with_trusted_code(
        spec,
        extension_status_resolver=lambda *_args: "disabled",
    )


def test_managed_reload_reason_uses_the_manager_extension_state(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_hash = "a" * 64
    operation = ExtensionRoutineOperation(
        script_name="lab.py",
        source_hash=source_hash,
        routine_id="normalize",
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
            script_name="scale.py",
            source_hash="a" * 64,
            routine_id="scale",
            routine_name="Scale",
            parameters={"scale": float("inf")},
        )

    with pytest.raises(ValueError, match="must be finite"):
        FileReplayCall(
            kind="extension_loader",
            target="scale.py",
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
            first.store.register_script(script_path)
        assert "scale.py" in second.model.extensions
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
            catalog, _source_hash = first.store.register_script(script_path)
        with qtbot.waitSignal(second.changed, timeout=3000):
            catalog = _validate_and_enable(
                first.store,
                "scale.py",
                expected_record_generation=(
                    catalog.extensions["scale.py"].record_generation
                ),
            )
        assert second.model.extensions["scale.py"].enabled

        with qtbot.waitSignal(second.changed, timeout=3000):
            catalog = first.store.update_script(
                "scale.py",
                expected_record_generation=(
                    catalog.extensions["scale.py"].record_generation
                ),
                embed_policy="always",
            )
        propagated = second.model.extensions["scale.py"]
        assert propagated.embed_policy == "always"

        with qtbot.waitSignal(second.changed, timeout=3000):
            first.store.set_routine_favorite("scale.py", "scale", favorite=True)
        assert second.model.routine_favorites == (("scale.py", "scale"),)

        with qtbot.waitSignal(second.changed, timeout=3000):
            first.store.remove_script(
                "scale.py", expected_record_generation=propagated.record_generation
            )
        assert "scale.py" not in second.model.extensions
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
        catalog, _source_hash = manager._extensions.catalog.store.register_script(
            script_path
        )
        catalog = _validate_and_enable(
            manager._extensions.catalog.store,
            "lab_loader.py",
            expected_record_generation=(
                catalog.extensions["lab_loader.py"].record_generation
            ),
        )
        manager._extensions.catalog.refresh()
        record = catalog.extensions["lab_loader.py"]
        loader_call = manager._extensions.execution.loader_call(
            record.script_name,
            record.source_hash,
            record.loaders[0].id,
        )
        resolved = _resolve_load_func(
            (loader_call, {}, FileDataSelection(kind="dataarray"))
        )
        if resolved is None:
            raise RuntimeError("The script loader did not resolve")
        script_path.write_text(
            """from pathlib import Path
import xarray as xr
from erlab.extensions import loader

@loader(id="load_data", extensions=(".txt",))
def read_data(path: Path) -> xr.DataArray:
    return xr.DataArray([2.0 * float(path.read_text())])
"""
        )
        new_source_hash = hashlib.sha256(script_path.read_bytes()).hexdigest()
        updated, changed = manager._extensions.catalog.store.reload_script(
            "lab_loader.py",
            expected_source_hash=new_source_hash,
            expected_record_generation=record.record_generation,
        )
        assert changed
        updated = _validate_and_enable(
            manager._extensions.catalog.store,
            "lab_loader.py",
            expected_record_generation=(
                updated.extensions["lab_loader.py"].record_generation
            ),
        )
        relocated_path = tmp_path / "relocated" / "lab_loader.py"
        relocated_path.parent.mkdir()
        script_path.replace(relocated_path)
        manager._extensions.catalog.store.relocate_script(
            "lab_loader.py",
            relocated_path,
            expected_record_generation=updated.extensions[
                "lab_loader.py"
            ].record_generation,
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


def test_loader_call_uses_pinned_bytes_after_unchanged_relocation(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "original" / "lab_loader.py"
    script_path.parent.mkdir()
    script_path.write_text(
        """from pathlib import Path
import xarray as xr
from erlab.extensions import loader

@loader(name="Lab data", extensions=(".txt",))
def load_data(path: Path) -> xr.DataArray:
    return xr.DataArray([float(path.read_text())], attrs={"origin": __file__})
"""
    )
    data_path = tmp_path / "value.txt"
    data_path.write_text("4")

    with manager_context() as manager:
        catalog, _source_hash = manager._extensions.catalog.store.register_script(
            script_path
        )
        catalog = _validate_and_enable(
            manager._extensions.catalog.store,
            "lab_loader.py",
            expected_record_generation=(
                catalog.extensions["lab_loader.py"].record_generation
            ),
        )
        manager._extensions.catalog.refresh()
        record = catalog.extensions["lab_loader.py"]
        loader_call = manager._extensions.execution.loader_call(
            record.script_name,
            record.source_hash,
            record.loaders[0].id,
        )
        relocated_path = tmp_path / "relocated" / "lab_loader.py"
        relocated_path.parent.mkdir()
        script_path.replace(relocated_path)
        manager._extensions.catalog.store.relocate_script(
            "lab_loader.py",
            relocated_path,
            expected_record_generation=record.record_generation,
        )
        manager._extensions.catalog.refresh()

        loaded_from_original_pin = loader_call(data_path)
        relocated_record = manager._extensions.catalog.model.extensions["lab_loader.py"]
        relocated_call = manager._extensions.execution.loader_call(
            relocated_record.script_name,
            relocated_record.source_hash,
            relocated_record.loaders[0].id,
        )
        loaded_from_relocated_registration = relocated_call(data_path)

    xr.testing.assert_identical(
        loaded_from_original_pin,
        xr.DataArray([4.0], attrs={"origin": str(script_path.resolve())}),
    )
    xr.testing.assert_identical(
        loaded_from_relocated_registration,
        xr.DataArray([4.0], attrs={"origin": str(relocated_path.resolve())}),
    )
    assert loader_call.registered_path == script_path.resolve()
    assert relocated_call.registered_path == relocated_path.resolve()


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
            load_code="load_script('/sender/workspace_loader.py').load_data()",
            replay_call=FileReplayCall(
                kind="extension_loader",
                target="workspace_loader.py",
                source_hash="a" * 64,
                capability_id="load_data",
                selection=FileDataSelection(kind="dataarray"),
            ),
        ),
    )

    assert spec.display_code() is None
    if spec.file_load_source is None:
        raise RuntimeError("The file provenance source was not stored")
    assert _load_source_details_from_provenance(spec.file_load_source).load_code is None
    nested = ToolProvenanceSpec(
        kind="script",
        start_label="Use embedded data",
        seed_code="derived = embedded_data",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="embedded_data",
                label="Embedded data",
                provenance_spec=spec,
            ),
        ),
    )
    assert nested.display_code() is None
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
        catalog, _source_hash = manager._extensions.catalog.store.register_script(
            script_path
        )
        _validate_and_enable(
            manager._extensions.catalog.store,
            "unavailable.py",
            expected_record_generation=(
                catalog.extensions["unavailable.py"].record_generation
            ),
        )
        manager._extensions.catalog.refresh()
        manager._extensions._sync_explorer_loaders()
        assert (
            manager._extensions.loader_by_name("unavailable.py:load_missing")
            is not None
        )
        assert any(
            script_name == "unavailable.py"
            for script_name, _descriptor in manager._extensions._enabled_routines()
        )

        script_path.unlink()
        manager._extensions._sync_explorer_loaders()

        assert manager._extensions.loader_by_name("unavailable.py:load_missing") is None
        assert all(
            script_name != "unavailable.py"
            for script_name, _descriptor in manager._extensions._enabled_routines()
        )
        assert all(
            getattr(func, "script_name", None) != "unavailable.py"
            for func, _defaults in manager._extensions.file_loaders().values()
        )


@pytest.mark.parametrize("failure", ["missing-source", "catalog-race"])
def test_loader_discovery_ignores_sources_that_cannot_be_pinned(
    manager_context,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    script_path = tmp_path / "loader.py"
    _loader_script(script_path, name="Lab data", extensions=(".dat",))

    with manager_context() as manager:
        controller = manager._extensions
        catalog, _source_hash = controller.catalog.store.register_script(script_path)
        _validate_and_enable(
            controller.catalog.store,
            "loader.py",
            expected_record_generation=(
                catalog.extensions["loader.py"].record_generation
            ),
        )
        controller.catalog.refresh()
        assert controller.loader_by_name("loader.py:load_data") is not None

        if failure == "missing-source":
            script_path.unlink()
        else:
            monkeypatch.setattr(
                controller.catalog.store,
                "resolve_script",
                lambda *_args: (_ for _ in ()).throw(
                    _ExtensionCatalogConflictError("changed by another manager")
                ),
            )

        controller._sync_explorer_loaders()

        assert controller.loader_by_name("loader.py:load_data") is None
        assert controller.file_loaders() == {}


def _set_workspace_script_state(
    manager: typing.Any,
    requirements: tuple[_WorkspaceScriptRequirement, ...],
    *,
    sources: dict[tuple[str, str], bytes] | None = None,
) -> None:
    """Install validated workspace script state for focused controller tests."""
    state = type(manager._workspace_state.extension_scripts)(requirements)
    for (script_name, source_hash), source in (sources or {}).items():
        state.remember_verified_source(script_name, source_hash, source)
    manager._workspace_state.extension_scripts.replace(state)


def test_workspace_requirement_resolution_does_not_import_embedded_code(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    marker = tmp_path / "imported"
    source = f"from pathlib import Path\nPath({str(marker)!r}).touch()\n".encode()
    source_hash = hashlib.sha256(source).hexdigest()
    requirement = _WorkspaceScriptRequirement(
        script_name="workspace_only.py",
        capability_id="routine",
        capability_name="Routine",
        capability_kind="routine",
        source_hash=source_hash,
        extension_api_version=1,
    )

    with manager_context() as manager:
        _set_workspace_script_state(
            manager,
            (requirement,),
            sources={(requirement.script_name, source_hash): source},
        )
        assert manager._extensions.resolved_workspace_requirements()[0].state == (
            "missing"
        )
        assert not marker.exists()


def test_workspace_requirements_dialog_refreshes_after_registration(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = b"# workspace source\n"
    source_hash = hashlib.sha256(source).hexdigest()
    requirement = _WorkspaceScriptRequirement(
        script_name="workspace_only.py",
        capability_id="routine",
        capability_name="Routine",
        capability_kind="routine",
        source_hash=source_hash,
        extension_api_version=1,
    )
    current = [_ResolvedWorkspaceRequirement(requirement=requirement, state="missing")]
    shown_dialogs = []

    with manager_context() as manager:
        _set_workspace_script_state(
            manager,
            (requirement,),
            sources={(requirement.script_name, source_hash): source},
        )
        monkeypatch.setattr(
            manager._extensions,
            "resolved_workspace_requirements",
            lambda: tuple(current),
        )

        def register(_script_name: str, _source_hash: str) -> bool:
            current[0] = current[0].model_copy(update={"state": "ready"})
            return True

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
    source_hash = hashlib.sha256(source).hexdigest()
    source_path.unlink()
    requirement = _WorkspaceScriptRequirement(
        script_name="workspace_scale.py",
        capability_id="scale",
        capability_name="Scale",
        capability_kind="routine",
        source_hash=source_hash,
        extension_api_version=1,
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
        operation = ExtensionRoutineOperation(
            script_name=requirement.script_name,
            source_hash=source_hash,
            routine_id="scale",
            routine_name="Scale",
            parameters={"scale": 3.0},
        )
        manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(xr.DataArray([1.0, 2.0])),
            show=False,
            provenance_spec=full_data(operation),
        )
        _set_workspace_script_state(
            manager,
            (requirement,),
            sources={(requirement.script_name, source_hash): source},
        )
        assert (
            manager._extensions.resolved_workspace_requirements()[0].state == "missing"
        )
        assert manager._extensions._save_and_register_embedded_script(
            requirement.script_name, source_hash
        )
        record = manager._extensions.catalog.store.read().extensions[
            requirement.script_name
        ]
        assert destination.read_bytes() == source
        assert record.script_name == destination.name
        assert record.enabled
        assert record.approved
        assert manager._extensions.resolved_workspace_requirements()[0].state == "ready"
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
    historical_hash = hashlib.sha256(historical_source).hexdigest()
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
        catalog, registered_hash = manager._extensions.catalog.store.register_script(
            script_path
        )
        assert registered_hash == historical_hash
        catalog = _validate_and_enable(
            manager._extensions.catalog.store,
            "shared.py",
            expected_record_generation=(
                catalog.extensions["shared.py"].record_generation
            ),
        )
        current_source = _script(script_path, "data + scale")
        current_hash = hashlib.sha256(current_source).hexdigest()
        catalog, changed = manager._extensions.catalog.store.reload_script(
            "shared.py",
            expected_source_hash=current_hash,
            expected_record_generation=catalog.extensions[
                "shared.py"
            ].record_generation,
        )
        assert changed
        catalog = _validate_and_enable(
            manager._extensions.catalog.store,
            "shared.py",
            expected_record_generation=(
                catalog.extensions["shared.py"].record_generation
            ),
        )
        manager._extensions.catalog.refresh()
        requirement = _WorkspaceScriptRequirement(
            script_name="shared.py",
            capability_id="scale",
            capability_name="Scale",
            capability_kind="routine",
            source_hash=historical_hash,
            extension_api_version=1,
        )
        manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(xr.DataArray([1.0])),
            show=False,
            provenance_spec=full_data(
                ExtensionRoutineOperation(
                    script_name="shared.py",
                    source_hash=historical_hash,
                    routine_id="scale",
                    routine_name="Scale",
                    parameters={},
                )
            ),
        )
        _set_workspace_script_state(
            manager,
            (requirement,),
            sources={("shared.py", historical_hash): historical_source},
        )
        assert manager._extensions._save_and_register_embedded_script(
            "shared.py", historical_hash
        )
        records = manager._extensions.catalog.store.read().extensions
        assert records["shared.py"].source_hash == current_hash
        workspace_record = records["shared_workspace.py"]
        assert workspace_record.enabled
        assert workspace_record.source_hash == historical_hash
        assert workspace_record.source_path == str(destination.resolve())
        resolved = manager._extensions.resolved_workspace_requirements()[0]
        assert resolved.state == "ready"
        assert resolved.requirement.script_name == workspace_record.script_name


def test_canceling_workspace_script_review_does_not_register_source(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "cancelled.py"
    source = _script(script_path)
    source_hash = hashlib.sha256(source).hexdigest()
    requirement = _WorkspaceScriptRequirement(
        script_name="cancelled.py",
        capability_id="scale",
        capability_name="Scale",
        capability_kind="routine",
        source_hash=source_hash,
        extension_api_version=1,
    )
    monkeypatch.setattr(
        extension_controller._SourceReviewDialog,
        "exec",
        lambda _dialog: QtWidgets.QDialog.DialogCode.Rejected,
    )

    with manager_context() as manager:
        _set_workspace_script_state(
            manager,
            (requirement,),
            sources={(requirement.script_name, source_hash): source},
        )
        assert not manager._extensions._save_and_register_embedded_script(
            requirement.script_name, source_hash
        )

        assert requirement.script_name not in (
            manager._extensions.catalog.store.read().extensions
        )
        assert (
            manager._extensions.resolved_workspace_requirements()[0].state == "missing"
        )


def test_canceling_workspace_script_save_does_not_register_source(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    source = _script(tmp_path / "cancelled.py")
    source_hash = hashlib.sha256(source).hexdigest()
    requirement = _WorkspaceScriptRequirement(
        script_name="cancelled.py",
        capability_id="scale",
        capability_name="Scale",
        capability_kind="routine",
        source_hash=source_hash,
        extension_api_version=1,
    )
    monkeypatch.setattr(
        extension_controller._SourceReviewDialog,
        "exec",
        lambda _dialog: QtWidgets.QDialog.DialogCode.Accepted,
    )
    monkeypatch.setattr(
        extension_controller.QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: ("", ""),
    )

    with manager_context() as manager:
        _set_workspace_script_state(
            manager,
            (requirement,),
            sources={(requirement.script_name, source_hash): source},
        )

        assert not manager._extensions._save_and_register_embedded_script(
            requirement.script_name, source_hash
        )
        assert manager._extensions.catalog.store.read().extensions == {}


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
            "_review_and_register",
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
        catalog, source_hash = manager._extensions.catalog.store.register_script(
            original_path
        )
        _validate_and_enable(
            manager._extensions.catalog.store,
            "analysis.py",
            expected_record_generation=(
                catalog.extensions["analysis.py"].record_generation
            ),
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

        manager._extensions._show_missing_script_recovery()
        dialog = manager._extensions._missing_scripts_dialog
        if dialog is None:
            raise RuntimeError("The missing-script dialog was not shown")
        assert dialog.tree.topLevelItemCount() == 1
        dialog.locate_button.click()
        qtbot.wait_until(
            lambda: manager._extensions._missing_scripts_dialog is None,
            timeout=3000,
        )

        snapshot = manager._extensions.catalog.store.resolve_script(
            "analysis.py", source_hash
        )
        assert snapshot.registered_path == relocated_path.resolve()


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
        catalog, _source_hash = manager._extensions.catalog.store.register_script(
            script_path
        )
        script_record = catalog.extensions["script.py"]
        dialog = manager._extensions._manage_dialog
        dialog.set_catalog(
            _ExtensionCatalogModel(extensions={"script.py": script_record}),
            {("script.py", script_record.source_hash): "Ready"},
        )

        def select(script_name: str) -> None:
            for index in range(dialog.tree.topLevelItemCount()):
                item = dialog.tree.topLevelItem(index)
                if item.data(0, QtCore.Qt.ItemDataRole.UserRole) == script_name:
                    dialog.tree.setCurrentItem(item)
                    return
            raise AssertionError(script_name)

        select("script.py")
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
        manager._extensions.catalog.store.register_script(script_path)
        manager._extensions.catalog.refresh()
        responses = iter(
            (
                QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Yes,
            )
        )
        monkeypatch.setattr(
            manager._extensions, "_removal_blocker", lambda _script_name: None
        )
        monkeypatch.setattr(
            QtWidgets.QMessageBox, "exec", lambda _dialog: next(responses)
        )

        manager._extensions._manage_action("remove", "restorable.py")
        assert "restorable.py" in manager._extensions.catalog.model.extensions
        assert script_path.is_file()

        manager._extensions._manage_action("remove", "restorable.py")
        assert "restorable.py" not in manager._extensions.catalog.model.extensions
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
        catalog, source_hash = manager._extensions.catalog.store.register_script(
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
                    target="source_loader.py",
                    source_hash=source_hash,
                    capability_id="source_loader",
                    selection=FileDataSelection(kind="dataarray"),
                ),
            ),
        )

        assert file_load_source_status(spec) == "extension-approval-required"
        assert not marker.exists()
        catalog = _validate_and_enable(
            manager._extensions.catalog.store,
            "source_loader.py",
            expected_record_generation=(
                catalog.extensions["source_loader.py"].record_generation
            ),
        )
        manager._extensions.catalog.refresh()
        marker.unlink()

        assert file_load_source_status(spec) == "loadable"
        assert not marker.exists()

        catalog = manager._extensions.catalog.store.update_script(
            "source_loader.py",
            expected_record_generation=(
                catalog.extensions["source_loader.py"].record_generation
            ),
            enabled=False,
        )
        assert file_load_source_status(spec) == "extension-disabled"
        catalog = manager._extensions.catalog.store.update_script(
            "source_loader.py",
            expected_record_generation=(
                catalog.extensions["source_loader.py"].record_generation
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
        assert file_load_source_status(missing_source) == ("extension-hash-mismatch")
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
        catalog, source_hash = manager._extensions.catalog.store.register_script(
            script_path
        )
        with pytest.raises(erlab.extensions.ExtensionImportError):
            manager._extensions.execution.validate_script(
                "broken_loader.py",
                source_hash,
                expected_record_generation=(
                    catalog.extensions["broken_loader.py"].record_generation
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
                    target="broken_loader.py",
                    source_hash=source_hash,
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
        catalog, source_hash = manager._extensions.catalog.store.register_script(
            script_path
        )
        requirement = _WorkspaceScriptRequirement(
            script_name="scale.py",
            capability_id="scale",
            capability_name="Scale",
            capability_kind="routine",
            source_hash=source_hash,
            extension_api_version=1,
        )
        manager._extensions.catalog.refresh()
        _set_workspace_script_state(manager, (requirement,))
        assert (
            manager._extensions.resolved_workspace_requirements(include_current=False)[
                0
            ].state
            == "approval-required"
        )

        catalog = _validate_and_enable(
            manager._extensions.catalog.store,
            "scale.py",
            expected_record_generation=(
                catalog.extensions["scale.py"].record_generation
            ),
        )
        manager._extensions.catalog.refresh()
        assert manager._extensions._resolve_requirement(requirement).state == "ready"
        assert manager._extensions.collect_workspace_requirements() == ()

        manager._extensions.catalog.store.update_script(
            "scale.py",
            expected_record_generation=(
                catalog.extensions["scale.py"].record_generation
            ),
            enabled=False,
        )
        manager._extensions.catalog.refresh()
        assert manager._extensions._resolve_requirement(requirement).state == "disabled"
        assert manager._extensions.collect_workspace_requirements() == ()

        _set_workspace_script_state(
            manager,
            (requirement.model_copy(update={"script_name": "missing.py"}),),
        )
        assert (
            manager._extensions.resolved_workspace_requirements(include_current=False)[
                0
            ].state
            == "missing"
        )
        _set_workspace_script_state(
            manager, (requirement.model_copy(update={"extension_api_version": 2}),)
        )
        assert (
            manager._extensions.resolved_workspace_requirements(include_current=False)[
                0
            ].state
            == "unsupported-api"
        )
        broken_path = tmp_path / "broken.py"
        broken_path.write_text("raise RuntimeError('broken import')\n")
        catalog, broken_hash = manager._extensions.catalog.store.register_script(
            broken_path
        )
        with pytest.raises(erlab.extensions.ExtensionImportError):
            manager._extensions.execution.validate_script(
                "broken.py",
                broken_hash,
                expected_record_generation=(
                    catalog.extensions["broken.py"].record_generation
                ),
            )
        manager._extensions.catalog.refresh()
        _set_workspace_script_state(
            manager,
            (
                requirement.model_copy(
                    update={
                        "script_name": "broken.py",
                        "source_hash": broken_hash,
                    }
                ),
            ),
        )
        assert (
            manager._extensions.resolved_workspace_requirements(include_current=False)[
                0
            ].state
            == "validation-failed"
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
        catalog, source_hash = manager._extensions.catalog.store.register_script(
            script_path
        )
        _validate_and_enable(
            manager._extensions.catalog.store,
            "catalog_source.py",
            expected_record_generation=(
                catalog.extensions["catalog_source.py"].record_generation
            ),
        )
        if stored_source is None:
            script_path.unlink()
        else:
            script_path.write_bytes(stored_source)
        manager._extensions.catalog.refresh()
        requirement = _WorkspaceScriptRequirement(
            script_name="catalog_source.py",
            capability_id="scale",
            capability_name="Scale",
            capability_kind="routine",
            source_hash=source_hash,
            extension_api_version=1,
        )
        _set_workspace_script_state(
            manager,
            (requirement,),
            sources={(requirement.script_name, source_hash): source},
        )

        assert manager._extensions.resolved_workspace_requirements()[0].state == (
            expected_state
        )


def test_removing_node_discards_only_its_workspace_requirements(
    manager_context,
) -> None:
    operation = ExtensionRoutineOperation(
        script_name="missing_extension.py",
        source_hash="a" * 64,
        routine_id="normalize",
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
        requirement = _WorkspaceScriptRequirement(
            script_name="missing_extension.py",
            capability_id="normalize",
            capability_name="Normalize",
            capability_kind="routine",
            source_hash="a" * 64,
            extension_api_version=1,
            referencing_nodes=(first_uid, second_uid),
        )
        _set_workspace_script_state(manager, (requirement,))

        manager.remove_imagetool(0)

        collected = manager._extensions.collect_workspace_requirements()
        assert len(collected) == 1
        assert collected[0].referencing_nodes == (second_uid,)

        manager.remove_imagetool(1)
        assert manager._extensions.collect_workspace_requirements() == ()


def test_collecting_requirements_reconciles_loaded_and_unresolved_nodes(
    manager_context,
) -> None:
    source_hash = "c" * 64
    operation = ExtensionRoutineOperation(
        script_name="workspace_routines.py",
        source_hash=source_hash,
        routine_id="normalize",
        routine_name="Normalize",
        parameters={},
    )

    with manager_context() as manager:
        tool = erlab.interactive.imagetool.ImageTool(xr.DataArray([1.0]))
        index = manager.add_imagetool(
            tool, show=False, provenance_spec=full_data(operation)
        )
        node = manager._node_for_target(index)
        requirement = _WorkspaceScriptRequirement(
            script_name="workspace_routines.py",
            capability_id="normalize",
            capability_name="Normalize",
            capability_kind="routine",
            source_hash=source_hash,
            extension_api_version=1,
            referencing_nodes=(node.uid, "unresolved-node"),
        )
        _set_workspace_script_state(manager, (requirement,))

        collected = manager._extensions.collect_workspace_requirements()
        assert collected[0].referencing_nodes == (node.uid, "unresolved-node")

        node.set_displayed_provenance(full_data())
        collected = manager._extensions.collect_workspace_requirements()
        assert collected[0].referencing_nodes == ("unresolved-node",)

        _set_workspace_script_state(
            manager,
            (requirement.model_copy(update={"referencing_nodes": (node.uid,)}),),
        )
        assert manager._extensions.collect_workspace_requirements() == ()


def test_collecting_requirements_merges_duplicate_loaded_capability(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    source = b"# embedded extension source\n"
    source_hash = hashlib.sha256(source).hexdigest()
    workspace_path = tmp_path / "merged-requirements.itws"
    operation = ExtensionRoutineOperation(
        script_name="shared_routines.py",
        source_hash=source_hash,
        routine_id="normalize",
        routine_name="Normalize",
        parameters={},
    )

    with manager_context() as manager:
        index = manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(xr.DataArray([1.0])),
            show=False,
            provenance_spec=full_data(operation),
        )
        loaded_uid = manager._node_for_target(index).uid
        base = _WorkspaceScriptRequirement(
            script_name="shared_routines.py",
            capability_id="normalize",
            capability_name="Normalize",
            capability_kind="routine",
            source_hash=source_hash,
            extension_api_version=1,
            referencing_nodes=("unresolved-existing",),
        )
        incoming = base.model_copy(
            update={
                "referencing_nodes": (loaded_uid,),
            }
        )
        _set_workspace_script_state(
            manager,
            (base, incoming),
            sources={(base.script_name, source_hash): source},
        )

        collected = manager._extensions.collect_workspace_requirements()
        manager._workspace_controller.saving._save_workspace_document(workspace_path)

    assert len(collected) == 1
    assert collected[0].referencing_nodes == (loaded_uid, "unresolved-existing")
    assert collected[0].script_name == "shared_routines.py"
    assert collected[0].capability_name == "Normalize"
    attrs = workspace_arrays._read_workspace_root_attrs_h5py(workspace_path)
    manifest = workspace_format._workspace_manifest_from_attrs(attrs)
    assert len(manifest["extension_requirements"]) == 1
    assert manifest["extension_requirements"][0]["referencing_nodes"] == [
        loaded_uid,
        "unresolved-existing",
    ]
    assert manifest["embedded_extension_sources"] == [
        {
            "script_name": "shared_routines.py",
            "source_hash": source_hash,
            "object_id": f"extension-source-{source_hash}",
        }
    ]


def test_collecting_requirements_merges_duplicate_unresolved_loaders(
    manager_context,
) -> None:
    source_hash = "1" * 64
    base = _WorkspaceScriptRequirement(
        script_name="shared_loaders.py",
        capability_id="load_data",
        capability_name="Load data",
        capability_kind="loader",
        source_hash=source_hash,
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
        _set_workspace_script_state(manager, (base, incoming))

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
    source_hash = "c" * 64
    data_path = tmp_path / "nested.txt"
    data_path.write_text("unused")
    operation = ExtensionRoutineOperation(
        script_name="nested_routines.py",
        source_hash=source_hash,
        routine_id="normalize",
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
                target="nested_loaders.py",
                source_hash=source_hash,
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
        (item.script_name, item.capability_kind, item.capability_id)
        for item in requirements
    } == {
        ("nested_loaders.py", "loader", "nested_loader"),
        ("nested_routines.py", "routine", "normalize"),
    }
    assert all(item.referencing_nodes == (node_uid,) for item in requirements)
    loader_requirement = next(
        item for item in requirements if item.capability_kind == "loader"
    )
    assert loader_requirement.file_sources == (str(data_path),)


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


def test_workspace_import_ignores_unselected_extension_requirements(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    requirement = _WorkspaceScriptRequirement(
        script_name="omitted_extension.py",
        capability_id="normalize",
        capability_name="Normalize",
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
        loader = manager._workspace_controller.loading
        loader._install_extension_scripts(
            loader._prepare_extension_scripts(
                tmp_path / "selected.itws",
                manifest,
                selected_paths={"0"},
            ),
            replace=True,
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

@loader(name="Counter", extensions=("tar.gz",))
def counter_loader(path: Path, scale: float = 1.0) -> xr.DataArray:
    global counter
    counter += 1
    return xr.DataArray([counter, float(path.read_text()) * scale])
"""
    )
    value_path = tmp_path / "value.TAR.GZ"
    value_path.write_text("4")

    with manager_context() as manager:
        catalog, _source_hash = manager._extensions.catalog.store.register_script(
            script_path
        )
        _validate_and_enable(
            manager._extensions.catalog.store,
            "counter_loader.py",
            expected_record_generation=(
                catalog.extensions["counter_loader.py"].record_generation
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
        assert "counter_loader.py:counter_loader" in (
            manager._extensions.explorer_loaders
        )
        manager._recent_name_filter = name_filter
        assert manager._recent_loader_name == "counter_loader.py:counter_loader"
        manager.ensure_explorer_initialized()
        assert (
            manager.explorer.current_explorer.loader_name
            == "counter_loader.py:counter_loader"
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


def test_decorated_loader_without_extensions_accepts_arbitrary_path(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "unrestricted_loader.py"
    script_path.write_text(
        """from pathlib import Path
import xarray as xr
from erlab.extensions import loader

@loader(name="Any file")
def unrestricted_loader(path: Path) -> xr.DataArray:
    return xr.DataArray([path.name])
"""
    )
    arbitrary_path = tmp_path / "value.arbitrary"

    with manager_context() as manager:
        catalog, _source_hash = manager._extensions.catalog.store.register_script(
            script_path
        )
        _validate_and_enable(
            manager._extensions.catalog.store,
            script_path.name,
            expected_record_generation=(
                catalog.extensions[script_path.name].record_generation
            ),
        )
        manager._extensions.catalog.refresh()

        assert tuple(manager._extensions.file_loaders(arbitrary_path)) == (
            "Any file (*)",
        )


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
        catalog, source_hash = manager._extensions.catalog.store.register_script(
            script_path
        )
        _validate_and_enable(
            manager._extensions.catalog.store,
            "reload_loader.py",
            expected_record_generation=(
                catalog.extensions["reload_loader.py"].record_generation
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
            script_name: str,
            source_hash: str,
            kind: str,
            capability_id: str,
        ) -> str:
            calls.append((script_name, source_hash, kind, capability_id))
            return original_status(script_name, source_hash, kind, capability_id)

        monkeypatch.setattr(
            manager._extensions,
            "capability_status",
            capability_status,
        )
        assert tool.slicer_area._direct_reloadable()
        assert calls == [("reload_loader.py", source_hash, "loader", "load_data")]
        assert tool.slicer_area._reload_unavailable_reason() is None

        current = manager._extensions.catalog.store.read().extensions[
            "reload_loader.py"
        ]
        manager._extensions.catalog.store.update_script(
            "reload_loader.py",
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
        first, _source_hash = manager._extensions.catalog.store.register_script(
            first_path
        )
        _validate_and_enable(
            manager._extensions.catalog.store,
            "first.py",
            expected_record_generation=(first.extensions["first.py"].record_generation),
        )
        second, _source_hash = manager._extensions.catalog.store.register_script(
            second_path
        )
        with pytest.raises(
            _ExtensionCatalogConflictError,
            match=r"conflicts with enabled script 'first\.py'",
        ):
            _validate_and_enable(
                manager._extensions.catalog.store,
                "second.py",
                expected_record_generation=(
                    second.extensions["second.py"].record_generation
                ),
            )
        manager._extensions.catalog.refresh()

        loaders = manager._extensions.file_loaders()
        assert tuple(loaders) == ("Lab Data (*.txt)",)
        rejected = manager._extensions.catalog.model.extensions["second.py"]
        assert not rejected.enabled
        assert (
            rejected.record_generation
            == second.extensions["second.py"].record_generation
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
        catalog, _source_hash = manager._extensions.catalog.store.register_script(
            script_path
        )
        with pytest.raises(
            _ExtensionCatalogConflictError,
            match="conflicts with built-in file dialog filters",
        ):
            _validate_and_enable(
                manager._extensions.catalog.store,
                "netcdf.py",
                expected_record_generation=(
                    catalog.extensions["netcdf.py"].record_generation
                ),
            )
        manager._extensions.catalog.refresh()

        loaders = manager._available_file_loaders()
        assert "NetCDF Files (*.nc *.nc4 *.cdf)" in loaders
        rejected = manager._extensions.catalog.model.extensions["netcdf.py"]
        assert not rejected.enabled
        assert (
            rejected.record_generation
            == catalog.extensions["netcdf.py"].record_generation
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
def slow(
    data: xr.DataArray,
    marker: pathlib.Path,
    started: pathlib.Path,
    release: pathlib.Path,
) -> xr.DataArray:
    started.touch()
    deadline = time.monotonic() + 10.0
    while not release.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Routine release was not received")
        time.sleep(0.01)
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
    started = tmp_path / "started"
    release = tmp_path / "release"
    failures: list[BaseException] = []
    loader_thread: threading.Thread | None = None

    with manager_context() as manager:
        catalog, source_hash = manager._extensions.catalog.store.register_script(
            script_path
        )
        _validate_and_enable(
            manager._extensions.catalog.store,
            "serialized.py",
            expected_record_generation=(
                catalog.extensions["serialized.py"].record_generation
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
            script_name="serialized.py",
            source_hash=source_hash,
            routine_id="slow",
            parameters={
                "marker": str(marker),
                "started": str(started),
                "release": str(release),
            },
            target=0,
        )
        qtbot.wait_until(started.exists, timeout=5000)
        assert manager._extensions.execution.active is not None

        def invoke_loader() -> None:
            try:
                call(marker)
            except BaseException as error:
                failures.append(error)

        execution = manager._extensions.execution

        def loader_is_admitted() -> bool:
            with execution._blocking_tasks_lock:
                return any(
                    getattr(task, "call", None) is call
                    for task in execution._blocking_tasks
                )

        try:
            loader_thread = threading.Thread(target=invoke_loader)
            loader_thread.start()
            qtbot.wait_until(loader_is_admitted, timeout=5000)
            current = manager._extensions.catalog.store.read().extensions[
                "serialized.py"
            ]
            manager._extensions.catalog.store.update_script(
                "serialized.py",
                expected_record_generation=current.record_generation,
                enabled=False,
            )
            manager._extensions.catalog.refresh()
        finally:
            release.touch()
            if loader_thread is not None:
                loader_thread.join(timeout=5)
                assert not loader_thread.is_alive()
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
    deadline = time.monotonic() + 5.0
    while path.read_text() != "release":
        if time.monotonic() >= deadline:
            raise TimeoutError("loader release was not received")
        time.sleep(0.01)
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
        catalog, source_hash = manager._extensions.catalog.store.register_script(
            script_path
        )
        _validate_and_enable(
            manager._extensions.catalog.store,
            "loader_first.py",
            expected_record_generation=(
                catalog.extensions["loader_first.py"].record_generation
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
            script_name="loader_first.py",
            source_hash=source_hash,
            routine_id="must_not_run",
            parameters={},
            target=0,
        )
        assert manager._extensions.execution.active is None
        assert len(manager._extensions.execution.queued) == 1
        current = manager._extensions.catalog.store.read().extensions["loader_first.py"]
        manager._extensions.catalog.store.update_script(
            "loader_first.py",
            expected_record_generation=current.record_generation,
            enabled=False,
        )
        manager._extensions.catalog.refresh()
        marker.write_text("release")

        qtbot.wait_until(lambda: not loader_thread.is_alive(), timeout=5000)
        loader_thread.join()
        qtbot.wait_until(
            lambda: manager._extensions.execution.active is None,
            timeout=5000,
        )
        assert len(loader_failures) == 1
        assert isinstance(loader_failures[0], erlab.extensions.ExtensionExecutionError)
        assert "disabled" in str(loader_failures[0])
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
        catalog, source_hash = manager._extensions.catalog.store.register_script(
            script_path
        )
        _validate_and_enable(
            manager._extensions.catalog.store,
            "queued.py",
            expected_record_generation=(
                catalog.extensions["queued.py"].record_generation
            ),
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
                script_name="queued.py",
                source_hash=source_hash,
                routine_id="scale",
                parameters={"scale": 2.0},
                target=0,
            )

            assert manager._extensions.execution.active is None
            assert [job.job_id for job in manager._extensions.execution.queued] == [
                job_id
            ]
            active = manager._extensions.execution._active
            if active is None:
                pytest.fail("The queued routine worker was not retained")
            queued_worker = active[1]

            manager._extensions.execution.remove_queued(job_id)

            assert queued_worker.done.is_set()
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
        catalog, source_hash = manager._extensions.catalog.store.register_script(
            script_path
        )
        _validate_and_enable(
            manager._extensions.catalog.store,
            "inspect_input.py",
            expected_record_generation=(
                catalog.extensions["inspect_input.py"].record_generation
            ),
        )
        manager._extensions.catalog.refresh()
        manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(data, _in_manager=True),
            show=False,
        )

        manager._extensions.execution.queue_routine(
            script_name="inspect_input.py",
            source_hash=source_hash,
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
        assert operation.source_hash == source_hash

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


def test_queued_routine_uses_pinned_bytes_after_unchanged_relocation(
    manager_context,
    qtbot: pytest.QtBot,
    tmp_path: pathlib.Path,
) -> None:
    class BlockingTask(QtCore.QRunnable):
        def __init__(self) -> None:
            super().__init__()
            self.started = threading.Event()
            self.release = threading.Event()

        def run(self) -> None:
            self.started.set()
            self.release.wait()

    script_path = tmp_path / "original" / "scale.py"
    script_path.parent.mkdir()
    _script(script_path)
    data = xr.DataArray([1.0, 2.0], dims="x")

    with manager_context() as manager:
        catalog, source_hash = manager._extensions.catalog.store.register_script(
            script_path
        )
        catalog = _validate_and_enable(
            manager._extensions.catalog.store,
            "scale.py",
            expected_record_generation=(
                catalog.extensions["scale.py"].record_generation
            ),
        )
        manager._extensions.catalog.refresh()
        manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(data, _in_manager=True), show=False
        )
        blocking_task = BlockingTask()
        manager._extensions.execution._pool.start(blocking_task)
        if not blocking_task.started.wait(timeout=2.0):
            blocking_task.release.set()
            pytest.fail("The blocking extension task did not start")
        try:
            manager._extensions.execution.queue_routine(
                script_name="scale.py",
                source_hash=source_hash,
                routine_id="scale",
                parameters={"scale": 3.0},
                target=0,
            )
            queued = manager._extensions.execution.queued
            assert len(queued) == 1
            assert queued[0].snapshot.registered_path == script_path.resolve()

            relocated_path = tmp_path / "relocated" / "scale.py"
            relocated_path.parent.mkdir()
            script_path.replace(relocated_path)
            manager._extensions.catalog.store.relocate_script(
                "scale.py",
                relocated_path,
                expected_record_generation=(
                    catalog.extensions["scale.py"].record_generation
                ),
            )
            manager._extensions.catalog.refresh()
        finally:
            blocking_task.release.set()

        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        xr.testing.assert_identical(manager._get_imagetool_data(1), data * 3.0)


def test_started_routine_uses_pinned_bytes_during_catalog_reload(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "scale.py"
    _script(script_path)
    data = xr.DataArray([1.0, 2.0], dims="x")

    with manager_context() as manager:
        execution = manager._extensions.execution
        store = manager._extensions.catalog.store
        catalog, source_hash = store.register_script(script_path)
        _validate_and_enable(
            store,
            "scale.py",
            expected_record_generation=(
                catalog.extensions["scale.py"].record_generation
            ),
        )
        manager._extensions.catalog.refresh()
        target = manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(data, _in_manager=True), show=False
        )
        node = manager._node_for_target(target)
        job = execution._routine_job(
            script_name="scale.py",
            source_hash=source_hash,
            routine_id="scale",
            parameters={"scale": 3.0},
            input_data=node.data_for_role("displayed"),
            input_uid=node.uid,
            input_snapshot=node.snapshot_token,
        )
        reloaded = False

        original_require_dataarray = extension_execution._require_dataarray

        def reload_after_invocation(value: typing.Any) -> xr.DataArray:
            nonlocal reloaded
            if not reloaded:
                reloaded = True
                changed_source = _script(script_path, "data * scale + 100.0")
                current = store.read().extensions["scale.py"]
                store.reload_script(
                    "scale.py",
                    expected_source_hash=hashlib.sha256(changed_source).hexdigest(),
                    expected_record_generation=current.record_generation,
                )
            return original_require_dataarray(value)

        monkeypatch.setattr(
            extension_execution, "_require_dataarray", reload_after_invocation
        )
        worker = _ExtensionRoutineWorker(
            job,
            manager_session_id=execution._manager_session_id,
            catalog_store=store,
            script_modules=execution._script_modules,
            source_is_healthy=lambda *_args: True,
        )

        worker.run()

        result = worker.result
        if result is None:
            raise RuntimeError("The routine worker did not return a result")
        assert reloaded
        assert job.snapshot.registered_path == script_path.resolve()
        assert result.status == "success"
        xr.testing.assert_identical(result.output, data * 3.0)

        execution._insert_if_current(result)
        assert manager.ntools == 1


def test_finished_routine_is_not_inserted_while_manager_closes(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    routine = erlab.extensions.RoutineDescriptor(
        id="scale",
        name="Scale",
        category="Lab",
        summary="",
        function_name="scale",
    )
    snapshot = _pinned_script(tmp_path / "scale.py", routines=(routine,))
    data = xr.DataArray([1.0, 2.0], dims="x")

    with manager_context() as manager:
        execution = manager._extensions.execution
        target = manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(data, _in_manager=True), show=False
        )
        node = manager._node_for_target(target)
        job = extension_execution._ExtensionRoutineJob(
            job_id="closing-job",
            snapshot=snapshot,
            routine=routine,
            parameters={},
            input_uid=node.uid,
            input_snapshot=node.snapshot_token,
            input_data=data,
        )
        result = extension_execution._ExtensionRoutineResult(
            job=job,
            output=data + 1.0,
            duration=0.0,
            status="success",
        )

        manager._workspace_state.closing_document = True
        try:
            execution._insert_if_current(result)
        finally:
            manager._workspace_state.closing_document = False
        assert manager.ntools == 1

        def start_close_during_publication(*_args, **_kwargs) -> None:
            manager._workspace_state.closing_document = True

        with monkeypatch.context() as context:
            context.setattr(
                execution,
                "_require_current_capability",
                start_close_during_publication,
            )
            try:
                execution._insert_if_current(result)
            finally:
                manager._workspace_state.closing_document = False
        assert manager.ntools == 1


def test_synchronous_routine_replay_rejects_a_source_changed_while_running(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "waiting.py"
    started_path = tmp_path / "started"
    release_path = tmp_path / "release"
    script_path.write_text(
        f"""import pathlib
import time
import xarray as xr
from erlab.extensions import routine

@routine()
def scale(data: xr.DataArray) -> xr.DataArray:
    pathlib.Path({str(started_path)!r}).touch()
    release = pathlib.Path({str(release_path)!r})
    while not release.exists():
        time.sleep(0.01)
    return data * 2.0
"""
    )
    data = xr.DataArray([1.0, 2.0])

    with manager_context() as manager:
        execution = manager._extensions.execution
        store = manager._extensions.catalog.store
        catalog, source_hash = store.register_script(script_path)
        _validate_and_enable(
            store,
            "waiting.py",
            expected_record_generation=(
                catalog.extensions["waiting.py"].record_generation
            ),
        )
        manager._extensions.catalog.refresh()
        operation = ExtensionRoutineOperation(
            script_name="waiting.py",
            source_hash=source_hash,
            routine_id="scale",
            routine_name="scale",
            parameters={},
        )
        reload_errors: list[BaseException] = []
        stop_reload = threading.Event()

        def reload_after_start() -> None:
            while not started_path.exists():
                if stop_reload.wait(0.005):
                    return
            try:
                changed_source = _script(script_path, "data + scale")
                current = store.read().extensions["waiting.py"]
                store.reload_script(
                    "waiting.py",
                    expected_source_hash=hashlib.sha256(changed_source).hexdigest(),
                    expected_record_generation=current.record_generation,
                )
            except BaseException as error:
                reload_errors.append(error)
            finally:
                release_path.touch()

        reload_thread = threading.Thread(target=reload_after_start)
        reload_thread.start()
        try:
            with pytest.raises(
                erlab.extensions.ExtensionExecutionError,
                match="became unavailable",
            ):
                execution.run_operation(operation, data)
        finally:
            release_path.touch()
            stop_reload.set()
            reload_thread.join(timeout=2.0)

        assert not reload_thread.is_alive()
        assert reload_errors == []


def test_started_loader_uses_pinned_bytes_during_catalog_reload(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "loader.py"
    _loader_script(script_path, name="Lab data", extensions=(".dat",))
    data_path = tmp_path / "value.dat"
    data_path.write_text("4")

    with manager_context() as manager:
        execution = manager._extensions.execution
        store = manager._extensions.catalog.store
        catalog, _source_hash = store.register_script(script_path)
        catalog = _validate_and_enable(
            store,
            "loader.py",
            expected_record_generation=(
                catalog.extensions["loader.py"].record_generation
            ),
        )
        manager._extensions.catalog.refresh()
        record = catalog.extensions["loader.py"]
        descriptor = record.loaders[0]
        call = execution.loader_call(
            record.script_name, record.source_hash, descriptor.id
        )
        reloaded = False

        original_require_loader_output = extension_execution._require_loader_output

        def reload_after_invocation(value: typing.Any) -> typing.Any:
            nonlocal reloaded
            if not reloaded:
                reloaded = True
                script_path.write_text(
                    """from pathlib import Path
import xarray as xr
from erlab.extensions import loader

@loader(name="Lab data", extensions=(".dat",))
def load_data(path: Path) -> xr.DataArray:
    return xr.DataArray([float(path.read_text()) + 100.0])
"""
                )
                changed_source = script_path.read_bytes()
                current = store.read().extensions["loader.py"]
                store.reload_script(
                    "loader.py",
                    expected_source_hash=hashlib.sha256(changed_source).hexdigest(),
                    expected_record_generation=current.record_generation,
                )
            return original_require_loader_output(value)

        monkeypatch.setattr(
            extension_execution,
            "_require_loader_output",
            reload_after_invocation,
        )

        with pytest.raises(
            erlab.extensions.ExtensionExecutionError,
            match="became unavailable",
        ):
            execution.run_loader(call, data_path, {})

        assert reloaded
        assert call.registered_path == script_path.resolve()


def test_active_and_queued_routines_reject_a_disabled_source(
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
        catalog, source_hash = manager._extensions.catalog.store.register_script(
            script_path
        )
        _validate_and_enable(
            manager._extensions.catalog.store,
            "slow.py",
            expected_record_generation=(
                catalog.extensions["slow.py"].record_generation
            ),
        )
        manager._extensions.catalog.refresh()
        manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(data, _in_manager=True), show=False
        )
        manager._extensions.execution.queue_routine(
            script_name="slow.py",
            source_hash=source_hash,
            routine_id="slow",
            parameters={"amount": 1.0, "delay": 0.2},
            target=0,
        )
        manager._extensions.execution.queue_routine(
            script_name="slow.py",
            source_hash=source_hash,
            routine_id="slow",
            parameters={"amount": 2.0, "delay": 0.0},
            target=0,
        )
        qtbot.wait_until(
            lambda: manager._extensions.execution.active is not None,
            timeout=2000,
        )
        current = manager._extensions.catalog.store.read().extensions["slow.py"]
        manager._extensions.catalog.store.update_script(
            "slow.py",
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
        assert manager.ntools == 1


def test_canceling_pending_loader_releases_waiter(tmp_path: pathlib.Path) -> None:
    call = _loader_call(
        tmp_path / "missing.py",
        erlab.extensions.LoaderDescriptor(
            id="load",
            name="Load",
            category="Lab",
            summary="",
            function_name="load",
        ),
        lambda *_args: xr.DataArray([1.0]),
    )
    worker = _ExtensionLoaderWorker(
        call,
        tmp_path / "data.txt",
        {},
        _ExtensionCatalogStore(tmp_path / "catalog"),
        {},
        source_is_healthy=lambda *_args: True,
    )

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
        catalog, source_hash = manager._extensions.catalog.store.register_script(
            script_path
        )
        result = manager._extensions.execution.validate_script(
            "validate_thread.py",
            source_hash,
            expected_record_generation=(
                catalog.extensions["validate_thread.py"].record_generation
            ),
        )

        assert result.extensions["validate_thread.py"].enabled
        assert int(marker_path.read_text()) != manager_thread

        failing_path = tmp_path / "stops.py"
        failing_path.write_text("raise SystemExit('extension requested exit')\n")
        catalog, source_hash = manager._extensions.catalog.store.register_script(
            failing_path
        )
        with pytest.raises(
            erlab.extensions.ExtensionExecutionError, match="SystemExit"
        ):
            manager._extensions.execution.validate_script(
                "stops.py",
                source_hash,
                expected_record_generation=(
                    catalog.extensions["stops.py"].record_generation
                ),
            )

        failed = manager._extensions.catalog.store.read().extensions["stops.py"]
        assert failed == catalog.extensions["stops.py"]
        assert "SystemExit: extension requested exit" in (
            manager._extensions.execution.validation_error(
                "stops.py", failed.source_hash
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
        catalog, source_hash = manager._extensions.catalog.store.register_script(
            script_path
        )
        manager._extensions.execution.validate_script(
            "owned_module.py",
            source_hash,
            expected_record_generation=(
                catalog.extensions["owned_module.py"].record_generation
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


def test_canceling_pending_validation_releases_waiter(tmp_path: pathlib.Path) -> None:
    worker = _ExtensionValidationWorker(
        "extension.py",
        "a" * 64,
        1,
        manager_session_id="manager",
        catalog_store=_ExtensionCatalogStore(tmp_path / "catalog"),
        script_modules={},
    )

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


def test_readonly_routine_input_preserves_lazy_array_backends() -> None:
    da = pytest.importorskip("dask.array")
    values = da.from_array(np.arange(3.0), chunks=2)
    coordinates = da.from_array(np.arange(3.0) + 10.0, chunks=2)
    source = xr.DataArray(
        values,
        dims="x",
        coords={"aux": ("x", coordinates)},
    )

    readonly = _readonly_array(source)

    assert readonly.data is values
    assert isinstance(readonly.coords["aux"].data, da.Array)
    assert readonly.coords["aux"].data.name == coordinates.name
    xr.testing.assert_identical(readonly.compute(), source.compute())


def test_extension_log_fields_accept_generic_dimension_mappings() -> None:
    fields = extension_execution._xarray_log_fields(
        types.SimpleNamespace(sizes={"x": 2, "y": 3})
    )

    assert fields["dimensions"] == ("x", "y")
    assert fields["shape"] == (2, 3)
    assert extension_execution._xarray_log_fields(
        types.SimpleNamespace(sizes=None)
    ) == {"type": "SimpleNamespace"}


def test_stale_routine_result_is_not_inserted(
    manager_context,
    qtbot: pytest.QtBot,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "slow.py"
    started_path = tmp_path / "started"
    release_path = tmp_path / "release"
    script_path.write_text(
        """import pathlib
import time
import xarray as xr
from erlab.extensions import routine

@routine(name="Slow")
def slow(
    data: xr.DataArray,
    started_path: str,
    release_path: str,
) -> xr.DataArray:
    pathlib.Path(started_path).touch()
    deadline = time.monotonic() + 10.0
    while not pathlib.Path(release_path).exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("The test did not release the routine")
        time.sleep(0.01)
    return data + 1.0
"""
    )
    data = xr.DataArray(np.arange(3.0), dims="x")

    with manager_context() as manager:
        catalog, source_hash = manager._extensions.catalog.store.register_script(
            script_path
        )
        _validate_and_enable(
            manager._extensions.catalog.store,
            "slow.py",
            expected_record_generation=(
                catalog.extensions["slow.py"].record_generation
            ),
        )
        manager._extensions.catalog.refresh()
        manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(data, _in_manager=True), show=False
        )
        manager._extensions.execution.queue_routine(
            script_name="slow.py",
            source_hash=source_hash,
            routine_id="slow",
            parameters={
                "started_path": str(started_path),
                "release_path": str(release_path),
            },
            target=0,
        )
        try:
            qtbot.wait_until(started_path.exists, timeout=5000)
            assert manager._extensions.execution.active is not None
            manager._tool_graph.root_wrappers[0]._advance_snapshot_token()
        finally:
            release_path.touch()

        qtbot.wait_until(
            lambda: manager._extensions.execution.active is None,
            timeout=10000,
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
        catalog, source_hash = manager._extensions.catalog.store.register_script(
            script_path
        )
        _validate_and_enable(
            manager._extensions.catalog.store,
            "invalid_shape.py",
            expected_record_generation=(
                catalog.extensions["invalid_shape.py"].record_generation
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
            script_name="invalid_shape.py",
            source_hash=source_hash,
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
    call = _loader_call(
        pathlib.Path("lab.py"),
        descriptor,
        lambda *_args, **_kwargs: xr.DataArray([1.0]),
    )
    adapter = extension_execution._DecoratedLoaderAdapter(call)

    with manager_context() as manager:
        assert manager._extensions.loader_name_for_callable(adapter.load) == (
            "lab.py:load_data"
        )
        assert manager._extensions.loader_name_for_callable(
            adapter.load_for_manager
        ) == ("lab.py:load_data")
        assert manager._extensions.loader_name_for_callable(call) == "lab.py:load_data"
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
        assert not manager._extensions._review_and_register(script_path)

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
        assert not manager._extensions._review_and_register(script_path)
        assert manager._extensions.catalog.model.extensions == {}

        monkeypatch.setattr(
            extension_controller._SourceReviewDialog,
            "exec",
            lambda _dialog: QtWidgets.QDialog.DialogCode.Accepted,
        )
        assert manager._extensions._review_and_register(script_path)
        record = manager._extensions.catalog.model.extensions["reviewed.py"]
        assert record.enabled
        assert record.approved


def test_reviewing_an_unchanged_unapproved_script_enables_it(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "reviewed.py"
    _script(script_path)

    with manager_context() as manager:
        manager._extensions.catalog.store.register_script(script_path)
        manager._extensions.catalog.refresh()
        monkeypatch.setattr(
            extension_controller._SourceReviewDialog,
            "exec",
            lambda _dialog: QtWidgets.QDialog.DialogCode.Accepted,
        )

        assert manager._extensions._review_and_register(script_path)

        record = manager._extensions.catalog.model.extensions["reviewed.py"]
        assert record.enabled
        assert record.approved


def test_manage_reload_paths(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "reloadable.py"
    _script(script_path)
    reviews: list[pathlib.Path] = []
    located: list[str] = []

    with manager_context() as manager:
        manager._extensions._manage_action("reload", "unknown.py")
        catalog, _source_hash = manager._extensions.catalog.store.register_script(
            script_path
        )
        manager._extensions.catalog.refresh()
        monkeypatch.setattr(
            manager._extensions,
            "_review_and_register",
            lambda path: reviews.append(path) or True,
        )
        manager._extensions._manage_action("reload", "reloadable.py")
        assert reviews == [script_path.resolve()]

        record = catalog.extensions["reloadable.py"]
        script_path.unlink()
        manager._extensions.catalog.refresh()
        monkeypatch.setattr(
            manager._extensions,
            "_locate_missing_script",
            lambda script_name: located.append(script_name) or True,
        )
        manager._extensions._manage_action("reload", record.script_name)

    assert located == ["reloadable.py"]


def test_catalog_change_refreshes_visible_extension_consumers(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class Explorer(QtWidgets.QWidget):
        def refresh_loader_choices(self) -> None:
            calls.append("explorer")

    with manager_context() as manager, monkeypatch.context() as test_patch:
        controller = manager._extensions
        menu = controller.menu
        if menu is None:
            raise RuntimeError("The manager extension menu was not created")
        test_patch.setattr(menu, "isVisible", lambda: True)
        test_patch.setattr(controller, "_populate_menu", lambda: calls.append("menu"))
        explorer = Explorer(manager)
        manager._standalone_app_windows["explorer"] = explorer
        tool = types.SimpleNamespace(
            _refresh_reload_data_action=lambda: calls.append("tool")
        )
        manager._tool_graph.nodes["extension-test-tool"] = types.SimpleNamespace(
            tool_window=tool
        )
        test_patch.setattr(manager, "_update_actions", lambda: calls.append("actions"))
        test_patch.setattr(manager, "_update_info", lambda: calls.append("details"))

        try:
            controller._catalog_changed(controller.catalog.model)
        finally:
            manager._tool_graph.nodes.pop("extension-test-tool")

    assert calls == ["menu", "explorer", "actions", "details", "tool"]


def test_workspace_resolution_distinguishes_missing_exact_sources(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "lab.py"
    _script(script_path)
    requested_source = b"requested source"
    requested_hash = hashlib.sha256(requested_source).hexdigest()
    requirement = _WorkspaceScriptRequirement(
        script_name=script_path.name,
        capability_id="analyze",
        capability_name="Analyze",
        capability_kind="routine",
        source_hash=requested_hash,
        extension_api_version=1,
    )

    with manager_context() as manager:
        catalog, _current_hash = manager._extensions.catalog.store.register_script(
            script_path
        )
        manager._extensions.catalog.refresh()
        scripts = workspace_state._WorkspaceScriptState((requirement,))
        scripts.remember_verified_source(
            requirement.script_name,
            requested_hash,
            requested_source,
        )
        manager._workspace_state.extension_scripts.replace(scripts)

        assert manager._extensions._resolve_requirement(requirement).state == (
            "hash-mismatch"
        )

        record = catalog.extensions[_script_name_key(script_path.name)]
        manager._extensions.catalog.store.remove_script(
            record.script_name,
            expected_record_generation=record.record_generation,
        )
        manager._extensions.catalog.refresh()
        recoverable = manager._extensions._resolve_requirement(requirement)
        assert recoverable.state == "missing"
        assert recoverable.detail == (
            "Save and register the script included with this workspace"
        )

        manager._workspace_state.extension_scripts.replace(
            workspace_state._WorkspaceScriptState((requirement,))
        )
        missing = manager._extensions._resolve_requirement(requirement)
        assert missing.state == "missing"
        assert missing.detail == "The required local script is unavailable"


def test_workspace_requirement_helpers_cover_empty_and_unavailable_nodes(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requirement = _WorkspaceScriptRequirement(
        script_name="lab.py",
        capability_id="analyze",
        capability_name="Analyze",
        capability_kind="routine",
        source_hash="a" * 64,
        extension_api_version=1,
        referencing_nodes=("node",),
    )

    with manager_context() as manager:
        scripts = manager._workspace_state.extension_scripts
        scripts.replace(workspace_state._WorkspaceScriptState((requirement,)))
        before = scripts.copy()
        scripts.rebase_nodes({})
        scripts.remove_node_references(())
        assert scripts.requirements == before.requirements

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
        store = manager._extensions.catalog.store
        catalog, source_hash = store.register_script(script_path)
        catalog = _validate_and_enable(
            store,
            script_path.name,
            expected_record_generation=(
                catalog.extensions[_script_name_key(script_path.name)].record_generation
            ),
        )
        store.update_script(
            script_path.name,
            expected_record_generation=catalog.extensions[
                _script_name_key(script_path.name)
            ].record_generation,
            embed_policy="always",
        )
        manager._extensions.catalog.refresh()
        operation = ExtensionRoutineOperation(
            script_name=script_path.name,
            source_hash=source_hash,
            routine_id="scale",
            routine_name="Scale",
            parameters={},
        )
        manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(xr.DataArray([1.0])),
            show=False,
            provenance_spec=full_data(operation),
        )

        extensions = manager._extensions
        requirements = extensions.collect_workspace_requirements()
        embedded_sources = extensions.collect_workspace_embedded_sources(
            (*requirements, *requirements)
        )

    assert len(requirements) == 1
    assert requirements[0].script_name == script_path.name
    assert embedded_sources == (
        (script_path.name, source_hash, script_path.read_bytes()),
    )


def test_never_policy_does_not_discard_a_different_required_source(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "changed.py"
    _script(script_path, "data + scale")
    historical_source = _script(tmp_path / "historical.py")
    historical_hash = hashlib.sha256(historical_source).hexdigest()

    with manager_context() as manager:
        store = manager._extensions.catalog.store
        catalog, _source_hash = store.register_script(script_path)
        store.update_script(
            script_path.name,
            expected_record_generation=catalog.extensions[
                _script_name_key(script_path.name)
            ].record_generation,
            embed_policy="never",
        )
        manager._extensions.catalog.refresh()
        manager._workspace_state.extension_scripts.remember_verified_source(
            script_path.name,
            historical_hash,
            historical_source,
        )
        requirement = _WorkspaceScriptRequirement(
            script_name=script_path.name,
            capability_id="scale",
            capability_name="Scale",
            capability_kind="routine",
            source_hash=historical_hash,
            extension_api_version=1,
            referencing_nodes=("missing-node",),
        )

        embedded_sources = manager._extensions.collect_workspace_embedded_sources(
            (requirement,)
        )

    assert embedded_sources == ((script_path.name, historical_hash, historical_source),)


def test_never_policy_preserves_recovery_source_when_local_file_changed(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "changed.py"
    original_source = _script(script_path)

    with manager_context() as manager:
        store = manager._extensions.catalog.store
        catalog, source_hash = store.register_script(script_path)
        store.update_script(
            script_path.name,
            expected_record_generation=catalog.extensions[
                _script_name_key(script_path.name)
            ].record_generation,
            embed_policy="never",
        )
        manager._workspace_state.extension_scripts.remember_verified_source(
            script_path.name,
            source_hash,
            original_source,
        )
        script_path.write_bytes(original_source + b"\n# changed\n")
        requirement = _WorkspaceScriptRequirement(
            script_name=script_path.name,
            capability_id="scale",
            capability_name="Scale",
            capability_kind="routine",
            source_hash=source_hash,
            extension_api_version=1,
            referencing_nodes=("missing-node",),
        )

        embedded_sources = manager._extensions.collect_workspace_embedded_sources(
            (requirement,)
        )

    assert embedded_sources == ((script_path.name, source_hash, original_source),)


@pytest.mark.parametrize("embed_policy", ["referenced", "never"])
def test_unused_script_follows_current_embedding_policy_without_requirements(
    manager_context,
    tmp_path: pathlib.Path,
    embed_policy: str,
) -> None:
    script_path = tmp_path / "unused.py"
    _script(script_path)

    with manager_context() as manager:
        store = manager._extensions.catalog.store
        catalog, source_hash = store.register_script(script_path)
        catalog = _validate_and_enable(
            store,
            script_path.name,
            expected_record_generation=catalog.extensions[
                _script_name_key(script_path.name)
            ].record_generation,
        )
        catalog = store.update_script(
            script_path.name,
            expected_record_generation=catalog.extensions[
                _script_name_key(script_path.name)
            ].record_generation,
            embed_policy="always",
        )
        manager._extensions.catalog.refresh()
        assert manager._extensions.collect_workspace_requirements() == ()
        assert manager._extensions.collect_workspace_embedded_sources(()) == (
            (script_path.name, source_hash, script_path.read_bytes()),
        )

        catalog = store.update_script(
            script_path.name,
            expected_record_generation=catalog.extensions[
                _script_name_key(script_path.name)
            ].record_generation,
            embed_policy=embed_policy,
        )
        manager._extensions.catalog.refresh()
        assert manager._extensions.collect_workspace_requirements() == ()
        assert manager._extensions.collect_workspace_embedded_sources(()) == ()
        assert manager._extensions._removal_blocker(script_path.name) is None

        store.update_script(
            script_path.name,
            expected_record_generation=catalog.extensions[
                _script_name_key(script_path.name)
            ].record_generation,
            enabled=False,
        )
        manager._extensions.catalog.refresh()
        assert manager._extensions.collect_workspace_requirements() == ()
        assert manager._extensions._removal_blocker(script_path.name) is None
        workspace_path = tmp_path / f"unused-{embed_policy}.itws"
        manager._workspace_controller.saving._save_workspace_document(workspace_path)

    attrs = workspace_arrays._read_workspace_root_attrs_h5py(workspace_path)
    manifest = workspace_format._workspace_manifest_from_attrs(attrs)
    assert manifest["extension_requirements"] == []
    with workspace_store.WorkspaceStore(workspace_path) as workspace:
        assert f"extension-source-{source_hash}" not in workspace.manifest_object_ids(
            manifest
        )


def test_unresolved_unused_requirement_is_preserved(manager_context) -> None:
    requirement = _WorkspaceScriptRequirement(
        script_name="missing_unused.py",
        capability_id="normalize",
        capability_name="Normalize",
        capability_kind="routine",
        source_hash="a" * 64,
        extension_api_version=1,
    )

    with manager_context() as manager:
        manager._workspace_state.extension_scripts.replace(
            workspace_state._WorkspaceScriptState((requirement,))
        )

        assert manager._extensions.collect_workspace_requirements() == (requirement,)
        assert manager._extensions._removal_blocker(requirement.script_name) is None


def test_workspace_registration_selects_the_script_requirement(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    source_path = tmp_path / "mixed.py"
    source = _script(source_path)
    source_hash = hashlib.sha256(source).hexdigest()
    base = _WorkspaceScriptRequirement(
        script_name=source_path.name,
        capability_id="scale",
        capability_name="Scale",
        capability_kind="routine",
        source_hash=source_hash,
        extension_api_version=1,
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
        scripts = workspace_state._WorkspaceScriptState((base,))
        scripts.remember_verified_source(base.script_name, source_hash, source)
        manager._workspace_state.extension_scripts.replace(scripts)
        assert manager._extensions._save_and_register_embedded_script(
            base.script_name, source_hash
        )
        requirements = manager._extensions.collect_workspace_requirements()

        assert (
            manager._extensions.capability_status(
                destination.name, source_hash, "routine", "scale"
            )
            == "ready"
        )
        assert (
            manager._extensions.catalog.model.extensions[
                _script_name_key(destination.name)
            ].script_name
            == destination.name
        )
        assert requirements == ()


def test_workspace_script_remap_updates_node_provenance_owners(
    manager_context,
) -> None:
    source_hash = "a" * 64
    operation = ExtensionRoutineOperation(
        script_name="workspace_routines.py",
        source_hash=source_hash,
        routine_id="normalize",
        routine_name="Normalize",
        parameters={},
    )
    old_spec = full_data(operation)

    with manager_context() as manager:
        manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(xr.DataArray([1.0])),
            show=False,
        )
        child_uid = manager.add_imagetool_child(
            erlab.interactive.imagetool.ImageTool(xr.DataArray([2.0])),
            0,
            show=False,
            source_spec=old_spec,
            provenance_spec=old_spec,
            source_state="stale",
        )
        input_tool = _ExtensionInputTool(xr.DataArray([3.0]))
        input_tool.set_script_inputs(
            (
                ScriptInput(
                    name="data",
                    source_spec=old_spec.model_dump(mode="json"),
                    provenance_spec=old_spec.model_dump(mode="json"),
                ),
            ),
            primary_input="data",
        )
        tool_uid = manager.add_childtool(
            input_tool,
            script_inputs={"data": child_uid},
            show=False,
        )
        input_tool._pending_script_inputs = input_tool.script_inputs
        manager._workspace_state.mark_clean()

        manager._extensions._remap_workspace_script(
            "workspace_routines.py", source_hash, "local_routines.py"
        )

        child = manager._tool_graph.nodes[child_uid]
        child_source = typing.cast(
            "ExtensionRoutineOperation", child.source_spec.operations[-1]
        )
        child_display = typing.cast(
            "ExtensionRoutineOperation", child.provenance_spec.operations[-1]
        )
        tool_source_spec = input_tool.script_inputs[0].parsed_source_spec()
        tool_input_spec = input_tool.script_inputs[0].parsed_provenance_spec()
        pending_inputs = input_tool._pending_script_inputs
        pending_tool_input_spec = (
            None
            if pending_inputs is None
            else pending_inputs[0].parsed_provenance_spec()
        )
        if (
            tool_source_spec is None
            or tool_input_spec is None
            or pending_tool_input_spec is None
        ):
            raise RuntimeError("ToolWindow input provenance was discarded")
        tool_source = typing.cast(
            "ExtensionRoutineOperation", tool_source_spec.operations[-1]
        )
        tool_input = typing.cast(
            "ExtensionRoutineOperation", tool_input_spec.operations[-1]
        )
        pending_tool_input = typing.cast(
            "ExtensionRoutineOperation", pending_tool_input_spec.operations[-1]
        )
        assert child_source.script_name == "local_routines.py"
        assert child_display.script_name == "local_routines.py"
        assert tool_source.script_name == "local_routines.py"
        assert tool_input.script_name == "local_routines.py"
        assert pending_tool_input.script_name == "local_routines.py"
        assert {child_uid, tool_uid}.issubset(manager._workspace_state.dirty_state)


def test_workspace_script_remap_updates_nested_and_loader_provenance(
    manager_context,
) -> None:
    source_hash = "a" * 64
    operation = ExtensionRoutineOperation(
        script_name="workspace.py",
        source_hash=source_hash,
        routine_id="normalize",
        routine_name="Normalize",
        parameters={},
    )
    nested = full_data(operation)
    provenance = ToolProvenanceSpec(
        kind="script",
        start_label="Use extension inputs",
        seed_code="result = nested",
        active_name="result",
        script_inputs=(
            ScriptInput(
                name="nested",
                provenance_spec=nested.model_dump(mode="json"),
            ),
        ),
        file_load_source=FileLoadSource(
            path="data.dat",
            loader_label="Workspace loader",
            loader_text="workspace.py: load_data",
            kwargs_text="",
            replay_call=FileReplayCall(
                kind="extension_loader",
                target="workspace.py",
                source_hash=source_hash,
                capability_id="load_data",
                selection=FileDataSelection(kind="dataarray"),
            ),
        ),
    )

    with manager_context() as manager:
        index = manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(xr.DataArray([1.0])),
            show=False,
            provenance_spec=provenance,
        )

        manager._extensions._remap_workspace_script(
            "workspace.py", source_hash, "registered.py"
        )

        remapped = manager._node_for_target(index).provenance_spec
        remapped_nested = remapped.script_inputs[0].parsed_provenance_spec()
        if remapped_nested is None:
            raise RuntimeError("The nested provenance was not retained")
        remapped_operation = typing.cast(
            "ExtensionRoutineOperation", remapped_nested.operations[-1]
        )
        assert remapped_operation.script_name == "registered.py"
        assert remapped.file_load_source is not None
        assert remapped.file_load_source.replay_call is not None
        assert remapped.file_load_source.replay_call.target == "registered.py"


def test_workspace_script_remap_updates_pending_tool_provenance(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    source_hash = "b" * 64
    operation = ExtensionRoutineOperation(
        script_name="embedded.py",
        source_hash=source_hash,
        routine_id="scale",
        routine_name="Scale",
        parameters={},
    )
    old_spec = full_data(operation)
    attrs = erlab.interactive.utils.ToolWindow._saved_script_input_attrs(
        (
            ScriptInput(
                name="data",
                source_spec=old_spec.model_dump(mode="json"),
                provenance_spec=old_spec.model_dump(mode="json"),
            ),
        ),
        "data",
    )

    with manager_context() as manager:
        manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(xr.DataArray([1.0])),
            show=False,
        )
        node = _ManagedWindowNode(
            manager,
            manager._next_node_uid("pending-extension-tool"),
            manager._tool_graph.root_wrappers[0].uid,
            None,
            window_kind="tool",
            name="Pending extension tool",
        )
        manager._register_child_node(node)
        node.set_pending_workspace_payload(
            "tool",
            tmp_path / "workspace.itws",
            "nodes/pending-extension-tool",
            payload_attrs=attrs,
        )
        manager._workspace_state.mark_clean()

        manager._extensions._remap_workspace_script(
            "embedded.py", source_hash, "saved_extension.py"
        )

        pending_attrs = node.pending_workspace_payload_attrs
        if pending_attrs is None:
            raise RuntimeError("Pending ToolWindow attributes were discarded")
        script_inputs, primary_input = (
            erlab.interactive.utils.ToolWindow._saved_script_input_metadata(
                pending_attrs
            )
        )
        assert primary_input == "data"
        source_spec = script_inputs[0].parsed_source_spec()
        input_spec = script_inputs[0].parsed_provenance_spec()
        if source_spec is None or input_spec is None:
            raise RuntimeError("Pending ToolWindow provenance was discarded")
        assert (
            typing.cast(
                "ExtensionRoutineOperation", source_spec.operations[-1]
            ).script_name
            == "saved_extension.py"
        )
        assert (
            typing.cast(
                "ExtensionRoutineOperation", input_spec.operations[-1]
            ).script_name
            == "saved_extension.py"
        )
        assert node.uid in manager._workspace_state.dirty_state


def test_workspace_script_remap_rolls_back_all_node_owners(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_hash = "c" * 64
    operation = ExtensionRoutineOperation(
        script_name="embedded.py",
        source_hash=source_hash,
        routine_id="scale",
        routine_name="Scale",
        parameters={},
    )
    old_spec = full_data(operation)
    requirement = _WorkspaceScriptRequirement(
        script_name="embedded.py",
        capability_id="scale",
        capability_name="Scale",
        capability_kind="routine",
        source_hash=source_hash,
        extension_api_version=1,
    )
    with manager_context() as manager:
        for value in (1.0, 2.0):
            manager.add_imagetool(
                erlab.interactive.imagetool.ImageTool(xr.DataArray([value])),
                show=False,
                provenance_spec=old_spec,
            )
        first = manager._tool_graph.root_wrappers[0]
        second = manager._tool_graph.root_wrappers[1]
        wrapper_type = type(second)
        original_remap = wrapper_type.remap_provenance_owners
        _set_workspace_script_state(manager, (requirement,))
        manager._workspace_state.mark_clean()

        def fail_second_node(
            node: _ManagedWindowNode,
            remap: Callable[[ToolProvenanceSpec], ToolProvenanceSpec],
        ):
            if node is second:
                raise RuntimeError("remap failed")
            return original_remap(node, remap)

        monkeypatch.setattr(
            wrapper_type,
            "remap_provenance_owners",
            fail_second_node,
        )
        with pytest.raises(RuntimeError, match="remap failed"):
            manager._extensions._remap_workspace_script(
                "embedded.py", source_hash, "saved_extension.py"
            )

        first_operation = typing.cast(
            "ExtensionRoutineOperation", first.provenance_spec.operations[-1]
        )
        assert first_operation.script_name == "embedded.py"
        assert manager._workspace_state.extension_scripts.requirements == (requirement,)
        assert not manager.is_workspace_modified


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
    source_hash = hashlib.sha256(source or b"unused").hexdigest()
    requirement = _WorkspaceScriptRequirement(
        script_name="unusable.py",
        capability_id="analyze",
        capability_name="Analyze",
        capability_kind="routine",
        source_hash=source_hash,
        extension_api_version=1,
    )
    warnings: list[None] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda *_args, **_kwargs: warnings.append(None),
    )

    with manager_context() as manager:
        scripts = workspace_state._WorkspaceScriptState(
            (requirement,) if requirements else ()
        )
        if source is not None:
            scripts.remember_verified_source(
                requirement.script_name, source_hash, source
            )
        manager._workspace_state.extension_scripts.replace(scripts)
        assert not manager._extensions._save_and_register_embedded_script(
            requirement.script_name, source_hash
        )

    assert bool(warnings) is warning_expected


def test_workspace_registration_reports_validation_failure(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    source_path = tmp_path / "failure.py"
    source = _script(source_path)
    source_hash = hashlib.sha256(source).hexdigest()
    requirement = _WorkspaceScriptRequirement(
        script_name=source_path.name,
        capability_id="scale",
        capability_name="Scale",
        capability_kind="routine",
        source_hash=source_hash,
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
        scripts = workspace_state._WorkspaceScriptState((requirement,))
        scripts.remember_verified_source(requirement.script_name, source_hash, source)
        manager._workspace_state.extension_scripts.replace(scripts)
        monkeypatch.setattr(
            manager._extensions.execution,
            "validate_script",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("validation failed")
            ),
        )
        assert not manager._extensions._save_and_register_embedded_script(
            requirement.script_name, source_hash
        )

    assert failures == ["The saved workspace extension could not be registered."]


def test_missing_workspace_script_recovery_can_repeat(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "analysis.py"
    _script(script_path)

    with manager_context() as manager:
        catalog, source_hash = manager._extensions.catalog.store.register_script(
            script_path
        )
        _validate_and_enable(
            manager._extensions.catalog.store,
            script_path.name,
            expected_record_generation=catalog.extensions[
                _script_name_key(script_path.name)
            ].record_generation,
        )
        manager._extensions.catalog.refresh()
        script_path.unlink()
        requirement = _WorkspaceScriptRequirement(
            script_name=script_path.name,
            capability_id="scale",
            capability_name="Scale",
            capability_kind="routine",
            source_hash=source_hash,
            extension_api_version=1,
        )
        manager._workspace_state.extension_scripts.replace(
            workspace_state._WorkspaceScriptState((requirement,))
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
    source_path = tmp_path / "workspace_scale.py"
    source = _script(source_path)
    source_hash = hashlib.sha256(source).hexdigest()
    requirement = _WorkspaceScriptRequirement(
        script_name=source_path.name,
        capability_id="scale",
        capability_name="Scale",
        capability_kind="routine",
        source_hash=source_hash,
        extension_api_version=1,
    )
    shown: list[None] = []

    with manager_context() as manager:
        scripts = workspace_state._WorkspaceScriptState((requirement,))
        scripts.remember_verified_source(requirement.script_name, source_hash, source)
        manager._workspace_state.extension_scripts.replace(scripts)
        monkeypatch.setattr(
            manager._extensions,
            "show_workspace_requirements",
            lambda: shown.append(None),
        )

        manager._extensions.notify_unavailable_workspace_requirements()

    assert shown == [None]


def test_workspace_notification_reports_unrecoverable_requirement(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requirement = _WorkspaceScriptRequirement(
        script_name="missing.py",
        capability_id="scale",
        capability_name="Scale",
        capability_kind="routine",
        source_hash="a" * 64,
        extension_api_version=1,
    )
    shown: list[erlab.interactive.utils.MessageDialog] = []
    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "exec",
        lambda dialog: shown.append(dialog),
    )

    with manager_context() as manager:
        manager._workspace_state.extension_scripts.replace(
            workspace_state._WorkspaceScriptState((requirement,))
        )

        manager._extensions.notify_unavailable_workspace_requirements()

    assert len(shown) == 1


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
    call = _loader_call(
        script_path,
        descriptor,
        lambda *_args, **_kwargs: xr.DataArray([1.0]),
        source=source,
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


def test_loader_import_failure_is_classified_as_a_source_failure(
    tmp_path: pathlib.Path,
) -> None:
    descriptor = erlab.extensions.LoaderDescriptor(
        id="load_data",
        name="Load data",
        category="Lab",
        summary="",
        function_name="load_data",
    )
    call = _ExtensionLoaderCall(
        manager_session_id="import-failure",
        snapshot=_pinned_script(
            tmp_path / "broken.py",
            b"raise RuntimeError('import failed')\n",
            loaders=(descriptor,),
        ),
        loader_id=descriptor.id,
        descriptor=descriptor,
        executor=lambda *_args: xr.DataArray([1.0]),
        publication_checker=lambda _call: None,
        publication_recorder=lambda _call: None,
    )

    with pytest.raises(extension_execution._ExtensionSourceLoadFailure) as exc_info:
        call._invoke(tmp_path / "data.txt", {}, {})

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_remove_queued_discards_pending_job(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "queued.py"
    _script(script_path)

    with manager_context() as manager:
        execution = manager._extensions.execution
        catalog, source_hash = execution._catalog.store.register_script(script_path)
        _validate_and_enable(
            execution._catalog.store,
            "queued.py",
            expected_record_generation=(
                catalog.extensions["queued.py"].record_generation
            ),
        )
        job = execution._routine_job(
            script_name="queued.py",
            source_hash=source_hash,
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
        catalog, source_hash = execution._catalog.store.register_script(script_path)
        _validate_and_enable(
            execution._catalog.store,
            "slow.py",
            expected_record_generation=(
                catalog.extensions["slow.py"].record_generation
            ),
        )
        execution._catalog.refresh()
        manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(
                xr.DataArray([1.0]), _in_manager=True
            ),
            show=False,
        )
        execution.queue_routine(
            script_name="slow.py",
            source_hash=source_hash,
            routine_id="slow",
            parameters={"delay": 0.4},
            target=0,
        )
        execution.queue_routine(
            script_name="slow.py",
            source_hash=source_hash,
            routine_id="slow",
            parameters={"delay": 0.4},
            target=0,
        )
        qtbot.wait_until(lambda: execution.active is not None, timeout=2000)

        call = _loader_call(
            script_path,
            loader_descriptor,
            lambda *_args, **_kwargs: xr.DataArray([1.0]),
            source=script_path.read_bytes(),
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

        catalog, _source_hash = controller.catalog.store.register_script(script_path)
        _validate_and_enable(
            controller.catalog.store,
            script_path.name,
            expected_record_generation=catalog.extensions[
                _script_name_key(script_path.name)
            ].record_generation,
        )
        controller.catalog.store.set_routine_favorite(
            script_path.name, "scale", favorite=True
        )
        controller.catalog.refresh()
        controller._recent.append((script_path.name, "scale"))

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
    tmp_path: pathlib.Path,
) -> None:
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    records: dict[str, _ScriptRecord] = {}
    for index in range(24):
        source_hash = f"{index + 1:064x}"
        script_name = f"extension-{index:02d}.py"
        records[_script_name_key(script_name)] = _ScriptRecord(
            script_name=script_name,
            source_path=os.fspath((tmp_path / script_name).resolve()),
            source_hash=source_hash,
            source_modified_at="2026-01-01T00:00:00+00:00",
            registered_at="2026-01-01T00:00:00+00:00",
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
        == "extension-12.py"
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

    assert dialog.selected_script_name == "extension-12.py"
    assert scroll_bar.value() == scroll_position
    dialog.search_edit.setText("extension-05")
    visible = [
        dialog.tree.topLevelItem(index)
        for index in range(dialog.tree.topLevelItemCount())
        if not dialog.tree.topLevelItem(index).isHidden()
    ]
    assert len(visible) == 1
    assert visible[0].data(0, QtCore.Qt.ItemDataRole.UserRole) == "extension-05.py"
    assert dialog.selected_script_name == "extension-05.py"


def test_extension_source_viewer_uses_python_editor(
    qtbot: pytest.QtBot,
) -> None:
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    viewer = extension_dialogs._SourceViewerDialog(
        "value = 1\n", parent, title="Source"
    )
    qtbot.addWidget(viewer)
    assert viewer.sizeHint() == QtCore.QSize(800, 600)
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
        controller.catalog.store.register_script(script_path)
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
            extension_controller,
            "live_manager_records",
            lambda *, strict=False, include_starting=False: (current, other),
        )
        assert controller._removal_blocker(script_path.name) is not None

        monkeypatch.setattr(
            extension_controller,
            "live_manager_records",
            lambda *, strict=False, include_starting=False: (current,),
        )
        monkeypatch.setattr(
            controller.execution, "uses_script", lambda script_name: True
        )
        assert controller._removal_blocker(script_path.name) is not None

        monkeypatch.setattr(
            controller.execution, "uses_script", lambda script_name: False
        )
        record = controller.catalog.model.extensions[_script_name_key(script_path.name)]
        requirement = _WorkspaceScriptRequirement(
            script_name=script_path.name,
            capability_id="scale",
            capability_name="Scale",
            capability_kind="routine",
            source_hash=record.source_hash,
            extension_api_version=1,
            referencing_nodes=("workspace-node",),
        )
        manager._workspace_state.extension_scripts.replace(
            workspace_state._WorkspaceScriptState((requirement,))
        )
        assert controller._removal_blocker(script_path.name) is not None

        manager._workspace_state.extension_scripts.clear()
        assert controller._removal_blocker(script_path.name) is None


def test_script_registration_reads_reviewed_bytes_inside_transaction(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = tmp_path / "transaction.py"
    reviewed_source = _script(script_path)
    reviewed_hash = hashlib.sha256(reviewed_source).hexdigest()
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    original_mutate = store.mutate

    def mutate_after_external_change(*args: typing.Any, **kwargs: typing.Any):
        script_path.write_bytes(reviewed_source + b"\n# changed\n")
        return original_mutate(*args, **kwargs)

    monkeypatch.setattr(store, "mutate", mutate_after_external_change)

    with pytest.raises(
        _ExtensionCatalogConflictError,
        match="changed after it was reviewed",
    ):
        store.register_script(script_path, expected_source_hash=reviewed_hash)

    assert store.read().extensions == {}


def test_validation_commit_rejects_a_script_changed_after_import(
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "changed_during_validation.py"
    source = _script(script_path)
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    catalog, source_hash = store.register_script(script_path)
    record = next(iter(catalog.extensions.values()))
    loaded = erlab.extensions.load_script(script_path)
    script_path.write_bytes(source + b"\n# changed\n")

    with pytest.raises(
        _ExtensionCatalogConflictError,
        match="changed during validation",
    ):
        store.commit_script_validation(
            record.script_name,
            source_hash=source_hash,
            expected_record_generation=record.record_generation,
            routines=tuple(item[0] for item in loaded.erlab.routines.values()),
            loaders=(),
        )


def test_loader_publication_hooks_are_explicit(tmp_path: pathlib.Path) -> None:
    script_path = tmp_path / "loader_publication.py"
    _loader_script(script_path, name="Numbers", extensions=(".txt",))
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    catalog, source_hash = store.register_script(script_path)
    record = next(iter(catalog.extensions.values()))
    snapshot = store.resolve_script(record.script_name, source_hash)
    loaded = erlab.extensions.load_script(script_path)
    descriptor = next(iter(loaded.erlab.loaders.values()))[0]
    events: list[str] = []
    call = extension_execution._ExtensionLoaderCall(
        manager_session_id="test-manager",
        snapshot=snapshot,
        loader_id=descriptor.id,
        descriptor=descriptor,
        executor=lambda _call, _path, _parameters: (
            events.append("execute") or xr.DataArray([1.0])
        ),
        publication_checker=lambda _call: events.append("check"),
        publication_recorder=lambda _call: events.append("record"),
    )

    result = call(tmp_path / "input.txt")

    assert result.identical(xr.DataArray([1.0]))
    assert events == ["execute"]
    call.require_current_for_publication()
    call.record_publication()
    assert events == ["execute", "check", "record"]


def test_loader_ingress_checks_and_records_at_the_insertion_boundary(
    manager_context,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = tmp_path / "numbers.py"
    _loader_script(script_path, name="Numbers", extensions=(".txt",))
    data_path = tmp_path / "numbers.txt"
    data_path.write_text("4")

    with manager_context() as manager:
        store = manager._extensions.catalog.store
        catalog, source_hash = store.register_script(script_path)
        catalog = _validate_and_enable(
            store,
            "numbers.py",
            expected_record_generation=(
                catalog.extensions["numbers.py"].record_generation
            ),
        )
        manager._extensions.catalog.refresh()
        call = manager._extensions.execution.loader_call(
            "numbers.py", source_hash, "load_data"
        )
        loaded = call(data_path)
        catalog = store.update_script(
            "numbers.py",
            expected_record_generation=(
                catalog.extensions["numbers.py"].record_generation
            ),
            enabled=False,
        )
        manager._extensions.catalog.refresh()
        monkeypatch.setattr(
            manager._data_ingress, "_error_creating_imagetool", lambda: None
        )

        assert manager._data_ingress.receive_data(
            [loaded], {}, show=False, _extension_publication=call
        ) == [False]
        assert manager.ntools == 0

        catalog = store.update_script(
            "numbers.py",
            expected_record_generation=(
                catalog.extensions["numbers.py"].record_generation
            ),
            enabled=True,
        )
        manager._extensions.catalog.refresh()
        call = manager._extensions.execution.loader_call(
            "numbers.py", source_hash, "load_data"
        )
        events: list[str] = []
        original_add = manager.add_imagetool

        def add_imagetool(*args: typing.Any, **kwargs: typing.Any) -> int:
            index = original_add(*args, **kwargs)
            events.append("insert")
            return index

        monkeypatch.setattr(manager, "add_imagetool", add_imagetool)
        monkeypatch.setattr(
            manager._workspace_state.extension_scripts,
            "remember_verified_source",
            lambda *_args, **_kwargs: events.append("record"),
        )

        assert manager._data_ingress.receive_data(
            [loaded], {}, show=False, _extension_publication=call
        ) == [True]
        assert events == ["insert", "record"]

        def fail_insertion(*_args: typing.Any, **_kwargs: typing.Any) -> typing.Never:
            raise RuntimeError("insertion failed")

        monkeypatch.setattr(manager, "add_imagetool", fail_insertion)
        assert manager._data_ingress.receive_data(
            [loaded], {}, show=False, _extension_publication=call
        ) == [False]
        assert events == ["insert", "record"]

        original_imagetool = manager_io.ImageTool

        def disable_script_during_construction(
            *args: typing.Any, **kwargs: typing.Any
        ) -> erlab.interactive.imagetool.ImageTool:
            tool = original_imagetool(*args, **kwargs)
            current = store.read().extensions["numbers.py"]
            store.update_script(
                current.script_name,
                expected_record_generation=current.record_generation,
                enabled=False,
            )
            manager._extensions.catalog.refresh()
            return tool

        monkeypatch.setattr(manager_io, "ImageTool", disable_script_during_construction)
        monkeypatch.setattr(manager, "add_imagetool", original_add)
        assert manager._data_ingress.receive_data(
            [loaded], {}, show=False, _extension_publication=call
        ) == [False]
        assert manager.ntools == 1
        assert events == ["insert", "record"]


def test_extension_publication_canonicalizes_workspace_script_filename(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "Analysis.py"
    source = _script(script_path)

    with manager_context() as manager:
        catalog, source_hash = manager._extensions.catalog.store.register_script(
            script_path
        )
        _validate_and_enable(
            manager._extensions.catalog.store,
            script_path.name,
            expected_record_generation=catalog.extensions[
                _script_name_key(script_path.name)
            ].record_generation,
        )
        manager._extensions.catalog.refresh()
        manager._workspace_state.mark_clean()
        manager._extensions.execution._require_current_capability(
            script_path.name,
            source_hash,
            "routine",
            "scale",
        )
        assert not manager.is_workspace_modified
        operation = ExtensionRoutineOperation(
            script_name="analysis.py",
            source_hash=source_hash,
            routine_id="scale",
            routine_name="Scale",
            parameters={},
        )
        uid = manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(xr.DataArray([1.0])),
            show=False,
            provenance_spec=full_data(operation),
        )
        requirement = _WorkspaceScriptRequirement(
            script_name="analysis.py",
            capability_id="scale",
            capability_name="Scale",
            capability_kind="routine",
            source_hash=source_hash,
            extension_api_version=1,
            referencing_nodes=(manager._tool_graph.root_wrappers[uid].uid,),
        )
        scripts = workspace_state._WorkspaceScriptState((requirement,))
        scripts.remember_verified_source("analysis.py", source_hash, source)
        manager._workspace_state.extension_scripts.replace(scripts)
        manager._workspace_state.mark_clean()

        manager._extensions.execution._require_current_capability(
            script_path.name,
            source_hash,
            "routine",
            "scale",
        )

        node = manager._tool_graph.root_wrappers[uid]
        remapped_operation = typing.cast(
            "ExtensionRoutineOperation", node.provenance_spec.operations[-1]
        )
        assert remapped_operation.script_name == script_path.name
        assert {
            requirement.script_name
            for requirement in manager._workspace_state.extension_scripts.requirements
        } == {script_path.name}
        assert {
            script_name
            for script_name, _source_hash in (
                manager._workspace_state.extension_scripts.verified_sources
            )
        } == {script_path.name}
        assert manager.is_workspace_modified


@pytest.mark.parametrize("script_name", ["", ".", "..", "bad\x00.py", "path/x.py"])
def test_script_name_key_rejects_invalid_basenames(script_name: str) -> None:
    with pytest.raises(ValueError, match=r"must be a \.py basename"):
        _script_name_key(script_name)


def test_extension_catalog_model_rejects_noncanonical_script_keys(
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "extension.py"
    record = _ScriptRecord(
        script_name=script_path.name,
        source_path=os.fspath(script_path),
        source_hash="a" * 64,
        source_modified_at="2026-01-01T00:00:00+00:00",
        registered_at="2026-01-01T00:00:00+00:00",
    )

    with pytest.raises(ValueError, match="key does not match"):
        _ExtensionCatalogModel(extensions={"wrong.py": record})
    with pytest.raises(ValueError, match="favorite must use a normalized"):
        _ExtensionCatalogModel(
            extensions={"extension.py": record},
            routine_favorites=(("Extension.py", "scale"),),
        )


def test_changed_script_is_not_offered_as_missing_and_requires_review(
    manager_context,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "analysis.py"
    source = _script(script_path)

    with manager_context() as manager:
        catalog, _source_hash = manager._extensions.catalog.store.register_script(
            script_path
        )
        _validate_and_enable(
            manager._extensions.catalog.store,
            script_path.name,
            expected_record_generation=catalog.extensions[
                _script_name_key(script_path.name)
            ].record_generation,
        )
        manager._extensions.catalog.refresh()
        script_path.write_bytes(source + b"\n# local update\n")

        assert manager._extensions._missing_script_records() == ()
        manager._extensions._refresh_manage_dialog()
        dialog = manager._extensions._manage_dialog
        assert dialog.selected_script_name == script_path.name
        assert dialog._buttons["reload"].property("extensionActionState") == "review"
        assert not dialog._buttons["view_source"].isEnabled()


def test_manage_dialog_preserves_selection_by_script_filename(
    qtbot,
    tmp_path: pathlib.Path,
) -> None:
    source_path = tmp_path / "Lab.py"
    source_path.write_bytes(b"pass\n")
    record = _pinned_script(source_path, b"pass\n").record
    updated = record.model_copy(
        update={"record_generation": record.record_generation + 1}
    )
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    dialog = extension_dialogs._ManageExtensionsDialog(parent)

    dialog.set_catalog(
        _ExtensionCatalogModel(
            extensions={_script_name_key(record.script_name): record}
        )
    )
    assert dialog.selected_script_name == "Lab.py"

    dialog.set_catalog(
        _ExtensionCatalogModel(
            extensions={_script_name_key(updated.script_name): updated}
        )
    )

    assert dialog.selected_script_name == "Lab.py"
    assert (
        dialog.findChild(QtWidgets.QPushButton, "manager_extension_open_folder_button")
        is None
    )


def test_replay_source_capture_commits_only_after_outer_publication(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    first = _pinned_script(tmp_path / "first.py", b"first\n")
    second = _pinned_script(tmp_path / "second.py", b"second\n")

    with manager_context() as manager:
        recorded: list[tuple[str, str, bytes]] = []
        monkeypatch.setattr(
            manager._workspace_state.extension_scripts,
            "remember_verified_source",
            lambda script_name, source_hash, source: recorded.append(
                (script_name, source_hash, source)
            ),
        )
        execution = manager._extensions.execution
        monkeypatch.setattr(execution, "_check_replay_capture", lambda _capture: None)

        with execution.capture_replay_sources() as outer:
            execution.stage_replay_source(first, "routine", "first")
            with execution.capture_replay_sources() as inner:
                execution.stage_replay_source(second, "loader", "second")
                inner.require_current_for_publication()
                inner.publish()
            assert recorded == []
            outer.require_current_for_publication()
            outer.publish()

        assert recorded == [
            (first.record.script_name, first.record.source_hash, first.source_bytes),
            (second.record.script_name, second.record.source_hash, second.source_bytes),
        ]

        def fail_after_publication() -> None:
            with execution.capture_replay_sources() as failed:
                execution.stage_replay_source(first, "routine", "first")
                failed.require_current_for_publication()
                failed.publish()
                raise RuntimeError("publication failed")

        with pytest.raises(RuntimeError, match="publication failed"):
            fail_after_publication()
        with execution.capture_replay_sources():
            execution.stage_replay_source(second, "loader", "second")

        assert len(recorded) == 2


def test_replay_source_capture_rechecks_every_nested_capability(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    first = _pinned_script(tmp_path / "first.py", b"first\n")
    second = _pinned_script(tmp_path / "second.py", b"second\n")
    current = {"first": True, "second": True}

    with manager_context() as manager:
        execution = manager._extensions.execution

        def check(capture) -> None:
            for *_identity, capability_id in capture.permits:
                if not current[capability_id]:
                    raise erlab.extensions.ExtensionExecutionError(
                        f"{capability_id} is no longer current"
                    )

        monkeypatch.setattr(execution, "_check_replay_capture", check)
        with execution.capture_replay_sources() as outer:
            execution.stage_replay_source(first, "routine", "first")
            with execution.capture_replay_sources() as inner:
                execution.stage_replay_source(second, "loader", "second")
                inner.require_current_for_publication()
                inner.publish()
            current["first"] = False
            with pytest.raises(erlab.extensions.ExtensionExecutionError, match="first"):
                outer.require_current_for_publication()


def test_provenance_edit_records_replay_source_after_replacement(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    snapshot = _pinned_script(tmp_path / "analysis.py", b"analysis\n")

    with manager_context() as manager:
        execution = manager._extensions.execution
        controller = manager._provenance_edit_controller
        events: list[str] = []

        def validate(*_args: typing.Any, **_kwargs: typing.Any) -> object:
            execution.stage_replay_source(snapshot, "routine", "analyze")
            return object()

        monkeypatch.setattr(controller, "_validated_edit", validate)
        monkeypatch.setattr(
            controller,
            "_apply_validated_edit",
            lambda _edit: events.append("replace"),
        )
        monkeypatch.setattr(
            manager._workspace_state.extension_scripts,
            "remember_verified_source",
            lambda *_args: events.append("record"),
        )
        monkeypatch.setattr(
            execution,
            "_check_replay_capture",
            lambda _capture: events.append("check"),
        )
        monkeypatch.setattr(
            controller,
            "_provenance_code_entries",
            lambda *_args, **_kwargs: (),
        )
        node = types.SimpleNamespace(
            displayed_provenance_spec=None,
            displayed_source_spec=None,
            parent_uid=None,
        )

        controller._validate_and_replace(
            typing.cast("typing.Any", node),
            "display",
            full_data(),
        )
        assert events == ["check", "replace", "record"]

        events.clear()

        def fail_replacement(_edit: object) -> typing.Never:
            events.append("replace")
            raise RuntimeError("replacement failed")

        monkeypatch.setattr(controller, "_apply_validated_edit", fail_replacement)
        with pytest.raises(RuntimeError, match="replacement failed"):
            controller._validate_and_replace(
                typing.cast("typing.Any", node),
                "display",
                full_data(),
            )
        assert events == ["check", "replace"]


def test_extension_models_validate_catalog_and_workspace_reference_identity(
    tmp_path: pathlib.Path,
) -> None:
    source_hash = "a" * 64
    record = _ScriptRecord(
        script_name="extension.py",
        source_path=os.fspath(tmp_path / "extension.py"),
        source_hash=source_hash,
        source_modified_at="2026-01-01T00:00:00+00:00",
        registered_at="2026-01-01T00:00:00+00:00",
    )
    with pytest.raises(pydantic.ValidationError, match="favorites must be unique"):
        _ExtensionCatalogModel(
            extensions={"extension.py": record},
            routine_favorites=(("extension.py", "value"),) * 2,
        )
    with pytest.raises(pydantic.ValidationError, match="unknown script"):
        _ExtensionCatalogModel(routine_favorites=(("extension.py", "value"),))
    with pytest.raises(pydantic.ValidationError, match="must not be empty"):
        _WorkspaceScriptRequirement(
            script_name="extension.py",
            capability_id="",
            capability_name="Value",
            capability_kind="routine",
            source_hash=source_hash,
            extension_api_version=1,
        )


def test_extension_dialog_helpers_handle_invalid_dates_and_empty_selection(
    qtbot,
) -> None:
    assert extension_dialogs._display_datetime(None) == ""
    assert extension_dialogs._display_datetime("not-a-date") == "not-a-date"
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    dialog = _RoutineSelectionDialog((), parent)
    qtbot.addWidget(dialog)
    dialog._toggle_favorite()


def test_catalog_reports_source_read_and_duplicate_favorite_failures(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _ExtensionCatalogStore(tmp_path / "catalog")
    source = tmp_path / "extension.py"
    _script(source)
    catalog, _source_hash = store.register_script(source)
    record = catalog.extensions["extension.py"]

    monkeypatch.setattr(
        extension_catalog.pathlib.Path,
        "open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("unreadable")),
    )
    with pytest.raises(FileNotFoundError, match=r"extension\.py"):
        extension_catalog._read_source_snapshot(source)
    monkeypatch.undo()

    unchanged = store.set_routine_favorite(record.script_name, "scale", favorite=False)
    assert unchanged.routine_favorites == ()
    with pytest.raises(KeyError):
        store.set_routine_favorite("missing.py", "scale", favorite=True)


def test_replay_source_capture_requires_validation_only_for_staged_sources(
    tmp_path: pathlib.Path,
) -> None:
    checked: list[object] = []
    capture = extension_execution._ReplaySourceCapture(
        publication_checker=checked.append
    )
    capture.publish()
    assert capture.published
    assert checked == []

    snapshot = _pinned_script(tmp_path / "extension.py")
    capture.permits[
        (
            "extension.py",
            snapshot.record.source_hash,
            "routine",
            "scale",
        )
    ] = snapshot
    with pytest.raises(RuntimeError, match="checked before"):
        capture.publish()
    capture.require_current_for_publication()
    capture.publish()
    assert checked == [capture]


def test_controller_saves_recovery_source_without_overwriting_user_files(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    source = b"print('workspace source')\n"
    warnings: list[pathlib.Path] = []
    failures: list[str] = []
    selected_path = ""

    monkeypatch.setattr(
        extension_controller.QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (selected_path, "Python scripts (*.py)"),
    )
    monkeypatch.setattr(
        extension_controller.QtWidgets.QMessageBox,
        "warning",
        lambda *_args, **_kwargs: warnings.append(pathlib.Path(selected_path)),
    )
    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        lambda _parent, _title, text, **_kwargs: failures.append(text),
    )

    with manager_context() as manager:
        controller = manager._extensions

        assert (
            controller._save_source_as_user_file(
                source, title="Save Source", suggested_name="../unsafe.py"
            )
            is None
        )

        selected_path = os.fspath(tmp_path / "saved_source")
        saved = controller._save_source_as_user_file(
            source, title="Save Source", suggested_name="extension.py"
        )
        assert saved == (tmp_path / "saved_source.py").resolve()
        assert saved.read_bytes() == source

        selected_path = os.fspath(saved)
        assert (
            controller._save_source_as_user_file(
                source, title="Save Source", suggested_name="extension.py"
            )
            == saved
        )

        saved.write_bytes(b"user source\n")
        assert (
            controller._save_source_as_user_file(
                source, title="Save Source", suggested_name="extension.py"
            )
            is None
        )
        assert saved.read_bytes() == b"user source\n"
        assert warnings == [saved]

        class FailingSaveFile:
            stage = "open"
            canceled = False

            def __init__(self, _path: str) -> None:
                pass

            def open(self, _mode: object) -> bool:
                return self.stage != "open"

            def write(self, value: bytes) -> int:
                return len(value) if self.stage != "write" else len(value) - 1

            def cancelWriting(self) -> None:
                type(self).canceled = True

            def commit(self) -> bool:
                return self.stage != "commit"

            def errorString(self) -> str:
                return f"{self.stage} failed"

        with monkeypatch.context() as save_patch:
            save_patch.setattr(
                extension_controller.QtCore, "QSaveFile", FailingSaveFile
            )
            for stage in ("open", "write", "commit"):
                FailingSaveFile.stage = stage
                selected_path = os.fspath(tmp_path / f"{stage}.py")
                assert (
                    controller._save_source_as_user_file(
                        source, title="Save Source", suggested_name="extension.py"
                    )
                    is None
                )

    assert FailingSaveFile.canceled
    assert failures == ["The extension script could not be saved."] * 3


def test_missing_script_recovery_reuses_and_releases_its_dialog(
    manager_context,
    qtbot: pytest.QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "missing.py"
    _script(script_path)

    with manager_context() as manager:
        catalog, _source_hash = manager._extensions.catalog.store.register_script(
            script_path
        )
        _validate_and_enable(
            manager._extensions.catalog.store,
            script_path.name,
            expected_record_generation=catalog.extensions[
                script_path.name
            ].record_generation,
        )
        manager._extensions.catalog.refresh()
        script_path.unlink()

        assert manager._extensions._show_missing_script_recovery()
        dialog = manager._extensions._missing_scripts_dialog
        if dialog is None:
            raise RuntimeError("The missing-script dialog was not shown")
        assert manager._extensions._show_missing_script_recovery()
        assert manager._extensions._missing_scripts_dialog is dialog
        located: list[str] = []
        monkeypatch.setattr(
            manager._extensions,
            "_locate_missing_script",
            lambda script_name: located.append(script_name) or False,
        )
        dialog.locate_requested.emit(script_path.name)
        assert located == [script_path.name]
        assert dialog.tree.topLevelItemCount() == 1

        dialog.reject()
        qtbot.wait_until(
            lambda: manager._extensions._missing_scripts_dialog is None,
            timeout=2000,
        )
        assert manager._extensions._missing_scripts_dialog_slots is None
        assert not manager._extensions._show_missing_script_recovery()
        assert manager._extensions._show_missing_script_recovery(repeat=True)
        repeated = manager._extensions._missing_scripts_dialog
        if repeated is None:
            raise RuntimeError("The missing-script dialog was not repeated")
        repeated.reject()


def test_execution_publication_and_admission_status_guards(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    descriptor = erlab.extensions.LoaderDescriptor(
        id="load_data",
        name="Load Data",
        category="Lab",
        summary="",
        function_name="load_data",
    )
    snapshot = _pinned_script(
        tmp_path / "loader.py", b"loader source\n", loaders=(descriptor,)
    )

    with manager_context() as manager:
        execution = manager._extensions.execution
        recorded: list[tuple[str, str, bytes]] = []
        monkeypatch.setattr(
            manager._workspace_state.extension_scripts,
            "remember_verified_source",
            lambda *args: recorded.append(typing.cast("tuple[str, str, bytes]", args)),
        )
        call = execution._loader_call_from_snapshot(snapshot, descriptor)
        execution._record_loader_publication(call)
        assert recorded == [
            (
                snapshot.record.script_name,
                snapshot.record.source_hash,
                snapshot.source_bytes,
            )
        ]

        execution.stage_replay_source(snapshot, "loader", descriptor.id)
        assert execution._replay_source_captures == []
        errors: list[tuple[str, str, str | None]] = []
        monkeypatch.setattr(
            execution,
            "_set_validation_error",
            lambda script_name, source_hash, detail: errors.append(
                (script_name, source_hash, detail)
            ),
        )

        def fail_source(task: object, **_kwargs: object) -> typing.Never:
            worker = typing.cast("_ExtensionLoaderWorker", task)
            worker.source_failure = True
            worker.traceback_text = "source traceback"
            raise RuntimeError("source failed")

        monkeypatch.setattr(execution, "_run_blocking_task", fail_source)
        with pytest.raises(RuntimeError, match="source failed"):
            execution.run_loader(call, tmp_path / "data.txt", {})
        assert errors == [(call.script_name, call.source_hash, "source traceback")]

        execution._accepting = False
        with pytest.raises(
            erlab.extensions.ExtensionExecutionError, match="shutting down"
        ):
            execution._require_current_capability(
                call.script_name, call.source_hash, "loader", call.loader_id
            )
        execution._accepting = True

        manager._extensions.catalog.load_error = "catalog failed"
        with pytest.raises(
            erlab.extensions.ExtensionExecutionError, match="catalog is unavailable"
        ):
            execution._routine_job(
                script_name="routine.py",
                source_hash=snapshot.record.source_hash,
                routine_id="scale",
                parameters={},
                input_data=xr.DataArray([1.0]),
                input_uid="input",
                input_snapshot="snapshot",
            )
        manager._extensions.catalog.load_error = None


def test_catalog_transactions_report_reachable_conflicts_and_no_ops(
    tmp_path: pathlib.Path,
) -> None:
    source_path = tmp_path / "Analysis.py"
    source = _script(source_path)
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("file")
    blocked_store = _ExtensionCatalogStore(blocked_parent / "catalog")
    with pytest.raises(extension_catalog._ExtensionCatalogLockError, match="directory"):
        blocked_store.register_script(source_path)
    watched_catalog = _ExtensionCatalog(directory=blocked_parent / "watched")
    try:
        watched_catalog.refresh()
        assert watched_catalog.load_error is not None
    finally:
        watched_catalog.close()

    store = _ExtensionCatalogStore(tmp_path / "catalog")
    catalog, source_hash = store.register_script(source_path)
    record = catalog.extensions["analysis.py"]
    with pytest.raises(_ExtensionCatalogConflictError, match="already registered"):
        store.register_script(source_path)

    same = store.relocate_script(
        record.script_name,
        source_path,
        expected_record_generation=record.record_generation,
    )
    assert same.generation == catalog.generation

    lower_case_path = tmp_path / "analysis.py"
    with pytest.raises(_ExtensionCatalogConflictError, match="different script name"):
        store.relocate_script(
            lower_case_path.name,
            lower_case_path,
            expected_record_generation=record.record_generation,
        )

    other_path = tmp_path / "other" / source_path.name
    other_path.parent.mkdir()
    other_path.write_bytes(b"different source\n")
    with pytest.raises(
        _ExtensionCatalogConflictError, match="different script contents"
    ):
        store.relocate_script(
            record.script_name,
            other_path,
            expected_record_generation=record.record_generation,
        )

    source_path.write_bytes(source + b"# changed\n")
    with pytest.raises(_ExtensionCatalogConflictError, match="after it was reviewed"):
        store.reload_script(
            record.script_name,
            expected_source_hash="0" * 64,
            expected_record_generation=record.record_generation,
        )
    source_path.write_bytes(source)

    unchanged = store.update_script(
        record.script_name,
        expected_record_generation=record.record_generation,
    )
    assert unchanged.generation == catalog.generation

    favored = store.set_routine_favorite(record.script_name, "scale", favorite=True)
    assert favored.routine_favorites == (("analysis.py", "scale"),)
    unfavored = store.set_routine_favorite(record.script_name, "scale", favorite=False)
    assert unfavored.routine_favorites == ()

    loaders = tuple(
        erlab.extensions.LoaderDescriptor(
            id=f"loader-{index}",
            name="Lab Data",
            category="Lab",
            summary="",
            function_name=f"load_{index}",
            extensions=(".txt",),
        )
        for index in range(2)
    )
    with pytest.raises(
        _ExtensionCatalogConflictError,
        match="duplicate file dialog filters",
    ):
        store.commit_script_validation(
            record.script_name,
            source_hash=source_hash,
            expected_record_generation=record.record_generation,
            routines=(),
            loaders=loaders,
        )

    removed = store.remove_script(
        record.script_name,
        expected_record_generation=record.record_generation,
    )
    assert removed.extensions == {}
    with pytest.raises(_ExtensionCatalogConflictError, match="another manager"):
        store.remove_script(
            record.script_name,
            expected_record_generation=record.record_generation,
        )


def test_controller_reports_action_and_source_failures(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "actions.py"
    _script(script_path)
    critical: list[str] = []
    warnings: list[str] = []
    information: list[str] = []
    refreshed: list[str] = []

    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        lambda _parent, _title, text, **_kwargs: critical.append(text),
    )
    monkeypatch.setattr(
        extension_controller.QtWidgets.QMessageBox,
        "warning",
        lambda _parent, _title, text, **_kwargs: warnings.append(text),
    )
    monkeypatch.setattr(
        extension_controller.QtWidgets.QMessageBox,
        "information",
        lambda _parent, _title, text, **_kwargs: information.append(text),
    )

    with manager_context() as manager:
        controller = manager._extensions
        catalog, _source_hash = controller.catalog.store.register_script(script_path)
        _validate_and_enable(
            controller.catalog.store,
            script_path.name,
            expected_record_generation=catalog.extensions[
                script_path.name
            ].record_generation,
        )
        controller.catalog.refresh()
        record = controller.catalog.model.extensions[script_path.name]
        monkeypatch.setattr(
            controller,
            "_refresh_manage_dialog",
            lambda: refreshed.append("manage"),
        )

        validation_calls: list[dict[str, typing.Any]] = []
        monkeypatch.setattr(
            controller.execution,
            "validation_error",
            lambda *_args: "validation traceback",
        )
        monkeypatch.setattr(
            controller.execution,
            "validate_script",
            lambda script_name, source_hash, **kwargs: validation_calls.append(
                {"script_name": script_name, "source_hash": source_hash, **kwargs}
            ),
        )
        controller._manage_action("toggle", record.script_name)
        assert validation_calls == [
            {
                "script_name": record.script_name,
                "source_hash": record.source_hash,
                "expected_record_generation": record.record_generation,
                "enable_script": False,
                "persist_result": False,
            }
        ]
        controller._manage_action("error", record.script_name)
        assert critical[-1] == "The extension could not be validated."

        monkeypatch.setattr(
            controller.catalog.store,
            "set_routine_favorite",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("failed")),
        )
        controller._set_routine_favorite(record.script_name, "scale", True)
        assert critical[-1] == "The routine favorite could not be changed."

        viewed: list[str] = []
        monkeypatch.setattr(
            extension_controller._SourceViewerDialog,
            "exec",
            lambda dialog: viewed.append(dialog.source.toPlainText()) or 0,
        )
        controller._show_source(record.script_name, record.source_hash)
        assert "def scale" in viewed[0]

        original_resolve = controller.catalog.store.resolve_script
        monkeypatch.setattr(
            controller.catalog.store,
            "resolve_script",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                FileNotFoundError(script_path)
            ),
        )
        controller._show_source(record.script_name, record.source_hash)
        assert critical[-1] == "The registered extension source could not be read."
        monkeypatch.setattr(
            controller.catalog.store, "resolve_script", original_resolve
        )

        shown_sources: list[tuple[str, str]] = []
        monkeypatch.setattr(
            controller,
            "_show_source",
            lambda script_name, source_hash: shown_sources.append(
                (script_name, source_hash)
            ),
        )
        controller._manage_action("view_source", record.script_name)
        assert shown_sources == [(record.script_name, record.source_hash)]

        opened: list[str] = []
        revealed: list[str] = []
        copied: list[str] = []
        monkeypatch.setattr(
            extension_controller.QtGui.QDesktopServices,
            "openUrl",
            lambda url: opened.append(url.toLocalFile()) or True,
        )
        monkeypatch.setattr(
            erlab.utils.misc,
            "open_in_file_manager",
            lambda path: revealed.append(os.fspath(path)),
        )
        monkeypatch.setattr(
            extension_controller.QtWidgets.QApplication,
            "clipboard",
            staticmethod(
                lambda: types.SimpleNamespace(
                    setText=lambda value: copied.append(value),
                    clear=lambda: None,
                )
            ),
        )
        controller._manage_action("open_source", record.script_name)
        controller._manage_action("reveal_source", record.script_name)
        controller._manage_action("copy_source", record.script_name)
        assert opened == [record.source_path]
        assert revealed == [record.source_path]
        assert copied == [record.source_path]

        script_path.unlink()
        controller._manage_action("open_source", record.script_name)
        assert information[-1] == "The registered source file is unavailable."

        original_update = controller.catalog.store.update_script
        monkeypatch.setattr(
            controller.catalog.store,
            "update_script",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                _ExtensionCatalogConflictError("changed")
            ),
        )
        controller._manage_action("embedding:always", record.script_name)
        assert warnings[-1] == "changed"
        monkeypatch.setattr(
            controller.catalog.store,
            "update_script",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("failed")),
        )
        controller._manage_action("embedding:always", record.script_name)
        assert critical[-1] == "The extension could not be changed."
        monkeypatch.setattr(controller.catalog.store, "update_script", original_update)

        assert not controller._locate_missing_script("unknown.py")
        monkeypatch.setattr(
            extension_controller.QtWidgets.QFileDialog,
            "getOpenFileName",
            lambda *_args, **_kwargs: ("", ""),
        )
        assert not controller._locate_missing_script(record.script_name)
        monkeypatch.setattr(
            extension_controller.QtWidgets.QFileDialog,
            "getOpenFileName",
            lambda *_args, **_kwargs: (record.source_path, "Python scripts (*.py)"),
        )
        monkeypatch.setattr(
            controller.catalog.store,
            "relocate_script",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("failed")),
        )
        assert not controller._locate_missing_script(record.script_name)
        assert critical[-1] == "The script location could not be updated."

        monkeypatch.setattr(
            controller.catalog.store,
            "resolve_script",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyError("unreadable")),
        )
        controller._refresh_manage_dialog = types.MethodType(
            extension_controller._ExtensionController._refresh_manage_dialog,
            controller,
        )
        controller._refresh_manage_dialog()

    assert refreshed


def test_extension_removal_rechecks_blockers_after_confirmation(
    manager_context,
    accept_dialog,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    script_path = tmp_path / "remove.py"
    _script(script_path)
    information: list[str] = []
    refreshes: list[None] = []

    with manager_context() as manager:
        controller = manager._extensions
        controller.catalog.store.register_script(script_path)
        controller.catalog.refresh()
        record = controller.catalog.model.extensions[script_path.name]
        with monkeypatch.context() as mp:
            mp.setattr(
                extension_controller,
                "live_manager_records",
                lambda **_kwargs: (_ for _ in ()).throw(
                    extension_controller.ImageToolManagerRegistryError("failed")
                ),
            )
            assert "could not be checked" in typing.cast(
                "str", controller._removal_blocker(record.script_name)
            )

            mp.setattr(
                controller,
                "_refresh_removal_eligibility",
                lambda *_args: refreshes.append(None),
            )
            mp.setattr(
                controller,
                "_removal_blocker",
                lambda _script_name: "initial blocker",
            )
            accept_dialog(
                lambda: controller._remove_extension(record),
                pre_call=lambda dialog: information.append(
                    typing.cast("QtWidgets.QMessageBox", dialog).text()
                ),
            )
            assert information[-1] == "initial blocker"

            blockers = iter((None, "new blocker"))
            mp.setattr(
                controller,
                "_removal_blocker",
                lambda _script_name: next(blockers),
            )
            accept_dialog(
                lambda: controller._remove_extension(record),
                chained_dialogs=2,
                pre_call=(
                    None,
                    lambda dialog: information.append(
                        typing.cast("QtWidgets.QMessageBox", dialog).text()
                    ),
                ),
                accept_call=(
                    lambda dialog: (
                        typing.cast("QtWidgets.QMessageBox", dialog)
                        .button(QtWidgets.QMessageBox.StandardButton.Yes)
                        .click()
                    ),
                    None,
                ),
            )
            assert information[-1] == "new blocker"
            assert script_path.name in controller.catalog.model.extensions

    assert len(refreshes) == 2


def test_execution_adapters_workers_and_result_lifecycle_guards(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    loader_descriptor = erlab.extensions.LoaderDescriptor(
        id="load_data",
        name="Load Data",
        category="Lab",
        summary="",
        function_name="load_data",
    )
    publications: list[str] = []
    loader_path = tmp_path / "loader.py"
    loader_call = _ExtensionLoaderCall(
        manager_session_id="manager",
        snapshot=_pinned_script(loader_path, loaders=(loader_descriptor,)),
        loader_id=loader_descriptor.id,
        descriptor=loader_descriptor,
        executor=lambda *_args: xr.DataArray([1.0]),
        publication_checker=lambda _call: publications.append("check"),
        publication_recorder=lambda _call: publications.append("record"),
    )
    adapter = extension_execution._DecoratedLoaderAdapter(loader_call)
    adapter.require_current_for_publication()
    adapter.record_publication()
    assert publications == ["check", "record"]
    monkeypatch.setattr(adapter, "load", lambda *_args, **_kwargs: object())
    with pytest.raises(
        erlab.extensions.ExtensionExecutionError, match="one xarray object"
    ):
        adapter.load_for_manager(tmp_path / "data.txt")

    import dask.array

    lazy_input = xr.DataArray(dask.array.from_array(np.arange(3)), dims="x")
    lazy_output = xr.DataArray(dask.array.from_array(np.arange(3) + 1), dims="x")
    detached = _detached_routine_output(lazy_output, lazy_input)
    assert isinstance(detached.data, dask.array.Array)

    loader_only_path = tmp_path / "loader_only.py"
    _loader_script(loader_only_path, name="Only Loader", extensions=(".txt",))
    loaded = erlab.extensions.load_script(loader_only_path)
    with pytest.raises(erlab.extensions.ExtensionExecutionError, match="missing from"):
        extension_execution._require_routine(loaded, "missing")

    routine = erlab.extensions.RoutineDescriptor(
        id="scale",
        name="Scale",
        category="Lab",
        summary="",
        function_name="scale",
    )
    bad_snapshot = _pinned_script(
        tmp_path / "bad.py",
        b"import os\nimport xarray as xr\n"
        b"from erlab.extensions import routine\n"
        b'if os.getenv("ERLAB_TEST_FAIL_EXTENSION_IMPORT"):\n'
        b'    raise RuntimeError("import failed")\n'
        b"@routine()\n"
        b"def scale(data: xr.DataArray) -> xr.DataArray:\n"
        b"    return data\n",
        routines=(routine,),
    )
    job = extension_execution._ExtensionRoutineJob(
        job_id="job",
        snapshot=bad_snapshot,
        routine=routine,
        parameters={},
        input_uid="input",
        input_snapshot="snapshot",
        input_data=xr.DataArray([1.0]),
    )
    worker = _ExtensionRoutineWorker(
        job,
        manager_session_id="worker-manager",
        catalog_store=_ExtensionCatalogStore(tmp_path / "worker-catalog"),
        script_modules={},
        source_is_healthy=lambda *_args: True,
    )
    monkeypatch.setattr(
        extension_execution,
        "_resolve_execution_capability",
        lambda *_args, **_kwargs: extension_execution._ExecutionCapability(
            "ready", bad_snapshot, routine
        ),
    )
    monkeypatch.setenv("ERLAB_TEST_FAIL_EXTENSION_IMPORT", "1")
    try:
        worker.run()
    finally:
        extension_execution._remove_manager_modules("worker-manager")
    if worker.result is None:
        raise RuntimeError("The routine worker did not return a result")
    assert worker.done.is_set()
    assert worker.result.source_failure
    assert worker.result.status == "failed"

    with manager_context() as manager:
        execution = manager._extensions.execution
        validation_changes: list[None] = []
        execution.validation_changed.connect(lambda: validation_changes.append(None))
        execution._set_validation_error("stale.py", "a" * 64, "failure")
        execution._set_validation_error("stale.py", "a" * 64, "updated failure")
        execution._set_validation_error("stale.py", "a" * 64, "updated failure")
        assert execution.validation_errors == {
            ("stale.py", "a" * 64): "updated failure"
        }
        assert validation_changes == [None, None]
        execution.prune_validation_errors(_ExtensionCatalogModel())
        assert execution.validation_errors == {}

        active_worker = _ExtensionRoutineWorker(
            job,
            manager_session_id="manager-worker",
            catalog_store=execution._catalog.store,
            script_modules={},
            source_is_healthy=lambda *_args: True,
        )
        active_worker.signals.finished.connect(execution._finished_slot)
        active_worker.signals.started.connect(execution._started_slot)
        execution._active = (job, active_worker)
        validation_errors: list[tuple[str, str, str | None]] = []
        dialogs: list[None] = []
        starts: list[None] = []
        with monkeypatch.context() as mp:
            mp.setattr(
                execution,
                "_set_validation_error",
                lambda *args: validation_errors.append(
                    typing.cast("tuple[str, str, str | None]", args)
                ),
            )
            mp.setattr(execution, "_start_next", lambda: starts.append(None))
            mp.setattr(
                extension_execution.erlab.interactive.utils.MessageDialog,
                "critical",
                lambda *_args, **_kwargs: dialogs.append(None),
            )
            execution._finished(
                extension_execution._ExtensionRoutineResult(
                    job=job,
                    output=None,
                    duration=0.0,
                    status="failed",
                    traceback_text="source traceback",
                    source_failure=True,
                )
            )
        assert validation_errors == [
            (job.script_name, job.source_hash, "source traceback")
        ]
        assert dialogs == [None]
        assert starts == [None]

        execution._accepting = False
        execution._insert_if_current(
            extension_execution._ExtensionRoutineResult(
                job=job,
                output=xr.DataArray([2.0]),
                duration=0.0,
                status="success",
            )
        )
        execution._accepting = True
        execution._pending.append(job)
        execution.shutdown()
        assert execution.queued == ()
