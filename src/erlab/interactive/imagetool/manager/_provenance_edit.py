from __future__ import annotations

import ast
import dataclasses
import pathlib
import traceback
import typing
import warnings

from qtpy import QtCore, QtWidgets

import erlab
from erlab.interactive.imagetool import dialogs, provenance
from erlab.interactive.imagetool._load_source import (
    _load_provenance_from_file_details,
    _loader_callable_text,
)
from erlab.interactive.imagetool.manager._dialogs import _LoaderOptionsWidget

if typing.TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    import xarray as xr

    from erlab.interactive.imagetool._mainwindow import ImageTool
    from erlab.interactive.imagetool.manager._mainwindow import ImageToolManager
    from erlab.interactive.imagetool.manager._wrapper import (
        _ImageToolWrapper,
        _ManagedWindowNode,
    )


@dataclasses.dataclass(frozen=True)
class _FileLoadBatchPeer:
    node: _ImageToolWrapper | _ManagedWindowNode
    scope: typing.Literal["display", "source"]
    spec: provenance.ToolProvenanceSpec
    original_path: pathlib.Path
    loader_summary: str


@dataclasses.dataclass(frozen=True)
class _ValidatedProvenanceEdit:
    node: _ImageToolWrapper | _ManagedWindowNode
    scope: typing.Literal["display", "source"]
    data: xr.DataArray
    spec: provenance.ToolProvenanceSpec
    filter_operation: provenance.ToolProvenanceOperation | None


class _ProvenanceReplayFailure(RuntimeError):
    def __init__(self, where: str, cause: Exception) -> None:
        super().__init__(f"{where}: {cause}")
        self.where = where
        self.cause = cause
        self.__cause__ = cause

    @property
    def missing_source_file(self) -> _MissingProvenanceSourceFileError | None:
        if isinstance(self.cause, _MissingProvenanceSourceFileError):
            return self.cause
        return None


class _MissingProvenanceSourceFileError(FileNotFoundError):
    def __init__(self, source_path: pathlib.Path) -> None:
        super().__init__(f"Recorded source file is no longer accessible: {source_path}")
        self.source_path = source_path


def _parse_loader_kwargs(text: str) -> dict[str, typing.Any]:
    text = text.strip()
    if not text or text == "(none)":
        return {}
    if text.startswith("{"):
        value = ast.literal_eval(text)
        if not isinstance(value, dict):
            raise TypeError("Loader kwargs must be a dictionary")
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Loader kwargs must use string keys")
        return dict(value)

    call = ast.parse(f"loader({text})", mode="eval").body
    if not isinstance(call, ast.Call) or call.args:
        raise TypeError("Use keyword arguments such as `key=value`")
    kwargs: dict[str, typing.Any] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            raise TypeError("Loader kwargs do not support ** unpacking")
        kwargs[keyword.arg] = ast.literal_eval(keyword.value)
    return kwargs


class _FileLoadEditDialog(QtWidgets.QDialog):
    def __init__(
        self,
        load_source: provenance.FileLoadSource,
        parent: QtWidgets.QWidget,
        *,
        batch_peers: Sequence[_FileLoadBatchPeer] = (),
    ) -> None:
        super().__init__(parent)
        self.setObjectName("managerProvenanceFileLoadEditDialog")
        self.setWindowTitle("Edit File Load")
        self.setModal(True)
        self._original_path = pathlib.Path(load_source.path).expanduser()
        self._batch_peers = tuple(batch_peers)
        self._batch_peers_by_uid = {peer.node.uid: peer for peer in self._batch_peers}
        replay_call = load_source.replay_call
        self._selection = (
            provenance.FileDataSelection(kind="dataarray")
            if replay_call is None
            else replay_call.selection
        )
        initial_kwargs = (
            _parse_loader_kwargs(load_source.kwargs_text)
            if replay_call is None
            else dict(replay_call.kwargs)
        )
        loader_extensions = initial_kwargs.pop("loader_extensions", None)

        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()
        form_layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        layout.addLayout(form_layout)

        self.path_edit = QtWidgets.QLineEdit(str(load_source.path), self)
        self.path_edit.setObjectName("managerProvenanceFilePathEdit")
        browse_button = QtWidgets.QToolButton(self)
        browse_button.setObjectName("managerProvenanceFileBrowseButton")
        browse_button.setText("…")
        browse_button.clicked.connect(self._browse)

        path_widget = QtWidgets.QWidget(self)
        path_layout = QtWidgets.QHBoxLayout(path_widget)
        path_layout.setContentsMargins(0, 0, 0, 0)
        path_layout.addWidget(self.path_edit, 1)
        path_layout.addWidget(browse_button)
        form_layout.addRow("File", path_widget)

        valid_loaders = erlab.interactive.utils.file_loaders(self.file_path())
        if not valid_loaders:
            valid_loaders = erlab.interactive.utils.file_loaders()

        selected_filter = None
        if replay_call is not None:
            for name_filter, (func, kwargs) in valid_loaders.items():
                if replay_call.kind == "erlab_loader":
                    loader = getattr(func, "__self__", None)
                    matches = (
                        isinstance(loader, erlab.io.dataloader.LoaderBase)
                        and loader.name == replay_call.target
                    )
                else:
                    matches = _loader_callable_text(func) == replay_call.target and all(
                        initial_kwargs.get(key) == value
                        for key, value in kwargs.items()
                    )
                if matches:
                    selected_filter = name_filter
                    break
        selected_filter = selected_filter or next(iter(valid_loaders), None)
        if selected_filter is not None:
            func, kwargs = valid_loaders[selected_filter]
            valid_loaders = dict(valid_loaders)
            valid_loaders[selected_filter] = (func, dict(kwargs) | initial_kwargs)

        self.loader_options = _LoaderOptionsWidget(
            self,
            valid_loaders,
            loader_extensions=(
                {selected_filter: dict(loader_extensions)}
                if selected_filter is not None and isinstance(loader_extensions, dict)
                else None
            ),
            sample_paths=(self.file_path(),),
        )
        self.loader_options.setObjectName("managerProvenanceLoaderOptionsWidget")
        self.loader_options.check_filter(selected_filter)
        self.kwargs_edit = self.loader_options.kwargs_line
        self.kwargs_edit.setObjectName("managerProvenanceLoaderKwargsEdit")
        self.loader_label = self.loader_options.func_label
        self.loader_label.setObjectName("managerProvenanceLoaderEdit")
        layout.addWidget(self.loader_options)

        self.batch_apply_check = QtWidgets.QCheckBox(
            "Also apply to matching file loads",
            self,
        )
        self.batch_apply_check.setObjectName("managerProvenanceBatchApplyCheck")
        self.batch_apply_check.setToolTip(
            "Apply the same file-load edit to other ImageTools loaded from the "
            "same folder with the same original loader arguments."
        )
        self.batch_apply_check.setEnabled(bool(self._batch_peers))
        layout.addWidget(self.batch_apply_check)

        self.batch_note = QtWidgets.QLabel(
            "No other matching file loads found",
            self,
        )
        self.batch_note.setObjectName("managerProvenanceBatchNote")
        self.batch_note.setVisible(not self._batch_peers)
        layout.addWidget(self.batch_note)

        self.batch_peer_tree = QtWidgets.QTreeWidget(self)
        self.batch_peer_tree.setObjectName("managerProvenanceBatchPeerTree")
        self.batch_peer_tree.setColumnCount(4)
        self.batch_peer_tree.setHeaderLabels(
            ["Tool", "Original Path", "New Path", "Loader"]
        )
        self.batch_peer_tree.setRootIsDecorated(False)
        self.batch_peer_tree.setUniformRowHeights(True)
        self.batch_peer_tree.setVisible(False)
        self.batch_peer_tree.setMinimumHeight(120)
        for peer in self._batch_peers:
            item = QtWidgets.QTreeWidgetItem(
                [
                    peer.node.display_text,
                    str(peer.original_path),
                    str(self._peer_path(peer)),
                    peer.loader_summary,
                ]
            )
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, peer.node.uid)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, QtCore.Qt.CheckState.Checked)
            self.batch_peer_tree.addTopLevelItem(item)
        self.batch_peer_tree.resizeColumnToContents(0)
        self.batch_peer_tree.resizeColumnToContents(1)
        layout.addWidget(self.batch_peer_tree)
        self.batch_apply_check.toggled.connect(self.batch_peer_tree.setVisible)
        self.path_edit.textChanged.connect(self._update_batch_peer_paths)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        self.button_box.setObjectName("managerProvenanceFileEditButtonBox")
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def file_path(self) -> pathlib.Path:
        return pathlib.Path(self.path_edit.text()).expanduser()

    def selected_batch_peers(self) -> tuple[_FileLoadBatchPeer, ...]:
        if not self.batch_apply_check.isChecked():
            return ()
        peers: list[_FileLoadBatchPeer] = []
        for row in range(self.batch_peer_tree.topLevelItemCount()):
            item = self.batch_peer_tree.topLevelItem(row)
            if item is None:
                continue
            if item.checkState(0) == QtCore.Qt.CheckState.Checked:
                uid = typing.cast("str", item.data(0, QtCore.Qt.ItemDataRole.UserRole))
                if uid in self._batch_peers_by_uid:
                    peers.append(self._batch_peers_by_uid[uid])
        return tuple(peers)

    def provenance_spec(
        self,
        *,
        active_name: str,
        replay_stages: tuple[provenance.ReplayStage, ...],
    ) -> provenance.ToolProvenanceSpec:
        return self._provenance_spec_for(
            path=self.file_path(),
            selection=self._selection,
            active_name=active_name,
            replay_stages=replay_stages,
        )

    def peer_provenance_spec(
        self,
        peer: _FileLoadBatchPeer,
    ) -> provenance.ToolProvenanceSpec:
        load_source = peer.spec.file_load_source
        replay_call = None if load_source is None else load_source.replay_call
        if replay_call is None:
            raise RuntimeError("Matching file load is no longer replayable")
        return self._provenance_spec_for(
            path=self._peer_path(peer),
            selection=replay_call.selection,
            active_name=peer.spec.active_name or "derived",
            replay_stages=peer.spec.replay_stages,
        )

    def _provenance_spec_for(
        self,
        *,
        path: pathlib.Path,
        selection: provenance.FileDataSelection,
        active_name: str,
        replay_stages: tuple[provenance.ReplayStage, ...],
    ) -> provenance.ToolProvenanceSpec:
        _filter_name, func, kwargs = self.loader_options.checked_filter()
        spec = _load_provenance_from_file_details(
            path,
            (func, kwargs, selection),
        )
        if spec is None:
            raise RuntimeError("Selected loader cannot be replayed")
        return spec.model_copy(
            update={
                "active_name": active_name,
                "replay_stages": replay_stages,
            }
        )

    def _peer_path(self, peer: _FileLoadBatchPeer) -> pathlib.Path:
        path = self.file_path()
        if _normalized_path(path) == _normalized_path(self._original_path):
            return peer.original_path
        return path.parent / f"{peer.original_path.stem}{path.suffix}"

    @QtCore.Slot()
    def _update_batch_peer_paths(self) -> None:
        for row in range(self.batch_peer_tree.topLevelItemCount()):
            item = self.batch_peer_tree.topLevelItem(row)
            if item is None:
                continue
            uid = typing.cast("str", item.data(0, QtCore.Qt.ItemDataRole.UserRole))
            if uid in self._batch_peers_by_uid:
                item.setText(2, str(self._peer_path(self._batch_peers_by_uid[uid])))

    @QtCore.Slot()
    def _browse(self) -> None:
        path, _selected_filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose File",
            str(self.file_path().parent),
        )
        if path:
            self.path_edit.setText(path)

    @QtCore.Slot()
    def accept(self) -> None:
        if not self.loader_options.validate_checked_values():
            return
        super().accept()


def _iter_dialog_classes(
    cls: type[dialogs._DataManipulationDialog],
) -> Iterator[type[dialogs._DataManipulationDialog]]:
    for subclass in cls.__subclasses__():
        yield subclass
        yield from _iter_dialog_classes(subclass)


def _dialog_class_for_operation(
    operation: provenance.ToolProvenanceOperation,
) -> type[dialogs._DataManipulationDialog] | None:
    for dialog_cls in (
        *_iter_dialog_classes(dialogs.DataTransformDialog),
        *_iter_dialog_classes(dialogs.DataFilterDialog),
    ):
        if not any(
            isinstance(operation, operation_type)
            for operation_type in dialog_cls.operation_types
        ):
            continue
        if (
            issubclass(dialog_cls, dialogs.DataTransformDialog)
            and dialog_cls.restore_transform_operation
            is dialogs.DataTransformDialog.restore_transform_operation
        ):
            continue
        if (
            issubclass(dialog_cls, dialogs.DataFilterDialog)
            and dialog_cls.restore_filter_operation
            is dialogs.DataFilterDialog.restore_filter_operation
        ):
            continue
        return dialog_cls
    return None


class _ProvenanceEditController:
    def __init__(self, manager: ImageToolManager) -> None:
        self._manager = manager

    def can_edit_row(
        self,
        row: provenance._ProvenanceDisplayRow | None,
    ) -> tuple[bool, str]:
        node = self._metadata_node()
        if row is None or row.edit_ref is None:
            return False, "This row does not support editing."
        if not self._node_editable(node):
            return False, "Select an available ImageTool row to edit provenance."
        node = typing.cast("_ImageToolWrapper | _ManagedWindowNode", node)
        if self._source_child_parent_row(node, row):
            return False, "Edit the parent ImageTool row directly."

        spec = self._display_spec_for_row(node, row)
        if spec is None:
            return False, "This row does not have replayable provenance."
        if spec.kind == "script":
            return False, "Script provenance rows are not editable in this version."
        if row.edit_ref.kind == "file_load":
            if spec.kind != "file" or spec.file_load_source is None:
                return False, "This row is not a file load step."
            if spec.file_load_source.replay_call is None:
                return False, "This file load step cannot be replayed."
            return True, ""
        operation = spec._operation_for_ref(row.edit_ref)
        if operation is None:
            return False, "This operation is not available."
        if isinstance(operation, provenance.ScriptCodeOperation):
            return False, "Free-form script steps are not editable."
        if (
            spec.kind in {"full_data", "public_data", "selection"}
            and row.scope != "source"
            and self._active_filter_ref(node, spec) != row.edit_ref
        ):
            return False, "This live row needs a parent source to replay."
        if _dialog_class_for_operation(operation) is None:
            return False, "No editing dialog is available for this step."
        return True, ""

    def can_revert_row(
        self,
        row: provenance._ProvenanceDisplayRow | None,
    ) -> tuple[bool, str]:
        node = self._metadata_node()
        if row is None or row.replay_ref is None:
            return False, "This row is not replayable."
        if not self._node_editable(node):
            return False, "Select an available ImageTool row to revert provenance."
        node = typing.cast("_ImageToolWrapper | _ManagedWindowNode", node)
        if self._source_child_parent_row(node, row):
            return False, "Revert the parent ImageTool row directly."
        if row.replay_ref.kind == "script_input":
            return False, "Dependency input rows are not revert targets."
        spec = self._display_spec_for_row(node, row)
        if spec is None:
            return False, "This row does not have replayable provenance."
        if spec.kind == "script":
            try:
                candidate = spec._prefix_through_ref(row.replay_ref)
            except ValueError:
                return False, "This script row is not a replayable step."
            if not provenance.script_provenance_replayable(candidate):
                return False, "This script step cannot be replayed automatically."
            if not all(
                self._manager._script_input_can_reload(
                    script_input,
                    target_node_uid=node.uid,
                )
                for script_input in candidate.script_inputs
            ):
                return False, "This script step has unavailable inputs."
            return True, ""
        if spec.kind in {"full_data", "public_data", "selection"}:
            if row.scope == "source" and node.parent_uid is not None:
                return True, ""
            return False, "This live row needs a parent source to replay."
        return True, ""

    def edit_row(self, row: provenance._ProvenanceDisplayRow | None) -> None:
        editable, reason = self.can_edit_row(row)
        if not editable:
            self._show_unavailable(reason)
            return
        node = typing.cast(
            "_ImageToolWrapper | _ManagedWindowNode",
            self._metadata_node(),
        )
        row = typing.cast("provenance._ProvenanceDisplayRow", row)
        ref = typing.cast("provenance._ProvenanceStepRef", row.edit_ref)
        try:
            if ref.kind == "file_load":
                self._edit_file_load_row(node, row)
            else:
                self._edit_operation_row(node, row)
        except Exception as exc:
            if self._handle_missing_source_file(
                node,
                row,
                title="Could Not Apply Provenance Edit",
                exc=exc,
            ):
                return
            self._show_failed("Could Not Apply Provenance Edit", exc)

    def revert_row(self, row: provenance._ProvenanceDisplayRow | None) -> None:
        revertible, reason = self.can_revert_row(row)
        if not revertible:
            self._show_unavailable(reason)
            return
        if not self._confirm_revert():
            return

        node = typing.cast(
            "_ImageToolWrapper | _ManagedWindowNode",
            self._metadata_node(),
        )
        row = typing.cast("provenance._ProvenanceDisplayRow", row)
        ref = typing.cast("provenance._ProvenanceStepRef", row.replay_ref)
        spec = self._display_spec_for_row(node, row)
        if spec is None:
            self._show_failed(
                "Could Not Revert Provenance Step",
                RuntimeError("No provenance spec is available"),
            )
            return
        candidate: provenance.ToolProvenanceSpec | None = None
        try:
            candidate = spec._prefix_through_ref(ref)
            self._validate_and_replace(
                node,
                row.scope,
                candidate,
                where="validating the provenance revert target",
            )
        except Exception as exc:
            if self._handle_missing_source_file(
                node,
                row,
                title="Could Not Revert Provenance Step",
                exc=exc,
                repair_spec=candidate,
            ):
                return
            self._show_failed("Could Not Revert Provenance Step", exc)

    def _metadata_node(self) -> _ImageToolWrapper | _ManagedWindowNode | None:
        uid = self._manager._metadata_node_uid
        if uid is None:
            return None
        return self._manager._tool_graph.nodes.get(uid)

    @staticmethod
    def _node_editable(
        node: _ImageToolWrapper | _ManagedWindowNode | None,
    ) -> bool:
        return node is not None and node.is_imagetool and node.imagetool is not None

    @staticmethod
    def _source_child_parent_row(
        node: _ImageToolWrapper | _ManagedWindowNode | None,
        row: provenance._ProvenanceDisplayRow,
    ) -> bool:
        return (
            node is not None
            and node.parent_uid is not None
            and node.source_spec is not None
            and row.scope == "display"
        )

    def _display_spec_for_row(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        row: provenance._ProvenanceDisplayRow,
    ) -> provenance.ToolProvenanceSpec | None:
        if row.scope == "source":
            return node.displayed_source_spec
        return node.displayed_provenance_spec

    def _file_load_source_edit_target(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        path: pathlib.Path,
    ) -> tuple[
        typing.Literal["display", "source"] | None,
        provenance.ToolProvenanceSpec | None,
        str,
    ]:
        target_path = _normalized_path(path)
        candidates: list[
            tuple[
                typing.Literal["display", "source"],
                provenance.ToolProvenanceSpec | None,
            ]
        ] = [("display", node.displayed_provenance_spec)]
        if node.parent_uid is not None and node.source_spec is not None:
            candidates.append(("source", node.displayed_source_spec))

        for scope, spec in candidates:
            if spec is None or spec.kind != "file" or spec.file_load_source is None:
                continue
            load_source = spec.file_load_source
            if _normalized_path(pathlib.Path(load_source.path)) != target_path:
                continue
            if load_source.replay_call is None:
                return None, None, "This file load step cannot be replayed."
            return (
                scope,
                spec,
                "Select the current source file and update the recorded "
                "file-load step.",
            )
        return (
            None,
            None,
            "This source was not recorded as an editable file-load step.",
        )

    def can_edit_file_load_source(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        path: pathlib.Path,
    ) -> tuple[bool, str]:
        if not self._node_editable(node):
            return False, "This ImageTool is not available for editing."
        _scope, spec, reason = self._file_load_source_edit_target(node, path)
        return spec is not None, reason

    def edit_file_load_source(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        path: pathlib.Path,
    ) -> None:
        if not self._node_editable(node):
            self._show_unavailable("This ImageTool is not available for editing.")
            return
        scope, spec, reason = self._file_load_source_edit_target(node, path)
        if scope is None or spec is None:
            self._show_unavailable(reason)
            return
        try:
            self._edit_file_load_spec(
                node,
                scope,
                spec,
                where="validating the edited file-load provenance",
                batch_peers=self._file_load_batch_peers(node, spec),
            )
        except Exception as exc:
            if isinstance(exc, _ProvenanceReplayFailure):
                missing = exc.missing_source_file
            else:
                missing = (
                    exc if isinstance(exc, _MissingProvenanceSourceFileError) else None
                )
            if missing is not None:
                self._show_missing_source_file(
                    "Could Not Update Source File",
                    exc,
                    missing,
                    can_edit=False,
                )
                return
            self._show_failed("Could Not Update Source File", exc)

    def _edit_file_load_row(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        row: provenance._ProvenanceDisplayRow,
    ) -> None:
        spec = self._display_spec_for_row(node, row)
        if spec is None or spec.kind != "file" or spec.file_load_source is None:
            raise RuntimeError("Selected row is not a file load step")
        self._edit_file_load_spec(
            node,
            row.scope,
            spec,
            where="validating the edited file-load provenance",
            batch_peers=self._file_load_batch_peers(node, spec),
        )

    def _edit_file_load_spec(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        scope: typing.Literal["display", "source"],
        spec: provenance.ToolProvenanceSpec,
        *,
        where: str,
        batch_peers: Sequence[_FileLoadBatchPeer] = (),
    ) -> None:
        if spec.kind != "file" or spec.file_load_source is None:
            raise RuntimeError("Selected provenance does not have a file load step")
        dialog = _FileLoadEditDialog(
            spec.file_load_source,
            self._manager,
            batch_peers=batch_peers,
        )
        if dialog.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return
        candidate = dialog.provenance_spec(
            active_name=spec.active_name or "derived",
            replay_stages=spec.replay_stages,
        )
        validated_edits = [
            self._validated_edit(node, scope, candidate, where=where),
        ]
        failures: list[tuple[_FileLoadBatchPeer, Exception]] = []
        for peer in dialog.selected_batch_peers():
            try:
                peer_where = (
                    "validating the edited file-load provenance for "
                    f"{peer.node.display_text}"
                )
                validated_edits.append(
                    self._validated_edit(
                        peer.node,
                        peer.scope,
                        dialog.peer_provenance_spec(peer),
                        where=peer_where,
                    )
                )
            except Exception as exc:
                failures.append((peer, exc))
        if failures and not self._confirm_apply_valid_batch(
            valid_peer_count=len(validated_edits) - 1,
            failures=failures,
        ):
            return
        for edit in validated_edits:
            self._apply_validated_edit(edit)

    def _edit_operation_row(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        row: provenance._ProvenanceDisplayRow,
    ) -> None:
        ref = typing.cast("provenance._ProvenanceStepRef", row.edit_ref)
        spec = self._display_spec_for_row(node, row)
        if spec is None:
            raise RuntimeError("No provenance spec is available")
        operation = spec._operation_for_ref(ref)
        if operation is None:
            raise RuntimeError("Selected operation is not available")
        dialog_cls = _dialog_class_for_operation(operation)
        if dialog_cls is None:
            raise RuntimeError("No editing dialog is available for this step")
        if self._active_filter_ref(node, spec) == ref:
            self._edit_active_filter(node, operation, dialog_cls)
            return

        input_spec = spec._prefix_before_ref(ref)
        try:
            input_data = self._replay_candidate(node, row.scope, input_spec)
        except Exception as exc:
            raise _ProvenanceReplayFailure(
                "preparing data before the selected provenance step",
                exc,
            ) from exc
        replacements = self._edited_operations_from_dialog(
            dialog_cls,
            operation,
            input_data,
        )
        if replacements is None:
            return
        candidate = spec._replace_operation_ref(ref, replacements)
        self._validate_and_replace(
            node,
            row.scope,
            candidate,
            where="validating the edited provenance step",
        )

    def _edited_operations_from_dialog(
        self,
        dialog_cls: type[dialogs._DataManipulationDialog],
        operation: provenance.ToolProvenanceOperation,
        input_data: xr.DataArray,
    ) -> list[provenance.ToolProvenanceOperation] | None:
        tool = typing.cast(
            "ImageTool | None",
            erlab.interactive.itool(input_data, manager=False, execute=False),
        )
        if tool is None:
            raise RuntimeError("Could not create a temporary ImageTool")
        tool.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        tool.hide()
        try:
            dialog = dialog_cls(tool.slicer_area)
            if isinstance(dialog, dialogs.DataFilterDialog):
                dialog.restore_filter_operation(operation)
                if dialog.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
                    return None
                new_operation = dialog.filter_operation()
                return [] if new_operation is None else [new_operation]

            transform_dialog = typing.cast("dialogs.DataTransformDialog", dialog)
            transform_dialog.restore_transform_operation(operation)
            replace_index = transform_dialog.launch_mode_combo.findData(
                "replace",
                QtCore.Qt.ItemDataRole.UserRole,
            )
            if replace_index >= 0:
                transform_dialog.launch_mode_combo.setCurrentIndex(replace_index)
            if transform_dialog.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
                return None
            return transform_dialog.source_operations()
        finally:
            tool.close()
            tool.deleteLater()

    def _edit_active_filter(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        operation: provenance.ToolProvenanceOperation,
        dialog_cls: type[dialogs._DataManipulationDialog],
    ) -> None:
        if node.imagetool is None or not issubclass(
            dialog_cls,
            dialogs.DataFilterDialog,
        ):
            raise RuntimeError("Active display filter is not editable")
        dialog = dialog_cls(node.slicer_area)
        dialog.restore_filter_operation(operation)
        if dialog.exec() == int(QtWidgets.QDialog.DialogCode.Accepted):
            self._manager._update_info(uid=node.uid)

    def _validate_and_replace(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        scope: typing.Literal["display", "source"],
        candidate: provenance.ToolProvenanceSpec,
        *,
        where: str = "validating the provenance change",
    ) -> None:
        self._apply_validated_edit(
            self._validated_edit(node, scope, candidate, where=where)
        )

    def _validated_edit(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        scope: typing.Literal["display", "source"],
        candidate: provenance.ToolProvenanceSpec,
        *,
        where: str,
    ) -> _ValidatedProvenanceEdit:
        try:
            candidate_data, candidate = self._replay_candidate_result(
                node,
                scope,
                candidate,
            )
        except Exception as exc:
            raise _ProvenanceReplayFailure(
                f"{where}: replaying the requested provenance",
                exc,
            ) from exc
        base_candidate, filter_operation = self._split_active_filter(node, candidate)
        if base_candidate == candidate:
            source_data = candidate_data
        else:
            try:
                source_data, base_candidate = self._replay_candidate_result(
                    node,
                    scope,
                    base_candidate,
                )
            except Exception as exc:
                raise _ProvenanceReplayFailure(
                    f"{where}: replaying provenance before the active display filter",
                    exc,
                ) from exc
        return _ValidatedProvenanceEdit(
            node=node,
            scope=scope,
            data=source_data,
            spec=base_candidate,
            filter_operation=filter_operation,
        )

    def _apply_validated_edit(self, edit: _ValidatedProvenanceEdit) -> None:
        self._replace_node_data(
            edit.node,
            edit.scope,
            edit.data,
            edit.spec,
            edit.filter_operation,
        )
        self._manager._update_info(uid=edit.node.uid)

    def _file_load_batch_peers(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        spec: provenance.ToolProvenanceSpec,
    ) -> tuple[_FileLoadBatchPeer, ...]:
        load_source = spec.file_load_source
        replay_call = None if load_source is None else load_source.replay_call
        if load_source is None or replay_call is None:
            return ()
        source_folder = _normalized_path(pathlib.Path(load_source.path).parent)
        peers: list[_FileLoadBatchPeer] = []
        for peer_node in self._manager._tool_graph.nodes.values():
            if (
                peer_node.uid == node.uid
                or not peer_node.is_imagetool
                or peer_node.imagetool is None
                or (
                    peer_node.parent_uid is not None
                    and peer_node.source_spec is not None
                )
            ):
                continue
            peer_spec = peer_node.displayed_provenance_spec
            if peer_spec is None or peer_spec.kind != "file":
                continue
            peer_load_source = peer_spec.file_load_source
            peer_replay_call = (
                None if peer_load_source is None else peer_load_source.replay_call
            )
            if peer_load_source is None or peer_replay_call is None:
                continue
            if (
                _normalized_path(pathlib.Path(peer_load_source.path).parent)
                != source_folder
            ):
                continue
            if not _same_replay_loader(replay_call, peer_replay_call):
                continue
            peers.append(
                _FileLoadBatchPeer(
                    node=peer_node,
                    scope="display",
                    spec=peer_spec,
                    original_path=pathlib.Path(peer_load_source.path).expanduser(),
                    loader_summary=_loader_summary(peer_load_source),
                )
            )
        return tuple(peers)

    def _replay_candidate(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        scope: typing.Literal["display", "source"],
        spec: provenance.ToolProvenanceSpec,
    ) -> xr.DataArray:
        data, _spec = self._replay_candidate_result(node, scope, spec)
        return data

    def _replay_candidate_result(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        scope: typing.Literal["display", "source"],
        spec: provenance.ToolProvenanceSpec,
    ) -> tuple[xr.DataArray, provenance.ToolProvenanceSpec]:
        if spec.kind == "file":
            return self._replay_file_candidate(spec), spec
        if spec.kind in {"full_data", "public_data", "selection"}:
            if scope != "source" or node.parent_uid is None:
                raise RuntimeError("Live provenance needs a parent source to replay")
            parent = self._manager._parent_node(node)
            return spec.apply(parent.current_source_data()), spec
        if spec.kind == "script":
            result = self._manager._rebuild_script_provenance(
                spec,
                target_node_uid=node.uid,
            )
            return result.data, result.provenance_spec
        raise RuntimeError("Unsupported provenance kind")

    def _replay_file_candidate(
        self,
        spec: provenance.ToolProvenanceSpec,
    ) -> xr.DataArray:
        load_source = spec.file_load_source
        if load_source is not None:
            source_path = pathlib.Path(load_source.path)
            if not source_path.exists():
                raise _MissingProvenanceSourceFileError(source_path)
        with warnings.catch_warnings(record=True) as replay_warnings:
            warnings.simplefilter("always")
            try:
                return provenance.replay_file_provenance(spec)
            except Exception as exc:
                if warning_details := _replay_warning_details(replay_warnings):
                    exc.add_note(
                        "Warnings emitted while replaying provenance:\n"
                        f"{warning_details}"
                    )
                raise

    def _replace_node_data(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        scope: typing.Literal["display", "source"],
        data: xr.DataArray,
        spec: provenance.ToolProvenanceSpec,
        filter_operation: provenance.ToolProvenanceOperation | None,
    ) -> None:
        preserve_filter = filter_operation is not None
        if scope == "source":
            live_spec = provenance.require_live_source_spec(spec)
            if live_spec is None or node.parent_uid is None:
                raise RuntimeError("Source-bound edits need live source provenance")
            parent = self._manager._parent_node(node)
            parent_data = parent.current_source_data()
            displayed = provenance.compose_display_provenance(
                parent.displayed_provenance_spec,
                live_spec,
                parent_data=parent_data,
            )
            node.set_source_binding(
                live_spec,
                auto_update=node.source_auto_update,
                state="fresh",
                provenance_spec=displayed,
            )
            node._replace_imagetool_data(
                data,
                displayed,
                propagate_descendants=True,
                preserve_filter=preserve_filter,
            )
            return
        node.replace_with_detached_data(
            data,
            spec,
            preserve_filter=preserve_filter,
        )

    def _active_filter_ref(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        spec: provenance.ToolProvenanceSpec,
    ) -> provenance._ProvenanceStepRef | None:
        if node.imagetool is None:
            return None
        active = node.slicer_area._accepted_filter_provenance_operation
        if active is None:
            return None
        if spec.kind == "file":
            for stage_index in range(len(spec.replay_stages) - 1, -1, -1):
                operations = spec.replay_stages[stage_index].operations
                for operation_index in range(len(operations) - 1, -1, -1):
                    if operations[operation_index] == active:
                        return provenance._ProvenanceStepRef(
                            "operation",
                            operation_index=operation_index,
                            stage_index=stage_index,
                        )
        elif spec.kind in {"full_data", "public_data", "selection", "script"}:
            for operation_index in range(len(spec.operations) - 1, -1, -1):
                if spec.operations[operation_index] == active:
                    return provenance._ProvenanceStepRef(
                        "operation",
                        operation_index=operation_index,
                    )
        return None

    def _split_active_filter(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        spec: provenance.ToolProvenanceSpec,
    ) -> tuple[
        provenance.ToolProvenanceSpec,
        provenance.ToolProvenanceOperation | None,
    ]:
        active_ref = self._active_filter_ref(node, spec)
        if active_ref is None:
            return spec, None
        active_operation = spec._operation_for_ref(active_ref)
        if active_operation is None:
            return spec, None
        base_spec = spec._replace_operation_ref(active_ref, ())
        if base_spec.kind == "file":
            stages = tuple(
                stage for stage in base_spec.replay_stages if stage.operations
            )
            base_spec = base_spec.model_copy(update={"replay_stages": stages})
        return base_spec, active_operation

    def _confirm_revert(self) -> bool:
        msg_box = QtWidgets.QMessageBox(self._manager)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setWindowTitle("Revert Provenance")
        msg_box.setText("Revert to the selected provenance step?")
        msg_box.setInformativeText(
            "Later provenance steps for this ImageTool will be dropped."
        )
        msg_box.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.Cancel
        )
        msg_box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        return msg_box.exec() == QtWidgets.QMessageBox.StandardButton.Yes

    def _show_unavailable(self, reason: str) -> None:
        QtWidgets.QMessageBox.information(
            self._manager,
            "Provenance Step Unavailable",
            reason,
        )

    def _show_failed(self, title: str, exc: Exception) -> None:
        exc_text = "".join(traceback.TracebackException.from_exception(exc).format())
        where = (
            exc.where
            if isinstance(exc, _ProvenanceReplayFailure)
            else "replaying the requested provenance"
        )
        dialog = erlab.interactive.utils.MessageDialog(
            self._manager,
            title=title,
            text="The provenance change could not be applied.",
            informative_text=(
                f"Failed while: {where}\n\n"
                "The current ImageTool data was left unchanged because the requested "
                "provenance could not be replayed. Use Revert to This Step to drop "
                "later provenance, or adjust the earlier steps so the full chain is "
                "valid again."
            ),
            detailed_text=erlab.interactive.utils._format_traceback(exc_text),
            buttons=QtWidgets.QDialogButtonBox.StandardButton.Ok,
            icon_pixmap=QtWidgets.QStyle.StandardPixmap.SP_MessageBoxWarning,
        )
        dialog.exec()

    def _handle_missing_source_file(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        row: provenance._ProvenanceDisplayRow,
        *,
        title: str,
        exc: Exception,
        repair_spec: provenance.ToolProvenanceSpec | None = None,
    ) -> bool:
        if isinstance(exc, _ProvenanceReplayFailure):
            missing = exc.missing_source_file
        else:
            missing = (
                exc if isinstance(exc, _MissingProvenanceSourceFileError) else None
            )
        if missing is None:
            return False
        spec = repair_spec
        if spec is None or spec.kind != "file" or spec.file_load_source is None:
            spec = self._display_spec_for_row(node, row)
        if spec is None or spec.kind != "file" or spec.file_load_source is None:
            self._show_missing_source_file(title, exc, missing, can_edit=False)
            return True
        if self._show_missing_source_file(title, exc, missing, can_edit=True):
            try:
                self._edit_file_load_spec(
                    node,
                    row.scope,
                    spec,
                    where=(
                        "validating provenance after selecting a replacement "
                        "source file"
                    ),
                    batch_peers=self._file_load_batch_peers(node, spec),
                )
            except Exception as repair_exc:
                if isinstance(repair_exc, _ProvenanceReplayFailure):
                    repair_missing = repair_exc.missing_source_file
                else:
                    repair_missing = (
                        repair_exc
                        if isinstance(
                            repair_exc,
                            _MissingProvenanceSourceFileError,
                        )
                        else None
                    )
                if repair_missing is None:
                    self._show_failed("Could Not Update Source File", repair_exc)
                else:
                    self._show_missing_source_file(
                        "Could Not Update Source File",
                        repair_exc,
                        repair_missing,
                        can_edit=True,
                    )
        return True

    def _show_missing_source_file(
        self,
        title: str,
        exc: Exception,
        missing: _MissingProvenanceSourceFileError,
        *,
        can_edit: bool,
    ) -> bool:
        where = (
            exc.where
            if isinstance(exc, _ProvenanceReplayFailure)
            else "replaying the requested provenance"
        )
        exc_text = "".join(traceback.TracebackException.from_exception(exc).format())
        buttons = QtWidgets.QDialogButtonBox.StandardButton.Ok
        if can_edit:
            buttons = (
                QtWidgets.QDialogButtonBox.StandardButton.Yes
                | QtWidgets.QDialogButtonBox.StandardButton.Cancel
            )
        dialog = erlab.interactive.utils.MessageDialog(
            self._manager,
            title=title,
            text="The recorded source file is no longer accessible.",
            informative_text=(
                f"Failed while: {where}\n\n"
                f"Missing file: {missing.source_path}\n\n"
                "Select the current location of the file to update the file-load "
                "step and replay the provenance again."
            ),
            detailed_text=erlab.interactive.utils._format_traceback(exc_text),
            buttons=buttons,
            default_button=(
                QtWidgets.QDialogButtonBox.StandardButton.Yes
                if can_edit
                else QtWidgets.QDialogButtonBox.StandardButton.Ok
            ),
            icon_pixmap=QtWidgets.QStyle.StandardPixmap.SP_MessageBoxWarning,
        )
        if can_edit:
            edit_button = dialog._button_box.button(
                QtWidgets.QDialogButtonBox.StandardButton.Yes
            )
            if edit_button is not None:
                edit_button.setText("Edit File Load…")
        result = dialog.exec()
        return can_edit and result == int(QtWidgets.QDialog.DialogCode.Accepted)

    def _confirm_apply_valid_batch(
        self,
        *,
        valid_peer_count: int,
        failures: Sequence[tuple[_FileLoadBatchPeer, Exception]],
    ) -> bool:
        details: list[str] = []
        for peer, exc in failures:
            details.append(f"{peer.node.display_text}\n{peer.original_path}")
            details.append(
                "".join(traceback.TracebackException.from_exception(exc).format())
            )
        valid_peer_text = (
            f" and {valid_peer_count} matching ImageTool(s)" if valid_peer_count else ""
        )
        dialog = erlab.interactive.utils.MessageDialog(
            self._manager,
            title="Some File Load Edits Failed",
            text="Some matching ImageTools could not be updated.",
            informative_text=(
                f"The current ImageTool{valid_peer_text} "
                "can be updated. Failed ImageTools will be left unchanged."
            ),
            detailed_text=erlab.interactive.utils._format_traceback(
                "\n\n".join(details)
            ),
            buttons=(
                QtWidgets.QDialogButtonBox.StandardButton.Yes
                | QtWidgets.QDialogButtonBox.StandardButton.Cancel
            ),
            default_button=QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            icon_pixmap=QtWidgets.QStyle.StandardPixmap.SP_MessageBoxWarning,
        )
        apply_button = dialog._button_box.button(
            QtWidgets.QDialogButtonBox.StandardButton.Yes
        )
        if apply_button is not None:
            apply_button.setText("Apply Valid Tools")
        return dialog.exec() == int(QtWidgets.QDialog.DialogCode.Accepted)


def _replay_warning_details(
    replay_warnings: list[warnings.WarningMessage],
) -> str:
    lines: list[str] = []
    for warning in replay_warnings:
        category_name = warning.category.__name__
        message = str(warning.message).strip()
        if not message:
            continue
        indented_message = "\n  ".join(message.splitlines())
        lines.append(f"- {category_name}: {indented_message}")
    return "\n".join(lines)


def _normalized_path(path: pathlib.Path) -> pathlib.Path:
    try:
        return path.expanduser().resolve(strict=False)
    except (OSError, RuntimeError):
        return path.expanduser().absolute()


def _same_replay_loader(
    left: provenance.FileReplayCall,
    right: provenance.FileReplayCall,
) -> bool:
    return (
        left.kind == right.kind
        and left.target == right.target
        and provenance.encode_provenance_value(left.kwargs)
        == provenance.encode_provenance_value(right.kwargs)
    )


def _loader_summary(load_source: provenance.FileLoadSource) -> str:
    if not load_source.kwargs_text or load_source.kwargs_text == "(none)":
        return load_source.loader_text
    return f"{load_source.loader_text} ({load_source.kwargs_text})"
