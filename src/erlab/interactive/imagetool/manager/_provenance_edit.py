from __future__ import annotations

import ast
import dataclasses
import pathlib
import traceback
import typing
import warnings

from qtpy import QtCore, QtWidgets

import erlab
import erlab.interactive.imagetool.slicer
import erlab.interactive.utils
from erlab.interactive.imagetool import _kspace_conversion, dialogs, provenance
from erlab.interactive.imagetool._load_source import (
    _load_provenance_from_file_details,
    _loader_callable_text,
)
from erlab.interactive.imagetool.manager._dialogs import _LoaderOptionsWidget
from erlab.interactive.imagetool.manager._widgets import _TrustedScriptReplayCancelled

if typing.TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    import xarray as xr

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
    script_input_path: tuple[int, ...] = ()
    display_label: str | None = None
    preserve_loader: bool = False

    @property
    def target_id(self) -> str:
        path = ".".join(str(index) for index in self.script_input_path)
        return f"{self.node.uid}\x1f{self.scope}\x1f{path}"

    @property
    def display_text(self) -> str:
        return self.display_label or self.node.display_text


@dataclasses.dataclass(frozen=True)
class _ValidatedProvenanceEdit:
    node: _ImageToolWrapper | _ManagedWindowNode
    scope: typing.Literal["display", "source"]
    data: xr.DataArray
    spec: provenance.ToolProvenanceSpec
    filter_operation: provenance.ToolProvenanceOperation | None


@dataclasses.dataclass(frozen=True)
class _OperationDialogMatch:
    dialog_cls: type[dialogs._DataManipulationDialog]
    start: int
    stop: int
    focus: str | None = None


class _ProvenanceReplayFailure(RuntimeError):
    def __init__(self, where: str, cause: Exception) -> None:
        super().__init__(f"{where}: {cause}")
        self.where = where
        self.cause = cause
        self.__cause__ = cause

    @property
    def missing_source_file(self) -> _MissingProvenanceSourceFileError | None:
        return _missing_source_file_error_from_exception(self.cause)


class _MissingProvenanceSourceFileError(FileNotFoundError):
    def __init__(self, source_path: pathlib.Path) -> None:
        super().__init__(f"Recorded source file is no longer accessible: {source_path}")
        self.source_path = source_path


def _missing_source_file_error_from_exception(
    exc: BaseException,
) -> _MissingProvenanceSourceFileError | None:
    seen: set[int] = set()

    def visit(
        current: BaseException | None,
    ) -> _MissingProvenanceSourceFileError | None:
        if current is None or id(current) in seen:
            return None
        seen.add(id(current))
        if isinstance(current, _MissingProvenanceSourceFileError):
            return current
        for nested in (
            getattr(current, "cause", None),
            current.__cause__,
            current.__context__,
        ):
            if not isinstance(nested, BaseException):
                continue
            if missing := visit(nested):
                return missing
        return None

    return visit(exc)


def _file_not_found_path_from_exception(exc: BaseException) -> pathlib.Path | None:
    seen: set[int] = set()

    def visit(current: BaseException | None) -> pathlib.Path | None:
        if current is None or id(current) in seen:
            return None
        seen.add(id(current))
        if (
            isinstance(current, FileNotFoundError)
            and not isinstance(current, _MissingProvenanceSourceFileError)
            and current.filename is not None
        ):
            try:
                return pathlib.Path(current.filename).expanduser()
            except TypeError:
                return None
        for nested in (
            getattr(current, "cause", None),
            current.__cause__,
            current.__context__,
        ):
            if not isinstance(nested, BaseException):
                continue
            if path := visit(nested):
                return path
        return None

    return visit(exc)


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
        batch_apply_default: bool = False,
        checked_batch_peer_ids: frozenset[str] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("managerProvenanceFileLoadEditDialog")
        self.setWindowTitle("Edit File Load")
        self.setModal(True)
        self._original_path = pathlib.Path(load_source.path).expanduser()
        self._batch_peers = tuple(batch_peers)
        self._batch_peers_by_target_id = {
            peer.target_id: peer for peer in self._batch_peers
        }
        self._batch_peer_paths = {
            peer.target_id: peer.original_path for peer in self._batch_peers
        }
        self._manual_batch_peer_paths: set[str] = set()
        self._updating_batch_peer_paths = False
        self._replay_call = load_source.replay_call
        self._selection = (
            provenance.FileDataSelection(kind="dataarray")
            if self._replay_call is None
            else self._replay_call.selection
        )
        self._initial_kwargs = (
            _parse_loader_kwargs(load_source.kwargs_text)
            if self._replay_call is None
            else dict(self._replay_call.kwargs)
        )
        loader_extensions = self._initial_kwargs.pop("loader_extensions", None)
        self._initial_loader_extensions = (
            dict(loader_extensions) if isinstance(loader_extensions, dict) else None
        )

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

        self.loader_options = self._make_loader_options(self.file_path())
        self._sync_loader_option_aliases()
        layout.addWidget(self.loader_options)

        self.batch_apply_check = QtWidgets.QCheckBox(
            "Also relink selected file loads",
            self,
        )
        self.batch_apply_check.setObjectName("managerProvenanceBatchApplyCheck")
        self.batch_apply_check.setToolTip(
            "Relink the checked file-load rows in the same update. Matching "
            "top-level ImageTools remain optional."
        )
        self.batch_apply_check.setEnabled(bool(self._batch_peers))
        self.batch_apply_check.setChecked(
            batch_apply_default and bool(self._batch_peers)
        )
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
        self.batch_peer_tree.setVisible(self.batch_apply_check.isChecked())
        self.batch_peer_tree.setMinimumHeight(120)
        for peer in self._batch_peers:
            item = QtWidgets.QTreeWidgetItem(
                [
                    peer.display_text,
                    str(peer.original_path),
                    str(self._peer_path(peer)),
                    peer.loader_summary,
                ]
            )
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, peer.target_id)
            item.setFlags(
                item.flags()
                | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                | QtCore.Qt.ItemFlag.ItemIsEditable
            )
            checked = (
                checked_batch_peer_ids is None
                or peer.target_id in checked_batch_peer_ids
            )
            item.setCheckState(
                0,
                (
                    QtCore.Qt.CheckState.Checked
                    if checked
                    else QtCore.Qt.CheckState.Unchecked
                ),
            )
            self.batch_peer_tree.addTopLevelItem(item)
        self.batch_peer_tree.resizeColumnToContents(0)
        self.batch_peer_tree.resizeColumnToContents(1)
        layout.addWidget(self.batch_peer_tree)
        self.batch_apply_check.toggled.connect(self.batch_peer_tree.setVisible)
        self.batch_peer_tree.itemChanged.connect(self._batch_peer_item_changed)
        self.path_edit.textChanged.connect(self._path_changed)

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

    def _make_loader_options(
        self,
        path: pathlib.Path,
        *,
        preferred_filter: str | None = None,
    ) -> _LoaderOptionsWidget:
        valid_loaders = erlab.interactive.utils.file_loaders(path)
        if not valid_loaders:
            valid_loaders = erlab.interactive.utils.file_loaders()

        selected_filter = (
            preferred_filter if preferred_filter in valid_loaders else None
        )
        if selected_filter is None and self._replay_call is not None:
            for name_filter, (func, kwargs) in valid_loaders.items():
                if self._replay_call.kind == "erlab_loader":
                    loader = getattr(func, "__self__", None)
                    matches = (
                        isinstance(loader, erlab.io.dataloader.LoaderBase)
                        and loader.name == self._replay_call.target
                    )
                else:
                    matches = _loader_callable_text(
                        func
                    ) == self._replay_call.target and all(
                        self._initial_kwargs.get(key) == value
                        for key, value in kwargs.items()
                    )
                if matches:
                    selected_filter = name_filter
                    break
        selected_filter = selected_filter or next(iter(valid_loaders), None)
        if selected_filter is not None:
            func, kwargs = valid_loaders[selected_filter]
            valid_loaders = dict(valid_loaders)
            valid_loaders[selected_filter] = (
                func,
                dict(kwargs) | self._initial_kwargs,
            )

        loader_options = _LoaderOptionsWidget(
            self,
            valid_loaders,
            loader_extensions=(
                {selected_filter: self._initial_loader_extensions}
                if selected_filter is not None
                and self._initial_loader_extensions is not None
                else None
            ),
            sample_paths=(path,),
        )
        loader_options.setObjectName("managerProvenanceLoaderOptionsWidget")
        loader_options.check_filter(selected_filter)
        return loader_options

    def _sync_loader_option_aliases(self) -> None:
        self.kwargs_edit = self.loader_options.kwargs_line
        self.kwargs_edit.setObjectName("managerProvenanceLoaderKwargsEdit")
        self.loader_label = self.loader_options.func_label
        self.loader_label.setObjectName("managerProvenanceLoaderEdit")

    def _checked_filter_name(self) -> str | None:
        checked_id = self.loader_options._button_group.checkedId()
        filters = tuple(self.loader_options._valid_loaders)
        if 0 <= checked_id < len(filters):
            return filters[checked_id]
        return None

    def _update_loader_options_for_path(self) -> None:
        path = self.file_path()
        current_filter = self._checked_filter_name()
        valid_loaders = erlab.interactive.utils.file_loaders(path)
        if not valid_loaders:
            valid_loaders = erlab.interactive.utils.file_loaders()
        if tuple(valid_loaders) == tuple(self.loader_options._valid_loaders):
            self.loader_options._sample_paths = (path,)
            return

        current_kwargs_text = self.kwargs_edit.text()
        current_extensions_text = {
            key: line.text()
            for key, line in self.loader_options.loader_extension_lines.items()
        }
        loader_options = self._make_loader_options(
            path, preferred_filter=current_filter
        )

        layout = typing.cast("QtWidgets.QVBoxLayout", self.layout())
        layout.replaceWidget(self.loader_options, loader_options)
        old_options = self.loader_options
        self.loader_options = loader_options
        self._sync_loader_option_aliases()
        old_options.deleteLater()

        selected_filter = self._checked_filter_name()
        if selected_filter == current_filter:
            self.kwargs_edit.setText(current_kwargs_text)
            for key, text in current_extensions_text.items():
                line = self.loader_options.loader_extension_lines.get(key)
                if line is not None:
                    line.setText(text)
        self.updateGeometry()

    def selected_batch_peers(self) -> tuple[_FileLoadBatchPeer, ...]:
        if not self.batch_apply_check.isChecked():
            return ()
        peers: list[_FileLoadBatchPeer] = []
        for row in range(self.batch_peer_tree.topLevelItemCount()):
            item = self.batch_peer_tree.topLevelItem(row)
            if item is None:
                continue
            if item.checkState(0) == QtCore.Qt.CheckState.Checked:
                target_id = typing.cast(
                    "str", item.data(0, QtCore.Qt.ItemDataRole.UserRole)
                )
                if peer := self._batch_peers_by_target_id.get(target_id):
                    peers.append(peer)
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
        if peer.preserve_loader:
            return _relinked_file_load_spec(peer.spec, self._peer_path(peer))
        return self._provenance_spec_for(
            path=self._peer_path(peer),
            selection=replay_call.selection,
            active_name=_file_load_edit_active_name(peer.spec),
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

    def _auto_peer_path(self, peer: _FileLoadBatchPeer) -> pathlib.Path:
        path = self.file_path()
        if _normalized_path(path) == _normalized_path(self._original_path):
            return peer.original_path
        return path.parent / peer.original_path.name

    def _peer_path(self, peer: _FileLoadBatchPeer) -> pathlib.Path:
        return self._batch_peer_paths.get(peer.target_id, self._auto_peer_path(peer))

    @QtCore.Slot(str)
    def _path_changed(self, _text: str) -> None:
        self._update_batch_peer_paths()
        self._update_loader_options_for_path()

    @QtCore.Slot()
    def _update_batch_peer_paths(self) -> None:
        self._updating_batch_peer_paths = True
        try:
            for row in range(self.batch_peer_tree.topLevelItemCount()):
                item = self.batch_peer_tree.topLevelItem(row)
                if item is None:
                    continue
                target_id = typing.cast(
                    "str", item.data(0, QtCore.Qt.ItemDataRole.UserRole)
                )
                if target_id not in self._batch_peers_by_target_id:
                    continue
                if target_id not in self._manual_batch_peer_paths:
                    peer = self._batch_peers_by_target_id[target_id]
                    self._batch_peer_paths[target_id] = self._auto_peer_path(peer)
                item.setText(2, str(self._batch_peer_paths[target_id]))
        finally:
            self._updating_batch_peer_paths = False

    @QtCore.Slot(QtWidgets.QTreeWidgetItem, int)
    def _batch_peer_item_changed(
        self, item: QtWidgets.QTreeWidgetItem, column: int
    ) -> None:
        if column != 2 or self._updating_batch_peer_paths:
            return
        target_id = typing.cast("str", item.data(0, QtCore.Qt.ItemDataRole.UserRole))
        if target_id not in self._batch_peers_by_target_id:
            return
        self._manual_batch_peer_paths.add(target_id)
        self._batch_peer_paths[target_id] = pathlib.Path(item.text(2)).expanduser()

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


def _operation_field_names(
    operation: provenance.ToolProvenanceOperation,
) -> tuple[str, ...]:
    return tuple(
        name for name in type(operation).model_fields if name not in {"op", "group"}
    )


def _operation_value_text(value: typing.Any) -> str:
    return repr(value)


def _parse_operation_value(text: str) -> typing.Any:
    text = text.strip()
    if not text:
        raise ValueError("Operation values must be Python literals.")
    return ast.literal_eval(text)


def _operation_from_field_text(
    operation: provenance.ToolProvenanceOperation,
    field_text: dict[str, str],
) -> provenance.ToolProvenanceOperation:
    payload = operation.model_dump(mode="python")
    for field_name, text in field_text.items():
        payload[field_name] = _parse_operation_value(text)
    return type(operation).model_validate(payload)


class _RecordedOperationEditDialog(QtWidgets.QDialog):
    def __init__(
        self,
        operations: Sequence[provenance.ToolProvenanceOperation],
        parent: QtWidgets.QWidget,
        *,
        focus: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("managerProvenanceRecordedOperationEditDialog")
        self.setWindowTitle("Edit Provenance Operation")
        self.setModal(True)
        self._operations = tuple(operations)
        self._field_edits: dict[tuple[int, str], QtWidgets.QLineEdit] = {}

        layout = QtWidgets.QVBoxLayout(self)
        for index, operation in enumerate(self._operations):
            group = QtWidgets.QGroupBox(_operation_title(operation), self)
            group.setObjectName(f"managerProvenanceOperationGroup{index}")
            form = QtWidgets.QFormLayout(group)
            form.setFieldGrowthPolicy(
                QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
            )
            field_names = _operation_field_names(operation)
            if not field_names:
                form.addRow(QtWidgets.QLabel("This operation has no editable values."))
            for field_name in field_names:
                edit = QtWidgets.QLineEdit(
                    _operation_value_text(getattr(operation, field_name)),
                    group,
                )
                edit.setObjectName(
                    f"managerProvenanceOperationField_{index}_{field_name}"
                )
                edit.setToolTip("Enter a Python literal value.")
                self._field_edits[(index, field_name)] = edit
                form.addRow(field_name, edit)
            layout.addWidget(group)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        self.button_box.setObjectName("managerProvenanceOperationEditButtonBox")
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)
        if focus is not None:
            self._focus_field(focus)

    def _focus_field(self, focus: str) -> None:
        for (_index, field_name), edit in self._field_edits.items():
            if field_name == focus:
                edit.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
                edit.selectAll()
                return

    def edited_operations(self) -> list[provenance.ToolProvenanceOperation]:
        edited: list[provenance.ToolProvenanceOperation] = []
        for index, operation in enumerate(self._operations):
            edited.append(
                _operation_from_field_text(
                    operation,
                    {
                        field_name: self._field_edits[(index, field_name)].text()
                        for field_name in _operation_field_names(operation)
                    },
                )
            )
        return edited

    @QtCore.Slot()
    def accept(self) -> None:
        try:
            self.edited_operations()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Operation Value",
                str(exc),
            )
            return
        super().accept()


class _ScriptCodeEditDialog(QtWidgets.QDialog):
    def __init__(
        self,
        operation: provenance.ScriptCodeOperation,
        parent: QtWidgets.QWidget,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("managerProvenanceScriptCodeEditDialog")
        self.setWindowTitle("Edit Python Code")
        self.setModal(True)
        layout = QtWidgets.QVBoxLayout(self)

        self.code_edit = erlab.interactive.utils.PythonCodeEditor(self)
        self.code_edit.setObjectName("managerProvenanceScriptCodeEditor")
        self.code_edit.setPlainText(operation.code or "")
        self.code_edit.setPlaceholderText("# Write Python code here")
        self.code_edit.setMinimumHeight(260)
        self.code_edit.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.NoWrap)
        self.code_edit.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.code_edit.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.code_edit.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        layout.addWidget(self.code_edit)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        self.button_box.setObjectName("managerProvenanceScriptCodeEditButtonBox")
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def code(self) -> str:
        return self.code_edit.toPlainText()

    @QtCore.Slot()
    def accept(self) -> None:
        try:
            ast.parse(self.code(), mode="exec")
        except SyntaxError as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Python Code",
                f"Python code is not valid: {exc}",
            )
            return
        super().accept()


def _operation_title(operation: provenance.ToolProvenanceOperation) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return operation.derivation_label()
        except Exception:
            return type(operation).__name__


def _recorded_operation_edit_unavailable_reason(
    operations: Sequence[provenance.ToolProvenanceOperation],
) -> str | None:
    for operation in operations:
        try:
            _operation_from_field_text(
                operation,
                {
                    field_name: _operation_value_text(getattr(operation, field_name))
                    for field_name in _operation_field_names(operation)
                },
            )
        except Exception as exc:
            return (
                f"{_operation_title(operation)} cannot be edited safely from "
                f"recorded metadata: {exc}"
            )
    return None


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
            issubclass(dialog_cls, dialogs.DataTransformDialog)
            and dialog_cls.grouped_operation_only
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


def _operations_for_ref(
    spec: provenance.ToolProvenanceSpec,
    ref: provenance._ProvenanceStepRef,
) -> tuple[provenance.ToolProvenanceOperation, ...]:
    if ref.kind != "operation" or ref.operation_index is None:
        return ()
    if ref.stage_index is None:
        return getattr(spec, "operations", ())
    replay_stages = getattr(spec, "replay_stages", ())
    if 0 <= ref.stage_index < len(replay_stages):
        return replay_stages[ref.stage_index].operations
    return ()


def _dialog_match_for_operation_ref(
    spec: provenance.ToolProvenanceSpec,
    ref: provenance._ProvenanceStepRef,
) -> _OperationDialogMatch | None:
    operation = spec._operation_for_ref(ref)
    if operation is None or ref.operation_index is None:
        return None
    operations = _operations_for_ref(spec, ref)
    if not operations:
        return None

    for dialog_base_cls in _iter_dialog_classes(dialogs.DataTransformDialog):
        dialog_cls = typing.cast("type[dialogs.DataTransformDialog]", dialog_base_cls)
        if not any(
            isinstance(operation, operation_type)
            for operation_type in dialog_cls.operation_types
        ):
            continue
        if (
            dialog_cls.restore_transform_operation
            is dialogs.DataTransformDialog.restore_transform_operation
            and dialog_cls.restore_transform_operations
            is dialogs.DataTransformDialog.restore_transform_operations
        ):
            continue
        group_kind = dialog_cls.operation_group_kind
        if group_kind is not None:
            marker = operation.group
            if marker is None or marker.kind != group_kind:
                continue
            group = dialog_cls.operation_group_for_edit(operations, ref.operation_index)
            if group is not None:
                return _OperationDialogMatch(
                    dialog_cls,
                    group[0],
                    group[1],
                    marker.focus,
                )
            continue
        group = dialog_cls.operation_group_for_edit(operations, ref.operation_index)
        if group is not None:
            return _OperationDialogMatch(dialog_cls, group[0], group[1])
        if dialog_cls.grouped_operation_only:
            continue
        return _OperationDialogMatch(
            dialog_cls,
            ref.operation_index,
            ref.operation_index + 1,
        )

    for dialog_base_cls in _iter_dialog_classes(dialogs.DataFilterDialog):
        dialog_cls = typing.cast("type[dialogs.DataFilterDialog]", dialog_base_cls)
        if not any(
            isinstance(operation, operation_type)
            for operation_type in dialog_cls.operation_types
        ):
            continue
        if (
            dialog_cls.restore_filter_operation
            is dialogs.DataFilterDialog.restore_filter_operation
        ):
            continue
        return _OperationDialogMatch(
            dialog_cls,
            ref.operation_index,
            ref.operation_index + 1,
        )
    return None


def _editable_group_range_for_ref(
    spec: provenance.ToolProvenanceSpec,
    ref: provenance._ProvenanceStepRef,
) -> tuple[int, int] | None:
    match = _dialog_match_for_operation_ref(spec, ref)
    if match is None or match.stop - match.start <= 1:
        return None
    return match.start, match.stop


def _uneditable_operation_reason(
    operation: provenance.ToolProvenanceOperation,
) -> str | None:
    return _kspace_conversion.incomplete_kspace_conversion_edit_reason(operation)


class _ProvenanceEditController:
    def __init__(self, manager: ImageToolManager) -> None:
        self._manager = manager

    def can_paste_steps(
        self,
        operations: Sequence[provenance.ToolProvenanceOperation] | None,
    ) -> tuple[bool, str]:
        if not operations:
            return False, "The clipboard does not contain copied ImageTool steps."
        if not self._paste_target_nodes():
            return False, "Select an available ImageTool row to paste provenance."
        return True, ""

    def paste_steps(
        self,
        operations: Sequence[provenance.ToolProvenanceOperation],
        *,
        active_name: str,
        contains_script: bool,
    ) -> None:
        paste_enabled, paste_reason = self.can_paste_steps(operations)
        if not paste_enabled:
            self._show_unavailable(paste_reason)
            return
        targets = self._paste_target_nodes()
        failures: list[tuple[_ImageToolWrapper | _ManagedWindowNode, Exception]] = []
        pasted_count = 0
        for node in targets:
            steps = provenance.restamp_operation_groups(operations)
            try:
                self._paste_steps_into_node(
                    node,
                    steps,
                    active_name=active_name,
                    contains_script=contains_script,
                )
            except _TrustedScriptReplayCancelled:
                return
            except Exception as exc:
                failures.append((node, exc))
            else:
                pasted_count += 1

        if pasted_count > 0:
            if failures:
                self._show_partial_paste_failures(pasted_count, failures)
            return

        if failures:
            self._show_failed(
                "Could Not Paste Provenance Steps",
                failures[0][1],
                text="The copied provenance steps could not be applied.",
                unchanged_reason=(
                    "The copied steps could not be replayed on the selected "
                    "ImageTool's current data, so nothing was changed. Check that "
                    "the destination data has the dimensions, coordinates, and "
                    "inputs expected by the copied steps."
                ),
            )

    def _paste_steps_into_node(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        steps: tuple[provenance.ToolProvenanceOperation, ...],
        *,
        active_name: str,
        contains_script: bool,
    ) -> None:
        if contains_script or any(not operation.live_applicable for operation in steps):
            self._paste_detached_steps(
                node,
                provenance.script(
                    *steps,
                    start_label="Start from current ImageTool data",
                    active_name=active_name or "derived",
                ),
                where="validating the pasted script provenance steps",
            )
            return
        self._paste_structured_steps(node, steps)

    def _paste_structured_steps(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        operations: tuple[provenance.ToolProvenanceOperation, ...],
    ) -> None:
        for operation in operations:
            if not operation.live_applicable:
                raise TypeError("Only live provenance operations can be pasted here")

        if node.parent_uid is not None and node.displayed_source_spec is not None:
            candidate = node.displayed_source_spec.append_replacement_operations(
                *operations
            )
            try:
                data, candidate = self._replay_candidate_result(
                    node,
                    "source",
                    candidate,
                )
                data = erlab.interactive.imagetool.slicer.ArraySlicer.validate_array(
                    data,
                    copy_values=False,
                )
            except Exception as exc:
                raise _ProvenanceReplayFailure(
                    "validating the pasted provenance steps: "
                    "replaying the requested provenance",
                    exc,
                ) from exc
            self._replace_node_data(node, "source", data, candidate, None)
            self._manager._update_info(uid=node.uid)
            return

        local = provenance.full_data(*operations)
        self._paste_detached_steps(
            node,
            local,
            where="validating the pasted provenance steps",
        )

    def _paste_detached_steps(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        local: provenance.ToolProvenanceSpec,
        *,
        where: str,
    ) -> None:
        current_data = node.current_source_data()
        try:
            if local.kind == "script":
                trusted_user_code = provenance.script_provenance_requires_trust(local)
                if trusted_user_code:
                    self._manager._ensure_script_provenance_trusted(
                        local,
                        reason="paste these provenance steps",
                    )
                data = provenance.replay_script_provenance(
                    local,
                    {
                        "data": current_data,
                        "derived": current_data,
                    },
                    trusted_user_code=trusted_user_code,
                )
            else:
                data = local.apply(current_data)
            data = erlab.interactive.imagetool.slicer.ArraySlicer.validate_array(
                data,
                copy_values=False,
            )
        except _TrustedScriptReplayCancelled:
            raise
        except Exception as exc:
            raise _ProvenanceReplayFailure(
                f"{where}: replaying the requested provenance",
                exc,
            ) from exc

        if local.kind == "script":
            spec = provenance.compose_full_provenance(
                node.displayed_provenance_spec,
                local,
                script_context_names=("data", "derived"),
            )
        else:
            spec = provenance.compose_full_provenance(
                node.displayed_provenance_spec,
                local,
            )
        if spec is None:
            spec = local.to_replay_spec()
        node.replace_with_detached_data(data, spec, preserve_filter=False)
        self._manager._update_info(uid=node.uid)

    def can_delete_row(
        self,
        row: provenance._ProvenanceDisplayRow | None,
    ) -> tuple[bool, str]:
        node = self._metadata_node()
        if row is None or row.replay_ref is None:
            return False, "This row cannot be deleted."
        if row.replay_ref.kind != "operation":
            return False, "Only operation rows can be deleted."
        if not self._node_editable(node):
            return False, "Select an available ImageTool row to delete provenance."
        node = typing.cast("_ImageToolWrapper | _ManagedWindowNode", node)
        if self._source_child_parent_row(node, row):
            return False, "Delete the parent ImageTool row directly."
        spec = self._display_spec_for_row(node, row)
        if spec is None:
            return False, "This row does not have replayable provenance."
        if spec._operation_for_ref(row.replay_ref) is None:
            return False, "This operation is not available."
        if spec.kind == "script":
            try:
                group = _editable_group_range_for_ref(spec, row.replay_ref)
                if group is None:
                    candidate = spec._replace_operation_ref(row.replay_ref, ())
                else:
                    candidate = spec._replace_operation_range_ref(
                        row.replay_ref,
                        group[0],
                        group[1],
                        (),
                    )
            except (IndexError, ValueError):
                return False, "This script row is not a replayable step."
            if not (
                provenance.script_provenance_replayable(candidate)
                or provenance.script_provenance_requires_trust(candidate)
            ):
                return False, "Deleting this script step would make replay invalid."
        if spec.kind in {"full_data", "public_data", "selection"}:
            if row.scope == "source" and node.parent_uid is not None:
                return True, ""
            return False, "This live row needs a parent source to replay."
        return True, ""

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
        if row.edit_ref.kind == "file_load":
            if spec.kind not in {"file", "script"} or spec.file_load_source is None:
                return False, "This row is not a file load step."
            if spec.file_load_source.replay_call is None:
                return False, "This file load step cannot be replayed."
            return True, ""
        operation = spec._operation_for_ref(row.edit_ref)
        if operation is None:
            return False, "This operation is not available."
        script_operation = (
            operation if isinstance(operation, provenance.ScriptCodeOperation) else None
        )
        if script_operation is not None:
            if script_operation.code is None or not script_operation.copyable:
                return False, "This script step does not contain editable code."
            if spec.kind == "script":
                return True, ""
        if spec.kind == "script":
            input_spec = spec._prefix_before_ref(row.edit_ref)
            if not (
                provenance.script_provenance_replayable(input_spec)
                or provenance.script_provenance_requires_trust(input_spec)
            ):
                return False, "This script step cannot be replayed."
        active_filter_ref = self._active_filter_ref(node, spec)
        if (
            spec.kind in {"full_data", "public_data", "selection"}
            and row.scope != "source"
            and active_filter_ref != row.edit_ref
            and node.detached_live_parent_data is None
        ):
            return False, "This live row needs a parent source to replay."
        if script_operation is not None:
            return True, ""
        dialog_match = _dialog_match_for_operation_ref(spec, row.edit_ref)
        if dialog_match is None:
            reason = _uneditable_operation_reason(operation)
            return False, reason or "No editing dialog is available for this step."
        if not row.script_input_path and active_filter_ref == row.edit_ref:
            return True, ""
        reason = _recorded_operation_edit_unavailable_reason(
            _operations_for_ref(spec, row.edit_ref)[
                dialog_match.start : dialog_match.stop
            ]
        )
        if reason is not None:
            return False, reason
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
        if row.replay_ref.kind == "operation":
            group = _editable_group_range_for_ref(spec, row.replay_ref)
            if group is not None and row.replay_ref.operation_index != group[1] - 1:
                return (
                    False,
                    "Revert is available from the final momentum-conversion row.",
                )
        try:
            candidate = spec._prefix_through_ref(row.replay_ref)
        except ValueError:
            if spec.kind == "script":
                return False, "This script row is not a replayable step."
            return False, "This row is not a replayable step."
        if candidate == spec:
            return False, "Already at this provenance step."
        if spec.kind == "script":
            if not (
                provenance.script_provenance_replayable(candidate)
                or provenance.script_provenance_requires_trust(candidate)
            ):
                return False, "This script step cannot be replayed."
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
        except _TrustedScriptReplayCancelled:
            return
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
        repair_root_candidate: provenance.ToolProvenanceSpec | None = None
        try:
            candidate = spec._prefix_through_ref(ref)
            repair_root_candidate = self._root_candidate_for_row(node, row, candidate)
            self._validate_and_replace(
                node,
                row.scope,
                repair_root_candidate,
                where="validating the provenance revert target",
            )
        except _TrustedScriptReplayCancelled:
            return
        except Exception as exc:
            if self._handle_missing_source_file(
                node,
                row,
                title="Could Not Revert Provenance Step",
                exc=exc,
                repair_spec=repair_root_candidate,
            ):
                return
            self._show_failed("Could Not Revert Provenance Step", exc)

    def delete_row(self, row: provenance._ProvenanceDisplayRow | None) -> None:
        deletable, reason = self.can_delete_row(row)
        if not deletable:
            self._show_unavailable(reason)
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
                "Could Not Delete Provenance Step",
                RuntimeError("No provenance spec is available"),
            )
            return
        repair_root_candidate: provenance.ToolProvenanceSpec | None = None
        try:
            group = _editable_group_range_for_ref(spec, ref)
            if group is None:
                candidate = spec._replace_operation_ref(ref, ())
            else:
                candidate = spec._replace_operation_range_ref(
                    ref,
                    group[0],
                    group[1],
                    (),
                )
            if candidate.kind in {"file", "script"}:
                candidate = candidate.model_copy(
                    update={
                        "replay_stages": tuple(
                            stage
                            for stage in candidate.replay_stages
                            if stage.operations
                        )
                    }
                )
            repair_root_candidate = self._root_candidate_for_row(node, row, candidate)
            self._validate_and_replace(
                node,
                row.scope,
                repair_root_candidate,
                where="validating the provenance delete target",
            )
        except _TrustedScriptReplayCancelled:
            return
        except Exception as exc:
            if self._handle_missing_source_file(
                node,
                row,
                title="Could Not Delete Provenance Step",
                exc=exc,
                repair_spec=repair_root_candidate,
            ):
                return
            self._show_failed("Could Not Delete Provenance Step", exc)

    def _metadata_node(self) -> _ImageToolWrapper | _ManagedWindowNode | None:
        uid = self._manager._metadata_node_uid
        if uid is None:
            return None
        return self._manager._tool_graph.nodes.get(uid)

    def _paste_target_nodes(self) -> list[_ImageToolWrapper | _ManagedWindowNode]:
        selected_targets = self._manager._selected_imagetool_targets()
        if selected_targets:
            nodes: list[_ImageToolWrapper | _ManagedWindowNode] = []
            seen_uids: set[str] = set()
            for target in selected_targets:
                try:
                    node = self._manager._node_for_target(target)
                except (KeyError, IndexError):
                    continue
                if node.uid in seen_uids or not self._node_editable(node):
                    continue
                seen_uids.add(node.uid)
                nodes.append(node)
            return nodes

        node = self._metadata_node()
        if not self._node_editable(node):
            return []
        return [typing.cast("_ImageToolWrapper | _ManagedWindowNode", node)]

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

    def _root_display_spec_for_row(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        row: provenance._ProvenanceDisplayRow,
    ) -> provenance.ToolProvenanceSpec | None:
        if row.scope == "source":
            return node.displayed_source_spec
        return node.displayed_provenance_spec

    def _display_spec_for_row(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        row: provenance._ProvenanceDisplayRow,
    ) -> provenance.ToolProvenanceSpec | None:
        spec = self._root_display_spec_for_row(node, row)
        if spec is None or not row.script_input_path:
            return spec
        return self._script_input_path_spec(spec, row.script_input_path)

    @staticmethod
    def _script_input_path_spec(
        spec: provenance.ToolProvenanceSpec,
        path: tuple[int, ...],
    ) -> provenance.ToolProvenanceSpec | None:
        current = spec
        for index in path:
            if index < 0 or index >= len(current.script_inputs):
                return None
            nested = current.script_inputs[index].parsed_provenance_spec()
            if nested is None:
                return None
            current = nested
        return current

    def _root_candidate_for_row(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        row: provenance._ProvenanceDisplayRow,
        candidate: provenance.ToolProvenanceSpec,
    ) -> provenance.ToolProvenanceSpec:
        if not row.script_input_path:
            return candidate
        root = self._root_display_spec_for_row(node, row)
        if root is None:
            raise RuntimeError("No root provenance spec is available")
        return self._replace_script_input_path_spec(
            root,
            row.script_input_path,
            candidate,
        )

    def _replace_script_input_path_spec(
        self,
        spec: provenance.ToolProvenanceSpec,
        path: tuple[int, ...],
        replacement: provenance.ToolProvenanceSpec,
    ) -> provenance.ToolProvenanceSpec:
        if not path:
            return replacement
        index = path[0]
        if index < 0 or index >= len(spec.script_inputs):
            raise IndexError("Script input provenance path is not available")
        script_input = spec.script_inputs[index]
        nested = script_input.parsed_provenance_spec()
        if nested is None:
            raise RuntimeError("Script input does not have replayable provenance")
        replaced = self._replace_script_input_path_spec(
            nested,
            path[1:],
            replacement,
        )
        script_inputs = list(spec.script_inputs)
        script_inputs[index] = script_input.model_copy(
            update={
                "node_uid": None,
                "node_snapshot_token": None,
                "provenance_spec": replaced.model_dump(mode="json"),
            }
        )
        return spec.model_copy(update={"script_inputs": tuple(script_inputs)})

    def _replace_file_load_target_spec(
        self,
        root: provenance.ToolProvenanceSpec,
        target: _FileLoadBatchPeer,
        replacement: provenance.ToolProvenanceSpec,
    ) -> provenance.ToolProvenanceSpec:
        if target.spec.kind == "script" and replacement.kind != "script":
            replacement = _replace_file_load_fields(target.spec, replacement)
        if not target.script_input_path:
            return replacement
        return self._replace_script_input_path_spec(
            root,
            target.script_input_path,
            replacement,
        )

    def _file_load_targets(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        scope: typing.Literal["display", "source"],
        spec: provenance.ToolProvenanceSpec | None,
    ) -> tuple[_FileLoadBatchPeer, ...]:
        targets: list[_FileLoadBatchPeer] = []
        self._append_file_load_targets(
            targets,
            node,
            scope,
            spec,
            script_input_path=(),
            display_label=node.display_text,
        )
        return tuple(targets)

    def _append_file_load_targets(
        self,
        targets: list[_FileLoadBatchPeer],
        node: _ImageToolWrapper | _ManagedWindowNode,
        scope: typing.Literal["display", "source"],
        spec: provenance.ToolProvenanceSpec | None,
        *,
        script_input_path: tuple[int, ...],
        display_label: str,
    ) -> None:
        if spec is None:
            return
        if spec.kind in {"file", "script"} and spec.file_load_source is not None:
            load_source = spec.file_load_source
            targets.append(
                _FileLoadBatchPeer(
                    node=node,
                    scope=scope,
                    spec=spec,
                    original_path=pathlib.Path(load_source.path).expanduser(),
                    loader_summary=_loader_summary(load_source),
                    script_input_path=script_input_path,
                    display_label=display_label,
                )
            )
            if spec.kind != "script":
                return
        if spec.kind != "script":
            return
        for index, script_input in enumerate(spec.script_inputs):
            nested = script_input.parsed_provenance_spec()
            self._append_file_load_targets(
                targets,
                node,
                scope,
                nested,
                script_input_path=(*script_input_path, index),
                display_label=f"{display_label}: {script_input.label}",
            )

    def _file_load_target_for_path(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        scope: typing.Literal["display", "source"],
        spec: provenance.ToolProvenanceSpec | None,
        path: pathlib.Path,
    ) -> _FileLoadBatchPeer | None:
        target_path = _normalized_path(path)
        for target in self._file_load_targets(node, scope, spec):
            if _normalized_path(target.original_path) == target_path:
                return target
        return None

    def _missing_file_load_repair_peers(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        scope: typing.Literal["display", "source"],
        root_spec: provenance.ToolProvenanceSpec | None,
        focused: _FileLoadBatchPeer,
    ) -> tuple[_FileLoadBatchPeer, ...]:
        peers: list[_FileLoadBatchPeer] = []
        for target in self._file_load_targets(node, scope, root_spec):
            if target.target_id == focused.target_id or not target.script_input_path:
                continue
            load_source = target.spec.file_load_source
            replay_call = None if load_source is None else load_source.replay_call
            if load_source is None or replay_call is None:
                continue
            if pathlib.Path(load_source.path).expanduser().exists():
                continue
            peers.append(dataclasses.replace(target, preserve_loader=True))
        return tuple(peers)

    @staticmethod
    def _root_spec_for_batch_peer(
        peer: _FileLoadBatchPeer,
    ) -> provenance.ToolProvenanceSpec:
        root = (
            peer.node.displayed_source_spec
            if peer.scope == "source"
            else peer.node.displayed_provenance_spec
        )
        if root is None:
            raise RuntimeError("Matching file load has no root provenance")
        return root

    def _file_load_source_edit_target(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        path: pathlib.Path,
    ) -> tuple[
        _ImageToolWrapper | _ManagedWindowNode | None,
        typing.Literal["display", "source"] | None,
        _FileLoadBatchPeer | None,
        str,
    ]:
        source_bound = node.parent_uid is not None and node.source_spec is not None
        candidates: tuple[
            tuple[
                typing.Literal["display", "source"],
                provenance.ToolProvenanceSpec | None,
            ],
            ...,
        ] = (
            (("source", node.displayed_source_spec),)
            if source_bound
            else (("display", node.displayed_provenance_spec),)
        )

        for scope, spec in candidates:
            target = self._file_load_target_for_path(node, scope, spec, path)
            if target is None:
                continue
            load_source = target.spec.file_load_source
            if load_source is None:
                continue
            if load_source.replay_call is None:
                return None, None, None, "This file load step cannot be replayed."
            return (
                node,
                scope,
                target,
                "Select the current source file and update the recorded "
                "file-load step.",
            )
        if source_bound:
            parent = self._manager._parent_node(node)
            parent_node, parent_scope, parent_spec, parent_reason = (
                self._file_load_source_edit_target(parent, path)
            )
            if parent_node is not None and parent_scope is not None:
                return parent_node, parent_scope, parent_spec, parent_reason
            return None, None, None, parent_reason
        return (
            None,
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
        _node, _scope, target, reason = self._file_load_source_edit_target(node, path)
        return target is not None, reason

    def edit_file_load_source(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        path: pathlib.Path,
    ) -> None:
        if not self._node_editable(node):
            self._show_unavailable("This ImageTool is not available for editing.")
            return
        edit_node, scope, target, reason = self._file_load_source_edit_target(
            node, path
        )
        if edit_node is None or scope is None or target is None:
            self._show_unavailable(reason)
            return
        row = (
            None
            if not target.script_input_path
            else provenance._ProvenanceDisplayRow(
                provenance.DerivationEntry("File load", None),
                scope=scope,
                script_input_path=target.script_input_path,
            )
        )
        try:
            self._edit_file_load_spec(
                edit_node,
                scope,
                target.spec,
                where="validating the edited file-load provenance",
                row=row,
                batch_peers=self._file_load_batch_peers(
                    edit_node, target.spec, row=row
                ),
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
        if (
            spec is None
            or spec.kind not in {"file", "script"}
            or spec.file_load_source is None
        ):
            raise RuntimeError("Selected row is not a file load step")
        self._edit_file_load_spec(
            node,
            row.scope,
            spec,
            where="validating the edited file-load provenance",
            row=row,
            batch_peers=(
                self._file_load_batch_peers(node, spec, row=row)
                if row.script_input_path
                else self._file_load_batch_peers(node, spec)
            ),
        )

    def _edit_file_load_spec(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        scope: typing.Literal["display", "source"],
        spec: provenance.ToolProvenanceSpec,
        *,
        where: str,
        row: provenance._ProvenanceDisplayRow | None = None,
        batch_peers: Sequence[_FileLoadBatchPeer] = (),
        batch_apply_default: bool = False,
        checked_batch_peer_ids: frozenset[str] | None = None,
        root_spec: provenance.ToolProvenanceSpec | None = None,
    ) -> None:
        if spec.kind not in {"file", "script"} or spec.file_load_source is None:
            raise RuntimeError("Selected provenance does not have a file load step")
        dialog = _FileLoadEditDialog(
            spec.file_load_source,
            self._manager,
            batch_peers=batch_peers,
            batch_apply_default=batch_apply_default,
            checked_batch_peer_ids=checked_batch_peer_ids,
        )
        if dialog.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return
        candidate = dialog.provenance_spec(
            active_name=_file_load_edit_active_name(spec),
            replay_stages=spec.replay_stages,
        )
        if spec.kind == "script":
            candidate = _replace_file_load_fields(spec, candidate)
        if row is None:
            edit_candidate = candidate
        elif root_spec is None:
            edit_candidate = self._root_candidate_for_row(node, row, candidate)
        else:
            edit_candidate = self._replace_script_input_path_spec(
                root_spec,
                row.script_input_path,
                candidate,
            )
        peer_edits = [
            (peer, dialog.peer_provenance_spec(peer))
            for peer in dialog.selected_batch_peers()
        ]
        deferred_peer_edits: list[
            tuple[_FileLoadBatchPeer, provenance.ToolProvenanceSpec]
        ] = []
        for peer, peer_candidate in peer_edits:
            if peer.node.uid == node.uid and peer.scope == scope:
                edit_candidate = self._replace_file_load_target_spec(
                    edit_candidate,
                    peer,
                    peer_candidate,
                )
            else:
                deferred_peer_edits.append((peer, peer_candidate))
        validated_edits = [
            self._validated_edit(node, scope, edit_candidate, where=where),
        ]
        failures: list[tuple[_FileLoadBatchPeer, Exception]] = []
        for peer, peer_candidate in deferred_peer_edits:
            try:
                peer_where = (
                    "validating the edited file-load provenance for "
                    f"{peer.node.display_text}"
                )
                if peer.script_input_path:
                    peer_candidate = self._replace_file_load_target_spec(
                        self._root_spec_for_batch_peer(peer),
                        peer,
                        peer_candidate,
                    )
                validated_edits.append(
                    self._validated_edit(
                        peer.node,
                        peer.scope,
                        peer_candidate,
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
        if isinstance(operation, provenance.ScriptCodeOperation):
            self._edit_script_code_operation_row(node, row, spec, ref, operation)
            return
        dialog_match = _dialog_match_for_operation_ref(spec, ref)
        if dialog_match is None:
            raise RuntimeError("No editing dialog is available for this step")
        if not row.script_input_path and self._active_filter_ref(node, spec) == ref:
            self._edit_active_filter(node, operation, dialog_match.dialog_cls)
            return

        operations = _operations_for_ref(spec, ref)[
            dialog_match.start : dialog_match.stop
        ]
        replacements = self._edited_recorded_operations(
            operations,
            focus=dialog_match.focus,
        )
        if replacements is None:
            return
        candidate = spec._replace_operation_range_ref(
            ref,
            dialog_match.start,
            dialog_match.stop,
            replacements,
        )
        root_candidate = self._root_candidate_for_row(node, row, candidate)
        self._validate_and_replace(
            node,
            row.scope,
            root_candidate,
            where="validating the edited provenance step",
        )

    def _edit_script_code_operation_row(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        row: provenance._ProvenanceDisplayRow,
        spec: provenance.ToolProvenanceSpec,
        ref: provenance._ProvenanceStepRef,
        operation: provenance.ScriptCodeOperation,
    ) -> None:
        dialog = _ScriptCodeEditDialog(operation, self._manager)
        if dialog.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return
        replacement = operation.model_copy(update={"code": dialog.code()})
        candidate = spec._replace_operation_ref(ref, (replacement,))
        root_candidate = self._root_candidate_for_row(node, row, candidate)
        self._validate_and_replace(
            node,
            row.scope,
            root_candidate,
            where="validating the edited Python code",
        )

    def _edited_recorded_operations(
        self,
        operations: Sequence[provenance.ToolProvenanceOperation],
        *,
        focus: str | None = None,
    ) -> list[provenance.ToolProvenanceOperation] | None:
        operations = tuple(operations)
        if not operations:
            raise ValueError("No provenance operations were provided for editing")
        if reason := _recorded_operation_edit_unavailable_reason(operations):
            raise RuntimeError(reason)
        dialog = _RecordedOperationEditDialog(operations, self._manager, focus=focus)
        if dialog.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return None
        return dialog.edited_operations()

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
        base_candidate, filter_operation = self._split_active_filter(node, candidate)
        try:
            source_data, base_candidate = self._replay_candidate_result(
                node,
                scope,
                base_candidate,
            )
        except _TrustedScriptReplayCancelled:
            raise
        except Exception as exc:
            replay_target = (
                "provenance before the active display filter"
                if filter_operation is not None
                else "the requested provenance"
            )
            raise _ProvenanceReplayFailure(
                f"{where}: replaying {replay_target}",
                exc,
            ) from exc
        if filter_operation is not None:
            self._validate_filter_operation(
                node,
                source_data,
                filter_operation,
                where=where,
            )
        return _ValidatedProvenanceEdit(
            node=node,
            scope=scope,
            data=source_data,
            spec=base_candidate,
            filter_operation=filter_operation,
        )

    def _validate_filter_operation(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        data: xr.DataArray,
        operation: provenance.ToolProvenanceOperation,
        *,
        where: str,
    ) -> None:
        try:
            slicer_area = getattr(node, "slicer_area", None)
            filter_result = getattr(
                slicer_area,
                "_filter_operation_result_for_replacement",
                None,
            )
            if node.imagetool is not None and callable(filter_result):
                filter_result(data, operation)
                return
            operation.apply(data, parent_data=data)
        except Exception as exc:
            raise _ProvenanceReplayFailure(
                f"{where}: validating the active display filter",
                exc,
            ) from exc

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
        *,
        row: provenance._ProvenanceDisplayRow | None = None,
    ) -> tuple[_FileLoadBatchPeer, ...]:
        load_source = spec.file_load_source
        replay_call = None if load_source is None else load_source.replay_call
        if load_source is None or replay_call is None:
            return ()
        source_folder = _normalized_path(pathlib.Path(load_source.path).parent)
        peers: list[_FileLoadBatchPeer] = []
        if row is not None and row.script_input_path:
            root = self._root_display_spec_for_row(node, row)
            for target in self._file_load_targets(node, row.scope, root):
                if target.script_input_path == row.script_input_path:
                    continue
                peer_load_source = target.spec.file_load_source
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
                peers.append(target)

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
            if peer_spec is None or peer_spec.kind not in {"file", "script"}:
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
            if any(
                isinstance(operation, provenance.ScriptCodeOperation)
                for operation in spec.operations
            ):
                return self._replay_live_script_candidate(node, scope, spec), spec
            if scope == "source" and node.parent_uid is not None:
                parent = self._manager._parent_node(node)
                return spec.apply(parent.current_source_data()), spec
            parent_data = node.detached_live_parent_data
            if parent_data is None:
                raise RuntimeError("Live provenance needs a parent source to replay")
            return spec.apply(parent_data), spec
        if spec.kind == "script":
            result = self._manager._rebuild_script_provenance(
                spec,
                target_node_uid=node.uid,
            )
            return result.data, result.provenance_spec
        raise RuntimeError("Unsupported provenance kind")

    def _replay_live_script_candidate(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        scope: typing.Literal["display", "source"],
        spec: provenance.ToolProvenanceSpec,
    ) -> xr.DataArray:
        if scope == "source" and node.parent_uid is not None:
            parent = self._manager._parent_node(node)
            parent_data = parent.current_source_data()
        else:
            parent_data = node.detached_live_parent_data
            if parent_data is None:
                raise RuntimeError("Live provenance needs a parent source to replay")
        data = provenance.ToolProvenanceSpec._starting_data_for_kind(
            typing.cast(
                "typing.Literal['full_data', 'public_data', 'selection']",
                spec.kind,
            ),
            parent_data,
        )
        for operation in spec.operations:
            if not isinstance(operation, provenance.ScriptCodeOperation):
                data = operation.apply(data, parent_data=parent_data)
                continue
            step_spec = provenance.script(
                operation,
                start_label=operation.label,
                seed_code="derived = data",
                active_name="derived",
            )
            replay_inputs = {
                "data": data,
                "derived": data,
                "parent_data": parent_data,
            }
            trusted_user_code = provenance.script_provenance_requires_trust(
                step_spec,
                external_input_names=set(replay_inputs),
            )
            if trusted_user_code:
                self._manager._ensure_script_provenance_trusted(
                    step_spec,
                    reason="apply this provenance step",
                    external_input_names=set(replay_inputs),
                )
            data = provenance.replay_script_provenance(
                step_spec,
                replay_inputs,
                trusted_user_code=trusted_user_code,
            )
        return data

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
        live_parent_data = None
        if spec.kind in {"full_data", "public_data", "selection"}:
            live_parent_data = node.detached_live_parent_data
        node.replace_with_detached_data(
            data,
            spec,
            preserve_filter=preserve_filter,
            live_parent_data=live_parent_data,
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
        for ref, operation in reversed(tuple(provenance.iter_operation_refs(spec))):
            if operation == active:
                return ref
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
        if base_spec.kind in {"file", "script"}:
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

    def _show_partial_paste_failures(
        self,
        pasted_count: int,
        failures: Sequence[tuple[_ImageToolWrapper | _ManagedWindowNode, Exception]],
    ) -> None:
        failed_labels = "\n".join(
            f"- {node.display_text}: {exc}" for node, exc in failures
        )
        dialog = erlab.interactive.utils.MessageDialog(
            self._manager,
            title="Some Provenance Steps Could Not Be Pasted",
            text=(
                f"Pasted provenance steps into {pasted_count} selected ImageTool "
                f"{'row' if pasted_count == 1 else 'rows'}."
            ),
            informative_text=(
                f"{len(failures)} selected ImageTool "
                f"{'row was' if len(failures) == 1 else 'rows were'} left unchanged."
            ),
            detailed_text=failed_labels,
            buttons=QtWidgets.QDialogButtonBox.StandardButton.Ok,
            icon_pixmap=QtWidgets.QStyle.StandardPixmap.SP_MessageBoxWarning,
        )
        dialog.exec()

    def _show_failed(
        self,
        title: str,
        exc: Exception,
        *,
        text: str = "The provenance change could not be applied.",
        unchanged_reason: str | None = None,
    ) -> None:
        exc_text = "".join(traceback.TracebackException.from_exception(exc).format())
        where = (
            exc.where
            if isinstance(exc, _ProvenanceReplayFailure)
            else "replaying the requested provenance"
        )
        if unchanged_reason is None:
            unchanged_reason = (
                "The current ImageTool data was left unchanged because the requested "
                "provenance could not be replayed. Use Revert to This Step to drop "
                "later provenance, or adjust the earlier steps so the full chain is "
                "valid again."
            )
        dialog = erlab.interactive.utils.MessageDialog(
            self._manager,
            title=title,
            text=text,
            informative_text=f"Failed while: {where}\n\n{unchanged_reason}",
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
        root_spec = repair_spec
        if (
            root_spec is not None
            and row.script_input_path
            and self._script_input_path_spec(root_spec, row.script_input_path) is None
        ):
            root_spec = self._root_candidate_for_row(node, row, root_spec)
        if root_spec is None:
            root_spec = self._root_display_spec_for_row(node, row)

        missing = _missing_source_file_error_from_exception(exc)
        if missing is None:
            missing_path = _file_not_found_path_from_exception(exc)
            if missing_path is None:
                return False
            missing_target = self._file_load_target_for_path(
                node,
                row.scope,
                root_spec,
                missing_path,
            )
            if missing_target is None:
                return False
            missing = _MissingProvenanceSourceFileError(missing_target.original_path)

        target = self._file_load_target_for_path(
            node,
            row.scope,
            root_spec,
            missing.source_path,
        )
        if target is None:
            self._show_missing_source_file(title, exc, missing, can_edit=False)
            return True
        repair_target = target
        prompt_title = title
        while self._show_missing_source_file(
            prompt_title,
            exc,
            missing,
            can_edit=True,
        ):
            try:
                target = (
                    self._file_load_target_for_path(
                        node,
                        row.scope,
                        root_spec,
                        missing.source_path,
                    )
                    or repair_target
                )
                repair_target = target
                repair_row = (
                    None
                    if not target.script_input_path
                    else provenance._ProvenanceDisplayRow(
                        provenance.DerivationEntry("File load", None),
                        scope=row.scope,
                        script_input_path=target.script_input_path,
                    )
                )
                matching_peers = self._file_load_batch_peers(
                    node,
                    target.spec,
                    row=repair_row,
                )
                required_repair_peers = self._missing_file_load_repair_peers(
                    node,
                    row.scope,
                    root_spec,
                    target,
                )
                batch_peer_by_id = {
                    peer.target_id: peer for peer in required_repair_peers
                }
                for peer in matching_peers:
                    batch_peer_by_id.setdefault(peer.target_id, peer)
                checked_batch_peer_ids = frozenset(
                    peer.target_id for peer in required_repair_peers
                )
                self._edit_file_load_spec(
                    node,
                    row.scope,
                    target.spec,
                    where=(
                        "validating provenance after selecting a replacement "
                        "source file"
                    ),
                    row=repair_row,
                    batch_peers=tuple(batch_peer_by_id.values()),
                    batch_apply_default=bool(checked_batch_peer_ids),
                    checked_batch_peer_ids=checked_batch_peer_ids,
                    root_spec=root_spec,
                )
                break
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
                    break
                prompt_title = "Could Not Update Source File"
                exc = repair_exc
                missing = repair_missing
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
                "Update the file load step and try again."
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


def _file_load_edit_active_name(spec: provenance.ToolProvenanceSpec) -> str:
    if spec.kind == "script":
        seed_output = spec._script_seed_output_name()
        if seed_output is not None:
            return seed_output
    return spec.active_name or "derived"


def _replace_file_load_fields(
    spec: provenance.ToolProvenanceSpec,
    replacement: provenance.ToolProvenanceSpec,
) -> provenance.ToolProvenanceSpec:
    if replacement.kind != "file" or replacement.file_load_source is None:
        raise RuntimeError("Replacement is not a file load provenance spec")
    return spec.model_copy(
        update={
            "start_label": replacement.start_label,
            "seed_code": replacement.seed_code,
            "file_load_source": replacement.file_load_source,
            "replay_stages": replacement.replay_stages,
        }
    )


def _replace_recorded_path_text(
    text: str | None,
    old_path: pathlib.Path,
    new_path: pathlib.Path,
) -> str | None:
    if text is None:
        return None
    old_text = str(old_path)
    new_text = str(new_path)
    return text.replace(repr(old_text), repr(new_text)).replace(old_text, new_text)


def _relinked_file_load_spec(
    spec: provenance.ToolProvenanceSpec,
    path: pathlib.Path,
) -> provenance.ToolProvenanceSpec:
    load_source = spec.file_load_source
    if load_source is None or load_source.replay_call is None:
        raise RuntimeError("Matching file load is no longer replayable")
    old_path = pathlib.Path(load_source.path).expanduser()
    new_path = pathlib.Path(path).expanduser()
    return spec.model_copy(
        update={
            "seed_code": _replace_recorded_path_text(
                spec.seed_code,
                old_path,
                new_path,
            ),
            "file_load_source": load_source.model_copy(
                update={
                    "path": str(new_path),
                    "load_code": _replace_recorded_path_text(
                        load_source.load_code,
                        old_path,
                        new_path,
                    ),
                }
            ),
        }
    )


def _loader_summary(load_source: provenance.FileLoadSource) -> str:
    if not load_source.kwargs_text or load_source.kwargs_text == "(none)":
        return load_source.loader_text
    return f"{load_source.loader_text} ({load_source.kwargs_text})"
