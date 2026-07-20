"""File-source provenance editing and relinking."""

from __future__ import annotations

import ast
import dataclasses
import pathlib
import typing

from qtpy import QtCore, QtWidgets

import erlab
import erlab.interactive.utils
from erlab.interactive.imagetool._load_source import (
    _load_provenance_from_file_details,
    _loader_callable_text,
    _migrate_legacy_file_data_selection,
)
from erlab.interactive.imagetool._provenance._model import (
    FileDataSelection,
    FileLoadSource,
    FileReplayCall,
    ReplayStep,
    ToolProvenanceSpec,
    encode_provenance_value,
)
from erlab.interactive.imagetool.manager._dialogs import _LoaderOptionsWidget

if typing.TYPE_CHECKING:
    from collections.abc import Sequence

    from erlab.interactive.imagetool.manager._wrapper import (
        _ImageToolWrapper,
        _ManagedWindowNode,
    )


@dataclasses.dataclass(frozen=True)
class _FileLoadBatchPeer:
    node: _ImageToolWrapper | _ManagedWindowNode
    scope: typing.Literal["display", "source"]
    spec: ToolProvenanceSpec
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
        load_source: FileLoadSource,
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
            FileDataSelection(kind="dataarray")
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
        replay_steps: tuple[ReplayStep, ...],
    ) -> ToolProvenanceSpec:
        return self._provenance_spec_for(
            path=self.file_path(),
            selection=self._selection,
            active_name=active_name,
            replay_steps=replay_steps,
        )

    def peer_provenance_spec(
        self,
        peer: _FileLoadBatchPeer,
    ) -> ToolProvenanceSpec:
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
            replay_steps=peer.spec.steps,
        )

    def _provenance_spec_for(
        self,
        *,
        path: pathlib.Path,
        selection: FileDataSelection,
        active_name: str,
        replay_steps: tuple[ReplayStep, ...],
    ) -> ToolProvenanceSpec:
        _filter_name, func, kwargs = self.loader_options.checked_filter()
        if selection.kind == "parsed_index":
            selection = _migrate_legacy_file_data_selection(
                path,
                (func, kwargs, selection),
            )
        spec = _load_provenance_from_file_details(
            path,
            (func, kwargs, selection),
        )
        if spec is None:
            raise RuntimeError("Selected loader cannot be replayed")
        return spec.model_copy(
            update={
                "active_name": active_name,
                "steps": replay_steps,
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


def _normalized_path(path: pathlib.Path) -> pathlib.Path:
    try:
        return path.expanduser().resolve(strict=False)
    except (OSError, RuntimeError):
        return path.expanduser().absolute()


def _same_replay_loader(
    left: FileReplayCall,
    right: FileReplayCall,
) -> bool:
    return (
        left.kind == right.kind
        and left.target == right.target
        and encode_provenance_value(left.kwargs)
        == encode_provenance_value(right.kwargs)
    )


def _file_load_edit_active_name(
    spec: ToolProvenanceSpec,
) -> str:
    if spec.kind == "script":
        seed_output = spec._script_seed_output_name()
        if seed_output is not None:
            return seed_output
    return spec.active_name or "derived"


def _replace_file_load_fields(
    spec: ToolProvenanceSpec,
    replacement: ToolProvenanceSpec,
) -> ToolProvenanceSpec:
    if replacement.kind != "file" or replacement.file_load_source is None:
        raise RuntimeError("Replacement is not a file load provenance spec")
    return spec.model_copy(
        update={
            "start_label": replacement.start_label,
            "seed_code": replacement.seed_code,
            "file_load_source": replacement.file_load_source,
            "steps": replacement.steps,
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
    spec: ToolProvenanceSpec,
    path: pathlib.Path,
) -> ToolProvenanceSpec:
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


def _loader_summary(
    load_source: FileLoadSource,
) -> str:
    if not load_source.kwargs_text or load_source.kwargs_text == "(none)":
        return load_source.loader_text
    return f"{load_source.loader_text} ({load_source.kwargs_text})"
