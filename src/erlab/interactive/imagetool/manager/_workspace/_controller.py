"""Workspace document lifecycle, persistence, and user-facing document actions."""

from __future__ import annotations

import contextlib
import copy
import functools
import logging
import os
import pathlib
import time
import typing
import uuid

from qtpy import QtCore, QtWidgets

import erlab
import erlab.interactive._options.core
import erlab.interactive.imagetool._serialization as imagetool_serialization
import erlab.interactive.imagetool.manager._workspace._arrays as workspace_arrays
import erlab.interactive.imagetool.manager._workspace._format as workspace_format
import erlab.interactive.imagetool.manager._workspace._loading as workspace_loading
import erlab.interactive.imagetool.manager._workspace._saving as workspace_saving
import erlab.interactive.imagetool.manager._workspace._state as workspace_state
import erlab.interactive.imagetool.manager._workspace._storage as workspace_storage
import erlab.interactive.imagetool.manager._workspace._store as workspace_store
import erlab.interactive.imagetool.manager._workspace._trust as workspace_trust
import erlab.interactive.imagetool.slicer
import erlab.interactive.imagetool.viewer_linking
from erlab.interactive import _shortcut_sequences
from erlab.interactive._code_trust import (
    approve_document_trust,
    bind_document_trust_manifest,
    commit_local_edit_trust,
    create_manifest,
    document_trust_description,
    document_trust_has_trusted_lineage,
    document_trust_is_trusted,
    document_trust_needs_review,
    execution_capability_allows,
    external_document_trust,
    issue_complete_execution_capability,
    issue_execution_capability,
    issue_local_edit_capability,
    manifest_has_code,
    merge_document_trust,
    relocate_document_trust,
    relocate_manifest_entries,
    trusted_location_document_trust,
    untrusted_document_trust,
)
from erlab.interactive._code_trust._application import (
    load_document_trust,
    load_imported_document_trust,
    save_document_trust,
)
from erlab.interactive._code_trust._ui import confirm_code_trust
from erlab.interactive.imagetool.manager import _desktop
from erlab.interactive.imagetool.manager._widgets import (
    _RECENT_WORKSPACES_SETTINGS_KEY,
    _WORKSPACE_SAVE_SHORTCUT_OBJECT_NAME,
    _WORKSPACE_SAVE_WAIT_DIALOG_THRESHOLD_SECONDS,
    _manager_settings,
    _show_workspace_file_lock_error,
    _window_title_with_modified_placeholder,
    _WorkspaceDocumentAccess,
    _WorkspacePropertiesDialog,
    _WorkspacePropertiesState,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Iterator, Mapping

    import h5py
    import xarray as xr

    from erlab.interactive._code_trust._api import _DocumentTrust
    from erlab.interactive._code_trust._core import CodeTrustEntry, CodeTrustEntrySource
    from erlab.interactive.imagetool._mainwindow import ImageTool
    from erlab.interactive.imagetool.manager._mainwindow import ImageToolManager
    from erlab.interactive.imagetool.manager._workspace._state import (
        _WorkspaceStateSnapshot,
    )
    from erlab.interactive.imagetool.manager._wrapper import (
        _ImageToolWrapper,
        _ManagedWindowNode,
    )
    from erlab.interactive.utils import ToolWindow
else:
    import lazy_loader as _lazy

    h5py = _lazy.load("h5py")

logger = logging.getLogger(__name__)
_WORKSPACE_SAVE_SUFFIX_WARNING = "ImageTool Manager saves workspaces as .itws files."


class _WorkspacePostSaveBindingError(RuntimeError):
    """Raised when a saved workspace cannot be rebound into the live session."""


def _show_itws_workspace_warning(parent: QtWidgets.QWidget) -> None:
    QtWidgets.QMessageBox.warning(
        parent,
        "Workspace Not Saved",
        _WORKSPACE_SAVE_SUFFIX_WARNING,
    )


class _WorkspaceController:
    def __init__(self, manager: ImageToolManager) -> None:
        self._manager = manager
        self._local_edit_capability: object | None = None
        self._loader_state = workspace_format.WorkspaceLoaderState()
        self.loading = workspace_loading._WorkspaceLoader(manager, self)
        self.saving = workspace_saving._WorkspaceSaver(manager, self)
        self._workspace_window_state_applied: tuple[str, str, bool] | None = None
        self._node_window_state_applied: dict[
            str, tuple[tuple[tuple[int, str], ...], bool]
        ] = {}
        self._pending_node_window_modified: dict[str, bool] = {}
        self._background_save_worker: workspace_saving._WorkspaceSaveWorker | None = (
            None
        )
        self._background_save_receiver: (
            workspace_saving._WorkspaceSaveResultReceiver | None
        ) = None
        self._background_save_requested = False
        self._workspace_store: workspace_store.WorkspaceStore | None = None
        self._imported_workspace_accesses: dict[
            pathlib.Path, _WorkspaceDocumentAccess
        ] = {}
        self._imported_workspace_dependents: dict[pathlib.Path, set[str]] = {}
        self._workspace_gc_worker: workspace_saving._WorkspaceGcWorker | None = None
        self._workspace_gc_receiver: (
            workspace_saving._WorkspaceGcResultReceiver | None
        ) = None
        self._workspace_gc_requested = False

    def _loaded_workspace_code_trust(
        self,
        path: str | os.PathLike[str],
        manifest: Mapping[str, typing.Any],
        *,
        selected_paths: set[str] | None,
    ) -> _DocumentTrust:
        if workspace_trust.workspace_path_is_trusted(path):
            return trusted_location_document_trust()
        try:
            imported_manifest = workspace_trust.workspace_code_trust_manifest(
                manifest, selected_paths=selected_paths
            )
        except (TypeError, ValueError):
            logger.warning(
                "Workspace Python content could not be inspected",
                exc_info=True,
            )
            return untrusted_document_trust()
        if selected_paths is None:
            return load_document_trust(imported_manifest)
        try:
            code_manifest = workspace_trust.workspace_code_trust_manifest(manifest)
        except (TypeError, ValueError):
            # An unavailable unselected tool must not prevent a safe selected import.
            return load_document_trust(imported_manifest)
        return load_imported_document_trust(code_manifest, imported_manifest)

    def _legacy_workspace_code_trust(
        self, path: str | os.PathLike[str]
    ) -> _DocumentTrust:
        if workspace_trust.workspace_path_is_trusted(path):
            return trusted_location_document_trust()
        return untrusted_document_trust()

    def _load_with_code_trust(
        self,
        incoming: _DocumentTrust,
        *,
        replace: bool,
        load: Callable[[], bool],
    ) -> bool:
        """Apply incoming trust while payloads load, and roll it back on failure."""
        previous = self._manager._workspace_state.code_trust
        if (
            not replace
            and document_trust_needs_review(previous)
            and document_trust_has_trusted_lineage(incoming)
        ):
            # Imported node paths can be rebased after this merge. Do not let a
            # signed pre-rebase location authorize an equal existing location.
            incoming = untrusted_document_trust(incoming.manifest)
        self._merge_workspace_code_trust(incoming, replace=replace)
        try:
            loaded = load()
        except Exception:
            self._set_workspace_code_trust(previous)
            raise
        if not loaded:
            self._set_workspace_code_trust(previous)
        return loaded

    def _set_workspace_code_trust(self, trust: _DocumentTrust) -> None:
        """Set one manager-owned decision and notify executable features."""
        state = self._manager._workspace_state
        previous = state.code_trust
        if previous == trust:
            return
        state.code_trust = trust
        self._refresh_code_trust_ui()
        self._notify_code_trust_changed()

    def _merge_workspace_code_trust(
        self, incoming: _DocumentTrust, *, replace: bool = False
    ) -> None:
        self._set_workspace_code_trust(
            merge_document_trust(
                self._manager._workspace_state.code_trust,
                incoming,
                replace=replace,
            )
        )

    def adopt_external_code(self, entries: CodeTrustEntrySource) -> None:
        """Merge executable content imported from outside the manager document."""
        resolved_entries = tuple(entries() if callable(entries) else entries)
        if not resolved_entries:
            return
        self._merge_workspace_code_trust(
            external_document_trust(
                create_manifest(
                    workspace_trust.WORKSPACE_CODE_TRUST_DOMAIN,
                    workspace_trust.WORKSPACE_CODE_TRUST_POLICY_VERSION,
                    resolved_entries,
                )
            )
        )

    def _notify_code_trust_changed(self) -> None:
        for node in self._manager._tool_graph.nodes.values():
            if node.tool_window is not None:
                try:
                    node.tool_window._code_trust_changed()
                except Exception:
                    logger.exception(
                        "Could not update %s after the workspace trust changed",
                        node.uid,
                    )

    def _record_saved_workspace_code_trust(
        self,
        manifest: Mapping[str, typing.Any],
        *,
        trusted_lineage: bool,
        current_document: bool = True,
    ) -> None:
        state = self._manager._workspace_state
        try:
            code_manifest = workspace_trust.workspace_code_trust_manifest(manifest)
        except (TypeError, ValueError):
            logger.warning(
                "Saved workspace Python content could not be inspected",
                exc_info=True,
            )
            self._manager._status_bar.showMessage(
                "Workspace saved, but its Python content could not be inspected",
                5000,
            )
            if current_document:
                self._set_workspace_code_trust(untrusted_document_trust())
            return
        saved_trust, signature_stored = save_document_trust(
            state.code_trust,
            code_manifest,
            saved_trusted_lineage=trusted_lineage,
        )
        if not signature_stored:
            self._manager._status_bar.showMessage(
                "Workspace saved, but code trust could not be stored", 5000
            )
        if current_document:
            self._set_workspace_code_trust(saved_trust)

    def review_and_approve_workspace_code_trust(self) -> None:
        """Review the current executable bundle and approve it as one unit."""
        try:
            manifest = workspace_trust.current_workspace_code_trust_manifest(
                self._manager
            )
        except (TypeError, ValueError):
            QtWidgets.QMessageBox.warning(
                self._manager,
                "Stored Code Cannot Be Reviewed",
                "ERLab cannot inspect the saved executable content of one or more "
                "tools. Install the missing tool extension or remove the affected "
                "tool before you trust this workspace.",
            )
            return
        if manifest_has_code(manifest) and not confirm_code_trust(
            self._manager,
            manifest,
            document_name="Workspace",
            object_name="manager_code_trust_review_dialog",
            window_title="Review Workspace Code",
        ):
            return
        self._set_workspace_code_trust(
            approve_document_trust(self._manager._workspace_state.code_trust, manifest)
        )

    def issue_code_execution_capability(
        self,
        entries: CodeTrustEntrySource,
        *,
        focus_on_block: bool = False,
        allow_partial: bool = False,
    ) -> object | None:
        """Issue a capability for one exact workspace execution inventory.

        Graph replay can request a partial capability because it checks each entry at
        its execution boundary. All other callers require the complete inventory.
        """
        resolved_entries = tuple(entries() if callable(entries) else entries)
        if self._local_edit_capability is not None and (
            execution_capability_allows(
                self._local_edit_capability,
                resolved_entries,
            )
            or (
                allow_partial
                and any(
                    execution_capability_allows(
                        self._local_edit_capability,
                        (entry,),
                    )
                    for entry in resolved_entries
                )
            )
        ):
            return self._local_edit_capability
        issuer = (
            issue_execution_capability
            if allow_partial
            else issue_complete_execution_capability
        )
        trust, capability = issuer(
            self._manager._workspace_state.code_trust,
            resolved_entries,
        )
        self._set_workspace_code_trust(trust)
        if capability is None and focus_on_block:
            self._manager.code_trust_banner.setFocus()
        return capability

    @contextlib.contextmanager
    def local_code_edit(
        self,
        execution_entries: CodeTrustEntrySource,
        *,
        edited_entries: CodeTrustEntrySource,
        focus_on_block: bool = False,
    ) -> Iterator[object | None]:
        """Authorize one explicit local edit and commit lineage on success."""
        state = self._manager._workspace_state
        previous = state.code_trust
        resolved_execution_entries = tuple(
            execution_entries() if callable(execution_entries) else execution_entries
        )
        resolved_edited_entries = tuple(
            edited_entries() if callable(edited_entries) else edited_entries
        )
        prospective, capability = issue_local_edit_capability(
            previous,
            resolved_execution_entries,
            edited_entries=resolved_edited_entries,
        )
        if not execution_capability_allows(capability, resolved_edited_entries):
            capability = None
        previous_manifest = previous.manifest
        if prospective != previous and (
            previous_manifest is None
            or previous_manifest.domain != workspace_trust.WORKSPACE_CODE_TRUST_DOMAIN
            or previous_manifest.policy_version
            != workspace_trust.WORKSPACE_CODE_TRUST_POLICY_VERSION
        ):
            previous_manifest = workspace_trust.current_workspace_code_trust_manifest(
                self._manager
            )
        if capability is None and focus_on_block:
            self._manager.code_trust_banner.setFocus()
        previous_capability = self._local_edit_capability
        self._local_edit_capability = capability
        try:
            yield capability
        finally:
            self._local_edit_capability = previous_capability
        if prospective != previous and state.code_trust == previous:
            try:
                manifest = workspace_trust.current_workspace_code_trust_manifest(
                    self._manager
                )
            except (TypeError, ValueError):
                logger.warning(
                    "Edited workspace Python content could not be inspected",
                    exc_info=True,
                )
                return
            committed = commit_local_edit_trust(
                previous,
                capability,
                () if previous_manifest is None else previous_manifest.entries,
                manifest.entries,
                edited_entries=resolved_edited_entries,
                document_manifest=manifest,
            )
            if committed != previous:
                self._set_workspace_code_trust(committed)

    def _refresh_code_trust_ui(self) -> None:
        trust = self._manager._workspace_state.code_trust
        banner = getattr(self._manager, "code_trust_banner", None)
        if banner is None:
            return
        banner.setVisible(not document_trust_is_trusted(trust))

    def _bind_current_workspace_manifest_if_review_needed(self) -> None:
        """Bind the full inventory after a node changes a paused workspace."""
        trust = self._manager._workspace_state.code_trust
        if not document_trust_needs_review(trust):
            return
        try:
            manifest = workspace_trust.current_workspace_code_trust_manifest(
                self._manager
            )
        except (TypeError, ValueError):
            return
        self._set_workspace_code_trust(bind_document_trust_manifest(trust, manifest))

    @staticmethod
    def _locate_tool_code_trust_entries(
        entries: Iterable[CodeTrustEntry], *, location_getter: Callable[[], str]
    ) -> tuple[CodeTrustEntry, ...]:
        """Map feature-relative tool entries to exact workspace locations."""
        return relocate_manifest_entries(
            create_manifest("erlab.workspace-tool-location", 1, entries),
            location_prefix=location_getter(),
        )

    def _configure_tool_code_trust(
        self,
        tool: ToolWindow,
        *,
        location_getter: Callable[[], str] | None = None,
    ) -> None:
        """Connect one materialized tool to the manager-owned trust decision."""
        incoming_trust = tool._document_trust
        current_trust = self._manager._workspace_state.code_trust
        if incoming_trust == current_trust:
            incoming_trust = None
        elif location_getter is not None:
            manifest = tool._current_code_trust_manifest()
            if manifest is not None:
                incoming_trust = relocate_document_trust(
                    incoming_trust,
                    manifest,
                    location_prefix=location_getter(),
                )
        if incoming_trust is not None:
            self._merge_workspace_code_trust(incoming_trust)
        tool._set_code_trust_host(
            self.issue_code_execution_capability,
            local_edit_context=self.local_code_edit,
            state_getter=lambda: self._manager._workspace_state.code_trust,
            entry_locator=(
                None
                if location_getter is None
                else functools.partial(
                    self._locate_tool_code_trust_entries,
                    location_getter=location_getter,
                )
            ),
        )

    def _locate_imagetool_code_trust_entries(
        self,
        entries: Iterable[CodeTrustEntry],
        *,
        node_getter: Callable[[], _ImageToolWrapper | _ManagedWindowNode],
    ) -> tuple[CodeTrustEntry, ...]:
        """Map runtime provenance entries to one ImageTool document location."""
        node = node_getter()
        location_prefix = (
            f"{self.saving._workspace_node_path_for_node(node)}/provenance"
        )
        return relocate_manifest_entries(
            create_manifest("erlab.workspace-imagetool-location", 1, entries),
            location_prefix=location_prefix,
            remove_location_prefix="runtime/",
        )

    def _configure_imagetool_code_trust(
        self,
        tool: ImageTool,
        *,
        node_getter: (
            Callable[[], _ImageToolWrapper | _ManagedWindowNode] | None
        ) = None,
    ) -> None:
        """Connect one ImageTool to the manager-owned trust decision."""

        def issue(entries: Iterable[CodeTrustEntry]) -> object | None:
            if node_getter is not None:
                entries = self._locate_imagetool_code_trust_entries(
                    entries,
                    node_getter=node_getter,
                )
            return self.issue_code_execution_capability(
                entries,
                focus_on_block=True,
                allow_partial=True,
            )

        tool.slicer_area._set_stored_code_authorizer(issue)

    def _tool_data_reference_matches_current_snapshot(
        self, reference: Mapping[str, typing.Any]
    ) -> bool:
        if reference.get("kind") != "manager_node":
            return True
        node_uid = reference.get("node_uid")
        if not isinstance(node_uid, str) or not node_uid:
            return False
        node = self._manager._tool_graph.nodes.get(node_uid)
        if node is None:
            return False
        data_role = reference.get("data_role", "displayed")
        if data_role not in {"source", "displayed"}:
            return False
        snapshot_token = reference.get("node_snapshot_token")
        return snapshot_token is None or (
            isinstance(snapshot_token, str)
            and snapshot_token != ""
            and snapshot_token == node.snapshot_token_for_role(data_role)
        )

    def _tool_data_reference_matches_current_data(
        self,
        reference: Mapping[str, typing.Any],
        data: xr.DataArray,
        *,
        owner_node: _ManagedWindowNode,
    ) -> bool:
        if not self._tool_data_reference_matches_current_snapshot(reference):
            return False
        if reference.get("kind") != "manager_node":
            return True
        node_uid = typing.cast("str", reference.get("node_uid"))
        node = self._manager._tool_graph.nodes[node_uid]
        data_role = typing.cast(
            "typing.Literal['source', 'displayed']",
            reference.get("data_role", "displayed"),
        )
        try:
            resolved = node.data_for_role(data_role)
            tool = owner_node.tool_window
            if tool is None:
                return False
            input_name = reference.get("input_name")
            script_input = next(
                (
                    (index, item)
                    for index, item in enumerate(tool.script_inputs)
                    if item.name == input_name
                ),
                None,
            )
            authorization = None
            if script_input is not None:
                index, item = script_input
                entries = tuple(
                    tool._source_spec_code_trust_entries(
                        item.parsed_source_spec(),
                        location_prefix=f"tool-inputs/{index}:{item.name}/source",
                    )
                )
                if entries:
                    entries = self._locate_tool_code_trust_entries(
                        entries,
                        location_getter=lambda: (
                            self.saving._workspace_node_path_for_node(owner_node)
                        ),
                    )
                    authorization = self.issue_code_execution_capability(entries)
                    if authorization is None:
                        return False
            resolved = (
                erlab.interactive.utils.ToolWindow._apply_saved_tool_data_reference(
                    reference,
                    resolved,
                    authorization=authorization,
                )
            )
        except Exception:
            return False
        return erlab.interactive.utils.ToolWindow._reference_resolves_current_tool_data(
            resolved, data
        )

    def _commit_saved_tool_data_references(
        self, snapshot: workspace_saving._WorkspaceSaveSnapshot
    ) -> None:
        for uid, snapshot_token, references in snapshot.serialized_tool_data_references:
            node = self._manager._tool_graph.nodes.get(uid)
            if (
                node is None
                or node.is_imagetool
                or node.snapshot_token != snapshot_token
                or not all(
                    self._tool_data_reference_matches_current_snapshot(reference)
                    for reference in references.values()
                )
            ):
                continue
            node._set_workspace_tool_data_references(references)

    @staticmethod
    def _normalize_recent_workspace_paths(
        paths: Iterable[str | os.PathLike[str]],
        *,
        limit: int,
    ) -> list[pathlib.Path]:
        recent_paths: list[pathlib.Path] = []
        seen: set[str] = set()
        for value in paths:
            path = pathlib.Path(value).expanduser().resolve()
            if path.suffix.lower() != ".itws":
                continue
            key = os.path.normcase(str(path))
            if key in seen:
                continue
            recent_paths.append(path)
            seen.add(key)
            if len(recent_paths) >= limit:
                break
        return recent_paths

    def _recent_workspace_paths(self) -> list[pathlib.Path]:
        limit = erlab.interactive.options.model.io.recent_workspace_limit
        settings = _manager_settings()
        settings.sync()
        values = settings.value(_RECENT_WORKSPACES_SETTINGS_KEY, [])
        if isinstance(values, str):
            stored_paths = [values] if values else []
        elif isinstance(values, (list, tuple)):
            stored_paths = [str(value) for value in values if value]
        else:
            stored_paths = []
        recent_paths = self._normalize_recent_workspace_paths(
            stored_paths,
            limit=limit,
        )
        if len(stored_paths) > limit:
            self._set_recent_workspace_paths(recent_paths)
        return recent_paths

    def _set_recent_workspace_paths(
        self, paths: Iterable[str | os.PathLike[str]]
    ) -> None:
        recent_paths = self._normalize_recent_workspace_paths(
            paths,
            limit=erlab.interactive.options.model.io.recent_workspace_limit,
        )
        settings = _manager_settings()
        if recent_paths:
            settings.setValue(
                _RECENT_WORKSPACES_SETTINGS_KEY,
                [str(path) for path in recent_paths],
            )
        else:
            settings.remove(_RECENT_WORKSPACES_SETTINGS_KEY)
        settings.sync()

    def _record_recent_workspace(self, fname: str | os.PathLike[str]) -> None:
        path = pathlib.Path(fname).expanduser().resolve()
        if path.suffix.lower() != ".itws":
            return
        path_key = os.path.normcase(str(path))
        paths = [
            existing
            for existing in self._recent_workspace_paths()
            if os.path.normcase(str(existing)) != path_key
        ]
        self._set_recent_workspace_paths([path, *paths])
        self._refresh_open_recent_menu_action()
        if erlab.utils.misc._IS_PACKAGED:
            _desktop.record_recent_workspace(path)

    def _clear_recent_workspaces(self) -> None:
        self._set_recent_workspace_paths([])
        self._populate_open_recent_menu()

    def _refresh_open_recent_menu_action(self) -> None:
        self._manager.open_recent_menu.setEnabled(
            bool(self._recent_workspace_paths())
            and not self._manager._workspace_state.save_in_progress
        )

    def _populate_open_recent_menu(self) -> None:
        self._manager.open_recent_menu.clear()
        paths = self._recent_workspace_paths()
        self._manager.open_recent_menu.setEnabled(
            bool(paths) and not self._manager._workspace_state.save_in_progress
        )
        if not paths:
            return

        name_counts: dict[str, int] = {}
        for path in paths:
            name_counts[path.name] = name_counts.get(path.name, 0) + 1

        for index, path in enumerate(paths):
            label = path.name
            if name_counts[path.name] > 1:
                label = f"{path.name} ({path.parent.name or path.parent})"
            action = QtWidgets.QAction(label, self._manager.open_recent_menu)
            action.setObjectName(f"manager_recent_workspace_action_{index}")
            action.setData(str(path))
            action.setToolTip(str(path))
            action.setStatusTip(str(path))
            action.triggered.connect(
                lambda _checked=False, recent_path=path: (
                    self._manager.open_recent_workspace(recent_path)
                )
            )
            self._manager.open_recent_menu.addAction(action)

        self._manager.open_recent_menu.addSeparator()
        clear_action = QtWidgets.QAction("Clear Menu", self._manager.open_recent_menu)
        clear_action.setObjectName("manager_clear_recent_workspaces_action")
        clear_action.triggered.connect(self._clear_recent_workspaces)
        self._manager.open_recent_menu.addAction(clear_action)

    def _load_workspace_path(self, path: pathlib.Path, *, native: bool = True) -> bool:
        if self._manager._workspace_state.save_in_progress:
            self._manager._status_bar.showMessage(
                "Workspace save in progress; open after it finishes", 3000
            )
            return False
        self._manager._recent_directory = str(path.parent)
        try:
            loaded = self.loading._load_workspace_file(
                path,
                replace=True,
                associate=True,
                mark_dirty=False,
                select=False,
                native=native,
            )
        except Exception as exc:
            if workspace_storage._is_workspace_file_lock_error(exc):
                logger.info(
                    "Workspace file is already open or locked: %s",
                    path,
                    extra={"suppress_ui_alert": True},
                )
                _show_workspace_file_lock_error(self._manager, path)
            else:
                logger.exception(
                    "Error while loading workspace",
                    extra={"suppress_ui_alert": True},
                )
                erlab.interactive.utils.MessageDialog.critical(
                    self._manager,
                    "Error",
                    "An error occurred while loading the workspace file.",
                )
            return False
        if loaded:
            self._record_recent_workspace(path)
        return loaded

    def _request_workspace_open(
        self, path: pathlib.Path, *, native: bool = True
    ) -> typing.Literal["opened", "scheduled", "cancelled", "failed"]:
        """Prompt for unsaved changes and open one recognized workspace path."""
        choice = self._dirty_workspace_save_choice(
            "Opening a workspace replaces the windows currently in this manager."
        )
        if choice == "cancel":
            return "cancelled"
        if choice in {"clean", "discard"}:
            loaded = self._load_workspace_path(path, native=native)
            return "opened" if loaded else "failed"

        def _continue_after_save(save_succeeded: bool) -> None:
            if save_succeeded and not self._manager.is_workspace_modified:
                self._load_workspace_path(path, native=native)

        return (
            "scheduled"
            if self.save(native=native, on_finished=_continue_after_save)
            else "failed"
        )

    def open_workspace_candidate(
        self, fname: str | os.PathLike[str], *, native: bool = True
    ) -> typing.Literal["not-workspace", "handled", "stop"]:
        """Recognize and open a path supplied by the general file-ingress flow.

        The return value tells data ingress whether the path is ordinary data,
        whether the workspace path was fully handled, or whether a cancel/deferred
        save means the rest of the current file batch must stop.
        """
        if self._manager._workspace_state.save_in_progress:
            self._manager._status_bar.showMessage(
                "Workspace save in progress; open after it finishes", 3000
            )
            return "stop"

        path = pathlib.Path(fname).expanduser().resolve()
        explicit_workspace = workspace_format._workspace_path_is_itws(path)
        try:
            tree = workspace_arrays.open_workspace_datatree(path, chunks=None)
        except Exception as exc:
            if workspace_storage._is_workspace_file_lock_error(exc):
                logger.info(
                    "Workspace file is already open or locked: %s",
                    path,
                    extra={"suppress_ui_alert": True},
                )
                _show_workspace_file_lock_error(self._manager, path)
                return "handled"
            if explicit_workspace:
                self._manager._show_operation_error(
                    "Error while loading workspace",
                    "An error occurred while loading the workspace file.",
                )
                return "handled"
            logger.debug("Failed to open %s as datatree workspace", path, exc_info=True)
            return "not-workspace"

        try:
            is_workspace = self.loading._is_datatree_workspace(tree)
        finally:
            tree.close()
        if not is_workspace:
            if not explicit_workspace:
                return "not-workspace"
            logger.error(
                "File with .itws extension is not an ImageTool workspace: %s",
                path,
                extra={"suppress_ui_alert": True},
            )
            erlab.interactive.utils.MessageDialog.critical(
                self._manager,
                "Error",
                "An error occurred while loading the workspace file.",
                f"{path.name} is not a valid ImageTool workspace file.",
            )
            return "handled"

        status = self._request_workspace_open(path, native=native)
        if status in {"cancelled", "scheduled"}:
            return "stop"
        return "handled"

    def _open_workspace_after_dirty_prompt(
        self, fname: str | os.PathLike[str], *, native: bool = True
    ) -> bool:
        if self._manager._workspace_state.save_in_progress:
            self._manager._status_bar.showMessage(
                "Workspace save in progress; open after it finishes", 3000
            )
            return False
        path = pathlib.Path(fname).expanduser().resolve()
        return self._request_workspace_open(path, native=native) in {
            "opened",
            "scheduled",
        }

    def open_recent_workspace(self, fname: str | os.PathLike[str]) -> bool:
        """Open a recently used workspace file."""
        if self._manager._workspace_state.save_in_progress:
            self._manager._status_bar.showMessage(
                "Workspace save in progress; open after it finishes", 3000
            )
            return False
        path = pathlib.Path(fname).expanduser().resolve()
        path_key = os.path.normcase(str(path))
        if not path.exists():
            self._set_recent_workspace_paths(
                existing
                for existing in self._recent_workspace_paths()
                if os.path.normcase(str(existing)) != path_key
            )
            self._refresh_open_recent_menu_action()
            QtWidgets.QMessageBox.warning(
                self._manager,
                "Workspace Not Found",
                f"The recent workspace file no longer exists:\n{path}",
            )
            return False
        if not workspace_format._workspace_path_is_itws(path):
            self._set_recent_workspace_paths(
                existing
                for existing in self._recent_workspace_paths()
                if os.path.normcase(str(existing)) != path_key
            )
            self._refresh_open_recent_menu_action()
            QtWidgets.QMessageBox.warning(
                self._manager,
                "Unsupported Workspace File",
                "ImageTool Manager opens workspace files with the .itws extension.",
            )
            return False
        return self._open_workspace_after_dirty_prompt(path)

    @property
    def workspace_path(self) -> str | None:
        """Path of the workspace document associated with this manager."""
        return (
            None
            if self._manager._workspace_state.path is None
            else str(self._manager._workspace_state.path)
        )

    def show_workspace_properties(self) -> None:
        """Show properties for the workspace associated with this manager."""
        _WorkspacePropertiesDialog(
            self._manager.workspace_path,
            state=self._workspace_properties_state(),
            review_code_callback=self.review_and_approve_workspace_code_trust,
            parent=self._manager,
        ).exec()

    def _workspace_properties_state(self) -> _WorkspacePropertiesState:
        trust = self._manager._workspace_state.code_trust
        return _WorkspacePropertiesState(
            is_modified=self._manager.is_workspace_modified,
            top_level_window_count=self._manager.ntools,
            code_trust_text=document_trust_description(trust),
            code_trust_review_available=document_trust_needs_review(trust),
        )

    @property
    def is_workspace_modified(self) -> bool:
        """Return whether this workspace has unsaved restorable changes."""
        return self._manager._workspace_state.is_modified(
            has_nodes=bool(self._manager._tool_graph.nodes)
        )

    def _refresh_manager_record(self, *, coalesce_if_busy: bool = True) -> None:
        self._manager._registry_heartbeat.request_refresh(
            self._manager.workspace_path,
            coalesce_if_busy=coalesce_if_busy,
        )

    def _workspace_window_state(self) -> tuple[str, str, bool]:
        if self._manager._workspace_state.path is None:
            window_file_path = ""
        else:
            window_file_path = typing.cast("str", self._manager.workspace_path)
        workspace_display_name = (
            "Untitled"
            if self._manager._workspace_state.path is None
            else self._manager._workspace_state.path.name
        )
        title = (
            f"{_window_title_with_modified_placeholder(workspace_display_name)}"
            f" - ImageTool Manager #{self._manager.manager_index}"
        )
        return window_file_path, title, self._manager.is_workspace_modified

    def _update_workspace_window_title(self, *, force: bool = True) -> None:
        if force:
            self._apply_workspace_window_title()
            return
        self._manager._queue_idle_work(
            ("workspace-window", "title"), self._apply_workspace_window_title
        )

    def _apply_workspace_window_title(self) -> None:
        window_file_path, title, modified = self._workspace_window_state()
        applied = self._workspace_window_state_applied
        if applied is None or applied[0] != window_file_path:
            self._manager.setWindowFilePath(window_file_path)
        if applied is None or applied[1] != title:
            self._manager.setWindowTitle(title)
        if applied is None or applied[2] != modified:
            self._manager.setWindowModified(modified)
        self._workspace_window_state_applied = (window_file_path, title, modified)

    def _release_workspace_lock(self) -> None:
        if self._workspace_store is not None:
            self._workspace_store.close()
            self._workspace_store = None
        if self._manager._workspace_state.lock is None:
            return
        self._manager._workspace_state.lock.unlock()
        self._manager._workspace_state.lock = None

    def _release_imported_workspace_accesses(self) -> None:
        """Release all source documents retained by imported lazy payloads."""
        accesses = tuple(self._imported_workspace_accesses.values())
        self._imported_workspace_accesses.clear()
        self._imported_workspace_dependents.clear()
        for access in accesses:
            access.release()

    @staticmethod
    def _workspace_node_source_paths(
        node: _ManagedWindowNode,
    ) -> frozenset[pathlib.Path]:
        """Return workspace paths directly referenced by one managed node."""
        paths: set[pathlib.Path] = set()
        pending = node.pending_workspace_payload
        if pending is not None:
            paths.add(pending[0].resolve())
        if node.is_imagetool and node.imagetool is not None:
            data_backing, source_paths = node.persistence_data_backing()
            if data_backing in {"dask", "file_lazy"}:
                paths.update(pathlib.Path(path).resolve() for path in source_paths)
        paths.update(
            path.resolve() for path, _group in node._workspace_reference_datasets
        )
        return frozenset(paths)

    def _retain_imported_workspace_access(
        self,
        access: _WorkspaceDocumentAccess,
        candidate_uids: Collection[str],
    ) -> None:
        """Retain an imported source only when loaded nodes still read from it."""
        path = access.path.resolve()
        dependents = {
            uid
            for uid in candidate_uids
            if (
                (node := self._manager._tool_graph.nodes.get(uid)) is not None
                and path in self._workspace_node_source_paths(node)
            )
        }
        if not dependents:
            return
        self._imported_workspace_dependents.setdefault(path, set()).update(dependents)
        if path in self._imported_workspace_accesses:
            return
        workspace_lock = access.take_lock()
        if workspace_lock is not None:
            self._imported_workspace_accesses[path] = _WorkspaceDocumentAccess(
                path, workspace_lock
            )

    def _release_unused_imported_workspace_accesses(self) -> None:
        """Release imported sources after all dependent payloads are rehomed."""
        for path, access in tuple(self._imported_workspace_accesses.items()):
            previous_dependents = self._imported_workspace_dependents.get(path, set())
            dependents: set[str] = set()
            for uid, node in self._manager._tool_graph.nodes.items():
                source_paths = self._workspace_node_source_paths(node)
                if path in source_paths:
                    dependents.add(uid)
                    continue
                if (
                    uid in previous_dependents
                    and node.is_imagetool
                    and node.imagetool is not None
                    and node.slicer_area._data.chunks is not None
                    and not source_paths
                ):
                    # Some Dask transformations discard xarray's source encoding.
                    # Keep ownership until a save rebinds or materializes this node.
                    dependents.add(uid)
            if dependents:
                self._imported_workspace_dependents[path] = dependents
                continue
            self._imported_workspace_accesses.pop(path, None)
            self._imported_workspace_dependents.pop(path, None)
            access.release()

    def _take_imported_workspace_lock(
        self, path: str | os.PathLike[str]
    ) -> QtCore.QLockFile | None:
        """Transfer a retained source lock to the associated workspace document."""
        resolved = pathlib.Path(path).resolve()
        access = self._imported_workspace_accesses.pop(resolved, None)
        self._imported_workspace_dependents.pop(resolved, None)
        return None if access is None else access.take_lock()

    def _take_workspace_access_lock(
        self, access: _WorkspaceDocumentAccess
    ) -> QtCore.QLockFile | None:
        """Transfer the lock that owns one workspace document."""
        workspace_lock = access.take_lock()
        if workspace_lock is None:
            workspace_lock = self._take_imported_workspace_lock(access.path)
        return workspace_lock

    def _current_workspace_document_path(self) -> pathlib.Path | None:
        path = self._manager._workspace_state.path
        if path is None or not workspace_format._workspace_path_is_itws(path):
            return None
        return path

    def _workspace_document_access(
        self, fname: str | os.PathLike[str]
    ) -> _WorkspaceDocumentAccess:
        workspace_path = pathlib.Path(fname).resolve()
        workspace_lock = None
        if (
            workspace_path != self._manager._workspace_state.path
            and workspace_path not in self._imported_workspace_accesses
        ):
            workspace_lock = workspace_storage._acquire_workspace_document_lock(
                workspace_path
            )
        return _WorkspaceDocumentAccess(workspace_path, workspace_lock)

    @contextlib.contextmanager
    def _workspace_document_access_context(
        self, fname: str | os.PathLike[str]
    ) -> Iterator[_WorkspaceDocumentAccess]:
        access = self._workspace_document_access(fname)
        try:
            yield access
        finally:
            access.release()

    def _set_workspace_path(
        self,
        fname: str | os.PathLike[str] | None,
        *,
        workspace_lock: QtCore.QLockFile | None = None,
        store: workspace_store.WorkspaceStore | None = None,
    ) -> None:
        workspace_path = None if fname is None else pathlib.Path(fname).resolve()
        if workspace_path == self._manager._workspace_state.path:
            if workspace_lock is not None:
                workspace_lock.unlock()
            if store is not None and store is not self._workspace_store:
                if self._workspace_store is not None:
                    self._workspace_store.close()
                self._workspace_store = store
            self._update_workspace_window_title()
            self._refresh_manager_record()
            return

        if workspace_path is not None and workspace_lock is None:
            raise RuntimeError(
                "Changing the workspace path requires a pre-acquired document lock"
            )
        old_workspace_path = self._manager._workspace_state.path
        if store is self._workspace_store:
            self._workspace_store = None
        self._release_workspace_lock()
        self._manager._workspace_state.lock = workspace_lock
        self._workspace_store = store
        self._manager._workspace_state.path = workspace_path
        self._manager._workspace_state.advance_document_identity()
        if old_workspace_path is not None and workspace_path is not None:
            self._repoint_pending_workspace_payloads(old_workspace_path, workspace_path)
        if self._manager._workspace_state.path is not None:
            self._manager._recent_directory = str(
                self._manager._workspace_state.path.parent
            )
        self._update_workspace_window_title()
        self._refresh_manager_record()

    def _repoint_pending_workspace_payloads(
        self,
        old_workspace_path: str | os.PathLike[str],
        new_workspace_path: str | os.PathLike[str],
    ) -> None:
        old_normalized = workspace_arrays._normalized_file_path(old_workspace_path)
        for node in self._manager._tool_graph.nodes.values():
            pending = node.pending_workspace_payload
            kind = node.pending_workspace_payload_kind
            if pending is None or kind is None:
                continue
            pending_workspace_path, payload_path = pending
            if (
                workspace_arrays._normalized_file_path(pending_workspace_path)
                == old_normalized
            ):
                node.set_pending_workspace_payload(
                    kind,
                    new_workspace_path,
                    payload_path,
                    payload_attrs=node.pending_workspace_payload_attrs,
                )

    def _repoint_saved_pending_workspace_payloads(
        self,
        workspace_path: str | os.PathLike[str],
        *,
        manifest: Mapping[str, typing.Any],
    ) -> None:
        payload_paths = {
            uid: payload_path
            for uid, _kind, payload_path in (
                workspace_format._workspace_manifest_payload_entries(manifest)
            )
        }
        for uid, node in self._manager._tool_graph.nodes.items():
            pending = node.pending_workspace_payload
            kind = node.pending_workspace_payload_kind
            payload_path = payload_paths.get(uid)
            if pending is None or kind is None or payload_path is None:
                continue
            node.set_pending_workspace_payload(
                kind,
                workspace_path,
                payload_path,
                payload_attrs=node.pending_workspace_payload_attrs,
            )

    def _adopt_committed_workspace_generation(
        self,
        workspace_path: str | os.PathLike[str],
        snapshot: workspace_saving._WorkspaceSaveSnapshot,
        *,
        manifest: Mapping[str, typing.Any],
    ) -> None:
        """Apply one committed generation to live manager references."""
        self._commit_saved_tool_data_references(snapshot)
        store = self._workspace_store
        if store is not None and not store.closed:
            store.rebind_legacy_readers(
                dict(snapshot.generation_plan.legacy_reader_rebindings)
            )
        self._repoint_saved_pending_workspace_payloads(
            workspace_path,
            manifest=manifest,
        )
        scripts = self._manager._workspace_state.extension_scripts
        for script_name, source_hash, source in snapshot.embedded_script_sources:
            scripts.remember_verified_source(script_name, source_hash, source)
        self._manager._workspace_state.schema_version = (
            workspace_format._current_workspace_schema_version()
        )

    def _pending_workspace_payload_snapshot(
        self,
    ) -> dict[
        str,
        tuple[
            typing.Literal["imagetool", "tool"],
            tuple[pathlib.Path, str],
            dict[str, typing.Any] | None,
        ],
    ]:
        snapshot: dict[
            str,
            tuple[
                typing.Literal["imagetool", "tool"],
                tuple[pathlib.Path, str],
                dict[str, typing.Any] | None,
            ],
        ] = {}
        for uid, node in self._manager._tool_graph.nodes.items():
            pending = node.pending_workspace_payload
            kind = node.pending_workspace_payload_kind
            if pending is not None and kind is not None:
                snapshot[uid] = (kind, pending, node.pending_workspace_payload_attrs)
        return snapshot

    def _restore_pending_workspace_payload_snapshot(
        self,
        snapshot: Mapping[
            str,
            tuple[
                typing.Literal["imagetool", "tool"],
                tuple[pathlib.Path, str],
                dict[str, typing.Any] | None,
            ],
        ],
    ) -> None:
        for uid, (kind, pending, attrs) in snapshot.items():
            node = self._manager._tool_graph.nodes.get(uid)
            if node is None:
                continue
            node.set_pending_workspace_payload(
                kind,
                pending[0],
                pending[1],
                payload_attrs=attrs,
            )

    def _dask_rebind_uids_after_full_save(
        self,
        *,
        backing_snapshot: Mapping[str, tuple[str, tuple[str, ...]]] | None,
        old_workspace_path: str | os.PathLike[str] | None,
        exclude_uids: Collection[str],
    ) -> frozenset[str]:
        """Return Dask nodes that still depend on data outside the document."""
        if backing_snapshot is None:
            return frozenset()
        old_path = workspace_arrays._normalized_file_path(old_workspace_path)
        rebind_uids: set[str] = set()
        for uid, (kind, source_paths) in backing_snapshot.items():
            node = self._manager._tool_graph.nodes.get(uid)
            if (
                kind != "dask"
                or uid in exclude_uids
                or node is None
                or not node.is_imagetool
                or node.imagetool is None
                or node.pending_workspace_memory_payload is not None
            ):
                continue
            uses_current_workspace_reader = (
                old_path is not None
                and bool(source_paths)
                and all(source_path == old_path for source_path in source_paths)
            )
            if not uses_current_workspace_reader:
                rebind_uids.add(uid)
        return frozenset(rebind_uids)

    def _live_imagetool_rebind_snapshot(
        self, uids: Collection[str]
    ) -> dict[str, tuple[_ManagedWindowNode, xr.DataArray, typing.Any, str]]:
        snapshot: dict[
            str, tuple[_ManagedWindowNode, xr.DataArray, typing.Any, str]
        ] = {}
        for uid in uids:
            node = self._manager._tool_graph.nodes.get(uid)
            if node is None or node.imagetool is None:
                continue
            snapshot[uid] = (
                node,
                node.slicer_area._data,
                copy.deepcopy(node.slicer_area.state),
                node.name,
            )
        return snapshot

    def _restore_live_imagetool_rebind_snapshot(
        self,
        snapshot: Mapping[
            str, tuple[_ManagedWindowNode, xr.DataArray, typing.Any, str]
        ],
    ) -> None:
        if not snapshot:
            return
        with self._workspace_load_context():
            for uid, (node, data, state, name) in snapshot.items():
                if uid not in self._manager._tool_graph.nodes or node.imagetool is None:
                    continue
                node.slicer_area.set_data(data, auto_compute=False)
                node.slicer_area.state = state
                node._set_name(name, manual=False)

    @staticmethod
    def _workspace_tool_references_include_uids(
        references: Mapping[str, Mapping[str, typing.Any]],
        uids: Collection[str],
        *,
        parent_uid: str | None,
    ) -> bool:
        if not uids:
            return False
        for reference in references.values():
            kind = reference.get("kind")
            if kind == "parent_source":
                if parent_uid in uids:
                    return True
            elif kind == "manager_node" and reference.get("node_uid") in uids:
                return True
        return False

    def _refresh_workspace_tool_data_after_full_save(
        self,
        old_workspace_path: str | os.PathLike[str] | None,
        workspace_path: str | os.PathLike[str],
        *,
        exclude_data_uids: Collection[str],
        source_snapshot: list[tuple[xr.Variable, bool, typing.Any]],
    ) -> None:
        """Rebind external Dask tool data and retarget saved references."""
        old_path = workspace_arrays._normalized_file_path(old_workspace_path)
        new_path = workspace_arrays._normalized_file_path(workspace_path)
        if new_path is None or old_path == new_path:
            return
        restore_entries: list[
            tuple[
                _ManagedWindowNode,
                typing.Any,
                xr.Dataset,
                dict[str, xr.DataArray],
                dict[tuple[pathlib.Path, str], xr.Dataset],
            ]
        ] = []
        try:
            for node in self._manager._tool_graph.nodes.values():
                tool = node.tool_window
                if (
                    node.is_imagetool
                    or tool is None
                    or not tool.can_save_and_load()
                    or node.uid in exclude_data_uids
                    or self._workspace_tool_references_include_uids(
                        node._workspace_tool_data_references,
                        exclude_data_uids,
                        parent_uid=node.parent_uid,
                    )
                ):
                    continue
                original_data_items = {
                    name: data.copy(deep=False)
                    for name, data in tool._persistence_data_items().items()
                }
                rebind_names: set[str] = set()
                retargets_current_workspace = False
                for name, data in original_data_items.items():
                    source_paths = workspace_arrays.dataarray_source_paths(data)
                    if old_path is not None and old_path in source_paths:
                        retargets_current_workspace = True
                    uses_current_workspace_reader = (
                        old_path is not None
                        and bool(source_paths)
                        and all(source_path == old_path for source_path in source_paths)
                    )
                    if data.chunks is not None and not uses_current_workspace_reader:
                        rebind_names.add(name)
                if (
                    not node._workspace_reference_datasets
                    and not rebind_names
                    and not retargets_current_workspace
                ):
                    continue

                restore_ds = tool.to_dataset()
                data_items = {
                    name: data.copy(deep=False)
                    for name, data in original_data_items.items()
                }
                original_reference_datasets = dict(node._workspace_reference_datasets)
                restore_entries.append(
                    (
                        node,
                        tool,
                        restore_ds,
                        original_data_items,
                        original_reference_datasets,
                    )
                )
                reference_datasets = dict(original_reference_datasets)
                opened: xr.Dataset | None = None
                try:
                    replace_ds = restore_ds
                    if rebind_names:
                        opened = workspace_arrays.open_workspace_dataset(
                            workspace_path,
                            self.loading._workspace_payload_path_for_uid(
                                workspace_path, node.uid
                            ),
                            chunks={},
                        )
                        replace_ds = workspace_format._restore_workspace_dataset_attrs(
                            opened
                        )
                        replace_ds = imagetool_serialization.restore_private_coords(
                            replace_ds, erlab.interactive.utils._SAVED_TOOL_DATA_NAME
                        )
                        source_parent_data, reference_resolver = (
                            self.loading._workspace_tool_restore_references(
                                replace_ds,
                                parent_target=node.parent_uid,
                                owner_node=node,
                                reference_datasets=reference_datasets,
                            )
                        )
                        rebound_items = type(tool)._tool_data_items_from_dataset(
                            replace_ds,
                            source_parent_data=source_parent_data,
                            reference_resolver=reference_resolver,
                            document_trust=self._manager._workspace_state.code_trust,
                            entry_locator=functools.partial(
                                self._locate_tool_code_trust_entries,
                                location_getter=functools.partial(
                                    self.saving._workspace_node_path_for_node,
                                    node,
                                ),
                            ),
                            variable_names=rebind_names,
                        )
                        for name in rebind_names:
                            data_items[name] = rebound_items[name]
                    for data in data_items.values():
                        workspace_arrays.set_workspace_xarray_sources(
                            data, workspace_path
                        )
                    with self._workspace_load_context(), tool._history_suppressed():
                        tool._replace_persistence_data_items(data_items, replace_ds)

                    retargeted_reference_datasets: dict[
                        tuple[pathlib.Path, str], xr.Dataset
                    ] = {}
                    for (
                        source_path,
                        group_path,
                    ), reference_ds in reference_datasets.items():
                        source_snapshot.extend(
                            workspace_arrays._workspace_xarray_source_snapshot(
                                reference_ds
                            )
                        )
                        if old_path is None:
                            workspace_arrays.set_workspace_xarray_sources(
                                reference_ds, new_path
                            )
                        else:
                            workspace_arrays.retarget_workspace_xarray_sources(
                                reference_ds, old_path, new_path
                            )
                        if old_path is None or (
                            workspace_arrays._normalized_file_path(source_path)
                            == old_path
                        ):
                            source_path = pathlib.Path(new_path)
                        retargeted_reference_datasets[(source_path, group_path)] = (
                            reference_ds
                        )
                    node._replace_workspace_reference_datasets(
                        retargeted_reference_datasets
                    )
                except Exception:
                    original_dataset_ids = {
                        id(dataset) for dataset in original_reference_datasets.values()
                    }
                    for dataset in reference_datasets.values():
                        if id(dataset) not in original_dataset_ids:
                            with contextlib.suppress(Exception):
                                dataset.close()
                    raise
                finally:
                    if opened is not None:
                        opened.close()
        except Exception as exc:
            for (
                restore_node,
                restore_tool,
                restore_ds,
                restore_data_items,
                restore_reference_datasets,
            ) in reversed(restore_entries):
                with contextlib.suppress(Exception):
                    restore_node._replace_workspace_reference_datasets(
                        restore_reference_datasets
                    )
                with (
                    contextlib.suppress(Exception),
                    self._workspace_load_context(),
                    restore_tool._history_suppressed(),
                ):
                    restore_tool._replace_persistence_data_items(
                        restore_data_items, restore_ds
                    )
            raise _WorkspacePostSaveBindingError(
                "Workspace file was saved, but live ToolWindow references could "
                "not be updated."
            ) from exc

    def _refresh_workspace_payload_bindings_after_full_save(
        self,
        workspace_path: str | os.PathLike[str],
        *,
        backing_snapshot: Mapping[str, tuple[str, tuple[str, ...]]] | None = None,
        old_workspace_path: str | os.PathLike[str] | None = None,
        skip_live_data_rebind_uids: Collection[str] = frozenset(),
    ) -> None:
        pending_snapshot = self._pending_workspace_payload_snapshot()
        live_imagetool_snapshot: dict[
            str, tuple[_ManagedWindowNode, xr.DataArray, typing.Any, str]
        ] = {}
        source_snapshot: list[tuple[xr.Variable, bool, typing.Any]] = []
        try:
            dask_rebind_uids = self._dask_rebind_uids_after_full_save(
                backing_snapshot=backing_snapshot,
                old_workspace_path=old_workspace_path,
                exclude_uids=skip_live_data_rebind_uids,
            )
            live_imagetool_snapshot = self._live_imagetool_rebind_snapshot(
                dask_rebind_uids
            )
            if dask_rebind_uids:
                self.loading._rebind_workspace_backed_imagetools(
                    workspace_path,
                    targets=dask_rebind_uids,
                    backing_snapshot=backing_snapshot,
                    old_workspace_path=old_workspace_path,
                    exclude_uids=skip_live_data_rebind_uids,
                )
            old_path = workspace_arrays._normalized_file_path(old_workspace_path)
            for uid, node in self._manager._tool_graph.nodes.items():
                if (
                    uid not in skip_live_data_rebind_uids
                    and node.is_imagetool
                    and node.imagetool is not None
                    and node.pending_workspace_memory_payload is None
                    and backing_snapshot is not None
                ):
                    backing = backing_snapshot.get(uid)
                    if backing is not None:
                        kind, source_paths = backing
                        should_retarget = uid not in dask_rebind_uids and (
                            kind == "dask"
                            or (
                                kind == "file_lazy"
                                and old_path is not None
                                and old_path in source_paths
                            )
                        )
                        if should_retarget:
                            source_snapshot.extend(
                                workspace_arrays._workspace_xarray_source_snapshot(
                                    node.slicer_area._data
                                )
                            )
                            workspace_arrays.set_workspace_xarray_sources(
                                node.slicer_area._data, workspace_path
                            )
            self._refresh_workspace_tool_data_after_full_save(
                old_workspace_path,
                workspace_path,
                exclude_data_uids=skip_live_data_rebind_uids,
                source_snapshot=source_snapshot,
            )
        except _WorkspacePostSaveBindingError:
            with contextlib.suppress(Exception):
                self._restore_pending_workspace_payload_snapshot(pending_snapshot)
            workspace_arrays._restore_workspace_xarray_source_snapshot(source_snapshot)
            with contextlib.suppress(Exception):
                self._restore_live_imagetool_rebind_snapshot(live_imagetool_snapshot)
            raise
        except Exception as exc:
            with contextlib.suppress(Exception):
                self._restore_pending_workspace_payload_snapshot(pending_snapshot)
            workspace_arrays._restore_workspace_xarray_source_snapshot(source_snapshot)
            with contextlib.suppress(Exception):
                self._restore_live_imagetool_rebind_snapshot(live_imagetool_snapshot)
            raise _WorkspacePostSaveBindingError(
                "Workspace file was saved, but live workspace data could not be "
                "rebound to the saved file."
            ) from exc

    def _imported_workspace_backing_snapshot(
        self,
    ) -> dict[str, tuple[str, tuple[str, ...]]] | None:
        """Capture data backing only when imported source ownership is active."""
        if not self._imported_workspace_accesses:
            return None
        return self.loading._workspace_data_backing_snapshot()

    def _rebind_imported_workspace_imagetools_after_save(
        self,
        workspace_path: str | os.PathLike[str],
        backing_snapshot: Mapping[str, tuple[str, tuple[str, ...]]] | None,
        *,
        exclude_uids: Collection[str],
    ) -> None:
        """Move imported lazy readers to objects committed by the current save."""
        if backing_snapshot is None:
            self._release_unused_imported_workspace_accesses()
            return
        imported_uids: set[str] = set().union(
            *self._imported_workspace_dependents.values()
        )
        dask_targets = {
            uid
            for uid, (kind, _source_paths) in backing_snapshot.items()
            if kind == "dask" and uid in imported_uids and uid not in exclude_uids
        }
        file_lazy_targets = {
            uid
            for uid, (kind, _source_paths) in backing_snapshot.items()
            if kind == "file_lazy" and uid in imported_uids and uid not in exclude_uids
        }
        targets = dask_targets | file_lazy_targets
        live_snapshot = self._live_imagetool_rebind_snapshot(targets)
        try:
            if dask_targets:
                self.loading._rebind_workspace_backed_imagetools(
                    workspace_path,
                    targets=dask_targets,
                    chunks={},
                    exclude_uids=exclude_uids,
                )
            if file_lazy_targets:
                self.loading._rebind_workspace_backed_imagetools(
                    workspace_path,
                    targets=file_lazy_targets,
                    chunks=None,
                    exclude_uids=exclude_uids,
                )
        except Exception as exc:
            with contextlib.suppress(Exception):
                self._restore_live_imagetool_rebind_snapshot(live_snapshot)
            raise _WorkspacePostSaveBindingError(
                "Workspace file was saved, but imported lazy data could not be "
                "rebound to the saved file."
            ) from exc
        self._release_unused_imported_workspace_accesses()

    def _active_managed_window(self) -> QtWidgets.QWidget | None:
        active_window = QtWidgets.QApplication.activeWindow()
        if not isinstance(active_window, QtWidgets.QWidget):
            return None
        if self._manager._node_uid_from_window(active_window) is None:
            return None
        if not erlab.interactive.utils.qt_is_valid(active_window):
            return None
        return active_window

    def _restore_focus_after_workspace_save(
        self, origin: QtWidgets.QWidget | None
    ) -> None:
        if (
            origin is None
            or not erlab.interactive.utils.qt_is_valid(origin)
            or not origin.isVisible()
        ):
            return
        active_window = QtWidgets.QApplication.activeWindow()
        if isinstance(active_window, QtWidgets.QWidget) and active_window not in (
            self._manager,
            origin,
        ):
            return
        origin.activateWindow()
        origin.raise_()
        focus_widget = origin.focusWidget()
        if isinstance(
            focus_widget, QtWidgets.QWidget
        ) and erlab.interactive.utils.qt_is_valid(focus_widget):
            focus_widget.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)

    def _dirty_details_text(self) -> str:
        def _node_names(uids: set[str]) -> tuple[str, ...]:
            return tuple(
                self._manager._tool_graph.nodes[uid].display_text
                for uid in sorted(uids)
                if uid in self._manager._tool_graph.nodes
            )

        sections = (
            ("Added", _node_names(self._manager._workspace_state.dirty_added)),
            (
                "Removed",
                tuple(dict.fromkeys(self._manager._workspace_state.dirty_removed)),
            ),
            (
                "Data modified",
                _node_names(
                    self._manager._workspace_state.dirty_data
                    - self._manager._workspace_state.dirty_added
                ),
            ),
            (
                "State modified",
                _node_names(
                    self._manager._workspace_state.dirty_state
                    - self._manager._workspace_state.dirty_data
                    - self._manager._workspace_state.dirty_added
                ),
            ),
            (
                "Structure modified",
                tuple(dict.fromkeys(self._manager._workspace_state.structure_reasons)),
            ),
            (
                "Layout modified",
                ("Manager window layout",)
                if self._manager._workspace_state.layout_modified
                else (),
            ),
            (
                "Settings modified",
                ("Workspace settings",)
                if self._manager._workspace_state.options_modified
                else (),
            ),
            (
                "Acquisition context modified",
                ("Acquisition context",)
                if self._manager._workspace_state.context_modified
                else (),
            ),
        )
        blocks: list[str] = []
        for label, items in sections:
            if items:
                blocks.append(f"{label}:\n" + "\n".join(f"- {item}" for item in items))
        return "\n\n".join(blocks)

    def _set_node_window_modified(self, uid: str, modified: bool) -> None:
        self._pending_node_window_modified.pop(uid, None)
        self._apply_node_window_modified(uid, modified)

    def _queue_node_window_modified(self, uid: str, modified: bool) -> None:
        self._pending_node_window_modified[uid] = modified
        self._manager._queue_idle_work(
            ("node-window", uid),
            functools.partial(self._flush_pending_node_window_modified, uid),
        )

    def _flush_pending_node_window_modified(self, uid: str) -> None:
        try:
            modified = self._pending_node_window_modified.pop(uid)
        except KeyError:
            return
        self._apply_node_window_modified(uid, modified)

    def _apply_node_window_modified(self, uid: str, modified: bool) -> None:
        node = self._manager._tool_graph.nodes.get(uid)
        if node is None:
            self._node_window_state_applied.pop(uid, None)
            return
        window = node.window
        if node.tool_window is not None:
            display_name = node.tool_window._tool_display_name
            base_title = (
                f"{node.tool_window.tool_name}: {display_name}"
                if display_name
                else node.tool_window.tool_name
            )
        else:
            base_title = node.label_text
        windows: list[tuple[QtWidgets.QWidget | None, str]] = [(window, base_title)]
        if node.tool_window is not None:
            windows.extend(node.tool_window._managed_secondary_windows())
        valid_windows: list[tuple[QtWidgets.QWidget, str]] = []
        for target_window, target_title in windows:
            if target_window is None or not erlab.interactive.utils.qt_is_valid(
                target_window
            ):
                continue
            valid_windows.append((target_window, target_title))
        target_state = (
            tuple(
                (id(target_window), target_title)
                for target_window, target_title in valid_windows
            ),
            modified,
        )
        if self._node_window_state_applied.get(uid) == target_state and all(
            target_window.windowTitle()
            == _window_title_with_modified_placeholder(target_title)
            for target_window, target_title in valid_windows
        ):
            return
        for target_window, target_title in valid_windows:
            title = _window_title_with_modified_placeholder(target_title)
            if title != target_window.windowTitle():
                target_window.setWindowTitle(title)
            target_window.setWindowModified(modified)
        self._node_window_state_applied[uid] = target_state

    def _apply_workspace_dirty_event(
        self, event: workspace_state._WorkspaceDirtyEvent
    ) -> bool:
        if event.uid is not None and (event.added or event.data or event.state):
            self._set_node_window_modified(event.uid, True)
        return self._manager._workspace_state.apply_dirty_event(event)

    def _mark_workspace_dirty(
        self,
        *,
        uid: str | None = None,
        data: bool = False,
        state: bool = False,
        added: bool = False,
        removed: str | None = None,
        structure: str | None = None,
    ) -> bool:
        if (
            self._manager._workspace_state.loading_depth > 0
            or self._manager._workspace_state.saving_depth > 0
        ):
            return False
        event = workspace_state._WorkspaceDirtyEvent(
            generation=self._manager._workspace_state.dirty_generation + 1,
            uid=uid,
            data=data,
            state=state,
            added=added,
            removed=removed,
            structure=structure,
        )
        was_modified = self._manager.is_workspace_modified
        node_was_modified = event.uid is not None and event.uid in (
            self._manager._workspace_state.dirty_added
            | self._manager._workspace_state.dirty_data
            | self._manager._workspace_state.dirty_state
        )
        if (
            event.uid is not None
            and (event.added or event.data or event.state)
            and not node_was_modified
        ):
            self._queue_node_window_modified(event.uid, True)
        dirty_changed = self._manager._workspace_state.mark_dirty(event)
        if not was_modified and self._manager.is_workspace_modified:
            self._update_workspace_window_title(force=False)
        return dirty_changed

    def _mark_node_added(self, uid: str) -> bool:
        changed = self._mark_workspace_dirty(
            uid=uid, added=True, structure="Added window"
        )
        self._bind_current_workspace_manifest_if_review_needed()
        return changed

    def _mark_node_data_dirty(self, uid: str) -> bool:
        changed = self._mark_workspace_dirty(uid=uid, data=True)
        if self._manager._workspace_state.loading_depth == 0:
            self._release_unused_imported_workspace_accesses()
        return changed

    def _mark_node_state_dirty(self, uid: str) -> bool:
        return self._mark_workspace_dirty(uid=uid, state=True)

    def _mark_tool_info_dirty(self, uid: str) -> bool:
        if uid not in self._manager._workspace_state.dirty_state:
            return self._mark_node_state_dirty(uid)
        return False

    def _mark_workspace_structure_dirty(self, reason: str) -> bool:
        return self._mark_workspace_dirty(structure=reason)

    def _mark_workspace_layout_dirty(self) -> None:
        if (
            not getattr(self._manager, "_manager_layout_tracking_enabled", False)
            or self._manager._workspace_state.path is None
            or self._manager._workspace_state.loading_depth > 0
            or self._manager._workspace_state.saving_depth > 0
            or self._manager._workspace_state.closing_document
        ):
            return
        if self._manager._workspace_state.mark_layout_dirty():
            self._update_workspace_window_title(force=False)

    def _mark_workspace_options_dirty(self) -> None:
        if (
            self._manager._workspace_state.loading_depth > 0
            or self._manager._workspace_state.saving_depth > 0
            or self._manager._workspace_state.closing_document
        ):
            return
        if self._manager._workspace_state.mark_options_dirty():
            self._update_workspace_window_title(force=False)

    def _mark_workspace_context_dirty(self) -> None:
        if (
            self._manager._workspace_state.loading_depth > 0
            or self._manager._workspace_state.saving_depth > 0
            or self._manager._workspace_state.closing_document
        ):
            return
        if self._manager._workspace_state.mark_context_dirty():
            self._update_workspace_window_title(force=False)

    def _mark_workspace_clean(self) -> None:
        self._manager._workspace_state.mark_clean()
        for uid in tuple(self._manager._tool_graph.nodes):
            self._set_node_window_modified(uid, False)
        self._update_workspace_window_title()

    def _restore_workspace_dirty_events(
        self, events: Iterable[workspace_state._WorkspaceDirtyEvent]
    ) -> None:
        retained_events = list(events)
        self._manager._workspace_state.mark_clean()
        for uid in tuple(self._manager._tool_graph.nodes):
            self._set_node_window_modified(uid, False)
        for event in retained_events:
            self._apply_workspace_dirty_event(event)
        self._manager._workspace_state.dirty_events = retained_events
        self._update_workspace_window_title()

    @contextlib.contextmanager
    def _workspace_load_context(self) -> Iterator[None]:
        with (
            self._manager._workspace_state.load_context(),
            self._manager._workspace_ui_refresh_context(),
        ):
            yield

    def _send_workspace_posted_events(self, event_type: QtCore.QEvent.Type) -> None:
        for _ in range(3):
            QtWidgets.QApplication.sendPostedEvents(None, int(event_type.value))

    def _drain_workspace_restore_events(self) -> None:
        self._send_workspace_posted_events(QtCore.QEvent.Type.MetaCall)
        self._send_workspace_posted_events(QtCore.QEvent.Type.DeferredDelete)

    def _drain_workspace_deferred_events(self) -> None:
        self._send_workspace_posted_events(QtCore.QEvent.Type.MetaCall)
        for _ in range(3):
            QtWidgets.QApplication.processEvents()
        self._manager._flush_idle_work(force=True)
        self._send_workspace_posted_events(QtCore.QEvent.Type.MetaCall)
        for _ in range(3):
            QtWidgets.QApplication.processEvents()

    def _workspace_state_snapshot(self) -> _WorkspaceStateSnapshot:
        return self._manager._workspace_state.snapshot(
            node_uid_counter=self._manager._tool_graph.uid_counter
        )

    def _install_workspace_save_shortcut(self, widget: QtWidgets.QWidget) -> None:
        for shortcut in widget.findChildren(QtWidgets.QShortcut):
            if shortcut.objectName() == _WORKSPACE_SAVE_SHORTCUT_OBJECT_NAME:
                return
        shortcut = QtWidgets.QShortcut(_shortcut_sequences.WORKSPACE_SAVE, widget)
        shortcut.setObjectName(_WORKSPACE_SAVE_SHORTCUT_OBJECT_NAME)
        shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        shortcut.activated.connect(self._manager.save)

    @staticmethod
    def _validated_standalone_app_state(
        key: str, state: Mapping[str, typing.Any]
    ) -> dict[str, typing.Any] | None:
        model_type: typing.Any
        if key == "explorer":
            from erlab.interactive.explorer._tabbed_explorer import DataExplorerState

            model_type = DataExplorerState
        elif key == "ptable":
            from erlab.interactive.ptable._window import PeriodicTableState

            model_type = PeriodicTableState
        else:
            return None
        try:
            return typing.cast(
                "dict[str, typing.Any]",
                model_type.model_validate(state).model_dump(
                    mode="json", exclude_none=True
                ),
            )
        except Exception:
            logger.warning(
                "Ignoring invalid %s standalone app state", key, exc_info=True
            )
            return None

    def _workspace_save_dialog(
        self,
        *,
        native: bool = True,
        caption: str = "Save Workspace",
        selected_file: str | os.PathLike[str] | None = None,
    ) -> str | None:
        dialog = QtWidgets.QFileDialog(self._manager, caption)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.AnyFile)
        dialog.setNameFilter("ImageTool Workspace Files (*.itws)")
        dialog.setDefaultSuffix("itws")
        if selected_file is not None:
            dialog.selectFile(str(selected_file))
        elif self._manager._workspace_state.path is not None:
            dialog.selectFile(str(self._manager._workspace_state.path))
        elif (directory := self._manager._recent_or_default_directory()) is not None:
            dialog.setDirectory(directory)
        if not native:  # pragma: no branch
            dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)

        if not dialog.exec():
            return None
        return dialog.selectedFiles()[0]

    def _dirty_workspace_save_choice(self, action_text: str) -> str:
        if not self._manager.is_workspace_modified:
            return "clean"

        msg_box = QtWidgets.QMessageBox(self._manager)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setText("Save changes to this workspace?")
        msg_box.setInformativeText(action_text)
        details = self._dirty_details_text()
        if details:
            msg_box.setDetailedText(details)
        msg_box.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Save
            | QtWidgets.QMessageBox.StandardButton.Discard
            | QtWidgets.QMessageBox.StandardButton.Cancel
        )
        msg_box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Save)
        result = msg_box.exec()
        if result == QtWidgets.QMessageBox.StandardButton.Save:
            return "save"
        if result == QtWidgets.QMessageBox.StandardButton.Discard:
            return "discard"
        return "cancel"

    def _run_after_dirty_workspace_saved_or_discarded(
        self,
        action_text: str,
        continuation: Callable[[], bool | None],
        *,
        native: bool = True,
    ) -> bool:
        choice = self._dirty_workspace_save_choice(action_text)
        if choice == "cancel":
            return False
        if choice in {"clean", "discard"}:
            return bool(continuation())

        def _continue_after_save(save_succeeded: bool) -> None:
            if save_succeeded and not self._manager.is_workspace_modified:
                continuation()

        return self.save(native=native, on_finished=_continue_after_save)

    def _show_legacy_workspace_upgrade_message(
        self, fname: str | os.PathLike[str]
    ) -> None:
        msg_box = QtWidgets.QMessageBox(self._manager)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Information)
        msg_box.setWindowTitle("Save Legacy Workspace")
        msg_box.setText("This workspace uses a legacy file format.")
        msg_box.setInformativeText(
            "Save it as an .itws file so ImageTool Manager can update it safely."
        )
        msg_box.setDetailedText(str(pathlib.Path(fname)))
        msg_box.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
        msg_box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Ok)
        msg_box.exec()

    def _save_legacy_workspace_as_current(
        self,
        fname: str | os.PathLike[str],
        *,
        native: bool = True,
        existing_access: _WorkspaceDocumentAccess | None = None,
    ) -> tuple[str, QtCore.QLockFile | None] | None:
        self._show_legacy_workspace_upgrade_message(fname)
        converted_fname = self._workspace_save_dialog(
            native=native,
            caption="Save Converted Workspace",
            selected_file=fname,
        )
        if converted_fname is None:
            return None
        converted_path = pathlib.Path(converted_fname).resolve()
        if not workspace_format._workspace_path_is_itws(converted_path):
            _show_itws_workspace_warning(self._manager)
            return None
        if existing_access is not None and converted_path == existing_access.path:
            with erlab.interactive.utils.wait_dialog(
                self._manager, "Saving workspace..."
            ):
                self.saving._save_workspace_document(
                    existing_access.path,
                    document_access=existing_access,
                )
            return str(existing_access.path), self._take_workspace_access_lock(
                existing_access
            )

        with self._workspace_document_access_context(converted_fname) as access:
            with erlab.interactive.utils.wait_dialog(
                self._manager, "Saving workspace..."
            ):
                self.saving._save_workspace_document(
                    access.path,
                    document_access=access,
                )
            return str(access.path), access.take_lock()

    def _associate_loaded_workspace_file(
        self,
        fname: str | os.PathLike[str],
        schema_version: int,
        *,
        native: bool = True,
        workspace_access: _WorkspaceDocumentAccess | None = None,
        rebind_data: bool = True,
    ) -> None:
        associated_fname = fname
        associated_lock: QtCore.QLockFile | None = None
        if workspace_format._workspace_schema_requires_conversion(schema_version):
            converted = self._save_legacy_workspace_as_current(
                fname, native=native, existing_access=workspace_access
            )
            if converted is None:
                self._set_workspace_path(None)
                self._mark_workspace_structure_dirty(
                    "Legacy workspace needs conversion"
                )
                return
            associated_fname, associated_lock = converted
            schema_version = workspace_format._current_workspace_schema_version()
        elif workspace_access is not None:
            associated_lock = self._take_workspace_access_lock(workspace_access)

        associated_store = None
        if schema_version >= workspace_format._WORKSPACE_LEGACY_SCHEMA_VERSION:
            associated_store = workspace_store.WorkspaceStore.active(associated_fname)
            if associated_store is None:
                associated_store = workspace_store.WorkspaceStore(associated_fname)
            if (
                workspace_format._workspace_schema_uses_immutable_generations(
                    schema_version
                )
                and not self._manager._workspace_state.save_as_only
            ):
                associated_store.clear_staging()

        self._set_workspace_path(
            associated_fname,
            workspace_lock=associated_lock,
            store=associated_store,
        )
        self._manager._workspace_state.schema_version = schema_version
        if rebind_data:
            self.loading._rebind_workspace_backed_imagetools(associated_fname)
        self._drain_workspace_restore_events()
        self._mark_workspace_clean()
        self._record_recent_workspace(associated_fname)

    def offload_to_workspace(
        self, targets: Iterable[int | str], *, native: bool = True
    ) -> bool:
        """Replace selected in-memory ImageTools with dask-backed workspace data.

        .. versionadded:: 3.23.0
        """
        if self._manager._workspace_state.save_in_progress:
            self._manager._status_bar.showMessage(
                "Workspace save in progress; offload after it finishes", 3000
            )
            return False
        offload_targets: list[int | str] = []
        for target in targets:
            node = self._manager._node_for_target(target)
            if (
                node.is_imagetool
                and node.imagetool is not None
                and not node.slicer_area.data_chunked
                and node.pending_workspace_memory_payload is None
            ):
                offload_targets.append(target)
        if not offload_targets:
            return False

        def _offload_after_save(save_succeeded: bool) -> None:
            if not save_succeeded or self._manager.is_workspace_modified:
                return
            if self._manager._workspace_state.path is None:
                return
            self._offload_targets_to_current_workspace(offload_targets)

        state = self._manager._workspace_state
        if state.path is None:
            return self.save_as(native=native, on_finished=_offload_after_save)
        if state.save_as_only:
            return self.save_as(native=native, on_finished=_offload_after_save)
        if (
            self._manager.is_workspace_modified
            or state.schema_version
            < workspace_format._current_workspace_schema_version()
        ):
            return self.save(native=native, on_finished=_offload_after_save)
        return self._offload_targets_to_current_workspace(offload_targets)

    def _offload_targets_to_current_workspace(
        self, offload_targets: Iterable[int | str]
    ) -> bool:
        workspace_path = self._manager._workspace_state.path
        if workspace_path is None:
            return False

        origin = self._active_managed_window()
        try:
            with erlab.interactive.utils.wait_dialog(
                origin or self._manager, "Offloading to workspace..."
            ):
                self.loading._rebind_workspace_backed_imagetools(
                    workspace_path,
                    targets=offload_targets,
                    chunks={},
                )
                self.saving._save_workspace_document(
                    workspace_path,
                    mark_clean=False,
                )
            self._manager._status_bar.showMessage("Data offloaded to workspace", 5000)
            self._schedule_workspace_gc()
        except Exception:
            logger.exception(
                "Could not offload data to the current workspace",
                extra={"suppress_ui_alert": True},
            )
            self._manager._show_operation_error(
                "Error while offloading to workspace",
                "An error occurred while reconnecting data from the workspace file.",
            )
            self._restore_focus_after_workspace_save(origin)
            return False

        self._restore_focus_after_workspace_save(origin)
        self._manager._update_actions()
        self._manager._update_info()
        return True

    def _set_workspace_save_actions_enabled(self, enabled: bool) -> tuple[bool, ...]:
        open_recent_action = self._manager.open_recent_menu.menuAction()
        previous = (
            self._manager.save_action.isEnabled(),
            self._manager.save_as_action.isEnabled(),
            self._manager.compact_workspace_action.isEnabled(),
            self._manager.load_action.isEnabled(),
            False if open_recent_action is None else open_recent_action.isEnabled(),
            self._manager.offload_action.isEnabled(),
            self._manager.import_workspace_action.isEnabled(),
        )
        self._manager.save_action.setEnabled(enabled and previous[0])
        self._manager.save_as_action.setEnabled(enabled and previous[1])
        self._manager.compact_workspace_action.setEnabled(enabled and previous[2])
        self._manager.load_action.setEnabled(enabled and previous[3])
        if open_recent_action is not None:
            open_recent_action.setEnabled(enabled and previous[4])
        self._manager.offload_action.setEnabled(enabled and previous[5])
        self._manager.import_workspace_action.setEnabled(enabled and previous[6])
        return previous

    def _restore_workspace_save_actions_enabled(
        self, previous: tuple[bool, ...]
    ) -> None:
        open_recent_action = self._manager.open_recent_menu.menuAction()
        self._manager.save_action.setEnabled(previous[0])
        self._manager.save_as_action.setEnabled(previous[1])
        self._manager.compact_workspace_action.setEnabled(previous[2])
        self._manager.load_action.setEnabled(previous[3])
        if open_recent_action is not None:
            open_recent_action.setEnabled(previous[4])
        self._manager.offload_action.setEnabled(previous[5])
        self._manager.import_workspace_action.setEnabled(previous[6])

    def _start_workspace_save_worker(
        self,
        fname: str | os.PathLike[str],
        snapshot: workspace_saving._WorkspaceSaveSnapshot,
        *,
        on_finished: Callable[
            [float, workspace_saving._WorkspaceSaveError | None], None
        ],
        on_start_error: Callable[[], None] | None = None,
    ) -> bool:
        thread_pool = QtCore.QThreadPool.globalInstance()
        if thread_pool is None:
            snapshot.close()
            if on_start_error is not None:
                on_start_error()
            return False

        target_path = pathlib.Path(fname).resolve()
        store = (
            self._workspace_store
            if self._workspace_store is not None
            and not self._workspace_store.closed
            and self._workspace_store.path == target_path
            else None
        )
        worker = workspace_saving._WorkspaceSaveWorker(
            fname,
            snapshot,
            store=store,
            reader_closers=self.saving._workspace_reader_closers(target_path),
        )
        previous_action_states = self._set_workspace_save_actions_enabled(False)

        def _finish(
            elapsed: float,
            error: workspace_saving._WorkspaceSaveError | None,
        ) -> None:
            self._manager._workspace_state.save_in_progress = False
            self._restore_workspace_save_actions_enabled(previous_action_states)
            self._manager._update_actions()
            receiver = self._background_save_receiver
            self._background_save_receiver = None
            self._background_save_worker = None
            if receiver is not None:
                receiver.deleteLater()
            try:
                on_finished(elapsed, error)
            except Exception:
                logger.exception(
                    "Error while finishing workspace save",
                    extra={"suppress_ui_alert": True},
                )
                self._manager._status_bar.clearMessage()
                self._manager._show_operation_error(
                    "Error while saving workspace",
                    "An error occurred while saving the workspace file.",
                )

        receiver = workspace_saving._WorkspaceSaveResultReceiver(
            callback=_finish,
            waiting_callback=lambda: self._manager._status_bar.showMessage(
                "Waiting for active workspace computations..."
            ),
            parent=self._manager,
        )
        worker.signals.waiting.connect(receiver.wait)
        worker.signals.finished.connect(receiver.finish)
        self._manager._workspace_state.save_in_progress = True
        self._background_save_worker = worker
        self._background_save_receiver = receiver
        try:
            thread_pool.start(worker)
        except Exception:
            self._manager._workspace_state.save_in_progress = False
            self._restore_workspace_save_actions_enabled(previous_action_states)
            self._background_save_worker = None
            self._background_save_receiver = None
            receiver.deleteLater()
            snapshot.close()
            if on_start_error is not None:
                on_start_error()
            return False
        return True

    def _schedule_workspace_gc(self) -> None:
        """Schedule bounded cleanup after the current save has finished."""
        self._workspace_gc_requested = True
        QtCore.QTimer.singleShot(0, self._start_workspace_gc)

    def _start_workspace_gc(self) -> None:
        if (
            not self._workspace_gc_requested
            or self._workspace_gc_worker is not None
            or self._manager._workspace_state.save_in_progress
        ):
            return
        store = self._workspace_store
        workspace_path = self._current_workspace_document_path()
        if (
            store is None
            or store.closed
            or workspace_path is None
            or store.path != workspace_path
        ):
            self._workspace_gc_requested = False
            return
        thread_pool = QtCore.QThreadPool.globalInstance()
        if thread_pool is None:
            return

        document_id = self._manager._workspace_state.document_id
        worker = workspace_saving._WorkspaceGcWorker(
            store,
            reader_closers=self.saving._workspace_reader_closers(workspace_path),
        )

        def _finish(more: bool, error: str | None) -> None:
            receiver = self._workspace_gc_receiver
            self._workspace_gc_receiver = None
            self._workspace_gc_worker = None
            if receiver is not None:
                receiver.deleteLater()
            if error is not None:
                logger.warning(
                    "Workspace cleanup failed:\n%s",
                    error,
                    extra={"suppress_ui_alert": True},
                )
                self._workspace_gc_requested = False
                return
            if self._manager._workspace_state.document_id != document_id:
                self._workspace_gc_requested = False
                return
            self._workspace_gc_requested = more
            if more:
                QtCore.QTimer.singleShot(0, self._start_workspace_gc)

        receiver = workspace_saving._WorkspaceGcResultReceiver(
            _finish,
            parent=self._manager,
        )
        worker.signals.finished.connect(receiver.finish)
        self._workspace_gc_requested = False
        self._workspace_gc_worker = worker
        self._workspace_gc_receiver = receiver
        try:
            thread_pool.start(worker)
        except Exception:
            self._workspace_gc_worker = None
            self._workspace_gc_receiver = None
            receiver.deleteLater()
            self._workspace_gc_requested = True
            logger.exception(
                "Could not start workspace cleanup",
                extra={"suppress_ui_alert": True},
            )

    def _show_workspace_post_save_binding_error(
        self, workspace_path: str | os.PathLike[str]
    ) -> None:
        self._manager._status_bar.clearMessage()
        self._manager._show_operation_error(
            "Workspace file saved but live references were not updated",
            "The workspace file was saved, but live tool data could not be "
            "updated to use the saved file. Reopen the workspace to continue "
            "from the saved version.",
        )

    def _mark_workspace_post_save_binding_refresh_failed(self) -> None:
        self._mark_workspace_structure_dirty(
            "Live workspace data references need refresh"
        )

    def _finish_workspace_save_result(
        self,
        *,
        document_id: str,
        workspace_path: pathlib.Path,
        snapshot: workspace_saving._WorkspaceSaveSnapshot,
        worker_elapsed: float,
        error: workspace_saving._WorkspaceSaveError | None,
        origin: QtWidgets.QWidget | None,
        snapshot_elapsed: float,
        started_at: float,
        restore_focus: bool,
        imported_backing_snapshot: Mapping[str, tuple[str, tuple[str, ...]]] | None,
    ) -> bool:
        total_elapsed = time.perf_counter() - started_at
        logger.debug(
            "Workspace save timing: snapshot %.3f s, write %.3f s, total %.3f s",
            snapshot_elapsed,
            worker_elapsed,
            total_elapsed,
        )
        if self._manager._workspace_state.document_id != document_id:
            logger.info(
                "Ignoring completed workspace save for inactive document: %s",
                workspace_path,
                extra={"suppress_ui_alert": True},
            )
            return False
        if error is not None:
            self._manager._status_bar.clearMessage()
            self._manager._show_workspace_save_worker_error(error)
            if restore_focus:
                self._restore_focus_after_workspace_save(origin)
            return False

        self._drain_workspace_deferred_events()
        post_save_events = tuple(
            event
            for event in self._manager._workspace_state.dirty_events
            if event.generation > snapshot.generation
        )
        has_new_dirty_generation = (
            self._manager._workspace_state.dirty_generation > snapshot.generation
            and self._manager.is_workspace_modified
        )
        self._adopt_committed_workspace_generation(
            workspace_path,
            snapshot,
            manifest=snapshot.generation_plan.manifest,
        )
        post_save_uids = frozenset(
            event.uid for event in post_save_events if event.uid is not None
        )
        try:
            self._rebind_imported_workspace_imagetools_after_save(
                workspace_path,
                imported_backing_snapshot,
                exclude_uids=post_save_uids,
            )
        except _WorkspacePostSaveBindingError:
            self._mark_workspace_post_save_binding_refresh_failed()
            self._show_workspace_post_save_binding_error(workspace_path)
            if restore_focus:
                self._restore_focus_after_workspace_save(origin)
            return False
        if post_save_events:
            self._restore_workspace_dirty_events(post_save_events)
            message = "Workspace saved; new changes remain unsaved"
        elif has_new_dirty_generation:
            message = "Workspace saved; new changes remain unsaved"
        else:
            self._mark_workspace_clean()
            message = (
                f"Workspace saved in {total_elapsed:.1f} s"
                if total_elapsed >= _WORKSPACE_SAVE_WAIT_DIALOG_THRESHOLD_SECONDS
                else "Workspace saved"
            )
        self._manager._status_bar.showMessage(message, 5000)
        self._record_saved_workspace_code_trust(
            snapshot.generation_plan.manifest,
            trusted_lineage=snapshot.trusted_lineage,
            current_document=not post_save_events and not has_new_dirty_generation,
        )
        if restore_focus:
            self._restore_focus_after_workspace_save(origin)
        self._record_recent_workspace(workspace_path)
        return True

    def _finish_background_workspace_save(
        self,
        *,
        document_id: str,
        workspace_path: pathlib.Path,
        snapshot: workspace_saving._WorkspaceSaveSnapshot,
        worker_elapsed: float,
        error: workspace_saving._WorkspaceSaveError | None,
        origin: QtWidgets.QWidget | None,
        snapshot_elapsed: float,
        started_at: float,
        restore_focus: bool,
        imported_backing_snapshot: Mapping[str, tuple[str, tuple[str, ...]]]
        | None = None,
        on_finished: Callable[[bool], None] | None = None,
    ) -> None:
        try:
            save_succeeded = self._finish_workspace_save_result(
                document_id=document_id,
                workspace_path=workspace_path,
                snapshot=snapshot,
                worker_elapsed=worker_elapsed,
                error=error,
                origin=origin,
                snapshot_elapsed=snapshot_elapsed,
                started_at=started_at,
                restore_focus=restore_focus,
                imported_backing_snapshot=imported_backing_snapshot,
            )
            queued = self._background_save_requested
            if (
                save_succeeded
                and queued
                and self._manager.is_workspace_modified
                and self._current_workspace_document_path() == workspace_path
            ):
                QtCore.QTimer.singleShot(0, self.save)
            elif save_succeeded:
                self._schedule_workspace_gc()
            if on_finished is not None:
                on_finished(save_succeeded)
        except Exception:
            logger.exception(
                "Error while finishing background workspace save",
                extra={"suppress_ui_alert": True},
            )
            self._manager._status_bar.clearMessage()
            self._manager._show_operation_error(
                "Error while saving workspace",
                "An error occurred while saving the workspace file.",
            )
            if on_finished is not None:
                on_finished(False)
        finally:
            self._background_save_requested = False

    def save(
        self,
        *,
        native: bool = True,
        on_finished: Callable[[bool], None] | None = None,
        restore_focus: bool = True,
    ) -> bool:
        """Start a non-blocking save for the current workspace document."""
        workspace_path = self._current_workspace_document_path()
        if workspace_path is None:
            return self.save_as(native=native, on_finished=on_finished)
        if self._manager._workspace_state.save_as_only:
            return self.save_as(native=native, on_finished=on_finished)
        if self._manager._workspace_state.save_in_progress:
            self._background_save_requested = True
            self._manager._status_bar.showMessage("Workspace save queued", 3000)
            return False

        origin = self._active_managed_window()
        document_id = self._manager._workspace_state.document_id
        imported_backing_snapshot = self._imported_workspace_backing_snapshot()
        self._manager._status_bar.showMessage("Saving workspace...")
        started_at = time.perf_counter()
        snapshot: workspace_saving._WorkspaceSaveSnapshot | None = None
        try:
            snapshot_started_at = time.perf_counter()
            snapshot = self.saving._workspace_save_snapshot(workspace_path)
            snapshot_elapsed = time.perf_counter() - snapshot_started_at
        except Exception:
            if snapshot is not None:
                snapshot.close()
            self._manager._status_bar.clearMessage()
            self._manager._show_operation_error(
                "Error while saving workspace",
                "An error occurred while saving the workspace file.",
            )
            if restore_focus:
                self._restore_focus_after_workspace_save(origin)
            if on_finished is not None:
                on_finished(False)
            return False
        if snapshot is None:  # pragma: no cover
            raise RuntimeError("Workspace save snapshot was not created")

        def _start_error() -> None:
            self._manager._status_bar.clearMessage()
            self._manager._show_operation_error(
                "Error while saving workspace",
                "An error occurred while saving the workspace file.",
            )
            if restore_focus:
                self._restore_focus_after_workspace_save(origin)
            if on_finished is not None:
                on_finished(False)

        self._background_save_requested = False
        return self._start_workspace_save_worker(
            workspace_path,
            snapshot,
            on_finished=lambda elapsed, error: self._finish_background_workspace_save(
                document_id=document_id,
                workspace_path=workspace_path,
                snapshot=snapshot,
                worker_elapsed=elapsed,
                error=error,
                origin=origin,
                snapshot_elapsed=snapshot_elapsed,
                started_at=started_at,
                restore_focus=restore_focus,
                imported_backing_snapshot=imported_backing_snapshot,
                on_finished=on_finished,
            ),
            on_start_error=_start_error,
        )

    def save_as(
        self,
        *,
        native: bool = True,
        on_finished: Callable[[bool], None] | None = None,
    ) -> bool:
        """Save the current workspace under a new path and bind to that path."""
        if self._manager._workspace_state.save_in_progress:
            self._manager._status_bar.showMessage(
                "Workspace save already in progress", 3000
            )
            if on_finished is not None:
                on_finished(False)
            return False
        origin = self._active_managed_window()
        fname = self._workspace_save_dialog(native=native, caption="Save Workspace As")
        if fname is None:
            if on_finished is not None:
                on_finished(False)
            return False
        if not workspace_format._workspace_path_is_itws(fname):
            _show_itws_workspace_warning(self._manager)
            if on_finished is not None:
                on_finished(False)
            return False
        if (
            self._manager._workspace_state.path is not None
            and pathlib.Path(fname).resolve() == self._manager._workspace_state.path
        ):
            if self._manager._workspace_state.save_as_only:
                self._manager._show_operation_error(
                    "Select a different workspace file",
                    "This workspace cannot overwrite the original file because "
                    "some extension content is unavailable.",
                )
                if on_finished is not None:
                    on_finished(False)
                return False
            return self.save(
                native=native,
                on_finished=on_finished,
            )
        old_workspace_path = self._manager._workspace_state.path
        document_id = self._manager._workspace_state.document_id
        backing_snapshot = self.loading._workspace_data_backing_snapshot()
        access: _WorkspaceDocumentAccess | None = None
        snapshot: workspace_saving._WorkspaceSaveSnapshot | None = None
        target_expected_state: workspace_storage._WorkspacePublicationState | None = (
            None
        )
        worker_path: pathlib.Path | None = None
        self._manager._status_bar.showMessage("Saving workspace...")
        started_at = time.perf_counter()
        try:
            access = self._workspace_document_access(fname)
            target_expected_state = workspace_storage._workspace_publication_state(
                access.path
            )
            worker_path = access.path.with_name(
                f".{access.path.name}.tmp-{uuid.uuid4().hex}"
            )
            self._drain_workspace_deferred_events()
            generation = self._manager._workspace_state.dirty_generation
            self._manager._workspace_state.saving_depth += 1
            try:
                snapshot_started_at = time.perf_counter()
                snapshot = self.saving._workspace_generation_save_snapshot(
                    generation,
                    fname=worker_path,
                )
                snapshot_elapsed = time.perf_counter() - snapshot_started_at
            finally:
                self._manager._workspace_state.saving_depth -= 1
        except Exception:
            if snapshot is not None:
                snapshot.close()
            if worker_path is not None:
                with contextlib.suppress(OSError):
                    worker_path.unlink()
            if access is not None:
                access.release()
            logger.exception(
                "Error while preparing workspace Save As snapshot",
                extra={"suppress_ui_alert": True},
            )
            self._manager._status_bar.clearMessage()
            self._manager._show_operation_error(
                "Error while saving workspace",
                "An error occurred while saving the workspace file.",
            )
            self._restore_focus_after_workspace_save(origin)
            if on_finished is not None:
                on_finished(False)
            return False
        if snapshot is None:  # pragma: no cover
            if access is not None:
                access.release()
            raise RuntimeError("Workspace save snapshot was not created")

        def _finish_save_as(
            worker_elapsed: float,
            error: workspace_saving._WorkspaceSaveError | None,
        ) -> None:
            nonlocal access
            total_elapsed = time.perf_counter() - started_at
            logger.debug(
                "Workspace save timing: snapshot %.3f s, write %.3f s, total %.3f s",
                snapshot_elapsed,
                worker_elapsed,
                total_elapsed,
            )
            if access is None:  # pragma: no cover
                if on_finished is not None:
                    on_finished(False)
                return
            if self._manager._workspace_state.document_id != document_id:
                logger.info(
                    "Ignoring completed Save As for inactive document: %s",
                    access.path,
                    extra={"suppress_ui_alert": True},
                )
                if worker_path is not None:
                    with contextlib.suppress(OSError):
                        worker_path.unlink()
                access.release()
                access = None
                if on_finished is not None:
                    on_finished(False)
                return
            if error is not None:
                if worker_path is not None:
                    with contextlib.suppress(OSError):
                        worker_path.unlink()
                self._manager._status_bar.clearMessage()
                self._manager._show_workspace_save_worker_error(error)
                access.release()
                self._restore_focus_after_workspace_save(origin)
                if on_finished is not None:
                    on_finished(False)
                return

            if target_expected_state is None:  # pragma: no cover
                raise RuntimeError("Save As publication state was not prepared")
            try:
                workspace_storage._replace_workspace_file(
                    worker_path,
                    access.path,
                    expected_state=target_expected_state,
                )
            except Exception:
                with contextlib.suppress(OSError):
                    worker_path.unlink()
                self._manager._status_bar.clearMessage()
                self._manager._show_operation_error(
                    "Error while saving workspace",
                    "The new workspace was written, but it could not replace the "
                    "selected destination.",
                )
                access.release()
                self._restore_focus_after_workspace_save(origin)
                if on_finished is not None:
                    on_finished(False)
                return

            self._drain_workspace_deferred_events()
            pre_rebind_post_save_events = tuple(
                event
                for event in self._manager._workspace_state.dirty_events
                if event.generation > snapshot.generation
            )
            post_save_uids = frozenset(
                event.uid
                for event in pre_rebind_post_save_events
                if event.uid is not None
            )
            old_store = self._workspace_store
            try:
                if old_store is not None:
                    old_store.switch_path(access.path)
                    new_store = old_store
                else:
                    new_store = workspace_store.WorkspaceStore(access.path)
            except Exception:
                logger.exception(
                    "Could not open the saved workspace store",
                    extra={"suppress_ui_alert": True},
                )
                self._show_workspace_post_save_binding_error(access.path)
                access.release()
                self._restore_focus_after_workspace_save(origin)
                if on_finished is not None:
                    on_finished(False)
                return

            saved_path = access.path
            self._set_workspace_path(
                saved_path,
                workspace_lock=self._take_workspace_access_lock(access),
                store=new_store,
            )
            self._manager._workspace_state.save_as_only = False
            self._manager._workspace_state.degraded_reasons = ()
            access = None
            self._adopt_committed_workspace_generation(
                saved_path,
                snapshot,
                manifest=snapshot.generation_plan.manifest,
            )
            try:
                self._refresh_workspace_payload_bindings_after_full_save(
                    saved_path,
                    backing_snapshot=backing_snapshot,
                    old_workspace_path=old_workspace_path,
                    skip_live_data_rebind_uids=post_save_uids,
                )
            except _WorkspacePostSaveBindingError:
                self._show_workspace_post_save_binding_error(saved_path)
                self._restore_focus_after_workspace_save(origin)
                if on_finished is not None:
                    on_finished(False)
                return
            self._release_unused_imported_workspace_accesses()
            self._drain_workspace_deferred_events()
            post_save_events = tuple(
                event
                for event in self._manager._workspace_state.dirty_events
                if event.generation > snapshot.generation
            )
            has_new_dirty_generation = (
                self._manager._workspace_state.dirty_generation > snapshot.generation
                and self._manager.is_workspace_modified
            )
            if post_save_events:
                self._restore_workspace_dirty_events(post_save_events)
            elif not has_new_dirty_generation:
                self._mark_workspace_clean()
            self._record_recent_workspace(saved_path)
            if post_save_events or has_new_dirty_generation:
                message = "Workspace saved; new changes remain unsaved"
            else:
                message = (
                    f"Workspace saved in {total_elapsed:.1f} s"
                    if total_elapsed >= _WORKSPACE_SAVE_WAIT_DIALOG_THRESHOLD_SECONDS
                    else "Workspace saved"
                )
            self._manager._status_bar.showMessage(message, 5000)
            self._record_saved_workspace_code_trust(
                snapshot.generation_plan.manifest,
                trusted_lineage=snapshot.trusted_lineage,
                current_document=(
                    not post_save_events and not has_new_dirty_generation
                ),
            )
            self._restore_focus_after_workspace_save(origin)
            self._schedule_workspace_gc()
            if on_finished is not None:
                on_finished(True)

        def _start_error() -> None:
            if worker_path is not None:
                with contextlib.suppress(OSError):
                    worker_path.unlink()
            if access is not None:
                access.release()
            self._manager._status_bar.clearMessage()
            self._manager._show_operation_error(
                "Error while saving workspace",
                "An error occurred while saving the workspace file.",
            )
            self._restore_focus_after_workspace_save(origin)
            if on_finished is not None:
                on_finished(False)

        if worker_path is None:  # pragma: no cover
            snapshot.close()
            access.release()
            raise RuntimeError("Save As worker path was not prepared")
        return self._start_workspace_save_worker(
            worker_path,
            snapshot,
            on_finished=_finish_save_as,
            on_start_error=_start_error,
        )

    def _confirm_compaction_with_exported_readers(
        self, parent: QtWidgets.QWidget
    ) -> bool:
        """Confirm that compaction can invalidate serialized Dask readers."""
        msg_box = QtWidgets.QMessageBox(parent)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setWindowTitle("Compact Workspace")
        msg_box.setText("This workspace was sent to Dask workers.")
        msg_box.setInformativeText(
            "Some Dask Futures can still need this data for retry or recomputation. "
            "ImageTool Manager cannot determine when every client has released it. "
            "Continue only if you no longer need those Futures."
        )
        cancel_button = msg_box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        compact_button = msg_box.addButton(
            "Compact Workspace", QtWidgets.QMessageBox.ButtonRole.DestructiveRole
        )
        msg_box.setDefaultButton(cancel_button)
        msg_box.exec()
        return msg_box.clickedButton() == compact_button

    def compact_workspace(self) -> bool:
        """Rewrite the current workspace with only reachable payload objects."""
        workspace_path = self._current_workspace_document_path()
        if workspace_path is None:
            return self.save_as()
        if self._manager._workspace_state.save_as_only:

            def _compact_after_save(save_succeeded: bool) -> None:
                if save_succeeded and not self._manager._workspace_state.save_as_only:
                    self.compact_workspace()

            return self.save_as(on_finished=_compact_after_save)
        if self._manager._workspace_state.save_in_progress:
            self._manager._status_bar.showMessage(
                "Workspace save already in progress", 3000
            )
            return False
        origin = self._active_managed_window()
        store = self._workspace_store
        if store is None or store.closed or store.path != workspace_path:
            self._manager._show_operation_error(
                "Error while compacting workspace",
                "The workspace file is not open. Reopen it and try again.",
            )
            return False
        serialized_reader_pins = store.serialized_reader_pin_snapshot()
        if (
            not serialized_reader_pins.empty
            and not self._confirm_compaction_with_exported_readers(
                origin or self._manager
            )
        ):
            self._restore_focus_after_workspace_save(origin)
            return False
        try:
            with erlab.interactive.utils.wait_dialog(
                origin or self._manager, "Compacting workspace..."
            ):
                self.saving._close_workspace_idle_readers(workspace_path)
                if (
                    self._manager.is_workspace_modified
                    or self._manager._workspace_state.schema_version
                    < workspace_format._current_workspace_schema_version()
                ):
                    self.saving._save_workspace_document(
                        workspace_path,
                        mark_clean=False,
                    )
                if store.leased_legacy_group_paths:
                    workspace_storage._rebind_equivalent_legacy_readers(
                        store,
                        store.current_generation().manifest,
                    )
                workspace_storage._compact_workspace_store(
                    store,
                    discard_serialized_reader_pins=serialized_reader_pins,
                )
            store.release_serialized_reader_pins(serialized_reader_pins)
            self._manager._status_bar.showMessage("Workspace compacted", 5000)
            self._mark_workspace_clean()
        except Exception as exc:
            logger.exception(
                "Could not compact workspace",
                extra={"suppress_ui_alert": True},
            )
            if isinstance(exc, workspace_store.WorkspaceStoreReopenError):
                store.close()
                if self._workspace_store is store:
                    self._workspace_store = None
                error_text = (
                    "The compacted workspace was saved, but it could not be "
                    "reopened. Close and reopen the workspace before you continue."
                )
            else:
                error_text = "An error occurred while compacting the workspace file."
            self._manager._show_operation_error(
                "Error while compacting workspace",
                error_text,
            )
            self._restore_focus_after_workspace_save(origin)
            return False

        self._restore_focus_after_workspace_save(origin)
        return True

    def load(self, *, native: bool = True) -> bool:
        """Replace this manager with a workspace file."""
        if self._manager._workspace_state.save_in_progress:
            self._manager._status_bar.showMessage(
                "Workspace save in progress; open after it finishes", 3000
            )
            return False
        dialog = QtWidgets.QFileDialog(self._manager)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptOpen)
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilter("ImageTool Workspace Files (*.itws)")
        if (directory := self._manager._recent_or_default_directory()) is not None:
            dialog.setDirectory(directory)
        if not native:  # pragma: no branch
            dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)

        if not dialog.exec():
            return False

        fname = dialog.selectedFiles()[0]
        return self._open_workspace_after_dirty_prompt(fname, native=native)

    def import_workspace(self, *, native: bool = True) -> bool:
        """Import selected windows from another workspace file."""
        if self._manager._workspace_state.save_in_progress:
            self._manager._status_bar.showMessage(
                "Workspace save in progress; import after it finishes", 3000
            )
            return False
        dialog = QtWidgets.QFileDialog(self._manager)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptOpen)
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilter("ImageTool Workspace Files (*.itws)")
        if (directory := self._manager._recent_or_default_directory()) is not None:
            dialog.setDirectory(directory)
        if not native:  # pragma: no branch
            dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)

        if not dialog.exec():
            return False
        fname = dialog.selectedFiles()[0]
        self._manager._recent_directory = os.path.dirname(fname)
        try:
            loaded = self.loading._load_workspace_file(
                fname,
                replace=False,
                associate=False,
                mark_dirty=True,
                select=True,
            )
        except Exception as exc:
            if workspace_storage._is_workspace_file_lock_error(exc):
                logger.info(
                    "Workspace file is already open or locked: %s",
                    fname,
                    extra={"suppress_ui_alert": True},
                )
                _show_workspace_file_lock_error(self._manager, fname)
            else:
                logger.exception(
                    "Error while importing workspace",
                    extra={"suppress_ui_alert": True},
                )
                erlab.interactive.utils.MessageDialog.critical(
                    self._manager,
                    "Error",
                    "An error occurred while importing the workspace file.",
                )
            return False
        else:
            if loaded:
                self._record_recent_workspace(fname)
            return loaded
