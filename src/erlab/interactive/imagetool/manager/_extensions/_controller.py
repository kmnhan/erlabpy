"""Qt controller for ImageTool Manager extensions."""

from __future__ import annotations

import contextlib
import copy
import fnmatch
import functools
import hashlib
import logging
import os
import pathlib
import traceback
import typing
from collections import defaultdict, deque

from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.extensions import EXTENSION_API_VERSION, LoaderDescriptor, RoutineDescriptor
from erlab.interactive.imagetool._load_source import _deserialize_loader_kwargs
from erlab.interactive.imagetool._provenance._model import iter_operation_refs
from erlab.interactive.imagetool.manager._extensions._catalog import (
    _ExtensionCatalog,
    _ExtensionCatalogConflictError,
    _safe_extension_id,
)
from erlab.interactive.imagetool.manager._extensions._dialogs import (
    _ExtensionParameterDialog,
    _ManageExtensionsDialog,
    _MetadataDialog,
    _RoutineSelectionDialog,
    _SourceReviewDialog,
    _WorkspaceRequirementsDialog,
)
from erlab.interactive.imagetool.manager._extensions._execution import (
    _DecoratedLoaderAdapter,
    _ExtensionExecutionController,
    _ExtensionLoaderCall,
)
from erlab.interactive.imagetool.manager._extensions._models import (
    _ExtensionMetadata,
    _ExtensionRecord,
    _ExtensionRevision,
    _ResolvedWorkspaceRequirement,
    _WorkspaceExtensionRequirement,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import xarray as xr

    from erlab.interactive.explorer._tabbed_explorer import _TabbedExplorer
    from erlab.interactive.imagetool._provenance._model import FileLoadSource
    from erlab.interactive.imagetool._provenance._operations import (
        ExtensionRoutineOperation,
    )
    from erlab.interactive.imagetool.manager._mainwindow import ImageToolManager

logger = logging.getLogger(__name__)


class _ExtensionSourceHashMismatchError(FileNotFoundError):
    """Report that available script bytes do not match the requested revision."""


class _ExtensionController(QtCore.QObject):
    """Own catalog UI, revision approval, and execution for one manager.

    The catalog is application-wide. Source approval and workspace ``Open Without``
    decisions that are not persisted stay on this controller and do not propagate to
    other managers.
    """

    def __init__(self, manager: ImageToolManager) -> None:
        super().__init__(manager)
        self._manager = manager
        self.catalog = _ExtensionCatalog(parent=self)
        if not erlab.utils.misc._IS_PACKAGED:
            try:
                self.catalog.store.refresh_environment_packages()
                self.catalog.refresh()
            except Exception:
                logger.exception(
                    "Could not refresh environment extensions during manager startup",
                    extra={"suppress_ui_alert": True},
                )
        self.execution = _ExtensionExecutionController(manager, self.catalog)
        self._catalog_changed_slot = self._catalog_changed
        self._manage_action_slot = self._manage_action
        self.catalog.changed.connect(self._catalog_changed_slot)
        self._recent: deque[tuple[str, str]] = deque(maxlen=8)
        self._workspace_requirements: tuple[_WorkspaceExtensionRequirement, ...] = ()
        self._workspace_embedded_sources: dict[tuple[str, str], bytes] = {}
        self._workspace_unresolved_embedded_objects: dict[
            str, tuple[bytes, str | None]
        ] = {}
        self._explorer_loaders: dict[str, erlab.io.dataloader.LoaderBase] = {}
        self._environment_loader_names: set[str] = set()
        self._closed = False
        self._unresolved_workspace_requirement_payloads: tuple[typing.Any, ...] = ()
        self._manage_dialog = _ManageExtensionsDialog(manager)
        self._manage_dialog.action_requested.connect(self._manage_action_slot)
        self._menu_show_slot = self._populate_menu
        self._context_menu_connections: list[
            tuple[QtWidgets.QMenu, Callable[[], None]]
        ] = []

        self.add_script_action = QtGui.QAction("Add Script…", manager)
        self.add_script_action.setObjectName("manager_add_extension_script_action")
        self._add_script_slot = self.add_script
        self.add_script_action.triggered.connect(self._add_script_slot)
        self.manage_action = QtGui.QAction("Manage Extensions", manager)
        self.manage_action.setObjectName("manager_manage_extensions_action")
        self._show_manager_slot = self.show_manager
        self.manage_action.triggered.connect(self._show_manager_slot)
        self.requirements_action = QtGui.QAction("Workspace Requirements", manager)
        self.requirements_action.setObjectName(
            "manager_workspace_extension_requirements_action"
        )
        self._show_requirements_slot = self.show_workspace_requirements
        self.requirements_action.triggered.connect(self._show_requirements_slot)
        self.progress_action = QtGui.QAction("Extension Jobs", manager)
        self.progress_action.setObjectName("manager_extension_jobs_action")
        self._show_progress_slot = self.execution.show_progress
        self.progress_action.triggered.connect(self._show_progress_slot)
        self.refresh_packages_action = QtGui.QAction(
            "Refresh Environment Packages", manager
        )
        self.refresh_packages_action.setObjectName(
            "manager_refresh_environment_extensions_action"
        )
        self._refresh_packages_slot = self.refresh_environment_packages
        self.refresh_packages_action.triggered.connect(self._refresh_packages_slot)
        self.refresh_packages_action.setVisible(not erlab.utils.misc._IS_PACKAGED)
        self.menu: QtWidgets.QMenu | None = None
        self._sync_explorer_loaders()

    def create_menu(self, menu_bar: QtWidgets.QMenuBar) -> QtWidgets.QMenu:
        menu = typing.cast("QtWidgets.QMenu", menu_bar.addMenu("&Extensions"))
        menu.setObjectName("manager_extensions_menu")
        menu.aboutToShow.connect(self._menu_show_slot)
        self.menu = menu
        self._populate_menu()
        return menu

    def add_context_submenu(self, parent_menu: QtWidgets.QMenu) -> QtWidgets.QMenu:
        menu = QtWidgets.QMenu("Extensions", parent_menu)
        menu.setObjectName("manager_extensions_context_menu")
        populate_slot = functools.partial(
            self._populate_routine_menu, menu, compact=True
        )
        menu.aboutToShow.connect(populate_slot)
        self._context_menu_connections.append((menu, populate_slot))
        self._populate_routine_menu(menu, compact=True)
        parent_menu.addMenu(menu)
        return menu

    def _enabled_routines(self) -> tuple[tuple[str, str, RoutineDescriptor], ...]:
        entries: list[tuple[str, str, RoutineDescriptor]] = []
        for record in self.catalog.model.extensions.values():
            if record.removed or not record.enabled:
                continue
            if record.source_type == "environment-package" and (
                erlab.utils.misc._IS_PACKAGED
            ):
                continue
            revision = record.revisions[record.current_revision]
            if not self.catalog.store.revision_available(
                record, record.current_revision
            ):
                continue
            entries.extend((record.id, record.name, item) for item in revision.routines)
        return tuple(entries)

    def file_loaders(
        self,
        paths: str | os.PathLike[str] | Iterable[str | os.PathLike[str]] | None = None,
    ) -> dict[str, tuple[Callable[..., typing.Any], dict[str, typing.Any]]]:
        """Return enabled decorated loaders in the standard file-dialog shape."""
        path_values = (
            ()
            if paths is None
            else (pathlib.Path(paths),)
            if isinstance(paths, (str, os.PathLike))
            else tuple(pathlib.Path(path) for path in paths)
        )
        entries: dict[str, tuple[Callable[..., typing.Any], dict[str, typing.Any]]] = {}
        owners: dict[str, str] = {}
        for record in self.catalog.model.extensions.values():
            if record.removed or not record.enabled:
                continue
            if record.source_type == "environment-package" and (
                erlab.utils.misc._IS_PACKAGED
            ):
                continue
            revision = record.revisions[record.current_revision]
            if not self.catalog.store.revision_available(
                record, record.current_revision
            ):
                continue
            for descriptor in revision.loaders:
                dialog_methods = revision.loader_dialog_methods
                if (
                    revision.entry_point_group == "erlab.io.loaders"
                    and not dialog_methods
                ):
                    continue
                if not dialog_methods:
                    patterns = tuple(
                        f"*{suffix}" for suffix in descriptor.extensions
                    ) or ("*",)
                    dialog_entries = (
                        (
                            f"{descriptor.name} ({' '.join(patterns)})",
                            None,
                            {
                                parameter.id: parameter.default
                                for parameter in descriptor.parameters
                                if not parameter.required
                            },
                        ),
                    )
                else:
                    dialog_entries = tuple(
                        (item.name_filter, item.method, dict(item.defaults))
                        for item in dialog_methods
                    )
                for name_filter, loader_method, defaults in dialog_entries:
                    patterns = erlab.interactive.utils._filter_to_patterns(name_filter)
                    if path_values and not all(
                        any(fnmatch.fnmatch(path.name, pattern) for pattern in patterns)
                        for path in path_values
                    ):
                        continue
                    previous_owner = owners.get(name_filter)
                    if previous_owner is not None:
                        raise ValueError(
                            f"Conflicting extension file dialog filter {name_filter!r} "
                            f"provided by {previous_owner!r} and {record.id!r}"
                        )
                    call = self._loader_call(
                        record,
                        revision,
                        descriptor,
                        loader_method=loader_method,
                    )
                    if revision.entry_point_group == "erlab.io.loaders" and (
                        loader_method is None or "." not in loader_method
                    ):
                        call_adapter = _DecoratedLoaderAdapter(call)
                        entries[name_filter] = (call_adapter.load, defaults)
                    else:
                        entries[name_filter] = (call, defaults)
                    owners[name_filter] = record.id
        return entries

    def _loader_call(
        self,
        record: _ExtensionRecord,
        revision: _ExtensionRevision,
        descriptor: LoaderDescriptor,
        *,
        revision_hash: str | None = None,
        loader_method: str | None = None,
    ) -> _ExtensionLoaderCall:
        """Create one manager-local call pinned to validated catalog state."""
        pinned_revision = (
            record.current_revision if revision_hash is None else revision_hash
        )
        return _ExtensionLoaderCall(
            manager_session_id=self._manager._manager_record.internal_id,
            catalog_generation=self.catalog.model.generation,
            extension_id=record.id,
            extension_name=record.name,
            revision_hash=pinned_revision,
            loader_id=descriptor.id,
            descriptor=descriptor,
            source_path=(
                self.catalog.store.source_path(record.id, pinned_revision)
                if record.source_type == "script"
                else None
            ),
            source_type=record.source_type,
            executor=self.execution.run_loader,
            entry_point_group=revision.entry_point_group,
            entry_point_name=revision.entry_point_name,
            entry_point_value=revision.entry_point_value,
            loader_method=loader_method,
            loader_always_single=(
                True
                if revision.loader_always_single is None
                else revision.loader_always_single
            ),
        )

    def replay_loader(
        self, load_source: FileLoadSource
    ) -> (
        xr.DataArray
        | xr.Dataset
        | xr.DataTree
        | list[xr.DataArray | xr.Dataset | xr.DataTree]
    ):
        """Run one exact file-provenance loader through the manager queue."""
        replay_call = load_source.replay_call
        if (
            replay_call is None
            or replay_call.kind != "extension_loader"
            or replay_call.revision is None
            or replay_call.capability_id is None
        ):
            raise erlab.extensions.ExtensionExecutionError(
                "Extension loader replay metadata is incomplete"
            )
        catalog = self.catalog.store.read()
        record = catalog.extensions.get(replay_call.target)
        if record is None or record.removed or not record.enabled:
            raise erlab.extensions.ExtensionExecutionError(
                "The extension is not enabled"
            )
        revision = record.revisions.get(replay_call.revision)
        if revision is None or not revision.approved:
            raise erlab.extensions.ExtensionExecutionError(
                "The extension loader revision is not available"
            )
        descriptor = next(
            (item for item in revision.loaders if item.id == replay_call.capability_id),
            None,
        )
        if descriptor is None:
            raise erlab.extensions.ExtensionExecutionError(
                f"Loader {replay_call.capability_id!r} is not available"
            )
        if revision.entry_point_group == "erlab.io.loaders":
            approved_methods = {
                None,
                *(item.method for item in revision.loader_dialog_methods),
            }
            if replay_call.loader_method not in approved_methods:
                raise erlab.extensions.ExtensionExecutionError(
                    "The requested loader method was not approved for this revision"
                )
        elif replay_call.loader_method is not None:
            raise erlab.extensions.ExtensionExecutionError(
                "Decorated extension loaders do not provide alternate methods"
            )
        call = self._loader_call(
            record,
            revision,
            descriptor,
            revision_hash=replay_call.revision,
            loader_method=replay_call.loader_method,
        )
        return self.execution.run_loader(
            call,
            pathlib.Path(load_source.path),
            _deserialize_loader_kwargs(replay_call.kwargs),
        )

    @property
    def explorer_loaders(self) -> dict[str, erlab.io.dataloader.LoaderBase]:
        """Manager-local loader adapters used by existing Data Explorer tabs."""
        return self._explorer_loaders

    @property
    def environment_loader_names(self) -> set[str]:
        """Names reserved by catalog-managed LoaderBase entry points."""
        return self._environment_loader_names

    def _sync_explorer_loaders(self) -> None:
        updated: dict[str, erlab.io.dataloader.LoaderBase] = {}
        for record in self.catalog.model.extensions.values():
            if record.removed or not record.enabled:
                continue
            if record.source_type == "environment-package" and (
                erlab.utils.misc._IS_PACKAGED
            ):
                continue
            revision = record.revisions[record.current_revision]
            if not self.catalog.store.revision_available(
                record, record.current_revision
            ):
                continue
            for descriptor in revision.loaders:
                adapter = _DecoratedLoaderAdapter(
                    self._loader_call(record, revision, descriptor)
                )
                updated[adapter.name] = adapter
        managed_names: set[str] = set()
        if not erlab.utils.misc._IS_PACKAGED:
            managed_names.update(
                descriptor.id
                for record in self.catalog.model.extensions.values()
                if record.source_type == "environment-package"
                for revision in record.revisions.values()
                if revision.entry_point_group == "erlab.io.loaders"
                for descriptor in revision.loaders
            )
            managed_names.update(
                revision.entry_point_name
                for record in self.catalog.model.extensions.values()
                if record.source_type == "environment-package"
                for revision in record.revisions.values()
                if revision.entry_point_group == "erlab.io.loaders"
                and revision.entry_point_name is not None
            )
        self._explorer_loaders.clear()
        self._explorer_loaders.update(updated)
        self._environment_loader_names.clear()
        self._environment_loader_names.update(managed_names)

    def loader_by_name(
        self, name: str
    ) -> tuple[Callable[..., typing.Any], dict[str, typing.Any]] | None:
        adapter = self._explorer_loaders.get(name)
        return None if adapter is None else (adapter.load, {})

    def loader_name_for_callable(self, call: Callable[..., typing.Any]) -> str | None:
        """Return the Data Explorer name for one extension loader call."""
        source = getattr(call, "__self__", call)
        if isinstance(source, _DecoratedLoaderAdapter):
            return source.name
        if isinstance(source, _ExtensionLoaderCall):
            return source.manager_loader_name
        return None

    def _populate_menu(self) -> None:
        menu = self.menu
        if menu is None:
            return
        self._populate_routine_menu(menu, compact=False)

    def _populate_routine_menu(self, menu: QtWidgets.QMenu, *, compact: bool) -> None:
        menu.clear()
        run_action = typing.cast("QtGui.QAction", menu.addAction("Run Routine…"))
        run_action.setObjectName("manager_run_extension_routine_action")
        run_action.triggered.connect(self.select_routine)
        routines = self._enabled_routines()
        favorite_keys = {
            (record.id, routine.id)
            for record in self.catalog.model.extensions.values()
            if record.favorite and record.enabled and not record.removed
            for routine in record.revisions[record.current_revision].routines
        }
        if favorite_keys:
            favorites = typing.cast("QtWidgets.QMenu", menu.addMenu("Favorites"))
            for entry in routines:
                if (entry[0], entry[2].id) in favorite_keys:
                    self._add_routine_action(favorites, entry)
        recent_keys = tuple(
            key for key in self._recent if key in {(e[0], e[2].id) for e in routines}
        )
        if recent_keys:
            recent_menu = typing.cast("QtWidgets.QMenu", menu.addMenu("Recent"))
            by_key = {(entry[0], entry[2].id): entry for entry in routines}
            for key in recent_keys:
                self._add_routine_action(recent_menu, by_key[key])
        categories: dict[str, list[tuple[str, str, RoutineDescriptor]]] = defaultdict(
            list
        )
        for entry in routines:
            categories[entry[2].category].append(entry)
        for category in sorted(categories):
            category_menu = typing.cast("QtWidgets.QMenu", menu.addMenu(category))
            for entry in sorted(categories[category], key=lambda item: item[2].name):
                self._add_routine_action(category_menu, entry)
        if compact:
            return
        menu.addSeparator()
        menu.addAction(self.add_script_action)
        menu.addAction(self.manage_action)
        menu.addAction(self.requirements_action)
        menu.addAction(self.progress_action)
        menu.addAction(self.refresh_packages_action)

    def _add_routine_action(
        self,
        menu: QtWidgets.QMenu,
        entry: tuple[str, str, RoutineDescriptor],
    ) -> None:
        extension_id, extension_name, descriptor = entry
        action = typing.cast("QtGui.QAction", menu.addAction(descriptor.name))
        action.setData((extension_id, descriptor.id))
        action.setToolTip(descriptor.summary or extension_name)
        action.triggered.connect(
            lambda _checked=False, ext=extension_id, routine=descriptor.id: (
                self.run_routine(ext, routine)
            )
        )

    @QtCore.Slot()
    def select_routine(self) -> None:
        routines = self._enabled_routines()
        if not routines:
            QtWidgets.QMessageBox.information(
                self._manager,
                "No Routines",
                "No enabled extension routines are available.",
            )
            return
        dialog = _RoutineSelectionDialog(routines, self._manager)
        if dialog.exec() and dialog.selection is not None:
            self.run_routine(*dialog.selection)

    def run_routine(self, extension_id: str, routine_id: str) -> None:
        targets = self._manager._selected_imagetool_targets()
        if len(targets) != 1:
            QtWidgets.QMessageBox.information(
                self._manager,
                "Select Data",
                "Select one ImageTool before you run a routine.",
            )
            return
        record = self.catalog.model.extensions.get(extension_id)
        if record is None or record.removed or not record.enabled:
            return
        descriptor = next(
            (
                item
                for item in record.revisions[record.current_revision].routines
                if item.id == routine_id
            ),
            None,
        )
        if descriptor is None:
            return
        dialog = _ExtensionParameterDialog(descriptor, self._manager)
        if not dialog.exec():
            return
        try:
            self.execution.queue_routine(
                extension_id=extension_id,
                routine_id=routine_id,
                parameters=dialog.parameters,
                target=targets[0],
            )
        except Exception:
            erlab.interactive.utils.MessageDialog.critical(
                self._manager,
                "Extension Error",
                f"{descriptor.name} could not be queued.",
                detailed_text=traceback.format_exc(),
            )
            return
        key = (extension_id, routine_id)
        with contextlib.suppress(ValueError):
            self._recent.remove(key)
        self._recent.appendleft(key)

    @QtCore.Slot()
    def add_script(self) -> None:
        path, _selected_filter = QtWidgets.QFileDialog.getOpenFileName(
            self._manager,
            "Add Extension Script",
            self._manager._recent_or_default_directory() or "",
            "Python scripts (*.py)",
        )
        if path:
            self._review_and_add(pathlib.Path(path))

    def _review_and_add(
        self, path: pathlib.Path, *, extension_id: str | None = None
    ) -> bool:
        try:
            reviewed_source = path.read_bytes()
            reviewed_revision = hashlib.sha256(reviewed_source).hexdigest()
            source_text = reviewed_source.decode("utf-8")
            catalog_extension_id = _safe_extension_id(extension_id or path.stem)
            existing = self.catalog.model.extensions.get(catalog_extension_id)
            if extension_id is None and existing is not None:
                current_revision = existing.revisions[existing.current_revision]
                same_source = (
                    existing.source_type == "script"
                    and current_revision.source_path is not None
                    and pathlib.Path(current_revision.source_path)
                    .expanduser()
                    .resolve()
                    == path.expanduser().resolve()
                )
                same_content = (
                    existing.source_type == "script"
                    and existing.current_revision == reviewed_revision
                )
                if not same_source and not same_content:
                    erlab.interactive.utils.MessageDialog.critical(
                        self._manager,
                        "Extension ID Already Exists",
                        (
                            f"A different extension already uses ID "
                            f"{catalog_extension_id!r}. Rename this script file or "
                            "use Reload for the existing extension."
                        ),
                    )
                    return False
            dialog = _SourceReviewDialog(
                None,
                self._manager,
                source_text=source_text,
            )
        except (OSError, UnicodeError):
            erlab.interactive.utils.MessageDialog.critical(
                self._manager,
                "Extension Error",
                "The extension source could not be read.",
                detailed_text=traceback.format_exc(),
            )
            return False
        if existing is not None:
            dialog.author_edit.setText(existing.metadata.author)
            dialog.contact_edit.setText(existing.metadata.contact)
            dialog.project_url_edit.setText(existing.metadata.project_url)
            dialog.change_summary_edit.setText(existing.metadata.change_summary)
            dialog.changelog_edit.setPlainText(existing.metadata.changelog)
        if not dialog.exec():
            return False
        try:
            _catalog, _revision, created = self.catalog.store.add_script(
                path,
                extension_id=extension_id,
                metadata=dialog.metadata,
                expected_revision=reviewed_revision,
                expected_record_generation=(
                    None if existing is None else existing.record_generation
                ),
                check_record_generation=True,
            )
            self.catalog.refresh()
            if not created:
                self._manager._status_bar.showMessage(
                    "The script contents did not change; no revision was created.",
                    4000,
                )
                return True
            record = self.catalog.model.extensions[catalog_extension_id]
            self.execution.validate_and_enable(
                record.id,
                expected_record_generation=record.record_generation,
            )
            self.catalog.refresh()
            return True  # noqa: TRY300 - success exits before the shared error path.
        except Exception:
            erlab.interactive.utils.MessageDialog.critical(
                self._manager,
                "Extension Error",
                "The extension could not be enabled.",
                detailed_text=traceback.format_exc(),
            )
            self.catalog.refresh()
            return False

    @QtCore.Slot()
    def show_manager(self) -> None:
        self._manage_dialog.set_catalog(
            self.catalog.model, self._catalog_source_states()
        )
        self._manage_dialog.show()
        self._manage_dialog.raise_()

    def _catalog_source_states(self) -> dict[tuple[str, str], str]:
        states: dict[tuple[str, str], str] = {}
        for record in self.catalog.model.extensions.values():
            for revision_hash, revision in record.revisions.items():
                key = (record.id, revision_hash)
                if record.source_type == "environment-package":
                    states[key] = (
                        "Editable environment package"
                        if revision.editable
                        else "Environment package"
                    )
                    continue
                try:
                    stored_path = self.catalog.store.source_path(
                        record.id, revision_hash
                    )
                except (FileNotFoundError, KeyError):
                    states[key] = "Stored source missing"
                    continue
                try:
                    stored_hash = hashlib.sha256(stored_path.read_bytes()).hexdigest()
                except OSError:
                    states[key] = "Stored source unreadable"
                    continue
                if stored_hash != revision_hash:
                    states[key] = "Stored source hash mismatch"
                    continue
                if revision.source_path is None:
                    states[key] = "Stored embedded source"
                    continue
                source_path = pathlib.Path(revision.source_path)
                if not source_path.is_file():
                    states[key] = "Stored source; original missing"
                    continue
                try:
                    current_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
                except OSError:
                    states[key] = "Stored source; original unreadable"
                else:
                    states[key] = (
                        "Stored source; original unchanged"
                        if current_hash == revision_hash
                        else "Stored source; original changed"
                    )
        return states

    @QtCore.Slot(str, str)
    def _manage_action(self, action_id: str, extension_id: str) -> None:
        record = self.catalog.model.extensions.get(extension_id)
        if record is None:
            return
        try:
            if action_id == "reload":
                source_path = record.revisions[record.current_revision].source_path
                if source_path is None:
                    raise FileNotFoundError(  # noqa: TRY301
                        "This extension has no script source path"
                    )
                self._review_and_add(
                    pathlib.Path(source_path), extension_id=extension_id
                )
                return
            if action_id == "toggle" and not record.enabled:
                self.execution.validate_and_enable(
                    extension_id,
                    expected_record_generation=record.record_generation,
                )
            elif action_id == "toggle":
                self.catalog.store.update_record(
                    extension_id,
                    expected_record_generation=record.record_generation,
                    enabled=False,
                )
            elif action_id == "favorite":
                self.catalog.store.update_record(
                    extension_id,
                    expected_record_generation=record.record_generation,
                    favorite=not record.favorite,
                )
            elif action_id == "remove":
                self.catalog.store.update_record(
                    extension_id,
                    expected_record_generation=record.record_generation,
                    removed=not record.removed,
                )
            elif action_id == "embedding":
                labels = {
                    "referenced": "Referenced scripts",
                    "always": "Always include",
                    "never": "Never embed",
                }
                selected, accepted = QtWidgets.QInputDialog.getItem(
                    self._manager,
                    "Script Embedding",
                    "Workspace embedding policy",
                    tuple(labels.values()),
                    tuple(labels).index(record.embed_policy),
                    False,
                )
                if not accepted:
                    return
                policy = next(key for key, value in labels.items() if value == selected)
                self.catalog.store.update_record(
                    extension_id,
                    expected_record_generation=record.record_generation,
                    embed_policy=typing.cast(
                        'typing.Literal["referenced", "always", "never"]', policy
                    ),
                )
            elif action_id == "metadata":
                dialog = _MetadataDialog(record.metadata, self._manager)
                if not dialog.exec():
                    return
                self.catalog.store.update_record(
                    extension_id,
                    expected_record_generation=record.record_generation,
                    metadata=dialog.metadata,
                )
            self.catalog.refresh()
        except _ExtensionCatalogConflictError as error:
            QtWidgets.QMessageBox.warning(
                self._manager,
                "Extension Changed",
                str(error),
            )
            self.catalog.refresh()
        except Exception:
            erlab.interactive.utils.MessageDialog.critical(
                self._manager,
                "Extension Error",
                "The extension could not be changed.",
                detailed_text=traceback.format_exc(),
            )
        self._manage_dialog.set_catalog(
            self.catalog.model, self._catalog_source_states()
        )

    @QtCore.Slot(object)
    def _catalog_changed(self, _model: object) -> None:
        self._sync_explorer_loaders()
        self._manage_dialog.set_catalog(
            self.catalog.model, self._catalog_source_states()
        )
        if self.menu is not None and self.menu.isVisible():
            self._populate_menu()
        explorer = self._manager._standalone_app_windows.get("explorer")
        if explorer is not None and erlab.interactive.utils.qt_is_valid(explorer):
            typing.cast("_TabbedExplorer", explorer).refresh_loader_choices()
        self._manager._update_actions()
        for node in self._manager._tool_graph.nodes.values():
            if node.tool_window is not None:
                node.tool_window._refresh_reload_data_action()

    def collect_workspace_requirements(
        self,
    ) -> tuple[_WorkspaceExtensionRequirement, ...]:
        """Rebuild loaded-node dependencies and retain unresolved references.

        Current provenance is authoritative for nodes in the graph. References to
        nodes that did not load and explicit requirements without node references
        remain unchanged so a degraded Save As does not discard them.
        """
        loaded_node_uids = set(self._manager._tool_graph.nodes)
        persisted: dict[
            tuple[str, str, str, str], list[_WorkspaceExtensionRequirement]
        ] = defaultdict(list)
        for item in self._workspace_requirements:
            persisted[
                (
                    item.extension_id,
                    item.revision_hash,
                    item.capability_kind,
                    item.capability_id,
                )
            ].append(item)

        def merged_persisted_requirement(
            key: tuple[str, str, str, str],
        ) -> _WorkspaceExtensionRequirement | None:
            """Merge per-node state for one immutable capability revision."""
            items = persisted.get(key)
            if not items:
                return None
            primary = items[0]
            metadata = dict(primary.metadata_snapshot)
            referencing_nodes: set[str] = set()
            file_sources: set[str] = set()
            embedded_object_id = primary.embedded_object_id
            for item in items:
                referencing_nodes.update(item.referencing_nodes)
                file_sources.update(item.file_sources)
                if embedded_object_id is None:
                    embedded_object_id = item.embedded_object_id
                for name, value in item.metadata_snapshot.items():
                    metadata.setdefault(name, value)
            return primary.model_copy(
                update={
                    "metadata_snapshot": metadata,
                    "embedded_object_id": embedded_object_id,
                    "referencing_nodes": tuple(sorted(referencing_nodes)),
                    "file_sources": tuple(sorted(file_sources)),
                }
            )

        references: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        operations: dict[tuple[str, str, str], typing.Any] = {}
        loader_references: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        loader_files: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        for uid, node in self._manager._tool_graph.nodes.items():
            spec = node.passive_displayed_provenance_spec
            pending_specs = deque(() if spec is None else (spec,))
            while pending_specs:
                current_spec = pending_specs.popleft()
                pending_specs.extend(
                    nested
                    for script_input in current_spec.script_inputs
                    if (nested := script_input.parsed_provenance_spec()) is not None
                )
                for _ref, operation in iter_operation_refs(current_spec):
                    if getattr(operation, "op", None) != "extension_routine":
                        continue
                    extension_operation = typing.cast(
                        "ExtensionRoutineOperation", operation
                    )
                    key = (
                        extension_operation.extension_id,
                        extension_operation.revision_hash,
                        extension_operation.routine_id,
                    )
                    references[key].add(uid)
                    operations[key] = extension_operation
                if current_spec.file_load_source is None:
                    continue
                replay_call = current_spec.file_load_source.replay_call
                if (
                    replay_call is None
                    or replay_call.kind != "extension_loader"
                    or replay_call.revision is None
                    or replay_call.capability_id is None
                ):
                    continue
                key = (
                    replay_call.target,
                    replay_call.revision,
                    replay_call.capability_id,
                )
                loader_references[key].add(uid)
                loader_files[key].add(current_spec.file_load_source.path)
        requirements: list[_WorkspaceExtensionRequirement] = []
        for key, node_uids in references.items():
            extension_id, revision_hash, routine_id = key
            record = self.catalog.model.extensions.get(extension_id)
            operation = typing.cast("ExtensionRoutineOperation", operations[key])
            previous = merged_persisted_requirement(
                (extension_id, revision_hash, "routine", routine_id)
            )
            if previous is not None:
                node_uids.update(
                    set(previous.referencing_nodes).difference(loaded_node_uids)
                )
            source_type: typing.Literal["script", "environment-package"] = (
                operation.source_type if previous is None else previous.source_type
            )
            metadata = {} if previous is None else dict(previous.metadata_snapshot)
            record_revision = (
                None
                if record is None or record.source_type != source_type
                else record.revisions.get(revision_hash)
            )
            if record is not None and record_revision is not None:
                metadata = {
                    **metadata,
                    "extension_name": record.name,
                    "routine_name": operation.routine_name,
                    **record.metadata.model_dump(),
                }
            elif previous is None:
                metadata = {
                    "extension_name": operation.extension_name,
                    "routine_name": operation.routine_name,
                }
            if (
                previous is None
                and record_revision is not None
                and record_revision.source_modified_at is not None
            ):
                metadata["source_modified_at"] = record_revision.source_modified_at
            requirements.append(
                _WorkspaceExtensionRequirement(
                    extension_id=extension_id,
                    capability_id=routine_id,
                    capability_kind="routine",
                    revision_hash=revision_hash,
                    extension_api_version=(
                        EXTENSION_API_VERSION
                        if previous is None
                        else previous.extension_api_version
                    ),
                    source_type=source_type,
                    metadata_snapshot=metadata,
                    embedded_object_id=self._embedded_script_object_id(
                        record=record,
                        revision_hash=revision_hash,
                        source_type=source_type,
                        previous=previous,
                    ),
                    referencing_nodes=tuple(sorted(node_uids)),
                )
            )
        for key, node_uids in loader_references.items():
            extension_id, revision_hash, loader_id = key
            record = self.catalog.model.extensions.get(extension_id)
            previous = merged_persisted_requirement(
                (extension_id, revision_hash, "loader", loader_id)
            )
            unresolved_node_uids: set[str] = set()
            if previous is not None:
                unresolved_node_uids = set(previous.referencing_nodes).difference(
                    loaded_node_uids
                )
                node_uids.update(unresolved_node_uids)
            loader_source_type: typing.Literal["script", "environment-package"] = (
                previous.source_type
                if previous is not None
                else "script"
                if record is None
                else record.source_type
            )
            loader_metadata = (
                {} if previous is None else dict(previous.metadata_snapshot)
            )
            loader_record_revision = (
                None
                if record is None or record.source_type != loader_source_type
                else record.revisions.get(revision_hash)
            )
            if record is not None and loader_record_revision is not None:
                loader_metadata = {
                    **loader_metadata,
                    "extension_name": record.name,
                    **record.metadata.model_dump(),
                }
            if (
                previous is None
                and loader_record_revision is not None
                and loader_record_revision.source_modified_at is not None
            ):
                loader_metadata["source_modified_at"] = (
                    loader_record_revision.source_modified_at
                )
            requirements.append(
                _WorkspaceExtensionRequirement(
                    extension_id=extension_id,
                    capability_id=loader_id,
                    capability_kind="loader",
                    revision_hash=revision_hash,
                    extension_api_version=(
                        EXTENSION_API_VERSION
                        if previous is None
                        else previous.extension_api_version
                    ),
                    source_type=loader_source_type,
                    metadata_snapshot=loader_metadata,
                    embedded_object_id=self._embedded_script_object_id(
                        record=record,
                        revision_hash=revision_hash,
                        source_type=loader_source_type,
                        previous=previous,
                    ),
                    referencing_nodes=tuple(sorted(node_uids)),
                    file_sources=tuple(
                        sorted(
                            loader_files[key]
                            | (
                                set(previous.file_sources)
                                if previous is not None and unresolved_node_uids
                                else set()
                            )
                        )
                    ),
                )
            )
        keys = {
            (
                item.extension_id,
                item.revision_hash,
                item.capability_kind,
                item.capability_id,
            )
            for item in requirements
        }
        for key in persisted:
            if key in keys:
                continue
            previous = merged_persisted_requirement(key)
            if previous is None:
                continue
            explicit = any(not item.referencing_nodes for item in persisted[key])
            remaining_node_uids = tuple(
                uid for uid in previous.referencing_nodes if uid not in loaded_node_uids
            )
            if explicit or remaining_node_uids:
                requirements.append(
                    previous.model_copy(
                        update={"referencing_nodes": remaining_node_uids}
                    )
                )
        keys.update(
            (
                item.extension_id,
                item.revision_hash,
                item.capability_kind,
                item.capability_id,
            )
            for item in requirements
        )
        for record in self.catalog.model.extensions.values():
            if (
                record.source_type != "script"
                or record.embed_policy != "always"
                or record.removed
            ):
                continue
            revision = record.revisions[record.current_revision]
            metadata = {
                "extension_name": record.name,
                **record.metadata.model_dump(),
            }
            for capability_kind, descriptors in (
                ("routine", revision.routines),
                ("loader", revision.loaders),
            ):
                for descriptor in descriptors:
                    key = (
                        record.id,
                        record.current_revision,
                        capability_kind,
                        descriptor.id,
                    )
                    if key in keys:
                        continue
                    requirements.append(
                        _WorkspaceExtensionRequirement(
                            extension_id=record.id,
                            capability_id=descriptor.id,
                            capability_kind=typing.cast(
                                'typing.Literal["routine", "loader"]',
                                capability_kind,
                            ),
                            revision_hash=record.current_revision,
                            extension_api_version=EXTENSION_API_VERSION,
                            source_type="script",
                            metadata_snapshot=(
                                metadata
                                if revision.source_modified_at is None
                                else {
                                    **metadata,
                                    "source_modified_at": revision.source_modified_at,
                                }
                            ),
                            embedded_object_id=self._embedded_script_object_id(
                                record=record,
                                revision_hash=record.current_revision,
                                source_type="script",
                                previous=None,
                            ),
                        )
                    )
        return tuple(requirements)

    def _embedded_script_object_id(
        self,
        *,
        record: _ExtensionRecord | None,
        revision_hash: str,
        source_type: typing.Literal["script", "environment-package"],
        previous: _WorkspaceExtensionRequirement | None,
    ) -> str | None:
        """Name an embedded object only when its source can be preserved."""
        if source_type != "script":
            return None
        if previous is not None and previous.embedded_object_id is not None:
            return previous.embedded_object_id
        if record is None or record.source_type != "script":
            return None
        if record.embed_policy == "never":
            return None
        try:
            self.revision_source_bytes(record.id, revision_hash)
        except (KeyError, OSError):
            return None
        return f"extension-{revision_hash}"

    def revision_source_bytes(self, extension_id: str, revision: str) -> bytes:
        """Return verified workspace or catalog bytes for an exact revision."""
        return self._verified_revision_source(extension_id, revision)

    def _verified_revision_source(
        self,
        extension_id: str,
        revision: str,
        *,
        include_embedded: bool = True,
        include_catalog: bool = True,
    ) -> bytes:
        """Select the first source whose bytes match the requested revision."""
        candidates: list[bytes] = []
        if include_embedded:
            embedded = self._workspace_embedded_sources.get((extension_id, revision))
            if embedded is not None:
                candidates.append(embedded)
        if include_catalog:
            with contextlib.suppress(KeyError, OSError):
                candidates.append(
                    self.catalog.store.source_path(extension_id, revision).read_bytes()
                )
        for source in candidates:
            if hashlib.sha256(source).hexdigest() == revision:
                return source
        if candidates:
            raise _ExtensionSourceHashMismatchError(
                f"Available source does not match {extension_id}:{revision}"
            )
        raise FileNotFoundError(f"No source is available for {extension_id}:{revision}")

    def set_workspace_requirements(
        self,
        requirements: Iterable[_WorkspaceExtensionRequirement],
        *,
        embedded_sources: dict[tuple[str, str], bytes] | None = None,
        unresolved_embedded_objects: dict[str, tuple[bytes, str | None]] | None = None,
        unresolved_payloads: Iterable[typing.Any] = (),
    ) -> None:
        self._workspace_requirements = tuple(requirements)
        self._workspace_embedded_sources = dict(embedded_sources or {})
        self._workspace_unresolved_embedded_objects = dict(
            unresolved_embedded_objects or {}
        )
        self._unresolved_workspace_requirement_payloads = tuple(
            copy.deepcopy(item) for item in unresolved_payloads
        )

    def workspace_requirement_state(
        self,
    ) -> tuple[
        tuple[_WorkspaceExtensionRequirement, ...],
        dict[tuple[str, str], bytes],
        dict[str, tuple[bytes, str | None]],
        tuple[typing.Any, ...],
    ]:
        """Copy workspace-owned extension state for transactional rollback."""
        return (
            self._workspace_requirements,
            dict(self._workspace_embedded_sources),
            dict(self._workspace_unresolved_embedded_objects),
            copy.deepcopy(self._unresolved_workspace_requirement_payloads),
        )

    def workspace_requirement_payloads(self) -> tuple[typing.Any, ...]:
        """Return validated and unresolved payloads for lossless saving."""
        return copy.deepcopy(
            (
                *(
                    item.model_dump(mode="json")
                    for item in self.collect_workspace_requirements()
                ),
                *self._unresolved_workspace_requirement_payloads,
            )
        )

    def resolved_workspace_requirements(
        self,
        *,
        include_current: bool = True,
    ) -> tuple[_ResolvedWorkspaceRequirement, ...]:
        requirements = (
            self.collect_workspace_requirements()
            if include_current
            else self._workspace_requirements
        )
        return tuple(self._resolve_requirement(item) for item in requirements)

    def unavailable_reason_for_node(self, node_uid: str) -> str | None:
        """Return why extension-dependent replay is disabled for one node."""
        for requirement in self.collect_workspace_requirements():
            if node_uid not in requirement.referencing_nodes:
                continue
            resolved = self._resolve_requirement(requirement)
            if resolved.state != "ready":
                return (
                    f"Extension {requirement.extension_id!r} is {resolved.state}. "
                    "Open Workspace Requirements for details."
                )
        return None

    def rebase_workspace_requirement_nodes(
        self,
        uid_map: dict[str, str],
        *,
        requirements: Iterable[_WorkspaceExtensionRequirement] | None = None,
    ) -> None:
        """Rebase only requirements owned by the workspace being loaded.

        An imported workspace can use the same saved UID as an existing manager node.
        The transient object-identity scope keeps that imported UID map from changing
        requirements that belonged to the open document before the import.
        """
        if not uid_map:
            return
        selected = None if requirements is None else {id(item) for item in requirements}
        self._workspace_requirements = tuple(
            requirement.model_copy(
                update={
                    "referencing_nodes": tuple(
                        uid_map.get(uid, uid) for uid in requirement.referencing_nodes
                    )
                }
            )
            if selected is None or id(requirement) in selected
            else requirement
            for requirement in self._workspace_requirements
        )

    def remove_workspace_node_references(self, node_uids: Iterable[str]) -> None:
        """Remove dependencies for nodes that the user explicitly deleted.

        Requirements for nodes that failed to load remain unchanged because those
        nodes do not pass through this method.
        """
        removed = set(node_uids)
        if not removed:
            return
        requirements: list[_WorkspaceExtensionRequirement] = []
        for requirement in self._workspace_requirements:
            if not requirement.referencing_nodes:
                requirements.append(requirement)
                continue
            remaining = tuple(
                uid for uid in requirement.referencing_nodes if uid not in removed
            )
            if not remaining:
                continue
            if remaining != requirement.referencing_nodes:
                requirement = requirement.model_copy(
                    update={"referencing_nodes": remaining}
                )
            requirements.append(requirement)
        self._workspace_requirements = tuple(requirements)

    def _resolve_requirement(
        self, requirement: _WorkspaceExtensionRequirement
    ) -> _ResolvedWorkspaceRequirement:
        if requirement.extension_api_version != EXTENSION_API_VERSION:
            return _ResolvedWorkspaceRequirement(
                requirement=requirement,
                state="unsupported-api",
                detail=f"API {requirement.extension_api_version} is not supported",
            )
        if (
            requirement.source_type == "environment-package"
            and erlab.utils.misc._IS_PACKAGED
        ):
            return _ResolvedWorkspaceRequirement(
                requirement=requirement,
                state="missing",
                detail="Environment package extensions are unavailable in this build",
            )
        record = self.catalog.model.extensions.get(requirement.extension_id)
        if record is not None and record.source_type != requirement.source_type:
            return _ResolvedWorkspaceRequirement(
                requirement=requirement,
                state="missing",
                detail="The catalog extension uses a different source type",
            )
        if record is None or record.removed:
            try:
                self._verified_revision_source(
                    requirement.extension_id,
                    requirement.revision_hash,
                    include_catalog=False,
                )
            except _ExtensionSourceHashMismatchError:
                return _ResolvedWorkspaceRequirement(
                    requirement=requirement, state="hash-mismatch"
                )
            except FileNotFoundError:
                return _ResolvedWorkspaceRequirement(
                    requirement=requirement, state="missing"
                )
            return _ResolvedWorkspaceRequirement(
                requirement=requirement, state="approval-required"
            )
        revision = record.revisions.get(requirement.revision_hash)
        if revision is None:
            try:
                self._verified_revision_source(
                    requirement.extension_id,
                    requirement.revision_hash,
                    include_catalog=False,
                )
            except _ExtensionSourceHashMismatchError:
                return _ResolvedWorkspaceRequirement(
                    requirement=requirement, state="hash-mismatch"
                )
            except FileNotFoundError:
                pass
            else:
                return _ResolvedWorkspaceRequirement(
                    requirement=requirement, state="approval-required"
                )
            return _ResolvedWorkspaceRequirement(
                requirement=requirement,
                state="missing",
                detail="The exact revision is not in the application catalog",
            )
        if record.source_type == "script":
            try:
                self._verified_revision_source(
                    record.id,
                    requirement.revision_hash,
                    include_embedded=False,
                )
            except _ExtensionSourceHashMismatchError:
                return _ResolvedWorkspaceRequirement(
                    requirement=requirement,
                    state="hash-mismatch",
                    detail="The stored revision source hash does not match",
                )
            except FileNotFoundError:
                return _ResolvedWorkspaceRequirement(
                    requirement=requirement,
                    state="missing",
                    detail="The stored revision source is unavailable",
                )
        else:
            try:
                self.catalog.store._entry_point_for_revision(revision)
            except Exception:
                return _ResolvedWorkspaceRequirement(
                    requirement=requirement,
                    state="missing",
                    detail="The environment package entry point is unavailable",
                )
        if revision.import_error:
            return _ResolvedWorkspaceRequirement(
                requirement=requirement,
                state="import-failed",
                detail=revision.import_error,
            )
        if not revision.approved:
            return _ResolvedWorkspaceRequirement(
                requirement=requirement, state="approval-required"
            )
        capabilities = (
            revision.routines
            if requirement.capability_kind == "routine"
            else revision.loaders
        )
        if all(item.id != requirement.capability_id for item in capabilities):
            return _ResolvedWorkspaceRequirement(
                requirement=requirement,
                state="missing",
                detail="The revision does not provide the required capability",
            )
        if not record.enabled:
            return _ResolvedWorkspaceRequirement(
                requirement=requirement, state="disabled"
            )
        return _ResolvedWorkspaceRequirement(requirement=requirement, state="ready")

    @QtCore.Slot()
    def show_workspace_requirements(self) -> None:
        dialog = _WorkspaceRequirementsDialog(
            self.resolved_workspace_requirements(),
            self._manager,
            approvable=set(self._workspace_embedded_sources),
        )

        def approve_slot(extension_id: str, revision: str) -> None:
            self._approve_embedded_script(extension_id, revision)
            if erlab.interactive.utils.qt_is_valid(dialog):
                dialog.set_requirements(self.resolved_workspace_requirements())

        dialog.approve_requested.connect(approve_slot)
        try:
            dialog.exec()
        finally:
            with contextlib.suppress(TypeError, RuntimeError):
                dialog.approve_requested.disconnect(approve_slot)

    @QtCore.Slot(str, str)
    def _approve_embedded_script(self, extension_id: str, revision: str) -> None:
        source = self._workspace_embedded_sources.get((extension_id, revision))
        if source is None:
            QtWidgets.QMessageBox.warning(
                self._manager,
                "Embedded Source Unavailable",
                "The selected requirement has no readable embedded script.",
            )
            return
        requirement = next(
            (
                item
                for item in self._workspace_requirements
                if item.extension_id == extension_id and item.revision_hash == revision
            ),
            None,
        )
        if requirement is None:
            return
        try:
            source_text = source.decode("utf-8")
        except UnicodeDecodeError:
            QtWidgets.QMessageBox.warning(
                self._manager,
                "Embedded Source Unavailable",
                "The embedded script is not valid UTF-8 source.",
            )
            return
        dialog = _SourceReviewDialog(None, self._manager, source_text=source_text)
        metadata_fields = _ExtensionMetadata.model_fields
        metadata = _ExtensionMetadata(
            **{
                key: value
                for key, value in requirement.metadata_snapshot.items()
                if key in metadata_fields and isinstance(value, str)
            }
        )
        dialog.author_edit.setText(metadata.author)
        dialog.contact_edit.setText(metadata.contact)
        dialog.project_url_edit.setText(metadata.project_url)
        dialog.change_summary_edit.setText(metadata.change_summary)
        dialog.changelog_edit.setPlainText(metadata.changelog)
        if not dialog.exec():
            return
        try:
            existing = self.catalog.model.extensions.get(extension_id)
            self.catalog.store.add_embedded_script(
                source,
                extension_id=extension_id,
                expected_revision=revision,
                name=str(
                    requirement.metadata_snapshot.get("extension_name", extension_id)
                ),
                metadata=dialog.metadata,
                source_modified_at=(
                    str(requirement.metadata_snapshot["source_modified_at"])
                    if requirement.metadata_snapshot.get("source_modified_at")
                    is not None
                    else None
                ),
                expected_record_generation=(
                    None if existing is None else existing.record_generation
                ),
                check_record_generation=True,
            )
            self.catalog.refresh()
            record = self.catalog.model.extensions[extension_id]
            self.execution.validate_and_enable(
                extension_id,
                expected_record_generation=record.record_generation,
            )
            self.catalog.refresh()
        except Exception:
            erlab.interactive.utils.MessageDialog.critical(
                self._manager,
                "Extension Error",
                "The embedded extension could not be enabled.",
                detailed_text=traceback.format_exc(),
            )

    def notify_unavailable_workspace_requirements(self) -> None:
        unavailable = [
            item
            for item in self.resolved_workspace_requirements()
            if item.state != "ready"
        ]
        if not unavailable:
            return
        states = ", ".join(
            f"{item.requirement.extension_id}: {item.state}" for item in unavailable
        )
        dialog = erlab.interactive.utils.MessageDialog(
            self._manager,
            title="Workspace Extensions Unavailable",
            text=(
                "Saved data was loaded, but some extension-dependent actions are "
                "disabled."
            ),
            informative_text=(
                f"{states}\nUse Save Workspace As to preserve the original file."
            ),
            icon_pixmap=QtWidgets.QStyle.StandardPixmap.SP_MessageBoxWarning,
        )
        dialog.exec()

    @QtCore.Slot()
    def refresh_environment_packages(self) -> None:
        if erlab.utils.misc._IS_PACKAGED:
            return
        try:
            self.catalog.store.refresh_environment_packages()
            self.catalog.refresh()
        except Exception:
            erlab.interactive.utils.MessageDialog.critical(
                self._manager,
                "Extension Error",
                "Environment packages could not be refreshed.",
                detailed_text=traceback.format_exc(),
            )

    def close(self) -> None:
        if self._closed:
            return
        self.execution.shutdown()
        if self.menu is not None and erlab.interactive.utils.qt_is_valid(self.menu):
            with contextlib.suppress(TypeError, RuntimeError):
                self.menu.aboutToShow.disconnect(self._menu_show_slot)
        for menu, slot in self._context_menu_connections:
            if erlab.interactive.utils.qt_is_valid(menu):
                with contextlib.suppress(TypeError, RuntimeError):
                    menu.aboutToShow.disconnect(slot)
        self._context_menu_connections.clear()
        for action, slot in (
            (self.add_script_action, self._add_script_slot),
            (self.manage_action, self._show_manager_slot),
            (self.requirements_action, self._show_requirements_slot),
            (self.progress_action, self._show_progress_slot),
            (self.refresh_packages_action, self._refresh_packages_slot),
        ):
            if erlab.interactive.utils.qt_is_valid(action):
                with contextlib.suppress(TypeError, RuntimeError):
                    action.triggered.disconnect(slot)
        if erlab.interactive.utils.qt_is_valid(self._manage_dialog):
            with contextlib.suppress(TypeError, RuntimeError):
                self._manage_dialog.action_requested.disconnect(
                    self._manage_action_slot
                )
            self._manage_dialog.close()
            self._manage_dialog.deleteLater()
        with contextlib.suppress(TypeError, RuntimeError):
            self.catalog.changed.disconnect(self._catalog_changed_slot)
        self.catalog.close()
        self._closed = True
