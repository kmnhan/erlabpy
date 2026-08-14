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
import uuid
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
    _MissingScriptsDialog,
    _RoutineSelectionDialog,
    _SourceReviewDialog,
    _SourceViewerDialog,
    _WorkspaceRequirementsDialog,
)
from erlab.interactive.imagetool.manager._extensions._execution import (
    _DecoratedLoaderAdapter,
    _ExtensionExecutionController,
    _ExtensionLoaderCall,
)
from erlab.interactive.imagetool.manager._extensions._models import (
    _ExtensionRecord,
    _ExtensionSource,
    _ResolvedWorkspaceRequirement,
    _WorkspaceExtensionRequirement,
)
from erlab.interactive.imagetool.manager._registry import live_manager_records

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable

    import xarray as xr

    from erlab.extensions._api import _CapabilityStatus
    from erlab.interactive.explorer._tabbed_explorer import _TabbedExplorer
    from erlab.interactive.imagetool._provenance._model import FileLoadSource
    from erlab.interactive.imagetool._provenance._operations import (
        ExtensionRoutineOperation,
    )
    from erlab.interactive.imagetool.manager._mainwindow import ImageToolManager

logger = logging.getLogger(__name__)


class _ExtensionSourceHashMismatchError(FileNotFoundError):
    """Report that available script bytes do not match the required source."""


class _ExtensionController(QtCore.QObject):
    """Own extension UI, script approval, and execution for one manager.

    Script registrations are application-wide. Workspace ``Open Without`` decisions
    stay on this controller and do not propagate to other managers.
    """

    def __init__(self, manager: ImageToolManager) -> None:
        super().__init__(manager)
        self._manager = manager
        self.catalog = _ExtensionCatalog(parent=self)
        self.execution = _ExtensionExecutionController(manager, self.catalog)
        self._catalog_changed_slot = self._catalog_changed
        self._manage_action_slot = self._manage_action
        self.catalog.changed.connect(self._catalog_changed_slot)
        self._recent: deque[tuple[str, str]] = deque(maxlen=8)
        self._routine_action_groups: list[
            tuple[QtWidgets.QMenu, list[QtGui.QAction]]
        ] = []
        self._workspace_requirements: tuple[_WorkspaceExtensionRequirement, ...] = ()
        self._workspace_embedded_sources: dict[tuple[str, str], bytes] = {}
        self._workspace_unresolved_embedded_objects: dict[
            str, tuple[bytes, str | None]
        ] = {}
        self._explorer_loaders: dict[str, erlab.io.dataloader.LoaderBase] = {}
        self._closed = False
        self._missing_script_prompt_shown = False
        self._missing_scripts_dialog: _MissingScriptsDialog | None = None
        self._missing_scripts_dialog_slots: (
            tuple[Callable[[str], None], Callable[[str], None], Callable[[int], None]]
            | None
        ) = None
        self._unresolved_workspace_requirement_payloads: tuple[typing.Any, ...] = ()
        self._manage_dialog = _ManageExtensionsDialog(manager)
        self._manage_dialog.action_requested.connect(self._manage_action_slot)
        self._manage_add_script_slot = self.add_script
        self._manage_open_folder_slot = self._open_extensions_folder
        self._manage_selection_slot = self._refresh_removal_eligibility
        self._manage_activated_slot = self._refresh_removal_eligibility
        self._execution_state_slot = self._refresh_removal_eligibility
        self._manage_dialog.add_script_requested.connect(self._manage_add_script_slot)
        self._manage_dialog.open_folder_requested.connect(self._manage_open_folder_slot)
        self._manage_dialog.selection_changed.connect(self._manage_selection_slot)
        self._manage_dialog.activated.connect(self._manage_activated_slot)
        self.execution.queue_changed.connect(self._execution_state_slot)
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
        self.menu: QtWidgets.QMenu | None = None
        self._sync_explorer_loaders()
        self._missing_script_prompt_slot = self._prompt_for_missing_scripts
        QtCore.QTimer.singleShot(0, self._missing_script_prompt_slot)

    def create_menu(self, menu_bar: QtWidgets.QMenuBar) -> QtWidgets.QMenu:
        menu = typing.cast("QtWidgets.QMenu", menu_bar.addMenu("&Extensions"))
        menu.setObjectName("manager_extensions_menu")
        menu.aboutToShow.connect(self._menu_show_slot)
        self.menu = menu
        self._populate_routine_menu(menu, compact=False, update_actions=False)
        return menu

    def add_context_submenu(self, parent_menu: QtWidgets.QMenu) -> QtWidgets.QMenu:
        menu = QtWidgets.QMenu("Extensions", parent_menu)
        menu.setObjectName("manager_extensions_context_menu")
        populate_slot = functools.partial(
            self._populate_routine_menu, menu, compact=True
        )
        menu.aboutToShow.connect(populate_slot)
        self._context_menu_connections.append((menu, populate_slot))
        self._populate_routine_menu(menu, compact=True, update_actions=False)
        parent_menu.addMenu(menu)
        return menu

    def _enabled_routines(self) -> tuple[tuple[str, str, RoutineDescriptor], ...]:
        entries: list[tuple[str, str, RoutineDescriptor]] = []
        for record in self.catalog.model.extensions.values():
            if not record.enabled:
                continue
            source = record.source
            if not self.catalog.store.source_available(record, source.source_hash):
                continue
            entries.extend((record.id, record.name, item) for item in source.routines)
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
            if not record.enabled:
                continue
            source = record.source
            if not self.catalog.store.source_available(record, source.source_hash):
                continue
            for descriptor in source.loaders:
                patterns = tuple(f"*{suffix}" for suffix in descriptor.extensions) or (
                    "*",
                )
                name_filter = f"{descriptor.name} ({' '.join(patterns)})"
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
                entries[name_filter] = (
                    self._loader_call(record, source, descriptor),
                    {
                        parameter.id: parameter.default
                        for parameter in descriptor.parameters
                        if not parameter.required
                    },
                )
                owners[name_filter] = record.id
        return entries

    def _loader_call(
        self,
        record: _ExtensionRecord,
        source: _ExtensionSource,
        descriptor: LoaderDescriptor,
        *,
        source_hash: str | None = None,
    ) -> _ExtensionLoaderCall:
        """Create one manager-local call pinned to validated catalog state."""
        pinned_source_hash = source.source_hash if source_hash is None else source_hash
        self._remember_registered_source(record, pinned_source_hash)
        return _ExtensionLoaderCall(
            manager_session_id=self._manager._manager_record.internal_id,
            catalog_generation=self.catalog.model.generation,
            extension_id=record.id,
            extension_name=record.name,
            source_hash=pinned_source_hash,
            loader_id=descriptor.id,
            descriptor=descriptor,
            source_path=self.catalog.store.executable_source_path(
                record.id, pinned_source_hash
            ),
            executor=self.execution.run_loader,
        )

    def replay_loader(
        self, load_source: FileLoadSource
    ) -> xr.DataArray | xr.Dataset | xr.DataTree:
        """Run one exact file-provenance loader through the manager queue."""
        replay_call = load_source.replay_call
        if (
            replay_call is None
            or replay_call.kind != "extension_loader"
            or replay_call.source_hash is None
            or replay_call.capability_id is None
        ):
            raise erlab.extensions.ExtensionExecutionError(
                "Extension loader replay metadata is incomplete"
            )
        catalog = self.catalog.store.read()
        record = catalog.extensions.get(replay_call.target)
        source = (
            record.source
            if record is not None
            and record.source.source_hash == replay_call.source_hash
            else None
        )
        try:
            global_status = self.catalog.store.capability_status(
                replay_call.target,
                replay_call.source_hash,
                "loader",
                replay_call.capability_id,
            )
        except KeyError:
            global_status = "missing-source"
        global_ready = (
            global_status == "ready"
            and record is not None
            and record.enabled
            and source is not None
            and source.approved
        )
        if not global_ready:
            raise erlab.extensions.ExtensionExecutionError(
                "The extension loader source is not available"
            )
        if record is None or source is None:
            raise erlab.extensions.ExtensionExecutionError(
                "The extension loader source is not available"
            )
        descriptor = next(
            (item for item in source.loaders if item.id == replay_call.capability_id),
            None,
        )
        if descriptor is None:
            raise erlab.extensions.ExtensionExecutionError(
                f"Loader {replay_call.capability_id!r} is not available"
            )
        call = self._loader_call(
            record,
            source,
            descriptor,
            source_hash=replay_call.source_hash,
        )
        return self.execution.run_loader(
            call,
            pathlib.Path(load_source.path),
            _deserialize_loader_kwargs(replay_call.kwargs),
        )

    def capability_status(
        self,
        extension_id: str,
        source_hash: str,
        kind: str,
        capability_id: str,
    ) -> _CapabilityStatus:
        """Resolve application catalog state for this manager."""
        try:
            global_status = self.catalog.store.capability_status(
                extension_id,
                source_hash,
                kind,
                capability_id,
            )
        except KeyError:
            global_status = "missing-source"
        return global_status

    @property
    def explorer_loaders(self) -> dict[str, erlab.io.dataloader.LoaderBase]:
        """Manager-local loader adapters used by existing Data Explorer tabs."""
        return self._explorer_loaders

    def _sync_explorer_loaders(self) -> None:
        updated: dict[str, erlab.io.dataloader.LoaderBase] = {}
        for record in self.catalog.model.extensions.values():
            if not record.enabled:
                continue
            source = record.source
            if not self.catalog.store.source_available(record, source.source_hash):
                continue
            for descriptor in source.loaders:
                adapter = _DecoratedLoaderAdapter(
                    self._loader_call(record, source, descriptor)
                )
                updated[adapter.name] = adapter
        self._explorer_loaders.clear()
        self._explorer_loaders.update(updated)

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

    def _populate_routine_menu(
        self,
        menu: QtWidgets.QMenu,
        *,
        compact: bool,
        update_actions: bool = True,
    ) -> None:
        menu.clear()
        retained = next(
            (
                actions
                for retained_menu, actions in self._routine_action_groups
                if retained_menu is menu
            ),
            None,
        )
        if retained is None:
            retained = []
            self._routine_action_groups.append((menu, retained))
        else:
            retained.clear()
        run_action = typing.cast("QtGui.QAction", menu.addAction("Run Routine…"))
        run_action.setObjectName("manager_run_extension_routine_action")
        run_action.setProperty("requiresImageTool", True)
        run_action.triggered.connect(self.select_routine)
        retained.append(run_action)
        routines = self._enabled_routines()
        favorite_keys = set(self.catalog.model.routine_favorites)
        if routines or not compact:
            menu.addSeparator()
        favorite_entries = tuple(
            entry for entry in routines if (entry[0], entry[2].id) in favorite_keys
        )
        if favorite_entries:
            favorites = typing.cast("QtWidgets.QMenu", menu.addMenu("Favorites"))
            favorites_action = favorites.menuAction()
            if favorites_action is not None:
                favorites_action.setProperty("requiresImageTool", True)
                retained.append(favorites_action)
            for entry in favorite_entries:
                self._add_routine_action(favorites, entry, retained)
        recent_keys = tuple(
            key for key in self._recent if key in {(e[0], e[2].id) for e in routines}
        )
        if recent_keys:
            recent_menu = typing.cast("QtWidgets.QMenu", menu.addMenu("Recent"))
            recent_action = recent_menu.menuAction()
            if recent_action is not None:
                recent_action.setProperty("requiresImageTool", True)
                retained.append(recent_action)
            by_key = {(entry[0], entry[2].id): entry for entry in routines}
            for key in recent_keys:
                self._add_routine_action(recent_menu, by_key[key], retained)
        categories: dict[str, list[tuple[str, str, RoutineDescriptor]]] = defaultdict(
            list
        )
        for entry in routines:
            categories[entry[2].category].append(entry)
        for category in sorted(categories):
            category_menu = typing.cast("QtWidgets.QMenu", menu.addMenu(category))
            category_action = category_menu.menuAction()
            if category_action is not None:
                category_action.setProperty("requiresImageTool", True)
                retained.append(category_action)
            for entry in sorted(categories[category], key=lambda item: item[2].name):
                self._add_routine_action(category_menu, entry, retained)
        if compact:
            if update_actions:
                self.update_actions()
            return
        if routines:
            menu.addSeparator()
        menu.addAction(self.add_script_action)
        menu.addAction(self.manage_action)
        menu.addAction(self.requirements_action)
        if update_actions:
            self.update_actions()

    def _add_routine_action(
        self,
        menu: QtWidgets.QMenu,
        entry: tuple[str, str, RoutineDescriptor],
        retained: list[QtGui.QAction],
    ) -> None:
        extension_id, extension_name, descriptor = entry
        action = typing.cast("QtGui.QAction", menu.addAction(descriptor.name))
        action.setData((extension_id, descriptor.id))
        action.setProperty("requiresImageTool", True)
        action.setToolTip(descriptor.summary or extension_name)
        action.triggered.connect(
            lambda _checked=False, ext=extension_id, routine=descriptor.id: (
                self.run_routine(ext, routine)
            )
        )
        retained.append(action)

    def update_actions(self) -> None:
        """Apply manager selection state to all retained routine actions."""
        enabled = len(self._manager._selected_imagetool_targets()) == 1
        live_groups: list[tuple[QtWidgets.QMenu, list[QtGui.QAction]]] = []
        for menu, actions in self._routine_action_groups:
            if not erlab.interactive.utils.qt_is_valid(menu):
                continue
            live_actions: list[QtGui.QAction] = []
            for action in actions:
                if not erlab.interactive.utils.qt_is_valid(action):
                    continue
                if action.property("requiresImageTool") is True:
                    action.setEnabled(enabled)
                live_actions.append(action)
            live_groups.append((menu, live_actions))
        self._routine_action_groups = live_groups

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
        dialog = _RoutineSelectionDialog(
            routines,
            self._manager,
            favorites=self.catalog.model.routine_favorites,
        )
        favorite_slot = self._set_routine_favorite
        dialog.favorite_requested.connect(favorite_slot)
        if dialog.exec() and dialog.selection is not None:
            self.run_routine(*dialog.selection)
        with contextlib.suppress(TypeError, RuntimeError):
            dialog.favorite_requested.disconnect(favorite_slot)

    @QtCore.Slot(str, str, bool)
    def _set_routine_favorite(
        self, extension_id: str, routine_id: str, favorite: bool
    ) -> None:
        try:
            self.catalog.store.set_routine_favorite(
                extension_id, routine_id, favorite=favorite
            )
            self.catalog.refresh()
        except Exception:
            erlab.interactive.utils.MessageDialog.critical(
                self._manager,
                "Extension Error",
                "The routine favorite could not be changed.",
                detailed_text=traceback.format_exc(),
            )

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
        if record is None or not record.enabled:
            return
        descriptor = next(
            (item for item in record.source.routines if item.id == routine_id),
            None,
        )
        if descriptor is None:
            return
        self._remember_registered_source(record, record.source.source_hash)
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

    def _remember_registered_source(
        self, record: _ExtensionRecord, source_hash: str
    ) -> None:
        """Retain script bytes used by this workspace before registration changes."""
        key = (record.id, source_hash)
        if key in self._workspace_embedded_sources:
            return
        try:
            path = self.catalog.store.executable_source_path(record.id, source_hash)
            source = path.read_bytes()
        except (KeyError, OSError, _ExtensionCatalogConflictError):
            return
        if hashlib.sha256(source).hexdigest() == source_hash:
            self._workspace_embedded_sources[key] = source

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
            reviewed_source_hash = hashlib.sha256(reviewed_source).hexdigest()
            source_text = reviewed_source.decode("utf-8")
            catalog_extension_id = _safe_extension_id(extension_id or path.stem)
            existing = self.catalog.model.extensions.get(catalog_extension_id)
            if extension_id is None:
                resolved_path = path.expanduser().resolve()
                matching_record = next(
                    (
                        record
                        for record in self.catalog.model.extensions.values()
                        if (source_path := record.source.source_path) is not None
                        and pathlib.Path(source_path).expanduser().resolve()
                        == resolved_path
                    ),
                    None,
                )
                if matching_record is not None:
                    catalog_extension_id = matching_record.id
                    existing = matching_record
            if extension_id is None and existing is not None:
                current_source = existing.source
                same_source = (
                    current_source.source_path is not None
                    and pathlib.Path(current_source.source_path).expanduser().resolve()
                    == resolved_path
                )
                if not same_source:
                    catalog_extension_id = _safe_extension_id(
                        f"{path.stem}-{uuid.uuid4().hex[:8]}"
                    )
                    existing = None
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
        if not dialog.exec():
            return False
        try:
            _catalog, _source_hash, changed = self.catalog.store.add_script(
                path,
                extension_id=catalog_extension_id,
                expected_source_hash=reviewed_source_hash,
                expected_record_generation=(
                    None if existing is None else existing.record_generation
                ),
                check_record_generation=True,
            )
            self.catalog.refresh()
            record = self.catalog.model.extensions[catalog_extension_id]
            source = record.source
            if not changed and record.enabled and source.approved:
                self._manager._status_bar.showMessage(
                    "The registered script contents did not change.",
                    4000,
                )
                return True
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

    def _missing_script_records(
        self, extension_ids: Collection[str] | None = None
    ) -> tuple[_ExtensionRecord, ...]:
        """Return enabled scripts whose registered current file cannot be read."""
        missing: list[_ExtensionRecord] = []
        for record in self.catalog.model.extensions.values():
            if not record.enabled or (
                extension_ids is not None and record.id not in extension_ids
            ):
                continue
            source = record.source
            if source.source_path is None:
                missing.append(record)
                continue
            try:
                pathlib.Path(source.source_path).read_bytes()
            except OSError:
                missing.append(record)
        return tuple(missing)

    @QtCore.Slot()
    def _prompt_for_missing_scripts(self) -> None:
        """Prompt for missing registered scripts after the event loop starts."""
        self._show_missing_script_recovery()

    def _show_missing_script_recovery(
        self,
        *,
        extension_ids: Collection[str] | None = None,
        repeat: bool = False,
    ) -> bool:
        """Offer one session-scoped recovery dialog for all missing scripts."""
        if self._closed or not erlab.interactive.utils.qt_is_valid(self._manager):
            return False
        current_dialog = self._missing_scripts_dialog
        if current_dialog is not None and erlab.interactive.utils.qt_is_valid(
            current_dialog
        ):
            current_dialog.raise_()
            return True
        if self._missing_script_prompt_shown and not repeat:
            return False
        records = self._missing_script_records(extension_ids)
        if not records:
            return False
        self._missing_script_prompt_shown = True
        dialog = _MissingScriptsDialog(records, self._manager)

        def refresh_dialog() -> None:
            if not erlab.interactive.utils.qt_is_valid(dialog):
                return
            remaining = self._missing_script_records(extension_ids)
            if not remaining:
                dialog.accept()
            else:
                dialog.set_records(remaining)

        def locate_slot(extension_id: str) -> None:
            self._locate_missing_script(extension_id)
            refresh_dialog()

        def restore_slot(extension_id: str) -> None:
            self._restore_missing_script(extension_id)
            refresh_dialog()

        def finished_slot(_result: int) -> None:
            if self._missing_scripts_dialog is not dialog:
                return
            with contextlib.suppress(TypeError, RuntimeError):
                dialog.locate_requested.disconnect(locate_slot)
            with contextlib.suppress(TypeError, RuntimeError):
                dialog.restore_requested.disconnect(restore_slot)
            with contextlib.suppress(TypeError, RuntimeError):
                dialog.finished.disconnect(finished_slot)
            self._missing_scripts_dialog = None
            self._missing_scripts_dialog_slots = None

        dialog.locate_requested.connect(locate_slot)
        dialog.restore_requested.connect(restore_slot)
        dialog.finished.connect(finished_slot)
        self._missing_scripts_dialog = dialog
        self._missing_scripts_dialog_slots = (
            locate_slot,
            restore_slot,
            finished_slot,
        )
        dialog.show()
        dialog.raise_()
        return True

    def _locate_missing_script(self, extension_id: str) -> bool:
        record = self.catalog.model.extensions.get(extension_id)
        if record is None:
            return False
        source = record.source
        initial = source.source_path or self._manager._recent_or_default_directory()
        path_value, _selected_filter = QtWidgets.QFileDialog.getOpenFileName(
            self._manager,
            f"Locate {record.name}",
            initial or "",
            "Python scripts (*.py)",
        )
        if not path_value:
            return False
        path = pathlib.Path(path_value).expanduser().resolve()
        try:
            source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            erlab.interactive.utils.MessageDialog.critical(
                self._manager,
                "Extension Error",
                "The selected script could not be read.",
                detailed_text=traceback.format_exc(),
            )
            return False
        if source_hash != source.source_hash:
            return self._review_and_add(path, extension_id=extension_id)
        try:
            self.catalog.store.add_script(
                path,
                extension_id=extension_id,
                expected_source_hash=source.source_hash,
                expected_record_generation=record.record_generation,
                check_record_generation=True,
            )
            self.catalog.refresh()
            return True  # noqa: TRY300 - success exits before the shared error path.
        except Exception:
            erlab.interactive.utils.MessageDialog.critical(
                self._manager,
                "Extension Error",
                "The script location could not be updated.",
                detailed_text=traceback.format_exc(),
            )
            self.catalog.refresh()
            return False

    def _save_source_as_user_file(
        self,
        source: bytes,
        *,
        title: str,
        suggested_name: str,
    ) -> pathlib.Path | None:
        """Write recovery source without replacing a different user file."""
        initial_directory = self._manager._recent_or_default_directory() or ""
        suggested_path = os.fspath(pathlib.Path(initial_directory) / suggested_name)
        destination_value, _selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self._manager,
            title,
            suggested_path,
            "Python scripts (*.py)",
        )
        if not destination_value:
            return None
        destination = pathlib.Path(destination_value).expanduser().resolve()
        if destination.suffix.lower() != ".py":
            destination = destination.with_suffix(".py")

        def write_new_file() -> None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            save_file = QtCore.QSaveFile(os.fspath(destination))
            if not save_file.open(QtCore.QIODevice.OpenModeFlag.WriteOnly):
                raise OSError(save_file.errorString())
            if save_file.write(source) != len(source):
                save_file.cancelWriting()
                raise OSError(save_file.errorString())
            if not save_file.commit():
                raise OSError(save_file.errorString())

        try:
            if destination.exists():
                if destination.read_bytes() != source:
                    QtWidgets.QMessageBox.warning(
                        self._manager,
                        "Choose a Different File",
                        "The selected file contains different source. Choose a new "
                        "filename so that it is not overwritten.",
                    )
                    return None
                return destination
            write_new_file()
        except OSError:
            erlab.interactive.utils.MessageDialog.critical(
                self._manager,
                "Extension Error",
                "The extension script could not be saved.",
                detailed_text=traceback.format_exc(),
            )
            return None
        return destination

    def _restore_missing_script(self, extension_id: str) -> bool:
        record = self.catalog.model.extensions.get(extension_id)
        if record is None:
            return False
        source_hash = record.source.source_hash
        try:
            source = self.catalog.store.recovery_source_path(
                extension_id, source_hash
            ).read_bytes()
        except OSError:
            erlab.interactive.utils.MessageDialog.critical(
                self._manager,
                "Extension Error",
                "The stored recovery source is unavailable.",
                detailed_text=traceback.format_exc(),
            )
            return False
        destination = self._save_source_as_user_file(
            source,
            title=f"Restore {record.name}",
            suggested_name=record.name,
        )
        if destination is None:
            return False
        try:
            self.catalog.store.add_script(
                destination,
                extension_id=extension_id,
                expected_source_hash=source_hash,
                expected_record_generation=record.record_generation,
                check_record_generation=True,
            )
            self.catalog.refresh()
            return True  # noqa: TRY300 - success exits before the shared error path.
        except Exception:
            erlab.interactive.utils.MessageDialog.critical(
                self._manager,
                "Extension Error",
                "The restored script could not be registered.",
                detailed_text=traceback.format_exc(),
            )
            self.catalog.refresh()
            return False

    @QtCore.Slot()
    def show_manager(self) -> None:
        self._refresh_manage_dialog()
        self._manage_dialog.show()
        self._manage_dialog.raise_()

    def _refresh_manage_dialog(self) -> None:
        managed_paths: dict[tuple[str, str], str] = {}
        for record in self.catalog.model.extensions.values():
            source_hash = record.source.source_hash
            with contextlib.suppress(FileNotFoundError, KeyError):
                managed_paths[(record.id, source_hash)] = os.fspath(
                    self.catalog.store.recovery_source_path(record.id, source_hash)
                )
        self._manage_dialog.set_catalog(
            self.catalog.model,
            self._catalog_source_states(),
            managed_paths=managed_paths,
        )
        self._refresh_removal_eligibility()

    @QtCore.Slot()
    def _open_extensions_folder(self) -> None:
        try:
            self.catalog.store.directory.mkdir(parents=True, exist_ok=True)
            erlab.utils.misc.open_in_file_manager(self.catalog.store.directory)
        except OSError:
            erlab.interactive.utils.MessageDialog.critical(
                self._manager,
                "Extension Error",
                "The extensions folder could not be opened.",
                detailed_text=traceback.format_exc(),
            )

    def _catalog_source_states(self) -> dict[tuple[str, str], str]:
        states: dict[tuple[str, str], str] = {}
        for record in self.catalog.model.extensions.values():
            source = record.source
            source_hash = source.source_hash
            key = (record.id, source_hash)
            if source.source_path is None:
                states[key] = "No registered source file"
                continue
            source_path = pathlib.Path(source.source_path)
            if not source_path.is_file():
                states[key] = "Source file missing"
                continue
            try:
                current_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
            except OSError:
                states[key] = "Source file unreadable"
            else:
                states[key] = (
                    "Ready" if current_hash == source_hash else "Source file changed"
                )
        return states

    @QtCore.Slot(str, str)
    def _manage_action(self, action_id: str, extension_id: str) -> None:
        record = self.catalog.model.extensions.get(extension_id)
        if record is None:
            return
        try:
            if action_id == "reload":
                source_path = record.source.source_path
                if source_path is None:
                    self._restore_missing_script(extension_id)
                    return
                if not pathlib.Path(source_path).is_file():
                    self._locate_missing_script(extension_id)
                    return
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
            elif action_id == "remove":
                self._remove_extension(record)
                return
            elif action_id.startswith("embedding:"):
                policy = action_id.removeprefix("embedding:")
                if policy not in {"referenced", "always", "never"}:
                    return
                self.catalog.store.update_record(
                    extension_id,
                    expected_record_generation=record.record_generation,
                    embed_policy=typing.cast(
                        'typing.Literal["referenced", "always", "never"]', policy
                    ),
                )
            elif action_id == "error":
                source = record.source
                if source.import_error:
                    erlab.interactive.utils.MessageDialog.critical(
                        self._manager,
                        "Extension Import Error",
                        "The extension could not be imported.",
                        detailed_text=source.import_error,
                    )
                return
            elif action_id == "view_source":
                self._show_source(record.id, record.source.source_hash)
                return
            elif action_id in {"open_source", "reveal_source", "copy_source"}:
                source_path = record.source.source_path
                if source_path is None or not pathlib.Path(source_path).is_file():
                    QtWidgets.QMessageBox.information(
                        self._manager,
                        "Source File Unavailable",
                        "The original source file is unavailable.",
                    )
                    return
                if action_id == "open_source":
                    QtGui.QDesktopServices.openUrl(
                        QtCore.QUrl.fromLocalFile(source_path)
                    )
                elif action_id == "reveal_source":
                    erlab.utils.misc.open_in_file_manager(source_path)
                else:
                    clipboard = QtWidgets.QApplication.clipboard()
                    if clipboard is not None:
                        clipboard.setText(source_path)
                return
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
        self._refresh_manage_dialog()

    def _show_source(self, extension_id: str, source_hash: str) -> None:
        record = self.catalog.model.extensions.get(extension_id)
        if record is None:
            return
        try:
            source_text = self.source_bytes(extension_id, source_hash).decode("utf-8")
        except (KeyError, OSError, UnicodeError):
            erlab.interactive.utils.MessageDialog.critical(
                self._manager,
                "Extension Error",
                "The stored extension source could not be read.",
                detailed_text=traceback.format_exc(),
            )
            return
        dialog = _SourceViewerDialog(
            source_text,
            self._manager,
            title=f"Running Source — {record.name}",
        )
        dialog.exec()

    @QtCore.Slot()
    @QtCore.Slot(str)
    def _refresh_removal_eligibility(self, _extension_id: str = "") -> None:
        extension_id = self._manage_dialog.selected_extension_id
        reason = None if extension_id is None else self._removal_blocker(extension_id)
        self._manage_dialog.set_removal_reason(reason)

    def _removal_blocker(self, extension_id: str) -> str | None:
        record = self.catalog.model.extensions.get(extension_id)
        if record is None:
            return None
        try:
            other_managers = tuple(
                manager
                for manager in live_manager_records()
                if manager.internal_id != self._manager._manager_record.internal_id
            )
        except Exception:
            return "Removal is unavailable because other managers could not be checked."
        if other_managers:
            descriptions = "; ".join(
                f"Manager {manager.index}"
                + (f" ({manager.workspace_path})" if manager.workspace_path else "")
                for manager in other_managers
            )
            return f"Close the other ImageTool Managers first: {descriptions}."
        if self.execution.uses_extension(extension_id):
            return "Wait for this extension's active or queued jobs to finish."
        requirements = tuple(
            requirement
            for requirement in self.collect_workspace_requirements()
            if requirement.extension_id == extension_id
        )
        if requirements:
            workspace_path = self._manager._manager_record.workspace_path
            location = (
                "the current workspace" if workspace_path is None else workspace_path
            )
            return f"Remove this extension from {location} before you delete it."
        return None

    def _remove_extension(self, record: _ExtensionRecord) -> None:
        blocker = self._removal_blocker(record.id)
        if blocker is not None:
            QtWidgets.QMessageBox.information(
                self._manager, "Extension Cannot Be Removed", blocker
            )
            self._refresh_removal_eligibility()
            return
        object_paths = {
            self.catalog.store.objects_directory / record.source.object_name
        }
        total_size = sum(path.stat().st_size for path in object_paths if path.is_file())
        current = record.source
        dialog = QtWidgets.QMessageBox(self._manager)
        dialog.setObjectName("manager_extension_remove_confirmation")
        dialog.setWindowTitle("Remove Extension")
        dialog.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        dialog.setText(f"Remove {record.name} from ERLab?")
        dialog.setInformativeText(
            "ERLab stores one recovery copy "
            f"({erlab.utils.formatting.format_nbytes(total_size)}). ERLab will delete "
            "that copy if no other extension uses it. The original source file will "
            "not be deleted."
        )
        dialog.setDetailedText(
            f"Original source: {current.source_path or 'No external source file'}\n\n"
            "Closed workspaces that did not embed this script can lose replay "
            "capability."
        )
        dialog.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.Cancel
        )
        dialog.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        if dialog.exec() != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        blocker = self._removal_blocker(record.id)
        if blocker is not None:
            QtWidgets.QMessageBox.information(
                self._manager, "Extension Cannot Be Removed", blocker
            )
            self._refresh_removal_eligibility()
            return
        _catalog, retained = self.catalog.store.remove_script(
            record.id,
            expected_record_generation=record.record_generation,
        )
        self.catalog.refresh()
        if retained is not None:
            QtWidgets.QMessageBox.warning(
                self._manager,
                "Extension Removed",
                "The extension was removed, but staged source files could not be "
                f"deleted. They remain at {retained}.",
            )

    @QtCore.Slot(object)
    def _catalog_changed(self, _model: object) -> None:
        self._sync_explorer_loaders()
        self._refresh_manage_dialog()
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
                    item.source_hash,
                    item.capability_kind,
                    item.capability_id,
                )
            ].append(item)

        def merged_persisted_requirement(
            key: tuple[str, str, str, str],
        ) -> _WorkspaceExtensionRequirement | None:
            """Merge per-node state for one immutable source dependency."""
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
                        extension_operation.source_hash,
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
                    or replay_call.source_hash is None
                    or replay_call.capability_id is None
                ):
                    continue
                key = (
                    replay_call.target,
                    replay_call.source_hash,
                    replay_call.capability_id,
                )
                loader_references[key].add(uid)
                loader_files[key].add(current_spec.file_load_source.path)
        requirements: list[_WorkspaceExtensionRequirement] = []
        for key, node_uids in references.items():
            extension_id, source_hash, routine_id = key
            record = self.catalog.model.extensions.get(extension_id)
            operation = typing.cast("ExtensionRoutineOperation", operations[key])
            previous = merged_persisted_requirement(
                (
                    extension_id,
                    source_hash,
                    "routine",
                    routine_id,
                )
            )
            if previous is not None:
                node_uids.update(
                    set(previous.referencing_nodes).difference(loaded_node_uids)
                )
            metadata = {} if previous is None else dict(previous.metadata_snapshot)
            record_source = (
                None
                if record is None or record.source.source_hash != source_hash
                else record.source
            )
            if record is not None and record_source is not None:
                for obsolete_key in (
                    "author",
                    "contact",
                    "project_url",
                    "change_summary",
                    "changelog",
                ):
                    metadata.pop(obsolete_key, None)
                metadata = {
                    **metadata,
                    "extension_name": record.name,
                    "routine_name": operation.routine_name,
                }
            elif previous is None:
                metadata = {
                    "extension_name": operation.extension_name,
                    "routine_name": operation.routine_name,
                }
            if (
                previous is None
                and record_source is not None
                and record_source.source_modified_at is not None
            ):
                metadata["source_modified_at"] = record_source.source_modified_at
            requirements.append(
                _WorkspaceExtensionRequirement(
                    extension_id=extension_id,
                    capability_id=routine_id,
                    capability_kind="routine",
                    source_hash=source_hash,
                    extension_api_version=(
                        EXTENSION_API_VERSION
                        if previous is None
                        else previous.extension_api_version
                    ),
                    metadata_snapshot=metadata,
                    embedded_object_id=self._embedded_script_object_id(
                        record=record,
                        source_hash=source_hash,
                        previous=previous,
                    ),
                    referencing_nodes=tuple(sorted(node_uids)),
                )
            )
        for key, node_uids in loader_references.items():
            extension_id, source_hash, loader_id = key
            record = self.catalog.model.extensions.get(extension_id)
            previous = merged_persisted_requirement(
                (
                    extension_id,
                    source_hash,
                    "loader",
                    loader_id,
                )
            )
            unresolved_node_uids: set[str] = set()
            if previous is not None:
                unresolved_node_uids = set(previous.referencing_nodes).difference(
                    loaded_node_uids
                )
                node_uids.update(unresolved_node_uids)
            loader_metadata = (
                {} if previous is None else dict(previous.metadata_snapshot)
            )
            loader_record_source = (
                None
                if record is None or record.source.source_hash != source_hash
                else record.source
            )
            if record is not None and loader_record_source is not None:
                for obsolete_key in (
                    "author",
                    "contact",
                    "project_url",
                    "change_summary",
                    "changelog",
                ):
                    loader_metadata.pop(obsolete_key, None)
                loader_metadata = {
                    **loader_metadata,
                    "extension_name": record.name,
                }
            if (
                previous is None
                and loader_record_source is not None
                and loader_record_source.source_modified_at is not None
            ):
                loader_metadata["source_modified_at"] = (
                    loader_record_source.source_modified_at
                )
            requirements.append(
                _WorkspaceExtensionRequirement(
                    extension_id=extension_id,
                    capability_id=loader_id,
                    capability_kind="loader",
                    source_hash=source_hash,
                    extension_api_version=(
                        EXTENSION_API_VERSION
                        if previous is None
                        else previous.extension_api_version
                    ),
                    metadata_snapshot=loader_metadata,
                    embedded_object_id=self._embedded_script_object_id(
                        record=record,
                        source_hash=source_hash,
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
                item.source_hash,
                item.capability_kind,
                item.capability_id,
            )
            for item in requirements
        }
        for key in persisted:
            if key in keys:
                continue
            previous = typing.cast(
                "_WorkspaceExtensionRequirement",
                merged_persisted_requirement(key),
            )
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
                item.source_hash,
                item.capability_kind,
                item.capability_id,
            )
            for item in requirements
        )
        for record in self.catalog.model.extensions.values():
            if record.embed_policy != "always":
                continue
            source = record.source
            metadata = {"extension_name": record.name}
            for capability_kind, descriptors in (
                ("routine", source.routines),
                ("loader", source.loaders),
            ):
                for descriptor in descriptors:
                    key = (
                        record.id,
                        source.source_hash,
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
                            source_hash=source.source_hash,
                            extension_api_version=EXTENSION_API_VERSION,
                            metadata_snapshot=(
                                metadata
                                if source.source_modified_at is None
                                else {
                                    **metadata,
                                    "source_modified_at": source.source_modified_at,
                                }
                            ),
                            embedded_object_id=self._embedded_script_object_id(
                                record=record,
                                source_hash=source.source_hash,
                                previous=None,
                            ),
                        )
                    )
        return tuple(requirements)

    def _embedded_script_object_id(
        self,
        *,
        record: _ExtensionRecord | None,
        source_hash: str,
        previous: _WorkspaceExtensionRequirement | None,
    ) -> str | None:
        """Name an embedded object only when its source can be preserved."""
        if previous is not None and previous.embedded_object_id is not None:
            return previous.embedded_object_id
        if record is None:
            return None
        if record.embed_policy == "never":
            return None
        try:
            self.source_bytes(record.id, source_hash)
        except (KeyError, OSError):
            return None
        return f"extension-{source_hash}"

    def source_bytes(self, extension_id: str, source_hash: str) -> bytes:
        """Return workspace or catalog bytes that match a source hash."""
        return self._verified_source(extension_id, source_hash)

    def _verified_source(
        self,
        extension_id: str,
        source_hash: str,
        *,
        include_embedded: bool = True,
        include_catalog: bool = True,
    ) -> bytes:
        """Select the first source whose bytes match the requested hash."""
        candidates: list[bytes] = []
        if include_embedded:
            embedded = self._workspace_embedded_sources.get((extension_id, source_hash))
            if embedded is not None:
                candidates.append(embedded)
        if include_catalog:
            with contextlib.suppress(KeyError, OSError):
                candidates.append(
                    self.catalog.store.recovery_source_path(
                        extension_id, source_hash
                    ).read_bytes()
                )
        for source in candidates:
            if hashlib.sha256(source).hexdigest() == source_hash:
                return source
        if candidates:
            raise _ExtensionSourceHashMismatchError(
                f"Available source does not match {extension_id}:{source_hash}"
            )
        raise FileNotFoundError(
            f"No source is available for {extension_id}:{source_hash}"
        )

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

    def _remap_workspace_extension(
        self, previous_id: str, source_hash: str, registered_id: str
    ) -> None:
        """Remap one workspace source after it is registered under a new ID."""

        def remap_spec(spec: typing.Any) -> typing.Any:
            candidate = spec
            for ref, operation in reversed(tuple(iter_operation_refs(spec))):
                if (
                    getattr(operation, "op", None) == "extension_routine"
                    and getattr(operation, "extension_id", None) == previous_id
                    and getattr(operation, "source_hash", None) == source_hash
                ):
                    candidate = candidate._replace_operation_ref(
                        ref,
                        (operation.model_copy(update={"extension_id": registered_id}),),
                    )

            script_inputs = []
            script_inputs_changed = False
            for script_input in candidate.script_inputs:
                nested = script_input.parsed_provenance_spec()
                if nested is None:
                    script_inputs.append(script_input)
                    continue
                remapped_nested = remap_spec(nested)
                if remapped_nested == nested:
                    script_inputs.append(script_input)
                    continue
                script_inputs_changed = True
                script_inputs.append(
                    script_input.model_copy(
                        update={
                            "provenance_spec": remapped_nested.model_dump(mode="json")
                        }
                    )
                )
            if script_inputs_changed:
                candidate = candidate.model_copy(
                    update={"script_inputs": tuple(script_inputs)}
                )

            load_source = candidate.file_load_source
            replay_call = None if load_source is None else load_source.replay_call
            if (
                replay_call is not None
                and replay_call.kind == "extension_loader"
                and replay_call.target == previous_id
                and replay_call.source_hash == source_hash
            ):
                candidate = candidate.model_copy(
                    update={
                        "file_load_source": load_source.model_copy(
                            update={
                                "replay_call": replay_call.model_copy(
                                    update={"target": registered_id}
                                )
                            }
                        )
                    }
                )
            return candidate

        for node in self._manager._tool_graph.nodes.values():
            spec = node.passive_displayed_provenance_spec
            if spec is None:
                continue
            remapped = remap_spec(spec)
            if remapped != spec:
                node.set_displayed_provenance(remapped, advance_snapshot=False)

        self._workspace_requirements = tuple(
            requirement.model_copy(update={"extension_id": registered_id})
            if requirement.extension_id == previous_id
            and requirement.source_hash == source_hash
            else requirement
            for requirement in self._workspace_requirements
        )
        embedded = self._workspace_embedded_sources.pop(
            (previous_id, source_hash), None
        )
        if embedded is not None:
            self._workspace_embedded_sources[(registered_id, source_hash)] = embedded

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
        record = self.catalog.model.extensions.get(requirement.extension_id)
        if record is None:
            try:
                self._verified_source(
                    requirement.extension_id,
                    requirement.source_hash,
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
                requirement=requirement,
                state="missing",
                detail="Save and register the script included with this workspace",
            )
        source = (
            record.source
            if record.source.source_hash == requirement.source_hash
            else None
        )
        if source is None:
            try:
                self._verified_source(
                    requirement.extension_id,
                    requirement.source_hash,
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
                    requirement=requirement,
                    state="missing",
                    detail=(
                        "Save and register the script included with this workspace"
                    ),
                )
            return _ResolvedWorkspaceRequirement(
                requirement=requirement,
                state="missing",
                detail="The required source is not registered",
            )
        try:
            self.catalog.store.executable_source_path(
                record.id, requirement.source_hash
            )
        except _ExtensionCatalogConflictError:
            return _ResolvedWorkspaceRequirement(
                requirement=requirement,
                state="hash-mismatch",
                detail="The registered script file changed",
            )
        except (FileNotFoundError, KeyError):
            return _ResolvedWorkspaceRequirement(
                requirement=requirement,
                state="missing",
                detail="The registered script file is unavailable",
            )
        if source.import_error:
            return _ResolvedWorkspaceRequirement(
                requirement=requirement,
                state="import-failed",
                detail=source.import_error,
            )
        if not source.approved:
            return _ResolvedWorkspaceRequirement(
                requirement=requirement, state="approval-required"
            )
        capabilities = (
            source.routines
            if requirement.capability_kind == "routine"
            else source.loaders
        )
        if all(item.id != requirement.capability_id for item in capabilities):
            return _ResolvedWorkspaceRequirement(
                requirement=requirement,
                state="missing",
                detail="The registered source does not provide the required capability",
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
            recoverable={
                (item.extension_id, item.source_hash)
                for item in self._workspace_requirements
                if (item.extension_id, item.source_hash)
                in self._workspace_embedded_sources
            },
        )

        def register_slot(extension_id: str, source_hash: str) -> None:
            self._save_and_register_embedded_script(extension_id, source_hash)
            if erlab.interactive.utils.qt_is_valid(dialog):
                dialog.set_requirements(self.resolved_workspace_requirements())

        dialog.register_requested.connect(register_slot)
        try:
            dialog.exec()
        finally:
            with contextlib.suppress(TypeError, RuntimeError):
                dialog.register_requested.disconnect(register_slot)

    @QtCore.Slot(str, str)
    def _save_and_register_embedded_script(
        self, extension_id: str, source_hash: str
    ) -> bool:
        """Save workspace source as a user file before registration and import."""
        source = self._workspace_embedded_sources.get((extension_id, source_hash))
        if source is None:
            QtWidgets.QMessageBox.warning(
                self._manager,
                "Embedded Source Unavailable",
                "The selected requirement has no readable embedded script.",
            )
            return False
        requirement = next(
            (
                item
                for item in self._workspace_requirements
                if item.extension_id == extension_id and item.source_hash == source_hash
            ),
            None,
        )
        if requirement is None:
            return False
        try:
            source_text = source.decode("utf-8")
        except UnicodeDecodeError:
            QtWidgets.QMessageBox.warning(
                self._manager,
                "Embedded Source Unavailable",
                "The embedded script is not valid UTF-8 source.",
            )
            return False
        dialog = _SourceReviewDialog(
            None,
            self._manager,
            source_text=source_text,
        )
        if not dialog.exec():
            return False
        extension_name = str(
            requirement.metadata_snapshot.get("extension_name", extension_id)
        )
        suggested_name = (
            extension_name
            if extension_name.lower().endswith(".py")
            else f"{extension_id}.py"
        )
        existing = self.catalog.model.extensions.get(extension_id)
        if existing is not None and existing.source.source_hash != source_hash:
            modified = requirement.metadata_snapshot.get("source_modified_at")
            try:
                date = (
                    QtCore.QDateTime.fromString(
                        str(modified), QtCore.Qt.DateFormat.ISODate
                    )
                    .date()
                    .toString("yyyy-MM-dd")
                )
            except (TypeError, ValueError):
                date = ""
            if not date:
                date = QtCore.QDate.currentDate().toString("yyyy-MM-dd")
            path_name = pathlib.Path(suggested_name)
            suggested_name = f"{path_name.stem}_workspace_{date}.py"
        destination = self._save_source_as_user_file(
            source,
            title="Save Workspace Extension Script",
            suggested_name=suggested_name,
        )
        if destination is None:
            return False
        try:
            registration_id = extension_id
            if existing is not None and existing.source.source_hash != source_hash:
                registration_id = _safe_extension_id(
                    f"{destination.stem}-{uuid.uuid4().hex[:8]}"
                )
            self.catalog.store.add_script(
                destination,
                extension_id=registration_id,
                expected_source_hash=source_hash,
                expected_record_generation=(
                    None
                    if registration_id != extension_id or existing is None
                    else existing.record_generation
                ),
                check_record_generation=True,
            )
            self.catalog.refresh()
            record = self.catalog.model.extensions[registration_id]
            if not record.source.approved:
                self.execution.validate_source(
                    registration_id,
                    source_hash,
                    expected_record_generation=record.record_generation,
                    enable_extension=True,
                )
                self.catalog.refresh()
            if registration_id != extension_id:
                self._remap_workspace_extension(
                    extension_id, source_hash, registration_id
                )
            return True  # noqa: TRY300 - success exits before the shared error path.
        except Exception:
            erlab.interactive.utils.MessageDialog.critical(
                self._manager,
                "Extension Error",
                "The saved workspace extension could not be registered.",
                detailed_text=traceback.format_exc(),
            )
            self.catalog.refresh()
            return False

    def notify_unavailable_workspace_requirements(self) -> None:
        unavailable = [
            item
            for item in self.resolved_workspace_requirements()
            if item.state != "ready"
        ]
        if not unavailable:
            return
        missing_registered_scripts = {
            item.requirement.extension_id
            for item in unavailable
            if item.state == "missing"
            and (
                record := self.catalog.model.extensions.get(
                    item.requirement.extension_id
                )
            )
            is not None
            and record.source.source_hash == item.requirement.source_hash
        }
        if missing_registered_scripts and self._show_missing_script_recovery(
            extension_ids=missing_registered_scripts,
            repeat=True,
        ):
            return
        if any(
            (
                item.requirement.extension_id,
                item.requirement.source_hash,
            )
            in self._workspace_embedded_sources
            for item in unavailable
        ):
            self.show_workspace_requirements()
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

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        missing_dialog = self._missing_scripts_dialog
        if missing_dialog is not None and erlab.interactive.utils.qt_is_valid(
            missing_dialog
        ):
            missing_dialog.close()
        self._missing_scripts_dialog = None
        self._missing_scripts_dialog_slots = None
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
        ):
            if erlab.interactive.utils.qt_is_valid(action):
                with contextlib.suppress(TypeError, RuntimeError):
                    action.triggered.disconnect(slot)
        if erlab.interactive.utils.qt_is_valid(self._manage_dialog):
            for signal, slot in (
                (self._manage_dialog.action_requested, self._manage_action_slot),
                (
                    self._manage_dialog.add_script_requested,
                    self._manage_add_script_slot,
                ),
                (
                    self._manage_dialog.open_folder_requested,
                    self._manage_open_folder_slot,
                ),
                (
                    self._manage_dialog.selection_changed,
                    self._manage_selection_slot,
                ),
                (self._manage_dialog.activated, self._manage_activated_slot),
            ):
                with contextlib.suppress(TypeError, RuntimeError):
                    signal.disconnect(slot)
            self._manage_dialog.close()
            self._manage_dialog.deleteLater()
        with contextlib.suppress(TypeError, RuntimeError):
            self.execution.queue_changed.disconnect(self._execution_state_slot)
        with contextlib.suppress(TypeError, RuntimeError):
            self.catalog.changed.disconnect(self._catalog_changed_slot)
        self._routine_action_groups.clear()
        self.catalog.close()
