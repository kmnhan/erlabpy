"""Qt controller for ImageTool Manager extensions."""

from __future__ import annotations

import contextlib
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
from erlab.extensions._models import _script_name_key
from erlab.interactive.imagetool._load_source import _deserialize_loader_kwargs
from erlab.interactive.imagetool._provenance._model import iter_operation_refs
from erlab.interactive.imagetool.manager._extensions._catalog import (
    _ExtensionCatalog,
    _ExtensionCatalogConflictError,
    _ExtensionCatalogError,
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
    _ExtensionCatalogModel,
    _ResolvedWorkspaceRequirement,
    _ScriptRecord,
    _WorkspaceScriptRequirement,
)
from erlab.interactive.imagetool.manager._registry import (
    ImageToolManagerRegistryError,
    live_manager_records,
)

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


class _ExtensionController(QtCore.QObject):
    """Own extension UI, script approval, and execution for one manager.

    Script registrations are application-wide. Validation failures and execution
    queues stay local to this manager.
    """

    def __init__(self, manager: ImageToolManager) -> None:
        super().__init__(manager)
        self._manager = manager
        self.catalog = _ExtensionCatalog(parent=self)
        self.execution = _ExtensionExecutionController(manager, self.catalog)
        self._catalog_changed_slot = self._catalog_changed
        self._catalog_read_failed_slot = self._show_catalog_read_error
        self._manage_action_slot = self._manage_action
        self.catalog.changed.connect(self._catalog_changed_slot)
        self.catalog.read_failed.connect(self._catalog_read_failed_slot)
        self._recent: deque[tuple[str, str]] = deque(maxlen=8)
        self._routine_action_groups: list[
            tuple[QtWidgets.QMenu, list[QtGui.QAction]]
        ] = []
        self._explorer_loaders: dict[str, erlab.io.dataloader.LoaderBase] = {}
        self._closed = False
        self._shown_catalog_errors: set[str] = set()
        self._missing_script_prompt_shown = False
        self._missing_scripts_dialog: _MissingScriptsDialog | None = None
        self._missing_scripts_dialog_slots: (
            tuple[Callable[[str], None], Callable[[int], None]] | None
        ) = None
        self._manage_dialog = _ManageExtensionsDialog(manager)
        self._manage_dialog.action_requested.connect(self._manage_action_slot)
        self._manage_add_script_slot = self.add_script
        self._manage_selection_slot = self._refresh_removal_eligibility
        self._manage_activated_slot = self._refresh_removal_eligibility
        self._execution_state_slot = self._refresh_removal_eligibility
        self._validation_changed_slot = self._session_validation_changed
        self._manage_dialog.add_script_requested.connect(self._manage_add_script_slot)
        self._manage_dialog.selection_changed.connect(self._manage_selection_slot)
        self._manage_dialog.activated.connect(self._manage_activated_slot)
        self.execution.queue_changed.connect(self._execution_state_slot)
        self.execution.validation_changed.connect(self._validation_changed_slot)
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
        self._initial_catalog_error_slot = self._show_initial_catalog_error
        QtCore.QTimer.singleShot(0, self._initial_catalog_error_slot)
        self._missing_script_prompt_slot = self._show_missing_script_recovery
        QtCore.QTimer.singleShot(0, self._missing_script_prompt_slot)

    @QtCore.Slot()
    def _show_initial_catalog_error(self) -> None:
        """Report startup failure only after the manager can own the dialog."""
        error = self.catalog.load_error
        if error is not None:
            self._show_catalog_read_error(error)

    @QtCore.Slot(str)
    def _show_catalog_read_error(self, error: str) -> None:
        """Show each distinct catalog read failure once during this manager session."""
        if self._closed or not erlab.interactive.utils.qt_is_valid(self._manager):
            return
        self._refresh_extension_state_views()
        if error in self._shown_catalog_errors:
            return
        self._shown_catalog_errors.add(error)
        erlab.interactive.utils.MessageDialog.critical(
            self._manager,
            "Extension Catalog Unavailable",
            "ImageTool Manager could not read the extension catalog. The existing "
            "catalog was not changed.",
            detailed_text=f"Catalog: {self.catalog.store.path}\n\n{error}",
        )

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

    def _enabled_routines(self) -> tuple[tuple[str, RoutineDescriptor], ...]:
        if self.catalog.load_error is not None:
            return ()
        entries: list[tuple[str, RoutineDescriptor]] = []
        for record in self.catalog.model.extensions.values():
            routines = self.execution.ready_routines(
                record.script_name, record.source_hash
            )
            entries.extend((record.script_name, item) for item in routines)
        return tuple(entries)

    def file_loaders(
        self,
        paths: str | os.PathLike[str] | Iterable[str | os.PathLike[str]] | None = None,
    ) -> dict[str, tuple[Callable[..., typing.Any], dict[str, typing.Any]]]:
        """Return enabled decorated loaders in the standard file-dialog shape."""
        if self.catalog.load_error is not None:
            return {}
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
            for call in self.execution.ready_loader_calls(
                record.script_name, record.source_hash
            ):
                descriptor = call.descriptor
                patterns = tuple(f"*{suffix}" for suffix in descriptor.extensions) or (
                    "*",
                )
                name_filter = f"{descriptor.name} ({' '.join(patterns)})"
                if path_values and not all(
                    not descriptor.extensions
                    or path.suffix.casefold() in descriptor.extensions
                    for path in path_values
                ):
                    continue
                previous_owner = owners.get(name_filter)
                if previous_owner is not None:
                    raise ValueError(
                        f"Conflicting extension file dialog filter {name_filter!r} "
                        f"provided by {previous_owner!r} and {record.script_name!r}"
                    )
                entries[name_filter] = (
                    call,
                    {
                        parameter.id: parameter.default
                        for parameter in descriptor.parameters
                        if not parameter.required
                    },
                )
                owners[name_filter] = record.script_name
        return entries

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
        if self.catalog.load_error is not None:
            raise erlab.extensions.ExtensionExecutionError(
                "The extension catalog is unavailable"
            )
        call = self.execution.loader_call(
            replay_call.target,
            replay_call.source_hash,
            replay_call.capability_id,
        )
        return self.execution.run_loader(
            call,
            pathlib.Path(load_source.path),
            _deserialize_loader_kwargs(replay_call.kwargs),
        )

    def capability_status(
        self,
        script_name: str,
        source_hash: str,
        kind: typing.Literal["routine", "loader"],
        capability_id: str,
    ) -> _CapabilityStatus:
        """Resolve application catalog state for this manager."""
        if self.catalog.load_error is not None:
            return "missing-source"
        return self.execution.capability_status(
            script_name,
            source_hash,
            kind,
            capability_id,
        )

    def routine_descriptor(
        self,
        script_name: str,
        source_hash: str,
        routine_id: str,
    ) -> RoutineDescriptor | None:
        """Return a routine descriptor from the exact registered script source."""
        try:
            snapshot = self.catalog.store.resolve_script(script_name, source_hash)
        except (FileNotFoundError, KeyError, _ExtensionCatalogError):
            return None
        return next(
            (item for item in snapshot.record.routines if item.id == routine_id),
            None,
        )

    def loader_descriptor(
        self,
        script_name: str,
        source_hash: str,
        loader_id: str,
    ) -> LoaderDescriptor | None:
        """Return a loader descriptor from the exact registered script source."""
        try:
            snapshot = self.catalog.store.resolve_script(script_name, source_hash)
        except (FileNotFoundError, KeyError, _ExtensionCatalogError):
            return None
        return next(
            (item for item in snapshot.record.loaders if item.id == loader_id),
            None,
        )

    @property
    def explorer_loaders(self) -> dict[str, erlab.io.dataloader.LoaderBase]:
        """Manager-local loader adapters used by existing Data Explorer tabs."""
        return self._explorer_loaders

    def _sync_explorer_loaders(self) -> None:
        updated: dict[str, erlab.io.dataloader.LoaderBase] = {}
        if self.catalog.load_error is not None:
            self._explorer_loaders.clear()
            return
        for record in self.catalog.model.extensions.values():
            for call in self.execution.ready_loader_calls(
                record.script_name, record.source_hash
            ):
                adapter = _DecoratedLoaderAdapter(call)
                updated[adapter.name] = adapter
        self._explorer_loaders.clear()
        self._explorer_loaders.update(updated)

    def loader_by_name(
        self, name: str
    ) -> tuple[Callable[..., typing.Any], dict[str, typing.Any]] | None:
        adapter = self._explorer_loaders.get(name)
        if not isinstance(adapter, _DecoratedLoaderAdapter):
            return None
        return adapter.load_for_manager, {}

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
            entry
            for entry in routines
            if (_script_name_key(entry[0]), entry[1].id) in favorite_keys
        )
        if favorite_entries:
            favorites = typing.cast("QtWidgets.QMenu", menu.addMenu("Favorites"))
            favorites_action = favorites.menuAction()
            if favorites_action is not None:
                favorites_action.setProperty("requiresImageTool", True)
                retained.append(favorites_action)
            for entry in favorite_entries:
                self._add_routine_action(favorites, entry, retained)
        routines_by_key = {
            (_script_name_key(entry[0]), entry[1].id): entry for entry in routines
        }
        recent_keys = tuple(
            (_script_name_key(script_name), routine_id)
            for script_name, routine_id in self._recent
            if (_script_name_key(script_name), routine_id) in routines_by_key
        )
        if recent_keys:
            recent_menu = typing.cast("QtWidgets.QMenu", menu.addMenu("Recent"))
            recent_action = recent_menu.menuAction()
            if recent_action is not None:
                recent_action.setProperty("requiresImageTool", True)
                retained.append(recent_action)
            for key in recent_keys:
                self._add_routine_action(recent_menu, routines_by_key[key], retained)
        categories: dict[str, list[tuple[str, RoutineDescriptor]]] = defaultdict(list)
        for entry in routines:
            categories[entry[1].category].append(entry)
        for category in sorted(categories):
            category_menu = typing.cast("QtWidgets.QMenu", menu.addMenu(category))
            category_action = category_menu.menuAction()
            if category_action is not None:
                category_action.setProperty("requiresImageTool", True)
                retained.append(category_action)
            for entry in sorted(categories[category], key=lambda item: item[1].name):
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
        entry: tuple[str, RoutineDescriptor],
        retained: list[QtGui.QAction],
    ) -> None:
        script_name, descriptor = entry
        action = typing.cast("QtGui.QAction", menu.addAction(descriptor.name))
        action.setData((script_name, descriptor.id))
        action.setProperty("requiresImageTool", True)
        action.setToolTip(descriptor.summary or script_name)
        action.triggered.connect(
            lambda _checked=False, script=script_name, routine=descriptor.id: (
                self.run_routine(script, routine)
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
            favorites=tuple(
                (script_name, descriptor.id)
                for script_name, descriptor in routines
                if (_script_name_key(script_name), descriptor.id)
                in self.catalog.model.routine_favorites
            ),
        )
        favorite_slot = self._set_routine_favorite
        dialog.favorite_requested.connect(favorite_slot)
        if dialog.exec() and dialog.selection is not None:
            self.run_routine(*dialog.selection)
        with contextlib.suppress(TypeError, RuntimeError):
            dialog.favorite_requested.disconnect(favorite_slot)

    @QtCore.Slot(str, str, bool)
    def _set_routine_favorite(
        self, script_name: str, routine_id: str, favorite: bool
    ) -> None:
        try:
            self.catalog.store.set_routine_favorite(
                script_name, routine_id, favorite=favorite
            )
            self.catalog.refresh()
        except Exception:
            erlab.interactive.utils.MessageDialog.critical(
                self._manager,
                "Extension Error",
                "The routine favorite could not be changed.",
                detailed_text=traceback.format_exc(),
            )

    def run_routine(self, script_name: str, routine_id: str) -> None:
        targets = self._manager._selected_imagetool_targets()
        if len(targets) != 1:
            QtWidgets.QMessageBox.information(
                self._manager,
                "Select Data",
                "Select one ImageTool before you run a routine.",
            )
            return
        try:
            snapshot = self.catalog.store.resolve_script(script_name)
        except (FileNotFoundError, KeyError, _ExtensionCatalogError):
            return
        record = snapshot.record
        if not record.enabled:
            return
        descriptor = next(
            (item for item in record.routines if item.id == routine_id),
            None,
        )
        if descriptor is None:
            return
        dialog = _ExtensionParameterDialog(descriptor, self._manager)
        if not dialog.exec():
            return
        try:
            self.execution.queue_routine(
                script_name=record.script_name,
                source_hash=record.source_hash,
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
        key = (record.script_name, routine_id)
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
            self._review_and_register(pathlib.Path(path))

    def _review_and_register(self, path: pathlib.Path) -> bool:
        """Review and register one user-owned local Python script."""
        try:
            resolved_path = path.expanduser().resolve()
            reviewed_source = resolved_path.read_bytes()
            reviewed_source_hash = hashlib.sha256(reviewed_source).hexdigest()
            source_text = reviewed_source.decode("utf-8")
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
            script_key = _script_name_key(resolved_path.name)
            existing = self.catalog.store.read().extensions.get(script_key)
            if existing is None:
                catalog, _source_hash = self.catalog.store.register_script(
                    resolved_path,
                    expected_source_hash=reviewed_source_hash,
                )
                changed = True
            elif pathlib.Path(existing.source_path) != resolved_path:
                QtWidgets.QMessageBox.warning(
                    self._manager,
                    "Script Already Registered",
                    f"{existing.script_name} is already registered at "
                    f"{existing.source_path}.",
                )
                return False
            else:
                catalog, changed = self.catalog.store.reload_script(
                    existing.script_name,
                    expected_source_hash=reviewed_source_hash,
                    expected_record_generation=existing.record_generation,
                )
            self.catalog.refresh()
            record = catalog.extensions[script_key]
            if not changed and record.enabled and record.approved:
                self._manager._status_bar.showMessage(
                    "The registered script contents did not change.",
                    4000,
                )
                return True
            self.execution.validate_script(
                record.script_name,
                record.source_hash,
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
        self, script_names: Collection[str] | None = None
    ) -> tuple[_ScriptRecord, ...]:
        """Return enabled scripts whose registered current file cannot be read."""
        requested_keys = (
            None
            if script_names is None
            else {_script_name_key(value) for value in script_names}
        )
        missing: list[_ScriptRecord] = []
        for record in self.catalog.model.extensions.values():
            if not record.enabled or (
                requested_keys is not None
                and _script_name_key(record.script_name) not in requested_keys
            ):
                continue
            try:
                self.catalog.store.resolve_script(
                    record.script_name, record.source_hash
                )
            except _ExtensionCatalogConflictError:
                continue
            except (FileNotFoundError, KeyError, _ExtensionCatalogError):
                missing.append(record)
        return tuple(missing)

    def _show_missing_script_recovery(
        self,
        *,
        script_names: Collection[str] | None = None,
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
        records = self._missing_script_records(script_names)
        if not records:
            return False
        self._missing_script_prompt_shown = True
        dialog = _MissingScriptsDialog(records, self._manager)

        def refresh_dialog() -> None:
            if not erlab.interactive.utils.qt_is_valid(dialog):
                return
            remaining = self._missing_script_records(script_names)
            if not remaining:
                dialog.accept()
            else:
                dialog.set_records(remaining)

        def locate_slot(script_name: str) -> None:
            self._locate_missing_script(script_name)
            refresh_dialog()

        def finished_slot(_result: int) -> None:
            if self._missing_scripts_dialog is not dialog:
                return
            with contextlib.suppress(TypeError, RuntimeError):
                dialog.locate_requested.disconnect(locate_slot)
            with contextlib.suppress(TypeError, RuntimeError):
                dialog.finished.disconnect(finished_slot)
            self._missing_scripts_dialog = None
            self._missing_scripts_dialog_slots = None

        dialog.locate_requested.connect(locate_slot)
        dialog.finished.connect(finished_slot)
        self._missing_scripts_dialog = dialog
        self._missing_scripts_dialog_slots = (
            locate_slot,
            finished_slot,
        )
        dialog.show()
        dialog.raise_()
        return True

    def _locate_missing_script(self, script_name: str) -> bool:
        record = self.catalog.model.extensions.get(_script_name_key(script_name))
        if record is None:
            return False
        path_value, _selected_filter = QtWidgets.QFileDialog.getOpenFileName(
            self._manager,
            f"Locate {record.script_name}",
            record.source_path,
            "Python scripts (*.py)",
        )
        if not path_value:
            return False
        path = pathlib.Path(path_value).expanduser().resolve()
        try:
            self.catalog.store.relocate_script(
                record.script_name,
                path,
                expected_record_generation=record.record_generation,
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
        safe_name = pathlib.Path(suggested_name).name
        suggested_path = os.fspath(pathlib.Path(initial_directory) / safe_name)
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

    @QtCore.Slot()
    def show_manager(self) -> None:
        self._refresh_manage_dialog()
        self._manage_dialog.show()
        self._manage_dialog.raise_()

    def _refresh_manage_dialog(self) -> None:
        if self.catalog.load_error is not None:
            self._manage_dialog.set_catalog(_ExtensionCatalogModel())
            self._refresh_removal_eligibility()
            return
        source_states: dict[tuple[str, str], str] = {}
        for record in self.catalog.model.extensions.values():
            key = (record.script_name, record.source_hash)
            try:
                self.catalog.store.resolve_script(
                    record.script_name, record.source_hash
                )
            except _ExtensionCatalogConflictError:
                source_states[key] = "Source file changed"
            except FileNotFoundError:
                source_states[key] = "Source file missing"
            except (KeyError, _ExtensionCatalogError):
                source_states[key] = "Source file unreadable"
            else:
                source_states[key] = "Ready"
        self._manage_dialog.set_catalog(
            self.catalog.model,
            source_states,
            validation_errors={
                (record.script_name, record.source_hash): error
                for record in self.catalog.model.extensions.values()
                if (
                    error := self.execution.validation_error(
                        record.script_name, record.source_hash
                    )
                )
                is not None
            },
        )
        self._refresh_removal_eligibility()

    @QtCore.Slot(str, str)
    def _manage_action(self, action_id: str, script_name: str) -> None:
        record = self.catalog.model.extensions.get(_script_name_key(script_name))
        if record is None:
            return
        try:
            if action_id == "reload":
                source_path = pathlib.Path(record.source_path)
                if not source_path.is_file():
                    self._locate_missing_script(record.script_name)
                    return
                self._review_and_register(source_path)
                return
            validation_error = self.execution.validation_error(
                record.script_name, record.source_hash
            )
            if action_id == "toggle" and validation_error is not None:
                if record.approved:
                    self.execution.validate_script(
                        record.script_name,
                        record.source_hash,
                        expected_record_generation=record.record_generation,
                        enable_script=False,
                        persist_result=False,
                    )
                else:
                    self.execution.validate_script(
                        record.script_name,
                        record.source_hash,
                        expected_record_generation=record.record_generation,
                    )
            elif action_id == "toggle" and not record.enabled:
                self.execution.validate_script(
                    record.script_name,
                    record.source_hash,
                    expected_record_generation=record.record_generation,
                )
            elif action_id == "toggle":
                self.catalog.store.update_script(
                    record.script_name,
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
                self.catalog.store.update_script(
                    record.script_name,
                    expected_record_generation=record.record_generation,
                    embed_policy=typing.cast(
                        'typing.Literal["referenced", "always", "never"]', policy
                    ),
                )
            elif action_id == "error":
                validation_error = self.execution.validation_error(
                    record.script_name, record.source_hash
                )
                if validation_error:
                    erlab.interactive.utils.MessageDialog.critical(
                        self._manager,
                        "Extension Validation Error",
                        "The extension could not be validated.",
                        detailed_text=validation_error,
                    )
                return
            elif action_id == "view_source":
                self._show_source(record.script_name, record.source_hash)
                return
            elif action_id in {"open_source", "reveal_source", "copy_source"}:
                source_path = record.source_path
                if not pathlib.Path(source_path).is_file():
                    QtWidgets.QMessageBox.information(
                        self._manager,
                        "Source File Unavailable",
                        "The registered source file is unavailable.",
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

    def _show_source(self, script_name: str, source_hash: str) -> None:
        try:
            snapshot = self.catalog.store.resolve_script(script_name, source_hash)
            source_text = snapshot.source_bytes.decode("utf-8")
        except (
            FileNotFoundError,
            KeyError,
            UnicodeError,
            _ExtensionCatalogError,
        ):
            erlab.interactive.utils.MessageDialog.critical(
                self._manager,
                "Extension Error",
                "The registered extension source could not be read.",
                detailed_text=traceback.format_exc(),
            )
            return
        dialog = _SourceViewerDialog(
            source_text,
            self._manager,
            title=f"Extension Source — {script_name}",
        )
        dialog.exec()

    @QtCore.Slot()
    @QtCore.Slot(str)
    def _refresh_removal_eligibility(self, _script_name: str = "") -> None:
        script_name = self._manage_dialog.selected_script_name
        reason = None if script_name is None else self._removal_blocker(script_name)
        self._manage_dialog.set_removal_reason(reason)

    def _removal_blocker(self, script_name: str) -> str | None:
        record = self.catalog.model.extensions.get(_script_name_key(script_name))
        if record is None:
            return None
        try:
            other_managers = tuple(
                manager
                for manager in live_manager_records(strict=True, include_starting=True)
                if manager.internal_id != self._manager._manager_record.internal_id
            )
        except ImageToolManagerRegistryError:
            return "Removal is unavailable because other managers could not be checked."
        if other_managers:
            descriptions = "; ".join(
                f"Manager {manager.index}"
                + (f" ({manager.workspace_path})" if manager.workspace_path else "")
                for manager in other_managers
            )
            return f"Close the other ImageTool Managers first: {descriptions}."
        if self.execution.uses_script(record.script_name):
            return "Wait for this script's active or queued jobs to finish."
        requirements = tuple(
            requirement
            for requirement in self.collect_workspace_requirements()
            if _script_name_key(requirement.script_name)
            == _script_name_key(record.script_name)
        )
        if requirements:
            workspace_path = self._manager._manager_record.workspace_path
            location = (
                "the current workspace" if workspace_path is None else workspace_path
            )
            return f"Remove this script from {location} before you delete it."
        if self._manager._workspace_state.extension_scripts.has_opaque_content:
            return (
                "This script can be referenced by workspace data that this ERLab "
                "version cannot inspect."
            )
        return None

    def _remove_extension(self, record: _ScriptRecord) -> None:
        blocker = self._removal_blocker(record.script_name)
        if blocker is not None:
            QtWidgets.QMessageBox.information(
                self._manager, "Extension Cannot Be Removed", blocker
            )
            self._refresh_removal_eligibility()
            return
        dialog = QtWidgets.QMessageBox(self._manager)
        dialog.setObjectName("manager_extension_remove_confirmation")
        dialog.setWindowTitle("Remove Extension")
        dialog.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        dialog.setText(f"Remove {record.script_name} from ERLab?")
        dialog.setInformativeText(
            "ERLab will remove this registration. The local Python file will not be "
            "deleted."
        )
        dialog.setDetailedText(
            f"Registered source: {record.source_path}\n\n"
            "Closed workspaces without an embedded copy can lose replay capability."
        )
        dialog.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.Cancel
        )
        dialog.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        if dialog.exec() != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        blocker = self._removal_blocker(record.script_name)
        if blocker is not None:
            QtWidgets.QMessageBox.information(
                self._manager, "Extension Cannot Be Removed", blocker
            )
            self._refresh_removal_eligibility()
            return
        self.catalog.store.remove_script(
            record.script_name,
            expected_record_generation=record.record_generation,
        )
        self.catalog.refresh()

    @QtCore.Slot(object)
    def _catalog_changed(self, _model: object) -> None:
        self.execution.prune_validation_errors(self.catalog.model)
        self._reconcile_persisted_workspace_requirements()
        self._refresh_extension_state_views()

    def _refresh_extension_state_views(self) -> None:
        """Apply global or manager-local extension state to all manager views."""
        self._sync_explorer_loaders()
        self._refresh_manage_dialog()
        if self.menu is not None and self.menu.isVisible():
            self._populate_menu()
        explorer = self._manager._standalone_app_windows.get("explorer")
        if explorer is not None and erlab.interactive.utils.qt_is_valid(explorer):
            typing.cast("_TabbedExplorer", explorer).refresh_loader_choices()
        self._manager._update_actions()
        self._manager._update_info()
        for node in self._manager._tool_graph.nodes.values():
            if node.tool_window is not None:
                node.tool_window._refresh_reload_data_action()

    @QtCore.Slot()
    def _session_validation_changed(self) -> None:
        """Refresh manager state after this session observes source health."""
        if self._closed or not erlab.interactive.utils.qt_is_valid(self._manager):
            return
        self._refresh_extension_state_views()

    def collect_workspace_requirements(
        self,
    ) -> tuple[_WorkspaceScriptRequirement, ...]:
        """Rebuild loaded-node dependencies and retain unresolved references.

        Current provenance is authoritative for nodes in the graph. References to
        nodes that did not load and unresolved requirements without node references
        remain available for a degraded Save As. Resolved dependencies that are not
        in the current graph are omitted without mutating document state.
        """
        loaded_node_uids = set(self._manager._tool_graph.nodes)
        persisted: dict[
            tuple[str, str, str, str], list[_WorkspaceScriptRequirement]
        ] = defaultdict(list)
        for item in self._manager._workspace_state.extension_scripts.requirements:
            persisted[
                (
                    _script_name_key(item.script_name),
                    item.source_hash,
                    item.capability_kind,
                    item.capability_id,
                )
            ].append(item)

        def merged_persisted_requirement(
            key: tuple[str, str, str, str],
        ) -> _WorkspaceScriptRequirement | None:
            """Merge per-node state for one immutable source dependency."""
            items = persisted.get(key)
            if not items:
                return None
            primary = items[0]
            referencing_nodes: set[str] = set()
            file_sources: set[str] = set()
            for item in items:
                referencing_nodes.update(item.referencing_nodes)
                file_sources.update(item.file_sources)
            return primary.model_copy(
                update={
                    "referencing_nodes": tuple(sorted(referencing_nodes)),
                    "file_sources": tuple(sorted(file_sources)),
                }
            )

        references: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        operations: dict[tuple[str, str, str], typing.Any] = {}
        loader_references: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        loader_files: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        loader_script_names: dict[tuple[str, str, str], str] = {}
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
                        _script_name_key(extension_operation.script_name),
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
                    _script_name_key(replay_call.target),
                    replay_call.source_hash,
                    replay_call.capability_id,
                )
                loader_references[key].add(uid)
                loader_files[key].add(current_spec.file_load_source.path)
                loader_script_names[key] = replay_call.target
        requirements: list[_WorkspaceScriptRequirement] = []
        for key, node_uids in references.items():
            _script_key, source_hash, routine_id = key
            operation = typing.cast("ExtensionRoutineOperation", operations[key])
            script_name = operation.script_name
            previous = merged_persisted_requirement(
                (
                    _script_name_key(script_name),
                    source_hash,
                    "routine",
                    routine_id,
                )
            )
            if previous is not None:
                node_uids.update(
                    set(previous.referencing_nodes).difference(loaded_node_uids)
                )
            requirements.append(
                _WorkspaceScriptRequirement(
                    script_name=script_name,
                    capability_id=routine_id,
                    capability_name=operation.routine_name,
                    capability_kind="routine",
                    source_hash=source_hash,
                    extension_api_version=(
                        EXTENSION_API_VERSION
                        if previous is None
                        else previous.extension_api_version
                    ),
                    referencing_nodes=tuple(sorted(node_uids)),
                )
            )
        for key, node_uids in loader_references.items():
            _script_key, source_hash, loader_id = key
            script_name = loader_script_names[key]
            previous = merged_persisted_requirement(
                (
                    _script_name_key(script_name),
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
            descriptor = self.loader_descriptor(script_name, source_hash, loader_id)
            requirements.append(
                _WorkspaceScriptRequirement(
                    script_name=script_name,
                    capability_id=loader_id,
                    capability_name=(
                        descriptor.name
                        if descriptor is not None
                        else previous.capability_name
                        if previous is not None
                        else loader_id
                    ),
                    capability_kind="loader",
                    source_hash=source_hash,
                    extension_api_version=(
                        EXTENSION_API_VERSION
                        if previous is None
                        else previous.extension_api_version
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
                _script_name_key(item.script_name),
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
                "_WorkspaceScriptRequirement",
                merged_persisted_requirement(key),
            )
            remaining_node_uids = tuple(
                uid for uid in previous.referencing_nodes if uid not in loaded_node_uids
            )
            if remaining_node_uids or (
                not previous.referencing_nodes
                and self._resolve_requirement(previous).state != "ready"
            ):
                requirements.append(
                    previous.model_copy(
                        update={"referencing_nodes": remaining_node_uids}
                    )
                )
        return tuple(requirements)

    def collect_workspace_embedded_sources(
        self, requirements: Iterable[_WorkspaceScriptRequirement]
    ) -> tuple[tuple[str, str, bytes], ...]:
        """Return exact normal sources that the current workspace must embed."""
        state = self._manager._workspace_state.extension_scripts
        try:
            records = self.catalog.store.read().extensions
        except _ExtensionCatalogError:
            records = {}
        wanted = set(state.explicit_sources)
        for requirement in requirements:
            record = records.get(_script_name_key(requirement.script_name))
            if (
                record is not None
                and record.source_hash == requirement.source_hash
                and record.embed_policy == "never"
            ):
                try:
                    self.catalog.store.resolve_script(
                        requirement.script_name, requirement.source_hash
                    )
                except (FileNotFoundError, KeyError, _ExtensionCatalogError):
                    pass
                else:
                    continue
            wanted.add((requirement.script_name, requirement.source_hash))
        wanted.update(
            (record.script_name, record.source_hash)
            for record in records.values()
            if record.embed_policy == "always"
        )
        verified_sources = {
            (_script_name_key(script_name), source_hash): source
            for (script_name, source_hash), (_entry, source) in (
                state.verified_sources.items()
            )
        }
        sources: list[tuple[str, str, bytes]] = []
        for script_name, source_hash in sorted(
            wanted, key=lambda item: (_script_name_key(item[0]), item[1])
        ):
            source = verified_sources.get((_script_name_key(script_name), source_hash))
            if source is None:
                try:
                    source = self.catalog.store.resolve_script(
                        script_name, source_hash
                    ).source_bytes
                except (FileNotFoundError, KeyError, _ExtensionCatalogError):
                    continue
            sources.append((script_name, source_hash, source))
        return tuple(sources)

    def _reconcile_persisted_workspace_requirements(self) -> None:
        """Retire resolved dependencies that have no document references."""
        state = self._manager._workspace_state.extension_scripts
        state.requirements = tuple(
            requirement
            for requirement in state.requirements
            if requirement.referencing_nodes
            or self._resolve_requirement(requirement).state != "ready"
        )
        state.explicit_sources = {
            key
            for key in state.explicit_sources
            if (record := self.catalog.model.extensions.get(_script_name_key(key[0])))
            is None
            or record.source_hash != key[1]
            or record.embed_policy != "always"
        }

    def _canonicalize_workspace_script_name(
        self, script_name: str, source_hash: str
    ) -> None:
        """Use one registered filename spelling across current document owners."""
        script_key = _script_name_key(script_name)
        requirements = self.collect_workspace_requirements()
        state = self._manager._workspace_state.extension_scripts
        source_hashes = {source_hash}
        source_hashes.update(
            requirement.source_hash
            for requirement in requirements
            if _script_name_key(requirement.script_name) == script_key
            and requirement.script_name != script_name
        )
        source_hashes.update(
            source_hash
            for stored_name, source_hash in state.verified_sources
            if _script_name_key(stored_name) == script_key
            and stored_name != script_name
        )
        for source_hash in sorted(source_hashes):
            self._remap_workspace_script(script_name, source_hash, script_name)

    def _remap_workspace_script(
        self, previous_name: str, source_hash: str, registered_name: str
    ) -> None:
        """Remap one workspace source after it is saved with a new filename."""
        previous_key = _script_name_key(previous_name)

        def remap_spec(spec: typing.Any) -> typing.Any:
            candidate = spec
            for ref, operation in reversed(tuple(iter_operation_refs(spec))):
                if (
                    getattr(operation, "op", None) == "extension_routine"
                    and isinstance(
                        operation_script_name := getattr(
                            operation, "script_name", None
                        ),
                        str,
                    )
                    and _script_name_key(operation_script_name) == previous_key
                    and getattr(operation, "source_hash", None) == source_hash
                ):
                    candidate = candidate._replace_operation_ref(
                        ref,
                        (
                            operation.model_copy(
                                update={"script_name": registered_name}
                            ),
                        ),
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
                and _script_name_key(replay_call.target) == previous_key
                and replay_call.source_hash == source_hash
            ):
                candidate = candidate.model_copy(
                    update={
                        "file_load_source": load_source.model_copy(
                            update={
                                "replay_call": replay_call.model_copy(
                                    update={"target": registered_name}
                                )
                            }
                        )
                    }
                )
            return candidate

        state = self._manager._workspace_state.extension_scripts
        original_state = state.copy()
        remapped_state = state.copy()
        remapped_state.remap_script(previous_name, source_hash, registered_name)
        state_changed = (
            remapped_state.requirements != original_state.requirements
            or remapped_state.verified_sources != original_state.verified_sources
            or remapped_state.explicit_sources != original_state.explicit_sources
        )

        applied: list[tuple[typing.Any, typing.Any]] = []
        try:
            for node in tuple(self._manager._tool_graph.nodes.values()):
                snapshot, changed = node.remap_provenance_owners(remap_spec)
                if changed:
                    applied.append((node, snapshot))
            if state_changed:
                state.replace(remapped_state)
        except Exception:
            with contextlib.suppress(Exception):
                state.replace(original_state)
            for node, snapshot in reversed(applied):
                with contextlib.suppress(Exception):
                    node.restore_provenance_owners(snapshot)
            raise
        for node, _snapshot in applied:
            node.commit_provenance_owner_remap()
        if state_changed and not applied:
            self._manager._mark_workspace_structure_dirty("Remapped extension script")

    def resolved_workspace_requirements(
        self,
        *,
        include_current: bool = True,
    ) -> tuple[_ResolvedWorkspaceRequirement, ...]:
        requirements = (
            self.collect_workspace_requirements()
            if include_current
            else self._manager._workspace_state.extension_scripts.requirements
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
                    f"Extension {requirement.script_name!r} is {resolved.state}. "
                    "Open Workspace Requirements for details."
                )
        return None

    def _resolve_requirement(
        self, requirement: _WorkspaceScriptRequirement
    ) -> _ResolvedWorkspaceRequirement:
        if requirement.extension_api_version != EXTENSION_API_VERSION:
            return _ResolvedWorkspaceRequirement(
                requirement=requirement,
                state="unsupported-api",
                detail=f"API {requirement.extension_api_version} is not supported",
            )
        if self.catalog.load_error is not None:
            return _ResolvedWorkspaceRequirement(
                requirement=requirement,
                state="missing",
                detail="The extension catalog is unavailable",
            )
        status = self.execution.capability_status(
            requirement.script_name,
            requirement.source_hash,
            requirement.capability_kind,
            requirement.capability_id,
        )
        if status == "ready":
            return _ResolvedWorkspaceRequirement(requirement=requirement, state="ready")
        if status == "validation-failed":
            return _ResolvedWorkspaceRequirement(
                requirement=requirement,
                state="validation-failed",
                detail=(
                    self.execution.validation_error(
                        requirement.script_name, requirement.source_hash
                    )
                    or "The script could not be validated"
                ),
            )
        if status == "approval-required":
            return _ResolvedWorkspaceRequirement(
                requirement=requirement, state="approval-required"
            )
        if status == "disabled":
            return _ResolvedWorkspaceRequirement(
                requirement=requirement, state="disabled"
            )
        if status == "unsupported-api":
            return _ResolvedWorkspaceRequirement(
                requirement=requirement,
                state="unsupported-api",
                detail="The registered capability uses an unsupported API",
            )
        if status == "hash-mismatch":
            return _ResolvedWorkspaceRequirement(
                requirement=requirement,
                state="hash-mismatch",
                detail="The registered script has different contents",
            )
        recoverable = any(
            _script_name_key(stored_name) == _script_name_key(requirement.script_name)
            and stored_hash == requirement.source_hash
            for stored_name, stored_hash in (
                self._manager._workspace_state.extension_scripts.verified_sources
            )
        )
        return _ResolvedWorkspaceRequirement(
            requirement=requirement,
            state="missing",
            detail=(
                "Save and register the script included with this workspace"
                if recoverable
                else "The required local script is unavailable"
            ),
        )

    @QtCore.Slot()
    def show_workspace_requirements(self) -> None:
        state = self._manager._workspace_state.extension_scripts
        resolved = self.resolved_workspace_requirements()
        dialog = _WorkspaceRequirementsDialog(
            resolved,
            self._manager,
            recoverable={
                (item.requirement.script_name, item.requirement.source_hash)
                for item in resolved
                if any(
                    _script_name_key(stored_name)
                    == _script_name_key(item.requirement.script_name)
                    and stored_hash == item.requirement.source_hash
                    for stored_name, stored_hash in state.verified_sources
                )
            },
            unresolved_count=self._unresolved_workspace_entry_count(),
        )

        def register_slot(script_name: str, source_hash: str) -> None:
            self._save_and_register_embedded_script(script_name, source_hash)
            if erlab.interactive.utils.qt_is_valid(dialog):
                dialog.set_requirements(
                    self.resolved_workspace_requirements(),
                    unresolved_count=self._unresolved_workspace_entry_count(),
                )

        dialog.register_requested.connect(register_slot)
        try:
            dialog.exec()
        finally:
            with contextlib.suppress(TypeError, RuntimeError):
                dialog.register_requested.disconnect(register_slot)

    def _unresolved_workspace_entry_count(self) -> int:
        """Return the visible count of opaque extension manifest entries."""
        state = self._manager._workspace_state.extension_scripts
        return (
            len(state.opaque_requirement_payloads)
            + len(state.opaque_source_payloads)
            + int(state.opaque_requirement_container is not None)
            + int(state.opaque_source_container is not None)
        )

    @QtCore.Slot(str, str)
    def _save_and_register_embedded_script(
        self, script_name: str, source_hash: str
    ) -> bool:
        """Save workspace source as a user file before registration and import."""
        state = self._manager._workspace_state.extension_scripts
        source = next(
            (
                value[1]
                for (stored_name, stored_hash), value in state.verified_sources.items()
                if _script_name_key(stored_name) == _script_name_key(script_name)
                and stored_hash == source_hash
            ),
            None,
        )
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
                for item in self.collect_workspace_requirements()
                if _script_name_key(item.script_name) == _script_name_key(script_name)
                and item.source_hash == source_hash
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
        suggested_name = requirement.script_name
        existing = self.catalog.model.extensions.get(_script_name_key(script_name))
        if existing is not None and existing.source_hash != source_hash:
            path_name = pathlib.Path(suggested_name)
            index = 1
            while True:
                suffix = "_workspace" if index == 1 else f"_workspace_{index}"
                candidate = f"{path_name.stem}{suffix}.py"
                if _script_name_key(candidate) not in self.catalog.model.extensions:
                    suggested_name = candidate
                    break
                index += 1
        destination = self._save_source_as_user_file(
            source,
            title="Save Workspace Extension Script",
            suggested_name=suggested_name,
        )
        if destination is None:
            return False
        try:
            destination_name = destination.name
            destination_key = _script_name_key(destination_name)
            current = self.catalog.store.read().extensions.get(destination_key)
            if current is None:
                catalog, _registered_hash = self.catalog.store.register_script(
                    destination,
                    expected_source_hash=source_hash,
                )
            elif current.source_hash != source_hash:
                QtWidgets.QMessageBox.warning(
                    self._manager,
                    "Script Name Already Used",
                    f"{current.script_name} is already registered with different "
                    "contents. Save this script with a different filename.",
                )
                return False
            elif pathlib.Path(current.source_path) != destination:
                catalog = self.catalog.store.relocate_script(
                    current.script_name,
                    destination,
                    expected_record_generation=current.record_generation,
                )
            else:
                catalog = self.catalog.store.read()
            self.catalog.refresh()
            record = catalog.extensions[destination_key]
            if not record.approved or not record.enabled:
                self.execution.validate_script(
                    record.script_name,
                    source_hash,
                    expected_record_generation=record.record_generation,
                )
                self.catalog.refresh()
            if destination_name != requirement.script_name:
                self._remap_workspace_script(
                    requirement.script_name,
                    source_hash,
                    destination_name,
                )
            self._reconcile_persisted_workspace_requirements()
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
        unresolved_count = self._unresolved_workspace_entry_count()
        unavailable = [
            item
            for item in self.resolved_workspace_requirements()
            if item.state != "ready"
        ]
        if not unavailable and unresolved_count == 0:
            return
        missing_registered_scripts = {
            item.requirement.script_name
            for item in unavailable
            if item.state == "missing"
            and (
                record := self.catalog.model.extensions.get(
                    _script_name_key(item.requirement.script_name)
                )
            )
            is not None
            and record.source_hash == item.requirement.source_hash
        }
        if missing_registered_scripts and self._show_missing_script_recovery(
            script_names=missing_registered_scripts,
            repeat=True,
        ):
            return
        state = self._manager._workspace_state.extension_scripts
        if (
            any(
                any(
                    _script_name_key(stored_name)
                    == _script_name_key(item.requirement.script_name)
                    and stored_hash == item.requirement.source_hash
                    for stored_name, stored_hash in state.verified_sources
                )
                for item in unavailable
            )
            or unresolved_count
        ):
            self.show_workspace_requirements()
            return
        states = ", ".join(
            f"{item.requirement.script_name}: {item.state}" for item in unavailable
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
            self.execution.validation_changed.disconnect(self._validation_changed_slot)
        with contextlib.suppress(TypeError, RuntimeError):
            self.catalog.changed.disconnect(self._catalog_changed_slot)
        with contextlib.suppress(TypeError, RuntimeError):
            self.catalog.read_failed.disconnect(self._catalog_read_failed_slot)
        self._routine_action_groups.clear()
        self.catalog.close()
