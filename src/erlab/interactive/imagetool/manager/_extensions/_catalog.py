"""Persistent script catalog for ImageTool Manager extensions."""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import os
import pathlib
import typing
import uuid

from qtpy import QtCore

from erlab.extensions import EXTENSION_API_VERSION, LoaderDescriptor, RoutineDescriptor
from erlab.extensions._api import (
    _CapabilityStatus,
    _RegisteredScriptCapability,
    _RegisteredScriptUnavailable,
    _remove_registered_script_backend,
    _set_registered_script_backend,
)
from erlab.interactive.imagetool.manager._extensions._models import (
    _ExtensionCatalogModel,
    _script_loader_name_filters,
    _script_name_key,
    _ScriptRecord,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable


class _ExtensionCatalogError(RuntimeError):
    pass


class _ExtensionCatalogConflictError(_ExtensionCatalogError):
    """The same script changed after an editor read it."""


class _ExtensionCatalogLockError(_ExtensionCatalogError):
    pass


@dataclasses.dataclass(frozen=True)
class _PinnedScript:
    """One catalog record and the exact local bytes read for it."""

    catalog_generation: int
    record: _ScriptRecord
    source_bytes: bytes = dataclasses.field(repr=False)

    @property
    def registered_path(self) -> pathlib.Path:
        """Return the absolute local file used for this snapshot."""
        return pathlib.Path(self.record.source_path)


def _default_catalog_directory() -> pathlib.Path:
    override = os.getenv("ERLAB_EXTENSION_CATALOG")
    if override:
        return pathlib.Path(override).expanduser().resolve()
    location = QtCore.QStandardPaths.writableLocation(
        QtCore.QStandardPaths.StandardLocation.GenericDataLocation
    )
    return pathlib.Path(location) / "ERLab" / "ImageTool Manager" / "extensions"


def _capability_descriptor(
    snapshot: _PinnedScript,
    kind: typing.Literal["routine", "loader"],
    capability_id: str,
) -> RoutineDescriptor | LoaderDescriptor | None:
    descriptors = (
        snapshot.record.routines if kind == "routine" else snapshot.record.loaders
    )
    return next((item for item in descriptors if item.id == capability_id), None)


def _capability_status(
    snapshot: _PinnedScript,
    kind: typing.Literal["routine", "loader"],
    capability_id: str,
    *,
    require_enabled: bool = True,
) -> _CapabilityStatus:
    record = snapshot.record
    if not record.approved:
        return "approval-required"
    descriptor = _capability_descriptor(snapshot, kind, capability_id)
    if descriptor is None:
        return "missing-capability"
    if descriptor.extension_api_version != EXTENSION_API_VERSION:
        return "unsupported-api"
    if require_enabled and not record.enabled:
        return "disabled"
    return "ready"


def _read_source_snapshot(path: pathlib.Path) -> tuple[bytes, str]:
    """Read source bytes and modification time from the same open local file."""
    try:
        with path.open("rb") as stream:
            source_bytes = stream.read()
            modified_timestamp = os.fstat(stream.fileno()).st_mtime
    except OSError as error:
        raise FileNotFoundError(path) from error
    modified_at = (
        datetime.datetime.fromtimestamp(modified_timestamp)
        .astimezone()
        .isoformat(timespec="seconds")
    )
    return source_bytes, modified_at


class _ExtensionCatalogStore:
    """Own catalog transactions and resolve exact local script snapshots.

    Every mutation re-reads the catalog while holding ``QLockFile``. A caller can
    merge an unrelated global change, but a changed ``record_generation`` rejects
    stale edits to the same script.
    """

    def __init__(self, directory: os.PathLike[str] | str | None = None) -> None:
        self.directory = (
            _default_catalog_directory()
            if directory is None
            else pathlib.Path(directory).expanduser().resolve()
        )
        self.path = self.directory / "catalog.json"
        self.lock_path = self.directory / "catalog.json.lock"

    def read(self) -> _ExtensionCatalogModel:
        if not self.path.exists():
            return _ExtensionCatalogModel()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return _ExtensionCatalogModel.model_validate(payload)
        except (OSError, TypeError, ValueError) as error:
            raise _ExtensionCatalogError(
                f"Could not read the extension catalog: {error}"
            ) from error

    def _lock(self) -> QtCore.QLockFile:
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise _ExtensionCatalogLockError(
                f"Could not create the extension catalog directory: {error}"
            ) from error
        lock = QtCore.QLockFile(os.fspath(self.lock_path))
        lock.setStaleLockTime(30_000)
        if not lock.tryLock(10_000):
            raise _ExtensionCatalogLockError(
                f"Could not lock the extension catalog: {lock.error()!s}"
            )
        return lock

    def _write_unlocked(self, catalog: _ExtensionCatalogModel) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = (catalog.model_dump_json(indent=2) + "\n").encode()
        save_file = QtCore.QSaveFile(os.fspath(self.path))
        if not save_file.open(QtCore.QIODevice.OpenModeFlag.WriteOnly):
            raise _ExtensionCatalogError(
                f"Could not open the extension catalog: {save_file.errorString()}"
            )
        if save_file.write(payload) != len(payload):
            save_file.cancelWriting()
            raise _ExtensionCatalogError(
                f"Could not write the extension catalog: {save_file.errorString()}"
            )
        if not save_file.commit():
            raise _ExtensionCatalogError(
                f"Could not commit the extension catalog: {save_file.errorString()}"
            )

    def mutate(
        self,
        script_name: str | None,
        callback: Callable[[_ExtensionCatalogModel], _ExtensionCatalogModel],
        *,
        expected_record_generation: int | None = None,
        check_record_generation: bool = False,
    ) -> _ExtensionCatalogModel:
        script_key = None if script_name is None else _script_name_key(script_name)
        lock = self._lock()
        try:
            current = self.read()
            if script_key is not None and (
                expected_record_generation is not None or check_record_generation
            ):
                record = current.extensions.get(script_key)
                actual = None if record is None else record.record_generation
                if actual != expected_record_generation:
                    raise _ExtensionCatalogConflictError(
                        f"Script {script_name!r} changed in another manager"
                    )
            updated = callback(current)
            if updated == current:
                return current
            updated = updated.model_copy(update={"generation": current.generation + 1})
            updated = _ExtensionCatalogModel.model_validate(
                updated.model_dump(mode="python")
            )
            self._write_unlocked(updated)
            return updated
        finally:
            lock.unlock()

    def resolve_script(
        self, script_name: str, expected_source_hash: str | None = None
    ) -> _PinnedScript:
        """Return one verified snapshot of the sole registered local script."""
        script_key = _script_name_key(script_name)
        lock = self._lock()
        try:
            catalog = self.read()
            record = catalog.extensions.get(script_key)
            if record is None:
                raise KeyError(script_name)
            if (
                expected_source_hash is not None
                and record.source_hash != expected_source_hash
            ):
                raise _ExtensionCatalogConflictError(
                    f"Registered script {record.script_name!r} has different contents"
                )
            source_path = pathlib.Path(record.source_path)
            try:
                source_bytes = source_path.read_bytes()
            except OSError as error:
                raise FileNotFoundError(source_path) from error
            if hashlib.sha256(source_bytes).hexdigest() != record.source_hash:
                raise _ExtensionCatalogConflictError(
                    f"Registered script {record.script_name!r} changed on disk"
                )
            return _PinnedScript(catalog.generation, record, source_bytes)
        finally:
            lock.unlock()

    def register_script(
        self,
        path: os.PathLike[str] | str,
        *,
        expected_source_hash: str | None = None,
    ) -> tuple[_ExtensionCatalogModel, str]:
        """Register a new local script by its case-insensitive filename."""
        source_path = pathlib.Path(path).expanduser().resolve()
        script_name = source_path.name
        script_key = _script_name_key(script_name)
        registered_at = (
            datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        )
        registered_source_hash: str | None = None

        def update(catalog: _ExtensionCatalogModel) -> _ExtensionCatalogModel:
            nonlocal registered_source_hash
            if script_key in catalog.extensions:
                existing = catalog.extensions[script_key]
                raise _ExtensionCatalogConflictError(
                    f"Script {existing.script_name!r} is already registered"
                )
            source_bytes, modified_at = _read_source_snapshot(source_path)
            source_hash = hashlib.sha256(source_bytes).hexdigest()
            if expected_source_hash is not None and source_hash != expected_source_hash:
                raise _ExtensionCatalogConflictError(
                    "The script source changed after it was reviewed"
                )
            registered_source_hash = source_hash
            records = dict(catalog.extensions)
            records[script_key] = _ScriptRecord(
                script_name=script_name,
                source_path=os.fspath(source_path),
                source_hash=source_hash,
                source_modified_at=modified_at,
                registered_at=registered_at,
                record_generation=1,
            )
            return catalog.model_copy(update={"extensions": records})

        catalog = self.mutate(None, update)
        return catalog, typing.cast("str", registered_source_hash)

    def relocate_script(
        self,
        script_name: str,
        path: os.PathLike[str] | str,
        *,
        expected_record_generation: int,
    ) -> _ExtensionCatalogModel:
        """Move one registration to an identical local script with the same name."""
        script_key = _script_name_key(script_name)
        source_path = pathlib.Path(path).expanduser().resolve()
        if source_path.name != script_name:
            raise _ExtensionCatalogConflictError(
                "The selected file has a different script name"
            )

        def update(catalog: _ExtensionCatalogModel) -> _ExtensionCatalogModel:
            record = catalog.extensions[script_key]
            if source_path.name != record.script_name:
                raise _ExtensionCatalogConflictError(
                    "The selected file has a different script name"
                )
            source_bytes, modified_at = _read_source_snapshot(source_path)
            source_hash = hashlib.sha256(source_bytes).hexdigest()
            if source_hash != record.source_hash:
                raise _ExtensionCatalogConflictError(
                    "The selected file has different script contents"
                )
            updated = record.model_copy(
                update={
                    "source_path": os.fspath(source_path),
                    "source_modified_at": modified_at,
                }
            )
            if updated == record:
                return catalog
            records = dict(catalog.extensions)
            records[script_key] = updated.model_copy(
                update={"record_generation": record.record_generation + 1}
            )
            return catalog.model_copy(update={"extensions": records})

        return self.mutate(
            script_name,
            update,
            expected_record_generation=expected_record_generation,
        )

    def reload_script(
        self,
        script_name: str,
        *,
        expected_source_hash: str,
        expected_record_generation: int,
    ) -> tuple[_ExtensionCatalogModel, bool]:
        """Record reviewed local contents and invalidate changed capabilities."""
        script_key = _script_name_key(script_name)
        changed: bool | None = None

        def update(catalog: _ExtensionCatalogModel) -> _ExtensionCatalogModel:
            nonlocal changed
            record = catalog.extensions[script_key]
            source_path = pathlib.Path(record.source_path)
            source_bytes, modified_at = _read_source_snapshot(source_path)
            source_hash = hashlib.sha256(source_bytes).hexdigest()
            if source_hash != expected_source_hash:
                raise _ExtensionCatalogConflictError(
                    "The script source changed after it was reviewed"
                )
            changed = source_hash != record.source_hash
            if not changed:
                return catalog
            records = dict(catalog.extensions)
            records[script_key] = record.model_copy(
                update={
                    "source_hash": source_hash,
                    "source_modified_at": modified_at,
                    "approved": False,
                    "enabled": False,
                    "routines": (),
                    "loaders": (),
                    "record_generation": record.record_generation + 1,
                }
            )
            return catalog.model_copy(update={"extensions": records})

        catalog = self.mutate(
            script_name,
            update,
            expected_record_generation=expected_record_generation,
        )
        return catalog, typing.cast("bool", changed)

    def commit_script_validation(
        self,
        script_name: str,
        *,
        source_hash: str,
        expected_record_generation: int,
        routines: tuple[RoutineDescriptor, ...],
        loaders: tuple[LoaderDescriptor, ...],
        enable_script: bool = True,
    ) -> _ExtensionCatalogModel:
        """Commit descriptors produced by execution-layer validation."""
        script_key = _script_name_key(script_name)

        def update(catalog: _ExtensionCatalogModel) -> _ExtensionCatalogModel:
            record = catalog.extensions[script_key]
            if source_hash != record.source_hash:
                raise _ExtensionCatalogConflictError(
                    f"Script {script_name!r} changed during validation"
                )
            source_bytes, _modified_at = _read_source_snapshot(
                pathlib.Path(record.source_path)
            )
            if hashlib.sha256(source_bytes).hexdigest() != source_hash:
                raise _ExtensionCatalogConflictError(
                    f"Script {script_name!r} changed during validation"
                )
            candidate = record.model_copy(
                update={"approved": True, "routines": routines, "loaders": loaders}
            )
            name_filters = _script_loader_name_filters(candidate)
            duplicate_filters = sorted(
                name_filter
                for name_filter in set(name_filters)
                if name_filters.count(name_filter) > 1
            )
            if duplicate_filters:
                joined = ", ".join(repr(value) for value in duplicate_filters)
                raise _ExtensionCatalogConflictError(
                    f"Script {script_name!r} provides duplicate file dialog filters: "
                    f"{joined}"
                )
            candidate_filters = set(name_filters)
            for other_key, other in catalog.extensions.items():
                if other_key == script_key or not other.enabled:
                    continue
                conflicts = sorted(
                    candidate_filters.intersection(_script_loader_name_filters(other))
                )
                if conflicts:
                    joined = ", ".join(repr(value) for value in conflicts)
                    raise _ExtensionCatalogConflictError(
                        f"Script {script_name!r} conflicts with enabled script "
                        f"{other.script_name!r} for file dialog filters: {joined}"
                    )
            records = dict(catalog.extensions)
            records[script_key] = candidate.model_copy(
                update={
                    "enabled": record.enabled or enable_script,
                    "record_generation": record.record_generation + 1,
                }
            )
            return catalog.model_copy(update={"extensions": records})

        return self.mutate(
            script_name,
            update,
            expected_record_generation=expected_record_generation,
        )

    def resolve_registered_capability(
        self,
        script_name: str,
        kind: typing.Literal["routine", "loader"],
        capability_id: str,
        *,
        source_hash: str | None = None,
        require_enabled: bool = True,
    ) -> _RegisteredScriptCapability:
        """Resolve one capability from one verified local-script snapshot."""
        try:
            snapshot = self.resolve_script(script_name, source_hash)
        except KeyError:
            raise
        except _ExtensionCatalogConflictError as error:
            raise _RegisteredScriptUnavailable("hash-mismatch") from error
        except (FileNotFoundError, _ExtensionCatalogError) as error:
            raise _RegisteredScriptUnavailable("missing-source") from error
        status = _capability_status(
            snapshot,
            kind,
            capability_id,
            require_enabled=require_enabled,
        )
        if status != "ready":
            raise _RegisteredScriptUnavailable(status)
        return _RegisteredScriptCapability(
            registered_path=snapshot.registered_path,
            script_name=snapshot.record.script_name,
            source_hash=snapshot.record.source_hash,
            descriptor=typing.cast(
                "RoutineDescriptor | LoaderDescriptor",
                _capability_descriptor(snapshot, kind, capability_id),
            ),
            source_bytes=snapshot.source_bytes,
        )

    def update_script(
        self,
        script_name: str,
        *,
        expected_record_generation: int,
        enabled: bool | None = None,
        embed_policy: typing.Literal["referenced", "always", "never"] | None = None,
    ) -> _ExtensionCatalogModel:
        script_key = _script_name_key(script_name)

        def update(catalog: _ExtensionCatalogModel) -> _ExtensionCatalogModel:
            record = catalog.extensions[script_key]
            values: dict[str, typing.Any] = {}
            if enabled is not None:
                values["enabled"] = enabled
            if embed_policy is not None:
                values["embed_policy"] = embed_policy
            updated = record.model_copy(update=values)
            if updated == record:
                return catalog
            records = dict(catalog.extensions)
            records[script_key] = updated.model_copy(
                update={"record_generation": record.record_generation + 1}
            )
            return catalog.model_copy(update={"extensions": records})

        return self.mutate(
            script_name,
            update,
            expected_record_generation=expected_record_generation,
        )

    def set_routine_favorite(
        self,
        script_name: str,
        routine_id: str,
        *,
        favorite: bool,
    ) -> _ExtensionCatalogModel:
        """Add or remove one routine from the application favorites."""
        script_key = _script_name_key(script_name)

        def update(catalog: _ExtensionCatalogModel) -> _ExtensionCatalogModel:
            if script_key not in catalog.extensions:
                raise KeyError(script_name)
            entry = (script_key, routine_id)
            favorites = list(catalog.routine_favorites)
            if favorite and entry not in favorites:
                favorites.append(entry)
            elif not favorite and entry in favorites:
                favorites.remove(entry)
            else:
                return catalog
            return catalog.model_copy(
                update={"routine_favorites": tuple(sorted(favorites))}
            )

        return self.mutate(None, update)

    def remove_script(
        self,
        script_name: str,
        *,
        expected_record_generation: int,
    ) -> _ExtensionCatalogModel:
        """Remove one registration without modifying its local Python file."""
        script_key = _script_name_key(script_name)

        def update(catalog: _ExtensionCatalogModel) -> _ExtensionCatalogModel:
            records = dict(catalog.extensions)
            del records[script_key]
            return catalog.model_copy(
                update={
                    "extensions": records,
                    "routine_favorites": tuple(
                        entry
                        for entry in catalog.routine_favorites
                        if entry[0] != script_key
                    ),
                }
            )

        return self.mutate(
            script_name,
            update,
            expected_record_generation=expected_record_generation,
        )


class _ExtensionCatalog(QtCore.QObject):
    """Observe one application catalog across active manager windows.

    Atomic replacement removes the file watch on some Qt backends. Each refresh
    therefore restores both the file and directory watches.
    """

    changed = QtCore.Signal(object)
    read_failed = QtCore.Signal(str)

    def __init__(
        self,
        *,
        directory: os.PathLike[str] | str | None = None,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._closed = False
        self.store = _ExtensionCatalogStore(directory)
        self.load_error: str | None = None
        try:
            self.model = self.store.read()
        except _ExtensionCatalogError as error:
            self.load_error = str(error)
            self.model = _ExtensionCatalogModel()
        self._watcher = QtCore.QFileSystemWatcher(self)
        self._schedule_refresh_slot = self._schedule_refresh
        self._watcher.fileChanged.connect(self._schedule_refresh_slot)
        self._watcher.directoryChanged.connect(self._schedule_refresh_slot)
        self._refresh_timer = QtCore.QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_slot = self.refresh
        self._refresh_timer.timeout.connect(self._refresh_slot)
        self._restore_watches()
        self._resolver_owner = uuid.uuid4().hex
        _set_registered_script_backend(self._resolver_owner, self.store)

    def _restore_watches(self) -> bool:
        try:
            self.store.directory.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            detail = f"Could not access the extension catalog directory: {error}"
            if detail != self.load_error:
                self.load_error = detail
                self.read_failed.emit(detail)
            return False
        wanted = {os.fspath(self.store.directory)}
        if self.store.path.exists():
            wanted.add(os.fspath(self.store.path))
        current = set(self._watcher.files()) | set(self._watcher.directories())
        stale = current - wanted
        if stale:
            self._watcher.removePaths(list(stale))
        missing = wanted - current
        if missing:
            self._watcher.addPaths(list(missing))
        return True

    @QtCore.Slot()
    def _schedule_refresh(self) -> None:
        if self._closed or self._refresh_timer.isActive():
            return
        self._refresh_timer.start(0)

    @QtCore.Slot()
    def refresh(self) -> None:
        if self._closed:
            return
        if not self._restore_watches():
            return
        try:
            model = self.store.read()
        except _ExtensionCatalogError as error:
            detail = str(error)
            if detail != self.load_error:
                self.load_error = detail
                self.model = _ExtensionCatalogModel()
                self.read_failed.emit(detail)
                self.changed.emit(self.model)
            return
        recovered = self.load_error is not None
        self.load_error = None
        if model == self.model and not recovered:
            return
        self.model = model
        self.changed.emit(model)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._refresh_timer.stop()
        self._watcher.fileChanged.disconnect(self._schedule_refresh_slot)
        self._watcher.directoryChanged.disconnect(self._schedule_refresh_slot)
        self._refresh_timer.timeout.disconnect(self._refresh_slot)
        _remove_registered_script_backend(self._resolver_owner)
