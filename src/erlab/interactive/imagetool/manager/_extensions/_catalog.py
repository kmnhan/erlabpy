"""Atomic global catalog for ImageTool Manager extensions."""

from __future__ import annotations

import datetime
import hashlib
import importlib.metadata
import logging
import os
import pathlib
import re
import types
import typing
import uuid

from qtpy import QtCore

from erlab.extensions import (
    ExtensionNotFoundError,
    LoaderDescriptor,
    RoutineDescriptor,
    load_script,
)
from erlab.extensions._api import (
    _CAPABILITY_ATTRIBUTE,
    _descriptor_for,
    _module_capabilities,
    _remove_resolvers,
    _set_capability_availability_resolver,
    _set_capability_resolver,
    _set_revision_resolver,
)
from erlab.extensions._entry_points import (
    _entry_point_revision,
    _entry_point_revision_payload,
    _EntryPointRevisionError,
    _load_entry_point_value,
)
from erlab.interactive.imagetool.manager._extensions._models import (
    _EnvironmentLoaderMethod,
    _ExtensionCatalogModel,
    _ExtensionMetadata,
    _ExtensionRecord,
    _ExtensionRevision,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class _ExtensionCatalogError(RuntimeError):
    pass


class _ExtensionCatalogConflictError(_ExtensionCatalogError):
    """The same extension changed after an editor read it."""


class _ExtensionCatalogLockError(_ExtensionCatalogError):
    pass


def _default_catalog_directory() -> pathlib.Path:
    override = os.getenv("ERLAB_EXTENSION_CATALOG")
    if override:
        return pathlib.Path(override).expanduser().resolve()
    location = QtCore.QStandardPaths.writableLocation(
        QtCore.QStandardPaths.StandardLocation.AppDataLocation
    )
    return pathlib.Path(location) / "extensions"


def _safe_extension_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-.")
    if not normalized:
        normalized = f"extension-{uuid.uuid4().hex[:8]}"
    return normalized


class _ExtensionCatalogStore:
    """Own catalog locking, generation checks, and atomic commits.

    Every mutation re-reads the catalog while holding ``QLockFile``. A caller can
    merge an unrelated global change, but a changed ``record_generation`` rejects
    stale edits to the same extension.
    """

    def __init__(self, directory: os.PathLike[str] | str | None = None) -> None:
        self.directory = (
            _default_catalog_directory()
            if directory is None
            else pathlib.Path(directory).expanduser().resolve()
        )
        self.path = self.directory / "catalog.json"
        self.lock_path = self.directory / "catalog.json.lock"
        self.objects_directory = self.directory / "objects"

    def read(self) -> _ExtensionCatalogModel:
        if not self.path.exists():
            return _ExtensionCatalogModel()
        try:
            return _ExtensionCatalogModel.model_validate_json(
                self.path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as error:
            raise _ExtensionCatalogError(
                f"Could not read the extension catalog: {error}"
            ) from error

    def _lock(self) -> QtCore.QLockFile:
        self.directory.mkdir(parents=True, exist_ok=True)
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
        extension_id: str | None,
        callback: Callable[[_ExtensionCatalogModel], _ExtensionCatalogModel],
        *,
        expected_record_generation: int | None = None,
        check_record_generation: bool = False,
    ) -> _ExtensionCatalogModel:
        lock = self._lock()
        try:
            current = self.read()
            if extension_id is not None and (
                expected_record_generation is not None or check_record_generation
            ):
                record = current.extensions.get(extension_id)
                actual = None if record is None else record.record_generation
                if actual != expected_record_generation:
                    raise _ExtensionCatalogConflictError(
                        f"Extension {extension_id!r} changed in another manager"
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

    def source_path(self, extension_id: str, revision: str) -> pathlib.Path:
        record = self.read().extensions.get(extension_id)
        if record is None or revision not in record.revisions:
            raise KeyError(f"Unknown extension revision {extension_id}:{revision}")
        path = self.objects_directory / record.revisions[revision].object_name
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def _store_script_source(self, source: bytes, revision_hash: str) -> str:
        """Store verified bytes and atomically repair a corrupt source object."""
        if hashlib.sha256(source).hexdigest() != revision_hash:
            raise ValueError("Extension source does not match its revision hash")
        object_name = f"{revision_hash}.py"
        self.objects_directory.mkdir(parents=True, exist_ok=True)
        object_path = self.objects_directory / object_name
        try:
            if hashlib.sha256(object_path.read_bytes()).hexdigest() == revision_hash:
                return object_name
        except OSError:
            pass
        save_file = QtCore.QSaveFile(os.fspath(object_path))
        if not save_file.open(QtCore.QIODevice.OpenModeFlag.WriteOnly):
            raise _ExtensionCatalogError(save_file.errorString())
        if save_file.write(source) != len(source):
            save_file.cancelWriting()
            raise _ExtensionCatalogError(save_file.errorString())
        if not save_file.commit():
            raise _ExtensionCatalogError(save_file.errorString())
        return object_name

    def add_script(
        self,
        path: os.PathLike[str] | str,
        *,
        extension_id: str | None = None,
        name: str | None = None,
        metadata: _ExtensionMetadata | None = None,
        approved: bool = False,
        expected_revision: str | None = None,
        expected_record_generation: int | None = None,
        check_record_generation: bool = False,
    ) -> tuple[_ExtensionCatalogModel, str, bool]:
        source_path = pathlib.Path(path).expanduser().resolve()
        source = source_path.read_bytes()
        revision_hash = hashlib.sha256(source).hexdigest()
        if expected_revision is not None and revision_hash != expected_revision:
            raise _ExtensionCatalogConflictError(
                "The script source changed after it was reviewed"
            )
        extension_id = _safe_extension_id(extension_id or source_path.stem)
        object_name = self._store_script_source(source, revision_hash)
        modified_at = (
            datetime.datetime.fromtimestamp(source_path.stat().st_mtime)
            .astimezone()
            .isoformat(timespec="seconds")
        )
        now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        created = False

        def update(catalog: _ExtensionCatalogModel) -> _ExtensionCatalogModel:
            nonlocal created
            records = dict(catalog.extensions)
            existing = records.get(extension_id)
            if existing is not None and existing.source_type != "script":
                raise _ExtensionCatalogConflictError(
                    f"Extension {extension_id!r} is an environment package"
                )
            if existing is not None and revision_hash == existing.current_revision:
                current = existing.revisions[revision_hash]
                updated_revision = current.model_copy(
                    update={
                        "source_path": os.fspath(source_path),
                        "source_modified_at": modified_at,
                    }
                )
                if updated_revision != current or (
                    metadata is not None and metadata != existing.metadata
                ):
                    revisions = dict(existing.revisions)
                    revisions[revision_hash] = updated_revision
                    records[extension_id] = existing.model_copy(
                        update={
                            "metadata": (
                                existing.metadata if metadata is None else metadata
                            ),
                            "revisions": revisions,
                            "record_generation": existing.record_generation + 1,
                        }
                    )
                    return catalog.model_copy(update={"extensions": records})
                return catalog
            if existing is not None and revision_hash in existing.revisions:
                records[extension_id] = existing.model_copy(
                    update={
                        "current_revision": revision_hash,
                        "enabled": False,
                        "metadata": metadata or existing.metadata,
                        "record_generation": existing.record_generation + 1,
                    }
                )
                created = True
                return catalog.model_copy(update={"extensions": records})
            revision = _ExtensionRevision(
                source_hash=revision_hash,
                object_name=object_name,
                source_path=os.fspath(source_path),
                source_modified_at=modified_at,
                created_at=now,
                approved=approved,
            )
            if existing is None:
                record = _ExtensionRecord(
                    id=extension_id,
                    name=name or source_path.stem.replace("_", " ").title(),
                    current_revision=revision_hash,
                    metadata=metadata or _ExtensionMetadata(),
                    revisions={revision_hash: revision},
                    record_generation=1,
                )
            else:
                revisions = dict(existing.revisions)
                revisions[revision_hash] = revision
                record = existing.model_copy(
                    update={
                        "current_revision": revision_hash,
                        "enabled": False,
                        "metadata": metadata or existing.metadata,
                        "revisions": revisions,
                        "record_generation": existing.record_generation + 1,
                    }
                )
            records[extension_id] = record
            created = True
            return catalog.model_copy(update={"extensions": records})

        catalog = self.mutate(
            extension_id,
            update,
            expected_record_generation=expected_record_generation,
            check_record_generation=check_record_generation,
        )
        return catalog, revision_hash, created

    def add_embedded_script(
        self,
        source: bytes,
        *,
        extension_id: str,
        expected_revision: str,
        name: str,
        metadata: _ExtensionMetadata,
        expected_record_generation: int | None = None,
        check_record_generation: bool = False,
    ) -> _ExtensionCatalogModel:
        """Copy approved workspace source into the application catalog."""
        actual_revision = hashlib.sha256(source).hexdigest()
        if actual_revision != expected_revision:
            raise ValueError(
                "Embedded extension source hash does not match its manifest"
            )
        extension_id = _safe_extension_id(extension_id)
        object_name = self._store_script_source(source, actual_revision)
        now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")

        def update(catalog: _ExtensionCatalogModel) -> _ExtensionCatalogModel:
            records = dict(catalog.extensions)
            existing = records.get(extension_id)
            if existing is not None and existing.source_type != "script":
                raise _ExtensionCatalogConflictError(
                    f"Extension {extension_id!r} is an environment package"
                )
            revision = _ExtensionRevision(
                source_hash=actual_revision,
                object_name=object_name,
                created_at=now,
                approved=False,
            )
            if existing is None:
                records[extension_id] = _ExtensionRecord(
                    id=extension_id,
                    name=name,
                    current_revision=actual_revision,
                    metadata=metadata,
                    revisions={actual_revision: revision},
                    record_generation=1,
                )
            else:
                revisions = dict(existing.revisions)
                revisions.setdefault(actual_revision, revision)
                records[extension_id] = existing.model_copy(
                    update={
                        "current_revision": actual_revision,
                        "enabled": False,
                        "removed": False,
                        "metadata": metadata,
                        "revisions": revisions,
                        "record_generation": existing.record_generation + 1,
                    }
                )
            return catalog.model_copy(update={"extensions": records})

        return self.mutate(
            extension_id,
            update,
            expected_record_generation=expected_record_generation,
            check_record_generation=check_record_generation,
        )

    def record_validation_failure(
        self,
        extension_id: str,
        *,
        revision_hash: str,
        expected_record_generation: int,
        import_error: str,
    ) -> _ExtensionCatalogModel:
        """Persist a failed import without changing a newer revision."""

        def update(catalog: _ExtensionCatalogModel) -> _ExtensionCatalogModel:
            current = catalog.extensions[extension_id]
            if current.current_revision != revision_hash:
                raise _ExtensionCatalogConflictError(
                    f"Extension {extension_id!r} changed during validation"
                )
            revisions = dict(current.revisions)
            revisions[revision_hash] = revisions[revision_hash].model_copy(
                update={"import_error": import_error, "approved": False}
            )
            records = dict(catalog.extensions)
            records[extension_id] = current.model_copy(
                update={
                    "enabled": False,
                    "revisions": revisions,
                    "record_generation": current.record_generation + 1,
                }
            )
            return catalog.model_copy(update={"extensions": records})

        return self.mutate(
            extension_id,
            update,
            expected_record_generation=expected_record_generation,
        )

    def enable_validated_revision(
        self,
        extension_id: str,
        *,
        revision_hash: str,
        expected_record_generation: int,
        routines: tuple[RoutineDescriptor, ...],
        loaders: tuple[LoaderDescriptor, ...],
        loader_always_single: bool | None,
        loader_dialog_methods: tuple[_EnvironmentLoaderMethod, ...],
    ) -> _ExtensionCatalogModel:
        """Commit descriptors produced by execution-layer validation."""

        def update(catalog: _ExtensionCatalogModel) -> _ExtensionCatalogModel:
            current = catalog.extensions[extension_id]
            if current.current_revision != revision_hash:
                raise _ExtensionCatalogConflictError(
                    f"Extension {extension_id!r} changed during validation"
                )
            revisions = dict(current.revisions)
            revision = revisions[revision_hash]
            revisions[revision_hash] = revision.model_copy(
                update={
                    "approved": True,
                    "routines": routines,
                    "loaders": loaders,
                    "import_error": None,
                    "loader_always_single": loader_always_single,
                    "loader_dialog_methods": loader_dialog_methods,
                }
            )
            records = dict(catalog.extensions)
            records[extension_id] = current.model_copy(
                update={
                    "enabled": True,
                    "removed": False,
                    "revisions": revisions,
                    "record_generation": current.record_generation + 1,
                }
            )
            return catalog.model_copy(update={"extensions": records})

        return self.mutate(
            extension_id,
            update,
            expected_record_generation=expected_record_generation,
        )

    def refresh_environment_packages(self) -> _ExtensionCatalogModel:
        """Refresh entry-point metadata without importing package code."""
        entries = tuple(
            entry
            for group in ("erlab.extensions", "erlab.io.loaders")
            for entry in importlib.metadata.entry_points().select(group=group)
        )
        discovered_ids: set[str] = set()

        def update(catalog: _ExtensionCatalogModel) -> _ExtensionCatalogModel:
            records = dict(catalog.extensions)
            changed = False
            for entry in entries:
                extension_id = _safe_extension_id(
                    f"environment.{entry.group}.{entry.name}"
                )
                discovered_ids.add(extension_id)
                try:
                    dist_name, dist_version, payload, editable = (
                        _entry_point_revision_payload(entry)
                    )
                except _EntryPointRevisionError:
                    logger.warning(
                        "Could not inspect environment extension %s:%s",
                        entry.group,
                        entry.name,
                        exc_info=True,
                        extra={"suppress_ui_alert": True},
                    )
                    continue
                revision_hash = hashlib.sha256(payload.encode()).hexdigest()
                existing = records.get(extension_id)
                if (
                    existing is not None
                    and existing.source_type != "environment-package"
                ):
                    continue
                if existing is not None and revision_hash in existing.revisions:
                    if revision_hash != existing.current_revision:
                        records[extension_id] = existing.model_copy(
                            update={
                                "current_revision": revision_hash,
                                "enabled": False,
                                "record_generation": (existing.record_generation + 1),
                            }
                        )
                        changed = True
                    continue
                revision = _ExtensionRevision(
                    source_hash=revision_hash,
                    object_name=entry.value,
                    created_at=datetime.datetime.now()
                    .astimezone()
                    .isoformat(timespec="seconds"),
                    entry_point_group=entry.group,
                    entry_point_name=entry.name,
                    entry_point_value=entry.value,
                    distribution_name=dist_name,
                    distribution_version=dist_version,
                    editable=editable,
                )
                if existing is None:
                    records[extension_id] = _ExtensionRecord(
                        id=extension_id,
                        name=entry.name.replace("_", " ").title(),
                        source_type="environment-package",
                        current_revision=revision_hash,
                        revisions={revision_hash: revision},
                        record_generation=1,
                    )
                else:
                    revisions = dict(existing.revisions)
                    revisions[revision_hash] = revision
                    records[extension_id] = existing.model_copy(
                        update={
                            "current_revision": revision_hash,
                            "enabled": False,
                            "removed": existing.removed,
                            "revisions": revisions,
                            "record_generation": existing.record_generation + 1,
                        }
                    )
                changed = True
            for extension_id, record in tuple(records.items()):
                if (
                    record.source_type == "environment-package"
                    and extension_id not in discovered_ids
                    and not record.removed
                ):
                    records[extension_id] = record.model_copy(
                        update={
                            "enabled": False,
                            "removed": True,
                            "record_generation": record.record_generation + 1,
                        }
                    )
                    changed = True
            if not changed:
                return catalog
            return catalog.model_copy(update={"extensions": records})

        return self.mutate(None, update)

    @staticmethod
    def _entry_point_for_revision(
        revision: _ExtensionRevision,
    ) -> importlib.metadata.EntryPoint:
        for entry_point in importlib.metadata.entry_points().select(
            group=revision.entry_point_group or ""
        ):
            if (
                entry_point.name == revision.entry_point_name
                and entry_point.value == revision.entry_point_value
            ):
                try:
                    revision_hash = _entry_point_revision(entry_point)
                except _EntryPointRevisionError:
                    continue
                if revision_hash == revision.source_hash:
                    return entry_point
        raise ImportError("The exact environment package revision is unavailable")

    def resolve_capability(
        self,
        extension_id: str,
        revision_hash: str,
        kind: str,
        capability_id: str,
    ) -> Callable[..., typing.Any]:
        """Resolve a pinned capability for public replay calls."""
        record = self.read().extensions.get(extension_id)
        if record is None or revision_hash not in record.revisions:
            raise KeyError(f"Unknown extension revision {extension_id}:{revision_hash}")
        revision = record.revisions[revision_hash]
        if record.removed or not record.enabled:
            raise ExtensionNotFoundError(f"Extension {extension_id!r} is disabled")
        if not revision.approved:
            raise ExtensionNotFoundError(
                f"Extension revision {extension_id}:{revision_hash} is not approved"
            )
        if record.source_type == "script":
            loaded = load_script(
                self.source_path(extension_id, revision_hash),
                expected_revision=revision_hash,
            )
            entries = loaded.routines if kind == "routine" else loaded.loaders
            try:
                return entries[capability_id][1]
            except KeyError as error:
                raise KeyError(
                    f"Unknown {kind} capability {capability_id!r}"
                ) from error
        entry_point = self._entry_point_for_revision(revision)
        value = _load_entry_point_value(entry_point, revision_hash)
        if entry_point.group == "erlab.io.loaders":
            if kind != "loader":
                raise KeyError(capability_id)
            from erlab.io.dataloader import LoaderBase

            if isinstance(value, type) and issubclass(value, LoaderBase):
                registered = value()
            elif isinstance(value, LoaderBase):
                registered = value
            else:
                raise TypeError("The entry point does not provide LoaderBase")
            if registered.name != capability_id:
                raise KeyError(capability_id)
            return registered.load
        if isinstance(value, types.ModuleType):
            routines, loaders = _module_capabilities(value)
            entries = routines if kind == "routine" else loaders
            return entries[capability_id][1]
        descriptor = _descriptor_for(value, getattr(value, _CAPABILITY_ATTRIBUTE))
        if descriptor.id != capability_id:
            raise KeyError(capability_id)
        if kind == "routine" and not isinstance(descriptor, RoutineDescriptor):
            raise TypeError("The capability is not a routine")
        if kind == "loader" and not isinstance(descriptor, LoaderDescriptor):
            raise TypeError("The capability is not a loader")
        return value

    def capability_available(
        self,
        extension_id: str,
        revision_hash: str,
        kind: str,
        capability_id: str,
    ) -> bool:
        """Check exact catalog metadata without importing extension code."""
        record = self.read().extensions.get(extension_id)
        if record is None or revision_hash not in record.revisions:
            raise KeyError(f"Unknown extension revision {extension_id}:{revision_hash}")
        revision = record.revisions[revision_hash]
        if record.removed or not record.enabled or not revision.approved:
            return False
        descriptors = revision.routines if kind == "routine" else revision.loaders
        if not any(descriptor.id == capability_id for descriptor in descriptors):
            return False
        return self.revision_available(record, revision_hash)

    def revision_available(self, record: _ExtensionRecord, revision_hash: str) -> bool:
        """Check an exact revision source without importing extension code."""
        revision = record.revisions.get(revision_hash)
        if revision is None:
            return False
        if record.source_type == "environment-package":
            try:
                self._entry_point_for_revision(revision)
            except ImportError:
                return False
            return True
        try:
            source = (self.objects_directory / revision.object_name).read_bytes()
        except OSError:
            return False
        return hashlib.sha256(source).hexdigest() == revision_hash

    def update_record(
        self,
        extension_id: str,
        *,
        expected_record_generation: int,
        enabled: bool | None = None,
        favorite: bool | None = None,
        removed: bool | None = None,
        embed_policy: typing.Literal["referenced", "always", "never"] | None = None,
        metadata: _ExtensionMetadata | None = None,
    ) -> _ExtensionCatalogModel:
        def update(catalog: _ExtensionCatalogModel) -> _ExtensionCatalogModel:
            record = catalog.extensions[extension_id]
            values: dict[str, typing.Any] = {
                "record_generation": record.record_generation + 1
            }
            if enabled is not None:
                values["enabled"] = enabled
            if favorite is not None:
                values["favorite"] = favorite
            if removed is not None:
                values["removed"] = removed
                if removed:
                    values["enabled"] = False
            if metadata is not None:
                values["metadata"] = metadata
            if embed_policy is not None:
                values["embed_policy"] = embed_policy
            records = dict(catalog.extensions)
            records[extension_id] = record.model_copy(update=values)
            return catalog.model_copy(update={"extensions": records})

        return self.mutate(
            extension_id,
            update,
            expected_record_generation=expected_record_generation,
        )


class _ExtensionCatalog(QtCore.QObject):
    """Observe one application catalog across active manager windows.

    Atomic replacement removes the file watch on some Qt backends. Each refresh
    therefore restores both the file and directory watches. A generation poll is a
    fallback for missed filesystem notifications.
    """

    changed = QtCore.Signal(object)

    def __init__(
        self,
        *,
        directory: os.PathLike[str] | str | None = None,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._closed = False
        self.store = _ExtensionCatalogStore(directory)
        self.model = self.store.read()
        self._watcher = QtCore.QFileSystemWatcher(self)
        self._schedule_refresh_slot = self._schedule_refresh
        self._watcher.fileChanged.connect(self._schedule_refresh_slot)
        self._watcher.directoryChanged.connect(self._schedule_refresh_slot)
        self._refresh_timer = QtCore.QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_slot = self.refresh
        self._refresh_timer.timeout.connect(self._refresh_slot)
        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.setInterval(2_000)
        self._poll_timer.timeout.connect(self._refresh_slot)
        self._poll_timer.start()
        self._restore_watches()
        self._resolver_owner = uuid.uuid4().hex
        _set_revision_resolver(self._resolver_owner, self.store.source_path)
        _set_capability_resolver(self._resolver_owner, self.store.resolve_capability)
        _set_capability_availability_resolver(
            self._resolver_owner, self.store.capability_available
        )

    def _restore_watches(self) -> None:
        self.store.directory.mkdir(parents=True, exist_ok=True)
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

    @QtCore.Slot()
    def _schedule_refresh(self) -> None:
        if self._closed or self._refresh_timer.isActive():
            return
        self._refresh_timer.start(0)

    @QtCore.Slot()
    def refresh(self) -> None:
        if self._closed:
            return
        self._restore_watches()
        model = self.store.read()
        if model.generation == self.model.generation:
            return
        self.model = model
        self.changed.emit(model)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._refresh_timer.stop()
        self._poll_timer.stop()
        self._watcher.fileChanged.disconnect(self._schedule_refresh_slot)
        self._watcher.directoryChanged.disconnect(self._schedule_refresh_slot)
        self._refresh_timer.timeout.disconnect(self._refresh_slot)
        self._poll_timer.timeout.disconnect(self._refresh_slot)
        _remove_resolvers(self._resolver_owner)
