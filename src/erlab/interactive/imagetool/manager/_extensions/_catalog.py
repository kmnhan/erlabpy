"""Atomic global catalog for ImageTool Manager extensions."""

from __future__ import annotations

import datetime
import hashlib
import importlib.metadata
import json
import logging
import os
import pathlib
import re
import shutil
import threading
import types
import typing
import uuid

from qtpy import QtCore

from erlab.extensions import (
    EXTENSION_API_VERSION,
    ExtensionNotFoundError,
    LoaderDescriptor,
    RoutineDescriptor,
    load_script,
)
from erlab.extensions._api import (
    _CAPABILITY_ATTRIBUTE,
    _CapabilityStatus,
    _descriptor_for,
    _module_capabilities,
    _remove_resolvers,
    _resolve_loader_method,
    _set_capability_resolver,
    _set_capability_status_resolver,
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
    _ExtensionRecord,
    _ExtensionRevision,
    _revision_loader_name_filters,
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

    Environment entry-point verification is shared by managers in this process.
    Startup and explicit refresh replace the cache. Routine status checks use the
    verified entries without scanning editable source trees again.
    """

    _environment_entry_points: typing.ClassVar[
        dict[tuple[str, str, str, str], importlib.metadata.EntryPoint]
    ] = {}
    _environment_entry_points_lock: typing.ClassVar[threading.RLock] = threading.RLock()

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
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return _ExtensionCatalogModel.model_validate(payload)
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

    def _commit_with_staged_objects(
        self, catalog: _ExtensionCatalogModel, object_names: set[str]
    ) -> pathlib.Path | None:
        """Stage managed objects until the atomic catalog commit succeeds."""
        staging_directory = self.directory / f".removal-{uuid.uuid4().hex}"
        moved: list[tuple[pathlib.Path, pathlib.Path]] = []
        invalid_object = next(
            (
                name
                for name in object_names
                if re.fullmatch(r"[0-9a-f]{64}\.py", name) is None
            ),
            None,
        )
        if invalid_object is not None:
            raise _ExtensionCatalogError(
                f"Invalid managed extension object name: {invalid_object!r}"
            )
        try:
            for object_name in sorted(object_names):
                source = self.objects_directory / object_name
                if not source.is_file():
                    continue
                staging_directory.mkdir(parents=True, exist_ok=True)
                staged = staging_directory / object_name
                os.replace(source, staged)
                moved.append((source, staged))
            self._write_unlocked(catalog)
        except Exception as error:
            restore_errors: list[OSError] = []
            for source, staged in reversed(moved):
                try:
                    source.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(staged, source)
                except OSError as restore_error:
                    restore_errors.append(restore_error)
            if staging_directory.exists() and not restore_errors:
                shutil.rmtree(staging_directory, ignore_errors=True)
            if restore_errors:
                raise _ExtensionCatalogError(
                    "Could not commit the extension catalog or restore staged source "
                    f"objects. The retained staging path is {staging_directory}"
                ) from error
            raise
        if not moved:
            return None
        try:
            shutil.rmtree(staging_directory)
        except OSError:
            return staging_directory
        return None

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
        change_summary: str | None = None,
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
                        "change_summary": (
                            current.change_summary
                            if change_summary is None
                            else change_summary
                        ),
                    }
                )
                if updated_revision != current:
                    revisions = dict(existing.revisions)
                    revisions[revision_hash] = updated_revision
                    records[extension_id] = existing.model_copy(
                        update={
                            "revisions": revisions,
                            "record_generation": existing.record_generation + 1,
                        }
                    )
                    return catalog.model_copy(update={"extensions": records})
                return catalog
            if existing is not None and revision_hash in existing.revisions:
                revisions = dict(existing.revisions)
                revisions[revision_hash] = revisions[revision_hash].model_copy(
                    update={
                        "source_path": os.fspath(source_path),
                        "source_modified_at": modified_at,
                        "change_summary": (
                            revisions[revision_hash].change_summary
                            if change_summary is None
                            else change_summary
                        ),
                    }
                )
                records[extension_id] = existing.model_copy(
                    update={
                        "current_revision": revision_hash,
                        "enabled": False,
                        "revisions": revisions,
                        "record_generation": existing.record_generation + 1,
                    }
                )
                created = True
                return catalog.model_copy(update={"extensions": records})
            revision = _ExtensionRevision(
                source_hash=revision_hash,
                object_name=object_name,
                change_summary=change_summary or "",
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
        change_summary: str | None = None,
        source_modified_at: str | None = None,
        expected_record_generation: int | None = None,
        check_record_generation: bool = False,
    ) -> _ExtensionCatalogModel:
        """Copy approved workspace source into this catalog."""
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
                change_summary=change_summary or "",
                source_modified_at=source_modified_at,
                created_at=now,
                approved=False,
            )
            if existing is None:
                records[extension_id] = _ExtensionRecord(
                    id=extension_id,
                    name=name,
                    current_revision=actual_revision,
                    revisions={actual_revision: revision},
                    record_generation=1,
                )
            else:
                revisions = dict(existing.revisions)
                current_revision = revisions.get(actual_revision)
                if current_revision is None:
                    revisions[actual_revision] = revision
                elif (
                    current_revision.source_modified_at is None
                    and source_modified_at is not None
                ):
                    revisions[actual_revision] = current_revision.model_copy(
                        update={
                            "source_modified_at": source_modified_at,
                            "change_summary": (
                                current_revision.change_summary
                                if change_summary is None
                                else change_summary
                            ),
                        }
                    )
                elif (
                    change_summary is not None
                    and current_revision.change_summary != change_summary
                ):
                    revisions[actual_revision] = current_revision.model_copy(
                        update={"change_summary": change_summary}
                    )
                records[extension_id] = existing.model_copy(
                    update={
                        "current_revision": actual_revision,
                        "enabled": False,
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
            validated_revision = revision.model_copy(
                update={
                    "approved": True,
                    "routines": routines,
                    "loaders": loaders,
                    "import_error": None,
                    "loader_always_single": loader_always_single,
                    "loader_dialog_methods": loader_dialog_methods,
                }
            )
            name_filters = _revision_loader_name_filters(validated_revision)
            duplicate_filters = sorted(
                name_filter
                for name_filter in set(name_filters)
                if name_filters.count(name_filter) > 1
            )
            if duplicate_filters:
                joined = ", ".join(repr(value) for value in duplicate_filters)
                raise _ExtensionCatalogConflictError(
                    f"Extension {extension_id!r} provides duplicate file dialog "
                    f"filters: {joined}"
                )
            candidate_filters = set(name_filters)
            candidate_loader_names = (
                {descriptor.id for descriptor in validated_revision.loaders}
                if validated_revision.entry_point_group == "erlab.io.loaders"
                else set()
            )
            for other in catalog.extensions.values():
                if other.id == extension_id or not other.enabled:
                    continue
                other_revision = other.revisions[other.current_revision]
                if other_revision.entry_point_group == "erlab.io.loaders":
                    loader_name_conflicts = sorted(
                        candidate_loader_names.intersection(
                            descriptor.id for descriptor in other_revision.loaders
                        )
                    )
                    if loader_name_conflicts:
                        joined = ", ".join(
                            repr(value) for value in loader_name_conflicts
                        )
                        raise _ExtensionCatalogConflictError(
                            f"Extension {extension_id!r} conflicts with enabled "
                            f"extension {other.id!r} for loader names: {joined}"
                        )
                conflicts = sorted(
                    candidate_filters.intersection(
                        _revision_loader_name_filters(other_revision)
                    )
                )
                if conflicts:
                    joined = ", ".join(repr(value) for value in conflicts)
                    raise _ExtensionCatalogConflictError(
                        f"Extension {extension_id!r} conflicts with enabled extension "
                        f"{other.id!r} for file dialog filters: {joined}"
                    )
            revisions[revision_hash] = validated_revision
            records = dict(catalog.extensions)
            records[extension_id] = current.model_copy(
                update={
                    "enabled": True,
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
        inspected_entries: list[
            tuple[importlib.metadata.EntryPoint, str, str, bool, str]
        ] = []
        inspection_failures: list[
            tuple[importlib.metadata.EntryPoint, str, str, str, str]
        ] = []
        for entry in entries:
            try:
                dist_name, dist_version, payload, editable = (
                    _entry_point_revision_payload(entry)
                )
            except _EntryPointRevisionError as error:
                logger.warning(
                    "Could not inspect environment extension %s:%s",
                    entry.group,
                    entry.name,
                    exc_info=True,
                    extra={"suppress_ui_alert": True},
                )
                distribution = entry.dist
                try:
                    dist_name = (
                        entry.name
                        if distribution is None
                        else str(distribution.metadata.get("Name", entry.name))
                    )
                    dist_version = "" if distribution is None else distribution.version
                except (AttributeError, TypeError, ValueError):
                    dist_name = entry.name
                    dist_version = ""
                detail = str(error) or type(error).__name__
                failed_payload = json.dumps(
                    {
                        "group": entry.group,
                        "name": entry.name,
                        "value": entry.value,
                        "inspection_error": detail,
                    },
                    sort_keys=True,
                )
                inspection_failures.append(
                    (
                        entry,
                        dist_name,
                        dist_version,
                        detail,
                        hashlib.sha256(failed_payload.encode()).hexdigest(),
                    )
                )
                continue
            inspected_entries.append(
                (
                    entry,
                    dist_name,
                    dist_version,
                    editable,
                    hashlib.sha256(payload.encode()).hexdigest(),
                )
            )
        available_ids = {
            _safe_extension_id(f"environment.{entry.group}.{entry.name}")
            for entry in entries
        }

        def update(catalog: _ExtensionCatalogModel) -> _ExtensionCatalogModel:
            records = dict(catalog.extensions)
            changed = False
            for (
                entry,
                dist_name,
                dist_version,
                editable,
                revision_hash,
            ) in inspected_entries:
                extension_id = _safe_extension_id(
                    f"environment.{entry.group}.{entry.name}"
                )
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
                                "record_generation": existing.record_generation + 1,
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
                            "revisions": revisions,
                            "record_generation": existing.record_generation + 1,
                        }
                    )
                changed = True
            for (
                entry,
                dist_name,
                dist_version,
                import_error,
                revision_hash,
            ) in inspection_failures:
                extension_id = _safe_extension_id(
                    f"environment.{entry.group}.{entry.name}"
                )
                existing = records.get(extension_id)
                if (
                    existing is not None
                    and existing.source_type != "environment-package"
                ):
                    continue
                if existing is not None and revision_hash in existing.revisions:
                    if existing.current_revision != revision_hash or existing.enabled:
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
                    import_error=import_error,
                    entry_point_group=entry.group,
                    entry_point_name=entry.name,
                    entry_point_value=entry.value,
                    distribution_name=dist_name,
                    distribution_version=dist_version,
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
                            "revisions": revisions,
                            "record_generation": existing.record_generation + 1,
                        }
                    )
                changed = True
            for extension_id, record in tuple(records.items()):
                if (
                    record.source_type == "environment-package"
                    and extension_id not in available_ids
                ):
                    del records[extension_id]
                    changed = True
            if not changed:
                return catalog
            return catalog.model_copy(
                update={
                    "extensions": records,
                    "routine_favorites": tuple(
                        favorite
                        for favorite in catalog.routine_favorites
                        if favorite[0] in records
                    ),
                }
            )

        catalog = self.mutate(None, update)
        verified_entries = {
            (entry.group, entry.name, entry.value, revision_hash): entry
            for (
                entry,
                _dist_name,
                _dist_version,
                _editable,
                revision_hash,
            ) in inspected_entries
        }
        with self._environment_entry_points_lock:
            type(self)._environment_entry_points = verified_entries
        return catalog

    def _entry_point_for_revision(
        self, revision: _ExtensionRevision
    ) -> importlib.metadata.EntryPoint:
        key = (
            revision.entry_point_group or "",
            revision.entry_point_name or "",
            revision.entry_point_value or "",
            revision.source_hash,
        )
        with self._environment_entry_points_lock:
            cached = self._environment_entry_points.get(key)
            if cached is not None:
                return cached
            for entry_point in importlib.metadata.entry_points().select(group=key[0]):
                if entry_point.name == key[1] and entry_point.value == key[2]:
                    try:
                        revision_hash = _entry_point_revision(entry_point)
                    except _EntryPointRevisionError:
                        continue
                    if revision_hash == revision.source_hash:
                        self._environment_entry_points[key] = entry_point
                        return entry_point
        raise ImportError("The exact environment package revision is unavailable")

    def resolve_capability(
        self,
        extension_id: str,
        revision_hash: str,
        kind: str,
        capability_id: str,
        method: str | None = None,
    ) -> Callable[..., typing.Any]:
        """Resolve a pinned capability and an approved loader method."""
        record = self.read().extensions.get(extension_id)
        if record is None or revision_hash not in record.revisions:
            raise KeyError(f"Unknown extension revision {extension_id}:{revision_hash}")
        revision = record.revisions[revision_hash]
        if not record.enabled:
            raise ExtensionNotFoundError(f"Extension {extension_id!r} is disabled")
        if not revision.approved:
            raise ExtensionNotFoundError(
                f"Extension revision {extension_id}:{revision_hash} is not approved"
            )
        if record.source_type == "script":
            if method is not None:
                raise ExtensionNotFoundError(
                    "Decorated extension loaders do not provide alternate methods"
                )
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
        if entry_point.group == "erlab.io.loaders":
            if kind != "loader":
                raise KeyError(capability_id)
            approved_methods = {
                None,
                *(item.method for item in revision.loader_dialog_methods),
            }
            if method not in approved_methods:
                raise ExtensionNotFoundError(
                    "The requested loader method was not approved for this revision"
                )
        elif method is not None:
            raise ExtensionNotFoundError(
                "Decorated extension loaders do not provide alternate methods"
            )
        value = _load_entry_point_value(entry_point, revision_hash)
        if entry_point.group == "erlab.io.loaders":
            from erlab.io.dataloader import LoaderBase

            if isinstance(value, type) and issubclass(value, LoaderBase):
                registered = value()
            elif isinstance(value, LoaderBase):
                registered = value
            else:
                raise TypeError("The entry point does not provide LoaderBase")
            if registered.name != capability_id:
                raise KeyError(capability_id)
            return _resolve_loader_method(registered.load, method)
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

    def capability_status(
        self,
        extension_id: str,
        revision_hash: str,
        kind: str,
        capability_id: str,
        source_type: str | None = None,
    ) -> _CapabilityStatus:
        """Resolve exact catalog state without importing extension code."""
        record = self.read().extensions.get(extension_id)
        if record is None or revision_hash not in record.revisions:
            raise KeyError(f"Unknown extension revision {extension_id}:{revision_hash}")
        if source_type is not None and record.source_type != source_type:
            return "missing-revision"
        revision = record.revisions[revision_hash]
        if record.source_type == "environment-package":
            try:
                self._entry_point_for_revision(revision)
            except ImportError:
                return "missing-revision"
        else:
            try:
                source = (self.objects_directory / revision.object_name).read_bytes()
            except OSError:
                return "missing-revision"
            if hashlib.sha256(source).hexdigest() != revision_hash:
                return "hash-mismatch"
        if revision.import_error:
            return "import-failed"
        if not revision.approved:
            return "approval-required"
        descriptors = revision.routines if kind == "routine" else revision.loaders
        descriptor = next(
            (item for item in descriptors if item.id == capability_id), None
        )
        if descriptor is None:
            return "missing-capability"
        if descriptor.extension_api_version != EXTENSION_API_VERSION:
            return "unsupported-api"
        if not record.enabled:
            return "disabled"
        return "ready"

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
        embed_policy: typing.Literal["referenced", "always", "never"] | None = None,
    ) -> _ExtensionCatalogModel:
        def update(catalog: _ExtensionCatalogModel) -> _ExtensionCatalogModel:
            record = catalog.extensions[extension_id]
            values: dict[str, typing.Any] = {
                "record_generation": record.record_generation + 1
            }
            if enabled is not None:
                values["enabled"] = enabled
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

    def set_routine_favorite(
        self,
        extension_id: str,
        routine_id: str,
        *,
        favorite: bool,
    ) -> _ExtensionCatalogModel:
        """Add or remove one routine from the application favorites."""

        def update(catalog: _ExtensionCatalogModel) -> _ExtensionCatalogModel:
            if extension_id not in catalog.extensions:
                raise KeyError(extension_id)
            entry = (extension_id, routine_id)
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
        extension_id: str,
        *,
        expected_record_generation: int,
    ) -> tuple[_ExtensionCatalogModel, pathlib.Path | None]:
        """Permanently remove one script record and its unshared source objects."""
        lock = self._lock()
        try:
            current = self.read()
            record = current.extensions.get(extension_id)
            actual_generation = None if record is None else record.record_generation
            if actual_generation != expected_record_generation:
                raise _ExtensionCatalogConflictError(
                    f"Extension {extension_id!r} changed in another manager"
                )
            if record is None:
                raise KeyError(extension_id)
            if record.source_type != "script":
                raise _ExtensionCatalogConflictError(
                    "Environment packages cannot be removed through ERLab"
                )
            records = dict(current.extensions)
            del records[extension_id]
            referenced_objects = {
                revision.object_name
                for remaining in records.values()
                if remaining.source_type == "script"
                for revision in remaining.revisions.values()
            }
            removable_objects = {
                revision.object_name for revision in record.revisions.values()
            }.difference(referenced_objects)
            updated = current.model_copy(
                update={
                    "generation": current.generation + 1,
                    "extensions": records,
                    "routine_favorites": tuple(
                        favorite
                        for favorite in current.routine_favorites
                        if favorite[0] != extension_id
                    ),
                }
            )
            updated = _ExtensionCatalogModel.model_validate(
                updated.model_dump(mode="python")
            )
            retained = self._commit_with_staged_objects(updated, removable_objects)
            return updated, retained
        finally:
            lock.unlock()


class _ExtensionCatalog(QtCore.QObject):
    """Observe one application catalog across active manager windows.

    Atomic replacement removes the file watch on some Qt backends. Each refresh
    therefore restores both the file and directory watches.
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
        self._restore_watches()
        self._resolver_owner = uuid.uuid4().hex
        _set_revision_resolver(self._resolver_owner, self.store.source_path)
        _set_capability_resolver(self._resolver_owner, self.store.resolve_capability)
        _set_capability_status_resolver(
            self._resolver_owner, self.store.capability_status
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
        self._watcher.fileChanged.disconnect(self._schedule_refresh_slot)
        self._watcher.directoryChanged.disconnect(self._schedule_refresh_slot)
        self._refresh_timer.timeout.disconnect(self._refresh_slot)
        _remove_resolvers(self._resolver_owner)
