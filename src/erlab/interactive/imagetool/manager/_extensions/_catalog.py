"""Persistent script catalog for ImageTool Manager extensions."""

from __future__ import annotations

import contextlib
import datetime
import hashlib
import json
import os
import pathlib
import re
import shutil
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
    _CapabilityStatus,
    _remove_resolvers,
    _set_capability_resolver,
    _set_capability_status_resolver,
    _set_script_capability_reference_resolver,
    _set_source_resolver,
)
from erlab.interactive.imagetool.manager._extensions._models import (
    _ExtensionCatalogModel,
    _ExtensionRecord,
    _ExtensionSource,
    _source_loader_name_filters,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable


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
        QtCore.QStandardPaths.StandardLocation.GenericDataLocation
    )
    return pathlib.Path(location) / "ERLab" / "ImageTool Manager" / "extensions"


def _safe_extension_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-.")
    if not normalized:
        normalized = f"extension-{uuid.uuid4().hex[:8]}"
    return normalized


def _catalog_payload_v1(payload: object) -> object:
    """Normalize catalogs written by unreleased extension prototypes."""
    if not isinstance(payload, dict) or payload.get("schema_version", 1) not in {
        1,
        2,
        3,
        4,
    }:
        return payload
    migrated = dict(payload)
    migrated["schema_version"] = 1
    raw_extensions = migrated.get("extensions", {})
    if not isinstance(raw_extensions, dict):
        return migrated
    favorites = {
        tuple(value)
        for value in migrated.get("routine_favorites", ())
        if isinstance(value, (list, tuple)) and len(value) == 2
    }
    extensions: dict[str, object] = {}
    for extension_id, raw_record in raw_extensions.items():
        if not isinstance(extension_id, str) or not isinstance(raw_record, dict):
            extensions[extension_id] = raw_record
            continue
        record = dict(raw_record)
        removed = record.pop("removed", False)
        favorite = record.pop("favorite", False)
        record.pop("metadata", None)
        source_type = record.get("source_type", "script")
        if source_type == "environment-package":
            continue
        record.pop("source_type", None)
        if removed:
            continue
        revisions = record.pop("revisions", None)
        current_revision = record.pop("current_revision", None)
        if isinstance(revisions, dict):
            current = revisions.get(current_revision)
            if isinstance(current, dict):
                current = dict(current)
                current.pop("change_summary", None)
                if "registered_at" not in current and "created_at" in current:
                    current["registered_at"] = current.pop("created_at")
                record["source"] = current
                if favorite:
                    for routine in current.get("routines", ()):
                        if isinstance(routine, dict) and isinstance(
                            routine.get("id"), str
                        ):
                            favorites.add((extension_id, routine["id"]))
        source = record.get("source")
        if isinstance(source, dict):
            source = dict(source)
            legacy_validation_error = source.pop("import_error", None)
            if source.get("validation_error") is None and legacy_validation_error:
                source["validation_error"] = legacy_validation_error
            source.pop("change_summary", None)
            for field in (
                "routine_call_references",
                "loader_call_references",
                "entry_point_group",
                "entry_point_name",
                "entry_point_value",
                "distribution_name",
                "distribution_version",
                "editable",
                "loader_always_single",
                "loader_dialog_methods",
            ):
                source.pop(field, None)
            if "registered_at" not in source and "created_at" in source:
                source["registered_at"] = source.pop("created_at")
            record["source"] = source
        extensions[extension_id] = record
    migrated["extensions"] = extensions
    migrated["routine_favorites"] = tuple(
        sorted(favorite for favorite in favorites if favorite[0] in extensions)
    )
    return migrated


def _catalog_with_canonical_names(
    catalog: _ExtensionCatalogModel,
) -> _ExtensionCatalogModel:
    """Use source filenames as persisted visible names."""
    records = dict(catalog.extensions)
    changed = False
    for extension_id, record in catalog.extensions.items():
        canonical_name = (
            pathlib.Path(record.source.source_path).name
            if record.source.source_path
            else None
        )
        if canonical_name and canonical_name != record.name:
            records[extension_id] = record.model_copy(update={"name": canonical_name})
            changed = True
    return catalog.model_copy(update={"extensions": records}) if changed else catalog


class _ExtensionCatalogStore:
    """Own catalog locking, generation checks, and atomic commits.

    Every mutation re-reads the catalog while holding ``QLockFile``. A caller can
    merge an unrelated global change, but a changed ``record_generation`` rejects
    stale edits to the same extension.

    Only script registrations and user preferences cross the persistence boundary.
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
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            catalog = _ExtensionCatalogModel.model_validate(
                _catalog_payload_v1(payload)
            )
        except (OSError, ValueError) as error:
            raise _ExtensionCatalogError(
                f"Could not read the extension catalog: {error}"
            ) from error
        return _catalog_with_canonical_names(catalog)

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

    def clean_unreleased_catalog(self) -> None:
        """Remove prototype fields and package records from the persistent file."""
        if not self.path.exists():
            return
        lock = self._lock()
        try:
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                normalized = _catalog_with_canonical_names(
                    _ExtensionCatalogModel.model_validate(_catalog_payload_v1(payload))
                )
            except (OSError, ValueError) as error:
                raise _ExtensionCatalogError(
                    f"Could not read the extension catalog: {error}"
                ) from error
            if payload != normalized.model_dump(mode="json"):
                self._write_unlocked(normalized)
        finally:
            lock.unlock()

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
            updated = _catalog_with_canonical_names(callback(current))
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

    def recovery_source_path(self, extension_id: str, source_hash: str) -> pathlib.Path:
        """Return the recovery copy when it matches the registered source."""
        record = self.read().extensions.get(extension_id)
        if record is None or record.source.source_hash != source_hash:
            raise KeyError(f"Unknown extension source {extension_id}:{source_hash}")
        path = self.objects_directory / record.source.object_name
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def executable_source_path(
        self, extension_id: str, source_hash: str
    ) -> pathlib.Path:
        """Return a registered user file only when it matches the source hash.

        Managed source objects are recovery copies. They must not become an
        implicit execution location when a user-owned script is missing or changed.
        """
        record = self.read().extensions.get(extension_id)
        if record is None or record.source.source_hash != source_hash:
            raise KeyError(f"Unknown extension source {extension_id}:{source_hash}")
        if record.source.source_path is None:
            raise FileNotFoundError(
                f"Extension {extension_id}:{source_hash} has no registered script file"
            )
        path = pathlib.Path(record.source.source_path).expanduser().resolve()
        try:
            source = path.read_bytes()
        except OSError as error:
            raise FileNotFoundError(path) from error
        if hashlib.sha256(source).hexdigest() != source_hash:
            raise _ExtensionCatalogConflictError(
                f"Registered script {path} does not match source {source_hash}"
            )
        return path

    def script_capability_reference(
        self,
        extension_id: str,
        kind: str,
        capability_id: str,
    ) -> tuple[pathlib.Path, str]:
        """Return the current validated user path and public function name.

        Copied code follows the locally registered script. Runtime checks use
        :meth:`executable_source_path` to reject changed source.
        """
        record = self.read().extensions.get(extension_id)
        if record is None:
            raise KeyError(extension_id)
        source = record.source
        descriptors = (
            source.routines
            if kind == "routine"
            else source.loaders
            if kind == "loader"
            else ()
        )
        descriptor = next(
            (item for item in descriptors if item.id == capability_id), None
        )
        if descriptor is None or not source.approved:
            raise KeyError(capability_id)
        try:
            path = self.executable_source_path(extension_id, source.source_hash)
        except _ExtensionCatalogConflictError as error:
            raise FileNotFoundError(
                f"Registered script {extension_id!r} changed on disk"
            ) from error
        return path, descriptor.function_name

    def _store_script_source(self, source: bytes, source_hash: str) -> str:
        """Store verified bytes and atomically repair a corrupt source object."""
        if hashlib.sha256(source).hexdigest() != source_hash:
            raise ValueError("Extension source does not match its source hash")
        object_name = f"{source_hash}.py"
        self.objects_directory.mkdir(parents=True, exist_ok=True)
        object_path = self.objects_directory / object_name
        try:
            if hashlib.sha256(object_path.read_bytes()).hexdigest() == source_hash:
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
        approved: bool = False,
        expected_source_hash: str | None = None,
        expected_record_generation: int | None = None,
        check_record_generation: bool = False,
    ) -> tuple[_ExtensionCatalogModel, str, bool]:
        source_path = pathlib.Path(path).expanduser().resolve()
        source = source_path.read_bytes()
        source_hash = hashlib.sha256(source).hexdigest()
        if expected_source_hash is not None and source_hash != expected_source_hash:
            raise _ExtensionCatalogConflictError(
                "The script source changed after it was reviewed"
            )
        extension_id = _safe_extension_id(extension_id or source_path.stem)
        object_path = self.objects_directory / f"{source_hash}.py"
        object_existed = object_path.is_file()
        object_name = self._store_script_source(source, source_hash)
        modified_at = (
            datetime.datetime.fromtimestamp(source_path.stat().st_mtime)
            .astimezone()
            .isoformat(timespec="seconds")
        )
        now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        changed = False
        replaced_object_name: str | None = None

        def update(catalog: _ExtensionCatalogModel) -> _ExtensionCatalogModel:
            nonlocal changed, replaced_object_name
            records = dict(catalog.extensions)
            existing = records.get(extension_id)
            if existing is not None and source_hash == existing.source.source_hash:
                updated_source = existing.source.model_copy(
                    update={
                        "source_path": os.fspath(source_path),
                        "source_modified_at": modified_at,
                    }
                )
                if (
                    updated_source != existing.source
                    or existing.name != source_path.name
                ):
                    records[extension_id] = existing.model_copy(
                        update={
                            "name": source_path.name,
                            "source": updated_source,
                            "record_generation": existing.record_generation + 1,
                        }
                    )
                    return catalog.model_copy(update={"extensions": records})
                return catalog
            registered_source = _ExtensionSource(
                source_hash=source_hash,
                object_name=object_name,
                source_path=os.fspath(source_path),
                source_modified_at=modified_at,
                registered_at=now,
                approved=approved,
            )
            if existing is None:
                record = _ExtensionRecord(
                    id=extension_id,
                    name=source_path.name,
                    source=registered_source,
                    record_generation=1,
                )
            else:
                replaced_object_name = existing.source.object_name
                record = existing.model_copy(
                    update={
                        "name": source_path.name,
                        "enabled": False,
                        "source": registered_source,
                        "record_generation": existing.record_generation + 1,
                    }
                )
            records[extension_id] = record
            changed = True
            return catalog.model_copy(update={"extensions": records})

        try:
            catalog = self.mutate(
                extension_id,
                update,
                expected_record_generation=expected_record_generation,
                check_record_generation=check_record_generation,
            )
        except Exception:
            if not object_existed:
                try:
                    current = self.read()
                except _ExtensionCatalogError:
                    current = None
                referenced = current is not None and any(
                    record.source.object_name == object_name
                    for record in current.extensions.values()
                )
                if not referenced:
                    with contextlib.suppress(OSError):
                        object_path.unlink(missing_ok=True)
            raise
        if (
            replaced_object_name is not None
            and replaced_object_name != object_name
            and not any(
                record.source.object_name == replaced_object_name
                for record in catalog.extensions.values()
            )
        ):
            with contextlib.suppress(OSError):
                (self.objects_directory / replaced_object_name).unlink(missing_ok=True)
        return catalog, source_hash, changed

    def record_source_validation_failure(
        self,
        extension_id: str,
        *,
        source_hash: str,
        expected_record_generation: int,
        validation_error: str,
    ) -> _ExtensionCatalogModel:
        """Persist a source-validation failure without changing a newer source."""

        def update(catalog: _ExtensionCatalogModel) -> _ExtensionCatalogModel:
            current = catalog.extensions[extension_id]
            if source_hash != current.source.source_hash:
                raise _ExtensionCatalogConflictError(
                    f"Extension {extension_id!r} changed during validation"
                )
            source = current.source.model_copy(
                update={"validation_error": validation_error, "approved": False}
            )
            records = dict(catalog.extensions)
            records[extension_id] = current.model_copy(
                update={
                    "source": source,
                    "enabled": False,
                    "record_generation": current.record_generation + 1,
                }
            )
            return catalog.model_copy(update={"extensions": records})

        return self.mutate(
            extension_id,
            update,
            expected_record_generation=expected_record_generation,
        )

    def enable_validated_source(
        self,
        extension_id: str,
        *,
        source_hash: str,
        expected_record_generation: int,
        routines: tuple[RoutineDescriptor, ...],
        loaders: tuple[LoaderDescriptor, ...],
        enable_extension: bool = True,
    ) -> _ExtensionCatalogModel:
        """Commit descriptors produced by execution-layer validation."""

        def update(catalog: _ExtensionCatalogModel) -> _ExtensionCatalogModel:
            current = catalog.extensions[extension_id]
            if source_hash != current.source.source_hash:
                raise _ExtensionCatalogConflictError(
                    f"Extension {extension_id!r} changed during validation"
                )
            validated_source = current.source.model_copy(
                update={
                    "approved": True,
                    "routines": routines,
                    "loaders": loaders,
                    "validation_error": None,
                }
            )
            name_filters = _source_loader_name_filters(validated_source)
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
            for other in catalog.extensions.values():
                if other.id == extension_id or not other.enabled:
                    continue
                other_source = other.source
                conflicts = sorted(
                    candidate_filters.intersection(
                        _source_loader_name_filters(other_source)
                    )
                )
                if conflicts:
                    joined = ", ".join(repr(value) for value in conflicts)
                    raise _ExtensionCatalogConflictError(
                        f"Extension {extension_id!r} conflicts with enabled extension "
                        f"{other.id!r} for file dialog filters: {joined}"
                    )
            records = dict(catalog.extensions)
            records[extension_id] = current.model_copy(
                update={
                    "enabled": current.enabled or enable_extension,
                    "source": validated_source,
                    "record_generation": current.record_generation + 1,
                }
            )
            return catalog.model_copy(update={"extensions": records})

        return self.mutate(
            extension_id,
            update,
            expected_record_generation=expected_record_generation,
        )

    def resolve_capability(
        self,
        extension_id: str,
        source_hash: str,
        kind: str,
        capability_id: str,
    ) -> Callable[..., typing.Any]:
        """Resolve a capability from the current approved source."""
        record = self.read().extensions.get(extension_id)
        if record is None or source_hash != record.source.source_hash:
            raise KeyError(f"Unknown extension source {extension_id}:{source_hash}")
        source = record.source
        if not record.enabled:
            raise ExtensionNotFoundError(f"Extension {extension_id!r} is disabled")
        if not source.approved:
            raise ExtensionNotFoundError(
                f"Extension source {extension_id}:{source_hash} is not approved"
            )
        loaded = load_script(
            self.executable_source_path(extension_id, source_hash),
            expected_source_hash=source_hash,
        )
        entries = loaded.routines if kind == "routine" else loaded.loaders
        try:
            return entries[capability_id][1]
        except KeyError as error:
            raise KeyError(f"Unknown {kind} capability {capability_id!r}") from error

    def capability_status(
        self,
        extension_id: str,
        source_hash: str,
        kind: str,
        capability_id: str,
    ) -> _CapabilityStatus:
        """Resolve exact catalog state without importing extension code."""
        record = self.read().extensions.get(extension_id)
        if record is None or source_hash != record.source.source_hash:
            raise KeyError(f"Unknown extension source {extension_id}:{source_hash}")
        source = record.source
        try:
            self.executable_source_path(extension_id, source_hash)
        except (FileNotFoundError, KeyError):
            return "missing-source"
        except _ExtensionCatalogConflictError:
            return "hash-mismatch"
        if source.validation_error:
            return "validation-failed"
        if not source.approved:
            return "approval-required"
        descriptors = source.routines if kind == "routine" else source.loaders
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

    def source_available(self, record: _ExtensionRecord, source_hash: str) -> bool:
        """Check the registered source without importing extension code."""
        if record.source.source_hash != source_hash:
            return False
        try:
            self.executable_source_path(record.id, source_hash)
        except (FileNotFoundError, KeyError, _ExtensionCatalogConflictError):
            return False
        return True

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
            records = dict(current.extensions)
            del records[extension_id]
            referenced_objects = {
                remaining.source.object_name for remaining in records.values()
            }
            removable_objects = {record.source.object_name}.difference(
                referenced_objects
            )
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
        self.store.clean_unreleased_catalog()
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
        _set_source_resolver(self._resolver_owner, self.store.executable_source_path)
        _set_script_capability_reference_resolver(
            self._resolver_owner, self.store.script_capability_reference
        )
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
        if model == self.model:
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
