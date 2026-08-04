"""Long-lived HDF5 storage for one ImageTool Manager workspace."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import pathlib
import threading
import typing
import uuid
import weakref
from dataclasses import dataclass
from typing import Self

import erlab

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import h5py
else:
    import lazy_loader as _lazy

    h5py = _lazy.load("h5py")


_WORKSPACE_OBJECTS_GROUP = "__itws_objects"
_WORKSPACE_STAGING_GROUP = "__itws_staging"
_WORKSPACE_GENERATIONS_GROUP = "__itws_generations"
_WORKSPACE_MANIFEST_DATASET = "manifest"
_WORKSPACE_GENERATION_WIDTH = 20


@dataclass(frozen=True)
class _WorkspaceGeneration:
    """One validated committed workspace generation."""

    sequence: int
    manifest: dict[str, typing.Any]


class WorkspaceStoreConflictError(RuntimeError):
    """The path no longer identifies the file opened by the store."""


class WorkspaceStore:
    """Own the HDF5 handle used by one open workspace document.

    Payload objects are immutable. A generation becomes visible only after its
    completed staging group moves into the generations group.
    """

    _active: weakref.WeakValueDictionary[str, WorkspaceStore] = (
        weakref.WeakValueDictionary()
    )
    _active_lock = threading.RLock()

    def __init__(self, path: str | os.PathLike[str], *, create: bool = False) -> None:
        self._path = pathlib.Path(path).resolve()
        self._lock = threading.RLock()
        self._write_lock = threading.RLock()
        self._object_leases: dict[str, int] = {}
        self._readers: weakref.WeakSet[typing.Any] = weakref.WeakSet()
        self._closed = True
        self._handle_generation = 0
        self._h5_file: typing.Any = None
        self._path_identity: tuple[int, int] | None = None
        try:
            self._open(create=create)
            self._register()
        except Exception:
            self._close_handle()
            raise

    @staticmethod
    def _key(path: str | os.PathLike[str]) -> str:
        return os.path.normcase(str(pathlib.Path(path).resolve()))

    @classmethod
    def active(cls, path: str | os.PathLike[str]) -> WorkspaceStore | None:
        """Return the store that owns *path*, if this process has one."""
        with cls._active_lock:
            store = cls._active.get(cls._key(path))
        if store is None or store.closed:
            return None
        return store

    @property
    def path(self) -> pathlib.Path:
        return self._path

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    @property
    def write_lock(self) -> threading.RLock:
        """Serialize complete saves and maintenance operations."""
        return self._write_lock

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def h5_file(self) -> typing.Any:
        if self._closed:
            raise RuntimeError("Workspace store is closed")
        return self._h5_file

    @property
    def handle_generation(self) -> int:
        """Return a value that changes each time the HDF5 handle reopens."""
        return self._handle_generation

    def _register(self) -> None:
        key = self._key(self._path)
        with self._active_lock:
            existing = self._active.get(key)
            if existing is not None and existing is not self and not existing.closed:
                self._close_handle()
                raise RuntimeError(
                    f"Workspace already has an active store: {self._path}"
                )
            self._active[key] = self

    def _unregister(self, path: pathlib.Path | None = None) -> None:
        key = self._key(self._path if path is None else path)
        with self._active_lock:
            if self._active.get(key) is self:
                self._active.pop(key, None)

    def _open(self, *, create: bool) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if create:
            self._h5_file = h5py.File(
                self._path,
                "w",
                libver="latest",
                fs_strategy="fsm",
                fs_persist=True,
            )
            self._h5_file.attrs["imagetool_workspace_schema_version"] = 5
            self._h5_file.attrs["erlab_version"] = str(erlab.__version__)
            self._h5_file.require_group(_WORKSPACE_OBJECTS_GROUP)
            self._h5_file.require_group(_WORKSPACE_STAGING_GROUP)
            self._h5_file.require_group(_WORKSPACE_GENERATIONS_GROUP)
            self._h5_file.flush()
        else:
            self._h5_file = h5py.File(self._path, "r+")
        self._handle_generation += 1
        path_stat = self._path.stat()
        self._path_identity = (path_stat.st_dev, path_stat.st_ino)
        self._closed = False

    def _close_handle(self) -> None:
        for reader in tuple(self._readers):
            with contextlib.suppress(Exception):
                reader._close_store_wrapper()
        h5_file = self._h5_file
        self._h5_file = None
        if h5_file is not None:
            with contextlib.suppress(Exception):
                h5_file.close()
        self._closed = True

    def require_current_path(self) -> None:
        """Fail if the open handle is no longer the file at ``path``."""
        try:
            path_stat = self._path.stat()
        except OSError as exc:
            raise WorkspaceStoreConflictError(
                f"Open workspace path is no longer available: {self._path}"
            ) from exc
        if (path_stat.st_dev, path_stat.st_ino) != self._path_identity:
            raise WorkspaceStoreConflictError(
                f"Open workspace path now identifies another file: {self._path}"
            )

    def close(self) -> None:
        """Close the document handle and remove this store from the registry."""
        with self._write_lock, self._lock:
            self._unregister()
            self._close_handle()

    def switch_path(self, path: str | os.PathLike[str]) -> None:
        """Open *path* through this store without invalidating store references."""
        new_path = pathlib.Path(path).resolve()
        with self._write_lock, self._lock:
            old_path = self._path
            self._unregister(old_path)
            self._close_handle()
            self._path = new_path
            try:
                self._open(create=False)
                self._register()
            except Exception:
                self._path = old_path
                self._open(create=False)
                self._register()
                raise

    def reopen(self) -> None:
        """Close and reopen the current file through the same store object."""
        with self._write_lock, self._lock:
            self._close_handle()
            self._open(create=False)

    def replace_from(
        self,
        prepared_path: str | os.PathLike[str],
        replace: Callable[[pathlib.Path, pathlib.Path], None],
        *,
        before_close: Callable[[], None] | None = None,
    ) -> None:
        """Replace this document while its HDF5 handle is closed.

        The store object stays stable, so existing lazy arrays use the reopened
        file after the replacement.
        """
        source = pathlib.Path(prepared_path).resolve()
        with self._write_lock, self._lock:
            if before_close is not None:
                before_close()
            path_identity = self._path_identity
            self._close_handle()
            try:
                try:
                    path_stat = self._path.stat()
                except OSError as exc:
                    raise WorkspaceStoreConflictError(
                        f"Open workspace path is no longer available: {self._path}"
                    ) from exc
                if (path_stat.st_dev, path_stat.st_ino) != path_identity:
                    raise WorkspaceStoreConflictError(
                        f"Open workspace path now identifies another file: {self._path}"
                    )
                replace(source, self._path)
            finally:
                self._open(create=False)

    def acquire_object(self, object_id: str) -> None:
        """Keep a payload object reachable while a lazy array uses it."""
        with self._lock:
            self._object_leases[object_id] = self._object_leases.get(object_id, 0) + 1

    def register_reader(self, reader: object) -> None:
        """Register a lazy reader that must refresh when this store reopens."""
        with self._lock:
            self._readers.add(reader)

    def unregister_reader(self, reader: object) -> None:
        """Remove a reader that no longer uses this store."""
        with self._lock:
            self._readers.discard(reader)

    def release_object(self, object_id: str) -> None:
        """Release one lazy-array reference to a payload object."""
        with self._lock:
            count = self._object_leases.get(object_id, 0)
            if count <= 1:
                self._object_leases.pop(object_id, None)
            else:
                self._object_leases[object_id] = count - 1

    @property
    def leased_object_ids(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._object_leases)

    @property
    def leased_legacy_group_paths(self) -> frozenset[str]:
        """Return old-format payload groups that still have lazy readers."""
        with self._lock:
            paths = {
                path
                for reader in self._readers
                if (path := reader.legacy_group_path) not in {None, "/"}
            }
            return frozenset(
                path
                for path in paths
                if path.strip("/") in self.h5_file
                and isinstance(self.h5_file[path], h5py.Group)
                and any(
                    isinstance(item, h5py.Dataset)
                    for item in self.h5_file[path].values()
                )
            )

    @staticmethod
    def object_path(object_id: str) -> str:
        if not object_id or "/" in object_id:
            raise ValueError("Workspace object ID must be one path component")
        return f"/{_WORKSPACE_OBJECTS_GROUP}/{object_id}"

    @staticmethod
    def _manifest_text(manifest: Mapping[str, typing.Any]) -> str:
        return json.dumps(manifest, sort_keys=True, separators=(",", ":"))

    @classmethod
    def _write_manifest(
        cls,
        group: typing.Any,
        manifest: Mapping[str, typing.Any],
    ) -> None:
        text = cls._manifest_text(manifest)
        dataset = group.create_dataset(
            _WORKSPACE_MANIFEST_DATASET,
            data=text,
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        dataset.attrs["sha256"] = hashlib.sha256(text.encode()).hexdigest()

    @classmethod
    def _read_manifest(cls, group: typing.Any) -> dict[str, typing.Any]:
        if _WORKSPACE_MANIFEST_DATASET not in group:
            raise ValueError("Workspace generation has no manifest")
        dataset = group[_WORKSPACE_MANIFEST_DATASET]
        raw = dataset.asstr()[()]
        if not isinstance(raw, str):
            raise TypeError("Workspace generation manifest is not text")
        expected = dataset.attrs.get("sha256")
        if isinstance(expected, bytes):
            expected = expected.decode()
        actual = hashlib.sha256(raw.encode()).hexdigest()
        if expected != actual:
            raise ValueError("Workspace generation manifest checksum does not match")
        manifest = json.loads(raw)
        if not isinstance(manifest, dict):
            raise TypeError("Workspace generation manifest is not an object")
        if int(manifest.get("schema_version", 0)) != 5:
            raise ValueError("Workspace generation has an unsupported schema")
        nodes = manifest.get("nodes")
        if not isinstance(nodes, list):
            raise TypeError("Workspace generation nodes are not a list")
        for entry in nodes:
            if not isinstance(entry, dict):
                raise TypeError("Workspace generation node is not an object")
            object_id = entry.get("payload_object_id")
            if not isinstance(object_id, str):
                raise TypeError("Workspace generation node has no payload object")
            object_path = cls.object_path(object_id)
            if entry.get("payload_path") != object_path:
                raise ValueError("Workspace generation payload path is not canonical")
            if object_path.strip("/") not in group.file:
                raise ValueError("Workspace generation payload object is missing")
        return manifest

    def generations(self) -> tuple[_WorkspaceGeneration, ...]:
        """Return all valid committed generations in ascending order."""
        with self._lock:
            root = self.h5_file.get(_WORKSPACE_GENERATIONS_GROUP)
            if root is None:
                return ()
            generations: list[_WorkspaceGeneration] = []
            for name in sorted(root):
                if len(name) != _WORKSPACE_GENERATION_WIDTH or not name.isdigit():
                    continue
                with contextlib.suppress(Exception):
                    generations.append(
                        _WorkspaceGeneration(int(name), self._read_manifest(root[name]))
                    )
            return tuple(generations)

    def current_generation(self) -> _WorkspaceGeneration:
        """Return the newest valid committed generation."""
        generations = self.generations()
        if not generations:
            raise ValueError("Workspace has no committed generation")
        return generations[-1]

    def publish(self, manifest: Mapping[str, typing.Any]) -> _WorkspaceGeneration:
        """Publish one completed manifest as the newest generation."""
        with self._write_lock, self._lock:
            self.require_current_path()
            generation_root = self.h5_file.require_group(_WORKSPACE_GENERATIONS_GROUP)
            existing_sequences = [
                int(name)
                for name in generation_root
                if len(name) == _WORKSPACE_GENERATION_WIDTH and name.isdigit()
            ]
            sequence = 1 if not existing_sequences else max(existing_sequences) + 1
            staging_root = self.h5_file.require_group(_WORKSPACE_STAGING_GROUP)
            staging_name = uuid.uuid4().hex
            staging = staging_root.create_group(staging_name)
            generation_manifest = dict(manifest)
            generation_manifest["schema_version"] = 5
            generation_manifest.pop("generation", None)
            try:
                self._write_manifest(staging, generation_manifest)
                generation_manifest = self._read_manifest(staging)
                self.flush()
            except Exception:
                with contextlib.suppress(Exception):
                    del staging_root[staging_name]
                    self.flush()
                raise

            generation_name = f"{sequence:0{_WORKSPACE_GENERATION_WIDTH}d}"
            self.h5_file.move(
                f"/{_WORKSPACE_STAGING_GROUP}/{staging_name}",
                f"/{_WORKSPACE_GENERATIONS_GROUP}/{generation_name}",
            )
            self.h5_file.attrs["imagetool_workspace_schema_version"] = 5
            self.h5_file.attrs["erlab_version"] = str(erlab.__version__)
            self.flush(durable=True)
            return _WorkspaceGeneration(sequence, generation_manifest)

    def flush(self, *, durable: bool = False) -> None:
        """Flush HDF5 buffers and optionally ask the operating system to sync."""
        self.h5_file.flush()
        if not durable:
            return
        with contextlib.suppress(Exception):
            handle = self.h5_file.id.get_vfd_handle()
            if isinstance(handle, tuple):
                handle = handle[0]
            os.fsync(int(handle))

    @staticmethod
    def manifest_object_ids(manifest: Mapping[str, typing.Any]) -> frozenset[str]:
        """Return payload object IDs referenced by a manifest."""
        object_ids: set[str] = set()
        nodes = manifest.get("nodes", ())
        if not isinstance(nodes, list):
            return frozenset()
        for entry in nodes:
            if not isinstance(entry, dict):
                continue
            object_id = entry.get("payload_object_id")
            if isinstance(object_id, str) and object_id:
                object_ids.add(object_id)
        return frozenset(object_ids)

    def collect_garbage(self, *, max_objects: int = 1) -> bool:
        """Retire old generations and unlink a bounded number of dead objects.

        Returns ``True`` when more obsolete objects remain.
        """
        if max_objects < 1:
            raise ValueError("max_objects must be positive")
        with self._write_lock, self._lock:
            self.require_current_path()
            generations = self.generations()
            retained = generations[-2:]
            retained_names = {
                f"{generation.sequence:0{_WORKSPACE_GENERATION_WIDTH}d}"
                for generation in retained
            }
            generation_root = self.h5_file.require_group(_WORKSPACE_GENERATIONS_GROUP)
            for name in list(generation_root):
                if name in retained_names:
                    continue
                del generation_root[name]

            reachable = set(self._object_leases)
            for generation in retained:
                reachable.update(self.manifest_object_ids(generation.manifest))

            object_root = self.h5_file.require_group(_WORKSPACE_OBJECTS_GROUP)
            obsolete = [name for name in object_root if name not in reachable]
            for name in obsolete[:max_objects]:
                del object_root[name]
            self.h5_file.flush()
            return len(obsolete) > max_objects

    def clear_staging(self) -> None:
        """Remove unpublished staging groups left by interrupted saves."""
        with self._write_lock, self._lock:
            staging_root = self.h5_file.require_group(_WORKSPACE_STAGING_GROUP)
            for name in list(staging_root):
                del staging_root[name]
            self.h5_file.flush()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
