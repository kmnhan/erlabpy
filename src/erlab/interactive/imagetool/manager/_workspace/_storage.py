"""Immutable workspace generations and legacy transaction recovery."""

from __future__ import annotations

import contextlib
import ctypes
import errno
import functools
import json
import logging
import os
import pathlib
import stat
import sys
import typing
import uuid
from dataclasses import dataclass

import numpy as np
from qtpy import QtCore

import erlab.interactive.imagetool.manager._workspace._arrays as workspace_arrays
import erlab.interactive.imagetool.manager._workspace._file_copy as workspace_file_copy
import erlab.interactive.imagetool.manager._workspace._format as workspace_format
import erlab.interactive.imagetool.manager._workspace._store as workspace_store
from erlab.interactive import _persistence_constants
from erlab.interactive.imagetool.manager._workspace._format import (
    _workspace_file_is_workspace,
    _workspace_path_is_itws,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping

    import h5py
    import xarray as xr

    from erlab.interactive._options.schema import WorkspaceCompressionMode
else:
    import lazy_loader as _lazy

    h5py = _lazy.load("h5py")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _WorkspaceObjectWrite:
    """One immutable payload object required by a generation."""

    object_id: str
    dataset: xr.Dataset | None = None
    source_file: str | None = None
    source_path: str | None = None
    blob: bytes | None = None
    blob_kind: str | None = None


@dataclass(frozen=True)
class _WorkspaceGroupCopy:
    """One live old-format group that a new document must preserve."""

    source_file: str
    source_path: str
    target_path: str


@dataclass(frozen=True)
class _WorkspaceGenerationPlan:
    """A complete manifest and the new objects that it references.

    ``legacy_object_links`` identifies unchanged legacy payloads that can become
    immutable objects through a hard link during an in-place format upgrade.
    ``legacy_reader_rebindings`` contains only legacy paths whose array values
    match the target object. Save As can hard-link these paths and retarget their
    live readers after publication without changing a Dask graph's result.
    """

    manifest: dict[str, typing.Any]
    objects: tuple[_WorkspaceObjectWrite, ...]
    preserved_groups: tuple[_WorkspaceGroupCopy, ...] = ()
    legacy_object_links: tuple[tuple[str, str], ...] = ()
    legacy_reader_rebindings: tuple[tuple[str, str], ...] = ()


def _workspace_object_copy_source(item: _WorkspaceObjectWrite) -> tuple[str, str]:
    if item.source_file is None or item.source_path is None:
        raise ValueError(f"Workspace object {item.object_id!r} has no source")
    return item.source_file, item.source_path


@contextlib.contextmanager
def _workspace_generation_copy_source(
    path: str | os.PathLike[str],
) -> Iterator[typing.Any]:
    active_store = workspace_store.WorkspaceStore.active(path)
    if active_store is not None:
        with active_store.read_session() as h5_file:
            yield h5_file
        return
    workspace_arrays.ensure_workspace_hdf5_filters_registered()
    with workspace_arrays._workspace_file_lock(path), h5py.File(path, "r") as h5_file:
        yield h5_file


def _existing_workspace_object_ids(
    path: str | os.PathLike[str], object_ids: Iterable[str]
) -> frozenset[str]:
    """Return only immutable objects that can be copied from a workspace.

    A degraded document can retain a manifest reference whose object was already
    missing. Save As must preserve that unresolved reference without attempting a
    copy that would abort the complete save.
    """
    candidates = frozenset(object_ids)
    if not candidates:
        return frozenset()
    with _workspace_generation_copy_source(path) as h5_file:
        return frozenset(
            object_id
            for object_id in candidates
            if workspace_store.WorkspaceStore.object_path(object_id).strip("/")
            in h5_file
        )


def _workspace_reused_payload_allows_byte_copy(
    h5_file: typing.Any, object_ids: Iterable[str], source_size: int
) -> bool:
    """Allow bulk copying only when it adds little unused data.

    Count allocated compressed bytes, without reading dataset values. A small
    metadata allowance covers ordinary tiny workspaces. Larger files must retain
    at least half their bytes to justify copying the complete source.
    """
    if source_size <= 0:
        return False
    stored_bytes = 0
    seen: set[int] = set()
    inspected = 0
    try:
        root_name = _persistence_constants.WORKSPACE_OBJECTS_GROUP
        if not isinstance(h5_file.get(root_name, getlink=True), h5py.HardLink):
            return False
        objects = h5_file[root_name]
        pending = []
        for object_id in object_ids:
            pending.append((objects, object_id))
            if len(pending) > 4096:
                return False
        while pending:
            parent, name = pending.pop()
            inspected += 1
            if inspected > 4096 or not isinstance(
                parent.get(name, getlink=True), h5py.HardLink
            ):
                return False
            item = parent[name]
            address = int(h5py.h5o.get_info(item.id).addr)
            if address in seen:
                continue
            seen.add(address)
            if isinstance(item, h5py.Dataset):
                if item.is_virtual or item.external:
                    return False
                stored_bytes += int(item.id.get_storage_size())
            elif isinstance(item, h5py.Group):
                for child_name in item:
                    pending.append((item, child_name))
                    if inspected + len(pending) > 4096:
                        return False
            else:
                return False
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return False
    return stored_bytes * 2 >= source_size or source_size - stored_bytes <= 1 << 20


def _seed_workspace_save_as(
    target_path: str | os.PathLike[str],
    plan: _WorkspaceGenerationPlan,
    *,
    on_contention: Callable[[], None] | None = None,
) -> bool:
    """Copy a stable immutable document before writing a new generation.

    Only unchanged objects from one document are retained. New dataset and blob
    writes still go through the generation writer, even if their IDs already
    exist in the copied document. Legacy and mixed-source plans keep the group
    copy path.
    """
    target = pathlib.Path(target_path).resolve()
    if target.exists() or (
        plan.preserved_groups
        or plan.legacy_object_links
        or plan.legacy_reader_rebindings
    ):
        return False
    object_ids = {item.object_id for item in plan.objects}
    if len(object_ids) != len(plan.objects):
        return False
    reused = tuple(
        item for item in plan.objects if item.dataset is None and item.blob is None
    )
    if not reused or any(
        item.source_file is None
        or item.source_path is None
        or f"/{item.source_path.strip('/')}"
        != workspace_store.WorkspaceStore.object_path(item.object_id)
        for item in reused
    ):
        return False
    source_paths = {
        pathlib.Path(typing.cast("str", item.source_file)).resolve() for item in reused
    }
    if len(source_paths) != 1:
        return False
    source = next(iter(source_paths))
    if source == target:
        return False
    source_store = workspace_store.WorkspaceStore.active(source)
    source_lock = (
        source_store.write_lock
        if source_store is not None
        else workspace_arrays._workspace_file_lock(source)
    )
    copied = False
    try:
        with source_lock, contextlib.ExitStack() as source_access:
            if source_store is not None:
                if (
                    source_store.closed
                    or source_store.path != source
                    or source_store.conflicted
                    or source_store.recovery_path is not None
                    or source_store.write_in_progress
                    or not source_store.locking_supported
                ):
                    return False
                try:
                    source_store.require_current_path()
                except workspace_store.WorkspaceStoreConflictError:
                    # Group copying can still use the retained recovery handle.
                    return False
            expected_source = _workspace_publication_state(source)
            source_h5 = source_access.enter_context(
                _workspace_generation_copy_source(source)
            )
            roots = {
                _persistence_constants.WORKSPACE_OBJECTS_GROUP,
                _persistence_constants.WORKSPACE_GENERATIONS_GROUP,
                _persistence_constants.WORKSPACE_STAGING_GROUP,
            }
            if (
                set(source_h5) != roots
                or not workspace_format._workspace_schema_uses_immutable_generations(
                    int(source_h5.attrs.get("imagetool_workspace_schema_version", 0))
                )
                or workspace_store.WorkspaceStore._workspace_id_from_file(source_h5)
                is None
            ):
                return False
            try:
                _, source_manifest = (
                    workspace_format._workspace_file_metadata_from_attrs(
                        workspace_arrays._read_workspace_root_attrs_h5py(source)
                    )
                )
            except (TypeError, ValueError):
                return False
            if source_manifest is None:
                return False
            source_nodes = tuple(
                workspace_format._iter_workspace_manifest_node_entries(source_manifest)
            )
            target_nodes = tuple(
                workspace_format._iter_workspace_manifest_node_entries(plan.manifest)
            )
            source_uids = {
                entry["uid"]
                for entry in source_nodes
                if isinstance(entry.get("uid"), str)
            }
            target_uids = {
                entry["uid"]
                for entry in target_nodes
                if isinstance(entry.get("uid"), str)
            }
            if (
                len(source_uids) != len(source_nodes)
                or len(target_uids) != len(target_nodes)
                or not source_uids.issubset(target_uids)
            ):
                # Exporting fewer rows should still copy only the selected payloads.
                return False
            keep_ids = {item.object_id for item in reused}
            if any(
                not isinstance(
                    source_h5.get(
                        workspace_store.WorkspaceStore.object_path(object_id)
                    ),
                    h5py.Group,
                )
                for object_id in keep_ids
            ):
                return False
            _require_workspace_publication_state(source, expected_source)
            if source_store is not None:
                try:
                    source_store.require_current_path()
                except workspace_store.WorkspaceStoreConflictError as exc:
                    raise _WorkspacePublicationConflictError(source) from exc
                # The write lock prevents saves, collection and handle replacement.
                # Release the read lock so lazy slices can continue during copying.
                source_access.close()
            target.parent.mkdir(parents=True, exist_ok=True)
            for allow_byte_copy in (False, True):
                if allow_byte_copy:
                    with _workspace_generation_copy_source(source) as estimate_h5:
                        worthwhile = _workspace_reused_payload_allows_byte_copy(
                            estimate_h5, keep_ids, expected_source[3]
                        )
                    _require_workspace_publication_state(source, expected_source)
                    if not worthwhile:
                        return False
                try:
                    copied = workspace_store._retry_file_access(
                        functools.partial(
                            workspace_file_copy.copy_workspace_file,
                            source,
                            target,
                            allow_byte_copy=allow_byte_copy,
                        ),
                        on_wait=on_contention,
                    )
                except FileExistsError:
                    raise
                except OSError:
                    # A partial file must not become the group writer's input.
                    # Cleanup errors remain fatal, including read-only targets.
                    workspace_file_copy._remove_incomplete_copy(target)
                    _require_workspace_publication_state(source, expected_source)
                    logger.debug(
                        "Could not copy workspace file; using group copying",
                        exc_info=True,
                    )
                    return False
                except BaseException:
                    with contextlib.suppress(OSError):
                        workspace_file_copy._remove_incomplete_copy(target)
                    raise
                _require_workspace_publication_state(source, expected_source)
                if copied:
                    break
            if not copied:
                return False

        with workspace_store._wait_for_hdf5_access(
            lambda: h5py.File(target, "r+", locking="best-effort"),
            on_wait=on_contention,
        ) as target_h5:
            objects = target_h5[_persistence_constants.WORKSPACE_OBJECTS_GROUP]
            for object_id in list(objects):
                if object_id not in keep_ids:
                    del objects[object_id]
            for root in (
                _persistence_constants.WORKSPACE_GENERATIONS_GROUP,
                _persistence_constants.WORKSPACE_STAGING_GROUP,
            ):
                for name in list(target_h5[root]):
                    del target_h5[root][name]
            target_h5.attrs[_persistence_constants.WORKSPACE_ID_ATTR] = uuid.uuid4().hex
            target_h5.flush()
    except BaseException as exc:
        if copied:
            with contextlib.suppress(OSError):
                target.unlink(missing_ok=True)
        if isinstance(exc, FileNotFoundError) and not source.exists():
            raise _WorkspaceBackingFileNotFoundError(source) from exc
        raise
    else:
        return True


def _copy_workspace_group(
    target_store: workspace_store.WorkspaceStore,
    *,
    source_file: str,
    source_path: str,
    target_path: str,
) -> None:
    try:
        with (
            _workspace_generation_copy_source(source_file) as source_h5_file,
            target_store.lock,
        ):
            copied = workspace_arrays._copy_workspace_h5_group_to_open_file(
                source_h5_file,
                target_store.h5_file,
                source_path,
                target_path,
                None,
            )
    except Exception as exc:
        with target_store.lock:
            workspace_arrays._delete_h5_path(target_store.h5_file, target_path)
            target_store.h5_file.flush()
        if isinstance(exc, FileNotFoundError):
            raise _WorkspaceBackingFileNotFoundError(source_file) from exc
        raise
    if not copied:
        with target_store.lock:
            workspace_arrays._delete_h5_path(target_store.h5_file, target_path)
            target_store.h5_file.flush()
        raise KeyError(
            f"Workspace payload group {source_path!r} is missing from {source_file!r}"
        )


def _link_workspace_group(
    target_store: workspace_store.WorkspaceStore,
    *,
    source_path: str,
    target_path: str,
) -> bool:
    """Create one HDF5 hard link and return whether it was created."""
    with target_store.lock:
        if target_path.strip("/") in target_store.h5_file:
            return False
        if source_path.strip("/") not in target_store.h5_file:
            raise KeyError(f"Workspace payload group {source_path!r} is missing")
        parent = workspace_arrays._ensure_h5_parent_group(
            target_store.h5_file, target_path
        )
        name = target_path.rstrip("/").rsplit("/", maxsplit=1)[-1]
        parent[name] = target_store.h5_file[source_path]
        return True


def _write_workspace_generation(
    target_store: workspace_store.WorkspaceStore,
    plan: _WorkspaceGenerationPlan,
    *,
    compression_mode: WorkspaceCompressionMode,
    on_contention: Callable[[], None] | None = None,
) -> workspace_store._WorkspaceGeneration:
    """Write new immutable objects and publish one generation."""
    with target_store.write_session(on_contention=on_contention):
        created_object_ids: list[str] = []
        created_legacy_links: list[str] = []
        try:
            with target_store.lock:
                target_store.require_current_path()
            legacy_object_links = {
                object_id: legacy_path
                for legacy_path, object_id in plan.legacy_object_links
            }
            legacy_reader_rebindings = dict(plan.legacy_reader_rebindings)
            for item in plan.preserved_groups:
                if item.target_path in legacy_reader_rebindings:
                    continue
                with target_store.lock:
                    if item.target_path.strip("/") in target_store.h5_file:
                        continue
                _copy_workspace_group(
                    target_store,
                    source_file=item.source_file,
                    source_path=item.source_path,
                    target_path=item.target_path,
                )
            for item in plan.objects:
                target_path = target_store.object_path(item.object_id)
                with target_store.lock:
                    if target_path.strip("/") in target_store.h5_file:
                        continue
                legacy_path = legacy_object_links.get(item.object_id)
                if legacy_path is not None:
                    if _link_workspace_group(
                        target_store,
                        source_path=legacy_path,
                        target_path=target_path,
                    ):
                        created_object_ids.append(item.object_id)
                    continue
                created_object_ids.append(item.object_id)
                if item.dataset is not None:
                    if workspace_arrays._workspace_dataset_can_write_h5py(item.dataset):
                        with target_store.lock:
                            workspace_arrays._write_workspace_dataset_group_to_file(
                                target_store.h5_file,
                                target_path,
                                item.dataset,
                                compression_mode=compression_mode,
                            )
                    else:
                        workspace_arrays._write_workspace_dataset_group_to_file(
                            target_store.h5_file,
                            target_path,
                            item.dataset,
                            compression_mode=compression_mode,
                        )
                    continue
                if item.blob is not None:
                    with target_store.lock:
                        group = target_store.h5_file.require_group(target_path)
                        if item.blob_kind is not None:
                            group.attrs["erlab_object_kind"] = item.blob_kind
                        group.create_dataset(
                            "source",
                            data=np.frombuffer(item.blob, dtype=np.uint8),
                        )
                    continue
                source_file, source_path = _workspace_object_copy_source(item)
                _copy_workspace_group(
                    target_store,
                    source_file=source_file,
                    source_path=source_path,
                    target_path=target_path,
                )
            for item in plan.preserved_groups:
                object_id = legacy_reader_rebindings.get(item.target_path)
                if object_id is None:
                    continue
                object_path = target_store.object_path(object_id)
                if _link_workspace_group(
                    target_store,
                    source_path=object_path,
                    target_path=item.target_path,
                ):
                    created_legacy_links.append(item.target_path)
            return target_store.publish(plan.manifest)
        except Exception:
            with contextlib.suppress(Exception), target_store.lock:
                for group_path in created_legacy_links:
                    workspace_arrays._delete_h5_path(
                        target_store.h5_file,
                        group_path,
                    )
                referenced_object_ids: set[str] = set()
                for generation in target_store.generations():
                    referenced_object_ids.update(
                        target_store.manifest_object_ids(generation.manifest)
                    )
                for object_id in created_object_ids:
                    if object_id not in referenced_object_ids:
                        workspace_arrays._delete_h5_path(
                            target_store.h5_file,
                            target_store.object_path(object_id),
                        )
                target_store.h5_file.flush()
            raise


def _read_workspace_blob(
    path: str | os.PathLike[str], object_id: str
) -> tuple[bytes, str | None]:
    """Read one immutable binary object without executing its contents."""
    object_path = workspace_store.WorkspaceStore.object_path(object_id)
    with _workspace_generation_copy_source(path) as h5_file:
        group = h5_file[object_path]
        raw = group["source"][()]
        kind = group.attrs.get("erlab_object_kind")
        if isinstance(kind, bytes):
            kind = kind.decode()
        return bytes(raw), kind if isinstance(kind, str) else None


def _workspace_dataset_values_equal(
    left: typing.Any,
    right: typing.Any,
    *,
    on_mismatch: Callable[[tuple[int, ...], np.ndarray], None] | None = None,
) -> bool:
    """Compare bounded blocks and optionally report one copied scalar mismatch."""
    if left.shape != right.shape or left.dtype != right.dtype:
        return False

    def _values_equal(left_values: np.ndarray, right_values: np.ndarray) -> bool:
        try:
            return bool(np.array_equal(left_values, right_values, equal_nan=True))
        except TypeError:
            return bool(np.array_equal(left_values, right_values))

    def _report_mismatch(
        left_values: np.ndarray,
        right_values: np.ndarray,
        offsets: tuple[int, ...],
    ) -> None:
        if on_mismatch is None:
            return
        left_values = np.asarray(left_values, dtype=left.dtype)
        right_values = np.asarray(right_values, dtype=right.dtype)
        if left_values.dtype.kind not in "biufcSU":
            return
        equal = np.equal(left_values, right_values)
        if left_values.dtype.kind in "fc":
            # array_equal treats complex values as equal if either component is
            # NaN in both values, including NaNs in different components.
            equal |= np.isnan(left_values) & np.isnan(right_values)
        if bool(np.all(equal)):
            return
        local_index = tuple(
            int(index) for index in np.unravel_index(int(np.argmin(equal)), equal.shape)
        )
        stored_scalar = np.array(
            left_values[local_index], dtype=left_values.dtype, copy=True
        )
        current_scalar = np.asarray(right_values[local_index])
        if _values_equal(stored_scalar, current_scalar):
            return
        on_mismatch(
            tuple(
                offset + index
                for offset, index in zip(offsets, local_index, strict=True)
            ),
            stored_scalar,
        )

    if not left.shape:
        left_values, right_values = left[()], right[()]
        equal = _values_equal(left_values, right_values)
        if not equal and on_mismatch is not None:
            _report_mismatch(left_values, right_values, ())
        return equal
    if any(size == 0 for size in left.shape):
        return True
    remaining_items = max(1, (16 * 1024 * 1024) // max(1, left.dtype.itemsize))
    reversed_block_shape: list[int] = []
    for size in reversed(left.shape):
        block_size = min(int(size), remaining_items)
        reversed_block_shape.append(block_size)
        remaining_items = max(1, remaining_items // block_size)
    block_shape = tuple(reversed(reversed_block_shape))
    block_counts = tuple(
        (int(size) + block_size - 1) // block_size
        for size, block_size in zip(left.shape, block_shape, strict=True)
    )
    for block_index in np.ndindex(block_counts):
        selection = tuple(
            slice(
                index * block_size,
                min((index + 1) * block_size, int(size)),
            )
            for index, block_size, size in zip(
                block_index, block_shape, left.shape, strict=True
            )
        )
        left_values, right_values = left[selection], right[selection]
        if not _values_equal(left_values, right_values):
            if on_mismatch is not None:
                _report_mismatch(
                    left_values,
                    right_values,
                    tuple(part.start for part in selection),
                )
            return False
    return True


def _workspace_groups_have_equal_datasets(
    left: typing.Any,
    right: typing.Any,
) -> bool:
    if left.id == right.id:
        return True
    left_datasets: dict[str, typing.Any] = {}
    right_datasets: dict[str, typing.Any] = {}

    def _collect_dataset(
        datasets: dict[str, typing.Any], name: str, item: typing.Any
    ) -> None:
        if isinstance(item, h5py.Dataset):
            datasets[name] = item

    left.visititems(lambda name, item: _collect_dataset(left_datasets, name, item))
    right.visititems(lambda name, item: _collect_dataset(right_datasets, name, item))
    if left_datasets.keys() != right_datasets.keys():
        return False
    return all(
        _workspace_dataset_values_equal(left_datasets[name], right_datasets[name])
        for name in left_datasets
    )


def _rebind_equivalent_legacy_readers(
    store: workspace_store.WorkspaceStore,
    manifest: Mapping[str, typing.Any],
) -> None:
    """Move live readers to equal immutable objects before compaction.

    Schema conversion temporarily preserves old payload paths for live xarray
    readers. Content comparison prevents a derived Dask graph from being retargeted
    to a different saved result.
    """
    legacy_reader_rebindings = (
        workspace_format._workspace_manifest_legacy_reader_rebindings(manifest)
    )
    legacy_paths = store.leased_legacy_group_paths
    if not legacy_reader_rebindings or not legacy_paths:
        return
    equivalent: dict[str, str] = {}
    with store.write_lock, store.lock, store.read_session() as h5_file:
        for legacy_path in legacy_paths:
            object_id = legacy_reader_rebindings.get(legacy_path)
            if object_id is None:
                continue
            object_path = store.object_path(object_id)
            if (
                legacy_path.strip("/") in h5_file
                and object_path.strip("/") in h5_file
                and _workspace_groups_have_equal_datasets(
                    h5_file[legacy_path], h5_file[object_path]
                )
            ):
                equivalent[legacy_path] = object_id
        store.rebind_legacy_readers(equivalent)


def _compact_workspace_store(
    store: workspace_store.WorkspaceStore,
    *,
    discard_serialized_reader_pins: (
        workspace_store._SerializedReaderPinSnapshot | None
    ) = None,
) -> None:
    """Rewrite a workspace while omitting confirmed obsolete reader pins."""
    workspace_path = store.path
    prepared_path = workspace_path.with_name(
        f".{workspace_path.name}.compact-{uuid.uuid4().hex}"
    )
    try:
        with store.write_lock, store.lock:
            current = store.current_generation()
            baseline_manifest = dict(current.manifest)
            baseline_manifest.pop("delta_save_count", None)
            baseline_manifest.pop("estimated_obsolete_bytes", None)
            baseline_manifest.pop("replacement_delta_count", None)
            baseline_manifest.pop("repack_estimate_known", None)
            object_ids = set(store.manifest_object_ids(baseline_manifest))
            node_object_ids = set(store.manifest_node_object_ids(baseline_manifest))
            extension_object_ids = set(
                store.manifest_extension_object_ids(baseline_manifest)
            )
            object_ids.update(store.leased_object_ids)
            serialized_pins = store.serialized_reader_pin_snapshot()
            discarded_pins = (
                discard_serialized_reader_pins
                or workspace_store._SerializedReaderPinSnapshot({}, {})
            )
            object_ids.update(
                object_id
                for object_id, version in serialized_pins.object_versions.items()
                if version > discarded_pins.object_versions.get(object_id, -1)
            )
            optional_missing_script_source_ids = (
                extension_object_ids
                - node_object_ids
                - set(serialized_pins.object_versions)
            )
            legacy_group_paths = store.leased_legacy_group_paths | {
                group_path
                for group_path, version in serialized_pins.legacy_group_versions.items()
                if version > discarded_pins.legacy_group_versions.get(group_path, -1)
            }
            expected_state = _workspace_publication_state(workspace_path)

            with (
                workspace_store.WorkspaceStore(
                    prepared_path,
                    create=True,
                    workspace_id=store.workspace_id,
                ) as compacted,
                compacted.write_session(),
            ):
                with compacted.lock:
                    for group_path in sorted(legacy_group_paths):
                        copied = workspace_arrays._copy_workspace_h5_group_to_open_file(
                            store.h5_file,
                            compacted.h5_file,
                            group_path,
                            group_path,
                            None,
                        )
                        if not copied:
                            raise KeyError(
                                f"Workspace payload group {group_path!r} is missing"
                            )
                    for object_id in sorted(object_ids):
                        object_path = store.object_path(object_id)
                        copied = workspace_arrays._copy_workspace_h5_group_to_open_file(
                            store.h5_file,
                            compacted.h5_file,
                            object_path,
                            object_path,
                            None,
                        )
                        if not copied:
                            if object_id in optional_missing_script_source_ids:
                                continue
                            raise KeyError(f"Workspace object {object_id!r} is missing")
                compacted.publish(baseline_manifest)
                compacted.publish(baseline_manifest)
                generations = compacted.generations()
                if (
                    len(generations) != 2
                    or generations[0].manifest != generations[1].manifest
                ):
                    raise RuntimeError(
                        "Compacted workspace baseline generations do not match"
                    )
                if any(
                    compacted.object_path(object_id).strip("/") not in compacted.h5_file
                    for object_id in object_ids - optional_missing_script_source_ids
                ):
                    raise RuntimeError(
                        "Compacted workspace is missing a payload object"
                    )
                compacted.flush(durable=True)

            expected_identity = expected_state[1:3]

            def _publish_compacted_workspace(
                source: pathlib.Path, destination: pathlib.Path
            ) -> None:
                closed_state = _workspace_publication_state(destination)
                if not closed_state[0] or closed_state[1:3] != expected_identity:
                    raise _WorkspacePublicationConflictError(destination)
                _replace_workspace_file(
                    source,
                    destination,
                    expected_state=closed_state,
                )

            store.replace_from(
                prepared_path,
                _publish_compacted_workspace,
                before_close=lambda: _require_workspace_publication_state(
                    workspace_path, expected_state
                ),
            )
    finally:
        if store.recovery_path != prepared_path:
            with contextlib.suppress(OSError):
                prepared_path.unlink()


_WorkspacePublicationState: typing.TypeAlias = tuple[bool, int, int, int, int]


class _WorkspaceBackingFileNotFoundError(FileNotFoundError):
    """A workspace payload source disappeared before it could be copied."""

    def __init__(self, source_path: str | os.PathLike[str]) -> None:
        self.source_path = os.fsdecode(source_path)
        super().__init__(
            errno.ENOENT,
            "Workspace backing file is missing",
            self.source_path,
        )


class _WorkspacePublicationConflictError(workspace_store.WorkspaceStoreConflictError):
    """A published workspace changed while a save was in progress."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = os.fsdecode(path)
        super().__init__(
            "Workspace changed outside ImageTool Manager while it was being saved: "
            f"{self.path}"
        )


def _workspace_publication_state(
    path: str | os.PathLike[str],
) -> _WorkspacePublicationState:
    """Return content-sensitive state used to detect an external file change.

    The state excludes ``ctime`` because cloud clients can change it when they
    update extended attributes without changing the workspace. File identity,
    size, and modification time still reject replacement and content changes.
    """
    try:
        stat_result = os.stat(path)
    except FileNotFoundError:
        return False, 0, 0, 0, 0
    return (
        True,
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
    )


def _replace_workspace_file(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    expected_state: _WorkspacePublicationState | None,
) -> None:
    """Publish a prepared workspace with bounded file-access retries."""

    def _replace() -> None:
        if (
            expected_state is not None
            and _workspace_publication_state(destination) != expected_state
        ):
            raise _WorkspacePublicationConflictError(destination)
        os.replace(source, destination)

    workspace_store._retry_file_access(_replace)
    _fsync_parent_directory(destination)


def _fsync_parent_directory(path: str | os.PathLike[str]) -> None:
    """Ask POSIX to persist a completed directory entry replacement."""
    if os.name != "posix":
        return
    with contextlib.suppress(OSError):
        file_descriptor = os.open(pathlib.Path(path).parent, os.O_RDONLY)
        try:
            os.fsync(file_descriptor)
        finally:
            os.close(file_descriptor)


def _require_workspace_publication_state(
    path: str | os.PathLike[str], expected_state: _WorkspacePublicationState
) -> None:
    if _workspace_publication_state(path) != expected_state:
        raise _WorkspacePublicationConflictError(path)


@dataclass(frozen=True)
class _WorkspaceDocumentLockInfo:
    path: str
    owner: str
    hostname: str
    appname: str
    pid: int | None


@contextlib.contextmanager
def _open_workspace_h5_file_for_update(
    fname: str | os.PathLike[str],
) -> Iterator[typing.Any]:
    with workspace_arrays._workspace_file_lock(fname), h5py.File(fname, "a") as h5_file:
        yield h5_file


def _iter_exception_chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _is_workspace_file_lock_error(exc: BaseException) -> bool:
    lock_errnos = {errno.EACCES, errno.EAGAIN}
    if hasattr(errno, "EWOULDBLOCK"):
        lock_errnos.add(errno.EWOULDBLOCK)

    for err in _iter_exception_chain(exc):
        if isinstance(err, BlockingIOError):
            return True
        message = str(err).lower()
        if not any(
            marker in message
            for marker in (
                "unable to lock file",
                "resource temporarily unavailable",
                "file is already open",
            )
        ):
            continue
        err_no = getattr(err, "errno", None)
        if err_no in lock_errnos or "unable to lock file" in message:
            return True
        if "file is already open" in message:
            return True
    return False


def _workspace_lock_path(fname: str | os.PathLike[str]) -> str:
    workspace_path = pathlib.Path(fname).resolve()
    return str(workspace_path.with_name(f".{workspace_path.name}.lock"))


def _hide_workspace_lock_file(lock_path: str) -> None:
    if sys.platform == "darwin":
        with contextlib.suppress(AttributeError, OSError):
            if not stat.S_ISREG(os.lstat(lock_path).st_mode):
                return
            os.chflags(lock_path, stat.UF_HIDDEN)
        return
    if os.name != "nt":
        return

    with contextlib.suppress(Exception):
        windll = getattr(ctypes, "windll", None)
        if windll is None:
            return
        windll.kernel32.SetFileAttributesW(
            str(lock_path),
            0x2,  # FILE_ATTRIBUTE_HIDDEN
        )


def _workspace_document_lock_info(
    fname: str | os.PathLike[str],
) -> _WorkspaceDocumentLockInfo:
    lock_path = _workspace_lock_path(fname)
    lock = QtCore.QLockFile(lock_path)
    locked = False
    pid = 0
    hostname = ""
    appname = ""
    with contextlib.suppress(Exception):
        locked, pid, hostname, appname = lock.getLockInfo()
    if not locked:
        pid = 0
        hostname = ""
        appname = ""
    owner = QtCore.QFileInfo(lock_path).owner()
    return _WorkspaceDocumentLockInfo(
        path=lock_path,
        owner=owner,
        hostname=hostname,
        appname=appname,
        pid=pid if pid > 0 else None,
    )


def _acquire_workspace_document_lock(
    fname: str | os.PathLike[str],
) -> QtCore.QLockFile:
    lock_path = _workspace_lock_path(fname)
    lock = QtCore.QLockFile(lock_path)
    # Document locks are long-lived; Qt uses 0 to disable age-based stale detection.
    lock.setStaleLockTime(0)
    # Do not block the UI thread when another manager already owns the workspace.
    if not lock.tryLock(0):
        raise BlockingIOError(
            errno.EAGAIN,
            f"Workspace file is already open or locked: {fname}",
        )
    _hide_workspace_lock_file(lock_path)
    return lock


def _workspace_txn_attr_target(h5_file, target_path: str):
    if target_path == "/":
        return h5_file.attrs
    group_path = target_path.strip("/")
    if group_path not in h5_file:
        return None
    return h5_file[group_path].attrs


def _write_workspace_attr_backup(
    txn_group, index: int, target_path: str, attrs
) -> None:
    backup_group = txn_group.require_group("attr_backups").create_group(str(index))
    backup_group.attrs["target_path"] = target_path
    attrs_group = backup_group.create_group("attrs")
    for key, value in attrs.items():
        attrs_group.attrs[key] = value


def _restore_workspace_attr_backups(h5_file, txn_group) -> None:
    if "attr_backups" not in txn_group:
        return
    backups = txn_group["attr_backups"]
    for key in sorted(backups, key=lambda value: int(value) if value.isdigit() else 0):
        backup_group = backups[key]
        target_path = backup_group.attrs.get("target_path")
        if isinstance(target_path, bytes):
            target_path = target_path.decode()
        if not isinstance(target_path, str) or "attrs" not in backup_group:
            continue
        target_attrs = _workspace_txn_attr_target(h5_file, target_path)
        if target_attrs is None:
            continue
        workspace_arrays._replace_h5_attrs(target_attrs, backup_group["attrs"].attrs)


def _workspace_transaction_operations(txn_group) -> dict[str, typing.Any]:
    raw_operations = txn_group.attrs.get("operations")
    if isinstance(raw_operations, bytes):
        raw_operations = raw_operations.decode()
    if isinstance(raw_operations, str):
        with contextlib.suppress(json.JSONDecodeError):
            operations = json.loads(raw_operations)
            if isinstance(operations, dict):
                return operations
    return {}


def _workspace_transaction_roots(txn_group) -> tuple[str | None, str | None]:
    pending_root = txn_group.attrs.get("pending_root")
    backup_root = txn_group.attrs.get("backup_root")
    if isinstance(pending_root, bytes):
        pending_root = pending_root.decode()
    if isinstance(backup_root, bytes):
        backup_root = backup_root.decode()
    return (
        pending_root if isinstance(pending_root, str) else None,
        backup_root if isinstance(backup_root, str) else None,
    )


def _cleanup_workspace_transaction_roots(
    h5_file,
    txn_path: str,
    *,
    pending_root: str | None,
    backup_root: str | None,
) -> None:
    for path in (pending_root, backup_root, txn_path):
        if path is not None:
            workspace_arrays._delete_h5_path(h5_file, path)


def _rollback_workspace_group_operations(
    h5_file,
    operations: Mapping[str, typing.Any],
) -> None:
    group_replacements = operations.get("group_replacements", ())
    if not isinstance(group_replacements, list):
        return
    for operation in reversed(group_replacements):
        if not isinstance(operation, dict):
            continue
        group_path = operation.get("group_path")
        backup_path = operation.get("backup_path")
        old_exists = bool(operation.get("old_exists", False))
        if not isinstance(group_path, str) or not isinstance(backup_path, str):
            continue
        backup_exists = workspace_arrays._h5_path_exists(h5_file, backup_path)
        if backup_exists:
            workspace_arrays._delete_h5_path(h5_file, group_path)
            workspace_arrays._ensure_h5_parent_group(h5_file, group_path)
            h5_file.move(backup_path.strip("/"), group_path.strip("/"))
        elif not old_exists:
            workspace_arrays._delete_h5_path(h5_file, group_path)


def _recover_open_workspace_transaction(h5_file, txn_path: str) -> None:
    txn_group = h5_file[txn_path]
    status = txn_group.attrs.get("status")
    if isinstance(status, bytes):
        status = status.decode()
    operations = _workspace_transaction_operations(txn_group)
    pending_root, backup_root = _workspace_transaction_roots(txn_group)

    if status == "committing":
        _rollback_workspace_group_operations(h5_file, operations)
        _restore_workspace_attr_backups(h5_file, txn_group)

    _cleanup_workspace_transaction_roots(
        h5_file,
        txn_path,
        pending_root=pending_root,
        backup_root=backup_root,
    )


def _cleanup_orphan_workspace_internal_groups(h5_file) -> None:
    transaction_roots: set[str] = set()
    for name in list(h5_file):
        if not name.startswith(
            _persistence_constants.WORKSPACE_TRANSACTION_GROUP_PREFIX
        ):
            continue
        pending_root, backup_root = _workspace_transaction_roots(h5_file[name])
        transaction_roots.update(
            root for root in (pending_root, backup_root) if root is not None
        )
    for name in list(h5_file):
        if (
            name.startswith(_persistence_constants.WORKSPACE_LEGACY_TEMP_GROUP_PREFIXES)
            and name not in transaction_roots
        ):
            del h5_file[name]


def _recover_workspace_transactions(fname: str | os.PathLike[str]) -> None:
    active_store = workspace_store.WorkspaceStore.active(fname)
    recovery_lock = (
        active_store.write_lock
        if active_store is not None
        else workspace_arrays._workspace_file_lock(fname)
    )
    with recovery_lock:
        if not pathlib.Path(fname).exists():
            return
        if not _workspace_path_is_itws(fname):
            return
        read_context = (
            active_store.read_session()
            if active_store is not None
            else h5py.File(fname, "r")
        )
        with read_context as h5_file:
            if not _workspace_file_is_workspace(h5_file):
                return
            transaction_names = {
                name
                for name in h5_file
                if name.startswith(
                    _persistence_constants.WORKSPACE_TRANSACTION_GROUP_PREFIX
                )
            }
            transaction_roots: set[str] = set()
            for name in transaction_names:
                pending_root, backup_root = _workspace_transaction_roots(h5_file[name])
                transaction_roots.update(
                    root for root in (pending_root, backup_root) if root is not None
                )
            has_orphan = any(
                name.startswith(
                    _persistence_constants.WORKSPACE_LEGACY_TEMP_GROUP_PREFIXES
                )
                and name not in transaction_roots
                for name in h5_file
            )
        if not transaction_names and not has_orphan:
            return
        file_context = (
            active_store.write_session()
            if active_store is not None
            else h5py.File(fname, "a")
        )
        with file_context as h5_file:
            if not _workspace_file_is_workspace(h5_file):
                return
            for name in list(h5_file):
                if name.startswith(
                    _persistence_constants.WORKSPACE_TRANSACTION_GROUP_PREFIX
                ):
                    _recover_open_workspace_transaction(h5_file, name)
            _cleanup_orphan_workspace_internal_groups(h5_file)
            h5_file.flush()
