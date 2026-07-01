"""ImageTool Manager workspace file format internals.

Workspace files use the ``.itws`` suffix and are HDF5 files written through
``xarray.DataTree``/``h5netcdf``. Runtime code should treat the root attributes as the
authoritative manifest for ordering and manager-level state; HDF5 group names are
payload locations, not the complete schema.

Current file layout
-------------------

::

    /
      attrs:
        imagetool_workspace_schema_version
        imagetool_workspace_manifest
        erlab_version
      <root index>/
        imagetool/
        childtools/
          <child uid>/
            imagetool/ or tool/
      __itws_txn_<id>/, __itws_pending_<id>/, __itws_backup_<id>/

Schema ownership
----------------

The root manifest envelope is assembled by :func:`_workspace_root_attrs_payload`;
non-obvious manifest entries are commented where they are built. Manager-level loader
state and the standalone app envelope are validated in this module. Data Explorer
and Periodic Table state schemas live with the app code that snapshots and restores
them. Shared Qt geometry/visibility state is implemented in
:mod:`erlab.interactive._qt_state`.

ImageTool and ToolWindow payload attrs live with the serializers that write them:
``BaseImageTool.to_dataset`` and ``ToolWindow._saved_tool_attrs``. New writers use
``*_window_state`` attrs; readers only fall back to released legacy ``itool_rect`` and
``tool_rect`` attrs for older workspace files.
"""

from __future__ import annotations

import base64
import collections.abc
import contextlib
import errno
import json
import logging
import math
import numbers
import os
import pathlib
import sys
import tempfile
import time
import traceback
import typing
import uuid
from dataclasses import dataclass

import numpy as np
import pydantic
from qtpy import QtCore

import erlab
from erlab.interactive.imagetool import _serialization
from erlab.interactive.imagetool.manager import _xarray

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, MutableMapping

    import h5py
    import xarray as xr

    from erlab.interactive._options.schema import WorkspaceCompressionMode
else:
    import lazy_loader as _lazy

    h5py = _lazy.load("h5py")

_WORKSPACE_SCHEMA_VERSION = 4
_WORKSPACE_LEGACY_SCHEMA_VERSION = 3
_WORKSPACE_MANIFEST_ATTR = "imagetool_workspace_manifest"
_WORKSPACE_TRANSACTION_PROTOCOL = "recoverable-delta-v1"
_WORKSPACE_PENDING_GROUP_PREFIX = "__itws_pending_"
_WORKSPACE_BACKUP_GROUP_PREFIX = "__itws_backup_"
_WORKSPACE_TRANSACTION_GROUP_PREFIX = "__itws_txn_"
_WORKSPACE_ENCODED_ATTRS_ATTR = "_erlab_workspace_encoded_attrs"
_WORKSPACE_ENCODED_ATTRS_VERSION = 1
_TOOL_DATA_BLOB_NAME_ATTR = _serialization.TOOL_DATA_BLOB_NAME_ATTR
_SAVED_TOOL_DATA_REFERENCE_DIM = _serialization.SAVED_TOOL_DATA_REFERENCE_DIM
_SAVED_TOOL_DATA_BLOB_DIM_PREFIX = _serialization.SAVED_TOOL_DATA_BLOB_DIM_PREFIX
_WORKSPACE_H5PY_DIMENSION_SCALE_ATTRS = frozenset(
    {"CLASS", "NAME", "DIMENSION_LIST", "REFERENCE_LIST"}
)


class WorkspaceLoaderState(pydantic.BaseModel):
    recent_directory: str | None = None
    # QFileDialog name filter selected in manager file-open flows.
    recent_name_filter: str | None = None
    # Manager file-open options are keyed by name filter, not loader name.
    manager_loader_kwargs_by_filter: dict[str, dict[str, typing.Any]] = pydantic.Field(
        default_factory=dict
    )
    # Loader extensions are saved separately because extend_loader applies them later.
    manager_loader_extensions_by_filter: dict[str, dict[str, typing.Any]] = (
        pydantic.Field(default_factory=dict)
    )
    # Data Explorer tabs select loaders by name, so their options use loader names.
    explorer_loader_kwargs_by_name: dict[str, dict[str, typing.Any]] = pydantic.Field(
        default_factory=dict
    )
    explorer_loader_extensions_by_name: dict[str, dict[str, typing.Any]] = (
        pydantic.Field(default_factory=dict)
    )

    model_config = pydantic.ConfigDict(extra="ignore")


class StandaloneAppsState(pydantic.BaseModel):
    schema_version: int = 1
    # Keys are manager standalone app ids such as "explorer" and "ptable".
    apps: dict[str, dict[str, typing.Any]] = pydantic.Field(default_factory=dict)

    model_config = pydantic.ConfigDict(extra="ignore")


class WorkspaceOptionOverridesState(pydantic.BaseModel):
    schema_version: int = 1
    overrides: dict[str, typing.Any] = pydantic.Field(default_factory=dict)

    model_config = pydantic.ConfigDict(extra="ignore")


@dataclass
class _WorkspaceSaveSnapshot:
    generation: int
    root_attrs: dict[str, typing.Any]
    delta_save_count: int
    estimated_obsolete_bytes: int = 0
    replacement_delta_count: int = 0
    repack_estimate_known: bool = True
    compression_mode: WorkspaceCompressionMode = "zstd1"
    file_repack: bool = False
    full_tree: xr.DataTree | None = None
    copy_source: str | None = None
    copy_groups: tuple[tuple[str, str, dict[str, typing.Any] | None], ...] = ()
    rewrite_groups: tuple[tuple[str, dict[str, xr.Dataset]], ...] = ()
    attr_updates: tuple[
        tuple[str, dict[str, typing.Any], tuple[str, dict[str, xr.Dataset]]], ...
    ] = ()

    def close(self) -> None:
        if self.full_tree is not None:
            self.full_tree.close()


@dataclass(frozen=True)
class _WorkspaceDirtyEvent:
    generation: int
    uid: str | None = None
    data: bool = False
    state: bool = False
    added: bool = False
    removed: str | None = None
    structure: str | None = None


@dataclass(frozen=True)
class _WorkspaceDocumentLockInfo:
    path: str
    owner: str
    hostname: str
    appname: str
    pid: int | None


class _WorkspaceSaveWorkerSignals(QtCore.QObject):
    finished = QtCore.Signal(bool, float, str)


class _WorkspaceSaveResultReceiver(QtCore.QObject):
    def __init__(
        self,
        *,
        callback: collections.abc.Callable[[bool, float, str], None] | None = None,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._callback = callback

    @QtCore.Slot(bool, float, str)
    def finish(self, ok: bool, elapsed: float, error_text: str) -> None:
        if self._callback is not None:
            self._callback(ok, elapsed, error_text)


class _WorkspaceSaveWorker(QtCore.QRunnable):
    def __init__(
        self,
        fname: str | os.PathLike[str],
        snapshot: _WorkspaceSaveSnapshot,
    ) -> None:
        super().__init__()
        self.signals = _WorkspaceSaveWorkerSignals()
        self._fname = fname
        self._snapshot = snapshot

    def run(self) -> None:
        start_time = time.perf_counter()
        error_text = ""
        ok = False
        try:
            if self._snapshot.file_repack:
                _write_full_workspace_tree_file(
                    self._fname,
                    None,
                    self._snapshot.root_attrs,
                    copy_source=self._snapshot.copy_source,
                    copy_groups=self._snapshot.copy_groups,
                    compression_mode=self._snapshot.compression_mode,
                )
            elif self._snapshot.full_tree is None:
                _write_workspace_transaction_file(
                    self._fname,
                    self._snapshot.rewrite_groups,
                    self._snapshot.attr_updates,
                    self._snapshot.root_attrs,
                    compression_mode=self._snapshot.compression_mode,
                )
            else:
                _write_full_workspace_tree_file(
                    self._fname,
                    self._snapshot.full_tree,
                    self._snapshot.root_attrs,
                    copy_source=self._snapshot.copy_source,
                    copy_groups=self._snapshot.copy_groups,
                    compression_mode=self._snapshot.compression_mode,
                )
            ok = True
        except Exception:
            error_text = traceback.format_exc()
        finally:
            with contextlib.suppress(Exception):
                self._snapshot.close()
        self.signals.finished.emit(ok, time.perf_counter() - start_time, error_text)


@contextlib.contextmanager
def _open_workspace_h5_file_for_update(
    fname: str | os.PathLike[str],
) -> Iterator[typing.Any]:
    with _xarray._workspace_file_lock(fname), h5py.File(fname, "a") as h5_file:
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
            import stat

            if not stat.S_ISREG(os.lstat(lock_path).st_mode):
                return
            os.chflags(lock_path, stat.UF_HIDDEN)
        return
    if os.name != "nt":
        return

    with contextlib.suppress(Exception):
        import ctypes

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


def _workspace_manifest_from_attrs(
    attrs: Mapping[typing.Any, typing.Any],
) -> dict[str, typing.Any]:
    raw_manifest = attrs.get(_WORKSPACE_MANIFEST_ATTR)
    if isinstance(raw_manifest, bytes):
        raw_manifest = raw_manifest.decode()
    if isinstance(raw_manifest, str):
        with contextlib.suppress(json.JSONDecodeError):
            manifest = json.loads(raw_manifest)
            if isinstance(manifest, dict):
                return manifest
    return {}


def _workspace_delta_save_count_from_attrs(
    attrs: Mapping[typing.Any, typing.Any],
) -> int:
    value = _workspace_manifest_from_attrs(attrs).get("delta_save_count", 0)
    with contextlib.suppress(TypeError, ValueError):
        return max(0, int(value))
    return 0


def _workspace_manifest_nonnegative_int(
    manifest: Mapping[str, typing.Any], key: str
) -> int:
    value = manifest.get(key, 0)
    with contextlib.suppress(TypeError, ValueError):
        return max(0, int(value))
    return 0


def _workspace_manifest_repack_estimate(
    manifest: Mapping[str, typing.Any] | None,
    *,
    delta_save_count: int,
) -> tuple[int, int, bool]:
    if delta_save_count <= 0:
        return 0, 0, True
    if manifest is None:
        return 0, 0, False
    has_estimate = (
        "estimated_obsolete_bytes" in manifest and "replacement_delta_count" in manifest
    )
    return (
        _workspace_manifest_nonnegative_int(manifest, "estimated_obsolete_bytes"),
        _workspace_manifest_nonnegative_int(manifest, "replacement_delta_count"),
        has_estimate,
    )


def _workspace_file_metadata_from_attrs(
    attrs: Mapping[typing.Any, typing.Any],
) -> tuple[int, int, dict[str, typing.Any] | None]:
    schema_version = int(attrs.get("imagetool_workspace_schema_version", 1))
    manifest = None
    if schema_version >= _WORKSPACE_SCHEMA_VERSION:
        manifest = _workspace_manifest_from_attrs(attrs) or None
    return schema_version, _workspace_delta_save_count_from_attrs(attrs), manifest


def _compacted_workspace_root_attrs(
    attrs: Mapping[typing.Any, typing.Any],
) -> dict[str, typing.Any]:
    schema_version, _delta_save_count, manifest = _workspace_file_metadata_from_attrs(
        attrs
    )
    if schema_version != _WORKSPACE_SCHEMA_VERSION or manifest is None:
        raise ValueError(
            "File-level workspace repack requires current workspace schema"
        )
    compacted_manifest = dict(manifest)
    compacted_manifest["schema_version"] = _WORKSPACE_SCHEMA_VERSION
    compacted_manifest["erlab_version"] = erlab.__version__
    compacted_manifest.pop("delta_save_count", None)
    compacted_manifest.pop("transaction_protocol", None)
    compacted_manifest.pop("estimated_obsolete_bytes", None)
    compacted_manifest.pop("replacement_delta_count", None)

    root_attrs = {str(key): value for key, value in attrs.items()}
    root_attrs["imagetool_workspace_schema_version"] = _WORKSPACE_SCHEMA_VERSION
    root_attrs[_WORKSPACE_MANIFEST_ATTR] = json.dumps(compacted_manifest)
    root_attrs["erlab_version"] = erlab.__version__
    return root_attrs


def _workspace_root_attrs_with_repack_estimate(
    attrs: Mapping[typing.Any, typing.Any],
    *,
    estimated_obsolete_bytes: int,
    replacement_delta_count: int,
    repack_estimate_known: bool = True,
) -> dict[str, typing.Any]:
    schema_version, delta_save_count, manifest = _workspace_file_metadata_from_attrs(
        attrs
    )
    if schema_version != _WORKSPACE_SCHEMA_VERSION or manifest is None:
        return {str(key): value for key, value in attrs.items()}
    updated_manifest = dict(manifest)
    if delta_save_count > 0 and repack_estimate_known:
        updated_manifest["estimated_obsolete_bytes"] = max(
            0, int(estimated_obsolete_bytes)
        )
        updated_manifest["replacement_delta_count"] = max(
            0, int(replacement_delta_count)
        )
    else:
        updated_manifest.pop("estimated_obsolete_bytes", None)
        updated_manifest.pop("replacement_delta_count", None)

    root_attrs = {str(key): value for key, value in attrs.items()}
    root_attrs[_WORKSPACE_MANIFEST_ATTR] = json.dumps(updated_manifest)
    return root_attrs


def _current_workspace_schema_version() -> int:
    return _WORKSPACE_SCHEMA_VERSION


def _workspace_path_is_itws(
    fname: str | os.PathLike[str],
) -> bool:
    return pathlib.Path(fname).suffix.lower() == ".itws"


def _set_legacy_workspace_schema(
    attrs: MutableMapping[typing.Hashable, typing.Any],
) -> None:
    attrs["imagetool_workspace_schema_version"] = _WORKSPACE_LEGACY_SCHEMA_VERSION


def _workspace_schema_requires_conversion(schema_version: int) -> bool:
    return schema_version < _WORKSPACE_LEGACY_SCHEMA_VERSION


def _workspace_schema_requires_full_save(schema_version: int) -> bool:
    return (
        _WORKSPACE_LEGACY_SCHEMA_VERSION <= schema_version < _WORKSPACE_SCHEMA_VERSION
    )


def _workspace_root_attrs_payload(
    *,
    root_order: Iterable[typing.Any],
    nodes: Iterable[Mapping[str, typing.Any]],
    delta_save_count: int,
    erlab_version: str,
    workspace_link_id: str | None = None,
    manager_layout: Mapping[str, typing.Any] | None = None,
    loader_state: Mapping[str, typing.Any] | None = None,
    standalone_apps: Mapping[str, typing.Any] | None = None,
    option_overrides: Mapping[str, typing.Any] | None = None,
    estimated_obsolete_bytes: int = 0,
    replacement_delta_count: int = 0,
    repack_estimate_known: bool = True,
) -> dict[str, typing.Any]:
    manifest: dict[str, typing.Any] = {
        "schema_version": _WORKSPACE_SCHEMA_VERSION,
        "erlab_version": erlab_version,
        # HDF5 group order is not the manager's authoritative top-level order.
        "root_order": list(root_order),
        # Stable manager UIDs map to payload groups and link/data metadata here.
        "nodes": list(nodes),
    }
    if workspace_link_id is not None:
        # Scopes watched-variable/source links to this workspace document.
        manifest["workspace_link_id"] = workspace_link_id
    if manager_layout is not None:
        # Stored at root so layout-only saves can avoid rewriting tool payloads.
        manifest["manager_layout"] = dict(manager_layout)
    if loader_state is not None:
        # Manager loader choices are independent of standalone Data Explorer state.
        manifest["loader_state"] = dict(loader_state)
    if standalone_apps is not None:
        # Restored on full workspace open; imports intentionally ignore app windows.
        manifest["standalone_apps"] = dict(standalone_apps)
    if option_overrides is not None:
        # Sparse interactive settings overrides portable with this workspace.
        manifest["interactive_option_overrides"] = dict(option_overrides)
    if delta_save_count > 0:
        # Marks files written through the recoverable in-place delta-save protocol.
        manifest["transaction_protocol"] = _WORKSPACE_TRANSACTION_PROTOCOL
        manifest["delta_save_count"] = int(delta_save_count)
        if repack_estimate_known:
            manifest["estimated_obsolete_bytes"] = max(0, int(estimated_obsolete_bytes))
            manifest["replacement_delta_count"] = max(0, int(replacement_delta_count))
    return {
        "imagetool_workspace_schema_version": _WORKSPACE_SCHEMA_VERSION,
        _WORKSPACE_MANIFEST_ATTR: json.dumps(manifest),
        "erlab_version": erlab_version,
    }


def _is_workspace_internal_group_name(name: typing.Any) -> bool:
    return str(name).startswith(
        (
            _WORKSPACE_PENDING_GROUP_PREFIX,
            _WORKSPACE_BACKUP_GROUP_PREFIX,
            _WORKSPACE_TRANSACTION_GROUP_PREFIX,
        )
    )


def _workspace_root_keys(
    tree: typing.Any, manifest: Mapping[str, typing.Any] | None
) -> list[str]:
    root_keys: list[str] = []
    if manifest is not None:
        raw_root_order = manifest.get("root_order", ())
        if isinstance(raw_root_order, list):
            root_keys.extend(
                str(item)
                for item in raw_root_order
                if str(item) not in root_keys
                and str(item) != "figures"
                and not _is_workspace_internal_group_name(item)
            )
    root_keys.extend(
        str(key)
        for key in tree
        if str(key) not in root_keys
        and str(key) != "figures"
        and not _is_workspace_internal_group_name(key)
    )
    return root_keys


def _workspace_file_is_workspace(h5_file) -> bool:
    if "imagetool_workspace_schema_version" in h5_file.attrs:
        return True
    return h5_file.attrs.get("is_itool_workspace", 0) == 1


def _workspace_path_is_unc(path: str | os.PathLike[str]) -> bool:
    path_str = os.fsdecode(path)
    return path_str.startswith(("\\\\", "//"))


def _workspace_path_is_likely_cloud_path(path: str | os.PathLike[str]) -> bool:
    cloud_markers = {
        "com~apple~clouddocs",
        "dropbox",
        "google drive",
        "icloud drive",
        "icloud~com~apple~clouddocs",
        "mobile documents",
        "onedrive",
        "one drive",
    }
    try:
        parts = pathlib.Path(path).resolve().parts
    except OSError:
        parts = pathlib.Path(path).absolute().parts
    normalized_parts = {part.casefold().replace("-", " ") for part in parts}
    return any(marker in part for part in normalized_parts for marker in cloud_markers)


def _workspace_path_is_likely_network_path(path: str | os.PathLike[str]) -> bool:
    if _workspace_path_is_unc(path):
        return True
    try:
        resolved = pathlib.Path(path).resolve()
    except OSError:
        resolved = pathlib.Path(path).absolute()
    parts = resolved.parts
    if sys.platform == "darwin" and len(parts) > 2 and parts[1] == "Volumes":
        return True
    return len(parts) > 1 and parts[1] in {"net", "nfs", "smb"}


def _workspace_path_is_high_risk(path: str | os.PathLike[str]) -> bool:
    return _workspace_path_is_likely_network_path(
        path
    ) or _workspace_path_is_likely_cloud_path(path)


def _workspace_use_incremental_enabled() -> bool:
    return bool(erlab.interactive.options.model.io.workspace.use_incremental)


def _workspace_incremental_save_on_remote_enabled() -> bool:
    return bool(erlab.interactive.options.model.io.workspace.incremental_save_on_remote)


def _workspace_requires_full_save(
    fname: str | os.PathLike[str],
    *,
    needs_full_save: bool,
    schema_version: int,
    structure_modified: bool,
    has_dirty_added: bool,
    has_dirty_removed: bool,
) -> bool:
    if not _workspace_use_incremental_enabled():
        return True
    if (
        not _workspace_incremental_save_on_remote_enabled()
        and _workspace_path_is_high_risk(fname)
    ):
        return True
    return (
        needs_full_save
        or not pathlib.Path(fname).exists()
        or schema_version != _WORKSPACE_SCHEMA_VERSION
        or structure_modified
        or has_dirty_added
        or has_dirty_removed
    )


def _h5_path_exists(h5_file, path: str) -> bool:
    stripped = path.strip("/")
    return stripped == "" or stripped in h5_file


def _delete_h5_path(h5_file, path: str) -> None:
    stripped = path.strip("/")
    if stripped and stripped in h5_file:
        del h5_file[stripped]


def _ensure_h5_parent_group(h5_file, path: str):
    parent = h5_file
    parts = [part for part in path.strip("/").split("/") if part]
    for part in parts[:-1]:
        parent = parent.require_group(part)
    return parent


def _workspace_serializable_attrs(
    attrs: Mapping[typing.Any, typing.Any],
) -> dict[str, typing.Any]:
    serializable: dict[str, typing.Any] = {}
    encoded_entries: list[list[typing.Any]] = []
    for key, value in attrs.items():
        if not isinstance(key, str) or not key:
            continue
        if key == _WORKSPACE_ENCODED_ATTRS_ATTR:
            existing_entries = _workspace_encoded_attr_entries(value)
            if existing_entries is not None:
                encoded_entries.extend(existing_entries)
                continue
        if (
            key != _WORKSPACE_ENCODED_ATTRS_ATTR
            and _workspace_attr_value_writes_natively(value)
        ):
            serializable[key] = value
            continue
        try:
            encoded_entries.append(
                [_workspace_encode_attr_key(key), _workspace_encode_attr_value(value)]
            )
        except TypeError:
            logger.warning(
                "Dropping workspace attribute %r with unsupported value type %s",
                key,
                type(value).__name__,
            )
    if encoded_entries:
        serializable[_WORKSPACE_ENCODED_ATTRS_ATTR] = json.dumps(
            {
                "version": _WORKSPACE_ENCODED_ATTRS_VERSION,
                "attrs": encoded_entries,
            },
            separators=(",", ":"),
        )
    return serializable


def _workspace_attr_value_writes_natively(value: typing.Any) -> bool:
    if isinstance(value, str):
        return True
    if isinstance(value, bytes):
        return b"\x00" not in value and _workspace_bytes_are_utf8(value)
    if isinstance(value, np.ndarray):
        return value.dtype.kind in "biufcSU"
    if isinstance(value, np.generic):
        return isinstance(value, (np.number, np.bool_))
    if isinstance(value, bool | int | float | complex):
        return True
    if isinstance(value, list | tuple):
        return _workspace_attr_sequence_writes_natively(value)
    return False


def _workspace_attr_sequence_writes_natively(
    value: list[typing.Any] | tuple[typing.Any, ...],
) -> bool:
    if not value:
        return True
    if all(isinstance(item, str) for item in value):
        return True
    if all(
        isinstance(item, bytes)
        and b"\x00" not in item
        and _workspace_bytes_are_utf8(item)
        for item in value
    ):
        return True
    return all(_workspace_attr_numeric_scalar_writes_natively(item) for item in value)


def _workspace_attr_scalar_writes_natively(value: typing.Any) -> bool:
    if isinstance(value, str):
        return True
    if isinstance(value, bytes):
        return b"\x00" not in value and _workspace_bytes_are_utf8(value)
    return _workspace_attr_numeric_scalar_writes_natively(value)


def _workspace_attr_numeric_scalar_writes_natively(value: typing.Any) -> bool:
    if isinstance(value, np.generic):
        return isinstance(value, (np.number, np.bool_))
    return isinstance(value, bool | int | float | complex)


def _workspace_bytes_are_utf8(value: bytes) -> bool:
    try:
        value.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _workspace_encode_attr_key(value: typing.Any) -> dict[str, typing.Any]:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return {"kind": "none"}
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "int", "value": value}
    if isinstance(value, float):
        return {"kind": "float", **_workspace_encode_float(value)}
    if isinstance(value, complex):
        return {
            "kind": "complex",
            "real": _workspace_encode_float(value.real),
            "imag": _workspace_encode_float(value.imag),
        }
    if isinstance(value, str):
        return {"kind": "str", "value": value}
    if isinstance(value, bytes):
        return {
            "kind": "bytes",
            "value": base64.b64encode(value).decode("ascii"),
        }
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [_workspace_encode_attr_key(item) for item in value],
        }
    raise TypeError(f"unsupported attr key type {type(value).__name__!r}")


def _workspace_decode_attr_key(value: typing.Any) -> typing.Hashable:
    decoded = _workspace_decode_attr_value(value)
    if not isinstance(decoded, collections.abc.Hashable):
        raise TypeError(f"decoded attr key is not hashable: {type(decoded).__name__!r}")
    return decoded


def _workspace_encode_attr_value(value: typing.Any) -> dict[str, typing.Any]:
    if isinstance(value, np.ndarray):
        return _workspace_encode_array(value, kind="ndarray")
    if isinstance(value, np.generic):
        return _workspace_encode_array(np.asarray(value), kind="numpy_scalar")
    if value is None:
        return {"kind": "none"}
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "int", "value": value}
    if isinstance(value, float):
        return {"kind": "float", **_workspace_encode_float(value)}
    if isinstance(value, complex):
        return {
            "kind": "complex",
            "real": _workspace_encode_float(value.real),
            "imag": _workspace_encode_float(value.imag),
        }
    if isinstance(value, str):
        return {"kind": "str", "value": value}
    if isinstance(value, bytes):
        return {
            "kind": "bytes",
            "value": base64.b64encode(value).decode("ascii"),
        }
    if isinstance(value, list):
        return {
            "kind": "list",
            "items": [_workspace_encode_attr_value(item) for item in value],
        }
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [_workspace_encode_attr_value(item) for item in value],
        }
    if isinstance(value, collections.abc.Mapping):
        return {
            "kind": "dict",
            "items": [
                [_workspace_encode_attr_key(key), _workspace_encode_attr_value(item)]
                for key, item in value.items()
            ],
        }
    if isinstance(value, numbers.Number):
        raise TypeError(f"unsupported numeric attr type {type(value).__name__!r}")
    raise TypeError(f"unsupported attr value type {type(value).__name__!r}")


def _workspace_decode_attr_value(value: typing.Any) -> typing.Any:
    if not isinstance(value, collections.abc.Mapping):
        raise TypeError("encoded workspace attr value must be a mapping")
    kind = value.get("kind")
    match kind:
        case "none":
            return None
        case "bool":
            return bool(value["value"])
        case "int":
            return int(value["value"])
        case "float":
            return _workspace_decode_float(value)
        case "complex":
            return complex(
                _workspace_decode_float(value["real"]),
                _workspace_decode_float(value["imag"]),
            )
        case "str":
            return str(value["value"])
        case "bytes":
            return base64.b64decode(str(value["value"]).encode("ascii"))
        case "list":
            return [_workspace_decode_attr_value(item) for item in value["items"]]
        case "tuple":
            return tuple(_workspace_decode_attr_value(item) for item in value["items"])
        case "dict":
            return {
                _workspace_decode_attr_key(key): _workspace_decode_attr_value(item)
                for key, item in value["items"]
            }
        case "ndarray":
            return _workspace_decode_array(value)
        case "numpy_scalar":
            return _workspace_decode_array(value)[()]
        case _:
            raise TypeError(f"unknown workspace attr value kind {kind!r}")


def _workspace_encode_float(value: float) -> dict[str, typing.Any]:
    if math.isnan(value):
        return {"special": "nan"}
    if math.isinf(value):
        return {"special": "inf" if value > 0 else "-inf"}
    return {"value": value}


def _workspace_decode_float(value: Mapping[str, typing.Any]) -> float:
    special = value.get("special")
    if special == "nan":
        return math.nan
    if special == "inf":
        return math.inf
    if special == "-inf":
        return -math.inf
    return float(value["value"])


def _workspace_encode_array(value, *, kind: str) -> dict[str, typing.Any]:
    array = np.asarray(value)
    payload: dict[str, typing.Any] = {
        "kind": kind,
        "dtype": array.dtype.str,
        "shape": list(array.shape),
    }
    if array.dtype.kind == "O":
        payload["items"] = _workspace_encode_attr_value(array.tolist())
        return payload
    contiguous = np.ascontiguousarray(array)
    payload["data"] = base64.b64encode(contiguous.tobytes()).decode("ascii")
    return payload


def _workspace_decode_array(value: Mapping[str, typing.Any]):
    dtype = np.dtype(typing.cast("str", value["dtype"]))
    shape = tuple(int(size) for size in typing.cast("list[typing.Any]", value["shape"]))
    if "items" in value:
        items = _workspace_decode_attr_value(value["items"])
        return np.asarray(items, dtype=object).reshape(shape)
    data = base64.b64decode(str(value["data"]).encode("ascii"))
    return np.frombuffer(data, dtype=dtype).copy().reshape(shape)


def _workspace_encoded_attr_entries(value: typing.Any) -> list[list[typing.Any]] | None:
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if not isinstance(value, str):
        return None
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("version") != _WORKSPACE_ENCODED_ATTRS_VERSION
        or not isinstance(payload.get("attrs"), list)
    ):
        return None
    entries = payload["attrs"]
    if not all(
        isinstance(entry, list) and len(entry) == 2
        for entry in typing.cast("list[typing.Any]", entries)
    ):
        return None
    return typing.cast("list[list[typing.Any]]", entries)


def _restore_workspace_serialized_attrs(
    attrs: Mapping[typing.Any, typing.Any],
) -> dict[typing.Any, typing.Any]:
    encoded_entries = _workspace_encoded_attr_entries(
        attrs.get(_WORKSPACE_ENCODED_ATTRS_ATTR)
    )
    if encoded_entries is None:
        return dict(attrs)
    restored = {
        key: value
        for key, value in attrs.items()
        if key != _WORKSPACE_ENCODED_ATTRS_ATTR
    }
    for key_payload, value_payload in encoded_entries:
        try:
            key = _workspace_decode_attr_key(key_payload)
            value = _workspace_decode_attr_value(value_payload)
        except (KeyError, TypeError, ValueError):
            logger.warning(
                "Ignoring invalid encoded workspace attribute", exc_info=True
            )
            continue
        if isinstance(key, str) and key:
            restored[key] = value
    return restored


def _sanitize_workspace_attr_names(ds: xr.Dataset) -> xr.Dataset:
    sanitized = ds.copy(deep=False)
    sanitized.attrs = _workspace_serializable_attrs(sanitized.attrs)
    for variable in sanitized.variables.values():
        variable.attrs = _workspace_serializable_attrs(variable.attrs)
    return sanitized


def _restore_workspace_dataset_attrs(ds: xr.Dataset) -> xr.Dataset:
    restored = ds.copy(deep=False)
    restored.attrs = _restore_workspace_serialized_attrs(restored.attrs)
    for variable in restored.variables.values():
        variable.attrs = _restore_workspace_serialized_attrs(variable.attrs)
    return restored


def _replace_h5_attrs(target_attrs, attrs: Mapping[typing.Any, typing.Any]) -> None:
    for key in list(target_attrs):
        del target_attrs[key]
    for key, value in _workspace_serializable_attrs(attrs).items():
        target_attrs[key] = value


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
        _replace_h5_attrs(target_attrs, backup_group["attrs"].attrs)


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
            _delete_h5_path(h5_file, path)


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
        backup_exists = _h5_path_exists(h5_file, backup_path)
        if backup_exists:
            _delete_h5_path(h5_file, group_path)
            _ensure_h5_parent_group(h5_file, group_path)
            h5_file.move(backup_path.strip("/"), group_path.strip("/"))
        elif not old_exists:
            _delete_h5_path(h5_file, group_path)


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
        if not name.startswith(_WORKSPACE_TRANSACTION_GROUP_PREFIX):
            continue
        pending_root, backup_root = _workspace_transaction_roots(h5_file[name])
        transaction_roots.update(
            root for root in (pending_root, backup_root) if root is not None
        )
    internal_prefixes = (
        _WORKSPACE_PENDING_GROUP_PREFIX,
        _WORKSPACE_BACKUP_GROUP_PREFIX,
    )
    for name in list(h5_file):
        if name.startswith(internal_prefixes) and name not in transaction_roots:
            del h5_file[name]


def _recover_workspace_transactions(fname: str | os.PathLike[str]) -> None:
    if not pathlib.Path(fname).exists():
        return
    if not _workspace_path_is_itws(fname):
        return
    with _xarray._workspace_file_lock(fname), h5py.File(fname, "a") as h5_file:
        if not _workspace_file_is_workspace(h5_file):
            return
        for name in list(h5_file):
            if name.startswith(_WORKSPACE_TRANSACTION_GROUP_PREFIX):
                _recover_open_workspace_transaction(h5_file, name)
        _cleanup_orphan_workspace_internal_groups(h5_file)
        h5_file.flush()


def _write_root_attrs_to_open_workspace_file(
    h5_file,
    attrs: Mapping[str, typing.Any],
    *,
    replace: bool = False,
) -> None:
    if replace:
        for key in list(h5_file.attrs):
            del h5_file.attrs[key]
    for key, value in attrs.items():
        h5_file.attrs[key] = value


def _write_workspace_root_attrs_to_file(
    fname: str | os.PathLike[str],
    attrs: Mapping[str, typing.Any],
    *,
    replace: bool = False,
) -> None:
    with _open_workspace_h5_file_for_update(fname) as h5_file:
        _write_root_attrs_to_open_workspace_file(h5_file, attrs, replace=replace)


def _h5py_attrs_to_dict(
    attrs: typing.Any, *, exclude: Iterable[typing.Hashable] = ()
) -> dict[typing.Hashable, typing.Any]:
    excluded = set(exclude)
    out: dict[typing.Hashable, typing.Any] = {}
    for key, value in attrs.items():
        if key in excluded:
            continue
        if isinstance(value, bytes):
            value = value.decode()
        out[key] = value
    return _restore_workspace_serialized_attrs(out)


def _read_workspace_root_attrs_h5py(
    fname: str | os.PathLike[str],
) -> dict[typing.Hashable, typing.Any]:
    _xarray.ensure_workspace_hdf5_filters_registered()
    with _xarray._workspace_file_lock(fname), h5py.File(fname, "r") as h5_file:
        if not _workspace_file_is_workspace(h5_file):
            raise ValueError("Not a valid workspace file")
        return _h5py_attrs_to_dict(h5_file.attrs)


def _workspace_live_root_group_copy_groups(
    fname: str | os.PathLike[str],
) -> tuple[tuple[str, str, dict[str, typing.Any] | None], ...]:
    _xarray.ensure_workspace_hdf5_filters_registered()
    with _xarray._workspace_file_lock(fname), h5py.File(fname, "r") as h5_file:
        if not _workspace_file_is_workspace(h5_file):
            raise ValueError("Not a valid workspace file")
        return tuple(
            (name, name, None)
            for name, item in h5_file.items()
            if isinstance(item, h5py.Group)
            and not _is_workspace_internal_group_name(name)
        )


def _workspace_h5_object_storage_size(obj: typing.Any) -> int:
    if isinstance(obj, h5py.Dataset):
        return max(0, int(obj.id.get_storage_size()))
    if isinstance(obj, h5py.Group):
        return sum(_workspace_h5_object_storage_size(child) for child in obj.values())
    return 0


def _workspace_h5_paths_storage_size(
    fname: str | os.PathLike[str],
    paths: Iterable[str],
) -> tuple[int, int]:
    total = 0
    existing_count = 0
    _xarray.ensure_workspace_hdf5_filters_registered()
    with _xarray._workspace_file_lock(fname), h5py.File(fname, "r") as h5_file:
        if not _workspace_file_is_workspace(h5_file):
            raise ValueError("Not a valid workspace file")
        for path in paths:
            path = path.strip("/")
            if path not in h5_file:
                continue
            existing_count += 1
            total += _workspace_h5_object_storage_size(h5_file[path])
    return total, existing_count


def _workspace_live_h5_storage_size(fname: str | os.PathLike[str]) -> int:
    total = 0
    _xarray.ensure_workspace_hdf5_filters_registered()
    with _xarray._workspace_file_lock(fname), h5py.File(fname, "r") as h5_file:
        if not _workspace_file_is_workspace(h5_file):
            raise ValueError("Not a valid workspace file")
        for name, item in h5_file.items():
            if isinstance(item, h5py.Group) and not _is_workspace_internal_group_name(
                name
            ):
                total += _workspace_h5_object_storage_size(item)
    return total


def _workspace_obsolete_estimate(
    fname: str | os.PathLike[str],
) -> int:
    try:
        file_size = pathlib.Path(fname).stat().st_size
    except OSError:
        return 0
    live_storage_size = _workspace_live_h5_storage_size(fname)
    return max(0, int(file_size) - live_storage_size)


def _workspace_file_repack_payload(
    fname: str | os.PathLike[str],
) -> tuple[
    dict[str, typing.Any], tuple[tuple[str, str, dict[str, typing.Any] | None], ...]
]:
    _recover_workspace_transactions(fname)
    root_attrs = _read_workspace_root_attrs_h5py(fname)
    return (
        _compacted_workspace_root_attrs(root_attrs),
        _workspace_live_root_group_copy_groups(fname),
    )


def _workspace_h5py_dataset_storage_supported(dataset: typing.Any) -> bool:
    return dataset.dtype.kind in "biufcS" or (
        dataset.dtype.kind == "O" and h5py.check_string_dtype(dataset.dtype) is not None
    )


def _workspace_h5py_coord_dims_fit(
    coord: xr.DataArray, data_array: xr.DataArray
) -> bool:
    return all(
        dim in data_array.sizes and coord.sizes[dim] == data_array.sizes[dim]
        for dim in coord.dims
    )


def _workspace_h5py_variable_payload(
    variable: xr.Variable, name: typing.Hashable
) -> tuple[typing.Any, dict[typing.Hashable, typing.Any], typing.Any] | None:
    import xarray as xr

    try:
        if variable.dtype.kind == "M":
            variable = xr.coders.CFDatetimeCoder().encode(variable, name=str(name))
        elif variable.dtype.kind == "m":
            variable = xr.coders.CFTimedeltaCoder().encode(variable, name=str(name))
    except Exception:
        return None

    data = np.asarray(variable.data)
    dtype = None
    if data.dtype.kind == "U":
        data = data.astype(object)
        dtype = h5py.string_dtype(encoding="utf-8")
    elif data.dtype.kind not in "biufcS":
        return None
    return data, dict(variable.attrs), dtype


def _workspace_h5py_dataarray_can_write(data_array: xr.DataArray) -> bool:
    if data_array.chunks is not None:
        return False
    return (
        _workspace_h5py_variable_payload(data_array.variable, data_array.name)
        is not None
    )


def _workspace_h5py_read_values(dataset: typing.Any) -> typing.Any:
    string_info = h5py.check_string_dtype(dataset.dtype)
    if dataset.dtype.kind == "O" and string_info is not None:
        values = np.asarray(dataset.asstr()[()])
        if values.dtype.kind == "O":
            values = values.astype(str)
        return values
    return np.asarray(dataset[()])


def _workspace_h5py_decode_coord_variable(
    variable: xr.Variable, name: str
) -> xr.Variable | None:
    import xarray as xr

    attrs = variable.attrs
    dtype_attr = attrs.get("dtype")
    units_attr = attrs.get("units")
    calendar_attr = attrs.get("calendar")
    try:
        if isinstance(dtype_attr, str) and dtype_attr.startswith("timedelta64["):
            return xr.coders.CFTimedeltaCoder().decode(variable, name=name)
        if (
            isinstance(units_attr, str)
            and isinstance(calendar_attr, str)
            and " since " in units_attr
        ):
            return xr.coders.CFDatetimeCoder().decode(variable, name=name)
    except Exception:
        return None
    return variable


def _workspace_h5py_dataset_variable(
    dataset: typing.Any,
    dims: tuple[str, ...],
    *,
    name: str,
    exclude_attrs: Iterable[typing.Hashable],
) -> xr.Variable | None:
    import xarray as xr

    if not _workspace_h5py_dataset_storage_supported(dataset):
        return None
    variable = xr.Variable(
        dims,
        _workspace_h5py_read_values(dataset),
        _h5py_attrs_to_dict(dataset.attrs, exclude=exclude_attrs),
    )
    return _workspace_h5py_decode_coord_variable(variable, name)


def _workspace_h5py_create_kwargs(
    encoding: Mapping[str, typing.Any] | None,
) -> dict[str, typing.Any]:
    if encoding is None:
        return {}
    kwargs: dict[str, typing.Any] = {}
    if "chunksizes" in encoding:
        kwargs["chunks"] = encoding["chunksizes"]
    for key in ("compression", "compression_opts", "shuffle", "fletcher32"):
        if key in encoding:
            kwargs[key] = encoding[key]
    return kwargs


def _workspace_h5py_filter_options(dataset: typing.Any) -> dict[int, tuple[int, ...]]:
    create_plist = dataset.id.get_create_plist()
    return {
        create_plist.get_filter(index)[0]: tuple(create_plist.get_filter(index)[2])
        for index in range(create_plist.get_nfilters())
    }


def _workspace_h5py_dataset_matches_encoding(
    dataset: typing.Any,
    encoding: Mapping[typing.Any, typing.Any],
) -> bool:
    filters = _workspace_h5py_filter_options(dataset)
    expected_filter = encoding.get("compression")
    if expected_filter is None:
        return not filters
    actual_options = filters.get(int(expected_filter))
    if actual_options is None:
        return False
    expected_options = encoding.get("compression_opts")
    return expected_options is None or actual_options == tuple(expected_options)


def _workspace_h5_group_matches_compression_mode(
    h5_file: typing.Any,
    group_path: str,
    ds: xr.Dataset,
    compression_mode: WorkspaceCompressionMode,
) -> bool:
    group_path = group_path.strip("/")
    if group_path not in h5_file:
        return False
    group = h5_file[group_path]
    encoding = _xarray.workspace_dataset_encoding(ds, compression_mode=compression_mode)
    for name in ds.data_vars:
        dataset_name = str(name)
        if dataset_name not in group:
            return False
        dataset = group[dataset_name]
        if not _workspace_h5py_dataset_matches_encoding(
            dataset, encoding.get(name, {})
        ):
            return False
    return True


def _workspace_h5py_attr_text(value: typing.Any) -> str | None:
    if isinstance(value, bytes):
        return value.decode()
    if hasattr(value, "decode"):
        decoded = value.decode()
        return str(decoded)
    if isinstance(value, str):
        return value
    return None


def _workspace_h5py_dataset_is_dimension_scale(dataset: typing.Any) -> bool:
    return _workspace_h5py_attr_text(dataset.attrs.get("CLASS")) == "DIMENSION_SCALE"


def _workspace_h5_group_matches_current_compression(
    h5_file: typing.Any,
    group_path: str,
    compression_mode: WorkspaceCompressionMode,
) -> bool:
    group_path = group_path.strip("/")
    if group_path not in h5_file:
        return False
    group = h5_file[group_path]
    if not isinstance(group, h5py.Group):
        return False
    compression_encoding = _xarray._workspace_blosc2_encoding(compression_mode)
    for item in group.values():
        if not isinstance(item, h5py.Dataset):
            continue
        if _workspace_h5py_dataset_is_dimension_scale(item):
            continue
        encoding = {}
        if (
            compression_encoding
            and item.dtype.kind in "iufc"
            and int(item.size) * int(item.dtype.itemsize)
            >= _xarray._WORKSPACE_COMPRESSION_MIN_BYTES
        ):
            encoding = compression_encoding
        if not _workspace_h5py_dataset_matches_encoding(item, encoding):
            return False
    return True


def _workspace_h5py_type_contains_reference(type_id: typing.Any) -> bool:
    type_class = type_id.get_class()
    if type_class == h5py.h5t.REFERENCE:
        return True
    if type_class in {h5py.h5t.ARRAY, h5py.h5t.VLEN}:
        super_type = type_id.get_super()
        try:
            return _workspace_h5py_type_contains_reference(super_type)
        finally:
            super_type.close()
    if type_class == h5py.h5t.COMPOUND:
        for index in range(type_id.get_nmembers()):
            member_type = type_id.get_member_type(index)
            try:
                if _workspace_h5py_type_contains_reference(member_type):
                    return True
            finally:
                member_type.close()
    return False


def _workspace_h5py_attr_contains_reference(
    source_attrs: typing.Any, key: typing.Hashable
) -> bool:
    attr_id = source_attrs.get_id(key)
    type_id = attr_id.get_type()
    try:
        return _workspace_h5py_type_contains_reference(type_id)
    finally:
        type_id.close()
        attr_id.close()


def _workspace_h5py_copy_regular_attrs(
    source_attrs: typing.Any,
    target_attrs: typing.Any,
    *,
    skip_dimension_scale_attrs: bool,
) -> None:
    for key, value in source_attrs.items():
        if skip_dimension_scale_attrs and key in _WORKSPACE_H5PY_DIMENSION_SCALE_ATTRS:
            continue
        if _workspace_h5py_attr_contains_reference(source_attrs, key):
            continue
        target_attrs[key] = value


def _workspace_h5py_rebuild_dimension_scales(
    source_group: typing.Any,
    target_group: typing.Any,
) -> None:
    _workspace_h5py_copy_regular_attrs(
        source_group.attrs,
        target_group.attrs,
        skip_dimension_scale_attrs=False,
    )
    scales_by_id: dict[int, typing.Any] = {}
    for name, source_obj in source_group.items():
        target_obj = target_group[name]
        if isinstance(source_obj, h5py.Group):
            _workspace_h5py_rebuild_dimension_scales(source_obj, target_obj)
            continue
        if not isinstance(source_obj, h5py.Dataset):
            continue
        _workspace_h5py_copy_regular_attrs(
            source_obj.attrs,
            target_obj.attrs,
            skip_dimension_scale_attrs=True,
        )
        if not _workspace_h5py_dataset_is_dimension_scale(source_obj):
            continue
        dim_id = source_obj.attrs.get("_Netcdf4Dimid")
        if dim_id is None:
            continue
        scale_name = _workspace_h5py_attr_text(source_obj.attrs.get("NAME")) or name
        target_obj.make_scale(scale_name)
        target_obj.attrs["_Netcdf4Dimid"] = np.int32(dim_id)
        if "_Netcdf4Coordinates" in source_obj.attrs:
            target_obj.attrs["_Netcdf4Coordinates"] = source_obj.attrs[
                "_Netcdf4Coordinates"
            ]
        scales_by_id[int(dim_id)] = target_obj

    for name, source_obj in source_group.items():
        if not isinstance(source_obj, h5py.Dataset):
            continue
        if _workspace_h5py_dataset_is_dimension_scale(source_obj):
            continue
        if "_Netcdf4Coordinates" not in source_obj.attrs:
            continue
        target_obj = target_group[name]
        coordinate_ids = np.asarray(source_obj.attrs["_Netcdf4Coordinates"]).reshape(-1)
        if len(coordinate_ids) != target_obj.ndim:
            continue
        for axis, dim_id in enumerate(coordinate_ids):
            scale = scales_by_id.get(int(dim_id))
            if scale is not None:
                target_obj.dims[axis].attach_scale(scale)


def _copy_workspace_h5_group_to_open_file(
    source_file: typing.Any,
    target_file: typing.Any,
    source_path: str,
    target_path: str,
    attrs: Mapping[str, typing.Any] | None,
) -> bool:
    source_path = source_path.strip("/")
    target_path = target_path.strip("/")
    if source_path not in source_file:
        return False
    source_group = source_file[source_path]
    if not isinstance(source_group, h5py.Group):
        return False
    parent = _ensure_h5_parent_group(target_file, target_path)
    target_name = target_path.rsplit("/", maxsplit=1)[-1]
    if target_name in parent:
        del parent[target_name]
    source_file.copy(source_path, parent, name=target_name, without_attrs=True)
    target_group = parent[target_name]
    _workspace_h5py_rebuild_dimension_scales(source_group, target_group)
    if attrs is not None:
        _replace_h5_attrs(target_group.attrs, attrs)
    return True


def _workspace_h5py_create_dataset(
    group: typing.Any,
    name: str,
    variable: xr.Variable,
    *,
    encoding: Mapping[str, typing.Any] | None = None,
) -> typing.Any | None:
    payload = _workspace_h5py_variable_payload(variable, name)
    if payload is None:
        return None
    data, attrs, dtype = payload
    kwargs = _workspace_h5py_create_kwargs(encoding)
    if dtype is not None:
        kwargs["dtype"] = dtype
    dataset = group.create_dataset(name, data=data, **kwargs)
    for key, value in attrs.items():
        dataset.attrs[key] = value
    return dataset


def _workspace_h5py_tool_data_blob_dim(variable_name: str) -> str:
    return f"{_SAVED_TOOL_DATA_BLOB_DIM_PREFIX}{variable_name.encode().hex()}>"


def _workspace_h5py_dataarray_is_tool_reference(data_array: xr.DataArray) -> bool:
    return (
        data_array.ndim == 1
        and data_array.dims == (_SAVED_TOOL_DATA_REFERENCE_DIM,)
        and data_array.size == 0
    )


def _workspace_h5py_dataarray_is_tool_blob(data_array: xr.DataArray) -> bool:
    return _TOOL_DATA_BLOB_NAME_ATTR in data_array.attrs


def _workspace_h5py_dataarray_is_independent_tool_item(
    data_array: xr.DataArray,
) -> bool:
    return _workspace_h5py_dataarray_can_write(data_array) and (
        _workspace_h5py_dataarray_is_tool_reference(data_array)
        or _workspace_h5py_dataarray_is_tool_blob(data_array)
    )


def _workspace_h5py_dataset_independent_tool_variable(
    dataset: typing.Any,
    variable_name: str,
    *,
    exclude_attrs: Iterable[typing.Hashable],
) -> xr.Variable | None:
    import xarray as xr

    if not _workspace_h5py_dataset_storage_supported(dataset):
        return None
    attrs = _h5py_attrs_to_dict(dataset.attrs, exclude=exclude_attrs)
    if _TOOL_DATA_BLOB_NAME_ATTR in attrs:
        dims = (_workspace_h5py_tool_data_blob_dim(variable_name),)
    elif dataset.ndim == 1 and dataset.size == 0:
        dims = (_SAVED_TOOL_DATA_REFERENCE_DIM,)
    else:
        return None
    return xr.Variable(dims, _workspace_h5py_read_values(dataset), attrs)


def _workspace_h5py_extra_tool_data_names(
    ds: xr.Dataset, data_name: typing.Hashable
) -> frozenset[typing.Hashable]:
    if data_name != _serialization.SAVED_TOOL_DATA_NAME:
        return frozenset()
    return frozenset(
        name
        for name, data_array in ds.data_vars.items()
        if name != data_name
        and _workspace_h5py_dataarray_is_independent_tool_item(data_array)
    )


def _workspace_h5py_dataset_has_only_independent_tool_items(
    ds: xr.Dataset, data_name: typing.Hashable
) -> bool:
    return (
        data_name == _serialization.SAVED_TOOL_DATA_NAME
        and not ds.coords
        and all(
            _workspace_h5py_dataarray_is_independent_tool_item(data_array)
            for data_array in ds.data_vars.values()
        )
    )


def _read_workspace_dataset_group_h5py(
    fname: str | os.PathLike[str],
    group_path: str,
    *,
    preferred_data_name: str | None = None,
) -> xr.Dataset | None:
    import xarray as xr

    _xarray.ensure_workspace_hdf5_filters_registered()
    group_path = group_path.strip("/")
    internal_attrs = (
        "CLASS",
        "DIMENSION_LIST",
        "NAME",
        "REFERENCE_LIST",
        "_Netcdf4Coordinates",
        "_Netcdf4Dimid",
        "coordinates",
    )
    with _xarray._workspace_file_lock(fname), h5py.File(fname, "r") as h5_file:
        if group_path not in h5_file or not isinstance(h5_file[group_path], h5py.Group):
            return None
        group = h5_file[group_path]
        datasets: dict[str, h5py.Dataset] = {}
        for name, obj in group.items():
            if not isinstance(
                obj, h5py.Dataset
            ) or not _workspace_h5py_dataset_storage_supported(obj):
                continue
            marker = obj.attrs.get("CLASS")
            if isinstance(marker, bytes):
                marker = marker.decode()
            if marker == "DIMENSION_SCALE":
                continue
            datasets[name] = obj
        if preferred_data_name is not None and preferred_data_name in datasets:
            data_name = preferred_data_name
        elif len(datasets) == 1:
            data_name = next(iter(datasets))
        else:
            return None

        if (
            preferred_data_name == _serialization.SAVED_TOOL_DATA_NAME
            and data_name == _serialization.SAVED_TOOL_DATA_NAME
        ):
            independent_data_vars: dict[typing.Hashable, xr.Variable] = {}
            for variable_name, dataset in datasets.items():
                variable = _workspace_h5py_dataset_independent_tool_variable(
                    dataset,
                    variable_name,
                    exclude_attrs=internal_attrs,
                )
                if variable is None:
                    break
                independent_data_vars[variable_name] = variable
            else:
                if _serialization.SAVED_TOOL_DATA_NAME in independent_data_vars:
                    return xr.Dataset(
                        independent_data_vars,
                        attrs=_h5py_attrs_to_dict(group.attrs),
                    )

        def _dataset_dims(dataset: h5py.Dataset) -> tuple[str, ...] | None:
            dims: list[str] = []
            for axis, dim in enumerate(dataset.dims):
                dim_keys = list(dim.keys())
                if len(dim_keys) != 1:
                    return None
                dim_name = str(dim_keys[0])
                scale = dim[dim_name]
                if (
                    not isinstance(scale, h5py.Dataset)
                    or scale.ndim != 1
                    or scale.shape[0] != dataset.shape[axis]
                    or not _workspace_h5py_dataset_storage_supported(scale)
                ):
                    return None
                dims.append(dim_name)
            return tuple(dims)

        data_dataset = datasets[data_name]
        if data_dataset.dtype.kind not in "biufc":
            return None
        dims: list[str] = []
        coords: dict[str, typing.Any] = {}
        for axis, dim in enumerate(data_dataset.dims):
            dim_keys = list(dim.keys())
            if len(dim_keys) != 1:
                return None
            dim_name = str(dim_keys[0])
            scale = dim[dim_name]
            if (
                not isinstance(scale, h5py.Dataset)
                or scale.ndim != 1
                or scale.shape[0] != data_dataset.shape[axis]
                or not _workspace_h5py_dataset_storage_supported(scale)
            ):
                return None
            dims.append(dim_name)
            coord_variable = _workspace_h5py_dataset_variable(
                scale,
                (dim_name,),
                name=dim_name,
                exclude_attrs=internal_attrs,
            )
            if coord_variable is None:
                return None
            coords[dim_name] = coord_variable

        scalar_coord_names = data_dataset.attrs.get("coordinates", "")
        if isinstance(scalar_coord_names, bytes):
            scalar_coord_names = scalar_coord_names.decode()
        legacy_spaced_coord_names = tuple(
            name
            for name, dataset in datasets.items()
            if name != data_name
            and _serialization.coord_name_needs_private_storage(name)
            and _workspace_h5py_dataset_storage_supported(dataset)
        )
        if isinstance(scalar_coord_names, str):
            for coord_name in scalar_coord_names.split():
                if coord_name not in group or not isinstance(
                    group[coord_name], h5py.Dataset
                ):
                    if legacy_spaced_coord_names:
                        continue
                    return None
                coord_dataset = group[coord_name]
                if coord_dataset.ndim == 0:
                    coord_dims = ()
                else:
                    coord_dims = _dataset_dims(coord_dataset)
                    if coord_dims is None or not all(dim in dims for dim in coord_dims):
                        return None
                coord_variable = _workspace_h5py_dataset_variable(
                    coord_dataset,
                    coord_dims,
                    name=coord_name,
                    exclude_attrs=internal_attrs,
                )
                if coord_variable is None:
                    return None
                coords[coord_name] = coord_variable

        data_attrs = _h5py_attrs_to_dict(data_dataset.attrs, exclude=internal_attrs)
        data_vars: dict[typing.Hashable, typing.Any] = {
            data_name: (
                tuple(dims),
                np.asarray(data_dataset[()]),
                data_attrs,
            )
        }

        private_records = _serialization.private_coord_records_from_attrs(data_attrs)
        for record in private_records or ():
            variable_name = record["variable_name"]
            if variable_name not in group or not isinstance(
                group[variable_name], h5py.Dataset
            ):
                return None
            coord_dataset = group[variable_name]
            coord_dims = tuple(record["dims"])
            if (
                coord_dataset.ndim != len(coord_dims)
                or not _workspace_h5py_dataset_storage_supported(coord_dataset)
                or not all(dim in dims for dim in coord_dims)
            ):
                return None
            coord_variable = _workspace_h5py_dataset_variable(
                coord_dataset,
                coord_dims,
                name=variable_name,
                exclude_attrs=internal_attrs,
            )
            if coord_variable is None:
                return None
            data_vars[variable_name] = coord_variable

        for variable_name in legacy_spaced_coord_names:
            if variable_name in data_vars:
                continue
            coord_dataset = datasets[variable_name]
            coord_dims = _dataset_dims(coord_dataset)
            if coord_dims is None or not all(dim in dims for dim in coord_dims):
                continue
            coord_variable = _workspace_h5py_dataset_variable(
                coord_dataset,
                coord_dims,
                name=variable_name,
                exclude_attrs=internal_attrs,
            )
            if coord_variable is None:
                continue
            data_vars[variable_name] = coord_variable

        if (
            preferred_data_name == _serialization.SAVED_TOOL_DATA_NAME
            and data_name == _serialization.SAVED_TOOL_DATA_NAME
        ):
            for variable_name, dataset in datasets.items():
                if (
                    variable_name == data_name
                    or variable_name in data_vars
                    or variable_name in coords
                ):
                    continue
                variable = _workspace_h5py_dataset_independent_tool_variable(
                    dataset,
                    variable_name,
                    exclude_attrs=internal_attrs,
                )
                if variable is None:
                    return None
                data_vars[variable_name] = variable

        return _serialization.restore_private_coords(
            xr.Dataset(
                data_vars,
                coords=coords,
                attrs=_h5py_attrs_to_dict(group.attrs),
            ),
            data_name,
        )


def _workspace_h5py_data_name(ds: xr.Dataset) -> typing.Hashable | None:
    if _serialization.ITOOL_DATA_NAME in ds.data_vars:
        return _serialization.ITOOL_DATA_NAME
    if _serialization.SAVED_TOOL_DATA_NAME in ds.data_vars:
        return _serialization.SAVED_TOOL_DATA_NAME
    if len(ds.data_vars) == 1:
        return next(iter(ds.data_vars))
    return None


def _workspace_dataset_can_write_h5py(ds: xr.Dataset) -> bool:
    data_name = _workspace_h5py_data_name(ds)
    if data_name is None:
        return False
    if _workspace_h5py_dataset_has_only_independent_tool_items(ds, data_name):
        return True
    private_data_names = set(_serialization.private_coord_variable_names(ds, data_name))
    extra_data_names = _workspace_h5py_extra_tool_data_names(ds, data_name)
    if any(
        name != data_name
        and name not in private_data_names
        and name not in extra_data_names
        for name in ds.data_vars
    ):
        return False
    data_array = ds[data_name]
    if data_array.chunks is not None or data_array.dtype.kind not in "biufc":
        return False
    for private_name in private_data_names:
        if private_name not in ds.data_vars:
            return False
        private_data = ds[private_name]
        if (
            private_data.chunks is not None
            or not _workspace_h5py_coord_dims_fit(private_data, data_array)
            or not _workspace_h5py_dataarray_can_write(private_data)
        ):
            return False
    for dim in data_array.dims:
        coord = ds.coords.get(dim)
        if (
            coord is None
            or coord.dims != (dim,)
            or not _workspace_h5py_dataarray_can_write(coord)
        ):
            return False
    for name, coord in ds.coords.items():
        if name in data_array.dims:
            continue
        if (
            _serialization.coord_name_needs_private_storage(name)
            or not _workspace_h5py_coord_dims_fit(coord, data_array)
            or not _workspace_h5py_dataarray_can_write(coord)
        ):
            return False
    return True


def _write_workspace_independent_tool_items_h5py(
    group: typing.Any,
    ds: xr.Dataset,
    *,
    encoding: Mapping[typing.Hashable, Mapping[str, typing.Any]] | None = None,
) -> bool:
    for variable_name, data_array in ds.data_vars.items():
        dataset = _workspace_h5py_create_dataset(
            group,
            str(variable_name),
            data_array.variable,
            encoding=None if encoding is None else encoding.get(variable_name),
        )
        if dataset is None:
            return False
    return True


def _write_workspace_dataset_group_h5py(
    fname: str | os.PathLike[str],
    group_path: str,
    ds: xr.Dataset,
    *,
    encoding: Mapping[typing.Hashable, Mapping[str, typing.Any]] | None = None,
) -> bool:
    if not _workspace_dataset_can_write_h5py(ds):
        return False
    ds = _sanitize_workspace_attr_names(ds)
    data_name = _workspace_h5py_data_name(ds)
    if data_name is None:
        return False
    private_data_names = _serialization.private_coord_variable_names(ds, data_name)

    _xarray.ensure_workspace_hdf5_filters_registered()
    group_path = group_path.strip("/")
    with h5py.File(fname, "a") as h5_file:
        if group_path in h5_file:
            del h5_file[group_path]
        parent = _ensure_h5_parent_group(h5_file, group_path)
        group_name = group_path.rsplit("/", maxsplit=1)[-1]
        group = parent.create_group(group_name)
        try:
            for key, value in ds.attrs.items():
                group.attrs[key] = value
            if encoding is None:
                encoding = _xarray.workspace_dataset_encoding(ds)
            if _workspace_h5py_dataset_has_only_independent_tool_items(ds, data_name):
                if not _write_workspace_independent_tool_items_h5py(
                    group, ds, encoding=encoding
                ):
                    del parent[group_name]
                    return False
                return True
            data_array = ds[data_name]
            dim_scales = []
            dim_scales_by_name = {}
            for dim_id, dim in enumerate(data_array.dims):
                coord = ds.coords[dim]
                coord_dataset = _workspace_h5py_create_dataset(
                    group, str(dim), coord.variable
                )
                if coord_dataset is None:
                    del parent[group_name]
                    return False
                coord_dataset.make_scale(str(dim))
                coord_dataset.attrs["_Netcdf4Dimid"] = np.int32(dim_id)
                coord_dataset.attrs["_Netcdf4Coordinates"] = np.asarray(
                    [dim_id], dtype=np.int32
                )
                dim_scales.append(coord_dataset)
                dim_scales_by_name[dim] = coord_dataset

            data_encoding = encoding.get(data_name, {})
            data_dataset = group.create_dataset(
                str(data_name),
                data=np.asarray(data_array.data),
                **_workspace_h5py_create_kwargs(data_encoding),
            )
            for dim_index, scale in enumerate(dim_scales):
                data_dataset.dims[dim_index].attach_scale(scale)
            data_dataset.attrs["_Netcdf4Coordinates"] = np.arange(
                len(dim_scales), dtype=np.int32
            )
            for key, value in data_array.attrs.items():
                data_dataset.attrs[key] = value

            scalar_coord_names: list[str] = []
            for name, coord in ds.coords.items():
                if name in data_array.dims:
                    continue
                coord_dataset = _workspace_h5py_create_dataset(
                    group, str(name), coord.variable
                )
                if coord_dataset is None:
                    del parent[group_name]
                    return False
                coordinate_ids = []
                for coord_dim_index, coord_dim in enumerate(coord.dims):
                    scale = dim_scales_by_name[coord_dim]
                    coord_dataset.dims[coord_dim_index].attach_scale(scale)
                    coordinate_ids.append(data_array.dims.index(coord_dim))
                if coordinate_ids:
                    coord_dataset.attrs["_Netcdf4Coordinates"] = np.asarray(
                        coordinate_ids, dtype=np.int32
                    )
                    coord_dataset.attrs["_Netcdf4Dimid"] = np.int32(coordinate_ids[0])
                scalar_coord_names.append(str(name))
            if scalar_coord_names:
                existing_coordinates = data_dataset.attrs.get("coordinates")
                if isinstance(existing_coordinates, bytes):
                    existing_coordinates = existing_coordinates.decode()
                coordinates = " ".join(scalar_coord_names)
                if isinstance(existing_coordinates, str) and existing_coordinates:
                    coordinates = f"{existing_coordinates} {coordinates}"
                data_dataset.attrs["coordinates"] = coordinates

            for private_name in private_data_names:
                private_data = ds[private_name]
                private_dataset = _workspace_h5py_create_dataset(
                    group, str(private_name), private_data.variable
                )
                if private_dataset is None:
                    del parent[group_name]
                    return False
                private_coordinate_ids: list[int] = []
                for private_dim_index, private_dim in enumerate(private_data.dims):
                    scale = dim_scales_by_name[private_dim]
                    private_dataset.dims[private_dim_index].attach_scale(scale)
                    private_coordinate_ids.append(data_array.dims.index(private_dim))
                if private_coordinate_ids:
                    private_dataset.attrs["_Netcdf4Coordinates"] = np.asarray(
                        private_coordinate_ids, dtype=np.int32
                    )
                    private_dataset.attrs["_Netcdf4Dimid"] = np.int32(
                        private_coordinate_ids[0]
                    )
            for extra_name in _workspace_h5py_extra_tool_data_names(ds, data_name):
                extra_data = ds[extra_name]
                extra_dataset = _workspace_h5py_create_dataset(
                    group,
                    str(extra_name),
                    extra_data.variable,
                    encoding=encoding.get(extra_name),
                )
                if extra_dataset is None:
                    del parent[group_name]
                    return False
        except Exception:
            del parent[group_name]
            return False
    return True


def _write_workspace_dataset_group_to_file(
    fname: str | os.PathLike[str],
    group_path: str,
    ds: xr.Dataset,
    *,
    lock_path: str | os.PathLike[str] | None = None,
    compression_mode: WorkspaceCompressionMode | None = None,
) -> None:
    encoding = _xarray.workspace_dataset_encoding(ds, compression_mode=compression_mode)
    if lock_path is not None:
        normalized_lock_path = _xarray._normalized_file_path(lock_path)
        if normalized_lock_path is not None and any(
            normalized_lock_path in _xarray.dataarray_source_paths(data_array)
            for data_array in (*ds.data_vars.values(), *ds.coords.values())
        ):
            ds = ds.load()

    data_name = _workspace_h5py_data_name(ds)
    if data_name is not None:
        ds = _serialization.encode_private_coords(ds, data_name)
    ds = _sanitize_workspace_attr_names(ds)
    stale_encoding_keys = {
        "chunksizes",
        "compression",
        "compression_opts",
        "contiguous",
        "fletcher32",
        "original_shape",
        "preferred_chunks",
        "shuffle",
        "source",
    }
    for variable in ds.variables.values():
        for key in stale_encoding_keys:
            variable.encoding.pop(key, None)

    maybe_lock = (
        _xarray._workspace_file_lock(lock_path)
        if lock_path is not None
        else contextlib.nullcontext()
    )
    with maybe_lock:
        if _write_workspace_dataset_group_h5py(
            fname, group_path, ds, encoding=encoding
        ):
            return
        ds.to_netcdf(
            fname,
            mode="a",
            engine="h5netcdf",
            group=f"/{group_path.strip('/')}",
            invalid_netcdf=True,
            encoding=encoding,
        )


def _write_workspace_constructor_groups_to_pending(
    fname: str | os.PathLike[str],
    constructor: Mapping[str, xr.Dataset],
    group_path: str,
    pending_path: str,
    *,
    compression_mode: WorkspaceCompressionMode | None = None,
) -> None:
    target_group_path = group_path.strip("/")
    pending_path = pending_path.strip("/")
    for constructor_group_path, ds in sorted(
        constructor.items(), key=lambda item: item[0].count("/")
    ):
        source_group_path = constructor_group_path.strip("/")
        if source_group_path != target_group_path and not source_group_path.startswith(
            f"{target_group_path}/"
        ):
            continue
        relative_path = source_group_path.removeprefix(target_group_path).strip("/")
        pending_group_path = (
            pending_path if not relative_path else f"{pending_path}/{relative_path}"
        )
        _write_workspace_dataset_group_to_file(
            fname,
            pending_group_path,
            ds,
            lock_path=fname,
            compression_mode=compression_mode,
        )


def _path_is_at_or_under(path: str, root_path: str) -> bool:
    path = path.strip("/")
    root_path = root_path.strip("/")
    return path == root_path or path.startswith(f"{root_path}/")


def _move_h5_path(h5_file, source_path: str, destination_path: str) -> None:
    _ensure_h5_parent_group(h5_file, destination_path)
    h5_file.move(source_path.strip("/"), destination_path.strip("/"))


def _set_workspace_transaction_status(h5_file, txn_path: str, status: str) -> None:
    h5_file[txn_path].attrs["status"] = status
    h5_file.flush()


def _prepare_workspace_transaction(
    fname: str | os.PathLike[str],
    txn_path: str,
    pending_root: str,
    backup_root: str,
    rewrite_map: dict[str, tuple[str, dict[str, xr.Dataset]]],
    attr_updates: tuple[
        tuple[str, dict[str, typing.Any], tuple[str, dict[str, xr.Dataset]]], ...
    ],
    root_attrs: Mapping[str, typing.Any],
) -> tuple[
    list[dict[str, typing.Any]],
    list[tuple[str, dict[str, typing.Any], tuple[str, dict[str, xr.Dataset]]]],
]:
    attr_updates_to_write: list[
        tuple[str, dict[str, typing.Any], tuple[str, dict[str, xr.Dataset]]]
    ] = []
    with _open_workspace_h5_file_for_update(fname) as h5_file:
        txn_group = h5_file.create_group(txn_path)
        txn_group.attrs["protocol"] = _WORKSPACE_TRANSACTION_PROTOCOL
        txn_group.attrs["status"] = "preparing"
        txn_group.attrs["pending_root"] = pending_root
        txn_group.attrs["backup_root"] = backup_root

        attr_backup_index = 0
        for payload_path, attrs, fallback in attr_updates:
            attr_path = payload_path.strip("/")
            if any(
                _path_is_at_or_under(attr_path, rewrite_path)
                for rewrite_path in rewrite_map
            ):
                continue
            if attr_path in h5_file:
                _write_workspace_attr_backup(
                    txn_group,
                    attr_backup_index,
                    payload_path,
                    h5_file[attr_path].attrs,
                )
                attr_backup_index += 1
                attr_updates_to_write.append((payload_path, attrs, fallback))
            else:
                fallback_path = fallback[0].strip("/")
                if not any(
                    _path_is_at_or_under(fallback_path, rewrite_path)
                    for rewrite_path in rewrite_map
                ):
                    rewrite_map[fallback_path] = fallback

        _write_workspace_attr_backup(txn_group, attr_backup_index, "/", h5_file.attrs)

        group_operations: list[dict[str, typing.Any]] = []
        for group_path in sorted(rewrite_map):
            pending_path = f"{pending_root}/{group_path}"
            backup_path = f"{backup_root}/{group_path}"
            group_operations.append(
                {
                    "group_path": group_path,
                    "pending_path": pending_path,
                    "backup_path": backup_path,
                    "old_exists": _h5_path_exists(h5_file, group_path),
                }
            )
        txn_group.attrs["operations"] = json.dumps(
            {"group_replacements": group_operations}
        )
        h5_file.flush()
    return group_operations, attr_updates_to_write


def _write_workspace_transaction_pending_groups(
    fname: str | os.PathLike[str],
    rewrite_map: Mapping[str, tuple[str, dict[str, xr.Dataset]]],
    pending_root: str,
    *,
    compression_mode: WorkspaceCompressionMode | None = None,
) -> None:
    try:
        for group_path, (rewrite_group_path, constructor) in sorted(
            rewrite_map.items()
        ):
            pending_group_path = f"{pending_root}/{group_path}"
            _write_workspace_constructor_groups_to_pending(
                fname,
                constructor,
                rewrite_group_path,
                pending_group_path,
                compression_mode=compression_mode,
            )
    except Exception:
        with _open_workspace_h5_file_for_update(fname) as h5_file:
            _delete_h5_path(h5_file, pending_root)
            h5_file.flush()
        raise


def _commit_workspace_transaction(
    fname: str | os.PathLike[str],
    txn_path: str,
    group_operations: Iterable[Mapping[str, typing.Any]],
    attr_updates: Iterable[
        tuple[str, dict[str, typing.Any], tuple[str, dict[str, xr.Dataset]]]
    ],
    root_attrs: Mapping[str, typing.Any],
) -> None:
    with _open_workspace_h5_file_for_update(fname) as h5_file:
        _set_workspace_transaction_status(h5_file, txn_path, "committing")
        for operation in group_operations:
            group_path = typing.cast("str", operation["group_path"])
            pending_path = typing.cast("str", operation["pending_path"])
            backup_path = typing.cast("str", operation["backup_path"])
            if not _h5_path_exists(h5_file, pending_path):
                raise KeyError(
                    f"Workspace pending group {pending_path!r} was not written"
                )
            if _h5_path_exists(h5_file, group_path):
                _move_h5_path(h5_file, group_path, backup_path)
            _move_h5_path(h5_file, pending_path, group_path)

        for payload_path, attrs, _fallback in attr_updates:
            target_attrs = _workspace_txn_attr_target(h5_file, payload_path)
            if target_attrs is not None:
                _replace_h5_attrs(target_attrs, attrs)

        _write_root_attrs_to_open_workspace_file(h5_file, root_attrs)
        _set_workspace_transaction_status(h5_file, txn_path, "committed")


def _write_workspace_transaction_file(
    fname: str | os.PathLike[str],
    rewrite_groups: Iterable[tuple[str, dict[str, xr.Dataset]]],
    attr_updates: Iterable[
        tuple[str, dict[str, typing.Any], tuple[str, dict[str, xr.Dataset]]]
    ],
    root_attrs: Mapping[str, typing.Any],
    *,
    compression_mode: WorkspaceCompressionMode | None = None,
) -> None:
    _recover_workspace_transactions(fname)
    rewrite_map = {
        group_path.strip("/"): (group_path, constructor)
        for group_path, constructor in rewrite_groups
    }
    attr_updates_tuple = tuple(attr_updates)
    txn_id = uuid.uuid4().hex
    txn_path = f"{_WORKSPACE_TRANSACTION_GROUP_PREFIX}{txn_id}"
    pending_root = f"{_WORKSPACE_PENDING_GROUP_PREFIX}{txn_id}"
    backup_root = f"{_WORKSPACE_BACKUP_GROUP_PREFIX}{txn_id}"
    group_operations, attr_updates_to_write = _prepare_workspace_transaction(
        fname,
        txn_path,
        pending_root,
        backup_root,
        rewrite_map,
        attr_updates_tuple,
        root_attrs,
    )
    try:
        _write_workspace_transaction_pending_groups(
            fname,
            rewrite_map,
            pending_root,
            compression_mode=compression_mode,
        )
        _commit_workspace_transaction(
            fname,
            txn_path,
            group_operations,
            attr_updates_to_write,
            root_attrs,
        )
    except Exception:
        _recover_workspace_transactions(fname)
        raise
    _recover_workspace_transactions(fname)


def _write_full_workspace_tree_file(
    fname: str | os.PathLike[str],
    tree: xr.DataTree | None,
    root_attrs: Mapping[str, typing.Any],
    *,
    copy_source: str | os.PathLike[str] | None = None,
    copy_groups: Iterable[tuple[str, str, dict[str, typing.Any] | None]] = (),
    compression_mode: WorkspaceCompressionMode | None = None,
) -> None:
    import shutil

    fname = os.fsdecode(fname)
    use_scratch = _workspace_path_is_high_risk(fname)
    tmp_dir: tempfile.TemporaryDirectory[str] | None = None
    destination_tmp: str | None = None
    if use_scratch:
        tmp_dir = tempfile.TemporaryDirectory(prefix="erlab-itws-")
        tmp_fname = str(pathlib.Path(tmp_dir.name) / pathlib.Path(fname).name)
    else:
        tmp_fname = f"{fname}.tmp-{uuid.uuid4().hex}"
    try:
        copied_paths: set[str] = set()
        _xarray.ensure_workspace_hdf5_filters_registered()
        copy_groups_tuple = tuple(copy_groups)
        if use_scratch and _workspace_path_is_likely_network_path(fname):
            if tree is None and copy_source is not None and copy_groups_tuple:
                raise ValueError(
                    "File-level workspace repack cannot run when HDF5 group "
                    "copy reuse is disabled"
                )
            copy_source = None
            copy_groups_tuple = ()
        if copy_source is not None and copy_groups_tuple:
            with (
                _xarray._workspace_file_lock(copy_source),
                h5py.File(copy_source, "r") as source_file,
                h5py.File(tmp_fname, "w") as tmp_file,
            ):
                _write_root_attrs_to_open_workspace_file(
                    tmp_file, root_attrs, replace=True
                )
                for source_path, destination_path, attrs in copy_groups_tuple:
                    destination_path = destination_path.strip("/")
                    if _copy_workspace_h5_group_to_open_file(
                        source_file,
                        tmp_file,
                        source_path,
                        destination_path,
                        attrs,
                    ):
                        copied_paths.add(destination_path)
        else:
            with h5py.File(tmp_fname, "w") as tmp_file:
                _write_root_attrs_to_open_workspace_file(
                    tmp_file, root_attrs, replace=True
                )

        if tree is not None:
            for node in sorted(tree.subtree, key=lambda value: value.path.count("/")):
                group_path = node.path.strip("/")
                if not group_path or group_path in copied_paths:
                    continue
                ds = node.to_dataset(inherit=False)
                if ds.variables or ds.attrs:
                    _write_workspace_dataset_group_to_file(
                        tmp_fname,
                        group_path,
                        ds,
                        compression_mode=compression_mode,
                    )

        _validate_workspace_h5_file(tmp_fname)
        _fsync_file(tmp_fname)
        if use_scratch:
            try:
                os.replace(tmp_fname, fname)
            except OSError as err:
                if err.errno != errno.EXDEV:
                    raise
                destination_tmp = f"{fname}.tmp-{uuid.uuid4().hex}"
                shutil.copyfile(tmp_fname, destination_tmp)
                _fsync_file(destination_tmp)
                os.replace(destination_tmp, fname)
            _fsync_parent_directory(fname)
        else:
            os.replace(tmp_fname, fname)
            _fsync_parent_directory(fname)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.remove(tmp_fname)
        if destination_tmp is not None:
            with contextlib.suppress(FileNotFoundError):
                os.remove(destination_tmp)
        if tmp_dir is not None:
            tmp_dir.cleanup()


def _validate_workspace_h5_file(fname: str | os.PathLike[str]) -> None:
    with h5py.File(fname, "r") as h5_file:
        if not _workspace_file_is_workspace(h5_file):
            raise ValueError(f"Temporary workspace file is not valid: {fname}")


def _fsync_file(fname: str | os.PathLike[str]) -> None:
    with contextlib.suppress(OSError):
        fd = os.open(fname, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _fsync_parent_directory(fname: str | os.PathLike[str]) -> None:
    if os.name != "posix":
        return
    with contextlib.suppress(OSError):
        fd = os.open(pathlib.Path(fname).parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
