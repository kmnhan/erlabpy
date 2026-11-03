"""Bounded metadata keys and mismatch witnesses for saved Figure inputs."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import pathlib
import typing
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

import erlab
import erlab.interactive._figurecomposer._source_storage as figure_source_storage
import erlab.interactive.imagetool.manager._workspace._arrays as workspace_arrays
import erlab.interactive.imagetool.manager._workspace._format as workspace_format
import erlab.interactive.imagetool.manager._workspace._storage as workspace_storage
import erlab.interactive.imagetool.manager._workspace._store as workspace_store
from erlab.interactive import _persistence_constants
from erlab.interactive.imagetool import _serialization

if typing.TYPE_CHECKING:
    import os
    from collections.abc import Callable, Hashable, Mapping

    from erlab.interactive.imagetool._provenance._model import ScriptInputDataRole

logger = logging.getLogger(__name__)

_METADATA_MAX_BYTES = 16 << 20
_SCALAR_MAX_BYTES = 4096


def _check_dtype(dtype: np.dtype) -> None:
    if (
        dtype.kind not in "biufcSUMm"
        or dtype.hasobject
        or dtype.fields is not None
        or dtype.subdtype is not None
        or dtype.metadata is not None
    ):
        raise TypeError("Figure comparison metadata needs a primitive fixed-size dtype")


def _check_value(value: Any, ancestors: set[int], size: int) -> int:
    """Reject lossy encodings and bound inputs before building their JSON."""
    kind = type(value)
    if kind in {type(None), bool, int, float, complex}:
        size += 32
    elif kind is str:
        size += len(value.encode("ascii", errors="backslashreplace"))
    elif kind is bytes:
        size += len(value)
    elif kind is np.ndarray or isinstance(value, np.generic):
        if kind is not np.ndarray and kind is not value.dtype.type:
            raise TypeError("NumPy scalar subclasses are not cacheable")
        _check_dtype(value.dtype)
        size += value.nbytes
    elif kind in {list, tuple, dict}:
        identity = id(value)
        if identity in ancestors:
            raise ValueError("Cyclic Figure comparison metadata is not cacheable")
        ancestors.add(identity)
        size += 16
        try:
            items = value.items() if kind is dict else enumerate(value)
            for key, item in items:
                if kind is dict:
                    size = _check_value(key, ancestors, size)
                size = _check_value(item, ancestors, size)
        finally:
            ancestors.remove(identity)
    else:
        raise TypeError(f"Unsupported Figure comparison metadata: {kind.__name__}")
    if size > _METADATA_MAX_BYTES:
        raise ValueError("Figure comparison metadata exceeds the cache budget")
    return size


def attrs_fingerprint(attrs: Mapping[Any, Any]) -> bytes:
    """Hash attributes without dropping unsupported values or retaining objects."""
    copied = dict(attrs)
    _check_value(copied, set(), 0)
    encoded = _serialization.encode_attr_value(copied)
    serialized = json.dumps(
        encoded, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    if len(serialized) > _METADATA_MAX_BYTES:
        raise ValueError(
            "Serialized Figure comparison metadata exceeds the cache budget"
        )
    return hashlib.sha256(serialized).digest()


def metadata_fingerprint(data: xr.DataArray) -> bytes:
    """Hash array metadata and bounded coordinate values, excluding intensity data."""
    _check_dtype(data.dtype)
    coordinates = []
    coordinate_bytes = 0
    for name, coordinate in data.coords.items():
        _check_dtype(coordinate.dtype)
        coordinate_bytes += coordinate.nbytes
        if coordinate_bytes > _METADATA_MAX_BYTES:
            raise ValueError("Figure coordinates exceed the comparison cache budget")
        coordinates.append(
            {
                "name": name,
                "dims": coordinate.dims,
                "shape": coordinate.shape,
                "dtype": coordinate.dtype.str,
                "attrs": coordinate.attrs,
                "values": coordinate.values,
            }
        )
    return attrs_fingerprint(
        {
            "name": data.name,
            "dims": data.dims,
            "shape": data.shape,
            "dtype": data.dtype.str,
            "attrs": data.attrs,
            "coordinates": coordinates,
        }
    )


@dataclass(frozen=True)
class _WorkspaceFigureMismatch:
    variable_name: str
    index: tuple[int, ...] | None = None
    value: np.ndarray | None = None
    metadata: bytes | None = None

    def __post_init__(self) -> None:
        if self.metadata is not None and type(self.metadata) is not bytes:
            raise TypeError(
                "A metadata witness must contain immutable fingerprint bytes"
            )
        if (self.index is None) != (self.value is None):
            raise ValueError("A scalar mismatch needs both an index and a value")
        if self.value is not None:
            if type(self.value) is not np.ndarray:
                raise TypeError("A scalar witness must be a plain NumPy array")
            value = np.asarray(self.value)
            _check_dtype(value.dtype)
            if value.ndim != 0 or not 0 < value.nbytes <= _SCALAR_MAX_BYTES:
                raise ValueError(
                    "A mismatch witness must be one small primitive scalar"
                )
            # Immutable bytes prevent a caller from re-enabling WRITEABLE later.
            frozen = np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(())
            object.__setattr__(self, "value", frozen)

    def still_differs(self, current: xr.DataArray) -> bool:
        """Recheck a mismatch against current data without retaining that data."""
        try:
            if self.metadata is not None:
                # Recorded only after a real metadata comparison failed. A
                # strict fingerprint difference alone does not prove inequality.
                return metadata_fingerprint(current) == self.metadata
            if self.index is None or self.value is None:
                return False
            actual = np.asarray(current.variable[self.index].values)
            _check_dtype(actual.dtype)
            if actual.ndim != 0:
                return False
            try:
                return not np.array_equal(self.value, actual, equal_nan=True)
            except TypeError:
                return not np.array_equal(self.value, actual)
        except Exception:
            return False


@dataclass(frozen=True)
class _WorkspaceFigureRepairResult:
    dataset: xr.Dataset | None = None
    mismatches: tuple[_WorkspaceFigureMismatch, ...] | None = None


@dataclass(frozen=True)
class _WorkspaceFigureSource:
    """A comparison source captured without retaining its manager or window."""

    node_uid: str
    snapshot_token: str
    data_role: ScriptInputDataRole
    data: xr.DataArray | None = None
    dataset: xr.Dataset | None = None
    pending: tuple[pathlib.Path, str] | None = None
    attrs: Mapping[Any, Any] | None = None
    name: str | None = None
    source_input_ndim: int | None = None
    source_paths: frozenset[pathlib.Path] = frozenset()


class _WorkspaceFigureRepair:
    """Inspect saved figure inputs using captured data and source descriptions."""

    def __init__(
        self,
        sources: Mapping[str, _WorkspaceFigureSource],
        cache: dict[str, tuple[bytes, tuple[_WorkspaceFigureMismatch, ...]]],
        compare: Callable[..., bool],
    ) -> None:
        self._sources = sources
        self._figure_repair_cache = cache
        self._compare = compare

    @staticmethod
    def _source_data(
        source: _WorkspaceFigureSource,
        opened: dict[tuple[pathlib.Path, str], xr.Dataset],
    ) -> xr.DataArray:
        if source.data is not None:
            return source.data
        dataset = source.dataset
        if dataset is None:
            if source.pending is None:
                raise ValueError("Figure comparison source has no data")
            dataset = opened.get(source.pending)
            if dataset is None:
                dataset = workspace_arrays.open_workspace_dataset(
                    *source.pending, chunks={}
                )
                opened[source.pending] = dataset
        from erlab.interactive.imagetool.manager._workspace._pending import (
            _PendingWorkspacePayloads,
        )

        return _PendingWorkspacePayloads._workspace_imagetool_source_data_from_payload(
            dataset,
            attrs=source.attrs,
            data_role=source.data_role,
            name=source.name,
            source_input_ndim=source.source_input_ndim,
        )

    @staticmethod
    def _close_reference_datasets(
        opened: Mapping[tuple[pathlib.Path, str], xr.Dataset],
    ) -> None:
        for dataset in opened.values():
            with contextlib.suppress(Exception):
                dataset.close()

    def _repair_workspace_figure_dataset(
        self, node_uid: str, ds: xr.Dataset
    ) -> _WorkspaceFigureRepairResult:
        """Inspect references and report only completed unequal comparisons."""
        sources = figure_source_storage.embedded_workspace_sources(ds.attrs)
        if not sources:
            return _WorkspaceFigureRepairResult()
        references = erlab.interactive.utils.ToolWindow._saved_tool_data_references(ds)
        repaired_names: set[str] = set()
        mismatches: list[_WorkspaceFigureMismatch] = []
        complete = True
        reference_datasets: dict[tuple[pathlib.Path, str], xr.Dataset] = {}
        tool_cls = erlab.interactive.utils.ToolWindow
        try:
            for variable_name, source in sources.items():
                if variable_name not in ds or not ds[variable_name].size:
                    complete = False
                    continue
                source_node = self._sources.get(variable_name)
                if source_node is None:
                    complete = False
                    continue
                stored: xr.DataArray | None = None
                try:
                    stored = tool_cls._tool_data_items_from_dataset(
                        ds,
                        source_parent_data=None,
                        reference_resolver=None,
                        variable_names=(variable_name,),
                    )[variable_name]
                    stored = figure_source_storage.restore_workspace_source_name(
                        ds.attrs, source.name, variable_name, stored
                    )
                    current = self._source_data(source_node, reference_datasets)
                    witnesses: list[_WorkspaceFigureMismatch] = []

                    def _remember_mismatch(
                        index: tuple[int, ...],
                        value: typing.Any,
                        *,
                        name: str = variable_name,
                        records: list[_WorkspaceFigureMismatch] = witnesses,
                    ) -> None:
                        records.append(_WorkspaceFigureMismatch(name, index, value))

                    def _remember_metadata_mismatch(
                        *,
                        name: str = variable_name,
                        data: xr.DataArray = current,
                        records: list[_WorkspaceFigureMismatch] = witnesses,
                    ) -> None:
                        records.append(
                            _WorkspaceFigureMismatch(
                                name, metadata=metadata_fingerprint(data)
                            )
                        )

                    if not self._compare(
                        stored,
                        current,
                        on_mismatch=_remember_mismatch,
                        on_metadata_mismatch=_remember_metadata_mismatch,
                    ):
                        if witnesses:
                            mismatches.extend(witnesses)
                        else:
                            complete = False
                        continue
                except (KeyError, OSError, RuntimeError, TypeError, ValueError):
                    complete = False
                    logger.debug(
                        "Could not compare saved figure input %s/%s",
                        node_uid,
                        source.name,
                        exc_info=True,
                    )
                    continue
                finally:
                    # Release this decoded blob before decoding the next input.
                    stored = None
                references[variable_name] = {
                    "kind": "manager_node",
                    "source": source.name,
                    "node_uid": source.node_uid,
                    "node_snapshot_token": source_node.snapshot_token,
                    "data_role": source.data_role,
                }
                repaired_names.add(variable_name)
        finally:
            self._close_reference_datasets(reference_datasets)
        if not repaired_names:
            return _WorkspaceFigureRepairResult(
                mismatches=tuple(mismatches) if complete else None
            )
        variables: dict[Hashable, xr.DataArray] = {}
        for name, data in ds.data_vars.items():
            if name in repaired_names:
                variables[name] = erlab.interactive.utils._tool_data_placeholder()
            elif name in references or (
                _persistence_constants.TOOL_DATA_BLOB_NAME_ATTR in data.attrs
            ):
                variables[name] = data.reset_coords(drop=True)
            else:
                variables[name] = data
        repaired = xr.Dataset(variables, attrs=ds.attrs)
        repaired.attrs[_persistence_constants.TOOL_DATA_REFERENCES_ATTR] = json.dumps(
            references
        )
        repaired.set_close(ds.close)
        return _WorkspaceFigureRepairResult(dataset=repaired)

    def _workspace_figure_repair_cache_state(
        self,
        payload: tuple[str | os.PathLike[str], str],
        attrs: Mapping[str, typing.Any],
        *,
        cached: tuple[bytes, tuple[_WorkspaceFigureMismatch, ...]] | None = None,
    ) -> tuple[bytes, bool] | None:
        """Validate an immutable payload and its current comparison inputs."""
        path = pathlib.Path(payload[0]).resolve()
        object_path = "/" + payload[1].strip("/")
        parts = object_path.strip("/").split("/")
        if (
            len(parts) != 2
            or parts[0] != _persistence_constants.WORKSPACE_OBJECTS_GROUP
            or not parts[1]
        ):
            return None
        references: dict[tuple[pathlib.Path, str], xr.Dataset] = {}
        try:
            sources = figure_source_storage.embedded_workspace_sources(attrs)
            if not sources:
                return None
            before = workspace_storage._workspace_publication_state(path)
            store = workspace_store.WorkspaceStore.active(path)
            with workspace_storage._workspace_generation_copy_source(path) as h5_file:
                if store is not None:
                    if (
                        store.closed
                        or store.path != path
                        or store.conflicted
                        or store.recovery_path is not None
                    ):
                        return None
                    store.require_current_path()
                workspace_id = workspace_store.WorkspaceStore._workspace_id_from_file(
                    h5_file
                )
                schema_version = int(
                    h5_file.attrs.get("imagetool_workspace_schema_version", 0)
                )
                immutable = (
                    workspace_format._workspace_schema_uses_immutable_generations(
                        schema_version
                    )
                )
                if workspace_id is None or object_path not in h5_file or not immutable:
                    return None
            # Dask workers can acquire this store lock when reading coordinates or
            # scalar samples. Compute their values outside the HDF5 read session.
            identity: list[typing.Any] = [
                str(path),
                before[1:3] if store is not None else before,
                workspace_id,
                object_path,
                attrs_fingerprint(attrs).hex(),
            ]
            current_arrays = {}
            for variable_name, source in sorted(sources.items()):
                source_node = self._sources.get(variable_name)
                if source_node is None:
                    return None
                current = self._source_data(source_node, references)
                pending = source_node.pending
                binding = (
                    None
                    if pending is None
                    else (str(pathlib.Path(pending[0]).resolve()), pending[1])
                )
                identity.append(
                    (
                        variable_name,
                        source.name,
                        source_node.node_uid,
                        source.data_role,
                        source_node.snapshot_token,
                        binding,
                        metadata_fingerprint(current).hex(),
                    )
                )
                current_arrays[variable_name] = current
            key = hashlib.sha256(
                json.dumps(identity, ensure_ascii=True).encode("ascii")
            ).digest()
            hit = (
                cached is not None
                and cached[0] == key
                and len(cached[1]) == len(sources)
                and all(
                    mismatch.still_differs(current_arrays[mismatch.variable_name])
                    for mismatch in cached[1]
                )
            )
            if store is not None:
                with store.read_session() as h5_file:
                    if store.path != path or store.recovery_path is not None:
                        return None
                    store.require_current_path()
                    if (
                        workspace_store.WorkspaceStore._workspace_id_from_file(h5_file)
                        != workspace_id
                        or object_path not in h5_file
                    ):
                        return None
            elif workspace_storage._workspace_publication_state(path) != before:
                return None
        except (KeyError, OSError, RuntimeError, TypeError, ValueError, RecursionError):
            return None
        else:
            return key, hit
        finally:
            self._close_reference_datasets(references)

    def _repair_saved_workspace_figure_dataset(
        self,
        node_uid: str,
        payload: tuple[str | os.PathLike[str], str],
        attrs: Mapping[str, typing.Any],
    ) -> xr.Dataset | None:
        """Inspect saved inputs without restoring a hidden figure window."""
        cached = self._figure_repair_cache.get(node_uid)
        state = self._workspace_figure_repair_cache_state(payload, attrs, cached=cached)
        if state is not None and state[1]:
            return None
        self._figure_repair_cache.pop(node_uid, None)
        opened: xr.Dataset | None = None
        try:
            # Do not retain encoded blobs after each comparison releases its input.
            opened = workspace_arrays.open_workspace_dataset(
                payload[0], payload[1], chunks=None, cache=False
            )
            captured = workspace_arrays._read_workspace_dataset_group_h5py(
                payload[0],
                payload[1],
                preferred_data_name=_persistence_constants.SAVED_TOOL_DATA_NAME,
                data_variables=opened.variables,
            )
            if captured is None:
                captured = workspace_format._restore_workspace_dataset_attrs(opened)
            captured.attrs = dict(attrs)
            result = self._repair_workspace_figure_dataset(node_uid, captured)
            repaired = result.dataset
            if repaired is None:
                if state is not None and result.mismatches:
                    entry = state[0], result.mismatches
                    verified = self._workspace_figure_repair_cache_state(
                        payload, attrs, cached=entry
                    )
                    if verified is not None and verified[1]:
                        self._figure_repair_cache[node_uid] = entry
                        if len(self._figure_repair_cache) > 64:
                            self._figure_repair_cache.pop(
                                next(iter(self._figure_repair_cache))
                            )
                return None
            repaired.set_close(opened.close)
            opened = None
        except (KeyError, OSError, RuntimeError, TypeError, ValueError):
            logger.debug(
                "Could not inspect saved figure inputs for %s", node_uid, exc_info=True
            )
            return None
        else:
            return repaired
        finally:
            if opened is not None:
                opened.close()
