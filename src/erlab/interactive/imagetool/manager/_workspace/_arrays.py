"""Array encoding and HDF5 payload storage for manager workspaces."""

from __future__ import annotations

import contextlib
import os
import pathlib
import threading
import typing

import h5netcdf
import hdf5plugin
import numpy as np
import xarray as xr
from xarray.backends import CachingFileManager, H5NetCDFStore

import erlab
from erlab.interactive.imagetool import _serialization

if typing.TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Iterator, Mapping

    import h5py

    from erlab.interactive._options.schema import WorkspaceCompressionMode
else:
    import lazy_loader as _lazy

    h5py = _lazy.load("h5py")

from erlab.interactive.imagetool.manager._workspace._format import (
    _is_workspace_internal_group_name,
    _restore_workspace_serialized_attrs,
    _sanitize_workspace_attr_names,
    _workspace_file_is_workspace,
    _workspace_serializable_attrs,
)

_WORKSPACE_FILE_LOCKS: dict[str, threading.RLock] = {}
_WORKSPACE_COMPRESSION_MIN_BYTES = 1 << 20  # 1 MiB
_TOOL_DATA_BLOB_NAME_ATTR = _serialization.TOOL_DATA_BLOB_NAME_ATTR
_SAVED_TOOL_DATA_REFERENCE_DIM = _serialization.SAVED_TOOL_DATA_REFERENCE_DIM
_SAVED_TOOL_DATA_BLOB_DIM_PREFIX = _serialization.SAVED_TOOL_DATA_BLOB_DIM_PREFIX
_WORKSPACE_H5PY_DIMENSION_SCALE_ATTRS = frozenset(
    {"CLASS", "NAME", "DIMENSION_LIST", "REFERENCE_LIST"}
)


def _replace_h5_attrs(target_attrs, attrs: Mapping[typing.Any, typing.Any]) -> None:
    for key in list(target_attrs):
        del target_attrs[key]
    for key, value in _workspace_serializable_attrs(attrs).items():
        target_attrs[key] = value


def _normalized_file_path(path: object) -> str | None:
    """Return an absolute normalized path string for path-like values."""
    if not isinstance(path, (str, bytes, os.PathLike)):
        return None
    try:
        path_str = os.fsdecode(path)
    except TypeError:
        return None
    if not path_str:
        return None
    try:
        return str(pathlib.Path(path_str).resolve())
    except OSError:
        return str(pathlib.Path(path_str).absolute())


def _workspace_file_lock(path: str | os.PathLike[str]) -> threading.RLock:
    target = _normalized_file_path(path)
    if target is None:
        target = os.fsdecode(path)
    lock = _WORKSPACE_FILE_LOCKS.get(target)
    if lock is None:
        lock = threading.RLock()
        _WORKSPACE_FILE_LOCKS[target] = lock
    return lock


def _workspace_file_identity(path: str | os.PathLike[str]) -> tuple[str, int, int, int]:
    target = _normalized_file_path(path)
    if target is None:
        target = os.fsdecode(path)
    try:
        stat_result = os.stat(target)
    except OSError:
        return target, 0, 0, 0
    return target, stat_result.st_dev, stat_result.st_ino, stat_result.st_mtime_ns


def _xarray_source_path(value: object) -> str | None:
    if not isinstance(value, (str, bytes, os.PathLike)):
        return None
    return _normalized_file_path(value)


def dataarray_source_paths(data_array: xr.DataArray) -> tuple[str, ...]:
    """Return normalized file sources referenced by a DataArray and its coords."""
    paths: list[str] = []

    def _append_source(value: object) -> None:
        source = _xarray_source_path(value)
        if source is not None and source not in paths:
            paths.append(source)

    _append_source(data_array.encoding.get("source"))
    for coord in data_array.coords.values():
        _append_source(coord.encoding.get("source"))
    return tuple(paths)


def dataarray_is_numpy_backed(data_array: xr.DataArray) -> bool:
    """Return True when a DataArray is already backed by an in-memory ndarray."""
    return isinstance(data_array.variable._data, (np.ndarray, np.generic))


def ensure_workspace_hdf5_filters_registered() -> None:
    """Register HDF5 filters needed by compressed workspace files."""
    hdf5plugin.register(force=False)


def workspace_compression_mode() -> WorkspaceCompressionMode:

    return erlab.interactive.options.model.io.workspace.compression


def _workspace_blosc2_encoding(
    compression_mode: WorkspaceCompressionMode,
) -> dict[str, typing.Any]:
    if compression_mode == "none":
        return {}

    ensure_workspace_hdf5_filters_registered()
    cname: typing.Literal["blosclz", "zstd"]
    match compression_mode:
        case "blosclz3":
            cname = "blosclz"
            clevel = 3
        case "zstd1":
            cname = "zstd"
            clevel = 1
        case _:
            raise ValueError(f"Unknown workspace compression mode: {compression_mode}")

    return dict(
        hdf5plugin.Blosc2(
            cname=cname,
            clevel=clevel,
            filters=hdf5plugin.Blosc2.SHUFFLE,
        )
    )


def _resolve_workspace_compression_mode(
    *,
    compression_mode: WorkspaceCompressionMode | None,
    compress: bool | None,
) -> WorkspaceCompressionMode:
    if compression_mode is not None:
        return compression_mode
    if compress is False:
        return "none"
    if compress is True:
        return "zstd1"
    return workspace_compression_mode()


def _should_compress_workspace_variable(
    variable: xr.Variable, *, min_bytes: int
) -> bool:
    if variable.dtype.kind not in "iufc":
        return False
    return int(variable.nbytes) >= min_bytes


def _workspace_chunksizes_for_dataarray(
    data_array: xr.DataArray,
) -> tuple[int, ...] | None:
    """Return a valid fixed HDF5 chunk shape for dask-backed workspace data."""
    chunks = data_array.chunks
    if chunks is None or data_array.ndim == 0:
        return None

    chunksizes: list[int] = []
    for size, dim_chunks in zip(data_array.shape, chunks, strict=True):
        if size <= 0 or len(dim_chunks) == 0:
            return None
        first = int(dim_chunks[0])
        if first <= 0:
            return None
        chunksizes.append(min(first, int(size)))
    return tuple(chunksizes)


def workspace_dataset_encoding(
    ds: xr.Dataset,
    *,
    min_bytes: int = _WORKSPACE_COMPRESSION_MIN_BYTES,
    compress: bool | None = None,
    compression_mode: WorkspaceCompressionMode | None = None,
) -> dict[Hashable, dict[str, typing.Any]]:
    """Return h5netcdf encodings for workspace data variables."""
    compression_mode = _resolve_workspace_compression_mode(
        compression_mode=compression_mode, compress=compress
    )
    compression_encoding = _workspace_blosc2_encoding(compression_mode)

    encoding: dict[Hashable, dict[str, typing.Any]] = {}
    for name, data_array in ds.data_vars.items():
        var_encoding: dict[str, typing.Any] = {}
        chunksizes = _workspace_chunksizes_for_dataarray(data_array)
        if chunksizes is not None:
            var_encoding["chunksizes"] = chunksizes
        if compression_encoding and _should_compress_workspace_variable(
            data_array.variable, min_bytes=min_bytes
        ):
            var_encoding.update(compression_encoding)
        if var_encoding:
            encoding[name] = var_encoding
    return encoding


class WorkspaceFileManager(CachingFileManager):
    """xarray file manager for manager-owned workspace readers."""

    def __init__(self, path: str | os.PathLike[str]) -> None:

        ensure_workspace_hdf5_filters_registered()
        target = _normalized_file_path(path)
        if target is None:
            target = os.fsdecode(path)
        identity = _workspace_file_identity(target)
        super().__init__(
            h5netcdf.File,
            target,
            mode="r+",
            kwargs={
                "invalid_netcdf": None,
                "phony_dims": "sort",
                "decode_vlen_strings": True,
            },
            lock=_workspace_file_lock(target),
            # xarray documents manager_id as a test/dependency-injection hook,
            # but the manager needs repeated workspace readers for the same
            # file identity to share one cached h5netcdf handle.
            manager_id=("erlab-workspace", *identity, "r+"),
        )
        self.workspace_path = target


def _iter_h5netcdf_group_paths(group: object, path: str = "/") -> Iterator[str]:
    yield path
    groups = getattr(group, "groups", {})
    for name, child in groups.items():
        child_path = f"/{name}" if path == "/" else f"{path}/{name}"
        yield from _iter_h5netcdf_group_paths(child, child_path)


def _open_workspace_dataset_from_manager(
    file_manager: WorkspaceFileManager,
    group: str,
    *,
    chunks: typing.Any,
) -> xr.Dataset:
    store = H5NetCDFStore(
        file_manager,
        group=group,
        mode="r+",
        lock=_workspace_file_lock(file_manager.workspace_path),
        autoclose=False,
    )
    if chunks is None:
        return xr.open_dataset(store)
    return xr.open_dataset(store, chunks=chunks)


def open_workspace_dataset(
    path: str | os.PathLike[str],
    group: str,
    *,
    chunks: typing.Any,
) -> xr.Dataset:
    """Open a workspace group through the manager-owned file manager."""
    target = _normalized_file_path(path)
    if target is None:
        target = os.fsdecode(path)
    return _open_workspace_dataset_from_manager(
        WorkspaceFileManager(target), group, chunks=chunks
    )


def open_workspace_datatree(
    path: str | os.PathLike[str], *, chunks: typing.Any
) -> xr.DataTree:
    """Open a workspace tree through the manager-owned file manager."""
    target = _normalized_file_path(path)
    if target is None:
        target = os.fsdecode(path)
    file_manager = WorkspaceFileManager(target)
    with file_manager.acquire_context() as h5_file:
        group_paths = tuple(_iter_h5netcdf_group_paths(h5_file))

    groups: dict[str, xr.Dataset] = {}
    try:
        for group_path in group_paths:
            groups[group_path] = _open_workspace_dataset_from_manager(
                file_manager, group_path, chunks=chunks
            )
        tree = xr.DataTree.from_dict(groups)
    except Exception:
        for ds in groups.values():
            ds.close()
        raise

    for group_path, ds in groups.items():
        tree[group_path].set_close(ds.close)
    return tree


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
    ensure_workspace_hdf5_filters_registered()
    with _workspace_file_lock(fname), h5py.File(fname, "r") as h5_file:
        if not _workspace_file_is_workspace(h5_file):
            raise ValueError("Not a valid workspace file")
        return _h5py_attrs_to_dict(h5_file.attrs)


def _workspace_live_root_group_copy_groups(
    fname: str | os.PathLike[str],
) -> tuple[tuple[str, str, dict[str, typing.Any] | None], ...]:
    ensure_workspace_hdf5_filters_registered()
    with _workspace_file_lock(fname), h5py.File(fname, "r") as h5_file:
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
    ensure_workspace_hdf5_filters_registered()
    with _workspace_file_lock(fname), h5py.File(fname, "r") as h5_file:
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
    ensure_workspace_hdf5_filters_registered()
    with _workspace_file_lock(fname), h5py.File(fname, "r") as h5_file:
        if not _workspace_file_is_workspace(h5_file):
            raise ValueError("Not a valid workspace file")
        for name, item in h5_file.items():
            if isinstance(item, h5py.Group) and not _is_workspace_internal_group_name(
                name
            ):
                total += _workspace_h5_object_storage_size(item)
    return total


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


def _workspace_h5py_blosc2_options_match(
    actual_options: tuple[int, ...], expected_options: tuple[int, ...]
) -> bool:
    if actual_options == expected_options:
        return True
    if len(actual_options) < 7 or len(expected_options) < 7:
        return False
    return actual_options[4:7] == expected_options[4:7]


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
    if expected_options is None:
        return True
    expected_options = tuple(expected_options)
    if int(expected_filter) == hdf5plugin.Blosc2.filter_id:
        return _workspace_h5py_blosc2_options_match(actual_options, expected_options)
    return actual_options == expected_options


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
    encoding = workspace_dataset_encoding(ds, compression_mode=compression_mode)
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


def _h5_group_matches_compression(
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
    compression_encoding = _workspace_blosc2_encoding(compression_mode)
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
            >= _WORKSPACE_COMPRESSION_MIN_BYTES
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
    ensure_workspace_hdf5_filters_registered()
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
    with _workspace_file_lock(fname), h5py.File(fname, "r") as h5_file:
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
        data_values = np.asarray(data_dataset[()])
        data_vars: dict[typing.Hashable, typing.Any] = {
            data_name: (
                tuple(dims),
                data_values,
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

    ensure_workspace_hdf5_filters_registered()
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
                encoding = workspace_dataset_encoding(ds)
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
    encoding = workspace_dataset_encoding(ds, compression_mode=compression_mode)
    if lock_path is not None:
        normalized_lock_path = _normalized_file_path(lock_path)
        if normalized_lock_path is not None and any(
            normalized_lock_path in dataarray_source_paths(data_array)
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
        _workspace_file_lock(lock_path)
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
