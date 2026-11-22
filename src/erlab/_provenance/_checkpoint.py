"""Own incoming data and verify the backing file of a lazy checkpoint."""

from __future__ import annotations

import copy
import inspect
import mmap
import os
import stat
import typing
import weakref
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
from xarray.core.indexing import PandasIndexingAdapter
from xarray.indexes import PandasIndex

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterator, Mapping


@dataclass(frozen=True)
class FileVersion:
    """Identity of the regular file that backs all deferred reads of an input."""

    path: str
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int

    @classmethod
    def capture(cls, path: str | os.PathLike[str]) -> FileVersion:
        """Capture the version before opening the array or scheduling any reads."""
        path = os.path.abspath(path)
        return cls.from_stat(path, os.stat(path))

    @classmethod
    def from_stat(cls, path: str, info: os.stat_result) -> FileVersion:
        """Capture the version at an absolute path from its open file handle."""
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(
                f"A versioned provenance input needs a regular file: {path}"
            )
        return cls(
            path,
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )

    def verify(self) -> None:
        try:
            observed = FileVersion.capture(self.path)
        except (OSError, ValueError) as exc:
            raise OSError(
                f"Recorded input {self.path!r} is unavailable. Restore the file or "
                "use a saved checkpoint of the original data."
            ) from exc
        if self != observed:
            raise OSError(
                f"Recorded input {self.path!r} has changed. Use a saved checkpoint of "
                "the original data, or reload its source to use the current file."
            )


def _buffer_owner(array: object) -> object:
    while isinstance(array, np.ndarray | memoryview):
        if isinstance(array, memoryview):
            array = array.obj
        elif type(array) is _OwnedBytesArray:
            if type(array.base) is bytes and not array.flags.writeable:
                return array
            array = array.base
        elif array.base is not None:
            array = array.base
        else:
            break
    return array


def _checked_chunk(block: np.ndarray, *, backing: FileVersion) -> np.ndarray:
    # Detach a mapped view before validating the bytes we will return.
    if isinstance(_buffer_owner(block), mmap.mmap):
        block = block.copy()
    backing.verify()
    return block


class _FileVariable:
    """Expose an unchunked xarray backend to Dask without loading the variable."""

    def __init__(self, source: typing.Any, backing: FileVersion) -> None:
        self.variable = source
        self.backing = backing
        self.shape = source.shape
        self.dtype = source.dtype
        self.ndim = source.ndim

    def __getitem__(self, key: tuple[slice, ...]) -> np.ndarray:
        self.backing.verify()
        return _checked_chunk(np.asarray(self.variable[key]), backing=self.backing)


def _owned_categories(values: pd.Categorical) -> pd.Categorical:
    return pd.Categorical.from_codes(
        values.codes.copy(), _owned_index(values.categories), ordered=values.ordered
    )


def _owned_index(index: pd.Index) -> pd.Index:
    # A deep pandas Index copy still shares Python objects in the index.
    if isinstance(index, pd.MultiIndex):
        return index.set_levels(
            [_owned_index(level) for level in index.levels], verify_integrity=False
        )
    if isinstance(index, pd.CategoricalIndex):
        return pd.CategoricalIndex(_owned_categories(index.array), name=index.name)
    if index.dtype != np.dtype(object):
        # The categorical codes and their labels have independent storage.
        return index.copy(deep=True)
    values = copy.deepcopy(index.to_numpy())
    if type(index) is pd.Index:
        return pd.Index(values, dtype=index.dtype, name=index.name, tupleize_cols=False)
    return type(index)(values, name=index.name)


def _own_index_data(
    data: xr.DataArray, *, copy_object_chunks: bool = False
) -> xr.DataArray:
    """Detach pandas index members and xarray's linked coordinate metadata."""
    variable = data.variable
    coordinates = data.coords.variables
    # Dask blocks with structured dtypes may also contain Python objects.
    has_objects = variable.dtype.kind in {"O", "V"}
    for coord in coordinates.values():
        has_objects |= coord.dtype.kind in {"O", "V"}
        if isinstance(coord, xr.IndexVariable):
            # Xarray recreates coordinate variables from their index and reuses
            # nested metadata even when the DataArray copy requests deep=True.
            if coord.attrs:
                coord.attrs = copy.deepcopy(coord.attrs)
            if coord.encoding:
                coord.encoding = copy.deepcopy(coord.encoding)
    if not has_objects:
        return data
    for item in (variable, *coordinates.values()):
        if copy_object_chunks and item.chunks is not None and item.dtype.hasobject:
            # Dask's final NumPy copy of a computed object array is shallow.
            array = item.data
            item.data = array.map_blocks(
                copy.deepcopy, dtype=array.dtype, meta=array._meta
            )
        if item.dtype.kind != "O" or isinstance(item, xr.IndexVariable):
            continue
        if isinstance(item.dtype, pd.CategoricalDtype):
            item.data = _owned_categories(item.data)
        elif isinstance(item._data, PandasIndexingAdapter):
            # drop_indexes can leave the pandas backing on an ordinary Variable.
            item.data = copy.deepcopy(item.data)
    if isinstance(variable, xr.IndexVariable) and variable.dtype.kind == "O":
        variable = variable.copy(deep=False, data=_owned_index(variable.to_index()))
    if not any(
        isinstance(coord, xr.IndexVariable) and coord.dtype.kind == "O"
        for coord in coordinates.values()
    ):
        return data if variable is data.variable else data._replace(variable)
    coords = dict(coordinates)
    indexes = dict(data.xindexes)
    for name, coord in coordinates.items():
        if (
            name not in indexes
            and isinstance(coord, xr.IndexVariable)
            and coord.dtype.kind == "O"
        ):
            coords[name] = coord.copy(deep=False, data=_owned_index(coord.to_index()))
    for index, variables in data.xindexes.group_by_index():
        if isinstance(index, PandasIndex) and any(
            coord.dtype.kind == "O" for coord in variables.values()
        ):
            owned = index._replace(_owned_index(index.to_pandas_index()))
            coords.update(owned.create_variables(variables))
            indexes.update(dict.fromkeys(variables, owned))
    return data._replace(variable, coords, indexes=indexes)


def editable_data(data: xr.DataArray) -> xr.DataArray:
    """Give the caller independent values, coordinates, and nested metadata."""
    return _own_index_data(data.copy(deep=True), copy_object_chunks=True)


# Used only while adopting several live views of the same external arrays.
_BufferMemo = dict[
    tuple[int, np.dtype, tuple[int, ...], tuple[int, ...]],
    tuple[np.ndarray, np.ndarray],
]


class _OwnedBytesArray(np.ndarray):
    """A privately captured bytes buffer, retained as the base of public views."""

    def __new__(
        cls, values: np.ndarray, order: typing.Literal["F", "C"]
    ) -> typing.Self:
        values = np.asarray(values)
        return super().__new__(
            cls, values.shape, values.dtype, buffer=values.tobytes(order), order=order
        )

    def __reduce_ex__(self, protocol: typing.SupportsIndex, /) -> typing.Any:
        # NumPy can unpickle writable arrays over bytes. Restore a plain ndarray
        # so that the graph captures it again at its next admission.
        return np.asarray(self).__reduce_ex__(protocol)


# Small captures are faster as bytes. Large numeric buffers can reuse an exact-sized
# NumPy allocation without the Python bytes header moving it into a larger size class.
_NUMERIC_BUFFER_MIN_BYTES = 1024**2


@dataclass(frozen=True, slots=True, eq=False)
class _ReadOnlyBuffer:
    """Own an isolated numeric allocation and expose its read-only array interface."""

    _root: np.ndarray

    def __init__(self, values: np.ndarray) -> None:
        # Export a byte view of our sole private allocation, including for non-native
        # dtypes. DLPack keeps it alive without exposing the writable producer.
        flat = np.asarray(values).flatten(order="A").view(np.uint8)
        root = np.from_dlpack(flat)
        root.setflags(write=False)
        object.__setattr__(self, "_root", root)

    @property
    def __array_interface__(self) -> dict[str, typing.Any]:
        return self._root.__array_interface__

    def __reduce__(self) -> tuple[type[_ReadOnlyBuffer], tuple[np.ndarray]]:
        # An unpickled owner must acquire its own immutable storage, too.
        return type(self), (self._root,)


def _immutable_values(
    values: np.ndarray, memo: _BufferMemo | None = None
) -> np.ndarray:
    """Share private storage or capture an external array once."""
    owner = _buffer_owner(values)
    if type(owner) in (_OwnedBytesArray, _ReadOnlyBuffer):
        return values.view()
    if memo is not None:
        # Pandas and xarray may return a new ndarray header for the same view.
        layout = (values.ctypes.data, values.dtype, values.shape, values.strides)
        if (saved := memo.get(layout)) is not None:
            return saved[1].view(values.dtype)
    order: typing.Literal["F", "C"] = "F" if values.flags.f_contiguous else "C"
    if values.nbytes >= _NUMERIC_BUFFER_MIN_BYTES and values.dtype.kind in "biufc":
        buffer = np.asarray(_ReadOnlyBuffer(values))
        result = np.frombuffer(buffer, dtype=values.dtype).reshape(
            values.shape, order=order
        )
    else:
        result = np.asarray(_OwnedBytesArray(values, order))
    if memo is not None:
        # Keep the original alive while its address is in use.
        memo[layout] = (values, result)
    return result


def _protected_variable(variable: xr.Variable, memo: _BufferMemo | None) -> xr.Variable:
    if not variable._in_memory:
        # A Dask Array supports assignment by replacing its graph. Give the caller
        # its own collection object. Keep unchunked backend variables deferred.
        result = variable.copy(deep=variable.chunks is not None)
    else:
        values = variable.data
        if isinstance(values, np.ndarray) and not values.dtype.hasobject:
            result = variable.copy(deep=False, data=_immutable_values(values, memo))
        else:
            result = variable.copy(deep=True)
    result.attrs = copy.deepcopy(variable.attrs)
    result.encoding = copy.deepcopy(variable.encoding)
    return result


def protected_data(
    data: xr.DataArray, *, memo: _BufferMemo | None = None
) -> xr.DataArray:
    """Isolate xarray metadata and indexes and share immutable eager buffers."""
    coords = dict(data.coords.variables)
    indexes = dict(data.xindexes)
    for key in coords.keys() - indexes.keys():
        coords[key] = _protected_variable(coords[key], memo)
    for index, variables in data.xindexes.group_by_index():
        if (
            type(index) is PandasIndex
            and type(index.index) is pd.Index
            and isinstance(index.index.dtype, np.dtype)
            and not index.index.dtype.hasobject
        ):
            idx = index.index
            owned = index._replace(
                pd.Index(
                    _immutable_values(idx.to_numpy(), memo), name=idx.name, copy=False
                )
            )
        else:
            owned = index.copy(deep=True)
        created = owned.create_variables(variables)
        for key, var in created.items():
            # A custom index can return one of its input variables.
            coord = var.copy(deep=False)
            coord.attrs = copy.deepcopy(var.attrs)
            coord.encoding = copy.deepcopy(var.encoding)
            coords[key] = coord
            indexes[key] = owned
    return _own_index_data(
        data._replace(
            _protected_variable(data.variable, memo), coords, indexes=indexes
        ),
        copy_object_chunks=True,
    )


# Dask copies preserve their graph. Keep no file, array or graph alive here.
_guarded_graphs: weakref.WeakValueDictionary[
    tuple[int, str, FileVersion], typing.Any
] = weakref.WeakValueDictionary()


def _guard_variable(
    variable: xr.Variable,
    backing: FileVersion,
    storage_key: str | None = None,
) -> None:
    if variable._in_memory and not isinstance(_buffer_owner(variable.data), mmap.mmap):
        return
    import dask.array as da

    if variable.chunks is None:
        preferred = variable.encoding.get("preferred_chunks") or {}
        if storage_key is None:
            chunks = (
                tuple(preferred.get(dim, "auto") for dim in variable.dims)
                if preferred
                else "auto"
            )
        else:
            chunks = tuple(
                preferred.get(dim, size)
                for dim, size in zip(variable.dims, variable.shape, strict=True)
            )
        array = da.from_array(
            _FileVariable(variable.copy(deep=False), backing),
            chunks=chunks,
            name=False if storage_key is None else "read-file-version-" + storage_key,
            meta=np.empty((0,) * variable.ndim, dtype=variable.dtype),
        )
        _guarded_graphs[id(array.dask), array.name, backing] = array.dask
    else:
        array = variable.data
        if isinstance(array, da.Array):
            key = (id(array.dask), array.name, backing)
            if _guarded_graphs.get(key) is array.dask:
                return
        array = array.map_blocks(
            _checked_chunk, backing=backing, dtype=array.dtype, meta=array._meta
        )
    variable.data = array


def _guard_file_variables(
    data: xr.DataArray,
    backing: FileVersion,
    *,
    storage_key: str | None = None,
    node_path: str = "/",
    data_key: tuple[Hashable, ...] = (),
) -> xr.DataArray:
    if storage_key is not None:
        from dask.base import tokenize

    result = data.copy(deep=False)
    variables = [(data_key, result.variable)]
    variables.extend(
        ((name,), coord) for name, coord in result.coords.variables.items()
    )
    for key, variable in variables:
        _guard_variable(
            variable,
            backing,
            tokenize(storage_key, node_path, key) if storage_key is not None else None,
        )
    return result


def _native_file_options(
    read: Callable[..., typing.Any],
    kwargs: Mapping[str, typing.Any],
    backing: FileVersion | None,
) -> tuple[Mapping[str, typing.Any], str | None]:
    """Open native storage chunks without a Dask layer between the guard and backend."""
    if backing is None or kwargs.get("chunks") not in ({}, "auto"):
        return kwargs, None
    manager = kwargs.get("chunked_array_type")
    if manager is None:
        manager = xr.get_options().get("chunk_manager", "dask")
    if (
        not any(
            read is opener
            for opener in (xr.open_dataset, xr.open_dataarray, xr.open_datatree)
        )
        or manager != "dask"
        or kwargs.get("from_array_kwargs")
        or kwargs.get("inline_array")
        or kwargs.get("overwrite_encoded_chunks")
        or (kwargs.get("backend_kwargs") or {}).get("overwrite_encoded_chunks")
    ):
        return kwargs, None
    from dask.base import tokenize

    if kwargs.get("engine") is None:
        guess = xr.backends.plugins.guess_engine
        # Newer DataTree readers also require group support from their backend.
        options = (
            {"must_support_groups": True}
            if read is xr.open_datatree
            and "must_support_groups" in inspect.signature(guess).parameters
            else {}
        )
        kwargs = {**kwargs, "engine": guess(backing.path, **options)}
    if kwargs["engine"] not in ("h5netcdf", "netcdf4", "scipy"):
        return kwargs, None
    token = tokenize(backing, read, kwargs)
    if kwargs.get("chunks") == "auto":
        from erlab._provenance._file_chunks import _VersionedDaskManager

        return {
            **kwargs,
            "chunked_array_type": _VersionedDaskManager(backing, token),
        }, token
    native = {**kwargs, "chunks": None}
    if native.get("cache") is None:
        native["cache"] = False
    return native, token


def _file_variables(
    opened: xr.DataArray | xr.Dataset | xr.DataTree,
) -> Iterator[tuple[str, tuple[Hashable, ...], xr.Variable]]:
    if isinstance(opened, xr.DataArray):
        yield "/", (), opened.variable
        for name, coord in opened.coords.variables.items():
            yield "/", (name,), coord
        return
    groups = (
        ((node.path, node.to_dataset(inherit=False)) for node in opened.subtree)
        if isinstance(opened, xr.DataTree)
        else (("/", opened),)
    )
    for node_path, dataset in groups:
        for name, variable in dataset.variables.items():
            yield node_path, (name,), variable


def _read_versioned_file(
    read: Callable[..., typing.Any],
    path: str | os.PathLike[str],
    kwargs: Mapping[str, typing.Any],
    backing: FileVersion | None,
) -> typing.Any:
    """Guard variables in a native file before exposing its container to ImageTool."""
    options, token = _native_file_options(read, kwargs, backing)
    opened = read(
        backing.path if token is not None and backing is not None else path, **options
    )
    if token is None or backing is None:
        return opened
    from dask.base import tokenize

    try:
        for node_path, key, variable in _file_variables(opened):
            _guard_variable(
                variable,
                backing,
                tokenize(token, node_path, key),
            )
        backing.verify()
    except BaseException:
        opened.close()
        raise
    return opened


def checkpoint_data(
    data: xr.DataArray, *, backing: FileVersion | None = None
) -> xr.DataArray:
    """Detach external buffers. Keep a declared, versioned file lazy."""
    if not isinstance(data, xr.DataArray):
        raise TypeError("Revision data must be a DataArray")
    if backing is not None:
        backing.verify()
        data = _guard_file_variables(data, backing)
    elif not data.variable._in_memory or any(
        not coord.variable._in_memory for coord in data.coords.values()
    ):
        # A delayed task can close over a mutable buffer. Evaluate all variables
        # together so data and coordinates that share a task only read it once.
        data = data.compute()
    captured = protected_data(data)
    if backing is not None:
        backing.verify()
    return captured
