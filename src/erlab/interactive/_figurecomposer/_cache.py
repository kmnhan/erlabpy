"""Bounded prepared-data caches for Figure Composer rendering."""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import datetime
import enum
import hashlib
import math
import typing
from collections.abc import Callable, Hashable, Iterator, Mapping, MutableMapping

import numpy as np
import pydantic
import xarray as xr

_DEFAULT_CACHE_MAX_ENTRIES = 32
_DEFAULT_CACHE_MAX_BYTES = 256 * 1024**2

_T = typing.TypeVar("_T")


class _UncacheableValueError(TypeError):
    """Raised when a semantic plan contains a value without a stable encoding."""


def _estimated_nbytes(value: object) -> int:
    if isinstance(value, xr.DataArray | xr.Dataset):
        return int(value.nbytes)
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return sum(
            _estimated_nbytes(getattr(value, field.name))
            for field in dataclasses.fields(value)
        )
    if isinstance(value, pydantic.BaseModel):
        return _estimated_nbytes(value.model_dump(mode="python"))
    if isinstance(value, Mapping):
        return sum(_estimated_nbytes(item) for item in value.values())
    if isinstance(value, list | tuple):
        return sum(_estimated_nbytes(item) for item in value)
    return 0


def _contains_dask_collection(value: object) -> bool:
    import dask.base

    if dask.base.is_dask_collection(value):
        return True
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return any(
            _contains_dask_collection(getattr(value, field.name))
            for field in dataclasses.fields(value)
        )
    if isinstance(value, pydantic.BaseModel):
        return _contains_dask_collection(value.model_dump(mode="python"))
    if isinstance(value, Mapping):
        return any(_contains_dask_collection(item) for item in value.values())
    if isinstance(value, list | tuple):
        return any(_contains_dask_collection(item) for item in value)
    return False


def _persist_dask_value(
    value: typing.Any, *, max_nbytes: int = _DEFAULT_CACHE_MAX_BYTES
) -> typing.Any:
    """Persist a reasonably sized nested dask value while preserving its structure."""
    if _estimated_nbytes(value) > max_nbytes or not _contains_dask_collection(value):
        return value
    import dask

    return dask.persist(value)[0]


def _freeze_cache_value(
    value: typing.Any,
    *,
    _ancestors: frozenset[int] = frozenset(),
) -> Hashable:
    may_recurse = (
        dataclasses.is_dataclass(value) and not isinstance(value, type)
    ) or isinstance(
        value,
        pydantic.BaseModel | np.ndarray | Mapping | list | tuple | frozenset,
    )
    if may_recurse:
        value_id = id(value)
        if value_id in _ancestors:
            raise _UncacheableValueError("render-cache plans cannot contain cycles")
        _ancestors = _ancestors | {value_id}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return (
            type(value).__module__,
            type(value).__qualname__,
            tuple(
                (
                    field.name,
                    _freeze_cache_value(
                        getattr(value, field.name),
                        _ancestors=_ancestors,
                    ),
                )
                for field in dataclasses.fields(value)
            ),
        )
    if isinstance(value, pydantic.BaseModel):
        return (
            type(value).__module__,
            type(value).__qualname__,
            _freeze_cache_value(
                value.model_dump(mode="python"),
                _ancestors=_ancestors,
            ),
        )
    if isinstance(value, slice):
        return (
            "slice",
            _freeze_cache_value(value.start, _ancestors=_ancestors),
            _freeze_cache_value(value.stop, _ancestors=_ancestors),
            _freeze_cache_value(value.step, _ancestors=_ancestors),
        )
    if isinstance(value, np.generic):
        return _freeze_cache_value(value.item(), _ancestors=_ancestors)
    if isinstance(value, enum.Enum):
        return (
            type(value).__module__,
            type(value).__qualname__,
            _freeze_cache_value(value.value, _ancestors=_ancestors),
        )
    if value is None:
        return ("none",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, bytes):
        return ("bytes", value)
    if isinstance(value, datetime.date | datetime.time | datetime.timedelta):
        return (type(value).__module__, type(value).__qualname__, value)
    if isinstance(value, float) and math.isnan(value):
        return ("float", "nan")
    if isinstance(value, float):
        return ("float", value)
    if isinstance(value, complex):
        return ("complex", value)
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            contents: Hashable = tuple(
                _freeze_cache_value(item, _ancestors=_ancestors)
                for item in value.reshape(-1).tolist()
            )
        else:
            contents = hashlib.blake2b(
                np.ascontiguousarray(value).tobytes(),
                digest_size=16,
            ).digest()
        return (
            "ndarray",
            value.dtype.str,
            tuple(value.shape),
            contents,
        )
    if isinstance(value, Mapping):
        return (
            "mapping",
            tuple(
                sorted(
                    (
                        (
                            _freeze_cache_value(key, _ancestors=_ancestors),
                            _freeze_cache_value(item, _ancestors=_ancestors),
                        )
                        for key, item in value.items()
                    ),
                    key=repr,
                )
            ),
        )
    if isinstance(value, list):
        return (
            "list",
            tuple(_freeze_cache_value(item, _ancestors=_ancestors) for item in value),
        )
    if isinstance(value, tuple):
        return (
            "tuple",
            tuple(_freeze_cache_value(item, _ancestors=_ancestors) for item in value),
        )
    if isinstance(value, frozenset):
        return (
            "frozenset",
            tuple(
                sorted(
                    (
                        _freeze_cache_value(item, _ancestors=_ancestors)
                        for item in value
                    ),
                    key=repr,
                )
            ),
        )
    raise _UncacheableValueError(
        f"{type(value).__module__}.{type(value).__qualname__} has no stable "
        "render-cache encoding"
    )


def _render_data_cache_key(stage: str, plan: object) -> tuple[str, Hashable]:
    """Return a stable cache key for a semantic prepared-data plan."""
    return stage, _freeze_cache_value(plan)


class _BoundedCache(MutableMapping[Hashable, typing.Any]):
    """Least-recently-used mapping bounded by entry count and logical bytes."""

    def __init__(
        self,
        *,
        max_entries: int = _DEFAULT_CACHE_MAX_ENTRIES,
        max_bytes: int = _DEFAULT_CACHE_MAX_BYTES,
        persist_dask: bool = False,
    ) -> None:
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._persist_dask = persist_dask
        self._values: collections.OrderedDict[Hashable, tuple[typing.Any, int]] = (
            collections.OrderedDict()
        )
        self._nbytes = 0

    def __getitem__(self, key: Hashable) -> typing.Any:
        value, nbytes = self._values.pop(key)
        self._values[key] = value, nbytes
        return value

    def __setitem__(self, key: Hashable, value: typing.Any) -> None:
        if self._persist_dask:
            value = _persist_dask_value(value, max_nbytes=self._max_bytes)
        nbytes = _estimated_nbytes(value)
        previous = self._values.pop(key, None)
        if previous is not None:
            self._nbytes -= previous[1]
        if nbytes > self._max_bytes:
            return
        self._values[key] = value, nbytes
        self._nbytes += nbytes
        while len(self._values) > self._max_entries or self._nbytes > self._max_bytes:
            _old_key, (_old_value, old_nbytes) = self._values.popitem(last=False)
            self._nbytes -= old_nbytes

    def __delitem__(self, key: Hashable) -> None:
        _value, nbytes = self._values.pop(key)
        self._nbytes -= nbytes

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def clear(self) -> None:
        self._values.clear()
        self._nbytes = 0


class _RenderDataCache:
    """Own prepared render data, invalidation, and render-session safety."""

    def __init__(self) -> None:
        self._values = _BoundedCache(persist_dask=True)
        self._source_revision = -1
        self._enabled = True

    def __len__(self) -> int:
        return len(self._values)

    def sync_source_revision(self, source_revision: int) -> None:
        """Discard entries when the source-payload generation changes."""
        if source_revision != self._source_revision:
            self.invalidate(source_revision)

    def invalidate(self, source_revision: int) -> None:
        """Discard all prepared data and record the active source generation."""
        self._values.clear()
        self._source_revision = source_revision

    @contextlib.contextmanager
    def render_session(
        self,
        *,
        source_revision: int,
        cache_safe: bool,
    ) -> Iterator[None]:
        """Apply one exception-safe cache policy for a complete render."""
        self.sync_source_revision(source_revision)
        previous_enabled = self._enabled
        self._enabled = previous_enabled and cache_safe
        if not self._enabled:
            self._values.clear()
        try:
            yield
        finally:
            self._enabled = previous_enabled
            if not cache_safe:
                self._values.clear()

    def get_or_compute(
        self,
        stage: str,
        plan: object,
        factory: Callable[[], _T],
        *,
        source_revision: int,
    ) -> _T:
        """Return one prepared value, bypassing cache for unsupported plans."""
        self.sync_source_revision(source_revision)
        if not self._enabled:
            return factory()
        try:
            key = _render_data_cache_key(stage, plan)
        except _UncacheableValueError:
            return factory()
        try:
            return typing.cast("_T", self._values[key])
        except KeyError:
            value = factory()
            self._values[key] = value
            return typing.cast("_T", self._values.get(key, value))
