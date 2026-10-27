"""Bounded metadata keys and mismatch witnesses for saved Figure inputs."""

from __future__ import annotations

import hashlib
import json
import typing
from dataclasses import dataclass
from typing import Any

import numpy as np

from erlab.interactive.imagetool import _serialization

if typing.TYPE_CHECKING:
    from collections.abc import Mapping

    import xarray as xr

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
