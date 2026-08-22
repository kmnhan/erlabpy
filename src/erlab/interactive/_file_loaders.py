"""Built-in file loader specifications shared by interactive tools."""

from __future__ import annotations

import dataclasses
import types
import typing

import xarray as xr

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Mapping


@dataclasses.dataclass(frozen=True, slots=True)
class BuiltinFileLoaderSpec:
    """Describe one built-in callable file loader."""

    id: str
    label: str
    description: str
    name_filter: str
    extensions: frozenset[str]
    load_func: Callable[..., xr.DataArray]
    default_kwargs: Mapping[str, typing.Any] = dataclasses.field(
        default_factory=lambda: types.MappingProxyType({})
    )
    file_dialog: bool = True


BUILTIN_FILE_LOADER_SPECS = (
    BuiltinFileLoaderSpec(
        id="builtin:xarray-hdf5",
        label="HDF5",
        description="Load an HDF5 file.",
        name_filter="xarray HDF5 Files (*.h5)",
        extensions=frozenset({".h5"}),
        load_func=xr.load_dataarray,
        default_kwargs=types.MappingProxyType({"engine": "h5netcdf"}),
    ),
    BuiltinFileLoaderSpec(
        id="builtin:xarray-netcdf",
        label="NetCDF",
        description="Load a NetCDF file.",
        name_filter="NetCDF Files (*.nc *.nc4 *.cdf)",
        extensions=frozenset({".nc", ".nc4", ".cdf"}),
        load_func=xr.load_dataarray,
    ),
    BuiltinFileLoaderSpec(
        id="builtin:xarray-zarr",
        label="Zarr",
        description="Load a Zarr store.",
        name_filter="xarray Zarr Stores (*.zarr)",
        extensions=frozenset({".zarr"}),
        load_func=xr.load_dataarray,
        default_kwargs=types.MappingProxyType({"engine": "zarr"}),
        file_dialog=False,
    ),
    BuiltinFileLoaderSpec(
        id="builtin:igor-binary-wave",
        label="IBW",
        description="Load an Igor Binary Wave.",
        name_filter="Igor Binary Waves (*.ibw)",
        extensions=frozenset({".ibw"}),
        load_func=xr.load_dataarray,
        default_kwargs=types.MappingProxyType({"engine": "erlab-igor"}),
    ),
    BuiltinFileLoaderSpec(
        id="builtin:igor-packed-experiment",
        label="PXT",
        description="Load a single-wave Igor packed experiment.",
        name_filter="Igor Packed Experiment Templates (*.pxt)",
        extensions=frozenset({".pxt"}),
        load_func=xr.load_dataarray,
        default_kwargs=types.MappingProxyType({"engine": "erlab-igor"}),
    ),
)

_BUILTIN_FILE_LOADER_BY_ID = {spec.id: spec for spec in BUILTIN_FILE_LOADER_SPECS}
_BUILTIN_FILE_LOADER_BY_NAME_FILTER = {
    spec.name_filter: spec for spec in BUILTIN_FILE_LOADER_SPECS
}

if len(_BUILTIN_FILE_LOADER_BY_ID) != len(BUILTIN_FILE_LOADER_SPECS):
    raise RuntimeError("Built-in file loader IDs must be unique")
if len(_BUILTIN_FILE_LOADER_BY_NAME_FILTER) != len(BUILTIN_FILE_LOADER_SPECS):
    raise RuntimeError("Built-in file loader filters must be unique")


def builtin_file_loader_for_id(loader_id: str) -> BuiltinFileLoaderSpec | None:
    """Return the built-in specification for a stable loader ID."""
    return _BUILTIN_FILE_LOADER_BY_ID.get(loader_id)


def builtin_file_loader_for_name_filter(
    name_filter: str,
) -> BuiltinFileLoaderSpec | None:
    """Return the built-in specification for a file-dialog filter."""
    return _BUILTIN_FILE_LOADER_BY_NAME_FILTER.get(name_filter)
