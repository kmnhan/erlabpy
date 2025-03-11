"""Backend for Igor Pro files."""

from __future__ import annotations

__all__ = ["IgorBackendEntrypoint", "load_experiment", "load_igor_hdf5", "load_wave"]

import os
import typing

import h5netcdf
import igor2.binarywave
import igor2.packed
import igor2.record
import numpy as np
import xarray as xr
from xarray.backends import BackendEntrypoint

import erlab

if typing.TYPE_CHECKING:
    from collections.abc import Iterable


class IgorBackendEntrypoint(BackendEntrypoint):
    """Backend for Igor Pro files.

    It can open ".pxt", ".pxp", and ".ibw" files using the `igor2` library.

    It also supports loading HDF5 files exported from Igor Pro.

    For more information about the underlying library, visit:
    https://github.com/AFM-analysis/igor2

    """

    description = "Open Igor Pro files (.pxt, .pxp and .ibw) in Xarray"
    url = "https://erlabpy.readthedocs.io/en/stable/generated/erlab.io.igor.html"

    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables: str | Iterable[str] | None = None,
        recursive: bool = False,
    ) -> xr.Dataset:
        if not isinstance(filename_or_obj, str | os.PathLike):
            raise TypeError("filename_or_obj must be a string or a path-like object")
        return _open_igor_ds(
            filename_or_obj, drop_variables=drop_variables, recursive=recursive
        )

    def guess_can_open(self, filename_or_obj) -> bool:
        if isinstance(filename_or_obj, str | os.PathLike):
            _, ext = os.path.splitext(filename_or_obj)
            return ext in {".pxt", ".pxp", ".ibw"}
        return False

    def open_datatree(
        self, filename_or_obj, *, recursive: bool = True, **kwargs
    ) -> xr.DataTree:
        if not isinstance(filename_or_obj, str | os.PathLike):
            raise TypeError("filename_or_obj must be a string or a path-like object")
        return xr.DataTree.from_dict(
            self.open_groups_as_dict(filename_or_obj, recursive=recursive)
        )

    def open_groups_as_dict(
        self, filename_or_obj, *, recursive: bool = True, **kwargs
    ) -> dict[str, xr.Dataset]:
        if not isinstance(filename_or_obj, str | os.PathLike):
            raise TypeError("filename_or_obj must be a string or a path-like object")
        return {
            k: v.to_dataset()
            for k, v in _load_experiment_raw(
                filename_or_obj, recursive=recursive
            ).items()
        }


def _open_igor_ds(
    filename: str | os.PathLike[typing.Any],
    drop_variables: str | Iterable[str] | None = None,
    recursive: bool = False,
) -> xr.Dataset:
    ext = os.path.splitext(filename)[-1]
    if ext.casefold() in {".pxp", ".pxt"}:
        if isinstance(drop_variables, str):
            drop_variables = [drop_variables]
        return load_experiment(filename, ignore=drop_variables, recursive=recursive)

    if ext.casefold() == ".ibw":
        ds = load_wave(filename).to_dataset()
    else:
        ds = load_igor_hdf5(filename)

    if drop_variables is not None:
        ds = ds.drop_vars(drop_variables)
    return ds


def _load_experiment_raw(
    filename: str | os.PathLike,
    folder: str | None = None,
    *,
    prefix: str | None = None,
    ignore: Iterable[str] | None = None,
    recursive: bool = False,
    **kwargs,
) -> dict[str, xr.DataArray]:
    expt = None
    for bo in [">", "=", "<"]:
        try:
            _, expt = igor2.packed.load(filename, initial_byte_order=bo)
            break
        except ValueError:
            continue

    if expt is None:
        raise OSError("Failed to load the experiment file. Please report this issue.")

    if folder is None:
        split_path: list[bytes] = []
    else:
        folder = folder.strip().strip("/")
        split_path = [n.encode() for n in folder.split("/")]

    if ignore is None:
        ignore = set()

    expt = expt["root"]
    for dirname in split_path:
        expt = expt[dirname]

    def _unpack_folders(contents: dict, parent: str = "") -> dict[str, xr.DataArray]:
        # drop: set = set()
        waves: dict[str, xr.DataArray] = {}

        for name, record in contents.items():
            decoded_name = name.decode() if isinstance(name, bytes) else name
            new_name = f"{parent}/{decoded_name}" if parent else decoded_name

            if isinstance(record, igor2.record.WaveRecord):
                if prefix is not None and not decoded_name.startswith(prefix):
                    continue
                if decoded_name in ignore:
                    continue
                waves[new_name] = load_wave(record, **kwargs)

            elif isinstance(record, dict) and recursive:
                waves.update(_unpack_folders(record, new_name))

        return waves

    return _unpack_folders(expt)


def load_experiment(
    filename: str | os.PathLike,
    folder: str | None = None,
    *,
    prefix: str | None = None,
    ignore: Iterable[str] | None = None,
    recursive: bool = False,
    **kwargs,
) -> xr.Dataset:
    """Load waves from an Igor experiment file(`.pxp` and `.pxt`).

    Use :func:`xarray.load_dataset` with ``backend="erlab-igor"`` for consistency.

    Parameters
    ----------
    filename
        The experiment file.
    folder
        Target folder within the experiment, given as a slash-separated string. If
        `None`, defaults to the root.
    prefix
        If given, only include waves with names that starts with the given string.
    ignore
        List of wave names to ignore.
    recursive
        If `True`, includes waves in child directories.
    **kwargs
        Extra arguments to :func:`load_wave`.

    Returns
    -------
    xarray.Dataset
        Dataset containing the waves.

    """
    return xr.Dataset(
        _load_experiment_raw(
            filename,
            folder,
            prefix=prefix,
            ignore=ignore,
            recursive=recursive,
            **kwargs,
        )
    )


def load_igor_hdf5(filename: str | os.PathLike) -> xr.Dataset:
    """Load a HDF5 file exported by Igor Pro into an `xarray.Dataset`.

    Use :func:`xarray.load_dataset` with ``backend="erlab-igor"`` for consistency.

    Parameters
    ----------
    filename
        The path to the file.

    Returns
    -------
    xarray.Dataset
        The loaded data.

    """
    ncf = h5netcdf.File(filename, mode="r", phony_dims="sort")
    ds = xr.open_dataset(xr.backends.H5NetCDFStore(ncf))
    for dv in ds.data_vars:
        wavescale = ds[dv].attrs["IGORWaveScaling"]
        ds = ds.assign_coords(
            {
                dim: wavescale[i + 1, 1]
                + wavescale[i + 1, 0] * np.arange(ds[dv].shape[i])
                for i, dim in enumerate(ds[dv].dims)
            }
        )
    return ds


def load_wave(
    wave: dict | igor2.record.WaveRecord | str | os.PathLike,
    data_dir: str | os.PathLike | None = None,
) -> xr.DataArray:
    """Load a wave from Igor binary format.

    Use :func:`xarray.load_dataarray` with ``backend="erlab-igor"`` for consistency.

    Parameters
    ----------
    wave
        The wave to load. It can be provided as a dictionary, an instance of
        `igor2.record.WaveRecord`, or a string representing the path to the wave file.
    data_dir
        The directory where the wave file is located. This parameter is only used if
        `wave` is a string or `PathLike` object. If `None`, `wave` must be a valid path.

    Returns
    -------
    xarray.DataArray
        The loaded wave.

    Raises
    ------
    ValueError
        If the wave file cannot be found or loaded.
    TypeError
        If the wave argument is of an unsupported type.

    """
    DEFAULT_DIMS = ["W", "X", "Y", "Z"]
    _MAXDIM = 4

    if isinstance(wave, dict):
        wave_dict = wave
    elif isinstance(wave, igor2.record.WaveRecord):
        wave_dict = wave.wave
    else:
        if data_dir is not None:
            wave = os.path.join(data_dir, wave)
        wave_dict = igor2.binarywave.load(wave)

    d = wave_dict["wave"]
    version = wave_dict["version"]
    dim_labels = [""] * _MAXDIM
    bin_header, wave_header = d["bin_header"], d["wave_header"]
    if version <= 3:
        shape = [wave_header["npnts"]] + [0] * (_MAXDIM - 1)
        sfA = [wave_header["hsA"]] + [0] * (_MAXDIM - 1)
        sfB = [wave_header["hsB"]] + [0] * (_MAXDIM - 1)
        # data_units = wave_header["dataUnits"]
        axis_units = [wave_header["xUnits"]]
        axis_units.extend([""] * (_MAXDIM - len(axis_units)))
    else:
        shape = wave_header["nDim"]
        sfA = wave_header["sfA"]
        sfB = wave_header["sfB"]
        if version >= 5:
            # data_units = d["data_units"].decode()
            axis_units = [b"".join(d).decode() for d in wave_header["dimUnits"]]
            units_sizes = bin_header["dimEUnitsSize"]
            sz_cum = 0
            for i, sz in enumerate(units_sizes):
                if sz != 0:
                    axis_units[i] = d["dimension_units"][sz_cum : sz_cum + sz].decode()
                sz_cum += sz
            for i, sz in enumerate(bin_header["dimLabelsSize"]):
                if sz != 0:
                    dim_labels[i] = b"".join(d["labels"][i]).decode()
        else:
            # data_units = d["data_units"].decode()
            axis_units = [d["dimension_units"].decode()]

    coords = {}
    for i, (a, b, c) in enumerate(zip(sfA, sfB, shape, strict=True)):
        if c == 0:
            continue

        dim, unit = dim_labels[i], axis_units[i]

        if dim == "":
            if unit == "":
                dim = DEFAULT_DIMS[i]
            else:
                # If dim is empty, but the unit is not, use the unit as the dim name
                dim, unit = unit, ""

        coords[dim] = np.linspace(b, b + a * (c - 1), c)
        if unit != "":
            coords[dim] = xr.DataArray(coords[dim], dims=(dim,), attrs={"units": unit})

    attrs: dict[str, int | float | str] = {}
    for ln in d.get("note", b"").decode().splitlines():
        if "=" in ln:
            key, value = ln.split("=", maxsplit=1)
            try:
                attrs[key] = int(value)
            except ValueError:
                try:
                    attrs[key] = float(value)
                except ValueError:
                    attrs[key] = value

    return xr.DataArray(
        erlab.utils.array.to_native_endian(d["wData"]),
        dims=coords.keys(),
        coords=coords,
        attrs=attrs,
    ).rename(wave_header["bname"].decode())
