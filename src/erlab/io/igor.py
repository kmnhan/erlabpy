"""Backend for Igor Pro files."""

from __future__ import annotations

__all__ = [
    "IgorBackendEntrypoint",
    "load_experiment",
    "load_igor_hdf5",
    "load_text",
    "load_wave",
    "save_wave",
    "set_scale",
]

import os
import pathlib
import re
import shlex
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

    import igorwriter
else:
    import lazy_loader as _lazy

    igorwriter = _lazy.load("igorwriter")


_WAVE_PATTERN = re.compile(r"WAVES\S*\s+[\"\']?([^\'\"\v:;]+)[\"\']?")  # 1D wave
_WAVE_SHAPE_PATTERN = re.compile(
    r"WAVES.*/N=\(([\d,]+)\)\s+[\"\']?([^\'\"\v:;]+)[\"\']?"
)  # 2D/3D/4D wave with shape
_SETSCALE_PATTERN = re.compile(r"SetScale\s*((?:/I)|(?:/P))?(.*)")


class IgorBackendEntrypoint(BackendEntrypoint):
    """Backend for Igor Pro files.

    This backend supports ".pxt", ".pxp", and ".ibw" files through `igor2
    <https://github.com/AFM-analysis/igor2>`_.

    Partial support for Igor Pro text files (`.itx`) is also provided, allowing loading
    `.itx` files that contain a single wave and well-defined ``SetScale`` commands.

    HDF5 files exported from Igor Pro are also partially supported.

    """

    description = "Open Igor Pro files (.pxt, .pxp, .ibw, and .itx) in Xarray"
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
            return ext in {".pxt", ".pxp", ".ibw", ".itx"}
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

    match ext.casefold():
        case ".pxp" | ".pxt":
            if isinstance(drop_variables, str):
                drop_variables = [drop_variables]
            return load_experiment(
                filename,
                ignore=drop_variables,
                recursive=recursive,
            )
        case ".itx":
            ds = load_text(filename).to_dataset()
        case ".ibw":
            ds = load_wave(filename).to_dataset()
        case _:
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

    coords: dict[str, np.ndarray | xr.DataArray] = {}
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


def save_wave(darr: xr.DataArray, filename: str | os.PathLike) -> None:
    """Save a wave to an Igor binary file.

    This function saves a single DataArray to an Igor binary file. It is the inverse of
    :func:`load_wave`. Only supports simple 1D, 2D, 3D, and 4D DataArrays without any
    associated coordinates and non-dimensional coordinates.

    Parameters
    ----------
    darr
        The DataArray to save as a wave.
    filename
        The path to the output file.

    Example
    -------
    >>> import xarray as xr
    >>> import erlab
    >>> darr = xr.DataArray([1, 2, 3], dims=["x"], coords={"x": [0, 1, 2]})
    >>> erlab.io.igor.save_wave(darr, "output.ibw")
    """
    wave = igorwriter.IgorWave(
        darr.values, name=darr.name if darr.name is not None else "wave0"
    )

    for dim_name, dim in zip(darr.dims, ("x", "y", "z", "t"), strict=False):
        coord = darr.coords[dim_name].values
        if not erlab.utils.array.is_uniform_spaced(coord):
            raise ValueError(
                "Failed to save the wave because the coordinate for dimension "
                f"'{dim_name}' is not evenly spaced. "
            )

        wave.set_dimscale(
            dim, start=coord[0], delta=coord[1] - coord[0], units=dim_name
        )

    for c in darr.coords:
        if c not in darr.dims:
            # If the coordinate is not a dimension, print a warning
            erlab.utils.misc.emit_user_level_warning(
                f"Coordinate '{c}' is not a DataArray dimension, and will not be "
                "written to the Igor binary wave file."
            )

    wave.set_note("\n".join(f"{k}={v}" for k, v in darr.attrs.items()))
    with pathlib.Path(filename).open("wb") as f:
        wave.save(f)


def _parse_wave_shape(wave_line: str) -> tuple[tuple[int, ...] | None, str]:
    """Get wave shape and name from a line like ``WAVES/S/N=(100,200) 'wave_name'``."""
    m = _WAVE_SHAPE_PATTERN.match(wave_line)

    if m:
        shape: tuple[int, ...] = tuple(int(n) for n in m.group(1).split(","))
        name: str = m.group(2)
        return shape, name

    # Failed to match shape, maybe it's a 1D wave?
    m = _WAVE_PATTERN.match(wave_line)
    if m:
        return None, m.group(1)

    raise ValueError(
        f"Invalid format: {wave_line}, unable to resolve wave shape. "
        "Please check the format."
    )


def _parse_setscale(darr: xr.DataArray, setscale_line: str) -> xr.DataArray:
    """Apply a SetScale command to a DataArray given a command string.

    This function parses the SetScale command string and applies the scale to the
    specified dimension of the DataArray using the `set_scale` function. See the
    documentation of `set_scale` for details on the parameters.

    Parameters
    ----------
    darr : DataArray
        The DataArray to which the scale will be applied.
    setscale_line : str
        The SetScale command string in the format: ``SetScale [/I|/P] dim num1 num2
        [units_str] [wave_name]``

    Returns
    -------
    DataArray
        The DataArray with the scale applied to the specified dimension.
    """
    m = _SETSCALE_PATTERN.match(setscale_line.strip())
    if not m:
        raise ValueError(f"Invalid SetScale line format: {setscale_line}")

    method = typing.cast("typing.Literal['/I', '/P', None]", m.group(1))

    # Split args while respecting quotes
    splitter = shlex.shlex(m.group(2), posix=True)
    splitter.whitespace += ","
    splitter.whitespace_split = True
    args = list(splitter)

    return set_scale(
        darr,
        method,
        typing.cast("typing.Literal['d', 't', 'x', 'y', 'z']", args[0]),
        args[1],
        args[2],
        *args[3:],
    )


def set_scale(
    darr: xr.DataArray,
    method: typing.Literal["/I", "/P", None],
    dim: typing.Literal["d", "t", "x", "y", "z"],
    num1: str | float,
    num2: str | float,
    units_str: str | None = None,
    wave_name: str | None = None,
) -> xr.DataArray:
    """Apply a scale to the specified dimension of the DataArray.

    This is the python equivalent of Igor Pro's `SetScale` command.

    Parameters
    ----------
    darr : DataArray
        The DataArray to which the scale will be applied.
    method : '/I', '/P', or None
        The method of scaling. See the Igor Pro documentation for details.
    dim : 'd', 't', 'x', 'y', or 'z'
        The dimension to which the scale will be applied. For 'd', nothing is done.
    num1 : str or float
        The first number for the scale. If a string, it will be converted to float.
    num2 : str or float
        The second number for the scale. If a string, it will be converted to float.
    units_str : str, optional
        If provided, the dimension will be renamed to this string.
    wave_name : str, optional
        Has no effect in this implementation, but can be used for consistency with Igor
        Pro.

    Returns
    -------
    DataArray
        The DataArray with the scale applied to the specified dimension.

    """
    if dim == "d":
        return darr

    # Convert num1 and num2 to float
    num1, num2 = float(num1), float(num2)

    valid_dims = ("x", "y", "z", "t")
    if dim not in valid_dims:
        raise ValueError(f"Invalid dimension: {dim}. Must be one of {valid_dims}.")
    dim_idx: int = valid_dims.index(dim)

    if len(darr.dims) <= dim_idx:
        # DataArray dimension is smaller than the specified dim
        return darr

    dim_name = darr.dims[dim_idx]

    match method:
        case "/I":
            vals = np.linspace(num1, num2, darr.shape[dim_idx])
        case "/P":
            vals = np.linspace(
                num1, num1 + num2 * (darr.shape[dim_idx] - 1), darr.shape[dim_idx]
            )
        case None:
            vals = np.linspace(num1, num2, darr.shape[dim_idx] + 1)[:-1]

    darr = darr.assign_coords({dim_name: vals})

    if units_str:
        darr = darr.rename({dim_name: units_str})

    return darr


def load_text(
    filename: str | os.PathLike, dtype=float, *, without_values: bool = False
) -> xr.DataArray:
    """Load an `.itx` file containing a *single wave* into a `xarray.DataArray`.

    This function reads basic `.itx` files exported from Igor Pro. Currently, it only
    supports files with a single wave and does not handle complex structures like
    multiple waves or nested structures.

    Parameters
    ----------
    filename
        The path to the `.itx` file.
    dtype
        The data type to use for the values. Defaults to `float`.
    without_values
        If `True`, the returned DataArray values will be filled with zeros. Use this to
        check the coords or attrs quickly without loading in the full data.

    Returns
    -------
    DataArray
        The loaded data.
    """
    comments: dict[str, str] = {}
    setscale_lines: list[str] = []

    shape: tuple[int, ...] | None = None
    wave_name: str | None = None

    skiprows: int | None = None
    max_rows: int | None = None

    with open(filename) as f:
        for i, line in enumerate(f):
            if line.startswith("BEGIN"):
                skiprows = i + 1
                continue
            if line.startswith("END"):
                max_rows = i - typing.cast("int", skiprows)
                continue
            if skiprows is not None and max_rows is None:
                # Data section, skip
                continue
            if line.startswith("X //"):
                # Parse key-value pairs from comment lines
                comment_line: str = line.removeprefix("X //").strip()

                if "=" in comment_line:
                    delim = "="
                elif ":" in comment_line:
                    delim = ":"
                else:
                    continue

                key, val = comment_line.split(delim, 1)
                if not val.strip() and delim == ":":
                    # Header line with no value, skipping
                    continue

                comments[key.strip()] = val.strip()
                continue
            if line.startswith("X SetScale"):
                setscale_lines.extend(line.removeprefix("X ").strip().split(";"))
                continue
            if line.startswith("WAVES"):
                if wave_name is not None:
                    erlab.utils.misc.emit_user_level_warning(
                        "Multiple wave definitions found in the file. "
                        "Only the first one will be loaded."
                    )
                    break
                shape, wave_name = _parse_wave_shape(line)
                continue

    if wave_name is None or skiprows is None or max_rows is None:
        raise ValueError(
            "No valid wave definition found in the file. Check the file format."
        )

    if shape is None:
        # 1D wave
        shape = (max_rows,)

    if without_values:
        arr = np.zeros(shape, dtype=dtype)
    else:
        arr = np.loadtxt(filename, dtype=dtype, skiprows=skiprows, max_rows=max_rows)

        if len(shape) >= 3:
            arr = arr.reshape(shape[-1], *shape[:-1]).transpose(
                *range(1, len(shape)), 0
            )
        else:
            arr = arr.reshape(shape)

    darr = xr.DataArray(arr, name=wave_name.strip(), attrs=comments)

    for setscale_line in setscale_lines:
        darr = _parse_setscale(darr, setscale_line)

    return darr
