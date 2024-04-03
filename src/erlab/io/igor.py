import os

import h5netcdf
import numpy as np
import xarray as xr

try:
    import igor2.binarywave
    import igor2.packed
    import igor2.record

except ImportError:
    import warnings

    warnings.warn(
        "igor2 is not installed. Some functions may not work.",
        ImportWarning,
        stacklevel=1,
    )

__all__ = ["load_experiment", "load_h5", "load_wave", "load_pxp", "load_ibw"]


def _load_experiment_raw(
    filename: str | os.PathLike,
    folder: str | None = None,
    *,
    prefix: str | None = None,
    ignore: list[str] | None = None,
    recursive: bool = False,
    **kwargs: dict,
) -> xr.Dataset:
    if folder is None:
        folder = []
    if ignore is None:
        ignore = []

    for bo in [">", "=", "<"]:
        try:
            _, expt = igor2.packed.load(filename, initial_byte_order=bo)
            break
        except ValueError:
            continue

    waves = {}
    if isinstance(folder, str):
        folder = folder.split("/")
    folder = [n.encode() for n in folder]

    expt = expt["root"]
    for dirname in folder:
        expt = expt[dirname]

    def unpack_folders(expt):
        for name, record in expt.items():
            if isinstance(record, igor2.record.WaveRecord):
                if prefix is not None:
                    if not name.decode().startswith(prefix):
                        continue
                if name.decode() in ignore:
                    continue
                waves[name.decode()] = load_wave(record, **kwargs)
            elif isinstance(record, dict):
                if recursive:
                    unpack_folders(record)

    unpack_folders(expt)
    return waves


def load_experiment(
    filename: str | os.PathLike,
    folder: str | None = None,
    *,
    prefix: str | None = None,
    ignore: list[str] | None = None,
    recursive: bool = False,
    **kwargs: dict,
) -> xr.Dataset:
    """Loads waves from an igor experiment (`.pxp`) file.

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


def load_h5(filename):
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
    wave,  #: dict | igor2.record.WaveRecord | str | os.PathLike,
    data_dir: str | os.PathLike | None = None,
) -> xr.DataArray:
    """Load a wave from Igor binary format.

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

    def get_dim_name(index):
        dim = dim_labels[index]
        unit = axis_units[index]
        if dim == "":
            if unit == "":
                return DEFAULT_DIMS[index]
            else:
                return unit
        elif unit == "":
            return dim
        else:
            return f"{dim} ({unit})"

    dims = [get_dim_name(i) for i in range(_MAXDIM)]
    coords = {
        dims[i]: np.linspace(b, b + a * (c - 1), c)
        for i, (a, b, c) in enumerate(zip(sfA, sfB, shape))
        if c != 0
    }

    attrs = {}
    for ln in d.get("note", "").decode().splitlines():
        if "=" in ln:
            k, v = ln.split("=", 1)
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
            attrs[k] = v

    return xr.DataArray(
        d["wData"], dims=coords.keys(), coords=coords, attrs=attrs
    ).rename(wave_header["bname"].decode())


load_pxp = load_experiment
load_ibw = load_wave

# from arpes import load_pxt
# import igor.igorpy


# def load_pxp_old(filename, recursive=False, silent=False, **kwargs):
#     expt = load_pxt.read_experiment(filename, **kwargs)
#     waves = {}

#     def process_wave(arr):
#         arr = arr.where(arr != 0)
#         for d in arr.dims:
#             arr = arr.sortby(d)
#         return arr

#     def unpack_folders(expt):
#         for e in expt:
#             try:
#                 arr = process_wave(load_pxt.wave_to_xarray(e))
#                 waves[arr.name] = arr
#                 if not silent:
#                     print(arr.name)
#             except AttributeError:
#                 pass
#             if recursive and isinstance(e, igor.igorpy.Folder):
#                 unpack_folders(e)

#     unpack_folders(expt)
#     return waves


# def load_ibw_old(filename, data_dir: str | os.PathLike | None = None):
#     try:
#         filename = find_first_file(filename, data_dir=data_dir)
#     except (ValueError, TypeError):
#         if data_dir is not None:
#             filename = os.path.join(data_dir, filename)

#     class ibwfile_wave(object):
#         def __init__(self, fname):
#             self.wave = load_pxt.read_single_ibw(fname)

#     return load_pxt.wave_to_xarray(igor.igorpy.Wave(ibwfile_wave(filename)))
