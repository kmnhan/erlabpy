"""General-purpose I/O utilities."""

__all__ = [
    "get_files",
    "load_hdf5",
    "load_krax",
    "open_hdf5",
    "save_as_hdf5",
    "save_as_netcdf",
    "showfitsinfo",
]

import importlib.util
import os
import pathlib
import struct
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt
import xarray as xr

import erlab


def showfitsinfo(path: str | os.PathLike) -> None:
    """Print raw metadata from a ``.fits`` file.

    Parameters
    ----------
    path
        Local path to ``.fits`` file.

    """
    if not importlib.util.find_spec("astropy"):
        raise ImportError("`astropy` needs to be installed to handle FITS files")

    from astropy.io import fits

    with fits.open(path, ignore_missing_end=True) as hdul:
        hdul.verify("silentfix+warn")
        hdul.info()
        for i in range(len(hdul)):
            # print(f'\nColumns in {i:d}: {hdul[i].columns.names!r}')
            print(f"\nHeaders in {i:d}:\n{hdul[i].header!r}")


def get_files(
    directory: str | os.PathLike,
    extensions: Iterable[str] | str | None = None,
    contains: str | None = None,
    notcontains: str | None = None,
    exclude: str | Iterable[str] | None = None,
) -> set[pathlib.Path]:
    """Return file names in a directory with the given extension(s).

    Directories are ignored.

    Parameters
    ----------
    directory
        Target directory.
    extensions
        List of extensions to filter for. If not provided, all files are returned.
    contains
        String to filter for in the file names.
    notcontains
        String to filter out of the file names.
    exclude
        Glob patterns to exclude from the search.

    Returns
    -------
    files : set of pathlib.Path
        Set of file objects in the directory.

    """
    files: set[pathlib.Path] = set()

    if isinstance(extensions, str):
        extensions = {extensions}

    dir_path = pathlib.Path(directory)

    excluded: set[pathlib.Path] = set()

    if exclude is not None:
        if isinstance(exclude, str):
            exclude = {exclude}

        for pattern in exclude:
            excluded.update(dir_path.glob(pattern))

    for f in dir_path.iterdir():
        if (
            f.is_dir()
            or (
                extensions is not None
                and (f.suffix == "" or f.suffix not in extensions)
            )
            or (contains is not None and contains not in f.name)
            or (notcontains is not None and notcontains in f.name)
        ):
            continue

        if f not in excluded:
            files.add(f)

    return files


def fix_attr_format(da: xr.DataArray):
    """Discards attributes that are incompatible with the ``netCDF4`` file format.

    Parameters
    ----------
    da : xarray.DataArray
        Target array.

    Returns
    -------
    out : xarray.Dataset object
        Target array with incompatible attributes removed.

    """
    valid_dtypes = ["S1", "i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8", "f4", "f8"]
    for key in da.attrs:
        isValid = 0
        for dt in valid_dtypes:
            isValid += np.array(da.attrs[key]).dtype == np.dtype(dt)
        if not isValid:
            try:
                da = da.assign_attrs({key: str(da.attrs[key])})
                erlab.utils.misc.emit_user_level_warning(
                    f"The attribute {key} with invalid type {dt}"
                    " will be converted to string",
                )
            except TypeError:
                # this is VERY unprobable...
                da = da.assign_attrs({key: ""})
                erlab.utils.misc.emit_user_level_warning(
                    f"The attribute {key} with invalid type {dt} will be removed",
                )
    return da


def open_hdf5(filename: str | os.PathLike, **kwargs) -> xr.DataArray | xr.Dataset:
    """Open data from an HDF5 file saved with `save_as_hdf5`.

    This is a thin wrapper around `xarray.open_dataarray` and `xarray.open_dataset`.

    Parameters
    ----------
    filename
        The path to the HDF5 file.
    **kwargs
        Extra arguments to `xarray.open_dataarray` or `xarray.open_dataset`.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        The opened data.
    """
    kwargs.setdefault("engine", "h5netcdf")
    try:
        return xr.open_dataarray(filename, **kwargs)
    except ValueError:
        return xr.open_dataset(filename, **kwargs)


def load_hdf5(filename: str | os.PathLike, **kwargs) -> xr.DataArray | xr.Dataset:
    """Load data from an HDF5 file saved with `save_as_hdf5`.

    This is a thin wrapper around `xarray.load_dataarray` and `xarray.load_dataset`.

    Parameters
    ----------
    filename
        The path to the HDF5 file.
    **kwargs
        Extra arguments to `xarray.load_dataarray` or `xarray.load_dataset`.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        The loaded data.
    """
    kwargs.setdefault("engine", "h5netcdf")
    try:
        return xr.load_dataarray(filename, **kwargs)
    except ValueError:
        return xr.load_dataset(filename, **kwargs)


def save_as_hdf5(
    data: xr.DataArray | xr.Dataset,
    filename: str | os.PathLike,
    igor_compat: bool = True,
    **kwargs,
) -> None:
    """Save data in ``HDF5`` format.

    Parameters
    ----------
    data
        `xarray.DataArray` to save.
    filename
        Target file name.
    igor_compat
        (*Experimental*) Make the resulting file compatible with Igor's `HDF5OpenFile`
        for DataArrays with up to 4 dimensions. A convenient Igor procedure is `included
        in the repository
        <https://github.com/kmnhan/erlabpy/blob/main/PythonInterface.ipf>`_. Default is
        `True`.
    **kwargs
        Extra arguments to `xarray.DataArray.to_netcdf`: refer to the `xarray`
        documentation for a list of all possible arguments.

    """
    kwargs.setdefault("engine", "h5netcdf")
    kwargs.setdefault("invalid_netcdf", True)

    data = data.copy(deep=True)
    for k, v in data.attrs.items():
        if v is None:
            data = data.assign_attrs({k: "None"})
        if isinstance(v, dict):
            data = data.assign_attrs({k: str(v)})

    if isinstance(data, xr.Dataset) or data.ndim > 4:
        igor_compat = False

    if igor_compat:
        # IGORWaveScaling order: chunk row column layer
        scaling = [[1, 0]]
        for i in range(data.ndim):
            coord: npt.NDArray = np.asarray(data[data.dims[i]].values)
            delta = coord[1] - coord[0]
            scaling.append([delta, coord[0]])
        if data.ndim == 4:
            scaling[0] = scaling.pop(-1)
        data.attrs["IGORWaveScaling"] = scaling

    data.to_netcdf(filename, **kwargs)


def save_as_netcdf(data: xr.DataArray, filename: str | os.PathLike, **kwargs) -> None:
    """Save data in ``netCDF4`` format.

    Discards invalid ``netCDF4`` attributes and produces a warning.

    Parameters
    ----------
    data
        `xarray.DataArray` to save.
    filename
        Target file name.
    **kwargs
        Extra arguments to `xarray.DataArray.to_netcdf`: refer to the `xarray`
        documentation for a list of all possible arguments.

    """
    kwargs.setdefault("engine", "h5netcdf")
    fix_attr_format(data).to_netcdf(
        filename,
        encoding={var: {"zlib": True, "complevel": 5} for var in data.coords},
        **kwargs,
    )


def load_krax(filename: str | os.PathLike):
    """Load MBS deflector maps in ``.krx`` format.

    Adapted to Python from the ``load_krax_FS.ipf`` Igor procedure by Felix Baumberger,
    University of Geneva.
    """

    def read_header(file, pos, size):
        file.seek(pos)
        return file.read(size).decode("utf-8")

    def header_to_dict(header_str):
        out = {}
        for row in header_str.strip().split("\r\n"):
            if row.startswith("DATA:"):
                break
            k, v = row.split("\t")
            try:
                v = float(v)
            except ValueError:
                pass
            else:
                if v.is_integer():
                    v = int(v)
            out[k] = v
        return out

    with open(filename, "rb") as file:
        file.seek(4)
        v0 = struct.unpack("<I", file.read(4))[0]
        is_64bit = v0 == 0

        if is_64bit:
            fmt = "<Q"
            num_bytes = 8
        else:
            fmt = "<I"
            num_bytes = 4

        file.seek(0)
        v1 = struct.unpack(fmt, file.read(num_bytes))[0]
        n_images = v1 // 3

        # File-position of first image
        file.seek(num_bytes)
        image_pos = struct.unpack(fmt, file.read(num_bytes))[0]

        # Parallel detection angle
        file.seek(2 * num_bytes)
        image_sizeY = struct.unpack(fmt, file.read(num_bytes))[0]

        # Energy coordinate
        file.seek(3 * num_bytes)
        image_sizeX = struct.unpack(fmt, file.read(num_bytes))[0]

        # Autodetect header format and get wave scaling from first header
        header_pos = (image_pos + image_sizeX * image_sizeY + 1) * 4
        header = read_header(file, header_pos, 1200)
        header = header[: header.find("DATA:")]
        header_dict = header_to_dict(header)

        if header.startswith("Lines\t"):
            e0 = header_dict.get("Start K.E.", None)
            e1 = header_dict.get("End K.E.", None)
            x0 = header_dict.get("ScaleMin", None)
            x1 = header_dict.get("ScaleMax", None)
            y0 = header_dict.get("MapStartX", None)
            y1 = header_dict.get("MapEndX", None)
        else:
            # Old header format
            e0 = header_dict.get("Start K.E.", None)
            e1 = header_dict.get("End K.E.", None)
            x0 = header_dict.get("XScaleMin", None)
            x1 = header_dict.get("XScaleMax", None)
            y0 = header_dict.get("YScaleMin", None)
            y1 = header_dict.get("YScaleMax", None)

        data = []
        for ii in range(n_images):
            file.seek((ii * 3 + 1) * num_bytes)
            image_pos = struct.unpack(fmt, file.read(num_bytes))[0]

            file.seek(image_pos * 4)
            buffer = np.frombuffer(
                file.read(image_sizeX * image_sizeY * 4), dtype=np.int32
            ).reshape((image_sizeY, image_sizeX))

            header_pos = (image_pos + image_sizeX * image_sizeY + 1) * 4
            file.seek(header_pos)
            header = file.read(1200).decode("utf-8")
            header = header[: header.find("DATA:")]
            header_dict = header_to_dict(header)

            data.append(
                xr.DataArray(
                    buffer,
                    dims=["angles", "energies"],
                    coords={
                        "angles": np.linspace(x0, x1, image_sizeY),
                        "energies": np.linspace(e0, e1, image_sizeX),
                    },
                    attrs=header_dict,
                )
            )

        if n_images == 1:
            return data[0]

        return xr.concat(data, dim="defl_angles").assign_coords(
            defl_angles=np.linspace(y0, y1, n_images)
        )
