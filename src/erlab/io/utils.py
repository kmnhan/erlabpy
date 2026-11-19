"""General-purpose I/O utilities."""

__all__ = [
    "get_files",
    "load_hdf5",
    "open_hdf5",
    "read_file_version",
    "save_as_hdf5",
    "save_as_netcdf",
    "showfitsinfo",
]

import contextlib
import functools
import importlib.util
import os
import pathlib
import typing
import warnings
from collections.abc import Callable, Iterable, Sequence

import numpy as np
import numpy.typing as npt
import xarray as xr

import erlab

_FileSource = typing.TypeVar("_FileSource")


def _close_file_handles(callbacks: tuple[Callable[[], None], ...]) -> None:
    with contextlib.ExitStack() as files:
        for callback in callbacks:
            files.callback(callback)


def _take_file_closer(source: object) -> Callable[[], None] | None:
    """Transfer xarray's file closers without retaining unrelated variables."""
    items = (
        source
        if isinstance(source, Sequence) and not isinstance(source, (str, bytes))
        else (source,)
    )
    callbacks = []
    for item in items:
        owners = item.subtree if isinstance(item, xr.DataTree) else (item,)
        for owner in owners:
            if (
                isinstance(owner, (xr.DataArray, xr.Dataset, xr.DataTree))
                and owner._close is not None
            ):
                # open_dataarray likewise transfers Dataset._close to its result.
                callbacks.append(owner._close)
                owner.set_close(None)
    if not callbacks:
        return None
    if len(callbacks) == 1:
        return callbacks[0]

    return functools.partial(_close_file_handles, tuple(callbacks))


def read_file_version(
    path: str | os.PathLike[str],
    read: Callable[[str], _FileSource],
    *,
    select: Callable[[_FileSource], xr.DataArray] | None = None,
    dtype: npt.DTypeLike | None = None,
    device: int,
    inode: int,
    size: int,
    modified_ns: int,
    changed_ns: int,
) -> xr.DataArray:
    """Read the selected array from a recorded version of a local file.

    ImageTool's copied Python passes the original file identity and timestamps.
    The reader receives the absolute path and transfers ownership of its open
    files. If it opens a Dataset, DataTree or sequence, pass a selection
    function that returns the desired DataArray. Closing the returned array
    releases the opened files. All deferred variables must come from this file.
    They verify its version on access. An optional dtype conversion is applied
    after the file checks are attached.

    The file identity is specific to its original filesystem. To analyze a copy
    on another machine, open that copy as a new source. To reproduce the original
    result after a file changes, supply an independent checkpoint of its data.

    Raises
    ------
    OSError
        If the path is missing or the file version has changed.
    """
    from erlab._provenance._checkpoint import FileVersion, _guard_file_variables

    version = FileVersion(
        os.path.abspath(path), device, inode, size, modified_ns, changed_ns
    )
    version.verify()
    try:
        opened = read(version.path)
    except Exception:
        version.verify()
        raise
    close_on_error = _take_file_closer(opened)
    try:
        data = opened if select is None else select(opened)
        if not isinstance(data, xr.DataArray):
            raise TypeError("The reader must select a DataArray.")
        result = _guard_file_variables(data, version)
        if dtype is not None:
            result = result.astype(dtype)
        version.verify()
        result.set_close(close_on_error)
        close_on_error = None
        return result
    finally:
        if close_on_error is not None:
            close_on_error()


def showfitsinfo(path: str | os.PathLike) -> None:  # pragma: no cover
    """Print raw metadata from a ``.fits`` file.

    .. deprecated:: 3.14.0

       Use `astropy.io.fits` directly to handle FITS files.

    Parameters
    ----------
    path
        Local path to ``.fits`` file.

    """
    warnings.warn(
        "`showfitsinfo` is deprecated, use `astropy.io.fits` directly",
        FutureWarning,
        stacklevel=1,
    )

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


def _fix_attr_format(da: xr.DataArray):  # pragma: no cover
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
                    " will be converted to string"
                )
            except TypeError:
                # this is VERY unprobable...
                da = da.assign_attrs({key: ""})
                erlab.utils.misc.emit_user_level_warning(
                    f"The attribute {key} with invalid type {dt} will be removed"
                )
    return da


def open_hdf5(
    filename: str | os.PathLike, **kwargs
) -> xr.DataArray | xr.Dataset:  # pragma: no cover
    """Open data from an HDF5 file saved with `save_as_hdf5`.

    This is a thin wrapper around `xarray.open_dataarray` and `xarray.open_dataset`.

    .. deprecated:: 3.14.0

       Use `xarray.open_dataarray` or `xarray.open_dataset` directly.

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


def load_hdf5(
    filename: str | os.PathLike, **kwargs
) -> xr.DataArray | xr.Dataset:  # pragma: no cover
    """Load data from an HDF5 file saved with `save_as_hdf5`.

    This is a thin wrapper around `xarray.load_dataarray` and `xarray.load_dataset`.

    .. deprecated:: 3.14.0

       Use `xarray.load_dataarray` or `xarray.load_dataset` directly.


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
) -> None:  # pragma: no cover
    """Save data in ``HDF5`` format.

    .. deprecated:: 3.14.0

       Use `xarray.DataArray.to_netcdf` or `xarray.Dataset.to_netcdf` directly. To save
       data in a format compatible with Igor, use :func:`erlab.io.igor.save_wave`.

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

    if igor_compat and not isinstance(data, xr.Dataset):
        singleton_dims: list[str] = [
            str(dim) for dim, size in data.sizes.items() if size == 1
        ]
        if singleton_dims:
            erlab.utils.misc.emit_user_level_warning(
                "Dimensions with length 1 were squeezed before saving with "
                "`igor_compat=True`: " + ", ".join(singleton_dims)
            )
            data = data.squeeze()

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


def save_as_netcdf(
    data: xr.DataArray, filename: str | os.PathLike, **kwargs
) -> None:  # pragma: no cover
    """Save data in ``netCDF4`` format.

    .. deprecated:: 3.14.0

       Use `xarray.DataArray.to_netcdf` or `xarray.Dataset.to_netcdf` directly.

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
    _fix_attr_format(data).to_netcdf(
        filename,
        encoding={var: {"zlib": True, "complevel": 5} for var in data.coords},
        **kwargs,
    )
