"""Utilities for reading NeXus files into xarray objects.

This module provides functions that can be used to extract coordinates and data from
NeXus files and convert them into xarray DataArrays conveniently. All functions in this
module require the `nexusformat <https://github.com/nexpy/nexusformat>`_ package to be
installed.
"""

import os
import typing
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence

import numpy as np
import xarray as xr

if typing.TYPE_CHECKING:
    import h5py
    from nexusformat import nexus

else:
    import lazy_loader as _lazy

    import erlab

    h5py = _lazy.load("h5py")

    nexus = erlab.utils.misc.LazyImport(
        "nexusformat.nexus",
        err_msg="The `nexusformat` package is required to read NeXus files",
    )


def _parse_value(value):
    """Convert numpy scalars or bytes to native Python types and regular strings."""
    if isinstance(value, bytes | bytearray):
        return value.decode("utf-8")
    if isinstance(value, nexus.NXattr):
        return _parse_value(value.nxdata)
    if isinstance(value, np.ndarray) and value.size == 1:
        return _parse_value(np.atleast_1d(value)[0])
    if isinstance(value, np.generic):
        # Convert to native Python type
        return value.item()
    return value


def _parse_field(field):
    """Convert a NeXus field to a native Python type."""
    if not isinstance(field, nexus.NXfield):
        return field

    if field.size == 1:
        return _parse_value(field.nxdata)

    return field.nxdata


def _remove_axis_attrs(attrs: Mapping[str, typing.Any]) -> dict[str, typing.Any]:
    out: dict[str, typing.Any] = {}
    for k, v in attrs.items():
        if k in ("axis", "primary", "target"):
            continue
        out[k] = _parse_value(v)

    return out


def _parse_h5py(obj: "h5py.Group | h5py.Dataset", out: dict) -> None:
    if isinstance(obj, h5py.Group):
        for v in obj.values():
            _parse_h5py(v, out)
    elif isinstance(obj, h5py.Dataset):
        out[obj.name] = _parse_value(obj[()])


def _parse_group(
    group: "nexus.NXgroup",
    out: dict[str, typing.Any],
    exclude: Sequence[str],
    parse: bool,
) -> None:
    """Recursively parse a NeXus group and its items into a nested dictionary.

    The keys are the absolute nxpath of the items, and the values are the corresponding
    NXfield objects. The function is called recursively for each sub-group. NXlink
    objects are skipped.

    Parameters
    ----------
    group
        The NeXus group to be parsed.
    out
        The dictionary to store the parsed items.
    exclude
        List of full paths to exclude from the output.
    parse
        Whether to parse the values of NXfields to native Python types.

    Note
    ----
    Sometimes the items in the group may be invalid NXobjects. In that case, the
    function will attempt to get the values from h5py as a fallback.

    """
    for item in group.values():
        if item.nxpath in exclude:
            continue

        if isinstance(item, nexus.NXgroup):
            _parse_group(item, out, exclude, parse)

        elif isinstance(item, nexus.NXlink):
            # Skip links
            continue

        elif isinstance(item, nexus.NXfield):
            if parse:
                out[item.nxpath] = _parse_field(item)
            else:
                out[item.nxpath] = item

        else:
            # Sometimes item may be an invalid NXobject
            # In that case, get the values from h5py as a fallback
            _parse_h5py(item.nxfile.get(item.nxpath), out)


def _get_primary_coords(group: "nexus.NXgroup", out: list["nexus.NXfield"]):
    for item in group.values():
        if isinstance(item, nexus.NXgroup):
            _get_primary_coords(item, out)
        elif isinstance(item, nexus.NXlink):
            # Skip links
            continue
        elif (
            isinstance(item, nexus.NXfield)
            and "primary" in item.attrs
            and int(item.primary) == 1
        ):
            out.append(item)


def _get_non_primary_coords(group: "nexus.NXgroup", out: list["nexus.NXfield"]):
    for item in group.values():
        if isinstance(item, nexus.NXgroup):
            _get_non_primary_coords(item, out)
        elif isinstance(item, nexus.NXlink):
            # Skip links
            continue
        elif (
            isinstance(item, nexus.NXfield)
            and "axis" in item.attrs
            and ("primary" not in item.attrs or int(item.primary) != 1)
        ):
            out.append(item)


def get_primary_coords(group: "nexus.NXgroup") -> list["nexus.NXfield"]:
    """Get all primary coordinates in a group.

    Retrieves all fields with the attribute `primary=1` in the group and its subgroups
    recursively. The output list is sorted by the `axis` attribute of the fields.

    Parameters
    ----------
    group
        The group to search for the coordinates.

    Returns
    -------
    fields_primary : list of NXfield

    """
    fields_primary: list[nexus.NXfield] = []
    _get_primary_coords(group, fields_primary)
    return sorted(fields_primary, key=lambda field: int(field.axis))


def get_non_primary_coords(group: "nexus.NXgroup") -> list["nexus.NXfield"]:
    """Get all non-primary coordinates in a group.

    Retrieves all fields with the attribute `axis` in the group and its subgroups
    recursively. The output list is sorted by the order of traversal.

    Parameters
    ----------
    group
        The group to search for the coordinates.

    Returns
    -------
    fields_non_primary : list of NXfield

    """
    fields_non_primary: list[nexus.NXfield] = []
    _get_non_primary_coords(group, fields_non_primary)
    return fields_non_primary


def get_primary_coord_dict(
    fields: list["nexus.NXfield"],
) -> tuple[tuple[str, ...], dict[str, xr.DataArray | tuple]]:
    """Generate a dictionary of primary coordinates from a list of NXfields.

    The output dictionary contains the absolute NXpath of the field as the key and the
    value is either a `xarray.DataArray` or a tuple of the associated dimensions, data,
    and attributes.

    The output arguments can be directly used as arguments for the `xarray.DataArray`
    constructor.

    If there are multiple fields with the same axis, one of the fields will depend on
    the other field.

    Parameters
    ----------
    fields : list of NXfield
        The list of primary coordinates to be converted. The list can be obtained from a
        NXgroup using :func:`get_primary_coords`.

    Returns
    -------
    dims : tuple of str
        The dimension names of the primary coordinates.
    coords : dict of str to DataArray or tuple
        The dictionary of primary coordinates in the group.
    """
    coords: dict[str, xr.DataArray | tuple] = {}

    # Dict to store processed axes
    # Ensure that if there are multiple fields with the same axis, only one is used
    processed_axes: dict[int, nexus.NXfield] = {}

    for field in fields:
        if field.ndim == 1:
            if int(field.axis) in processed_axes:
                # Axis already added, make this coordinate depend on that dimension
                coords[field.nxpath] = (
                    processed_axes[int(field.axis)].nxpath,
                    field.nxdata,
                    _remove_axis_attrs(field.attrs),
                )
                continue

            coords[field.nxpath] = nxfield_to_xarray(field)

        else:
            # Multidimensional coordinate (like kinetic energy in hv-dependent scans)
            assoc_dims: list[str] = []
            for num in field.shape:
                for field in fields:
                    if field.ndim == 1 and field.shape == (num,):
                        assoc_dims.append(field.nxpath)
                        break
                else:
                    # Shape not in list, must be new dimension
                    assoc_dims.append(field.nxpath)

            coords[field.nxpath] = (
                tuple(assoc_dims),
                field.nxdata,
                _remove_axis_attrs(field.attrs),
            )

        processed_axes[int(field.axis)] = field

    dims = tuple(f.nxpath for f in dict(sorted(processed_axes.items())).values())

    return dims, coords


def get_coord_dict(
    group: "nexus.NXgroup",
) -> tuple[tuple[str, ...], dict[str, xr.DataArray | tuple]]:
    """Generate a dictionary of coordinates from a NeXus group.

    Parameters
    ----------
    group : NXgroup
        The NeXus group from which to extract coordinates.

    Returns
    -------
    dims : tuple of str
        The dimension names of the primary coordinates.
    coords : dict of str to DataArray or tuple
        The dictionary of all coordinates in the group.
    """
    fields_primary: list[nexus.NXfield] = get_primary_coords(group)
    fields_non_primary: list[nexus.NXfield] = get_non_primary_coords(group)

    dims, coords = get_primary_coord_dict(fields_primary)

    for field in fields_non_primary:
        if field.size == 1:
            # Scalar coordinate
            coords[field.nxpath] = nxfield_to_xarray(field, no_dims=True)

        else:
            # Coord depends on some other primary coordinate
            associated_primary: nexus.NXfield = group.nxroot[dims[int(field.axis) - 1]]
            coords[field.nxpath] = (
                associated_primary.nxpath,
                field.nxdata,
                _remove_axis_attrs(field.attrs),
            )

    return dims, coords


def nexus_group_to_dict(
    group: "nexus.NXgroup",
    exclude: Sequence[str] | None,
    relative: bool = True,
    replace_slash: bool = True,
    parse: bool = False,
) -> dict[str, typing.Any]:
    """Convert a NeXus group to a dictionary.

    This function takes a NeXus group and converts it into a dictionary where the keys
    are the paths of the group's items relative to the group's root path, and the values
    are the corresponding items.

    Parameters
    ----------
    group
        The NeXus group to be converted.
    exclude
        List of paths to exclude from the output.
    relative
        Whether to use the relative or absolute paths of the items. If `True`, the keys
        are the paths of the items relative to the path of the group. If `False`, the
        keys are the absolute paths of the items relative to the root of the NeXus file.
    replace_slash
        Whether to replace the slashes in the paths with dots.
    parse
        Whether to coerce the values of NXfields to native Python types.

    """
    if exclude is None:
        exclude = []

    prefix: str = group.nxpath
    out: dict[str, typing.Any] = {}
    _parse_group(group, out, exclude, parse)

    if relative:
        out = {k.removeprefix(prefix).lstrip("/"): v for k, v in out.items()}

    if replace_slash:
        out = {k.replace("/", "."): v for k, v in out.items()}

    return out


def nxfield_to_xarray(field: "nexus.NXfield", no_dims: bool = False) -> xr.DataArray:
    """Convert a coord-like 1D NeXus field to a single `xarray.DataArray`.

    Parameters
    ----------
    field
        The NeXus field to be converted.

    Returns
    -------
    DataArray

    """
    attrs = _remove_axis_attrs(field.attrs)

    if no_dims:
        if not field.size == 1:
            raise ValueError(
                "The field must have a single value "
                "to be converted to a non-dimensional array"
            )
        return xr.DataArray(field.nxdata.item(), attrs=attrs)

    if field.ndim > 1:
        raise ValueError("The field must have a single dimension to be converted")

    return xr.DataArray(field, dims=(field.nxpath,), attrs=attrs)


def nxgroup_to_xarray(
    group: "nexus.NXgroup",
    data: str | Callable[["nexus.NXgroup"], "nexus.NXfield"],
    without_values: bool = False,
) -> xr.DataArray:
    """Convert a NeXus group to an xarray DataArray.

    Parameters
    ----------
    group : NXgroup
        The NeXus group to be converted.
    data : str or callable
        The location of the data values which can be specified in two ways:

        - If a string, it must be the relative path from ``group`` to the `NXfield
          <nexusformat.nexus.tree.NXfield>` containing the data values.

        - If a callable, it must be a function that takes ``group`` as an argument and
          returns the `NXfield <nexusformat.nexus.tree.NXfield>` containing the data
          values.
    without_values
        If `True`, the returned DataArray values will be filled with zeros. Use this to
        check the coords or attrs quickly without loading in the full data.

    Returns
    -------
    DataArray
        The DataArray containing the data. Dimension and coordinate names are the
        relative paths of the corresponding NXfields, with the slashes replaced by dots.

    """
    if callable(data):
        values: nexus.NXfield = data(group)
    else:
        values = group[data]

    if isinstance(values, nexus.NXlink):
        values = values.nxlink

    dims, coords = get_coord_dict(group)
    excluded_paths = [*list(coords.keys()), values.nxpath]

    attrs = nexus_group_to_dict(group, exclude=excluded_paths, parse=True)

    # Strip path prefix from dimensions and coordinates
    prefix: str = group.nxpath

    def _make_relative(s: Hashable) -> str:
        return str(s).removeprefix(prefix).lstrip("/").replace("/", ".")

    def _make_coord_relative(t: xr.DataArray | tuple) -> xr.DataArray | tuple:
        if isinstance(t, xr.DataArray):
            if len(t.dims) == 1:
                return t.rename({t.dims[0]: _make_relative(t.dims[0])})

            return t

        _tmp_list = list(t)
        if isinstance(t[0], str):
            _tmp_list[0] = _make_relative(t[0])
        elif isinstance(t[0], Iterable):
            _tmp_list[0] = tuple(_make_relative(s) for s in t[0])
        return tuple(_tmp_list)

    dims = tuple(_make_relative(d) for d in dims)
    coords = {_make_relative(k): _make_coord_relative(v) for k, v in coords.items()}

    if without_values:
        values = np.zeros(values.shape, values.dtype)
    return xr.DataArray(values, dims=dims, coords=coords, attrs=attrs)


def get_entry(filename: str | os.PathLike, entry: str | None = None) -> "nexus.NXentry":
    """Get an NXentry object from a NeXus file.

    Parameters
    ----------
    filename
        The path to the NeXus file.
    entry
        The path of the entry to get. If `None`, the first entry in the file is
        returned.

    Returns
    -------
    entry : NXentry
        The NXentry object obtained from the file.

    """
    root = nexus.nxload(filename)
    if entry is None:
        return next(iter(root.entries.values()))

    return root[entry]
