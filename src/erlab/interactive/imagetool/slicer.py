"""Helper functions for fast slicing :class:`xarray.DataArray` objects."""

from __future__ import annotations

import copy
import functools
import typing
import warnings

import numpy as np
import numpy.typing as npt
from qtpy import QtCore, QtWidgets

import erlab

if typing.TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence

    import dask.array
    import xarray as xr


class ArraySlicerState(typing.TypedDict):
    """A dictionary containing the state of cursors in an :class:`ArraySlicer`."""

    dims: tuple[Hashable, ...]
    bins: list[list[int]]
    indices: list[list[int]]
    values: list[list[float]]
    snap_to_data: bool
    twin_coord_names: typing.NotRequired[tuple[Hashable, ...]]
    cursor_color_params: typing.NotRequired[
        tuple[tuple[Hashable, ...], Hashable, str, bool, float, float] | None
    ]


def check_cursors_compatible(old: xr.DataArray, new: xr.DataArray) -> bool:
    """Check if the cursor positions of the old array can be applied to the new array.

    The two arrays must have the same dimensions, and the coordinate values for each
    dimension of the old array must be included in the coordinate values of the same
    dimension in the new array.

    Parameters
    ----------
    old
        The original DataArray.
    new
        The new DataArray.
    """
    old, new = _cursor_compatibility_pair(old, new)
    if set(old.dims) != set(new.dims):
        return False
    for d in old.dims:
        old_coord = old.coords.get(d)
        new_coord = new.coords.get(d)
        if old_coord is None or new_coord is None:
            if old.sizes[d] != new.sizes[d]:
                return False
            continue
        if not np.isin(old_coord.values, new_coord.values, assume_unique=True).all():
            return False
    return True


def _drop_unmatched_stack_dim(data: xr.DataArray, other: xr.DataArray) -> xr.DataArray:
    if (
        "stack_dim" in data.dims
        and "stack_dim" not in other.dims
        and data.sizes["stack_dim"] == 1
    ):
        return data.squeeze("stack_dim", drop=True)
    return data


def _cursor_compatibility_pair(
    old: xr.DataArray, new: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
    old_view = erlab.utils.array._restore_nonuniform_dims(old.copy(deep=False))
    new_view = erlab.utils.array._restore_nonuniform_dims(new.copy(deep=False))
    old_view = _drop_unmatched_stack_dim(old_view, new_view)
    new_view = _drop_unmatched_stack_dim(new_view, old_view)
    return old_view, new_view


def _get_inc(coord):
    try:
        return coord[1] - coord[0]
    except IndexError:
        # Coord size is 1, assume increment of 1
        return 1


def _display_value_abs_limit() -> float:
    return float(
        erlab.interactive.options.qsettings.value("colors/max_rendered_abs_value", 1e30)
    )


def _limits_require_display_mask(mn: float, mx: float, limit: float) -> bool:
    return np.isinf(mn) or np.isinf(mx) or mn < -limit or mx > limit


def _display_safe_values(values, limit: float | None = None):
    """Return values safe for Qt rendering without modifying the source array."""
    arr = np.asarray(values)
    if (
        arr.size == 0
        or not np.issubdtype(arr.dtype, np.number)
        or np.issubdtype(arr.dtype, np.complexfloating)
    ):
        return values

    if limit is None:
        limit = _display_value_abs_limit()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
        mn, mx = float(np.nanmin(arr)), float(np.nanmax(arr))
    if not _limits_require_display_mask(mn, mx, limit):
        return values

    with np.errstate(all="ignore"):
        return np.where(np.isfinite(arr) & (np.abs(arr) <= limit), arr, np.nan)


def _display_safe_float(value, limit: float | None = None) -> float:
    """Return a scalar display value after applying ImageTool display masking."""
    arr = np.asarray(value)
    if arr.size == 0:
        return np.nan
    if limit is None:
        limit = _display_value_abs_limit()
    if arr.size == 1:
        out = float(arr.reshape(()))
        if np.isfinite(out) and abs(out) <= limit:
            return out
        return np.nan

    arr = np.asarray(_display_safe_values(arr, limit))
    if np.isnan(arr).all():
        return np.nan
    return float(np.nanmean(arr))


def _center_index_for_size(size: int) -> int:
    return size // 2 - (1 if size % 2 == 0 else 0)


def _center_indices_for_shape(shape: Sequence[int]) -> list[int]:
    return [_center_index_for_size(int(size)) for size in shape]


def _normalized_axis_index(index: int, size: int) -> int:
    if 0 <= index < size:
        return index
    return _center_index_for_size(size)


def _bin_slice_for_axis(center: int, window: int, size: int) -> slice:
    start = center - window // 2
    stop = center + (window - 1) // 2 + 1
    start = max(0, start)
    stop = min(size, stop)
    selected_size = stop - start
    if selected_size < window:
        missing = window - selected_size
        if start == 0:
            stop = min(size, stop + missing)
        elif stop == size:
            start = max(0, start - missing)
    return slice(start, stop)


def _hidden_axes_for_display(ndim: int, display_axes: Sequence[int]) -> tuple[int, ...]:
    key_set = set(display_axes)
    return tuple(ax for ax in range(ndim) if ax not in key_set)


def _reduced_axes_selection(
    shape: Sequence[int],
    reduced_axes: Sequence[int],
    indices: Sequence[int],
    bins: Sequence[int],
    binned: Sequence[bool],
) -> tuple[tuple[slice | int, ...], tuple[int, ...], bool, bool]:
    ndim = len(shape)
    selection: list[slice | int] = [slice(None)] * ndim
    any_binned = False
    all_binned = True
    dropped = 0
    selected_axes: list[int] = []
    for ax in reduced_axes:
        if binned[ax]:
            any_binned = True
            selection[ax] = _bin_slice_for_axis(indices[ax], bins[ax], int(shape[ax]))
            selected_axes.append(ax - dropped)
        else:
            all_binned = False
            selection[ax] = indices[ax]
            dropped += 1
    return tuple(selection), tuple(selected_axes), any_binned, all_binned


def _display_safe_minmax(
    data: xr.DataArray,
    raw_limits: tuple[float, float] | None = None,
    limit: float | None = None,
) -> tuple[float, float]:
    """Return display-safe finite limits for an ImageTool DataArray."""
    if limit is None:
        limit = _display_value_abs_limit()
    mn, mx = raw_limits if raw_limits is not None else _minmax_darr_quiet(data)
    if _limits_require_display_mask(mn, mx, limit):
        if data.chunks is None:
            return _display_safe_minmax_eager(data.values, limit)
        with np.errstate(all="ignore"):
            mn, mx = _minmax_darr_quiet(
                data.where(np.isfinite(data) & (np.abs(data) <= limit))
            )
    if np.isfinite(mn) and np.isfinite(mx):
        return mn, mx
    return 0.0, 1.0


def _display_safe_minmax_eager(values, limit: float) -> tuple[float, float]:
    mn, mx = np.inf, -np.inf
    chunks = (values,) if values.ndim <= 2 else values
    for chunk in chunks:
        chunk = _display_safe_values(chunk, limit)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
            chunk_mn = float(np.nanmin(chunk))
            chunk_mx = float(np.nanmax(chunk))
        if np.isfinite(chunk_mn):
            mn = min(mn, chunk_mn)
        if np.isfinite(chunk_mx):
            mx = max(mx, chunk_mx)
    if np.isfinite(mn) and np.isfinite(mx):
        return mn, mx
    return 0.0, 1.0


def _minmax_darr_quiet(data: xr.DataArray) -> tuple[float, float]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
        return erlab.utils.array.minmax_darr(data)


def qsel_args_from_indexers(
    data: xr.DataArray,
    indexers: Mapping[Hashable, slice | int],
    binned_dims: Sequence[Hashable],
) -> dict[Hashable, float]:
    """Build ``qsel`` keyword arguments from index selections.

    Parameters
    ----------
    data
        Data whose current coordinates should be used for the returned ``qsel`` values.
    indexers
        Dimension names mapped to integer indices or index slices.
    binned_dims
        Dimensions whose index slices should become ``qsel`` center and width
        arguments.

    Returns
    -------
    dict
        Keyword arguments suitable for :meth:`xarray.DataArray.qsel`.

    Raises
    ------
    ValueError
        If a requested dimension is missing or a binned selection cannot be expressed
        by the current coordinate values.
    """
    out: dict[Hashable, float] = {}
    binned_dim_set = set(binned_dims)

    for dim, selector in indexers.items():
        if dim not in data.dims:
            raise ValueError(f"Dimension `{dim}` not found in data")
        coord_values = data[dim].values
        inc = _get_inc(coord_values)
        order = 3 if inc == 0 else erlab.utils.array.effective_decimals(inc)

        if dim in binned_dim_set:
            coord = data[dim][selector].values
            center = float(np.round(coord.mean(), order))
            width = float(np.abs(np.round(coord[-1] - coord[0] + inc, order)))

            out[dim] = center
            slice_obj = slice(center - width / 2, center + width / 2)
            if coord[0] > coord[-1]:
                slice_obj = slice(
                    slice_obj.stop,
                    slice_obj.start,
                    -slice_obj.step if slice_obj.step is not None else None,
                )

            if not np.allclose(data[dim].sel({dim: slice_obj}).values, coord):
                raise ValueError(
                    "Bin does not contain the same values as the original data."
                )

            out[str(dim) + "_width"] = width
        else:
            out[dim] = float(np.round(coord_values[selector], order))

    return out


class ArraySlicer(QtCore.QObject):
    """Internal class used to slice a :class:`xarray.DataArray` rapidly.

    Computes binned line and image profiles from multiple cursors. This class also
    stores the data indices and the number of bins for each cursor. Automatic conversion
    of non-uniform dimensions are also handled here.

    Parameters
    ----------
    xarray_obj
        A :class:`xarray.DataArray` with up to 4 dimensions.

    Signals
    -------
    sigIndexChanged(int or tuple of int, tuple or None)
        Emitted when the cursor index is changed. The first argument is the cursor
        index, and the second is a tuple containing the changed axes.
    sigBinChanged(int, tuple)
        Emitted when the bin size is changed. The first argument is the cursor index,
        and the second is a tuple containing the changed axes.
    sigCursorCountChanged(int)
        Emitted when the number of cursors is changed. Emits the new number of cursors.
    sigShapeChanged()
        Emitted when the underlying `xarray.DataArray` is transposed.
    sigTwinChanged()
        Emitted when the coordinates to be displayed in the twin axes are changed.

    Note
    ----
    The original intent of this class was a xarray accessor. This is why `ArraySlicer`
    does not depend on a :class:`ImageSlicerArea
    <erlab.interactive.imagetool.viewer.ImageSlicerArea>` but rather on the underlying
    `xarray.DataArray`. Originally, when loading a different array, a different instance
    of `ArraySlicer` had to be created. This was a terrible design choice since it
    messed up signals every time the instance was replaced. Hence, the behaviour was
    modified (23/06/19) so that the underlying `xarray.DataArray` of `ArraySlicer` could
    be swapped. As a consequence, each instance of :class:`ImageSlicerArea
    <erlab.interactive.imagetool.viewer.ImageSlicerArea>` now corresponds to exactly one
    instance of `ArraySlicer`, regardless of the data. In the future, `ArraySlicer`
    might be changed so that it relies on its one-to-one correspondence with
    :class:`ImageSlicerArea <erlab.interactive.imagetool.viewer.ImageSlicerArea>`
    for the signals.

    """

    sigIndexChanged = QtCore.Signal(object, object)  #: :meta private:
    sigBinChanged = QtCore.Signal(int, tuple)  #: :meta private:
    sigCursorCountChanged = QtCore.Signal(int)  #: :meta private:
    sigShapeChanged = QtCore.Signal()  #: :meta private:
    sigTwinChanged = QtCore.Signal()  #: :meta private:

    def __init__(
        self,
        xarray_obj: xr.DataArray,
        parent: QtCore.QObject,
        *,
        display_value_abs_limit: float | None = None,
    ) -> None:
        super().__init__(parent)
        self.display_value_abs_limit = float(
            display_value_abs_limit
            if display_value_abs_limit is not None
            else _display_value_abs_limit()
        )
        self.snap_act = QtWidgets.QAction("&Snap to Pixels", self)
        self.snap_act.setShortcut("S")
        self.snap_act.setCheckable(True)
        self.snap_act.setChecked(False)
        self.snap_act.setToolTip("Snap cursors to data points")

        self.set_array(xarray_obj, validate=True, reset=True, copy_values=False)

    @property
    def snap_to_data(self) -> bool:
        return self.snap_act.isChecked()

    @snap_to_data.setter
    def snap_to_data(self, value: bool) -> None:
        self.snap_act.setChecked(value)

    @property
    def twin_coord_names(self) -> set[Hashable]:
        return self._twin_coord_names

    @twin_coord_names.setter
    def twin_coord_names(self, coord_names: set[Hashable]) -> None:
        self._twin_coord_names: set[Hashable] = set(coord_names)
        self.sigTwinChanged.emit()

    def set_array(
        self,
        xarray_obj: xr.DataArray,
        validate: bool = True,
        reset: bool = False,
        *,
        copy_values: bool = True,
        preserve_dims: Sequence[Hashable] | None = None,
    ) -> bool:
        """Set the DataArray object to be sliced.

        Parameters
        ----------
        xarray_obj : DataArray
            The array to slice.
        validate
            If True, validate the array before setting it. This flag is intended for
            internal use where the new array is guaranteed to be valid, such as when
            transposing an already valid array.
        reset
            If True, reset cursors, bins, indices, and values.
        copy_values
            If `True`, copy the underlying array values while validating. Set to
            `False` to reuse the current values buffer when the caller already manages
            ownership.
        preserve_dims
            Dimension order to preserve after validation. This is used when rebuilding
            the slicer from the public source array without disturbing the current view
            layout.

        Returns
        -------
        bool
            True if the cursors were reset, False if only the data was updated.
        """
        obj_original = getattr(self, "_obj", None)
        if obj_original is not None:
            # Shallow copy is enough: we only compare dims/coords for cursor
            # compatibility.
            obj_original = obj_original.copy(deep=False)

        if validate:
            self._obj = self.validate_array(xarray_obj, copy_values=copy_values)
        else:
            self._obj = xarray_obj

        if (
            preserve_dims is not None
            and tuple(self._obj.dims) != tuple(preserve_dims)
            and set(self._obj.dims) == set(preserve_dims)
        ):
            self._obj = self._obj.transpose(*preserve_dims)

        if (obj_original is not None) and reset:
            # If same coords, keep cursors
            if check_cursors_compatible(obj_original, self._obj):
                if set(self._obj.dims) == set(obj_original.dims):
                    self._obj = self._obj.transpose(*obj_original.dims)
                reset = False
            del obj_original

        self.clear_dim_cache()
        self._refresh_array_layout_cache()
        if validate:
            self.clear_val_cache()

        if reset:
            self._bins: list[list[int]] = [[1] * self._obj.ndim]
            # Keep an explicit boolean cache alongside the integer bin widths so the
            # hot slicing paths do not rebuild `b != 1` tuples on every query.
            self._binned: list[tuple[bool, ...]] = [
                tuple(False for _ in self._all_axes)
            ]
            self._indices: list[list[int]] = [
                [s // 2 - (1 if s % 2 == 0 else 0) for s in self._obj.shape]
            ]
            self._values: list[list[np.floating]] = [
                [c[i] for c, i in zip(self.coords, self._indices[0], strict=True)]
            ]
            self._twin_coord_names = set()
            self._cursor_color_params: (
                tuple[tuple[Hashable, ...], Hashable, str, bool, float, float] | None
            ) = None
            self.snap_to_data = False
        else:
            # Preserve cursor bin widths when the array is replaced, but rebuild the
            # derived boolean cache against the current axis order.
            self._normalize_cursor_axis_state()
        return reset

    @functools.cached_property
    def associated_coord_dims(self) -> dict[Hashable, tuple[Hashable, ...]]:
        """Numeric non-dimension coordinates that can be plotted with data profiles."""
        dims = set(self._obj.dims)
        out: dict[Hashable, tuple[Hashable, ...]] = {}
        for name, coord in self._obj.coords.items():
            coord_dims = tuple(coord.dims)
            if (
                name not in dims
                and coord_dims
                and set(coord_dims).issubset(dims)
                and np.issubdtype(coord.dtype, np.number)
                and not np.issubdtype(coord.dtype, np.complexfloating)
            ):
                out[name] = coord_dims
        return out

    def associated_coord_profile(
        self, coord_name: Hashable, cursor: int, display_axis: Sequence[int]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None:
        """Return a 1D associated-coordinate profile for a profile plot."""
        if len(display_axis) != 1:
            return None

        dims = self.associated_coord_dims.get(coord_name)
        if dims is None:
            return None

        axis = display_axis[0]
        display_dim = self._obj.dims[axis]
        if display_dim not in dims:
            return None

        coord = self._obj.coords[coord_name]
        isel_kw: dict[Hashable, slice | int] = {}
        mean_dims: list[Hashable] = []
        for dim in dims:
            if dim == display_dim:
                continue
            axis_idx = self._dim_indices.get(dim)
            if axis_idx is None:  # pragma: no cover
                return None
            if self._binned[cursor][axis_idx]:
                isel_kw[dim] = self._bin_slice(cursor, axis_idx)
                mean_dims.append(dim)
            else:
                isel_kw[dim] = self._indices[cursor][axis_idx]

        selected = coord.isel(isel_kw)
        if mean_dims:
            selected = selected.mean(dim=mean_dims, skipna=True)

        return (
            self.coords_uniform[axis].astype(np.float64),
            selected.transpose(display_dim).values.astype(np.float64),
        )

    def associated_coord_point_value(
        self, coord_name: Hashable, cursor: int, binned: bool = True
    ) -> np.floating | dask.array.Array | None:
        """Return an associated-coordinate value at the cursor position."""
        dims = self.associated_coord_dims.get(coord_name)
        if dims is None:
            return None

        coord = self._obj.coords[coord_name]
        isel_kw: dict[Hashable, slice | int] = {}
        mean_dims: list[Hashable] = []
        for dim in dims:
            axis_idx = self._dim_indices.get(dim)
            if axis_idx is None:  # pragma: no cover
                return None
            if binned and self._binned[cursor][axis_idx]:
                isel_kw[dim] = self._bin_slice(cursor, axis_idx)
                mean_dims.append(dim)
            else:
                isel_kw[dim] = self._indices[cursor][axis_idx]

        selected = coord.isel(isel_kw)
        if mean_dims:
            selected = selected.mean(dim=mean_dims, skipna=True)
        return selected.data

    def cursor_color_coord(
        self, cursor: int, coord_dims: tuple[Hashable, ...], coord_name: Hashable
    ) -> tuple[tuple[Hashable, ...], npt.NDArray[np.float64], float] | None:
        """Return dims, all values, and cursor value for coordinate-based coloring."""
        if coord_name in self._dim_indices and coord_dims == (coord_name,):
            axis_idx = self._dim_indices[coord_name]
            values = self.coords[axis_idx].astype(np.float64)
        else:
            if self.associated_coord_dims.get(coord_name) != coord_dims:
                return None
            values = self._obj.coords[coord_name].values.astype(np.float64)

        cursor_index: list[int] = []
        for dim in coord_dims:
            coord_axis_idx = self._dim_indices.get(dim)
            if coord_axis_idx is None:  # pragma: no cover
                return None
            cursor_index.append(self._indices[cursor][coord_axis_idx])
        return coord_dims, values, float(values[tuple(cursor_index)])

    @functools.cached_property
    def coords(self) -> tuple[npt.NDArray[np.floating], ...]:
        """Coordinate values of each dimension in the array."""
        if self._nonuniform_axes:
            return tuple(
                (
                    self.values_of_dim(str(dim).removesuffix("_idx"))
                    if i in self._nonuniform_axes
                    else self.values_of_dim(dim)
                )
                for i, dim in enumerate(self._obj.dims)
            )
        return self.coords_uniform

    @functools.cached_property
    def coords_uniform(self) -> tuple[npt.NDArray[np.floating], ...]:
        """Coordinate values of each dimension in the array.

        Non-uniform coordinates are converted to uniform indices.
        """
        return tuple(self.values_of_dim(dim) for dim in self._obj.dims)

    @functools.cached_property
    def incs(self) -> tuple[np.floating, ...]:
        """Increment of each dimension in the array.

        Returns the step for each dimension coordinate in the array. Non-uniform
        dimensions will return the absolute average of all non-zero step sizes.
        """
        if self._nonuniform_axes:
            return tuple(
                (
                    erlab.interactive.imagetool.fastslicing._avg_nonzero_abs_diff(coord)
                    if i in self._nonuniform_axes
                    else _get_inc(coord)
                )
                for i, coord in enumerate(self.coords)
            )
        return tuple(_get_inc(coord) for coord in self.coords)

    @functools.cached_property
    def incs_uniform(self) -> tuple[np.floating, ...]:
        """Increment size of each dimension in the array.

        Non-uniform dimensions will increment by 1.
        """
        return tuple(_get_inc(coord) for coord in self.coords_uniform)

    @functools.cached_property
    def lims(self) -> tuple[tuple[np.floating, np.floating], ...]:
        """Coordinate bounds of each dimension in the array."""
        if self._nonuniform_axes:
            return tuple(
                (
                    (np.amin(coord), np.amax(coord))
                    if i in self._nonuniform_axes
                    else (coord[0], coord[-1])
                )
                for i, coord in enumerate(self.coords)
            )
        return tuple((coord[0], coord[-1]) for coord in self.coords)

    @functools.cached_property
    def lims_uniform(self) -> tuple[tuple[np.floating, np.floating], ...]:
        """Coordinate bounds of each dimension in the array.

        Non-uniform dimensions are converted to uniform indices.
        """
        return tuple((coord[0], coord[-1]) for coord in self.coords_uniform)

    @functools.cached_property
    def uniform_index_params(
        self,
    ) -> tuple[tuple[np.floating, np.floating, int], ...]:
        """Cached uniform-axis indexing parameters for fast value lookups."""
        return tuple(
            (lims[0], inc, size - 1)
            for lims, inc, size in zip(
                self.lims_uniform, self.incs_uniform, self._obj.shape, strict=True
            )
        )

    @property
    def nanmax(self) -> float:
        return self.limits[1]

    @property
    def nanmin(self) -> float:
        return self.limits[0]

    @functools.cached_property
    def _raw_limits(self) -> tuple[float, float]:
        return _minmax_darr_quiet(self._obj)

    def display_safe_values(self, values):
        """Return values safe for Qt rendering for this slicer instance."""
        return _display_safe_values(values, self.display_value_abs_limit)

    def display_safe_float(self, value) -> float:
        """Return a scalar display value for this slicer instance."""
        return _display_safe_float(value, self.display_value_abs_limit)

    def display_safe_minmax(
        self, data: xr.DataArray, raw_limits: tuple[float, float] | None = None
    ) -> tuple[float, float]:
        """Return display-safe finite limits for this slicer instance."""
        return _display_safe_minmax(data, raw_limits, self.display_value_abs_limit)

    @property
    def display_values_known_safe(self) -> bool:
        if self._obj.chunks is not None:
            return False
        raw_limits = self.__dict__.get("_raw_limits")
        return raw_limits is not None and not _limits_require_display_mask(
            raw_limits[0], raw_limits[1], self.display_value_abs_limit
        )

    @functools.cached_property
    def limits(self) -> tuple[float, float]:
        """Return the display-safe global minimum and maximum of the data."""
        return self.display_safe_minmax(self._obj, self._raw_limits)

    @property
    def n_cursors(self) -> int:
        """The number of cursors."""
        return len(self._bins)

    @property
    def state(self) -> ArraySlicerState:
        return {
            "dims": copy.deepcopy(self._obj.dims),
            "bins": copy.deepcopy(self._bins),
            "indices": copy.deepcopy(self._indices),
            "values": erlab.utils.misc._convert_to_native(self._values),
            "snap_to_data": bool(self.snap_to_data),
            "twin_coord_names": tuple(self.twin_coord_names),
            "cursor_color_params": copy.deepcopy(self._cursor_color_params),
        }

    @state.setter
    def state(self, state: ArraySlicerState) -> None:
        if self._obj.dims != state["dims"]:
            self._obj = self._obj.transpose(*state["dims"])

        self.snap_to_data = state["snap_to_data"]
        self.clear_cache()
        # The setters below depend on eager layout lookups such as
        # `_nonuniform_axes_set` and `_dim_indices`, so rebuild them immediately after
        # any transpose and before restoring cursor state.
        self._refresh_array_layout_cache()

        for i, (bins, indices, values) in enumerate(
            zip(state["bins"], state["indices"], state["values"], strict=True)
        ):
            self.center_cursor(i, update=False)
            self.set_indices(i, indices, update=False)
            self.set_values(i, values, update=False)
            self.set_bins(i, bins, update=False)

            # We call set_array below so use update=False

        self.twin_coord_names = set(state.get("twin_coord_names", set()))
        cursor_color_params = state.get("cursor_color_params", None)
        if cursor_color_params is not None:
            coord_dims, coord_name, cmap, reverse, vmin, vmax = cursor_color_params
            if isinstance(coord_dims, list):
                coord_dims = tuple(coord_dims)
            elif not isinstance(coord_dims, tuple):
                coord_dims = (coord_dims,)
            self._cursor_color_params = (
                coord_dims,
                coord_name,
                cmap,
                reverse,
                vmin,
                vmax,
            )
        else:
            self._cursor_color_params = None

        self.set_array(self._obj, validate=False)
        # Not sure why the last line is needed but the cursor is not fully restored
        # without it, and requires a transpose before the cursor position and the
        # dimension label is correct. Must be a bug in the order of operations or
        # something... reproduce by archiving a 3D array with one non-uniform axis.

    @staticmethod
    def validate_array(data: xr.DataArray, copy_values: bool = True) -> xr.DataArray:
        """Validate a given :class:`xarray.DataArray`.

        If data has two momentum axes (``kx`` and ``ky``), set them (and ``eV`` if
        exists) as the first two (or three) dimensions. Then, checks the data for
        non-uniform coordinates, which are converted to indices. Finally, converts the
        coordinates to C-contiguous arrays.

        If input data values are neither float32 nor float64, a conversion to float64 is
        attempted.

        Parameters
        ----------
        data
            Input array with at least two dimensions.
        copy_values
            If `True`, copy the underlying values while validating. Set to `False` when
            the caller intentionally wants the validated array to share the same values
            buffer as the source array.

        Returns
        -------
        xarray.DataArray
            The converted data. Non-uniform dimensions are converted to be dependent on
            uniform index dimensions, suffixed with ``'_idx'``.

        """
        # Keep metadata copying shallow to avoid deepcopy/GC issues.
        data = data.copy(deep=False)
        if copy_values:
            # Make the backing array independent so in-place updates in ImageTool do
            # not mutate the caller's original DataArray.
            data._variable = data.variable.copy(deep=False, data=data.data.copy())
        if data.size == 0:
            raise ValueError("Data must not be empty.")

        if data.ndim == 1:
            # Promote 1D data to 2D by adding a dummy dimension
            data = data.expand_dims("stack_dim", axis=-1)

        if data.ndim > 4:
            # Try squeezing
            data = data.squeeze()

        if data.ndim < 2:
            raise ValueError("Data must have at least two dimensions.")

        if data.ndim > 4:
            raise ValueError("Data must have at most four dimensions.")

        for d in data.dims:
            if data[d].ndim != 1:
                raise ValueError(f"Coordinate of dimension {d} is not one-dimensional.")

        # Handle loading non-uniform data saved in older versions.
        # erlab>=3.2.0 should not save non-uniform data in the first place.
        data = erlab.utils.array._restore_nonuniform_dims(data)

        # Convert coords to C-contiguous array
        data = erlab.utils.array.sort_coord_order(
            data.assign_coords(
                {d: data[d].astype(data[d].dtype, order="C") for d in data.dims}
            ),
            keys=data.coords.keys(),
            dims_first=False,
        )

        # Cast to float64 if not a floating point type (e.g. int)
        if data.dtype not in (np.float32, np.float64):
            data = data.astype(np.float64)

        return erlab.utils.array._make_dims_uniform(data)

    @classmethod
    def preflight_array(cls, data: xr.DataArray) -> None:
        """Check display compatibility without exposing renderer normalization."""
        cls.validate_array(data, copy_values=False)

    def _reset_property_cache(self, propname: str) -> None:
        self.__dict__.pop(propname, None)

    def _refresh_array_layout_cache(self) -> None:
        """Rebuild eager lookup tables derived from the current DataArray layout."""
        # These small lookup caches are coupled to the current dimension order and are
        # rebuilt wholesale whenever the backing DataArray changes.
        # Identify non-uniform axes created by the private conversion while avoiding
        # false positives when users provide their own *_idx dimensions.
        self._nonuniform_axes = []
        for i, d in enumerate(self._obj.dims):
            if erlab.utils.array._nonuniform_dim_name(self._obj, d) is not None:
                self._nonuniform_axes.append(i)
        self._nonuniform_axes_set: set[int] = set(self._nonuniform_axes)
        self._all_axes: tuple[int, ...] = tuple(range(self._obj.ndim))
        self._dim_indices: dict[Hashable, int] = {
            dim: i for i, dim in enumerate(self._obj.dims)
        }
        self._hidden_axes_cache: dict[tuple[int, ...], tuple[int, ...]] = {}
        self._hidden_axes_has_nonuniform_cache: dict[tuple[int, ...], bool] = {}

    def clear_dim_cache(self) -> None:
        """Clear cached properties related to dimensions.

        This method clears cached coordinate values, increments, limits, and the small
        argument-keyed memo tables derived from the current dimension layout.
        """
        for prop in (
            "coords",
            "associated_coord_dims",
            "coords_uniform",
            "incs",
            "incs_uniform",
            "lims",
            "lims_uniform",
            "uniform_index_params",
        ):
            self._reset_property_cache(prop)
        self._reset_hidden_axes_cache()

    def clear_val_cache(self) -> None:
        """Clear cached properties related to data values.

        This method clears the cached properties that depend on the data values, such as
        the global minima and maxima.
        """
        self._reset_property_cache("_raw_limits")
        self._reset_property_cache("limits")

    def clear_cache(self) -> None:
        """Clear all cached properties."""
        self.clear_dim_cache()
        self.clear_val_cache()

    def _reset_hidden_axes_cache(self) -> None:
        """Clear memoized hidden-axis selections derived from display tuples."""
        if hasattr(self, "_hidden_axes_cache"):
            self._hidden_axes_cache.clear()
        if hasattr(self, "_hidden_axes_has_nonuniform_cache"):
            self._hidden_axes_has_nonuniform_cache.clear()

    def _refresh_cursor_binned(self, cursor: int) -> None:
        """Refresh the cached binned-state tuple after mutating bin widths."""
        self._binned[cursor] = tuple(b != 1 for b in self._bins[cursor])

    def _normalize_cursor_axis_state(self) -> None:
        center_indices = _center_indices_for_shape(self._obj.shape)
        if not self._bins:
            self._bins.append([1] * self._obj.ndim)
        cursor_count = len(self._bins)
        del self._indices[cursor_count:]
        del self._values[cursor_count:]
        while len(self._indices) < cursor_count:
            self._indices.append(list(center_indices))
        while len(self._values) < cursor_count:
            self._values.append(
                [
                    coord[index]
                    for coord, index in zip(self.coords, center_indices, strict=True)
                ]
            )

        for cursor in range(cursor_count):
            bins = self._bins[cursor]
            indices = self._indices[cursor]
            values = self._values[cursor]
            normalized_bins: list[int] = []
            normalized_indices: list[int] = []
            normalized_values: list[np.floating] = []
            for axis in self._all_axes:
                normalized_bins.append(int(bins[axis]) if axis < len(bins) else 1)
                index_missing = axis >= len(indices)
                index = center_indices[axis] if index_missing else int(indices[axis])
                normalized_index = _normalized_axis_index(index, self._obj.shape[axis])
                index_clamped = normalized_index != index
                index = normalized_index
                normalized_indices.append(index)
                if index_missing or index_clamped or axis >= len(values):
                    normalized_values.append(self.coords[axis][index])
                else:
                    normalized_values.append(values[axis])
            self._bins[cursor] = normalized_bins
            self._indices[cursor] = normalized_indices
            self._values[cursor] = normalized_values

        self._binned = [tuple(b != 1 for b in bins) for bins in self._bins]

    def _hidden_axes_for_disp(self, disp: Sequence[int]) -> tuple[int, ...]:
        """Return cached hidden axes for a given displayed-axis selection."""
        key = tuple(disp)
        hidden = self._hidden_axes_cache.get(key)
        if hidden is None:
            # Display-axis combinations repeat frequently during cursor motion, so keep
            # a tiny per-layout memo instead of rebuilding these tuples every time.
            hidden = _hidden_axes_for_display(self._obj.ndim, key)
            self._hidden_axes_cache[key] = hidden
            self._hidden_axes_has_nonuniform_cache[key] = any(
                ax in self._nonuniform_axes_set for ax in hidden
            )
        return hidden

    def _hidden_axes_have_nonuniform(self, disp: Sequence[int]) -> bool:
        """Return whether any hidden axis uses non-uniform coordinates."""
        key = tuple(disp)
        has_nonuniform = self._hidden_axes_has_nonuniform_cache.get(key)
        if has_nonuniform is None:
            self._hidden_axes_for_disp(key)
            return self._hidden_axes_has_nonuniform_cache[key]
        return has_nonuniform

    def values_of_dim(self, dim: Hashable) -> npt.NDArray[np.floating]:
        """Fast equivalent of :code:`self._obj[dim].values`.

        Returns the cached pointer of the underlying coordinate array, achieving a ~80x
        speedup. This should work most of the time since we only assume floating point
        values, but does require further testing. May break for future versions of
        :obj:`pandas` or :obj:`xarray`. See Notes.

        Parameters
        ----------
        dim
            Name of the dimension to get the values from.

        Returns
        -------
        numpy.ndarray

        Notes
        -----
        Looking at the implementation, I think this may return a pandas array in some
        cases, but I'm not sure so I'll just leave it this way. When something breaks,
        replacing with :code:`self._obj._coords[dim]._data.array.array._ndarray` may
        do the trick.

        """
        return self._obj._coords[dim]._data.array._data  # type: ignore[union-attr]

    def get_significant(self, axis: int, uniform: bool = False) -> int:
        """Return the number of significant digits for a given axis."""
        if uniform and axis in self._nonuniform_axes_set:
            return 0  # Index axis, no decimals
        step = self.incs[axis]
        if step == 0:
            return 3  # Default to 3 decimal places for zero step size
        return erlab.utils.array.effective_decimals(step)

    def add_cursor(self, like_cursor: int = -1, update: bool = True) -> None:
        self._bins.append(list(self.get_bins(like_cursor)))
        self._binned.append(self._binned[like_cursor])
        new_ind = self.get_indices(like_cursor)
        self._indices.append(list(new_ind))
        self._values.append([c[i] for c, i in zip(self.coords, new_ind, strict=True)])
        if update:
            self.sigCursorCountChanged.emit(self.n_cursors)

    def remove_cursor(self, index: int, update: bool = True) -> None:
        if self.n_cursors == 1:
            raise ValueError("There must be at least one cursor.")
        self._bins.pop(index)
        self._binned.pop(index)
        self._indices.pop(index)
        self._values.pop(index)
        if update:
            self.sigCursorCountChanged.emit(self.n_cursors)

    def center_cursor(self, cursor: int, update: bool = True) -> None:
        self.set_indices(
            cursor,
            [s // 2 - (1 if s % 2 == 0 else 0) for s in self._obj.shape],
            update=update,
        )

    @QtCore.Slot(int, result=list)
    def get_bins(self, cursor: int) -> list[int]:
        return self._bins[cursor]

    @QtCore.Slot(int, result=list)
    def get_bin_values(self, cursor: int) -> list[float | None]:
        bins = self.get_bins(cursor)
        if self._nonuniform_axes:
            return [
                None if i in self._nonuniform_axes else b * np.abs(inc)
                for i, (b, inc) in enumerate(zip(bins, self.incs_uniform, strict=True))
            ]
        return [b * np.abs(inc) for b, inc in zip(bins, self.incs_uniform, strict=True)]

    def set_bins(
        self, cursor: int, value: list[int] | list[int | None], update: bool = True
    ) -> None:
        if not len(value) == self._obj.ndim:
            raise ValueError("length of bin array must match the number of dimensions.")
        axes: list[int | None] = []
        for i, x in enumerate(value):
            axes += self.set_bin(cursor, i, x, update=False)
        if update:
            self.sigBinChanged.emit(cursor, tuple(axes))

    @QtCore.Slot(int, int, int, bool, result=list)
    def set_bin(
        self, cursor: int, axis: int, value: int | None, update: bool = True
    ) -> list[int]:
        if value is None:
            return []
        if int(value) != value:
            raise TypeError("bins must have integer type")
        self._bins[cursor][axis] = int(value)
        self._refresh_cursor_binned(cursor)
        if update:
            self.sigBinChanged.emit(cursor, (axis,))
            return []
        return [axis]

    @QtCore.Slot(int, result=tuple)
    def get_binned(self, cursor: int) -> tuple[bool, ...]:
        """Return whether each axis is binned for the given cursor."""
        return self._binned[cursor]

    def is_binned(self, cursor: int) -> bool:
        """Return whether any axis is binned for the given cursor."""
        return any(self._binned[cursor])

    @QtCore.Slot(int, result=list)
    def get_indices(self, cursor: int) -> list[int]:
        return self._indices[cursor]

    @QtCore.Slot(int, int, result=int)
    def get_index(self, cursor: int, axis: int) -> int:
        return self._indices[cursor][axis]

    def set_indices(self, cursor: int, value: list[int], update: bool = True) -> None:
        if not len(value) == self._obj.ndim:
            raise ValueError(
                "length of index array must match the number of dimensions"
            )
        axes: list[int | None] = []
        for i, x in enumerate(value):
            axes += self.set_index(cursor, i, x, update=False)
        if update:
            self.sigIndexChanged.emit(cursor, tuple(axes))

    @QtCore.Slot(int, int, int, bool, result=list)
    def set_index(
        self, cursor: int, axis: int, value: int | None, update: bool = True
    ) -> list[int | None]:
        if value is None:
            return []
        if int(value) != value:
            raise TypeError("indices must have integer type")
        self._indices[cursor][axis] = int(value)
        self._values[cursor][axis] = self.coords[axis][int(value)]
        if update:
            self.sigIndexChanged.emit(cursor, (axis,))
            return []
        return [axis]

    @QtCore.Slot(int, int, int, bool)
    def step_index(
        self, cursor: int, axis: int, value: int, update: bool = True
    ) -> None:
        self._indices[cursor][axis] += value
        if (
            self._indices[cursor][axis] >= self._obj.shape[axis]
            or self._indices[cursor][axis] < 0
        ):
            self._indices[cursor][axis] -= value
            return
        self._values[cursor][axis] = self.coords[axis][self._indices[cursor][axis]]
        if update:
            self.sigIndexChanged.emit(cursor, (axis,))

    @QtCore.Slot(int, bool, result=list)
    def get_values(self, cursor: int, uniform: bool = False) -> list[np.floating]:
        if uniform and self._nonuniform_axes:
            val = list(self._values[cursor])
            for ax in self._nonuniform_axes:
                val[ax] = np.float32(self._indices[cursor][ax])
            return val
        return self._values[cursor]

    @QtCore.Slot(int, int, bool, result=float)
    def get_value(self, cursor: int, axis: int, uniform: bool = False) -> float:
        if uniform and axis in self._nonuniform_axes_set:
            return float(self._indices[cursor][axis])
        return float(self._values[cursor][axis])

    def set_values(self, cursor: int, values: list[float], update: bool = True) -> None:
        if not len(values) == self._obj.ndim:
            raise ValueError("length of values must match the number of dimensions")
        axes: list[int | None] = []
        for i, x in enumerate(values):
            axes += self.set_value(cursor, i, x, update=False)
        if update:
            self.sigIndexChanged.emit(cursor, tuple(axes))

    @QtCore.Slot(int, int, float, bool, bool, result=list)
    def set_value(
        self,
        cursor: int,
        axis: int,
        value: float | None,
        update: bool = True,
        uniform: bool = False,
    ) -> list[int | None]:
        if value is None:
            return []
        indices = self._indices[cursor]
        values = self._values[cursor]
        index = self.index_of_value(axis, value, uniform=uniform)
        indices[axis] = index
        if self.snap_to_data or (axis in self._nonuniform_axes_set):
            new = self.coords[axis][index]
            if values[axis] == new:
                return []
            values[axis] = new
        else:
            values[axis] = np.float64(value)
        if update:
            self.sigIndexChanged.emit(cursor, (axis,))
            return []
        return [axis]

    @QtCore.Slot(int, bool)
    def point_value(
        self, cursor: int, binned: bool = True
    ) -> npt.NDArray[np.floating] | np.floating | dask.array.Array:
        if binned:
            return self.extract_avg_slice(cursor, self._all_axes)
        return self._obj[tuple(self._indices[cursor])].values

    @QtCore.Slot(int, int)
    def swap_axes(self, ax1: int, ax2: int) -> None:
        if not 0 <= ax1 < self._obj.ndim or not 0 <= ax2 < self._obj.ndim:
            raise IndexError(
                f"axis indices {ax1} and {ax2} are incompatible with "
                f"{self._obj.ndim}-dimensional data"
            )
        self._normalize_cursor_axis_state()
        for i in range(self.n_cursors):
            self._bins[i][ax1], self._bins[i][ax2] = (
                self._bins[i][ax2],
                self._bins[i][ax1],
            )
            self._refresh_cursor_binned(i)
            self._values[i][ax1], self._values[i][ax2] = (
                self._values[i][ax2],
                self._values[i][ax1],
            )
            self._indices[i][ax1], self._indices[i][ax2] = (
                self._indices[i][ax2],
                self._indices[i][ax1],
            )
        dims_new = list(self._obj.dims)
        dims_new[ax1], dims_new[ax2] = dims_new[ax2], dims_new[ax1]

        self.set_array(self._obj.transpose(*dims_new), validate=False)

        self.sigShapeChanged.emit()

    def array_rect(
        self, i: int | None = None, j: int | None = None
    ) -> (
        tuple[np.floating, np.floating, np.floating, np.floating]
        | npt.NDArray[np.floating]
    ):
        if i is None:
            i = 0
        if j is None:
            return self.coords_uniform[i]
        return erlab.interactive.imagetool.fastslicing._array_rect(
            i, j, self.lims_uniform, self.incs_uniform
        )

    def value_of_index(
        self, axis: int, value: int, uniform: bool = False
    ) -> np.floating:
        """Get the value of the coordinate at the given index.

        Parameters
        ----------
        axis
            The axis to index into.
        value
            The index to get the value from.
        uniform
            This flag is only applied when the given `axis` corresponds to a
            non-uniform coordinate. If `True`, an index is returned. If `False`, the
            coordinate value at the given index is returned.

        """
        if uniform or (axis not in self._nonuniform_axes_set):
            return self.coords_uniform[axis][value]
        return self.coords[axis][value]

    def index_of_value(self, axis: int, value: float, uniform: bool = False) -> int:
        """Get the index of the coordinate closest to the given value.

        Parameters
        ----------
        axis
            The axis to search.
        value
            The value to search for.
        uniform
            This flag is only applied when the given `axis` corresponds to a non-uniform
            coordinate. If `True`, `value` is treated as an index. If `False`, the index
            corresponding to the coordinate with the closest value to `value` is
            returned.

        """
        if uniform or (axis not in self._nonuniform_axes_set):
            start, delta, upper = self.uniform_index_params[axis]
            if delta == 0:
                return 0
            index = round((value - start) / delta)
            if index <= 0:
                return 0
            return min(index, upper)

        return erlab.interactive.imagetool.fastslicing._index_of_value_nonuniform(
            self.coords[axis], value
        )

    def isel_args(
        self,
        cursor: int,
        disp: Sequence[int],
        int_if_one: bool = False,
        uniform: bool = False,
    ) -> dict[Hashable, slice | int]:
        out: dict[Hashable, slice | int] = {}
        for ax in self._hidden_axes_for_disp(disp):
            dim_name = self._obj.dims[ax]
            if ax in self._nonuniform_axes_set and not uniform:
                dim_name = str(dim_name).removesuffix("_idx")
            out[dim_name] = self._bin_slice(cursor, ax, int_if_one)
        return out

    def qsel_args(self, cursor: int, disp: Sequence[int]) -> dict:
        binned = self.get_binned(cursor)
        indexers = self.isel_args(cursor, disp, int_if_one=True)
        binned_dims: list[Hashable] = []

        for dim in indexers:
            axis_idx = self._dim_indices.get(dim)
            if axis_idx is None:
                axis_idx = self._obj.dims.index(dim)
            if binned[axis_idx]:
                binned_dims.append(dim)

        return qsel_args_from_indexers(self._obj, indexers, binned_dims)

    def qsel_code(self, cursor: int, disp: Sequence[int]) -> str:
        if self._hidden_axes_have_nonuniform(disp):
            # Has non-uniform axes, fallback to isel
            return self.isel_code(cursor, disp)

        try:
            qsel_kw = self.qsel_args(cursor, disp)
        except ValueError:
            return self.isel_code(cursor, disp)
        kwargs_str = erlab.interactive.utils.format_kwargs(qsel_kw)
        if kwargs_str:
            return f".qsel({kwargs_str})"
        return ""

    def isel_code(self, cursor: int, disp: Sequence[int]) -> str:
        kwargs_str = erlab.interactive.utils.format_kwargs(
            self.isel_args(cursor, disp, int_if_one=True)
        )
        if kwargs_str:
            return f".isel({kwargs_str})"
        return ""

    def xslice(self, cursor: int, disp: Sequence[int]) -> xr.DataArray:
        if not self._hidden_axes_have_nonuniform(disp):
            return self._obj.qsel(self.qsel_args(cursor, disp))

        isel_kw = self.isel_args(cursor, disp, int_if_one=False, uniform=True)
        binned_dims: list[Hashable] = [
            k
            for k, v in zip(self._obj.dims, self.get_binned(cursor), strict=True)
            if (v and (k in isel_kw))
        ]  # Select only relevant binned dimensions
        sliced = self._obj.isel(isel_kw).qsel.mean(binned_dims)
        if self._nonuniform_axes:
            return erlab.utils.array._restore_nonuniform_dims(sliced)
        return sliced

    @QtCore.Slot(int, tuple, result=np.ndarray)
    def slice_with_coord(
        self, cursor: int, disp: Sequence[int]
    ) -> tuple[
        tuple[np.floating, np.floating, np.floating, np.floating]
        | npt.NDArray[np.floating],
        npt.NDArray[np.floating] | np.floating | dask.array.Array,
    ]:
        axis = self._hidden_axes_for_disp(disp)
        data = self.extract_avg_slice(cursor, axis)
        if len(disp) == 2 and 0 in disp:
            data = data.T
        return self.array_rect(*disp), data

    def extract_avg_slice(
        self, cursor: int, axis: Sequence[int]
    ) -> npt.NDArray[np.floating] | np.floating | dask.array.Array:
        match len(axis):
            case 0:
                return self._obj.data
            case 1:
                return self._bin_along_axis(cursor, axis[0])
            case _:
                return self._bin_along_multiaxis(cursor, axis)

    def span_bounds(self, cursor: int, axis: int) -> npt.NDArray[np.floating]:
        slc = self._bin_slice(cursor, axis)
        if isinstance(slc, int):
            return self.coords_uniform[axis][slc : slc + 1]
        lb = max(0, slc.start)
        ub = min(self._obj.shape[axis] - 1, slc.stop - 1)
        return self.coords_uniform[axis][[lb, ub]]

    def _bin_slice(
        self, cursor: int, axis: int, int_if_one: bool = False
    ) -> slice | int:
        center = self._indices[cursor][axis]
        if self._binned[cursor][axis]:
            return _bin_slice_for_axis(
                center, self._bins[cursor][axis], self._obj.shape[axis]
            )
        if int_if_one:
            return center
        return slice(center, center + 1)

    def _axis_selection_key(
        self, cursor: int, axis: int
    ) -> int | tuple[int | None, int | None, int | None]:
        """Return the source selection identity for a cursor and axis."""
        if self._binned[cursor][axis]:
            selection = self._bin_slice(cursor, axis)
            if isinstance(selection, slice):
                return (selection.start, selection.stop, selection.step)
            return selection
        return self._indices[cursor][axis]

    def _bin_along_axis(
        self, cursor: int, axis: int
    ) -> npt.NDArray[np.floating] | np.floating | dask.array.Array:
        center = self._indices[cursor][axis]
        if not self._binned[cursor][axis]:
            return self._obj.data[(slice(None),) * axis + (center,)]
        selection = self._bin_slice(cursor, axis)
        return erlab.interactive.imagetool.fastbinning.fast_nanmean_skipcheck(
            self._obj.data[(slice(None),) * axis + (selection,)],
            axis=axis,
        )

    def _bin_along_multiaxis(
        self, cursor: int, axis: Sequence[int]
    ) -> npt.NDArray[np.floating] | np.floating | dask.array.Array:
        """Extract and optionally average over multiple hidden axes.

        Reduced axes with bin size ``1`` are indexed by integer, which drops them from
        the selected view. Reduced axes with larger bins are sliced and then averaged.
        The reduction axes therefore need to be remapped only in the mixed case where
        some reduced axes are indexed away while others remain for averaging.

        """
        # Internal callers pass sorted, unique axes.
        reduced_axes: Sequence[int] = axis
        selection, selected_axis, any_binned, all_binned = _reduced_axes_selection(
            self._obj.shape,
            reduced_axes,
            self._indices[cursor],
            self._bins[cursor],
            self._binned[cursor],
        )

        selected = self._obj.data[tuple(selection)]
        if not any_binned:
            return selected

        if all_binned:
            # No axis remapping is needed when every reduced axis is still present.
            return erlab.interactive.imagetool.fastbinning.fast_nanmean_skipcheck(
                selected, axis=reduced_axes
            )

        if selected.ndim == 1:
            # Mixed point-value requests can leave a single remaining binned axis.
            return erlab.interactive.imagetool.fastbinning.fast_nanmean(
                selected, axis=0
            )

        reduction_axis: int | tuple[int, ...] = (
            selected_axis[0] if len(selected_axis) == 1 else tuple(selected_axis)
        )
        return erlab.interactive.imagetool.fastbinning.fast_nanmean_skipcheck(
            selected, axis=reduction_axis
        )
