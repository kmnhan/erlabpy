"""Helper functions for fast slicing :class:`xarray.DataArray` objects."""

from __future__ import annotations

import copy
import functools
import importlib
import typing

import numpy as np
import numpy.typing as npt
import xarray as xr
from qtpy import QtCore, QtWidgets

import erlab

if typing.TYPE_CHECKING:
    from collections.abc import Hashable, Sequence


class ArraySlicerState(typing.TypedDict):
    """A dictionary containing the state of cursors in an :class:`ArraySlicer`."""

    dims: tuple[Hashable, ...]
    bins: list[list[int]]
    indices: list[list[int]]
    values: list[list[float]]
    snap_to_data: bool
    twin_coord_names: typing.NotRequired[tuple[Hashable, ...]]


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
    if set(old.dims) != set(new.dims):
        return False
    for d in old.dims:
        if not np.isin(old[d].values, new[d].values, assume_unique=True).all():
            return False
    return True


def make_dims_uniform(darr: xr.DataArray) -> xr.DataArray:
    """Ensure that all dimensions of the given DataArray are uniform.

    This function checks each dimension of the input DataArray to determine if its
    coordinate is evenly spaced. If a dimension is found to be non-uniform, a new
    coordinate named ``{dim}_idx`` is created with indices ranging from 0 to N-1, where
    N is the length of the original coordinate. The original dimension is then swapped
    with the new uniform dimension, and is left in the DataArray as a coordinate.

    Parameters
    ----------
    darr
        The input DataArray to be processed.

    Returns
    -------
    DataArray
        A new DataArray with all dimensions made uniform.
    """
    nonuniform_dims: list[str] = [
        str(d)
        for d in darr.dims
        if not erlab.interactive.imagetool.fastslicing._is_uniform(
            darr[d].values.astype(np.float64)
        )
    ]
    for d in nonuniform_dims:
        darr = darr.assign_coords(
            {d + "_idx": (d, list(np.arange(len(darr[d]), dtype=np.float32)))}
        ).swap_dims({d: d + "_idx"})

    return darr


def restore_nonuniform_dims(darr: xr.DataArray) -> xr.DataArray:
    """Undo the effect of :func:`make_dims_uniform`.

    Restore non-uniform dimensions by swapping dimensions that end with ``'_idx'`` with
    their corresponding coordinates and dropping the uniform dimensions.

    Parameters
    ----------
    darr
        The input DataArray with dimensions that may end with ``'_idx'``.

    Returns
    -------
    DataArray
        The DataArray with ``'_idx'`` dimensions swapped with their corresponding
        coordinates and the ``'_idx'`` dimensions dropped.
    """
    nonuniform_dims: list[Hashable] = []
    for d in darr.dims:
        if str(d).endswith("_idx"):
            stripped = str(d).removesuffix("_idx")
            if stripped in darr.coords:
                nonuniform_dims.append(d)
                darr = darr.swap_dims({d: stripped})
    return darr.drop_vars(nonuniform_dims)


def _get_inc(coord):
    try:
        return coord[1] - coord[0]
    except IndexError:
        return 0


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
    sigIndexChanged(int, tuple)
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
    <erlab.interactive.imagetool.core.ImageSlicerArea>` but rather on the underlying
    `xarray.DataArray`. Originally, when loading a different array, a different instance
    of `ArraySlicer` had to be created. This was a terrible design choice since it
    messed up signals every time the instance was replaced. Hence, the behaviour was
    modified (23/06/19) so that the underlying `xarray.DataArray` of `ArraySlicer` could
    be swapped. As a consequence, each instance of :class:`ImageSlicerArea
    <erlab.interactive.imagetool.core.ImageSlicerArea>` now corresponds to exactly one
    instance of `ArraySlicer`, regardless of the data. In the future, `ArraySlicer`
    might be changed so that it relies on its one-to-one correspondence with
    :class:`ImageSlicerArea <erlab.interactive.imagetool.core.ImageSlicerArea>` for the
    signals.

    """

    sigIndexChanged = QtCore.Signal(int, object)  #: :meta private:
    sigBinChanged = QtCore.Signal(int, tuple)  #: :meta private:
    sigCursorCountChanged = QtCore.Signal(int)  #: :meta private:
    sigShapeChanged = QtCore.Signal()  #: :meta private:
    sigTwinChanged = QtCore.Signal()  #: :meta private:

    def __init__(self, xarray_obj: xr.DataArray) -> None:
        super().__init__()
        self.snap_act = QtWidgets.QAction("&Snap to Pixels", self)
        self.snap_act.setShortcut("S")
        self.snap_act.setCheckable(True)
        self.snap_act.setChecked(False)
        self.snap_act.setToolTip("Snap cursors to data points")

        self.set_array(xarray_obj, validate=True, reset=True)

        # Preload to prevent hanging on initial bin
        importlib.import_module("erlab.interactive.imagetool.fastbinning")

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
        self, xarray_obj: xr.DataArray, validate: bool = True, reset: bool = False
    ) -> None:
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

        """
        obj_original: xr.DataArray | None = None
        if hasattr(self, "_obj"):
            obj_original = self._obj.copy()
            del self._obj

        if validate:
            self._obj: xr.DataArray = self.validate_array(xarray_obj)
        else:
            self._obj = xarray_obj

        if (obj_original is not None) and reset:
            # If same coords, keep cursors
            if check_cursors_compatible(obj_original, self._obj):
                self._obj = self._obj.transpose(*obj_original.dims)
                reset = False
            del obj_original

        # TODO: This is not robust, may break if user supplies dim that ends with "_idx"
        # Need to find a better way to handle this.
        self._nonuniform_axes: list[int] = [
            i for i, d in enumerate(self._obj.dims) if str(d).endswith("_idx")
        ]

        self.clear_dim_cache(include_vals=True)
        if validate:
            self.clear_val_cache(include_vals=False)

        if reset:
            self._bins: list[list[int]] = [[1] * self._obj.ndim]
            self._indices: list[list[int]] = [
                [s // 2 - (1 if s % 2 == 0 else 0) for s in self._obj.shape]
            ]
            self._values: list[list[np.floating]] = [
                [c[i] for c, i in zip(self.coords, self._indices[0], strict=True)]
            ]
            self._twin_coord_names = set()
            self.snap_to_data = False
        else:
            # Update twin axes on reload
            self.sigTwinChanged.emit()

    @functools.cached_property
    def associated_coords(
        self,
    ) -> dict[str, dict[str, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]]:
        out: dict[
            str, dict[str, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]
        ] = {str(d): {} for d in self._obj.dims}
        for k, coord in self._obj.coords.items():
            if (
                isinstance(coord, xr.DataArray)
                and len(coord.dims) == 1
                and str(coord.dims[0]) != k
            ):
                out[str(coord.dims[0])][str(k)] = (
                    coord[coord.dims[0]].values.astype(np.float64),
                    coord.values.astype(np.float64),
                )
        return out

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
        """Increment size of each dimension in the array.

        Returns the step size of each dimension in the array. Non-uniform dimensions
        will return the absolute average of all non-zero step sizes.
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
                    (min(coord), max(coord))
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
    def data_vals_T(self) -> npt.NDArray[np.floating]:
        """Transposed data values.

        This property is used for fast slicing and binning operations.
        """
        return erlab.interactive.imagetool.fastslicing._transposed(self._obj.values)

    # Benchmarks result in 10~20x slower speeds for bottleneck and numbagg compared to
    # numpy on arm64 mac with Accelerate BLAS. Needs confirmation on intel systems.
    @functools.cached_property
    def nanmax(self) -> float:
        return float(np.nanmax(self._obj.values))

    @functools.cached_property
    def nanmin(self) -> float:
        return float(np.nanmin(self._obj.values))

    @functools.cached_property
    def absnanmax(self) -> float:
        return max(abs(self.nanmin), abs(self.nanmax))

    @property
    def limits(self) -> tuple[float, float]:
        """Return the global minima and maxima of the data."""
        return self.nanmin, self.nanmax

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
        }

    @state.setter
    def state(self, state: ArraySlicerState) -> None:
        if self._obj.dims != state["dims"]:
            self._obj = self._obj.transpose(*state["dims"])

        self.snap_to_data = state["snap_to_data"]
        self.clear_cache()

        for i, (bins, indices, values) in enumerate(
            zip(state["bins"], state["indices"], state["values"], strict=True)
        ):
            self.center_cursor(i)
            self.set_indices(i, indices, update=False)
            self.set_values(i, values, update=True)
            self.set_bins(i, bins, update=True)

        self.twin_coord_names = set(state.get("twin_coord_names", set()))

        self.set_array(self._obj, validate=False)
        # Not sure why the last line is needed but the cursor is not fully restored
        # without it, and requires a transpose before the cursor position and the
        # dimension label is correct. Must be a bug in the order of operations or
        # something... reproduce by archiving a 3D array with one non-uniform axis.

    @staticmethod
    def validate_array(data: xr.DataArray) -> xr.DataArray:
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

        Returns
        -------
        xarray.DataArray
            The converted data.

        """
        data = data.copy().squeeze()

        if data.size == 0:
            raise ValueError("Data must not be empty.")

        if data.ndim < 2:
            raise ValueError("Data must have at least two dimensions.")

        if data.ndim > 4:
            raise ValueError("Data must have at most four dimensions.")

        for d in data.dims:
            if data[d].ndim != 1:
                raise ValueError(f"Coordinate of dimension {d} is not one-dimensional.")

        # Handle loading non-uniform data saved in older versions.
        # erlab>=3.2.0 should not save non-uniform data in the first place.
        data = restore_nonuniform_dims(data)

        # Convert coords to C-contiguous array
        data = data.assign_coords(
            {d: data[d].astype(data[d].dtype, order="C") for d in data.dims}
        )

        if data.dtype not in (np.float32, np.float64):
            data = data.astype(np.float64)

        return make_dims_uniform(data)

    def _reset_property_cache(self, propname: str) -> None:
        self.__dict__.pop(propname, None)

    def clear_dim_cache(self, include_vals: bool = False) -> None:
        """Clear cached properties related to dimensions.

        This method clears the cached coordinate values, increments, and limits.

        Parameters
        ----------
        include_vals
            Whether to clear the cache that contains the transposed data values.

        """
        for prop in (
            "coords",
            "associated_coords",
            "coords_uniform",
            "incs",
            "incs_uniform",
            "lims",
            "lims_uniform",
        ):
            self._reset_property_cache(prop)

        if include_vals:
            self._reset_property_cache("data_vals_T")

    def clear_val_cache(self, include_vals: bool = False) -> None:
        """Clear cached properties related to data values.

        This method clears the cached properties that depend on the data values, such as
        the global minima and maxima.

        Parameters
        ----------
        include_vals
            Whether to clear the cache that contains the transposed data values.

        """
        for prop in ("nanmax", "nanmin", "absnanmax"):
            self._reset_property_cache(prop)

        if include_vals:
            self._reset_property_cache("data_vals_T")

    def clear_cache(self) -> None:
        """Clear all cached properties."""
        self.clear_dim_cache()
        self.clear_val_cache(include_vals=True)

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
        if uniform and axis in self._nonuniform_axes:
            step = self.incs_uniform[axis]
        else:
            step = self.incs[axis]
        if step == 0:
            return 3  # Default to 3 decimal places for zero step size
        return erlab.utils.array.effective_decimals(step)

    def add_cursor(self, like_cursor: int = -1, update: bool = True) -> None:
        self._bins.append(list(self.get_bins(like_cursor)))
        new_ind = self.get_indices(like_cursor)
        self._indices.append(list(new_ind))
        self._values.append([c[i] for c, i in zip(self.coords, new_ind, strict=True)])
        if update:
            self.sigCursorCountChanged.emit(self.n_cursors)

    def remove_cursor(self, index: int, update: bool = True) -> None:
        if self.n_cursors == 1:
            raise ValueError("There must be at least one cursor.")
        self._bins.pop(index)
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
        if update:
            self.sigBinChanged.emit(cursor, (axis,))
            return []
        return [axis]

    @QtCore.Slot(int, result=tuple)
    def get_binned(self, cursor: int) -> tuple[bool, ...]:
        return tuple(b != 1 for b in self.get_bins(cursor))

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
        if uniform and axis in self._nonuniform_axes:
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
        self._indices[cursor][axis] = self.index_of_value(axis, value, uniform=uniform)
        if self.snap_to_data or (axis in self._nonuniform_axes):
            new = self.coords[axis][self._indices[cursor][axis]]
            if self._values[cursor][axis] == new:
                return []
            self._values[cursor][axis] = new
        else:
            self._values[cursor][axis] = np.float64(value)
        if update:
            self.sigIndexChanged.emit(cursor, (axis,))
            return []
        return [axis]

    @QtCore.Slot(int, bool, result=np.floating)
    def point_value(
        self, cursor: int, binned: bool = True
    ) -> npt.NDArray[np.floating] | np.floating:
        if binned:
            return self.extract_avg_slice(cursor, tuple(range(self._obj.ndim)))
        return self._obj.values[tuple(self.get_indices(cursor))]

    @QtCore.Slot(int, int)
    def swap_axes(self, ax1: int, ax2: int) -> None:
        for i in range(self.n_cursors):
            self._bins[i][ax1], self._bins[i][ax2] = (
                self._bins[i][ax2],
                self._bins[i][ax1],
            )
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
        if uniform or (axis not in self._nonuniform_axes):
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
        if uniform or (axis not in self._nonuniform_axes):
            return erlab.interactive.imagetool.fastslicing._index_of_value(
                axis, value, self.lims_uniform, self.incs_uniform, self._obj.shape
            )

        return erlab.interactive.imagetool.fastslicing._index_of_value_nonuniform(
            self.coords[axis], value
        )

    def isel_args(
        self,
        cursor: int,
        disp: Sequence[int],
        int_if_one: bool = False,
        uniform: bool = False,
    ) -> dict[str, slice | int]:
        axis: list[int] = sorted(set(range(self._obj.ndim)) - set(disp))
        return {
            str(self._obj.dims[ax]).removesuffix("_idx")
            if (ax in self._nonuniform_axes and not uniform)
            else str(self._obj.dims[ax]): self._bin_slice(cursor, ax, int_if_one)
            for ax in axis
        }

    def qsel_args(self, cursor: int, disp: Sequence[int]) -> dict:
        out: dict[str, float] = {}
        binned = self.get_binned(cursor)

        for dim, selector in self.isel_args(cursor, disp, int_if_one=True).items():
            axis_idx = self._obj.dims.index(dim)
            inc = self.incs[axis_idx]
            # Estimate minimum number of decimal places required to represent selection
            order = self.get_significant(axis_idx)

            if binned[axis_idx]:
                coord = self._obj[dim][selector].values

                out[dim] = float(np.round(coord.mean(), order))
                width = float(np.round(abs(coord[-1] - coord[0]) + inc, order))

                if not np.allclose(
                    self._obj[dim]
                    .sel({dim: slice(out[dim] - width / 2, out[dim] + width / 2)})
                    .values,
                    coord,
                ):
                    raise ValueError(
                        "Bin does not contain the same values as the original data."
                    )

                out[dim + "_width"] = width

            else:
                out[dim] = float(np.round(self._obj[dim].values[selector], order))

        return out

    def qsel_code(self, cursor: int, disp: Sequence[int]) -> str:
        if any(
            a in self._nonuniform_axes for a in set(range(self._obj.ndim)) - set(disp)
        ):
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
        if not any(
            a in self._nonuniform_axes for a in set(range(self._obj.ndim)) - set(disp)
        ):
            return self._obj.qsel(self.qsel_args(cursor, disp))

        isel_kw = self.isel_args(cursor, disp, int_if_one=False, uniform=True)
        binned_dims: list[Hashable] = [
            k
            for k, v in zip(self._obj.dims, self.get_binned(cursor), strict=True)
            if (v and (k in isel_kw))
        ]  # Select only relevant binned dimensions
        binned_coords_averaged: dict[str, xr.DataArray] = {
            str(k): self._obj[k][isel_kw[str(k)]].mean() for k in binned_dims
        }
        # !TODO: we may lose some coords here, like dims that depend on the binned dims
        sliced = (
            self._obj.isel(isel_kw)
            .mean(binned_dims)
            .assign_coords(binned_coords_averaged)
            .squeeze()  # is squeeze needed here?
        )
        if self._nonuniform_axes:
            return restore_nonuniform_dims(sliced)
        return sliced

    @QtCore.Slot(int, tuple, result=np.ndarray)
    def slice_with_coord(
        self, cursor: int, disp: Sequence[int]
    ) -> tuple[
        tuple[np.floating, np.floating, np.floating, np.floating]
        | npt.NDArray[np.floating],
        npt.NDArray[np.floating] | np.floating,
    ]:
        axis = sorted(set(range(self._obj.ndim)) - set(disp))
        return self.array_rect(*disp), self.extract_avg_slice(cursor, axis)

    def extract_avg_slice(
        self, cursor: int, axis: Sequence[int]
    ) -> npt.NDArray[np.floating] | np.floating:
        if len(axis) == 0:
            return self.data_vals_T
        if len(axis) == 1:
            return self._bin_along_axis(cursor, axis[0])
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
        center = self.get_indices(cursor)[axis]
        if self.get_binned(cursor)[axis]:
            window = self.get_bins(cursor)[axis]
            return slice(center - window // 2, center + (window - 1) // 2 + 1)
        if int_if_one:
            return center
        return slice(center, center + 1)

    def _bin_along_axis(
        self, cursor: int, axis: int
    ) -> npt.NDArray[np.floating] | np.floating:
        axis_val = (axis - 1) % self._obj.ndim
        if not self.get_binned(cursor)[axis]:
            return self.data_vals_T[
                (slice(None),) * axis_val + (self._bin_slice(cursor, axis),)
            ].squeeze(axis=axis_val)
        return erlab.interactive.imagetool.fastbinning.fast_nanmean_skipcheck(
            self.data_vals_T[
                (slice(None),) * axis_val + (self._bin_slice(cursor, axis),)
            ],
            axis=axis_val,
        )

    def _bin_along_multiaxis(
        self, cursor: int, axis: Sequence[int]
    ) -> npt.NDArray[np.floating] | np.floating:
        if any(self.get_binned(cursor)):
            slices = tuple(self._bin_slice(cursor, ax) for ax in axis)
        else:
            slices = tuple(self.get_indices(cursor)[i] for i in axis)
        axis = tuple((ax - 1) % self._obj.ndim for ax in axis)
        selected = self.data_vals_T[
            tuple(
                slices[axis.index(d)] if d in axis else slice(None)
                for d in range(self._obj.ndim)
            )
        ]
        if any(self.get_binned(cursor)):
            return erlab.interactive.imagetool.fastbinning.fast_nanmean_skipcheck(
                selected, axis=axis
            )
        return selected
