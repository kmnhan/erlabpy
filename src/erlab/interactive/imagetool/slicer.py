"""Helper functions for fast slicing :class:`xarray.DataArray` objects."""

from __future__ import annotations

__all__ = ["ArraySlicer"]

import copy
import functools
from typing import TYPE_CHECKING, TypedDict

import numba
import numpy as np
import numpy.typing as npt
from qtpy import QtCore

from erlab.interactive.imagetool.fastbinning import fast_nanmean_skipcheck
from erlab.interactive.utils import format_kwargs

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

    import xarray as xr

    class ArraySlicerState(TypedDict):
        dims: tuple[Hashable, ...]
        bins: list[list[int]]
        indices: list[list[int]]
        values: list[list[np.float32]]
        snap_to_data: bool


VALID_NDIM = (2, 3, 4)

_signature_array_rect = [
    numba.types.UniTuple(numba.float32, 4)(
        numba.int64,
        numba.int64,
        numba.types.UniTuple(numba.types.UniTuple(numba.float32, 2), i),
        numba.types.UniTuple(numba.float32, i),
    )
    for i in VALID_NDIM
]
_signature_index_of_value = [
    numba.int64(
        numba.int64,
        ftype,
        numba.types.UniTuple(numba.types.UniTuple(ftype, 2), i),
        numba.types.UniTuple(ftype, i),
        numba.types.UniTuple(numba.int64, i),
    )
    for ftype in (numba.float32, numba.float64)
    for i in VALID_NDIM
]


@numba.njit(_signature_array_rect, cache=True, fastmath=True)
def _array_rect(
    i: int,
    j: int,
    lims: tuple[tuple[np.float32, np.float32], ...],
    incs: tuple[np.float32, ...],
) -> tuple[np.float32, np.float32, np.float32, np.float32]:
    x = lims[i][0] - incs[i]
    y = lims[j][0] - incs[j]
    w = lims[i][-1] - x
    h = lims[j][-1] - y
    x += np.float32(0.5 * incs[i])
    y += np.float32(0.5 * incs[j])
    return x, y, w, h


@numba.njit(_signature_index_of_value, cache=True)
def _index_of_value(
    axis: int,
    val: np.floating,
    lims: tuple[tuple[np.floating, np.floating], ...],
    incs: tuple[np.floating, ...],
    shape: tuple[int],
) -> int:
    ind = min(
        round((val - lims[axis][0]) / incs[axis]),
        shape[axis] - 1,
    )
    if ind < 0:
        return 0
    return ind


@numba.njit(cache=True)
def _transposed(arr: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    if arr.ndim == 2:
        return arr.T
    elif arr.ndim == 3:
        return arr.transpose(1, 2, 0)
    else:
        return arr.transpose(1, 2, 3, 0)


@numba.njit(
    [numba.boolean(numba.float32[::1]), numba.boolean(numba.int64[::1])], cache=True
)
def _is_uniform(arr: npt.NDArray[np.float32]) -> bool:
    dif = np.diff(arr)
    return np.allclose(dif, dif[0], rtol=3e-05, atol=3e-05, equal_nan=True)


@numba.njit(
    [
        numba.int64(numba.float32[::1], numba.float32),
        numba.int64(numba.float64[::1], numba.float64),
    ],
    cache=True,
)
def _index_of_value_nonuniform(
    arr: npt.NDArray[np.float32], val: np.float32
) -> np.int_:
    return np.searchsorted((arr[:-1] + arr[1:]) / 2, val)


class ArraySlicer(QtCore.QObject):
    """Internal class used to slice a :class:`xarray.DataArray` rapidly.

    Computes binned line and image profiles from multiple cursors. Also handles the data
    indices and the number of bins for each cursor. Automatic conversion of non-uniform
    dimensions are also handled here.

    Parameters
    ----------
    xarray_obj
        A :class:`xarray.DataArray` with up to 4 dimensions.

    Signals
    -------
    sigIndexChanged(int, tuple)
    sigBinChanged(int, tuple)
    sigCursorCountChanged(int)
    sigShapeChanged()

    Note
    ----
    The original intent of this class was a xarray accessor. This is why `ArraySlicer`
    does not depend on a `ImageSlicerArea` but rather on the underlying
    `xarray.DataArray`. Originally, when loading a different array, a different instance
    of `ArraySlicer` had to be created. This was a terrible design choice since it
    messed up signals every time the instance was replaced. Hence, the behaviour was
    modified (23/06/19) so that the underlying `xarray.DataArray` of `ArraySlicer` could
    be swapped. As a consequence, each instance of `ImageSlicerArea` now corresponds to
    exactly one instance of `ArraySlicer`, regardless of the data. In the future,
    `ArraySlicer` might be changed so that it relies on its one-to-one correspondence
    with `ImageSlicerArea` for the signals.

    """

    sigIndexChanged = QtCore.Signal(int, object)  #: :meta private:
    sigBinChanged = QtCore.Signal(int, tuple)  #: :meta private:
    sigCursorCountChanged = QtCore.Signal(int)  #: :meta private:
    sigShapeChanged = QtCore.Signal()  #: :meta private:

    def __init__(self, xarray_obj: xr.DataArray):
        super().__init__()
        self.set_array(xarray_obj, validate=True, reset=True)

    @functools.cached_property
    def coords(self) -> tuple[npt.NDArray[np.float32], ...]:
        if self._nonuniform_axes:
            return tuple(
                (
                    self.values_of_dim(str(dim)[:-4])
                    if i in self._nonuniform_axes
                    else self.values_of_dim(dim)
                )
                for i, dim in enumerate(self._obj.dims)
            )
        else:
            return self.coords_uniform

    @functools.cached_property
    def coords_uniform(self) -> tuple[npt.NDArray[np.float32], ...]:
        return tuple(self.values_of_dim(dim) for dim in self._obj.dims)

    @functools.cached_property
    def incs(self) -> tuple[np.float32, ...]:
        return tuple(coord[1] - coord[0] for coord in self.coords)

    @functools.cached_property
    def incs_uniform(self) -> tuple[np.float32, ...]:
        return tuple(coord[1] - coord[0] for coord in self.coords_uniform)

    @functools.cached_property
    def lims(self) -> tuple[tuple[np.float32, np.float32], ...]:
        if self._nonuniform_axes:
            return tuple(
                (
                    (min(coord), max(coord))
                    if i in self._nonuniform_axes
                    else (coord[0], coord[-1])
                )
                for i, coord in enumerate(self.coords)
            )
        else:
            return tuple((coord[0], coord[-1]) for coord in self.coords)

    @functools.cached_property
    def lims_uniform(self) -> tuple[tuple[np.float32, np.float32], ...]:
        return tuple((coord[0], coord[-1]) for coord in self.coords_uniform)

    @functools.cached_property
    def data_vals_T(self) -> npt.NDArray[np.floating]:
        return _transposed(self._obj.values)

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

    @functools.cached_property
    def absnanmin(self) -> float:
        mn, mx = self.nanmin, self.nanmax
        if mn * mx <= np.float32(0.0):
            return 0.0
        elif mn < np.float32(0.0):
            return -mx
        else:
            return mn

    @property
    def limits(self) -> tuple[float, float]:
        """Return the global minima and maxima of the data."""
        return self.nanmin, self.nanmax

    @staticmethod
    def validate_array(data: xr.DataArray) -> xr.DataArray:
        """Validate a given :class:`xarray.DataArray`.

        If data has two momentum axes (``kx`` and ``ky``), set them (and ``eV`` if
        exists) as the first two (or three) dimensions. Then, checks the data for
        non-uniform coordinates, which are converted to indices. Finally, converts the
        coordinates to C-contiguous float32. If input data values neither float32 nor
        float64, a conversion to float64 is attempted.

        Parameters
        ----------
        data
            Input array with at least two dimensions.

        Returns
        -------
        xarray.DataArray
            The converted data.

        """
        if data.ndim < 2:
            raise ValueError("Data must have at least two dimensions.")

        if data.ndim > 4:
            raise ValueError("Data must have at most four dimensions.")

        # convert coords to C-contiguous float32
        data = data.assign_coords(
            {d: data[d].astype(np.float32, order="C") for d in data.dims}
        )

        if data.dtype not in (np.float32, np.float64):
            data = data.astype(np.float64)

        new_dims: tuple[str, ...] = ("kx", "ky")
        if all(d in data.dims for d in new_dims):
            # if data has kx and ky axis, transpose
            if "eV" in data.dims:
                new_dims += ("eV",)
            new_dims += tuple(str(d) for d in data.dims if d not in new_dims)
            data = data.transpose(*new_dims)

        nonuniform_dims: list[str] = [
            str(d) for d in data.dims if not _is_uniform(data[d].values)
        ]
        for d in nonuniform_dims:
            data = data.assign_coords(
                {d + "_idx": (d, list(np.arange(len(data[d]), dtype=np.float32)))}
            ).swap_dims({d: d + "_idx"})

        return data

    def reset_property_cache(self, propname: str) -> None:
        self.__dict__.pop(propname, None)

    def clear_dim_cache(self, include_vals: bool = False):
        for prop in (
            "coords",
            "coords_uniform",
            "incs",
            "incs_uniform",
            "lims",
            "lims_uniform",
        ):
            self.reset_property_cache(prop)

        if include_vals:
            self.reset_property_cache("data_vals_T")

    def clear_val_cache(self, include_vals: bool = False):
        for prop in ("nanmax", "nanmin", "absnanmax", "absnanmin"):
            self.reset_property_cache(prop)

        if include_vals:
            self.reset_property_cache("data_vals_T")

    def clear_cache(self):
        self.clear_dim_cache()
        self.clear_val_cache(include_vals=True)

    def set_array(
        self, xarray_obj: xr.DataArray, validate: bool = True, reset: bool = False
    ) -> None:
        if hasattr(self, "_obj"):
            del self._obj

        if validate:
            self._obj: xr.DataArray = self.validate_array(xarray_obj)
        else:
            self._obj = xarray_obj
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
            self._values: list[list[np.float32]] = [
                [c[i] for c, i in zip(self.coords, self._indices[0], strict=True)]
            ]
            self.snap_to_data: bool = False

    def values_of_dim(self, dim: Hashable) -> npt.NDArray[np.float32]:
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
        else:
            return [
                b * np.abs(inc) for b, inc in zip(bins, self.incs_uniform, strict=True)
            ]

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
        self, cursor: int, axis: int, value: int, update: bool = True
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
        self, cursor: int, axis: int, value: int, update: bool = True
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
    def get_values(self, cursor: int, uniform: bool = False) -> list[np.float32]:
        if uniform and self._nonuniform_axes:
            val = list(self._values[cursor])
            for ax in self._nonuniform_axes:
                val[ax] = np.float32(self._indices[cursor][ax])
            return val
        else:
            return self._values[cursor]

    @QtCore.Slot(int, int, bool, result=np.float32)
    def get_value(self, cursor: int, axis: int, uniform: bool = False) -> np.float32:
        if uniform and axis in self._nonuniform_axes:
            return np.float32(self._indices[cursor][axis])
        else:
            return self._values[cursor][axis]

    def set_values(
        self, cursor: int, value: list[np.float32], update: bool = True
    ) -> None:
        if not len(value) == self._obj.ndim:
            raise ValueError(
                "length of value array must match the number of dimensions"
            )
        axes: list[int | None] = []
        for i, x in enumerate(value):
            axes += self.set_value(cursor, i, x, update=False)
        if update:
            self.sigIndexChanged.emit(cursor, tuple(axes))

    @QtCore.Slot(int, int, np.float32, bool, bool, result=list)
    def set_value(
        self,
        cursor: int,
        axis: int,
        value: np.float32,
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
            self._values[cursor][axis] = np.float32(value)
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
        else:
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
        tuple[np.float32, np.float32, np.float32, np.float32] | npt.NDArray[np.float32]
    ):
        if i is None:
            i = 0
        if j is None:
            return self.coords_uniform[i]
        return _array_rect(i, j, self.lims_uniform, self.incs_uniform)

    def value_of_index(
        self, axis: int, value: int, uniform: bool = False
    ) -> np.float32:
        if uniform or (axis not in self._nonuniform_axes):
            return self.coords_uniform[axis][value]
        else:
            return self.coords[axis][value]

    def index_of_value(
        self, axis: int, value: np.float32, uniform: bool = False
    ) -> int:
        if uniform or (axis not in self._nonuniform_axes):
            return _index_of_value(
                axis, value, self.lims_uniform, self.incs_uniform, self._obj.shape
            )
        else:
            return _index_of_value_nonuniform(self.coords[axis], value)

    def isel_args(
        self, cursor: int, disp: Sequence[int], int_if_one: bool = False
    ) -> dict[str, slice | int]:
        axis: list[int] = sorted(set(range(self._obj.ndim)) - set(disp))
        return {
            str(self._obj.dims[ax]).rstrip("_idx")
            if ax in self._nonuniform_axes
            else str(self._obj.dims[ax]): self._bin_slice(cursor, ax, int_if_one)
            for ax in axis
        }

    def qsel_args(self, cursor: int, disp: Sequence[int]) -> dict:
        out: dict[str, float] = {}
        binned = self.get_binned(cursor)

        for dim, selector in self.isel_args(cursor, disp, int_if_one=True).items():
            inc = self.incs[self._obj.dims.index(dim)]
            order = int(-np.floor(np.log10(inc)) + 1)

            if binned[self._obj.dims.index(dim)]:
                coord = self._obj[dim][selector].values

                out[dim] = np.round(coord.mean(), order)
                width = np.round(abs(coord[-1] - coord[0]) + inc, order)

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
                out[dim] = np.round(self._obj[dim].values[selector], order)

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

        return f".qsel({format_kwargs(qsel_kw)})"

    def isel_code(self, cursor: int, disp: Sequence[int]) -> str:
        return f".isel({format_kwargs(self.isel_args(cursor, disp, int_if_one=True))})"

    def xslice(self, cursor: int, disp: Sequence[int]) -> xr.DataArray:
        isel_kw = self.isel_args(cursor, disp, int_if_one=False)
        binned_coord_average: dict[str, xr.DataArray] = {
            str(k): self._obj[k][isel_kw[str(k)]].mean()
            for k, v in zip(self._obj.dims, self.get_binned(cursor), strict=True)
            if v
        }
        return (
            self._obj.isel(isel_kw)
            .squeeze()
            .mean(binned_coord_average.keys())
            .assign_coords(binned_coord_average)
        )

    @QtCore.Slot(int, tuple, result=np.ndarray)
    def slice_with_coord(
        self, cursor: int, disp: Sequence[int]
    ) -> tuple[
        tuple[np.float32, np.float32, np.float32, np.float32] | npt.NDArray[np.float32],
        npt.NDArray[np.float32] | np.float32,
    ]:
        axis = sorted(set(range(self._obj.ndim)) - set(disp))
        return self.array_rect(*disp), self.extract_avg_slice(cursor, axis)

    def extract_avg_slice(
        self, cursor: int, axis: Sequence[int]
    ) -> npt.NDArray[np.floating] | np.floating:
        if len(axis) == 0:
            return self.data_vals_T
        elif len(axis) == 1:
            return self._bin_along_axis(cursor, axis[0])
        else:
            return self._bin_along_multiaxis(cursor, axis)

    def span_bounds(self, cursor: int, axis: int) -> npt.NDArray[np.float32]:
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
        elif int_if_one:
            return center
        else:
            return slice(center, center + 1)

    def _bin_along_axis(
        self, cursor: int, axis: int
    ) -> npt.NDArray[np.floating] | np.floating:
        axis_val = (axis - 1) % self._obj.ndim
        if not self.get_binned(cursor)[axis]:
            return self.data_vals_T[
                (slice(None),) * axis_val + (self._bin_slice(cursor, axis),)
            ].squeeze(axis=axis_val)
        else:
            return fast_nanmean_skipcheck(
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
            return fast_nanmean_skipcheck(selected, axis=axis)
        else:
            return selected

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
            "values": copy.deepcopy(self._values),
            "snap_to_data": bool(self.snap_to_data),
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
