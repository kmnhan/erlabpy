"""Helper functions for fast slicing xarray.DataArray objects."""

import numba
import numbagg
import numpy as np
import numpy.typing as npt
import xarray as xr
from PySide6 import QtCore  # , QtGui, QtWidgets


@numba.njit(fastmath=True, cache=True)
def array_rect_jit(
    i: int, j: int, lims: tuple[tuple[float]], incs: tuple[float]
) -> tuple[float]:
    x = lims[i][0] - incs[i]
    y = lims[j][0] - incs[j]
    w = lims[i][-1] - x
    h = lims[j][-1] - y
    x += 0.5 * incs[i]
    y += 0.5 * incs[j]
    return x, y, w, h


@numba.njit(fastmath=True, cache=True)
def index_of_value_jit(
    axis: int,
    val: float,
    lims: tuple[tuple[float]],
    incs: tuple[float],
    shape: tuple[int],
) -> int:
    ind = min(
        round((val - lims[axis][0]) / incs[axis]),
        shape[axis] - 1,
    )
    if ind < 0:
        return 0
    return ind


@numba.njit(fastmath=True, cache=True)
def index_of_value_regular_jit(arr, val) -> int:
    return np.searchsorted((arr[:-1] + arr[1:]) / 2, val)


@numba.njit(fastmath=True, cache=True)
def transposed_jit(arr):
    if arr.ndim == 2:
        return arr.T
    elif arr.ndim == 3:
        return arr.transpose(1, 2, 0)
    elif arr.ndim == 4:
        return arr.transpose(1, 2, 3, 0)


class SlicerArray(QtCore.QObject):

    sigIndexChanged = QtCore.Signal(int, tuple)
    sigBinChanged = QtCore.Signal(int, tuple)
    sigCursorCountChanged = QtCore.Signal(int)
    sigShapeChanged = QtCore.Signal()

    def __init__(self, xarray_obj: xr.DataArray):
        super().__init__()
        self._obj = xarray_obj
        self._bins = [[1] * self._obj.ndim]
        self._indices = [[s // 2 - (1 if s % 2 == 0 else 0) for s in self._obj.shape]]
        self._values = [[c[i] for c, i in zip(self.coords, self._indices[0])]]
        self._snap_to_data = False

    def add_cursor(self, like_cursor: int = -1, update=True):
        self._bins.append(list(self.get_bins(like_cursor)))
        new_ind = self.get_indices(like_cursor)
        self._indices.append(list(new_ind))
        self._values.append([c[i] for c, i in zip(self.coords, new_ind)])
        if update:
            self.sigCursorCountChanged.emit(self.n_cursors)

    def remove_cursor(self, index, update=True):
        if self.n_cursors == 1:
            raise ValueError("There must be at least one cursor.")
        self._bins.pop(index)
        self._indices.pop(index)
        self._values.pop(index)
        if update:
            self.sigCursorCountChanged.emit(self.n_cursors)

    @property
    def snap_to_data(self) -> bool:
        return self._snap_to_data

    @snap_to_data.setter
    def snap_to_data(self, value: bool):
        self._snap_to_data = value

    @property
    def n_cursors(self) -> int:
        return len(self._bins)

    @QtCore.Slot(int, result=list[int])
    def get_bins(self, cursor) -> list[int]:
        return self._bins[cursor]

    def set_bins(self, cursor, value, update: bool = True):
        if not len(value) == self._obj.ndim:
            raise ValueError("length of bin array must match the number of dimensions.")
        axes = []
        for i, x in enumerate(value):
            axes += self.set_bin(cursor, i, x, update=False)
        if update:
            self.sigBinChanged.emit(cursor, tuple(axes))

    @QtCore.Slot(int, int, int, bool, result=list[int | None])
    def set_bin(self, cursor, axis, value, update: bool = True) -> list[int | None]:
        if value is None:
            return []
        if int(value) != value:
            raise TypeError("bins must have integer type")
        self._bins[cursor][axis] = int(value)
        if update:
            self.sigBinChanged.emit(cursor, self.get_bins(cursor))
        return [axis]

    @QtCore.Slot(int, result=tuple[bool])
    def get_binned(self, cursor) -> tuple[bool]:
        return tuple(b != 1 for b in self.get_bins(cursor))

    @QtCore.Slot(int, result=list[int])
    def get_indices(self, cursor) -> list[int]:
        return self._indices[cursor]

    def set_indices(self, cursor, value, update: bool = True):
        if not len(value) == self._obj.ndim:
            raise ValueError(
                "length of index array must match the number of dimensions"
            )
        axes = []
        for i, x in enumerate(value):
            axes += self.set_index(cursor, i, x, update=False)
        if update:
            self.sigIndexChanged.emit(cursor, tuple(axes))

    @QtCore.Slot(int, int, int, bool, result=list[int | None])
    def set_index(self, cursor, axis, value, update: bool = True) -> list[int | None]:
        if value is None:
            return []
        if int(value) != value:
            raise TypeError("indices must have integer type")
        self._indices[cursor][axis] = int(value)
        self._values[cursor][axis] = self.coords[axis][int(value)]
        if update:
            self.sigIndexChanged.emit(cursor, (axis,))
        return [axis]

    @QtCore.Slot(int, result=list[float])
    def get_values(self, cursor) -> list[float]:
        return self._values[cursor]

    def set_values(self, cursor, value, update: bool = True):
        if not len(value) == self._obj.ndim:
            raise ValueError(
                "length of value array must match the number of dimensions"
            )
        axes = []
        for i, x in enumerate(value):
            axes += self.set_value(cursor, i, x, update=False)
        if update:
            self.sigIndexChanged.emit(cursor, tuple(axes))

    @QtCore.Slot(int, int, float, bool, result=list[int | None])
    def set_value(self, cursor, axis, value, update: bool = True) -> list[int | None]:
        if value is None:
            return []
        self._indices[cursor][axis] = self.index_of_value(axis, value)
        if self.snap_to_data:
            self._values[cursor][axis] = self.coords[axis][self._indices[cursor][axis]]
        else:
            self._values[cursor][axis] = float(value)
        if update:
            self.sigIndexChanged.emit(cursor, (axis,))
        return [axis]

    @property
    def coords(self) -> tuple[npt.NDArray[np.float64]]:
        return tuple(self._obj[dim].values for dim in self._obj.dims)

    @property
    def incs(self) -> tuple[float]:
        return tuple(coord[1] - coord[0] for coord in self.coords)

    @property
    def lims(self) -> tuple[tuple[float]]:
        return tuple((coord[0], coord[-1]) for coord in self.coords)

    @property
    def data_vals_T(self):
        return transposed_jit(self._obj.values)

    def absnanmax(self, *args, **kwargs):
        return numbagg.nanmax(np.abs(self._obj.values), *args, **kwargs)

    def absnanmin(self, *args, **kwargs):
        return numbagg.nanmin(np.abs(self._obj.values), *args, **kwargs)

    def nanmax(self, *args, **kwargs):
        return numbagg.nanmax(self._obj.values, *args, **kwargs)

    def nanmin(self, *args, **kwargs):
        return numbagg.nanmin(self._obj.values, *args, **kwargs)

    def current_value(self, cursor):
        return self._obj.values[tuple(self.get_indices(cursor))]

    def swap_axes(self, ax1, ax2):
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
        self._obj = self._obj.transpose(*dims_new)

        self.sigShapeChanged.emit()

    def array_rect(self, i=None, j=None):
        if i is None:
            i = 0
        if j is None:
            return self.coords[i]
        return array_rect_jit(i, j, self.lims, self.incs)

    def index_of_value(self, axis, val):
        return index_of_value_jit(axis, val, self.lims, self.incs, self._obj.shape)

    @QtCore.Slot(int, tuple, result=npt.NDArray[np.float64])
    def slice_with_coord(self, cursor: int, axis: tuple):
        domain = sorted(set(range(self._obj.ndim)) - set(axis))
        return self.array_rect(*axis), self.extract_avg_slice(cursor, domain)

    def extract_avg_slice(self, cursor, axis=None):
        if axis is None:
            return self.data_vals_T
        if not np.iterable(axis):
            return self._bin_along_axis(cursor, axis)
        return self._bin_along_multiaxis(cursor, axis)

    def span_bounds(self, cursor, axis):
        slc = self._bin_slice(cursor, axis)
        lb = max(0, slc.start)
        ub = min(self._obj.shape[axis] - 1, slc.stop - 1)
        # ub = min(len(self.coords[axis]) - 1, slc.stop - 1)
        return self.coords[axis][[lb, ub]]

    def _bin_slice(self, cursor, axis):
        center = self.get_indices(cursor)[axis]
        if self.get_binned(cursor)[axis]:
            window = self.get_bins(cursor)[axis]
            return slice(center - window // 2, center + (window - 1) // 2 + 1)
        else:
            return slice(center, center + 1)

    def _bin_along_axis(self, cursor, axis):
        axis -= 1
        if not self.get_binned(cursor)[axis + 1]:
            return self.data_vals_T[
                (slice(None),) * (axis % self._obj.ndim)
                + (self._bin_slice(cursor, axis + 1),)
            ].squeeze(axis=axis)
        else:
            return numbagg.nanmean(
                self.data_vals_T[
                    (slice(None),) * (axis % self._obj.ndim)
                    + (self._bin_slice(cursor, axis + 1),)
                ],
                axis=axis,
            )

    def _bin_along_multiaxis(self, cursor, axis):
        if any(self.get_binned(cursor)):
            slices = [self._bin_slice(cursor, ax) for ax in axis]
        else:
            slices = [self.get_indices(cursor)[i] for i in axis]
        axis = [(ax - 1) % self._obj.ndim for ax in axis]
        selected = self.data_vals_T[
            tuple(
                slices[axis.index(d)] if d in axis else slice(None)
                for d in range(self._obj.ndim)
            )
        ]
        if any(self.get_binned(cursor)):
            return numbagg.nanmean(selected, axis=axis)
        else:
            return selected
