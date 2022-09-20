"""Interactive plotting, mainly with Qt and pyqtgraph."""
# from . import utilities

import xarray as xr
import numpy as np
import numbagg
import numba


@numba.njit(fastmath=True, cache=True)
def array_rect_jit(i, j, lims, incs):
    x = lims[i][0] - incs[i]
    y = lims[j][0] - incs[j]
    w = lims[i][-1] - x
    h = lims[j][-1] - y
    x += 0.5 * incs[i]
    y += 0.5 * incs[j]
    return x, y, w, h

@numba.njit(fastmath=True, cache=True)
def index_of_value_jit(axis, val, lims, incs, shape):
    ind = min(
        round((val - lims[axis][0]) / incs[axis]),
        shape[axis] - 1,
    )
    if ind < 0:
        return 0
    return ind

@numba.njit(fastmath=True, cache=True)
def index_of_value_regular_jit(arr, val):
    return np.searchsorted((arr[:-1] + arr[1:]) / 2, val)


@numba.njit(fastmath=True, cache=True)
def return_transposed_jit(arr):
    if arr.ndim == 2:
        return arr.T
    elif arr.ndim == 3:
        return arr.transpose(1, 2, 0)
    elif arr.ndim == 4:
        return arr.transpose(1, 2, 3, 0)



@xr.register_dataarray_accessor("_slicer")
class SlicerArray:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._bins = [[1] * self._obj.ndim]
        self._indices = [[s // 2 - (1 if s % 2 == 0 else 0) for s in self._obj.shape]]
        self._values = [[c[i] for c, i in zip(self.coords, self._indices[0])]]
        self._current_cursor = 0
        self._snap_to_data = False

    def add_cursor(self):
        self._bins.append([1] * self._obj.ndim)
        new_ind = self.indices
        self._indices.append(new_ind)
        self._values.append([c[i] for c, i in zip(self.coords, new_ind)])
        self.current_cursor = self.n_cursors - 1

    def remove_cursor(self, i):
        if self.n_cursors == 1:
            raise ValueError("There must be at least one cursor.")
        if self.current_cursor == i:
            if i == 0:
                self.current_cursor = 1
            self.current_cursor -= 1
        self._bins.pop(i)
        self._indices.pop(i)
        self._values.pop(i)

    @property
    def snap_to_data(self):
        return self._snap_to_data

    @snap_to_data.setter
    def snap_to_data(self, value: bool):
        self._snap_to_data = value

    @property
    def current_cursor(self):
        return self._current_cursor

    @current_cursor.setter
    def current_cursor(self, value: int):
        value = value % self.n_cursors
        if int(value) >= self.n_cursors:
            raise IndexError
        self._current_cursor = int(value)

    @property
    def n_cursors(self):
        return len(self._bins)

    @property
    def bins(self):
        return self._bins[self.current_cursor]

    @bins.setter
    def bins(self, value):
        if not len(value) == self._obj.ndim:
            raise ValueError("length of bin array must match the number of dimensions.")
        for i, x in enumerate(value):
            if x is None:
                continue
            if int(x) != x:
                raise TypeError("bins must have integer type")
            self._bins[self.current_cursor][i] = int(x)

    @property
    def binned(self):
        return tuple(b != 1 for b in self.bins)

    @property
    def indices(self):
        return self._indices[self.current_cursor]

    @indices.setter
    def indices(self, value):
        if not len(value) == self._obj.ndim:
            raise ValueError(
                "length of index array must match the number of dimensions"
            )
        for i, x in enumerate(value):
            if x is None:
                continue
            if int(x) != x:
                raise TypeError("indices must have integer type")
            self._indices[self.current_cursor][i] = int(x)
            self._values[self.current_cursor][i] = self.coords[i][int(x)]

    @property
    def values(self):
        return self._values[self.current_cursor]

    @values.setter
    def values(self, value):
        if not len(value) == self._obj.ndim:
            raise ValueError(
                "length of value array must match the number of dimensions"
            )
        for i, x in enumerate(value):
            if x is None:
                continue
            self._indices[self.current_cursor][i] = self.index_of_value(i, x)
            if self.snap_to_data:
                self._values[self.current_cursor][i] = self.coords[i][
                    self._indices[self.current_cursor][i]
                ]
            else:
                self._values[self.current_cursor][i] = float(x)
    
    @property
    def coords(self):
        return tuple(self._obj[dim].values for dim in self._obj.dims)

    @property
    def incs(self):
        return tuple(coord[1] - coord[0] for coord in self.coords)

    @property
    def lims(self):
        return tuple((coord[0], coord[-1]) for coord in self.coords)

    @property
    def data_vals_T(self):
        return return_transposed_jit(self._obj.values)
    
    def nanmax(self, *args, **kwargs):
        return numbagg.nanmax(self._obj.values, *args, **kwargs)

    def nanmin(self, *args, **kwargs):
        return numbagg.nanmin(self._obj.values, *args, **kwargs)
    
    def current_value(self):
        return self._obj.values[tuple(self.indices)]
    
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

    def array_rect(self, i=None, j=None):
        if i is None:
            i = 0
        if j is None:
            return self.coords[i]
        return array_rect_jit(i, j, self.lims, self.incs)

    def index_of_value(self, axis, val):
        return index_of_value_jit(axis, val, self.lims, self.incs, self._obj.shape)

    def slice_with_coord(self, axis: tuple):
        domain = sorted(set(range(self._obj.ndim)) - set(axis))
        return self.array_rect(*axis), self.extract_avg_slice(domain)

    def extract_avg_slice(self, axis=None):
        if axis is None:
            return self.data_vals_T
        if not np.iterable(axis):
            return self._bin_along_axis(axis)
        return self._bin_along_multiaxis(axis)

    def span_bounds(self, axis):
        slc = self._bin_slice(axis)
        lb = max(0, slc.start)
        ub = min(self._obj.shape[axis] - 1, slc.stop - 1)
        return self.coords[axis][[lb, ub]]

    def _bin_slice(self, axis):
        center = self.indices[axis]
        if self.binned[axis]:
            window = self.bins[axis]
            return slice(center - window // 2, center + (window - 1) // 2 + 1)
        else:
            return slice(center, center + 1)

    def _bin_along_axis(self, axis):
        axis -= 1
        if not self.binned[axis + 1]:
            return self.data_vals_T[
                (slice(None),) * (axis % self._obj.ndim) + (self._bin_slice(axis + 1),)
            ].squeeze(axis=axis)
        else:
            return numbagg.nanmean(
                self.data_vals_T[
                    (slice(None),) * (axis % self._obj.ndim)
                    + (self._bin_slice(axis + 1),)
                ],
                axis=axis,
            )

    def _bin_along_multiaxis(self, axis):
        if any(self.binned):
            slices = [self._bin_slice(ax) for ax in axis]
        else:
            slices = [self.indices[i] for i in axis]
        axis = [(ax - 1) % self._obj.ndim for ax in axis]
        selected = self.data_vals_T[
            tuple(
                slices[axis.index(d)] if d in axis else slice(None)
                for d in range(self._obj.ndim)
            )
        ]
        if any(self.binned):
            return numbagg.nanmean(selected, axis=axis)
        else:
            return selected
