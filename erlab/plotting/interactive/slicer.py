"""Helper functions for fast slicing xarray.DataArray objects."""
from collections.abc import Iterable, Sequence

import numba
import numbagg
import numpy as np
import numpy.typing as npt
import xarray as xr
from PySide6 import QtCore

VALID_NDIM = (2, 3, 4)

signature_array_rect = [
    numba.types.UniTuple(numba.float64, 4)(
        numba.int64,
        numba.int64,
        numba.types.UniTuple(numba.types.UniTuple(numba.float64, 2), i),
        numba.types.UniTuple(numba.float64, i),
    )
    for i in VALID_NDIM
]

signature_index_of_value = [
    numba.int64(
        numba.int64,
        numba.float64,
        numba.types.UniTuple(numba.types.UniTuple(numba.float64, 2), i),
        numba.types.UniTuple(numba.float64, i),
        numba.types.UniTuple(numba.int64, i),
    )
    for i in VALID_NDIM
]

signature_transposed = [
    numba.float64[[slice(None) for _ in range(i)]](
        numba.float64[[slice(None) for _ in range(i)]]
    )
    for i in VALID_NDIM
]


@numba.njit(
    signature_array_rect,
    fastmath=True,
    cache=True,
)
def _array_rect(
    i: int, j: int, lims: tuple[tuple[float, float], ...], incs: tuple[float, ...]
) -> tuple[float, float, float, float]:
    x = lims[i][0] - incs[i]
    y = lims[j][0] - incs[j]
    w = lims[i][-1] - x
    h = lims[j][-1] - y
    x += 0.5 * incs[i]
    y += 0.5 * incs[j]
    return x, y, w, h


@numba.njit(
    signature_index_of_value,
    fastmath=True,
    cache=True,
)
def _index_of_value(
    axis: int,
    val: float,
    lims: tuple[tuple[float, float], ...],
    incs: tuple[float, ...],
    shape: tuple[int],
) -> int:
    ind = min(
        round((val - lims[axis][0]) / incs[axis]),
        shape[axis] - 1,
    )
    if ind < 0:
        return 0
    return ind


@numba.njit(signature_transposed, fastmath=True, cache=True)
def _transposed(arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    if arr.ndim == 2:
        return arr.T
    elif arr.ndim == 3:
        return arr.transpose(1, 2, 0)
    else:
        return arr.transpose(1, 2, 3, 0)


@numba.njit(numba.boolean(numba.float64[::1]), fastmath=True, cache=True)
def _is_uniform(arr: npt.NDArray[np.float64]) -> bool:
    dif = np.diff(arr)
    return np.allclose(dif, dif[0], rtol=1e-05, atol=1e-08, equal_nan=False)


@numba.njit(numba.int64(numba.float64[:], numba.float64), fastmath=True, cache=True)
def _index_of_value_nonuniform(arr: npt.NDArray[np.float64], val: float) -> int:
    return np.searchsorted((arr[:-1] + arr[1:]) / 2, val)


class ArraySlicer(QtCore.QObject):
    sigIndexChanged = QtCore.Signal(int, tuple)
    sigBinChanged = QtCore.Signal(int, tuple)
    sigCursorCountChanged = QtCore.Signal(int)
    sigShapeChanged = QtCore.Signal()

    def __init__(self, xarray_obj: xr.DataArray):
        super().__init__()
        self.set_array(xarray_obj)
        self._bins = [[1] * self._obj.ndim]
        self._indices = [[s // 2 - (1 if s % 2 == 0 else 0) for s in self._obj.shape]]
        self._values = [[c[i] for c, i in zip(self.coords, self._indices[0])]]
        self.snap_to_data = False

    def set_array(self, xarray_obj: xr.DataArray, validate=True):
        if validate:
            self._obj = self._validate_array(xarray_obj)
        else:
            self._obj = xarray_obj
        self._nonuniform_axes = [
            i for i, d in enumerate(self._obj.dims) if str(d).endswith("_idx")
        ]

    @staticmethod
    def _validate_array(data: xr.DataArray):
        new_dims = ("kx", "ky")
        if all(d in data.dims for d in new_dims):
            # if data has kx and ky axis, transpose
            if "eV" in data.dims:
                new_dims += ("eV",)
            new_dims += tuple(d for d in data.dims if d not in new_dims)
            data = data.transpose(*new_dims)

        nonuniform_dims = [str(d) for d in data.dims if not _is_uniform(data[d].values)]
        for d in nonuniform_dims:
            data = data.assign_coords(
                {d + "_idx": (d, list(np.arange(len(data[d]), dtype=np.float64)))}
            ).swap_dims({d: d + "_idx"})

        return data.astype(np.float64, order="C")

    def add_cursor(self, like_cursor: int = -1, update: bool = True):
        self._bins.append(list(self.get_bins(like_cursor)))
        new_ind = self.get_indices(like_cursor)
        self._indices.append(list(new_ind))
        self._values.append([c[i] for c, i in zip(self.coords, new_ind)])
        if update:
            self.sigCursorCountChanged.emit(self.n_cursors)

    def remove_cursor(self, index: int, update: bool = True):
        if self.n_cursors == 1:
            raise ValueError("There must be at least one cursor.")
        self._bins.pop(index)
        self._indices.pop(index)
        self._values.pop(index)
        if update:
            self.sigCursorCountChanged.emit(self.n_cursors)

    @property
    def n_cursors(self) -> int:
        return len(self._bins)

    @QtCore.Slot(int, result=list[int])
    def get_bins(self, cursor: int) -> list[int]:
        return self._bins[cursor]

    def center_cursor(self, cursor: int, update: bool = True):
        self.set_indices(
            cursor,
            [s // 2 - (1 if s % 2 == 0 else 0) for s in self._obj.shape],
            update=update,
        )

    def set_bins(self, cursor: int, value: list[int | None], update: bool = True):
        if not len(value) == self._obj.ndim:
            raise ValueError("length of bin array must match the number of dimensions.")
        axes = []
        for i, x in enumerate(value):
            axes += self.set_bin(cursor, i, x, update=False)
        if update:
            self.sigBinChanged.emit(cursor, tuple(axes))

    @QtCore.Slot(int, int, int, bool, result=list[int | None])
    def set_bin(
        self, cursor: int, axis: int, value: int, update: bool = True
    ) -> list[int | None]:
        if value is None:
            return []
        if int(value) != value:
            raise TypeError("bins must have integer type")
        self._bins[cursor][axis] = int(value)
        if update:
            self.sigBinChanged.emit(cursor, (axis,))
            return []
        return [axis]

    @QtCore.Slot(int, result=tuple[bool, ...])
    def get_binned(self, cursor: int) -> tuple[bool, ...]:
        return tuple(b != 1 for b in self.get_bins(cursor))

    @QtCore.Slot(int, result=list[int])
    def get_indices(self, cursor: int) -> list[int]:
        return self._indices[cursor]

    @QtCore.Slot(int, int, result=int)
    def get_index(self, cursor: int, axis: int) -> int:
        return self._indices[cursor][axis]

    def set_indices(self, cursor: int, value: list[int], update: bool = True):
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
    def step_index(self, cursor: int, axis: int, amount: int, update: bool = True):
        self._indices[cursor][axis] += amount
        if (
            self._indices[cursor][axis] >= self._obj.shape[axis]
            or self._indices[cursor][axis] < 0
        ):
            self._indices[cursor][axis] -= amount
            return
        self._values[cursor][axis] = self.coords[axis][self._indices[cursor][axis]]
        if update:
            self.sigIndexChanged.emit(cursor, (axis,))

    @QtCore.Slot(int, bool, result=list[float])
    def get_values(self, cursor: int, uniform: bool = False) -> list[float]:
        if uniform and self._nonuniform_axes:
            val = list(self._values[cursor])
            for ax in self._nonuniform_axes:
                val[ax] = float(self._indices[cursor][ax])
            return val
        else:
            return self._values[cursor]

    @QtCore.Slot(int, int, bool, result=float)
    def get_value(self, cursor: int, axis: int, uniform: bool = False) -> float:
        if uniform and axis in self._nonuniform_axes:
            return float(self._indices[cursor][axis])
        else:
            return self._values[cursor][axis]

    def set_values(self, cursor: int, value: list[float], update: bool = True):
        if not len(value) == self._obj.ndim:
            raise ValueError(
                "length of value array must match the number of dimensions"
            )
        axes = []
        for i, x in enumerate(value):
            axes += self.set_value(cursor, i, x, update=False)
        if update:
            self.sigIndexChanged.emit(cursor, tuple(axes))

    @QtCore.Slot(int, int, float, bool, bool, result=list[int | None])
    def set_value(
        self,
        cursor: int,
        axis: int,
        value: float,
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
            self._values[cursor][axis] = float(value)
        if update:
            self.sigIndexChanged.emit(cursor, (axis,))
            return []
        return [axis]

    @property
    def coords(self) -> tuple[npt.NDArray[np.float64], ...]:
        if self._nonuniform_axes:
            return tuple(
                self._obj[str(dim)[:-4]].values.astype(np.float64)
                if i in self._nonuniform_axes
                else self._obj[dim].values.astype(np.float64)
                for i, dim in enumerate(self._obj.dims)
            )
        else:
            return self.coords_uniform

    @property
    def coords_uniform(self) -> tuple[npt.NDArray[np.float64], ...]:
        return tuple(self._obj[dim].values.astype(np.float64) for dim in self._obj.dims)

    @property
    def incs(self) -> tuple[float, ...]:
        return tuple(coord[1] - coord[0] for coord in self.coords)

    @property
    def incs_uniform(self) -> tuple[float, ...]:
        return tuple(coord[1] - coord[0] for coord in self.coords_uniform)

    @property
    def lims(self) -> tuple[tuple[float, float]]:
        return tuple((coord[0], coord[-1]) for coord in self.coords)

    @property
    def lims_uniform(self) -> tuple[tuple[float, float]]:
        return tuple((coord[0], coord[-1]) for coord in self.coords_uniform)

    @property
    def data_vals_T(self) -> npt.NDArray[np.float64]:
        return _transposed(self._obj.values)
    
    @property
    def limits(self) -> tuple[float, float]:
        return self.nanmin(), self.nanmax()

    def absnanmax(self, *args, **kwargs) -> float:
        return numbagg.nanmax(np.abs(self._obj.values), *args, **kwargs)

    def absnanmin(self, *args, **kwargs) -> float:
        return numbagg.nanmin(np.abs(self._obj.values), *args, **kwargs)

    def nanmax(self, *args, **kwargs) -> float:
        return numbagg.nanmax(self._obj.values, *args, **kwargs)

    def nanmin(self, *args, **kwargs) -> float:
        return numbagg.nanmin(self._obj.values, *args, **kwargs)
    
    @QtCore.Slot(int, bool, result=float)
    def point_value(
        self, cursor: int, binned: bool = True
    ) -> npt.NDArray[np.float64] | float:
        if binned:
            return self.extract_avg_slice(cursor, tuple(range(self._obj.ndim)))
        else:
            return self._obj.values[tuple(self.get_indices(cursor))]

    @QtCore.Slot(int, int)
    def swap_axes(self, ax1: int, ax2: int):
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
    ) -> tuple[float, float, float, float] | npt.NDArray[np.float64]:
        if i is None:
            i = 0
        if j is None:
            return self.coords_uniform[i]
        return _array_rect(i, j, self.lims_uniform, self.incs_uniform)

    def index_of_value(self, axis: int, val: float, uniform: bool = False) -> int:
        if uniform or (axis not in self._nonuniform_axes):
            return _index_of_value(
                axis, val, self.lims_uniform, self.incs_uniform, self._obj.shape
            )
        else:
            return _index_of_value_nonuniform(self.coords[axis], val)

    @QtCore.Slot(int, tuple, result=npt.NDArray[np.float64])
    def slice_with_coord(
        self, cursor: int, axis: Sequence[int]
    ) -> tuple[
        tuple[float, float, float, float] | npt.NDArray[np.float64],
        npt.NDArray[np.float64] | float,
    ]:
        domain = sorted(set(range(self._obj.ndim)) - set(axis))
        return self.array_rect(*axis), self.extract_avg_slice(cursor, domain)

    def extract_avg_slice(
        self, cursor: int, axis: int | Sequence[int] | None = None
    ) -> npt.NDArray[np.float64] | float:
        if axis is None:
            return self.data_vals_T
        if isinstance(axis, int):
            return self._bin_along_axis(cursor, axis)
        return self._bin_along_multiaxis(cursor, axis)

    def span_bounds(self, cursor: int, axis: int):
        slc = self._bin_slice(cursor, axis)
        lb = max(0, slc.start)
        ub = min(self._obj.shape[axis] - 1, slc.stop - 1)
        return self.coords_uniform[axis][[lb, ub]]

    def _bin_slice(self, cursor: int, axis: int) -> slice:
        center = self.get_indices(cursor)[axis]
        if self.get_binned(cursor)[axis]:
            window = self.get_bins(cursor)[axis]
            return slice(center - window // 2, center + (window - 1) // 2 + 1)
        else:
            return slice(center, center + 1)

    def _bin_along_axis(
        self, cursor: int, axis: int
    ) -> npt.NDArray[np.float64] | float:
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

    def _bin_along_multiaxis(
        self, cursor: int, axis: Sequence[int]
    ) -> npt.NDArray[np.float64] | float:
        if any(self.get_binned(cursor)):
            slices = [self._bin_slice(cursor, ax) for ax in axis]
        else:
            slices = [self.get_indices(cursor)[i] for i in axis]
        axis = tuple((ax - 1) % self._obj.ndim for ax in axis)
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
