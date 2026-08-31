import concurrent.futures

import dask.callbacks
import numpy as np
import pytest
import xarray as xr
import xarray.testing

import erlab.analysis.transform
from erlab.analysis.transform import rotate, shift, symmetrize, symmetrize_nfold


def _symmetrize_nfold_reference(
    darr: xr.DataArray,
    fold: int,
    *,
    order: int,
    monkeypatch: pytest.MonkeyPatch,
    center: tuple[float, float] = (0.0, 0.0),
    mode: str = "constant",
) -> xr.DataArray:
    if not (
        np.issubdtype(darr.dtype, np.floating)
        or np.issubdtype(darr.dtype, np.complexfloating)
    ):
        darr = darr.astype(np.result_type(darr.dtype, float))
    with monkeypatch.context() as scipy_path:
        scipy_path.setattr(
            erlab.analysis.transform, "_NUMBA_AFFINE_DTYPES", frozenset()
        )
        return xr.concat(
            [
                rotate(
                    darr,
                    360.0 * idx / fold,
                    axes=("y", "x"),
                    center=center,
                    reshape=False,
                    order=order,
                    mode=mode,
                    cval=np.nan,
                    prefilter=order > 1,
                )
                for idx in range(fold)
            ],
            dim="rotation",
        ).mean("rotation", skipna=True)


@pytest.mark.parametrize("use_dask", [False, True], ids=["no_dask", "dask"])
def test_rotate(use_dask) -> None:
    input_arr = xr.DataArray(
        np.arange(12).reshape((3, 4)).astype(float),
        dims=("y", "x"),
        coords={"y": [0.0, 1.0, 2.0], "x": [0.0, 1.0, 2.0, 3.0]},
    )
    if use_dask:
        input_arr = input_arr.chunk()
    expected_output = xr.DataArray(
        np.array([[3, 7, 11], [2, 6, 10], [1, 5, 9], [0, 4, 8]], dtype=float),
        dims=("y", "x"),
        coords={"y": [-3.0, -2.0, -1.0, 0.0], "x": [0.0, 1.0, 2.0]},
    )

    xarray.testing.assert_allclose(
        rotate(input_arr, 90, reshape=True, order=1), expected_output
    )
    xarray.testing.assert_allclose(
        rotate(
            input_arr,
            90,
            axes=("y", "x"),
            center={"x": 0, "y": 0},
            reshape=True,
            order=1,
        ),
        expected_output,
    )
    xarray.testing.assert_allclose(
        rotate(input_arr, 90, center={"x": 3, "y": 2}, reshape=True, order=1),
        xr.DataArray(
            np.array([[3, 7, 11], [2, 6, 10], [1, 5, 9], [0, 4, 8]], dtype=float),
            dims=("y", "x"),
            coords={"y": [2.0, 3.0, 4.0, 5.0], "x": [1.0, 2.0, 3.0]},
        ),
    )

    xarray.testing.assert_allclose(
        rotate(input_arr, 90, reshape=False, order=1),
        xr.DataArray(
            np.array(
                [
                    [0, 4, 8, np.nan],
                    [np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan],
                ],
                dtype=float,
            ),
            dims=("y", "x"),
            coords={"y": [0.0, 1.0, 2.0], "x": [0.0, 1.0, 2.0, 3.0]},
        ),
    )

    # Higher dimensional array
    input_arr = xr.DataArray(
        np.arange(24).reshape((3, 4, 2)).astype(float),
        dims=("y", "x", "z"),
        coords={"y": [0.0, 1.0, 2.0], "x": [0.0, 1.0, 2.0, 3.0], "z": [0.0, 1.0]},
    )
    if use_dask:
        input_arr = input_arr.chunk()
    xarray.testing.assert_allclose(
        rotate(input_arr, 90, reshape=True, order=1),
        xr.DataArray(
            np.array(
                [
                    [[6, 7], [14, 15], [22, 23]],
                    [[4, 5], [12, 13], [20, 21]],
                    [[2, 3], [10, 11], [18, 19]],
                    [[0, 1], [8, 9], [16, 17]],
                ],
                dtype=float,
            ),
            dims=("y", "x", "z"),
            coords={
                "y": [-3.0, -2.0, -1.0, 0.0],
                "x": [0.0, 1.0, 2.0],
                "z": [0.0, 1.0],
            },
        ),
    )

    # Test with associated coordinates
    input_arr = xr.DataArray(
        np.arange(12).reshape((3, 4)).astype(float),
        dims=("y", "x"),
        coords={
            "y": [0.0, 1.0, 2.0],
            "x": [0.0, 1.0, 2.0, 3.0],
            "yy": ("y", [0.0, 1.0, 2.0]),
        },
    )
    if use_dask:
        input_arr = input_arr.chunk()
    expected_output = xr.DataArray(
        np.array([[3, 7, 11], [2, 6, 10], [1, 5, 9], [0, 4, 8]], dtype=float),
        dims=("y", "x"),
        coords={"y": [-3.0, -2.0, -1.0, 0.0], "x": [0.0, 1.0, 2.0]},
    )

    with pytest.raises(
        ValueError, match="center must have keys matching the two rotation axes"
    ):
        rotate(input_arr, 90, center={"x": 0, "z": 0})

    with pytest.raises(
        ValueError, match="all coordinates along axes must be evenly spaced"
    ):
        rotate(
            xr.DataArray(
                np.arange(12).reshape((3, 4)).astype(float),
                dims=("y", "x"),
                coords={"y": [0.0, 1.0, 3.0], "x": [0.0, 1.0, 2.0, 3.0]},
            ),
            90,
        )


@pytest.mark.parametrize(
    ("dtype", "array_order", "read_only"),
    [
        pytest.param(np.float32, "C", False, id="float32-c-writeable"),
        pytest.param(np.float64, "F", True, id="float64-f-read-only"),
        pytest.param(np.complex64, "C", True, id="complex64-c-read-only"),
        pytest.param(np.complex128, "F", False, id="complex128-f-writeable"),
    ],
)
def test_rotate_linear_numba_dtype(dtype, array_order, read_only, monkeypatch) -> None:
    values = np.array(np.arange(63).reshape(7, 9), dtype=dtype, order=array_order)
    if np.issubdtype(dtype, np.complexfloating):
        values *= 1.0 + 0.5j
    values[2, 5] = np.nan
    values.setflags(write=not read_only)
    original = values.copy()
    darr = xr.DataArray(
        values,
        dims=("y", "x"),
        coords={"y": np.linspace(-1.2, 1.2, 7), "x": np.linspace(-2.0, 2.0, 9)},
    )
    kwargs = {
        "angle": 33.0,
        "axes": ("y", "x"),
        "center": (0.15, -0.2),
        "reshape": False,
        "order": 1,
        "mode": "constant",
        "cval": 0.25,
        "prefilter": False,
    }

    with monkeypatch.context() as scipy_path:
        scipy_path.setattr(
            erlab.analysis.transform, "_NUMBA_AFFINE_DTYPES", frozenset()
        )
        expected = rotate(darr, **kwargs)
    actual = rotate(darr, **kwargs)

    assert actual.dtype == np.dtype(dtype)
    xr.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(values, original)
    assert values.flags.writeable == (not read_only)


@pytest.mark.parametrize(
    "missing_value",
    [
        pytest.param(complex(np.nan, 7.0), id="real-nan"),
        pytest.param(complex(7.0, np.nan), id="imaginary-nan"),
    ],
)
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_rotate_linear_numba_complex_nan_components(
    dtype, missing_value, monkeypatch
) -> None:
    values = np.arange(15, dtype=dtype).reshape(3, 5) + 1j * np.arange(
        100, 115, dtype=dtype
    ).reshape(3, 5)
    values[1, 2] = missing_value
    darr = xr.DataArray(
        values,
        dims=("y", "x"),
        coords={"y": np.arange(-1.0, 2.0), "x": np.arange(-2.0, 3.0)},
    )
    kwargs = {
        "angle": 0.0,
        "axes": ("y", "x"),
        "reshape": False,
        "order": 1,
        "mode": "constant",
        "cval": np.nan,
        "prefilter": False,
    }

    with monkeypatch.context() as scipy_path:
        scipy_path.setattr(
            erlab.analysis.transform, "_NUMBA_AFFINE_DTYPES", frozenset()
        )
        expected = rotate(darr, **kwargs)
    actual = rotate(darr, **kwargs)

    xr.testing.assert_allclose(actual.real, expected.real)
    xr.testing.assert_allclose(actual.imag, expected.imag)


@pytest.mark.parametrize(
    ("order", "mode", "dtype"),
    [
        pytest.param(1, "constant", np.int16, id="integer"),
        pytest.param(0, "constant", np.float64, id="nearest-order"),
        pytest.param(3, "constant", np.float64, id="spline-order"),
        pytest.param(1, "nearest", np.float64, id="boundary-mode"),
    ],
)
def test_rotate_scipy_fallback(order, mode, dtype, monkeypatch) -> None:
    def _fail_fast_path(*args, **kwargs) -> None:
        raise AssertionError("Numba fast path was used")

    monkeypatch.setattr(
        erlab.analysis.transform, "_apply_affine_linear", _fail_fast_path
    )
    darr = xr.DataArray(
        np.arange(63).reshape(7, 9).astype(dtype),
        dims=("y", "x"),
        coords={"y": np.linspace(-1.2, 1.2, 7), "x": np.linspace(-2.0, 2.0, 9)},
    )

    actual = rotate(
        darr,
        33.0,
        axes=("y", "x"),
        center=(0.15, -0.2),
        reshape=False,
        order=order,
        mode=mode,
        cval=0.0,
        prefilter=order > 1,
    )

    assert actual.shape == darr.shape
    assert actual.dtype == np.dtype(dtype)


@pytest.mark.parametrize(
    ("shape", "angle", "reshape", "nan_index"),
    [
        pytest.param((3, 5), 0.0, False, (2, 1), id="identity-stencil"),
        pytest.param((7, 9), 90.0, True, (4, 1), id="right-angle-reshape"),
    ],
)
def test_rotate_linear_numba_scipy_boundary_rounding(
    shape, angle, reshape, nan_index, monkeypatch
) -> None:
    values = np.arange(np.prod(shape), dtype=float).reshape(shape)
    values[nan_index] = np.nan
    darr = xr.DataArray(
        values,
        dims=("y", "x"),
        coords={
            "y": (np.arange(shape[0]) - (shape[0] - 1) / 2) * 0.4,
            "x": (np.arange(shape[1]) - (shape[1] - 1) / 2) * 0.4,
        },
    )
    kwargs = {
        "angle": angle,
        "axes": ("y", "x"),
        "center": (0.0, 0.0),
        "reshape": reshape,
        "order": 1,
        "mode": "constant",
        "cval": np.nan,
        "prefilter": False,
    }

    with monkeypatch.context() as scipy_path:
        scipy_path.setattr(
            erlab.analysis.transform, "_NUMBA_AFFINE_DTYPES", frozenset()
        )
        expected = rotate(darr, **kwargs)
    actual = rotate(darr, **kwargs)

    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    xr.testing.assert_allclose(actual, expected)


def test_rotate_linear_numba_dask_matches_eager(monkeypatch) -> None:
    values = np.arange(126, dtype=float).reshape(2, 7, 9)
    values[0, 2, 5] = np.nan
    darr = xr.DataArray(
        values,
        dims=("eV", "y", "x"),
        coords={
            "eV": [-0.1, 0.0],
            "y": np.linspace(-1.2, 1.2, 7),
            "x": np.linspace(-2.0, 2.0, 9),
        },
    )
    kwargs = {
        "angle": 33.0,
        "axes": ("y", "x"),
        "center": (0.15, -0.2),
        "reshape": False,
        "order": 1,
        "mode": "constant",
        "cval": np.nan,
        "prefilter": False,
    }
    expected = rotate(darr, **kwargs)

    def _fail_parallel_kernel(*args, **kwargs) -> None:
        raise AssertionError("Parallel Numba kernel was used inside a Dask task")

    monkeypatch.setattr(
        erlab.analysis.transform, "_apply_affine_linear_numba", _fail_parallel_kernel
    )
    construction_tasks = []

    with dask.callbacks.Callback(
        pretask=lambda key, *args: construction_tasks.append(key)
    ):
        actual = rotate(darr.chunk({"eV": 1, "y": -1, "x": -1}), **kwargs)

    assert construction_tasks == []
    assert actual.chunks is not None
    compute_tasks = []
    with dask.callbacks.Callback(pretask=lambda key, *args: compute_tasks.append(key)):
        computed = actual.compute(scheduler="threads")
    kernel_tasks = [
        key
        for key in compute_tasks
        if "apply_affine_linear" in str(key[0] if isinstance(key, tuple) else key)
    ]
    assert len(kernel_tasks) == 2
    xr.testing.assert_allclose(computed, expected)


def test_affine_linear_numba_worker_threads_use_serial(monkeypatch) -> None:
    darr = xr.DataArray(
        np.arange(63, dtype=float).reshape(7, 9),
        dims=("y", "x"),
        coords={"y": np.linspace(-1.2, 1.2, 7), "x": np.linspace(-2.0, 2.0, 9)},
    )
    rotate_kwargs = {
        "angle": 33.0,
        "axes": ("y", "x"),
        "center": (0.15, -0.2),
        "reshape": False,
    }
    symmetrize_kwargs = {
        "fold": 3,
        "axes": ("y", "x"),
        "center": (0.15, -0.2),
        "reshape": False,
    }
    expected_rotate = rotate(darr, **rotate_kwargs)
    expected_symmetrized = symmetrize_nfold(darr, **symmetrize_kwargs)

    def _fail_parallel_kernel(*args, **kwargs) -> None:
        raise AssertionError("Parallel Numba kernel was used inside a worker thread")

    monkeypatch.setattr(
        erlab.analysis.transform, "_apply_affine_linear_numba", _fail_parallel_kernel
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        rotate_future = executor.submit(rotate, darr, **rotate_kwargs)
        symmetrize_future = executor.submit(symmetrize_nfold, darr, **symmetrize_kwargs)
        actual_rotate = rotate_future.result()
        actual_symmetrized = symmetrize_future.result()

    xr.testing.assert_allclose(actual_rotate, expected_rotate)
    xr.testing.assert_allclose(actual_symmetrized, expected_symmetrized)


def test_rotate_reshape_dask_is_lazy_and_geometry_based() -> None:
    values = np.full((2, 3, 5), np.nan)
    darr = xr.DataArray(
        values,
        dims=("eV", "y", "x"),
        coords={
            "eV": [-0.1, 0.0],
            "y": np.arange(-1.0, 2.0),
            "x": np.arange(-2.0, 3.0),
        },
    )
    lazy = darr.chunk({"eV": 1, "y": -1, "x": -1})
    construction_tasks = []

    with dask.callbacks.Callback(
        pretask=lambda key, *args: construction_tasks.append(key)
    ):
        actual = rotate(
            lazy,
            30.0,
            axes=("y", "x"),
            center=(0.0, 0.0),
            reshape=True,
            order=1,
            mode="constant",
            cval=np.nan,
            prefilter=False,
        )

    assert construction_tasks == []
    assert actual.chunks is not None
    assert actual.sizes == {"eV": 2, "y": 5, "x": 6}
    np.testing.assert_allclose(actual.y, np.arange(-2.0, 3.0), atol=1e-14)
    np.testing.assert_allclose(actual.x, np.arange(-2.5, 3.0), atol=1e-14)

    expected = rotate(
        darr,
        30.0,
        axes=("y", "x"),
        center=(0.0, 0.0),
        reshape=True,
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )
    computed = actual.compute(scheduler="threads")
    xr.testing.assert_allclose(computed, expected)


@pytest.mark.parametrize("use_dask", [False, True], ids=["no_dask", "dask"])
def test_shift(use_dask) -> None:
    # Create a test input DataArray
    darr = xr.DataArray(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(float), dims=["x", "y"]
    )
    if use_dask:
        darr = darr.chunk()

    # Create a test shift DataArray
    shift_arr = xr.DataArray([1, 0, 2], dims=["x"])

    # Perform the shift operation
    shifted = shift(darr, shift_arr, along="y")

    # Define the expected result
    expected = xr.DataArray(
        np.array([[np.nan, 1.0, 2.0], [4.0, 5.0, 6.0], [np.nan, np.nan, 7.0]]),
        dims=["x", "y"],
    )

    # Check if the shifted array matches the expected result
    assert np.allclose(shifted, expected, equal_nan=True)


def test_shift_coords_ignores_nan_shifts_for_rigid_coordinate_shift() -> None:
    data = xr.DataArray(
        np.arange(6, dtype=float).reshape(2, 3),
        dims=("x", "y"),
        coords={"y": [10.0, 11.0, 12.0]},
    )
    shifts = xr.DataArray([2.0, np.nan], dims="x")

    shifted = shift(data, shifts, along="y", shift_coords=True)

    np.testing.assert_allclose(shifted.y, [12.0, 13.0, 14.0])


def test_shift_order1_optimized() -> None:
    arr = xr.DataArray(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        dims=["x", "y"],
        coords={"x": [0, 1], "y": [0, 1, 2]},
    )

    shifted = shift(
        arr,
        shift=1.0,
        along="y",
        shift_coords=False,
        order=1,
        mode="constant",
        prefilter=False,
    )

    expected = np.array([[np.nan, 1.0, 2.0], [np.nan, 4.0, 5.0]])
    np.testing.assert_allclose(shifted.values, expected, equal_nan=True)


@pytest.mark.parametrize("use_dask", [False, True], ids=["no_dask", "dask"])
def test_symmetrize_nfold(use_dask) -> None:
    coords = np.arange(-2.0, 3.0, dtype=float)
    darr = xr.DataArray(
        np.zeros((5, 5), dtype=int),
        dims=("ky", "kx"),
        coords={"ky": coords, "kx": coords},
    )
    darr.loc[{"ky": 0.0, "kx": 1.0}] = 1

    if use_dask:
        darr = darr.chunk()

    expected = xr.DataArray(
        np.zeros((5, 5), dtype=float),
        dims=("ky", "kx"),
        coords={"ky": coords, "kx": coords},
    )
    for ky, kx in ((0.0, 1.0), (1.0, 0.0), (0.0, -1.0), (-1.0, 0.0)):
        expected.loc[{"ky": ky, "kx": kx}] = 0.25

    sym = symmetrize_nfold(
        darr,
        4,
        axes=("ky", "kx"),
        center={"ky": 0.0, "kx": 0.0},
        reshape=False,
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )

    assert np.issubdtype(sym.dtype, np.floating)
    xr.testing.assert_allclose(sym, expected)


@pytest.mark.parametrize(
    "dtype", [np.int16, np.float32, np.float64, np.complex64, np.complex128]
)
@pytest.mark.parametrize("array_order", ["C", "F"], ids=["c_order", "f_order"])
@pytest.mark.parametrize("read_only", [False, True], ids=["writeable", "read_only"])
def test_symmetrize_nfold_linear_numba_dtype(
    dtype, array_order, read_only, monkeypatch
) -> None:
    values = np.array(np.arange(63).reshape(7, 9), dtype=dtype, order=array_order)
    if np.issubdtype(dtype, np.complexfloating):
        values *= 1.0 + 0.5j
    values[2, 5] = np.nan if not np.issubdtype(dtype, np.integer) else 0
    values.setflags(write=not read_only)
    original = values.copy()
    darr = xr.DataArray(
        values,
        dims=("y", "x"),
        coords={"y": np.linspace(-1.2, 1.2, 7), "x": np.linspace(-2.0, 2.0, 9)},
    )
    center = (0.15, -0.2)

    expected = _symmetrize_nfold_reference(
        darr, 3, order=1, monkeypatch=monkeypatch, center=center
    )
    actual = symmetrize_nfold(
        darr,
        3,
        axes=("y", "x"),
        center=center,
        reshape=False,
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )

    expected_dtype = (
        np.dtype(np.float64) if np.issubdtype(dtype, np.integer) else np.dtype(dtype)
    )
    assert actual.dtype == expected_dtype
    xr.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(values, original)
    assert values.flags.writeable == (not read_only)


@pytest.mark.parametrize(
    ("order", "mode"),
    [(0, "constant"), (1, "constant"), (3, "constant"), (1, "nearest")],
)
def test_symmetrize_nfold_interpolation_orders(order, mode, monkeypatch) -> None:
    darr = xr.DataArray(
        np.arange(63, dtype=float).reshape(7, 9),
        dims=("y", "x"),
        coords={"y": np.linspace(-1.2, 1.2, 7), "x": np.linspace(-2.0, 2.0, 9)},
    )
    center = (0.15, -0.2)

    expected = _symmetrize_nfold_reference(
        darr,
        3,
        order=order,
        monkeypatch=monkeypatch,
        center=center,
        mode=mode,
    )
    actual = symmetrize_nfold(
        darr,
        3,
        axes=("y", "x"),
        center=center,
        reshape=False,
        order=order,
        mode=mode,
        cval=np.nan,
        prefilter=order > 1,
    )

    xr.testing.assert_allclose(actual, expected)


def test_symmetrize_nfold_linear_numba_nan_stencil(monkeypatch) -> None:
    values = np.arange(15, dtype=float).reshape(3, 5)
    values[1, 2] = np.nan
    darr = xr.DataArray(
        values,
        dims=("y", "x"),
        coords={"y": np.arange(-1.0, 2.0), "x": np.arange(-2.0, 3.0)},
    )

    expected = _symmetrize_nfold_reference(darr, 2, order=1, monkeypatch=monkeypatch)
    actual = symmetrize_nfold(
        darr,
        2,
        axes=("y", "x"),
        center=(0.0, 0.0),
        reshape=False,
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )

    xr.testing.assert_allclose(actual, expected)


def test_symmetrize_nfold_reshape_dask_is_lazy_and_geometry_based() -> None:
    values = np.full((2, 3, 5), np.nan)
    darr = xr.DataArray(
        values,
        dims=("eV", "y", "x"),
        coords={
            "eV": [-0.1, 0.0],
            "y": np.arange(-1.0, 2.0),
            "x": np.arange(-2.0, 3.0),
        },
    )
    lazy = darr.chunk({"eV": 1, "y": -1, "x": -1})
    executed_tasks = []

    with dask.callbacks.Callback(pretask=lambda key, *args: executed_tasks.append(key)):
        actual = symmetrize_nfold(
            lazy,
            4,
            axes=("y", "x"),
            center=(0.0, 0.0),
            reshape=True,
            order=1,
            mode="constant",
            cval=np.nan,
            prefilter=False,
        )

    assert executed_tasks == []
    assert actual.chunks is not None
    assert actual.sizes == {"eV": 2, "y": 5, "x": 5}
    expected = symmetrize_nfold(
        darr,
        4,
        axes=("y", "x"),
        center=(0.0, 0.0),
        reshape=True,
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )
    compute_tasks = []
    with dask.callbacks.Callback(pretask=lambda key, *args: compute_tasks.append(key)):
        computed = actual.compute(scheduler="threads")
    kernel_tasks = [
        key
        for key in compute_tasks
        if "apply_affine_linear" in str(key[0] if isinstance(key, tuple) else key)
    ]
    assert len(kernel_tasks) == 2
    xr.testing.assert_allclose(computed, expected)


def test_symmetrize_nfold_reshape_numba_matches_scipy(monkeypatch) -> None:
    darr = xr.DataArray(
        np.arange(15, dtype=float).reshape(3, 5),
        dims=("y", "x"),
        coords={
            "y": (np.arange(3) - 1) * 0.4,
            "x": (np.arange(5) - 2) * 0.4,
        },
    )
    kwargs = {
        "fold": 3,
        "axes": ("y", "x"),
        "center": (0.0, 0.0),
        "reshape": True,
        "order": 1,
        "mode": "constant",
        "cval": np.nan,
        "prefilter": False,
    }

    with monkeypatch.context() as scipy_path:
        scipy_path.setattr(
            erlab.analysis.transform, "_NUMBA_AFFINE_DTYPES", frozenset()
        )
        expected = symmetrize_nfold(darr, **kwargs)
    actual = symmetrize_nfold(darr, **kwargs)

    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    xr.testing.assert_allclose(actual, expected)


def test_symmetrize_nfold_reshape_geometry_matches_scipy_support() -> None:
    darr = xr.DataArray(
        np.ones((11, 11)),
        dims=("y", "x"),
        coords={
            "y": (np.arange(11) - 5) * -0.4,
            "x": (np.arange(11) - 5) * 0.7,
        },
    )

    actual = symmetrize_nfold(darr, 3, axes=("y", "x"), reshape=True)

    assert actual.sizes == {"y": 21, "x": 11}
    np.testing.assert_allclose(actual.y, np.linspace(4.0, -4.0, 21), atol=1e-14)
    np.testing.assert_allclose(actual.x, np.linspace(-3.5, 3.5, 11), atol=1e-14)


def test_rotation_geometry_bounds_no_intersection() -> None:
    matrices = np.array([[[1.0, 0.0, 100.0], [0.0, 1.0, 100.0], [0.0, 0.0, 1.0]]])

    bounds = erlab.analysis.transform._rotation_geometry_bounds(
        matrices, (3, 3), (4, 5)
    )

    assert bounds == (0, 0, 0, 0)


@pytest.mark.parametrize(
    ("mode", "cval"),
    [pytest.param("constant", 0.25, id="finite-cval"), pytest.param("nearest", np.nan)],
)
def test_symmetrize_nfold_reshape_keeps_finite_padding(mode, cval) -> None:
    darr = xr.DataArray(
        np.ones((3, 5)),
        dims=("y", "x"),
        coords={"y": np.arange(-1.0, 2.0), "x": np.arange(-2.0, 3.0)},
    )

    actual = symmetrize_nfold(
        darr, 3, axes=("y", "x"), reshape=True, mode=mode, cval=cval
    )

    assert actual.sizes == {"y": 7, "x": 7}


def test_symmetrize_nfold_broadcasts_over_remaining_dims() -> None:
    coords = np.arange(-2.0, 3.0, dtype=float)
    darr = xr.DataArray(
        np.zeros((2, 5, 5), dtype=float),
        dims=("eV", "ky", "kx"),
        coords={"eV": [-0.1, 0.0], "ky": coords, "kx": coords},
    )
    darr.loc[{"eV": -0.1, "ky": 0.0, "kx": 1.0}] = 1.0
    darr.loc[{"eV": 0.0, "ky": 0.0, "kx": 2.0}] = 2.0

    expected = xr.DataArray(
        np.zeros((2, 5, 5), dtype=float),
        dims=("eV", "ky", "kx"),
        coords={"eV": [-0.1, 0.0], "ky": coords, "kx": coords},
    )
    for ky, kx in ((0.0, 1.0), (1.0, 0.0), (0.0, -1.0), (-1.0, 0.0)):
        expected.loc[{"eV": -0.1, "ky": ky, "kx": kx}] = 0.25
    for ky, kx in ((0.0, 2.0), (2.0, 0.0), (0.0, -2.0), (-2.0, 0.0)):
        expected.loc[{"eV": 0.0, "ky": ky, "kx": kx}] = 0.5

    sym = symmetrize_nfold(
        darr,
        4,
        axes=("ky", "kx"),
        center={"ky": 0.0, "kx": 0.0},
        reshape=False,
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )

    xr.testing.assert_allclose(sym, expected)


def test_symmetrize_nfold_prefilter_matches_rotate_default() -> None:
    coords = np.arange(-4.0, 5.0, dtype=float)
    darr = xr.DataArray(
        np.zeros((9, 9), dtype=float),
        dims=("y", "x"),
        coords={"y": coords, "x": coords},
    )
    darr.loc[{"y": 0.0, "x": 1.0}] = 1.0

    expected = xr.concat(
        [
            rotate(
                darr,
                90.0 * idx,
                axes=("y", "x"),
                center={"y": 0.0, "x": 0.0},
                reshape=False,
                order=3,
                mode="constant",
                cval=np.nan,
            )
            for idx in range(4)
        ],
        dim="_preview_symmetry",
    ).mean("_preview_symmetry", skipna=True, keep_attrs=True)

    sym = symmetrize_nfold(
        darr,
        4,
        axes=("y", "x"),
        center={"y": 0.0, "x": 0.0},
        reshape=False,
        order=3,
        mode="constant",
        cval=np.nan,
    )

    xr.testing.assert_allclose(sym, expected)
    assert sym.sel(y=0.0, x=1.0).item() == pytest.approx(0.25)


def test_symmetrize_nfold_invalid_fold() -> None:
    darr = xr.DataArray(
        np.zeros((5, 5), dtype=float),
        dims=("y", "x"),
        coords={"y": np.arange(-2.0, 3.0), "x": np.arange(-2.0, 3.0)},
    )

    with pytest.raises(ValueError, match="fold must be at least 2"):
        symmetrize_nfold(darr, 1)


def test_symmetrize_nfold_non_uniform() -> None:
    darr = xr.DataArray(
        np.zeros((4, 4), dtype=float),
        dims=("y", "x"),
        coords={"y": [0.0, 1.0, 3.0, 6.0], "x": [0.0, 1.0, 2.0, 3.0]},
    )

    with pytest.raises(
        ValueError, match="all coordinates along axes must be evenly spaced"
    ):
        symmetrize_nfold(darr, 4)


def test_symmetrize_nfold_invalid_center() -> None:
    darr = xr.DataArray(
        np.zeros((5, 5), dtype=float),
        dims=("y", "x"),
        coords={"y": np.arange(-2.0, 3.0), "x": np.arange(-2.0, 3.0)},
    )

    with pytest.raises(
        ValueError, match="center must have keys matching the two rotation axes"
    ):
        symmetrize_nfold(darr, 4, center={"x": 0.0, "z": 0.0})


def test_symmetrize_nfold_preserves_attrs_and_drops_rotated_axis_coords() -> None:
    darr = xr.DataArray(
        np.zeros((5, 5), dtype=float),
        dims=("y", "x"),
        coords={
            "y": np.arange(5.0),
            "x": np.arange(5.0),
            "yy": ("y", np.arange(5.0)),
            "label": "sample",
        },
        attrs={"test_attr": 1},
    )
    darr.loc[{"y": 2.0, "x": 3.0}] = 1.0

    sym = symmetrize_nfold(darr, 4, center={"y": 2.0, "x": 2.0})

    assert "yy" not in sym.coords
    assert sym.coords["label"].item() == "sample"
    assert sym.attrs == {"test_attr": 1}


@pytest.mark.parametrize("use_dask", [False, True], ids=["no_dask", "dask"])
def test_symmetrize_nfold_defaults_to_reshape(use_dask) -> None:
    darr = xr.DataArray(
        np.zeros((3, 5), dtype=float),
        dims=("y", "x"),
        coords={"y": np.arange(-1.0, 2.0), "x": np.arange(-2.0, 3.0)},
    )
    darr.loc[{"y": 0.0, "x": 2.0}] = 1.0

    if use_dask:
        darr = darr.chunk()

    sym = symmetrize_nfold(
        darr,
        4,
        axes=("y", "x"),
        center={"y": 0.0, "x": 0.0},
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )

    expected = xr.DataArray(
        np.array(
            [
                [np.nan, 0.0, 0.5, 0.0, np.nan],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0, 0.0, 0.5],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [np.nan, 0.0, 0.5, 0.0, np.nan],
            ],
            dtype=float,
        ),
        dims=("y", "x"),
        coords={"y": np.arange(-2.0, 3.0), "x": np.arange(-2.0, 3.0)},
    )

    assert sym.sizes["y"] > darr.sizes["y"]
    xr.testing.assert_allclose(sym, expected)


@pytest.mark.parametrize(
    ("mode", "part", "expected"),
    [
        ("valid", "both", [3.0, 5.0, 7.0, 9.0, 10.0, 10.0, 9.0, 7.0, 5.0, 3.0]),
        ("valid", "below", [3.0, 5.0, 7.0, 9.0, 10.0]),
        ("valid", "above", [10.0, 9.0, 7.0, 5.0, 3.0]),
        (
            "full",
            "both",
            [1.5, 3.0, 5.0, 7.0, 9.0, 10.0, 10.0, 9.0, 7.0, 5.0, 3.0, 1.5],
        ),
        ("full", "below", [1.5, 3.0, 5.0, 7.0, 9.0, 10.0]),
        ("full", "above", [10.0, 9.0, 7.0, 5.0, 3.0, 1.5]),
    ],
)
def test_symmetrize(mode, part, expected):
    da = xr.DataArray(
        np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0], dtype=float),
        dims="x",
        coords={"x": np.linspace(-6, 5, 12)},
    )
    sym_da = symmetrize(da, "x", center=0.0, mode=mode, part=part)
    expected = np.array(expected, dtype=float)
    np.testing.assert_allclose(sym_da.values, expected, rtol=1e-5)


@pytest.mark.parametrize(
    ("mode", "part", "expected"),
    [
        ("valid", "both", [3.0, 5.0, 7.0, 9.0, 10.0, 10.0, 9.0, 7.0, 5.0, 3.0]),
        ("valid", "below", [10.0, 9.0, 7.0, 5.0, 3.0]),
        ("valid", "above", [3.0, 5.0, 7.0, 9.0, 10.0]),
        (
            "full",
            "both",
            [1.5, 3.0, 5.0, 7.0, 9.0, 10.0, 10.0, 9.0, 7.0, 5.0, 3.0, 1.5],
        ),
        ("full", "below", [10.0, 9.0, 7.0, 5.0, 3.0, 1.5]),
        ("full", "above", [1.5, 3.0, 5.0, 7.0, 9.0, 10.0]),
    ],
)
def test_symmetrize_inverted(mode, part, expected):
    da = xr.DataArray(
        np.array([0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1], dtype=float),
        dims="x",
        coords={"x": np.linspace(5, -6, 12)},
    )
    sym_da = symmetrize(da, "x", center=0.0, mode=mode, part=part)
    expected = np.array(expected, dtype=float)
    np.testing.assert_allclose(sym_da.values, expected, rtol=1e-5)


@pytest.mark.parametrize(
    ("mode", "part", "expected"),
    [
        (
            "valid",
            "both",
            [np.nan, np.nan, 7.0, 9.0, 10.0, 10.0, 9.0, 7.0, np.nan, np.nan],
        ),
        ("valid", "below", [np.nan, np.nan, 7.0, 9.0, 10.0]),
        ("valid", "above", [10.0, 9.0, 7.0, np.nan, np.nan]),
        (
            "full",
            "both",
            [1.5, 2.5, 3.5, 7.0, 9.0, 10.0, 10.0, 9.0, 7.0, 3.5, 2.5, 1.5],
        ),
        ("full", "below", [1.5, 2.5, 3.5, 7.0, 9.0, 10.0]),
        ("full", "above", [10.0, 9.0, 7.0, 3.5, 2.5, 1.5]),
    ],
)
def test_symmetrize_na(mode, part, expected):
    da = xr.DataArray(
        np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, np.nan, np.nan], dtype=float),
        dims="x",
        coords={"x": np.linspace(-6, 5, 12)},
    )
    sym_da = symmetrize(da, "x", center=0.0, mode=mode, part=part)
    expected = np.array(expected, dtype=float)
    np.testing.assert_allclose(sym_da.values, expected, rtol=1e-5)


def test_symmetrize_subtract():
    da = xr.DataArray(
        np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0], dtype=float),
        dims="x",
        coords={"x": np.linspace(-6, 5, 12)},
    )
    sym_da = symmetrize(da, "x", center=0.0, subtract=True)
    expected = np.array(
        [1.5, 2.0, 2.0, 2.0, 2.0, 1.0, -1.0, -2.0, -2.0, -2.0, -2.0, -1.5], dtype=float
    )
    np.testing.assert_allclose(sym_da.values, expected, rtol=1e-5)


@pytest.mark.parametrize("use_dask", [False, True], ids=["numpy", "dask"])
def test_symmetrize_average_multidimensional_singleton_halves(
    use_dask: bool,
) -> None:
    da = xr.DataArray(
        [[1.0, 4.0], [2.0, 6.0]],
        dims=("batch", "x"),
        coords={"batch": ["a", "b"], "x": [1.1, 1.2]},
        name="intensity",
        attrs={"units": "counts"},
    )
    if use_dask:
        da = da.chunk()
    expected_sum = da.copy(data=[[5.0, 5.0], [8.0, 8.0]])
    expected_average = da.copy(data=[[2.5, 2.5], [4.0, 4.0]])

    summed = symmetrize(da, "x", center=1.15)
    averaged = symmetrize(da, "x", center=1.15, average=True)

    xr.testing.assert_allclose(summed, expected_sum)
    xr.testing.assert_allclose(averaged, expected_average)
    assert averaged.name == da.name
    assert averaged.attrs == da.attrs


@pytest.mark.parametrize(
    ("mode", "subtract", "expected"),
    [
        (
            "full",
            False,
            [1.5, 1.5, 2.5, 3.5, 4.5, 5.0, 5.0, 4.5, 3.5, 2.5, 1.5, 1.5],
        ),
        (
            "full",
            True,
            [1.5, 1.0, 1.0, 1.0, 1.0, 0.5, -0.5, -1.0, -1.0, -1.0, -1.0, -1.5],
        ),
        (
            "valid",
            False,
            [1.5, 2.5, 3.5, 4.5, 5.0, 5.0, 4.5, 3.5, 2.5, 1.5],
        ),
    ],
)
def test_symmetrize_average(mode, subtract, expected):
    da = xr.DataArray(
        np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0], dtype=float),
        dims="x",
        coords={"x": np.linspace(-6, 5, 12)},
        name="intensity",
    )
    sym_da = symmetrize(
        da,
        "x",
        center=0.0,
        subtract=subtract,
        average=True,
        mode=mode,
    )
    np.testing.assert_allclose(sym_da.values, expected, rtol=1e-5)
    assert sym_da.name == da.name


def test_symmetrize_non_uniform() -> None:
    # Test that symmetrize raises an error when the coordinate is non-uniform.
    da = xr.DataArray(
        np.array([1, 2, 3, 4], dtype=float),
        dims="x",
        coords={"x": np.array([0.0, 1.0, 3.0, 6.0])},  # non-evenly spaced
    )
    with pytest.raises(
        ValueError, match="Coordinate along dimension x must be uniformly spaced"
    ):
        symmetrize(da, "x", center=0.0)


def test_symmetrize_singleton_coord() -> None:
    da = xr.DataArray(
        np.array([1.0], dtype=float),
        dims="x",
        coords={"x": np.array([0.0])},
    )
    with pytest.raises(
        ValueError,
        match="Coordinate along dimension x must contain at least two values",
    ):
        symmetrize(da, "x", center=0.0)
