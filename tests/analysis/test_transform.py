import numpy as np
import pytest
import xarray as xr
import xarray.testing

from erlab.analysis.transform import rotate, shift, symmetrize, symmetrize_nfold


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
