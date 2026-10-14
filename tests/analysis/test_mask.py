import numpy as np
import pytest
import xarray as xr

from erlab.analysis.mask import (
    _spherical_mask,
    mask_with_hex_bz,
    mask_with_polygon,
    mask_with_radius,
)


def test_mask_with_hex_bz() -> None:
    rng = np.random.default_rng(1)
    kxymap = xr.DataArray(
        rng.random((5, 5)),
        dims=("kx", "ky"),
        coords={"kx": np.linspace(-1, 1, 5), "ky": np.linspace(-1, 1, 5)},
    )

    # Call the function
    masked_map = mask_with_hex_bz(kxymap, a=5, rotate=0.0, invert=False)

    # Assert the shape of the masked map is the same as the input map
    assert masked_map.shape == kxymap.shape
    # Assert only the edges are masked
    assert np.isnan(masked_map[0, :]).all()
    assert np.isnan(masked_map[-1, :]).all()
    assert np.isnan(masked_map[:, 0]).all()
    assert np.isnan(masked_map[:, -1]).all()
    xr.testing.assert_equal(masked_map[1:-1, 1:-1], kxymap[1:-1, 1:-1])


def test_mask_with_polygon() -> None:
    # Create a sample input array
    arr = xr.DataArray(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        dims=("kx", "ky"),
        coords={"kx": [0, 1, 2], "ky": [0, 1, 2]},
    )

    # Define the vertices of the polygon
    vertices = np.array([[0.5, 0.5], [0.5, 2.5], [2.5, 2.5], [2.5, 0.5]])

    # Apply the mask with the polygon
    masked_arr = mask_with_polygon(arr, vertices)

    # Define the expected masked array
    expected_mask = xr.DataArray(
        np.array([[False, False, False], [False, True, True], [False, True, True]]),
        dims=("kx", "ky"),
        coords={"kx": [0, 1, 2], "ky": [0, 1, 2]},
    )

    # Check if the masked array matches the expected result
    xr.testing.assert_equal(masked_arr, arr.where(expected_mask))


def test_spherical_mask() -> None:
    # Create a test data array
    darr = xr.DataArray(np.arange(25).reshape(5, 5), dims=("x", "y"))

    # Test case 1: Single radius, multiple dimensions
    mask = _spherical_mask(darr, radius=2, x=2, y=2)
    expected_mask = xr.DataArray(
        [
            [False, False, True, False, False],
            [False, True, True, True, False],
            [True, True, True, True, True],
            [False, True, True, True, False],
            [False, False, True, False, False],
        ],
        dims=("x", "y"),
    )
    xr.testing.assert_equal(mask, expected_mask)

    # Test case 2: Single radius, single dimension
    mask = _spherical_mask(darr, radius=1, x=2)
    expected_mask = xr.DataArray([False, True, True, True, False], dims=("x",))
    xr.testing.assert_equal(mask, expected_mask)

    # Test case 3: Dictionary of radii, multiple dimensions
    mask = _spherical_mask(darr, radius={"x": 2, "y": 1}, x=2, y=2)
    expected_mask = xr.DataArray(
        [
            [False, False, True, False, False],
            [False, False, True, False, False],
            [False, True, True, True, False],
            [False, False, True, False, False],
            [False, False, True, False, False],
        ],
        dims=("x", "y"),
    )
    xr.testing.assert_equal(mask, expected_mask)


@pytest.mark.parametrize("invert", [False, True])
@pytest.mark.parametrize("drop", [False, True])
@pytest.mark.parametrize("use_dask", [False, True], ids=["no_dask", "dask"])
@pytest.mark.parametrize("scalar_params", [False, True], ids=["vector", "scalar"])
def test_mask_with_radius_ignores_parameter_coordinates(
    invert: bool, drop: bool, use_dask: bool, scalar_params: bool
) -> None:
    data = xr.DataArray(
        np.arange(15.0).reshape(5, 3),
        dims=("x", "y"),
        coords={
            "x": np.arange(-2.0, 3.0),
            "y": [10.0, 20.0, 30.0],
            "sample": "data",
            "sample_position": ("y", [1.0, 2.0, 3.0]),
        },
        name="intensity",
        attrs={"units": "counts"},
    )
    if use_dask:
        data = data.chunk({"x": 2, "y": 1})
    center = (
        xr.DataArray(0.0)
        if scalar_params
        else xr.DataArray([-1.0, 0.0, 1.0], dims="y", coords={"y": data.y})
    )
    radius = (
        xr.DataArray(1.0)
        if scalar_params
        else xr.DataArray([0.5, 1.0, 1.5], dims="y", coords={"y": data.y})
    )
    extra_center = center.assign_coords(sample="reference", center_source="fit")
    extra_radius = radius.assign_coords(sample="reference", radius_source="fit")
    if not scalar_params:
        extra_center = extra_center.assign_coords(
            sample_position=("y", [4.0, 5.0, 6.0]),
            center_position=("y", [7.0, 8.0, 9.0]),
        )

    expected = mask_with_radius(data, {"x": radius}, x=center, invert=invert, drop=drop)
    actual = mask_with_radius(
        data, {"x": extra_radius}, x=extra_center, invert=invert, drop=drop
    )

    xr.testing.assert_identical(actual, expected)
    assert bool(actual.chunks) == use_dask
