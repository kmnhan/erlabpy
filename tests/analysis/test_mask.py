import numpy as np
import xarray as xr
from erlab.analysis.mask import mask_with_hex_bz, mask_with_polygon, spherical_mask


def test_mask_with_hex_bz():
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


def test_mask_with_polygon():
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


def test_spherical_mask():
    # Create a test data array
    darr = xr.DataArray(np.arange(25).reshape(5, 5), dims=("x", "y"))

    # Test case 1: Single radius, multiple dimensions
    mask = spherical_mask(darr, radius=2, x=2, y=2)
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
    mask = spherical_mask(darr, radius=1, x=2)
    expected_mask = xr.DataArray([False, True, True, True, False], dims=("x",))
    xr.testing.assert_equal(mask, expected_mask)

    # Test case 3: Dictionary of radii, multiple dimensions
    mask = spherical_mask(darr, radius={"x": 2, "y": 1}, x=2, y=2)
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
