import numpy as np
import pytest
import xarray as xr
from astropy.io import fits

from erlab.io.fitsutils import fits_to_xarray, process_fits_dataset


def test_bintable_image_trpix_is_one_based(tmp_path) -> None:
    column = fits.Column(
        name="image",
        format="6D",
        dim="(2,3)",
        array=[np.arange(6, dtype=np.float64)],
    )
    hdu = fits.BinTableHDU.from_columns([column], name="data")
    hdu.header["TDESC1"] = "(x,y)"
    hdu.header["TRPIX1"] = "(1,1)"
    hdu.header["TRVAL1"] = "(10,20)"
    hdu.header["TDELT1"] = "(2,3)"

    path = tmp_path / "image.fits"
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)

    data = next(iter(fits_to_xarray(path).children.values())).dataset["image"]

    np.testing.assert_allclose(data.coords["x"], [10.0, 12.0])
    np.testing.assert_allclose(data.coords["y"], [20.0, 23.0, 26.0])


def test_bintable_image_without_axis_scale_has_no_image_coords(tmp_path) -> None:
    column = fits.Column(
        name="image",
        format="6D",
        dim="(2,3)",
        array=[np.arange(6, dtype=np.float64)],
    )
    hdu = fits.BinTableHDU.from_columns([column], name="data")
    hdu.header["TDESC1"] = "(x,y)"

    path = tmp_path / "image.fits"
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)

    data = next(iter(fits_to_xarray(path).children.values())).dataset["image"]

    assert data.dims == ("x", "y", "dim_0")
    assert "x" not in data.coords
    assert "y" not in data.coords


def test_process_fits_dataset_without_scan_type_is_unchanged() -> None:
    data = xr.Dataset({"signal": xr.DataArray(np.arange(3.0), dims=("scan",))})

    xr.testing.assert_identical(process_fits_dataset(data), data)


def test_process_fits_dataset_without_motor_name_is_unchanged() -> None:
    data = xr.Dataset(
        {"signal": xr.DataArray(np.arange(3.0), dims=("scan",))},
        attrs={"LWLVNM": "Beamline", "NM_0_0": " "},
    )

    xr.testing.assert_identical(process_fits_dataset(data), data)


def test_process_fits_dataset_single_motor_without_readback() -> None:
    data = xr.Dataset(
        {
            "signal": xr.DataArray(
                np.zeros((2, 3)),
                dims=("energy", "scan"),
                coords={"energy": [0.0, 1.0]},
            )
        },
        attrs={
            "LWLVNM": "Beamline",
            "NM_0_0": "Alpha",
            "ST_0_0": 0.0,
            "EN_0_0": 4.0,
            "N_0_0": 5,
        },
    )

    processed = process_fits_dataset(data)

    assert processed.signal.dims == ("energy", "Alpha")
    np.testing.assert_allclose(processed.Alpha, [0.0, 1.0, 2.0])
    assert "Alpha_readback" not in processed.coords


def test_process_fits_dataset_single_point_scan_uses_start_value() -> None:
    data = xr.Dataset(
        {"signal": xr.DataArray(np.arange(1.0), dims=("scan",))},
        attrs={
            "LWLVNM": "Beamline",
            "NM_0_0": "Alpha",
            "ST_0_0": 2.0,
            "EN_0_0": 9.0,
            "N_0_0": 1,
        },
    )

    processed = process_fits_dataset(data)

    np.testing.assert_allclose(processed.Alpha, [2.0])


def test_process_fits_dataset_requires_unindexed_scan_dimension() -> None:
    data = xr.Dataset(
        {
            "signal": xr.DataArray(
                np.arange(3.0),
                dims=("Alpha",),
                coords={"Alpha": [0.0, 1.0, 2.0]},
            )
        },
        attrs={"LWLVNM": "Beamline", "NM_0_0": "Alpha"},
    )

    with pytest.raises(ValueError, match="unindexed scan dimension"):
        process_fits_dataset(data)


def test_process_fits_dataset_rejects_unsupported_scan_type() -> None:
    data = xr.Dataset(
        {"signal": xr.DataArray(np.arange(3.0), dims=("scan",))},
        attrs={"LWLVNM": "Two Motor"},
    )

    with pytest.raises(ValueError, match="Only single motor scans"):
        process_fits_dataset(data)
