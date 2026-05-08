import numpy as np
from astropy.io import fits

from erlab.io.fitsutils import fits_to_xarray


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
