from __future__ import annotations

__all__ = ["fits_to_xarray", "load_fits7"]
import typing

import numpy as np
import numpy.typing as npt
import xarray as xr

import erlab

if typing.TYPE_CHECKING:
    import os
    from collections.abc import Hashable

    from astropy.io import fits
else:
    fits = erlab.utils.misc.LazyImport(
        "astropy.io.fits",
        err_msg="The `astropy` package is required to handle FITS files",
    )


class AxisInfo(typing.TypedDict):
    ctype: str
    crpix: float
    crval: float
    cdelt: float
    unit: str


def parse_tdim(tdim_str: str) -> tuple[int, ...]:
    # Example: '(32,32)'
    tdim_str = tdim_str.strip("() ")
    return tuple(int(x) for x in tdim_str.split(","))


def parse_tfloat(tfloat_str: str) -> list[float]:
    # Example: '(0.0,0.0)'
    tfloat_str = tfloat_str.strip("() ")
    return [float(x) for x in tfloat_str.split(",")]


def parse_tstring(tstr: str) -> list[str]:
    # Example: '(X, Y)'
    tstr = tstr.strip("() ")
    return [x.strip() for x in tstr.split(",")]


def get_axis_info(header: fits.Header, i: int) -> AxisInfo:
    # i is 1-based
    return {
        "ctype": header.get(f"CTYPE{i}", f"AXIS{i}"),
        "crpix": header.get(f"CRPIX{i}", 1.0) - 1.0,
        "crval": header.get(f"CRVAL{i}", 0.0),
        "cdelt": header.get(f"CDELT{i}", 1.0),
        "unit": header.get(f"CUNIT{i}", ""),
    }


def make_coords(header, shape) -> tuple[dict[str, npt.NDArray], tuple[str, ...]]:
    coords: dict[str, npt.NDArray] = {}
    dims: tuple[str, ...] = ()
    for i, n in enumerate(shape):
        axis: AxisInfo = get_axis_info(header, i + 1)
        start = axis["crval"] - axis["crpix"] * axis["cdelt"]
        stop = start + n * axis["cdelt"]
        label = axis["ctype"].strip() or f"dim_{i}"
        coords[label] = np.linspace(start, stop - axis["cdelt"], n)
        dims = (*dims, label)
    return coords, dims


def parse_bintable_column(header, col, nrows):
    # Handle TDIM, TRVAL, TRPIX, TDELT, TDESC, TUNIT
    tdim = header.get(f"TDIM{col}", None)
    tdesc = header.get(f"TDESC{col}", None)
    trpix = header.get(f"TRPIX{col}", None)
    trval = header.get(f"TRVAL{col}", None)
    tdelt = header.get(f"TDELT{col}", None)
    tunit = header.get(f"TUNIT{col}", None)
    coords = {}
    dims = []
    if tdim:
        shape = parse_tdim(tdim)
        dims = (
            parse_tstring(tdesc) if tdesc else [f"dim_{i}" for i in range(len(shape))]
        )
        # Add row dimension
        dims.append(f"dim_{col - 1}")
        shape = (*shape, nrows)
        # Coordinates for image axes
        if trpix and trval and tdelt:
            pix = parse_tfloat(trpix)
            val = parse_tfloat(trval)
            delt = parse_tfloat(tdelt)
            for i, d in enumerate(dims[:-1]):
                start = val[i] - (pix[i]) * delt[i]
                coords[d] = np.arange(shape[i]) * delt[i] + start

    else:
        # 1D vector or scalar
        dims = [header.get(f"TTYPE{col}", f"col_{col}")]
        shape = (nrows,)
    return coords, dims, shape, tunit


def fits_to_xarray(filename: str | os.PathLike) -> xr.DataTree:
    """Load the contents of a FITS file into a :class:`xarray.DataTree`.

    This function corresponds to the functionality of the ``Fits Loader`` Igor
    procedure; it loads the contents of a FITS file into a :class:`xarray.DataTree` with
    minimal processing. To load ALS BL7 or BL10 data, use :func:`load_fits7` which wraps
    this function and processes the data further.

    Parameters
    ----------
    filename
        Path to the FITS file.


    Returns
    -------
    xarray.DataTree

    See Also
    --------
    load_fits7

    """
    hdul = fits.open(filename)
    result: dict[str, typing.Any] = {}

    main_header: fits.Header | None = None
    for hdu_idx, hdu in enumerate(hdul):
        hname = hdu.name if hdu.name.strip() else f"EXT{hdu_idx}"
        if isinstance(hdu, fits.PrimaryHDU | fits.ImageHDU):
            if hdu.data is None:
                # Primary HDU; no data, just store header
                hdu.verify("silentfix")
                main_header = hdu.header
                continue

            data = hdu.data
            coords, dims = make_coords(hdu.header, data.shape)
            da = xr.DataArray(data, coords=coords, dims=dims, name=hname)
            # Assign units if present
            bunit = hdu.header.get("BUNIT", None)
            if bunit:
                da.attrs["units"] = bunit
            result[hname] = da.to_dataset(name=hname)

        elif isinstance(hdu, fits.BinTableHDU):
            result[hname] = {}
            nrows = hdu.data.shape[0]
            for col_idx, col in enumerate(hdu.columns, 1):
                colname = col.name
                arr = erlab.utils.array.to_native_endian(hdu.data[colname].T)

                coords, dims, shape, tunit = parse_bintable_column(
                    hdu.header, col_idx, nrows
                )

                if arr.shape != shape:
                    raise ValueError(
                        f"Shape mismatch for column {colname} in {hname}: "
                        f"expected {shape}, got {arr.shape}"
                    )

                da = xr.DataArray(arr, coords=coords, dims=dims, name=colname)
                if tunit:
                    da.attrs["units"] = tunit

                result[hname][colname] = da

            result[hname] = xr.Dataset(result[hname], attrs=main_header)

    hdul.close()

    return xr.DataTree.from_dict(result)


def process_fits_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Process a FITS ARPES dataset by assigning proper coordinates.

    Assumes a format used by beamlines 10.0.1 and 7.0.1 at ALS, and possibly others.
    Roughly Corresponds to ``RedimAndScaleData`` in ``LoadFits7.ipf``.
    """
    # Make all auxiliary coordinates dependent on time
    coords_to_update: list[Hashable] = [
        c
        for c in ds.coords
        if c not in next(iter(ds.data_vars.values())).dims and c != "time"
    ]
    ds = ds.assign_coords(
        {
            c: xr.DataArray(
                ds.coords[c].data,
                coords={"time": ds.coords["time"]},
                attrs=ds.coords[c].attrs,
            )
            for c in coords_to_update
        }
    )

    # Remove comment section from attributes
    if "COMMENT" in ds.attrs:
        del ds.attrs["COMMENT"]

    # Assign proper scan coordinates to data
    scan_type = ds.attrs.get("LWLVNM", None)
    if scan_type is not None:
        match scan_type:
            case "None":
                return ds
            case (
                "One Motor"
                | "Beamline"
                | "DAC"
                | "Manual"
                | "Beta_Compensated"
                | "SES"
                | "Line Scan Fine"
            ):
                # Get dim name without coordinates
                for d in next(iter(ds.data_vars.values())).dims:
                    if d not in ds.coords:
                        d_name = d
                        break

                # Assign coordinate to scan axis
                ds = ds.rename({d_name: "time"})

                motor_axis = ds.attrs.get("NM_0_0", "")

                if motor_axis not in ds.coords:
                    # If motor axis is not in coords, create it
                    # This is not accessed in most cases but is added for safety
                    n0 = int(ds.attrs.get("N_0_0", 0))
                    st0 = float(ds.attrs.get("ST_0_0", 0))
                    en0 = float(ds.attrs.get("EN_0_0", 0))

                    ds = ds.assign_coords(
                        {motor_axis: ("time", np.linspace(st0, en0, n0))}
                    )

                ds = ds.swap_dims({"time": motor_axis})

            case _:
                raise ValueError(
                    "Only single motor scans are supported for FITS files. "
                    f"Given scan type is {scan_type}."
                )

    return ds


def load_fits7(filename: str | os.PathLike) -> xr.Dataset | xr.DataArray:
    """Load ARPES data saved to a FITS file.

    Loads ARPES data assuming a format used by beamlines 10.0.1 and 7.0.1 at ALS, and
    possibly others.

    Parameters
    ----------
    filename
        Path to the FITS file.

    """
    dt = fits_to_xarray(filename)

    if len(dt.children) != 1:
        raise ValueError(
            "The FITS file contains multiple HDUs. It may be not supported by this "
            "function. Please report this issue with a minimal example file."
        )

    ds = process_fits_dataset(next(iter(dt.children.values())).dataset)
    if len(ds.data_vars) == 1:
        # If only one data variable, return it directly
        return next(iter(ds.data_vars.values())).assign_attrs(ds.attrs).squeeze()

    return ds
