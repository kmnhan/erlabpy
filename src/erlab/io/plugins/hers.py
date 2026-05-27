"""Data loader for beamline 10.0.1 at ALS."""

__all__ = ["HERSLoader"]

import re
import typing

import xarray as xr

import erlab
from erlab.io.dataloader import LoaderBase


class HERSLoader(LoaderBase):
    name = "hers"
    description = "ALS Beamline 10.0.1 HERS"
    extensions: typing.ClassVar[set[str]] = {".fits"}

    name_map: typing.ClassVar[dict] = {
        "beta": "Alpha",  # Analyzer rotation
        "hv": ("mono_eV", "MONOEV", "BL_E"),
    }
    coordinate_attrs = ("beta", "hv")
    additional_attrs: typing.ClassVar[dict] = {"configuration": 1}

    skip_validate: bool = True
    always_single: bool = True

    @property
    def file_dialog_methods(self):
        return {"ALS BL10.0.1 Raw Data (*.fits)": (self.load, {})}

    def identify(self, num, data_dir):
        pattern = re.compile(rf"(\d+)_{str(num).zfill(5)}.fits")
        matches = [
            path
            for path in erlab.io.utils.get_files(data_dir, ".fits")
            if pattern.match(path.name)
        ]
        return matches, {}

    def load_single(
        self,
        file_path,
        *,
        without_values: bool = False,
        convert_to_angles: bool = True,
        b_field_correction: bool = True,
    ) -> xr.Dataset | xr.DataArray:
        """Load a single HERS dataset from a FITS file.

        Parameters
        ----------
        file_path : str or Path
            Path to the FITS file to load.
        without_values : bool, optional
            Unused parameter for compatibility with the base class. Defaults to False.
        convert_to_angles : bool, optional
            Whether to convert pixel coordinates to angles. Defaults to True.
        b_field_correction : bool, optional
            Whether to apply correction for warping due to the magnetic field. Only
            applies if ``convert_to_angles`` is True. Defaults to True.

        """
        data = typing.cast("xr.DataArray", erlab.io.fitsutils.load_fits7(file_path))
        if convert_to_angles:  # pragma: no branch
            data = pixel_to_angle(data)
            if b_field_correction:  # pragma: no branch
                data = correct_b_field(data)
        return data


def pixel_to_angle(data: xr.DataArray) -> xr.DataArray:
    """Convert pixel coordinates to angles. Assumes A30 lens mode."""
    if "pixel" in data.coords:  # pragma: no branch
        data = data.assign_coords(pixel=0.04631 * (data.pixel - 500)).rename(
            {"pixel": "alpha"}
        )
    return data


def _extract_distortion(
    data: xr.DataArray,
    energy: float = -4.5,
    angle_range: tuple[float, float] = (16.0, 24.0),
) -> xr.DataArray:
    near_edge = data.interp(eV=energy).sel(alpha=slice(*angle_range))
    if angle_range[0] < 0:  # pragma: no cover
        bkg = near_edge.isel(alpha=slice(None, 5)).mean("alpha")
    else:
        bkg = near_edge.isel(alpha=slice(-5, None)).mean("alpha")
    near_edge = near_edge - bkg
    edge = erlab.analysis.interpolate.leading_edge(
        near_edge,
        dim="alpha",
        direction="negative" if angle_range[0] < 0 else "positive",
    )
    degree = min(5, max(0, int(edge.count().item()) - 1))

    return -xr.polyval(
        data.Alpha,
        edge.polyfit(dim="Alpha", deg=degree, full=True).polyfit_coefficients,
    )


def correct_b_field(
    data: xr.DataArray,
    energy: float = -4.5,
    angle_range: tuple[float, float] = (16.0, 24.0),
) -> xr.DataArray:
    """Apply correction to some warping in the data due to the magnetic field.

    Fixed & adapted from the original Bcor() procedure.
    """
    if all(d in data.dims for d in ("alpha", "Alpha", "eV")):
        shift = _extract_distortion(data, energy=energy, angle_range=angle_range)
        shift = shift - shift.mean()
        data = erlab.analysis.transform.shift(
            data,
            shift=shift,
            along="alpha",
            shift_coords=False,
            assume_sorted=True,
        )

    return data
