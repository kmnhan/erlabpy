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
        self, file_path, without_values: bool = False
    ) -> xr.Dataset | xr.DataArray:
        return erlab.io.fitsutils.load_fits7(file_path)
