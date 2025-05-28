"""Data loader for System 1 (SARPES) at Seoul National University."""

__all__ = ["System1Loader"]

import typing

import xarray as xr

import erlab
from erlab.io.dataloader import LoaderBase


class System1Loader(LoaderBase):
    name = "snu1"
    description = "System 1 at Seoul National University"
    extensions: typing.ClassVar[set[str]] = {".itx"}

    name_map: typing.ClassVar[dict] = {
        "alpha": "Non-Energy Channel [deg]",
        "eV": "Kinetic Energy [eV]",
        "hv": "Excitation Energy",
    }
    coordinate_attrs = ("hv",)
    additional_attrs: typing.ClassVar[dict] = {"configuration": 2}
    additional_coords: typing.ClassVar[dict] = {"hv": 21.2182}

    skip_validate: bool = True
    always_single: bool = True

    @property
    def file_dialog_methods(self):
        return {"SNU System 1 Raw Data (*.itx)": (self.load, {})}

    def load_single(
        self, file_path, without_values: bool = False
    ) -> xr.Dataset | xr.DataArray:
        return erlab.io.igor.load_text(file_path, without_values=without_values)
