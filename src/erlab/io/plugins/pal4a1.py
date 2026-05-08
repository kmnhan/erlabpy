"""Data loader for beamline 4A1 at PAL."""

__all__ = ["PAL4A1Loader"]

import os
import re
import typing

import xarray as xr

import erlab
from erlab.io.dataloader import LoaderBase

_NUMERIC_UNIT_PATTERN = re.compile(
    r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)"
    r"\s*(?:[A-Za-z%][A-Za-z0-9_./%()[\]-]*|\[[^\]]+\])?\s*$"
)


def _parse_pal_scalar(value: typing.Any) -> typing.Any:
    if not isinstance(value, str):
        return value

    value = value.strip()
    if "..." in value:
        return value

    m = _NUMERIC_UNIT_PATTERN.match(value)
    if m is None:
        return value

    number = m.group(1)
    if re.search(r"[.eE]", number):
        return float(number)
    return int(number)


class PAL4A1Loader(LoaderBase):
    name = "pal4a1"
    description = "PAL Beamline 4A1"
    extensions: typing.ClassVar[set[str]] = {".itx"}

    name_map: typing.ClassVar[dict] = {
        "eV": "x",
        "alpha": ["y", "Phi"],
        "beta": ["z", "Theta"],
        "hv": "Photon Energy",
        "x": "X position",
        "y": "Y position",
        "z": "Z position",
    }
    coordinate_attrs = ("alpha", "beta", "hv", "x", "y", "z")
    additional_attrs: typing.ClassVar[dict] = {"configuration": 1}

    skip_validate: bool = True
    always_single: bool = True

    @property
    def file_dialog_methods(self):
        return {"PAL 4A1 Raw Data (*.itx)": (self.load, {})}

    def load_single(
        self, file_path: str | os.PathLike, without_values: bool = False
    ) -> xr.DataArray:
        return erlab.io.igor.load_text(file_path, without_values=without_values)

    def post_process(self, data: xr.DataArray) -> xr.DataArray:
        data = data.assign_attrs(
            {key: _parse_pal_scalar(value) for key, value in data.attrs.items()}
        )
        return super().post_process(data)
