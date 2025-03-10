"""Data loader for beamline 20 LOREA at ALBA."""

__all__ = ["LOREALoader"]

import pathlib
import re
import typing

import xarray as xr

import erlab
from erlab.io.dataloader import LoaderBase


def _get_data(group):
    return group.get_default()[group.get_default().signal]


class LOREALoader(LoaderBase):
    name = "lorea"
    description = "ALBA Beamline 20 LOREA"
    extensions: typing.ClassVar[set[str]] = {".nxs", ".krx"}

    aliases = ("alba_bl20",)

    name_map: typing.ClassVar[dict] = {
        "eV": ["instrument.analyser.energies", "energies"],
        "alpha": ["instrument.analyser.angles", "angles"],
        "beta": ["instrument.analyser.defl_angles", "defl_angles"],
        "delta": "instrument.manipulator.saazimuth",  # azi
        "chi": "instrument.manipulator.sapolar",  # polar
        "xi": "instrument.manipulator.satilt",  # tilt
        "x": "instrument.manipulator.sax",
        "y": "instrument.manipulator.say",
        "z": "instrument.manipulator.saz",
        "hv": "instrument.monochromator.energy",
        "sample_temp": "sample.temperature",
    }

    coordinate_attrs = ("beta", "delta", "chi", "xi", "hv", "x", "y", "z")
    additional_attrs: typing.ClassVar[dict] = {"configuration": 3}

    skip_validate: bool = True
    always_single: bool = True

    @property
    def file_dialog_methods(self):
        return {"ALBA BL20 LOREA Raw Data (*.nxs *.krx)": (self.load, {})}

    def load_single(self, file_path, without_values: bool = False) -> xr.DataArray:
        if pathlib.Path(file_path).suffix == ".krx":
            return erlab.io.plugins.mbs.load_krax(file_path)

        return erlab.io.nexusutils.nxgroup_to_xarray(
            erlab.io.nexusutils.get_entry(file_path), _get_data, without_values
        )

    def identify(self, num, data_dir, krax=False):
        if krax:
            target_files = erlab.io.utils.get_files(data_dir, ".krx")
            pattern = re.compile(rf".+-\d-{str(num).zfill(5)}_\d.krx")
        else:
            target_files = erlab.io.utils.get_files(data_dir, ".nxs")
            pattern = re.compile(rf"{str(num).zfill(3)}_.+.nxs")

        matches = [path for path in target_files if pattern.match(path.name)]

        return matches, {}
