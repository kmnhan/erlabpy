"""Data loader for beamline I05 at Diamond."""

__all__ = ["I05Loader"]

import typing

import xarray as xr

import erlab
from erlab.io.dataloader import LoaderBase


class I05Loader(LoaderBase):
    name = "i05"
    description = "Diamond Beamline I05"
    extensions: typing.ClassVar[set[str]] = {".nxs"}

    aliases = ("diamond_i05",)

    name_map: typing.ClassVar[dict] = {
        "eV": "instrument.analyser.energies",
        "alpha": "instrument.analyser.angles",
        "beta": [
            "instrument.deflector_x.deflector_x",
            "instrument.analyser.deflector_x",
        ],
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
        return {"Diamond I05 Raw Data (*.nxs)": (self.load, {})}

    def load_single(self, file_path, without_values=False) -> xr.DataArray:
        out = erlab.io.nexusutils.nxgroup_to_xarray(
            erlab.io.nexusutils.get_entry(file_path), "analyser/data", without_values
        ).squeeze()

        if (
            "instrument.centre_energy.centre_energy" in out.dims
            and "instrument.monochromator.energy" in out.coords
        ):
            # swap_dims not working for 2D coords, workaround
            out = (
                out.assign_coords(
                    {
                        "instrument.monochromator.energy": out[
                            "instrument.centre_energy.centre_energy"
                        ],
                        "instrument.centre_energy.centre_energy": out[
                            "instrument.monochromator.energy"
                        ],
                    }
                )
                .rename(
                    {
                        "instrument.monochromator.energy": "energy_temp",
                        "instrument.centre_energy.centre_energy": "hv_temp",
                    }
                )
                .rename(
                    {
                        "hv_temp": "instrument.monochromator.energy",
                        "energy_temp": "instrument.centre_energy.centre_energy",
                    }
                )
            )
        return out
