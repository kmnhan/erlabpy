"""Data loader for beamline I05 at Diamond."""

from typing import ClassVar

import xarray as xr

from erlab.io.dataloader import LoaderBase
from erlab.io.nexusutils import get_entry, nxgroup_to_xarray


class I05Loader(LoaderBase):
    name = "i05"

    aliases = ("diamond_i05",)

    name_map: ClassVar[dict] = {
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
        "temp_sample": "sample.temperature",
    }

    coordinate_attrs = ("beta", "delta", "chi", "xi", "hv", "x", "y", "z")
    additional_attrs: ClassVar[dict] = {"configuration": 3}

    skip_validate: bool = True
    always_single: bool = True

    @property
    def file_dialog_methods(self):
        return {"Diamond I05 Raw Data (*.nxs)": (self.load, {})}

    def load_single(self, file_path) -> xr.DataArray:
        out = nxgroup_to_xarray(get_entry(file_path), "analyser/data").squeeze()

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
