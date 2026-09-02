"""Data loader for TPS Beamline 39A at NSRRC."""

__all__ = ["TPS39ALoader"]

import typing

from erlab.io.plugins.da30 import DA30Loader


class TPS39ALoader(DA30Loader):
    """Load Scienta analyzer data from TPS Beamline 39A at NSRRC."""

    name = "tps39a"
    description = "NSRRC TPS Beamline 39A"
    coordinate_attrs = ("beta", "hv")
    additional_attrs: typing.ClassVar[dict] = {"configuration": 4}

    @property
    def file_dialog_methods(self):
        return {"TPS 39A Data (*.ibw *.pxt *.zip)": (self.load, {})}
