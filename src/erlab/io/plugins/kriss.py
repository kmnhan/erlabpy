"""Data loader for KRISS."""

__all__ = ["KRISSLoader"]

import typing

from erlab.io.plugins.da30 import DA30Loader


class KRISSLoader(DA30Loader):
    name = "kriss"
    description = "KRISS ARPES-MBE"
    aliases = ("KRISS",)
    coordinate_attrs = ("beta", "chi", "xi", "hv")
    additional_attrs: typing.ClassVar[dict] = {"configuration": 4}

    @property
    def file_dialog_methods(self):
        return {"KRISS Data (*.ibw *.pxt *.zip)": (self.load, {})}

    @property
    def name_map(self):
        return super().name_map | {"chi": "R1", "xi": ["R2", "Point [deg]"]}
