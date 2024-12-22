"""Data loader for KRISS."""

from typing import ClassVar

from erlab.io.plugins.da30 import DA30Loader


class KRISSLoader(DA30Loader):
    name = "kriss"
    description = "KRISS ARPES-MBE"
    aliases = ("KRISS",)
    coordinate_attrs = ("beta", "chi", "xi", "hv", "x", "y", "z")
    additional_attrs: ClassVar[dict] = {"configuration": 4}

    @property
    def name_map(self):
        return super().name_map | {"chi": "ThetaY", "xi": "ThetaX"}
