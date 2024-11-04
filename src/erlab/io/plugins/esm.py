"""Data loader for beamline ID21 ESM at NSLS-II."""

from typing import ClassVar

from erlab.io.plugins.da30 import DA30Loader


class ESMLoader(DA30Loader):
    name = "esm"

    aliases = ("bnl", "id21")

    coordinate_attrs = ("beta", "hv")

    additional_attrs: ClassVar[dict] = {"configuration": 3}
