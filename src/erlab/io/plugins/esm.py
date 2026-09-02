"""Data loader for beamline ID21 ESM at NSLS-II."""

__all__ = ["ESMLoader"]

import typing

from erlab.io.plugins.da30 import DA30Loader


class ESMLoader(DA30Loader):
    """Load data from the ESM beamline at NSLS-II."""

    name = "esm"
    description = "NSLS-II Beamline ID21 ESM"

    aliases = ("bnl", "id21")

    coordinate_attrs = ("beta", "hv")

    additional_attrs: typing.ClassVar[dict] = {"configuration": 3}
