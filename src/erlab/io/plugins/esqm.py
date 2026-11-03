"""Data loader for the ESQM setup at Ajou University."""

__all__ = ["ESQMLoader"]

import typing

from erlab.io.plugins.fesuma import FeSuMaLoader


class ESQMLoader(FeSuMaLoader):
    """Load FeSuMa data from ESQM."""

    name = "esqm"
    description = "ESQM Lab at Ajou University"
    additional_attrs: typing.ClassVar[dict] = {"configuration": 4}

    fourier16_coefficients = (0.0, 2.6953, 0.0099912, -0.0031449)
    fourier28_coefficients = (0.0, 1.5212, -0.00075013, -0.001731)
    fourier32_coefficients = (0.0, 1.6609, 0.013249, -0.0022113)

    @property
    def file_dialog_methods(self):
        return {"ESQM Raw Data (*.h5)": (self.load, {})}
