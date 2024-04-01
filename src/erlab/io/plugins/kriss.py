"""Plugin for data acquired at KRISS."""

import os
import re

import numpy as np
import numpy.typing as npt

import erlab.io.utilities
from erlab.io.plugins.da30 import DA30Loader


class KRISSLoader(DA30Loader):
    name: str = "kriss"

    aliases: list[str] = ["KRISS"]

    coordinate_attrs: tuple[str, ...] = ("beta", "chi", "xi", "hv", "x", "y", "z")

    additional_attrs: dict[str, str | int | float] = {"configuration": 4}

    @property
    def name_map(self):
        return super().name_map | {"chi": "ThetaY", "xi": "ThetaX"}

    def identify(
        self, num: int, data_dir: str | os.PathLike
    ) -> tuple[list[str], dict[str, npt.NDArray[np.float64]]]:
        for file in erlab.io.utilities.get_files(data_dir, extensions=(".ibw", ".zip")):
            if file.endswith(".zip"):
                match = re.match(r"(.*?)" + str(num).zfill(4) + r".zip", file)
            else:
                match = re.match(
                    r"(.*?)" + str(num).zfill(4) + str(num).zfill(3) + r".ibw", file
                )

            if match is not None:
                return [file], {}

        raise FileNotFoundError(f"No files found for scan {num} in {data_dir}")
