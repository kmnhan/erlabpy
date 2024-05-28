"""Plugin for data acquired at KRISS."""

import os
import re
from collections.abc import Iterable
from typing import ClassVar

import erlab.io.utils
from erlab.io.plugins.da30 import DA30Loader


class KRISSLoader(DA30Loader):
    name = "kriss"

    aliases = ("KRISS",)

    coordinate_attrs = ("beta", "chi", "xi", "hv", "x", "y", "z")

    additional_attrs: ClassVar[dict] = {"configuration": 4}

    @property
    def name_map(self):
        return super().name_map | {"chi": "ThetaY", "xi": "ThetaX"}

    def identify(
        self, num: int, data_dir: str | os.PathLike
    ) -> tuple[list[str], dict[str, Iterable]]:
        for file in erlab.io.utils.get_files(data_dir, extensions=(".ibw", ".zip")):
            if file.endswith(".zip"):
                match = re.match(r"(.*?)" + str(num).zfill(4) + r".zip", file)
            else:
                match = re.match(
                    r"(.*?)" + str(num).zfill(4) + str(num).zfill(3) + r".ibw", file
                )

            if match is not None:
                return [file], {}

        raise FileNotFoundError(f"No files found for scan {num} in {data_dir}")
