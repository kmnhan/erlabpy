"""Data loader for beamline ID21 ESM at NSLS-II."""

import os
import re
from typing import ClassVar

import erlab.io.utils
from erlab.io.plugins.da30 import DA30Loader


class ESMLoader(DA30Loader):
    name = "esm"

    aliases = ("bnl", "id21")

    coordinate_attrs = ("beta", "hv")

    additional_attrs: ClassVar[dict] = {"configuration": 3}

    def identify(self, num: int, data_dir: str | os.PathLike):
        for file in erlab.io.utils.get_files(
            data_dir, extensions=(".ibw", ".pxt", ".zip")
        ):
            match file.suffix:
                case ".zip":
                    m = re.match(r"(.*?)" + str(num).zfill(4) + r".zip", file.name)

                case ".pxt":
                    m = re.match(r"(.*?)" + str(num).zfill(4) + r".pxt", file.name)

                case ".ibw":
                    m = re.match(
                        r"(.*?)" + str(num).zfill(4) + str(num).zfill(3) + r".ibw",
                        file.name,
                    )

            if m is not None:
                return [file], {}

        return None
