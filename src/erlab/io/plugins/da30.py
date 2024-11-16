"""Loader for Scienta Omicron DA30 analyzer with SES.

Provides a base class for implementing loaders that can load data acquired with Scienta
Omicron's DA30 analyzer using ``SES.exe``. Subclass to implement the actual loading.
"""

import configparser
import os
import pathlib
import re
import tempfile
import zipfile
from collections.abc import Iterable
from typing import ClassVar

import numpy as np
import xarray as xr

import erlab.io
from erlab.io.dataloader import LoaderBase


class CasePreservingConfigParser(configparser.ConfigParser):
    """ConfigParser that preserves the case of the keys."""

    def optionxform(self, optionstr):
        return str(optionstr)


class DA30Loader(LoaderBase):
    name = "da30"
    aliases: Iterable[str] = ["DA30"]

    name_map: ClassVar[dict] = {
        "eV": ["Kinetic Energy [eV]", "Energy [eV]"],
        "alpha": ["Y-Scale [deg]", "Thetax [deg]"],
        "beta": ["Thetay [deg]"],
        "hv": ["BL Energy", "Excitation Energy"],
    }
    additional_attrs: ClassVar[dict] = {}
    always_single = True
    skip_validate = True

    @property
    def file_dialog_methods(self):
        return {"DA30 Raw Data (*.ibw *.pxt *.zip)": (self.load, {})}

    def load_single(
        self, file_path: str | os.PathLike, without_values: bool = False
    ) -> xr.DataArray | xr.Dataset | xr.DataTree:
        file_path = pathlib.Path(file_path)

        match file_path.suffix:
            case ".ibw":
                data: xr.DataArray | xr.Dataset | xr.DataTree = xr.load_dataarray(
                    file_path, engine="erlab-igor"
                )

            case ".pxt":
                data = xr.load_dataset(file_path, engine="erlab-igor")

                if len(data.data_vars) == 1:
                    # Get DataArray if single region
                    data = data[next(iter(data.data_vars))]

            case ".zip":
                data = load_zip(file_path, without_values)

            case _:
                raise ValueError(f"Unsupported file extension {file_path.suffix}")

        return data

    def identify(self, num: int, data_dir: str | os.PathLike):
        matches = []

        pattern = re.compile(r"(.*?)" + str(num).zfill(4))
        pattern_ibw = re.compile(
            r"(.*?)" + str(num).zfill(4) + ".*" + str(num).zfill(3)
        )
        for file in sorted(
            erlab.io.utils.get_files(data_dir, extensions=(".ibw", ".pxt", ".zip"))
        ):
            match file.suffix:
                case ".ibw":
                    m = pattern_ibw.match(file.stem)
                case _:
                    m = pattern.match(file.stem)

            if m is not None:
                matches.append(file)

        return matches, None

    def post_process(self, data: xr.DataArray) -> xr.DataArray:
        data = super().post_process(data)

        if "beta" not in data.coords:
            data = data.assign_coords(beta=0.0)

        return data

    def files_for_summary(self, data_dir: str | os.PathLike):
        return sorted(
            erlab.io.utils.get_files(data_dir, extensions=(".pxt", ".ibw", ".zip"))
        )


def load_zip(
    filename: str | os.PathLike, without_values: bool = False
) -> xr.DataArray | xr.Dataset | xr.DataTree:
    """Load data from a ``.zip`` file from a Scienta Omicron DA30 analyzer.

    If the file contains a single region, a DataArray is returned. If the file contains
    multiple regions that can be merged without conflicts, a Dataset is returned. If the
    regions cannot be merged without conflicts, a DataTree containing all regions is
    returned.
    """
    with zipfile.ZipFile(filename) as z:
        regions: list[str] = [
            fn[9:-4]
            for fn in z.namelist()
            if fn.startswith("Spectrum_") and fn.endswith(".bin")
        ]
        out: list[xr.DataArray] = []
        for region in regions:
            with tempfile.TemporaryDirectory() as tmp_dir:
                z.extract(f"Spectrum_{region}.ini", tmp_dir)
                z.extract(f"{region}.ini", tmp_dir)

                unzipped = pathlib.Path(tmp_dir)

                region_info = parse_ini(unzipped / f"Spectrum_{region}.ini")["spectrum"]
                attrs = {}
                for d in parse_ini(unzipped / f"{region}.ini").values():
                    attrs.update(d)

                if not without_values:
                    z.extract(f"Spectrum_{region}.bin", tmp_dir)
                    arr = np.fromfile(
                        unzipped / f"Spectrum_{region}.bin", dtype=np.float32
                    )

            shape = []
            coords = {}
            for d in ("depth", "height", "width"):
                n = int(region_info[d])
                offset = float(region_info[f"{d}offset"])
                delta = float(region_info[f"{d}delta"])
                shape.append(n)
                coords[region_info[f"{d}label"]] = np.linspace(
                    offset, offset + (n - 1) * delta, n
                )

            if not without_values:
                arr = arr.reshape(shape)
            else:
                arr = np.zeros(shape, dtype=np.float32)

            out.append(
                xr.DataArray(arr, coords=coords, name=region_info["name"], attrs=attrs)
            )

    if len(out) == 1:
        return out[0]

    try:
        # Try to merge the data without conflicts
        return xr.merge(out, join="exact")
    except:  # noqa: E722
        # On failure, combine into DataTree
        return xr.DataTree.from_dict(
            {str(da.name): da.to_dataset(promote_attrs=True) for da in out}
        )


def _parse_value(value):
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
    return value


def parse_ini(filename: str | os.PathLike) -> dict:
    """Parse an ``.ini`` file into a dictionary."""
    parser = CasePreservingConfigParser(strict=False)
    out = {}
    with open(filename, encoding="utf-8") as f:
        parser.read_file(f)
        for section in parser.sections():
            out[section] = {k: _parse_value(v) for k, v in parser.items(section)}
    return out
