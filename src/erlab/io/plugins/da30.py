"""Loader for Scienta Omicron DA30 analyzer with SES.

Provides a base class for implementing loaders that can load data acquired with Scienta
Omicron's DA30 analyzer using ``SES.exe``. Subclass to implement the actual loading.
"""

import configparser
import os
import tempfile
import zipfile
from collections.abc import Iterable
from typing import ClassVar

import numpy as np
import xarray as xr
from xarray.core.datatree import DataTree

from erlab.io.dataloader import LoaderBase


class CasePreservingConfigParser(configparser.ConfigParser):
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
        self, file_path: str | os.PathLike
    ) -> xr.DataArray | xr.Dataset | DataTree:
        ext = os.path.splitext(file_path)[-1]

        match ext:
            case ".ibw":
                data: xr.DataArray | xr.Dataset | DataTree = xr.load_dataarray(
                    file_path, engine="erlab-igor"
                )

            case ".pxt":
                data = xr.load_dataset(file_path, engine="erlab-igor")

                if len(data.data_vars) == 1:
                    # Get DataArray if single region
                    data = data[next(iter(data.data_vars))]

            case ".zip":
                data = load_zip(file_path)

            case _:
                raise ValueError(f"Unsupported file extension {ext}")

        return data

    def post_process(self, data: xr.DataArray) -> xr.DataArray:
        data = super().post_process(data)

        if "beta" not in data.coords:
            data = data.assign_coords(beta=0.0)

        return data


def load_zip(
    filename: str | os.PathLike,
) -> xr.DataArray | xr.Dataset | DataTree:
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
                z.extract(f"Spectrum_{region}.bin", tmp_dir)

                region_info = parse_ini(
                    os.path.join(tmp_dir, f"Spectrum_{region}.ini")
                )["spectrum"]
                attrs = {}
                for d in parse_ini(os.path.join(tmp_dir, f"{region}.ini")).values():
                    attrs.update(d)

                arr = np.fromfile(
                    os.path.join(tmp_dir, f"Spectrum_{region}.bin"), dtype=np.float32
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

            out.append(
                xr.DataArray(
                    arr.reshape(shape),
                    coords=coords,
                    name=region_info["name"],
                    attrs=attrs,
                )
            )

    if len(out) == 1:
        return out[0]

    try:
        # Try to merge the data without conflicts
        return xr.merge(out, join="exact")
    except:  # noqa: E722
        # On failure, combine into DataTree
        return DataTree.from_dict(
            {str(da.name): da.to_dataset(promote_attrs=True) for da in out}
        )


def parse_ini(filename: str | os.PathLike) -> dict:
    parser = CasePreservingConfigParser(strict=False)
    out = {}
    with open(filename, encoding="utf-8") as f:
        parser.read_file(f)
        for section in parser.sections():
            out[section] = dict(parser.items(section))
    return out
