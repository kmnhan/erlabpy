"""Loader for Scienta Omicron DA30L analyzer with SES.

Provides a base class for implementing loaders that can load data acquired with Scienta
Omicron's DA30L analyzer using ``SES.exe``. Must be subclassed to implement the actual
loading.
"""

import configparser
import os
import tempfile
import zipfile
from typing import ClassVar

import numpy as np
import xarray as xr

from erlab.io.dataloader import LoaderBase
from erlab.io.igor import load_experiment, load_wave


class CasePreservingConfigParser(configparser.ConfigParser):
    def optionxform(self, optionstr):
        return str(optionstr)


class DA30Loader(LoaderBase):
    name = "da30"
    aliases = ("DA30",)

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

    def load_single(self, file_path: str | os.PathLike) -> xr.DataArray:
        ext = os.path.splitext(file_path)[-1]

        match ext:
            case ".ibw":
                data = load_wave(file_path)
            case ".pxt":
                data = load_experiment(file_path)
            case ".zip":
                data = load_zip(file_path)
            case _:
                raise ValueError(f"Unsupported file extension {ext}")

        return self.post_process_general(data)

    def post_process(self, data: xr.DataArray) -> xr.DataArray:
        data = super().post_process(data)

        if "beta" not in data.coords:
            data = data.assign_coords(beta=0.0)

        return data


def load_zip(
    filename: str | os.PathLike,
) -> xr.DataArray | xr.Dataset | list[xr.DataArray]:
    with zipfile.ZipFile(filename) as z:
        regions: list[str] = [
            fn[9:-4]
            for fn in z.namelist()
            if fn.startswith("Spectrum_") and fn.endswith(".bin")
        ]
        out = []
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

            arr = xr.DataArray(
                arr.reshape(shape), coords=coords, name=region_info["name"], attrs=attrs
            )
            out.append(arr)

    if len(out) == 1:
        return out[0]

    try:
        # Try to merge the data without conflicts
        return xr.merge(out)
    except:  # noqa: E722
        # On failure, return a list
        return out


def parse_ini(filename: str | os.PathLike) -> dict:
    parser = CasePreservingConfigParser(strict=False)
    out = {}
    with open(filename) as f:
        parser.read_file(f)
        for section in parser.sections():
            out[section] = dict(parser.items(section))
    return out
