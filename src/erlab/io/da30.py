"""
Loader for `.ibw`, `.pxt`, and `.zip` files produced by the DA30 hemispherical electron
analyzer. 

"""

import configparser
import os
import tempfile
import zipfile

import numpy as np
import xarray as xr

from erlab.io.igor import load_experiment, load_wave
from erlab.io.utilities import find_first_file

COORDS_MAPPING = {
    "Kinetic Energy [eV]": "eV",
    "Energy [eV]": "eV",
    "Y-Scale [deg]": "phi",
    "Thetax [deg]": "phi",
    "Thetay [deg]": "theta",
}

ATTRS_MAPPING = {"bl_energy": "hv", "excitation_energy": "hv"}


def load(filename, data_dir=None, contains=None, **kwargs):
    xr.set_options(keep_attrs=True)
    try:
        filename = find_first_file(filename, data_dir=data_dir, contains=contains)
    except ValueError:
        pass

    if filename.endswith(".ibw"):
        wave = load_wave(filename, **kwargs)
    elif filename.endswith(".pxt"):
        wave = load_experiment(filename, **kwargs)
    elif filename.endswith(".zip"):
        wave = load_zip(filename, **kwargs)
    else:
        raise ValueError("Unsupported file type")

    if isinstance(wave, list):
        wave = [process_array(w) for w in wave]
    else:
        wave = process_array(wave)
    return wave


def process_array(wave: xr.DataArray | xr.Dataset):
    wave = wave.rename({k: v for k, v in COORDS_MAPPING.items() if k in wave.dims})

    if isinstance(wave, xr.Dataset):
        if len(wave.data_vars) == 1:
            wave = list(wave.data_vars.values())[0]

    wave = wave.assign_coords(phi=np.deg2rad(wave.phi))
    if "theta" in wave.coords:
        wave = wave.assign_coords(theta=np.deg2rad(wave.theta))
    if "hv" in wave.attrs:
        wave = wave.assign_coords(
            eV=wave.eV - float(wave.attrs["hv"]), hv=float(wave.attrs["hv"])
        )

    return wave


def load_zip(filename, merge: bool = True) -> xr.DataArray | list[xr.DataArray]:
    regions: list[str] = []

    with zipfile.ZipFile(filename) as z:
        for fn in z.namelist():
            if fn.startswith("Spectrum_") and fn.endswith(".bin"):
                regions.append(fn[9:-4])

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
            coords = dict()
            for d in ("depth", "height", "width"):
                n = int(region_info[d])
                offset = float(region_info[f"{d}offset"])
                delta = float(region_info[f"{d}delta"])
                shape.append(n)
                coords[region_info[f"{d}label"]] = np.linspace(
                    offset, offset + (n - 1) * delta, n
                )

            attrs = {k.lower().replace(" ", "_"): v for k, v in attrs.items()}
            attrs = {ATTRS_MAPPING.get(k, k): v for k, v in attrs.items()}
            arr = xr.DataArray(
                arr.reshape(shape), coords=coords, name=region_info["name"], attrs=attrs
            )
            out.append(arr)
    if merge:
        out = xr.merge(out)
    return out


def parse_ini(filename):
    parser = configparser.ConfigParser(strict=False)
    out = dict()
    with open(filename, "r") as f:
        parser.read_file(f)
        for section in parser.sections():
            out[section] = dict(parser.items(section))
    return out