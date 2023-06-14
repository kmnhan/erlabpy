import os
from datetime import datetime

import arpes
import arpes.preparation
import numpy as np
import pandas as pd
import xarray as xr
from arpes.endstations.plugin.merlin import BL403ARPESEndstation

from erlab.io.igor import load_wave


def load_live(filename, data_dir=None):
    wave = load_wave(filename, data_dir)
    wave = wave.rename(
        {
            k: v
            for k, v in {
                "Energy": "eV",
                "Sample Y (Vert)": "y",
                "Sample X": "x",
                "Scienta": "phi",
                "Polar": "theta",
            }.items()
            if k in wave.dims
        }
    )
    wave = wave.assign_coords(eV=wave["eV"] - wave.attrs["hv"])

    endstation = BL403ARPESEndstation()
    return endstation.postprocess_final(
        endstation.postprocess(xr.Dataset(dict(spectrum=wave)))
    )


def load(filename, data_dir=None, **kwargs):
    return arpes.io.load_data(filename, location="BL4", data_dir=data_dir, **kwargs)


def folder_summary(data_dir, exclude_live=False):
    fnames = dict()
    for fname in BL403ARPESEndstation.files_for_search(data_dir):
        try:
            name = BL403ARPESEndstation.find_first_file(
                fname[2:5], dict(file=fname[2:5]), data_dir=data_dir
            )
            fnames[name] = os.path.splitext(os.path.basename(name))[0][:5]
        except ValueError:
            name = os.path.join(data_dir, fname)
            fnames[name] = os.path.splitext(os.path.basename(name))[0]

    if not exclude_live:
        for fname in os.listdir(data_dir):
            if fname.endswith(".ibw"):
                name = os.path.join(data_dir, fname)
                fnames[name] = os.path.splitext(os.path.basename(name))[0]

    data_attrs = {
        "spectrum_type": "Type",
        "lens_mode_name": "Lens Mode",
        "acquisition_mode": "Scan Type",
        "temperature": "T(K)",
        "pass_energy": "Pass E",
        "slit_number": "Analyzer Slit",
        "undulator_polarization": "Polarization",
        "hv": "hv",
        "entrance_slit": "Entrance Slit",
        "exit_slit": "Exit Slit",
        "x": "x",
        "y": "y",
        "z": "z",
        "beta": "tilt",
        "chi": "azi",
    }

    data_info = []
    time_list = []
    for path, name in fnames.items():
        if os.path.splitext(path)[1] == ".ibw":
            data = load_live(path)
        else:
            data = load(path)
        time_list.append(
            datetime.strptime(
                f"{data.attrs['date']} {data.attrs['time']}", "%d/%m/%Y %I:%M:%S %p"
            )  # .replace(tzinfo=ZoneInfo("America/Los_Angeles"))
        )
        data_info.append([name])

        for k in data_attrs.keys():
            val = data.attrs[k]
            if k == "lens_mode_name":
                val = val.replace("Angular", "A")
            elif k in ("entrance_slit", "exit_slit"):
                val = round(val)
            elif k == "undulator_polarization":
                val = {0: "LH", 2: "LV", -1: "RC", 1: "LC"}[val]
            elif k in ("chi", "beta"):
                val = np.rad2deg(val)
            data_info[-1].append(val)
        del data

    return pd.DataFrame(
        data_info, index=time_list, columns=["File Name"] + list(data_attrs.values())
    ).sort_index()
