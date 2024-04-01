"""Data loader for beamline 4.0.3 at ALS."""

import glob
import os
import re
import datetime
import numpy as np
import numpy.typing as npt
import xarray as xr
import pandas as pd


import erlab.io.utilities
from erlab.io.dataloader import LoaderBase
from erlab.io.igor import load_experiment, load_wave


class BL403Loader(LoaderBase):
    name: str = "merlin"
    aliases: list[str] = ["ALS_BL4", "als_bl4", "BL403", "bl403"]

    name_map: dict[str, str | list[str]] = {
        "alpha": "deg",
        "beta": ["Polar", "Polar Compens"],
        "delta": "Azimuth",
        "xi": "Tilt",
        "x": "Sample X",
        "y": "Sample Y (Vert)",
        "z": "Sample Z",
        "hv": "BL Energy",
        "polarization": "EPU POL",
        "temp_sample": "Temperature Sensor B",
        "photon_flux": "Mesh Current",
    }
    coordinate_attrs: tuple[str, ...] = (
        "beta",
        "delta",
        "xi",
        "hv",
        "x",
        "y",
        "z",
        "polarization",
        "photon_flux",
    )
    additional_attrs: dict[str, str | int | float] = {
        "configuration": 1,
        "sample_workfunction": 4.44,
    }

    def load_single(self, file_path: str | os.PathLike) -> xr.DataArray:
        data = load_experiment(file_path)
        # One file always corresponds to single region, so assume only one data variable
        data: xr.DataArray = data.data_vars[next(iter(data.data_vars.keys()))]

        return self.process_keys(data)

    def identify(
        self, num: int, data_dir: str | os.PathLike
    ) -> tuple[list[str], dict[str, npt.NDArray[np.float64]]]:
        coord_dict: dict[str, npt.NDArray[np.float64]] = {}

        # Look for scans
        files = glob.glob(f"*_{str(num).zfill(3)}_S*.pxt", root_dir=data_dir)
        files.sort()
        # Assume files sorted by scan #

        if len(files) == 0:
            # Look for multiregion scan
            files = glob.glob(f"*_{str(num).zfill(3)}_R*.pxt", root_dir=data_dir)
            files.sort()
        elif len(files) > 1:
            prefix: str = re.match(
                r"(.*?)_" + str(num).zfill(3) + r"(?:_S\d{3})?.pxt", files[0]
            ).group(1)

            motor_file = os.path.join(
                data_dir, f"{prefix}_{str(num).zfill(3)}_Motor_Pos.txt"
            )

            coord_arr = np.loadtxt(motor_file, skiprows=1)
            with open(motor_file) as f:
                header = f.readline().strip().split("\t")

            if coord_arr.ndim == 1:
                coord_arr = coord_arr.reshape(-1, 1)

            for i, hdr in enumerate(header):
                key = self.name_map_reversed.get(hdr, hdr)
                coord_dict[key] = coord_arr[: len(files), i].astype(np.float64)

        if len(files) == 0:
            # Look for single file scan
            files = glob.glob(f"*_{str(num).zfill(3)}.pxt", root_dir=data_dir)

        if len(files) == 0:
            raise FileNotFoundError(f"No files found for scan {num} in {data_dir}")

        files = [os.path.join(data_dir, f) for f in files]

        return files, coord_dict

    def infer_index(self, name: str) -> int | None:
        try:
            scan_num: str = re.match(r".*?(\d{3})(?:_S\d{3})?", name).group(1)
        except (AttributeError, IndexError):
            return None

        if scan_num.isdigit():
            return int(scan_num)
        else:
            return None

    def post_process(self, data: xr.DataArray) -> xr.DataArray:
        data = super().post_process(data)

        if "eV" in data.coords:
            data = data.assign_coords(eV=-data.eV.values)

        return data

    def load_live(self, filename, data_dir=None):
        wave = load_wave(filename, data_dir)
        wave = wave.rename(
            {
                k: v
                for k, v in {"Energy": "eV", "Scienta": "alpha"}.items()
                if k in wave.dims
            }
        )
        wave = wave.assign_coords(eV=-wave["eV"] + wave.attrs["BL Energy"])

        return self.post_process(wave)

    def generate_summary(
        self, data_dir: str | os.PathLike, exclude_live: bool = False
    ) -> pd.DataFrame:
        files: dict[str, str] = {}

        for pth in erlab.io.utilities.get_files(data_dir, extensions=(".pxt",)):
            data_name = os.path.splitext(os.path.basename(pth))[0]
            name_match = re.match(r"(.*?_\d{3})_(?:_S\d{3})?", data_name)
            if name_match is not None:
                data_name = name_match.group(1)
            files[data_name] = pth

        if not exclude_live:
            for file in os.listdir(data_dir):
                if file.endswith(".ibw"):
                    data_name = os.path.splitext(file)[0]
                    path = os.path.join(data_dir, file)
                    files[data_name] = path

        summary_attrs: dict[str, str] = {
            "Lens Mode": "Lens Mode",
            "Scan Type": "Acquisition Mode",
            "T(K)": "temp_sample",
            "Pass E": "Pass Energy",
            "Analyzer Slit": "Slit Plate",
            "Polarization": "polarization",
            "hv": "hv",
            "Entrance Slit": "Entrance Slit",
            "Exit Slit": "Exit Slit",
            "x": "x",
            "y": "y",
            "z": "z",
            "polar": "beta",
            "tilt": "xi",
            "azi": "delta",
        }

        cols = ["Time", "File Name", "Type", *summary_attrs.keys()]

        data: list[dict] = []
        data_info = []

        for name, path in files.items():
            if os.path.splitext(path)[1] == ".ibw":
                data = self.load_live(path)
                if "beta" in data.dims:
                    data_type = "LP"
                else:
                    data_type = "LXY"
            else:
                data = self.load(path)
                if "beta" in data.dims:
                    data_type = "map"
                if "hv" in data.dims:
                    data_type = "hvdep"

            data_info.append(
                [
                    datetime.datetime.strptime(
                        f"{data.attrs['Date']} {data.attrs['Time']}",
                        "%d/%m/%Y %I:%M:%S %p",
                    ),
                    name,
                    data_type,
                ]
            )

            for k, v in summary_attrs.items():
                try:
                    val = data.attrs[v]
                except KeyError:
                    try:
                        val = data.coords[v].values
                        if val.size == 1:
                            val = val.item()
                    except KeyError:
                        val = ""

                if k == "Lens Mode":
                    val = val.replace("Angular", "A")

                elif k in ("Entrance Slit", "Exit Slit"):
                    val = round(val)

                elif k == "Polarization":
                    if np.iterable(val):
                        val = np.asarray(val).astype(int)
                    else:
                        val = [round(val)]
                    val = [{0: "LH", 2: "LV", -1: "RC", 1: "LC"}.get(v, v) for v in val]

                    if len(val) == 1:
                        val = val[0]

                data_info[-1].append(val)

            del data

        return (
            pd.DataFrame(data_info, columns=cols)
            .sort_values("Time")
            .set_index("File Name")
        )
