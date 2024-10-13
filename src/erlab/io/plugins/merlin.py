"""Data loader for beamline 4.0.3 at ALS."""

import datetime
import glob
import os
import re
import warnings
from collections.abc import Callable
from typing import Any, ClassVar, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

import erlab.io.utils
from erlab.io.dataloader import LoaderBase


def _format_polarization(val) -> str:
    val = round(float(val))
    return {0: "LH", 2: "LV", -1: "RC", 1: "LC"}.get(val, str(val))


class MERLINLoader(LoaderBase):
    name = "merlin"

    aliases = ("ALS_BL4", "als_bl4", "BL403", "bl403")

    name_map: ClassVar[dict] = {
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
        "mesh_current": "Mesh Current",
    }
    coordinate_attrs = (
        "beta",
        "delta",
        "xi",
        "hv",
        "x",
        "y",
        "z",
        "polarization",
        "mesh_current",
        "temp_sample",
    )
    additional_attrs: ClassVar[dict] = {
        "configuration": 1,
        "sample_workfunction": 4.44,
    }
    formatters: ClassVar[dict[str, Callable]] = {
        "polarization": _format_polarization,
        "Lens Mode": lambda x: x.replace("Angular", "A"),
        "Entrance Slit": round,
        "Exit Slit": round,
        "Slit Plate": round,
    }
    always_single = False

    @property
    def file_dialog_methods(self):
        return {"ALS BL4.0.3 Raw Data (*.pxt, *.ibw)": (self.load, {})}

    def load_single(self, file_path: str | os.PathLike) -> xr.DataArray:
        if os.path.splitext(file_path)[1] == ".ibw":
            return self._load_live(file_path)

        # One file always corresponds to single region
        return xr.open_dataarray(file_path, engine="erlab-igor")

    def identify(self, num: int, data_dir: str | os.PathLike):
        coord_dict: dict[str, npt.NDArray[np.float64]] = {}

        # Look for multi-file scans
        files = glob.glob(f"*_{str(num).zfill(3)}_S*.pxt", root_dir=data_dir)
        files.sort()  # this should sort files by scan number

        if len(files) == 0:
            # Look for multiregion scans
            files = glob.glob(f"*_{str(num).zfill(3)}_R*.pxt", root_dir=data_dir)
            files.sort()

        elif len(files) > 1:
            match_prefix = re.match(
                r"(.*?)_" + str(num).zfill(3) + r"(?:_S\d{3})?.pxt", files[0]
            )
            if match_prefix is None:
                raise RuntimeError(f"Failed to determine prefix from {files[0]}")
            prefix: str = match_prefix.group(1)

            motor_file = os.path.join(
                data_dir, f"{prefix}_{str(num).zfill(3)}_Motor_Pos.txt"
            )

            coord_arr = np.loadtxt(motor_file, skiprows=1)
            with open(motor_file, encoding="utf-8") as f:
                header = f.readline().strip().split("\t")  # motor coordinate names

            if coord_arr.ndim == 1:
                coord_arr = coord_arr.reshape(-1, 1)  # ensure 2D

            for i, dim in enumerate(header):
                # Trim coord to number of files
                coord_dict[dim] = coord_arr[: len(files), i]

        if len(files) == 0:
            # Look for single file scan
            files = glob.glob(f"*_{str(num).zfill(3)}.pxt", root_dir=data_dir)

        if len(files) == 0:
            return None

        files = [os.path.join(data_dir, f) for f in files]

        return files, coord_dict

    def infer_index(self, name: str) -> tuple[int | None, dict[str, Any]]:
        try:
            match_scan = re.match(r".*?(\d{3})(?:_S\d{3})?", name)
            if match_scan is None:
                return None, {}

            scan_num: str = match_scan.group(1)
        except IndexError:
            return None, {}

        if scan_num.isdigit():
            return int(scan_num), {}
        return None, {}

    def post_process(self, data: xr.DataArray) -> xr.DataArray:
        data = super().post_process(data)

        if "eV" in data.coords:
            data = data.assign_coords(eV=-data.eV.values)

        if "temp_sample" in data.coords:
            # Add temperature to attributes, for backwards compatibility
            temp = float(data.temp_sample.mean())
            data = data.assign_attrs(temp_sample=temp)

        return data

    def load_live(self, identifier, data_dir):
        warnings.warn(
            "load_live is deprecated, use load instead",
            DeprecationWarning,
            stacklevel=1,
        )
        return self.load(identifier, data_dir)

    def _load_live(self, file_path: str | os.PathLike) -> xr.DataArray:
        wave = xr.load_dataarray(file_path, engine="erlab-igor")
        wave = wave.rename(
            {
                k: v
                for k, v in {"Energy": "eV", "Scienta": "alpha"}.items()
                if k in wave.dims
            }
        )
        return wave.assign_coords(eV=-wave["eV"] + wave.attrs["BL Energy"])

    def generate_summary(
        self, data_dir: str | os.PathLike, exclude_live: bool = False
    ) -> pd.DataFrame:
        files: dict[str, str] = {}

        for path in erlab.io.utils.get_files(data_dir, extensions=(".pxt",)):
            data_name = os.path.splitext(os.path.basename(path))[0]
            name_match = re.match(r"(.*?_\d{3})_(?:_S\d{3})?", data_name)
            if name_match is not None:
                data_name = name_match.group(1)
            files[data_name] = path

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

        cols = ["File Name", "Path", "Time", "Type", *summary_attrs.keys()]

        data_info = []
        processed_indices: list[int] = []

        def _add_darr(dname: str, file: str, darr: xr.DataArray, live: bool = False):
            if live:
                data_type = "LP" if "beta" in darr.dims else "LXY"
            else:
                data_type = "core"
                if "alpha" in darr.dims:
                    data_type = "cut"
                if "beta" in darr.dims:
                    data_type = "map"
                if "hv" in darr.dims:
                    data_type = "hvdep"
            data_info.append(
                [
                    dname,
                    file,
                    datetime.datetime.strptime(
                        f"{darr.attrs['Date']} {darr.attrs['Time']}",
                        "%d/%m/%Y %I:%M:%S %p",
                    ),
                    data_type,
                    *(
                        self.get_formatted_attr_or_coord(darr, k)
                        for k in summary_attrs.values()
                    ),
                ]
            )

        for name, path in files.items():
            if os.path.splitext(path)[1] == ".ibw":
                _add_darr(
                    name, path, darr=cast(xr.DataArray, self.load(path)), live=True
                )
            else:
                idx, _ = self.infer_index(os.path.splitext(os.path.basename(path))[0])
                if idx in processed_indices:
                    continue

                if idx is not None:
                    processed_indices.append(idx)

                data = cast(xr.DataArray | xr.Dataset, self.load(path))

                if isinstance(data, xr.Dataset):
                    for k, darr in data.data_vars.items():
                        _add_darr(f"{name}_{k}", path, darr)
                else:
                    _add_darr(name, path, data)
                del data

        return (
            pd.DataFrame(data_info, columns=cols)
            .sort_values("Time")
            .set_index("File Name")
        )
