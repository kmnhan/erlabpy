"""Data loader for beamline 4.0.3 at ALS."""

__all__ = ["MERLINLoader"]

import datetime
import os
import pathlib
import re
import typing
import warnings
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import xarray as xr

import erlab
from erlab.io.dataloader import LoaderBase


def _format_polarization(val) -> str:
    val = round(float(val))
    return {0: "LH", 2: "LV", -1: "RC", 1: "LC"}.get(val, str(val))


def _parse_time(darr: xr.DataArray) -> datetime.datetime:
    return datetime.datetime.strptime(
        f"{darr.attrs['Date']} {darr.attrs['Time']}", "%d/%m/%Y %I:%M:%S %p"
    )


def _determine_kind(data: xr.DataArray) -> str:
    if "scan_type" in data.attrs and data.attrs["scan_type"] == "live":
        return "LP" if "beta" in data.dims else "LXY"

    data_type = "xps"
    if "alpha" in data.dims:
        data_type = "cut"
    if "beta" in data.dims:
        data_type = "map"
    if "hv" in data.dims:
        data_type = "hvdep"
    return data_type


class MERLINLoader(LoaderBase):
    name = "merlin"
    description = "ALS Beamline 4.0.3 MERLIN"
    extensions: typing.ClassVar[set[str]] = {".pxt", ".ibw"}

    aliases = ("ALS_BL4", "als_bl4", "BL403", "bl403")

    name_map: typing.ClassVar[dict] = {
        "alpha": "deg",
        "beta": ["Polar", "Polar Compens"],
        "delta": "Azimuth",
        "xi": "Tilt",
        "x": "Sample X",
        "y": "Sample Y (Vert)",
        "z": "Sample Z",
        "hv": "BL Energy",
        "polarization": "EPU POL",
        "sample_temp": "Temperature Sensor B",
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
        "sample_temp",
    )
    additional_attrs: typing.ClassVar[dict] = {"configuration": 1}

    formatters: typing.ClassVar[dict[str, Callable]] = {
        "polarization": _format_polarization,
        "Lens Mode": lambda x: x.replace("Angular", "A"),
        "Entrance Slit": round,
        "Exit Slit": round,
        "Slit Plate": round,
    }

    summary_attrs: typing.ClassVar[
        dict[str, str | Callable[[xr.DataArray], typing.Any]]
    ] = {
        "time": _parse_time,
        "type": _determine_kind,
        "lens mode": "Lens Mode",
        "mode": "Acquisition Mode",
        "temperature": "sample_temp",
        "pass energy": "Pass Energy",
        "analyzer slit": "Slit Plate",
        "pol": "polarization",
        "hv": "hv",
        "entrance slit": "Entrance Slit",
        "exit slit": "Exit Slit",
        "polar": "beta",
        "tilt": "xi",
        "azi": "delta",
        "x": "x",
        "y": "y",
        "z": "z",
    }

    summary_sort = "time"

    always_single = False

    @property
    def file_dialog_methods(self):
        return {
            "ALS BL4.0.3 Data (*.pxt *.ibw)": (self.load, {}),
            "ALS BL4.0.3 Single File (*.pxt)": (self.load, {"single": True}),
        }

    def load_single(
        self, file_path: str | os.PathLike, without_values: bool = False
    ) -> xr.DataArray:
        file_path = pathlib.Path(file_path)

        if file_path.suffix == ".ibw":
            return self._load_live(file_path)

        # One file always corresponds to single region
        return xr.open_dataarray(file_path, engine="erlab-igor")

    def identify(self, num: int, data_dir: str | os.PathLike):
        data_dir = pathlib.Path(data_dir)

        coord_dict: dict[str, npt.NDArray[np.float64]] = {}

        # Look for multi-file scan
        files = sorted(data_dir.glob(f"*_{str(num).zfill(3)}_S*.pxt"))

        if len(files) == 0:
            # Look for multiregion scan
            files = sorted(data_dir.glob(f"*_{str(num).zfill(3)}_R*.pxt"))

        elif len(files) > 1:
            # Extract motor positions for multi-file scan
            match_prefix = re.match(
                r"(.*?)_" + str(num).zfill(3) + r"(?:_S\d{3})?.pxt", files[0].name
            )
            if match_prefix is None:
                raise RuntimeError(f"Failed to determine prefix from {files[0]}")
            prefix: str = match_prefix.group(1)

            motor_file = data_dir / f"{prefix}_{str(num).zfill(3)}_Motor_Pos.txt"

            coord_arr = np.loadtxt(motor_file, skiprows=1)

            with motor_file.open(encoding="utf-8") as f:
                header = f.readline().strip().split("\t")  # motor coordinate names

            if coord_arr.ndim <= 1:
                coord_arr = coord_arr.reshape(-1, 1)  # ensure 2D

            if len(files) > coord_arr.shape[0]:
                raise RuntimeError(
                    f"Number of motor positions ({coord_arr.shape[0]}) "
                    f"does not match number of files ({len(files)})"
                )

            for i, dim in enumerate(header):
                # Trim coord to number of files
                coord_dict[dim] = coord_arr[: len(files), i]

        if len(files) == 0:
            # Look for single file scan
            files = sorted(data_dir.glob(f"*_{str(num).zfill(3)}.pxt"))

            # If there is more than one file found, this indicates a conflict
            if len(files) > 1:
                erlab.utils.misc.emit_user_level_warning(
                    f"Multiple files found for scan {num}, using {files[0]}"
                )
                files = files[:1]

        if len(files) == 0:
            return None

        return files, coord_dict

    def infer_index(self, name: str) -> tuple[int | None, dict[str, typing.Any]]:
        try:
            match_scan = re.match(r".*?(\d{3})(?:_S\d{3})", name)
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

        return data

    def load_live(self, identifier, data_dir):
        warnings.warn(
            "load_live is deprecated, use load instead", FutureWarning, stacklevel=1
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
        wave = wave.assign_attrs(scan_type="live")
        return wave.assign_coords(eV=-wave["eV"] + wave.attrs["BL Energy"])

    def files_for_summary(self, data_dir: str | os.PathLike):
        return sorted(erlab.io.utils.get_files(data_dir, extensions=(".pxt", ".ibw")))
