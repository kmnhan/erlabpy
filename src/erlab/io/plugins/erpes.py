"""Data loader for our homelab system."""

__all__ = ["ERPESLoader"]

import datetime
import os
import pathlib
import re
import typing
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import xarray as xr

import erlab
from erlab.io.plugins.da30 import DA30Loader


def _determine_kind(data: xr.DataArray) -> str:
    data_type = "cut"
    if "beta" in data.dims:
        data_type = "map"

    misc_dims = set(data.dims) - {"alpha", "beta", "eV"}
    if len(misc_dims) > 0:
        misc_dims_str = ", ".join(sorted(str(d) for d in misc_dims))
        data_type = f"{misc_dims_str} {data_type}"
    return data_type


def _get_start_time(data: xr.DataArray) -> datetime.datetime:
    return xr.apply_ufunc(
        lambda x, y: datetime.datetime.fromisoformat(f"{x} {y}"),
        data["Date"],
        data["Time"],
        vectorize=True,
    )


def _get_seq_start(data: xr.DataArray) -> datetime.datetime:
    return xr.apply_ufunc(
        lambda x: datetime.datetime.fromisoformat(x), data["seq_start"], vectorize=True
    )


def _get_attrs_time(data: xr.DataArray) -> datetime.datetime:
    return xr.apply_ufunc(
        lambda x: datetime.datetime.fromisoformat(x), data["attrs_time"], vectorize=True
    )


def _emit_ambiguous_file_warning(num, file_to_use):
    erlab.utils.misc.emit_user_level_warning(
        f"Multiple files found for scan {num}, using {file_to_use}. "
        "Try providing the `prefix` argument to specify."
    )


class ERPESLoader(DA30Loader):
    name = "erpes"
    description = "KAIST home lab setup"
    extensions: typing.ClassVar[set[str]] = {".pxt", ".zip", ""}

    name_map: typing.ClassVar[dict] = {
        "eV": ["Kinetic Energy [eV]", "Energy [eV]"],
        "alpha": ["Y-Scale [deg]", "Thetax [deg]"],
        "beta": ["Thetay [deg]", "ThetaY"],
        "hv": ["BL Energy", "Excitation Energy"],
        "sample_temp": "TB",
    }
    coordinate_attrs: tuple[str, ...] = (
        "beta",
        "chi",
        # "xi",
        "hv",
        "sample_temp",
        "TA",
        "TC",
        "TD",
        "T1",
        "T2",
        "T3",
        "T4",
        "T5",
        "T6",
        "T7",
        "T8",
        "T0",
        "ch1",
        "ch2",
        "ch3",
        "ch4",
        "ch5",
        "ch6",
        "torr_main",
        "torr_middle",
        "torr_loadlock",
        "seq_start",
        "attrs_time",
        "Date",  # Promote Date and Time
        "Time",  # Convert to single datetime in additional_coords
    )

    additional_attrs: typing.ClassVar[dict] = {"configuration": 4}

    additional_coords: typing.ClassVar[dict] = {
        "hv": 6.0187,
        "datetime": _get_start_time,
        "seq_start": _get_seq_start,
        "attrs_time": _get_attrs_time,
    }

    overridden_coords: tuple[str, ...] = (
        "seq_start",
        "attrs_time",
    )

    summary_attrs: typing.ClassVar[
        dict[str, str | Callable[[xr.DataArray], typing.Any]]
    ] = {
        "time": _get_start_time,
        "type": _determine_kind,
        "lens mode": "Lens Mode",
        "mode": "Acquisition Mode",
        "pass energy": "Pass Energy",
        "analyzer slit": "slit_number",
        "temperature": "sample_temp",
        "coldfinger": "TA",
        "beta": "beta",
        "ch1": "ch1",
        "ch2": "ch2",
        "ch3": "ch3",
        "ch4": "ch4",
        "ch5": "ch5",
        "ch6": "ch6",
    }
    summary_sort = "time"

    always_single = False

    _PATTERN_MULTIFILE = re.compile(r".*\d{4}_S\d{5}.(pxt|zip)")
    _PATTERN_PREFIX = re.compile(r"(.*?)\d{4}(?:_S\d{5})?.(pxt|zip)")
    _PATTERN_FILENO = re.compile(r".*?(\d{4})(?:_S\d{5})?")
    _PATTERN_PREFIX_FILENO = re.compile(r"(.*?)(\d{4})(?:_S\d{5})?")

    @property
    def file_dialog_methods(self) -> dict[str, tuple[Callable, dict[str, typing.Any]]]:
        return {
            "1KARPES Data (*.pxt *.zip)": (self.load, {}),
            "1KARPES Single File (*.pxt *.zip)": (self.load, {"single": True}),
        }

    def identify(
        self, num: int, data_dir: str | os.PathLike, prefix: str | None = None
    ) -> tuple[list[pathlib.Path], dict[str, npt.NDArray]] | None:
        prefix_pattern = r".*" if prefix is None else re.escape(prefix)

        coord_dict: dict[str, npt.NDArray] = {}

        data_dir = pathlib.Path(data_dir)

        all_files: set[pathlib.Path] = erlab.io.utils.get_files(
            data_dir, extensions=(".pxt", ".zip")
        )  # All data with valid extensions

        # Look for multifile scans
        pattern_multifile_n = re.compile(
            prefix_pattern + str(num).zfill(4) + r"_S\d{5}.(pxt|zip)"
        )
        files: list[pathlib.Path] = [
            f for f in all_files if pattern_multifile_n.match(f.name)
        ]
        files.sort()  # Now, safe to assume files sorted by scan #

        if len(files) >= 1:
            # Found files from multifile scan
            if prefix is None:
                prefixes: set[str] = set()
                for i, file in enumerate(files):
                    match_prefix = self._PATTERN_PREFIX.match(file.name)
                    if match_prefix is not None:
                        # This should never be None, but mypy doesn't know that
                        this_prefix = match_prefix.group(1)
                        prefixes.add(this_prefix)
                        if i > 0 and this_prefix not in prefixes:
                            erlab.utils.misc.emit_user_level_warning(
                                f"Multiple prefixes found for scan {num}, "
                                f"using {next(iter(prefixes))}. "
                                "Provide `prefix` argument to specify."
                            )
                            break
                prefix = next(iter(prefixes))

            motor_file = data_dir / f"{prefix}{str(num).zfill(4)}_motors.csv"

            # Load the coordinates from the csv file
            coord_arr = np.loadtxt(motor_file, delimiter=",", skiprows=1)

            with open(motor_file) as f:
                header = f.readline().strip().split(",")

            if coord_arr.ndim <= 1:
                coord_arr = coord_arr.reshape(1, -1)  # ensure 2D

            if len(files) > coord_arr.shape[0]:
                raise RuntimeError(
                    f"Number of motor positions ({coord_arr.shape[0]}) "
                    f"does not match number of files ({len(files)})"
                )

            # Each header entry will contain a dimension name
            for i, dim in enumerate(header[1:]):
                coord_dict[dim] = coord_arr[: len(files), i + 1].astype(np.float64)

        if (
            len(files) == 0 or prefix is None
        ):  # If prefix not given, need to check for ambiguity
            pattern_singlefile = re.compile(
                prefix_pattern + str(num).zfill(4) + r".(pxt|zip)"
            )
            # Look for single file scan
            files_single = [
                f
                for f in all_files
                if pattern_singlefile.match(f.name)
                and not self._PATTERN_MULTIFILE.match(f.name)
            ]
            files_single.sort()

            if len(files) != 0:
                if len(files_single) != 0:
                    _emit_ambiguous_file_warning(num, files[0])
                return files, coord_dict

            files = files_single

            single_scan_ambiguous: bool = (len(files) > 1) and not (
                len(files) == 2 and files[0].stem == files[1].stem
            )  # Latter: DA maps and cuts within single region
            if single_scan_ambiguous:
                _emit_ambiguous_file_warning(num, files[0])
                files = files[:1]

        if len(files) == 0:
            return None

        return files, coord_dict

    def infer_index(self, name: str) -> tuple[int | None, dict[str, typing.Any]]:
        try:
            match_scan = self._PATTERN_PREFIX_FILENO.match(name)
            if match_scan is None:
                return None, {}

            prefix: str = match_scan.group(1)
            scan_num: str = match_scan.group(2)
        except IndexError:
            return None, {}

        if scan_num.isdigit():
            return int(scan_num), {"prefix": prefix}
        return None, {}

    def post_process(self, data: xr.DataArray) -> xr.DataArray:
        return super().post_process(data).drop_vars(["Date", "Time"], errors="ignore")
