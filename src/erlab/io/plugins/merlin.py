"""Data loader for beamline 4.0.3 at ALS."""

__all__ = ["MERLINLoader", "load_bcs"]

import csv
import datetime
import json
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

_BCSHeader = dict[str, typing.Any]
_BCSRow = dict[str, str]
_FloatArray = npt.NDArray[np.float64]


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


def _unique_name(name: str, existing: set[str]) -> str:
    if name not in existing:
        existing.add(name)
        return name

    candidate = f"{name} raw"
    if candidate not in existing:
        existing.add(candidate)
        return candidate

    index = 1
    while f"{candidate} {index}" in existing:
        index += 1

    unique = f"{candidate} {index}"
    existing.add(unique)
    return unique


def load_bcs(path: str | os.PathLike) -> xr.DataArray | xr.DataTree:
    """Load a beamline control system scan.

    Parameters
    ----------
    path
        Path to the BCS text file. The image files referenced by the data table are
        resolved relative to the text file and, when needed, from a sibling directory
        named ``"<text file stem> Images"``.

    Returns
    -------
    xarray.DataArray or xarray.DataTree
        A data array containing the compiled image stack for BCS files with one image
        column. If a file contains multiple image columns, each image stream is loaded
        into a separate child node of a data tree.
    """
    path = pathlib.Path(path)

    lines = path.read_text(encoding="utf-8-sig").splitlines()
    try:
        header_start = next(
            i for i, line in enumerate(lines) if line.strip() == "HEADER"
        )
        data_start = next(i for i, line in enumerate(lines) if line.strip() == "DATA")
    except StopIteration as err:
        raise ValueError(f"{path} is not a valid BCS data file") from err

    if data_start <= header_start or data_start + 1 >= len(lines):
        raise ValueError(f"{path} is missing BCS data table")

    header = typing.cast(
        "_BCSHeader",
        json.loads("\n".join(lines[header_start + 1 : data_start]).strip()),
    )
    reader = csv.DictReader(lines[data_start + 1 :], delimiter="\t")
    columns = list(reader.fieldnames or ())
    rows = typing.cast("list[_BCSRow]", list(reader))
    if not rows:
        raise ValueError(f"{path} contains no BCS data rows")

    image_columns = [
        column
        for column in columns
        if any(str(row.get(column, "")).lower().endswith(".png") for row in rows)
    ]
    if not image_columns:
        raise ValueError(f"{path} contains no BCS image columns")

    numeric_columns: dict[str, _FloatArray] = {}
    for column in columns:
        if column in image_columns:
            continue

        values: list[float] = []
        for row in rows:
            value = row.get(column)
            if value is None or value == "":
                break
            try:
                values.append(float(value))
            except ValueError:
                break
        else:
            numeric_columns[column] = np.asarray(values, dtype=np.float64)

    scan_type = header.get("Scan Type", "")
    scan_info = header.get(scan_type, {})
    scan_motor = scan_info.get("X Motor") if isinstance(scan_info, dict) else None
    scan_dim = str(scan_motor) if scan_motor else "step"
    goal_column: str | None = f"{scan_dim} Goal"
    if goal_column in numeric_columns:
        scan_values = numeric_columns[goal_column]
    elif scan_dim in numeric_columns:
        scan_values = numeric_columns[scan_dim]
        goal_column = scan_dim
    else:
        scan_values = np.arange(len(rows), dtype=np.float64)
        goal_column = None

    constant_values: dict[str, float] = {}
    varying_columns: dict[str, _FloatArray] = {}
    for column, column_values in numeric_columns.items():
        if column == goal_column:
            continue
        if np.all(column_values == column_values[0]):
            constant_values[column] = float(column_values[0])
        else:
            varying_columns[column] = column_values

    try:
        from PIL import Image
    except ImportError as err:
        raise ImportError(
            "PIL is required to load BCS files with image columns. "
            "Please install `erlab[io]` or Pillow to use this feature."
        ) from err

    data: dict[str, xr.DataArray] = {}
    for image_column in image_columns:
        images: list[npt.NDArray[np.generic]] = []
        for row in rows:
            image_ref = row[image_column]
            normalized_ref = pathlib.Path(image_ref.strip().replace("\\", "/"))
            if normalized_ref.is_absolute():
                candidates: tuple[pathlib.Path, ...] = (normalized_ref,)
            else:
                candidates = (
                    path.parent / normalized_ref,
                    path.parent / f"{path.stem} Images" / normalized_ref.name,
                )

            image_path = next(
                (candidate for candidate in candidates if candidate.is_file()), None
            )
            if image_path is None:
                raise FileNotFoundError(
                    f"Could not find BCS image {image_ref!r} for {path.name}"
                )

            with Image.open(image_path) as image:
                images.append(np.asarray(image).copy())

        try:
            image_stack = np.stack(images, axis=0)
        except ValueError as err:
            raise ValueError("All BCS images must have the same shape") from err

        image_shape = image_stack.shape[1:]
        if len(image_shape) == 2:
            dims: tuple[str, ...] = (scan_dim, "y", "x")
        elif len(image_shape) == 3:
            dims = (scan_dim, "y", "x", "channel")
        else:
            raise ValueError("BCS images must be two-dimensional or RGB/RGBA images")

        coords: dict[str, typing.Any] = {
            scan_dim: scan_values,
            "y": np.arange(image_shape[0]),
            "x": np.arange(image_shape[1]),
        }
        existing_coords = set(coords)
        if len(image_shape) == 3:
            coords["channel"] = np.arange(image_shape[2])
            existing_coords.add("channel")

        for column, coord_values in varying_columns.items():
            coords[_unique_name(column, existing_coords)] = (scan_dim, coord_values)

        for column, coord_values in constant_values.items():
            coords[_unique_name(column, existing_coords)] = coord_values

        motors = header.get("Motors", {})
        if isinstance(motors, dict):
            for motor_name, motor_value in motors.items():
                if motor_name not in existing_coords:
                    coords[motor_name] = motor_value
                    existing_coords.add(motor_name)

        attrs: dict[str, typing.Any] = {}
        existing_attrs = set(attrs)
        for attr_name, attr_value in header.items():
            if attr_name in {"General", "Motors"}:
                continue
            if attr_name not in existing_coords:
                attrs[_unique_name(attr_name, existing_attrs)] = attr_value

        data[image_column] = xr.DataArray(
            image_stack,
            dims=dims,
            coords=coords,
            name=image_column,
            attrs=attrs,
        )

    if len(image_columns) == 1:
        return data[image_columns[0]]

    existing: set[str] = set()
    return xr.DataTree.from_dict(
        {
            _unique_name(image_column.replace("/", "_"), existing): darr.to_dataset()
            for image_column, darr in data.items()
        }
    )


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
            "ALS BL4.0.3 BCS Data (*.txt)": (load_bcs, {}),
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

    def _parse_motor_file(
        self, data_dir: pathlib.Path, prefix: str, num: int
    ) -> tuple[list[str], npt.NDArray[np.float64]]:
        motor_file = data_dir / f"{prefix}_{str(num).zfill(3)}_Motor_Pos.txt"

        # Load as string first to avoid issues with empty lines and trailing spaces
        coord_arr = np.loadtxt(motor_file, dtype=str, skiprows=1).astype(np.float64)

        with motor_file.open(encoding="utf-8") as f:
            header = f.readline().strip().split("\t")  # motor coordinate names

        if coord_arr.ndim <= 1:
            coord_arr = coord_arr.reshape(-1, 1)  # ensure 2D

        return header, coord_arr

    def _get_prefix(self, file_name: str, num: int) -> str:
        match_prefix = re.match(
            r"(.*?)_" + str(num).zfill(3) + r"(_R\d)?(?:_S\d{3})?.pxt", file_name
        )
        if match_prefix is None:
            raise RuntimeError(f"Failed to determine prefix from {file_name}")
        return match_prefix.group(1)

    def identify(self, num: int, data_dir: str | os.PathLike):
        data_dir = pathlib.Path(data_dir)

        coord_dict: dict[str, npt.NDArray[np.float64]] = {}

        # Look for multi-file scan
        files = sorted(data_dir.glob(f"*_{str(num).zfill(3)}_S*.pxt"))

        if len(files) == 0:
            # Look for multiregion AND multi-file scan, like f_001_R0_S001.pxt
            files = sorted(data_dir.glob(f"*_{str(num).zfill(3)}_R*_S*.pxt"))

            if len(files) != 0:
                # Parse motor file for multiregion multi-file scan
                prefix = self._get_prefix(files[0].name, num)
                header, coord_arr = self._parse_motor_file(data_dir, prefix, num)

                prefix = re.escape(prefix)

                region_list: list[int] = []
                coords_tmp: dict[str, list[float]] = {dim: [] for dim in header}

                for f in files:
                    match_r_s = re.match(
                        f"{prefix}_" + str(num).zfill(3) + r"_R(\d)_S(\d{3}).pxt",
                        f.name,
                    )
                    if match_r_s is None:
                        raise RuntimeError(f"Failed to parse file name {f.name}")

                    region = int(match_r_s.group(1))
                    scan_idx = int(match_r_s.group(2)) - 1

                    region_list.append(region)
                    for i, dim in enumerate(header):
                        coords_tmp[dim].append(coord_arr[scan_idx, i])

                for dim in header:
                    coord_dict[dim] = np.array(coords_tmp[dim])
                coord_dict["__region"] = np.array(region_list)

            else:
                # Look for multiregion scan like f_001_R0.pxt
                files = sorted(data_dir.glob(f"*_{str(num).zfill(3)}_R*.pxt"))
                if len(files) != 0:
                    region_numbers = []
                    for f in files:
                        match_r = re.match(
                            rf".*?_{str(num).zfill(3)}_R(\d).pxt", f.name
                        )
                        if match_r is None:
                            raise RuntimeError(f"Failed to parse file name {f.name}")
                        region_numbers.append(int(match_r.group(1)))
                    coord_dict["__region"] = np.array(region_numbers)

        elif len(files) > 1:
            # Extract motor positions for multi-file scan
            prefix = self._get_prefix(files[0].name, num)
            header, coord_arr = self._parse_motor_file(data_dir, prefix, num)

            if len(files) > coord_arr.shape[0]:
                if header == ["Fake Motor"] and coord_arr.shape[0] > 0:
                    # Allow incomplete motor scan files
                    coord_arr = np.arange(len(files)).reshape(-1, 1) + coord_arr[0, 0]
                elif len(header) == 1 and coord_arr.shape[0] > 1:
                    start = coord_arr[0, 0]
                    step = coord_arr[1, 0] - start
                    erlab.utils.misc.emit_user_level_warning(
                        f"Number of motor positions ({coord_arr.shape[0]}) "
                        f"less than the number of files ({len(files)}). "
                        f"Assuming step size of {step} to generate motor positions."
                    )
                    coord_arr = np.arange(len(files)).reshape(-1, 1) * step + start
                else:
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
            match_scan = re.match(r".*?(\d{3})(_R\d)?(?:_S\d{3})?", name)
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
            data = self._fix_energy_axis(data)

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
        return wave.assign_coords(eV=-wave["eV"])

    def files_for_summary(self, data_dir: str | os.PathLike):
        return sorted(erlab.io.utils.get_files(data_dir, extensions=(".pxt", ".ibw")))

    @staticmethod
    def _fix_energy_axis(data: xr.DataArray) -> xr.DataArray:
        """Fix the energy axis by taking the energy step from the attributes."""
        step = data.attrs.get("Energy Step")
        if step and data.attrs.get("scan_type") != "live":
            old_eV = data.eV.values
            return data.assign_coords(
                eV=data.eV.copy(
                    data=np.linspace(
                        -old_eV[0], -old_eV[0] + step * (len(old_eV) - 1), len(old_eV)
                    )
                )
            )
        return data.assign_coords(eV=-data.eV)

    def pre_combine_multiple(
        self, data_list, coord_dict
    ) -> tuple[list[xr.DataArray] | list[xr.DataTree], dict]:
        if "__region" in coord_dict:
            # Group data with same scan index into single DataTree
            regions = coord_dict.pop("__region")

            unique_regions = np.unique(regions)

            # Group data by region to set equal energy axis for each region
            data_for_region: list[list[xr.DataArray]] = [
                [
                    data
                    for r, data in zip(regions, data_list, strict=True)
                    if r == region
                ]
                for region in unique_regions
            ]
            data_for_region = [
                self.pre_combine_multiple(d, {})[0] for d in data_for_region
            ]

            # Combine regions that correspond to same scan index into single DataTree
            scan_length = len(data_list) // len(unique_regions)
            combined_list = [
                xr.DataTree.from_dict(
                    {
                        f"R{int(region)}": data_for_region[n][scan_idx].to_dataset()
                        for n, region in enumerate(unique_regions)
                    }
                )
                for scan_idx in range(scan_length)
            ]
            coord_dict = {k: v[:scan_length] for k, v in coord_dict.items()}
            return combined_list, coord_dict

        # Energy start & step may have very small offsets on the order of 1e-6 eV, which
        # can cause issues with merging. We just assume that the energy axis is the same
        # for all data arrays in the list, so we can assign the first one to all of
        # them. This is also the behavior when using `Assemble` from the LoadSESb GUI in
        # Igor Pro.
        return [
            d.assign_coords(eV=d.eV.copy(data=data_list[0].eV.values))
            for d in data_list
        ], coord_dict
