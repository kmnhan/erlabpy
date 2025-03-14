"""Data loader for beamline 7.0.2 at ALS.

.. note::

    This loader assumes vertical analyzer slit orientation and deflector mapping. For
    other configurations, the ``configuration`` attribute and coordinate names must be
    changed accordingly.

"""

__all__ = ["MAESTROMicroLoader"]

import datetime
import os
import re
import typing
from collections.abc import Callable
from pathlib import Path

import numpy as np
import xarray as xr

import erlab
from erlab.io.dataloader import LoaderBase

if typing.TYPE_CHECKING:
    from collections.abc import Hashable


def get_cache_file(file_path: str | os.PathLike) -> Path:
    file_path = Path(file_path)
    data_dir = file_path.parent
    cache_dir = data_dir.with_name(f".{data_dir.name}_cache")
    return cache_dir.joinpath(file_path.stem + "_2D_Data" + file_path.suffix)


def cache_as_float32(file_path: str | os.PathLike, data: xr.Dataset) -> xr.DataArray:
    """Cache and return the 2D part of the data as a float32 DataArray.

    If the cache file exists, it is loaded and returned.

    Loading MAESTRO `.h5` files is slow because the data is stored in double precision.
    This function caches the 2D Data part in float32 to speed up subsequent loading.

    Caching is disabled in read-only file systems.

    """
    cache_file = get_cache_file(file_path)

    if cache_file.is_file():
        return xr.open_dataarray(cache_file, engine="h5netcdf")

    writable: bool = os.access(cache_file.parent.parent, os.W_OK)

    if writable and not cache_file.parent.is_dir():
        cache_file.parent.mkdir(parents=True)

    if len(data.data_vars) > 1:
        erlab.utils.misc.emit_user_level_warning(
            "More than one data variable is present in the data."
            "Only the first one will be used"
        )

    # Get the first data variable
    data = data[next(iter(data.data_vars))]

    if writable:
        # Save cache
        data = data.astype(np.float32)
        data.to_netcdf(cache_file, engine="h5netcdf")

    return data


class MAESTROMicroLoader(LoaderBase):
    name = "maestro"
    description = "ALS Beamline 7.0.2.1 MAESTRO"
    extensions: typing.ClassVar[set[str]] = {".h5"}

    aliases = ("ALS_BL7", "als_bl7", "BL702", "bl702")

    name_map: typing.ClassVar[dict] = {
        "x": "Motors_Logical.X",
        "y": "Motors_Logical.Y",
        "z": "Motors_Logical.Z",
        "chi": "Motors_Logical.Theta",  # polar
        "xi": "Motors_Logical.Beta",  # tilt
        "delta": "Motors_Logical.Phi",  # azimuth
        "beta": ("Slit Defl", "Motors_Logical.Slit Defl"),
        "hv": ("MONOEV", "Beamline.Beamline Energy"),
        "sample_temp": "Cryostat_A",  # A/B/C/D is Sample/Shield/SupportTube/Coldtip
    }
    coordinate_attrs = (
        "beta",
        "delta",
        "hv",
        "sample_temp",
        "chi",
        "xi",
        "x",
        "y",
        "z",
    )
    additional_attrs: typing.ClassVar[dict] = {}

    skip_validate: bool = True
    always_single: bool = True

    formatters: typing.ClassVar[dict[str, Callable]] = {
        "Main.START_T": lambda x: datetime.datetime.strptime(x, "%m/%d/%Y %I:%M:%S %p"),
        "scan_type": lambda x: "" if x == "None" else x,
        "DAQ_Swept.lens mode name": lambda x: x.replace("Angular", "A"),
    }

    summary_attrs: typing.ClassVar[
        dict[str, str | Callable[[xr.DataArray], typing.Any]]
    ] = {
        "time": "Main.START_T",
        "type": "scan_type",
        "pre": "pre_scan",
        "post": "post_scan",
        "lens mode": "DAQ_Swept.lens mode name",
        "region": "DAQ_Swept.SS Region name",
        "temperature": "sample_temp",
        "pass energy": "DAQ_Swept.pass energy",
        "analyzer slit": "DAQ_Swept.Electron Spectrometer  Entrance Slit",
        "pol": "Beamline.EPU Polarization",
        "hv": "hv",
        "polar": "chi",
        "tilt": "xi",
        "azi": "delta",
        "deflector": "beta",
        "x": "x",
        "y": "y",
        "z": "z",
    }

    @property
    def file_dialog_methods(self):
        return {"ALS BL7.0.2 Raw Data (*.h5)": (self.load, {})}

    def identify(self, num, data_dir):
        pattern = re.compile(rf"(\d+)_{str(num).zfill(5)}.h5")
        matches = [
            path
            for path in erlab.io.utils.get_files(data_dir, ".h5")
            if pattern.match(path.name)
        ]
        return matches, {}

    def load_single(self, file_path, without_values: bool = False) -> xr.DataArray:
        groups = xr.open_groups(file_path, engine="h5netcdf", phony_dims="sort")

        if "PreScan" in groups["/Comments"]:
            pre_scan: str = groups["/Comments"]["PreScan"].item()[0].decode()
        else:
            pre_scan = ""

        if "PostScan" in groups["/Comments"]:
            post_scan: str = groups["/Comments"]["PostScan"].item()[0].decode()
        else:
            post_scan = ""

        def _parse_attr(v) -> str | int | float:
            """Strip quotes and convert numerical strings to int or float."""
            v = v.strip().decode()
            try:
                v = float(v)
                if v.is_integer():
                    return int(v)
            except ValueError:
                if v.startswith("'") and v.endswith("'"):
                    return v[1:-1]
                return v
            else:
                return v

        nested_attrs: dict[Hashable, dict[str, tuple[str, str | int | float]]] = {}
        for key, val in groups["/Headers"].data_vars.items():
            # v given as (longname, name, value, comment)
            # we want to extract the name, comment and value
            nested_attrs[key] = {
                str(_parse_attr(v[1])): (str(_parse_attr(v[3])), _parse_attr(v[2]))
                for v in val.values
            }

        human_readable_attrs: dict[str, str | int | float] = {}
        # Final attributes are stored here
        # Keys are in the form "group_name.commment"

        for group_name, contents_dict in nested_attrs.items():
            for k, v in contents_dict.items():
                new_key = f"{group_name}.{k}" if v[0] == "" else f"{group_name}.{v[0]}"
                human_readable_attrs[new_key] = v[1]

        scan_attrs: dict[str, str | int | float] = {
            k: v[1] for k, v in nested_attrs.get("Low_Level_Scan", {}).items()
        }

        if "LWLVNM" in scan_attrs:
            scan_type: str = str(scan_attrs["LWLVNM"])

            lwlvlpn = int(scan_attrs["LWLVLPN"])  # number of low level loops
            motors: list[str] = []
            motor_shape: list[int] = []
            for i in range(lwlvlpn):
                nmsbdv = int(
                    scan_attrs[f"NMSBDV{i}"]
                )  # number of subdevices in i-th loop

                for j in range(nmsbdv):
                    nm = str(
                        scan_attrs[f"NM_{i}_{j}"]
                    )  # name of j-th subdevice in i-th loop
                    nm = nm.replace("CRYO-", "").strip()
                    motors.append(nm)
                    motor_shape.append(int(scan_attrs[f"N_{i}_{j}"]))

        else:
            scan_type = "unknown"
            motors = ["XY"]

        # Get coords
        coords = groups["/0D_Data"].rename({"phony_dim_0": "phony_dim_3"})
        if len(motors) == 1:
            coords = coords.swap_dims({"phony_dim_3": motors[0]})

        if without_values:
            data = groups["/2D_Data"]
            data = data[next(iter(data.data_vars))]
            data = xr.DataArray(
                np.zeros(data.shape, dtype=np.uint8),
                dims=data.dims,
                attrs=data.attrs,
                name=data.name,
            )
        else:
            # Create or load cache
            data = cache_as_float32(file_path, groups["/2D_Data"])

        coord_dict = {
            name: np.linspace(offset, offset + (size - 1) * delta, size)
            for delta, offset, size, name in zip(
                reversed(data.attrs["scaleDelta"]),
                reversed(data.attrs["scaleOffset"]),
                data.shape,
                reversed(data.attrs["unitNames"]),
                strict=False,
            )
        }

        data = data.rename(
            {f"phony_dim_{i + 1}": k for i, k in enumerate(coord_dict.keys())}
        ).assign_coords(coord_dict)

        if len(motors) == 1:
            data = data.rename(phony_dim_3=motors[0]).assign_coords(coords)
        else:
            # Just keep phony dims for now
            data = data.assign_coords(coords)  # .set_xindex(motors)
            # data = data.set_index(phony_dim_3=motors)
            # data = data.unstack("phony_dim_3")

        # The configuration is hardcoded to 3, which is for vertical analyzer slit with
        # deflector map. For horizontal slit configuration and/or tilt/polar maps,
        # coordinates and the attribute must be changed accordingly.
        data.attrs = {
            "scan_type": scan_type,
            "pre_scan": pre_scan,
            "post_scan": post_scan,
            "configuration": 3,
        }
        return data.assign_attrs(human_readable_attrs).squeeze()

    def files_for_summary(self, data_dir):
        return sorted(erlab.io.utils.get_files(data_dir, extensions=(".h5",)))
