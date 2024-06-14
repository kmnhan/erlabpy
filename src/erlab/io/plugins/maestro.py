"""Data loader for beamline 7.0.2 at ALS.

.. note::

    This loader assumes vertical analyzer slit orientation and deflector mapping. For
    other configurations, the ``configuration`` attribute and coordinate names must be
    changed accordingly.

"""

import os
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from xarray.backends.api import open_datatree

import erlab.io
from erlab.io.dataloader import LoaderBase

if TYPE_CHECKING:
    from collections.abc import Hashable


def get_cache_file(file_path):
    file_path = Path(file_path)
    data_dir = file_path.parent
    cache_dir = data_dir.with_name(f".{data_dir.name}_cache")
    cache_file = cache_dir.joinpath(file_path.stem + "_2D_Data" + file_path.suffix)

    return cache_file


def cache_as_float32(file_path):
    """Cache and return the 2D part of the data as a float32 DataArray.

    Loading MAESTRO `.h5` files is slow because the data is stored in double precision.
    This function caches the 2D Data part in float32 to speed up subsequent loading. As
    a consequence, the loader will fail in read-only file systems.

    """
    dt = open_datatree(file_path, engine="h5netcdf", phony_dims="sort")

    cache_file = get_cache_file(file_path)
    if cache_file.is_file():
        return

    if not cache_file.parent.is_dir():
        cache_file.parent.mkdir(parents=True)

    data = dt["2D_Data"].load().to_dataset().astype(np.float32)

    if len(data.data_vars) > 1:
        warnings.warn(
            "More than one data variable is present in the data."
            "Only the first one will be used",
            stacklevel=2,
        )

    # Get the first data variable
    data = data[next(iter(data.data_vars))]

    # Save cache
    erlab.io.save_as_hdf5(data, cache_file, igor_compat=False)

    return data


class MAESTROMicroLoader(LoaderBase):
    name = "maestro"

    aliases = ("ALS_BL7", "als_bl7", "BL702", "bl702")

    name_map: ClassVar[dict] = {
        "x": "LMOTOR0",
        "y": "LMOTOR1",
        "z": "LMOTOR2",
        "chi": "LMOTOR3",  # Theta, polar
        "xi": "LMOTOR4",  # Beta, tilt
        "delta": "LMOTOR5",  # Phi, azimuth
        "beta": ("Slit Defl", "LMOTOR9"),
        "hv": ("MONOEV", "BL_E"),
        "temp_sample": "Cryostat_A",
        "polarization": "EPU Polarization",
    }
    coordinate_attrs = ("beta", "delta", "chi", "xi", "hv", "x", "y", "z")
    additional_attrs: ClassVar[dict] = {}

    skip_validate: bool = True
    always_single: bool = True

    @property
    def file_dialog_methods(self):
        return {"ALS BL7.0.2 Raw Data (*.h5)": (self.load, {})}

    def identify(self, num, data_dir):
        file = None
        for f in erlab.io.utils.get_files(data_dir, ".h5"):
            if re.match(rf"(\d+)_{str(num).zfill(5)}.h5", os.path.basename(f)):
                file = f

        if file is None:
            raise ValueError(f"No file found in {data_dir} for {num}")

        return [file], {}

    def load_single(self, file_path):
        cache_file = get_cache_file(file_path)
        dt = open_datatree(file_path, engine="h5netcdf", phony_dims="sort")

        if "PreScan" in dt["Comments"]:
            comment: str = dt["Comments"]["PreScan"].item()[0].decode()
        else:
            comment = ""

        def _parse_attr(v) -> str | int | float:
            """Strip quotes and convert numerical strings to int or float."""
            v = v.strip().decode()
            try:
                v = float(v)
                if v.is_integer():
                    return int(v)
                else:
                    return v
            except ValueError:
                if v.startswith("'") and v.endswith("'"):
                    return v[1:-1]
                return v

        nested_attrs: dict[Hashable, dict[str, tuple[str, str | int | float]]] = {}
        combined_attrs: dict[str, str | int | float] = {}
        for key, val in dt["Headers"].data_vars.items():
            # v given as (longname, name, value, comment)
            # we want to extract the name, comment and value
            nested_attrs[key] = {
                str(_parse_attr(v[1])): (str(_parse_attr(v[3])), _parse_attr(v[2]))
                for v in val.values
            }

            combined_attrs = {
                **combined_attrs,
                **{k: v[1] for k, v in nested_attrs[key].items()},
            }

        if "LWLVNM" in combined_attrs:
            scan_type: str = str(combined_attrs["LWLVNM"])

            lwlvlpn = int(combined_attrs["LWLVLPN"])  # number of low level loops
            motors: list[str] = []
            motor_shape: list[int] = []
            for i in range(lwlvlpn):
                nmsbdv = int(
                    combined_attrs[f"NMSBDV{i}"]
                )  # number of subdevices in i-th loop

                for j in range(nmsbdv):
                    nm = str(
                        combined_attrs[f"NM_{i}_{j}"]
                    )  # name of j-th subdevice in i-th loop
                    nm = nm.replace("CRYO-", "").strip()
                    motors.append(nm)
                    motor_shape.append(combined_attrs[f"N_{i}_{j}"])

        else:
            scan_type = "unknown"
            motors = ["XY"]

        # Get coords
        coords = (
            dt["0D_Data"].load().rename({"phony_dim_0": "phony_dim_3"}).to_dataset()
        )
        if len(motors) == 1:
            coords = coords.swap_dims({"phony_dim_3": motors[0]})

        if cache_file.is_file():
            # Load cache
            data = erlab.io.load_hdf5(cache_file)
        else:
            # Create cache
            data = cache_as_float32(file_path)

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
            # Stack and unstack to get the correct n-dimensional data
            data = (
                data.assign_coords(coords)
                .set_index(phony_dim_3=motors)
                .unstack("phony_dim_3")
            )

        # The configuration is hardcoded to 3, which is for vertical analyzer slit and
        # deflector map. For horizontal slit configuration or beta maps, coordinates and
        # the attribute must be changed accordingly.
        data.attrs = {
            "scan_type": scan_type,
            "comment": comment,
            "configuration": 3,
            "nested_attrs": nested_attrs,
        }
        data = data.assign_attrs(combined_attrs).squeeze()

        return data
