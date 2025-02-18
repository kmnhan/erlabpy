"""Data loader for beamline 5-2 at SSRL."""

__all__ = ["SSRL52Loader"]

import datetime
import os
import re
import typing
from collections.abc import Callable

import numpy as np
import xarray as xr

import erlab
from erlab.io.dataloader import LoaderBase

if typing.TYPE_CHECKING:
    import h5netcdf
else:
    import lazy_loader as _lazy

    h5netcdf = _lazy.load("h5netcdf")


def _format_polarization(val) -> str:
    val = float(np.round(val, 3))
    return {0.0: "LH", 0.5: "LV", 0.25: "RC", -0.25: "LC"}.get(val, str(val))


def _parse_value(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


class SSRL52Loader(LoaderBase):
    name = "ssrl52"
    description = "SSRL Beamline 5-2"
    extensions: typing.ClassVar[set[str]] = {".h5"}

    aliases = ("ssrl", "bl5-2")

    name_map: typing.ClassVar[dict] = {
        "eV": ["Kinetic Energy", "Binding Energy"],
        "alpha": "ThetaX",
        "beta": ["ThetaY", "YDeflection", "DeflectionY"],
        "delta": ["A", "a"],  # azi
        "chi": ["T", "t"],  # polar
        "xi": ["F", "f"],  # tilt
        "x": "X",
        "y": "Y",
        "z": "Z",
        "hv": ["energy", "photon_energy"],
        "sample_temp": ["TB", "sample_stage_temperature"],
        "sample_workfunction": "WorkFunction",
    }

    coordinate_attrs = (
        "beta",
        "delta",
        "chi",
        "xi",
        "hv",
        "x",
        "y",
        "z",
        "polarization",
        "sample_temp",
    )

    additional_attrs: typing.ClassVar[dict] = {
        "configuration": 3,
        "sample_workfunction": 4.5,
    }

    formatters: typing.ClassVar[dict[str, Callable]] = {
        "CreationTimeStamp": datetime.datetime.fromtimestamp,
        "PassEnergy": round,
        "polarization": _format_polarization,
    }

    summary_attrs: typing.ClassVar[
        dict[str, str | Callable[[xr.DataArray], typing.Any]]
    ] = {
        "time": "CreationTimeStamp",
        "type": "Description",
        "lens mode": "LensModeName",
        "region": "RegionName",
        "temperature": "sample_temp",
        "pass energy": "PassEnergy",
        "pol": "polarization",
        "hv": "hv",
        "polar": "chi",
        "tilt": "xi",
        "azi": "delta",
        "deflector": "beta",
        "x": "x",
        "y": "y",
        "z": "z",
    }

    summary_sort = "time"

    always_single: bool = True
    skip_validate: bool = True

    @property
    def file_dialog_methods(self):
        return {"SSRL BL5-2 Raw Data (*.h5)": (self.load, {})}

    def load_single(
        self, file_path: str | os.PathLike, without_values: bool = False
    ) -> xr.DataArray:
        is_hvdep: bool = False

        dim_mapping: dict[str, str] = {}

        with h5netcdf.File(file_path, mode="r", phony_dims="sort") as ncf:
            attrs = dict(ncf.attrs)
            compat_mode = "data" in ncf.groups  # Compatibility with older data

            for k, v in ncf.groups.items():
                ds = xr.open_dataset(xr.backends.H5NetCDFStore(v, autoclose=True))

                # if k.casefold() == "Beamline".casefold():
                #     attrs[k] = ds.attrs
                #     attrs["polarization"] = ds.attrs.get("polarization")

                # else:
                # Merge group attributes
                attrs = attrs | ds.attrs

                if k.casefold() == "Data".casefold():
                    if compat_mode:
                        if "exposure" in ds.variables:
                            ds = ds.rename_vars(counts="spectrum", exposure="time")
                        else:
                            ds = ds.rename_vars(counts="spectrum")
                    elif "Time" in ds.variables:
                        ds = ds.rename_vars(Count="spectrum", Time="time")
                    else:
                        ds = ds.rename_vars(Count="spectrum")

                    # List of dicts containing scale and label info for each axis
                    axes: list[dict[str, float | int | str]] = [
                        dict(v.groups[g].attrs) for g in v.groups
                    ]

                    for i, ax in enumerate(axes):
                        # Unify case for compatibility with old data
                        axes[i] = {name.lower(): val for name, val in ax.items()}

                    # Apply dim labels
                    dim_mapping = {
                        f"phony_dim_{i}": str(ax["label"]) for i, ax in enumerate(axes)
                    }
                    data = ds.rename_dims(dim_mapping)

                    # Apply coordinates
                    for i, ax in enumerate(axes):
                        if compat_mode:
                            cnt = v.dimensions[f"phony_dim_{i}"].size
                        else:
                            cnt = int(ax["count"])

                        if isinstance(ax["offset"], str):
                            if ax["label"] == "energy":
                                data = data.assign_coords(
                                    {
                                        ax["label"]: np.array(
                                            ncf["MapInfo"]["Beamline:energy"]
                                        )
                                    }
                                )
                                # Axes2 may have some values... not sure what they are
                                # For now, just ignore them and use beamline attributes
                                continue
                            if ax["label"] != "Kinetic Energy":
                                erlab.utils.misc.emit_user_level_warning(
                                    "Undefined offset for non-energy axis. This was "
                                    "not taken into account while writing the loader "
                                    "code. Please report this issue. Resulting data "
                                    "may be incorrect"
                                )
                                continue
                            is_hvdep = True

                            # For hv dep scans, EKin is given for each scan
                            data = data.rename({ax["label"]: "Binding Energy"})
                            ax["label"] = "Binding Energy"

                            # ax['offset'] will be something like:
                            # "MapInfo:Data:Axes0:Offset"
                            offset_key: str = ax["offset"][8:]

                            # Take first kinetic energy
                            offset = np.array(ncf["MapInfo"][offset_key])[0]

                            if isinstance(ax["delta"], str):
                                delta = np.array(ncf["MapInfo"][ax["delta"][8:]])
                                # may be ~1e-8 difference between values
                                if not np.allclose(delta, delta[0], atol=1e-7):
                                    erlab.utils.misc.emit_user_level_warning(
                                        "Non-uniform delta for hv-dependent scan. This "
                                        "was not taken into account while writing the "
                                        "loader code. Please report this issue. "
                                        "Resulting data may be incorrect"
                                    )
                                delta = delta[0]
                            else:
                                delta = float(ax["delta"])

                        else:
                            offset = float(ax["offset"])
                            delta = float(ax["delta"])

                        mn, mx = (offset, offset + (cnt - 1) * delta)
                        coord = np.linspace(mn, mx, cnt)

                        if len(data[ax["label"]]) != cnt:
                            # For premature data
                            coord = coord[: len(data[ax["label"]])]

                        data = data.assign_coords({ax["label"]: coord})

            attrs = {k: _parse_value(v) for k, v in attrs.items()}

            coord_names = list(data.coords.keys())
            coord_sizes = [len(data[coord]) for coord in coord_names]
            coord_attrs: dict = {}
            for k, v in dict(attrs).items():
                if isinstance(v, str) and v.startswith("MapInfo:"):
                    del attrs[k]
                    var = np.array(ncf["MapInfo"][v[8:]])
                    same_length_indices = [
                        i for i, s in enumerate(coord_sizes) if s == len(var)
                    ]
                    for idx in list(same_length_indices):
                        # Attributes should not be dependent on these dims
                        if coord_names[idx] in (
                            "ThetaX",
                            "Kinetic Energy",
                            "Binding Energy",
                        ):
                            same_length_indices.remove(idx)
                    if len(same_length_indices) != 1:
                        # Multiple dimensions with the same length, ambiguous
                        erlab.utils.misc.emit_user_level_warning(
                            f"Ambiguous length for {k}. This was not taken into "
                            "account while writing the loader code. Please report this "
                            "issue. Resulting data may be incorrect"
                        )
                    idx = same_length_indices[-1]
                    coord_attrs[k] = xr.DataArray(var, dims=[coord_names[idx]])

            if is_hvdep:
                data = data.assign_coords(
                    {
                        "Binding Energy": data["Binding Energy"]
                        - data["energy"].values[0]
                        + attrs.get("WorkFunction", 4.465)
                    }
                )

                # data = data.rename(energy="hv")

            darr = data["spectrum"]

            if not without_values:
                darr = darr.load()  # Load into memory before closing file
                if "time" in data.variables:
                    # Normalize by dwell time
                    darr = darr / data["time"]

            else:
                darr = xr.DataArray(
                    np.zeros(darr.shape, darr.dtype),
                    coords=darr.coords,
                    dims=darr.dims,
                    attrs=darr.attrs,
                    name=darr.name,
                )

            darr = darr.assign_attrs(attrs)

        return darr.assign_coords(coord_attrs)

    def post_process(self, data: xr.DataArray) -> xr.DataArray:
        data = super().post_process(data)

        # Convert to binding energy
        if (
            "sample_workfunction" in data.attrs
            and "eV" in data.dims
            and data["eV"].min() > 0
        ):
            data = data.assign_coords(
                eV=data["eV"] - float(data["hv"]) + data.attrs["sample_workfunction"]
            )

        return data

    def identify(self, num: int, data_dir: str | os.PathLike, zap: bool = False):
        if zap:
            target_files = erlab.io.utils.get_files(data_dir, ".h5", contains="zap")
        else:
            target_files = erlab.io.utils.get_files(data_dir, ".h5", notcontains="zap")

        pattern = re.compile(r"(.*?)_" + str(num).zfill(4) + r".h5")
        matches = [path for path in target_files if pattern.match(path.name)]

        return matches, {}

    def load_zap(self, identifier, data_dir):
        return self.load(identifier, data_dir, zap=True)

    def files_for_summary(self, data_dir):
        return sorted(erlab.io.utils.get_files(data_dir, extensions=(".h5",)))
