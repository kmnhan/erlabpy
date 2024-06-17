"""Data loader for beamline 5-2 at SSRL."""

import datetime
import os
import re
import warnings
from typing import ClassVar

import h5netcdf
import numpy as np
import pandas as pd
import xarray as xr

import erlab.io.utils
from erlab.io.dataloader import LoaderBase


class SSRL52Loader(LoaderBase):
    name = "ssrl"
    aliases = ("ssrl52", "bl5-2")

    name_map: ClassVar[dict] = {
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
        "temp_sample": ["TB", "sample_stage_temperature"],
        "sample_workfunction": "WorkFunction",
    }

    coordinate_attrs = ("beta", "delta", "chi", "xi", "hv", "x", "y", "z")

    additional_attrs: ClassVar[dict] = {
        "configuration": 3,
        "sample_workfunction": 4.5,
    }

    always_single: bool = True
    skip_validate: bool = True

    @property
    def file_dialog_methods(self):
        return {"SSRL BL5-2 Raw Data (*.h5)": (self.load, {})}

    def load_single(self, file_path: str | os.PathLike) -> xr.DataArray:
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
                        f"phony_dim_{i}": ax["label"] for i, ax in enumerate(axes)
                    }
                    data = ds.rename_dims(dim_mapping).load()

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
                            elif ax["label"] != "Kinetic Energy":
                                warnings.warn(
                                    "Undefined offset for non-energy axis. This was "
                                    "not taken into account while writing the loader "
                                    "code. Please report this issue. Resulting data "
                                    "may be incorrect",
                                    stacklevel=1,
                                )
                                continue
                            is_hvdep = True
                            # For hv dep scans, EKin is given for each scan
                            data = data.rename({ax["label"]: "Binding Energy"})
                            ax["label"] = "Binding Energy"
                            # ax['offset'] will be something like "MapInfo:Data:Axes0:Offset"
                            offset_key: str = ax["offset"][8:]
                            # Take first kinetic energy
                            offset = np.array(ncf["MapInfo"][offset_key])[0]

                            if isinstance(ax["delta"], str):
                                delta = np.array(ncf["MapInfo"][ax["delta"][8:]])
                                # may be ~1e-8 difference between values
                                if not np.allclose(delta, delta[0], atol=1e-7):
                                    warnings.warn(
                                        "Non-uniform delta for hv-dependent scan. This "
                                        "was not taken into account while writing the "
                                        "loader code. Please report this issue. "
                                        "Resulting data may be incorrect",
                                        stacklevel=1,
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
                        warnings.warn(
                            f"Ambiguous length for {k}. This was not taken into account "
                            "while writing the loader code. Please report this issue. "
                            "Resulting data may be incorrect",
                            stacklevel=1,
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

        if "time" in data.variables:
            # Normalize by dwell time
            data = data["spectrum"] / data["time"]
        else:
            data = data["spectrum"]

        data = data.assign_attrs(attrs)
        data = data.assign_coords(coord_attrs)

        return self.process_keys(data)

    def post_process(self, data: xr.DataArray) -> xr.DataArray:
        data = super().post_process(data)

        if "temp_sample" in data.coords:
            # Add temperature to attributes
            temp = float(data.temp_sample.mean())
            data = data.assign_attrs(temp_sample=temp)

        # Convert to binding energy
        if "sample_workfunction" in data.attrs and "eV" in data.dims:
            if data.eV.min() > 0:
                data = data.assign_coords(
                    eV=data.eV - float(data.hv) + data.attrs["sample_workfunction"]
                )

        return data

    def identify(
        self,
        num: int,
        data_dir: str | os.PathLike,
        zap: bool = False,
    ):
        if zap:
            target_files = erlab.io.utils.get_files(
                data_dir, extensions=(".h5",), contains="zap"
            )
        else:
            target_files = erlab.io.utils.get_files(
                data_dir, extensions=(".h5",), notcontains="zap"
            )

        for file in target_files:
            match = re.match(r"(.*?)_" + str(num).zfill(4) + r".h5", file)
            if match is not None:
                return [file], {}

        raise FileNotFoundError(f"No files found for scan {num} in {data_dir}")

    # def post_process(
    #     self, data: xr.DataArray | xr.Dataset
    # ) -> xr.DataArray | xr.Dataset:
    #     data = super().post_process(data)

    #     if "eV" in data.coords:
    #         data = data.assign_coords(eV=-data.eV.values)

    #     return data

    def load_zap(self, identifier, data_dir):
        return self.load(identifier, data_dir, zap=True)

    def generate_summary(
        self, data_dir: str | os.PathLike, exclude_zap: bool = False
    ) -> pd.DataFrame:
        files: dict[str, str] = {}

        if exclude_zap:
            target_files = erlab.io.utils.get_files(
                data_dir, extensions=(".h5",), notcontains="zap"
            )
        else:
            target_files = erlab.io.utils.get_files(data_dir, extensions=(".h5",))

        for pth in target_files:
            base_name = os.path.splitext(os.path.basename(pth))[0]
            files[base_name] = pth

        summary_attrs: dict[str, str] = {
            "Type": "Description",
            "Lens Mode": "LensModeName",
            "Region": "RegionName",
            "T(K)": "temp_sample",
            "Pass E": "PassEnergy",
            "Polarization": "polarization",
            "hv": "hv",
            # "Entrance Slit": "Entrance Slit",
            # "Exit Slit": "Exit Slit",
            "x": "x",
            "y": "y",
            "z": "z",
            "polar": "chi",
            "tilt": "xi",
            "azi": "delta",
            "DA": "beta",
        }

        cols = ["File Name", "Path", "Time", *summary_attrs.keys()]

        data_info = []

        for name, path in files.items():
            data = self.load(path)

            data_info.append(
                [
                    name,
                    path,
                    datetime.datetime.fromtimestamp(data.attrs["CreationTimeStamp"]),
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

                if k == "Pass E":
                    val = round(val)

                elif k == "Polarization":
                    if np.iterable(val):
                        val = np.round(np.asarray(val), 3).astype(float)
                    else:
                        val = [float(np.round(val, 3))]
                    val = [
                        {0.0: "LH", 0.5: "LV", 0.25: "RC", -0.25: "LC"}.get(v, v)
                        for v in val
                    ]

                    if len(val) == 1:
                        val = val[0]

                data_info[-1].append(val)

            del data

        return (
            pd.DataFrame(data_info, columns=cols)
            .sort_values("Time")
            .set_index("File Name")
        )
