import itertools

import h5netcdf
import numpy as np
import xarray as xr

from erlab.io.utilities import find_first_file


def load(filename, data_dir=None, contains=None):
    try:
        filename = find_first_file(filename, data_dir=data_dir, contains=contains)
    except ValueError:
        pass

    with h5netcdf.File(filename, mode="r", phony_dims="sort") as ncf:
        attrs = dict(ncf.attrs)
        attr_keys_mapping = {
            "BL_energy": "hv",
            "X": "x",
            "Y": "y",
            "Z": "z",
            "A": "delta",  # azi
            "T": "chi",  # polar
            "F": "xi",  # tilt
            "TA": "temperature_cryotip",  # cold head temp
            "TB": "temperature",  # sample temp
            "CreationTime": "creation_time",
            "HDF5Version": "HDF5_Version",
            "H5pyVersion": "h5py_version",
            "Description": "description",
            "Notes": "notes",
            "Sample": "sample",
            "User": "user",
            "Model": "analyzer_name",
            "SerialNumber": "serial_number",
            "Version": "version",
            "StartTime": "start_time",
            "StopTime": "stop_time",
            "UpdateTime": "update_time",
            "Duration": "duration",
            "LensModeName": "lens_mode",
            "CameraMode": "camera_mode",
            "AcquisitionTime": "acquisition_time",
            "WorkFunction": "sample_workfunction",
            "MeasurementMode": "acquisition_mode",
            "PassEnergy": "pass_energy",
            "MCP": "mcp_voltage",
            "BL_pexit": "exit_slit",
            "YDeflection": "beta",
        }
        coords_keys_mapping = {
            "Kinetic Energy": "eV",
            "ThetaX": "alpha",
            "ThetaY": "beta",
        }
        fixed_attrs = {"analyzer_type": "hemispherical"}
        attr_to_coords = ["hv"]

        for k, v in fixed_attrs.items():
            attrs[k] = v

        compat_mode = "data" in ncf.groups  # compatibility with older data

        if compat_mode:
            attr_keys_mapping = {
                "BL_photon_energy": "hv",
                "a": "delta",  # azi
                "t": "chi",  # polar
                "f": "xi",  # tilt
                "cryo_temperature": "temperature_cryotip",  # cold head temp
                "cold_head_temperature": "temperature",  # sample temp
                "creationtime": "creation_time",
                "model": "analyzer_name",
                "MCPVoltage": "mcp_voltage",
                "DeflectionY": "beta",
            }

        for k, v in ncf.groups.items():
            ds = xr.open_dataset(xr.backends.H5NetCDFStore(v, autoclose=True))
            if k.casefold() == "Beamline".casefold():
                ds.attrs = {f"BL_{kk}": vv for kk, vv in ds.attrs.items()}

            attrs = attrs | ds.attrs
            if k.casefold() == "Data".casefold():
                if compat_mode:
                    if "exposure" in ds.variables:
                        ds = ds.rename_vars(counts="spectrum", exposure="time")
                    else:
                        ds = ds.rename_vars(counts="spectrum")
                else:
                    if "Time" in ds.variables:
                        ds = ds.rename_vars(Count="spectrum", Time="time")
                    else:
                        ds = ds.rename_vars(Count="spectrum")

                axes = [dict(v.groups[g].attrs) for g in v.groups]

                for i, ax in enumerate(axes):
                    axes[i] = {
                        name.lower(): val for name, val in ax.items()
                    }  # unify case

                # if not compat_mode:
                #     # some data have mismatching order between axes and phony dims
                #     # some even have mismatching counts... those are currently not supported
                #     axes_new = [None] * len(axes)
                #     for ax in axes:
                #         if ax["count"] in ds.spectrum.shape:
                #             try:
                #                 axes_new[ds.spectrum.shape.index(ax["count"])] = ax
                #             except ValueError:
                #                 pass
                #     axes = axes_new

                data = ds.rename_dims(
                    {f"phony_dim_{i}": ax["label"] for i, ax in enumerate(axes)}
                ).load()
                ds.close()

                for i, ax in enumerate(axes):
                    if compat_mode:
                        cnt = v.dimensions[f"phony_dim_{i}"].size
                    else:
                        cnt = ax["count"]
                    mn, mx = (
                        ax["offset"],
                        ax["offset"] + (cnt - 1) * ax["delta"],
                    )
                    data = data.assign_coords({ax["label"]: np.linspace(mn, mx, cnt)})

        for k in list(attrs.keys()):
            if k in attr_keys_mapping.keys():
                attrs[attr_keys_mapping[k]] = attrs.pop(k)

        data = data.rename(
            {k: v for k, v in coords_keys_mapping.items() if k in data.dims}
        )

        # if compat:
        #     if "theta" not in itertools.product(attrs.keys(), data.dims):
        #         attrs["theta"] = 0.0

        #     attrs["alpha"] = 90.0
        #     attrs["psi"] = 0.0

        #     for a in ["alpha", "beta", "theta", "theta_DA", "chi", "phi", "psi"]:
        #         try:
        #             data = data.assign_coords({a: np.deg2rad(data[a])})
        #         except KeyError:
        #             try:
        #                 data = data.assign_coords({a: np.deg2rad(attrs.pop(a))})
        #             except KeyError:
        #                 continue
        
        for a in ["chi", "xi", "alpha", "beta", "delta"]:
            try:
                data = data.assign_coords({a: data[a]})
            except KeyError:
                try:
                    data = data.assign_coords({a: attrs.pop(a)})
                except KeyError:
                    continue

        # Assume that nobody will perform polar mapping at SSRL 5-2.
        attrs["configuration"] = 3  # Type I with DA

        for c in attr_to_coords:
            data = data.assign_coords({c: attrs.pop(c)})

        # data.attrs = attrs
        # data.spectrum.attrs = attrs
        if "time" in data.variables:
            out = data.spectrum / data.time
        else:
            out = data.spectrum
        out.attrs = attrs
    return out
