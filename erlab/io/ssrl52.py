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

    ncf = h5netcdf.File(filename, mode="r", phony_dims="sort")
    attrs = dict(ncf.attrs)
    attr_keys_mapping = {
        "BL_energy": "hv",
        "X": "x",
        "Y": "y",
        "Z": "z",
        "A": "chi",  # azi
        "T": "beta",  # polar
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
        "BL_I0": "photon_flux",  # 이거 맞나..?
        "BL_spear": "beam_current",  # 이거 맞나..?
        "YDeflection": "theta_DA",
    }
    coords_keys_mapping = {
        "Kinetic Energy": "eV",
        "ThetaX": "phi",
        "ThetaY": "theta",
    }
    fixed_attrs = {"analyzer_type": "hemispherical"}
    attr_to_coords = ["hv"]

    for k, v in fixed_attrs.items():
        attrs[k] = v

    compat_mode = "data" in ncf.groups  # compatibility with older data

    if compat_mode:
        attr_keys_mapping = {
            "BL_photon_energy": "hv",
            "a": "chi",  # azi
            "t": "beta",  # polar
            "cryo_temperature": "temperature_cryotip",  # cold head temp
            "cold_head_temperature": "temperature",  # sample temp
            "creationtime": "creation_time",
            "model": "analyzer_name",
            "MCPVoltage": "mcp_voltage",
            "BL_i0": "photon_flux",  # 이거 맞나..?
            "BL_spear_current": "beam_current",  # 이거 맞나..?
            "DeflectionY": "theta_DA",
        }

    for k, v in ncf.groups.items():
        ds = xr.open_dataset(xr.backends.H5NetCDFStore(v))
        if k.casefold() == "Beamline".casefold():
            ds.attrs = {f"BL_{kk}": vv for kk, vv in ds.attrs.items()}

        attrs = attrs | ds.attrs
        if k.casefold() == "Data".casefold():
            axes = [dict(v.groups[g].attrs) for g in v.groups]
            for i, ax in enumerate(axes):
                axes[i] = {name.lower(): val for name, val in ax.items()}  # unify case
            data = ds.rename_dims(
                {f"phony_dim_{i}": ax["label"] for i, ax in enumerate(axes)}
            )
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
            if compat_mode:
                if "exposure" in data.variables:
                    data = data.rename_vars(counts="spectrum", exposure="time")
                else:
                    data = data.rename_vars(counts="spectrum")
            else:
                if "Time" in data.variables:
                    data = data.rename_vars(Count="spectrum", Time="time")
                else:
                    data = data.rename_vars(Count="spectrum")

    for k in list(attrs.keys()):
        if k in attr_keys_mapping.keys():
            attrs[attr_keys_mapping[k]] = attrs.pop(k)

    data = data.rename({k: v for k, v in coords_keys_mapping.items() if k in data.dims})

    if "theta" not in itertools.product(attrs.keys(), data.dims):
        attrs["theta"] = 0.0

    attrs["alpha"] = 90.0
    attrs["psi"] = 0.0

    for a in ["alpha", "beta", "theta", "theta_DA", "chi", "phi", "psi"]:
        try:
            data = data.assign_coords({a: np.deg2rad(data[a])})
        except KeyError:
            try:
                data = data.assign_coords({a: np.deg2rad(attrs.pop(a))})
            except KeyError:
                continue

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
