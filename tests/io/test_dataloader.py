import csv
import datetime
import errno
import os
import pathlib
import re
import tempfile
from typing import ClassVar

import numpy as np
import pytest
import xarray as xr

import erlab
from erlab.interactive.imagetool.manager import ImageToolManager
from erlab.io.dataloader import LoaderBase
from erlab.io.exampledata import generate_data_angles


def make_data(beta=5.0, temp=20.0, hv=50.0, bandshift=0.0):
    data = generate_data_angles(
        shape=(250, 1, 300),
        angrange={"alpha": (-15, 15), "beta": (beta, beta)},
        hv=hv,
        configuration=1,
        temp=temp,
        bandshift=bandshift,
        assign_attributes=False,
        seed=1,
    ).T

    # Rename coordinates. The loader must rename them back to the original names.
    data = data.rename(
        {
            "alpha": "ThetaX",
            "beta": "Polar",
            "eV": "BindingEnergy",
            "hv": "PhotonEnergy",
            "xi": "Tilt",
            "delta": "Azimuth",
        }
    )
    dt = datetime.datetime.now()

    # Assign some attributes that real data would have
    return data.assign_attrs(
        {
            "LensMode": "Angular30",  # Lens mode of the analyzer
            "SpectrumType": "Fixed",  # Acquisition mode of the analyzer
            "PassEnergy": 10,  # Pass energy of the analyzer
            "UndPol": 0,  # Undulator polarization
            "Date": dt.strftime(r"%d/%m/%Y"),  # Date of the measurement
            "Time": dt.strftime("%I:%M:%S %p"),  # Time of the measurement
            "TB": temp,
            "X": 0.0,
            "Y": 0.0,
            "Z": 0.0,
        }
    )


def _format_polarization(val) -> str:
    val = round(float(val))
    return {0: "LH", 2: "LV", -1: "RC", 1: "LC"}.get(val, str(val))


def _parse_time(darr: xr.DataArray) -> datetime.datetime:
    return datetime.datetime.strptime(
        f"{darr.attrs['Date']} {darr.attrs['Time']}",
        r"%d/%m/%Y %I:%M:%S %p",
    )


def _determine_kind(darr: xr.DataArray) -> str:
    if "scan_type" in darr.attrs and darr.attrs["scan_type"] == "live":
        return "LP" if "beta" in darr.dims else "LXY"

    data_type = "xps"
    if "alpha" in darr.dims:
        data_type = "cut"
    if "beta" in darr.dims:
        data_type = "map"
    if "hv" in darr.dims:
        data_type = "hvdep"
    return data_type


def test_loader(qtbot) -> None:
    # Create a temporary directory
    tmp_dir = tempfile.TemporaryDirectory()

    # Generate a map
    beta_coords = np.linspace(2, 7, 10)

    # Generate and save cuts with different beta values
    for i, beta in enumerate(beta_coords):
        data = make_data(beta=beta, temp=20.0 + i, hv=50.0)
        filename = f"{tmp_dir.name}/data_001_S{str(i + 1).zfill(3)}.h5"
        data.to_netcdf(filename, engine="h5netcdf")

    # Write scan coordinates to a csv file
    with open(f"{tmp_dir.name}/data_001_axis.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Index", "Polar"])

        for i, beta in enumerate(beta_coords):
            writer.writerow([i + 1, beta])

    # Generate some cuts with different band shifts
    for i in range(4):
        data = make_data(beta=5.0, temp=20.0, hv=50.0, bandshift=-i * 0.05)
        filename = f"{tmp_dir.name}/data_{str(i + 2).zfill(3)}.h5"
        data.to_netcdf(filename, engine="h5netcdf")

    class ExampleLoader(LoaderBase):
        name = "example"
        description = "Example loader for testing purposes"

        name_map: ClassVar[dict] = {
            "eV": "BindingEnergy",
            "alpha": "ThetaX",
            "beta": [
                "Polar",
                "Polar Compens",
            ],  # Can have multiple names assigned to the same name
            # If both are present in the data, a ValueError will be raised
            "delta": "Azimuth",
            "xi": "Tilt",
            "x": "X",
            "y": "Y",
            "z": "Z",
            "hv": "PhotonEnergy",
            "polarization": "UndPol",
            "sample_temp": "TB",
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
            "photon_flux",
            "sample_temp",
        )
        # Attributes to be used as coordinates. Place all attributes that we don't want
        # to lose when merging multiple file scans here.

        additional_attrs: ClassVar[dict] = {
            "configuration": 1,  # Experimental geometry required for kspace conversion
            "sample_workfunction": 4.3,
        }  # Any additional metadata you want to add to the data

        formatters: ClassVar[dict] = {
            "polarization": _format_polarization,
            "LensMode": lambda x: x.replace("Angular", "A"),
        }

        summary_attrs: ClassVar[dict] = {
            "Time": _parse_time,
            "Type": _determine_kind,
            "Lens Mode": "LensMode",
            "Scan Type": "SpectrumType",
            "T(K)": "sample_temp",
            "Pass E": "PassEnergy",
            "Polarization": "polarization",
            "hv": "hv",
            "x": "x",
            "y": "y",
            "z": "z",
            "polar": "beta",
            "tilt": "xi",
            "azi": "delta",
        }

        summary_sort = "Time"

        skip_validate = False

        always_single = False

        def identify(self, num, data_dir):
            coord_dict = {}
            data_dir = pathlib.Path(data_dir)

            # Look for scans with data_###_S###.h5, and sort them
            files = list(data_dir.glob(f"data_{str(num).zfill(3)}_S*.h5"))
            files.sort()

            if len(files) == 0:
                # If no files found, look for data_###.h5
                files = list(data_dir.glob(f"data_{str(num).zfill(3)}.h5"))
            else:
                # If files found, extract coordinate values from the filenames
                axis_file = data_dir / f"data_{str(num).zfill(3)}_axis.csv"
                with axis_file.open("r", encoding="locale") as f:
                    header = f.readline().strip().split(",")

                coord_arr = np.loadtxt(axis_file, delimiter=",", skiprows=1)

                for i, hdr in enumerate(header[1:]):
                    coord_dict[hdr] = coord_arr[: len(files), i + 1].astype(np.float64)

            if len(files) == 0:
                # If no files found up to this point, return None
                return None

            return files, coord_dict

        def load_single(self, file_path, without_values=False):
            darr = xr.open_dataarray(file_path, engine="h5netcdf")

            if without_values:
                # Do not load the data into memory
                return xr.DataArray(
                    np.zeros(darr.shape, darr.dtype),
                    coords=darr.coords,
                    dims=darr.dims,
                    attrs=darr.attrs,
                )

            return darr

        def post_process(self, data: xr.DataArray) -> xr.DataArray:
            data = super().post_process(data)

            if "sample_temp" in data.coords:
                # Add temperature to attributes, for backwards compatibility
                temp = float(data.sample_temp.mean())
                data = data.assign_attrs(sample_temp=temp)

            return data

        def infer_index(self, name):
            # Get the scan number from file name
            try:
                scan_num: str = re.match(r".*?(\d{3})(?:_S\d{3})?", name).group(1)
            except (AttributeError, IndexError):
                return None, None

            if scan_num.isdigit():
                return int(scan_num), {}
            return None, None

        def files_for_summary(self, data_dir):
            return erlab.io.utils.get_files(data_dir, extensions=[".h5"])

    with erlab.io.loader_context("example", tmp_dir.name):
        erlab.io.load(1)

    with pytest.raises(
        FileNotFoundError,
        match=re.escape(
            str(
                FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), "some_nonexistent_dir"
                )
            )
        ),
    ):
        erlab.io.loaders.set_data_dir("some_nonexistent_dir")

    # Test if the reprs are working
    assert repr(erlab.io.loaders).startswith("Name")
    assert erlab.io.loaders._repr_html_().startswith("<div><style>")

    # Set loader
    erlab.io.set_loader("example")
    erlab.io.set_data_dir(tmp_dir.name)
    erlab.io.load(2)

    erlab.io.set_loader(None)

    # Set with attribute
    erlab.io.loaders.current_loader = "example"
    erlab.io.loaders.current_data_dir = tmp_dir.name
    erlab.io.load(2)

    # Test if coordinate_attrs are correctly assigned
    mfdata = erlab.io.load(1)
    assert np.allclose(mfdata["sample_temp"].values, 20.0 + np.arange(len(beta_coords)))

    df = erlab.io.summarize(display=False)
    assert len(df.index) == 5

    # Test if pretty printing works
    erlab.io.loaders.current_loader.get_styler(df)._repr_html_()

    # Interactive summary
    erlab.io.loaders.current_loader._isummarize(df)

    # Interactive summary with imagetool manager
    manager = ImageToolManager()
    qtbot.addWidget(manager)

    erlab.io.loaders.current_loader._isummarize(df)
    manager.close()

    # Cleanup the temporary directory
    tmp_dir.cleanup()
