import csv
import datetime
import glob
import os
import re
import tempfile
from typing import ClassVar

import erlab.io
import numpy as np
import pandas as pd
import pytest
from erlab.io.dataloader import LoaderBase
from erlab.io.exampledata import generate_data_angles


def make_data(beta=5.0, temp=20.0, hv=50.0, bandshift=0.0):
    data = generate_data_angles(
        shape=(250, 1, 300),
        angrange={"alpha": (-15, 15), "beta": (beta, beta)},
        hv=hv,
        configuration=1,  # Configuration, see
        temp=temp,
        bandshift=bandshift,
        count=1000,
        assign_attributes=False,
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

    # Assign some attributes that real data would have
    data = data.assign_attrs(
        {
            "LensMode": "Angular30",  # Lens mode of the analyzer
            "SpectrumType": "Fixed",  # Acquisition mode of the analyzer
            "PassEnergy": 10,  # Pass energy of the analyzer
            "UndPol": 0,  # Undulator polarization
            "DateTime": datetime.datetime.now().isoformat(),  # Acquisition time
            "TB": temp,
            "X": 0.0,
            "Y": 0.0,
            "Z": 0.0,
        }
    )
    return data


def test_loader():
    # Create a temporary directory
    tmp_dir = tempfile.TemporaryDirectory()

    # Generate a map
    beta_coords = np.linspace(2, 7, 10)

    for i, beta in enumerate(beta_coords):
        erlab.io.save_as_hdf5(
            make_data(beta=beta, temp=20.0, hv=50.0),
            filename=f"{tmp_dir.name}/data_001_S{str(i + 1).zfill(3)}.h5",
            igor_compat=False,
        )

    # Write scan coordinates to a csv file
    with open(f"{tmp_dir.name}/data_001_axis.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Index", "Polar"])

        for i, beta in enumerate(beta_coords):
            writer.writerow([i + 1, beta])

    # Generate a cut
    erlab.io.save_as_hdf5(
        make_data(beta=5.0, temp=20.0, hv=50.0),
        filename=f"{tmp_dir.name}/data_002.h5",
        igor_compat=False,
    )

    # List the generated files
    sorted(os.listdir(tmp_dir.name))

    class ExampleLoader(LoaderBase):
        name = "example"

        aliases = ("Ex",)

        name_map: ClassVar[dict] = {
            "eV": "BindingEnergy",
            "alpha": "ThetaX",
            "beta": [
                "Polar",
                "Polar Compens",
            ],  # Can have multiple names assigned to the same name
            "delta": "Azimuth",
            "xi": "Tilt",
            "x": "X",
            "y": "Y",
            "z": "Z",
            "hv": "PhotonEnergy",
            "polarization": "UndPol",
            "temp_sample": "TB",
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
        )
        # Attributes to be used as coordinates. Place all attributes that we don't want to
        # lose when merging multiple file scans here.

        additional_attrs: ClassVar[dict] = {
            "configuration": 1,  # Experimental geometry. Required for momentum conversion
            "sample_workfunction": 4.3,
        }  # Any additional metadata you want to add to the data

        skip_validate = False

        always_single = False

        def identify(self, num, data_dir):
            coord_dict = {}

            # Look for scans with data_###_S###.h5, and sort them
            files = glob.glob(f"data_{str(num).zfill(3)}_S*.h5", root_dir=data_dir)
            files.sort()

            if len(files) == 0:
                # If no files found, look for data_###.h5
                files = glob.glob(f"data_{str(num).zfill(3)}.h5", root_dir=data_dir)
            else:
                # If files found, extract coordinate values from the filenames
                axis_file = f"{data_dir}/data_{str(num).zfill(3)}_axis.csv"
                with open(axis_file) as f:
                    header = f.readline().strip().split(",")

                coord_arr = np.loadtxt(axis_file, delimiter=",", skiprows=1)

                for i, hdr in enumerate(header[1:]):
                    key = self.name_map_reversed.get(hdr, hdr)
                    coord_dict[key] = coord_arr[: len(files), i + 1].astype(np.float64)

            if len(files) == 0:
                # If no files found up to this point, raise an error
                raise FileNotFoundError(f"No files found for scan {num} in {data_dir}")

            # Files must be full paths
            files = [os.path.join(data_dir, f) for f in files]

            return files, coord_dict

        def load_single(self, file_path):
            data = erlab.io.load_hdf5(file_path)

            # To prevent conflicts when merging multiple scans, we rename the coordinates
            # prior to concatenation
            return self.process_keys(data)

        def infer_index(self, name):
            # Get the scan number from file name
            try:
                scan_num: str = re.match(r".*?(\d{3})(?:_S\d{3})?", name).group(1)
            except (AttributeError, IndexError):
                return None, None

            if scan_num.isdigit():
                return int(scan_num), {}
            else:
                return None, None

        def generate_summary(self, data_dir):
            # Get all valid data files in directory
            files = {}
            for path in erlab.io.utils.get_files(data_dir, extensions=[".h5"]):
                # Base name
                data_name = os.path.splitext(os.path.basename(path))[0]

                # If multiple scans, strip the _S### part
                name_match = re.match(r"(.*?_\d{3})_(?:_S\d{3})?", data_name)
                if name_match is not None:
                    data_name = name_match.group(1)

                files[data_name] = path

            # Map dataframe column names to data attributes
            attrs_mapping = {
                "Lens Mode": "LensMode",
                "Scan Type": "SpectrumType",
                "T(K)": "temp_sample",
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
            column_names = ["File Name", "Path", "Time", "Type", *attrs_mapping.keys()]

            data_info = []

            processed_indices = set()
            for name, path in files.items():
                # Skip already processed multi-file scans
                index, _ = self.infer_index(name)
                if index in processed_indices:
                    continue
                elif index is not None:
                    processed_indices.add(index)

                # Load data
                data = self.load(path)

                # Determine type of scan
                data_type = "core"
                if "alpha" in data.dims:
                    data_type = "cut"
                if "beta" in data.dims:
                    data_type = "map"
                if "hv" in data.dims:
                    data_type = "hvdep"

                data_info.append(
                    [
                        name,
                        path,
                        datetime.datetime.fromisoformat(data.attrs["DateTime"]),
                        data_type,
                    ]
                )

                for k, v in attrs_mapping.items():
                    # Try to get the attribute from the data, then from the coordinates
                    try:
                        val = data.attrs[v]
                    except KeyError:
                        try:
                            val = data.coords[v].values
                            if val.size == 1:
                                val = val.item()
                        except KeyError:
                            val = ""

                    # Convert polarization values to human readable form
                    if k == "Polarization":
                        if np.iterable(val):
                            val = np.asarray(val).astype(int)
                        else:
                            val = [round(val)]
                        val = [
                            {0: "LH", 2: "LV", -1: "RC", 1: "LC"}.get(v, v) for v in val
                        ]
                        if len(val) == 1:
                            val = val[0]

                    data_info[-1].append(val)

                del data

            # Sort by time and set index
            return (
                pd.DataFrame(data_info, columns=column_names)
                .sort_values("Time")
                .set_index("File Name")
            )

    with erlab.io.loader_context("example", tmp_dir.name):
        erlab.io.load(1)

    with pytest.raises(
        FileNotFoundError, match="Directory some_nonexistent_dir not found"
    ):
        erlab.io.loaders.set_data_dir("some_nonexistent_dir")

    # Test if the reprs are working
    assert erlab.io.loaders.__repr__().startswith("Registered data loaders")
    assert erlab.io.loaders._repr_html_().startswith("<table><thead>")

    erlab.io.set_loader("example")
    erlab.io.set_data_dir(tmp_dir.name)
    erlab.io.load(1)
    erlab.io.load(2)

    df = erlab.io.summarize(display=False)
    assert len(df.index) == 2

    # Test if pretty printing works
    erlab.io.loaders.current_loader.get_styler(df)._repr_html_()

    # Interactive summary
    erlab.io.loaders.current_loader.isummarize(df)
