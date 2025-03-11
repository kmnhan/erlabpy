import csv
import datetime
import errno
import os
import pathlib
import re
import tempfile
import typing

import numpy as np
import pytest
import xarray as xr
from qtpy import QtCore

import erlab
from erlab.io.dataloader import LoaderBase, UnsupportedFileError
from erlab.io.exampledata import generate_data_angles

if typing.TYPE_CHECKING:
    from erlab.interactive.explorer import _DataExplorer


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
        f"{darr.attrs['Date']} {darr.attrs['Time']}", r"%d/%m/%Y %I:%M:%S %p"
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


def test_loader(qtbot, accept_dialog) -> None:
    # Create a temporary directory
    tmp_dir = tempfile.TemporaryDirectory()

    # Generate a map
    beta_coords = np.linspace(2, 7, 10)

    # Generate and save cuts with different beta values
    data_2d = []
    for i, beta in enumerate(beta_coords):
        data = make_data(beta=beta, temp=20.0 + i, hv=50.0)
        filename = f"{tmp_dir.name}/data_001_S{str(i + 1).zfill(3)}.h5"
        data.to_netcdf(filename, engine="h5netcdf")
        data_2d.append(data)

    data_2d = xr.concat(data_2d, dim="Polar")

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

    # Save map data
    data_2d.to_netcdf(f"{tmp_dir.name}/data_006.h5", engine="h5netcdf")

    # Save XPS data
    data_2d.isel(Polar=0, ThetaX=0).to_netcdf(
        f"{tmp_dir.name}/data_007.h5", engine="h5netcdf"
    )

    # Save data with wrong file extension
    wrong_file: str = f"{tmp_dir.name}/data_010.nc"
    data.to_netcdf(wrong_file, engine="h5netcdf")

    class ExampleLoader(LoaderBase):
        name = "example"
        description = "Example loader for testing purposes"
        extensions: typing.ClassVar[set[str]] = {".h5"}

        name_map: typing.ClassVar[dict] = {
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

        additional_attrs: typing.ClassVar[dict] = {
            "configuration": 1,  # Experimental geometry required for kspace conversion
            "sample_workfunction": 4.3,
        }  # Any additional metadata you want to add to the data

        formatters: typing.ClassVar[dict] = {
            "polarization": _format_polarization,
            "LensMode": lambda x: x.replace("Angular", "A"),
        }

        summary_attrs: typing.ClassVar[dict] = {
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
                    name=darr.name,
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

        @property
        def file_dialog_methods(self):
            return {"Example Raw Data (*.h5)": (self.load, {})}

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
        ).replace("some_nonexistent_dir", ".*some_nonexistent_dir"),
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

    # Loading nonexistent indices
    nonexistent_file = "data_099.h5"
    with pytest.raises(
        FileNotFoundError,
        match=re.escape(
            str(
                FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), nonexistent_file
                )
            )
        ).replace(re.escape(nonexistent_file), f".*{re.escape(nonexistent_file)}"),
    ):
        erlab.io.load("data_099.h5")

    with pytest.raises(
        UnsupportedFileError,
        match=UnsupportedFileError._make_msg(
            "example",
            pathlib.Path(wrong_file),
            erlab.io.loaders.example.extensions,
        ),
    ):
        erlab.io.load(wrong_file, single=True)

    # Test if coordinate_attrs are correctly assigned
    mfdata = erlab.io.load(1)
    assert np.allclose(mfdata["sample_temp"].values, 20.0 + np.arange(len(beta_coords)))

    df = erlab.io.summarize(display=False)
    assert len(df.index) == 7

    # Test if pretty printing works
    erlab.io.loaders.current_loader.get_styler(df)._repr_html_()

    # Interactive summary
    box = erlab.io.loaders.current_loader._isummarize(df)
    btn_box = box.children[0].children[0]
    assert len(btn_box.children) == 3  # prev, next, load full
    btn_box.children[2].click()  # load full
    btn_box.children[1].click()  # next
    btn_box.children[1].click()  # next
    del box, btn_box

    # Interactive summary with imagetool manager
    erlab.interactive.imagetool.manager.main(execute=False)
    manager = erlab.interactive.imagetool.manager._manager_instance
    qtbot.addWidget(manager)
    with qtbot.waitExposed(manager):
        manager.show()
        manager.activateWindow()

    box = erlab.io.loaders.current_loader._isummarize(df)
    btn_box = box.children[0].children[0]
    assert len(btn_box.children) == 4  # prev, next, load full, imagetool
    btn_box.children[3].click()  # imagetool

    qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
    manager.remove_tool(0)
    qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

    del box, btn_box

    # Test data explorer

    # Initialize data explorer
    manager.ensure_explorer_initialized()
    assert hasattr(manager, "explorer")
    explorer: _DataExplorer = manager.explorer

    # Set the recent directory and name filter
    manager._recent_directory = tmp_dir.name
    manager._recent_name_filter = next(
        iter(erlab.io.loaders["example"].file_dialog_methods.keys())
    )

    # Show data explorer
    manager.show_explorer()
    qtbot.wait_exposed(explorer)

    # Enable preview
    explorer._preview_check.setChecked(True)

    assert explorer.loader_name == "example"
    assert explorer._fs_model.file_system.path == pathlib.Path(tmp_dir.name)

    # Reload folder
    explorer._reload_act.trigger()

    # Set show hidden files
    explorer._tree_view.model().set_show_hidden(False)

    # Sort by name
    explorer._tree_view.sortByColumn(0, QtCore.Qt.SortOrder.DescendingOrder)

    def select_files(indices: list[int], deselect: bool = False) -> None:
        selection_model = explorer._tree_view.selectionModel()

        for index in indices:
            idx_start = explorer._tree_view.model().index(index, 0)
            idx_end = explorer._tree_view.model().index(
                index, explorer._tree_view.model().columnCount() - 1
            )
            selection_model.select(
                QtCore.QItemSelection(idx_start, idx_end),
                QtCore.QItemSelectionModel.SelectionFlag.Deselect
                if deselect
                else QtCore.QItemSelectionModel.SelectionFlag.Select,
            )
            if deselect:
                qtbot.wait_until(
                    lambda idx=idx_end: idx not in explorer._tree_view.selectedIndexes()
                )
            else:
                qtbot.wait_until(
                    lambda idx=idx_end: idx in explorer._tree_view.selectedIndexes()
                )

    assert explorer._text_edit.toPlainText() == explorer.TEXT_NONE_SELECTED

    # Multiple selection
    select_files([2, 3, 4])

    # Show multiple in manager
    explorer.to_manager()

    # Clear selection
    select_files([2, 3, 4], deselect=True)

    # Single selection
    select_files([2])

    # Show single in manager
    explorer.to_manager()

    qtbot.wait_until(lambda: manager.ntools == 4)

    # Reload data in manager
    qmodelindex = manager.list_view._model._row_index(3)
    manager.list_view.selectionModel().select(
        QtCore.QItemSelection(qmodelindex, qmodelindex),
        QtCore.QItemSelectionModel.SelectionFlag.Select,
    )
    with qtbot.wait_signal(manager.get_tool(3).slicer_area.sigDataChanged):
        manager.reload_action.trigger()
    qtbot.wait(100)

    # Clear selection
    select_files([2], deselect=True)

    # Single selection multiple times
    for i in range(1, 5):
        with qtbot.wait_signal(explorer._preview._sigDataChanged, timeout=2000):
            select_files([i])
        select_files([i], deselect=True)

    # Test sorting by different columns
    for i in range(4):
        explorer._tree_view.sortByColumn(i, QtCore.Qt.SortOrder.AscendingOrder)

    # # Trigger open in file explorer
    # explorer._finder_act.trigger()

    explorer.close()

    # Close imagetool manager
    _handler = accept_dialog(manager.close)
    erlab.interactive.imagetool.manager._manager_instance = None

    # Cleanup the temporary directory
    tmp_dir.cleanup()
