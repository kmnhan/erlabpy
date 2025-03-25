import csv
import datetime
import functools
import logging
import os
import pathlib
import re
import sys
import threading
import time
import typing
from collections.abc import Callable, Sequence

import lmfit
import numpy as np
import pooch
import pytest
import xarray as xr
from numpy.testing import assert_almost_equal
from qtpy import QtCore, QtWidgets

import erlab
from erlab.interactive.utils import _WaitDialog
from erlab.io.dataloader import LoaderBase
from erlab.io.exampledata import generate_data_angles, generate_gold_edge

DATA_COMMIT_HASH = "19c743559a3008e0cb74f5c8e2ef87334b0e7dc1"
"""The commit hash of the commit to retrieve from `kmnhan/erlabpy-data`."""

DATA_KNOWN_HASH = "5c60d819a8dd740121cb5e2c6170bc0ef38e31a753fe57f6a8a7056ed7359418"
"""The SHA-256 checksum of the `.tar.gz` file."""

log = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def test_data_dir() -> pathlib.Path:
    path = os.getenv("ERLAB_TEST_DATA_DIR", None)
    if path is None:
        cache_folder = pooch.os_cache("erlabpy")
        pooch.retrieve(
            "https://api.github.com/repos/kmnhan/erlabpy-data/tarball/"
            + DATA_COMMIT_HASH,
            known_hash=DATA_KNOWN_HASH,
            path=cache_folder,
            processor=pooch.Untar(extract_dir=""),
        )
        path = cache_folder / f"kmnhan-erlabpy-data-{DATA_COMMIT_HASH[:7]}"

    return pathlib.Path(path)


def _exp_decay(t, n0, tau=1):
    return n0 * np.exp(-t / tau)


@pytest.fixture
def exp_decay_model():
    return lmfit.Model(_exp_decay)


@pytest.fixture(scope="session")
def fit_test_darr():
    t = np.arange(0, 5, 0.5)
    da = xr.DataArray(
        np.stack([_exp_decay(t, 3, 3), _exp_decay(t, 5, 4), np.nan * t], axis=-1),
        dims=("t", "x"),
        coords={"t": t, "x": [0, 1, 2]},
    )
    da[0, 0] = np.nan
    return da


@pytest.fixture(scope="session")
def anglemap():
    return generate_data_angles(shape=(10, 10, 10), assign_attributes=True)


@pytest.fixture(scope="session")
def gold():
    return generate_gold_edge(
        (15, 150), temp=100, Eres=1e-2, edge_coeffs=(0.04, 1e-5, -3e-4), noise=False
    )


class _DialogDetectionThread(QtCore.QThread):
    sigUpdate = QtCore.Signal(object)
    sigTimeout = QtCore.Signal(int)
    sigTrigger = QtCore.Signal(int, object)
    sigPreCall = QtCore.Signal(int, object)

    def __init__(self, index: int, pre_call: Callable | None, timeout: float) -> None:
        super().__init__()
        self.pre_call = pre_call
        self.index = index
        self.timeout = timeout
        self._precall_called = threading.Event()

    def precall_called(self):
        if self.isRunning():
            self.mutex.lock()
        self._precall_called.set()
        if self.isRunning():
            self.mutex.unlock()

    def run(self):
        self.mutex = QtCore.QMutex()
        time.sleep(0.001)
        start_time = time.perf_counter()

        dialog = None

        log.debug("looking for dialog %d...", self.index)
        while (
            dialog is None or isinstance(dialog, _WaitDialog)
        ) and time.perf_counter() - start_time < self.timeout:
            dialog = QtWidgets.QApplication.activeModalWidget()
            time.sleep(0.01)

        if dialog is None or isinstance(dialog, _WaitDialog):
            log.debug("emitting timeout %d", self.index)
            self.sigTimeout.emit(self.index)
            return

        log.debug("dialog %d detected: %s", self.index, dialog)

        if self.pre_call is not None:
            log.debug("pre_call %d...", self.index)
            self.sigPreCall.emit(self.index, dialog)
            while not self._precall_called.is_set():
                time.sleep(0.01)
            log.debug("pre_call %d done", self.index)

        log.debug("emitting trigger for %d", self.index + 1)
        self.sigTrigger.emit(self.index + 1, dialog)


class _DialogHandler(QtCore.QObject):
    """Accept a dialog during testing.

    If there is no dialog, it waits until one is created for a maximum of 5 seconds (by
    default). Adapted from `this issue comment on pytest-qt
    <https://github.com/pytest-dev/pytest-qt/issues/256#issuecomment-1915675942>`_.

    Parameters
    ----------
    dialog_trigger
        Callable that triggers the dialog creation.
    timeout
        Maximum time (seconds) to wait for the dialog creation.
    pre_call
        Callable that takes the dialog as a single argument. If provided, it is executed
        before calling ``.accept()`` on the dialog. If a sequence of callables of length
        equal to ``chained_dialog`` is provided, each callable will be called before
        each dialog is accepted.
    accept_call
        If provided, it is called instead of ``.accept()`` on the dialog. If a sequence
        of callables of length equal to ``chained_dialog`` is provided, each callable
        will be called instead of ``.accept()`` on each dialog.
    chained_dialog
        If 2, a new dialog is expected to be created right after the dialog is accepted.
        The new dialog will also be accepted. Numbers greater than 1 will accept
        multiple dialogs.
    """

    def __init__(
        self,
        dialog_trigger: Callable,
        timeout: float = 5.0,
        pre_call: Callable | Sequence[Callable | None] | None = None,
        accept_call: Callable | Sequence[Callable | None] | None = None,
        chained_dialogs: int = 1,
    ):
        super().__init__()

        self.timeout: float = timeout
        self._timed_out = False

        if not isinstance(pre_call, Sequence):
            pre_call = [pre_call] + [None] * (chained_dialogs - 1)

        if not isinstance(accept_call, Sequence):
            accept_call = [accept_call] + [None] * (chained_dialogs - 1)
        self._pre_call_list = pre_call
        self._accept_call_list = accept_call
        self._max_index = chained_dialogs - 1

        self.trigger_index(0, dialog_trigger)

    @QtCore.Slot(int)
    def _timeout(self, index: int) -> None:
        log.debug("timeout %d", index)
        self._timed_out = True
        if hasattr(self, "_handler") and self._handler.isRunning():
            self._handler.wait()
            self._handler = None

        pytest.fail(
            f"No dialog for index {index} was created after {self.timeout} seconds."
        )

    @QtCore.Slot(int, object)
    def trigger_index(
        self, index: int, dialog_or_trigger: QtWidgets.QDialog | Callable
    ) -> None:
        """
        Trigger the dialog creation.

        Parameters
        ----------
        index
            The index of the dialog to trigger, starting from 0. If the index is greater
            than 0, ``dialog_or_trigger`` should be a dialog.
        dialog_or_trigger
            The callable that triggers the dialog creation or a prviously created dialog
            which will create the next dialog upon acceptance.
        """
        log.debug("index %d triggered", index)

        if index <= self._max_index:
            if hasattr(self, "_handler") and self._handler.isRunning():
                self._handler.wait()
                self._handler = None

            self._handler = _DialogDetectionThread(
                index, self._pre_call_list[index], self.timeout
            )
            self._handler.sigTimeout.connect(self._timeout)
            self._handler.sigTrigger.connect(self.trigger_index)
            self._handler.sigPreCall.connect(self.handle_pre_call)
            self._handler.start()

        if isinstance(dialog_or_trigger, QtWidgets.QDialog):
            accept_call = self._accept_call_list[index - 1]

            if accept_call is not None:
                accept_call(dialog_or_trigger)
            else:
                if (
                    isinstance(dialog_or_trigger, QtWidgets.QMessageBox)
                    and dialog_or_trigger.defaultButton() is not None
                ):
                    dialog_or_trigger.defaultButton().click()
                else:
                    dialog_or_trigger.accept()
            log.debug("finished %d", index - 1)

        else:
            dialog_or_trigger()

    @QtCore.Slot(int, object)
    def handle_pre_call(self, index: int, dialog: QtWidgets.QDialog) -> None:
        log.debug("pre-call callable received")
        self._pre_call_list[index](dialog)
        log.debug("pre-call successfully called")
        self._handler.precall_called()


@pytest.fixture
def accept_dialog():
    return _DialogHandler


def _move_and_compare_values(bot, win, expected, cursor=0, target_win=None):
    if target_win is None:
        target_win = win
    with bot.waitExposed(win):
        target_win.show()
        target_win.activateWindow()
        target_win.setFocus()

    assert_almost_equal(win.array_slicer.point_value(cursor), expected[0])

    x_ax = win.slicer_area.main_image.display_axis[0]
    y_ax = win.slicer_area.main_image.display_axis[1]

    x0, y0 = (
        win.slicer_area.get_current_index(x_ax),
        win.slicer_area.get_current_index(y_ax),
    )

    # Move left
    win.slicer_area.step_index(x_ax, -1)
    bot.waitUntil(
        lambda: win.slicer_area.get_current_index(x_ax) == x0 - 1, timeout=2000
    )
    assert_almost_equal(win.array_slicer.point_value(cursor), expected[1])

    # Move down
    win.slicer_area.step_index(y_ax, -1)
    bot.waitUntil(
        lambda: win.slicer_area.get_current_index(y_ax) == y0 - 1, timeout=2000
    )
    assert_almost_equal(win.array_slicer.point_value(cursor), expected[2])

    # Move right
    win.slicer_area.step_index(x_ax, 1)
    bot.waitUntil(lambda: win.slicer_area.get_current_index(x_ax) == x0, timeout=2000)
    assert_almost_equal(win.array_slicer.point_value(cursor), expected[3])

    # Move up
    win.slicer_area.step_index(y_ax, 1)
    bot.waitUntil(lambda: win.slicer_area.get_current_index(y_ax) == y0, timeout=2000)
    assert_almost_equal(win.array_slicer.point_value(cursor), expected[0])


@pytest.fixture
def move_and_compare_values():
    return _move_and_compare_values


@pytest.fixture(autouse=True)
def cover_qthreads(monkeypatch, qtbot):
    # https://github.com/nedbat/coveragepy/issues/686#issuecomment-2286288111
    from qtpy.QtCore import QThread

    base_constructor = QThread.__init__

    def run_with_trace(self):  # pragma: no cover
        if "coverage" in sys.modules:
            # https://github.com/nedbat/coveragepy/issues/686#issuecomment-634932753
            sys.settrace(threading._trace_hook)
        self._base_run()

    def init_with_trace(self, *args, **kwargs):
        base_constructor(self, *args, **kwargs)
        self._base_run = self.run
        self.run = functools.partial(run_with_trace, self)

    monkeypatch.setattr(QThread, "__init__", init_with_trace)


@pytest.fixture(autouse=True)
def cover_qthreadpool(monkeypatch, qtbot):
    # https://github.com/nedbat/coveragepy/issues/686#issuecomment-2435049275
    from qtpy.QtCore import QThreadPool

    base_constructor = QThreadPool.globalInstance().start

    def run_with_trace(self):  # pragma: no cover
        if "coverage" in sys.modules:
            # https://github.com/nedbat/coveragepy/issues/686#issuecomment-634932753
            sys.settrace(threading._trace_hook)
        self._base_run()

    def _start(worker, *args, **kwargs):
        worker._base_run = worker.run
        worker.run = functools.partial(run_with_trace, worker)
        return base_constructor(worker, *args, **kwargs)

    monkeypatch.setattr(QThreadPool.globalInstance(), "start", _start)


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


@pytest.fixture(scope="session")
def example_data_dir(tmp_path_factory) -> pathlib.Path:
    tmp_dir: pathlib.Path = tmp_path_factory.mktemp("example_data")

    # Generate a map
    beta_coords = np.linspace(2, 7, 10)

    # Generate and save cuts with different beta values
    data_2d = []
    for i, beta in enumerate(beta_coords):
        data = make_data(beta=beta, temp=20.0 + i, hv=50.0)
        filename = tmp_dir / f"data_001_S{str(i + 1).zfill(3)}.h5"
        data.to_netcdf(filename, engine="h5netcdf")
        data_2d.append(data)

    data_2d = xr.concat(data_2d, dim="Polar")

    # Write scan coordinates to a csv file
    with open(tmp_dir / "data_001_axis.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Index", "Polar"])

        for i, beta in enumerate(beta_coords):
            writer.writerow([i + 1, beta])

    # Generate some cuts with different band shifts
    for i in range(4):
        data = make_data(beta=5.0, temp=20.0, hv=50.0, bandshift=-i * 0.05)
        filename = tmp_dir / f"data_{str(i + 2).zfill(3)}.h5"
        data.to_netcdf(filename, engine="h5netcdf")

    # Save map data
    data_2d.to_netcdf(tmp_dir / "data_006.h5", engine="h5netcdf")

    # Save XPS data
    data_2d.isel(Polar=0, ThetaX=0).to_netcdf(
        tmp_dir / "data_007.h5", engine="h5netcdf"
    )

    # Save data with wrong file extension
    wrong_file = tmp_dir / "data_010.nc"
    data.to_netcdf(wrong_file, engine="h5netcdf")

    return tmp_dir


@pytest.fixture(scope="session")
def example_loader():
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

        summary_sort = "File Name"

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

    return ExampleLoader
