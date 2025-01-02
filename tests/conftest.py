import logging
import os
import pathlib
import threading
import time
from collections.abc import Callable, Sequence

import lmfit
import numpy as np
import pooch
import pytest
import xarray as xr
from numpy.testing import assert_almost_equal
from qtpy import QtCore, QtWidgets

from erlab.interactive.utils import _WaitDialog
from erlab.io.exampledata import generate_data_angles, generate_gold_edge

DATA_COMMIT_HASH = "9408f73f3562a5c1e5f6e01dec25bcd16832264e"
"""The commit hash of the commit to retrieve from `kmnhan/erlabpy-data`."""

DATA_KNOWN_HASH = "75b31cd538ea4847c6eb34017f5d69bed324081329fcc0eece5089677e37df4f"
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


@pytest.fixture
def fit_test_darr():
    t = np.arange(0, 5, 0.5)
    da = xr.DataArray(
        np.stack([_exp_decay(t, 3, 3), _exp_decay(t, 5, 4), np.nan * t], axis=-1),
        dims=("t", "x"),
        coords={"t": t, "x": [0, 1, 2]},
    )
    da[0, 0] = np.nan
    return da


@pytest.fixture
def anglemap():
    return generate_data_angles(shape=(10, 10, 10), assign_attributes=True)


@pytest.fixture
def gold():
    return generate_gold_edge(
        (15, 150), temp=100, Eres=1e-2, edge_coeffs=(0.04, 1e-5, -3e-4), noise=False
    )


class _DialogDetectionThread(QtCore.QThread):
    sigUpdate = QtCore.Signal(object)
    sigTimeout = QtCore.Signal(int)
    sigTrigger = QtCore.Signal(int, object)
    sigPreCall = QtCore.Signal(int, object)

    def __init__(
        self,
        index: int,
        pre_call: Callable | None,
        timeout: float,
    ) -> None:
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

        self._mutex = QtCore.QMutex()

        dialog = None

        log.debug("handling dialog %d", self.index)
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

        log.debug("emitting trigger for %d", self.index)
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
        log.debug("trigger index %d", index)

        if index <= self._max_index:
            if hasattr(self, "_handler") and self._handler.isRunning():
                self._handler.wait()

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


def _move_and_compare_values(qtbot, win, expected, cursor=0, target_win=None):
    if target_win is None:
        target_win = win
    with qtbot.waitExposed(win):
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
    qtbot.waitUntil(
        lambda: win.slicer_area.get_current_index(x_ax) == x0 - 1, timeout=2000
    )
    assert_almost_equal(win.array_slicer.point_value(cursor), expected[1])

    # Move down
    win.slicer_area.step_index(y_ax, -1)
    qtbot.waitUntil(
        lambda: win.slicer_area.get_current_index(y_ax) == y0 - 1, timeout=2000
    )
    assert_almost_equal(win.array_slicer.point_value(cursor), expected[2])

    # Move right
    win.slicer_area.step_index(x_ax, 1)
    qtbot.waitUntil(lambda: win.slicer_area.get_current_index(x_ax) == x0, timeout=2000)
    assert_almost_equal(win.array_slicer.point_value(cursor), expected[3])

    # Move up
    win.slicer_area.step_index(y_ax, 1)
    qtbot.waitUntil(lambda: win.slicer_area.get_current_index(y_ax) == y0, timeout=2000)
    assert_almost_equal(win.array_slicer.point_value(cursor), expected[0])


@pytest.fixture
def move_and_compare_values():
    return _move_and_compare_values
