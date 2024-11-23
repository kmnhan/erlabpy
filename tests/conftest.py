import os
import pathlib
import time
from collections.abc import Callable

import lmfit
import numpy as np
import pooch
import pytest
import xarray as xr
from numpy.testing import assert_almost_equal
from qtpy import QtCore, QtWidgets

from erlab.io.exampledata import generate_data_angles, generate_gold_edge

DATA_COMMIT_HASH = "bd2c597a49dfbcb91961bef3dcf988179dbe1151"
"""The commit hash of the commit to retrieve from `kmnhan/erlabpy-data`."""

DATA_KNOWN_HASH = "434534c4e4d595aac073860289e2fcee09b31ca7655cb9b68a6143e34eecbae4"
"""The hash of the `.tar.gz` file."""


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
        temp=100, Eres=1e-2, nx=15, ny=150, edge_coeffs=(0.04, 1e-5, -3e-4), noise=False
    )


def _accept_dialog(
    dialog_trigger: Callable,
    time_out: int = 5,
    pre_call: Callable | None = None,
    accept_call: Callable | None = None,
) -> None:
    """Accept a dialog during testing.

    If there is no dialog, it waits until one is created for a maximum of 5 seconds (by
    default). Adapted from `this issue comment on pytest-qt
    <https://github.com/pytest-dev/pytest-qt/issues/256#issuecomment-1915675942>`_.

    Parameters
    ----------
    dialog_trigger
        Callable that triggers the dialog creation.
    time_out
        Maximum time (seconds) to wait for the dialog creation.
    pre_call
        Callable that takes the dialog as a single argument. If provided, it is executed
        before calling ``.accept()`` on the dialog.
    accept_call
        If provided, it is called instead of ``.accept()`` on the dialog.
    """
    dialog = None
    start_time = time.time()

    # Helper function to catch the dialog instance and hide it
    def dialog_creation():
        # Wait for the dialog to be created or timeout
        nonlocal dialog
        while dialog is None and time.time() - start_time < time_out:
            dialog = QtWidgets.QApplication.activeModalWidget()

        # Avoid errors when dialog is not created
        if isinstance(dialog, QtWidgets.QDialog):
            if pre_call is not None:
                pre_call(dialog)

            if accept_call is not None:
                accept_call(dialog)
            elif (
                isinstance(dialog, QtWidgets.QMessageBox)
                and dialog.defaultButton() is not None
            ):
                dialog.defaultButton().click()
            else:
                dialog.accept()

    # Create a thread to get the dialog instance and call dialog_creation trigger
    QtCore.QTimer.singleShot(1, dialog_creation)
    dialog_trigger()

    assert isinstance(
        dialog, QtWidgets.QDialog
    ), f"No dialog was created after {time_out} seconds. Dialog type: {type(dialog)}"


@pytest.fixture
def accept_dialog():
    return _accept_dialog


def _move_and_compare_values(qtbot, win, expected, cursor=0, target_win=None):
    if target_win is None:
        target_win = win
    assert_almost_equal(win.array_slicer.point_value(cursor), expected[0])

    # Move left
    qtbot.keyClick(
        target_win, QtCore.Qt.Key.Key_Left, QtCore.Qt.KeyboardModifier.ShiftModifier
    )
    assert_almost_equal(win.array_slicer.point_value(cursor), expected[1])

    # Move down
    qtbot.keyClick(
        target_win, QtCore.Qt.Key.Key_Down, QtCore.Qt.KeyboardModifier.ShiftModifier
    )
    assert_almost_equal(win.array_slicer.point_value(cursor), expected[2])

    # Move right
    qtbot.keyClick(
        target_win, QtCore.Qt.Key.Key_Right, QtCore.Qt.KeyboardModifier.ShiftModifier
    )
    assert_almost_equal(win.array_slicer.point_value(cursor), expected[3])

    # Move up
    qtbot.keyClick(
        target_win, QtCore.Qt.Key.Key_Up, QtCore.Qt.KeyboardModifier.ShiftModifier
    )
    assert_almost_equal(win.array_slicer.point_value(cursor), expected[0])


@pytest.fixture
def move_and_compare_values():
    return _move_and_compare_values
