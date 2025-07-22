import tempfile

import lmfit
import numpy as np
import pytest
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

from erlab.interactive.utils import (
    IconActionButton,
    IdentifierValidator,
    load_fit_ui,
    save_fit_ui,
)


@pytest.fixture
def action():
    action = QtGui.QAction("Test Action")
    action.setCheckable(True)
    return action


def test_icon_action_button_initialization(qtbot, action) -> None:
    button = IconActionButton(action, on="mdi6.plus", off="mdi6.minus")
    assert button._action == action
    assert button.icon_key_on == "mdi6.plus"
    assert button.icon_key_off == "mdi6.minus"
    assert button.isCheckable()


def test_icon_action_button_set_action(qtbot, action) -> None:
    button = IconActionButton(action, on="mdi6.plus", off="mdi6.minus")
    new_action = QtGui.QAction("New Action")
    new_action.setCheckable(True)
    button.setAction(new_action)
    assert button._action == new_action


def test_icon_action_button_update_from_action(qtbot, action) -> None:
    button = IconActionButton(action, on="mdi6.plus", off="mdi6.minus")
    action.setText("Updated Action")
    action.setEnabled(False)
    action.setChecked(True)
    action.setToolTip("Updated Tooltip")
    assert not button.isEnabled()
    assert button.isChecked()
    assert button.toolTip() == "Updated Tooltip"


def test_icon_action_button_click(qtbot, action) -> None:
    button = IconActionButton(action, on="mdi6.plus", off="mdi6.minus")
    qtbot.addWidget(button)
    qtbot.mouseClick(button, QtCore.Qt.LeftButton)
    assert action.isChecked()


def test_icon_action_button_toggle(qtbot, action) -> None:
    button = IconActionButton(action, on="mdi6.plus", off="mdi6.minus")
    qtbot.addWidget(button)
    qtbot.mouseClick(button, QtCore.Qt.LeftButton)
    assert button.isChecked()
    qtbot.mouseClick(button, QtCore.Qt.LeftButton)
    assert not button.isChecked()


@pytest.fixture
def fit_result_ds():
    rng = np.random.default_rng(1)
    xvals = np.linspace(0, 20, 100)
    yvals = 2 * np.sin(xvals) + 0.1 * rng.normal(size=xvals.shape)
    test_data = xr.DataArray(yvals, dims=["x"], coords={"x": xvals})

    def model_func(x, a, b):
        return a * np.sin(x) + b

    model = lmfit.Model(model_func, independent_vars=["x"])
    return test_data.xlm.modelfit("x", model=model, params={"a": 2, "b": 0})


@pytest.mark.parametrize("file_ext", ["nc", "h5"])
def test_save_fit_ui(qtbot, accept_dialog, fit_result_ds, file_ext):
    tmp_dir = tempfile.TemporaryDirectory()
    filename = f"{tmp_dir.name}/fit_save.{file_ext}"

    def _go_to_file(dialog: QtWidgets.QFileDialog):
        dialog.setDirectory(tmp_dir.name)
        dialog.selectFile(filename)
        focused = dialog.focusWidget()
        if isinstance(focused, QtWidgets.QLineEdit):
            focused.setText(f"fit_save.{file_ext}")

    # Save fit
    _handler_save = accept_dialog(
        lambda: save_fit_ui(fit_result_ds), pre_call=_go_to_file
    )

    # Load fit
    _handler_load = accept_dialog(lambda: load_fit_ui(), pre_call=_go_to_file)

    tmp_dir.cleanup()


@pytest.mark.parametrize(
    ("input_str", "expected_state"),
    [
        ("valid_name", QtGui.QValidator.State.Acceptable),
        ("", QtGui.QValidator.State.Intermediate),
        ("1invalid", QtGui.QValidator.State.Invalid),
        ("with space", QtGui.QValidator.State.Invalid),
        ("for", QtGui.QValidator.State.Intermediate),  # Python keyword
        ("_hidden", QtGui.QValidator.State.Acceptable),
        ("invalid-char!", QtGui.QValidator.State.Invalid),
        (None, QtGui.QValidator.State.Intermediate),
    ],
)
def test_identifier_validator_validate(input_str, expected_state):
    validator = IdentifierValidator()
    state, out_str, pos = validator.validate(input_str, 0)
    assert state == expected_state


@pytest.mark.parametrize(
    ("input_str", "expected"),
    [
        ("valid_name", "valid_name"),
        ("1invalid", "_1invalid"),
        ("with space", "with_space"),
        ("for", "for_"),
        ("", "var"),
        ("___", "var"),
        ("!@#", "var"),
        ("class", "class_"),
        (None, "var"),
    ],
)
def test_identifier_validator_fixup(input_str, expected):
    validator = IdentifierValidator()
    assert validator.fixup(input_str) == expected
