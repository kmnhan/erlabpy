import tempfile

import lmfit
import numpy as np
import pytest
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

from erlab.interactive.utils import (
    IconActionButton,
    IdentifierValidator,
    _TracebackDialog,
    load_fit_ui,
    save_fit_ui,
    show_traceback,
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
    accept_dialog(lambda: save_fit_ui(fit_result_ds), pre_call=_go_to_file)

    # Load fit
    accept_dialog(lambda: load_fit_ui(), pre_call=_go_to_file)

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
    state, _, _ = validator.validate(input_str, 0)
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


class CustomException(Exception):
    pass


@pytest.fixture
def raise_exception():
    def _func():
        raise CustomException("Dialog test error")

    return _func


@pytest.mark.parametrize(
    "informative_text", ["Additional info", ""], ids=["with_info", "no_info"]
)
def test_traceback_dialog_basic(qtbot, raise_exception, informative_text):
    # Simulate an exception for traceback
    try:
        raise_exception()
    except CustomException:
        parent = QtWidgets.QWidget()
        qtbot.addWidget(parent)
        parent.show()

        dialog = _TracebackDialog(
            parent=parent,
            title="Error Title",
            text="Error occurred",
            informative_text=informative_text,
            buttons=QtWidgets.QDialogButtonBox.StandardButton.Ok,
            icon_pixmap=QtWidgets.QStyle.StandardPixmap.SP_MessageBoxCritical,
        )
        qtbot.addWidget(dialog)
        dialog.show()

        assert dialog.windowTitle() == "Error Title"
        assert dialog._text_label.text() == "Error occurred"
        assert dialog._info_label.text() == informative_text
        # Details container should be hidden initially if details_visible
        assert not dialog._details_container.isVisible()

        dialog._details_toggle.setChecked(True)
        qtbot.wait_until(lambda: dialog._details_container.isVisible(), timeout=100)
        dialog._details_toggle.setChecked(False)
        qtbot.wait_until(lambda: not dialog._details_container.isVisible(), timeout=100)

        # Accept dialog
        dialog._button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).click()
        assert dialog.result() == QtWidgets.QDialog.Accepted


def test_traceback_dialog_no_exception(qtbot):
    # sys.exception() returns None, so details should be hidden
    parent = QtWidgets.QWidget()
    dialog = _TracebackDialog(
        parent=parent,
        title="No Exception",
        text="No error",
        informative_text="",
        buttons=QtWidgets.QDialogButtonBox.StandardButton.Ok,
        icon_pixmap=QtWidgets.QStyle.StandardPixmap.SP_MessageBoxInformation,
    )
    qtbot.addWidget(dialog)
    assert not dialog._details_toggle.isVisible()
    assert not dialog._details_container.isVisible()
    dialog._button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).click()
    assert dialog.result() == QtWidgets.QDialog.Accepted


def test_traceback_dialog_default_buttons_and_icon(qtbot, raise_exception):
    try:
        raise_exception()
    except CustomException:
        parent = QtWidgets.QWidget()
        dialog = _TracebackDialog(
            parent=parent,
            title="Default Buttons",
            text="Default test",
        )
        qtbot.addWidget(dialog)
        # Should default to Ok button and critical icon
        ok_btn = dialog._button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        assert ok_btn is not None
        ok_btn.click()
        assert dialog.result() == QtWidgets.QDialog.Accepted


def test_show_traceback(qtbot, raise_exception, accept_dialog):
    try:
        raise_exception()
    except CustomException:
        parent = QtWidgets.QWidget()
        qtbot.addWidget(parent)
        parent.show()

        def _call_show_traceback():
            show_traceback(
                parent=parent,
                title="Show Traceback",
                text="An error occurred",
                informative_text="Some info",
            )

        accept_dialog(_call_show_traceback)
