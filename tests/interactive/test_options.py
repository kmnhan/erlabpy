import pytest
from qtpy import QtWidgets

from erlab.interactive._options import OptionDialog, options
from erlab.interactive._options.defaults import DEFAULT_OPTIONS


@pytest.fixture
def dialog(qtbot):
    dlg = OptionDialog()
    qtbot.addWidget(dlg)
    return dlg


def test_dialog_initial_settings(dialog: OptionDialog):
    # Should match current OptionManager settings
    assert dialog.current_options == options.option_dict


def test_dialog_modified_property(dialog: OptionDialog):
    # Should be False initially
    assert not dialog.modified
    # Change a value in the parameter tree
    param = dialog.tree.parameter.child("colors").child("cmap").child("name")
    param.setValue("bwr")
    assert dialog.modified


def test_apply_button_enables_on_change(dialog: OptionDialog, qtbot):
    param = dialog.tree.parameter.child("colors").child("cmap").child("name")
    param.setValue("bwr")
    assert dialog.btn_apply.isEnabled()


def test_apply_saves_settings(dialog: OptionDialog, qtbot):
    param = dialog.tree.parameter.child("colors").child("cmap").child("name")
    param.setValue("bwr")
    dialog.apply()
    assert options.option_dict["colors"]["cmap"]["name"] == "bwr"
    assert not dialog.modified


def test_restore_defaults(dialog: OptionDialog, qtbot):
    # Change a value
    param = dialog.tree.parameter.child("colors").child("cmap").child("name")
    param.setValue("bwr")
    dialog.restore()
    # Should restore to DEFAULT_OPTIONS
    assert dialog.current_options == DEFAULT_OPTIONS


def test_reject_with_modifications(dialog: OptionDialog, qtbot, accept_dialog):
    # Reset to defaults first
    dialog.restore()
    dialog.apply()
    assert options.option_dict["colors"]["cmap"]["name"] != "bwr"

    # Change a value
    param = dialog.tree.parameter.child("colors").child("cmap").child("name")
    param.setValue("bwr")
    # Patch QMessageBox to simulate user clicking "No"

    def _msgbox_close(dialog: QtWidgets.QMessageBox):
        dialog.button(QtWidgets.QMessageBox.StandardButton.Cancel).click()

    _h0 = accept_dialog(dialog.reject, accept_call=_msgbox_close)

    # Dialog should remain open, and changes should not be saved
    assert dialog.current_options["colors"]["cmap"]["name"] == "bwr"
    assert options.option_dict == DEFAULT_OPTIONS

    def _msgbox_savechanges(dialog: QtWidgets.QMessageBox):
        dialog.button(QtWidgets.QMessageBox.StandardButton.Ok).click()

    _h1 = accept_dialog(dialog.reject, accept_call=_msgbox_savechanges)

    # Changes should be saved now
    assert options.option_dict["colors"]["cmap"]["name"] == "bwr"

    # Reset settings before next test
    options.restore()
