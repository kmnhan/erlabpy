import pytest
from qtpy import QtWidgets

from erlab.interactive._options import OptionDialog, options
from erlab.interactive._options.schema import AppOptions


@pytest.fixture
def dialog(qtbot):
    dlg = OptionDialog()
    qtbot.addWidget(dlg)
    return dlg


def test_dialog_initial_settings(dialog: OptionDialog):
    # Should match current OptionManager settings
    assert dialog.current_options.model_dump() == options.model.model_dump()


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


def test_dialog_parameter_tips_apply_to_label_and_row_widgets(
    dialog: OptionDialog,
):
    param = dialog.tree.parameter.child("io").child("dask").child("compute_threshold")
    tip = param.opts["tip"]
    (item,) = tuple(param.items)

    assert item.toolTip(0) == tip
    assert item.toolTip(1) == tip
    assert item.layoutWidget.toolTip() == tip
    assert item.displayLabel.toolTip() == tip
    assert item.widget.toolTip() == tip


def test_apply_saves_settings(dialog: OptionDialog, qtbot):
    dialog.tree.parameter.child("colors").child("cmap").child("name").setValue("bwr")
    dialog.apply()

    # Test with igor colormap that requires combobox repopulation
    dialog.tree.parameter.child("colors").child("cmap").child("name").setValue(
        "RainbowLight"
    )
    dialog.tree.parameter.child("colors").child("cmap").child("reverse").setValue(True)
    dialog.apply()

    assert options.model.colors.cmap.name == "RainbowLight"
    assert options.model.colors.cmap.reverse
    assert not dialog.modified

    options.restore()  # Reset settings after test


def test_restore_defaults(dialog: OptionDialog, qtbot):
    # Change a value
    param = dialog.tree.parameter.child("colors").child("cmap").child("name")
    param.setValue("bwr")
    dialog.restore()
    # Should restore to DEFAULT_OPTIONS
    assert dialog.current_options.model_dump() == AppOptions().model_dump()

    # Accept dialog to save changes
    dialog.accept()


def test_reject_with_modifications(dialog: OptionDialog, qtbot, accept_dialog):
    # Reset to defaults first
    dialog.restore()
    dialog.apply()
    assert options.model.colors.cmap.name != "bwr"

    # Change a value
    param = dialog.tree.parameter.child("colors").child("cmap").child("name")
    param.setValue("bwr")
    # Patch QMessageBox to simulate user clicking "No"

    def _msgbox_close(dialog: QtWidgets.QMessageBox):
        dialog.button(QtWidgets.QMessageBox.StandardButton.Cancel).click()

    accept_dialog(dialog.reject, accept_call=_msgbox_close)

    # Dialog should remain open, and changes should not be saved
    assert dialog.current_options.colors.cmap.name == "bwr"
    assert options.model.model_dump() == AppOptions().model_dump()

    def _msgbox_savechanges(dialog: QtWidgets.QMessageBox):
        dialog.button(QtWidgets.QMessageBox.StandardButton.Ok).click()

    accept_dialog(dialog.reject, accept_call=_msgbox_savechanges)

    # Changes should be saved now
    assert options.model.colors.cmap.name == "bwr"

    options.restore()  # Reset settings after test


def test_options_get_set():
    options.restore()

    # Check initial values
    assert options["colors/cmap/name"] == AppOptions().colors.cmap.name
    assert options["io/workspace/compress"] is True
    assert options["io/workspace/use_incremental"] is True
    assert options["io/workspace/incremental_save_on_remote"] is False

    # Set a new value
    options["colors/cmap/name"] = "viridis"
    options["io/workspace/compress"] = False
    options["io/workspace/use_incremental"] = False
    options["io/workspace/incremental_save_on_remote"] = True

    # Check if the value was set correctly
    assert options["colors/cmap/name"] == "viridis"
    assert options["io/workspace/compress"] is False
    assert options["io/workspace/use_incremental"] is False
    assert options["io/workspace/incremental_save_on_remote"] is True
    assert not options.model.io.workspace.compress
    assert not options.model.io.workspace.use_incremental
    assert options.model.io.workspace.incremental_save_on_remote

    options.restore()  # Reset settings after test
