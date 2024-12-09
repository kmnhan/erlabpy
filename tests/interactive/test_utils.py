import pytest
from qtpy import QtCore, QtGui

from erlab.interactive.utils import IconActionButton


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
