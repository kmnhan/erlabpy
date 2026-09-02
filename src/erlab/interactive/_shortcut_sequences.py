"""Shared shortcut sequences for interactive applications."""

from __future__ import annotations

import sys

from qtpy import QtGui

WINDOW_CLOSE = "Ctrl+W"
WORKSPACE_SAVE = QtGui.QKeySequence.StandardKey.Save

MANAGER_OPEN_WORKSPACE = QtGui.QKeySequence.StandardKey.Open
MANAGER_SAVE_WORKSPACE = WORKSPACE_SAVE
MANAGER_SAVE_WORKSPACE_AS = QtGui.QKeySequence.StandardKey.SaveAs
MANAGER_WORKSPACE_PROPERTIES = "Alt+Return"
MANAGER_DATA_EXPLORER = "Ctrl+E"
MANAGER_PERIODIC_TABLE = "Ctrl+Shift+P"
MANAGER_HIDE_WINDOWS = WINDOW_CLOSE
MANAGER_REMOVE_WINDOWS = QtGui.QKeySequence.StandardKey.Delete
MANAGER_RELOAD_DATA = QtGui.QKeySequence.StandardKey.Refresh
MANAGER_LINK_WINDOWS = "Ctrl+L"
MANAGER_UNLINK_WINDOWS = "Ctrl+Shift+L"
MANAGER_CONSOLE = "Ctrl+J"
MANAGER_SETTINGS = QtGui.QKeySequence.StandardKey.Preferences


def manager_rename_selection() -> tuple[str, ...]:
    if sys.platform == "darwin":
        return ("Return", "Enter")
    return ("F2",)


def manager_show_selection() -> tuple[str, ...]:
    if sys.platform == "darwin":
        return ("Ctrl+Down",)
    return ("Return", "Enter")


IMAGETOOL_OPEN = QtGui.QKeySequence.StandardKey.Open
IMAGETOOL_SAVE_AS = QtGui.QKeySequence.StandardKey.SaveAs
IMAGETOOL_CLOSE = WINDOW_CLOSE
IMAGETOOL_REMOVE = QtGui.QKeySequence.StandardKey.Delete
IMAGETOOL_REVEAL_MANAGER = "Ctrl+Shift+M"
IMAGETOOL_SETTINGS = QtGui.QKeySequence.StandardKey.Preferences
IMAGETOOL_RELOAD = QtGui.QKeySequence.StandardKey.Refresh
IMAGETOOL_VIEW_ALL = "Ctrl+A"
IMAGETOOL_TRANSPOSE = "T"
IMAGETOOL_ADD_CURSOR = "Shift+A"
IMAGETOOL_REMOVE_CURSOR = "Shift+R"
IMAGETOOL_TOGGLE_CURSORS = "Shift+V"
IMAGETOOL_UNDO = QtGui.QKeySequence.StandardKey.Undo
IMAGETOOL_REDO = QtGui.QKeySequence.StandardKey.Redo
IMAGETOOL_CENTER_CURSOR = "Shift+C"
IMAGETOOL_CENTER_ALL_CURSORS = "Alt+Shift+C"
IMAGETOOL_REVERSE_COLORMAP = "R"
IMAGETOOL_SNAP_TO_PIXELS = "S"
IMAGETOOL_COPY_CURSOR_VALUES = "Ctrl+Shift+C"
IMAGETOOL_COPY_CURSOR_INDICES = "Ctrl+Alt+C"
IMAGETOOL_MOVE_CURSOR = (
    "Shift+Up",
    "Shift+Down",
    "Shift+Right",
    "Shift+Left",
    "Ctrl+Shift+Up",
    "Ctrl+Shift+Down",
    "Ctrl+Shift+Right",
    "Ctrl+Shift+Left",
)
IMAGETOOL_MOVE_ALL_CURSORS = (
    "Alt+Shift+Up",
    "Alt+Shift+Down",
    "Alt+Shift+Right",
    "Alt+Shift+Left",
    "Ctrl+Alt+Shift+Up",
    "Ctrl+Alt+Shift+Down",
    "Ctrl+Alt+Shift+Right",
    "Ctrl+Alt+Shift+Left",
)

EXPLORER_OPEN_IN_MANAGER = QtGui.QKeySequence.StandardKey.Open
EXPLORER_CLOSE = WINDOW_CLOSE
EXPLORER_OPEN_FOLDER = "Ctrl+Shift+O"
EXPLORER_RELOAD_FOLDER = QtGui.QKeySequence.StandardKey.Refresh
EXPLORER_NEW_TAB = QtGui.QKeySequence.StandardKey.AddTab


def explorer_enclosing_folder() -> str:
    return "Ctrl+Up" if sys.platform == "darwin" else "Alt+Up"


def explorer_next_tab() -> str:
    return "Meta+Tab" if sys.platform == "darwin" else "Ctrl+Tab"


def explorer_previous_tab() -> str:
    return "Meta+Shift+Tab" if sys.platform == "darwin" else "Ctrl+Shift+Tab"


FIGURE_COMPOSER_UNDO = QtGui.QKeySequence.StandardKey.Undo
FIGURE_COMPOSER_REDO = QtGui.QKeySequence.StandardKey.Redo
FIGURE_COMPOSER_SAVE_WORKSPACE = WORKSPACE_SAVE
FIGURE_COMPOSER_CLOSE = WINDOW_CLOSE
FIGURE_COMPOSER_RENAME_SOURCE = "F2"
FIGURE_COMPOSER_COMPLETION = "Ctrl+Space"
