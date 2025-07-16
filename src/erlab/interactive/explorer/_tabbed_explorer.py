"""Tabbed window for managing multiple data explorer tabs."""

from __future__ import annotations

import pathlib
import typing
import weakref

from qtpy import QtCore, QtGui, QtWidgets

from erlab.interactive.explorer._base_explorer import _DataExplorer

if typing.TYPE_CHECKING:
    import collections.abc


def _menu_to_dict(menu: QtWidgets.QMenuBar | QtWidgets.QMenu) -> dict[str, typing.Any]:
    result: dict[str, typing.Any] = {}
    result = {}
    for action in menu.actions():
        child_menu = action.menu()
        if child_menu:
            # Recursively get submenu structure
            result[action.text()] = _menu_to_dict(child_menu)
        else:
            result[action.text()] = action
    return result


def _get_all_values(
    d: dict[str, typing.Any],
) -> collections.abc.Generator[typing.Any, None, None]:
    """Recursively yield all values from a nested dictionary."""
    for value in d.values():
        if isinstance(value, dict):
            yield from _get_all_values(value)
        else:
            yield value


class _TabbedExplorer(QtWidgets.QMainWindow):
    def __init__(self, parent: QtWidgets.QWidget | None = None, **kwargs) -> None:
        super().__init__(parent=parent)
        self.setWindowTitle("Explorer Tabs Example")
        self._setup_actions()
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the main window UI."""
        self.central = QtWidgets.QWidget(self)
        self.setCentralWidget(self.central)
        self.layout_ = QtWidgets.QVBoxLayout(self.central)

        self.tab_widget = QtWidgets.QTabWidget(self)
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self.close_tab)

        self.layout_.addWidget(self.tab_widget)

        button_layout = QtWidgets.QHBoxLayout()
        self.add_tab_btn = QtWidgets.QPushButton("Add Tab", self)
        self.add_tab_btn.clicked.connect(self.add_tab)
        button_layout.addWidget(self.add_tab_btn)
        button_layout.addStretch()
        self.layout_.addLayout(button_layout)

        self.add_tab()

    def _setup_actions(self) -> None:
        # Set up menubar
        self._main_menubar = typing.cast("QtWidgets.QMenuBar", self.menuBar())

        # Add file menu
        file_menu = typing.cast("QtWidgets.QMenu", self._main_menubar.addMenu("&File"))

        # Store global actions that should be available across all tabs here
        self._global_actions: weakref.WeakSet[QtWidgets.QAction] = weakref.WeakSet()

        self._addtab_act = QtWidgets.QAction("New Tab", self)
        self._addtab_act.triggered.connect(self.add_tab)
        self._addtab_act.setShortcut(QtGui.QKeySequence.StandardKey.AddTab)
        self._global_actions.add(self._addtab_act)
        file_menu.addAction(self._addtab_act)

    def get_explorer(self, index: int) -> _DataExplorer:
        """Get the DataExplorer instance for the given tab index."""
        current_tab = typing.cast("QtWidgets.QWidget", self.tab_widget.widget(index))
        return current_tab._explorer  # type: ignore[attr-defined]

    @QtCore.Slot()
    def add_tab(self) -> None:
        new_explorer: _DataExplorer = _DataExplorer(self)
        tab_idx: int = self.tab_widget.addTab(
            QtWidgets.QWidget(), f"Tab {self.tab_widget.count() + 1}"
        )
        self.tab_widget.setCurrentIndex(tab_idx)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        content = new_explorer.centralWidget()
        if content:
            # Embed the DataExplorer's central widget into the tab
            layout.addWidget(content)
        current_tab = typing.cast("QtWidgets.QWidget", self.tab_widget.widget(tab_idx))
        current_tab.setLayout(layout)
        self.tab_widget.setTabText(tab_idx, new_explorer.current_directory.name)
        self.tab_widget.setTabToolTip(tab_idx, str(new_explorer.current_directory))

        # Store a reference to the DataExplorer instance in the tab widget
        current_tab._explorer = new_explorer  # type: ignore[attr-defined]

        new_explorer.sigDirectoryChanged.connect(
            lambda path, idx=tab_idx: self.update_tab_title(idx, path)
        )
        self.update_menubar()

    def update_menubar(self) -> None:
        """Update the menubar to reflect the current tab's actions.

        Combines global actions with the current tab's actions, respecting the top-level
        names of the menus. For example, if a tab has a menu "File" with actions, those
        actions will be added to the global "File" menu if it exists.
        """

        def _remove_non_global_act(menu):
            """Remove all actions except top-level global actions."""
            for action in menu.actions():
                child_menu = action.menu()
                if child_menu:
                    # Is a submenu, continue recursively
                    _remove_non_global_act(child_menu)
                else:
                    # Is an action, check if it's global
                    if action not in self._global_actions:
                        menu.removeAction(action)

        _remove_non_global_act(self._main_menubar)
        idx = self.tab_widget.currentIndex()
        if idx != -1:
            # Tab is selected, get its DataExplorer instance
            subwindow = self.get_explorer(idx)
            for sub_action in subwindow.menu_bar.actions():
                # Iterate through top-level actions, check if they have a submenu
                added: bool = False
                child_menu = sub_action.menu()
                if child_menu:
                    # Action has a submenu, check if menu with same title exists
                    title = sub_action.text()
                    for global_act in self._main_menubar.actions():
                        if global_act.text() == title:
                            # If found, add submenu actions to the global menu
                            target_menu = global_act.menu()
                            if target_menu:
                                for sub_sub_action in child_menu.actions():
                                    target_menu.addAction(sub_sub_action)
                            added = True
                            break
                if not added:
                    # Add new actions or menus directly
                    self._main_menubar.addAction(sub_action)

    @QtCore.Slot(int, str)
    def update_tab_title(self, index: int, path: str):
        """Update the tab title with the current directory name."""
        if index < self.tab_widget.count():
            dir_name = pathlib.Path(path).name if path else path
            self.tab_widget.setTabText(index, dir_name)

    def close_tab(self, index: int | _DataExplorer) -> None:
        if self.tab_widget.count() == 1:
            self.close()
        else:
            if isinstance(index, _DataExplorer):
                # If index is a DataExplorer instance, find its tab index by iteration
                for i in range(self.tab_widget.count()):
                    if self.get_explorer(i) is index:
                        index = i
                        break
            if isinstance(index, int):
                self.tab_widget.removeTab(index)
        self.update_menubar()
