"""Tabbed window for managing multiple data explorer tabs."""

from __future__ import annotations

import sys
import typing
import weakref

from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive.explorer._base_explorer import _DataExplorer


class _TabbedExplorer(QtWidgets.QMainWindow):
    def __init__(self, parent: QtWidgets.QWidget | None = None, **kwargs) -> None:
        super().__init__(parent=parent)
        self._setup_actions()
        self._setup_ui(**kwargs)

    def _setup_ui(self, **kwargs) -> None:
        """Set up the main window UI."""
        self.tab_widget = QtWidgets.QTabWidget(self)
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.setMovable(True)
        self.tab_widget.setDocumentMode(True)

        self.tab_widget.tabCloseRequested.connect(self.close_tab)
        self.tab_widget.currentChanged.connect(self.update_menubar)

        add_btn = erlab.interactive.utils.IconActionButton(
            self._addtab_act, "mdi6.plus", icon_kw={"scale_factor": 1.2}
        )
        add_btn.setFlat(True)
        self.tab_widget.setCornerWidget(add_btn, QtCore.Qt.Corner.TopRightCorner)

        self.setCentralWidget(self.tab_widget)

        self.add_tab(**kwargs)

        self.setMinimumWidth(487)
        self.setMinimumHeight(301)
        self.resize(974, 602)

    def _setup_actions(self) -> None:
        """Set up the main window actions."""
        # Set up menubar
        self._main_menubar = typing.cast("QtWidgets.QMenuBar", self.menuBar())

        # Add file menu
        file_menu = typing.cast("QtWidgets.QMenu", self._main_menubar.addMenu("&File"))

        self._addtab_act = QtWidgets.QAction("New Tab", self)
        self._addtab_act.triggered.connect(self.add_tab)
        self._addtab_act.setShortcut(QtGui.QKeySequence.StandardKey.AddTab)
        file_menu.addAction(self._addtab_act)

        self._next_act = QtWidgets.QAction("Next Tab", self)
        self._next_act.triggered.connect(
            lambda: self.tab_widget.setCurrentIndex(
                (self.tab_widget.currentIndex() + 1) % self.tab_widget.count()
            )
        )
        self._next_act.setShortcut(
            QtGui.QKeySequence("Meta+Tab" if sys.platform == "darwin" else "Ctrl+Tab")
        )
        file_menu.addAction(self._next_act)

        self._prev_act = QtWidgets.QAction("Previous Tab", self)
        self._prev_act.triggered.connect(
            lambda: self.tab_widget.setCurrentIndex(
                (self.tab_widget.currentIndex() - 1) % self.tab_widget.count()
            )
        )
        self._prev_act.setShortcut(
            QtGui.QKeySequence(
                "Meta+Shift+Tab" if sys.platform == "darwin" else "Ctrl+Shift+Tab"
            )
        )
        file_menu.addAction(self._prev_act)

        self._file_sep = QtWidgets.QAction(self)
        self._file_sep.setSeparator(True)
        file_menu.addAction(self._file_sep)

        # Store global actions that should be available across all tabs here
        self._global_actions: weakref.WeakSet[QtWidgets.QAction] = weakref.WeakSet()
        self._global_actions.add(self._addtab_act)
        self._global_actions.add(self._next_act)
        self._global_actions.add(self._prev_act)
        self._global_actions.add(self._file_sep)

    @property
    def current_explorer(self) -> _DataExplorer | None:
        """Get the currently active DataExplorer instance."""
        current_index = self.tab_widget.currentIndex()
        if current_index >= 0:
            return self.get_explorer(current_index)
        return None

    def get_explorer(self, index: int) -> _DataExplorer | None:
        """Get the DataExplorer instance for the given tab index."""
        current_tab = typing.cast("QtWidgets.QWidget", self.tab_widget.widget(index))
        if hasattr(current_tab, "_explorer"):
            # If the tab has an _explorer attribute, return it
            return typing.cast("_DataExplorer", current_tab._explorer)
        return None

    @QtCore.Slot()
    def add_tab(self, **kwargs) -> None:
        """Add a new tab with a DataExplorer instance.

        Parameters
        ----------
        kwargs
            Additional keyword arguments to pass to the DataExplorer constructor.
            If `root_path` or `loader_name` is not provided, they will default to the
            currently active tab's directory and loader name, respectively.
        """
        if self.current_explorer is not None:
            # Take current tab's directory and loader name as defaults
            kwargs.setdefault("root_path", self.current_explorer.current_directory)
            kwargs.setdefault("loader_name", self.current_explorer.loader_name)

        new_explorer: _DataExplorer = _DataExplorer(self, **kwargs)
        tab_idx: int = self.tab_widget.addTab(
            QtWidgets.QWidget(), str(new_explorer.current_directory.name)
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

        new_explorer.sigDirectoryChanged.connect(self.update_title)
        new_explorer.sigCloseRequested.connect(self.close_tab)
        self.update_menubar()

    @QtCore.Slot()
    def update_menubar(self) -> None:
        """Update the menubar to reflect the current tab's actions.

        Combines global actions with the current tab's actions, respecting the top-level
        names of the menus. For example, if a tab has a menu "File" with actions, those
        actions will be added to the global "File" menu if it exists.
        """
        self.update_title()

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
        subwindow = self.get_explorer(idx)
        if not subwindow:
            # No tab is selected or selected tab is still being initialized?
            return

        # Tab is selected, get its DataExplorer instance
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

    @QtCore.Slot()
    def update_title(self) -> None:
        """Update the tab and window title with the current directory name."""
        tab_idx = self.tab_widget.currentIndex()
        explorer = self.get_explorer(tab_idx)
        if not explorer:
            self.setWindowTitle("Data Explorer")
        else:
            curr_dir = explorer.current_directory
            self.setWindowTitle(f"Data Explorer — {curr_dir.name}")
            self.tab_widget.setTabText(tab_idx, curr_dir.name)
            self.tab_widget.setTabToolTip(tab_idx, str(curr_dir))

    @QtCore.Slot(int)
    @QtCore.Slot(object)
    def close_tab(self, index: int | _DataExplorer) -> None:
        if self.tab_widget.count() == 1:
            self.hide()
        else:
            if isinstance(index, _DataExplorer):
                # If index is a DataExplorer instance, find its tab index by iteration
                index = -1
                for i in range(self.tab_widget.count()):
                    if self.get_explorer(i) is index:
                        index = i
                        break
            self.tab_widget.removeTab(
                self.tab_widget.currentIndex() if index < 0 else index
            )
        self.update_menubar()

    def dragEnterEvent(
        self, event: QtGui.QDragEnterEvent | None
    ) -> None:  # pragma: no cover
        """Pass drag events to the current explorer."""
        if self.current_explorer:
            self.current_explorer.dragEnterEvent(event)
        super().dragEnterEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent | None) -> None:  # pragma: no cover
        """Pass drop events to the current explorer."""
        if self.current_explorer:
            self.current_explorer.dropEvent(event)
        super().dropEvent(event)
