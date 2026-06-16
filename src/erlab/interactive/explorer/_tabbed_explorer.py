"""Tabbed window for managing multiple data explorer tabs."""

from __future__ import annotations

import pathlib
import sys
import typing
import weakref

import pydantic
from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive import _qt_state
from erlab.interactive.explorer._base_explorer import (
    DataExplorerTabState,
    _DataExplorer,
)


class DataExplorerState(pydantic.BaseModel):
    # The tabbed Data Explorer top-level window uses the shared Qt window schema.
    window_state: _qt_state.QtWindowState | None = None
    tabs: tuple[DataExplorerTabState, ...] = ()
    active_tab: int = 0
    # Shared loader options for tabs in this Data Explorer window.
    loader_kwargs_by_name: dict[str, dict[str, typing.Any]] = pydantic.Field(
        default_factory=dict
    )
    loader_extensions_by_name: dict[str, dict[str, typing.Any]] = pydantic.Field(
        default_factory=dict
    )

    model_config = pydantic.ConfigDict(extra="ignore")

    @pydantic.field_validator("tabs", mode="before")
    @classmethod
    def _tabs_tuple(cls, value: object) -> object:
        if value is None:
            return ()
        return tuple(value) if isinstance(value, (list, tuple)) else value


class _TabbedExplorer(QtWidgets.QMainWindow):
    sigStateChanged = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None, **kwargs) -> None:
        super().__init__(parent=parent)
        self._workspace_state_restoring = False
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
        self.tab_widget.currentChanged.connect(self._emit_workspace_state_changed)

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

    def _emit_workspace_state_changed(self, *_args: object) -> None:
        if not self._workspace_state_restoring:
            self.sigStateChanged.emit()

    def loader_kwargs_by_name(self) -> dict[str, dict[str, typing.Any]]:
        loader_kwargs: dict[str, dict[str, typing.Any]] = {}
        for index in range(self.tab_widget.count()):
            explorer = self.get_explorer(index)
            if explorer is not None:
                loader_kwargs.update(explorer.loader_kwargs_by_name())
        return loader_kwargs

    def loader_extensions_by_name(self) -> dict[str, dict[str, typing.Any]]:
        loader_extensions: dict[str, dict[str, typing.Any]] = {}
        for index in range(self.tab_widget.count()):
            explorer = self.get_explorer(index)
            if explorer is not None:
                loader_extensions.update(explorer.loader_extensions_by_name())
        return loader_extensions

    def apply_loader_state(
        self,
        *,
        kwargs_by_name: dict[str, dict[str, typing.Any]],
        extensions_by_name: dict[str, dict[str, typing.Any]],
    ) -> None:
        for index in range(self.tab_widget.count()):
            explorer = self.get_explorer(index)
            if explorer is not None:
                explorer.apply_loader_state(
                    kwargs_by_name=kwargs_by_name,
                    extensions_by_name=extensions_by_name,
                )

    def workspace_state_payload(self) -> dict[str, typing.Any]:
        return DataExplorerState(
            window_state=_qt_state.qt_window_state(self),
            tabs=tuple(
                DataExplorerTabState.model_validate(explorer.workspace_state_payload())
                for index in range(self.tab_widget.count())
                if (explorer := self.get_explorer(index)) is not None
            ),
            active_tab=int(self.tab_widget.currentIndex()),
            loader_kwargs_by_name=self.loader_kwargs_by_name(),
            loader_extensions_by_name=self.loader_extensions_by_name(),
        ).model_dump(mode="json", exclude_none=True)

    def restore_workspace_state(self, state: DataExplorerState) -> None:
        state = DataExplorerState.model_validate(state)
        self._workspace_state_restoring = True
        try:
            loader_kwargs_by_name = {
                str(name): dict(value)
                for name, value in state.loader_kwargs_by_name.items()
            }
            loader_extensions_by_name = {
                str(name): dict(value)
                for name, value in state.loader_extensions_by_name.items()
            }

            if state.tabs:
                self._clear_tabs()
                for tab_state in state.tabs:
                    self.add_tab(
                        root_path=tab_state.root_path,
                        loader_name=tab_state.loader_name,
                    )
                    explorer = self.current_explorer
                    if explorer is not None:
                        explorer.restore_workspace_state(tab_state)
            self.apply_loader_state(
                kwargs_by_name=loader_kwargs_by_name,
                extensions_by_name=loader_extensions_by_name,
            )

            if self.tab_widget.count() > 0:
                self.tab_widget.setCurrentIndex(
                    max(0, min(state.active_tab, self.tab_widget.count() - 1))
                )

            _qt_state.restore_qt_window_state(self, state.window_state)
        finally:
            self._workspace_state_restoring = False
        self._emit_workspace_state_changed()

    def show_path(self, path: str | pathlib.Path) -> None:
        source_path = pathlib.Path(path).resolve()
        root_path = source_path if source_path.is_dir() else source_path.parent
        self.add_tab(root_path=root_path)
        explorer = self.current_explorer
        if explorer is None:
            return
        selected_paths = () if source_path.is_dir() else (str(source_path),)
        explorer.restore_workspace_state(
            DataExplorerTabState(
                root_path=str(root_path),
                loader_name=explorer.loader_name,
                selected_paths=selected_paths,
            )
        )

    def _clear_tabs(self) -> None:
        while self.tab_widget.count() > 0:
            self._discard_tab(0)

    def _discard_tab(self, index: int) -> None:
        tab = self.tab_widget.widget(index)
        explorer = self.get_explorer(index)
        self.tab_widget.removeTab(index)
        if explorer is not None:
            if explorer._stop_preview_workers():
                explorer.deleteLater()
            else:
                explorer._delete_when_preview_workers_done()
        if tab is not None:
            tab.deleteLater()

    def _stop_preview_workers(self) -> None:
        for index in range(self.tab_widget.count()):
            explorer = self.get_explorer(index)
            if explorer is not None:
                explorer._stop_preview_workers()

    def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:
        self._stop_preview_workers()
        super().closeEvent(event)

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

        loader_kwargs = self.loader_kwargs_by_name()
        loader_extensions = self.loader_extensions_by_name()
        new_explorer: _DataExplorer = _DataExplorer(self, **kwargs)
        new_explorer.apply_loader_state(
            kwargs_by_name=loader_kwargs,
            extensions_by_name=loader_extensions,
        )
        tab_idx: int = self.tab_widget.addTab(
            QtWidgets.QWidget(), str(new_explorer.current_directory.name)
        )
        self.tab_widget.setCurrentIndex(tab_idx)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        content = new_explorer.takeCentralWidget()
        if content:
            # Transfer UI ownership from the hidden QMainWindow wrapper to the tab.
            layout.addWidget(content)
        current_tab = typing.cast("QtWidgets.QWidget", self.tab_widget.widget(tab_idx))
        current_tab.setLayout(layout)
        self.tab_widget.setTabText(tab_idx, new_explorer.current_directory.name)
        self.tab_widget.setTabToolTip(tab_idx, str(new_explorer.current_directory))

        # Store a reference to the DataExplorer instance in the tab widget
        current_tab._explorer = new_explorer  # type: ignore[attr-defined]

        new_explorer.sigDirectoryChanged.connect(self.update_title)
        new_explorer.sigCloseRequested.connect(self.close_tab)
        new_explorer.sigStateChanged.connect(self._emit_workspace_state_changed)
        self.update_menubar()
        self._emit_workspace_state_changed()

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
            erlab.interactive.utils._hide_or_close_with_manager(self)
        else:
            if isinstance(index, _DataExplorer):
                # If index is a DataExplorer instance, find its tab index by iteration
                index = -1
                for i in range(self.tab_widget.count()):
                    if self.get_explorer(i) is index:
                        index = i
                        break
            self._discard_tab(self.tab_widget.currentIndex() if index < 0 else index)
        self.update_menubar()
        self._emit_workspace_state_changed()

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
