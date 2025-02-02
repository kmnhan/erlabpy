"""Quickly browse and load ARPES data files with a file manager-like interface."""

from __future__ import annotations

import os
import pathlib
import sys
import time
import typing
import weakref

from qtpy import QtCore, QtGui, QtWidgets

import erlab

if typing.TYPE_CHECKING:
    from collections.abc import Callable

_IGOR_PRO_MIME_TYPES = {
    "pxt": "Igor Pro Packed Stationery",
    "pxp": "Igor Pro Packed Experiment",
    "ibw": "Igor Pro Binary Wave",
    "itx": "Igor Pro Text Data",
}

_TRANSLATE_MIME_TYPES = {
    "application/octet-stream": "Unknown",
    "text/csv": "CSV Document",
}


class _FileSystem:
    """Represents a file system.

    Parameters
    ----------
    path
        Path to the file or directory.
    show_hidden
        Whether to show hidden files and directories (starting with a dot).
    """

    def __init__(self, path: str | os.PathLike, show_hidden: bool = False) -> None:
        self._path = pathlib.Path(path)
        self._children: list[_FileSystem] | None = None

        self._show_hidden = show_hidden

    @property
    def path(self) -> pathlib.Path:
        """Path to the file or directory."""
        return self._path

    @path.setter
    def path(self, path: str | os.PathLike) -> None:
        self._path = pathlib.Path(path)
        self._children = None

    @property
    def show_hidden(self) -> bool:
        """Whether files prefixed with a dot are taken into account."""
        return self._show_hidden

    @show_hidden.setter
    def show_hidden(self, show: bool) -> None:
        self._show_hidden = show
        self._children = None

    @property
    def has_children(self) -> bool:
        """Whether the file system can have children (is a directory).

        This being `True` does not guarantee that the children have been fetched. Use
        the `can_fetch_children` property to check.
        """
        return self.path.is_dir()

    @property
    def children(self) -> list[_FileSystem]:
        """List of children of the file system.

        This property is undefined if the file system has no children.
        """
        if self._children is None:
            self.reload()
        return typing.cast(list[_FileSystem], self._children)

    @property
    def can_fetch_children(self) -> bool:
        """Whether the children has been fetched since initialization."""
        return self.has_children and self._children is None

    def reload(self) -> None:
        """Reload the children of the file system."""
        if self.has_children:
            self._children = []
            for p in self.path.iterdir():
                if not self.show_hidden and p.name.startswith("."):
                    continue
                self._children.append(_FileSystem(p))

    def __getitem__(self, key: str) -> _FileSystem:
        """Get a child of the file system by its name."""
        if self.has_children:
            for child in self.children:
                if child.path.name == key:
                    return child
        raise KeyError(key)

    def __repr__(self) -> str:
        return f"FileSystem({self.path})"

    def child_from_path(self, path: str | pathlib.Path) -> _FileSystem:
        path = pathlib.Path(path).relative_to(self.path)
        child = self
        for part in path.parts:
            child = child[part]
        return child

    def sort_recursive(
        self, key: Callable[[_FileSystem], typing.Any], reverse: bool = False
    ) -> None:
        if self.has_children and not self.can_fetch_children:  # sort only loaded
            self._children = sorted(self.children, key=key, reverse=reverse)
            for child in self._children:
                child.sort_recursive(key, reverse)


class _DataExplorerModel(QtCore.QAbstractItemModel):
    """Model for a file system explorer.

    Add to a QTreeView to display a file system in a file manager-like interface.

    This model works similarly to a QFileSystemModel, but lacks watching for changes in
    the file system. Use the `refresh` method to update the model.

    Although QFileSystemModel has more features, upon opening a folder in a cloud
    storage or network drive, it tries to download all the files. This is critical for
    handling large data. This model only fetches the file name using standard library
    calls, which is much more efficient and safe.
    """

    def __init__(
        self,
        root_path: str | os.PathLike,
        file_browser: _DataExplorer,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.file_system: _FileSystem = _FileSystem(root_path)
        self._file_browser = weakref.ref(file_browser)

        self._icon_provider: QtWidgets.QFileIconProvider = QtWidgets.QFileIconProvider()
        self._mime_database: QtCore.QMimeDatabase = QtCore.QMimeDatabase()
        self._sort_column: int = 0
        self._sort_order: QtCore.Qt.SortOrder = QtCore.Qt.SortOrder.AscendingOrder

    @property
    def file_browser(self) -> _DataExplorer:
        """Parent DataExplorer widget."""
        _file_browser = self._file_browser()
        if _file_browser:
            return _file_browser
        raise LookupError("Parent was destroyed")

    def set_root_path(self, root_path: str | os.PathLike) -> None:
        self.file_system.path = pathlib.Path(root_path)
        self.reload()

    def set_show_hidden(self, show: bool) -> None:
        self.file_system.show_hidden = show
        self.reload()

    def file_path(self, index: QtCore.QModelIndex) -> str:
        """Get the string representing the path of the file at given index."""
        return str(self.get_fs(index).path)

    def file_info(self, index: QtCore.QModelIndex) -> QtCore.QFileInfo:
        """Get the :class:`QtCore.QFileInfo` of the file at given index."""
        return QtCore.QFileInfo(self.file_path(index))

    def mime_type(self, index: QtCore.QModelIndex) -> str:
        file_info = self.file_info(index)
        if file_info.suffix() in _IGOR_PRO_MIME_TYPES:
            mime: str = _IGOR_PRO_MIME_TYPES[file_info.suffix()]

        else:
            mime = self._mime_database.mimeTypeForFile(
                file_info, QtCore.QMimeDatabase.MatchMode.MatchExtension
            ).comment()

        return _TRANSLATE_MIME_TYPES.get(mime, mime)

    def date_modified(self, index: QtCore.QModelIndex) -> str:
        """Get the date modified of the file at the index."""
        path: pathlib.Path = index.internalPointer().path
        if path.exists():
            return time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime(os.path.getmtime(path)),
            )
        return ""

    def hasChildren(self, parent: QtCore.QModelIndex | None = None) -> bool:
        """Whether the index has children."""
        if parent is None:
            parent = QtCore.QModelIndex()
        return self.get_fs(parent).has_children

    def get_fs(self, index: QtCore.QModelIndex) -> _FileSystem:
        """Get the underlying file system object from the index."""
        if not index.isValid():
            return self.file_system

        return typing.cast(_FileSystem, index.internalPointer())

    def fetchMore(self, parent: QtCore.QModelIndex) -> None:
        """Fetch more children of the file system."""
        self.get_fs(parent).reload()

    def canFetchMore(self, parent: QtCore.QModelIndex) -> bool:
        return self.get_fs(parent).can_fetch_children

    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlag:
        default_flags = QtCore.Qt.ItemFlag.ItemIsDropEnabled | super().flags(index)

        if index.isValid():
            flags = QtCore.Qt.ItemFlag.ItemIsDragEnabled | default_flags
            ext: str = self.get_fs(index).path.suffix
            loader = erlab.io.loaders[self.file_browser.loader_name]
            if (
                loader.extensions is not None
                and ext != ""
                and ext not in loader.extensions
            ):
                flags = flags & ~QtCore.Qt.ItemFlag.ItemIsEnabled
            return flags
        return default_flags

    def index(
        self, row: int, column: int, parent: QtCore.QModelIndex | None = None
    ) -> QtCore.QModelIndex:
        if parent is None:
            parent = QtCore.QModelIndex()
        if not self.hasIndex(row, column, parent):
            return QtCore.QModelIndex()

        return self.createIndex(row, column, self.get_fs(parent).children[row])

    @typing.overload
    def parent(self, child: QtCore.QModelIndex) -> QtCore.QModelIndex: ...

    @typing.overload
    def parent(self) -> QtCore.QObject | None: ...

    def parent(
        self, child: QtCore.QModelIndex | None = None
    ) -> QtCore.QModelIndex | QtCore.QObject | None:
        if child is None:
            return super().parent()

        if not child.isValid():
            return QtCore.QModelIndex()
        return self._find_parent_index(self.get_fs(child))

    def _find_index(self, child_item: _FileSystem) -> QtCore.QModelIndex:
        try:
            parent_item = self.file_system.child_from_path(child_item.path.parent)
        except (KeyError, ValueError):
            return QtCore.QModelIndex()

        row: int = parent_item.children.index(child_item)
        return self.createIndex(row, 0, child_item)

    def _find_parent_index(self, child_item: _FileSystem) -> QtCore.QModelIndex:
        try:
            parent_item = self.file_system.child_from_path(child_item.path.parent)
        except (KeyError, ValueError):
            return QtCore.QModelIndex()

        if parent_item == self.file_system:
            return QtCore.QModelIndex()

        try:
            grandparent_item = self.file_system.child_from_path(parent_item.path.parent)
        except (KeyError, ValueError):
            return QtCore.QModelIndex()

        row: int = grandparent_item.children.index(parent_item)
        return self.createIndex(row, 0, parent_item)

    def rowCount(self, parent: QtCore.QModelIndex | None = None) -> int:
        if parent is None:
            parent = QtCore.QModelIndex()
        if parent.column() > 0:
            return 0
        return len(self.get_fs(parent).children)

    def columnCount(self, parent: QtCore.QModelIndex | None = None) -> int:
        return 4

    def data(
        self, index: QtCore.QModelIndex, role: int = QtCore.Qt.ItemDataRole.DisplayRole
    ) -> typing.Any:
        if not index.isValid():
            return None

        item: _FileSystem = self.get_fs(index)
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            match index.column():
                case 0:
                    return item.path.name
                case 1:
                    if item.path.is_dir() or not item.path.exists():
                        return "--"
                    return erlab.utils.formatting.format_nbytes(
                        os.path.getsize(item.path)
                    )
                case 2:
                    return self.mime_type(index)
                case 3:
                    return self.date_modified(index)

        elif role == QtCore.Qt.ItemDataRole.DecorationRole:
            match index.column():
                case 0:
                    return self._icon_provider.icon(self.file_info(index))

        return None

    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> typing.Any:
        if (
            orientation == QtCore.Qt.Orientation.Horizontal
            and role == QtCore.Qt.ItemDataRole.DisplayRole
        ):
            match section:
                case 0:
                    return "Name"
                case 1:
                    return "Size"
                case 2:
                    return "Type"
                case 3:
                    return "Date Modified"
        return None

    @QtCore.Slot()
    def reload(self) -> None:
        self.beginResetModel()
        self.file_system.reload()
        self._sort_fs()
        self.endResetModel()

    @QtCore.Slot()
    def climb_up(self) -> None:
        self.set_root_path(self.file_system.path.parent)

    def sort(
        self,
        column: int,
        order: QtCore.Qt.SortOrder = QtCore.Qt.SortOrder.AscendingOrder,
    ) -> None:
        self._sort_column = column
        self._sort_order = order
        self.layoutAboutToBeChanged.emit()
        self._sort_fs()
        self.layoutChanged.emit()

    def _sort_fs(self) -> None:
        self.file_system.sort_recursive(
            key=self._get_sort_key_func(self._sort_column),
            reverse=self._sort_order == QtCore.Qt.SortOrder.DescendingOrder,
        )

    def _get_sort_key_func(
        self, column: int
    ) -> typing.Callable[[_FileSystem], typing.Any]:
        def name_key(item: _FileSystem) -> str:
            return item.path.name.casefold()

        def size_key(item: _FileSystem) -> int:
            return os.path.getsize(item.path) if item.path.is_file() else -1

        def type_key(item: _FileSystem) -> str:
            return self.mime_type(self._find_index(item)).casefold()

        def date_key(item: _FileSystem) -> float:
            return os.path.getmtime(item.path) if item.path.exists() else 0

        return [name_key, size_key, type_key, date_key][column]


class _DataExplorerTreeView(QtWidgets.QTreeView):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSortingEnabled(True)
        self.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.setDragEnabled(True)
        self.setAcceptDrops(False)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragOnly)

    def model(self) -> _DataExplorerModel:
        return typing.cast(_DataExplorerModel, super().model())

    def selectionModel(self) -> QtCore.QItemSelectionModel:
        return typing.cast(QtCore.QItemSelectionModel, super().selectionModel())

    @property
    def selected_paths(self) -> list[pathlib.Path]:
        return [
            self.model().get_fs(idx).path
            for idx in self.selectedIndexes()
            if idx.column() == 0  # Unique rows
        ]

    def startDrag(self, supportedActions: QtCore.Qt.DropAction) -> None:
        drag = QtGui.QDrag(self)
        mime_data = QtCore.QMimeData()
        mime_data.setUrls(
            [
                QtCore.QUrl.fromLocalFile(str(path.resolve()))
                for path in self.selected_paths
            ]
        )
        drag.setMimeData(mime_data)
        drag.exec(QtCore.Qt.DropAction.CopyAction)


class _ReprFetcherSignals(QtCore.QObject):
    fetched = QtCore.Signal(str, str)


class _ReprFetcher(QtCore.QRunnable):
    """Worker to fetch information about a single ARPES data file.

    Parameters
    ----------
    file_path
        Path to the file.
    loader_name
        Name of the loader plugin to use.
    """

    def __init__(self, file_path: str | pathlib.Path, load_method) -> None:
        super().__init__()
        self.signals = _ReprFetcherSignals()
        self.file_path = pathlib.Path(file_path)
        self.load_method = load_method

    @QtCore.Slot()
    def run(self) -> None:
        file_path = str(self.file_path)
        try:
            dat = self.load_method(
                file_path, single=True, load_kwargs={"without_values": True}
            )
            text = erlab.utils.formatting.format_darr_html(dat, additional_info=[])
            del dat
        except Exception as e:
            text = "Error loading file:\n" + f"{type(e).__name__}: {e}"
        self.signals.fetched.emit(file_path, text)


class _LoaderInfoModel(QtCore.QAbstractTableModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def data(
        self, index: QtCore.QModelIndex, role: int = QtCore.Qt.ItemDataRole.DisplayRole
    ) -> typing.Any:
        if not index.isValid():
            return None
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            loader_name: str = list(erlab.io.loaders.keys())[index.row()]

            match index.column():
                case 0:
                    return loader_name
                case 1:
                    loader = erlab.io.loaders[loader_name]
                    return loader.description if hasattr(loader, "description") else ""
        return None

    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> typing.Any:
        if (
            role == QtCore.Qt.ItemDataRole.DisplayRole
            and orientation == QtCore.Qt.Orientation.Horizontal
        ):
            match section:
                case 0:
                    return "Name"
                case 1:
                    return "Description"
        return None

    def rowCount(self, index: QtCore.QModelIndex | None = None) -> int:
        return len(erlab.io.loaders.keys())

    def columnCount(self, index: QtCore.QModelIndex | None = None) -> int:
        return 2


class _LoaderWidget(QtWidgets.QComboBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        model = _LoaderInfoModel()
        view = QtWidgets.QTableView()
        view.setCornerButtonEnabled(False)
        view.verticalHeader().hide()
        view.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )

        self.setModel(model)
        self.setView(view)
        view.resizeColumnsToContents()
        view.setMinimumWidth(
            sum(view.columnWidth(i) for i in range(model.columnCount(0)))
        )


class _DataExplorer(QtWidgets.QWidget):
    TEXT_NONE_SELECTED: str = (
        "Select a folder or drag and drop a folder into the window "
        "to browse its contents."
    )
    TEXT_MULTIPLE_SELECTED: str = "Multiple files selected"

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        root_path: str | os.PathLike | None = None,
        loader_name: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setWindowTitle("Data Explorer")
        root_path = root_path if root_path else os.getcwd()
        self._fs_model = _DataExplorerModel(root_path, self)
        self._fs_model.modelReset.connect(
            lambda: QtCore.QTimer.singleShot(1, self._dir_loaded)
        )

        self.menu_bar = QtWidgets.QMenuBar(self)

        self._setup_actions()
        self._setup_ui()

        # Selection that was used to display the current file info
        self._displayed_selection: list[pathlib.Path] = []

        if loader_name:
            self._loader_combo.setCurrentText(loader_name)

        self._dir_loaded()

    @property
    def loader_name(self) -> str:
        """Name of the selected loader."""
        return self._loader_combo.currentText()

    def _setup_actions(self) -> None:
        self._to_manager_act = QtWidgets.QAction("&Open in Manager", self)
        self._to_manager_act.triggered.connect(self.to_manager)
        self._to_manager_act.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        self._to_manager_act.setToolTip("Open the selected file(s) in ImageToolManager")

        self._close_act = QtWidgets.QAction("&Close Window", self)
        self._close_act.triggered.connect(self.close)
        self._close_act.setShortcut(QtGui.QKeySequence.StandardKey.Close)

        fm_name = "Finder" if sys.platform == "darwin" else "File Explorer"
        self._finder_act = QtWidgets.QAction(f"Reveal in {fm_name}", self)
        self._finder_act.setToolTip(f"Open the current folder in {fm_name}")
        self._finder_act.triggered.connect(
            lambda: erlab.utils.misc.open_in_file_manager(
                self._fs_model.file_system.path
            )
        )

        self._open_dir_act = QtWidgets.QAction("&Open Folder...", self)
        self._open_dir_act.triggered.connect(self._choose_directory)
        self._open_dir_act.setShortcut(QtGui.QKeySequence("Ctrl+Shift+O"))
        self._open_dir_act.setToolTip("Choose a directory to browse")

        self._reload_act = QtWidgets.QAction("Reload Folder", self)
        self._reload_act.triggered.connect(self._fs_model.reload)
        self._reload_act.setShortcut(QtGui.QKeySequence.StandardKey.Refresh)
        self._reload_act.setToolTip("Refresh the current directory contents")

        self._climb_up_act = QtWidgets.QAction("Enclosing Folder", self)
        self._climb_up_act.triggered.connect(self._fs_model.climb_up)
        self._climb_up_act.setShortcut(
            QtGui.QKeySequence("Ctrl+Up" if sys.platform == "darwin" else "Alt+Up")
        )
        self._climb_up_act.setToolTip("Go up one directory level")

        # Populate the menu bar
        file_menu = typing.cast(QtWidgets.QMenu, self.menu_bar.addMenu("&File"))
        file_menu.addAction(self._to_manager_act)
        file_menu.addAction(self._close_act)
        file_menu.addAction(self._finder_act)
        file_menu.addSeparator()
        file_menu.addAction(self._open_dir_act)
        file_menu.addAction(self._reload_act)
        file_menu.addAction(self._climb_up_act)

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        self.setLayout(layout)

        top_widget = QtWidgets.QWidget(self)
        top_layout = QtWidgets.QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_widget.setLayout(top_layout)
        top_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Maximum
        )
        layout.addWidget(top_widget)

        top_layout.addWidget(
            erlab.interactive.utils.IconActionButton(self._open_dir_act, "mdi6.folder")
        )

        top_layout.addWidget(
            erlab.interactive.utils.IconActionButton(self._reload_act, "mdi6.refresh")
        )

        top_layout.addWidget(
            erlab.interactive.utils.IconActionButton(
                self._climb_up_act, "mdi6.arrow-up"
            )
        )

        top_layout.addWidget(
            erlab.interactive.utils.IconActionButton(
                self._to_manager_act, "mdi6.chart-tree"
            )
        )

        top_layout.addWidget(
            erlab.interactive.utils.IconActionButton(
                self._finder_act, "mdi6.apple-finder"
            )
        )

        top_layout.addStretch()

        top_layout.addWidget(QtWidgets.QLabel("Loader"))
        self._loader_combo = _LoaderWidget()
        # self._loader_combo = QtWidgets.QComboBox()
        # self._loader_combo.addItems(erlab.io.loaders.keys())
        # for i, k in enumerate(erlab.io.loaders.keys()):
        #     loader = erlab.io.loaders[k]
        #     if hasattr(loader, "description"):
        #         self._loader_combo.setItemData(
        #             i, loader.description, QtCore.Qt.ItemDataRole.ToolTipRole
        #         )
        self._loader_combo.currentIndexChanged.connect(self._on_selection_changed)

        top_layout.addWidget(self._loader_combo)

        splitter = QtWidgets.QSplitter(self)
        splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        self._tree_view = _DataExplorerTreeView(self)
        self._tree_view.setModel(self._fs_model)
        self._tree_view.selectionModel().selectionChanged.connect(
            self._on_selection_changed
        )
        self._tree_view.doubleClicked.connect(self.to_manager)
        splitter.addWidget(self._tree_view)

        self._text_edit = QtWidgets.QTextEdit()
        self._text_edit.setText(self.TEXT_NONE_SELECTED)
        self._text_edit.setReadOnly(True)
        splitter.addWidget(self._text_edit)

        self.setMinimumWidth(487)
        self.setMinimumHeight(301)
        self.resize(974, 602)

    @QtCore.Slot()
    def _dir_loaded(self):
        """Slot to be called when a directory is loaded."""
        self._tree_view.resizeColumnToContents(0)

    @property
    def _threadpool(self) -> QtCore.QThreadPool:
        return typing.cast(QtCore.QThreadPool, QtCore.QThreadPool.globalInstance())

    @property
    def _current_selection(self) -> list[pathlib.Path]:
        """Currently selected files."""
        return self._tree_view.selected_paths

    @property
    def _up_to_date(self) -> bool:
        """Whether the displayed file info is up to date."""
        return set(self._displayed_selection) == set(self._current_selection)

    @QtCore.Slot()
    def _choose_directory(self) -> None:
        """Open a dialog to choose a directory."""
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Directory", str(self._fs_model.file_system.path)
        )
        if directory:
            self._fs_model.set_root_path(directory)

    @QtCore.Slot()
    def _on_selection_changed(self) -> None:
        selected_files: list[pathlib.Path] = self._current_selection
        n_sel = len(selected_files)

        if n_sel == 1:
            # Show loading text only if loading takes more than 100 ms
            QtCore.QTimer.singleShot(100, self._show_loading_text_if_needed)
            worker = _ReprFetcher(
                selected_files[0],
                erlab.io.loaders[self.loader_name].load,
            )
            worker.signals.fetched.connect(self._show_file_info)
            self._threadpool.start(worker)

        else:
            self._text_edit.setText(
                self.TEXT_NONE_SELECTED if n_sel == 0 else self.TEXT_MULTIPLE_SELECTED
            )
            self._displayed_selection = selected_files

    @QtCore.Slot()
    def _show_loading_text_if_needed(self) -> None:
        if not self._up_to_date:
            self._text_edit.setText("Loading...")

    @QtCore.Slot(str, str)
    def _show_file_info(self, file_path: str, text: str) -> None:
        selected_files: list[pathlib.Path] = self._current_selection
        if len(selected_files) == 1 and selected_files[0] == pathlib.Path(file_path):
            self._text_edit.setHtml(self._parse_file_info(text))
            self._displayed_selection = selected_files

    @staticmethod
    def _parse_file_info(text: str) -> str:
        if hasattr(QtGui.QPalette.ColorRole, "Accent"):
            accent_color = QtWidgets.QApplication.palette().accent().color().name()
            text = text.replace(
                erlab.utils.formatting._DEFAULT_ACCENT_COLOR, accent_color
            )
        return text

    @QtCore.Slot()
    def to_manager(self) -> None:
        """Open the selected files in ImageTool Manager."""
        if not erlab.interactive.imagetool.manager.is_running():
            QtWidgets.QMessageBox.critical(
                self,
                "ImageTool Manager not running",
                "The ImageTool Manager is not running. "
                "Start the ImageTool Manager application and try again.",
            )
        else:
            erlab.interactive.imagetool.manager.load_in_manager(
                self._current_selection, self.loader_name
            )

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent | None) -> None:
        """Handle drag-and-drop operations entering the window."""
        if event:
            mime_data: QtCore.QMimeData | None = event.mimeData()
            if mime_data and mime_data.hasUrls():
                event.acceptProposedAction()
            else:
                event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent | None) -> None:
        """Handle drag-and-drop operations dropping folders into the window."""
        if event:
            mime_data: QtCore.QMimeData | None = event.mimeData()
            if mime_data and mime_data.hasUrls():
                urls = mime_data.urls()
                file_paths: list[pathlib.Path] = [
                    pathlib.Path(url.toLocalFile()) for url in urls
                ]
                if len(file_paths) == 1 and file_paths[0].is_dir():
                    self._fs_model.set_root_path(file_paths[0])


def data_explorer(
    directory: str | os.PathLike | None = None,
    loader_name: str | None = None,
    *,
    execute: bool | None = None,
) -> None:
    """Start the data explorer.

    Data explorer is a tool to browse and load ARPES data files with a file manager-like
    interface. Data attributes of supported files can be quickly inspected, and can be
    loaded into ImageToolManager for further analysis.

    The data explorer can be started from the command line as a standalone application
    with the following command:

    .. code-block:: bash

        python -m erlab.interactive.explorer

    Also, it can be opened from the GUI by selecting "File" -> "Data Explorer" in
    ImageToolManager.

    Parameters
    ----------
    directory
        Initial directory to display in the explorer.
    loader_name
        Name of the loader to use to load the data. The loader must be registered in
        :attr:`erlab.io.loaders`.
    """
    with erlab.interactive.utils.setup_qapp(execute):
        win = _DataExplorer(root_path=directory, loader_name=loader_name)
        win.show()
        win.raise_()
        win.activateWindow()


if __name__ == "__main__":
    data_explorer()
