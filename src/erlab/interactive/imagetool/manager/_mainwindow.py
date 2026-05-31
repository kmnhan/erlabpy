from __future__ import annotations

import gc
import logging
import sys
import typing

from qtpy import QtCore, QtGui, QtWidgets

import erlab
import erlab.interactive.imagetool.slicer
from erlab.interactive._dask import DaskMenu
from erlab.interactive.imagetool.manager import _server as _manager_server
from erlab.interactive.imagetool.manager._actions import _ActionsMixin
from erlab.interactive.imagetool.manager._dependency import _ManagerDependencyTracker
from erlab.interactive.imagetool.manager._details_panel import _DetailsPanelMixin
from erlab.interactive.imagetool.manager._lineage import _LineageMixin
from erlab.interactive.imagetool.manager._linking import _ManagerLinkRegistry
from erlab.interactive.imagetool.manager._metadata import _ManagerToolMetadataQueue
from erlab.interactive.imagetool.manager._modelview import _ImageToolWrapperTreeView
from erlab.interactive.imagetool.manager._registry import (
    activate_manager_record,
    reserve_manager_record,
    unregister_manager_record,
)
from erlab.interactive.imagetool.manager._tool_graph import _ManagerToolGraph
from erlab.interactive.imagetool.manager._widgets import (
    _LINKER_COLORS,
    _SHM_NAME,
    _ApplicationQuitFilter,
    _HeightForWidthFrame,
    _MetadataDerivationListWidget,
    _SingleImagePreview,
    _StandaloneAppSpec,
    _WarningEmitter,
    _WarningNotificationHandler,
    _WidgetsMixin,
)
from erlab.interactive.imagetool.manager._workspace_io import _WorkspaceIOMixin
from erlab.interactive.imagetool.manager._workspace_state import _ManagerWorkspaceState
from erlab.interactive.imagetool.manager._wrapper import (
    _ImageToolWrapper,
    _ManagedWindowNode,
)

if typing.TYPE_CHECKING:
    import datetime

    import numpy as np

    from erlab.interactive.imagetool._mainwindow import ImageTool
    from erlab.interactive.imagetool.manager._io import _MultiFileHandler
    from erlab.interactive.imagetool.provenance_framework import ToolProvenanceSpec
    from erlab.interactive.imagetool.provenance_operations import (
        ImageToolSelectionSourceBinding,
    )

logger = logging.getLogger(__name__)


class ImageToolManager(
    _WorkspaceIOMixin,
    _LineageMixin,
    _DetailsPanelMixin,
    _ActionsMixin,
    _WidgetsMixin,
):
    """The ImageToolManager window.

    This class implements a GUI application for managing multiple ImageTool windows.

    Users do not need to create an instance of this class directly. Instead, use the
    command line script ``itool-manager`` or the function :func:`main
    <erlab.interactive.imagetool.manager.main>` to start the application.

    Signals
    -------
    sigLinkersChanged()
        Signal emitted when the linker state is changed.

    """

    sigLinkersChanged = QtCore.Signal()  #: :meta private:
    _sigReloadLinkers = QtCore.Signal()  #: Emitted when linker state needs refreshing

    _sigDataReplaced = QtCore.Signal()  #: :meta private:
    # Signal emitted when data is replaced in the manager, for testing purposes.

    _sigReplyData = QtCore.Signal(object)  #: :meta private:
    # Signal emitted to reply data requests.

    _sigWatchedDataEdited = QtCore.Signal(str, str, str)  #: :meta private:
    # Signal emitted to notify ipython watchers of data changes.

    def __init__(self) -> None:
        super().__init__()

        # Initialize warning notifications
        self._warning_emitter = _WarningEmitter(self)
        self._warning_emitter.warning_received.connect(self._show_alert)
        self._warning_handler = _WarningNotificationHandler(self._warning_emitter)
        logging.getLogger().addHandler(self._warning_handler)
        self._alert_dialogs: list[erlab.interactive.utils.MessageDialog] = []
        self._ignored_warning_messages: set[str] = set()

        # Setup uncaught exception handler
        self._previous_excepthook = sys.excepthook
        sys.excepthook = self._handle_uncaught_exception

        self._manager_record = reserve_manager_record(host=_manager_server.HOST_IP)
        self.manager_index = self._manager_record.index
        self._tool_graph = _ManagerToolGraph()
        self._dependency_tracker = _ManagerDependencyTracker(self._tool_graph)

        try:
            (
                self.server,
                self.watcher_server,
                port,
                watch_port,
            ) = self._start_manager_servers()
            self._manager_record = activate_manager_record(
                self._manager_record.internal_id,
                port=port,
                watch_port=watch_port,
            )
        except Exception:
            unregister_manager_record(self._manager_record.internal_id)
            raise

        self._registry_heartbeat_timer = QtCore.QTimer(self)
        self._registry_heartbeat_timer.setInterval(3000)
        self._registry_heartbeat_timer.timeout.connect(self._refresh_manager_record)
        self._registry_heartbeat_timer.start()

        # Shared memory for detecting multiple instances
        # No longer used starting from v3.8.2, but kept for backward compatibility
        self._shm = QtCore.QSharedMemory(_SHM_NAME)
        self._shm.create(1)  # Create segment so that it can be attached to

        self.menu_bar: QtWidgets.QMenuBar = typing.cast(
            "QtWidgets.QMenuBar", self.menuBar()
        )

        self._workspace_state = _ManagerWorkspaceState()
        self._tool_metadata_queue = _ManagerToolMetadataQueue(
            self, self._flush_pending_tool_metadata_updates
        )
        self._update_workspace_window_title()

        qapp = QtWidgets.QApplication.instance()
        self._application_quit_filter: _ApplicationQuitFilter | None = None
        if isinstance(qapp, QtWidgets.QApplication):
            self._application_quit_filter = _ApplicationQuitFilter(self)
            qapp.installEventFilter(self._application_quit_filter)

        self._link_registry = _ManagerLinkRegistry()

        # Stores additional analysis tools opened from child ImageTool windows
        self._additional_windows: dict[str, QtWidgets.QWidget] = {}
        self._standalone_app_windows: dict[str, QtWidgets.QWidget] = {}
        self._standalone_app_specs: dict[str, _StandaloneAppSpec] = {
            "explorer": _StandaloneAppSpec(
                key="explorer",
                menu="file",
                text="Data Explorer",
                tooltip="Show the data explorer window",
                shortcut="Ctrl+E",
                icon_name="drive-harddisk",
                factory=self._create_explorer_window,
            ),
            "ptable": _StandaloneAppSpec(
                key="ptable",
                menu="apps",
                text="Periodic Table",
                tooltip="Show the periodic table window",
                shortcut="Ctrl+Shift+P",
                icon_name="applications-science",
                factory=self._create_ptable_window,
            ),
        }
        self._standalone_app_actions: dict[str, QtWidgets.QAction] = {}

        # Store progress bar widgets
        self._progress_bars: dict[int, QtWidgets.QProgressDialog] = {}

        self._bulk_remove_depth: int = 0

        # Initialize actions
        self.settings_action = QtWidgets.QAction("Settings", self)
        self.settings_action.triggered.connect(self.open_settings)
        self.settings_action.setShortcut(QtGui.QKeySequence.StandardKey.Preferences)
        self.settings_action.setToolTip("Open settings")
        self.settings_action.setIcon(QtGui.QIcon.fromTheme("preferences-system"))

        self.show_action = QtWidgets.QAction("Show", self)
        self.show_action.triggered.connect(self.show_selected)
        self.show_action.setShortcut(
            "Return"
            if sys.platform == "darwin"
            else QtGui.QKeySequence.StandardKey.InsertParagraphSeparator
        )
        self.show_action.setToolTip("Show selected windows")

        self.hide_action = QtWidgets.QAction("Hide", self)
        self.hide_action.triggered.connect(self.hide_selected)
        self.hide_action.setShortcut("Ctrl+W")
        self.hide_action.setToolTip("Hide selected windows")

        self.gc_action = QtWidgets.QAction("Run Garbage Collection", self)
        self.gc_action.triggered.connect(self.garbage_collect)
        self.gc_action.setToolTip("Run garbage collection to free up memory")
        self.gc_action.setIcon(QtGui.QIcon.fromTheme("user-trash"))

        self.open_action = QtWidgets.QAction("Add &Data Files…", self)
        self.open_action.setObjectName("manager_add_data_files_action")
        self.open_action.triggered.connect(self.open)
        self.open_action.setToolTip("Load data files as new ImageTool rows")

        self.new_manager_action = QtWidgets.QAction("New Manager Window", self)
        self.new_manager_action.setObjectName("manager_new_instance_action")
        self.new_manager_action.triggered.connect(self.open_new_manager_instance)
        self.new_manager_action.setToolTip("Open another ImageTool Manager window")
        self.new_manager_action.setIcon(QtGui.QIcon.fromTheme("window-new"))

        self.save_action = QtWidgets.QAction("&Save", self)
        self.save_action.setObjectName("manager_save_workspace_action")
        self.save_action.setShortcut(QtGui.QKeySequence.StandardKey.Save)
        self.save_action.setToolTip("Save this workspace")
        self.save_action.setIcon(QtGui.QIcon.fromTheme("document-save"))
        self.save_action.triggered.connect(self.save)

        self.save_as_action = QtWidgets.QAction("Save Workspace &As…", self)
        self.save_as_action.setObjectName("manager_save_workspace_as_action")
        self.save_as_action.setShortcut(QtGui.QKeySequence.StandardKey.SaveAs)
        self.save_as_action.setToolTip(
            "Save this workspace to a new file and use that file for future saves"
        )
        self.save_as_action.setIcon(QtGui.QIcon.fromTheme("document-save-as"))
        self.save_as_action.triggered.connect(self.save_as)

        self.compact_workspace_action = QtWidgets.QAction("Compact Workspace", self)
        self.compact_workspace_action.setObjectName("manager_compact_workspace_action")
        self.compact_workspace_action.setToolTip(
            "Rewrite this workspace file to remove unused space"
        )
        self.compact_workspace_action.triggered.connect(self.compact_workspace)

        self.workspace_properties_action = QtWidgets.QAction(
            "Workspace Properties", self
        )
        self.workspace_properties_action.setObjectName(
            "manager_workspace_properties_action"
        )
        self.workspace_properties_action.setMenuRole(QtWidgets.QAction.MenuRole.NoRole)
        self.workspace_properties_action.setShortcut(QtGui.QKeySequence("Alt+Return"))
        self.workspace_properties_action.setToolTip(
            "Show properties for the current workspace"
        )
        self.workspace_properties_action.setIcon(
            QtGui.QIcon.fromTheme("document-properties")
        )
        self.workspace_properties_action.triggered.connect(
            self.show_workspace_properties
        )

        self.load_action = QtWidgets.QAction("&Open Workspace…", self)
        self.load_action.setObjectName("manager_open_workspace_action")
        self.load_action.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        self.load_action.setToolTip("Replace this workspace with a workspace file")
        self.load_action.setIcon(QtGui.QIcon.fromTheme("document-open"))
        self.load_action.triggered.connect(self.load)

        self.open_recent_menu = QtWidgets.QMenu("Open &Recent", self)
        self.open_recent_menu.setObjectName("manager_open_recent_menu")
        open_recent_action = self.open_recent_menu.menuAction()
        # Qt creates a menu action for every QMenu; this keeps type narrowing explicit.
        if open_recent_action is None:  # pragma: no cover
            raise RuntimeError("Open Recent menu action was not created")
        open_recent_action.setObjectName("manager_open_recent_menu_action")
        open_recent_action.setIcon(QtGui.QIcon.fromTheme("document-open-recent"))
        self.open_recent_menu.setToolTipsVisible(True)
        self.open_recent_menu.aboutToShow.connect(self._populate_open_recent_menu)
        self._refresh_open_recent_menu_action()

        self.import_workspace_action = QtWidgets.QAction(
            "Add Windows From &Workspace…", self
        )
        self.import_workspace_action.setObjectName(
            "manager_add_windows_from_workspace_action"
        )
        self.import_workspace_action.setToolTip(
            "Add selected windows from another workspace file"
        )
        self.import_workspace_action.setIcon(QtGui.QIcon.fromTheme("list-add"))
        self.import_workspace_action.triggered.connect(self.import_workspace)

        self.remove_action = QtWidgets.QAction("Remove", self)
        self.remove_action.triggered.connect(self.remove_selected)
        self.remove_action.setShortcut(QtGui.QKeySequence.StandardKey.Delete)
        self.remove_action.setToolTip("Remove selected windows")

        self.rename_action = QtWidgets.QAction("Rename", self)
        self.rename_action.triggered.connect(self.rename_selected)
        self.rename_action.setToolTip("Rename selected windows")

        self.duplicate_action = QtWidgets.QAction("Duplicate", self)
        self.duplicate_action.triggered.connect(self.duplicate_selected)
        self.duplicate_action.setToolTip("Duplicate selected windows")
        self.duplicate_action.setIcon(QtGui.QIcon.fromTheme("edit-copy"))

        self.promote_action = QtWidgets.QAction("Promote Window", self)
        self.promote_action.triggered.connect(self.promote_selected)
        self.promote_action.setToolTip(
            "Promote the selected nested ImageTool to a top-level window"
        )
        self.promote_action.setIcon(QtGui.QIcon.fromTheme("go-up"))

        self.reindex_action = QtWidgets.QAction("Reset Index", self)
        self.reindex_action.triggered.connect(self.reindex)
        self.reindex_action.setToolTip("Reset indices of all windows")

        self.link_action = QtWidgets.QAction("Link", self)
        self.link_action.triggered.connect(lambda _checked=False: self.link_selected())
        self.link_action.setShortcut(QtGui.QKeySequence("Ctrl+L"))
        self.link_action.setToolTip("Link selected windows")

        self.unlink_action = QtWidgets.QAction("Unlink", self)
        self.unlink_action.triggered.connect(
            lambda _checked=False: self.unlink_selected()
        )
        self.unlink_action.setShortcut(QtGui.QKeySequence("Ctrl+Shift+L"))
        self.unlink_action.setToolTip("Unlink selected windows")

        self.offload_action = QtWidgets.QAction("Offload to Workspace", self)
        self.offload_action.triggered.connect(self.offload_selected_to_workspace)
        self.offload_action.setToolTip(
            "Free this data from memory and use dask-backed data from the "
            "workspace file"
        )

        self.console_action = QtWidgets.QAction("Console", self)
        self.console_action.triggered.connect(self.toggle_console)
        self.console_action.setShortcut(QtGui.QKeySequence("Ctrl+J"))
        self.console_action.setToolTip("Toggle console window")
        self.console_action.setIcon(QtGui.QIcon.fromTheme("utilities-terminal"))

        self.preview_action = QtWidgets.QAction("Preview on Hover", self)
        self.preview_action.setCheckable(True)
        self.preview_action.setToolTip("Show preview on hover")

        self.store_action = QtWidgets.QAction("Store with IPython", self)
        self.store_action.triggered.connect(self.store_selected)
        self.store_action.setToolTip("Store selected data with IPython")

        self.explorer_action = self._create_standalone_app_action("explorer")
        self.ptable_action = self._create_standalone_app_action("ptable")

        self.concat_action = QtWidgets.QAction("Concatenate", self)
        self.concat_action.triggered.connect(self.concat_selected)
        self.concat_action.setToolTip("Concatenate data in selected windows")

        self.reload_action = QtWidgets.QAction("Reload Data", self)
        self.reload_action.triggered.connect(self.reload_selected)
        self.reload_action.setToolTip(
            "Reload selected data from its saved files, parent, or inputs"
        )
        self.reload_action.setIcon(QtGui.QIcon.fromTheme("view-refresh"))
        self.reload_action.setVisible(False)

        self.unwatch_action = QtWidgets.QAction("Stop Watching", self)
        self.unwatch_action.triggered.connect(self.unwatch_selected)
        self.unwatch_action.setToolTip("Stop watching selected windows")
        self.unwatch_action.setIcon(QtGui.QIcon.fromTheme("process-stop"))
        self.unwatch_action.setVisible(False)

        self.source_update_action = QtWidgets.QAction("Automatic Updates…", self)
        self.source_update_action.triggered.connect(self.show_selected_source_updates)
        self.source_update_action.setToolTip(
            "Turn automatic updates on or off for the selected child window"
        )
        self.source_update_action.setIcon(QtGui.QIcon.fromTheme("sync-synchronizing"))
        self.source_update_action.setVisible(False)

        self.about_action = QtWidgets.QAction("About", self)
        self.about_action.setIcon(QtGui.QIcon.fromTheme("help-about"))
        self.about_action.triggered.connect(self.about)

        self.check_update_action = QtWidgets.QAction("Check for Updates", self)
        self.check_update_action.setMenuRole(
            QtWidgets.QAction.MenuRole.ApplicationSpecificRole
        )
        self.check_update_action.triggered.connect(self.check_for_updates)
        self.check_update_action.setIcon(
            QtGui.QIcon.fromTheme("software-update-available")
        )
        self.check_update_action.setVisible(erlab.utils.misc._IS_PACKAGED)

        release_notes_action, open_docs_action, report_issue_action = (
            erlab.interactive.utils.make_help_actions(self)
        )

        self.open_log_folder_action = QtWidgets.QAction("Open Log Directory", self)
        self.open_log_folder_action.triggered.connect(self.open_log_directory)

        # Populate menu bar
        self.file_menu: QtWidgets.QMenu = typing.cast(
            "QtWidgets.QMenu", self.menu_bar.addMenu("&File")
        )
        self._file_menu_action = self.file_menu.menuAction()
        self.file_menu.setObjectName("manager_file_menu")
        self.file_menu.aboutToShow.connect(self._refresh_open_recent_menu_action)
        self.file_menu.addAction(self.load_action)
        self.file_menu.addMenu(self.open_recent_menu)
        self.file_menu.addAction(self.save_action)
        self.file_menu.addAction(self.save_as_action)
        self.file_menu.addAction(self.compact_workspace_action)
        self.file_menu.addAction(self.workspace_properties_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.open_action)
        self.file_menu.addAction(self.import_workspace_action)
        self.file_menu.addAction(self.explorer_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.new_manager_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.store_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.gc_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.show_action)
        self.file_menu.addAction(self.hide_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.remove_action)
        self.file_menu.addAction(self.offload_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.settings_action)

        self.edit_menu: QtWidgets.QMenu = typing.cast(
            "QtWidgets.QMenu", self.menu_bar.addMenu("&Edit")
        )
        self._edit_menu_action = self.edit_menu.menuAction()
        self.edit_menu.setObjectName("manager_edit_menu")
        self.edit_menu.addAction(self.reindex_action)
        self.edit_menu.addSeparator()
        self.edit_menu.addAction(self.concat_action)
        self.edit_menu.addAction(self.duplicate_action)
        self.edit_menu.addAction(self.promote_action)
        self.edit_menu.addSeparator()
        self.edit_menu.addAction(self.rename_action)
        self.edit_menu.addAction(self.link_action)
        self.edit_menu.addAction(self.unlink_action)

        self.view_menu: QtWidgets.QMenu = typing.cast(
            "QtWidgets.QMenu", self.menu_bar.addMenu("&View")
        )
        self._view_menu_action = self.view_menu.menuAction()
        self.view_menu.setObjectName("manager_view_menu")
        self.view_menu.addAction(self.console_action)
        self.view_menu.addSeparator()
        self.view_menu.addAction(self.preview_action)
        self.view_menu.addSeparator()

        self.apps_menu: QtWidgets.QMenu = typing.cast(
            "QtWidgets.QMenu", self.menu_bar.addMenu("&Apps")
        )
        self._apps_menu_action = self.apps_menu.menuAction()
        self.apps_menu.setObjectName("manager_apps_menu")
        self.apps_menu.addAction(self.ptable_action)

        self._dask_menu = DaskMenu(self, "Dask")
        self.menu_bar.addMenu(self._dask_menu)
        self._dask_menu_action = self._dask_menu.menuAction()

        self.help_menu: QtWidgets.QMenu = typing.cast(
            "QtWidgets.QMenu", self.menu_bar.addMenu("&Help")
        )
        self._help_menu_action = self.help_menu.menuAction()
        self.help_menu.setObjectName("manager_help_menu")
        self.help_menu.addAction(self.about_action)
        self.help_menu.addAction(self.check_update_action)
        self.help_menu.addAction(release_notes_action)
        self.help_menu.addSeparator()
        self.help_menu.addAction(open_docs_action)
        self.help_menu.addAction(report_issue_action)
        self.help_menu.addSeparator()
        self.help_menu.addAction(self.open_log_folder_action)

        # Initialize sidebar buttons linked to actions
        self.open_button = erlab.interactive.utils.IconActionButton(
            self.open_action, "mdi6.folder-file"
        )
        self.remove_button = erlab.interactive.utils.IconActionButton(
            self.remove_action, "mdi6.window-close"
        )
        self.rename_button = erlab.interactive.utils.IconActionButton(
            self.rename_action, "mdi6.rename"
        )
        self.link_button = erlab.interactive.utils.IconActionButton(
            self.link_action, "mdi6.link-variant"
        )
        self.unlink_button = erlab.interactive.utils.IconActionButton(
            self.unlink_action, "mdi6.link-variant-off"
        )
        self.preview_button = erlab.interactive.utils.IconActionButton(
            self.preview_action, on="ph.eye", off="ph.eye-slash"
        )

        # Initialize GUI
        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.setCentralWidget(main_splitter)

        # Construct left side of splitter
        left_container = QtWidgets.QWidget()
        left_layout = QtWidgets.QHBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        main_splitter.addWidget(left_container)

        titlebar = QtWidgets.QWidget()
        titlebar_layout = QtWidgets.QVBoxLayout()
        titlebar.setLayout(titlebar_layout)
        titlebar_layout.addWidget(self.open_button)
        titlebar_layout.addWidget(self.remove_button)
        titlebar_layout.addWidget(self.rename_button)
        titlebar_layout.addWidget(self.link_button)
        titlebar_layout.addWidget(self.unlink_button)
        titlebar_layout.addStretch()
        left_layout.addWidget(titlebar)

        self.tree_view = _ImageToolWrapperTreeView(self)
        self.tree_view._selection_model.selectionChanged.connect(self._update_actions)
        self.tree_view._selection_model.selectionChanged.connect(self._update_info)
        self.tree_view._model.dataChanged.connect(self._update_info)
        left_layout.addWidget(self.tree_view)

        # Construct right side of splitter
        right_panel = QtWidgets.QWidget(self)
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        main_splitter.addWidget(right_panel)

        right_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        right_layout.addWidget(right_splitter, 1)

        self.text_box = QtWidgets.QTextEdit(self)
        self.text_box.setReadOnly(True)
        right_splitter.addWidget(self.text_box)

        self.preview_widget = _SingleImagePreview(self)
        right_splitter.addWidget(self.preview_widget)

        self.metadata_group = _HeightForWidthFrame(self)
        self.metadata_group.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.metadata_group.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )
        metadata_layout = QtWidgets.QVBoxLayout(self.metadata_group)
        metadata_layout.setContentsMargins(0, 0, 0, 0)
        metadata_layout.setSpacing(4)
        self.metadata_group.setLayout(metadata_layout)

        self.metadata_details_widget = _HeightForWidthFrame(self.metadata_group)
        self.metadata_details_layout = QtWidgets.QGridLayout(
            self.metadata_details_widget
        )
        self.metadata_details_layout.setContentsMargins(0, 0, 0, 0)
        self.metadata_details_layout.setHorizontalSpacing(8)
        self.metadata_details_layout.setVerticalSpacing(2)
        self.metadata_details_layout.setColumnStretch(1, 1)
        self.metadata_details_widget.setLayout(self.metadata_details_layout)
        self.metadata_details_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )
        self.metadata_details_widget.setVisible(False)
        metadata_layout.addWidget(self.metadata_details_widget)
        self._metadata_detail_labels: dict[str, QtWidgets.QLabel] = {}
        self._metadata_monospace_font = QtGui.QFontDatabase.systemFont(
            QtGui.QFontDatabase.SystemFont.FixedFont
        )

        self.metadata_derivation_list = _MetadataDerivationListWidget(
            self.metadata_group
        )
        self.metadata_derivation_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.metadata_derivation_list.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.metadata_derivation_list.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.metadata_derivation_list.setVerticalScrollMode(
            QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        self.metadata_derivation_list.setTextElideMode(
            QtCore.Qt.TextElideMode.ElideRight
        )
        self.metadata_derivation_list.setUniformItemSizes(True)
        self.metadata_derivation_list.setAlternatingRowColors(False)
        self.metadata_derivation_list.copy_requested.connect(
            self._copy_selected_derivation_code
        )
        self.metadata_derivation_list.context_menu_requested.connect(
            self._show_metadata_derivation_menu
        )
        self.metadata_derivation_list.setVisible(False)
        metadata_layout.addWidget(self.metadata_derivation_list)
        right_layout.addWidget(self.metadata_group, 0)

        # Set initial splitter sizes
        right_splitter.setSizes([280, 140])
        main_splitter.setSizes([100, 150])

        # Store most recent name filter and directory for new windows
        self._recent_name_filter: str | None = None
        self._recent_directory: str | None = None
        self._recent_loader_extensions_by_filter: dict[str, dict[str, typing.Any]] = {}
        self._metadata_full_code_available = False
        self._metadata_node_uid: str | None = None
        self._metadata_copy_selected_action = QtGui.QAction("Copy Selected Code", self)
        self._metadata_copy_selected_action.setObjectName(
            "manager_copy_selected_code_action"
        )
        self._metadata_copy_selected_action.triggered.connect(
            self._copy_selected_derivation_code
        )
        self._metadata_copy_full_action = QtGui.QAction("Copy Full Code", self)
        self._metadata_copy_full_action.setObjectName("manager_copy_full_code_action")
        self._metadata_copy_full_action.triggered.connect(
            self._copy_full_derivation_code
        )

        self.sigLinkersChanged.connect(self._update_actions)
        self.sigLinkersChanged.connect(self.tree_view.refresh)
        self._sigReloadLinkers.connect(self._request_reload_linkers)
        self._update_actions()
        self._update_info()

        # Golden ratio :)
        self.setMinimumWidth(301)
        self.setMinimumHeight(487)
        self.resize(487, 487)

        # Install event filter for keyboard shortcuts
        self._kb_filter = erlab.interactive.utils.KeyboardEventFilter(self)
        for widget in (
            self.text_box,
            self.metadata_derivation_list,
        ):
            widget.installEventFilter(self._kb_filter)

        # File handlers for multithreaded file loading
        self._file_handlers: set[_MultiFileHandler] = set()

        # Initialize status bar
        self._status_bar.showMessage("")

    def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:
        """Handle proper termination of resources before closing the application."""
        logger.debug("Closing ImageTool Manager...")
        previous_closing_workspace_document = self._workspace_state.closing_document
        self._workspace_state.closing_document = True
        try:
            if not self._confirm_save_dirty_workspace(
                "Closing this manager will discard unsaved workspace changes."
            ):
                if event:
                    event.ignore()
                return

            logger.debug("Waiting for file handlers to finish...")
            if len(self._file_handlers) > 0:  # pragma: no cover
                with erlab.interactive.utils.wait_dialog(
                    self, "Waiting for file operations to finish..."
                ):
                    for handler in list(self._file_handlers):
                        handler.wait()

            logger.debug("Stopping servers...")
            self._registry_heartbeat_timer.stop()
            self._stop_servers()
            unregister_manager_record(self._manager_record.internal_id)

            self._compact_workspace_before_shutdown()

            logger.debug("Removing all ImageTool windows...")
            with self._workspace_load_context():
                self.remove_all_tools()

            logger.debug("Closing additional windows...")
            for widget in dict(self._additional_windows).values():
                widget.close()
                widget.deleteLater()

            logger.debug("Removing event filters...")
            qapp = QtWidgets.QApplication.instance()
            if (
                isinstance(qapp, QtWidgets.QApplication)
                and self._application_quit_filter is not None
            ):
                qapp.removeEventFilter(self._application_quit_filter)
                self._application_quit_filter = None
            for widget in (
                self.text_box,
                self.metadata_derivation_list,
            ):
                widget.removeEventFilter(self._kb_filter)
            self.tree_view._delegate._cleanup_filter()

            if hasattr(self, "console"):
                logger.debug("Shutting down console kernel...")
                self.console._console_widget.shutdown_kernel()
                self.console.close()
                self.console.deleteLater()

            if self._standalone_app_windows:
                logger.debug("Closing standalone apps...")
                self._close_standalone_apps()

            logger.debug("Releasing workspace lock...")
            self._release_workspace_lock()

            logger.debug("Closing dask client (if any)...")
            self._dask_menu.close_client()

            root_logger = logging.getLogger()
            if self._warning_handler in root_logger.handlers:  # pragma: no branch
                root_logger.removeHandler(self._warning_handler)

            self._clear_all_alerts()

            if sys.excepthook == self._handle_uncaught_exception:
                sys.excepthook = self._previous_excepthook

            super().closeEvent(event)
        finally:
            self._workspace_state.closing_document = previous_closing_workspace_document

    @property
    def ntools(self) -> int:
        """Number of ImageTool windows being handled by the manager."""
        return self._tool_graph.ntools

    @property
    def next_idx(self) -> int:
        """Index for the next window."""
        return self._tool_graph.next_index

    def _next_node_uid(self, preferred: str | None = None) -> str:
        return self._tool_graph.next_uid(preferred)

    def _consume_node_uid(self, uid: str) -> None:
        self._tool_graph.consume_uid(uid)

    def _register_root_wrapper(self, wrapper: _ImageToolWrapper) -> None:
        self._tool_graph.register_root(wrapper)

    def _register_child_node(self, node: _ManagedWindowNode) -> None:
        self._tool_graph.register_child(node)
        if node.tool_window is not None:
            node.tool_window._refresh_reload_data_action()

    def _unregister_node(self, uid: str) -> None:
        node = self._tool_graph.unregister_node(uid)
        if node is None:
            return
        self._dependency_tracker.clear_uid(uid)
        self._refresh_dependency_dependents(uid)

    def color_for_linker(
        self, linker: erlab.interactive.imagetool.viewer_linking.SlicerLinkProxy
    ) -> QtGui.QColor:
        """Get the color that should represent the given linker."""
        idx = self._link_registry.index(linker)
        return _LINKER_COLORS[idx % len(_LINKER_COLORS)]

    def linker_index(
        self, linker: erlab.interactive.imagetool.viewer_linking.SlicerLinkProxy
    ) -> int:
        return self._link_registry.index(linker)

    def add_imagetool(
        self,
        tool: ImageTool,
        *,
        show: bool = True,
        activate: bool = False,
        watched_var: tuple[str, str] | None = None,
        watched_workspace_link_id: str | None = None,
        watched_source_label: str | None = None,
        watched_source_uid: str | None = None,
        watched_connected: bool = True,
        source_input_ndim: int | None = None,
        source_input_dtype: np.dtype[typing.Any] | str | None = None,
        uid: str | None = None,
        provenance_spec: ToolProvenanceSpec | None = None,
        source_spec: ToolProvenanceSpec | None = None,
        source_binding: ImageToolSelectionSourceBinding | None = None,
        source_auto_update: bool = False,
        source_state: _ManagedWindowNode._source_state_type = "fresh",
        index: int | None = None,
        snapshot_token: str | None = None,
        created_time: datetime.datetime | str | bytes | None = None,
    ) -> int:
        """Add a new ImageTool window to the manager and show it.

        Parameters
        ----------
        tool
            ImageTool object to be added.
        show
            Whether to show the window after adding, by default `True`.
        activate
            Whether to focus on the window after adding, by default `False`.
        watched_var
            If the tool is created from a watched variable, this should be a tuple of
            the variable name and its unique ID.
        source_input_ndim
            Original dimensionality of the bound source before ImageTool-specific
            promotion (for example, promoted 1D inputs).

        Returns
        -------
        int
            Index of the added ImageTool window.
        """
        if provenance_spec is not None:
            tool.set_provenance_spec(provenance_spec)
        if index is None or index in self._tool_graph.root_wrappers:
            index = int(self.next_idx)
        else:
            index = int(index)
        wrapper = _ImageToolWrapper(
            self,
            index,
            self._next_node_uid(uid),
            tool,
            watched_var=watched_var,
            watched_workspace_link_id=watched_workspace_link_id,
            watched_source_label=watched_source_label,
            watched_source_uid=watched_source_uid,
            watched_connected=watched_connected,
            source_input_ndim=source_input_ndim,
            source_input_dtype=source_input_dtype,
            provenance_spec=provenance_spec,
            source_spec=source_spec,
            source_binding=source_binding,
            source_auto_update=source_auto_update,
            source_state=source_state,
            snapshot_token=snapshot_token,
            created_time=created_time,
        )
        self._register_root_wrapper(wrapper)
        wrapper.update_title()

        self._sigReloadLinkers.emit()

        if show:
            tool.show()

        if activate:
            tool.activateWindow()
            tool.raise_()

        # Add to view after initialization
        self.tree_view.imagetool_added(index)
        self._mark_node_added(wrapper.uid)

        return index

    @QtCore.Slot()
    def garbage_collect(self) -> None:
        """Run garbage collection to free up memory."""
        gc.collect()  # pragma: no cover

    # def __del__(self):
    # """Ensure proper cleanup of server thread when the manager is deleted."""
    # self._stop_server()
