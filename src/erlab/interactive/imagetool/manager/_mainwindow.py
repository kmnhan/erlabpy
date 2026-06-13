from __future__ import annotations

import contextlib
import gc
import logging
import re
import sys
import typing

from qtpy import QtCore, QtGui, QtWidgets

import erlab
import erlab.interactive.imagetool.slicer
from erlab.interactive._dask import DaskMenu
from erlab.interactive.imagetool.manager import _server as _manager_server
from erlab.interactive.imagetool.manager._actions import _ActionsController
from erlab.interactive.imagetool.manager._base import _ImageToolManagerBase
from erlab.interactive.imagetool.manager._dependency import _ManagerDependencyTracker
from erlab.interactive.imagetool.manager._details_panel import _DetailsPanelController
from erlab.interactive.imagetool.manager._heartbeat import _RegistryHeartbeatController
from erlab.interactive.imagetool.manager._lineage import _LineageController
from erlab.interactive.imagetool.manager._linking import _ManagerLinkRegistry
from erlab.interactive.imagetool.manager._metadata import _ManagerToolMetadataQueue
from erlab.interactive.imagetool.manager._modelview import _ImageToolWrapperTreeView
from erlab.interactive.imagetool.manager._provenance_edit import (
    _ProvenanceEditController,
)
from erlab.interactive.imagetool.manager._registry import (
    activate_manager_record,
    reserve_manager_record,
    unregister_manager_record,
)
from erlab.interactive.imagetool.manager._tool_graph import _ManagerToolGraph
from erlab.interactive.imagetool.manager._widgets import (
    _LINKER_COLORS,
    _SHM_NAME,
    _WORKSPACE_REBIND_KEEP_CHUNKS,
    _ApplicationQuitFilter,
    _HeightForWidthFrame,
    _manager_settings,
    _MetadataDerivationListWidget,
    _SingleImagePreview,
    _StandaloneAppSpec,
    _WarningEmitter,
    _WarningNotificationHandler,
    _WidgetsController,
)
from erlab.interactive.imagetool.manager._workspace_io import _WorkspaceIOController
from erlab.interactive.imagetool.manager._workspace_state import _ManagerWorkspaceState
from erlab.interactive.imagetool.manager._wrapper import (
    _ImageToolWrapper,
    _ManagedWindowNode,
)

if typing.TYPE_CHECKING:
    import datetime
    import os
    import pathlib
    from collections.abc import Callable, Iterable, Iterator, Mapping

    import numpy as np
    import xarray as xr

    from erlab.interactive.imagetool import provenance
    from erlab.interactive.imagetool._load_source import _LoadSourceDetails
    from erlab.interactive.imagetool._mainwindow import ImageTool
    from erlab.interactive.imagetool.manager import _workspace as _manager_workspace
    from erlab.interactive.imagetool.manager._dependency import _DependencyStatus
    from erlab.interactive.imagetool.manager._io import _MultiFileHandler
    from erlab.interactive.imagetool.manager._server import (
        _ManagerServer,
        _WatcherServer,
    )
    from erlab.interactive.imagetool.manager._widgets import (
        _ScriptRebuildResult,
        _WorkspaceDocumentAccess,
        _WorkspacePropertiesState,
    )
    from erlab.interactive.imagetool.manager._workspace_state import (
        _WorkspaceStateSnapshot,
    )
    from erlab.interactive.imagetool.manager._wrapper import _MetadataField
    from erlab.interactive.imagetool.viewer import ImageSlicerArea

logger = logging.getLogger(__name__)

_NEW_FIGURE_TARGET = "__new_figure__"
_FIGURE_VIEW_MODE_SETTINGS_KEY = "figures/view_mode"
_FIGURE_GALLERY_SIZE_SETTINGS_KEY = "figures/gallery_thumbnail_size"
_FIGURE_VIEW_MODE_LIST = "list"
_FIGURE_VIEW_MODE_GALLERY = "gallery"
_FIGURE_VIEW_MODES = (_FIGURE_VIEW_MODE_LIST, _FIGURE_VIEW_MODE_GALLERY)
_FIGURE_GALLERY_SIZE_MEDIUM = "medium"
_FIGURE_GALLERY_THUMBNAIL_SIZES = {
    "small": (112, 84),
    _FIGURE_GALLERY_SIZE_MEDIUM: (152, 114),
    "large": (216, 162),
}


class _AppendFigureTargetDialog(QtWidgets.QDialog):
    """Prompt for a Figure Composer target figure and axes selection."""

    def __init__(
        self,
        manager: ImageToolManager,
        figure_uids: tuple[str, ...],
        operation: typing.Any | None,
        *,
        allow_new_figure: bool = False,
    ) -> None:
        from erlab.interactive._figurecomposer._widgets import (
            _AxesSelectorWidget,
            _GridSpecViewWidget,
        )

        super().__init__(manager)
        self._manager = manager
        self._figure_uids = figure_uids
        self._operation = operation
        self.setObjectName("managerAppendFigureTargetDialog")
        self.setWindowTitle("Figure" if allow_new_figure else "Append to Figure")
        self.setModal(True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetFixedSize)

        form = QtWidgets.QFormLayout()
        form.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        layout.addLayout(form)

        self.figure_combo = QtWidgets.QComboBox(self)
        self.figure_combo.setObjectName("managerAppendFigureCombo")
        if allow_new_figure:
            self.figure_combo.addItem("New Figure", _NEW_FIGURE_TARGET)
        for uid in figure_uids:
            self.figure_combo.addItem(manager._child_node(uid).display_text, uid)
        self.figure_combo.setVisible(self.figure_combo.count() > 1)
        if self.figure_combo.count() > 1:
            form.addRow("Figure", self.figure_combo)
        else:
            form.addRow(
                "Figure",
                QtWidgets.QLabel(
                    manager._child_node(figure_uids[0]).display_text, self
                ),
            )

        self.selector_stack = QtWidgets.QStackedWidget(self)
        self.selector_stack.setObjectName("managerAppendAxesSelectorStack")
        self.axes_selector = _AxesSelectorWidget(self)
        self.axes_selector.setObjectName("managerAppendAxesSelector")
        self.gridspec_axes_selector = _GridSpecViewWidget(self, mode="select")
        self.gridspec_axes_selector.setObjectName("managerAppendGridSpecAxesSelector")
        self.selector_stack.addWidget(self.axes_selector)
        self.selector_stack.addWidget(self.gridspec_axes_selector)
        form.addRow("Axes", self.selector_stack)

        self.status_label = QtWidgets.QLabel(self)
        self.status_label.setObjectName("managerAppendAxesStatusLabel")
        layout.addWidget(self.status_label)

        action_layout = QtWidgets.QHBoxLayout()
        self.all_axes_button = QtWidgets.QToolButton(self)
        self.all_axes_button.setObjectName("managerAppendAllAxesButton")
        self.all_axes_button.setText("All axes")
        self.all_axes_button.setToolTip("Select every available axes in this figure.")
        action_layout.addWidget(self.all_axes_button)
        self.clear_axes_button = QtWidgets.QToolButton(self)
        self.clear_axes_button.setObjectName("managerAppendClearAxesButton")
        self.clear_axes_button.setText("Clear")
        self.clear_axes_button.setToolTip("Clear the current axes selection.")
        action_layout.addWidget(self.clear_axes_button)
        action_layout.addStretch(1)
        layout.addLayout(action_layout)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        self.button_box.setObjectName("managerAppendFigureButtonBox")
        layout.addWidget(self.button_box)

        self.figure_combo.currentIndexChanged.connect(self._figure_changed)
        self.axes_selector.sigSelectionChanged.connect(self._selection_changed)
        self.axes_selector.sigAddRowRequested.connect(self._add_subplot_row)
        self.axes_selector.sigAddColumnRequested.connect(self._add_subplot_column)
        self.gridspec_axes_selector.sigSelectionChanged.connect(self._selection_changed)
        self.all_axes_button.clicked.connect(self._select_all_axes)
        self.clear_axes_button.clicked.connect(self._clear_axes)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self._figure_changed()

    def figure_uid(self) -> str:
        uid = self.figure_combo.currentData()
        if isinstance(uid, str):
            return uid
        return self._figure_uids[0]

    def is_new_figure(self) -> bool:
        return self.figure_combo.currentData() == _NEW_FIGURE_TARGET

    def axes_selection(self) -> typing.Any | None:
        from erlab.interactive._figurecomposer import FigureAxesSelectionState

        if self.is_new_figure():
            return None
        tool = self._figure_tool()
        if tool is None:
            return None
        setup = tool.tool_status.setup
        if setup.layout_mode == "gridspec":
            axes_ids = self.gridspec_axes_selector.selected_axes_ids()
            if not axes_ids:
                return None
            return FigureAxesSelectionState(axes_ids=axes_ids)
        axes = self.axes_selector.selected_axes()
        if not axes:
            return None
        return FigureAxesSelectionState(axes=axes)

    def selected_target(self) -> tuple[str, typing.Any] | None:
        if self.is_new_figure():
            return None
        selection = self.axes_selection()
        if selection is None:
            return None
        return self.figure_uid(), selection

    @QtCore.Slot()
    def _figure_changed(self) -> None:
        if self.is_new_figure():
            self.selector_stack.setVisible(False)
            self.all_axes_button.setVisible(False)
            self.clear_axes_button.setVisible(False)
            self._ok_button().setEnabled(True)
            self.status_label.setText("A new figure will be created.")
            return
        self.selector_stack.setVisible(True)
        self.all_axes_button.setVisible(True)
        self.clear_axes_button.setVisible(True)
        tool = self._figure_tool()
        if tool is None:
            self._ok_button().setEnabled(False)
            self.status_label.setText("The selected figure is unavailable.")
            return
        setup = tool.tool_status.setup
        if setup.layout_mode == "gridspec":
            from erlab.interactive._figurecomposer._gridspec import (
                _gridspec_all_axes_ids,
                _gridspec_axis_display_names,
                _gridspec_valid_axes_ids,
            )

            axes_ids = _gridspec_valid_axes_ids(setup, _gridspec_all_axes_ids(setup))
            labels = dict(
                zip(
                    axes_ids,
                    _gridspec_axis_display_names(setup, axes_ids),
                    strict=True,
                )
            )
            self.gridspec_axes_selector.set_layout(setup.gridspec.root, labels)
            self.selector_stack.setCurrentWidget(self.gridspec_axes_selector)
            self.gridspec_axes_selector.set_selected_axes_ids(
                self._default_gridspec_axes(axes_ids)
            )
        else:
            labels = {
                (row, col): f"{row}, {col}"
                for row in range(setup.nrows)
                for col in range(setup.ncols)
            }
            self.axes_selector.set_grid(setup.nrows, setup.ncols, labels)
            self.selector_stack.setCurrentWidget(self.axes_selector)
            self.axes_selector.set_selected_axes(
                self._default_subplot_axes(
                    self._manager._figure_all_axes(setup.nrows, setup.ncols)
                )
            )
        self._selection_changed()

    @QtCore.Slot()
    def _selection_changed(self, *_args: typing.Any) -> None:
        selection = self.axes_selection()
        self._ok_button().setEnabled(selection is not None)
        if selection is None:
            self.status_label.setText("Select at least one target axes.")
            return
        count = len(selection.axes_ids) if selection.axes_ids else len(selection.axes)
        suffix = "axis" if count == 1 else "axes"
        self.status_label.setText(f"{count} target {suffix} selected.")

    @QtCore.Slot()
    def _select_all_axes(self) -> None:
        tool = self._figure_tool()
        if tool is None:
            return
        setup = tool.tool_status.setup
        if setup.layout_mode == "gridspec":
            from erlab.interactive._figurecomposer._gridspec import (
                _gridspec_all_axes_ids,
                _gridspec_valid_axes_ids,
            )

            self.gridspec_axes_selector.set_selected_axes_ids(
                _gridspec_valid_axes_ids(setup, _gridspec_all_axes_ids(setup)),
                emit=True,
            )
        else:
            self.axes_selector.set_selected_axes(
                self._manager._figure_all_axes(setup.nrows, setup.ncols),
                emit=True,
            )

    @QtCore.Slot()
    def _clear_axes(self) -> None:
        if self.selector_stack.currentWidget() is self.gridspec_axes_selector:
            self.gridspec_axes_selector.set_selected_axes_ids((), emit=True)
        else:
            self.axes_selector.set_selected_axes((), emit=True)

    @QtCore.Slot()
    def _add_subplot_row(self) -> None:
        self._grow_subplot_grid("row")

    @QtCore.Slot()
    def _add_subplot_column(self) -> None:
        self._grow_subplot_grid("column")

    def _grow_subplot_grid(self, direction: typing.Literal["row", "column"]) -> None:
        tool = self._figure_tool()
        if tool is None or tool.tool_status.setup.layout_mode != "subplots":
            return
        selected = self.axes_selector.selected_axes()
        if not tool._grow_subplot_grid(direction):
            return
        setup = tool.tool_status.setup
        labels = {
            (row, col): f"{row}, {col}"
            for row in range(setup.nrows)
            for col in range(setup.ncols)
        }
        self.axes_selector.set_grid(setup.nrows, setup.ncols, labels)
        self.axes_selector.set_selected_axes(selected or ((0, 0),), emit=True)

    def _figure_tool(self) -> typing.Any | None:
        from erlab.interactive._figurecomposer import FigureComposerTool

        if not self._manager._is_figure_uid(self.figure_uid()):
            return None
        tool = self._manager._child_node(self.figure_uid()).tool_window
        return tool if isinstance(tool, FigureComposerTool) else None

    def _ok_button(self) -> QtWidgets.QPushButton:
        button = self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        if button is None:
            raise RuntimeError("Append dialog OK button is unavailable")
        return button

    def _default_subplot_axes(
        self, axes: tuple[tuple[int, int], ...]
    ) -> tuple[tuple[int, int], ...]:
        if self._operation_defaults_to_all_axes() or len(axes) <= 1:
            return axes
        return axes[:1]

    def _default_gridspec_axes(self, axes_ids: tuple[str, ...]) -> tuple[str, ...]:
        if self._operation_defaults_to_all_axes() or len(axes_ids) <= 1:
            return axes_ids
        return axes_ids[:1]

    def _operation_defaults_to_all_axes(self) -> bool:
        if self._operation is None:
            return True
        from erlab.interactive._figurecomposer import FigureOperationKind

        return self._operation.kind == FigureOperationKind.PLOT_SLICES


class ImageToolManager(_ImageToolManagerBase):
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
        self._lineage_controller = _LineageController(self)
        self._provenance_edit_controller = _ProvenanceEditController(self)
        self._details_panel = _DetailsPanelController(self)
        self._actions_controller = _ActionsController(self)
        self._widgets_controller = _WidgetsController(self)

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

        self._registry_heartbeat = _RegistryHeartbeatController(
            self._manager_record.internal_id,
            parent=self,
        )
        self._registry_heartbeat_timer = QtCore.QTimer(self)
        self._registry_heartbeat_timer.setInterval(3000)
        self._registry_heartbeat_timer.timeout.connect(self._registry_heartbeat_tick)

        # Shared memory for detecting multiple instances
        # No longer used starting from v3.8.2, but kept for backward compatibility
        self._shm = QtCore.QSharedMemory(_SHM_NAME)
        self._shm.create(1)  # Create segment so that it can be attached to

        self.menu_bar: QtWidgets.QMenuBar = typing.cast(
            "QtWidgets.QMenuBar", self.menuBar()
        )

        self._workspace_state = _ManagerWorkspaceState()
        self._workspace_controller = _WorkspaceIOController(self)
        self._tool_metadata_queue = _ManagerToolMetadataQueue(
            self, self._flush_pending_tool_metadata_updates
        )
        self._update_workspace_window_title()
        self._registry_heartbeat_timer.start()

        qapp = QtWidgets.QApplication.instance()
        self._application_quit_filter: _ApplicationQuitFilter | None = None
        if isinstance(qapp, QtWidgets.QApplication):
            self._application_quit_filter = _ApplicationQuitFilter(self)
            qapp.installEventFilter(self._application_quit_filter)

        self._link_registry = _ManagerLinkRegistry()

        # Stores additional analysis tools opened from child ImageTool windows
        self._additional_windows: dict[str, QtWidgets.QWidget] = {}
        self._standalone_app_windows: dict[str, QtWidgets.QWidget] = {}
        self._standalone_app_event_filters: dict[str, QtCore.QObject] = {}
        self._standalone_app_pending_states: dict[str, dict[str, typing.Any]] = {}
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
        self._figure_view_mode = self._read_figure_view_mode_setting()
        self._figure_gallery_thumbnail_size_name = (
            self._read_figure_gallery_size_setting()
        )
        self._updating_figure_view_controls = False

        # Store progress bar widgets
        self._progress_bars: dict[int, QtWidgets.QProgressDialog] = {}

        self._bulk_remove_depth: int = 0
        self._manager_layout_tracking_enabled = False

        # Initialize actions
        self.settings_action = QtWidgets.QAction("Settings", self)
        self.settings_action.triggered.connect(self.open_settings)
        self.settings_action.setShortcut(QtGui.QKeySequence.StandardKey.Preferences)
        self.settings_action.setToolTip("Open settings")
        self.settings_action.setIcon(QtGui.QIcon.fromTheme("preferences-system"))

        self.show_action = QtWidgets.QAction("Show", self)
        self.show_action.triggered.connect(self.show_selected)
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

        self.batch_action = QtWidgets.QAction("Batch Operation…", self)
        self.batch_action.setObjectName("manager_batch_operation_action")
        self.batch_action.triggered.connect(self.show_batch_operations)
        self.batch_action.setToolTip("Apply an operation to multiple ImageTools")

        self.create_figure_action = QtWidgets.QAction("Figure", self)
        self.create_figure_action.setObjectName("manager_figure_action")
        self.create_figure_action.triggered.connect(self.create_figure_from_selection)
        self.create_figure_action.setToolTip(
            "Create or append an editable Matplotlib figure from the selected data"
        )
        self.create_figure_action.setIcon(QtGui.QIcon.fromTheme("insert-image"))

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
        self.edit_menu.addAction(self.batch_action)
        self.edit_menu.addAction(self.create_figure_action)
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
        self.batch_button = erlab.interactive.utils.IconActionButton(
            self.batch_action, "mdi6.table-edit"
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
        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.main_splitter.splitterMoved.connect(
            lambda _pos, _index: self._mark_workspace_layout_dirty()
        )
        self.setCentralWidget(self.main_splitter)

        # Construct left side of splitter
        left_container = QtWidgets.QWidget()
        left_layout = QtWidgets.QHBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        self.main_splitter.addWidget(left_container)

        titlebar = QtWidgets.QWidget()
        titlebar_layout = QtWidgets.QVBoxLayout()
        titlebar.setLayout(titlebar_layout)
        titlebar_layout.addWidget(self.open_button)
        titlebar_layout.addWidget(self.remove_button)
        titlebar_layout.addWidget(self.rename_button)
        titlebar_layout.addWidget(self.batch_button)
        titlebar_layout.addWidget(self.link_button)
        titlebar_layout.addWidget(self.unlink_button)
        titlebar_layout.addStretch()
        left_layout.addWidget(titlebar)

        self.tree_view = _ImageToolWrapperTreeView(self)
        self.tree_view.setObjectName("manager_data_tree_view")
        self._install_selection_shortcuts(self.tree_view)
        self.tree_view._selection_model.selectionChanged.connect(self._update_actions)
        self.tree_view._selection_model.selectionChanged.connect(self._update_info)
        self.tree_view._selection_model.selectionChanged.connect(
            self._clear_figure_selection_from_tree
        )
        self.tree_view._model.dataChanged.connect(self._update_info)

        self.left_tabs = QtWidgets.QTabWidget(left_container)
        self.left_tabs.setObjectName("manager_left_tabs")
        self.left_tabs.setDocumentMode(True)
        left_tab_bar = self.left_tabs.tabBar()
        if left_tab_bar is not None:  # pragma: no branch
            left_tab_bar.hide()
        self.left_tabs.addTab(self.tree_view, "Data/Tools")

        left_layout.addWidget(self.left_tabs)

        # Construct right side of splitter
        right_panel = QtWidgets.QWidget(self)
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        self.main_splitter.addWidget(right_panel)

        self.right_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.right_splitter.splitterMoved.connect(
            lambda _pos, _index: self._mark_workspace_layout_dirty()
        )
        right_layout.addWidget(self.right_splitter, 1)

        self.text_box = QtWidgets.QTextEdit(self)
        self.text_box.setReadOnly(True)
        self.right_splitter.addWidget(self.text_box)

        self.preview_widget = _SingleImagePreview(self)
        self.right_splitter.addWidget(self.preview_widget)

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
        self.right_splitter.setSizes([280, 140])
        self.main_splitter.setSizes([100, 150])

        # Store most recent name filter and directory for new windows
        self._recent_name_filter: str | None = None
        self._recent_directory: str | None = None
        self._recent_loader_kwargs_by_filter: dict[str, dict[str, typing.Any]] = {}
        self._recent_loader_extensions_by_filter: dict[str, dict[str, typing.Any]] = {}
        self._metadata_full_code_available = False
        self._metadata_node_uid: str | None = None
        self._refreshing_figure_list = False
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
        self._metadata_edit_step_action = QtGui.QAction("Edit Step…", self)
        self._metadata_edit_step_action.setObjectName(
            "manager_edit_provenance_step_action"
        )
        self._metadata_edit_step_action.triggered.connect(
            self._edit_selected_derivation_step
        )
        self._metadata_revert_step_action = QtGui.QAction("Revert to This Step…", self)
        self._metadata_revert_step_action.setObjectName(
            "manager_revert_provenance_step_action"
        )
        self._metadata_revert_step_action.triggered.connect(
            self._revert_selected_derivation_step
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
        self._manager_layout_tracking_enabled = True

    def event(self, event: QtCore.QEvent | None) -> bool:
        handled = super().event(event)
        if event is not None and event.type() in (
            QtCore.QEvent.Type.Move,
            QtCore.QEvent.Type.Resize,
            QtCore.QEvent.Type.WindowStateChange,
        ):
            self._mark_workspace_layout_dirty()
        return handled

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
            self._registry_heartbeat.stop()
            self._stop_servers()
            unregister_manager_record(self._manager_record.internal_id)

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

    def _register_figure_node(self, node: _ManagedWindowNode) -> None:
        self._tool_graph.register_figure(node)
        if node.tool_window is not None:
            node.tool_window._refresh_reload_data_action()

    def _unregister_node(self, uid: str) -> None:
        node = self._tool_graph.unregister_node(uid)
        if node is None:
            return
        self._dependency_tracker.clear_uid(uid)
        self._refresh_dependency_dependents(uid)

    def _iter_descendant_uids(self, uid: str) -> list[str]:
        return self._tool_graph.descendant_uids(uid)

    def _mark_removed_subtree_dirty(self, uid: str) -> None:
        for node_uid in self._tool_graph.subtree_uids(uid):
            node = self._tool_graph.nodes.get(node_uid)
            if node is not None:
                self._set_node_window_modified(node_uid, False)
                self._mark_workspace_dirty(
                    removed=node.display_text, structure="Removed window"
                )

    def _remove_uid_target(self, uid: str) -> None:
        if uid not in self._tool_graph.nodes:
            return
        subtree = self._tool_graph.subtree_uids(uid)
        subtree.reverse()
        for child_uid in subtree:
            child = self._tool_graph.nodes.get(child_uid)
            if child is None or isinstance(child, _ImageToolWrapper):
                continue
            self._unregister_node(child_uid)
            if child.tool_window is not None:
                child.tool_window.set_source_parent_fetcher(None)
                child.tool_window.set_input_provenance_parent_fetcher(None)
            child.dispose()

    def _figure_uids(self) -> list[str]:
        return [
            uid
            for uid in self._tool_graph.figure_uids
            if uid in self._tool_graph.nodes and self._is_figure_uid(uid)
        ]

    @staticmethod
    def _settings_string(key: str, default: str) -> str:
        value = _manager_settings().value(key, default)
        return value if isinstance(value, str) else default

    def _read_figure_view_mode_setting(self) -> str:
        mode = self._settings_string(
            _FIGURE_VIEW_MODE_SETTINGS_KEY, _FIGURE_VIEW_MODE_LIST
        )
        return mode if mode in _FIGURE_VIEW_MODES else _FIGURE_VIEW_MODE_LIST

    def _read_figure_gallery_size_setting(self) -> str:
        size_name = self._settings_string(
            _FIGURE_GALLERY_SIZE_SETTINGS_KEY, _FIGURE_GALLERY_SIZE_MEDIUM
        )
        if size_name in _FIGURE_GALLERY_THUMBNAIL_SIZES:
            return size_name
        return _FIGURE_GALLERY_SIZE_MEDIUM

    def _style_pixel_metric(self, metric: QtWidgets.QStyle.PixelMetric) -> int:
        return self._qt_style().pixelMetric(metric)

    def _qt_style(self) -> QtWidgets.QStyle:
        style = self.style() or QtWidgets.QApplication.style()
        if style is None:  # pragma: no cover
            raise RuntimeError("No active Qt style")
        return style

    def _create_figures_ui(self) -> None:
        if hasattr(self, "figure_tab"):
            return

        self.figure_tab = QtWidgets.QWidget(self.left_tabs)
        self.figure_tab.setObjectName("manager_figures_tab")
        figure_layout = QtWidgets.QVBoxLayout(self.figure_tab)
        figure_layout.setContentsMargins(0, 0, 0, 0)
        figure_layout.setSpacing(0)
        self.figure_view_controls = QtWidgets.QWidget(self.figure_tab)
        self.figure_view_controls.setObjectName("manager_figures_view_controls")
        figure_view_layout = QtWidgets.QHBoxLayout(self.figure_view_controls)
        figure_view_layout.setContentsMargins(0, 0, 0, 0)
        figure_view_layout.setSpacing(
            self._style_pixel_metric(
                QtWidgets.QStyle.PixelMetric.PM_LayoutHorizontalSpacing
            )
        )
        self.figure_view_button_group = QtWidgets.QButtonGroup(
            self.figure_view_controls
        )
        self.figure_view_button_group.setExclusive(True)
        self.figure_view_list_button = self._figure_view_mode_button(
            "List",
            erlab.interactive.utils.qtawesome.icon("ph.list-bullets"),
            _FIGURE_VIEW_MODE_LIST,
        )
        self.figure_view_gallery_button = self._figure_view_mode_button(
            "Gallery",
            erlab.interactive.utils.qtawesome.icon("ph.grid-four"),
            _FIGURE_VIEW_MODE_GALLERY,
        )
        for button in (self.figure_view_list_button, self.figure_view_gallery_button):
            self.figure_view_button_group.addButton(button)
            figure_view_layout.addWidget(button)
        self.figure_gallery_size_label = QtWidgets.QLabel(
            "Thumbnail", self.figure_view_controls
        )
        self.figure_gallery_size_combo = QtWidgets.QComboBox(self.figure_view_controls)
        self.figure_gallery_size_combo.setObjectName("manager_figures_gallery_size")
        self.figure_gallery_size_combo.setToolTip("Choose gallery thumbnail size.")
        self.figure_gallery_size_label.setBuddy(self.figure_gallery_size_combo)
        for label, key in (
            ("Small", "small"),
            ("Medium", "medium"),
            ("Large", "large"),
        ):
            self.figure_gallery_size_combo.addItem(label, key)
        self.figure_gallery_size_combo.currentIndexChanged.connect(
            self._figure_gallery_size_changed
        )
        figure_view_layout.addStretch(1)
        figure_view_layout.addWidget(self.figure_gallery_size_label)
        figure_view_layout.addWidget(self.figure_gallery_size_combo)
        figure_layout.addWidget(self.figure_view_controls)
        self.figure_list = QtWidgets.QListWidget(self.figure_tab)
        self.figure_list.setObjectName("manager_figures_list")
        self.figure_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.figure_list.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.SelectedClicked
        )
        self.figure_list.setContextMenuPolicy(
            QtCore.Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.figure_list.itemSelectionChanged.connect(self._figure_selection_changed)
        self.figure_list.itemChanged.connect(self._figure_item_changed)
        self.figure_list.itemDoubleClicked.connect(self._show_figure_item)
        self.figure_list.customContextMenuRequested.connect(self._show_figure_menu)
        self._install_selection_shortcuts(self.figure_list)
        figure_layout.addWidget(self.figure_list)
        self._apply_figure_view_controls()
        self._apply_figure_list_view_configuration()

    def _install_selection_shortcuts(self, widget: QtWidgets.QWidget) -> None:
        def add_shortcut(
            sequence: str,
            callback: Callable[[], None],
        ) -> None:
            shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(sequence), widget)
            shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetShortcut)
            shortcut.activated.connect(callback)

        if sys.platform == "darwin":
            add_shortcut("Return", self.rename_selected)
            add_shortcut("Enter", self.rename_selected)
            add_shortcut("Ctrl+Down", self.show_selected)
        else:
            add_shortcut("F2", self.rename_selected)
            add_shortcut("Return", self.show_selected)
            add_shortcut("Enter", self.show_selected)

    def _destroy_figures_ui(self) -> None:
        figure_tab = getattr(self, "figure_tab", None)
        if figure_tab is None:
            return

        self._refreshing_figure_list = True
        figure_list = getattr(self, "figure_list", None)
        if figure_list is not None and erlab.interactive.utils.qt_is_valid(figure_list):
            figure_list.blockSignals(True)
            with contextlib.suppress(TypeError, RuntimeError):
                figure_list.itemSelectionChanged.disconnect(
                    self._figure_selection_changed
                )
            with contextlib.suppress(TypeError, RuntimeError):
                figure_list.itemChanged.disconnect(self._figure_item_changed)
            with contextlib.suppress(TypeError, RuntimeError):
                figure_list.itemDoubleClicked.disconnect(self._show_figure_item)
            with contextlib.suppress(TypeError, RuntimeError):
                figure_list.customContextMenuRequested.disconnect(
                    self._show_figure_menu
                )
        gallery_size_combo = getattr(self, "figure_gallery_size_combo", None)
        if gallery_size_combo is not None and erlab.interactive.utils.qt_is_valid(
            gallery_size_combo
        ):
            with contextlib.suppress(TypeError, RuntimeError):
                gallery_size_combo.currentIndexChanged.disconnect(
                    self._figure_gallery_size_changed
                )
        for button_name in (
            "figure_view_list_button",
            "figure_view_gallery_button",
        ):
            button = getattr(self, button_name, None)
            if button is not None and erlab.interactive.utils.qt_is_valid(button):
                with contextlib.suppress(TypeError, RuntimeError):
                    button.clicked.disconnect()

        tab_index = self.left_tabs.indexOf(figure_tab)
        if tab_index >= 0:
            if self.left_tabs.currentIndex() == tab_index:
                self.left_tabs.setCurrentIndex(0)
            self.left_tabs.removeTab(tab_index)
        figure_tab.hide()
        figure_tab.deleteLater()
        for attr in (
            "figure_tab",
            "figure_view_controls",
            "figure_view_button_group",
            "figure_view_list_button",
            "figure_view_gallery_button",
            "figure_gallery_size_label",
            "figure_gallery_size_combo",
            "figure_list",
        ):
            if hasattr(self, attr):
                delattr(self, attr)
        self._refreshing_figure_list = False

    def _figure_view_mode_button(
        self, text: str, icon: QtGui.QIcon, mode: str
    ) -> QtWidgets.QToolButton:
        button = QtWidgets.QToolButton(self.figure_view_controls)
        button.setObjectName(f"manager_figures_{mode}_view_button")
        button.setIcon(icon)
        button.setCheckable(True)
        button.setAutoRaise(True)
        button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)
        button.setAccessibleName(f"{text} view")
        button.setToolTip(f"Show figures in {text.lower()} view.")
        button.clicked.connect(
            lambda _checked=False, mode=mode: self._set_figure_view_mode(mode)
        )
        return button

    def _set_figure_view_mode(self, mode: str) -> None:
        if mode not in _FIGURE_VIEW_MODES or mode == self._figure_view_mode:
            return
        self._figure_view_mode = mode
        _manager_settings().setValue(_FIGURE_VIEW_MODE_SETTINGS_KEY, mode)
        self._apply_figure_view_controls()
        self._sync_figures_ui()

    @QtCore.Slot(int)
    def _figure_gallery_size_changed(self, _index: int) -> None:
        if self._updating_figure_view_controls:
            return
        size_name = self.figure_gallery_size_combo.currentData()
        if (
            not isinstance(size_name, str)
            or size_name not in _FIGURE_GALLERY_THUMBNAIL_SIZES
            or size_name == self._figure_gallery_thumbnail_size_name
        ):
            return
        self._figure_gallery_thumbnail_size_name = size_name
        _manager_settings().setValue(_FIGURE_GALLERY_SIZE_SETTINGS_KEY, size_name)
        self._apply_figure_view_controls()
        self._sync_figures_ui()

    def _apply_figure_view_controls(self) -> None:
        if not hasattr(self, "figure_view_list_button"):
            return
        self._updating_figure_view_controls = True
        try:
            self.figure_view_list_button.setChecked(
                self._figure_view_mode == _FIGURE_VIEW_MODE_LIST
            )
            self.figure_view_gallery_button.setChecked(
                self._figure_view_mode == _FIGURE_VIEW_MODE_GALLERY
            )
            gallery_mode = self._figure_view_mode == _FIGURE_VIEW_MODE_GALLERY
            self.figure_gallery_size_label.setVisible(gallery_mode)
            self.figure_gallery_size_combo.setVisible(gallery_mode)
            self.figure_gallery_size_combo.setEnabled(gallery_mode)
            for index in range(self.figure_gallery_size_combo.count()):
                if (
                    self.figure_gallery_size_combo.itemData(index)
                    == self._figure_gallery_thumbnail_size_name
                ):
                    self.figure_gallery_size_combo.setCurrentIndex(index)
                    break
        finally:
            self._updating_figure_view_controls = False

    def _figure_gallery_thumbnail_size(self) -> QtCore.QSize:
        width, height = _FIGURE_GALLERY_THUMBNAIL_SIZES[
            self._figure_gallery_thumbnail_size_name
        ]
        return QtCore.QSize(width, height)

    def _figure_gallery_grid_size(self) -> QtCore.QSize:
        thumbnail_size = self._figure_gallery_thumbnail_size()
        spacing = self._style_pixel_metric(
            QtWidgets.QStyle.PixelMetric.PM_LayoutHorizontalSpacing
        )
        label_height = self.figure_list.fontMetrics().height() * 2
        return QtCore.QSize(
            thumbnail_size.width() + spacing * 4,
            thumbnail_size.height() + label_height + spacing * 3,
        )

    def _apply_figure_list_view_configuration(self) -> None:
        if self._figure_view_mode == _FIGURE_VIEW_MODE_GALLERY:
            self.figure_list.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
            self.figure_list.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
            self.figure_list.setMovement(QtWidgets.QListView.Movement.Static)
            self.figure_list.setFlow(QtWidgets.QListView.Flow.LeftToRight)
            self.figure_list.setWrapping(True)
            self.figure_list.setSpacing(
                self._style_pixel_metric(
                    QtWidgets.QStyle.PixelMetric.PM_LayoutHorizontalSpacing
                )
            )
            self.figure_list.setUniformItemSizes(True)
            self.figure_list.setIconSize(self._figure_gallery_thumbnail_size())
            self.figure_list.setGridSize(self._figure_gallery_grid_size())
            self.figure_list.setTextElideMode(QtCore.Qt.TextElideMode.ElideMiddle)
            return
        self.figure_list.setViewMode(QtWidgets.QListView.ViewMode.ListMode)
        self.figure_list.setResizeMode(QtWidgets.QListView.ResizeMode.Fixed)
        self.figure_list.setMovement(QtWidgets.QListView.Movement.Static)
        self.figure_list.setFlow(QtWidgets.QListView.Flow.TopToBottom)
        self.figure_list.setWrapping(False)
        self.figure_list.setSpacing(0)
        self.figure_list.setUniformItemSizes(False)
        self.figure_list.setIconSize(QtCore.QSize())
        self.figure_list.setGridSize(QtCore.QSize())

    def _set_figures_tab_available(self, available: bool) -> None:
        if not available:
            self._destroy_figures_ui()
            left_tab_bar = self.left_tabs.tabBar()
            if left_tab_bar is not None:  # pragma: no branch
                left_tab_bar.setVisible(False)
            self.left_tabs.updateGeometry()
            return

        self._create_figures_ui()
        tab_index = self.left_tabs.indexOf(self.figure_tab)
        if tab_index < 0:
            tab_index = self.left_tabs.addTab(self.figure_tab, "Figures")
        self.figure_tab.show()
        self.left_tabs.setTabVisible(tab_index, True)
        left_tab_bar = self.left_tabs.tabBar()
        if left_tab_bar is not None:  # pragma: no branch
            left_tab_bar.setVisible(True)
        self.left_tabs.updateGeometry()

    def _figure_gallery_icon(self, uid: str) -> QtGui.QIcon:
        if not erlab.interactive.utils.qt_is_valid(self):
            return QtGui.QIcon()
        if not self._is_figure_uid(uid):
            return QtGui.QIcon(self._figure_gallery_placeholder_pixmap())
        node = self._child_node(uid)
        tool_window = node.tool_window
        if tool_window is None or not erlab.interactive.utils.qt_is_valid(tool_window):
            return QtGui.QIcon(self._figure_gallery_placeholder_pixmap())
        if getattr(tool_window, "preview_pixmap_stale", False):
            request_preview = getattr(
                tool_window, "request_preview_pixmap_update", None
            )
            if callable(request_preview):
                request_preview()
        thumbnail_pixmap = self._figure_gallery_tool_thumbnail_pixmap(tool_window)
        if thumbnail_pixmap is None or thumbnail_pixmap.isNull():
            return QtGui.QIcon(self._figure_gallery_placeholder_pixmap())
        return QtGui.QIcon(thumbnail_pixmap)

    def _figure_gallery_placeholder_pixmap(self) -> QtGui.QPixmap:
        thumbnail_size = self._figure_gallery_thumbnail_size()
        pixmap = QtGui.QPixmap(thumbnail_size)
        pixmap.fill(self.palette().color(QtGui.QPalette.ColorRole.Base))
        painter = QtGui.QPainter(pixmap)
        try:
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            rect = QtCore.QRectF(pixmap.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
            painter.setPen(self.palette().color(QtGui.QPalette.ColorRole.Mid))
            painter.drawRoundedRect(rect, 3.0, 3.0)
        finally:
            painter.end()
        return pixmap

    def _figure_gallery_tool_thumbnail_pixmap(
        self, tool_window: object
    ) -> QtGui.QPixmap | None:
        if isinstance(
            tool_window, QtCore.QObject
        ) and not erlab.interactive.utils.qt_is_valid(tool_window):
            return None
        thumbnail_size = self._figure_gallery_thumbnail_size()
        thumbnail_provider = getattr(tool_window, "preview_thumbnail_pixmap", None)
        if callable(thumbnail_provider):
            thumbnail = thumbnail_provider(thumbnail_size)
            if thumbnail is not None and not thumbnail.isNull():
                return self._figure_gallery_thumbnail_pixmap(thumbnail)
        preview_pixmap = getattr(tool_window, "preview_pixmap", None)
        if preview_pixmap is None or preview_pixmap.isNull():
            return None
        return self._figure_gallery_thumbnail_pixmap(preview_pixmap)

    def _figure_gallery_thumbnail_pixmap(
        self, source_pixmap: QtGui.QPixmap
    ) -> QtGui.QPixmap:
        thumbnail_size = self._figure_gallery_thumbnail_size()
        canvas = QtGui.QPixmap(thumbnail_size)
        canvas.fill(self.palette().color(QtGui.QPalette.ColorRole.Base))
        scaled = source_pixmap
        if source_pixmap.size() != thumbnail_size:
            scaled = source_pixmap.scaled(
                thumbnail_size,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
        painter = QtGui.QPainter(canvas)
        try:
            x_pos = (thumbnail_size.width() - scaled.width()) // 2
            y_pos = (thumbnail_size.height() - scaled.height()) // 2
            painter.drawPixmap(x_pos, y_pos, scaled)
        finally:
            painter.end()
        return canvas

    def _figure_list_item(self, uid: str) -> QtWidgets.QListWidgetItem:
        node = self._child_node(uid)
        item = QtWidgets.QListWidgetItem(node.display_text)
        item.setData(QtCore.Qt.ItemDataRole.UserRole, uid)
        item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
        if self._figure_view_mode == _FIGURE_VIEW_MODE_GALLERY:
            item.setIcon(self._figure_gallery_icon(uid))
            item.setTextAlignment(
                QtCore.Qt.AlignmentFlag.AlignHCenter
                | QtCore.Qt.AlignmentFlag.AlignBottom
            )
        return item

    def _next_figure_display_name(self) -> str:
        highest = 0
        for uid in self._figure_uids():
            match = re.fullmatch(r"Figure (\d+)", self._child_node(uid).display_text)
            if match is not None:
                highest = max(highest, int(match.group(1)))
        return f"Figure {highest + 1}"

    def _sync_figures_ui(self, *, select_uid: str | None = None) -> None:
        figure_uids = self._figure_uids()
        selected_uids = (
            {select_uid}
            if select_uid is not None
            else set(self._selected_figure_uids())
        )

        has_figures = bool(figure_uids)
        self._set_figures_tab_available(has_figures)
        if not has_figures:
            return
        self.figure_view_controls.setVisible(has_figures)
        self._apply_figure_view_controls()
        self._apply_figure_list_view_configuration()

        self._refreshing_figure_list = True
        self.figure_list.blockSignals(True)
        try:
            self.figure_list.clear()
            for uid in figure_uids:
                item = self._figure_list_item(uid)
                self.figure_list.addItem(item)
                if uid in selected_uids:
                    item.setSelected(True)
                    self.figure_list.setCurrentItem(item)
        finally:
            self.figure_list.blockSignals(False)
            self._refreshing_figure_list = False

        if select_uid is not None:
            self.left_tabs.setCurrentWidget(self.figure_tab)

    def _figure_list_item_for_uid(self, uid: str) -> QtWidgets.QListWidgetItem | None:
        if not hasattr(self, "figure_list"):
            return None
        for row in range(self.figure_list.count()):
            item = self.figure_list.item(row)
            if item is not None and self._figure_uid_from_item(item) == uid:
                return item
        return None

    def _update_figure_gallery_icon(self, uid: str) -> None:
        if (
            not erlab.interactive.utils.qt_is_valid(self)
            or not hasattr(self, "figure_list")
            or not erlab.interactive.utils.qt_is_valid(self.figure_list)
            or self._refreshing_figure_list
            or self._figure_view_mode != _FIGURE_VIEW_MODE_GALLERY
            or not self._is_figure_uid(uid)
        ):
            return
        item = self._figure_list_item_for_uid(uid)
        if item is not None:
            self.figure_list.blockSignals(True)
            try:
                item.setIcon(self._figure_gallery_icon(uid))
            finally:
                self.figure_list.blockSignals(False)

    def _figure_uid_from_item(
        self, item: QtWidgets.QListWidgetItem | None
    ) -> str | None:
        if item is None:
            return None
        uid = item.data(QtCore.Qt.ItemDataRole.UserRole)
        return uid if isinstance(uid, str) else None

    def _select_figure_uid(self, uid: str) -> None:
        self._sync_figures_ui(select_uid=uid)
        self.tree_view.deselect_all()
        self._update_actions()
        self._update_info()

    @QtCore.Slot()
    def _clear_figure_selection_from_tree(self) -> None:
        if self._refreshing_figure_list or not hasattr(self, "figure_list"):
            return
        if not self.tree_view.selectedIndexes():
            return
        self.figure_list.blockSignals(True)
        try:
            self.figure_list.clearSelection()
        finally:
            self.figure_list.blockSignals(False)

    @QtCore.Slot()
    def _figure_selection_changed(self) -> None:
        if self._refreshing_figure_list or not hasattr(self, "figure_list"):
            return
        if self.figure_list.selectedItems():
            selection_model = self.tree_view.selectionModel()
            if selection_model is None:  # pragma: no cover
                return
            selection_model.blockSignals(True)
            try:
                self.tree_view.clearSelection()
            finally:
                selection_model.blockSignals(False)
        self._update_actions()
        self._update_info()

    @QtCore.Slot(QtWidgets.QListWidgetItem)
    def _figure_item_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        if self._refreshing_figure_list or not hasattr(self, "figure_list"):
            return
        uid = self._figure_uid_from_item(item)
        if uid is None or not self._is_figure_uid(uid):
            return
        self._child_node(uid).name = item.text()
        self._update_info(uid=uid)

    @QtCore.Slot(QtWidgets.QListWidgetItem)
    def _show_figure_item(self, item: QtWidgets.QListWidgetItem) -> None:
        if not hasattr(self, "figure_list"):
            return
        uid = self._figure_uid_from_item(item)
        if uid is not None and self._is_figure_uid(uid):
            self.show_childtool(uid)

    @QtCore.Slot(QtCore.QPoint)
    def _show_figure_menu(self, position: QtCore.QPoint) -> None:
        if not hasattr(self, "figure_list"):
            return
        menu = QtWidgets.QMenu("Figures", self.figure_list)
        menu.addAction(self.show_action)
        menu.addAction(self.hide_action)
        menu.addSeparator()
        menu.addAction(self.duplicate_action)
        menu.addAction(self.remove_action)
        menu.addAction(self.rename_action)
        viewport = self.figure_list.viewport()
        if viewport is not None:  # pragma: no branch
            menu.popup(viewport.mapToGlobal(position))

    @staticmethod
    def _figure_all_axes(nrows: int, ncols: int) -> tuple[tuple[int, int], ...]:
        return tuple((row, col) for row in range(nrows) for col in range(ncols))

    @staticmethod
    def _figure_middle_coordinate_value(data: xr.DataArray, dim: str) -> float | None:
        coord = data.coords.get(dim)
        if coord is None or coord.size == 0:
            return None
        with contextlib.suppress(TypeError, ValueError):
            return float(coord.values[int(coord.size // 2)])
        return None

    def _make_figure_operations_for_sources(
        self,
        source_data: Mapping[str, xr.DataArray],
        *,
        setup: typing.Any,
    ) -> tuple[typing.Any, ...]:
        from erlab.interactive._figurecomposer import (
            FigureAxesSelectionState,
            FigureOperationState,
        )
        from erlab.interactive._figurecomposer._sources import _public_source_data

        if not source_data:
            return ()

        source_names = tuple(source_data)
        all_axes = FigureAxesSelectionState(
            axes=self._figure_all_axes(setup.nrows, setup.ncols)
        )
        squeezed = [
            _public_source_data(data).squeeze(drop=True)
            for data in source_data.values()
        ]

        if all(data.ndim > 1 for data in squeezed):
            first = squeezed[0]
            slice_dim = None
            slice_values: tuple[float, ...] = ()
            if first.ndim > 2:
                slice_dim = str(first.dims[0])
                value = self._figure_middle_coordinate_value(first, slice_dim)
                if value is not None:
                    slice_values = (value,)
            operation = FigureOperationState.plot_slices(
                label="plot_slices",
                sources=source_names,
                axes=all_axes,
                slice_dim=slice_dim,
                slice_values=slice_values,
            ).model_copy(update={"order": "F"} if len(source_names) > 1 else {})
            return (operation,)

        operations = []
        for index, source_name in enumerate(source_names):
            row = min(index, setup.nrows - 1)
            operations.append(
                FigureOperationState.line(
                    label=source_name,
                    source=source_name,
                    axes=FigureAxesSelectionState(axes=((row, 0),)),
                )
            )
        return tuple(operations)

    def _figure_plot_slices_operation_from_targets(
        self, targets: tuple[int | str, ...], source_names: tuple[str, ...]
    ) -> typing.Any | None:
        from erlab.interactive._figurecomposer import FigureOperationKind
        from erlab.interactive._figurecomposer._seeding import (
            plot_slices_operation_with_source_styles,
        )

        if not source_names:
            return None
        source_operations: list[typing.Any] = []
        for index, target in enumerate(targets):
            if index >= len(source_names):
                break
            node = self._node_for_target(target)
            tool = node.imagetool
            if tool is None or not tool.slicer_area.axes:
                return None
            plot = typing.cast("typing.Any", tool.slicer_area.axes[0])
            if not plot.is_image:
                return None
            source_operation = plot.figure_composer_operation(
                source_name=source_names[index]
            )
            if source_operation.kind != FigureOperationKind.PLOT_SLICES:
                return None
            source_operations.append(source_operation)
        if not source_operations:
            return None

        if any(operation.map_selections for operation in source_operations):
            from erlab.interactive._figurecomposer._exceptions import (
                FigureComposerPlotSlicesSelectionError,
            )

            raise FigureComposerPlotSlicesSelectionError
        operation = source_operations[0]
        updates: dict[str, typing.Any] = {"sources": source_names}
        if len(source_names) > 1:
            updates["order"] = "F"
        expanded_operation = operation.model_copy(update=updates)
        return plot_slices_operation_with_source_styles(
            expanded_operation,
            tuple(source_operations),
            selections_per_source=1,
        )

    def _show_figure_plot_slices_selection_error(self, error: Exception) -> None:
        from erlab.interactive._figurecomposer._exceptions import (
            PLOT_SLICES_SELECTION_ERROR_TITLE,
        )

        QtWidgets.QMessageBox.warning(
            self, PLOT_SLICES_SELECTION_ERROR_TITLE, str(error)
        )

    @staticmethod
    def _figure_plot_slices_grid_shape(operation: typing.Any) -> tuple[int, int]:
        map_count = (
            len(operation.map_selections)
            if operation.map_selections
            else len(operation.sources)
        )
        map_count = max(map_count, 1)
        slice_count = max(len(operation.slice_values), 1)
        if operation.order == "F":
            return slice_count, map_count
        return map_count, slice_count

    def _figure_setup_for_operation(
        self, operation: typing.Any | None, source_data: Mapping[str, xr.DataArray]
    ) -> typing.Any:
        from erlab.interactive._figurecomposer import (
            FigureOperationKind,
            FigureSubplotsState,
        )
        from erlab.interactive._figurecomposer._sources import _public_source_data

        if operation is not None and operation.kind == FigureOperationKind.PLOT_SLICES:
            nrows, ncols = self._figure_plot_slices_grid_shape(operation)
            return FigureSubplotsState(nrows=nrows, ncols=ncols)

        squeezed = [
            _public_source_data(data).squeeze(drop=True)
            for data in source_data.values()
        ]
        if squeezed and all(data.ndim > 1 for data in squeezed):
            return FigureSubplotsState(
                nrows=max(len(squeezed), 1),
                ncols=1,
            )
        return FigureSubplotsState(nrows=max(len(squeezed), 1), ncols=1)

    def create_figure_from_targets(
        self,
        targets: Iterable[int | str],
        *,
        operation: typing.Any | None = None,
        custom_code: str | None = None,
        title: str | None = None,
        show: bool = True,
    ) -> str | None:
        from erlab.interactive._figurecomposer import (
            FigureAxesSelectionState,
            FigureComposerTool,
            FigureOperationKind,
            FigureOperationState,
            FigureSourceState,
        )
        from erlab.interactive._figurecomposer._sources import _public_source_data

        resolved_targets = tuple(dict.fromkeys(targets))
        if not resolved_targets:
            return None

        source_data: dict[str, xr.DataArray] = {}
        sources: list[FigureSourceState] = []

        for target in resolved_targets:
            node = self._node_for_target(target)
            data = node.current_source_data()
            script_input = self._script_input_for_node(node)
            source = FigureSourceState.from_script_input(script_input)
            source_data[source.name] = data
            sources.append(source)

        primary_source = sources[0].name
        source_names = tuple(source.name for source in sources)
        auto_operation = None
        if (
            operation is None
            and custom_code is None
            and all(
                _public_source_data(data).squeeze(drop=True).ndim > 1
                for data in source_data.values()
            )
        ):
            from erlab.interactive._figurecomposer._exceptions import (
                FigureComposerPlotSlicesSelectionError,
            )

            try:
                auto_operation = self._figure_plot_slices_operation_from_targets(
                    resolved_targets, source_names
                )
            except FigureComposerPlotSlicesSelectionError as exc:
                self._show_figure_plot_slices_selection_error(exc)
                return None
        setup = self._figure_setup_for_operation(
            None if custom_code is not None else operation or auto_operation,
            source_data,
        )
        operations: tuple[FigureOperationState, ...]
        if operation is not None:
            if (
                operation.kind == FigureOperationKind.PLOT_SLICES
                and not operation.axes.expression
            ):
                operation = operation.model_copy(
                    update={
                        "axes": FigureAxesSelectionState(
                            axes=self._figure_all_axes(setup.nrows, setup.ncols)
                        )
                    }
                )
            operations = (operation,)
        elif custom_code is not None:
            operations = (
                FigureOperationState.custom(
                    label=title or "custom code",
                    code=custom_code,
                    trusted=True,
                ),
            )
        elif auto_operation is not None:
            if not auto_operation.axes.expression:
                auto_operation = auto_operation.model_copy(
                    update={
                        "axes": FigureAxesSelectionState(
                            axes=self._figure_all_axes(setup.nrows, setup.ncols)
                        )
                    }
                )
            operations = (auto_operation,)
        else:
            operations = typing.cast(
                "tuple[FigureOperationState, ...]",
                self._make_figure_operations_for_sources(
                    source_data,
                    setup=setup,
                ),
            )

        tool = FigureComposerTool.from_sources(
            source_data,
            sources=tuple(sources),
            operations=operations,
            setup=setup,
            primary_source=primary_source,
        )
        tool._tool_display_name = (
            title if title is not None else self._next_figure_display_name()
        )
        uid = self.add_figuretool(tool, show=show)
        self._select_figure_uid(uid)
        return uid

    @QtCore.Slot()
    def create_figure_from_selection(self) -> None:
        targets = self._selected_figure_source_targets()
        if not targets:
            return
        figure_uids = tuple(self._figure_uids())
        if not figure_uids:
            self.create_figure_from_targets(targets)
            return

        dialog = _AppendFigureTargetDialog(
            self,
            figure_uids,
            None,
            allow_new_figure=True,
        )
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        if dialog.is_new_figure():
            self.create_figure_from_targets(targets)
            return
        target = dialog.selected_target()
        if target is None:
            return
        figure_uid, axes_selection = target
        self.append_figure_from_targets(
            targets,
            figure_uid=figure_uid,
            axes_selection=axes_selection,
        )

    def create_figure_from_slicer_area(
        self,
        slicer_area: ImageSlicerArea,
        *,
        operation: typing.Any | None = None,
        custom_code: str | None = None,
        title: str | None = None,
        show: bool = True,
    ) -> str | None:
        target = self.target_from_slicer_area(slicer_area)
        if target is None:
            return None
        return self.create_figure_from_targets(
            (target,),
            operation=operation,
            custom_code=custom_code,
            title=title,
            show=show,
        )

    def _append_single_axis_selection(self, figure_uid: str) -> typing.Any | None:
        from erlab.interactive._figurecomposer import FigureAxesSelectionState
        from erlab.interactive._figurecomposer._gridspec import (
            _gridspec_all_axes_ids,
            _gridspec_valid_axes_ids,
        )

        tool = self._child_node(figure_uid).tool_window
        if tool is None:
            return None
        setup = tool.tool_status.setup
        if setup.layout_mode == "gridspec":
            axes_ids = _gridspec_valid_axes_ids(setup, _gridspec_all_axes_ids(setup))
            if len(axes_ids) == 1:
                return FigureAxesSelectionState(axes_ids=axes_ids)
            return None
        all_axes = self._figure_all_axes(setup.nrows, setup.ncols)
        if len(all_axes) == 1:
            return FigureAxesSelectionState(axes=all_axes)
        return None

    def _prompt_append_figure_target(
        self, operation: typing.Any | None, *, figure_uid: str | None = None
    ) -> tuple[str, typing.Any] | None:
        figure_uids = self._figure_uids()
        if not figure_uids:
            return None
        if figure_uid is not None:
            if not self._is_figure_uid(figure_uid):
                return None
            automatic = self._append_single_axis_selection(figure_uid)
            if automatic is not None:
                return figure_uid, automatic
            figure_uids = (figure_uid,)
        elif len(figure_uids) == 1:
            automatic = self._append_single_axis_selection(figure_uids[0])
            if automatic is not None:
                return figure_uids[0], automatic

        dialog = _AppendFigureTargetDialog(self, tuple(figure_uids), operation)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        return dialog.selected_target()

    def append_figure_from_targets(
        self,
        targets: Iterable[int | str],
        *,
        figure_uid: str | None = None,
        axes_selection: typing.Any | None = None,
        operation: typing.Any | None = None,
        show: bool = True,
    ) -> bool:
        from erlab.interactive._figurecomposer import (
            FigureComposerTool,
            FigureSourceState,
        )
        from erlab.interactive._figurecomposer._sources import _public_source_data

        resolved_targets = tuple(dict.fromkeys(targets))
        if not resolved_targets:
            return False

        source_data: dict[str, xr.DataArray] = {}
        sources = []
        for target in resolved_targets:
            source_node = self._node_for_target(target)
            data = source_node.current_source_data()
            source = self._script_input_for_node(source_node)
            source_model = FigureSourceState.from_script_input(source)
            source_data[source_model.name] = data
            sources.append(source_model)

        source_names = tuple(source.name for source in sources)
        auto_operation = None
        if operation is None and all(
            _public_source_data(data).squeeze(drop=True).ndim > 1
            for data in source_data.values()
        ):
            from erlab.interactive._figurecomposer._exceptions import (
                FigureComposerPlotSlicesSelectionError,
            )

            try:
                auto_operation = self._figure_plot_slices_operation_from_targets(
                    resolved_targets, source_names
                )
            except FigureComposerPlotSlicesSelectionError as exc:
                self._show_figure_plot_slices_selection_error(exc)
                return False
        prompt_operation = operation or auto_operation

        if axes_selection is None:
            prompt = self._prompt_append_figure_target(
                prompt_operation, figure_uid=figure_uid
            )
            if prompt is None:
                return False
            resolved_figure_uid, axes_selection = prompt
        else:
            if figure_uid is None or not self._is_figure_uid(figure_uid):
                return False
            resolved_figure_uid = figure_uid

        node = self._child_node(resolved_figure_uid)
        tool = node.tool_window
        if not isinstance(tool, FigureComposerTool):
            return False

        tool.add_sources(tuple(sources), source_data)
        operations = (
            (operation,)
            if operation is not None
            else (auto_operation,)
            if auto_operation is not None
            else self._make_figure_operations_for_sources(
                source_data, setup=tool.tool_status.setup
            )
        )
        for appended in operations:
            tool.add_operation(appended.model_copy(update={"axes": axes_selection}))

        self._mark_workspace_dirty(uid=resolved_figure_uid, state=True)
        self._select_figure_uid(resolved_figure_uid)
        if show:
            node.show()
        return True

    def append_figure_from_slicer_area(
        self,
        slicer_area: ImageSlicerArea,
        *,
        operation: typing.Any,
        show: bool = True,
    ) -> bool:
        target = self.target_from_slicer_area(slicer_area)
        if target is None:
            return False
        return self.append_figure_from_targets(
            (target,), operation=operation, show=show
        )

    @QtCore.Slot()
    def reindex(self) -> None:
        """Reset indices of ImageTool windows to be consecutive in display order."""
        with self._reindex_lock:
            self._tool_graph.reindex_roots()

        self.tree_view.refresh()
        self._mark_workspace_structure_dirty("Reindexed root windows")

    @QtCore.Slot(int)
    def remove_imagetool(self, index: int, *, update_view: bool = True) -> None:
        """Remove the ImageTool window corresponding to the given index."""
        if index not in self._tool_graph.root_wrappers:
            return
        wrapper = self._tool_graph.root_wrappers[index]
        self._mark_removed_subtree_dirty(wrapper.uid)
        descendant_uids = list(wrapper._childtool_indices)
        if update_view:
            self.tree_view.imagetool_removed(index)

        for uid in list(descendant_uids):
            self._remove_uid_target(uid)

        self._tool_graph.unregister_root(index)
        self._refresh_dependency_dependents(wrapper.uid)
        wrapper.dispose()
        wrapper.deleteLater()

    @contextlib.contextmanager
    def _bulk_remove_context(self) -> Iterator[None]:
        outermost = self._bulk_remove_depth == 0
        self._bulk_remove_depth += 1
        if outermost:
            self._link_registry.clear_pending_cleanup()
            self.setUpdatesEnabled(False)
            self.tree_view.setUpdatesEnabled(False)
        try:
            yield
        finally:
            self._bulk_remove_depth -= 1
            if outermost:
                self.tree_view.setUpdatesEnabled(True)
                self.setUpdatesEnabled(True)

                if self._link_registry.pop_pending_cleanup():
                    self._cleanup_linkers()

                self._update_actions()
                self._update_info()

    def _remove_imagetools(
        self,
        indices: list[int | str],
        *,
        child_uids: list[str] | None = None,
        clear_view: bool = False,
    ) -> None:
        root_indices: list[int] = []
        child_targets: list[str] = []
        covered_child_uids: set[str] = set()
        for target in indices:
            if isinstance(target, int):
                root_indices.append(target)
                wrapper = self._tool_graph.root_wrappers.get(target)
                if wrapper is not None:
                    direct_children = list(wrapper._childtool_indices)
                    covered_child_uids.update(direct_children)
                    for child_uid in direct_children:
                        covered_child_uids.update(self._iter_descendant_uids(child_uid))
            else:
                child_targets.append(target)

        for uid in child_uids or []:
            if uid not in covered_child_uids and uid not in child_targets:
                child_targets.append(uid)

        if len(root_indices) == 0 and len(child_targets) == 0:
            return

        with self._bulk_remove_context():
            if clear_view:
                self.tree_view.clear_imagetools()

            for index in root_indices:
                self.remove_imagetool(index, update_view=not clear_view)

            for uid in child_targets:
                self._remove_childtool(uid)

    def remove_all_tools(self) -> None:
        """Remove all ImageTool windows."""
        self._remove_imagetools(
            list(self._tool_graph.root_wrappers.keys()),
            child_uids=list(self._tool_graph.figure_uids),
            clear_view=True,
        )

    @QtCore.Slot(int)
    def show_imagetool(self, index: int) -> None:
        """Show the ImageTool window corresponding to the given index."""
        if index in self._tool_graph.root_wrappers:
            self._tool_graph.root_wrappers[index].show()

    @QtCore.Slot()
    def _request_reload_linkers(self) -> None:
        if self._link_registry.request_cleanup(defer=self._bulk_remove_depth > 0):
            self.sigLinkersChanged.emit()

    @QtCore.Slot()
    def _cleanup_linkers(self) -> None:
        """Remove linkers with one or no children."""
        self._link_registry.cleanup_stale()
        self.sigLinkersChanged.emit()

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

    def _node_info_html(self, node: _ImageToolWrapper | _ManagedWindowNode) -> str:
        return self._details_panel._node_info_html(node)

    def _clear_metadata(self) -> None:
        self._details_panel._clear_metadata()

    def _set_metadata_node(self, node: _ImageToolWrapper | _ManagedWindowNode) -> None:
        self._details_panel._set_metadata_node(node)

    def _set_metadata_fields(self, fields: list[_MetadataField]) -> None:
        self._details_panel._set_metadata_fields(fields)

    def _show_load_source_details(self, details: _LoadSourceDetails) -> None:
        self._details_panel._show_load_source_details(details)

    def _load_source_for_replay(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> tuple[str, str] | None:
        return self._details_panel._load_source_for_replay(node)

    def _prompt_replay_input_name(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> str | None:
        return self._details_panel._prompt_replay_input_name(node)

    def _update_metadata_pane(self) -> None:
        self._details_panel._update_metadata_pane()

    def _selected_derivation_items(self) -> list[QtWidgets.QListWidgetItem]:
        return self._details_panel._selected_derivation_items()

    def _selected_derivation_code(self) -> str | None:
        return self._details_panel._selected_derivation_code()

    def _selected_derivation_row(
        self,
    ) -> provenance._ProvenanceDisplayRow | None:
        return self._details_panel._selected_derivation_row()

    def _build_metadata_derivation_menu(self) -> QtWidgets.QMenu | None:
        return self._details_panel._build_metadata_derivation_menu()

    def _show_metadata_derivation_menu(self, pos: QtCore.QPoint) -> None:
        self._details_panel._show_metadata_derivation_menu(pos)

    def _copy_selected_derivation_code(self) -> None:
        self._details_panel._copy_selected_derivation_code()

    def _copy_full_derivation_code(self) -> None:
        self._details_panel._copy_full_derivation_code()

    def _edit_selected_derivation_step(self) -> None:
        self._details_panel._edit_selected_derivation_step()

    def _revert_selected_derivation_step(self) -> None:
        self._details_panel._revert_selected_derivation_step()

    def _update_info(self, *, uid: str | None = None) -> None:
        self._details_panel._update_info(uid=uid)

    def _schedule_tool_metadata_update(self, uid: str) -> None:
        self._details_panel._schedule_tool_metadata_update(uid)

    def _flush_pending_tool_metadata_updates(self, pending: set[str]) -> None:
        self._details_panel._flush_pending_tool_metadata_updates(pending)

    def _update_actions(self) -> None:
        self._details_panel._update_actions()

    def about(self) -> None:
        self._widgets_controller.about()

    def updated(self, old_version: str, new_version: str) -> None:
        self._widgets_controller.updated(old_version, new_version)

    def open_log_directory(self) -> None:
        self._widgets_controller.open_log_directory()

    def _parse_progressbar(self, message: str) -> None:
        self._widgets_controller._parse_progressbar(message)

    def _show_alert(
        self, levelname: str, levelno: int, message: str, formatted_traceback: str
    ) -> None:
        self._widgets_controller._show_alert(
            levelname, levelno, message, formatted_traceback
        )

    def _ignore_warning_message(self, message: str) -> None:
        self._widgets_controller._ignore_warning_message(message)

    def _unregister_alert(self, alert: erlab.interactive.utils.MessageDialog) -> None:
        self._widgets_controller._unregister_alert(alert)

    def _clear_all_alerts(self) -> None:
        self._widgets_controller._clear_all_alerts()

    def _handle_uncaught_exception(
        self,
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_traceback: typing.Any,
    ) -> None:
        self._widgets_controller._handle_uncaught_exception(
            exc_type, exc_value, exc_traceback
        )

    def _start_server_pair(
        self, *, port: int, watch_port: int
    ) -> tuple[_ManagerServer, _WatcherServer, int, int]:
        return self._widgets_controller._start_server_pair(
            port=port, watch_port=watch_port
        )

    def _start_manager_servers(
        self,
    ) -> tuple[_ManagerServer, _WatcherServer, int, int]:
        return self._widgets_controller._start_manager_servers()

    def _stop_servers(self) -> None:
        self._widgets_controller._stop_servers()

    def open_settings(self) -> None:
        self._widgets_controller.open_settings()

    def open_new_manager_instance(self) -> None:
        self._widgets_controller.open_new_manager_instance()

    def check_for_updates(self) -> None:
        self._widgets_controller.check_for_updates()

    @staticmethod
    def _normalize_recent_workspace_paths(
        paths: Iterable[str | os.PathLike[str]],
    ) -> list[pathlib.Path]:
        return _WorkspaceIOController._normalize_recent_workspace_paths(paths)

    def _recent_workspace_paths(self) -> list[pathlib.Path]:
        return self._workspace_controller._recent_workspace_paths()

    def _set_recent_workspace_paths(
        self, paths: Iterable[str | os.PathLike[str]]
    ) -> None:
        self._workspace_controller._set_recent_workspace_paths(paths)

    def _record_recent_workspace(self, fname: str | os.PathLike[str]) -> None:
        self._workspace_controller._record_recent_workspace(fname)

    def _clear_recent_workspaces(self) -> None:
        self._workspace_controller._clear_recent_workspaces()

    def _refresh_open_recent_menu_action(self) -> None:
        self._workspace_controller._refresh_open_recent_menu_action()

    def _populate_open_recent_menu(self) -> None:
        self._workspace_controller._populate_open_recent_menu()

    def open_recent_workspace(self, fname: str | os.PathLike[str]) -> bool:
        return self._workspace_controller.open_recent_workspace(fname)

    @property
    def workspace_path(self) -> str | None:
        return self._workspace_controller.workspace_path

    def show_workspace_properties(self) -> None:
        self._workspace_controller.show_workspace_properties()

    def _workspace_properties_state(self) -> _WorkspacePropertiesState:
        return self._workspace_controller._workspace_properties_state()

    @property
    def is_workspace_modified(self) -> bool:
        return self._workspace_controller.is_workspace_modified

    def _registry_heartbeat_tick(self) -> None:
        self._workspace_controller._refresh_manager_record(coalesce_if_busy=False)

    def _refresh_manager_record(self, *, coalesce_if_busy: bool = True) -> None:
        self._workspace_controller._refresh_manager_record(
            coalesce_if_busy=coalesce_if_busy
        )

    def _update_workspace_window_title(self) -> None:
        self._workspace_controller._update_workspace_window_title()

    def _release_workspace_lock(self) -> None:
        self._workspace_controller._release_workspace_lock()

    def _workspace_document_access(
        self, fname: str | os.PathLike[str]
    ) -> _WorkspaceDocumentAccess:
        return self._workspace_controller._workspace_document_access(fname)

    @contextlib.contextmanager
    def _workspace_document_access_context(
        self, fname: str | os.PathLike[str]
    ) -> Iterator[_WorkspaceDocumentAccess]:
        with self._workspace_controller._workspace_document_access_context(
            fname
        ) as access:
            yield access

    def _set_workspace_path(
        self,
        fname: str | os.PathLike[str] | None,
        *,
        workspace_lock: QtCore.QLockFile | None = None,
    ) -> None:
        self._workspace_controller._set_workspace_path(
            fname, workspace_lock=workspace_lock
        )

    def _adopt_workspace_path(self, fname: str | os.PathLike[str]) -> None:
        self._workspace_controller._adopt_workspace_path(fname)

    def _active_managed_window(self) -> QtWidgets.QWidget | None:
        return self._workspace_controller._active_managed_window()

    def _restore_focus_after_workspace_save(
        self, origin: QtWidgets.QWidget | None
    ) -> None:
        self._workspace_controller._restore_focus_after_workspace_save(origin)

    def _dirty_details_text(self) -> str:
        return self._workspace_controller._dirty_details_text()

    def _set_node_window_modified(self, uid: str, modified: bool) -> None:
        self._workspace_controller._set_node_window_modified(uid, modified)

    def _apply_workspace_dirty_event(
        self, event: _manager_workspace._WorkspaceDirtyEvent
    ) -> bool:
        return self._workspace_controller._apply_workspace_dirty_event(event)

    def _mark_workspace_dirty(
        self,
        *,
        uid: str | None = None,
        data: bool = False,
        state: bool = False,
        added: bool = False,
        removed: str | None = None,
        structure: str | None = None,
    ) -> None:
        self._workspace_controller._mark_workspace_dirty(
            uid=uid,
            data=data,
            state=state,
            added=added,
            removed=removed,
            structure=structure,
        )

    def _mark_node_added(self, uid: str) -> None:
        self._workspace_controller._mark_node_added(uid)

    def _mark_node_data_dirty(self, uid: str) -> None:
        self._workspace_controller._mark_node_data_dirty(uid)

    def _mark_node_state_dirty(self, uid: str) -> None:
        self._workspace_controller._mark_node_state_dirty(uid)

    def _mark_tool_info_dirty(self, uid: str) -> None:
        self._workspace_controller._mark_tool_info_dirty(uid)

    def _mark_workspace_structure_dirty(self, reason: str) -> None:
        self._workspace_controller._mark_workspace_structure_dirty(reason)

    def _mark_workspace_clean(self) -> None:
        self._workspace_controller._mark_workspace_clean()

    def _restore_workspace_dirty_events(
        self, events: Iterable[_manager_workspace._WorkspaceDirtyEvent]
    ) -> None:
        self._workspace_controller._restore_workspace_dirty_events(events)

    @contextlib.contextmanager
    def _workspace_load_context(self) -> Iterator[None]:
        with self._workspace_controller._workspace_load_context():
            yield

    def _drain_workspace_deferred_events(self) -> None:
        self._workspace_controller._drain_workspace_deferred_events()

    def _drain_workspace_restore_events(self) -> None:
        self._workspace_controller._drain_workspace_restore_events()

    def _workspace_state_snapshot(self) -> _WorkspaceStateSnapshot:
        return self._workspace_controller._workspace_state_snapshot()

    def _restore_workspace_state_snapshot(
        self, snapshot: _WorkspaceStateSnapshot
    ) -> None:
        self._workspace_controller._restore_workspace_state_snapshot(snapshot)

    def _install_workspace_save_shortcut(self, widget: QtWidgets.QWidget) -> None:
        self._workspace_controller._install_workspace_save_shortcut(widget)

    def _annotate_workspace_dataset(
        self,
        ds: xr.Dataset,
        node: _ImageToolWrapper | _ManagedWindowNode,
        *,
        kind: typing.Literal["imagetool", "tool"],
    ) -> xr.Dataset:
        return self._workspace_controller._annotate_workspace_dataset(
            ds, node, kind=kind
        )

    def _serialize_workspace_node(
        self,
        constructor: dict[str, xr.Dataset],
        node: _ImageToolWrapper | _ManagedWindowNode,
        path: str,
        *,
        include_children: bool,
    ) -> None:
        self._workspace_controller._serialize_workspace_node(
            constructor, node, path, include_children=include_children
        )

    def _to_datatree(
        self, close: bool = False, include_children: bool = True
    ) -> xr.DataTree:
        return self._workspace_controller._to_datatree(close, include_children)

    def _load_workspace_figures(
        self,
        tree: xr.DataTree,
        *,
        manifest: dict[str, typing.Any] | None = None,
        workspace_file_path: str | os.PathLike[str] | None = None,
        loaded_targets_by_uid: dict[str, int | str] | None = None,
    ) -> None:
        self._workspace_controller._load_workspace_figures(
            tree,
            manifest=manifest,
            workspace_file_path=workspace_file_path,
            loaded_targets_by_uid=loaded_targets_by_uid,
        )

    def _load_workspace_imagetool_dataset(
        self,
        ds: xr.Dataset,
        *,
        parent_target: int | str | None,
        node_path: str | None,
        loaded_targets_by_uid: dict[str, int | str] | None = None,
    ) -> int | str:
        return self._workspace_controller._load_workspace_imagetool_dataset(
            ds,
            parent_target=parent_target,
            node_path=node_path,
            loaded_targets_by_uid=loaded_targets_by_uid,
        )

    def _load_workspace_tool_dataset(
        self,
        ds: xr.Dataset,
        *,
        parent_target: int | str | None,
        loaded_targets_by_uid: dict[str, int | str] | None = None,
    ) -> int | str:
        return self._workspace_controller._load_workspace_tool_dataset(
            ds,
            parent_target=parent_target,
            loaded_targets_by_uid=loaded_targets_by_uid,
        )

    @staticmethod
    def _workspace_saved_uid_from_dataset(ds: xr.Dataset) -> str | None:
        return _WorkspaceIOController._workspace_saved_uid_from_dataset(ds)

    def _record_workspace_loaded_imagetool_target(
        self,
        ds: xr.Dataset,
        target: int | str,
        loaded_targets_by_uid: dict[str, int | str] | None,
    ) -> None:
        self._workspace_controller._record_workspace_loaded_imagetool_target(
            ds, target, loaded_targets_by_uid
        )

    def _record_workspace_loaded_tool_target(
        self,
        ds: xr.Dataset,
        target: int | str,
        loaded_targets_by_uid: dict[str, int | str] | None,
    ) -> None:
        self._workspace_controller._record_workspace_loaded_tool_target(
            ds, target, loaded_targets_by_uid
        )

    def _restore_workspace_link_groups(
        self,
        manifest: Mapping[str, typing.Any] | None,
        loaded_targets_by_uid: Mapping[str, int | str],
    ) -> None:
        self._workspace_controller._restore_workspace_link_groups(
            manifest, loaded_targets_by_uid
        )

    def _load_workspace_node(
        self,
        node_tree: xr.DataTree,
        *,
        parent_target: int | str | None = None,
        selection_item: QtWidgets.QTreeWidgetItem | None = None,
        manifest: dict[str, typing.Any] | None = None,
        node_path: str | None = None,
        workspace_file_path: str | os.PathLike[str] | None = None,
        loaded_targets_by_uid: dict[str, int | str] | None = None,
    ) -> int | str:
        return self._workspace_controller._load_workspace_node(
            node_tree,
            parent_target=parent_target,
            selection_item=selection_item,
            manifest=manifest,
            node_path=node_path,
            workspace_file_path=workspace_file_path,
            loaded_targets_by_uid=loaded_targets_by_uid,
        )

    def _load_workspace_roots(
        self,
        tree: xr.DataTree,
        root_keys: Iterable[str],
        *,
        root_item: QtWidgets.QTreeWidgetItem | None = None,
        manifest: dict[str, typing.Any] | None = None,
        workspace_file_path: str | os.PathLike[str] | None = None,
        loaded_targets_by_uid: dict[str, int | str] | None = None,
    ) -> None:
        self._workspace_controller._load_workspace_roots(
            tree,
            root_keys,
            root_item=root_item,
            manifest=manifest,
            workspace_file_path=workspace_file_path,
            loaded_targets_by_uid=loaded_targets_by_uid,
        )

    def _from_h5py_workspace_file(
        self,
        fname: str | os.PathLike[str],
        manifest: Mapping[str, typing.Any],
        *,
        replace: bool,
        mark_dirty: bool,
    ) -> bool:
        return self._workspace_controller._from_h5py_workspace_file(
            fname, manifest, replace=replace, mark_dirty=mark_dirty
        )

    def _restore_replaced_workspace(
        self, backup_tree: xr.DataTree, snapshot: _WorkspaceStateSnapshot
    ) -> None:
        self._workspace_controller._restore_replaced_workspace(backup_tree, snapshot)

    def _from_datatree(
        self,
        tree: xr.DataTree,
        *,
        replace: bool = False,
        mark_dirty: bool = True,
        select: bool = True,
        workspace_file_path: str | os.PathLike[str] | None = None,
    ) -> bool:
        return self._workspace_controller._from_datatree(
            tree,
            replace=replace,
            mark_dirty=mark_dirty,
            select=select,
            workspace_file_path=workspace_file_path,
        )

    def _parse_datatree_compat_v1(self, tree: xr.DataTree) -> xr.DataTree:
        return self._workspace_controller._parse_datatree_compat_v1(tree)

    def _parse_datatree_compat_v2(self, tree: xr.DataTree) -> xr.DataTree:
        return self._workspace_controller._parse_datatree_compat_v2(tree)

    def _is_datatree_workspace(self, tree: xr.DataTree) -> bool:
        return self._workspace_controller._is_datatree_workspace(tree)

    def _workspace_node_path(self, uid: str) -> str:
        return self._workspace_controller._workspace_node_path(uid)

    def _workspace_payload_path(self, uid: str) -> str:
        return self._workspace_controller._workspace_payload_path(uid)

    def _workspace_root_indices(self) -> tuple[int, ...]:
        return self._workspace_controller._workspace_root_indices()

    def _workspace_link_metadata_by_uid(self) -> dict[str, tuple[int, bool]]:
        return self._workspace_controller._workspace_link_metadata_by_uid()

    def _workspace_node_manifest_entries(self) -> list[dict[str, typing.Any]]:
        return self._workspace_controller._workspace_node_manifest_entries()

    @staticmethod
    def _tree_item_child_by_key(
        item: QtWidgets.QTreeWidgetItem | None, key: str
    ) -> QtWidgets.QTreeWidgetItem | None:
        return _WorkspaceIOController._tree_item_child_by_key(item, key)

    def _workspace_root_attrs_payload(
        self, *, delta_save_count: int | None = None
    ) -> dict[str, typing.Any]:
        return self._workspace_controller._workspace_root_attrs_payload(
            delta_save_count=delta_save_count
        )

    def _workspace_layout_snapshot(self) -> dict[str, typing.Any]:
        return self._workspace_controller._workspace_layout_snapshot()

    def _restore_workspace_layout(
        self, manifest: Mapping[str, typing.Any] | None
    ) -> None:
        self._workspace_controller._restore_workspace_layout(manifest)

    def _write_full_workspace_file(self, fname: str | os.PathLike[str]) -> None:
        self._workspace_controller._write_full_workspace_file(fname)

    def _workspace_highest_dirty_data_roots(self) -> list[str]:
        return self._workspace_controller._workspace_highest_dirty_data_roots()

    def _save_workspace_delta(self, fname: str | os.PathLike[str]) -> None:
        self._workspace_controller._save_workspace_delta(fname)

    def _save_workspace_document(
        self,
        fname: str | os.PathLike[str],
        *,
        force_full: bool = False,
        document_access: _WorkspaceDocumentAccess | None = None,
    ) -> None:
        self._workspace_controller._save_workspace_document(
            fname, force_full=force_full, document_access=document_access
        )

    def _workspace_save_dialog(
        self,
        *,
        native: bool = True,
        caption: str = "Save Workspace",
        selected_file: str | os.PathLike[str] | None = None,
    ) -> str | None:
        return self._workspace_controller._workspace_save_dialog(
            native=native, caption=caption, selected_file=selected_file
        )

    def _confirm_save_dirty_workspace(self, action_text: str) -> bool:
        return self._workspace_controller._confirm_save_dirty_workspace(action_text)

    def _show_legacy_workspace_upgrade_message(
        self, fname: str | os.PathLike[str]
    ) -> None:
        self._workspace_controller._show_legacy_workspace_upgrade_message(fname)

    def _save_legacy_workspace_as_v4(
        self,
        fname: str | os.PathLike[str],
        *,
        native: bool = True,
        existing_access: _WorkspaceDocumentAccess | None = None,
    ) -> tuple[str, QtCore.QLockFile | None] | None:
        return self._workspace_controller._save_legacy_workspace_as_v4(
            fname, native=native, existing_access=existing_access
        )

    def _associate_loaded_workspace_file(
        self,
        fname: str | os.PathLike[str],
        schema_version: int,
        *,
        native: bool = True,
        delta_save_count: int = 0,
        workspace_access: _WorkspaceDocumentAccess | None = None,
        rebind_data: bool = True,
    ) -> None:
        self._workspace_controller._associate_loaded_workspace_file(
            fname,
            schema_version,
            native=native,
            delta_save_count=delta_save_count,
            workspace_access=workspace_access,
            rebind_data=rebind_data,
        )

    def _workspace_rebind_data_for_uid(
        self, fname: str | os.PathLike[str], uid: str, *, chunks: typing.Any
    ) -> xr.DataArray:
        return self._workspace_controller._workspace_rebind_data_for_uid(
            fname, uid, chunks=chunks
        )

    def _workspace_data_backing_snapshot(
        self,
    ) -> dict[str, tuple[str, tuple[str, ...]]]:
        return self._workspace_controller._workspace_data_backing_snapshot()

    def _rebind_workspace_backed_imagetools(
        self,
        fname: str | os.PathLike[str],
        *,
        targets: Iterable[int | str] | None = None,
        chunks: typing.Any = _WORKSPACE_REBIND_KEEP_CHUNKS,
        backing_snapshot: Mapping[str, tuple[str, tuple[str, ...]]] | None = None,
        old_workspace_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self._workspace_controller._rebind_workspace_backed_imagetools(
            fname,
            targets=targets,
            chunks=chunks,
            backing_snapshot=backing_snapshot,
            old_workspace_path=old_workspace_path,
        )

    def offload_to_workspace(
        self, targets: Iterable[int | str], *, native: bool = True
    ) -> bool:
        return self._workspace_controller.offload_to_workspace(targets, native=native)

    def _workspace_requires_full_save(self, fname: str | os.PathLike[str]) -> bool:
        return self._workspace_controller._workspace_requires_full_save(fname)

    def _workspace_rewrite_group_snapshot(
        self, uid: str
    ) -> tuple[str, dict[str, xr.Dataset]]:
        return self._workspace_controller._workspace_rewrite_group_snapshot(uid)

    def _workspace_attr_update_snapshot(
        self, uid: str
    ) -> tuple[str, dict[str, typing.Any], tuple[str, dict[str, xr.Dataset]]] | None:
        return self._workspace_controller._workspace_attr_update_snapshot(uid)

    def _workspace_delta_save_snapshot(
        self,
        generation: int,
        root_attrs: dict[str, typing.Any],
        delta_save_count: int,
    ) -> _manager_workspace._WorkspaceSaveSnapshot:
        return self._workspace_controller._workspace_delta_save_snapshot(
            generation, root_attrs, delta_save_count
        )

    def _workspace_save_snapshot(
        self, fname: str | os.PathLike[str]
    ) -> _manager_workspace._WorkspaceSaveSnapshot:
        return self._workspace_controller._workspace_save_snapshot(fname)

    def _workspace_full_save_snapshot(
        self, generation: int
    ) -> _manager_workspace._WorkspaceSaveSnapshot:
        return self._workspace_controller._workspace_full_save_snapshot(generation)

    def _workspace_full_save_copy_groups(
        self, tree: xr.DataTree
    ) -> tuple[str | None, tuple[tuple[str, str, dict[str, typing.Any] | None], ...]]:
        return self._workspace_controller._workspace_full_save_copy_groups(tree)

    def _open_workspace_save_wait_dialog(
        self,
        parent: QtWidgets.QWidget,
        *,
        title: str = "Saving Workspace",
        label_text: str = "Saving workspace...",
    ) -> QtWidgets.QDialog:
        return self._workspace_controller._open_workspace_save_wait_dialog(
            parent, title=title, label_text=label_text
        )

    def _set_workspace_save_actions_enabled(
        self, enabled: bool
    ) -> tuple[bool, bool, bool]:
        return self._workspace_controller._set_workspace_save_actions_enabled(enabled)

    def _restore_workspace_save_actions_enabled(
        self, previous: tuple[bool, bool, bool]
    ) -> None:
        self._workspace_controller._restore_workspace_save_actions_enabled(previous)

    def _run_workspace_save_worker(
        self,
        fname: str | os.PathLike[str],
        snapshot: _manager_workspace._WorkspaceSaveSnapshot,
        origin: QtWidgets.QWidget | None,
        *,
        wait_dialog_title: str = "Saving Workspace",
        wait_dialog_text: str = "Saving workspace...",
    ) -> tuple[bool, float, str]:
        return self._workspace_controller._run_workspace_save_worker(
            fname,
            snapshot,
            origin,
            wait_dialog_title=wait_dialog_title,
            wait_dialog_text=wait_dialog_text,
        )

    def save(self, *, native: bool = True) -> bool:
        return self._workspace_controller.save(native=native)

    def save_as(self, *, native: bool = True) -> bool:
        return self._workspace_controller.save_as(native=native)

    def _compact_workspace_before_shutdown(self) -> None:
        self._workspace_controller._compact_workspace_before_shutdown()

    def compact_workspace(self) -> bool:
        return self._workspace_controller.compact_workspace()

    def _save_to_file(self, fname: str) -> None:
        self._workspace_controller._save_to_file(fname)

    def _load_workspace_file(
        self,
        fname: str | os.PathLike[str],
        *,
        replace: bool,
        associate: bool,
        mark_dirty: bool,
        select: bool,
        native: bool = True,
    ) -> bool:
        return self._workspace_controller._load_workspace_file(
            fname,
            replace=replace,
            associate=associate,
            mark_dirty=mark_dirty,
            select=select,
            native=native,
        )

    def load(self, *, native: bool = True) -> bool:
        return self._workspace_controller.load(native=native)

    def import_workspace(self, *, native: bool = True) -> bool:
        return self._workspace_controller.import_workspace(native=native)

    def open(self, *, native: bool = True) -> None:
        self._workspace_controller.open(native=native)

    def _data_recv(
        self,
        data: list[xr.DataArray] | list[xr.Dataset],
        kwargs: dict[str, typing.Any],
        *,
        watched_var: tuple[str, str] | None = None,
        watched_metadata: Mapping[str, typing.Any] | None = None,
        show: bool | None = None,
    ) -> list[bool]:
        return self._workspace_controller._data_recv(
            data,
            kwargs,
            watched_var=watched_var,
            watched_metadata=watched_metadata,
            show=show,
        )

    def _dependency_refs_for_uid(
        self, uid: str
    ) -> tuple[provenance.ScriptInputDependencyRef, ...]:
        return self._lineage_controller._dependency_refs_for_uid(uid)

    def dependency_status_for_uid(self, uid: str) -> _DependencyStatus | None:
        return self._lineage_controller.dependency_status_for_uid(uid)

    def dependency_status_label_for_uid(self, uid: str) -> str | None:
        return self._lineage_controller.dependency_status_label_for_uid(uid)

    def dependency_status_badge_for_uid(self, uid: str) -> str | None:
        return self._lineage_controller.dependency_status_badge_for_uid(uid)

    def dependency_status_tooltip_for_uid(self, uid: str) -> str | None:
        return self._lineage_controller.dependency_status_tooltip_for_uid(uid)

    def dependency_input_summary_for_uid(self, uid: str) -> str | None:
        return self._lineage_controller.dependency_input_summary_for_uid(uid)

    def _show_dependency_reload_dialog(self, target: int | str) -> None:
        self._lineage_controller._show_dependency_reload_dialog(target)

    @staticmethod
    def _script_input_has_recorded_file(script_input: provenance.ScriptInput) -> bool:
        return _LineageController._script_input_has_recorded_file(script_input)

    @staticmethod
    def _dependency_ref_has_recorded_file(
        spec: provenance.ToolProvenanceSpec | None,
        ref: provenance.ScriptInputDependencyRef,
    ) -> bool:
        return _LineageController._dependency_ref_has_recorded_file(spec, ref)

    def _missing_dependencies_have_recorded_file(self, uid: str) -> bool:
        return self._lineage_controller._missing_dependencies_have_recorded_file(uid)

    def _dependency_dependent_uids(self, uid: str) -> list[str]:
        return self._lineage_controller._dependency_dependent_uids(uid)

    def _refresh_dependency_dependents(self, uid: str) -> None:
        self._lineage_controller._refresh_dependency_dependents(uid)

    def _script_input_name_for_node(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> str:
        return self._lineage_controller._script_input_name_for_node(node)

    def _script_input_for_node(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> provenance.ScriptInput:
        return self._lineage_controller._script_input_for_node(node)

    def _multi_input_script_provenance(
        self,
        input_targets: Iterable[int | str],
        *,
        operation_label: str,
        operation_code: str,
        active_name: str = "derived",
        start_label: str = "Run ImageTool manager action",
    ) -> provenance.ToolProvenanceSpec:
        return self._lineage_controller._multi_input_script_provenance(
            input_targets,
            operation_label=operation_label,
            operation_code=operation_code,
            active_name=active_name,
            start_label=start_label,
        )

    def _show_multi_input_script_result(
        self,
        data: xr.DataArray,
        input_targets: Iterable[int | str],
        *,
        operation_label: str,
        operation_code: str,
    ) -> int | None:
        return self._lineage_controller._show_multi_input_script_result(
            data,
            input_targets,
            operation_label=operation_label,
            operation_code=operation_code,
        )

    def _script_provenance_inputs_current(
        self, spec: provenance.ToolProvenanceSpec
    ) -> bool:
        return self._lineage_controller._script_provenance_inputs_current(spec)

    def _resolve_live_script_input_for_reload(
        self,
        script_input: provenance.ScriptInput,
        *,
        target_node_uid: str | None = None,
    ) -> tuple[xr.DataArray, provenance.ScriptInput] | None:
        return self._lineage_controller._resolve_live_script_input_for_reload(
            script_input,
            target_node_uid=target_node_uid,
        )

    def _script_input_can_reload(
        self,
        script_input: provenance.ScriptInput,
        *,
        target_node_uid: str | None = None,
    ) -> bool:
        return self._lineage_controller._script_input_can_reload(
            script_input,
            target_node_uid=target_node_uid,
        )

    def _rebuild_script_provenance(
        self,
        spec: provenance.ToolProvenanceSpec,
        *,
        target_node_uid: str | None = None,
    ) -> _ScriptRebuildResult:
        return self._lineage_controller._rebuild_script_provenance(
            spec,
            target_node_uid=target_node_uid,
        )

    def _node_can_reload_script_inputs(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> bool:
        return self._lineage_controller._node_can_reload_script_inputs(node)

    def _script_reload_from_slicer_area(
        self,
        slicer_area: ImageSlicerArea,
        *,
        execute: bool,
    ) -> bool:
        return self._lineage_controller._script_reload_from_slicer_area(
            slicer_area,
            execute=execute,
        )

    def _workspace_loaded_uid_map(
        self, loaded_targets_by_uid: Mapping[str, int | str]
    ) -> dict[str, str]:
        return self._lineage_controller._workspace_loaded_uid_map(loaded_targets_by_uid)

    def _rebase_loaded_workspace_dependency_refs(
        self, loaded_targets_by_uid: Mapping[str, int | str]
    ) -> None:
        self._lineage_controller._rebase_loaded_workspace_dependency_refs(
            loaded_targets_by_uid
        )

    def _selected_reload_targets(
        self,
    ) -> tuple[list[int | str], dict[int | str, list[str]]] | None:
        return self._lineage_controller._selected_reload_targets()

    def _reload_target_for_child(self, uid: str) -> int | str | None:
        return self._lineage_controller._reload_target_for_child(uid)

    def _reload_source_chain_for_child(self, uid: str) -> bool:
        return self._lineage_controller._reload_source_chain_for_child(uid)

    def show_selected_source_updates(self) -> None:
        self._lineage_controller.show_selected_source_updates()

    def _child_targets_of(self, target: int | str) -> list[str]:
        return self._lineage_controller._child_targets_of(target)

    def _refresh_source_chain_to_uid(self, uid: str) -> bool:
        return self._lineage_controller._refresh_source_chain_to_uid(uid)

    def _resume_pending_source_refreshes(self, uid: str) -> None:
        self._lineage_controller._resume_pending_source_refreshes(uid)

    def _parent_source_data_for_uid(self, uid: str) -> xr.DataArray:
        return self._lineage_controller._parent_source_data_for_uid(uid)

    def _mark_descendants_source_state(
        self,
        uid: str,
        state: _ManagedWindowNode._source_state_type,
    ) -> None:
        self._lineage_controller._mark_descendants_source_state(uid, state)

    def _mark_descendants_source_unavailable(self, uid: str) -> None:
        self._lineage_controller._mark_descendants_source_unavailable(uid)

    def _propagate_source_change_from_uid(
        self, uid: str, parent_data: xr.DataArray | None = None
    ) -> None:
        self._lineage_controller._propagate_source_change_from_uid(uid, parent_data)

    def show_selected(self) -> None:
        self._lineage_controller.show_selected()

    def hide_selected(self) -> None:
        self._lineage_controller.hide_selected()

    def hide_all(self) -> None:
        self._lineage_controller.hide_all()

    def reload_selected(self) -> None:
        self._lineage_controller.reload_selected()

    @staticmethod
    def _reload_incompatibility_details(
        current: xr.DataArray, rebuilt: xr.DataArray
    ) -> str:
        return _LineageController._reload_incompatibility_details(current, rebuilt)

    def _prompt_incompatible_reload_commit(self, details: str) -> str:
        return self._lineage_controller._prompt_incompatible_reload_commit(details)

    def _replace_script_reload_target(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        result: _ScriptRebuildResult,
    ) -> None:
        self._lineage_controller._replace_script_reload_target(node, result)

    def _reload_script_derived_target(self, target: int | str) -> bool:
        return self._lineage_controller._reload_script_derived_target(target)

    def remove_selected(self) -> None:
        self._lineage_controller.remove_selected()

    def rename_selected(self) -> None:
        self._actions_controller.rename_selected()

    def duplicate_selected(self) -> None:
        self._actions_controller.duplicate_selected()

    def promote_selected(self) -> None:
        self._actions_controller.promote_selected()

    def promote_child_imagetool(self, uid: str) -> int:
        return self._actions_controller.promote_child_imagetool(uid)

    def link_selected(self, link_colors: bool = True, deselect: bool = True) -> None:
        self._actions_controller.link_selected(
            link_colors=link_colors, deselect=deselect
        )

    def unlink_selected(self, deselect: bool = True) -> None:
        self._actions_controller.unlink_selected(deselect=deselect)

    def offload_selected_to_workspace(self) -> None:
        self._actions_controller.offload_selected_to_workspace()

    def concat_selected(self) -> None:
        self._actions_controller.concat_selected()

    def batch_target_count(self) -> int:
        return self._actions_controller.batch_target_count()

    def show_batch_operations(self) -> None:
        self._actions_controller.show_batch_operations()

    def apply_batch_transform_dialog(
        self,
        dialog: typing.Any,
        launch_mode: typing.Literal["replace", "detach", "nest"],
    ) -> bool:
        return self._actions_controller.apply_batch_transform_dialog(
            dialog,
            launch_mode,
        )

    def apply_batch_filter_dialog(self, dialog: typing.Any) -> bool:
        return self._actions_controller.apply_batch_filter_dialog(dialog)

    def store_selected(self) -> None:
        self._actions_controller.store_selected()

    def unwatch_selected(self) -> None:
        self._actions_controller.unwatch_selected()

    def rename_imagetool(self, index: int, new_name: str) -> None:
        self._actions_controller.rename_imagetool(index, new_name)

    def _duplicate_subtree(
        self, target: int | str, *, parent_override: int | str | None = None
    ) -> int | str:
        return self._actions_controller._duplicate_subtree(
            target, parent_override=parent_override
        )

    def duplicate_imagetool(self, index: int | str) -> int | str:
        return self._actions_controller.duplicate_imagetool(index)

    def duplicate_childtool(self, uid: str) -> str:
        return self._actions_controller.duplicate_childtool(uid)

    def link_imagetools(self, *indices: int | str, link_colors: bool = True) -> None:
        self._actions_controller.link_imagetools(*indices, link_colors=link_colors)

    def name_of_imagetool(self, index: int) -> str:
        return self._actions_controller.name_of_imagetool(index)

    def label_of_imagetool(self, index: int) -> str:
        return self._actions_controller.label_of_imagetool(index)

    def _data_load(
        self, paths: list[str], loader_name: str, kwargs: dict[str, typing.Any]
    ) -> None:
        self._actions_controller._data_load(paths, loader_name, kwargs)

    def _data_replace(
        self, data_list: list[xr.DataArray], indices: list[int | str]
    ) -> None:
        self._actions_controller._data_replace(data_list, indices)

    def _find_watched_idx(self, uid: str) -> int | None:
        return self._actions_controller._find_watched_idx(uid)

    def _watched_source_color_key(self, wrapper: _ImageToolWrapper) -> str:
        return self._actions_controller._watched_source_color_key(wrapper)

    def color_for_watched_var_source(self, wrapper: _ImageToolWrapper) -> QtGui.QColor:
        return self._actions_controller.color_for_watched_var_source(wrapper)

    def _remove_watched(self, uid: str) -> None:
        self._actions_controller._remove_watched(uid)

    def _show_watched(self, uid: str) -> None:
        self._actions_controller._show_watched(uid)

    def _data_watched_update(
        self,
        varname: str,
        uid: str,
        darr: xr.DataArray,
        watched_metadata: Mapping[str, typing.Any] | None = None,
    ) -> None:
        self._actions_controller._data_watched_update(
            varname, uid, darr, watched_metadata
        )

    def _data_unwatch(self, uid: str) -> None:
        self._actions_controller._data_unwatch(uid)

    def _get_imagetool_data(self, index_or_uid: int | str) -> xr.DataArray | None:
        return self._actions_controller._get_imagetool_data(index_or_uid)

    def _send_imagetool_data(self, index_or_uid: int | str) -> None:
        self._actions_controller._send_imagetool_data(index_or_uid)

    def _watch_info(self) -> dict[str, typing.Any]:
        return self._actions_controller._watch_info()

    def _send_watch_info(self) -> None:
        self._actions_controller._send_watch_info()

    def ensure_console_initialized(self) -> None:
        self._actions_controller.ensure_console_initialized()

    def toggle_console(self) -> None:
        self._actions_controller.toggle_console()

    @property
    def _recent_loader_name(self) -> str | None:
        return self._actions_controller._recent_loader_name

    def ensure_explorer_initialized(self) -> None:
        self._actions_controller.ensure_explorer_initialized()

    def show_explorer(self) -> None:
        self._actions_controller.show_explorer()

    def show_ptable(self) -> None:
        self._actions_controller.show_ptable()

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent | None) -> None:
        self._actions_controller.dragEnterEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent | None) -> None:
        self._actions_controller.dropEvent(event)

    def _handle_dropped_files(self, file_paths: list[pathlib.Path]) -> None:
        self._actions_controller._handle_dropped_files(file_paths)

    def _show_loaded_info(
        self,
        loaded: list[pathlib.Path],
        canceled: list[pathlib.Path],
        failed: list[pathlib.Path],
        retry_callback: Callable[[list[pathlib.Path]], typing.Any],
    ) -> None:
        self._actions_controller._show_loaded_info(
            loaded, canceled, failed, retry_callback
        )

    def open_multiple_files(
        self, queued: list[pathlib.Path], try_workspace: bool = False
    ) -> None:
        self._actions_controller.open_multiple_files(
            queued, try_workspace=try_workspace
        )

    def _error_creating_imagetool(self) -> None:
        self._actions_controller._error_creating_imagetool()

    def _show_operation_error(self, log_message: str, text: str) -> None:
        self._actions_controller._show_operation_error(log_message, text)

    def _show_workspace_save_worker_error(self, error_text: str) -> None:
        self._actions_controller._show_workspace_save_worker_error(error_text)

    def _add_from_multiple_files(
        self,
        loaded: list[pathlib.Path],
        queued: list[pathlib.Path],
        failed: list[pathlib.Path],
        func: Callable[..., typing.Any],
        kwargs: dict[str, typing.Any],
        retry_callback: Callable[..., typing.Any],
    ) -> None:
        self._actions_controller._add_from_multiple_files(
            loaded, queued, failed, func, kwargs, retry_callback
        )

    def add_widget(self, widget: QtWidgets.QWidget) -> None:
        self._actions_controller.add_widget(widget)

    def add_childtool(
        self,
        tool: erlab.interactive.utils.ToolWindow,
        index: int | str,
        *,
        show: bool = True,
        uid: str | None = None,
        snapshot_token: str | None = None,
        created_time: datetime.datetime | str | bytes | None = None,
    ) -> str:
        return self._actions_controller.add_childtool(
            tool,
            index,
            show=show,
            uid=uid,
            snapshot_token=snapshot_token,
            created_time=created_time,
        )

    def add_figuretool(
        self,
        tool: erlab.interactive.utils.ToolWindow,
        *,
        show: bool = True,
        uid: str | None = None,
        snapshot_token: str | None = None,
        created_time: datetime.datetime | str | bytes | None = None,
    ) -> str:
        node = _ManagedWindowNode(
            self,
            self._next_node_uid(uid),
            None,
            tool,
            snapshot_token=snapshot_token,
            created_time=created_time,
        )
        if not tool._tool_display_name:
            tool._tool_display_name = self._next_figure_display_name()
        self._register_figure_node(node)
        self._mark_node_added(node.uid)
        self._sync_figures_ui(select_uid=node.uid if show else None)
        if show:
            node.show()
        return node.uid

    def add_imagetool_child(
        self,
        tool: ImageTool,
        parent: int | str,
        *,
        show: bool = True,
        activate: bool = False,
        uid: str | None = None,
        provenance_spec: provenance.ToolProvenanceSpec | None = None,
        source_spec: provenance.ToolProvenanceSpec | None = None,
        source_binding: provenance.ImageToolSelectionSourceBinding | None = None,
        source_auto_update: bool = False,
        source_state: _ManagedWindowNode._source_state_type = "fresh",
        output_id: str | None = None,
        snapshot_token: str | None = None,
        created_time: datetime.datetime | str | bytes | None = None,
    ) -> str:
        return self._actions_controller.add_imagetool_child(
            tool,
            parent,
            show=show,
            activate=activate,
            uid=uid,
            provenance_spec=provenance_spec,
            source_spec=source_spec,
            source_binding=source_binding,
            source_auto_update=source_auto_update,
            source_state=source_state,
            output_id=output_id,
            snapshot_token=snapshot_token,
            created_time=created_time,
        )

    def index_from_slicer_area(self, slicer_area: ImageSlicerArea) -> int | None:
        return self._actions_controller.index_from_slicer_area(slicer_area)

    def wrapper_from_slicer_area(
        self, slicer_area: ImageSlicerArea
    ) -> _ImageToolWrapper | None:
        return self._actions_controller.wrapper_from_slicer_area(slicer_area)

    def node_from_slicer_area(
        self, slicer_area: ImageSlicerArea
    ) -> _ImageToolWrapper | _ManagedWindowNode | None:
        return self._actions_controller.node_from_slicer_area(slicer_area)

    def target_from_slicer_area(self, slicer_area: ImageSlicerArea) -> int | str | None:
        return self._actions_controller.target_from_slicer_area(slicer_area)

    def _add_childtool_from_slicerarea(
        self,
        tool: QtWidgets.QWidget,
        parent_slicer_area: ImageSlicerArea,
    ) -> None:
        self._actions_controller._add_childtool_from_slicerarea(
            tool, parent_slicer_area
        )

    def _get_childtool_and_parent(
        self, uid: str
    ) -> tuple[erlab.interactive.utils.ToolWindow, int]:
        return self._actions_controller._get_childtool_and_parent(uid)

    def get_childtool(self, uid: str) -> erlab.interactive.utils.ToolWindow:
        return self._actions_controller.get_childtool(uid)

    def show_childtool(self, uid: str) -> None:
        self._actions_controller.show_childtool(uid)

    def _remove_childtool(self, uid: str) -> None:
        self._actions_controller._remove_childtool(uid)

    def eventFilter(
        self, obj: QtCore.QObject | None = None, event: QtCore.QEvent | None = None
    ) -> bool:
        return self._actions_controller.eventFilter(obj, event)

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
        provenance_spec: provenance.ToolProvenanceSpec | None = None,
        source_spec: provenance.ToolProvenanceSpec | None = None,
        source_binding: provenance.ImageToolSelectionSourceBinding | None = None,
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
