"""Manager for multiple ImageTool windows.

.. image:: ../images/manager_light.png
    :align: center
    :alt: ImageToolManager window screenshot
    :class: only-light
    :width: 600px

.. only:: format_html

    .. image:: ../images/manager_dark.png
        :align: center
        :alt: ImageToolManager window screenshot
        :class: only-dark
        :width: 600px

This module provides a GUI application for managing multiple ImageTool windows. The
application can be started by running the script `itool-manager` from the command line
in the environment where the package is installed.

Python scripts communicate with managers using ZeroMQ connections. The first manager
uses the default request port 45555 and watch port 45556 when those ports are free;
additional managers use dynamically assigned ports and can be selected by 0-based
manager index.

"""

from __future__ import annotations

__all__ = [
    "HOST_IP",
    "PORT",
    "PORT_WATCH",
    "ImageToolManager",
    "ImageToolManagerAmbiguousError",
    "ImageToolManagerHandle",
    "ImageToolManagerNotFoundError",
    "ImageToolManagerRegistry",
    "clear_default_manager",
    "fetch",
    "get_default_manager",
    "get_log_file_path",
    "is_running",
    "load_in_manager",
    "main",
    "manager_selection_info",
    "managers",
    "maybe_push",
    "replace_data",
    "set_default_manager",
    "show_in_manager",
    "shutdown",
    "use_manager",
    "watch",
    "watch_info",
    "watched_variables",
]


import logging
import os
import pathlib
import shutil
import sys
import typing
import warnings
from dataclasses import dataclass

from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive.imagetool.manager import _desktop
from erlab.interactive.imagetool.manager._logging import (
    configure_logging,
    get_log_file_path,
)
from erlab.interactive.imagetool.manager._mainwindow import ImageToolManager
from erlab.interactive.imagetool.manager._server import (
    HOST_IP,
    PORT,
    PORT_WATCH,
    ImageToolManagerAmbiguousError,
    ImageToolManagerHandle,
    ImageToolManagerNotFoundError,
    ImageToolManagerRegistry,
    _load_in_manager_startup,
    _unwatch_data,
    _watch_data,
    clear_default_manager,
    fetch,
    get_default_manager,
    is_running,
    load_in_manager,
    manager_selection_info,
    managers,
    replace_data,
    set_default_manager,
    show_in_manager,
    use_manager,
    watch_info,
)
from erlab.interactive.imagetool.manager._watcher import (
    maybe_push,
    shutdown,
    watch,
    watched_variables,
)
from erlab.interactive.imagetool.manager._widgets import _ICON_PATH

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from collections.abc import Iterable


_manager_instance: ImageToolManager | None = None
"""Reference to the running manager instance."""

_always_use_socket: bool = False
"""Internal flag to use sockets within same process for test coverage."""

_STARTUP_FORWARD_TIMEOUT_MS = 1500
_STARTUP_TARGET_NEW: typing.Literal["new"] = "new"
_STARTUP_TARGET_CANCEL: typing.Literal["cancel"] = "cancel"


def _runtime_window_icon() -> QtGui.QIcon:
    # Avoid QTBUG-147602 crashes in QImage::toCGImage() color-space handling.
    image = QtGui.QImage(_ICON_PATH).convertToFormat(
        QtGui.QImage.Format.Format_ARGB32_Premultiplied
    )
    image.setColorSpace(QtGui.QColorSpace())
    return QtGui.QIcon(QtGui.QPixmap.fromImage(image))


@dataclass(frozen=True)
class _StartupArgs:
    files: list[pathlib.Path]
    open_workspace_dialog: bool = False
    force_new_manager: bool = False


def watch_data(varname: str, uid: str, data, show: bool = False) -> None:
    """Compatibility wrapper for the internal watch transport API.

    .. deprecated:: 3.20.0
       Use :func:`watch` instead.
    """
    warnings.warn(
        "`watch_data` is deprecated and will become private. "
        "Use `erlab.interactive.imagetool.manager.watch` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    _watch_data(varname, uid, data, show=show)


def unwatch_data(uid: str, remove: bool = False):
    """Compatibility wrapper for the internal unwatch transport API.

    .. deprecated:: 3.20.0
       Use :func:`watch` with ``stop`` options instead.
    """
    warnings.warn(
        "`unwatch_data` is deprecated and will become private. "
        "Use `erlab.interactive.imagetool.manager.watch(..., stop=True)` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _unwatch_data(uid, remove=remove)


class _ManagerApp(QtWidgets.QApplication):
    def __init__(self, argv: list[str]) -> None:
        super().__init__(argv)
        self._pending_files: list[pathlib.Path] = []

    def event(self, e: QtCore.QEvent | None) -> bool:  # pragma: no cover
        if e and e.type() == QtCore.QEvent.Type.FileOpen:
            # Happens both at first launch (before your window shows)
            # and when the app is already running

            file_event = typing.cast("QtGui.QFileOpenEvent", e)
            if file_event.url().isLocalFile():
                self._handle_open_file(pathlib.Path(file_event.url().toLocalFile()))
                return True
        return super().event(e)

    def _handle_open_file(self, path: pathlib.Path) -> None:  # pragma: no cover
        if _manager_instance:
            _manager_instance._handle_dropped_files([path])
        else:
            self._pending_files.append(path)


def _get_updater_settings() -> QtCore.QSettings:  # pragma: no cover
    """Get the QSettings object for the updater."""
    return QtCore.QSettings(
        QtCore.QSettings.Format.IniFormat,
        QtCore.QSettings.Scope.UserScope,
        "erlabpy",
        "imagetool-manager-updater",
    )


def _cleanup_update_tmp_dirs(settings) -> None:
    """Clean up any leftover temporary update directories from previous runs."""
    tmp_dirs = settings.value("update_tmp_dirs", "")
    tmp_dir_list = tmp_dirs.strip().split(",") if tmp_dirs else []
    for d in tmp_dir_list:
        p = pathlib.Path(d)
        if p.exists() and p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
    settings.setValue("update_tmp_dirs", "")


def _parse_startup_args(args: Iterable[str]) -> _StartupArgs:
    file_args: list[pathlib.Path] = []
    open_workspace_dialog = False
    force_new_manager = False

    for arg in args:
        if arg == _desktop.OPEN_WORKSPACE_DIALOG_ARG:
            open_workspace_dialog = True
            continue
        if arg == _desktop.NEW_MANAGER_WINDOW_ARG:
            force_new_manager = True
            continue
        if arg.startswith("-"):
            continue
        path = pathlib.Path(arg)
        if path.exists():
            file_args.append(path)

    return _StartupArgs(
        files=file_args,
        open_workspace_dialog=open_workspace_dialog,
        force_new_manager=force_new_manager,
    )


def _startup_path_is_workspace(path: pathlib.Path) -> bool:
    suffix = path.suffix.lower()
    if suffix == ".itws":
        return True
    if suffix != ".h5":
        return False
    try:
        import h5py

        with h5py.File(path, "r") as h5_file:
            return (
                "imagetool_workspace_schema_version" in h5_file.attrs
                or h5_file.attrs.get("is_itool_workspace", 0) == 1
            )
    except Exception:
        return False


def _startup_pending_files(
    file_open_event_files: list[pathlib.Path], startup_files: list[pathlib.Path]
) -> list[pathlib.Path]:
    pending: list[pathlib.Path] = []
    seen: set[pathlib.Path] = set()
    for path in [*file_open_event_files, *startup_files]:
        try:
            key = path.resolve()
        except OSError:
            key = path
        if key in seen:
            continue
        pending.append(path)
        seen.add(key)
    return pending


def _choose_startup_manager_target(
    selection_info: dict[str, object],
    parent: QtWidgets.QWidget | None = None,
) -> int | typing.Literal["new", "cancel"]:
    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle("Open Files")

    layout = QtWidgets.QVBoxLayout(dialog)
    layout.addWidget(
        QtWidgets.QLabel(
            "Choose where to open these files. Select an existing manager to add "
            "them there, or open a new manager window."
        )
    )

    target_list = QtWidgets.QListWidget(dialog)
    target_list.setObjectName("manager_startup_target_list")
    target_list.setSelectionMode(
        QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
    )
    layout.addWidget(target_list)

    new_item = QtWidgets.QListWidgetItem("New Manager")
    new_item.setData(QtCore.Qt.ItemDataRole.UserRole, _STARTUP_TARGET_NEW)
    target_list.addItem(new_item)

    managers_info = typing.cast("list[dict[str, object]]", selection_info["managers"])
    for record in managers_info:
        index = typing.cast("int", record["index"])
        workspace_path = record.get("workspace_path")
        label = (
            f"Manager #{index} - {pathlib.Path(str(workspace_path)).name}"
            if workspace_path
            else f"Manager #{index} - Unsaved workspace"
        )
        tooltip_parts = [
            f"Manager #{index}",
            f"Endpoint: {record.get('host')}:{record.get('port')}",
            f"Process ID: {record.get('pid')}",
        ]
        if workspace_path:
            tooltip_parts.append(f"Workspace: {workspace_path}")

        item = QtWidgets.QListWidgetItem(label)
        item.setToolTip("\n".join(tooltip_parts))
        item.setData(QtCore.Qt.ItemDataRole.UserRole, index)
        target_list.addItem(item)

    target_list.setCurrentRow(0)

    buttons = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.StandardButton.Open
        | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
        parent=dialog,
    )
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    target_list.itemDoubleClicked.connect(lambda _item: dialog.accept())
    layout.addWidget(buttons)

    if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return _STARTUP_TARGET_CANCEL
    current_item = target_list.currentItem()
    if current_item is None:
        return _STARTUP_TARGET_CANCEL
    target = current_item.data(QtCore.Qt.ItemDataRole.UserRole)
    if target == _STARTUP_TARGET_NEW:
        return _STARTUP_TARGET_NEW
    return int(target)


def _try_forward_startup_files(
    paths: list[pathlib.Path],
    *,
    parent: QtWidgets.QWidget | None = None,
) -> bool:
    if not paths:
        return False
    if any(_startup_path_is_workspace(path) for path in paths):
        return False

    try:
        selection_info = manager_selection_info(
            lock_timeout_ms=_STARTUP_FORWARD_TIMEOUT_MS
        )
        resolved_index = selection_info.get("resolved_index")
        if resolved_index is not None:
            target: int | typing.Literal["new", "cancel"] = typing.cast(
                "int", resolved_index
            )
        elif selection_info.get("needs_selection"):
            target = _choose_startup_manager_target(selection_info, parent=parent)
        else:
            return False
    except Exception:
        logger.debug("Could not inspect live managers for startup file forwarding")
        return False

    if target == _STARTUP_TARGET_CANCEL:
        return True
    if target == _STARTUP_TARGET_NEW:
        return False

    try:
        _load_in_manager_startup(
            paths,
            target=target,
            timeout_ms=_STARTUP_FORWARD_TIMEOUT_MS,
        )
    except Exception:
        logger.info(
            "Could not forward startup files to ImageTool manager #%s",
            target,
            exc_info=True,
            extra={"suppress_ui_alert": True},
        )
        return False
    return True


def main(execute: bool = True) -> None:
    """Start the ImageToolManager application.

    Running ``itool-manager`` from a shell will invoke this function.
    """
    global _manager_instance

    startup_args = _parse_startup_args(sys.argv[1:])
    # Files passed as command-line arguments also handle opening files on Windows.

    if erlab.utils.misc._IS_PACKAGED:  # pragma: no cover
        _desktop.configure_process()

    qapp = typing.cast(
        "QtWidgets.QApplication | None", QtWidgets.QApplication.instance()
    )
    startup_arg_files: list[pathlib.Path] = []
    if not qapp:
        qapp = _ManagerApp(sys.argv)
        qapp.setStyle("Fusion")
        qapp.setAttribute(QtCore.Qt.ApplicationAttribute.AA_DontShowIconsInMenus, False)
        startup_arg_files = startup_args.files

    if not qapp.property("_erlab_itool_manager_configured") and (
        sys.platform != "darwin" or not erlab.utils.misc._IS_PACKAGED
    ):
        # Ignore if running in a PyInstaller bundle on macOS.
        qapp.setWindowIcon(_runtime_window_icon())
        qapp.setApplicationName("ImageTool Manager")
        qapp.setApplicationDisplayName("ImageTool Manager")
        qapp.setApplicationVersion(erlab.__version__)
        qapp.setProperty("_erlab_itool_manager_configured", True)

    configure_logging()

    if isinstance(qapp, _ManagerApp) and _manager_instance is None:
        qapp.processEvents()
    file_open_event_files = (
        qapp._pending_files.copy() if isinstance(qapp, _ManagerApp) else []
    )
    startup_files = _startup_pending_files(file_open_event_files, startup_arg_files)
    if (
        startup_files
        and not startup_args.force_new_manager
        and _try_forward_startup_files(startup_files)
    ):
        if isinstance(qapp, _ManagerApp):
            qapp._pending_files.clear()
        return

    _manager_instance = ImageToolManager()
    _manager_instance.show()
    _manager_instance.activateWindow()

    if erlab.utils.misc._IS_PACKAGED:  # pragma: no cover
        _desktop.install_macos_dock_menu(_manager_instance)

    if startup_files:  # pragma: no cover
        _manager_instance._handle_dropped_files(startup_files)
        if isinstance(qapp, _ManagerApp):
            qapp._pending_files.clear()

    if startup_args.open_workspace_dialog:
        QtCore.QTimer.singleShot(0, _manager_instance.load)

    if erlab.utils.misc._IS_PACKAGED:  # pragma: no cover
        # Handle cleanup after a successful application update
        updater_settings = _get_updater_settings()
        new_version = str(erlab.__version__)
        old_version = updater_settings.value("version_before_update", "")
        if old_version != new_version:
            _manager_instance.updated(old_version, new_version)
            _cleanup_update_tmp_dirs(updater_settings)
            updater_settings.setValue("version_before_update", new_version)
            updater_settings.sync()

        # Suppress warnings on console initialization
        os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

    if execute:
        qapp.exec()
        _manager_instance = None


def _get_recent_directory() -> str:
    """Return the most recent directory used by the ImageToolManager.

    Used internally to set the default directory for various file dialogs in tools
    launched inside the manager. Returns an empty string if no directory has been set
    yet, or if the manager is running in a different process.

    """
    if (
        _manager_instance is not None
        and _manager_instance._recent_directory is not None
    ):
        return str(_manager_instance._recent_directory)
    return ""
