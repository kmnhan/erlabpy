from __future__ import annotations

import ctypes
import ctypes.util
import logging
import pathlib
import sys
import typing

from qtpy import QtWidgets

logger = logging.getLogger(__name__)

APP_USER_MODEL_ID = "dev.kmnhan.erlabpy.imagetoolmanager"
OPEN_WORKSPACE_DIALOG_ARG = "--open-workspace-dialog"
NEW_MANAGER_WINDOW_ARG = "--new-manager-window"

_SHARD_PATHW = 0x00000003
_KDC_RECENT = 2

_WINDOWS_JUMP_LIST_TASKS = (
    (
        "Open Workspace…",
        (OPEN_WORKSPACE_DIALOG_ARG,),
        "Open an ImageTool workspace file",
    ),
    (
        "New Manager Window",
        (NEW_MANAGER_WINDOW_ARG,),
        "Open another ImageTool Manager window",
    ),
)

if typing.TYPE_CHECKING:
    import os


def configure_process() -> None:
    """Install process-level desktop integration for packaged manager launches."""
    if not sys.platform.startswith("win"):
        return
    set_windows_app_user_model_id()
    install_windows_jump_list()


def record_recent_workspace(fname: str | os.PathLike[str]) -> None:
    """Record a workspace with the platform's shell recent-document list."""
    path = pathlib.Path(fname).expanduser().resolve()
    if path.suffix.lower() != ".itws":
        return

    try:
        if sys.platform == "darwin":
            _record_macos_recent_document(path)
        elif sys.platform.startswith("win"):
            shell32 = typing.cast("typing.Any", ctypes).windll.shell32
            shell32.SHAddToRecentDocs(_SHARD_PATHW, ctypes.c_wchar_p(str(path)))
    except Exception:
        logger.debug("Could not record workspace with desktop shell", exc_info=True)


def install_macos_dock_menu(manager: QtWidgets.QWidget) -> QtWidgets.QMenu | None:
    """Install running-app macOS Dock actions for the manager."""
    if sys.platform != "darwin":
        return None

    dock_menu = QtWidgets.QMenu(manager)
    dock_menu.setObjectName("manager_macos_dock_menu")

    open_action = QtWidgets.QAction("Open Workspace…", dock_menu)
    open_action.setObjectName("manager_dock_open_workspace_action")
    open_action.triggered.connect(typing.cast("typing.Any", manager).load)
    dock_menu.addAction(open_action)

    new_manager_action = QtWidgets.QAction("New Manager Window", dock_menu)
    new_manager_action.setObjectName("manager_dock_new_manager_action")
    new_manager_action.triggered.connect(
        typing.cast("typing.Any", manager).open_new_manager_instance
    )
    dock_menu.addAction(new_manager_action)

    try:
        dock_menu.setAsDockMenu()
    except Exception:
        logger.debug("Could not install macOS Dock menu", exc_info=True)
        dock_menu.deleteLater()
        return None

    # Keep the menu alive; Qt hands a native delegate to the Dock.
    typing.cast("typing.Any", manager)._macos_dock_menu = dock_menu
    return dock_menu


def set_windows_app_user_model_id(app_id: str = APP_USER_MODEL_ID) -> None:
    """Set the process AppUserModelID used for taskbar grouping."""
    if not sys.platform.startswith("win"):
        return
    try:
        shell32 = typing.cast("typing.Any", ctypes).windll.shell32
        result = shell32.SetCurrentProcessExplicitAppUserModelID(
            ctypes.c_wchar_p(app_id)
        )
    except Exception:
        logger.debug("Could not set Windows AppUserModelID", exc_info=True)
        return
    if result:
        logger.debug("SetCurrentProcessExplicitAppUserModelID returned %s", result)


def install_windows_jump_list() -> None:
    """Install Windows Jump List tasks for packaged manager launches."""
    if not sys.platform.startswith("win"):
        return
    try:
        import pythoncom
        from win32com.propsys import propsys, pscon
        from win32com.shell import shell
    except ImportError:
        logger.debug("pywin32 is unavailable; skipping Windows Jump List tasks")
        return

    exe_path = pathlib.Path(sys.executable)
    try:
        destination_list = pythoncom.CoCreateInstance(
            shell.CLSID_DestinationList,
            None,
            pythoncom.CLSCTX_INPROC_SERVER,
            shell.IID_ICustomDestinationList,
        )
        if hasattr(destination_list, "SetAppID"):
            destination_list.SetAppID(APP_USER_MODEL_ID)
        destination_list.BeginList()
        destination_list.AppendKnownCategory(_KDC_RECENT)

        collection = pythoncom.CoCreateInstance(
            shell.CLSID_EnumerableObjectCollection,
            None,
            pythoncom.CLSCTX_INPROC_SERVER,
            shell.IID_IObjectCollection,
        )
        for title, arguments, description in _WINDOWS_JUMP_LIST_TASKS:
            link = pythoncom.CoCreateInstance(
                shell.CLSID_ShellLink,
                None,
                pythoncom.CLSCTX_INPROC_SERVER,
                shell.IID_IShellLink,
            )
            link.SetPath(str(exe_path))
            link.SetArguments(" ".join(arguments))
            link.SetDescription(description)
            link.SetIconLocation(str(exe_path), 0)
            properties = link.QueryInterface(propsys.IID_IPropertyStore)
            properties.SetValue(pscon.PKEY_Title, propsys.PROPVARIANTType(title))
            properties.Commit()
            collection.AddObject(link)
        destination_list.AddUserTasks(collection)
        destination_list.CommitList()
    except Exception:
        logger.debug("Could not install Windows Jump List tasks", exc_info=True)


def _record_macos_recent_document(path: pathlib.Path) -> None:
    try:
        from AppKit import NSDocumentController  # type: ignore[import-not-found]
        from Foundation import NSURL  # type: ignore[import-not-found]
    except ImportError:
        _record_macos_recent_document_ctypes(path)
        return

    url = NSURL.fileURLWithPath_(str(path))
    NSDocumentController.sharedDocumentController().noteNewRecentDocumentURL_(url)


# Last-resort bridge for macOS builds without PyObjC; exercised manually on macOS.
def _record_macos_recent_document_ctypes(
    path: pathlib.Path,
) -> None:  # pragma: no cover
    objc_path = ctypes.util.find_library("objc")
    appkit_path = ctypes.util.find_library("AppKit")
    foundation_path = ctypes.util.find_library("Foundation")
    if objc_path is None or appkit_path is None or foundation_path is None:
        raise RuntimeError("Objective-C runtime libraries are unavailable")

    ctypes.cdll.LoadLibrary(foundation_path)
    ctypes.cdll.LoadLibrary(appkit_path)
    objc = ctypes.cdll.LoadLibrary(objc_path)
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    objc.objc_msgSend.restype = ctypes.c_void_p

    def objc_class(name: bytes) -> int:
        cls = objc.objc_getClass(name)
        if not cls:
            raise RuntimeError(f"Objective-C class unavailable: {name!r}")
        return typing.cast("int", cls)

    def objc_selector(name: bytes) -> int:
        selector = objc.sel_registerName(name)
        if not selector:
            raise RuntimeError(f"Objective-C selector unavailable: {name!r}")
        return typing.cast("int", selector)

    def send_id(
        receiver: int,
        selector: bytes,
        *args,
        argtypes: list[typing.Any] | None = None,
    ) -> int:
        objc.objc_msgSend.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            *(argtypes or [ctypes.c_void_p] * len(args)),
        ]
        result = objc.objc_msgSend(receiver, objc_selector(selector), *args)
        if not result:
            raise RuntimeError(f"Objective-C call returned nil: {selector!r}")
        return typing.cast("int", result)

    ns_path = send_id(
        objc_class(b"NSString"),
        b"stringWithUTF8String:",
        path.as_posix().encode(),
        argtypes=[ctypes.c_char_p],
    )
    url = send_id(objc_class(b"NSURL"), b"fileURLWithPath:", ns_path)
    controller = send_id(
        objc_class(b"NSDocumentController"), b"sharedDocumentController"
    )

    objc.objc_msgSend.restype = None
    objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    objc.objc_msgSend(
        controller,
        objc_selector(b"noteNewRecentDocumentURL:"),
        url,
    )
