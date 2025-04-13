"""Jupyter console widget for ImageToolManager."""

from __future__ import annotations

__all__ = ["ToolNamespace", "ToolsNamespace", "_ImageToolManagerJupyterConsole"]

import importlib
import typing
import weakref

import numpy as np
import qtconsole.inprocess
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

import erlab

if typing.TYPE_CHECKING:
    from erlab.interactive.imagetool import ImageTool
    from erlab.interactive.imagetool.manager import ImageToolManager
    from erlab.interactive.imagetool.manager._wrapper import _ImageToolWrapper


class ToolNamespace:
    """A console interface that represents a single ImageTool object.

    In the manager console, this namespace can be accessed with the variable
    ``tools[idx]``, where ``idx`` is the index of the ImageTool to access.

    Examples
    --------
    - Access the underlying DataArray of an ImageTool:

      >>> tools[1].data

    - Setting a new DataArray:

      >>> tools[1].data = new_data

    """

    def __init__(self, wrapper: _ImageToolWrapper) -> None:
        self._wrapper_ref = weakref.ref(wrapper)

    @property
    def _wrapper(self) -> _ImageToolWrapper:
        wrapper = self._wrapper_ref()
        if wrapper:
            return wrapper
        raise LookupError("Parent was destroyed")

    @property
    def tool(self) -> ImageTool:
        """The underlying ImageTool object."""
        if self._wrapper.archived:
            self._wrapper.unarchive()
        return typing.cast("ImageTool", self._wrapper.tool)

    @property
    def data(self) -> xr.DataArray:
        """The DataArray associated with the ImageTool."""
        return self.tool.slicer_area._data

    @data.setter
    def data(self, value: xr.DataArray) -> None:
        self.tool.slicer_area.set_data(value)

    def __getattr__(self, attr):  # implicitly wrap methods from ImageToolWrapper
        if hasattr(self._wrapper, attr):
            m = getattr(self._wrapper, attr)
            if callable(m):
                return m
        raise AttributeError(attr)

    def __repr__(self) -> str:
        time_repr = self._wrapper._created_time.isoformat(sep=" ", timespec="seconds")
        out = f"ImageTool {self._wrapper.index}: {self._wrapper.name}\n"
        out += f"  Added: {time_repr}\n"
        out += f"  Archived: {self._wrapper.archived}\n"
        if not self._wrapper.archived:
            out += f"  Linked: {self.tool.slicer_area.is_linked}\n"
        return out


class ToolsNamespace:
    """A console interface that represents the ImageToolManager and its tools.

    In the manager console, this namespace can be accessed with the variable `tools`.

    Examples
    --------
    - Print the list of tools:

      >>> tools

    - Access :class:`ToolNamespace` by index:

      >>> tools[1]

    """

    def __init__(self, manager: ImageToolManager) -> None:
        self._manager_ref = weakref.ref(manager)

    @property
    def _manager(self) -> ImageToolManager:
        """Access the ImageToolManager instance."""
        manager = self._manager_ref()
        if manager:
            return manager
        raise LookupError("Parent was destroyed")

    @property
    def selected_data(self) -> list[xr.DataArray]:
        """Get a list of DataArrays from the selected windows."""
        return [
            self._manager.get_tool(idx).slicer_area._data
            for idx in self._manager.list_view.selected_tool_indices
        ]

    def __getitem__(self, index: int) -> ToolNamespace | None:
        """Access a specific ImageTool object by its index."""
        if index not in self._manager._tool_wrappers:
            print(f"Tool {index} not found")
            return None

        return ToolNamespace(self._manager._tool_wrappers[index])

    def __repr__(self) -> str:
        output = []
        for index, wrapper in self._manager._tool_wrappers.items():
            output.append(f"{index}: {wrapper.name}")
        if not output:
            return "No tools"
        return "\n".join(output)


class _JupyterConsoleWidget(qtconsole.inprocess.QtInProcessRichJupyterWidget):
    """A Jupyter console widget for ImageToolManager.

    This widget is derived from qtconsole with some modifications such as:

    - Support for dark mode

    - Custom banner text

    - Lazy kernel initialization, including lazily evaluated namespace injection

    - Automated storing of data from ImageTools with the ``%store`` magic command


    Parameters
    ----------
    parent
        The parent widget for the console.
    namespace
        A dictionary of objects to inject into the console namespace. The keys are the
        names of the objects in the console, and the values are the objects themselves.
        If the value is a string, it is imported as a module upon kernel initialization,
        improving startup time for the ImageToolManager application.

    """

    def __init__(
        self, parent=None, namespace: dict[str, typing.Any] | None = None
    ) -> None:
        super().__init__(parent)
        self.kernel_manager = qtconsole.inprocess.QtInProcessKernelManager()
        self._namespace = namespace
        self._kernel_banner_default: str = ""

    def initialize_kernel(self) -> None:
        if not self.kernel_manager.kernel:
            self.kernel_manager.start_kernel()
            self.kernel_client = self.kernel_manager.client()
            self.kernel_client.start_channels()

            self.execute(r"%load_ext storemagic", hidden=True)

            if self._namespace is not None:
                self.kernel_manager.kernel.shell.push(
                    {
                        name: importlib.import_module(module)
                        if isinstance(module, str)
                        else module
                        for name, module in self._namespace.items()
                    }
                )

    def store_data_as(self, tool_index: int, name: str) -> None:
        """Store the data in an ImageTool with IPython to reuse in other scripts."""
        self.initialize_kernel()
        store_commands = (
            f"{name} = tools[{tool_index}].data",
            f"get_ipython().run_line_magic('store', '{name}')",
            f"del {name}",
        )
        self.execute("\n".join(store_commands), hidden=True)

    @QtCore.Slot()
    def shutdown_kernel(self) -> None:
        if self.kernel_manager.kernel:
            self.kernel_client.stop_channels()
            self.kernel_manager.shutdown_kernel()

    def _banner_default(self) -> str:
        banner = super()._banner_default()
        return banner.strip() + f" | ERLabPy {erlab.__version__}\n"

    @property
    def kernel_banner(self) -> str:
        def _command_ansi(title: str, command_list: list[str]):
            out = f"\033[1m* {title}\033[0m"
            for command in command_list:
                out += f"\n  {command}"
            return out

        info_str = (
            _command_ansi("Access data", ["tools[<index>].data", "tools.selected_data"])
            + "\n"
            + _command_ansi("Change data", ["tools[<index>].data = <value>"])
            + "\n"
            + _command_ansi(
                "Control window visibility",
                ["tools[<index>].show(), .close(), .dispose()"],
            )
            + "\n"
        )

        return f"{self._kernel_banner_default}{info_str}"

    @kernel_banner.setter
    def kernel_banner(self, value: str) -> None:
        self._kernel_banner_default = value

    def _update_colors(self) -> None:
        """Detect dark mode and update the console colors accordingly."""
        if self.kernel_manager.kernel:
            is_dark: bool = (
                self.palette().color(QtGui.QPalette.ColorRole.Base).value() < 128
            )  # dark detection based on base color, adapted from pyqtgraph ReplWidget
            colors = "linux" if is_dark else "lightbg"
            self.set_default_style(colors)
            self._syntax_style_changed()
            self._style_sheet_changed()
            self._execute(
                f"""
from IPython.core.ultratb import VerboseTB
if getattr(VerboseTB, 'tb_highlight_style', None) is not None:
    VerboseTB.tb_highlight_style = '{self.syntax_style}'
elif getattr(VerboseTB, '_tb_highlight_style', None) is not None:
    VerboseTB._tb_highlight_style = '{self.syntax_style}'
else:
    get_ipython().run_line_magic('colors', '{colors}')
del VerboseTB
""",
                True,
            )  # Adapted from qtconsole.mainwindow.MainWindow.set_syntax_style

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(300, 186)


class _ImageToolManagerJupyterConsole(QtWidgets.QDockWidget):
    """A dock widget containing the Jupyter console."""

    def __init__(self, manager: ImageToolManager) -> None:
        super().__init__("Console", manager, flags=QtCore.Qt.WindowType.Window)

        self._console_widget = _JupyterConsoleWidget(
            parent=self,
            namespace={
                "np": np,
                "xr": xr,
                "erlab": erlab,
                "eri": erlab.interactive,
                "tools": ToolsNamespace(manager),
                "era": "erlab.analysis",
                "eplt": "erlab.plotting",
                "plt": "matplotlib.pyplot",
            },
        )
        qapp = QtWidgets.QApplication.instance()

        if qapp:
            # Shutdown kernel when application quits
            qapp.aboutToQuit.connect(self._console_widget.shutdown_kernel)

        self.setWidget(self._console_widget)
        manager.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self)
        self.setFloating(False)
        self.hide()

        # Start kernel when console is shown
        self._console_widget.installEventFilter(self)

    def eventFilter(
        self, obj: QtCore.QObject | None = None, event: QtCore.QEvent | None = None
    ) -> bool:
        if (
            hasattr(self, "_console_widget")
            and obj == self._console_widget
            and event is not None
            and event.type() == QtCore.QEvent.Type.Show
        ):
            self._console_widget.initialize_kernel()
            self._console_widget._update_colors()
        return super().eventFilter(obj, event)

    def changeEvent(self, evt: QtCore.QEvent | None) -> None:
        if evt is not None and evt.type() == QtCore.QEvent.Type.PaletteChange:
            self._console_widget._update_colors()

        super().changeEvent(evt)
