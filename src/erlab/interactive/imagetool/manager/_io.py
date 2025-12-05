"""Helper classes for loading multiple files sequentially in a separate thread."""

from __future__ import annotations

__all__ = ["_MultiFileHandler"]

import collections
import contextlib
import logging
import pathlib
import traceback
import typing
import weakref

from qtpy import QtCore, QtWidgets

import erlab

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from collections.abc import Callable

    import xarray as xr

    from erlab.interactive.imagetool.manager import ImageToolManager


class _DataLoaderSignals(QtCore.QObject):
    sigLoaded = QtCore.Signal(pathlib.Path, list)
    sigFailed = QtCore.Signal(pathlib.Path, str)


class _DataLoader(QtCore.QRunnable):
    """Load data from a file and emit signals when the loading is complete.

    - If the loading process is successful, the signal ``sigLoaded`` is emitted with the
      file path and the loaded data.

    - If the loading process fails, the signal ``sigFailed`` is emitted with the file
      path and the formatted exception message string.

    Parameters
    ----------
    file_path
        Path to the file to be loaded.
    func
        The function to be called for loading the file.The function should accept the
        file path as the first positional argument.
    kwargs
        Additional keyword arguments to be passed to ``func``.

    """

    def __init__(
        self, file_path: pathlib.Path, func: Callable, kwargs: dict[str, typing.Any]
    ):
        super().__init__()
        self.signals: _DataLoaderSignals = _DataLoaderSignals()

        self._file_path = file_path
        self._func = func
        self._kwargs = kwargs

    def run(self) -> None:
        try:
            data_list: list[xr.DataArray] = (
                erlab.interactive.imagetool.core._parse_input(
                    self._func(self._file_path, **self._kwargs)
                )
            )
        except Exception:
            logger.debug(
                "Error loading data from %s", self._file_path
            )  # Use debug level to avoid duplicate popup
            self.signals.sigFailed.emit(self._file_path, traceback.format_exc())
        else:
            self.signals.sigLoaded.emit(self._file_path, data_list)


class _MultiFileHandler(QtCore.QObject):
    """Manage the loading of multiple files in a separate thread.

    Parameters
    ----------
    manager
        The manager instance.
    file_list
        List of file paths to be loaded.
    func
        The function to be called for loading each file.
    kwargs
        Additional keyword arguments to be passed to ``func``.

    Attributes
    ----------
    loaded
        List of successfully loaded files.
    failed
        List of files that failed to load.

    Signals
    -------
    sigFinished(loaded, aborted, failed)
        Emitted when the loading process has finished. The signal is emitted even if the
        loading process was stopped prematurely using the ``abort`` method or due to an
        unexpected error.

    """

    sigFinished = QtCore.Signal(object, object, object)  #: :meta private:

    def __init__(
        self,
        manager: ImageToolManager,
        file_list: list[pathlib.Path],
        func: Callable,
        kwargs: dict[str, typing.Any],
    ):
        super().__init__(manager)

        self._manager = weakref.ref(manager)
        self._queue: collections.deque[pathlib.Path] = collections.deque(file_list)
        self._func = func
        self._kwargs = kwargs

        self.loaded: list[pathlib.Path] = []
        self.failed: list[pathlib.Path] = []

        self._abort: bool = False

        self._threadpool = QtCore.QThreadPool(self)
        self._threadpool.setExpiryTimeout(0)

    @property
    def manager(self) -> ImageToolManager:
        """Access the parent manager instance."""
        manager = self._manager()
        if manager is None:
            raise LookupError("Parent was destroyed")
        return manager

    @property
    def queued(self) -> list[pathlib.Path]:
        """List of files that are yet to be loaded."""
        return list(self._queue)

    def start(self) -> None:
        """Initiate the loading process.

        This method should be only called once.
        """
        self._load_next()

    @QtCore.Slot()
    def _load_next(self) -> None:
        """Load the next file in the queue."""
        if len(self._queue) == 0 or self._abort:
            self.sigFinished.emit(self.loaded, self.queued, self.failed)
            return

        file_path = self._queue.popleft()

        self.manager._status_bar.showMessage(f"Loading {file_path.name}...")

        loader = _DataLoader(file_path, self._func, self._kwargs)
        loader.signals.sigLoaded.connect(self._on_loaded)
        loader.signals.sigFailed.connect(self._on_failed)
        self._threadpool.start(loader)

    @QtCore.Slot(pathlib.Path, list)
    def _on_loaded(
        self, file_path: pathlib.Path, data_list: list[xr.DataArray]
    ) -> None:
        self.manager._status_bar.showMessage("")
        self.loaded.append(file_path)
        self.manager._recent_directory = str(file_path.parent)
        QtCore.QTimer.singleShot(
            0, lambda: self._deliver_and_queue(file_path, data_list)
        )

    def _deliver_and_queue(
        self, file_path: pathlib.Path, data_list: list[xr.DataArray]
    ) -> None:
        func: Callable | str = self._func
        func_instance = getattr(func, "__self__", None)
        if isinstance(func_instance, erlab.io.dataloader.LoaderBase):
            func = func_instance.name

        self.manager._data_recv(
            data_list,
            kwargs={"file_path": file_path, "load_func": (func, self._kwargs.copy())},
        )
        QtCore.QTimer.singleShot(0, self._load_next)

    @QtCore.Slot(pathlib.Path, str)
    def _on_failed(self, file_path: pathlib.Path, exc_str: str) -> None:
        self.failed.append(file_path)
        self.manager._status_bar.showMessage("")

        dialog = erlab.interactive.utils.MessageDialog(
            self.manager,
            text=f"Failed to load {file_path.name}",
            informative_text="Do you want to skip this file and continue loading?",
            buttons=QtWidgets.QDialogButtonBox.StandardButton.Abort
            | QtWidgets.QDialogButtonBox.StandardButton.Yes,
            default_button=QtWidgets.QDialogButtonBox.StandardButton.Yes,
        )
        dialog.setDetailedText(erlab.interactive.utils._format_traceback(exc_str))
        dialog.adjustSize()
        match dialog.exec():
            case QtWidgets.QDialog.DialogCode.Accepted:
                self._load_next()
            case _:
                self.sigFinished.emit(self.loaded, self.queued, self.failed)

    def abort(self) -> None:
        """Abort the loading process.

        Files that are already being loaded will be completed, but no further files
        will be loaded.
        """
        self._abort = True
        self._queue.clear()
        if hasattr(self._threadpool, "clear"):
            self._threadpool.clear()

    def wait(self) -> None:
        """Block until all files are loaded or the loading process is aborted."""
        if hasattr(self._threadpool, "waitForDone"):  # Qt 6.8+
            self._threadpool.waitForDone()
        else:  # pragma: no cover
            with contextlib.suppress(KeyboardInterrupt):
                while self._threadpool.activeThreadCount() > 0:
                    QtCore.QThread.msleep(100)
