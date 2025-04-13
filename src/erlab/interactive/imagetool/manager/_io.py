"""Helper classes for loading multiple files sequentially in a separate thread."""

from __future__ import annotations

__all__ = ["_MultiFileHandler"]

import collections
import logging
import pathlib
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

    @QtCore.Slot()
    def run(self) -> None:
        try:
            data_list: list[xr.DataArray] = (
                erlab.interactive.imagetool.core._parse_input(
                    self._func(self._file_path, **self._kwargs)
                )
            )
        except Exception as e:
            logger.exception("Error loading data from %s", self._file_path)
            self.signals.sigFailed.emit(self._file_path, f"{type(e).__name__}: {e}")
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
    sigFinished()
        Emitted when the loading process has finished. The signal is emitted even if the
        loading process was aborted due to an error.

    """

    sigFinished = QtCore.Signal()  #: :meta private:

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

    @property
    def _threadpool(self) -> QtCore.QThreadPool:
        return typing.cast("QtCore.QThreadPool", QtCore.QThreadPool.globalInstance())

    @property
    def manager(self) -> ImageToolManager:
        """Access the parent manager instance."""
        _manager = self._manager()
        if _manager is None:
            raise LookupError("Parent was destroyed")
        return _manager

    @property
    def queued(self) -> list[pathlib.Path]:
        """List of files that are yet to be loaded."""
        return list(self._queue)

    def start(self) -> None:
        """Initiate the loading process.

        This method should be only called once.
        """
        self._load_next()

    def _load_next(self) -> None:
        """Load the next file in the queue."""
        if len(self._queue) == 0:
            self.sigFinished.emit()
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
        self.manager._data_recv(
            data_list,
            kwargs={"file_path": file_path, "_disable_reload": len(data_list) > 1},
        )
        self.loaded.append(file_path)
        self.manager._recent_directory = str(file_path.parent)
        self._load_next()

    @QtCore.Slot(pathlib.Path, str)
    def _on_failed(self, file_path: pathlib.Path, exc_str: str) -> None:
        self.manager._status_bar.showMessage("")
        self.failed.append(file_path)

        msg_box = QtWidgets.QMessageBox(self.manager)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setText(f"Failed to load {file_path.name}")
        msg_box.setInformativeText(
            "Do you want to skip this file and continue loading?"
        )
        msg_box.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Abort
            | QtWidgets.QMessageBox.StandardButton.Yes
        )
        msg_box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Yes)
        msg_box.setDetailedText(exc_str)
        match msg_box.exec():
            case QtWidgets.QMessageBox.StandardButton.Yes:
                self._load_next()
            case QtWidgets.QMessageBox.StandardButton.Abort:
                self.sigFinished.emit()
