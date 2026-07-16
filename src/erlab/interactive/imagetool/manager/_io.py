"""File selection, external data ingestion, and sequential file loading."""

from __future__ import annotations

__all__ = ["_DataIngressController", "_MultiFileHandler"]

import collections
import collections.abc
import contextlib
import logging
import os
import pathlib
import traceback
import typing
import weakref

import xarray as xr
from qtpy import QtCore, QtWidgets

import erlab
from erlab.interactive.imagetool._mainwindow import ImageTool
from erlab.interactive.imagetool._provenance._model import FileDataSelection
from erlab.interactive.imagetool.manager._dialogs import _is_loader_func

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from erlab.interactive.explorer._tabbed_explorer import _TabbedExplorer
    from erlab.interactive.imagetool._provenance._model import ToolProvenanceOperation
    from erlab.interactive.imagetool.manager._mainwindow import ImageToolManager


class _DataIngressController:
    """Open external data and construct manager-owned ImageTool windows."""

    def __init__(self, manager: ImageToolManager) -> None:
        self._manager = manager

    def open(self, *, native: bool = True) -> None:
        """Prompt for one or more data files and queue them for loading."""
        dialog = QtWidgets.QFileDialog(self._manager)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptOpen)
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFiles)
        valid_loaders: dict[str, tuple[Callable, dict[str, typing.Any]]] = (
            erlab.interactive.utils.file_loaders()
        )
        dialog.setNameFilters(valid_loaders.keys())
        if not native:
            dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)

        preferred_name_filter = self._manager._preferred_name_filter(valid_loaders)
        if preferred_name_filter is not None:
            dialog.selectNameFilter(preferred_name_filter)
        if (directory := self._manager._recent_or_default_directory()) is not None:
            dialog.setDirectory(directory)

        if not dialog.exec():
            return

        file_names = dialog.selectedFiles()
        self._manager._recent_name_filter = dialog.selectedNameFilter()
        self._manager._recent_directory = os.path.dirname(file_names[0])
        func, kwargs = valid_loaders[self._manager._recent_name_filter]
        if _is_loader_func(func):
            selected = self._manager._select_loader_options(
                {self._manager._recent_name_filter: (func, kwargs)},
                self._manager._recent_name_filter,
                sample_paths=file_names,
            )
            if selected is None:
                return
            self._manager._recent_name_filter, func, kwargs = selected
        self.add_from_multiple_files(
            loaded=[],
            queued=[pathlib.Path(path) for path in file_names],
            failed=[],
            func=func,
            kwargs=kwargs,
            retry_callback=lambda _: self.open(native=native),
        )

    def receive_data(
        self,
        data: list[xr.DataArray] | list[xr.Dataset],
        kwargs: dict[str, typing.Any],
        *,
        watched_var: tuple[str, str] | None = None,
        watched_metadata: Mapping[str, typing.Any] | None = None,
        show: bool | None = None,
    ) -> list[bool]:
        """Construct manager-owned ImageTools from received arrays or datasets."""
        flags: list[bool] = []
        if erlab.utils.misc.is_sequence_of(data, xr.Dataset):
            for ds in data:
                try:
                    self._manager.add_imagetool(
                        ImageTool.from_dataset(
                            ds,
                            _in_manager=True,
                            options_model=self._manager.effective_interactive_options,
                        ),
                        activate=True,
                    )
                except Exception:
                    flags.append(False)
                    logger.exception(
                        "Error creating ImageTool window",
                        extra={"suppress_ui_alert": True},
                    )
                    self._error_creating_imagetool()
                else:
                    flags.append(True)
            return flags

        try:
            prepared_data = (
                erlab.interactive.imagetool.viewer_state._prepare_input_data(
                    typing.cast("list[xr.DataArray]", data),
                    self._manager,
                    allow_dialog=watched_var is None,
                )
            )
        except ValueError:
            logger.exception(
                "Error creating ImageTool window",
                extra={"suppress_ui_alert": True},
            )
            self._error_creating_imagetool()
            return [False for _item in data]
        if prepared_data is None:
            return [False for _item in data]

        link = kwargs.pop("link", False)
        link_colors = kwargs.pop("link_colors", True)
        indices: list[int] = []
        kwargs["_in_manager"] = True
        kwargs.setdefault("options_model", self._manager.effective_interactive_options)

        load_func = kwargs.pop("load_func", None)
        load_selections = kwargs.pop("load_selections", None)
        load_preparation_operations = kwargs.pop("preparation_operations", None)
        source_input_ndims = kwargs.pop("source_input_ndims", None)
        source_input_dtypes = kwargs.pop("source_input_dtypes", None)
        if show is None:
            show = len(prepared_data) == 1
        watched_metadata = dict(watched_metadata or {})
        if watched_var is not None:
            watched_metadata.setdefault(
                "workspace_link_id", self._manager._workspace_state.link_id
            )

        embedded_load_selection: FileDataSelection | None = None
        if load_func is not None:
            if len(load_func) not in (2, 3):
                raise ValueError(
                    "load_func must contain a loader and kwargs, optionally followed "
                    "by one selection"
                )
            if len(load_func) == 2 and load_selections is None:
                raise ValueError(
                    "A two-item load_func requires explicit load_selections"
                )
            if len(load_func) == 3:
                if not isinstance(load_func[2], FileDataSelection):
                    raise TypeError("load_func selection must be a FileDataSelection")
                if load_selections is None and len(prepared_data) != 1:
                    raise ValueError(
                        "A load_func selection can only describe one prepared array"
                    )
                embedded_load_selection = load_func[2]

        normalized_load_selections: tuple[FileDataSelection, ...] | None = None
        if load_selections is not None:
            if isinstance(load_selections, str | bytes) or not isinstance(
                load_selections, collections.abc.Sequence
            ):
                raise TypeError("load_selections must be a sequence")
            normalized_load_selections = tuple(load_selections)
            if len(normalized_load_selections) != len(prepared_data):
                raise ValueError(
                    "load_selections must contain one selection per prepared array"
                )
            if any(
                not isinstance(selection, FileDataSelection)
                for selection in normalized_load_selections
            ):
                raise TypeError(
                    "load_selections must contain FileDataSelection instances"
                )

        for i, prepared in enumerate(prepared_data):
            load_selection = (
                normalized_load_selections[i]
                if normalized_load_selections is not None
                else (
                    embedded_load_selection
                    if embedded_load_selection is not None
                    else prepared.selection
                )
            )
            this_load_func = (*load_func[:2], load_selection) if load_func else None
            preparation_operations = (
                tuple(
                    typing.cast(
                        "Sequence[ToolProvenanceOperation]",
                        load_preparation_operations,
                    )[i]
                )
                if load_preparation_operations is not None
                else prepared.operations
            )
            source_input_ndim = (
                typing.cast("Sequence[int]", source_input_ndims)[i]
                if source_input_ndims is not None
                else prepared.source_ndim
            )
            source_input_dtype = (
                typing.cast("Sequence[typing.Any]", source_input_dtypes)[i]
                if source_input_dtypes is not None
                else prepared.source_dtype
            )
            try:
                indices.append(
                    self._manager.add_imagetool(
                        ImageTool(
                            prepared.data,
                            **kwargs,
                            load_func=this_load_func,
                            preparation_operations=preparation_operations,
                        ),
                        show=show,
                        activate=show,
                        watched_var=watched_var,
                        watched_workspace_link_id=typing.cast(
                            "str | None",
                            watched_metadata.get("workspace_link_id"),
                        ),
                        watched_source_label=typing.cast(
                            "str | None", watched_metadata.get("source_label")
                        ),
                        watched_source_uid=typing.cast(
                            "str | None", watched_metadata.get("source_uid")
                        ),
                        watched_connected=bool(
                            watched_metadata.get("connected", watched_var is not None)
                        ),
                        source_input_ndim=source_input_ndim,
                        source_input_dtype=source_input_dtype,
                    )
                )
                if watched_var is not None:
                    node = self._manager._node_for_target(indices[-1])
                    if node.imagetool is not None:
                        node.imagetool._update_title()
            except Exception:
                flags.append(False)
                logger.exception(
                    "Error creating ImageTool window",
                    extra={"suppress_ui_alert": True},
                )
                self._error_creating_imagetool()
            else:
                flags.append(True)

        if link:
            self._manager.link_imagetools(*indices, link_colors=link_colors)

        return flags

    def _show_loaded_info(
        self,
        loaded: list[pathlib.Path],
        canceled: list[pathlib.Path],
        failed: list[pathlib.Path],
        retry_callback: Callable[[list[pathlib.Path]], typing.Any],
    ) -> None:
        """Report aggregate file-loading results and offer to retry failures."""
        loaded, canceled, failed = (
            list(dict.fromkeys(loaded)),
            list(dict.fromkeys(canceled)),
            list(dict.fromkeys(failed)),
        )
        n_done, n_fail = len(loaded), len(failed)
        self._manager._status_bar.showMessage(
            f"Loaded {n_done} {'file' if n_done == 1 else 'files'}", 5000
        )
        if n_fail == 0:
            return

        message = f"Loaded {n_done} {'file' if n_done == 1 else 'files'}"
        message += f" with {n_fail} {'failure' if n_fail == 1 else 'failures'}."
        msg_box = QtWidgets.QMessageBox(self._manager)
        msg_box.setText(message)
        details = ""
        for label, paths in (
            ("Loaded", loaded),
            ("Failed", failed),
            ("Canceled", canceled),
        ):
            if paths:
                names = "\n".join(path.name for path in paths)
                details += f"{label}:\n{names}\n\n"
        msg_box.setDetailedText(details)
        msg_box.setInformativeText("Do you want to retry loading the failed files?")
        msg_box.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Retry
            | QtWidgets.QMessageBox.StandardButton.Cancel
        )
        msg_box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Retry)
        if msg_box.exec() == QtWidgets.QMessageBox.StandardButton.Retry:
            retry_callback(failed)

    def _error_creating_imagetool(self) -> None:
        """Show one user-facing error after ImageTool construction fails."""
        erlab.interactive.utils.MessageDialog.critical(
            self._manager,
            "Error",
            "An error occurred while creating the ImageTool window.",
            "The data may be incompatible with ImageTool.",
        )

    def open_multiple_files(
        self, queued: list[pathlib.Path], try_workspace: bool = False
    ) -> None:
        """Open paths as workspaces, ordinary data files, or directories."""
        paths = list(queued)
        if try_workspace and self._manager._workspace_state.save_in_progress:
            self._manager._status_bar.showMessage(
                "Workspace save in progress; open after it finishes", 3000
            )
            return

        n_files = len(paths)
        loaded: list[pathlib.Path] = []
        failed: list[pathlib.Path] = []
        if try_workspace:
            remaining: list[pathlib.Path] = []
            for path in paths:
                result = self._manager._workspace_controller.open_workspace_candidate(
                    path
                )
                if result == "not-workspace":
                    remaining.append(path)
                    continue
                if result == "stop":
                    return
                loaded.append(path)
            paths = remaining

        if not paths:
            return

        valid_loaders: dict[str, tuple[Callable, dict[str, typing.Any]]] = (
            erlab.interactive.utils.file_loaders(paths)
        )
        if not valid_loaders:
            if all(path.is_dir() for path in paths):
                explorer = typing.cast(
                    "_TabbedExplorer", self._manager._show_standalone_app("explorer")
                )
                for path in paths:
                    explorer.add_tab(root_path=path)
                return

            singular = n_files == 1
            QtWidgets.QMessageBox.critical(
                self._manager,
                "Error",
                f"The selected {'file' if singular else 'files'} "
                f"with extension '{paths[0].suffix}' {'is' if singular else 'are'} "
                "not supported by any available plugin.",
            )
            return

        if len(valid_loaders) == 1:
            name_filter, (func, kwargs) = next(iter(valid_loaders.items()))
            if _is_loader_func(func):
                selected = self._manager._select_loader_options(
                    valid_loaders, name_filter, sample_paths=paths
                )
                if selected is None:
                    return
                self._manager._recent_name_filter, func, kwargs = selected
            else:
                self._manager._recent_name_filter = name_filter
        else:
            selected = self._manager._select_loader_options(
                valid_loaders, sample_paths=paths
            )
            if selected is None:
                return
            self._manager._recent_name_filter, func, kwargs = selected

        self.add_from_multiple_files(
            loaded,
            paths,
            failed,
            func,
            kwargs,
            self.open_multiple_files,
        )

    def add_from_multiple_files(
        self,
        loaded: list[pathlib.Path],
        queued: list[pathlib.Path],
        failed: list[pathlib.Path],
        func: Callable,
        kwargs: dict[str, typing.Any],
        retry_callback: Callable,
    ) -> None:
        """Start a multi-file load and report the aggregate result."""
        handler = _MultiFileHandler(self._manager, queued, func, kwargs)
        self._manager._file_handlers.add(handler)

        def _finished_callback(loaded_new, aborted, failed_new) -> None:
            self._show_loaded_info(
                loaded + loaded_new,
                aborted,
                failed + failed_new,
                retry_callback=retry_callback,
            )
            self._manager._file_handlers.remove(handler)

        handler.sigFinished.connect(_finished_callback)
        handler.start()


class _DataLoaderSignals(QtCore.QObject):
    sigLoaded = QtCore.Signal(pathlib.Path, object)
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
            data = self._func(self._file_path, **self._kwargs)
        except Exception:
            logger.debug(
                "Error loading data from %s", self._file_path
            )  # Use debug level to avoid duplicate popup
            self.signals.sigFailed.emit(self._file_path, traceback.format_exc())
        else:
            self.signals.sigLoaded.emit(self._file_path, data)


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
    canceled
        List of files that were loaded but not opened.
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
        self.n_total: int = len(file_list)
        self._queue: collections.deque[pathlib.Path] = collections.deque(file_list)
        self._func = func
        self._kwargs = kwargs

        self.loaded: list[pathlib.Path] = []
        self.canceled: list[pathlib.Path] = []
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
            self.sigFinished.emit(self.loaded, self.canceled + self.queued, self.failed)
            return

        file_path = self._queue.popleft()

        self.manager._status_bar.showMessage(f"Loading {file_path.name}...")

        loader = _DataLoader(file_path, self._func, self._kwargs)
        loader.signals.sigLoaded.connect(self._on_loaded)
        loader.signals.sigFailed.connect(self._on_failed)
        self._threadpool.start(loader)

    @QtCore.Slot(pathlib.Path, object)
    def _on_loaded(self, file_path: pathlib.Path, data: object) -> None:
        self.manager._status_bar.showMessage("")
        self.manager._recent_directory = str(file_path.parent)
        try:
            selected_data = (
                erlab.interactive.imagetool.viewer_state._select_input_dataarrays(
                    typing.cast(
                        "xr.DataArray | xr.Dataset | xr.DataTree | list[xr.DataArray]",
                        data,
                    ),
                    self.manager,
                )
            )
        except Exception:
            self._on_failed(file_path, traceback.format_exc())
            return
        if selected_data is None:
            self.canceled.append(file_path)
            erlab.interactive.utils.single_shot(self, 0, self._load_next)
            return

        self.loaded.append(file_path)
        erlab.interactive.utils.single_shot(
            self, 0, lambda: self._deliver_and_queue(file_path, selected_data)
        )

    def _deliver_and_queue(
        self,
        file_path: pathlib.Path,
        selected_data: tuple[typing.Any, ...],
    ) -> None:
        func: Callable | str = self._func
        func_instance = getattr(func, "__self__", None)
        if isinstance(func_instance, erlab.io.dataloader.LoaderBase):
            func = func_instance.name

        self.manager._data_ingress.receive_data(
            [prepared.data for prepared in selected_data],
            kwargs={
                "file_path": file_path,
                "load_func": (func, self._kwargs.copy()),
                "load_selections": tuple(
                    prepared.selection for prepared in selected_data
                ),
                "preparation_operations": tuple(
                    prepared.operations for prepared in selected_data
                ),
                "source_input_ndims": tuple(
                    prepared.source_ndim for prepared in selected_data
                ),
                "source_input_dtypes": tuple(
                    prepared.source_dtype for prepared in selected_data
                ),
            },
            show=(self.n_total == 1),
        )
        erlab.interactive.utils.single_shot(self, 0, self._load_next)

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
                self.sigFinished.emit(
                    self.loaded, self.canceled + self.queued, self.failed
                )

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
