import pathlib
import typing
from collections.abc import Callable

from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive.explorer._base_explorer import (
    _IGOR_PRO_MIME_TYPES,
    _PREVIEW_WORKER_STOP_TIMEOUT_MS,
    DataExplorerTabState,
    _DataExplorer,
    _FileSystem,
    _ReprFetcher,
)
from erlab.interactive.explorer._tabbed_explorer import (
    DataExplorerState,
    _TabbedExplorer,
)
from erlab.interactive.imagetool.manager import _dialogs


class _PreviewTrackingExplorer(QtWidgets.QWidget):
    def __init__(self, name: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.current_directory = pathlib.Path(name)
        self.menu_bar = QtWidgets.QMenuBar(self)
        self.stopped_preview_workers = 0
        self._preview_stopping = False

    def _stop_preview_workers(self) -> bool:
        self.stopped_preview_workers += 1
        return True


class _DeferredPreviewTrackingExplorer(_PreviewTrackingExplorer):
    def __init__(self, name: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(name, parent)
        self.deferred_delete_count = 0
        self.defer_preview_stop = True

    def _stop_preview_workers(self) -> bool:
        self.stopped_preview_workers += 1
        if self.defer_preview_stop:
            self._preview_stopping = True
            return False
        return True

    def _delete_when_preview_workers_done(self) -> None:
        self.deferred_delete_count += 1


class _PreviewTrackingTabbedExplorer(_TabbedExplorer):
    def add_tab(self, **kwargs) -> None:
        tab = QtWidgets.QWidget()
        explorer = _PreviewTrackingExplorer(
            f"tab-{self.tab_widget.count()}", parent=tab
        )
        tab._explorer = explorer  # type: ignore[attr-defined]
        self.tab_widget.addTab(tab, explorer.current_directory.name)
        self.tab_widget.setCurrentIndex(self.tab_widget.count() - 1)


class _DeferredPreviewTrackingTabbedExplorer(_TabbedExplorer):
    def add_tab(self, **kwargs) -> None:
        tab = QtWidgets.QWidget()
        explorer = _DeferredPreviewTrackingExplorer(
            f"tab-{self.tab_widget.count()}", parent=tab
        )
        tab._explorer = explorer  # type: ignore[attr-defined]
        self.tab_widget.addTab(tab, explorer.current_directory.name)
        self.tab_widget.setCurrentIndex(self.tab_widget.count() - 1)


def test_explorer_last_tab_closes_without_manager(qtbot, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(erlab.interactive.imagetool.manager, "_manager_instance", None)

    class _TrackingTabbedExplorer(_TabbedExplorer):
        def __init__(self, *args, **kwargs) -> None:
            self.close_event_count = 0
            super().__init__(*args, **kwargs)

        def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:
            self.close_event_count += 1
            super().closeEvent(event)

    win = _TrackingTabbedExplorer(root_path=tmp_path)
    qtbot.addWidget(win)
    with qtbot.waitExposed(win):
        win.show()

    win.close_tab(0)

    qtbot.wait_until(lambda: not win.isVisible())
    assert win.close_event_count == 1


def test_tabbed_explorer_close_tab_stops_preview_workers(
    qtbot,
) -> None:
    win = _PreviewTrackingTabbedExplorer()
    qtbot.addWidget(win)
    discarded_explorer = win.get_explorer(0)
    assert isinstance(discarded_explorer, _PreviewTrackingExplorer)

    win.add_tab()

    win.close_tab(0)

    assert win.tab_widget.count() == 1
    assert discarded_explorer.stopped_preview_workers == 1


def test_tabbed_explorer_clear_tabs_discards_each_tab(qtbot) -> None:
    win = _PreviewTrackingTabbedExplorer()
    qtbot.addWidget(win)
    win.add_tab()
    win.add_tab()
    explorers = [win.get_explorer(index) for index in range(win.tab_widget.count())]
    assert all(isinstance(explorer, _PreviewTrackingExplorer) for explorer in explorers)

    win._clear_tabs()

    assert win.tab_widget.count() == 0
    assert [
        typing.cast("_PreviewTrackingExplorer", explorer).stopped_preview_workers
        for explorer in explorers
    ] == [1, 1, 1]


def test_tabbed_explorer_discard_tab_handles_missing_explorer(qtbot) -> None:
    win = _PreviewTrackingTabbedExplorer()
    qtbot.addWidget(win)
    empty_tab = QtWidgets.QWidget()
    empty_index = win.tab_widget.addTab(empty_tab, "empty")

    win._discard_tab(empty_index)

    assert win.tab_widget.count() == 1

    win._discard_tab(win.tab_widget.count())

    assert win.tab_widget.count() == 1


def test_tabbed_explorer_discard_tab_defers_busy_preview_workers(qtbot) -> None:
    win = _PreviewTrackingTabbedExplorer()
    qtbot.addWidget(win)
    tab = QtWidgets.QWidget()
    explorer = _DeferredPreviewTrackingExplorer("busy", tab)
    tab._explorer = explorer  # type: ignore[attr-defined]
    busy_index = win.tab_widget.addTab(tab, "busy")

    win._discard_tab(busy_index)

    assert win.tab_widget.count() == 1
    assert explorer.stopped_preview_workers == 1
    assert explorer.deferred_delete_count == 1


def test_tabbed_explorer_close_stops_preview_workers_without_removing_tabs(
    qtbot,
) -> None:
    win = _PreviewTrackingTabbedExplorer()
    qtbot.addWidget(win)
    explorer = win.current_explorer
    assert isinstance(explorer, _PreviewTrackingExplorer)

    win.close()

    assert win.tab_widget.count() == 1
    assert win.current_explorer is explorer
    assert explorer.stopped_preview_workers == 1


def test_tabbed_explorer_close_ignores_busy_preview_workers(qtbot) -> None:
    win = _DeferredPreviewTrackingTabbedExplorer()
    qtbot.addWidget(win)
    explorer = win.current_explorer
    assert isinstance(explorer, _DeferredPreviewTrackingExplorer)
    empty_tab = QtWidgets.QWidget()
    win.tab_widget.addTab(empty_tab, "empty")
    event = QtGui.QCloseEvent()

    win.closeEvent(event)

    assert not event.isAccepted()
    assert win.tab_widget.count() == 2
    assert win.current_explorer is explorer
    assert explorer.stopped_preview_workers == 1
    assert not explorer._preview_stopping

    win.closeEvent(None)

    assert explorer.stopped_preview_workers == 2
    assert not explorer._preview_stopping
    explorer.defer_preview_stop = False
    win.close()


def test_explorer_close_stops_preview_workers(
    qtbot,
) -> None:
    class _TrackingDataExplorer(_DataExplorer):
        def __init__(self) -> None:
            QtWidgets.QMainWindow.__init__(self)
            self.stopped_preview_workers = False

        def _stop_preview_workers(
            self, timeout_ms: int = _PREVIEW_WORKER_STOP_TIMEOUT_MS
        ) -> bool:
            del timeout_ms
            self.stopped_preview_workers = True
            return True

    explorer = _TrackingDataExplorer()
    qtbot.addWidget(explorer)
    event = QtGui.QCloseEvent()

    explorer.closeEvent(event)

    assert explorer.stopped_preview_workers
    assert event.isAccepted()


def test_explorer_close_ignores_busy_preview_workers(
    qtbot, example_loader, example_data_dir: pathlib.Path
) -> None:
    class _BusyDataExplorer(_DataExplorer):
        def __init__(self, *args, **kwargs) -> None:
            self.stopped_preview_workers = False
            self.defer_preview_stop = True
            super().__init__(*args, **kwargs)

        def _stop_preview_workers(
            self, timeout_ms: int = _PREVIEW_WORKER_STOP_TIMEOUT_MS
        ) -> bool:
            self.stopped_preview_workers = True
            if self.defer_preview_stop:
                self._preview_stopping = True
                return False
            return True

    explorer = _BusyDataExplorer(root_path=example_data_dir, loader_name="example")
    qtbot.addWidget(explorer)
    event = QtGui.QCloseEvent()

    explorer.closeEvent(event)

    assert explorer.stopped_preview_workers
    assert not event.isAccepted()
    assert not explorer._preview_stopping

    explorer.stopped_preview_workers = False
    explorer.closeEvent(None)

    assert explorer.stopped_preview_workers
    assert not explorer._preview_stopping
    explorer.defer_preview_stop = False
    explorer.close()


def test_tabbed_explorer_show_path_adds_selected_file_tab(
    qtbot, example_loader, example_data_dir: pathlib.Path
) -> None:
    source_path = example_data_dir / "data_002.h5"
    win = _TabbedExplorer(root_path=example_data_dir, loader_name="example")
    qtbot.addWidget(win)
    initial_count = win.tab_widget.count()

    win.show_path(source_path)

    assert win.tab_widget.count() == initial_count + 1
    explorer = win.current_explorer
    assert explorer is not None
    assert explorer.current_directory == example_data_dir
    qtbot.wait_until(lambda: explorer._tree_view.selected_paths == [source_path])
    qtbot.wait_until(lambda: explorer._displayed_selection == [source_path])


def test_explorer_general(
    qtbot,
    example_loader,
    example_data_dir: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()

        # Set the recent directory and name filter
        manager._recent_directory = str(example_data_dir)
        manager._recent_name_filter = next(
            iter(erlab.io.loaders["example"].file_dialog_methods.keys())
        )

        # Initialize data explorer
        manager.ensure_explorer_initialized()
        assert hasattr(manager, "explorer")
        tabbed_explorer: _TabbedExplorer = manager.explorer

        # Add tab
        tabbed_explorer.add_tab()
        qtbot.wait_until(lambda: tabbed_explorer.tab_widget.count() == 2)

        # Remove tab
        tabbed_explorer.get_explorer(1).try_close()
        qtbot.wait_until(lambda: tabbed_explorer.tab_widget.count() == 1)

        explorer: _DataExplorer = tabbed_explorer.get_explorer(0)

        # Show data explorer
        with qtbot.wait_exposed(tabbed_explorer):
            manager.show_explorer()

        # Enable preview
        explorer._preview_check.setChecked(True)

        assert explorer.loader_name == "example"
        assert explorer._fs_model.file_system.path == example_data_dir

        # Reload folder
        explorer._reload_act.trigger()

        # Set show hidden files
        explorer._tree_view.model().set_show_hidden(False)

        # Sort by name
        explorer._tree_view.sortByColumn(0, QtCore.Qt.SortOrder.DescendingOrder)

        def select_files(indices: list[int], deselect: bool = False) -> None:
            selection_model = explorer._tree_view.selectionModel()

            for index in indices:
                idx_start = explorer._tree_view.model().index(index, 0)
                idx_end = explorer._tree_view.model().index(
                    index, explorer._tree_view.model().columnCount() - 1
                )
                selection_model.select(
                    QtCore.QItemSelection(idx_start, idx_end),
                    QtCore.QItemSelectionModel.SelectionFlag.Deselect
                    if deselect
                    else QtCore.QItemSelectionModel.SelectionFlag.Select,
                )
                if deselect:
                    qtbot.wait_until(
                        lambda idx=idx_end: (
                            idx not in explorer._tree_view.selectedIndexes()
                        )
                    )
                else:
                    qtbot.wait_until(
                        lambda idx=idx_end: idx in explorer._tree_view.selectedIndexes()
                    )

        assert explorer._text_edit.toPlainText() == explorer.TEXT_NONE_SELECTED

        # Multiple selection
        select_files([2, 3, 4])
        assert (
            len(explorer._tree_view.selectedIndexes())
            == 3 * explorer._tree_view.model().columnCount()
        )

        # Show multiple in manager
        explorer.to_manager()

        # Wait until the manager has updated
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=20000)

        # Clear selection
        select_files([2, 3, 4], deselect=True)

        # Single selection
        select_files([2])
        assert (
            len(explorer._tree_view.selectedIndexes())
            == explorer._tree_view.model().columnCount()
        )

        # Show single in manager
        explorer.to_manager()
        qtbot.wait_until(lambda: manager.ntools == 4, timeout=20000)

        # Show single in manager with socket
        erlab.interactive.imagetool.manager._always_use_socket = True
        explorer.to_manager()
        qtbot.wait_until(lambda: manager.ntools == 5, timeout=20000)
        erlab.interactive.imagetool.manager._always_use_socket = False

        # Reload data in manager
        # Choose tool 3
        qmodelindex = manager.tree_view._model._row_index(3)
        manager.tree_view.selectionModel().select(
            QtCore.QItemSelection(qmodelindex, qmodelindex),
            QtCore.QItemSelectionModel.SelectionFlag.Select,
        )

        # Lock levels to check if they are preserved after reloading
        manager.get_imagetool(3).slicer_area.lock_levels(True)
        manager.get_imagetool(3).slicer_area.levels = (1.0, 23.0)

        old_state = manager.get_imagetool(3).slicer_area.state.copy()

        with qtbot.wait_signal(manager.get_imagetool(3).slicer_area.sigDataChanged):
            manager.reload_action.trigger()
        qtbot.wait(200)

        assert manager.get_imagetool(3).slicer_area.state == old_state

        # Clear selection
        select_files([2], deselect=True)

        # Single selection multiple times
        for i in range(1, 5):
            with qtbot.wait_signal(explorer._preview._sigDataChanged, timeout=2000):
                select_files([i])
            select_files([i], deselect=True)

        # Test sorting by different columns
        for i in range(4):
            explorer._tree_view.sortByColumn(i, QtCore.Qt.SortOrder.AscendingOrder)

        # # Trigger open in file explorer
        # explorer._finder_act.trigger()

        explorer.close()


def test_explorer_loader_extensions_apply_only_to_manager_loads(
    qtbot,
    monkeypatch,
    example_loader,
    example_data_dir: pathlib.Path,
) -> None:
    explorer = _DataExplorer(root_path=example_data_dir, loader_name="example")
    qtbot.addWidget(explorer)

    file_path = example_data_dir / "data_002.h5"
    assert explorer._manager_load_kwargs({"single": True}) == {"single": True}
    explorer._loader_extensions_by_name["example"] = {
        "additional_coords": {"gui_extra": 7.0}
    }
    assert explorer._manager_load_kwargs({"loader_extensions": ("ignored",)}) == {
        "loader_extensions": {"additional_coords": {"gui_extra": 7.0}}
    }

    load_calls: list[tuple[list[pathlib.Path], str | None, dict[str, typing.Any]]] = []
    monkeypatch.setattr(
        erlab.interactive.imagetool.manager,
        "is_running",
        lambda: True,
    )
    monkeypatch.setattr(
        erlab.interactive.imagetool.manager,
        "load_in_manager",
        lambda files, loader_name=None, **kwargs: load_calls.append(
            (list(files), loader_name, kwargs)
        ),
    )

    explorer.to_manager(files=[file_path])
    assert load_calls == [
        (
            [file_path],
            "example",
            {"loader_extensions": {"additional_coords": {"gui_extra": 7.0}}},
        )
    ]

    preview_calls: list[dict[str, typing.Any]] = []

    def _preview_loader(_file_path, **kwargs):
        preview_calls.append(kwargs)
        return erlab.io.loaders["example"].load(
            _file_path,
            **kwargs,
        )

    worker = _ReprFetcher(file_path, _preview_loader, include_values=False)
    worker.run()
    assert preview_calls == [{"single": True, "load_kwargs": {"without_values": True}}]


def test_repr_fetcher_keeps_coordinate_values_without_preview(
    tmp_path: pathlib.Path,
) -> None:
    import numpy as np
    import xarray as xr

    data = xr.DataArray(
        np.zeros((3, 4)),
        dims=("x", "y"),
        coords={
            "x": np.array([10.0, 12.0, 14.0]),
            "y": np.array([0.0, 0.5, 1.0, 1.5]),
        },
        name="demo",
    )
    fetched: list[tuple[str, object]] = []

    worker = _ReprFetcher(tmp_path / "data.h5", lambda *_args, **_kwargs: data, False)
    worker.signals.fetched.connect(
        lambda _path, text, preview_data: fetched.append((text, preview_data))
    )

    worker.run()

    assert len(fetched) == 1
    text, preview_data = fetched[0]
    assert preview_data is None
    assert "10 : 2 : 14" in text
    assert "0 : 0.5 : 1.5" in text
    assert "float64 [3]" not in text
    assert "float64 [4]" not in text


def test_repr_fetcher_aborted_before_run_skips_loader(
    tmp_path: pathlib.Path,
) -> None:
    events: list[str] = []

    def fail_loader(*_args, **_kwargs):
        raise AssertionError("aborted preview worker should not load data")

    worker = _ReprFetcher(tmp_path / "data.h5", fail_loader, include_values=False)
    worker.signals.fetched.connect(lambda *_args: events.append("fetched"))
    worker.signals.finished.connect(lambda *_args: events.append("finished"))

    worker.abort()
    worker.run()

    assert events == ["finished"]


def test_repr_fetcher_keeps_python_ownership_until_finished(
    tmp_path: pathlib.Path,
) -> None:
    worker = _ReprFetcher(tmp_path / "data.h5", object(), include_values=False)

    assert not worker.autoDelete()


def test_repr_fetcher_aborted_after_load_skips_formatting(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    import xarray as xr

    events: list[str] = []

    def fail_format(*_args, **_kwargs):
        raise AssertionError("aborted preview worker should not format data")

    monkeypatch.setattr(erlab.utils.formatting, "format_darr_html", fail_format)

    worker: _ReprFetcher

    def load_then_abort(*_args, **_kwargs):
        worker.abort()
        return xr.DataArray([1.0], dims=("x",))

    worker = _ReprFetcher(tmp_path / "data.h5", load_then_abort, include_values=False)
    worker.signals.fetched.connect(lambda *_args: events.append("fetched"))
    worker.signals.finished.connect(lambda *_args: events.append("finished"))

    worker.run()

    assert events == ["finished"]


def test_explorer_stop_preview_workers_aborts_with_bounded_wait(
    tmp_path: pathlib.Path,
) -> None:
    class _FakeThreadPool:
        def __init__(self) -> None:
            self.clear_called = False
            self.wait_timeout_ms: int | None = None

        def clear(self) -> None:
            self.clear_called = True

        def waitForDone(self, timeout_ms: int) -> bool:
            self.wait_timeout_ms = timeout_ms
            return False

    class _ExplorerDouble:
        def __init__(self, worker: _ReprFetcher) -> None:
            self._preview_stopping = False
            self._preview_workers = {worker}
            self._preview_threadpool = _FakeThreadPool()

    worker = _ReprFetcher(tmp_path / "data.h5", object(), include_values=False)
    explorer = _ExplorerDouble(worker)

    assert not _DataExplorer._stop_preview_workers(
        typing.cast("_DataExplorer", explorer)
    )
    assert worker._aborted.is_set()
    assert explorer._preview_threadpool.clear_called
    assert (
        explorer._preview_threadpool.wait_timeout_ms == _PREVIEW_WORKER_STOP_TIMEOUT_MS
    )


def test_explorer_stop_preview_workers_disconnects_finished_workers(
    tmp_path: pathlib.Path,
) -> None:
    class _FakeThreadPool:
        def __init__(self) -> None:
            self.clear_called = False
            self.wait_timeout_ms: int | None = None

        def clear(self) -> None:
            self.clear_called = True

        def waitForDone(self, timeout_ms: int) -> bool:
            self.wait_timeout_ms = timeout_ms
            return True

    class _ExplorerDouble:
        def __init__(self, worker: _ReprFetcher) -> None:
            self._preview_stopping = False
            self._preview_workers = {worker}
            self._preview_threadpool = _FakeThreadPool()
            self.disconnected_workers: list[_ReprFetcher] = []

        def _disconnect_preview_worker(self, worker: _ReprFetcher) -> None:
            self.disconnected_workers.append(worker)

    worker = _ReprFetcher(tmp_path / "data.h5", object(), include_values=False)
    explorer = _ExplorerDouble(worker)

    assert _DataExplorer._stop_preview_workers(typing.cast("_DataExplorer", explorer))
    assert worker._aborted.is_set()
    assert explorer._preview_threadpool.clear_called
    assert (
        explorer._preview_threadpool.wait_timeout_ms == _PREVIEW_WORKER_STOP_TIMEOUT_MS
    )
    assert explorer.disconnected_workers == [worker]
    assert explorer._preview_workers == set()


def test_explorer_selection_change_ignores_stopping_preview() -> None:
    class _ExplorerDouble:
        _preview_stopping = True

    _DataExplorer._on_selection_changed(typing.cast("_DataExplorer", _ExplorerDouble()))


def test_explorer_show_file_info_ignores_stopping_preview() -> None:
    class _ExplorerDouble:
        _preview_stopping = True

    _DataExplorer._show_file_info(
        typing.cast("_DataExplorer", _ExplorerDouble()),
        "data.h5",
        "<b>new</b>",
        None,
    )


def test_explorer_preview_worker_finished_removes_worker(
    tmp_path: pathlib.Path,
) -> None:
    class _ExplorerDouble:
        def __init__(self, worker: _ReprFetcher) -> None:
            self._preview_workers = {worker}

        def _disconnect_preview_worker(self, worker: _ReprFetcher) -> None:
            return None

    worker = _ReprFetcher(tmp_path / "data.h5", object(), include_values=False)
    explorer = _ExplorerDouble(worker)

    _DataExplorer._preview_worker_finished(
        typing.cast("_DataExplorer", explorer), worker
    )
    assert explorer._preview_workers == set()


def test_explorer_preview_worker_finished_disconnects_worker_signals(
    qtbot,
    example_loader,
    example_data_dir: pathlib.Path,
) -> None:
    class _TrackingDataExplorer(_DataExplorer):
        def __init__(self, *args, **kwargs) -> None:
            self.preview_calls: list[tuple[str, str, object]] = []
            super().__init__(*args, **kwargs)

        def _show_file_info(self, file_path: str, text: str, data: object) -> None:
            self.preview_calls.append((file_path, text, data))

    explorer = _TrackingDataExplorer(root_path=example_data_dir, loader_name="example")
    qtbot.addWidget(explorer)
    worker = _ReprFetcher(example_data_dir / "data_001.h5", object(), False)
    worker.signals.fetched.connect(explorer._show_file_info)
    worker.signals.finished.connect(explorer._preview_worker_finished)
    explorer._preview_workers.add(worker)

    explorer._preview_worker_finished(worker)
    worker.signals.fetched.emit(str(worker.file_path), "preview", None)

    assert explorer._preview_workers == set()
    assert explorer.preview_calls == []


def test_explorer_delete_when_preview_workers_done_deletes_immediately(
    tmp_path: pathlib.Path,
) -> None:
    class _FakeThreadPool:
        def __init__(self) -> None:
            self.clear_called = False

        def clear(self) -> None:
            self.clear_called = True

        def activeThreadCount(self) -> int:
            return 0

        def waitForDone(self, _timeout_ms: int) -> bool:
            return True

    class _ExplorerDouble:
        def __init__(self, worker: _ReprFetcher) -> None:
            self.deleted_later = False
            self._preview_workers = {worker}
            self._preview_threadpool = _FakeThreadPool()

        def deleteLater(self) -> None:
            self.deleted_later = True

    worker = _ReprFetcher(tmp_path / "data.h5", object(), include_values=False)
    explorer = _ExplorerDouble(worker)

    _DataExplorer._delete_when_preview_workers_done(
        typing.cast("_DataExplorer", explorer)
    )

    assert explorer._preview_threadpool.clear_called
    assert explorer._preview_workers == set()
    assert explorer.deleted_later


def test_explorer_delete_when_preview_workers_done_defers_active_pool(
    monkeypatch,
    tmp_path: pathlib.Path,
) -> None:
    class _FakeThreadPool:
        def __init__(self) -> None:
            self.clear_called = False

        def clear(self) -> None:
            self.clear_called = True

        def activeThreadCount(self) -> int:
            return 1

        def waitForDone(self, _timeout_ms: int) -> bool:
            return False

    class _ExplorerDouble:
        def __init__(self, worker: _ReprFetcher) -> None:
            self._preview_workers = {worker}
            self._preview_threadpool = _FakeThreadPool()

        def _delete_when_preview_workers_done(self) -> None:
            _DataExplorer._delete_when_preview_workers_done(
                typing.cast("_DataExplorer", self)
            )

    worker = _ReprFetcher(tmp_path / "data.h5", object(), include_values=False)
    explorer = _ExplorerDouble(worker)
    callbacks: list[tuple[object, int, object]] = []
    monkeypatch.setattr(
        erlab.interactive.utils,
        "single_shot",
        lambda receiver, msec, callback, *guards: callbacks.append(
            (receiver, msec, callback)
        ),
    )

    _DataExplorer._delete_when_preview_workers_done(
        typing.cast("_DataExplorer", explorer)
    )

    assert explorer._preview_threadpool.clear_called
    assert callbacks == [(explorer, 100, explorer._delete_when_preview_workers_done)]


def test_explorer_workspace_state_restores_selection(
    qtbot,
    example_loader,
    example_data_dir: pathlib.Path,
) -> None:
    explorer = _DataExplorer(root_path=example_data_dir, loader_name="example")
    qtbot.addWidget(explorer)
    qtbot.wait_until(lambda: explorer._tree_view.model().rowCount() > 0)

    assert (
        DataExplorerTabState.model_validate(
            {
                "root_path": str(example_data_dir),
                "selected_paths": None,
                "splitter_sizes": None,
                "preview_splitter_sizes": None,
            }
        ).selected_paths
        == ()
    )
    assert DataExplorerState.model_validate({"tabs": None}).tabs == ()

    explorer._tree_view.selectionModel().selectionChanged.disconnect(
        explorer._on_selection_changed
    )
    file_paths = (
        example_data_dir / "data_002.h5",
        example_data_dir / "data_003.h5",
    )
    missing_path = example_data_dir / "missing.h5"
    explorer.restore_workspace_state(
        DataExplorerTabState(
            root_path=str(example_data_dir),
            loader_name="example",
            selected_paths=(*map(str, file_paths), str(missing_path)),
        )
    )

    qtbot.wait_until(lambda: set(file_paths).issubset(explorer._current_selection))
    assert set(file_paths).issubset(explorer._current_selection)
    assert missing_path not in explorer._current_selection
    assert not explorer._model_index_for_path(missing_path).isValid()


def test_explorer_workspace_state_missing_root_is_empty(
    qtbot,
    example_loader,
    tmp_path: pathlib.Path,
) -> None:
    explorer = _DataExplorer(root_path=tmp_path, loader_name="example")
    qtbot.addWidget(explorer)

    missing_root = tmp_path / "disconnected-share"
    missing_path = missing_root / "data_001.h5"
    explorer.restore_workspace_state(
        DataExplorerTabState(
            root_path=str(missing_root),
            loader_name="example",
            selected_paths=(str(missing_path),),
        )
    )
    qtbot.wait(10)

    assert explorer.current_directory == missing_root
    assert explorer._tree_view.model().rowCount() == 0
    assert explorer.workspace_state_payload()["root_path"] == str(missing_root)
    assert explorer._current_selection == []
    assert not explorer._model_index_for_path(missing_path).isValid()


def test_explorer_filesystem_read_error_stays_empty(
    monkeypatch,
    tmp_path: pathlib.Path,
) -> None:
    read_error = OSError("network drive unavailable")
    original_iterdir = pathlib.Path.iterdir

    def _iterdir(path: pathlib.Path):
        if path == tmp_path:
            raise read_error
        return original_iterdir(path)

    monkeypatch.setattr(pathlib.Path, "iterdir", _iterdir)

    file_system = _FileSystem(tmp_path)
    file_system.reload()

    assert file_system.children == []
    assert file_system.children_error is read_error


def test_explorer_filesystem_directory_probe_error_is_cached(
    monkeypatch,
    tmp_path: pathlib.Path,
) -> None:
    stale_path = tmp_path / "stale-volume"
    probe_error = TimeoutError(60, "Operation timed out", str(stale_path))
    original_is_dir = pathlib.Path.is_dir
    probe_count = 0

    def _is_dir(path: pathlib.Path) -> bool:
        nonlocal probe_count
        if path == stale_path:
            probe_count += 1
            raise probe_error
        return original_is_dir(path)

    monkeypatch.setattr(pathlib.Path, "is_dir", _is_dir)

    file_system = _FileSystem(stale_path)

    assert file_system.has_children is False
    assert file_system.can_fetch_children is False
    assert file_system.children == []
    assert file_system.children_error is probe_error

    assert file_system.has_children is False
    assert probe_count == 1


def test_explorer_model_directory_probe_error_does_not_raise(
    qtbot,
    monkeypatch,
    example_loader,
    tmp_path: pathlib.Path,
) -> None:
    explorer = _DataExplorer(root_path=tmp_path, loader_name="example")
    qtbot.addWidget(explorer)
    stale_path = tmp_path / "stale-volume"
    probe_error = TimeoutError(60, "Operation timed out", str(stale_path))
    original_is_dir = pathlib.Path.is_dir
    probe_count = 0

    def _is_dir(path: pathlib.Path) -> bool:
        nonlocal probe_count
        if path == stale_path:
            probe_count += 1
            raise probe_error
        return original_is_dir(path)

    monkeypatch.setattr(pathlib.Path, "is_dir", _is_dir)

    model = explorer._fs_model
    stale_item = _FileSystem(stale_path)
    index = model.createIndex(0, 0, stale_item)

    assert model.hasChildren(index) is False
    assert model.canFetchMore(index) is False
    assert model.hasChildren(index) is False
    assert probe_count == 1


def test_explorer_metadata_helpers_handle_edge_paths(
    qtbot,
    example_loader,
    tmp_path: pathlib.Path,
) -> None:
    explorer = _DataExplorer(root_path=tmp_path, loader_name="example")
    qtbot.addWidget(explorer)
    model = explorer._fs_model

    missing_path = tmp_path / "missing.h5"
    missing_item = _FileSystem(missing_path)

    assert model._stat_path(missing_path) is None
    assert model.data(model.createIndex(0, 1, missing_item)) == "--"
    assert model.date_modified(model.createIndex(0, 3, missing_item)) == ""
    assert model._get_sort_key_func(1)(missing_item) == -1
    assert model._get_sort_key_func(3)(missing_item) == 0
    assert (
        model._mime_type_for_path(tmp_path / "packed.pxt")
        == _IGOR_PRO_MIME_TYPES["pxt"]
    )

    model.file_system._children = []
    model.file_system._children_error = OSError("permission denied")
    assert explorer._current_directory_notice() is not None


def test_explorer_type_sort_uses_file_paths(
    qtbot,
    monkeypatch,
    example_loader,
    example_data_dir: pathlib.Path,
) -> None:
    explorer = _DataExplorer(root_path=example_data_dir, loader_name="example")
    qtbot.addWidget(explorer)
    qtbot.wait_until(lambda: explorer._tree_view.model().rowCount() > 0)

    def _fail_find_index(_item):
        raise AssertionError("type sorting should not walk model indexes")

    monkeypatch.setattr(explorer._fs_model, "_find_index", _fail_find_index)

    explorer._tree_view.sortByColumn(2, QtCore.Qt.SortOrder.AscendingOrder)


def test_explorer_loader_options_dialog_updates_kwargs(
    qtbot,
    monkeypatch,
    example_loader,
    example_data_dir: pathlib.Path,
) -> None:
    explorer = _DataExplorer(root_path=example_data_dir, loader_name="example")
    qtbot.addWidget(explorer)
    explorer._loader_kwargs_by_name["example"] = {"single": True}
    explorer._loader_extensions_by_name["example"] = {
        "coordinate_attrs": ("old_coord",)
    }

    dialogs = []
    exec_results = [False, True]

    class _FakeNameFilterDialog:
        def __init__(
            self, parent, valid_loaders, *, loader_extensions=None, sample_paths=None
        ) -> None:
            assert parent is explorer
            assert valid_loaders == {
                "example": (erlab.io.loaders["example"].load, {"single": True})
            }
            assert loader_extensions == {
                "example": {"coordinate_attrs": ("old_coord",)}
            }
            assert list(sample_paths or ()) == []
            self.checked_name = None
            dialogs.append(self)

        def check_filter(self, name_filter: str | None) -> None:
            self.checked_name = name_filter

        def exec(self) -> bool:
            return exec_results.pop(0)

        def checked_filter(self):
            return (
                "example",
                erlab.io.loaders["example"].load,
                {
                    "single": False,
                    "loader_extensions": {"additional_coords": {"gui_extra": 7.0}},
                },
            )

    monkeypatch.setattr(_dialogs, "_NameFilterDialog", _FakeNameFilterDialog)

    explorer._open_loader_options()
    assert dialogs[-1].checked_name == "example"
    assert explorer._loader_kwargs_by_name["example"] == {"single": True}
    assert explorer._loader_extensions_by_name["example"] == {
        "coordinate_attrs": ("old_coord",)
    }

    explorer._open_loader_options()
    assert dialogs[-1].checked_name == "example"
    assert explorer._loader_kwargs_by_name["example"] == {"single": False}
    assert explorer._loader_extensions_by_name["example"] == {
        "additional_coords": {"gui_extra": 7.0}
    }
