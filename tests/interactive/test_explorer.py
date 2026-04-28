import pathlib
import typing
from collections.abc import Callable

from qtpy import QtCore, QtGui

import erlab
from erlab.interactive.explorer._base_explorer import _DataExplorer, _ReprFetcher
from erlab.interactive.explorer._tabbed_explorer import _TabbedExplorer
from erlab.interactive.imagetool.manager import _dialogs


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
        def __init__(self, parent, valid_loaders, *, loader_extensions=None) -> None:
            assert parent is explorer
            assert valid_loaders == {
                "example": (erlab.io.loaders["example"].load, {"single": True})
            }
            assert loader_extensions == {
                "example": {"coordinate_attrs": ("old_coord",)}
            }
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
