import pathlib
import typing

from qtpy import QtCore

import erlab

if typing.TYPE_CHECKING:
    from erlab.interactive.explorer._base_explorer import _DataExplorer
    from erlab.interactive.explorer._tabbed_explorer import _TabbedExplorer


def test_explorer_general(
    qtbot, example_loader, example_data_dir: pathlib.Path
) -> None:
    erlab.interactive.imagetool.manager.main(execute=False)
    manager = erlab.interactive.imagetool.manager._manager_instance
    qtbot.add_widget(manager)

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
    manager.show_explorer()
    qtbot.wait_exposed(explorer)

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
                    lambda idx=idx_end: idx not in explorer._tree_view.selectedIndexes()
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
    qmodelindex = manager.list_view._model._row_index(3)
    manager.list_view.selectionModel().select(
        QtCore.QItemSelection(qmodelindex, qmodelindex),
        QtCore.QItemSelectionModel.SelectionFlag.Select,
    )

    # Lock levels to check if they are preserved after reloading
    manager.get_tool(3).slicer_area.lock_levels(True)
    manager.get_tool(3).slicer_area.levels = (1.0, 23.0)

    old_state = manager.get_tool(3).slicer_area.state.copy()

    with qtbot.wait_signal(manager.get_tool(3).slicer_area.sigDataChanged):
        manager.reload_action.trigger()
    qtbot.wait(200)

    assert manager.get_tool(3).slicer_area.state == old_state

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

    # Close imagetool manager
    manager.remove_all_tools()
    qtbot.wait_until(lambda: manager.ntools == 0)
    manager.close()
    erlab.interactive.imagetool.manager._manager_instance = None
