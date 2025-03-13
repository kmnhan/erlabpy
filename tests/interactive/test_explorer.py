import pathlib
import typing

from qtpy import QtCore

import erlab

if typing.TYPE_CHECKING:
    from erlab.interactive.explorer import _DataExplorer


def test_explorer(qtbot, example_loader, example_data_dir: pathlib.Path) -> None:
    erlab.interactive.imagetool.manager.main(execute=False)
    manager = erlab.interactive.imagetool.manager._manager_instance
    qtbot.add_widget(manager)

    # Initialize data explorer
    manager.ensure_explorer_initialized()
    assert hasattr(manager, "explorer")
    explorer: _DataExplorer = manager.explorer

    # Set the recent directory and name filter
    manager._recent_directory = str(example_data_dir)
    manager._recent_name_filter = next(
        iter(erlab.io.loaders["example"].file_dialog_methods.keys())
    )

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

    # Show multiple in manager
    explorer.to_manager()

    # Clear selection
    select_files([2, 3, 4], deselect=True)

    # Single selection
    select_files([2])

    # Show single in manager
    explorer.to_manager()
    qtbot.wait_until(lambda: manager.ntools == 4)

    # Show single in manager with socket
    erlab.interactive.imagetool.manager._always_use_socket = True
    explorer.to_manager()
    qtbot.wait_until(lambda: manager.ntools == 5)
    erlab.interactive.imagetool.manager._always_use_socket = False

    # Reload data in manager
    qmodelindex = manager.list_view._model._row_index(3)
    manager.list_view.selectionModel().select(
        QtCore.QItemSelection(qmodelindex, qmodelindex),
        QtCore.QItemSelectionModel.SelectionFlag.Select,
    )
    with qtbot.wait_signal(manager.get_tool(3).slicer_area.sigDataChanged):
        manager.reload_action.trigger()
    qtbot.wait(100)

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

    manager.close()
    erlab.interactive.imagetool.manager._manager_instance = None
