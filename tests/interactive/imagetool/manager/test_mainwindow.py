# ruff: noqa: F403, F405
from ._shared import *


def test_manager_metadata_full_code_generated_only_when_copied(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    prov = erlab.interactive.imagetool.provenance_framework
    calls: list[str] = []
    copied: list[str] = []

    def fake_derivation_code(wrapper):
        calls.append(wrapper.uid)
        return "derived = xr.DataArray([1.0])"

    monkeypatch.setattr(
        erlab.interactive.utils,
        "copy_to_clipboard",
        lambda text: copied.append(text) or text,
    )

    with manager_context() as manager:
        manager.show()
        itool(xr.DataArray([1.0], dims=("x",)), manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        wrapper = manager._imagetool_wrappers[0]
        wrapper.set_detached_provenance(
            prov.full_data(prov.RenameOperation(name="renamed")).to_replay_spec()
        )

        monkeypatch.setattr(
            type(wrapper),
            "derivation_code",
            property(fake_derivation_code),
        )
        manager._set_metadata_node(wrapper)

        assert not calls
        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        trigger_menu_action(menu, manager._metadata_copy_full_action)
        assert calls == [wrapper.uid]
        assert copied == ["derived = xr.DataArray([1.0])"]


def test_load_source_details_dialog_kwargs_editor_wraps_and_highlights(
    qtbot, tmp_path
) -> None:
    kwargs_text = (
        'engine="h5netcdf", '
        "very_long_keyword_argument_name=123, "
        'another_long_keyword_argument_name="abcdef"'
    )
    dialog = _LoadSourceDetailsDialog(
        _LoadSourceDetails(
            path=tmp_path / "scan.nc",
            loader_label="Loader",
            loader_text="xarray.load_dataarray",
            kwargs_text=kwargs_text,
            load_code=None,
        )
    )
    qtbot.addWidget(dialog)
    dialog.show()

    qtbot.wait_until(lambda: dialog.kwargs_edit._visual_row_count() > 1, timeout=2000)

    expected_rows = min(
        dialog.kwargs_edit._MAX_VISIBLE_ROWS,
        dialog.kwargs_edit._visual_row_count(),
    )
    assert dialog.kwargs_edit.height() == (
        expected_rows * dialog.kwargs_edit.fontMetrics().lineSpacing()
        + dialog.kwargs_edit._VERTICAL_PADDING
    )
    assert isinstance(
        dialog.kwargs_highlighter, erlab.interactive.utils.PythonHighlighter
    )
    dialog.kwargs_edit.setPlainText(None)
    assert dialog.kwargs_edit.toPlainText() == ""


def test_workspace_properties_dialog_actions(qtbot, monkeypatch, tmp_path) -> None:
    workspace_path = (tmp_path / "workspace.itws").resolve()
    workspace_path.write_bytes(b"itws")
    copied: list[str] = []
    revealed: list[pathlib.Path] = []

    monkeypatch.setattr(
        erlab.interactive.utils,
        "copy_to_clipboard",
        lambda content: copied.append(str(content)) or str(content),
    )
    monkeypatch.setattr(
        erlab.utils.misc,
        "open_in_file_manager",
        lambda path: revealed.append(pathlib.Path(path)),
    )

    dialog = _WorkspacePropertiesDialog(
        workspace_path,
        state=_WorkspacePropertiesState(
            is_modified=True,
            top_level_window_count=3,
        ),
    )
    qtbot.addWidget(dialog)
    dialog.show()

    assert not dialog.findChildren(QtWidgets.QLineEdit)
    path_label = dialog.findChild(
        QtWidgets.QLabel, "manager_workspace_path_value_label"
    )
    assert path_label is not None
    assert path_label.text() == str(workspace_path)
    assert path_label.toolTip() == str(workspace_path)
    assert path_label.textInteractionFlags() == (
        QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
    )
    assert dialog.value_labels["open_windows"].text() == "3"
    assert dialog.value_labels["size"].text()
    assert dialog.value_labels["modified"].text()
    assert (
        dialog.findChild(
            QtWidgets.QPlainTextEdit, "manager_workspace_dirty_details_edit"
        )
        is None
    )

    assert dialog.copy_path_button is not None
    assert dialog.reveal_button is not None

    dialog.copy_path_button.click()
    dialog.reveal_button.click()

    assert copied == [str(workspace_path)]
    assert revealed == [workspace_path]


def test_workspace_properties_dialog_without_associated_file(qtbot) -> None:
    dialog = _WorkspacePropertiesDialog(
        None,
        state=_WorkspacePropertiesState(is_modified=False, top_level_window_count=0),
    )
    qtbot.addWidget(dialog)
    dialog.show()

    assert not dialog.findChildren(QtWidgets.QLineEdit)
    assert (
        dialog.findChild(QtWidgets.QLabel, "manager_workspace_path_value_label") is None
    )
    assert dialog.value_labels["open_windows"].text() == "0"
    assert dialog.copy_path_button is None
    assert dialog.reveal_button is None
    assert (
        dialog.findChild(
            QtWidgets.QAbstractButton, "manager_copy_workspace_path_button"
        )
        is None
    )
    assert (
        dialog.findChild(
            QtWidgets.QAbstractButton, "manager_reveal_workspace_path_button"
        )
        is None
    )
    dialog._copy_path()
    dialog._reveal_path()


def test_workspace_properties_dialog_file_detail_branches(
    qtbot, monkeypatch, tmp_path
) -> None:
    missing_workspace = (tmp_path / "missing.itws").resolve()
    state = _WorkspacePropertiesState(is_modified=False, top_level_window_count=1)
    dialog = _WorkspacePropertiesDialog(
        missing_workspace,
        state=state,
    )
    qtbot.addWidget(dialog)

    assert dialog.value_labels["size"].text() == "File not found"
    assert dialog.value_labels["modified"].text() == "File not found"
    assert dialog.value_labels["open_windows"].text() == "1"
    assert dialog._status_text(missing_workspace, state) == "Associated file"

    assert manager_mainwindow._workspace_file_type_text(None) == (
        "Unsaved ImageTool workspace"
    )
    assert manager_mainwindow._workspace_file_type_text(
        tmp_path / "workspace.itws"
    ) == ("ImageTool Workspace (.itws)")
    assert manager_mainwindow._workspace_file_type_text(tmp_path / "workspace.h5") == (
        "xarray HDF5 Workspace (.h5)"
    )
    assert manager_mainwindow._workspace_file_type_text(tmp_path / "notes.txt") == (
        "TXT file"
    )
    assert manager_mainwindow._workspace_file_type_text(tmp_path / "README") == "File"

    assert manager_mainwindow._format_workspace_file_size(1) == "1 byte"
    assert manager_mainwindow._format_workspace_file_size(999) == "999 bytes"
    assert manager_mainwindow._format_workspace_file_size(1_000).startswith("1.00 KB")
    assert manager_mainwindow._format_workspace_file_size(10**16).startswith(
        "10000.00 PB"
    )

    monkeypatch.setattr(manager_mainwindow.sys, "platform", "darwin")
    assert manager_mainwindow._workspace_file_manager_action_text() == (
        "Reveal in Finder"
    )
    monkeypatch.setattr(manager_mainwindow.sys, "platform", "win32")
    assert manager_mainwindow._workspace_file_manager_action_text() == (
        "Reveal in File Explorer"
    )
    monkeypatch.setattr(manager_mainwindow.sys, "platform", "linux")
    assert manager_mainwindow._workspace_file_manager_action_text() == (
        "Open Containing Folder"
    )


@pytest.mark.parametrize("use_socket", [False, True], ids=["no_socket", "socket"])
def test_manager(
    qtbot,
    accept_dialog,
    test_data,
    use_socket,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context(use_socket=use_socket) as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        logger.info("Manager is running, adding test data")
        test_data.qshow(manager=True)

        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        logger.info("Confirmed tool is added, checking data")
        assert manager.get_imagetool(0).array_slicer.point_value(0) == 12.0

        logger.info("Checking data retrieval via fetch")

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(fetch, 0)

            # Let the GUI thread keep processing events while waiting
            qtbot.waitUntil(lambda: fut.done(), timeout=10000)

            result = fut.result()

        xr.testing.assert_identical(result, test_data)

        logger.info("Confirmed fetch works, adding more tools")
        # Add two tools
        for tool in itool(
            [test_data, test_data], link=False, execute=False, manager=False
        ):
            tool.move_to_manager()

        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)

        # Linking
        select_tools(manager, [1, 2])
        manager.link_selected()
        manager.tree_view.refresh(None)

        # Unlinking one unlinks both
        select_tools(manager, [1])
        manager.unlink_selected()
        manager.tree_view.refresh(None)
        assert not manager.get_imagetool(1).slicer_area.is_linked
        assert not manager.get_imagetool(2).slicer_area.is_linked

        # Linking again
        select_tools(manager, [1, 2])
        manager.link_selected()
        manager.tree_view.refresh(None)
        assert manager.get_imagetool(1).slicer_area.is_linked
        assert manager.get_imagetool(2).slicer_area.is_linked

        # Toggle visibility
        geometry = manager.get_imagetool(1).geometry()
        manager._imagetool_wrappers[1].hide()
        assert not manager.get_imagetool(1).isVisible()
        manager._imagetool_wrappers[1].show()
        assert manager.get_imagetool(1).geometry() == geometry

        # Removing tool
        manager.remove_imagetool(0)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        # Batch renaming
        select_tools(manager, [1, 2])

        def _handle_renaming(dialog: _RenameDialog):
            dialog._new_name_lines[1].setText("new_name_1")
            dialog._new_name_lines[2].setText("new_name_2")

        accept_dialog(manager.rename_action.trigger, pre_call=_handle_renaming)
        assert manager._imagetool_wrappers[1].name == "new_name_1"
        assert manager._imagetool_wrappers[2].name == "new_name_2"

        # Rename single
        select_tools(manager, [2], deselect=True)
        select_tools(manager, [1])
        manager.rename_action.trigger()

        qtbot.wait_until(
            lambda: (
                manager.tree_view.state()
                == QtWidgets.QAbstractItemView.State.EditingState
            ),
            timeout=5000,
        )
        delegate = manager.tree_view.itemDelegate()
        assert isinstance(delegate, _ImageToolWrapperItemDelegate)
        assert isinstance(delegate._current_editor, QtWidgets.QLineEdit)
        delegate._current_editor.setText("new_name_1_single")
        qtbot.keyClick(delegate._current_editor, QtCore.Qt.Key.Key_Return)
        qtbot.wait_until(
            lambda: manager._imagetool_wrappers[1].name == "new_name_1_single",
            timeout=5000,
        )

        # Select single tool
        select_tools(manager, [1])

        # Update info panel
        bring_manager_to_top(qtbot, manager)
        manager._update_info()

        # Batch show/hide
        select_tools(manager, [1, 2])
        manager.hide_action.trigger()
        manager.show_action.trigger()

        assert manager.get_imagetool(1).isVisible()
        assert manager.get_imagetool(2).isVisible()

        # Add third tool
        xr.concat(
            [
                manager.get_imagetool(1).slicer_area._data,
                manager.get_imagetool(2).slicer_area._data,
            ],
            "concat_dim",
        ).qshow(manager=True)
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)

        # Update info panel
        bring_manager_to_top(qtbot, manager)
        manager.tree_view.clearSelection()
        select_tools(manager, [1, 2, 3])
        manager._update_info()

        # Show goldtool
        logger.info("Opening goldtool")
        manager.get_imagetool(3).slicer_area.images[2].open_in_goldtool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[3]._childtools) == 1, timeout=5000
        )
        assert isinstance(
            next(iter(manager._imagetool_wrappers[3]._childtools.values())), GoldTool
        )
        logger.info("Confirmed goldtool is added")

        # Trigger paint event
        manager.tree_view.expandAll()

        # Test rename goldtool
        goldtool_uid: str = manager._imagetool_wrappers[3]._childtool_indices[0]

        # Bring manager to top
        manager.tree_view.clearSelection()
        bring_manager_to_top(qtbot, manager)
        select_child_tool(manager, goldtool_uid)

        manager._update_actions()
        assert manager.rename_action.isEnabled()
        manager.rename_action.trigger()
        qtbot.wait_until(
            lambda: (
                manager.tree_view.state()
                == QtWidgets.QAbstractItemView.State.EditingState
            ),
            timeout=5000,
        )
        delegate = manager.tree_view.itemDelegate()
        assert isinstance(delegate, _ImageToolWrapperItemDelegate)
        assert isinstance(delegate._current_editor, QtWidgets.QLineEdit)
        delegate._current_editor.setText("new_goldtool_name")
        qtbot.keyClick(delegate._current_editor, QtCore.Qt.Key.Key_Return)
        qtbot.wait_until(
            lambda: (
                next(
                    iter(manager._imagetool_wrappers[3]._childtools.values())
                )._tool_display_name
                == "new_goldtool_name"
            ),
            timeout=5000,
        )

        # Close goldtool
        logger.info("Closing goldtool")
        manager._remove_childtool(
            next(iter(manager._imagetool_wrappers[3]._childtools.keys()))
        )

        # Show dtool
        logger.info("Opening dtool")
        manager.get_imagetool(3).slicer_area.images[2].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[3]._childtools) == 1, timeout=5000
        )
        assert isinstance(
            next(iter(manager._imagetool_wrappers[3]._childtools.values())),
            DerivativeTool,
        )
        logger.info("Confirmed dtool is added")
        manager.tree_view.expandAll()
        tool_uid: str = manager._imagetool_wrappers[3]._childtool_indices[0]

        # Show dtool
        manager.show_childtool(tool_uid)

        # Tool and parent
        logger.info("Checking parent and childtool retrieval")
        tool, idx = manager._get_childtool_and_parent(tool_uid)
        assert isinstance(tool, DerivativeTool)
        assert idx == 3

        # Check dtool info printing
        bring_manager_to_top(qtbot, manager)
        manager.tree_view.clearSelection()
        select_child_tool(manager, tool_uid)
        manager._update_info(uid=tool_uid)

        # Duplicate dtool
        logger.info("Duplicating dtool")
        bring_manager_to_top(qtbot, manager)
        manager.tree_view.clearSelection()
        select_child_tool(manager, tool_uid)
        manager.duplicate_selected()
        manager.tree_view.refresh(None)

        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[3]._childtools) == 2, timeout=5000
        )

        # Check calling invalid indices
        logger.info("Checking invalid index handling")
        parent_qindex = manager.tree_view._model._row_index(3)
        assert manager.tree_view._model.index(1, 0, parent_qindex).isValid()
        assert not manager.tree_view._model.index(4, 0, parent_qindex).isValid()

        valid_but_wrong_pointer_type = manager.tree_view._model.createIndex(
            parent_qindex.row(), parent_qindex.column(), "invalid data"
        )
        assert not manager.tree_view._model.index(
            1, 0, valid_but_wrong_pointer_type
        ).isValid()

        # Close dtools
        logger.info("Closing dtools")
        for uid in list(manager._imagetool_wrappers[3]._childtools.keys()):
            manager._remove_childtool(uid)

        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[3]._childtools) == 0, timeout=5000
        )
        logger.info("Confirmed dtools are removed")

        # Bring manager to top
        logger.info("Testing mouse events")
        with qtbot.waitExposed(manager):
            manager.hide_all()  # Prevent windows from obstructing the manager
            manager.activateWindow()
            manager.raise_()
            manager.preview_action.setChecked(True)

        # Test mouse hover over list view
        # This may not work on all systems due to the way the mouse events are generated
        delegate._force_hover = True

        first_index = manager.tree_view.model().index(0, 0)
        first_rect_center = manager.tree_view.visualRect(first_index).center()
        qtbot.mouseMove(manager.tree_view.viewport())
        qtbot.mouseMove(manager.tree_view.viewport(), first_rect_center)
        qtbot.mouseMove(
            manager.tree_view.viewport(), first_rect_center - QtCore.QPoint(10, 10)
        )
        qtbot.mouseMove(manager.tree_view.viewport())  # move to blank should hide popup

        # Remove third tool
        select_tools(manager, [3])
        accept_dialog(manager.remove_action.trigger)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        # Test concatenate
        concat_data = xr.concat(
            [
                manager.get_imagetool(1).slicer_area._data,
                manager.get_imagetool(2).slicer_area._data,
            ],
            "concat_dim",
        )
        select_tools(manager, [1, 2])
        accept_dialog(manager.concat_action.trigger)
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)
        xr.testing.assert_identical(
            manager.get_imagetool(3).slicer_area._data, concat_data
        )

        # Test concatenate (remove originals)
        select_tools(manager, [1, 2])

        def _handle_concat(dialog: _ConcatDialog):
            dialog._remove_original_check.setChecked(True)

        accept_dialog(manager.concat_action.trigger, pre_call=_handle_concat)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        xr.testing.assert_identical(
            manager.get_imagetool(4).slicer_area._data, concat_data
        )

        # Remove all selected
        select_tools(manager, [3, 4])
        accept_dialog(manager.remove_action.trigger)
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        # Show about dialog
        accept_dialog(manager.about)


def test_remove_from_window_shortcut(
    qtbot,
    accept_dialog,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        test_data.qshow(manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        tool = manager.get_imagetool(0)

        with qtbot.waitExposed(tool):
            tool.activateWindow()
            tool.raise_()
            tool.setFocus()

        assert tool.remove_act.isVisible()

        accept_dialog(lambda: qtbot.keyClick(tool, QtCore.Qt.Key.Key_Delete))
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)


def test_remove_childtool_delete_shortcut(
    qtbot,
    accept_dialog,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        test_data.qshow(manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_dtool()

        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 1, timeout=5000
        )
        wrapper = manager._imagetool_wrappers[0]
        uid, child = next(iter(wrapper._childtools.items()))

        with qtbot.waitExposed(child):
            child.activateWindow()
            child.raise_()
            child.setFocus()

        accept_dialog(lambda: qtbot.keyClick(child, QtCore.Qt.Key.Key_Delete))
        qtbot.wait_until(lambda: uid not in wrapper._childtools, timeout=5000)


def test_manager_childtool_type_badge_only_for_tool_windows(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        named_data = test_data.rename("source")
        itool(named_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        model = typing.cast("_ImageToolWrapperItemModel", manager.tree_view.model())
        delegate = typing.cast(
            "_ImageToolWrapperItemDelegate", manager.tree_view.itemDelegate()
        )
        parent = manager._imagetool_wrappers[0]
        parent_tool = manager.get_imagetool(0)
        root_index = model._row_index(0)
        assert root_index.data(_TOOL_TYPE_ROLE) is None

        parent_tool.slicer_area.images[0].open_in_new_window()
        qtbot.wait_until(lambda: len(parent._childtool_indices) == 1, timeout=5000)
        image_uid = parent._childtool_indices[0]
        image_node = manager._child_node(image_uid)
        image_index = model._row_index(image_uid)
        assert image_node.is_imagetool
        assert image_index.data(_TOOL_TYPE_ROLE) is None

        option = QtWidgets.QStyleOptionViewItem()
        option.rect = QtCore.QRect(0, 0, 360, 25)
        option.font = manager.tree_view.font()
        option.palette = manager.tree_view.palette()
        assert delegate._compute_tool_type_info(option, image_node) == (
            None,
            None,
            None,
        )

        parent_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(lambda: len(parent._childtool_indices) == 2, timeout=5000)
        tool_uid = next(uid for uid in parent._childtool_indices if uid != image_uid)
        tool_node = manager._child_node(tool_uid)
        tool = manager.get_childtool(tool_uid)
        tool_index = model._row_index(tool_uid)
        assert isinstance(tool, DerivativeTool)
        assert tool_index.data(_TOOL_TYPE_ROLE) == tool.tool_name
        assert tool_index.data(QtCore.Qt.ItemDataRole.DisplayRole) == (
            tool._tool_display_name
        )
        assert tool_index.data(QtCore.Qt.ItemDataRole.EditRole) == (
            tool._tool_display_name
        )

        type_rect, type_text, _ = delegate._compute_tool_type_info(option, tool_node)
        assert type_rect is not None
        assert type_text == tool.tool_name

        tooltip_text = None

        def _show_tooltip(*args, **kwargs) -> None:
            nonlocal tooltip_text
            tooltip_text = args[1]

        monkeypatch.setattr(QtWidgets.QToolTip, "showText", _show_tooltip)
        help_event = QtGui.QHelpEvent(
            QtCore.QEvent.Type.ToolTip,
            type_rect.center(),
            manager.tree_view.viewport().mapToGlobal(type_rect.center()),
        )
        assert delegate.helpEvent(help_event, manager.tree_view, option, tool_index)
        assert_nonempty_tooltip(tooltip_text)

        manager.tree_view.expand(root_index)
        actual_option = QtWidgets.QStyleOptionViewItem()
        delegate.initStyleOption(actual_option, tool_index)
        actual_option.rect = manager.tree_view.visualRect(tool_index)
        actual_type_rect, _, _ = delegate._compute_tool_type_info(
            actual_option, tool_node
        )
        assert actual_type_rect is not None
        show_calls: list[str] = []
        monkeypatch.setattr(
            manager, "show_childtool", lambda uid: show_calls.append(uid)
        )
        click_tree_view_pos(manager.tree_view, actual_type_rect.center())
        assert show_calls == [tool_uid]

        editor = QtWidgets.QLineEdit(manager.tree_view.viewport())
        delegate.updateEditorGeometry(editor, option, tool_index)
        assert editor.geometry().left() > type_rect.right()
        editor.deleteLater()

        pixmap = QtGui.QPixmap(option.rect.size())
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(pixmap)
        delegate.paint(painter, option, tool_index)
        painter.end()

        assert model.setData(
            tool_index,
            "renamed_dtool",
            QtCore.Qt.ItemDataRole.EditRole,
        )
        assert tool._tool_display_name == "renamed_dtool"
        assert tool_index.data(_TOOL_TYPE_ROLE) == tool.tool_name
        assert tool_index.data(QtCore.Qt.ItemDataRole.DisplayRole) == "renamed_dtool"
        assert tool_index.data(QtCore.Qt.ItemDataRole.EditRole) == "renamed_dtool"


def test_manager_multi_data_not_shown(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool([test_data, test_data], manager=True)

        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        assert not manager.get_imagetool(0).isVisible()
        assert not manager.get_imagetool(1).isVisible()


def test_manager_replace(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        # Open a tool with the manager
        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        assert manager.get_imagetool(0).array_slicer.point_value(0) == 12.0

        # Replace data in the tool
        with qtbot.wait_signal(manager._sigDataReplaced):
            itool(test_data**2, manager=True, replace=0)

        assert manager.get_imagetool(0).array_slicer.point_value(0) == 144.0

        # Replacing 1 should create a new tool
        itool(test_data**2, manager=True, replace=1)
        qtbot.wait_until(lambda: manager.ntools == 2)
        assert manager.get_imagetool(1).array_slicer.point_value(0) == 144.0

        # Negative indexing
        with qtbot.wait_signal(manager._sigDataReplaced):
            itool(test_data, manager=True, replace=-1)

        assert manager.get_imagetool(1).array_slicer.point_value(0) == 12.0


def test_manager_childtool_source_updates(
    qtbot,
    accept_dialog,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 1, timeout=5000
        )

        wrapper = manager._imagetool_wrappers[0]
        uid, child = next(iter(wrapper._childtools.items()))
        assert isinstance(child, DerivativeTool)
        assert child.source_spec is not None

        initial = test_data.transpose("eV", "alpha")
        xr.testing.assert_identical(child.tool_data, initial)

        replaced = test_data.copy(deep=True)
        replaced.data = np.asarray(replaced.data) * 3

        with qtbot.wait_signal(manager._sigDataReplaced):
            itool(replaced, manager=True, replace=0)

        qtbot.wait_until(lambda: child.source_state == "stale", timeout=5000)
        xr.testing.assert_identical(child.tool_data, initial)

        badge_rect, badge_text, index = child_status_badge(manager, uid)
        assert badge_text == "Stale"
        delegate = typing.cast(
            "_ImageToolWrapperItemDelegate", manager.tree_view.itemDelegate()
        )
        option = QtWidgets.QStyleOptionViewItem()
        option.rect = manager.tree_view.visualRect(index)
        option.font = manager.tree_view.font()
        tooltip_text = None

        def _show_tooltip(*args, **kwargs) -> None:
            nonlocal tooltip_text
            tooltip_text = args[1]

        monkeypatch.setattr(QtWidgets.QToolTip, "showText", _show_tooltip)
        help_event = QtGui.QHelpEvent(
            QtCore.QEvent.Type.ToolTip,
            badge_rect.center(),
            manager.tree_view.viewport().mapToGlobal(badge_rect.center()),
        )
        assert delegate.helpEvent(help_event, manager.tree_view, option, index)
        assert_nonempty_tooltip(tooltip_text)
        assert index == manager.tree_view.indexAt(badge_rect.center())

        def _enable_auto_update(dialog: QtWidgets.QDialog) -> None:
            dialog.auto_update_check.setChecked(True)  # type: ignore[attr-defined]

        def _update_now(dialog: QtWidgets.QDialog) -> None:
            dialog.update_button.click()  # type: ignore[attr-defined]

        refresh_calls: list[str] = []
        original_refresh_chain = manager._refresh_source_chain_to_uid

        def _track_refresh_chain(refresh_uid: str) -> bool:
            refresh_calls.append(refresh_uid)
            return original_refresh_chain(refresh_uid)

        monkeypatch.setattr(
            manager, "_refresh_source_chain_to_uid", _track_refresh_chain
        )

        click_child_status_badge(
            manager,
            uid,
            accept_dialog,
            pre_call=_enable_auto_update,
            accept_call=_update_now,
        )

        assert refresh_calls == [uid]
        assert child.source_state == "fresh"
        assert child.source_auto_update is True
        xr.testing.assert_identical(child.tool_data, replaced.transpose("eV", "alpha"))
        auto_badge_rect, auto_badge_text, _ = child_status_badge(manager, uid)
        assert auto_badge_text == "Auto"
        assert not child._source_status_bar.isHidden()
        assert child.source_status_text == "Automatic Updates Enabled"

        tooltip_text = None
        auto_help_event = QtGui.QHelpEvent(
            QtCore.QEvent.Type.ToolTip,
            auto_badge_rect.center(),
            manager.tree_view.viewport().mapToGlobal(auto_badge_rect.center()),
        )
        assert delegate.helpEvent(auto_help_event, manager.tree_view, option, index)
        assert_nonempty_tooltip(tooltip_text)

        def _disable_auto_update(dialog: QtWidgets.QDialog) -> None:
            dialog.auto_update_check.setChecked(False)  # type: ignore[attr-defined]

        click_child_status_badge(
            manager,
            uid,
            accept_dialog,
            pre_call=_disable_auto_update,
        )

        assert child.source_state == "fresh"
        assert child.source_auto_update is False
        assert child._source_status_bar.isHidden()

        replaced2 = replaced.copy(deep=True)
        replaced2.data = np.asarray(replaced2.data) + 5

        with qtbot.wait_signal(manager._sigDataReplaced):
            itool(replaced2, manager=True, replace=0)

        qtbot.wait_until(lambda: child.source_state == "stale", timeout=5000)
        xr.testing.assert_identical(child.tool_data, replaced.transpose("eV", "alpha"))

        accept_dialog(
            lambda: child._source_status_button.click(), accept_call=_update_now
        )

        assert refresh_calls == [uid, uid]
        assert child.source_state == "fresh"
        xr.testing.assert_identical(child.tool_data, replaced2.transpose("eV", "alpha"))

        child._set_source_state("unavailable")
        qtbot.wait_until(lambda: child.source_state == "unavailable", timeout=5000)
        unavailable_badge_rect, unavailable_badge_text, _ = child_status_badge(
            manager, uid
        )
        assert isinstance(unavailable_badge_text, str)
        assert unavailable_badge_text.strip()

        tooltip_text = None
        unavailable_help_event = QtGui.QHelpEvent(
            QtCore.QEvent.Type.ToolTip,
            unavailable_badge_rect.center(),
            manager.tree_view.viewport().mapToGlobal(unavailable_badge_rect.center()),
        )
        assert delegate.helpEvent(
            unavailable_help_event, manager.tree_view, option, index
        )
        assert_nonempty_tooltip(tooltip_text)


def test_manager_reload_selected_preserves_manual_root_name(
    qtbot,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source = test_data.astype(float).rename("scan")
    file_path = tmp_path / "scan.h5"
    source.to_netcdf(file_path, engine="h5netcdf")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(
            source,
            manager=True,
            file_path=file_path,
            load_func=(xr.load_dataarray, {"engine": "h5netcdf"}, 0),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        model = typing.cast("_ImageToolWrapperItemModel", manager.tree_view.model())
        root_index = model._row_index(0)
        assert model.setData(
            root_index,
            "manual root name",
            QtCore.Qt.ItemDataRole.EditRole,
        )
        root_tool = manager.get_imagetool(0)
        assert manager.name_of_imagetool(0) == "manual root name"
        assert root_tool.slicer_area._data.name == "manual root name"
        assert root_index.data(QtCore.Qt.ItemDataRole.EditRole) == "manual root name"
        assert (
            root_index.data(QtCore.Qt.ItemDataRole.DisplayRole)
            == "0: manual root name (scan)"
        )
        assert root_tool.windowTitle() == "0: manual root name (scan)"

        updated = (source + 100.0).rename("reloaded_scan")
        updated.to_netcdf(file_path, engine="h5netcdf")

        manager.tree_view.clearSelection()
        select_tools(manager, [0])
        manager._update_actions()
        assert manager.reload_action.isVisible()

        with qtbot.wait_signal(root_tool.slicer_area.sigDataChanged, timeout=5000):
            manager.reload_selected()

        assert manager.name_of_imagetool(0) == "manual root name"
        assert root_tool.windowTitle() == "0: manual root name (scan)"
        xr.testing.assert_identical(fetch(0), updated.rename("manual root name"))


def test_manager_file_suffix_does_not_seed_unnamed_root_name(
    qtbot,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source = test_data.astype(float).rename(None)
    file_path = tmp_path / "scan.h5"
    source.to_netcdf(file_path, engine="h5netcdf")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(
            source,
            manager=True,
            file_path=file_path,
            load_func=(xr.load_dataarray, {"engine": "h5netcdf"}, 0),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        model = typing.cast("_ImageToolWrapperItemModel", manager.tree_view.model())
        root_index = model._row_index(0)

        assert manager.name_of_imagetool(0) == ""
        assert root_index.data(QtCore.Qt.ItemDataRole.EditRole) == ""
        assert root_index.data(QtCore.Qt.ItemDataRole.DisplayRole) == "0 (scan)"
        assert manager.get_imagetool(0).windowTitle() == "0 (scan)"


def test_manager_reload_selected_preserves_manual_child_imagetool_name(
    qtbot,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    prov = erlab.interactive.imagetool.provenance_framework
    source = xr.DataArray(
        np.arange(24, dtype=float).reshape((6, 4)),
        dims=["x", "y"],
        coords={"x": np.arange(6), "y": np.arange(4)},
        name="scan",
    )
    file_path = tmp_path / "scan.h5"
    source.to_netcdf(file_path, engine="h5netcdf")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(
            source,
            manager=True,
            file_path=file_path,
            load_func=(xr.load_dataarray, {"engine": "h5netcdf"}, 0),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child_tool = itool(source.copy(deep=False), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=prov.full_data(),
            source_auto_update=False,
        )

        model = typing.cast("_ImageToolWrapperItemModel", manager.tree_view.model())
        child_index = model._row_index(child_uid)
        assert model.setData(
            child_index,
            "manual child name",
            QtCore.Qt.ItemDataRole.EditRole,
        )
        child_node = manager._child_node(child_uid)
        assert child_node.name == "manual child name"
        assert child_tool.slicer_area._data.name == "manual child name"
        assert child_index.data(QtCore.Qt.ItemDataRole.EditRole) == "manual child name"
        assert (
            child_index.data(QtCore.Qt.ItemDataRole.DisplayRole) == "manual child name"
        )
        assert child_tool.windowTitle() == "manual child name"

        updated = (source + 50.0).rename("reloaded_scan")
        updated.to_netcdf(file_path, engine="h5netcdf")

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_actions()
        assert manager.reload_action.isVisible()

        with qtbot.wait_signal(child_tool.slicer_area.sigDataChanged, timeout=5000):
            manager.reload_selected()

        assert child_node.source_state == "fresh"
        assert child_node.name == "manual child name"
        assert child_tool.windowTitle() == "manual child name"
        xr.testing.assert_identical(
            fetch(child_uid), updated.rename("manual child name")
        )


def test_manager_workspace_reload_preserves_manual_root_name(
    qtbot,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source = test_data.astype(float).rename("scan")
    source_path = tmp_path / "scan.h5"
    source.to_netcdf(source_path, engine="h5netcdf")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(
            source,
            manager=True,
            file_path=source_path,
            load_func=(xr.load_dataarray, {"engine": "h5netcdf"}, 0),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        manager.rename_imagetool(0, "saved manual root")
        workspace_path = tmp_path / "manual-root-name.itws"
        manager._save_workspace_document(workspace_path, force_full=True)

        assert manager._load_workspace_file(
            workspace_path,
            replace=True,
            associate=True,
            mark_dirty=False,
            select=False,
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        assert manager.name_of_imagetool(0) == "saved manual root"
        assert manager.get_imagetool(0).slicer_area._data.name == "saved manual root"

        updated = (source + 200.0).rename("updated_scan")
        updated.to_netcdf(source_path, engine="h5netcdf")

        manager.tree_view.clearSelection()
        select_tools(manager, [0])
        with qtbot.wait_signal(
            manager.get_imagetool(0).slicer_area.sigDataChanged, timeout=5000
        ):
            manager.reload_selected()

        assert manager.name_of_imagetool(0) == "saved manual root"
        xr.testing.assert_identical(fetch(0), updated.rename("saved manual root"))


def test_manager_workspace_reload_preserves_manual_child_imagetool_name(
    qtbot,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    prov = erlab.interactive.imagetool.provenance_framework
    source = xr.DataArray(
        np.arange(24, dtype=float).reshape((6, 4)),
        dims=["x", "y"],
        coords={"x": np.arange(6), "y": np.arange(4)},
        name="scan",
    )
    source_path = tmp_path / "scan.h5"
    source.to_netcdf(source_path, engine="h5netcdf")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(
            source,
            manager=True,
            file_path=source_path,
            load_func=(xr.load_dataarray, {"engine": "h5netcdf"}, 0),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child_tool = itool(source.copy(deep=False), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=prov.full_data(),
            source_auto_update=False,
        )
        manager._child_node(child_uid).name = "saved manual child"

        workspace_path = tmp_path / "manual-child-name.itws"
        manager._save_workspace_document(workspace_path, force_full=True)

        assert manager._load_workspace_file(
            workspace_path,
            replace=True,
            associate=True,
            mark_dirty=False,
            select=False,
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        loaded_child_uid = manager._imagetool_wrappers[0]._childtool_indices[0]
        loaded_child_node = manager._child_node(loaded_child_uid)
        loaded_child_tool = manager.get_imagetool(loaded_child_uid)
        assert loaded_child_node.name == "saved manual child"
        assert loaded_child_tool.slicer_area._data.name == "saved manual child"

        updated = (source + 125.0).rename("updated_scan")
        updated.to_netcdf(source_path, engine="h5netcdf")

        manager.tree_view.clearSelection()
        select_child_tool(manager, loaded_child_uid)
        with qtbot.wait_signal(
            loaded_child_tool.slicer_area.sigDataChanged, timeout=5000
        ):
            manager.reload_selected()

        assert loaded_child_node.source_state == "fresh"
        assert loaded_child_node.name == "saved manual child"
        xr.testing.assert_identical(
            fetch(loaded_child_uid), updated.rename("saved manual child")
        )


@pytest.mark.parametrize("auto_update", [False, True], ids=["manual", "auto"])
def test_manager_reload_selected_child_tool_refreshes_from_file_parent(
    qtbot,
    tmp_path: pathlib.Path,
    test_data,
    auto_update: bool,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source = test_data.rename("scan")
    file_path = tmp_path / "scan.h5"
    source.to_netcdf(file_path, engine="h5netcdf")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(
            source,
            manager=True,
            file_path=file_path,
            load_func=(xr.load_dataarray, {"engine": "h5netcdf"}, 0),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 1, timeout=5000
        )

        child_uid = manager._imagetool_wrappers[0]._childtool_indices[0]
        child = manager.get_childtool(child_uid)
        assert isinstance(child, DerivativeTool)
        child.set_source_binding(child.source_spec, auto_update=auto_update)

        updated = source.copy(deep=True)
        updated.data = np.asarray(updated.data) + 100.0
        updated.to_netcdf(file_path, engine="h5netcdf")

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_actions()
        assert manager.reload_action.isVisible()

        with qtbot.wait_signal(child.sigDataChanged, timeout=5000):
            manager.reload_selected()

        assert child.source_state == "fresh"
        xr.testing.assert_identical(fetch(0), updated)
        xr.testing.assert_identical(child.tool_data, updated.transpose("eV", "alpha"))


def test_managed_child_tool_file_menu_reload_refreshes_file_parent(
    qtbot,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source = test_data.rename("scan")
    file_path = tmp_path / "scan.h5"
    source.to_netcdf(file_path, engine="h5netcdf")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(
            source,
            manager=True,
            file_path=file_path,
            load_func=(xr.load_dataarray, {"engine": "h5netcdf"}, 0),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 1, timeout=5000
        )

        child_uid = manager._imagetool_wrappers[0]._childtool_indices[0]
        child = manager.get_childtool(child_uid)
        assert isinstance(child, DerivativeTool)
        assert child.source_state == "fresh"

        reload_action = child.findChild(QtGui.QAction, "tool_reload_data_action")
        file_menu = child._tool_file_menu
        assert reload_action is not None
        assert child._source_status_bar.isHidden()
        file_menu.aboutToShow.emit()
        assert file_menu.menuAction().isVisible()
        assert reload_action.isVisible()
        assert reload_action.isEnabled()
        refresh_key = QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Refresh)
        assert (
            reload_action.shortcut().matches(refresh_key)
            == QtGui.QKeySequence.SequenceMatch.ExactMatch
        )

        updated = source.copy(deep=True)
        updated.data = np.asarray(updated.data) + 100.0
        updated.to_netcdf(file_path, engine="h5netcdf")

        with qtbot.wait_signal(child.sigDataChanged, timeout=5000):
            reload_action.trigger()

        assert child.source_state == "fresh"
        xr.testing.assert_identical(fetch(0), updated)
        xr.testing.assert_identical(child.tool_data, updated.transpose("eV", "alpha"))


def test_managed_nested_child_tool_file_menu_reload_refreshes_file_ancestor(
    qtbot,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    prov = erlab.interactive.imagetool.provenance_framework
    source = test_data.rename("scan")
    file_path = tmp_path / "scan.h5"
    source.to_netcdf(file_path, engine="h5netcdf")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(
            source,
            manager=True,
            file_path=file_path,
            load_func=(xr.load_dataarray, {"engine": "h5netcdf"}, 0),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child_source_spec = prov.selection(
            prov.IselOperation(kwargs={"alpha": slice(0, 4)})
        )
        child_tool = itool(
            child_source_spec.apply(source), manager=False, execute=False
        )
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=child_source_spec,
            source_auto_update=False,
        )

        child_node = manager._child_node(child_uid)
        child_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(lambda: len(child_node._childtools) == 1, timeout=5000)

        tool_uid = child_node._childtool_indices[0]
        tool = manager.get_childtool(tool_uid)
        assert isinstance(tool, DerivativeTool)
        assert child_node.source_state == "fresh"
        assert tool.source_state == "fresh"

        reload_action = tool.findChild(QtGui.QAction, "tool_reload_data_action")
        assert reload_action is not None
        assert reload_action.isEnabled()

        updated = source.copy(deep=True)
        updated.data = np.asarray(updated.data) + 100.0
        updated.to_netcdf(file_path, engine="h5netcdf")

        with qtbot.wait_signal(tool.sigDataChanged, timeout=5000):
            reload_action.trigger()

        expected_child = updated.isel(alpha=slice(0, 4))
        assert child_node.source_state == "fresh"
        assert tool.source_state == "fresh"
        xr.testing.assert_identical(fetch(0), updated)
        xr.testing.assert_identical(fetch(child_uid), expected_child)
        xr.testing.assert_identical(
            tool.tool_data, expected_child.transpose("eV", "alpha")
        )


def test_managed_child_tool_hides_reload_without_reloadable_ancestor(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 1, timeout=5000
        )

        child_uid = manager._imagetool_wrappers[0]._childtool_indices[0]
        child = manager.get_childtool(child_uid)
        assert isinstance(child, DerivativeTool)

        reload_action = child.findChild(QtGui.QAction, "tool_reload_data_action")
        assert reload_action is not None
        child._tool_file_menu.aboutToShow.emit()
        assert not child._tool_file_menu.menuAction().isVisible()
        assert not reload_action.isVisible()
        assert not reload_action.isEnabled()
        assert not child.reload_source_data()
        assert not manager._reload_source_chain_for_child(child_uid)


def test_manager_reload_selected_nested_child_refreshes_from_file_ancestor(
    qtbot,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    prov = erlab.interactive.imagetool.provenance_framework
    source = xr.DataArray(
        np.arange(24, dtype=float).reshape((6, 4)),
        dims=["x", "y"],
        coords={"x": np.arange(6), "y": np.arange(4)},
        name="scan",
    )
    file_path = tmp_path / "scan.h5"
    source.to_netcdf(file_path, engine="h5netcdf")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(
            source,
            manager=True,
            file_path=file_path,
            load_func=(xr.load_dataarray, {"engine": "h5netcdf"}, 0),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child_tool = itool(source.copy(deep=False), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=prov.full_data(),
            source_auto_update=False,
        )

        grandchild_tool = itool(
            source.isel(y=slice(0, 2)), manager=False, execute=False
        )
        assert isinstance(grandchild_tool, erlab.interactive.imagetool.ImageTool)
        grandchild_uid = manager.add_imagetool_child(
            grandchild_tool,
            child_uid,
            show=False,
            source_spec=prov.selection(prov.IselOperation(kwargs={"y": slice(0, 2)})),
            source_auto_update=False,
        )

        child_node = manager._child_node(child_uid)
        grandchild_node = manager._child_node(grandchild_uid)
        updated = source + 50.0
        updated.to_netcdf(file_path, engine="h5netcdf")

        manager.tree_view.clearSelection()
        select_child_tool(manager, grandchild_uid)
        manager._update_actions()
        assert manager.reload_action.isVisible()

        with qtbot.wait_signal(
            grandchild_tool.slicer_area.sigDataChanged, timeout=5000
        ):
            manager.reload_selected()

        assert child_node.source_state == "fresh"
        assert grandchild_node.source_state == "fresh"
        xr.testing.assert_identical(fetch(child_uid), updated)
        xr.testing.assert_identical(fetch(grandchild_uid), updated.isel(y=slice(0, 2)))


def test_manager_reload_multi_selected_children_dedupes_file_ancestor(
    qtbot,
    monkeypatch,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    prov = erlab.interactive.imagetool.provenance_framework
    source = xr.DataArray(
        np.arange(24, dtype=float).reshape((6, 4)),
        dims=["x", "y"],
        coords={"x": np.arange(6), "y": np.arange(4)},
        name="scan",
    )
    file_path = tmp_path / "scan.h5"
    source.to_netcdf(file_path, engine="h5netcdf")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(
            source,
            manager=True,
            file_path=file_path,
            load_func=(xr.load_dataarray, {"engine": "h5netcdf"}, 0),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        first_tool = itool(source.isel(x=slice(0, 2)), manager=False, execute=False)
        second_tool = itool(source.isel(x=slice(2, 4)), manager=False, execute=False)
        assert isinstance(first_tool, erlab.interactive.imagetool.ImageTool)
        assert isinstance(second_tool, erlab.interactive.imagetool.ImageTool)
        first_uid = manager.add_imagetool_child(
            first_tool,
            0,
            show=False,
            source_spec=prov.selection(prov.IselOperation(kwargs={"x": slice(0, 2)})),
            source_auto_update=False,
        )
        second_uid = manager.add_imagetool_child(
            second_tool,
            0,
            show=False,
            source_spec=prov.selection(prov.IselOperation(kwargs={"x": slice(2, 4)})),
            source_auto_update=False,
        )

        root_node = manager._imagetool_wrappers[0]
        root_area = root_node.slicer_area
        original_reload = root_area._reload
        reload_calls: list[int] = []

        def _track_reload() -> bool:
            reload_calls.append(root_node.index)
            return original_reload()

        monkeypatch.setattr(root_area, "_reload", _track_reload)

        updated = source + 100.0
        updated.to_netcdf(file_path, engine="h5netcdf")

        manager.tree_view.clearSelection()
        select_child_tool(manager, first_uid)
        select_child_tool(manager, second_uid)
        manager._update_actions()
        assert manager.reload_action.isVisible()

        manager.reload_selected()

        assert reload_calls == [0]
        qtbot.wait_until(
            lambda: (
                manager._child_node(first_uid).source_state == "fresh"
                and manager._child_node(second_uid).source_state == "fresh"
            ),
            timeout=5000,
        )
        xr.testing.assert_identical(fetch(first_uid), updated.isel(x=slice(0, 2)))
        xr.testing.assert_identical(fetch(second_uid), updated.isel(x=slice(2, 4)))


def test_manager_reload_mixed_child_selection_requires_all_children_eligible(
    qtbot,
    monkeypatch,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    prov = erlab.interactive.imagetool.provenance_framework
    source = xr.DataArray(
        np.arange(12, dtype=float).reshape((3, 4)),
        dims=["x", "y"],
        coords={"x": np.arange(3), "y": np.arange(4)},
        name="scan",
    )
    file_path = tmp_path / "scan.h5"
    source.to_netcdf(file_path, engine="h5netcdf")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(
            source,
            manager=True,
            file_path=file_path,
            load_func=(xr.load_dataarray, {"engine": "h5netcdf"}, 0),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        eligible_tool = itool(source.isel(x=slice(0, 2)), manager=False, execute=False)
        unbound_tool = itool(source.isel(x=slice(1, 3)), manager=False, execute=False)
        assert isinstance(eligible_tool, erlab.interactive.imagetool.ImageTool)
        assert isinstance(unbound_tool, erlab.interactive.imagetool.ImageTool)
        eligible_uid = manager.add_imagetool_child(
            eligible_tool,
            0,
            show=False,
            source_spec=prov.selection(prov.IselOperation(kwargs={"x": slice(0, 2)})),
            source_auto_update=False,
        )
        unbound_uid = manager.add_imagetool_child(unbound_tool, 0, show=False)

        root_node = manager._imagetool_wrappers[0]
        reload_calls: list[int] = []
        monkeypatch.setattr(
            root_node.slicer_area,
            "_reload",
            lambda: reload_calls.append(root_node.index) or True,
        )

        updated = source + 100.0
        updated.to_netcdf(file_path, engine="h5netcdf")

        manager.tree_view.clearSelection()
        select_child_tool(manager, eligible_uid)
        select_child_tool(manager, unbound_uid)
        manager._update_actions()
        assert not manager.reload_action.isVisible()

        manager.reload_selected()

        assert reload_calls == []
        xr.testing.assert_identical(fetch(0), source)
        xr.testing.assert_identical(fetch(eligible_uid), source.isel(x=slice(0, 2)))


def test_manager_selected_reload_targets_handles_stale_selection(
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        monkeypatch.setattr(
            type(manager.tree_view),
            "selected_imagetool_indices",
            property(lambda _view: []),
        )
        monkeypatch.setattr(
            type(manager.tree_view),
            "selected_childtool_uids",
            property(lambda _view: ["stale-child"]),
        )
        monkeypatch.setattr(
            manager,
            "_child_node",
            lambda _uid: (_ for _ in ()).throw(KeyError("missing")),
        )

        assert manager._selected_reload_targets() is None

        child_node = types.SimpleNamespace(has_source_binding=True)
        monkeypatch.setattr(manager, "_child_node", lambda _uid: child_node)
        monkeypatch.setattr(
            manager,
            "_parent_node",
            lambda _node: (_ for _ in ()).throw(KeyError("missing-parent")),
        )

        assert manager._selected_reload_targets() is None


def test_manager_reload_selected_skips_child_refresh_when_parent_reload_fails(
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        node = types.SimpleNamespace(
            imagetool=object(),
            slicer_area=types.SimpleNamespace(_reload=lambda: False),
        )
        refreshed: list[str] = []
        monkeypatch.setattr(
            manager, "_selected_reload_targets", lambda: ([0], {0: ["child"]})
        )
        monkeypatch.setattr(manager, "_node_for_target", lambda _target: node)
        monkeypatch.setattr(manager, "_refresh_source_chain_to_uid", refreshed.append)

        manager.reload_selected()

        assert refreshed == []


def test_manager_full_data_childtool_updates_follow_transposed_view(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.transpose_main_image()
        assert parent_tool.slicer_area.data.dims == ("eV", "alpha")

        parent_tool.slicer_area.open_in_meshtool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 1, timeout=5000
        )

        child = next(iter(manager._imagetool_wrappers[0]._childtools.values()))
        xarray.testing.assert_identical(child.tool_data, parent_tool.slicer_area.data)

        replaced = test_data.copy(deep=True)
        replaced.data = np.asarray(replaced.data) * 2

        with qtbot.wait_signal(manager._sigDataReplaced):
            itool(replaced, manager=True, replace=0)

        qtbot.wait_until(lambda: child.source_state == "stale", timeout=5000)
        assert child._update_from_parent_source() is True
        xarray.testing.assert_identical(child.tool_data, parent_tool.slicer_area.data)

        child.set_source_binding(child.source_spec, auto_update=True, state="fresh")
        replaced2 = replaced.copy(deep=True)
        replaced2.data = np.asarray(replaced2.data) + 5

        with qtbot.wait_signal(manager._sigDataReplaced):
            itool(replaced2, manager=True, replace=0)

        qtbot.wait_until(lambda: child.source_state == "fresh", timeout=5000)
        xarray.testing.assert_identical(child.tool_data, parent_tool.slicer_area.data)


def test_manager_selection_child_binding_survives_coordinate_shift_and_workspace_reload(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(4 * 5 * 3, dtype=float).reshape((4, 5, 3)),
        dims=("x", "y", "z"),
        coords={"x": np.arange(4), "y": np.arange(5), "z": np.arange(3)},
        name="scan",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(data, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.set_value(axis=2, value=1.0, cursor=0)
        parent_tool.array_slicer.set_bin(0, axis=2, value=3, update=True)
        parent_tool.slicer_area.images[0].open_in_new_window()

        root = manager._imagetool_wrappers[0]
        qtbot.wait_until(lambda: len(root._childtool_indices) == 1, timeout=5000)
        child_uid = root._childtool_indices[0]
        child_node = manager._child_node(child_uid)
        assert child_node.source_binding is not None

        tree = manager._to_datatree()
        child_attrs = tree[f"0/childtools/{child_uid}/imagetool"].attrs
        assert "manager_node_live_source_binding" in child_attrs
        assert "manager_node_live_source_spec" in child_attrs

        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)
        for node in tree.values():
            manager._load_workspace_node(typing.cast("xr.DataTree", node))
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        loaded_child = manager._child_node(child_uid)
        assert loaded_child.source_binding is not None

        shifted = data.assign_coords(z=[10.0, 11.0, 12.0])
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, shifted)

        qtbot.wait_until(lambda: loaded_child.source_state == "stale", timeout=5000)
        assert loaded_child._update_from_parent_source() is True
        child_data = manager.get_imagetool(child_uid).slicer_area._data.rename(None)
        assert not np.isnan(child_data.values).all()
        xarray.testing.assert_identical(
            child_data,
            shifted.qsel(z=11.0, z_width=3.0).rename(None),
        )


def test_manager_add_imagetool_child_materializes_source_binding_without_spec(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    prov = erlab.interactive.imagetool.provenance_framework
    data = xr.DataArray(
        np.arange(12.0).reshape(3, 4),
        dims=("x", "y"),
        coords={"x": np.arange(3.0), "y": np.arange(4.0)},
        name="scan",
    )
    source_binding = prov.ImageToolSelectionSourceBinding(
        selection_mode="isel",
        selection_indexers={"y": slice(1, 3)},
    )
    source_spec = source_binding.materialize(data)

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(data, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child = itool(source_spec.apply(data), manager=False, execute=False)
        assert isinstance(child, erlab.interactive.imagetool.ImageTool)
        qtbot.addWidget(child)
        child_uid = manager.add_imagetool_child(
            child,
            0,
            show=False,
            source_binding=source_binding,
        )
        child_node = manager._child_node(child_uid)

        assert child_node.source_binding == source_binding
        assert child_node.source_spec == source_spec


def test_manager_goldtool_output_itool_nests_under_tool(
    qtbot,
    monkeypatch,
    gold,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(gold, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child_uid = manager.add_childtool(
            GoldTool(gold.copy(deep=True), data_name="gold_input"),
            0,
            show=False,
        )
        child = manager.get_childtool(child_uid)
        assert isinstance(child, GoldTool)
        configure_goldtool_child(child, fitted=True, spline=False)

        child.open_itool()

        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)

        output_uid = child_node._childtool_indices[0]
        output_node = manager._child_node(output_uid)
        assert manager.ntools == 1
        assert output_node.is_imagetool
        assert output_node.parent_uid == child_uid
        assert output_node.output_id == "goldtool.corrected"
        assert output_node.source_spec is None
        assert output_node.provenance_spec is not None
        xr.testing.assert_identical(fetch(output_uid), child.corrected)
        manager.tree_view.clearSelection()
        select_child_tool(manager, output_uid)
        manager._update_info(uid=output_uid)
        assert metadata_derivation_texts(manager) == [
            "Start from current goldtool input data",
            "Fit and correct current data with the polynomial edge model",
        ]
        copied = copy_full_code_for_uid(monkeypatch, manager, output_uid)
        assert "era.gold.poly(" in copied
        assert "corrected = era.gold.correct_with_edge(" in copied


def test_manager_dtool_output_itool_nests_under_tool(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 1, timeout=5000
        )

        child_uid = manager._imagetool_wrappers[0]._childtool_indices[0]
        child = manager.get_childtool(child_uid)
        assert isinstance(child, DerivativeTool)

        child.open_itool()

        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)

        output_uid = child_node._childtool_indices[0]
        output_node = manager._child_node(output_uid)
        assert manager.ntools == 1
        assert output_node.is_imagetool
        assert output_node.parent_uid == child_uid
        assert output_node.output_id == "dtool.result"
        assert output_node.source_spec is None
        assert output_node.provenance_spec is not None
        xr.testing.assert_identical(fetch(output_uid), child.result.T)
        manager.tree_view.clearSelection()
        select_child_tool(manager, output_uid)
        manager._update_info(uid=output_uid)
        derivation = metadata_derivation_texts(manager)
        assert derivation[0] == "Start from selected parent ImageTool data"
        assert derivation[-2:] == [
            "Compute derivative output",
            "Transpose derivative output for ImageTool display",
        ]
        copied = copy_full_code_for_uid(monkeypatch, manager, output_uid)
        namespace = _exec_generated_code(
            copied,
            {"data": parent_tool.slicer_area.data.copy(deep=True)},
        )
        result = namespace["result"]
        assert isinstance(result, xr.DataArray)
        xr.testing.assert_identical(result, child.result.T)


def test_manager_ktool_output_itool_nests_under_tool(
    qtbot,
    monkeypatch,
    anglemap,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(anglemap, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.open_in_ktool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 1, timeout=5000
        )

        child_uid = manager._imagetool_wrappers[0]._childtool_indices[0]
        child = typing.cast("typing.Any", manager.get_childtool(child_uid))
        child.show_converted()

        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)

        output_uid = child_node._childtool_indices[0]
        output_node = manager._child_node(output_uid)
        assert manager.ntools == 1
        assert output_node.parent_uid == child_uid
        assert output_node.output_id == "ktool.converted_output"
        assert output_node.source_spec is None
        assert output_node.provenance_spec is not None
        xr.testing.assert_identical(fetch(output_uid), child._itool.slicer_area.data)
        copied = copy_full_code_for_uid(monkeypatch, manager, output_uid)
        assert ".kspace.set_normal(" in copied
        assert "_kconv = " in copied

        replaced = anglemap.copy(deep=True)
        replaced.data = np.asarray(replaced.data) + 1.0
        with qtbot.wait_signal(manager._sigDataReplaced):
            itool(replaced, link=False, manager=True, replace=0)

        qtbot.wait_until(lambda: child.source_state == "stale", timeout=5000)
        qtbot.wait_until(lambda: output_node.source_state == "stale", timeout=5000)

        child.set_source_binding(child.source_spec, auto_update=True, state="fresh")
        output_node.set_output_binding(
            typing.cast("str", output_node.output_id),
            provenance_spec=output_node.provenance_spec,
            auto_update=True,
            state="fresh",
        )

        replaced2 = replaced.copy(deep=True)
        replaced2.data = np.asarray(replaced2.data) + 2.0
        with qtbot.wait_signal(manager._sigDataReplaced):
            itool(replaced2, link=False, manager=True, replace=0)

        qtbot.wait_until(lambda: child.source_state == "fresh", timeout=5000)
        qtbot.wait_until(lambda: output_node.source_state == "fresh", timeout=5000)
        xr.testing.assert_identical(fetch(output_uid), child._converted_output())


def test_manager_ktool_output_itool_marks_stale_without_recomputing(
    qtbot,
    monkeypatch,
    anglemap,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(anglemap, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.open_in_ktool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 1, timeout=5000
        )

        child_uid = manager._imagetool_wrappers[0]._childtool_indices[0]
        child = typing.cast("typing.Any", manager.get_childtool(child_uid))
        child.show_converted()

        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)

        output_uid = child_node._childtool_indices[0]
        output_node = manager._child_node(output_uid)
        assert output_node.source_auto_update is False

        before = fetch(output_uid).copy(deep=True)
        wait_ms = int(1000 / child._UPDATE_LIMIT_HZ) + 50
        qtbot.wait(wait_ms)

        call_count = 0
        original_converted_output = child._converted_output

        def _counting_converted_output():
            nonlocal call_count
            call_count += 1
            return original_converted_output()

        monkeypatch.setattr(child, "_converted_output", _counting_converted_output)

        delta_spin = child._offset_spins["delta"]
        delta_spin.setValue(delta_spin.value() + 0.01)

        qtbot.wait_until(lambda: output_node.source_state == "stale", timeout=5000)
        qtbot.wait(wait_ms)

        assert call_count == 0
        xr.testing.assert_identical(fetch(output_uid), before)


def test_manager_reused_output_child_keeps_stale_state(
    qtbot,
    monkeypatch,
    anglemap,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(anglemap, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.open_in_ktool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 1, timeout=5000
        )

        child_uid = manager._imagetool_wrappers[0]._childtool_indices[0]
        child = typing.cast("typing.Any", manager.get_childtool(child_uid))
        child.show_converted()

        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)

        output_uid = child_node._childtool_indices[0]
        output_node = manager._child_node(output_uid)

        child.set_source_binding(child.source_spec, auto_update=False, state="stale")
        output_node.set_output_binding(
            typing.cast("str", output_node.output_id),
            provenance_spec=output_node.provenance_spec,
            auto_update=False,
            state="stale",
        )

        monkeypatch.setattr(
            child, "_prompt_existing_output_imagetool", lambda: "update"
        )
        child.show_converted()

        qtbot.wait_until(lambda: output_node.source_state == "stale", timeout=5000)
        xr.testing.assert_identical(fetch(output_uid), child._converted_output())


def test_manager_dtool_output_itool_refreshes_with_parent_updates(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 1, timeout=5000
        )

        child_uid = manager._imagetool_wrappers[0]._childtool_indices[0]
        child = manager.get_childtool(child_uid)
        assert isinstance(child, DerivativeTool)

        child.open_itool()

        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)

        output_uid = child_node._childtool_indices[0]
        output_node = manager._child_node(output_uid)

        replaced = test_data.copy(deep=True)
        replaced.data = np.asarray(replaced.data) * 2
        with qtbot.wait_signal(manager._sigDataReplaced):
            itool(replaced, link=False, manager=True, replace=0)

        qtbot.wait_until(lambda: child.source_state == "stale", timeout=5000)
        qtbot.wait_until(lambda: output_node.source_state == "stale", timeout=5000)

        child.set_source_binding(child.source_spec, auto_update=True, state="fresh")
        output_node.set_output_binding(
            typing.cast("str", output_node.output_id),
            provenance_spec=output_node.provenance_spec,
            auto_update=True,
            state="fresh",
        )

        replaced2 = replaced.copy(deep=True)
        replaced2.data = np.asarray(replaced2.data) + 5.0
        with qtbot.wait_signal(manager._sigDataReplaced):
            itool(replaced2, link=False, manager=True, replace=0)

        qtbot.wait_until(lambda: child.source_state == "fresh", timeout=5000)
        qtbot.wait_until(lambda: output_node.source_state == "fresh", timeout=5000)
        xr.testing.assert_identical(fetch(output_uid), child.result.T)


def test_manager_output_itool_auto_update_can_be_disabled_from_auto_badge(
    qtbot,
    accept_dialog,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 1, timeout=5000
        )

        child_uid = manager._imagetool_wrappers[0]._childtool_indices[0]
        child = manager.get_childtool(child_uid)
        assert isinstance(child, DerivativeTool)
        child.open_itool()

        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)
        output_uid = child_node._childtool_indices[0]
        output_node = manager._child_node(output_uid)

        child.set_source_binding(child.source_spec, auto_update=True, state="fresh")

        replaced = test_data.copy(deep=True)
        replaced.data = np.asarray(replaced.data) * 2
        with qtbot.wait_signal(manager._sigDataReplaced):
            itool(replaced, link=False, manager=True, replace=0)

        qtbot.wait_until(lambda: child.source_state == "fresh", timeout=5000)
        qtbot.wait_until(lambda: output_node.source_state == "stale", timeout=5000)

        def _enable_auto_update(dialog: QtWidgets.QDialog) -> None:
            dialog.auto_update_check.setChecked(True)  # type: ignore[attr-defined]

        def _update_now(dialog: QtWidgets.QDialog) -> None:
            dialog.update_button.click()  # type: ignore[attr-defined]

        click_child_status_badge(
            manager,
            output_uid,
            accept_dialog,
            pre_call=_enable_auto_update,
            accept_call=_update_now,
        )

        qtbot.wait_until(lambda: output_node.source_state == "fresh", timeout=5000)
        assert output_node.source_auto_update is True
        xr.testing.assert_identical(fetch(output_uid), child.result.T)
        refreshed_output = fetch(output_uid).copy(deep=True)
        _, badge_text, _ = child_status_badge(manager, output_uid)
        assert badge_text == "Auto"

        def _disable_auto_update(dialog: QtWidgets.QDialog) -> None:
            dialog.auto_update_check.setChecked(False)  # type: ignore[attr-defined]

        click_child_status_badge(
            manager,
            output_uid,
            accept_dialog,
            pre_call=_disable_auto_update,
        )
        assert output_node.source_auto_update is False

        replaced2 = replaced.copy(deep=True)
        replaced2.data = np.asarray(replaced2.data) + 5.0
        with qtbot.wait_signal(manager._sigDataReplaced):
            itool(replaced2, link=False, manager=True, replace=0)

        qtbot.wait_until(lambda: child.source_state == "fresh", timeout=5000)
        qtbot.wait_until(lambda: output_node.source_state == "stale", timeout=5000)
        xr.testing.assert_identical(fetch(output_uid), refreshed_output)
