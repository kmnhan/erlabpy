# ruff: noqa: F403, F405

from ._common import *


def _source_context_action(
    tool: FigureComposerTool, object_name: str
) -> tuple[QtWidgets.QMenu, QtGui.QAction]:
    tool._show_source_context_menu(QtCore.QPoint(0, 0))
    assert tool._source_context_menu is not None
    action = next(
        action
        for action in tool._source_context_menu.actions()
        if action.objectName() == object_name
    )
    return tool._source_context_menu, action


def test_figure_composer_source_alias_candidate_rejects_unusable_names(
    monkeypatch,
) -> None:
    assert (
        figurecomposer_sources._source_alias_candidate(
            xr.DataArray(np.arange(2), dims=("x",), name=" ")
        )
        is None
    )
    assert (
        figurecomposer_sources._source_alias_candidate(
            xr.DataArray(np.arange(2), dims=("x",), name="!!!")
        )
        is None
    )
    assert (
        figurecomposer_sources._source_alias_error(
            erlab.interactive.utils._SAVED_TOOL_DATA_NAME
        )
        is not None
    )

    class InvalidIdentifierValidator:
        def fixup(self, _text: str) -> str:
            return "bad name"

    monkeypatch.setattr(
        erlab.interactive.utils, "IdentifierValidator", InvalidIdentifierValidator
    )
    assert (
        figurecomposer_sources._source_alias_candidate(
            xr.DataArray(np.arange(2), dims=("x",), name="bad name")
        )
        is None
    )


def test_figure_composer_plot_source_move_button_uses_disabled_icon_color(
    qtbot, monkeypatch
) -> None:
    records: list[tuple[str, dict[str, typing.Any]]] = []

    def record_icon(name: str, **kwargs: typing.Any) -> QtGui.QIcon:
        records.append((name, kwargs))
        pixmap = QtGui.QPixmap(12, 12)
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        return QtGui.QIcon(pixmap)

    monkeypatch.setattr(erlab.interactive.utils.qtawesome, "icon", record_icon)
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    button = figurecomposer_plot_slices._PlotSourceMoveButton("up", parent)
    palette = button.palette()
    disabled_text = QtGui.QColor("#6f7782")
    palette.setColor(
        QtGui.QPalette.ColorGroup.Disabled,
        QtGui.QPalette.ColorRole.ButtonText,
        disabled_text,
    )
    button.setPalette(palette)
    button.setEnabled(False)

    assert records[-1][0] == "mdi6.arrow-up"
    assert records[-1][1]["color_disabled"] == disabled_text


def test_figure_composer_secondary_source_roundtrip_ignores_stale_backend_encoding(
    qtbot,
    tmp_path,
) -> None:
    primary = xr.DataArray(
        np.arange(3.0),
        dims=("x",),
        coords={"x": [0.0, 1.0, 2.0]},
        name="primary",
    )
    secondary = xr.DataArray(
        np.arange(3.0) + 10.0,
        dims=("x",),
        coords={"x": [0.0, 1.0, 2.0]},
        name="secondary",
    )
    secondary.encoding["compression"] = "unknown"
    secondary.encoding["source"] = "stale-source.nc"
    secondary.coords["x"].encoding["compression"] = "unknown"
    recipe = FigureRecipeState(
        sources=(
            FigureSourceState(name="primary", label="primary"),
            FigureSourceState(name="secondary", label="secondary"),
        ),
        primary_source="primary",
    )
    tool = FigureComposerTool(
        primary,
        recipe=recipe,
        source_data={"primary": primary, "secondary": secondary},
    )
    qtbot.addWidget(tool)

    restored = _restored_figure_composer_from_netcdf(tool, qtbot, tmp_path)

    xr.testing.assert_equal(restored.source_data()["secondary"], secondary)
    assert secondary.encoding["compression"] == "unknown"
    assert secondary.coords["x"].encoding["compression"] == "unknown"


@pytest.mark.parametrize("retain_base_data", (False, True))
def test_figure_composer_selected_source_roundtrip_applies_selection_once(
    qtbot, retain_base_data: bool
) -> None:
    base = xr.DataArray(
        np.arange(10.0),
        dims=("x",),
        coords={"x": np.arange(10.0)},
        name="profile",
    )
    selected = base.isel(x=slice(2, 8))
    source = FigureSourceState(
        name="profile",
        isel={"x": slice(2, 8)},
        selection_source="profile",
    )
    tool = FigureComposerTool.from_sources(
        {"profile": selected},
        sources=(source,),
        operations=(FigureOperationState.line(label="line", source="profile"),),
        primary_source="profile",
    )
    qtbot.addWidget(tool)
    if retain_base_data:
        tool._source_selection_base_data["profile"] = base

    restored = erlab.interactive.utils.ToolWindow.from_dataset(tool.to_dataset())
    qtbot.addWidget(restored)
    assert isinstance(restored, FigureComposerTool)
    xr.testing.assert_identical(restored.source_data()["profile"], selected)
    assert ("profile" in restored._source_selection_base_data) is retain_base_data

    restored_again = erlab.interactive.utils.ToolWindow.from_dataset(
        restored.to_dataset()
    )
    qtbot.addWidget(restored_again)
    assert isinstance(restored_again, FigureComposerTool)
    xr.testing.assert_identical(restored_again.source_data()["profile"], selected)


def test_figure_composer_selected_source_reference_restores_from_base_data(
    qtbot,
) -> None:
    base = xr.DataArray(
        np.arange(10.0),
        dims=("x",),
        coords={"x": np.arange(10.0)},
        name="profile",
    )
    selected = base.isel(x=slice(2, 8))
    tool = FigureComposerTool.from_sources(
        {"profile": selected},
        sources=(
            FigureSourceState(
                name="profile",
                isel={"x": slice(2, 8)},
                selection_source="profile",
                node_uid="profile-node",
            ),
        ),
        operations=(FigureOperationState.line(label="line", source="profile"),),
        primary_source="profile",
    )
    qtbot.addWidget(tool)

    with tool._save_tool_data_reference_context({"profile-node"}):
        saved = tool.to_dataset()
    restored = erlab.interactive.utils.ToolWindow.from_dataset(
        saved,
        _tool_data_reference_resolver=lambda _reference: base,
    )
    qtbot.addWidget(restored)
    assert isinstance(restored, FigureComposerTool)
    xr.testing.assert_identical(restored.source_data()["profile"], selected)
    xr.testing.assert_identical(restored._source_selection_base_data["profile"], base)


def test_figure_composer_selected_alias_roundtrip_uses_source_alias_base(
    qtbot,
) -> None:
    base = xr.DataArray(
        np.arange(10.0),
        dims=("x",),
        coords={"x": np.arange(10.0)},
        name="profile",
    )
    selected = base.isel(x=slice(2, 8))
    tool = FigureComposerTool.from_sources(
        {"base": base, "selected": selected},
        sources=(
            FigureSourceState(name="base"),
            FigureSourceState(
                name="selected",
                isel={"x": slice(2, 8)},
                selection_source="base",
            ),
        ),
        operations=(FigureOperationState.line(label="line", source="selected"),),
        primary_source="base",
    )
    qtbot.addWidget(tool)

    restored = erlab.interactive.utils.ToolWindow.from_dataset(tool.to_dataset())
    qtbot.addWidget(restored)
    assert isinstance(restored, FigureComposerTool)
    xr.testing.assert_identical(restored.source_data()["selected"], selected)
    xr.testing.assert_identical(restored._source_selection_base_data["selected"], base)


def test_figure_composer_persistence_metadata_fallbacks(qtbot) -> None:
    primary = xr.DataArray(np.arange(2.0), dims=("x",), name="primary")
    fallback = xr.DataArray(np.arange(2.0) + 2.0, dims=("x",), name="fallback")
    tool = FigureComposerTool.from_sources(
        {"primary": primary, "fallback": fallback},
        sources=(
            FigureSourceState(name="primary"),
            FigureSourceState(name="fallback"),
            FigureSourceState(name="missing"),
        ),
        primary_source="primary",
    )
    qtbot.addWidget(tool)
    del tool._source_data["primary"]

    items = tool._persistence_data_items()
    xr.testing.assert_identical(
        items[erlab.interactive.utils._SAVED_TOOL_DATA_NAME], fallback
    )
    assert tool._embedded_selected_source_names(xr.Dataset()) == ()

    attr = figurecomposer_tool_module._PERSISTED_SELECTED_SOURCE_DATA_ATTR
    for payload in ("{", json.dumps({"selected": True})):
        assert (
            tool._persisted_selected_source_names(xr.Dataset(attrs={attr: payload}))
            == frozenset()
        )


def test_figure_composer_source_ui_keeps_aliases_as_internal_keys(qtbot) -> None:
    first = _figure_composer_image_source("first")
    second = _figure_composer_image_source("second")
    tool = FigureComposerTool(
        first,
        recipe=FigureRecipeState(
            sources=(
                FigureSourceState(name="data_0", label="ImageTool 0: sample_map"),
                FigureSourceState(name="data_1", label="ImageTool 1: reference_map"),
            ),
            operations=(
                FigureOperationState.plot_slices(
                    label="plot_slices",
                    sources=("data_0",),
                    axes=FigureAxesSelectionState(),
                    slice_dim="eV",
                    slice_values=(0.0,),
                ),
            ),
            primary_source="data_0",
        ),
        source_data={"data_0": first, "data_1": second},
    )
    qtbot.addWidget(tool)

    first_item = tool.source_list.topLevelItem(0)
    assert first_item is not None
    assert first_item.data(0, QtCore.Qt.ItemDataRole.UserRole) == "data_0"
    assert first_item.data(0, QtCore.Qt.ItemDataRole.UserRole + 1) is True
    assert not first_item.font(0).bold()
    assert first_item.text(0) == "data_0"
    second_item = tool.source_list.topLevelItem(1)
    assert second_item is not None
    assert second_item.data(0, QtCore.Qt.ItemDataRole.UserRole + 1) is False
    assert second_item.text(0) == "data_1"

    tool._select_step_section("sources")
    checks = _plot_source_checks(tool)
    assert checks["data_0"].property("figure_source_name") == "data_0"

    assert tool.tool_status.operations[0].sources == ("data_0",)
    assert "data_0" in tool.generated_code()


def test_figure_composer_source_ui_uses_shared_shape_formatter(
    qtbot, monkeypatch
) -> None:
    data = _figure_composer_image_source("data")
    calls: list[tuple[tuple[str, ...], bool, str | None]] = []

    def _format_shape(darr: xr.DataArray, show_size: bool = False) -> str:
        calls.append((tuple(str(dim) for dim in darr.dims), show_size, darr.name))
        return "<p>formatted shape</p>"

    monkeypatch.setattr(
        erlab.utils.formatting,
        "format_darr_shape_html",
        _format_shape,
    )
    tool = FigureComposerTool(data)
    qtbot.addWidget(tool)

    first_item = tool.source_list.topLevelItem(0)
    assert first_item is not None
    shape_widget = tool.source_list.itemWidget(first_item, 1)
    assert isinstance(shape_widget, QtWidgets.QLabel)
    assert shape_widget.text() == "<p>formatted shape</p>"
    assert tool.source_list.toolTip() == ""
    assert first_item.toolTip(0)
    assert first_item.toolTip(1) == first_item.toolTip(0)
    assert shape_widget.toolTip() == ""
    assert (tuple(str(dim) for dim in data.dims), False, None) in calls
    assert any(call[1] for call in calls)


def test_figure_composer_source_refresh_controls_use_live_source_callbacks(
    qtbot,
) -> None:
    image = _figure_composer_image_source("image")
    profile = _figure_composer_profile_source("profile")
    tool = FigureComposerTool(
        image,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=1),
            sources=(
                FigureSourceState(
                    name="data_0",
                    label="ImageTool 0: image",
                    node_uid="node-0",
                ),
                FigureSourceState(name="profile", label="Detached"),
            ),
            operations=(),
            primary_source="data_0",
        ),
        source_data={"data_0": image, "profile": profile},
    )
    qtbot.addWidget(tool)

    assert tool.source_list.columnCount() == 2
    assert tool.source_list.findChildren(QtWidgets.QToolButton) == []
    assert tool.refresh_sources_button.accessibleName() == "Refresh Selected Sources"
    assert tool.refresh_sources_button.menu() is None
    assert not tool.refresh_sources_button.isEnabled()
    assert tool._source_refresh_label("data_0") is None
    tool._refresh_selected_sources_from_button()
    assert not tool.source_status_label.isHidden()

    refreshed: list[str] = []

    def can_refresh_source(name: str) -> bool:
        return name == "data_0"

    def refresh_source(name: str) -> bool:
        refreshed.append(name)
        return True

    def source_label(name: str) -> str | None:
        return "ImageTool 0: image" if name == "data_0" else None

    tool._set_source_refresh_callbacks(
        can_refresh_source=can_refresh_source,
        refresh_source=refresh_source,
        source_label=source_label,
    )

    first_item = tool.source_list.topLevelItem(0)
    second_item = tool.source_list.topLevelItem(1)
    assert first_item is not None
    assert second_item is not None
    assert first_item.icon(0).isNull()
    assert second_item.icon(0).isNull()
    assert first_item.data(0, QtCore.Qt.ItemDataRole.AccessibleDescriptionRole)
    assert tool.refresh_sources_button.isEnabled()

    tool.refresh_sources_button.click()
    assert refreshed == ["data_0"]
    assert not tool.source_status_label.isHidden()

    refreshed.clear()
    tool._refresh_all_sources_from_button()
    assert refreshed == ["data_0"]
    assert not tool.source_status_label.isHidden()

    tool._set_selected_source_names_silent({"data_0", "profile"}, "data_0")
    tool._refresh_source_controls()
    refreshed.clear()
    tool.refresh_sources_button.click()
    assert refreshed == ["data_0"]
    assert "profile" in tool.source_status_label.text()

    def raise_refresh(name: str) -> bool:
        raise RuntimeError(f"{name} is incompatible")

    tool._set_selected_source_names_silent({"data_0"}, "data_0")
    tool._set_source_refresh_callbacks(
        can_refresh_source=can_refresh_source,
        refresh_source=raise_refresh,
        source_label=source_label,
    )
    tool.refresh_sources_button.click()
    assert "incompatible" in tool.source_status_label.text()

    tool._set_source_refresh_callbacks(
        can_refresh_source=lambda _name: False,
        refresh_source=refresh_source,
        source_label=source_label,
    )
    assert not tool.refresh_sources_button.isEnabled()
    refreshed.clear()
    tool._refresh_selected_sources_from_button()
    tool._refresh_all_sources_from_button()
    assert refreshed == []

    def raise_lookup(name: str) -> bool:
        raise LookupError(name)

    def raise_lookup_label(name: str) -> str | None:
        raise LookupError(name)

    tool._set_source_refresh_callbacks(
        can_refresh_source=raise_lookup,
        refresh_source=refresh_source,
        source_label=raise_lookup_label,
    )
    assert not tool._source_refresh_available("data_0")
    assert tool._source_refresh_label("data_0") is None

    tool._set_source_refresh_callbacks(
        can_refresh_source=can_refresh_source,
        refresh_source=refresh_source,
        source_label=lambda _name: "",
    )
    assert tool._source_refresh_label("data_0") is None


def test_figure_composer_source_remove_controls_disable_used_sources(qtbot) -> None:
    image = _figure_composer_image_source("image")
    profile = _figure_composer_profile_source("profile")
    unused = xr.DataArray(
        np.array([10.0, 11.0]),
        dims=("q",),
        coords={"q": [0.0, 1.0]},
        name="unused",
    )
    tool = FigureComposerTool(
        image,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=1),
            sources=(
                FigureSourceState(name="data_0", label="ImageTool 0: image"),
                FigureSourceState(name="profile", label="Profile"),
                FigureSourceState(name="unused", label="Unused"),
            ),
            operations=(
                FigureOperationState.plot_slices(
                    label="plot_slices",
                    sources=("data_0",),
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                    slice_dim="eV",
                    slice_values=(0.0,),
                ),
                FigureOperationState.line(
                    label="profile",
                    source="profile",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ),
            ),
            primary_source="data_0",
        ),
        source_data={"data_0": image, "profile": profile, "unused": unused},
    )
    qtbot.addWidget(tool)
    tool.operation_list.setCurrentItem(tool.operation_list.topLevelItem(0))

    before_status = tool.tool_status
    before_source_data = tool.source_data()
    tool._set_selected_source_names_silent({"profile"}, "profile")
    tool._refresh_source_controls()
    tool._refresh_source_detail_panel()
    assert not tool.remove_selected_source_button.isEnabled()
    assert tool.source_detail_content.property("figureComposerSourceUsageCount") == 1
    tool._remove_selected_sources()
    assert tool.tool_status == before_status
    assert set(tool.source_data()) == set(before_source_data)
    assert not tool.remove_source("profile")
    assert not tool.remove_source("missing")

    tool._set_selected_source_names_silent({"unused"}, "unused")
    tool._refresh_source_controls()
    assert tool.remove_selected_source_button.isEnabled()
    tool.remove_selected_source_button.click()
    assert "unused" not in tool.source_data()
    assert tool.source_status_label.text() == ""
    assert tool.source_status_label.isHidden()


def test_figure_composer_source_list_has_plain_rows_and_balanced_columns(
    qtbot,
) -> None:
    data = _figure_composer_profile_source("data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(
                FigureSourceState(name="data", node_uid="node-data"),
                FigureSourceState(name="missing", node_uid="node-missing"),
            ),
            primary_source="data",
        ),
        source_data={"data": data},
    )
    qtbot.addWidget(tool)
    tool._set_source_refresh_callbacks(
        can_refresh_source=lambda _name: True,
        refresh_source=lambda _name: True,
        source_label=lambda name: name,
    )

    data_item = tool.source_list.topLevelItem(0)
    missing_item = tool.source_list.topLevelItem(1)
    assert data_item is not None
    assert missing_item is not None
    assert data_item.icon(0).isNull()
    assert missing_item.icon(0).isNull()
    assert missing_item.data(0, QtCore.Qt.ItemDataRole.AccessibleDescriptionRole)

    tool.resize(900, 650)
    tool.show()
    QtWidgets.QApplication.processEvents()
    header = tool.source_list.header()
    assert header is not None
    assert header.sectionResizeMode(0) == QtWidgets.QHeaderView.ResizeMode.Stretch
    assert header.sectionResizeMode(1) == QtWidgets.QHeaderView.ResizeMode.Interactive
    assert header.sectionSize(0) >= 100
    assert header.sectionSize(1) == 150


def test_figure_composer_remove_source_updates_state_history_and_code(qtbot) -> None:
    image = _figure_composer_image_source("image")
    extra = xr.DataArray(
        np.array([1.0, 2.0]),
        dims=("q",),
        coords={"q": [0.0, 1.0]},
        name="extra",
    )
    operation = FigureOperationState.plot_slices(
        label="plot_slices",
        sources=("data_0",),
        axes=FigureAxesSelectionState(axes=((0, 0),)),
        slice_dim="eV",
        slice_values=(0.0,),
    )
    tool = FigureComposerTool(
        image,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=1),
            sources=(
                FigureSourceState(name="data_0", label="ImageTool 0: image"),
                FigureSourceState(name="extra", label="Extra"),
            ),
            operations=(operation,),
            primary_source="extra",
        ),
        source_data={"data_0": image, "extra": extra},
    )
    qtbot.addWidget(tool)
    data_changed: list[None] = []
    info_changed: list[None] = []
    tool.sigDataChanged.connect(lambda: data_changed.append(None))
    tool.sigInfoChanged.connect(lambda: info_changed.append(None))

    assert not tool.remove_source("data_0")
    assert tool.remove_source("extra")

    assert data_changed == [None]
    assert info_changed == [None]
    assert tuple(source.name for source in tool.source_states()) == ("data_0",)
    assert set(tool.source_data()) == {"data_0"}
    assert tool.tool_status.operations == (operation,)
    assert tool.tool_status.primary_source == "data_0"
    assert tool.source_status_label.text() == ""
    assert not tool.remove_selected_source_button.isEnabled()

    _render_figure_composer_rgba(tool)
    assert tool._operation_render_errors == {}
    namespace = _exec_generated_code(tool.generated_code(), {"data_0": image})
    assert isinstance(namespace["fig"], Figure)
    restored = FigureComposerTool(
        image,
        recipe=tool.tool_status,
        source_data=tool.source_data(),
    )
    qtbot.addWidget(restored)
    assert tuple(source.name for source in restored.source_states()) == ("data_0",)
    assert set(restored.source_data()) == {"data_0"}

    assert tool.undoable
    tool.undo()
    assert tuple(source.name for source in tool.source_states()) == ("data_0", "extra")
    assert tool.tool_status.primary_source == "extra"
    xr.testing.assert_identical(tool.source_data()["extra"], extra)
    assert tool.redoable
    tool.redo()
    assert tuple(source.name for source in tool.source_states()) == ("data_0",)
    assert set(tool.source_data()) == {"data_0"}
    assert tool.tool_status.primary_source == "data_0"
    tool._refresh_source_list()
    assert tool.source_list.findChildren(QtWidgets.QToolButton) == []


def test_figure_composer_source_alias_editor_renames_references(qtbot) -> None:
    first = _figure_composer_image_source("first")
    second = _figure_composer_image_source("second")
    tool = FigureComposerTool(
        first,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=1),
            sources=(
                FigureSourceState(name="data_0", label="Legacy first"),
                FigureSourceState(name="data_1", label="Legacy second"),
            ),
            operations=(
                FigureOperationState.plot_slices(
                    label="plot_slices",
                    sources=("data_0", "data_1"),
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                    slice_dim="eV",
                    slice_values=(0.0,),
                ),
                FigureOperationState.line(
                    label="profile",
                    source="data_0",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ),
            ),
            primary_source="data_0",
        ),
        source_data={"data_0": first, "data_1": second},
    )
    qtbot.addWidget(tool)
    tool._set_selected_source_names_silent({"data_0"}, "data_0")
    tool._refresh_source_detail_panel()
    tool._refresh_source_selection_editor()

    alias_edit = tool.source_alias_edit
    assert alias_edit.text() == "data_0"
    tool._rename_source_alias("data_0", "renamed")

    assert tuple(source.name for source in tool.source_states()) == (
        "renamed",
        "data_1",
    )
    assert tool.tool_status.primary_source == "renamed"
    assert tuple(tool.source_data()) == ("renamed", "data_1")
    assert tool.tool_status.operations[0].sources == ("renamed", "data_1")
    assert tool.tool_status.operations[1].line_source == "renamed"
    assert tool.source_list.topLevelItem(0).text(0) == "renamed"
    assert "Legacy" not in tool.source_list.topLevelItem(0).text(0)

    assert tool._source_alias_error("data_1", current="renamed") is not None


def test_figure_composer_source_duplicate_and_reorder_controls(qtbot) -> None:
    first = _figure_composer_profile_source("first")
    second = _figure_composer_profile_source("second")
    third = _figure_composer_profile_source("third")
    tool = FigureComposerTool.from_sources(
        {"first": first, "second": second, "third": third},
        sources=(
            FigureSourceState(name="first"),
            FigureSourceState(name="second", isel={"x": 0}, selection_source="first"),
            FigureSourceState(name="third"),
        ),
        operations=(FigureOperationState.line(label="profile", source="first"),),
        setup=FigureSubplotsState(),
        primary_source="first",
    )
    qtbot.addWidget(tool)
    tool._source_selection_base_data["second"] = first
    tool._set_selected_source_names_silent({"second"}, "second")
    tool._refresh_source_controls()
    assert (
        tool.findChild(QtWidgets.QToolButton, "figureComposerDuplicateSourceButton")
        is None
    )
    assert (
        tool.findChild(QtWidgets.QToolButton, "figureComposerMoveSourceUpButton")
        is None
    )
    assert (
        tool.findChild(QtWidgets.QToolButton, "figureComposerMoveSourceDownButton")
        is None
    )
    assert (
        tool.source_list.dragDropMode()
        == QtWidgets.QAbstractItemView.DragDropMode.InternalMove
    )
    assert tool.source_list.defaultDropAction() == QtCore.Qt.DropAction.MoveAction
    assert tool.source_list.showDropIndicator()

    menu, move_up_action = _source_context_action(
        tool, "figureComposerContextMoveSourceUpAction"
    )
    move_down_action = next(
        action
        for action in menu.actions()
        if action.objectName() == "figureComposerContextMoveSourceDownAction"
    )
    assert move_up_action.text() == "Move Up"
    assert move_down_action.text() == "Move Down"
    assert move_up_action.isEnabled()
    assert move_down_action.isEnabled()
    menu.close()

    menu, duplicate_action = _source_context_action(
        tool, "figureComposerContextDuplicateSourceAction"
    )
    duplicate_action.trigger()
    menu.close()

    assert tuple(source.name for source in tool.source_states()) == (
        "first",
        "second",
        "second_copy",
        "third",
    )
    duplicate = tool._source_by_name()["second_copy"]
    assert duplicate.isel == {"x": 0}
    assert duplicate.selection_source == "first"
    xr.testing.assert_identical(tool.source_data()["second_copy"], second)
    xr.testing.assert_identical(tool._source_selection_base_data["second_copy"], first)
    assert tool.source_list.currentItem().text(0) == "second_copy"

    menu, move_up_action = _source_context_action(
        tool, "figureComposerContextMoveSourceUpAction"
    )
    move_up_action.trigger()
    menu.close()
    assert tuple(source.name for source in tool.source_states()) == (
        "first",
        "second_copy",
        "second",
        "third",
    )
    assert tool.source_list.currentItem().text(0) == "second_copy"

    menu, move_down_action = _source_context_action(
        tool, "figureComposerContextMoveSourceDownAction"
    )
    move_down_action.trigger()
    menu.close()
    menu, move_down_action = _source_context_action(
        tool, "figureComposerContextMoveSourceDownAction"
    )
    move_down_action.trigger()
    menu.close()
    assert tuple(source.name for source in tool.source_states()) == (
        "first",
        "second",
        "third",
        "second_copy",
    )
    assert tool.source_list.currentItem().text(0) == "second_copy"


def test_figure_composer_source_list_internal_reorder_updates_recipe(qtbot) -> None:
    data = {
        name: _figure_composer_profile_source(name) for name in ("a", "b", "c", "d")
    }
    tool = FigureComposerTool.from_sources(
        data,
        sources=tuple(FigureSourceState(name=name) for name in data),
        setup=FigureSubplotsState(),
        primary_source="a",
    )
    qtbot.addWidget(tool)
    tool._set_selected_source_names_silent({"b", "c"}, "b")

    first_moved = tool.source_list.takeTopLevelItem(1)
    second_moved = tool.source_list.takeTopLevelItem(1)
    assert first_moved is not None
    assert second_moved is not None
    tool.source_list.insertTopLevelItem(2, first_moved)
    tool.source_list.insertTopLevelItem(3, second_moved)
    tool._set_selected_source_names_silent({"b", "c"}, "b")
    tool.source_list._queue_rows_reordered()

    assert tuple(source.name for source in tool.source_states()) == (
        "a",
        "b",
        "c",
        "d",
    )
    qtbot.waitUntil(
        lambda: (
            tuple(source.name for source in tool.source_states())
            == ("a", "d", "b", "c")
        )
    )

    assert tuple(source.name for source in tool.source_states()) == (
        "a",
        "d",
        "b",
        "c",
    )
    assert tool.source_list.currentItem().text(0) == "b"
    assert {item.text(0) for item in tool.source_list.selectedItems()} == {"b", "c"}


def test_figure_composer_source_list_keyboard_context_menu(qtbot) -> None:
    data = {name: _figure_composer_profile_source(name) for name in ("a", "b")}
    tool = FigureComposerTool.from_sources(
        data,
        sources=tuple(FigureSourceState(name=name) for name in data),
        setup=FigureSubplotsState(),
        primary_source="a",
    )
    qtbot.addWidget(tool)
    tool._set_selected_source_names_silent({"b"}, "b")
    refreshed: list[str] = []

    def refresh_source(name: str) -> bool:
        refreshed.append(name)
        return True

    tool._set_source_refresh_callbacks(
        can_refresh_source=lambda _name: True,
        refresh_source=refresh_source,
    )

    event = QtGui.QKeyEvent(
        QtCore.QEvent.Type.KeyPress,
        QtCore.Qt.Key.Key_F10,
        QtCore.Qt.KeyboardModifier.ShiftModifier,
    )
    tool.source_list.keyPressEvent(event)

    assert event.isAccepted()
    assert tool._source_context_menu is not None
    action_names = {
        action.objectName()
        for action in tool._source_context_menu.actions()
        if action.objectName()
    }
    assert "figureComposerContextMoveSourceUpAction" in action_names
    assert "figureComposerContextMoveSourceDownAction" in action_names
    refresh_all_action = next(
        action
        for action in tool._source_context_menu.actions()
        if action.objectName() == "figureComposerContextRefreshAllSourcesAction"
    )
    assert refresh_all_action.isEnabled()
    refresh_all_action.trigger()
    assert refreshed == ["a", "b"]
    tool._source_context_menu.close()


def test_figure_composer_source_list_edge_events(qtbot) -> None:
    source_list = figurecomposer_tool_module._FigureComposerSourceList()
    qtbot.addWidget(source_list)
    emitted: list[tuple[object, object, object]] = []
    source_list.rows_reordered.connect(
        lambda names, selected, current: emitted.append((names, selected, current))
    )

    source_list.addTopLevelItem(QtWidgets.QTreeWidgetItem(["missing-role"]))
    source_list._emit_rows_reordered()
    source_list._queue_rows_reordered()
    source_list.keyPressEvent(None)
    source_list.dropEvent(None)
    qtbot.wait(1)

    assert emitted == []

    event = QtGui.QKeyEvent(
        QtCore.QEvent.Type.KeyPress,
        QtCore.Qt.Key.Key_A,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    source_list.keyPressEvent(event)


def test_figure_composer_source_alias_editor_commit_paths(
    qtbot, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = {name: _figure_composer_profile_source(name) for name in ("first", "second")}
    tool = FigureComposerTool.from_sources(
        data,
        sources=tuple(FigureSourceState(name=name) for name in data),
        setup=FigureSubplotsState(),
        primary_source="first",
    )
    qtbot.addWidget(tool)
    tool._set_selected_source_names_silent({"first"}, "first")
    tool._refresh_source_detail_panel()
    tool._refresh_source_selection_editor()
    alias_edit = tool.source_alias_edit

    monkeypatch.setattr(tool, "sender", lambda: alias_edit)
    alias_edit.setText("first")
    tool._commit_source_alias_edit()
    assert tuple(source.name for source in tool.source_states()) == ("first", "second")

    alias_edit.setText("second")
    tool._commit_source_alias_edit()
    assert not tool.source_validation_label.isHidden()
    assert tool.source_status_label.isHidden()
    assert alias_edit.text() == "second"

    alias_edit.setText("renamed")
    tool._commit_source_alias_edit()
    assert tuple(source.name for source in tool.source_states()) == (
        "renamed",
        "second",
    )
    assert "renamed" in tool._source_data


def test_figure_composer_remove_selected_sources_removes_available_rows(qtbot) -> None:
    data = {name: _figure_composer_profile_source(name) for name in ("first", "second")}
    tool = FigureComposerTool.from_sources(
        data,
        sources=tuple(FigureSourceState(name=name) for name in data),
        operations=(FigureOperationState.line(label="line", source="first"),),
        setup=FigureSubplotsState(),
        primary_source="first",
    )
    qtbot.addWidget(tool)
    tool._set_selected_source_names_silent({"second"}, "second")

    tool._remove_selected_sources()

    assert tuple(source.name for source in tool.source_states()) == ("first",)
    assert "second" not in tool._source_data


def test_figure_composer_source_add_callbacks_handle_unavailable_paths(
    qtbot,
) -> None:
    data = _figure_composer_profile_source("data")
    tool = FigureComposerTool(data)
    qtbot.addWidget(tool)
    figure_window = figurecomposer_widgets._FigureComposerDisplayWindow(
        FigureSubplotsState()
    )
    qtbot.addWidget(figure_window)
    tool._figure_window = figure_window
    mime = QtCore.QMimeData()
    calls: list[str] = []

    tool._set_source_add_callbacks(
        add_sources=lambda: calls.append("add") or True,
        can_drop_sources=lambda data: data is mime,
        drop_sources=lambda data: calls.append("drop") or data is mime,
    )
    assert tool._source_add_available()
    tool._request_add_sources_from_button()
    assert calls == ["add"]
    assert not tool._source_drop_available(None)
    assert tool._source_drop_available(mime)
    assert not tool._add_sources_from_mime(None)
    assert tool._add_sources_from_mime(mime)
    assert calls == ["add", "drop"]

    def raise_value() -> bool:
        raise ValueError("unavailable")

    def raise_lookup(_mime: QtCore.QMimeData) -> bool:
        raise LookupError("unavailable")

    tool._set_source_add_callbacks(
        add_sources=lambda: True,
        can_add_sources=raise_value,
        can_drop_sources=raise_lookup,
        drop_sources=raise_lookup,
    )
    assert not tool._source_add_available()
    tool._request_add_sources_from_button()
    tool._figure_window = None
    assert not tool._source_drop_available(mime)
    assert not tool._add_sources_from_mime(mime)

    tool._set_source_add_callbacks()
    assert not tool._source_add_available()
    tool._request_add_sources_from_button()


def test_figure_composer_source_structure_edge_paths(qtbot) -> None:
    data = {name: _figure_composer_profile_source(name) for name in ("first", "second")}
    tool = FigureComposerTool.from_sources(
        data,
        sources=tuple(FigureSourceState(name=name) for name in data),
        setup=FigureSubplotsState(),
        primary_source="first",
    )
    qtbot.addWidget(tool)

    def clear_source_current() -> None:
        tool.source_list.clearSelection()
        tool.source_list.setCurrentIndex(QtCore.QModelIndex())

    clear_source_current()
    tool._remove_selected_sources()
    clear_source_current()
    tool._duplicate_selected_sources()
    clear_source_current()
    tool._move_selected_sources(1)
    assert not tool._source_move_possible(1)
    assert tuple(source.name for source in tool.source_states()) == (
        "first",
        "second",
    )

    tool._set_selected_source_names_silent({"first"}, "first")
    tool._move_selected_sources(-1)
    assert tuple(source.name for source in tool.source_states()) == (
        "first",
        "second",
    )
    tool._rename_source_alias("missing", "renamed")
    assert tuple(source.name for source in tool.source_states()) == (
        "first",
        "second",
    )

    assert tool._source_alias_error("") is not None
    assert tool._source_alias_error("bad name") is not None
    assert tool._source_alias_error(erlab.interactive.utils._SAVED_TOOL_DATA_NAME)
    assert tool._source_alias_error("fig") is not None
    assert tool._source_alias_error("second", current="first") is not None
    reserved = {"first_copy"}
    assert tool._source_copy_alias("first", reserved) == "first_copy_2"
    reserved = {"first"}
    assert tool._source_unique_alias("first", reserved) == "first_2"

    tool._refresh_source_detail_panel()
    tool._refresh_source_selection_editor()
    alias_edit = tool.source_alias_edit
    alias_edit.setText("bvec")
    alias_edit.editingFinished.emit()
    assert alias_edit.text() == "bvec"
    assert not tool.source_validation_label.isHidden()
    assert tool.source_status_label.isHidden()
    tool._focus_source_alias_editor()
    first_item = tool.source_list.topLevelItem(0)
    assert first_item is not None
    tool._source_list_item_double_clicked(first_item, 0)
    tool._rename_source_alias("first", "renamed")
    assert tuple(source.name for source in tool.source_states()) == (
        "renamed",
        "second",
    )

    tool._set_selected_source_names_silent({"renamed", "second"}, "renamed")
    tool._focus_source_alias_editor()
    tool._source_list_reordered("not-a-sequence", set(), None)
    tool._source_list_reordered(("renamed", "renamed"), set(), None)
    tool._source_list_reordered(("renamed", "second"), set(), None)
    tool._source_list_reordered(("second", "renamed"), (), None)
    assert tuple(source.name for source in tool.source_states()) == (
        "second",
        "renamed",
    )


def test_figure_composer_source_selection_editor_edge_paths(qtbot) -> None:
    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0, 2.0]},
        name="data",
    )
    other = data.rename("other")
    scalar = xr.DataArray(1.0, name="scalar")
    tool = FigureComposerTool.from_sources(
        {"data": data, "other": other, "scalar": scalar},
        sources=(
            FigureSourceState(name="data", isel={"x": 0}),
            FigureSourceState(name="other", qsel={"x": 1.0}),
            FigureSourceState(name="scalar"),
        ),
        setup=FigureSubplotsState(),
        primary_source="data",
    )
    qtbot.addWidget(tool)

    def flush_deferred_editor_deletes() -> None:
        QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        QtWidgets.QApplication.processEvents()

    tool.source_list.clearSelection()
    tool.source_list.setCurrentIndex(QtCore.QModelIndex())
    tool._refresh_source_detail_panel()
    tool._refresh_source_selection_editor()
    flush_deferred_editor_deletes()
    assert tool.source_selection_controls_layout.count() == 0
    assert (
        tool.source_detail_content.property("figureComposerSourceEditorMode") == "empty"
    )

    tool._set_selected_source_names_silent({"scalar"}, "scalar")
    tool._refresh_source_detail_panel()
    tool._refresh_source_selection_editor()
    flush_deferred_editor_deletes()
    item = tool.source_selection_controls_layout.itemAt(
        tool.source_selection_controls_layout.rowCount() - 1,
        QtWidgets.QFormLayout.ItemRole.FieldRole,
    )
    assert item is not None
    assert isinstance(item.widget(), QtWidgets.QLabel)

    tool.source_list.clearSelection()
    first_item = tool.source_list.topLevelItem(0)
    second_item = tool.source_list.topLevelItem(1)
    assert first_item is not None
    assert second_item is not None
    tool.source_list.setCurrentItem(first_item)
    first_item.setSelected(True)
    second_item.setSelected(True)
    tool._refresh_source_detail_panel()
    tool._refresh_source_selection_editor()
    flush_deferred_editor_deletes()
    combo = None
    for row in range(tool.source_selection_controls_layout.rowCount()):
        item = tool.source_selection_controls_layout.itemAt(
            row, QtWidgets.QFormLayout.ItemRole.FieldRole
        )
        widget = None if item is None else item.widget()
        if widget is None or not widget.objectName().startswith(
            "figureComposerSourceSelectionDimRow"
        ):
            continue
        combo = widget.findChild(
            QtWidgets.QComboBox, "figureComposerSourceSelectionModeCombo0"
        )
        if combo is not None:
            break
    assert combo is not None
    assert combo.currentData() is _editor_controls.MIXED_VALUE
    assert "Size:" in combo.toolTip()
    for mode in ("isel", "qsel"):
        index = combo.findData(mode)
        assert index >= 0
        assert combo.itemData(index, QtCore.Qt.ItemDataRole.ToolTipRole)
    assert (
        tool.source_detail_content.property("figureComposerSourceEditorMode")
        == "multiple"
    )
    assert (
        tool.source_detail_content.property("figureComposerSourceSelectionCount") == 2
    )
    assert not tool.source_alias_controls.isVisible()
    assert tool.source_inspector.isHidden()

    tool._source_data.clear()
    tool._update_selected_source_dimension("x", "isel", "0", "")
    assert tool._source_data == {}


def test_figure_composer_source_selection_helper_edges(qtbot) -> None:
    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("eV", "kx"),
        coords={"eV": [0.0, 1.0], "kx": [0.0, 1.0, 2.0]},
        name="data",
    )
    tool = FigureComposerTool.from_sources(
        {"data": data},
        sources=(
            FigureSourceState(name="data"),
            FigureSourceState(name="derived", selection_source="data"),
        ),
        setup=FigureSubplotsState(),
        primary_source="data",
    )
    qtbot.addWidget(tool)
    assert (
        tool._source_selection_input_data("derived", tool._source_by_name()["derived"])
        is data
    )

    operation = FigureOperationState.plot_slices(
        label="slice",
        sources=("data",),
        slice_dim="eV",
    )
    normalized = FigureComposerTool._plot_slices_operation_with_legacy_qsel(
        operation,
        {"eV": 1.0, "eV_width": 0.5},
    )
    assert normalized.slice_values == (1.0,)
    assert normalized.slice_width == 0.5
    assert normalized.slice_kwargs == {}

    assert (
        FigureComposerTool._legacy_selection_fallback_source(
            FigureOperationState.line(label="line", source="data"),
            "data",
        )
        is None
    )
    assert FigureComposerTool._operation_without_map_selections(
        operation,
        "data",
    ).sources == ("data",)
    assert (
        FigureComposerTool._operation_without_map_selections(
            FigureOperationState.custom(label="custom", code="pass", trusted=True),
            None,
        ).map_selections
        == ()
    )

    tool.add_sources((FigureSourceState(name="missing"),), {})
    assert tuple(source.name for source in tool.source_states()) == ("data", "derived")
    tool.add_sources((FigureSourceState(name="data"),), {"data": data})
    assert tuple(source.name for source in tool.source_states()) == (
        "data",
        "derived",
        "data_2",
    )


def test_figure_composer_legacy_source_selection_normalization_edges(
    qtbot,
) -> None:
    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("eV", "kx"),
        coords={"eV": [0.0, 1.0], "kx": [0.0, 1.0, 2.0]},
        name="data",
    )
    selected = data.qsel(eV=1.0)
    selected.name = data.name
    reused = FigureSourceState(
        name="data_selected",
        selection_source="data",
        qsel={"eV": 1.0},
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data"), reused),
            operations=(
                FigureOperationState.plot_array(
                    label="array",
                    source="data",
                    map_selections=(
                        FigureDataSelectionState(source="data", qsel={"eV": 1.0}),
                    ),
                ),
                FigureOperationState.plot_array(
                    label="array_empty",
                    source="data",
                    map_selections=(FigureDataSelectionState(source="data"),),
                ),
                FigureOperationState.plot_array(
                    label="array_multi",
                    source="data",
                    map_selections=(
                        FigureDataSelectionState(source="data", qsel={"eV": 0.0}),
                        FigureDataSelectionState(source="missing", qsel={"eV": 0.0}),
                    ),
                ),
                FigureOperationState.plot_slices(
                    label="slice",
                    sources=("data",),
                    map_selections=(
                        FigureDataSelectionState(source="data", qsel={"eV": 1.0}),
                    ),
                    slice_dim="eV",
                ),
                FigureOperationState.plot_slices(
                    label="slice_alias",
                    sources=("data",),
                    map_selections=(
                        FigureDataSelectionState(source="data", isel={"kx": 1}),
                    ),
                    slice_dim="eV",
                ),
            ),
            primary_source="data",
        ),
        source_data={"data": data},
    )
    qtbot.addWidget(tool)

    sources = {source.name: source for source in tool.source_states()}
    assert "data_selected" in sources
    xr.testing.assert_identical(tool.source_data()["data_selected"], selected)
    assert tool.tool_status.operations[0].sources == ("data_selected",)
    assert tool.tool_status.operations[0].map_selections == ()
    assert tool.tool_status.operations[1].sources == ("data",)
    assert tool.tool_status.operations[1].map_selections == ()
    assert tool.tool_status.operations[2].sources == ("data",)
    assert tool.tool_status.operations[2].map_selections == ()
    assert tool.tool_status.operations[3].slice_values == (1.0,)
    assert tool.tool_status.operations[3].map_selections == ()
    assert tool.tool_status.operations[4].sources == ("data_selected_2",)
    assert tool.tool_status.operations[4].map_selections == ()
    assert sources["data_selected_2"].isel == {"kx": 1}


def test_figure_composer_source_display_helpers_use_alias_only() -> None:
    source = FigureSourceState(name="data_0", label="ImageTool 0: sample_map")
    assert figurecomposer_sources._source_display_label(source, "data_0") == "data_0"
    assert (
        figurecomposer_sources._source_display_tooltip(source, "data_0")
        == "Alias: data_0"
    )
    assert (
        figurecomposer_sources._source_display_label(
            FigureSourceState(name="data_0", label="ImageTool"),
            "data_0",
            disambiguate=True,
        )
        == "data_0"
    )


def test_figure_composer_replace_source_preserves_alias_and_generated_code(
    qtbot, monkeypatch
) -> None:
    original = _figure_composer_image_source("original")
    replacement = original.copy(
        data=np.asarray(original.data) + 10.0,
        deep=True,
    )
    replacement.name = "replacement"
    old_snapshot_id = "old-snapshot"
    new_snapshot_id = "new-snapshot"
    tool = FigureComposerTool(
        original,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=1),
            sources=(
                FigureSourceState(
                    name="data_0",
                    label="ImageTool 0: original",
                    node_uid="old-node",
                    node_snapshot_token=old_snapshot_id,
                    provenance_spec={"source": "old"},
                ),
            ),
            operations=(
                FigureOperationState.plot_slices(
                    label="plot_slices",
                    sources=("data_0",),
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                    slice_dim="eV",
                    slice_values=(0.0,),
                ),
                FigureOperationState.line(
                    label="profile",
                    source="data_0",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ).model_copy(
                    update={
                        "line_selection": {
                            "eV": 0.0,
                            "beta": float(original.coords["beta"].values[0]),
                        },
                        "line_x": "alpha",
                    }
                ),
            ),
            primary_source="data_0",
        ),
        source_data={"data_0": original},
    )
    qtbot.addWidget(tool)
    tool.operation_list.setCurrentItem(tool.operation_list.topLevelItem(0))

    assert "data_0" in tool.operation_list.topLevelItem(0).text(0)
    assert "data_0" in tool.operation_list.topLevelItem(1).text(0)
    assert tool.step_section_buttons["sources"].text() == "Sources: data_0"

    replaced = tool.replace_source(
        "data_0",
        FigureSourceState(
            name="data_1",
            label="ImageTool 1: replacement",
            node_uid="new-node",
            node_snapshot_token=new_snapshot_id,
            provenance_spec={"source": "new"},
        ),
        replacement,
    )

    assert replaced is True
    [source] = tool.source_states()
    assert source.name == "data_0"
    assert source.node_uid == "new-node"
    assert source.node_snapshot_token == new_snapshot_id
    assert source.provenance_spec == {"source": "new"}
    xr.testing.assert_identical(tool.source_data()["data_0"], replacement)
    assert tool.tool_status.operations[0].sources == ("data_0",)
    assert tool.tool_status.operations[1].line_source == "data_0"
    assert "data_0" in tool.operation_list.topLevelItem(0).text(0)
    assert "ImageTool 1: replacement" not in tool.operation_list.topLevelItem(0).text(0)
    assert "data_0" in tool.operation_list.topLevelItem(1).text(0)
    assert "ImageTool 1: replacement" not in tool.operation_list.topLevelItem(1).text(0)
    assert tool.step_section_buttons["sources"].text() == "Sources: data_0"

    _render_figure_composer_rgba(tool)
    assert tool._operation_render_errors == {}

    captured_maps: list[tuple[xr.DataArray, ...]] = []

    def capture_plot_slices(maps, **_kwargs):
        captured_maps.append(tuple(maps))

    monkeypatch.setattr(eplt, "plot_slices", capture_plot_slices)
    exec(tool.generated_code(), {"data_0": replacement})  # noqa: S102

    assert captured_maps
    captured = captured_maps[0][0]
    xr.testing.assert_identical(
        captured,
        replacement.sel(eV=float(captured.coords["eV"])),
    )

    assert not tool.replace_source(
        "missing",
        FigureSourceState(name="data_2", label="Missing"),
        replacement,
    )
    tool._source_data["orphan"] = original
    assert tool.replace_source(
        "orphan",
        FigureSourceState(name="data_2", label="Orphan"),
        original,
    )
    assert tuple(source.name for source in tool.source_states()) == (
        "data_0",
        "orphan",
    )
    assert not hasattr(tool.source_states()[-1], "label")


def test_figure_composer_source_refresh_applies_saved_selection(
    qtbot,
) -> None:
    original = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("eV", "alpha"),
        coords={"eV": [-1.0, 0.0], "alpha": [0.0, 1.0, 2.0]},
        name="map",
    )
    selected = original.qsel(eV=0.0)
    tool = FigureComposerTool.from_sources(
        {"data": selected},
        sources=(
            FigureSourceState(
                name="data",
                label="data",
                qsel={"eV": 0.0},
                selection_source="data",
                node_uid="node",
            ),
        ),
        operations=(FigureOperationState.plot_array(label="array", source="data"),),
        primary_source="data",
    )
    qtbot.addWidget(tool)

    replacement = original + 10.0
    assert tool.replace_source(
        "data",
        FigureSourceState(name="data", label="replacement", node_uid="node"),
        replacement,
    )
    xr.testing.assert_identical(tool.source_data()["data"], replacement.qsel(eV=0.0))
    [source] = tool.source_states()
    assert source.qsel == {"eV": 0.0}
    assert source.selection_source == "data"

    stale = tool.source_data()["data"]
    incompatible = xr.DataArray(
        np.arange(3.0),
        dims=("alpha",),
        coords={"alpha": [0.0, 1.0, 2.0]},
        name="map",
    )
    assert not tool.replace_source(
        "data",
        FigureSourceState(name="data", label="bad", node_uid="node"),
        incompatible,
    )
    xr.testing.assert_identical(tool.source_data()["data"], stale)
    assert "Could not refresh source" in tool.source_status_label.text()

    refreshed = original + 20.0
    tool.refresh_from_sources({"data": refreshed})
    xr.testing.assert_identical(tool.source_data()["data"], refreshed.qsel(eV=0.0))
    assert tool.source_status_label.text() == ""


def test_figure_composer_readding_linked_source_preserves_selection(qtbot) -> None:
    original = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("eV", "alpha"),
        coords={"eV": [-1.0, 0.0], "alpha": [0.0, 1.0, 2.0]},
        name="map",
    )
    selected = original.qsel(eV=0.0)
    tool = FigureComposerTool.from_sources(
        {"data": selected},
        sources=(
            FigureSourceState(
                name="data",
                qsel={"eV": 0.0},
                selection_source="data",
                node_uid="node",
            ),
        ),
        operations=(FigureOperationState.plot_array(label="array", source="data"),),
        primary_source="data",
    )
    qtbot.addWidget(tool)

    refreshed = original + 10.0
    tool.add_sources(
        (FigureSourceState(name="data", node_uid="node"),),
        {"data": refreshed},
    )

    [source] = tool.source_states()
    assert source.qsel == {"eV": 0.0}
    assert source.selection_source == "data"
    xr.testing.assert_identical(tool.source_data()["data"], refreshed.qsel(eV=0.0))
    xr.testing.assert_identical(tool._source_selection_base_data["data"], refreshed)

    state_before = tool.tool_status
    data_before = tool.source_data()["data"]
    incompatible = xr.DataArray(np.arange(3.0), dims=("alpha",), name="map")
    tool.add_sources(
        (FigureSourceState(name="data", node_uid="node"),),
        {"data": incompatible},
    )
    assert tool.tool_status == state_before
    xr.testing.assert_identical(tool.source_data()["data"], data_before)
    assert not tool.source_status_label.isHidden()


def test_figure_composer_replace_different_source_preserves_compatible_selection(
    qtbot,
) -> None:
    original = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("eV", "alpha"),
        coords={"eV": [-1.0, 0.0], "alpha": [0.0, 1.0, 2.0]},
        name="original",
    )
    tool = FigureComposerTool.from_sources(
        {"data": original.qsel(eV=0.0)},
        sources=(
            FigureSourceState(
                name="data",
                qsel={"eV": 0.0},
                selection_source="original_base",
                node_uid="old-node",
            ),
        ),
        operations=(FigureOperationState.plot_array(label="array", source="data"),),
        primary_source="data",
    )
    qtbot.addWidget(tool)

    replacement = (original + 20.0).rename("replacement")
    assert tool.replace_source(
        "data",
        FigureSourceState(name="replacement", node_uid="new-node"),
        replacement,
    )

    [source] = tool.source_states()
    assert source.node_uid == "new-node"
    assert source.qsel == {"eV": 0.0}
    assert source.selection_source == "data"
    xr.testing.assert_identical(tool.source_data()["data"], replacement.qsel(eV=0.0))
    xr.testing.assert_identical(tool._source_selection_base_data["data"], replacement)

    selected_replacement = (original + 40.0).rename("selected_replacement")
    assert tool.replace_source(
        "data",
        FigureSourceState(
            name="selected_replacement",
            qsel={"eV": -1.0},
            selection_source="selected_replacement",
            node_uid="newer-node",
        ),
        selected_replacement,
    )
    [source] = tool.source_states()
    assert source.qsel == {"eV": -1.0}
    assert source.selection_source == "data"
    xr.testing.assert_identical(
        tool.source_data()["data"], selected_replacement.qsel(eV=-1.0)
    )


def test_figure_composer_refresh_from_sources_covers_edge_paths(qtbot) -> None:
    first = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("eV", "alpha"),
        coords={"eV": [-1.0, 0.0], "alpha": [0.0, 1.0, 2.0]},
        name="first",
    )
    second = xr.DataArray(
        np.arange(3.0),
        dims=("alpha",),
        coords={"alpha": [0.0, 1.0, 2.0]},
        name="second",
    )
    tool = FigureComposerTool.from_sources(
        {"first": first, "second": second},
        sources=(
            FigureSourceState(
                name="first",
                qsel={"eV": 0.0},
                selection_source="first",
            ),
            FigureSourceState(name="second"),
        ),
        operations=(FigureOperationState.plot_array(label="array", source="first"),),
        primary_source="first",
    )
    qtbot.addWidget(tool)

    tool.refresh_from_sources({})
    assert tool.source_status_label.text() == ""

    incompatible = second + 10.0
    tool.refresh_from_sources({"first": incompatible})
    assert "Could not refresh source data for: first" in tool.source_status_label.text()

    refreshed_second = second + 20.0
    tool._source_selection_base_data["second"] = second
    tool.refresh_from_sources({"second": refreshed_second})
    xr.testing.assert_identical(tool.source_data()["second"], refreshed_second)
    assert "second" not in tool._source_selection_base_data
    assert tool.source_status_label.text() == ""

    extra = xr.DataArray(np.arange(2.0), dims=("x",), name="extra")
    tool.refresh_from_sources({"extra": extra})
    xr.testing.assert_identical(tool.source_data()["extra"], extra)


def test_figure_composer_batch_source_selection_skips_incompatible_sources(
    qtbot,
) -> None:
    first = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0, 2.0]},
        name="first",
    )
    second = xr.DataArray(
        np.arange(3.0),
        dims=("y",),
        coords={"y": [0.0, 1.0, 2.0]},
        name="second",
    )
    tool = FigureComposerTool.from_sources(
        {"first": first, "second": second},
        sources=(
            FigureSourceState(name="first", label="first"),
            FigureSourceState(name="second", label="second"),
        ),
        operations=(FigureOperationState.plot_array(label="array", source="first"),),
        primary_source="first",
    )
    qtbot.addWidget(tool)

    tool.source_list.clearSelection()
    first_item = tool.source_list.topLevelItem(0)
    second_item = tool.source_list.topLevelItem(1)
    assert first_item is not None
    assert second_item is not None
    first_item.setSelected(True)
    second_item.setSelected(True)

    tool._update_selected_source_dimension("x", "isel", "1", "")

    source_by_name = {source.name: source for source in tool.source_states()}
    assert source_by_name["first"].isel == {"x": 1}
    assert source_by_name["second"].isel == {}
    xr.testing.assert_identical(tool.source_data()["first"], first.isel(x=1))
    xr.testing.assert_identical(tool.source_data()["second"], second)
    assert not tool.source_validation_label.isHidden()
    assert "second" in tool.source_validation_label.text()


def test_figure_composer_source_provenance_helper_edges(qtbot) -> None:
    base_data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("eV", "alpha"),
        coords={"eV": [-1.0, 0.0], "alpha": [0.0, 1.0, 2.0]},
        name="base",
    )
    base_spec = provenance.public_data().model_dump(mode="json")
    base_source = FigureSourceState(
        name="base",
        node_uid="base-node",
        provenance_spec=base_spec,
    )
    selected_source = FigureSourceState(
        name="base_selected",
        selection_source="base",
        qsel={"eV": 0.0},
        mean_dims=("alpha",),
        provenance_spec=base_spec,
    )
    tool = FigureComposerTool.from_sources(
        {"base": base_data, "base_selected": base_data.qsel(eV=0.0).mean("alpha")},
        sources=(base_source, selected_source),
        operations=(
            FigureOperationState.plot_array(label="array", source="base_selected"),
        ),
        setup=FigureSubplotsState(),
        primary_source="base",
    )
    qtbot.addWidget(tool)

    operation_types = tuple(
        type(operation)
        for operation in tool._source_selection_replay_operations(
            selected_source.model_copy(update={"isel": {"alpha": 1}})
        )
    )
    assert operation_types == (
        provenance.IselOperation,
        provenance.QSelOperation,
        provenance.QSelAggregationOperation,
    )
    assert tool._source_code_name_candidate("ImageTool 3: data_5") == "source_5"
    assert tool._source_code_name_candidate("2 sample map") == "source_2_sample_map"
    assert tool._source_code_name_candidate("class") is None
    assert tool._source_code_name_candidate(" !!! ") is None

    used_names = {"source"}
    assert (
        tool._source_display_code_name(
            FigureSourceState(name=" !!! "), used_names=used_names
        )
        == "source_2"
    )
    assert "source_2" in used_names

    gridspec_data = base_data.rename("gs0")
    gridspec_tool = FigureComposerTool.from_sources(
        {"source_data": gridspec_data},
        sources=(FigureSourceState(name="source_data", provenance_spec=base_spec),),
        operations=(
            FigureOperationState.plot_array(label="array", source="source_data"),
        ),
        setup=FigureSubplotsState(
            layout_mode="gridspec",
            gridspec=FigureGridSpecLayoutState(
                root=FigureGridSpecGridState(
                    grid_id="root",
                    nrows=1,
                    ncols=1,
                    axes=(
                        FigureGridSpecAxesState(
                            axes_id="main",
                            span=FigureGridSpecSpanState(
                                row_start=0,
                                row_stop=1,
                                col_start=0,
                                col_stop=1,
                            ),
                        ),
                    ),
                )
            ),
        ),
        primary_source="source_data",
    )
    qtbot.addWidget(gridspec_tool)
    script_inputs, _skip_names, source_name_map = (
        gridspec_tool._display_code_source_plan()
    )
    assert script_inputs[0].name == "gs0_source"
    assert source_name_map["source_data"] == "gs0_source"

    script_input = provenance.ScriptInput(name="base", provenance_spec=base_spec)
    assert tool._script_input_with_name(script_input, "base") is script_input
    assert tool._script_input_with_name(script_input, "renamed").name == "renamed"

    renamed_input = tool._selected_source_script_input(
        base_source,
        display_name="display_base",
        source_by_name={"base": base_source},
    )
    assert renamed_input is not None
    assert renamed_input.name == "display_base"

    selected_input = tool._selected_source_script_input(
        selected_source,
        display_name="display_selected",
        source_by_name={"base": base_source, "base_selected": selected_source},
    )
    assert selected_input is not None
    assert selected_input.name == "display_selected"
    selected_spec = selected_input.parsed_provenance_spec()
    assert selected_spec is not None
    assert len(selected_spec.replay_stages) == 1
    assert len(selected_spec.replay_stages[0].operations) == 2

    assert (
        tool._selected_source_script_input(
            FigureSourceState(name="live_selected", selection_source="missing"),
            display_name="live_selected",
            source_by_name={},
        )
        is None
    )
    assert (
        tool._selected_source_script_input(
            FigureSourceState(name="invalid", node_uid="node"),
            display_name="invalid",
            source_by_name={},
        )
        is None
    )


def test_figure_composer_source_helpers_cover_selection_contract() -> None:
    unnamed = xr.DataArray(np.arange(2.0), dims=("x",), name=None)
    invalid_name = xr.DataArray(np.arange(2.0), dims=("x",), name="bad name")
    punct_name = xr.DataArray(np.arange(2.0), dims=("x",), name=" !!! ")
    keyword_name = xr.DataArray(np.arange(2.0), dims=("x",), name="class")
    leading_digit_name = xr.DataArray(np.arange(2.0), dims=("x",), name="2 sample")
    mixed_name = xr.DataArray(np.arange(2.0), dims=("x",), name="Sample-Map")
    assert figurecomposer_sources._source_name(unnamed) == "data"
    assert figurecomposer_sources._source_label(unnamed) == "data"
    assert figurecomposer_sources._source_name(invalid_name) == "bad_name"
    assert figurecomposer_sources._source_name(punct_name) == "data"
    assert figurecomposer_sources._source_name(keyword_name) == "class_"
    assert figurecomposer_sources._source_name(leading_digit_name) == "_2_sample"
    assert figurecomposer_sources._source_name(mixed_name) == "sample_map"
    reserved_name = xr.DataArray(np.arange(2.0), dims=("x",), name="profiles")
    assert figurecomposer_sources._source_name(reserved_name) == "profiles_2"
    bvec_name = xr.DataArray(np.arange(2.0), dims=("x",), name="bvec")
    assert figurecomposer_sources._source_name(bvec_name) == "bvec_2"
    assert figurecomposer_sources._source_alias_error("erlab") is not None
    assert (
        figurecomposer_sources._source_alias_error("line_color_values_norm") is not None
    )
    reserved = {"sample_map"}
    assert figurecomposer_sources._source_unique_name("sample_map", reserved) == (
        "sample_map_2"
    )
    assert "sample_map_2" in reserved
    reserved = set()
    assert figurecomposer_sources._source_unique_name("fig", reserved) == "fig_2"
    reserved = set()
    assert (
        figurecomposer_sources._source_unique_name("profiles", reserved) == "profiles_2"
    )
    reserved = set()
    assert figurecomposer_sources._source_unique_name("gs0", reserved) == "gs0_source"
    assert figurecomposer_sources._source_display_tooltip(None, "data_0") == (
        "Alias: data_0"
    )
    assert (
        "Original name: bad name"
        in figurecomposer_source_inspector.source_metadata_tooltip(
            FigureSourceState(name="bad_name"),
            "bad_name",
            invalid_name,
        )
    )
    with pytest.raises(ValueError, match="not a valid variable"):
        figurecomposer_sources._valid_source_variable("bad name")
    assert figurecomposer_plot_slices._source_names(
        FigureOperationState.plot_slices(
            label="mapped",
            sources=("extra",),
            map_selections=(FigureDataSelectionState(source="extra"),),
        )
    ) == ("extra",)

    method = FigureOperationState.method(
        family=FigureMethodFamily.AXES,
        name="set",
        args=(slice(1, 5, 2),),
        kwargs={
            "indexer": slice(None, 3),
            "metadata": {"kind": "slice", "start": 1, "stop": 2},
        },
    )
    restored_method = FigureOperationState.model_validate_json(method.model_dump_json())
    assert restored_method.method_args == (slice(1, 5, 2),)
    assert restored_method.method_kwargs["indexer"] == slice(None, 3)
    assert restored_method.method_kwargs["metadata"] == {
        "kind": "slice",
        "start": 1,
        "stop": 2,
    }
    legacy_source = FigureSourceState.model_validate(
        {"name": "legacy", "isel": {"x": {"kind": "slice", "start": 1}}}
    )
    assert legacy_source.isel == {"x": slice(1, None)}

    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [10.0, 20.0, 30.0]},
        name="map",
    )
    assert figurecomposer_sources._available_source_dims(
        {"data": data}, ("missing", "data")
    ) == ["x", "y"]
    assert (
        figurecomposer_sources._selected_data(
            {"data": data}, FigureDataSelectionState(source="missing")
        )
        is None
    )
    selected = figurecomposer_sources._selected_data(
        {"data": data},
        FigureDataSelectionState(
            source="data",
            isel={"x": {"kind": "slice", "start": 0, "stop": 1}},
            qsel={"y": 20.0},
            mean_dims=("x",),
        ),
    )
    assert selected is not None
    assert selected.ndim == 0
    assert float(selected) == 1.0

    selected_multi_mean = figurecomposer_sources._selected_data(
        {"data": data},
        FigureDataSelectionState(source="data", mean_dims=("x", "y")),
    )
    assert selected_multi_mean is not None
    assert selected_multi_mean.ndim == 0

    nonuniform_public = xr.DataArray(
        np.arange(24.0).reshape(2, 3, 4),
        dims=("alpha", "eV", "sample_temp"),
        coords={
            "alpha": [0.0, 1.0],
            "eV": [-0.1, 0.0, 0.1],
            "sample_temp": [10.0, 15.0, 30.0, 60.0],
        },
        name="map",
    )
    nonuniform_internal = erlab.interactive.imagetool.slicer.make_dims_uniform(
        nonuniform_public
    )
    assert nonuniform_internal.dims == ("alpha", "eV", "sample_temp_idx")
    assert figurecomposer_sources._available_source_dims(
        {"data": nonuniform_internal}, ("data",)
    ) == ["alpha", "eV", "sample_temp"]
    nonuniform_selected = figurecomposer_sources._selected_data(
        {"data": nonuniform_internal},
        FigureDataSelectionState(
            source="data",
            isel={"sample_temp": {"kind": "slice", "start": 1, "stop": 3}},
            mean_dims=("sample_temp",),
        ),
    )
    assert nonuniform_selected is not None
    xr.testing.assert_identical(
        nonuniform_selected,
        nonuniform_public.isel(sample_temp=slice(1, 3)).qsel.mean("sample_temp"),
    )

    assert (
        figurecomposer_sources._middle_coord_value(
            xr.DataArray([], dims=("x",), coords={"x": []}), "x"
        )
        is None
    )
    assert (
        figurecomposer_sources._middle_coord_value(
            xr.DataArray([1, 2], dims=("x",), coords={"x": ["a", "b"]}), "x"
        )
        is None
    )


def test_figure_composer_raw_sources_use_public_nonuniform_dims(qtbot) -> None:
    public = xr.DataArray(
        np.arange(24.0).reshape(2, 3, 4),
        dims=("alpha", "eV", "sample_temp"),
        coords={
            "alpha": [0.0, 1.0],
            "eV": [-0.1, 0.0, 0.1],
            "sample_temp": [10.0, 15.0, 30.0, 60.0],
        },
        name="map",
    )
    internal = erlab.interactive.imagetool.slicer.make_dims_uniform(public)
    operation = FigureOperationState.plot_slices(
        label="plot_slices",
        sources=("data",),
    )

    tool = FigureComposerTool.from_sources(
        {"data": internal},
        sources=(FigureSourceState(name="data", label="map"),),
        operations=(operation,),
        setup=FigureSubplotsState(),
        primary_source="data",
    )
    qtbot.addWidget(tool)

    shape = figurecomposer_plot_slices._plot_slices_shape(tool, operation)
    assert "sample_temp" in shape.source_text
    assert "sample_temp_idx" not in shape.source_text
    shape_item = tool.source_list.topLevelItem(0)
    assert shape_item is not None
    shape_label = tool.source_list.itemWidget(shape_item, 1)
    assert isinstance(shape_label, QtWidgets.QLabel)
    assert "sample_temp_idx" not in shape_label.text()


def test_figure_composer_source_inspector_is_sources_tab_compact(
    qtbot, monkeypatch
) -> None:
    first = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("eV", "kx"),
        coords={"eV": [-0.1, 0.1], "kx": [0.0, 0.5, 1.0]},
        attrs={"sample": "TiSe2", "long": "x" * 200},
        name="original map",
    )
    second = xr.DataArray(
        np.arange(4.0),
        dims=("delay",),
        coords={"delay": [0.0, 1.0, 2.0, 3.0]},
        attrs={"scan": 12},
        name="profile",
    )
    format_calls: list[tuple[tuple[str, ...], dict[str, typing.Any]]] = []

    def fake_format_darr_html(
        data: xr.DataArray,
        **kwargs: typing.Any,
    ) -> str:
        format_calls.append((tuple(str(dim) for dim in data.dims), kwargs))
        return "<p>formatted details</p>"

    monkeypatch.setattr(
        erlab.utils.formatting,
        "format_darr_html",
        fake_format_darr_html,
    )
    tool = FigureComposerTool.from_sources(
        {"map": first, "profile": second},
        sources=(
            FigureSourceState(name="map", label="Map source", node_uid="node-map"),
            FigureSourceState(name="profile", label="Profile source"),
        ),
        operations=(FigureOperationState.plot_slices(label="maps", sources=("map",)),),
        primary_source="map",
    )
    qtbot.addWidget(tool)

    assert tool.source_splitter.orientation() == QtCore.Qt.Orientation.Horizontal
    assert not tool.source_splitter.childrenCollapsible()
    assert tool.source_inspector.parentWidget() is tool.source_detail_content
    assert tool.source_detail_scroll.widget() is tool.source_detail_content
    assert tool.editor_tabs.indexOf(tool.sources_page) == 0
    assert not hasattr(tool, "step_detail_splitter")
    assert tool.source_inspector.source_name() == "map"
    assert tool.source_inspector.property("figureComposerSourceAlias") == "map"
    assert tool.source_inspector.property("figureComposerSourceDims") == ("eV", "kx")
    assert tool.source_inspector.property("figureComposerSourceDtype") == "float64"
    summary_html = tool.source_inspector.subtitle_label.text()
    assert summary_html.startswith("original map<br>")
    assert "Original name:" not in summary_html
    assert "<p>" not in summary_html
    assert "<br><br>" not in summary_html
    assert not tool.source_inspector.details_button.isChecked()
    assert not tool.source_inspector.details_label.isVisibleTo(tool.source_inspector)
    assert tool.source_inspector.details_html() == ""
    assert format_calls == []
    assert (
        tool.source_detail_content.property("figureComposerSourceEditorMode")
        == "single"
    )
    assert tool.source_detail_content.property("figureComposerSourceUsageCount") == 1
    assert tool.source_alias_edit.text() == "map"
    assert (
        tool.source_list.selectionMode()
        == QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
    )
    assert (
        tool.source_list.selectionBehavior()
        == QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
    )
    first_item = tool.source_list.topLevelItem(0)
    assert first_item is not None
    assert first_item.flags() & QtCore.Qt.ItemFlag.ItemIsSelectable
    assert tool.source_list.selectedItems() == [first_item]
    shape_label = tool.source_list.itemWidget(first_item, 1)
    assert isinstance(shape_label, QtWidgets.QLabel)
    assert shape_label.testAttribute(
        QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents
    )

    tool.source_inspector.details_button.setChecked(True)
    assert tool.source_inspector.details_label.isVisibleTo(tool.source_inspector)
    assert tool.source_inspector.property("figureComposerSourceDetailsExpanded") is True
    assert tool.source_inspector.details_html() == "<p>formatted details</p>"
    assert format_calls == [(("eV", "kx"), {"show_size": True, "show_summary": False})]
    tool.source_inspector.details_button.setChecked(False)
    tool.source_inspector.details_button.setChecked(True)
    assert len(format_calls) == 1

    second_item = tool.source_list.topLevelItem(1)
    assert second_item is not None
    tool.source_list.clearSelection()
    tool.source_list.setCurrentItem(second_item)
    assert tool.source_list.selectedItems() == [second_item]
    assert tool.source_inspector.source_name() == "profile"
    assert tool.source_inspector.property("figureComposerSourceAlias") == "profile"
    assert tool.source_inspector.property("figureComposerSourceDims") == ("delay",)
    assert len(format_calls) == 2


def test_figure_composer_source_inspector_details_show_coord_values(qtbot) -> None:
    data = xr.DataArray(
        np.arange(3.0),
        dims=("x",),
        coords={"x": [0.0, 1.0, 2.0], "temperature": 20.0},
        name="profile",
    )
    tool = FigureComposerTool.from_sources(
        {"profile": data},
        sources=(FigureSourceState(name="profile", label="Profile"),),
        primary_source="profile",
    )
    qtbot.addWidget(tool)

    tool.source_inspector.details_button.setChecked(True)
    html = tool.source_inspector.details_html()

    assert "0 : 1 : 2" in html
    assert ">20</td>" in html
    assert "float64 [3]" not in html
    assert "float64 scalar" not in html


def test_figure_composer_source_inspector_details_fallback_is_metadata_only(
    qtbot, monkeypatch
) -> None:
    data = xr.DataArray(
        np.arange(3.0),
        dims=("x",),
        coords={"x": [0.0, 1.0, 2.0]},
        name="profile",
    )
    format_calls: list[dict[str, typing.Any]] = []

    def raise_load_coords(data: xr.DataArray) -> xr.DataArray:
        raise RuntimeError("coordinate metadata unavailable")

    def fake_format_darr_html(
        data: xr.DataArray,
        **kwargs: typing.Any,
    ) -> str:
        format_calls.append(kwargs)
        return "<p>fallback details</p>"

    monkeypatch.setattr(
        figurecomposer_source_inspector,
        "_source_data_with_loaded_coords",
        raise_load_coords,
    )
    monkeypatch.setattr(
        erlab.utils.formatting,
        "format_darr_html",
        fake_format_darr_html,
    )

    tool = FigureComposerTool.from_sources(
        {"profile": data},
        sources=(FigureSourceState(name="profile", label="Profile"),),
        primary_source="profile",
    )
    qtbot.addWidget(tool)

    tool.source_inspector.details_button.setChecked(True)
    assert tool.source_inspector.details_html() == "<p>fallback details</p>"
    assert format_calls[-1] == {
        "show_size": True,
        "show_summary": False,
        "load_values": False,
    }


def test_figure_composer_source_list_shape_cell_click_selects_row(qtbot) -> None:
    first = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("eV", "kx"),
        name="map",
    )
    second = xr.DataArray(np.arange(4.0), dims=("delay",), name="profile")
    tool = FigureComposerTool.from_sources(
        {"map": first, "profile": second},
        sources=(
            FigureSourceState(name="map", label="Map source"),
            FigureSourceState(name="profile", label="Profile source"),
        ),
        operations=(FigureOperationState.plot_slices(label="maps", sources=("map",)),),
        primary_source="map",
    )
    qtbot.addWidget(tool)
    tool.editor_tabs.setCurrentWidget(tool.sources_page)
    with qtbot.waitExposed(tool):
        tool.show()

    second_item = tool.source_list.topLevelItem(1)
    assert second_item is not None
    index = tool.source_list.indexFromItem(second_item, 1)
    rect = tool.source_list.visualRect(index)
    assert rect.isValid()

    qtbot.mouseClick(
        tool.source_list.viewport(),
        QtCore.Qt.MouseButton.LeftButton,
        pos=rect.center(),
    )

    assert tool.source_list.currentItem() is second_item
    assert tool.source_list.selectedItems() == [second_item]
    assert tool.source_inspector.source_name() == "profile"


def test_figure_composer_source_inspector_uses_public_nonuniform_dims(qtbot) -> None:
    public = xr.DataArray(
        np.arange(24.0).reshape(2, 3, 4),
        dims=("alpha", "eV", "sample_temp"),
        coords={
            "alpha": [0.0, 1.0],
            "eV": [-0.1, 0.0, 0.1],
            "sample_temp": [10.0, 15.0, 30.0, 60.0],
        },
        name="map",
    )
    internal = erlab.interactive.imagetool.slicer.make_dims_uniform(public)
    tool = FigureComposerTool.from_sources(
        {"data": internal},
        sources=(FigureSourceState(name="data", label="map"),),
        operations=(FigureOperationState.plot_slices(label="maps", sources=("data",)),),
        primary_source="data",
    )
    qtbot.addWidget(tool)

    dims = tool.source_inspector.property("figureComposerSourceDims")
    assert dims == ("alpha", "eV", "sample_temp")
    tool.source_inspector.details_button.setChecked(True)
    assert "sample_temp_idx" not in tool.source_inspector.details_html()
    assert "sample_temp" in tool.source_inspector.details_html()


def test_figure_composer_source_inspector_tracks_source_tab_selection(qtbot) -> None:
    first = xr.DataArray([1.0, 2.0], dims=("x",), name="first")
    second = xr.DataArray([3.0, 4.0], dims=("x",), name="second")
    tool = FigureComposerTool.from_sources(
        {"first": first, "second": second},
        sources=(
            FigureSourceState(name="first", label="first"),
            FigureSourceState(name="second", label="second"),
        ),
        operations=(
            FigureOperationState.line(label="first line", source="first"),
            FigureOperationState.custom(label="custom", code="pass", trusted=True),
        ),
        primary_source="first",
    )
    qtbot.addWidget(tool)

    assert tool.source_inspector.source_name() == "first"
    tool.operation_list.setCurrentItem(tool.operation_list.topLevelItem(1))
    assert tool.source_inspector.source_name() == "first"
    assert tool.source_status_label.text() == ""
    assert tool.source_status_label.isHidden()
    assert tool.step_source_status_label.isHidden()
    first_item = tool.source_list.topLevelItem(0)
    assert first_item is not None
    tool.source_list.setCurrentItem(first_item)
    assert tool.source_inspector.source_name() == "first"
    tool.operation_list.setCurrentItem(tool.operation_list.topLevelItem(0))
    assert tool.source_inspector.source_name() == "first"


def test_figure_composer_source_inspector_updates_without_render_or_history(
    qtbot, monkeypatch
) -> None:
    x_data = xr.DataArray([0.0, 1.0, 2.0], dims=("x",), name="x")
    y_data = xr.DataArray([1.0, 2.0], dims=("y",), name="y")
    operation = FigureOperationState.method(
        family=FigureMethodFamily.AXES,
        name="plot",
    ).model_copy(
        update={
            "method_plot_data_mode": "from_data",
            "method_plot_x": FigureMethodPlotValueState(source="x", kind="data"),
            "method_plot_y": FigureMethodPlotValueState(source="y", kind="data"),
        }
    )
    tool = FigureComposerTool.from_sources(
        {"x": x_data, "y": y_data},
        sources=(
            FigureSourceState(name="x", label="x"),
            FigureSourceState(name="y", label="y"),
        ),
        operations=(operation,),
        primary_source="y",
    )
    qtbot.addWidget(tool)
    render_calls: list[object] = []
    write_calls: list[object] = []
    monkeypatch.setattr(
        figurecomposer_tool_module,
        "_render_preview",
        lambda *args, **kwargs: render_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(tool, "_write_state", lambda: write_calls.append(None))

    y_item = tool.source_list.topLevelItem(1)
    assert y_item is not None
    tool.source_list.setCurrentItem(y_item)

    assert render_calls == []
    assert write_calls == []
    assert tool.source_inspector.source_name() == "y"
    assert tool.source_inspector.property("figureComposerSourceAlias") == "y"


def test_figure_composer_source_inspector_helper_edges(qtbot) -> None:
    data = xr.DataArray(
        [1.0, 2.0],
        dims=("x",),
        coords={"x": [0.0, 1.0], 7: ("x", [10.0, 20.0])},
        attrs={"note": "kept"},
        name="source",
    )
    source = FigureSourceState(name="data", label="display")

    missing_metadata = figurecomposer_source_inspector.source_metadata_tooltip(
        source, "data", None
    )
    assert missing_metadata.startswith("data")
    assert "unavailable" in missing_metadata

    metadata = figurecomposer_source_inspector.source_metadata_tooltip(
        source, "data", data
    )
    assert "Dims: x: 2" in metadata
    assert "Shape: 2" in metadata
    assert "Dtype: float64" in metadata

    assert "source is missing" in figurecomposer_source_inspector.source_value_tooltip(
        None, ("data", None), axis="x"
    )
    assert "Use DataArray values" in (
        figurecomposer_source_inspector.source_value_tooltip(
            data, ("data", None), axis="y"
        )
    )
    xerr_tooltip = figurecomposer_source_inspector.source_value_tooltip(
        data, ("data", None), axis="xerr"
    )
    yerr_tooltip = figurecomposer_source_inspector.source_value_tooltip(
        data, ("data", None), axis="yerr"
    )
    assert "XERR" not in xerr_tooltip
    assert "YERR" not in yerr_tooltip
    assert xerr_tooltip.replace("X error", "error") == yerr_tooltip.replace(
        "Y error", "error"
    )
    assert "Coordinate 'missing' is not available" in (
        figurecomposer_source_inspector.source_value_tooltip(
            data, ("coord", "missing"), axis="x"
        )
    )
    numeric_coord_tooltip = figurecomposer_source_inspector.source_value_tooltip(
        data, ("coord", "7"), axis="x"
    )
    assert "Coord dims: x" in numeric_coord_tooltip

    inspector = figurecomposer_source_inspector.SourceInspectorWidget()
    qtbot.addWidget(inspector)
    inspector.set_context(
        source_name=None,
        data=None,
    )
    assert inspector.source_name() is None
    assert inspector.property("figureComposerSourceDims") == ()
    assert not inspector.details_button.isEnabled()

    inspector.set_context(
        source_name="data",
        data=None,
    )
    assert inspector.source_name() == "data"
    assert inspector.property("figureComposerSourceDims") == ()
    assert not inspector.details_button.isEnabled()


def test_figure_composer_source_inspector_target_fallbacks(qtbot) -> None:
    data = xr.DataArray([1.0, 2.0], dims=("x",), name="data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="saved", label="Saved"),),
            primary_source="missing",
        ),
        source_data={},
    )
    qtbot.addWidget(tool)

    tool._source_data.clear()
    tool._select_source_list_row_silent(None)
    assert tool._default_source_inspector_target() == "saved"
    tool._refresh_source_detail_panel()
    assert tool.source_inspector.source_name() is None
    assert (
        tool.source_detail_content.property("figureComposerSourceEditorMode") == "empty"
    )

    tool._select_source_list_row_silent(None)
    assert tool.source_list.currentItem() is None


def test_figure_composer_line_source_combo_uses_alias_data_and_updates_recipe(
    qtbot,
) -> None:
    first = _figure_composer_profile_source("first")
    second = _figure_composer_profile_source("second")
    tool = FigureComposerTool(
        first,
        recipe=FigureRecipeState(
            sources=(
                FigureSourceState(name="data_0", label="ImageTool 0: first"),
                FigureSourceState(name="data_1", label="ImageTool 1: second"),
            ),
            operations=(FigureOperationState.line(label="line", source="data_0"),),
            primary_source="data_0",
        ),
        source_data={"data_0": first, "data_1": second},
    )
    qtbot.addWidget(tool)
    tool._select_step_section("sources")

    source_combos = tool.step_source_controls.findChildren(
        QtWidgets.QComboBox, "figureComposerLineSourceCombo"
    )
    source_combo = next(
        (
            combo
            for combo in source_combos
            if combo.property("figure_composer_editor_generation")
            == tool._operation_editor_generation
        ),
        None,
    )
    assert source_combo is not None
    first_index = source_combo.findData("data_0")
    second_index = source_combo.findData("data_1")
    assert first_index >= 0
    assert second_index >= 0
    assert source_combo.itemData(first_index) == "data_0"
    assert source_combo.itemData(second_index) == "data_1"

    _activate_combo_index(source_combo, second_index)

    assert tool.tool_status.operations[0].line_source == "data_1"


def test_figure_composer_rebases_source_node_uids(qtbot) -> None:
    data = _figure_composer_image_source("data")
    nested_spec = provenance.ToolProvenanceSpec(
        kind="script",
        start_label="nested",
        active_name="nested",
        script_inputs=(
            provenance.ScriptInput(
                name="nested",
                label="nested",
                node_uid="old-nested",
            ),
        ),
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(
                FigureSourceState(
                    name="data",
                    label="data",
                    node_uid="old-source",
                    provenance_spec=nested_spec.model_dump(mode="json"),
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    tool.rebase_source_node_uids(
        {"old-source": "new-source", "old-nested": "new-nested"}
    )

    source = tool.tool_status.sources[0]
    assert source.node_uid == "new-source"
    rebased_spec = provenance.parse_tool_provenance_spec(source.provenance_spec)
    assert rebased_spec is not None
    assert rebased_spec.script_inputs[0].node_uid == "new-nested"


def test_figure_composer_provenance_includes_sources_and_build_step(
    qtbot, tmp_path: Path
) -> None:
    data = xr.DataArray(
        np.arange(9.0).reshape(3, 3),
        dims=("x", "y"),
        coords={"x": np.arange(3), "y": np.arange(3)},
        name="map",
    )
    file_path = tmp_path / "map.nc"
    data.to_netcdf(file_path)
    source_spec = _file_load_provenance(file_path)
    operation = FigureOperationState.plot_slices(
        label="plot_slices",
        sources=("data_0",),
        axes=FigureAxesSelectionState(axes=((0, 0),)),
    ).model_copy(update={"annotate": False})
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(
                FigureSourceState(
                    name="data_0",
                    label="test source",
                    provenance_spec=source_spec.model_dump(mode="json"),
                ),
            ),
            operations=(operation,),
            primary_source="data_0",
        ),
        source_data={"data_0": data},
    )
    qtbot.addWidget(tool)

    spec = tool.current_provenance_spec()
    assert spec is not None
    assert spec.active_name == "fig"
    assert tuple(script_input.name for script_input in spec.script_inputs) == ("map",)
    entries = spec.display_entries()
    assert len(entries) == 3
    assert entries[1].code is None
    assert not entries[1].copyable
    assert entries[2].code is not None
    assert entries[2].copyable

    code = spec.display_code()
    assert code is not None
    namespace = _exec_generated_code(code, {})
    assert isinstance(namespace["fig"], Figure)


def test_figure_composer_full_code_composes_selected_file_sources(
    qtbot,
    monkeypatch,
    tmp_path: Path,
) -> None:
    first = xr.DataArray(
        np.arange(8.0).reshape(2, 2, 2),
        dims=("hv", "x", "y"),
        coords={"hv": [39.274, 43.698], "x": [0.0, 1.0], "y": [0.0, 1.0]},
        name="QV12 LH gap15",
    )
    second = (first + 10.0).rename("QV12 LH gap15 ALSU")
    first_path = tmp_path / "first.nc"
    second_path = tmp_path / "second.nc"
    first.to_netcdf(first_path)
    second.to_netcdf(second_path)
    first_source = FigureSourceState(
        name="data_3",
        label="ImageTool 3: QV12 LH gap15",
        provenance_spec=_file_load_provenance(first_path).model_dump(mode="json"),
    )
    second_source = FigureSourceState(
        name="data_4",
        label="ImageTool 4: QV12 LH gap15 ALSU",
        provenance_spec=_file_load_provenance(second_path).model_dump(mode="json"),
    )
    first_selected = FigureSourceState(
        name="data_3_selected",
        label="QV12 LH gap15 selection",
        selection_source="data_3",
        qsel={"hv": 39.274},
        provenance_spec=first_source.provenance_spec,
    )
    second_selected = FigureSourceState(
        name="data_4_selected",
        label="QV12 LH gap15 ALSU selection",
        selection_source="data_4",
        qsel={"hv": 43.698},
        provenance_spec=second_source.provenance_spec,
    )
    tool = FigureComposerTool(
        first,
        recipe=FigureRecipeState(
            sources=(first_source, first_selected, second_source, second_selected),
            operations=(
                FigureOperationState.plot_array(
                    label="plot_array",
                    source="data_3_selected",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ),
                FigureOperationState.plot_array(
                    label="plot_array",
                    source="data_4_selected",
                    axes=FigureAxesSelectionState(axes=((0, 1),)),
                ),
            ),
            setup=FigureSubplotsState(nrows=1, ncols=2),
            primary_source="data_3",
        ),
        source_data={
            "data_3": first,
            "data_3_selected": first.qsel(hv=39.274),
            "data_4": second,
            "data_4_selected": second.qsel(hv=43.698),
        },
    )
    qtbot.addWidget(tool)

    spec = tool.current_provenance_spec()
    assert spec is not None
    assert tuple(script_input.name for script_input in spec.script_inputs) == (
        "qv12_lh_gap15",
        "qv12_lh_gap15_alsu",
    )
    code = spec.display_code()
    assert code is not None
    lines = [line for line in code.splitlines() if line.strip()]
    assert lines[:3] == [
        "import xarray",
        "import matplotlib.pyplot as plt",
        "import erlab.plotting as eplt",
    ]
    assert "_itool_replay_" not in code
    assert "data_3_selected =" not in code
    assert "data_4_selected =" not in code
    assert "qv12_lh_gap15 = xarray.load_dataarray" in code
    assert "qv12_lh_gap15_alsu = xarray.load_dataarray" in code
    assert code.count(".qsel(hv=") == 2
    assert "eplt.plot_array(qv12_lh_gap15" in code
    assert "eplt.plot_array(qv12_lh_gap15_alsu" in code

    captured: list[xr.DataArray] = []
    monkeypatch.setattr(
        eplt,
        "plot_array",
        lambda arr, **_kwargs: captured.append(arr),
    )
    namespace = _exec_generated_code(code, {})
    assert isinstance(namespace["fig"], Figure)
    xr.testing.assert_identical(captured[0], first.qsel(hv=39.274))
    xr.testing.assert_identical(captured[1], second.qsel(hv=43.698))


def test_figure_composer_full_code_keeps_needed_base_and_selected_sources(
    qtbot,
    tmp_path: Path,
) -> None:
    data = xr.DataArray(
        np.arange(8.0).reshape(2, 2, 2),
        dims=("hv", "x", "y"),
        coords={"hv": [39.274, 43.698], "x": [0.0, 1.0], "y": [0.0, 1.0]},
        name="fig",
    )
    path = tmp_path / "source.nc"
    data.to_netcdf(path)
    source = FigureSourceState(
        name="data_3",
        label="ImageTool 3: fig",
        provenance_spec=_file_load_provenance(path).model_dump(mode="json"),
    )
    selected = FigureSourceState(
        name="data_3_selected",
        label="fig selection",
        selection_source="data_3",
        qsel={"hv": 39.274},
        provenance_spec=source.provenance_spec,
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(source, selected),
            operations=(
                FigureOperationState.plot_array(
                    label="raw",
                    source="data_3",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ),
                FigureOperationState.plot_array(
                    label="selected",
                    source="data_3_selected",
                    axes=FigureAxesSelectionState(axes=((0, 1),)),
                ),
            ),
            setup=FigureSubplotsState(nrows=1, ncols=2),
            primary_source="data_3",
        ),
        source_data={"data_3": data, "data_3_selected": data.qsel(hv=39.274)},
    )
    qtbot.addWidget(tool)

    spec = tool.current_provenance_spec()
    assert spec is not None
    assert tuple(script_input.name for script_input in spec.script_inputs) == (
        "fig_2",
        "fig_3",
    )
    code = spec.display_code()
    assert code is not None
    assert "fig =" not in code
    assert "fig_2 = xarray.load_dataarray" in code
    assert "fig_3 = fig_2.qsel(hv=39.274)" in code
    assert "eplt.plot_array(fig_2" in code
    assert "eplt.plot_array(fig_3" in code


def test_figure_composer_full_code_falls_back_for_live_selected_source(
    qtbot,
    monkeypatch,
) -> None:
    data = xr.DataArray(
        np.arange(8.0).reshape(2, 2, 2),
        dims=("hv", "x", "y"),
        coords={"hv": [39.274, 43.698], "x": [0.0, 1.0], "y": [0.0, 1.0]},
        name="live",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(
                FigureSourceState(name="data_3", label="live", node_uid="node"),
                FigureSourceState(
                    name="data_3_selected",
                    label="live selection",
                    selection_source="data_3",
                    qsel={"hv": 39.274},
                ),
            ),
            operations=(
                FigureOperationState.plot_array(
                    label="selected",
                    source="data_3_selected",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ),
            ),
            primary_source="data_3",
        ),
        source_data={"data_3": data, "data_3_selected": data.qsel(hv=39.274)},
    )
    qtbot.addWidget(tool)

    spec = tool.current_provenance_spec()
    assert spec is not None
    assert tuple(script_input.name for script_input in spec.script_inputs) == (
        "data_3",
    )
    assert spec.display_code() is None

    code = tool.generated_code()
    assert "data_3_selected = data_3.qsel(hv=39.274)" in code
    assert "eplt.plot_array(data_3_selected" in code

    captured: list[xr.DataArray] = []
    monkeypatch.setattr(
        eplt,
        "plot_array",
        lambda arr, **_kwargs: captured.append(arr),
    )
    namespace = _exec_generated_code(code, {"data_3": data})
    assert isinstance(namespace["fig"], Figure)
    xr.testing.assert_identical(captured[0], data.qsel(hv=39.274))


def test_figure_composer_copy_paste_steps_carries_same_process_source_data(
    qtbot,
) -> None:
    source_data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
        name="map",
    )
    source_tool = FigureComposerTool(
        source_data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="map", label="map"),),
            operations=(
                FigureOperationState.plot_slices(
                    label="plot",
                    sources=("map",),
                ),
            ),
            primary_source="map",
        ),
        source_data={"map": source_data},
    )
    destination_data = xr.DataArray(
        np.arange(3.0),
        dims=("kx",),
        coords={"kx": [0.0, 1.0, 2.0]},
        name="existing",
    )
    destination = FigureComposerTool(
        destination_data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="existing", label="existing"),),
            operations=(),
            primary_source="existing",
        ),
        source_data={"existing": destination_data},
    )
    qtbot.addWidget(source_tool)
    qtbot.addWidget(destination)
    _clear_clipboard()
    data_changed: list[None] = []
    destination.sigDataChanged.connect(lambda: data_changed.append(None))

    _select_operation_rows(source_tool, (0,))
    source_tool.copy_operation_button.click()
    destination._paste_operations_from_clipboard()

    assert data_changed == [None]
    assert [source.name for source in destination.tool_status.sources] == [
        "existing",
        "map",
    ]
    assert destination.tool_status.operations[-1].sources == ("map",)
    xr.testing.assert_identical(destination.source_data()["map"], source_data)

    destination.undo()
    assert [source.name for source in destination.tool_status.sources] == ["existing"]
    assert set(destination.source_data()) == {"existing"}

    destination.redo()
    assert [source.name for source in destination.tool_status.sources] == [
        "existing",
        "map",
    ]
    assert destination.tool_status.operations[-1].sources == ("map",)
    xr.testing.assert_identical(destination.source_data()["map"], source_data)


def test_figure_composer_cut_paste_steps_preserves_same_composer_sources(
    qtbot,
) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.plot_slices(
                    label="plot",
                    sources=("data",),
                    map_selections=(FigureDataSelectionState(source="data"),),
                ),
            ),
            primary_source="data",
        ),
        source_data={"data": data},
    )
    qtbot.addWidget(tool)
    _clear_clipboard()

    _select_operation_rows(tool, (0,))
    original_id = tool.tool_status.operations[0].operation_id
    tool.cut_operation_button.click()
    tool.paste_operation_button.click()

    assert [source.name for source in tool.tool_status.sources] == ["data"]
    assert set(tool.source_data()) == {"data"}
    pasted_operation = tool.tool_status.operations[0]
    assert pasted_operation.sources == ("data",)
    assert pasted_operation.map_selections == ()
    assert pasted_operation.operation_id != original_id


def test_figure_composer_cut_paste_steps_renames_cross_composer_sources(
    qtbot,
) -> None:
    image = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
        name="incoming_image",
    )
    profile = xr.DataArray(
        np.arange(3.0),
        dims=("kx",),
        coords={"kx": [0.0, 1.0, 2.0]},
        name="incoming_profile",
    )
    source_tool = FigureComposerTool(
        image,
        recipe=FigureRecipeState(
            sources=(
                FigureSourceState(name="data", label="image"),
                FigureSourceState(name="profile", label="profile"),
            ),
            operations=(
                FigureOperationState.plot_slices(
                    label="plot",
                    sources=("data",),
                    map_selections=(FigureDataSelectionState(source="data"),),
                ),
                FigureOperationState.line(label="line", source="profile"),
            ),
            primary_source="data",
        ),
        source_data={"data": image, "profile": profile},
    )
    existing_image = image.copy(data=np.full((2, 2), -1.0))
    existing_profile = profile.copy(data=np.full(3, -1.0))
    destination = FigureComposerTool(
        existing_image,
        recipe=FigureRecipeState(
            sources=(
                FigureSourceState(name="data", label="existing image"),
                FigureSourceState(name="profile", label="existing profile"),
            ),
            operations=(),
            primary_source="data",
        ),
        source_data={"data": existing_image, "profile": existing_profile},
    )
    qtbot.addWidget(source_tool)
    qtbot.addWidget(destination)
    _clear_clipboard()

    _select_operation_rows(source_tool, (0, 1))
    source_tool.cut_operation_button.click()
    destination._paste_operations_from_clipboard()

    assert [operation.label for operation in source_tool.tool_status.operations] == []
    assert [source.name for source in destination.tool_status.sources] == [
        "data",
        "profile",
        "data_copy",
        "profile_copy",
    ]
    pasted_plot, pasted_line = destination.tool_status.operations
    assert pasted_plot.sources == ("data_copy",)
    assert pasted_plot.map_selections == ()
    assert pasted_line.line_source == "profile_copy"
    xr.testing.assert_identical(destination.source_data()["data"], existing_image)
    xr.testing.assert_identical(destination.source_data()["profile"], existing_profile)
    xr.testing.assert_identical(destination.source_data()["data_copy"], image)
    xr.testing.assert_identical(destination.source_data()["profile_copy"], profile)


def test_figure_composer_copy_paste_steps_renames_source_conflicts(qtbot) -> None:
    image = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
        name="incoming_image",
    )
    profile = xr.DataArray(
        np.arange(3.0),
        dims=("kx",),
        coords={"kx": [0.0, 1.0, 2.0]},
        name="incoming_profile",
    )
    source_tool = FigureComposerTool(
        image,
        recipe=FigureRecipeState(
            sources=(
                FigureSourceState(name="data", label="image"),
                FigureSourceState(name="profile", label="profile"),
            ),
            operations=(
                FigureOperationState.plot_slices(
                    label="plot",
                    sources=("data",),
                    map_selections=(FigureDataSelectionState(source="data"),),
                ),
                FigureOperationState.line(label="line", source="profile"),
            ),
            primary_source="data",
        ),
        source_data={"data": image, "profile": profile},
    )
    existing_image = image.copy(data=np.full((2, 2), -1.0))
    existing_profile = profile.copy(data=np.full(3, -1.0))
    destination = FigureComposerTool(
        existing_image,
        recipe=FigureRecipeState(
            sources=(
                FigureSourceState(name="data", label="existing image"),
                FigureSourceState(name="profile", label="existing profile"),
            ),
            operations=(),
            primary_source="data",
        ),
        source_data={"data": existing_image, "profile": existing_profile},
    )
    qtbot.addWidget(source_tool)
    qtbot.addWidget(destination)
    _clear_clipboard()

    _select_operation_rows(source_tool, (0, 1))
    source_tool.copy_operation_button.click()
    destination._paste_operations_from_clipboard()

    assert [source.name for source in destination.tool_status.sources] == [
        "data",
        "profile",
        "data_copy",
        "profile_copy",
    ]
    pasted_plot, pasted_line = destination.tool_status.operations
    assert pasted_plot.sources == ("data_copy",)
    assert pasted_plot.map_selections == ()
    assert pasted_line.line_source == "profile_copy"
    xr.testing.assert_identical(destination.source_data()["data"], existing_image)
    xr.testing.assert_identical(destination.source_data()["profile"], existing_profile)
    xr.testing.assert_identical(destination.source_data()["data_copy"], image)
    xr.testing.assert_identical(destination.source_data()["profile_copy"], profile)


def test_figure_composer_copy_paste_steps_typed_payload_has_missing_source(
    qtbot,
) -> None:
    remote_data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
        name="remote",
    )
    source_tool = FigureComposerTool(
        remote_data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="remote", label="remote"),),
            operations=(
                FigureOperationState.plot_slices(
                    label="plot",
                    sources=("remote",),
                ),
            ),
            primary_source="remote",
        ),
        source_data={"remote": remote_data},
    )
    destination_data = xr.DataArray(
        np.arange(3.0),
        dims=("kx",),
        coords={"kx": [0.0, 1.0, 2.0]},
        name="local",
    )
    destination = FigureComposerTool(
        destination_data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="local", label="local"),),
            operations=(),
            primary_source="local",
        ),
        source_data={"local": destination_data},
    )
    qtbot.addWidget(source_tool)
    qtbot.addWidget(destination)
    clipboard = _clear_clipboard()

    _select_operation_rows(source_tool, (0,))
    source_tool.copy_operation_button.click()
    copied_mime = clipboard.mimeData()
    payload_text = bytes(
        copied_mime.data(figurecomposer_tool_module._STEPS_CLIPBOARD_MIME).data()
    ).decode("utf-8")
    assert payload_text.startswith(
        '{\n  "type": "erlab.figure_composer.steps",\n  "version": 1,'
    )
    assert json.loads(payload_text)["type"] == "erlab.figure_composer.steps"
    assert copied_mime.text()
    assert not copied_mime.text().lstrip().startswith("{")
    with pytest.raises(json.JSONDecodeError):
        json.loads(copied_mime.text())
    mime = QtCore.QMimeData()
    mime.setData(
        figurecomposer_tool_module._STEPS_CLIPBOARD_MIME,
        payload_text.encode("utf-8"),
    )
    mime.setText(copied_mime.text())
    clipboard.setMimeData(mime)
    destination._paste_operations_from_clipboard()

    assert [source.name for source in destination.tool_status.sources] == [
        "local",
        "remote",
    ]
    assert destination.tool_status.operations[-1].sources == ("remote",)
    assert "remote" not in destination.source_data()


def test_figure_composer_source_data_history_helpers(qtbot) -> None:
    data = xr.DataArray(np.arange(3.0), dims=("x",), name="data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(_custom_order_step("a"),),
            primary_source="data",
        ),
        source_data={"data": data},
    )
    qtbot.addWidget(tool)

    tool._reset_history_stack()
    assert list(tool._prev_source_data_states[-1]) == ["data"]
    assert not tool.undoable
    assert not tool.redoable
    tool.undo()
    tool.redo()
    tool._write_state()
    assert len(tool._prev_source_data_states) == 1

    with tool._history_suppressed():
        tool._write_state()
        tool._replace_last_state()
    assert len(tool._prev_source_data_states) == 1

    replacement = data.copy(data=np.full(3, 2.0))
    tool.set_source_data({"data": replacement})
    tool._replace_last_state()
    xr.testing.assert_identical(tool._prev_source_data_states[-1]["data"], replacement)

    tool._prev_states.clear()
    tool._prev_source_data_states.clear()
    tool._replace_last_state()
    assert len(tool._prev_states) == 1
    xr.testing.assert_identical(tool._prev_source_data_states[-1]["data"], replacement)


def test_figure_composer_copy_paste_source_and_insert_fallbacks(
    qtbot, monkeypatch
) -> None:
    data = xr.DataArray(np.arange(3.0), dims=("x",), name="data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(_custom_order_step("a"),),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    operation = FigureOperationState.plot_slices(
        label="plot",
        sources=("data",),
        map_selections=(FigureDataSelectionState(source="extra"),),
    )
    assert tool._operation_source_names(operation) == ("data",)

    renamed_sources, _, _ = tool._renamed_pasted_sources(
        (
            FigureSourceState(name="extra", label="extra"),
            FigureSourceState(name="extra", label="extra"),
        ),
        {},
    )
    assert [source.name for source in renamed_sources] == ["extra"]

    clipboard = _clear_clipboard()
    clipboard.setText(
        figurecomposer_tool_module._step_clipboard_payload_text(
            (_custom_order_step("b"),), ()
        )
    )
    tool.operation_list.setCurrentItem(tool.operation_list.topLevelItem(0))
    with monkeypatch.context() as patch:
        patch.setattr(tool, "_operation_id_for_item", lambda item: "missing")
        tool._paste_operations_from_clipboard()
    assert [operation.label for operation in tool.tool_status.operations] == ["a", "b"]

    def existing_source_payload(sources, source_data, *, preserve_existing=False):
        return (FigureSourceState(name="data", label="data"),), {}, {}

    with monkeypatch.context() as patch:
        patch.setattr(tool, "_renamed_pasted_sources", existing_source_payload)
        clipboard.setText(
            figurecomposer_tool_module._step_clipboard_payload_text(
                (_custom_order_step("c"),), ()
            )
        )
        tool._paste_operations_from_clipboard()
    assert [source.name for source in tool.tool_status.sources] == ["data"]

    tool._source_data["data_copy"] = data
    renamed_sources, rename_map, _ = tool._renamed_pasted_sources(
        (FigureSourceState(name="data", label="data"),),
        {},
    )
    assert [source.name for source in renamed_sources] == ["data_copy_2"]
    assert rename_map == {"data": "data_copy_2"}

    tool._source_data["extra"] = data
    renamed_sources, rename_map, renamed_source_data = tool._renamed_pasted_sources(
        (FigureSourceState(name="extra", label="extra"),),
        {"extra": data},
        preserve_existing=True,
    )
    assert [source.name for source in renamed_sources] == ["extra"]
    assert rename_map == {"extra": "extra"}
    assert renamed_source_data == {}

    metadata_source = FigureSourceState(name="metadata_only", label="metadata")
    tool._recipe = tool._recipe.model_copy(
        update={"sources": (*tool.tool_status.sources, metadata_source)}
    )
    renamed_sources, rename_map, renamed_source_data = tool._renamed_pasted_sources(
        (metadata_source,),
        {"metadata_only": data},
        preserve_existing=True,
    )
    assert renamed_sources == ()
    assert rename_map == {"metadata_only": "metadata_only"}
    xr.testing.assert_identical(renamed_source_data["metadata_only"], data)


def test_figure_composer_toolbar_image_target_combo_elides_long_sources(qtbot) -> None:
    source_count = 6
    source_names = tuple(
        f"source_{index}_with_an_intentionally_long_alias_name"
        for index in range(source_count)
    )
    source_labels = tuple(
        f"Long data label {index} with additional descriptive text"
        for index in range(source_count)
    )
    source_data = {name: _figure_composer_image_source(name) for name in source_names}
    operation = FigureOperationState.plot_slices(
        label="plot_slices",
        sources=source_names,
        axes=FigureAxesSelectionState(
            axes=tuple((0, index) for index in range(source_count))
        ),
    )
    tool = FigureComposerTool(
        source_data[source_names[0]],
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(ncols=source_count),
            sources=tuple(
                FigureSourceState(name=name, label=label)
                for name, label in zip(source_names, source_labels, strict=True)
            ),
            operations=(operation,),
            primary_source=source_names[0],
        ),
        source_data=source_data,
    )
    qtbot.addWidget(tool)
    tool.show_figure_window(activate=False)

    tool._show_axes_customize_dialog()
    dialog = tool._axes_customize_dialog
    assert isinstance(dialog, QtWidgets.QDialog)
    selector = dialog.findChild(figurecomposer_widgets._AxesSelectorWidget)
    target_combo = dialog.findChild(
        QtWidgets.QComboBox, "figureComposerToolbarImageTargetCombo"
    )
    assert selector is not None
    assert target_combo is not None
    selector.set_selected_axes(
        tuple((0, index) for index in range(source_count)), emit=True
    )

    assert target_combo.count() == 1
    visible_text = target_combo.itemText(0)
    assert "Image slices" in visible_text
    assert f"{source_count} panels" in visible_text
    for text in (*source_names, *source_labels):
        assert text not in visible_text

    tooltip = target_combo.itemData(0, QtCore.Qt.ItemDataRole.ToolTipRole)
    assert isinstance(tooltip, str)
    assert all(name in tooltip for name in source_names)
    assert all(label not in tooltip for label in source_labels)
    assert target_combo.toolTip() == tooltip
    assert target_combo.view().textElideMode() == QtCore.Qt.TextElideMode.ElideMiddle
    assert (
        target_combo.sizeAdjustPolicy()
        == QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    )
    assert (
        target_combo.sizeHint().width()
        < target_combo.fontMetrics().horizontalAdvance(tooltip)
    )
    custom_label = figurecomposer_toolbar_dialogs._image_style_target_label(
        1,
        operation.model_copy(update={"label": "Custom image step"}),
        (figurecomposer_plot_slices._PlotSlicesPanelKey(0, 1, "ignored"),),
    )
    assert custom_label == "Step 2: Custom image step: panel 1.2"


def test_figure_composer_batch_line_source_dependent_combos_disable(
    qtbot,
) -> None:
    first = xr.DataArray(
        np.array([1.0, 2.0]),
        dims=("kx",),
        coords={"kx": [0.0, 1.0]},
        name="first",
    )
    second = xr.DataArray(
        np.array([1.0, 2.0]),
        dims=("ky",),
        coords={"ky": [0.0, 1.0]},
        name="second",
    )
    tool = FigureComposerTool(
        first,
        recipe=FigureRecipeState(
            sources=(
                FigureSourceState(name="first", label="first"),
                FigureSourceState(name="second", label="second"),
            ),
            operations=(
                FigureOperationState.line(label="first", source="first"),
                FigureOperationState.line(label="second", source="second"),
            ),
            primary_source="first",
        ),
        source_data={"first": first, "second": second},
    )
    qtbot.addWidget(tool)

    _select_operation_rows(tool, (0, 1))
    tool._select_step_section("selection")
    selection_page = tool.step_editor_stack.currentWidget()
    coordinate_combo = selection_page.findChild(
        QtWidgets.QComboBox, "figureComposerProfileCoordinateCombo"
    )
    assert coordinate_combo is not None
    assert coordinate_combo.isEnabled() is False
    assert "different valid choices" in coordinate_combo.toolTip()


def test_figure_composer_restore_skips_missing_nonprimary_source_reference(
    qtbot, monkeypatch
) -> None:
    primary = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="primary",
    )
    stale = xr.DataArray(
        np.arange(4.0, 8.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="stale",
    )
    tool = FigureComposerTool.from_sources(
        {"primary": primary, "stale": stale},
        sources=(
            FigureSourceState(name="primary", label="Primary", node_uid="n-primary"),
            FigureSourceState(name="stale", label="Stale", node_uid="n-stale"),
        ),
        operations=(FigureOperationState.line(label="line", source="primary"),),
        primary_source="primary",
    )
    qtbot.addWidget(tool)
    assert tool.refresh_preview_pixmap(allow_offscreen=True) is not None

    with tool._save_tool_data_reference_context({"n-primary", "n-stale"}):
        ds = tool.to_dataset()

    references = json.loads(
        ds.attrs[erlab.interactive.utils._TOOL_DATA_REFERENCES_ATTR]
    )
    assert references["stale"]["node_uid"] == "n-stale"
    assert ds["stale"].size == 0

    with pytest.raises(ValueError, match="'n-primary'"):
        erlab.interactive.utils.ToolWindow.from_dataset(
            ds,
            _tool_data_reference_resolver=lambda _reference: None,
        )

    def fail_render(*_args, **_kwargs) -> None:
        pytest.fail("restoring a cached figure preview must not render the recipe")

    monkeypatch.setattr(figurecomposer_tool_module, "_render_preview", fail_render)

    restored = erlab.interactive.utils.ToolWindow.from_dataset(
        ds,
        _tool_data_reference_resolver=lambda reference: (
            primary if reference.get("node_uid") == "n-primary" else None
        ),
    )
    qtbot.addWidget(restored)

    assert isinstance(restored, FigureComposerTool)
    source_data = restored.source_data()
    xr.testing.assert_identical(source_data["primary"], primary)
    assert "stale" not in source_data
    assert any(source.name == "stale" for source in restored.tool_status.sources)
    assert restored.preview_pixmap is not None
    assert not restored.preview_pixmap_stale
