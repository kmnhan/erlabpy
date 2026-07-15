# ruff: noqa: F403, F405

import erlab.interactive._figurecomposer._codegen as figurecomposer_codegen
import erlab.interactive._figurecomposer._source_panel as figurecomposer_source_panel
from erlab.interactive._figurecomposer._exceptions import FigureComposerInputError

from ._common import *


def _source_context_action(
    tool: FigureComposerTool, object_name: str
) -> tuple[QtWidgets.QMenu, QtGui.QAction]:
    existing_menus = tool.source_panel.source_list.findChildren(QtWidgets.QMenu)
    tool.source_panel._show_context_menu(QtCore.QPoint(0, 0))
    menu = next(
        menu
        for menu in tool.source_panel.source_list.findChildren(QtWidgets.QMenu)
        if all(menu is not existing_menu for existing_menu in existing_menus)
    )
    action = menu.findChild(QtGui.QAction, object_name)
    assert action is not None
    return menu, action


def test_figure_composer_source_alias_candidate_normalizes_usable_names() -> None:
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

    assert (
        figurecomposer_sources._source_alias_candidate(
            xr.DataArray(np.arange(2), dims=("x",), name="bad name")
        )
        == "bad_name"
    )


def test_figure_composer_source_state_normalizes_legacy_self_selection_parent() -> None:
    source = FigureSourceState.model_validate(
        {
            "name": "selected",
            "qsel": {"x": 1.0},
            "selection_source": "selected",
        }
    )

    assert source.selection_source is None
    assert "selection_source" not in source.model_dump(mode="json")

    immutable_source = FigureSourceState.model_validate(
        types.MappingProxyType(
            {
                "name": "immutable_selected",
                "label": "immutable_selected",
                "selection_source": "immutable_selected",
            }
        )
    )
    assert immutable_source.selection_source is None

    with pytest.raises(ValueError):
        FigureSourceState.model_validate("not a source mapping")

    legacy_source = FigureSourceState(name="selected").model_copy(
        update={"selection_source": "selected"}
    )
    selected_source = figurecomposer_sources._source_with_selection(
        legacy_source,
        FigureDataSelectionState(source="selected", isel={"x": 0}),
    )
    assert selected_source.selection_source is None


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
        tool._document.source_selection_base_data["profile"] = base

    restored = erlab.interactive.utils.ToolWindow.from_dataset(tool.to_dataset())
    qtbot.addWidget(restored)
    assert isinstance(restored, FigureComposerTool)
    xr.testing.assert_identical(restored.source_data()["profile"], selected)
    assert (
        "profile" in restored._document.source_selection_base_data
    ) is retain_base_data

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
    xr.testing.assert_identical(
        restored._document.source_selection_base_data["profile"], base
    )


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
    xr.testing.assert_identical(
        restored._document.source_selection_base_data["selected"], base
    )


def test_figure_composer_restore_rebuilds_selected_source_chain_out_of_order(
    qtbot,
) -> None:
    base = xr.DataArray(
        np.arange(24.0).reshape(2, 3, 4),
        dims=("u", "v", "w"),
        coords={"u": [0.0, 1.0], "v": [0.0, 1.0, 2.0], "w": np.arange(4)},
        name="base",
    )
    tool = FigureComposerTool.from_sources(
        {"base": base},
        sources=(
            FigureSourceState(
                name="selected_v",
                selection_source="selected_u",
                qsel={"v": 2.0},
            ),
            FigureSourceState(
                name="selected_u",
                selection_source="base",
                qsel={"u": 1.0},
            ),
            FigureSourceState(name="base"),
        ),
        operations=(FigureOperationState.line(label="line", source="selected_v"),),
        primary_source="base",
    )
    qtbot.addWidget(tool)

    tool._restore_persistence_data_items({}, xr.Dataset())

    selected_u = base.qsel(u=1.0)
    selected_v = selected_u.qsel(v=2.0)
    xr.testing.assert_identical(tool.source_data()["selected_u"], selected_u)
    xr.testing.assert_identical(tool.source_data()["selected_v"], selected_v)
    xr.testing.assert_identical(
        tool._document.source_selection_base_data["selected_u"], base
    )
    xr.testing.assert_identical(
        tool._document.source_selection_base_data["selected_v"], selected_u
    )


def test_figure_composer_restore_recomputes_cached_selected_descendants(qtbot) -> None:
    tool, base, selected_u, selected_v = _transitive_selected_source_tool()
    qtbot.addWidget(tool)
    replacement = base + 100.0

    tool._restore_persistence_data_items(
        {
            erlab.interactive.utils._SAVED_TOOL_DATA_NAME: replacement,
            "selected_u": base,
            "selected_v": selected_u,
        },
        xr.Dataset(),
    )

    expected_u = replacement.qsel(u=1.0)
    expected_v = expected_u.qsel(v=2.0)
    xr.testing.assert_identical(tool.source_data()["base"], replacement)
    xr.testing.assert_identical(tool.source_data()["selected_u"], expected_u)
    xr.testing.assert_identical(tool.source_data()["selected_v"], expected_v)
    assert not tool.source_data()["selected_u"].identical(selected_u)
    assert not tool.source_data()["selected_v"].identical(selected_v)
    xr.testing.assert_identical(
        tool._document.source_selection_base_data["selected_u"], replacement
    )
    xr.testing.assert_identical(
        tool._document.source_selection_base_data["selected_v"], expected_u
    )


def test_figure_composer_restore_fallbacks_are_parent_first(qtbot) -> None:
    stable = xr.DataArray(np.arange(2.0), dims=("x",), name="stable")
    base = xr.DataArray(
        np.arange(24.0).reshape(2, 3, 4),
        dims=("u", "v", "w"),
        coords={"u": [0.0, 1.0], "v": [0.0, 1.0, 2.0], "w": np.arange(4)},
        name="base",
    )
    selected_u = base.qsel(u=1.0)
    tool = FigureComposerTool.from_sources(
        {"stable": stable},
        sources=(
            FigureSourceState(
                name="selected_v",
                selection_source="selected_u",
                qsel={"v": 2.0},
            ),
            FigureSourceState(
                name="selected_u",
                selection_source="missing_base",
                qsel={"u": 1.0},
            ),
            FigureSourceState(name="missing_base"),
            FigureSourceState(name="stable"),
        ),
        operations=(FigureOperationState.line(label="line", source="selected_v"),),
        primary_source="stable",
    )
    qtbot.addWidget(tool)

    tool._restore_persistence_data_items(
        {"selected_v": selected_u, "selected_u": base}, xr.Dataset()
    )

    xr.testing.assert_identical(tool.source_data()["selected_u"], selected_u)
    xr.testing.assert_identical(
        tool.source_data()["selected_v"], selected_u.qsel(v=2.0)
    )


def test_figure_composer_restore_selected_source_cycle_stays_unresolved(qtbot) -> None:
    stable = xr.DataArray(np.arange(3.0), dims=("x",), name="stable")
    tool = FigureComposerTool.from_sources(
        {"stable": stable},
        sources=(
            FigureSourceState(name="stable"),
            FigureSourceState(
                name="cycle_a", selection_source="cycle_b", isel={"x": 0}
            ),
            FigureSourceState(
                name="cycle_b", selection_source="cycle_a", isel={"x": 0}
            ),
        ),
        operations=(),
        primary_source="stable",
    )
    qtbot.addWidget(tool)

    tool._restore_persistence_data_items({}, xr.Dataset())

    assert set(tool.source_data()) == {"stable"}
    assert tool._document.source_selection_base_data == {}


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
    del tool._document.source_data["primary"]

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

    first_item = tool.source_panel.source_list.topLevelItem(0)
    assert first_item is not None
    assert first_item.data(0, QtCore.Qt.ItemDataRole.UserRole) == "data_0"
    assert first_item.data(0, QtCore.Qt.ItemDataRole.UserRole + 1) is True
    assert not first_item.font(0).bold()
    second_item = tool.source_panel.source_list.topLevelItem(1)
    assert second_item is not None
    assert second_item.data(0, QtCore.Qt.ItemDataRole.UserRole) == "data_1"
    assert second_item.data(0, QtCore.Qt.ItemDataRole.UserRole + 1) is False

    tool.operation_panel.select_section("sources")
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

    first_item = tool.source_panel.source_list.topLevelItem(0)
    assert first_item is not None
    shape_widget = tool.source_panel.source_list.itemWidget(first_item, 1)
    assert isinstance(shape_widget, QtWidgets.QLabel)
    assert tool.source_panel.source_list.toolTip() == ""
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

    assert tool.source_panel.source_list.columnCount() == 2
    assert tool.source_panel.source_list.findChildren(QtWidgets.QToolButton) == []
    assert (
        tool.source_panel.refresh_sources_button.accessibleName()
        == "Refresh Selected Sources"
    )
    assert tool.source_panel.refresh_sources_button.menu() is None
    assert not tool.source_panel.refresh_sources_button.isEnabled()
    assert tool._source_refresh_label("data_0") is None
    tool.source_panel.refresh_requested.emit(tool.source_panel.selected_names())
    assert not tool.source_panel.source_status_label.isHidden()

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

    first_item = tool.source_panel.source_list.topLevelItem(0)
    second_item = tool.source_panel.source_list.topLevelItem(1)
    assert first_item is not None
    assert second_item is not None
    assert first_item.icon(0).isNull()
    assert second_item.icon(0).isNull()
    assert first_item.data(0, QtCore.Qt.ItemDataRole.AccessibleDescriptionRole)
    assert tool.source_panel.refresh_sources_button.isEnabled()

    tool.source_panel.refresh_sources_button.click()
    assert refreshed == ["data_0"]
    assert not tool.source_panel.source_status_label.isHidden()

    refreshed.clear()
    tool._refresh_all_sources_from_button()
    assert refreshed == ["data_0"]
    assert not tool.source_panel.source_status_label.isHidden()

    tool.source_panel.set_selected_names({"data_0", "profile"}, current_name="data_0")
    tool._refresh_source_controls()
    refreshed.clear()
    tool.source_panel.refresh_sources_button.click()
    assert refreshed == ["data_0"]
    assert not tool.source_panel.source_status_label.isHidden()

    def raise_refresh(name: str) -> bool:
        raise RuntimeError(f"{name} is incompatible")

    tool.source_panel.set_selected_names({"data_0"}, current_name="data_0")
    tool._set_source_refresh_callbacks(
        can_refresh_source=can_refresh_source,
        refresh_source=raise_refresh,
        source_label=source_label,
    )
    tool.source_panel.refresh_sources_button.click()
    assert not tool.source_panel.source_status_label.isHidden()

    tool._set_source_refresh_callbacks(
        can_refresh_source=lambda _name: False,
        refresh_source=refresh_source,
        source_label=source_label,
    )
    assert not tool.source_panel.refresh_sources_button.isEnabled()
    refreshed.clear()
    tool.source_panel.refresh_requested.emit(tool.source_panel.selected_names())
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
    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )

    before_status = tool.tool_status
    before_source_data = tool.source_data()
    tool.source_panel.set_selected_names({"profile"}, current_name="profile")
    tool._refresh_source_controls()
    tool._refresh_source_detail_panel()
    assert not tool.source_panel.remove_selected_source_button.isEnabled()
    assert (
        tool.source_panel.source_detail_content.property(
            "figureComposerSourceUsageCount"
        )
        == 1
    )
    tool.source_panel.remove_requested.emit(tool.source_panel.selected_names())
    assert tool.tool_status == before_status
    assert set(tool.source_data()) == set(before_source_data)
    assert not tool.remove_source("profile")
    assert not tool.remove_source("missing")

    tool.source_panel.set_selected_names({"unused"}, current_name="unused")
    tool._refresh_source_controls()
    assert tool.source_panel.remove_selected_source_button.isEnabled()
    tool.source_panel.remove_selected_source_button.click()
    assert "unused" not in tool.source_data()
    assert tool.source_panel.source_status_label.isHidden()


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

    data_item = tool.source_panel.source_list.topLevelItem(0)
    missing_item = tool.source_panel.source_list.topLevelItem(1)
    assert data_item is not None
    assert missing_item is not None
    assert data_item.icon(0).isNull()
    assert missing_item.icon(0).isNull()
    assert missing_item.data(0, QtCore.Qt.ItemDataRole.AccessibleDescriptionRole)

    tool.resize(900, 650)
    tool.show()
    QtWidgets.QApplication.processEvents()
    header = tool.source_panel.source_list.header()
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
    assert tool.source_panel.source_status_label.isHidden()
    assert not tool.source_panel.remove_selected_source_button.isEnabled()

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
    assert tool.source_panel.source_list.findChildren(QtWidgets.QToolButton) == []


def test_figure_composer_remove_selected_sources_coalesces_history(qtbot) -> None:
    data = xr.DataArray(np.arange(3.0), dims=("x",), name="data")
    first_unused = data.copy(data=np.arange(3.0) + 10.0).rename("first_unused")
    second_unused = data.copy(data=np.arange(3.0) + 20.0).rename("second_unused")
    tool = FigureComposerTool.from_sources(
        {
            "data": data,
            "first_unused": first_unused,
            "second_unused": second_unused,
        },
        sources=(
            FigureSourceState(name="data"),
            FigureSourceState(name="first_unused"),
            FigureSourceState(name="second_unused"),
        ),
        operations=(FigureOperationState.line(label="line", source="data"),),
        primary_source="data",
    )
    qtbot.addWidget(tool)
    tool._reset_history_stack()
    tool.source_panel.set_selected_names(
        {"first_unused", "second_unused"}, current_name="first_unused"
    )

    tool.source_panel.remove_requested.emit(tool.source_panel.selected_names())

    assert tuple(tool.source_data()) == ("data",)
    assert len(tool._prev_states) == 2
    tool.undo()
    assert tuple(tool.source_data()) == (
        "data",
        "first_unused",
        "second_unused",
    )
    assert not tool.undoable
    assert tool.redoable

    tool.redo()
    assert tuple(tool.source_data()) == ("data",)
    assert tool.undoable
    assert not tool.redoable


def test_figure_composer_cannot_remove_source_used_by_selected_alias(qtbot) -> None:
    base = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("eV", "x"),
        coords={"eV": [-1.0, 0.0], "x": [0.0, 1.0, 2.0]},
        name="base",
    )
    other = xr.DataArray(np.arange(3.0), dims=("x",), name="other")
    tool = FigureComposerTool.from_sources(
        {
            "base": base,
            "selected": base.qsel(eV=0.0),
            "other": other,
        },
        sources=(
            FigureSourceState(name="base"),
            FigureSourceState(
                name="selected",
                qsel={"eV": 0.0},
                selection_source="base",
            ),
            FigureSourceState(name="other"),
        ),
        operations=(FigureOperationState.line(label="line", source="other"),),
        primary_source="other",
    )
    qtbot.addWidget(tool)

    assert not tool.remove_source("base")
    assert tool.remove_source("selected")
    assert tool.remove_source("base")


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
    tool.source_panel.set_selected_names({"data_0"}, current_name="data_0")
    tool._refresh_source_detail_panel()
    tool._refresh_source_selection_editor()

    alias_edit = tool.source_panel.source_alias_edit
    assert alias_edit.text() == "data_0"
    tool.source_panel.rename_requested.emit("data_0", "renamed")

    assert tuple(source.name for source in tool.source_states()) == (
        "renamed",
        "data_1",
    )
    assert tool.tool_status.primary_source == "renamed"
    assert tuple(tool.source_data()) == ("renamed", "data_1")
    assert tool.tool_status.operations[0].sources == ("renamed", "data_1")
    assert tool.tool_status.operations[1].line_source == "renamed"
    assert tool._document.source_by_name()["renamed"].label == "Legacy first"
    assert (
        tool.source_panel.source_list.topLevelItem(0).data(
            0, QtCore.Qt.ItemDataRole.UserRole
        )
        == "renamed"
    )

    assert tool._document.source_alias_error("data_1", current="renamed") is not None


def test_figure_composer_source_alias_rename_keeps_own_selection_implicit(
    qtbot,
) -> None:
    base = xr.DataArray(np.arange(4.0), dims=("x",), name="base")
    selected = base.isel(x=slice(1, None))
    tool = FigureComposerTool.from_sources(
        {"selected": selected},
        sources=(
            FigureSourceState(
                name="selected",
                selection_source="selected",
                isel={"x": slice(1, None)},
            ),
        ),
        operations=(FigureOperationState.line(label="line", source="selected"),),
        primary_source="selected",
    )
    qtbot.addWidget(tool)
    tool._document.source_selection_base_data["selected"] = base

    assert tool._document.rename_source("selected", "renamed")

    [source] = tool.source_states()
    assert source.name == "renamed"
    assert source.selection_source is None
    xr.testing.assert_identical(tool.source_data()["renamed"], selected)
    xr.testing.assert_identical(
        tool._document.source_selection_base_data["renamed"], base
    )


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
    tool._document.source_selection_base_data["second"] = first
    tool.source_panel.set_selected_names({"second"}, current_name="second")
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
        tool.source_panel.source_list.dragDropMode()
        == QtWidgets.QAbstractItemView.DragDropMode.InternalMove
    )
    assert (
        tool.source_panel.source_list.defaultDropAction()
        == QtCore.Qt.DropAction.MoveAction
    )
    assert tool.source_panel.source_list.showDropIndicator()

    menu, move_up_action = _source_context_action(
        tool, "figureComposerContextMoveSourceUpAction"
    )
    move_down_action = next(
        action
        for action in menu.actions()
        if action.objectName() == "figureComposerContextMoveSourceDownAction"
    )
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
    duplicate = tool._document.source_by_name()["second_copy"]
    assert duplicate.label == "second_copy"
    assert duplicate.isel == {"x": 0}
    assert duplicate.selection_source == "first"
    xr.testing.assert_identical(tool.source_data()["second_copy"], second)
    xr.testing.assert_identical(
        tool._document.source_selection_base_data["second_copy"], first
    )
    assert tool.source_panel.current_name() == "second_copy"

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
    assert tool.source_panel.current_name() == "second_copy"

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
    assert tool.source_panel.current_name() == "second_copy"


def test_figure_composer_duplicate_source_preserves_historical_label(qtbot) -> None:
    source_data = {
        name: xr.DataArray(np.arange(2.0), dims=("x",), name=name)
        for name in ("automatic", "historical", "keeper")
    }
    tool = FigureComposerTool.from_sources(
        source_data,
        sources=(
            FigureSourceState(name="automatic"),
            FigureSourceState(name="historical", label="Historical display label"),
            FigureSourceState(name="keeper"),
        ),
        operations=(),
        primary_source="keeper",
    )
    qtbot.addWidget(tool)
    tool.source_panel.set_selected_names(
        {"automatic", "historical"}, current_name="automatic"
    )

    tool.source_panel.duplicate_requested.emit(tool.source_panel.selected_names())

    source_by_name = tool._document.source_by_name()
    assert source_by_name["automatic_copy"].label == "automatic_copy"
    assert source_by_name["historical_copy"].label == "Historical display label"


def test_figure_composer_duplicate_selected_source_generated_code_uses_raw_base(
    qtbot, monkeypatch
) -> None:
    base = xr.DataArray(
        np.arange(8.0).reshape(2, 2, 2),
        dims=("hv", "x", "y"),
        coords={"hv": [40.0, 50.0], "x": [0.0, 1.0], "y": [0.0, 1.0]},
        name="data",
    )
    selected = base.qsel(hv=40.0)
    tool = FigureComposerTool.from_sources(
        {"data": selected},
        sources=(
            FigureSourceState(
                name="data",
                qsel={"hv": 40.0},
                selection_source="data",
            ),
        ),
        operations=(
            FigureOperationState.plot_array(
                label="original",
                source="data",
                axes=FigureAxesSelectionState(axes=((0, 0),)),
            ),
        ),
        setup=FigureSubplotsState(nrows=1, ncols=2),
        primary_source="data",
    )
    qtbot.addWidget(tool)
    tool._document.source_selection_base_data["data"] = base
    tool.source_panel.set_selected_names({"data"}, current_name="data")
    tool.source_panel.duplicate_requested.emit(tool.source_panel.selected_names())
    assert tool._document.source_by_name()["data_copy"].selection_source is None
    tool.tool_status = tool.tool_status.model_copy(
        update={
            "operations": (
                *tool.tool_status.operations,
                FigureOperationState.plot_array(
                    label="copy",
                    source="data_copy",
                    axes=FigureAxesSelectionState(axes=((0, 1),)),
                ),
            )
        }
    )

    captured: list[xr.DataArray] = []
    monkeypatch.setattr(
        eplt,
        "plot_array",
        lambda data, **_kwargs: captured.append(data),
    )
    namespace = _exec_generated_code(
        tool.generated_code(), {"data": base, "data_copy": base}
    )

    assert isinstance(namespace["fig"], Figure)
    assert len(captured) == 2
    xr.testing.assert_identical(captured[0], selected)
    xr.testing.assert_identical(captured[1], selected)


def test_figure_composer_generated_code_resolves_selected_source_dependencies(
    qtbot, monkeypatch
) -> None:
    base = xr.DataArray(
        np.arange(24.0).reshape(2, 3, 2, 2),
        dims=("u", "v", "x", "y"),
        coords={
            "u": [0.0, 1.0],
            "v": [10.0, 20.0, 30.0],
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
        },
        name="base",
    )
    selected_u = base.qsel(u=1.0)
    selected_v = selected_u.qsel(v=20.0)
    tool = FigureComposerTool.from_sources(
        {
            "base": base,
            "selected_u": selected_u,
            "selected_v": selected_v,
        },
        sources=(
            FigureSourceState(
                name="selected_v",
                selection_source="selected_u",
                qsel={"v": 20.0},
            ),
            FigureSourceState(name="base"),
            FigureSourceState(
                name="selected_u",
                selection_source="base",
                qsel={"u": 1.0},
            ),
        ),
        operations=(
            FigureOperationState.plot_array(label="selected", source="selected_v"),
        ),
        primary_source="base",
    )
    qtbot.addWidget(tool)
    captured: list[xr.DataArray] = []
    monkeypatch.setattr(
        eplt,
        "plot_array",
        lambda data, **_kwargs: captured.append(data),
    )

    code = tool.generated_code()

    assert code.index("selected_u = base.qsel(u=1.0)") < code.index(
        "selected_v = selected_u.qsel(v=20.0)"
    )
    _exec_generated_code(code, {"base": base})
    xr.testing.assert_identical(captured[-1], selected_v)

    code_with_materialized_parent = figurecomposer_codegen.generated_code(
        tool,
        skip_source_selection_names=frozenset({"selected_u"}),
    )

    assert "selected_u =" not in code_with_materialized_parent
    assert "base.qsel" not in code_with_materialized_parent
    _exec_generated_code(
        code_with_materialized_parent,
        {"selected_u": selected_u},
    )
    xr.testing.assert_identical(captured[-1], selected_v)

    cyclic_sources = tuple(
        source.model_copy(update={"selection_source": "selected_v"})
        if source.name == "selected_u"
        else source
        for source in tool.source_states()
    )
    tool.tool_status = tool.tool_status.model_copy(update={"sources": cyclic_sources})
    with pytest.raises(
        FigureComposerInputError,
        match=r"dependency cycle: selected_v -> selected_u -> selected_v",
    ):
        tool.generated_code()


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
    tool.source_panel.set_selected_names({"b", "c"}, current_name="b")

    first_moved = tool.source_panel.source_list.takeTopLevelItem(1)
    second_moved = tool.source_panel.source_list.takeTopLevelItem(1)
    assert first_moved is not None
    assert second_moved is not None
    tool.source_panel.source_list.insertTopLevelItem(2, first_moved)
    tool.source_panel.source_list.insertTopLevelItem(3, second_moved)
    tool.source_panel.set_selected_names({"b", "c"}, current_name="b")
    tool.source_panel.source_list._queue_rows_reordered()

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
    assert tool.source_panel.current_name() == "b"
    assert set(tool.source_panel.selected_names()) == {"b", "c"}


def test_figure_composer_source_list_keyboard_context_menu(qtbot) -> None:
    data = {name: _figure_composer_profile_source(name) for name in ("a", "b")}
    tool = FigureComposerTool.from_sources(
        data,
        sources=tuple(FigureSourceState(name=name) for name in data),
        setup=FigureSubplotsState(),
        primary_source="a",
    )
    qtbot.addWidget(tool)
    tool.source_panel.set_selected_names({"b"}, current_name="b")
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
    tool.source_panel.source_list.keyPressEvent(event)

    assert event.isAccepted()
    refresh_all_action = tool.source_panel.source_list.findChild(
        QtGui.QAction, "figureComposerContextRefreshAllSourcesAction"
    )
    assert refresh_all_action is not None
    menu = refresh_all_action.parent()
    assert isinstance(menu, QtWidgets.QMenu)
    action_names = {
        action.objectName() for action in menu.actions() if action.objectName()
    }
    assert "figureComposerContextMoveSourceUpAction" in action_names
    assert "figureComposerContextMoveSourceDownAction" in action_names
    assert refresh_all_action.isEnabled()
    refresh_all_action.trigger()
    assert refreshed == ["a", "b"]
    menu.close()


def test_figure_composer_data_only_source_context_menu_disables_rename(qtbot) -> None:
    source = _figure_composer_profile_source("source")
    data_only = _figure_composer_profile_source("data_only")
    tool = FigureComposerTool.from_sources(
        {"source": source, "data_only": data_only},
        sources=(FigureSourceState(name="source"),),
        operations=(),
        setup=FigureSubplotsState(),
        primary_source="source",
    )
    qtbot.addWidget(tool)
    tool.source_panel.set_selected_names({"data_only"}, current_name="data_only")
    tool._refresh_source_detail_panel()

    assert not tool.source_panel.source_alias_edit.isEnabled()
    menu, rename_action = _source_context_action(
        tool, "figureComposerContextRenameSourceAction"
    )
    triggered: list[bool] = []
    rename_action.triggered.connect(triggered.append)

    assert not rename_action.isEnabled()
    rename_action.trigger()
    assert triggered == []
    menu.close()


def test_figure_composer_source_context_menus_are_released(qtbot) -> None:
    tool = FigureComposerTool.from_sources(
        {"a": _figure_composer_profile_source("a")},
        sources=(FigureSourceState(name="a"),),
        setup=FigureSubplotsState(),
        primary_source="a",
    )
    qtbot.addWidget(tool)

    tool.source_panel._show_context_menu(QtCore.QPoint(0, 0))
    first_menus = tool.source_panel.source_list.findChildren(QtWidgets.QMenu)
    assert len(first_menus) == 1
    first_menu = first_menus[0]
    tool.source_panel._show_context_menu(QtCore.QPoint(0, 0))
    menus = tool.source_panel.source_list.findChildren(QtWidgets.QMenu)
    assert len(menus) == 2
    second_menu = next(menu for menu in menus if menu is not first_menu)

    first_menu.close()
    qtbot.waitUntil(lambda: not erlab.interactive.utils.qt_is_valid(first_menu))
    assert tool.source_panel.source_list.findChildren(QtWidgets.QMenu) == [second_menu]

    second_menu.close()
    qtbot.waitUntil(lambda: not erlab.interactive.utils.qt_is_valid(second_menu))
    assert tool.source_panel.source_list.findChildren(QtWidgets.QMenu) == []


def test_figure_composer_source_list_edge_events(qtbot) -> None:
    source_list = figurecomposer_source_panel._FigureSourceList()
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
    tool.source_panel.set_selected_names({"first"}, current_name="first")
    tool._refresh_source_detail_panel()
    tool._refresh_source_selection_editor()
    alias_edit = tool.source_panel.source_alias_edit

    alias_edit.setText("first")
    alias_edit.editingFinished.emit()
    assert tuple(source.name for source in tool.source_states()) == ("first", "second")

    alias_edit.setText("second")
    alias_edit.editingFinished.emit()
    assert not tool.source_panel.source_validation_label.isHidden()
    assert tool.source_panel.source_status_label.isHidden()
    assert alias_edit.text() == "second"

    alias_edit.setText("not valid")
    alias_edit.editingFinished.emit()
    assert tuple(source.name for source in tool.source_states()) == ("first", "second")
    assert not tool.source_panel.source_validation_label.isHidden()
    assert alias_edit.text() == "not valid"

    alias_edit.setText("list")
    alias_edit.editingFinished.emit()
    assert tuple(source.name for source in tool.source_states()) == ("first", "second")
    assert not tool.source_panel.source_validation_label.isHidden()
    assert alias_edit.text() == "list"

    with monkeypatch.context() as patch:
        patch.setattr(tool._document, "rename_source", lambda *_args: False)
        alias_edit.setText("third")
        alias_edit.editingFinished.emit()
    assert alias_edit.text() == "first"

    tool.source_panel.set_selected_names(set(), current_name=None)
    tool.source_panel.focus_alias_editor()
    tool.source_panel.set_selected_names({"first"}, current_name="first")
    tool._refresh_source_detail_panel()
    tool.source_panel.source_alias_edit.setEnabled(False)
    tool.source_panel.focus_alias_editor()
    assert not tool.source_panel.source_alias_edit.isEnabled()
    tool.source_panel.source_alias_edit.setEnabled(True)

    alias_edit.setText("renamed")
    alias_edit.editingFinished.emit()
    assert tuple(source.name for source in tool.source_states()) == (
        "renamed",
        "second",
    )
    assert tool._document.source_by_name()["renamed"].label == "renamed"
    assert "renamed" in tool._document.source_data


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
    tool.source_panel.set_selected_names({"second"}, current_name="second")

    tool.source_panel.remove_requested.emit(tool.source_panel.selected_names())

    assert tuple(source.name for source in tool.source_states()) == ("first",)
    assert "second" not in tool._document.source_data


def test_figure_composer_remove_selected_sources_resolves_dependencies(qtbot) -> None:
    base = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0, 2.0]},
        name="base",
    )
    selected = base.qsel(x=1.0)
    other = xr.DataArray(np.arange(3.0), dims=("y",), name="other")
    tool = FigureComposerTool.from_sources(
        {"other": other, "base": base, "selected": selected},
        sources=(
            FigureSourceState(name="other"),
            FigureSourceState(name="base"),
            FigureSourceState(
                name="selected", selection_source="base", qsel={"x": 1.0}
            ),
        ),
        operations=(),
        primary_source="other",
    )
    qtbot.addWidget(tool)
    tool.source_panel.set_selected_names({"base", "selected"}, current_name="base")

    tool.source_panel.remove_requested.emit(tool.source_panel.selected_names())

    assert tuple(source.name for source in tool.source_states()) == ("other",)
    assert set(tool.source_data()) == {"other"}


def test_figure_composer_remove_all_selected_sources_keeps_last_row(qtbot) -> None:
    source_data = {
        name: xr.DataArray(np.arange(2.0), dims=("x",), name=name)
        for name in ("first", "second", "third")
    }
    tool = FigureComposerTool.from_sources(
        source_data,
        sources=tuple(FigureSourceState(name=name) for name in source_data),
        operations=(),
        primary_source="first",
    )
    qtbot.addWidget(tool)
    tool.source_panel.set_selected_names(set(source_data), current_name="first")

    tool.source_panel.remove_requested.emit(tool.source_panel.selected_names())

    assert tuple(source.name for source in tool.source_states()) == ("third",)
    assert set(tool.source_data()) == {"third"}


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
        tool.source_panel.source_list.clearSelection()
        tool.source_panel.source_list.setCurrentIndex(QtCore.QModelIndex())

    clear_source_current()
    tool.source_panel.remove_requested.emit(tool.source_panel.selected_names())
    clear_source_current()
    tool.source_panel.duplicate_requested.emit(tool.source_panel.selected_names())
    clear_source_current()
    tool.source_panel.move_requested.emit(tool.source_panel.selected_names(), 1)
    assert not tool._document.can_move_sources((), 1)
    assert tuple(source.name for source in tool.source_states()) == (
        "first",
        "second",
    )

    tool.source_panel.set_selected_names({"first"}, current_name="first")
    tool.source_panel.move_requested.emit(tool.source_panel.selected_names(), -1)
    assert tuple(source.name for source in tool.source_states()) == (
        "first",
        "second",
    )
    tool.source_panel.rename_requested.emit("missing", "renamed")
    assert tuple(source.name for source in tool.source_states()) == (
        "first",
        "second",
    )

    assert tool._document.source_alias_error("") is not None
    assert tool._document.source_alias_error("bad name") is not None
    assert tool._document.source_alias_error(
        erlab.interactive.utils._SAVED_TOOL_DATA_NAME
    )
    assert tool._document.source_alias_error("fig") is not None
    assert tool._document.source_alias_error("second", current="first") is not None

    tool._refresh_source_detail_panel()
    tool._refresh_source_selection_editor()
    alias_edit = tool.source_panel.source_alias_edit
    alias_edit.setText("bvec")
    alias_edit.editingFinished.emit()
    assert alias_edit.text() == "bvec"
    assert not tool.source_panel.source_validation_label.isHidden()
    assert tool.source_panel.source_status_label.isHidden()
    tool.source_panel.focus_alias_editor()
    first_item = tool.source_panel.source_list.topLevelItem(0)
    assert first_item is not None
    tool.source_panel.source_list.itemDoubleClicked.emit(first_item, 0)
    tool.source_panel.rename_requested.emit("first", "renamed")
    assert tuple(source.name for source in tool.source_states()) == (
        "renamed",
        "second",
    )

    tool.source_panel.set_selected_names({"renamed", "second"}, current_name="renamed")
    tool.source_panel.focus_alias_editor()
    tool._source_list_reordered("not-a-sequence", set(), None)
    tool._source_list_reordered(("renamed", "renamed"), set(), None)
    tool._source_list_reordered(("renamed", "second"), set(), None)
    tool._source_list_reordered(("second", "renamed"), (), None)
    assert tuple(source.name for source in tool.source_states()) == (
        "second",
        "renamed",
    )


def test_figure_composer_source_selection_editor_widget_visibility(qtbot) -> None:
    panel = figurecomposer_source_panel.FigureSourcePanel()
    qtbot.addWidget(panel)
    row = figurecomposer_source_panel.FigureSourceSelectionRow(
        dimension="x",
        tooltip="",
        mode=None,
        mode_mixed=False,
        value_text="",
        value_mixed=False,
        width_text="",
        width_mixed=False,
    )

    conditional_widgets = (
        panel._selection_section,
        panel._selection_message_label,
        panel._selection_message,
    )
    assert all(widget.isHidden() for widget in conditional_widgets)

    panel.set_selection_editor((row,))
    assert not panel._selection_section.isHidden()
    assert panel._selection_message_label.isHidden()
    assert panel._selection_message.isHidden()

    panel.set_selection_editor((), message="Unavailable")
    assert all(not widget.isHidden() for widget in conditional_widgets)

    panel.set_selection_editor(())
    assert all(widget.isHidden() for widget in conditional_widgets)


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

    tool.source_panel.source_list.clearSelection()
    tool.source_panel.source_list.setCurrentIndex(QtCore.QModelIndex())
    tool._refresh_source_detail_panel()
    tool._refresh_source_selection_editor()
    flush_deferred_editor_deletes()
    assert tool.source_panel.source_selection_controls_layout.count() == 0
    assert (
        tool.source_panel.source_detail_content.property(
            "figureComposerSourceEditorMode"
        )
        == "empty"
    )

    tool.source_panel.set_selected_names({"scalar"}, current_name="scalar")
    tool._refresh_source_detail_panel()
    tool._refresh_source_selection_editor()
    flush_deferred_editor_deletes()
    item = tool.source_panel.source_selection_controls_layout.itemAt(
        tool.source_panel.source_selection_controls_layout.rowCount() - 1,
        QtWidgets.QFormLayout.ItemRole.FieldRole,
    )
    assert item is not None
    assert isinstance(item.widget(), QtWidgets.QLabel)

    tool.source_panel.source_list.clearSelection()
    first_item = tool.source_panel.source_list.topLevelItem(0)
    second_item = tool.source_panel.source_list.topLevelItem(1)
    assert first_item is not None
    assert second_item is not None
    tool.source_panel.source_list.setCurrentItem(first_item)
    first_item.setSelected(True)
    second_item.setSelected(True)
    tool._refresh_source_detail_panel()
    tool._refresh_source_selection_editor()
    flush_deferred_editor_deletes()
    combo = None
    for row in range(tool.source_panel.source_selection_controls_layout.rowCount()):
        item = tool.source_panel.source_selection_controls_layout.itemAt(
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
        tool.source_panel.source_detail_content.property(
            "figureComposerSourceEditorMode"
        )
        == "multiple"
    )
    assert (
        tool.source_panel.source_detail_content.property(
            "figureComposerSourceSelectionCount"
        )
        == 2
    )
    assert not tool.source_panel.source_alias_controls.isVisible()
    assert tool.source_panel.source_inspector.isHidden()

    tool._document.source_data.clear()
    tool.source_panel.selection_dimension_requested.emit(
        tool.source_panel.selected_names(), "x", "isel", "0", ""
    )
    assert tool._document.source_data == {}


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
    assert tool._document.source_selection_input_data("derived") is data

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
    assert tool._document.source_by_name()["data_2"].label == "data_2"
    tool.add_sources(
        (FigureSourceState(name="data", label="Historical incoming"),),
        {"data": data},
    )
    assert tuple(source.name for source in tool.source_states()) == (
        "data",
        "derived",
        "data_2",
        "data_3",
    )
    assert tool._document.source_by_name()["data_3"].label == "Historical incoming"


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
    assert sources["data_selected_2"].label == "data_selected_2"
    assert sources["data_selected_2"].isel == {"kx": 1}


@pytest.mark.parametrize(
    ("operation_sources", "selection_values", "expected"),
    (
        (("first",), (("first", 0.0), ("first", 2.0)), (0.0, 2.0)),
        (
            ("first", "second"),
            (("first", 1.0),),
            (1.0, "second"),
        ),
        (
            ("first", "first"),
            (("first", 0.0), ("first", 2.0)),
            (0.0, 2.0),
        ),
        ((), (("first", 2.0), ("first", 0.0)), (2.0, 0.0)),
    ),
)
def test_figure_composer_plot_slices_legacy_selections_preserve_source_order(
    qtbot,
    operation_sources: tuple[str, ...],
    selection_values: tuple[tuple[str, float], ...],
    expected: tuple[float | str, ...],
) -> None:
    first = xr.DataArray(
        np.arange(12.0).reshape(3, 2, 2),
        dims=("cut", "x", "y"),
        coords={"cut": [0.0, 1.0, 2.0], "x": [0.0, 1.0], "y": [0.0, 1.0]},
        name="first",
    )
    second = (first + 100.0).rename("second")
    operation = FigureOperationState.plot_slices(
        label="maps",
        sources=operation_sources,
        map_selections=tuple(
            FigureDataSelectionState(source=source, qsel={"cut": value})
            for source, value in selection_values
        ),
    )
    tool = FigureComposerTool.from_sources(
        {"first": first, "second": second},
        sources=(FigureSourceState(name="first"), FigureSourceState(name="second")),
        operations=(operation,),
        primary_source="first",
    )
    qtbot.addWidget(tool)

    [loaded_operation] = tool.tool_status.operations
    assert loaded_operation.map_selections == ()
    assert len(loaded_operation.sources) == len(expected)
    for source_name, expected_value in zip(
        loaded_operation.sources, expected, strict=True
    ):
        if expected_value == "second":
            assert source_name == "second"
            xr.testing.assert_identical(tool.source_data()[source_name], second)
        else:
            selected = first.qsel(cut=expected_value)
            xr.testing.assert_identical(tool.source_data()[source_name], selected)
            source = tool._document.source_by_name()[source_name]
            assert source.selection_source == "first"
            assert source.qsel == {"cut": expected_value}


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
    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )

    assert tool.tool_status.operations[0].sources == ("data_0",)
    assert tool.tool_status.operations[1].line_source == "data_0"

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
    assert source.label == "ImageTool 1: replacement"
    assert source.node_uid == "new-node"
    assert source.node_snapshot_token == new_snapshot_id
    assert source.provenance_spec == {"source": "new"}
    xr.testing.assert_identical(tool.source_data()["data_0"], replacement)
    assert tool.tool_status.operations[0].sources == ("data_0",)
    assert tool.tool_status.operations[1].line_source == "data_0"

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
    tool._document.source_data["orphan"] = original
    assert tool.replace_source(
        "orphan",
        FigureSourceState(name="data_2", label="Historical orphan"),
        original,
    )
    tool._document.source_data["default_orphan"] = original
    assert tool.replace_source(
        "default_orphan",
        FigureSourceState(name="data_3"),
        original,
    )
    assert tuple(source.name for source in tool.source_states()) == (
        "data_0",
        "orphan",
        "default_orphan",
    )
    assert tool._document.source_by_name()["orphan"].label == "Historical orphan"
    assert tool._document.source_by_name()["default_orphan"].label == "default_orphan"


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
    assert source.selection_source is None

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
    assert not tool.source_panel.source_status_label.isHidden()

    refreshed = original + 20.0
    tool.refresh_from_sources({"data": refreshed})
    xr.testing.assert_identical(tool.source_data()["data"], refreshed.qsel(eV=0.0))
    assert tool.source_panel.source_status_label.isHidden()


def _transitive_selected_source_tool() -> tuple[
    FigureComposerTool,
    xr.DataArray,
    xr.DataArray,
    xr.DataArray,
]:
    base = xr.DataArray(
        np.arange(24.0).reshape(2, 3, 4),
        dims=("u", "v", "w"),
        coords={"u": [0.0, 1.0], "v": [0.0, 1.0, 2.0], "w": np.arange(4)},
        name="base",
    )
    selected_u = base.qsel(u=1.0)
    selected_v = selected_u.qsel(v=2.0)
    tool = FigureComposerTool.from_sources(
        {"base": base, "selected_u": selected_u, "selected_v": selected_v},
        sources=(
            FigureSourceState(name="base", node_uid="base-node"),
            FigureSourceState(
                name="selected_u",
                selection_source="base",
                qsel={"u": 1.0},
            ),
            FigureSourceState(
                name="selected_v",
                selection_source="selected_u",
                qsel={"v": 2.0},
            ),
        ),
        operations=(FigureOperationState.line(label="line", source="selected_v"),),
        primary_source="base",
    )
    tool._document.source_selection_base_data.update(
        {"selected_u": base, "selected_v": selected_u}
    )
    return tool, base, selected_u, selected_v


@pytest.mark.parametrize("mutation", ("replace", "add", "refresh"))
def test_figure_composer_source_refresh_recomputes_transitive_selected_sources(
    qtbot,
    mutation: str,
) -> None:
    tool, base, _selected_u, _selected_v = _transitive_selected_source_tool()
    qtbot.addWidget(tool)
    replacement = base + 100.0

    if mutation == "replace":
        assert tool.replace_source(
            "base",
            FigureSourceState(name="incoming", node_uid="replacement-node"),
            replacement,
        )
    elif mutation == "add":
        tool.add_sources(
            (FigureSourceState(name="base", node_uid="base-node"),),
            {"base": replacement},
        )
    else:
        tool.refresh_from_sources({"base": replacement})

    expected_u = replacement.qsel(u=1.0)
    expected_v = expected_u.qsel(v=2.0)
    xr.testing.assert_identical(tool.source_data()["base"], replacement)
    xr.testing.assert_identical(tool.source_data()["selected_u"], expected_u)
    xr.testing.assert_identical(tool.source_data()["selected_v"], expected_v)
    xr.testing.assert_identical(
        tool._document.source_selection_base_data["selected_u"], replacement
    )
    xr.testing.assert_identical(
        tool._document.source_selection_base_data["selected_v"], expected_u
    )
    if mutation == "replace":
        assert {source.node_uid for source in tool.source_states()} == {
            "replacement-node"
        }
    assert tool.source_panel.source_status_label.isHidden()


def test_figure_composer_refreshing_selected_alias_updates_linked_root(qtbot) -> None:
    tool, base, _selected_u, _selected_v = _transitive_selected_source_tool()
    qtbot.addWidget(tool)
    replacement = base + 100.0
    provenance_spec = {"kind": "new-origin"}
    snapshot_id = "new-snapshot"

    assert tool.replace_source(
        "selected_v",
        FigureSourceState(
            name="manager_default",
            node_uid="base-node",
            node_snapshot_token=snapshot_id,
            provenance_spec=provenance_spec,
        ),
        replacement,
    )

    source_by_name = tool._document.source_by_name()
    assert source_by_name["base"].selection_source is None
    assert source_by_name["selected_u"].selection_source == "base"
    assert source_by_name["selected_v"].selection_source == "selected_u"
    assert {source.node_uid for source in source_by_name.values()} == {"base-node"}
    assert {source.node_snapshot_token for source in source_by_name.values()} == {
        snapshot_id
    }
    assert all(
        source.provenance_spec == provenance_spec for source in source_by_name.values()
    )
    expected_u = replacement.qsel(u=1.0)
    expected_v = expected_u.qsel(v=2.0)
    xr.testing.assert_identical(tool.source_data()["base"], replacement)
    xr.testing.assert_identical(tool.source_data()["selected_u"], expected_u)
    xr.testing.assert_identical(tool.source_data()["selected_v"], expected_v)


def test_figure_composer_readding_shared_link_updates_root_without_new_alias(
    qtbot,
) -> None:
    tool, base, _selected_u, _selected_v = _transitive_selected_source_tool()
    qtbot.addWidget(tool)
    replacement = base + 100.0

    result = tool.add_sources(
        (FigureSourceState(name="manager_default", node_uid="base-node"),),
        {"manager_default": replacement},
    )

    assert result.added == ()
    assert result.updated == (("manager_default", "base"),)
    assert result.skipped == ()
    assert result.name_map == {"manager_default": "base"}
    assert tuple(tool._document.source_by_name()) == (
        "base",
        "selected_u",
        "selected_v",
    )
    expected_u = replacement.qsel(u=1.0)
    expected_v = expected_u.qsel(v=2.0)
    xr.testing.assert_identical(tool.source_data()["base"], replacement)
    xr.testing.assert_identical(tool.source_data()["selected_u"], expected_u)
    xr.testing.assert_identical(tool.source_data()["selected_v"], expected_v)


def test_figure_composer_add_sources_reports_partial_batch_outcome(qtbot) -> None:
    tool, base, _selected_u, _selected_v = _transitive_selected_source_tool()
    qtbot.addWidget(tool)
    incompatible = base.isel(v=0, drop=True) + 100.0
    extra = xr.DataArray(np.arange(3.0), dims=("x",), name="extra")

    result = tool.add_sources(
        (
            FigureSourceState(name="incoming", node_uid="base-node"),
            FigureSourceState(name="extra", node_uid="extra-node"),
        ),
        {"incoming": incompatible, "extra": extra},
    )

    assert result.added == (("extra", "extra"),)
    assert result.updated == ()
    assert result.skipped[0][0] == "incoming"
    assert "base" in result.skipped[0][1]
    assert result.name_map == {"extra": "extra"}
    assert result
    xr.testing.assert_identical(tool.source_data()["base"], base)
    xr.testing.assert_identical(tool.source_data()["extra"], extra)
    assert not tool.source_panel.source_status_label.isHidden()


@pytest.mark.parametrize("mutation", ("replace", "add", "refresh"))
def test_figure_composer_source_refresh_is_atomic_when_dependent_selection_fails(
    qtbot,
    mutation: str,
) -> None:
    tool, base, _selected_u, _selected_v = _transitive_selected_source_tool()
    qtbot.addWidget(tool)
    original_status = tool.tool_status
    original_data = dict(tool.source_data())
    original_bases = dict(tool._document.source_selection_base_data)
    incompatible = base.isel(v=0, drop=True) + 100.0

    if mutation == "replace":
        assert not tool.replace_source(
            "base",
            FigureSourceState(name="incoming", node_uid="replacement-node"),
            incompatible,
        )
    elif mutation == "add":
        tool.add_sources(
            (FigureSourceState(name="base", node_uid="base-node"),),
            {"base": incompatible},
        )
    else:
        tool.refresh_from_sources({"base": incompatible})

    assert tool.tool_status == original_status
    assert set(tool.source_data()) == set(original_data)
    for source_name, data in original_data.items():
        xr.testing.assert_identical(tool.source_data()[source_name], data)
    assert set(tool._document.source_selection_base_data) == set(original_bases)
    for source_name, data in original_bases.items():
        xr.testing.assert_identical(
            tool._document.source_selection_base_data[source_name], data
        )
    assert not tool.source_panel.source_status_label.isHidden()


def test_figure_composer_selection_edit_recomputes_transitive_selected_sources(
    qtbot,
) -> None:
    tool, base, _selected_u, _selected_v = _transitive_selected_source_tool()
    qtbot.addWidget(tool)
    tool.source_panel.set_selected_names({"selected_u"}, current_name="selected_u")

    tool.source_panel.selection_dimension_requested.emit(
        tool.source_panel.selected_names(), "u", "qsel", "0.0", ""
    )

    expected_u = base.qsel(u=0.0)
    expected_v = expected_u.qsel(v=2.0)
    assert tool._document.source_by_name()["selected_u"].qsel == {"u": 0.0}
    xr.testing.assert_identical(tool.source_data()["selected_u"], expected_u)
    xr.testing.assert_identical(tool.source_data()["selected_v"], expected_v)
    xr.testing.assert_identical(
        tool._document.source_selection_base_data["selected_u"], base
    )
    xr.testing.assert_identical(
        tool._document.source_selection_base_data["selected_v"], expected_u
    )
    assert tool.source_panel.source_validation_label.isHidden()


def test_figure_composer_selection_edit_is_atomic_when_dependent_fails(
    qtbot,
) -> None:
    tool, _base, _selected_u, _selected_v = _transitive_selected_source_tool()
    qtbot.addWidget(tool)
    tool.source_panel.set_selected_names({"selected_u"}, current_name="selected_u")
    original_status = tool.tool_status
    original_data = dict(tool.source_data())
    original_bases = dict(tool._document.source_selection_base_data)

    tool.source_panel.selection_dimension_requested.emit(
        tool.source_panel.selected_names(), "v", "mean", "", ""
    )

    assert tool.tool_status == original_status
    for source_name, data in original_data.items():
        xr.testing.assert_identical(tool.source_data()[source_name], data)
    for source_name, data in original_bases.items():
        xr.testing.assert_identical(
            tool._document.source_selection_base_data[source_name], data
        )
    assert not tool.source_panel.source_validation_label.isHidden()


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
    assert source.selection_source is None
    xr.testing.assert_identical(tool.source_data()["data"], refreshed.qsel(eV=0.0))
    xr.testing.assert_identical(
        tool._document.source_selection_base_data["data"], refreshed
    )

    state_before = tool.tool_status
    data_before = tool.source_data()["data"]
    incompatible = xr.DataArray(np.arange(3.0), dims=("alpha",), name="map")
    tool.add_sources(
        (FigureSourceState(name="data", node_uid="node"),),
        {"data": incompatible},
    )
    assert tool.tool_status == state_before
    xr.testing.assert_identical(tool.source_data()["data"], data_before)
    assert not tool.source_panel.source_status_label.isHidden()


def test_figure_composer_readding_renamed_linked_source_updates_alias(qtbot) -> None:
    original = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("eV", "x"),
        coords={"eV": [-1.0, 0.0], "x": [0.0, 1.0, 2.0]},
        name="map",
    )
    tool = FigureComposerTool.from_sources(
        {"custom_alias": original.qsel(eV=0.0)},
        sources=(
            FigureSourceState(
                name="custom_alias",
                qsel={"eV": 0.0},
                selection_source="custom_alias",
                node_uid="node",
            ),
        ),
        operations=(
            FigureOperationState.plot_array(label="array", source="custom_alias"),
        ),
        primary_source="custom_alias",
    )
    qtbot.addWidget(tool)

    refreshed = original + 10.0
    tool.add_sources(
        (FigureSourceState(name="manager_default", node_uid="node"),),
        {"manager_default": refreshed},
    )

    assert tuple(source.name for source in tool.source_states()) == ("custom_alias",)
    assert tool.source_states()[0].label == "custom_alias"
    assert tool.source_states()[0].qsel == {"eV": 0.0}
    assert tool.source_states()[0].selection_source is None
    xr.testing.assert_identical(
        tool.source_data()["custom_alias"], refreshed.qsel(eV=0.0)
    )
    xr.testing.assert_identical(
        tool._document.source_selection_base_data["custom_alias"], refreshed
    )


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
    assert source.selection_source is None
    xr.testing.assert_identical(tool.source_data()["data"], replacement.qsel(eV=0.0))
    xr.testing.assert_identical(
        tool._document.source_selection_base_data["data"], replacement
    )

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
    assert source.selection_source is None
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
    assert tool.source_panel.source_status_label.isHidden()

    incompatible = second + 10.0
    tool.refresh_from_sources({"first": incompatible})
    assert not tool.source_panel.source_status_label.isHidden()

    refreshed_second = second + 20.0
    tool._document.source_selection_base_data["second"] = second
    tool.refresh_from_sources({"second": refreshed_second})
    xr.testing.assert_identical(tool.source_data()["second"], refreshed_second)
    assert "second" not in tool._document.source_selection_base_data
    assert tool.source_panel.source_status_label.isHidden()

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

    tool.source_panel.source_list.clearSelection()
    first_item = tool.source_panel.source_list.topLevelItem(0)
    second_item = tool.source_panel.source_list.topLevelItem(1)
    assert first_item is not None
    assert second_item is not None
    first_item.setSelected(True)
    second_item.setSelected(True)

    tool.source_panel.selection_dimension_requested.emit(
        tool.source_panel.selected_names(), "x", "isel", "1", ""
    )

    source_by_name = {source.name: source for source in tool.source_states()}
    assert source_by_name["first"].isel == {"x": 1}
    assert source_by_name["second"].isel == {}
    xr.testing.assert_identical(tool.source_data()["first"], first.isel(x=1))
    xr.testing.assert_identical(tool.source_data()["second"], second)
    assert not tool.source_panel.source_validation_label.isHidden()


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
    renamed_script_input = tool._script_input_with_name(script_input, "renamed")
    assert renamed_script_input.name == "renamed"
    assert renamed_script_input.label == "renamed"
    historical_input = script_input.model_copy(update={"label": "Historical input"})
    renamed_historical_input = tool._script_input_with_name(historical_input, "renamed")
    assert renamed_historical_input.name == "renamed"
    assert renamed_historical_input.label == "Historical input"
    historical_source = FigureSourceState.from_script_input(historical_input)
    assert historical_source.label == "Historical input"
    historical_roundtrip = historical_source.to_script_input()
    assert historical_roundtrip is not None
    assert historical_roundtrip.label == "Historical input"

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
    assert selected_input.label == "display_selected"
    selected_spec = selected_input.parsed_provenance_spec()
    assert selected_spec is not None
    assert len(selected_spec.replay_stages) == 1
    assert len(selected_spec.replay_stages[0].operations) == 2

    historical_selected_source = selected_source.model_copy(
        update={"label": "ImageTool 7: gap scan"}
    )
    historical_selected_input = tool._selected_source_script_input(
        historical_selected_source,
        display_name="display_selected",
        source_by_name={
            "base": base_source,
            "base_selected": historical_selected_source,
        },
    )
    assert historical_selected_input is not None
    assert historical_selected_input.label == "ImageTool 7: gap scan"

    live_base = FigureSourceState(name="live_base", node_uid="live-node")
    live_selected_source = FigureSourceState(
        name="live_selected",
        selection_source="live_base",
        qsel={"eV": 0.0},
    )
    live_selected_input = tool._selected_source_script_input(
        live_selected_source,
        display_name="live_selected",
        source_by_name={
            "live_base": live_base,
            "live_selected": live_selected_source,
        },
    )
    assert live_selected_input is not None
    assert live_selected_input.node_uid is None
    live_selected_spec = live_selected_input.parsed_provenance_spec()
    assert live_selected_spec is not None
    [live_dependency] = provenance.script_input_dependency_refs(live_selected_spec)
    assert live_dependency.node_uid == "live-node"
    live_selected_graph = _replay_graph.compile_replay_graph(
        live_selected_spec,
        live_input_resolver=lambda script_input: (
            (base_data, script_input) if script_input.node_uid == "live-node" else None
        ),
    )
    xr.testing.assert_identical(
        _replay_graph.execute_replay_graph(live_selected_graph),
        base_data.qsel(eV=0.0),
    )

    cycle_a = selected_source.model_copy(
        update={"name": "cycle_a", "selection_source": "cycle_b"}
    )
    cycle_b = selected_source.model_copy(
        update={"name": "cycle_b", "selection_source": "cycle_a"}
    )
    assert (
        tool._selected_source_script_input(
            cycle_a,
            display_name="cycle",
            source_by_name={"cycle_a": cycle_a, "cycle_b": cycle_b},
        )
        is None
    )

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
            FigureSourceState(name="invalid"),
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
    builtin_name = xr.DataArray(np.arange(2.0), dims=("x",), name="list")
    assert figurecomposer_sources._source_name(builtin_name) == "list_2"
    assert figurecomposer_sources._source_alias_error("erlab") is not None
    assert figurecomposer_sources._source_alias_error("list") is not None
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
    assert figurecomposer_operation_metadata.declared_operation_source_names(
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
    ordinary_source = FigureSourceState(name="ordinary")
    ordinary_payload = ordinary_source.model_dump(mode="json")
    assert ordinary_payload["label"] == "ordinary"
    assert "isel" not in ordinary_payload
    assert "qsel" not in ordinary_payload
    assert "mean_dims" not in ordinary_payload
    assert "selection_source" not in ordinary_payload

    current_saved_recipe = FigureRecipeState.model_validate(
        {
            "sources": (
                {
                    "name": "legacy",
                    "isel": {"x": {"kind": "slice", "start": 1}},
                    "selection_source": "base",
                },
            ),
            "primary_source": "legacy",
        }
    )
    [legacy_source] = current_saved_recipe.sources
    assert legacy_source.label == "legacy"
    assert legacy_source.isel == {"x": slice(1, None)}
    [legacy_payload] = current_saved_recipe.model_dump(mode="json")["sources"]
    assert legacy_payload["label"] == "legacy"
    assert legacy_payload["isel"] == {
        "x": {"__erlab_figure_composer_slice__": [1, None, None]}
    }
    assert legacy_payload["selection_source"] == "base"

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
    nonuniform_internal = erlab.utils.array._make_dims_uniform(nonuniform_public)
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
    internal = erlab.utils.array._make_dims_uniform(public)
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
    shape_item = tool.source_panel.source_list.topLevelItem(0)
    assert shape_item is not None
    shape_label = tool.source_panel.source_list.itemWidget(shape_item, 1)
    assert isinstance(shape_label, QtWidgets.QLabel)


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

    assert (
        tool.source_panel.source_splitter.orientation()
        == QtCore.Qt.Orientation.Horizontal
    )
    assert not tool.source_panel.source_splitter.childrenCollapsible()
    assert (
        tool.source_panel.source_inspector.parentWidget()
        is tool.source_panel.source_detail_content
    )
    assert (
        tool.source_panel.source_detail_scroll.widget()
        is tool.source_panel.source_detail_content
    )
    assert tool.editor_tabs.indexOf(tool.source_panel) == 0
    assert not hasattr(tool, "step_detail_splitter")
    assert tool.source_panel.source_inspector.source_name() == "map"
    assert (
        tool.source_panel.source_inspector.property("figureComposerSourceAlias")
        == "map"
    )
    assert tool.source_panel.source_inspector.property("figureComposerSourceDims") == (
        "eV",
        "kx",
    )
    assert (
        tool.source_panel.source_inspector.property("figureComposerSourceDtype")
        == "float64"
    )
    assert tool.source_panel.source_inspector.subtitle_label.isVisibleTo(
        tool.source_panel.source_inspector
    )
    assert not tool.source_panel.source_inspector.details_button.isChecked()
    assert not tool.source_panel.source_inspector.details_label.isVisibleTo(
        tool.source_panel.source_inspector
    )
    assert tool.source_panel.source_inspector.details_html() == ""
    assert format_calls == []
    assert (
        tool.source_panel.source_detail_content.property(
            "figureComposerSourceEditorMode"
        )
        == "single"
    )
    assert (
        tool.source_panel.source_detail_content.property(
            "figureComposerSourceUsageCount"
        )
        == 1
    )
    assert tool.source_panel.source_alias_edit.text() == "map"
    assert (
        tool.source_panel.source_list.selectionMode()
        == QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
    )
    assert (
        tool.source_panel.source_list.selectionBehavior()
        == QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
    )
    first_item = tool.source_panel.source_list.topLevelItem(0)
    assert first_item is not None
    assert first_item.flags() & QtCore.Qt.ItemFlag.ItemIsSelectable
    assert tool.source_panel.source_list.selectedItems() == [first_item]
    shape_label = tool.source_panel.source_list.itemWidget(first_item, 1)
    assert isinstance(shape_label, QtWidgets.QLabel)
    assert shape_label.testAttribute(
        QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents
    )

    tool.source_panel.source_inspector.details_button.setChecked(True)
    assert tool.source_panel.source_inspector.details_label.isVisibleTo(
        tool.source_panel.source_inspector
    )
    assert (
        tool.source_panel.source_inspector.property(
            "figureComposerSourceDetailsExpanded"
        )
        is True
    )
    assert (
        tool.source_panel.source_inspector.details_html() == "<p>formatted details</p>"
    )
    assert format_calls == [(("eV", "kx"), {"show_size": True, "show_summary": False})]
    tool.source_panel.source_inspector.details_button.setChecked(False)
    tool.source_panel.source_inspector.details_button.setChecked(True)
    assert len(format_calls) == 1

    second_item = tool.source_panel.source_list.topLevelItem(1)
    assert second_item is not None
    tool.source_panel.source_list.clearSelection()
    tool.source_panel.source_list.setCurrentItem(second_item)
    assert tool.source_panel.source_list.selectedItems() == [second_item]
    assert tool.source_panel.source_inspector.source_name() == "profile"
    assert (
        tool.source_panel.source_inspector.property("figureComposerSourceAlias")
        == "profile"
    )
    assert tool.source_panel.source_inspector.property("figureComposerSourceDims") == (
        "delay",
    )
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

    tool.source_panel.source_inspector.details_button.setChecked(True)
    html = tool.source_panel.source_inspector.details_html()

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

    tool.source_panel.source_inspector.details_button.setChecked(True)
    assert (
        tool.source_panel.source_inspector.details_html() == "<p>fallback details</p>"
    )
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
    tool.editor_tabs.setCurrentWidget(tool.source_panel)
    with qtbot.waitExposed(tool):
        tool.show()

    second_item = tool.source_panel.source_list.topLevelItem(1)
    assert second_item is not None
    index = tool.source_panel.source_list.indexFromItem(second_item, 1)
    rect = tool.source_panel.source_list.visualRect(index)
    assert rect.isValid()

    qtbot.mouseClick(
        tool.source_panel.source_list.viewport(),
        QtCore.Qt.MouseButton.LeftButton,
        pos=rect.center(),
    )

    assert tool.source_panel.source_list.currentItem() is second_item
    assert tool.source_panel.source_list.selectedItems() == [second_item]
    assert tool.source_panel.source_inspector.source_name() == "profile"


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
    internal = erlab.utils.array._make_dims_uniform(public)
    tool = FigureComposerTool.from_sources(
        {"data": internal},
        sources=(FigureSourceState(name="data", label="map"),),
        operations=(FigureOperationState.plot_slices(label="maps", sources=("data",)),),
        primary_source="data",
    )
    qtbot.addWidget(tool)

    dims = tool.source_panel.source_inspector.property("figureComposerSourceDims")
    assert dims == ("alpha", "eV", "sample_temp")
    tool.source_panel.source_inspector.details_button.setChecked(True)
    assert "sample_temp_idx" not in tool.source_panel.source_inspector.details_html()
    assert "sample_temp" in tool.source_panel.source_inspector.details_html()


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

    assert tool.source_panel.source_inspector.source_name() == "first"
    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(1)
    )
    assert tool.source_panel.source_inspector.source_name() == "first"
    assert tool.source_panel.source_status_label.isHidden()
    assert tool.step_source_status_label.isHidden()
    first_item = tool.source_panel.source_list.topLevelItem(0)
    assert first_item is not None
    tool.source_panel.source_list.setCurrentItem(first_item)
    assert tool.source_panel.source_inspector.source_name() == "first"
    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )
    assert tool.source_panel.source_inspector.source_name() == "first"


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

    y_item = tool.source_panel.source_list.topLevelItem(1)
    assert y_item is not None
    tool.source_panel.source_list.setCurrentItem(y_item)

    assert render_calls == []
    assert write_calls == []
    assert tool.source_panel.source_inspector.source_name() == "y"
    assert (
        tool.source_panel.source_inspector.property("figureComposerSourceAlias") == "y"
    )


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

    inspector._ensure_details_html()
    assert inspector.details_html() == ""


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

    tool._document.source_data.clear()
    tool.source_panel.set_selected_names((), current_name=None)
    assert tool._default_source_inspector_target() == "saved"
    tool._refresh_source_detail_panel()
    assert tool.source_panel.source_inspector.source_name() is None
    assert (
        tool.source_panel.source_detail_content.property(
            "figureComposerSourceEditorMode"
        )
        == "empty"
    )

    tool.source_panel.set_selected_names((), current_name=None)
    assert tool.source_panel.source_list.currentItem() is None


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
    tool.operation_panel.select_section("sources")

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


def test_figure_composer_provenance_replays_transitive_selected_source_chain(
    qtbot,
    tmp_path: Path,
) -> None:
    base = xr.DataArray(
        np.arange(24.0).reshape(2, 3, 4),
        dims=("u", "v", "w"),
        coords={"u": [0.0, 1.0], "v": [0.0, 1.0, 2.0], "w": np.arange(4)},
        name="sample_map",
    )
    selected_u = base.qsel(u=1.0)
    selected_v = selected_u.qsel(v=2.0)
    path = tmp_path / "sample_map.nc"
    base.to_netcdf(path)
    source_spec = _file_load_provenance(path).model_dump(mode="json")
    tool = FigureComposerTool.from_sources(
        {"base": base, "selected_u": selected_u, "selected_v": selected_v},
        sources=(
            FigureSourceState(
                name="base", node_uid="base-node", provenance_spec=source_spec
            ),
            FigureSourceState(
                name="selected_u",
                selection_source="base",
                qsel={"u": 1.0},
                provenance_spec=source_spec,
            ),
            FigureSourceState(
                name="selected_v",
                selection_source="selected_u",
                qsel={"v": 2.0},
                provenance_spec=source_spec,
            ),
        ),
        operations=(
            FigureOperationState.custom(
                label="reserve source name",
                code="sample_map = -1",
                trusted=True,
            ),
            FigureOperationState.line(label="profile", source="selected_v"),
        ),
        primary_source="base",
    )
    qtbot.addWidget(tool)

    script_inputs, skipped_names, source_name_map = tool._display_code_source_plan()
    assert skipped_names == {"selected_v"}
    assert source_name_map == {"selected_v": "sample_map_2"}
    assert len(script_inputs) == 1
    assert script_inputs[0].name == "sample_map_2"
    assert script_inputs[0].label == "sample_map_2"
    assert script_inputs[0].node_uid is None
    selected_spec = script_inputs[0].parsed_provenance_spec()
    assert selected_spec is not None
    assert len(selected_spec.replay_stages) == 2
    assert all(len(stage.operations) == 1 for stage in selected_spec.replay_stages)
    [dependency] = provenance.script_input_dependency_refs(selected_spec)
    assert dependency.node_uid == "base-node"

    resolved_names: list[str] = []

    def resolve_live(
        script_input: provenance.ScriptInput,
    ) -> tuple[xr.DataArray, provenance.ScriptInput] | None:
        if script_input.node_uid != "base-node":
            return None
        resolved_names.append(script_input.name)
        return base, script_input

    selected_graph = _replay_graph.compile_replay_graph(
        selected_spec,
        live_input_resolver=resolve_live,
    )
    live_selected = _replay_graph.execute_replay_graph(selected_graph)
    xr.testing.assert_identical(live_selected, selected_v)
    assert resolved_names == ["base"]

    figure_spec = tool.current_provenance_spec()
    assert figure_spec is not None
    code = figure_spec.display_code()
    assert code is not None
    namespace = _exec_generated_code(code, {})

    assert namespace["sample_map"] == -1
    xr.testing.assert_identical(namespace["sample_map_2"], selected_v)
    lines = namespace["fig"].axes[0].lines
    assert len(lines) == 1
    np.testing.assert_allclose(lines[0].get_ydata(), selected_v.values)


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
    assert tuple(script_input.name for script_input in spec.script_inputs) == ("live",)
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
    source_tool.operation_panel.copy_button.click()
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


def test_figure_composer_copy_paste_steps_carries_selection_dependencies(
    qtbot, monkeypatch
) -> None:
    base = xr.DataArray(
        np.arange(8.0).reshape(2, 2, 2),
        dims=("hv", "x", "y"),
        coords={"hv": [40.0, 50.0], "x": [0.0, 1.0], "y": [0.0, 1.0]},
        name="base",
    )
    selected = base.qsel(hv=40.0)
    source_tool = FigureComposerTool.from_sources(
        {"base": base, "selected": selected},
        sources=(
            FigureSourceState(name="base"),
            FigureSourceState(
                name="selected",
                qsel={"hv": 40.0},
                selection_source="base",
            ),
        ),
        operations=(FigureOperationState.plot_array(label="plot", source="selected"),),
        primary_source="base",
    )
    source_tool._document.source_selection_base_data["selected"] = base

    existing = xr.DataArray(np.zeros((2, 2)), dims=("x", "y"), name="existing")
    destination = FigureComposerTool.from_sources(
        {"base": existing, "selected": existing},
        sources=(FigureSourceState(name="base"), FigureSourceState(name="selected")),
        operations=(),
        primary_source="base",
    )
    qtbot.addWidget(source_tool)
    qtbot.addWidget(destination)
    _clear_clipboard()

    _select_operation_rows(source_tool, (0,))
    source_tool._copy_selected_operations()
    destination._paste_operations_from_clipboard()

    source_by_name = destination._document.source_by_name()
    assert source_by_name["selected_copy"].selection_source == "base_copy"
    xr.testing.assert_identical(destination.source_data()["base_copy"], base)
    xr.testing.assert_identical(destination.source_data()["selected_copy"], selected)
    xr.testing.assert_identical(
        destination._document.source_selection_base_data["selected_copy"], base
    )
    assert destination.tool_status.operations[-1].sources == ("selected_copy",)

    captured: list[xr.DataArray] = []
    monkeypatch.setattr(
        eplt,
        "plot_array",
        lambda data, **_kwargs: captured.append(data),
    )
    namespace = _exec_generated_code(
        destination.generated_code(),
        {"base_copy": destination.source_data()["base_copy"]},
    )
    assert isinstance(namespace["fig"], Figure)
    xr.testing.assert_identical(captured[-1], selected)


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
    tool.operation_panel.cut_button.click()
    tool.operation_panel.paste_button.click()

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
    source_tool.operation_panel.cut_button.click()
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
    source_tool.operation_panel.copy_button.click()
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
    source_tool.operation_panel.copy_button.click()
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
    assert list(tool._prev_source_data_states[-1][0]) == ["data"]
    assert tool._prev_source_data_states[-1][1] == {}
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
    xr.testing.assert_identical(
        tool._prev_source_data_states[-1][0]["data"], replacement
    )

    tool._prev_states.clear()
    tool._prev_source_data_states.clear()
    tool._replace_last_state()
    assert len(tool._prev_states) == 1
    xr.testing.assert_identical(
        tool._prev_source_data_states[-1][0]["data"], replacement
    )


def test_figure_composer_history_restores_source_selection_backing(qtbot) -> None:
    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("eV", "x"),
        coords={"eV": [-1.0, 0.0], "x": [0.0, 1.0, 2.0]},
        name="data",
    )
    tool = FigureComposerTool.from_sources(
        {"data": data},
        sources=(FigureSourceState(name="data"),),
        operations=(FigureOperationState.plot_array(label="plot", source="data"),),
        primary_source="data",
    )
    qtbot.addWidget(tool)
    tool.source_panel.set_selected_names({"data"}, current_name="data")

    tool.source_panel.selection_dimension_requested.emit(
        tool.source_panel.selected_names(), "eV", "qsel", "0.0", ""
    )
    assert tool.source_data()["data"].dims == ("x",)
    xr.testing.assert_identical(tool._document.source_selection_base_data["data"], data)

    tool.undo()
    assert tool.source_states()[0].qsel == {}
    assert "data" not in tool._document.source_selection_base_data
    xr.testing.assert_identical(tool.source_data()["data"], data)

    tool.redo()
    assert tool.source_states()[0].qsel == {"eV": 0.0}
    xr.testing.assert_identical(tool._document.source_selection_base_data["data"], data)

    tool.source_panel.selection_dimension_requested.emit(
        tool.source_panel.selected_names(), "eV", "keep", "", ""
    )
    assert tool.source_states()[0].qsel == {}
    xr.testing.assert_identical(tool.source_data()["data"], data)


def test_figure_composer_copy_paste_source_and_insert_fallbacks(qtbot) -> None:
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
    assert tool._document.operation_source_names(operation) == ("data",)

    clipboard = _clear_clipboard()
    clipboard.setText(
        figurecomposer_tool_module._step_clipboard_payload_text(
            (_custom_order_step("b"),), ()
        )
    )
    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )
    tool._paste_operations_from_clipboard()
    assert [operation.label for operation in tool.tool_status.operations] == ["a", "b"]


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
    tool.operation_panel.select_section("selection")
    selection_page = tool.operation_panel.editor_stack.currentWidget()
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


def test_figure_composer_source_structure_defensive_paths(qtbot) -> None:
    first = xr.DataArray(np.arange(3.0), dims=("x",), name="first")
    second = xr.DataArray(np.arange(3.0) + 10.0, dims=("x",), name="second")
    tool = FigureComposerTool.from_sources(
        {"first": first, "second": second},
        sources=(FigureSourceState(name="first"), FigureSourceState(name="second")),
        operations=(),
        primary_source="first",
    )
    qtbot.addWidget(tool)

    tool.source_panel.set_selected_names(set(), current_name=None)
    assert tool.source_panel.selected_names() == ()

    refreshed: list[str] = []

    def refresh_source(name: str) -> bool:
        if name == "first":
            tool._set_source_panel_status("refresh rejected")
            return False
        refreshed.append(name)
        return True

    tool._set_source_refresh_callbacks(
        can_refresh_source=lambda name: name in {"first", "second"},
        refresh_source=refresh_source,
    )
    tool._refresh_source_names(("first", "second"))
    assert refreshed == ["second"]
    assert not tool.source_panel.source_status_label.isHidden()

    legacy_source = FigureSourceState(name="legacy").model_copy(
        update={"selection_source": "legacy"}
    )
    renamed = tool._document.source_with_renamed_references(
        legacy_source, {"legacy": "renamed"}
    )
    assert renamed.name == "renamed"
    assert renamed.selection_source is None

    tool._source_list_reordered(("second", "first"), (), "second")
    assert tuple(source.name for source in tool.source_states()) == ("second", "first")
    assert tool.source_panel.selected_names() == ("second",)

    first_source = tool._document.source_by_name()["first"].model_copy(
        update={"selection_source": "second"}
    )
    second_source = tool._document.source_by_name()["second"].model_copy(
        update={"selection_source": "first"}
    )
    tool.tool_status = tool.tool_status.model_copy(
        update={"sources": (first_source, second_source)}
    )
    assert tool._document.source_dependency_names(("first",)) == ("second", "first")


def test_figure_composer_legacy_source_selection_defensive_paths(qtbot) -> None:
    base = xr.DataArray(np.arange(3.0), dims=("x",), name="base")
    selected = FigureSourceState(
        name="selected",
        selection_source="base",
        qsel={"missing": 0.0},
    )
    tool = FigureComposerTool.from_sources(
        {"base": base},
        sources=(FigureSourceState(name="base"), selected),
        operations=(),
        primary_source="base",
    )
    qtbot.addWidget(tool)

    source_list = list(tool.source_states())
    source_by_name = {source.name: source for source in source_list}
    source_data = dict(tool.source_data())
    selection_base_data = dict(tool._document.source_selection_base_data)
    alias = tool._source_alias_for_legacy_selection(
        FigureDataSelectionState(source="base", qsel={"missing": 0.0}),
        source_list=source_list,
        source_by_name=source_by_name,
        reserved=set(source_by_name),
        source_data=source_data,
        selection_base_data=selection_base_data,
    )
    assert alias == "selected"
    assert "selected" not in source_data
    xr.testing.assert_identical(selection_base_data["selected"], base)

    assert tool._document.source_lineage_names("selected", {"selected": selected}) == (
        "selected",
    )
    cycle_a = selected.model_copy(
        update={"name": "cycle_a", "selection_source": "cycle_b"}
    )
    cycle_b = selected.model_copy(
        update={"name": "cycle_b", "selection_source": "cycle_a"}
    )
    with pytest.raises(ValueError, match="cycle"):
        tool._document.source_lineage_names(
            "cycle_a", {"cycle_a": cycle_a, "cycle_b": cycle_b}
        )
    propagated = tool._document.source_states_with_propagated_link_metadata(
        {"selected": selected}, ("missing",)
    )
    assert propagated["selected"] is selected
    with pytest.raises(ValueError, match="unavailable"):
        tool._document.recompute_source_dependents(
            {}, {}, ("base",), source_by_name=source_by_name
        )

    linked = FigureSourceState(name="linked", selection_source="base")
    recomputed_data, recomputed_bases = tool._document.recompute_source_dependents(
        {"base": base},
        {"linked": base},
        ("base",),
        source_by_name={"base": source_by_name["base"], "linked": linked},
    )
    xr.testing.assert_identical(recomputed_data["linked"], base)
    assert "linked" not in recomputed_bases

    empty_operation = FigureOperationState.plot_slices(label="empty", sources=())
    converted_empty = tool._plot_slices_operation_with_shared_legacy_selection(
        empty_operation
    )
    assert converted_empty is not None
    assert converted_empty.map_selections == ()

    invalid_operation = FigureOperationState.plot_slices(
        label="invalid",
        sources=("base",),
        map_selections=(
            FigureDataSelectionState(source="base", qsel={"missing": 0.0}),
        ),
    )
    assert (
        tool._plot_slices_operation_with_shared_legacy_selection(invalid_operation)
        is None
    )

    no_slice_dimension = FigureOperationState.plot_slices(
        label="legacy", sources=("base",)
    ).model_copy(update={"slice_dim": None})
    converted_slice = FigureComposerTool._plot_slices_operation_with_legacy_qsel(
        no_slice_dimension,
        {"x": 1.0, "x_width": 0.5},
    )
    assert converted_slice.slice_dim == "x"
    assert converted_slice.slice_values == (1.0,)
    assert converted_slice.slice_width == 0.5

    no_selection_operation = FigureOperationState.plot_slices(
        label="no selection",
        sources=("base",),
        map_selections=(FigureDataSelectionState(source="base"),),
    )
    converted_sources = tool._plot_slices_operation_with_legacy_source_aliases(
        no_selection_operation,
        source_list=[source_by_name["base"]],
        source_by_name={"base": source_by_name["base"]},
        reserved={"base"},
        source_data=dict(tool.source_data()),
        selection_base_data=dict(tool._document.source_selection_base_data),
    )
    assert converted_sources.sources == ("base",)
    assert converted_sources.map_selections == ()
    assert (
        FigureComposerTool._legacy_selection_fallback_source(
            FigureOperationState.custom(label="custom", code="pass", trusted=True),
            "base",
        )
        is None
    )


def test_figure_composer_legacy_selection_reports_replaced_backing_data(qtbot) -> None:
    base = xr.DataArray(np.arange(3.0), dims=("x",), name="base")
    selected = FigureSourceState(
        name="selected",
        selection_source="base",
        qsel={"missing": 0.0},
    )
    tool = FigureComposerTool.from_sources(
        {"base": base},
        sources=(FigureSourceState(name="base"), selected),
        operations=(),
        primary_source="base",
    )
    qtbot.addWidget(tool)
    stale_base = base + 10
    tool._document.source_selection_base_data["selected"] = stale_base
    tool._document.append_operation(
        FigureOperationState.plot_array(
            label="legacy",
            source="base",
            map_selections=(
                FigureDataSelectionState(source="base", qsel={"missing": 0.0}),
            ),
        )
    )

    assert tool._normalize_operation_source_selections()
    xr.testing.assert_identical(
        tool._document.source_selection_base_data["selected"], base
    )


def test_figure_composer_add_and_replace_tolerate_broken_link_cycle(qtbot) -> None:
    left_data = xr.DataArray(np.arange(3.0), dims=("x",), name="left")
    right_data = left_data.rename("right")
    left = FigureSourceState(name="left", node_uid="shared").model_copy(
        update={"selection_source": "right"}
    )
    right = FigureSourceState(name="right", node_uid="shared").model_copy(
        update={"selection_source": "left"}
    )
    tool = FigureComposerTool.from_sources(
        {"left": left_data, "right": right_data},
        sources=(left, right),
        operations=(),
        primary_source="left",
    )
    qtbot.addWidget(tool)

    incoming = left_data + 10.0
    result = tool.add_sources(
        (FigureSourceState(name="incoming", node_uid="shared"),),
        {"incoming": incoming},
    )
    assert result.added == (("incoming", "incoming"),)
    xr.testing.assert_identical(tool.source_data()["incoming"], incoming)

    replacement = left_data + 20.0
    assert tool.replace_source(
        "left",
        FigureSourceState(name="replacement", node_uid="shared"),
        replacement,
    )
    xr.testing.assert_identical(tool.source_data()["left"], replacement)

    unavailable = tool.add_sources((FigureSourceState(name="missing"),), {})
    assert not unavailable
    assert unavailable.skipped[0][0] == "missing"
    assert not tool.source_panel.source_status_label.isHidden()


def test_figure_composer_restore_persisted_selection_failure_paths(qtbot) -> None:
    data = xr.DataArray(np.arange(3.0), dims=("x",), name="saved")

    primary_tool = FigureComposerTool.from_sources(
        {"primary": data},
        sources=(FigureSourceState(name="primary"),),
        operations=(),
        primary_source="primary",
    )
    qtbot.addWidget(primary_tool)
    invalid_primary = primary_tool.source_states()[0].model_copy(
        update={"qsel": {"missing": 0.0}}
    )
    primary_tool.tool_status = primary_tool.tool_status.model_copy(
        update={"sources": (invalid_primary,)}
    )
    primary_tool.set_source_data({})
    primary_tool._restore_persistence_data_items(
        {erlab.interactive.utils._SAVED_TOOL_DATA_NAME: data},
        xr.Dataset(attrs={"tool_data_name": "restored"}),
    )
    assert primary_tool.source_data()["primary"].name == "restored"
    assert "primary" not in primary_tool._document.source_selection_base_data

    invalid_child = FigureSourceState(
        name="child",
        selection_source="root",
        qsel={"missing": 0.0},
    )
    descendant_tool = FigureComposerTool.from_sources(
        {"root": data},
        sources=(FigureSourceState(name="root"), invalid_child),
        operations=(),
        primary_source="root",
    )
    qtbot.addWidget(descendant_tool)
    descendant_tool.set_source_data({})
    descendant_tool._restore_persistence_data_items(
        {erlab.interactive.utils._SAVED_TOOL_DATA_NAME: data}, xr.Dataset()
    )
    assert "root" in descendant_tool.source_data()
    assert "child" not in descendant_tool.source_data()

    orphan = FigureSourceState(
        name="orphan",
        selection_source="missing",
        qsel={"missing": 0.0},
    )
    pending_tool = FigureComposerTool.from_sources(
        {"stable": data},
        sources=(FigureSourceState(name="stable"), orphan),
        operations=(),
        primary_source="stable",
    )
    qtbot.addWidget(pending_tool)
    pending_tool._restore_persistence_data_items({"orphan": data}, xr.Dataset())
    assert "stable" in pending_tool.source_data()
    assert "orphan" not in pending_tool.source_data()


def test_figure_composer_selected_source_codegen_fallback_uses_base_input(
    qtbot,
) -> None:
    base_data = xr.DataArray(np.arange(3.0), dims=("x",), name="base")
    selected_data = base_data.isel(x=0)
    base_source = FigureSourceState(name="base")
    selected_source = FigureSourceState(
        name="selected",
        selection_source="base",
        isel={"x": 0},
    )
    tool = FigureComposerTool.from_sources(
        {"base": base_data, "selected": selected_data},
        sources=(base_source, selected_source),
        operations=(FigureOperationState.plot_array(label="plot", source="selected"),),
        primary_source="base",
    )
    qtbot.addWidget(tool)

    malformed_provenance = {"kind": "not-a-provenance-kind"}
    tool.tool_status = tool.tool_status.model_copy(
        update={
            "sources": (
                base_source.model_copy(
                    update={"provenance_spec": malformed_provenance}
                ),
                selected_source.model_copy(
                    update={"provenance_spec": malformed_provenance}
                ),
            )
        }
    )
    script_inputs, skipped_names, source_name_map = tool._display_code_source_plan()
    assert tuple(script_input.name for script_input in script_inputs) == ("base",)
    assert skipped_names == frozenset()
    assert source_name_map == {}

    cleanup_tool = FigureComposerTool.from_sources(
        {"base": base_data, "selected": selected_data},
        sources=(base_source, selected_source),
        operations=(),
        primary_source="base",
    )
    qtbot.addWidget(cleanup_tool)
    cleanup_tool._document.source_selection_base_data["selected"] = base_data
    cleanup_tool.set_missing_sources({"selected"})
    assert "selected" not in cleanup_tool.source_data()
    assert "selected" not in cleanup_tool._document.source_selection_base_data


def test_figure_composer_source_selection_control_guard_paths(qtbot) -> None:
    plain = xr.DataArray(np.arange(2.0), dims=("x",), name="plain")
    other = xr.DataArray(np.arange(2.0), dims=("y",), name="other")
    empty = xr.DataArray(
        np.empty(0), dims=("empty",), coords={"empty": np.empty(0)}, name="empty"
    )
    with_coordinates = xr.DataArray(
        np.arange(2.0),
        dims=("x",),
        coords={"x": [0.0, 1.0]},
        name="with_coordinates",
    )
    source_names = ("plain", "other", "empty", "with_coordinates", "unavailable")
    tool = FigureComposerTool.from_sources(
        {
            "plain": plain,
            "other": other,
            "empty": empty,
            "with_coordinates": with_coordinates,
        },
        sources=tuple(FigureSourceState(name=name) for name in source_names),
        operations=(),
        primary_source="plain",
    )
    qtbot.addWidget(tool)

    assert tool._source_selection_dimension_tooltip("x", source_names)
    assert tool._common_source_selection_dims(("unavailable", "plain", "other")) == ()
    tool.source_panel.set_selected_names(
        {"with_coordinates"}, current_name="with_coordinates"
    )
    assert tool._default_source_inspector_target() == "with_coordinates"

    tool._refresh_source_detail_panel()
    tool._refresh_source_selection_editor()
    mode_combo = tool.source_panel.source_selection_controls.findChild(
        QtWidgets.QComboBox, "figureComposerSourceSelectionModeCombo0"
    )
    value_edit = tool.source_panel.source_selection_controls.findChild(
        QtWidgets.QLineEdit, "figureComposerSourceSelectionValueEdit0"
    )
    width_edit = tool.source_panel.source_selection_controls.findChild(
        QtWidgets.QLineEdit, "figureComposerSourceSelectionWidthEdit0"
    )
    assert mode_combo is not None
    assert value_edit is not None
    assert width_edit is not None
    before = tool.source_states()
    mode_combo.addItem("unsupported", object())
    unsupported_index = mode_combo.count() - 1
    mode_combo.setCurrentIndex(unsupported_index)
    mode_combo.activated.emit(unsupported_index)
    assert tool.source_states() == before

    qsel_index = mode_combo.findData("qsel")
    assert qsel_index >= 0
    mode_combo.setCurrentIndex(qsel_index)
    mode_combo.activated.emit(qsel_index)
    _editor_controls.LineEditControlAdapter(value_edit).set_mixed(True)
    value_edit.editingFinished.emit()
    assert tool.source_states() == before

    _editor_controls.LineEditControlAdapter(value_edit).set_mixed(False)
    value_edit.setText("1.0")
    _editor_controls.LineEditControlAdapter(width_edit).set_mixed(True)
    value_edit.editingFinished.emit()
    assert tool.source_states() == before

    _editor_controls.LineEditControlAdapter(width_edit).set_mixed(False)
    value_edit.setText("[")
    width_edit.setText("0.5")
    value_edit.editingFinished.emit()
    assert not tool.source_panel.source_validation_label.isHidden()

    value_edit.setText("1.0")
    width_edit.editingFinished.emit()
    selected_source = tool._document.source_by_name()["with_coordinates"]
    assert selected_source.qsel
    assert "with_coordinates" in tool._document.source_selection_base_data

    detached_selected = FigureSourceState(
        name="selected", selection_source="missing_base"
    )
    detached_tool = FigureComposerTool.from_sources(
        {"selected": with_coordinates},
        sources=(FigureSourceState(name="missing_base"), detached_selected),
        operations=(),
        primary_source="selected",
    )
    qtbot.addWidget(detached_tool)
    detached_tool.source_panel.set_selected_names({"selected"}, current_name="selected")
    detached_before = detached_tool.source_states()
    detached_tool.source_panel.selection_dimension_requested.emit(
        detached_tool.source_panel.selected_names(), "x", "qsel", "1.0", ""
    )
    assert detached_tool.source_states() == detached_before

    tool.source_panel.set_selected_names({"unavailable"}, current_name="unavailable")
    tool._document.recipe = tool.tool_status.model_copy(
        update={
            "sources": tuple(
                source
                for source in tool.source_states()
                if source.name != "unavailable"
            )
        }
    )
    tool._refresh_source_detail_panel()
    assert not tool.source_panel.source_alias_edit.isEnabled()


def test_figure_composer_restore_persisted_source_pending_paths(qtbot) -> None:
    data = xr.DataArray(np.arange(3.0), dims=("x",), name="saved")
    blocked_tool = FigureComposerTool.from_sources(
        {"stable": data, "parent": data},
        sources=(
            FigureSourceState(name="stable"),
            FigureSourceState(name="parent", selection_source="missing_parent"),
            FigureSourceState(name="child", selection_source="parent"),
        ),
        operations=(),
        primary_source="stable",
    )
    qtbot.addWidget(blocked_tool)
    blocked_tool._restore_persistence_data_items({"child": data}, xr.Dataset())
    assert "parent" in blocked_tool.source_data()
    assert "child" not in blocked_tool.source_data()

    fallback_tool = FigureComposerTool.from_sources(
        {"stable": data},
        sources=(
            FigureSourceState(name="stable"),
            FigureSourceState(name="orphan", selection_source="missing_parent"),
        ),
        operations=(),
        primary_source="stable",
    )
    qtbot.addWidget(fallback_tool)
    fallback_tool._restore_persistence_data_items({"orphan": data}, xr.Dataset())
    xr.testing.assert_identical(fallback_tool.source_data()["orphan"], data)


def test_figure_composer_selected_source_script_input_skips_nonreplayable_input(
    qtbot, monkeypatch
) -> None:
    data = xr.DataArray(
        np.arange(2.0), dims=("x",), coords={"x": [0.0, 1.0]}, name="base"
    )
    provenance_spec = provenance.public_data().model_dump(mode="json")
    base = FigureSourceState(name="base", provenance_spec=provenance_spec)
    selected = FigureSourceState(
        name="selected",
        selection_source="base",
        qsel={"x": 1.0},
        provenance_spec=provenance_spec,
    )
    tool = FigureComposerTool.from_sources(
        {"base": data, "selected": data.qsel(x=1.0)},
        sources=(base, selected),
        operations=(FigureOperationState.line(label="line", source="selected"),),
        primary_source="base",
    )
    qtbot.addWidget(tool)

    monkeypatch.setattr(provenance, "to_replay_provenance_spec", lambda _spec: None)
    assert (
        tool._selected_source_script_input(
            selected,
            display_name="selected_display",
            source_by_name={"base": base, "selected": selected},
        )
        is None
    )
