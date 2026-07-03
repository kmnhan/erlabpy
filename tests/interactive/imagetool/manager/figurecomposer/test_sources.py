# ruff: noqa: F403, F405

from ._common import *


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
    assert first_item.font(0).bold()
    assert first_item.text(1) == "data_0"
    second_item = tool.source_list.topLevelItem(1)
    assert second_item is not None
    assert second_item.data(0, QtCore.Qt.ItemDataRole.UserRole + 1) is False
    assert second_item.text(1) == "data_1"

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
    shape_widget = tool.source_list.itemWidget(first_item, 2)
    assert isinstance(shape_widget, QtWidgets.QLabel)
    assert shape_widget.text() == "<p>formatted shape</p>"
    assert tool.source_list.toolTip() == ""
    assert first_item.toolTip(0) == ""
    assert first_item.toolTip(1) == ""
    assert first_item.toolTip(2) == ""
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

    buttons = _source_refresh_buttons(tool)
    assert set(buttons) == {"data_0", "profile"}
    assert buttons["data_0"].property("figure_source_name") == "data_0"
    assert buttons["data_0"].accessibleName() == "Refresh Source"
    assert not buttons["data_0"].isEnabled()
    assert (
        buttons["data_0"].toolTip() == "This source is not linked to an open ImageTool"
    )
    assert tool.refresh_sources_button.accessibleName() == "Refresh Sources"
    assert not tool.refresh_sources_button.isEnabled()
    assert tool.refresh_sources_button.toolTip() == (
        "No sources are linked to open ImageTools"
    )
    assert tool._source_refresh_label("data_0") is None
    tool._refresh_source_from_button("data_0")
    tool._refresh_sources_from_button()

    refreshed: list[tuple[str, str | tuple[str, ...]]] = []

    def can_refresh_source(name: str) -> bool:
        return name == "data_0"

    def refresh_source(name: str) -> bool:
        refreshed.append(("one", name))
        return True

    def refresh_sources(names: Sequence[str]) -> int:
        refreshed.append(("many", tuple(names)))
        return len(names)

    def source_label(name: str) -> str | None:
        return "ImageTool 0: image" if name == "data_0" else None

    tool._set_source_refresh_callbacks(
        can_refresh_source=can_refresh_source,
        refresh_source=refresh_source,
        refresh_sources=refresh_sources,
        source_label=source_label,
    )

    buttons = _source_refresh_buttons(tool)
    assert buttons["data_0"].isEnabled()
    assert buttons["data_0"].toolTip() == (
        "Refresh “ImageTool 0: image” from ImageTool 0: image"
    )
    assert not buttons["profile"].isEnabled()
    assert buttons["profile"].toolTip() == (
        "This source is not linked to an open ImageTool"
    )
    assert tool.refresh_sources_button.isEnabled()
    assert tool.refresh_sources_button.toolTip() == (
        "Refresh all sources linked to open ImageTools"
    )

    status_before_refresh = (
        tool.source_status_label.text(),
        tool.source_status_label.isVisible(),
    )
    buttons["data_0"].click()
    assert refreshed[-1] == ("one", "data_0")
    assert (
        tool.source_status_label.text(),
        tool.source_status_label.isVisible(),
    ) == status_before_refresh

    tool.refresh_sources_button.click()
    assert refreshed[-1] == ("many", ("data_0",))
    assert (
        tool.source_status_label.text(),
        tool.source_status_label.isVisible(),
    ) == status_before_refresh

    def refresh_no_sources(names: Sequence[str]) -> int:
        refreshed.append(("none", tuple(names)))
        return 0

    tool._set_source_refresh_callbacks(
        can_refresh_source=can_refresh_source,
        refresh_source=refresh_source,
        refresh_sources=refresh_no_sources,
        source_label=source_label,
    )
    tool._set_source_status_text("diagnostic")
    tool.refresh_sources_button.click()
    assert refreshed[-1] == ("none", ("data_0",))
    assert tool.source_status_label.text() == "diagnostic"

    tool._set_source_refresh_callbacks(
        can_refresh_source=lambda _name: False,
        refresh_source=refresh_source,
        refresh_sources=refresh_sources,
        source_label=source_label,
    )
    refreshed.clear()
    tool._refresh_source_from_button("data_0")
    tool._refresh_sources_from_button()
    assert refreshed == []

    def raise_lookup(name: str) -> bool:
        raise LookupError(name)

    def raise_lookup_label(name: str) -> str | None:
        raise LookupError(name)

    tool._set_source_refresh_callbacks(
        can_refresh_source=raise_lookup,
        refresh_source=refresh_source,
        refresh_sources=refresh_sources,
        source_label=raise_lookup_label,
    )
    assert not tool._source_refresh_available("data_0")
    assert tool._source_refresh_label("data_0") is None

    tool._set_source_refresh_callbacks(
        can_refresh_source=can_refresh_source,
        refresh_source=refresh_source,
        refresh_sources=refresh_sources,
        source_label=lambda _name: "",
    )
    assert tool._source_refresh_label("data_0") is None

    direct_item = QtWidgets.QTreeWidgetItem(["direct", "", "", ""])
    tool.source_list.addTopLevelItem(direct_item)
    direct_button = QtWidgets.QToolButton(tool.source_list)
    direct_button.setObjectName("figureComposerRefreshSourceButton")
    tool.source_list.setItemWidget(direct_item, 3, direct_button)
    assert (
        tool._source_list_row_button(direct_item, "figureComposerRefreshSourceButton")
        is direct_button
    )
    assert (
        tool._source_list_row_button(direct_item, "figureComposerRemoveSourceButton")
        is None
    )
    tool._refresh_source_controls()


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
    tool.operation_list.setCurrentRow(0)

    buttons = _source_remove_buttons(tool)
    assert set(buttons) == {"data_0", "profile", "unused"}
    assert buttons["unused"].property("figure_source_name") == "unused"
    assert buttons["unused"].accessibleName() == "Remove Source"
    assert not buttons["data_0"].isEnabled()
    assert buttons["data_0"].toolTip() == "This source is used by one or more steps"
    assert not buttons["profile"].isEnabled()
    assert buttons["profile"].toolTip() == "This source is used by one or more steps"
    assert buttons["unused"].isEnabled()
    assert buttons["unused"].toolTip() == "Remove “Unused” from this figure"

    before_status = tool.tool_status
    before_source_data = tool.source_data()
    buttons["profile"].click()
    assert tool.tool_status == before_status
    assert set(tool.source_data()) == set(before_source_data)
    tool._remove_source_from_button("profile")
    assert tool.tool_status == before_status
    assert set(tool.source_data()) == set(before_source_data)
    assert not tool.remove_source("profile")
    assert not tool.remove_source("missing")

    buttons["unused"].click()
    assert "unused" not in tool.source_data()
    assert tool.source_status_label.text() == ""
    assert tool.source_status_label.isHidden()


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
    buttons = _source_remove_buttons(tool)
    assert set(buttons) == {"data_0"}
    assert not buttons["data_0"].isEnabled()
    assert buttons["data_0"].toolTip() == "This source is used by one or more steps"

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
    assert set(_source_remove_buttons(tool)) == {"data_0"}


def test_figure_composer_source_display_helpers_keep_alias_secondary() -> None:
    source = FigureSourceState(name="data_0", label="ImageTool 0: sample_map")
    assert (
        figurecomposer_sources._source_display_label(source, "data_0")
        == "ImageTool 0: sample_map"
    )
    assert (
        figurecomposer_sources._source_display_tooltip(source, "data_0")
        == "ImageTool 0: sample_map\nAlias: data_0"
    )

    duplicates = figurecomposer_sources._source_duplicate_labels(
        (
            FigureSourceState(name="data_0", label="ImageTool"),
            FigureSourceState(name="data_1", label="ImageTool"),
        )
    )
    assert duplicates == frozenset(("ImageTool",))
    assert (
        figurecomposer_sources._source_display_label(
            FigureSourceState(name="data_0", label="ImageTool"),
            "data_0",
            disambiguate=True,
        )
        == "ImageTool (data_0)"
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
    tool.operation_list.setCurrentRow(0)

    assert "ImageTool 0: original" in tool.operation_list.item(0).text()
    assert "ImageTool 0: original" in tool.operation_list.item(1).text()
    assert tool.step_section_buttons["sources"].text() == (
        "Sources: ImageTool 0: original"
    )

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
    assert "ImageTool 1: replacement" in tool.operation_list.item(0).text()
    assert "ImageTool 0: original" not in tool.operation_list.item(0).text()
    assert "ImageTool 1: replacement" in tool.operation_list.item(1).text()
    assert "ImageTool 0: original" not in tool.operation_list.item(1).text()
    assert tool.step_section_buttons["sources"].text() == (
        "Sources: ImageTool 1: replacement"
    )

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
    assert tool.source_states()[-1].label == "Orphan"


def test_figure_composer_source_helpers_cover_selection_contract() -> None:
    unnamed = xr.DataArray(np.arange(2.0), dims=("x",), name=None)
    invalid_name = xr.DataArray(np.arange(2.0), dims=("x",), name="bad name")
    assert figurecomposer_sources._source_name(unnamed) == "data"
    assert figurecomposer_sources._source_label(unnamed) == "data"
    assert figurecomposer_sources._source_name(invalid_name) == "data"
    assert figurecomposer_sources._source_display_tooltip(None, "data_0") == (
        "Alias: data_0"
    )
    with pytest.raises(ValueError, match="not a valid variable"):
        figurecomposer_sources._valid_source_variable("bad name")
    assert figurecomposer_plot_slices._source_names(
        FigureOperationState.plot_slices(
            label="mapped",
            sources=(),
            map_selections=(FigureDataSelectionState(source="extra"),),
        )
    ) == ("extra",)

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
    shape_label = tool.source_list.itemWidget(shape_item, 2)
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
        name="map",
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

    assert tool.source_inspector.parentWidget() is tool.step_sources_page
    assert not hasattr(tool, "step_detail_splitter")
    assert tool.source_inspector.source_name() == "map"
    assert tool.source_inspector.property("figureComposerSourceAlias") == "map"
    assert tool.source_inspector.property("figureComposerSourceDims") == ("eV", "kx")
    assert tool.source_inspector.property("figureComposerSourceDtype") == "float64"
    assert tool.source_inspector.property("figureComposerSourceUsedByStep") is True
    assert not tool.source_inspector.details_button.isChecked()
    assert not tool.source_inspector.details_scroll.isVisibleTo(tool.source_inspector)
    assert tool.source_inspector.details_html() == "<p>formatted details</p>"
    assert format_calls[-1] == (
        ("eV", "kx"),
        {"show_size": True, "show_summary": False, "load_values": False},
    )
    assert (
        tool.source_list.selectionMode()
        == QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
    )
    assert (
        tool.source_list.selectionBehavior()
        == QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
    )
    first_item = tool.source_list.topLevelItem(0)
    assert first_item is not None
    assert first_item.flags() & QtCore.Qt.ItemFlag.ItemIsSelectable
    assert tool.source_list.selectedItems() == [first_item]
    shape_label = tool.source_list.itemWidget(first_item, 2)
    assert isinstance(shape_label, QtWidgets.QLabel)
    assert shape_label.testAttribute(
        QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents
    )

    tool.source_inspector.details_button.setChecked(True)
    assert tool.source_inspector.details_scroll.isVisibleTo(tool.source_inspector)
    assert tool.source_inspector.property("figureComposerSourceDetailsExpanded") is True

    second_item = tool.source_list.topLevelItem(1)
    assert second_item is not None
    tool.source_list.setCurrentItem(second_item)
    assert tool.source_list.selectedItems() == [second_item]
    assert tool.source_inspector.source_name() == "profile"
    assert tool.source_inspector.property("figureComposerSourceAlias") == "profile"
    assert tool.source_inspector.property("figureComposerSourceDims") == ("delay",)


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
    with qtbot.waitExposed(tool):
        tool.show()

    second_item = tool.source_list.topLevelItem(1)
    assert second_item is not None
    index = tool.source_list.indexFromItem(second_item, 2)
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
            FigureOperationState.line(label="second line", source="second"),
        ),
        primary_source="first",
    )
    qtbot.addWidget(tool)

    assert tool.source_inspector.source_name() == "first"
    tool.operation_list.setCurrentRow(1)
    assert tool.source_inspector.source_name() == "second"
    first_item = tool.source_list.topLevelItem(0)
    assert first_item is not None
    tool.source_list.setCurrentItem(first_item)
    assert tool.source_inspector.source_name() == "first"
    tool.operation_list.setCurrentRow(0)
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
    assert tool.source_inspector.property("figureComposerSourceUsedByStep") is True
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
    assert "Alias: data" in missing_metadata
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
        source_state=None,
        data=None,
        operation_source_names=(),
    )
    assert inspector.source_name() is None
    assert inspector.property("figureComposerSourceDims") == ()
    assert inspector.property("figureComposerSourceUsedByStep") is False
    assert not inspector.details_button.isEnabled()

    inspector.set_context(
        source_name="data",
        source_state=source,
        data=None,
        operation_source_names=("data",),
    )
    assert inspector.source_name() == "data"
    assert inspector.property("figureComposerSourceUsedByStep") is True
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
    tool._refresh_source_inspector()
    assert tool.source_inspector.source_name() == "saved"

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
    assert tuple(script_input.name for script_input in spec.script_inputs) == (
        "data_0",
    )
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
    assert pasted_operation.map_selections[0].source == "data"
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
    assert pasted_plot.map_selections[0].source == "data_copy"
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
    assert pasted_plot.map_selections[0].source == "data_copy"
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
    assert tool._operation_source_names(operation) == ("data", "extra")

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
    tool.operation_list.setCurrentRow(0)
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
    assert all(label in tooltip for label in source_labels)
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
