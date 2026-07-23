"""Plot Slices state behavior tests."""

from ._plot_slices_common import (
    FigureAxesSelectionState,
    FigureComposerPlotSlicesSelectionError,
    FigureComposerTool,
    FigureDataSelectionState,
    FigureOperationKind,
    FigureOperationState,
    FigureRecipeState,
    FigureSourceState,
    FigureSubplotsState,
    QtCore,
    QtWidgets,
    _figure_composer_image_source,
    _plot_slices_selection_migration_data,
    _plot_source_checks,
    _plot_source_move_buttons,
    _set_unsupported_plot_slices_cursor_state,
    _unsupported_plot_slices_data,
    eplt,
    erlab,
    figurecomposer_adapter,
    np,
    plot_slices_model,
    plot_slices_operation_with_source_styles,
    pytest,
    typing,
    xr,
)


def test_figure_composer_plot_slices_migrates_shared_map_selection_to_slice_state(
    qtbot,
) -> None:
    data = _plot_slices_selection_migration_data()
    operation = FigureOperationState.plot_slices(
        label="plot_slices",
        sources=("first",),
        map_selections=(FigureDataSelectionState(source="first", qsel={"y": 0.0}),),
        axes=FigureAxesSelectionState(),
    )
    tool = FigureComposerTool.from_sources(
        {"first": data},
        sources=(FigureSourceState(name="first", label="first"),),
        operations=(operation,),
        primary_source="first",
    )
    qtbot.addWidget(tool)

    [loaded_operation] = tool.tool_status.operations
    assert loaded_operation.sources == ("first",)
    assert loaded_operation.map_selections == ()
    assert loaded_operation.slice_dim == "y"
    assert loaded_operation.slice_values == (0.0,)
    assert loaded_operation.slice_kwargs == {}
    assert plot_slices_model._plot_slices_shape(tool._document, loaded_operation).valid

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )
    tool.operation_editor.select_section("selection")
    assert (
        tool.findChild(
            QtWidgets.QWidget,
            "figureComposerPlotSlicesInputSelectionSection",
        )
        is None
    )


def test_figure_composer_plot_slices_migrates_shared_multi_map_selection_to_slice_state(
    qtbot,
) -> None:
    first = _plot_slices_selection_migration_data()
    second = first + 100.0
    second.name = "second"
    tool = FigureComposerTool.from_sources(
        {"first": first, "second": second},
        sources=(
            FigureSourceState(name="first", label="first"),
            FigureSourceState(name="second", label="second"),
        ),
        operations=(
            FigureOperationState.plot_slices(
                label="plot_slices",
                sources=("first", "second"),
                map_selections=(
                    FigureDataSelectionState(source="first", qsel={"y": 0.0}),
                    FigureDataSelectionState(source="second", qsel={"y": 0.0}),
                ),
                axes=FigureAxesSelectionState(),
            ),
        ),
        primary_source="first",
    )
    qtbot.addWidget(tool)

    [loaded_operation] = tool.tool_status.operations
    assert loaded_operation.sources == ("first", "second")
    assert loaded_operation.map_selections == ()
    assert loaded_operation.slice_dim == "y"
    assert loaded_operation.slice_values == (0.0,)
    assert loaded_operation.slice_kwargs == {}
    shape = plot_slices_model._plot_slices_shape(tool._document, loaded_operation)
    assert shape.valid
    assert shape.plot_ndim == 2


def test_figure_composer_plot_slices_migrates_shared_selection_before_sources_restore(
    qtbot,
) -> None:
    first = _plot_slices_selection_migration_data()
    second = first + 100.0
    second.name = "second"
    primary = first.rename("primary")
    tool = FigureComposerTool.from_sources(
        {"primary": primary},
        sources=(
            FigureSourceState(name="primary", label="primary"),
            FigureSourceState(name="first", label="first"),
            FigureSourceState(name="second", label="second"),
        ),
        operations=(
            FigureOperationState.plot_slices(
                label="plot_slices",
                sources=("first", "second"),
                map_selections=(
                    FigureDataSelectionState(source="first", qsel={"y": 0.0}),
                    FigureDataSelectionState(source="second", qsel={"y": 0.0}),
                ),
                axes=FigureAxesSelectionState(),
            ),
        ),
        primary_source="primary",
    )
    qtbot.addWidget(tool)

    [loaded_operation] = tool.tool_status.operations
    assert loaded_operation.sources == ("first", "second")
    assert loaded_operation.map_selections == ()
    assert loaded_operation.slice_dim == "y"
    assert loaded_operation.slice_values == (0.0,)
    tool.set_source_data({"primary": primary, "first": first, "second": second})
    shape = plot_slices_model._plot_slices_shape(tool._document, loaded_operation)
    assert shape.valid
    assert shape.plot_ndim == 2


def test_figure_composer_plot_slices_migrates_per_source_map_selections_to_aliases(
    qtbot,
) -> None:
    first = xr.DataArray(
        np.arange(24.0).reshape(2, 3, 4),
        dims=("x", "hv", "y"),
        coords={"x": [0.0, 1.0], "hv": [10.0, 20.0, 30.0], "y": [-1.0, 0.0, 1.0, 2.0]},
        name="first",
    )
    second = first + 100.0
    second.name = "second"
    tool = FigureComposerTool.from_sources(
        {"first": first, "second": second},
        sources=(
            FigureSourceState(name="first", label="first"),
            FigureSourceState(name="second", label="second"),
        ),
        operations=(
            FigureOperationState.plot_slices(
                label="plot_slices",
                sources=("first", "second"),
                map_selections=(
                    FigureDataSelectionState(source="first", qsel={"y": 0.0}),
                    FigureDataSelectionState(source="second", qsel={"y": 2.0}),
                ),
                axes=FigureAxesSelectionState(),
            ),
        ),
        primary_source="first",
    )
    qtbot.addWidget(tool)

    [loaded_operation] = tool.tool_status.operations
    assert loaded_operation.sources == ("first_selected", "second_selected")
    assert loaded_operation.map_selections == ()
    source_by_name = {source.name: source for source in tool.source_states()}
    assert source_by_name["first_selected"].selection_source == "first"
    assert source_by_name["first_selected"].qsel == {"y": 0.0}
    assert source_by_name["second_selected"].selection_source == "second"
    assert source_by_name["second_selected"].qsel == {"y": 2.0}
    xr.testing.assert_identical(tool.source_data()["first_selected"], first.qsel(y=0.0))
    xr.testing.assert_identical(
        tool.source_data()["second_selected"], second.qsel(y=2.0)
    )


def test_figure_composer_plot_slices_restores_deferred_source_alias_data(
    qtbot,
) -> None:
    first = _plot_slices_selection_migration_data()
    second = first + 100.0
    second.name = "second"
    primary = first.rename("primary")
    tool = FigureComposerTool.from_sources(
        {"primary": primary},
        sources=(
            FigureSourceState(name="primary", label="primary"),
            FigureSourceState(name="first", label="first"),
            FigureSourceState(name="second", label="second"),
        ),
        operations=(
            FigureOperationState.plot_slices(
                label="plot_slices",
                sources=("first", "second"),
                map_selections=(
                    FigureDataSelectionState(source="first", qsel={"y": 0.0}),
                    FigureDataSelectionState(source="second", qsel={"y": 2.0}),
                ),
                axes=FigureAxesSelectionState(),
            ),
        ),
        primary_source="primary",
    )
    qtbot.addWidget(tool)

    [loaded_operation] = tool.tool_status.operations
    assert loaded_operation.sources == ("first_selected", "second_selected")
    assert "first_selected" not in tool.source_data()
    tool._restore_persistence_data_items(
        {"first": first, "second": second},
        xr.Dataset(),
    )
    xr.testing.assert_identical(tool.source_data()["first_selected"], first.qsel(y=0.0))
    xr.testing.assert_identical(
        tool.source_data()["second_selected"], second.qsel(y=2.0)
    )
    shape = plot_slices_model._plot_slices_shape(tool._document, loaded_operation)
    assert shape.valid
    assert shape.plot_ndim == 2


def test_figure_composer_plot_slices_source_selector_updates_sliced_sources(
    qtbot,
) -> None:
    first = _figure_composer_image_source("first")
    second = _figure_composer_image_source("second")
    tool = FigureComposerTool.from_sources(
        {"first": first, "second": second},
        sources=(
            FigureSourceState(name="first", label="first"),
            FigureSourceState(name="second", label="second"),
        ),
        operations=(
            FigureOperationState.plot_slices(
                label="plot_slices",
                sources=("first",),
                map_selections=(
                    FigureDataSelectionState(source="first", qsel={"eV": 0.0}),
                ),
            ),
        ),
        primary_source="first",
    )
    qtbot.addWidget(tool)
    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )
    tool.operation_editor.select_section("sources")

    checks = _plot_source_checks(tool)
    checks["second"].setCheckState(QtCore.Qt.CheckState.Checked)

    updated = tool.tool_status.operations[0]
    assert updated.sources == ("first", "second")
    assert updated.map_selections == ()

    qtbot.waitUntil(
        lambda: _plot_source_move_buttons(tool)[("second", "up")].isEnabled(),
        timeout=1000,
    )
    _plot_source_move_buttons(tool)[("second", "up")].click()
    qtbot.waitUntil(
        lambda: (
            tool.tool_status.operations[0].sources
            == (
                "second",
                "first",
            )
        ),
        timeout=1000,
    )

    updated = tool.tool_status.operations[0]
    assert updated.sources == ("second", "first")
    assert updated.map_selections == ()


def test_figure_composer_plot_slices_source_selector_updates_sources(
    qtbot, monkeypatch
) -> None:
    first = _figure_composer_image_source("first")
    second = _figure_composer_image_source("second")
    tool = FigureComposerTool(
        first,
        recipe=FigureRecipeState(
            sources=(
                FigureSourceState(name="first_source", label="first"),
                FigureSourceState(name="second_source", label="second"),
            ),
            operations=(
                FigureOperationState.plot_slices(
                    label="plot_slices",
                    sources=("first_source",),
                    axes=FigureAxesSelectionState(),
                    slice_dim="eV",
                    slice_values=(0.0,),
                ),
            ),
            primary_source="first_source",
        ),
        source_data={"first_source": first, "second_source": second},
    )
    qtbot.addWidget(tool)
    tool.operation_editor.select_section("sources")

    assert not tool.operation_editor.source_controls.findChildren(QtWidgets.QLineEdit)
    checks = _plot_source_checks(tool)
    assert set(checks) == {"first_source", "second_source"}
    assert checks["first_source"].checkState() == QtCore.Qt.CheckState.Checked
    assert checks["second_source"].checkState() == QtCore.Qt.CheckState.Unchecked
    assert tool.source_panel.source_status_label.isHidden()

    checks["second_source"].setCheckState(QtCore.Qt.CheckState.Checked)

    assert tool.tool_status.operations[0].sources == (
        "first_source",
        "second_source",
    )
    qtbot.waitUntil(
        lambda: _plot_source_move_buttons(tool)[("second_source", "up")].isEnabled(),
        timeout=1000,
    )
    buttons = _plot_source_move_buttons(tool)
    assert buttons[("first_source", "up")].isEnabled() is False
    assert buttons[("first_source", "down")].isEnabled() is True
    assert buttons[("second_source", "up")].isEnabled() is True
    assert buttons[("second_source", "down")].isEnabled() is False

    buttons[("second_source", "up")].click()

    assert tool.tool_status.operations[0].sources == (
        "second_source",
        "first_source",
    )
    captured_maps: list[tuple[str, ...]] = []

    def capture_plot_slices(maps, **_kwargs):
        captured_maps.append(tuple(data.name for data in maps))

    monkeypatch.setattr(eplt, "plot_slices", capture_plot_slices)
    namespace: dict[str, typing.Any] = {
        "first_source": first,
        "second_source": second,
    }
    exec(tool.generated_code(), namespace)  # noqa: S102
    assert captured_maps == [("second", "first")]


def test_figure_composer_plot_slices_selection_error_details() -> None:
    plain = str(FigureComposerPlotSlicesSelectionError())
    assert "Details:" not in plain

    detailed = str(FigureComposerPlotSlicesSelectionError("bad key"))
    assert plain in detailed
    assert "Details: bad key" in detailed


def test_figure_composer_opens_plot_slices_selection_on_nonuniform_data(
    qtbot,
) -> None:
    public = xr.DataArray(
        np.arange(24.0).reshape(4, 2, 3),
        dims=("sample_temp", "alpha", "eV"),
        coords={
            "sample_temp": [10.0, 15.0, 30.0, 60.0],
            "alpha": [0.0, 1.0],
            "eV": [-0.1, 0.0, 0.1],
        },
        name="map",
    )
    internal = erlab.utils.array._make_dims_uniform(public)
    operation = FigureOperationState.plot_slices(
        label="plot_slices",
        sources=("data",),
        map_selections=(
            FigureDataSelectionState(source="data", isel={"sample_temp": 1}),
        ),
    )

    tool = FigureComposerTool.from_sources(
        {"data": internal},
        sources=(FigureSourceState(name="data", label="map"),),
        operations=(operation,),
        setup=FigureSubplotsState(),
        primary_source="data",
    )
    qtbot.addWidget(tool)

    assert tool.operation_panel.operation_list.topLevelItemCount() == 1
    assert "sample_temp_idx" not in tool.generated_code()


def test_imagetool_main_image_seeds_nonuniform_plot_slices_selection(qtbot) -> None:
    public = xr.DataArray(
        np.arange(24.0).reshape(4, 2, 3),
        dims=("sample_temp", "alpha", "eV"),
        coords={
            "sample_temp": [10.0, 15.0, 30.0, 60.0],
            "alpha": [0.0, 1.0],
            "eV": [-0.1, 0.0, 0.1],
        },
        name="map",
    )
    tool = erlab.interactive.itool(public, manager=False, execute=False)
    assert isinstance(tool, erlab.interactive.imagetool.ImageTool)
    qtbot.addWidget(tool)

    tool.slicer_area.set_value(axis=0, value=30.0, cursor=0)
    operation = figurecomposer_adapter.build_figure_composer_operation(
        tool.slicer_area.images[2], source_name="data"
    )

    assert operation.kind == FigureOperationKind.PLOT_ARRAY
    assert len(operation.map_selections) == 1
    assert operation.map_selections[0].source == "data"
    assert "sample_temp_idx" not in operation.model_dump_json()


def test_imagetool_main_image_seeds_plot_slices_selection_with_spaced_dim(
    qtbot,
) -> None:
    data = xr.DataArray(
        np.arange(12.0).reshape(3, 2, 2),
        dims=("Track Shift", "kx", "ky"),
        coords={
            "Track Shift": [0.0, 1.0, 2.0],
            "kx": [0.0, 1.0],
            "ky": [0.0, 1.0],
        },
        name="map",
    )
    tool = erlab.interactive.itool(data, manager=False, execute=False)
    assert isinstance(tool, erlab.interactive.imagetool.ImageTool)
    qtbot.addWidget(tool)

    tool.slicer_area.set_value(axis=0, value=1.0, cursor=0)
    operation = figurecomposer_adapter.build_figure_composer_operation(
        tool.slicer_area.images[2], source_name="data"
    )

    assert operation.kind == FigureOperationKind.PLOT_ARRAY
    assert operation.map_selections == (
        FigureDataSelectionState(source="data", qsel={"Track Shift": 1.0}),
    )


def test_imagetool_rejects_uneditable_plot_slices_selection(qtbot) -> None:
    tool = erlab.interactive.imagetool.ImageTool(_unsupported_plot_slices_data())
    qtbot.addWidget(tool)

    _set_unsupported_plot_slices_cursor_state(tool)

    with pytest.raises(
        FigureComposerPlotSlicesSelectionError,
        match="Cannot plot when more than one dimension",
    ):
        figurecomposer_adapter.build_figure_composer_operation(
            tool.slicer_area.images[0], source_name="data"
        )


def test_imagetool_plot_slices_selection_warning_shows_error_detail(
    qtbot, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)

    messages: list[tuple[QtWidgets.QWidget | None, str, str]] = []

    def warning(
        parent: QtWidgets.QWidget | None, title: str, text: str
    ) -> QtWidgets.QMessageBox.StandardButton:
        messages.append((parent, title, text))
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", warning)

    figurecomposer_adapter.show_plot_slices_selection_error(
        parent,
        FigureComposerPlotSlicesSelectionError(
            "Cannot plot when more than one dimension"
        ),
    )

    assert len(messages) == 1
    assert messages[0][0] is parent
    assert messages[0][1] == "Cannot Create Plot Slices Figure"
    assert "Details:" in messages[0][2]
    assert "Cannot plot when more than one dimension" in messages[0][2]


def test_figure_composer_plot_slices_qsel_kwargs_display_in_selection(qtbot) -> None:
    data = _figure_composer_image_source("data")
    operation = FigureOperationState.plot_slices(
        label="image",
        sources=("data",),
    ).model_copy(
        update={
            "extra_kwargs": {
                "eV": 0.0,
                "eV_width": 0.1,
                "beta": slice(-0.5, 0.5),
                "zorder": 2,
            },
        }
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )
    tool.operation_editor.select_section("selection")
    selection_page = tool.operation_editor.stack.currentWidget()
    dimension_combo = selection_page.findChild(
        QtWidgets.QComboBox, "figureComposerPlotSlicesDimensionCombo"
    )
    values_edit = selection_page.findChild(
        QtWidgets.QLineEdit, "figureComposerPlotSlicesValuesEdit"
    )
    width_edit = selection_page.findChild(
        QtWidgets.QLineEdit, "figureComposerPlotSlicesWidthEdit"
    )
    slice_kwargs_edit = selection_page.findChild(
        QtWidgets.QLineEdit, "figureComposerPlotSlicesSliceKwargsEdit"
    )
    assert dimension_combo is not None
    assert values_edit is not None
    assert values_edit.text() == "0"
    assert width_edit is not None
    assert width_edit.text() == "0.1"
    assert slice_kwargs_edit is not None
    assert slice_kwargs_edit.text() == "beta=slice(-0.5, 0.5)"

    tool.operation_editor.select_section("advanced")
    extra_kwargs_edit = tool.operation_editor.stack.currentWidget().findChild(
        QtWidgets.QLineEdit, "figureComposerExtraKwEdit"
    )
    assert extra_kwargs_edit is not None
    assert extra_kwargs_edit.text() == "zorder=2"
    code = tool.generated_code()
    assert "eV=[0.0]" in code
    assert "eV_width=0.1" in code
    assert "beta=slice(-0.5, 0.5)" in code


def test_figure_composer_plot_slices_advanced_qsel_kwargs_move_to_selection(
    qtbot,
) -> None:
    data = _figure_composer_image_source("data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.plot_slices(
                    label="image",
                    sources=("data",),
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )
    tool.operation_editor.select_section("advanced")
    extra_kwargs_edit = tool.operation_editor.stack.currentWidget().findChild(
        QtWidgets.QLineEdit, "figureComposerExtraKwEdit"
    )
    assert extra_kwargs_edit is not None
    extra_kwargs_edit.setText("eV=0.5, eV_width=0.25, beta=slice(-0.5, 0.5), zorder=2")
    extra_kwargs_edit.editingFinished.emit()

    operation = tool.tool_status.operations[0]
    assert operation.slice_dim == "eV"
    assert operation.slice_values == (0.5,)
    assert operation.slice_width == pytest.approx(0.25)
    assert operation.extra_kwargs == {"zorder": 2}
    beta_slice = operation.slice_kwargs["beta"]
    assert isinstance(beta_slice, slice)
    assert beta_slice.start == pytest.approx(-0.5)
    assert beta_slice.stop == pytest.approx(0.5)
    assert beta_slice.step is None

    qtbot.waitUntil(
        lambda: not tool._operation_editor_update_pending,
        timeout=1000,
    )
    tool.operation_editor.select_section("selection")
    slice_kwargs_edit = tool.operation_editor.stack.currentWidget().findChild(
        QtWidgets.QLineEdit, "figureComposerPlotSlicesSliceKwargsEdit"
    )
    assert slice_kwargs_edit is not None
    assert slice_kwargs_edit.text() == "beta=slice(-0.5, 0.5)"


def test_plot_slices_source_styles_keep_default_cmap_panels() -> None:
    operation = FigureOperationState.plot_slices(
        label="plot_slices",
        sources=("data_0", "data_1"),
    )
    source_operations = (
        FigureOperationState.plot_slices(
            label="first",
            sources=("data_0",),
        ).model_copy(update={"cmap": "magma"}),
        FigureOperationState.plot_slices(
            label="second",
            sources=("data_1",),
        ),
    )

    seeded = plot_slices_operation_with_source_styles(
        operation,
        source_operations,
        selections_per_source=1,
    )

    assert seeded.cmap is None
    assert seeded.panel_styles_enabled
    assert {
        (style.map_index, style.slice_index): style.cmap
        for style in seeded.panel_styles
    } == {(0, 0): "magma"}
