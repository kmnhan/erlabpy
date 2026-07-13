# ruff: noqa: F403, F405

from ._common import *


def test_figure_composer_line_style_helpers_update_recipe(qtbot) -> None:
    assert "" not in figurecomposer_line_style.LINE_STYLE_OPTIONS
    assert " " not in figurecomposer_line_style.LINE_STYLE_OPTIONS
    assert "None" not in figurecomposer_line_style.LINE_STYLE_OPTIONS
    assert "none" in figurecomposer_line_style.LINE_STYLE_OPTIONS
    assert "" not in figurecomposer_line_style.LINE_MARKER_OPTIONS
    assert " " not in figurecomposer_line_style.LINE_MARKER_OPTIONS
    assert "None" not in figurecomposer_line_style.LINE_MARKER_OPTIONS
    assert "none" in figurecomposer_line_style.LINE_MARKER_OPTIONS

    assert figurecomposer_line_style.color_kw_value_from_text("") is None
    assert figurecomposer_line_style.color_kw_value_from_text("tab:blue") == "tab:blue"
    assert figurecomposer_line_style.color_kw_value_from_text("['red', 'blue']") == (
        "red",
        "blue",
    )
    assert figurecomposer_line_style.color_kw_value_from_text("('red', 'blue')") == (
        "red",
        "blue",
    )
    assert figurecomposer_line_style.color_kw_value_from_text("[bad") == "[bad"

    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    combo = QtWidgets.QComboBox(parent)
    figurecomposer_line_style.configure_style_combo(
        combo,
        figurecomposer_line_style.LINE_STYLE_OPTIONS,
        None,
    )
    assert combo.itemData(0) is None
    assert figurecomposer_line_style.style_combo_value(combo) is None
    figurecomposer_line_style.set_style_combo_value(combo, "")
    assert figurecomposer_line_style.style_combo_value(combo) == "none"
    figurecomposer_line_style.set_style_combo_value(combo, " ")
    assert figurecomposer_line_style.style_combo_value(combo) == "none"
    figurecomposer_line_style.set_style_combo_value(combo, "None")
    assert figurecomposer_line_style.style_combo_value(combo) == "none"
    figurecomposer_line_style.set_style_combo_value(combo, "custom-dash")
    assert figurecomposer_line_style.style_combo_value(combo) == "custom-dash"
    assert combo.itemText(combo.count() - 1) == "custom-dash"

    spinbox = figurecomposer_line_style.optional_positive_spinbox(None, parent=parent)
    assert (
        figurecomposer_line_style.optional_positive_spinbox_value(spinbox.value())
        is None
    )
    spinbox.setValue(2.5)
    assert (
        figurecomposer_line_style.optional_positive_spinbox_value(spinbox.value())
        == 2.5
    )

    operation = FigureOperationState.plot_slices(
        label="line-slices",
        sources=("data",),
    ).model_copy(update={"line_kw": {"lw": "2", "custom": 1}, "cmap": "magma"})
    assert figurecomposer_line_style.line_kw_text(operation, "linewidth", "lw") == "2"
    assert figurecomposer_line_style.line_kw_float(operation, "linewidth", "lw") == 2.0
    assert figurecomposer_line_style.extra_line_kw(operation) == {"custom": 1}
    bad_operation = operation.model_copy(update={"line_kw": {"linewidth": "bad"}})
    assert figurecomposer_line_style.line_kw_float(bad_operation, "linewidth") is None

    data = xr.DataArray(np.arange(3.0), dims=("x",), name="data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    tool.operation_list.setCurrentItem(tool.operation_list.topLevelItem(0))

    tool._updating_controls = True
    figurecomposer_line_style.update_current_line_kw(
        tool, "color", "red", clear_stale_cmap=True
    )
    tool._updating_controls = False
    assert tool.tool_status.operations[0].cmap == "magma"

    figurecomposer_line_style.update_current_line_kw(
        tool,
        "color",
        "red",
        aliases=("c",),
        clear_stale_cmap=True,
    )
    updated = tool.tool_status.operations[0]
    assert updated.line_kw["color"] == "red"
    assert "c" not in updated.line_kw
    assert updated.cmap is None

    tool._updating_controls = True
    figurecomposer_line_style.update_current_extra_line_kw(tool, {"zorder": 5})
    tool._updating_controls = False
    assert "zorder" not in tool.tool_status.operations[0].line_kw

    figurecomposer_line_style.update_current_extra_line_kw(tool, {"zorder": 5})
    assert tool.tool_status.operations[0].line_kw == {
        "lw": "2",
        "color": "red",
        "zorder": 5,
    }


def test_figure_composer_step_editor_section_headers_are_native_subgroups(
    qtbot,
) -> None:
    image = _figure_composer_image_source("image")
    line_map = _figure_composer_line_slice_source("line_map")
    recipe = FigureRecipeState(
        sources=(
            FigureSourceState(name="image", label="image"),
            FigureSourceState(name="line_map", label="line map"),
        ),
        operations=(
            FigureOperationState.line(label="profiles", source="line_map").model_copy(
                update={"line_iter_dim": "eV", "line_color_mode": "coordinate"}
            ),
            FigureOperationState.plot_slices(
                label="line slices",
                sources=("line_map",),
                slice_dim="eV",
                slice_values=(-0.5, 0.5),
            ).model_copy(update={"gradient": True}),
            FigureOperationState.plot_slices(
                label="image slices",
                sources=("image",),
                slice_dim="eV",
                slice_values=(0.0,),
            ).model_copy(update={"colorbar": "right"}),
            FigureOperationState.method(
                family=FigureMethodFamily.AXES,
                name="plot",
                axes=FigureAxesSelectionState(axes=((0, 0),)),
            ),
            FigureOperationState.plot_array(
                label="image plot",
                source="image",
            ).model_copy(update={"colorbar": "right"}),
        ),
        primary_source="image",
    )
    tool = FigureComposerTool(
        image,
        recipe=recipe,
        source_data={"image": image, "line_map": line_map},
    )
    qtbot.addWidget(tool)

    tool.operation_list.setCurrentItem(tool.operation_list.topLevelItem(0))
    tool._update_operation_editor()
    tool._select_step_section("selection")
    line_selection_page = tool.step_editor_stack.currentWidget()
    assert line_selection_page is not None
    _assert_step_editor_section(
        line_selection_page, "figureComposerLineSelectionDataSection"
    )
    _assert_step_editor_section(
        line_selection_page, "figureComposerLineSelectionProfilesSection"
    )
    tool._select_step_section("view")
    line_view_page = tool.step_editor_stack.currentWidget()
    assert line_view_page is not None
    assert (
        line_view_page.findChild(
            QtWidgets.QWidget, "figureComposerLineViewPlacementSection"
        )
        is None
    )
    tool._select_step_section("style")
    line_style_page = tool.step_editor_stack.currentWidget()
    assert line_style_page is not None
    _assert_step_editor_section(line_style_page, "figureComposerLineStyleLegendSection")
    _assert_step_editor_section(line_style_page, "figureComposerLineStyleColorSection")
    _assert_step_editor_section(line_style_page, "figureComposerLineStyleLineSection")
    _assert_step_editor_section(line_style_page, "figureComposerLineStyleFillSection")
    tool._select_step_section("other")
    line_other_page = tool.step_editor_stack.currentWidget()
    assert line_other_page is not None
    assert tool.step_section_buttons["other"].property("section_title") == "Transform"
    assert (
        line_other_page.findChild(
            QtWidgets.QWidget, "figureComposerLineOtherTransformSection"
        )
        is None
    )

    tool.operation_list.setCurrentItem(tool.operation_list.topLevelItem(1))
    tool._update_operation_editor()
    tool._select_step_section("selection")
    line_slices_selection_page = tool.step_editor_stack.currentWidget()
    assert line_slices_selection_page is not None
    _assert_step_editor_section(
        line_slices_selection_page,
        "figureComposerPlotSlicesSelectionDimensionsSection",
    )
    _assert_step_editor_section(
        line_slices_selection_page,
        "figureComposerPlotSlicesSelectionValuesSection",
    )
    tool._select_step_section("view")
    line_slices_view_page = tool.step_editor_stack.currentWidget()
    assert line_slices_view_page is not None
    _assert_step_editor_section(
        line_slices_view_page, "figureComposerPlotSlicesViewPanelsSection"
    )
    _assert_step_editor_section(
        line_slices_view_page, "figureComposerPlotSlicesViewAxesSection"
    )
    tool._select_step_section("colors")
    line_slices_colors_page = tool.step_editor_stack.currentWidget()
    assert line_slices_colors_page is not None
    _assert_step_editor_section(
        line_slices_colors_page,
        "figureComposerPlotSlicesStyleLegendSection",
    )
    _assert_step_editor_section(
        line_slices_colors_page,
        "figureComposerPlotSlicesStyleLineSection",
    )
    _assert_step_editor_section(
        line_slices_colors_page,
        "figureComposerPlotSlicesStyleFillSection",
    )
    _assert_step_editor_section(
        line_slices_colors_page,
        "figureComposerPlotSlicesStylePanelOverridesSection",
    )
    tool._select_step_section("transform")
    line_slices_transform_page = tool.step_editor_stack.currentWidget()
    assert line_slices_transform_page is not None
    assert (
        line_slices_transform_page.objectName()
        == "figureComposerPlotSlicesTransformPage"
    )
    assert (
        line_slices_transform_page.findChild(
            QtWidgets.QWidget, "figureComposerPlotSlicesColorsTransformSection"
        )
        is None
    )

    tool.operation_list.setCurrentItem(tool.operation_list.topLevelItem(2))
    tool._update_operation_editor()
    tool._select_step_section("colors")
    image_slices_colors_page = tool.step_editor_stack.currentWidget()
    assert image_slices_colors_page is not None
    _assert_step_editor_section(
        image_slices_colors_page,
        "figureComposerPlotSlicesColorsImageColorSection",
    )
    _assert_step_editor_section(
        image_slices_colors_page,
        "figureComposerPlotSlicesColorsColorbarSection",
    )
    _assert_step_editor_section(
        image_slices_colors_page,
        "figureComposerPlotSlicesColorsPanelOverridesSection",
    )

    tool.operation_list.setCurrentItem(tool.operation_list.topLevelItem(3))
    tool._update_operation_editor()
    tool._select_step_section("method")
    method_page = tool.step_editor_stack.currentWidget()
    assert method_page is not None
    _assert_step_editor_section(method_page, "figureComposerMethodCallSection")
    _assert_step_editor_section(method_page, "figureComposerMethodValuesSection")
    _assert_step_editor_section(method_page, "figureComposerMethodAdvancedSection")

    tool.operation_list.setCurrentItem(tool.operation_list.topLevelItem(4))
    tool._update_operation_editor()
    tool._select_step_section("view")
    plot_array_view_page = tool.step_editor_stack.currentWidget()
    assert plot_array_view_page is not None
    _assert_step_editor_section(
        plot_array_view_page, "figureComposerPlotArrayViewImageSection"
    )
    _assert_step_editor_section(
        plot_array_view_page, "figureComposerPlotArrayViewAxesSection"
    )
    tool._select_step_section("colors")
    plot_array_colors_page = tool.step_editor_stack.currentWidget()
    assert plot_array_colors_page is not None
    _assert_step_editor_section(
        plot_array_colors_page,
        "figureComposerPlotArrayColorsImageColorSection",
    )
    _assert_step_editor_section(
        plot_array_colors_page,
        "figureComposerPlotArrayColorsColorbarSection",
    )


def test_figure_composer_norm_helpers_cover_structured_and_custom_norms(
    monkeypatch,
) -> None:
    assert figurecomposer_norms._norm_module_prefix("Normalize") == "mcolors"
    assert figurecomposer_norms._norm_module_prefix("CenteredPowerNorm") == "eplt"
    assert figurecomposer_norms._norm_combo_choices("CustomNorm")[-1] == "CustomNorm"
    assert figurecomposer_norms._norm_kwarg_fields("CustomNorm") == ("gamma",)
    assert figurecomposer_norms._norm_float_value(None) is None
    assert figurecomposer_norms._norm_updates_from_kwargs(
        {"gamma": 2, "halfrange": 1, "clip": None, "extra": "value"}
    ) == {
        "norm_gamma": 2.0,
        "halfrange": 1.0,
        "norm_clip": None,
        "norm_kwargs": {"extra": "value"},
    }
    assert figurecomposer_norms._cmap_base_and_reverse("magma_r") == ("magma", True)
    assert figurecomposer_norms._cmap_with_reverse("", False) is None
    assert figurecomposer_norms._cmap_with_reverse("magma_r", False) == "magma"

    power_from_plot_gamma = FigureOperationState.plot_slices(
        label="power",
        sources=("data",),
    ).model_copy(update={"gamma": 0.5})
    assert (
        figurecomposer_norms._norm_constructor_kwargs(power_from_plot_gamma)["gamma"]
        == 0.5
    )

    two_slope = FigureOperationState.plot_slices(
        label="two-slope",
        sources=("data",),
    ).model_copy(update={"norm_name": "TwoSlopeNorm"})
    assert figurecomposer_norms._norm_constructor_kwargs(two_slope)["vcenter"] == 0.0

    custom_calls: list[tuple[tuple[typing.Any, ...], dict[str, typing.Any]]] = []

    class CustomNorm:
        def __init__(self, *args, **kwargs) -> None:
            custom_calls.append((args, kwargs))

    monkeypatch.setattr(eplt, "CustomNorm", CustomNorm, raising=False)
    custom = FigureOperationState.plot_slices(
        label="custom",
        sources=("data",),
    ).model_copy(
        update={
            "norm_name": "CustomNorm",
            "norm_gamma": 2.0,
            "norm_kwargs": {"alpha": 3},
        }
    )
    figurecomposer_norms._norm_object(custom)
    assert custom_calls[-1] == ((2.0,), {"alpha": 3})
    assert figurecomposer_norms._norm_code(custom) == "eplt.CustomNorm(2.0, alpha=3)"

    custom_no_gamma = custom.model_copy(update={"norm_gamma": None})
    figurecomposer_norms._norm_object(custom_no_gamma)
    assert custom_calls[-1] == ((), {"alpha": 3})
    assert figurecomposer_norms._norm_code(custom_no_gamma) == (
        "eplt.CustomNorm(alpha=3)"
    )


def test_figure_composer_cmap_helpers_normalize_colorcet_names() -> None:
    pytest.importorskip("colorcet")

    erlab.interactive.colors.load_all_colormaps()

    assert figurecomposer_norms._cmap_base_and_reverse("CET_C1_r") == (
        "cet_CET_C1",
        True,
    )
    assert figurecomposer_norms._cmap_with_reverse("fire", True) == "cet_fire_r"


def test_figure_composer_axes_helpers_parse_safe_expressions() -> None:
    axs = np.empty((2, 2), dtype=object)
    axs[0, 0] = "ax00"
    axs[0, 1] = "ax01"
    axs[1, 0] = "ax10"
    axs[1, 1] = "ax11"

    assert figurecomposer_axes._compact_axes_code(()) is None
    assert figurecomposer_axes._compact_axes_iterable_code((), nrows=2, ncols=2) is None
    assert figurecomposer_axes._axes_expression_value("axs", axs) is axs
    assert figurecomposer_axes._axes_expression_value("ax", axs) == "ax00"
    assert figurecomposer_axes._axes_expression_value("axs[-1, 0]", axs) == "ax10"
    assert figurecomposer_axes._axes_expression_value(
        "axs[[0, 1], 0]", axs
    ).tolist() == [
        "ax00",
        "ax10",
    ]
    with pytest.raises(ValueError, match="Unsupported axes name"):
        figurecomposer_axes._axes_expression_value("figure", axs)
    with pytest.raises(ValueError, match="integer indices"):
        figurecomposer_axes._axes_expression_value("axs[1.5]", axs)
    with pytest.raises(ValueError, match="integer indices"):
        figurecomposer_axes._axes_expression_value("axs[-None]", axs)
    with pytest.raises(ValueError, match="Unsupported axes expression"):
        figurecomposer_axes._axes_expression_value("axs + axs", axs)


def test_figure_composer_line_transform_helpers_cover_edge_cases() -> None:
    profile = xr.DataArray(
        [0.0, 0.0],
        dims=("kx",),
        coords={"kx": [0.0, 1.0], "center": 2.0},
    )
    assert _line_transform.line_normalize_from_text("unknown") == "none"
    with pytest.raises(ValueError, match="Cannot normalize profile by max"):
        _line_transform.normalize_line_data(profile, "max")
    assert _line_transform.line_transform_values((), 0, default=1.0) == ()
    assert _line_transform.line_transform_values((), 2, default=1.0) == (1.0, 1.0)
    assert _line_transform.line_transform_values((2.0,), 2, default=1.0) == (
        2.0,
        2.0,
    )
    assert _line_transform.line_transform_values((1.0, 2.0), 2, default=1.0) == (
        1.0,
        2.0,
    )
    with pytest.raises(ValueError, match="one value or one per profile"):
        _line_transform.line_transform_values((1.0, 2.0, 3.0), 2, default=1.0)

    coordinate_operation = FigureOperationState.line(
        label="profile",
        source="data",
    ).model_copy(update={"line_offset_source": "coordinate"})
    with pytest.raises(ValueError, match="One profile per"):
        _line_transform.line_offset_coordinate_name(coordinate_operation)

    associated_operation = coordinate_operation.model_copy(
        update={"line_offset_source": "associated"}
    )
    with pytest.raises(ValueError, match="require a coordinate"):
        _line_transform.line_offset_coordinate_name(associated_operation)
    with pytest.raises(ValueError, match="no coordinate"):
        _line_transform.profile_scalar_coord_value(profile, "missing")
    with pytest.raises(ValueError, match="not scalar"):
        _line_transform.profile_scalar_coord_value(profile, "kx")

    offset_operation = associated_operation.model_copy(
        update={"line_offset_coord": "center", "line_offset_scale": 0.5}
    )
    assert _line_transform.line_offsets_for_profiles(offset_operation, (profile,)) == (
        1.0,
    )

    code_operation = FigureOperationState.line(
        label="profile",
        source="data",
    ).model_copy(
        update={
            "line_scales": (2.0,),
            "line_offsets": (1.0,),
            "line_normalize": "none",
        }
    )
    assert _line_transform.profile_transform_code_lines(code_operation) == [
        "profiles = [",
        "    1.0 + 2.0 * profile",
        "    for profile in profiles",
        "]",
    ]
    normalized_operation = code_operation.model_copy(update={"line_normalize": "max"})
    with pytest.raises(ValueError, match="Cannot normalize profile by max"):
        _line_transform.profile_transform_code_lines(
            normalized_operation,
            profiles=(profile,),
        )

    stack_data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("cut", "kx"),
        coords={"cut": [1.0, 2.0], "kx": [-1.0, 0.0, 1.0]},
    )
    stack_operation = FigureOperationState.line(
        label="profile",
        source="data",
    ).model_copy(
        update={
            "line_iter_dim": "cut",
            "line_normalize": "mean",
            "line_scales": (0.5,),
            "line_offset_source": "coordinate",
            "line_offset_scale": 2.0,
        }
    )
    assert _line_transform.profile_stack_transform_code(
        stack_operation,
        data_name="profile_data",
        line_data=stack_data,
    ) == (
        "2.0 * profile_data['cut'] + "
        "0.5 * (profile_data / profile_data.mean('kx', skipna=True))"
    )
    multi_dim_stack = xr.DataArray(
        np.arange(12.0).reshape(2, 2, 3),
        dims=("cut", "eV", "kx"),
        coords={"cut": [1.0, 2.0], "eV": [0.0, 1.0], "kx": [-1.0, 0.0, 1.0]},
    )
    assert (
        _line_transform.profile_stack_transform_code(
            stack_operation,
            data_name="profile_data",
            line_data=multi_dim_stack,
        )
        is None
    )
    stack_code_operation = code_operation.model_copy(update={"line_iter_dim": "cut"})
    assert (
        _line_transform.profile_stack_transform_code(
            stack_code_operation,
            data_name="profile_data",
            line_data=stack_data,
        )
        == "1.0 + 2.0 * profile_data"
    )
    manual_multi_offset = code_operation.model_copy(
        update={"line_scales": (), "line_offsets": (1.0, 2.0)}
    )
    assert (
        _line_transform.profile_stack_transform_code(
            manual_multi_offset,
            data_name="profile_data",
            line_data=stack_data,
        )
        is None
    )
    assert _line_transform.profile_transform_code_lines(manual_multi_offset) == [
        "profiles = [",
        "    offset + profile",
        "    for profile, offset in zip(",
        "        profiles,",
        "        [1.0, 2.0],",
        "        strict=True,",
        "    )",
        "]",
    ]
    index_offset = code_operation.model_copy(
        update={
            "line_scales": (),
            "line_offsets": (),
            "line_offset_source": "index",
            "line_offset_scale": 3.0,
        }
    )
    assert (
        _line_transform.profile_stack_transform_code(
            index_offset,
            data_name="profile_data",
            line_data=stack_data,
        )
        is None
    )
    assert _line_transform.profile_transform_code_lines(index_offset)[4] == (
        "        [3.0 * float(index) for index in range(len(profiles))],"
    )


def test_figure_composer_codegen_axes_helpers_cover_invalid_targets(qtbot) -> None:
    data = xr.DataArray(np.arange(2.0), dims=("x",), name="data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=1),
            sources=(FigureSourceState(name="data", label="data"),),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    with pytest.raises(ValueError, match="outside the current layout"):
        figurecomposer_code._axes_sequence_code(
            tool._document, FigureAxesSelectionState(axes=((1, 0),))
        )
    with pytest.raises(ValueError, match="No axes"):
        figurecomposer_code._axes_sequence_code(
            tool._document, FigureAxesSelectionState(axes=())
        )

    root = FigureGridSpecGridState(
        grid_id="root",
        nrows=1,
        ncols=2,
        axes=(
            FigureGridSpecAxesState(
                axes_id="axis-a",
                span=FigureGridSpecSpanState(
                    row_start=0,
                    row_stop=1,
                    col_start=0,
                    col_stop=1,
                ),
            ),
            FigureGridSpecAxesState(
                axes_id="axis-b",
                span=FigureGridSpecSpanState(
                    row_start=0,
                    row_stop=1,
                    col_start=1,
                    col_stop=2,
                ),
            ),
        ),
    )
    grid_tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(
                layout_mode="gridspec",
                gridspec=FigureGridSpecLayoutState(root=root),
            ),
            sources=(FigureSourceState(name="data", label="data"),),
            primary_source="data",
        ),
    )
    qtbot.addWidget(grid_tool)

    assert (
        figurecomposer_code._axes_code(
            grid_tool._document,
            FigureAxesSelectionState(axes_ids=("axis-a", "axis-b")),
            for_plot_slices=False,
        )
        == "[ax0, ax1]"
    )
    with pytest.raises(ValueError, match="No axes"):
        figurecomposer_code._axes_sequence_code(
            grid_tool._document, FigureAxesSelectionState(axes_ids=())
        )


def test_figure_composer_gridspec_helpers_cover_naming_and_region_edges() -> None:
    invalid_child = FigureGridSpecGridState(
        grid_id="child",
        nrows=1,
        ncols=1,
        span=None,
        axes=(
            FigureGridSpecAxesState(
                axes_id="nested-axis",
                span=FigureGridSpecSpanState(
                    row_start=0,
                    row_stop=1,
                    col_start=0,
                    col_stop=1,
                ),
            ),
        ),
    )
    root = FigureGridSpecGridState(
        grid_id="root",
        nrows=1,
        ncols=2,
        axes=(
            FigureGridSpecAxesState(
                axes_id="axis-a",
                span=FigureGridSpecSpanState(
                    row_start=0,
                    row_stop=1,
                    col_start=0,
                    col_stop=1,
                ),
            ),
            FigureGridSpecAxesState(
                axes_id="axis-b",
                label="custom_axis",
                span=FigureGridSpecSpanState(
                    row_start=0,
                    row_stop=1,
                    col_start=1,
                    col_stop=2,
                ),
            ),
        ),
        child_grids=(invalid_child,),
    )
    setup = FigureSubplotsState(
        layout_mode="gridspec",
        gridspec=FigureGridSpecLayoutState(root=root),
    )

    assert (
        figurecomposer_gridspec._gridspec_axis_variable_name_error(setup, "axis-b", "")
        == ""
    )
    assert (
        figurecomposer_gridspec._gridspec_axis_variable_name_error(
            setup, "axis-b", "ax0"
        )
        == "Variable name conflicts with an autogenerated name."
    )
    overlap_child = FigureGridSpecGridState(
        grid_id="overlap-child",
        nrows=1,
        ncols=1,
        span=FigureGridSpecSpanState(
            row_start=0,
            row_stop=1,
            col_start=0,
            col_stop=1,
        ),
    )
    overlap_root = FigureGridSpecGridState(
        grid_id="root",
        nrows=1,
        ncols=1,
        child_grids=(overlap_child,),
    )
    assert figurecomposer_gridspec._gridspec_region_overlaps(
        overlap_root,
        FigureGridSpecSpanState(
            row_start=0,
            row_stop=1,
            col_start=0,
            col_stop=1,
        ),
    )
    assert tuple(figurecomposer_gridspec._iter_valid_axes_with_grid(root)) == (
        (root, root.axes[0]),
        (root, root.axes[1]),
    )
    assert (
        figurecomposer_gridspec._axis_code_name_for_validation(
            setup,
            "axis-b",
            FigureGridSpecAxesState(
                axes_id="missing-axis",
                span=FigureGridSpecSpanState(
                    row_start=0,
                    row_stop=1,
                    col_start=0,
                    col_stop=1,
                ),
            ),
            0,
            reserved_names=(),
        )
        == "ax0_1"
    )
    assert (
        figurecomposer_gridspec._unique_axis_code_name("ax", {"ax", "ax_1"}) == "ax_2"
    )


def test_figure_composer_editor_control_adapters_cover_mixed_states(qtbot) -> None:
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)

    edit = QtWidgets.QLineEdit(parent)
    line_adapter = _editor_controls.LineEditControlAdapter(edit)
    assert line_adapter.mixed_row_widget(mixed=False) is edit
    mixed_widget = line_adapter.mixed_row_widget(mixed=True, parent=parent)
    assert (
        mixed_widget.findChild(QtWidgets.QLabel, "figureComposerMixedValueMarker")
        is not None
    )
    line_adapter.set_mixed(True)
    assert line_adapter.unchanged_mixed()
    edit.setText("edited")
    edit.setModified(True)
    assert not line_adapter.unchanged_mixed()

    plain = QtWidgets.QPlainTextEdit(parent)
    plain_adapter = _editor_controls.PlainTextControlAdapter(plain)
    plain_adapter.set_mixed(True)
    assert plain_adapter.unchanged_mixed()
    plain.document().setModified(True)
    assert not plain_adapter.unchanged_mixed()

    combo = QtWidgets.QComboBox(parent)
    combo.addItems(["a", "b"])
    combo_adapter = _editor_controls.ComboBoxControlAdapter(combo)
    combo_adapter.set_mixed(True)
    assert combo.currentData() is _editor_controls.MIXED_VALUE
    assert not combo.model().item(0).isEnabled()

    committed_values: list[str] = []
    commit_combo = QtWidgets.QComboBox(parent)
    commit_combo.addItems(["a", "b"])
    _editor_controls.ComboBoxControlAdapter(commit_combo).connect_commit(
        lambda _widget, signal, callback: signal.connect(callback),
        committed_values.append,
    )
    commit_combo.setCurrentIndex(1)
    assert committed_values == []
    commit_combo.activated.emit(1)
    assert committed_values == ["b"]

    check = QtWidgets.QCheckBox(parent)
    check_adapter = _editor_controls.CheckBoxControlAdapter(check)
    check_adapter.set_mixed(True)
    assert check.checkState() == QtCore.Qt.CheckState.PartiallyChecked

    destroyed = QtWidgets.QLineEdit()
    destroyed_adapter = _editor_controls.LineEditControlAdapter(destroyed)
    del destroyed
    gc.collect()
    with pytest.raises(RuntimeError, match="destroyed"):
        _ = destroyed_adapter.widget


def test_figure_composer_state_validators_cover_invalid_values() -> None:
    with pytest.raises(ValueError, match="before zero"):
        FigureGridSpecSpanState(
            row_start=-1,
            row_stop=1,
            col_start=0,
            col_stop=1,
        )
    with pytest.raises(ValueError, match="at least one cell"):
        FigureGridSpecSpanState(
            row_start=1,
            row_stop=1,
            col_start=0,
            col_stop=1,
        )
    with pytest.raises(ValueError, match="at least one row"):
        FigureGridSpecGridState(nrows=0, ncols=1)
    with pytest.raises(ValueError, match="ratios must be positive"):
        FigureGridSpecGridState(nrows=1, ncols=1, width_ratios=(0.0,))
    with pytest.raises(ValueError, match="at least one row"):
        FigureSubplotsState(nrows=0)
    with pytest.raises(ValueError, match="figsize values"):
        FigureSubplotsState(figsize=(0.0, 1.0))
    with pytest.raises(ValueError, match="dpi must be positive"):
        FigureSubplotsState(dpi=0.0)
    with pytest.raises(ValueError, match="ratios must be positive"):
        FigureSubplotsState(width_ratios=(-1.0,))
    with pytest.raises(ValueError, match="export dpi must be positive"):
        FigureExportState(dpi=0.0)

    empty_selection = FigureAxesSelectionState(axes=())
    assert empty_selection.bounded(FigureSubplotsState()).axes == ((0, 0),)


def test_figure_composer_rendering_helpers_cover_selection_edges(qtbot) -> None:
    data = xr.DataArray(np.arange(2.0), dims=("x",), name="data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=1, sharex=False, sharey=False),
            sources=(FigureSourceState(name="data", label="data"),),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    assert "sharex" not in figurecomposer_rendering._setup_kwargs(tool._document)
    assert "sharey" not in figurecomposer_rendering._setup_kwargs(tool._document)

    fig, axs = plt.subplots(1, 1, squeeze=False)
    with pytest.raises(ValueError, match="outside the current figure"):
        figurecomposer_rendering._axes_from_selection(
            tool,
            FigureAxesSelectionState(axes=((2, 0),)),
            axs,
            for_plot_slices=False,
        )
    with pytest.raises(ValueError, match="No axes"):
        figurecomposer_rendering._axes_from_selection(
            tool,
            FigureAxesSelectionState(axes=()),
            axs,
            for_plot_slices=False,
        )
    plt.close(fig)

    grid_axis = FigureGridSpecAxesState(
        axes_id="axis-a",
        span=FigureGridSpecSpanState(
            row_start=0,
            row_stop=1,
            col_start=0,
            col_stop=1,
        ),
    )
    grid_tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(
                layout_mode="gridspec",
                gridspec=FigureGridSpecLayoutState(
                    root=FigureGridSpecGridState(
                        grid_id="root",
                        nrows=1,
                        ncols=1,
                        axes=(grid_axis,),
                    )
                ),
            ),
            sources=(FigureSourceState(name="data", label="data"),),
            primary_source="data",
        ),
    )
    qtbot.addWidget(grid_tool)
    grid_fig, grid_axs = plt.subplots(1, 1)
    axes_by_id = {"axis-a": grid_axs}
    with pytest.raises(ValueError, match="Advanced axes expressions"):
        figurecomposer_rendering._axes_from_selection(
            grid_tool,
            FigureAxesSelectionState(expression="axs"),
            axes_by_id,
            for_plot_slices=False,
        )
    with pytest.raises(ValueError, match="outside the current GridSpec"):
        figurecomposer_rendering._axes_from_selection(
            grid_tool,
            FigureAxesSelectionState(axes_ids=("missing",)),
            axes_by_id,
            for_plot_slices=False,
        )
    with pytest.raises(ValueError, match="No axes"):
        figurecomposer_rendering._axes_from_selection(
            grid_tool,
            FigureAxesSelectionState(axes_ids=()),
            axes_by_id,
            for_plot_slices=False,
        )
    assert figurecomposer_rendering._iter_axes({"axis-a": grid_axs}) == (grid_axs,)
    assert figurecomposer_rendering._iter_axes([grid_axs]) == (grid_axs,)
    assert figurecomposer_rendering._render_error_text(RuntimeError()) == "RuntimeError"
    plt.close(grid_fig)
