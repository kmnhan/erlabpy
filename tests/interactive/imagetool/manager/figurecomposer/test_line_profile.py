# ruff: noqa: F403, F405

import erlab.interactive._figurecomposer._labels as figurecomposer_labels
import erlab.interactive._figurecomposer._ui._color_widgets as color_widgets
import erlab.interactive.imagetool._figurecomposer_adapter as figurecomposer_adapter
from erlab.interactive._figurecomposer._model._document import FigureDocument
from erlab.interactive._figurecomposer._operations._plot_slices import (
    _codegen as plot_slices_codegen,
)
from erlab.interactive._figurecomposer._operations._plot_slices import (
    _model as plot_slices_model,
)

from ._common import *


def test_figure_composer_line_migrates_single_map_selection_to_source_alias(
    qtbot,
) -> None:
    data = xr.DataArray(
        np.arange(12.0).reshape(3, 4),
        dims=("cut", "kx"),
        coords={"cut": [0.0, 1.0, 2.0], "kx": [-1.0, -0.5, 0.5, 1.0]},
        name="profile",
    )
    operation = FigureOperationState.line(
        label="profiles", source="profile"
    ).model_copy(
        update={
            "map_selections": (
                FigureDataSelectionState(source="profile", qsel={"cut": 0.0}),
            ),
            "line_x": "kx",
            "line_y": "cut",
            "line_selection": {"kx": 0.5},
            "line_iter_dim": "cut",
            "line_reduce": "both",
            "line_reduce_coarsen": 3,
            "line_reduce_thin": 4,
        }
    )
    tool = FigureComposerTool.from_sources(
        {"profile": data},
        sources=(FigureSourceState(name="profile", label="profile"),),
        operations=(operation,),
        primary_source="profile",
    )
    qtbot.addWidget(tool)

    [loaded_operation] = tool.tool_status.operations
    assert loaded_operation.line_source == "profile_selected"
    assert loaded_operation.map_selections == ()
    assert loaded_operation.line_y is None
    assert loaded_operation.line_selection == {}
    assert loaded_operation.line_iter_dim is None
    assert loaded_operation.line_reduce == "disabled"
    assert loaded_operation.line_reduce_coarsen == 2
    assert loaded_operation.line_reduce_thin == 2
    source_by_name = {source.name: source for source in tool.source_states()}
    assert source_by_name["profile_selected"].selection_source == "profile"
    assert source_by_name["profile_selected"].qsel == {"cut": 0.0}
    xr.testing.assert_identical(
        tool.source_data()["profile_selected"], data.qsel(cut=0.0)
    )
    [profile] = figurecomposer_line_profile._line_data_items(tool, loaded_operation)
    xr.testing.assert_identical(profile, data.qsel(cut=0.0))

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )
    tool.operation_editor.select_section("selection")

    assert tool.findChild(QtWidgets.QLineEdit, "figureComposerLineSelectionEdit")
    assert (
        tool.findChild(QtWidgets.QWidget, "figureComposerLineInputSelectionSection")
        is None
    )


def test_figure_composer_line_preserves_multi_cursor_legacy_selections(
    qtbot,
) -> None:
    data = xr.DataArray(
        np.arange(12.0).reshape(3, 4),
        dims=("cut", "kx"),
        coords={"cut": [0.0, 1.0, 2.0], "kx": [-1.0, -0.5, 0.5, 1.0]},
        name="profile",
    )
    selections = (
        FigureDataSelectionState(source="profile", qsel={"cut": 0.0}),
        FigureDataSelectionState(source="profile", qsel={"cut": 2.0}),
    )
    operation = FigureOperationState.line(
        label="profiles", source="profile"
    ).model_copy(
        update={
            "map_selections": selections,
            "line_x": "kx",
            "line_labels": ("first", "second"),
        }
    )
    tool = FigureComposerTool.from_sources(
        {"profile": data},
        sources=(FigureSourceState(name="profile"),),
        operations=(operation,),
        primary_source="profile",
    )
    qtbot.addWidget(tool)

    [loaded_operation] = tool.tool_status.operations
    assert loaded_operation.map_selections == selections
    assert figurecomposer_operation_metadata.declared_operation_source_names(
        loaded_operation
    ) == ("profile",)
    profiles = figurecomposer_line_profile._line_data_items(tool, loaded_operation)
    assert len(profiles) == 2
    xr.testing.assert_identical(profiles[0], data.qsel(cut=0.0))
    xr.testing.assert_identical(profiles[1], data.qsel(cut=2.0))

    one_per_axis_operation = loaded_operation.model_copy(
        update={
            "axes": FigureAxesSelectionState(axes=((0, 0),)),
            "line_placement": "one_per_axis",
        }
    )
    selection_code = figurecomposer_line_profile._line_selection_code(
        tool, one_per_axis_operation
    )
    assert "profiles = [" in selection_code
    assert "for profile, label in zip(" in selection_code

    figure = plt.figure()
    try:
        figurecomposer_rendering._render_into_figure(tool, figure, sync_visible=False)
        rendered_lines = figure.axes[0].lines
        assert len(rendered_lines) == 2
        np.testing.assert_allclose(rendered_lines[0].get_ydata(), data.qsel(cut=0.0))
        np.testing.assert_allclose(rendered_lines[1].get_ydata(), data.qsel(cut=2.0))
    finally:
        plt.close(figure)

    namespace = _exec_generated_code(tool.generated_code(), {"profile": data})
    generated_lines = namespace["fig"].axes[0].lines
    assert len(generated_lines) == 2
    np.testing.assert_allclose(generated_lines[0].get_ydata(), data.qsel(cut=0.0))
    np.testing.assert_allclose(generated_lines[1].get_ydata(), data.qsel(cut=2.0))

    operation_payload = loaded_operation.model_dump(mode="json")
    assert operation_payload["map_selections"] == [
        selection.model_dump(mode="json") for selection in selections
    ]
    assert "map_selections" not in FigureOperationState.line(
        label="plain", source="profile"
    ).model_dump(mode="json")
    restored_operation = FigureOperationState.model_validate_json(
        loaded_operation.model_dump_json()
    )
    assert restored_operation.map_selections == selections

    restored_tool = erlab.interactive.utils.ToolWindow.from_dataset(tool.to_dataset())
    assert isinstance(restored_tool, FigureComposerTool)
    qtbot.addWidget(restored_tool)
    assert restored_tool.tool_status.operations[0].map_selections == selections
    restored_profiles = figurecomposer_line_profile._line_data_items(
        restored_tool, restored_tool.tool_status.operations[0]
    )
    xr.testing.assert_identical(restored_profiles[0], data.qsel(cut=0.0))
    xr.testing.assert_identical(restored_profiles[1], data.qsel(cut=2.0))


def test_imagetool_line_profile_seeds_public_nonuniform_coordinate(qtbot) -> None:
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
    source_tool = erlab.interactive.itool(public, manager=False, execute=False)
    assert isinstance(source_tool, erlab.interactive.imagetool.ImageTool)
    qtbot.addWidget(source_tool)

    operation = figurecomposer_adapter.build_figure_composer_operation(
        source_tool.slicer_area.profiles[0], source_name="data"
    )

    assert operation.kind == FigureOperationKind.LINE
    assert operation.line_x == "sample_temp"
    assert "sample_temp_idx" not in operation.model_dump_json()

    composer = FigureComposerTool.from_sources(
        {"data": source_tool.slicer_area._tool_source_parent_data()},
        sources=(FigureSourceState(name="data", label="map"),),
        operations=(operation,),
        setup=FigureSubplotsState(),
        primary_source="data",
    )
    qtbot.addWidget(composer)

    figure = plt.figure()
    try:
        figurecomposer_rendering._render_into_figure(
            composer, figure, sync_visible=False
        )
        assert composer._operation_render_errors == {}
        assert any(axis.lines for axis in figure.axes)
    finally:
        plt.close(figure)


def test_figure_composer_line_profile_uses_public_nonuniform_dims(qtbot) -> None:
    public = xr.DataArray(
        np.arange(8.0).reshape(4, 2),
        dims=("sample_temp", "alpha"),
        coords={
            "sample_temp": [10.0, 15.0, 30.0, 60.0],
            "alpha": [0.0, 1.0],
        },
        name="profile",
    )
    internal = erlab.utils.array._make_dims_uniform(public)
    operation = FigureOperationState.line(
        label="line",
        source="data",
    ).model_copy(update={"line_selection": {"sample_temp": 15.0}, "line_x": "alpha"})

    tool = FigureComposerTool.from_sources(
        {"data": internal},
        sources=(FigureSourceState(name="data", label="profile"),),
        operations=(operation,),
        setup=FigureSubplotsState(),
        primary_source="data",
    )
    qtbot.addWidget(tool)

    coordinate_names = figurecomposer_line_profile._available_line_coordinate_names(
        tool._document, operation
    )
    assert "alpha" in coordinate_names
    assert "sample_temp_idx" not in coordinate_names
    line_items = figurecomposer_line_profile._line_data_items(tool, operation)
    assert len(line_items) == 1
    assert line_items[0].dims == ("alpha",)


def test_figure_composer_line_offset_coords_empty_without_iter_dim() -> None:
    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("cut", "kx"),
        coords={"cut": [0.0, 1.0], "kx": [0.0, 0.5, 1.0]},
        name="profiles",
    )
    operation = FigureOperationState.line(label="line", source="profiles")

    assert figurecomposer_line_profile._line_offset_coord_names(data, operation) == []


def test_figure_composer_line_profile_operation_uses_semantic_sections(
    qtbot,
) -> None:
    data = _figure_composer_line_slice_source("profile_data")
    operation = FigureOperationState.line(
        label="profiles",
        source="profile_data",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
    ).model_copy(
        update={
            "line_x": "kx",
            "line_iter_dim": "eV",
            "line_selection": {"eV": slice(-0.5, 0.5)},
            "line_reduce": "both",
            "line_labels": ("left", "right"),
            "line_colors": ("C0", "C1"),
            "line_kw": {"linestyle": "--", "marker": "o"},
            "line_normalize": "max",
            "line_scales": (2.0,),
            "line_offsets": (0.0, 1.0),
            "xlim": (-0.5, 0.5),
            "ylim": (0.0, None),
        }
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="profile_data", label="profile_data"),),
            operations=(operation,),
            primary_source="profile_data",
        ),
    )
    qtbot.addWidget(tool)

    assert tool.operation_editor.section_keys == (
        "sources",
        "axes",
        "selection",
        "view",
        "style",
        "other",
    )
    assert (
        _operation_section_button(tool, "other").property("section_title")
        == "Transform"
    )
    assert [
        tool.operation_editor.stack.widget(index).objectName()
        for index in range(tool.operation_editor.stack.count())
    ] == [
        "figureComposerStepSourcesPage",
        "figureComposerTargetAxesPage",
        "figureComposerLineSelectionPage",
        "figureComposerLineViewPage",
        "figureComposerLineStylePage",
        "figureComposerLineOtherPage",
    ]

    selection_page = tool.findChild(
        QtWidgets.QWidget, "figureComposerLineSelectionPage"
    )
    view_page = tool.findChild(QtWidgets.QWidget, "figureComposerLineViewPage")
    style_page = tool.findChild(QtWidgets.QWidget, "figureComposerLineStylePage")
    other_page = tool.findChild(QtWidgets.QWidget, "figureComposerLineOtherPage")
    assert selection_page is not None
    assert view_page is not None
    assert style_page is not None
    assert other_page is not None

    section_controls = (
        (
            selection_page,
            (
                "figureComposerProfileCoordinateCombo",
                "figureComposerProfileValuesCombo",
                "figureComposerLineSelectionEdit",
                "figureComposerProfileIterDimCombo",
                "figureComposerProfileReduceCombo",
            ),
        ),
        (
            view_page,
            (
                "figureComposerProfilePlacementCombo",
                "figureComposerDataValuesAxisCombo",
                "figureComposerLineXLimEdit",
                "figureComposerLineYLimEdit",
            ),
        ),
        (
            style_page,
            (
                "figureComposerLineLabelsEdit",
                "figureComposerLineColorModeCombo",
                "figureComposerLineColorsEdit",
                "figureComposerLineStyleCombo",
                "figureComposerLineWidthSpin",
                "figureComposerLineMarkerCombo",
                "figureComposerLineMarkerSizeSpin",
                "figureComposerLineMarkerFaceColorEdit",
                "figureComposerLineMarkerEdgeColorEdit",
                "figureComposerLineGradientCheck",
            ),
        ),
        (
            other_page,
            (
                "figureComposerLineNormalizeCombo",
                "figureComposerLineScalesEdit",
                "figureComposerLineOffsetSourceCombo",
                "figureComposerLineOffsetsEdit",
            ),
        ),
    )
    pages = (selection_page, view_page, style_page, other_page)
    for expected_page, control_names in section_controls:
        for name in control_names:
            widget = tool.findChild(QtWidgets.QWidget, name)
            assert widget is not None
            for page in pages:
                if page is expected_page:
                    assert page.isAncestorOf(widget)
                else:
                    assert not page.isAncestorOf(widget)

    for key, page in (
        ("selection", selection_page),
        ("view", view_page),
        ("style", style_page),
        ("other", other_page),
    ):
        tool.operation_editor.select_section(key)
        assert tool.operation_editor.stack.currentWidget() is page
        assert tool.tool_status.operations[0] == operation


def test_figure_composer_line_values_axis_swaps_regular_profile(qtbot) -> None:
    profile = xr.DataArray(
        np.array([2.0, 4.0, 8.0]),
        dims=("kx",),
        coords={"kx": [-1.0, 0.0, 1.0]},
        name="profile",
    )
    operation = FigureOperationState.line(
        label="profile",
        source="profile",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
    ).model_copy(update={"line_x": "kx", "line_values_axis": "x", "gradient": True})
    tool = FigureComposerTool(
        profile,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(),
            sources=(FigureSourceState(name="profile", label="profile"),),
            operations=(operation,),
            primary_source="profile",
        ),
    )
    qtbot.addWidget(tool)

    fig = tool.figure
    figurecomposer_rendering._render_into_figure(tool, fig, sync_visible=False)
    line = fig.axes[0].lines[0]
    np.testing.assert_allclose(line.get_xdata(), profile.values)
    np.testing.assert_allclose(line.get_ydata(), profile["kx"].values)
    assert len(fig.axes[0].images) == 1
    np.testing.assert_allclose(
        fig.axes[0].images[0].cmap(1.0),
        mcolors.to_rgba(line.get_color()),
        rtol=1e-7,
    )

    code = tool.generated_code()
    assert "import erlab.plotting as eplt" in code
    assert "_line = axs[0, 0].plot(profile, profile['kx'])[0]" in code
    assert "transpose=True" in code
    code_lines = code.splitlines()
    plot_index = code_lines.index(
        "    _line = axs[0, 0].plot(profile, profile['kx'])[0]"
    )
    assert code_lines[plot_index + 1] == "    eplt.gradient_fill("

    namespace: dict[str, typing.Any] = {"profile": profile}
    exec(code, namespace)  # noqa: S102
    line = namespace["fig"].axes[0].lines[0]
    np.testing.assert_allclose(line.get_xdata(), profile.values)
    np.testing.assert_allclose(line.get_ydata(), profile["kx"].values)
    assert len(namespace["fig"].axes[0].images) == 1
    np.testing.assert_allclose(
        namespace["fig"].axes[0].images[0].cmap(1.0),
        mcolors.to_rgba(line.get_color()),
        rtol=1e-7,
    )


def test_figure_composer_line_profile_helper_contracts(qtbot) -> None:
    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("cut", "kx"),
        coords={
            "cut": [0.0, 1.0],
            "kx": [-1.0, 0.0, 1.0],
            "temperature": ("cut", [20.0, 30.0]),
            "signal": (("cut", "kx"), np.arange(6.0).reshape(2, 3) + 10.0),
        },
        name="profile",
    )
    operation = FigureOperationState.line(
        label="profile",
        source="profile",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
    ).model_copy(
        update={
            "line_x": "kx",
            "line_y": "signal",
            "line_iter_dim": "cut",
            "line_selection": {"kx": slice(-1.0, 1.0)},
            "line_values_axis": "x",
            "line_labels": ("a", "b"),
            "line_colors": ("red", "blue"),
            "line_kw": {"lw": 2.0, "c": "black", "marker": "o"},
            "xlim": (-2.0, 2.0),
            "ylim": (0.0, 20.0),
        }
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(),
            sources=(FigureSourceState(name="profile", label="profile"),),
            operations=(operation,),
            primary_source="profile",
        ),
    )
    qtbot.addWidget(tool)
    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )

    figurecomposer_line_profile._line_limit_update_callback(
        tool.operation_editor, "xlim"
    )("0, 1")
    assert tool.tool_status.operations[0].xlim == (0.0, 1.0)
    figurecomposer_line_profile._line_limit_update_callback(
        tool.operation_editor, "ylim"
    )("")
    assert tool.tool_status.operations[0].ylim is None

    assert (
        figurecomposer_line_profile._line_placement_text("one_per_axis")
        == "One profile per axis"
    )
    assert (
        figurecomposer_line_profile._line_placement_from_text("One profile per axis")
        == "one_per_axis"
    )
    assert (
        figurecomposer_line_profile._line_placement_from_text("anything else")
        == "all_axes"
    )
    assert (
        figurecomposer_line_profile._line_choice_data(
            tool._document,
            operation.model_copy(update={"line_source": None}),
            values=False,
        )
        is None
    )
    assert (
        figurecomposer_line_profile._available_line_value_names(
            tool._document, operation.model_copy(update={"line_source": "missing"})
        )
        == []
    )
    assert set(
        figurecomposer_line_profile._available_line_value_names(
            tool._document, operation
        )
    ) >= {"cut", "kx", "temperature", "signal"}
    assert figurecomposer_line_profile._available_line_coordinate_names(
        tool._document, operation
    ) == ["kx"]
    assert figurecomposer_line_profile._available_line_offset_coords(
        tool._document, operation
    ) == ["temperature"]

    profiles = figurecomposer_line_profile._line_data_items(tool, operation)
    assert len(profiles) == 2
    assert all(profile.dims == ("kx",) for profile in profiles)
    selected_operation = operation.model_copy(
        update={
            "map_selections": (
                FigureDataSelectionState(source="profile", qsel={"cut": 0.0}),
            ),
            "line_labels": (),
        }
    )
    assert (
        len(figurecomposer_line_profile._line_data_items(tool, selected_operation)) == 2
    )
    assert (
        figurecomposer_line_profile._line_data_items(
            tool, operation.model_copy(update={"line_source": None})
        )
        == []
    )
    assert (
        figurecomposer_line_profile._line_data_items(
            tool, operation.model_copy(update={"line_source": "missing"})
        )
        == []
    )
    with pytest.raises(ValueError, match="one-dimensional"):
        figurecomposer_line_profile._line_coordinate(data, None)

    assert figurecomposer_line_profile._line_text_values((), 0, default=None) == ()
    assert figurecomposer_line_profile._line_text_values(
        ("shared",), 2, default=None
    ) == (
        "shared",
        "shared",
    )
    with pytest.raises(ValueError, match="one value or one per profile"):
        figurecomposer_line_profile._line_text_values(("a", "b", "c"), 2, default=None)
    assert figurecomposer_line_profile._line_profile_style_kwargs(operation) == {
        "linewidth": 2.0,
        "marker": "o",
    }

    loop_names = ["profile"]
    loop_values = ["profiles"]
    style_lines, kwargs_text = figurecomposer_line_profile._line_style_code(
        operation,
        profiles=profiles,
        sources=("profile", "profile"),
        loop_names=loop_names,
        loop_values=loop_values,
    )
    assert loop_names == ["profile", "label", "color"]
    assert loop_values == ["profiles", "['a', 'b']", "['red', 'blue']"]
    assert style_lines == []
    assert "linewidth=2.0" in kwargs_text
    assert "label=label" in kwargs_text
    assert "color=color" in kwargs_text

    assert (
        figurecomposer_line_profile._line_code(
            tool, operation.model_copy(update={"line_source": None})
        )
        == []
    )
    selection_lines = figurecomposer_line_profile._line_code(tool, selected_operation)
    assert not any(line == "profiles = [" for line in selection_lines)
    assert any(".qsel(kx=slice(-1.0, 1.0))" in line for line in selection_lines)
    one_per_axis_lines = figurecomposer_line_profile._line_code(
        tool, operation.model_copy(update={"line_placement": "one_per_axis"})
    )
    assert not any("if len(target_axes)" in line for line in one_per_axis_lines)
    assert not any(line.startswith("target_axes =") for line in one_per_axis_lines)
    assert any(
        "axs[0, 0].plot(profile, profile['kx']" in line for line in one_per_axis_lines
    )
    assert one_per_axis_lines[-1] == "axs[0, 0].set(xlim=(-2.0, 2.0), ylim=(0.0, 20.0))"


def test_figure_composer_profile_lines_support_per_profile_style_and_offsets(
    qtbot,
) -> None:
    profile_data = xr.DataArray(
        np.arange(12.0).reshape(3, 4),
        dims=("cut", "kx"),
        coords={
            "cut": [0.0, 1.0, 2.0],
            "kx": [-1.0, 0.0, 1.0, 2.0],
            "temperature": ("cut", [10.0, 20.0, 30.0]),
        },
        name="profile_data",
    )
    operation = FigureOperationState.line(
        label="profiles",
        source="profile_data",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
    ).model_copy(
        update={
            "line_x": "kx",
            "line_iter_dim": "cut",
            "line_labels": ("a", "b", "c"),
            "line_colors": ("red", "green", "blue"),
            "line_kw": {
                "linestyle": "--",
                "linewidth": 1.5,
                "marker": "o",
                "markersize": 6.0,
                "markerfacecolor": "yellow",
                "markeredgecolor": "black",
            },
            "line_offset_source": "associated",
            "line_offset_coord": "temperature",
            "line_offset_scale": 0.01,
            "gradient": True,
        }
    )
    tool = FigureComposerTool(
        profile_data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(),
            sources=(FigureSourceState(name="profile_data", label="profile_data"),),
            operations=(operation,),
            primary_source="profile_data",
        ),
    )
    qtbot.addWidget(tool)
    profiles = figurecomposer_line_profile._line_data_items(tool, operation)

    tool.operation_editor.select_section("selection")
    selection_page = tool.operation_editor.stack.currentWidget()
    reduce_combo = selection_page.findChild(
        QtWidgets.QComboBox, "figureComposerProfileReduceCombo"
    )
    assert reduce_combo is not None
    assert (
        selection_page.findChild(
            QtWidgets.QSpinBox, "figureComposerProfileReduceCoarsenSpin"
        )
        is None
    )
    _activate_combo_text(reduce_combo, "Both")
    qtbot.waitUntil(
        lambda: (
            tool.operation_editor.stack.currentWidget().findChild(
                QtWidgets.QSpinBox, "figureComposerProfileReduceCoarsenSpin"
            )
            is not None
        ),
        timeout=1000,
    )
    selection_page = tool.operation_editor.stack.currentWidget()
    coarsen_spin = selection_page.findChild(
        QtWidgets.QSpinBox, "figureComposerProfileReduceCoarsenSpin"
    )
    thin_spin = selection_page.findChild(
        QtWidgets.QSpinBox, "figureComposerProfileReduceThinSpin"
    )
    assert tool.tool_status.operations[0].line_reduce == "both"
    assert coarsen_spin is not None
    assert coarsen_spin.value() == 2
    assert thin_spin is not None
    thin_spin.setValue(3)
    assert tool.tool_status.operations[0].line_reduce_thin == 3
    tool._replace_operation(0, operation)
    tool.operation_editor.select_section("other")
    other_page = tool.operation_editor.stack.currentWidget()
    offset_source_combo = other_page.findChild(
        QtWidgets.QComboBox, "figureComposerLineOffsetSourceCombo"
    )
    assert offset_source_combo is not None
    assert (
        other_page.findChild(
            QtWidgets.QComboBox, "figureComposerLineOffsetCoordinateCombo"
        )
        is not None
    )
    assert (
        other_page.findChild(
            QtWidgets.QDoubleSpinBox, "figureComposerLineOffsetScaleEdit"
        )
        is not None
    )
    assert (
        other_page.findChild(QtWidgets.QLineEdit, "figureComposerLineOffsetsEdit")
        is None
    )
    tool.operation_editor.select_section("style")
    style_page = tool.operation_editor.stack.currentWidget()
    line_style_combo = style_page.findChild(
        QtWidgets.QComboBox, "figureComposerLineStyleCombo"
    )
    line_width_spin = style_page.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerLineWidthSpin"
    )
    marker_combo = style_page.findChild(
        QtWidgets.QComboBox, "figureComposerLineMarkerCombo"
    )
    marker_size_spin = style_page.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerLineMarkerSizeSpin"
    )
    marker_face_edit = style_page.findChild(
        QtWidgets.QLineEdit, "figureComposerLineMarkerFaceColorEdit"
    )
    marker_edge_edit = style_page.findChild(
        QtWidgets.QLineEdit, "figureComposerLineMarkerEdgeColorEdit"
    )
    gradient_check = style_page.findChild(
        QtWidgets.QCheckBox, "figureComposerLineGradientCheck"
    )
    marker_face_button = style_page.findChild(
        color_widgets._ColorPickerButton,
        "figureComposerLineMarkerFaceColorButton",
    )
    assert line_style_combo is not None
    assert line_width_spin is not None
    assert line_width_spin.value() == 1.5
    assert marker_combo is not None
    assert marker_size_spin is not None
    assert marker_size_spin.value() == 6.0
    assert marker_face_edit is not None
    assert marker_face_edit.text() == "yellow"
    assert marker_face_button is not None
    assert marker_edge_edit is not None
    assert marker_edge_edit.text() == "black"
    assert gradient_check is not None
    assert gradient_check.isChecked()

    color_text_edit = style_page.findChild(
        QtWidgets.QLineEdit, "figureComposerLineColorsEdit"
    )
    first_color_edit = style_page.findChild(
        QtWidgets.QLineEdit, "figureComposerLineColorItemEdit_0"
    )
    assert color_text_edit is not None
    assert color_text_edit.text() == "red, green, blue"
    assert first_color_edit is not None
    first_color_edit.setText("tab:blue")
    first_color_edit.setModified(True)
    first_color_edit.editingFinished.emit()
    assert tool.tool_status.operations[0].line_colors == (
        "tab:blue",
        "green",
        "blue",
    )
    assert color_text_edit.text() == "tab:blue, green, blue"
    color_text_edit.setText("C0, C1, C2")
    color_text_edit.setModified(True)
    color_text_edit.editingFinished.emit()
    assert tool.tool_status.operations[0].line_colors == ("C0", "C1", "C2")

    operation = tool.tool_status.operations[0]

    tool.operation_editor.select_section("other")
    other_page = tool.operation_editor.stack.currentWidget()
    offset_source_combo = other_page.findChild(
        QtWidgets.QComboBox, "figureComposerLineOffsetSourceCombo"
    )
    assert offset_source_combo is not None
    _activate_combo_text(offset_source_combo, "manual")
    qtbot.waitUntil(
        lambda: (
            tool.operation_editor.stack.currentWidget().findChild(
                QtWidgets.QComboBox, "figureComposerLineOffsetCoordinateCombo"
            )
            is None
        ),
        timeout=1000,
    )
    other_page = tool.operation_editor.stack.currentWidget()
    assert tool.tool_status.operations[0].line_offset_source == "manual"
    assert tool.tool_status.operations[0].line_offset_scale == 1.0
    assert (
        other_page.findChild(
            QtWidgets.QComboBox, "figureComposerLineOffsetCoordinateCombo"
        )
        is None
    )
    assert (
        other_page.findChild(
            QtWidgets.QDoubleSpinBox, "figureComposerLineOffsetScaleEdit"
        )
        is None
    )
    assert (
        other_page.findChild(QtWidgets.QLineEdit, "figureComposerLineOffsetsEdit")
        is not None
    )

    offset_source_combo = other_page.findChild(
        QtWidgets.QComboBox, "figureComposerLineOffsetSourceCombo"
    )
    assert offset_source_combo is not None
    _activate_combo_text(offset_source_combo, "index")
    qtbot.waitUntil(
        lambda: (
            tool.operation_editor.stack.currentWidget().findChild(
                QtWidgets.QDoubleSpinBox, "figureComposerLineOffsetScaleEdit"
            )
            is not None
        ),
        timeout=1000,
    )
    other_page = tool.operation_editor.stack.currentWidget()
    assert tool.tool_status.operations[0].line_offset_source == "index"
    assert (
        other_page.findChild(
            QtWidgets.QComboBox, "figureComposerLineOffsetCoordinateCombo"
        )
        is None
    )
    assert (
        other_page.findChild(
            QtWidgets.QDoubleSpinBox, "figureComposerLineOffsetScaleEdit"
        )
        is not None
    )
    assert (
        other_page.findChild(QtWidgets.QLineEdit, "figureComposerLineOffsetsEdit")
        is None
    )
    tool._replace_operation(0, operation)

    assert figurecomposer_line_profile._available_line_offset_coords(
        tool._document, operation
    ) == ["temperature"]
    assert _line_transform.line_offsets_for_profiles(
        operation.model_copy(
            update={"line_offset_source": "index", "line_offset_scale": 2.0}
        ),
        profiles,
    ) == (0.0, 2.0, 4.0)
    assert _line_transform.line_offsets_for_profiles(
        operation.model_copy(
            update={"line_offset_source": "coordinate", "line_offset_scale": 0.5}
        ),
        profiles,
    ) == (0.0, 0.5, 1.0)
    assert _line_transform.line_offsets_for_profiles(operation, profiles) == (
        0.1,
        0.2,
        0.3,
    )

    fig = tool.figure
    figurecomposer_rendering._render_into_figure(tool, fig, sync_visible=False)
    assert fig.axes[0].get_legend() is None
    assert len(fig.axes[0].images) == 3
    for index, line in enumerate(fig.axes[0].lines):
        np.testing.assert_allclose(line.get_xdata(), profile_data["kx"].values)
        np.testing.assert_allclose(
            line.get_ydata(), profile_data.isel(cut=index).values + 0.1 * (index + 1)
        )
        assert line.get_label() == ("a", "b", "c")[index]
        assert line.get_color() == ("C0", "C1", "C2")[index]
        assert line.get_linestyle() == "--"
        assert line.get_linewidth() == 1.5
        assert line.get_marker() == "o"
        assert line.get_markersize() == 6.0
        assert line.get_markerfacecolor() == "yellow"
        assert line.get_markeredgecolor() == "black"
        np.testing.assert_allclose(
            fig.axes[0].images[index].cmap(1.0),
            mcolors.to_rgba(line.get_color()),
            rtol=1e-7,
        )

    namespace: dict[str, typing.Any] = {"profile_data": profile_data}
    code = tool.generated_code()
    assert "import erlab.plotting as eplt" in code
    assert "ax.legend()" not in code
    assert "profile_offsets =" not in code
    assert "0.01 * profile_data['temperature'] + profile_data" in code
    assert "for ax in" not in code
    assert "_line = axs[0, 0].plot(profile['kx'], profile" in code
    assert "for _line in" not in code
    assert "ax.lines" not in code
    code_lines = code.splitlines()
    plot_index = next(
        index
        for index, line in enumerate(code_lines)
        if "_line = axs[0, 0].plot(" in line
    )
    assert code_lines[plot_index + 1] == "    eplt.gradient_fill("
    exec(code, namespace)  # noqa: S102
    assert len(namespace["fig"].axes[0].images) == 3
    for index, line in enumerate(namespace["fig"].axes[0].lines):
        np.testing.assert_allclose(line.get_xdata(), profile_data["kx"].values)
        np.testing.assert_allclose(
            line.get_ydata(), profile_data.isel(cut=index).values + 0.1 * (index + 1)
        )
        assert line.get_label() == ("a", "b", "c")[index]
        assert line.get_color() == ("C0", "C1", "C2")[index]
        assert line.get_linestyle() == "--"
        assert line.get_linewidth() == 1.5
        assert line.get_marker() == "o"
        assert line.get_markersize() == 6.0
        assert line.get_markerfacecolor() == "yellow"
        assert line.get_markeredgecolor() == "black"
        np.testing.assert_allclose(
            namespace["fig"].axes[0].images[index].cmap(1.0),
            mcolors.to_rgba(line.get_color()),
            rtol=1e-7,
        )

    shared_label_operation = operation.model_copy(
        update={"line_labels": ("shared",), "line_colors": ()}
    )
    shared_label_tool = FigureComposerTool(
        profile_data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(),
            sources=(FigureSourceState(name="profile_data", label="profile_data"),),
            operations=(shared_label_operation,),
            primary_source="profile_data",
        ),
    )
    qtbot.addWidget(shared_label_tool)

    shared_fig = shared_label_tool.figure
    figurecomposer_rendering._render_into_figure(
        shared_label_tool, shared_fig, sync_visible=False
    )
    assert [line.get_label() for line in shared_fig.axes[0].lines] == [
        "shared",
        "shared",
        "shared",
    ]

    namespace = {"profile_data": profile_data}
    shared_code = shared_label_tool.generated_code()
    assert "profile_label =" not in shared_code
    exec(shared_code, namespace)  # noqa: S102
    assert [line.get_label() for line in namespace["fig"].axes[0].lines] == [
        "shared",
        "shared",
        "shared",
    ]


def test_figure_composer_line_label_placeholders_render_and_codegen(qtbot) -> None:
    eV = np.array([-0.1, 0.2])
    kx = np.array([-1.0, 0.0, 1.0])
    data = xr.DataArray(
        np.arange(eV.size * kx.size, dtype=float).reshape(eV.size, kx.size),
        dims=("eV", "kx"),
        coords={"eV": eV, "kx": kx},
        name="data",
    )
    operation = FigureOperationState.line(
        label="profiles",
        source="data",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
    ).model_copy(
        update={
            "line_x": "kx",
            "line_iter_dim": "eV",
            "line_label_text": r"$k_{F}$, $E-E_F = {eV:g}$ eV",
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

    expected_labels = [
        r"$k_{F}$, $E-E_F = -0.1$ eV",
        r"$k_{F}$, $E-E_F = 0.2$ eV",
    ]
    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    assert [line.get_label() for line in tool.figure.axes[0].lines] == expected_labels

    code = tool.generated_code()
    assert expected_labels[0] not in code
    assert expected_labels[1] not in code
    assert "profile.coords['eV'].values.item()" in code
    assert "profile_labels" not in code
    namespace: dict[str, typing.Any] = {"data": data}
    exec(code, namespace)  # noqa: S102
    assert [
        line.get_label() for line in namespace["fig"].axes[0].lines
    ] == expected_labels


def test_figure_composer_line_label_placeholders_accept_spaced_names(
    qtbot,
) -> None:
    sample_temp = np.array([20.0, 30.0])
    kx = np.array([-1.0, 0.0, 1.0])
    data = xr.DataArray(
        np.arange(sample_temp.size * kx.size, dtype=float).reshape(
            sample_temp.size, kx.size
        ),
        dims=("sample temp", "kx"),
        coords={"sample temp": sample_temp, "kx": kx},
        attrs={"sample label": "annealed"},
        name="data",
    )
    operation = FigureOperationState.line(
        label="profiles",
        source="data",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
    ).model_copy(
        update={
            "line_x": "kx",
            "line_iter_dim": "sample temp",
            "line_label_text": "{sample_temp + 1.5:g} K, {sample_label}",
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

    expected_labels = ["21.5 K, annealed", "31.5 K, annealed"]
    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    assert [line.get_label() for line in tool.figure.axes[0].lines] == expected_labels

    code = tool.generated_code()
    assert expected_labels[0] not in code
    assert expected_labels[1] not in code
    assert "profile.coords['sample temp'].values.item() + 1.5" in code
    assert "profile.attrs['sample label']" in code
    namespace: dict[str, typing.Any] = {"data": data}
    exec(code, namespace)  # noqa: S102
    assert [
        line.get_label() for line in namespace["fig"].axes[0].lines
    ] == expected_labels


def test_figure_composer_line_label_missing_placeholder_errors(qtbot) -> None:
    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("eV", "kx"),
        coords={"eV": [-0.1, 0.2], "kx": [-1.0, 0.0, 1.0]},
        name="data",
    )
    operation = FigureOperationState.line(
        label="profiles",
        source="data",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
    ).model_copy(
        update={
            "line_x": "kx",
            "line_iter_dim": "eV",
            "line_label_text": "eV={eV:g}, temperature={temperature:g}",
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

    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    render_error = tool._operation_render_errors[operation.operation_id]
    assert "temperature" in render_error
    assert "Available placeholders" not in render_error
    with pytest.raises(ValueError, match="temperature") as exc_info:
        tool.generated_code()
    codegen_error = str(exc_info.value)
    assert "temperature" in codegen_error
    assert "Available placeholders" not in codegen_error


def test_figure_composer_line_profile_coordinate_colormap_render_and_codegen(
    qtbot,
) -> None:
    eV = np.array([-0.1, 0.2])
    kx = np.array([-1.0, 0.0, 1.0])
    data = xr.DataArray(
        np.arange(eV.size * kx.size, dtype=float).reshape(eV.size, kx.size),
        dims=("eV", "kx"),
        coords={"eV": eV, "kx": kx},
        name="data",
    )
    operation = FigureOperationState.line(
        label="profiles",
        source="data",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
    ).model_copy(
        update={
            "line_x": "kx",
            "line_iter_dim": "eV",
            "line_color_mode": "coordinate",
            "line_color_cmap": "plasma",
            "line_color_cmap_trim_lower": 0.1,
            "line_color_cmap_trim_upper": 0.2,
            "line_colors": ("black",),
            "line_kw": {"color": "red", "linestyle": "--"},
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

    expected_colors = _expected_line_colormap_colors(eV, "plasma", trim=(0.1, 0.2))
    assert figurecomposer_line_profile._available_line_color_coords(
        tool._document, tool._source_display_name, operation
    ) == ["eV"]
    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    rendered_lines = tool.figure.axes[0].lines
    assert [line.get_linestyle() for line in rendered_lines] == ["--", "--"]
    np.testing.assert_allclose(
        np.asarray([mcolors.to_rgba(line.get_color()) for line in rendered_lines]),
        expected_colors,
    )

    tool.operation_editor.select_section("style")
    style_page = tool.operation_editor.stack.currentWidget()
    assert (
        style_page.findChild(QtWidgets.QComboBox, "figureComposerLineColorModeCombo")
        is not None
    )
    assert (
        style_page.findChild(QtWidgets.QComboBox, "figureComposerLineColorCoordCombo")
        is not None
    )
    assert (
        style_page.findChild(
            erlab.interactive.colors.ColorMapComboBox,
            "figureComposerLineColorCmapCombo",
        )
        is not None
    )
    trim_lower_spin = style_page.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerLineColorCmapTrimLowerSpin"
    )
    trim_upper_spin = style_page.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerLineColorCmapTrimUpperSpin"
    )
    assert trim_lower_spin is not None
    assert trim_upper_spin is not None
    assert trim_lower_spin.value() == pytest.approx(0.1)
    assert trim_upper_spin.value() == pytest.approx(0.2)
    assert not trim_lower_spin.keyboardTracking()
    assert not trim_upper_spin.keyboardTracking()
    assert trim_lower_spin.toolTip()
    assert trim_upper_spin.toolTip()
    assert (
        style_page.findChild(QtWidgets.QLineEdit, "figureComposerLineColorsEdit")
        is None
    )

    code = tool.generated_code()
    assert "import matplotlib.colors as mcolors" in code
    assert "line_color_values =" in code
    assert "line_colors = plt.get_cmap('plasma')(" in code
    assert "0.1 + 0.7 * line_color_values_norm(line_color_values)" in code
    assert "black" not in code
    namespace: dict[str, typing.Any] = {"data": data}
    exec(code, namespace)  # noqa: S102
    generated_lines = namespace["fig"].axes[0].lines
    np.testing.assert_allclose(
        np.asarray([mcolors.to_rgba(line.get_color()) for line in generated_lines]),
        expected_colors,
    )
    trim_lower_spin.setValue(0.25)
    trim_upper_spin.setValue(0.3)
    assert tool.tool_status.operations[0].line_color_cmap_trim_lower == pytest.approx(
        0.25
    )
    assert tool.tool_status.operations[0].line_color_cmap_trim_upper == pytest.approx(
        0.3
    )


def test_figure_composer_line_colormap_equal_values_use_midpoint() -> None:
    colors = figurecomposer_line_colormap.colors_from_values(
        (2.0, 2.0), "viridis", trim=(0.2, 0.3)
    )
    expected = plt.get_cmap("viridis")([0.45, 0.45])
    np.testing.assert_allclose(colors, expected)


def test_figure_composer_line_colormap_normalizes_colorcet_name() -> None:
    pytest.importorskip("colorcet")

    erlab.interactive.colors.load_all_colormaps()
    operation = FigureOperationState.line(label="profiles", source="data").model_copy(
        update={
            "line_color_cmap": "fire",
            "line_color_cmap_reverse": True,
        }
    )

    assert figurecomposer_line_colormap.effective_line_color_cmap(operation) == (
        "cet_fire_r"
    )


def test_figure_composer_line_colormap_rejects_invalid_trim() -> None:
    with pytest.raises(ValueError, match="trim"):
        figurecomposer_line_colormap.colors_from_values(
            (1.0,), "viridis", trim=(0.5, 0.5)
        )
    with pytest.raises(ValueError, match="trim"):
        figurecomposer_line_colormap.colormap_code_lines(
            "[1.0]", "viridis", trim=(-0.1, 0.0)
        )
    operation = FigureOperationState.line(label="profiles", source="data").model_copy(
        update={"line_color_cmap_trim_lower": 0.5, "line_color_cmap_trim_upper": 0.5}
    )
    assert figurecomposer_line_colormap.line_color_cmap_trim_control_values(
        operation
    ) == (0.0, 0.0)


def test_figure_composer_line_colormap_helper_edges() -> None:
    contexts = (
        {"numeric": 1.0, "text": "a", "nan": np.nan, "index": 0},
        {"numeric": 2.0, "text": "b", "nan": 1.0, "index": 1},
    )
    assert figurecomposer_line_colormap.numeric_context_field_names(contexts) == (
        "numeric",
    )
    assert figurecomposer_line_colormap.values_from_contexts(
        contexts, "numeric", item_name="line"
    ) == (1.0, 2.0)
    with pytest.raises(ValueError, match="Choose a coordinate"):
        figurecomposer_line_colormap.values_from_contexts(
            contexts, None, item_name="line"
        )
    with pytest.raises(ValueError, match="missing"):
        figurecomposer_line_colormap.values_from_contexts(
            contexts, "missing", item_name="line"
        )
    with pytest.raises(ValueError, match="numeric scalars"):
        figurecomposer_line_colormap.values_from_contexts(
            contexts, "text", item_name="line"
        )
    with pytest.raises(ValueError, match="finite numeric scalars"):
        figurecomposer_line_colormap.values_from_contexts(
            contexts, "nan", item_name="line"
        )
    assert figurecomposer_line_colormap.colors_from_values((), "viridis") == ()
    assert figurecomposer_line_colormap.colormap_code_lines("[0.0, 1.0]", "viridis")[
        -2:
    ] == ["else:", "    line_colors = []"]


def test_figure_composer_line_color_mode_widget_updates_state(qtbot) -> None:
    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("eV", "kx"),
        coords={"eV": [0.1, 0.2], "kx": [-1.0, 0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(data)
    qtbot.addWidget(tool)
    operation = FigureOperationState.line(
        label="profiles",
        source="data",
    ).model_copy(
        update={
            "line_iter_dim": "eV",
            "line_color_mode": "manual",
            "line_color_coord": "missing",
            "line_color_cmap": "viridis_r",
        }
    )
    widget = figurecomposer_toolbar_dialogs._LineColorModeWidget(
        tool, operation, line_kind="profile"
    )
    qtbot.addWidget(widget)
    emitted: list[FigureOperationState] = []
    widget.sigOperationChanged.connect(emitted.append)

    assert widget._mode_from_text("By coordinate") == "manual"
    widget.mode_combo.setItemData(widget.mode_combo.currentIndex(), None)
    assert widget._mode_from_text("By coordinate") == "coordinate"
    widget.mode_combo.setItemData(widget.mode_combo.currentIndex(), "manual")

    widget.mode_combo.setCurrentIndex(widget.mode_combo.findData("coordinate"))
    widget.mode_combo.activated.emit(widget.mode_combo.currentIndex())
    assert emitted[-1].line_color_mode == "coordinate"
    assert emitted[-1].line_color_coord == "missing"
    assert not widget.coordinate_page.isHidden()

    widget.coord_combo.setCurrentIndex(widget.coord_combo.findData("eV"))
    widget.coord_combo.activated.emit(widget.coord_combo.currentIndex())
    assert emitted[-1].line_color_coord == "eV"

    widget.cmap_combo.setCurrentText("plasma")
    widget.cmap_combo.activated.emit(widget.cmap_combo.currentIndex())
    assert emitted[-1].line_color_cmap == "plasma"
    assert emitted[-1].line_color_cmap_reverse is True

    widget.reverse_check.setCheckState(QtCore.Qt.CheckState.Unchecked)
    assert emitted[-1].line_color_cmap_reverse is False
    widget.trim_lower_spin.setValue(0.1)
    widget.trim_upper_spin.setValue(0.2)
    assert emitted[-1].line_color_cmap_trim_lower == pytest.approx(0.1)
    assert emitted[-1].line_color_cmap_trim_upper == pytest.approx(0.2)

    emitted.clear()
    widget._updating = True
    widget._mode_changed(0)
    widget._coord_changed(0)
    widget._cmap_changed(0)
    widget._reverse_changed(QtCore.Qt.CheckState.Checked.value)
    widget._trim_lower_changed(0.3)
    widget._trim_upper_changed(0.4)
    assert emitted == []
    widget._updating = False

    plot_slices_operation = FigureOperationState.plot_slices(
        label="slices",
        sources=("data",),
        slice_dim="eV",
        slice_values=(0.1, 0.2),
    ).model_copy(update={"line_color_mode": "manual"})
    plot_slices_widget = figurecomposer_toolbar_dialogs._LineColorModeWidget(
        tool, plot_slices_operation, line_kind="plot_slices"
    )
    qtbot.addWidget(plot_slices_widget)
    plot_slices_widget.mode_combo.setItemData(
        plot_slices_widget.mode_combo.currentIndex(), None
    )
    assert plot_slices_widget._mode_from_text("By coordinate") == "coordinate"


def test_figure_composer_line_profile_helper_edges(qtbot) -> None:
    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("eV", "kx"),
        coords={
            "eV": [0.1, 0.2],
            "kx": [-1.0, 0.0, 1.0],
            "offset": ("eV", [10.0, 20.0]),
        },
        name="data",
    )
    context = FigureDocument(FigureRecipeState(), source_data={"data": data})
    operation = FigureOperationState.line(label="profiles", source="data").model_copy(
        update={"line_iter_dim": "eV", "line_x": "kx"}
    )

    assert figurecomposer_line_profile._line_reduce_text("unknown") == "Disabled"
    assert figurecomposer_line_profile._line_reduce_from_text("unknown") == "disabled"
    assert figurecomposer_line_profile._line_color_mode_text(
        "unknown"
    ) == figurecomposer_line_profile._line_color_mode_text("manual")
    assert (
        figurecomposer_line_profile._line_color_mode_from_text("By coordinate")
        == "coordinate"
    )
    assert figurecomposer_line_profile._line_color_mode_from_text("Manual") == "manual"
    assert (
        figurecomposer_line_profile._line_selection_sources(
            operation.model_copy(update={"line_source": None})
        )
        == ()
    )
    assert (
        figurecomposer_line_profile._available_line_value_names(
            context, operation.model_copy(update={"line_source": None})
        )
        == []
    )
    assert (
        figurecomposer_line_profile._available_line_value_names(
            context, operation.model_copy(update={"line_source": "missing"})
        )
        == []
    )
    assert set(
        figurecomposer_line_profile._available_line_coordinate_names(context, operation)
    ) >= {"kx"}
    assert figurecomposer_line_profile._available_line_offset_coords(
        context, operation
    ) == ["offset"]
    assert (
        figurecomposer_line_profile._available_line_offset_coords(
            context, operation.model_copy(update={"line_iter_dim": None})
        )
        == []
    )

    cmap_tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(cmap_tool)
    cmap_tool._updating_controls = True
    figurecomposer_line_profile._update_current_line_color_cmap(
        cmap_tool.operation_editor, "viridis", True
    )
    assert cmap_tool.tool_status.operations[0].line_color_cmap is None
    assert not cmap_tool.tool_status.operations[0].line_color_cmap_reverse
    cmap_tool._updating_controls = False
    figurecomposer_line_profile._update_current_line_color_cmap(
        cmap_tool.operation_editor, "viridis_r", True
    )
    assert cmap_tool.tool_status.operations[0].line_color_cmap == "viridis"
    assert cmap_tool.tool_status.operations[0].line_color_cmap_reverse


def test_figure_composer_line_profile_style_codegen_helper_edges() -> None:
    profiles = [
        xr.DataArray(
            [1.0, 2.0],
            dims=("kx",),
            coords={"kx": [0.0, 1.0], "eV": value},
            name="profile",
        )
        for value in (0.1, 0.2)
    ]
    sources = ["first", "second"]
    operation = FigureOperationState.line(label="profiles", source="data").model_copy(
        update={
            "line_iter_dim": "eV",
            "line_label_text": "{source}:{number}:{dim}:{value:g}:{eV:g}",
            "line_color_mode": "coordinate",
            "line_color_cmap": "plasma",
            "line_color_cmap_trim_lower": 0.1,
            "line_color_cmap_trim_upper": 0.2,
        }
    )
    loop_names = ["profile"]
    loop_values = ["profiles"]
    lines, kwargs = figurecomposer_line_profile._line_style_code(
        operation,
        profiles=profiles,
        sources=sources,
        loop_names=loop_names,
        loop_values=loop_values,
    )
    assert loop_names == ["profile", "index", "source", "color"]
    assert loop_values[1] == "range(len(profiles))"
    assert loop_values[2] == "['first', 'second']"
    assert loop_values[3] == "line_colors"
    assert any("line_color_values =" in line for line in lines)
    assert 'label=f"{source}:{index + 1}' in kwargs
    assert "color=color" in kwargs

    with pytest.raises(ValueError, match="Choose a coordinate"):
        figurecomposer_line_profile._line_style_code(
            operation.model_copy(update={"line_iter_dim": None, "line_label_text": ""}),
            profiles=profiles,
            sources=sources,
            loop_names=["profile"],
            loop_values=["profiles"],
        )

    single_color_operation = FigureOperationState.line(
        label="profiles", source="data"
    ).model_copy(update={"line_colors": ("red",)})
    _lines, single_kwargs = figurecomposer_line_profile._line_style_code(
        single_color_operation,
        profiles=profiles,
        sources=sources,
        loop_names=["profile"],
        loop_values=["profiles"],
    )
    assert "color='red'" in single_kwargs

    multi_color_loop_names = ["profile"]
    multi_color_loop_values = ["profiles"]
    _lines, multi_kwargs = figurecomposer_line_profile._line_style_code(
        single_color_operation.model_copy(update={"line_colors": ("red", "blue")}),
        profiles=profiles,
        sources=sources,
        loop_names=multi_color_loop_names,
        loop_values=multi_color_loop_values,
    )
    assert multi_color_loop_names == ["profile", "color"]
    assert multi_color_loop_values[1] == "['red', 'blue']"
    assert "color=color" in multi_kwargs


def test_figure_composer_default_line_labels_use_property_labels() -> None:
    assert (
        figurecomposer_labels.default_label_text(
            "sample_temp", (20.0,), fallback="profile {number}"
        )
        == r"$T = {sample_temp:g}$ K"
    )
    assert (
        figurecomposer_labels.default_label_text(
            "eV", (-0.1,), fallback="profile {number}"
        )
        == r"$E-E_F = {eV:g}$ eV"
    )
    kx_label = figurecomposer_labels.default_label_text(
        "kx", (1.0,), fallback="profile {number}"
    )
    assert figurecomposer_labels.labels_from_text(
        kx_label,
        ({"kx": 1.25, "index": 0, "number": 1},),
    ) == (r"$k_x = 1.25$ Å${}^{-1}$",)
    phase_label = figurecomposer_labels.default_label_text(
        "phase", fallback="profile {number}"
    )
    assert phase_label == r"$phase = {phase}$"
    assert figurecomposer_labels.labels_from_text(
        phase_label,
        ({"phase": "A", "index": 0, "number": 1},),
    ) == (r"$phase = A$",)


def test_figure_composer_line_label_help_button_opens_structured_dialog(
    qtbot, monkeypatch
) -> None:
    icon_names: list[str] = []

    def themed_icon(name: str) -> QtGui.QIcon:
        icon_names.append(name)
        pixmap = QtGui.QPixmap(12, 12)
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        return QtGui.QIcon(pixmap) if name == "help-faq" else QtGui.QIcon()

    monkeypatch.setattr(QtGui.QIcon, "fromTheme", themed_icon)
    data = xr.DataArray(
        [1.0, 2.0, 3.0],
        dims=("kx",),
        coords={"kx": [-1.0, 0.0, 1.0], "sample temp": 20.0},
        attrs={"sample label": "annealed"},
        name="profile",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="profile", label="profile"),),
            operations=(
                FigureOperationState.line(
                    label="profile",
                    source="profile",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ),
            ),
            primary_source="profile",
        ),
    )
    qtbot.addWidget(tool)

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )
    tool.operation_editor.select_section("style")
    editor = tool.operation_editor.stack.currentWidget()
    labels_edit = editor.findChild(QtWidgets.QLineEdit, "figureComposerLineLabelsEdit")
    help_button = editor.findChild(
        QtWidgets.QToolButton, "figureComposerLineLabelsHelpButton"
    )
    assert labels_edit is not None
    assert help_button is not None
    assert "\n" not in labels_edit.toolTip()
    assert "\n" not in help_button.toolTip()
    assert "help-faq" in icon_names
    assert help_button.toolButtonStyle() == (
        QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly
    )
    assert not help_button.icon().isNull()

    help_button.click()
    qtbot.waitUntil(
        lambda: (
            tool.findChild(QtWidgets.QDialog, "figureComposerLegendLabelsHelpDialog")
            is not None
        ),
        timeout=1000,
    )
    dialog = tool.findChild(QtWidgets.QDialog, "figureComposerLegendLabelsHelpDialog")
    assert dialog is not None
    table = dialog.findChild(
        QtWidgets.QTableWidget, "figureComposerLegendLabelsHelpTable"
    )
    examples = dialog.findChildren(
        QtWidgets.QWidget, "figureComposerLegendLabelsHelpExample"
    )
    examples_list = dialog.findChild(
        QtWidgets.QListWidget, "figureComposerLegendLabelsHelpExamples"
    )
    assert table is not None
    assert len(examples) == 4
    assert examples_list is None
    rows = {
        table.item(row, 0).data(QtCore.Qt.ItemDataRole.UserRole): table.item(
            row, 1
        ).data(QtCore.Qt.ItemDataRole.UserRole + 1)
        for row in range(table.rowCount())
    }
    assert rows["sample_temp"] == "coord"
    assert rows["sample_label"] == "attr"
    dialog.close()


def test_figure_composer_plot_slices_line_label_placeholders_codegen(
    qtbot,
) -> None:
    eV = np.array([-0.1, 0.2])
    kx = np.array([-1.0, 0.0, 1.0])
    data = xr.DataArray(
        np.arange(eV.size * kx.size, dtype=float).reshape(eV.size, kx.size),
        dims=("eV", "kx"),
        coords={"eV": eV, "kx": kx},
        name="data",
    )
    operation = FigureOperationState.plot_slices(
        label="slices",
        sources=("data",),
        axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
        slice_dim="eV",
        slice_values=tuple(float(value) for value in eV),
    ).model_copy(update={"line_label_text": r"$E-E_F = {eV:g}$ eV"})
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    expected_labels = [r"$E-E_F = -0.1$ eV", r"$E-E_F = 0.2$ eV"]
    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    assert [axis.lines[0].get_label() for axis in tool.figure.axes] == expected_labels

    code = tool.generated_code()
    assert expected_labels[0] not in code
    assert expected_labels[1] not in code
    assert "slice_value" in code
    namespace: dict[str, typing.Any] = {"data": data}
    exec(code, namespace)  # noqa: S102
    assert [
        axis.lines[0].get_label() for axis in namespace["fig"].axes
    ] == expected_labels


def test_figure_composer_plot_slices_line_label_placeholders_accept_spaced_dim(
    qtbot,
) -> None:
    sample_temp = np.array([20.0, 30.0])
    kx = np.array([-1.0, 0.0, 1.0])
    data = xr.DataArray(
        np.arange(sample_temp.size * kx.size, dtype=float).reshape(
            sample_temp.size, kx.size
        ),
        dims=("sample temp", "kx"),
        coords={"sample temp": sample_temp, "kx": kx},
        name="data",
    )
    operation = FigureOperationState.plot_slices(
        label="slices",
        sources=("data",),
        axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
        slice_dim="sample temp",
        slice_values=tuple(float(value) for value in sample_temp),
    ).model_copy(update={"line_label_text": "{sample_temp + 1.5:g} K"})
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    expected_labels = ["21.5 K", "31.5 K"]
    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    assert [axis.lines[0].get_label() for axis in tool.figure.axes] == expected_labels

    code = tool.generated_code()
    assert expected_labels[0] not in code
    assert expected_labels[1] not in code
    assert "slice_value" in code
    namespace: dict[str, typing.Any] = {"data": data}
    exec(code, namespace)  # noqa: S102
    assert [
        axis.lines[0].get_label() for axis in namespace["fig"].axes
    ] == expected_labels


def test_figure_composer_plot_slices_line_color_codegen_helper_variants(qtbot) -> None:
    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("eV", "kx"),
        coords={"eV": [0.1, 0.2], "kx": [-1.0, 0.0, 1.0]},
        name="data",
    )
    line = xr.DataArray([1.0, 2.0], dims=("kx",), name="line")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=tuple(
                FigureSourceState(name=name, label=name.upper())
                for name in ("a", "b", "line")
            ),
            operations=(),
            primary_source="a",
        ),
        source_data={"a": data, "b": data + 10.0, "line": line},
    )
    qtbot.addWidget(tool)

    assert (
        plot_slices_codegen._plot_slices_line_color_code_lines(
            tool,
            FigureOperationState.plot_slices(
                label="manual",
                sources=("a",),
                slice_dim="eV",
                slice_values=(0.1,),
            ),
        )
        == []
    )

    with pytest.raises(ValueError, match="Choose a coordinate"):
        plot_slices_codegen._plot_slices_line_color_code_lines(
            tool,
            FigureOperationState.plot_slices(
                label="missing", sources=("line",)
            ).model_copy(update={"line_color_mode": "coordinate"}),
        )
    with pytest.raises(ValueError, match="Cannot color slices"):
        plot_slices_codegen._plot_slices_line_color_code_lines(
            tool,
            FigureOperationState.plot_slices(
                label="bad",
                sources=("a",),
                slice_dim="eV",
                slice_values=(0.1,),
            ).model_copy(
                update={"line_color_mode": "coordinate", "line_color_coord": "kx"}
            ),
        )

    one_slice = FigureOperationState.plot_slices(
        label="one",
        sources=("a", "b"),
        slice_dim="eV",
        slice_values=(0.1,),
    ).model_copy(update={"line_color_mode": "coordinate"})
    assert any(
        "for _ in range(2)" in line
        for line in plot_slices_codegen._plot_slices_line_color_code_lines(
            tool, one_slice
        )
    )

    fortran = one_slice.model_copy(update={"slice_values": (0.1, 0.2), "order": "F"})
    assert any(
        "for slice_value in [0.1, 0.2] for _ in range(2)" in line
        for line in plot_slices_codegen._plot_slices_line_color_code_lines(
            tool, fortran
        )
    )

    c_order = fortran.model_copy(update={"order": "C"})
    assert any(
        "for _ in range(2) for slice_value in [0.1, 0.2]" in line
        for line in plot_slices_codegen._plot_slices_line_color_code_lines(
            tool, c_order
        )
    )

    missing_key_operation = FigureOperationState.plot_slices(
        label="missing-key",
        sources=("missing",),
        slice_dim="eV",
        slice_values=(0.1,),
    ).model_copy(update={"line_color_mode": "coordinate"})
    assert (
        plot_slices_codegen._plot_slices_line_color_code_lines(
            tool, missing_key_operation
        )
        == []
    )

    label_operation = FigureOperationState.plot_slices(
        label="labels",
        sources=("a",),
        slice_dim="eV",
        slice_values=(0.1,),
    ).model_copy(
        update={
            "line_color_mode": "coordinate",
            "line_labels": ("literal",),
        }
    )
    label_line_kw_code = plot_slices_codegen._plot_slices_line_kw_code(
        tool, label_operation
    )
    assert label_line_kw_code is not None
    assert "'label': 'literal'" in label_line_kw_code
    assert "'color': line_colors[0]" in label_line_kw_code


def test_figure_composer_one_profile_per_axis_codegen_executes(qtbot) -> None:
    cut_values = np.array([0.0, 1.0, 2.0])
    energy = np.array([-0.2, 0.0, 0.2])
    kx = np.array([-1.0, 0.0, 1.0, 2.0])
    values = np.arange(cut_values.size * energy.size * kx.size, dtype=float).reshape(
        cut_values.size, energy.size, kx.size
    )
    data = xr.DataArray(
        values,
        dims=("cut", "eV", "kx"),
        coords={"cut": cut_values, "eV": energy, "kx": kx},
        name="data",
    )
    profile_operation = FigureOperationState.line(
        label="mdc overlay",
        source="data",
        axes=FigureAxesSelectionState(axes=((0, 0), (0, 1), (0, 2))),
    ).model_copy(
        update={
            "line_placement": "one_per_axis",
            "line_x": "kx",
            "line_selection": {"cut": cut_values.tolist(), "eV": 0.0},
            "line_iter_dim": "cut",
            "line_normalize": "max",
            "line_colors": ("black",),
            "line_scales": (0.1, 0.2, 0.3),
            "line_offsets": (-0.2, 0.0, 0.2),
        }
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=3),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.plot_slices(
                    label="selection",
                    sources=("data",),
                    axes=FigureAxesSelectionState(axes=((0, 0), (0, 1), (0, 2))),
                    slice_dim="cut",
                    slice_values=tuple(float(value) for value in cut_values),
                ),
                profile_operation,
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    code = tool.generated_code()
    assert "if len(target_axes)" not in code
    assert "profiles =" in code
    assert "profile_scales =" not in code
    assert "profile_offsets =" not in code
    assert "[0.1, 0.2, 0.3]," in code
    assert "[-0.2, 0.0, 0.2]," in code
    assert (
        "profiles = [\n    offset + scale * (profile / profile.max(skipna=True))"
    ) in code
    assert "for profile, scale, offset in zip(" in code
    assert "for ax, profile in zip(" in code
    assert "ax.plot(profile['kx'], profile" in code
    assert "_line = " not in code

    namespace: dict[str, typing.Any] = {"data": data}
    exec(code, namespace)  # noqa: S102
    axs = namespace["axs"]
    for index, axis in enumerate(axs.flat):
        line = axis.lines[0]
        profile = data.qsel(cut=float(cut_values[index]), eV=0.0).squeeze(drop=True)
        expected = profile_operation.line_offsets[
            index
        ] + profile_operation.line_scales[index] * (profile / profile.max(skipna=True))
        np.testing.assert_allclose(line.get_xdata(), kx)
        np.testing.assert_allclose(line.get_ydata(), expected.values)


def test_figure_composer_line_gradient_fill_one_profile_per_axis_executes(
    qtbot,
) -> None:
    cut_values = np.array([0.0, 1.0])
    kx = np.array([-1.0, 0.0, 1.0])
    data = xr.DataArray(
        np.array([[1.0, 2.0, 1.0], [0.5, 1.5, 0.5]]),
        dims=("cut", "kx"),
        coords={"cut": cut_values, "kx": kx},
        name="data",
    )
    operation = FigureOperationState.line(
        label="profiles",
        source="data",
        axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
    ).model_copy(
        update={
            "line_placement": "one_per_axis",
            "line_x": "kx",
            "line_iter_dim": "cut",
            "line_values_axis": "x",
            "line_colors": ("red", "blue"),
            "gradient": True,
        }
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    fig = tool.figure
    figurecomposer_rendering._render_into_figure(tool, fig, sync_visible=False)
    for index, axis in enumerate(fig.axes):
        profile = data.isel(cut=index)
        assert len(axis.lines) == 1
        assert len(axis.images) == 1
        np.testing.assert_allclose(axis.lines[0].get_xdata(), profile.values)
        np.testing.assert_allclose(axis.lines[0].get_ydata(), kx)
        np.testing.assert_allclose(
            axis.images[0].cmap(1.0),
            mcolors.to_rgba(axis.lines[0].get_color()),
            rtol=1e-7,
        )

    code = tool.generated_code()
    assert "for ax, profile, color in zip(" in code
    assert "_line = ax.plot(profile, profile['kx'], color=color)[0]" in code
    assert "transpose=True" in code
    assert "for _line in" not in code
    assert "ax.lines" not in code
    code_lines = code.splitlines()
    plot_index = code_lines.index(
        "    _line = ax.plot(profile, profile['kx'], color=color)[0]"
    )
    assert code_lines[plot_index + 1] == "    eplt.gradient_fill("

    namespace: dict[str, typing.Any] = {"data": data}
    exec(code, namespace)  # noqa: S102
    for index, axis in enumerate(namespace["fig"].axes):
        profile = data.isel(cut=index)
        assert len(axis.lines) == 1
        assert len(axis.images) == 1
        np.testing.assert_allclose(axis.lines[0].get_xdata(), profile.values)
        np.testing.assert_allclose(axis.lines[0].get_ydata(), kx)
        np.testing.assert_allclose(
            axis.images[0].cmap(1.0),
            mcolors.to_rgba(axis.lines[0].get_color()),
            rtol=1e-7,
        )


def test_figure_composer_line_gradient_fill_codegen_preserves_line_source(
    qtbot,
) -> None:
    data = xr.DataArray(
        np.array([1.0, 2.0, 1.0]),
        dims=("kx",),
        coords={"kx": [-1.0, 0.0, 1.0]},
        name="_line",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(),
            sources=(FigureSourceState(name="_line", label="_line"),),
            operations=(
                FigureOperationState.line(
                    label="gradient",
                    source="_line",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ).model_copy(update={"gradient": True}),
                FigureOperationState.line(
                    label="later",
                    source="_line",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ),
            ),
            primary_source="_line",
        ),
    )
    qtbot.addWidget(tool)

    code = tool.generated_code()
    code_lines = code.splitlines()
    assert not any(line.startswith("    _line = profile.plot(") for line in code_lines)
    assert "_line_2 = axs[0, 0].plot(profile[profile.dims[0]], profile)[0]" in code
    assert "profile_data = _line" in code

    namespace: dict[str, typing.Any] = {"_line": data}
    exec(code, namespace)  # noqa: S102
    assert len(namespace["fig"].axes[0].lines) == 2
    assert len(namespace["fig"].axes[0].images) == 1


def test_figure_composer_profile_reduce_codegen_executes(qtbot) -> None:
    cut_values = np.arange(6.0)
    kx = np.array([-1.0, 0.0, 1.0])
    data = xr.DataArray(
        np.arange(cut_values.size * kx.size, dtype=float).reshape(
            cut_values.size, kx.size
        ),
        dims=("cut", "kx"),
        coords={"cut": cut_values, "kx": kx},
        name="data",
    )
    operation = FigureOperationState.line(
        label="reduced profiles",
        source="data",
        axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
    ).model_copy(
        update={
            "line_placement": "one_per_axis",
            "line_x": "kx",
            "line_iter_dim": "cut",
            "line_reduce": "both",
            "line_reduce_coarsen": 2,
            "line_reduce_thin": 2,
        }
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    reduced = data.coarsen(cut=2, boundary="trim").mean().thin(cut=2)
    profiles = figurecomposer_line_profile._line_data_items(tool, operation)
    assert len(profiles) == 2
    xr.testing.assert_identical(profiles[0], reduced.isel(cut=0))
    xr.testing.assert_identical(profiles[1], reduced.isel(cut=1))

    fig = tool.figure
    figurecomposer_rendering._render_into_figure(tool, fig, sync_visible=False)
    for index, axis in enumerate(fig.axes):
        np.testing.assert_allclose(axis.lines[0].get_xdata(), kx)
        np.testing.assert_allclose(axis.lines[0].get_ydata(), profiles[index].values)

    code = tool.generated_code()
    assert "if len(target_axes)" not in code
    assert (
        'profile_data = data.coarsen(cut=2, boundary="trim").mean().thin(cut=2)' in code
    )
    assert "profile_data = profile_data." not in code
    namespace: dict[str, typing.Any] = {"data": data}
    exec(code, namespace)  # noqa: S102
    for index, axis in enumerate(namespace["axs"].flat):
        np.testing.assert_allclose(axis.lines[0].get_xdata(), kx)
        np.testing.assert_allclose(axis.lines[0].get_ydata(), profiles[index].values)


def test_figure_composer_one_profile_per_axis_codegen_broadcasts_profiles(
    qtbot,
) -> None:
    cut_values = np.array([0.0, 1.0, 2.0])
    kx = np.array([-1.0, 0.0, 1.0])
    data = xr.DataArray(
        np.arange(cut_values.size * kx.size, dtype=float).reshape(
            cut_values.size, kx.size
        ),
        dims=("cut", "kx"),
        coords={"cut": cut_values, "kx": kx},
        name="data",
    )

    many_profiles_operation = FigureOperationState.line(
        label="profiles",
        source="data",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
    ).model_copy(
        update={
            "line_placement": "one_per_axis",
            "line_x": "kx",
            "line_iter_dim": "cut",
            "line_offset_source": "index",
            "xlim": (-0.5, 0.5),
        }
    )
    many_profiles_tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(many_profiles_operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(many_profiles_tool)

    namespace: dict[str, typing.Any] = {"data": data}
    many_profiles_code = many_profiles_tool.generated_code()
    assert "if len(target_axes)" not in many_profiles_code
    assert "target_axes =" not in many_profiles_code
    assert "for profile in profiles:" in many_profiles_code
    assert "axs[0, 0].plot(profile['kx'], profile)" in many_profiles_code
    assert "axs[0, 0].set(xlim=(-0.5, 0.5))" in many_profiles_code
    exec(many_profiles_code, namespace)  # noqa: S102
    lines = namespace["fig"].axes[0].lines
    assert len(lines) == 3
    assert namespace["fig"].axes[0].get_xlim() == pytest.approx((-0.5, 0.5))
    for index, line in enumerate(lines):
        np.testing.assert_allclose(line.get_xdata(), kx)
        np.testing.assert_allclose(
            line.get_ydata(), data.isel(cut=index).values + index
        )
    figurecomposer_rendering._render_into_figure(
        many_profiles_tool, many_profiles_tool.figure, sync_visible=False
    )
    rendered_lines = many_profiles_tool.figure.axes[0].lines
    assert len(rendered_lines) == 3
    for index, line in enumerate(rendered_lines):
        np.testing.assert_allclose(line.get_xdata(), kx)
        np.testing.assert_allclose(
            line.get_ydata(), data.isel(cut=index).values + index
        )

    single_profile_operation = FigureOperationState.line(
        label="profile",
        source="data",
        axes=FigureAxesSelectionState(axes=((0, 0), (0, 1), (0, 2))),
    ).model_copy(
        update={
            "line_placement": "one_per_axis",
            "line_x": "kx",
            "line_selection": {"cut": 1.0},
            "line_offset_source": "index",
            "xlim": (-0.5, 0.5),
        }
    )
    single_profile_tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=3),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(single_profile_operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(single_profile_tool)

    namespace = {"data": data}
    single_profile_code = single_profile_tool.generated_code()
    assert "if len(target_axes)" not in single_profile_code
    assert "target_axes =" not in single_profile_code
    assert "profiles * 3," in single_profile_code
    exec(single_profile_code, namespace)  # noqa: S102
    profile = data.qsel(cut=1.0).squeeze(drop=True)
    for index, axis in enumerate(namespace["axs"].flat):
        assert len(axis.lines) == 1
        assert axis.get_xlim() == pytest.approx((-0.5, 0.5))
        np.testing.assert_allclose(axis.lines[0].get_xdata(), kx)
        np.testing.assert_allclose(axis.lines[0].get_ydata(), profile.values + index)
    figurecomposer_rendering._render_into_figure(
        single_profile_tool, single_profile_tool.figure, sync_visible=False
    )
    for index, axis in enumerate(single_profile_tool.figure.axes):
        assert len(axis.lines) == 1
        assert axis.get_xlim() == pytest.approx((-0.5, 0.5))
        np.testing.assert_allclose(axis.lines[0].get_xdata(), kx)
        np.testing.assert_allclose(axis.lines[0].get_ydata(), profile.values + index)


def test_figure_composer_regular_line_profiles_render_on_each_selected_axis(
    qtbot,
) -> None:
    cut_values = np.array([0.0, 1.0])
    kx = np.array([-1.0, 0.0, 1.0])
    data = xr.DataArray(
        np.arange(cut_values.size * kx.size, dtype=float).reshape(
            cut_values.size, kx.size
        ),
        dims=("cut", "kx"),
        coords={"cut": cut_values, "kx": kx},
        name="data",
    )
    operation = FigureOperationState.line(
        label="profiles",
        source="data",
        axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
    ).model_copy(
        update={
            "line_x": "kx",
            "line_iter_dim": "cut",
            "line_values_axis": "x",
            "line_labels": ("low", "high"),
            "line_colors": ("red", "blue"),
            "line_scales": (2.0,),
            "ylim": (-1.5, 1.5),
        }
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    fig = tool.figure
    figurecomposer_rendering._render_into_figure(tool, fig, sync_visible=False)
    for axis in fig.axes:
        assert len(axis.lines) == 2
        assert axis.get_ylim() == pytest.approx((-1.5, 1.5))
        for index, line in enumerate(axis.lines):
            np.testing.assert_allclose(
                line.get_xdata(), 2.0 * data.isel(cut=index).values
            )
            np.testing.assert_allclose(line.get_ydata(), kx)
            assert line.get_label() == ("low", "high")[index]
            assert line.get_color() == ("red", "blue")[index]

    code = tool.generated_code()
    assert "for ax in" in code
    assert "ax.plot(profile, profile['kx']" in code
    namespace: dict[str, typing.Any] = {"data": data}
    exec(code, namespace)  # noqa: S102
    for axis in namespace["fig"].axes:
        assert len(axis.lines) == 2
        for index, line in enumerate(axis.lines):
            np.testing.assert_allclose(
                line.get_xdata(), 2.0 * data.isel(cut=index).values
            )
            np.testing.assert_allclose(line.get_ydata(), kx)

    test_fig = Figure()
    test_axis = test_fig.subplots()
    with pytest.warns(UserWarning, match="identical low and high xlims"):
        figurecomposer_line_profile._set_axis_xlim(test_axis, 1.0)
    figurecomposer_line_profile._set_axis_ylim(test_axis, 2.0)
    assert test_axis.get_xlim()[0] < 1.0 < test_axis.get_xlim()[1]
    assert test_axis.get_ylim()[0] == pytest.approx(2.0)


def test_figure_composer_regular_line_gradient_fill_codegen_executes(
    qtbot,
) -> None:
    cut_values = np.array([0.0, 1.0])
    kx = np.array([-1.0, 0.0, 1.0])
    data = xr.DataArray(
        np.arange(cut_values.size * kx.size, dtype=float).reshape(
            cut_values.size, kx.size
        ),
        dims=("cut", "kx"),
        coords={"cut": cut_values, "kx": kx},
        name="data",
    )
    operation = FigureOperationState.line(
        label="profiles",
        source="data",
        axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
    ).model_copy(
        update={
            "line_x": "kx",
            "line_iter_dim": "cut",
            "line_colors": ("red", "blue"),
            "gradient": True,
        }
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    fig = tool.figure
    figurecomposer_rendering._render_into_figure(tool, fig, sync_visible=False)
    for axis in fig.axes:
        assert len(axis.lines) == 2
        assert len(axis.images) == 2
        for index, line in enumerate(axis.lines):
            np.testing.assert_allclose(line.get_xdata(), kx)
            np.testing.assert_allclose(line.get_ydata(), data.isel(cut=index).values)
            assert line.get_color() == ("red", "blue")[index]

    code = tool.generated_code()
    assert "_line = ax.plot(profile['kx'], profile, color=color)[0]" in code
    code_lines = code.splitlines()
    plot_index = code_lines.index(
        "        _line = ax.plot(profile['kx'], profile, color=color)[0]"
    )
    assert code_lines[plot_index + 1] == "        eplt.gradient_fill("

    namespace: dict[str, typing.Any] = {"data": data}
    exec(code, namespace)  # noqa: S102
    for axis in namespace["fig"].axes:
        assert len(axis.lines) == 2
        assert len(axis.images) == 2
        for index, line in enumerate(axis.lines):
            np.testing.assert_allclose(line.get_xdata(), kx)
            np.testing.assert_allclose(line.get_ydata(), data.isel(cut=index).values)


def test_figure_composer_line_gradient_fill_uses_new_line_after_existing_line(
    qtbot,
) -> None:
    kx = np.array([-1.0, 0.0, 1.0])
    data = xr.DataArray(
        np.array([1.0, 2.0, 1.0]),
        dims=("kx",),
        coords={"kx": kx},
        name="data",
    )
    base_operation = FigureOperationState.line(
        label="base",
        source="data",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
    ).model_copy(update={"line_x": "kx", "line_colors": ("black",)})
    gradient_operation = FigureOperationState.line(
        label="gradient",
        source="data",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
    ).model_copy(
        update={
            "line_x": "kx",
            "line_colors": ("red",),
            "gradient": True,
        }
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(base_operation, gradient_operation),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    fig = tool.figure
    figurecomposer_rendering._render_into_figure(tool, fig, sync_visible=False)
    axis = fig.axes[0]
    assert len(axis.lines) == 2
    assert len(axis.images) == 1
    assert axis.lines[0].get_color() == "black"
    assert axis.lines[1].get_color() == "red"
    np.testing.assert_allclose(
        axis.images[0].cmap(1.0),
        mcolors.to_rgba(axis.lines[1].get_color()),
        rtol=1e-7,
    )

    code = tool.generated_code()
    assert "profile.plot(" not in code
    namespace: dict[str, typing.Any] = {"data": data}
    exec(code, namespace)  # noqa: S102
    axis = namespace["fig"].axes[0]
    assert len(axis.lines) == 2
    assert len(axis.images) == 1
    assert axis.lines[0].get_color() == "black"
    assert axis.lines[1].get_color() == "red"
    np.testing.assert_allclose(
        axis.images[0].cmap(1.0),
        mcolors.to_rgba(axis.lines[1].get_color()),
        rtol=1e-7,
    )


def test_figure_composer_line_labels_auto_add_axes_legend_step(
    qtbot, monkeypatch
) -> None:
    profile = xr.DataArray(
        np.array([1.0, 2.0, 3.0]),
        dims=("kx",),
        coords={"kx": [-1.0, 0.0, 1.0]},
        name="profile",
    )
    tool = FigureComposerTool(
        profile,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="profile", label="profile"),),
            operations=(
                FigureOperationState.line(
                    label="profile",
                    source="profile",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ),
            ),
            primary_source="profile",
        ),
    )
    qtbot.addWidget(tool)

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )
    tool.operation_editor.select_section("style")
    labels_edit = tool.operation_editor.stack.currentWidget().findChild(
        QtWidgets.QLineEdit, "figureComposerLineLabelsEdit"
    )
    assert labels_edit is not None

    rebuild_calls: list[None] = []
    monkeypatch.setattr(
        tool,
        "_update_operation_editor",
        lambda: rebuild_calls.append(None),
    )
    labels_edit.setText("profile A")
    labels_edit.editingFinished.emit()

    assert rebuild_calls == []
    assert tool._current_operation_index() == 0
    assert len(tool.tool_status.operations) == 2
    line_operation, legend_operation = tool.tool_status.operations
    assert line_operation.line_label_text == "profile A"
    assert line_operation.line_labels == ()
    assert legend_operation.kind == FigureOperationKind.METHOD
    assert legend_operation.method_family == FigureMethodFamily.AXES
    assert legend_operation.method_name == "legend"
    assert legend_operation.axes == line_operation.axes

    labels_edit.setText("profile B")
    labels_edit.editingFinished.emit()

    assert len(tool.tool_status.operations) == 2
    assert tool.tool_status.operations[0].line_label_text == "profile B"
    assert tool.tool_status.operations[0].line_labels == ()


def test_figure_composer_disabled_line_labels_do_not_add_legend_or_render(
    qtbot,
    monkeypatch,
) -> None:
    profile = xr.DataArray(
        np.array([1.0, 2.0, 3.0]),
        dims=("kx",),
        coords={"kx": [-1.0, 0.0, 1.0]},
        name="profile",
    )
    tool = FigureComposerTool(
        profile,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="profile", label="profile"),),
            operations=(
                FigureOperationState.line(
                    label="profile",
                    source="profile",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ).model_copy(update={"enabled": False}),
            ),
            primary_source="profile",
        ),
    )
    qtbot.addWidget(tool)

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )
    tool.operation_editor.select_section("style")
    labels_edit = tool.operation_editor.stack.currentWidget().findChild(
        QtWidgets.QLineEdit, "figureComposerLineLabelsEdit"
    )
    assert labels_edit is not None
    render_calls: list[tuple[object, ...]] = []
    info_changed: list[None] = []
    tool.sigInfoChanged.connect(lambda: info_changed.append(None))
    monkeypatch.setattr(
        figurecomposer_tool_module,
        "_render_preview",
        lambda *args, **_kwargs: render_calls.append(args),
    )

    labels_edit.setText("disabled profile")
    labels_edit.editingFinished.emit()

    assert len(tool.tool_status.operations) == 1
    assert tool.tool_status.operations[0].line_label_text == "disabled profile"
    assert tool.tool_status.operations[0].line_labels == ()
    assert render_calls == []
    assert not tool._preview_render_update_pending
    assert info_changed == [None]


def test_figure_composer_batch_line_labels_add_one_legend_per_axes_group(
    qtbot,
) -> None:
    profile = xr.DataArray(
        np.array([1.0, 2.0, 3.0]),
        dims=("kx",),
        coords={"kx": [-1.0, 0.0, 1.0]},
        name="profile",
    )
    first_axes = FigureAxesSelectionState(axes=((0, 0),))
    second_axes = FigureAxesSelectionState(axes=((0, 1),))
    tool = FigureComposerTool(
        profile,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(ncols=2),
            sources=(FigureSourceState(name="profile", label="profile"),),
            operations=(
                FigureOperationState.line(
                    label="first",
                    source="profile",
                    axes=first_axes,
                ),
                FigureOperationState.line(
                    label="second",
                    source="profile",
                    axes=first_axes,
                ),
                FigureOperationState.line(
                    label="third",
                    source="profile",
                    axes=second_axes,
                ),
            ),
            primary_source="profile",
        ),
    )
    qtbot.addWidget(tool)

    _select_operation_rows(tool, (0, 1, 2))
    tool.operation_editor.select_section("style")
    labels_edit = tool.operation_editor.stack.currentWidget().findChild(
        QtWidgets.QLineEdit, "figureComposerLineLabelsEdit"
    )
    assert labels_edit is not None
    labels_edit.setText("profile")
    labels_edit.editingFinished.emit()

    operations = tool.tool_status.operations
    assert len(operations) == 5
    assert [
        operation.line_label_text
        for operation in operations
        if operation.kind == FigureOperationKind.LINE
    ] == ["profile", "profile", "profile"]
    assert operations[2].method_name == "legend"
    assert operations[2].axes == first_axes
    assert operations[4].method_name == "legend"
    assert operations[4].axes == second_axes


def test_figure_composer_line_mean_normalization_executes(qtbot) -> None:
    profile = xr.DataArray(
        np.array([2.0, 4.0, 6.0]),
        dims=("kx",),
        coords={"kx": [-1.0, 0.0, 1.0]},
        name="profile",
    )
    operation = FigureOperationState.line(
        label="profile",
        source="profile",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
    ).model_copy(update={"line_x": "kx", "line_normalize": "mean"})
    tool = FigureComposerTool(
        profile,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(),
            sources=(FigureSourceState(name="profile", label="profile"),),
            operations=(operation,),
            primary_source="profile",
        ),
    )
    qtbot.addWidget(tool)

    expected = profile / profile.mean(skipna=True)
    fig = tool.figure
    figurecomposer_rendering._render_into_figure(tool, fig, sync_visible=False)
    line = fig.axes[0].lines[0]
    np.testing.assert_allclose(line.get_xdata(), profile["kx"].values)
    np.testing.assert_allclose(line.get_ydata(), expected.values)

    code = tool.generated_code()
    assert "profile.mean(skipna=True)" in code
    namespace: dict[str, typing.Any] = {"profile": profile}
    exec(code, namespace)  # noqa: S102
    line = namespace["fig"].axes[0].lines[0]
    np.testing.assert_allclose(line.get_xdata(), profile["kx"].values)
    np.testing.assert_allclose(line.get_ydata(), expected.values)


def test_figure_composer_line_normalization_reports_zero_scale(
    qtbot, monkeypatch
) -> None:
    profile = xr.DataArray(
        np.array([0.0, 0.0]),
        dims=("kx",),
        coords={"kx": [-1.0, 1.0]},
        name="profile",
    )
    operation = FigureOperationState.line(
        label="profile",
        source="profile",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
    ).model_copy(update={"line_x": "kx", "line_normalize": "max"})
    tool = FigureComposerTool(
        profile,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(),
            sources=(FigureSourceState(name="profile", label="profile"),),
            operations=(operation,),
            primary_source="profile",
        ),
    )
    qtbot.addWidget(tool)

    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)

    error_text = tool._operation_render_errors[operation.operation_id]
    assert "Cannot normalize profile by max" in error_text
    with pytest.raises(ValueError, match="Cannot normalize profile by max"):
        tool.generated_code()

    dialogs: list[typing.Any] = []

    class _RecordingMessageDialog:
        def __init__(self, parent=None, **kwargs) -> None:
            self.parent = parent
            self.kwargs = kwargs
            dialogs.append(self)

        def exec(self) -> int:
            return int(QtWidgets.QDialog.DialogCode.Accepted)

    monkeypatch.setattr(
        erlab.interactive.utils, "MessageDialog", _RecordingMessageDialog
    )
    tool.copy_code()
    assert dialogs
    assert dialogs[0].kwargs["title"] == "Cannot Copy Figure Code"
    assert "Cannot normalize profile by max" in dialogs[0].kwargs["text"]
    assert "Traceback" in dialogs[0].kwargs["detailed_text"]


def test_figure_composer_label_helper_edges() -> None:
    assert figurecomposer_labels.label_coord_placeholder_name("") == "field"
    assert figurecomposer_labels.label_coord_placeholder_name("1 eV") == "_1_eV"
    assert figurecomposer_labels.label_coord_placeholder_name("class") == "class_"
    assert figurecomposer_labels._explicit_field_names("{{x}} {sample_temp:.1f}") == {
        "sample_temp"
    }

    profile = xr.DataArray(
        [1.0],
        dims=("x",),
        coords={"x": [0.0], "sample_temp": 20.0},
        name="profile",
    )
    context = figurecomposer_labels.label_context(
        profile,
        index=2,
        source="map",
        dim="sample_temp",
        value=np.asarray([20.0]),
    )
    assert context["number"] == 3
    assert context["source"] == "map"
    assert context["sample_temp"] == 20.0
    assert context["value"] == 20.0
    spaced_profile = xr.DataArray(
        [1.0],
        dims=("x",),
        coords={"x": [0.0], "sample temp": 20.0},
        attrs={"sample label": "annealed"},
        name="profile",
    )
    spaced_context = figurecomposer_labels.label_context(
        spaced_profile,
        index=0,
        dim="sample temp",
        value=np.asarray([20.0]),
    )
    assert spaced_context["sample_temp"] == 20.0
    assert spaced_context["sample_label"] == "annealed"
    assert figurecomposer_labels.labels_from_text(
        "{sample_temp + 1.5:g} K, {sample_label}", (spaced_context,)
    ) == ("21.5 K, annealed",)
    with pytest.raises(ValueError, match=r"Use \{sample_temp:g\}") as exc_info:
        figurecomposer_labels.labels_from_text("{sample temp:g}", (spaced_context,))
    error_message = str(exc_info.value)
    assert "Use {sample_temp:g}" in error_message
    assert "Available placeholders" not in error_message
    label_code = figurecomposer_labels.label_fstring_code(
        "{sample_temp + 1.5:g} K, {sample_label}",
        {
            "sample_temp": figurecomposer_labels.coord_value_expression("sample temp"),
            "sample_label": figurecomposer_labels.attr_value_expression("sample label"),
        },
    )
    assert eval(label_code, {"profile": spaced_profile}) == "21.5 K, annealed"  # noqa: S307
    collision_profile = xr.DataArray(
        [1.0],
        dims=("x",),
        coords={
            "x": [0.0],
            "sample temp": 20.0,
            "sample-temp": 30.0,
            "sample_temp": 40.0,
            "class": 1.0,
            "2 theta": 2.0,
        },
        name="profile",
    )
    collision_context = figurecomposer_labels.label_context(collision_profile, index=0)
    assert collision_context["sample_temp"] == 20.0
    assert collision_context["sample_temp_2"] == 30.0
    assert collision_context["sample_temp_3"] == 40.0
    assert collision_context["class_"] == 1.0
    assert collision_context["_2_theta"] == 2.0
    mixed_context = {
        figurecomposer_labels._LABEL_FIELD_SOURCES_KEY: {
            "sample_temp": "attr",
            "field": "custom",
        },
        figurecomposer_labels._LABEL_FIELD_ORIGINAL_NAMES_KEY: {
            "sample_temp": "other temp",
            "field": "other field",
        },
    }
    invalid_metadata_context = {
        figurecomposer_labels._LABEL_FIELD_SOURCES_KEY: "bad",
        figurecomposer_labels._LABEL_FIELD_ORIGINAL_NAMES_KEY: "bad",
    }
    assert figurecomposer_labels.label_context_field_sources(
        (spaced_context, invalid_metadata_context, mixed_context)
    ) == {
        "x": "coord",
        "sample_temp": "mixed",
        "sample_label": "attr",
        "field": "custom",
    }
    assert figurecomposer_labels.label_context_original_field_names(
        (spaced_context, invalid_metadata_context, mixed_context)
    ) == {
        "x": "x",
        "sample_temp": "",
        "sample_label": "sample label",
        "field": "other field",
    }
    assert (
        figurecomposer_labels.label_context_coord_alias(
            {
                figurecomposer_labels._LABEL_COORD_ALIASES_KEY: {"x": 1},
                "x": 0.0,
            },
            "x",
        )
        == "x"
    )
    assert figurecomposer_labels.label_context_coord_alias({}, "missing") is None
    assert figurecomposer_labels.labels_from_text("", (), default="fallback") == ()
    assert figurecomposer_labels.labels_from_text(
        "", (context, context), literal_values=("one",), default=None
    ) == ("one", "one")
    with pytest.raises(ValueError, match="one value per profile"):
        figurecomposer_labels.labels_from_text(
            "", (context, context), literal_values=("a", "b", "c")
        )
    assert figurecomposer_labels.labels_from_text(
        "$k_{F}$", (context,), default=None
    ) == (r"$k_{F}$",)
    assert figurecomposer_labels.labels_from_text(
        "{{literal}} {missing", (context,), default=None
    ) == ("{{literal}} {missing",)
    assert figurecomposer_labels.labels_from_text(
        "{source!r}:{sample_temp!s}:{sample_temp!a}:{sample_temp:.1f}",
        (context,),
    ) == ("'map':20.0:20.0:20.0",)
    format_context = {**context, "width": 5}
    assert figurecomposer_labels.labels_from_text(
        "{sample_temp:{width}.1f}", (format_context,)
    ) == (" 20.0",)
    assert figurecomposer_labels.labels_from_text(
        "{sample_temp + 1.5:.1f}", (context,)
    ) == ("21.5",)
    with pytest.raises(ValueError, match="basic operators"):
        figurecomposer_labels.labels_from_text("{abs(sample_temp)}", (context,))
    with pytest.raises(ValueError, match="not available"):
        figurecomposer_labels.labels_from_text("{missing + 1}", (context,))
    with pytest.raises(ValueError, match="missing") as exc_info:
        figurecomposer_labels.labels_from_text("{missing:g}", (context,))
    error_message = str(exc_info.value)
    assert "not available" in error_message
    assert "missing" in error_message
    assert "Available placeholders" not in error_message
    with pytest.raises(ValueError, match="Could not format"):
        figurecomposer_labels.labels_from_text("{source:g}", (context,))

    assert not figurecomposer_labels.label_text_uses_placeholders("", (context,))
    assert not figurecomposer_labels.label_text_uses_placeholders("plain", (context,))
    assert (
        figurecomposer_labels.label_fstring_code(
            r"{source}\{missing}", {"source": "source"}
        )
        == r'f"{source}\\{{missing}}"'
    )
    assert (
        figurecomposer_labels.label_fstring_code("{{literal}} {missing", {})
        == r'f"{{literal}} {{missing"'
    )
    formatted_code = figurecomposer_labels.label_fstring_code(
        "{sample_temp!r}:{sample_temp:{width}.1f}",
        {
            "sample_temp": figurecomposer_labels.coord_value_expression("sample temp"),
            "width": "5",
        },
    )
    assert eval(formatted_code, {"profile": spaced_profile}) == "20.0: 20.0"  # noqa: S307
    with pytest.raises(ValueError, match="missing") as exc_info:
        figurecomposer_labels.label_fstring_code("{missing:g}", {"source": "source"})
    error_message = str(exc_info.value)
    assert "not available" in error_message
    assert "missing" in error_message
    assert "Available placeholders" not in error_message
    with pytest.raises(ValueError, match="basic operators"):
        figurecomposer_labels.label_fstring_code(
            "{sample_temp.real}",
            {
                "sample_temp": figurecomposer_labels.coord_value_expression(
                    "sample temp"
                )
            },
        )
    assert figurecomposer_labels.coord_value_expression("sample_temp") == (
        'profile.coords["sample_temp"].values.item()'
    )
    assert figurecomposer_labels.string_literal_expression("a'b") == '"a\'b"'
    assert (
        figurecomposer_labels.default_label_text(
            None, fallback="profile {number}", source_count=2
        )
        == "{source}, profile {number}"
    )
    assert (
        figurecomposer_labels.default_label_text(
            "phase", ("A",), fallback="profile {number}", source_count=2
        )
        == r"{source}, $phase = {phase}$"
    )
    assert "comma-separated" in figurecomposer_labels.label_text_tooltip(
        (), item_name="profile"
    )
    tooltip = figurecomposer_labels.label_text_tooltip(
        (spaced_context,), item_name="profile"
    )
    assert "\n" not in tooltip
    assert "data names" not in tooltip.lower()
    placeholder_rows = {
        row.placeholder: row
        for row in figurecomposer_labels.label_text_help_placeholder_rows(
            (spaced_context,), item_name="profile"
        )
    }
    assert placeholder_rows["sample_temp"].kind == "coord"
    assert placeholder_rows["sample_temp"].description.endswith("'sample temp'")
    assert placeholder_rows["sample_label"].kind == "attr"
    assert placeholder_rows["sample_label"].description.endswith("'sample label'")
    many_context = figurecomposer_labels.label_context(
        xr.DataArray(
            [1.0],
            dims=("x",),
            coords={
                "x": [0.0],
                **{f"coord {index}": float(index) for index in range(80)},
            },
        ),
        index=0,
    )
    many_rows = figurecomposer_labels.label_text_help_placeholder_rows(
        (many_context,), item_name="profile"
    )
    row_names = {row.placeholder for row in many_rows}
    assert "more" not in {row.kind for row in many_rows}
    assert {f"coord_{index}" for index in range(80)}.issubset(row_names)
    assert (
        figurecomposer_labels.label_editor_text(
            FigureOperationState.line(label="line", source="data").model_copy(
                update={"line_labels": ("A", "B")}
            )
        )
        == "A, B"
    )


def test_figure_composer_line_coordinate_colormap_rejects_bad_values(qtbot) -> None:
    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("eV", "kx"),
        coords={"eV": [0.0, np.nan], "kx": [-1.0, 0.0, 1.0]},
        name="data",
    )
    operation = FigureOperationState.line(
        label="profiles",
        source="data",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
    ).model_copy(
        update={
            "line_x": "kx",
            "line_iter_dim": "eV",
            "line_color_mode": "coordinate",
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

    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    render_error = tool._operation_render_errors[operation.operation_id]
    assert "finite numeric scalars" in render_error
    with pytest.raises(ValueError, match="finite numeric scalars"):
        tool.generated_code()

    plot_operation = FigureOperationState.plot_slices(
        label="slices",
        sources=("data",),
        axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
        slice_dim="eV",
        slice_values=(0.0, np.nan),
    ).model_copy(update={"line_color_mode": "coordinate"})
    plot_tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(plot_operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(plot_tool)
    with pytest.raises(ValueError, match="finite numeric scalars"):
        plot_slices_model._panel_line_kw_argument(plot_tool, plot_operation)
    with pytest.raises(ValueError, match="finite numeric scalars"):
        plot_tool.generated_code()


def test_figure_composer_line_action_seeds_from_selected_slice_step(
    qtbot,
) -> None:
    data = xr.DataArray(
        np.arange(18.0).reshape(3, 2, 3),
        dims=("cut", "eV", "kx"),
        coords={
            "cut": [0.0, 1.0, 2.0],
            "eV": [-0.1, 0.1],
            "kx": [-1.0, 0.0, 1.0],
        },
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=3),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.plot_slices(
                    label="selection",
                    sources=("data",),
                    axes=FigureAxesSelectionState(axes=((0, 0), (0, 1), (0, 2))),
                    slice_dim="cut",
                    slice_values=(0.0, 1.0, 2.0),
                    slice_width=0.25,
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    profile_action = next(
        action
        for action in tool.operation_panel.add_step_menu.actions()
        if action.data() == "line"
    )
    profile_action.trigger()

    operation = tool.tool_status.operations[-1]
    assert operation.kind == FigureOperationKind.LINE
    assert operation.line_placement == "one_per_axis"
    assert operation.line_normalize == "max"
    assert operation.line_colors == ("black",)
    assert operation.line_x == "kx"
    assert operation.line_iter_dim == "cut"
    assert operation.line_selection == {"cut": [0.0, 1.0, 2.0], "cut_width": 0.25}
    assert operation.axes.axes == ((0, 0), (0, 1), (0, 2))

    tool.operation_editor.select_section("view")
    placement_combo = tool.operation_editor.stack.currentWidget().findChild(
        QtWidgets.QComboBox, "figureComposerProfilePlacementCombo"
    )
    assert placement_combo is not None

    tool.operation_editor.select_section("other")
    normalize_combo = tool.operation_editor.stack.currentWidget().findChild(
        QtWidgets.QComboBox, "figureComposerLineNormalizeCombo"
    )
    assert normalize_combo is not None
    assert "each extracted 1D profile independently" in normalize_combo.toolTip()

    unseeded_tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            primary_source="data",
        ),
    )
    qtbot.addWidget(unseeded_tool)
    line_action = next(
        action
        for action in unseeded_tool.operation_panel.add_step_menu.actions()
        if action.data() == "line"
    )
    line_action.trigger()
    assert unseeded_tool.tool_status.operations[-1].line_placement == "all_axes"


def test_figure_composer_batch_line_edits_update_selected_steps(qtbot) -> None:
    profile = xr.DataArray(
        np.array([1.0, 2.0, 3.0]),
        dims=("kx",),
        coords={"kx": [-1.0, 0.0, 1.0]},
        name="profile",
    )
    tool = FigureComposerTool(
        profile,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="profile", label="profile"),),
            operations=(
                FigureOperationState.line(label="first", source="profile"),
                FigureOperationState.line(label="second", source="profile"),
                FigureOperationState.line(label="third", source="profile"),
                FigureOperationState.line(label="unselected", source="profile"),
            ),
            primary_source="profile",
        ),
    )
    qtbot.addWidget(tool)

    _select_operation_rows(tool, (0, 1, 2))
    tool.operation_editor.select_section("other")
    other_page = tool.operation_editor.stack.currentWidget()
    normalize_combo = other_page.findChild(
        QtWidgets.QComboBox, "figureComposerLineNormalizeCombo"
    )
    assert normalize_combo is not None
    _activate_combo_text(normalize_combo, "Each profile by mean")

    assert [operation.line_normalize for operation in tool.tool_status.operations] == [
        "mean",
        "mean",
        "mean",
        "none",
    ]
    assert _selected_operation_rows(tool) == (0, 1, 2)


def test_figure_composer_batch_line_mixed_values_do_not_overwrite_on_blur(
    qtbot,
) -> None:
    profile = xr.DataArray(
        np.array([1.0, 2.0, 3.0]),
        dims=("kx",),
        coords={"kx": [-1.0, 0.0, 1.0]},
        name="profile",
    )
    tool = FigureComposerTool(
        profile,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="profile", label="profile"),),
            operations=(
                FigureOperationState.line(label="first", source="profile").model_copy(
                    update={"line_colors": ("C0",)}
                ),
                FigureOperationState.line(label="second", source="profile").model_copy(
                    update={"line_colors": ("C1",)}
                ),
            ),
            primary_source="profile",
        ),
    )
    qtbot.addWidget(tool)

    _select_operation_rows(tool, (0, 1))
    tool.operation_editor.select_section("style")
    style_page = tool.operation_editor.stack.currentWidget()
    color_edit = style_page.findChild(
        QtWidgets.QLineEdit, "figureComposerLineColorsEdit"
    )
    assert color_edit is not None
    assert color_edit.text() == ""
    assert color_edit.placeholderText() == "(multiple values)"

    color_edit.editingFinished.emit()
    assert [operation.line_colors for operation in tool.tool_status.operations] == [
        ("C0",),
        ("C1",),
    ]

    color_edit.setText("black")
    color_edit.setModified(True)
    color_edit.editingFinished.emit()
    assert [operation.line_colors for operation in tool.tool_status.operations] == [
        ("black",),
        ("black",),
    ]
