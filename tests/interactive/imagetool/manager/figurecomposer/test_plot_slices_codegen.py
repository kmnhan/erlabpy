"""Plot Slices codegen behavior tests."""

from ._plot_slices_common import (
    _SCRIPT_REPLAY_ALLOWED_BUILTINS,
    FigureAxesSelectionState,
    FigureComposerTool,
    FigureMethodFamily,
    FigureOperationState,
    FigurePlotSlicesPanelStyleState,
    FigureRecipeState,
    FigureSourceState,
    FigureSubplotsState,
    QtWidgets,
    _expected_line_colormap_colors,
    _render_figure_composer_rgba,
    _select_operation_rows,
    _validate_script_code_names,
    eplt,
    erlab,
    figurecomposer_adapter,
    figurecomposer_rendering,
    mcolors,
    np,
    plot_slices_codegen,
    plot_slices_model,
    plt,
    pytest,
    typing,
    xr,
)


def test_figure_composer_plot_slices_line_coordinate_colormap_codegen(
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
    ).model_copy(
        update={
            "line_color_mode": "coordinate",
            "line_color_cmap": "viridis",
            "line_color_cmap_trim_lower": 0.1,
            "line_color_cmap_trim_upper": 0.15,
            "line_kw": {"color": "red", "linestyle": "--"},
            "line_label_text": r"$E-E_F = {eV:g}$ eV",
            "panel_styles_enabled": True,
            "panel_styles": (
                FigurePlotSlicesPanelStyleState(
                    map_index=0,
                    slice_index=1,
                    line_kw={"linewidth": 2.5, "color": "black"},
                ),
            ),
        }
    )
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

    expected_colors = _expected_line_colormap_colors(eV, "viridis", trim=(0.1, 0.15))
    assert plot_slices_model._available_plot_slices_line_color_coords(
        tool._document, tool._source_display_name, operation
    ) == ["eV"]
    line_kw = plot_slices_model._panel_line_kw_argument(tool, operation)
    assert isinstance(line_kw, list)
    assert line_kw[0][0]["linestyle"] == "--"
    assert line_kw[0][1]["linewidth"] == 2.5
    assert "c" not in line_kw[0][0]
    np.testing.assert_allclose(
        [line_kw[0][0]["color"], line_kw[0][1]["color"]],
        expected_colors,
    )

    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    rendered_lines = [axis.lines[0] for axis in tool.figure.axes]
    np.testing.assert_allclose(
        np.asarray([mcolors.to_rgba(line.get_color()) for line in rendered_lines]),
        expected_colors,
    )
    assert [line.get_linestyle() for line in rendered_lines] == ["--", "--"]
    assert rendered_lines[1].get_linewidth() == 2.5

    tool.operation_editor.select_section("colors")
    colors_page = tool.operation_editor.stack.currentWidget()
    assert (
        colors_page.findChild(
            QtWidgets.QComboBox, "figureComposerPlotSlicesLineColorModeCombo"
        )
        is not None
    )
    assert (
        colors_page.findChild(
            QtWidgets.QComboBox, "figureComposerPlotSlicesLineColorCoordCombo"
        )
        is not None
    )
    assert (
        colors_page.findChild(
            erlab.interactive.colors.ColorMapComboBox,
            "figureComposerPlotSlicesLineColorCmapCombo",
        )
        is not None
    )
    trim_lower_spin = colors_page.findChild(
        QtWidgets.QDoubleSpinBox,
        "figureComposerPlotSlicesLineColorCmapTrimLowerSpin",
    )
    trim_upper_spin = colors_page.findChild(
        QtWidgets.QDoubleSpinBox,
        "figureComposerPlotSlicesLineColorCmapTrimUpperSpin",
    )
    assert trim_lower_spin is not None
    assert trim_upper_spin is not None
    assert trim_lower_spin.value() == pytest.approx(0.1)
    assert trim_upper_spin.value() == pytest.approx(0.15)
    assert not trim_lower_spin.keyboardTracking()
    assert not trim_upper_spin.keyboardTracking()
    assert trim_lower_spin.toolTip()
    assert trim_upper_spin.toolTip()
    assert (
        colors_page.findChild(
            QtWidgets.QLineEdit, "figureComposerPlotSlicesLineColorEdit"
        )
        is None
    )

    code = tool.generated_code()
    assert "import matplotlib.colors as mcolors" in code
    assert "line_color_values =" in code
    assert "line_colors = plt.get_cmap('viridis')(" in code
    assert "0.1 + 0.75 * line_color_values_norm(line_color_values)" in code
    assert "red" not in code
    assert "black" not in code
    assert "line_colors[0]" in code
    namespace: dict[str, typing.Any] = {"data": data}
    exec(code, namespace)  # noqa: S102
    generated_lines = [axis.lines[0] for axis in namespace["fig"].axes]
    np.testing.assert_allclose(
        np.asarray([mcolors.to_rgba(line.get_color()) for line in generated_lines]),
        expected_colors,
    )
    assert [line.get_linestyle() for line in generated_lines] == ["--", "--"]
    assert generated_lines[1].get_linewidth() == 2.5
    trim_lower_spin.setValue(0.2)
    trim_upper_spin.setValue(0.25)
    assert tool.tool_status.operations[0].line_color_cmap_trim_lower == pytest.approx(
        0.2
    )
    assert tool.tool_status.operations[0].line_color_cmap_trim_upper == pytest.approx(
        0.25
    )


def test_figure_composer_all_coordinate_multisource_codegen_executes(qtbot) -> None:
    first = xr.DataArray(
        np.arange(6.0).reshape(2, 3) + 1.0,
        dims=("eV", "kx"),
        coords={"eV": [0.0, 1.0], "kx": [-1.0, 0.0, 1.0]},
        name="first",
    )
    second = (first + 10.0).rename("second")
    operation = FigureOperationState.plot_slices(
        label="all coordinates",
        sources=("first", "second"),
        axes=FigureAxesSelectionState(axes=((0, 0), (0, 1), (1, 0), (1, 1))),
        slice_dim="eV",
    ).model_copy(
        update={
            "slice_values_mode": "all",
            "line_normalize": "max",
            "line_label_text": "{index}: {source} {dim}={value:g}",
        }
    )
    tool = FigureComposerTool.from_sources(
        {"first": first, "second": second},
        sources=(
            FigureSourceState(name="first", label="first"),
            FigureSourceState(name="second", label="second"),
        ),
        operations=(operation,),
        setup=FigureSubplotsState(nrows=2, ncols=2),
        primary_source="first",
    )
    qtbot.addWidget(tool)

    namespace: dict[str, typing.Any] = {"first": first, "second": second}
    exec(tool.generated_code(), namespace)  # noqa: S102

    figure = namespace["fig"]
    try:
        lines = [axis.lines[0] for axis in figure.axes]
        assert [line.get_label() for line in lines] == [
            "0: first eV=0",
            "1: first eV=1",
            "2: second eV=0",
            "3: second eV=1",
        ]
        for line in lines:
            assert np.max(line.get_ydata()) == pytest.approx(1.0)
    finally:
        plt.close(figure)


def test_figure_composer_plot_slices_label_codegen_helper_variants(qtbot) -> None:
    operation = FigureOperationState.plot_slices(
        label="slices",
        sources=("a", "b"),
        slice_dim="eV",
        slice_values=(0.1, 0.2),
    ).model_copy(update={"line_label_text": "{source}:{number}:{eV:g}"})
    fields = {"source", "number", "eV"}
    single_key = (plot_slices_model._PlotSlicesPanelKey(0, 0, "A"),)
    by_slice_keys = tuple(
        plot_slices_model._PlotSlicesPanelKey(0, index, f"A {index}")
        for index in range(2)
    )
    by_source_keys = tuple(
        plot_slices_model._PlotSlicesPanelKey(index, 0, f"S {index}")
        for index in range(2)
    )
    grid_keys = tuple(
        plot_slices_model._PlotSlicesPanelKey(map_index, slice_index, "")
        for map_index in range(2)
        for slice_index in range(2)
    )
    namespace = {"slice_values": [0.1, 0.2]}

    single = plot_slices_codegen._plot_slices_label_line_kw_comprehension_code(
        operation, single_key, ("alpha",), "slice_values", fields
    )
    assert eval(single, namespace) == {"label": "alpha:1:0.1"}  # noqa: S307

    by_slice = plot_slices_codegen._plot_slices_label_line_kw_comprehension_code(
        operation, by_slice_keys, ("alpha",), "slice_values", fields
    )
    assert eval(by_slice, namespace) == [  # noqa: S307
        {"label": "alpha:1:0.1"},
        {"label": "alpha:2:0.2"},
    ]

    by_source = plot_slices_codegen._plot_slices_label_line_kw_comprehension_code(
        operation, by_source_keys, ("alpha", "beta"), "slice_values", fields
    )
    assert eval(by_source, namespace) == [  # noqa: S307
        {"label": "alpha:1:0.1"},
        {"label": "beta:2:0.1"},
    ]

    grid = plot_slices_codegen._plot_slices_label_line_kw_comprehension_code(
        operation, grid_keys, ("alpha", "beta"), "slice_values", fields
    )
    assert eval(grid, namespace) == [  # noqa: S307
        [{"label": "alpha:1:0.1"}, {"label": "alpha:2:0.2"}],
        [{"label": "beta:3:0.1"}, {"label": "beta:4:0.2"}],
    ]

    fortran_grid = plot_slices_codegen._plot_slices_label_line_kw_comprehension_code(
        operation.model_copy(update={"order": "F"}),
        grid_keys,
        ("alpha", "beta"),
        "slice_values",
        fields,
    )
    assert eval(fortran_grid, namespace) == [  # noqa: S307
        [{"label": "alpha:1:0.1"}, {"label": "beta:2:0.1"}],
        [{"label": "alpha:3:0.2"}, {"label": "beta:4:0.2"}],
    ]
    for code in (by_slice, by_source, grid, fortran_grid):
        available_names = {
            *_SCRIPT_REPLAY_ALLOWED_BUILTINS,
            "slice_values",
        }
        _validate_script_code_names(
            f"line_kw = {code}",
            available_names,
            {},
        )

    styled = plot_slices_codegen._plot_slices_styled_label_line_kw_code(
        operation.model_copy(
            update={
                "line_kw": {"linewidth": 1.5},
                "panel_styles": (
                    FigurePlotSlicesPanelStyleState(
                        map_index=1, slice_index=1, line_kw={"color": "red"}
                    ),
                ),
            }
        ),
        grid_keys,
        ("alpha", "beta"),
        "slice_values",
        {
            (1, 1): FigurePlotSlicesPanelStyleState(
                map_index=1, slice_index=1, line_kw={"color": "red"}
            )
        },
        fields,
    )
    styled_value = eval(styled, namespace)  # noqa: S307
    assert styled_value[1][1]["label"] == "beta:4:0.2"
    assert styled_value[1][1]["color"] == "red"
    assert styled_value[0][0]["linewidth"] == 1.5

    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("eV", "kx"),
        coords={"eV": [0.1, 0.2], "kx": [-1.0, 0.0, 1.0]},
        name="data",
    )
    styled_operation = operation.model_copy(
        update={
            "sources": ("a",),
            "panel_styles_enabled": True,
            "panel_styles": (
                FigurePlotSlicesPanelStyleState(
                    map_index=0, slice_index=1, line_kw={"color": "blue"}
                ),
            ),
        }
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="a", label="a"),),
            operations=(styled_operation,),
            primary_source="a",
        ),
        source_data={"a": data},
    )
    qtbot.addWidget(tool)
    styled_line_kw = plot_slices_codegen._plot_slices_label_line_kw_code(
        tool, styled_operation
    )
    assert styled_line_kw is not None
    styled_line_kw_value = eval(styled_line_kw, {}, namespace)  # noqa: S307
    assert styled_line_kw_value[0][1]["label"] == "a:2:0.2"
    assert styled_line_kw_value[0][1]["color"] == "blue"

    assert (
        plot_slices_codegen._plot_slices_panel_index_expr(
            "F",
            map_count=2,
            slice_count=3,
            map_index_expr="map_index",
            slice_index_expr="slice_index",
        )
        == "slice_index * 2 + map_index"
    )
    assert (
        plot_slices_codegen._line_kw_dict_code({"alpha": 0.5}, "label_code")
        == "{**{'alpha': 0.5}, 'label': label_code}"
    )
    assert (
        plot_slices_codegen._line_kw_dict_code({}, "label_code")
        == "{'label': label_code}"
    )


def test_figure_composer_plot_slices_line_transforms_codegen_executes(
    qtbot,
) -> None:
    data = xr.DataArray(
        np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]]),
        dims=("eV", "kx"),
        coords={"eV": [0.0, 1.0, 2.0], "kx": [0.0, 1.0]},
        name="data",
    )
    operation = FigureOperationState.plot_slices(
        label="line_slices",
        sources=("data",),
        axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
        slice_dim="eV",
        slice_values=(0.0, 1.0),
        slice_width=0.1,
    ).model_copy(
        update={
            "line_normalize": "max",
            "line_scales": (0.5, 2.0),
            "line_offsets": (1.0, -1.0),
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

    assert "transform" in tool.operation_editor.section_keys
    tool.operation_editor.select_section("transform")
    transform_page = tool.operation_editor.stack.currentWidget()
    assert transform_page.objectName() == "figureComposerPlotSlicesTransformPage"
    assert (
        transform_page.findChild(
            QtWidgets.QWidget, "figureComposerPlotSlicesLineTransformGroup"
        )
        is None
    )
    assert transform_page.findChild(
        QtWidgets.QComboBox, "figureComposerPlotSlicesLineNormalizeCombo"
    )
    assert transform_page.findChild(
        QtWidgets.QLineEdit, "figureComposerPlotSlicesLineScalesEdit"
    )
    assert transform_page.findChild(
        QtWidgets.QLineEdit, "figureComposerPlotSlicesLineOffsetsEdit"
    )
    assert (
        transform_page.findChild(QtWidgets.QCheckBox, "figureComposerGradientCheck")
        is None
    )

    tool.operation_editor.select_section("colors")
    colors_page = tool.operation_editor.stack.currentWidget()
    assert (
        colors_page.findChild(
            QtWidgets.QComboBox, "figureComposerPlotSlicesLineNormalizeCombo"
        )
        is None
    )
    assert colors_page.findChild(QtWidgets.QCheckBox, "figureComposerGradientCheck")

    kwargs = plot_slices_model._plot_slices_kwargs(tool, tool.tool_status.operations[0])
    assert "line_normalize" not in kwargs
    assert "line_scales" not in kwargs
    assert "line_offsets" not in kwargs

    code = tool.generated_code()
    assert "import xarray as xr" in code
    assert "data.qsel(eV=0.0, eV_width=0.1)" in code
    assert "data.qsel(eV=0.0, eV_width=0.1).squeeze(drop=True)" not in code
    assert "profile_scales =" not in code
    assert "profile_offsets =" not in code
    assert "[0.5, 2.0]," in code
    assert "[1.0, -1.0]," in code
    assert (
        "profiles = [\n    offset + scale * (profile / profile.max(skipna=True))"
    ) in code
    assert "xr.IndexVariable" not in code
    assert "plot_map =" not in code
    assert "plot_maps" not in code
    assert "eplt.plot_slices(\n    xr.concat(" in code
    assert 'dim="eV"' in code
    assert 'coords="different"' in code
    assert 'compat="equals"' in code
    assert '.assign_coords({"eV": [0.0, 1.0]})' in code
    assert "eV_width" not in code.split("eplt.plot_slices(", maxsplit=1)[1]

    namespace: dict[str, typing.Any] = {"data": data}
    exec(code, namespace)  # noqa: S102
    axs = namespace["axs"]
    np.testing.assert_allclose(axs[0, 0].lines[0].get_ydata(), [1.25, 1.5])
    np.testing.assert_allclose(axs[0, 1].lines[0].get_ydata(), [0.0, 1.0])


def test_figure_composer_plot_slices_image_panel_styles_codegen_executes(
    qtbot,
) -> None:
    import matplotlib.colors as mcolors

    data = xr.DataArray(
        np.arange(12.0).reshape(3, 2, 2),
        dims=("eV", "kx", "ky"),
        coords={"eV": [0.0, 1.0, 2.0], "kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    operation = FigureOperationState.plot_slices(
        label="image_panels",
        sources=("data",),
        axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
        slice_dim="eV",
        slice_values=(0.0, 1.0),
    ).model_copy(
        update={
            "panel_styles_enabled": True,
            "panel_styles": (
                FigurePlotSlicesPanelStyleState(
                    map_index=0,
                    slice_index=0,
                    cmap="viridis",
                    norm_name="Normalize",
                    vmin=0.0,
                    vmax=5.0,
                ),
                FigurePlotSlicesPanelStyleState(
                    map_index=0,
                    slice_index=1,
                    cmap="magma_r",
                    norm_name="CenteredPowerNorm",
                    norm_gamma=0.5,
                    halfrange=1.0,
                ),
            ),
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
        source_data={"data": data},
    )
    qtbot.addWidget(tool)

    kwargs = plot_slices_model._plot_slices_kwargs(tool, tool.tool_status.operations[0])
    assert kwargs["cmap"] == [["viridis", "magma_r"]]
    assert isinstance(kwargs["norm"][0][0], mcolors.Normalize)
    assert isinstance(kwargs["norm"][0][1], eplt.CenteredPowerNorm)

    namespace: dict[str, typing.Any] = {"data": data}
    exec(tool.generated_code(), namespace)  # noqa: S102
    axs = namespace["axs"]
    assert axs[0, 0].images[0].cmap.name == "viridis"
    assert axs[0, 1].images[0].cmap.name == "magma_r"
    assert isinstance(axs[0, 0].images[0].norm, mcolors.Normalize)
    assert isinstance(axs[0, 1].images[0].norm, eplt.CenteredPowerNorm)
    assert axs[0, 1].images[0].norm.gamma == 0.5
    assert axs[0, 1].images[0].norm.halfrange == 1.0


def test_figure_composer_plot_slices_line_panel_styles_codegen_executes(
    qtbot,
) -> None:
    data = xr.DataArray(
        np.arange(6.0).reshape(3, 2),
        dims=("eV", "kx"),
        coords={"eV": [0.0, 1.0, 2.0], "kx": [0.0, 1.0]},
        name="data",
    )
    operation = FigureOperationState.plot_slices(
        label="line_panels",
        sources=("data",),
        axes=FigureAxesSelectionState(axes=((0, 0), (1, 0))),
        slice_dim="eV",
        slice_values=(0.0, 1.0),
    ).model_copy(
        update={
            "order": "F",
            "panel_styles_enabled": True,
            "panel_styles": (
                FigurePlotSlicesPanelStyleState(
                    map_index=0,
                    slice_index=0,
                    line_kw={"color": "red", "linestyle": "--"},
                ),
                FigurePlotSlicesPanelStyleState(
                    map_index=0,
                    slice_index=1,
                    line_kw={"color": "blue", "marker": "o", "linewidth": 2.0},
                ),
            ),
        }
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=2, ncols=1),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
        source_data={"data": data},
    )
    qtbot.addWidget(tool)

    kwargs = plot_slices_model._plot_slices_kwargs(tool, tool.tool_status.operations[0])
    assert kwargs["line_kw"] == [
        [{"color": "red", "linestyle": "--"}],
        [{"color": "blue", "marker": "o", "linewidth": 2.0}],
    ]
    assert kwargs["line_order"] == "F"

    namespace: dict[str, typing.Any] = {"data": data}
    exec(tool.generated_code(), namespace)  # noqa: S102
    first_line = namespace["axs"][0, 0].lines[0]
    second_line = namespace["axs"][1, 0].lines[0]
    assert first_line.get_color() == "red"
    assert first_line.get_linestyle() == "--"
    assert second_line.get_color() == "blue"
    assert second_line.get_marker() == "o"
    assert second_line.get_linewidth() == 2.0


def test_figure_composer_dict_inputs_prefer_keyword_form(qtbot) -> None:
    data = xr.DataArray(
        np.arange(6.0).reshape(3, 2),
        dims=("eV", "kx"),
        coords={"eV": [0.0, 1.0, 2.0], "kx": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.plot_slices(
                    label="image",
                    sources=("data",),
                ).model_copy(
                    update={
                        "annotate_kw": {"fontsize": 8, "color": "black"},
                        "colorbar_kw": {"fraction": 0.05, "pad": 0.02},
                        "norm_kwargs": {"custom": "extra"},
                        "extra_kwargs": {"alpha": 0.5, "zorder": 2},
                    }
                ),
                FigureOperationState.plot_slices(
                    label="line",
                    sources=("data",),
                    slice_dim="eV",
                    slice_values=(0.0,),
                ).model_copy(
                    update={
                        "gradient_kw": {"color": "C0", "alpha": 0.25},
                    }
                ),
                FigureOperationState.line(
                    label="profile",
                    source="data",
                ).model_copy(update={"line_selection": {"eV": 0.0, "eV_width": 0.1}}),
                FigureOperationState.method(
                    family=FigureMethodFamily.ERLAB,
                    name="set_titles",
                ).model_copy(update={"method_kwargs": {"fontsize": 9}}),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )
    tool.operation_editor.select_section("view")
    annotate_kwargs_edit = tool.findChild(
        QtWidgets.QLineEdit, "figureComposerAnnotateKwEdit"
    )
    assert annotate_kwargs_edit is not None
    assert annotate_kwargs_edit.text() == 'fontsize=8, color="black"'

    tool.operation_editor.select_section("colors")
    colorbar_kwargs_edit = tool.findChild(
        QtWidgets.QLineEdit, "figureComposerColorbarKwEdit"
    )
    norm_kwargs_edit = tool.findChild(
        QtWidgets.QLineEdit, "figureComposerNormKwargsEdit"
    )
    assert colorbar_kwargs_edit is not None
    assert colorbar_kwargs_edit.text() == "fraction=0.05, pad=0.02"
    assert norm_kwargs_edit is not None
    assert norm_kwargs_edit.text() == 'custom="extra"'

    tool.operation_editor.select_section("advanced")
    extra_kwargs_edit = tool.findChild(QtWidgets.QLineEdit, "figureComposerExtraKwEdit")
    assert extra_kwargs_edit is not None
    assert extra_kwargs_edit.text() == "alpha=0.5, zorder=2"

    _select_operation_rows(tool, (1,))
    tool.operation_editor.select_section("colors")
    gradient_kwargs_edit = tool.findChild(
        QtWidgets.QLineEdit, "figureComposerGradientKwEdit"
    )
    assert gradient_kwargs_edit is not None
    assert gradient_kwargs_edit.text() == 'color="C0", alpha=0.25'

    _select_operation_rows(tool, (2,))
    tool.operation_editor.select_section("selection")
    selection_page = tool.operation_editor.stack.currentWidget()
    assert (
        selection_page.findChild(
            QtWidgets.QComboBox, "figureComposerProfileReduceCombo"
        )
        is None
    )
    line_selection_edit = selection_page.findChild(
        QtWidgets.QLineEdit, "figureComposerLineSelectionEdit"
    )
    assert line_selection_edit is not None
    assert line_selection_edit.text() == "eV=0.0, eV_width=0.1"
    line_selection_edit.setText("eV=slice(0.0, 1.0), kx=0.0")
    line_selection_edit.editingFinished.emit()
    assert tool.tool_status.operations[2].line_selection == {
        "eV": slice(0.0, 1.0),
        "kx": 0.0,
    }

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(3)
    )
    tool.operation_editor.select_section("method")
    erlab_method_kwargs_edit = tool.findChild(
        QtWidgets.QLineEdit, "figureComposerERLabMethodKwEdit"
    )
    assert erlab_method_kwargs_edit is not None
    assert erlab_method_kwargs_edit.text() == "fontsize=9"


def test_figure_composer_imagetool_norm_parser_uses_structured_fields() -> None:
    updates = figurecomposer_adapter._norm_updates(
        "|eplt.CenteredPowerNorm(0.5, vcenter=0.0, halfrange=1.0)|"
    )

    assert updates == {
        "norm_kwargs": {},
        "norm_name": "CenteredPowerNorm",
        "norm_gamma": 0.5,
        "vcenter": 0.0,
        "halfrange": 1.0,
    }


def test_figure_composer_imagetool_value_and_norm_parsers_cover_edges() -> None:
    class Floatable:
        def __float__(self) -> float:
            return 1.5

    fallback = object()
    assert figurecomposer_adapter._plain_value(None) is None
    assert figurecomposer_adapter._plain_value(np.bool_(True)) is True
    assert figurecomposer_adapter._plain_value([np.int64(1)]) == [1]
    assert figurecomposer_adapter._plain_value((np.float64(2.0),)) == (2.0,)
    assert figurecomposer_adapter._plain_value({"a": np.int64(3)}) == {"a": 3}
    assert figurecomposer_adapter._plain_value(np.int64(4)) == 4
    assert figurecomposer_adapter._plain_value(np.float64(5.0)) == 5.0
    assert figurecomposer_adapter._plain_value(Floatable()) == 1.5
    assert figurecomposer_adapter._plain_value(fallback) is fallback
    assert figurecomposer_adapter._indexer_state(slice(1, 3, 2)) == {
        "kind": "slice",
        "start": 1,
        "stop": 3,
        "step": 2,
    }
    assert figurecomposer_adapter._indexer_state(2) == 2

    invalid_norms = (
        "not valid python",
        "1",
        "eplt",
        "mcolors.PowerNorm(1)",
        "eplt.PowerNorm(1)",
        "eplt.CenteredPowerNorm(**kwargs)",
    )
    for norm_code in invalid_norms:
        assert figurecomposer_adapter._norm_updates(norm_code) is None

    assert figurecomposer_adapter._operation_updates({"norm": object()}) is None
    assert (
        figurecomposer_adapter._operation_updates({"norm": "eplt.PowerNorm(1)"}) is None
    )
    assert figurecomposer_adapter._operation_updates({"cmap": "|dynamic_cmap|"}) is None
    assert figurecomposer_adapter._operation_updates(
        {"gamma": np.float64(0.5), "alpha": np.int64(2)}
    ) == {
        "norm_name": "PowerNorm",
        "norm_gamma": 0.5,
        "extra_kwargs": {"alpha": 2},
    }


def test_figure_composer_plot_slices_spaced_qsel_dimension_codegen_executes(
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
        name="data",
    )
    operation = FigureOperationState.plot_slices(
        label="spaced",
        sources=("data",),
        slice_dim="Track Shift",
        slice_values=(1.0,),
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
        source_data={"data": data},
    )
    qtbot.addWidget(tool)

    _render_figure_composer_rgba(tool)
    code = tool.generated_code()
    assert "Track Shift" in code
    assert "**{" in code

    namespace: dict[str, typing.Any] = {"data": data}
    exec(code, namespace)  # noqa: S102
    assert len(namespace["axs"][0, 0].images) == 1


def test_figure_composer_powernorm_codegen_uses_plot_kwargs(qtbot) -> None:
    import matplotlib.colors as mcolors

    data = xr.DataArray(
        np.arange(12.0).reshape(3, 2, 2),
        dims=("eV", "kx", "ky"),
        coords={"eV": [0.0, 1.0, 2.0], "kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.plot_slices(
                    label="power",
                    sources=("data",),
                    slice_dim="eV",
                    slice_values=(0.0,),
                ).model_copy(
                    update={
                        "norm_name": "PowerNorm",
                        "norm_gamma": 0.5,
                        "vmin": 0.0,
                        "vmax": 10.0,
                    }
                ),
            ),
            primary_source="data",
        ),
        source_data={"data": data},
    )
    qtbot.addWidget(tool)

    code = tool.generated_code()
    assert "import matplotlib.colors as mcolors" not in code
    assert "norm=mcolors.PowerNorm" not in code
    assert "gamma=0.5" in code
    assert "vmin=0.0" in code
    assert "vmax=10.0" in code

    namespace: dict[str, typing.Any] = {"data": data}
    exec(code, namespace)  # noqa: S102
    norm = namespace["axs"][0, 0].images[0].norm
    assert isinstance(norm, mcolors.PowerNorm)
    assert norm.gamma == 0.5
    assert norm.vmin == 0.0
    assert norm.vmax == 10.0


def test_figure_composer_explicit_norm_codegen_executes(qtbot) -> None:
    import matplotlib.colors as mcolors

    import erlab.plotting as eplt

    data = xr.DataArray(
        np.arange(24.0).reshape(3, 4, 2),
        dims=("eV", "kx", "ky"),
        coords={
            "eV": [0.0, 1.0, 2.0],
            "kx": [0.0, 1.0, 2.0, 3.0],
            "ky": [0.0, 1.0],
        },
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.plot_slices(
                    label="power",
                    sources=("data",),
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                    slice_dim="eV",
                    slice_values=(0.0,),
                ).model_copy(
                    update={
                        "norm_name": "PowerNorm",
                        "norm_gamma": 0.5,
                        "vmin": 0.0,
                        "vmax": 10.0,
                        "norm_clip": True,
                    }
                ),
                FigureOperationState.plot_slices(
                    label="inverse",
                    sources=("data",),
                    axes=FigureAxesSelectionState(axes=((0, 1),)),
                    slice_dim="eV",
                    slice_values=(1.0,),
                ).model_copy(
                    update={
                        "norm_name": "InversePowerNorm",
                        "norm_gamma": 0.75,
                        "vmin": 0.0,
                        "vmax": 20.0,
                    }
                ),
            ),
            primary_source="data",
        ),
        source_data={"data": data},
    )
    qtbot.addWidget(tool)

    code = tool.generated_code()
    assert "import matplotlib.colors as mcolors" in code
    assert "norm=mcolors.PowerNorm" in code
    namespace = {"data": data}
    exec(code, namespace)  # noqa: S102

    left_norm = namespace["axs"][0, 0].images[0].norm
    right_norm = namespace["axs"][0, 1].images[0].norm
    assert isinstance(left_norm, mcolors.PowerNorm)
    assert left_norm.gamma == 0.5
    assert left_norm.vmin == 0.0
    assert left_norm.vmax == 10.0
    assert left_norm.clip is True
    assert isinstance(right_norm, eplt.InversePowerNorm)
    assert right_norm.gamma == 0.75
    assert right_norm.vmin == 0.0
    assert right_norm.vmax == 20.0
