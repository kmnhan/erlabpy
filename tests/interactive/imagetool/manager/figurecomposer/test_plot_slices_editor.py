"""Plot Slices editor behavior tests."""

from ._plot_slices_common import (
    Figure,
    FigureAxesSelectionState,
    FigureComposerTool,
    FigureDataSelectionState,
    FigureDocument,
    FigureMethodFamily,
    FigureOperationState,
    FigurePlotSlicesPanelStyleState,
    FigureRecipeState,
    FigureSourceState,
    FigureSubplotsState,
    QtCore,
    QtWidgets,
    _method_float_pair_args,
    _method_spec,
    _plot_source_checks,
    _plot_source_move_buttons,
    eplt,
    figurecomposer_norms,
    figurecomposer_rendering,
    figurecomposer_text,
    figurecomposer_tool_module,
    mpl,
    np,
    options,
    plot_slices_codegen,
    plot_slices_editor,
    plot_slices_model,
    plot_slices_panel_style_editor,
    plot_slices_render,
    plot_slices_spec,
    plt,
    pytest,
    typing,
    warnings,
    xr,
)


def test_figure_composer_plot_slices_panel_helpers_cover_style_contract(
    qtbot,
) -> None:
    source = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("eV", "kx"),
        coords={
            "eV": [0.0, 1.0],
            "kx": [-1.0, 0.0, 1.0],
            "temperature": ("eV", [20.0, 30.0]),
        },
        name="line_map",
    )
    tool = FigureComposerTool(
        source,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=2),
            sources=(FigureSourceState(name="data", label="line_map"),),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    operation = FigureOperationState.plot_slices(
        label="selection",
        sources=("data",),
        slice_dim="eV",
        slice_values=(0.0, 1.0),
    ).model_copy(
        update={
            "cmap": "viridis",
            "line_kw": {"linewidth": 1.5},
            "panel_styles_enabled": True,
            "panel_styles": (
                FigurePlotSlicesPanelStyleState(
                    map_index=0,
                    slice_index=0,
                    cmap="magma",
                    norm_name="Normalize",
                    vmin=0.0,
                    vmax=5.0,
                    line_kw={"color": "red"},
                ),
                FigurePlotSlicesPanelStyleState(
                    map_index=0,
                    slice_index=1,
                    cmap="plasma",
                    norm_name="PowerNorm",
                    norm_gamma=0.5,
                    line_kw={"color": "blue"},
                ),
            ),
            "line_normalize": "max",
            "line_offset_source": "coordinate",
            "line_iter_dim": "eV",
        }
    )

    keys = plot_slices_model._plot_slices_panel_keys(
        tool._document, tool._source_display_name, operation
    )
    assert [(key.map_index, key.slice_index) for key in keys] == [(0, 0), (0, 1)]
    assert plot_slices_model._plot_slices_slice_count(tool._document, operation) == 2
    assert plot_slices_model._plot_slices_slice_labels(operation, 2) == (
        "eV=0",
        "eV=1",
    )
    assert plot_slices_model._panel_cmap_argument(tool, operation) == [
        ["magma", "plasma"]
    ]
    assert plot_slices_model._effective_panel_cmap(
        FigureOperationState.plot_slices(label="default", sources=("data",)),
        FigurePlotSlicesPanelStyleState(map_index=0, slice_index=0),
    ) == figurecomposer_norms._matplotlib_cmap_name(options.model.colors.cmap.name)
    assert plot_slices_model._panel_line_kw_argument(tool, operation) == [
        [{"linewidth": 1.5, "color": "red"}, {"linewidth": 1.5, "color": "blue"}]
    ]
    assert plot_slices_model._has_panel_line_kw_overrides(tool, operation)
    norm_argument = plot_slices_model._panel_norm_argument(tool, operation)
    assert isinstance(norm_argument, list)
    assert plot_slices_codegen._panel_norm_uses_matplotlib_colors(tool, operation)
    assert "mcolors.Normalize" in (
        plot_slices_codegen._panel_norm_code(tool, operation) or ""
    )

    profiles, profile_keys = plot_slices_model._plot_slices_line_profiles(
        tool._document,
        tool._source_display_name,
        operation,
        maps=(source,),
    )
    assert len(profiles) == 2
    assert [(key.map_index, key.slice_index) for key in profile_keys] == [
        (0, 0),
        (0, 1),
    ]
    assert plot_slices_model._plot_slices_uses_transformed_line_maps(tool, operation)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="In a future version of xarray the default value for .*",
            category=FutureWarning,
        )
        transformed_maps = plot_slices_model._plot_slices_transformed_maps(
            tool,
            operation,
            (source,),
        )
    assert len(transformed_maps) == 1
    assert transformed_maps[0].dims == ("eV", "kx")
    assert set(
        plot_slices_editor._available_plot_slices_offset_coords(
            tool.operation_editor, operation
        )
    ) >= {"eV", "temperature"}

    code_operation = operation.model_copy(
        update={"axes": FigureAxesSelectionState(axes=((0, 0), (0, 1)))}
    )
    fig, axs = plt.subplots(1, 2, squeeze=False)
    namespace: dict[str, typing.Any] = {
        "data": source,
        "eplt": eplt,
        "xr": xr,
        "axs": axs,
    }
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="In a future version of xarray the default value for .*",
            category=FutureWarning,
        )
        exec(  # noqa: S102
            "\n".join(
                plot_slices_codegen._plot_slices_transformed_code_lines(
                    tool, code_operation
                )
            ),
            namespace,
        )
    assert len(axs[0, 0].lines) == 1
    assert len(axs[0, 1].lines) == 1
    plt.close(fig)

    single_panel_operation = operation.model_copy(update={"slice_values": (0.0,)})
    assert (
        plot_slices_model._panel_cmap_argument(tool, single_panel_operation) == "magma"
    )
    single_norm = plot_slices_model._panel_norm_argument(tool, single_panel_operation)
    assert isinstance(single_norm, mpl.colors.Normalize)
    assert plot_slices_model._panel_line_kw_argument(tool, single_panel_operation) == {
        "linewidth": 1.5,
        "color": "red",
    }

    no_override_operation = operation.model_copy(
        update={"panel_styles_enabled": False, "panel_styles": ()}
    )
    assert plot_slices_model._panel_norm_argument(tool, no_override_operation) is None
    assert plot_slices_codegen._panel_norm_code(tool, no_override_operation) is None
    assert plot_slices_model._panel_line_kw_argument(tool, no_override_operation) == {
        "linewidth": 1.5
    }

    same_cmap_operation = operation.model_copy(
        update={
            "panel_styles": (
                FigurePlotSlicesPanelStyleState(
                    map_index=0, slice_index=0, cmap="magma"
                ),
                FigurePlotSlicesPanelStyleState(
                    map_index=0, slice_index=1, cmap="magma"
                ),
            )
        }
    )
    assert plot_slices_model._panel_cmap_argument(tool, same_cmap_operation) == "magma"
    same_line_operation = operation.model_copy(
        update={
            "panel_styles": (
                FigurePlotSlicesPanelStyleState(
                    map_index=0, slice_index=0, line_kw={"color": "red"}
                ),
                FigurePlotSlicesPanelStyleState(
                    map_index=0, slice_index=1, line_kw={"color": "red"}
                ),
            )
        }
    )
    assert plot_slices_model._panel_line_kw_argument(tool, same_line_operation) == {
        "linewidth": 1.5,
        "color": "red",
    }

    selection_operation = operation.model_copy(
        update={
            "slice_dim": None,
            "slice_values": (),
            "slice_kwargs": {"kx": [-1.0, 0.0], "kx_width": 0.1},
        }
    )
    assert plot_slices_model._plot_slices_slice_labels(selection_operation, 2) == (
        "kx[0]",
        "kx[1]",
    )

    no_override_operation = operation.model_copy(
        update={
            "panel_styles_enabled": True,
            "panel_styles": (
                FigurePlotSlicesPanelStyleState(map_index=0, slice_index=0),
            ),
        }
    )
    assert (
        plot_slices_model._panel_cmap_argument(tool, no_override_operation) == "viridis"
    )
    assert plot_slices_model._panel_norm_argument(tool, no_override_operation) is None
    assert plot_slices_codegen._panel_norm_code(tool, no_override_operation) is None
    assert not plot_slices_codegen._panel_norm_uses_matplotlib_colors(
        tool, no_override_operation
    )
    assert plot_slices_model._panel_line_kw_argument(tool, no_override_operation) == {
        "linewidth": 1.5
    }


def test_figure_composer_plot_slices_edge_helper_contracts(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        figurecomposer_tool_module,
        "_render_preview",
        lambda *_args, **_kwargs: None,
    )
    image = xr.DataArray(
        np.arange(24.0).reshape(2, 3, 4),
        dims=("eV", "kx", "ky"),
        coords={"eV": [0.0, 1.0], "kx": [-1.0, 0.0, 1.0], "ky": range(4)},
        name="image",
    )
    line = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("eV", "kx"),
        coords={"eV": [0.0, 1.0], "kx": [-1.0, 0.0, 1.0]},
        name="line",
    )
    other = xr.DataArray(
        np.arange(8.0).reshape(2, 4),
        dims=("eV", "phi"),
        coords={"eV": [0.0, 1.0], "phi": range(4)},
        name="other",
    )
    line_operation = FigureOperationState.plot_slices(
        label="line",
        sources=("line",),
        slice_dim="eV",
        slice_values=(0.0, 1.0),
    ).model_copy(
        update={
            "line_kw": {"linewidth": 1.5},
            "panel_styles_enabled": True,
            "panel_styles": (
                FigurePlotSlicesPanelStyleState(
                    map_index=0,
                    slice_index=0,
                    line_kw={"color": "red"},
                ),
                FigurePlotSlicesPanelStyleState(
                    map_index=0,
                    slice_index=1,
                    line_kw={"color": "blue"},
                ),
            ),
            "order": "F",
            "gradient": True,
            "gradient_kw": {"alpha": 0.2},
            "line_normalize": "mean",
        }
    )
    image_operation = FigureOperationState.plot_slices(
        label="image",
        sources=("image",),
        slice_dim="eV",
        slice_values=(0.0, 1.0),
        axes=FigureAxesSelectionState(expression="axs[0, :]"),
    ).model_copy(
        update={
            "transpose": True,
            "xlim": (-1.0, None),
            "ylim": 0.5,
            "crop": False,
            "same_limits": True,
            "axis": "x",
            "show_all_labels": True,
            "colorbar": "right",
            "hide_colorbar_ticks": False,
            "annotate": False,
            "cmap": "magma",
            "norm_name": "PowerNorm",
            "norm_gamma": 0.5,
            "vmin": 0.0,
            "vmax": 10.0,
            "order": "F",
            "cmap_order": "F",
            "norm_order": "F",
            "subplot_kw": {"sharex": True},
            "annotate_kw": {"fontsize": 8},
            "colorbar_kw": {"ticks": [0.0, 1.0]},
            "extra_kwargs": {"alpha": 0.9},
        }
    )
    tool = FigureComposerTool(
        image,
        recipe=FigureRecipeState(
            sources=(
                FigureSourceState(name="image", label="image"),
                FigureSourceState(name="line", label="line"),
                FigureSourceState(name="other", label="other"),
            ),
            operations=(line_operation, image_operation),
            primary_source="image",
        ),
        source_data={"image": image, "line": line, "other": other},
    )
    qtbot.addWidget(tool)

    assert (
        plot_slices_model._plot_slices_batch_panel_kind(
            tool._document,
            ((0, line_operation), (1, image_operation)),
            line_operation,
        )
        == "mixed"
    )
    assert (
        plot_slices_model._plot_slices_batch_panel_kind(
            tool._document, (), line_operation
        )
        == "line"
    )

    keys = plot_slices_model._plot_slices_panel_keys(
        tool._document, tool._source_display_name, line_operation
    )
    assert [(key.map_index, key.slice_index) for key in keys] == [(0, 0), (0, 1)]
    assert plot_slices_model._plot_slices_slice_labels(
        line_operation.model_copy(update={"slice_values": ()}),
        2,
    ) == ("slice 1", "slice 2")
    slice_kwarg_operation = line_operation.model_copy(
        update={
            "slice_dim": None,
            "slice_values": (),
            "slice_kwargs": {"eV": [0.0, 1.0], "eV_width": 0.2},
        }
    )
    assert (
        plot_slices_model._plot_slices_slice_count(
            tool._document, slice_kwarg_operation
        )
        == 2
    )
    shape = plot_slices_model._plot_slices_shape(tool._document, slice_kwarg_operation)
    assert shape.valid
    assert shape.panel_count == 2
    range_shape = plot_slices_model._plot_slices_shape(
        tool._document,
        line_operation.model_copy(
            update={
                "slice_dim": None,
                "slice_values": (),
                "slice_kwargs": {"kx": slice(-1.0, 1.0), "eV": 0.0},
            }
        ),
    )
    assert range_shape.valid

    missing_shape = plot_slices_model._plot_slices_shape(
        tool._document,
        FigureOperationState.plot_slices(label="missing", sources=("missing",)),
    )
    assert not missing_shape.valid
    mismatched_shape = plot_slices_model._plot_slices_shape(
        tool._document,
        FigureOperationState.plot_slices(label="mixed", sources=("line", "other")),
    )
    assert not mismatched_shape.valid
    invalid_cut_shape = plot_slices_model._plot_slices_shape(
        tool._document,
        line_operation.model_copy(update={"slice_dim": "missing", "slice_values": ()}),
    )
    assert invalid_cut_shape.valid
    incomplete_cut_shape = plot_slices_model._plot_slices_shape(
        tool._document,
        line_operation.model_copy(update={"slice_values": ()}),
    )
    assert incomplete_cut_shape.valid

    image_kwargs = plot_slices_model._plot_slices_kwargs(tool, image_operation)
    assert image_kwargs["transpose"] is True
    assert image_kwargs["xlim"] == (-1.0, None)
    assert image_kwargs["ylim"] == 0.5
    assert image_kwargs["crop"] is False
    assert image_kwargs["same_limits"] is True
    assert image_kwargs["axis"] == "x"
    assert image_kwargs["show_all_labels"] is True
    assert image_kwargs["colorbar"] == "right"
    assert image_kwargs["hide_colorbar_ticks"] is False
    assert image_kwargs["annotate"] is False
    assert image_kwargs["cmap"] == "magma"
    assert image_kwargs["gamma"] == 0.5
    assert image_kwargs["vmin"] == 0.0
    assert image_kwargs["vmax"] == 10.0
    assert image_kwargs["order"] == "F"
    assert image_kwargs["cmap_order"] == "F"
    assert image_kwargs["norm_order"] == "F"

    set_xlim_operation = FigureOperationState.method(
        family=FigureMethodFamily.AXES,
        name="set_xlim",
        args=(0.0, None),
    )
    assert _method_float_pair_args(
        tool.operation_editor,
        set_xlim_operation,
        _method_spec(set_xlim_operation),
    ) == (0.0, None)
    assert image_kwargs["subplot_kw"] == {"sharex": True}
    assert image_kwargs["annotate_kw"] == {"fontsize": 8}
    assert image_kwargs["colorbar_kw"] == {"ticks": [0.0, 1.0]}
    assert image_kwargs["alpha"] == 0.9

    explicit_norm_kwargs = plot_slices_model._plot_slices_kwargs(
        tool,
        image_operation.model_copy(
            update={"norm_name": "Normalize", "norm_gamma": None}
        ),
    )
    assert "norm" in explicit_norm_kwargs
    panel_norm_kwargs = plot_slices_model._plot_slices_kwargs(
        tool,
        image_operation.model_copy(
            update={
                "panel_styles_enabled": True,
                "panel_styles": (
                    FigurePlotSlicesPanelStyleState(
                        map_index=0,
                        slice_index=0,
                        norm_name="Normalize",
                    ),
                ),
            }
        ),
    )
    assert "norm" in panel_norm_kwargs

    line_kwargs = plot_slices_model._plot_slices_kwargs(tool, line_operation)
    assert line_kwargs["line_kw"] == [
        [{"linewidth": 1.5, "color": "red"}],
        [{"linewidth": 1.5, "color": "blue"}],
    ]
    assert line_kwargs["line_order"] == "F"
    assert line_kwargs["gradient"] is True
    assert line_kwargs["gradient_kw"] == {"alpha": 0.2}
    transformed_kwargs = plot_slices_model._plot_slices_transformed_kwargs(
        tool,
        line_operation,
    )
    assert "eV_width" not in transformed_kwargs
    assert transformed_kwargs["eV"] == [0.0, 1.0]

    flat_axes = np.empty(4, dtype=object)
    reshaped_axes = plot_slices_render._plot_slices_axes(
        line_operation.model_copy(update={"sources": ("line", "other")}),
        (line, other),
        flat_axes,
    )
    assert isinstance(reshaped_axes, np.ndarray)
    assert reshaped_axes.shape == (2, 2)
    mismatched_axes = np.empty(3, dtype=object)
    assert (
        plot_slices_render._plot_slices_axes(
            line_operation,
            (line,),
            mismatched_axes,
        )
        is mismatched_axes
    )
    assert (
        plot_slices_render._plot_slices_axes(line_operation, (line,), object())
        is not flat_axes
    )

    selection_operation = FigureOperationState.plot_slices(
        label="selection",
        sources=("image",),
        map_selections=(
            FigureDataSelectionState(source="image", isel={"eV": 0}),
            FigureDataSelectionState(source="image", qsel={"eV": 1.0}),
        ),
    )
    assert (
        len(plot_slices_model._operation_maps(tool._document, selection_operation)) == 1
    )
    selection_lines = plot_slices_codegen._plot_slices_code_lines(
        tool,
        selection_operation,
    )
    assert all("selected_maps" not in line for line in selection_lines)
    assert any("eplt.plot_slices" in line for line in selection_lines)
    single_selection_operation = FigureOperationState.plot_slices(
        label="single_selection",
        sources=("image",),
        map_selections=(FigureDataSelectionState(source="image", qsel={"eV": 1.0}),),
    )
    single_selection_lines = plot_slices_codegen._plot_slices_code_lines(
        tool,
        single_selection_operation,
    )
    assert len(single_selection_lines) == 1
    assert "selected_maps" not in single_selection_lines[0]
    assert "eplt.plot_slices(image" in single_selection_lines[0]
    assert (
        plot_slices_codegen._plot_slices_code_lines(
            tool,
            FigureOperationState.plot_slices(label="empty", sources=()),
        )
        == []
    )

    transform_lines = plot_slices_codegen._plot_slices_transformed_code_lines(
        tool,
        line_operation,
    )
    assert transform_lines[0] == "profiles = ["
    assert any("eplt.plot_slices" in line for line in transform_lines)
    no_slice_map_lines, no_slice_maps_code = (
        plot_slices_codegen._plot_slices_transformed_maps_code(
            line_operation.model_copy(update={"slice_dim": None, "slice_values": ()}),
            keys[:1],
        )
    )
    assert no_slice_map_lines == []
    assert no_slice_maps_code == "profiles[0]"

    assert plot_slices_editor._bool_or_text("True") is True
    assert plot_slices_editor._bool_or_text("False") is False
    assert plot_slices_editor._bool_or_text("row") == "row"
    assert plot_slices_editor._optional_number_or_text("vmin", "") is None
    assert plot_slices_editor._optional_number_or_text("cmap", "magma") == "magma"
    assert plot_slices_editor._optional_number_or_text("vmax", "1.5") == 1.5
    assert (
        plot_slices_editor._norm_field_placeholder(
            image_operation.model_copy(update={"norm_name": "CenteredPowerNorm"}),
            "vcenter",
        )
        == "0"
    )
    assert (
        plot_slices_editor._norm_field_placeholder(
            image_operation.model_copy(update={"vcenter": 1.0}),
            "vcenter",
        )
        == ""
    )
    placeholder_operation = FigureOperationState.plot_slices(
        label="image",
        sources=("image",),
        slice_dim="eV",
        slice_values=(0.0,),
        axes=FigureAxesSelectionState(axes=((0, 0),)),
    )
    placeholder_tool = FigureComposerTool(
        image,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="image", label="image"),),
            operations=(placeholder_operation,),
            primary_source="image",
        ),
        source_data={"image": image},
    )
    qtbot.addWidget(placeholder_tool)
    placeholder_tool.show_figure_window(activate=False)
    figurecomposer_rendering._render_preview(placeholder_tool, show_window=True)
    assert plot_slices_editor._plot_slices_color_limit_placeholders(
        placeholder_tool.operation_editor,
        placeholder_operation,
    ) == {"vmin": "0", "vmax": "11"}
    assert (
        plot_slices_editor._norm_gamma_value(
            image_operation.model_copy(update={"norm_gamma": None, "gamma": None})
        )
        == 1.0
    )
    assert plot_slices_model._norm_clip_text(None) == "default"
    assert plot_slices_model._norm_clip_from_text("True") is True
    assert plot_slices_model._norm_clip_from_text("False") is False
    assert plot_slices_model._norm_clip_from_text("default") is None

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(1)
    )
    plot_slices_editor._update_current_norm_name(
        tool.operation_editor, "CenteredPowerNorm"
    )
    assert tool.tool_status.operations[1].norm_name == "CenteredPowerNorm"
    plot_slices_editor._update_current_norm_gamma(tool.operation_editor, 0.75)
    assert tool.tool_status.operations[1].norm_gamma == 0.75
    plot_slices_editor._update_current_norm_kwargs(
        tool.operation_editor,
        "halfrange=2.0, clip=True, custom=1",
    )
    assert tool.tool_status.operations[1].halfrange == 2.0
    assert tool.tool_status.operations[1].norm_clip is True
    assert tool.tool_status.operations[1].norm_kwargs == {"custom": 1}
    plot_slices_editor._update_current_slice_kwargs(
        tool.operation_editor,
        "eV=[0, 1], eV_width=0.2",
    )
    assert tool.tool_status.operations[1].slice_dim == "eV"
    assert tool.tool_status.operations[1].slice_width == 0.2
    plot_slices_editor._update_current_extra_kwargs(
        tool.operation_editor,
        "kx=0.0, alpha=0.5",
    )
    assert tool.tool_status.operations[1].slice_kwargs["kx"] == 0.0
    assert tool.tool_status.operations[1].extra_kwargs == {"alpha": 0.5}
    plot_slices_editor._update_current_cmap(
        tool.operation_editor, base="viridis", reverse=True
    )
    assert tool.tool_status.operations[1].cmap == "viridis_r"
    plot_slices_editor._update_current_panel_styles_enabled(
        tool.operation_editor, False
    )
    assert not tool.tool_status.operations[1].panel_styles_enabled
    plot_slices_editor._update_current_panel_styles(
        tool.operation_editor,
        (FigurePlotSlicesPanelStyleState(map_index=0, slice_index=0, cmap="plasma"),),
    )
    assert tool.tool_status.operations[1].panel_styles_enabled
    assert tool.tool_status.operations[1].panel_styles[0].cmap == "plasma"
    plot_slices_editor._update_current_panel_styles(tool.operation_editor, ())
    assert not tool.tool_status.operations[1].panel_styles_enabled
    assert tool.tool_status.operations[1].panel_styles == ()


def test_figure_composer_plot_slices_all_coordinate_values_with_thin(
    qtbot,
) -> None:
    data = xr.DataArray(
        np.arange(5 * 3 * 2.0).reshape(5, 3, 2),
        dims=("eV", "kx", "ky"),
        coords={
            "eV": np.linspace(-1.0, 1.0, 5),
            "kx": [0.0, 1.0, 2.0],
            "ky": [10.0, 20.0],
        },
        name="data",
    )
    operation = FigureOperationState.plot_slices(
        label="image",
        sources=("data",),
        slice_dim="eV",
        slice_values=(0.0,),
        axes=FigureAxesSelectionState(expression="axs"),
    ).model_copy(
        update={
            "slice_values_mode": "all",
            "slice_values_thin": 2,
        }
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=3),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
        source_data={"data": data},
    )
    qtbot.addWidget(tool)

    expected_values = data.thin({"eV": 2}).coords["eV"].values
    kwargs = plot_slices_model._plot_slices_kwargs(tool, operation)
    np.testing.assert_allclose(kwargs["eV"], expected_values)
    shape = plot_slices_model._plot_slices_shape(tool._document, operation)
    assert shape.panel_count == expected_values.size
    assert shape.plot_dims == ("kx", "ky")

    code_kwargs = plot_slices_codegen._plot_slices_code_kwargs(tool, operation)
    assert isinstance(code_kwargs["eV"], figurecomposer_text._RawCode)
    captured: list[dict[str, typing.Any]] = []

    class PlotSlicesCapture:
        @staticmethod
        def plot_slices(_maps, **plot_kwargs):
            captured.append(plot_kwargs)

    exec(  # noqa: S102
        "\n".join(plot_slices_codegen._plot_slices_code_lines(tool, operation)),
        {
            "data": data,
            "eplt": PlotSlicesCapture,
            "axs": object(),
        },
    )
    assert len(captured) == 1
    np.testing.assert_allclose(captured[0]["eV"], expected_values)

    full_values_operation = operation.model_copy(update={"slice_values_thin": 1})
    full_kwargs = plot_slices_model._plot_slices_kwargs(
        tool,
        full_values_operation,
    )
    np.testing.assert_allclose(full_kwargs["eV"], data.coords["eV"].values)

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )
    tool._update_operation_editor()
    tool.operation_editor.select_section("selection")
    selection_page = tool.operation_editor.stack.currentWidget()
    assert selection_page is not None
    values_edit = selection_page.findChild(
        QtWidgets.QLineEdit, "figureComposerPlotSlicesValuesEdit"
    )
    assert values_edit is None
    coordinate_summary = selection_page.findChild(
        QtWidgets.QLabel, "figureComposerPlotSlicesCoordinateSummary"
    )
    assert coordinate_summary is not None
    thin_spin = selection_page.findChild(
        QtWidgets.QAbstractSpinBox, "figureComposerPlotSlicesValuesThinSpin"
    )
    assert thin_spin is not None
    assert thin_spin.isEnabled()
    assert typing.cast("typing.Any", thin_spin).value() == 2


def test_figure_composer_plot_slices_shape_and_source_editor_contracts(
    qtbot,
    monkeypatch,
) -> None:
    first = xr.DataArray(
        np.arange(24.0).reshape(2, 3, 4),
        dims=("eV", "kx", "ky"),
        coords={"eV": [0.0, 1.0], "kx": [0.0, 1.0, 2.0], "ky": range(4)},
        name="first",
    )
    second = xr.DataArray(
        np.arange(12.0).reshape(2, 6),
        dims=("eV", "kz"),
        coords={"eV": [0.0, 1.0], "kz": range(6)},
        name="second",
    )
    first_operation = FigureOperationState.plot_slices(
        label="first",
        sources=("first",),
        axes=FigureAxesSelectionState(axes=((0, 0),), expression="axs[0, 0]"),
    ).model_copy(
        update={
            "transpose": True,
            "slice_kwargs": {
                "eV": slice(0.0, 1.0),
                "kx": 1.0,
                "ky": [0.0, 1.0],
                "ky_width": 0.2,
            },
        }
    )
    second_operation = FigureOperationState.plot_slices(
        label="second",
        sources=("second",),
        axes=FigureAxesSelectionState(axes=((0, 0),)),
    )
    tool = FigureComposerTool(
        first,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=1),
            sources=(
                FigureSourceState(name="first", label="first"),
                FigureSourceState(name="second", label="second"),
            ),
            operations=(first_operation, second_operation),
            primary_source="first",
        ),
        source_data={"first": first, "second": second},
    )
    qtbot.addWidget(tool)

    shape = plot_slices_model._plot_slices_shape(tool._document, first_operation)
    assert shape.source_text == "eV, kx, ky"
    assert shape.panel_text == "eV (1D line)"
    assert shape.selection_text == ""
    assert shape.plot_ndim == 1
    assert shape.panel_count == 2
    assert shape.valid
    invalid_shape = plot_slices_model._plot_slices_shape(
        tool._document,
        first_operation.model_copy(
            update={
                "slice_kwargs": {"eV": 0.0, "kx": 1.0, "ky": 2.0},
            }
        ),
    )
    assert invalid_shape.plot_ndim == 0
    assert not invalid_shape.valid
    assert (
        plot_slices_spec._section_summary(tool, "selection", first_operation)
        == "additional"
    )
    assert plot_slices_spec._section_summary(tool, "view", first_operation) == "auto"
    assert plot_slices_spec._section_summary(tool, "advanced", first_operation) == ""
    assert plot_slices_spec._section_summary(tool, "unknown", first_operation) == ""

    mixed_operation = first_operation.model_copy(
        update={"sources": ("first", "second")}
    )
    mixed_shape = plot_slices_model._plot_slices_shape(tool._document, mixed_operation)
    assert not mixed_shape.valid
    assert mixed_shape.plot_ndim is None

    empty_tool = FigureComposerTool(
        first,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(),
            sources=(),
            operations=(
                FigureOperationState.plot_slices(label="missing", sources=("missing",)),
            ),
            primary_source="missing",
        ),
        source_data={},
    )
    qtbot.addWidget(empty_tool)
    empty_tool._document.replace_source_payloads({}, {})
    empty_shape = plot_slices_model._plot_slices_shape(
        empty_tool._document, empty_tool.tool_status.operations[0]
    )
    assert not empty_shape.valid
    assert empty_shape.panel_count == 0

    with monkeypatch.context() as context:
        context.setattr(
            tool,
            "_editable_operations",
            lambda: ((0, first_operation), (1, second_operation)),
        )
        assert (
            plot_slices_editor._plot_source_check_state(
                tool.operation_editor, first_operation, "first"
            )
            == QtCore.Qt.CheckState.PartiallyChecked
        )

    checks = _plot_source_checks(tool)
    assert set(checks) == {"first", "second"}
    assert checks["first"].checkState() == QtCore.Qt.CheckState.Checked
    assert checks["second"].checkState() == QtCore.Qt.CheckState.Unchecked

    checks["second"].setCheckState(QtCore.Qt.CheckState.Checked)
    assert tool.tool_status.operations[0].sources == ("first", "second")

    _plot_source_checks(tool)["first"].setCheckState(QtCore.Qt.CheckState.Unchecked)
    assert tool.tool_status.operations[0].sources == ("second",)

    _plot_source_checks(tool)["first"].setCheckState(QtCore.Qt.CheckState.Checked)
    assert tool.tool_status.operations[0].sources == ("first", "second")

    qtbot.waitUntil(
        lambda: _plot_source_move_buttons(tool)[("first", "down")].isEnabled(),
        timeout=1000,
    )
    _plot_source_move_buttons(tool)[("first", "down")].click()
    qtbot.waitUntil(
        lambda: tool.tool_status.operations[0].sources[:2] == ("second", "first"),
        timeout=1000,
    )


def test_figure_composer_plot_slices_panel_override_controls_stay_live(
    qtbot,
) -> None:
    operation = FigureOperationState.plot_slices(
        label="image",
        sources=("data",),
    ).model_copy(update={"cmap": "viridis", "norm_name": "PowerNorm"})
    keys = (plot_slices_model._PlotSlicesPanelKey(0, 0, "panel 1"),)
    editor = plot_slices_panel_style_editor._PanelStyleEditorWidget(
        operation,
        keys,
        lambda _owner, signal, slot: signal.connect(slot),
        "viridis",
    )
    qtbot.addWidget(editor)
    emitted: list[tuple[FigurePlotSlicesPanelStyleState, ...]] = []
    editor.sigPanelStylesChanged.connect(emitted.append)

    editor.cmap_combo.activated.emit(editor.cmap_combo.currentIndex())
    editor.cmap_reverse_check.setCheckState(QtCore.Qt.CheckState.Checked)
    editor.norm_combo.activated.emit(editor.norm_combo.currentIndex())
    editor.gamma_edit.setText("2")
    editor.gamma_edit.setModified(True)
    editor.gamma_edit.editingFinished.emit()
    editor.clip_combo.activated.emit(editor.clip_combo.currentIndex())
    editor.norm_kwargs_edit.setText("clip=False")
    editor.norm_kwargs_edit.setModified(True)
    editor.norm_kwargs_edit.editingFinished.emit()
    assert emitted == []

    assert not editor.cmap_override_check.isTristate()
    editor.cmap_override_check.click()
    assert editor.cmap_override_check.checkState() == QtCore.Qt.CheckState.Checked
    assert editor.cmap_combo.isEnabled()
    assert emitted[-1] == (
        FigurePlotSlicesPanelStyleState(
            map_index=0,
            slice_index=0,
            cmap="viridis",
        ),
    )

    editor.norm_override_check.click()
    assert editor.norm_override_check.checkState() == QtCore.Qt.CheckState.Checked
    assert editor.norm_combo.isEnabled()
    assert emitted[-1] == (
        FigurePlotSlicesPanelStyleState(
            map_index=0,
            slice_index=0,
            cmap="viridis",
            norm_name="PowerNorm",
        ),
    )


def test_figure_composer_plot_slices_mappable_tagging_edges() -> None:
    operation = FigureOperationState.plot_slices(label="data", sources=("data",))
    key = plot_slices_model._PlotSlicesPanelKey(0, 0, "panel")
    figure = Figure()
    axis = figure.subplots()
    old_mappable = axis.imshow(np.arange(4.0).reshape(2, 2))
    old_ids = plot_slices_render._axis_mappable_ids((axis,))

    plot_slices_render._tag_plot_slices_mappables(
        operation,
        (axis,),
        (key, plot_slices_model._PlotSlicesPanelKey(0, 1, "extra")),
        old_ids,
    )

    assert not hasattr(
        old_mappable,
        plot_slices_render._PLOT_SLICES_MAPPABLE_OPERATION_ID_ATTR,
    )

    new_mappable = axis.imshow(np.arange(4.0).reshape(2, 2))
    plot_slices_render._tag_plot_slices_mappables(
        operation,
        (axis,),
        (key,),
        old_ids,
    )

    assert not hasattr(
        old_mappable,
        plot_slices_render._PLOT_SLICES_MAPPABLE_OPERATION_ID_ATTR,
    )
    assert (
        getattr(
            new_mappable,
            plot_slices_render._PLOT_SLICES_MAPPABLE_OPERATION_ID_ATTR,
        )
        == operation.operation_id
    )
    assert getattr(
        new_mappable,
        plot_slices_render._PLOT_SLICES_MAPPABLE_PANEL_KEY_ATTR,
    ) == (0, 0)


def test_figure_composer_plot_slices_all_coordinate_helper_edges() -> None:
    data = xr.DataArray(
        np.arange(15.0).reshape(5, 3),
        dims=("eV", "kx"),
        coords={"eV": np.linspace(-0.2, 0.2, 5), "kx": [-1.0, 0.0, 1.0]},
        name="data",
    )
    context = FigureDocument(
        FigureRecipeState(),
        source_data={"data": data},
    )
    operation = FigureOperationState.plot_slices(
        label="slices",
        sources=("data",),
    ).model_copy(update={"slice_values_mode": "all"})

    assert plot_slices_model._all_coordinate_slice_values(context, operation) == ()
    assert plot_slices_codegen._all_coordinate_slice_values_code(operation) is None
    assert (
        plot_slices_codegen._first_plot_slices_source_code(
            operation.model_copy(update={"sources": ()})
        )
        is None
    )
    selection_operation = operation.model_copy(
        update={
            "map_selections": (
                FigureDataSelectionState(source="data", qsel={"eV": 0.1}),
            )
        }
    )
    assert (
        plot_slices_codegen._first_plot_slices_source_code(selection_operation)
        == "data"
    )
    assert (
        plot_slices_codegen._all_coordinate_slice_values_code(
            operation.model_copy(update={"sources": (), "slice_dim": "eV"})
        )
        is None
    )
    assert plot_slices_model._slice_values_mode_from_text("not a mode") == "manual"
    assert plot_slices_model._line_color_mode_from_text("By coordinate") == "coordinate"
    assert plot_slices_model._line_color_mode_from_text("Manual") == "manual"
    assert (
        plot_slices_model._all_coordinate_slice_values_error(
            context, operation, data.dims
        )
        == "Choose a dimension before using all coordinate values."
    )
    assert (
        plot_slices_model._all_coordinate_slice_values_summary(context, operation)
        == "Choose a dimension."
    )

    missing_dim = operation.model_copy(update={"slice_dim": "missing"})
    assert plot_slices_model._all_coordinate_slice_values(context, missing_dim) == ()
    assert "'missing' is not an input dimension" in (
        plot_slices_model._all_coordinate_slice_values_error(
            context, missing_dim, data.dims
        )
    )
    assert "'missing' is not an input dimension" in (
        plot_slices_model._all_coordinate_slice_values_summary(context, missing_dim)
    )

    missing_source_context = FigureDocument(FigureRecipeState())
    assert (
        plot_slices_model._all_coordinate_slice_values(
            missing_source_context,
            operation.model_copy(update={"slice_dim": "eV"}),
        )
        == ()
    )
    assert (
        plot_slices_model._all_coordinate_slice_values_summary(
            missing_source_context, missing_dim
        )
        == "Select at least one valid source."
    )

    string_coord = xr.DataArray(
        np.ones((2, 3)),
        dims=("label", "kx"),
        coords={"label": ["a", "b"], "kx": [-1.0, 0.0, 1.0]},
        name="string_coord",
    )
    string_context = FigureDocument(
        FigureRecipeState(),
        source_data={"data": string_coord},
    )
    string_operation = operation.model_copy(update={"slice_dim": "label"})
    assert (
        plot_slices_model._all_coordinate_slice_values(string_context, string_operation)
        == ()
    )
    assert "numeric and non-empty" in (
        plot_slices_model._all_coordinate_slice_values_error(
            string_context, string_operation, string_coord.dims
        )
    )
    assert "numeric and non-empty" in (
        plot_slices_model._all_coordinate_slice_values_summary(
            string_context, string_operation
        )
    )

    thinned = operation.model_copy(update={"slice_dim": "eV", "slice_values_thin": 2})
    assert plot_slices_model._all_coordinate_slice_values(
        context, thinned
    ) == pytest.approx(tuple(data.thin({"eV": 2}).coords["eV"].values))
    assert (
        plot_slices_model._all_coordinate_slice_values_summary(context, thinned)
        == "eV: 5 values, 3 plotted"
    )
    assert (
        plot_slices_codegen._all_coordinate_slice_values_code(thinned)
        == 'data.thin({"eV": 2}).coords["eV"].values'
    )
    assert (
        plot_slices_codegen._all_coordinate_slice_values_code(
            thinned.model_copy(update={"slice_values_thin": 1})
        )
        == 'data.coords["eV"].values'
    )
    assert (
        plot_slices_model._all_coordinate_slice_values_summary(
            context, thinned.model_copy(update={"slice_values_thin": 1})
        )
        == "eV: 5 values"
    )
    assert (
        plot_slices_codegen._plot_slices_slice_values_code(
            context, operation.model_copy(update={"slice_values_mode": "manual"})
        )
        == "[None]"
    )
    assert (
        plot_slices_codegen._plot_slices_slice_values_code(
            context,
            operation.model_copy(
                update={
                    "slice_values_mode": "manual",
                    "slice_dim": "eV",
                    "slice_values": (0.1,),
                }
            ),
        )
        == "[0.1]"
    )
    assert plot_slices_model._plot_slices_panel_qsel_kwargs(
        operation.model_copy(
            update={"slice_dim": "eV", "slice_values": (0.1,), "slice_width": 0.2}
        ),
        plot_slices_model._PlotSlicesPanelKey(0, 0, ""),
    ) == {"eV": 0.1, "eV_width": 0.2}


def test_figure_composer_plot_slices_line_transform_rejects_zero_scale(
    qtbot,
) -> None:
    data = xr.DataArray(
        np.zeros((2, 2)),
        dims=("eV", "kx"),
        coords={"eV": [0.0, 1.0], "kx": [0.0, 1.0]},
        name="data",
    )
    operation = FigureOperationState.plot_slices(
        label="line_slices",
        sources=("data",),
        axes=FigureAxesSelectionState(axes=((0, 0),)),
        slice_dim="eV",
        slice_values=(0.0,),
    ).model_copy(update={"line_normalize": "max"})
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    with pytest.raises(ValueError, match="Cannot normalize profile by max"):
        tool.generated_code()
