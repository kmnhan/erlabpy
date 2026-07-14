# ruff: noqa: F403, F405

from ._common import *


def test_figure_composer_method_doc_url_uses_family_templates() -> None:
    assert figurecomposer_method._method_doc_url(
        figurecomposer_method.AXES_METHODS["text"]
    ) == ("https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html")
    assert figurecomposer_method._method_doc_url(
        figurecomposer_method.FIGURE_METHODS["supxlabel"]
    ) == (
        "https://matplotlib.org/stable/api/_as_gen/"
        "matplotlib.figure.Figure.supxlabel.html"
    )
    assert figurecomposer_method._method_doc_url(
        figurecomposer_method.ERLAB_METHODS["clean_labels"]
    ) == (
        "https://erlabpy.readthedocs.io/en/stable/erlab.plotting.html"
        "#erlab.plotting.clean_labels"
    )
    spec = figurecomposer_method.MethodSpec(
        family=FigureMethodFamily.ERLAB,
        name="call_name",
        label="call_name",
        tooltip="test",
        target_domain=figurecomposer_method.MethodTargetDomain.FIGURE,
        call_policy=figurecomposer_method.MethodCallPolicy.PLAIN_CALL,
        doc_name="documented_name",
    )
    assert figurecomposer_method._method_doc_url(spec) == (
        "https://erlabpy.readthedocs.io/en/stable/erlab.plotting.html"
        "#erlab.plotting.documented_name"
    )


def test_figure_composer_method_docs_button_opens_current_url(
    qtbot, monkeypatch
) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="text",
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.FIGURE,
                    name="supxlabel",
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.ERLAB,
                    name="clean_labels",
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    opened_urls: list[str] = []

    def record_url(url: QtCore.QUrl) -> bool:
        opened_urls.append(url.toString())
        return True

    monkeypatch.setattr(QtGui.QDesktopServices, "openUrl", record_url)

    for row, expected in (
        (
            0,
            "https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html",
        ),
        (
            1,
            "https://matplotlib.org/stable/api/_as_gen/"
            "matplotlib.figure.Figure.supxlabel.html",
        ),
        (
            2,
            "https://erlabpy.readthedocs.io/en/stable/erlab.plotting.html"
            "#erlab.plotting.clean_labels",
        ),
    ):
        tool.operation_list.setCurrentItem(tool.operation_list.topLevelItem(row))
        tool._select_step_section("method")
        button = tool.step_editor_stack.currentWidget().findChild(
            QtWidgets.QToolButton, "figureComposerMethodDocsButton"
        )
        assert button is not None
        assert button.isEnabled()
        assert not button.autoRaise()
        assert button.property("figure_method_doc_url") == expected
        button.click()

    assert opened_urls == [
        "https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html",
        (
            "https://matplotlib.org/stable/api/_as_gen/"
            "matplotlib.figure.Figure.supxlabel.html"
        ),
        (
            "https://erlabpy.readthedocs.io/en/stable/erlab.plotting.html"
            "#erlab.plotting.clean_labels"
        ),
    ]


def test_figure_composer_method_helper_edge_contracts(
    qtbot, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=2, layout=None),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="set_xlim",
                    args=(0.0, 1.0),
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.FIGURE,
                    name="set_layout_engine",
                    args=("compressed",),
                    kwargs={"hspace": 0.2, "pad": 0.1},
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    render_calls: list[FigureComposerTool] = []
    monkeypatch.setattr(
        figurecomposer_tool_module,
        "_render_preview",
        lambda tool: render_calls.append(tool),
    )
    tool.tool_status = tool.tool_status
    assert render_calls == [tool]

    with pytest.raises(ValueError, match="Unsupported axes method"):
        figurecomposer_method._method_spec(
            FigureOperationState.method(family=FigureMethodFamily.AXES, name="missing")
        )
    text_spec = figurecomposer_method.AXES_METHODS["set_xlim"]
    assert figurecomposer_method._method_selector_text(text_spec) == text_spec.name

    colorbar_operation = FigureOperationState.method(
        family=FigureMethodFamily.ERLAB,
        name="nice_colorbar",
    )
    colorbar_spec = figurecomposer_method._method_spec(colorbar_operation)
    with pytest.raises(ValueError, match="Unsupported call policy"):
        figurecomposer_method._effective_call_policy(
            colorbar_operation.model_copy(update={"method_call_policy": "bad-policy"}),
            colorbar_spec,
        )
    with pytest.raises(ValueError, match="not available"):
        figurecomposer_method._effective_call_policy(
            colorbar_operation.model_copy(
                update={
                    "method_call_policy": (
                        figurecomposer_method.MethodCallPolicy.PLAIN_CALL.value
                    )
                }
            ),
            colorbar_spec,
        )
    assert (
        figurecomposer_method._effective_call_policy(
            colorbar_operation.model_copy(
                update={
                    "method_call_policy": (
                        figurecomposer_method.MethodCallPolicy.AX_KEYWORD.value
                    )
                }
            ),
            colorbar_spec,
        )
        == figurecomposer_method.MethodCallPolicy.AX_KEYWORD
    )

    assert figurecomposer_method._live_layout_axes(
        tool, render_if_missing=True
    ).shape == (1, 2)
    assert (
        figurecomposer_method._first_live_axis(
            tool,
            FigureAxesSelectionState(expression="axs[3, 3]"),
        )
        is None
    )
    assert (
        figurecomposer_method._method_float_pair_args(
            tool,
            FigureOperationState.method(
                family=FigureMethodFamily.AXES,
                name="set_xlim",
                axes=FigureAxesSelectionState(expression="axs[3, 3]"),
            ),
            figurecomposer_method.AXES_METHODS["set_xlim"],
        )
        is None
    )

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
    live_layout_axes = figurecomposer_method._live_layout_axes
    assert live_layout_axes(grid_tool) is None
    grid_axes = live_layout_axes(grid_tool, render_if_missing=True)
    assert isinstance(grid_axes, dict)
    assert set(grid_axes) == {"axis-a"}
    monkeypatch.setattr(
        figurecomposer_method,
        "_live_layout_axes",
        lambda _tool, *, render_if_missing=False: grid_axes,
    )
    assert (
        figurecomposer_method._first_live_axis(
            grid_tool,
            FigureAxesSelectionState(axes=(), axes_ids=()),
        )
        is grid_axes["axis-a"]
    )
    assert live_layout_axes(grid_tool) is None
    assert (
        figurecomposer_method._limit_method_default_args(
            grid_tool,
            figurecomposer_method.AXES_METHODS["set_xlim"],
            FigureAxesSelectionState(axes=(), axes_ids=("missing-axis",)),
        )
        == ()
    )

    scheduled_calls: list[tuple[QtCore.QObject, int, Callable[[], None]]] = []
    monkeypatch.setattr(grid_tool, "_saved_tool_window_visible", lambda _ds: True)
    monkeypatch.setattr(grid_tool, "_auto_redraw_enabled", lambda: True)
    monkeypatch.setattr(
        grid_tool, "_defer_restore_work", lambda *_args, **_kwargs: False
    )
    monkeypatch.setattr(
        erlab.interactive.utils,
        "single_shot",
        lambda *args: scheduled_calls.append(args),
    )
    grid_tool._queue_post_restore_redraw_if_needed(xr.Dataset())
    assert len(scheduled_calls) == 1

    int_control = figurecomposer_method.MethodControlSpec(
        kind=figurecomposer_method.MethodControlKind.INT_ARG,
        label="Count",
        tooltip="count tooltip",
        object_name="count",
        default=None,
        step=2,
    )
    int_spin = figurecomposer_method._int_spinbox(None, int_control, parent=tool)
    assert int_spin.value() == 0
    assert int_spin.singleStep() == 2
    float_control = figurecomposer_method.MethodControlSpec(
        kind=figurecomposer_method.MethodControlKind.FLOAT_ARG,
        label="Value",
        tooltip="value tooltip",
        object_name="value",
        default=None,
        decimals=2,
        step=0.25,
    )
    float_spin = figurecomposer_method._float_spinbox(None, float_control, parent=tool)
    assert float_spin.value() == pytest.approx(0.0)
    assert float_spin.decimals() == 2
    assert float_spin.singleStep() == pytest.approx(0.25)
    assert "multiple values" in figurecomposer_method._numeric_control_tooltip(
        float_control,
        mixed=True,
    )

    default_from_window = figurecomposer_method._subplots_adjust_default(tool, "left")
    assert default_from_window == pytest.approx(
        tool.figure_window.figure.subplotpars.left
    )
    spin_operation = FigureOperationState.method(
        family=FigureMethodFamily.FIGURE,
        name="subplots_adjust",
        kwargs={"left": "bad"},
    )
    adjust_spin = figurecomposer_method._subplots_adjust_spinbox(
        tool,
        spin_operation,
        "left",
        mixed=False,
        parent=tool,
    )
    assert adjust_spin.value() == pytest.approx(default_from_window)
    adjust_tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.FIGURE,
                    name="subplots_adjust",
                    kwargs={"left": 0.2},
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(adjust_tool)
    figurecomposer_method._update_current_subplots_adjust_kwarg(
        adjust_tool,
        "left",
        None,
    )
    assert "left" not in adjust_tool.tool_status.operations[0].method_kwargs

    layout_spec = figurecomposer_method.FIGURE_METHODS["set_layout_engine"]
    assert (
        figurecomposer_method._layout_engine_name(
            FigureOperationState.method(
                family=FigureMethodFamily.FIGURE,
                name="set_layout_engine",
            ),
            layout_spec,
        )
        == "none"
    )
    assert figurecomposer_method._filter_layout_engine_kwargs(
        (),
        {"pad": 0.1},
    ) == {"pad": 0.1}
    assert figurecomposer_method._filter_layout_engine_kwargs(
        ("tight",),
        {"pad": 0.1, "hspace": 0.2},
    ) == {"pad": 0.1}

    with pytest.raises(ValueError, match="argument index"):
        figurecomposer_method._control_arg_index(
            figurecomposer_method.MethodControlSpec(
                kind=figurecomposer_method.MethodControlKind.TEXT_ARG,
                label="Missing",
                tooltip="missing",
                object_name="missing",
            )
        )
    with pytest.raises(ValueError, match="keyword name"):
        figurecomposer_method._control_key(
            figurecomposer_method.MethodControlSpec(
                kind=figurecomposer_method.MethodControlKind.TEXT_KWARG,
                label="Missing",
                tooltip="missing",
                object_name="missing",
            )
        )

    assert figurecomposer_method._empty_text_as_none("") is None
    assert figurecomposer_method._empty_text_as_none("title") == "title"
    assert figurecomposer_method._string_tuple_from_text_or_none("") is None
    assert figurecomposer_method._string_tuple_from_text_or_none("a, b") == ("a", "b")
    assert figurecomposer_method._format_int_value(None) == ""
    assert figurecomposer_method._format_int_value(2.0) == "2"
    assert figurecomposer_method._format_float_value(None) == ""
    assert figurecomposer_method._format_float_value(1.25) == "1.25"
    assert figurecomposer_method._format_literal_value(None) == ""
    assert figurecomposer_method._format_literal_value({"alpha": 0.5}) == "alpha=0.5"
    assert figurecomposer_method._format_literal_value((1, 2)) == "1, 2"
    assert figurecomposer_method._format_literal_value("literal") == '"literal"'
    assert figurecomposer_method._format_plot_sequence(3.0) == "3.0"
    assert figurecomposer_method._plot_sequence_from_text("") == ()
    assert figurecomposer_method._plot_sequence_from_text("1, 2") == (1, 2)
    assert figurecomposer_method._format_aspect_value(None) == ""
    assert figurecomposer_method._format_aspect_value("equal") == "equal"
    assert figurecomposer_method._format_aspect_value(2) == "2"
    assert figurecomposer_method._format_aspect_value(("custom",)) == '("custom",)'
    assert figurecomposer_method._literal_value_from_text("alpha=0.5") == {"alpha": 0.5}
    assert figurecomposer_method._literal_value_from_text("") is None
    assert figurecomposer_method._literal_value_from_text("[1, 2]") == [1, 2]
    assert figurecomposer_method._aspect_value_from_text("") is None
    assert figurecomposer_method._aspect_value_from_text("equal") == "equal"
    assert figurecomposer_method._aspect_value_from_text("2") == 2.0
    assert figurecomposer_method._aspect_value_from_text("custom") == "custom"
    assert figurecomposer_method._aspect_value_from_text("'manual'") == "manual"
    assert figurecomposer_method._aspect_value_from_text("[1, 2]") == "[1, 2]"
    assert figurecomposer_method._optional_literal_from_text("") is None
    assert figurecomposer_method._optional_literal_from_text("alpha=0.5") == {
        "alpha": 0.5
    }
    assert figurecomposer_method._optional_float_from_text("") is None
    assert figurecomposer_method._optional_float_from_text("1.5") == 1.5
    assert figurecomposer_method._optional_int_from_text("") is None
    assert figurecomposer_method._optional_int_from_text("3") == 3

    assert figurecomposer_method._plot_y_arg_value(
        FigureOperationState.method(family=FigureMethodFamily.AXES, name="plot"),
        figurecomposer_method.AXES_METHODS["plot"],
    ) == (0.0, 1.0)
    empty_default_spec = figurecomposer_method.MethodSpec(
        family=FigureMethodFamily.AXES,
        name="empty_default",
        label="empty_default",
        tooltip="empty",
        target_domain=figurecomposer_method.MethodTargetDomain.AXES,
        call_policy=figurecomposer_method.MethodCallPolicy.BOUND_EACH_AXIS,
    )
    assert (
        figurecomposer_method._plot_y_arg_value(
            FigureOperationState.method(
                family=FigureMethodFamily.AXES,
                name="empty_default",
            ),
            empty_default_spec,
        )
        == ()
    )

    assert figurecomposer_method._family_from_label("bad") == FigureMethodFamily.ERLAB
    assert (
        figurecomposer_method._family_from_label("Figure Method")
        == FigureMethodFamily.FIGURE
    )
    assert (
        figurecomposer_method._call_policy_from_label("Selected axes together")
        == figurecomposer_method.MethodCallPolicy.AX_KEYWORD
    )
    assert (
        figurecomposer_method._call_policy_from_label("plain_call")
        == figurecomposer_method.MethodCallPolicy.PLAIN_CALL
    )
    assert (
        figurecomposer_method._method_selector_text(
            figurecomposer_method.AXES_METHODS["set_xlabel"]
        )
        == "set_xlabel"
    )
    assert (
        figurecomposer_method._method_selector_text(
            figurecomposer_method.FIGURE_METHODS["supxlabel"]
        )
        == "supxlabel"
    )
    assert (
        figurecomposer_method._method_selector_text(
            figurecomposer_method.ERLAB_METHODS["clean_labels"]
        )
        == "clean_labels"
    )
    assert (
        figurecomposer_method._method_combo_object_name(FigureMethodFamily.FIGURE)
        == "figureComposerFigureMethodCombo"
    )
    assert (
        figurecomposer_method._method_kwargs_object_name(FigureMethodFamily.ERLAB)
        == "figureComposerERLabMethodKwEdit"
    )
    assert (
        figurecomposer_method._method_display(
            FigureOperationState.method(
                family=FigureMethodFamily.FIGURE,
                name="supxlabel",
            )
        )
        == "fig.supxlabel"
    )
    assert (
        figurecomposer_method._method_display(
            FigureOperationState.method(
                family=FigureMethodFamily.ERLAB,
                name="clean_labels",
            )
        )
        == "eplt.clean_labels"
    )
    assert (
        figurecomposer_method._callable_display(
            figurecomposer_method.FIGURE_METHODS["supxlabel"]
        )
        == "fig.supxlabel"
    )
    assert (
        figurecomposer_method._callable_display(colorbar_spec)
        == "erlab.plotting.nice_colorbar"
    )
    assert (
        figurecomposer_method._method_doc_url(
            figurecomposer_method.MethodSpec(
                family=FigureMethodFamily.ERLAB,
                name="custom_doc",
                label="custom_doc",
                tooltip="custom",
                target_domain=figurecomposer_method.MethodTargetDomain.NONE,
                call_policy=figurecomposer_method.MethodCallPolicy.PLAIN_CALL,
                doc_url="https://example.test/docs",
            )
        )
        == "https://example.test/docs"
    )

    transform_figure, transform_axis = plt.subplots()
    try:
        assert (
            figurecomposer_method._transform_component(
                transform_figure,
                transform_axis,
                "figure",
            )
            is transform_figure.transFigure
        )
        assert (
            figurecomposer_method._transform_component(
                transform_figure,
                transform_axis,
                "dpi",
            )
            is transform_figure.dpi_scale_trans
        )
        assert figurecomposer_method._transform_component_code("figure") == (
            "fig.transFigure"
        )
        assert (
            figurecomposer_method._transform_component_code("dpi")
            == "fig.dpi_scale_trans"
        )
        with pytest.raises(ValueError, match="Unknown transform component"):
            figurecomposer_method._transform_component(
                transform_figure,
                transform_axis,
                "bad",
            )
        with pytest.raises(ValueError, match="Unknown transform component"):
            figurecomposer_method._transform_component_code("bad")
        assert (
            figurecomposer_method._render_method_transform(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="plot",
                ),
                figurecomposer_method.AXES_METHODS["plot"],
                figure=transform_figure,
                axis=transform_axis,
            )
            is None
        )
        assert (
            figurecomposer_method._method_transform_code(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="plot",
                ),
                figurecomposer_method.AXES_METHODS["plot"],
            )
            is None
        )
    finally:
        plt.close(transform_figure)

    positional_text_spec = figurecomposer_method.MethodSpec(
        family=FigureMethodFamily.AXES,
        name="text_values_positional",
        label="text_values_positional",
        tooltip="test",
        target_domain=figurecomposer_method.MethodTargetDomain.AXES,
        call_policy=figurecomposer_method.MethodCallPolicy.BOUND_EACH_AXIS,
        text_values_policy=figurecomposer_method.MethodTextValuesPolicy.POSITIONAL,
    )
    keyword_text_spec = figurecomposer_method.MethodSpec(
        family=FigureMethodFamily.AXES,
        name="text_values_keyword",
        label="text_values_keyword",
        tooltip="test",
        target_domain=figurecomposer_method.MethodTargetDomain.AXES,
        call_policy=figurecomposer_method.MethodCallPolicy.BOUND_EACH_AXIS,
        text_values_policy=figurecomposer_method.MethodTextValuesPolicy.KWARG,
        text_values_kwarg="labels",
    )
    text_operation = FigureOperationState.method(
        family=FigureMethodFamily.AXES,
        name="text_values",
    ).model_copy(update={"text_values": ("A", "B")})
    assert figurecomposer_method._render_args_kwargs(
        tool,
        text_operation,
        positional_text_spec,
    )[0] == (["A", "B"],)
    assert figurecomposer_method._render_args_kwargs(
        tool,
        text_operation,
        keyword_text_spec,
    )[1] == {"labels": ["A", "B"]}
    assert figurecomposer_method._code_args_kwargs(
        tool,
        text_operation,
        keyword_text_spec,
    )[1] == {"labels": ["A", "B"]}

    no_operation_tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(),
            primary_source="data",
        ),
    )
    qtbot.addWidget(no_operation_tool)
    figurecomposer_method._update_current_method_name(no_operation_tool, "plot")

    tool.operation_list.setCurrentItem(tool.operation_list.topLevelItem(1))
    tool._select_step_section("method")
    figure_method_combo = tool.findChild(
        QtWidgets.QComboBox, "figureComposerFigureMethodCombo"
    )
    assert figure_method_combo is not None
    assert figure_method_combo.currentText() == "set_layout_engine"
    assert figure_method_combo.currentData() == "set_layout_engine"

    figurecomposer_method._update_current_method_args(tool, ("none",))
    assert tool.tool_status.operations[1].method_args == ("none",)
    figurecomposer_method._update_current_layout_engine(tool, 0, "tight")
    assert tool.tool_status.operations[1].method_args == ("tight",)
    assert tool.tool_status.operations[1].method_kwargs == {"pad": 0.1}
    figurecomposer_method._update_current_method_arg(tool, 2, "third")
    assert tool.tool_status.operations[1].method_args == ("tight", None, "third")
    figurecomposer_method._update_current_method_string_tuple_arg(tool, 4, "tail")
    assert tool.tool_status.operations[1].method_args == (
        "tight",
        None,
        "third",
        (),
        ("tail",),
    )
    figurecomposer_method._update_current_method_string_tuple_arg(tool, 1, "a, b")
    assert tool.tool_status.operations[1].method_args == (
        "tight",
        ("a", "b"),
        "third",
        (),
        ("tail",),
    )
    figurecomposer_method._update_current_method_string_tuple_arg(tool, 1, "")
    assert tool.tool_status.operations[1].method_args == ("tight",)
    figurecomposer_method._update_current_method_kwarg(tool, "pad", None)
    assert tool.tool_status.operations[1].method_kwargs == {}
    figurecomposer_method._update_current_method_kwarg(tool, "pad", 0.3)
    assert tool.tool_status.operations[1].method_kwargs == {"pad": 0.3}
    figurecomposer_method._operation_trust_update_callback(tool)(True)
    assert tool.tool_status.operations[1].trusted is True
    figurecomposer_method._method_float_pair_args_update_callback(tool)("0, 1")
    assert tool.tool_status.operations[1].method_args == (0.0, 1.0)
    figurecomposer_method._update_current_method_family(tool, FigureMethodFamily.AXES)
    assert tool.tool_status.operations[1].method_family == FigureMethodFamily.AXES
    figurecomposer_method._update_current_method_family(tool, FigureMethodFamily.FIGURE)
    assert tool.tool_status.operations[1].method_family == FigureMethodFamily.FIGURE
    tool.operation_list.setCurrentItem(tool.operation_list.topLevelItem(1))
    tool._select_step_section("method")
    figurecomposer_method._update_current_method_name(tool, "set_layout_engine")
    assert tool.tool_status.operations[1].method_name == "set_layout_engine"
    figurecomposer_method._update_current_method_call_policy(
        tool,
        figurecomposer_method.MethodCallPolicy.PLAIN_CALL,
    )
    assert tool.tool_status.operations[1].method_call_policy == "plain_call"
    figurecomposer_method._update_current_method_call_policy(
        tool,
        figurecomposer_method.MethodCallPolicy.BOUND_FIGURE,
    )
    assert tool.tool_status.operations[1].method_call_policy is None
    figurecomposer_method._update_current_method_text_values(tool, "\nlabel\n")
    assert tool.tool_status.operations[1].text_values == ("label",)


def test_figure_composer_method_framework_dispatch_policies(qtbot, monkeypatch) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    calls: list[tuple[str, object, dict[str, object]]] = []

    def ax_keyword_helper(*args: object, ax: object, **kwargs: object) -> None:
        calls.append(("ax_keyword", ax, kwargs))

    def fig_keyword_helper(*args: object, fig: object, **kwargs: object) -> None:
        calls.append(("fig_keyword", fig, kwargs))

    def each_axis_helper(*args: object, ax: object, **kwargs: object) -> None:
        calls.append(("each_axis", ax, kwargs))

    def plain_helper(*args: object, **kwargs: object) -> None:
        calls.append(("plain", args, kwargs))

    monkeypatch.setattr(
        eplt, "composer_ax_keyword_test", ax_keyword_helper, raising=False
    )
    monkeypatch.setattr(
        eplt, "composer_fig_keyword_test", fig_keyword_helper, raising=False
    )
    monkeypatch.setattr(
        eplt, "composer_each_axis_test", each_axis_helper, raising=False
    )
    monkeypatch.setattr(eplt, "composer_plain_test", plain_helper, raising=False)
    monkeypatch.setitem(
        figurecomposer_method.ERLAB_METHODS,
        "composer_ax_keyword_test",
        figurecomposer_method.MethodSpec(
            family=FigureMethodFamily.ERLAB,
            name="composer_ax_keyword_test",
            label="composer_ax_keyword_test",
            tooltip="test",
            target_domain=figurecomposer_method.MethodTargetDomain.AXES,
            call_policy=figurecomposer_method.MethodCallPolicy.AX_KEYWORD,
        ),
    )
    monkeypatch.setitem(
        figurecomposer_method.ERLAB_METHODS,
        "composer_fig_keyword_test",
        figurecomposer_method.MethodSpec(
            family=FigureMethodFamily.ERLAB,
            name="composer_fig_keyword_test",
            label="composer_fig_keyword_test",
            tooltip="test",
            target_domain=figurecomposer_method.MethodTargetDomain.FIGURE,
            call_policy=figurecomposer_method.MethodCallPolicy.FIG_KEYWORD,
        ),
    )
    monkeypatch.setitem(
        figurecomposer_method.ERLAB_METHODS,
        "composer_each_axis_test",
        figurecomposer_method.MethodSpec(
            family=FigureMethodFamily.ERLAB,
            name="composer_each_axis_test",
            label="composer_each_axis_test",
            tooltip="test",
            target_domain=figurecomposer_method.MethodTargetDomain.AXES,
            call_policy=figurecomposer_method.MethodCallPolicy.EACH_AXIS_AX_KEYWORD,
        ),
    )
    monkeypatch.setitem(
        figurecomposer_method.ERLAB_METHODS,
        "composer_plain_test",
        figurecomposer_method.MethodSpec(
            family=FigureMethodFamily.ERLAB,
            name="composer_plain_test",
            label="composer_plain_test",
            tooltip="test",
            target_domain=figurecomposer_method.MethodTargetDomain.NONE,
            call_policy=figurecomposer_method.MethodCallPolicy.PLAIN_CALL,
            default_args=("value",),
        ),
    )
    operations = (
        FigureOperationState.method(
            family=FigureMethodFamily.ERLAB,
            name="composer_ax_keyword_test",
            axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
            kwargs={"alpha": 0.5},
        ),
        FigureOperationState.method(
            family=FigureMethodFamily.ERLAB,
            name="composer_fig_keyword_test",
            kwargs={"label": "figure"},
        ),
        FigureOperationState.method(
            family=FigureMethodFamily.ERLAB,
            name="composer_each_axis_test",
            axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
            kwargs={"color": "red"},
        ),
        FigureOperationState.method(
            family=FigureMethodFamily.ERLAB,
            name="composer_plain_test",
        ),
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=operations,
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    figure = Figure()
    figurecomposer_rendering._render_into_figure(tool, figure, sync_visible=False)

    assert [call[0] for call in calls] == [
        "ax_keyword",
        "fig_keyword",
        "each_axis",
        "each_axis",
        "plain",
    ]
    assert isinstance(calls[0][1], np.ndarray)
    assert calls[0][2] == {"alpha": 0.5}
    assert calls[1][1] is figure
    assert calls[1][2] == {"label": "figure"}
    assert calls[2][2] == {"color": "red"}
    assert calls[3][2] == {"color": "red"}
    assert calls[4][1] == ("value",)

    lines = [
        line
        for operation in operations
        for line in figurecomposer_method._method_code(tool, operation)
    ]
    assert lines == [
        "eplt.composer_ax_keyword_test(alpha=0.5, ax=axs)",
        'eplt.composer_fig_keyword_test(label="figure", fig=fig)',
        "for ax in axs.flat:",
        '    eplt.composer_each_axis_test(color="red", ax=ax)',
        'eplt.composer_plain_test("value")',
    ]


def test_figure_composer_axes_plot_data_update_helper(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="plot",
                    args=((0.0, 1.0),),
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    figurecomposer_method._update_current_plot_data_arg(tool, "x", "0, 1")
    assert tool.tool_status.operations[0].method_args == ((0, 1), (0.0, 1.0))

    figurecomposer_method._update_current_plot_data_arg(tool, "y", "2, 3")
    assert tool.tool_status.operations[0].method_args == ((0, 1), (2, 3))

    figurecomposer_method._update_current_plot_data_arg(tool, "x", "")
    assert tool.tool_status.operations[0].method_args == ((2, 3),)


def test_figure_composer_tick_params_controls_update_state(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="tick_params",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    _select_operation_rows(tool, (0,))
    tool._select_step_section("method")
    method_page = tool.step_editor_stack.currentWidget()
    assert method_page is not None
    editor = method_page.findChild(
        figurecomposer_tick_params.TickParamsEditorWidget,
        "figureComposerAxesMethodTickParamsEditor",
    )
    assert editor is not None
    assert editor.tick_params() == {}
    editor.setToolTip("generic tick params tooltip")
    assert editor.toolTip() == ""
    axis_control = editor.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodTickParamsAxisCombo"
    )
    reset_control = editor.findChild(
        QtWidgets.QCheckBox, "figureComposerAxesMethodTickParamsResetCombo"
    )
    bottom_control = editor.findChild(
        QtWidgets.QCheckBox, "figureComposerAxesMethodTickParamsBottomCombo"
    )
    length_edit = editor.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodTickParamsLengthEdit"
    )
    colors_edit = editor.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodTickParamsColorsEdit"
    )
    grid_disclosure = editor.findChild(
        QtWidgets.QWidget, "figureComposerAxesMethodTickParamsGridDisclosure"
    )
    assert axis_control is not None
    assert reset_control is not None
    assert bottom_control is not None
    assert length_edit is not None
    assert colors_edit is not None
    assert grid_disclosure is not None
    tick_tooltips = (
        axis_control.toolTip(),
        reset_control.toolTip(),
        bottom_control.toolTip(),
        length_edit.toolTip(),
        colors_edit.toolTip(),
        grid_disclosure.toolTip(),
    )
    assert all(tick_tooltips)
    assert len(set(tick_tooltips)) == len(tick_tooltips)
    assert tool.tool_status.operations[0].method_kwargs == {}

    _click_tick_params_segment(
        editor,
        "figureComposerAxesMethodTickParamsAxisCombo",
        "x",
    )
    _click_tick_params_segment(
        editor,
        "figureComposerAxesMethodTickParamsDirectionCombo",
        "inout",
    )
    _set_tick_params_button(
        editor,
        "figureComposerAxesMethodTickParamsBottomCombo",
        False,
    )
    _finish_tick_params_edit(
        editor,
        "figureComposerAxesMethodTickParamsLengthEdit",
        "4.5",
    )
    _finish_tick_params_edit(
        editor,
        "figureComposerAxesMethodTickParamsLabelSizeEdit",
        "9",
    )
    _finish_tick_params_edit(
        editor,
        "figureComposerAxesMethodTickParamsColorsEdit",
        "tab:red",
    )

    assert tool.tool_status.operations[0].method_kwargs == {
        "axis": "x",
        "direction": "inout",
        "bottom": False,
        "length": 4.5,
        "labelsize": 9,
        "colors": "tab:red",
    }

    _set_tick_params_button(
        editor,
        "figureComposerAxesMethodTickParamsBottomCombo",
        None,
    )
    _finish_tick_params_edit(editor, "figureComposerAxesMethodTickParamsLengthEdit", "")

    assert tool.tool_status.operations[0].method_kwargs == {
        "axis": "x",
        "direction": "inout",
        "labelsize": 9,
        "colors": "tab:red",
    }


def test_figure_composer_axes_methods_render_and_codegen(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="text",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ).model_copy(
                    update={
                        "method_args": (0.1, 0.9, "Panel"),
                        "method_transform": "axes",
                        "method_kwargs": {"ha": "left", "va": "top"},
                    }
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="axvline",
                    axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
                ).model_copy(
                    update={
                        "method_args": (0.5,),
                        "method_kwargs": {
                            "color": "red",
                            "linestyle": "--",
                        },
                    }
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="axvspan",
                    axes=FigureAxesSelectionState(axes=((0, 1),)),
                ).model_copy(
                    update={
                        "method_args": (0.2, 0.4),
                        "method_kwargs": {"alpha": 0.25},
                    }
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="set_xticks",
                    axes=FigureAxesSelectionState(axes=((0, 1),)),
                ).model_copy(
                    update={
                        "method_args": ((0.0, 1.0), ("left", "right")),
                    }
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="grid",
                    axes=FigureAxesSelectionState(axes=((0, 1),)),
                ).model_copy(
                    update={
                        "method_args": (True,),
                        "method_kwargs": {"which": "major", "axis": "x"},
                    }
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="set_axis_off",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="set_xscale",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="set_yscale",
                    axes=FigureAxesSelectionState(axes=((0, 1),)),
                ).model_copy(update={"method_args": ("linear",)}),
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="set_title",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ).model_copy(
                    update={
                        "method_args": ("Left title",),
                        "method_kwargs": {"loc": "left", "pad": 2.0},
                    }
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="set_xlabel",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ).model_copy(
                    update={
                        "method_args": ("Momentum",),
                        "method_kwargs": {"loc": "right", "labelpad": 3.0},
                    }
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="set_ylabel",
                    axes=FigureAxesSelectionState(axes=((0, 1),)),
                ).model_copy(
                    update={
                        "method_args": ("Energy",),
                        "method_kwargs": {"loc": "top", "labelpad": 4.0},
                    }
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="margins",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ).model_copy(
                    update={
                        "method_kwargs": {"x": 0.1, "y": 0.2, "tight": False},
                    }
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="set_aspect",
                    axes=FigureAxesSelectionState(axes=((0, 1),)),
                ).model_copy(
                    update={
                        "method_args": (2.0,),
                        "method_kwargs": {"share": True},
                    }
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="tick_params",
                    axes=FigureAxesSelectionState(axes=((0, 1),)),
                ).model_copy(
                    update={
                        "method_kwargs": {
                            "axis": "x",
                            "which": "both",
                            "direction": "in",
                            "length": 6.0,
                            "width": 1.5,
                            "colors": "red",
                            "pad": 3.0,
                            "labelrotation": 45.0,
                            "labelsize": 8,
                            "bottom": True,
                            "top": True,
                            "labelbottom": True,
                            "labeltop": False,
                            "grid_color": "blue",
                            "grid_alpha": 0.25,
                            "grid_linewidth": 0.75,
                            "grid_linestyle": "--",
                        },
                    }
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    tool.operation_list.setCurrentItem(tool.operation_list.topLevelItem(0))
    tool._select_step_section("method")
    method_combo = tool.findChild(QtWidgets.QComboBox, "figureComposerAxesMethodCombo")
    transform_combo = tool.findChild(
        QtWidgets.QComboBox, "figureComposerMethodTransformModeCombo"
    )
    text_edit = tool.findChild(
        QtWidgets.QPlainTextEdit, "figureComposerAxesMethodTextEdit"
    )
    kwargs_edit = tool.findChild(QtWidgets.QLineEdit, "figureComposerAxesMethodKwEdit")
    assert method_combo is not None
    assert transform_combo is not None
    assert text_edit is not None
    assert kwargs_edit is not None
    assert method_combo.currentText() == "text"
    assert method_combo.currentData() == "text"
    assert transform_combo.currentText() == "axes"
    assert text_edit.toPlainText() == "Panel"
    assert kwargs_edit.text() == 'ha="left", va="top"'
    assert tool.step_section_buttons["method"].text() == "ax.text"

    tool.operation_list.setCurrentItem(tool.operation_list.topLevelItem(4))
    tool._select_step_section("method")
    grid_visible_combo = tool.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodGridVisibleCombo"
    )
    grid_which_combo = tool.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodWhichCombo"
    )
    grid_axis_combo = tool.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodAxisCombo"
    )
    assert grid_visible_combo is not None
    assert grid_which_combo is not None
    assert grid_axis_combo is not None
    assert grid_visible_combo.currentText() == "True"
    assert grid_which_combo.currentText() == "major"
    assert grid_axis_combo.currentText() == "x"

    scale_names = tuple(mscale.get_scale_names())
    assert "log" in scale_names
    tool.operation_list.setCurrentItem(tool.operation_list.topLevelItem(6))
    tool._select_step_section("method")
    xscale_combo = tool.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodXScaleCombo"
    )
    assert xscale_combo is not None
    assert (
        tuple(xscale_combo.itemText(index) for index in range(xscale_combo.count()))
        == scale_names
    )
    assert xscale_combo.currentText() == "log"
    assert tool.tool_status.operations[6].method_args == ()

    tool.operation_list.setCurrentItem(tool.operation_list.topLevelItem(7))
    tool._select_step_section("method")
    yscale_combo = tool.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodYScaleCombo"
    )
    assert yscale_combo is not None
    assert (
        tuple(yscale_combo.itemText(index) for index in range(yscale_combo.count()))
        == scale_names
    )
    y_scale = "linear" if "linear" in scale_names else scale_names[0]
    _activate_combo_text(yscale_combo, y_scale)
    assert tool.tool_status.operations[7].method_args == (y_scale,)

    tool.operation_list.setCurrentItem(tool.operation_list.topLevelItem(8))
    tool._select_step_section("method")
    title_edit = tool.findChild(
        QtWidgets.QPlainTextEdit, "figureComposerAxesMethodTitleEdit"
    )
    title_loc_combo = tool.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodTitleLocCombo"
    )
    title_pad_edit = tool.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerAxesMethodTitlePadEdit"
    )
    assert title_edit is not None
    assert title_loc_combo is not None
    assert title_pad_edit is not None
    assert title_edit.toPlainText() == "Left title"
    assert title_loc_combo.currentText() == "left"
    assert title_pad_edit.value() == 2.0

    tool.operation_list.setCurrentItem(tool.operation_list.topLevelItem(11))
    tool._select_step_section("method")
    x_margin_edit = tool.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerAxesMethodXMarginEdit"
    )
    y_margin_edit = tool.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerAxesMethodYMarginEdit"
    )
    tight_combo = tool.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodMarginsTightCombo"
    )
    assert x_margin_edit is not None
    assert y_margin_edit is not None
    assert tight_combo is not None
    assert x_margin_edit.value() == pytest.approx(0.1)
    assert y_margin_edit.value() == pytest.approx(0.2)
    assert tight_combo.currentText() == "False"

    tool.operation_list.setCurrentItem(tool.operation_list.topLevelItem(12))
    tool._select_step_section("method")
    aspect_edit = tool.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodAspectEdit"
    )
    aspect_share_combo = tool.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodAspectShareCombo"
    )
    assert aspect_edit is not None
    assert aspect_share_combo is not None
    assert aspect_edit.text() == "2"
    assert aspect_share_combo.currentText() == "True"
    aspect_edit.setText("2.5")
    aspect_edit.editingFinished.emit()
    assert tool.tool_status.operations[12].method_args == (2.5,)

    tool.operation_list.setCurrentItem(tool.operation_list.topLevelItem(13))
    tool._select_step_section("method")
    tick_editor = tool.findChild(
        figurecomposer_tick_params.TickParamsEditorWidget,
        "figureComposerAxesMethodTickParamsEditor",
    )
    assert tick_editor is not None
    tick_top_button = tick_editor.findChild(
        QtWidgets.QCheckBox, "figureComposerAxesMethodTickParamsTopCombo"
    )
    tick_labeltop_button = tick_editor.findChild(
        QtWidgets.QCheckBox, "figureComposerAxesMethodTickParamsLabelTopCombo"
    )
    tick_length_edit = tick_editor.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodTickParamsLengthEdit"
    )
    tick_label_size_edit = tick_editor.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodTickParamsLabelSizeEdit"
    )
    tick_colors_edit = tick_editor.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodTickParamsColorsEdit"
    )
    assert tick_top_button is not None
    assert tick_labeltop_button is not None
    assert tick_length_edit is not None
    assert tick_label_size_edit is not None
    assert tick_colors_edit is not None
    assert tick_editor.tick_params()["axis"] == "x"
    assert tick_editor.tick_params()["which"] == "both"
    assert tick_editor.tick_params()["direction"] == "in"
    assert tick_top_button.property("tick_params_value") is True
    assert tick_labeltop_button.property("tick_params_value") is False
    assert tick_length_edit.text() == "6"
    assert tick_label_size_edit.text() == "8"
    assert tick_colors_edit.text() == "red"

    fig = tool.figure
    figurecomposer_rendering._render_into_figure(tool, fig, sync_visible=False)
    fig.canvas.draw()
    assert fig.axes[0].texts[0].get_text() == "Panel"
    assert fig.axes[0].texts[0].get_transform() == fig.axes[0].transAxes
    assert fig.axes[0].lines[0].get_color() == "red"
    assert fig.axes[0].axison is False
    assert fig.axes[1].lines[0].get_color() == "red"
    assert [tick.get_text() for tick in fig.axes[1].get_xticklabels()] == [
        "left",
        "right",
    ]
    assert len(fig.axes[1].patches) == 1
    assert fig.axes[0].get_xscale() == "log"
    assert fig.axes[1].get_yscale() == y_scale
    assert fig.axes[0].get_title(loc="left") == "Left title"
    assert fig.axes[0].get_xlabel() == "Momentum"
    assert fig.axes[1].get_ylabel() == "Energy"
    assert fig.axes[0].margins() == (0.1, 0.2)
    assert fig.axes[1].get_aspect() == 2.5
    tick = fig.axes[1].xaxis.get_major_ticks()[0]
    assert tick.tick1line.get_markersize() == 6.0
    assert tick.tick1line.get_markeredgewidth() == 1.5
    assert tick.tick1line.get_color() == "red"
    assert tick.tick2line.get_visible() is True
    assert tick.label1.get_fontsize() == 8.0
    assert tick.label1.get_rotation() == 45.0
    assert tick.label1.get_color() == "red"
    assert tick.label2.get_visible() is False

    code = tool.generated_code()
    assert (
        'axs[0, 0].text(0.1, 0.9, "Panel", ha="left", va="top", '
        "transform=axs[0, 0].transAxes)"
    ) in code
    assert (
        'for ax in axs.flat:\n    ax.axvline(0.5, color="red", linestyle="--")' in code
    )
    assert "axs[0, 1].axvspan(0.2, 0.4, alpha=0.25)" in code
    assert 'axs[0, 1].set_xticks((0.0, 1.0), ("left", "right"))' in code
    assert 'axs[0, 1].grid(True, which="major", axis="x")' in code
    assert "axs[0, 0].set_axis_off()" in code
    assert 'axs[0, 0].set_xscale("log")' in code
    assert f'axs[0, 1].set_yscale("{y_scale}")' in code
    assert 'axs[0, 0].set_title("Left title", loc="left", pad=2.0)' in code
    assert 'axs[0, 0].set_xlabel("Momentum", loc="right", labelpad=3.0)' in code
    assert 'axs[0, 1].set_ylabel("Energy", loc="top", labelpad=4.0)' in code
    assert "axs[0, 0].margins(x=0.1, y=0.2, tight=False)" in code
    assert "axs[0, 1].set_aspect(2.5, share=True)" in code
    assert (
        'axs[0, 1].tick_params(axis="x", which="both", direction="in", '
        'length=6.0, width=1.5, colors="red", pad=3.0, labelrotation=45.0, '
        "labelsize=8, bottom=True, top=True, labelbottom=True, "
        'labeltop=False, grid_color="blue", grid_alpha=0.25, '
        'grid_linewidth=0.75, grid_linestyle="--")'
    ) in code
    assert "for ax in (axs[0, 0],):" not in code
    assert "for ax in (axs[0, 1],):" not in code

    namespace: dict[str, typing.Any] = {}
    exec(code, namespace)  # noqa: S102
    namespace["fig"].canvas.draw()
    axs = namespace["axs"]
    assert axs[0, 0].texts[0].get_text() == "Panel"
    assert axs[0, 0].axison is False
    assert [tick.get_text() for tick in axs[0, 1].get_xticklabels()] == [
        "left",
        "right",
    ]
    assert axs[0, 0].get_xscale() == "log"
    assert axs[0, 1].get_yscale() == y_scale
    assert axs[0, 0].get_title(loc="left") == "Left title"
    assert axs[0, 0].get_xlabel() == "Momentum"
    assert axs[0, 1].get_ylabel() == "Energy"
    assert axs[0, 0].margins() == (0.1, 0.2)
    assert axs[0, 1].get_aspect() == 2.5
    tick = axs[0, 1].xaxis.get_major_ticks()[0]
    assert tick.tick1line.get_markersize() == 6.0
    assert tick.tick1line.get_markeredgewidth() == 1.5
    assert tick.tick1line.get_color() == "red"
    assert tick.tick2line.get_visible() is True
    assert tick.label1.get_fontsize() == 8.0
    assert tick.label1.get_rotation() == 45.0
    assert tick.label1.get_color() == "red"
    assert tick.label2.get_visible() is False


def test_figure_composer_axes_plot_method_render_and_codegen(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="plot",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ).model_copy(
                    update={
                        "method_args": (
                            (0.0, 0.5, 1.0),
                            (1.0, 0.5, 0.0),
                        ),
                        "method_kwargs": {
                            "color": "C1",
                            "linestyle": "--",
                            "linewidth": 2.5,
                            "marker": "o",
                            "markersize": 4.0,
                            "markerfacecolor": "white",
                            "markeredgecolor": "black",
                            "alpha": 0.75,
                            "label": "manual",
                            "zorder": 4.0,
                            "clip_on": False,
                            "transform": "ignored",
                        },
                        "method_transform": "blend",
                        "method_transform_x": "data",
                        "method_transform_y": "axes",
                    }
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="plot",
                    axes=FigureAxesSelectionState(axes=((0, 1),)),
                ).model_copy(
                    update={
                        "method_args": ((0.0, 0.25, 1.0),),
                        "method_transform": "custom",
                        "method_transform_expression": "ax.transAxes",
                        "trusted": True,
                    }
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    tool.operation_list.setCurrentItem(tool.operation_list.topLevelItem(0))
    tool._select_step_section("method")
    method_page = tool.step_editor_stack.currentWidget()
    method_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodCombo"
    )
    x_edit = method_page.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodPlotXEdit"
    )
    y_edit = method_page.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodPlotYEdit"
    )
    color_edit = method_page.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodPlotColorEdit"
    )
    color_button = method_page.findChild(
        QtWidgets.QWidget, "figureComposerAxesMethodPlotColorEditButton"
    )
    style_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodPlotLineStyleCombo"
    )
    width_spin = method_page.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerAxesMethodPlotLineWidthSpin"
    )
    marker_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodPlotMarkerCombo"
    )
    transform_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerMethodTransformModeCombo"
    )
    transform_x_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerMethodTransformXCombo"
    )
    transform_y_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerMethodTransformYCombo"
    )
    kwargs_edit = method_page.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodKwEdit"
    )
    assert method_combo is not None
    assert x_edit is not None
    assert y_edit is not None
    assert color_edit is not None
    assert color_button is not None
    assert style_combo is not None
    assert width_spin is not None
    assert marker_combo is not None
    assert transform_combo is not None
    assert transform_x_combo is not None
    assert transform_y_combo is not None
    assert kwargs_edit is not None
    assert method_combo.currentText() == "plot"
    assert method_combo.currentData() == "plot"
    assert x_edit.text() == "0.0, 0.5, 1.0"
    assert y_edit.text() == "1.0, 0.5, 0.0"
    assert color_edit.text() == "C1"
    assert style_combo.currentText() == "--"
    assert width_spin.value() == pytest.approx(2.5)
    assert marker_combo.currentText() == "o"
    assert transform_combo.currentText() == "blend"
    assert transform_x_combo.currentText() == "data"
    assert transform_y_combo.currentText() == "axes"
    assert kwargs_edit.text() == "clip_on=False"

    color_edit.setText("tab:blue")
    color_edit.setModified(True)
    color_widget = color_edit.parentWidget()
    assert isinstance(color_widget, figurecomposer_widgets._ColorLineEditWidget)
    color_widget.editingFinished.emit()
    assert tool.tool_status.operations[0].method_kwargs["color"] == "tab:blue"
    kwargs_edit.setText('clip_on=True, transform="ax.transData"')
    kwargs_edit.setModified(True)
    kwargs_edit.editingFinished.emit()
    assert tool.tool_status.operations[0].method_kwargs["clip_on"] is True
    assert tool.tool_status.operations[0].method_kwargs["transform"] == "ignored"

    tool.operation_list.setCurrentItem(tool.operation_list.topLevelItem(1))
    tool._select_step_section("method")
    method_page = tool.step_editor_stack.currentWidget()
    custom_transform_edit = method_page.findChild(
        QtWidgets.QLineEdit, "figureComposerMethodTransformExpressionEdit"
    )
    custom_transform_trusted = method_page.findChild(
        QtWidgets.QCheckBox, "figureComposerMethodTransformTrustedCheck"
    )
    assert custom_transform_edit is not None
    assert custom_transform_trusted is not None
    assert custom_transform_edit.text() == "ax.transAxes"
    assert custom_transform_trusted.isChecked()

    fig = tool.figure
    figurecomposer_rendering._render_into_figure(tool, fig, sync_visible=False)
    assert tool._operation_render_errors == {}
    line = fig.axes[0].lines[0]
    np.testing.assert_allclose(line.get_xdata(), (0.0, 0.5, 1.0))
    np.testing.assert_allclose(line.get_ydata(), (1.0, 0.5, 0.0))
    assert line.get_color() == "tab:blue"
    assert line.get_linestyle() == "--"
    assert line.get_linewidth() == pytest.approx(2.5)
    assert line.get_marker() == "o"
    assert line.get_markersize() == pytest.approx(4.0)
    assert line.get_markerfacecolor() == "white"
    assert line.get_markeredgecolor() == "black"
    assert line.get_alpha() == pytest.approx(0.75)
    assert line.get_label() == "manual"
    assert line.get_zorder() == pytest.approx(4.0)
    assert line.get_clip_on() is True
    assert isinstance(line.get_transform(), mtransforms.BlendedGenericTransform)
    assert fig.axes[1].lines[0].get_transform() == fig.axes[1].transAxes

    code = tool.generated_code()
    assert "import matplotlib.transforms as mtransforms" in code
    assert "axs[0, 0].plot((0.0, 0.5, 1.0), (1.0, 0.5, 0.0)" in code
    assert 'color="tab:blue"' in code
    assert "clip_on=True" in code
    assert "transform=ax.transData" not in code
    assert (
        "transform=mtransforms.blended_transform_factory("
        "axs[0, 0].transData, axs[0, 0].transAxes)"
    ) in code
    assert "ax.plot((0.0, 0.25, 1.0), transform=ax.transAxes)" in code

    namespace: dict[str, typing.Any] = {}
    exec(code, namespace)  # noqa: S102
    axs = namespace["axs"]
    assert len(axs[0, 0].lines) == 1
    assert axs[0, 0].lines[0].get_color() == "tab:blue"
    assert axs[0, 1].lines[0].get_transform() == axs[0, 1].transAxes


def test_figure_composer_axes_errorbar_method_render_and_codegen(qtbot) -> None:
    data = xr.DataArray(
        np.arange(3.0),
        dims=("kx",),
        coords={"kx": [0.0, 0.5, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="errorbar",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ).model_copy(
                    update={
                        "method_args": (
                            (0.0, 0.5, 1.0),
                            (1.0, 0.5, 0.0),
                        ),
                        "method_kwargs": {
                            "xerr": (0.05, 0.1, 0.15),
                            "yerr": 0.2,
                            "color": "C1",
                            "linestyle": "--",
                            "marker": "o",
                            "capsize": 3.0,
                            "label": "manual errorbar",
                        },
                    }
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    tool.operation_list.setCurrentItem(tool.operation_list.topLevelItem(0))
    tool._select_step_section("method")
    method_page = tool.step_editor_stack.currentWidget()
    method_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodCombo"
    )
    x_edit = method_page.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodErrorbarXEdit"
    )
    y_edit = method_page.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodErrorbarYEdit"
    )
    xerr_edit = method_page.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodErrorbarXErrorEdit"
    )
    yerr_edit = method_page.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodErrorbarYErrorEdit"
    )
    capsize_spin = method_page.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerAxesMethodErrorbarCapSizeSpin"
    )
    kwargs_edit = method_page.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodKwEdit"
    )
    assert method_combo is not None
    assert x_edit is not None
    assert y_edit is not None
    assert xerr_edit is not None
    assert yerr_edit is not None
    assert capsize_spin is not None
    assert kwargs_edit is not None
    assert method_combo.currentData() == "errorbar"
    assert x_edit.text() == "0.0, 0.5, 1.0"
    assert y_edit.text() == "1.0, 0.5, 0.0"
    assert xerr_edit.text() == "0.05, 0.1, 0.15"
    assert yerr_edit.text() == "0.2"
    assert capsize_spin.value() == pytest.approx(3.0)
    assert kwargs_edit.text() == ""

    xerr_edit.setText("0.1, 0.2, 0.3")
    xerr_edit.setModified(True)
    xerr_edit.editingFinished.emit()
    assert tool.tool_status.operations[0].method_kwargs["xerr"] == (0.1, 0.2, 0.3)
    capsize_spin.setValue(4.0)
    assert tool.tool_status.operations[0].method_kwargs["capsize"] == pytest.approx(4.0)

    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    assert tool._operation_render_errors == {}
    container = tool.figure.axes[0].containers[0]
    assert isinstance(container, ErrorbarContainer)
    line = container.lines[0]
    np.testing.assert_allclose(
        np.asarray(line.get_xdata(), dtype=float), (0.0, 0.5, 1.0)
    )
    np.testing.assert_allclose(
        np.asarray(line.get_ydata(), dtype=float), (1.0, 0.5, 0.0)
    )
    assert line.get_color() == "C1"
    assert line.get_linestyle() == "--"
    assert line.get_marker() == "o"
    assert container.get_label() == "manual errorbar"
    _assert_errorbar_xerr(container, (0.0, 0.5, 1.0), (1.0, 0.5, 0.0), (0.1, 0.2, 0.3))
    _assert_errorbar_capsize(container, 4.0)

    code = tool.generated_code()
    assert "axs[0, 0].errorbar((0.0, 0.5, 1.0), (1.0, 0.5, 0.0)" in code
    assert "xerr=(0.1, 0.2, 0.3)" in code
    assert "yerr=0.2" in code
    assert "capsize=4.0" in code

    namespace: dict[str, typing.Any] = {}
    exec(code, namespace)  # noqa: S102
    generated_container = namespace["axs"][0, 0].containers[0]
    assert isinstance(generated_container, ErrorbarContainer)
    generated_line = generated_container.lines[0]
    np.testing.assert_allclose(
        np.asarray(generated_line.get_xdata(), dtype=float), (0.0, 0.5, 1.0)
    )
    np.testing.assert_allclose(
        np.asarray(generated_line.get_ydata(), dtype=float), (1.0, 0.5, 0.0)
    )
    _assert_errorbar_xerr(
        generated_container,
        (0.0, 0.5, 1.0),
        (1.0, 0.5, 0.0),
        (0.1, 0.2, 0.3),
    )
    _assert_errorbar_capsize(generated_container, 4.0)


@pytest.mark.parametrize(
    "method_args",
    [
        ((1.0, 0.5, 0.0),),
        ((), (1.0, 0.5, 0.0)),
        ((0.0, 0.5, 1.0), ()),
    ],
)
def test_figure_composer_axes_errorbar_method_requires_entered_x_and_y(
    qtbot,
    method_args: tuple[tuple[float, ...], ...],
) -> None:
    data = xr.DataArray(
        np.arange(3.0),
        dims=("kx",),
        coords={"kx": [0.0, 0.5, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="errorbar",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ).model_copy(
                    update={
                        "method_args": method_args,
                        "method_kwargs": {
                            "xerr": (0.1, 0.2, 0.3),
                            "yerr": (0.2, 0.2, 0.2),
                            "marker": "o",
                        },
                    }
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    errors = tuple(tool._operation_render_errors.values())
    assert len(errors) == 1
    assert "Enter X and Y values for ax.errorbar" in errors[0]
    with pytest.raises(ValueError, match=r"Enter X and Y values for ax\.errorbar"):
        tool.generated_code()


def test_figure_composer_axes_plot_data_mode_switches_editor(qtbot) -> None:
    data = xr.DataArray(
        np.array([1.0, 2.0, 3.0]),
        dims=("kx",),
        coords={"kx": [0.0, 0.5, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="plot",
                ),
            ),
            primary_source="data",
        ),
        source_data={"data": data},
    )
    qtbot.addWidget(tool)

    tool.operation_list.setCurrentItem(tool.operation_list.topLevelItem(0))
    tool._select_step_section("method")
    method_page = tool.step_editor_stack.currentWidget()
    mode_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodPlotDataModeCombo"
    )
    assert mode_combo is not None
    assert mode_combo.currentData() == "entered"
    assert (
        method_page.findChild(QtWidgets.QLineEdit, "figureComposerAxesMethodPlotXEdit")
        is not None
    )
    assert (
        method_page.findChild(QtWidgets.QLineEdit, "figureComposerAxesMethodPlotYEdit")
        is not None
    )

    _activate_combo_index(mode_combo, mode_combo.findData("from_data"))
    qtbot.wait_until(
        lambda: (
            tool.step_editor_stack.currentWidget().findChild(
                QtWidgets.QComboBox, "figureComposerAxesMethodPlotYSourceCombo"
            )
            is not None
        ),
        timeout=5000,
    )

    operation = tool.tool_status.operations[0]
    assert operation.method_plot_data_mode == "from_data"
    assert operation.method_plot_y == FigureMethodPlotValueState(
        source="data", kind="data"
    )
    method_page = tool.step_editor_stack.currentWidget()
    assert (
        method_page.findChild(QtWidgets.QLineEdit, "figureComposerAxesMethodPlotXEdit")
        is None
    )
    x_source_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodPlotXSourceCombo"
    )
    x_values_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodPlotXValuesCombo"
    )
    y_source_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodPlotYSourceCombo"
    )
    y_values_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodPlotYValuesCombo"
    )
    assert x_source_combo is not None
    assert x_values_combo is not None
    assert y_source_combo is not None
    assert y_values_combo is not None
    assert x_source_combo.property("figureComposerPlotDataRole") == "x_source"
    assert x_values_combo.property("figureComposerPlotDataRole") == "x_values"
    assert y_source_combo.property("figureComposerPlotDataRole") == "y_source"
    assert y_values_combo.property("figureComposerPlotDataRole") == "y_values"
    assert x_source_combo.toolTip()
    assert x_values_combo.toolTip()
    assert y_source_combo.toolTip()
    assert y_values_combo.toolTip()
    assert x_source_combo.toolTip() != x_values_combo.toolTip()
    assert y_source_combo.toolTip() != y_values_combo.toolTip()
    assert x_source_combo.itemText(0) != x_values_combo.itemText(0)
    assert x_source_combo.currentData() is None
    assert y_source_combo.currentData() == "data"
    y_value_options = {
        y_values_combo.itemData(index) for index in range(y_values_combo.count())
    }
    assert y_value_options >= {
        ("data", None),
        ("coord", "kx"),
    }
    assert all(
        y_values_combo.itemData(index, QtCore.Qt.ItemDataRole.ToolTipRole)
        for index in range(y_values_combo.count())
        if y_values_combo.itemData(index) is not None
    )

    figurecomposer_method._update_current_method_name(tool, "errorbar")
    tool._select_step_section("method")
    qtbot.wait_until(
        lambda: (
            tool.step_editor_stack.currentWidget().findChild(
                QtWidgets.QComboBox,
                "figureComposerAxesMethodErrorbarXErrorSourceCombo",
            )
            is not None
        ),
        timeout=5000,
    )
    operation = tool.tool_status.operations[0]
    assert operation.method_name == "errorbar"
    assert operation.method_plot_data_mode == "from_data"
    assert operation.method_plot_y == FigureMethodPlotValueState(
        source="data", kind="data"
    )
    method_page = tool.step_editor_stack.currentWidget()
    xerr_source_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodErrorbarXErrorSourceCombo"
    )
    xerr_values_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodErrorbarXErrorValuesCombo"
    )
    yerr_source_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodErrorbarYErrorSourceCombo"
    )
    yerr_values_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodErrorbarYErrorValuesCombo"
    )
    assert xerr_source_combo is not None
    assert xerr_values_combo is not None
    assert yerr_source_combo is not None
    assert yerr_values_combo is not None
    assert xerr_source_combo.property("figureComposerPlotDataRole") == "xerr_source"
    assert xerr_values_combo.property("figureComposerPlotDataRole") == "xerr_values"
    assert yerr_source_combo.property("figureComposerPlotDataRole") == "yerr_source"
    assert yerr_values_combo.property("figureComposerPlotDataRole") == "yerr_values"
    assert xerr_source_combo.currentData() is None
    assert yerr_source_combo.currentData() is None

    figurecomposer_method._update_current_method_name(tool, "plot")
    operation = tool.tool_status.operations[0]
    assert operation.method_name == "plot"
    assert operation.method_plot_data_mode == "from_data"
    assert operation.method_plot_xerr is None
    assert operation.method_plot_yerr is None


def test_figure_composer_axes_plot_method_picks_data_render_and_codegen(qtbot) -> None:
    x_source = xr.DataArray(
        np.array([10.0, 20.0, 30.0]),
        dims=("kx",),
        coords={"kx": [0.0, 0.5, 1.0]},
        name="x_source",
    )
    y_source = xr.DataArray(
        np.array([1.0, 4.0, 9.0]),
        dims=("point",),
        coords={"point": [0, 1, 2]},
        name="y_source",
    )
    tool = FigureComposerTool(
        y_source,
        recipe=FigureRecipeState(
            sources=(
                FigureSourceState(name="x_data", label="x data"),
                FigureSourceState(name="y_data", label="y data"),
            ),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="plot",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ).model_copy(
                    update={
                        "method_plot_data_mode": "from_data",
                        "method_plot_x": FigureMethodPlotValueState(
                            source="x_data", kind="coord", name="kx"
                        ),
                        "method_plot_y": FigureMethodPlotValueState(
                            source="y_data", kind="data"
                        ),
                        "method_kwargs": {"color": "C2", "label": "picked"},
                    }
                ),
            ),
            primary_source="y_data",
        ),
        source_data={"x_data": x_source, "y_data": y_source},
    )
    qtbot.addWidget(tool)

    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    assert tool._operation_render_errors == {}
    line = tool.figure.axes[0].lines[0]
    np.testing.assert_allclose(line.get_xdata(), x_source.coords["kx"].values)
    np.testing.assert_allclose(line.get_ydata(), y_source.values)
    assert line.get_color() == "C2"
    assert line.get_label() == "picked"

    code = tool.generated_code()
    assert 'x_data.coords["kx"].values' in code
    assert "y_data.values" in code
    namespace: dict[str, typing.Any] = {
        "x_data": x_source,
        "y_data": y_source,
    }
    exec(code, namespace)  # noqa: S102
    generated_line = namespace["axs"][0, 0].lines[0]
    np.testing.assert_allclose(generated_line.get_xdata(), x_source.coords["kx"].values)
    np.testing.assert_allclose(generated_line.get_ydata(), y_source.values)


def test_figure_composer_axes_errorbar_method_picks_data_render_and_codegen(
    qtbot,
) -> None:
    values_source = xr.DataArray(
        np.array([0.1, 0.2, 0.3]),
        dims=("scan",),
        coords={"scan": [10.0, 20.0, 30.0]},
        name="values",
    )
    stderr_source = xr.DataArray(
        np.array([0.01, 0.02, 0.03]),
        dims=("scan",),
        coords={"scan": [10.0, 20.0, 30.0]},
        name="stderr",
    )
    yerr_source = xr.DataArray(
        np.array([1.0, 2.0, 3.0]),
        dims=("scan",),
        coords={"scan": [10.0, 20.0, 30.0]},
        name="yerr",
    )
    tool = FigureComposerTool(
        values_source,
        recipe=FigureRecipeState(
            sources=(
                FigureSourceState(name="values_data", label="values data"),
                FigureSourceState(name="stderr_data", label="stderr data"),
                FigureSourceState(name="yerr_data", label="y error data"),
            ),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="errorbar",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ).model_copy(
                    update={
                        "method_plot_data_mode": "from_data",
                        "method_plot_x": FigureMethodPlotValueState(
                            source="values_data", kind="data"
                        ),
                        "method_plot_y": FigureMethodPlotValueState(
                            source="values_data", kind="coord", name="scan"
                        ),
                        "method_plot_xerr": FigureMethodPlotValueState(
                            source="stderr_data", kind="data"
                        ),
                        "method_plot_yerr": FigureMethodPlotValueState(
                            source="yerr_data", kind="data"
                        ),
                        "method_kwargs": {
                            "xerr": 99.0,
                            "yerr": 99.0,
                            "color": "C3",
                            "label": "picked errorbar",
                            "linestyle": "none",
                            "marker": "o",
                        },
                    }
                ),
            ),
            primary_source="values_data",
        ),
        source_data={
            "values_data": values_source,
            "stderr_data": stderr_source,
            "yerr_data": yerr_source,
        },
    )
    qtbot.addWidget(tool)

    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    assert tool._operation_render_errors == {}
    container = tool.figure.axes[0].containers[0]
    assert isinstance(container, ErrorbarContainer)
    line = container.lines[0]
    np.testing.assert_allclose(line.get_xdata(), values_source.values)
    np.testing.assert_allclose(line.get_ydata(), values_source.coords["scan"].values)
    assert line.get_color() == "C3"
    assert container.get_label() == "picked errorbar"
    _assert_errorbar_xerr(
        container,
        values_source.values,
        values_source.coords["scan"].values,
        stderr_source.values,
    )
    _assert_errorbar_yerr(
        container,
        values_source.values,
        values_source.coords["scan"].values,
        yerr_source.values,
    )

    code = tool.generated_code()
    assert "values_data.values" in code
    assert 'values_data.coords["scan"].values' in code
    assert "xerr=stderr_data.values" in code
    assert "yerr=yerr_data.values" in code
    assert "xerr=99.0" not in code
    assert "yerr=99.0" not in code
    namespace: dict[str, typing.Any] = {
        "values_data": values_source,
        "stderr_data": stderr_source,
        "yerr_data": yerr_source,
    }
    exec(code, namespace)  # noqa: S102
    generated_container = namespace["axs"][0, 0].containers[0]
    assert isinstance(generated_container, ErrorbarContainer)
    generated_line = generated_container.lines[0]
    np.testing.assert_allclose(generated_line.get_xdata(), values_source.values)
    np.testing.assert_allclose(
        generated_line.get_ydata(), values_source.coords["scan"].values
    )
    _assert_errorbar_xerr(
        generated_container,
        values_source.values,
        values_source.coords["scan"].values,
        stderr_source.values,
    )
    _assert_errorbar_yerr(
        generated_container,
        values_source.values,
        values_source.coords["scan"].values,
        yerr_source.values,
    )


def test_figure_composer_axes_errorbar_method_requires_picked_x_and_y(
    qtbot,
) -> None:
    values_source = xr.DataArray(
        np.array([1.0, 4.0, 9.0]),
        dims=("scan",),
        coords={"scan": [10.0, 20.0, 30.0]},
        name="values",
    )
    stderr_source = xr.DataArray(
        np.array([0.1, 0.2, 0.3]),
        dims=("scan",),
        coords={"scan": [10.0, 20.0, 30.0]},
        name="stderr",
    )
    yerr_source = xr.DataArray(
        np.array([0.4, 0.5, 0.6]),
        dims=("scan",),
        coords={"scan": [10.0, 20.0, 30.0]},
        name="yerr",
    )
    tool = FigureComposerTool(
        values_source,
        recipe=FigureRecipeState(
            sources=(
                FigureSourceState(name="values_data", label="values data"),
                FigureSourceState(name="stderr_data", label="stderr data"),
                FigureSourceState(name="yerr_data", label="y error data"),
            ),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="errorbar",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ).model_copy(
                    update={
                        "method_plot_data_mode": "from_data",
                        "method_plot_y": FigureMethodPlotValueState(
                            source="values_data", kind="data"
                        ),
                        "method_plot_xerr": FigureMethodPlotValueState(
                            source="stderr_data", kind="data"
                        ),
                        "method_plot_yerr": FigureMethodPlotValueState(
                            source="yerr_data", kind="data"
                        ),
                    }
                ),
            ),
            primary_source="values_data",
        ),
        source_data={
            "values_data": values_source,
            "stderr_data": stderr_source,
            "yerr_data": yerr_source,
        },
    )
    qtbot.addWidget(tool)

    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    errors = tuple(tool._operation_render_errors.values())
    assert len(errors) == 1
    assert "Choose X values for ax.errorbar" in errors[0]
    with pytest.raises(ValueError, match=r"Choose X values for ax\.errorbar"):
        tool.generated_code()


@pytest.mark.parametrize(
    ("source_data", "x_state", "y_state", "message"),
    [
        (
            {"data": xr.DataArray([1.0, 2.0], dims=("x",), name="data")},
            None,
            FigureMethodPlotValueState(source="data", kind="coord", name="missing"),
            "Coordinate 'missing' is not available",
        ),
        (
            {"image": xr.DataArray(np.ones((2, 2)), dims=("x", "y"), name="image")},
            None,
            FigureMethodPlotValueState(source="image", kind="data"),
            "one-dimensional",
        ),
        (
            {
                "x": xr.DataArray([0.0, 1.0, 2.0], dims=("x",), name="x"),
                "y": xr.DataArray([1.0, 2.0], dims=("y",), name="y"),
            },
            FigureMethodPlotValueState(source="x", kind="data"),
            FigureMethodPlotValueState(source="y", kind="data"),
            "same length",
        ),
        (
            {"data": xr.DataArray([1.0, 2.0], dims=("x",), name="data")},
            None,
            FigureMethodPlotValueState(source="missing", kind="data"),
            "is not available",
        ),
    ],
)
def test_figure_composer_axes_plot_picked_data_invalid_inputs(
    qtbot,
    source_data: dict[str, xr.DataArray],
    x_state: FigureMethodPlotValueState | None,
    y_state: FigureMethodPlotValueState,
    message: str,
) -> None:
    primary_source = next(iter(source_data))
    tool = FigureComposerTool(
        source_data[primary_source],
        recipe=FigureRecipeState(
            sources=tuple(
                FigureSourceState(name=name, label=name) for name in source_data
            ),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="plot",
                ).model_copy(
                    update={
                        "method_plot_data_mode": "from_data",
                        "method_plot_x": x_state,
                        "method_plot_y": y_state,
                    }
                ),
            ),
            primary_source=primary_source,
        ),
        source_data=source_data,
    )
    qtbot.addWidget(tool)

    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    errors = tuple(tool._operation_render_errors.values())
    assert len(errors) == 1
    assert message in errors[0]
    with pytest.raises(ValueError, match=message):
        tool.generated_code()


@pytest.mark.parametrize("axis", ["xerr", "yerr"])
def test_figure_composer_axes_errorbar_picked_error_data_invalid_lengths(
    qtbot,
    axis: str,
) -> None:
    values_source = xr.DataArray(
        np.array([0.1, 0.2]),
        dims=("scan",),
        coords={"scan": [10.0, 20.0]},
        name="values",
    )
    stderr_source = xr.DataArray(
        np.array([0.01, 0.02, 0.03]),
        dims=("scan",),
        name="stderr",
    )
    tool = FigureComposerTool(
        values_source,
        recipe=FigureRecipeState(
            sources=(
                FigureSourceState(name="values_data", label="values data"),
                FigureSourceState(name="stderr_data", label="stderr data"),
            ),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="errorbar",
                ).model_copy(
                    update={
                        "method_plot_data_mode": "from_data",
                        "method_plot_x": FigureMethodPlotValueState(
                            source="values_data", kind="data"
                        ),
                        "method_plot_y": FigureMethodPlotValueState(
                            source="values_data", kind="coord", name="scan"
                        ),
                        f"method_plot_{axis}": FigureMethodPlotValueState(
                            source="stderr_data", kind="data"
                        ),
                    }
                ),
            ),
            primary_source="values_data",
        ),
        source_data={"values_data": values_source, "stderr_data": stderr_source},
    )
    qtbot.addWidget(tool)

    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    errors = tuple(tool._operation_render_errors.values())
    assert len(errors) == 1
    assert "same length" in errors[0]
    with pytest.raises(ValueError, match="same length"):
        tool.generated_code()


def test_figure_composer_axes_plot_picked_data_sources_are_renamed() -> None:
    operation = FigureOperationState.method(
        family=FigureMethodFamily.AXES,
        name="plot",
    ).model_copy(
        update={
            "method_plot_data_mode": "from_data",
            "method_plot_x": FigureMethodPlotValueState(
                source="x_data", kind="coord", name="kx"
            ),
            "method_plot_y": FigureMethodPlotValueState(source="y_data", kind="data"),
        }
    )

    assert figurecomposer_operation_metadata.declared_operation_source_names(
        operation
    ) == ("x_data", "y_data")
    renamed = figurecomposer_operation_metadata.rename_operation_sources(
        operation, {"x_data": "x_copy", "y_data": "y_copy"}
    )

    assert renamed.method_plot_x == FigureMethodPlotValueState(
        source="x_copy", kind="coord", name="kx"
    )
    assert renamed.method_plot_y == FigureMethodPlotValueState(
        source="y_copy", kind="data"
    )

    errorbar_operation = operation.model_copy(
        update={
            "method_name": "errorbar",
            "method_plot_xerr": FigureMethodPlotValueState(
                source="stderr_data", kind="data"
            ),
            "method_plot_yerr": FigureMethodPlotValueState(
                source="stderr_data", kind="coord", name="kx"
            ),
        }
    )
    assert figurecomposer_operation_metadata.declared_operation_source_names(
        errorbar_operation
    ) == (
        "x_data",
        "y_data",
        "stderr_data",
    )
    renamed_errorbar = figurecomposer_operation_metadata.rename_operation_sources(
        errorbar_operation,
        {"x_data": "x_copy", "y_data": "y_copy", "stderr_data": "stderr_copy"},
    )
    assert renamed_errorbar.method_plot_xerr == FigureMethodPlotValueState(
        source="stderr_copy", kind="data"
    )
    assert renamed_errorbar.method_plot_yerr == FigureMethodPlotValueState(
        source="stderr_copy", kind="coord", name="kx"
    )


def test_figure_composer_axes_plot_data_helper_edges() -> None:
    source = xr.DataArray(
        np.arange(6.0).reshape(1, 6),
        dims=("scan", "point"),
        coords={"scan": [0], "point": np.linspace(0.0, 1.0, 6)},
        name="source",
    )
    coord_grid = xr.DataArray(
        np.ones((1, 6)),
        dims=("scan", "point"),
        coords=source.coords,
        name="coord_grid",
    )
    source_with_grid_coord = source.assign_coords(grid=coord_grid)
    source_with_2d_coord = xr.DataArray(
        np.ones((2, 6)),
        dims=("scan", "point"),
        coords={"scan": [0, 1], "point": np.linspace(0.0, 1.0, 6)},
        name="source_with_2d_coord",
    ).assign_coords(
        grid=(
            ("scan", "point"),
            np.arange(12.0).reshape(2, 6),
        )
    )
    tool = types.SimpleNamespace(
        _document=types.SimpleNamespace(
            source_data={"source": source_with_grid_coord},
            source_names=lambda: ("source",),
        ),
        _source_display_name=lambda name: f"display {name}",
        _source_tooltip=lambda name: f"tooltip {name}",
    )

    assert figurecomposer_method._plot_data_mode_text("unknown") == "Enter values"
    assert figurecomposer_method._plot_value_display(None) == "Choose values"
    assert figurecomposer_method._plot_value_options(tool, None) == ()
    assert figurecomposer_method._plot_value_options(tool, "missing") == ()
    option_tool = types.SimpleNamespace(
        _document=types.SimpleNamespace(
            source_data={"source": source_with_2d_coord},
            source_names=lambda: ("source",),
        ),
        _source_display_name=lambda name: f"display {name}",
        _source_tooltip=lambda name: f"tooltip {name}",
    )
    assert ("coord", "grid") not in {
        value
        for _text, value in figurecomposer_method._plot_value_options(
            option_tool, "source"
        )
    }
    assert figurecomposer_method._plot_coord_by_name(source, "point") is not None
    assert figurecomposer_method._plot_source_combo_tooltip("x", has_sources=False)
    xerr_source_tooltip = figurecomposer_method._plot_source_combo_tooltip(
        "xerr", has_sources=True
    )
    yerr_source_tooltip = figurecomposer_method._plot_source_combo_tooltip(
        "yerr", has_sources=True
    )
    assert xerr_source_tooltip.replace("x error", "error") == (
        yerr_source_tooltip.replace("y error", "error")
    )
    assert figurecomposer_method._plot_values_combo_tooltip(
        "y", source="source", value_options_match=False
    )
    xerr_values_tooltip = figurecomposer_method._plot_values_combo_tooltip(
        "xerr", source="source", value_options_match=True
    )
    yerr_values_tooltip = figurecomposer_method._plot_values_combo_tooltip(
        "yerr", source="source", value_options_match=True
    )
    assert xerr_values_tooltip.replace("x error", "error") == (
        yerr_values_tooltip.replace("y error", "error")
    )
    with pytest.raises(ValueError, match="Unknown plot value selection"):
        figurecomposer_method._plot_value_combo_data_parts(("bad", None))

    data_state = FigureMethodPlotValueState(source="source", kind="data")
    code, value = figurecomposer_method._plot_value_code_and_data(tool, data_state)
    assert str(code) == "source.squeeze(drop=True).values"
    assert value.dims == ("point",)

    coord_state = FigureMethodPlotValueState(
        source="source", kind="coord", name="point"
    )
    coord_code, coord_value = figurecomposer_method._plot_value_code_and_data(
        tool, coord_state
    )
    assert str(coord_code) == 'source.coords["point"].values'
    assert coord_value.dims == ("point",)

    with pytest.raises(ValueError, match="Choose a coordinate"):
        figurecomposer_method._plot_value_data(
            tool, FigureMethodPlotValueState(source="source", kind="coord")
        )
    with pytest.raises(ValueError, match="one-dimensional"):
        figurecomposer_method._plot_value_data(
            option_tool,
            FigureMethodPlotValueState(source="source", kind="coord", name="grid"),
        )
    with pytest.raises(ValueError, match="Choose Y values"):
        figurecomposer_method._picked_plot_args(
            tool,
            FigureOperationState.method(
                family=FigureMethodFamily.AXES,
                name="plot",
            ).model_copy(update={"method_plot_data_mode": "from_data"}),
            figurecomposer_method.AXES_METHODS["plot"],
        )

    operation = FigureOperationState.method(
        family=FigureMethodFamily.AXES,
        name="plot",
    ).model_copy(
        update={
            "method_plot_data_mode": "from_data",
            "method_plot_y": data_state,
        }
    )
    args = figurecomposer_method._picked_plot_args(
        tool, operation, figurecomposer_method.AXES_METHODS["plot"]
    )
    assert len(args) == 1
    np.testing.assert_allclose(args[0], source.values.reshape(-1))
    code_args = figurecomposer_method._picked_plot_code_args(
        tool, operation, figurecomposer_method.AXES_METHODS["plot"]
    )
    assert tuple(str(arg) for arg in code_args) == ("source.squeeze(drop=True).values",)


def test_figure_composer_axes_plot_data_update_helpers() -> None:
    operation = FigureOperationState.method(
        family=FigureMethodFamily.AXES,
        name="plot",
    ).model_copy(
        update={
            "method_plot_data_mode": "entered",
            "method_plot_x": FigureMethodPlotValueState(source="old", kind="data"),
        }
    )
    data = xr.DataArray([1.0, 2.0], dims=("x",), name="data")
    calls: list[bool] = []

    class UpdateTool:
        def __init__(self) -> None:
            self._document = types.SimpleNamespace(
                source_data={"data": data}, source_names=lambda: ("data",)
            )
            self.operation = operation

        def _update_operations(
            self,
            updater: Callable[[int, FigureOperationState], FigureOperationState],
            *,
            rebuild_editor: bool = False,
        ) -> None:
            calls.append(rebuild_editor)
            self.operation = updater(0, self.operation)

    tool = UpdateTool()
    figurecomposer_method._update_current_plot_data_mode(tool, "invalid")
    assert calls == []

    figurecomposer_method._update_current_plot_data_mode(tool, "from_data")
    assert calls[-1] is True
    assert tool.operation.method_plot_data_mode == "from_data"
    assert tool.operation.method_plot_y == FigureMethodPlotValueState(
        source="data", kind="data"
    )

    figurecomposer_method._update_current_plot_value_source(tool, "x", None)
    assert tool.operation.method_plot_x is None
    figurecomposer_method._update_current_plot_value_source(tool, "x", "data")
    assert tool.operation.method_plot_x == FigureMethodPlotValueState(
        source="data", kind="data"
    )
    figurecomposer_method._update_current_plot_value_selection(
        tool, "x", ("coord", "x")
    )
    assert tool.operation.method_plot_x == FigureMethodPlotValueState(
        source="data", kind="coord", name="x"
    )
    figurecomposer_method._update_current_plot_value_selection(tool, "x", None)
    assert tool.operation.method_plot_x is None

    assert (
        figurecomposer_operation_metadata.declared_operation_source_names(
            FigureOperationState.method(
                family=FigureMethodFamily.AXES,
                name="missing",
            )
        )
        == ()
    )


def test_figure_composer_axes_plot_optional_bool_editor_branch(qtbot) -> None:
    operation = FigureOperationState.method(
        family=FigureMethodFamily.AXES,
        name="tick_params",
        kwargs={"reset": True},
    )
    layout_parent = QtWidgets.QWidget()
    qtbot.addWidget(layout_parent)
    layout = QtWidgets.QFormLayout(layout_parent)
    updates: list[FigureOperationState] = []

    class EditorTool:
        def _batch_is_mixed(
            self,
            _operation: FigureOperationState,
            _getter: Callable[[FigureOperationState], object],
        ) -> bool:
            return False

        def _optional_name_combo(
            self,
            names: Sequence[str],
            current: str | None,
            none_label: str,
            callback: Callable[[str | None], None],
            *,
            parent: QtWidgets.QWidget | None = None,
            mixed: bool = False,
        ) -> QtWidgets.QComboBox:
            combo = QtWidgets.QComboBox(parent)
            combo.addItem(none_label, None)
            for name in names:
                combo.addItem(name, name)
            if current is not None:
                combo.setCurrentText(current)
            combo.setProperty("figureComposerMixedValue", mixed)
            combo.activated.connect(lambda _index: callback(combo.currentData()))
            return combo

        def _add_form_row(
            self,
            form: QtWidgets.QFormLayout,
            _label: str,
            widget: QtWidgets.QWidget,
            _tooltip: str,
        ) -> None:
            form.addRow(widget)

        def _update_operations(
            self,
            updater: Callable[[int, FigureOperationState], FigureOperationState],
        ) -> None:
            updates.append(updater(0, operation))

    control = figurecomposer_method._optional_bool_kwarg_combo(
        "Reset",
        "reset",
        "figureComposerTestResetCombo",
        "tooltip",
    )
    figurecomposer_method._add_method_control_row(
        EditorTool(),
        layout,
        operation,
        figurecomposer_method.AXES_METHODS["tick_params"],
        control,
    )
    combo = layout_parent.findChild(QtWidgets.QComboBox, "figureComposerTestResetCombo")
    assert combo is not None
    assert combo.currentText() == "True"

    combo.setCurrentText("False")
    combo.activated.emit(combo.currentIndex())
    assert updates[-1].method_kwargs["reset"] is False


def test_figure_composer_axes_plot_combo_helper_edges(qtbot) -> None:
    source = xr.DataArray(
        np.arange(6.0).reshape(1, 6),
        dims=("scan", "point"),
        coords={
            "scan": [0],
            "point": np.linspace(0.0, 1.0, 6),
            "scan_coord": ("scan", [10.0]),
            "row_coord": (("scan", "point"), np.arange(6.0).reshape(1, 6)),
        },
        name="source",
    )

    class ComboTool:
        def __init__(self) -> None:
            self._document = types.SimpleNamespace(
                source_data={"source": source}, source_names=lambda: ("source",)
            )
            self.operation_editor = QtWidgets.QWidget()
            self.operation = FigureOperationState.method(
                family=FigureMethodFamily.AXES,
                name="plot",
            )

        def _mark_editor_control(self, _widget: QtWidgets.QWidget) -> None:
            return None

        def _connect_editor_signal(
            self,
            _widget: QtWidgets.QWidget,
            signal: typing.Any,
            callback: Callable[..., None],
        ) -> None:
            signal.connect(callback)

        def _source_names(self) -> tuple[str, ...]:
            return ("source",)

        def _source_display_name(self, name: str) -> str:
            return f"display {name}"

        def _source_tooltip(self, name: str) -> str:
            return f"tooltip {name}"

        def _update_operations(
            self,
            updater: Callable[[int, FigureOperationState], FigureOperationState],
        ) -> None:
            self.operation = updater(0, self.operation)

    tool = ComboTool()
    qtbot.addWidget(tool.operation_editor)
    changed: list[object] = []

    mode_combo = figurecomposer_method._plot_data_mode_combo(
        tool,
        current=None,
        changed=changed.append,
        parent=tool.operation_editor,
        mixed=True,
    )
    assert mode_combo.currentData() is _editor_controls.MIXED_VALUE
    assert not typing.cast("typing.Any", mode_combo.model()).item(0).isEnabled()

    source_combo = figurecomposer_method._plot_source_combo(
        tool,
        current=None,
        changed=changed.append,
        axis="y",
        parent=tool.operation_editor,
        allow_none=False,
        mixed=True,
    )
    assert source_combo.currentData() is _editor_controls.MIXED_VALUE
    assert not typing.cast("typing.Any", source_combo.model()).item(0).isEnabled()

    values_combo = figurecomposer_method._plot_values_combo(
        tool,
        source="source",
        current=None,
        changed=changed.append,
        axis="y",
        parent=tool.operation_editor,
        allow_none=False,
        mixed=True,
    )
    assert values_combo.currentData() is _editor_controls.MIXED_VALUE
    assert not typing.cast("typing.Any", values_combo.model()).item(0).isEnabled()

    no_source_values = figurecomposer_method._plot_values_combo(
        tool,
        source=None,
        current=None,
        changed=changed.append,
        axis="y",
        parent=tool.operation_editor,
        allow_none=False,
    )
    assert no_source_values.itemData(0) is None
    assert not no_source_values.isEnabled()
    assert figurecomposer_method._plot_values_combo_tooltip(
        "y", source=None, value_options_match=True
    )

    squeezed_coord_state = FigureMethodPlotValueState(
        source="source", kind="coord", name="row_coord"
    )
    code, value = figurecomposer_method._plot_value_code_and_data(
        tool, squeezed_coord_state
    )
    assert str(code) == 'source.coords["row_coord"].squeeze(drop=True).values'
    assert value.dims == ("point",)

    with pytest.raises(ValueError, match="Choose a coordinate"):
        figurecomposer_method._plot_value_code_and_data(
            tool, FigureMethodPlotValueState(source="source", kind="coord")
        )
    with pytest.raises(ValueError, match="one-dimensional"):
        figurecomposer_method._plot_value_code_and_data(
            tool,
            FigureMethodPlotValueState(
                source="source", kind="coord", name="scan_coord"
            ),
        )
    with pytest.raises(ValueError, match="Choose Y values"):
        figurecomposer_method._picked_plot_code_args(
            tool,
            FigureOperationState.method(
                family=FigureMethodFamily.AXES,
                name="plot",
            ).model_copy(update={"method_plot_data_mode": "from_data"}),
            figurecomposer_method.AXES_METHODS["plot"],
        )

    callback = figurecomposer_method._method_optional_bool_kwarg_callback(
        tool, "visible"
    )
    callback(None)
    assert "visible" not in tool.operation.method_kwargs
    callback("True")
    assert tool.operation.method_kwargs["visible"] is True

    optional_bool_control = figurecomposer_method.MethodControlSpec(
        kind=figurecomposer_method.MethodControlKind.OPTIONAL_BOOL_KWARG_COMBO,
        label="Visible",
        tooltip="",
        object_name="visible",
    )
    assert figurecomposer_method._control_accepts_value(optional_bool_control, None)
    assert figurecomposer_method._control_accepts_value(optional_bool_control, False)
    assert not figurecomposer_method._control_accepts_value(
        optional_bool_control, "False"
    )


@pytest.mark.parametrize(
    ("mode", "code_fragment"),
    [
        ("figure", "fig.transFigure"),
        ("dpi", "fig.dpi_scale_trans"),
        ("xaxis", "ax.get_xaxis_transform()"),
        ("yaxis", "ax.get_yaxis_transform()"),
    ],
)
def test_figure_composer_method_transform_presets(
    mode: str, code_fragment: str
) -> None:
    figure, axis = plt.subplots()
    spec = figurecomposer_method.AXES_METHODS["plot"]
    operation = FigureOperationState.method(
        family=FigureMethodFamily.AXES,
        name="plot",
    ).model_copy(update={"method_transform": mode})

    transform = figurecomposer_method._render_method_transform(
        operation,
        spec,
        figure=figure,
        axis=axis,
    )
    code = figurecomposer_method._method_transform_code(operation, spec)

    assert isinstance(transform, mtransforms.Transform)
    assert repr(code) == code_fragment
    plt.close(figure)


def test_figure_composer_method_custom_transform_errors() -> None:
    figure, axis = plt.subplots()
    spec = figurecomposer_method.AXES_METHODS["plot"]
    untrusted = FigureOperationState.method(
        family=FigureMethodFamily.AXES,
        name="plot",
    ).model_copy(
        update={
            "method_transform": "custom",
            "method_transform_expression": "ax.transAxes",
        }
    )
    empty = untrusted.model_copy(
        update={"trusted": True, "method_transform_expression": ""}
    )
    not_transform = untrusted.model_copy(
        update={"trusted": True, "method_transform_expression": "1"}
    )

    with pytest.raises(ValueError, match="not trusted"):
        figurecomposer_method._method_transform_code(untrusted, spec)
    with pytest.raises(ValueError, match="empty"):
        figurecomposer_method._method_transform_code(empty, spec)
    with pytest.raises(ValueError, match="empty"):
        figurecomposer_method._render_method_transform(
            empty,
            spec,
            figure=figure,
            axis=axis,
        )
    with pytest.raises(TypeError, match="Transform"):
        figurecomposer_method._render_method_transform(
            not_transform,
            spec,
            figure=figure,
            axis=axis,
        )
    plt.close(figure)


def test_figure_composer_loaded_custom_method_transform_requires_trust(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="plot",
                ).model_copy(
                    update={
                        "method_args": ((0.0, 1.0),),
                        "method_transform": "custom",
                        "method_transform_expression": "ax.transAxes",
                        "trusted": True,
                    }
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    loaded = erlab.interactive.utils.ToolWindow.from_dataset(tool.to_dataset())
    qtbot.addWidget(loaded)
    assert isinstance(loaded, FigureComposerTool)
    operation = loaded.tool_status.operations[0]
    assert operation.method_transform == "custom"
    assert operation.method_transform_expression == "ax.transAxes"
    assert operation.trusted is False

    figurecomposer_rendering._render_into_figure(
        loaded, loaded.figure, sync_visible=False
    )
    assert "not trusted" in loaded._operation_render_errors[operation.operation_id]


def test_figure_composer_limit_methods_default_to_current_axis_limits(qtbot) -> None:
    tool = FigureComposerTool(_figure_composer_profile_source("data"))
    qtbot.addWidget(tool)
    layout_axes = figurecomposer_rendering._live_layout_axes(
        tool, render_if_missing=True
    )
    assert layout_axes is not None

    axis = figurecomposer_rendering._iter_axes(layout_axes)[0]
    expected_xlim = tuple(float(value) for value in axis.get_xlim())
    expected_ylim = tuple(float(value) for value in axis.get_ylim())
    assert tool._figure_window is None

    tool._add_operation("method:axes")
    figurecomposer_method._update_current_method_name(tool, "set_xlim")
    tool._select_step_section("method")
    qtbot.wait_until(
        lambda: (
            tool.step_editor_stack.currentWidget().findChild(
                QtWidgets.QLineEdit, "figureComposerAxesMethodLimitsEdit"
            )
            is not None
        ),
        timeout=5000,
    )
    limits_edit = tool.step_editor_stack.currentWidget().findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodLimitsEdit"
    )
    assert limits_edit is not None
    assert tool.tool_status.operations[-1].method_args == pytest.approx(expected_xlim)
    assert limits_edit.text() == f"{expected_xlim[0]:g}, {expected_xlim[1]:g}"

    tool._add_operation("method:axes")
    figurecomposer_method._update_current_method_name(tool, "set_ylim")
    tool._select_step_section("method")
    qtbot.wait_until(
        lambda: (
            tool.step_editor_stack.currentWidget().findChild(
                QtWidgets.QLineEdit, "figureComposerAxesMethodLimitsEdit"
            )
            is not None
        ),
        timeout=5000,
    )
    limits_edit = tool.step_editor_stack.currentWidget().findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodLimitsEdit"
    )
    assert limits_edit is not None
    assert tool.tool_status.operations[-1].method_args == pytest.approx(expected_ylim)
    assert limits_edit.text() == f"{expected_ylim[0]:g}, {expected_ylim[1]:g}"


def test_figure_composer_method_selector_preserves_compatible_values(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=1),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="set_xlabel",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ).model_copy(
                    update={
                        "method_args": ("Momentum",),
                        "method_kwargs": {
                            "loc": "right",
                            "labelpad": 3.0,
                            "fontsize": 12,
                        },
                    }
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    tool.operation_list.setCurrentItem(tool.operation_list.topLevelItem(0))
    tool._select_step_section("method")

    method_page = tool.step_editor_stack.currentWidget()
    method_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodCombo"
    )
    assert method_combo is not None
    initial_operation = tool.tool_status.operations[0]
    method_combo.activated.emit(method_combo.currentIndex())
    assert tool.tool_status.operations[0] == initial_operation

    ylabel_index = method_combo.findData("set_ylabel")
    assert ylabel_index >= 0
    _activate_combo_index(method_combo, ylabel_index)
    operation = tool.tool_status.operations[0]
    assert operation.method_name == "set_ylabel"
    assert operation.method_args == ("Momentum",)
    assert operation.method_kwargs == {
        "loc": "top",
        "labelpad": 3.0,
        "fontsize": 12,
    }
    assert operation.axes == FigureAxesSelectionState(axes=((0, 0),))

    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    assert not tool._operation_render_errors
    assert tool.figure.axes[0].get_ylabel() == "Momentum"

    namespace: dict[str, typing.Any] = {}
    exec(tool.generated_code(), namespace)  # noqa: S102
    assert namespace["axs"][0, 0].get_ylabel() == "Momentum"

    figurecomposer_method._update_current_method_name(tool, "set_xlim")
    operation = tool.tool_status.operations[0]
    assert operation.method_name == "set_xlim"
    assert operation.method_args != ("Momentum",)
    assert operation.method_kwargs == {}


def test_figure_composer_method_transfer_edge_contracts(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="set_xlim",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    initial_operation = tool.tool_status.operations[0]
    figurecomposer_method._update_current_method_family(tool, FigureMethodFamily.AXES)
    assert tool.tool_status.operations[0] == initial_operation

    tool._updating_controls = True
    try:
        figurecomposer_method._update_current_method_name(tool, "set_ylim")
    finally:
        tool._updating_controls = False
    assert tool.tool_status.operations[0] == initial_operation

    combo_no_none = figurecomposer_method.MethodControlSpec(
        kind=figurecomposer_method.MethodControlKind.ARG_COMBO,
        label="Choice",
        tooltip="choice",
        object_name="choice",
        arg_index=0,
        options=("keep",),
    )
    combo_with_none = figurecomposer_method.MethodControlSpec(
        kind=figurecomposer_method.MethodControlKind.ARG_COMBO,
        label="Choice",
        tooltip="choice",
        object_name="choice",
        arg_index=0,
        options=("keep",),
        none_label="None",
    )
    bool_combo = figurecomposer_method.MethodControlSpec(
        kind=figurecomposer_method.MethodControlKind.BOOL_KWARG_COMBO,
        label="Flag",
        tooltip="flag",
        object_name="flag",
        key="flag",
    )
    assert not figurecomposer_method._control_accepts_value(combo_no_none, None)
    assert figurecomposer_method._control_accepts_value(combo_with_none, None)
    assert figurecomposer_method._control_accepts_value(bool_combo, True)
    assert not figurecomposer_method._control_accepts_value(bool_combo, "True")

    assert (
        figurecomposer_method._transfer_axis_label_loc(
            figurecomposer_method.AXES_METHODS["set_ylabel"],
            figurecomposer_method.AXES_METHODS["set_xlabel"],
            "top",
        )
        == "right"
    )
    assert (
        figurecomposer_method._transfer_axis_label_loc(
            figurecomposer_method.AXES_METHODS["set_title"],
            figurecomposer_method.AXES_METHODS["set_xlabel"],
            "unchanged",
        )
        == "unchanged"
    )

    float_pair_updates = figurecomposer_method._method_transfer_updates(
        tool,
        FigureOperationState.method(
            family=FigureMethodFamily.AXES,
            name="set_xlim",
            args=(1.0, 2.0),
            axes=FigureAxesSelectionState(axes=((0, 0),)),
        ),
        figurecomposer_method.AXES_METHODS["set_ylim"],
    )
    assert float_pair_updates["method_args"] == (1.0, 2.0)

    default_arg_updates = figurecomposer_method._method_transfer_updates(
        tool,
        FigureOperationState.method(
            family=FigureMethodFamily.AXES,
            name="set_xlabel",
            args=("x",),
            axes=FigureAxesSelectionState(axes=((0, 0),)),
        ),
        figurecomposer_method.AXES_METHODS["set_ylabel"],
    )
    assert default_arg_updates["method_args"] == ("y",)

    plot_operation = FigureOperationState.method(
        family=FigureMethodFamily.AXES,
        name="plot",
        args=((0.0, 1.0), (1.0, 2.0)),
        kwargs={"custom": "value"},
        axes=FigureAxesSelectionState(axes=((0, 0),)),
    ).model_copy(
        update={
            "method_transform": "custom",
            "method_transform_x": "figure",
            "method_transform_y": "data",
            "method_transform_expression": "ax.transData",
        }
    )
    plot_updates = figurecomposer_method._method_transfer_updates(
        tool,
        plot_operation,
        figurecomposer_method.AXES_METHODS["plot"],
    )
    assert plot_updates["method_args"] == plot_operation.method_args
    assert plot_updates["method_kwargs"] == {"custom": "value"}
    assert plot_updates["method_transform"] == "custom"
    assert plot_updates["method_transform_x"] == "figure"
    assert plot_updates["method_transform_y"] == "data"
    assert plot_updates["method_transform_expression"] == "ax.transData"

    errorbar_operation = plot_operation.model_copy(
        update={
            "method_name": "errorbar",
            "method_plot_data_mode": "from_data",
            "method_plot_x": FigureMethodPlotValueState(source="x", kind="data"),
            "method_plot_y": FigureMethodPlotValueState(source="y", kind="data"),
            "method_plot_xerr": FigureMethodPlotValueState(
                source="stderr", kind="data"
            ),
            "method_kwargs": {"xerr": (0.1, 0.2), "custom": "value"},
        }
    )
    errorbar_to_plot_updates = figurecomposer_method._method_transfer_updates(
        tool,
        errorbar_operation,
        figurecomposer_method.AXES_METHODS["plot"],
    )
    assert errorbar_to_plot_updates["method_plot_data_mode"] == "from_data"
    assert errorbar_to_plot_updates["method_plot_x"] == FigureMethodPlotValueState(
        source="x", kind="data"
    )
    assert errorbar_to_plot_updates["method_plot_y"] == FigureMethodPlotValueState(
        source="y", kind="data"
    )
    assert errorbar_to_plot_updates["method_plot_xerr"] is None
    assert "xerr" not in errorbar_to_plot_updates["method_kwargs"]

    errorbar_to_errorbar_updates = figurecomposer_method._method_transfer_updates(
        tool,
        errorbar_operation,
        figurecomposer_method.AXES_METHODS["errorbar"],
    )
    assert errorbar_to_errorbar_updates["method_plot_xerr"] == (
        FigureMethodPlotValueState(source="stderr", kind="data")
    )
    assert errorbar_to_errorbar_updates["method_kwargs"]["xerr"] == (0.1, 0.2)

    text_arg = figurecomposer_method.MethodControlSpec(
        kind=figurecomposer_method.MethodControlKind.TEXT_ARG,
        label="Text",
        tooltip="text",
        object_name="text",
        arg_index=0,
    )
    accepted_combo = figurecomposer_method.MethodControlSpec(
        kind=figurecomposer_method.MethodControlKind.ARG_COMBO,
        label="Accepted",
        tooltip="accepted",
        object_name="accepted",
        arg_index=2,
        options=("keep",),
    )
    source_rejected_combo = figurecomposer_method.MethodControlSpec(
        kind=figurecomposer_method.MethodControlKind.ARG_COMBO,
        label="Rejected",
        tooltip="rejected",
        object_name="rejected",
        arg_index=3,
        options=("bad",),
    )
    target_rejected_combo = figurecomposer_method.MethodControlSpec(
        kind=figurecomposer_method.MethodControlKind.ARG_COMBO,
        label="Rejected",
        tooltip="rejected",
        object_name="rejected",
        arg_index=3,
        options=("ok",),
    )
    same_kwarg = figurecomposer_method.MethodControlSpec(
        kind=figurecomposer_method.MethodControlKind.FLOAT_KWARG,
        label="Same",
        tooltip="same",
        object_name="same",
        key="same",
    )
    source_mismatch_kwarg = figurecomposer_method.MethodControlSpec(
        kind=figurecomposer_method.MethodControlKind.TEXT_KWARG,
        label="Mismatch",
        tooltip="mismatch",
        object_name="mismatch",
        key="mismatch",
    )
    target_mismatch_kwarg = figurecomposer_method.MethodControlSpec(
        kind=figurecomposer_method.MethodControlKind.FLOAT_KWARG,
        label="Mismatch",
        tooltip="mismatch",
        object_name="mismatch",
        key="mismatch",
    )
    transform_control = figurecomposer_method.MethodControlSpec(
        kind=figurecomposer_method.MethodControlKind.TRANSFORM,
        label="Transform",
        tooltip="transform",
        object_name="transform",
    )
    source_spec = figurecomposer_method.MethodSpec(
        family=FigureMethodFamily.AXES,
        name="transfer_source",
        label="transfer_source",
        tooltip="test",
        target_domain=figurecomposer_method.MethodTargetDomain.AXES,
        call_policy=figurecomposer_method.MethodCallPolicy.BOUND_EACH_AXIS,
        allowed_call_policies=(
            figurecomposer_method.MethodCallPolicy.BOUND_EACH_AXIS,
            figurecomposer_method.MethodCallPolicy.AX_KEYWORD,
        ),
        default_args=("default",),
        controls=(
            text_arg,
            accepted_combo,
            source_rejected_combo,
            same_kwarg,
            source_mismatch_kwarg,
            transform_control,
        ),
        text_values_policy=figurecomposer_method.MethodTextValuesPolicy.POSITIONAL,
    )
    target_spec = figurecomposer_method.MethodSpec(
        family=FigureMethodFamily.AXES,
        name="transfer_target",
        label="transfer_target",
        tooltip="test",
        target_domain=figurecomposer_method.MethodTargetDomain.AXES,
        call_policy=figurecomposer_method.MethodCallPolicy.BOUND_EACH_AXIS,
        allowed_call_policies=(
            figurecomposer_method.MethodCallPolicy.BOUND_EACH_AXIS,
            figurecomposer_method.MethodCallPolicy.AX_KEYWORD,
        ),
        controls=(
            text_arg,
            accepted_combo,
            target_rejected_combo,
            same_kwarg,
            target_mismatch_kwarg,
            transform_control,
        ),
        text_values_policy=figurecomposer_method.MethodTextValuesPolicy.POSITIONAL,
    )
    monkeypatch.setitem(
        figurecomposer_method.AXES_METHODS, "transfer_source", source_spec
    )
    monkeypatch.setitem(
        figurecomposer_method.AXES_METHODS, "transfer_target", target_spec
    )
    transfer_operation = FigureOperationState.method(
        family=FigureMethodFamily.AXES,
        name="transfer_source",
        args=("default", "ignored", "keep", "bad"),
        kwargs={"same": 2.0, "mismatch": "skip"},
        axes=FigureAxesSelectionState(axes=((0, 0),)),
    ).model_copy(
        update={
            "method_call_policy": (
                figurecomposer_method.MethodCallPolicy.AX_KEYWORD.value
            ),
            "text_values": ("A", "B"),
            "method_transform": "custom",
            "method_transform_x": "figure",
            "method_transform_y": "data",
            "method_transform_expression": "ax.transData",
        }
    )

    transfer_updates = figurecomposer_method._method_transfer_updates(
        tool,
        transfer_operation,
        target_spec,
    )
    assert transfer_updates["method_args"] == (None, None, "keep")
    assert transfer_updates["method_kwargs"] == {"same": 2.0}
    assert (
        transfer_updates["method_call_policy"]
        == figurecomposer_method.MethodCallPolicy.AX_KEYWORD.value
    )
    assert transfer_updates["text_values"] == ("A", "B")
    assert transfer_updates["method_transform"] == "custom"
    assert transfer_updates["method_transform_x"] == "figure"
    assert transfer_updates["method_transform_y"] == "data"
    assert transfer_updates["method_transform_expression"] == "ax.transData"


def test_figure_composer_batch_same_method_edits_selected_steps(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="set_title",
                    args=("left",),
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="set_title",
                    args=("right",),
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="set_xlabel",
                    args=("unchanged",),
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    _select_operation_rows(tool, (0, 1))
    tool._select_step_section("method")
    method_page = tool.step_editor_stack.currentWidget()
    title_edit = method_page.findChild(
        QtWidgets.QPlainTextEdit, "figureComposerAxesMethodTitleEdit"
    )
    assert title_edit is not None
    assert title_edit.toPlainText() == ""
    assert title_edit.placeholderText() == "(multiple values)"

    title_edit.textChanged.emit()
    assert tool.tool_status.operations[0].method_args == ("left",)
    assert tool.tool_status.operations[1].method_args == ("right",)

    title_edit.setPlainText("shared")
    assert tool.tool_status.operations[0].method_args == ("shared",)
    assert tool.tool_status.operations[1].method_args == ("shared",)
    assert tool.tool_status.operations[2].method_args == ("unchanged",)
    assert _selected_operation_rows(tool) == (0, 1)


def test_figure_composer_method_text_args_accept_real_newlines(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="set_xlabel",
                    args=(r"h\nu",),
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    tool.operation_list.setCurrentItem(tool.operation_list.topLevelItem(0))
    tool._select_step_section("method")
    label_edit = tool.step_editor_stack.currentWidget().findChild(
        QtWidgets.QPlainTextEdit, "figureComposerAxesMethodXLabelEdit"
    )
    assert label_edit is not None
    assert label_edit.toPlainText() == r"h\nu"

    label_edit.setPlainText("Energy\n(eV)")
    assert tool.tool_status.operations[0].method_args == ("Energy\n(eV)",)
    restored_status = FigureRecipeState.model_validate_json(
        tool.tool_status.model_dump_json()
    )
    assert restored_status.operations[0].method_args == ("Energy\n(eV)",)

    namespace: dict[str, typing.Any] = {"data": data}
    exec(tool.generated_code(), namespace)  # noqa: S102
    assert namespace["fig"].axes[0].get_xlabel() == "Energy\n(eV)"

    label_edit.setPlainText(r"h\nu")
    assert tool.tool_status.operations[0].method_args == (r"h\nu",)


def test_figure_composer_batch_same_plot_method_edits_selected_steps(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="plot",
                ).model_copy(
                    update={
                        "method_kwargs": {"color": "red", "linewidth": 1.0},
                        "method_transform": "axes",
                    }
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="plot",
                ).model_copy(
                    update={
                        "method_kwargs": {"color": "blue", "linewidth": 3.0},
                        "method_transform": "figure",
                    }
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="set_title",
                    args=("unchanged",),
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    _select_operation_rows(tool, (0, 1))
    tool._select_step_section("method")
    method_page = tool.step_editor_stack.currentWidget()
    color_edit = method_page.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodPlotColorEdit"
    )
    transform_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerMethodTransformModeCombo"
    )
    width_spin = method_page.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerAxesMethodPlotLineWidthSpin"
    )
    assert color_edit is not None
    assert transform_combo is not None
    assert width_spin is not None
    assert color_edit.text() == ""
    assert color_edit.placeholderText() == "(multiple values)"
    assert transform_combo.currentText() == "(multiple values)"
    assert width_spin.value() == pytest.approx(float(mpl.rcParams["lines.linewidth"]))
    width_spin_container = width_spin.parentWidget()
    assert width_spin_container is not None
    assert width_spin_container.findChild(
        QtWidgets.QLabel, "figureComposerMixedValueMarker"
    )

    color_edit.editingFinished.emit()
    assert tool.tool_status.operations[0].method_kwargs["color"] == "red"
    assert tool.tool_status.operations[1].method_kwargs["color"] == "blue"
    assert tool.tool_status.operations[0].method_kwargs["linewidth"] == pytest.approx(
        1.0
    )
    assert tool.tool_status.operations[1].method_kwargs["linewidth"] == pytest.approx(
        3.0
    )

    color_edit.setText("tab:green")
    color_edit.setModified(True)
    color_edit.editingFinished.emit()
    assert tool.tool_status.operations[0].method_kwargs["color"] == "tab:green"
    assert tool.tool_status.operations[1].method_kwargs["color"] == "tab:green"
    width_spin.setValue(4.5)
    assert tool.tool_status.operations[0].method_kwargs["linewidth"] == pytest.approx(
        4.5
    )
    assert tool.tool_status.operations[1].method_kwargs["linewidth"] == pytest.approx(
        4.5
    )

    _activate_combo_text(transform_combo, "blend")
    assert tool.tool_status.operations[0].method_transform == "blend"
    assert tool.tool_status.operations[1].method_transform == "blend"
    assert tool.tool_status.operations[2].method_args == ("unchanged",)
    assert _selected_operation_rows(tool) == (0, 1)


def test_figure_composer_batch_incompatible_methods_disable_editor(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="set_title",
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="set_ylabel",
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    _select_operation_rows(tool, (0, 1))

    assert tool.step_editor_stack.currentWidget().objectName() == (
        "figureComposerIncompatibleBatchPage"
    )
    assert (
        tool.step_editor_stack.currentWidget().findChild(
            QtWidgets.QPlainTextEdit, "figureComposerAxesMethodTitleEdit"
        )
        is None
    )


def test_figure_composer_figure_method_has_no_axes_target(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.FIGURE,
                    name="supxlabel",
                    args=("Momentum",),
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    tool.operation_list.setCurrentItem(tool.operation_list.topLevelItem(0))
    assert "axes" not in tool.step_section_buttons
    assert tool.operation_list.topLevelItem(0).text(0) == "fig.supxlabel"
    assert tool.step_section_buttons["method"].text() == "fig.supxlabel"

    fig = tool.figure
    figurecomposer_rendering._render_into_figure(tool, fig, sync_visible=False)
    assert fig._supxlabel is not None
    assert fig._supxlabel.get_text() == "Momentum"

    code = tool.generated_code()
    assert 'fig.supxlabel("Momentum")' in code
    namespace: dict[str, typing.Any] = {}
    exec(code, namespace)  # noqa: S102
    assert namespace["fig"]._supxlabel.get_text() == "Momentum"


def test_figure_composer_figure_layout_methods_render_and_codegen(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    adjust_tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(ncols=2, layout="none"),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.FIGURE,
                    name="subplots_adjust",
                    kwargs={
                        "left": 0.2,
                        "bottom": 0.15,
                        "right": 0.8,
                        "top": 0.85,
                        "wspace": 0.4,
                        "hspace": 0.3,
                    },
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(adjust_tool)

    adjust_tool.operation_list.setCurrentItem(
        adjust_tool.operation_list.topLevelItem(0)
    )
    adjust_tool._select_step_section("method")
    adjust_page = adjust_tool.step_editor_stack.currentWidget()
    left_spin = adjust_page.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerFigureSubplotsAdjustLeftEdit"
    )
    top_spin = adjust_page.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerFigureSubplotsAdjustTopEdit"
    )
    assert left_spin is not None
    assert top_spin is not None
    assert left_spin.value() == pytest.approx(0.2)
    assert left_spin.minimum() == pytest.approx(0.0)
    assert left_spin.maximum() == pytest.approx(0.8 - 10.0 ** -left_spin.decimals())
    assert left_spin.decimals() == 3
    assert left_spin.singleStep() == pytest.approx(0.005)
    assert not left_spin.keyboardTracking()
    top_spin.setValue(0.9)
    assert adjust_tool.tool_status.operations[0].method_kwargs["top"] == 0.9

    default_tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(ncols=2, layout="none"),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.FIGURE,
                    name="subplots_adjust",
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(default_tool)
    default_tool.operation_list.setCurrentItem(
        default_tool.operation_list.topLevelItem(0)
    )
    default_tool._select_step_section("method")
    default_page = default_tool.step_editor_stack.currentWidget()
    default_left_spin = default_page.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerFigureSubplotsAdjustLeftEdit"
    )
    assert default_left_spin is not None
    default_figure = Figure(
        figsize=default_tool.tool_status.setup.figsize,
        dpi=default_tool.tool_status.setup.dpi,
        layout=default_tool.tool_status.setup.layout,
    )
    assert default_left_spin.value() == pytest.approx(default_figure.subplotpars.left)

    fig = adjust_tool.figure
    figurecomposer_rendering._render_into_figure(adjust_tool, fig, sync_visible=False)
    assert fig.subplotpars.left == pytest.approx(0.2)
    assert fig.subplotpars.bottom == pytest.approx(0.15)
    assert fig.subplotpars.right == pytest.approx(0.8)
    assert fig.subplotpars.top == pytest.approx(0.9)
    assert fig.subplotpars.wspace == pytest.approx(0.4)
    assert fig.subplotpars.hspace == pytest.approx(0.3)

    code = adjust_tool.generated_code()
    assert (
        "fig.subplots_adjust(left=0.2, bottom=0.15, right=0.8, "
        "top=0.9, wspace=0.4, hspace=0.3)"
    ) in code
    namespace = {"data": data}
    exec(code, namespace)  # noqa: S102
    assert namespace["fig"].subplotpars.top == pytest.approx(0.9)

    engine_tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.FIGURE,
                    name="set_layout_engine",
                    args=("tight",),
                    kwargs={"pad": 0.5, "hspace": 0.2},
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(engine_tool)

    engine_tool.operation_list.setCurrentItem(
        engine_tool.operation_list.topLevelItem(0)
    )
    engine_tool._select_step_section("method")
    engine_page = engine_tool.step_editor_stack.currentWidget()
    engine_combo = engine_page.findChild(
        QtWidgets.QComboBox, "figureComposerFigureLayoutEngineCombo"
    )
    pad_edit = engine_page.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerFigureLayoutEnginePadEdit"
    )
    hspace_edit = engine_page.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerFigureLayoutEngineHspaceEdit"
    )
    assert engine_combo is not None
    assert pad_edit is not None
    assert hspace_edit is None
    assert engine_combo.currentText() == "tight"
    assert pad_edit.value() == pytest.approx(0.5)
    assert "hspace" not in engine_tool.generated_code()

    _activate_combo_text(engine_combo, "compressed")
    qtbot.waitUntil(
        lambda: (
            engine_tool.step_editor_stack.currentWidget().findChild(
                QtWidgets.QDoubleSpinBox, "figureComposerFigureLayoutEngineHspaceEdit"
            )
            is not None
        ),
        timeout=1000,
    )
    operation = engine_tool.tool_status.operations[0]
    assert operation.method_args == ("compressed",)
    assert operation.method_kwargs == {"hspace": 0.2}

    engine_page = engine_tool.step_editor_stack.currentWidget()
    assert (
        engine_page.findChild(
            QtWidgets.QDoubleSpinBox, "figureComposerFigureLayoutEnginePadEdit"
        )
        is None
    )
    hspace_edit = engine_page.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerFigureLayoutEngineHspaceEdit"
    )
    rect_edit = engine_page.findChild(
        QtWidgets.QLineEdit, "figureComposerFigureLayoutEngineRectEdit"
    )
    assert hspace_edit is not None
    assert rect_edit is not None
    assert hspace_edit.value() == pytest.approx(0.2)
    rect_edit.setText("0, 0, 0.9, 1")
    rect_edit.editingFinished.emit()
    assert engine_tool.tool_status.operations[0].method_kwargs == {
        "hspace": 0.2,
        "rect": (0, 0, 0.9, 1),
    }

    fig = engine_tool.figure
    figurecomposer_rendering._render_into_figure(engine_tool, fig, sync_visible=False)
    assert fig.get_layout_engine().__class__.__name__ == "ConstrainedLayoutEngine"

    code = engine_tool.generated_code()
    assert (
        'fig.set_layout_engine("compressed", hspace=0.2, rect=(0, 0, 0.9, 1))'
    ) in code
    namespace = {"data": data}
    exec(code, namespace)  # noqa: S102
    assert namespace["fig"].get_layout_engine().__class__.__name__ == (
        "ConstrainedLayoutEngine"
    )


def test_figure_composer_layout_engine_none_is_post_creation_method(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(ncols=2, layout="none"),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.FIGURE,
                    name="set_layout_engine",
                    args=("none",),
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.FIGURE,
                    name="subplots_adjust",
                    kwargs={"left": 0.25},
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    fig = tool.figure
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        figurecomposer_rendering._render_into_figure(tool, fig, sync_visible=False)
    assert not any(
        "incompatible with subplots_adjust" in str(item.message) for item in caught
    )
    assert fig.get_layout_engine() is None
    assert fig.subplotpars.left == pytest.approx(0.25)

    code = tool.generated_code()
    assert 'layout="none"' in code
    assert 'fig.set_layout_engine("none")' in code
    namespace = {"data": data}
    with warnings.catch_warnings(record=True) as generated_caught:
        warnings.simplefilter("always")
        exec(code, namespace)  # noqa: S102
    assert not any(
        "incompatible with subplots_adjust" in str(item.message)
        for item in generated_caught
    )
    assert namespace["fig"].get_layout_engine() is None
    assert namespace["fig"].subplotpars.left == pytest.approx(0.25)


def test_figure_composer_legend_methods_render_and_codegen(qtbot) -> None:
    profile = xr.DataArray(
        np.array([1.0, 2.0, 3.0]),
        dims=("kx",),
        coords={"kx": [-1.0, 0.0, 1.0]},
        name="profile",
    )
    line_operation = FigureOperationState.line(
        label="profile",
        source="profile",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
    ).model_copy(update={"line_x": "kx", "line_labels": ("profile",)})
    axes_legend_operation = FigureOperationState.method(
        family=FigureMethodFamily.AXES,
        name="legend",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
    ).model_copy(
        update={
            "method_kwargs": {
                "loc": "upper right",
                "ncols": 1,
                "title": "Axis legend",
                "frameon": False,
                "fontsize": "small",
                "title_fontsize": "medium",
                "markerscale": 1.5,
                "labelspacing": 0.2,
                "handlelength": 1.0,
                "handletextpad": 0.3,
                "columnspacing": 0.5,
                "bbox_to_anchor": (1.0, 1.0),
            }
        }
    )
    figure_legend_operation = FigureOperationState.method(
        family=FigureMethodFamily.FIGURE,
        name="legend",
    ).model_copy(
        update={
            "method_kwargs": {
                "loc": "lower center",
                "ncols": 1,
                "title": "Figure legend",
                "frameon": True,
                "bbox_to_anchor": (0.5, 0.0),
            }
        }
    )
    tool = FigureComposerTool(
        profile,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="profile", label="profile"),),
            operations=(
                line_operation,
                axes_legend_operation,
                figure_legend_operation,
            ),
            primary_source="profile",
        ),
    )
    qtbot.addWidget(tool)

    _select_operation_rows(tool, (1,))
    tool._select_step_section("method")
    loc_combo = tool.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodLegendLocCombo"
    )
    columns_edit = tool.findChild(
        QtWidgets.QSpinBox, "figureComposerAxesMethodLegendColumnsEdit"
    )
    title_edit = tool.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodLegendTitleEdit"
    )
    assert loc_combo is not None
    assert columns_edit is not None
    assert title_edit is not None
    assert loc_combo.currentText() == "upper right"
    assert columns_edit.value() == 1
    assert title_edit.text() == "Axis legend"

    tool.operation_list.setCurrentItem(tool.operation_list.topLevelItem(2))
    tool._select_step_section("method")
    assert "axes" not in tool.step_section_buttons
    figure_loc_combo = tool.findChild(
        QtWidgets.QComboBox, "figureComposerFigureMethodLegendLocCombo"
    )
    assert figure_loc_combo is not None
    assert figure_loc_combo.currentText() == "lower center"

    fig = tool.figure
    figurecomposer_rendering._render_into_figure(tool, fig, sync_visible=False)
    axis_legend = fig.axes[0].get_legend()
    assert axis_legend is not None
    assert axis_legend.get_title().get_text() == "Axis legend"
    assert axis_legend.get_frame_on() is False
    assert [text.get_text() for text in axis_legend.get_texts()] == ["profile"]
    figure_legend = fig.legends[0]
    assert figure_legend.get_title().get_text() == "Figure legend"
    assert figure_legend.get_frame_on() is True

    code = tool.generated_code()
    assert 'axs[0, 0].legend(loc="upper right", ncols=1' in code
    assert 'title="Axis legend"' in code
    assert "bbox_to_anchor=(1.0, 1.0)" in code
    assert 'fig.legend(loc="lower center", ncols=1' in code
    assert 'title="Figure legend"' in code

    namespace: dict[str, typing.Any] = {"profile": profile}
    exec(code, namespace)  # noqa: S102
    assert namespace["fig"].axes[0].get_legend() is not None
    assert namespace["fig"].legends[0].get_title().get_text() == "Figure legend"


def test_figure_composer_label_generated_code_escapes_backslashes(qtbot) -> None:
    label = "$\\Delta y$"
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
                FigureOperationState.method(
                    family=FigureMethodFamily.FIGURE,
                    name="supylabel",
                ).model_copy(update={"method_args": (label,)}),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    code = tool.generated_code()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", SyntaxWarning)
        compiled = compile(code, "<figure-composer-generated>", "exec")

    assert not any(warning.category is SyntaxWarning for warning in caught)
    namespace: dict[str, typing.Any] = {}
    exec(compiled, namespace)  # noqa: S102
    assert namespace["fig"].get_supylabel() == label


def test_figure_composer_colorbar_method_target_policy(qtbot) -> None:
    data = xr.DataArray(
        np.arange(8.0).reshape(2, 2, 2),
        dims=("eV", "kx", "ky"),
        coords={"eV": [0.0, 1.0], "kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.plot_slices(
                    label="maps",
                    sources=("data",),
                    axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
                    slice_dim="eV",
                    slice_values=(0.0, 1.0),
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.ERLAB,
                    name="nice_colorbar",
                    axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    tool.operation_list.setCurrentItem(tool.operation_list.topLevelItem(1))
    tool._select_step_section("method")
    policy_combo = tool.findChild(
        QtWidgets.QComboBox, "figureComposerMethodCallPolicyCombo"
    )
    assert policy_combo is not None
    assert policy_combo.currentText() == "Each selected axis"
    assert "for ax in axs.flat:" in tool.generated_code()

    _activate_combo_text(policy_combo, "Selected axes together")
    assert tool.tool_status.operations[1].method_call_policy == "ax_keyword"
    assert "eplt.nice_colorbar(ax=axs)" in tool.generated_code()


def test_figure_composer_erlab_method_controls_update_recipe(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    axes = FigureAxesSelectionState(axes=((0, 0), (0, 1)))
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.ERLAB,
                    name="clean_labels",
                    axes=axes,
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.ERLAB,
                    name="label_subplots",
                    axes=axes,
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.ERLAB,
                    name="label_subplot_properties",
                    axes=axes,
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.ERLAB,
                    name="nice_colorbar",
                    axes=axes,
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.ERLAB,
                    name="proportional_colorbar",
                    axes=axes,
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.ERLAB,
                    name="set_titles",
                    axes=axes,
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.ERLAB,
                    name="fermiline",
                    axes=axes,
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.ERLAB,
                    name="mark_points",
                    axes=axes,
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.ERLAB,
                    name="scale_units",
                    axes=axes,
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.ERLAB,
                    name="fancy_labels",
                    axes=axes,
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.ERLAB,
                    name="integer_ticks",
                    axes=axes,
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.ERLAB,
                    name="sizebar",
                    axes=axes,
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.ERLAB,
                    name="unify_clim",
                    axes=axes,
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    def select_method(row: int) -> QtWidgets.QWidget:
        tool.operation_list.setCurrentItem(tool.operation_list.topLevelItem(row))
        tool._select_step_section("method")
        page = tool.step_editor_stack.currentWidget()
        assert page is not None
        return page

    def line_edit(page: QtWidgets.QWidget, name: str) -> QtWidgets.QLineEdit:
        widget = page.findChild(QtWidgets.QLineEdit, name)
        assert widget is not None
        return widget

    def combo_box(page: QtWidgets.QWidget, name: str) -> QtWidgets.QComboBox:
        widget = page.findChild(QtWidgets.QComboBox, name)
        assert widget is not None
        return widget

    def spin_box(page: QtWidgets.QWidget, name: str) -> QtWidgets.QSpinBox:
        widget = page.findChild(QtWidgets.QSpinBox, name)
        assert widget is not None
        return widget

    def double_spin_box(page: QtWidgets.QWidget, name: str) -> QtWidgets.QDoubleSpinBox:
        widget = page.findChild(QtWidgets.QDoubleSpinBox, name)
        assert widget is not None
        return widget

    def set_line_edit(page: QtWidgets.QWidget, name: str, text: str) -> None:
        edit = line_edit(page, name)
        edit.setText(text)
        edit.editingFinished.emit()

    def set_combo(page: QtWidgets.QWidget, name: str, text: str) -> None:
        _activate_combo_text(combo_box(page, name), text)

    def operation(row: int) -> FigureOperationState:
        return tool.tool_status.operations[row]

    page = select_method(0)
    set_combo(page, "figureComposerERLabCleanLabelsRemoveInnerTicksCombo", "True")
    assert operation(0).method_args == (True,)

    page = select_method(1)
    spin_box(page, "figureComposerERLabLabelSubplotsStartEdit").setValue(3)
    set_combo(page, "figureComposerERLabLabelSubplotsOrderCombo", "F")
    set_combo(page, "figureComposerERLabLabelSubplotsLocCombo", "lower right")
    set_line_edit(page, "figureComposerERLabLabelSubplotsOffsetEdit", "1, 2")
    set_line_edit(page, "figureComposerERLabLabelSubplotsPrefixEdit", "(")
    set_line_edit(page, "figureComposerERLabLabelSubplotsSuffixEdit", ")")
    set_combo(page, "figureComposerERLabLabelSubplotsNumericCombo", "True")
    set_combo(page, "figureComposerERLabLabelSubplotsCapitalCombo", "True")
    set_combo(page, "figureComposerERLabLabelSubplotsFontWeightCombo", "bold")
    set_line_edit(page, "figureComposerERLabLabelSubplotsFontSizeEdit", "8")
    assert operation(1).method_kwargs == {
        "startfrom": 3,
        "order": "F",
        "loc": "lower right",
        "offset": (1.0, 2.0),
        "prefix": "(",
        "suffix": ")",
        "numeric": True,
        "capital": True,
        "fontweight": "bold",
        "fontsize": 8,
    }

    page = select_method(2)
    set_line_edit(page, "figureComposerERLabLabelPropertiesValuesEdit", "eV=[0, 1]")
    set_line_edit(page, "figureComposerERLabLabelPropertiesDecimalsEdit", "2")
    spin_box(page, "figureComposerERLabLabelPropertiesSiEdit").setValue(-3)
    set_line_edit(page, "figureComposerERLabLabelPropertiesNameEdit", "Energy")
    set_line_edit(page, "figureComposerERLabLabelPropertiesUnitEdit", "eV")
    set_combo(page, "figureComposerERLabLabelPropertiesOrderCombo", "F")
    set_combo(page, "figureComposerERLabLabelPropertiesLocCombo", "lower left")
    set_line_edit(page, "figureComposerERLabLabelPropertiesOffsetEdit", "3, 4")
    set_line_edit(page, "figureComposerERLabLabelPropertiesPrefixEdit", "[")
    set_line_edit(page, "figureComposerERLabLabelPropertiesSuffixEdit", "]")
    set_combo(page, "figureComposerERLabLabelPropertiesFontWeightCombo", "bold")
    set_line_edit(page, "figureComposerERLabLabelPropertiesFontSizeEdit", "9")
    assert operation(2).method_args == ({"eV": [0, 1]},)
    assert operation(2).method_kwargs == {
        "decimals": 2,
        "si": -3,
        "name": "Energy",
        "unit": "eV",
        "order": "F",
        "loc": "lower left",
        "offset": (3.0, 4.0),
        "prefix": "[",
        "suffix": "]",
        "fontweight": "bold",
        "fontsize": 9,
    }

    page = select_method(3)
    double_spin_box(page, "figureComposerERLabNiceColorbarWidthEdit").setValue(10.0)
    double_spin_box(page, "figureComposerERLabNiceColorbarAspectEdit").setValue(4.0)
    double_spin_box(page, "figureComposerERLabNiceColorbarPadEdit").setValue(2.0)
    set_combo(page, "figureComposerERLabNiceColorbarMinMaxCombo", "True")
    set_combo(page, "figureComposerERLabNiceColorbarOrientationCombo", "horizontal")
    set_combo(page, "figureComposerERLabNiceColorbarFloatingCombo", "True")
    set_line_edit(page, "figureComposerERLabNiceColorbarTicksEdit", "0, 0.5, 1")
    set_line_edit(
        page, "figureComposerERLabNiceColorbarTickLabelsEdit", "low, mid, high"
    )
    assert operation(3).method_kwargs == {
        "width": 10.0,
        "aspect": 4.0,
        "pad": 2.0,
        "minmax": True,
        "orientation": "horizontal",
        "floating": True,
        "ticks": (0, 0.5, 1),
        "ticklabels": ("low", "mid", "high"),
    }

    page = select_method(4)
    spin_box(page, "figureComposerERLabProportionalColorbarIndexEdit").setValue(0)
    set_combo(page, "figureComposerERLabProportionalColorbarImageOnlyCombo", "True")
    set_line_edit(page, "figureComposerERLabProportionalColorbarTicksEdit", "[0, 1]")
    assert operation(4).method_kwargs == {
        "index": 0,
        "image_only": True,
        "ticks": [0, 1],
    }

    page = select_method(5)
    set_combo(page, "figureComposerERLabSetTitlesOrderCombo", "F")
    assert operation(5).method_kwargs == {"order": "F"}

    page = select_method(6)
    double_spin_box(page, "figureComposerERLabFermilineValueEdit").setValue(0.1)
    set_combo(page, "figureComposerERLabFermilineOrientationCombo", "v")
    assert operation(6).method_kwargs == {"value": 0.1, "orientation": "v"}
    set_line_edit(page, "figureComposerERLabFermilineColorEdit", "tab:red")
    set_combo(page, "figureComposerERLabFermilineLineStyleCombo", "--")
    set_line_edit(page, "figureComposerERLabFermilineLineWidthEdit", "1.5")
    assert operation(6).method_kwargs == {
        "value": 0.1,
        "orientation": "v",
        "color": "tab:red",
        "linestyle": "--",
        "linewidth": 1.5,
    }

    page = select_method(7)
    set_line_edit(page, "figureComposerERLabMarkPointsPointsEdit", "0, 1")
    set_line_edit(page, "figureComposerERLabMarkPointsLabelsEdit", "G, M")
    set_line_edit(page, "figureComposerERLabMarkPointsYEdit", "0.25, 0.5")
    set_line_edit(page, "figureComposerERLabMarkPointsPadEdit", "1, 2")
    set_combo(page, "figureComposerERLabMarkPointsLiteralCombo", "True")
    set_combo(page, "figureComposerERLabMarkPointsRomanCombo", "False")
    set_combo(page, "figureComposerERLabMarkPointsBarCombo", "True")
    assert operation(7).method_args == ((0, 1), ("G", "M"))
    assert operation(7).method_kwargs == {
        "y": (0.25, 0.5),
        "pad": (1.0, 2.0),
        "literal": True,
        "roman": False,
        "bar": True,
    }

    page = select_method(8)
    set_combo(page, "figureComposerERLabScaleUnitsAxisCombo", "y")
    spin_box(page, "figureComposerERLabScaleUnitsSiEdit").setValue(3)
    set_combo(page, "figureComposerERLabScaleUnitsPrefixCombo", "False")
    set_combo(page, "figureComposerERLabScaleUnitsPowerCombo", "True")
    assert operation(8).method_args == ("y", 3)
    assert operation(8).method_kwargs == {"prefix": False, "power": True}

    page = select_method(9)
    set_combo(page, "figureComposerERLabFancyLabelsRadiansCombo", "True")
    assert operation(9).method_kwargs == {"radians": True}

    page = select_method(10)
    assert (
        page.findChild(QtWidgets.QLineEdit, "figureComposerERLabMethodKwEdit") is None
    )

    page = select_method(11)
    assert operation(11).method_kwargs == {}
    assert double_spin_box(page, "figureComposerERLabSizebarValueEdit").value() == 1.0
    assert line_edit(page, "figureComposerERLabSizebarUnitEdit").text() == "m"
    double_spin_box(page, "figureComposerERLabSizebarValueEdit").setValue(2.0)
    set_line_edit(page, "figureComposerERLabSizebarUnitEdit", "m")
    spin_box(page, "figureComposerERLabSizebarSiEdit").setValue(-6)
    double_spin_box(page, "figureComposerERLabSizebarResolutionEdit").setValue(0.001)
    spin_box(page, "figureComposerERLabSizebarDecimalsEdit").setValue(1)
    set_line_edit(page, "figureComposerERLabSizebarLabelEdit", "200 um")
    set_combo(page, "figureComposerERLabSizebarLocCombo", "lower left")
    double_spin_box(page, "figureComposerERLabSizebarPadEdit").setValue(0.2)
    double_spin_box(page, "figureComposerERLabSizebarBorderPadEdit").setValue(0.6)
    double_spin_box(page, "figureComposerERLabSizebarSepEdit").setValue(4.0)
    set_combo(page, "figureComposerERLabSizebarFrameCombo", "True")
    assert operation(11).method_kwargs == {
        "value": 2.0,
        "unit": "m",
        "si": -6,
        "resolution": 0.001,
        "decimals": 1,
        "label": "200 um",
        "loc": "lower left",
        "pad": 0.2,
        "borderpad": 0.6,
        "sep": 4.0,
        "frameon": True,
    }

    page = select_method(12)
    set_combo(page, "figureComposerERLabUnifyClimImageOnlyCombo", "True")
    set_combo(page, "figureComposerERLabUnifyClimAutoscaleCombo", "True")
    set_line_edit(page, "figureComposerERLabUnifyClimVminEdit", "0")
    set_line_edit(page, "figureComposerERLabUnifyClimVmaxEdit", "1")
    assert operation(12).method_kwargs == {
        "image_only": True,
        "autoscale": True,
        "vmin": 0.0,
        "vmax": 1.0,
    }

    code = tool.generated_code()
    assert "ticks=(0, 0.5, 1)" in code
    assert "ticks=[0, 1]" in code
    assert "eplt.fancy_labels(axs, radians=True)" in code
    assert "eplt.integer_ticks(axs)" in code
    assert "eplt.label_subplot_properties(" in code
    assert 'loc="lower left"' in code
    assert "offset=(3.0, 4.0)" in code
    assert 'prefix="["' in code
    assert 'suffix="]"' in code
    assert 'fontweight="bold"' in code
    assert "fontsize=9" in code
    assert "eplt.sizebar(" in code
    assert "value=2.0" in code
    assert 'unit="m"' in code
    assert "eplt.unify_clim(" in code
    assert "image_only=True" in code
    assert "autoscale=True" in code


def test_figure_composer_tick_params_editor_edge_commits(qtbot) -> None:
    editor = figurecomposer_tick_params.TickParamsEditorWidget(
        {"length": 1.0, "grid_alpha": 0.5, "grid_linestyle": "--"}
    )
    qtbot.addWidget(editor)
    emitted: list[dict[str, typing.Any]] = []
    editor.sigTickParamsChanged.connect(emitted.append)

    editor.setToolTip("ignored")
    assert editor.toolTip() == ""

    _finish_tick_params_edit(editor, "figureComposerAxesMethodTickParamsLengthEdit", "")
    assert "length" not in editor.tick_params()
    assert emitted[-1]["grid_alpha"] == 0.5

    emitted.clear()
    _finish_tick_params_edit(
        editor, "figureComposerAxesMethodTickParamsWidthEdit", "not-a-number"
    )
    _finish_tick_params_edit(
        editor, "figureComposerAxesMethodTickParamsWidthEdit", "-1"
    )
    assert emitted == []

    _finish_tick_params_edit(
        editor, "figureComposerAxesMethodTickParamsWidthEdit", "2.5"
    )
    assert emitted[-1]["width"] == pytest.approx(2.5)

    emitted.clear()
    _finish_tick_params_edit(
        editor, "figureComposerAxesMethodTickParamsGridAlphaEdit", "2"
    )
    assert emitted == []
    _finish_tick_params_edit(
        editor, "figureComposerAxesMethodTickParamsGridAlphaEdit", "0.25"
    )
    assert emitted[-1]["grid_alpha"] == pytest.approx(0.25)

    emitted.clear()
    _finish_tick_params_edit(
        editor, "figureComposerAxesMethodTickParamsLabelSizeEdit", "not valid("
    )
    assert emitted == []
    _finish_tick_params_edit(
        editor, "figureComposerAxesMethodTickParamsLabelSizeEdit", "'small'"
    )
    assert emitted[-1]["labelsize"] == "small"
    _finish_tick_params_edit(
        editor, "figureComposerAxesMethodTickParamsLabelSizeEdit", ""
    )
    assert "labelsize" not in emitted[-1]

    _finish_tick_params_edit(
        editor, "figureComposerAxesMethodTickParamsLabelFontEdit", "Arial"
    )
    assert emitted[-1]["labelfontfamily"] == "Arial"
    _finish_tick_params_edit(
        editor, "figureComposerAxesMethodTickParamsLabelFontEdit", ""
    )
    assert "labelfontfamily" not in emitted[-1]

    colors_edit = editor.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodTickParamsColorsEdit"
    )
    assert colors_edit is not None
    colors_edit.setText("")
    colors_edit.editingFinished.emit()
    assert "colors" not in emitted[-1]

    combo = editor.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodTickParamsGridLineStyleCombo"
    )
    assert combo is not None
    figurecomposer_tick_params.TickParamsEditorWidget._set_combo_value(
        combo, "not-present"
    )
    assert combo.currentIndex() == 0

    editor._updating = True
    editor._set_kwarg("axis", "x")
    assert emitted[-1].get("axis") != "x"
    editor._updating = False

    tristate = figurecomposer_tick_params._TriStateCheckBox("test")
    qtbot.addWidget(tristate)
    tristate.set_value("bad")
    assert tristate.value() is None


def test_figure_composer_erlab_method_allows_empty_text_values(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.ERLAB,
                    name="set_titles",
                    axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
                ).model_copy(update={"text_values": ("Left", "Right")}),
                FigureOperationState.method(
                    family=FigureMethodFamily.ERLAB,
                    name="set_xlabels",
                    axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
                ).model_copy(update={"text_values": ("initial",)}),
                FigureOperationState.method(
                    family=FigureMethodFamily.ERLAB,
                    name="set_ylabels",
                    axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    tool.operation_list.setCurrentItem(tool.operation_list.topLevelItem(0))
    tool._select_step_section("method")
    title_edit = tool.step_editor_stack.currentWidget().findChild(
        QtWidgets.QPlainTextEdit
    )
    assert title_edit is not None
    title_edit.setPlainText("Left\n")
    assert tool.tool_status.operations[0].text_values == ("Left", "")

    tool.operation_list.setCurrentItem(tool.operation_list.topLevelItem(1))
    tool._select_step_section("method")
    xlabel_edit = tool.step_editor_stack.currentWidget().findChild(
        QtWidgets.QPlainTextEdit
    )
    assert xlabel_edit is not None
    xlabel_edit.setPlainText("")
    assert tool.tool_status.operations[1].text_values == ("",)

    namespace: dict[str, typing.Any] = {"data": data}
    exec(tool.generated_code(), namespace)  # noqa: S102
    axs = namespace["axs"]
    assert axs[0, 0].get_title() == "Left"
    assert axs[0, 1].get_title() == ""
    assert axs[0, 0].get_xlabel() == ""
    assert axs[0, 1].get_xlabel() == ""
    assert axs[0, 0].get_ylabel() == ""
    assert axs[0, 1].get_ylabel() == ""


def test_figure_composer_method_draw_time_text_error_is_non_modal(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    plot_operation = FigureOperationState.plot_array(label="plot", source="data")
    title_operation = FigureOperationState.method(
        family=FigureMethodFamily.ERLAB,
        name="set_titles",
        axes=FigureAxesSelectionState(axes=((0, 0),)),
    ).model_copy(update={"text_values": ("ALS, $$39.3 eV",)})
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(plot_operation, title_operation),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    tool.operation_list.setCurrentItem(tool.operation_list.topLevelItem(1))

    tool._redraw_plot(show_window=True)

    render_error = tool._operation_render_errors[title_operation.operation_id]
    assert "ValueError" in render_error
    assert "ParseException" in render_error
    item = tool.operation_list.topLevelItem(1)
    assert item is not None
    assert _operation_status_codes(tool, 1) == ("render_error",)
    assert "ParseException" in item.toolTip(
        figurecomposer_tool_module._OPERATION_LIST_STATUS_COLUMN
    )

    tool._replace_operation(
        1,
        title_operation.model_copy(update={"text_values": ("ALS, $39.3$ eV",)}),
    )
    tool._redraw_plot(show_window=True)

    assert title_operation.operation_id not in tool._operation_render_errors
    item = tool.operation_list.topLevelItem(1)
    assert item is not None
    assert _operation_status_codes(tool, 1) == ()
