# ruff: noqa: F403, F405

from erlab.interactive._figurecomposer._operations._method import (
    _catalog as method_catalog,
)
from erlab.interactive._figurecomposer._operations._method import (
    _execution as method_execution,
)

from ._common import *


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
        method_catalog.ERLAB_METHODS,
        "composer_ax_keyword_test",
        method_catalog.MethodSpec(
            family=FigureMethodFamily.ERLAB,
            name="composer_ax_keyword_test",
            label="composer_ax_keyword_test",
            tooltip="test",
            target_domain=method_catalog.MethodTargetDomain.AXES,
            call_policy=method_catalog.MethodCallPolicy.AX_KEYWORD,
        ),
    )
    monkeypatch.setitem(
        method_catalog.ERLAB_METHODS,
        "composer_fig_keyword_test",
        method_catalog.MethodSpec(
            family=FigureMethodFamily.ERLAB,
            name="composer_fig_keyword_test",
            label="composer_fig_keyword_test",
            tooltip="test",
            target_domain=method_catalog.MethodTargetDomain.FIGURE,
            call_policy=method_catalog.MethodCallPolicy.FIG_KEYWORD,
        ),
    )
    monkeypatch.setitem(
        method_catalog.ERLAB_METHODS,
        "composer_each_axis_test",
        method_catalog.MethodSpec(
            family=FigureMethodFamily.ERLAB,
            name="composer_each_axis_test",
            label="composer_each_axis_test",
            tooltip="test",
            target_domain=method_catalog.MethodTargetDomain.AXES,
            call_policy=method_catalog.MethodCallPolicy.EACH_AXIS_AX_KEYWORD,
        ),
    )
    monkeypatch.setitem(
        method_catalog.ERLAB_METHODS,
        "composer_plain_test",
        method_catalog.MethodSpec(
            family=FigureMethodFamily.ERLAB,
            name="composer_plain_test",
            label="composer_plain_test",
            tooltip="test",
            target_domain=method_catalog.MethodTargetDomain.NONE,
            call_policy=method_catalog.MethodCallPolicy.PLAIN_CALL,
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
        for line in method_execution._method_code(tool, operation)
    ]
    calls.clear()
    axs = np.asarray(figure.axes, dtype=object).reshape(1, 2)
    exec("\n".join(lines), {"axs": axs, "eplt": eplt, "fig": figure})  # noqa: S102

    assert [call[0] for call in calls] == [
        "ax_keyword",
        "fig_keyword",
        "each_axis",
        "each_axis",
        "plain",
    ]
    assert calls[0][1] is axs
    assert calls[0][2] == {"alpha": 0.5}
    assert calls[1][1] is figure
    assert calls[1][2] == {"label": "figure"}
    assert calls[2][1] is axs.flat[0]
    assert calls[2][2] == {"color": "red"}
    assert calls[3][1] is axs.flat[1]
    assert calls[3][2] == {"color": "red"}
    assert calls[4][1] == ("value",)


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

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )
    tool.operation_editor.select_section("method")
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
    assert method_combo.currentData() == "text"
    assert text_edit.toPlainText() == "Panel"
    assert kwargs_edit.text() == 'ha="left", va="top"'

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(4)
    )
    tool.operation_editor.select_section("method")
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
    assert tool.tool_status.operations[4].method_args == (True,)
    assert tool.tool_status.operations[4].method_kwargs == {
        "which": "major",
        "axis": "x",
    }

    scale_names = tuple(mscale.get_scale_names())
    assert "log" in scale_names
    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(6)
    )
    tool.operation_editor.select_section("method")
    xscale_combo = tool.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodXScaleCombo"
    )
    assert xscale_combo is not None
    assert (
        tuple(xscale_combo.itemText(index) for index in range(xscale_combo.count()))
        == scale_names
    )
    assert tool.tool_status.operations[6].method_args == ()

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(7)
    )
    tool.operation_editor.select_section("method")
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

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(8)
    )
    tool.operation_editor.select_section("method")
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
    assert title_pad_edit.value() == 2.0

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(11)
    )
    tool.operation_editor.select_section("method")
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

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(12)
    )
    tool.operation_editor.select_section("method")
    aspect_edit = tool.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodAspectEdit"
    )
    aspect_share_combo = tool.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodAspectShareCombo"
    )
    assert aspect_edit is not None
    assert aspect_share_combo is not None
    assert aspect_edit.text() == "2"
    aspect_edit.setText("2.5")
    aspect_edit.editingFinished.emit()
    assert tool.tool_status.operations[12].method_args == (2.5,)

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(13)
    )
    tool.operation_editor.select_section("method")
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
    spec = method_catalog.AXES_METHODS["plot"]
    operation = FigureOperationState.method(
        family=FigureMethodFamily.AXES,
        name="plot",
    ).model_copy(update={"method_transform": mode})

    transform = method_execution._render_method_transform(
        operation,
        spec,
        figure=figure,
        axis=axis,
    )
    code = method_execution._method_transform_code(operation, spec)

    assert isinstance(transform, mtransforms.Transform)
    assert repr(code) == code_fragment
    plt.close(figure)


def test_figure_composer_method_custom_transform_errors() -> None:
    figure, axis = plt.subplots()
    spec = method_catalog.AXES_METHODS["plot"]
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
        method_execution._method_transform_code(untrusted, spec)
    with pytest.raises(ValueError, match="empty"):
        method_execution._method_transform_code(empty, spec)
    with pytest.raises(ValueError, match="empty"):
        method_execution._render_method_transform(
            empty,
            spec,
            figure=figure,
            axis=axis,
        )
    with pytest.raises(TypeError, match="Transform"):
        method_execution._render_method_transform(
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

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )
    assert "axes" not in tool.operation_editor.section_keys
    assert tool.tool_status.operations[0].method_name == "supxlabel"

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

    adjust_tool.operation_panel.operation_list.setCurrentItem(
        adjust_tool.operation_panel.operation_list.topLevelItem(0)
    )
    adjust_tool.operation_editor.select_section("method")
    adjust_page = adjust_tool.operation_editor.stack.currentWidget()
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
    default_tool.operation_panel.operation_list.setCurrentItem(
        default_tool.operation_panel.operation_list.topLevelItem(0)
    )
    default_tool.operation_editor.select_section("method")
    default_page = default_tool.operation_editor.stack.currentWidget()
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

    engine_tool.operation_panel.operation_list.setCurrentItem(
        engine_tool.operation_panel.operation_list.topLevelItem(0)
    )
    engine_tool.operation_editor.select_section("method")
    engine_page = engine_tool.operation_editor.stack.currentWidget()
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
    assert pad_edit.value() == pytest.approx(0.5)
    assert "hspace" not in engine_tool.generated_code()

    _activate_combo_text(engine_combo, "compressed")
    qtbot.waitUntil(
        lambda: (
            engine_tool.operation_editor.stack.currentWidget().findChild(
                QtWidgets.QDoubleSpinBox, "figureComposerFigureLayoutEngineHspaceEdit"
            )
            is not None
        ),
        timeout=1000,
    )
    operation = engine_tool.tool_status.operations[0]
    assert operation.method_args == ("compressed",)
    assert operation.method_kwargs == {"hspace": 0.2}

    engine_page = engine_tool.operation_editor.stack.currentWidget()
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
    tool.operation_editor.select_section("method")
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
    assert columns_edit.value() == 1
    assert title_edit.text() == "Axis legend"

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(2)
    )
    tool.operation_editor.select_section("method")
    assert "axes" not in tool.operation_editor.section_keys
    figure_loc_combo = tool.findChild(
        QtWidgets.QComboBox, "figureComposerFigureMethodLegendLocCombo"
    )
    assert figure_loc_combo is not None

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

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(1)
    )
    tool.operation_editor.select_section("method")
    policy_combo = tool.findChild(
        QtWidgets.QComboBox, "figureComposerMethodCallPolicyCombo"
    )
    assert policy_combo is not None
    assert "for ax in axs.flat:" in tool.generated_code()

    _activate_combo_text(policy_combo, "Selected axes together")
    assert tool.tool_status.operations[1].method_call_policy == "ax_keyword"
    assert "eplt.nice_colorbar(ax=axs)" in tool.generated_code()


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
    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(1)
    )

    tool._redraw_plot(show_window=True)

    render_error = tool._operation_render_errors[title_operation.operation_id]
    assert "ValueError" in render_error
    assert "ParseException" in render_error
    item = tool.operation_panel.operation_list.topLevelItem(1)
    assert item is not None
    assert _operation_status_codes(tool, 1) == ("render_error",)
    assert "ParseException" in item.toolTip(
        figurecomposer_operation_panel._OPERATION_LIST_STATUS_COLUMN
    )

    tool._replace_operation(
        1,
        title_operation.model_copy(update={"text_values": ("ALS, $39.3$ eV",)}),
    )
    tool._redraw_plot(show_window=True)

    assert title_operation.operation_id not in tool._operation_render_errors
    item = tool.operation_panel.operation_list.topLevelItem(1)
    assert item is not None
    assert _operation_status_codes(tool, 1) == ()
