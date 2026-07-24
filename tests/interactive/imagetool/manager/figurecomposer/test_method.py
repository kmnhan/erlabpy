from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

import erlab.interactive._figurecomposer._rendering as figurecomposer_rendering
import erlab.interactive._figurecomposer._tool as figurecomposer_tool_module
import erlab.interactive._figurecomposer._ui._tick_params as figurecomposer_tick_params
import erlab.interactive._stylesheets
import erlab.plotting as eplt
from erlab.interactive._figurecomposer import (
    FigureAxesSelectionState,
    FigureComposerTool,
    FigureGridSpecAxesState,
    FigureGridSpecGridState,
    FigureGridSpecLayoutState,
    FigureGridSpecSpanState,
    FigureMethodFamily,
    FigureMethodPlotValueState,
    FigureOperationState,
    FigureRecipeState,
    FigureSourceState,
    FigureSubplotsState,
)
from erlab.interactive._figurecomposer._operations._method import (
    _catalog as method_catalog,
)
from erlab.interactive._figurecomposer._operations._method import (
    _editor as method_editor,
)
from erlab.interactive._figurecomposer._operations._method import (
    _execution as method_execution,
)
from erlab.interactive._figurecomposer._operations._method import (
    _plot_editor as method_plot_editor,
)
from erlab.interactive._figurecomposer._operations._method import _state as method_state

from ._common import (
    _activate_combo_index,
    _activate_combo_text,
    _click_tick_params_segment,
    _figure_composer_profile_source,
    _finish_tick_params_edit,
    _select_operation_rows,
    _selected_operation_rows,
    _set_tick_params_button,
)


@pytest.mark.parametrize(
    ("family", "methods", "target_domain"),
    [
        (
            FigureMethodFamily.AXES,
            method_catalog.AXES_METHODS,
            method_catalog.MethodTargetDomain.AXES,
        ),
        (
            FigureMethodFamily.FIGURE,
            method_catalog.FIGURE_METHODS,
            method_catalog.MethodTargetDomain.FIGURE,
        ),
        (
            FigureMethodFamily.ERLAB,
            method_catalog.ERLAB_METHODS,
            method_catalog.MethodTargetDomain.AXES,
        ),
    ],
)
def test_figure_composer_method_catalog_matches_document_target_semantics(
    family, methods, target_domain
) -> None:
    assert methods
    for name, spec in methods.items():
        assert spec.family == family
        assert spec.name == name
        assert spec.target_domain == target_domain


def test_figure_composer_method_doc_url_uses_family_templates() -> None:
    assert method_catalog._method_doc_url(method_catalog.AXES_METHODS["text"]) == (
        "https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html"
    )
    assert method_catalog._method_doc_url(
        method_catalog.FIGURE_METHODS["supxlabel"]
    ) == (
        "https://matplotlib.org/stable/api/_as_gen/"
        "matplotlib.figure.Figure.supxlabel.html"
    )
    assert method_catalog._method_doc_url(
        method_catalog.ERLAB_METHODS["clean_labels"]
    ) == (
        "https://erlabpy.readthedocs.io/en/stable/erlab.plotting.html"
        "#erlab.plotting.clean_labels"
    )
    spec = method_catalog.MethodSpec(
        family=FigureMethodFamily.ERLAB,
        name="call_name",
        label="call_name",
        tooltip="test",
        target_domain=method_catalog.MethodTargetDomain.FIGURE,
        call_policy=method_catalog.MethodCallPolicy.PLAIN_CALL,
        doc_name="documented_name",
    )
    assert method_catalog._method_doc_url(spec) == (
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
        tool.operation_panel.operation_list.setCurrentItem(
            tool.operation_panel.operation_list.topLevelItem(row)
        )
        tool.operation_editor.select_section("method")
        button = tool.operation_editor.stack.currentWidget().findChild(
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
        method_catalog._method_spec(
            FigureOperationState.method(family=FigureMethodFamily.AXES, name="missing")
        )
    text_spec = method_catalog.AXES_METHODS["set_xlim"]
    assert method_catalog._method_selector_text(text_spec) == text_spec.name

    colorbar_operation = FigureOperationState.method(
        family=FigureMethodFamily.ERLAB,
        name="nice_colorbar",
    )
    colorbar_spec = method_catalog._method_spec(colorbar_operation)
    with pytest.raises(ValueError, match="Unsupported call policy"):
        method_catalog._effective_call_policy(
            colorbar_operation.model_copy(update={"method_call_policy": "bad-policy"}),
            colorbar_spec,
        )
    with pytest.raises(ValueError, match="not available"):
        method_catalog._effective_call_policy(
            colorbar_operation.model_copy(
                update={
                    "method_call_policy": (
                        method_catalog.MethodCallPolicy.PLAIN_CALL.value
                    )
                }
            ),
            colorbar_spec,
        )
    assert (
        method_catalog._effective_call_policy(
            colorbar_operation.model_copy(
                update={
                    "method_call_policy": (
                        method_catalog.MethodCallPolicy.AX_KEYWORD.value
                    )
                }
            ),
            colorbar_spec,
        )
        == method_catalog.MethodCallPolicy.AX_KEYWORD
    )

    assert figurecomposer_rendering._live_layout_axes(
        tool, render_if_missing=True
    ).shape == (1, 2)
    assert (
        method_execution._first_live_axis(
            tool,
            FigureAxesSelectionState(expression="axs[3, 3]"),
        )
        is None
    )
    assert (
        method_editor._method_float_pair_args(
            tool.operation_editor,
            FigureOperationState.method(
                family=FigureMethodFamily.AXES,
                name="set_xlim",
                axes=FigureAxesSelectionState(expression="axs[3, 3]"),
            ),
            method_catalog.AXES_METHODS["set_xlim"],
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
    live_layout_axes = figurecomposer_rendering._live_layout_axes
    assert live_layout_axes(grid_tool) is None
    grid_axes = live_layout_axes(grid_tool, render_if_missing=True)
    assert isinstance(grid_axes, dict)
    assert set(grid_axes) == {"axis-a"}
    monkeypatch.setattr(
        method_execution,
        "_live_layout_axes",
        lambda _tool, *, render_if_missing=False: grid_axes,
    )
    assert (
        method_execution._first_live_axis(
            grid_tool,
            FigureAxesSelectionState(axes=(), axes_ids=()),
        )
        is grid_axes["axis-a"]
    )
    assert live_layout_axes(grid_tool) is None
    assert (
        method_state._limit_method_default_args(
            method_catalog.AXES_METHODS["set_xlim"],
            None,
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

    int_control = method_catalog.MethodControlSpec(
        kind=method_catalog.MethodControlKind.INT_ARG,
        label="Count",
        tooltip="count tooltip",
        object_name="count",
        default=None,
        step=2,
    )
    int_spin = method_editor._int_spinbox(None, int_control, parent=tool)
    assert int_spin.value() == 0
    assert int_spin.singleStep() == 2
    float_control = method_catalog.MethodControlSpec(
        kind=method_catalog.MethodControlKind.FLOAT_ARG,
        label="Value",
        tooltip="value tooltip",
        object_name="value",
        default=None,
        decimals=2,
        step=0.25,
    )
    float_spin = method_editor._float_spinbox(None, float_control, parent=tool)
    assert float_spin.value() == pytest.approx(0.0)
    assert float_spin.decimals() == 2
    assert float_spin.singleStep() == pytest.approx(0.25)
    assert "multiple values" in method_editor._numeric_control_tooltip(
        float_control,
        mixed=True,
    )

    default_from_window = tool.operation_editor.subplot_parameter_default("left")
    assert default_from_window == pytest.approx(
        tool.figure_window.figure.subplotpars.left
    )
    spin_operation = FigureOperationState.method(
        family=FigureMethodFamily.FIGURE,
        name="subplots_adjust",
        kwargs={"left": "bad"},
    )
    adjust_spin = method_editor._subplots_adjust_spinbox(
        tool.operation_editor,
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
    method_editor._update_current_subplots_adjust_kwarg(
        adjust_tool.operation_editor,
        "left",
        None,
    )
    assert "left" not in adjust_tool.tool_status.operations[0].method_kwargs

    layout_spec = method_catalog.FIGURE_METHODS["set_layout_engine"]
    assert (
        method_state._layout_engine_name(
            FigureOperationState.method(
                family=FigureMethodFamily.FIGURE,
                name="set_layout_engine",
            ),
            layout_spec,
        )
        == "none"
    )
    assert method_state._filter_layout_engine_kwargs(
        (),
        {"pad": 0.1},
    ) == {"pad": 0.1}
    assert method_state._filter_layout_engine_kwargs(
        ("tight",),
        {"pad": 0.1, "hspace": 0.2},
    ) == {"pad": 0.1}

    with pytest.raises(ValueError, match="argument index"):
        method_editor._control_arg_index(
            method_catalog.MethodControlSpec(
                kind=method_catalog.MethodControlKind.TEXT_ARG,
                label="Missing",
                tooltip="missing",
                object_name="missing",
            )
        )
    with pytest.raises(ValueError, match="keyword name"):
        method_editor._control_key(
            method_catalog.MethodControlSpec(
                kind=method_catalog.MethodControlKind.TEXT_KWARG,
                label="Missing",
                tooltip="missing",
                object_name="missing",
            )
        )

    assert method_state._empty_text_as_none("") is None
    assert method_state._empty_text_as_none("title") == "title"
    assert method_state._string_tuple_from_text_or_none("") is None
    assert method_state._string_tuple_from_text_or_none("a, b") == ("a", "b")
    assert method_state._format_int_value(None) == ""
    assert method_state._format_int_value(2.0) == "2"
    assert method_state._format_float_value(None) == ""
    assert method_state._format_float_value(1.25) == "1.25"
    assert method_state._format_literal_value(None) == ""
    assert method_state._format_literal_value({"alpha": 0.5}) == "alpha=0.5"
    assert method_state._format_literal_value((1, 2)) == "1, 2"
    assert method_state._format_literal_value("literal") == '"literal"'
    assert method_plot_editor._format_plot_sequence(3.0) == "3.0"
    assert method_plot_editor._plot_sequence_from_text("") == ()
    assert method_plot_editor._plot_sequence_from_text("1, 2") == (1, 2)
    assert method_state._format_aspect_value(None) == ""
    assert method_state._format_aspect_value("equal") == "equal"
    assert method_state._format_aspect_value(2) == "2"
    assert method_state._format_aspect_value(("custom",)) == '("custom",)'
    assert method_state._literal_value_from_text("alpha=0.5") == {"alpha": 0.5}
    assert method_state._literal_value_from_text("") is None
    assert method_state._literal_value_from_text("[1, 2]") == [1, 2]
    assert method_state._aspect_value_from_text("") is None
    assert method_state._aspect_value_from_text("equal") == "equal"
    assert method_state._aspect_value_from_text("2") == 2.0
    assert method_state._aspect_value_from_text("custom") == "custom"
    assert method_state._aspect_value_from_text("'manual'") == "manual"
    assert method_state._aspect_value_from_text("[1, 2]") == "[1, 2]"
    assert method_state._optional_literal_from_text("") is None
    assert method_state._optional_literal_from_text("alpha=0.5") == {"alpha": 0.5}
    assert method_state._optional_float_from_text("") is None
    assert method_state._optional_float_from_text("1.5") == 1.5
    assert method_state._optional_int_from_text("") is None
    assert method_state._optional_int_from_text("3") == 3

    assert method_plot_editor._plot_y_arg_value(
        FigureOperationState.method(family=FigureMethodFamily.AXES, name="plot"),
        method_catalog.AXES_METHODS["plot"],
    ) == (0.0, 1.0)
    empty_default_spec = method_catalog.MethodSpec(
        family=FigureMethodFamily.AXES,
        name="empty_default",
        label="empty_default",
        tooltip="empty",
        target_domain=method_catalog.MethodTargetDomain.AXES,
        call_policy=method_catalog.MethodCallPolicy.BOUND_EACH_AXIS,
    )
    assert (
        method_plot_editor._plot_y_arg_value(
            FigureOperationState.method(
                family=FigureMethodFamily.AXES,
                name="empty_default",
            ),
            empty_default_spec,
        )
        == ()
    )

    assert method_editor._family_from_label("bad") == FigureMethodFamily.ERLAB
    assert (
        method_editor._family_from_label("Figure Method") == FigureMethodFamily.FIGURE
    )
    assert (
        method_editor._call_policy_from_label("Selected axes together")
        == method_catalog.MethodCallPolicy.AX_KEYWORD
    )
    assert (
        method_editor._call_policy_from_label("plain_call")
        == method_catalog.MethodCallPolicy.PLAIN_CALL
    )
    assert (
        method_catalog._method_selector_text(method_catalog.AXES_METHODS["set_xlabel"])
        == "set_xlabel"
    )
    assert (
        method_catalog._method_selector_text(method_catalog.FIGURE_METHODS["supxlabel"])
        == "supxlabel"
    )
    assert (
        method_catalog._method_selector_text(
            method_catalog.ERLAB_METHODS["clean_labels"]
        )
        == "clean_labels"
    )
    assert (
        method_editor._method_combo_object_name(FigureMethodFamily.FIGURE)
        == "figureComposerFigureMethodCombo"
    )
    assert (
        method_editor._method_kwargs_object_name(FigureMethodFamily.ERLAB)
        == "figureComposerERLabMethodKwEdit"
    )
    assert (
        method_catalog._method_display(
            FigureOperationState.method(
                family=FigureMethodFamily.FIGURE,
                name="supxlabel",
            )
        )
        == "fig.supxlabel"
    )
    assert (
        method_catalog._method_display(
            FigureOperationState.method(
                family=FigureMethodFamily.ERLAB,
                name="clean_labels",
            )
        )
        == "eplt.clean_labels"
    )
    assert (
        method_catalog._callable_display(method_catalog.FIGURE_METHODS["supxlabel"])
        == "fig.supxlabel"
    )
    assert (
        method_catalog._callable_display(colorbar_spec)
        == "erlab.plotting.nice_colorbar"
    )
    assert (
        method_catalog._method_doc_url(
            method_catalog.MethodSpec(
                family=FigureMethodFamily.ERLAB,
                name="custom_doc",
                label="custom_doc",
                tooltip="custom",
                target_domain=method_catalog.MethodTargetDomain.NONE,
                call_policy=method_catalog.MethodCallPolicy.PLAIN_CALL,
                doc_url="https://example.test/docs",
            )
        )
        == "https://example.test/docs"
    )

    transform_figure, transform_axis = plt.subplots()
    try:
        assert (
            method_execution._transform_component(
                transform_figure,
                transform_axis,
                "figure",
            )
            is transform_figure.transFigure
        )
        assert (
            method_execution._transform_component(
                transform_figure,
                transform_axis,
                "dpi",
            )
            is transform_figure.dpi_scale_trans
        )
        assert method_execution._transform_component_code("figure") == (
            "fig.transFigure"
        )
        assert (
            method_execution._transform_component_code("dpi") == "fig.dpi_scale_trans"
        )
        assert (
            method_execution._render_method_transform(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="plot",
                ),
                method_catalog.AXES_METHODS["plot"],
                figure=transform_figure,
                axis=transform_axis,
            )
            is None
        )
        assert (
            method_execution._method_transform_code(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="plot",
                ),
                method_catalog.AXES_METHODS["plot"],
            )
            is None
        )
    finally:
        plt.close(transform_figure)

    positional_text_spec = method_catalog.MethodSpec(
        family=FigureMethodFamily.AXES,
        name="text_values_positional",
        label="text_values_positional",
        tooltip="test",
        target_domain=method_catalog.MethodTargetDomain.AXES,
        call_policy=method_catalog.MethodCallPolicy.BOUND_EACH_AXIS,
        text_values_policy=method_catalog.MethodTextValuesPolicy.POSITIONAL,
    )
    keyword_text_spec = method_catalog.MethodSpec(
        family=FigureMethodFamily.AXES,
        name="text_values_keyword",
        label="text_values_keyword",
        tooltip="test",
        target_domain=method_catalog.MethodTargetDomain.AXES,
        call_policy=method_catalog.MethodCallPolicy.BOUND_EACH_AXIS,
        text_values_policy=method_catalog.MethodTextValuesPolicy.KWARG,
        text_values_kwarg="labels",
    )
    text_operation = FigureOperationState.method(
        family=FigureMethodFamily.AXES,
        name="text_values",
    ).model_copy(update={"text_values": ("A", "B")})
    assert method_execution._render_args_kwargs(
        tool,
        text_operation,
        positional_text_spec,
    )[0] == (["A", "B"],)
    assert method_execution._render_args_kwargs(
        tool,
        text_operation,
        keyword_text_spec,
    )[1] == {"labels": ["A", "B"]}
    assert method_execution._code_args_kwargs(
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
    method_editor._update_current_method_name(
        no_operation_tool.operation_editor, "plot"
    )

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(1)
    )
    tool.operation_editor.select_section("method")
    figure_method_combo = tool.findChild(
        QtWidgets.QComboBox, "figureComposerFigureMethodCombo"
    )
    assert figure_method_combo is not None
    assert figure_method_combo.currentData() == "set_layout_engine"

    method_editor._update_current_method_args(tool.operation_editor, ("none",))
    assert tool.tool_status.operations[1].method_args == ("none",)
    method_editor._update_current_layout_engine(tool.operation_editor, 0, "tight")
    assert tool.tool_status.operations[1].method_args == ("tight",)
    assert tool.tool_status.operations[1].method_kwargs == {"pad": 0.1}
    method_editor._update_current_method_arg(tool.operation_editor, 2, "third")
    assert tool.tool_status.operations[1].method_args == ("tight", None, "third")
    method_editor._update_current_method_string_tuple_arg(
        tool.operation_editor, 4, "tail"
    )
    assert tool.tool_status.operations[1].method_args == (
        "tight",
        None,
        "third",
        (),
        ("tail",),
    )
    method_editor._update_current_method_string_tuple_arg(
        tool.operation_editor, 1, "a, b"
    )
    assert tool.tool_status.operations[1].method_args == (
        "tight",
        ("a", "b"),
        "third",
        (),
        ("tail",),
    )
    method_editor._update_current_method_string_tuple_arg(tool.operation_editor, 1, "")
    assert tool.tool_status.operations[1].method_args == ("tight",)
    method_editor._update_current_method_kwarg(tool.operation_editor, "pad", None)
    assert tool.tool_status.operations[1].method_kwargs == {}
    method_editor._update_current_method_kwarg(tool.operation_editor, "pad", 0.3)
    assert tool.tool_status.operations[1].method_kwargs == {"pad": 0.3}
    method_editor._operation_trust_update_callback(tool.operation_editor)(True)
    assert tool.tool_status.operations[1].trusted is True
    method_editor._method_float_pair_args_update_callback(tool.operation_editor)("0, 1")
    assert tool.tool_status.operations[1].method_args == (0.0, 1.0)
    method_editor._update_current_method_family(
        tool.operation_editor, FigureMethodFamily.AXES
    )
    assert tool.tool_status.operations[1].method_family == FigureMethodFamily.AXES
    method_editor._update_current_method_family(
        tool.operation_editor, FigureMethodFamily.FIGURE
    )
    assert tool.tool_status.operations[1].method_family == FigureMethodFamily.FIGURE
    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(1)
    )
    tool.operation_editor.select_section("method")
    method_editor._update_current_method_name(
        tool.operation_editor, "set_layout_engine"
    )
    assert tool.tool_status.operations[1].method_name == "set_layout_engine"
    method_editor._update_current_method_call_policy(
        tool.operation_editor,
        method_catalog.MethodCallPolicy.PLAIN_CALL,
    )
    assert tool.tool_status.operations[1].method_call_policy == "plain_call"
    method_editor._update_current_method_call_policy(
        tool.operation_editor,
        method_catalog.MethodCallPolicy.BOUND_FIGURE,
    )
    assert tool.tool_status.operations[1].method_call_policy is None
    method_editor._update_current_method_text_values(tool.operation_editor, "\nlabel\n")
    assert tool.tool_status.operations[1].text_values == ("label",)


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
    tool.operation_editor.select_section("method")
    method_page = tool.operation_editor.stack.currentWidget()
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
    method_editor._update_current_method_name(tool.operation_editor, "set_xlim")
    tool.operation_editor.select_section("method")
    qtbot.wait_until(
        lambda: (
            tool.operation_editor.stack.currentWidget().findChild(
                QtWidgets.QLineEdit, "figureComposerAxesMethodLimitsEdit"
            )
            is not None
        ),
        timeout=5000,
    )
    limits_edit = tool.operation_editor.stack.currentWidget().findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodLimitsEdit"
    )
    assert limits_edit is not None
    assert tool.tool_status.operations[-1].method_args == pytest.approx(expected_xlim)
    assert limits_edit.text() == f"{expected_xlim[0]:g}, {expected_xlim[1]:g}"

    tool._add_operation("method:axes")
    method_editor._update_current_method_name(tool.operation_editor, "set_ylim")
    tool.operation_editor.select_section("method")
    qtbot.wait_until(
        lambda: (
            tool.operation_editor.stack.currentWidget().findChild(
                QtWidgets.QLineEdit, "figureComposerAxesMethodLimitsEdit"
            )
            is not None
        ),
        timeout=5000,
    )
    limits_edit = tool.operation_editor.stack.currentWidget().findChild(
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
    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )
    tool.operation_editor.select_section("method")

    method_page = tool.operation_editor.stack.currentWidget()
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

    method_editor._update_current_method_name(tool.operation_editor, "set_xlim")
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
    method_editor._update_current_method_family(
        tool.operation_editor, FigureMethodFamily.AXES
    )
    assert tool.tool_status.operations[0] == initial_operation

    tool._updating_controls = True
    try:
        method_editor._update_current_method_name(tool.operation_editor, "set_ylim")
    finally:
        tool._updating_controls = False
    assert tool.tool_status.operations[0] == initial_operation

    combo_no_none = method_catalog.MethodControlSpec(
        kind=method_catalog.MethodControlKind.ARG_COMBO,
        label="Choice",
        tooltip="choice",
        object_name="choice",
        arg_index=0,
        options=("keep",),
    )
    combo_with_none = method_catalog.MethodControlSpec(
        kind=method_catalog.MethodControlKind.ARG_COMBO,
        label="Choice",
        tooltip="choice",
        object_name="choice",
        arg_index=0,
        options=("keep",),
        none_label="None",
    )
    bool_combo = method_catalog.MethodControlSpec(
        kind=method_catalog.MethodControlKind.BOOL_KWARG_COMBO,
        label="Flag",
        tooltip="flag",
        object_name="flag",
        key="flag",
    )
    assert not method_state._control_accepts_value(combo_no_none, None)
    assert method_state._control_accepts_value(combo_with_none, None)
    assert method_state._control_accepts_value(bool_combo, True)
    assert not method_state._control_accepts_value(bool_combo, "True")

    assert (
        method_state._transfer_axis_label_loc(
            method_catalog.AXES_METHODS["set_ylabel"],
            method_catalog.AXES_METHODS["set_xlabel"],
            "top",
        )
        == "right"
    )
    assert (
        method_state._transfer_axis_label_loc(
            method_catalog.AXES_METHODS["set_title"],
            method_catalog.AXES_METHODS["set_xlabel"],
            "unchanged",
        )
        == "unchanged"
    )

    float_pair_updates = method_state._method_transfer_updates(
        FigureOperationState.method(
            family=FigureMethodFamily.AXES,
            name="set_xlim",
            args=(1.0, 2.0),
            axes=FigureAxesSelectionState(axes=((0, 0),)),
        ),
        method_catalog.AXES_METHODS["set_ylim"],
        default_axis=None,
    )
    assert float_pair_updates["method_args"] == (1.0, 2.0)

    default_arg_updates = method_state._method_transfer_updates(
        FigureOperationState.method(
            family=FigureMethodFamily.AXES,
            name="set_xlabel",
            args=("x",),
            axes=FigureAxesSelectionState(axes=((0, 0),)),
        ),
        method_catalog.AXES_METHODS["set_ylabel"],
        default_axis=None,
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
    plot_updates = method_state._method_transfer_updates(
        plot_operation,
        method_catalog.AXES_METHODS["plot"],
        default_axis=None,
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
    errorbar_to_plot_updates = method_state._method_transfer_updates(
        errorbar_operation,
        method_catalog.AXES_METHODS["plot"],
        default_axis=None,
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

    errorbar_to_errorbar_updates = method_state._method_transfer_updates(
        errorbar_operation,
        method_catalog.AXES_METHODS["errorbar"],
        default_axis=None,
    )
    assert errorbar_to_errorbar_updates["method_plot_xerr"] == (
        FigureMethodPlotValueState(source="stderr", kind="data")
    )
    assert errorbar_to_errorbar_updates["method_kwargs"]["xerr"] == (0.1, 0.2)

    text_arg = method_catalog.MethodControlSpec(
        kind=method_catalog.MethodControlKind.TEXT_ARG,
        label="Text",
        tooltip="text",
        object_name="text",
        arg_index=0,
    )
    accepted_combo = method_catalog.MethodControlSpec(
        kind=method_catalog.MethodControlKind.ARG_COMBO,
        label="Accepted",
        tooltip="accepted",
        object_name="accepted",
        arg_index=2,
        options=("keep",),
    )
    source_rejected_combo = method_catalog.MethodControlSpec(
        kind=method_catalog.MethodControlKind.ARG_COMBO,
        label="Rejected",
        tooltip="rejected",
        object_name="rejected",
        arg_index=3,
        options=("bad",),
    )
    target_rejected_combo = method_catalog.MethodControlSpec(
        kind=method_catalog.MethodControlKind.ARG_COMBO,
        label="Rejected",
        tooltip="rejected",
        object_name="rejected",
        arg_index=3,
        options=("ok",),
    )
    same_kwarg = method_catalog.MethodControlSpec(
        kind=method_catalog.MethodControlKind.FLOAT_KWARG,
        label="Same",
        tooltip="same",
        object_name="same",
        key="same",
    )
    source_mismatch_kwarg = method_catalog.MethodControlSpec(
        kind=method_catalog.MethodControlKind.TEXT_KWARG,
        label="Mismatch",
        tooltip="mismatch",
        object_name="mismatch",
        key="mismatch",
    )
    target_mismatch_kwarg = method_catalog.MethodControlSpec(
        kind=method_catalog.MethodControlKind.FLOAT_KWARG,
        label="Mismatch",
        tooltip="mismatch",
        object_name="mismatch",
        key="mismatch",
    )
    transform_control = method_catalog.MethodControlSpec(
        kind=method_catalog.MethodControlKind.TRANSFORM,
        label="Transform",
        tooltip="transform",
        object_name="transform",
    )
    source_spec = method_catalog.MethodSpec(
        family=FigureMethodFamily.AXES,
        name="transfer_source",
        label="transfer_source",
        tooltip="test",
        target_domain=method_catalog.MethodTargetDomain.AXES,
        call_policy=method_catalog.MethodCallPolicy.BOUND_EACH_AXIS,
        allowed_call_policies=(
            method_catalog.MethodCallPolicy.BOUND_EACH_AXIS,
            method_catalog.MethodCallPolicy.AX_KEYWORD,
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
        text_values_policy=method_catalog.MethodTextValuesPolicy.POSITIONAL,
    )
    target_spec = method_catalog.MethodSpec(
        family=FigureMethodFamily.AXES,
        name="transfer_target",
        label="transfer_target",
        tooltip="test",
        target_domain=method_catalog.MethodTargetDomain.AXES,
        call_policy=method_catalog.MethodCallPolicy.BOUND_EACH_AXIS,
        allowed_call_policies=(
            method_catalog.MethodCallPolicy.BOUND_EACH_AXIS,
            method_catalog.MethodCallPolicy.AX_KEYWORD,
        ),
        controls=(
            text_arg,
            accepted_combo,
            target_rejected_combo,
            same_kwarg,
            target_mismatch_kwarg,
            transform_control,
        ),
        text_values_policy=method_catalog.MethodTextValuesPolicy.POSITIONAL,
    )
    monkeypatch.setitem(method_catalog.AXES_METHODS, "transfer_source", source_spec)
    monkeypatch.setitem(method_catalog.AXES_METHODS, "transfer_target", target_spec)
    transfer_operation = FigureOperationState.method(
        family=FigureMethodFamily.AXES,
        name="transfer_source",
        args=("default", "ignored", "keep", "bad"),
        kwargs={"same": 2.0, "mismatch": "skip"},
        axes=FigureAxesSelectionState(axes=((0, 0),)),
    ).model_copy(
        update={
            "method_call_policy": (method_catalog.MethodCallPolicy.AX_KEYWORD.value),
            "text_values": ("A", "B"),
            "method_transform": "custom",
            "method_transform_x": "figure",
            "method_transform_y": "data",
            "method_transform_expression": "ax.transData",
        }
    )

    transfer_updates = method_state._method_transfer_updates(
        transfer_operation,
        target_spec,
        default_axis=None,
    )
    assert transfer_updates["method_args"] == (None, None, "keep")
    assert transfer_updates["method_kwargs"] == {"same": 2.0}
    assert (
        transfer_updates["method_call_policy"]
        == method_catalog.MethodCallPolicy.AX_KEYWORD.value
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
    tool.operation_editor.select_section("method")
    method_page = tool.operation_editor.stack.currentWidget()
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

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )
    tool.operation_editor.select_section("method")
    label_edit = tool.operation_editor.stack.currentWidget().findChild(
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

    assert tool.operation_editor.stack.currentWidget().objectName() == (
        "figureComposerIncompatibleBatchPage"
    )
    assert (
        tool.operation_editor.stack.currentWidget().findChild(
            QtWidgets.QPlainTextEdit, "figureComposerAxesMethodTitleEdit"
        )
        is None
    )


@pytest.mark.parametrize(
    ("method_name", "method_args", "text_values"),
    [
        (
            "label_subplots",
            (),
            tuple(f"label-{index}" for index in range(9)),
        ),
        (
            "label_subplot_properties",
            ({"value": list(range(9))},),
            (),
        ),
        (
            "set_titles",
            (),
            tuple(f"title-{index}" for index in range(9)),
        ),
        (
            "set_xlabels",
            (),
            tuple(f"x-{index}" for index in range(9)),
        ),
        (
            "set_ylabels",
            (),
            tuple(f"y-{index}" for index in range(9)),
        ),
    ],
)
def test_figure_composer_ordered_methods_preserve_rectangular_axes(
    qtbot, monkeypatch, method_name, method_args, text_values
) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    axes = FigureAxesSelectionState(
        axes=tuple((row, col) for row in range(3) for col in range(3))
    )
    operation = FigureOperationState.method(
        family=FigureMethodFamily.ERLAB,
        name=method_name,
        axes=axes,
    ).model_copy(
        update={
            "method_args": method_args,
            "method_kwargs": {"order": "F"},
            "text_values": text_values,
        }
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=3, ncols=3),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    calls: list[tuple[tuple[int, ...], str | None]] = []

    def capture_axes(axes, *_args, **kwargs) -> None:
        calls.append((np.asarray(axes, dtype=object).shape, kwargs.get("order")))

    monkeypatch.setattr(eplt, method_name, capture_axes)

    preview_figure = plt.figure()
    figurecomposer_rendering._render_into_figure(
        tool, preview_figure, sync_visible=False
    )
    assert calls == [((3, 3), "F")]

    calls.clear()
    namespace: dict[str, typing.Any] = {"data": data}
    exec(tool.generated_code(), namespace)  # noqa: S102
    assert calls == [((3, 3), "F")]

    plt.close(preview_figure)
    plt.close(namespace["fig"])


def test_figure_composer_label_subplots_preserves_empty_text_rows(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    operation = FigureOperationState.method(
        family=FigureMethodFamily.ERLAB,
        name="label_subplots",
        axes=FigureAxesSelectionState(axes=((0, 0), (0, 1), (0, 2))),
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=3),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    tool.operation_editor.select_section("method")
    text_edit = tool.operation_editor.stack.currentWidget().findChild(
        QtWidgets.QPlainTextEdit,
        "figureComposerMethodTextValuesEdit",
    )
    assert text_edit is not None

    text_edit.setPlainText("first\n\nthird")
    operation = tool.tool_status.operations[0]
    assert operation.text_values == ("first", "", "third")
    _args, kwargs = method_execution._render_args_kwargs(
        tool,
        operation,
        method_catalog._method_spec(operation),
    )
    assert kwargs["values"] == ["first", "", "third"]

    namespace: dict[str, typing.Any] = {"data": data}
    exec(tool.generated_code(), namespace)  # noqa: S102
    assert all(len(axis.artists) == 1 for axis in namespace["axs"].flat)
    plt.close(namespace["fig"])

    text_edit.setPlainText("")
    assert tool.tool_status.operations[0].text_values == ()


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
        tool.operation_panel.operation_list.setCurrentItem(
            tool.operation_panel.operation_list.topLevelItem(row)
        )
        tool.operation_editor.select_section("method")
        page = tool.operation_editor.stack.currentWidget()
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
        combo = combo_box(page, name)
        index = combo.findData(text)
        if index >= 0:
            _activate_combo_index(combo, index)
        else:
            _activate_combo_text(combo, text)

    def operation(row: int) -> FigureOperationState:
        return tool.tool_status.operations[row]

    page = select_method(0)
    set_combo(page, "figureComposerERLabCleanLabelsRemoveInnerTicksCombo", "True")
    assert operation(0).method_args == (True,)

    page = select_method(1)
    order_combo = combo_box(page, "figureComposerERLabLabelSubplotsOrderCombo")
    assert [order_combo.itemData(index) for index in range(order_combo.count())] == [
        "C",
        "F",
    ]
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

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )
    tool.operation_editor.select_section("method")
    title_edit = tool.operation_editor.stack.currentWidget().findChild(
        QtWidgets.QPlainTextEdit
    )
    assert title_edit is not None
    title_edit.setPlainText("Left\n")
    assert tool.tool_status.operations[0].text_values == ("Left", "")

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(1)
    )
    tool.operation_editor.select_section("method")
    xlabel_edit = tool.operation_editor.stack.currentWidget().findChild(
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
