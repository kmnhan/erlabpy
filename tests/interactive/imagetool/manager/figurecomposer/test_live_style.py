import typing

import pytest

import erlab.interactive.utils
from erlab.interactive._figurecomposer import (
    FigureComposerTool,
    FigureMethodFamily,
    FigureOperationKind,
    FigureOperationState,
    FigurePlotSlicesPanelStyleState,
    FigureSourceState,
)
from erlab.interactive._figurecomposer import _tool as figurecomposer_tool
from erlab.interactive._figurecomposer._operations._plot_slices._render import (
    _PLOT_SLICES_MAPPABLE_OPERATION_ID_ATTR,
)

from ._common import _figure_composer_image_source


def _visible_image_tool(
    qtbot,
    kind: FigureOperationKind,
    *,
    operation_updates: dict[str, object] | None = None,
    later_operations: tuple[FigureOperationState, ...] = (),
) -> FigureComposerTool:
    data = _figure_composer_image_source("data").isel(eV=0)
    if kind == FigureOperationKind.PLOT_ARRAY:
        operation = FigureOperationState.plot_array(label="plot_array", source="data")
    else:
        operation = FigureOperationState.plot_slices(
            label="plot_slices", sources=("data",)
        )
    operation = operation.model_copy(
        update={"cmap": "viridis", **(operation_updates or {})}
    )
    tool = FigureComposerTool.from_sources(
        {"data": data},
        sources=(FigureSourceState(name="data", label="data"),),
        operations=(operation, *later_operations),
        primary_source="data",
    )
    qtbot.addWidget(tool)
    tool.show_figure_window(activate=False)
    qtbot.waitUntil(lambda: tool.figure_window.isVisible(), timeout=1000)
    return tool


def _operation_mappables(
    tool: FigureComposerTool, operation_id: str
) -> tuple[typing.Any, ...]:
    return tuple(
        mappable
        for axis in tool.figure.axes
        for mappable in (*axis.images, *axis.collections)
        if getattr(mappable, _PLOT_SLICES_MAPPABLE_OPERATION_ID_ATTR, None)
        == operation_id
    )


@pytest.mark.parametrize(
    "kind", [FigureOperationKind.PLOT_ARRAY, FigureOperationKind.PLOT_SLICES]
)
def test_figure_composer_updates_live_image_colormap_in_place(
    qtbot, monkeypatch, kind
) -> None:
    tool = _visible_image_tool(qtbot, kind)
    operation = tool.tool_status.operations[0]
    original_mappables = _operation_mappables(tool, operation.operation_id)
    assert len(original_mappables) == 1
    draw_idle_calls: list[None] = []

    monkeypatch.setattr(
        tool,
        "_redraw_plot",
        lambda **_kwargs: pytest.fail("a safe colormap change must not rerender"),
    )
    monkeypatch.setattr(tool.canvas, "draw_idle", lambda: draw_idle_calls.append(None))

    assert tool._update_operations_by_ids(
        (operation.operation_id,),
        lambda _index, target: target.model_copy(update={"cmap": "magma"}),
    )

    updated_mappables = _operation_mappables(tool, operation.operation_id)
    assert len(updated_mappables) == 1
    assert updated_mappables[0] is original_mappables[0]
    assert updated_mappables[0].get_cmap().name == "magma"
    assert tool.tool_status.operations[0].cmap == "magma"
    assert draw_idle_calls == [None]
    assert tool.preview_pixmap_stale


def test_figure_composer_live_colormap_falls_back_after_arbitrary_step(
    qtbot, monkeypatch
) -> None:
    later_method = FigureOperationState.method(
        family=FigureMethodFamily.AXES,
        name="set_title",
        args=("later",),
    )
    tool = _visible_image_tool(
        qtbot,
        FigureOperationKind.PLOT_ARRAY,
        later_operations=(later_method,),
    )
    operation = tool.tool_status.operations[0]
    mappable = _operation_mappables(tool, operation.operation_id)[0]
    redraw_calls: list[None] = []
    monkeypatch.setattr(
        tool, "_redraw_plot", lambda **_kwargs: redraw_calls.append(None)
    )

    assert tool._update_operations_by_ids(
        (operation.operation_id,),
        lambda _index, target: target.model_copy(update={"cmap": "magma"}),
    )

    assert redraw_calls == [None]
    assert mappable.get_cmap().name == "viridis"


def test_figure_composer_live_colormap_falls_back_for_panel_styles(
    qtbot, monkeypatch
) -> None:
    panel_style = FigurePlotSlicesPanelStyleState(
        map_index=0,
        slice_index=0,
        cmap="plasma",
    )
    tool = _visible_image_tool(
        qtbot,
        FigureOperationKind.PLOT_SLICES,
        operation_updates={
            "panel_styles_enabled": True,
            "panel_styles": (panel_style,),
        },
    )
    operation = tool.tool_status.operations[0]
    mappable = _operation_mappables(tool, operation.operation_id)[0]
    redraw_calls: list[None] = []
    monkeypatch.setattr(
        tool, "_redraw_plot", lambda **_kwargs: redraw_calls.append(None)
    )

    assert tool._update_operations_by_ids(
        (operation.operation_id,),
        lambda _index, target: target.model_copy(update={"cmap": "magma"}),
    )

    assert redraw_calls == [None]
    assert mappable.get_cmap().name == "plasma"


def test_figure_composer_live_colormap_does_not_handle_normalization_changes(
    qtbot, monkeypatch
) -> None:
    tool = _visible_image_tool(qtbot, FigureOperationKind.PLOT_ARRAY)
    operation = tool.tool_status.operations[0]
    mappable = _operation_mappables(tool, operation.operation_id)[0]
    previous_clim = mappable.get_clim()
    redraw_calls: list[None] = []
    monkeypatch.setattr(
        tool, "_redraw_plot", lambda **_kwargs: redraw_calls.append(None)
    )

    assert tool._update_operations_by_ids(
        (operation.operation_id,),
        lambda _index, target: target.model_copy(update={"vmin": 1.0}),
    )

    assert redraw_calls == [None]
    assert mappable.get_clim() == previous_clim


def test_figure_composer_live_colormap_uses_configured_default(
    qtbot, monkeypatch
) -> None:
    tool = _visible_image_tool(qtbot, FigureOperationKind.PLOT_ARRAY)
    operation = tool.tool_status.operations[0]
    mappable = _operation_mappables(tool, operation.operation_id)[0]
    monkeypatch.setattr(
        tool,
        "_redraw_plot",
        lambda **_kwargs: pytest.fail("a safe colormap change must not rerender"),
    )
    monkeypatch.setattr(
        tool,
        "_editor_styled_rcparams_value",
        lambda key: "plasma" if key == "image.cmap" else pytest.fail(key),
    )

    assert tool._update_operations_by_ids(
        (operation.operation_id,),
        lambda _index, target: target.model_copy(update={"cmap": None}),
    )

    assert mappable.get_cmap().name == "plasma"


def test_figure_composer_live_colormap_falls_back_for_extra_kwarg_override(
    qtbot, monkeypatch
) -> None:
    tool = _visible_image_tool(
        qtbot,
        FigureOperationKind.PLOT_ARRAY,
        operation_updates={"extra_kwargs": {"cmap": "plasma"}},
    )
    operation = tool.tool_status.operations[0]
    mappable = _operation_mappables(tool, operation.operation_id)[0]
    redraw_calls: list[None] = []
    monkeypatch.setattr(
        tool, "_redraw_plot", lambda **_kwargs: redraw_calls.append(None)
    )

    assert tool._update_operations_by_ids(
        (operation.operation_id,),
        lambda _index, target: target.model_copy(update={"cmap": "magma"}),
    )

    assert redraw_calls == [None]
    assert mappable.get_cmap().name == "plasma"


@pytest.mark.parametrize("missing_tag", [False, True])
def test_figure_composer_live_colormap_falls_back_when_artist_is_unusable(
    qtbot, monkeypatch, missing_tag
) -> None:
    tool = _visible_image_tool(qtbot, FigureOperationKind.PLOT_ARRAY)
    operation = tool.tool_status.operations[0]
    mappable = _operation_mappables(tool, operation.operation_id)[0]
    if missing_tag:
        delattr(mappable, _PLOT_SLICES_MAPPABLE_OPERATION_ID_ATTR)
        cmap = "magma"
    else:
        cmap = "missing_colormap"
    redraw_calls: list[None] = []
    monkeypatch.setattr(
        tool, "_redraw_plot", lambda **_kwargs: redraw_calls.append(None)
    )

    assert tool._update_operations_by_ids(
        (operation.operation_id,),
        lambda _index, target: target.model_copy(update={"cmap": cmap}),
    )

    assert redraw_calls == [None]
    assert mappable.get_cmap().name == "viridis"


def test_figure_composer_live_colormap_falls_back_for_invalid_canvas(
    qtbot, monkeypatch
) -> None:
    tool = _visible_image_tool(qtbot, FigureOperationKind.PLOT_ARRAY)
    operation = tool.tool_status.operations[0]
    mappable = _operation_mappables(tool, operation.operation_id)[0]
    canvas = tool.canvas
    redraw_calls: list[None] = []
    monkeypatch.setattr(
        erlab.interactive.utils,
        "qt_is_valid",
        lambda *objects: all(obj is not canvas for obj in objects),
    )
    monkeypatch.setattr(
        tool, "_redraw_plot", lambda **_kwargs: redraw_calls.append(None)
    )

    assert tool._update_operations_by_ids(
        (operation.operation_id,),
        lambda _index, target: target.model_copy(update={"cmap": "magma"}),
    )

    assert redraw_calls == [None]
    assert mappable.get_cmap().name == "viridis"


def test_figure_composer_live_colormap_falls_back_when_layout_lookup_fails(
    qtbot, monkeypatch
) -> None:
    tool = _visible_image_tool(qtbot, FigureOperationKind.PLOT_SLICES)
    operation = tool.tool_status.operations[0]
    mappable = _operation_mappables(tool, operation.operation_id)[0]
    redraw_calls: list[None] = []

    def fail_layout_lookup(*_args, **_kwargs):
        raise RuntimeError("unavailable layout")

    monkeypatch.setattr(figurecomposer_tool, "_plot_slices_shape", fail_layout_lookup)
    monkeypatch.setattr(
        tool, "_redraw_plot", lambda **_kwargs: redraw_calls.append(None)
    )

    assert tool._update_operations_by_ids(
        (operation.operation_id,),
        lambda _index, target: target.model_copy(update={"cmap": "magma"}),
    )

    assert redraw_calls == [None]
    assert mappable.get_cmap().name == "viridis"


def test_figure_composer_live_colormap_respects_disabled_auto_redraw(
    qtbot, monkeypatch
) -> None:
    tool = _visible_image_tool(qtbot, FigureOperationKind.PLOT_ARRAY)
    operation = tool.tool_status.operations[0]
    mappable = _operation_mappables(tool, operation.operation_id)[0]
    tool.auto_redraw_check.setChecked(False)
    monkeypatch.setattr(
        tool,
        "_redraw_plot",
        lambda **_kwargs: pytest.fail("disabled auto-redraw must not redraw"),
    )

    assert tool._update_operations_by_ids(
        (operation.operation_id,),
        lambda _index, target: target.model_copy(update={"cmap": "magma"}),
    )

    assert mappable.get_cmap().name == "viridis"
    assert tool._auto_redraw_dirty
    assert tool.preview_pixmap_stale
