"""Prepared-selection cache behavior for Figure Composer Plot Slices steps."""

import numpy as np
import xarray as xr
from matplotlib.figure import Figure

import erlab.accessors.general as accessor_general
import erlab.interactive._figurecomposer._operations._plot_slices._cache as cache_module
import erlab.interactive._figurecomposer._rendering as figurecomposer_rendering
from erlab.interactive._figurecomposer import (
    FigureAxesSelectionState,
    FigureComposerTool,
    FigureOperationState,
    FigureRecipeState,
    FigureSourceState,
    FigureSubplotsState,
)
from erlab.interactive._figurecomposer._operations._plot_slices._cache import (
    _PlotSlicesSelectionCache,
)

from ._common import _figure_composer_image_source


def _slices_tool(
    data: xr.DataArray,
    *operations: FigureOperationState,
    ncols: int = 2,
) -> FigureComposerTool:
    return FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=ncols),
            sources=(FigureSourceState(name="data"),),
            operations=operations,
            primary_source="data",
        ),
        source_data={"data": data},
    )


def _slice_operation(
    label: str,
    *,
    axes: tuple[tuple[int, int], ...],
    transpose: bool = False,
) -> FigureOperationState:
    return FigureOperationState.plot_slices(
        label=label,
        sources=("data",),
        axes=FigureAxesSelectionState(axes=axes),
        slice_dim="eV",
        slice_values=(0.0, 0.5),
    ).model_copy(update={"transpose": transpose})


def test_plot_slices_selection_cache_is_bounded(monkeypatch) -> None:
    monkeypatch.setattr(cache_module, "_MAX_ENTRIES", 2)
    monkeypatch.setattr(cache_module, "_MAX_BYTES", 32)
    cache = _PlotSlicesSelectionCache()
    first = (xr.DataArray(np.arange(2.0), dims="x"),)
    second = (xr.DataArray(np.arange(2.0), dims="x"),)
    third = (xr.DataArray(np.arange(2.0), dims="x"),)

    cache["first"] = first
    cache["second"] = second
    assert cache["first"] is first
    cache["third"] = third

    assert tuple(cache) == ("first", "third")
    cache["oversized"] = (xr.DataArray(np.arange(10.0), dims="x"),)
    assert "oversized" not in cache


def test_plot_slices_transpose_change_uses_a_new_selection(qtbot, monkeypatch) -> None:
    data = _figure_composer_image_source("data")
    axes = ((0, 0), (0, 1))
    operation = _slice_operation("slices", axes=axes)
    tool = _slices_tool(data, operation)
    qtbot.addWidget(tool)
    calls = 0
    original_qsel = accessor_general.SelectionAccessor.__call__

    def counted_qsel(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original_qsel(self, *args, **kwargs)

    monkeypatch.setattr(
        accessor_general.SelectionAccessor,
        "__call__",
        counted_qsel,
    )

    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    tool._document.replace_operation(
        0,
        operation.model_copy(update={"transpose": True}),
    )
    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    assert calls == 2


def test_plot_slices_selection_cache_tracks_selection_and_source_changes(
    qtbot, monkeypatch
) -> None:
    data = _figure_composer_image_source("data")
    operation = _slice_operation("slices", axes=((0, 0), (0, 1)))
    tool = _slices_tool(data, operation)
    qtbot.addWidget(tool)
    calls = 0
    original_qsel = accessor_general.SelectionAccessor.__call__

    def counted_qsel(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original_qsel(self, *args, **kwargs)

    monkeypatch.setattr(
        accessor_general.SelectionAccessor,
        "__call__",
        counted_qsel,
    )

    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    tool._document.replace_operation(
        0,
        operation.model_copy(update={"slice_values": (0.25, 0.75)}),
    )
    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    assert calls == 2

    tool.set_source_data({"data": data + 1.0})
    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    assert calls == 3

    tool.tool_data.values[:] += 1.0
    tool.touch_source_data()
    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    assert calls == 4


def test_custom_code_disables_plot_slices_selection_cache(qtbot) -> None:
    data = _figure_composer_image_source("data")
    first = _slice_operation("before", axes=((0, 0),))
    custom = FigureOperationState.custom(
        label="mutate",
        code="data.values[:] += 1.0",
        trusted=True,
    )
    second = _slice_operation("after", axes=((0, 1),))
    tool = _slices_tool(data, first, custom, second)
    qtbot.addWidget(tool)

    figure = Figure()
    figurecomposer_rendering._render_into_figure(tool, figure, sync_visible=False)

    before = np.asarray(figure.axes[0].images[0].get_array())
    after = np.asarray(figure.axes[1].images[0].get_array())
    np.testing.assert_allclose(after, before + 1.0)
    assert len(tool._plot_slices_cache) == 0


def test_plot_slices_persists_lazy_selection_across_redraws(qtbot) -> None:
    import dask
    import dask.array as da

    eager = _figure_composer_image_source("data")
    compute_calls = 0

    @dask.delayed
    def load_values() -> np.ndarray:
        nonlocal compute_calls
        compute_calls += 1
        return eager.values

    data = eager.copy(
        data=da.from_delayed(load_values(), shape=eager.shape, dtype=eager.dtype)
    )
    operation = _slice_operation("slices", axes=((0, 0), (0, 1)))
    tool = _slices_tool(data, operation)
    qtbot.addWidget(tool)

    with dask.config.set(scheduler="synchronous"):
        figurecomposer_rendering._render_into_figure(
            tool, tool.figure, sync_visible=False
        )
        figurecomposer_rendering._render_into_figure(
            tool, tool.figure, sync_visible=False
        )

    assert compute_calls == 1
