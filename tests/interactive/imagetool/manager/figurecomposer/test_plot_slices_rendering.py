"""Plot Slices rendering behavior tests."""

from ._plot_slices_common import (
    Figure,
    FigureAxesSelectionState,
    FigureComposerTool,
    FigureOperationState,
    FigureRecipeState,
    FigureSourceState,
    FigureSubplotsState,
    _figure_composer_image_source,
    accessor_general,
    figurecomposer_rendering,
    np,
    pytest,
    xr,
)


@pytest.mark.parametrize("transpose", [False, True])
def test_figure_composer_plot_slices_reuses_selection_cache_across_redraws(
    qtbot, monkeypatch, transpose: bool
) -> None:
    data = _figure_composer_image_source("data")
    axes = FigureAxesSelectionState(axes=((0, 0), (0, 1)))
    first_operation = FigureOperationState.plot_slices(
        label="first",
        sources=("data",),
        axes=axes,
        slice_dim="eV",
        slice_values=(0.0, 0.5),
    ).model_copy(update={"cmap": "viridis", "transpose": transpose})
    second_operation = FigureOperationState.plot_slices(
        label="second",
        sources=("data",),
        axes=axes,
        slice_dim="eV",
        slice_values=(0.0, 0.5),
    ).model_copy(update={"cmap": "magma", "transpose": transpose})
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(first_operation, second_operation),
            primary_source="data",
        ),
        source_data={"data": data},
    )
    qtbot.addWidget(tool)

    calls: list[dict[str, object]] = []
    original_qsel = accessor_general.SelectionAccessor.__call__

    def counted_qsel(self, *args, **kwargs):
        calls.append(dict(kwargs))
        return original_qsel(self, *args, **kwargs)

    monkeypatch.setattr(
        accessor_general.SelectionAccessor,
        "__call__",
        counted_qsel,
    )

    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)

    assert len(calls) == 1
    assert calls[0]["eV"] == [0.0, 0.5]
    assert [len(axis.images) for axis in tool.figure.axes] == [2, 2]
    assert [image.get_cmap().name for image in tool.figure.axes[0].images] == [
        "viridis",
        "magma",
    ]

    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)

    assert len(calls) == 1

    cached_image = np.asarray(tool.figure.axes[0].images[0].get_array()).copy()
    data.values[:] += 1000.0
    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)

    assert len(calls) == 1
    np.testing.assert_allclose(tool.figure.axes[0].images[0].get_array(), cached_image)

    replacement = data + 1.0
    tool._document.replace_source_payloads({"data": replacement}, {})
    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)

    assert len(calls) == 2

    custom = FigureOperationState.custom(label="custom", code="pass", trusted=True)
    tool._document.replace_recipe(
        tool.tool_status.model_copy(
            update={"operations": (*tool.tool_status.operations, custom)}
        )
    )
    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)

    assert len(calls) == 6

    tool._document.replace_recipe(
        tool.tool_status.model_copy(
            update={"operations": tool.tool_status.operations[:-1]}
        )
    )
    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)

    assert len(calls) == 7


def test_figure_composer_custom_code_disables_selection_reuse_within_render(
    qtbot,
) -> None:
    data = _figure_composer_image_source("data")
    first = FigureOperationState.plot_slices(
        label="before",
        sources=("data",),
        axes=FigureAxesSelectionState(axes=((0, 0),)),
        slice_dim="eV",
        slice_values=(0.0,),
    )
    custom = FigureOperationState.custom(
        label="mutate",
        code="data.values[:] += 1.0",
        trusted=True,
    )
    second = FigureOperationState.plot_slices(
        label="after",
        sources=("data",),
        axes=FigureAxesSelectionState(axes=((0, 1),)),
        slice_dim="eV",
        slice_values=(0.0,),
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=2),
            sources=(FigureSourceState(name="data"),),
            operations=(first, custom, second),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    figure = Figure()
    figurecomposer_rendering._render_into_figure(tool, figure, sync_visible=False)

    before = np.asarray(figure.axes[0].images[0].get_array())
    after = np.asarray(figure.axes[1].images[0].get_array())
    np.testing.assert_allclose(after, before + 1.0)
    assert len(tool._render_data_cache) == 0


def test_figure_composer_plot_slices_cache_preserves_repeated_source_order(
    qtbot,
) -> None:
    first_data = _figure_composer_image_source("first")
    second_data = (first_data + 1000.0).rename("second")
    axes = FigureAxesSelectionState(axes=((0, 0), (0, 1), (0, 2)))
    first = FigureOperationState.plot_slices(
        label="first",
        sources=("first", "second", "first"),
        axes=axes,
        slice_dim="eV",
        slice_values=(0.0,),
    )
    second = FigureOperationState.plot_slices(
        label="second",
        sources=("first", "second", "second"),
        axes=axes,
        slice_dim="eV",
        slice_values=(0.0,),
    )
    tool = FigureComposerTool(
        first_data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=3),
            sources=(
                FigureSourceState(name="first"),
                FigureSourceState(name="second"),
            ),
            operations=(first, second),
            primary_source="first",
        ),
        source_data={"first": first_data, "second": second_data},
    )
    qtbot.addWidget(tool)

    figure = Figure()
    figurecomposer_rendering._render_into_figure(tool, figure, sync_visible=False)

    expected_first = first_data.qsel(eV=[0.0]).isel(eV=0).values
    expected_second = second_data.qsel(eV=[0.0]).isel(eV=0).values
    np.testing.assert_allclose(figure.axes[2].images[0].get_array(), expected_first)
    np.testing.assert_allclose(figure.axes[2].images[1].get_array(), expected_second)


def test_figure_composer_plot_slices_uses_persisted_selection_on_first_render(
    qtbot,
) -> None:
    import dask
    import dask.array as da

    eager = _figure_composer_image_source("data")
    compute_calls: list[None] = []

    @dask.delayed
    def load_values() -> np.ndarray:
        compute_calls.append(None)
        return eager.values

    data = eager.copy(
        data=da.from_delayed(load_values(), shape=eager.shape, dtype=eager.dtype)
    )
    operation = FigureOperationState.plot_slices(
        label="slices",
        sources=("data",),
        axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
        slice_dim="eV",
        slice_values=(0.0, 0.5),
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

    with dask.config.set(scheduler="synchronous"):
        figurecomposer_rendering._render_into_figure(
            tool, tool.figure, sync_visible=False
        )
        assert compute_calls == [None]

        figurecomposer_rendering._render_into_figure(
            tool, tool.figure, sync_visible=False
        )

    assert compute_calls == [None]


def test_figure_composer_plot_slices_order_change_matches_cold_transform_cache(
    qtbot,
) -> None:
    first = xr.DataArray(
        np.array([[1.0, 2.0], [2.0, 4.0]]),
        dims=("eV", "kx"),
        coords={"eV": [0.0, 1.0], "kx": [0.0, 1.0]},
        name="first",
    )
    second = xr.DataArray(
        np.array([[3.0, 6.0], [4.0, 8.0]]),
        dims=("eV", "kx"),
        coords={"eV": [0.0, 1.0], "kx": [0.0, 1.0]},
        name="second",
    )
    operation = FigureOperationState.plot_slices(
        label="line_slices",
        sources=("first", "second"),
        axes=FigureAxesSelectionState(axes=((0, 0), (0, 1), (1, 0), (1, 1))),
        slice_dim="eV",
        slice_values=(0.0, 1.0),
    ).model_copy(update={"line_scales": (1.0, 10.0, 100.0, 1000.0)})
    tool = FigureComposerTool(
        first,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=2, ncols=2),
            sources=(
                FigureSourceState(name="first"),
                FigureSourceState(name="second"),
            ),
            operations=(operation,),
            primary_source="first",
        ),
        source_data={"first": first, "second": second},
    )
    qtbot.addWidget(tool)

    figurecomposer_rendering._render_into_figure(tool, Figure(), sync_visible=False)
    tool._document.replace_operation(0, operation.model_copy(update={"order": "F"}))

    warm_figure = Figure()
    figurecomposer_rendering._render_into_figure(
        tool,
        warm_figure,
        sync_visible=False,
    )
    warm_values = tuple(axis.lines[0].get_ydata() for axis in warm_figure.axes)

    tool._render_data_cache.invalidate(tool._document.source_revision)
    cold_figure = Figure()
    figurecomposer_rendering._render_into_figure(
        tool,
        cold_figure,
        sync_visible=False,
    )
    cold_values = tuple(axis.lines[0].get_ydata() for axis in cold_figure.axes)

    for warm, cold in zip(warm_values, cold_values, strict=True):
        np.testing.assert_allclose(warm, cold)
    for actual, expected in zip(
        warm_values,
        (
            [1.0, 2.0],
            [30.0, 60.0],
            [200.0, 400.0],
            [4000.0, 8000.0],
        ),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected)
