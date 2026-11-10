import types
import typing

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

import erlab.interactive._figurecomposer._rendering as figurecomposer_rendering
import erlab.interactive._stylesheets
from erlab.interactive._figurecomposer import (
    FigureAxesSelectionState,
    FigureComposerTool,
    FigureDataSelectionState,
    FigureMethodFamily,
    FigureMethodPlotValueState,
    FigureOperationKind,
    FigureOperationState,
    FigureRecipeState,
    FigureSourceState,
    FigureSubplotsState,
    _seeding,
)
from erlab.interactive._figurecomposer._trust import figure_code_trust_entries
from erlab.interactive.imagetool._figurecomposer_adapter import (
    build_figure_composer_operation,
)
from erlab.interactive.imagetool.manager._figurecomposer._controller import (
    _FigureComposerWorkflowController,
)
from tests.interactive.imagetool.manager.helpers import _exec_generated_code


@pytest.mark.parametrize(
    ("dtype", "multicursor"),
    [
        ("bool", False),
        ("bool", True),
        ("int16", False),
        ("float16", False),
        ("float32", False),
        ("float64", False),
    ],
)
def test_manager_figure_seed_preserves_display_dtype(qtbot, dtype, multicursor) -> None:
    if dtype == "bool":
        values = np.arange(100).reshape(10, 10) % 2 == 0
        if multicursor:
            values = np.stack(
                (
                    values,
                    np.zeros((10, 10), dtype=bool),
                    np.ones((10, 10), dtype=bool),
                ),
                axis=2,
            )
    else:
        values = (np.arange(100).reshape(10, 10) / 4).astype(dtype)
        values[-1, :] = -32768 if dtype == "int16" else 1000
    data = (
        xr.DataArray(
            values, dims=("y", "x", "slice"), coords={"slice": [0.0, 1.0, 2.0]}
        )
        if multicursor
        else xr.DataArray(values, dims=("y", "x"))
    )
    original = data.copy(deep=True)
    options = erlab.interactive.options.model.model_copy(deep=True)
    options.colors.max_rendered_abs_value = 75.0
    tool = erlab.interactive.itool(
        data, manager=False, execute=False, options_model=options
    )
    qtbot.addWidget(tool)
    area = tool.slicer_area
    if multicursor:
        area.add_cursor()
        area.add_cursor()
        for cursor in range(3):
            area.set_index(2, cursor, cursor=cursor)
    area.set_colormap(levels_robust=True)
    source = area.images[0]
    expected_dtype = np.dtype("float32" if dtype == "float32" else "float64")
    expected_arrays = []
    expected_levels = []
    for cursor, item in enumerate(source.slicer_data_items):
        selected = data.isel(slice=cursor) if multicursor else data
        expected = selected.astype(expected_dtype).transpose(
            source.axis_dims[1], source.axis_dims[0]
        )
        expected = expected.where(np.isfinite(expected) & (abs(expected) <= 75.0))
        expected_arrays.append(expected.values)
        expected_levels.append(item.getLevels())
        assert item.image.dtype == expected_dtype
        np.testing.assert_array_equal(item.image, expected.values)
    operation = build_figure_composer_operation(source, source_name="data")
    assert operation.kind == (
        FigureOperationKind.PLOT_SLICES
        if multicursor
        else FigureOperationKind.PLOT_ARRAY
    )
    operation = operation.model_copy(
        update={
            "axes": FigureAxesSelectionState(
                axes=tuple((0, i) for i in range(area.n_cursors))
            )
        }
    )
    composer = FigureComposerTool.from_sources(
        {"data": data},
        sources=(FigureSourceState(name="data"),),
        operations=(operation,),
        setup=FigureSubplotsState(nrows=1, ncols=area.n_cursors),
        primary_source="data",
    )
    qtbot.addWidget(composer)
    figure = Figure()
    figurecomposer_rendering._render_into_figure(composer, figure, sync_visible=False)
    assert not composer._operation_render_errors
    for axis, expected, levels in zip(
        figure.axes, expected_arrays, expected_levels, strict=True
    ):
        image = axis.images[0]
        assert image.get_array().dtype == expected_dtype
        np.testing.assert_array_equal(image.get_array().filled(np.nan), expected)
        np.testing.assert_array_equal(image.get_clim(), levels)
    namespace = {"data": data}
    exec(composer.generated_code(), namespace, namespace)  # noqa: S102
    try:
        for axis, expected, levels in zip(
            namespace["fig"].axes, expected_arrays, expected_levels, strict=True
        ):
            image = axis.images[0]
            assert image.get_array().dtype == expected_dtype
            np.testing.assert_array_equal(image.get_array().filled(np.nan), expected)
            np.testing.assert_array_equal(image.get_clim(), levels)
    finally:
        plt.close(namespace["fig"])
    xr.testing.assert_identical(data, original)


@pytest.mark.parametrize("multicursor", [False, True])
@pytest.mark.parametrize(
    "case",
    ["positive", "negative", "configured", "binned", "locked", "ordinary-masked"],
)
def test_manager_figure_seed_preserves_display_mask_after_reduction(
    qtbot, multicursor, case
) -> None:
    limit = 75.0 if case in {"configured", "locked"} else 1.0e30
    options = erlab.interactive.options.model.model_copy(deep=True)
    options.colors.max_rendered_abs_value = limit
    values = np.stack(
        [np.arange(100.0).reshape(10, 10) - 20 * index for index in range(4)],
        axis=2,
    )
    if case == "positive":
        values[-1, :, :] = 1.0e31
    elif case == "negative":
        values[-1, :, :] = -1.0e31
    elif case == "ordinary-masked":
        values[:] = 1.0e31
    elif case == "binned":
        # These values cancel before masking. Masking source data first loses them.
        values[0, :, 0::2] = -1.0e31
        values[0, :, 1::2] = 1.0e31
        # This average is unsafe, even though one input value is safe.
        values[1, :, 0::2] = 0.0
        values[1, :, 1::2] = 4.0e30
    data = xr.DataArray(
        values,
        dims=("y", "x", "slice"),
        coords={"y": np.arange(10.0), "x": np.arange(10.0), "slice": np.arange(4.0)},
    )
    original = data.copy(deep=True)
    tool = erlab.interactive.itool(
        data, manager=False, execute=False, options_model=options
    )
    qtbot.addWidget(tool)
    area = tool.slicer_area
    if multicursor:
        area.add_cursor()
    if case == "binned":
        area.set_bin_all(2, 2)
    for cursor in range(area.n_cursors):
        area.set_index(2, 2 * cursor + (case == "binned"), cursor=cursor)
    centered = case in {"negative", "configured", "locked"}
    locked = case == "locked"
    robust = case != "ordinary-masked"
    area.set_colormap(
        levels_robust=robust,
        zero_centered=centered,
        levels_locked=locked,
        levels=(-10.0, 60.0) if locked else None,
    )
    source = area.images[0]
    if multicursor and case == "positive":
        source.vb.setRange(xRange=(2.0, 5.0), yRange=(1.0, 7.0), padding=0.0)
    expected_arrays = []
    expected_levels = []
    for cursor, item in enumerate(source.slicer_data_items):
        selected = (
            data.isel(slice=slice(2 * cursor, 2 * cursor + 2)).mean("slice")
            if case == "binned"
            else data.isel(slice=2 * cursor)
        )
        selected = selected.transpose(source.axis_dims[1], source.axis_dims[0])
        expected = selected.where(np.isfinite(selected) & (abs(selected) <= limit))
        expected_arrays.append(expected.values)
        np.testing.assert_allclose(item.image, expected.values, equal_nan=True)
        levels = (
            (-10.0, 60.0)
            if locked
            else (0.0, 255.0)
            if case == "ordinary-masked"
            else np.nanpercentile(expected, [2, 98])
        )
        expected_levels.append(levels)
        np.testing.assert_allclose(item.getLevels(), levels)
    if case == "binned":
        assert np.isfinite(expected_arrays[0]).any()
        assert np.count_nonzero(expected_arrays[0] == 0.0) >= 10
        assert np.count_nonzero(np.isnan(expected_arrays[0])) == 10

    operation = build_figure_composer_operation(source, source_name="data")
    assert operation.kind == (
        FigureOperationKind.PLOT_SLICES
        if multicursor
        else FigureOperationKind.PLOT_ARRAY
    )
    assert operation.robust is (robust and not locked)
    assert operation.image_value_abs_limit == limit
    assert not operation.crop
    if multicursor and case == "positive":
        assert operation.xlim is not None
        assert operation.ylim is not None
    operation = operation.model_copy(
        update={
            "axes": FigureAxesSelectionState(
                axes=tuple((0, i) for i in range(area.n_cursors))
            )
        }
    )
    recipe = FigureRecipeState(
        sources=(FigureSourceState(name="data"),),
        operations=(operation,),
        setup=FigureSubplotsState(nrows=1, ncols=area.n_cursors),
        primary_source="data",
    )
    restored = FigureRecipeState.model_validate_json(recipe.model_dump_json())
    assert restored == recipe
    assert not figure_code_trust_entries(restored, location_prefix="test")
    composer = FigureComposerTool.from_sources(
        {"data": area.displayed_data},
        sources=restored.sources,
        operations=restored.operations,
        setup=restored.setup,
        primary_source="data",
    )
    qtbot.addWidget(composer)
    figure = Figure()
    canvas = FigureCanvasAgg(figure)
    figurecomposer_rendering._render_into_figure(composer, figure, sync_visible=False)
    canvas.draw()
    assert not composer._operation_render_errors
    for axis, expected, levels in zip(
        figure.axes, expected_arrays, expected_levels, strict=True
    ):
        image = axis.images[0]
        np.testing.assert_allclose(
            image.get_array().filled(np.nan), expected, equal_nan=True
        )
        np.testing.assert_allclose(image.get_clim(), levels)
        if multicursor and case == "positive":
            np.testing.assert_allclose(axis.get_xlim(), operation.xlim)
            np.testing.assert_allclose(axis.get_ylim(), operation.ylim)
    code = composer.generated_code()
    assert code.count("import numpy as np") == 1
    namespace = {"data": data}
    exec(code, namespace, namespace)  # noqa: S102
    try:
        for axis, expected, levels in zip(
            namespace["fig"].axes, expected_arrays, expected_levels, strict=True
        ):
            image = axis.images[0]
            np.testing.assert_allclose(
                image.get_array().filled(np.nan), expected, equal_nan=True
            )
            np.testing.assert_allclose(image.get_clim(), levels)
    finally:
        plt.close(namespace["fig"])
    if case == "positive":
        changed = data * 2.0
        namespace = {"data": changed}
        exec(code, namespace, namespace)  # noqa: S102
        try:
            for axis, levels in zip(
                namespace["fig"].axes, expected_levels, strict=True
            ):
                np.testing.assert_allclose(
                    axis.images[0].get_clim(), np.asarray(levels) * 2.0
                )
        finally:
            plt.close(namespace["fig"])
    xr.testing.assert_identical(data, original)


@pytest.mark.parametrize("robust", [False, True])
@pytest.mark.parametrize("locked", [False, True])
@pytest.mark.parametrize(
    ("high_contrast", "zero_centered", "expected_mapping"),
    [
        (False, False, np.sqrt([0.0, 0.25, 0.5, 0.75, 1.0])),
        (True, False, [0.0, 0.4375, 0.75, 0.9375, 1.0]),
        (
            False,
            True,
            [0.0, 0.5 - np.sqrt(0.5) / 2, 0.5, 0.5 + np.sqrt(0.5) / 2, 1.0],
        ),
        (True, True, [0.0, 0.125, 0.5, 0.875, 1.0]),
    ],
)
def test_manager_figure_image_seed_preserves_automatic_and_locked_levels(
    qtbot, robust, locked, high_contrast, zero_centered, expected_mapping
) -> None:
    values = np.arange(-20.0, 80.0).reshape(10, 10)
    values[-1, -1] = 1.0e6
    data = xr.DataArray(values, dims=("y", "x"))
    tool = erlab.interactive.itool(data, manager=False, execute=False)
    qtbot.addWidget(tool)
    area = tool.slicer_area
    area.set_colormap(
        cmap="coolwarm",
        gamma=0.5,
        high_contrast=high_contrast,
        zero_centered=zero_centered,
        levels_robust=robust,
        levels_locked=locked,
        levels=(-30.0, 70.0) if locked else None,
    )
    source = area.images[0]
    visible_levels = source.slicer_data_items[0].getLevels()
    assert visible_levels is not None
    expected_levels = (
        (-30.0, 70.0)
        if locked
        else np.percentile(values, [2, 98])
        if robust
        else (values.min(), values.max())
    )
    np.testing.assert_allclose(visible_levels, expected_levels)
    operation = build_figure_composer_operation(source, source_name="data")
    assert operation.kind == FigureOperationKind.PLOT_ARRAY
    assert operation.robust is (robust and not locked)
    assert "robust" not in operation.extra_kwargs
    if zero_centered:
        assert operation.vcenter == pytest.approx(np.mean(visible_levels))
        if not locked:
            assert operation.halfrange is None

    composer = FigureComposerTool.from_sources(
        {"data": area.displayed_data},
        sources=(FigureSourceState(name="data"),),
        operations=(operation,),
        primary_source="data",
    )
    qtbot.addWidget(composer)
    figure = Figure()
    canvas = FigureCanvasAgg(figure)
    figurecomposer_rendering._render_into_figure(composer, figure, sync_visible=False)
    canvas.draw()
    assert not composer._operation_render_errors
    image = figure.axes[0].images[0]
    sample_values = np.linspace(*visible_levels, 5)
    np.testing.assert_allclose(image.get_clim(), visible_levels)
    np.testing.assert_allclose(image.norm(sample_values), expected_mapping)

    namespace = _exec_generated_code(composer.generated_code(), {"data": data})
    try:
        copied_image = namespace["fig"].axes[0].images[0]
        np.testing.assert_allclose(copied_image.get_clim(), visible_levels)
        np.testing.assert_allclose(copied_image.norm(sample_values), expected_mapping)
    finally:
        plt.close(namespace["fig"])


@pytest.mark.parametrize("zero_centered", [False, True])
@pytest.mark.parametrize("case", ["constant", "nonfinite", "equal-percentiles"])
def test_manager_figure_image_seed_preserves_robust_fallback_levels(
    qtbot, zero_centered, case
) -> None:
    values = np.full((10, 10), np.nan if case == "nonfinite" else 5.0)
    if case == "equal-percentiles":
        values[-1, -1] = 1.0e6
    data = xr.DataArray(values, dims=("y", "x"))
    tool = erlab.interactive.itool(data, manager=False, execute=False)
    qtbot.addWidget(tool)
    area = tool.slicer_area
    area.set_colormap(levels_robust=True, zero_centered=zero_centered)
    source = area.images[0]
    visible_levels = source.slicer_data_items[0].getLevels()
    np.testing.assert_allclose(visible_levels, (0.0, 255.0))
    operation = build_figure_composer_operation(source, source_name="data")
    assert operation.robust
    composer = FigureComposerTool.from_sources(
        {"data": area.displayed_data},
        sources=(FigureSourceState(name="data"),),
        operations=(operation,),
        primary_source="data",
    )
    qtbot.addWidget(composer)
    figure = Figure()
    canvas = FigureCanvasAgg(figure)
    figurecomposer_rendering._render_into_figure(composer, figure, sync_visible=False)
    canvas.draw()
    assert not composer._operation_render_errors
    np.testing.assert_allclose(figure.axes[0].images[0].get_clim(), visible_levels)
    namespace = _exec_generated_code(composer.generated_code(), {"data": data})
    try:
        np.testing.assert_allclose(
            namespace["fig"].axes[0].images[0].get_clim(), visible_levels
        )
    finally:
        plt.close(namespace["fig"])


@pytest.mark.parametrize("robust", [False, True])
@pytest.mark.parametrize("locked", [False, True])
@pytest.mark.parametrize(
    ("high_contrast", "zero_centered", "expected_mapping"),
    [
        (False, False, np.sqrt([0.0, 0.25, 0.5, 0.75, 1.0])),
        (True, False, [0.0, 0.4375, 0.75, 0.9375, 1.0]),
        (
            False,
            True,
            [0.0, 0.5 - np.sqrt(0.5) / 2, 0.5, 0.5 + np.sqrt(0.5) / 2, 1.0],
        ),
        (True, True, [0.0, 0.125, 0.5, 0.875, 1.0]),
    ],
)
def test_manager_figure_multicursor_seed_preserves_automatic_and_locked_levels(
    qtbot, robust, locked, high_contrast, zero_centered, expected_mapping
) -> None:
    values = np.stack(
        (
            np.arange(-20.0, 80.0).reshape(10, 10),
            np.arange(-120.0, -20.0).reshape(10, 10),
        ),
        axis=2,
    )
    values[-1, -1, 0] = 1.0e6
    values[0, 0, 1] = -2.0e6
    data = xr.DataArray(
        values,
        dims=("y", "x", "slice"),
        coords={"slice": [0.0, 1.0]},
    )
    tool = erlab.interactive.itool(data, manager=False, execute=False)
    qtbot.addWidget(tool)
    area = tool.slicer_area
    area.add_cursor()
    area.set_index(2, 0, cursor=0)
    area.set_index(2, 1, cursor=1)
    area.set_colormap(
        cmap="coolwarm",
        gamma=0.5,
        high_contrast=high_contrast,
        zero_centered=zero_centered,
        levels_robust=robust,
        levels_locked=locked,
        levels=(-30.0, 70.0) if locked else None,
    )
    source = area.images[0]
    expected_levels = (
        [(-30.0, 70.0)] * 2
        if locked
        else [np.percentile(values[:, :, cursor], [2, 98]) for cursor in range(2)]
        if robust
        else [
            (values[:, :, cursor].min(), values[:, :, cursor].max())
            for cursor in range(2)
        ]
    )
    for item, levels in zip(source.slicer_data_items, expected_levels, strict=True):
        np.testing.assert_allclose(item.getLevels(), levels)
    operation = build_figure_composer_operation(source, source_name="data")
    assert operation.kind == FigureOperationKind.PLOT_SLICES
    assert operation.slice_dim == "slice"
    assert operation.slice_values == (0.0, 1.0)
    assert operation.robust is (robust and not locked)
    assert "robust" not in operation.extra_kwargs
    assert operation.same_limits is locked
    if zero_centered and not locked:
        assert operation.panel_styles_enabled
        assert [style.vcenter for style in operation.panel_styles] == pytest.approx(
            [np.mean(levels) for levels in expected_levels]
        )
        assert all(style.halfrange is None for style in operation.panel_styles)

    operation = operation.model_copy(
        update={"axes": FigureAxesSelectionState(axes=((0, 0), (0, 1)))}
    )
    composer = FigureComposerTool.from_sources(
        {"data": area.displayed_data},
        sources=(FigureSourceState(name="data"),),
        operations=(operation,),
        setup=FigureSubplotsState(nrows=1, ncols=2),
        primary_source="data",
    )
    qtbot.addWidget(composer)
    figure = Figure()
    canvas = FigureCanvasAgg(figure)
    figurecomposer_rendering._render_into_figure(composer, figure, sync_visible=False)
    canvas.draw()
    assert not composer._operation_render_errors
    assert len(figure.axes) == 2
    for axis, levels in zip(figure.axes, expected_levels, strict=True):
        image = axis.images[0]
        np.testing.assert_allclose(image.get_clim(), levels)
        np.testing.assert_allclose(
            image.norm(np.linspace(*levels, 5)), expected_mapping
        )

    namespace = _exec_generated_code(composer.generated_code(), {"data": data})
    try:
        assert len(namespace["fig"].axes) == 2
        for axis, levels in zip(namespace["fig"].axes, expected_levels, strict=True):
            image = axis.images[0]
            np.testing.assert_allclose(image.get_clim(), levels)
            np.testing.assert_allclose(
                image.norm(np.linspace(*levels, 5)), expected_mapping
            )
    finally:
        plt.close(namespace["fig"])


@pytest.mark.parametrize("zero_centered", [False, True])
@pytest.mark.parametrize("singleton", [False, True])
@pytest.mark.parametrize("case", ["constant", "nonfinite", "equal-percentiles"])
def test_manager_figure_multicursor_seed_preserves_robust_fallback_levels(
    qtbot, zero_centered, singleton, case
) -> None:
    values = np.stack(
        (
            np.full((10, 10), np.nan if case == "nonfinite" else 5.0),
            np.arange(-20.0, 80.0).reshape(10, 10),
        ),
        axis=2,
    )
    if case == "equal-percentiles":
        values[-1, -1, 0] = 1.0e6
    values[-1, -1, 1] = 2.0e6
    data = xr.DataArray(
        values,
        dims=("y", "x", "slice"),
        coords={"slice": [0.0, 1.0]},
    )
    if singleton:
        data = data.expand_dims(singleton=[7.0], other_singleton=[11.0])
    tool = erlab.interactive.itool(data, manager=False, execute=False)
    qtbot.addWidget(tool)
    area = tool.slicer_area
    area.add_cursor()
    area.set_index(2, 0, cursor=0)
    area.set_index(2, 1, cursor=1)
    area.set_colormap(levels_robust=True, zero_centered=zero_centered)
    source = area.images[0]
    expected_levels = [(0.0, 255.0), np.percentile(values[:, :, 1], [2, 98])]
    for item, levels in zip(source.slicer_data_items, expected_levels, strict=True):
        np.testing.assert_allclose(item.getLevels(), levels)
    operation = build_figure_composer_operation(source, source_name="data")
    assert operation.kind == FigureOperationKind.PLOT_SLICES
    assert operation.robust
    assert "robust" not in operation.extra_kwargs
    assert operation.panel_styles_enabled
    fallback_style = operation.panel_styles[0]
    assert (fallback_style.map_index, fallback_style.slice_index) == (0, 0)
    if zero_centered:
        assert fallback_style.vcenter == 127.5
        assert fallback_style.halfrange == 127.5
        assert operation.panel_styles[1].vcenter == pytest.approx(
            np.mean(expected_levels[1])
        )
        assert operation.panel_styles[1].halfrange is None
        assert (
            operation.panel_styles[1].map_index,
            operation.panel_styles[1].slice_index,
        ) == ((1, 0) if singleton else (0, 1))
    else:
        assert (fallback_style.vmin, fallback_style.vmax) == (0.0, 255.0)

    operation = operation.model_copy(
        update={"axes": FigureAxesSelectionState(axes=((0, 0), (0, 1)))}
    )
    composer = FigureComposerTool.from_sources(
        {"data": area.displayed_data},
        sources=(FigureSourceState(name="data"),),
        operations=(operation,),
        setup=FigureSubplotsState(nrows=1, ncols=2),
        primary_source="data",
    )
    qtbot.addWidget(composer)
    figure = Figure()
    canvas = FigureCanvasAgg(figure)
    figurecomposer_rendering._render_into_figure(composer, figure, sync_visible=False)
    canvas.draw()
    assert not composer._operation_render_errors
    assert len(figure.axes) == 2
    for axis, levels in zip(figure.axes, expected_levels, strict=True):
        np.testing.assert_allclose(axis.images[0].get_clim(), levels)

    namespace = _exec_generated_code(composer.generated_code(), {"data": data})
    try:
        assert len(namespace["fig"].axes) == 2
        for axis, levels in zip(namespace["fig"].axes, expected_levels, strict=True):
            np.testing.assert_allclose(axis.images[0].get_clim(), levels)
    finally:
        plt.close(namespace["fig"])


def test_manager_figure_operation_source_name_mapping_updates_all_fields() -> None:
    method_x = FigureMethodPlotValueState(source="old", kind="data")
    method_y = FigureMethodPlotValueState(source="other", kind="data")
    operation = FigureOperationState.method(
        family=FigureMethodFamily.AXES,
        name="plot",
    ).model_copy(
        update={
            "sources": ("old", "other"),
            "map_selections": (
                FigureDataSelectionState(source="old", qsel={"x": 0.0}),
            ),
            "line_source": "old",
            "hv_overlay_source": "old",
            "method_plot_x": method_x,
            "method_plot_y": method_y,
        }
    )

    assert _seeding._operation_with_source_names(operation, {}) is operation
    renamed = _seeding._operation_with_source_names(operation, {"old": "new"})

    assert renamed.sources == ("new", "other")
    assert renamed.map_selections == (
        FigureDataSelectionState(source="new", qsel={"x": 0.0}),
    )
    assert renamed.line_source == "new"
    assert renamed.hv_overlay_source == "new"
    assert renamed.method_plot_x == method_x.model_copy(update={"source": "new"})
    assert renamed.method_plot_y is method_y


def test_manager_default_figure_seed_uses_public_nonuniform_dims() -> None:
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

    operation = _seeding._make_operations_for_sources(
        {"data": public},
        setup=FigureSubplotsState(),
    )[0]

    assert operation.kind == FigureOperationKind.PLOT_SLICES
    assert operation.slice_dim == "sample_temp"
    assert "sample_temp_idx" not in operation.model_dump_json()


def test_manager_default_figure_seed_keeps_mixed_higher_dimensional_sources() -> None:
    profile = xr.DataArray(np.arange(4.0), dims=("eV",), name="profile")
    image_stack = xr.DataArray(
        np.arange(24.0).reshape(3, 2, 4),
        dims=("sample_temp", "alpha", "eV"),
        coords={
            "sample_temp": [10.0, 15.0, 30.0],
            "alpha": [0.0, 1.0],
            "eV": [-0.1, 0.0, 0.1, 0.2],
        },
        name="map",
    )

    line_operation, map_operation = _seeding._make_operations_for_sources(
        {"profile": profile, "map": image_stack},
        setup=FigureSubplotsState(nrows=2, ncols=1),
    )

    assert line_operation.kind == FigureOperationKind.LINE
    assert line_operation.line_source == "profile"
    assert line_operation.axes.axes == ((0, 0),)
    assert map_operation.kind == FigureOperationKind.PLOT_SLICES
    assert map_operation.sources == ("map",)
    assert map_operation.axes.axes == ((1, 0),)
    assert map_operation.slice_dim == "sample_temp"
    assert map_operation.slice_values == (15.0,)


def test_manager_plot_slices_setup_honors_order_for_horizontal_seeding() -> None:
    operation = FigureOperationState.plot_slices(
        label="plot_slices",
        sources=("data",),
        slice_dim="eV",
        slice_values=(0.0, 1.0, 2.0),
    )
    assert _seeding._plot_slices_grid_shape(operation) == (1, 3)

    multi_source_operation = FigureOperationState.plot_slices(
        label="plot_slices",
        sources=("data_0", "data_1", "data_2"),
        slice_dim="eV",
        slice_values=(0.0,),
    ).model_copy(update={"order": "F"})
    assert _seeding._plot_slices_grid_shape(multi_source_operation) == (1, 3)


def test_manager_figure_operation_helpers_cover_multi_image_edges() -> None:
    first_image = xr.DataArray(np.arange(4.0).reshape(2, 2), dims=("y", "x"))
    second_image = xr.DataArray(np.arange(4.0, 8.0).reshape(2, 2), dims=("y", "x"))

    operations = _seeding._make_operations_for_sources(
        {"first": first_image, "second": second_image},
        setup=FigureSubplotsState(nrows=2, ncols=1),
    )

    assert [operation.kind for operation in operations] == [
        FigureOperationKind.PLOT_ARRAY,
        FigureOperationKind.PLOT_ARRAY,
    ]
    assert [operation.axes.axes for operation in operations] == [
        ((0, 0),),
        ((1, 0),),
    ]

    split_by_axes_id = _seeding._operations_with_append_axes(
        typing.cast("tuple[typing.Any, ...]", operations),
        FigureAxesSelectionState(axes_ids=("left", "right")),
    )
    assert [operation.axes.axes_ids for operation in split_by_axes_id] == [
        ("left",),
        ("right",),
    ]


def test_manager_figure_image_target_helpers_cover_plot_slices_edges(
    monkeypatch,
) -> None:
    plot_slices = FigureOperationState.plot_slices(
        label="slice",
        sources=("old",),
        slice_dim="eV",
        slice_values=(0.0,),
    )
    plot_array = FigureOperationState.plot_array(label="array", source="old")

    class _FakePlot:
        is_image = True

        def __init__(self, operation: FigureOperationState) -> None:
            self.operation = operation

    nodes = {
        "slice_a": types.SimpleNamespace(
            imagetool=types.SimpleNamespace(
                slicer_area=types.SimpleNamespace(axes=(_FakePlot(plot_slices),))
            )
        ),
        "slice_b": types.SimpleNamespace(
            imagetool=types.SimpleNamespace(
                slicer_area=types.SimpleNamespace(axes=(_FakePlot(plot_slices),))
            )
        ),
        "array": types.SimpleNamespace(
            imagetool=types.SimpleNamespace(
                slicer_area=types.SimpleNamespace(axes=(_FakePlot(plot_array),))
            )
        ),
    }
    manager = types.SimpleNamespace(
        _node_for_target=lambda target: nodes[target],
        get_imagetool=lambda target: nodes[target].imagetool,
    )
    controller = typing.cast(
        "_FigureComposerWorkflowController",
        types.SimpleNamespace(_host=manager),
    )
    monkeypatch.setattr(
        "erlab.interactive.imagetool._figurecomposer_adapter.build_figure_composer_operation",
        lambda plot, *, source_name: plot.operation.model_copy(
            update={"sources": (source_name,)}
        ),
    )
    operations_from_targets = (
        _FigureComposerWorkflowController._figure_operations_from_image_targets
    )

    mixed = operations_from_targets(
        controller,
        ("slice_a", "array"),
        ("first", "second"),
    )
    assert mixed is not None
    assert [operation.kind for operation in mixed] == [
        FigureOperationKind.PLOT_SLICES,
        FigureOperationKind.PLOT_ARRAY,
    ]
    combined = operations_from_targets(
        controller,
        ("slice_a", "slice_b"),
        ("first", "second"),
    )

    assert combined is not None
    assert len(combined) == 1
    assert combined[0].kind == FigureOperationKind.PLOT_SLICES
    assert combined[0].sources == ("first", "second")
    assert combined[0].order == "F"


@pytest.mark.parametrize("equal_aspect", [False, True])
def test_manager_combined_image_steps_preserve_direction_and_aspect(
    qtbot, equal_aspect
) -> None:
    data = xr.DataArray(
        np.arange(27.0).reshape(3, 3, 3),
        dims=("alpha", "eV", "beta"),
        coords={dim: np.arange(3.0) for dim in ("alpha", "eV", "beta")},
    )
    tools = [
        erlab.interactive.itool(data, manager=False, execute=False) for _ in range(2)
    ]
    for tool in tools:
        qtbot.addWidget(tool)
        viewbox = tool.slicer_area.main_image.getViewBox()
        viewbox.invertX(True)
        viewbox.setAspectLocked(equal_aspect)
    controller = typing.cast(
        "_FigureComposerWorkflowController",
        types.SimpleNamespace(
            _host=types.SimpleNamespace(get_imagetool=lambda target: tools[target])
        ),
    )

    operations = (
        _FigureComposerWorkflowController._figure_operations_from_image_targets(
            controller, (0, 1), ("first", "second")
        )
    )

    assert operations is not None
    assert len(operations) == 1
    operation = operations[0]
    assert operation.kind == FigureOperationKind.PLOT_SLICES
    assert operation.invert_x is True
    assert operation.invert_y is False
    assert operation.axis == ("image" if equal_aspect else "auto")
    assert "aspect" not in operation.extra_kwargs
    for tool in tools:
        tool.close()


def test_manager_bz_overlay_ignores_converted_output_without_ktool_parent() -> None:
    class FakeNode:
        output_id = "ktool.converted_output"

    class FakeParent:
        tool_window = object()

    class FakeManager:
        def _node_for_target(self, target: int | str) -> FakeNode:
            assert target == "converted"
            return FakeNode()

        def _parent_node(self, node: FakeNode) -> FakeParent:
            assert isinstance(node, FakeNode)
            return FakeParent()

    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [-1.0, 1.0], "ky": [-2.0, 2.0]},
        name="momentum",
    )
    axes = FigureAxesSelectionState(axes=((0, 0),))
    controller = typing.cast(
        "_FigureComposerWorkflowController",
        types.SimpleNamespace(_host=FakeManager()),
    )

    assert (
        _FigureComposerWorkflowController._figure_bz_overlay_operation_from_targets(
            controller,
            ("first", "second"),
            {"momentum": data},
            axes=axes,
        )
        is None
    )
    assert (
        _FigureComposerWorkflowController._figure_bz_overlay_operation_from_targets(
            controller,
            ("converted",),
            {"first": data, "second": data},
            axes=axes,
        )
        is None
    )

    assert (
        _FigureComposerWorkflowController._figure_bz_overlay_operation_from_target(
            controller,
            "converted",
            data,
            axes=axes,
        )
        is None
    )


def _all_image_style_values() -> dict[str, typing.Any]:
    from erlab.interactive._figurecomposer import FigureImageStyleState

    values = {
        "norm_name": "CenteredInversePowerNorm",
        "norm_gamma": 0.75,
        "norm_clip": True,
        "norm_kwargs": {"clip": True},
        "vmin": -3.0,
        "vmax": 7.0,
        "vcenter": 2.0,
        "halfrange": 5.0,
        "cmap": "magma",
        "transpose": True,
        "invert_x": True,
        "invert_y": True,
        "xlim": (-1.0, 1.0),
        "ylim": (-2.0, 2.0),
        "crop": False,
        "aspect": 2.0,
        "same_limits": "all",
        "axis": "off",
        "show_all_labels": True,
        "colorbar": "right",
        "hide_colorbar_ticks": False,
        "annotate": False,
        "robust": True,
        "image_value_abs_limit": 100.0,
        "gamma": 0.5,
        "annotate_kw": {"color": "red"},
        "colorbar_kw": {"pad": 0.1},
        "extra_kwargs": {"alpha": 0.7},
    }
    assert set(values) == set(FigureImageStyleState.model_fields)
    defaults = FigureImageStyleState()
    assert all(value != getattr(defaults, name) for name, value in values.items())
    return values


def _combined_image_operation_states(monkeypatch, operations):
    controller = typing.cast(
        "_FigureComposerWorkflowController",
        types.SimpleNamespace(
            _host=types.SimpleNamespace(
                get_imagetool=lambda index: types.SimpleNamespace(
                    slicer_area=types.SimpleNamespace(
                        axes=(
                            types.SimpleNamespace(
                                is_image=True, operation=operations[index]
                            ),
                        )
                    )
                )
            )
        ),
    )
    monkeypatch.setattr(
        "erlab.interactive.imagetool._figurecomposer_adapter.build_figure_composer_operation",
        lambda plot, *, source_name: plot.operation,
    )
    return _FigureComposerWorkflowController._figure_operations_from_image_targets(
        controller,
        tuple(range(len(operations))),
        tuple(operation.sources[0] for operation in operations),
    )


def test_manager_combined_image_style_round_trip_is_field_complete(monkeypatch) -> None:
    from erlab.interactive._figurecomposer import FigureImageStyleState

    style = FigureImageStyleState(**_all_image_style_values())
    sources = tuple(
        FigureOperationState.plot_array(
            label=name,
            source=name,
            map_selections=(FigureDataSelectionState(source=name, isel={"z": 1}),),
        ).with_image_style(style)
        for name in ("first", "second")
    )
    operations = _combined_image_operation_states(monkeypatch, sources)
    assert operations is not None
    assert len(operations) == 1
    assert operations[0].kind == FigureOperationKind.PLOT_SLICES
    assert operations[0].image_style == style
    flat = operations[0].model_dump(mode="json")
    assert "image_style" not in flat
    assert set(FigureImageStyleState.model_fields).issubset(flat)
    assert FigureOperationState.model_validate(flat).image_style == style
    recipe = FigureRecipeState(operations=operations)
    assert FigureRecipeState.model_validate_json(recipe.model_dump_json()) == recipe
    # Existing recipes keep all style fields at the operation level.
    legacy = {
        "kind": "plot_array",
        "label": "legacy",
        "sources": ["first"],
        **style.model_dump(mode="json"),
    }
    assert FigureOperationState.model_validate(legacy).image_style == style
    sources[0].norm_kwargs["source_only"] = True
    assert "source_only" not in operations[0].norm_kwargs
    assert "source_only" not in sources[1].norm_kwargs


def test_manager_combined_image_styles_compare_every_operation_field(
    monkeypatch,
) -> None:
    from erlab.interactive._figurecomposer import (
        FigureImageColorStyleState,
        FigureImageStyleState,
    )

    first = FigureOperationState.plot_array(
        label="first",
        source="first",
        map_selections=(FigureDataSelectionState(source="first", isel={"z": 1}),),
    ).with_image_style(FigureImageStyleState())
    for name, value in _all_image_style_values().items():
        if name in FigureImageColorStyleState.model_fields or name == "gamma":
            continue
        second = FigureOperationState.plot_array(
            label="second",
            source="second",
            map_selections=(FigureDataSelectionState(source="second", isel={"z": 1}),),
        ).with_image_style(FigureImageStyleState(**{name: value}))
        operations = _combined_image_operation_states(monkeypatch, (first, second))
        assert operations == (first, second), name


@pytest.mark.parametrize(
    ("multicursor", "difference"),
    [
        (multicursor, difference)
        for multicursor in (False, True)
        for difference in ("shared", "robust", "mask", "norm", "locked", "view")
    ]
    + [(False, "cursors")],
)
def test_manager_new_figure_preserves_each_image_style(
    qtbot, manager_context, multicursor, difference
) -> None:
    from ._common import _select_operation_rows

    arrays = {}
    original_arrays = {}
    expected = {}
    image_tools = {}
    image_buffers = {}
    with manager_context() as manager:
        for source_index, name in enumerate(("first", "second")):
            values = np.arange(-150.0, 150.0).reshape(10, 10, 3)
            values[0, 0, :] = 1.0e31
            values[-1, -1, :] = 1.0e6
            data = xr.DataArray(values + source_index, dims=("y", "x", "z"), name=name)
            arrays[name] = data
            original_arrays[name] = data.copy(deep=True)
            options = erlab.interactive.options.model.model_copy(deep=True)
            options.colors.max_rendered_abs_value = (
                50.0 if difference == "mask" and source_index else 200.0
            )
            image_tool = erlab.interactive.imagetool.ImageTool(
                data, _in_manager=True, options_model=options
            )
            manager.add_imagetool(image_tool, show=False)
            area = image_tool.slicer_area
            if multicursor or (difference == "cursors" and source_index == 1):
                area.add_cursor()
                area.set_index(2, 0, cursor=0)
                area.set_index(2, 2, cursor=1)
            area.set_colormap(
                "magma" if source_index else "viridis",
                levels_robust=not (difference == "robust" and source_index),
                zero_centered=difference == "norm" and source_index == 1,
                high_contrast=difference == "norm" and source_index == 1,
                levels_locked=difference == "locked" and source_index == 1,
                levels=(-20.0, 30.0)
                if difference == "locked" and source_index
                else None,
            )
            if difference == "view" and source_index:
                area.main_image.getViewBox().invertX(True)
            image_tools[name] = image_tool
            image_buffers[name] = tuple(
                item.image for item in area.images[0].slicer_data_items
            )
        figure_uid = manager.create_figure_from_targets((0, 1), show=False)
        assert figure_uid is not None
        composer = manager._child_node(figure_uid).tool_window
        assert isinstance(composer, FigureComposerTool)
        for name, image_tool in image_tools.items():
            items = image_tool.slicer_area.images[0].slicer_data_items
            for item, buffer in zip(items, image_buffers[name], strict=True):
                assert item.image is buffer
                item.refresh_data()
            expected[name] = tuple(
                (item.image.copy(), item.getLevels()) for item in items
            )
        operations = composer.tool_status.operations
        expected_combined = difference in {"shared", "norm"}
        assert len(operations) == (1 if expected_combined else 2)
        assert composer.tool_status.setup.nrows == (
            2 if multicursor or difference == "cursors" else 1
        )
        assert composer.tool_status.setup.ncols == 2
        saved = composer.tool_status.model_dump_json()
        restored = FigureRecipeState.model_validate_json(saved)
        assert restored == composer.tool_status
        _select_operation_rows(composer, tuple(range(len(operations))))
        for section in ("colors", "view", "sources"):
            composer.operation_editor.select_section(section)
        assert composer.tool_status.model_dump_json() == saved
        figure = Figure()
        figurecomposer_rendering._render_into_figure(
            composer, figure, sync_visible=False
        )
        assert not composer._operation_render_errors
        namespace = _exec_generated_code(composer.generated_code(), arrays)
        try:
            for rendered in (figure, namespace["fig"]):
                assert len(rendered.axes) == (
                    4 if multicursor or difference == "cursors" else 2
                )
                for source_index, name in enumerate(arrays):
                    for cursor, (values, levels) in enumerate(expected[name]):
                        axis = rendered.axes[cursor * 2 + source_index]
                        [image] = axis.images
                        assert bool(axis.get_xlabel()) is (
                            cursor == len(expected[name]) - 1
                        )
                        assert bool(axis.get_ylabel()) is (
                            not expected_combined or source_index == 0
                        )
                        np.testing.assert_array_equal(
                            image.get_array().filled(np.nan), values
                        )
                        np.testing.assert_allclose(
                            image.get_clim(), levels, rtol=0.0, atol=1.0e-12
                        )
                        assert bool(axis.xaxis_inverted()) is (
                            difference == "view" and source_index == 1
                        )
            for name, original in original_arrays.items():
                xr.testing.assert_identical(arrays[name], original)
                xr.testing.assert_identical(composer.source_data()[name], original)
        finally:
            plt.close(namespace["fig"])


def test_manager_combined_panel_color_style_is_field_complete(monkeypatch) -> None:
    from erlab.interactive._figurecomposer import FigureImageColorStyleState

    first = FigureOperationState.plot_array(
        label="first",
        source="first",
        map_selections=(FigureDataSelectionState(source="first", isel={"z": 1}),),
    )
    values = _all_image_style_values()
    color_values = {
        name: values[name] for name in FigureImageColorStyleState.model_fields
    }
    second = FigureOperationState.plot_array(
        label="second",
        source="second",
        map_selections=(FigureDataSelectionState(source="second", isel={"z": 1}),),
    ).with_image_style(first.image_style.model_copy(update=color_values))
    operations = _combined_image_operation_states(monkeypatch, (first, second))
    assert operations is not None
    assert len(operations) == 1
    [combined] = operations
    assert combined.panel_styles_enabled
    panel_style = next(style for style in combined.panel_styles if style.map_index == 1)
    assert {
        name: getattr(panel_style, name)
        for name in FigureImageColorStyleState.model_fields
    } == color_values
    restored = FigureOperationState.model_validate_json(combined.model_dump_json())
    assert restored == combined
    assert all(isinstance(style, type(panel_style)) for style in restored.panel_styles)


def test_manager_separate_image_steps_keep_each_source_column() -> None:
    operations = tuple(
        FigureOperationState.plot_slices(
            label=name, sources=(name,), slice_dim="z", slice_values=values
        )
        for name, values in (("first", (0.0, 1.0)), ("second", (0.0, 1.0, 2.0)))
    )
    setup = _seeding._setup_for_image_operations(operations, {})
    assert (setup.nrows, setup.ncols) == (3, 2)
    placed = _seeding._operations_with_append_axes(
        operations,
        FigureAxesSelectionState(
            axes=tuple((row, col) for row in range(3) for col in range(2))
        ),
    )
    assert all(operation.order == "F" for operation in placed)
    assert all(
        operation.cmap_order == "C" and operation.norm_order is None
        for operation in placed
    )
    assert placed[0].axes.axes == ((0, 0), (1, 0))
    assert placed[1].axes.axes == ((0, 1), (1, 1), (2, 1))
