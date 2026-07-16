import types
import typing

import numpy as np
import xarray as xr

import erlab.interactive._stylesheets
from erlab.interactive._figurecomposer import (
    FigureAxesSelectionState,
    FigureDataSelectionState,
    FigureMethodFamily,
    FigureMethodPlotValueState,
    FigureOperationKind,
    FigureOperationState,
    FigureSubplotsState,
    _seeding,
)
from erlab.interactive.imagetool.manager._figurecomposer._controller import (
    _FigureComposerWorkflowController,
)


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
    internal = erlab.utils.array._make_dims_uniform(public)

    operation = _seeding._make_operations_for_sources(
        {"data": internal},
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

    assert (
        operations_from_targets(
            controller,
            ("slice_a", "array"),
            ("first", "second"),
        )
        is None
    )
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
