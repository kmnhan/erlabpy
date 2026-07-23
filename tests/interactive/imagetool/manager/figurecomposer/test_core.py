from __future__ import annotations

import ast
import typing
from pathlib import Path

if typing.TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np
import pytest
import xarray as xr
from qtpy import QtGui, QtWidgets

import erlab.interactive._figurecomposer._cache as figurecomposer_cache
import erlab.interactive._figurecomposer._model._gridspec as figurecomposer_gridspec
import erlab.interactive._figurecomposer._rendering as figurecomposer_rendering
import erlab.interactive._figurecomposer._text as figurecomposer_text
import erlab.interactive._figurecomposer._tool as figurecomposer_tool_module
import erlab.interactive._stylesheets
from erlab.interactive._figurecomposer import (
    FigureAxesSelectionState,
    FigureComposerTool,
    FigureDataSelectionState,
    FigureGridSpecAxesState,
    FigureGridSpecGridState,
    FigureGridSpecLayoutState,
    FigureGridSpecSpanState,
    FigureMethodFamily,
    FigureMethodPlotValueState,
    FigureOperationKind,
    FigureOperationState,
    FigureRecipeState,
    FigureSourceState,
    FigureSubplotsState,
)
from erlab.interactive._figurecomposer._exceptions import FigureComposerInputError
from erlab.interactive._figurecomposer._model import (
    _operation_metadata as figurecomposer_operation_metadata,
)
from erlab.interactive._figurecomposer._model._document import FigureDocument
from erlab.interactive._figurecomposer._operations import (
    _custom_code as figurecomposer_custom_code_operation,
)
from erlab.interactive._figurecomposer._operations import (
    _line_profile as figurecomposer_line_profile,
)
from erlab.interactive._figurecomposer._operations import (
    _photon_energy as figurecomposer_photon_energy,
)
from erlab.interactive._figurecomposer._operations import (
    _set_palette as figurecomposer_set_palette,
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
    _plot_data as method_plot_data,
)
from erlab.interactive._figurecomposer._operations._method import (
    _plot_editor as method_plot_editor,
)
from erlab.interactive._figurecomposer._operations._method import _spec as method_spec
from erlab.interactive._figurecomposer._operations._method import _state as method_state
from erlab.interactive._figurecomposer._operations._plot_slices import (
    _codegen as plot_slices_codegen,
)
from erlab.interactive._figurecomposer._operations._plot_slices import (
    _editor as plot_slices_editor,
)
from erlab.interactive._figurecomposer._operations._plot_slices import (
    _model as plot_slices_model,
)
from erlab.interactive._figurecomposer._operations._plot_slices import (
    _panel_style_editor as plot_slices_panel_style_editor,
)
from erlab.interactive._figurecomposer._operations._plot_slices import (
    _render as plot_slices_render,
)
from erlab.interactive._figurecomposer._operations._plot_slices import (
    _spec as plot_slices_spec,
)

from ._common import _activate_combo_index, _select_operation_rows


def test_figure_document_orders_source_dependencies_and_rejects_cycles() -> None:
    source = xr.DataArray(np.arange(3.0), dims="x", name="source")
    sources = (
        FigureSourceState(name="base"),
        FigureSourceState(name="selected", selection_source="base"),
    )
    document = FigureDocument(
        FigureRecipeState(sources=sources, primary_source="base"),
        source_data={"base": source, "selected": source},
    )

    assert document.source_dependency_names(("selected",)) == ("base", "selected")

    document.replace_recipe(
        document.recipe.model_copy(
            update={
                "sources": (
                    sources[0].model_copy(update={"selection_source": "selected"}),
                    sources[1],
                )
            }
        )
    )
    with pytest.raises(FigureComposerInputError, match="selected -> base -> selected"):
        document.source_dependency_names(("selected",), reject_cycles=True)


def test_figure_document_replaces_source_payloads_together() -> None:
    data = xr.DataArray(np.arange(3.0), dims="x", name="data")
    base = xr.DataArray(np.arange(4.0), dims="x", name="base")
    source_data = {"data": data}
    selection_base_data = {"selected": base}
    document = FigureDocument(FigureRecipeState())

    document.replace_source_payloads(source_data, selection_base_data)
    source_data.clear()
    selection_base_data.clear()

    xr.testing.assert_identical(document.source_data["data"], data)
    xr.testing.assert_identical(
        document.source_selection_base_data["selected"],
        base,
    )
    assert document.source_data["data"] is not data
    assert document.source_selection_base_data["selected"] is not base

    data.values[:] = -1.0
    base.values[:] = -1.0

    np.testing.assert_array_equal(document.source_data["data"], np.arange(3.0))
    np.testing.assert_array_equal(
        document.source_selection_base_data["selected"],
        np.arange(4.0),
    )

    document.replace_source_payloads({"data": data}, {})

    xr.testing.assert_identical(document.source_data["data"], data)
    assert document.source_data["data"] is not data
    assert document.source_selection_base_data == {}


def test_figure_document_owns_constructor_source_payloads() -> None:
    data = xr.DataArray(
        np.arange(3.0),
        dims="x",
        attrs={"nested": {"label": "original"}},
    )
    document = FigureDocument(
        FigureRecipeState(),
        source_data={"data": data},
        source_selection_base_data={"data": data},
    )

    assert document.source_data["data"] is document.source_selection_base_data["data"]

    data.values[:] = -1.0
    data.attrs["nested"]["label"] = "mutated"

    np.testing.assert_array_equal(document.source_data["data"], np.arange(3.0))
    assert document.source_data["data"].attrs["nested"]["label"] == "original"


def test_figure_document_discards_owned_source_payloads() -> None:
    data = xr.DataArray(np.arange(3.0), dims="x", name="data")
    document = FigureDocument(
        FigureRecipeState(
            sources=(
                FigureSourceState(name="data"),
                FigureSourceState(name="other"),
            )
        ),
        source_data={"data": data, "other": data + 1.0},
        source_selection_base_data={"data": data},
    )
    revision = document.source_revision

    assert document.discard_source_data(("missing",)) == ()
    assert document.source_revision == revision
    assert document.discard_source_data(("data", "data")) == ("data",)
    assert document.source_revision == revision + 1
    assert set(document.source_data) == {"other"}
    assert document.source_selection_base_data == {}


def test_figure_document_attaches_source_observer_after_construction() -> None:
    revisions: list[int] = []
    document = FigureDocument(
        FigureRecipeState(),
        source_data={"data": xr.DataArray([1.0], dims="x")},
    )

    document.set_source_changed_callback(revisions.append)
    document.touch_source_payloads()

    assert revisions == [2]

    document.set_source_changed_callback(None)
    document.touch_source_payloads()
    assert revisions == [2]


def test_figure_document_owns_assigned_recipe_state() -> None:
    recipe = FigureRecipeState(
        setup=FigureSubplotsState(figsize=(4.0, 3.0)),
        operations=(
            FigureOperationState.plot_array(
                label="image",
                source="data",
            ).model_copy(update={"extra_kwargs": {"alpha": 0.5}}),
        ),
    )
    document = FigureDocument(recipe)

    recipe.setup.figsize = (8.0, 6.0)
    recipe.operations[0].extra_kwargs["alpha"] = 0.75
    assert document.recipe.setup.figsize == (4.0, 3.0)
    assert document.recipe.operations[0].extra_kwargs["alpha"] == 0.5

    replacement = document.recipe.model_copy(deep=True)
    replacement.setup.figsize = (5.0, 4.0)
    assert document.replace_recipe(replacement)
    revision = document.recipe_revision

    replacement.setup.figsize = (9.0, 7.0)
    replacement.operations[0].extra_kwargs["alpha"] = 1.0
    assert document.recipe.setup.figsize == (5.0, 4.0)
    assert document.recipe.operations[0].extra_kwargs["alpha"] == 0.5
    assert document.recipe_revision == revision


def test_figure_composer_prepared_data_cache_is_bounded_and_persists_dask() -> None:
    import dask
    import dask.array as da

    cache = figurecomposer_cache._BoundedCache(max_entries=2, max_bytes=32)
    cache["first"] = np.arange(2.0)
    cache["second"] = np.arange(2.0)
    assert cache["first"].tolist() == [0.0, 1.0]
    cache["third"] = np.arange(2.0)
    assert list(cache) == ["first", "third"]

    cache["first"] = np.arange(10.0)
    assert "first" not in cache

    compute_calls: list[None] = []

    @dask.delayed
    def load_values() -> np.ndarray:
        compute_calls.append(None)
        return np.arange(4.0)

    lazy = xr.DataArray(
        da.from_delayed(load_values(), shape=(4,), dtype=float),
        dims="x",
        name="lazy",
    )
    persisted_cache = figurecomposer_cache._BoundedCache(
        max_entries=2, max_bytes=1024, persist_dask=True
    )
    with dask.config.set(scheduler="synchronous"):
        persisted_cache["lazy"] = lazy
        np.testing.assert_allclose(persisted_cache["lazy"].values, np.arange(4.0))
        np.testing.assert_allclose(persisted_cache["lazy"].values, np.arange(4.0))
    assert compute_calls == [None]


def test_figure_composer_render_cache_fails_closed_and_restores_session_policy() -> (
    None
):
    class UnsupportedSelector:
        __hash__ = None

        def __repr__(self) -> str:
            return "truncated selector"

    cache = figurecomposer_cache._RenderDataCache()
    compute_calls: list[int] = []
    cyclic_plan: dict[str, object] = {}
    cyclic_plan["self"] = cyclic_plan

    def compute() -> int:
        compute_calls.append(len(compute_calls))
        return compute_calls[-1]

    assert figurecomposer_cache._freeze_cache_value(1) != (
        figurecomposer_cache._freeze_cache_value(1.0)
    )
    assert figurecomposer_cache._freeze_cache_value([1]) != (
        figurecomposer_cache._freeze_cache_value((1,))
    )
    assert (
        cache.get_or_compute(
            "selection",
            {"selector": UnsupportedSelector()},
            compute,
            source_revision=1,
        )
        == 0
    )
    assert (
        cache.get_or_compute(
            "selection",
            {"selector": UnsupportedSelector()},
            compute,
            source_revision=1,
        )
        == 1
    )
    assert len(cache) == 0

    assert (
        cache.get_or_compute(
            "selection",
            cyclic_plan,
            compute,
            source_revision=1,
        )
        == 2
    )
    assert len(cache) == 0

    assert (
        cache.get_or_compute("selection", {"selector": 1.0}, compute, source_revision=1)
        == 3
    )
    assert (
        cache.get_or_compute("selection", {"selector": 1.0}, compute, source_revision=1)
        == 3
    )
    assert len(cache) == 1

    with cache.render_session(source_revision=1, cache_safe=False):
        assert (
            cache.get_or_compute(
                "selection", {"selector": 1.0}, compute, source_revision=1
            )
            == 4
        )
        assert (
            cache.get_or_compute(
                "selection", {"selector": 1.0}, compute, source_revision=1
            )
            == 5
        )
        assert len(cache) == 0

    assert (
        cache.get_or_compute("selection", {"selector": 1.0}, compute, source_revision=1)
        == 6
    )
    assert len(cache) == 1

    with (
        pytest.raises(RuntimeError, match="render failed"),
        cache.render_session(source_revision=1, cache_safe=False),
    ):
        raise RuntimeError("render failed")

    assert (
        cache.get_or_compute("selection", {"selector": 1.0}, compute, source_revision=1)
        == 7
    )
    assert len(cache) == 1


def test_figure_composer_render_cache_nested_values_have_explicit_semantics() -> None:
    import dataclasses
    import datetime
    import enum

    import dask.array as da
    import pydantic

    @dataclasses.dataclass(frozen=True)
    class Box:
        value: object

    class BoxModel(pydantic.BaseModel):
        value: object

    class Mode(enum.Enum):
        VALUE = "value"

    array = np.arange(2.0)
    lazy = xr.DataArray(da.arange(2, chunks=1), dims="x")

    assert figurecomposer_cache._estimated_nbytes(Box(array)) == array.nbytes
    assert figurecomposer_cache._estimated_nbytes(BoxModel(value=array)) == array.nbytes
    assert figurecomposer_cache._contains_dask_collection(Box(lazy))
    assert figurecomposer_cache._contains_dask_collection(BoxModel(value=lazy))

    explicit_values = (
        np.int64(1),
        Mode.VALUE,
        b"value",
        datetime.date(2026, 7, 23),
        1.0 + 2.0j,
        np.array(["value", 1], dtype=object),
        frozenset({"value"}),
    )
    for value in explicit_values:
        assert isinstance(figurecomposer_cache._freeze_cache_value(value), tuple)


def test_figure_document_keeps_selected_dask_sources_lazy() -> None:
    import dask
    import dask.array as da

    compute_calls: list[None] = []

    @dask.delayed
    def load_values() -> np.ndarray:
        compute_calls.append(None)
        return np.arange(12.0).reshape(3, 4)

    lazy = xr.DataArray(
        da.from_delayed(load_values(), shape=(3, 4), dtype=float),
        dims=("x", "y"),
        coords={"x": np.arange(3.0), "y": np.arange(4.0)},
        name="lazy",
    )
    first_source = FigureSourceState(
        name="first", selection_source="lazy", qsel={"x": 1.0}
    )
    second_source = FigureSourceState(
        name="second", selection_source="lazy", qsel={"x": 2.0}
    )
    with dask.config.set(scheduler="synchronous"):
        first = FigureDocument.source_data_from_selection(lazy, first_source)
        second = FigureDocument.source_data_from_selection(lazy, second_source)
        assert compute_calls == []
        assert dask.base.is_dask_collection(second)
        np.testing.assert_allclose(first.values, np.arange(4.0, 8.0))
    assert compute_calls == [None]


def test_figure_document_dask_selection_errors_remain_lazy() -> None:
    import dask
    import dask.array as da

    @dask.delayed
    def fail_to_load() -> np.ndarray:
        raise RuntimeError("backend read failed")

    lazy = xr.DataArray(
        da.from_delayed(fail_to_load(), shape=(2, 3), dtype=float),
        dims=("x", "y"),
        coords={"x": [0, 1], "y": [0, 1, 2]},
        name="lazy",
    )
    document = FigureDocument(
        FigureRecipeState(
            sources=(FigureSourceState(name="lazy"),), primary_source="lazy"
        ),
        source_data={"lazy": lazy},
    )

    with dask.config.set(scheduler="synchronous"):
        result = document.update_source_selection_dimension(("lazy",), "x", "isel", 0)
        assert result.updated == ("lazy",)
        with pytest.raises(RuntimeError, match="backend read failed"):
            _ = document.source_data["lazy"].values


def test_figure_document_replaces_only_valid_layout_setup() -> None:
    document = FigureDocument(FigureRecipeState())
    invalid = document.recipe.setup.model_copy(update={"nrows": 0})

    with pytest.raises(ValueError, match="at least one row"):
        document.replace_setup(invalid)

    assert document.recipe.setup.nrows == 1
    updated = document.recipe.setup.model_copy(update={"nrows": 2})
    assert document.replace_setup(updated)
    assert not document.replace_setup(updated)


def test_figure_document_converts_layout_and_operation_targets_atomically() -> None:
    axes_operation = FigureOperationState.plot_array(
        label="image",
        source="data",
        axes=FigureAxesSelectionState(axes=((0, 1), (1, 0)), expression="custom_axes"),
    )
    figure_operation = FigureOperationState.method(
        family=FigureMethodFamily.FIGURE,
        name="supxlabel",
        axes=FigureAxesSelectionState(axes=((1, 1),), expression="figure_expression"),
    )
    palette_operation = FigureOperationState.set_palette()
    document = FigureDocument(
        FigureRecipeState(
            setup=FigureSubplotsState(nrows=2, ncols=2),
            operations=(axes_operation, figure_operation, palette_operation),
        )
    )

    assert document.convert_layout_mode("gridspec")
    gridspec_setup = document.recipe.setup
    axes_ids = figurecomposer_gridspec._gridspec_all_axes_ids(gridspec_setup)
    converted = document.recipe.operations
    assert gridspec_setup.layout_mode == "gridspec"
    assert converted[0].axes.axes_ids == (axes_ids[1], axes_ids[2])
    assert converted[0].axes.expression == ""
    assert converted[1] == figure_operation
    assert converted[2] == palette_operation
    assert tuple(operation.operation_id for operation in converted) == tuple(
        operation.operation_id
        for operation in (axes_operation, figure_operation, palette_operation)
    )
    assert not document.convert_layout_mode("gridspec")

    assert document.convert_layout_mode("subplots")
    assert document.recipe.setup.layout_mode == "subplots"
    assert document.recipe.operations[0].axes.axes == ((0, 1), (1, 0))
    assert document.recipe.operations[1] == figure_operation
    with pytest.raises(ValueError, match="unknown figure layout mode"):
        document.convert_layout_mode("unknown")


def test_figure_document_converts_nested_gridspec_targets_to_root_spans() -> None:
    first = FigureGridSpecAxesState(
        axes_id="nested-first",
        span=FigureGridSpecSpanState(row_start=0, row_stop=1, col_start=0, col_stop=1),
    )
    second = FigureGridSpecAxesState(
        axes_id="nested-second",
        span=FigureGridSpecSpanState(row_start=0, row_stop=1, col_start=1, col_stop=2),
    )
    nested = FigureGridSpecGridState(
        grid_id="nested",
        nrows=1,
        ncols=2,
        span=FigureGridSpecSpanState(row_start=0, row_stop=1, col_start=0, col_stop=2),
        axes=(first, second),
    )
    setup = FigureSubplotsState(
        layout_mode="gridspec",
        gridspec=FigureGridSpecLayoutState(
            root=FigureGridSpecGridState(
                grid_id="root", nrows=2, ncols=2, child_grids=(nested,)
            )
        ),
    )
    operation = FigureOperationState.line(
        label="profiles",
        source="data",
        axes=FigureAxesSelectionState(
            axes=((1, 1),),
            axes_ids=("nested-first", "nested-second", "removed"),
            expression="advanced",
        ),
    )
    document = FigureDocument(FigureRecipeState(setup=setup, operations=(operation,)))

    assert document.convert_layout_mode("subplots")
    assert document.recipe.operations[0].axes.axes == ((0, 0), (0, 1))
    assert document.recipe.operations[0].axes.expression == ""


def test_figure_document_renames_source_references_atomically() -> None:
    data = xr.DataArray(np.arange(3.0), dims="x", name="data")
    document = FigureDocument(
        FigureRecipeState(
            sources=(
                FigureSourceState(name="data", label="data"),
                FigureSourceState(name="selected", selection_source="data"),
            ),
            operations=(
                FigureOperationState.plot_array(label="array", source="selected"),
                FigureOperationState.custom(
                    label="python", code="result = data + selected", trusted=True
                ),
            ),
            primary_source="data",
        ),
        source_data={"data": data, "selected": data},
        source_selection_base_data={"selected": data},
    )

    assert document.source_usage_count("data") == 2
    assert not document.source_is_removable("data")
    assert document.rename_source("data", "renamed")
    assert tuple(document.source_by_name()) == ("renamed", "selected")
    assert document.source_by_name()["renamed"].label == "renamed"
    assert document.source_by_name()["selected"].selection_source == "renamed"
    assert document.recipe.primary_source == "renamed"
    assert tuple(document.source_data) == ("renamed", "selected")
    assert document.recipe.operations[1].code == "result = renamed + selected"

    recipe = document.recipe.model_copy(
        update={
            "operations": (
                FigureOperationState.custom(
                    label="ambiguous", code="renamed = renamed + 1", trusted=True
                ),
            )
        }
    )
    document.replace_recipe(recipe)
    before_data = dict(document.source_data)
    with pytest.raises(FigureComposerInputError, match="also binds"):
        document.rename_source("renamed", "other")
    assert document.recipe == recipe
    assert document.source_data == before_data


def test_figure_document_duplicate_move_reorder_and_remove_sources() -> None:
    data = xr.DataArray(np.arange(3.0), dims="x", name="data")
    document = FigureDocument(
        FigureRecipeState(
            sources=(
                FigureSourceState(name="base"),
                FigureSourceState(name="selected", selection_source="base"),
                FigureSourceState(name="other"),
            ),
            operations=(FigureOperationState.line(label="line", source="other"),),
            primary_source="base",
        ),
        source_data={"base": data, "selected": data, "other": data},
        source_selection_base_data={"selected": data},
    )

    assert document.duplicate_sources(("selected",)) == ("selected_copy",)
    assert document.source_by_name()["selected_copy"].selection_source == "base"
    assert document.source_data["selected_copy"] is not data
    assert document.can_move_sources(("selected_copy",), 1)
    assert document.move_sources(("selected_copy",), 1)
    with pytest.raises(ValueError, match="must be -1 or 1"):
        document.can_move_sources(("selected_copy",), 0)
    with pytest.raises(ValueError, match="must be -1 or 1"):
        document.move_sources(("selected_copy",), 2)
    assert tuple(document.source_by_name()) == (
        "base",
        "selected",
        "other",
        "selected_copy",
    )
    assert document.reorder_sources(("other", "base", "selected", "selected_copy"))
    with pytest.raises(ValueError, match="exact permutation"):
        document.reorder_sources(("other", "base"))

    assert document.remove_sources(("base", "selected", "selected_copy")) == (
        "selected",
        "selected_copy",
        "base",
    )
    assert tuple(document.source_by_name()) == ("other",)
    assert document.recipe.primary_source == "other"
    assert not document.remove_source("other")


def test_figure_document_updates_and_inserts_operations_by_identity() -> None:
    operations = tuple(
        FigureOperationState(
            kind=FigureOperationKind.SET_PALETTE,
            label=label,
            operation_id=operation_id,
        )
        for operation_id, label in (
            ("first-id", "first"),
            ("second-id", "second"),
            ("third-id", "third"),
        )
    )
    duplicate_recipe = FigureRecipeState(operations=(operations[0], operations[0]))
    with pytest.raises(ValueError, match="must be unique"):
        FigureDocument(duplicate_recipe)

    document = FigureDocument(FigureRecipeState(operations=operations))
    before = document.recipe
    with pytest.raises(ValueError, match="must be unique"):
        document.replace_recipe(duplicate_recipe)
    assert document.recipe == before

    assert document.operation_index("second-id") == 1
    assert document.operation_index("missing-id") is None
    stored_operation = document.operation_by_id("second-id")
    assert stored_operation == operations[1]
    assert stored_operation is not operations[1]
    assert document.operation_by_id("missing-id") is None

    updated_indices: list[int] = []

    def update_label(
        index: int, operation: FigureOperationState
    ) -> FigureOperationState:
        updated_indices.append(index)
        return operation.model_copy(update={"label": f"{operation.label} updated"})

    assert document.update_operations_by_ids(
        ("third-id", "missing-id", "second-id", "second-id"), update_label
    )
    assert updated_indices == [1, 2]
    assert tuple(
        operation.operation_id for operation in document.recipe.operations
    ) == (
        "first-id",
        "second-id",
        "third-id",
    )
    assert tuple(operation.label for operation in document.recipe.operations) == (
        "first",
        "second updated",
        "third updated",
    )
    assert not document.update_operations_by_ids(
        ("second-id",), lambda _index, operation: operation
    )
    before = document.recipe
    with pytest.raises(ValueError, match="cannot change its ID"):
        document.update_operations_by_ids(
            ("second-id",),
            lambda _index, operation: operation.model_copy(
                update={"operation_id": "replacement-id"}
            ),
        )
    assert document.recipe == before

    replacement = document.recipe.operations[0].model_copy(
        update={"label": "replacement"}
    )
    assert document.replace_operation(0, replacement)
    assert not document.replace_operation(0, replacement)
    assert document.recipe.operations[0].operation_id == "first-id"
    assert not document.replace_operations(document.recipe.operations)
    before = document.recipe
    with pytest.raises(ValueError, match="must be unique"):
        document.replace_operations(
            (document.recipe.operations[0], document.recipe.operations[0])
        )
    assert document.recipe == before
    with pytest.raises(IndexError, match="operation index"):
        document.replace_operation(-1, replacement)
    with pytest.raises(ValueError, match="already in use"):
        document.replace_operation(
            0,
            replacement.model_copy(update={"operation_id": "second-id"}),
        )

    appended = FigureOperationState(
        kind=FigureOperationKind.SET_PALETTE,
        label="appended",
        operation_id="appended-id",
    )
    assert document.append_operation(appended) == 3
    inserted = (
        FigureOperationState(
            kind=FigureOperationKind.SET_PALETTE,
            label="inserted one",
            operation_id="inserted-one-id",
        ),
        FigureOperationState(
            kind=FigureOperationKind.SET_PALETTE,
            label="inserted two",
            operation_id="inserted-two-id",
        ),
    )
    assert document.insert_operations(1, inserted) == (
        "inserted-one-id",
        "inserted-two-id",
    )
    assert tuple(
        operation.operation_id for operation in document.recipe.operations
    ) == (
        "first-id",
        "inserted-one-id",
        "inserted-two-id",
        "second-id",
        "third-id",
        "appended-id",
    )
    assert document.insert_operations(0, ()) == ()
    with pytest.raises(IndexError, match="insertion index"):
        document.insert_operations(len(document.recipe.operations) + 1, inserted)
    with pytest.raises(ValueError, match="already in use"):
        document.append_operation(appended)
    with pytest.raises(ValueError, match="must be unique"):
        document.insert_operations(0, (inserted[0], inserted[0]))


def test_figure_document_duplicates_removes_reorders_and_moves_operations() -> None:
    operations = tuple(
        FigureOperationState(
            kind=FigureOperationKind.SET_PALETTE,
            label=operation_id,
            operation_id=operation_id,
        )
        for operation_id in ("a", "b", "c", "d", "e")
    )
    document = FigureDocument(FigureRecipeState(operations=operations))

    duplicate_ids = document.duplicate_operations((3, 1))
    assert len(duplicate_ids) == 2
    assert set(duplicate_ids).isdisjoint(
        operation.operation_id for operation in operations
    )
    assert (
        document.recipe.operations[4].model_copy(update={"operation_id": "b"})
        == operations[1]
    )
    assert (
        document.recipe.operations[5].model_copy(update={"operation_id": "d"})
        == operations[3]
    )
    assert tuple(
        operation.operation_id for operation in document.recipe.operations
    ) == (
        "a",
        "b",
        "c",
        "d",
        *duplicate_ids,
        "e",
    )
    before = document.recipe
    with pytest.raises(IndexError, match="operation index"):
        document.duplicate_operations((-1,))
    assert document.recipe == before

    assert document.remove_operation_indices((5, 2, 2)) == (
        "c",
        duplicate_ids[1],
    )
    assert tuple(
        operation.operation_id for operation in document.recipe.operations
    ) == (
        "a",
        "b",
        "d",
        duplicate_ids[0],
        "e",
    )
    assert document.remove_operation_indices(()) == ()
    with pytest.raises(IndexError, match="operation index"):
        document.remove_operation_indices((99,))

    reordered = ("e", "a", "b", "d", duplicate_ids[0])
    assert document.reorder_operations(reordered)
    assert not document.reorder_operations(reordered)
    before = document.recipe
    with pytest.raises(ValueError, match="exact permutation"):
        document.reorder_operations(("e", "a", "b", "d"))
    assert document.recipe == before

    assert document.can_move_operations(("a", "b"), 1)
    assert document.move_operations(("a", "b"), 1)
    assert tuple(
        operation.operation_id for operation in document.recipe.operations
    ) == (
        "e",
        "d",
        "a",
        "b",
        duplicate_ids[0],
    )
    assert document.move_operations(("a", "b"), -1)
    assert tuple(
        operation.operation_id for operation in document.recipe.operations
    ) == (
        "e",
        "a",
        "b",
        "d",
        duplicate_ids[0],
    )
    assert not document.can_move_operations(("e",), -1)
    assert not document.move_operations(("e",), -1)
    assert not document.can_move_operations(("missing",), 1)
    assert not document.move_operations(("missing",), 1)
    with pytest.raises(ValueError, match="must be -1 or 1"):
        document.can_move_operations(("a",), 0)
    with pytest.raises(ValueError, match="must be -1 or 1"):
        document.move_operations(("a",), 2)


def test_figure_document_pastes_operations_and_sources_atomically() -> None:
    data = xr.DataArray(np.arange(3.0), dims=("x",), name="data")
    document = FigureDocument(
        FigureRecipeState(
            sources=(FigureSourceState(name="data"),),
            operations=(
                FigureOperationState(
                    kind=FigureOperationKind.SET_PALETTE,
                    label="palette",
                    operation_id="existing-operation",
                ),
            ),
        ),
        source_data={"data": data, "data_copy": data},
    )
    copied = FigureOperationState.plot_array(label="image", source="data")
    before = document.recipe
    with pytest.raises(IndexError, match="insertion index"):
        document.paste_operations(
            3,
            (copied,),
            (FigureSourceState(name="data"),),
            {"data": data},
            {"data": data},
        )
    assert document.recipe == before

    result = document.paste_operations(
        1,
        (copied,),
        (
            FigureSourceState(name="data"),
            FigureSourceState(name="data"),
        ),
        {"data": data},
        {"data": data},
    )
    assert result.source_data_changed
    assert len(result.operation_ids) == 1
    assert result.operation_ids[0] not in {
        "existing-operation",
        copied.operation_id,
    }
    assert tuple(source.name for source in document.recipe.sources) == (
        "data",
        "data_copy_2",
    )
    assert document.recipe.operations[1].sources == ("data_copy_2",)
    xr.testing.assert_identical(document.source_data["data_copy_2"], data)
    xr.testing.assert_identical(
        document.source_selection_base_data["data_copy_2"], data
    )

    metadata_source = FigureSourceState(name="metadata_only")
    preserved = FigureDocument(
        FigureRecipeState(sources=(metadata_source,)),
        source_data={"existing_data": data},
    )
    preserved_result = preserved.paste_operations(
        0,
        (FigureOperationState.custom(label="code", code="pass", trusted=True),),
        (metadata_source,),
        {"metadata_only": data},
        {},
        preserve_existing=True,
    )
    assert preserved_result.source_data_changed
    assert tuple(source.name for source in preserved.recipe.sources) == (
        "metadata_only",
    )
    xr.testing.assert_identical(preserved.source_data["metadata_only"], data)

    preserved_data = FigureDocument(
        FigureRecipeState(sources=(FigureSourceState(name="data"),)),
        source_data={"data": data},
    )
    replacement = data + 10
    preserved_data_result = preserved_data.paste_operations(
        0,
        (FigureOperationState.plot_array(label="image", source="data"),),
        (FigureSourceState(name="data"),),
        {"data": replacement},
        {},
        preserve_existing=True,
    )
    assert not preserved_data_result.source_data_changed
    xr.testing.assert_identical(preserved_data.source_data["data"], data)

    fresh_base = data + 20
    stale_base = data + 30
    selected_source = FigureSourceState(
        name="selected",
        selection_source="data",
        qsel={"x": 0.0},
    )
    preserved_selection = FigureDocument(
        FigureRecipeState(
            sources=(FigureSourceState(name="data"), selected_source),
        ),
        source_data={"data": fresh_base, "selected": fresh_base.qsel(x=0.0)},
        source_selection_base_data={"selected": fresh_base},
    )
    preserved_selection_result = preserved_selection.paste_operations(
        0,
        (FigureOperationState.plot_array(label="image", source="selected"),),
        (FigureSourceState(name="data"), selected_source),
        {"data": stale_base, "selected": stale_base.qsel(x=0.0)},
        {"selected": stale_base},
        preserve_existing=True,
    )
    assert not preserved_selection_result.source_data_changed
    xr.testing.assert_identical(
        preserved_selection.source_selection_base_data["selected"], fresh_base
    )


def test_figure_document_copies_only_accepted_source_inputs(monkeypatch) -> None:
    data = xr.DataArray(np.arange(3.0), dims="x", name="data")
    document = FigureDocument(
        FigureRecipeState(sources=(FigureSourceState(name="data"),)),
        source_data={"data": data},
    )
    rejected_replacement = data + 10.0
    preserved_paste = data + 20.0
    accepted_replacement = data + 30.0
    deep_copy_count = 0
    original_copy = xr.DataArray.copy

    def counted_copy(self, *args, **kwargs):
        nonlocal deep_copy_count
        if kwargs.get("deep", True):
            deep_copy_count += 1
        return original_copy(self, *args, **kwargs)

    monkeypatch.setattr(xr.DataArray, "copy", counted_copy)

    result = document.replace_source(
        "missing",
        FigureSourceState(name="missing"),
        rejected_replacement,
    )
    assert not result
    assert deep_copy_count == 0

    result = document.paste_operations(
        0,
        (FigureOperationState.plot_array(label="image", source="data"),),
        (FigureSourceState(name="data"),),
        {"data": preserved_paste},
        {},
        preserve_existing=True,
    )
    assert not result.source_data_changed
    assert deep_copy_count == 0

    result = document.replace_source(
        "data",
        FigureSourceState(name="data"),
        accepted_replacement,
    )
    assert result.updated == ("data",)
    assert deep_copy_count == 1
    accepted_replacement.values[:] = -1.0
    np.testing.assert_array_equal(document.source_data["data"], data.values + 30.0)


def test_figure_document_add_and_replace_recompute_dependents_atomically() -> None:
    base = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("u", "v"),
        coords={"u": [0.0, 1.0], "v": [0.0, 1.0, 2.0]},
        name="base",
    )
    selected = base.qsel(u=1.0)
    document = FigureDocument(
        FigureRecipeState(
            sources=(
                FigureSourceState(name="base", node_uid="node"),
                FigureSourceState(
                    name="selected",
                    selection_source="base",
                    qsel={"u": 1.0},
                ),
            ),
            primary_source="base",
        ),
        source_data={"base": base, "selected": selected},
        source_selection_base_data={"selected": base},
    )

    refreshed = base + 10.0
    result = document.add_sources(
        (FigureSourceState(name="incoming", node_uid="node"),),
        {"incoming": refreshed},
    )
    assert result.updated == (("incoming", "base"),)
    xr.testing.assert_identical(document.source_data["selected"], refreshed.qsel(u=1.0))

    before_recipe = document.recipe
    before_data = dict(document.source_data)
    incompatible = refreshed.isel(u=0, drop=True)
    result = document.replace_source(
        "base", FigureSourceState(name="bad", node_uid="other"), incompatible
    )
    assert not result
    assert result.skipped
    assert document.recipe == before_recipe
    for name, expected in before_data.items():
        xr.testing.assert_identical(document.source_data[name], expected)

    replacement = base + 20.0
    result = document.replace_source(
        "base", FigureSourceState(name="replacement", node_uid="other"), replacement
    )
    assert result.updated == ("base",)
    xr.testing.assert_identical(
        document.source_data["selected"], replacement.qsel(u=1.0)
    )
    expected_selected = replacement.qsel(u=1.0).copy(deep=True)
    replacement.values[:] = -1.0
    xr.testing.assert_identical(document.source_data["selected"], expected_selected)


def test_figure_document_updates_source_selection_per_compatible_source() -> None:
    data = xr.DataArray(
        np.arange(3.0),
        dims="x",
        coords={"x": [0.0, 1.0, 2.0]},
        name="data",
    )
    scalar = xr.DataArray(1.0, name="scalar")
    document = FigureDocument(
        FigureRecipeState(
            sources=(
                FigureSourceState(name="data"),
                FigureSourceState(name="scalar"),
            ),
            primary_source="data",
        ),
        source_data={"data": data, "scalar": scalar},
    )

    result = document.update_source_selection_dimension(
        ("data", "scalar", "missing"), "x", "isel", 1
    )

    assert result.updated == ("data",)
    assert {name for name, _detail in result.skipped} == {"scalar", "missing"}
    assert document.source_by_name()["data"].isel == {"x": 1}
    xr.testing.assert_identical(document.source_data["data"], data.isel(x=1))
    xr.testing.assert_identical(document.source_data["scalar"], scalar)

    recipe = document.recipe
    source_data = dict(document.source_data)
    result = document.update_source_selection_dimension(
        ("data",), "x", "qsel", slice(0.0, 1.0), 0.1
    )
    assert not result
    assert result.skipped
    assert document.recipe == recipe
    for name, expected in source_data.items():
        xr.testing.assert_identical(document.source_data[name], expected)


def test_operation_metadata_covers_every_operation_kind() -> None:
    plot_array = FigureOperationState.plot_array(
        label="array", source="first"
    ).model_copy(update={"sources": ("first", "first", "second")})
    plot_slices = FigureOperationState.plot_slices(
        label="slices",
        sources=("second", "first", "second"),
        map_selections=(FigureDataSelectionState(source="ignored"),),
    )
    line = FigureOperationState.line(label="line", source="ignored").model_copy(
        update={
            "map_selections": (
                FigureDataSelectionState(source="second"),
                FigureDataSelectionState(source="first"),
                FigureDataSelectionState(source="second"),
            )
        }
    )
    photon_energy = FigureOperationState.photon_energy_overlay(source="first")
    method = FigureOperationState.method(
        family=FigureMethodFamily.AXES, name="errorbar"
    ).model_copy(
        update={
            "method_plot_data_mode": "from_data",
            "method_plot_x": FigureMethodPlotValueState(source="first"),
            "method_plot_y": FigureMethodPlotValueState(source="second"),
            "method_plot_xerr": FigureMethodPlotValueState(source="second"),
            "method_plot_yerr": FigureMethodPlotValueState(source="third"),
        }
    )
    custom = FigureOperationState.custom(
        label="custom", code="result = second + first", trusted=False
    )
    operations = (
        FigureOperationState(kind=FigureOperationKind.SET_PALETTE, label="palette"),
        plot_array,
        plot_slices,
        line,
        FigureOperationState.bz_overlay(),
        photon_energy,
        method,
        custom,
    )
    expected = {
        FigureOperationKind.SET_PALETTE: (),
        FigureOperationKind.PLOT_ARRAY: ("first", "second"),
        FigureOperationKind.PLOT_SLICES: ("second", "first"),
        FigureOperationKind.LINE: ("second", "first"),
        FigureOperationKind.BZ_OVERLAY: (),
        FigureOperationKind.PHOTON_ENERGY_OVERLAY: ("first",),
        FigureOperationKind.METHOD: ("first", "second", "third"),
        FigureOperationKind.CUSTOM: (),
    }

    assert {operation.kind for operation in operations} == set(FigureOperationKind)
    assert {
        operation.kind: figurecomposer_operation_metadata.operation_uses_axes(operation)
        for operation in operations
    } == {
        FigureOperationKind.SET_PALETTE: False,
        FigureOperationKind.PLOT_ARRAY: True,
        FigureOperationKind.PLOT_SLICES: True,
        FigureOperationKind.LINE: True,
        FigureOperationKind.BZ_OVERLAY: True,
        FigureOperationKind.PHOTON_ENERGY_OVERLAY: True,
        FigureOperationKind.METHOD: True,
        FigureOperationKind.CUSTOM: False,
    }
    assert not figurecomposer_operation_metadata.operation_uses_axes(
        FigureOperationState.method(family=FigureMethodFamily.FIGURE, name="supxlabel")
    )
    assert figurecomposer_operation_metadata.operation_uses_axes(
        FigureOperationState.method(
            family=FigureMethodFamily.ERLAB, name="clean_labels"
        )
    )
    assert {
        operation.kind: (
            figurecomposer_operation_metadata.declared_operation_source_names(operation)
        )
        for operation in operations
    } == expected
    assert figurecomposer_operation_metadata.recipe_operation_source_names(
        custom, ("first", "second", "unused")
    ) == ("first", "second")
    assert figurecomposer_operation_metadata.declared_operation_source_names(
        line.model_copy(
            update={"map_selections": (FigureDataSelectionState(source="second"),)}
        )
    ) == ("ignored",)
    assert (
        figurecomposer_operation_metadata.declared_operation_source_names(
            photon_energy.model_copy(update={"hv_overlay_source": None})
        )
        == ()
    )
    for non_source_method in (
        method.model_copy(update={"method_plot_data_mode": "entered"}),
        method.model_copy(update={"method_name": "legend"}),
        method.model_copy(update={"method_family": FigureMethodFamily.FIGURE}),
    ):
        assert (
            figurecomposer_operation_metadata.declared_operation_source_names(
                non_source_method
            )
            == ()
        )

    document = FigureDocument(
        FigureRecipeState(
            sources=(
                FigureSourceState(name="first"),
                FigureSourceState(name="second"),
            ),
            operations=(custom,),
        )
    )
    assert document.operation_source_names(custom) == ("first", "second")
    assert document.direct_sources_used_by_recipe() == {"first", "second"}
    assert document.direct_sources_used_by_recipe(executable_only=True) == set()


def test_figure_composer_operation_modules_use_editor_signal_contract() -> None:
    modules = (
        figurecomposer_custom_code_operation,
        figurecomposer_line_profile,
        method_catalog,
        method_editor,
        method_execution,
        method_spec,
        method_plot_data,
        method_plot_editor,
        method_state,
        figurecomposer_photon_energy,
        plot_slices_codegen,
        plot_slices_editor,
        plot_slices_model,
        plot_slices_panel_style_editor,
        plot_slices_render,
        plot_slices_spec,
        figurecomposer_set_palette,
    )
    direct_connects: list[str] = []
    for module in modules:
        module_file = module.__file__
        assert module_file is not None
        tree = ast.parse(Path(module_file).read_text())
        direct_connects.extend(
            f"{Path(module_file).name}:{node.lineno}"
            for node in ast.walk(tree)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "connect"
            )
        )
    assert direct_connects == []


def test_figure_composer_text_helpers_parse_user_inputs() -> None:
    assert figurecomposer_text._float_pair_from_text("") is None
    assert figurecomposer_text._float_pair_from_text("1, 2.5") == (1.0, 2.5)
    with pytest.raises(FigureComposerInputError, match="two"):
        figurecomposer_text._float_pair_from_text("1")
    with pytest.raises(FigureComposerInputError, match="two"):
        figurecomposer_text._float_pair_from_text("1, bad")

    assert figurecomposer_text._plot_limit_from_text("") is None
    assert figurecomposer_text._plot_limit_from_text("1.5") == 1.5
    assert figurecomposer_text._plot_limit_from_text("None") is None
    assert figurecomposer_text._plot_limit_from_text("[2]") == 2.0
    assert figurecomposer_text._plot_limit_from_text("(1, 2)") == (1.0, 2.0)
    assert figurecomposer_text._plot_limit_from_text("0, None") == (0.0, None)
    assert figurecomposer_text._plot_limit_from_text("(None, 2)") == (None, 2.0)
    assert figurecomposer_text._limit_pair_from_text("0, None") == (0.0, None)
    assert figurecomposer_text._limit_pair_from_text("") is None
    assert figurecomposer_text._format_plot_limit((0.0, None)) == "0, None"
    with pytest.raises(FigureComposerInputError, match="one"):
        figurecomposer_text._plot_limit_from_text("(1, 2, 3)")
    with pytest.raises(FigureComposerInputError, match="one"):
        figurecomposer_text._plot_limit_from_text("(1, 'bad')")
    with pytest.raises(FigureComposerInputError, match="two"):
        figurecomposer_text._limit_pair_from_text("1")
    with pytest.raises(FigureComposerInputError, match="two"):
        figurecomposer_text._limit_pair_from_text("(1,)")
    with pytest.raises(FigureComposerInputError, match="two"):
        figurecomposer_text._limit_pair_from_text("(1, 'bad')")

    assert figurecomposer_text._float_tuple_from_text("1, 2") == (1.0, 2.0)
    with pytest.raises(FigureComposerInputError, match="numbers"):
        figurecomposer_text._float_tuple_from_text("1, bad")
    assert figurecomposer_text._literal_sequence_from_text("") == ()
    assert figurecomposer_text._literal_sequence_from_text("[1, 2]") == (1, 2)
    assert figurecomposer_text._literal_sequence_from_text("(1)") == (1,)
    assert figurecomposer_text._literal_sequence_from_text("'x'") == ("x",)
    assert figurecomposer_text._literal_sequence_from_text("1, 2") == (1, 2)

    assert figurecomposer_text._string_tuple_from_text("") == ()
    assert figurecomposer_text._string_tuple_from_text("alpha, beta") == (
        "alpha",
        "beta",
    )
    assert figurecomposer_text._string_tuple_from_text("('alpha')") == ("alpha",)
    assert figurecomposer_text._string_tuple_from_text("['alpha', 2]") == (
        "alpha",
        "2",
    )
    with pytest.raises(FigureComposerInputError, match="text"):
        figurecomposer_text._string_tuple_from_text("(1)")
    assert figurecomposer_text._text_tuple_from_text("a\n\nb") == ("a", "b")
    assert figurecomposer_text._text_tuple_from_text("a\n\nb", preserve_empty=True) == (
        "a",
        "",
        "b",
    )

    assert figurecomposer_text._dict_from_text("") == {}
    assert figurecomposer_text._dict_from_text(
        "a=1, b=slice(0, 2)", allow_slice=True
    ) == {
        "a": 1,
        "b": slice(0, 2),
    }
    assert figurecomposer_text._dict_from_text("{'a': 1}") == {"a": 1}
    with pytest.raises(FigureComposerInputError, match="keyword"):
        figurecomposer_text._dict_from_text("a=")
    with pytest.raises(FigureComposerInputError, match="keyword"):
        figurecomposer_text._dict_from_text("a=object()")
    with pytest.raises(FigureComposerInputError, match="keyword"):
        figurecomposer_text._dict_from_text("1")
    with pytest.raises(FigureComposerInputError, match="keyword"):
        figurecomposer_text._dict_from_text("{1, 2}")
    with pytest.raises(FigureComposerInputError, match="explicit"):
        figurecomposer_text._dict_from_text("**{'a': 1}")
    with pytest.raises(FigureComposerInputError, match="explicit"):
        figurecomposer_text._dict_from_text("{**{'a': 1}}")
    with pytest.raises(FigureComposerInputError, match="keyword"):
        figurecomposer_text._dict_from_text("{alpha: 1}")

    assert figurecomposer_text._format_pair(None) == ""
    assert figurecomposer_text._format_limit_pair(None) == ""
    assert figurecomposer_text._format_plot_limit(2.0) == "2"
    assert (
        figurecomposer_text._format_dim_sizes(
            xr.DataArray(np.zeros((2, 3)), dims=("alpha", "beta"))
        )
        == "alpha=2, beta=3"
    )
    assert figurecomposer_text._format_axes_tuple((), nrows=2, ncols=2) == "none"
    assert figurecomposer_text._selection_value_count(np.arange(6).reshape(2, 3)) == 6
    assert figurecomposer_text._selection_value_count([1, 2]) == 2
    assert figurecomposer_text._selection_value_count(slice(None)) is None


def test_figure_composer_redraw_and_preview_cache_edges(qtbot, monkeypatch) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(data)
    qtbot.addWidget(tool)

    render_calls: list[dict[str, object]] = []
    single_shot_calls: list[tuple[int, Callable[[], None]]] = []
    monkeypatch.setattr(
        figurecomposer_tool_module,
        "_render_preview",
        lambda _tool, **kwargs: render_calls.append(kwargs),
    )
    monkeypatch.setattr(
        erlab.interactive.utils,
        "single_shot",
        lambda _owner, delay, callback: single_shot_calls.append((delay, callback)),
    )

    tool.auto_redraw_check.setChecked(False)
    assert not tool._maybe_redraw_plot(show_window=True)
    assert tool._auto_redraw_dirty
    assert tool._preview_pixmap_stale
    assert render_calls == []

    tool._queue_preview_render_update()
    assert tool._auto_redraw_dirty
    assert single_shot_calls == []

    tool._run_queued_preview_render_update(tool._preview_render_update_generation)
    assert tool._auto_redraw_dirty
    assert tool._preview_pixmap_stale
    assert render_calls == []

    tool.auto_redraw_check.setChecked(True)
    assert render_calls == [{}]
    assert not tool._auto_redraw_dirty

    tool._redraw_plot_requested()
    assert render_calls[-1] == {}

    assert not tool._saved_tool_window_visible(xr.Dataset())
    assert tool._saved_tool_window_visible(xr.Dataset(attrs={"tool_visible": True}))

    tool.auto_redraw_check.setChecked(False)
    visible_ds = xr.Dataset(attrs={"tool_visible": True})
    tool._queue_post_restore_redraw_if_needed(visible_ds)
    assert single_shot_calls == []
    tool.auto_redraw_check.setChecked(True)
    tool._queue_post_restore_redraw_if_needed(visible_ds)
    assert single_shot_calls[-1][0] == 0
    single_shot_calls[-1][1]()
    assert render_calls[-1] == {"show_window": True}

    assert tool._persisted_preview_cache_pixmap() is None
    preview = QtGui.QPixmap(
        figurecomposer_tool_module._PERSISTED_PREVIEW_CACHE_SIZE[0] + 20,
        figurecomposer_tool_module._PERSISTED_PREVIEW_CACHE_SIZE[1] + 20,
    )
    preview.fill(QtGui.QColor("red"))
    tool._preview_pixmap_cache = preview
    persisted = tool._persisted_preview_cache_pixmap()
    assert persisted is not None
    assert (
        persisted.width() <= figurecomposer_tool_module._PERSISTED_PREVIEW_CACHE_SIZE[0]
    )
    saved = tool._append_persistence_payload(xr.Dataset())
    assert figurecomposer_tool_module._PERSISTED_PREVIEW_CACHE_ATTR in saved.attrs

    restored = FigureComposerTool(data)
    qtbot.addWidget(restored)
    restored._restore_persisted_preview_cache(
        xr.Dataset(
            attrs={figurecomposer_tool_module._PERSISTED_PREVIEW_CACHE_ATTR: "bad"}
        )
    )
    assert restored._preview_pixmap_cache is None
    restored._restore_persisted_preview_cache(saved)
    assert restored._preview_pixmap_cache is not None
    assert (
        restored._preview_pixmap_stale
        == saved.attrs[figurecomposer_tool_module._PERSISTED_PREVIEW_CACHE_STALE_ATTR]
    )


def test_figure_composer_visible_redraw_draws_once_and_skips_unchanged(
    qtbot, monkeypatch
) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(data)
    qtbot.addWidget(tool)
    tool.show_figure_window(activate=False)

    canvas = tool.figure_window.canvas
    original_draw = canvas.draw
    draw_calls: list[None] = []

    def record_draw() -> None:
        draw_calls.append(None)
        original_draw()

    monkeypatch.setattr(canvas, "draw", record_draw)

    tool._redraw_plot(force=True)
    assert draw_calls == [None]
    assert not tool.preview_pixmap_stale

    tool._redraw_plot()
    assert draw_calls == [None]

    tool._mark_preview_pixmap_stale()
    tool._redraw_plot()
    assert draw_calls == [None, None]

    operation = tool.tool_status.operations[0]
    tool._document.replace_operation(0, operation.model_copy(update={"cmap": "plasma"}))
    tool._redraw_plot()
    assert draw_calls == [None, None, None]

    tool._redraw_plot(force=True)
    assert draw_calls == [None, None, None, None]

    operation = tool.tool_status.operations[0]
    tool._rendering = True
    try:
        tool._document.replace_operation(
            0, operation.model_copy(update={"cmap": "magma"})
        )
        tool._redraw_plot()
    finally:
        tool._rendering = False

    assert draw_calls == [None, None, None, None]
    assert tool.figure.axes[0].images[0].get_cmap().name == "plasma"
    assert tool._last_preview_render_signature is None
    assert tool._auto_redraw_dirty
    assert tool.preview_pixmap_stale

    tool._redraw_plot()

    assert draw_calls == [None, None, None, None, None]
    assert tool.figure.axes[0].images[0].get_cmap().name == "magma"
    assert not tool._auto_redraw_dirty
    assert not tool.preview_pixmap_stale


def test_figure_composer_tool_status_is_detached_from_live_recipe(
    qtbot, monkeypatch
) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(data)
    qtbot.addWidget(tool)
    tool.show_figure_window(activate=False)

    draw_calls: list[None] = []
    original_draw = tool.figure_window.canvas.draw

    def record_draw() -> None:
        draw_calls.append(None)
        original_draw()

    monkeypatch.setattr(tool.figure_window.canvas, "draw", record_draw)
    tool._redraw_plot(force=True)
    tool._redraw_plot()
    assert draw_calls == [None]

    updated = tool.tool_status
    updated.operations[0].extra_kwargs["alpha"] = 0.5
    tool._redraw_plot()
    assert draw_calls == [None]
    assert "alpha" not in tool.tool_status.operations[0].extra_kwargs

    source_snapshot = tool.source_states()[0]
    source_snapshot.qsel["x"] = 0.0
    assert tool.source_states()[0].qsel == {}

    tool.tool_status = updated
    assert draw_calls == [None, None]
    assert tool.tool_status.operations[0].extra_kwargs["alpha"] == 0.5

    assigned_revision = tool._document.recipe_revision
    assigned_figsize = tool.tool_status.setup.figsize
    updated.setup.figsize = (13.0, 9.0)
    updated.operations[0].extra_kwargs["alpha"] = 0.75
    assert tool.tool_status.setup.figsize == assigned_figsize
    assert tool.tool_status.operations[0].extra_kwargs["alpha"] == 0.5
    assert tool._document.recipe_revision == assigned_revision


def test_figure_composer_public_source_data_is_read_only(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
            "aux": (("x", "y"), np.arange(4.0).reshape(2, 2)),
        },
        attrs={"nested": {"value": 1}},
        name="data",
    )
    tool = FigureComposerTool(data)
    qtbot.addWidget(tool)

    primary = tool.tool_data
    named = tool.source_data()["data"]
    owned = tool._document.source_data["data"]
    assert np.shares_memory(primary.values, owned.values)
    assert np.shares_memory(named.values, owned.values)
    assert not np.shares_memory(owned.values, data.values)
    with pytest.raises(ValueError, match="read-only"):
        primary.values[:] = -1.0
    with pytest.raises(ValueError, match="read-only"):
        named.coords["aux"].values[:] = -1.0

    primary.attrs["nested"]["value"] = 2
    assert tool._document.source_data["data"].attrs["nested"]["value"] == 1

    revision = tool._document.source_revision
    with tool.edit_source_data() as sources:
        sources["data"].values[:] += 1.0
    assert tool._document.source_revision == revision + 1
    np.testing.assert_allclose(tool.tool_data.values, data.values + 1.0)
    np.testing.assert_array_equal(data.values, np.arange(4.0).reshape(2, 2))


def test_figure_composer_subplot_defaults_without_figure_window(qtbot) -> None:
    tool = FigureComposerTool(xr.DataArray(np.arange(2.0), dims="x", name="data"))
    qtbot.addWidget(tool)
    figure_window = tool._figure_window
    tool._figure_window = None
    try:
        defaults = figurecomposer_rendering._subplot_adjust_defaults(tool)
    finally:
        tool._figure_window = figure_window

    assert set(defaults) == {"left", "bottom", "right", "top", "wspace", "hspace"}
    assert all(np.isfinite(value) for value in defaults.values())


def test_figure_composer_preview_draw_error_ignores_missing_operation(qtbot) -> None:
    data = xr.DataArray(np.arange(2.0), dims=("x",), name="data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data"),),
            operations=(),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    figurecomposer_rendering._set_preview_draw_error(tool, RuntimeError("boom"))

    assert tool._operation_render_errors == {}


def test_figure_composer_pipeline_codegen_executes(qtbot) -> None:
    data = xr.DataArray(
        np.arange(12.0).reshape(3, 2, 2),
        dims=("eV", "kx", "ky"),
        coords={"eV": [0.0, 1.0, 2.0], "kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    profile = xr.DataArray(
        np.arange(2.0),
        dims=("kx",),
        coords={"kx": [0.0, 1.0]},
        name="profile",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=2),
            sources=(
                FigureSourceState(name="data", label="data"),
                FigureSourceState(name="profile", label="profile"),
            ),
            operations=(
                FigureOperationState.plot_slices(
                    label="left",
                    sources=("data",),
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                    slice_dim="eV",
                    slice_values=(0.0,),
                ),
                FigureOperationState.plot_slices(
                    label="right",
                    sources=("data",),
                    axes=FigureAxesSelectionState(axes=((0, 1),)),
                    slice_dim="eV",
                    slice_values=(1.0,),
                ),
                FigureOperationState.line(
                    label="profile",
                    source="profile",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ).model_copy(update={"line_x": "kx", "xlim": (0.25, 0.75)}),
                FigureOperationState.method(
                    family=FigureMethodFamily.ERLAB,
                    name="clean_labels",
                    axes=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
                ),
            ),
            primary_source="data",
        ),
        source_data={"data": data, "profile": profile},
    )
    qtbot.addWidget(tool)

    _select_operation_rows(tool, (2,))
    tool.operation_editor.select_section("selection")
    selection_page = tool.operation_editor.stack.currentWidget()
    profile_coordinate_combo = selection_page.findChild(
        QtWidgets.QComboBox, "figureComposerProfileCoordinateCombo"
    )
    profile_values_combo = selection_page.findChild(
        QtWidgets.QComboBox, "figureComposerProfileValuesCombo"
    )
    assert profile_coordinate_combo is not None
    assert profile_values_combo is not None
    assert profile_coordinate_combo.itemData(0) is None
    assert profile_values_combo.itemData(0) is None
    assert profile_coordinate_combo.findData("kx") >= 0
    assert profile_values_combo.findData("kx") >= 0
    _activate_combo_index(
        profile_coordinate_combo, profile_coordinate_combo.findData("kx")
    )
    assert tool.tool_status.operations[2].line_x == "kx"
    _activate_combo_index(profile_coordinate_combo, 0)
    assert tool.tool_status.operations[2].line_x is None
    _activate_combo_index(profile_values_combo, profile_values_combo.findData("kx"))
    assert tool.tool_status.operations[2].line_y == "kx"
    selection_page = tool.operation_editor.stack.currentWidget()
    profile_values_combo = selection_page.findChild(
        QtWidgets.QComboBox, "figureComposerProfileValuesCombo"
    )
    assert profile_values_combo is not None
    _activate_combo_index(profile_values_combo, 0)
    assert tool.tool_status.operations[2].line_y is None
    selection_page = tool.operation_editor.stack.currentWidget()
    profile_coordinate_combo = selection_page.findChild(
        QtWidgets.QComboBox, "figureComposerProfileCoordinateCombo"
    )
    profile_values_combo = selection_page.findChild(
        QtWidgets.QComboBox, "figureComposerProfileValuesCombo"
    )
    assert profile_coordinate_combo is not None
    assert profile_values_combo is not None
    assert profile_coordinate_combo.toolTip()
    assert profile_values_combo.toolTip()
    assert all(
        widget.toolTip() for widget in selection_page.findChildren(QtWidgets.QLineEdit)
    )
    assert all(
        widget.toolTip() for widget in selection_page.findChildren(QtWidgets.QCheckBox)
    )
    tool.operation_editor.select_section("view")
    view_page = tool.operation_editor.stack.currentWidget()
    data_values_axis_combo = view_page.findChild(
        QtWidgets.QComboBox, "figureComposerDataValuesAxisCombo"
    )
    assert data_values_axis_combo is not None
    assert data_values_axis_combo.toolTip()
    assert all(
        widget.toolTip() for widget in view_page.findChildren(QtWidgets.QLineEdit)
    )
    assert all(
        widget.toolTip() for widget in view_page.findChildren(QtWidgets.QCheckBox)
    )

    _select_operation_rows(tool, (3,))
    assert tool.tool_status.operations[3].method_name == "clean_labels"
    tool.operation_editor.select_section("method")
    erlab_method_page = tool.operation_editor.stack.currentWidget()
    assert all(
        widget.toolTip()
        for widget in erlab_method_page.findChildren(QtWidgets.QComboBox)
    )
    assert all(
        widget.toolTip()
        for widget in erlab_method_page.findChildren(QtWidgets.QPlainTextEdit)
    )
    assert all(
        widget.toolTip()
        for widget in erlab_method_page.findChildren(QtWidgets.QLineEdit)
    )

    namespace = {"data": data, "profile": profile}
    exec(tool.generated_code(), namespace)  # noqa: S102
    assert namespace["axs"].shape == (1, 2)
    assert namespace["axs"][0, 0].get_xlim() == pytest.approx((0.25, 0.75))


def test_figure_composer_generated_code_skips_disabled_operations(qtbot) -> None:
    data = xr.DataArray(np.arange(3.0), dims=("x",), name="data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data"),),
            operations=(
                FigureOperationState.custom(
                    label="disabled Python",
                    code="raise RuntimeError('must not run')",
                    trusted=True,
                ).model_copy(update={"enabled": False}),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    code = tool.generated_code()

    assert "must not run" not in code
