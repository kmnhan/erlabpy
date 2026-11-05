from __future__ import annotations

import subprocess
import sys
import textwrap
from dataclasses import dataclass

import numpy as np
import pytest
import xarray as xr

from erlab._provenance import Follow, ProvenanceGraph, Recipe


@dataclass(frozen=True)
class Scale:
    factor: int

    def input_slots(self):
        return ()

    def apply(self, data):
        return self.factor * data


class Add:
    def input_slots(self):
        return ("other",)

    def apply_with_inputs(self, data, inputs, *, authorization=None):
        return data + inputs["other"]


@pytest.fixture
def graph() -> ProvenanceGraph:
    return ProvenanceGraph()


def test_replace_preserves_recorded_input_until_refresh(graph) -> None:
    b_buffer = np.array([1, 2])
    a0 = graph.add_source("A", xr.DataArray([10, 20]))
    b0 = graph.add_source("B", xr.DataArray(b_buffer))
    c0 = graph.derive("C", Recipe.bind(Scale(factor=2), Follow("B")), auto=False)

    def check(entry_id, revision_id, values):
        assert graph.entry(entry_id).entry_id == entry_id
        assert graph.entry(entry_id).revision_id == revision_id
        xr.testing.assert_identical(graph.data(revision_id), xr.DataArray(values))

    check("A", a0, [10, 20])
    check("B", b0, [1, 2])
    check("C", c0, [2, 4])
    assert graph.revision(c0).producer.inputs == (("data", b0),)
    assert graph.next_inputs("C") == (("data", b0),)
    assert not graph.is_stale("C")

    b1 = graph.replace("B", Recipe.bind(Add(), Follow("B"), other=Follow("A")))
    assert b1 != b0
    check("A", a0, [10, 20])
    check("B", b1, [11, 22])
    check("C", c0, [2, 4])
    assert graph.entry_ids == ("A", "B", "C")
    assert graph.revision(b1).producer.inputs == (("data", b0), ("other", a0))
    assert graph.revision(c0).producer.inputs == (("data", b0),)
    assert graph.next_inputs("C") == (("data", b1),)
    assert graph.is_stale("C")
    assert not graph.entry("C").auto
    b_buffer[:] = -1
    assert graph.is_virtual(b0)
    xr.testing.assert_identical(graph.data(b0), xr.DataArray([1, 2]))

    c1 = graph.refresh("C")
    assert c1 != c0
    check("A", a0, [10, 20])
    check("B", b1, [11, 22])
    check("C", c1, [22, 44])
    assert graph.entry_ids == ("A", "B", "C")
    assert graph.revision(c1).producer.inputs == (("data", b1),)
    assert graph.next_inputs("C") == (("data", b1),)
    assert not graph.is_stale("C")


def test_replacing_a_derived_entry_reapplies_its_previous_step(graph) -> None:
    from erlab._provenance import EarlierStep
    from erlab.interactive.imagetool._provenance._operations import IselOperation

    a0 = graph.add_source("A", xr.DataArray([1, 2, 3], dims="x"))
    b0 = graph.derive(
        "B", Recipe.bind(IselOperation(kwargs={"x": slice(1, None)}), Follow("A"))
    )
    b1 = graph.replace("B", Recipe.bind(Scale(factor=2), Follow("B")))
    assert graph.revision(b1).producer.inputs == (("data", b0),)
    assert graph.next_inputs("B") == (("data", EarlierStep(b0)),)
    a1 = graph.replace("A", Recipe.bind(Scale(factor=10), Follow("A")))
    assert graph.is_stale("B")

    b2 = graph.refresh("B")
    ((_, crop),) = graph.revision(b2).producer.inputs
    assert crop != b0
    assert graph.is_virtual(crop)
    assert graph.revision(crop).producer.inputs == (("data", a1),)
    assert graph.revision(b0).producer.inputs == (("data", a0),)
    xr.testing.assert_identical(graph.data(b0), xr.DataArray([2, 3], dims="x"))
    xr.testing.assert_identical(graph.data(b2), xr.DataArray([40, 60], dims="x"))
    assert not graph.is_stale("B")


def test_failed_refresh_keeps_displayed_data_and_prerequisites(graph) -> None:
    from erlab.interactive.imagetool._provenance._operations import IselOperation

    graph.add_source("A", xr.DataArray([1, 2], dims="x"))
    b0 = graph.derive("B", Recipe.bind(Scale(factor=2), Follow("A")))
    c0 = graph.derive("C", Recipe.bind(Scale(factor=3), Follow("B")))
    graph.replace("A", Recipe.bind(Scale(factor=10), Follow("A")))
    graph.set_recipe("C", Recipe.bind(IselOperation(kwargs={"x": 5}), Follow("B")))
    before = graph.revision_ids
    with pytest.raises(IndexError):
        graph.refresh("C")

    assert graph.revision_ids == before
    assert graph.entry("B").revision_id == b0
    assert graph.is_stale("B")
    assert graph.entry("C").revision_id == c0
    assert graph.is_stale("C")
    assert graph.revision(c0).producer.operation.factor == 3
    xr.testing.assert_identical(graph.data(c0), xr.DataArray([6, 12], dims="x"))
    graph.set_recipe("C", Recipe.bind(Scale(factor=4), Follow("B")))
    c1 = graph.refresh("C")
    assert graph.revision(c1).producer.inputs == (
        ("data", graph.entry("B").revision_id),
    )
    xr.testing.assert_identical(graph.data(c1), xr.DataArray([80, 160], dims="x"))


def test_shared_file_input_is_read_once_per_refresh(tmp_path) -> None:
    from erlab._provenance import FileLoad
    from erlab.interactive.imagetool._provenance._model import (
        FileDataSelection,
        FileReplayCall,
    )

    path = tmp_path / "source.nc"
    xr.Dataset({"first": ("x", [1, 2]), "second": ("x", [10, 20])}).to_netcdf(
        path, engine="h5netcdf"
    )
    reads = []

    def read_file(source, call):
        reads.append((source.path, call.selection.value))
        return xr.load_dataset(source.path, **call.kwargs)[call.selection.value]

    graph = ProvenanceGraph(read_file=read_file)
    source = graph.file_source(str(path))
    first = FileLoad(
        source,
        FileReplayCall(
            kind="callable",
            target="xarray.load_dataset",
            kwargs={"engine": "h5netcdf"},
            selection=FileDataSelection(kind="dataset_variable", value="first"),
        ),
    )
    f0 = graph.derive("F", first)
    graph.derive("L", Recipe.bind(Scale(factor=2), Follow("F")))
    graph.derive("R", Recipe.bind(Scale(factor=3), Follow("F")))
    graph.derive("C", Recipe.bind(Add(), Follow("L"), other=Follow("R")))
    assert graph.revision(f0).producer.inputs == (("file", source.source_id),)
    assert graph.revision(f0).producer.call == first.call

    second = FileLoad(
        source,
        first.call.model_copy(
            update={
                "selection": FileDataSelection(kind="dataset_variable", value="second")
            }
        ),
    )
    graph.set_recipe("F", second)
    assert graph.is_stale("C")
    assert graph.next_inputs("C") == (("data", Follow("L")), ("other", Follow("R")))
    c1 = graph.refresh("C")
    f1 = graph.entry("F").revision_id
    assert f0 != f1
    assert reads == [(str(path), "first"), (str(path), "second")]
    assert graph.revision(f1).producer.call == second.call
    for name in ["L", "R"]:
        assert graph.revision(graph.entry(name).revision_id).producer.inputs == (
            ("data", f1),
        )
    xr.testing.assert_identical(
        graph.data(c1), xr.DataArray([50, 100], dims="x", name="second")
    )


def test_ordered_slots_and_applied_code_with_repeated_input(graph) -> None:
    from erlab.interactive.imagetool._provenance._operations import ConcatOperation

    left = xr.DataArray([1, 2], dims="x")
    right = xr.DataArray([10, 20], dims="x")
    a = graph.add_source("A", left)
    b = graph.add_source("B", right)
    op = ConcatOperation(dim="scan", input_count=3)
    result = graph.derive(
        "C", Recipe.bind(op, Follow("B"), other_1=Follow("B"), other_0=Follow("A"))
    )
    applied = graph.revision(result).producer
    assert applied.inputs == (("data", b), ("other_0", a), ("other_1", b))
    ns = {"xr": xr, "left": left, "right": right}
    code = applied.code({a: "left", b: "right"}, output_name="combined")
    exec(code, ns)  # noqa: S102
    expected = xr.concat([right, left, right], dim="scan")
    xr.testing.assert_identical(ns["combined"], expected)
    xr.testing.assert_identical(graph.data(result), expected)


def test_cycle_checks_apply_to_followed_entries_only(graph) -> None:
    from erlab._provenance import Pin

    a0 = graph.add_source("A", xr.DataArray([1]))
    b0 = graph.derive("B", Recipe.bind(Scale(factor=2), Follow("A")), parent_id="A")
    with pytest.raises(ValueError, match="cycle"):
        graph.replace("A", Recipe.bind(Add(), Follow("A"), other=Follow("B")))
    with pytest.raises(ValueError, match="cycle"):
        graph.set_recipe("A", Recipe.bind(Scale(factor=2), Follow("A")))
    assert graph.entry("A").revision_id == a0
    assert not graph.is_stale("B")

    a1 = graph.replace("A", Recipe.bind(Add(), Follow("A"), other=Pin(b0)))
    assert graph.revision(a1).producer.inputs == (("data", a0), ("other", b0))
    graph.set_auto("B", enabled=True)
    assert graph.entry("B").auto
    assert graph.entry("B").parent_id == "A"
    assert graph.is_stale("B")
    xr.testing.assert_identical(graph.data(graph.refresh("B")), xr.DataArray([6]))


def test_invalid_ids_and_slots_are_rejected_before_publication(graph) -> None:
    from erlab._provenance import Pin

    a = graph.add_source("A", xr.DataArray([1]))
    with pytest.raises(ValueError, match="already exists"):
        graph.add_source("A", xr.DataArray([2]))
    with pytest.raises(ValueError, match="Invalid provenance ID"):
        graph.add_source("../bad", xr.DataArray([2]))
    with pytest.raises(KeyError, match="unknown"):
        graph.derive("C", Recipe.bind(Add(), Pin("unknown"), other=Follow("A")))
    with pytest.raises(KeyError, match="absent"):
        graph.derive("C", Recipe.bind(Scale(factor=2), Follow("absent")))
    with pytest.raises(ValueError, match="slots"):
        Recipe.bind(Add(), Follow("A"), target=Pin(a))
    with pytest.raises(ValueError, match="ordered"):
        Recipe(Add(), (("other", Pin(a)), ("data", Follow("A"))))
    with pytest.raises(KeyError, match="C"):
        graph.entry("C")


def test_checkpoints_and_nested_operation_settings_are_owned(graph) -> None:
    from erlab.interactive.imagetool._provenance._operations import IselOperation

    original = xr.DataArray([10, 20], dims="x", attrs={"labels": ["source"]})
    a = graph.add_source("A", original)
    op = IselOperation(kwargs={"x": [1, 0]})
    b = graph.derive("B", Recipe.bind(op, Follow("A")))
    original.attrs["labels"].clear()
    original.values[:] = -1
    graph.data(a).values[:] = -2
    op.kwargs["x"].reverse()
    graph.revision(b).producer.operation.kwargs["x"].clear()
    graph.entry("B").recipe.operation.kwargs["x"].clear()
    xr.testing.assert_identical(
        graph.data(graph.refresh("B")),
        xr.DataArray([20, 10], dims="x", attrs={"labels": ["source"]}),
    )


def test_acceptance_path_imports_no_qt_or_manager() -> None:
    code = textwrap.dedent("""
        import runpy
        import sys

        class BlockQt:
            def find_spec(self, fullname, path, target=None):
                blocked = {"qtpy", "PyQt6", "PySide6", "pyqtgraph"}
                if fullname.partition(".")[0] in blocked:
                    raise AssertionError(f"Unexpected Qt import: {fullname}")
                return None

        sys.meta_path.insert(0, BlockQt())
        suite = runpy.run_path(sys.argv[1])
        suite["test_replace_preserves_recorded_input_until_refresh"](suite["ProvenanceGraph"]())
        assert "erlab.interactive.imagetool.manager" not in sys.modules
    """)
    subprocess.run([sys.executable, "-c", code, __file__], check=True)


def test_status_and_provenance_queries_preserve_lazy_data(graph) -> None:
    import dask.array as da
    from dask import delayed

    calls = []

    class LazyDouble(Scale):
        def apply(self, data):
            @delayed
            def read():
                calls.append(True)
                return data.values * self.factor

            return data.copy(
                data=da.from_delayed(read(), shape=data.shape, dtype=data.dtype)
            )

    a0 = graph.add_source("A", xr.DataArray([1, 2]))
    b0 = graph.derive("B", Recipe.bind(LazyDouble(2), Follow("A")))
    a1 = graph.replace("A", Recipe.bind(Scale(3), Follow("A")))
    assert graph.revision(b0).producer.inputs == (("data", a0),)
    assert graph.next_inputs("B") == (("data", a1),)
    assert graph.is_stale("B")
    assert graph.data(b0).chunks is not None
    assert calls == []
    np.testing.assert_array_equal(graph.data(b0).values, [2, 4])
    assert calls == [True]


def test_chain_and_diamond_visit_only_affected_entries(graph, monkeypatch) -> None:
    from collections import Counter

    from erlab._provenance import Pin

    visits = []
    calls = []

    class CountScale(Scale):
        def apply(self, data):
            calls.append(self.factor)
            return super().apply(data)

    # Long enough to catch a recursive graph walk.
    length = 1100
    graph.add_source("A", xr.DataArray(1))
    parent = "A"
    for index in range(length):
        child = f"chain{index}"
        graph.derive(child, Recipe.bind(CountScale(1), Follow(parent)))
        parent = child
    graph.derive("L", Recipe.bind(CountScale(2), Follow(parent)))
    graph.derive("R", Recipe.bind(CountScale(3), Follow(parent)))
    graph.derive("C", Recipe.bind(Add(), Follow("L"), other=Follow("R")))

    other = graph.add_source("other", xr.DataArray(90))
    graph.derive("independent", Recipe.bind(CountScale(7), Follow("other")))
    graph.derive("pinned", Recipe.bind(CountScale(7), Pin(other)))
    calls.clear()
    plan_input = graph._plan_input

    def visit(binding, *, refresh):
        visits.append(binding)
        return plan_input(binding, refresh=refresh)

    monkeypatch.setattr(graph, "_plan_input", visit)
    followers_read = []

    class CountFollowers(dict):
        def get(self, entry_id, default=None):
            followers_read.append(entry_id)
            return super().get(entry_id, default)

    monkeypatch.setattr(graph, "_followers", CountFollowers(graph._followers))
    a1 = graph.replace("A", Recipe.bind(CountScale(4), Follow("A")))
    assert graph.is_stale("C")
    assert not graph.is_stale("independent")
    assert not graph.is_stale("pinned")
    affected = {"C", "L", "R", *(f"chain{index}" for index in range(length))}
    assert Counter(followers_read) == dict.fromkeys(affected | {"A"}, 1)
    visits.clear()
    followers_read.clear()
    c1 = graph.refresh("C")
    assert Counter(followers_read) == dict.fromkeys(affected, 1)
    assert Counter(calls) == {4: 1, 1: length, 2: 1, 3: 1}
    assert len(visits) == len(set(visits)) == length + 4
    assert set(visits) == {
        Follow("A"),
        Pin(a1),
        Follow("L"),
        Follow("R"),
        *(Follow(f"chain{index}") for index in range(length)),
    }
    xr.testing.assert_identical(graph.data(c1), xr.DataArray(20))


def test_superseded_calculation_cannot_publish(graph) -> None:
    from erlab._provenance import Pin

    a0 = graph.add_source("A", xr.DataArray([1]))

    class ReplaceInputDuringApply(Scale):
        def apply(self, data):
            graph.replace("A", Recipe.bind(Scale(10), Pin(a0)))
            return super().apply(data)

    with pytest.raises(RuntimeError, match="changed during evaluation"):
        graph.derive("B", Recipe.bind(ReplaceInputDuringApply(2), Follow("A")))
    with pytest.raises(KeyError, match="B"):
        graph.entry("B")
    xr.testing.assert_identical(
        graph.data(graph.entry("A").revision_id), xr.DataArray([10])
    )


def test_only_owner_can_publish(graph) -> None:
    from concurrent.futures import ThreadPoolExecutor

    a = graph.add_source("A", xr.DataArray(1))
    with ThreadPoolExecutor(max_workers=1) as pool:
        attempt = pool.submit(graph.replace, "A", Recipe.bind(Scale(2), Follow("A")))
        with pytest.raises(RuntimeError, match="owner thread"):
            attempt.result()
    assert graph.entry("A").revision_id == a


def test_editing_a_binding_keeps_the_recorded_step_and_pinned_input(graph) -> None:
    from erlab._provenance import Pin

    a = graph.add_source("A", xr.DataArray(1))
    graph.add_source("B", xr.DataArray(10))
    b = graph.derive("B view", Recipe.bind(Scale(2), Follow("B")))
    c = graph.derive("C", Recipe.bind(Scale(3), Follow("A")), parent_id="A")
    graph.set_recipe("C", Recipe.bind(Scale(5), Pin(b)))
    graph.replace("B view", Recipe.bind(Scale(4), Follow("B")))
    assert graph.revision(c).producer.inputs == (("data", a),)
    assert graph.next_inputs("C") == (("data", b),)
    updated = graph.refresh("C")
    xr.testing.assert_identical(graph.data(updated), xr.DataArray(100))
    graph.replace("A", Recipe.bind(Scale(10), Follow("A")))
    assert not graph.is_stale("C")
    assert graph.entry("C").parent_id == "A"


def test_replacing_a_stale_derived_value_records_what_was_on_screen(graph) -> None:
    graph.add_source("A", xr.DataArray([1]))
    b0 = graph.derive("B", Recipe.bind(Scale(2), Follow("A")))
    graph.replace("A", Recipe.bind(Scale(10), Follow("A")))
    b1 = graph.replace("B", Recipe.bind(Scale(3), Follow("B")))
    assert graph.revision(b1).producer.inputs == (("data", b0),)
    xr.testing.assert_identical(graph.data(b1), xr.DataArray([6]))
    assert graph.is_stale("B")
    xr.testing.assert_identical(graph.data(graph.refresh("B")), xr.DataArray([60]))


def test_retained_history_and_stale_results_own_file_checkpoints() -> None:
    from erlab._provenance import FileLoad
    from erlab.interactive.imagetool._provenance._model import (
        FileDataSelection,
        FileReplayCall,
    )

    values = iter(range(5))
    graph = ProvenanceGraph(read_file=lambda _source, _call: xr.DataArray(next(values)))
    file_recipe = FileLoad(
        graph.file_source("input"),
        FileReplayCall(
            kind="callable",
            target="xarray.load_dataarray",
            selection=FileDataSelection(kind="dataarray"),
        ),
    )
    f0 = graph.derive("F", file_recipe)
    c0 = graph.derive("C", Recipe.bind(Scale(2), Follow("F")))
    with graph.retain(c0):
        for _ in range(4):
            graph.refresh("F")
            assert graph.is_stale("C")
            graph.refresh("C")
        xr.testing.assert_identical(graph.data(c0), xr.DataArray(0))
        xr.testing.assert_identical(graph.data(f0), xr.DataArray(0))
        assert len(graph.revision_ids) == 4
    assert len(graph.revision_ids) == 2
    assert sum(graph.revision(node).checkpointed for node in graph.revision_ids) == 1


@pytest.mark.parametrize("derived", [True, False])
def test_replace_keeps_pending_producer_and_its_pinned_input(graph, derived) -> None:
    from erlab._provenance import Pin

    a = graph.add_source("A", xr.DataArray(1))
    pin = graph.derive("input", Recipe.bind(Scale(5), Follow("A")))
    if derived:
        b0 = graph.derive("B", Recipe.bind(Scale(2), Follow("A")))
    else:
        b0 = graph.add_source("B", xr.DataArray(2))
    graph.set_recipe("B", Recipe.bind(Scale(3), Pin(pin)))
    graph.set_recipe("input", Recipe.bind(Scale(7), Follow("A")))
    graph.refresh("input")

    b1 = graph.replace("B", Recipe.bind(Scale(4), Follow("B")))
    assert graph.revision(b1).producer.inputs == (("data", b0),)
    xr.testing.assert_identical(graph.data(b1), xr.DataArray(8))
    assert graph.is_stale("B")
    # A second Replace must also consume what is on screen.
    b2 = graph.replace("B", Recipe.bind(Scale(2), Follow("B")))
    xr.testing.assert_identical(graph.data(b2), xr.DataArray(16))
    assert graph.is_stale("B")

    refreshed = graph.refresh("B")
    xr.testing.assert_identical(graph.data(refreshed), xr.DataArray(120))
    assert not graph.is_stale("B")
    ((_, middle),) = graph.revision(refreshed).producer.inputs
    ((_, earlier),) = graph.revision(middle).producer.inputs
    assert graph.revision(earlier).producer.inputs == (("data", pin),)
    assert graph.revision(pin).producer.inputs == (("data", a),)
    assert graph.revision(b1).producer.inputs == (("data", b0),)
    graph.replace("B", Recipe.bind(Scale(1), Follow("B")))
    assert not graph.is_stale("B")


def test_replaced_entry_follows_the_edited_source(graph) -> None:
    from erlab.interactive.imagetool._provenance._operations import IselOperation

    graph.add_source("A", xr.DataArray([1, 2], dims="x"))
    graph.add_source("X", xr.DataArray([10, 20], dims="x"))
    b0 = graph.derive("B", Recipe.bind(IselOperation(kwargs={"x": 0}), Follow("A")))
    graph.set_recipe("B", Recipe.bind(IselOperation(kwargs={"x": 1}), Follow("X")))
    b1 = graph.replace("B", Recipe.bind(Scale(2), Follow("B")))
    assert graph.is_stale("B")
    assert graph.revision(b1).producer.inputs == (("data", b0),)
    xr.testing.assert_identical(graph.data(b1), xr.DataArray(2))

    # Inspecting the pending edit must not expose the graph's operation parameters.
    ((_, earlier),) = graph.next_inputs("B")
    earlier.pending.recipe.operation.kwargs["x"] = 0
    xr.testing.assert_identical(graph.data(graph.refresh("B")), xr.DataArray(40))
    graph.replace("A", Recipe.bind(Scale(2), Follow("A")))
    assert not graph.is_stale("B")
    graph.replace("X", Recipe.bind(Scale(3), Follow("X")))
    assert graph.is_stale("B")
    xr.testing.assert_identical(graph.data(graph.refresh("B")), xr.DataArray(120))
