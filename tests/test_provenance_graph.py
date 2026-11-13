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
    assert not graph.is_stale("B")
    xr.testing.assert_identical(
        graph.data(graph.entry("B").revision_id), xr.DataArray([6])
    )


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
        suite["test_reload_from_memory_keeps_the_source_checkpoint"](suite["ProvenanceGraph"]())
        suite["test_auto_coalesces_updates_and_waits_for_auto_off_prerequisites"]()
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
    assert graph.plan_update("B", command="reload").entries == ("A", "B")
    assert graph.recorded_lineage(b0)[0].revision_id == a0
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

    def visit(binding, **kwargs):
        visits.append(binding)
        return plan_input(binding, **kwargs)

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


@pytest.fixture
def file_graph(tmp_path):
    from erlab._provenance import FileLoad
    from erlab.interactive.imagetool._provenance._model import (
        FileDataSelection,
        FileReplayCall,
    )

    path = tmp_path / "scan.nc"
    reads = []

    def read(source, call):
        reads.append(source.path)
        return xr.load_dataarray(source.path, **call.kwargs)

    graph = ProvenanceGraph(read_file=read)
    load = FileLoad(
        graph.file_source(str(path)),
        FileReplayCall(
            kind="callable",
            target="xarray.load_dataarray",
            kwargs={"engine": "h5netcdf"},
            selection=FileDataSelection(kind="dataarray"),
        ),
    )
    xr.DataArray([1, 2]).to_netcdf(path, engine="h5netcdf")
    return graph, path, load, reads


@pytest.mark.parametrize("command", ["refresh", "reload"])
def test_stale_result_uses_the_inputs_shown_in_its_command_plan(file_graph, command):
    graph, path, load, reads = file_graph
    a = graph.add_source("A", xr.DataArray([10, 20]))
    b0 = graph.derive("B", load)
    c0 = graph.derive("C", Recipe.bind(Scale(2), Follow("B")))
    b1 = graph.replace("B", Recipe.bind(Add(), Follow("B"), other=Follow("A")))
    sibling = graph.derive("sibling", Recipe.bind(Scale(5), Follow("B")))
    xr.DataArray([3, 4]).to_netcdf(path, engine="h5netcdf")

    with graph.retain(c0):
        plan = graph.plan_update("C", command=command)
        assert graph.revision(c0).producer.inputs == (("data", b0),)
        assert graph.next_inputs("C") == (("data", b1),)
        assert graph.is_stale("C")
        assert reads == [str(path)]
        if command == "reload":
            assert plan.entries == ("B", "C")
            assert plan.affected_followers == {"sibling"}
            assert plan.file_loads == (load,)
            assert plan.current_inputs == (("A", a),)
            c1 = graph.reload("C")
            b2 = graph.entry("B").revision_id
            assert b2 != b1
            ((_, original), (_, used_a)) = graph.revision(b2).producer.inputs
            assert used_a == a
            assert original != b0
            assert graph.revision(original).producer == load
            xr.testing.assert_identical(graph.data(b2), xr.DataArray([13, 24]))
            xr.testing.assert_identical(graph.data(c1), xr.DataArray([26, 48]))
            assert reads == [str(path), str(path)]
            assert graph.is_stale("sibling")
        else:
            assert plan.entries == ("C",)
            assert not plan.affected_followers
            assert plan.file_loads == ()
            assert plan.current_inputs == (("B", b1),)
            c1 = graph.refresh("C")
            assert graph.entry("B").revision_id == b1
            xr.testing.assert_identical(graph.data(c1), xr.DataArray([22, 44]))
            assert reads == [str(path)]
            assert not graph.is_stale("sibling")
        assert graph.entry("sibling").revision_id == sibling
        assert graph.revision(c1).producer.inputs == (
            ("data", graph.entry("B").revision_id),
        )
        assert not graph.is_stale("C")
        xr.testing.assert_identical(graph.data(c0), xr.DataArray([2, 4]))
        xr.testing.assert_identical(graph.data(b0), xr.DataArray([1, 2]))


def test_reload_shares_live_producer_and_preserves_pinned_file_revision(file_graph):
    from erlab._provenance import Pin

    graph, path, load, reads = file_graph
    f0 = graph.derive("F", load)
    graph.derive("L", Recipe.bind(Scale(2), Follow("F")))
    graph.derive("R", Recipe.bind(Add(), Follow("F"), other=Pin(f0)))
    graph.derive("C", Recipe.bind(Add(), Follow("L"), other=Follow("R")))
    p = graph.derive("P", Recipe.bind(Scale(5), Pin(f0)))
    xr.DataArray([10, 20]).to_netcdf(path, engine="h5netcdf")

    assert graph.plan_update("C", command="reload").pinned_inputs == (f0,)
    for request in range(2):
        c = graph.reload("C")
        assert len(reads) == 2 + request
        f = graph.entry("F").revision_id
        assert f != f0
        assert graph.revision(graph.entry("R").revision_id).producer.inputs == (
            ("data", f),
            ("other", f0),
        )
        assert graph.revision(graph.entry("L").revision_id).producer.inputs == (
            ("data", f),
        )
        xr.testing.assert_identical(graph.data(c), xr.DataArray([31, 62]))
    assert graph.entry("P").revision_id == p
    assert not graph.is_stale("P")
    assert graph.plan_update("P", command="reload").file_loads == ()
    graph.reload("P")
    assert len(reads) == 3


def test_missing_file_leaves_recorded_result_available(file_graph):
    from erlab._provenance import MissingFileSource

    graph, path, load, _ = file_graph
    f = graph.derive("F", load)
    c = graph.derive("C", Recipe.bind(Scale(2), Follow("F")))
    graph.set_recipe("F", load)
    path.unlink()
    with pytest.raises(MissingFileSource) as error:
        graph.reload("C")
    assert error.value.source == load.source
    assert error.value.displayed_revision_id == c
    assert str(path) in str(error.value)
    assert graph.entry("F").revision_id == f
    assert graph.entry("C").revision_id == c
    assert graph.is_stale("F")
    assert graph.is_stale("C")
    xr.testing.assert_identical(graph.data(c), xr.DataArray([2, 4]))


def test_cancelled_reload_discards_already_prepared_prerequisite(file_graph):
    from concurrent.futures import CancelledError

    graph, path, load, reads = file_graph
    f0 = graph.derive("F", load)
    c0 = graph.derive("C", Recipe.bind(Scale(2), Follow("F")))
    graph.set_recipe("F", load)
    xr.DataArray([10, 20]).to_netcdf(path, engine="h5netcdf")
    before = graph.revision_ids
    with pytest.raises(CancelledError):
        graph.reload("C", is_cancelled=lambda: len(reads) > 1)

    assert len(reads) == 2
    assert graph.revision_ids == before
    assert graph.entry("F").revision_id == f0
    assert graph.entry("C").revision_id == c0
    assert graph.is_stale("F")
    assert graph.is_stale("C")
    c1 = graph.reload("C")
    xr.testing.assert_identical(graph.data(c1), xr.DataArray([20, 40]))


def test_cancellation_after_last_calculation_keeps_displayed_value(graph):
    from concurrent.futures import CancelledError

    cancelled = False

    class CancelDuringApply(Scale):
        def apply(self, data):
            nonlocal cancelled
            cancelled = True
            return super().apply(data)

    a0 = graph.add_source("A", xr.DataArray(1))
    with pytest.raises(CancelledError):
        graph.replace(
            "A",
            Recipe.bind(CancelDuringApply(2), Follow("A")),
            is_cancelled=lambda: cancelled,
        )
    assert graph.revision_ids == {a0}
    assert graph.entry("A").revision_id == a0


def test_unrelated_publication_during_calculation_is_allowed(graph):
    class UpdateOther(Scale):
        def apply(self, data):
            graph.replace("other", Recipe.bind(Scale(5), Follow("other")))
            return super().apply(data)

    graph.add_source("A", xr.DataArray(1))
    graph.add_source("other", xr.DataArray(10))
    b = graph.derive("B", Recipe.bind(UpdateOther(2), Follow("A")))
    xr.testing.assert_identical(graph.data(b), xr.DataArray(2))
    xr.testing.assert_identical(
        graph.data(graph.entry("other").revision_id), xr.DataArray(50)
    )


def test_auto_coalesces_updates_and_waits_for_auto_off_prerequisites():
    callbacks = []
    calls = []

    class Count(Scale):
        def apply(self, data):
            calls.append(self.factor)
            return super().apply(data)

    graph = ProvenanceGraph(schedule_auto=callbacks.append)
    graph.add_source("A", xr.DataArray(1))
    b0 = graph.derive("B", Recipe.bind(Count(2), Follow("A")), auto=True)
    c0 = graph.derive("C", Recipe.bind(Count(3), Follow("B")), auto=True)
    graph.derive("D", Recipe.bind(Count(4), Follow("B")), auto=True)
    graph.derive("E", Recipe.bind(Add(), Follow("C"), other=Follow("D")), auto=True)
    calls.clear()

    graph.replace("A", Recipe.bind(Scale(10), Follow("A")))
    graph.replace("A", Recipe.bind(Scale(2), Follow("A")))
    assert len(callbacks) == 1
    assert not calls
    assert graph.entry("B").revision_id == b0
    graph.set_auto("B", enabled=False)
    callbacks.pop()()
    assert graph.entry("B").revision_id == b0
    assert graph.entry("C").revision_id == c0
    assert graph.is_stale("E")
    assert graph.auto_blockers("C") == {"B"}
    assert not calls
    assert not callbacks

    # Explicit Refresh of B makes its Auto-enabled followers eligible.
    graph.refresh("B")
    assert len(callbacks) == 1
    callbacks.pop()()
    assert not callbacks
    assert sorted(calls) == [2, 3, 4]
    xr.testing.assert_identical(
        graph.data(graph.entry("E").revision_id), xr.DataArray(280)
    )
    assert not graph.is_stale("E")


@pytest.mark.parametrize("drain_while_paused", [False, True])
def test_auto_pause_retains_latest_input_and_resumes_once(drain_while_paused):
    callbacks = []
    graph = ProvenanceGraph(schedule_auto=callbacks.append)
    graph.add_source("A", xr.DataArray(1))
    b0 = graph.derive("B", Recipe.bind(Scale(2), Follow("A")), auto=True)
    graph.replace("A", Recipe.bind(Scale(10), Follow("A")))
    assert len(callbacks) == 1

    graph.set_auto_paused(True)
    if drain_while_paused:
        callbacks.pop()()
    graph.replace("A", Recipe.bind(Scale(2), Follow("A")))
    assert len(callbacks) == (0 if drain_while_paused else 1)
    assert graph.current_revision("B") == b0
    assert graph.is_stale("B")

    graph.set_auto_paused(False)
    assert len(callbacks) == 1
    callbacks.pop()()
    assert not callbacks
    assert not graph.is_stale("B")
    xr.testing.assert_identical(
        graph.data(graph.current_revision("B")), xr.DataArray(40)
    )
    graph.replace("A", Recipe.bind(Scale(3), Follow("A")))
    assert len(callbacks) == 1
    callbacks.pop()()
    xr.testing.assert_identical(
        graph.data(graph.current_revision("B")), xr.DataArray(120)
    )


def test_checkpoint_reader_owns_sealed_values_after_entry_removal():
    graph = ProvenanceGraph()
    data = xr.DataArray([1.0, 2.0], dims="x")
    revision = graph.prepare_import("A", data, (), live=True)()
    assert graph.checkpoint_reader(revision) is None
    graph.derive("B", Recipe.bind(Scale(2), Follow("A")))
    reader = graph.checkpoint_reader(revision)
    assert reader is not None
    graph.remove_entry("B")
    graph.remove_entry("A")
    data.values[:] = 99
    first = reader()
    xr.testing.assert_identical(first, xr.DataArray([1.0, 2.0], dims="x"))
    first.values[:] = -1
    xr.testing.assert_identical(reader(), xr.DataArray([1.0, 2.0], dims="x"))


def test_auto_failure_is_isolated_and_retried_on_next_change(graph):
    class FailOnTwo(Scale):
        def apply(self, data):
            if data.item() == 2:
                raise ValueError("input equals two")
            return super().apply(data)

    graph.add_source("A", xr.DataArray(1))
    b0 = graph.derive("B", Recipe.bind(FailOnTwo(2), Follow("A")), auto=True)
    c0 = graph.derive("C", Recipe.bind(Scale(3), Follow("B")), auto=True)
    d0 = graph.derive("D", Recipe.bind(Scale(4), Follow("A")), auto=True)
    a1 = graph.replace("A", Recipe.bind(Scale(2), Follow("A")))
    assert graph.entry("A").revision_id == a1
    assert graph.entry("B").revision_id == b0
    assert graph.entry("C").revision_id == c0
    assert graph.is_stale("B")
    assert graph.is_stale("C")
    assert graph.entry("D").revision_id != d0
    assert graph.auto_error("B") == "input equals two"
    graph.replace("A", Recipe.bind(Scale(2), Follow("A")))
    assert graph.auto_error("B") is None
    assert not graph.is_stale("C")
    xr.testing.assert_identical(
        graph.data(graph.entry("C").revision_id), xr.DataArray(24)
    )


def test_full_code_uses_recorded_inputs_and_verified_file_or_checkpoint(file_graph):
    import hashlib

    from erlab.interactive.imagetool._provenance._graph import ReplayGraphError
    from erlab.interactive.imagetool._provenance._operations import (
        ConcatOperation,
        IselOperation,
    )
    from erlab.interactive.imagetool._provenance._recorded_code import full_code

    graph, path, load, reads = file_graph
    original_hash = hashlib.sha256(path.read_bytes()).digest()
    f0 = graph.derive("F", load)
    a0 = graph.add_source("A", xr.DataArray([10, 20]))
    b0 = graph.derive(
        "B",
        Recipe.bind(
            ConcatOperation(dim="scan", input_count=3),
            Follow("F"),
            other_0=Follow("A"),
            other_1=Follow("F"),
        ),
    )
    c0 = graph.derive("C", Recipe.bind(IselOperation(kwargs={"dim_0": 1}), Follow("B")))
    graph.set_recipe("C", Recipe.bind(IselOperation(kwargs={"dim_0": 0}), Follow("B")))
    graph.replace(
        "B", Recipe.bind(IselOperation(kwargs={"scan": slice(1, None)}), Follow("B"))
    )
    assert graph.is_stale("C")
    assert [node.revision_id for node in graph.recorded_lineage(c0)] == [f0, a0, b0, c0]

    def verified(node):
        return (
            node.revision_id == f0
            and hashlib.sha256(path.read_bytes()).digest() == original_hash
        )

    code = full_code(
        graph, c0, output_name="cut", source_names={a0: "sample"}, verify_file=verified
    )
    namespace = {"sample": graph.data(a0)}
    exec(code, namespace)  # noqa: S102
    xr.testing.assert_identical(namespace["cut"], graph.data(c0))
    assert len(reads) == 1

    xr.DataArray([100, 200]).to_netcdf(path, engine="h5netcdf")
    with pytest.raises(ReplayGraphError, match=f0):
        full_code(
            graph,
            c0,
            output_name="cut",
            source_names={a0: "sample"},
            verify_file=verified,
        )
    # The historical Python still works when the original file is gone.
    path.unlink()
    with pytest.raises(ReplayGraphError, match=f0):
        full_code(
            graph,
            c0,
            output_name="cut",
            source_names={a0: "sample"},
            verify_file=verified,
        )
    code = full_code(
        graph, c0, output_name="cut", source_names={a0: "sample", f0: "original_scan"}
    )
    namespace = {"sample": graph.data(a0), "original_scan": graph.data(f0)}
    exec(code, namespace)  # noqa: S102
    xr.testing.assert_identical(namespace["cut"], graph.data(c0))
    assert graph.entry("C").revision_id == c0


def test_full_code_requires_unambiguous_checkpoint_names(graph):
    from erlab.interactive.imagetool._provenance._graph import ReplayGraphError
    from erlab.interactive.imagetool._provenance._recorded_code import full_code

    a = graph.add_source("A", xr.DataArray(1))
    b = graph.add_source("B", xr.DataArray(2))
    c = graph.derive("C", Recipe.bind(Add(), Follow("A"), other=Follow("B")))
    with pytest.raises(ValueError, match="Distinct revisions"):
        full_code(graph, c, output_name="result", source_names={a: "data", b: "data"})
    with pytest.raises(ValueError, match="Distinct revisions"):
        full_code(
            graph,
            c,
            output_name="result",
            source_names={a: "mu", b: "\N{FULLWIDTH LATIN SMALL LETTER M}u"},
        )
    with pytest.raises(ValueError, match="recorded origins"):
        full_code(graph, c, output_name="result", source_names={c: "result"})
    with pytest.raises(ReplayGraphError, match=a):
        full_code(graph, a, output_name="result")
    code = full_code(graph, a, output_name="result", source_names={a: "source"})
    namespace = {"source": graph.data(a)}
    exec(code, namespace)  # noqa: S102
    xr.testing.assert_identical(namespace["result"], graph.data(a))


@pytest.mark.parametrize("scheduler", ["default", "inline", "deferred"])
@pytest.mark.parametrize("fail_on_obsolete", [False, True])
def test_auto_keeps_latest_input_after_repeated_supersession(
    scheduler, fail_on_obsolete
):
    callbacks = []
    schedule = {
        "default": None,
        "inline": lambda callback: callback(),
        "deferred": callbacks.append,
    }
    graph = ProvenanceGraph(schedule_auto=schedule[scheduler])
    updates = 0
    evaluated = []

    class UpdateSource(Scale):
        def apply(self, data):
            nonlocal updates
            evaluated.append(data.item())
            if updates:
                updates -= 1
                graph.replace("A", Recipe.bind(Scale(2), Follow("A")))
                if fail_on_obsolete:
                    raise ValueError("failed on obsolete input")
            return super().apply(data)

    graph.add_source("A", xr.DataArray(1))
    b0 = graph.derive("B", Recipe.bind(UpdateSource(3), Follow("A")), auto=True)
    updates = 3
    graph.replace("A", Recipe.bind(Scale(5), Follow("A")))
    if scheduler == "deferred":
        for value in [5, 10, 20]:
            assert len(callbacks) == 1
            callbacks.pop()()
            assert evaluated[-1] == value
            assert graph.entry("B").revision_id == b0
            assert graph.auto_error("B") is None
        assert len(callbacks) == 1
        callbacks.pop()()
        assert not callbacks
    a = graph.entry("A").revision_id
    b = graph.entry("B").revision_id
    assert graph.revision(b).producer.inputs == (("data", a),)
    xr.testing.assert_identical(graph.data(b), xr.DataArray(120))
    assert evaluated == [1, 5, 10, 20, 40]
    assert graph.auto_error("B") is None
    assert not graph.is_stale("B")


def test_scheduling_failure_preserves_source_and_allows_retry():
    def schedule(callback):
        if fail:
            raise RuntimeError("executor unavailable")
        callback()

    fail = True
    graph = ProvenanceGraph(schedule_auto=schedule)
    graph.add_source("A", xr.DataArray(1))
    b = graph.derive("B", Recipe.bind(Scale(2), Follow("A")), auto=True)
    a = graph.replace("A", Recipe.bind(Scale(5), Follow("A")))
    assert graph.entry("A").revision_id == a
    assert graph.entry("B").revision_id == b
    assert graph.is_stale("B")
    assert "executor unavailable" in graph.auto_error("B")
    fail = False
    graph.set_auto("B", enabled=True)
    assert not graph.is_stale("B")
    assert graph.auto_error("B") is None


def test_full_code_preserves_a_shared_auxiliary_input(graph):
    from erlab.interactive.imagetool._provenance._operations import (
        ConcatOperation,
        KspaceWorkFunctionOperation,
    )
    from erlab.interactive.imagetool._provenance._recorded_code import full_code

    data = xr.DataArray([1, 2], dims="x", attrs={"analyzer_work_function": 3.0})
    a = graph.add_source("A", data)
    graph.derive(
        "B", Recipe.bind(KspaceWorkFunctionOperation(work_function=4.0), Follow("A"))
    )
    c = graph.derive(
        "C",
        Recipe.bind(
            ConcatOperation(input_count=2, dim="k", combine_attrs="drop_conflicts"),
            Follow("B"),
            other_0=Follow("A"),
        ),
    )
    ns = {"sample": graph.data(a)}
    exec(full_code(graph, c, source_names={a: "sample"}, output_name="combined"), ns)  # noqa: S102
    xr.testing.assert_identical(ns["combined"], graph.data(c))
    xr.testing.assert_identical(ns["sample"], data)


def test_reload_from_memory_keeps_the_source_checkpoint(graph):
    a = graph.add_source("A", xr.DataArray(1))
    b = graph.derive("B", Recipe.bind(Scale(2), Follow("A")))
    c = graph.derive("C", Recipe.bind(Scale(3), Follow("B")))
    assert graph.reload("A") == a
    plan = graph.plan_update("C", command="reload")
    assert plan.current_inputs == (("A", a),)
    assert plan.entries == ("B", "C")
    assert not plan.file_loads
    assert graph.reload("C") != c
    assert graph.entry("B").revision_id != b
    assert graph.entry("A").revision_id == a
    xr.testing.assert_identical(
        graph.data(graph.entry("C").revision_id), xr.DataArray(6)
    )


def test_edit_supersedes_reload_before_downstream_operation():
    from erlab._provenance import FileLoad
    from erlab.interactive.imagetool._provenance._model import (
        FileDataSelection,
        FileReplayCall,
    )

    edit = False
    calls = []

    class Count(Scale):
        def apply(self, data):
            calls.append(self.factor)
            return super().apply(data)

    def read(_source, _call):
        if edit:
            graph.set_recipe("C", Recipe.bind(Count(3), Follow("F")))
        return xr.DataArray(10 if edit else 1)

    graph = ProvenanceGraph(read_file=read)
    f = graph.derive(
        "F",
        FileLoad(
            graph.file_source("input"),
            FileReplayCall(
                kind="callable",
                target="xarray.load_dataarray",
                selection=FileDataSelection(kind="dataarray"),
            ),
        ),
    )
    c = graph.derive("C", Recipe.bind(Count(2), Follow("F")))
    edit = True
    with pytest.raises(RuntimeError, match="changed during evaluation"):
        graph.reload("C")
    assert graph.entry("F").revision_id == f
    assert graph.entry("C").revision_id == c
    assert calls == [2]
    assert graph.is_stale("C")
    xr.testing.assert_identical(graph.data(graph.refresh("C")), xr.DataArray(3))


@pytest.mark.parametrize("earlier_first", [True, False])
def test_follow_and_earlier_step_share_one_reload(file_graph, earlier_first):
    from erlab._provenance import EarlierStep

    graph, path, load, reads = file_graph
    f0 = graph.derive("F", load)
    earlier = EarlierStep(f0)
    inputs = (earlier, Follow("F")) if earlier_first else (Follow("F"), earlier)
    graph.derive("C", Recipe.bind(Add(), inputs[0], other=inputs[1]))
    plan = graph.plan_update("C", command="reload")
    assert plan.entries == ("F", "C")
    assert plan.earlier_steps == (earlier,)
    assert plan.file_loads == (load,)
    assert reads == [str(path)]

    for value in [4, 8]:
        xr.DataArray(value).to_netcdf(path, engine="h5netcdf")
        before = len(reads)
        c = graph.reload("C")
        assert len(reads) == before + 1
        f = graph.entry("F").revision_id
        assert graph.revision(c).producer.inputs == (("data", f), ("other", f))
        xr.testing.assert_identical(graph.data(c), xr.DataArray(2 * value))


def test_mixed_bindings_with_edited_recipe_evaluate_separately(graph):
    from erlab._provenance import EarlierStep

    graph.add_source("A", xr.DataArray(2))
    b0 = graph.derive("B", Recipe.bind(Scale(3), Follow("A")))
    graph.derive("C", Recipe.bind(Add(), EarlierStep(b0), other=Follow("B")))
    graph.set_recipe("B", Recipe.bind(Scale(5), Follow("A")))
    c = graph.reload("C")
    (_, past), (_, live) = graph.revision(c).producer.inputs
    assert past != live
    assert live == graph.entry("B").revision_id
    xr.testing.assert_identical(graph.data(past), xr.DataArray(6))
    xr.testing.assert_identical(graph.data(live), xr.DataArray(10))
    xr.testing.assert_identical(graph.data(c), xr.DataArray(16))


@pytest.mark.parametrize(
    "submission", ["original", "resubmit", "round_trip", "edited", "independent"]
)
def test_array_indexer_sharing_depends_on_the_captured_snapshot(
    graph, monkeypatch, submission
) -> None:
    from erlab._provenance import EarlierStep
    from erlab.interactive.imagetool._provenance._operations import IselOperation

    graph.add_source("A", xr.DataArray([1, 2, 3], dims="x"))
    indices = np.array([0, 2])
    recipe = Recipe.bind(IselOperation(kwargs={"x": indices}), Follow("A"))
    b = graph.derive("B", recipe)
    graph.derive("C", Recipe.bind(Add(), Follow("B"), other=EarlierStep(b)))
    indices[:] = 1
    if submission == "resubmit":
        graph.set_recipe("B", recipe)
    elif submission == "round_trip":
        graph.set_recipe("B", graph.entry("B").recipe)
    elif submission == "edited":
        editor = graph.entry("B").recipe
        writable = editor.operation.kwargs["x"]
        writable[:] = [1, 2]
        graph.set_recipe("B", editor)
        writable[:] = 0
    elif submission == "independent":
        graph.set_recipe(
            "B", Recipe.bind(IselOperation(kwargs={"x": np.array([0, 2])}), Follow("A"))
        )

    calls = 0
    apply = IselOperation.apply

    def count_apply(self, data):
        nonlocal calls
        calls += 1
        return apply(self, data)

    monkeypatch.setattr(IselOperation, "apply", count_apply)
    c = graph.reload("C")
    (_, followed), (_, earlier) = graph.revision(c).producer.inputs
    separate = submission in {"independent", "edited"}
    assert calls == (2 if separate else 1)
    assert (followed != earlier) == separate
    expected = [3, 6] if submission == "edited" else [2, 6]
    xr.testing.assert_identical(graph.data(c), xr.DataArray(expected, dims="x"))


def test_mixed_bindings_keep_current_and_pinned_values_out_of_reload_cache(file_graph):
    from erlab._provenance import EarlierStep, Pin

    graph, path, load, reads = file_graph
    f = graph.derive("F", load)
    graph.derive("mixed", Recipe.bind(Add(), Follow("F"), other=EarlierStep(f)))
    graph.derive("with pin", Recipe.bind(Add(), Follow("mixed"), other=Pin(f)))
    xr.DataArray([10, 20]).to_netcdf(path, engine="h5netcdf")

    # Refresh reuses the displayed F and runs just the earlier step.
    plan = graph.plan_update("mixed", command="refresh")
    assert plan.current_inputs == (("F", f),)
    assert plan.entries == ("mixed",)
    mixed = graph.refresh("mixed")
    (_, current), (_, earlier) = graph.revision(mixed).producer.inputs
    assert current == f
    assert earlier != current
    assert graph.entry("F").revision_id == f
    xr.testing.assert_identical(graph.data(mixed), xr.DataArray([11, 22]))

    result = graph.reload("with pin")
    new_f = graph.entry("F").revision_id
    new_mixed = graph.entry("mixed").revision_id
    assert graph.revision(result).producer.inputs == (("data", new_mixed), ("other", f))
    assert graph.revision(new_mixed).producer.inputs == (
        ("data", new_f),
        ("other", new_f),
    )
    xr.testing.assert_identical(graph.data(result), xr.DataArray([21, 42]))
    assert reads == [str(path)] * 3


def test_independent_producers_with_identical_file_settings_remain_distinct(file_graph):
    from erlab._provenance import EarlierStep
    from erlab._provenance._records import PendingRecipe

    graph, path, load, reads = file_graph
    f = graph.derive("F", load)
    g = graph.derive("G", load)
    # Even a shared recipe snapshot does not identify two independent producers.
    edit = PendingRecipe(load)
    left, right = EarlierStep(f, edit), EarlierStep(g, edit)
    graph.derive("C", Recipe.bind(Add(), left, other=right))
    plan = graph.plan_update("C", command="reload")
    assert plan.file_loads == (load, load)
    c = graph.reload("C")
    (_, first), (_, second) = graph.revision(c).producer.inputs
    assert first != second
    assert graph.entry("F").revision_id == f
    assert graph.entry("G").revision_id == g
    assert reads == [str(path)] * 4


def test_refresh_shares_a_stale_derived_producer(graph):
    from erlab._provenance import EarlierStep

    calls = []

    class Count(Scale):
        def apply(self, data):
            calls.append(data.item())
            return super().apply(data)

    graph.add_source("A", xr.DataArray(2))
    b = graph.derive("B", Recipe.bind(Count(3), Follow("A")))
    graph.derive("C", Recipe.bind(Add(), Follow("B"), other=EarlierStep(b)))
    graph.replace("A", Recipe.bind(Scale(5), Follow("A")))
    plan = graph.plan_update("C", command="refresh")
    assert plan.entries == ("B", "C")
    assert plan.earlier_steps == (EarlierStep(b),)
    c = graph.refresh("C")
    updated_b = graph.entry("B").revision_id
    assert calls == [2, 10]
    assert graph.revision(c).producer.inputs == (
        ("data", updated_b),
        ("other", updated_b),
    )
    xr.testing.assert_identical(graph.data(c), xr.DataArray(60))


def test_current_producer_failure_does_not_spin_on_redundant_auto_request(graph):
    attempts = []

    class Fail(Scale):
        def apply(self, data):
            if data.item() == 2:
                attempts.append(2)
                if len(attempts) == 1:
                    graph.set_auto("B", enabled=True)
                raise RuntimeError("Provenance changed during evaluation")
            return super().apply(data)

    graph.add_source("A", xr.DataArray(1))
    b = graph.derive("B", Recipe.bind(Fail(3), Follow("A")), auto=True)
    graph.replace("A", Recipe.bind(Scale(2), Follow("A")))
    assert attempts == [2]
    assert graph.entry("B").revision_id == b
    assert graph.is_stale("B")
    assert graph.auto_error("B") == "Provenance changed during evaluation"


@pytest.mark.parametrize("edited", [False, True])
def test_reload_compares_effective_file_settings_for_the_same_producer(
    tmp_path, edited
):
    from erlab._provenance import EarlierStep, FileLoad
    from erlab.interactive.imagetool._provenance._model import (
        FileDataSelection,
        FileReplayCall,
    )

    path = tmp_path / "offsets.nc"
    xr.Dataset({"a": ("x", [1]), "b": ("x", [5])}).to_netcdf(path, engine="h5netcdf")
    reads = []

    def read(source, call):
        reads.append(call.selection.value)
        return xr.load_dataset(source.path, **call.kwargs)[call.selection.value]

    graph = ProvenanceGraph(read_file=read)
    source = graph.file_source(str(path))

    def load(variable):
        return FileLoad(
            source,
            FileReplayCall(
                kind="callable",
                target="xarray.load_dataset",
                kwargs={"engine": "h5netcdf"},
                selection=FileDataSelection(kind="dataset_variable", value=variable),
            ),
        )

    f = graph.derive("F", load("a"))
    graph.derive("C", Recipe.bind(Add(), Follow("F"), other=EarlierStep(f)))
    graph.set_recipe("F", load("b" if edited else "a"))
    plan = graph.plan_update("C", command="reload")
    reads.clear()
    c = graph.reload("C")
    assert len(plan.file_loads) == len(reads)
    (_, live), (_, earlier) = graph.revision(c).producer.inputs
    if edited:
        assert sorted(reads) == ["a", "b"]
        assert live != earlier
        assert graph.data(c).item() == 6
    else:
        assert reads == ["a"]
        assert live == earlier
        assert graph.data(c).item() == 2


def test_preview_reuses_captured_file_kwargs_without_copying_the_array(tmp_path):
    from erlab._provenance import EarlierStep, FileLoad
    from erlab.interactive.imagetool._provenance._model import (
        FileDataSelection,
        FileReplayCall,
    )

    copies = []
    reads = []

    class Variables(np.ndarray):
        def __deepcopy__(self, memo):
            copies.append(self.item())
            return self.copy()

        def __eq__(self, other):
            raise AssertionError("Planning compared array contents")

        def __hash__(self):
            raise AssertionError("Planning hashed an array")

    path = tmp_path / "source.nc"
    xr.Dataset({"a": 1, "b": 2}).to_netcdf(path, engine="h5netcdf")

    def read(source, call):
        reads.append(call.kwargs["drop_variables"].item())
        data = xr.load_dataset(source.path, **call.kwargs)[call.selection.value]
        call.kwargs["drop_variables"][:] = "a"
        return data

    graph = ProvenanceGraph(read_file=read)
    names = np.array(["b"]).view(Variables)
    load = FileLoad(
        graph.file_source(str(path)),
        FileReplayCall(
            kind="callable",
            target="xarray.load_dataset",
            kwargs={"engine": "h5netcdf", "drop_variables": names},
            selection=FileDataSelection(kind="dataset_variable", value="a"),
        ),
    )
    f = graph.derive("F", load)
    graph.derive("C", Recipe.bind(Add(), Follow("F"), other=EarlierStep(f)))
    names[:] = "a"
    copies.clear()
    graph.set_recipe("F", graph.entry("F").recipe)
    plan = graph.plan_update("C", command="reload")
    assert len(plan.file_loads) == 1
    assert reads == ["b"]
    assert copies == []

    c = graph.reload("C")
    assert reads == ["b", "b"]
    (_, live), (_, earlier) = graph.revision(c).producer.inputs
    assert live == earlier
    assert graph.data(c).item() == 2


def test_edit_shared_ancestor_rebuilds_only_selected_branch(graph):
    graph.add_source("A", xr.DataArray(2))
    b = graph.derive("B", Recipe.bind(Scale(3), Follow("A")))
    c = graph.derive("C", Recipe.bind(Scale(4), Follow("B")))
    sibling = graph.derive("sibling", Recipe.bind(Scale(5), Follow("B")))

    changed = graph.edit_operation(
        "C", displayed_revision=c, edited_revision=b, operations=[Scale(7)]
    )
    assert graph.data(changed).item() == 56
    assert graph.entry("B").revision_id == b
    assert graph.entry("sibling").revision_id == sibling
    assert graph.follows("C") == {"A"}
    assert graph.data(b).item() == 6
    with pytest.raises(RuntimeError, match="changed while editing"):
        graph.edit_operation(
            "C", displayed_revision=c, edited_revision=b, operations=[Scale(8)]
        )
    graph.replace("A", Recipe.bind(Scale(10), Follow("A")))
    assert graph.data(graph.refresh("C")).item() == 560
    assert graph.entry("B").revision_id == b


def test_group_edit_replaces_the_chain_and_preserves_other_input_slots(graph):
    from erlab._provenance import Pin

    graph.add_source("source", xr.DataArray(2))
    pinned = graph.add_source("offset", xr.DataArray(3))
    first = graph.derive(
        "first", Recipe.bind(Add(), Follow("source"), other=Pin(pinned))
    )
    last = graph.derive("last", Recipe.bind(Scale(10), Follow("first")))
    shown = graph.derive(
        "shown", Recipe.bind(Add(), Follow("last"), other=Follow("first"))
    )

    class Fail(Scale):
        def apply(self, data):
            raise ValueError("invalid edited settings")

    def edit(operations, prefix=(first,)):
        return graph.edit_operation(
            "shown",
            displayed_revision=shown,
            edited_revision=last,
            group_prefix=prefix,
            operations=operations,
        )

    with pytest.raises(ValueError, match="contiguous"):
        edit([Add()], prefix=(pinned,))
    with pytest.raises(ValueError, match="invalid edited settings"):
        edit([Add(), Fail(1)])
    assert graph.current_revision("shown") == shown

    # Remove the final scaling operation. The other slot consumes the first
    # step independently and keeps that binding.
    changed = edit([Add()])
    assert graph.data(changed).item() == 10
    assert graph.current_revision("first") == first
    assert graph.current_revision("last") == last
    assert graph.follows("shown") == {"source", "first"}
    with pytest.raises(RuntimeError, match="changed while editing"):
        edit([Add(), Scale(4)])
    graph.replace("offset", Recipe.bind(Scale(2), Follow("offset")))
    graph.replace("source", Recipe.bind(Scale(2), Follow("source")))
    assert graph.data(graph.refresh("shown")).item() == 14


def test_filter_tracks_displayed_and_unfiltered_consumers(graph):
    from erlab._provenance import Pin

    original = graph.add_source("A", xr.DataArray(2))
    shown = graph.derive("shown", Recipe.bind(Scale(3), Follow("A")))
    raw = graph.derive("raw", Recipe.bind(Scale(3), Follow("A", source=True)))
    filtered = graph.prepare_import(
        "A", xr.DataArray(10), (Recipe.bind(Scale(5), Pin(original)),), filtered=True
    )()
    assert graph.is_stale("shown")
    assert not graph.is_stale("raw")
    assert graph.entry("raw").revision_id == raw
    assert graph.next_inputs("raw") == (("data", original),)
    assert graph.next_inputs("shown") == (("data", filtered),)
    graph.reload_entries(["shown", "raw"])
    assert graph.data(graph.entry("shown").revision_id).item() == 30
    assert graph.data(graph.entry("raw").revision_id).item() == 6
    assert graph.entry("shown").revision_id != shown
    graph.remove_entry("A")
    assert graph.data(graph.reload("raw")).item() == 6
    assert graph.data(graph.reload("shown")).item() == 30
    graph.remove_entry("shown")
    graph.remove_entry("raw")
    assert not graph.revision_ids
    assert not graph.is_closed("A")


def test_transposed_filtered_source_follows_refresh_duplicate_and_close(graph):
    from erlab._provenance import EarlierStep
    from erlab.interactive.imagetool._provenance._operations import TransposeOperation

    data = xr.DataArray(np.arange(6.0).reshape(2, 3), dims=("x", "y"))
    graph.add_source("input", data)
    base = graph.derive("view", Recipe.bind(Scale(2), Follow("input")))
    graph.prepare_import(
        "view",
        (data * 10).T,
        (
            Recipe(Scale(5), (("data", EarlierStep(base)),), view_role="filter"),
            Recipe(
                TransposeOperation(dims=("y", "x")),
                (("data", Follow("view")),),
                view_role="layout",
            ),
        ),
    )()
    source = Follow("view", source=True)
    graph.derive("consumer", Recipe.bind(Scale(3), source))
    graph.replace("input", Recipe.bind(Scale(2), Follow("input")))
    xr.testing.assert_identical(graph.data(graph.refresh("consumer")), (data * 12).T)
    xr.testing.assert_identical(
        graph.data(graph.current_revision("view")), (data * 20).T
    )
    assert graph.revision(graph.current_revision("consumer")).producer.inputs == (
        ("data", graph.followed_revision(source)),
    )
    graph.duplicate_entries({"view": "copy", "consumer": "copy_consumer"})
    copied = Follow("copy", source=True)
    assert graph.followed_revision(copied) != graph.followed_revision(source)
    graph.remove_entry("copy")
    xr.testing.assert_identical(
        graph.data(graph.reload("copy_consumer")), (data * 12).T
    )


def test_installed_conversion_follows_the_pending_recipe():
    from contextlib import nullcontext

    from erlab._provenance import EarlierStep, Pin
    from erlab.interactive.imagetool._provenance._operations._coordinates import (
        ImageToolInputOperation,
    )

    def convert(_uid, key, data):
        if data.dtype.kind == "i":
            return Recipe.bind(
                ImageToolInputOperation(cast_float64=True), EarlierStep(key)
            )
        return None

    graph = ProvenanceGraph(install=lambda *_: nullcontext({}), installed_step=convert)
    old = graph.add_source("A", xr.DataArray(1))
    graph.add_source("B", xr.DataArray(11))
    graph.add_source("result", xr.DataArray(0))
    displayed = graph.prepare_import(
        "result",
        None,
        (Recipe.bind(Scale(2), Pin(old)),),
        next_recipe=Recipe.bind(Scale(3), Follow("B")),
    )()
    shown = graph.data(displayed)
    assert shown.dtype == np.float64
    assert shown.item() == 2
    assert graph.is_stale("result")
    refreshed = graph.data(graph.refresh("result"))
    assert refreshed.dtype == np.float64
    assert refreshed.item() == 33


def test_worker_owns_its_captured_input_and_completed_output(graph):
    data = xr.DataArray([1.0, 2.0], attrs={"sample": "original"})
    key = graph.add_source("A", data)
    recipe = Recipe.bind(Scale(factor=3), Follow("A"))
    task = graph.capture_worker("B", recipe, lambda op, value: op.apply(value))
    with pytest.raises(ValueError, match="no completed calculation"):
        graph.retain_completed(task)
    task.data.attrs.clear()
    returned = task.run()
    returned.data = np.zeros(2)
    returned.attrs["other"] = True
    with pytest.raises(RuntimeError, match="already completed"):
        task.run()
    result = graph.retain_completed(task)
    assert graph.revision(result.revision_id).producer.inputs == (("data", key),)
    xr.testing.assert_identical(result.calculated_data, data * 3)


def test_prepared_publication_rejects_changed_input(graph):
    a = graph.add_source("A", xr.DataArray(1))
    publish = graph.prepare_import(
        "C", xr.DataArray(2), [Recipe.bind(Scale(2), Follow("A"))]
    )
    graph.replace("A", Recipe.bind(Scale(3), Follow("A")))
    with pytest.raises(RuntimeError, match="changed during publication"):
        publish()
    assert "C" not in graph
    assert a in graph.revision_ids


@pytest.mark.parametrize("action", ["change input", "unrelated", "cancel"])
def test_publication_rechecks_request_while_host_can_roll_back(action):
    from concurrent.futures import CancelledError
    from contextlib import contextmanager

    trigger = cancelled = False
    windows = {}

    @contextmanager
    def install(entries, values):
        nonlocal trigger, cancelled
        before = windows.copy()
        windows.update({name: values[key].item() for name, key in entries.items()})
        try:
            if trigger:
                trigger = False
                if action == "cancel":
                    cancelled = True
                else:
                    target = "A" if action == "change input" else "other"
                    graph.prepare_import(target, xr.DataArray(3), ())()
            yield {}
        except Exception:
            windows.clear()
            windows.update(before)
            raise

    graph = ProvenanceGraph(install=install)
    graph.add_source("A", xr.DataArray(1))
    initial = graph.derive("B", Recipe.bind(Scale(2), Follow("A")))
    graph.prepare_import("A", xr.DataArray(2), ())()
    trigger = True
    if action == "unrelated":
        updated = graph.refresh("B")
        assert windows["B"] == graph.data(updated).item() == 4
        assert not graph.is_stale("B")
    else:
        with pytest.raises(CancelledError if action == "cancel" else RuntimeError):
            graph.refresh("B", is_cancelled=lambda: cancelled)
        assert windows["B"] == 2
        assert graph.current_revision("B") == initial
        assert graph.is_stale("B")
        if action == "change input":
            assert graph.data(graph.refresh("B")).item() == 6


def test_lazy_file_checkpoint_rejects_changed_bytes(tmp_path):
    import dask.array as da
    from dask import delayed

    path = tmp_path / "mutable.npy"
    original = np.arange(8, dtype=np.float64)
    np.save(path, original)
    reads = []

    @delayed
    def read_block(index):
        reads.append(index)
        return np.load(path)[index : index + 2]

    values = da.concatenate(
        [
            da.from_delayed(read_block(i), shape=(2,), dtype="float64")
            for i in range(0, 8, 2)
        ]
    )
    data = xr.DataArray(values, dims="x")
    data.encoding["source"] = path
    graph = ProvenanceGraph()
    key = graph.prepare_import("source", data, (), live=True)()
    assert not reads
    retained = graph.data(key)
    assert not reads
    np.testing.assert_array_equal(retained[:2].values, original[:2])
    assert reads == [0]
    np.save(path, original + 20)
    with pytest.raises(OSError, match=r"Recorded lazy input .* has changed"):
        _ = retained[:2].values


def test_lazy_file_guard_covers_values_when_coordinate_is_already_guarded(tmp_path):
    import dask.array as da

    from erlab._provenance._records import capture_display_data

    path = tmp_path / "source.bin"
    path.write_bytes(b"original")
    coordinate = capture_display_data(
        xr.DataArray(da.arange(4, chunks=2), dims="x"), source_path=str(path)
    )
    data = xr.DataArray(
        da.ones(4, chunks=2), dims="x", coords={"aux": ("x", coordinate.data)}
    )
    recorded = capture_display_data(data, source_path=str(path))
    np.testing.assert_array_equal(recorded.data.compute(), np.ones(4))
    path.write_bytes(b"changed")
    with pytest.raises(OSError, match="unavailable or has changed"):
        recorded.data.compute()


def test_auto_retries_if_installation_error_belongs_to_a_superseded_input():
    from contextlib import contextmanager

    update = False
    callbacks = []

    @contextmanager
    def install(_entries, _values):
        nonlocal update
        if update:
            update = False
            graph.prepare_import("A", xr.DataArray(3), ())()
            raise ValueError("Failed to install the superseded result")
        yield {}

    graph = ProvenanceGraph(install=install, schedule_auto=callbacks.append)
    graph.add_source("A", xr.DataArray(1))
    initial = graph.derive("B", Recipe.bind(Scale(2), Follow("A")), auto=True)
    graph.prepare_import("A", xr.DataArray(2), ())()
    update = True
    callbacks.pop()()
    assert graph.current_revision("B") == initial
    assert graph.is_stale("B")
    assert graph.auto_error("B") is None
    callbacks.pop()()
    assert not graph.is_stale("B")
    assert graph.data(graph.current_revision("B")).item() == 6


def test_captured_lazy_pin_is_not_loaded_on_publication():
    import dask.array as da
    from dask import delayed

    from erlab._provenance import Pin
    from erlab._provenance._records import capture_display_data
    from erlab.interactive.imagetool._provenance._operations import IselOperation

    reads = []

    def chunk(index):
        reads.append(index)
        return np.array([index], dtype=float)

    source = xr.DataArray(
        da.concatenate(
            [
                da.from_delayed(delayed(chunk)(i), shape=(1,), dtype=float)
                for i in range(16)
            ]
        ),
        dims="x",
    )
    pin = capture_display_data(source)
    selected = pin.isel(x=3).compute()
    assert reads == [3]
    graph = ProvenanceGraph()
    key = graph.prepare_import(
        "B",
        selected,
        [Recipe.bind(IselOperation(kwargs={"x": 3}), Pin("old"))],
        checkpoints={"old": ("A", pin)},
    )()
    assert reads == [3]
    del source
    xr.testing.assert_identical(graph.data(key), selected)
    xr.testing.assert_equal(
        graph.data("old").compute(), xr.DataArray(np.arange(16.0), dims="x")
    )


def test_live_frame_copy_is_elided_only_for_immutable_backing():
    from erlab._provenance import Pin
    from erlab.interactive.imagetool._provenance._operations import IselOperation

    parent = np.arange(12)
    mutable_view = parent.view()
    mutable_view.flags.writeable = False
    immutable = np.frombuffer(parent.tobytes(), dtype=parent.dtype)
    graph = ProvenanceGraph()
    for name, buffer, share in (("A", mutable_view, False), ("B", immutable, True)):
        data = xr.DataArray(buffer, dims="x")
        revision = graph.prepare_import(name, data, (), live=True)()
        graph.derive(
            f"read-{name}", Recipe.bind(IselOperation(kwargs={"x": 1}), Pin(revision))
        )
        assert np.shares_memory(buffer, graph._displayed[revision].data) == share
    parent[:] = 99
    xr.testing.assert_equal(
        graph.data(graph.current_revision("A")), xr.DataArray(np.arange(12), dims="x")
    )


def test_live_input_owns_values_coordinates_and_metadata_at_acceptance():
    values = np.arange(12.0).reshape(3, 4)[:, ::2]
    offsets = np.arange(3.0)
    data = xr.DataArray(
        values,
        dims=("x", "y"),
        coords={"offset": ("x", offsets)},
        attrs={"nested": {"value": 1}},
    )
    expected = data.copy(deep=True)
    graph = ProvenanceGraph()
    revision = graph.prepare_import("source", data, (), live=True)()
    values[:] = -1
    offsets[:] = -2
    data.attrs["nested"]["value"] = 3
    xr.testing.assert_identical(graph.data(revision), expected)


def test_recorded_code_substitutes_only_the_branch_with_a_changed_lazy_file(tmp_path):
    import dask.array as da
    from dask import delayed

    from erlab._provenance import FileLoad
    from erlab.interactive.imagetool._provenance._model import (
        FileDataSelection,
        FileReplayCall,
    )
    from erlab.interactive.imagetool._provenance._operations import (
        ConcatOperation,
        IselOperation,
    )
    from erlab.interactive.imagetool._provenance._recorded_code import full_code

    path = tmp_path / "source.npy"
    values = np.arange(12).reshape(3, 4)
    np.save(path, values)
    data = xr.DataArray(
        da.from_delayed(delayed(np.load)(path), shape=values.shape, dtype=values.dtype),
        dims=("scan", "x"),
    )
    graph = ProvenanceGraph()
    file = FileLoad(
        graph.file_source(str(path)),
        FileReplayCall(
            kind="callable",
            target="xarray.open_dataarray",
            selection=FileDataSelection(kind="dataarray"),
        ),
    )
    root = graph.prepare_import("A", data, (file,), live=True)()
    selection = IselOperation(kwargs={"scan": 0})
    selected = selection.apply(data).compute()
    child = graph.prepare_import(
        "B", selected, (Recipe.bind(selection, Follow("A")),)
    )()
    other = graph.add_source("C", selected + 2)
    result = graph.derive(
        "D",
        Recipe.bind(
            ConcatOperation(input_count=2, dim="side", combine_attrs="drop_conflicts"),
            Follow("C"),
            other_0=Follow("B"),
        ),
    )
    assert graph.recorded_code_inputs(result) == set()
    np.save(path, values + 100)
    assert graph.recorded_code_inputs(result) == {child}
    assert root not in [
        node.revision_id for node in graph.recorded_lineage(result, stop_at={child})
    ]
    ns = {"selection": graph.data(child), "other_input": graph.data(other)}
    code = full_code(
        graph,
        result,
        output_name="combined",
        source_names={child: "selection", other: "other_input"},
    )
    exec(code, ns)  # noqa: S102
    xr.testing.assert_identical(ns["combined"], graph.data(result))


def test_pending_input_owns_only_the_revision_it_used(graph):
    import gc

    original = graph.add_source("source", xr.DataArray(1))
    holder = Scale(2)
    graph.capture_input(Follow("source"), "console input", holder)
    graph.prepare_import("source", xr.DataArray(3), ())()
    assert graph.captured_input("console input") == original
    del holder
    gc.collect()
    assert graph.captured_input("console input") is None
    assert not graph.has_revision(original)


def test_prepared_result_rejects_unknown_recorded_input_slot(graph):
    original = graph.add_source("source", xr.DataArray(1))
    with pytest.raises(ValueError, match="published operation slot"):
        graph.prepare_import(
            "result",
            xr.DataArray(2),
            (Recipe.bind(Scale(2), Follow("source")),),
            recorded_inputs={(0, "unknown"): original},
        )
    assert "result" not in graph


def test_multiple_history_edits_and_restores_publish_together(graph):
    graph.add_source("source", xr.DataArray(2))
    graph.add_source("other", xr.DataArray(3))
    first = graph.derive("first", Recipe.bind(Scale(2), Follow("source")))
    second = graph.derive("second", Recipe.bind(Scale(3), Follow("other")))
    before = graph.generation
    graph.apply_operation_edits(
        {
            "first": (
                first,
                Recipe.bind(Add(), Follow("other"), other=Follow("other")),
            ),
            "second": (second, Recipe.bind(Scale(4), Follow("first"))),
        }
    )
    assert graph.generation == before + 1
    a, b = (graph.current_revision(uid) for uid in ("first", "second"))
    assert graph.data(a).item() == 6
    assert graph.data(b).item() == 24
    assert graph.revision(b).producer.inputs == (("data", a),)
    assert graph.follows("first") == {"other"}
    assert graph.follows("second") == {"first"}
    origin = graph.current_revision("other")
    graph.restore_revisions({"first": (a, origin), "second": (b, a)})
    assert graph.data(graph.current_revision("first")).item() == 3
    assert graph.data(graph.current_revision("second")).item() == 6
    assert graph.follows("first") == set()
    assert graph.follows("second") == {"other"}


def test_failed_batch_edit_preserves_all_results(graph):
    from erlab.interactive.imagetool._provenance._operations import IselOperation

    graph.add_source("input", xr.DataArray([1, 2], dims="x"))
    first = graph.derive("first", Recipe.bind(Scale(2), Follow("input")))
    second = graph.derive("second", Recipe.bind(Scale(3), Follow("input")))
    with pytest.raises(IndexError):
        graph.apply_operation_edits(
            {
                "first": (first, Recipe.bind(Scale(4), Follow("input"))),
                "second": (
                    second,
                    Recipe.bind(IselOperation(kwargs={"x": 9}), Follow("first")),
                ),
            }
        )
    assert graph.current_revision("first") == first
    assert graph.current_revision("second") == second
    assert graph.follows("second") == {"input"}


def test_duplicate_subtree_keeps_stale_inputs_and_follows_copied_entries(graph):
    graph.add_source("root", xr.DataArray(2))
    used = graph.derive(
        "child", Recipe.bind(Scale(3), Follow("root")), parent_id="root"
    )
    graph.derive("sum", Recipe.bind(Add(), Follow("child"), other=Follow("child")))
    graph.prepare_import("root", xr.DataArray(5), ())()
    copies = graph.duplicate_entries({"root": "R", "child": "C", "sum": "S"})
    assert graph.is_stale("S")
    assert graph.entry("C").parent_id == "R"
    assert graph.follows("S") == {"C"}
    assert graph.revision(graph.current_revision("S")).producer.inputs == (
        ("data", copies[used]),
        ("other", copies[used]),
    )
    assert graph.data(graph.current_revision("S")).item() == 12
    graph.prepare_import("R", xr.DataArray(7), ())()
    graph.promote_entry("C")
    assert graph.entry("C").parent_id is None
    assert graph.data(graph.refresh("S")).item() == 42
    assert graph.data(graph.current_revision("sum")).item() == 12


def test_duplicate_during_publication_updates_auto_from_the_copied_source():
    def duplicate_on_change(entries):
        if entries == ("root",) and "child" in graph:
            graph.duplicate_entries({"child": "C", "root": "R"})

    graph = ProvenanceGraph(published=duplicate_on_change)
    graph.add_source("root", xr.DataArray(2))
    graph.derive("child", Recipe.bind(Scale(3), Follow("root")), auto=True)
    graph.prepare_import("root", xr.DataArray(5), ())()
    copied = graph.current_revision("C")
    assert graph.revision(copied).producer.inputs == (
        ("data", graph.current_revision("R")),
    )
    assert graph.data(copied).item() == 15
    assert not graph.is_stale("C")
