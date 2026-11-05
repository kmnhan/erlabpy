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

    def visit(binding, *, mode):
        visits.append(binding)
        return plan_input(binding, mode=mode)

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
