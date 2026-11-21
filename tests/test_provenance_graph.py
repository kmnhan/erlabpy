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
    from erlab._provenance._operations import IselOperation

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
    from erlab._provenance._operations import IselOperation

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
    from erlab._provenance._model import FileDataSelection, FileReplayCall

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
    from erlab._provenance._operations import ConcatOperation

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
    from erlab._provenance._operations import IselOperation

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
    from erlab._provenance._model import FileDataSelection, FileReplayCall

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
    from erlab._provenance._operations import IselOperation

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
    from erlab._provenance._model import FileDataSelection, FileReplayCall

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

    from erlab._provenance._operations import ConcatOperation, IselOperation
    from erlab.interactive.imagetool._provenance._graph import ReplayGraphError
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
    with pytest.raises(OSError, match="has changed"):
        exec(code, {"sample": graph.data(a0)})  # noqa: S102
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
    with pytest.raises(OSError, match="unavailable"):
        exec(code, {"sample": graph.data(a0)})  # noqa: S102
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
    from erlab._provenance._operations import (
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
    from erlab._provenance._model import FileDataSelection, FileReplayCall

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
    from erlab._provenance._operations import IselOperation

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
    from erlab._provenance._model import FileDataSelection, FileReplayCall

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
    from erlab._provenance._model import FileDataSelection, FileReplayCall

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


def test_versioned_copy_keeps_unchunked_selected_variable_and_cast_lazy(
    tmp_path, monkeypatch
):
    import psutil
    from xarray.backends.h5netcdf_ import H5NetCDFArrayWrapper

    from erlab._provenance import FileLoad, FileVersion
    from erlab._provenance._model import FileDataSelection, FileReplayCall
    from erlab.interactive.imagetool._provenance._recorded_code import full_code

    path = tmp_path / "counts.nc"
    counts = xr.DataArray(np.arange(12, dtype=np.int16), dims="x", name="counts")
    dataset = xr.Dataset({"counts": counts, "other": counts * 3})
    dataset.to_netcdf(
        path, engine="h5netcdf", encoding={"counts": {"chunksizes": (2,)}}
    )
    version = FileVersion.capture(path)
    call = FileReplayCall(
        kind="callable",
        target="xarray.open_dataset",
        kwargs={"engine": "h5netcdf"},
        selection=FileDataSelection(kind="dataset_variable", value="counts"),
        cast_float64=True,
    )
    graph = ProvenanceGraph()
    graph.adopt_checkpoint(
        "source",
        "original",
        counts.astype("float64"),
        file_load=FileLoad(graph.file_source(str(path)), call),
        backing=version,
    )
    code = full_code(
        graph, "original", output_name="measured", verify_file=lambda _: True
    )
    assert "measured = erlab.io.read_file_version(\n" in code
    assert "\n    xarray.open_dataset," in code
    reads = []
    read = H5NetCDFArrayWrapper._getitem

    def read_chunk(self, key):
        reads.append((self.variable_name, key))
        return read(self, key)

    monkeypatch.setattr(H5NetCDFArrayWrapper, "_getitem", read_chunk)
    notebook = {}
    exec(code, notebook)  # noqa: S102
    result = notebook["measured"]
    assert result.dtype == np.float64
    assert reads == []
    xr.testing.assert_identical(result.isel(x=0).compute(), counts[0].astype("float64"))
    assert reads == [("counts", (slice(0, 2, 1),))]

    result.close()
    assert str(path.resolve()) not in {f.path for f in psutil.Process().open_files()}
    path.unlink()
    with pytest.raises(OSError, match="unavailable"):
        result.isel(x=-1).compute()


def test_recorded_file_code_distinguishes_none_and_tuple_variable_keys(
    tmp_path, monkeypatch
):
    import dask

    from erlab._provenance import FileLoad, FileVersion
    from erlab._provenance._model import FileDataSelection, FileReplayCall
    from erlab.interactive.imagetool._provenance._recorded_code import full_code

    path = tmp_path / "names.nc"
    values = xr.Dataset({"raw": ("x", [3.0, 4.0]), "clean": ("x", [1.0, 2.0])})
    values.to_netcdf(path, engine="h5netcdf")
    version = FileVersion.capture(path)
    open_dataset = xr.open_dataset

    def restore_keys(path, **kwargs):
        # NetCDF variable names are strings. Restore the xarray keys that a
        # previously recorded reader can expose after opening the file.
        dataset = open_dataset(path, **kwargs)
        renamed = dataset.rename_vars({"raw": None, "clean": ("counts", "cleaned")})
        renamed.set_close(dataset.close)
        return renamed

    graph = ProvenanceGraph()
    codes = []
    for entry, key in (("raw", None), ("clean", ("counts", "cleaned"))):
        call = FileReplayCall(
            kind="callable",
            target="xarray.open_dataset",
            kwargs={"engine": "h5netcdf", "chunks": {}},
            selection=FileDataSelection(kind="dataset_variable", value=key),
        )
        restored = FileReplayCall.model_validate_json(call.model_dump_json())
        graph.adopt_checkpoint(
            entry,
            entry,
            values[entry].rename(key),
            file_load=FileLoad(graph.file_source(str(path)), restored),
            backing=version,
        )
        codes.append(
            full_code(graph, entry, output_name="measured", verify_file=lambda _: True)
        )
    monkeypatch.setattr(xr, "open_dataset", restore_keys)
    arrays = []
    for code in codes:
        notebook = {}
        exec(code, notebook)  # noqa: S102
        arrays.append(notebook["measured"])
    raw, clean = arrays
    assert raw.name is None
    assert clean.name == ("counts", "cleaned")
    assert raw.data.name != clean.data.name
    assert dask.compute(raw.data[1], clean.data[1]) == (4.0, 2.0)
    raw.close()
    clean.close()


def test_file_replay_imports_without_optional_eager_datatree_loader() -> None:
    code = textwrap.dedent("""
        import xarray as xr

        if hasattr(xr, "load_datatree"):
            del xr.load_datatree
        from erlab._provenance._files import _SUPPORTED_CALLABLE_LOADERS

        assert _SUPPORTED_CALLABLE_LOADERS["xarray.open_datatree"] is xr.open_datatree
        assert "xarray.load_datatree" not in _SUPPORTED_CALLABLE_LOADERS
    """)
    subprocess.run([sys.executable, "-c", code], check=True)


def test_xarray_file_capture_bounds_layout_checks(tmp_path, monkeypatch):
    import h5py

    from erlab._provenance import _files

    path = tmp_path / "multiple.h5"
    with h5py.File(path, "w") as f:
        f["first"] = [0]
        f["second"] = [1]
    monkeypatch.setattr(_files, "_MAX_STORAGE_LINKS", 1)
    assert _files._capture_loader_file(path, xr.open_dataarray, {}) is None
    monkeypatch.setattr(_files, "_MAX_STORAGE_LINKS", 2)
    assert _files._capture_loader_file(path, xr.open_dataarray, {}) is not None


@pytest.mark.parametrize("layout", ["virtual", "external_link", "external_storage"])
def test_xarray_loader_does_not_pin_only_the_container_of_multifile_data(
    graph, tmp_path, layout
):
    import h5py

    from erlab._provenance import FileLoad
    from erlab._provenance._files import _capture_loader_file
    from erlab._provenance._model import FileDataSelection, FileReplayCall
    from erlab.interactive.imagetool._provenance._graph import ReplayGraphError
    from erlab.interactive.imagetool._provenance._recorded_code import full_code

    path = tmp_path / "container.h5"
    other = tmp_path / "values.h5"
    values = np.array([1.0, 2.0])
    with h5py.File(other, "w") as f:
        f["values"] = values
    with h5py.File(path, "w") as f:
        if layout == "virtual":
            mapping = h5py.VirtualLayout(shape=values.shape, dtype=values.dtype)
            mapping[:] = h5py.VirtualSource(str(other), "/values", shape=values.shape)
            f.create_virtual_dataset("values", mapping)
        elif layout == "external_link":
            f["values"] = h5py.ExternalLink(str(other), "/values")
        else:
            f.create_dataset(
                "values",
                data=values,
                external=[(str(tmp_path / "raw"), 0, values.nbytes)],
            )

    kwargs = {"engine": "h5netcdf", "phony_dims": "sort", "chunks": {}}
    backing = _capture_loader_file(path, xr.open_dataset, kwargs)
    assert backing is None
    with xr.open_dataset(path, **kwargs) as ds:
        revision = graph.add_source("A", ds["values"], backing=backing)
    call = FileReplayCall(
        kind="callable",
        target="xarray.open_dataset",
        kwargs=kwargs,
        selection=FileDataSelection(kind="dataset_variable", value="values"),
    )
    graph.adopt_checkpoint(
        "file",
        "old_file",
        graph.data(revision),
        file_load=FileLoad(graph.file_source(str(path)), call),
    )
    with pytest.raises(ReplayGraphError, match="Supply a notebook name"):
        full_code(graph, "old_file", output_name="sample", verify_file=lambda _: True)
    code = full_code(
        graph, "old_file", output_name="sample", source_names={"old_file": "saved"}
    )
    with h5py.File(path, "r+") as f:
        f["values"][:] = values + 2
    namespace = {"saved": graph.data("old_file")}
    exec(code, namespace)  # noqa: S102
    np.testing.assert_equal(namespace["sample"].values, values)
    np.testing.assert_equal(graph.data(revision).values, values)


@pytest.mark.parametrize(
    "layout", ["index", "categorical", "multi_index", "values", "dropped_index"]
)
def test_snapshots_own_mutable_members_of_pandas_coordinates(graph, layout):
    from dataclasses import dataclass, field

    import pandas as pd

    @dataclass(frozen=True)
    class Label:
        key: str
        notes: list[str] = field(compare=False)

    labels = [Label("a", ["original"]), Label("b", [])]
    if layout == "values":
        incoming = xr.DataArray(pd.Categorical(labels, categories=labels), dims="x")

        def first(array):
            return array.values[0]

    else:
        if layout == "multi_index":
            index = pd.MultiIndex.from_product(
                [labels, [1]], names=["sample", "channel"]
            )
            coords = xr.Coordinates.from_pandas_multiindex(index, "x")
        else:
            index = (
                pd.CategoricalIndex(labels, categories=labels)
                if layout == "categorical"
                else pd.Index(labels)
            )
            coords = {"x": index}
        incoming = xr.DataArray([1, 2], dims="x", coords=coords)

        if layout == "dropped_index":
            incoming = incoming.drop_indexes("x")

        def first(array):
            if layout == "dropped_index":
                return array.x.values[0]
            item = array.indexes["x"][0]
            return item[0] if layout == "multi_index" else item

    revision = graph.add_source("A", incoming)
    labels[0].notes.clear()
    public = graph.data(revision)
    assert first(public).notes == ["original"]
    first(public).notes.append("edited")
    saved = graph.data(revision)
    assert first(saved).notes == ["original"]
    assert saved.dtype == incoming.dtype
    if layout not in {"values", "dropped_index"}:
        assert type(saved.indexes["x"]) is type(incoming.indexes["x"])
        assert saved.coords.keys() == incoming.coords.keys()


@pytest.mark.parametrize("replace_source", [False, True])
def test_mutating_repeated_operand_preserves_source_and_sibling(graph, replace_source):
    from dataclasses import field

    from erlab._provenance._operations import IselOperation

    @dataclass(frozen=True)
    class Label:
        name: str
        notes: list[str] = field(compare=False)

    class EditNotes:
        def input_slots(self):
            return ("other",)

        def apply_with_inputs(self, data, inputs):
            other = inputs["other"]
            assert other is data
            other.values[0]["notes"].append("edited")
            other.x.values[0].notes.append("edited")
            other.x.attrs["notes"].append("edited")
            other.x.encoding["notes"].append("edited")
            return data

    incoming = xr.DataArray(
        np.array([{"notes": []}], dtype=object),
        dims="x",
        coords={"x": [Label("sample", [])]},
    )
    incoming.x.attrs["notes"] = []
    incoming.x.encoding["notes"] = []
    source = graph.add_source("A", incoming)
    sibling = graph.derive(
        "B", Recipe.bind(IselOperation(kwargs={"x": slice(None)}), Follow("A"))
    )
    recipe = Recipe.bind(EditNotes(), Follow("A"), other=Follow("A"))
    result = graph.replace("A", recipe) if replace_source else graph.derive("C", recipe)
    assert graph.revision(result).producer.inputs == (
        ("data", source),
        ("other", source),
    )
    for revision, notes in ((result, ["edited"]), (source, []), (sibling, [])):
        array = graph.data(revision)
        assert array.values[0]["notes"] == notes
        assert array.x.values[0].notes == notes
        assert array.x.attrs["notes"] == notes
        assert array.x.encoding["notes"] == notes


def test_unversioned_source_and_public_reads_isolate_mutable_buffers(graph, tmp_path):
    import dask.array as da
    from dask import delayed

    values = np.array([{"items": [1]}], dtype=object)
    incoming = xr.DataArray(
        da.from_delayed(delayed(lambda: values)(), shape=(1,), dtype=object),
        dims="x",
        coords={"x": [1], "label": ("x", [12])},
        attrs={"labels": ["sample"]},
    )
    # Xarray can propagate the source encoding through an arbitrary calculation.
    # This path says nothing about the buffer closed over by the delayed task.
    incoming.encoding["source"] = str(tmp_path / "old-source")
    incoming.x.attrs["calibration"] = {"labels": ["position"]}
    incoming.x.encoding["metadata"] = {"codes": [3]}
    revision = graph.add_source("A", incoming)
    values[0]["items"].append(2)
    incoming.attrs["labels"].clear()
    incoming.label.values[:] = -1
    incoming.x.attrs["calibration"]["labels"].clear()
    incoming.x.encoding["metadata"]["codes"].clear()
    editable = graph.data(revision)
    editable.values[0]["items"].clear()
    editable.attrs["labels"].clear()
    editable.label.values[:] = -2
    editable.x.attrs["calibration"]["labels"].append("edited")
    editable.x.encoding["metadata"]["codes"].append(4)

    original = graph.data(revision)
    assert original.values[0] == {"items": [1]}
    assert original.x.attrs["calibration"]["labels"] == ["position"]
    assert original.x.encoding["metadata"]["codes"] == [3]
    assert original.attrs == {"labels": ["sample"]}
    np.testing.assert_array_equal(original.label.values, [12])


@pytest.mark.parametrize("chunked", [False, True])
def test_versioned_source_reads_only_requested_chunks_and_checks_file(
    graph, tmp_path, monkeypatch, chunked
):
    from xarray.backends.h5netcdf_ import H5NetCDFArrayWrapper

    from erlab._provenance import FileVersion
    from erlab._provenance._operations import SelectCoordOperation

    path = tmp_path / "source.nc"
    volume = np.arange(8 * 12).reshape(8, 12)
    source = xr.DataArray(
        volume, dims=("y", "x"), coords={"line": ("x", np.arange(12))}, name="volume"
    )
    source.to_netcdf(
        path, engine="h5netcdf", encoding={"volume": {"chunksizes": (2, 4)}}
    )
    reads = []
    original_get = H5NetCDFArrayWrapper._getitem

    def record_get(self, key):
        reads.append((self.variable_name, key))
        return original_get(self, key)

    monkeypatch.setattr(H5NetCDFArrayWrapper, "_getitem", record_get)
    version = FileVersion.capture(path)
    with xr.open_dataarray(
        path, engine="h5netcdf", chunks={} if chunked else None
    ) as incoming:
        revision = graph.add_source("A", incoming, backing=version)
    line = graph.derive(
        "line", Recipe.bind(SelectCoordOperation(coord_name="line"), Follow("A"))
    )
    assert graph.entry("line").revision_id == line
    assert reads == []
    np.testing.assert_array_equal(graph.data(line).values, np.arange(12))
    result = graph.data(revision)
    assert result.chunks == ((2, 2, 2, 2), (4, 4, 4))
    tile = result.isel(y=slice(2, 4), x=slice(8, 12))
    np.testing.assert_array_equal(tile.values, volume[2:4, 8:12])
    assert [key for name, key in reads if name == "volume"] == [
        (slice(2, 4, 1), slice(8, 12, 1))
    ]

    # Replace the file after the caller has obtained a deferred result.
    replacement = tmp_path / "replacement.nc"
    (source + 1000).to_netcdf(replacement, engine="h5netcdf")
    replacement.replace(path)
    with pytest.raises(OSError, match="has changed"):
        tile.compute()


def test_file_backed_checkpoint_releases_source_after_last_owner(tmp_path):
    import gc
    import weakref

    import psutil

    from erlab._provenance import FileLoad, FileVersion
    from erlab._provenance._model import FileDataSelection, FileReplayCall

    path = tmp_path / "scan.nc"
    data = xr.DataArray(np.arange(4))
    data.to_netcdf(path, engine="h5netcdf")

    def read_file(source, call):
        version = FileVersion.capture(source.path)
        with xr.open_dataarray(source.path, engine="h5netcdf") as opened:
            return opened, version

    graph = ProvenanceGraph(read_file=read_file)
    for _ in range(6):
        source = graph.file_source(str(path))
        source_ref = weakref.ref(source)
        load = FileLoad(
            source,
            FileReplayCall(
                kind="callable",
                target="xarray.open_dataarray",
                selection=FileDataSelection(kind="dataarray"),
            ),
        )
        revision = graph.derive("scan", load)
        with graph.retain(revision):
            graph.remove_entry("scan")
            del load, source
            assert source_ref() is graph.file_source(str(path))
            np.testing.assert_array_equal(graph.data(revision).values, data.values)
        gc.collect()
        assert source_ref() is None
        assert graph.revision_ids == frozenset()
        assert str(path.resolve()) not in {
            f.path for f in psutil.Process().open_files()
        }


def test_saved_operations_without_qt(tmp_path):
    # Import the actual catalog only after installing the Qt import check.
    code = textwrap.dedent("""
        import pathlib
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
        suite["_check_saved_operations"]()
        suite["_check_saved_file_call"](pathlib.Path(sys.argv[2]))
    """)
    subprocess.run([sys.executable, "-c", code, __file__, str(tmp_path)], check=True)


def _check_saved_operations():
    import erlab
    from erlab._provenance import Application
    from erlab._provenance._model import parse_tool_provenance_operation
    from erlab._provenance._operations import (
        AffineCoordOperation,
        ConcatOperation,
        KspaceWorkFunctionOperation,
        NormalizeOperation,
        RemoveMeshOperation,
        RenameOperation,
        TransposeOperation,
    )

    data = xr.DataArray(
        np.array([-1, 1, 2, 6], dtype=np.float32).reshape(2, 2),
        dims=("y", "x"),
        coords={"x": np.arange(2, dtype=np.float32) * 1e-6, "ref": 21.2},
        name="signal",
        attrs={"sample": {"labels": ["original"]}},
    )
    before = data.copy(deep=True)
    # This expectation does not invoke the normalization helper under test.
    normalized = data.copy(
        data=np.array([[np.nan, np.nan], [0.5, 1.5]], dtype="float32")
    )
    setter_result = data.copy(deep=True)
    setter_result.kspace.analyzer_work_function = 4.2
    operand = (("data", "A"),)
    with xr.set_options(keep_attrs=True):
        cases = [
            (RenameOperation(name="data"), operand, {"A": data}, data.rename("data")),
            (
                TransposeOperation(dims=("x", "y")),
                operand,
                {"A": data},
                data.transpose("x", "y"),
            ),
            (
                AffineCoordOperation(
                    coord_name="x",
                    scale=1,
                    offset=21.2,
                    offset_coord="ref",
                    offset_coord_sign=-1,
                ),
                operand,
                {"A": data},
                data,
            ),
            (NormalizeOperation(dims=("x",)), operand, {"A": data}, normalized),
            (
                KspaceWorkFunctionOperation(work_function=4.2),
                operand,
                {"A": data},
                setter_result,
            ),
            (
                ConcatOperation(dim="part", input_count=3),
                (*operand, ("other_0", "B"), ("other_1", "A")),
                {"A": data, "B": -data},
                xr.concat([data, -data, data], dim="part"),
            ),
        ]
        spectrum = xr.DataArray(
            np.arange(1, 65, dtype=float).reshape(8, 8),
            dims=("alpha", "eV"),
            coords={"alpha": np.arange(8), "eV": np.arange(8)},
        )
        peaks = ((4, 4), (4, 6), (4, 2))
        args = {"first_order_peaks": peaks, "order": 1, "n_pad": 0, "roi_hw": 2}
        corrected, mesh = erlab.analysis.mesh.remove_mesh(spectrum, **args)
        for output, expected in (("corrected", corrected), ("mesh", mesh)):
            cases.append(
                (
                    RemoveMeshOperation(**args, output=output),
                    operand,
                    {"A": spectrum},
                    expected,
                )
            )
        for operation, slots, inputs, expected in cases:
            restored = parse_tool_provenance_operation(
                operation.model_dump(mode="json")
            )
            applied = Application(restored, slots)
            names = {key: f"input_{key}" for key in inputs}
            namespace = {names[k]: v.copy(deep=True) for k, v in inputs.items()}
            namespace.update(xr=xr, np=np, era=erlab.analysis)
            with xr.set_options(keep_attrs=False):
                live = applied.apply(inputs)
                exec(applied.code(names, output_name="result"), namespace)  # noqa: S102
                for result in (live, namespace["result"]):
                    assert result.dtype == expected.dtype
                    xr.testing.assert_identical(result, expected)
    xr.testing.assert_identical(data, before)


def _check_saved_file_call(tmp_path):
    from erlab._provenance._files import _load_file_source_data, _resolve_load_func
    from erlab._provenance._model import (
        FileDataSelection,
        FileLoadSource,
        FileReplayCall,
    )

    expected = xr.DataArray(
        np.array([3, 7], dtype="float32"),
        dims="energy",
        name="counts",
        attrs={"unit": "Hz"},
    )
    path = tmp_path / "scan.nc"
    xr.Dataset({"other": -expected, "counts": expected}).to_netcdf(
        path, engine="h5netcdf"
    )
    selection = FileDataSelection(kind="dataset_variable", value="counts")
    resolved = _resolve_load_func((xr.load_dataset, {"engine": "h5netcdf"}, selection))
    call = FileReplayCall.model_validate_json(resolved.replay_call().model_dump_json())
    record = FileLoadSource(
        path=str(path),
        replay_call=call,
        loader_text=call.target,
        loader_label="Loader",
        kwargs_text="",
    )
    actual = _load_file_source_data(record)
    assert actual.dtype == expected.dtype
    xr.testing.assert_identical(actual, expected)
    from erlab._provenance._files import _registered_local_callable_loader

    saved_loader = _registered_local_callable_loader(call.target)
    restored = _resolve_load_func((saved_loader, call.kwargs, call.selection))
    namespace = {}
    exec(restored.load_code(path, assign="result"), namespace)  # noqa: S102
    xr.testing.assert_identical(namespace["result"], expected)


def test_checkpoint_reuses_only_immutable_numeric_values():
    from erlab._provenance._checkpoint import checkpoint_data

    mutable = np.arange(12)
    for values, share in (
        (np.frombuffer(mutable.tobytes(), dtype=mutable.dtype), True),
        (mutable.view(), False),
    ):
        values.setflags(write=False)
        source = xr.DataArray(values, dims="x", attrs={"groups": [1]})
        owned = checkpoint_data(source)
        assert np.shares_memory(owned.values, source.values) is share
        owned.attrs["groups"].clear()
        assert source.attrs == {"groups": [1]}


@pytest.mark.parametrize("structured", [False, True])
def test_public_read_detaches_objects_in_deferred_results(graph, structured):
    import dask.array as da
    from dask import delayed

    class DeferredObjects:
        def input_slots(self):
            return ()

        def apply(self, data):
            # One durable delayed task can hand out the same Python objects.
            values = (
                np.array([([data.item()],)], dtype=[("scan", object)])
                if structured
                else np.array([{"scan": [data.item()]}], dtype=object)
            )
            return xr.DataArray(
                da.from_delayed(
                    delayed(lambda: values)(), shape=(1,), dtype=values.dtype
                )
            )

    graph.add_source("A", xr.DataArray(12))
    revision = graph.derive("B", Recipe.bind(DeferredObjects(), Follow("A")))
    graph.data(revision).values[0]["scan"].clear()
    assert graph.data(revision).values[0]["scan"] == [12]


@pytest.mark.parametrize("size", [1000, 10000])
def test_unrelated_workspace_rows_do_not_add_refresh_work(graph, size, monkeypatch):
    from erlab._provenance import Pin

    arr = xr.DataArray(1)
    source = graph.add_source("source", arr)
    graph.derive("child", Recipe.bind(Scale(2), Follow("source")))
    for n in range(size - 2):
        graph.add_source(f"row{n}", arr)
    visits = []
    plan_input = graph._plan_input

    def visit(binding, *, mode):
        visits.append(binding)
        return plan_input(binding, mode=mode)

    monkeypatch.setattr(graph, "_plan_input", visit)
    revision = graph.refresh("child")
    assert visits == [Follow("source"), Pin(source)]
    xr.testing.assert_identical(graph.data(revision), xr.DataArray(2))


def test_call_keeps_nested_input_references_distinct_from_literals():
    from erlab._provenance._calls import Call, InputRef

    primary = xr.DataArray(np.ones(2), dims="x")
    source = xr.DataArray([1, 2], dims="x", name="a")
    literal = np.array([3, 5], dtype=np.int16)
    call = Call.method(
        "assign_coords",
        args=({"data": InputRef("grid"), "|grid|": ("x", literal), "label": "grid"},),
    )
    result = call.evaluate({"data": primary, "grid": source})
    scope = {"a": primary.copy(), "b": source.copy(), "np": np}
    code = call.expression({"data": "a", "grid": "b"})
    exec("result = " + code, scope)  # noqa: S102
    expected = primary.assign_coords(
        {"data": source, "|grid|": ("x", literal), "label": "grid"}
    )
    for actual in (result, scope["result"]):
        xr.testing.assert_identical(actual, expected)
        assert actual.coords["|grid|"].dtype == np.int16


def test_recorded_code_preserves_each_steps_settings():
    from erlab._provenance._operations import NormalizeOperation
    from erlab.interactive.imagetool._provenance._recorded_code import full_code

    graph = ProvenanceGraph()
    source = xr.DataArray([1.0, 3.0], dims="x", attrs={"unit": "count"})
    original = graph.add_source("A", source)
    with xr.set_options(keep_attrs=True):
        normalized = graph.derive(
            "B", Recipe.bind(NormalizeOperation(dims=("x",)), Follow("A"))
        )
    with xr.set_options(keep_attrs=False):
        code = full_code(
            graph,
            normalized,
            output_name="result",
            source_names={original: "input_data"},
        )
        scope = {"input_data": source.copy(deep=True)}
        exec(code, scope)  # noqa: S102
        expected = source.copy(data=[0.5, 1.5])
        xr.testing.assert_identical(scope["result"], expected)
        xr.testing.assert_identical(graph.data(normalized), expected)
        edited = graph.edit_application(
            "B", normalized, NormalizeOperation(dims=("x",))
        )
        xr.testing.assert_identical(graph.data(edited), expected)
        refreshed = graph.refresh("B")
        xr.testing.assert_identical(graph.data(refreshed), expected)
        assert xr.get_options()["keep_attrs"] is False


def test_worker_application_keeps_saved_options_at_dispatch():
    from erlab._provenance import Application
    from erlab._provenance._operations import ExtensionRoutineOperation

    operation = ExtensionRoutineOperation(
        script_name="routine.py",
        source_hash="a" * 64,
        routine_id="routine",
        routine_name="Routine",
        parameters={},
        execution_options={"keep_attrs": False},
    )
    with xr.set_options(keep_attrs=True, arithmetic_broadcast=False):
        recipe = Recipe.bind(operation, Follow("A"))
        applied = Application(operation, (("data", "revision"),))
    assert applied.execution == recipe._captured.execution
    assert applied.execution.as_dict()["keep_attrs"] is False
    assert applied.execution.as_dict()["arithmetic_broadcast"] is False
    assert applied.operation.execution_options == applied.execution.as_dict()
    assert applied.local_execution.options == ()
    code = applied.code({"revision": "data"}, output_name="result")
    assert "with xr.set_options" not in code
    assert "'keep_attrs': False" in code


def test_copied_array_method_uses_edited_arguments():
    from erlab._provenance._operations import RenameOperation, TransposeOperation

    source = xr.DataArray([[1], [2]], dims=("x", "y"), name="source")
    rename = RenameOperation(name="first")
    assert rename.apply(source).name == "first"
    updated = rename.model_copy(update={"name": "next"})
    assert updated.apply(source).name == "next"
    assert updated.model_dump(mode="json") == RenameOperation(name="next").model_dump(
        mode="json"
    )
    transpose = TransposeOperation(dims=("y", "x"))
    assert transpose.apply(source).dims == ("y", "x")
    assert (
        transpose.model_copy(update={"dims": ("x", "y")}, deep=True).apply(source).dims
        == source.dims
    )


def test_foreign_dask_graph_with_guarded_name_keeps_its_actual_tasks(tmp_path):
    import dask.array as da

    from erlab._provenance._checkpoint import (
        FileVersion,
        _guard_file_variables,
        _read_versioned_file,
    )

    path = tmp_path / "source.nc"
    original = xr.DataArray(np.arange(8), dims="x", name="scan")
    original.to_netcdf(path, engine="h5netcdf")
    version = FileVersion.capture(path)
    with _read_versioned_file(
        xr.open_dataarray, path, {"engine": "h5netcdf", "chunks": {}}, version
    ) as source:
        # A different graph may use exactly the same public collection name.
        name = source.data.name
        foreign = da.Array(
            {(name, 0): np.arange(8)[::-1]}, name, ((8,),), dtype=source.dtype
        )
        guarded = _guard_file_variables(xr.DataArray(foreign, dims="x"), version)
        np.testing.assert_array_equal(guarded.compute(), np.arange(8)[::-1])
        replacement = tmp_path / "changed.nc"
        replacement.touch()
        replacement.replace(path)
        with pytest.raises(OSError, match="has changed"):
            guarded.compute()


@pytest.mark.parametrize("command", ["refresh", "reload"])
def test_closed_producer_retains_last_displayed_revision_for_each_slot(graph, command):
    source = graph.add_source("input", xr.DataArray(2))
    assert graph.next_inputs("input") == ()
    produced = graph.derive("scale", Recipe.bind(Scale(3), Follow("input")))
    graph.derive("sum", Recipe.bind(Add(), Follow("scale"), other=Follow("scale")))
    graph.replace("input", Recipe.bind(Scale(5), Follow("input")))
    assert graph.is_stale("scale")
    graph.remove_entry("scale")

    expected_slots = (("data", produced), ("other", produced))
    assert graph.next_inputs("sum") == expected_slots
    assert graph.is_virtual(produced)
    plan = graph.plan_update("sum", command=command)
    assert plan.current_inputs == (("scale", produced),)
    assert plan.entries == ("sum",)
    updated = getattr(graph, command)("sum")
    assert graph.revision(updated).producer.inputs == expected_slots
    assert [r.revision_id for r in graph.recorded_lineage(updated)] == [
        source,
        produced,
        updated,
    ]
    xr.testing.assert_identical(graph.data(updated), xr.DataArray(12))

    graph.remove_entry("sum")
    for query in (graph.auto_error, graph.is_stale):
        with pytest.raises(KeyError, match="sum"):
            query("sum")
    with pytest.raises(KeyError, match=updated):
        graph.is_virtual(updated)


def test_checkpoint_notifications_and_editor_errors_preserve_current_data(graph):
    from erlab._provenance._operations import RenameOperation

    data = xr.DataArray([1, 2], name="input")
    graph.adopt_checkpoint("source", "saved", data)
    graph.derive("copy", Recipe.bind(RenameOperation(name="first"), Follow("source")))
    graph.adopt_checkpoint("source", "saved", data)
    assert not graph.is_stale("copy")
    before = graph.current_revision("copy")
    sources = graph.followed_entries("copy")
    with pytest.raises(TypeError, match="checkpoint must be a DataArray"):
        graph.adopt_checkpoint("source", "invalid", [3, 4])
    with pytest.raises(TypeError, match="no editable operation"):
        graph.edit_application("source", "saved", RenameOperation(name="bad"))
    with pytest.raises(ValueError, match="Reopen the editor"):
        graph.edit_application("copy", "obsolete", RenameOperation(name="bad"))
    with pytest.raises(KeyError, match="missing-parent"):
        graph.add_source("other", data, parent_id="missing-parent")
    with pytest.raises(ValueError, match="Expected refresh or reload"):
        graph.plan_update("copy", command="replace")

    assert graph.entry_ids == ("source", "copy")
    assert graph.revision_ids == {"saved", before}
    assert graph.current_revision("source") == "saved"
    assert graph.followed_entries("copy") == sources == {"source"}
    assert not graph.is_stale("copy")
    xr.testing.assert_identical(graph.data(before), data.rename("first"))


def test_admission_rechecks_row_after_copying_external_metadata(graph):
    class LateArrival:
        def __deepcopy__(self, memo):
            graph.add_source("row", xr.DataArray([17]))
            return "copied"

    arriving = xr.DataArray([99], attrs={"metadata": LateArrival()})
    with pytest.raises(ValueError, match="already exists"):
        graph.add_source("row", arriving)
    current = graph.current_revision("row")
    assert graph.revision_ids == {current}
    xr.testing.assert_identical(graph.data(current), xr.DataArray([17]))


def test_failed_file_admission_and_replacement_preserve_existing_row(file_graph):
    from erlab._provenance import FileLoad, MissingFileSource

    graph, path, load, reads = file_graph
    path.unlink()
    with pytest.raises(MissingFileSource) as failure:
        graph.derive("file", load)
    assert failure.value.source == load.source
    assert failure.value.displayed_revision_id is None
    assert graph.entry_ids == ()
    assert graph.revision_ids == set()
    xr.DataArray([2]).to_netcdf(path, engine="h5netcdf")
    source = graph.add_source("existing", xr.DataArray([3]))
    loaded = graph.replace("existing", load)
    assert loaded != source
    assert graph.next_inputs("existing") == (("file", load.source.source_id),)
    xr.testing.assert_identical(graph.data(loaded), xr.DataArray([2]))

    unknown = ProvenanceGraph()
    with pytest.raises(ValueError, match="needs a path"):
        unknown.file_source("")
    with pytest.raises(ValueError, match="Unknown file source"):
        unknown.derive("file", load)
    with pytest.raises(ValueError, match="No file reader"):
        unknown.derive("file", FileLoad(unknown.file_source(str(path)), load.call))
    assert not unknown.revision_ids
    assert reads == [str(path), str(path)]


@pytest.mark.parametrize("failure", ["non-array", "missing-operation-input"])
def test_operation_failures_do_not_publish_or_report_an_unrelated_source(
    graph, failure
):
    from erlab._provenance import MissingFileSource

    class Fails(Scale):
        def apply(self, data):
            if failure == "non-array":
                return data.values
            raise FileNotFoundError("analysis calibration")

    original = graph.add_source("source", xr.DataArray(7))
    old = graph.derive("child", Recipe.bind(Scale(2), Follow("source")))
    graph.set_recipe("child", Recipe.bind(Fails(2), Follow("source")))
    error = TypeError if failure == "non-array" else FileNotFoundError
    message = (
        "must produce a DataArray" if failure == "non-array" else "analysis calibration"
    )
    with pytest.raises(error, match=message) as caught:
        graph.refresh("child")
    assert not isinstance(caught.value, MissingFileSource)
    assert graph.current_revision("child") == old
    assert graph.is_stale("child")
    assert graph.revision_ids == {old, original}
    xr.testing.assert_identical(graph.data(old), xr.DataArray(14))


@pytest.mark.parametrize("when", ["choosing", "installing"])
def test_reload_destination_conflict_preserves_independent_publication(
    file_graph, when
):
    from contextlib import contextmanager

    from erlab._provenance import FileLoad

    _, path, load, _ = file_graph
    active = False

    @contextmanager
    def install(updates):
        if active and when == "installing":
            assert {u.entry_id for u in updates} == {"fork"}
            graph.add_source("fork", xr.DataArray(19))
        yield

    graph = ProvenanceGraph(
        read_file=lambda source, call: xr.load_dataarray(source.path, **call.kwargs),
        install=install,
    )
    original = graph.derive("source", FileLoad(graph.file_source(str(path)), load.call))
    xr.DataArray([3, 5]).to_netcdf(path, engine="h5netcdf")
    active = True

    def choose(array):
        xr.testing.assert_identical(array, xr.DataArray([3, 5]))
        if when == "choosing":
            graph.add_source("fork", xr.DataArray(19))
        return "fork"

    with pytest.raises(ValueError, match="already exists"):
        graph.reload("source", choose_destination=choose)
    fork = graph.current_revision("fork")
    assert graph.revision_ids == {original, fork}
    assert graph.current_revision("source") == original
    xr.testing.assert_identical(graph.data(original), xr.DataArray([1, 2]))
    xr.testing.assert_identical(graph.data(fork), xr.DataArray(19))


@pytest.mark.parametrize("edited", [False, True])
def test_repeated_producer_compares_nested_coordinate_selections(
    graph, monkeypatch, edited
):
    from erlab._provenance import EarlierStep
    from erlab._provenance._operations import SelOperation

    coords = [0.5, 1.5, 2.5]
    graph.add_source(
        "source",
        xr.DataArray(
            np.arange(9).reshape(3, 3),
            dims=("x", "y"),
            coords={"x": coords, "y": coords},
        ),
    )

    def selection(end):
        return Recipe.bind(
            SelOperation(kwargs={"x": slice(0.5, end), "y": [0.5, 2.5]}),
            Follow("source"),
        )

    old = graph.derive("selected", selection(2.5))
    graph.derive("sum", Recipe.bind(Add(), Follow("selected"), other=EarlierStep(old)))
    graph.set_recipe("selected", selection(1.5 if edited else 2.5))
    calls = []
    apply = SelOperation.apply

    def count_apply(operation, data):
        calls.append(operation.kwargs["x"].stop)
        return apply(operation, data)

    monkeypatch.setattr(SelOperation, "apply", count_apply)
    refreshed = graph.reload("sum")
    expected = np.array([[0, 4], [6, 10], [12, 16]])
    if edited:
        expected = expected[:2]
    actual = graph.data(refreshed)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual.x, coords[: expected.shape[0]])
    np.testing.assert_array_equal(actual.y, [0.5, 2.5])
    assert sorted(calls) == ([1.5, 2.5] if edited else [2.5])


def test_invalid_refresh_bindings_preserve_the_previous_recipe_and_value(graph):
    from erlab._provenance import EarlierStep

    source = graph.add_source("source", xr.DataArray(5))
    child = graph.derive("child", Recipe.bind(Scale(3), Follow("source")))
    with pytest.raises(ValueError, match="source revision has no earlier"):
        graph.set_recipe("child", Recipe.bind(Scale(7), EarlierStep(source)))
    with pytest.raises(TypeError, match="Invalid refresh binding"):
        graph.set_recipe("child", Recipe.bind(Scale(7), "source"))
    with pytest.raises(TypeError, match="Expected an operation recipe"):
        graph.set_recipe("child", Follow("source"))
    assert graph.revision_ids == {source, child}
    assert not graph.is_stale("child")
    assert graph.next_inputs("child") == (("data", source),)
    xr.testing.assert_identical(graph.data(graph.refresh("child")), xr.DataArray(15))


def test_derived_revision_captures_buffers_retained_by_operation(graph):
    retained = []

    class RetainResult(Scale):
        def apply(self, data):
            result = (self.factor * data).assign_coords(detector=("x", [2.0, 4.0]))
            result.attrs["labels"] = ["accepted"]
            retained.append(result)
            return result

    graph.add_source("input", xr.DataArray([3.0, 5.0], dims="x"))
    key = graph.derive("output", Recipe.bind(RetainResult(2), Follow("input")))
    original = retained.pop()
    expected = original.copy(deep=True)
    original.values[:] = -1
    original.detector.values[:] = -2
    original.attrs["labels"].clear()
    xr.testing.assert_identical(graph.data(key), expected)


def test_repeated_earlier_producer_updates_once_while_an_unrelated_row_changes(graph):
    from erlab._provenance import EarlierStep

    calls = []
    updating = False

    class RecordScale(Scale):
        def apply(self, data):
            if updating:
                calls.append(data.item())
                graph.adopt_checkpoint("independent", "second", xr.DataArray(17))
            return super().apply(data)

    graph.add_source("source", xr.DataArray(2))
    graph.adopt_checkpoint("independent", "first", xr.DataArray(11))
    earlier = graph.derive("scaled", Recipe.bind(RecordScale(3), Follow("source")))
    last = EarlierStep(earlier)
    graph.derive("sum", Recipe.bind(Add(), last, other=last))
    graph.replace("source", Recipe.bind(Scale(5), Follow("source")))
    updating = True
    result = graph.reload("sum")
    used = dict(graph.revision(result).producer.inputs)
    assert used["data"] == used["other"] != earlier
    assert calls == [10]
    assert graph.current_revision("independent") == "second"
    assert graph.current_revision("scaled") == earlier
    xr.testing.assert_identical(graph.data(result), xr.DataArray(60))
