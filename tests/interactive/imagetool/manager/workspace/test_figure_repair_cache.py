import contextlib
import gc
import json
import pathlib
import shutil
import threading
import types
import typing
import uuid
import weakref
from dataclasses import dataclass

import dask
import dask.array as da
import h5py
import numpy as np
import pytest
import xarray as xr
from dask import delayed

import erlab.interactive.imagetool.manager._workspace._arrays as workspace_arrays
import erlab.interactive.imagetool.manager._workspace._figure_repair as figure_repair
import erlab.interactive.imagetool.manager._workspace._storage as workspace_storage
import erlab.interactive.imagetool.manager._workspace._store as workspace_store
import erlab.interactive.utils as interactive_utils
from erlab.interactive import _persistence_constants
from erlab.interactive._figurecomposer import FigureComposerTool, FigureSourceState
from erlab.interactive.imagetool import itool
from tests.interactive.imagetool.manager.helpers import adopt_workspace_path
from tests.interactive.imagetool.manager.workspace._support import (
    _current_workspace_payload_attrs,
    _current_workspace_payload_path,
)


@dataclass
class _RepairCase:
    manager: typing.Any
    saver: typing.Any
    sources: tuple[typing.Any, ...]
    node: typing.Any
    captured: xr.DataArray
    path: pathlib.Path
    payload: tuple[pathlib.Path, str]
    attrs: dict[str, typing.Any]

    def probe(self, *, payload=None, attrs=None) -> xr.Dataset | None:
        return self.saver._repair_saved_workspace_figure_dataset(
            self.node,
            self.payload if payload is None else payload,
            self.attrs if attrs is None else attrs,
        )


@pytest.fixture
def figure_repair_case(qtbot, monkeypatch, tmp_path, manager_context, request):
    difference = getattr(request, "param", "values")
    data = xr.DataArray(
        np.arange(48.0).reshape(6, 8),
        dims=("x", "y"),
        coords={
            "x": np.arange(6.0),
            "y": np.arange(8.0),
            "auxiliary": ("x", np.arange(6.0) * 2.0),
        },
        attrs={
            "units": "counts",
            "calibration": np.array([1.0, 2.0]),
            "zero": 0.0,
        },
        name="cut",
    )
    data.coords["x"].attrs["units"] = "degrees"
    captured = data.copy(deep=True)
    if difference == "values":
        captured.values[0, 0] = -1.0
    elif difference == "attrs":
        captured.attrs["units"] = "normalized"
    elif difference == "coords":
        values = captured.coords["x"].values.copy()
        values[-1] = 5.5
        captured.coords["x"] = captured.coords["x"].copy(data=values)
    elif difference == "coord_attrs":
        captured.coords["x"].attrs["units"] = "radians"
    elif difference == "name":
        captured = captured.rename("historical_cut")
    elif difference == "dtype":
        captured = captured.astype(np.float32)
    else:
        raise ValueError(difference)

    with manager_context() as manager:
        sources = []
        for _ in range(2):
            index = manager.add_imagetool(
                itool(data.copy(deep=True), manager=False, execute=False), show=False
            )
            sources.append(manager._node_for_target(index))
        figure = FigureComposerTool.from_sources(
            {"first": captured, "second": captured.copy(deep=True)},
            sources=tuple(
                FigureSourceState(
                    name=name,
                    node_uid=source.uid,
                    node_snapshot_token=source.snapshot_token,
                )
                for name, source in zip(("first", "second"), sources, strict=True)
            ),
            operations=(),
            primary_source="first",
        )
        uid = manager.add_figuretool(figure, show=False)
        for source in sources:
            source._advance_snapshot_token()
        saver = manager._workspace_controller.saving
        path = tmp_path / "historical-figure.itws"
        # Preserve the historical embedded payload on the first save.
        with monkeypatch.context() as context:
            context.setattr(
                saver,
                "_repair_workspace_figure_dataset",
                lambda _node, _ds: figure_repair._WorkspaceFigureRepairResult(),
            )
            saver._save_workspace_document(path)
        adopt_workspace_path(manager, path)
        payload = (path, _current_workspace_payload_path(path, f"figures/{uid}"))
        attrs = _current_workspace_payload_attrs(path, f"figures/{uid}")
        saved = workspace_arrays._read_workspace_dataset_group_h5py(
            *payload, preferred_data_name=_persistence_constants.SAVED_TOOL_DATA_NAME
        )
        assert saved is not None
        assert saved[_persistence_constants.SAVED_TOOL_DATA_NAME].size == data.size
        assert _persistence_constants.TOOL_DATA_BLOB_NAME_ATTR in saved["second"].attrs
        assert not FigureComposerTool._saved_tool_data_references(saved)
        saved.close()
        yield _RepairCase(
            manager,
            saver,
            tuple(sources),
            manager._child_node(uid),
            captured,
            path,
            payload,
            attrs,
        )


@pytest.fixture
def repair_counts(monkeypatch, figure_repair_case):
    counts = {"decode": 0, "compare": 0}
    tool_cls = interactive_utils.ToolWindow
    original_decode = tool_cls._tool_data_items_from_dataset
    original_compare = figure_repair_case.saver._workspace_figure_arrays_identical

    def decode(cls, *args, **kwargs):
        counts["decode"] += 1
        return original_decode(*args, **kwargs)

    def compare(*args, **kwargs):
        counts["compare"] += 1
        return original_compare(*args, **kwargs)

    monkeypatch.setattr(tool_cls, "_tool_data_items_from_dataset", classmethod(decode))
    monkeypatch.setattr(
        figure_repair_case.saver, "_workspace_figure_arrays_identical", compare
    )
    return counts


def test_unchanged_historical_figure_skips_decode_and_compare(
    figure_repair_case, repair_counts
):
    assert figure_repair_case.probe() is None
    assert repair_counts["decode"] == repair_counts["compare"] == 2
    first = repair_counts.copy()
    assert figure_repair_case.probe() is None
    assert repair_counts == first


@pytest.mark.parametrize(
    "change", ["token", "attrs", "numpy_attrs", "coord_values", "coord_attrs", "role"]
)
def test_current_input_changes_invalidate_negative_repair(
    figure_repair_case, repair_counts, change
):
    case = figure_repair_case
    assert case.probe() is None
    first = repair_counts.copy()
    source = case.sources[0]
    token = source.snapshot_token
    attrs = None
    if change == "token":
        source._advance_snapshot_token()
    elif change == "attrs":
        source.slicer_area._data.attrs["units"] = "normalized"
    elif change == "numpy_attrs":
        source.slicer_area._data.attrs["calibration"][1] = 3.0
    elif change == "coord_values":
        coordinate = source.slicer_area._data.coords["x"]
        values = coordinate.values.copy()
        values[-1] = 5.5
        source.slicer_area._data.coords["x"] = coordinate.copy(data=values)
    elif change == "coord_attrs":
        source.slicer_area._data.coords["x"].attrs["units"] = "radians"
    elif change == "role":
        attrs = dict(case.attrs)
        state = json.loads(attrs["tool_state"])
        state["sources"][0]["data_role"] = "source"
        state["sources"][0]["provider_selection"]["data_role"] = "source"
        attrs["tool_state"] = json.dumps(state)
    if change != "token":
        assert source.snapshot_token == token
    assert case.probe(attrs=attrs) is None
    assert repair_counts["decode"] > first["decode"]
    assert repair_counts["compare"] > first["compare"]


@pytest.mark.parametrize(
    "change", ["workspace_path", "payload_path", "workspace_id", "file"]
)
def test_saved_payload_identity_changes_invalidate_negative_repair(
    figure_repair_case, repair_counts, tmp_path, change
):
    case = figure_repair_case
    copy_path = tmp_path / "copy.itws"
    shutil.copyfile(case.path, copy_path)
    payload = (copy_path, case.payload[1])
    if change == "payload_path":
        with h5py.File(copy_path, "r+") as h5_file:
            h5_file.copy(case.payload[1], "/objects/second-figure-object")
    assert case.probe(payload=payload) is None
    first = repair_counts.copy()
    if change == "workspace_path":
        payload = case.payload
    elif change == "payload_path":
        payload = (copy_path, "/objects/second-figure-object")
    elif change == "workspace_id":
        with h5py.File(copy_path, "r+") as h5_file:
            h5_file.attrs[_persistence_constants.WORKSPACE_ID_ATTR] = uuid.uuid4().hex
    elif change == "file":
        replacement = tmp_path / "replacement.itws"
        shutil.copyfile(copy_path, replacement)
        replacement.replace(copy_path)
    assert case.probe(payload=payload) is None
    assert repair_counts["decode"] > first["decode"]
    assert repair_counts["compare"] > first["compare"]


def test_pending_source_binding_changes_invalidate_negative_repair(
    figure_repair_case, repair_counts, tmp_path
):
    case = figure_repair_case
    source = case.sources[0]
    source_payload = _current_workspace_payload_path(case.path, "0")
    source_attrs = _current_workspace_payload_attrs(case.path, "0")
    source.set_pending_workspace_memory_payload(case.path, source_payload, source_attrs)
    assert case.probe() is None
    first = repair_counts.copy()
    token = source.snapshot_token
    copy_path = tmp_path / "other-source.itws"
    shutil.copyfile(case.path, copy_path)
    source.set_pending_workspace_memory_payload(copy_path, source_payload, source_attrs)
    assert source.snapshot_token == token
    assert case.probe() is None
    assert repair_counts["decode"] > first["decode"]
    assert repair_counts["compare"] > first["compare"]


def test_lazy_source_metadata_and_witness_release_workspace_lock(
    figure_repair_case, repair_counts, monkeypatch
):
    case = figure_repair_case
    store = workspace_store.WorkspaceStore.active(case.path)
    owned_store = store is None
    if store is None:
        store = workspace_store.WorkspaceStore(case.path)
    blocked = []
    seen = {"metadata": 0, "witness": 0}

    def check_worker_can_read(stage):
        acquired = []

        def acquire_from_worker():
            locked = store.lock.acquire(blocking=False)
            acquired.append(locked)
            if locked:
                store.lock.release()

        worker = threading.Thread(target=acquire_from_worker)
        worker.start()
        worker.join(timeout=2.0)
        assert not worker.is_alive()
        blocked.extend(not value for value in acquired)
        seen[stage] += 1

    original_metadata = figure_repair.metadata_fingerprint
    original_witness = figure_repair._WorkspaceFigureMismatch.still_differs

    def metadata(data):
        check_worker_can_read("metadata")
        # A regression must fail the worker-lock assertion without hanging pytest.
        with dask.config.set(scheduler="synchronous"):
            return original_metadata(data)

    def witness(mismatch, current):
        check_worker_can_read("witness")
        with dask.config.set(scheduler="synchronous"):
            return original_witness(mismatch, current)

    monkeypatch.setattr(figure_repair, "metadata_fingerprint", metadata)
    monkeypatch.setattr(
        figure_repair._WorkspaceFigureMismatch, "still_differs", witness
    )
    try:
        for index, source in enumerate(case.sources):
            source.set_pending_workspace_memory_payload(
                case.path,
                _current_workspace_payload_path(case.path, str(index)),
                _current_workspace_payload_attrs(case.path, str(index)),
            )
            current = source.data_for_role("displayed")
            assert isinstance(current.data, da.Array)
            assert isinstance(current.coords["auxiliary"].data, da.Array)
        assert case.probe() is None
        first = repair_counts.copy()
        assert case.probe() is None
        assert repair_counts == first
        assert seen["metadata"] > 0
        assert seen["witness"] > 0
        assert not any(blocked)
    finally:
        if owned_store:
            store.close()


@pytest.mark.parametrize(
    "figure_repair_case",
    ["attrs", "coords", "coord_attrs", "name", "dtype"],
    indirect=True,
)
def test_metadata_mismatch_is_cached_and_later_matching_source_is_repaired(
    figure_repair_case, repair_counts
):
    case = figure_repair_case
    assert case.probe() is None
    first = repair_counts.copy()
    assert case.probe() is None
    assert repair_counts == first
    for source in case.sources:
        source.slicer_area.replace_source_data(case.captured.copy(deep=True))
    repaired = case.probe()
    assert repaired is not None
    try:
        references = FigureComposerTool._saved_tool_data_references(repaired)
        assert len(references) == 2
        assert all(repaired[name].size == 0 for name in references)
    finally:
        repaired.close()
    assert repair_counts["compare"] > first["compare"]


def test_in_place_values_invalidate_mismatch_witness_without_token_change(
    figure_repair_case, repair_counts
):
    case = figure_repair_case
    assert case.probe() is None
    first = repair_counts.copy()
    tokens = tuple(source.snapshot_token for source in case.sources)
    for source in case.sources:
        source.slicer_area._data.values[0, 0] = case.captured.values[0, 0]
    assert tuple(source.snapshot_token for source in case.sources) == tokens
    repaired = case.probe()
    assert repaired is not None
    try:
        assert len(FigureComposerTool._saved_tool_data_references(repaired)) == 2
    finally:
        repaired.close()
    assert repair_counts["compare"] > first["compare"]


@pytest.mark.parametrize("difference", ["attrs_order", "signed_zero"])
def test_equivalent_metadata_digest_difference_is_not_a_mismatch_proof(
    figure_repair_case, repair_counts, monkeypatch, difference
):
    case = figure_repair_case
    tokens = tuple(source.snapshot_token for source in case.sources)
    for source in case.sources:
        attrs = source.slicer_area._data.attrs
        if difference == "attrs_order":
            source.slicer_area._data.attrs = dict(reversed(tuple(attrs.items())))
        else:
            attrs["zero"] = -0.0
        assert xr.Dataset(attrs=case.captured.attrs).identical(
            xr.Dataset(attrs=source.slicer_area._data.attrs)
        )
        assert figure_repair.metadata_fingerprint(
            case.captured
        ) != figure_repair.metadata_fingerprint(source.slicer_area._data)
    original = workspace_storage._workspace_dataset_values_equal

    def compare_without_scalar_witness(left, right, **kwargs):
        kwargs.pop("on_mismatch", None)
        return original(left, right, **kwargs)

    monkeypatch.setattr(
        workspace_storage,
        "_workspace_dataset_values_equal",
        compare_without_scalar_witness,
    )
    assert case.probe() is None
    first = repair_counts.copy()
    assert case.probe() is None
    assert repair_counts["decode"] > first["decode"]
    for source in case.sources:
        source.slicer_area._data.values[0, 0] = case.captured.values[0, 0]
    assert tuple(source.snapshot_token for source in case.sources) == tokens
    repaired = case.probe()
    assert repaired is not None
    try:
        assert len(FigureComposerTool._saved_tool_data_references(repaired)) == 2
    finally:
        repaired.close()


def test_negative_repair_cache_evicts_oldest_entry(figure_repair_case, repair_counts):
    case = figure_repair_case
    nodes = [types.SimpleNamespace(uid=f"cache-figure-{index}") for index in range(65)]
    for node in nodes:
        assert (
            case.saver._repair_saved_workspace_figure_dataset(
                node, case.payload, case.attrs
            )
            is None
        )
    assert len(case.saver._figure_repair_cache) == 64
    first = repair_counts.copy()
    assert (
        case.saver._repair_saved_workspace_figure_dataset(
            nodes[-1], case.payload, case.attrs
        )
        is None
    )
    assert repair_counts == first
    assert (
        case.saver._repair_saved_workspace_figure_dataset(
            nodes[0], case.payload, case.attrs
        )
        is None
    )
    assert repair_counts["decode"] > first["decode"]


def test_update_values_preview_keeps_source_and_negative_cache_valid(
    figure_repair_case, repair_counts
):
    case = figure_repair_case
    assert case.probe() is None
    first = repair_counts.copy()
    for source in case.sources:
        before = source.data_for_role("displayed").copy(deep=True)
        token = source.snapshot_token
        source.slicer_area.update_values(case.captured.values.copy(), update=False)
        xr.testing.assert_identical(source.data_for_role("displayed"), before)
        assert source.snapshot_token == token
    assert case.probe() is None
    assert repair_counts == first


@pytest.mark.parametrize("location", ["source", "payload"])
def test_unsupported_metadata_is_compared_again_instead_of_dropped_from_key(
    figure_repair_case, repair_counts, location
):
    case = figure_repair_case
    attrs = dict(case.attrs)
    if location == "source":
        case.sources[0].slicer_area._data.attrs["custom_metadata"] = object()
    else:
        attrs["custom_metadata"] = object()
    assert case.probe(attrs=attrs) is None
    first = repair_counts.copy()
    assert case.probe(attrs=attrs) is None
    assert repair_counts["decode"] > first["decode"]
    assert repair_counts["compare"] > first["compare"]


def test_source_change_during_comparison_cannot_publish_stale_negative_cache(
    figure_repair_case, repair_counts, monkeypatch
):
    case = figure_repair_case
    original = case.saver._workspace_figure_arrays_identical

    def compare_and_change_source(stored, current, **kwargs):
        result = original(stored, current, **kwargs)
        current.values[0, 0] = stored.values[0, 0]
        return result

    with monkeypatch.context() as context:
        context.setattr(
            case.saver, "_workspace_figure_arrays_identical", compare_and_change_source
        )
        assert case.probe() is None
    first = repair_counts.copy()
    repaired = case.probe()
    assert repaired is not None
    try:
        assert len(FigureComposerTool._saved_tool_data_references(repaired)) == 2
    finally:
        repaired.close()
    assert repair_counts["compare"] > first["compare"]


@pytest.mark.parametrize("stage", ["read", "decode", "compare"])
def test_failed_repair_probes_retry_instead_of_caching(
    figure_repair_case, repair_counts, monkeypatch, stage
):
    case = figure_repair_case
    if stage == "read":
        target, name = workspace_arrays, "_read_workspace_dataset_group_h5py"
    elif stage == "decode":
        target, name = interactive_utils.ToolWindow, "_tool_data_items_from_dataset"
    else:
        target, name = case.saver, "_workspace_figure_arrays_identical"
    original = getattr(target, name)
    failed = False

    def fail_once(*args, **kwargs):
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("Temporary input read failure")
        return original(*args, **kwargs)

    if stage == "decode":
        monkeypatch.setattr(target, name, staticmethod(fail_once))
    else:
        monkeypatch.setattr(target, name, fail_once)
    assert case.probe() is None
    assert failed
    first = repair_counts.copy()
    assert case.probe() is None
    assert repair_counts["decode"] > first["decode"]
    second = repair_counts.copy()
    assert case.probe() is None
    assert repair_counts == second


def test_negative_cache_releases_datasets_and_decoded_blobs(
    figure_repair_case, monkeypatch
):
    case = figure_repair_case
    references = []
    blobs = []
    original_open = workspace_arrays.open_workspace_dataset
    original_read = workspace_arrays._read_workspace_dataset_group_h5py
    original_blob = interactive_utils._tool_data_from_blob

    def track_open(*args, **kwargs):
        dataset = original_open(*args, **kwargs)
        references.append(weakref.ref(dataset))
        return dataset

    def track_read(*args, **kwargs):
        dataset = original_read(*args, **kwargs)
        if dataset is not None:
            references.append(weakref.ref(dataset))
        return dataset

    def track_blob(blob):
        blobs.append(weakref.ref(blob.values))
        decoded = original_blob(blob)
        references.append(weakref.ref(decoded))
        blobs.append(weakref.ref(decoded.values))
        return decoded

    monkeypatch.setattr(workspace_arrays, "open_workspace_dataset", track_open)
    monkeypatch.setattr(
        workspace_arrays, "_read_workspace_dataset_group_h5py", track_read
    )
    monkeypatch.setattr(interactive_utils, "_tool_data_from_blob", track_blob)
    assert case.probe() is None
    assert len(references) >= 3
    assert len(blobs) == 2
    gc.collect()
    assert all(reference() is None for reference in references)
    assert all(reference() is None for reference in blobs)
    assert case.probe() is None
    assert len(blobs) == 2


@pytest.mark.parametrize("payload_path", ["objects", "tools/figure", "objects/a/child"])
def test_negative_cache_requires_one_immutable_object_path(
    figure_repair_case, payload_path
):
    case = figure_repair_case
    assert (
        case.saver._workspace_figure_repair_cache_state(
            (case.path, payload_path), case.attrs
        )
        is None
    )


def test_negative_cache_ignores_nonfigure_payloads(figure_repair_case):
    case = figure_repair_case
    assert case.saver._workspace_figure_repair_cache_state(case.payload, {}) is None


@pytest.mark.parametrize("source_kind", ["missing", "non_imagetool"])
def test_unavailable_source_does_not_cache_incomplete_comparison(
    figure_repair_case, repair_counts, source_kind
):
    case = figure_repair_case
    attrs = dict(case.attrs)
    state = json.loads(attrs["tool_state"])
    state["sources"][0]["node_uid"] = (
        "missing-source" if source_kind == "missing" else case.node.uid
    )
    attrs["tool_state"] = json.dumps(state)
    assert case.probe(attrs=attrs) is None
    assert case.node.uid not in case.saver._figure_repair_cache
    first = repair_counts.copy()
    assert case.probe(attrs=attrs) is None
    assert repair_counts["decode"] > first["decode"]


@pytest.mark.parametrize("change", ["identity", "object", "schema"])
def test_negative_cache_rejects_incomplete_or_mutable_backing(
    figure_repair_case, tmp_path, change
):
    case = figure_repair_case
    path = tmp_path / "unsupported-backing.itws"
    shutil.copyfile(case.path, path)
    with h5py.File(path, "r+") as h5_file:
        if change == "identity":
            del h5_file.attrs[_persistence_constants.WORKSPACE_ID_ATTR]
        elif change == "object":
            del h5_file[case.payload[1]]
        else:
            h5_file.attrs["imagetool_workspace_schema_version"] = 4
    assert (
        case.saver._workspace_figure_repair_cache_state(
            (path, case.payload[1]), case.attrs
        )
        is None
    )


@pytest.mark.parametrize("change", ["closed", "path", "conflicted", "recovery"])
def test_negative_cache_rejects_store_changes_before_reading_inputs(
    figure_repair_case, monkeypatch, tmp_path, change
):
    case = figure_repair_case
    store = workspace_store.WorkspaceStore.active(case.path)
    assert store is not None
    original = workspace_storage._workspace_generation_copy_source

    @contextlib.contextmanager
    def changing_store(path):
        with original(path) as h5_file, monkeypatch.context() as context:
            if change in {"closed", "conflicted"}:
                context.setattr(store, "_state", change)
            elif change == "path":
                context.setattr(store, "_path", tmp_path / "other.itws")
            else:
                context.setattr(store, "_recovery_path", tmp_path / "recovery.itws")
            yield h5_file

    monkeypatch.setattr(
        workspace_storage, "_workspace_generation_copy_source", changing_store
    )
    assert (
        case.saver._workspace_figure_repair_cache_state(case.payload, case.attrs)
        is None
    )


@pytest.mark.parametrize("change", ["path", "recovery", "identity", "object", "file"])
def test_negative_cache_rechecks_backing_after_reading_inputs(
    figure_repair_case, monkeypatch, tmp_path, change
):
    case = figure_repair_case
    store = workspace_store.WorkspaceStore.active(case.path)
    assert store is not None
    payload = case.payload
    if change == "file":
        path = tmp_path / "external.itws"
        shutil.copyfile(case.path, path)
        payload = path, case.payload[1]
    original = figure_repair.metadata_fingerprint
    changed = False

    def change_backing(data):
        nonlocal changed
        fingerprint = original(data)
        if not changed:
            changed = True
            if change == "path":
                context.setattr(store, "_path", tmp_path / "other.itws")
            elif change == "recovery":
                context.setattr(store, "_recovery_path", tmp_path / "recovery.itws")
            else:
                session = (
                    h5py.File(payload[0], "r+")
                    if change == "file"
                    else store.write_session()
                )
                with session as h5_file:
                    if change == "object":
                        del h5_file[payload[1]]
                    else:
                        h5_file.attrs[_persistence_constants.WORKSPACE_ID_ATTR] = (
                            uuid.uuid4().hex
                        )
        return fingerprint

    # Restore path/recovery changes before the Manager fixture closes its store.
    with monkeypatch.context() as context:
        context.setattr(figure_repair, "metadata_fingerprint", change_backing)
        assert (
            case.saver._workspace_figure_repair_cache_state(payload, case.attrs) is None
        )
    assert changed


def test_negative_cache_uses_xarray_fallback_for_unsupported_direct_reads(
    figure_repair_case, repair_counts, monkeypatch
):
    case = figure_repair_case
    monkeypatch.setattr(
        workspace_arrays, "_read_workspace_dataset_group_h5py", lambda *_a, **_kw: None
    )
    assert case.probe() is None
    assert repair_counts["compare"] == 2
    first = repair_counts.copy()
    assert case.probe() is None
    assert repair_counts == first


@pytest.mark.parametrize("difference", ["name", "attrs"])
def test_array_comparison_accepts_omitted_cache_callbacks(
    figure_repair_case, difference
):
    case = figure_repair_case
    current = case.captured.copy(deep=True)
    if difference == "name":
        current = current.rename("other")
    else:
        current.attrs["units"] = "normalized"
    assert not case.saver._workspace_figure_arrays_identical(case.captured, current)


def test_repair_ignores_datasets_without_embedded_figure_sources(figure_repair_case):
    case = figure_repair_case
    result = case.saver._repair_workspace_figure_dataset(case.node, xr.Dataset())
    assert result.dataset is None
    assert result.mismatches is None


@pytest.mark.parametrize("missing_data", ["absent", "empty"])
def test_missing_captured_input_prevents_complete_negative_repair(
    figure_repair_case, missing_data
):
    case = figure_repair_case
    dataset = workspace_arrays._read_workspace_dataset_group_h5py(
        *case.payload, preferred_data_name=_persistence_constants.SAVED_TOOL_DATA_NAME
    )
    assert dataset is not None
    try:
        dataset = dataset.drop_vars("second")
        if missing_data == "empty":
            dataset["second"] = interactive_utils._tool_data_placeholder()
        result = case.saver._repair_workspace_figure_dataset(case.node, dataset)
        assert result.dataset is None
        assert result.mismatches is None
    finally:
        dataset.close()


def test_partial_repair_preserves_other_captured_inputs_and_variables(
    figure_repair_case,
):
    case = figure_repair_case
    dataset = workspace_arrays._read_workspace_dataset_group_h5py(
        *case.payload, preferred_data_name=_persistence_constants.SAVED_TOOL_DATA_NAME
    )
    assert dataset is not None
    dataset["annotation"] = xr.DataArray([10, 20], dims="annotation_index")
    case.sources[0].slicer_area.replace_source_data(case.captured.copy(deep=True))
    try:
        result = case.saver._repair_workspace_figure_dataset(case.node, dataset)
        repaired = result.dataset
        assert repaired is not None
        try:
            references = FigureComposerTool._saved_tool_data_references(repaired)
            assert set(references) == {_persistence_constants.SAVED_TOOL_DATA_NAME}
            assert repaired[_persistence_constants.SAVED_TOOL_DATA_NAME].size == 0
            xr.testing.assert_identical(
                repaired["second"], dataset["second"].reset_coords(drop=True)
            )
            xr.testing.assert_identical(repaired["annotation"], dataset["annotation"])
        finally:
            repaired.close()
    finally:
        dataset.close()


@pytest.mark.parametrize(
    "dtype",
    [
        np.dtype(object),
        np.dtype([("field", np.float64)]),
        np.dtype((np.float64, (2,))),
        np.dtype(np.float64, metadata={"meaning": "calibrated"}),
    ],
)
def test_metadata_cache_rejects_nonprimitive_dtype(dtype):
    with pytest.raises(TypeError, match="primitive fixed-size dtype"):
        figure_repair._check_dtype(dtype)


@pytest.mark.parametrize("base", [list, dict])
def test_metadata_cache_rejects_mutable_container_subclasses(base):
    class CustomMetadata(base):
        pass

    with pytest.raises(TypeError, match="Unsupported Figure comparison metadata"):
        figure_repair.attrs_fingerprint({"custom": CustomMetadata()})


def test_metadata_cache_rejects_numpy_scalar_subclasses():
    class CustomScalar(np.float64):
        pass

    with pytest.raises(TypeError, match="NumPy scalar subclasses"):
        figure_repair.attrs_fingerprint({"custom": CustomScalar(1.0)})


def test_metadata_cache_preserves_bytes_and_numpy_scalar_values():
    attrs = {"raw": b"\xff", "offset": np.float64(1.0)}
    before = figure_repair.attrs_fingerprint(attrs)
    assert figure_repair.attrs_fingerprint(dict(attrs)) == before
    attrs["raw"] = b"\x00"
    assert figure_repair.attrs_fingerprint(attrs) != before


@pytest.mark.parametrize("container", [list, dict])
def test_metadata_cache_rejects_cycles(container):
    cyclic = container()
    if container is list:
        cyclic.append(cyclic)
    else:
        cyclic["self"] = cyclic
    with pytest.raises(ValueError, match="Cyclic Figure comparison metadata"):
        figure_repair.attrs_fingerprint({"cycle": cyclic})


@pytest.mark.parametrize("kind", ["raw", "serialized"])
def test_metadata_cache_bounds_raw_and_serialized_values(monkeypatch, kind):
    monkeypatch.setattr(figure_repair, "_METADATA_MAX_BYTES", 64)
    value = b"x" * 65 if kind == "raw" else "\\" * 32
    expected = "^Figure comparison metadata exceeds" if kind == "raw" else "^Serialized"
    with pytest.raises(ValueError, match=expected):
        figure_repair.attrs_fingerprint({"value": value})


def test_metadata_fingerprint_excludes_intensity_but_tracks_nested_attrs():
    data = xr.DataArray(
        np.arange(4.0), dims="x", attrs={"nested": {"values": [1.0, 2.0]}}
    )
    before = figure_repair.metadata_fingerprint(data)
    data.values[:] = -1.0
    assert figure_repair.metadata_fingerprint(data) == before
    data.attrs["nested"]["values"][1] = 3.0
    assert figure_repair.metadata_fingerprint(data) != before


def test_coordinate_cache_budget_applies_to_all_coordinates(monkeypatch):
    data = xr.DataArray(
        np.zeros((5, 5)),
        dims=("x", "y"),
        coords={"x": np.arange(5.0), "y": np.arange(5.0)},
    )
    monkeypatch.setattr(figure_repair, "_METADATA_MAX_BYTES", 64)
    assert all(coordinate.nbytes < 64 for coordinate in data.coords.values())
    with pytest.raises(ValueError, match="coordinates exceed"):
        figure_repair.metadata_fingerprint(data)


@pytest.mark.parametrize(
    "value",
    [np.array([1.0]), np.array(b"x", dtype="S4097"), np.ndarray((), dtype="S0")],
)
def test_mismatch_witness_requires_one_bounded_scalar(value):
    with pytest.raises(ValueError, match="one small primitive scalar"):
        figure_repair._WorkspaceFigureMismatch("input", (0,), value)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"metadata": bytearray(b"fingerprint")}, TypeError, "immutable fingerprint"),
        ({"index": (0,)}, ValueError, "both an index and a value"),
        ({"value": np.array(1.0)}, ValueError, "both an index and a value"),
        ({"index": (0,), "value": np.float64(1.0)}, TypeError, "plain NumPy array"),
    ],
)
def test_mismatch_witness_rejects_incomplete_or_mutable_records(kwargs, error, message):
    with pytest.raises(error, match=message):
        figure_repair._WorkspaceFigureMismatch("input", **kwargs)


def test_mismatch_witness_without_proof_requires_full_comparison():
    witness = figure_repair._WorkspaceFigureMismatch("input")
    assert not witness.still_differs(xr.DataArray([1.0], dims="x"))


@pytest.mark.parametrize("index", [(0,), (5, 0)])
def test_mismatch_witness_rejects_partial_or_out_of_bounds_indices(index):
    current = xr.DataArray(np.zeros((2, 3)), dims=("x", "y"))
    witness = figure_repair._WorkspaceFigureMismatch("input", index, np.array(1.0))
    assert not witness.still_differs(current)


def test_mismatch_witness_owns_immutable_scalar_storage():
    value = np.array(b"before", dtype="S4096")
    witness = figure_repair._WorkspaceFigureMismatch("input", (), value)
    value[()] = b"after"
    assert witness.value is not None
    assert witness.value.item() == b"before"
    assert witness.value.nbytes == 4096
    with pytest.raises(ValueError, match="WRITEABLE"):
        witness.value.setflags(write=True)


def test_metadata_witness_requires_same_confirmed_unequal_configuration():
    current = xr.DataArray([1.0], dims="x", attrs={"units": "normalized"})
    witness = figure_repair._WorkspaceFigureMismatch(
        "input", metadata=figure_repair.metadata_fingerprint(current)
    )
    assert witness.still_differs(current)
    current.attrs["units"] = "counts"
    assert not witness.still_differs(current)
    current.attrs["units"] = "normalized"
    assert witness.still_differs(current)


@pytest.mark.parametrize(
    ("value", "different"),
    [
        (np.array(np.nan), 1.0),
        (np.array(complex(np.nan, 1.0)), complex(1.0, 1.0)),
        (np.array("a"), "b"),
        (np.array(np.datetime64("NaT", "ns")), np.datetime64("2026-01-01", "ns")),
    ],
)
def test_mismatch_witness_handles_special_scalar_values(value, different):
    current = xr.DataArray(value.reshape((1,)).copy(), dims="x")
    witness = figure_repair._WorkspaceFigureMismatch("input", (0,), value)
    assert not witness.still_differs(current)
    current.values[0] = different
    assert witness.still_differs(current)


def test_mismatch_witness_read_failure_requires_full_comparison():
    @delayed
    def unavailable_values():
        raise OSError("Source is temporarily unavailable")

    current = xr.DataArray(
        da.from_delayed(unavailable_values(), shape=(2,), dtype=np.float64), dims="x"
    )
    witness = figure_repair._WorkspaceFigureMismatch("input", (0,), np.array(1.0))
    assert not witness.still_differs(current)
