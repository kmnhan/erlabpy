import pathlib
import typing
import warnings

import pytest
import xarray as xr
from qtpy import QtWidgets

import erlab
import erlab.interactive.imagetool.manager._widgets as manager_widgets
from erlab.interactive.imagetool._provenance._model import (
    DerivationEntry,
    FileLoadSource,
    ReplayStep,
    ScriptInput,
    ToolProvenanceSpec,
    _ProvenanceDisplayRow,
    _ProvenanceStepRef,
    compose_full_provenance,
    full_data,
    script,
    selection,
)
from erlab.interactive.imagetool._provenance._operations import (
    AverageOperation,
    IselOperation,
    RestoreNonuniformDimsOperation,
    ScriptCodeOperation,
    SortByOperation,
)
from erlab.interactive.imagetool.manager._provenance_edit import (
    _controller as provenance_edit_controller,
)
from erlab.interactive.imagetool.manager._provenance_edit import (
    _files as provenance_edit_files,
)

from ._common import (
    _fake_edit_controller,
    _fake_edit_node,
    _manager_provenance_file_spec,
    _manager_replay_file_spec,
)


def test_manager_provenance_file_load_batch_peer_matching(
    tmp_path: pathlib.Path,
) -> None:
    old_dir = tmp_path / "old"
    other_dir = tmp_path / "other"
    old_dir.mkdir()
    other_dir.mkdir()
    current = _fake_edit_node(
        _manager_replay_file_spec(old_dir / "a.h5"),
        uid="current",
        display_text="Current",
    )
    matching = _fake_edit_node(
        _manager_replay_file_spec(old_dir / "b.h5"),
        uid="matching",
        display_text="Matching",
    )
    pending_matching = _fake_edit_node(
        _manager_replay_file_spec(old_dir / "pending.h5"),
        uid="pending-matching",
        display_text="Pending Matching",
    )
    pending_matching.imagetool = None
    pending_matching.pending_workspace_memory_payload = (
        pathlib.Path("workspace.itws"),
        "1/imagetool",
    )
    other_folder = _fake_edit_node(
        _manager_replay_file_spec(other_dir / "c.h5"),
        uid="other-folder",
    )
    different_kwargs_spec = _manager_replay_file_spec(old_dir / "d.h5")
    assert different_kwargs_spec.file_load_source is not None
    different_replay_call = different_kwargs_spec.file_load_source.replay_call
    assert different_replay_call is not None
    different_kwargs_spec = different_kwargs_spec.model_copy(
        update={
            "file_load_source": different_kwargs_spec.file_load_source.model_copy(
                update={
                    "replay_call": different_replay_call.model_copy(
                        update={"kwargs": {"engine": "scipy"}},
                    ),
                }
            )
        }
    )
    different_kwargs = _fake_edit_node(
        different_kwargs_spec,
        uid="different-kwargs",
    )
    source_bound = _fake_edit_node(
        _manager_replay_file_spec(old_dir / "e.h5"),
        uid="source-bound",
        parent_uid="parent",
        source_spec=full_data(),
    )
    no_spec = _fake_edit_node(None, uid="no-spec")
    non_file = _fake_edit_node(full_data(), uid="non-file")
    no_load_source = _fake_edit_node(
        _manager_replay_file_spec(old_dir / "f.h5").model_copy(
            update={"file_load_source": None},
        ),
        uid="no-load-source",
    )
    no_replay_file = _manager_replay_file_spec(old_dir / "g.h5")
    assert no_replay_file.file_load_source is not None
    no_replay = _fake_edit_node(
        no_replay_file.model_copy(
            update={
                "file_load_source": no_replay_file.file_load_source.model_copy(
                    update={"replay_call": None},
                )
            }
        ),
        uid="no-replay",
    )
    controller = _fake_edit_controller(
        current,
        nodes={
            node.uid: node
            for node in (
                current,
                matching,
                pending_matching,
                other_folder,
                different_kwargs,
                source_bound,
                no_spec,
                non_file,
                no_load_source,
                no_replay,
            )
        },
        metadata_uid="current",
    )

    assert current.displayed_provenance_spec is not None
    peers = controller._file_load_batch_peers(
        typing.cast("typing.Any", current),
        current.displayed_provenance_spec,
    )

    assert [peer.node.uid for peer in peers] == ["matching", "pending-matching"]
    assert peers[0].original_path == old_dir / "b.h5"
    assert peers[1].original_path == old_dir / "pending.h5"

    assert (
        controller._file_load_batch_peers(
            typing.cast("typing.Any", current),
            full_data(),
        )
        == ()
    )


def test_manager_provenance_collects_nested_file_load_targets(
    tmp_path: pathlib.Path,
) -> None:
    first_spec = _manager_replay_file_spec(tmp_path / "a.h5")
    second_file_spec = _manager_replay_file_spec(tmp_path / "b.h5")
    second_parent_spec = script(
        ScriptCodeOperation(
            label="Use nested file",
            code="derived = nested",
        ),
        start_label="Run nested script",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="nested",
                label="Nested",
                provenance_spec=second_file_spec,
            ),
        ),
    )
    root_spec = script(
        ScriptCodeOperation(
            label="Combine inputs",
            code="derived = data_0 + data_1",
        ),
        start_label="Run manager script",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="data_0",
                label="First",
                provenance_spec=first_spec,
            ),
            ScriptInput(
                name="data_1",
                label="Second",
                provenance_spec=second_parent_spec,
            ),
        ),
    )
    node = _fake_edit_node(root_spec, display_text="Derived")
    controller = _fake_edit_controller(node)

    targets = controller._file_load_targets(
        typing.cast("typing.Any", node),
        "display",
        root_spec,
    )

    assert [target.script_input_path for target in targets] == [(0,), (1, 0)]
    assert [target.original_path for target in targets] == [
        tmp_path / "a.h5",
        tmp_path / "b.h5",
    ]
    assert [target.display_text for target in targets] == [
        "Derived: First",
        "Derived: Second: Nested",
    ]


def test_manager_provenance_nested_file_load_batch_keeps_top_level_matches(
    tmp_path: pathlib.Path,
) -> None:
    old_dir = tmp_path / "old"
    old_dir.mkdir()
    first_spec = _manager_replay_file_spec(old_dir / "a.h5")
    second_spec = _manager_replay_file_spec(old_dir / "b.h5")
    matching_spec = _manager_replay_file_spec(old_dir / "c.h5")
    root_spec = script(
        ScriptCodeOperation(
            label="Combine inputs",
            code="derived = data_0 + data_1",
        ),
        start_label="Run manager script",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="data_0",
                label="First",
                provenance_spec=first_spec,
            ),
            ScriptInput(
                name="data_1",
                label="Second",
                provenance_spec=second_spec,
            ),
        ),
    )
    current = _fake_edit_node(root_spec, uid="current", display_text="Current")
    matching = _fake_edit_node(
        matching_spec,
        uid="matching",
        display_text="Matching",
    )
    controller = _fake_edit_controller(
        current,
        nodes={"current": current, "matching": matching},
        metadata_uid="current",
    )
    first_file_row = root_spec.display_rows()[1].children[0]

    peers = controller._file_load_batch_peers(
        typing.cast("typing.Any", current),
        first_spec,
        row=first_file_row,
    )

    assert [(peer.node.uid, peer.script_input_path) for peer in peers] == [
        ("current", (1,)),
        ("matching", ()),
    ]


def test_manager_provenance_missing_repair_peers_include_nonmatching_nested(
    tmp_path: pathlib.Path,
) -> None:
    old_a_dir = tmp_path / "old-a"
    old_b_dir = tmp_path / "old-b"
    old_a_dir.mkdir()
    old_b_dir.mkdir()
    first_spec = _manager_replay_file_spec(old_a_dir / "a.h5")
    second_spec = _manager_replay_file_spec(old_b_dir / "b.h5")
    assert second_spec.file_load_source is not None
    second_replay_call = second_spec.file_load_source.replay_call
    assert second_replay_call is not None
    second_spec = second_spec.model_copy(
        update={
            "file_load_source": second_spec.file_load_source.model_copy(
                update={
                    "kwargs_text": "engine='scipy'",
                    "replay_call": second_replay_call.model_copy(
                        update={"kwargs": {"engine": "scipy"}},
                    ),
                }
            )
        }
    )
    root_spec = script(
        ScriptCodeOperation(
            label="Combine inputs",
            code="derived = data_0 + data_1",
        ),
        start_label="Run manager script",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="data_0",
                label="First",
                provenance_spec=first_spec,
            ),
            ScriptInput(
                name="data_1",
                label="Second",
                provenance_spec=second_spec,
            ),
        ),
    )
    node = _fake_edit_node(root_spec)
    controller = _fake_edit_controller(node)
    focused = controller._file_load_target_for_path(
        typing.cast("typing.Any", node),
        "display",
        root_spec,
        old_a_dir / "a.h5",
    )
    assert focused is not None

    matching_peers = controller._file_load_batch_peers(
        typing.cast("typing.Any", node),
        first_spec,
        row=root_spec.display_rows()[1].children[0],
    )
    repair_peers = controller._missing_file_load_repair_peers(
        typing.cast("typing.Any", node),
        "display",
        root_spec,
        focused,
    )

    assert matching_peers == ()
    assert [peer.script_input_path for peer in repair_peers] == [(1,)]
    assert [peer.preserve_loader for peer in repair_peers] == [True]


def test_manager_provenance_missing_repair_peers_skip_available_or_unreplayable(
    tmp_path: pathlib.Path,
) -> None:
    old_dir = tmp_path / "old"
    old_dir.mkdir()
    available_path = old_dir / "available.h5"
    available_path.touch()
    node = _fake_edit_node(full_data())
    focused = provenance_edit_files._FileLoadBatchPeer(
        node=typing.cast("typing.Any", node),
        scope="display",
        spec=_manager_replay_file_spec(old_dir / "focused.h5"),
        original_path=old_dir / "focused.h5",
        loader_summary="xarray.load_dataarray",
        script_input_path=(0,),
    )
    unavailable_spec = _manager_replay_file_spec(old_dir / "unavailable.h5")
    assert unavailable_spec.file_load_source is not None
    unavailable_spec = unavailable_spec.model_copy(
        update={
            "file_load_source": unavailable_spec.file_load_source.model_copy(
                update={"replay_call": None},
            )
        }
    )
    unavailable = provenance_edit_files._FileLoadBatchPeer(
        node=typing.cast("typing.Any", node),
        scope="display",
        spec=unavailable_spec,
        original_path=old_dir / "unavailable.h5",
        loader_summary="xarray.load_dataarray",
        script_input_path=(1,),
    )
    available = provenance_edit_files._FileLoadBatchPeer(
        node=typing.cast("typing.Any", node),
        scope="display",
        spec=_manager_replay_file_spec(available_path),
        original_path=available_path,
        loader_summary="xarray.load_dataarray",
        script_input_path=(2,),
    )
    controller = _fake_edit_controller(node)
    controller._file_load_targets = lambda *_args, **_kwargs: (
        focused,
        unavailable,
        available,
    )

    assert (
        controller._missing_file_load_repair_peers(
            typing.cast("typing.Any", node),
            "display",
            full_data(),
            focused,
        )
        == ()
    )


def test_manager_provenance_nested_file_load_batch_skips_unmatched_targets(
    tmp_path: pathlib.Path,
) -> None:
    old_dir = tmp_path / "old"
    old_dir.mkdir()
    focused_spec = _manager_replay_file_spec(old_dir / "focused.h5")
    node = _fake_edit_node(full_data())
    focused = provenance_edit_files._FileLoadBatchPeer(
        node=typing.cast("typing.Any", node),
        scope="display",
        spec=focused_spec,
        original_path=old_dir / "focused.h5",
        loader_summary="xarray.load_dataarray",
        script_input_path=(0,),
    )
    unavailable_spec = _manager_replay_file_spec(old_dir / "unavailable.h5")
    assert unavailable_spec.file_load_source is not None
    unavailable_spec = unavailable_spec.model_copy(
        update={
            "file_load_source": unavailable_spec.file_load_source.model_copy(
                update={"replay_call": None},
            )
        }
    )
    unavailable = provenance_edit_files._FileLoadBatchPeer(
        node=typing.cast("typing.Any", node),
        scope="display",
        spec=unavailable_spec,
        original_path=old_dir / "unavailable.h5",
        loader_summary="xarray.load_dataarray",
        script_input_path=(1,),
    )
    different_loader_spec = _manager_replay_file_spec(old_dir / "different.h5")
    assert different_loader_spec.file_load_source is not None
    replay_call = different_loader_spec.file_load_source.replay_call
    assert replay_call is not None
    different_loader_spec = different_loader_spec.model_copy(
        update={
            "file_load_source": different_loader_spec.file_load_source.model_copy(
                update={
                    "kwargs_text": "engine='scipy'",
                    "replay_call": replay_call.model_copy(
                        update={"kwargs": {"engine": "scipy"}},
                    ),
                }
            )
        }
    )
    different_loader = provenance_edit_files._FileLoadBatchPeer(
        node=typing.cast("typing.Any", node),
        scope="display",
        spec=different_loader_spec,
        original_path=old_dir / "different.h5",
        loader_summary="xarray.load_dataarray",
        script_input_path=(2,),
    )
    controller = _fake_edit_controller(node)
    row = _ProvenanceDisplayRow(
        DerivationEntry("Focused", None),
        scope="display",
        script_input_path=(0,),
    )
    controller._root_display_spec_for_row = lambda *_args, **_kwargs: full_data()
    controller._file_load_targets = lambda *_args, **_kwargs: (
        focused,
        unavailable,
        different_loader,
    )

    assert (
        controller._file_load_batch_peers(
            typing.cast("typing.Any", node),
            focused_spec,
            row=row,
        )
        == ()
    )


def test_manager_provenance_file_load_batch_helper_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = _fake_edit_node(None)
    controller = _fake_edit_controller(node)
    assert (
        controller._file_load_targets(typing.cast("typing.Any", node), "display", None)
        == ()
    )

    replacement = full_data()
    assert (
        controller._replace_file_load_target_spec(
            replacement,
            provenance_edit_files._FileLoadBatchPeer(
                node=typing.cast("typing.Any", node),
                scope="display",
                spec=replacement,
                original_path=pathlib.Path("scan.h5"),
                loader_summary="xarray.load_dataarray",
            ),
            replacement,
        )
        is replacement
    )

    source_spec = selection()
    source_node = _fake_edit_node(
        full_data(),
        source_display_spec=source_spec,
    )
    assert (
        controller._root_spec_for_batch_peer(
            provenance_edit_files._FileLoadBatchPeer(
                node=typing.cast("typing.Any", source_node),
                scope="source",
                spec=source_spec,
                original_path=pathlib.Path("scan.h5"),
                loader_summary="xarray.load_dataarray",
            )
        )
        is source_spec
    )
    with pytest.raises(RuntimeError, match="root provenance"):
        controller._root_spec_for_batch_peer(
            provenance_edit_files._FileLoadBatchPeer(
                node=typing.cast("typing.Any", node),
                scope="display",
                spec=full_data(),
                original_path=pathlib.Path("scan.h5"),
                loader_summary="xarray.load_dataarray",
            )
        )

    stale_target = provenance_edit_files._FileLoadBatchPeer(
        node=typing.cast("typing.Any", node),
        scope="display",
        spec=full_data(),
        original_path=pathlib.Path("scan.h5"),
        loader_summary="xarray.load_dataarray",
    )
    monkeypatch.setattr(
        controller,
        "_file_load_target_for_path",
        lambda *_args, **_kwargs: stale_target,
    )
    assert controller._file_load_source_edit_target(
        typing.cast("typing.Any", node),
        pathlib.Path("scan.h5"),
    ) == (
        None,
        None,
        None,
        "This source was not recorded as an editable file-load step.",
    )

    source = FileLoadSource(
        path="scan.h5",
        loader_label="Load Function",
        loader_text="xarray.load_dataarray",
        kwargs_text="",
        replay_call=None,
    )
    assert provenance_edit_files._loader_summary(source) == "xarray.load_dataarray"
    assert (
        provenance_edit_files._loader_summary(
            source.model_copy(update={"kwargs_text": "(none)"})
        )
        == "xarray.load_dataarray"
    )

    def _raise_resolve(
        self: pathlib.Path,
        *,
        strict: bool = False,
    ) -> pathlib.Path:
        del self, strict
        raise RuntimeError("resolve failed")

    monkeypatch.setattr(pathlib.Path, "resolve", _raise_resolve)

    assert (
        provenance_edit_files._normalized_path(pathlib.Path("scan.h5"))
        == pathlib.Path("scan.h5").expanduser().absolute()
    )

    empty_warning = warnings.WarningMessage(
        "",
        UserWarning,
        "scan.py",
        1,
    )
    useful_warning = warnings.WarningMessage(
        "first line\nsecond line",
        RuntimeWarning,
        "scan.py",
        2,
    )
    assert (
        provenance_edit_controller._replay_warning_details(
            [empty_warning, useful_warning]
        )
        == "- RuntimeWarning: first line\n  second line"
    )


def test_manager_provenance_edit_file_load_source_uses_display_spec(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    source_path = tmp_path / "missing.h5"
    node = _fake_edit_node(_manager_replay_file_spec(source_path))
    controller = _fake_edit_controller(node)
    calls: list[
        tuple[
            object,
            typing.Literal["display", "source"],
            ToolProvenanceSpec,
            dict[str, typing.Any],
        ]
    ] = []

    def _edit_file_load_spec(
        edit_node: object,
        scope: typing.Literal["display", "source"],
        spec: ToolProvenanceSpec,
        **kwargs: typing.Any,
    ) -> None:
        calls.append((edit_node, scope, spec, kwargs))

    monkeypatch.setattr(controller, "_edit_file_load_spec", _edit_file_load_spec)

    editable, reason = controller.can_edit_file_load_source(
        typing.cast("typing.Any", node),
        source_path,
    )
    assert editable
    assert reason

    controller.edit_file_load_source(typing.cast("typing.Any", node), source_path)

    assert len(calls) == 1
    edit_node, scope, spec, kwargs = calls[0]
    assert edit_node is node
    assert scope == "display"
    assert spec is node.displayed_provenance_spec
    assert kwargs["where"] == "validating the edited file-load provenance"
    assert kwargs["batch_peers"] == ()


def test_manager_provenance_edit_file_load_source_uses_script_display_spec(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    source_path = tmp_path / "missing.h5"
    script_spec = compose_full_provenance(
        _manager_replay_file_spec(source_path),
        script(
            ScriptCodeOperation(
                label="Calculate result",
                code="result = derived + 1",
            ),
            start_label="Run script",
            active_name="result",
        ),
    )
    assert script_spec is not None
    node = _fake_edit_node(script_spec)
    controller = _fake_edit_controller(node)
    calls: list[ToolProvenanceSpec] = []

    def _edit_file_load_spec(
        _edit_node: object,
        _scope: typing.Literal["display", "source"],
        spec: ToolProvenanceSpec,
        **_kwargs: typing.Any,
    ) -> None:
        calls.append(spec)

    monkeypatch.setattr(controller, "_edit_file_load_spec", _edit_file_load_spec)

    editable, reason = controller.can_edit_file_load_source(
        typing.cast("typing.Any", node),
        source_path,
    )
    assert editable
    assert reason

    controller.edit_file_load_source(typing.cast("typing.Any", node), source_path)

    assert calls == [script_spec]


def test_manager_provenance_edit_file_load_source_falls_back_to_source_spec(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    source_path = tmp_path / "source.h5"
    display_path = tmp_path / "display.h5"
    source_display_spec = _manager_replay_file_spec(source_path)
    node = _fake_edit_node(
        _manager_replay_file_spec(display_path),
        source_spec=full_data(),
        source_display_spec=source_display_spec,
        parent_uid="parent",
    )
    controller = _fake_edit_controller(node)
    calls: list[
        tuple[
            object,
            typing.Literal["display", "source"],
            ToolProvenanceSpec,
        ]
    ] = []
    monkeypatch.setattr(
        controller,
        "_edit_file_load_spec",
        lambda edit_node, scope, spec, **_kwargs: calls.append(
            (edit_node, scope, spec)
        ),
    )

    editable, reason = controller.can_edit_file_load_source(
        typing.cast("typing.Any", node),
        source_path,
    )
    assert editable
    assert reason

    controller.edit_file_load_source(typing.cast("typing.Any", node), source_path)

    assert calls == [(node, "source", source_display_spec)]


def test_manager_provenance_edit_file_load_source_prefers_source_spec(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    source_path = tmp_path / "source.h5"
    source_display_spec = _manager_replay_file_spec(source_path)
    node = _fake_edit_node(
        _manager_replay_file_spec(source_path),
        source_spec=full_data(),
        source_display_spec=source_display_spec,
        parent_uid="parent",
    )
    controller = _fake_edit_controller(node)
    calls: list[
        tuple[
            object,
            typing.Literal["display", "source"],
            ToolProvenanceSpec,
        ]
    ] = []
    monkeypatch.setattr(
        controller,
        "_edit_file_load_spec",
        lambda edit_node, scope, spec, **_kwargs: calls.append(
            (edit_node, scope, spec)
        ),
    )

    editable, reason = controller.can_edit_file_load_source(
        typing.cast("typing.Any", node),
        source_path,
    )
    assert editable
    assert reason

    controller.edit_file_load_source(typing.cast("typing.Any", node), source_path)

    assert calls == [(node, "source", source_display_spec)]


def test_manager_provenance_edit_file_load_source_uses_parent_display_spec(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    source_path = tmp_path / "source.h5"
    parent_spec = _manager_replay_file_spec(source_path)
    parent = _fake_edit_node(parent_spec, uid="parent")
    node = _fake_edit_node(
        _manager_replay_file_spec(source_path),
        uid="child",
        source_spec=full_data(),
        source_display_spec=full_data(),
        parent_uid="parent",
    )
    controller = _fake_edit_controller(
        node,
        nodes={"parent": parent, "child": node},
        parent=parent,
        metadata_uid="child",
    )
    calls: list[
        tuple[
            object,
            typing.Literal["display", "source"],
            ToolProvenanceSpec,
        ]
    ] = []
    monkeypatch.setattr(
        controller,
        "_edit_file_load_spec",
        lambda edit_node, scope, spec, **_kwargs: calls.append(
            (edit_node, scope, spec)
        ),
    )

    editable, reason = controller.can_edit_file_load_source(
        typing.cast("typing.Any", node),
        source_path,
    )
    assert editable
    assert reason

    controller.edit_file_load_source(typing.cast("typing.Any", node), source_path)

    assert calls == [(parent, "display", parent_spec)]


def test_manager_provenance_edit_file_load_source_rejects_source_bound_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    parent = _fake_edit_node(_manager_replay_file_spec(tmp_path / "parent.h5"))
    node = _fake_edit_node(
        _manager_replay_file_spec(tmp_path / "child.h5"),
        source_spec=full_data(),
        source_display_spec=full_data(),
        parent_uid="parent",
    )
    controller = _fake_edit_controller(
        node,
        nodes={"parent": parent, "node": node},
        parent=parent,
    )
    unavailable: list[str] = []
    monkeypatch.setattr(controller, "_show_unavailable", unavailable.append)
    monkeypatch.setattr(
        controller,
        "_edit_file_load_spec",
        lambda *_args, **_kwargs: pytest.fail("unexpected edit"),
    )

    editable, reason = controller.can_edit_file_load_source(
        typing.cast("typing.Any", node),
        tmp_path / "other.h5",
    )
    assert not editable
    assert reason

    controller.edit_file_load_source(
        typing.cast("typing.Any", node),
        tmp_path / "other.h5",
    )

    assert unavailable == [reason]


def test_manager_provenance_edit_file_load_source_rejects_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    source_path = tmp_path / "source.h5"
    node = _fake_edit_node(_manager_replay_file_spec(source_path))
    controller = _fake_edit_controller(node)
    unavailable: list[str] = []
    monkeypatch.setattr(controller, "_show_unavailable", unavailable.append)
    monkeypatch.setattr(
        controller,
        "_edit_file_load_spec",
        lambda *_args, **_kwargs: pytest.fail("unexpected edit"),
    )

    editable, reason = controller.can_edit_file_load_source(
        typing.cast("typing.Any", node),
        tmp_path / "other.h5",
    )
    assert not editable
    assert reason

    controller.edit_file_load_source(
        typing.cast("typing.Any", node),
        tmp_path / "other.h5",
    )

    assert unavailable == [reason]


def test_manager_provenance_edit_file_load_source_rejects_unreplayable_file(
    tmp_path: pathlib.Path,
) -> None:
    source_path = tmp_path / "source.h5"
    spec = _manager_replay_file_spec(source_path)
    assert spec.file_load_source is not None
    node = _fake_edit_node(
        spec.model_copy(
            update={
                "file_load_source": spec.file_load_source.model_copy(
                    update={"replay_call": None}
                )
            }
        )
    )
    controller = _fake_edit_controller(node)

    editable, reason = controller.can_edit_file_load_source(
        typing.cast("typing.Any", node),
        source_path,
    )

    assert not editable
    assert "replay" in reason


def test_manager_provenance_edit_file_load_source_rejects_unavailable_node(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    node = _fake_edit_node(_manager_replay_file_spec(tmp_path / "source.h5"))
    node.imagetool = None
    controller = _fake_edit_controller(node)
    unavailable: list[str] = []
    monkeypatch.setattr(controller, "_show_unavailable", unavailable.append)
    monkeypatch.setattr(
        controller,
        "_edit_file_load_spec",
        lambda *_args, **_kwargs: pytest.fail("unexpected edit"),
    )

    editable, reason = controller.can_edit_file_load_source(
        typing.cast("typing.Any", node),
        tmp_path / "source.h5",
    )
    assert not editable
    assert reason

    controller.edit_file_load_source(
        typing.cast("typing.Any", node),
        tmp_path / "source.h5",
    )

    assert unavailable == [reason]


def test_manager_provenance_edit_controller_availability_branches() -> None:
    edit_ref = _ProvenanceStepRef("operation", operation_index=0)
    replay_ref = _ProvenanceStepRef("operation", operation_index=0)
    row = _ProvenanceDisplayRow(
        DerivationEntry("row", None),
        edit_ref=edit_ref,
        replay_ref=replay_ref,
    )

    assert not _fake_edit_controller().can_edit_row(row)[0]
    assert not _fake_edit_controller().can_revert_row(row)[0]

    node = _fake_edit_node(full_data(IselOperation()))
    node.parent_uid = "parent"
    node.source_spec = selection()
    controller = _fake_edit_controller(node)
    assert not controller.can_edit_row(row)[0]
    assert not controller.can_revert_row(row)[0]

    nested_parent_history = script(
        AverageOperation(dims=("x",)),
        IselOperation(kwargs={"y": 0}),
        start_label="Load parent input",
        seed_code="data_0 = xr.DataArray([[1.0]], dims=['x', 'y'])",
        active_name="data_0",
    )
    parent_context = script(
        ScriptCodeOperation(
            label="Use parent input",
            code="derived = data_0",
        ),
        start_label="Run parent script",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="data_0",
                label="Parent input",
                provenance_spec=nested_parent_history,
            ),
        ),
    )
    nested_parent_row = parent_context.display_rows()[1].children[1]
    node = _fake_edit_node(
        parent_context,
        source_spec=selection(),
        parent_uid="parent",
    )
    controller = _fake_edit_controller(node)
    assert (
        provenance_edit_controller._ProvenanceEditController._source_child_parent_row(
            typing.cast("typing.Any", node),
            nested_parent_row,
        )
    )
    assert not controller.can_edit_row(nested_parent_row)[0]
    assert not controller.can_revert_row(nested_parent_row)[0]
    assert not controller.can_delete_row(nested_parent_row)[0]

    node = _fake_edit_node(None)
    controller = _fake_edit_controller(node)
    assert not controller.can_edit_row(row)[0]
    assert not controller.can_revert_row(row)[0]

    file_row = _ProvenanceDisplayRow(
        DerivationEntry("load", None),
        edit_ref=_ProvenanceStepRef("file_load"),
        replay_ref=_ProvenanceStepRef("file_load"),
    )
    controller = _fake_edit_controller(_fake_edit_node(full_data()))
    assert not controller.can_edit_row(file_row)[0]

    no_replay_file = _manager_provenance_file_spec(pathlib.Path("scan.h5")).model_copy(
        update={
            "file_load_source": FileLoadSource(
                path="scan.h5",
                loader_label="Load Function",
                loader_text="load",
                kwargs_text="",
                replay_call=None,
            )
        }
    )
    controller = _fake_edit_controller(_fake_edit_node(no_replay_file))
    assert not controller.can_edit_row(file_row)[0]

    script_file = compose_full_provenance(
        _manager_provenance_file_spec(pathlib.Path("scan.h5")),
        script(
            ScriptCodeOperation(
                label="Calculate result",
                code="result = derived + 1",
            ),
            start_label="Run script",
            active_name="result",
        ),
    )
    assert script_file is not None
    script_file_row = script_file.display_rows()[0]
    assert script_file_row.edit_ref == _ProvenanceStepRef("file_load")
    controller = _fake_edit_controller(_fake_edit_node(script_file))
    assert controller.can_edit_row(script_file_row) == (True, "")

    missing_operation_row = _ProvenanceDisplayRow(
        DerivationEntry("missing", None),
        edit_ref=_ProvenanceStepRef("operation", operation_index=10),
        replay_ref=replay_ref,
    )
    controller = _fake_edit_controller(_fake_edit_node(full_data()))
    assert not controller.can_edit_row(missing_operation_row)[0]

    script_operation_spec = ToolProvenanceSpec(
        kind="full_data",
        operations=(ScriptCodeOperation(label="script", code="derived = data"),),
    )
    script_row = _ProvenanceDisplayRow(
        DerivationEntry("script", None),
        edit_ref=edit_ref,
        replay_ref=replay_ref,
    )
    controller = _fake_edit_controller(
        _fake_edit_node(script_operation_spec, parent_uid="parent")
    )
    assert not controller.can_edit_row(script_row)[0]

    script_with_structured_step = script(
        ScriptCodeOperation(
            label="Create derived data",
            code="derived = data_0 + 1.0",
        ),
        AverageOperation(dims=("x",)),
        start_label="Run script",
        active_name="derived",
        script_inputs=(ScriptInput(name="data_0", label="Input"),),
    )
    controller = _fake_edit_controller(_fake_edit_node(script_with_structured_step))
    script_code_row, structured_row = script_with_structured_step.display_rows()[2:]
    assert controller.can_edit_row(script_code_row) == (True, "")
    assert controller.can_edit_row(structured_row) == (True, "")

    controller = _fake_edit_controller(
        _fake_edit_node(script_with_structured_step),
        script_input_can_reload=lambda *_args, **_kwargs: False,
    )
    assert controller.can_edit_row(structured_row) == (True, "")

    script_parent = script(
        ScriptCodeOperation(
            label="Concatenate selected ImageTools",
            code="derived = data_0 + data_1",
        ),
        start_label="Run ImageTool manager action",
        active_name="derived",
        script_inputs=(
            ScriptInput(name="data_0", label="ImageTool 0: scan"),
            ScriptInput(name="data_1", label="ImageTool 1: scan"),
        ),
    )
    script_with_composed_structured_step = compose_full_provenance(
        script_parent,
        full_data(SortByOperation(variables=("x",))),
    )
    assert script_with_composed_structured_step is not None
    script_code_row, sort_row = script_with_composed_structured_step.display_rows()[3:]
    assert controller.can_edit_row(script_code_row) == (True, "")
    controller = _fake_edit_controller(
        _fake_edit_node(script_with_composed_structured_step)
    )
    assert controller.can_edit_row(sort_row) == (True, "")

    active_filter = AverageOperation(dims=("x",))
    active_filter_spec = script(
        ScriptCodeOperation(
            label="Create derived data",
            code="derived = data_0 + 1.0",
        ),
        active_filter,
        start_label="Run script",
        active_name="derived",
        script_inputs=(ScriptInput(name="data_0", label="Input"),),
    )
    active_row = active_filter_spec.display_rows()[3]
    controller = _fake_edit_controller(
        _fake_edit_node(active_filter_spec, active_filter=active_filter),
        script_input_can_reload=lambda *_args, **_kwargs: False,
    )
    assert controller.can_edit_row(active_row) == (True, "")

    source_row = _ProvenanceDisplayRow(
        DerivationEntry("source", None),
        edit_ref=edit_ref,
        replay_ref=replay_ref,
        scope="source",
    )

    unsupported_spec = full_data(RestoreNonuniformDimsOperation())
    controller = _fake_edit_controller(
        _fake_edit_node(
            full_data(),
            source_display_spec=unsupported_spec,
            parent_uid="parent",
        )
    )
    assert not controller.can_edit_row(source_row)[0]

    live_row = _ProvenanceDisplayRow(
        DerivationEntry("live", None),
        edit_ref=edit_ref,
        replay_ref=replay_ref,
    )
    controller = _fake_edit_controller(_fake_edit_node(selection(IselOperation())))
    assert not controller.can_edit_row(live_row)[0]
    assert not controller.can_revert_row(live_row)[0]

    script_input_row = _ProvenanceDisplayRow(
        DerivationEntry("input", None),
        replay_ref=_ProvenanceStepRef("script_input", script_input_index=0),
    )
    assert not controller.can_revert_row(script_input_row)[0]

    earlier_source_row = _ProvenanceDisplayRow(
        DerivationEntry("source", None),
        edit_ref=edit_ref,
        replay_ref=edit_ref,
        scope="source",
    )
    controller = _fake_edit_controller(
        _fake_edit_node(
            selection(IselOperation()),
            source_display_spec=selection(
                IselOperation(kwargs={"x": 0}),
                IselOperation(kwargs={"y": 0}),
            ),
            parent_uid="parent",
        )
    )
    assert controller.can_revert_row(earlier_source_row)[0]


def test_manager_provenance_edit_nested_script_input_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_spec = script(
        AverageOperation(dims=("x",)),
        start_label="Load parent",
        seed_code="data_0 = xr.DataArray([1.0, 2.0], dims=['x'])",
        active_name="data_0",
    )
    root_spec = script(
        ScriptCodeOperation(
            label="Use parent",
            code="derived = data_0",
        ),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="data_0",
                label="Parent",
                node_uid="parent",
                node_snapshot_token=str(object()),
                provenance_spec=parent_spec,
            ),
        ),
    )
    node = _fake_edit_node(root_spec)
    controller = _fake_edit_controller(node)
    nested_row = root_spec.display_rows()[1].children[1]

    assert controller._display_spec_for_row(node, nested_row) == parent_spec
    assert controller.can_edit_row(nested_row) == (True, "")

    replacement = AverageOperation(dims=("y",))
    replaced: list[ToolProvenanceSpec] = []
    monkeypatch.setattr(
        controller,
        "_replay_candidate",
        lambda *_args, **_kwargs: pytest.fail("editing must not replay on open"),
    )
    monkeypatch.setattr(
        controller,
        "_edited_native_operations",
        lambda *_args, **_kwargs: [replacement],
    )
    monkeypatch.setattr(
        controller,
        "_validate_and_replace",
        lambda _node, _scope, candidate, **_kwargs: replaced.append(candidate),
    )

    controller._edit_operation_row(node, nested_row)

    assert len(replaced) == 1
    edited_input = replaced[0].script_inputs[0]
    assert edited_input.node_uid is None
    assert edited_input.node_snapshot_token is None
    edited_parent = edited_input.parsed_provenance_spec()
    assert edited_parent is not None
    assert edited_parent.operations == (replacement,)


def test_manager_provenance_nested_script_input_revert_delete_and_file_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_spec = _manager_provenance_file_spec(pathlib.Path("scan.h5"))
    parent_spec = script(
        AverageOperation(dims=("x",)),
        IselOperation(kwargs={"y": 0}),
        start_label="Use file parent",
        active_name="data_0",
        script_inputs=(
            ScriptInput(
                name="file_parent",
                label="File parent",
                provenance_spec=file_spec,
            ),
        ),
    )
    root_spec = script(
        ScriptCodeOperation(
            label="Use parent",
            code="derived = data_0",
        ),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="data_0",
                label="Parent",
                node_uid="parent",
                node_snapshot_token=str(object()),
                provenance_spec=parent_spec,
            ),
        ),
    )
    node = _fake_edit_node(root_spec)
    controller = _fake_edit_controller(node)
    nested_rows = root_spec.display_rows()[1].children
    nested_revert_row = nested_rows[2]
    nested_delete_row = nested_rows[3]

    assert controller._display_spec_for_row(node, nested_revert_row) == parent_spec
    assert controller._script_input_path_spec(root_spec, (99,)) is None
    assert controller._script_input_path_spec(root_spec, (0, 99)) is None
    no_history = root_spec.model_copy(
        update={
            "script_inputs": (
                root_spec.script_inputs[0].model_copy(update={"provenance_spec": None}),
            )
        }
    )
    assert controller._script_input_path_spec(no_history, (0,)) is None
    assert (
        controller._root_candidate_for_row(
            node,
            root_spec.display_rows()[2],
            parent_spec,
        )
        is parent_spec
    )

    rootless_controller = _fake_edit_controller(_fake_edit_node(None))
    with pytest.raises(RuntimeError, match="No root provenance"):
        rootless_controller._root_candidate_for_row(
            typing.cast("typing.Any", rootless_controller._metadata_node()),
            nested_revert_row,
            parent_spec,
        )
    with pytest.raises(IndexError, match="Script input provenance path"):
        controller._replace_script_input_path_spec(root_spec, (99,), parent_spec)
    with pytest.raises(RuntimeError, match="does not have replayable provenance"):
        controller._replace_script_input_path_spec(no_history, (0,), parent_spec)

    replaced: list[ToolProvenanceSpec] = []
    monkeypatch.setattr(controller, "can_revert_row", lambda _row: (True, ""))
    monkeypatch.setattr(controller, "_confirm_revert", lambda: True)
    monkeypatch.setattr(
        controller,
        "_validate_and_replace",
        lambda _node, _scope, candidate, **_kwargs: replaced.append(candidate),
    )

    controller.revert_row(nested_revert_row)
    assert len(replaced) == 1
    reverted_parent = replaced[-1].script_inputs[0].parsed_provenance_spec()
    assert reverted_parent is not None
    assert [operation.op for operation in reverted_parent.operations] == ["average"]
    assert replaced[-1].script_inputs[0].node_uid is None

    replaced.clear()
    monkeypatch.setattr(controller, "can_delete_row", lambda _row: (True, ""))
    controller.delete_row(nested_delete_row)
    assert len(replaced) == 1
    deleted_parent = replaced[-1].script_inputs[0].parsed_provenance_spec()
    assert deleted_parent is not None
    assert [operation.op for operation in deleted_parent.operations] == ["average"]

    file_load_row = nested_rows[1].children[0]
    file_load_calls: list[
        tuple[
            object,
            typing.Literal["display", "source"],
            ToolProvenanceSpec,
            _ProvenanceDisplayRow | None,
            tuple[object, ...],
        ]
    ] = []
    monkeypatch.setattr(
        controller,
        "_edit_file_load_spec",
        lambda edit_node, scope, spec, *, row=None, batch_peers=(), **_kwargs: (
            file_load_calls.append((edit_node, scope, spec, row, tuple(batch_peers)))
        ),
    )

    controller._edit_file_load_row(node, file_load_row)
    assert file_load_calls == [(node, "display", file_spec, file_load_row, ())]


def test_manager_provenance_nested_file_load_batch_replaces_one_root_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"
    old_dir.mkdir()
    new_dir.mkdir()
    old_a = old_dir / "a.h5"
    old_b = old_dir / "b.h5"
    new_a = new_dir / "a.h5"
    new_b = new_dir / "b.h5"
    first_spec = _manager_replay_file_spec(old_a)
    second_spec = _manager_replay_file_spec(old_b)
    root_spec = script(
        ScriptCodeOperation(
            label="Combine inputs",
            code="derived = data_0 + data_1",
        ),
        start_label="Run manager script",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="data_0",
                label="First",
                node_uid="first",
                node_snapshot_token=str(object()),
                provenance_spec=first_spec,
            ),
            ScriptInput(
                name="data_1",
                label="Second",
                node_uid="second",
                node_snapshot_token=str(object()),
                provenance_spec=second_spec,
            ),
        ),
    )
    node = _fake_edit_node(root_spec)
    controller = _fake_edit_controller(node)
    first_file_row = root_spec.display_rows()[1].children[0]
    peers = controller._file_load_batch_peers(
        typing.cast("typing.Any", node),
        first_spec,
        row=first_file_row,
    )
    assert [peer.script_input_path for peer in peers] == [(1,)]

    replacement_first = _manager_replay_file_spec(new_a)
    replacement_second = _manager_replay_file_spec(new_b)

    class _Dialog:
        def __init__(
            self,
            _load_source: FileLoadSource,
            _parent: QtWidgets.QWidget,
            *,
            batch_peers: tuple[provenance_edit_files._FileLoadBatchPeer, ...],
            **_kwargs: typing.Any,
        ) -> None:
            assert batch_peers == peers

        def exec(self) -> int:
            return int(QtWidgets.QDialog.DialogCode.Accepted)

        def provenance_spec(
            self,
            *,
            active_name: str,
            replay_steps: tuple[ReplayStep, ...],
        ) -> ToolProvenanceSpec:
            del active_name, replay_steps
            return replacement_first

        def selected_batch_peers(
            self,
        ) -> tuple[provenance_edit_files._FileLoadBatchPeer, ...]:
            return peers

        def peer_provenance_spec(
            self,
            peer: provenance_edit_files._FileLoadBatchPeer,
        ) -> ToolProvenanceSpec:
            assert peer is peers[0]
            return replacement_second

    validated: list[ToolProvenanceSpec] = []
    applied: list[str] = []
    monkeypatch.setattr(provenance_edit_controller, "_FileLoadEditDialog", _Dialog)
    monkeypatch.setattr(
        controller,
        "_validated_edit",
        lambda edit_node, scope, candidate, **_kwargs: (
            validated.append(candidate)
            or provenance_edit_controller._ValidatedProvenanceEdit(
                node=typing.cast("typing.Any", edit_node),
                scope=scope,
                data=xr.DataArray([1.0], dims=("x",)),
                spec=candidate,
                filter_operation=None,
            )
        ),
    )
    monkeypatch.setattr(
        controller,
        "_apply_validated_edit",
        lambda edit: applied.append(edit.node.uid),
    )

    controller._edit_file_load_row(typing.cast("typing.Any", node), first_file_row)

    assert applied == ["node"]
    assert len(validated) == 1
    edited_inputs = validated[0].script_inputs
    assert edited_inputs[0].node_uid is None
    assert edited_inputs[0].node_snapshot_token is None
    assert edited_inputs[1].node_uid is None
    assert edited_inputs[1].node_snapshot_token is None
    edited_first = edited_inputs[0].parsed_provenance_spec()
    edited_second = edited_inputs[1].parsed_provenance_spec()
    assert edited_first is not None
    assert edited_second is not None
    assert edited_first.file_load_source is not None
    assert edited_second.file_load_source is not None
    assert pathlib.Path(edited_first.file_load_source.path) == new_a
    assert pathlib.Path(edited_second.file_load_source.path) == new_b


def test_manager_provenance_file_load_batch_replaces_nested_peer_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"
    old_dir.mkdir()
    new_dir.mkdir()
    current_spec = _manager_replay_file_spec(old_dir / "current.h5")
    peer_file_spec = _manager_replay_file_spec(old_dir / "peer.h5")
    peer_root = script(
        ScriptCodeOperation(
            label="Use file input",
            code="derived = data_0",
        ),
        start_label="Run manager script",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="data_0",
                label="Peer file",
                provenance_spec=peer_file_spec,
            ),
        ),
    )
    current_node = _fake_edit_node(current_spec, uid="current")
    peer_node = _fake_edit_node(peer_root, uid="peer")
    peer = provenance_edit_files._FileLoadBatchPeer(
        node=typing.cast("typing.Any", peer_node),
        scope="display",
        spec=peer_file_spec,
        original_path=old_dir / "peer.h5",
        loader_summary="xarray.load_dataarray",
        script_input_path=(0,),
    )
    controller = _fake_edit_controller(
        current_node,
        nodes={"current": current_node, "peer": peer_node},
        metadata_uid="current",
    )
    replacement_current = _manager_replay_file_spec(new_dir / "current.h5")
    replacement_peer = _manager_replay_file_spec(new_dir / "peer.h5")

    class _Dialog:
        def __init__(
            self,
            _load_source: FileLoadSource,
            _parent: QtWidgets.QWidget,
            *,
            batch_peers: tuple[provenance_edit_files._FileLoadBatchPeer, ...],
            **_kwargs: typing.Any,
        ) -> None:
            assert batch_peers == (peer,)

        def exec(self) -> int:
            return int(QtWidgets.QDialog.DialogCode.Accepted)

        def provenance_spec(
            self,
            *,
            active_name: str,
            replay_steps: tuple[ReplayStep, ...],
        ) -> ToolProvenanceSpec:
            del active_name, replay_steps
            return replacement_current

        def selected_batch_peers(
            self,
        ) -> tuple[provenance_edit_files._FileLoadBatchPeer, ...]:
            return (peer,)

        def peer_provenance_spec(
            self,
            selected_peer: provenance_edit_files._FileLoadBatchPeer,
        ) -> ToolProvenanceSpec:
            assert selected_peer is peer
            return replacement_peer

    validated: list[tuple[str, ToolProvenanceSpec]] = []
    monkeypatch.setattr(provenance_edit_controller, "_FileLoadEditDialog", _Dialog)
    monkeypatch.setattr(
        controller,
        "_validated_edit",
        lambda edit_node, scope, candidate, **_kwargs: (
            validated.append((edit_node.uid, candidate))
            or provenance_edit_controller._ValidatedProvenanceEdit(
                node=typing.cast("typing.Any", edit_node),
                scope=scope,
                data=xr.DataArray([1.0], dims=("x",)),
                spec=candidate,
                filter_operation=None,
            )
        ),
    )
    monkeypatch.setattr(controller, "_apply_validated_edit", lambda _edit: None)

    controller._edit_file_load_spec(
        typing.cast("typing.Any", current_node),
        "display",
        current_spec,
        where="validating edited file load",
        row=None,
        batch_peers=(peer,),
    )

    assert [uid for uid, _candidate in validated] == ["current", "peer"]
    assert validated[0][1] == replacement_current
    edited_peer = validated[1][1].script_inputs[0].parsed_provenance_spec()
    assert edited_peer is not None
    assert edited_peer.file_load_source is not None
    assert pathlib.Path(edited_peer.file_load_source.path) == new_dir / "peer.h5"


@pytest.mark.parametrize(
    ("action", "configure_result", "expected_attempts", "expected_applied"),
    [
        ("retry", False, 2, 1),
        ("configure", True, 2, 1),
        ("configure", False, 1, 0),
        ("cancel", False, 1, 0),
    ],
)
def test_manager_provenance_spreadsheet_failure_is_recoverable_on_demand(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    action: typing.Literal["retry", "configure", "cancel"],
    configure_result: bool,
    expected_attempts: int,
    expected_applied: int,
) -> None:
    spec = _manager_replay_file_spec(tmp_path / "scan.h5")
    node = _fake_edit_node(spec)
    controller = _fake_edit_controller(node)
    source = erlab.io.metadata.ExcelMetadataSource(
        tmp_path / "metadata.xlsx",
        sheet_name="Measurements",
        file_name_column="File",
        coordinate_mapping={"Temperature": "sample_temp"},
        row_range=(20, 27),
    )
    configure_calls: list[bool] = []

    class _LoaderOptions:
        def spreadsheet_metadata_source(self):
            return source

        def configure_spreadsheet_metadata(self, *, load_on_open: bool) -> bool:
            configure_calls.append(load_on_open)
            return configure_result

    class _Dialog:
        loader_options = _LoaderOptions()

        def __init__(self, *_args: typing.Any, **_kwargs: typing.Any) -> None:
            pass

        def exec(self) -> int:
            return int(QtWidgets.QDialog.DialogCode.Accepted)

        def provenance_spec(self, **_kwargs: typing.Any) -> ToolProvenanceSpec:
            return spec

        def selected_batch_peers(
            self,
        ) -> tuple[provenance_edit_files._FileLoadBatchPeer, ...]:
            return ()

    validation_attempts = 0
    applied: list[provenance_edit_controller._ValidatedProvenanceEdit] = []

    def validate(
        edit_node: object,
        scope: typing.Literal["display", "source"],
        candidate: ToolProvenanceSpec,
        **_kwargs: typing.Any,
    ) -> provenance_edit_controller._ValidatedProvenanceEdit:
        nonlocal validation_attempts
        validation_attempts += 1
        if validation_attempts == 1:
            source_error = erlab.io.metadata.SpreadsheetMetadataError(
                "temporary source failure"
            )
            raise provenance_edit_controller._ProvenanceReplayFailure(
                "validating spreadsheet replay",
                source_error,
            ) from source_error
        return provenance_edit_controller._ValidatedProvenanceEdit(
            node=typing.cast("typing.Any", edit_node),
            scope=scope,
            data=xr.DataArray([1.0], dims=("x",)),
            spec=candidate,
            filter_operation=None,
        )

    monkeypatch.setattr(provenance_edit_controller, "_FileLoadEditDialog", _Dialog)
    monkeypatch.setattr(controller, "_validated_edit", validate)
    monkeypatch.setattr(
        controller,
        "_prompt_spreadsheet_metadata_recovery",
        lambda selected_source, _error: (
            action if selected_source is source else pytest.fail("wrong source")
        ),
    )
    monkeypatch.setattr(controller, "_apply_validated_edit", applied.append)

    controller._edit_file_load_spec(
        typing.cast("typing.Any", node),
        "display",
        spec,
        where="validating spreadsheet replay",
    )

    assert validation_attempts == expected_attempts
    assert len(applied) == expected_applied
    assert configure_calls == ([False] if action == "configure" else [])


@pytest.mark.parametrize("action", ["retry", "configure", "cancel"])
def test_manager_provenance_spreadsheet_recovery_prompt_actions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    action: typing.Literal["retry", "configure", "cancel"],
) -> None:
    real_message_box = QtWidgets.QMessageBox
    boxes: list[object] = []

    class _Button:
        def __init__(self) -> None:
            self.object_name = ""

        def setObjectName(self, name: str) -> None:
            self.object_name = name

    class _MessageBox:
        Icon = real_message_box.Icon
        StandardButton = real_message_box.StandardButton
        ButtonRole = real_message_box.ButtonRole

        def __init__(self, _parent: object) -> None:
            self.object_name = ""
            self.buttons: dict[object, _Button] = {}
            self.clicked: _Button | None = None
            boxes.append(self)

        def setObjectName(self, name: str) -> None:
            self.object_name = name

        def setIcon(self, _icon: object) -> None:
            pass

        def setWindowTitle(self, _title: str) -> None:
            pass

        def setText(self, _text: str) -> None:
            pass

        def setInformativeText(self, _text: str) -> None:
            pass

        def setDetailedText(self, _text: str) -> None:
            pass

        def addButton(self, button: object, role: object | None = None) -> _Button:
            created = _Button()
            self.buttons["configure" if role is not None else button] = created
            return created

        def setDefaultButton(self, _button: object) -> None:
            pass

        def exec(self) -> None:
            key: object = {
                "retry": real_message_box.StandardButton.Retry,
                "configure": "configure",
                "cancel": real_message_box.StandardButton.Cancel,
            }[action]
            self.clicked = self.buttons[key]

        def clickedButton(self) -> _Button | None:
            return self.clicked

    monkeypatch.setattr(
        provenance_edit_controller.QtWidgets,
        "QMessageBox",
        _MessageBox,
    )
    controller = _fake_edit_controller(_fake_edit_node(full_data()))
    source = erlab.io.metadata.ExcelMetadataSource(tmp_path / "metadata.xlsx")

    result = controller._prompt_spreadsheet_metadata_recovery(
        source,
        erlab.io.metadata.SpreadsheetMetadataError("source unavailable"),
    )

    assert result == action
    box = typing.cast("typing.Any", boxes[0])
    assert box.object_name == "managerSpreadsheetMetadataRecoveryDialog"
    assert (
        box.buttons["configure"].object_name
        == "managerSpreadsheetMetadataRecoveryConfigureButton"
    )


def test_manager_provenance_revert_rejects_current_prefixes(
    tmp_path: pathlib.Path,
) -> None:
    invalid_operation_row = _ProvenanceDisplayRow(
        DerivationEntry("invalid", None),
        replay_ref=_ProvenanceStepRef("operation"),
    )
    controller = _fake_edit_controller(_fake_edit_node(full_data()))
    revertible, reason = controller.can_revert_row(invalid_operation_row)
    assert not revertible
    assert reason

    file_load_row = _ProvenanceDisplayRow(
        DerivationEntry("load", None),
        replay_ref=_ProvenanceStepRef("file_load"),
    )
    controller = _fake_edit_controller(
        _fake_edit_node(_manager_replay_file_spec(tmp_path / "scan.h5"))
    )
    revertible, reason = controller.can_revert_row(file_load_row)
    assert not revertible
    assert reason

    file_stage_spec = _manager_replay_file_spec(
        tmp_path / "scan.h5",
        IselOperation(kwargs={"x": 0}),
        IselOperation(kwargs={"y": 0}),
    )
    earlier_stage_row = _ProvenanceDisplayRow(
        DerivationEntry("isel", None),
        replay_ref=_ProvenanceStepRef(
            "operation",
            operation_index=0,
        ),
    )
    latest_stage_row = _ProvenanceDisplayRow(
        DerivationEntry("isel", None),
        replay_ref=_ProvenanceStepRef(
            "operation",
            operation_index=1,
        ),
    )
    controller = _fake_edit_controller(_fake_edit_node(file_stage_spec))
    assert controller.can_revert_row(earlier_stage_row)[0]
    revertible, reason = controller.can_revert_row(latest_stage_row)
    assert not revertible
    assert reason

    script_spec = script(
        ScriptCodeOperation(
            label="Run script",
            code="derived = xr.DataArray([1.0], dims=('x',))",
        ),
        IselOperation(kwargs={"x": 0}),
        start_label="Run script",
        active_name="derived",
    )
    controller = _fake_edit_controller(_fake_edit_node(script_spec))
    revertible, reason = controller.can_revert_row(invalid_operation_row)
    assert not revertible
    assert reason

    latest_script_row = _ProvenanceDisplayRow(
        DerivationEntry("isel", None),
        replay_ref=_ProvenanceStepRef(
            "operation",
            operation_index=1,
        ),
    )
    revertible, reason = controller.can_revert_row(latest_script_row)
    assert not revertible
    assert reason

    source_latest_row = _ProvenanceDisplayRow(
        DerivationEntry("source", None),
        replay_ref=_ProvenanceStepRef("operation", operation_index=1),
        scope="source",
    )
    source_spec = selection(
        IselOperation(kwargs={"x": 0}),
        IselOperation(kwargs={"y": 0}),
    )
    controller = _fake_edit_controller(
        _fake_edit_node(
            full_data(),
            source_display_spec=source_spec,
            parent_uid="parent",
        )
    )
    revertible, reason = controller.can_revert_row(source_latest_row)
    assert not revertible
    assert reason


def test_manager_provenance_edit_controller_error_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = _fake_edit_node(full_data())
    controller = _fake_edit_controller(node)
    unavailable: list[str] = []
    failed: list[tuple[str, Exception]] = []
    monkeypatch.setattr(controller, "_show_unavailable", unavailable.append)
    monkeypatch.setattr(
        controller, "_show_failed", lambda title, exc: failed.append((title, exc))
    )

    controller.edit_row(None)
    controller.revert_row(None)
    assert len(unavailable) == 2

    row = _ProvenanceDisplayRow(
        DerivationEntry("row", None),
        replay_ref=_ProvenanceStepRef("operation", operation_index=0),
    )
    monkeypatch.setattr(controller, "can_revert_row", lambda _row: (True, ""))
    monkeypatch.setattr(controller, "_confirm_revert", lambda: True)
    monkeypatch.setattr(controller, "_display_spec_for_row", lambda _node, _row: None)
    controller.revert_row(row)
    assert len(failed) == 1

    monkeypatch.setattr(
        controller,
        "_display_spec_for_row",
        lambda _node, _row: full_data(),
    )
    monkeypatch.setattr(
        controller,
        "_validate_and_replace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    controller.revert_row(row)
    assert len(failed) == 2
    assert "boom" in str(failed[-1][1])


@pytest.mark.parametrize("apply_valid_tools", [True, False])
def test_manager_provenance_file_load_batch_partial_failure_decision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    apply_valid_tools: bool,
) -> None:
    current_spec = _manager_replay_file_spec(tmp_path / "a.h5")
    valid_spec = _manager_replay_file_spec(tmp_path / "b.h5")
    failed_spec = _manager_replay_file_spec(tmp_path / "c.h5")
    current = _fake_edit_node(current_spec, uid="current")
    valid = _fake_edit_node(valid_spec, uid="valid")
    failed = _fake_edit_node(failed_spec, uid="failed")
    controller = _fake_edit_controller(
        current,
        nodes={node.uid: node for node in (current, valid, failed)},
        metadata_uid="current",
    )
    assert valid_spec.file_load_source is not None
    assert failed_spec.file_load_source is not None
    peers = (
        provenance_edit_files._FileLoadBatchPeer(
            node=typing.cast("typing.Any", valid),
            scope="display",
            spec=valid_spec,
            original_path=tmp_path / "b.h5",
            loader_summary="xarray.load_dataarray",
        ),
        provenance_edit_files._FileLoadBatchPeer(
            node=typing.cast("typing.Any", failed),
            scope="display",
            spec=failed_spec,
            original_path=tmp_path / "c.h5",
            loader_summary="xarray.load_dataarray",
        ),
    )

    class _Dialog:
        def __init__(
            self,
            _load_source: FileLoadSource,
            _parent: QtWidgets.QWidget,
            *,
            batch_peers: tuple[provenance_edit_files._FileLoadBatchPeer, ...],
            **_kwargs: typing.Any,
        ) -> None:
            assert batch_peers == peers

        def exec(self) -> int:
            return int(QtWidgets.QDialog.DialogCode.Accepted)

        def provenance_spec(
            self,
            *,
            active_name: str,
            replay_steps: tuple[ReplayStep, ...],
        ) -> ToolProvenanceSpec:
            del active_name, replay_steps
            return current_spec

        def selected_batch_peers(
            self,
        ) -> tuple[provenance_edit_files._FileLoadBatchPeer, ...]:
            return peers

        def peer_provenance_spec(
            self,
            peer: provenance_edit_files._FileLoadBatchPeer,
        ) -> ToolProvenanceSpec:
            return peer.spec

    monkeypatch.setattr(provenance_edit_controller, "_FileLoadEditDialog", _Dialog)
    monkeypatch.setattr(
        controller,
        "_file_load_batch_peers",
        lambda _node, _spec: peers,
    )
    monkeypatch.setattr(
        controller,
        "_confirm_apply_valid_batch",
        lambda **_kwargs: apply_valid_tools,
    )

    def _validated_edit(
        node: typing.Any,
        scope: typing.Literal["display", "source"],
        candidate: ToolProvenanceSpec,
        *,
        where: str,
    ) -> provenance_edit_controller._ValidatedProvenanceEdit:
        del where
        if node.uid == "failed":
            raise RuntimeError("peer failed")
        return provenance_edit_controller._ValidatedProvenanceEdit(
            node=typing.cast("typing.Any", node),
            scope=scope,
            data=xr.DataArray([1.0], dims=("x",)),
            spec=candidate,
            filter_operation=None,
        )

    applied: list[str] = []
    monkeypatch.setattr(controller, "_validated_edit", _validated_edit)
    monkeypatch.setattr(
        controller,
        "_apply_validated_edit",
        lambda edit: applied.append(edit.node.uid),
    )
    row = _ProvenanceDisplayRow(
        DerivationEntry("load", None),
        edit_ref=_ProvenanceStepRef("file_load"),
    )

    controller._edit_file_load_row(typing.cast("typing.Any", current), row)

    if apply_valid_tools:
        assert applied == ["current", "valid"]
    else:
        assert applied == []


def test_manager_provenance_file_load_batch_current_failure_aborts_all(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    current_spec = _manager_replay_file_spec(tmp_path / "a.h5")
    peer_spec = _manager_replay_file_spec(tmp_path / "b.h5")
    current = _fake_edit_node(current_spec, uid="current")
    peer = _fake_edit_node(peer_spec, uid="peer")
    controller = _fake_edit_controller(
        current,
        nodes={node.uid: node for node in (current, peer)},
        metadata_uid="current",
    )
    batch_peer = provenance_edit_files._FileLoadBatchPeer(
        node=typing.cast("typing.Any", peer),
        scope="display",
        spec=peer_spec,
        original_path=tmp_path / "b.h5",
        loader_summary="xarray.load_dataarray",
    )

    class _Dialog:
        def __init__(self, *_args: typing.Any, **_kwargs: typing.Any) -> None:
            pass

        def exec(self) -> int:
            return int(QtWidgets.QDialog.DialogCode.Accepted)

        def provenance_spec(
            self,
            *,
            active_name: str,
            replay_steps: tuple[ReplayStep, ...],
        ) -> ToolProvenanceSpec:
            del active_name, replay_steps
            return current_spec

        def selected_batch_peers(
            self,
        ) -> tuple[provenance_edit_files._FileLoadBatchPeer, ...]:
            return (batch_peer,)

        def peer_provenance_spec(
            self,
            peer: provenance_edit_files._FileLoadBatchPeer,
        ) -> ToolProvenanceSpec:
            return peer.spec

    monkeypatch.setattr(provenance_edit_controller, "_FileLoadEditDialog", _Dialog)
    monkeypatch.setattr(
        controller,
        "_file_load_batch_peers",
        lambda _node, _spec: (batch_peer,),
    )
    monkeypatch.setattr(
        controller,
        "_validated_edit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("current failed")),
    )
    monkeypatch.setattr(
        controller,
        "_confirm_apply_valid_batch",
        lambda **_kwargs: pytest.fail("peer failures should not be considered"),
    )
    applied: list[str] = []
    monkeypatch.setattr(
        controller,
        "_apply_validated_edit",
        lambda edit: applied.append(edit.node.uid),
    )
    row = _ProvenanceDisplayRow(
        DerivationEntry("load", None),
        edit_ref=_ProvenanceStepRef("file_load"),
    )

    with pytest.raises(RuntimeError, match="current failed"):
        controller._edit_file_load_row(typing.cast("typing.Any", current), row)

    assert applied == []


@pytest.mark.parametrize(
    ("dialog_result", "expected"),
    [
        (int(QtWidgets.QDialog.DialogCode.Accepted), True),
        (int(QtWidgets.QDialog.DialogCode.Rejected), False),
    ],
)
def test_manager_provenance_file_load_batch_failure_confirmation_dialog(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    dialog_result: int,
    expected: bool,
) -> None:
    controller = _fake_edit_controller(_fake_edit_node(full_data()))
    peer_spec = _manager_replay_file_spec(tmp_path / "b.h5")
    peer_node = _fake_edit_node(peer_spec, uid="peer", display_text="Peer")
    peer = provenance_edit_files._FileLoadBatchPeer(
        node=typing.cast("typing.Any", peer_node),
        scope="display",
        spec=peer_spec,
        original_path=tmp_path / "b.h5",
        loader_summary="xarray.load_dataarray",
    )
    dialogs: list[dict[str, typing.Any]] = []

    class _RecordingMessageDialog:
        def __init__(self, parent: typing.Any, **kwargs: typing.Any) -> None:
            self._button_box = QtWidgets.QDialogButtonBox(kwargs["buttons"])
            dialogs.append({"parent": parent, "dialog": self, **kwargs})

        def exec(self) -> int:
            return dialog_result

    monkeypatch.setattr(
        erlab.interactive.utils,
        "MessageDialog",
        _RecordingMessageDialog,
    )

    assert (
        controller._confirm_apply_valid_batch(
            valid_peer_count=1,
            failures=[(peer, RuntimeError("peer failed"))],
        )
        is expected
    )

    assert len(dialogs) == 1
    assert dialogs[0]["parent"] is controller._manager
    assert dialogs[0]["buttons"] == (
        QtWidgets.QDialogButtonBox.StandardButton.Yes
        | QtWidgets.QDialogButtonBox.StandardButton.Cancel
    )
    assert (
        dialogs[0]["default_button"] == QtWidgets.QDialogButtonBox.StandardButton.Cancel
    )
    assert (
        dialogs[0]["icon_pixmap"]
        == QtWidgets.QStyle.StandardPixmap.SP_MessageBoxWarning
    )

    assert "RuntimeError" in dialogs[0]["detailed_text"]
    assert "peer failed" in dialogs[0]["detailed_text"]


def test_manager_provenance_file_load_batch_failure_confirmation_button_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    controller = _fake_edit_controller(_fake_edit_node(full_data()))
    peer_spec = _manager_replay_file_spec(tmp_path / "b.h5")
    peer_node = _fake_edit_node(peer_spec, uid="peer", display_text="Peer")
    peer = provenance_edit_files._FileLoadBatchPeer(
        node=typing.cast("typing.Any", peer_node),
        scope="display",
        spec=peer_spec,
        original_path=tmp_path / "b.h5",
        loader_summary="xarray.load_dataarray",
    )

    class _MissingButtonBox:
        def button(
            self,
            _button: QtWidgets.QDialogButtonBox.StandardButton,
        ) -> QtWidgets.QAbstractButton | None:
            return None

    class _RecordingMessageDialog:
        def __init__(self, *_args: typing.Any, **_kwargs: typing.Any) -> None:
            self._button_box = _MissingButtonBox()

        def exec(self) -> int:
            return int(QtWidgets.QDialog.DialogCode.Accepted)

    monkeypatch.setattr(
        erlab.interactive.utils,
        "MessageDialog",
        _RecordingMessageDialog,
    )

    assert controller._confirm_apply_valid_batch(
        valid_peer_count=0,
        failures=[(peer, RuntimeError("peer failed"))],
    )


def test_manager_provenance_edit_controller_failed_dialog_uses_message_dialog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _fake_edit_controller(_fake_edit_node(full_data()))
    dialogs: list[dict[str, typing.Any]] = []

    class _RecordingMessageDialog:
        def __init__(self, parent: typing.Any, **kwargs: typing.Any) -> None:
            dialogs.append({"parent": parent, **kwargs})

        def exec(self) -> int:
            return int(QtWidgets.QDialog.DialogCode.Accepted)

    monkeypatch.setattr(
        erlab.interactive.utils,
        "MessageDialog",
        _RecordingMessageDialog,
    )

    cause = RuntimeError("missing coord")
    failure = provenance_edit_controller._ProvenanceReplayFailure(
        "validating edited provenance",
        cause,
    )
    failure.add_note(
        "Warnings emitted while replaying provenance:\n"
        "- UserWarning: inferred scan warning"
    )

    controller._show_failed("Could Not Apply Provenance Edit", failure)

    assert len(dialogs) == 1
    assert dialogs[0]["parent"] is controller._manager
    assert "provenance" in dialogs[0]["text"].lower()
    assert "validating edited provenance" in dialogs[0]["informative_text"]
    assert "replay" in dialogs[0]["informative_text"].lower()
    assert "unchanged" in dialogs[0]["informative_text"].lower()
    assert "RuntimeError" in dialogs[0]["detailed_text"]
    assert "missing coord" in dialogs[0]["detailed_text"]
    assert "inferred scan warning" in dialogs[0]["detailed_text"]
    assert (
        dialogs[0]["icon_pixmap"]
        == QtWidgets.QStyle.StandardPixmap.SP_MessageBoxWarning
    )

    controller._show_failed(
        "Could Not Paste Provenance Steps",
        failure,
        text="The copied provenance steps could not be applied.",
        unchanged_reason="The copied steps could not be replayed.",
    )

    assert len(dialogs) == 2
    assert dialogs[1]["text"] == "The copied provenance steps could not be applied."
    assert "The copied steps could not be replayed." in dialogs[1]["informative_text"]


@pytest.mark.parametrize(
    ("dialog_result", "expected_opened"),
    [
        (int(QtWidgets.QDialog.DialogCode.Accepted), 1),
        (int(QtWidgets.QDialog.DialogCode.Rejected), 0),
    ],
)
def test_manager_provenance_missing_source_after_edit_ok_opens_file_load_editor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    dialog_result: int,
    expected_opened: int,
) -> None:
    missing_path = tmp_path / "missing.h5"
    peer_path = tmp_path / "peer.h5"
    spec = _manager_replay_file_spec(
        missing_path,
        IselOperation(kwargs={"x": 0}),
    )
    node = _fake_edit_node(spec, uid="current", display_text="Current")
    peer = _fake_edit_node(
        _manager_replay_file_spec(peer_path),
        uid="peer",
        display_text="Peer",
    )
    controller = _fake_edit_controller(
        node,
        nodes={node.uid: node, peer.uid: peer},
        metadata_uid=node.uid,
    )
    row = _ProvenanceDisplayRow(
        DerivationEntry("isel", None),
        edit_ref=_ProvenanceStepRef(
            "operation",
            operation_index=0,
        ),
        replay_ref=_ProvenanceStepRef(
            "operation",
            operation_index=0,
        ),
    )
    dialogs: list[dict[str, typing.Any]] = []
    opened: list[
        tuple[
            typing.Any,
            str,
            ToolProvenanceSpec,
            tuple[provenance_edit_files._FileLoadBatchPeer, ...],
        ]
    ] = []

    class _RecordingMessageDialog:
        def __init__(self, parent: typing.Any, **kwargs: typing.Any) -> None:
            self._button_box = QtWidgets.QDialogButtonBox(kwargs["buttons"])
            dialogs.append({"parent": parent, "dialog": self, **kwargs})

        def exec(self) -> int:
            return dialog_result

    monkeypatch.setattr(
        erlab.interactive.utils,
        "MessageDialog",
        _RecordingMessageDialog,
    )
    monkeypatch.setattr(
        controller,
        "_edited_native_operations",
        lambda *_args, **_kwargs: [IselOperation(kwargs={"x": 0})],
    )
    missing = provenance_edit_files._MissingProvenanceSourceFileError(missing_path)
    failure = provenance_edit_controller._ProvenanceReplayFailure(
        "validating the edited provenance step",
        missing,
    )
    monkeypatch.setattr(
        controller,
        "_validate_and_replace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
    )
    monkeypatch.setattr(
        controller,
        "_edit_file_load_spec",
        lambda node_arg, scope, spec_arg, **kwargs: opened.append(
            (node_arg, scope, spec_arg, kwargs["batch_peers"])
        ),
    )

    controller.edit_row(row)

    assert len(dialogs) == 1
    assert dialogs[0]["parent"] is controller._manager
    assert "recorded source file" in dialogs[0]["text"].lower()
    assert "validating the edited provenance step" in (dialogs[0]["informative_text"])
    assert str(missing_path) in dialogs[0]["informative_text"]
    assert "Revert to This Step" not in dialogs[0]["informative_text"]
    assert dialogs[0]["buttons"] == (
        QtWidgets.QDialogButtonBox.StandardButton.Yes
        | QtWidgets.QDialogButtonBox.StandardButton.Cancel
    )
    assert len(opened) == expected_opened
    if opened:
        edit_node, scope, opened_spec, batch_peers = opened[0]
        assert (edit_node, scope, opened_spec) == (node, "display", spec)
        assert [batch_peer.node.uid for batch_peer in batch_peers] == [peer.uid]


def test_manager_provenance_missing_source_revert_repairs_revert_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    missing_path = tmp_path / "missing.h5"
    spec = _manager_replay_file_spec(
        missing_path,
        IselOperation(kwargs={"x": 0}),
    )
    node = _fake_edit_node(spec)
    controller = _fake_edit_controller(node)
    row = _ProvenanceDisplayRow(
        DerivationEntry("load", None),
        replay_ref=_ProvenanceStepRef("file_load"),
    )
    opened: list[ToolProvenanceSpec] = []

    class _AcceptingMessageDialog:
        def __init__(self, *_args: typing.Any, **kwargs: typing.Any) -> None:
            self._button_box = QtWidgets.QDialogButtonBox(kwargs["buttons"])

        def exec(self) -> int:
            return int(QtWidgets.QDialog.DialogCode.Accepted)

    monkeypatch.setattr(controller, "_confirm_revert", lambda: True)
    monkeypatch.setattr(
        erlab.interactive.utils,
        "MessageDialog",
        _AcceptingMessageDialog,
    )
    monkeypatch.setattr(
        controller,
        "_edit_file_load_spec",
        lambda _node, _scope, spec_arg, **_kwargs: opened.append(spec_arg),
    )

    controller.revert_row(row)

    assert len(opened) == 1
    assert opened[0].kind == "file"
    assert opened[0].steps == ()


def test_manager_provenance_script_file_revert_reports_missing_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    missing_path = tmp_path / "missing.h5"
    file_spec = _manager_replay_file_spec(
        missing_path,
        IselOperation(kwargs={"x": 0}),
    )
    script_spec = script(
        ScriptCodeOperation(
            label="Use loaded data",
            code="derived = derived + 1.0",
        ),
        start_label=file_spec.start_label,
        seed_code=file_spec.seed_code,
        active_name=file_spec.active_name,
        file_load_source=file_spec.file_load_source,
        steps=file_spec.steps,
    )
    node = _fake_edit_node(script_spec)
    controller = _fake_edit_controller(node)
    row = script_spec.display_rows()[0]
    dialogs: list[dict[str, typing.Any]] = []

    class _RecordingMessageDialog:
        def __init__(self, _parent: typing.Any, **kwargs: typing.Any) -> None:
            self._button_box = QtWidgets.QDialogButtonBox(kwargs["buttons"])
            dialogs.append(kwargs)

        def exec(self) -> int:
            return int(QtWidgets.QDialog.DialogCode.Rejected)

    def raise_missing_rebuild(*_args: typing.Any, **_kwargs: typing.Any) -> None:
        missing = FileNotFoundError(
            2,
            "No such file or directory",
            str(missing_path),
        )
        rebuild_error = manager_widgets._ScriptRebuildError(
            "Could not reload data.",
            details=str(missing),
        )
        raise rebuild_error from missing

    monkeypatch.setattr(controller, "_confirm_revert", lambda: True)
    monkeypatch.setattr(
        erlab.interactive.utils,
        "MessageDialog",
        _RecordingMessageDialog,
    )
    monkeypatch.setattr(
        controller._manager,
        "_rebuild_script_provenance",
        raise_missing_rebuild,
    )
    monkeypatch.setattr(
        controller,
        "_show_failed",
        lambda *_args, **_kwargs: pytest.fail("generic failure dialog was shown"),
    )
    monkeypatch.setattr(
        controller,
        "_edit_file_load_spec",
        lambda *_args, **_kwargs: pytest.fail("cancel should not edit file load"),
    )

    controller.revert_row(row)

    assert len(dialogs) == 1
    assert "recorded source file" in dialogs[0]["text"].lower()
    assert "validating the provenance revert target" in dialogs[0]["informative_text"]
    assert str(missing_path) in dialogs[0]["informative_text"]
    assert (
        "current ImageTool data was left unchanged"
        not in (dialogs[0]["informative_text"])
    )


def test_manager_provenance_missing_source_without_file_load_shows_dedicated_dialog(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    controller = _fake_edit_controller(_fake_edit_node(full_data()))
    missing = provenance_edit_files._MissingProvenanceSourceFileError(
        tmp_path / "missing.h5",
    )
    exc = provenance_edit_controller._ProvenanceReplayFailure(
        "repairing source", missing
    )
    exc.__cause__ = missing
    row = _ProvenanceDisplayRow(DerivationEntry("row", None))
    dialogs: list[dict[str, typing.Any]] = []

    class _RecordingMessageDialog:
        def __init__(self, _parent: typing.Any, **kwargs: typing.Any) -> None:
            self._button_box = QtWidgets.QDialogButtonBox(kwargs["buttons"])
            dialogs.append(kwargs)

        def exec(self) -> int:
            return int(QtWidgets.QDialog.DialogCode.Accepted)

    monkeypatch.setattr(
        erlab.interactive.utils,
        "MessageDialog",
        _RecordingMessageDialog,
    )
    monkeypatch.setattr(
        controller,
        "_edit_file_load_spec",
        lambda *_args, **_kwargs: pytest.fail("no file-load editor is available"),
    )

    assert controller._handle_missing_source_file(
        typing.cast("typing.Any", _fake_edit_node(full_data())),
        row,
        title="Could Not Apply Provenance Edit",
        exc=exc,
    )

    assert len(dialogs) == 1
    assert dialogs[0]["buttons"] == QtWidgets.QDialogButtonBox.StandardButton.Ok
    assert dialogs[0]["default_button"] == (
        QtWidgets.QDialogButtonBox.StandardButton.Ok
    )
    assert "repairing source" in dialogs[0]["informative_text"]


def test_manager_provenance_missing_source_dialog_button_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    controller = _fake_edit_controller(_fake_edit_node(full_data()))
    missing = provenance_edit_files._MissingProvenanceSourceFileError(
        tmp_path / "missing.h5",
    )

    class _MissingButtonBox:
        def button(
            self,
            _button: QtWidgets.QDialogButtonBox.StandardButton,
        ) -> QtWidgets.QAbstractButton | None:
            return None

    class _RecordingMessageDialog:
        def __init__(self, *_args: typing.Any, **_kwargs: typing.Any) -> None:
            self._button_box = _MissingButtonBox()

        def exec(self) -> int:
            return int(QtWidgets.QDialog.DialogCode.Accepted)

    monkeypatch.setattr(
        erlab.interactive.utils,
        "MessageDialog",
        _RecordingMessageDialog,
    )

    assert controller._show_missing_source_file(
        "Could Not Apply Provenance Edit",
        missing,
        missing,
        can_edit=True,
    )


def test_manager_provenance_missing_nested_source_uses_batch_relink_dialog(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    old_dir = tmp_path / "old"
    old_dir.mkdir()
    first_spec = _manager_replay_file_spec(old_dir / "a.h5")
    second_spec = _manager_replay_file_spec(old_dir / "b.h5")
    root_spec = script(
        ScriptCodeOperation(
            label="Combine inputs",
            code="derived = data_0 + data_1",
        ),
        start_label="Run manager script",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="data_0",
                label="First",
                provenance_spec=first_spec,
            ),
            ScriptInput(
                name="data_1",
                label="Second",
                provenance_spec=second_spec,
            ),
        ),
    )
    node = _fake_edit_node(root_spec)
    controller = _fake_edit_controller(node)
    missing = provenance_edit_files._MissingProvenanceSourceFileError(old_dir / "a.h5")
    exc = provenance_edit_controller._ProvenanceReplayFailure(
        "replaying script input",
        missing,
    )
    script_row = root_spec.display_rows()[-1]
    dialog_calls: list[
        tuple[
            pathlib.Path,
            tuple[tuple[int, ...], ...],
            tuple[tuple[int, ...], ...],
            bool,
        ]
    ] = []

    class _AcceptingMessageDialog:
        def __init__(self, _parent: typing.Any, **kwargs: typing.Any) -> None:
            self._button_box = QtWidgets.QDialogButtonBox(kwargs["buttons"])

        def exec(self) -> int:
            return int(QtWidgets.QDialog.DialogCode.Accepted)

    class _Dialog:
        def __init__(
            self,
            load_source: FileLoadSource,
            _parent: QtWidgets.QWidget,
            *,
            batch_peers: tuple[provenance_edit_files._FileLoadBatchPeer, ...],
            batch_apply_default: bool = False,
            checked_batch_peer_ids: frozenset[str] | None = None,
        ) -> None:
            checked_batch_peer_ids = checked_batch_peer_ids or frozenset()
            dialog_calls.append(
                (
                    pathlib.Path(load_source.path),
                    tuple(peer.script_input_path for peer in batch_peers),
                    tuple(
                        peer.script_input_path
                        for peer in batch_peers
                        if peer.target_id in checked_batch_peer_ids
                    ),
                    batch_apply_default,
                )
            )
            assert [peer.preserve_loader for peer in batch_peers] == [True]

        def exec(self) -> int:
            return int(QtWidgets.QDialog.DialogCode.Rejected)

    monkeypatch.setattr(
        erlab.interactive.utils,
        "MessageDialog",
        _AcceptingMessageDialog,
    )
    monkeypatch.setattr(provenance_edit_controller, "_FileLoadEditDialog", _Dialog)

    assert controller._handle_missing_source_file(
        typing.cast("typing.Any", node),
        script_row,
        title="Could Not Apply Provenance Edit",
        exc=exc,
    )

    assert dialog_calls == [(old_dir / "a.h5", ((1,),), ((1,),), True)]


def test_manager_provenance_missing_nested_repair_relinks_nonmatching_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    old_a_dir = tmp_path / "old-a"
    old_b_dir = tmp_path / "old-b"
    new_dir = tmp_path / "new"
    old_a_dir.mkdir()
    old_b_dir.mkdir()
    new_dir.mkdir()
    old_a_path = old_a_dir / "a.h5"
    old_b_path = old_b_dir / "b.h5"
    new_a_path = new_dir / "a.h5"
    new_b_path = new_dir / "b.h5"
    first_spec = _manager_replay_file_spec(old_a_path)
    second_spec = _manager_replay_file_spec(
        old_b_path,
        IselOperation(kwargs={"x": 0}),
    )
    assert second_spec.file_load_source is not None
    second_replay_call = second_spec.file_load_source.replay_call
    assert second_replay_call is not None
    second_spec = second_spec.model_copy(
        update={
            "file_load_source": second_spec.file_load_source.model_copy(
                update={
                    "kwargs_text": "engine='scipy'",
                    "replay_call": second_replay_call.model_copy(
                        update={"kwargs": {"engine": "scipy"}},
                    ),
                }
            )
        }
    )
    root_spec = script(
        ScriptCodeOperation(
            label="Combine inputs",
            code="derived = data_0 + data_1",
        ),
        start_label="Run manager script",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="data_0",
                label="First",
                node_uid="deleted-a",
                node_snapshot_token=str(object()),
                provenance_spec=first_spec,
            ),
            ScriptInput(
                name="data_1",
                label="Second",
                node_uid="deleted-b",
                node_snapshot_token=str(object()),
                provenance_spec=second_spec,
            ),
        ),
    )
    node = _fake_edit_node(root_spec)
    controller = _fake_edit_controller(node)
    missing = provenance_edit_files._MissingProvenanceSourceFileError(old_a_path)
    exc = provenance_edit_controller._ProvenanceReplayFailure(
        "replaying script input",
        missing,
    )
    script_row = root_spec.display_rows()[-1]
    validated: list[ToolProvenanceSpec] = []
    applied: list[ToolProvenanceSpec] = []

    class _AcceptingMessageDialog:
        def __init__(self, _parent: typing.Any, **kwargs: typing.Any) -> None:
            self._button_box = QtWidgets.QDialogButtonBox(kwargs["buttons"])

        def exec(self) -> int:
            return int(QtWidgets.QDialog.DialogCode.Accepted)

    class _Dialog:
        def __init__(
            self,
            load_source: FileLoadSource,
            _parent: QtWidgets.QWidget,
            *,
            batch_peers: tuple[provenance_edit_files._FileLoadBatchPeer, ...],
            batch_apply_default: bool = False,
            checked_batch_peer_ids: frozenset[str] | None = None,
        ) -> None:
            assert pathlib.Path(load_source.path) == old_a_path
            assert batch_apply_default is True
            assert [peer.script_input_path for peer in batch_peers] == [(1,)]
            assert [peer.preserve_loader for peer in batch_peers] == [True]
            assert checked_batch_peer_ids == frozenset(
                peer.target_id for peer in batch_peers
            )
            self._batch_peers = batch_peers

        def exec(self) -> int:
            return int(QtWidgets.QDialog.DialogCode.Accepted)

        def provenance_spec(
            self,
            *,
            active_name: str,
            replay_steps: tuple[ReplayStep, ...],
        ) -> ToolProvenanceSpec:
            del active_name, replay_steps
            return _manager_replay_file_spec(new_a_path)

        def selected_batch_peers(
            self,
        ) -> tuple[provenance_edit_files._FileLoadBatchPeer, ...]:
            return self._batch_peers

        def peer_provenance_spec(
            self,
            peer: provenance_edit_files._FileLoadBatchPeer,
        ) -> ToolProvenanceSpec:
            return provenance_edit_files._relinked_file_load_spec(
                peer.spec,
                new_b_path,
            )

    monkeypatch.setattr(
        erlab.interactive.utils,
        "MessageDialog",
        _AcceptingMessageDialog,
    )
    monkeypatch.setattr(provenance_edit_controller, "_FileLoadEditDialog", _Dialog)
    monkeypatch.setattr(
        controller,
        "_validated_edit",
        lambda edit_node, scope, candidate, **_kwargs: (
            validated.append(candidate)
            or provenance_edit_controller._ValidatedProvenanceEdit(
                node=typing.cast("typing.Any", edit_node),
                scope=scope,
                data=xr.DataArray([1.0], dims=("x",)),
                spec=candidate,
                filter_operation=None,
            )
        ),
    )
    monkeypatch.setattr(
        controller,
        "_apply_validated_edit",
        lambda edit: applied.append(edit.spec),
    )

    assert controller._handle_missing_source_file(
        typing.cast("typing.Any", node),
        script_row,
        title="Could Not Apply Provenance Edit",
        exc=exc,
    )

    assert len(validated) == 1
    assert applied == validated
    first_relinked = validated[0].script_inputs[0].parsed_provenance_spec()
    second_relinked = validated[0].script_inputs[1].parsed_provenance_spec()
    assert first_relinked is not None
    assert second_relinked is not None
    assert first_relinked.file_load_source is not None
    assert second_relinked.file_load_source is not None
    assert pathlib.Path(first_relinked.file_load_source.path) == new_a_path
    assert pathlib.Path(second_relinked.file_load_source.path) == new_b_path
    assert second_relinked.steps == second_spec.steps
    second_replay_call = second_relinked.file_load_source.replay_call
    assert second_replay_call is not None
    assert second_replay_call.kwargs == {"engine": "scipy"}
    assert validated[0].script_inputs[0].node_uid is None
    assert validated[0].script_inputs[1].node_uid is None


def test_manager_provenance_missing_nested_repair_partial_selection_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"
    old_dir.mkdir()
    new_dir.mkdir()
    old_a_path = old_dir / "a.h5"
    old_b_path = old_dir / "b.h5"
    new_a_path = new_dir / "a.h5"
    first_spec = _manager_replay_file_spec(old_a_path)
    second_spec = _manager_replay_file_spec(old_b_path)
    root_spec = script(
        ScriptCodeOperation(
            label="Combine inputs",
            code="derived = data_0 + data_1",
        ),
        start_label="Run manager script",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="data_0",
                label="First",
                provenance_spec=first_spec,
            ),
            ScriptInput(
                name="data_1",
                label="Second",
                provenance_spec=second_spec,
            ),
        ),
    )
    node = _fake_edit_node(root_spec)
    controller = _fake_edit_controller(node)
    missing = provenance_edit_files._MissingProvenanceSourceFileError(old_a_path)
    exc = provenance_edit_controller._ProvenanceReplayFailure(
        "replaying script input",
        missing,
    )
    script_row = root_spec.display_rows()[-1]
    validated: list[ToolProvenanceSpec] = []
    dialog_results = [
        int(QtWidgets.QDialog.DialogCode.Accepted),
        int(QtWidgets.QDialog.DialogCode.Rejected),
    ]

    class _MessageDialog:
        def __init__(self, _parent: typing.Any, **kwargs: typing.Any) -> None:
            self._button_box = QtWidgets.QDialogButtonBox(kwargs["buttons"])

        def exec(self) -> int:
            return dialog_results.pop(0)

    class _Dialog:
        def __init__(
            self,
            load_source: FileLoadSource,
            _parent: QtWidgets.QWidget,
            *,
            batch_peers: tuple[provenance_edit_files._FileLoadBatchPeer, ...],
            batch_apply_default: bool = False,
            checked_batch_peer_ids: frozenset[str] | None = None,
        ) -> None:
            assert pathlib.Path(load_source.path) == old_a_path
            assert batch_apply_default is True
            assert [peer.script_input_path for peer in batch_peers] == [(1,)]
            assert checked_batch_peer_ids == frozenset(
                peer.target_id for peer in batch_peers
            )

        def exec(self) -> int:
            return int(QtWidgets.QDialog.DialogCode.Accepted)

        def provenance_spec(
            self,
            *,
            active_name: str,
            replay_steps: tuple[ReplayStep, ...],
        ) -> ToolProvenanceSpec:
            del active_name, replay_steps
            return _manager_replay_file_spec(new_a_path)

        def selected_batch_peers(
            self,
        ) -> tuple[provenance_edit_files._FileLoadBatchPeer, ...]:
            return ()

    def _validated_edit(
        _edit_node: typing.Any,
        _scope: typing.Literal["display", "source"],
        candidate: ToolProvenanceSpec,
        **_kwargs: typing.Any,
    ) -> provenance_edit_controller._ValidatedProvenanceEdit:
        validated.append(candidate)
        second_candidate = candidate.script_inputs[1].parsed_provenance_spec()
        assert second_candidate is not None
        assert second_candidate.file_load_source is not None
        assert pathlib.Path(second_candidate.file_load_source.path) == old_b_path
        still_missing = provenance_edit_files._MissingProvenanceSourceFileError(
            old_b_path,
        )
        raise provenance_edit_controller._ProvenanceReplayFailure(
            "validating replacement",
            still_missing,
        ) from still_missing

    monkeypatch.setattr(erlab.interactive.utils, "MessageDialog", _MessageDialog)
    monkeypatch.setattr(provenance_edit_controller, "_FileLoadEditDialog", _Dialog)
    monkeypatch.setattr(controller, "_validated_edit", _validated_edit)
    monkeypatch.setattr(
        controller,
        "_apply_validated_edit",
        lambda _edit: pytest.fail("invalid partial repair must not apply"),
    )

    assert controller._handle_missing_source_file(
        typing.cast("typing.Any", node),
        script_row,
        title="Could Not Apply Provenance Edit",
        exc=exc,
    )

    assert len(validated) == 1
    first_candidate = validated[0].script_inputs[0].parsed_provenance_spec()
    assert first_candidate is not None
    assert first_candidate.file_load_source is not None
    assert pathlib.Path(first_candidate.file_load_source.path) == new_a_path
    assert node.displayed_provenance_spec == root_spec


def test_manager_provenance_missing_nested_source_repair_keeps_script_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    file_spec = _manager_replay_file_spec(tmp_path / "missing.h5")
    root_spec = script(
        ScriptCodeOperation(
            label="Use file input",
            code="derived = data_0",
        ),
        start_label="Run manager script",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="data_0",
                label="File input",
                provenance_spec=file_spec,
            ),
        ),
    )
    node = _fake_edit_node(root_spec)
    controller = _fake_edit_controller(node)
    file_row = root_spec.display_rows()[1].children[0]
    missing = provenance_edit_files._MissingProvenanceSourceFileError(
        tmp_path / "missing.h5"
    )
    exc = provenance_edit_controller._ProvenanceReplayFailure(
        "replaying nested file input",
        missing,
    )
    edit_calls: list[
        tuple[
            typing.Literal["display", "source"],
            ToolProvenanceSpec,
            _ProvenanceDisplayRow | None,
        ]
    ] = []

    class _AcceptingMessageDialog:
        def __init__(self, _parent: typing.Any, **kwargs: typing.Any) -> None:
            self._button_box = QtWidgets.QDialogButtonBox(kwargs["buttons"])

        def exec(self) -> int:
            return int(QtWidgets.QDialog.DialogCode.Accepted)

    monkeypatch.setattr(
        erlab.interactive.utils,
        "MessageDialog",
        _AcceptingMessageDialog,
    )
    monkeypatch.setattr(
        controller,
        "_edit_file_load_spec",
        lambda _node, scope, spec, *, row=None, **_kwargs: edit_calls.append(
            (scope, spec, row)
        ),
    )

    assert controller._handle_missing_source_file(
        typing.cast("typing.Any", node),
        file_row,
        title="Could Not Apply Provenance Edit",
        exc=exc,
        repair_spec=file_spec,
    )

    assert len(edit_calls) == 1
    scope, edited_spec, repair_row = edit_calls[0]
    assert scope == "display"
    assert edited_spec == file_spec
    assert repair_row is not None
    assert repair_row.script_input_path == file_row.script_input_path


def test_manager_provenance_missing_source_repair_relinks_repair_root_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    old_path = tmp_path / "missing.h5"
    new_path = tmp_path / "current.h5"
    old_file_spec = _manager_replay_file_spec(old_path)
    new_file_spec = _manager_replay_file_spec(new_path)
    current_root = script(
        ScriptCodeOperation(
            label="Use file input",
            code="derived = data_0",
        ),
        AverageOperation(dims=("x",)),
        IselOperation(kwargs={"y": 0}),
        start_label="Run manager script",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="data_0",
                label="File input",
                provenance_spec=old_file_spec,
            ),
        ),
    )
    repair_root = current_root.model_copy(
        update={"operations": current_root.operations[:-1]}
    )
    assert current_root.operations != repair_root.operations
    node = _fake_edit_node(current_root)
    controller = _fake_edit_controller(node)
    missing = provenance_edit_files._MissingProvenanceSourceFileError(old_path)
    exc = provenance_edit_controller._ProvenanceReplayFailure(
        "replaying revert target",
        missing,
    )
    row = _ProvenanceDisplayRow(
        DerivationEntry("Aggregate", None),
    )
    validated: list[ToolProvenanceSpec] = []

    class _AcceptingMessageDialog:
        def __init__(self, _parent: typing.Any, **kwargs: typing.Any) -> None:
            self._button_box = QtWidgets.QDialogButtonBox(kwargs["buttons"])

        def exec(self) -> int:
            return int(QtWidgets.QDialog.DialogCode.Accepted)

    class _Dialog:
        def __init__(
            self,
            load_source: FileLoadSource,
            _parent: QtWidgets.QWidget,
            **_kwargs: typing.Any,
        ) -> None:
            assert pathlib.Path(load_source.path) == old_path

        def exec(self) -> int:
            return int(QtWidgets.QDialog.DialogCode.Accepted)

        def provenance_spec(
            self,
            *,
            active_name: str,
            replay_steps: tuple[ReplayStep, ...],
        ) -> ToolProvenanceSpec:
            del active_name, replay_steps
            return new_file_spec

        def selected_batch_peers(
            self,
        ) -> tuple[provenance_edit_files._FileLoadBatchPeer, ...]:
            return ()

    monkeypatch.setattr(
        erlab.interactive.utils,
        "MessageDialog",
        _AcceptingMessageDialog,
    )
    monkeypatch.setattr(provenance_edit_controller, "_FileLoadEditDialog", _Dialog)
    monkeypatch.setattr(
        controller,
        "_validated_edit",
        lambda edit_node, scope, candidate, **_kwargs: (
            validated.append(candidate)
            or provenance_edit_controller._ValidatedProvenanceEdit(
                node=typing.cast("typing.Any", edit_node),
                scope=scope,
                data=xr.DataArray([1.0], dims=("x",)),
                spec=candidate,
                filter_operation=None,
            )
        ),
    )
    monkeypatch.setattr(controller, "_apply_validated_edit", lambda _edit: None)

    assert controller._handle_missing_source_file(
        typing.cast("typing.Any", node),
        row,
        title="Could Not Revert Provenance Step",
        exc=exc,
        repair_spec=repair_root,
    )

    assert len(validated) == 1
    assert validated[0].operations == repair_root.operations
    edited_input = validated[0].script_inputs[0].parsed_provenance_spec()
    assert edited_input is not None
    assert edited_input.file_load_source is not None
    assert pathlib.Path(edited_input.file_load_source.path) == new_path


@pytest.mark.parametrize("missing_again", [False, True])
def test_manager_provenance_missing_source_repair_failure_branches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    missing_again: bool,
) -> None:
    spec = _manager_replay_file_spec(tmp_path / "missing.h5")
    assert spec.file_load_source is not None
    node = _fake_edit_node(spec)
    controller = _fake_edit_controller(node)
    missing = provenance_edit_files._MissingProvenanceSourceFileError(
        tmp_path / "missing.h5",
    )
    exc = provenance_edit_controller._ProvenanceReplayFailure("replaying file", missing)
    exc.__cause__ = missing
    row = _ProvenanceDisplayRow(DerivationEntry("row", None))
    failed: list[Exception] = []
    dialogs: list[str] = []
    repair_attempts = 0
    dialog_results = [int(QtWidgets.QDialog.DialogCode.Accepted)]
    if missing_again:
        dialog_results.extend(
            [
                int(QtWidgets.QDialog.DialogCode.Accepted),
                int(QtWidgets.QDialog.DialogCode.Rejected),
            ]
        )

    class _AcceptingMessageDialog:
        def __init__(self, _parent: typing.Any, **kwargs: typing.Any) -> None:
            self._button_box = QtWidgets.QDialogButtonBox(kwargs["buttons"])
            dialogs.append(kwargs["text"])

        def exec(self) -> int:
            return dialog_results.pop(0)

    def _raise_repair_failure(*_args: typing.Any, **_kwargs: typing.Any) -> None:
        nonlocal repair_attempts
        repair_attempts += 1
        if not missing_again:
            raise RuntimeError("repair failed")
        replacement_missing = provenance_edit_files._MissingProvenanceSourceFileError(
            tmp_path / "still-missing.h5",
        )
        repair_exc = provenance_edit_controller._ProvenanceReplayFailure(
            "validating replacement",
            replacement_missing,
        )
        raise repair_exc from replacement_missing

    monkeypatch.setattr(
        erlab.interactive.utils,
        "MessageDialog",
        _AcceptingMessageDialog,
    )
    monkeypatch.setattr(controller, "_edit_file_load_spec", _raise_repair_failure)
    monkeypatch.setattr(
        controller,
        "_show_failed",
        lambda _title, failure: failed.append(failure),
    )

    assert controller._handle_missing_source_file(
        typing.cast("typing.Any", node),
        row,
        title="Could Not Apply Provenance Edit",
        exc=exc,
    )

    if missing_again:
        assert len(dialogs) == 3
        assert repair_attempts == 2
        assert failed == []
    else:
        assert len(dialogs) == 1
        assert repair_attempts == 1
        assert len(failed) == 1
        assert "repair failed" in str(failed[0])


def test_manager_provenance_file_replay_validation_prechecks_missing_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    controller = _fake_edit_controller(_fake_edit_node(full_data()))
    spec = _manager_provenance_file_spec(tmp_path / "missing.h5")

    monkeypatch.setattr(
        provenance_edit_controller,
        "replay_file_provenance",
        lambda _spec: pytest.fail("missing files should fail before loader replay"),
    )

    with pytest.raises(FileNotFoundError, match="no longer accessible"):
        controller._replay_file_candidate(spec)
    with pytest.raises(provenance_edit_files._MissingProvenanceSourceFileError) as exc:
        controller._replay_file_candidate(spec)
    assert exc.value.source_path == tmp_path / "missing.h5"


def test_manager_provenance_file_replay_validation_captures_loader_warnings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    file_path = tmp_path / "scan.h5"
    file_path.touch()
    controller = _fake_edit_controller(_fake_edit_node(full_data()))
    spec = _manager_provenance_file_spec(file_path)

    def _warn_then_fail(_spec: ToolProvenanceSpec) -> xr.DataArray:
        warnings.warn(
            "Loading f_003_S001 with inferred index 3 resulted in an error.",
            UserWarning,
            stacklevel=1,
        )
        raise RuntimeError("real replay failure")

    monkeypatch.setattr(
        provenance_edit_controller,
        "replay_file_provenance",
        _warn_then_fail,
    )

    with warnings.catch_warnings(record=True) as escaped_warnings:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match="real replay failure") as exc_info:
            controller._replay_file_candidate(spec)

    assert not escaped_warnings
    notes = "\n".join(getattr(exc_info.value, "__notes__", ()))
    assert "Warnings emitted while replaying provenance" in notes
    assert "inferred index 3" in notes


def test_manager_provenance_edit_controller_private_error_branches() -> None:
    controller = _fake_edit_controller(_fake_edit_node(None))
    file_row = _ProvenanceDisplayRow(
        DerivationEntry("load", None),
        edit_ref=_ProvenanceStepRef("file_load"),
    )
    with pytest.raises(RuntimeError, match="file load"):
        controller._edit_file_load_row(
            typing.cast("typing.Any", _fake_edit_node(None)), file_row
        )
    with pytest.raises(RuntimeError, match="file load"):
        controller._edit_file_load_spec(
            typing.cast("typing.Any", _fake_edit_node(full_data())),
            "display",
            full_data(),
            where="testing",
        )

    operation_row = _ProvenanceDisplayRow(
        DerivationEntry("op", None),
        edit_ref=_ProvenanceStepRef("operation", operation_index=0),
    )
    with pytest.raises(RuntimeError, match="No provenance"):
        controller._edit_operation_row(
            typing.cast("typing.Any", _fake_edit_node(None)), operation_row
        )
    with pytest.raises(RuntimeError, match="not available"):
        controller._edit_operation_row(
            typing.cast("typing.Any", _fake_edit_node(full_data())),
            operation_row,
        )
    with pytest.raises(RuntimeError, match="No editing dialog"):
        controller._edit_operation_row(
            typing.cast(
                "typing.Any",
                _fake_edit_node(full_data(RestoreNonuniformDimsOperation())),
            ),
            operation_row,
        )
