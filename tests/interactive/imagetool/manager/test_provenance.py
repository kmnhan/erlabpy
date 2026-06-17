import enum
import json
import pathlib
import types
import typing
import warnings
from collections.abc import Callable

import numpy as np
import pydantic
import pytest
import xarray
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

import erlab
import erlab.interactive.imagetool.manager._details_panel as manager_details_panel
import erlab.interactive.imagetool.manager._dialogs as manager_dialogs
import erlab.interactive.imagetool.manager._provenance_edit as manager_provenance_edit
import erlab.interactive.imagetool.manager._widgets as manager_widgets
import erlab.interactive.imagetool.manager._workspace_io as manager_workspace_io
import erlab.interactive.imagetool.manager._wrapper as manager_wrapper
from erlab.interactive._fit2d import Fit2DTool
from erlab.interactive._mesh import MeshTool
from erlab.interactive.derivative import DerivativeTool
from erlab.interactive.fermiedge import GoldTool, ResolutionTool
from erlab.interactive.imagetool import itool, provenance
from erlab.interactive.imagetool.dialogs import SelectionDialog
from erlab.interactive.imagetool.manager import fetch, replace_data
from erlab.interactive.imagetool.manager._modelview import (
    _NODE_UID_ROLE,
    _ImageToolWrapperItemDelegate,
)
from erlab.interactive.imagetool.manager._server import _remove_idx, _show_idx

from .helpers import (
    InMemoryClipboard,
    _assert_modelfit_code_replays_source,
    _exec_generated_code,
    child_status_badge,
    click_child_status_badge,
    configure_goldtool_child,
    copy_full_code_for_uid,
    install_in_memory_clipboard,
    make_fit1d_child,
    make_fit2d_child,
    manager_preview_pixmap,
    menu_map_by_object_name,
    metadata_derivation_texts,
    metadata_detail_map,
    select_child_tool,
    select_metadata_rows,
    select_tools,
    set_transform_launch_mode,
    trigger_menu_action,
)


@pytest.fixture(autouse=True)
def isolate_qt_clipboard(monkeypatch: pytest.MonkeyPatch) -> InMemoryClipboard:
    return install_in_memory_clipboard(monkeypatch)


def _manager_provenance_file_spec(path: pathlib.Path):
    return provenance.file_load(
        start_label="Load source",
        seed_code=f"derived = xr.load_dataarray({str(path)!r})",
        file_load_source=provenance.FileLoadSource(
            path=str(path),
            loader_label="xarray.load_dataarray",
            loader_text="xarray.load_dataarray",
            kwargs_text="",
            replay_call=provenance.FileReplayCall(
                kind="callable",
                target="xarray.load_dataarray",
                selected_index=0,
            ),
        ),
    )


def _manager_replay_file_spec(
    path: pathlib.Path,
    *operations: provenance.ToolProvenanceOperation,
) -> provenance.ToolProvenanceSpec:
    spec = provenance.file_load(
        start_label=f"Load data from file {path.name!r}",
        seed_code=(
            "import xarray\n\n"
            f"derived = xarray.load_dataarray({str(path)!r}, engine='h5netcdf')"
        ),
        file_load_source=provenance.FileLoadSource(
            path=str(path),
            loader_label="Load Function",
            loader_text="xarray.load_dataarray",
            kwargs_text="engine='h5netcdf'",
            replay_call=provenance.FileReplayCall(
                kind="callable",
                target="xarray.load_dataarray",
                kwargs={"engine": "h5netcdf"},
                selected_index=0,
            ),
        ),
    )
    if operations:
        spec = spec.append_replay_stage(provenance.full_data(*operations))
    return spec


def _add_file_replay_tool(
    manager: erlab.interactive.imagetool.manager.ImageToolManager,
    data: xr.DataArray,
    spec: provenance.ToolProvenanceSpec,
) -> erlab.interactive.imagetool.ImageTool:
    tool = itool(data, manager=False, execute=False)
    assert isinstance(tool, erlab.interactive.imagetool.ImageTool)
    manager.add_imagetool(tool, show=False, provenance_spec=spec)
    return tool


def _provenance_paste_test_data(name: str = "data") -> xr.DataArray:
    return xr.DataArray(
        np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5),
        dims=("x", "y", "z"),
        coords={"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0, 3.0]},
        name=name,
    )


def test_file_load_edit_dialog_uses_loader_options_widget(qtbot) -> None:
    load_source = provenance.FileLoadSource(
        path="scan.h5",
        loader_label="Load Function",
        loader_text="xarray.load_dataarray",
        kwargs_text="engine='h5netcdf'",
        replay_call=provenance.FileReplayCall(
            kind="callable",
            target="xarray.load_dataarray",
            kwargs={"engine": "h5netcdf"},
            selected_index=0,
        ),
    )
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    dialog = manager_provenance_edit._FileLoadEditDialog(load_source, parent)
    qtbot.addWidget(dialog)

    assert isinstance(dialog.loader_options, manager_dialogs._LoaderOptionsWidget)
    assert dialog.kwargs_edit is dialog.loader_options.kwargs_line

    dialog.kwargs_edit.setText("engine='h5netcdf', chunks={'x': 1}")

    assert dialog.loader_options.checked_filter()[2] == {
        "engine": "h5netcdf",
        "chunks": {"x": 1},
    }


def test_file_load_edit_dialog_allows_loader_change(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    example_loader,
) -> None:
    loader = example_loader()
    monkeypatch.setitem(erlab.io.loaders._loaders, loader.name, loader)
    monkeypatch.setitem(erlab.io.loaders._alias_mapping, loader.name, loader.name)
    load_source = provenance.FileLoadSource(
        path=str(tmp_path / "scan.h5"),
        loader_label="Load Function",
        loader_text="xarray.load_dataarray",
        kwargs_text="engine='h5netcdf'",
        replay_call=provenance.FileReplayCall(
            kind="callable",
            target="xarray.load_dataarray",
            kwargs={"engine": "h5netcdf"},
            selected_index=0,
        ),
    )
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    dialog = manager_provenance_edit._FileLoadEditDialog(load_source, parent)
    qtbot.addWidget(dialog)

    loader_filters = list(dialog.loader_options._valid_loaders)
    example_filter = next(
        name
        for name, (func, _kwargs) in dialog.loader_options._valid_loaders.items()
        if getattr(func, "__self__", None) is loader
    )
    dialog.loader_options._button_group.button(
        loader_filters.index(example_filter)
    ).setChecked(True)
    dialog.kwargs_edit.setText("single=True")

    spec = dialog.provenance_spec(active_name="derived", replay_stages=())

    assert spec.file_load_source is not None
    replay_call = spec.file_load_source.replay_call
    assert replay_call is not None
    assert replay_call.kind == "erlab_loader"
    assert replay_call.target == "example"
    assert replay_call.kwargs == {"single": True}
    assert replay_call.selection == provenance.FileDataSelection(
        kind="parsed_index", value=0
    )


def test_file_load_edit_dialog_updates_loader_options_for_replacement_path(
    qtbot,
    tmp_path: pathlib.Path,
) -> None:
    load_source = provenance.FileLoadSource(
        path=str(tmp_path / "scan.h5"),
        loader_label="Load Function",
        loader_text="xarray.load_dataarray",
        kwargs_text="engine='h5netcdf'",
        replay_call=provenance.FileReplayCall(
            kind="callable",
            target="xarray.load_dataarray",
            kwargs={"engine": "h5netcdf"},
            selected_index=0,
        ),
    )
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    dialog = manager_provenance_edit._FileLoadEditDialog(load_source, parent)
    qtbot.addWidget(dialog)

    initial_filter = dialog._checked_filter_name()
    assert initial_filter is not None
    dialog.kwargs_edit.setText("engine='h5netcdf', chunks={'x': 1}")

    same_suffix_path = tmp_path / "replacement.h5"
    dialog.path_edit.setText(str(same_suffix_path))

    assert dialog.loader_options._sample_paths == (same_suffix_path,)
    assert dialog._checked_filter_name() == initial_filter
    assert dialog.kwargs_edit.text() == "engine='h5netcdf', chunks={'x': 1}"

    new_suffix_path = tmp_path / "replacement.nc"
    dialog.path_edit.setText(str(new_suffix_path))

    loader_filters = tuple(dialog.loader_options._valid_loaders)
    assert dialog.loader_options._sample_paths == (new_suffix_path,)
    assert initial_filter not in loader_filters
    assert any("*.nc" in name for name in loader_filters)


def test_file_load_edit_dialog_batch_targets_and_path_mapping(
    qtbot,
    tmp_path: pathlib.Path,
) -> None:
    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"
    old_dir.mkdir()
    new_dir.mkdir()
    current_spec = _manager_replay_file_spec(old_dir / "a.h5")
    peer_spec = _manager_replay_file_spec(old_dir / "b.h5")
    assert current_spec.file_load_source is not None
    peer_node = types.SimpleNamespace(uid="peer", display_text="Peer")
    peer = manager_provenance_edit._FileLoadBatchPeer(
        node=typing.cast("typing.Any", peer_node),
        scope="display",
        spec=peer_spec,
        original_path=old_dir / "b.h5",
        loader_summary="xarray.load_dataarray",
    )
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    dialog = manager_provenance_edit._FileLoadEditDialog(
        current_spec.file_load_source,
        parent,
        batch_peers=(peer,),
    )
    qtbot.addWidget(dialog)

    assert dialog.batch_apply_check.isEnabled()
    assert dialog.batch_peer_tree.isHidden()
    assert dialog.selected_batch_peers() == ()

    dialog.batch_apply_check.setChecked(True)
    item = dialog.batch_peer_tree.topLevelItem(0)
    assert not dialog.batch_peer_tree.isHidden()
    assert item.checkState(0) == QtCore.Qt.CheckState.Checked
    assert pathlib.Path(item.text(2)) == old_dir / "b.h5"
    assert dialog.selected_batch_peers() == (peer,)

    dialog.kwargs_edit.setText("engine='h5netcdf', chunks={'x': 1}")
    assert pathlib.Path(item.text(2)) == old_dir / "b.h5"

    dialog.path_edit.setText(str(new_dir / "a.nc"))
    assert pathlib.Path(item.text(2)) == new_dir / "b.nc"

    item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
    assert dialog.selected_batch_peers() == ()


def test_file_load_edit_dialog_batch_defensive_branches(
    qtbot,
    tmp_path: pathlib.Path,
) -> None:
    current_spec = _manager_replay_file_spec(tmp_path / "a.h5")
    assert current_spec.file_load_source is not None
    peer_spec = _manager_replay_file_spec(tmp_path / "b.h5")
    peer_node = types.SimpleNamespace(uid="peer", display_text="Peer")
    peer = manager_provenance_edit._FileLoadBatchPeer(
        node=typing.cast("typing.Any", peer_node),
        scope="display",
        spec=peer_spec,
        original_path=tmp_path / "b.h5",
        loader_summary="xarray.load_dataarray",
    )
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    dialog = manager_provenance_edit._FileLoadEditDialog(
        current_spec.file_load_source,
        parent,
        batch_peers=(peer,),
    )
    qtbot.addWidget(dialog)
    dialog.batch_apply_check.setChecked(True)

    class _MissingUidItem:
        def checkState(self, _column: int) -> QtCore.Qt.CheckState:
            return QtCore.Qt.CheckState.Checked

        def data(self, _column: int, _role: QtCore.Qt.ItemDataRole) -> str:
            return "missing"

        def setText(self, _column: int, _text: str) -> None:
            pytest.fail("unknown batch peer rows should not be updated")

    class _SparseTree:
        def topLevelItemCount(self) -> int:
            return 2

        def topLevelItem(self, row: int) -> _MissingUidItem | None:
            return None if row == 0 else _MissingUidItem()

    dialog.batch_peer_tree = typing.cast("typing.Any", _SparseTree())

    assert dialog.selected_batch_peers() == ()
    dialog._update_batch_peer_paths()


def test_file_load_edit_dialog_rejects_stale_batch_peer(qtbot) -> None:
    current_spec = _manager_replay_file_spec(pathlib.Path("a.h5"))
    assert current_spec.file_load_source is not None
    stale_spec = current_spec.model_copy(
        update={
            "file_load_source": current_spec.file_load_source.model_copy(
                update={"replay_call": None},
            )
        }
    )
    stale_peer = manager_provenance_edit._FileLoadBatchPeer(
        node=typing.cast(
            "typing.Any",
            types.SimpleNamespace(uid="stale", display_text="Stale"),
        ),
        scope="display",
        spec=stale_spec,
        original_path=pathlib.Path("b.h5"),
        loader_summary="xarray.load_dataarray",
    )
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    dialog = manager_provenance_edit._FileLoadEditDialog(
        current_spec.file_load_source,
        parent,
        batch_peers=(stale_peer,),
    )
    qtbot.addWidget(dialog)

    with pytest.raises(RuntimeError, match="no longer replayable"):
        dialog.peer_provenance_spec(stale_peer)


def test_file_load_edit_dialog_batch_disabled_without_peers(qtbot) -> None:
    spec = _manager_replay_file_spec(pathlib.Path("a.h5"))
    assert spec.file_load_source is not None
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    dialog = manager_provenance_edit._FileLoadEditDialog(
        spec.file_load_source,
        parent,
    )
    qtbot.addWidget(dialog)

    assert not dialog.batch_apply_check.isEnabled()
    assert not dialog.batch_note.isHidden()
    assert dialog.batch_peer_tree.topLevelItemCount() == 0


@pytest.mark.parametrize(
    "operation",
    [
        provenance.RotateOperation(
            angle=5.0,
            axes=("x", "y"),
            center=(0.0, 1.0),
        ),
        provenance.QSelOperation(kwargs={"x": 1.0}),
        provenance.IselOperation(kwargs={"x": 0}),
        provenance.SelOperation(kwargs={"x": 1.0}),
        provenance.SortByOperation(variables=("x",)),
        provenance.AverageOperation(dims=("x",)),
        provenance.QSelAggregationOperation(dims=("x",), func="mean"),
        provenance.InterpolationOperation(dim="x", values=[0.0, 1.0]),
        provenance.LeadingEdgeOperation(dim="x", fraction=0.5),
        provenance.CoarsenOperation(
            dim={"x": 2},
            boundary="trim",
            side="left",
            coord_func="mean",
            reducer="mean",
        ),
        provenance.ThinOperation(mode="global", factor=2),
        provenance.SymmetrizeOperation(dim="x", center=0.0),
        provenance.SymmetrizeNfoldOperation(
            fold=4,
            axes=("x", "y"),
            center={"x": 0.0, "y": 1.0},
        ),
        provenance.NormalizeOperation(dims=("x",), mode="area"),
        provenance.DivideByCoordOperation(coord_name="x"),
        provenance.GaussianFilterOperation(sigma={"x": 1.0}),
        provenance.SwapDimsOperation(mapping={"x": "kx"}),
        provenance.RenameDimsCoordsOperation(mapping={"x": "kx"}),
        provenance.AffineCoordOperation(coord_name="x", scale=1.0, offset=0.0),
        provenance.AssignCoordsOperation(coord_name="x", values=[0.0, 1.0]),
        provenance.AssignScalarCoordOperation(coord_name="temperature", value=20.0),
        provenance.AssignCoord1DOperation(
            coord_name="kx",
            dim="x",
            values=[0.0, 1.0],
        ),
        provenance.AssignAttrsOperation(attrs={"note": "edited"}),
    ],
)
def test_manager_provenance_structured_operations_have_edit_dialogs(
    operation: provenance.ToolProvenanceOperation,
) -> None:
    assert manager_provenance_edit._dialog_class_for_operation(operation) is not None


def test_manager_provenance_loader_kwargs_parser_and_file_dialog_branches(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    parse = manager_provenance_edit._parse_loader_kwargs

    assert parse("") == {}
    assert parse("(none)") == {}
    assert parse("{'engine': 'h5netcdf'}") == {"engine": "h5netcdf"}
    assert parse("engine='h5netcdf', chunks={'x': 1}") == {
        "engine": "h5netcdf",
        "chunks": {"x": 1},
    }
    with pytest.raises(TypeError, match="dictionary"):
        parse("{'not-a-dict'}")
    with pytest.raises(TypeError, match="string keys"):
        parse("{1: 'bad'}")
    with pytest.raises(TypeError, match="keyword arguments"):
        parse("'bad'")
    with pytest.raises(TypeError, match="unpacking"):
        parse("**kwargs")

    load_source = provenance.FileLoadSource(
        path=str(tmp_path / "old.h5"),
        loader_label="Load Function",
        loader_text="xarray.load_dataarray",
        kwargs_text="(none)",
        replay_call=None,
    )
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    dialog = manager_provenance_edit._FileLoadEditDialog(load_source, parent)
    qtbot.addWidget(dialog)
    assert dialog.loader_options.checked_filter()[2] == {"engine": "h5netcdf"}

    new_path = tmp_path / "new.h5"
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (str(new_path), ""),
    )
    dialog._browse()
    assert dialog.file_path() == new_path

    warnings: list[object] = []
    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        lambda *_args, **_kwargs: warnings.append(object()),
    )
    dialog.kwargs_edit.setText("'bad'")
    dialog.accept()
    assert len(warnings) == 1


def test_manager_provenance_dialog_class_lookup_skips_base_restore_hooks() -> None:
    class _NoRestoreTransformDialog(
        manager_provenance_edit.dialogs.DataTransformDialog
    ):
        operation_types = (provenance.ScriptCodeOperation,)

    class _NoRestoreFilterDialog(manager_provenance_edit.dialogs.DataFilterDialog):
        operation_types = (provenance.ScriptCodeOperation,)

    assert (
        manager_provenance_edit._dialog_class_for_operation(
            provenance.ScriptCodeOperation(label="script", code="derived = data")
        )
        is None
    )

    del _NoRestoreTransformDialog, _NoRestoreFilterDialog


def _fake_edit_controller(
    node: typing.Any | None = None,
    *,
    parent: typing.Any | None = None,
    nodes: dict[str, typing.Any] | None = None,
    metadata_uid: str | None = None,
) -> manager_provenance_edit._ProvenanceEditController:
    graph_nodes = (
        nodes if nodes is not None else ({} if node is None else {"node": node})
    )
    if metadata_uid is None:
        metadata_uid = "node" if node is not None else None

    def _parent_node(_node: typing.Any) -> typing.Any:
        if parent is None:
            raise RuntimeError("missing parent")
        return parent

    manager = types.SimpleNamespace(
        _metadata_node_uid=metadata_uid,
        _tool_graph=types.SimpleNamespace(nodes=graph_nodes),
        _parent_node=_parent_node,
        _script_input_can_reload=lambda *_args, **_kwargs: True,
        _rebuild_script_provenance=lambda spec, **_kwargs: types.SimpleNamespace(
            data=xr.DataArray([1.0], dims=("x",)),
            provenance_spec=spec,
        ),
        _update_info=lambda **_kwargs: None,
    )
    return manager_provenance_edit._ProvenanceEditController(
        typing.cast("typing.Any", manager)
    )


def _fake_edit_node(
    spec: provenance.ToolProvenanceSpec | None,
    *,
    uid: str = "node",
    display_text: str = "Node",
    source_spec: provenance.ToolProvenanceSpec | None = None,
    source_display_spec: provenance.ToolProvenanceSpec | None = None,
    parent_uid: str | None = None,
    active_filter: provenance.ToolProvenanceOperation | None = None,
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        uid=uid,
        display_text=display_text,
        is_imagetool=True,
        imagetool=object(),
        parent_uid=parent_uid,
        source_spec=source_spec,
        displayed_provenance_spec=spec,
        displayed_source_spec=source_display_spec,
        source_auto_update=True,
        slicer_area=types.SimpleNamespace(
            _accepted_filter_provenance_operation=active_filter
        ),
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
        source_spec=provenance.full_data(),
    )
    no_spec = _fake_edit_node(None, uid="no-spec")
    non_file = _fake_edit_node(provenance.full_data(), uid="non-file")
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

    assert [peer.node.uid for peer in peers] == ["matching"]
    assert peers[0].original_path == old_dir / "b.h5"

    assert (
        controller._file_load_batch_peers(
            typing.cast("typing.Any", current),
            provenance.full_data(),
        )
        == ()
    )


def test_manager_provenance_file_load_batch_helper_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = provenance.FileLoadSource(
        path="scan.h5",
        loader_label="Load Function",
        loader_text="xarray.load_dataarray",
        kwargs_text="",
        replay_call=None,
    )
    assert manager_provenance_edit._loader_summary(source) == "xarray.load_dataarray"
    assert (
        manager_provenance_edit._loader_summary(
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
        manager_provenance_edit._normalized_path(pathlib.Path("scan.h5"))
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
        manager_provenance_edit._replay_warning_details([empty_warning, useful_warning])
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
            provenance.ToolProvenanceSpec,
            dict[str, typing.Any],
        ]
    ] = []

    def _edit_file_load_spec(
        edit_node: object,
        scope: typing.Literal["display", "source"],
        spec: provenance.ToolProvenanceSpec,
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


def test_manager_provenance_edit_file_load_source_falls_back_to_source_spec(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    source_path = tmp_path / "source.h5"
    display_path = tmp_path / "display.h5"
    source_display_spec = _manager_replay_file_spec(source_path)
    node = _fake_edit_node(
        _manager_replay_file_spec(display_path),
        source_spec=provenance.full_data(),
        source_display_spec=source_display_spec,
        parent_uid="parent",
    )
    controller = _fake_edit_controller(node)
    calls: list[
        tuple[
            object,
            typing.Literal["display", "source"],
            provenance.ToolProvenanceSpec,
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
        source_spec=provenance.full_data(),
        source_display_spec=source_display_spec,
        parent_uid="parent",
    )
    controller = _fake_edit_controller(node)
    calls: list[
        tuple[
            object,
            typing.Literal["display", "source"],
            provenance.ToolProvenanceSpec,
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
        source_spec=provenance.full_data(),
        source_display_spec=provenance.full_data(),
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
            provenance.ToolProvenanceSpec,
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
        source_spec=provenance.full_data(),
        source_display_spec=provenance.full_data(),
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
    edit_ref = provenance._ProvenanceStepRef("operation", operation_index=0)
    replay_ref = provenance._ProvenanceStepRef("operation", operation_index=0)
    row = provenance._ProvenanceDisplayRow(
        provenance.DerivationEntry("row", None),
        edit_ref=edit_ref,
        replay_ref=replay_ref,
    )

    assert not _fake_edit_controller().can_edit_row(row)[0]
    assert not _fake_edit_controller().can_revert_row(row)[0]

    node = _fake_edit_node(provenance.full_data(provenance.IselOperation()))
    node.parent_uid = "parent"
    node.source_spec = provenance.selection()
    controller = _fake_edit_controller(node)
    assert not controller.can_edit_row(row)[0]
    assert not controller.can_revert_row(row)[0]

    node = _fake_edit_node(None)
    controller = _fake_edit_controller(node)
    assert not controller.can_edit_row(row)[0]
    assert not controller.can_revert_row(row)[0]

    file_row = provenance._ProvenanceDisplayRow(
        provenance.DerivationEntry("load", None),
        edit_ref=provenance._ProvenanceStepRef("file_load"),
        replay_ref=provenance._ProvenanceStepRef("file_load"),
    )
    controller = _fake_edit_controller(_fake_edit_node(provenance.full_data()))
    assert not controller.can_edit_row(file_row)[0]

    no_replay_file = _manager_provenance_file_spec(pathlib.Path("scan.h5")).model_copy(
        update={
            "file_load_source": provenance.FileLoadSource(
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

    missing_operation_row = provenance._ProvenanceDisplayRow(
        provenance.DerivationEntry("missing", None),
        edit_ref=provenance._ProvenanceStepRef("operation", operation_index=10),
        replay_ref=replay_ref,
    )
    controller = _fake_edit_controller(_fake_edit_node(provenance.full_data()))
    assert not controller.can_edit_row(missing_operation_row)[0]

    script_operation_spec = provenance.ToolProvenanceSpec(
        kind="full_data",
        operations=(
            provenance.ScriptCodeOperation(label="script", code="derived = data"),
        ),
    )
    script_row = provenance._ProvenanceDisplayRow(
        provenance.DerivationEntry("script", None),
        edit_ref=edit_ref,
        replay_ref=replay_ref,
    )
    controller = _fake_edit_controller(
        _fake_edit_node(script_operation_spec, parent_uid="parent")
    )
    assert not controller.can_edit_row(script_row)[0]

    script_with_structured_step = provenance.script(
        provenance.ScriptCodeOperation(
            label="Create derived data",
            code="derived = xr.DataArray([1.0, 2.0], dims=('x',))",
        ),
        provenance.AverageOperation(dims=("x",)),
        start_label="Run script",
        active_name="derived",
    )
    controller = _fake_edit_controller(_fake_edit_node(script_with_structured_step))
    script_code_row, structured_row = script_with_structured_step.display_rows()[1:]
    assert not controller.can_edit_row(script_code_row)[0]
    assert controller.can_edit_row(structured_row) == (True, "")

    source_row = provenance._ProvenanceDisplayRow(
        provenance.DerivationEntry("source", None),
        edit_ref=edit_ref,
        replay_ref=replay_ref,
        scope="source",
    )

    unsupported_spec = provenance.full_data(provenance.RestoreNonuniformDimsOperation())
    controller = _fake_edit_controller(
        _fake_edit_node(
            provenance.full_data(),
            source_display_spec=unsupported_spec,
            parent_uid="parent",
        )
    )
    assert not controller.can_edit_row(source_row)[0]

    live_row = provenance._ProvenanceDisplayRow(
        provenance.DerivationEntry("live", None),
        edit_ref=edit_ref,
        replay_ref=replay_ref,
    )
    controller = _fake_edit_controller(
        _fake_edit_node(provenance.selection(provenance.IselOperation()))
    )
    assert not controller.can_edit_row(live_row)[0]
    assert not controller.can_revert_row(live_row)[0]

    script_input_row = provenance._ProvenanceDisplayRow(
        provenance.DerivationEntry("input", None),
        replay_ref=provenance._ProvenanceStepRef("script_input", script_input_index=0),
    )
    assert not controller.can_revert_row(script_input_row)[0]

    earlier_source_row = provenance._ProvenanceDisplayRow(
        provenance.DerivationEntry("source", None),
        edit_ref=edit_ref,
        replay_ref=edit_ref,
        scope="source",
    )
    controller = _fake_edit_controller(
        _fake_edit_node(
            provenance.selection(provenance.IselOperation()),
            source_display_spec=provenance.selection(
                provenance.IselOperation(kwargs={"x": 0}),
                provenance.IselOperation(kwargs={"y": 0}),
            ),
            parent_uid="parent",
        )
    )
    assert controller.can_revert_row(earlier_source_row)[0]


def test_manager_provenance_revert_rejects_current_prefixes(
    tmp_path: pathlib.Path,
) -> None:
    invalid_operation_row = provenance._ProvenanceDisplayRow(
        provenance.DerivationEntry("invalid", None),
        replay_ref=provenance._ProvenanceStepRef("operation"),
    )
    controller = _fake_edit_controller(_fake_edit_node(provenance.full_data()))
    revertible, reason = controller.can_revert_row(invalid_operation_row)
    assert not revertible
    assert reason

    file_load_row = provenance._ProvenanceDisplayRow(
        provenance.DerivationEntry("load", None),
        replay_ref=provenance._ProvenanceStepRef("file_load"),
    )
    controller = _fake_edit_controller(
        _fake_edit_node(_manager_replay_file_spec(tmp_path / "scan.h5"))
    )
    revertible, reason = controller.can_revert_row(file_load_row)
    assert not revertible
    assert reason

    file_stage_spec = _manager_replay_file_spec(
        tmp_path / "scan.h5",
        provenance.IselOperation(kwargs={"x": 0}),
        provenance.IselOperation(kwargs={"y": 0}),
    )
    earlier_stage_row = provenance._ProvenanceDisplayRow(
        provenance.DerivationEntry("isel", None),
        replay_ref=provenance._ProvenanceStepRef(
            "operation",
            operation_index=0,
            stage_index=0,
        ),
    )
    latest_stage_row = provenance._ProvenanceDisplayRow(
        provenance.DerivationEntry("isel", None),
        replay_ref=provenance._ProvenanceStepRef(
            "operation",
            operation_index=1,
            stage_index=0,
        ),
    )
    controller = _fake_edit_controller(_fake_edit_node(file_stage_spec))
    assert controller.can_revert_row(earlier_stage_row)[0]
    revertible, reason = controller.can_revert_row(latest_stage_row)
    assert not revertible
    assert reason

    script_spec = provenance.script(
        provenance.ScriptCodeOperation(
            label="Run script",
            code="derived = xr.DataArray([1.0], dims=('x',))",
        ),
        provenance.IselOperation(kwargs={"x": 0}),
        start_label="Run script",
        active_name="derived",
    )
    controller = _fake_edit_controller(_fake_edit_node(script_spec))
    revertible, reason = controller.can_revert_row(invalid_operation_row)
    assert not revertible
    assert reason

    latest_script_row = provenance._ProvenanceDisplayRow(
        provenance.DerivationEntry("isel", None),
        replay_ref=provenance._ProvenanceStepRef(
            "operation",
            operation_index=1,
        ),
    )
    revertible, reason = controller.can_revert_row(latest_script_row)
    assert not revertible
    assert reason

    source_latest_row = provenance._ProvenanceDisplayRow(
        provenance.DerivationEntry("source", None),
        replay_ref=provenance._ProvenanceStepRef("operation", operation_index=1),
        scope="source",
    )
    source_spec = provenance.selection(
        provenance.IselOperation(kwargs={"x": 0}),
        provenance.IselOperation(kwargs={"y": 0}),
    )
    controller = _fake_edit_controller(
        _fake_edit_node(
            provenance.full_data(),
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
    node = _fake_edit_node(provenance.full_data())
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

    row = provenance._ProvenanceDisplayRow(
        provenance.DerivationEntry("row", None),
        replay_ref=provenance._ProvenanceStepRef("operation", operation_index=0),
    )
    monkeypatch.setattr(controller, "can_revert_row", lambda _row: (True, ""))
    monkeypatch.setattr(controller, "_confirm_revert", lambda: True)
    monkeypatch.setattr(controller, "_display_spec_for_row", lambda _node, _row: None)
    controller.revert_row(row)
    assert len(failed) == 1

    monkeypatch.setattr(
        controller,
        "_display_spec_for_row",
        lambda _node, _row: provenance.full_data(),
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
        manager_provenance_edit._FileLoadBatchPeer(
            node=typing.cast("typing.Any", valid),
            scope="display",
            spec=valid_spec,
            original_path=tmp_path / "b.h5",
            loader_summary="xarray.load_dataarray",
        ),
        manager_provenance_edit._FileLoadBatchPeer(
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
            _load_source: provenance.FileLoadSource,
            _parent: QtWidgets.QWidget,
            *,
            batch_peers: tuple[manager_provenance_edit._FileLoadBatchPeer, ...],
        ) -> None:
            assert batch_peers == peers

        def exec(self) -> int:
            return int(QtWidgets.QDialog.DialogCode.Accepted)

        def provenance_spec(
            self,
            *,
            active_name: str,
            replay_stages: tuple[provenance.ReplayStage, ...],
        ) -> provenance.ToolProvenanceSpec:
            del active_name, replay_stages
            return current_spec

        def selected_batch_peers(
            self,
        ) -> tuple[manager_provenance_edit._FileLoadBatchPeer, ...]:
            return peers

        def peer_provenance_spec(
            self,
            peer: manager_provenance_edit._FileLoadBatchPeer,
        ) -> provenance.ToolProvenanceSpec:
            return peer.spec

    monkeypatch.setattr(manager_provenance_edit, "_FileLoadEditDialog", _Dialog)
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
        candidate: provenance.ToolProvenanceSpec,
        *,
        where: str,
    ) -> manager_provenance_edit._ValidatedProvenanceEdit:
        del where
        if node.uid == "failed":
            raise RuntimeError("peer failed")
        return manager_provenance_edit._ValidatedProvenanceEdit(
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
    row = provenance._ProvenanceDisplayRow(
        provenance.DerivationEntry("load", None),
        edit_ref=provenance._ProvenanceStepRef("file_load"),
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
    batch_peer = manager_provenance_edit._FileLoadBatchPeer(
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
            replay_stages: tuple[provenance.ReplayStage, ...],
        ) -> provenance.ToolProvenanceSpec:
            del active_name, replay_stages
            return current_spec

        def selected_batch_peers(
            self,
        ) -> tuple[manager_provenance_edit._FileLoadBatchPeer, ...]:
            return (batch_peer,)

        def peer_provenance_spec(
            self,
            peer: manager_provenance_edit._FileLoadBatchPeer,
        ) -> provenance.ToolProvenanceSpec:
            return peer.spec

    monkeypatch.setattr(manager_provenance_edit, "_FileLoadEditDialog", _Dialog)
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
    row = provenance._ProvenanceDisplayRow(
        provenance.DerivationEntry("load", None),
        edit_ref=provenance._ProvenanceStepRef("file_load"),
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
    controller = _fake_edit_controller(_fake_edit_node(provenance.full_data()))
    peer_spec = _manager_replay_file_spec(tmp_path / "b.h5")
    peer_node = _fake_edit_node(peer_spec, uid="peer", display_text="Peer")
    peer = manager_provenance_edit._FileLoadBatchPeer(
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
    controller = _fake_edit_controller(_fake_edit_node(provenance.full_data()))
    peer_spec = _manager_replay_file_spec(tmp_path / "b.h5")
    peer_node = _fake_edit_node(peer_spec, uid="peer", display_text="Peer")
    peer = manager_provenance_edit._FileLoadBatchPeer(
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
    controller = _fake_edit_controller(_fake_edit_node(provenance.full_data()))
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
    failure = manager_provenance_edit._ProvenanceReplayFailure(
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
def test_manager_provenance_missing_source_edit_opens_file_load_editor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    dialog_result: int,
    expected_opened: int,
) -> None:
    missing_path = tmp_path / "missing.h5"
    peer_path = tmp_path / "peer.h5"
    spec = _manager_replay_file_spec(
        missing_path,
        provenance.IselOperation(kwargs={"x": 0}),
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
    row = provenance._ProvenanceDisplayRow(
        provenance.DerivationEntry("isel", None),
        edit_ref=provenance._ProvenanceStepRef(
            "operation",
            operation_index=0,
            stage_index=0,
        ),
        replay_ref=provenance._ProvenanceStepRef(
            "operation",
            operation_index=0,
            stage_index=0,
        ),
    )
    dialogs: list[dict[str, typing.Any]] = []
    opened: list[
        tuple[
            typing.Any,
            str,
            provenance.ToolProvenanceSpec,
            tuple[manager_provenance_edit._FileLoadBatchPeer, ...],
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
        "_edit_file_load_spec",
        lambda node_arg, scope, spec_arg, **kwargs: opened.append(
            (node_arg, scope, spec_arg, kwargs["batch_peers"])
        ),
    )

    controller.edit_row(row)

    assert len(dialogs) == 1
    assert dialogs[0]["parent"] is controller._manager
    assert "source file" in dialogs[0]["text"].lower()
    assert (
        "preparing data before the selected provenance step"
        in (dialogs[0]["informative_text"])
    )
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
        provenance.IselOperation(kwargs={"x": 0}),
    )
    node = _fake_edit_node(spec)
    controller = _fake_edit_controller(node)
    row = provenance._ProvenanceDisplayRow(
        provenance.DerivationEntry("load", None),
        replay_ref=provenance._ProvenanceStepRef("file_load"),
    )
    opened: list[provenance.ToolProvenanceSpec] = []

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
    assert opened[0].replay_stages == ()


def test_manager_provenance_missing_source_without_file_load_shows_dedicated_dialog(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    controller = _fake_edit_controller(_fake_edit_node(provenance.full_data()))
    missing = manager_provenance_edit._MissingProvenanceSourceFileError(
        tmp_path / "missing.h5",
    )
    exc = manager_provenance_edit._ProvenanceReplayFailure("repairing source", missing)
    exc.__cause__ = missing
    row = provenance._ProvenanceDisplayRow(provenance.DerivationEntry("row", None))
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
        typing.cast("typing.Any", _fake_edit_node(provenance.full_data())),
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
    controller = _fake_edit_controller(_fake_edit_node(provenance.full_data()))
    missing = manager_provenance_edit._MissingProvenanceSourceFileError(
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
    missing = manager_provenance_edit._MissingProvenanceSourceFileError(
        tmp_path / "missing.h5",
    )
    exc = manager_provenance_edit._ProvenanceReplayFailure("replaying file", missing)
    exc.__cause__ = missing
    row = provenance._ProvenanceDisplayRow(provenance.DerivationEntry("row", None))
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
        replacement_missing = manager_provenance_edit._MissingProvenanceSourceFileError(
            tmp_path / "still-missing.h5",
        )
        repair_exc = manager_provenance_edit._ProvenanceReplayFailure(
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
    controller = _fake_edit_controller(_fake_edit_node(provenance.full_data()))
    spec = _manager_provenance_file_spec(tmp_path / "missing.h5")

    monkeypatch.setattr(
        manager_provenance_edit.provenance,
        "replay_file_provenance",
        lambda _spec: pytest.fail("missing files should fail before loader replay"),
    )

    with pytest.raises(FileNotFoundError, match="no longer accessible"):
        controller._replay_file_candidate(spec)
    with pytest.raises(
        manager_provenance_edit._MissingProvenanceSourceFileError
    ) as exc:
        controller._replay_file_candidate(spec)
    assert exc.value.source_path == tmp_path / "missing.h5"


def test_manager_provenance_file_replay_validation_captures_loader_warnings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    file_path = tmp_path / "scan.h5"
    file_path.touch()
    controller = _fake_edit_controller(_fake_edit_node(provenance.full_data()))
    spec = _manager_provenance_file_spec(file_path)

    def _warn_then_fail(_spec: provenance.ToolProvenanceSpec) -> xr.DataArray:
        warnings.warn(
            "Loading f_003_S001 with inferred index 3 resulted in an error.",
            UserWarning,
            stacklevel=1,
        )
        raise RuntimeError("real replay failure")

    monkeypatch.setattr(
        manager_provenance_edit.provenance,
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
    file_row = provenance._ProvenanceDisplayRow(
        provenance.DerivationEntry("load", None),
        edit_ref=provenance._ProvenanceStepRef("file_load"),
    )
    with pytest.raises(RuntimeError, match="file load"):
        controller._edit_file_load_row(
            typing.cast("typing.Any", _fake_edit_node(None)), file_row
        )
    with pytest.raises(RuntimeError, match="file load"):
        controller._edit_file_load_spec(
            typing.cast("typing.Any", _fake_edit_node(provenance.full_data())),
            "display",
            provenance.full_data(),
            where="testing",
        )

    operation_row = provenance._ProvenanceDisplayRow(
        provenance.DerivationEntry("op", None),
        edit_ref=provenance._ProvenanceStepRef("operation", operation_index=0),
    )
    with pytest.raises(RuntimeError, match="No provenance"):
        controller._edit_operation_row(
            typing.cast("typing.Any", _fake_edit_node(None)), operation_row
        )
    with pytest.raises(RuntimeError, match="not available"):
        controller._edit_operation_row(
            typing.cast("typing.Any", _fake_edit_node(provenance.full_data())),
            operation_row,
        )
    with pytest.raises(RuntimeError, match="No editing dialog"):
        controller._edit_operation_row(
            typing.cast(
                "typing.Any",
                _fake_edit_node(
                    provenance.full_data(provenance.RestoreNonuniformDimsOperation())
                ),
            ),
            operation_row,
        )


def test_manager_provenance_edit_controller_dialog_execution_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _fake_edit_controller()
    data = xr.DataArray(np.arange(6, dtype=float).reshape((2, 3)), dims=("x", "y"))
    operation = provenance.NormalizeOperation(dims=("x",), mode="area")
    original_itool = erlab.interactive.itool

    monkeypatch.setattr(erlab.interactive, "itool", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="temporary ImageTool"):
        controller._edited_operations_from_dialog(
            manager_provenance_edit.dialogs.NormalizeDialog,
            operation,
            data,
        )

    monkeypatch.setattr(erlab.interactive, "itool", original_itool)
    monkeypatch.setattr(
        manager_provenance_edit.dialogs.NormalizeDialog,
        "exec",
        lambda self: int(QtWidgets.QDialog.DialogCode.Rejected),
    )
    assert (
        controller._edited_operations_from_dialog(
            manager_provenance_edit.dialogs.NormalizeDialog,
            operation,
            data,
        )
        is None
    )

    monkeypatch.setattr(
        manager_provenance_edit.dialogs.NormalizeDialog,
        "exec",
        lambda self: int(QtWidgets.QDialog.DialogCode.Accepted),
    )
    assert controller._edited_operations_from_dialog(
        manager_provenance_edit.dialogs.NormalizeDialog,
        operation,
        data,
    ) == [operation]

    monkeypatch.setattr(
        manager_provenance_edit.dialogs.NormalizeDialog,
        "filter_operation",
        lambda self: None,
    )
    assert (
        controller._edited_operations_from_dialog(
            manager_provenance_edit.dialogs.NormalizeDialog,
            operation,
            data,
        )
        == []
    )

    with pytest.raises(RuntimeError, match="Active display filter"):
        controller._edit_active_filter(
            typing.cast("typing.Any", _fake_edit_node(provenance.full_data())),
            operation,
            manager_provenance_edit.dialogs.AggregateDialog,
        )


def test_manager_provenance_edit_controller_skips_deleted_temp_tool_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _fake_edit_controller()
    data = xr.DataArray(np.arange(6, dtype=float).reshape((2, 3)), dims=("x", "y"))
    operation = provenance.NormalizeOperation(dims=("x",), mode="area")
    original_itool = erlab.interactive.itool
    original_qt_is_valid = erlab.interactive.utils.qt_is_valid
    created_tools: list[QtWidgets.QWidget] = []

    def recording_itool(*args, **kwargs):
        tool = original_itool(*args, **kwargs)
        created_tools.append(tool)
        return tool

    def fake_qt_is_valid(*objects: object) -> bool:
        if created_tools and any(obj is created_tools[-1] for obj in objects):
            return False
        return original_qt_is_valid(*objects)

    monkeypatch.setattr(
        manager_provenance_edit.dialogs.NormalizeDialog,
        "exec",
        lambda self: int(QtWidgets.QDialog.DialogCode.Rejected),
    )
    with monkeypatch.context() as context:
        context.setattr(erlab.interactive, "itool", recording_itool)
        context.setattr(erlab.interactive.utils, "qt_is_valid", fake_qt_is_valid)
        assert (
            controller._edited_operations_from_dialog(
                manager_provenance_edit.dialogs.NormalizeDialog,
                operation,
                data,
            )
            is None
        )

    for tool in created_tools:
        if original_qt_is_valid(tool):
            tool.close()
            tool.deleteLater()


def test_manager_provenance_validation_reports_active_filter_replay_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _fake_edit_controller(_fake_edit_node(provenance.full_data()))
    data = xr.DataArray([1.0], dims=("x",))
    candidate = provenance.full_data()
    base_candidate = provenance.full_data(provenance.IselOperation(kwargs={"x": 0}))
    calls = 0

    def _replay_candidate_result(
        _node: typing.Any,
        _scope: typing.Literal["display", "source"],
        spec: provenance.ToolProvenanceSpec,
    ) -> tuple[xr.DataArray, provenance.ToolProvenanceSpec]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return data, spec
        raise RuntimeError("base replay failed")

    monkeypatch.setattr(
        controller,
        "_replay_candidate_result",
        _replay_candidate_result,
    )
    monkeypatch.setattr(
        controller,
        "_split_active_filter",
        lambda _node, _spec: (
            base_candidate,
            provenance.NormalizeOperation(dims=("x",), mode="area"),
        ),
    )

    with pytest.raises(manager_provenance_edit._ProvenanceReplayFailure) as exc:
        controller._validated_edit(
            typing.cast("typing.Any", _fake_edit_node(provenance.full_data())),
            "display",
            candidate,
            where="validating edited filter",
        )

    assert "active display filter" in str(exc.value)


def test_manager_provenance_edit_controller_live_replay_and_replace() -> None:
    parent_data = xr.DataArray(
        np.arange(6, dtype=float).reshape((2, 3)), dims=("x", "y")
    )
    parent = types.SimpleNamespace(
        displayed_provenance_spec=provenance.full_data(),
        current_source_data=lambda: parent_data,
    )
    node = _fake_edit_node(
        provenance.full_data(),
        parent_uid="parent",
        source_display_spec=provenance.selection(),
    )
    replaced: list[tuple[xr.DataArray, provenance.ToolProvenanceSpec, bool, bool]] = []
    source_bindings: list[
        tuple[
            provenance.ToolProvenanceSpec,
            bool,
            str,
            provenance.ToolProvenanceSpec,
        ]
    ] = []
    node._replace_imagetool_data = (
        lambda data, spec, *, propagate_descendants, preserve_filter: replaced.append(
            (data, spec, propagate_descendants, preserve_filter)
        )
    )
    node.set_source_binding = lambda spec, *, auto_update, state, provenance_spec: (
        source_bindings.append((spec, auto_update, state, provenance_spec))
    )
    controller = _fake_edit_controller(node, parent=parent)
    spec = provenance.selection(provenance.IselOperation(kwargs={"x": 1}))

    data, replayed_spec = controller._replay_candidate_result(
        typing.cast("typing.Any", node),
        "source",
        spec,
    )
    xr.testing.assert_identical(data, parent_data.isel(x=1))
    assert replayed_spec == spec

    controller._replace_node_data(
        typing.cast("typing.Any", node),
        "source",
        data,
        spec,
        provenance.NormalizeOperation(dims=("y",), mode="area"),
    )

    assert source_bindings
    assert source_bindings[-1][0] == spec
    assert source_bindings[-1][1:] == (
        True,
        "fresh",
        provenance.compose_display_provenance(
            parent.displayed_provenance_spec,
            spec,
            parent_data=parent_data,
        ),
    )
    assert replaced[-1][2:] == (True, True)

    with pytest.raises(RuntimeError, match="Live provenance"):
        controller._replay_candidate_result(
            typing.cast("typing.Any", node),
            "display",
            spec,
        )
    with pytest.raises(RuntimeError, match="Source-bound edits"):
        controller._replace_node_data(
            typing.cast("typing.Any", node),
            "source",
            data,
            typing.cast("typing.Any", None),
            None,
        )

    unsupported = spec.model_copy(update={"kind": "bad"})
    with pytest.raises(RuntimeError, match="Unsupported provenance kind"):
        controller._replay_candidate_result(
            typing.cast("typing.Any", node),
            "display",
            unsupported,
        )


def test_manager_provenance_edit_controller_active_filter_refs_and_split() -> None:
    controller = _fake_edit_controller()
    active = provenance.NormalizeOperation(dims=("x",), mode="area")
    node = _fake_edit_node(provenance.full_data(), active_filter=active)

    assert (
        controller._active_filter_ref(
            typing.cast("typing.Any", _fake_edit_node(provenance.full_data())),
            provenance.full_data(active),
        )
        is None
    )

    live_spec = provenance.full_data(provenance.IselOperation(kwargs={"x": 0}), active)
    assert controller._active_filter_ref(
        typing.cast("typing.Any", node),
        live_spec,
    ) == provenance._ProvenanceStepRef("operation", operation_index=1)

    file_spec = _manager_provenance_file_spec(
        pathlib.Path("scan.h5")
    ).append_replay_stage(provenance.full_data(active))
    assert controller._active_filter_ref(
        typing.cast("typing.Any", node),
        file_spec,
    ) == provenance._ProvenanceStepRef(
        "operation",
        operation_index=0,
        stage_index=0,
    )
    base_spec, split_operation = controller._split_active_filter(
        typing.cast("typing.Any", node),
        file_spec,
    )
    assert split_operation == active
    assert base_spec.replay_stages == ()

    node.slicer_area._accepted_filter_provenance_operation = None
    assert controller._split_active_filter(
        typing.cast("typing.Any", node),
        live_spec,
    ) == (live_spec, None)


def test_tool_provenance_spec_row_reference_helpers_cover_edge_branches() -> None:
    isel = provenance.IselOperation(kwargs={"x": 0})
    sel = provenance.SelOperation(kwargs={"y": 1.0})
    spec = provenance.selection(isel, sel)
    start_ref = provenance._ProvenanceStepRef("start")
    first_ref = provenance._ProvenanceStepRef("operation", operation_index=0)
    missing_ref = provenance._ProvenanceStepRef("operation", operation_index=20)
    script_input_ref = provenance._ProvenanceStepRef(
        "script_input",
        script_input_index=0,
    )

    assert spec._operation_for_ref(start_ref) is None
    assert spec._operation_for_ref(first_ref) == isel
    assert spec._operation_for_ref(missing_ref) is None
    with pytest.raises(ValueError, match="Expected an operation"):
        spec._replace_operation_ref(start_ref, ())
    assert spec._replace_operation_ref(first_ref, (sel,)).operations == (sel, sel)
    assert spec._prefix_through_ref(start_ref).operations == ()
    with pytest.raises(ValueError, match="cannot be replayed"):
        spec._prefix_through_ref(script_input_ref)
    assert spec._prefix_before_ref(start_ref).operations == ()
    assert spec._prefix_before_ref(first_ref).operations == ()
    assert spec._streamlined_operations("selection", spec.operations) == spec.operations

    file_spec = _manager_provenance_file_spec(
        pathlib.Path("scan.h5")
    ).append_replay_stage(provenance.full_data(isel, sel))
    stage_ref = provenance._ProvenanceStepRef(
        "operation",
        operation_index=1,
        stage_index=0,
    )
    assert file_spec._operation_for_ref(stage_ref) == sel
    assert (
        file_spec._operation_for_ref(
            provenance._ProvenanceStepRef("operation", operation_index=1, stage_index=2)
        )
        is None
    )
    assert (
        file_spec._prefix_through_ref(
            provenance._ProvenanceStepRef("file_load")
        ).replay_stages
        == ()
    )
    assert file_spec._prefix_before_ref(stage_ref).replay_stages[0].operations == (
        isel,
    )


def test_elided_value_label_keeps_full_text_during_resize(qtbot) -> None:
    class _FallbackStyleLabel(manager_widgets._ElidedValueLabel):
        def style(self) -> QtWidgets.QStyle | None:
            return None

    label = _FallbackStyleLabel("/very/long/path/to/data/scan_with_long_name.h5")
    qtbot.addWidget(label)
    label.setMargin(2)
    label.setIndent(4)
    label.resize(90, label.sizeHint().height())
    label.show()

    assert label.text() == label.full_text
    assert label.toolTip() == label.full_text
    assert label.sizeHint().width() > label.minimumSizeHint().width()
    assert label.grab().isNull() is False
    assert label.text() == label.full_text

    label.setText(None)
    assert label.text() == ""
    assert label.full_text == ""
    assert not hasattr(label, "clicked")
    assert label.cursor().shape() != QtCore.Qt.CursorShape.PointingHandCursor
    assert label.textInteractionFlags() == (
        QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
    )


def test_manager_metadata_added_label_does_not_force_splitter_width(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    long_time = "2024-01-02 03:04:05 Pacific Daylight Time (-0700)"

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        manager._set_metadata_fields(
            [
                manager_wrapper._MetadataField(
                    "Added",
                    long_time,
                    monospace=True,
                )
            ]
        )
        manager._update_metadata_pane()

        label = manager._metadata_detail_labels["Added"]
        assert isinstance(label, manager_widgets._ElidedValueLabel)
        assert label.text() == long_time
        assert label.full_text == long_time
        assert label.toolTip() == long_time
        assert label.textInteractionFlags() == (
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        assert (
            label.sizePolicy().horizontalPolicy()
            == QtWidgets.QSizePolicy.Policy.Ignored
        )
        key_label = typing.cast(
            "QtWidgets.QLabel",
            manager.metadata_details_layout.itemAtPosition(0, 0).widget(),
        )
        assert key_label.alignment() == (
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        assert label.alignment() == (
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        assert label.sizeHint().width() < label.fontMetrics().horizontalAdvance(
            long_time
        )
        assert manager.metadata_details_widget.minimumSizeHint().width() < (
            label.fontMetrics().horizontalAdvance(long_time)
        )


def test_manager_file_label_helpers_and_file_replay_rename_update(tmp_path) -> None:
    paths = [
        tmp_path / "scan_a.h5",
        tmp_path / "scan_b.h5",
        tmp_path / "scan_c.h5",
    ]

    assert manager_wrapper._compact_file_suffix(paths) == " (scan_a, scan_b, +1)"

    spec = _manager_provenance_file_spec(paths[0]).append_replay_stage(
        provenance.full_data(
            provenance.AverageOperation(dims=("x",))
        ).append_final_rename("old")
    )
    renamed = manager_wrapper._spec_with_final_data_name(spec, "new")

    assert renamed.kind == "file"
    assert renamed.replay_stages
    assert renamed.replay_stages[-1].operations[-1] == provenance.RenameOperation(
        name="new"
    )
    assert renamed.replay_stages[-1].operations[:-1] == (
        provenance.AverageOperation(dims=("x",)),
    )


def test_manager_childtool_from_filtered_parent_uses_display_provenance(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["alpha", "eV"],
        coords={"alpha": np.arange(5, dtype=float), "eV": np.arange(5, dtype=float)},
    )
    operation = provenance.GaussianFilterOperation(sigma={"alpha": 1.0})
    expected = operation.apply(data, parent_data=data)

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.apply_filter_operation(operation)
        parent_tool.slicer_area.open_in_meshtool()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[0]._childtools) == 1,
            timeout=5000,
        )

        child_uid = manager._tool_graph.root_wrappers[0]._childtool_indices[0]
        child = manager.get_childtool(child_uid)
        assert child.input_provenance_spec is not None
        display_code = child.input_provenance_spec.display_code()
        assert display_code is not None
        assert "gaussian_filter" in display_code
        namespace = {"data": data.copy(deep=True)}
        exec(  # noqa: S102
            display_code,
            {"np": np, "xr": xr, "erlab": erlab, "era": erlab.analysis},
            namespace,
        )
        xr.testing.assert_identical(namespace["derived"], expected)


def test_manager_filtered_parent_updates_source_bound_child(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["alpha", "eV"],
        coords={"alpha": np.arange(5, dtype=float), "eV": np.arange(5, dtype=float)},
    )
    operation = provenance.GaussianFilterOperation(sigma={"alpha": 1.0})
    expected = operation.apply(data, parent_data=data)

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root_tool = itool(data, manager=False, execute=False)
        assert isinstance(root_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root_tool, show=False)

        child_tool = itool(data.copy(deep=False), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=provenance.full_data(),
            source_auto_update=True,
        )
        child_node = manager._child_node(child_uid)

        root_tool.slicer_area.apply_filter_operation(operation, emit_edited=True)

        qtbot.wait_until(
            lambda: (
                child_node.source_state == "fresh"
                and fetch(child_uid).identical(expected)
            ),
            timeout=5000,
        )
        xr.testing.assert_identical(fetch(child_uid), expected)
        assert child_node.provenance_spec is not None
        code = child_node.provenance_spec.display_code()
        assert code is not None
        namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
        xr.testing.assert_identical(namespace["derived"], expected)


def test_manager_filtered_source_bound_child_refresh_keeps_filter(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["alpha", "eV"],
        coords={"alpha": np.arange(5, dtype=float), "eV": np.arange(5, dtype=float)},
    )
    updated = data + 100.0
    operation = provenance.GaussianFilterOperation(sigma={"alpha": 1.0})
    expected = operation.apply(updated, parent_data=updated)

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root_tool = itool(data, manager=False, execute=False)
        assert isinstance(root_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root_tool, show=False)

        child_tool = itool(data.copy(deep=False), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=provenance.full_data(),
            source_auto_update=True,
        )
        child_node = manager._child_node(child_uid)
        child_tool.slicer_area.apply_filter_operation(operation, emit_edited=True)

        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated)

        qtbot.wait_until(
            lambda: (
                child_node.source_state == "fresh"
                and fetch(child_uid).identical(expected)
            ),
            timeout=5000,
        )
        xr.testing.assert_identical(fetch(child_uid), expected)
        display_spec = child_node.displayed_provenance_spec
        assert display_spec is not None
        display_code = display_spec.display_code()
        assert display_code is not None
        assert "gaussian_filter" in display_code
        namespace = _exec_generated_code(
            display_code, {"data": updated.copy(deep=True)}
        )
        xr.testing.assert_identical(namespace["derived"], expected)


def test_manager_filtered_source_bound_child_failed_refresh_keeps_filter(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["x", "y"],
        coords={"x": np.arange(5, dtype=float), "y": np.arange(5, dtype=float)},
    )
    bad_update = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["u", "y"],
        coords={"u": np.arange(5, dtype=float), "y": np.arange(5, dtype=float)},
    )
    operation = provenance.GaussianFilterOperation(sigma={"x": 1.0})
    expected = operation.apply(data, parent_data=data)

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root_tool = itool(data, manager=False, execute=False)
        assert isinstance(root_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root_tool, show=False)

        child_tool = itool(data.copy(deep=False), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=provenance.full_data(),
            source_auto_update=True,
        )
        child_node = manager._child_node(child_uid)
        child_tool.slicer_area.apply_filter_operation(operation, emit_edited=True)

        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, bad_update)

        qtbot.wait_until(
            lambda: child_node.source_state == "unavailable",
            timeout=5000,
        )
        xr.testing.assert_identical(fetch(child_uid), expected)
        assert child_tool.slicer_area._accepted_filter_provenance_operation == operation


def test_manager_duplicate_filtered_child_records_filter_once(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["alpha", "eV"],
        coords={"alpha": np.arange(5, dtype=float), "eV": np.arange(5, dtype=float)},
    )
    operation = provenance.GaussianFilterOperation(sigma={"alpha": 1.0})
    expected = operation.apply(data, parent_data=data)

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root_tool = itool(data, manager=False, execute=False)
        assert isinstance(root_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root_tool, show=False)

        child_tool = itool(data.copy(deep=False), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=provenance.full_data(),
            source_auto_update=True,
        )
        child_tool.slicer_area.apply_filter_operation(operation, emit_edited=True)

        duplicated_uid = manager.duplicate_childtool(child_uid)
        duplicated_node = manager._child_node(duplicated_uid)
        duplicated_tool = manager.get_imagetool(duplicated_uid)

        assert duplicated_node.source_spec is not None
        assert [op.op for op in duplicated_node.source_spec.operations] == []
        displayed_source = duplicated_node.displayed_source_spec
        assert displayed_source is not None
        assert [op.op for op in displayed_source.operations] == ["gaussian_filter"]
        display_code = duplicated_node.displayed_provenance_spec.display_code()
        assert display_code is not None
        assert display_code.count("gaussian_filter") == 1
        xr.testing.assert_identical(duplicated_tool.slicer_area.data, expected)


def test_manager_workspace_roundtrip_filtered_child_records_filter_once(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["alpha", "eV"],
        coords={"alpha": np.arange(5, dtype=float), "eV": np.arange(5, dtype=float)},
    )
    operation = provenance.GaussianFilterOperation(sigma={"alpha": 1.0})

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root_tool = itool(data, manager=False, execute=False)
        assert isinstance(root_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root_tool, show=False)

        child_tool = itool(data.copy(deep=False), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=provenance.full_data(),
            source_auto_update=True,
        )
        child_tool.slicer_area.apply_filter_operation(operation, emit_edited=True)

        tree = manager._to_datatree()
        saved = typing.cast(
            "xr.DataTree", tree[f"0/childtools/{child_uid}/imagetool"]
        ).to_dataset(inherit=False)
        state = json.loads(saved.attrs["itool_state"])
        assert state["filter_operation"]["op"] == "gaussian_filter"
        source_payload = json.loads(saved.attrs["manager_node_live_source_spec"])
        assert source_payload["operations"] == []

        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)
        for node in tree.values():
            manager._load_workspace_node(typing.cast("xr.DataTree", node))
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        loaded_node = manager._child_node(child_uid)
        displayed_source = loaded_node.displayed_source_spec
        assert displayed_source is not None
        assert [op.op for op in displayed_source.operations] == ["gaussian_filter"]

        updated = data + 10.0
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated)
        expected = operation.apply(updated, parent_data=updated)
        qtbot.wait_until(
            lambda: fetch(child_uid).identical(expected),
            timeout=5000,
        )
        xr.testing.assert_identical(fetch(child_uid), expected)


def test_manager_operation_filter_preserves_output_binding(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:

    class _OutputToolState(pydantic.BaseModel):
        pass

    class _OutputTool(erlab.interactive.utils.ToolWindow[_OutputToolState]):
        StateModel = _OutputToolState
        tool_name = "output-dummy"

        def __init__(self, data: xr.DataArray) -> None:
            super().__init__()
            self._data = data
            self._status = _OutputToolState()

        @property
        def tool_status(self) -> _OutputToolState:
            return self._status

        @tool_status.setter
        def tool_status(self, status: _OutputToolState) -> None:
            self._status = status

        @property
        def tool_data(self) -> xr.DataArray:
            return self._data

        def output_imagetool_data(
            self, output_id: str | enum.Enum
        ) -> xr.DataArray | None:
            assert output_id == "out"
            return self._data + 10.0

        def output_imagetool_provenance(
            self, output_id: str | enum.Enum, data: xr.DataArray
        ) -> provenance.ToolProvenanceSpec | None:
            assert output_id == "out"
            del data
            return provenance.script(
                provenance.ScriptCodeOperation(
                    label="Use output", code="result = data + 10"
                ),
                start_label="Start from parent",
                active_name="result",
            )

    data = xr.DataArray(
        np.arange(12, dtype=float).reshape((3, 4)),
        dims=["x", "y"],
        coords={"x": np.arange(3), "y": np.arange(4)},
        name="scan",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root_tool = itool(data, manager=False, execute=False)
        assert isinstance(root_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root_tool, show=False)

        child = _OutputTool(data)
        child_uid = manager.add_childtool(child, 0, show=False)
        child_node = manager._child_node(child_uid)
        assert child_node.displayed_source_spec == child_node.source_spec
        initial_output = typing.cast("xr.DataArray", child.output_imagetool_data("out"))
        output_tool = itool(initial_output, manager=False, execute=False)
        assert isinstance(output_tool, erlab.interactive.imagetool.ImageTool)
        output_uid = manager.add_imagetool_child(
            output_tool,
            child_uid,
            show=False,
            provenance_spec=child.output_imagetool_provenance("out", initial_output),
            source_state="fresh",
            output_id="out",
        )
        operation = provenance.GaussianFilterOperation(sigma={"x": 1.0})
        output_tool.slicer_area.apply_filter_operation(operation, emit_edited=True)
        expected = operation.apply(initial_output, parent_data=initial_output)

        duplicated_uid = manager.duplicate_childtool(output_uid)
        duplicated_node = manager._child_node(duplicated_uid)
        assert duplicated_node.output_id == "out"
        assert duplicated_node.source_spec is None
        xr.testing.assert_identical(fetch(duplicated_uid), expected)

        tree = manager._to_datatree()
        saved = typing.cast(
            "xr.DataTree",
            tree[f"0/childtools/{child_uid}/childtools/{output_uid}/imagetool"],
        ).to_dataset(inherit=False)
        assert saved.attrs["manager_node_output_id"] == "out"
        state = json.loads(saved.attrs["itool_state"])
        assert state["filter_operation"]["op"] == "gaussian_filter"
        xr.testing.assert_identical(
            saved[manager_workspace_io._ITOOL_DATA_NAME].rename(initial_output.name),
            initial_output,
        )


def test_manager_non_imagetool_node_displayed_provenance_uses_tool_provenance(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:

    class _StaticToolState(pydantic.BaseModel):
        value: int = 0

    class _StaticTool(erlab.interactive.utils.ToolWindow[_StaticToolState]):
        StateModel = _StaticToolState
        tool_name = "static-dummy"

        def __init__(
            self,
            data: xr.DataArray,
            provenance_spec: provenance.ToolProvenanceSpec,
        ) -> None:
            super().__init__()
            self._data = data
            self._status = _StaticToolState()
            self._provenance_spec = provenance_spec

        @property
        def tool_status(self) -> _StaticToolState:
            return self._status

        @tool_status.setter
        def tool_status(self, status: _StaticToolState) -> None:
            self._status = status

        @property
        def tool_data(self) -> xr.DataArray:
            return self._data

        def update_data(self, new_data: xr.DataArray) -> bool:
            self._data = new_data
            return True

        def current_provenance_spec(
            self,
        ) -> provenance.ToolProvenanceSpec | None:
            return self._provenance_spec

    data = xr.DataArray(np.arange(4.0), dims=("x",))
    provenance_spec = provenance.script(
        provenance.ScriptCodeOperation(label="Double data", code="result = data * 2"),
        start_label="Start from data",
        seed_code="data = source",
        active_name="result",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root_tool = itool(data, manager=False, execute=False)
        assert isinstance(root_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root_tool, show=False)

        child_uid = manager.add_childtool(
            _StaticTool(data, provenance_spec),
            0,
            show=False,
        )
        child_node = manager._child_node(child_uid)

        assert child_node.displayed_provenance_spec == provenance_spec


def test_manager_goldtool_output_itool_stales_when_fit_results_change(
    qtbot,
    gold,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(gold, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child_uid = manager.add_childtool(
            GoldTool(gold.copy(deep=True), data_name="gold_input"),
            0,
            show=False,
        )
        child = manager.get_childtool(child_uid)
        assert isinstance(child, GoldTool)
        configure_goldtool_child(child, fitted=True, spline=False)
        child.open_itool()

        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)

        output_uid = child_node._childtool_indices[0]
        output_node = manager._child_node(output_uid)
        before = fetch(output_uid).copy(deep=True)

        child.post_fit(child.edge_center + 1, child.edge_stderr)

        qtbot.wait_until(lambda: output_node.source_state == "stale", timeout=5000)
        xr.testing.assert_identical(fetch(output_uid), before)


def test_manager_ximageitem_open_itool_creates_independent_top_level_window(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.open_in_meshtool()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[0]._childtools) == 1,
            timeout=5000,
        )

        child_uid = manager._tool_graph.root_wrappers[0]._childtool_indices[0]
        child = typing.cast("typing.Any", manager.get_childtool(child_uid))
        assert child.main_image.data_array is not None

        child.main_image.open_itool()

        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        child_node = manager._child_node(child_uid)
        assert child_node._childtool_indices == []
        output_node = manager._tool_graph.root_wrappers[1]
        assert output_node.parent_uid is None
        assert output_node.output_id is None
        assert output_node.source_spec is None
        assert output_node.provenance_spec is not None
        assert output_node.provenance_spec.display_code() is not None
        xr.testing.assert_identical(fetch(1), child.main_image.data_array.T)

        monkeypatch.setattr(
            child,
            "_prompt_existing_output_imagetool",
            lambda: pytest.fail("unbound xImageItem opens should not prompt"),
        )
        updated = (child.main_image.data_array * 2).rename(
            child.main_image.data_array.name
        )
        child.main_image.setDataArray(updated)
        child.main_image.open_itool()

        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)
        assert child_node._childtool_indices == []
        second_output_node = manager._tool_graph.root_wrappers[2]
        assert second_output_node.parent_uid is None
        assert second_output_node.output_id is None
        assert second_output_node.source_spec is None
        assert second_output_node.provenance_spec is not None
        assert second_output_node.provenance_spec.display_code() is not None
        xr.testing.assert_identical(fetch(2), updated.T)


def test_manager_workspace_roundtrip_independent_unbound_imagetool(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.open_in_meshtool()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[0]._childtools) == 1,
            timeout=5000,
        )

        child_uid = manager._tool_graph.root_wrappers[0]._childtool_indices[0]
        child = typing.cast("typing.Any", manager.get_childtool(child_uid))
        expected = child.main_image.data_array.T.copy(deep=True)

        child.main_image.open_itool()
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        tree = manager._to_datatree()

        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        for node in tree.values():
            manager._load_workspace_node(typing.cast("xr.DataTree", node))

        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        matching_roots = [
            wrapper
            for index, wrapper in manager._tool_graph.root_wrappers.items()
            if wrapper.parent_uid is None
            and wrapper.source_spec is None
            and wrapper.provenance_spec is not None
            and wrapper.output_id is None
            and wrapper._childtool_indices == []
            and fetch(index).identical(expected)
        ]
        assert len(matching_roots) == 1
        assert matching_roots[0].provenance_spec is not None
        assert matching_roots[0].provenance_spec.display_code() is not None


def test_manager_metadata_uses_streamlined_child_derivation(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        data = xr.DataArray(
            np.arange(25).reshape((5, 5)),
            dims=["alpha", "eV"],
            coords={"alpha": np.arange(5), "eV": np.arange(5)},
        )
        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[0]._childtool_indices) == 1,
            timeout=5000,
        )

        child_uid = manager._tool_graph.root_wrappers[0]._childtool_indices[0]
        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)

        derivation = metadata_derivation_texts(manager)
        assert derivation[0] == "Start from selected parent ImageTool data"
        assert not any(line == "isel()" for line in derivation)
        assert not any("Sort coordinates" in line for line in derivation)
        assert any(line.startswith("transpose(") for line in derivation)

        copied = copy_full_code_for_uid(monkeypatch, manager, child_uid)
        namespace = _exec_generated_code(
            copied,
            {"data": parent_tool.slicer_area.data.copy(deep=True)},
        )
        result = namespace["result"]
        assert isinstance(result, xr.DataArray)
        xr.testing.assert_identical(result, manager.get_childtool(child_uid).result)
        assert ".isel()" not in copied
        assert "sort_coord_order" not in copied
        assert ".transpose(" in copied


def test_manager_nested_imagetool_refresh_updates_descendant_dependency(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    base = xr.DataArray(
        np.arange(16, dtype=float).reshape((4, 4)),
        dims=["x", "y"],
        coords={"x": np.arange(4), "y": np.arange(4)},
        name="scan",
    )
    initial_root_spec = provenance.selection(
        provenance.IselOperation(kwargs={"x": slice(0, 2)})
    )
    updated_root_spec = provenance.selection(
        provenance.IselOperation(kwargs={"x": slice(1, 3)})
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root_data = base.isel(x=slice(0, 2))
        root_tool = itool(root_data, manager=False, execute=False)
        assert isinstance(root_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root_tool, show=False, provenance_spec=initial_root_spec)

        child_tool = itool(root_data.copy(deep=False), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=provenance.full_data(),
            source_auto_update=True,
        )

        grandchild_data = root_data.isel(y=slice(0, 2))
        grandchild_tool = itool(grandchild_data, manager=False, execute=False)
        assert isinstance(grandchild_tool, erlab.interactive.imagetool.ImageTool)
        grandchild_uid = manager.add_imagetool_child(
            grandchild_tool,
            child_uid,
            show=False,
            source_spec=provenance.selection(
                provenance.IselOperation(kwargs={"y": slice(0, 2)})
            ),
            source_auto_update=True,
        )

        root_node = manager._tool_graph.root_wrappers[0]
        grandchild_node = manager._child_node(grandchild_uid)
        assert grandchild_node.provenance_spec is not None
        assert "slice(0, 2)" in typing.cast(
            "str", grandchild_node.provenance_spec.derivation_code()
        )

        root_node.set_detached_provenance(updated_root_spec)
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, base.isel(x=slice(1, 3)))

        qtbot.wait_until(
            lambda: (
                grandchild_node.provenance_spec is not None
                and grandchild_node.provenance_spec.derivation_code() is not None
                and "slice(1, 3)"
                in typing.cast("str", grandchild_node.provenance_spec.derivation_code())
            ),
            timeout=5000,
        )
        code = typing.cast("str", grandchild_node.provenance_spec.derivation_code())
        assert "derived = derived.isel(x=slice(1, 3))" in code
        assert "derived = derived.isel(x=slice(0, 2))" not in code


def test_manager_nested_imagetool_auto_update_can_be_disabled_from_auto_badge(
    qtbot,
    accept_dialog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    base = xr.DataArray(
        np.arange(24, dtype=float).reshape((6, 4)),
        dims=["x", "y"],
        coords={"x": np.arange(6), "y": np.arange(4)},
        name="scan",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root_data = base.isel(x=slice(0, 2))
        root_tool = itool(root_data, manager=False, execute=False)
        assert isinstance(root_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(
            root_tool,
            show=False,
            provenance_spec=provenance.selection(
                provenance.IselOperation(kwargs={"x": slice(0, 2)})
            ),
        )

        child_tool = itool(root_data.copy(deep=False), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=provenance.full_data(),
            source_auto_update=False,
        )
        child_node = manager._child_node(child_uid)

        updated = base.isel(x=slice(2, 4))
        manager._tool_graph.root_wrappers[0].set_detached_provenance(
            provenance.selection(provenance.IselOperation(kwargs={"x": slice(2, 4)}))
        )
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated)

        qtbot.wait_until(lambda: child_node.source_state == "stale", timeout=5000)
        xr.testing.assert_identical(fetch(child_uid), root_data)

        def _enable_auto_update(dialog: QtWidgets.QDialog) -> None:
            dialog.auto_update_check.setChecked(True)  # type: ignore[attr-defined]

        def _update_now(dialog: QtWidgets.QDialog) -> None:
            dialog.update_button.click()  # type: ignore[attr-defined]

        click_child_status_badge(
            manager,
            child_uid,
            accept_dialog,
            pre_call=_enable_auto_update,
            accept_call=_update_now,
        )

        qtbot.wait_until(lambda: child_node.source_state == "fresh", timeout=5000)
        assert child_node.source_auto_update is True
        xr.testing.assert_identical(fetch(child_uid), updated)
        _, badge_text, _ = child_status_badge(manager, child_uid)
        assert badge_text == "Auto"

        def _disable_auto_update(dialog: QtWidgets.QDialog) -> None:
            dialog.auto_update_check.setChecked(False)  # type: ignore[attr-defined]

        click_child_status_badge(
            manager,
            child_uid,
            accept_dialog,
            pre_call=_disable_auto_update,
        )
        assert child_node.source_auto_update is False

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_actions()
        assert manager.source_update_action.isVisible()
        assert manager.source_update_action.isEnabled()

        unbound_tool = itool(updated.copy(deep=False), manager=False, execute=False)
        assert isinstance(unbound_tool, erlab.interactive.imagetool.ImageTool)
        unbound_uid = manager.add_imagetool_child(unbound_tool, 0, show=False)
        manager.tree_view.clearSelection()
        select_child_tool(manager, unbound_uid)
        manager._update_actions()
        assert not manager.source_update_action.isVisible()

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        select_child_tool(manager, unbound_uid)
        manager._update_actions()
        assert not manager.source_update_action.isVisible()

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_actions()
        assert manager.source_update_action.isVisible()
        select_tools(manager, [0])
        manager._update_actions()
        assert not manager.source_update_action.isVisible()

        updated2 = base.isel(x=slice(4, 6))
        manager._tool_graph.root_wrappers[0].set_detached_provenance(
            provenance.selection(provenance.IselOperation(kwargs={"x": slice(4, 6)}))
        )
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated2)

        qtbot.wait_until(lambda: child_node.source_state == "stale", timeout=5000)
        xr.testing.assert_identical(fetch(child_uid), updated)


def test_manager_nested_stale_imagetool_marks_grandchildren_stale(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    base = xr.DataArray(
        np.arange(16, dtype=float).reshape((4, 4)),
        dims=["x", "y"],
        coords={"x": np.arange(4), "y": np.arange(4)},
        name="scan",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root_data = base.isel(x=slice(0, 2))
        root_tool = itool(root_data, manager=False, execute=False)
        assert isinstance(root_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(
            root_tool,
            show=False,
            provenance_spec=provenance.selection(
                provenance.IselOperation(kwargs={"x": slice(0, 2)})
            ),
        )

        child_tool = itool(root_data.copy(deep=False), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=provenance.full_data(),
            source_auto_update=False,
        )

        grandchild_tool = itool(
            root_data.isel(y=slice(0, 2)), manager=False, execute=False
        )
        assert isinstance(grandchild_tool, erlab.interactive.imagetool.ImageTool)
        grandchild_uid = manager.add_imagetool_child(
            grandchild_tool,
            child_uid,
            show=False,
            source_spec=provenance.selection(
                provenance.IselOperation(kwargs={"y": slice(0, 2)})
            ),
            source_auto_update=True,
        )

        root_node = manager._tool_graph.root_wrappers[0]
        child_node = manager._child_node(child_uid)
        grandchild_node = manager._child_node(grandchild_uid)

        root_node.set_detached_provenance(
            provenance.selection(provenance.IselOperation(kwargs={"x": slice(1, 3)}))
        )
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, base.isel(x=slice(1, 3)))

        qtbot.wait_until(lambda: child_node.source_state == "stale", timeout=5000)
        qtbot.wait_until(lambda: grandchild_node.source_state == "stale", timeout=5000)


def test_manager_manual_nested_refresh_updates_stale_ancestors(
    qtbot,
    accept_dialog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    base = xr.DataArray(
        np.arange(24, dtype=float).reshape((6, 4)),
        dims=["x", "y"],
        coords={"x": np.arange(6), "y": np.arange(4)},
        name="scan",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root_data = base.isel(x=slice(0, 2))
        root_tool = itool(root_data, manager=False, execute=False)
        assert isinstance(root_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(
            root_tool,
            show=False,
            provenance_spec=provenance.selection(
                provenance.IselOperation(kwargs={"x": slice(0, 2)})
            ),
        )

        child_tool = itool(root_data.copy(deep=False), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=provenance.full_data(),
            source_auto_update=False,
        )

        grandchild_tool = itool(
            root_data.isel(y=slice(0, 2)), manager=False, execute=False
        )
        assert isinstance(grandchild_tool, erlab.interactive.imagetool.ImageTool)
        grandchild_uid = manager.add_imagetool_child(
            grandchild_tool,
            child_uid,
            show=False,
            source_spec=provenance.selection(
                provenance.IselOperation(kwargs={"y": slice(0, 2)})
            ),
            source_auto_update=False,
        )

        child_node = manager._child_node(child_uid)
        grandchild_node = manager._child_node(grandchild_uid)
        updated_root = base.isel(x=slice(2, 4))

        manager._tool_graph.root_wrappers[0].set_detached_provenance(
            provenance.selection(provenance.IselOperation(kwargs={"x": slice(2, 4)}))
        )
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated_root)

        qtbot.wait_until(lambda: child_node.source_state == "stale", timeout=5000)
        qtbot.wait_until(lambda: grandchild_node.source_state == "stale", timeout=5000)

        def _update_now(dialog: QtWidgets.QDialog) -> None:
            dialog.update_button.click()  # type: ignore[attr-defined]

        click_child_status_badge(
            manager,
            grandchild_uid,
            accept_dialog,
            accept_call=_update_now,
        )

        qtbot.wait_until(lambda: child_node.source_state == "fresh", timeout=5000)
        qtbot.wait_until(lambda: grandchild_node.source_state == "fresh", timeout=5000)
        assert child_node.source_auto_update is False
        assert grandchild_node.source_auto_update is False
        xr.testing.assert_identical(fetch(child_uid), updated_root)
        xr.testing.assert_identical(
            fetch(grandchild_uid), updated_root.isel(y=slice(0, 2))
        )


def test_manager_manual_nested_refresh_resumes_after_deferred_parent(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:

    class _DeferredToolState(pydantic.BaseModel):
        value: int = 0

    class _DeferredTool(erlab.interactive.utils.ToolWindow[_DeferredToolState]):
        StateModel = _DeferredToolState
        tool_name = "deferred-dummy"

        def __init__(self, data: xr.DataArray) -> None:
            super().__init__()
            self._data = data
            self._status = _DeferredToolState()
            self.pending_data: xr.DataArray | None = None

        @property
        def tool_status(self) -> _DeferredToolState:
            return self._status

        @tool_status.setter
        def tool_status(self, status: _DeferredToolState) -> None:
            self._status = status

        @property
        def tool_data(self) -> xr.DataArray:
            return self._data

        def update_data(self, new_data: xr.DataArray) -> bool:
            self.pending_data = new_data
            self._source_refresh_deferred = self.has_source_binding
            return False

        def finish_deferred_update(self) -> None:
            if self.pending_data is None:
                raise RuntimeError("No deferred data is pending")
            self._data = self.pending_data
            self.pending_data = None
            self.finalize_source_refresh()

    base = xr.DataArray(
        np.arange(24, dtype=float).reshape((6, 4)),
        dims=["x", "y"],
        coords={"x": np.arange(6), "y": np.arange(4)},
        name="scan",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root_data = base.isel(x=slice(0, 2))
        root_tool = itool(root_data, manager=False, execute=False)
        assert isinstance(root_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root_tool, show=False)

        parent_tool = _DeferredTool(root_data)
        parent_uid = manager.add_childtool(parent_tool, 0, show=False)
        parent_tool.set_source_binding(provenance.full_data(), auto_update=False)

        leaf_tool = itool(root_data.isel(y=slice(0, 2)), manager=False, execute=False)
        assert isinstance(leaf_tool, erlab.interactive.imagetool.ImageTool)
        leaf_uid = manager.add_imagetool_child(
            leaf_tool,
            parent_uid,
            show=False,
            source_spec=provenance.selection(
                provenance.IselOperation(kwargs={"y": slice(0, 2)})
            ),
            source_auto_update=False,
        )

        parent_node = manager._child_node(parent_uid)
        leaf_node = manager._child_node(leaf_uid)
        updated_root = base.isel(x=slice(2, 4))

        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated_root)

        qtbot.wait_until(lambda: parent_node.source_state == "stale", timeout=5000)
        qtbot.wait_until(lambda: leaf_node.source_state == "stale", timeout=5000)

        assert manager._refresh_source_chain_to_uid(leaf_uid) is False
        assert parent_tool.pending_data is not None
        xr.testing.assert_identical(fetch(leaf_uid), root_data.isel(y=slice(0, 2)))

        parent_tool.finish_deferred_update()

        qtbot.wait_until(lambda: parent_node.source_state == "fresh", timeout=5000)
        qtbot.wait_until(lambda: leaf_node.source_state == "fresh", timeout=5000)
        xr.testing.assert_identical(parent_tool.tool_data, updated_root)
        xr.testing.assert_identical(fetch(leaf_uid), updated_root.isel(y=slice(0, 2)))
        assert not manager._dependency_tracker.has_pending_source_refreshes()


def test_manager_meshtool_output_itools_use_distinct_output_ids(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.open_in_meshtool()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[0]._childtools) == 1,
            timeout=5000,
        )

        child_uid = manager._tool_graph.root_wrappers[0]._childtool_indices[0]
        child = typing.cast("typing.Any", manager.get_childtool(child_uid))
        monkeypatch.setattr(
            child,
            "_prompt_existing_output_imagetool",
            lambda: (_ for _ in ()).throw(AssertionError("prompt should not open")),
        )

        child._corrected = child.tool_data.copy(deep=True) + 1
        child._mesh = child.tool_data.copy(deep=True) - 1

        child._corr_itool()
        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)

        child._mesh_itool()
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 2, timeout=5000)

        corr_uid, mesh_uid = child_node._childtool_indices
        corr_node = manager._child_node(corr_uid)
        mesh_node = manager._child_node(mesh_uid)
        assert manager.ntools == 1
        assert corr_node.parent_uid == child_uid
        assert mesh_node.parent_uid == child_uid
        assert corr_node.output_id == "meshtool.corrected_output"
        assert mesh_node.output_id == "meshtool.mesh_output"
        assert corr_node.source_spec is None
        assert corr_node.provenance_spec is not None
        assert mesh_node.source_spec is None
        assert mesh_node.provenance_spec is not None
        xr.testing.assert_identical(fetch(corr_uid), child._corrected)
        xr.testing.assert_identical(fetch(mesh_uid), child._mesh)


def test_manager_selection_dialog_opens_child_with_source_spec(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(3 * 4 * 5 * 6).reshape((3, 4, 5, 6)).astype(float),
        dims=["alpha", "eV", "beta", "hv"],
        coords={
            "alpha": np.arange(3, dtype=float),
            "eV": np.arange(4, dtype=float),
            "beta": np.arange(5, dtype=float),
            "hv": np.linspace(20.0, 70.0, 6),
        },
        name="scan",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.set_index(3, 2)
        dialog = SelectionDialog(parent_tool.slicer_area)
        assert (
            dialog.launch_mode_combo.currentData(QtCore.Qt.ItemDataRole.UserRole)
            == "replace"
        )
        set_transform_launch_mode(dialog, "nest")

        dialog.accept()

        parent = manager._tool_graph.root_wrappers[0]
        qtbot.wait_until(lambda: len(parent._childtool_indices) == 1, timeout=5000)
        child_uid = parent._childtool_indices[0]
        child_node = manager._child_node(child_uid)
        child_tool = child_node.imagetool
        expected = data.qsel(hv=40.0)

        assert child_tool is not None
        assert child_node.source_spec is not None
        assert [op.op for op in child_node.source_spec.operations] == ["qsel"]
        xarray.testing.assert_identical(
            child_node.source_spec.apply(parent_tool.slicer_area.data), expected
        )
        xarray.testing.assert_identical(
            child_tool.slicer_area._data.rename(None), expected.rename(None)
        )


@pytest.mark.parametrize(
    ("output_id", "expected_name"),
    [
        ("meshtool.corrected_output", "corrected"),
        ("meshtool.mesh_output", "mesh"),
    ],
)
def test_manager_meshtool_output_child_qsel_copy_code_tracks_selected_output_id(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
    output_id: str,
    expected_name: str,
) -> None:

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.open_in_meshtool()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[0]._childtools) == 1,
            timeout=5000,
        )

        child_uid = manager._tool_graph.root_wrappers[0]._childtool_indices[0]
        child = typing.cast("typing.Any", manager.get_childtool(child_uid))
        child._corrected = child.tool_data.copy(deep=True) + 1
        child._mesh = child.tool_data.copy(deep=True) - 1

        if output_id == "meshtool.corrected_output":
            child._corr_itool()
        else:
            child._mesh_itool()

        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)

        output_uid = child_node._childtool_indices[0]
        output_data = fetch(output_uid)
        nested_tool = itool(
            output_data.qsel(alpha=1, alpha_width=1), manager=False, execute=False
        )
        nested_uid = manager.add_imagetool_child(
            nested_tool,
            output_uid,
            show=False,
            source_spec=provenance.selection(
                provenance.QSelOperation(kwargs={"alpha": 1, "alpha_width": 1})
            ),
            source_auto_update=True,
        )

        copied = copy_full_code_for_uid(monkeypatch, manager, nested_uid)
        assert "corrected, mesh =" in copied
        assert "era.mesh.remove_mesh(" in copied
        assert not any(
            line == f"derived = {expected_name}" for line in copied.splitlines()
        )
        assert ")[0]" not in copied
        assert ")[1]" not in copied
        assert f"derived = {expected_name}.qsel(alpha=1, alpha_width=1)" in copied


def test_manager_fit2d_output_itools_use_distinct_output_ids(
    qtbot,
    monkeypatch,
    exp_decay_model,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        parent_tool = manager.get_imagetool(0)
        parent_tool.set_provenance_spec(
            provenance.script(
                provenance.ScriptCodeOperation(
                    label="Prepare parent data",
                    code="prepared_parent = data + 1",
                ),
                start_label="Start from test data",
                active_name="prepared_parent",
            )
        )

        child_uid, child = make_fit2d_child(manager, 0, exp_decay_model)
        monkeypatch.setattr(
            child,
            "_prompt_existing_output_imagetool",
            lambda: (_ for _ in ()).throw(AssertionError("prompt should not open")),
        )

        param_names = [
            child.param_plot_combo.itemText(index)
            for index in range(child.param_plot_combo.count())
        ]
        first_param_name, second_param_name = param_names[:2]
        first_param_index = child.param_plot_combo.findText(first_param_name)
        second_param_index = child.param_plot_combo.findText(second_param_name)
        assert first_param_index >= 0
        assert second_param_index >= 0
        params_full = []
        for index in range(len(child._params_full)):
            params = child._params.copy()
            params[first_param_name].set(value=1.0 + index)
            params[first_param_name].stderr = 0.01 + index
            params[second_param_name].set(value=10.0 + index)
            params[second_param_name].stderr = 0.1 + index
            params_full.append(params)
        child._params_full = params_full
        child._result_ds_full = [xr.Dataset() for _ in params_full]

        child.param_plot_combo.setCurrentIndex(first_param_index)
        assert child.param_plot_combo.currentText() == first_param_name
        first_values = child._param_plot_dataarray(first_param_name, stderr=False)
        child.param_plot_combo.setCurrentIndex(second_param_index)
        assert child.param_plot_combo.currentText() == second_param_name
        second_values = child._param_plot_dataarray(second_param_name, stderr=False)
        assert not second_values.identical(first_values)

        child.param_plot_combo.setCurrentIndex(first_param_index)
        child.param_plot._show_parameter_values()
        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)

        child.param_plot_combo.setCurrentIndex(second_param_index)
        assert child.param_plot_combo.currentText() == second_param_name
        child.param_plot._show_parameter_values()
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 2, timeout=5000)

        child.param_plot_combo.setCurrentIndex(first_param_index)
        assert child.param_plot_combo.currentText() == first_param_name
        child.param_plot._show_parameter_stderr()
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 3, timeout=5000)

        first_values_uid, second_values_uid, stderr_uid = child_node._childtool_indices
        first_values_node = manager._child_node(first_values_uid)
        second_values_node = manager._child_node(second_values_uid)
        stderr_node = manager._child_node(stderr_uid)
        assert manager.ntools == 1
        assert first_values_node.parent_uid == child_uid
        assert second_values_node.parent_uid == child_uid
        assert stderr_node.parent_uid == child_uid
        assert first_values_node.output_id == Fit2DTool._parameter_output_id(
            Fit2DTool.Output.PARAMETER_VALUES, first_param_name
        )
        assert second_values_node.output_id == Fit2DTool._parameter_output_id(
            Fit2DTool.Output.PARAMETER_VALUES, second_param_name
        )
        assert stderr_node.output_id == Fit2DTool._parameter_output_id(
            Fit2DTool.Output.PARAMETER_STDERR, first_param_name
        )
        assert first_values_node.source_spec is None
        assert first_values_node.provenance_spec is not None
        assert second_values_node.source_spec is None
        assert second_values_node.provenance_spec is not None
        assert stderr_node.source_spec is None
        assert stderr_node.provenance_spec is not None
        xr.testing.assert_identical(fetch(first_values_uid), first_values)
        xr.testing.assert_identical(fetch(second_values_uid), second_values)
        xr.testing.assert_identical(
            fetch(stderr_uid),
            child._param_plot_dataarray(first_param_name, stderr=True),
        )
        child.param_plot_combo.setCurrentIndex(second_param_index)
        assert first_values_node._update_from_parent_source()
        xr.testing.assert_identical(fetch(first_values_uid), first_values)
        assert not fetch(first_values_uid).identical(second_values)

        values_code = copy_full_code_for_uid(monkeypatch, manager, first_values_uid)
        second_values_code = copy_full_code_for_uid(
            monkeypatch, manager, second_values_uid
        )
        stderr_code = copy_full_code_for_uid(monkeypatch, manager, stderr_uid)
        assert "prepared_parent = data + 1" in values_code
        assert "prepared_parent = data + 1" in second_values_code
        assert "prepared_parent = data + 1" in stderr_code
        assert f".modelfit_coefficients.sel(param={first_param_name!r})" in values_code
        assert (
            f".modelfit_coefficients.sel(param={second_param_name!r})"
            in second_values_code
        )
        assert f".modelfit_stderr.sel(param={first_param_name!r})" in stderr_code


def test_manager_output_refresh_updates_stale_parent_source(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:

    class _OutputToolState(pydantic.BaseModel):
        value: int = 0

    class _OutputTool(erlab.interactive.utils.ToolWindow[_OutputToolState]):
        StateModel = _OutputToolState
        tool_name = "output-dummy"

        def __init__(self, data: xr.DataArray) -> None:
            super().__init__()
            self._data = data
            self._status = _OutputToolState()
            self.refreshed_inputs: list[xr.DataArray] = []

        @property
        def tool_status(self) -> _OutputToolState:
            return self._status

        @tool_status.setter
        def tool_status(self, status: _OutputToolState) -> None:
            self._status = status

        @property
        def tool_data(self) -> xr.DataArray:
            return self._data

        def update_data(self, new_data: xr.DataArray) -> bool:
            self.refreshed_inputs.append(new_data)
            self._data = new_data
            return True

        def output_imagetool_data(
            self, output_id: str | enum.Enum
        ) -> xr.DataArray | None:
            assert output_id == "out"
            return self._data + 10.0

        def output_imagetool_provenance(
            self, output_id: str | enum.Enum, data: xr.DataArray
        ) -> provenance.ToolProvenanceSpec | None:
            assert output_id == "out"
            return provenance.script(
                provenance.ScriptCodeOperation(
                    label="Use output", code="result = data + 10"
                ),
                start_label="Start from parent",
                active_name="result",
            )

    data = xr.DataArray(
        np.arange(12, dtype=float).reshape((3, 4)),
        dims=["x", "y"],
        coords={"x": np.arange(3), "y": np.arange(4)},
        name="scan",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root_tool = itool(data, manager=False, execute=False)
        assert isinstance(root_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root_tool, show=False)

        child = _OutputTool(data)
        child_uid = manager.add_childtool(child, 0, show=False)
        child.set_source_binding(provenance.full_data(), auto_update=False)

        initial_output = typing.cast("xr.DataArray", child.output_imagetool_data("out"))
        output_tool = itool(initial_output, manager=False, execute=False)
        assert isinstance(output_tool, erlab.interactive.imagetool.ImageTool)
        output_uid = manager.add_imagetool_child(
            output_tool,
            child_uid,
            show=False,
            provenance_spec=child.output_imagetool_provenance("out", initial_output),
            source_state="fresh",
            output_id="out",
        )

        child_node = manager._child_node(child_uid)
        output_node = manager._child_node(output_uid)
        updated = data * 2.0

        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated)

        qtbot.wait_until(lambda: child_node.source_state == "stale", timeout=5000)
        qtbot.wait_until(lambda: output_node.source_state == "stale", timeout=5000)
        xr.testing.assert_identical(fetch(output_uid), initial_output)

        assert manager._refresh_source_chain_to_uid(output_uid) is True
        assert child.refreshed_inputs
        assert child.source_state == "fresh"
        assert output_node.source_state == "fresh"
        xr.testing.assert_identical(fetch(output_uid), updated + 10.0)


def test_manager_fit2d_unbound_output_itool_creates_independent_top_level_windows(
    qtbot,
    monkeypatch,
    exp_decay_model,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child_uid, child = make_fit2d_child(manager, 0, exp_decay_model)
        initial = xr.DataArray(
            np.arange(4.0), dims=("x",), coords={"x": np.arange(4)}, name="initial"
        )
        updated = xr.DataArray(
            np.arange(4.0) + 10,
            dims=("x",),
            coords={"x": np.arange(4)},
            name="updated",
        )

        child._show_dataarray_in_itool(initial)
        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        assert child_node._childtool_indices == []
        first_output_node = manager._tool_graph.root_wrappers[1]
        assert first_output_node.parent_uid is None
        assert first_output_node.output_id is None
        assert first_output_node.source_spec is None
        assert first_output_node.provenance_spec is None
        assert not first_output_node.reloadable
        xr.testing.assert_identical(fetch(1), initial)
        monkeypatch.setattr(
            child,
            "_prompt_existing_output_imagetool",
            lambda: pytest.fail("unbound fit2d opens should not prompt"),
        )

        child._show_dataarray_in_itool(updated)

        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)
        assert child_node._childtool_indices == []
        second_output_node = manager._tool_graph.root_wrappers[2]
        assert second_output_node.parent_uid is None
        assert second_output_node.output_id is None
        assert second_output_node.source_spec is None
        assert second_output_node.provenance_spec is None
        assert not second_output_node.reloadable
        xr.testing.assert_identical(fetch(2), updated)


def test_manager_open_in_new_window_nests_image_tool_children(
    qtbot,
    monkeypatch,
    accept_dialog,
    tmp_path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        file_dir = tmp_path / ("very_long_directory_name_" * 4)
        file_dir.mkdir(parents=True)
        file_path = file_dir / "scan_with_a_long_name.h5"
        test_data.to_netcdf(file_path, engine="h5netcdf")

        itool(
            test_data,
            manager=True,
            file_path=file_path,
            load_func=(xr.load_dataarray, {"engine": "h5netcdf"}, 0),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent = manager._tool_graph.root_wrappers[0]
        manager.tree_view.clearSelection()
        select_tools(manager, [0])
        manager._update_info()
        root_index = manager.tree_view._model._row_index(0)
        assert root_index.data(_NODE_UID_ROLE) == parent.uid
        details = metadata_detail_map(manager)
        assert details["Kind"] == "ImageTool"
        assert details["File"] == str(file_path)
        assert "Chunks" not in details
        assert "Added" in details
        assert metadata_derivation_texts(manager) == [
            "Load data from file 'scan_with_a_long_name.h5'"
        ]
        assert manager._build_metadata_derivation_menu() is not None

        copied: list[str] = []
        monkeypatch.setattr(
            erlab.interactive.utils,
            "copy_to_clipboard",
            lambda text: copied.append(text) or text,
        )
        file_label = manager._metadata_detail_labels["File"]
        assert file_label.toolTip() == str(file_path)
        file_label.setFixedWidth(84)
        qtbot.wait(10)
        assert getattr(file_label, "full_text", file_label.text()) == str(file_path)
        assert file_label.text() == str(file_path)
        assert metadata_detail_map(manager)["File"] == str(file_path)
        details_button = manager.metadata_details_widget.findChild(
            QtWidgets.QToolButton,
            "manager_metadata_file_details_button",
        )
        assert details_button is not None

        def _inspect_source_dialog(dialog: QtWidgets.QDialog) -> None:
            assert not dialog.findChildren(QtWidgets.QLineEdit)
            assert not dialog.findChildren(QtWidgets.QPlainTextEdit)
            assert (
                dialog.findChild(
                    QtWidgets.QLabel, "manager_load_source_path_value_label"
                ).text()  # type: ignore[union-attr]
                == str(file_path)
            )
            assert (
                dialog.findChild(
                    QtWidgets.QLabel, "manager_load_source_loader_value_label"
                ).text()  # type: ignore[union-attr]
            ).endswith("xarray.load_dataarray")
            assert (
                dialog.findChild(
                    QtWidgets.QLabel, "manager_load_source_arguments_value_label"
                ).text()  # type: ignore[union-attr]
                == 'engine="h5netcdf"'
            )
            dialog.copy_code_button.click()  # type: ignore[attr-defined]

        accept_dialog(
            lambda: qtbot.mouseClick(
                details_button,
                QtCore.Qt.MouseButton.LeftButton,
            ),
            pre_call=_inspect_source_dialog,
        )
        assert copied
        load_namespace = _exec_generated_code(copied[-1], {})
        xr.testing.assert_identical(
            load_namespace["data"],
            xr.load_dataarray(file_path, engine="h5netcdf"),
        )

        manager.get_imagetool(0).slicer_area.images[0].open_in_new_window()
        qtbot.wait_until(lambda: len(parent._childtool_indices) == 1, timeout=5000)

        child_uid = parent._childtool_indices[0]
        child_node = manager._child_node(child_uid)
        child_tool = manager.get_imagetool(child_uid)

        assert child_node.is_imagetool
        assert child_node.parent_uid == parent.uid
        assert child_node.source_spec is not None
        xr.testing.assert_identical(fetch(child_uid), child_tool.slicer_area._data)

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)
        child_index = manager.tree_view._model._row_index(child_uid)
        assert child_index.data(_NODE_UID_ROLE) == child_uid
        child_details = metadata_detail_map(manager)
        assert child_details["Kind"] == "ImageTool"
        assert "Added" in child_details
        assert child_details["File"] == str(file_path)
        assert "Chunks" not in child_details
        assert metadata_derivation_texts(manager)

        child_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)

        nested_uid = child_node._childtool_indices[0]
        nested_tool = manager.get_childtool(nested_uid)
        assert isinstance(nested_tool, DerivativeTool)

        manager.tree_view.clearSelection()
        select_child_tool(manager, nested_uid)
        manager._update_info(uid=nested_uid)
        nested_details = metadata_detail_map(manager)
        assert nested_details["Kind"] == nested_tool.tool_name
        assert "Added" in nested_details
        assert metadata_derivation_texts(manager)
        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        monkeypatch.setattr(
            manager,
            "_prompt_replay_input_name",
            lambda _node: pytest.fail("file-backed replay should not prompt"),
        )
        trigger_menu_action(menu, manager._metadata_copy_full_action)
        assert copied
        assert not provenance.uses_default_replay_input(copied[-1])
        namespace = _exec_generated_code(copied[-1], {})
        result = namespace["result"]
        assert isinstance(result, xr.DataArray)
        xr.testing.assert_identical(result, nested_tool.result)


def test_manager_promote_action_enablement_and_menus(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent = manager._tool_graph.root_wrappers[0]
        manager.get_imagetool(0).slicer_area.images[0].open_in_new_window()
        qtbot.wait_until(lambda: len(parent._childtool_indices) == 1, timeout=5000)

        child_uid = parent._childtool_indices[0]
        child_tool = manager.get_imagetool(child_uid)
        child_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._child_node(child_uid)._childtool_indices) == 1,
            timeout=5000,
        )
        nested_uid = manager._child_node(child_uid)._childtool_indices[0]

        menus = menu_map_by_object_name(manager.menu_bar)
        assert manager.promote_action in menus["manager_edit_menu"].actions()
        assert manager.promote_action in manager.tree_view._menu.actions()

        manager.tree_view.clearSelection()
        manager._update_actions()
        assert not manager.promote_action.isEnabled()

        select_tools(manager, [0])
        manager._update_actions()
        assert not manager.promote_action.isEnabled()

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_actions()
        assert manager.promote_action.isEnabled()

        select_tools(manager, [0])
        manager._update_actions()
        assert not manager.promote_action.isEnabled()

        manager.tree_view.clearSelection()
        select_child_tool(manager, nested_uid)
        manager._update_actions()
        assert not manager.promote_action.isEnabled()

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_actions()
        assert manager.promote_action.isEnabled()


def test_manager_rename_action_enablement_for_child_selection(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent = manager._tool_graph.root_wrappers[0]
        manager.get_imagetool(0).slicer_area.images[0].open_in_new_window()
        qtbot.wait_until(lambda: len(parent._childtool_indices) == 1, timeout=5000)

        child_uid = parent._childtool_indices[0]
        child_tool = manager.get_imagetool(child_uid)
        child_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._child_node(child_uid)._childtool_indices) == 1,
            timeout=5000,
        )
        nested_uid = manager._child_node(child_uid)._childtool_indices[0]

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_actions()
        assert manager.rename_action.isEnabled()

        select_tools(manager, [0])
        manager._update_actions()
        assert not manager.rename_action.isEnabled()

        manager.tree_view.clearSelection()
        select_child_tool(manager, nested_uid)
        manager._update_actions()
        assert manager.rename_action.isEnabled()

        manager.rename_action.trigger()
        qtbot.wait_until(
            lambda: (
                manager.tree_view.state()
                == QtWidgets.QAbstractItemView.State.EditingState
            ),
            timeout=5000,
        )
        delegate = manager.tree_view.itemDelegate()
        assert isinstance(delegate, _ImageToolWrapperItemDelegate)
        assert isinstance(delegate._current_editor, QtWidgets.QLineEdit)
        delegate._current_editor.setText("renamed_child_tool")
        qtbot.keyClick(delegate._current_editor, QtCore.Qt.Key.Key_Return)
        qtbot.wait_until(
            lambda: (
                manager.get_childtool(nested_uid)._tool_display_name
                == "renamed_child_tool"
            ),
            timeout=5000,
        )

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        select_child_tool(manager, nested_uid)
        manager._update_actions()
        assert not manager.rename_action.isEnabled()


def test_manager_promote_selected_cancel_keeps_nested_imagetool(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent = manager._tool_graph.root_wrappers[0]
        manager.get_imagetool(0).slicer_area.images[0].open_in_new_window()
        qtbot.wait_until(lambda: len(parent._childtool_indices) == 1, timeout=5000)

        child_uid = parent._childtool_indices[0]
        select_child_tool(manager, child_uid)

        captured: dict[str, str] = {}

        def _cancel_prompt(
            dialog: QtWidgets.QMessageBox,
        ) -> QtWidgets.QMessageBox.StandardButton:
            captured["text"] = dialog.text()
            captured["info"] = dialog.informativeText()
            return QtWidgets.QMessageBox.StandardButton.Cancel

        monkeypatch.setattr(QtWidgets.QMessageBox, "exec", _cancel_prompt)

        manager.promote_action.trigger()

        assert captured["text"] == "Promote selected ImageTool to a top-level window?"
        assert "live update linkage" in captured["info"].lower()
        assert "detached history" in captured["info"].lower()
        assert manager.ntools == 1
        assert parent._childtool_indices == [child_uid]
        assert manager._child_node(child_uid).parent_uid == parent.uid


def test_manager_promote_child_imagetool_rehomes_subtree_and_detaches_provenance(
    qtbot,
    monkeypatch,
    accept_dialog,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(60).reshape((3, 4, 5)).astype(float),
        dims=["x", "y", "z"],
        coords={"x": np.arange(3), "y": np.arange(4), "z": np.arange(5)},
        name="scan",
    )
    file_path = tmp_path / "scan.h5"
    data.to_netcdf(file_path, engine="h5netcdf")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(
            data,
            manager=True,
            file_path=file_path,
            load_func=(xr.load_dataarray, {"engine": "h5netcdf"}, 0),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)

        def _nest_average(dialog) -> None:
            dialog.dim_checks["x"].setChecked(True)
            set_transform_launch_mode(dialog, "nest")

        accept_dialog(parent_tool.mnb._average, pre_call=_nest_average)

        parent = manager._tool_graph.root_wrappers[0]
        qtbot.wait_until(lambda: len(parent._childtool_indices) == 1, timeout=5000)

        child_uid = parent._childtool_indices[0]
        child_node = manager._child_node(child_uid)
        child_tool = manager.get_imagetool(child_uid)
        child_node.name = "averaged child"
        assert (
            manager_workspace_io._strip_workspace_modified_placeholder(
                child_tool.windowTitle()
            )
            == "averaged child"
        )
        child_before = fetch(child_uid).copy(deep=True)

        child_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)
        nested_uid = child_node._childtool_indices[0]

        select_child_tool(manager, child_uid)
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "exec",
            lambda _: QtWidgets.QMessageBox.StandardButton.Yes,
        )

        manager.promote_action.trigger()

        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        promoted_index = 1
        promoted = manager._tool_graph.root_wrappers[promoted_index]
        assert promoted.uid == child_uid
        assert child_uid not in parent._childtool_indices
        assert promoted.parent_uid is None
        assert promoted.source_spec is None
        assert promoted.provenance_spec is not None
        assert promoted._childtool_indices == [nested_uid]
        assert manager._child_node(nested_uid).parent_uid == child_uid
        qtbot.wait_until(
            lambda: manager.tree_view.selected_imagetool_indices == [promoted_index],
            timeout=5000,
        )
        assert manager.tree_view.selected_imagetool_indices == [promoted_index]
        assert manager.tree_view.selected_childtool_uids == []
        assert manager._root_wrapper_for_uid(nested_uid).index == promoted_index
        assert (
            manager.get_imagetool(promoted_index).windowTitle()
            == f"{promoted_index}: averaged child (scan)"
        )
        xr.testing.assert_identical(fetch(child_uid), child_before)
        xr.testing.assert_identical(
            manager._parent_source_data_for_uid(nested_uid),
            manager.get_imagetool(promoted_index).slicer_area._data,
        )

        manager._update_info()
        derivation = metadata_derivation_texts(manager)
        assert any("Aggregate" in line for line in derivation)

        updated = data.copy(deep=True)
        updated.data = np.asarray(updated.data) + 10
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated)

        assert promoted.source_state == "fresh"
        xr.testing.assert_identical(fetch(child_uid), child_before)


def test_manager_replace_current_sets_provenance_on_provenance_free_root(
    qtbot,
    accept_dialog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(60).reshape((3, 4, 5)).astype(float),
        dims=["x", "y", "z"],
        coords={"x": np.arange(3), "y": np.arange(4), "z": np.arange(5)},
        name="scan",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        root = manager._tool_graph.root_wrappers[0]
        root_tool = manager.get_imagetool(0)
        assert root.provenance_spec is None

        def _replace_average(dialog) -> None:
            dialog.dim_checks["x"].setChecked(True)
            set_transform_launch_mode(dialog, "replace")

        accept_dialog(root_tool.mnb._average, pre_call=_replace_average)

        assert root.source_spec is None
        assert root.provenance_spec is not None
        assert root.provenance_spec.derivation_code() == (
            'derived = data\nderived = derived.qsel.mean("x")'
        )
        xr.testing.assert_identical(
            root_tool.slicer_area._data.rename(None),
            data.qsel.mean("x").rename(None),
        )

        manager.tree_view.clearSelection()
        select_tools(manager, [0])
        manager._update_info()
        derivation = metadata_derivation_texts(manager)
        assert derivation == [
            "Start from current parent ImageTool data",
            'Aggregate(dims=("x",), func="mean")',
        ]


def test_manager_aggregate_child_refreshes_from_parent(
    qtbot,
    accept_dialog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(60).reshape((3, 4, 5)).astype(float),
        dims=["x", "y", "z"],
        coords={"x": np.arange(3), "y": np.arange(4), "z": np.arange(5)},
        name="scan",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)

        def _nest_sum(dialog) -> None:
            dialog.dim_checks["x"].setChecked(True)
            dialog.reducer_combo.setCurrentText("Sum")
            set_transform_launch_mode(dialog, "nest")

        accept_dialog(parent_tool.mnb._aggregate, pre_call=_nest_sum)

        parent = manager._tool_graph.root_wrappers[0]
        qtbot.wait_until(lambda: len(parent._childtool_indices) == 1, timeout=5000)
        child_uid = parent._childtool_indices[0]
        child_node = manager._child_node(child_uid)

        assert child_node.source_spec is not None
        assert [op.op for op in child_node.source_spec.operations] == [
            "qsel_aggregate",
        ]
        xr.testing.assert_identical(
            fetch(child_uid).rename(None), data.qsel.sum("x").rename(None)
        )

        updated = data + 10
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated)

        qtbot.wait_until(lambda: child_node.source_state == "stale", timeout=5000)
        assert child_node._update_from_parent_source() is True
        xr.testing.assert_identical(
            fetch(child_uid).rename(None), updated.qsel.sum("x").rename(None)
        )


def test_manager_replace_transform_on_filtered_source_child_keeps_live_source(
    qtbot,
    accept_dialog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(12).reshape((3, 4)).astype(float),
        dims=["x", "y"],
        coords={"x": np.arange(3, dtype=float), "y": np.arange(4, dtype=float)},
        name="scan",
    )
    operation = provenance.GaussianFilterOperation(sigma={"x": 1.0})

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root_tool = itool(data, manager=False, execute=False)
        assert isinstance(root_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root_tool, show=False)

        child_tool = itool(data.copy(deep=False), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=provenance.full_data(),
            source_auto_update=True,
        )
        child_node = manager._child_node(child_uid)
        child_tool.slicer_area.apply_filter_operation(operation, emit_edited=True)

        def _replace_average(dialog) -> None:
            dialog.dim_checks["x"].setChecked(True)
            set_transform_launch_mode(dialog, "replace")

        accept_dialog(child_tool.mnb._average, pre_call=_replace_average)

        filtered = operation.apply(data, parent_data=data)
        expected = filtered.qsel.mean("x")
        xr.testing.assert_identical(fetch(child_uid), expected)
        assert child_node.source_spec is not None
        assert child_node.source_spec.is_live_source
        assert [op.op for op in child_node.source_spec.operations] == [
            "gaussian_filter",
            "qsel_aggregate",
        ]

        updated = data + 10.0
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated)

        updated_filtered = operation.apply(updated, parent_data=updated)
        updated_expected = updated_filtered.qsel.mean("x")
        qtbot.wait_until(
            lambda: (
                child_node.source_state == "fresh"
                and fetch(child_uid).identical(updated_expected)
            ),
            timeout=5000,
        )
        xr.testing.assert_identical(
            fetch(child_uid),
            updated_expected,
        )


def test_manager_file_backed_replace_current_keeps_file_provenance(
    qtbot,
    monkeypatch,
    accept_dialog,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    file_path = tmp_path / "scan.h5"
    test_data.to_netcdf(file_path, engine="h5netcdf")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(
            test_data,
            manager=True,
            file_path=file_path,
            load_func=(xr.load_dataarray, {"engine": "h5netcdf"}, 0),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        root = manager._tool_graph.root_wrappers[0]
        root_tool = manager.get_imagetool(0)
        assert root.provenance_spec is not None
        assert root.provenance_spec.display_entries()[0].label == (
            "Load data from file 'scan.h5'"
        )

        def _replace_average(dialog) -> None:
            dialog.dim_checks["alpha"].setChecked(True)
            set_transform_launch_mode(dialog, "replace")

        accept_dialog(root_tool.mnb._average, pre_call=_replace_average)

        assert root.provenance_spec is not None
        assert root.provenance_spec.kind == "file"
        assert len(root.provenance_spec.replay_stages) == 1
        assert root.provenance_spec.replay_stages[0].source_kind == "full_data"
        assert [op.op for op in root.provenance_spec.replay_stages[0].operations] == [
            "qsel_aggregate",
        ]
        entries = root.provenance_spec.display_entries()
        assert entries[0].label == "Load data from file 'scan.h5'"
        assert any("Aggregate" in entry.label for entry in entries)

        manager.tree_view.clearSelection()
        select_tools(manager, [0])
        manager._update_info()
        assert metadata_derivation_texts(manager)[0] == "Load data from file 'scan.h5'"

        copied: list[str] = []
        monkeypatch.setattr(
            erlab.interactive.utils,
            "copy_to_clipboard",
            lambda text: copied.append(text) or text,
        )
        monkeypatch.setattr(
            manager,
            "_prompt_replay_input_name",
            lambda _node: pytest.fail("file-backed replay should not prompt"),
        )
        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        trigger_menu_action(menu, manager._metadata_copy_full_action)
        assert copied
        assert "scan.h5" in copied[-1]

        namespace = _exec_generated_code(copied[-1], {})
        derived = namespace["derived"]
        assert isinstance(derived, xr.DataArray)
        xr.testing.assert_identical(
            derived.rename(None),
            xr.load_dataarray(file_path, engine="h5netcdf")
            .astype(np.float64)
            .qsel.mean("alpha")
            .rename(None),
        )


def test_manager_provenance_file_load_edit_accept_and_cancel(
    qtbot,
    accept_dialog,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    first_path = tmp_path / "first.h5"
    second_path = tmp_path / "second.h5"
    first = test_data.copy(deep=True)
    second = (test_data * 3).rename(test_data.name)
    first.to_netcdf(first_path, engine="h5netcdf")
    second.to_netcdf(second_path, engine="h5netcdf")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(
            first,
            manager=True,
            file_path=first_path,
            load_func=(xr.load_dataarray, {"engine": "h5netcdf"}, 0),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        root = manager._tool_graph.root_wrappers[0]
        select_tools(manager, [0])
        manager._update_info()
        select_metadata_rows(manager, [0])
        row = manager._selected_derivation_row()
        assert row is not None
        editable, _reason = manager._provenance_edit_controller.can_edit_row(row)
        assert editable

        before_spec = root.provenance_spec
        before_data = root.slicer_area._data.copy(deep=True)

        accept_dialog(
            manager._edit_selected_derivation_step,
            accept_call=lambda dialog: dialog.reject(),
        )
        xr.testing.assert_identical(root.slicer_area._data, before_data)
        assert root.provenance_spec == before_spec

        def _edit_file(dialog: QtWidgets.QDialog) -> None:
            dialog.path_edit.setText(str(second_path))  # type: ignore[attr-defined]

        accept_dialog(manager._edit_selected_derivation_step, pre_call=_edit_file)

        assert root.provenance_spec is not None
        assert root.provenance_spec.file_load_source is not None
        assert pathlib.Path(root.provenance_spec.file_load_source.path) == second_path
        xr.testing.assert_identical(
            root.slicer_area._data.rename(None),
            second.astype(np.float64).rename(None),
        )


def test_manager_provenance_file_load_batch_edit_updates_matching_peer(
    qtbot,
    accept_dialog,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"
    old_dir.mkdir()
    new_dir.mkdir()
    paths = {
        "old_a": old_dir / "a.h5",
        "old_b": old_dir / "b.h5",
        "new_a": new_dir / "a.h5",
        "new_b": new_dir / "b.h5",
    }
    old_a = test_data.copy(deep=True)
    old_b = (test_data + 10).rename(test_data.name)
    new_a = (test_data + 100).rename(test_data.name)
    new_b = (test_data + 200).rename(test_data.name)
    for data, path in (
        (old_a, paths["old_a"]),
        (old_b, paths["old_b"]),
        (new_a, paths["new_a"]),
        (new_b, paths["new_b"]),
    ):
        data.to_netcdf(path, engine="h5netcdf")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        first_tool = _add_file_replay_tool(
            manager,
            old_a,
            _manager_replay_file_spec(paths["old_a"]),
        )
        second_tool = _add_file_replay_tool(
            manager,
            old_b,
            _manager_replay_file_spec(paths["old_b"]),
        )
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        select_tools(manager, [0])
        manager._update_info()
        select_metadata_rows(manager, [0])

        def _edit_batch(dialog: QtWidgets.QDialog) -> None:
            dialog.path_edit.setText(str(paths["new_a"]))  # type: ignore[attr-defined]
            dialog.batch_apply_check.setChecked(True)  # type: ignore[attr-defined]

        accept_dialog(manager._edit_selected_derivation_step, pre_call=_edit_batch)

        xr.testing.assert_identical(
            first_tool.slicer_area._data.rename(None),
            new_a.astype(np.float64).rename(None),
        )
        xr.testing.assert_identical(
            second_tool.slicer_area._data.rename(None),
            new_b.astype(np.float64).rename(None),
        )
        first_spec = manager._tool_graph.root_wrappers[0].provenance_spec
        second_spec = manager._tool_graph.root_wrappers[1].provenance_spec
        assert first_spec is not None
        assert first_spec.file_load_source is not None
        assert second_spec is not None
        assert second_spec.file_load_source is not None
        assert pathlib.Path(first_spec.file_load_source.path) == paths["new_a"]
        assert pathlib.Path(second_spec.file_load_source.path) == paths["new_b"]


def test_manager_provenance_rows_dim_when_not_activatable(
    qtbot,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    file_path = tmp_path / "scan.h5"
    test_data.to_netcdf(file_path, engine="h5netcdf")

    script_data = xr.DataArray(
        np.arange(6, dtype=float).reshape((2, 3)),
        dims=("x", "y"),
        name="scripted",
    )
    script_spec = provenance.script(
        provenance.ScriptCodeOperation(label="Copy source", code="derived = data"),
        provenance.QSelAggregationOperation(dims=("x",), func="mean"),
        start_label="Run script",
        active_name="derived",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        _add_file_replay_tool(manager, test_data, _manager_replay_file_spec(file_path))
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        select_tools(manager, [0])
        manager._update_info()

        load_item = manager.metadata_derivation_list.item(0)
        assert load_item is not None
        assert (
            load_item.data(manager_widgets._METADATA_DERIVATION_ACTIVATABLE_ROLE)
            is True
        )
        assert (
            load_item.data(manager_widgets._METADATA_DERIVATION_COPYABLE_ROLE) is False
        )
        assert load_item.foreground().style() == QtCore.Qt.BrushStyle.NoBrush
        select_metadata_rows(manager, [0])
        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        assert menu.defaultAction() is manager._metadata_edit_step_action

        tool = itool(script_data.qsel.mean("x"), manager=False, execute=False)
        assert isinstance(tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(tool, show=False, provenance_spec=script_spec)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        manager.tree_view.clearSelection()
        select_tools(manager, [1])
        manager._update_info()

        script_operation_item = None
        for row in range(manager.metadata_derivation_list.count()):
            item = manager.metadata_derivation_list.item(row)
            if item is None:
                continue
            if item.data(
                manager_widgets._METADATA_DERIVATION_COPYABLE_ROLE
            ) and not item.data(manager_widgets._METADATA_DERIVATION_ACTIVATABLE_ROLE):
                script_operation_item = item
                break
        assert script_operation_item is not None
        assert script_operation_item.toolTip()
        assert script_operation_item.flags() & QtCore.Qt.ItemFlag.ItemIsEnabled
        assert script_operation_item.foreground().color() == (
            manager.metadata_derivation_list.palette().color(
                QtGui.QPalette.ColorGroup.Disabled,
                QtGui.QPalette.ColorRole.Text,
            )
        )
        select_metadata_rows(
            manager,
            [manager.metadata_derivation_list.row(script_operation_item)],
        )
        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        assert menu.defaultAction() is None


def test_manager_provenance_row_activation_uses_edit_default_action(
    qtbot,
    monkeypatch,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    file_path = tmp_path / "scan.h5"
    test_data.to_netcdf(file_path, engine="h5netcdf")
    spec = _manager_replay_file_spec(
        file_path,
        provenance.QSelAggregationOperation(dims=("alpha",), func="mean"),
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        _add_file_replay_tool(manager, test_data.qsel.mean("alpha"), spec)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        select_tools(manager, [0])
        manager._update_info()
        activated_rows: list[provenance._ProvenanceDisplayRow | None] = []
        monkeypatch.setattr(
            manager._provenance_edit_controller,
            "edit_row",
            activated_rows.append,
        )

        select_metadata_rows(manager, [0])
        manager.metadata_derivation_list.setFocus()
        qtbot.keyClick(
            manager.metadata_derivation_list,
            QtCore.Qt.Key.Key_Return,
        )
        assert activated_rows == [manager._selected_derivation_row()]

        activated_rows.clear()
        item = manager.metadata_derivation_list.item(0)
        assert item is not None
        manager.metadata_derivation_list.itemActivated.emit(item)
        assert activated_rows == [manager._selected_derivation_row()]

        activated_rows.clear()
        manager.metadata_derivation_list.clearSelection()
        selection_model = manager.metadata_derivation_list.selectionModel()
        assert selection_model is not None
        for row in (0, 1):
            selection_model.select(
                manager.metadata_derivation_list.model().index(row, 0),
                QtCore.QItemSelectionModel.SelectionFlag.Select,
            )
        selection_model.setCurrentIndex(
            manager.metadata_derivation_list.model().index(0, 0),
            QtCore.QItemSelectionModel.SelectionFlag.NoUpdate,
        )
        manager.metadata_derivation_list.itemActivated.emit(item)
        assert activated_rows == []


def test_manager_provenance_row_activation_ignores_noneditable_row(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(6, dtype=float).reshape((2, 3)),
        dims=("x", "y"),
        name="scan",
    )
    spec = provenance.script(
        provenance.ScriptCodeOperation(label="Copy source", code="derived = data"),
        provenance.QSelAggregationOperation(dims=("x",), func="mean"),
        start_label="Run script",
        active_name="derived",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        tool = itool(data.qsel.mean("x"), manager=False, execute=False)
        assert isinstance(tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(tool, show=False, provenance_spec=spec)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        select_tools(manager, [0])
        manager._update_info()

        monkeypatch.setattr(
            manager._provenance_edit_controller,
            "edit_row",
            lambda _row: pytest.fail("non-editable activation should be ignored"),
        )
        item = None
        for row in range(manager.metadata_derivation_list.count()):
            candidate = manager.metadata_derivation_list.item(row)
            if candidate is None:
                continue
            if candidate.data(
                manager_widgets._METADATA_DERIVATION_COPYABLE_ROLE
            ) and not candidate.data(
                manager_widgets._METADATA_DERIVATION_ACTIVATABLE_ROLE
            ):
                item = candidate
                break
        assert item is not None
        select_metadata_rows(manager, [manager.metadata_derivation_list.row(item)])
        manager.metadata_derivation_list.itemActivated.emit(item)


def test_manager_provenance_context_menu_preserves_extended_selection(
    qtbot,
    monkeypatch,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    file_path = tmp_path / "scan.h5"
    test_data.to_netcdf(file_path, engine="h5netcdf")
    operation = provenance.AssignAttrsOperation(attrs={"note": "selected"})
    spec = _manager_replay_file_spec(
        file_path,
        provenance.QSelAggregationOperation(dims=("alpha",), func="mean"),
        operation,
    )
    displayed = operation.apply(
        test_data.qsel.mean("alpha"),
        parent_data=test_data.qsel.mean("alpha"),
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        _add_file_replay_tool(manager, displayed, spec)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        select_tools(manager, [0])
        manager._update_info()
        assert manager.metadata_derivation_list.count() >= 3
        select_metadata_rows(manager, [0, 1])
        target_item = manager.metadata_derivation_list.item(2)
        assert target_item is not None
        assert not target_item.isSelected()

        build_metadata_derivation_menu = manager._build_metadata_derivation_menu
        menu = build_metadata_derivation_menu()
        assert menu is not None
        assert not manager._metadata_edit_step_action.isEnabled()
        assert not manager._metadata_revert_step_action.isEnabled()
        assert not manager._metadata_delete_step_action.isEnabled()

        captured_selection: list[list[int]] = []

        def _capture_menu() -> None:
            captured_selection.append(
                [
                    manager.metadata_derivation_list.row(item)
                    for item in manager.metadata_derivation_list.selectedItems()
                ]
            )
            return

        monkeypatch.setattr(manager, "_build_metadata_derivation_menu", _capture_menu)
        pos = manager.metadata_derivation_list.visualItemRect(target_item).center()
        manager._show_metadata_derivation_menu(pos)

        assert captured_selection == [[0, 1]]
        assert [
            manager.metadata_derivation_list.row(item)
            for item in manager.metadata_derivation_list.selectedItems()
        ] == [0, 1]
        assert not target_item.isSelected()

        manager.metadata_derivation_list.setCurrentItem(target_item)
        manager.metadata_derivation_list.clearSelection()
        assert manager.metadata_derivation_list.currentItem() is target_item
        assert manager.metadata_derivation_list.selectedItems() == []
        menu = build_metadata_derivation_menu()
        assert menu is not None
        assert not manager._metadata_edit_step_action.isEnabled()
        assert not manager._metadata_revert_step_action.isEnabled()
        assert not manager._metadata_copy_selected_action.isEnabled()
        assert not manager._metadata_delete_step_action.isEnabled()


def test_manager_provenance_context_menu_groups_row_commands(
    qtbot,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    file_path = tmp_path / "scan.h5"
    test_data.to_netcdf(file_path, engine="h5netcdf")
    spec = _manager_replay_file_spec(
        file_path,
        provenance.QSelAggregationOperation(dims=("alpha",), func="mean"),
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        _add_file_replay_tool(manager, test_data.qsel.mean("alpha"), spec)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        select_tools(manager, [0])
        manager._update_info()
        select_metadata_rows(manager, [1])
        menu = manager._build_metadata_derivation_menu()
        assert menu is not None

        assert manager._metadata_delete_step_action.isEnabled()

        actions = menu.actions()
        assert actions[0] is manager._metadata_edit_step_action
        assert actions[1] is manager._metadata_revert_step_action
        assert actions[2].isSeparator()
        assert actions[3] is manager._metadata_copy_selected_action
        assert actions[4] is manager._metadata_paste_steps_action
        if manager._metadata_copy_full_action in actions:
            assert actions[5] is manager._metadata_copy_full_action
            separator_index = 6
        else:
            separator_index = 5
        assert actions[separator_index].isSeparator()
        assert actions[separator_index + 1] is manager._metadata_delete_step_action


def test_manager_provenance_script_operation_rows_are_not_editable_v1(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(6, dtype=float).reshape((2, 3)),
        dims=("x", "y"),
        name="scan",
    )
    spec = provenance.script(
        provenance.ScriptCodeOperation(label="Copy source", code="derived = data"),
        provenance.QSelAggregationOperation(dims=("x",), func="mean"),
        start_label="Run script",
        active_name="derived",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        tool = itool(data.qsel.mean("x"), manager=False, execute=False)
        assert isinstance(tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(tool, show=False, provenance_spec=spec)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        select_tools(manager, [0])
        manager._update_info()
        select_metadata_rows(manager, [2])
        row = manager._selected_derivation_row()
        assert row is not None
        assert row.edit_ref is not None

        editable, _reason = manager._provenance_edit_controller.can_edit_row(row)
        assert not editable


def test_manager_provenance_script_structured_row_can_revert(
    qtbot,
    monkeypatch,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(3 * 4 * 2 * 2, dtype=float).reshape((3, 4, 2, 2)),
        dims=("x", "y", "z", "w"),
        coords={
            "x": [0.0, 1.0, 2.0],
            "y": np.arange(4),
            "z": [0.0, 1.0],
            "w": [0.0, 1.0],
        },
        name="scan",
    )
    file_path = tmp_path / "scan.h5"
    data.to_netcdf(file_path, engine="h5netcdf")
    file_spec = _manager_replay_file_spec(file_path)
    spec = provenance.script(
        provenance.ScriptCodeOperation(
            label="Use source",
            code="derived = source.copy()",
        ),
        provenance.QSelAggregationOperation(dims=("x",), func="mean"),
        provenance.IselOperation(kwargs={"y": 0}),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            provenance.ScriptInput(
                name="source",
                label="Recorded source",
                provenance_spec=file_spec,
            ),
        ),
    )
    initial = provenance.replay_script_provenance(spec, {})

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        tool = itool(initial, manager=False, execute=False)
        assert isinstance(tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(tool, show=False, provenance_spec=spec)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        root = manager._tool_graph.root_wrappers[0]
        select_tools(manager, [0])
        manager._update_info()
        rows = root.derivation_display_rows
        aggregate_row = next(
            row
            for row in rows
            if row.replay_ref
            == provenance._ProvenanceStepRef("operation", operation_index=1)
        )
        row_index = rows.index(aggregate_row)
        select_metadata_rows(manager, [row_index])
        row = manager._selected_derivation_row()
        assert row is not None

        revertible, _reason = manager._provenance_edit_controller.can_revert_row(row)
        assert revertible

        monkeypatch.setattr(
            manager._provenance_edit_controller,
            "_confirm_revert",
            lambda: True,
        )
        manager._revert_selected_derivation_step()

        assert root.provenance_spec is not None
        assert [operation.op for operation in root.provenance_spec.operations] == [
            "script_code",
            "qsel_aggregate",
        ]
        xr.testing.assert_identical(
            tool.slicer_area._data.rename(None),
            data.qsel.mean("x").rename(None),
        )


def test_manager_provenance_structured_operation_edit_accept_and_cancel(
    qtbot,
    accept_dialog,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(24, dtype=float).reshape((3, 4, 2)),
        dims=("x", "y", "z"),
        coords={"x": [0.0, 1.0, 2.0], "y": np.arange(4), "z": [0.0, 1.0]},
        name="scan",
    )
    file_path = tmp_path / "scan.h5"
    data.to_netcdf(file_path, engine="h5netcdf")
    spec = _manager_replay_file_spec(
        file_path,
        provenance.QSelAggregationOperation(dims=("y",), func="mean"),
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        tool = _add_file_replay_tool(
            manager,
            provenance.replay_file_provenance(spec),
            spec,
        )
        root = manager._tool_graph.root_wrappers[0]

        select_tools(manager, [0])
        manager._update_info()
        select_metadata_rows(manager, [1])
        row = manager._selected_derivation_row()
        assert row is not None
        assert manager._provenance_edit_controller.can_edit_row(row)[0]

        before_spec = root.provenance_spec
        before_data = tool.slicer_area._data.copy(deep=True)
        accept_dialog(
            manager._edit_selected_derivation_step,
            accept_call=lambda dialog: dialog.reject(),
        )
        assert root.provenance_spec == before_spec
        xr.testing.assert_identical(tool.slicer_area._data, before_data)

        def _edit_aggregate(dialog: QtWidgets.QDialog) -> None:
            for check in dialog.dim_checks.values():  # type: ignore[attr-defined]
                check.setChecked(False)
            dialog.dim_checks["x"].setChecked(True)  # type: ignore[attr-defined]
            reducer_index = dialog.reducer_combo.findData("sum")  # type: ignore[attr-defined]
            dialog.reducer_combo.setCurrentIndex(reducer_index)  # type: ignore[attr-defined]

        accept_dialog(manager._edit_selected_derivation_step, pre_call=_edit_aggregate)

        assert root.provenance_spec is not None
        stage = root.provenance_spec.replay_stages[0]
        assert stage.operations == (
            provenance.QSelAggregationOperation(dims=("x",), func="sum"),
        )
        xr.testing.assert_identical(
            tool.slicer_area._data.rename(None),
            data.qsel.sum("x").rename(None),
        )


def test_manager_provenance_script_derived_structured_step_is_editable(
    qtbot,
    accept_dialog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(24, dtype=float).reshape((3, 4, 2)),
        dims=("x", "y", "z"),
        coords={"x": [0.0, 1.0, 2.0], "y": np.arange(4), "z": [0.0, 1.0]},
        name="scan",
    )

    with manager_context() as manager:
        manager.show()
        input_tool = typing.cast(
            "erlab.interactive.imagetool.ImageTool",
            itool(data, manager=False, execute=False),
        )
        manager.add_imagetool(input_tool, show=False)
        input_node = manager._tool_graph.root_wrappers[0]
        derived_data = (data + 1.0).qsel.mean("y")
        spec = provenance.script(
            provenance.ScriptCodeOperation(
                label="Evaluate console expression",
                code="derived = data_0 + 1.0",
            ),
            provenance.QSelAggregationOperation(dims=("y",), func="mean"),
            start_label="Run ImageTool manager console code",
            active_name="derived",
            script_inputs=(
                provenance.ScriptInput(
                    name="data_0",
                    label="ImageTool 0: scan",
                    node_uid=input_node.uid,
                    node_snapshot_token=input_node.snapshot_token,
                ),
            ),
        )
        derived_tool = typing.cast(
            "erlab.interactive.imagetool.ImageTool",
            itool(derived_data, manager=False, execute=False),
        )
        manager.add_imagetool(derived_tool, show=False, provenance_spec=spec)
        derived_node = manager._tool_graph.root_wrappers[1]

        select_tools(manager, [1])
        manager._update_info()
        script_code_row, structured_row = spec.display_rows()[2:]
        assert not manager._provenance_edit_controller.can_edit_row(script_code_row)[0]
        assert manager._provenance_edit_controller.can_edit_row(structured_row) == (
            True,
            "",
        )
        select_metadata_rows(manager, [3])

        def _edit_aggregate(dialog: QtWidgets.QDialog) -> None:
            for check in dialog.dim_checks.values():  # type: ignore[attr-defined]
                check.setChecked(False)
            dialog.dim_checks["x"].setChecked(True)  # type: ignore[attr-defined]
            reducer_index = dialog.reducer_combo.findData("sum")  # type: ignore[attr-defined]
            dialog.reducer_combo.setCurrentIndex(reducer_index)  # type: ignore[attr-defined]

        accept_dialog(manager._edit_selected_derivation_step, pre_call=_edit_aggregate)

        assert derived_node.provenance_spec is not None
        assert derived_node.provenance_spec.operations == (
            provenance.ScriptCodeOperation(
                label="Evaluate console expression",
                code="derived = data_0 + 1.0",
            ),
            provenance.QSelAggregationOperation(dims=("x",), func="sum"),
        )
        xr.testing.assert_identical(
            derived_tool.slicer_area._data.rename(None),
            (data + 1.0).qsel.sum("x").rename(None),
        )


def test_manager_provenance_active_filter_edit_accept_and_cancel(
    qtbot,
    accept_dialog,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(24, dtype=float).reshape((3, 4, 2)) + 1.0,
        dims=("x", "y", "z"),
        coords={"x": [0.0, 1.0, 2.0], "y": np.arange(4), "z": [0.0, 1.0]},
        name="scan",
    )
    file_path = tmp_path / "scan.h5"
    data.to_netcdf(file_path, engine="h5netcdf")
    spec = _manager_replay_file_spec(file_path)

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        tool = _add_file_replay_tool(manager, data, spec)
        operation = provenance.NormalizeOperation(dims=("x",), mode="area")
        tool.slicer_area.apply_filter_operation(operation, emit_edited=True)

        select_tools(manager, [0])
        manager._update_info()
        select_metadata_rows(manager, [1])
        row = manager._selected_derivation_row()
        assert row is not None
        assert manager._provenance_edit_controller.can_edit_row(row)[0]

        before_source = tool.slicer_area._data.copy(deep=True)
        before_filter = tool.slicer_area._accepted_filter_provenance_operation
        accept_dialog(
            manager._edit_selected_derivation_step,
            accept_call=lambda dialog: dialog.reject(),
        )
        xr.testing.assert_identical(tool.slicer_area._data, before_source)
        assert tool.slicer_area._accepted_filter_provenance_operation == before_filter

        def _edit_filter(dialog: QtWidgets.QDialog) -> None:
            for check in dialog.dim_checks.values():  # type: ignore[attr-defined]
                check.setChecked(False)
            dialog.dim_checks["y"].setChecked(True)  # type: ignore[attr-defined]
            dialog.opts[2].setChecked(True)  # type: ignore[attr-defined]

        accept_dialog(manager._edit_selected_derivation_step, pre_call=_edit_filter)

        assert tool.slicer_area._accepted_filter_provenance_operation == (
            provenance.NormalizeOperation(dims=("y",), mode="min")
        )
        xr.testing.assert_identical(tool.slicer_area._data, before_source)


def test_manager_provenance_edit_rejects_incompatible_downstream_and_reverts(
    qtbot,
    accept_dialog,
    monkeypatch,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(48, dtype=float).reshape((3, 4, 2, 2)) + 1.0,
        dims=("x", "y", "z", "w"),
        coords={
            "x": [1.0, 2.0, 4.0],
            "y": np.arange(4),
            "z": [0.0, 1.0],
            "w": [0.0, 1.0],
        },
        name="scan",
    )
    file_path = tmp_path / "scan.h5"
    data.to_netcdf(file_path, engine="h5netcdf")
    spec = _manager_replay_file_spec(
        file_path,
        provenance.QSelAggregationOperation(dims=("x",), func="mean"),
        provenance.IselOperation(kwargs={"y": 0}),
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        initial = provenance.replay_file_provenance(spec)
        tool = _add_file_replay_tool(manager, initial, spec)
        root = manager._tool_graph.root_wrappers[0]

        select_tools(manager, [0])
        manager._update_info()
        select_metadata_rows(manager, [1])

        before_spec = root.provenance_spec
        before_data = tool.slicer_area._data.copy(deep=True)
        failures: list[tuple[str, Exception]] = []
        monkeypatch.setattr(
            manager._provenance_edit_controller,
            "_show_failed",
            lambda title, exc: failures.append((title, exc)),
        )
        monkeypatch.setattr(
            manager._provenance_edit_controller,
            "_edited_operations_from_dialog",
            lambda _dialog_cls, _operation, _input_data: [
                provenance.QSelAggregationOperation(dims=("y",), func="mean")
            ],
        )

        manager._edit_selected_derivation_step()
        assert failures
        assert root.provenance_spec == before_spec
        xr.testing.assert_identical(tool.slicer_area._data, before_data)

        accept_dialog(manager._revert_selected_derivation_step)
        assert root.provenance_spec == before_spec
        xr.testing.assert_identical(tool.slicer_area._data, before_data)

        def _confirm_revert(dialog: QtWidgets.QDialog) -> None:
            button = typing.cast(
                "QtWidgets.QMessageBox",
                dialog,
            ).button(QtWidgets.QMessageBox.StandardButton.Yes)
            assert button is not None
            button.click()

        accept_dialog(
            manager._revert_selected_derivation_step,
            accept_call=_confirm_revert,
        )

        assert root.provenance_spec is not None
        assert root.provenance_spec.replay_stages[0].operations == (
            provenance.QSelAggregationOperation(dims=("x",), func="mean"),
        )
        xr.testing.assert_identical(
            tool.slicer_area._data.rename(None),
            data.qsel.mean("x").rename(None),
        )


def test_manager_detached_file_provenance_metadata_and_reload_roundtrip(
    qtbot,
    accept_dialog,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    file_path = tmp_path / "scan.h5"
    test_data.to_netcdf(file_path, engine="h5netcdf")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(
            test_data,
            manager=True,
            file_path=file_path,
            load_func=(xr.load_dataarray, {"engine": "h5netcdf"}, 0),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        root_tool = manager.get_imagetool(0)

        def _detach_average(dialog) -> None:
            dialog.dim_checks["alpha"].setChecked(True)
            set_transform_launch_mode(dialog, "detach")

        accept_dialog(root_tool.mnb._average, pre_call=_detach_average)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        tree = manager._to_datatree()
        provenance_payload = json.loads(
            tree["1/imagetool"].attrs["manager_node_provenance_spec"]
        )
        assert provenance_payload["schema_version"] == 2
        assert provenance_payload["kind"] == "file"
        assert provenance_payload["operations"] == []
        assert len(provenance_payload["replay_stages"]) == 1
        assert provenance_payload["replay_stages"][0]["source_kind"] == "full_data"
        assert [
            operation["op"]
            for operation in provenance_payload["replay_stages"][0]["operations"]
        ] == ["qsel_aggregate"]
        assert (
            provenance_payload["file_load_source"]["replay_call"]["target"]
            == "xarray.load_dataarray"
        )
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        for node in tree.values():
            manager._load_workspace_node(typing.cast("xr.DataTree", node))

        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        detached = manager._tool_graph.root_wrappers[1]
        detached_tool = manager.get_imagetool(1)
        assert detached.parent_uid is None
        assert detached.output_id is None
        assert detached.source_spec is None
        assert detached.provenance_spec is not None
        assert detached.provenance_spec.file_load_source is not None
        assert detached.reloadable
        assert detached_tool.slicer_area._file_path is None

        manager.tree_view.clearSelection()
        select_tools(manager, [1])
        manager._update_actions()
        manager._update_info()
        assert metadata_detail_map(manager)["File"] == str(file_path)
        assert metadata_derivation_texts(manager)[0] == "Load data from file 'scan.h5'"
        assert manager.reload_action.isVisible()

        updated = test_data + 100
        updated.to_netcdf(file_path, engine="h5netcdf")

        with qtbot.wait_signal(detached_tool.slicer_area.sigDataChanged):
            manager.reload_selected()

        assert detached.parent_uid is None
        assert detached.output_id is None
        assert detached.source_spec is None
        assert detached.provenance_spec is not None
        assert detached_tool.slicer_area._file_path is None
        xr.testing.assert_identical(
            fetch(1).rename(None),
            updated.astype(np.float64).qsel.mean("alpha").rename(None),
        )

        file_path.unlink()
        manager._update_actions()
        assert not detached.reloadable
        assert not manager.reload_action.isVisible()


def test_manager_workspace_loads_legacy_321_provenance_payload(
    qtbot,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    file_path = tmp_path / "scan.h5"
    test_data.to_netcdf(file_path, engine="h5netcdf")
    legacy_payload = {
        "schema_version": 1,
        "kind": "script",
        "start_label": "Load data from file 'scan.h5'",
        "seed_code": (
            "import xarray\n\n"
            f"derived = xarray.load_dataarray({str(file_path)!r}, "
            'engine="h5netcdf").astype("float64")'
        ),
        "active_name": "derived",
        "file_load_source": {
            "path": str(file_path),
            "loader_label": "Load Function",
            "loader_text": "xarray.load_dataarray",
            "kwargs_text": 'engine="h5netcdf"',
            "load_code": (
                "import xarray\n\n"
                f"data = xarray.load_dataarray({str(file_path)!r}, "
                'engine="h5netcdf").astype("float64")'
            ),
        },
        "operations": [
            {
                "op": "script_code",
                "label": 'Average(dims=("alpha",))',
                "code": 'derived = derived.qsel.average("alpha")',
                "copyable": True,
            }
        ],
    }

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        tree = manager._to_datatree()
        tree["0/imagetool"].attrs["manager_node_provenance_spec"] = json.dumps(
            legacy_payload
        )
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        for node in tree.values():
            manager._load_workspace_node(typing.cast("xr.DataTree", node))

        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        loaded = manager._tool_graph.root_wrappers[0]
        assert loaded.provenance_spec is not None
        assert loaded.provenance_spec.schema_version == 2
        assert loaded.provenance_spec.kind == "script"
        assert loaded.provenance_spec.file_load_source is not None
        assert loaded.provenance_spec.file_load_source.replay_call is None
        assert not loaded.reloadable

        manager.tree_view.clearSelection()
        select_tools(manager, [0])
        manager._update_info()
        assert metadata_detail_map(manager)["File"] == str(file_path)
        assert metadata_derivation_texts(manager) == [
            "Load data from file 'scan.h5'",
            'Average(dims=("alpha",))',
        ]


def test_manager_prompt_replay_input_name_accept_cancel_and_invalid(
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    class _Node:
        def __init__(self, name: str | None) -> None:
            self.data = xr.DataArray(np.arange(2), dims=("x",), name=name)

        def _metadata_data(self) -> xr.DataArray:
            return self.data

    class _FakeLineEdit:
        def __init__(self) -> None:
            self.validator_set = False
            self.selected = False

        def setValidator(self, _validator) -> None:
            self.validator_set = True

        def selectAll(self) -> None:
            self.selected = True

    class _FakeInputDialog:
        InputMode = QtWidgets.QInputDialog.InputMode
        responses: typing.ClassVar[list[tuple[int, str]]] = []
        instances: typing.ClassVar[list[typing.Any]] = []

        def __init__(self, _parent) -> None:
            self.line_edit = _FakeLineEdit()
            self.initial_text = ""
            self._result, self._text = self.responses.pop(0)
            self.instances.append(self)

        def setWindowTitle(self, _title: str) -> None:
            pass

        def setLabelText(self, _text: str) -> None:
            pass

        def setTextValue(self, text: str) -> None:
            self.initial_text = text

        def setInputMode(self, _mode) -> None:
            pass

        def findChild(self, _cls):
            return self.line_edit

        def exec(self) -> int:
            return self._result

        def textValue(self) -> str:
            return self._text

    accepted = int(QtWidgets.QDialog.DialogCode.Accepted)
    rejected = int(QtWidgets.QDialog.DialogCode.Rejected)
    _FakeInputDialog.responses = [
        (rejected, ""),
        (accepted, "bad-name"),
        (accepted, " custom_source "),
    ]
    monkeypatch.setattr(QtWidgets, "QInputDialog", _FakeInputDialog)

    with manager_context() as manager:
        assert manager._prompt_replay_input_name(_Node("data")) is None
        assert _FakeInputDialog.instances[0].initial_text == "source_data"
        assert _FakeInputDialog.instances[0].line_edit.validator_set
        assert _FakeInputDialog.instances[0].line_edit.selected

        assert manager._prompt_replay_input_name(_Node("valid_name")) is None
        assert _FakeInputDialog.instances[1].initial_text == "valid_name"

        assert manager._prompt_replay_input_name(_Node(None)) == "custom_source"
        assert _FakeInputDialog.instances[2].initial_text == "source_data"


def test_manager_nonuniform_transform_children_refresh_from_public_data(
    qtbot,
    accept_dialog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(20).reshape((5, 4)).astype(float),
        dims=["x", "y"],
        coords={"x": [0.0, 0.2, 0.8, 1.4, 2.0], "y": np.arange(4)},
        name="scan",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        assert parent_tool.slicer_area.data.dims == ("x_idx", "y")

        def _nest_coarsen(dialog) -> None:
            assert "x_idx" not in dialog.dim_checks
            dialog.dim_checks["x"].setChecked(True)
            dialog.window_spins["x"].setValue(2)
            dialog.boundary_combo.setCurrentText("trim")
            dialog.side_combo.setCurrentText("left")
            dialog.coord_func_combo.setCurrentText("mean")
            dialog.reducer_combo.setCurrentText("mean")
            set_transform_launch_mode(dialog, "nest")

        accept_dialog(parent_tool.mnb._coarsen, pre_call=_nest_coarsen)

        parent = manager._tool_graph.root_wrappers[0]
        qtbot.wait_until(lambda: len(parent._childtool_indices) == 1, timeout=5000)

        child_uid = parent._childtool_indices[0]
        child_node = manager._child_node(child_uid)
        child_tool = manager.get_imagetool(child_uid)

        assert child_node.source_spec is not None
        assert child_node.source_spec.kind == "public_data"
        xr.testing.assert_identical(
            child_tool.slicer_area._data.rename(None),
            data.coarsen(x=2, boundary="trim", side="left", coord_func="mean")
            .mean()
            .rename(None),
        )

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)
        derivation = metadata_derivation_texts(manager)
        assert derivation[0] == "Start from current parent ImageTool data"
        assert len(derivation) == 2
        assert "Coarsen" in derivation[1]

        updated = data.copy(deep=True)
        updated.data = np.asarray(updated.data) * 2

        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated)

        qtbot.wait_until(lambda: child_node.source_state == "stale", timeout=5000)
        assert child_node._update_from_parent_source() is True
        xr.testing.assert_identical(
            child_tool.slicer_area._data.rename(None),
            updated.coarsen(x=2, boundary="trim", side="left", coord_func="mean")
            .mean()
            .rename(None),
        )


def test_manager_transform_launch_modes_refresh_nested_and_detached(
    qtbot,
    monkeypatch,
    accept_dialog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(60).reshape((3, 4, 5)).astype(float),
        dims=["x", "y", "z"],
        coords={"x": np.arange(3), "y": np.arange(4), "z": np.arange(5)},
        name="scan",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)

        def _nest_average(dialog) -> None:
            set_transform_launch_mode(dialog, "nest")
            assert dialog.launch_mode_combo.toolTip()
            dialog.dim_checks["x"].setChecked(True)

        accept_dialog(parent_tool.mnb._average, pre_call=_nest_average)

        parent = manager._tool_graph.root_wrappers[0]
        qtbot.wait_until(lambda: len(parent._childtool_indices) == 1, timeout=5000)

        child_uid = parent._childtool_indices[0]
        child_node = manager._child_node(child_uid)
        child_tool = manager.get_imagetool(child_uid)
        xr.testing.assert_identical(
            child_tool.slicer_area._data.rename(None),
            data.qsel.mean("x").rename(None),
        )

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)
        details = metadata_detail_map(manager)
        assert details["Kind"] == "ImageTool"
        assert "Added" in details
        derivation = metadata_derivation_texts(manager)
        assert any("Aggregate" in line for line in derivation)
        assert not any("rename(" in line for line in derivation)

        copied: list[str] = []
        monkeypatch.setattr(
            erlab.interactive.utils,
            "copy_to_clipboard",
            lambda text: copied.append(text) or text,
        )

        def _replace_average(dialog) -> None:
            dialog.dim_checks["y"].setChecked(True)
            set_transform_launch_mode(dialog, "replace")

        accept_dialog(child_tool.mnb._average, pre_call=_replace_average)

        transforms = [
            op
            for op in typing.cast(
                "provenance.ToolProvenanceSpec",
                child_node.source_spec,
            ).operations
            if op.op == "qsel_aggregate"
        ]
        assert [op.op for op in transforms] == ["qsel_aggregate", "qsel_aggregate"]
        xr.testing.assert_identical(
            child_tool.slicer_area._data.rename(None),
            data.qsel.mean("x").qsel.mean("y").rename(None),
        )

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)
        derivation = metadata_derivation_texts(manager)
        assert derivation[0] == "Start from current parent ImageTool data"
        assert len(derivation) == 3
        assert "Aggregate" in derivation[1]
        assert "dims=" in derivation[1]
        assert "Aggregate" in derivation[2]
        assert "dims=" in derivation[2]
        manager.metadata_derivation_list.setFocus()
        clipboard = QtWidgets.QApplication.clipboard()
        select_metadata_rows(manager, [0])
        clipboard.clear()
        qtbot.keyClick(
            manager.metadata_derivation_list,
            QtCore.Qt.Key.Key_C,
            QtCore.Qt.KeyboardModifier.ControlModifier,
        )
        assert copied == []
        assert clipboard.text() == ""

        select_metadata_rows(manager, [1, 2])
        qtbot.keyClick(
            manager.metadata_derivation_list,
            QtCore.Qt.Key.Key_C,
            QtCore.Qt.KeyboardModifier.ControlModifier,
        )
        selected_namespace = _exec_generated_code(
            clipboard.text(),
            {"derived": data.copy(deep=True)},
        )
        assert clipboard.mimeData().hasFormat(
            manager_details_panel._PROVENANCE_STEPS_CLIPBOARD_MIME
        )
        selected_result = selected_namespace["derived"]
        assert isinstance(selected_result, xr.DataArray)
        xr.testing.assert_identical(
            selected_result.rename(None),
            data.qsel.mean("x").qsel.mean("y").rename(None),
        )

        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        trigger_menu_action(menu, manager._metadata_copy_selected_action)
        selected_namespace = _exec_generated_code(
            clipboard.text(),
            {"derived": data.copy(deep=True)},
        )
        selected_result = selected_namespace["derived"]
        assert isinstance(selected_result, xr.DataArray)
        xr.testing.assert_identical(
            selected_result.rename(None),
            data.qsel.mean("x").qsel.mean("y").rename(None),
        )

        monkeypatch.setattr(
            manager,
            "_prompt_replay_input_name",
            lambda _node: "source_data",
        )
        trigger_menu_action(menu, manager._metadata_copy_full_action)
        full_namespace = _exec_generated_code(
            copied[-1],
            {"source_data": data.copy(deep=True)},
        )
        assert not provenance.uses_default_replay_input(copied[-1])
        full_result = full_namespace["derived"]
        assert isinstance(full_result, xr.DataArray)
        xr.testing.assert_identical(
            full_result.rename(None),
            data.qsel.mean("x").qsel.mean("y").rename(None),
        )
        assert ".rename(" not in copied[-1]

        manual = xr.DataArray(
            np.arange(5, dtype=float) + 100.0,
            dims=["z"],
            coords={"z": data["z"].values},
            name=child_tool.slicer_area._data.name,
        )
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(child_uid, manual)
        xr.testing.assert_identical(fetch(child_uid), manual)

        updated = data.copy(deep=True)
        updated.data = np.asarray(updated.data) * 2

        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated)

        qtbot.wait_until(lambda: child_node.source_state == "stale", timeout=5000)
        assert child_node._update_from_parent_source() is True
        xr.testing.assert_identical(
            child_tool.slicer_area._data.rename(None),
            updated.qsel.mean("x").qsel.mean("y").rename(None),
        )

        def _detach_average(dialog) -> None:
            dialog.dim_checks["x"].setChecked(True)
            set_transform_launch_mode(dialog, "detach")

        accept_dialog(parent_tool.mnb._average, pre_call=_detach_average)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        detached = manager._tool_graph.root_wrappers[1]
        assert detached.source_spec is None
        assert detached.provenance_spec is not None
        detached_tool = manager.get_imagetool(1)
        detached_derivation_before = detached.provenance_spec.derivation_code()

        def _replace_detached_average(dialog) -> None:
            dialog.dim_checks["y"].setChecked(True)
            set_transform_launch_mode(dialog, "replace")

        accept_dialog(detached_tool.mnb._average, pre_call=_replace_detached_average)
        assert detached.source_spec is None
        assert detached.provenance_spec is not None
        detached_transforms = [
            op
            for op in detached.provenance_spec.operations
            if op.op == "qsel_aggregate"
        ]
        assert [op.op for op in detached_transforms] == [
            "qsel_aggregate",
            "qsel_aggregate",
        ]
        assert detached.provenance_spec.derivation_code() == (
            "derived = data\n"
            'derived = derived.qsel.mean("x")\n'
            'derived = derived.qsel.mean("y")'
        )
        assert detached.provenance_spec.derivation_code() != detached_derivation_before
        xr.testing.assert_identical(
            detached_tool.slicer_area._data.rename(None),
            updated.qsel.mean("x").qsel.mean("y").rename(None),
        )
        detached_before = detached_tool.slicer_area._data.copy(deep=True)

        manager.tree_view.clearSelection()
        select_tools(manager, [1])
        manager._update_info()
        detached_derivation = metadata_derivation_texts(manager)
        assert detached_derivation[0] == "Start from current parent ImageTool data"
        assert len(detached_derivation) == 3
        assert "Aggregate" in detached_derivation[1]
        assert "Aggregate" in detached_derivation[2]

        duplicated_detached_index = typing.cast("int", manager.duplicate_imagetool(1))
        duplicated_detached = manager._tool_graph.root_wrappers[
            duplicated_detached_index
        ]
        assert duplicated_detached.source_spec is None
        assert duplicated_detached.provenance_spec == detached.provenance_spec
        xr.testing.assert_identical(
            manager.get_imagetool(duplicated_detached_index).slicer_area._data.rename(
                None
            ),
            detached_tool.slicer_area._data.rename(None),
        )

        updated2 = data.copy(deep=True)
        updated2.data = np.asarray(updated2.data) * 3
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated2)

        qtbot.wait_until(lambda: child_node.source_state == "stale", timeout=5000)
        xr.testing.assert_identical(detached_tool.slicer_area._data, detached_before)

        manager.tree_view.clearSelection()
        manager._update_info()
        assert metadata_detail_map(manager) == {}
        assert metadata_derivation_texts(manager) == []
        assert manager._build_metadata_derivation_menu() is None

        select_tools(manager, [0])
        select_child_tool(manager, child_uid)
        manager._update_info()
        assert metadata_detail_map(manager) == {}
        assert metadata_derivation_texts(manager) == []
        assert manager._build_metadata_derivation_menu() is None


def test_manager_provenance_step_clipboard_payload_validation() -> None:
    operation_payload = provenance.AverageOperation(dims=("z",)).model_dump(mode="json")
    valid_payload = {
        "type": manager_details_panel._PROVENANCE_STEPS_CLIPBOARD_PAYLOAD_TYPE,
        "version": manager_details_panel._PROVENANCE_STEPS_CLIPBOARD_PAYLOAD_VERSION,
        "operations": [operation_payload],
    }

    mime_data = QtCore.QMimeData()
    mime_data.setData(
        manager_details_panel._PROVENANCE_STEPS_CLIPBOARD_MIME,
        json.dumps(valid_payload).encode("utf-8"),
    )
    payload = manager_details_panel._provenance_step_clipboard_payload(mime_data)
    assert payload is not None
    assert payload[0] == (provenance.AverageOperation(dims=("z",)),)
    assert payload[1] == "derived"
    assert not payload[2]

    text_mime_data = QtCore.QMimeData()
    text_mime_data.setText(json.dumps({**valid_payload, "active_name": "result"}))
    payload = manager_details_panel._provenance_step_clipboard_payload(text_mime_data)
    assert payload is not None
    assert payload[1] == "result"

    plain_text_mime_data = QtCore.QMimeData()
    plain_text_mime_data.setText("derived = data")
    assert (
        manager_details_panel._provenance_step_clipboard_payload(plain_text_mime_data)
        is None
    )
    assert (
        manager_details_panel._provenance_step_clipboard_payload(QtCore.QMimeData())
        is None
    )

    invalid_utf8_mime_data = QtCore.QMimeData()
    invalid_utf8_mime_data.setData(
        manager_details_panel._PROVENANCE_STEPS_CLIPBOARD_MIME,
        b"\xff",
    )
    assert (
        manager_details_panel._provenance_step_clipboard_payload(invalid_utf8_mime_data)
        is None
    )

    for malformed_payload in (
        [],
        {**valid_payload, "type": "other"},
        {**valid_payload, "version": -1},
        {**valid_payload, "operations": {}},
        {**valid_payload, "operations": [{"op": "unknown"}]},
    ):
        malformed_mime_data = QtCore.QMimeData()
        malformed_mime_data.setData(
            manager_details_panel._PROVENANCE_STEPS_CLIPBOARD_MIME,
            json.dumps(malformed_payload).encode("utf-8"),
        )
        assert (
            manager_details_panel._provenance_step_clipboard_payload(
                malformed_mime_data
            )
            is None
        )


def test_manager_selected_derivation_step_payload_filters_rows() -> None:
    def item(
        row: provenance._ProvenanceDisplayRow | str,
        *,
        copyable: bool = True,
        code: str | None = "derived = data",
    ) -> QtWidgets.QListWidgetItem:
        list_item = QtWidgets.QListWidgetItem()
        list_item.setData(manager_details_panel._METADATA_DERIVATION_ROW_ROLE, row)
        list_item.setData(
            manager_details_panel._METADATA_DERIVATION_COPYABLE_ROLE,
            copyable,
        )
        list_item.setData(manager_details_panel._METADATA_DERIVATION_CODE_ROLE, code)
        return list_item

    operation_ref = provenance._ProvenanceStepRef("operation", operation_index=0)
    script_row_a = provenance._ProvenanceDisplayRow(
        provenance.DerivationEntry("script a", None),
        replay_ref=operation_ref,
    )
    script_row_b = provenance._ProvenanceDisplayRow(
        provenance.DerivationEntry("script b", None),
        replay_ref=operation_ref,
    )
    missing_operation_row = provenance._ProvenanceDisplayRow(
        provenance.DerivationEntry("missing op", None),
        replay_ref=provenance._ProvenanceStepRef(
            "operation",
            operation_index=10,
        ),
    )
    non_copyable_script_row = provenance._ProvenanceDisplayRow(
        provenance.DerivationEntry("non-copyable script", None),
        replay_ref=operation_ref,
    )
    non_live_row = provenance._ProvenanceDisplayRow(
        provenance.DerivationEntry("non-live", None),
        replay_ref=operation_ref,
    )
    row_to_spec = {
        script_row_a: provenance.script(
            provenance.ScriptCodeOperation(label="script a", code="a = data"),
            start_label="Start script a",
            active_name="a",
        ),
        script_row_b: provenance.script(
            provenance.ScriptCodeOperation(label="script b", code="b = data"),
            start_label="Start script b",
            active_name="b",
        ),
        missing_operation_row: provenance.full_data(
            provenance.AverageOperation(dims=("x",))
        ),
        non_copyable_script_row: types.SimpleNamespace(
            _operation_for_ref=lambda _ref: provenance.ScriptCodeOperation(
                label="hidden script",
                code="derived = data",
                copyable=False,
            )
        ),
        non_live_row: types.SimpleNamespace(
            _operation_for_ref=lambda _ref: types.SimpleNamespace(live_applicable=False)
        ),
    }
    edit_controller = types.SimpleNamespace(
        _display_spec_for_row=lambda _node, row: row_to_spec.get(row)
    )
    selected_items: list[QtWidgets.QListWidgetItem] = []
    manager = types.SimpleNamespace(
        _metadata_node_uid="node",
        _tool_graph=types.SimpleNamespace(nodes={"node": object()}),
        _selected_derivation_items=lambda: selected_items,
        _provenance_edit_controller=edit_controller,
    )
    controller = manager_details_panel._DetailsPanelController(
        typing.cast("typing.Any", manager)
    )

    manager._metadata_node_uid = None
    assert controller._selected_derivation_step_payload() is None

    manager._metadata_node_uid = "missing"
    assert controller._selected_derivation_step_payload() is None

    manager._metadata_node_uid = "node"
    selected_items = [
        item("not a row"),
        item(
            provenance._ProvenanceDisplayRow(provenance.DerivationEntry("start", None))
        ),
        item(
            provenance._ProvenanceDisplayRow(
                provenance.DerivationEntry("file", None),
                replay_ref=provenance._ProvenanceStepRef("file_load"),
            )
        ),
        item(script_row_a, copyable=False),
        item(script_row_a, code=None),
        item(
            provenance._ProvenanceDisplayRow(
                provenance.DerivationEntry("no spec", None),
                replay_ref=operation_ref,
            )
        ),
        item(missing_operation_row),
        item(non_copyable_script_row),
        item(non_live_row),
    ]
    assert controller._selected_derivation_step_payload() is None

    selected_items = [item(script_row_a), item(script_row_b)]
    assert controller._selected_derivation_step_payload() is None


def test_manager_copy_selected_derivation_code_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = types.SimpleNamespace(_selected_derivation_code=lambda: "derived = data")
    controller = manager_details_panel._DetailsPanelController(
        typing.cast("typing.Any", manager)
    )

    copied: list[str] = []
    monkeypatch.setattr(
        erlab.interactive.utils,
        "copy_to_clipboard",
        lambda text: copied.append(text) or text,
    )
    monkeypatch.setattr(
        controller,
        "_selected_derivation_step_payload",
        lambda: None,
    )
    controller._copy_selected_derivation_code()
    assert copied == ["derived = data"]

    monkeypatch.setattr(
        controller,
        "_selected_derivation_step_payload",
        lambda: ((provenance.AverageOperation(dims=("x",)),), "derived", False),
    )
    monkeypatch.setattr(QtWidgets.QApplication, "clipboard", lambda: None)
    controller._copy_selected_derivation_code()


def test_manager_derivation_context_menu_ignores_empty_space(
    qtbot,
) -> None:
    list_widget = QtWidgets.QListWidget()
    qtbot.addWidget(list_widget)
    manager = types.SimpleNamespace(
        metadata_derivation_list=list_widget,
        _build_metadata_derivation_menu=lambda: pytest.fail("unexpected menu"),
    )
    controller = manager_details_panel._DetailsPanelController(
        typing.cast("typing.Any", manager)
    )

    controller._show_metadata_derivation_menu(QtCore.QPoint(10, 10))


def test_manager_paste_steps_validation_error_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _fake_edit_controller(None, metadata_uid=None)
    unavailable: list[str] = []
    monkeypatch.setattr(controller, "_show_unavailable", unavailable.append)
    controller.paste_steps(
        (provenance.AverageOperation(dims=("x",)),),
        active_name="derived",
        contains_script=False,
    )
    assert unavailable

    node = _fake_edit_node(
        provenance.full_data(),
        parent_uid="parent",
        source_display_spec=provenance.full_data(),
    )
    with pytest.raises(TypeError, match="Only live provenance operations"):
        controller._paste_structured_steps(
            node,
            (provenance.ScriptCodeOperation(label="script", code="derived = data"),),
        )

    monkeypatch.setattr(
        controller,
        "_replay_candidate_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad replay")),
    )
    with pytest.raises(manager_provenance_edit._ProvenanceReplayFailure) as exc_info:
        controller._paste_structured_steps(
            node,
            (provenance.AverageOperation(dims=("x",)),),
        )
    assert "pasted provenance steps" in exc_info.value.where


def test_manager_paste_detached_steps_uses_replay_spec_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _provenance_paste_test_data()
    local = provenance.full_data(provenance.AverageOperation(dims=("z",)))
    controller = _fake_edit_controller(None, metadata_uid=None)
    replaced: list[tuple[xr.DataArray, provenance.ToolProvenanceSpec, bool]] = []
    node = types.SimpleNamespace(
        uid="node",
        displayed_provenance_spec=provenance.full_data(),
        current_source_data=lambda: data,
        replace_with_detached_data=lambda data, spec, preserve_filter: replaced.append(
            (data, spec, preserve_filter)
        ),
    )
    monkeypatch.setattr(
        provenance,
        "compose_full_provenance",
        lambda _parent, _local: None,
    )

    controller._paste_detached_steps(
        typing.cast("typing.Any", node),
        local,
        where="testing fallback",
    )

    assert len(replaced) == 1
    xr.testing.assert_identical(replaced[0][0], local.apply(data))
    assert replaced[0][1] == local.to_replay_spec()
    assert replaced[0][2] is False


def test_manager_can_delete_row_reports_unavailable_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation_ref = provenance._ProvenanceStepRef(
        "operation",
        operation_index=0,
        stage_index=0,
    )
    row = provenance._ProvenanceDisplayRow(
        provenance.DerivationEntry("Average", None),
        replay_ref=operation_ref,
    )

    controller = _fake_edit_controller(None, metadata_uid=None)
    deletable, reason = controller.can_delete_row(row)
    assert not deletable
    assert reason

    source_child = _fake_edit_node(
        provenance.full_data(provenance.AverageOperation(dims=("x",))),
        parent_uid="parent",
        source_spec=provenance.full_data(),
    )
    controller = _fake_edit_controller(source_child)
    deletable, reason = controller.can_delete_row(row)
    assert not deletable
    assert reason

    controller = _fake_edit_controller(_fake_edit_node(None))
    deletable, reason = controller.can_delete_row(row)
    assert not deletable
    assert reason

    missing_ref_row = provenance._ProvenanceDisplayRow(
        provenance.DerivationEntry("Missing", None),
        replay_ref=provenance._ProvenanceStepRef(
            "operation",
            operation_index=99,
            stage_index=0,
        ),
    )
    controller = _fake_edit_controller(
        _fake_edit_node(provenance.full_data(provenance.AverageOperation(dims=("x",))))
    )
    deletable, reason = controller.can_delete_row(missing_ref_row)
    assert not deletable
    assert reason

    broken_script_spec = types.SimpleNamespace(
        kind="script",
        _operation_for_ref=lambda _ref: provenance.ScriptCodeOperation(
            label="Broken",
            code="derived = data",
        ),
        _replace_operation_ref=lambda *_args: (_ for _ in ()).throw(
            ValueError("bad ref")
        ),
    )
    controller = _fake_edit_controller(
        _fake_edit_node(typing.cast("typing.Any", broken_script_spec))
    )
    deletable, reason = controller.can_delete_row(row)
    assert not deletable
    assert reason

    valid_script_spec = provenance.script(
        provenance.ScriptCodeOperation(
            label="Offset",
            code="derived = derived + 1",
        ),
        provenance.ScriptCodeOperation(
            label="Scale",
            code="derived = derived * 2",
        ),
        start_label="Start from data",
        seed_code="derived = data_0",
        active_name="derived",
        script_inputs=(provenance.ScriptInput(name="data_0", label="Input"),),
    )
    controller = _fake_edit_controller(_fake_edit_node(valid_script_spec))
    deletable, reason = controller.can_delete_row(valid_script_spec.display_rows()[2])
    assert deletable
    assert reason == ""

    source_live_spec = provenance.full_data(provenance.AverageOperation(dims=("x",)))
    live_row = source_live_spec.display_rows(scope="source")[1]
    source_bound = _fake_edit_node(
        provenance.full_data(),
        parent_uid="parent",
        source_display_spec=source_live_spec,
    )
    controller = _fake_edit_controller(source_bound)
    deletable, reason = controller.can_delete_row(live_row)
    assert deletable
    assert reason == ""

    controller = _fake_edit_controller(_fake_edit_node(source_live_spec))
    deletable, reason = controller.can_delete_row(source_live_spec.display_rows()[1])
    assert not deletable
    assert reason


def test_manager_delete_row_error_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = provenance._ProvenanceDisplayRow(
        provenance.DerivationEntry("Average", None),
        replay_ref=provenance._ProvenanceStepRef(
            "operation",
            operation_index=0,
            stage_index=0,
        ),
    )
    node = _fake_edit_node(
        provenance.full_data(provenance.AverageOperation(dims=("x",)))
    )
    controller = _fake_edit_controller(node)
    monkeypatch.setattr(controller, "can_delete_row", lambda _row: (True, ""))

    failures: list[tuple[str, Exception]] = []
    monkeypatch.setattr(
        controller,
        "_show_failed",
        lambda title, exc: failures.append((title, exc)),
    )
    monkeypatch.setattr(controller, "_display_spec_for_row", lambda *_args: None)
    controller.delete_row(row)
    assert failures

    failures.clear()
    monkeypatch.setattr(
        controller,
        "_display_spec_for_row",
        lambda *_args: provenance.full_data(provenance.AverageOperation(dims=("x",))),
    )
    monkeypatch.setattr(
        controller,
        "_validate_and_replace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad replay")),
    )
    monkeypatch.setattr(
        controller,
        "_handle_missing_source_file",
        lambda *_args, **_kwargs: False,
    )
    controller.delete_row(row)
    assert failures

    failures.clear()
    monkeypatch.setattr(
        controller,
        "_handle_missing_source_file",
        lambda *_args, **_kwargs: True,
    )
    controller.delete_row(row)
    assert failures == []

    live_row = provenance.full_data(
        provenance.AverageOperation(dims=("x",))
    ).display_rows()[1]
    replaced: list[
        tuple[
            object,
            typing.Literal["display", "source"],
            provenance.ToolProvenanceSpec,
        ]
    ] = []
    monkeypatch.setattr(controller, "can_delete_row", lambda _row: (True, ""))
    monkeypatch.setattr(
        controller,
        "_display_spec_for_row",
        lambda *_args: provenance.full_data(provenance.AverageOperation(dims=("x",))),
    )
    monkeypatch.setattr(
        controller,
        "_validate_and_replace",
        lambda edit_node, scope, candidate, **_kwargs: replaced.append(
            (edit_node, scope, candidate)
        ),
    )
    controller.delete_row(live_row)
    assert len(replaced) == 1
    assert replaced[0][2].operations == ()


def test_manager_delete_provenance_step_preserves_later_operations(
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = _provenance_paste_test_data("scan")
    file_path = tmp_path / "scan.h5"
    data.to_netcdf(file_path, engine="h5netcdf")
    operations = (
        provenance.AverageOperation(dims=("z",)),
        provenance.IselOperation(kwargs={"y": 1}),
    )
    spec = _manager_replay_file_spec(file_path, *operations)
    displayed = provenance.replay_file_provenance(spec)

    with manager_context() as manager:
        tool = _add_file_replay_tool(manager, displayed, spec)
        index = 0

        select_tools(manager, [index])
        manager._update_info()
        select_metadata_rows(manager, [1])
        row = manager._selected_derivation_row()
        deletable, reason = manager._provenance_edit_controller.can_delete_row(row)
        assert deletable, reason
        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        trigger_menu_action(menu, manager._metadata_delete_step_action)

        expected_spec = spec._replace_operation_ref(
            provenance._ProvenanceStepRef(
                "operation",
                operation_index=0,
                stage_index=0,
            ),
            (),
        )
        expected = provenance.replay_file_provenance(expected_spec)
        xr.testing.assert_identical(tool.slicer_area._data, expected)
        root = manager._tool_graph.root_wrappers[index]
        assert root.provenance_spec == expected_spec
        assert metadata_derivation_texts(manager) == [
            f"Load data from file {file_path.name!r}",
            operations[1].derivation_entry().label,
        ]


def test_manager_delete_invalid_script_step_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = _provenance_paste_test_data("scan")
    operations = (
        provenance.ScriptCodeOperation(
            label="Create result",
            code="result = derived + 1.0",
        ),
        provenance.ScriptCodeOperation(
            label="Use result",
            code="result = result * 2.0",
        ),
    )
    spec = provenance.script(
        *operations,
        start_label="Start from data",
        seed_code="derived = data",
        active_name="result",
    )
    displayed = provenance.replay_script_provenance(spec, {"data": data})

    with manager_context() as manager:
        tool = itool(displayed, manager=False, execute=False)
        assert isinstance(tool, erlab.interactive.imagetool.ImageTool)
        index = manager.add_imagetool(tool, show=False, provenance_spec=spec)

        select_tools(manager, [index])
        manager._update_info()
        select_metadata_rows(manager, [1])
        row = manager._selected_derivation_row()
        deletable, reason = manager._provenance_edit_controller.can_delete_row(row)
        assert not deletable
        assert reason

        unavailable: list[str] = []
        monkeypatch.setattr(
            manager._provenance_edit_controller,
            "_show_unavailable",
            unavailable.append,
        )
        before = tool.slicer_area._data.copy(deep=True)
        before_spec = manager._tool_graph.root_wrappers[index].provenance_spec
        manager._delete_selected_derivation_step()

        assert unavailable
        xr.testing.assert_identical(tool.slicer_area._data, before)
        assert manager._tool_graph.root_wrappers[index].provenance_spec == before_spec


def test_manager_copy_paste_structured_provenance_steps(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source_base = _provenance_paste_test_data("source")
    source_operations = (
        provenance.AssignAttrsOperation(attrs={"copied": "yes"}),
        provenance.AverageOperation(dims=("z",)),
    )
    source_spec = provenance.full_data(*source_operations)
    source_data = source_spec.apply(source_base)

    dest_base = _provenance_paste_test_data("dest") + 100.0
    dest_seed_op = provenance.ScriptCodeOperation(
        label="Keep existing destination provenance",
        code="derived = derived.assign_attrs({'existing': 'yes'})",
    )
    dest_spec = provenance.script(
        dest_seed_op,
        start_label="Start from destination data",
        seed_code="derived = data",
        active_name="derived",
    )
    dest_data = provenance.replay_script_provenance(dest_spec, {"data": dest_base})

    with manager_context() as manager:
        source_tool = itool(source_data, manager=False, execute=False)
        assert isinstance(source_tool, erlab.interactive.imagetool.ImageTool)
        source_index = manager.add_imagetool(
            source_tool,
            show=False,
            provenance_spec=source_spec,
        )
        dest_tool = itool(dest_data, manager=False, execute=False)
        assert isinstance(dest_tool, erlab.interactive.imagetool.ImageTool)
        dest_index = manager.add_imagetool(
            dest_tool,
            show=False,
            provenance_spec=dest_spec,
        )

        clipboard = QtWidgets.QApplication.clipboard()
        clipboard.clear()
        manager.tree_view.clearSelection()
        select_tools(manager, [source_index])
        manager._update_info()
        select_metadata_rows(manager, [1, 2])
        qtbot.keyClick(
            manager.metadata_derivation_list,
            QtCore.Qt.Key.Key_C,
            QtCore.Qt.KeyboardModifier.ControlModifier,
        )

        payload = manager_details_panel._provenance_step_clipboard_payload(
            clipboard.mimeData()
        )
        assert payload is not None
        payload_operations, _active_name, contains_script = payload
        assert payload_operations == source_operations
        assert not contains_script
        assert clipboard.mimeData().hasFormat(
            manager_details_panel._PROVENANCE_STEPS_CLIPBOARD_MIME
        )
        copied_namespace = _exec_generated_code(
            clipboard.text(),
            {"derived": dest_data.copy(deep=True)},
        )
        copied_result = copied_namespace["derived"]
        assert isinstance(copied_result, xr.DataArray)
        expected = source_spec.apply(dest_data)
        xr.testing.assert_identical(copied_result, expected)

        manager.tree_view.clearSelection()
        select_tools(manager, [dest_index])
        manager._update_info()
        select_metadata_rows(manager, [0])
        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        assert manager._metadata_paste_steps_action in menu.actions()
        assert manager._metadata_paste_steps_action.isEnabled()

        manager.metadata_derivation_list.setFocus()
        qtbot.keyClick(
            manager.metadata_derivation_list,
            QtCore.Qt.Key.Key_V,
            QtCore.Qt.KeyboardModifier.ControlModifier,
        )
        xr.testing.assert_identical(dest_tool.slicer_area._data, expected)

        dest_node = manager._tool_graph.root_wrappers[dest_index]
        assert dest_node.displayed_provenance_spec is not None
        assert dest_node.displayed_provenance_spec.script_context_bindings == ()
        replay_namespace = _exec_generated_code(
            dest_node.displayed_provenance_spec.derivation_code(),
            {"data": dest_base.copy(deep=True)},
        )
        replayed = replay_namespace["derived"]
        assert isinstance(replayed, xr.DataArray)
        xr.testing.assert_identical(replayed, expected)

        bad_mime = QtCore.QMimeData()
        bad_mime.setData(
            manager_details_panel._PROVENANCE_STEPS_CLIPBOARD_MIME,
            b"{not-json",
        )
        clipboard.setMimeData(bad_mime)
        failures: list[str] = []
        monkeypatch.setattr(
            manager._provenance_edit_controller,
            "_show_unavailable",
            lambda reason: failures.append(reason),
        )
        before = dest_tool.slicer_area._data.copy(deep=True)
        manager._paste_provenance_steps_from_clipboard()
        assert failures
        xr.testing.assert_identical(dest_tool.slicer_area._data, before)


def test_manager_paste_structured_steps_preserves_source_binding(
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    parent_data = _provenance_paste_test_data("parent")
    copied_operation = provenance.AverageOperation(dims=("z",))
    source_spec = provenance.full_data(copied_operation)
    source_data = source_spec.apply(parent_data)
    child_base_spec = provenance.full_data(
        provenance.AssignAttrsOperation(attrs={"child": "bound"})
    )
    child_data = child_base_spec.apply(parent_data)

    with manager_context() as manager:
        parent_tool = itool(parent_data, manager=False, execute=False)
        assert isinstance(parent_tool, erlab.interactive.imagetool.ImageTool)
        parent_index = manager.add_imagetool(parent_tool, show=False)
        child_tool = itool(child_data, manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            parent_index,
            show=False,
            source_spec=child_base_spec,
            source_auto_update=True,
        )
        child_node = manager._child_node(child_uid)
        filter_operation = provenance.GaussianFilterOperation(sigma={"z": 1.0})
        child_tool.slicer_area.apply_filter_operation(filter_operation)
        displayed_child_source = child_node.displayed_source_spec
        assert displayed_child_source is not None
        source_tool = itool(source_data, manager=False, execute=False)
        assert isinstance(source_tool, erlab.interactive.imagetool.ImageTool)
        source_index = manager.add_imagetool(
            source_tool,
            show=False,
            provenance_spec=source_spec,
        )

        manager.tree_view.clearSelection()
        select_tools(manager, [source_index])
        manager._update_info()
        select_metadata_rows(manager, [1])
        manager._copy_selected_derivation_code()

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info()
        manager._paste_provenance_steps_from_clipboard()

        child_node = manager._child_node(child_uid)
        assert (
            child_node.parent_uid == manager._tool_graph.root_wrappers[parent_index].uid
        )
        assert child_node.source_auto_update is True
        expected_source_spec = displayed_child_source.append_replacement_operations(
            copied_operation
        )
        assert child_node.source_spec == expected_source_spec
        assert child_tool.slicer_area._accepted_filter_provenance_operation is None
        expected = child_node.source_spec.apply(parent_data)
        xr.testing.assert_identical(child_tool.slicer_area._data, expected)


def test_manager_copy_paste_script_provenance_steps_detaches_and_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = _provenance_paste_test_data("scan")
    child_source_spec = provenance.full_data(
        provenance.AssignAttrsOperation(attrs={"before": "script paste"})
    )
    child_data = child_source_spec.apply(data)
    script_operations = (
        provenance.ScriptCodeOperation(
            label="Offset data",
            code="result = derived + 2.0",
        ),
        provenance.AverageOperation(dims=("z",)),
    )
    script_spec = provenance.script(
        *script_operations,
        start_label="Run copied script",
        seed_code="derived = data",
        active_name="result",
    )
    script_data = provenance.replay_script_provenance(script_spec, {"data": data})

    with manager_context() as manager:
        parent_tool = itool(data, manager=False, execute=False)
        assert isinstance(parent_tool, erlab.interactive.imagetool.ImageTool)
        parent_index = manager.add_imagetool(parent_tool, show=False)
        child_tool = itool(child_data, manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            parent_index,
            show=False,
            source_spec=child_source_spec,
            source_auto_update=True,
        )
        source_tool = itool(script_data, manager=False, execute=False)
        assert isinstance(source_tool, erlab.interactive.imagetool.ImageTool)
        source_index = manager.add_imagetool(
            source_tool,
            show=False,
            provenance_spec=script_spec,
        )

        manager.tree_view.clearSelection()
        select_tools(manager, [source_index])
        manager._update_info()
        select_metadata_rows(manager, [1, 2])
        manager._copy_selected_derivation_code()
        payload = manager_details_panel._provenance_step_clipboard_payload(
            QtWidgets.QApplication.clipboard().mimeData()
        )
        assert payload is not None
        payload_operations, _active_name, contains_script = payload
        assert payload_operations == script_operations
        assert contains_script

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info()
        manager._paste_provenance_steps_from_clipboard()

        child_node = manager._child_node(child_uid)
        expected = provenance.replay_script_provenance(
            script_spec,
            {"data": child_data},
        )
        xr.testing.assert_identical(child_tool.slicer_area._data, expected)
        assert child_node.source_spec is None
        assert child_node.source_auto_update is False
        pasted_code = child_node.displayed_provenance_spec.derivation_code()
        assert pasted_code is not None
        replay_namespace = _exec_generated_code(
            pasted_code,
            {"data": data.copy(deep=True)},
        )
        replayed = replay_namespace["result"]
        assert isinstance(replayed, xr.DataArray)
        xr.testing.assert_identical(replayed, expected)
        derivation = metadata_derivation_texts(manager)
        assert script_operations[0].derivation_entry().label in derivation
        assert script_operations[1].derivation_entry().label in derivation
        assert not any("Paste provenance steps" in label for label in derivation)
        assert not any("derived = data" in label for label in derivation)

        bad_operation = provenance.ScriptCodeOperation(
            label="Use unavailable scratch value",
            code="result = scratch + 1.0",
        )
        bad_spec = provenance.script(
            bad_operation,
            start_label="Run invalid copied script",
            seed_code="derived = data",
            active_name="result",
        )
        bad_tool = itool(data, manager=False, execute=False)
        assert isinstance(bad_tool, erlab.interactive.imagetool.ImageTool)
        bad_index = manager.add_imagetool(
            bad_tool,
            show=False,
            provenance_spec=bad_spec,
        )

        manager.tree_view.clearSelection()
        select_tools(manager, [bad_index])
        manager._update_info()
        select_metadata_rows(manager, [1])
        manager._copy_selected_derivation_code()

        failures: list[tuple[str, Exception, dict[str, str]]] = []
        monkeypatch.setattr(
            manager._provenance_edit_controller,
            "_show_failed",
            lambda title, exc, **kwargs: failures.append((title, exc, kwargs)),
        )
        before = child_tool.slicer_area._data.copy(deep=True)
        before_spec = child_node.provenance_spec
        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info()
        manager._paste_provenance_steps_from_clipboard()

        assert failures
        assert failures[0][0] == "Could Not Paste Provenance Steps"
        assert failures[0][2]["text"] == (
            "The copied provenance steps could not be applied."
        )
        assert (
            "dimensions, coordinates, and inputs" in failures[0][2]["unchanged_reason"]
        )
        assert "Revert to This Step" not in failures[0][2]["unchanged_reason"]
        xr.testing.assert_identical(child_tool.slicer_area._data, before)
        assert child_node.provenance_spec == before_spec
        assert child_node.source_spec is None


def test_manager_paste_script_steps_replays_from_current_output_name(
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = _provenance_paste_test_data("scan")
    dest_base_spec = provenance.script(
        provenance.ScriptCodeOperation(
            label="Compute destination result",
            code="result = derived + 1.0",
        ),
        start_label="Start from destination data",
        seed_code="derived = data",
        active_name="result",
    )
    dest_data = provenance.replay_script_provenance(dest_base_spec, {"data": data})
    copied_operation = provenance.ScriptCodeOperation(
        label="Offset copied result",
        code="result = derived + 2.0",
    )
    source_spec = provenance.script(
        copied_operation,
        start_label="Run copied script",
        seed_code="derived = data",
        active_name="result",
    )
    source_data = provenance.replay_script_provenance(source_spec, {"data": data})

    with manager_context() as manager:
        dest_tool = itool(dest_data, manager=False, execute=False)
        assert isinstance(dest_tool, erlab.interactive.imagetool.ImageTool)
        dest_index = manager.add_imagetool(
            dest_tool,
            show=False,
            provenance_spec=dest_base_spec,
        )
        source_tool = itool(source_data, manager=False, execute=False)
        assert isinstance(source_tool, erlab.interactive.imagetool.ImageTool)
        source_index = manager.add_imagetool(
            source_tool,
            show=False,
            provenance_spec=source_spec,
        )

        manager.tree_view.clearSelection()
        select_tools(manager, [source_index])
        manager._update_info()
        select_metadata_rows(manager, [1])
        manager._copy_selected_derivation_code()

        manager.tree_view.clearSelection()
        select_tools(manager, [dest_index])
        manager._update_info()
        manager._paste_provenance_steps_from_clipboard()

        expected = dest_data + 2.0
        displayed_expected = (
            erlab.interactive.imagetool.slicer.ArraySlicer.validate_array(
                expected,
                copy_values=False,
            )
        )
        xr.testing.assert_identical(dest_tool.slicer_area._data, displayed_expected)
        dest_node = manager._tool_graph.root_wrappers[dest_index]
        assert dest_node.displayed_provenance_spec is not None
        assert [
            operation.derivation_entry().label
            for operation in dest_node.displayed_provenance_spec.operations
        ] == [
            "Compute destination result",
            "Offset copied result",
        ]
        assert [
            binding.model_dump(mode="json")
            for binding in dest_node.displayed_provenance_spec.script_context_bindings
        ] == [{"operation_index": 1, "names": ["data", "derived"]}]

        code = dest_node.displayed_provenance_spec.derivation_code()
        assert code is not None
        assert "Start from current ImageTool data" not in code
        assert "derived = derived" not in code
        replay_namespace = _exec_generated_code(
            code,
            {"data": data.copy(deep=True)},
        )
        replayed = replay_namespace["result"]
        assert isinstance(replayed, xr.DataArray)
        xr.testing.assert_identical(replayed, expected)


def test_manager_divide_by_coord_child_refresh_and_code(
    qtbot,
    accept_dialog,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(12, dtype=float).reshape((3, 4)) + 1.0,
        dims=["x", "y"],
        coords={
            "x": np.arange(3),
            "y": np.arange(4),
            "mesh_current": ("x", [1.0, 2.0, 4.0]),
        },
        name="scan",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)

        def _nest_divide(dialog) -> None:
            set_transform_launch_mode(dialog, "nest")
            dialog.coord_combo.setCurrentText("mesh_current")

        accept_dialog(parent_tool.mnb._divide_by_coord, pre_call=_nest_divide)

        parent = manager._tool_graph.root_wrappers[0]
        qtbot.wait_until(lambda: len(parent._childtool_indices) == 1, timeout=5000)
        child_uid = parent._childtool_indices[0]
        child_node = manager._child_node(child_uid)
        child_tool = manager.get_imagetool(child_uid)

        expected = (data / data.mesh_current).rename("scan")
        xr.testing.assert_identical(child_tool.slicer_area._data, expected)
        assert child_node.source_spec is not None
        operations = [
            op for op in child_node.source_spec.operations if op.op == "divide_by_coord"
        ]
        assert len(operations) == 1
        assert operations[0].coord_name == "mesh_current"

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)
        derivation = metadata_derivation_texts(manager)
        assert any("Divide by Coordinate" in line for line in derivation)

        copied: list[str] = []
        monkeypatch.setattr(
            erlab.interactive.utils,
            "copy_to_clipboard",
            lambda text: copied.append(text) or text,
        )
        monkeypatch.setattr(
            manager,
            "_prompt_replay_input_name",
            lambda _node: "source_data",
        )
        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        trigger_menu_action(menu, manager._metadata_copy_full_action)
        assert not provenance.uses_default_replay_input(copied[-1])
        assert ".rename(" not in copied[-1]

        namespace = _exec_generated_code(
            copied[-1], {"source_data": data.copy(deep=True)}
        )
        xr.testing.assert_identical(
            namespace["derived"].rename(None), expected.rename(None)
        )

        updated = data.copy(deep=True)
        updated.data = np.asarray(updated.data) * 2
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated)

        qtbot.wait_until(lambda: child_node.source_state == "stale", timeout=5000)
        assert child_node._update_from_parent_source() is True
        xr.testing.assert_identical(
            child_tool.slicer_area._data.rename(None),
            (updated / updated.mesh_current).rename(None),
        )


def test_manager_affine_coord_child_refreshes_from_formula(
    qtbot,
    accept_dialog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(12, dtype=float).reshape((3, 4)),
        dims=["x", "y"],
        coords={"x": np.arange(3), "y": np.arange(4)},
        name="scan",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)

        def _nest_affine(dialog) -> None:
            set_transform_launch_mode(dialog, "nest")
            dialog._coord_combo.setCurrentText("y")
            dialog.coord_widget.edit_mode_tabs.setCurrentIndex(1)
            dialog.coord_widget.scale_spin.setValue(2.0)
            dialog.coord_widget.offset_spin.setValue(0.5)

        accept_dialog(parent_tool.mnb._assign_coords, pre_call=_nest_affine)

        parent = manager._tool_graph.root_wrappers[0]
        qtbot.wait_until(lambda: len(parent._childtool_indices) == 1, timeout=5000)
        child_uid = parent._childtool_indices[0]
        child_node = manager._child_node(child_uid)
        child_tool = manager.get_imagetool(child_uid)

        operation = provenance.AffineCoordOperation(
            coord_name="y",
            scale=2.0,
            offset=0.5,
        )
        expected = operation.apply(data, parent_data=data).rename("scan")
        xr.testing.assert_identical(child_tool.slicer_area._data, expected)

        assert child_node.source_spec is not None
        operations = [
            op for op in child_node.source_spec.operations if op.op == "affine_coord"
        ]
        assert len(operations) == 1
        assert operations[0] == operation

        updated = data.assign_coords(y=np.arange(4, dtype=float) + 10.0)
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated)

        qtbot.wait_until(lambda: child_node.source_state == "stale", timeout=5000)
        assert child_node._update_from_parent_source() is True
        xr.testing.assert_identical(
            child_tool.slicer_area._data.rename(None),
            operation.apply(updated, parent_data=updated).rename(None),
        )


def test_manager_assign_attrs_child_refreshes_from_operation(
    qtbot,
    accept_dialog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(12, dtype=float).reshape((3, 4)),
        dims=["x", "y"],
        coords={"x": np.arange(3), "y": np.arange(4)},
        attrs={"source": "old"},
        name="scan",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)

        def _nest_attrs(dialog) -> None:
            set_transform_launch_mode(dialog, "nest")
            source_row = next(
                row
                for row in range(dialog.table.rowCount())
                if dialog._row_key(row) == "source"
            )
            dialog.table.item(source_row, 2).setText("new")
            dialog._add_empty_row()
            flag_row = dialog.table.rowCount() - 1
            dialog.table.item(flag_row, 0).setText("flag")
            typing.cast(
                "QtWidgets.QComboBox", dialog.table.cellWidget(flag_row, 1)
            ).setCurrentText("Bool")
            dialog.table.item(flag_row, 2).setText("True")

        accept_dialog(parent_tool.mnb._assign_attrs, pre_call=_nest_attrs)

        parent = manager._tool_graph.root_wrappers[0]
        qtbot.wait_until(lambda: len(parent._childtool_indices) == 1, timeout=5000)
        child_uid = parent._childtool_indices[0]
        child_node = manager._child_node(child_uid)
        child_tool = manager.get_imagetool(child_uid)

        operation = provenance.AssignAttrsOperation(
            attrs={"source": "new", "flag": True}
        )
        expected = operation.apply(data, parent_data=data).rename("scan")
        xr.testing.assert_identical(child_tool.slicer_area._data, expected)

        assert child_node.source_spec is not None
        operations = [
            op for op in child_node.source_spec.operations if op.op == "assign_attrs"
        ]
        assert operations == [operation]

        updated = data.assign_attrs(source="updated", count=2)
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated)

        qtbot.wait_until(lambda: child_node.source_state == "stale", timeout=5000)
        assert child_node._update_from_parent_source() is True
        xr.testing.assert_identical(
            child_tool.slicer_area._data.rename(None),
            operation.apply(updated, parent_data=updated).rename(None),
        )


def test_manager_reindex(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        # Open a tool with the manager
        itool([test_data, test_data, test_data], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 3)

        assert manager._tool_graph.displayed_indices == [0, 1, 2]

        # Remove tool at index 1
        manager.remove_imagetool(1)
        qtbot.wait_until(lambda: manager.ntools == 2)

        assert manager._tool_graph.displayed_indices == [0, 2]

        # Reindex
        manager.reindex_action.trigger()
        qtbot.wait_until(
            lambda: manager._tool_graph.displayed_indices == [0, 1], timeout=5000
        )


def test_manager_server_show_remove(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        # Open a tool with the manager
        itool([test_data, test_data], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        # Show tool at index 0
        _show_idx(0)

        # Remove tool at index 0
        _remove_idx(0)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)


def test_manager_data_watched_update_replaces_existing_tool_source_data(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        manager._data_recv([test_data], {}, watched_var=("data", "kernel-0"))
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        tool = manager.get_imagetool(0)
        updated = test_data.copy(deep=True)
        updated.data = np.asarray(updated.data) * 11

        with qtbot.wait_signal(tool.slicer_area.sigSourceDataReplaced):
            manager._data_watched_update("data", "kernel-0", updated)

        xr.testing.assert_identical(tool.slicer_area.data, updated)


def test_manager_workspace_roundtrip_preserves_watched_binding(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        manager._data_recv(
            [test_data],
            {},
            watched_var=("data", "watch:stable-data"),
            watched_metadata={
                "workspace_link_id": manager._workspace_state.link_id,
                "source_label": "notebook-a",
                "source_uid": "kernel-a",
                "connected": True,
            },
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        workspace_link_id = manager._workspace_state.link_id
        tree = manager._to_datatree()
        tree.attrs.update(manager._workspace_root_attrs_payload(delta_save_count=0))
        manifest = json.loads(tree.attrs["imagetool_workspace_manifest"])
        assert manifest["workspace_link_id"] == workspace_link_id
        attrs = tree["0/imagetool"].attrs
        assert attrs["manager_node_watched_varname"] == "data"
        assert attrs["manager_node_watched_uid"] == "watch:stable-data"
        assert attrs["manager_node_watched_workspace_link_id"] == workspace_link_id
        assert attrs["manager_node_watched_source_label"] == "notebook-a"
        assert attrs["manager_node_watched_source_uid"] == "kernel-a"

        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)
        manager._workspace_state.link_id = "different-workspace-link"

        manager._load_workspace_node(typing.cast("xr.DataTree", tree["0"]))
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        wrapper = manager._tool_graph.root_wrappers[0]
        assert wrapper.watched
        assert wrapper._watched_varname == "data"
        assert wrapper._watched_uid == "watch:stable-data"
        assert wrapper._watched_workspace_link_id == workspace_link_id
        assert wrapper._watched_source_label == "notebook-a"
        assert wrapper._watched_source_uid == "kernel-a"
        assert wrapper._watched_connected is False

        with qtbot.wait_signal(manager._sigReplyData) as blocker:
            manager._send_watch_info()
        assert blocker.args[0]["workspace_link_id"] == "different-workspace-link"
        assert blocker.args[0]["watched"][0]["workspace_link_id"] == workspace_link_id
        assert blocker.args[0]["watched"][0]["source_label"] == "notebook-a"
        assert blocker.args[0]["watched"][0]["source_uid"] == "kernel-a"

        manager._from_datatree(tree, replace=True, select=False)
        assert manager._workspace_state.link_id == workspace_link_id


def test_manager_workspace_watched_attrs_skip_missing_workspace_link(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(test_data, _in_manager=True),
            watched_var=("data", "watch:stable-data"),
            watched_workspace_link_id=None,
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        attrs = manager._to_datatree()["0/imagetool"].attrs
        assert attrs["manager_node_watched_varname"] == "data"
        assert attrs["manager_node_watched_uid"] == "watch:stable-data"
        assert "manager_node_watched_workspace_link_id" not in attrs


def test_manager_watched_badge_color_groups_by_source_uid(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        for varname, uid, source_uid in (
            ("left", "watch:left", "kernel-a"),
            ("right", "watch:right", "kernel-a"),
            ("other", "watch:other", "kernel-b"),
        ):
            manager.add_imagetool(
                erlab.interactive.imagetool.ImageTool(test_data, _in_manager=True),
                watched_var=(varname, uid),
                watched_source_uid=source_uid,
            )
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)

        left = manager._tool_graph.root_wrappers[0]
        right = manager._tool_graph.root_wrappers[1]
        other = manager._tool_graph.root_wrappers[2]

        assert (
            manager.color_for_watched_var_source(left)
            == manager_widgets._WATCHED_VAR_COLORS[0]
        )
        assert (
            manager.color_for_watched_var_source(right)
            == manager_widgets._WATCHED_VAR_COLORS[0]
        )
        assert (
            manager.color_for_watched_var_source(other)
            == manager_widgets._WATCHED_VAR_COLORS[1]
        )


def test_manager_watched_badge_color_falls_back_to_source_label(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(test_data, _in_manager=True),
            watched_var=("left", "watch:left"),
            watched_source_label="notebook-a",
        )
        manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(test_data, _in_manager=True),
            watched_var=("right", "watch:right"),
            watched_source_label="notebook-a",
        )
        manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(test_data, _in_manager=True),
            watched_var=("other", "watch:other"),
            watched_source_label="notebook-b",
        )
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)

        left = manager._tool_graph.root_wrappers[0]
        right = manager._tool_graph.root_wrappers[1]
        other = manager._tool_graph.root_wrappers[2]
        assert manager.color_for_watched_var_source(
            left
        ) == manager.color_for_watched_var_source(right)
        assert manager.color_for_watched_var_source(
            other
        ) != manager.color_for_watched_var_source(left)


def test_manager_watched_badge_color_uses_legacy_uid_suffix(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(test_data, _in_manager=True),
            watched_var=("left", "left legacy-kernel"),
        )
        manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(test_data, _in_manager=True),
            watched_var=("right", "right legacy-kernel"),
        )
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        left = manager._tool_graph.root_wrappers[0]
        right = manager._tool_graph.root_wrappers[1]
        assert manager.color_for_watched_var_source(
            left
        ) == manager.color_for_watched_var_source(right)


def test_manager_watched_root_provenance_uses_variable_name(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        manager._data_recv([test_data], {}, watched_var=("my_data", "kernel-0"))
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        node = manager._tool_graph.root_wrappers[0]
        provenance = node.provenance_spec
        assert provenance is not None
        code = provenance.display_code()
        assert code is not None
        namespace = _exec_generated_code(code, {"my_data": test_data.copy(deep=True)})
        derived = namespace["derived"]
        assert isinstance(derived, xr.DataArray)
        xr.testing.assert_identical(derived, manager.get_imagetool(0).slicer_area.data)
        assert provenance.display_entries()[0].label == (
            "Start from watched variable 'my_data'"
        )

        copied: list[str] = []
        monkeypatch.setattr(
            erlab.interactive.utils,
            "copy_to_clipboard",
            lambda text: copied.append(text) or text,
        )
        monkeypatch.setattr(
            manager,
            "_prompt_replay_input_name",
            lambda _node: pytest.fail("watched roots should not prompt"),
        )
        manager.tree_view.clearSelection()
        select_tools(manager, [0])
        manager._update_info(uid=node.uid)
        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        trigger_menu_action(menu, manager._metadata_copy_full_action)
        namespace = _exec_generated_code(
            copied[-1],
            {"my_data": test_data.copy(deep=True)},
        )
        derived = namespace["derived"]
        assert isinstance(derived, xr.DataArray)
        xr.testing.assert_identical(derived, manager.get_imagetool(0).slicer_area.data)


def test_manager_non_watched_full_code_prompts_for_source_variable(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        manager._data_recv([test_data], {})
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        node = manager._tool_graph.root_wrappers[0]
        node.set_detached_provenance(
            provenance.full_data(provenance.AverageOperation(dims=("alpha",)))
        )

        copied: list[str] = []
        prompted: list[str] = []
        monkeypatch.setattr(
            erlab.interactive.utils,
            "copy_to_clipboard",
            lambda text: copied.append(text) or text,
        )
        monkeypatch.setattr(
            manager,
            "_prompt_replay_input_name",
            lambda prompt_node: prompted.append(prompt_node.uid) or "source_data",
        )
        manager.tree_view.clearSelection()
        select_tools(manager, [0])
        manager._update_info(uid=node.uid)
        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        trigger_menu_action(menu, manager._metadata_copy_full_action)

        assert prompted == [node.uid]
        assert copied
        assert not provenance.uses_default_replay_input(copied[-1])
        namespace = _exec_generated_code(
            copied[-1], {"source_data": test_data.copy(deep=True)}
        )
        xr.testing.assert_identical(namespace["derived"], test_data.qsel.mean("alpha"))


def test_manager_non_watched_full_code_prompt_cancel_does_not_copy(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        manager._data_recv([test_data], {})
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        node = manager._tool_graph.root_wrappers[0]
        node.set_detached_provenance(
            provenance.full_data(provenance.AverageOperation(dims=("alpha",)))
        )

        copied: list[str] = []
        monkeypatch.setattr(
            erlab.interactive.utils,
            "copy_to_clipboard",
            lambda text: copied.append(text) or text,
        )
        monkeypatch.setattr(manager, "_prompt_replay_input_name", lambda _node: None)
        manager.tree_view.clearSelection()
        select_tools(manager, [0])
        manager._update_info(uid=node.uid)
        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        trigger_menu_action(menu, manager._metadata_copy_full_action)

        assert copied == []


def test_manager_file_backed_full_code_uses_load_code(
    qtbot,
    monkeypatch,
    tmp_path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    file_path = tmp_path / "scan.h5"
    test_data.to_netcdf(file_path, engine="h5netcdf")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(
            test_data,
            manager=True,
            file_path=file_path,
            load_func=(xr.load_dataarray, {"engine": "h5netcdf"}, 0),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        node = manager._tool_graph.root_wrappers[0]
        node.set_detached_provenance(
            provenance.full_data(provenance.AverageOperation(dims=("alpha",)))
        )

        copied: list[str] = []
        monkeypatch.setattr(
            erlab.interactive.utils,
            "copy_to_clipboard",
            lambda text: copied.append(text) or text,
        )
        monkeypatch.setattr(
            manager,
            "_prompt_replay_input_name",
            lambda _node: pytest.fail("file-backed replay should not prompt"),
        )
        manager.tree_view.clearSelection()
        select_tools(manager, [0])
        manager._update_info(uid=node.uid)
        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        trigger_menu_action(menu, manager._metadata_copy_full_action)

        assert copied
        assert not provenance.uses_default_replay_input(copied[-1])
        namespace = _exec_generated_code(copied[-1], {})
        xr.testing.assert_identical(namespace["derived"], test_data.qsel.mean("alpha"))


def test_manager_file_backed_full_code_prefers_scan_number_loader(
    qtbot,
    monkeypatch,
    example_loader,
    example_data_dir: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    file_path = example_data_dir / "data_002.h5"
    data = erlab.io.loaders["example"].load(file_path)

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(
            data,
            manager=True,
            file_path=file_path,
            load_func=("example", {}, 0),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        node = manager._tool_graph.root_wrappers[0]
        node.set_detached_provenance(
            provenance.full_data(provenance.AverageOperation(dims=("alpha",)))
        )

        copied: list[str] = []
        monkeypatch.setattr(
            erlab.interactive.utils,
            "copy_to_clipboard",
            lambda text: copied.append(text) or text,
        )
        monkeypatch.setattr(
            manager,
            "_prompt_replay_input_name",
            lambda _node: pytest.fail("file-backed replay should not prompt"),
        )
        manager.tree_view.clearSelection()
        select_tools(manager, [0])
        manager._update_info(uid=node.uid)
        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        trigger_menu_action(menu, manager._metadata_copy_full_action)

        assert copied
        assert f"erlab.io.load(2, data_dir={str(example_data_dir)!r})" in copied[-1]
        namespace = _exec_generated_code(copied[-1], {})
        xr.testing.assert_identical(namespace["derived"], data.qsel.mean("alpha"))


def test_manager_watched_root_child_tool_copy_code_uses_variable_name(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        manager._data_recv([test_data], {}, watched_var=("my_data", "kernel-0"))
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[0]._childtool_indices) == 1,
            timeout=5000,
        )

        child_uid = manager._tool_graph.root_wrappers[0]._childtool_indices[0]
        copied = copy_full_code_for_uid(monkeypatch, manager, child_uid)
        namespace = _exec_generated_code(
            copied,
            {"my_data": test_data.copy(deep=True)},
        )
        result = namespace["result"]
        assert isinstance(result, xr.DataArray)
        child_tool = manager.get_childtool(child_uid)
        assert isinstance(child_tool, DerivativeTool)
        xr.testing.assert_identical(result, child_tool.result)


def test_manager_watched_root_child_imagetool_copy_code_uses_variable_name(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        manager._data_recv([test_data], {}, watched_var=("my_data", "kernel-0"))
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_new_window()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[0]._childtool_indices) == 1,
            timeout=5000,
        )

        child_uid = manager._tool_graph.root_wrappers[0]._childtool_indices[0]
        copied = copy_full_code_for_uid(monkeypatch, manager, child_uid)
        namespace = _exec_generated_code(
            copied,
            {"my_data": test_data.copy(deep=True)},
        )
        derived = namespace["derived"]
        assert isinstance(derived, xr.DataArray)
        xr.testing.assert_identical(derived, fetch(child_uid))


def test_manager_watched_root_ftool_copy_code_1d_omits_duplicate_seed_and_noop_squeeze(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        manager._data_recv([test_data], {}, watched_var=("my_data", "kernel-0"))
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_ftool()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[0]._childtool_indices) == 1,
            timeout=5000,
        )

        child_uid = manager._tool_graph.root_wrappers[0]._childtool_indices[0]
        child_tool = manager.get_childtool(child_uid)
        assert isinstance(child_tool, Fit2DTool)

        copied: list[str] = []
        monkeypatch.setattr(
            erlab.interactive.utils,
            "copy_to_clipboard",
            lambda text: copied.append(text) or text,
        )
        child_tool.copy_code_1d()

        assert copied
        _assert_modelfit_code_replays_source(copied[-1], "my_data", test_data)


def test_manager_selecting_unfit_ftool_child_does_not_warn(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        manager._data_recv([test_data], {}, watched_var=("my_data", "kernel-0"))
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_ftool()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[0]._childtool_indices) == 1,
            timeout=5000,
        )

        child_uid = manager._tool_graph.root_wrappers[0]._childtool_indices[0]
        child_tool = manager.get_childtool(child_uid)
        assert isinstance(child_tool, Fit2DTool)

        warnings: list[tuple[str, str]] = []
        monkeypatch.setattr(
            child_tool,
            "_show_warning",
            lambda title, text: warnings.append((title, text)),
        )

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)

        qtbot.wait_until(
            lambda: "ftool_2d" in manager.text_box.toHtml().lower(), timeout=5000
        )
        assert "ftool_2d" in manager.text_box.toPlainText().lower()
        assert manager.preview_widget.isVisible()
        assert not manager_preview_pixmap(manager).isNull()
        assert metadata_detail_map(manager)["Kind"] == "ftool_2d"
        assert not manager._metadata_full_code_available
        assert not warnings


def test_manager_fit1d_child_side_panel(
    qtbot,
    exp_decay_model,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child_uid, _ = make_fit1d_child(manager, 0, exp_decay_model)

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)

        qtbot.wait_until(
            lambda: "ftool_1d" in manager.text_box.toHtml().lower(), timeout=5000
        )
        assert "ftool_1d" in manager.text_box.toPlainText().lower()
        assert "not fit yet" in manager.text_box.toPlainText().lower()
        assert not manager.preview_widget.isVisible()
        assert metadata_detail_map(manager)["Kind"] == "ftool_1d"


def test_manager_fit2d_child_side_panel_live_refresh(
    qtbot,
    exp_decay_model,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child_uid, child = make_fit2d_child(manager, 0, exp_decay_model)

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)

        qtbot.wait_until(
            lambda: "ftool_2d" in manager.text_box.toHtml().lower(), timeout=5000
        )
        old_html = manager.text_box.toHtml()
        new_index = (
            child.y_min_spin.value()
            if child._current_idx != child.y_min_spin.value()
            else child.y_max_spin.value()
        )
        child.y_index_spin.setValue(new_index)

        qtbot.wait_until(lambda: manager.text_box.toHtml() != old_html, timeout=5000)
        assert f"index {new_index}" in manager.text_box.toPlainText().lower()


def test_manager_goldtool_child_side_panel(
    qtbot,
    gold,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(gold, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child = GoldTool(gold.copy(deep=True), data_name="gold_input")
        child_uid = manager.add_childtool(child, 0, show=False)
        configure_goldtool_child(child, fitted=True, spline=True)

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)

        qtbot.wait_until(
            lambda: "goldtool" in manager.text_box.toHtml().lower(), timeout=5000
        )
        assert "goldtool" in manager.text_box.toPlainText().lower()
        assert manager.preview_widget.isVisible()
        assert not manager_preview_pixmap(manager).isNull()
        assert metadata_detail_map(manager)["Kind"] == "goldtool"


def test_manager_goldtool_child_side_panel_live_refresh(
    qtbot,
    gold,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(gold, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child = GoldTool(gold.copy(deep=True), data_name="gold_input")
        child_uid = manager.add_childtool(child, 0, show=False)
        configure_goldtool_child(child, fitted=True, spline=False)

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)

        qtbot.wait_until(
            lambda: "goldtool" in manager.text_box.toHtml().lower(), timeout=5000
        )
        old_html = manager.text_box.toHtml()
        child.params_tab.setCurrentIndex(1)

        qtbot.wait_until(lambda: manager.text_box.toHtml() != old_html, timeout=5000)
        assert "spline" in manager.text_box.toPlainText().lower()


def test_manager_restool_child_side_panel(
    qtbot,
    gold,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(gold, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child_uid = manager.add_childtool(
            ResolutionTool(gold.copy(deep=True), data_name="gold_input"),
            0,
            show=False,
        )

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)

        qtbot.wait_until(
            lambda: "restool" in manager.text_box.toHtml().lower(), timeout=5000
        )
        assert "restool" in manager.text_box.toPlainText().lower()
        assert manager.preview_widget.isVisible()
        assert not manager_preview_pixmap(manager).isNull()
        assert metadata_detail_map(manager)["Kind"] == "restool"


def test_manager_restool_child_side_panel_live_refresh(
    qtbot,
    gold,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(gold, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child_uid = manager.add_childtool(
            ResolutionTool(gold.copy(deep=True), data_name="gold_input"),
            0,
            show=False,
        )
        child = manager.get_childtool(child_uid)
        assert isinstance(child, ResolutionTool)

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)

        qtbot.wait_until(
            lambda: "restool" in manager.text_box.toHtml().lower(), timeout=5000
        )
        old_html = manager.text_box.toHtml()
        step = max(child.x0_spin.singleStep(), 10**-child._x_decimals)
        new_value = min(child.x0_spin.value() + step, child.x1_spin.value())
        if new_value == child.x0_spin.value():
            new_value = max(child._x_range[0], child.x0_spin.value() - step)
        child.x0_spin.setValue(new_value)

        qtbot.wait_until(lambda: manager.text_box.toHtml() != old_html, timeout=5000)


def test_manager_meshtool_child_side_panel(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child_uid = manager.add_childtool(
            MeshTool(test_data.copy(deep=True), data_name="mesh_input"),
            0,
            show=False,
        )

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)

        qtbot.wait_until(
            lambda: "meshtool" in manager.text_box.toHtml().lower(), timeout=5000
        )
        assert "meshtool" in manager.text_box.toPlainText().lower()
        assert manager.preview_widget.isVisible()
        assert not manager_preview_pixmap(manager).isNull()
        assert metadata_detail_map(manager)["Kind"] == "meshtool"


def test_manager_meshtool_child_side_panel_live_refresh(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child_uid = manager.add_childtool(
            MeshTool(test_data.copy(deep=True), data_name="mesh_input"),
            0,
            show=False,
        )
        child = manager.get_childtool(child_uid)
        assert isinstance(child, MeshTool)

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)

        qtbot.wait_until(
            lambda: "meshtool" in manager.text_box.toHtml().lower(), timeout=5000
        )
        old_html = manager.text_box.toHtml()
        child.order_spin.setValue(child.order_spin.value() + 1)

        qtbot.wait_until(lambda: manager.text_box.toHtml() != old_html, timeout=5000)


def test_manager_watched_1d_root_ftool_copy_code_omits_synthetic_squeeze(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(np.arange(5), dims=("x",), coords={"x": np.arange(5)})

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        manager._data_recv([data], {}, watched_var=("my_1d", "kernel-0"))
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_ftool()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[0]._childtool_indices) == 1,
            timeout=5000,
        )

        child_uid = manager._tool_graph.root_wrappers[0]._childtool_indices[0]
        child_tool = manager.get_childtool(child_uid)

        copied: list[str] = []
        monkeypatch.setattr(
            erlab.interactive.utils,
            "copy_to_clipboard",
            lambda text: copied.append(text) or text,
        )
        child_tool.copy_code()

        assert copied
        _assert_modelfit_code_replays_source(copied[-1], "my_1d", data)


def test_manager_watched_update_to_1d_refreshes_copy_code_cleanup(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    updated = xr.DataArray(np.arange(5), dims=("x",), coords={"x": np.arange(5)})

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        manager._data_recv([test_data], {}, watched_var=("my_data", "kernel-0"))
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        with qtbot.wait_signal(parent_tool.slicer_area.sigSourceDataReplaced):
            manager._data_watched_update("my_data", "kernel-0", updated)

        parent_tool.slicer_area.images[0].open_in_ftool()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[0]._childtool_indices) == 1,
            timeout=5000,
        )

        child_uid = manager._tool_graph.root_wrappers[0]._childtool_indices[0]
        child_tool = manager.get_childtool(child_uid)

        copied: list[str] = []
        monkeypatch.setattr(
            erlab.interactive.utils,
            "copy_to_clipboard",
            lambda text: copied.append(text) or text,
        )
        child_tool.copy_code()

        assert copied
        _assert_modelfit_code_replays_source(copied[-1], "my_data", updated)


def test_manager_duplicate_watched_1d_root_preserves_copy_code_cleanup(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(np.arange(5), dims=("x",), coords={"x": np.arange(5)})

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        manager._data_recv([data], {}, watched_var=("my_1d", "kernel-0"))
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        duplicated = manager.duplicate_imagetool(0)
        assert isinstance(duplicated, int)

        parent_tool = manager.get_imagetool(duplicated)
        parent_tool.slicer_area.images[0].open_in_ftool()
        qtbot.wait_until(
            lambda: (
                len(manager._tool_graph.root_wrappers[duplicated]._childtool_indices)
                == 1
            ),
            timeout=5000,
        )

        child_uid = manager._tool_graph.root_wrappers[duplicated]._childtool_indices[0]
        child_tool = manager.get_childtool(child_uid)

        copied: list[str] = []
        monkeypatch.setattr(
            erlab.interactive.utils,
            "copy_to_clipboard",
            lambda text: copied.append(text) or text,
        )
        child_tool.copy_code()

        assert copied
        _assert_modelfit_code_replays_source(copied[-1], "my_1d", data)


def test_manager_workspace_roundtrip_watched_1d_root_preserves_copy_code_cleanup(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(np.arange(5), dims=("x",), coords={"x": np.arange(5)})

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        manager._data_recv([data], {}, watched_var=("my_1d", "kernel-0"))
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        tree = manager._to_datatree()
        assert tree["0/imagetool"].attrs["manager_node_source_input_ndim"] == 1

        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        for node in tree.values():
            manager._load_workspace_node(typing.cast("xr.DataTree", node))

        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_ftool()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[0]._childtool_indices) == 1,
            timeout=5000,
        )

        child_uid = manager._tool_graph.root_wrappers[0]._childtool_indices[0]
        child_tool = manager.get_childtool(child_uid)

        copied: list[str] = []
        monkeypatch.setattr(
            erlab.interactive.utils,
            "copy_to_clipboard",
            lambda text: copied.append(text) or text,
        )
        child_tool.copy_code()

        assert copied
        _assert_modelfit_code_replays_source(copied[-1], "my_1d", data)
