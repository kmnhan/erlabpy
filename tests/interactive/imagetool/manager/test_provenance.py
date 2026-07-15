import ast
import enum
import json
import pathlib
import types
import typing
import warnings
from collections.abc import Callable, Sequence

import lmfit
import numpy as np
import pydantic
import pytest
import xarray
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

import erlab
import erlab.interactive.imagetool._highdim as imagetool_highdim
import erlab.interactive.imagetool.manager._details_panel as manager_details_panel
import erlab.interactive.imagetool.manager._dialogs as manager_dialogs
import erlab.interactive.imagetool.manager._lineage as manager_lineage
import erlab.interactive.imagetool.manager._mainwindow as manager_mainwindow
import erlab.interactive.imagetool.manager._provenance_edit as manager_provenance_edit
import erlab.interactive.imagetool.manager._widgets as manager_widgets
import erlab.interactive.imagetool.manager._workspace_io as manager_workspace_io
import erlab.interactive.imagetool.manager._wrapper as manager_wrapper
from erlab.interactive._fit2d import Fit2DTool
from erlab.interactive._mesh import MeshTool
from erlab.interactive.derivative import DerivativeTool
from erlab.interactive.fermiedge import GoldTool, ResolutionTool
from erlab.interactive.imagetool import _kspace_conversion, itool
from erlab.interactive.imagetool._provenance._code import uses_default_replay_input
from erlab.interactive.imagetool._provenance._execution import (
    replay_file_provenance,
    replay_script_provenance,
)
from erlab.interactive.imagetool._provenance._model import (
    DerivationEntry,
    FileDataSelection,
    FileLoadSource,
    FileReplayCall,
    ReplayStage,
    ScriptInput,
    ToolProvenanceOperation,
    ToolProvenanceSpec,
    _ProvenanceDisplayRow,
    _ProvenanceStepRef,
    compose_display_provenance,
    compose_full_provenance,
    file_load,
    full_data,
    operation_group_range,
    script,
    selection,
    stamp_operation_group,
    strip_operation_groups,
)
from erlab.interactive.imagetool._provenance._operations import (
    AffineCoordOperation,
    AssignAttrsOperation,
    AssignCoord1DOperation,
    AssignCoordsOperation,
    AssignScalarCoordOperation,
    AverageOperation,
    CoarsenOperation,
    DivideByCoordOperation,
    GaussianFilterOperation,
    InterpolationOperation,
    IselOperation,
    KspaceConvertOperation,
    KspaceSetNormalOperation,
    KspaceWorkFunctionOperation,
    LeadingEdgeOperation,
    NormalizeOperation,
    QSelAggregationOperation,
    QSelOperation,
    RenameDimsCoordsOperation,
    RenameOperation,
    RestoreNonuniformDimsOperation,
    RotateOperation,
    ScriptCodeOperation,
    SelOperation,
    SortByOperation,
    SwapDimsOperation,
    SymmetrizeNfoldOperation,
    SymmetrizeOperation,
    ThinOperation,
    TransposeOperation,
)
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
    return file_load(
        start_label="Load source",
        seed_code=f"derived = xr.load_dataarray({str(path)!r})",
        file_load_source=FileLoadSource(
            path=str(path),
            loader_label="xarray.load_dataarray",
            loader_text="xarray.load_dataarray",
            kwargs_text="",
            replay_call=FileReplayCall(
                kind="callable",
                target="xarray.load_dataarray",
                selected_index=0,
            ),
        ),
    )


def _manager_replay_file_spec(
    path: pathlib.Path,
    *operations: ToolProvenanceOperation,
) -> ToolProvenanceSpec:
    spec = file_load(
        start_label=f"Load data from file {path.name!r}",
        seed_code=(
            "import xarray\n\n"
            f"derived = xarray.load_dataarray({str(path)!r}, engine='h5netcdf')"
        ),
        file_load_source=FileLoadSource(
            path=str(path),
            loader_label="Load Function",
            loader_text="xarray.load_dataarray",
            kwargs_text="engine='h5netcdf'",
            replay_call=FileReplayCall(
                kind="callable",
                target="xarray.load_dataarray",
                kwargs={"engine": "h5netcdf"},
                selected_index=0,
            ),
        ),
    )
    if operations:
        spec = spec.append_replay_stage(full_data(*operations))
    return spec


def _add_file_replay_tool(
    manager: erlab.interactive.imagetool.manager.ImageToolManager,
    data: xr.DataArray,
    spec: ToolProvenanceSpec,
) -> erlab.interactive.imagetool.ImageTool:
    tool = itool(data, manager=False, execute=False)
    assert isinstance(tool, erlab.interactive.imagetool.ImageTool)
    manager.add_imagetool(tool, show=False, provenance_spec=spec)
    return tool


def _set_combo_data(combo: QtWidgets.QComboBox, data: object) -> None:
    index = combo.findData(data, QtCore.Qt.ItemDataRole.UserRole)
    assert index >= 0
    combo.setCurrentIndex(index)


def _selection_dialog_row(dialog: QtWidgets.QDialog, dim: str) -> typing.Any:
    assert isinstance(dialog, SelectionDialog)
    row = next(row for row in dialog.rows if row.dim == dim)
    row.use_check.setChecked(True)
    return row


def _set_selection_point(
    dialog: QtWidgets.QDialog,
    *,
    dim: str,
    method: str,
    value: float,
) -> None:
    row = _selection_dialog_row(dialog, dim)
    _set_combo_data(row.method_combo, method)
    _set_combo_data(row.kind_combo, "point")
    row.value_start_spin.setValue(value)


def _set_selection_range(
    dialog: QtWidgets.QDialog,
    *,
    dim: str,
    method: str,
    start: float,
    stop: float,
) -> None:
    row = _selection_dialog_row(dialog, dim)
    _set_combo_data(row.method_combo, method)
    _set_combo_data(row.kind_combo, "range")
    row.value_start_spin.setValue(start)
    row.value_stop_spin.setValue(stop)


def _set_aggregate(
    dialog: QtWidgets.QDialog,
    *,
    dims: tuple[str, ...],
    func: str,
) -> None:
    assert isinstance(dialog, manager_provenance_edit.dialogs.AggregateDialog)
    for dim, check in dialog.dim_checks.items():
        check.setChecked(dim in dims)
    _set_combo_data(dialog.reducer_combo, func)


def _provenance_paste_test_data(name: str = "data") -> xr.DataArray:
    return xr.DataArray(
        np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5),
        dims=("x", "y", "z"),
        coords={"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0, 3.0]},
        name=name,
    )


def _set_provenance_steps_clipboard(
    operations: tuple[ToolProvenanceOperation, ...],
    *,
    active_name: str = "derived",
) -> None:
    payload_text = json.dumps(
        {
            "type": manager_details_panel._PROVENANCE_STEPS_CLIPBOARD_PAYLOAD_TYPE,
            "version": (
                manager_details_panel._PROVENANCE_STEPS_CLIPBOARD_PAYLOAD_VERSION
            ),
            "active_name": active_name,
            "operations": [
                operation.model_dump(mode="json") for operation in operations
            ],
        },
        separators=(",", ":"),
    )
    mime_data = QtCore.QMimeData()
    mime_data.setData(
        manager_details_panel._PROVENANCE_STEPS_CLIPBOARD_MIME,
        payload_text.encode("utf-8"),
    )
    QtWidgets.QApplication.clipboard().setMimeData(mime_data)


def _fit2d_param_result_dataset(params: typing.Any) -> xr.Dataset:
    params = params.copy()
    param_args = ", ".join(("x", *params.keys()))
    namespace = {"np": np}
    exec(  # noqa: S102
        f"def _model_func({param_args}):\n    return np.zeros_like(x, dtype=float)\n",
        namespace,
    )
    result = lmfit.model.ModelResult(
        lmfit.Model(namespace["_model_func"]),
        params,
        data=np.zeros(3),
        fcn_args=(np.arange(3, dtype=float),),
        max_nfev=1,
    )
    result.params = params.copy()
    result.nfev = 1
    return xr.Dataset({"modelfit_results": xr.DataArray(result, dims=())})


def _seed_fit2d_param_results(child: Fit2DTool, params_list: list[typing.Any]) -> None:
    child._params_full = [params.copy() for params in params_list]
    child._result_ds_full = [
        _fit2d_param_result_dataset(params) for params in params_list
    ]
    child._update_param_plot_options()


def test_file_load_edit_dialog_uses_loader_options_widget(qtbot) -> None:
    load_source = FileLoadSource(
        path="scan.h5",
        loader_label="Load Function",
        loader_text="xarray.load_dataarray",
        kwargs_text="engine='h5netcdf'",
        replay_call=FileReplayCall(
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


def test_file_load_edit_dialog_open_is_metadata_only(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_source = FileLoadSource(
        path="scan.h5",
        loader_label="Load Function",
        loader_text="xarray.load_dataarray",
        kwargs_text="engine='h5netcdf'",
        replay_call=FileReplayCall(
            kind="callable",
            target="xarray.load_dataarray",
            kwargs={"engine": "h5netcdf"},
            selected_index=0,
        ),
    )
    monkeypatch.setattr(
        manager_provenance_edit,
        "replay_file_provenance",
        lambda *_args, **_kwargs: pytest.fail(
            "opening file-load editor must not replay the file"
        ),
    )
    monkeypatch.setattr(
        manager_provenance_edit,
        "_load_provenance_from_file_details",
        lambda *_args, **_kwargs: pytest.fail(
            "opening file-load editor must not rebuild load provenance"
        ),
    )
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)

    dialog = manager_provenance_edit._FileLoadEditDialog(load_source, parent)
    qtbot.addWidget(dialog)

    assert dialog.file_path() == pathlib.Path("scan.h5")


def test_file_load_edit_dialog_allows_loader_change(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    example_loader,
) -> None:
    loader = example_loader()
    monkeypatch.setitem(erlab.io.loaders._loaders, loader.name, loader)
    monkeypatch.setitem(erlab.io.loaders._alias_mapping, loader.name, loader.name)
    file_path = tmp_path / "scan.h5"
    xr.DataArray(
        np.ones((2, 3)),
        dims=("ThetaX", "BindingEnergy"),
    ).to_netcdf(file_path, engine="h5netcdf")
    load_source = FileLoadSource(
        path=str(file_path),
        loader_label="Load Function",
        loader_text="xarray.load_dataarray",
        kwargs_text="engine='h5netcdf'",
        replay_call=FileReplayCall(
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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", erlab.io.dataloader.ValidationWarning)
        spec = dialog.provenance_spec(active_name="derived", replay_stages=())

    assert spec.file_load_source is not None
    replay_call = spec.file_load_source.replay_call
    assert replay_call is not None
    assert replay_call.kind == "erlab_loader"
    assert replay_call.target == "example"
    assert replay_call.kwargs == {"single": True}
    assert replay_call.selection == FileDataSelection(kind="dataarray")


def test_file_load_edit_dialog_updates_loader_options_for_replacement_path(
    qtbot,
    tmp_path: pathlib.Path,
) -> None:
    load_source = FileLoadSource(
        path=str(tmp_path / "scan.h5"),
        loader_label="Load Function",
        loader_text="xarray.load_dataarray",
        kwargs_text="engine='h5netcdf'",
        replay_call=FileReplayCall(
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
    multi_suffix_peer_spec = _manager_replay_file_spec(old_dir / "c.scan.h5")
    assert current_spec.file_load_source is not None
    peer_node = types.SimpleNamespace(uid="peer", display_text="Peer")
    multi_suffix_peer_node = types.SimpleNamespace(
        uid="peer-multi", display_text="Peer Multi"
    )
    peer = manager_provenance_edit._FileLoadBatchPeer(
        node=typing.cast("typing.Any", peer_node),
        scope="display",
        spec=peer_spec,
        original_path=old_dir / "b.h5",
        loader_summary="xarray.load_dataarray",
    )
    multi_suffix_peer = manager_provenance_edit._FileLoadBatchPeer(
        node=typing.cast("typing.Any", multi_suffix_peer_node),
        scope="display",
        spec=multi_suffix_peer_spec,
        original_path=old_dir / "c.scan.h5",
        loader_summary="xarray.load_dataarray",
    )
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    dialog = manager_provenance_edit._FileLoadEditDialog(
        current_spec.file_load_source,
        parent,
        batch_peers=(peer, multi_suffix_peer),
    )
    qtbot.addWidget(dialog)

    assert dialog.batch_apply_check.isEnabled()
    assert dialog.batch_peer_tree.isHidden()
    assert dialog.selected_batch_peers() == ()

    dialog.batch_apply_check.setChecked(True)
    item = dialog.batch_peer_tree.topLevelItem(0)
    multi_suffix_item = dialog.batch_peer_tree.topLevelItem(1)
    assert not dialog.batch_peer_tree.isHidden()
    assert item.checkState(0) == QtCore.Qt.CheckState.Checked
    assert multi_suffix_item.checkState(0) == QtCore.Qt.CheckState.Checked
    assert pathlib.Path(item.text(2)) == old_dir / "b.h5"
    assert pathlib.Path(multi_suffix_item.text(2)) == old_dir / "c.scan.h5"
    assert dialog.selected_batch_peers() == (peer, multi_suffix_peer)

    dialog.kwargs_edit.setText("engine='h5netcdf', chunks={'x': 1}")
    assert pathlib.Path(item.text(2)) == old_dir / "b.h5"
    assert pathlib.Path(multi_suffix_item.text(2)) == old_dir / "c.scan.h5"

    dialog.path_edit.setText(str(old_dir / "a.nc"))
    assert pathlib.Path(item.text(2)) == old_dir / "b.h5"
    assert pathlib.Path(multi_suffix_item.text(2)) == old_dir / "c.scan.h5"

    dialog.path_edit.setText(str(new_dir / "a.nc"))
    assert pathlib.Path(item.text(2)) == new_dir / "b.h5"
    assert pathlib.Path(multi_suffix_item.text(2)) == new_dir / "c.scan.h5"

    item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
    multi_suffix_item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
    assert dialog.selected_batch_peers() == ()


def test_file_load_edit_dialog_batch_manual_path_override(
    qtbot,
    tmp_path: pathlib.Path,
) -> None:
    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"
    manual_dir = tmp_path / "manual"
    old_dir.mkdir()
    new_dir.mkdir()
    manual_dir.mkdir()
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
        script_input_path=(1,),
    )
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    dialog = manager_provenance_edit._FileLoadEditDialog(
        current_spec.file_load_source,
        parent,
        batch_peers=(peer,),
        batch_apply_default=True,
    )
    qtbot.addWidget(dialog)

    item = dialog.batch_peer_tree.topLevelItem(0)
    assert not dialog.batch_peer_tree.isHidden()
    assert dialog.selected_batch_peers() == (peer,)

    manual_path = manual_dir / "explicit.h5"
    xr.DataArray(np.ones((2, 3)), dims=("x", "y")).to_netcdf(
        manual_path,
        engine="h5netcdf",
    )
    item.setText(2, str(manual_path))
    dialog.path_edit.setText(str(new_dir / "a.h5"))

    assert pathlib.Path(item.text(2)) == manual_path
    assert dialog.peer_provenance_spec(peer).file_load_source is not None
    assert pathlib.Path(dialog.peer_provenance_spec(peer).file_load_source.path) == (
        manual_path
    )


def test_file_load_edit_dialog_repair_checks_required_and_preserves_peer_loader(
    qtbot,
    tmp_path: pathlib.Path,
) -> None:
    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"
    old_dir.mkdir()
    new_dir.mkdir()
    current_spec = _manager_replay_file_spec(old_dir / "a.h5")
    required_spec = _manager_replay_file_spec(
        old_dir / "b.h5",
        IselOperation(kwargs={"x": 0}),
    )
    optional_spec = _manager_replay_file_spec(old_dir / "c.h5")
    assert current_spec.file_load_source is not None
    assert required_spec.file_load_source is not None
    required_replay_call = required_spec.file_load_source.replay_call
    assert required_replay_call is not None
    required_spec = required_spec.model_copy(
        update={
            "file_load_source": required_spec.file_load_source.model_copy(
                update={
                    "kwargs_text": "engine='scipy'",
                    "replay_call": required_replay_call.model_copy(
                        update={"kwargs": {"engine": "scipy"}},
                    ),
                }
            )
        }
    )
    peer_node = types.SimpleNamespace(uid="peer", display_text="Peer")
    required_peer = manager_provenance_edit._FileLoadBatchPeer(
        node=typing.cast("typing.Any", peer_node),
        scope="display",
        spec=required_spec,
        original_path=old_dir / "b.h5",
        loader_summary="xarray.load_dataarray (engine='scipy')",
        script_input_path=(1,),
        preserve_loader=True,
    )
    optional_peer = manager_provenance_edit._FileLoadBatchPeer(
        node=typing.cast("typing.Any", peer_node),
        scope="display",
        spec=optional_spec,
        original_path=old_dir / "c.h5",
        loader_summary="xarray.load_dataarray",
        script_input_path=(2,),
    )
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    dialog = manager_provenance_edit._FileLoadEditDialog(
        current_spec.file_load_source,
        parent,
        batch_peers=(required_peer, optional_peer),
        batch_apply_default=True,
        checked_batch_peer_ids=frozenset({required_peer.target_id}),
    )
    qtbot.addWidget(dialog)

    required_item = dialog.batch_peer_tree.topLevelItem(0)
    optional_item = dialog.batch_peer_tree.topLevelItem(1)
    assert required_item.checkState(0) == QtCore.Qt.CheckState.Checked
    assert optional_item.checkState(0) == QtCore.Qt.CheckState.Unchecked
    assert dialog.selected_batch_peers() == (required_peer,)

    new_b = new_dir / "manual-b.h5"
    required_item.setText(2, str(new_b))
    relinked = dialog.peer_provenance_spec(required_peer)

    assert relinked.active_name == required_spec.active_name
    assert relinked.replay_stages == required_spec.replay_stages
    assert relinked.file_load_source is not None
    relinked_replay_call = relinked.file_load_source.replay_call
    assert relinked_replay_call is not None
    assert pathlib.Path(relinked.file_load_source.path) == new_b
    assert relinked_replay_call.kwargs == {"engine": "scipy"}
    assert relinked.seed_code is not None
    assert str(new_b) in relinked.seed_code
    assert str(old_dir / "b.h5") not in relinked.seed_code


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
    dialog._batch_peer_item_changed(typing.cast("typing.Any", _MissingUidItem()), 2)


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
    with pytest.raises(RuntimeError, match="no longer replayable"):
        manager_provenance_edit._relinked_file_load_spec(
            stale_spec,
            pathlib.Path("current-b.h5"),
        )


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
        RotateOperation(
            angle=5.0,
            axes=("x", "y"),
            center=(0.0, 1.0),
        ),
        QSelOperation(kwargs={"x": 1.0}),
        IselOperation(kwargs={"x": 0}),
        SelOperation(kwargs={"x": 1.0}),
        SortByOperation(variables=("x",)),
        AverageOperation(dims=("x",)),
        QSelAggregationOperation(dims=("x",), func="mean"),
        InterpolationOperation(dim="x", values=[0.0, 1.0]),
        LeadingEdgeOperation(dim="x", fraction=0.5),
        CoarsenOperation(
            dim={"x": 2},
            boundary="trim",
            side="left",
            coord_func="mean",
            reducer="mean",
        ),
        ThinOperation(mode="global", factor=2),
        SymmetrizeOperation(dim="x", center=0.0),
        SymmetrizeNfoldOperation(
            fold=4,
            axes=("x", "y"),
            center={"x": 0.0, "y": 1.0},
        ),
        NormalizeOperation(dims=("x",), mode="area"),
        DivideByCoordOperation(coord_name="x"),
        GaussianFilterOperation(sigma={"x": 1.0}),
        SwapDimsOperation(mapping={"x": "kx"}),
        RenameDimsCoordsOperation(mapping={"x": "kx"}),
        AffineCoordOperation(coord_name="x", scale=1.0, offset=0.0),
        AssignCoordsOperation(coord_name="x", values=[0.0, 1.0]),
        AssignScalarCoordOperation(coord_name="temperature", value=20.0),
        AssignCoord1DOperation(
            coord_name="kx",
            dim="x",
            values=[0.0, 1.0],
        ),
        AssignAttrsOperation(attrs={"note": "edited"}),
    ],
)
def test_manager_provenance_structured_operations_have_edit_dialogs(
    operation: ToolProvenanceOperation,
) -> None:
    assert manager_provenance_edit._dialog_class_for_operation(operation) is not None


def test_manager_provenance_operation_editor_contract_is_valid() -> None:
    manager_provenance_edit._validate_operation_editor_contract()


def test_manager_provenance_editor_contract_rejects_group_without_matcher() -> None:
    class _MissingGroupMatcherDialog(
        manager_provenance_edit.dialogs.DataTransformDialog
    ):
        __module__ = manager_provenance_edit.dialogs.__name__
        operation_types = (ScriptCodeOperation,)
        grouped_operation_only = True

        def restore_transform_operations(
            self,
            operations: Sequence[ToolProvenanceOperation],
        ) -> None:
            del operations

    try:
        errors = manager_provenance_edit._operation_editor_contract_errors()
    finally:
        _MissingGroupMatcherDialog.operation_types = ()

    assert (
        "_MissingGroupMatcherDialog declares ScriptCodeOperation as a grouped editor "
        "but does not override operation_group_for_edit"
    ) in errors


def test_manager_provenance_editor_contract_rejects_mixed_editors() -> None:
    class _StandaloneScriptDialog(manager_provenance_edit.dialogs.DataTransformDialog):
        __module__ = manager_provenance_edit.dialogs.__name__
        operation_types = (ScriptCodeOperation,)

        def restore_transform_operation(
            self,
            operation: ToolProvenanceOperation,
        ) -> None:
            del operation

    class _GroupedScriptDialog(manager_provenance_edit.dialogs.DataTransformDialog):
        __module__ = manager_provenance_edit.dialogs.__name__
        operation_types = (ScriptCodeOperation,)
        grouped_operation_only = True

        @classmethod
        def operation_group_for_edit(
            cls,
            operations: Sequence[ToolProvenanceOperation],
            operation_index: int,
        ) -> tuple[int, int] | None:
            del cls, operations, operation_index
            return (0, 1)

        def restore_transform_operations(
            self,
            operations: Sequence[ToolProvenanceOperation],
        ) -> None:
            del operations

    try:
        errors = manager_provenance_edit._operation_editor_contract_errors()
    finally:
        _StandaloneScriptDialog.operation_types = ()
        _GroupedScriptDialog.operation_types = ()

    assert (
        "ScriptCodeOperation has both standalone and grouped editors: "
        "_StandaloneScriptDialog; _GroupedScriptDialog"
    ) in errors


def test_manager_provenance_editor_contract_rejects_ambiguous_editors() -> None:
    class _StandaloneScriptDialogA(manager_provenance_edit.dialogs.DataTransformDialog):
        __module__ = manager_provenance_edit.dialogs.__name__
        operation_types = (ScriptCodeOperation,)

        def restore_transform_operation(
            self,
            operation: ToolProvenanceOperation,
        ) -> None:
            del operation

    class _StandaloneScriptDialogB(manager_provenance_edit.dialogs.DataTransformDialog):
        __module__ = manager_provenance_edit.dialogs.__name__
        operation_types = (ScriptCodeOperation,)

        def restore_transform_operation(
            self,
            operation: ToolProvenanceOperation,
        ) -> None:
            del operation

    try:
        errors = manager_provenance_edit._operation_editor_contract_errors()
        with pytest.raises(RuntimeError, match="Multiple standalone"):
            manager_provenance_edit._standalone_editor_dialog_class_for_operation_type(
                ScriptCodeOperation
            )
    finally:
        _StandaloneScriptDialogA.operation_types = ()
        _StandaloneScriptDialogB.operation_types = ()

    assert (
        "ScriptCodeOperation has multiple standalone editors: "
        "_StandaloneScriptDialogA, _StandaloneScriptDialogB"
    ) in errors


def test_manager_provenance_editor_contract_rejects_multiple_grouped_editors() -> None:
    class _GroupedScriptDialogA(manager_provenance_edit.dialogs.DataTransformDialog):
        __module__ = manager_provenance_edit.dialogs.__name__
        operation_types = (ScriptCodeOperation,)
        grouped_operation_only = True

        @classmethod
        def operation_group_for_edit(
            cls,
            operations: Sequence[ToolProvenanceOperation],
            operation_index: int,
        ) -> tuple[int, int] | None:
            del cls, operations, operation_index
            return (0, 1)

        def restore_transform_operations(
            self,
            operations: Sequence[ToolProvenanceOperation],
        ) -> None:
            del operations

    class _GroupedScriptDialogB(manager_provenance_edit.dialogs.DataTransformDialog):
        __module__ = manager_provenance_edit.dialogs.__name__
        operation_types = (ScriptCodeOperation,)
        grouped_operation_only = True

        @classmethod
        def operation_group_for_edit(
            cls,
            operations: Sequence[ToolProvenanceOperation],
            operation_index: int,
        ) -> tuple[int, int] | None:
            del cls, operations, operation_index
            return (0, 1)

        def restore_transform_operations(
            self,
            operations: Sequence[ToolProvenanceOperation],
        ) -> None:
            del operations

    try:
        errors = manager_provenance_edit._operation_editor_contract_errors()
    finally:
        _GroupedScriptDialogA.operation_types = ()
        _GroupedScriptDialogB.operation_types = ()

    assert (
        "ScriptCodeOperation has multiple grouped editors: "
        "_GroupedScriptDialogA, _GroupedScriptDialogB"
    ) in errors


def test_manager_provenance_editor_contract_rejects_missing_editor() -> None:
    class _ScriptDialogWithoutRestore(
        manager_provenance_edit.dialogs.DataTransformDialog
    ):
        __module__ = manager_provenance_edit.dialogs.__name__
        operation_types = (ScriptCodeOperation,)

    class _ScriptFilterWithoutRestore(manager_provenance_edit.dialogs.DataFilterDialog):
        __module__ = manager_provenance_edit.dialogs.__name__
        operation_types = (ScriptCodeOperation,)

    try:
        errors = manager_provenance_edit._operation_editor_contract_errors()
        with pytest.raises(RuntimeError, match="Invalid ImageTool"):
            manager_provenance_edit._validate_operation_editor_contract()
    finally:
        _ScriptDialogWithoutRestore.operation_types = ()
        _ScriptFilterWithoutRestore.operation_types = ()

    assert "ScriptCodeOperation has no provenance editor" in errors


@pytest.mark.parametrize(
    "operation",
    [
        SelOperation(kwargs={"x": slice(0.0, 1.0)}),
        IselOperation(kwargs={"x": slice(0, 1)}),
        QSelOperation(kwargs={"x": 1.0}),
    ],
)
def test_manager_provenance_selection_operations_use_selection_dialog(
    operation: ToolProvenanceOperation,
) -> None:
    assert (
        manager_provenance_edit._dialog_class_for_operation(operation)
        is SelectionDialog
    )


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

    load_source = FileLoadSource(
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
        operation_types = (ScriptCodeOperation,)

    class _NoRestoreFilterDialog(manager_provenance_edit.dialogs.DataFilterDialog):
        operation_types = (ScriptCodeOperation,)

    assert (
        manager_provenance_edit._dialog_class_for_operation(
            ScriptCodeOperation(label="script", code="derived = data")
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
    script_input_can_reload: Callable[..., bool] | None = None,
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
        _selected_imagetool_targets=lambda: (),
        _node_for_target=lambda target: graph_nodes[target],
        _parent_node=_parent_node,
        _script_input_can_reload=(
            script_input_can_reload
            if script_input_can_reload is not None
            else lambda *_args, **_kwargs: True
        ),
        _rebuild_script_provenance=lambda spec, **_kwargs: types.SimpleNamespace(
            data=xr.DataArray([1.0], dims=("x",)),
            provenance_spec=spec,
        ),
        _ensure_script_provenance_trusted=lambda *_args, **_kwargs: None,
        _update_info=lambda **_kwargs: None,
    )
    return manager_provenance_edit._ProvenanceEditController(
        typing.cast("typing.Any", manager)
    )


def _fake_edit_node(
    spec: ToolProvenanceSpec | None,
    *,
    uid: str = "node",
    display_text: str = "Node",
    source_spec: ToolProvenanceSpec | None = None,
    source_display_spec: ToolProvenanceSpec | None = None,
    parent_uid: str | None = None,
    active_filter: ToolProvenanceOperation | None = None,
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        uid=uid,
        display_text=display_text,
        is_imagetool=True,
        imagetool=object(),
        pending_workspace_memory_payload=None,
        parent_uid=parent_uid,
        detached_live_parent_data=None,
        source_spec=source_spec,
        displayed_provenance_spec=spec,
        displayed_source_spec=source_display_spec,
        source_auto_update=True,
        materialize_pending_workspace_payload=lambda: True,
        slicer_area=types.SimpleNamespace(
            _accepted_filter_provenance_operation=active_filter
        ),
    )


def _trust_required_script_spec() -> ToolProvenanceSpec:
    return script(
        ScriptCodeOperation(
            label="User code",
            code="import os\nderived = data_0 + int(os.path.exists(os.devnull))",
        ),
        start_label="Run user code",
        active_name="derived",
        script_inputs=(ScriptInput(name="data_0", label="Input"),),
    )


def test_manager_selection_provenance_edit_restores_from_high_dimensional_source(
    qtbot: typing.Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_data = xr.DataArray(
        np.zeros((2, 3, 4, 5, 6)),
        dims=("scan", "eV", "kx", "ky", "temperature"),
    )
    spec = full_data(IselOperation(kwargs={"scan": 1}))
    node = _fake_edit_node(spec)
    node.detached_live_parent_data = parent_data

    manager = QtWidgets.QWidget()
    qtbot.addWidget(manager)
    manager._metadata_node_uid = "node"
    manager._tool_graph = types.SimpleNamespace(nodes={"node": node})
    manager._selected_imagetool_targets = lambda: ()
    manager._node_for_target = lambda target: manager._tool_graph.nodes[target]
    manager._parent_node = lambda _node: pytest.fail("parent data should be detached")
    manager._script_input_can_reload = lambda *_args, **_kwargs: True
    manager._rebuild_script_provenance = lambda *_args, **_kwargs: pytest.fail(
        "selection edit should not rebuild script provenance"
    )
    manager._ensure_script_provenance_trusted = lambda *_args, **_kwargs: None
    manager._update_info = lambda **_kwargs: None
    controller = manager_provenance_edit._ProvenanceEditController(
        typing.cast("typing.Any", manager)
    )

    monkeypatch.setattr(
        erlab.interactive.imagetool,
        "ImageTool",
        lambda *_args, **_kwargs: pytest.fail(
            "selection provenance edits should not create a temporary ImageTool"
        ),
    )

    captured: dict[str, typing.Any] = {}

    def exec_dialog(dialog: SelectionDialog) -> int:
        scan_row = next(row for row in dialog.rows if row.dim == "scan")
        captured["parent"] = dialog.parent()
        captured["dims"] = dialog.public_data.dims
        captured["scan_checked"] = scan_row.use_check.isChecked()
        captured["scan_method"] = scan_row.method
        captured["scan_index"] = int(scan_row.index_start_spin.value())
        return int(QtWidgets.QDialog.DialogCode.Accepted)

    monkeypatch.setattr(SelectionDialog, "exec", exec_dialog)

    row = spec.display_rows(parent_data=parent_data)[1]
    assert row.edit_ref is not None
    dialog_match = manager_provenance_edit._dialog_match_for_operation_ref(
        spec,
        row.edit_ref,
    )
    assert dialog_match is not None

    replacements = controller._edited_native_operations(
        node,
        row,
        spec,
        row.edit_ref,
        dialog_match,
    )

    assert captured == {
        "parent": manager,
        "dims": parent_data.dims,
        "scan_checked": True,
        "scan_method": "isel",
        "scan_index": 1,
    }
    assert replacements == [IselOperation(kwargs={"scan": 1})]


def test_manager_source_bound_derivation_rows_are_metadata_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def record_apply(
        self: ToolProvenanceOperation,
        data: xr.DataArray,
        *,
        parent_data: xr.DataArray,
    ) -> xr.DataArray:
        calls.append(self.op)
        return data

    monkeypatch.setattr(
        SymmetrizeNfoldOperation,
        "apply",
        record_apply,
    )
    monkeypatch.setattr(
        KspaceConvertOperation,
        "apply",
        record_apply,
    )
    source_spec = full_data(
        SymmetrizeNfoldOperation(fold=4, axes=("x", "y")),
        KspaceConvertOperation(),
    )
    parent = _fake_edit_node(full_data(), uid="parent")
    parent.current_source_data = lambda: pytest.fail(
        "metadata rendering must not compute parent source data"
    )
    child = _fake_edit_node(
        None,
        uid="child",
        parent_uid="parent",
        source_spec=source_spec,
        source_display_spec=source_spec,
    )
    child.manager = types.SimpleNamespace(_parent_node=lambda _node: parent)

    rows = manager_wrapper._ManagedWindowNode.derivation_display_rows.fget(child)

    assert calls == []
    assert rows is not None
    assert any(row.entry.label.startswith("Rotational Symmetrize") for row in rows)
    assert any(row.entry.label.startswith("Convert to momentum") for row in rows)


def test_manager_trusted_script_replay_prompt_is_session_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = types.SimpleNamespace(_trusted_script_replay_keys=set())
    controller = manager_lineage._LineageController(typing.cast("typing.Any", manager))
    spec = _trust_required_script_spec()
    prompts: list[str] = []
    monkeypatch.setattr(
        controller,
        "_prompt_trusted_script_replay",
        lambda _spec, *, reason: prompts.append(reason) or True,
    )

    controller._ensure_script_provenance_trusted(spec, reason="reload this result")
    controller._ensure_script_provenance_trusted(spec, reason="reload this result")
    changed_spec = spec.model_copy(
        update={
            "operations": (
                ScriptCodeOperation(
                    label="User code",
                    code=(
                        "import os\n"
                        "derived = data_0 + int(os.path.exists(os.devnull)) + 1"
                    ),
                ),
            ),
        }
    )
    controller._ensure_script_provenance_trusted(changed_spec, reason="reload")

    assert prompts == ["reload this result", "reload"]


def test_manager_trusted_script_replay_prompt_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = types.SimpleNamespace(_trusted_script_replay_keys=set())
    controller = manager_lineage._LineageController(typing.cast("typing.Any", manager))
    monkeypatch.setattr(
        controller,
        "_prompt_trusted_script_replay",
        lambda _spec, *, reason: False,
    )

    with pytest.raises(manager_widgets._TrustedScriptReplayCancelled):
        controller._ensure_script_provenance_trusted(
            _trust_required_script_spec(),
            reason="reload this result",
        )


def test_manager_trusted_script_replay_safe_and_prompt_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = types.SimpleNamespace(_trusted_script_replay_keys=set())
    controller = manager_lineage._LineageController(typing.cast("typing.Any", manager))
    safe_spec = script(
        ScriptCodeOperation(label="Offset", code="derived = data + 1"),
        start_label="Run safe script",
        seed_code="derived = data",
        active_name="derived",
    )
    monkeypatch.setattr(
        controller,
        "_prompt_trusted_script_replay",
        lambda *_args, **_kwargs: pytest.fail("safe replay should not prompt"),
    )

    controller._ensure_script_provenance_trusted(safe_spec, reason="reload")
    prompt_controller = manager_lineage._LineageController(
        typing.cast("typing.Any", manager)
    )
    detailed_texts: list[str] = []

    class _FakeMessageBox:
        class Icon(enum.IntEnum):
            Warning = 1

        class ButtonRole(enum.IntEnum):
            AcceptRole = 1

        class StandardButton(enum.IntEnum):
            Cancel = 1

        def __init__(self, _parent: typing.Any = None) -> None:
            self._run_button = object()
            self._cancel_button = object()

        def setObjectName(self, _name: str) -> None:
            pass

        def setIcon(self, _icon: enum.IntEnum) -> None:
            pass

        def setWindowTitle(self, _title: str) -> None:
            pass

        def setText(self, _text: str) -> None:
            pass

        def setInformativeText(self, _text: str) -> None:
            pass

        def setDetailedText(self, text: str) -> None:
            detailed_texts.append(text)

        def addButton(
            self, button: str | enum.IntEnum, _role: enum.IntEnum | None = None
        ) -> object:
            if button == "Run Code":
                return self._run_button
            return self._cancel_button

        def setDefaultButton(self, _button: typing.Any) -> None:
            pass

        def exec(self) -> None:
            pass

        def clickedButton(self) -> object:
            return self._run_button

    monkeypatch.setattr(manager_lineage.QtWidgets, "QMessageBox", _FakeMessageBox)
    assert prompt_controller._prompt_trusted_script_replay(
        safe_spec,
        reason="reload this result",
    )
    assert detailed_texts == ["derived = data\nderived = data + 1"]


def test_manager_provenance_lightweight_helper_edges() -> None:
    parent = types.SimpleNamespace(
        uid="parent",
        parent_uid=None,
        _childtool_indices=(),
        is_imagetool=True,
        type_badge_text="",
        name="",
    )
    child = types.SimpleNamespace(
        uid="child",
        parent_uid="parent",
        is_imagetool=True,
        type_badge_text="",
        name="",
    )
    panel = types.SimpleNamespace(
        _manager=types.SimpleNamespace(
            _tool_graph=types.SimpleNamespace(
                nodes={"parent": parent},
                root_wrappers={},
            )
        )
    )

    assert (
        manager_details_panel._DetailsPanelController._script_input_current_node_label(
            typing.cast("typing.Any", panel),
            child,
        )
        == "ImageTool child"
    )

    script_input_row = _ProvenanceDisplayRow(
        DerivationEntry("Use data_0 from Input", None),
        replay_ref=_ProvenanceStepRef(
            "script_input",
            script_input_index=0,
        ),
        script_input_path=(0,),
    )
    assert (
        manager_details_panel._DetailsPanelController._script_input_for_row(
            full_data(),
            script_input_row,
        )
        is None
    )
    assert (
        manager_details_panel._DetailsPanelController._script_input_for_row(
            script(
                start_label="Run script",
                active_name="derived",
                script_inputs=(ScriptInput(name="data_0", label="Input"),),
            ),
            script_input_row,
        )
        is None
    )

    forwarded: list[
        tuple[
            ToolProvenanceSpec,
            str,
            set[str] | None,
        ]
    ] = []
    spec = script(
        start_label="Run script",
        active_name="derived",
    )

    def ensure_script_provenance_trusted(
        spec_arg: ToolProvenanceSpec,
        *,
        reason: str,
        external_input_names: set[str] | None = None,
    ) -> None:
        forwarded.append((spec_arg, reason, external_input_names))

    manager = types.SimpleNamespace(
        _lineage_controller=types.SimpleNamespace(
            _ensure_script_provenance_trusted=ensure_script_provenance_trusted
        )
    )

    manager_mainwindow.ImageToolManager._ensure_script_provenance_trusted(
        typing.cast("typing.Any", manager),
        spec,
        reason="reload this result",
        external_input_names={"data_0"},
    )

    assert forwarded == [(spec, "reload this result", {"data_0"})]


def test_manager_trust_required_script_can_reload_and_rebuilds_trusted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _trust_required_script_spec()
    ensured: list[str] = []
    trusted_flags: list[bool] = []
    manager = types.SimpleNamespace(
        _script_input_can_reload=lambda *_args, **_kwargs: True,
        _ensure_script_provenance_trusted=lambda _spec, *, reason: ensured.append(
            reason
        ),
        _resolve_live_script_input_for_reload=lambda *_args, **_kwargs: None,
    )
    controller = manager_lineage._LineageController(typing.cast("typing.Any", manager))
    node = types.SimpleNamespace(
        is_imagetool=True,
        imagetool=object(),
        provenance_spec=spec,
        uid="node",
    )

    def rebuild_script_provenance(
        spec_arg: ToolProvenanceSpec,
        **kwargs: typing.Any,
    ) -> tuple[xr.DataArray, ToolProvenanceSpec]:
        trusted_flags.append(bool(kwargs["trusted_user_code"]))
        return xr.DataArray([2.0], dims=("x",)), spec_arg

    monkeypatch.setattr(
        manager_lineage,
        "rebuild_script_provenance",
        rebuild_script_provenance,
    )

    assert controller._node_can_reload_script_inputs(typing.cast("typing.Any", node))
    result = controller._rebuild_script_provenance(spec)

    assert ensured == ["reload this result"]
    assert trusted_flags == [True]
    xr.testing.assert_identical(result.data, xr.DataArray([2.0], dims=("x",)))


def test_manager_provenance_edit_file_load_helper_edges(tmp_path: pathlib.Path) -> None:
    source_path = tmp_path / "scan.nc"
    replacement_path = tmp_path / "replacement.nc"
    script_spec = script(
        start_label="Load script",
        seed_code=f"loaded = xr.load_dataarray({str(source_path)!r})",
        active_name="loaded",
        file_load_source=FileLoadSource(
            path=str(source_path),
            loader_label="xarray.load_dataarray",
            loader_text="xarray.load_dataarray",
            kwargs_text="",
            replay_call=FileReplayCall(
                kind="callable",
                target="xarray.load_dataarray",
                selected_index=0,
            ),
        ),
    )
    replacement = _manager_provenance_file_spec(replacement_path)

    assert manager_provenance_edit._file_load_edit_active_name(script_spec) == "loaded"
    replaced = manager_provenance_edit._replace_file_load_fields(
        script_spec,
        replacement,
    )
    assert replaced.kind == "script"
    assert replaced.start_label == replacement.start_label
    assert replaced.seed_code == replacement.seed_code
    assert replaced.file_load_source == replacement.file_load_source
    with pytest.raises(RuntimeError, match="not a file load"):
        manager_provenance_edit._replace_file_load_fields(
            script_spec,
            full_data(),
        )

    invalid_filename = FileNotFoundError()
    invalid_filename.filename = object()
    assert (
        manager_provenance_edit._file_not_found_path_from_exception(invalid_filename)
        is None
    )
    nested_missing = FileNotFoundError(2, "No such file", str(source_path))
    wrapper = RuntimeError("wrapped")
    wrapper.__cause__ = nested_missing
    assert (
        manager_provenance_edit._file_not_found_path_from_exception(wrapper)
        == source_path
    )


def test_manager_script_code_edit_dialog_uses_python_code_editor(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    dialog = manager_provenance_edit._ScriptCodeEditDialog(
        ScriptCodeOperation(
            label="Evaluate console expression",
            code="derived = data_0 + 1",
        ),
        parent,
    )
    qtbot.addWidget(dialog)

    code_edit = dialog.findChild(
        erlab.interactive.utils.PythonCodeEditor,
        "managerProvenanceScriptCodeEditor",
    )
    assert code_edit is not None
    assert isinstance(code_edit.highlighter, erlab.interactive.utils.PythonHighlighter)
    assert code_edit.lineWrapMode() == QtWidgets.QTextEdit.LineWrapMode.NoWrap

    warnings: list[str] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda *_args: warnings.append("warning"),
    )
    code_edit.setPlainText("derived =")
    dialog.accept()

    assert warnings == ["warning"]
    assert dialog.result() != int(QtWidgets.QDialog.DialogCode.Accepted)


def test_manager_edit_script_code_row_replaces_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = script(
        ScriptCodeOperation(
            label="Evaluate console expression",
            code="derived = data_0 + 1",
        ),
        start_label="Run ImageTool manager console code",
        active_name="derived",
        script_inputs=(ScriptInput(name="data_0", label="Input"),),
    )
    controller = _fake_edit_controller(_fake_edit_node(spec))
    candidates: list[ToolProvenanceSpec] = []

    class FakeDialog:
        def __init__(
            self,
            operation: ScriptCodeOperation,
            _parent: QtWidgets.QWidget,
        ) -> None:
            assert operation.code == "derived = data_0 + 1"

        def exec(self) -> int:
            return int(QtWidgets.QDialog.DialogCode.Accepted)

        def code(self) -> str:
            return "derived = data_0 + 2"

    monkeypatch.setattr(manager_provenance_edit, "_ScriptCodeEditDialog", FakeDialog)
    monkeypatch.setattr(
        controller,
        "_validate_and_replace",
        lambda _node, _scope, candidate, **_kwargs: candidates.append(candidate),
    )

    controller.edit_row(spec.display_rows()[2])

    assert len(candidates) == 1
    assert isinstance(candidates[0].operations[0], ScriptCodeOperation)
    assert candidates[0].operations[0].code == "derived = data_0 + 2"


def test_manager_edit_nested_script_code_row_replaces_nested_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nested = script(
        ScriptCodeOperation(label="Offset input", code="derived = data + 1"),
        start_label="Build nested",
        seed_code="data = xr.DataArray([1.0], dims=('x',))",
        active_name="derived",
    )
    spec = script(
        ScriptCodeOperation(label="Use nested", code="derived = data_0"),
        start_label="Run parent",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="data_0",
                label="Nested",
                provenance_spec=nested,
            ),
        ),
    )
    controller = _fake_edit_controller(_fake_edit_node(spec))
    candidates: list[ToolProvenanceSpec] = []

    class FakeDialog:
        def __init__(self, *_args: typing.Any) -> None:
            pass

        def exec(self) -> int:
            return int(QtWidgets.QDialog.DialogCode.Accepted)

        def code(self) -> str:
            return "derived = data + 3"

    monkeypatch.setattr(manager_provenance_edit, "_ScriptCodeEditDialog", FakeDialog)
    monkeypatch.setattr(
        controller,
        "_validate_and_replace",
        lambda _node, _scope, candidate, **_kwargs: candidates.append(candidate),
    )

    controller.edit_row(spec.display_rows()[1].children[1])

    nested_candidate = candidates[0].script_inputs[0].parsed_provenance_spec()
    assert nested_candidate is not None
    assert nested_candidate.operations[0].code == "derived = data + 3"


def test_manager_edit_live_script_code_row_replays_with_data_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_data = xr.DataArray([1.0, 2.0], dims=("x",))
    spec = full_data(
        ScriptCodeOperation(
            label="Live user code",
            code="derived = derived + 1",
        )
    )
    node = _fake_edit_node(spec)
    node.detached_live_parent_data = parent_data
    controller = _fake_edit_controller(node)
    applied: list[tuple[xr.DataArray, ToolProvenanceSpec]] = []

    class FakeDialog:
        def __init__(
            self,
            operation: ScriptCodeOperation,
            _parent: QtWidgets.QWidget,
        ) -> None:
            assert operation.code == "derived = derived + 1"

        def exec(self) -> int:
            return int(QtWidgets.QDialog.DialogCode.Accepted)

        def code(self) -> str:
            return "derived = derived + 3"

    monkeypatch.setattr(manager_provenance_edit, "_ScriptCodeEditDialog", FakeDialog)
    monkeypatch.setattr(
        controller,
        "_replace_node_data",
        lambda _node, _scope, data, candidate, _filter: applied.append(
            (data, candidate)
        ),
    )

    row = spec.display_rows()[1]
    assert controller.can_edit_row(row) == (True, "")
    controller.edit_row(row)

    assert len(applied) == 1
    xr.testing.assert_identical(applied[0][0], parent_data + 3)
    assert applied[0][1].kind == "full_data"
    assert applied[0][1].operations[0].code == "derived = derived + 3"


def test_manager_edit_script_code_trust_cancel_does_not_show_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = script(
        ScriptCodeOperation(label="User code", code="derived = data_0 + 1"),
        start_label="Run user code",
        active_name="derived",
        script_inputs=(ScriptInput(name="data_0", label="Input"),),
    )
    controller = _fake_edit_controller(_fake_edit_node(spec))
    failures: list[object] = []

    class FakeDialog:
        def __init__(self, *_args: typing.Any) -> None:
            pass

        def exec(self) -> int:
            return int(QtWidgets.QDialog.DialogCode.Accepted)

        def code(self) -> str:
            return "import os\nderived = data_0 + int(os.path.exists(os.devnull))"

    def cancel_validate(*_args: typing.Any, **_kwargs: typing.Any) -> None:
        raise manager_widgets._TrustedScriptReplayCancelled

    monkeypatch.setattr(manager_provenance_edit, "_ScriptCodeEditDialog", FakeDialog)
    monkeypatch.setattr(controller, "_validate_and_replace", cancel_validate)
    monkeypatch.setattr(controller, "_show_failed", lambda *args: failures.append(args))

    controller.edit_row(spec.display_rows()[2])

    assert failures == []


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
    focused = manager_provenance_edit._FileLoadBatchPeer(
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
    unavailable = manager_provenance_edit._FileLoadBatchPeer(
        node=typing.cast("typing.Any", node),
        scope="display",
        spec=unavailable_spec,
        original_path=old_dir / "unavailable.h5",
        loader_summary="xarray.load_dataarray",
        script_input_path=(1,),
    )
    available = manager_provenance_edit._FileLoadBatchPeer(
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
    focused = manager_provenance_edit._FileLoadBatchPeer(
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
    unavailable = manager_provenance_edit._FileLoadBatchPeer(
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
    different_loader = manager_provenance_edit._FileLoadBatchPeer(
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
            manager_provenance_edit._FileLoadBatchPeer(
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
            manager_provenance_edit._FileLoadBatchPeer(
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
            manager_provenance_edit._FileLoadBatchPeer(
                node=typing.cast("typing.Any", node),
                scope="display",
                spec=full_data(),
                original_path=pathlib.Path("scan.h5"),
                loader_summary="xarray.load_dataarray",
            )
        )

    stale_target = manager_provenance_edit._FileLoadBatchPeer(
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
    assert manager_provenance_edit._ProvenanceEditController._source_child_parent_row(
        typing.cast("typing.Any", node),
        nested_parent_row,
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
            batch_peers: tuple[manager_provenance_edit._FileLoadBatchPeer, ...],
            **_kwargs: typing.Any,
        ) -> None:
            assert batch_peers == peers

        def exec(self) -> int:
            return int(QtWidgets.QDialog.DialogCode.Accepted)

        def provenance_spec(
            self,
            *,
            active_name: str,
            replay_stages: tuple[ReplayStage, ...],
        ) -> ToolProvenanceSpec:
            del active_name, replay_stages
            return replacement_first

        def selected_batch_peers(
            self,
        ) -> tuple[manager_provenance_edit._FileLoadBatchPeer, ...]:
            return peers

        def peer_provenance_spec(
            self,
            peer: manager_provenance_edit._FileLoadBatchPeer,
        ) -> ToolProvenanceSpec:
            assert peer is peers[0]
            return replacement_second

    validated: list[ToolProvenanceSpec] = []
    applied: list[str] = []
    monkeypatch.setattr(manager_provenance_edit, "_FileLoadEditDialog", _Dialog)
    monkeypatch.setattr(
        controller,
        "_validated_edit",
        lambda edit_node, scope, candidate, **_kwargs: (
            validated.append(candidate)
            or manager_provenance_edit._ValidatedProvenanceEdit(
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
    peer = manager_provenance_edit._FileLoadBatchPeer(
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
            batch_peers: tuple[manager_provenance_edit._FileLoadBatchPeer, ...],
            **_kwargs: typing.Any,
        ) -> None:
            assert batch_peers == (peer,)

        def exec(self) -> int:
            return int(QtWidgets.QDialog.DialogCode.Accepted)

        def provenance_spec(
            self,
            *,
            active_name: str,
            replay_stages: tuple[ReplayStage, ...],
        ) -> ToolProvenanceSpec:
            del active_name, replay_stages
            return replacement_current

        def selected_batch_peers(
            self,
        ) -> tuple[manager_provenance_edit._FileLoadBatchPeer, ...]:
            return (peer,)

        def peer_provenance_spec(
            self,
            selected_peer: manager_provenance_edit._FileLoadBatchPeer,
        ) -> ToolProvenanceSpec:
            assert selected_peer is peer
            return replacement_peer

    validated: list[tuple[str, ToolProvenanceSpec]] = []
    monkeypatch.setattr(manager_provenance_edit, "_FileLoadEditDialog", _Dialog)
    monkeypatch.setattr(
        controller,
        "_validated_edit",
        lambda edit_node, scope, candidate, **_kwargs: (
            validated.append((edit_node.uid, candidate))
            or manager_provenance_edit._ValidatedProvenanceEdit(
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
            stage_index=0,
        ),
    )
    latest_stage_row = _ProvenanceDisplayRow(
        DerivationEntry("isel", None),
        replay_ref=_ProvenanceStepRef(
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
            _load_source: FileLoadSource,
            _parent: QtWidgets.QWidget,
            *,
            batch_peers: tuple[manager_provenance_edit._FileLoadBatchPeer, ...],
            **_kwargs: typing.Any,
        ) -> None:
            assert batch_peers == peers

        def exec(self) -> int:
            return int(QtWidgets.QDialog.DialogCode.Accepted)

        def provenance_spec(
            self,
            *,
            active_name: str,
            replay_stages: tuple[ReplayStage, ...],
        ) -> ToolProvenanceSpec:
            del active_name, replay_stages
            return current_spec

        def selected_batch_peers(
            self,
        ) -> tuple[manager_provenance_edit._FileLoadBatchPeer, ...]:
            return peers

        def peer_provenance_spec(
            self,
            peer: manager_provenance_edit._FileLoadBatchPeer,
        ) -> ToolProvenanceSpec:
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
        candidate: ToolProvenanceSpec,
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
            replay_stages: tuple[ReplayStage, ...],
        ) -> ToolProvenanceSpec:
            del active_name, replay_stages
            return current_spec

        def selected_batch_peers(
            self,
        ) -> tuple[manager_provenance_edit._FileLoadBatchPeer, ...]:
            return (batch_peer,)

        def peer_provenance_spec(
            self,
            peer: manager_provenance_edit._FileLoadBatchPeer,
        ) -> ToolProvenanceSpec:
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
    controller = _fake_edit_controller(_fake_edit_node(full_data()))
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
            stage_index=0,
        ),
        replay_ref=_ProvenanceStepRef(
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
            ToolProvenanceSpec,
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
        "_edited_native_operations",
        lambda *_args, **_kwargs: [IselOperation(kwargs={"x": 0})],
    )
    missing = manager_provenance_edit._MissingProvenanceSourceFileError(missing_path)
    failure = manager_provenance_edit._ProvenanceReplayFailure(
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
    assert opened[0].replay_stages == ()


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
        replay_stages=file_spec.replay_stages,
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
    missing = manager_provenance_edit._MissingProvenanceSourceFileError(
        tmp_path / "missing.h5",
    )
    exc = manager_provenance_edit._ProvenanceReplayFailure("repairing source", missing)
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
    missing = manager_provenance_edit._MissingProvenanceSourceFileError(
        old_dir / "a.h5"
    )
    exc = manager_provenance_edit._ProvenanceReplayFailure(
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
            batch_peers: tuple[manager_provenance_edit._FileLoadBatchPeer, ...],
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
    monkeypatch.setattr(manager_provenance_edit, "_FileLoadEditDialog", _Dialog)

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
    missing = manager_provenance_edit._MissingProvenanceSourceFileError(old_a_path)
    exc = manager_provenance_edit._ProvenanceReplayFailure(
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
            batch_peers: tuple[manager_provenance_edit._FileLoadBatchPeer, ...],
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
            replay_stages: tuple[ReplayStage, ...],
        ) -> ToolProvenanceSpec:
            del active_name, replay_stages
            return _manager_replay_file_spec(new_a_path)

        def selected_batch_peers(
            self,
        ) -> tuple[manager_provenance_edit._FileLoadBatchPeer, ...]:
            return self._batch_peers

        def peer_provenance_spec(
            self,
            peer: manager_provenance_edit._FileLoadBatchPeer,
        ) -> ToolProvenanceSpec:
            return manager_provenance_edit._relinked_file_load_spec(
                peer.spec,
                new_b_path,
            )

    monkeypatch.setattr(
        erlab.interactive.utils,
        "MessageDialog",
        _AcceptingMessageDialog,
    )
    monkeypatch.setattr(manager_provenance_edit, "_FileLoadEditDialog", _Dialog)
    monkeypatch.setattr(
        controller,
        "_validated_edit",
        lambda edit_node, scope, candidate, **_kwargs: (
            validated.append(candidate)
            or manager_provenance_edit._ValidatedProvenanceEdit(
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
    assert second_relinked.replay_stages == second_spec.replay_stages
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
    missing = manager_provenance_edit._MissingProvenanceSourceFileError(old_a_path)
    exc = manager_provenance_edit._ProvenanceReplayFailure(
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
            batch_peers: tuple[manager_provenance_edit._FileLoadBatchPeer, ...],
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
            replay_stages: tuple[ReplayStage, ...],
        ) -> ToolProvenanceSpec:
            del active_name, replay_stages
            return _manager_replay_file_spec(new_a_path)

        def selected_batch_peers(
            self,
        ) -> tuple[manager_provenance_edit._FileLoadBatchPeer, ...]:
            return ()

    def _validated_edit(
        _edit_node: typing.Any,
        _scope: typing.Literal["display", "source"],
        candidate: ToolProvenanceSpec,
        **_kwargs: typing.Any,
    ) -> manager_provenance_edit._ValidatedProvenanceEdit:
        validated.append(candidate)
        second_candidate = candidate.script_inputs[1].parsed_provenance_spec()
        assert second_candidate is not None
        assert second_candidate.file_load_source is not None
        assert pathlib.Path(second_candidate.file_load_source.path) == old_b_path
        still_missing = manager_provenance_edit._MissingProvenanceSourceFileError(
            old_b_path,
        )
        raise manager_provenance_edit._ProvenanceReplayFailure(
            "validating replacement",
            still_missing,
        ) from still_missing

    monkeypatch.setattr(erlab.interactive.utils, "MessageDialog", _MessageDialog)
    monkeypatch.setattr(manager_provenance_edit, "_FileLoadEditDialog", _Dialog)
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
    missing = manager_provenance_edit._MissingProvenanceSourceFileError(
        tmp_path / "missing.h5"
    )
    exc = manager_provenance_edit._ProvenanceReplayFailure(
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
    missing = manager_provenance_edit._MissingProvenanceSourceFileError(old_path)
    exc = manager_provenance_edit._ProvenanceReplayFailure(
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
            replay_stages: tuple[ReplayStage, ...],
        ) -> ToolProvenanceSpec:
            del active_name, replay_stages
            return new_file_spec

        def selected_batch_peers(
            self,
        ) -> tuple[manager_provenance_edit._FileLoadBatchPeer, ...]:
            return ()

    monkeypatch.setattr(
        erlab.interactive.utils,
        "MessageDialog",
        _AcceptingMessageDialog,
    )
    monkeypatch.setattr(manager_provenance_edit, "_FileLoadEditDialog", _Dialog)
    monkeypatch.setattr(
        controller,
        "_validated_edit",
        lambda edit_node, scope, candidate, **_kwargs: (
            validated.append(candidate)
            or manager_provenance_edit._ValidatedProvenanceEdit(
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
    missing = manager_provenance_edit._MissingProvenanceSourceFileError(
        tmp_path / "missing.h5",
    )
    exc = manager_provenance_edit._ProvenanceReplayFailure("replaying file", missing)
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
    controller = _fake_edit_controller(_fake_edit_node(full_data()))
    spec = _manager_provenance_file_spec(tmp_path / "missing.h5")

    monkeypatch.setattr(
        manager_provenance_edit,
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
        manager_provenance_edit,
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


def test_manager_provenance_native_transform_edit_mode_uses_dialog_operations(
    qtbot,
) -> None:
    data = xr.DataArray(
        np.arange(3 * 4 * 2, dtype=float).reshape((3, 4, 2)),
        dims=("x", "y", "z"),
        coords={"x": [0.0, 1.0, 2.0], "y": np.arange(4), "z": [0.0, 1.0]},
    )
    tool = erlab.interactive.imagetool.ImageTool(data)
    qtbot.addWidget(tool)
    dialog = manager_provenance_edit.dialogs.AggregateDialog(
        tool.slicer_area,
        provenance_edit_mode=True,
    )

    assert not hasattr(dialog, "launch_mode_combo")
    manager_provenance_edit._ProvenanceEditController._restore_native_edit_dialog(
        dialog,
        (QSelAggregationOperation(dims=("y",), func="mean"),),
        "dims",
    )
    _set_aggregate(dialog, dims=("x",), func="sum")

    assert dialog.provenance_edit_operations() == [
        QSelAggregationOperation(dims=("x",), func="sum")
    ]


def test_manager_provenance_native_edit_mode_uses_dialog_accept_validation(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = xr.DataArray(
        np.arange(3 * 4, dtype=float).reshape((3, 4)),
        dims=("x", "y"),
        coords={"x": np.arange(3), "y": np.arange(4)},
    )
    tool = erlab.interactive.imagetool.ImageTool(data)
    qtbot.addWidget(tool)
    dialog = manager_provenance_edit.dialogs.AggregateDialog(
        tool.slicer_area,
        provenance_edit_mode=True,
    )
    _set_aggregate(dialog, dims=("x", "y"), func="mean")

    warnings_shown: list[tuple[object, ...]] = []

    def _record_warning(*args: object) -> QtWidgets.QMessageBox.StandardButton:
        warnings_shown.append(args)
        return QtWidgets.QMessageBox.StandardButton.Ok

    def _unexpected_provenance_accept() -> None:
        raise AssertionError("dialog validation must run before edit acceptance")

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _record_warning)
    monkeypatch.setattr(
        dialog,
        "_accept_provenance_edit",
        _unexpected_provenance_accept,
    )

    dialog.buttonBox.accepted.emit()

    assert warnings_shown
    assert dialog.result() != int(QtWidgets.QDialog.DialogCode.Accepted)


@pytest.mark.parametrize("mode", ["empty", "error", "valid"])
def test_manager_provenance_base_edit_mode_accept_paths(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    data = xr.DataArray([1.0, 2.0], dims=("x",))
    tool = erlab.interactive.imagetool.ImageTool(data)
    qtbot.addWidget(tool)

    class _EditModeDialog(manager_provenance_edit.dialogs._DataManipulationDialog):
        def provenance_edit_operations(
            self,
        ) -> list[ToolProvenanceOperation]:
            if mode == "empty":
                return []
            if mode == "error":
                raise RuntimeError("operation failure")
            return [IselOperation(kwargs={"x": 0})]

    dialog = _EditModeDialog(tool.slicer_area, provenance_edit_mode=True)
    warnings_shown: list[tuple[object, ...]] = []
    critical_shown: list[tuple[object, ...]] = []

    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda *args: (
            warnings_shown.append(args) or QtWidgets.QMessageBox.StandardButton.Ok
        ),
    )
    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        lambda *args, **_kwargs: critical_shown.append(args),
    )

    dialog.accept()

    if mode == "valid":
        assert dialog.result() == int(QtWidgets.QDialog.DialogCode.Accepted)
        assert warnings_shown == []
        assert critical_shown == []
    else:
        assert dialog.result() != int(QtWidgets.QDialog.DialogCode.Accepted)
        assert bool(warnings_shown) is (mode == "empty")
        assert bool(critical_shown) is (mode == "error")


def test_manager_provenance_base_edit_operations_must_be_implemented(qtbot) -> None:
    data = xr.DataArray([1.0, 2.0], dims=("x",))
    tool = erlab.interactive.imagetool.ImageTool(data)
    qtbot.addWidget(tool)
    dialog = manager_provenance_edit.dialogs._DataManipulationDialog(tool.slicer_area)

    with pytest.raises(NotImplementedError):
        dialog.provenance_edit_operations()


def test_manager_provenance_filter_edit_mode_accept_skips_preview_apply(qtbot) -> None:
    data = xr.DataArray(np.arange(4.0), dims=("x",))
    tool = erlab.interactive.imagetool.ImageTool(data)
    qtbot.addWidget(tool)
    dialog = manager_provenance_edit.dialogs.NormalizeDialog(
        tool.slicer_area,
        provenance_edit_mode=True,
    )
    dialog.dim_checks["x"].setChecked(True)

    assert not hasattr(dialog, "preview_button")

    dialog.accept()

    assert dialog.result() == int(QtWidgets.QDialog.DialogCode.Accepted)
    assert tool.slicer_area._accepted_filter_provenance_operation is None


def test_manager_provenance_native_selection_edit_restores_slice_operations(
    qtbot,
) -> None:
    data = xr.DataArray(
        np.arange(3 * 4, dtype=float).reshape((3, 4)),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0, 2.0], "y": np.arange(4.0)},
    )
    tool = erlab.interactive.imagetool.ImageTool(data)
    qtbot.addWidget(tool)
    dialog = SelectionDialog(tool.slicer_area, provenance_edit_mode=True)

    assert not hasattr(dialog, "launch_mode_combo")
    manager_provenance_edit._ProvenanceEditController._restore_native_edit_dialog(
        dialog,
        (SelOperation(kwargs={"y": slice(1.0, 3.0)}),),
        None,
    )

    assert dialog.provenance_edit_operations() == [
        SelOperation(kwargs={"y": slice(1.0, 3.0)})
    ]


def _native_current_seed_data() -> xr.DataArray:
    return xr.DataArray(
        np.arange(12, dtype=float).reshape((3, 4)) + 1.0,
        dims=("x", "eV"),
        coords={
            "x": [0.0, 1.0, 2.0],
            "eV": [-1.0, 0.0, 1.0, 2.0],
            "scale": ("x", [1.0, 2.0, 4.0]),
            "order": ("x", [2.0, 1.0, 3.0]),
            "meta": "scan",
        },
    )


@pytest.mark.parametrize(
    ("operation", "dialog_cls"),
    [
        pytest.param(
            NormalizeOperation(
                dims=("x",),
                mode="minmax",
                denominator_rtol=1e-7,
            ),
            manager_provenance_edit.dialogs.NormalizeDialog,
            id="normalize",
        ),
        pytest.param(
            GaussianFilterOperation(sigma={"x": 0.25}),
            manager_provenance_edit.dialogs.GaussianFilterDialog,
            id="gaussian",
        ),
        pytest.param(
            DivideByCoordOperation(coord_name="scale"),
            manager_provenance_edit.dialogs.DivideByCoordDialog,
            id="divide_by_coord",
        ),
        pytest.param(
            SortByOperation(variables=("order",), ascending=False),
            manager_provenance_edit.dialogs.SortByDialog,
            id="sortby",
        ),
    ],
)
def test_manager_terminal_current_data_edit_opens_without_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    operation: ToolProvenanceOperation,
    dialog_cls: type[manager_provenance_edit.dialogs._DataManipulationDialog],
) -> None:
    base = _native_current_seed_data()
    current = operation.apply(base, parent_data=base)
    spec = _manager_replay_file_spec(tmp_path / "source.h5", operation)
    node = _fake_edit_node(spec)
    node.current_source_data = lambda: current
    controller = _fake_edit_controller(node)
    monkeypatch.setattr(
        controller,
        "_replay_candidate_result",
        lambda *_args, **_kwargs: pytest.fail("opening the editor should not replay"),
    )

    captured: dict[str, typing.Any] = {}

    def exec_dialog(
        dialog: manager_provenance_edit.dialogs._DataManipulationDialog,
    ) -> int:
        captured["dialog_cls"] = type(dialog)
        if isinstance(dialog, manager_provenance_edit.dialogs.NormalizeDialog):
            captured["dims"] = tuple(
                dim for dim, check in dialog.dim_checks.items() if check.isChecked()
            )
            captured["mode"] = dialog._mode
            captured["denominator_rtol"] = dialog.denominator_rtol
        elif isinstance(dialog, manager_provenance_edit.dialogs.GaussianFilterDialog):
            captured["dims"] = tuple(
                dim for dim, check in dialog.dim_checks.items() if check.isChecked()
            )
            captured["sigma"] = {
                dim: dialog._spin_value(dialog.sigma_spins[dim])
                for dim in dialog.sigma_spins
            }
        elif isinstance(dialog, manager_provenance_edit.dialogs.DivideByCoordDialog):
            captured["coord_name"] = dialog._selected_coord_name
        elif isinstance(dialog, manager_provenance_edit.dialogs.SortByDialog):
            captured["sort_keys"] = dialog._sort_keys
            captured["ascending"] = dialog.ascending_combo.currentData(
                QtCore.Qt.ItemDataRole.UserRole
            )
        return int(QtWidgets.QDialog.DialogCode.Rejected)

    monkeypatch.setattr(dialog_cls, "exec", exec_dialog)

    row = spec.display_rows()[1]
    assert row.edit_ref is not None
    dialog_match = manager_provenance_edit._dialog_match_for_operation_ref(
        spec,
        row.edit_ref,
    )
    assert dialog_match is not None

    assert (
        controller._edited_native_operations(
            node,
            row,
            spec,
            row.edit_ref,
            dialog_match,
        )
        is None
    )

    assert captured["dialog_cls"] is dialog_cls
    if isinstance(operation, NormalizeOperation):
        assert captured["dims"] == ("x",)
        assert captured["mode"] == "minmax"
        assert captured["denominator_rtol"] == pytest.approx(1e-7)
    elif isinstance(operation, GaussianFilterOperation):
        assert captured["dims"] == ("x",)
        assert captured["sigma"]["x"] == pytest.approx(0.25)
    elif isinstance(operation, DivideByCoordOperation):
        assert captured["coord_name"] == "scale"
    elif isinstance(operation, SortByOperation):
        assert captured["sort_keys"] == ("order",)
        assert captured["ascending"] is False


def test_manager_terminal_current_data_edit_accept_still_replays_for_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    base = _native_current_seed_data()
    operation = NormalizeOperation(
        dims=("x",),
        mode="minmax",
        denominator_rtol=1e-7,
    )
    current = operation.apply(base, parent_data=base)
    spec = _manager_replay_file_spec(tmp_path / "source.h5", operation)
    node = _fake_edit_node(spec)
    node.current_source_data = lambda: current
    controller = _fake_edit_controller(node)
    replayed: list[ToolProvenanceSpec] = []
    replaced: list[ToolProvenanceSpec] = []

    def replay_candidate_result(
        _node: typing.Any,
        _scope: typing.Literal["display", "source"],
        candidate: ToolProvenanceSpec,
    ) -> tuple[xr.DataArray, ToolProvenanceSpec]:
        replayed.append(candidate)
        return current, candidate

    monkeypatch.setattr(controller, "_replay_candidate_result", replay_candidate_result)
    monkeypatch.setattr(
        controller,
        "_replace_node_data",
        lambda _node, _scope, _data, candidate, _filter: replaced.append(candidate),
    )
    monkeypatch.setattr(
        manager_provenance_edit.dialogs.NormalizeDialog,
        "exec",
        lambda _dialog: int(QtWidgets.QDialog.DialogCode.Accepted),
    )

    row = spec.display_rows()[1]
    controller.edit_row(row)

    assert replayed == [spec]
    assert replaced == [spec]


def test_manager_terminal_current_data_edit_seed_rejects_grouped_operations(
    tmp_path: pathlib.Path,
) -> None:
    data = _native_current_seed_data()
    operation = NormalizeOperation(dims=("x",), mode="minmax")
    operations: tuple[ToolProvenanceOperation, ...] = (
        operation,
        SortByOperation(variables=("x",)),
    )
    spec = _manager_replay_file_spec(tmp_path / "source.h5", *operations)
    node = _fake_edit_node(spec)
    node.current_source_data = lambda: data
    controller = _fake_edit_controller(node)
    row = spec.display_rows()[1]
    assert row.edit_ref is not None

    seed = controller._terminal_current_data_edit_seed(
        node,
        spec,
        row.edit_ref,
        manager_provenance_edit._OperationDialogMatch(
            manager_provenance_edit.dialogs.NormalizeDialog,
            0,
            len(operations),
        ),
        operations,
    )

    assert seed is None


@pytest.mark.parametrize(
    ("operation", "dialog_cls", "current_data"),
    [
        pytest.param(
            NormalizeOperation(dims=("missing",), mode="minmax"),
            manager_provenance_edit.dialogs.NormalizeDialog,
            _native_current_seed_data(),
            id="normalize-missing-dim",
        ),
        pytest.param(
            GaussianFilterOperation(sigma={"missing": 0.25}),
            manager_provenance_edit.dialogs.GaussianFilterDialog,
            _native_current_seed_data(),
            id="gaussian-missing-dim",
        ),
        pytest.param(
            GaussianFilterOperation(sigma={"x": 0.25}),
            manager_provenance_edit.dialogs.GaussianFilterDialog,
            _native_current_seed_data().isel(x=slice(0, 1)),
            id="gaussian-degenerate-coord",
        ),
        pytest.param(
            GaussianFilterOperation(sigma={"x": 0.25}),
            manager_provenance_edit.dialogs.GaussianFilterDialog,
            _native_current_seed_data().assign_coords(x=["a", "b", "c"]),
            id="gaussian-nonnumeric-coord",
        ),
        pytest.param(
            DivideByCoordOperation(coord_name="missing"),
            manager_provenance_edit.dialogs.DivideByCoordDialog,
            _native_current_seed_data(),
            id="divide-by-missing-coord",
        ),
        pytest.param(
            DivideByCoordOperation(coord_name="label"),
            manager_provenance_edit.dialogs.DivideByCoordDialog,
            _native_current_seed_data().assign_coords(label=("x", ["a", "b", "c"])),
            id="divide-by-nonnumeric-coord",
        ),
        pytest.param(
            SortByOperation(variables=("missing",)),
            manager_provenance_edit.dialogs.SortByDialog,
            _native_current_seed_data(),
            id="sortby-missing-key",
        ),
    ],
)
def test_manager_terminal_current_data_edit_seed_rejects_invalid_metadata(
    tmp_path: pathlib.Path,
    operation: ToolProvenanceOperation,
    dialog_cls: type[manager_provenance_edit.dialogs._DataManipulationDialog],
    current_data: xr.DataArray,
) -> None:
    spec = _manager_replay_file_spec(tmp_path / "source.h5", operation)
    node = _fake_edit_node(spec)
    node.current_source_data = lambda: current_data
    controller = _fake_edit_controller(node)
    row = spec.display_rows()[1]
    assert row.edit_ref is not None

    seed = controller._terminal_current_data_edit_seed(
        node,
        spec,
        row.edit_ref,
        manager_provenance_edit._OperationDialogMatch(dialog_cls, 0, 1),
        (operation,),
    )

    assert seed is None


def test_manager_affine_coord_edit_opens_without_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    base = xr.DataArray(
        np.arange(6, dtype=float).reshape((2, 3)),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [10.0, 20.0, 30.0]},
    )
    operation = AffineCoordOperation(
        coord_name="y",
        scale=2.0,
        offset=0.5,
    )
    current = operation.apply(base, parent_data=base)
    spec = _manager_replay_file_spec(tmp_path / "source.h5", operation)
    node = _fake_edit_node(spec)
    node.current_source_data = lambda: current
    controller = _fake_edit_controller(node)
    monkeypatch.setattr(
        controller,
        "_replay_candidate_result",
        lambda *_args, **_kwargs: pytest.fail("opening the editor should not replay"),
    )

    captured: dict[str, typing.Any] = {}

    def exec_dialog(dialog: manager_provenance_edit.dialogs.AssignCoordsDialog) -> int:
        captured["coord_name"] = dialog.current_coord_name
        captured["scale"] = float(dialog.coord_widget.scale_spin.value())
        captured["offset"] = float(dialog.coord_widget.offset_spin.value())
        captured["reference_coord"] = dialog.coord_widget._old_coord.copy()
        captured["dialog_coord"] = dialog.slicer_area.data["y"].values.copy()
        return int(QtWidgets.QDialog.DialogCode.Rejected)

    monkeypatch.setattr(
        manager_provenance_edit.dialogs.AssignCoordsDialog,
        "exec",
        exec_dialog,
    )

    row = spec.display_rows()[1]
    assert row.edit_ref is not None
    dialog_match = manager_provenance_edit._dialog_match_for_operation_ref(
        spec,
        row.edit_ref,
    )
    assert dialog_match is not None

    assert (
        controller._edited_native_operations(
            node,
            row,
            spec,
            row.edit_ref,
            dialog_match,
        )
        is None
    )

    assert captured["coord_name"] == "y"
    assert captured["scale"] == 2.0
    assert captured["offset"] == 0.5
    np.testing.assert_allclose(captured["reference_coord"], base.y.values)
    np.testing.assert_allclose(captured["dialog_coord"], base.y.values)


def test_manager_affine_coord_edit_accept_still_replays_for_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    base = xr.DataArray(
        np.arange(6, dtype=float).reshape((2, 3)),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [10.0, 20.0, 30.0]},
    )
    operation = AffineCoordOperation(
        coord_name="y",
        scale=2.0,
        offset=0.5,
    )
    current = operation.apply(base, parent_data=base)
    spec = _manager_replay_file_spec(tmp_path / "source.h5", operation)
    node = _fake_edit_node(spec)
    node.current_source_data = lambda: current
    controller = _fake_edit_controller(node)
    replayed: list[ToolProvenanceSpec] = []
    replaced: list[ToolProvenanceSpec] = []

    def replay_candidate_result(
        _node: typing.Any,
        _scope: typing.Literal["display", "source"],
        candidate: ToolProvenanceSpec,
    ) -> tuple[xr.DataArray, ToolProvenanceSpec]:
        replayed.append(candidate)
        return current, candidate

    monkeypatch.setattr(controller, "_replay_candidate_result", replay_candidate_result)
    monkeypatch.setattr(
        controller,
        "_replace_node_data",
        lambda _node, _scope, _data, candidate, _filter: replaced.append(candidate),
    )
    monkeypatch.setattr(
        manager_provenance_edit.dialogs.AssignCoordsDialog,
        "exec",
        lambda _dialog: int(QtWidgets.QDialog.DialogCode.Accepted),
    )

    row = spec.display_rows()[1]
    controller.edit_row(row)

    assert replayed == [spec]
    assert replaced == [spec]


@pytest.mark.parametrize(
    ("operations", "current_data"),
    [
        pytest.param(
            (
                AffineCoordOperation(
                    coord_name="y",
                    scale=2.0,
                    offset=0.5,
                ),
                SortByOperation(variables=("y",)),
            ),
            xr.DataArray(
                np.arange(6, dtype=float).reshape((2, 3)),
                dims=("x", "y"),
                coords={"x": [0.0, 1.0], "y": [20.5, 40.5, 60.5]},
            ),
            id="grouped-operations",
        ),
        pytest.param(
            (NormalizeOperation(dims=("x",), mode="minmax"),),
            xr.DataArray(
                np.arange(6, dtype=float).reshape((2, 3)),
                dims=("x", "y"),
                coords={"x": [0.0, 1.0], "y": [10.0, 20.0, 30.0]},
            ),
            id="wrong-operation",
        ),
        pytest.param(
            (
                AffineCoordOperation(
                    coord_name="y",
                    scale=2.0,
                    offset=0.5,
                ),
            ),
            xr.DataArray(
                np.arange(6, dtype=float).reshape((2, 3)),
                dims=("x", "y"),
                coords={"x": [0.0, 1.0], "y": [10.0 + 0.0j, 20.0 + 0.0j, 30.0 + 0.0j]},
            ),
            id="complex-coordinate",
        ),
        pytest.param(
            (
                AffineCoordOperation(
                    coord_name="y",
                    scale=2.0,
                    offset=0.5,
                ),
            ),
            xr.DataArray(
                np.arange(6, dtype=float).reshape((2, 3)),
                dims=("x", "y"),
                coords={"x": [0.0, 1.0], "y": [10.0, np.inf, 30.0]},
            ),
            id="nonfinite-coordinate",
        ),
        pytest.param(
            (
                AffineCoordOperation(
                    coord_name="missing",
                    scale=2.0,
                    offset=0.5,
                ),
            ),
            xr.DataArray(
                np.arange(6, dtype=float).reshape((2, 3)),
                dims=("x", "y"),
                coords={"x": [0.0, 1.0], "y": [10.0, 20.0, 30.0]},
            ),
            id="missing-coordinate",
        ),
    ],
)
def test_manager_affine_coord_edit_seed_rejects_unsafe_current_data(
    tmp_path: pathlib.Path,
    operations: tuple[ToolProvenanceOperation, ...],
    current_data: xr.DataArray,
) -> None:
    spec = _manager_replay_file_spec(tmp_path / "source.h5", *operations)
    node = _fake_edit_node(spec)
    node.current_source_data = lambda: current_data
    controller = _fake_edit_controller(node)
    row = spec.display_rows()[1]
    assert row.edit_ref is not None

    seed = controller._terminal_affine_coord_edit_seed(
        node,
        spec,
        row.edit_ref,
        manager_provenance_edit._OperationDialogMatch(
            manager_provenance_edit.dialogs.AssignCoordsDialog,
            0,
            len(operations),
        ),
        operations,
    )

    assert seed is None


@pytest.mark.parametrize(
    ("operations", "current_data"),
    [
        pytest.param(
            (
                AffineCoordOperation(
                    coord_name="y",
                    scale=2.0,
                    offset=0.5,
                ),
                TransposeOperation(dims=("y", "x")),
            ),
            xr.DataArray(
                np.arange(6, dtype=float).reshape((2, 3)),
                dims=("x", "y"),
                coords={"x": [0.0, 1.0], "y": [20.5, 40.5, 60.5]},
            ).transpose("y", "x"),
            id="nonterminal",
        ),
        pytest.param(
            (
                AffineCoordOperation(
                    coord_name="y",
                    scale=0.0,
                    offset=0.5,
                ),
            ),
            xr.DataArray(
                np.arange(6, dtype=float).reshape((2, 3)),
                dims=("x", "y"),
                coords={"x": [0.0, 1.0], "y": [0.5, 0.5, 0.5]},
            ),
            id="zero-scale",
        ),
    ],
)
def test_manager_affine_coord_edit_falls_back_to_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    operations: tuple[ToolProvenanceOperation, ...],
    current_data: xr.DataArray,
) -> None:
    replay_data = xr.DataArray(
        np.arange(6, dtype=float).reshape((2, 3)),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [10.0, 20.0, 30.0]},
    )
    spec = _manager_replay_file_spec(tmp_path / "source.h5", *operations)
    node = _fake_edit_node(spec)
    node.current_source_data = lambda: current_data
    controller = _fake_edit_controller(node)
    replayed: list[ToolProvenanceSpec] = []

    def replay_candidate_result(
        _node: typing.Any,
        _scope: typing.Literal["display", "source"],
        candidate: ToolProvenanceSpec,
    ) -> tuple[xr.DataArray, ToolProvenanceSpec]:
        replayed.append(candidate)
        return replay_data, candidate

    monkeypatch.setattr(controller, "_replay_candidate_result", replay_candidate_result)
    monkeypatch.setattr(
        manager_provenance_edit.dialogs.AssignCoordsDialog,
        "exec",
        lambda _dialog: int(QtWidgets.QDialog.DialogCode.Rejected),
    )

    row = spec.display_rows()[1]
    assert row.edit_ref is not None
    dialog_match = manager_provenance_edit._dialog_match_for_operation_ref(
        spec,
        row.edit_ref,
    )
    assert dialog_match is not None

    assert (
        controller._edited_native_operations(
            node,
            row,
            spec,
            row.edit_ref,
            dialog_match,
        )
        is None
    )
    assert len(replayed) == 1


@pytest.mark.parametrize(
    "case",
    ["nonterminal", "current-source-unavailable", "leading-edge-missing-dim"],
)
def test_manager_terminal_current_data_edit_falls_back_to_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    case: str,
) -> None:
    replay_data = _native_current_seed_data()
    operation: ToolProvenanceOperation
    if case == "nonterminal":
        operation = NormalizeOperation(dims=("x",), mode="minmax")
        operations: tuple[ToolProvenanceOperation, ...] = (
            operation,
            TransposeOperation(dims=("eV", "x")),
        )
        current_data = operation.apply(replay_data, parent_data=replay_data).transpose(
            "eV",
            "x",
        )
        dialog_cls = manager_provenance_edit.dialogs.NormalizeDialog
    elif case == "current-source-unavailable":
        operation = NormalizeOperation(dims=("x",), mode="minmax")
        operations = (operation,)
        current_data = None
        dialog_cls = manager_provenance_edit.dialogs.NormalizeDialog
    else:
        operation = LeadingEdgeOperation(
            dim="eV",
            fraction=0.25,
            direction="negative",
        )
        operations = (operation,)
        current_data = operation.apply(replay_data, parent_data=replay_data)
        dialog_cls = manager_provenance_edit.dialogs.LeadingEdgeDialog

    spec = _manager_replay_file_spec(tmp_path / "source.h5", *operations)
    node = _fake_edit_node(spec)
    if current_data is None:
        node.current_source_data = lambda: (_ for _ in ()).throw(
            RuntimeError("missing current data")
        )
    else:
        node.current_source_data = lambda: current_data
    controller = _fake_edit_controller(node)
    replayed: list[ToolProvenanceSpec] = []

    def replay_candidate_result(
        _node: typing.Any,
        _scope: typing.Literal["display", "source"],
        candidate: ToolProvenanceSpec,
    ) -> tuple[xr.DataArray, ToolProvenanceSpec]:
        replayed.append(candidate)
        return replay_data, candidate

    monkeypatch.setattr(controller, "_replay_candidate_result", replay_candidate_result)
    monkeypatch.setattr(
        dialog_cls,
        "exec",
        lambda _dialog: int(QtWidgets.QDialog.DialogCode.Rejected),
    )

    row = spec.display_rows()[1]
    assert row.edit_ref is not None
    dialog_match = manager_provenance_edit._dialog_match_for_operation_ref(
        spec,
        row.edit_ref,
    )
    assert dialog_match is not None

    assert (
        controller._edited_native_operations(
            node,
            row,
            spec,
            row.edit_ref,
            dialog_match,
        )
        is None
    )
    assert len(replayed) == 1


def test_manager_provenance_edit_controller_native_dialog_error_branches() -> None:
    controller = _fake_edit_controller()
    operation = NormalizeOperation(dims=("x",), mode="area")
    spec = full_data(QSelAggregationOperation(dims=("x",), func="mean"))
    row = _ProvenanceDisplayRow(
        DerivationEntry("Aggregate", None),
    )
    ref = _ProvenanceStepRef("operation", operation_index=0)

    with pytest.raises(RuntimeError, match="Active display filter"):
        controller._edit_active_filter(
            typing.cast("typing.Any", _fake_edit_node(full_data())),
            operation,
            manager_provenance_edit.dialogs.AggregateDialog,
        )
    with pytest.raises(ValueError, match="No provenance operations"):
        controller._edited_native_operations(
            typing.cast("typing.Any", _fake_edit_node(spec)),
            row,
            spec,
            ref,
            manager_provenance_edit._OperationDialogMatch(
                manager_provenance_edit.dialogs.AggregateDialog,
                0,
                0,
            ),
        )


def test_manager_provenance_native_operation_editor_cancel_and_replay_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _fake_edit_controller()
    data = xr.DataArray(np.arange(4.0), dims=("x",))
    operation = IselOperation(kwargs={"x": 0})
    spec = full_data(operation)
    row = _ProvenanceDisplayRow(
        DerivationEntry("isel", None),
        edit_ref=_ProvenanceStepRef("operation", operation_index=0),
    )
    ref = typing.cast("_ProvenanceStepRef", row.edit_ref)
    dialog_match = manager_provenance_edit._OperationDialogMatch(
        SelectionDialog,
        0,
        1,
    )
    replayed_specs: list[ToolProvenanceSpec] = []

    def _replay_candidate_result(
        _node: typing.Any,
        _scope: typing.Literal["display", "source"],
        candidate: ToolProvenanceSpec,
    ) -> tuple[xr.DataArray, ToolProvenanceSpec]:
        replayed_specs.append(candidate)
        return data, candidate

    monkeypatch.setattr(
        controller,
        "_replay_candidate_result",
        _replay_candidate_result,
    )
    monkeypatch.setattr(
        QtWidgets.QDialog,
        "exec",
        lambda _dialog: int(QtWidgets.QDialog.DialogCode.Rejected),
    )

    assert (
        controller._edited_native_operations(
            typing.cast("typing.Any", _fake_edit_node(spec)),
            row,
            spec,
            ref,
            dialog_match,
        )
        is None
    )
    assert replayed_specs == [full_data()]

    def _raise_replay_cancelled(
        *_args: object,
        **_kwargs: object,
    ) -> tuple[xr.DataArray, ToolProvenanceSpec]:
        raise manager_provenance_edit._TrustedScriptReplayCancelled

    monkeypatch.setattr(
        controller,
        "_replay_candidate_result",
        _raise_replay_cancelled,
    )
    with pytest.raises(manager_provenance_edit._TrustedScriptReplayCancelled):
        controller._edited_native_operations(
            typing.cast("typing.Any", _fake_edit_node(spec)),
            row,
            spec,
            ref,
            dialog_match,
        )

    monkeypatch.setattr(
        controller,
        "_replay_candidate_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("replay failed")),
    )
    with pytest.raises(manager_provenance_edit._ProvenanceReplayFailure) as exc:
        controller._edited_native_operations(
            typing.cast("typing.Any", _fake_edit_node(spec)),
            row,
            spec,
            ref,
            dialog_match,
        )
    assert "opening the provenance editor" in str(exc.value)


def test_manager_provenance_restore_native_edit_dialog_rejects_bad_dialogs(
    qtbot,
) -> None:
    data = xr.DataArray(np.arange(4.0), dims=("x",))
    tool = erlab.interactive.imagetool.ImageTool(data)
    qtbot.addWidget(tool)
    filter_dialog = manager_provenance_edit.dialogs.NormalizeDialog(tool.slicer_area)
    base_dialog = manager_provenance_edit.dialogs._DataManipulationDialog(
        tool.slicer_area
    )
    filter_operation = NormalizeOperation(dims=("x",), mode="area")

    manager_provenance_edit._ProvenanceEditController._restore_native_edit_dialog(
        filter_dialog,
        (filter_operation,),
        None,
    )

    assert filter_dialog.provenance_edit_operations() == [filter_operation]

    with pytest.raises(ValueError, match="one operation"):
        manager_provenance_edit._ProvenanceEditController._restore_native_edit_dialog(
            filter_dialog,
            (
                NormalizeOperation(dims=("x",), mode="area"),
                NormalizeOperation(dims=("x",), mode="min"),
            ),
            None,
        )
    with pytest.raises(TypeError, match="transform or filter"):
        manager_provenance_edit._ProvenanceEditController._restore_native_edit_dialog(
            base_dialog,
            (IselOperation(kwargs={"x": 0}),),
            None,
        )


@pytest.mark.parametrize(
    "operation",
    [
        QSelOperation(kwargs={"x": slice(0.0, 2.0)}),
        SelOperation(kwargs={"y": slice(1.0, 3.0)}),
        IselOperation(kwargs={"y": slice(1, 3)}),
    ],
)
def test_manager_provenance_slice_selection_rows_remain_editable(
    operation: ToolProvenanceOperation,
) -> None:
    spec = selection(operation)
    controller = _fake_edit_controller(
        _fake_edit_node(
            full_data(),
            source_spec=spec,
            source_display_spec=spec,
            parent_uid="parent",
        )
    )
    row = spec.display_rows(scope="source")[1]

    assert controller.can_edit_row(row) == (True, "")


def test_manager_provenance_validation_preserves_active_filter_with_one_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _fake_edit_controller(_fake_edit_node(full_data()))
    data = xr.DataArray([1.0, 2.0], dims=("x",))
    candidate = full_data()
    base_candidate = full_data()
    filter_operation = NormalizeOperation(dims=("x",), mode="area")
    replayed_specs: list[ToolProvenanceSpec] = []

    def _replay_candidate_result(
        _node: typing.Any,
        _scope: typing.Literal["display", "source"],
        spec: ToolProvenanceSpec,
    ) -> tuple[xr.DataArray, ToolProvenanceSpec]:
        replayed_specs.append(spec)
        return data, spec

    monkeypatch.setattr(
        controller,
        "_replay_candidate_result",
        _replay_candidate_result,
    )
    monkeypatch.setattr(
        controller,
        "_split_active_filter",
        lambda _node, _spec: (base_candidate, filter_operation),
    )

    edit = controller._validated_edit(
        typing.cast("typing.Any", _fake_edit_node(full_data())),
        "display",
        candidate,
        where="validating edited filter",
    )

    assert replayed_specs == [base_candidate]
    assert edit.data is data
    assert edit.spec == base_candidate
    assert edit.filter_operation == filter_operation


def test_manager_provenance_filter_validation_uses_live_slicer_result() -> None:
    controller = _fake_edit_controller(_fake_edit_node(full_data()))
    data = xr.DataArray([1.0, 2.0], dims=("x",))
    operation = NormalizeOperation(dims=("x",), mode="area")
    calls: list[tuple[xr.DataArray, ToolProvenanceOperation]] = []
    node = types.SimpleNamespace(
        imagetool=object(),
        slicer_area=types.SimpleNamespace(
            _filter_operation_result_for_replacement=lambda data, operation: (
                calls.append((data, operation))
            )
        ),
    )

    controller._validate_filter_operation(
        typing.cast("typing.Any", node),
        data,
        operation,
        where="validating edited filter",
    )

    assert calls == [(data, operation)]


def test_manager_provenance_validation_reports_active_filter_validation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _fake_edit_controller(_fake_edit_node(full_data()))
    data = xr.DataArray([1.0], dims=("x",))
    candidate = full_data()
    base_candidate = full_data()
    calls = 0

    def _replay_candidate_result(
        _node: typing.Any,
        _scope: typing.Literal["display", "source"],
        spec: ToolProvenanceSpec,
    ) -> tuple[xr.DataArray, ToolProvenanceSpec]:
        nonlocal calls
        calls += 1
        return data, spec

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
            DivideByCoordOperation(coord_name="missing"),
        ),
    )

    with pytest.raises(manager_provenance_edit._ProvenanceReplayFailure) as exc:
        controller._validated_edit(
            typing.cast("typing.Any", _fake_edit_node(full_data())),
            "display",
            candidate,
            where="validating edited filter",
        )

    assert "active display filter" in str(exc.value)
    assert calls == 1


def test_manager_provenance_edit_controller_live_replay_and_replace() -> None:
    parent_data = xr.DataArray(
        np.arange(6, dtype=float).reshape((2, 3)), dims=("x", "y")
    )
    parent = types.SimpleNamespace(
        displayed_provenance_spec=full_data(),
        current_source_data=lambda: parent_data,
    )
    node = _fake_edit_node(
        full_data(),
        parent_uid="parent",
        source_display_spec=selection(),
    )
    replaced: list[tuple[xr.DataArray, ToolProvenanceSpec, bool, bool]] = []
    source_bindings: list[
        tuple[
            ToolProvenanceSpec,
            bool,
            str,
            ToolProvenanceSpec,
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
    spec = selection(IselOperation(kwargs={"x": 1}))

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
        NormalizeOperation(dims=("y",), mode="area"),
    )

    assert source_bindings
    assert source_bindings[-1][0] == spec
    assert source_bindings[-1][1:] == (
        True,
        "fresh",
        compose_display_provenance(
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
    active = NormalizeOperation(dims=("x",), mode="area")
    node = _fake_edit_node(full_data(), active_filter=active)

    assert (
        controller._active_filter_ref(
            typing.cast("typing.Any", _fake_edit_node(full_data())),
            full_data(active),
        )
        is None
    )

    live_spec = full_data(IselOperation(kwargs={"x": 0}), active)
    assert controller._active_filter_ref(
        typing.cast("typing.Any", node),
        live_spec,
    ) == _ProvenanceStepRef("operation", operation_index=1)

    file_spec = _manager_provenance_file_spec(
        pathlib.Path("scan.h5")
    ).append_replay_stage(full_data(active))
    assert controller._active_filter_ref(
        typing.cast("typing.Any", node),
        file_spec,
    ) == _ProvenanceStepRef(
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

    script_file_spec = script(
        start_label="Load source",
        seed_code=typing.cast("str", file_spec.seed_code),
        active_name="derived",
        file_load_source=file_spec.file_load_source,
        replay_stages=file_spec.replay_stages,
    )
    assert controller._active_filter_ref(
        typing.cast("typing.Any", node),
        script_file_spec,
    ) == _ProvenanceStepRef(
        "operation",
        operation_index=0,
        stage_index=0,
    )
    base_spec, split_operation = controller._split_active_filter(
        typing.cast("typing.Any", node),
        script_file_spec,
    )
    assert split_operation == active
    assert base_spec.kind == "script"
    assert base_spec.replay_stages == ()

    node.slicer_area._accepted_filter_provenance_operation = None
    assert controller._split_active_filter(
        typing.cast("typing.Any", node),
        live_spec,
    ) == (live_spec, None)


def test_tool_provenance_spec_row_reference_helpers_cover_edge_branches() -> None:
    isel = IselOperation(kwargs={"x": 0})
    sel = SelOperation(kwargs={"y": 1.0})
    spec = selection(isel, sel)
    start_ref = _ProvenanceStepRef("start")
    first_ref = _ProvenanceStepRef("operation", operation_index=0)
    missing_ref = _ProvenanceStepRef("operation", operation_index=20)
    script_input_ref = _ProvenanceStepRef(
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
    ).append_replay_stage(full_data(isel, sel))
    stage_ref = _ProvenanceStepRef(
        "operation",
        operation_index=1,
        stage_index=0,
    )
    assert file_spec._operation_for_ref(stage_ref) == sel
    assert (
        file_spec._operation_for_ref(
            _ProvenanceStepRef("operation", operation_index=1, stage_index=2)
        )
        is None
    )
    assert (
        file_spec._prefix_through_ref(_ProvenanceStepRef("file_load")).replay_stages
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


def test_manager_metadata_derivation_list_has_visible_splitter(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    rows = [
        _ProvenanceDisplayRow(DerivationEntry(f"Step {index}", "derived = data", True))
        for index in range(8)
    ]

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        assert manager.right_splitter.count() == 3
        assert manager.right_splitter.widget(2) is manager.metadata_group
        assert (
            manager.metadata_details_widget.parentWidget()
            is manager.metadata_details_page
        )
        assert (
            manager.metadata_derivation_list.parentWidget()
            is manager.metadata_provenance_page
        )
        assert not isinstance(
            manager.metadata_derivation_list.parentWidget(), QtWidgets.QSplitter
        )

        manager._set_metadata_node(
            typing.cast(
                "typing.Any",
                types.SimpleNamespace(
                    uid="node",
                    display_text="Node",
                    note="",
                    has_note=False,
                    is_imagetool=True,
                    type_badge_text="",
                    displayed_provenance_spec=full_data(),
                    metadata_fields=[
                        manager_wrapper._MetadataField("Kind", "ImageTool")
                    ],
                    derivation_display_rows=rows,
                ),
            )
        )

        assert manager.metadata_group.isVisible()
        assert manager.inspector_tabs.currentWidget() is manager.metadata_details_page
        assert manager.metadata_details_widget.isVisible()
        assert not manager.metadata_derivation_list.isVisible()
        manager.inspector_tabs.setCurrentWidget(manager.metadata_provenance_page)
        QtWidgets.QApplication.processEvents()
        assert manager.metadata_derivation_list.isVisible()
        handle = manager.right_splitter.handle(2)
        assert handle is not None
        qtbot.wait_until(handle.isVisible, timeout=5000)
        assert manager.metadata_derivation_list.minimumHeight() > 0
        assert (
            manager.metadata_derivation_list.maximumHeight()
            == manager_widgets._QWIDGETSIZE_MAX
        )

        manager.resize(640, 700)
        manager.right_splitter.setSizes([200, 160, 260])
        QtWidgets.QApplication.processEvents()
        assert manager.right_splitter.sizes()[2] > (
            manager.metadata_derivation_list.minimumHeight()
        )
        before_right_sizes = manager.right_splitter.sizes()
        before_list_height = manager.metadata_derivation_list.height()
        manager.right_splitter.moveSplitter(
            before_right_sizes[0] + before_right_sizes[1] - 40, 2
        )
        QtWidgets.QApplication.processEvents()
        after_right_sizes = manager.right_splitter.sizes()
        assert after_right_sizes[2] > before_right_sizes[2]
        assert manager.metadata_derivation_list.height() > before_list_height


def test_manager_file_label_helpers_and_file_replay_rename_update(tmp_path) -> None:
    paths = [
        tmp_path / "scan_a.h5",
        tmp_path / "scan_b.h5",
        tmp_path / "scan_c.h5",
    ]

    assert manager_wrapper._compact_file_suffix(paths) == " (scan_a, scan_b, +1)"

    spec = _manager_provenance_file_spec(paths[0]).append_replay_stage(
        full_data(AverageOperation(dims=("x",))).append_final_rename("old")
    )
    renamed = manager_wrapper._spec_with_final_data_name(spec, "new")

    assert renamed.kind == "file"
    assert renamed.replay_stages
    assert renamed.replay_stages[-1].operations[-1] == RenameOperation(name="new")
    assert renamed.replay_stages[-1].operations[:-1] == (AverageOperation(dims=("x",)),)

    script_spec = script(
        start_label="Load source",
        seed_code=typing.cast("str", spec.seed_code),
        active_name="derived",
        file_load_source=spec.file_load_source,
        replay_stages=spec.replay_stages,
    )
    script_renamed = manager_wrapper._spec_with_final_data_name(script_spec, "newer")

    assert script_renamed.kind == "script"
    assert script_renamed.replay_stages
    assert script_renamed.operations == ()
    script_stage_operations = script_renamed.replay_stages[-1].operations
    assert script_stage_operations[-1] == RenameOperation(name="newer")
    assert script_stage_operations[:-1] == (AverageOperation(dims=("x",)),)


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
    operation = GaussianFilterOperation(sigma={"alpha": 1.0})
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
    operation = GaussianFilterOperation(sigma={"alpha": 1.0})
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
            source_spec=full_data(),
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
    operation = GaussianFilterOperation(sigma={"alpha": 1.0})
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
            source_spec=full_data(),
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
    operation = GaussianFilterOperation(sigma={"x": 1.0})
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
            source_spec=full_data(),
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
    operation = GaussianFilterOperation(sigma={"alpha": 1.0})
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
            source_spec=full_data(),
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
    operation = GaussianFilterOperation(sigma={"alpha": 1.0})

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
            source_spec=full_data(),
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
        ) -> ToolProvenanceSpec | None:
            assert output_id == "out"
            del data
            return script(
                ScriptCodeOperation(label="Use output", code="result = data + 10"),
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
        operation = GaussianFilterOperation(sigma={"x": 1.0})
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
            provenance_spec: ToolProvenanceSpec,
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
            self, *, flush_deferred_restore: bool = True
        ) -> ToolProvenanceSpec | None:
            del flush_deferred_restore
            return self._provenance_spec

    data = xr.DataArray(np.arange(4.0), dims=("x",))
    provenance_spec = script(
        ScriptCodeOperation(label="Double data", code="result = data * 2"),
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
        parent_node = manager._tool_graph.root_wrappers[0]
        original_current_source_data = type(parent_node).current_source_data

        def fail_parent_current_source_data(self):
            if self is parent_node:
                raise AssertionError(
                    "metadata rendering must not compute parent source data"
                )
            return original_current_source_data(self)

        monkeypatch.setattr(
            type(parent_node),
            "current_source_data",
            fail_parent_current_source_data,
        )
        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)

        derivation = metadata_derivation_texts(manager)
        assert derivation[0] == "Start from selected parent ImageTool data"
        assert not any(line == "isel()" for line in derivation)
        assert not any("Sort coordinates" in line for line in derivation)
        assert any(line.startswith("transpose(") for line in derivation)

        monkeypatch.setattr(
            type(parent_node),
            "current_source_data",
            original_current_source_data,
        )
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
    initial_root_spec = selection(IselOperation(kwargs={"x": slice(0, 2)}))
    updated_root_spec = selection(IselOperation(kwargs={"x": slice(1, 3)}))

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
            source_spec=full_data(),
            source_auto_update=True,
        )

        grandchild_data = root_data.isel(y=slice(0, 2))
        grandchild_tool = itool(grandchild_data, manager=False, execute=False)
        assert isinstance(grandchild_tool, erlab.interactive.imagetool.ImageTool)
        grandchild_uid = manager.add_imagetool_child(
            grandchild_tool,
            child_uid,
            show=False,
            source_spec=selection(IselOperation(kwargs={"y": slice(0, 2)})),
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
        assert ".isel(x=slice(1, 3))" in code
        assert ".isel(x=slice(0, 2))" not in code
        namespace = {"data": base}
        exec(code, namespace)  # noqa: S102
        xr.testing.assert_identical(
            namespace["derived"],
            base.isel(x=slice(1, 3), y=slice(0, 2)),
        )


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
            provenance_spec=selection(IselOperation(kwargs={"x": slice(0, 2)})),
        )

        child_tool = itool(root_data.copy(deep=False), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=full_data(),
            source_auto_update=False,
        )
        child_node = manager._child_node(child_uid)

        updated = base.isel(x=slice(2, 4))
        manager._tool_graph.root_wrappers[0].set_detached_provenance(
            selection(IselOperation(kwargs={"x": slice(2, 4)}))
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
            selection(IselOperation(kwargs={"x": slice(4, 6)}))
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
            provenance_spec=selection(IselOperation(kwargs={"x": slice(0, 2)})),
        )

        child_tool = itool(root_data.copy(deep=False), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=full_data(),
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
            source_spec=selection(IselOperation(kwargs={"y": slice(0, 2)})),
            source_auto_update=True,
        )

        root_node = manager._tool_graph.root_wrappers[0]
        child_node = manager._child_node(child_uid)
        grandchild_node = manager._child_node(grandchild_uid)

        root_node.set_detached_provenance(
            selection(IselOperation(kwargs={"x": slice(1, 3)}))
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
            provenance_spec=selection(IselOperation(kwargs={"x": slice(0, 2)})),
        )

        child_tool = itool(root_data.copy(deep=False), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=full_data(),
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
            source_spec=selection(IselOperation(kwargs={"y": slice(0, 2)})),
            source_auto_update=False,
        )

        child_node = manager._child_node(child_uid)
        grandchild_node = manager._child_node(grandchild_uid)
        updated_root = base.isel(x=slice(2, 4))

        manager._tool_graph.root_wrappers[0].set_detached_provenance(
            selection(IselOperation(kwargs={"x": slice(2, 4)}))
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
        parent_tool.set_source_binding(full_data(), auto_update=False)

        leaf_tool = itool(root_data.isel(y=slice(0, 2)), manager=False, execute=False)
        assert isinstance(leaf_tool, erlab.interactive.imagetool.ImageTool)
        leaf_uid = manager.add_imagetool_child(
            leaf_tool,
            parent_uid,
            show=False,
            source_spec=selection(IselOperation(kwargs={"y": slice(0, 2)})),
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


def test_manager_batch_selection_replace_qsel_remains_editable(
    qtbot,
    accept_dialog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(3 * 4 * 2, dtype=float).reshape((3, 4, 2)),
        dims=("x", "y", "z"),
        coords={
            "x": [0.0, 1.0, 2.0],
            "y": np.arange(4, dtype=float),
            "z": np.arange(2, dtype=float),
        },
        name="scan",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        first = itool(data, manager=False, execute=False)
        second = itool(data + 100.0, manager=False, execute=False)
        assert isinstance(first, erlab.interactive.imagetool.ImageTool)
        assert isinstance(second, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(first, show=False)
        manager.add_imagetool(second, show=False)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        select_tools(manager, [0, 1])
        dialog = SelectionDialog(first.slicer_area, batch_manager=manager)
        row = dialog.rows[0]
        row.use_check.setChecked(True)
        row.value_start_spin.setValue(1.0)

        assert manager.apply_batch_transform_dialog(dialog, "replace")
        xarray.testing.assert_identical(
            first.slicer_area._data.rename(None),
            data.qsel(x=1.0).rename(None),
        )

        select_tools(manager, [0, 1])
        dialog = SelectionDialog(first.slicer_area, batch_manager=manager)
        row = dialog.rows[0]
        row.use_check.setChecked(True)
        row.kind_combo.setCurrentIndex(
            row.kind_combo.findData("range", QtCore.Qt.ItemDataRole.UserRole)
        )
        row.value_start_spin.setValue(1.0)
        row.value_stop_spin.setValue(3.0)

        assert manager.apply_batch_transform_dialog(dialog, "replace")
        xarray.testing.assert_identical(
            first.slicer_area._data.rename(None),
            data.qsel(x=1.0).qsel(y=slice(1.0, 3.0)).rename(None),
        )

        manager.tree_view.clearSelection()
        select_tools(manager, [0])
        manager._update_info()
        select_metadata_rows(manager, [1])
        selected_row = manager._selected_derivation_row()
        assert selected_row is not None
        assert manager._provenance_edit_controller.can_edit_row(selected_row) == (
            True,
            "",
        )

        def _edit_qsel(dialog: QtWidgets.QDialog) -> None:
            _set_selection_point(dialog, dim="x", method="qsel", value=2.0)

        accept_dialog(manager._edit_selected_derivation_step, pre_call=_edit_qsel)

        assert first.slicer_area._data.sizes["z"] == data.sizes["z"]
        xarray.testing.assert_identical(
            first.slicer_area._data.rename(None),
            data.qsel(x=2.0).qsel(y=slice(1.0, 3.0)).rename(None),
        )

        manager._update_info()
        select_metadata_rows(manager, [2])
        selected_row = manager._selected_derivation_row()
        assert selected_row is not None
        assert manager._provenance_edit_controller.can_edit_row(selected_row) == (
            True,
            "",
        )

        def _edit_qsel_range(dialog: QtWidgets.QDialog) -> None:
            _set_selection_range(
                dialog,
                dim="y",
                method="qsel",
                start=0.0,
                stop=2.0,
            )

        accept_dialog(
            manager._edit_selected_derivation_step,
            pre_call=_edit_qsel_range,
        )

        xarray.testing.assert_identical(
            first.slicer_area._data.rename(None),
            data.qsel(x=2.0).qsel(y=slice(0.0, 2.0)).rename(None),
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
            source_spec=selection(QSelOperation(kwargs={"alpha": 1, "alpha_width": 1})),
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
            script(
                ScriptCodeOperation(
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

        first_param_name, second_param_name = list(child._params.keys())[:2]
        params_full = []
        for index in range(len(child._params_full)):
            params = child._params.copy()
            params[first_param_name].set(value=1.0 + index)
            params[first_param_name].stderr = 0.01 + index
            params[second_param_name].set(value=10.0 + index)
            params[second_param_name].stderr = 0.1 + index
            params_full.append(params)
        _seed_fit2d_param_results(child, params_full)

        first_param_index = child.param_plot_combo.findText(first_param_name)
        second_param_index = child.param_plot_combo.findText(second_param_name)
        assert first_param_index >= 0
        assert second_param_index >= 0

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

        def selected_fit_output(code: str) -> tuple[str, str]:
            for call in (
                node for node in ast.walk(ast.parse(code)) if isinstance(node, ast.Call)
            ):
                if (
                    not isinstance(call.func, ast.Attribute)
                    or call.func.attr != "sel"
                    or not isinstance(call.func.value, ast.Attribute)
                    or call.func.value.attr
                    not in {"modelfit_coefficients", "modelfit_stderr"}
                ):
                    continue
                param_keyword = next(
                    (
                        keyword
                        for keyword in call.keywords
                        if keyword.arg == "param"
                        and isinstance(keyword.value, ast.Constant)
                        and isinstance(keyword.value.value, str)
                    ),
                    None,
                )
                if param_keyword is not None:
                    return call.func.value.attr, typing.cast(
                        "str", param_keyword.value.value
                    )
            raise AssertionError("generated fit code does not select a parameter")

        assert selected_fit_output(values_code) == (
            "modelfit_coefficients",
            first_param_name,
        )
        assert selected_fit_output(second_values_code) == (
            "modelfit_coefficients",
            second_param_name,
        )
        assert selected_fit_output(stderr_code) == (
            "modelfit_stderr",
            first_param_name,
        )


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
        ) -> ToolProvenanceSpec | None:
            assert output_id == "out"
            return script(
                ScriptCodeOperation(label="Use output", code="result = data + 10"),
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
        child.set_source_binding(full_data(), auto_update=False)

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
            load_func=(
                xr.load_dataarray,
                {"engine": "h5netcdf"},
                FileDataSelection(kind="dataarray"),
            ),
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
        assert not uses_default_replay_input(copied[-1])
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
            load_func=(
                xr.load_dataarray,
                {"engine": "h5netcdf"},
                FileDataSelection(kind="dataarray"),
            ),
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
        derivation_code = root.provenance_spec.derivation_code()
        assert derivation_code.count("derived =") == 1
        namespace = _exec_generated_code(
            derivation_code,
            {"data": data.copy(deep=True)},
        )
        derived = namespace["derived"]
        assert isinstance(derived, xr.DataArray)
        xr.testing.assert_identical(
            derived.rename(None), data.qsel.mean("x").rename(None)
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
    operation = GaussianFilterOperation(sigma={"x": 1.0})

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
            source_spec=full_data(),
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
            load_func=(
                xr.load_dataarray,
                {"engine": "h5netcdf"},
                FileDataSelection(kind="dataarray"),
            ),
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
            load_func=(
                xr.load_dataarray,
                {"engine": "h5netcdf"},
                FileDataSelection(kind="dataarray"),
            ),
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


def test_manager_provenance_nested_file_load_batch_relinks_deleted_parents(
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
    old_a_path = old_dir / "a.h5"
    old_b_path = old_dir / "b.h5"
    new_a_path = new_dir / "a.h5"
    new_b_path = new_dir / "b.h5"
    old_a = test_data.copy(deep=True)
    old_b = (test_data + 10).rename(test_data.name)
    new_a = (test_data + 100).rename(test_data.name)
    new_b = (test_data + 200).rename(test_data.name)
    new_a.to_netcdf(new_a_path, engine="h5netcdf")
    new_b.to_netcdf(new_b_path, engine="h5netcdf")

    first_spec = _manager_replay_file_spec(old_a_path)
    second_spec = _manager_replay_file_spec(old_b_path)
    root_spec = script(
        ScriptCodeOperation(
            label="Combine deleted parent tools",
            code="derived = data_0 + data_1",
        ),
        start_label="Run ImageTool manager console code",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="data_0",
                label="ImageTool 0: a",
                node_uid="deleted-a",
                node_snapshot_token=str(object()),
                provenance_spec=first_spec,
            ),
            ScriptInput(
                name="data_1",
                label="ImageTool 1: b",
                node_uid="deleted-b",
                node_snapshot_token=str(object()),
                provenance_spec=second_spec,
            ),
        ),
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        tool = _add_file_replay_tool(
            manager,
            old_a + old_b,
            root_spec,
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        root = manager._tool_graph.root_wrappers[0]

        select_tools(manager, [0])
        manager._update_info()
        select_metadata_rows(manager, [2])
        row = manager._selected_derivation_row()
        assert row is not None
        assert row.script_input_path == (0,)
        assert manager._provenance_edit_controller.can_edit_row(row)[0]

        def _edit_batch(dialog: QtWidgets.QDialog) -> None:
            dialog.path_edit.setText(str(new_a_path))  # type: ignore[attr-defined]
            dialog.batch_apply_check.setChecked(True)  # type: ignore[attr-defined]

        accept_dialog(manager._edit_selected_derivation_step, pre_call=_edit_batch)

        xr.testing.assert_identical(
            tool.slicer_area._data.rename(None),
            (new_a + new_b).astype(np.float64).rename(None),
        )
        assert root.provenance_spec is not None
        first_input, second_input = root.provenance_spec.script_inputs
        assert first_input.node_uid is None
        assert second_input.node_uid is None
        first_relinked = first_input.parsed_provenance_spec()
        second_relinked = second_input.parsed_provenance_spec()
        assert first_relinked is not None
        assert second_relinked is not None
        assert first_relinked.file_load_source is not None
        assert second_relinked.file_load_source is not None
        assert pathlib.Path(first_relinked.file_load_source.path) == new_a_path
        assert pathlib.Path(second_relinked.file_load_source.path) == new_b_path


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
    script_spec = script(
        ScriptCodeOperation(label="Copy source", code="derived = data"),
        QSelAggregationOperation(dims=("x",), func="mean"),
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
        QSelAggregationOperation(dims=("alpha",), func="mean"),
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        _add_file_replay_tool(manager, test_data.qsel.mean("alpha"), spec)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        select_tools(manager, [0])
        manager._update_info()
        activated_rows: list[_ProvenanceDisplayRow | None] = []
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
        manager.metadata_derivation_list.itemActivated.emit(item, 0)
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
        manager.metadata_derivation_list.itemActivated.emit(item, 0)
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
    spec = script(
        ScriptCodeOperation(label="Copy source", code="derived = data"),
        QSelAggregationOperation(dims=("x",), func="mean"),
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
        manager.metadata_derivation_list.itemActivated.emit(item, 0)


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
    operation = AssignAttrsOperation(attrs={"note": "selected"})
    spec = _manager_replay_file_spec(
        file_path,
        QSelAggregationOperation(dims=("alpha",), func="mean"),
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

        def _capture_menu(*, include_row_actions: bool = True) -> None:
            assert include_row_actions
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


def test_manager_provenance_context_menu_on_empty_space_keeps_paste(
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
        QSelAggregationOperation(dims=("alpha",), func="mean"),
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        _add_file_replay_tool(manager, test_data.qsel.mean("alpha"), spec)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        select_tools(manager, [0])
        manager._update_info()
        select_metadata_rows(manager, [1])
        _set_provenance_steps_clipboard(
            (AssignAttrsOperation(attrs={"copied": "yes"}),)
        )
        menu = manager._build_metadata_derivation_menu(include_row_actions=False)
        assert menu is not None
        assert manager._metadata_paste_steps_action in menu.actions()
        assert manager._metadata_paste_steps_action.isEnabled()
        assert not manager._metadata_edit_step_action.isEnabled()
        assert not manager._metadata_revert_step_action.isEnabled()
        assert not manager._metadata_copy_selected_action.isEnabled()
        assert not manager._metadata_delete_step_action.isEnabled()


def test_manager_provenance_paste_filter_respects_focus_guards(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = _provenance_paste_test_data()

    with manager_context() as manager:
        tool = itool(data, manager=False, execute=False)
        assert isinstance(tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(tool, show=False, provenance_spec=full_data())
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        assert manager._provenance_paste_filter is not None
        paste_event = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_V,
            QtCore.Qt.KeyboardModifier.ControlModifier,
        )
        paste_calls = 0

        def _record_paste() -> None:
            nonlocal paste_calls
            paste_calls += 1

        monkeypatch.setattr(
            manager,
            "_paste_provenance_steps_from_clipboard",
            _record_paste,
        )
        monkeypatch.setattr(
            manager._provenance_paste_filter,
            "_should_handle_paste",
            lambda: True,
        )

        assert manager._provenance_paste_filter.eventFilter(
            manager.tree_view, paste_event
        )
        assert paste_calls == 1

        paste_event = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_V,
            QtCore.Qt.KeyboardModifier.ControlModifier,
        )
        monkeypatch.setattr(
            manager._provenance_paste_filter,
            "_should_handle_paste",
            lambda: False,
        )
        assert not manager._provenance_paste_filter.eventFilter(
            manager.tree_view, paste_event
        )
        assert paste_calls == 1

        assert manager_mainwindow._widget_accepts_text_paste(
            manager.text_box, stop_at=manager
        )
        assert not manager_mainwindow._widget_accepts_text_paste(
            manager.tree_view, stop_at=manager
        )


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
        QSelAggregationOperation(dims=("alpha",), func="mean"),
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


def test_manager_provenance_unresolved_script_prefix_blocks_structured_edit(
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
    spec = script(
        ScriptCodeOperation(label="Copy source", code="derived = data"),
        QSelAggregationOperation(dims=("x",), func="mean"),
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
    spec = script(
        ScriptCodeOperation(
            label="Use source",
            code="derived = source.copy()",
        ),
        QSelAggregationOperation(dims=("x",), func="mean"),
        IselOperation(kwargs={"y": 0}),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="source",
                label="Recorded source",
                provenance_spec=file_spec,
            ),
        ),
    )
    initial = replay_script_provenance(spec, {})

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
            if row.replay_ref == _ProvenanceStepRef("operation", operation_index=1)
        )
        aggregate_item = None
        for row_index in range(manager.metadata_derivation_list.count()):
            item = manager.metadata_derivation_list.item(row_index)
            if (
                item is not None
                and item.data(manager_details_panel._METADATA_DERIVATION_ROW_ROLE)
                == aggregate_row
            ):
                aggregate_item = item
                break
        assert aggregate_item is not None
        select_metadata_rows(
            manager,
            [manager.metadata_derivation_list.row(aggregate_item)],
        )
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
        QSelAggregationOperation(dims=("y",), func="mean"),
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        tool = _add_file_replay_tool(
            manager,
            replay_file_provenance(spec),
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
            _set_aggregate(dialog, dims=("x",), func="sum")

        accept_dialog(manager._edit_selected_derivation_step, pre_call=_edit_aggregate)

        assert root.provenance_spec is not None
        stage = root.provenance_spec.replay_stages[0]
        assert stage.operations == (QSelAggregationOperation(dims=("x",), func="sum"),)
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
        spec = script(
            ScriptCodeOperation(
                label="Evaluate console expression",
                code="derived = data_0 + 1.0",
            ),
            QSelAggregationOperation(dims=("y",), func="mean"),
            start_label="Run ImageTool manager console code",
            active_name="derived",
            script_inputs=(
                ScriptInput(
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
        assert manager._provenance_edit_controller.can_edit_row(script_code_row) == (
            True,
            "",
        )
        assert manager._provenance_edit_controller.can_edit_row(structured_row) == (
            True,
            "",
        )
        select_metadata_rows(manager, [3])

        def _edit_aggregate(dialog: QtWidgets.QDialog) -> None:
            _set_aggregate(dialog, dims=("x",), func="sum")

        accept_dialog(manager._edit_selected_derivation_step, pre_call=_edit_aggregate)

        assert derived_node.provenance_spec is not None
        assert derived_node.provenance_spec.operations == (
            ScriptCodeOperation(
                label="Evaluate console expression",
                code="derived = data_0 + 1.0",
            ),
            QSelAggregationOperation(dims=("x",), func="sum"),
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
        operation = NormalizeOperation(dims=("x",), mode="area")
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
            NormalizeOperation(dims=("y",), mode="min")
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
        QSelAggregationOperation(dims=("x",), func="mean"),
        IselOperation(kwargs={"y": 0}),
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        initial = replay_file_provenance(spec)
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
            "_edited_native_operations",
            lambda *_args, **_kwargs: [
                QSelAggregationOperation(dims=("y",), func="mean")
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
            QSelAggregationOperation(dims=("x",), func="mean"),
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
            load_func=(
                xr.load_dataarray,
                {"engine": "h5netcdf"},
                FileDataSelection(kind="dataarray"),
            ),
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
        assert manager.reload_action.isVisible()
        assert manager.reload_action.isEnabled()
        assert str(file_path) in manager.reload_action.toolTip()


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
                "ToolProvenanceSpec",
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
        assert not uses_default_replay_input(copied[-1])
        full_result = full_namespace["derived"]
        assert isinstance(full_result, xr.DataArray)
        xr.testing.assert_identical(
            full_result.rename(None),
            data.qsel.mean("x").qsel.mean("y").rename(None),
        )

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
        detached_derivation = detached.provenance_spec.derivation_code()
        assert detached_derivation.count("derived =") == 1
        detached_namespace = _exec_generated_code(
            detached_derivation,
            {"data": updated.copy(deep=True)},
        )
        detached_result = detached_namespace["derived"]
        assert isinstance(detached_result, xr.DataArray)
        xr.testing.assert_identical(
            detached_result.rename(None),
            updated.qsel.mean("x").qsel.mean("y").rename(None),
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
    operation_payload = AverageOperation(dims=("z",)).model_dump(mode="json")
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
    assert payload[0] == (AverageOperation(dims=("z",)),)
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
        row: _ProvenanceDisplayRow | str,
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

    operation_ref = _ProvenanceStepRef("operation", operation_index=0)
    script_row_a = _ProvenanceDisplayRow(
        DerivationEntry("script a", None),
        replay_ref=operation_ref,
    )
    script_row_b = _ProvenanceDisplayRow(
        DerivationEntry("script b", None),
        replay_ref=operation_ref,
    )
    missing_operation_row = _ProvenanceDisplayRow(
        DerivationEntry("missing op", None),
        replay_ref=_ProvenanceStepRef(
            "operation",
            operation_index=10,
        ),
    )
    non_copyable_script_row = _ProvenanceDisplayRow(
        DerivationEntry("non-copyable script", None),
        replay_ref=operation_ref,
    )
    non_live_row = _ProvenanceDisplayRow(
        DerivationEntry("non-live", None),
        replay_ref=operation_ref,
    )
    row_to_spec = {
        script_row_a: script(
            ScriptCodeOperation(label="script a", code="a = data"),
            start_label="Start script a",
            active_name="a",
        ),
        script_row_b: script(
            ScriptCodeOperation(label="script b", code="b = data"),
            start_label="Start script b",
            active_name="b",
        ),
        missing_operation_row: full_data(AverageOperation(dims=("x",))),
        non_copyable_script_row: types.SimpleNamespace(
            _operation_for_ref=lambda _ref: ScriptCodeOperation(
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
        item(_ProvenanceDisplayRow(DerivationEntry("start", None))),
        item(
            _ProvenanceDisplayRow(
                DerivationEntry("file", None),
                replay_ref=_ProvenanceStepRef("file_load"),
            )
        ),
        item(script_row_a, copyable=False),
        item(script_row_a, code=None),
        item(
            _ProvenanceDisplayRow(
                DerivationEntry("no spec", None),
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


def test_manager_metadata_derivation_rows_render_as_tree(qtbot) -> None:
    child_row = _ProvenanceDisplayRow(
        DerivationEntry("Offset parent", "data_0 = data_0 + 1", True),
        edit_ref=_ProvenanceStepRef("operation", operation_index=0),
        replay_ref=_ProvenanceStepRef("operation", operation_index=0),
        script_input_path=(0,),
    )
    parent_row = _ProvenanceDisplayRow(
        DerivationEntry("Use data_0 from Parent", None, False),
        replay_ref=_ProvenanceStepRef(
            "script_input",
            script_input_index=0,
        ),
        children=(child_row,),
    )
    sibling_row = _ProvenanceDisplayRow(
        DerivationEntry("Use derived data", None, False),
        replay_ref=_ProvenanceStepRef("operation", operation_index=0),
    )
    derivation_list = manager_widgets._MetadataDerivationListWidget()
    qtbot.addWidget(derivation_list)
    manager = types.SimpleNamespace(
        metadata_derivation_list=derivation_list,
        _provenance_edit_controller=types.SimpleNamespace(
            can_edit_row=lambda row: (row is child_row, "")
        ),
        _set_metadata_fields=lambda _fields: None,
        _update_metadata_pane=lambda: None,
    )
    controller = manager_details_panel._DetailsPanelController(
        typing.cast("typing.Any", manager)
    )
    node = types.SimpleNamespace(
        uid="node",
        displayed_provenance_spec=full_data(),
        metadata_fields=[],
        derivation_display_rows=[parent_row, sibling_row],
    )

    controller._set_metadata_node(node)

    parent_item = derivation_list.topLevelItem(0)
    assert parent_item is not None
    assert parent_item.text() == "Use data_0 from Parent"
    parent_tree_item = typing.cast(
        "manager_widgets._MetadataDerivationTreeItem",
        parent_item,
    )
    parent_tree_item.setText("Use data_0 from Renamed Parent")
    assert parent_tree_item.text() == "Use data_0 from Renamed Parent"
    parent_tree_item.setText(0, "Use data_0 from Parent")
    assert parent_tree_item.text(0) == "Use data_0 from Parent"
    parent_tree_item.setData(QtCore.Qt.ItemDataRole.UserRole + 20, "one-arg")
    assert parent_tree_item.data(QtCore.Qt.ItemDataRole.UserRole + 20) == "one-arg"
    parent_tree_item.setData(0, QtCore.Qt.ItemDataRole.UserRole + 21, "two-arg")
    assert parent_tree_item.data(0, QtCore.Qt.ItemDataRole.UserRole + 21) == "two-arg"
    parent_tree_item.setToolTip("one-arg tooltip")
    assert parent_tree_item.toolTip() == "one-arg tooltip"
    parent_tree_item.setToolTip(0, "two-arg tooltip")
    assert parent_tree_item.toolTip(0) == "two-arg tooltip"
    parent_tree_item.setForeground(QtGui.QColor("red"))
    assert parent_tree_item.foreground().color() == QtGui.QColor("red")
    parent_tree_item.setForeground(0, QtGui.QColor("blue"))
    assert parent_tree_item.foreground(0).color() == QtGui.QColor("blue")
    with pytest.raises(TypeError):
        parent_tree_item.setText()
    with pytest.raises(TypeError):
        parent_tree_item.data()
    with pytest.raises(TypeError):
        parent_tree_item.setData(QtCore.Qt.ItemDataRole.UserRole)
    with pytest.raises(TypeError):
        parent_tree_item.setToolTip()
    with pytest.raises(TypeError):
        parent_tree_item.setForeground()

    assert parent_item.childCount() == 1
    child_item = parent_item.child(0)
    assert child_item is not None
    assert child_item.text() == "Offset parent"
    assert (
        child_item.data(manager_details_panel._METADATA_DERIVATION_ROW_ROLE)
        is child_row
    )
    sibling_item = derivation_list.topLevelItem(1)
    assert sibling_item is not None
    assert (
        sibling_item.data(manager_details_panel._METADATA_DERIVATION_ROW_ROLE)
        is sibling_row
    )
    assert derivation_list.count() == 3
    assert derivation_list.item(0) is parent_item
    assert derivation_list.item(1) is child_item
    assert derivation_list.item(2) is sibling_item
    assert derivation_list.item(3) is None
    assert derivation_list.row(parent_item) == 0
    assert derivation_list.row(child_item) == 1
    assert derivation_list.row(sibling_item) == 2
    assert derivation_list.display_order(child_item) == 1
    assert (
        derivation_list.display_order(
            manager_widgets._MetadataDerivationTreeItem("orphan")
        )
        == -1
    )
    derivation_list.setUniformItemSizes(True)
    assert derivation_list.uniformRowHeights()


def test_manager_metadata_script_input_labels_use_current_nodes(qtbot) -> None:
    source_spec = script(
        start_label="Build source",
        seed_code="derived = xr.DataArray([1.0], dims=('x',))",
        active_name="derived",
    )
    spec = script(
        ScriptCodeOperation(label="Copy", code="derived = data_10"),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="data_10",
                label="ImageTool 10: stale",
                node_uid="n16",
                provenance_spec=source_spec,
            ),
        ),
    )
    derivation_list = manager_widgets._MetadataDerivationListWidget()
    qtbot.addWidget(derivation_list)
    source_node = types.SimpleNamespace(
        uid="n16",
        parent_uid=None,
        is_imagetool=True,
        name="3 V",
        type_badge_text=None,
        _childtool_indices=[],
    )
    manager = types.SimpleNamespace(
        metadata_derivation_list=derivation_list,
        _tool_graph=types.SimpleNamespace(
            nodes={"n16": source_node},
            root_wrappers={4: source_node},
        ),
        _provenance_edit_controller=types.SimpleNamespace(
            can_edit_row=lambda _row: (False, "")
        ),
        _set_metadata_fields=lambda _fields: None,
        _update_metadata_pane=lambda: None,
    )
    controller = manager_details_panel._DetailsPanelController(
        typing.cast("typing.Any", manager)
    )
    node = types.SimpleNamespace(
        uid="node",
        displayed_provenance_spec=spec,
        metadata_fields=[],
        derivation_display_rows=spec.display_rows(),
    )

    controller._set_metadata_node(node)

    input_item = derivation_list.topLevelItem(1)
    assert input_item is not None
    assert input_item.text() == "Use data_10 from ImageTool 4: 3 V"
    assert "ImageTool 10" not in input_item.text()


def test_manager_metadata_missing_script_input_uses_neutral_label(qtbot) -> None:
    spec = script(
        ScriptCodeOperation(label="Copy", code="derived = data_10"),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="data_10",
                label="ImageTool 10: stale",
                node_uid="missing",
            ),
        ),
    )
    derivation_list = manager_widgets._MetadataDerivationListWidget()
    qtbot.addWidget(derivation_list)
    manager = types.SimpleNamespace(
        metadata_derivation_list=derivation_list,
        _tool_graph=types.SimpleNamespace(nodes={}, root_wrappers={}),
        _provenance_edit_controller=types.SimpleNamespace(
            can_edit_row=lambda _row: (False, "")
        ),
        _set_metadata_fields=lambda _fields: None,
        _update_metadata_pane=lambda: None,
    )
    controller = manager_details_panel._DetailsPanelController(
        typing.cast("typing.Any", manager)
    )
    node = types.SimpleNamespace(
        uid="node",
        displayed_provenance_spec=spec,
        metadata_fields=[],
        derivation_display_rows=spec.display_rows(),
    )

    controller._set_metadata_node(node)
    details = controller._unavailable_replay_code_details(
        typing.cast("typing.Any", node)
    )

    input_item = derivation_list.topLevelItem(1)
    assert input_item is not None
    assert input_item.text() == "Missing source for data_10"
    assert "ImageTool 10" not in input_item.text()
    assert "Missing source for data_10 (recorded as ImageTool 10: stale)" in details


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
        lambda: ((AverageOperation(dims=("x",)),), "derived", False),
    )
    monkeypatch.setattr(QtWidgets.QApplication, "clipboard", lambda: None)
    controller._copy_selected_derivation_code()


def test_manager_unavailable_replay_details_skip_replayable_script_inputs() -> None:
    source_spec = script(
        start_label="Build source",
        seed_code="derived = xr.DataArray([1.0], dims=('x',))",
        active_name="derived",
    )
    spec = script(
        ScriptCodeOperation(
            label="Run opaque code",
            code=None,
            copyable=False,
        ),
        start_label="Build figure",
        active_name="fig",
        script_inputs=(
            ScriptInput(
                name="data_3",
                label="ImageTool 3: D10cu",
                provenance_spec=source_spec,
            ),
        ),
    )
    node = types.SimpleNamespace(
        displayed_provenance_spec=spec,
        derivation_entries=spec.derivation_entries(),
        derivation_display_rows=spec.display_rows(),
    )
    controller = manager_details_panel._DetailsPanelController(
        typing.cast("typing.Any", types.SimpleNamespace())
    )

    details = controller._unavailable_replay_code_details(
        typing.cast("typing.Any", node)
    )

    assert "Run opaque code" in details
    assert "Use data_3 from ImageTool 3: D10cu" not in details


def test_manager_derivation_context_menu_ignores_empty_space(
    qtbot,
) -> None:
    list_widget = QtWidgets.QListWidget()
    qtbot.addWidget(list_widget)
    calls: list[bool] = []
    exec_positions: list[QtCore.QPoint] = []

    class _Menu:
        def exec(self, pos: QtCore.QPoint) -> None:
            exec_positions.append(pos)

    def build_menu(*, include_row_actions: bool) -> QtWidgets.QMenu:
        calls.append(include_row_actions)
        return typing.cast("QtWidgets.QMenu", _Menu())

    manager = types.SimpleNamespace(
        metadata_derivation_list=list_widget,
        _build_metadata_derivation_menu=build_menu,
    )
    controller = manager_details_panel._DetailsPanelController(
        typing.cast("typing.Any", manager)
    )

    controller._show_metadata_derivation_menu(QtCore.QPoint(10, 10))

    assert calls == [False]
    assert len(exec_positions) == 1


def test_manager_paste_steps_validation_error_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _fake_edit_controller(None, metadata_uid=None)
    unavailable: list[str] = []
    monkeypatch.setattr(controller, "_show_unavailable", unavailable.append)
    controller.paste_steps(
        (AverageOperation(dims=("x",)),),
        active_name="derived",
        contains_script=False,
    )
    assert unavailable

    node = _fake_edit_node(
        full_data(),
        parent_uid="parent",
        source_display_spec=full_data(),
    )
    with pytest.raises(TypeError, match="Only live provenance operations"):
        controller._paste_structured_steps(
            node,
            (ScriptCodeOperation(label="script", code="derived = data"),),
        )

    monkeypatch.setattr(
        controller,
        "_replay_candidate_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad replay")),
    )
    with pytest.raises(manager_provenance_edit._ProvenanceReplayFailure) as exc_info:
        controller._paste_structured_steps(
            node,
            (AverageOperation(dims=("x",)),),
        )
    assert "pasted provenance steps" in exc_info.value.where


def test_manager_paste_steps_targets_editable_imagetools_only() -> None:
    editable = _fake_edit_node(full_data(), uid="editable")
    pending = _fake_edit_node(full_data(), uid="pending")
    pending.imagetool = None
    pending.pending_workspace_memory_payload = (
        pathlib.Path("workspace.itws"),
        "0/imagetool",
    )
    unavailable = _fake_edit_node(full_data(), uid="unavailable")
    unavailable.imagetool = None
    non_imagetool = _fake_edit_node(full_data(), uid="non-imagetool")
    non_imagetool.is_imagetool = False
    graph_nodes = {
        editable.uid: editable,
        pending.uid: pending,
        unavailable.uid: unavailable,
        non_imagetool.uid: non_imagetool,
    }
    controller = _fake_edit_controller(
        editable,
        nodes=graph_nodes,
        metadata_uid=None,
    )
    controller._manager._selected_imagetool_targets = lambda: tuple(graph_nodes)

    assert controller._paste_target_nodes() == [editable, pending]
    assert controller.can_paste_steps(
        (AssignAttrsOperation(attrs={"copied": "yes"}),)
    ) == (True, "")


def test_manager_paste_detached_steps_uses_replay_spec_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _provenance_paste_test_data()
    local = full_data(AverageOperation(dims=("z",)))
    controller = _fake_edit_controller(None, metadata_uid=None)
    replaced: list[tuple[xr.DataArray, ToolProvenanceSpec, bool]] = []
    node = types.SimpleNamespace(
        uid="node",
        displayed_provenance_spec=full_data(),
        current_source_data=lambda: data,
        replace_with_detached_data=lambda data, spec, preserve_filter: replaced.append(
            (data, spec, preserve_filter)
        ),
    )
    monkeypatch.setattr(
        manager_provenance_edit,
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
    operation_ref = _ProvenanceStepRef(
        "operation",
        operation_index=0,
        stage_index=0,
    )
    row = _ProvenanceDisplayRow(
        DerivationEntry("Average", None),
        replay_ref=operation_ref,
    )

    controller = _fake_edit_controller(None, metadata_uid=None)
    deletable, reason = controller.can_delete_row(row)
    assert not deletable
    assert reason

    source_child = _fake_edit_node(
        full_data(AverageOperation(dims=("x",))),
        parent_uid="parent",
        source_spec=full_data(),
    )
    controller = _fake_edit_controller(source_child)
    deletable, reason = controller.can_delete_row(row)
    assert not deletable
    assert reason

    controller = _fake_edit_controller(_fake_edit_node(None))
    deletable, reason = controller.can_delete_row(row)
    assert not deletable
    assert reason

    missing_ref_row = _ProvenanceDisplayRow(
        DerivationEntry("Missing", None),
        replay_ref=_ProvenanceStepRef(
            "operation",
            operation_index=99,
            stage_index=0,
        ),
    )
    controller = _fake_edit_controller(
        _fake_edit_node(full_data(AverageOperation(dims=("x",))))
    )
    deletable, reason = controller.can_delete_row(missing_ref_row)
    assert not deletable
    assert reason

    broken_script_spec = types.SimpleNamespace(
        kind="script",
        _operation_for_ref=lambda _ref: ScriptCodeOperation(
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

    valid_script_spec = script(
        ScriptCodeOperation(
            label="Offset",
            code="derived = derived + 1",
        ),
        ScriptCodeOperation(
            label="Scale",
            code="derived = derived * 2",
        ),
        start_label="Start from data",
        seed_code="derived = data_0",
        active_name="derived",
        script_inputs=(ScriptInput(name="data_0", label="Input"),),
    )
    controller = _fake_edit_controller(_fake_edit_node(valid_script_spec))
    deletable, reason = controller.can_delete_row(valid_script_spec.display_rows()[2])
    assert deletable
    assert reason == ""

    source_live_spec = full_data(AverageOperation(dims=("x",)))
    live_row = source_live_spec.display_rows(scope="source")[1]
    source_bound = _fake_edit_node(
        full_data(),
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
    row = _ProvenanceDisplayRow(
        DerivationEntry("Average", None),
        replay_ref=_ProvenanceStepRef(
            "operation",
            operation_index=0,
            stage_index=0,
        ),
    )
    node = _fake_edit_node(full_data(AverageOperation(dims=("x",))))
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
        lambda *_args: full_data(AverageOperation(dims=("x",))),
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

    live_row = full_data(AverageOperation(dims=("x",))).display_rows()[1]
    replaced: list[
        tuple[
            object,
            typing.Literal["display", "source"],
            ToolProvenanceSpec,
        ]
    ] = []
    monkeypatch.setattr(controller, "can_delete_row", lambda _row: (True, ""))
    monkeypatch.setattr(
        controller,
        "_display_spec_for_row",
        lambda *_args: full_data(AverageOperation(dims=("x",))),
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
        AverageOperation(dims=("z",)),
        IselOperation(kwargs={"y": 1}),
    )
    spec = _manager_replay_file_spec(file_path, *operations)
    displayed = replay_file_provenance(spec)

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
            _ProvenanceStepRef(
                "operation",
                operation_index=0,
                stage_index=0,
            ),
            (),
        )
        expected = replay_file_provenance(expected_spec)
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
        ScriptCodeOperation(
            label="Create result",
            code="result = derived + 1.0",
        ),
        ScriptCodeOperation(
            label="Use result",
            code="result = result * 2.0",
        ),
    )
    spec = script(
        *operations,
        start_label="Start from data",
        seed_code="derived = data",
        active_name="result",
    )
    displayed = replay_script_provenance(spec, {"data": data})

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
        AssignAttrsOperation(attrs={"copied": "yes"}),
        AverageOperation(dims=("z",)),
    )
    source_spec = full_data(*source_operations)
    source_data = source_spec.apply(source_base)

    dest_base = _provenance_paste_test_data("dest") + 100.0
    dest_seed_op = ScriptCodeOperation(
        label="Keep existing destination provenance",
        code="derived = derived.assign_attrs({'existing': 'yes'})",
    )
    dest_spec = script(
        dest_seed_op,
        start_label="Start from destination data",
        seed_code="derived = data",
        active_name="derived",
    )
    dest_data = replay_script_provenance(dest_spec, {"data": dest_base})

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


def test_manager_paste_structured_provenance_steps_into_selected_imagetools(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    operations = (
        AssignAttrsOperation(attrs={"copied": "yes"}),
        AverageOperation(dims=("z",)),
    )
    source_spec = full_data(*operations)
    dest_data_a = _provenance_paste_test_data("dest_a") + 100.0
    dest_data_b = _provenance_paste_test_data("dest_b") + 200.0
    expected_a = source_spec.apply(dest_data_a)
    expected_b = source_spec.apply(dest_data_b)

    with manager_context() as manager:
        tool_a = itool(dest_data_a.copy(deep=True), manager=False, execute=False)
        tool_b = itool(dest_data_b.copy(deep=True), manager=False, execute=False)
        assert isinstance(tool_a, erlab.interactive.imagetool.ImageTool)
        assert isinstance(tool_b, erlab.interactive.imagetool.ImageTool)
        index_a = manager.add_imagetool(
            tool_a,
            show=False,
            provenance_spec=full_data(),
        )
        index_b = manager.add_imagetool(
            tool_b,
            show=False,
            provenance_spec=full_data(),
        )
        _set_provenance_steps_clipboard(operations)

        manager.tree_view.clearSelection()
        select_tools(manager, [index_a, index_b])
        manager._paste_provenance_steps_from_clipboard()

        xr.testing.assert_identical(tool_a.slicer_area._data, expected_a)
        xr.testing.assert_identical(tool_b.slicer_area._data, expected_b)
        assert manager._tool_graph.root_wrappers[index_a].displayed_provenance_spec
        assert manager._tool_graph.root_wrappers[index_b].displayed_provenance_spec


def test_manager_paste_structured_provenance_steps_into_pending_memory_imagetool(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = _provenance_paste_test_data("pending_dest").assign_coords(z=np.arange(5))
    operations = (AssignAttrsOperation(attrs={"copied": "yes"}),)
    expected = full_data(*operations).apply(data)

    with manager_context() as manager:
        tool = itool(data.copy(deep=True), manager=False, execute=False)
        assert isinstance(tool, erlab.interactive.imagetool.ImageTool)
        index = manager.add_imagetool(
            tool,
            show=False,
            provenance_spec=full_data(),
        )
        tool.hide()

        workspace_path = tmp_path / "pending-provenance-paste.itws"
        manager._save_workspace_document(workspace_path, force_full=True)
        assert manager._load_workspace_file(
            workspace_path, replace=True, associate=True, mark_dirty=False, select=False
        )
        node = manager._tool_graph.root_wrappers[index]
        assert node.pending_workspace_memory_payload is not None
        assert node.imagetool is None

        materialize_calls = 0
        materialize = manager._materialize_pending_workspace_payload

        def _record_materialize(
            target: manager_wrapper._ImageToolWrapper
            | manager_wrapper._ManagedWindowNode,
        ) -> bool:
            nonlocal materialize_calls
            materialize_calls += 1
            return materialize(target)

        monkeypatch.setattr(
            manager,
            "_materialize_pending_workspace_payload",
            _record_materialize,
        )

        _set_provenance_steps_clipboard(operations)
        manager.tree_view.clearSelection()
        select_tools(manager, [index])
        manager._update_info()
        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        assert manager._metadata_paste_steps_action in menu.actions()
        assert manager._metadata_paste_steps_action.isEnabled()
        assert materialize_calls == 0
        assert node.pending_workspace_memory_payload is not None

        manager._paste_provenance_steps_from_clipboard()

        assert materialize_calls == 1
        assert node.pending_workspace_memory_payload is None
        assert node.imagetool is not None
        xr.testing.assert_identical(node.slicer_area._data, expected)
        assert node.displayed_provenance_spec is not None


def test_manager_paste_structured_provenance_steps_reports_partial_failures(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    operations = (
        AssignAttrsOperation(attrs={"copied": "yes"}),
        AverageOperation(dims=("z",)),
    )
    source_spec = full_data(*operations)
    good_data = _provenance_paste_test_data("good") + 100.0
    bad_data = xr.DataArray(
        np.arange(12, dtype=float).reshape(3, 4),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0, 3.0]},
        name="bad",
    )
    expected_good = source_spec.apply(good_data)

    with manager_context() as manager:
        good_tool = itool(good_data.copy(deep=True), manager=False, execute=False)
        bad_tool = itool(bad_data.copy(deep=True), manager=False, execute=False)
        assert isinstance(good_tool, erlab.interactive.imagetool.ImageTool)
        assert isinstance(bad_tool, erlab.interactive.imagetool.ImageTool)
        good_index = manager.add_imagetool(
            good_tool,
            show=False,
            provenance_spec=full_data(),
        )
        bad_index = manager.add_imagetool(
            bad_tool,
            show=False,
            provenance_spec=full_data(),
        )
        reports: list[
            tuple[
                int,
                tuple[tuple[str, Exception], ...],
            ]
        ] = []

        def _record_partial(
            pasted_count: int,
            failures: typing.Sequence[
                tuple[manager_wrapper._ImageToolWrapper, Exception]
            ],
        ) -> None:
            reports.append(
                (
                    pasted_count,
                    tuple((node.uid, exc) for node, exc in failures),
                )
            )

        monkeypatch.setattr(
            manager._provenance_edit_controller,
            "_show_partial_paste_failures",
            _record_partial,
        )
        _set_provenance_steps_clipboard(operations)

        manager.tree_view.clearSelection()
        select_tools(manager, [good_index, bad_index])
        manager._paste_provenance_steps_from_clipboard()

        xr.testing.assert_identical(good_tool.slicer_area._data, expected_good)
        xr.testing.assert_identical(bad_tool.slicer_area._data, bad_data)
        assert len(reports) == 1
        assert reports[0][0] == 1
        assert len(reports[0][1]) == 1
        assert reports[0][1][0][0] == manager._tool_graph.root_wrappers[bad_index].uid


def test_manager_copy_paste_kspace_conversion_steps_remain_group_editable(
    qtbot,
    anglemap,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source_base = anglemap.qsel(eV=-0.1).copy(deep=True)
    source_operations = stamp_operation_group(
        (
            KspaceWorkFunctionOperation(work_function=4.2),
            KspaceSetNormalOperation(alpha=1.0, beta=2.0, delta=3.0),
            KspaceConvertOperation(bounds=None, resolution=None),
        ),
        kind=_kspace_conversion.KSPACE_CONVERSION_GROUP_KIND,
    )
    source_spec = full_data(*source_operations)
    source_data = source_spec.apply(source_base)
    dest_base = source_base.copy(deep=True)
    dest_base.values = dest_base.values + 100.0
    expected = source_spec.apply(dest_base)
    child_base_spec = full_data()

    with manager_context() as manager:
        source_tool = itool(source_data, manager=False, execute=False)
        assert isinstance(source_tool, erlab.interactive.imagetool.ImageTool)
        source_index = manager.add_imagetool(
            source_tool,
            show=False,
            provenance_spec=source_spec,
        )
        parent_tool = itool(dest_base, manager=False, execute=False)
        assert isinstance(parent_tool, erlab.interactive.imagetool.ImageTool)
        parent_index = manager.add_imagetool(parent_tool, show=False)
        child_tool = itool(dest_base, manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            parent_index,
            show=False,
            source_spec=child_base_spec,
        )
        partial_child_tool = itool(dest_base, manager=False, execute=False)
        assert isinstance(partial_child_tool, erlab.interactive.imagetool.ImageTool)
        partial_child_uid = manager.add_imagetool_child(
            partial_child_tool,
            parent_index,
            show=False,
            source_spec=child_base_spec,
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
        partial_payload = manager_details_panel._provenance_step_clipboard_payload(
            clipboard.mimeData()
        )
        assert partial_payload is not None
        partial_operations, _active_name, contains_script = partial_payload
        assert [operation.op for operation in partial_operations] == [
            "kspace_work_function",
            "kspace_set_normal",
        ]
        assert all(operation.group is None for operation in partial_operations)
        assert not contains_script

        manager.tree_view.clearSelection()
        select_child_tool(manager, partial_child_uid)
        manager._update_info(uid=partial_child_uid)
        manager._paste_provenance_steps_from_clipboard()
        partial_node = manager._child_node(partial_child_uid)
        assert partial_node.displayed_source_spec is not None
        assert partial_node.displayed_source_spec.operations == partial_operations
        manager._update_info(uid=partial_child_uid)
        for row_index in (1, 2):
            row_item = manager.metadata_derivation_list.item(row_index)
            assert row_item is not None
            row = row_item.data(manager_details_panel._METADATA_DERIVATION_ROW_ROLE)
            assert isinstance(row, _ProvenanceDisplayRow)
            editable, reason = manager._provenance_edit_controller.can_edit_row(row)
            assert not editable
            assert "complete editable momentum-conversion group" in reason
            assert "`Set normal emission`" in reason
            assert "`Convert to momentum`" in reason

        manager.tree_view.clearSelection()
        select_tools(manager, [source_index])
        manager._update_info()
        select_metadata_rows(manager, [1, 2, 3])
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

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)
        manager._paste_provenance_steps_from_clipboard()

        xr.testing.assert_allclose(child_tool.slicer_area._data, expected)
        child_node = manager._child_node(child_uid)
        assert child_node.displayed_source_spec is not None
        pasted_operations = child_node.displayed_source_spec.operations
        assert strip_operation_groups(pasted_operations) == strip_operation_groups(
            source_operations
        )
        assert operation_group_range(
            pasted_operations,
            0,
            kind=_kspace_conversion.KSPACE_CONVERSION_GROUP_KIND,
        ) == (0, len(pasted_operations))
        source_group_ids = {
            operation.group.id for operation in source_operations if operation.group
        }
        pasted_group_ids = {
            operation.group.id for operation in pasted_operations if operation.group
        }
        assert len(pasted_group_ids) == 1
        assert pasted_group_ids.isdisjoint(source_group_ids)

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)
        for row_index in (1, 2, 3):
            row_item = manager.metadata_derivation_list.item(row_index)
            assert row_item is not None
            row = row_item.data(manager_details_panel._METADATA_DERIVATION_ROW_ROLE)
            assert isinstance(row, _ProvenanceDisplayRow)
            editable, reason = manager._provenance_edit_controller.can_edit_row(row)
            assert editable, reason


def test_manager_paste_structured_steps_preserves_source_binding(
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    parent_data = _provenance_paste_test_data("parent")
    copied_operation = AverageOperation(dims=("z",))
    source_spec = full_data(copied_operation)
    source_data = source_spec.apply(parent_data)
    child_base_spec = full_data(AssignAttrsOperation(attrs={"child": "bound"}))
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
        filter_operation = GaussianFilterOperation(sigma={"z": 1.0})
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
    child_source_spec = full_data(
        AssignAttrsOperation(attrs={"before": "script paste"})
    )
    child_data = child_source_spec.apply(data)
    script_operations = (
        ScriptCodeOperation(
            label="Offset data",
            code="result = derived + 2.0",
        ),
        AverageOperation(dims=("z",)),
    )
    script_spec = script(
        *script_operations,
        start_label="Run copied script",
        seed_code="derived = data",
        active_name="result",
    )
    script_data = replay_script_provenance(script_spec, {"data": data})

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
        expected = replay_script_provenance(
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

        bad_operation = ScriptCodeOperation(
            label="Use unavailable scratch value",
            code="result = scratch + 1.0",
        )
        bad_spec = script(
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
    dest_base_spec = script(
        ScriptCodeOperation(
            label="Compute destination result",
            code="result = derived + 1.0",
        ),
        start_label="Start from destination data",
        seed_code="derived = data",
        active_name="result",
    )
    dest_data = replay_script_provenance(dest_base_spec, {"data": data})
    copied_operation = ScriptCodeOperation(
        label="Offset copied result",
        code="result = derived + 2.0",
    )
    source_spec = script(
        copied_operation,
        start_label="Run copied script",
        seed_code="derived = data",
        active_name="result",
    )
    source_data = replay_script_provenance(source_spec, {"data": data})

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
        assert not uses_default_replay_input(copied[-1])

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

        operation = AffineCoordOperation(
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

        operation = AssignAttrsOperation(attrs={"source": "new", "flag": True})
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


def test_manager_high_dimensional_watched_data_errors_without_reduction_dialog(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    shape = (2, 3, 4, 5, 6)
    data = xr.DataArray(
        np.arange(np.prod(shape), dtype=float).reshape(shape),
        dims=("scan", "pol", "z", "y", "x"),
        coords={
            dim: np.arange(size, dtype=float)
            for dim, size in zip(("scan", "pol", "z", "y", "x"), shape, strict=True)
        },
        name="cube",
    )
    valid = xr.DataArray(
        np.arange(25, dtype=float).reshape(5, 5),
        dims=("y", "x"),
        name="valid",
    )
    create_errors: list[None] = []
    update_errors: list[tuple[object, ...]] = []

    class _ReductionDialog:
        def __init__(self, *_args: object) -> None:
            raise AssertionError("watched variables must not open reduction dialogs")

    def _critical(*args: object, **_kwargs: object) -> None:
        update_errors.append(args)

    monkeypatch.setattr(
        imagetool_highdim,
        "_HighDimensionalReductionDialog",
        _ReductionDialog,
    )
    monkeypatch.setattr(erlab.interactive.utils.MessageDialog, "critical", _critical)

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        monkeypatch.setattr(
            manager,
            "_error_creating_imagetool",
            lambda: create_errors.append(None),
        )

        manager._data_watched_update("cube", "kernel-0", data)
        assert manager.ntools == 0
        assert len(create_errors) == 1

        manager._data_recv([valid], {}, watched_var=("valid", "kernel-1"))
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        tool = manager.get_imagetool(0)
        previous = tool.slicer_area.data.copy(deep=True)
        manager._data_watched_update("valid", "kernel-1", data)
        xr.testing.assert_identical(tool.slicer_area.data, previous)
        assert update_errors


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
        node.set_detached_provenance(full_data(AverageOperation(dims=("alpha",))))

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
        assert not uses_default_replay_input(copied[-1])
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
        node.set_detached_provenance(full_data(AverageOperation(dims=("alpha",))))

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
            load_func=(
                xr.load_dataarray,
                {"engine": "h5netcdf"},
                FileDataSelection(kind="dataarray"),
            ),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        node = manager._tool_graph.root_wrappers[0]
        node.set_detached_provenance(full_data(AverageOperation(dims=("alpha",))))

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
        assert not uses_default_replay_input(copied[-1])
        namespace = _exec_generated_code(copied[-1], {})
        xr.testing.assert_identical(namespace["derived"], test_data.qsel.mean("alpha"))


def test_manager_data_recv_validates_load_selection_metadata(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    selection = FileDataSelection(kind="dataarray")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        with pytest.raises(ValueError, match="loader and kwargs"):
            manager._data_recv([test_data], {"load_func": (xr.load_dataarray,)})
        with pytest.raises(TypeError, match="selection must be a FileDataSelection"):
            manager._data_recv(
                [test_data],
                {"load_func": (xr.load_dataarray, {}, 0)},
            )
        with pytest.raises(ValueError, match="only describe one prepared array"):
            manager._data_recv(
                [test_data, test_data],
                {"load_func": (xr.load_dataarray, {}, selection)},
            )
        with pytest.raises(ValueError, match="requires explicit load_selections"):
            manager._data_recv(
                [test_data],
                {"load_func": (xr.load_dataarray, {})},
            )
        with pytest.raises(TypeError, match="must be a sequence"):
            manager._data_recv(
                [test_data],
                {"load_func": (xr.load_dataarray, {}), "load_selections": 0},
            )
        with pytest.raises(ValueError, match="one selection per prepared array"):
            manager._data_recv(
                [test_data],
                {"load_func": (xr.load_dataarray, {}), "load_selections": ()},
            )
        with pytest.raises(TypeError, match="FileDataSelection instances"):
            manager._data_recv(
                [test_data],
                {"load_func": (xr.load_dataarray, {}), "load_selections": (0,)},
            )


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
            load_func=("example", {}, FileDataSelection(kind="dataarray")),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        node = manager._tool_graph.root_wrappers[0]
        node.set_detached_provenance(full_data(AverageOperation(dims=("alpha",))))

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


def test_manager_dependency_summary_coalesces_matching_input_name_and_label() -> None:
    """Avoid repeating an input's name when its label is identical."""
    ref = types.SimpleNamespace(
        name="data",
        label="data",
        node_uid="closed-parent",
        node_snapshot_token=None,
    )
    manager = types.SimpleNamespace(
        _dependency_refs_for_uid=lambda _uid: (ref,),
        _tool_graph=types.SimpleNamespace(nodes={}),
        _dependency_ref_has_recorded_file=lambda _spec, _ref: False,
    )
    controller = manager_lineage._LineageController(typing.cast("typing.Any", manager))

    assert controller.dependency_input_summary_for_uid("derived") == (
        "data (parent no longer open)"
    )
