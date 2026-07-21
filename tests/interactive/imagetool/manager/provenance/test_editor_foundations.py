import enum
import pathlib
import types
import typing
import warnings
from collections.abc import Sequence

import numpy as np
import pytest
import xarray as xr
from qtpy import QtCore, QtWidgets

import erlab
import erlab.interactive.imagetool.dialogs as imagetool_dialogs
import erlab.interactive.imagetool.manager._details_panel as manager_details_panel
import erlab.interactive.imagetool.manager._dialogs as manager_dialogs
import erlab.interactive.imagetool.manager._lineage as manager_lineage
import erlab.interactive.imagetool.manager._mainwindow as manager_mainwindow
import erlab.interactive.imagetool.manager._widgets as manager_widgets
import erlab.interactive.imagetool.manager._wrapper as manager_wrapper
from erlab.interactive.imagetool._provenance._model import (
    DerivationEntry,
    FileDataSelection,
    FileLoadSource,
    FileReplayCall,
    ScriptInput,
    ToolProvenanceOperation,
    ToolProvenanceSpec,
    _ProvenanceDisplayRow,
    _ProvenanceStepRef,
    full_data,
    script,
)
from erlab.interactive.imagetool._provenance._operations import (
    AffineCoordOperation,
    AssignAttrsOperation,
    AssignCoord1DOperation,
    AssignCoordsOperation,
    AssignScalarCoordOperation,
    AverageOperation,
    BoxcarFilterOperation,
    CoarsenOperation,
    DivideByCoordOperation,
    GaussianFilterOperation,
    ImageDerivativeOperation,
    InterpolationOperation,
    IselOperation,
    KspaceConvertOperation,
    LeadingEdgeOperation,
    NormalizeOperation,
    QSelAggregationOperation,
    QSelOperation,
    RemoveMeshOperation,
    RenameDimsCoordsOperation,
    RotateOperation,
    ScriptCodeOperation,
    SelOperation,
    SortByOperation,
    SwapDimsOperation,
    SymmetrizeNfoldOperation,
    SymmetrizeOperation,
    ThinOperation,
    UniformInterpolationOperation,
)
from erlab.interactive.imagetool.dialogs import SelectionDialog
from erlab.interactive.imagetool.manager._provenance_edit import (
    _controller as provenance_edit_controller,
)
from erlab.interactive.imagetool.manager._provenance_edit import (
    _editors as provenance_editors,
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
    dialog = provenance_edit_files._FileLoadEditDialog(load_source, parent)
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
        provenance_edit_files,
        "_load_provenance_from_file_details",
        lambda *_args, **_kwargs: pytest.fail(
            "opening file-load editor must not rebuild load provenance"
        ),
    )
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)

    dialog = provenance_edit_files._FileLoadEditDialog(load_source, parent)
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
    dialog = provenance_edit_files._FileLoadEditDialog(load_source, parent)
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
        spec = dialog.provenance_spec(active_name="derived", replay_steps=())

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
    dialog = provenance_edit_files._FileLoadEditDialog(load_source, parent)
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
    peer = provenance_edit_files._FileLoadBatchPeer(
        node=typing.cast("typing.Any", peer_node),
        scope="display",
        spec=peer_spec,
        original_path=old_dir / "b.h5",
        loader_summary="xarray.load_dataarray",
    )
    multi_suffix_peer = provenance_edit_files._FileLoadBatchPeer(
        node=typing.cast("typing.Any", multi_suffix_peer_node),
        scope="display",
        spec=multi_suffix_peer_spec,
        original_path=old_dir / "c.scan.h5",
        loader_summary="xarray.load_dataarray",
    )
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    dialog = provenance_edit_files._FileLoadEditDialog(
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
    peer = provenance_edit_files._FileLoadBatchPeer(
        node=typing.cast("typing.Any", peer_node),
        scope="display",
        spec=peer_spec,
        original_path=old_dir / "b.h5",
        loader_summary="xarray.load_dataarray",
        script_input_path=(1,),
    )
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    dialog = provenance_edit_files._FileLoadEditDialog(
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
    required_peer = provenance_edit_files._FileLoadBatchPeer(
        node=typing.cast("typing.Any", peer_node),
        scope="display",
        spec=required_spec,
        original_path=old_dir / "b.h5",
        loader_summary="xarray.load_dataarray (engine='scipy')",
        script_input_path=(1,),
        preserve_loader=True,
    )
    optional_peer = provenance_edit_files._FileLoadBatchPeer(
        node=typing.cast("typing.Any", peer_node),
        scope="display",
        spec=optional_spec,
        original_path=old_dir / "c.h5",
        loader_summary="xarray.load_dataarray",
        script_input_path=(2,),
    )
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    dialog = provenance_edit_files._FileLoadEditDialog(
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
    assert relinked.steps == required_spec.steps
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
    peer = provenance_edit_files._FileLoadBatchPeer(
        node=typing.cast("typing.Any", peer_node),
        scope="display",
        spec=peer_spec,
        original_path=tmp_path / "b.h5",
        loader_summary="xarray.load_dataarray",
    )
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    dialog = provenance_edit_files._FileLoadEditDialog(
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
    stale_peer = provenance_edit_files._FileLoadBatchPeer(
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
    dialog = provenance_edit_files._FileLoadEditDialog(
        current_spec.file_load_source,
        parent,
        batch_peers=(stale_peer,),
    )
    qtbot.addWidget(dialog)

    with pytest.raises(RuntimeError, match="no longer replayable"):
        dialog.peer_provenance_spec(stale_peer)
    with pytest.raises(RuntimeError, match="no longer replayable"):
        provenance_edit_files._relinked_file_load_spec(
            stale_spec,
            pathlib.Path("current-b.h5"),
        )


def test_file_load_edit_dialog_batch_disabled_without_peers(qtbot) -> None:
    spec = _manager_replay_file_spec(pathlib.Path("a.h5"))
    assert spec.file_load_source is not None
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    dialog = provenance_edit_files._FileLoadEditDialog(
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
        UniformInterpolationOperation(sizes={"x": 3}),
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
        BoxcarFilterOperation(size={"x": 3}),
        ImageDerivativeOperation(
            method="diffn",
            kwargs={"coord": "x", "order": 2},
        ),
        RemoveMeshOperation(
            first_order_peaks=((1, 1), (1, 2), (1, 0)),
            order=1,
            n_pad=0,
            roi_hw=1,
        ),
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
    assert provenance_editors._dialog_class_for_operation(operation) is not None


def test_manager_provenance_operation_editor_contract_is_valid() -> None:
    provenance_editors._validate_operation_editor_contract()


def test_manager_provenance_editor_contract_rejects_group_without_matcher() -> None:
    class _MissingGroupMatcherDialog(imagetool_dialogs.DataTransformDialog):
        __module__ = imagetool_dialogs.__name__
        operation_types = (ScriptCodeOperation,)
        grouped_operation_only = True

        def restore_transform_operations(
            self,
            operations: Sequence[ToolProvenanceOperation],
        ) -> None:
            del operations

    try:
        errors = provenance_editors._operation_editor_contract_errors()
    finally:
        _MissingGroupMatcherDialog.operation_types = ()

    assert (
        "_MissingGroupMatcherDialog declares ScriptCodeOperation as a grouped editor "
        "but does not override operation_group_for_edit"
    ) in errors


def test_manager_provenance_editor_contract_rejects_mixed_editors() -> None:
    class _StandaloneScriptDialog(imagetool_dialogs.DataTransformDialog):
        __module__ = imagetool_dialogs.__name__
        operation_types = (ScriptCodeOperation,)

        def restore_transform_operation(
            self,
            operation: ToolProvenanceOperation,
        ) -> None:
            del operation

    class _GroupedScriptDialog(imagetool_dialogs.DataTransformDialog):
        __module__ = imagetool_dialogs.__name__
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
        errors = provenance_editors._operation_editor_contract_errors()
    finally:
        _StandaloneScriptDialog.operation_types = ()
        _GroupedScriptDialog.operation_types = ()

    assert (
        "ScriptCodeOperation has both standalone and grouped editors: "
        "_StandaloneScriptDialog; _GroupedScriptDialog"
    ) in errors


def test_manager_provenance_editor_contract_rejects_ambiguous_editors() -> None:
    class _StandaloneScriptDialogA(imagetool_dialogs.DataTransformDialog):
        __module__ = imagetool_dialogs.__name__
        operation_types = (ScriptCodeOperation,)

        def restore_transform_operation(
            self,
            operation: ToolProvenanceOperation,
        ) -> None:
            del operation

    class _StandaloneScriptDialogB(imagetool_dialogs.DataTransformDialog):
        __module__ = imagetool_dialogs.__name__
        operation_types = (ScriptCodeOperation,)

        def restore_transform_operation(
            self,
            operation: ToolProvenanceOperation,
        ) -> None:
            del operation

    try:
        errors = provenance_editors._operation_editor_contract_errors()
        with pytest.raises(RuntimeError, match="Multiple standalone"):
            provenance_editors._standalone_editor_dialog_class_for_operation_type(
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
    class _GroupedScriptDialogA(imagetool_dialogs.DataTransformDialog):
        __module__ = imagetool_dialogs.__name__
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

    class _GroupedScriptDialogB(imagetool_dialogs.DataTransformDialog):
        __module__ = imagetool_dialogs.__name__
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
        errors = provenance_editors._operation_editor_contract_errors()
    finally:
        _GroupedScriptDialogA.operation_types = ()
        _GroupedScriptDialogB.operation_types = ()

    assert (
        "ScriptCodeOperation has multiple grouped editors: "
        "_GroupedScriptDialogA, _GroupedScriptDialogB"
    ) in errors


def test_manager_provenance_editor_contract_rejects_missing_editor() -> None:
    class _ScriptDialogWithoutRestore(imagetool_dialogs.DataTransformDialog):
        __module__ = imagetool_dialogs.__name__
        operation_types = (ScriptCodeOperation,)

    class _ScriptFilterWithoutRestore(imagetool_dialogs.DataFilterDialog):
        __module__ = imagetool_dialogs.__name__
        operation_types = (ScriptCodeOperation,)

    try:
        errors = provenance_editors._operation_editor_contract_errors()
        with pytest.raises(RuntimeError, match="Invalid ImageTool"):
            provenance_editors._validate_operation_editor_contract()
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
    assert provenance_editors._dialog_class_for_operation(operation) is SelectionDialog


def test_manager_provenance_loader_kwargs_parser_and_file_dialog_branches(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    parse = provenance_edit_files._parse_loader_kwargs

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
    dialog = provenance_edit_files._FileLoadEditDialog(load_source, parent)
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
    class _NoRestoreTransformDialog(imagetool_dialogs.DataTransformDialog):
        operation_types = (ScriptCodeOperation,)

    class _NoRestoreFilterDialog(imagetool_dialogs.DataFilterDialog):
        operation_types = (ScriptCodeOperation,)

    assert (
        provenance_editors._dialog_class_for_operation(
            ScriptCodeOperation(label="script", code="derived = data")
        )
        is None
    )

    del _NoRestoreTransformDialog, _NoRestoreFilterDialog


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
    node.detached_replay_source_data = parent_data

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
    controller = provenance_edit_controller._ProvenanceEditController(
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
    dialog_match = provenance_editors._dialog_match_for_operation_ref(
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


def test_manager_provenance_edit_file_load_helper_edges(
    tmp_path: pathlib.Path,
) -> None:
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

    assert provenance_edit_files._file_load_edit_active_name(script_spec) == "loaded"
    replaced = provenance_edit_files._replace_file_load_fields(
        script_spec,
        replacement,
    )
    assert replaced.kind == "script"
    assert replaced.start_label == replacement.start_label
    assert replaced.seed_code == replacement.seed_code
    assert replaced.file_load_source == replacement.file_load_source
    with pytest.raises(RuntimeError, match="not a file load"):
        provenance_edit_files._replace_file_load_fields(
            script_spec,
            full_data(),
        )

    invalid_filename = FileNotFoundError()
    invalid_filename.filename = object()
    assert (
        provenance_edit_files._file_not_found_path_from_exception(invalid_filename)
        is None
    )
    nested_missing = FileNotFoundError(2, "No such file", str(source_path))
    wrapper = RuntimeError("wrapped")
    wrapper.__cause__ = nested_missing
    assert (
        provenance_edit_files._file_not_found_path_from_exception(wrapper)
        == source_path
    )


def test_manager_script_code_edit_dialog_uses_python_code_editor(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    dialog = provenance_editors._ScriptCodeEditDialog(
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

    monkeypatch.setattr(provenance_edit_controller, "_ScriptCodeEditDialog", FakeDialog)
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

    monkeypatch.setattr(provenance_edit_controller, "_ScriptCodeEditDialog", FakeDialog)
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
    node.detached_replay_source_data = parent_data
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

    monkeypatch.setattr(provenance_edit_controller, "_ScriptCodeEditDialog", FakeDialog)
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

    monkeypatch.setattr(provenance_edit_controller, "_ScriptCodeEditDialog", FakeDialog)
    monkeypatch.setattr(controller, "_validate_and_replace", cancel_validate)
    monkeypatch.setattr(controller, "_show_failed", lambda *args: failures.append(args))

    controller.edit_row(spec.display_rows()[2])

    assert failures == []
