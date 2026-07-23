import pathlib
import types
import typing

import numpy as np
import pytest
import xarray as xr
from qtpy import QtCore, QtWidgets

import erlab
import erlab.interactive.imagetool.dialogs as imagetool_dialogs
import erlab.interactive.imagetool.manager._widgets as manager_widgets
from erlab.interactive._fit1d import Fit1DTool
from erlab.interactive._fit2d import Fit2DTool
from erlab.interactive._mesh import MeshTool
from erlab.interactive.derivative import DerivativeTool
from erlab.interactive.fermiedge import GoldTool, ResolutionTool
from erlab.interactive.imagetool import _kspace_conversion
from erlab.interactive.imagetool._provenance._model import (
    DerivationEntry,
    ToolProvenanceOperation,
    ToolProvenanceSpec,
    _ProvenanceDisplayRow,
    _ProvenanceStepRef,
    compose_display_provenance,
    full_data,
    script,
    selection,
    stamp_operation_group,
)
from erlab.interactive.imagetool._provenance._operations import (
    AffineCoordOperation,
    DivideByCoordOperation,
    GaussianFilterOperation,
    ImageDerivativeOperation,
    IselOperation,
    KspaceConfigurationOperation,
    KspaceConvertOperation,
    KspaceSetNormalOperation,
    LeadingEdgeOperation,
    ModelFitOperation,
    NormalizeOperation,
    QSelAggregationOperation,
    QSelOperation,
    RemoveMeshOperation,
    ScriptCodeOperation,
    SelOperation,
    SortByOperation,
    TransposeOperation,
    UniformInterpolationOperation,
    _ModelFitParameterSpec,
)
from erlab.interactive.imagetool.dialogs import SelectionDialog
from erlab.interactive.imagetool.manager._provenance_edit import (
    _controller as provenance_edit_controller,
)
from erlab.interactive.imagetool.manager._provenance_edit import (
    _editors as provenance_editors,
)
from erlab.interactive.kspace import KspaceTool

from ._common import (
    _fake_edit_controller,
    _fake_edit_node,
    _manager_provenance_file_spec,
    _manager_replay_file_spec,
    _set_aggregate,
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
    dialog = imagetool_dialogs.AggregateDialog(
        tool.slicer_area,
        provenance_edit_mode=True,
    )

    assert not hasattr(dialog, "launch_mode_combo")
    provenance_edit_controller._ProvenanceEditController._restore_native_edit_dialog(
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
    dialog = imagetool_dialogs.AggregateDialog(
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

    class _EditModeDialog(imagetool_dialogs._DataManipulationDialog):
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
    dialog = imagetool_dialogs._DataManipulationDialog(tool.slicer_area)

    with pytest.raises(NotImplementedError):
        dialog.provenance_edit_operations()


def test_manager_provenance_filter_edit_mode_accept_skips_preview_apply(qtbot) -> None:
    data = xr.DataArray(np.arange(4.0), dims=("x",))
    tool = erlab.interactive.imagetool.ImageTool(data)
    qtbot.addWidget(tool)
    dialog = imagetool_dialogs.NormalizeDialog(
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
    provenance_edit_controller._ProvenanceEditController._restore_native_edit_dialog(
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
            imagetool_dialogs.NormalizeDialog,
            id="normalize",
        ),
        pytest.param(
            GaussianFilterOperation(sigma={"x": 0.25}),
            imagetool_dialogs.GaussianFilterDialog,
            id="gaussian",
        ),
        pytest.param(
            DivideByCoordOperation(coord_name="scale"),
            imagetool_dialogs.DivideByCoordDialog,
            id="divide_by_coord",
        ),
        pytest.param(
            SortByOperation(variables=("order",), ascending=False),
            imagetool_dialogs.SortByDialog,
            id="sortby",
        ),
    ],
)
def test_manager_terminal_current_data_edit_opens_without_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    operation: ToolProvenanceOperation,
    dialog_cls: type[imagetool_dialogs._DataManipulationDialog],
) -> None:
    base = _native_current_seed_data()
    current = operation.apply(base)
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
        dialog: imagetool_dialogs._DataManipulationDialog,
    ) -> int:
        captured["dialog_cls"] = type(dialog)
        if isinstance(dialog, imagetool_dialogs.NormalizeDialog):
            captured["dims"] = tuple(
                dim for dim, check in dialog.dim_checks.items() if check.isChecked()
            )
            captured["mode"] = dialog._mode
            captured["denominator_rtol"] = dialog.denominator_rtol
        elif isinstance(dialog, imagetool_dialogs.GaussianFilterDialog):
            captured["dims"] = tuple(
                dim for dim, check in dialog.dim_checks.items() if check.isChecked()
            )
            captured["sigma"] = {
                dim: dialog._spin_value(dialog.sigma_spins[dim])
                for dim in dialog.sigma_spins
            }
        elif isinstance(dialog, imagetool_dialogs.DivideByCoordDialog):
            captured["coord_name"] = dialog._selected_coord_name
        elif isinstance(dialog, imagetool_dialogs.SortByDialog):
            captured["sort_keys"] = dialog._sort_keys
            captured["ascending"] = dialog.ascending_combo.currentData(
                QtCore.Qt.ItemDataRole.UserRole
            )
        return int(QtWidgets.QDialog.DialogCode.Rejected)

    monkeypatch.setattr(dialog_cls, "exec", exec_dialog)

    row = spec.display_rows()[1]
    assert row.edit_ref is not None
    dialog_match = provenance_editors._dialog_match_for_operation_ref(
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
    current = operation.apply(base)
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
        imagetool_dialogs.NormalizeDialog,
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
        provenance_editors._OperationDialogMatch(
            imagetool_dialogs.NormalizeDialog,
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
            imagetool_dialogs.NormalizeDialog,
            _native_current_seed_data(),
            id="normalize-missing-dim",
        ),
        pytest.param(
            GaussianFilterOperation(sigma={"missing": 0.25}),
            imagetool_dialogs.GaussianFilterDialog,
            _native_current_seed_data(),
            id="gaussian-missing-dim",
        ),
        pytest.param(
            GaussianFilterOperation(sigma={"x": 0.25}),
            imagetool_dialogs.GaussianFilterDialog,
            _native_current_seed_data().isel(x=slice(0, 1)),
            id="gaussian-degenerate-coord",
        ),
        pytest.param(
            GaussianFilterOperation(sigma={"x": 0.25}),
            imagetool_dialogs.GaussianFilterDialog,
            _native_current_seed_data().assign_coords(x=["a", "b", "c"]),
            id="gaussian-nonnumeric-coord",
        ),
        pytest.param(
            DivideByCoordOperation(coord_name="missing"),
            imagetool_dialogs.DivideByCoordDialog,
            _native_current_seed_data(),
            id="divide-by-missing-coord",
        ),
        pytest.param(
            DivideByCoordOperation(coord_name="label"),
            imagetool_dialogs.DivideByCoordDialog,
            _native_current_seed_data().assign_coords(label=("x", ["a", "b", "c"])),
            id="divide-by-nonnumeric-coord",
        ),
        pytest.param(
            SortByOperation(variables=("missing",)),
            imagetool_dialogs.SortByDialog,
            _native_current_seed_data(),
            id="sortby-missing-key",
        ),
    ],
)
def test_manager_terminal_current_data_edit_seed_rejects_invalid_metadata(
    tmp_path: pathlib.Path,
    operation: ToolProvenanceOperation,
    dialog_cls: type[imagetool_dialogs._DataManipulationDialog],
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
        provenance_editors._OperationDialogMatch(dialog_cls, 0, 1),
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
    current = operation.apply(base)
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

    def exec_dialog(dialog: imagetool_dialogs.AssignCoordsDialog) -> int:
        captured["coord_name"] = dialog.current_coord_name
        captured["scale"] = float(dialog.coord_widget.scale_spin.value())
        captured["offset"] = float(dialog.coord_widget.offset_spin.value())
        captured["reference_coord"] = dialog.coord_widget._old_coord.copy()
        captured["dialog_coord"] = dialog.slicer_area.data["y"].values.copy()
        return int(QtWidgets.QDialog.DialogCode.Rejected)

    monkeypatch.setattr(
        imagetool_dialogs.AssignCoordsDialog,
        "exec",
        exec_dialog,
    )

    row = spec.display_rows()[1]
    assert row.edit_ref is not None
    dialog_match = provenance_editors._dialog_match_for_operation_ref(
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
    current = operation.apply(base)
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
        imagetool_dialogs.AssignCoordsDialog,
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
        provenance_editors._OperationDialogMatch(
            imagetool_dialogs.AssignCoordsDialog,
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
        imagetool_dialogs.AssignCoordsDialog,
        "exec",
        lambda _dialog: int(QtWidgets.QDialog.DialogCode.Rejected),
    )

    row = spec.display_rows()[1]
    assert row.edit_ref is not None
    dialog_match = provenance_editors._dialog_match_for_operation_ref(
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
        current_data = operation.apply(replay_data).transpose(
            "eV",
            "x",
        )
        dialog_cls = imagetool_dialogs.NormalizeDialog
    elif case == "current-source-unavailable":
        operation = NormalizeOperation(dims=("x",), mode="minmax")
        operations = (operation,)
        current_data = None
        dialog_cls = imagetool_dialogs.NormalizeDialog
    else:
        operation = LeadingEdgeOperation(
            dim="eV",
            fraction=0.25,
            direction="negative",
        )
        operations = (operation,)
        current_data = operation.apply(replay_data)
        dialog_cls = imagetool_dialogs.LeadingEdgeDialog

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
    dialog_match = provenance_editors._dialog_match_for_operation_ref(
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
            imagetool_dialogs.AggregateDialog,
        )
    with pytest.raises(ValueError, match="No provenance operations"):
        controller._edited_native_operations(
            typing.cast("typing.Any", _fake_edit_node(spec)),
            row,
            spec,
            ref,
            provenance_editors._OperationDialogMatch(
                imagetool_dialogs.AggregateDialog,
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
    dialog_match = provenance_editors._OperationDialogMatch(
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
        raise manager_widgets._TrustedScriptReplayCancelled

    monkeypatch.setattr(
        controller,
        "_replay_candidate_result",
        _raise_replay_cancelled,
    )
    with pytest.raises(manager_widgets._TrustedScriptReplayCancelled):
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
    with pytest.raises(provenance_edit_controller._ProvenanceReplayFailure) as exc:
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
    filter_dialog = imagetool_dialogs.NormalizeDialog(tool.slicer_area)
    base_dialog = imagetool_dialogs._DataManipulationDialog(tool.slicer_area)
    filter_operation = NormalizeOperation(dims=("x",), mode="area")

    provenance_edit_controller._ProvenanceEditController._restore_native_edit_dialog(
        filter_dialog,
        (filter_operation,),
        None,
    )

    assert filter_dialog.provenance_edit_operations() == [filter_operation]

    with pytest.raises(ValueError, match="one operation"):
        provenance_edit_controller._ProvenanceEditController._restore_native_edit_dialog(
            filter_dialog,
            (
                NormalizeOperation(dims=("x",), mode="area"),
                NormalizeOperation(dims=("x",), mode="min"),
            ),
            None,
        )
    with pytest.raises(TypeError, match="transform or filter"):
        provenance_edit_controller._ProvenanceEditController._restore_native_edit_dialog(
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


def test_manager_provenance_validation_rejects_highdim_reorder_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
    )
    expand = ScriptCodeOperation(
        label="Expand dimensions",
        code="derived = derived.expand_dims(a=2, b=2, c=2)",
    )
    reduce = ScriptCodeOperation(
        label="Reduce added dimensions",
        code=(
            "derived = derived.mean([dim for dim in ('a', 'b', 'c') "
            "if dim in derived.dims])"
        ),
    )
    original = script(
        expand,
        reduce,
        start_label="Start from source",
        seed_code="derived = data",
        active_name="derived",
    )
    reordered = original.model_copy(update={"steps": tuple(reversed(original.steps))})
    manager = types.SimpleNamespace(
        _ensure_script_provenance_trusted=lambda *_args, **_kwargs: None,
    )
    controller = provenance_edit_controller._ProvenanceEditController(
        typing.cast("typing.Any", manager)
    )
    node = types.SimpleNamespace(
        imagetool=None,
        resolved_replay_source_data=lambda: source,
    )
    applied_edits: list[typing.Any] = []
    monkeypatch.setattr(controller, "_apply_validated_edit", applied_edits.append)

    with pytest.raises(
        provenance_edit_controller._ProvenanceReplayFailure,
        match="validating the replayed ImageTool data",
    ):
        controller._validate_and_replace(
            typing.cast("typing.Any", node),
            "display",
            reordered,
            where="validating the reordered provenance",
        )
    assert applied_edits == []


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

    with pytest.raises(provenance_edit_controller._ProvenanceReplayFailure) as exc:
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
    replaced: list[
        tuple[xr.DataArray, ToolProvenanceSpec, bool, xr.DataArray | None, bool]
    ] = []
    source_bindings: list[
        tuple[
            ToolProvenanceSpec,
            bool,
            str,
            ToolProvenanceSpec,
        ]
    ] = []

    def replace_imagetool_data(
        data,
        spec,
        *,
        propagate_descendants,
        replay_source_data,
        preserve_filter,
    ) -> None:
        replaced.append(
            (
                data,
                spec,
                propagate_descendants,
                replay_source_data,
                preserve_filter,
            )
        )

    node._replace_imagetool_data = replace_imagetool_data
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
    assert replaced[-1][2:] == (True, None, True)

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
    )
    base_spec, split_operation = controller._split_active_filter(
        typing.cast("typing.Any", node),
        file_spec,
    )
    assert split_operation == active
    assert base_spec.steps == ()

    script_file_spec = script(
        start_label="Load source",
        seed_code=typing.cast("str", file_spec.seed_code),
        active_name="derived",
        file_load_source=file_spec.file_load_source,
        steps=file_spec.steps,
    )
    assert controller._active_filter_ref(
        typing.cast("typing.Any", node),
        script_file_spec,
    ) == _ProvenanceStepRef(
        "operation",
        operation_index=0,
    )
    base_spec, split_operation = controller._split_active_filter(
        typing.cast("typing.Any", node),
        script_file_spec,
    )
    assert split_operation == active
    assert base_spec.kind == "script"
    assert base_spec.steps == ()

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
    )
    assert file_spec._operation_for_ref(stage_ref) == sel
    assert file_spec._prefix_through_ref(_ProvenanceStepRef("file_load")).steps == ()
    assert file_spec._prefix_before_ref(stage_ref).operations == (isel,)


def _analysis_tool_provenance_cases() -> tuple[
    tuple[
        str,
        type,
        tuple[ToolProvenanceOperation, ...],
        str,
    ],
    ...,
]:
    kspace_operations = stamp_operation_group(
        (
            KspaceConfigurationOperation(configuration=2),
            KspaceSetNormalOperation(alpha=1.0, beta=2.0, delta=3.0),
            KspaceConvertOperation(bounds=None, resolution=None),
        ),
        kind=_kspace_conversion.KSPACE_CONVERSION_GROUP_KIND,
    )
    model_fit = ModelFitOperation(
        fit_dim="x",
        model="PolynomialModel",
        model_kwargs={"degree": 1},
        parameters={
            "c0": _ModelFitParameterSpec(value=0.0),
            "c1": _ModelFitParameterSpec(value=1.0),
        },
        method="leastsq",
        parameter="c1",
        output="value",
    )
    remove_mesh = RemoveMeshOperation(
        first_order_peaks=((5, 5), (5, 7), (5, 3)),
        order=1,
        n_pad=0,
        roi_hw=2,
        output="corrected",
    )
    return (
        (
            DerivativeTool.tool_name,
            DerivativeTool,
            (
                UniformInterpolationOperation(sizes={"x": 5}),
                GaussianFilterOperation(sigma={"x": 1.0}),
                ImageDerivativeOperation(
                    method="diffn",
                    kwargs={"coord": "x", "order": 2},
                ),
                TransposeOperation(),
            ),
            "result",
        ),
        (
            KspaceTool.tool_name,
            KspaceTool,
            kspace_operations,
            "converted",
        ),
        (
            GoldTool.tool_name,
            GoldTool,
            (
                ScriptCodeOperation(
                    label="Fit and correct current data",
                    code="corrected = derived",
                ),
            ),
            "corrected",
        ),
        (
            ResolutionTool.tool_name,
            ResolutionTool,
            (
                ScriptCodeOperation(
                    label="Fit the current averaged edge distribution",
                    code="result = derived",
                ),
            ),
            "result",
        ),
        (
            Fit1DTool.tool_name,
            Fit1DTool,
            (
                ScriptCodeOperation(
                    label="Fit current data with the current model",
                    code="result = derived",
                ),
            ),
            "result",
        ),
        (
            Fit2DTool.tool_name,
            Fit2DTool,
            (
                IselOperation(kwargs={"y": slice(0, 3)}),
                model_fit,
            ),
            "parameter_values",
        ),
        (
            f"{Fit2DTool.tool_name}-stderr",
            Fit2DTool,
            (
                IselOperation(kwargs={"y": slice(0, 3)}),
                model_fit.model_copy(update={"output": "stderr"}),
            ),
            "parameter_stderr",
        ),
        (
            f"{MeshTool.tool_name}-corrected",
            MeshTool,
            (remove_mesh,),
            "corrected",
        ),
        (
            f"{MeshTool.tool_name}-mesh",
            MeshTool,
            (remove_mesh.model_copy(update={"output": "mesh"}),),
            "mesh",
        ),
    )


@pytest.mark.parametrize("file_backed", [False, True], ids=["memory", "file"])
@pytest.mark.parametrize(
    ("case_name", "tool_cls", "operations", "active_name"),
    _analysis_tool_provenance_cases(),
    ids=[case[0] for case in _analysis_tool_provenance_cases()],
)
def test_analysis_tool_provenance_prefixes_are_editable_and_replayable(
    case_name: str,
    tool_cls: type,
    operations: tuple[ToolProvenanceOperation, ...],
    active_name: str,
    file_backed: bool,
) -> None:
    assert tool_cls.COPY_PROVENANCE is not None
    file_spec = _manager_provenance_file_spec(pathlib.Path("scan.h5"))
    seed_code = (
        typing.cast("str", file_spec.seed_code) if file_backed else "derived = data"
    )
    file_load_source = file_spec.file_load_source if file_backed else None
    if file_load_source is not None:
        file_load_source = file_load_source.model_copy(
            update={"load_code": seed_code.replace("derived =", "data =", 1)}
        )
    spec = script(
        *operations,
        start_label=f"Start from current {case_name} input data",
        seed_code=seed_code,
        active_name=active_name,
        file_load_source=file_load_source,
    )
    node = _fake_edit_node(spec)
    if not file_backed:
        node.replay_source_data = xr.DataArray(
            np.arange(5.0),
            dims=("x",),
        )
        node.has_replay_source = True
    controller = _fake_edit_controller(node)

    rows = list(spec.display_rows())
    for row in rows:
        rows.extend(row.children)

    for row in rows:
        if row.replay_ref is not None:
            prefix = spec._prefix_through_ref(row.replay_ref)
            assert controller._script_spec_replayable_for_node(node, prefix), (
                f"{case_name} row {row.replay_ref!r} has a non-replayable prefix"
            )
        if row.edit_ref is None:
            continue
        if row.edit_ref.kind == "file_load":
            should_be_editable = True
        else:
            operation = spec._operation_for_ref(row.edit_ref)
            should_be_editable = (
                isinstance(operation, ScriptCodeOperation)
                and operation.copyable
                and operation.code is not None
            ) or provenance_editors._dialog_match_for_operation_ref(
                spec,
                row.edit_ref,
            ) is not None
        if should_be_editable:
            assert controller.can_edit_row(row) == (True, "")
        if row.replay_ref is None or row.replay_ref.kind != "operation":
            continue
        group = provenance_edit_controller._editable_group_range_for_ref(
            spec,
            row.replay_ref,
        )
        if group is None:
            deletion_candidate = spec._replace_operation_ref(row.replay_ref, ())
        else:
            deletion_candidate = spec._replace_operation_range_ref(
                row.replay_ref,
                group[0],
                group[1],
                (),
            )
        should_be_deletable = controller._script_spec_replayable_for_node(
            node,
            deletion_candidate,
        )
        assert controller.can_delete_row(row)[0] is should_be_deletable
