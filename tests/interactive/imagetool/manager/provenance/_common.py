import json
import pathlib
import types
import typing
from collections.abc import Callable

import lmfit
import numpy as np
import xarray as xr
from qtpy import QtCore, QtWidgets

import erlab
import erlab.interactive.imagetool.dialogs as imagetool_dialogs
import erlab.interactive.imagetool.manager._details_panel as manager_details_panel
from erlab.interactive._fit2d import Fit2DTool
from erlab.interactive.imagetool import itool
from erlab.interactive.imagetool._provenance._model import (
    FileLoadSource,
    FileReplayCall,
    ToolProvenanceOperation,
    ToolProvenanceSpec,
    file_load,
    full_data,
)
from erlab.interactive.imagetool.dialogs import SelectionDialog
from erlab.interactive.imagetool.manager._provenance_edit import (
    _controller as provenance_edit_controller,
)


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
    assert isinstance(dialog, imagetool_dialogs.AggregateDialog)
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


def _fake_edit_controller(
    node: typing.Any | None = None,
    *,
    parent: typing.Any | None = None,
    nodes: dict[str, typing.Any] | None = None,
    metadata_uid: str | None = None,
    script_input_can_reload: Callable[..., bool] | None = None,
) -> provenance_edit_controller._ProvenanceEditController:
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
    return provenance_edit_controller._ProvenanceEditController(
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
        replay_source_data=None,
        has_replay_source=False,
        source_spec=source_spec,
        displayed_provenance_spec=spec,
        displayed_source_spec=source_display_spec,
        source_auto_update=True,
        materialize_pending_workspace_payload=lambda: True,
        slicer_area=types.SimpleNamespace(
            _accepted_filter_provenance_operation=active_filter
        ),
    )
