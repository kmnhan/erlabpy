import json
import pathlib
import typing
from collections.abc import Callable

import numpy as np
import xarray as xr
from qtpy import QtCore, QtWidgets

import erlab
import erlab.interactive.imagetool.manager._details_panel as manager_details_panel
from erlab.interactive.imagetool import itool
from erlab.interactive.imagetool._provenance._code import uses_default_replay_input
from erlab.interactive.imagetool._provenance._model import (
    FileDataSelection,
    ToolProvenanceSpec,
)
from erlab.interactive.imagetool.manager import fetch, replace_data
from tests.interactive.imagetool.manager.helpers import (
    _exec_generated_code,
    metadata_derivation_texts,
    metadata_detail_map,
    select_child_tool,
    select_metadata_rows,
    select_tools,
    set_transform_launch_mode,
    trigger_menu_action,
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

        tree = manager._workspace_controller.saving._to_datatree()
        provenance_payload = json.loads(
            tree["1/imagetool"].attrs["manager_node_provenance_spec"]
        )
        assert provenance_payload["schema_version"] == 3
        assert provenance_payload["kind"] == "file"
        assert provenance_payload["operations"] == []
        assert len(provenance_payload["steps"]) == 1
        assert provenance_payload["steps"][0]["input_policy"] == "current"
        assert provenance_payload["steps"][0]["operation"]["op"] == "qsel_aggregate"
        assert (
            provenance_payload["file_load_source"]["replay_call"]["target"]
            == "xarray.load_dataarray"
        )
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        for node in tree.values():
            manager._workspace_controller.loading._load_workspace_node(
                typing.cast("xr.DataTree", node)
            )

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

        tree = manager._workspace_controller.saving._to_datatree()
        tree["0/imagetool"].attrs["manager_node_provenance_spec"] = json.dumps(
            legacy_payload
        )
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        for node in tree.values():
            manager._workspace_controller.loading._load_workspace_node(
                typing.cast("xr.DataTree", node)
            )

        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        loaded = manager._tool_graph.root_wrappers[0]
        assert loaded.provenance_spec is not None
        assert loaded.provenance_spec.schema_version == 3
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
        assert detached.replay_source_data is not None
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
        assert duplicated_detached.replay_source_data is not None
        xr.testing.assert_identical(
            duplicated_detached.replay_source_data,
            detached.replay_source_data,
        )
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
