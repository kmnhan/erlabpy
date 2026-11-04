import json
import pathlib
import threading
import types
import typing
from collections.abc import Callable

import numpy as np
import pytest
import xarray as xr

import erlab
import erlab.interactive.imagetool._highdim as imagetool_highdim
import erlab.interactive.imagetool.manager._details_panel as manager_details_panel
import erlab.interactive.imagetool.manager._lineage as manager_lineage
import erlab.interactive.imagetool.manager._widgets as manager_widgets
from erlab.interactive import _persistence_constants
from erlab.interactive._fit1d import Fit1DTool
from erlab.interactive._fit2d import Fit2DTool
from erlab.interactive._mesh import MeshTool
from erlab.interactive.derivative import DerivativeTool
from erlab.interactive.fermiedge import GoldTool, ResolutionTool
from erlab.interactive.imagetool import itool
from erlab.interactive.imagetool._provenance._model import (
    FileDataSelection,
    ScriptInput,
    full_data,
)
from erlab.interactive.imagetool._provenance._operations import (
    AverageOperation,
    RenameOperation,
)
from erlab.interactive.imagetool.manager import fetch
from erlab.interactive.imagetool.manager._server import _remove_idx, _show_idx
from tests.interactive.imagetool.manager.helpers import (
    _assert_modelfit_code_replays_source,
    _exec_generated_code,
    configure_goldtool_child,
    copy_full_code_for_uid,
    make_fit1d_child,
    make_fit2d_child,
    manager_preview_pixmap,
    metadata_detail_map,
    select_child_tool,
    select_metadata_rows,
    select_tools,
    trigger_menu_action,
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

        manager._data_ingress.receive_data(
            [test_data], {}, watched_var=("data", "kernel-0")
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        tool = manager.get_imagetool(0)
        wrapper = manager._tool_graph.root_wrappers[0]
        assert wrapper.replay_source_data is None
        xr.testing.assert_identical(
            wrapper.resolved_replay_source_data(), test_data.rename("data")
        )
        wrapper.set_detached_provenance(
            full_data(AverageOperation(dims=("alpha",))),
            replay_source_data=test_data,
        )
        updated = test_data.copy(deep=True)
        updated.data = np.asarray(updated.data) * 11

        with qtbot.wait_signal(tool.slicer_area.sigSourceDataReplaced):
            manager._data_watched_update("data", "kernel-0", updated)

        xr.testing.assert_identical(tool.slicer_area.data, updated.rename("data"))
        assert wrapper.provenance_spec is not None
        assert wrapper.provenance_spec.operations == (RenameOperation(name="data"),)
        assert wrapper.replay_source_data is None
        xr.testing.assert_identical(
            wrapper.resolved_replay_source_data(), updated.rename("data")
        )


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
            manager._data_ingress,
            "_error_creating_imagetool",
            lambda: create_errors.append(None),
        )

        manager._data_watched_update("cube", "kernel-0", data)
        assert manager.ntools == 0
        assert len(create_errors) == 1

        manager._data_ingress.receive_data(
            [valid], {}, watched_var=("valid", "kernel-1")
        )
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

        manager._data_ingress.receive_data(
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
        tree = manager._workspace_controller.saving._to_datatree()
        manifest = manager._workspace_controller.saving._workspace_manifest()
        tree.attrs["imagetool_workspace_schema_version"] = 4
        tree.attrs[_persistence_constants.WORKSPACE_MANIFEST_ATTR] = json.dumps(
            manifest
        )
        assert manifest["workspace_link_id"] == workspace_link_id
        attrs = tree["0/imagetool"].attrs
        assert attrs["manager_node_watched_varname"] == "data"
        assert attrs["manager_node_watched_uid"] == "watch:stable-data"
        assert attrs["manager_node_watched_workspace_link_id"] == workspace_link_id
        assert attrs["manager_node_watched_source_label"] == "notebook-a"
        assert attrs["manager_node_watched_source_uid"] == "kernel-a"
        assert "manager_node_provenance_spec" not in attrs
        assert "<manager-replay-source-data>" not in tree["0/imagetool"]

        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)
        manager._workspace_state.link_id = "different-workspace-link"

        manager._workspace_controller.loading._load_workspace_node(
            typing.cast("xr.DataTree", tree["0"])
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        wrapper = manager._tool_graph.root_wrappers[0]
        assert wrapper.watched
        assert wrapper._watched_varname == "data"
        assert wrapper._watched_uid == "watch:stable-data"
        assert wrapper._watched_workspace_link_id == workspace_link_id
        assert wrapper._watched_source_label == "notebook-a"
        assert wrapper._watched_source_uid == "kernel-a"
        assert wrapper._watched_connected is False
        assert wrapper.replay_source_data is None
        xr.testing.assert_identical(
            wrapper.resolved_replay_source_data(), test_data.rename("data")
        )

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

        attrs = manager._workspace_controller.saving._to_datatree()["0/imagetool"].attrs
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
            manager.color_for_watched_var_source(left).getRgb()[:3]
            == manager_widgets._WATCHED_VAR_COLORS[0]
        )
        assert (
            manager.color_for_watched_var_source(right).getRgb()[:3]
            == manager_widgets._WATCHED_VAR_COLORS[0]
        )
        assert (
            manager.color_for_watched_var_source(other).getRgb()[:3]
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
    source = test_data.rename("source_array")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        manager._data_ingress.receive_data(
            [source], {}, watched_var=("my_data", "kernel-0")
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        node = manager._tool_graph.root_wrappers[0]
        assert source.name == "source_array"
        assert node.name == "my_data"
        provenance = node.provenance_spec
        assert provenance is not None
        code = provenance.display_code()
        assert code is not None
        namespace = _exec_generated_code(code, {"my_data": source.copy(deep=True)})
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
            {"my_data": source.copy(deep=True)},
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

        manager._data_ingress.receive_data([test_data], {})
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        node = manager._tool_graph.root_wrappers[0]
        node.set_detached_provenance(
            full_data(AverageOperation(dims=("alpha",))),
            replay_source_data=None,
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

        manager._data_ingress.receive_data([test_data], {})
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        node = manager._tool_graph.root_wrappers[0]
        node.set_detached_provenance(
            full_data(AverageOperation(dims=("alpha",))),
            replay_source_data=None,
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
            load_func=(
                xr.load_dataarray,
                {"engine": "h5netcdf"},
                FileDataSelection(kind="dataarray"),
            ),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        node = manager._tool_graph.root_wrappers[0]
        node.set_detached_provenance(
            full_data(AverageOperation(dims=("alpha",))),
            replay_source_data=None,
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
        namespace = _exec_generated_code(copied[-1], {})
        xr.testing.assert_identical(namespace["derived"], test_data.qsel.mean("alpha"))


def test_manager_data_ingress_validates_load_selection_metadata(
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
            manager._data_ingress.receive_data(
                [test_data], {"load_func": (xr.load_dataarray,)}
            )
        with pytest.raises(TypeError, match="selection must be a FileDataSelection"):
            manager._data_ingress.receive_data(
                [test_data],
                {"load_func": (xr.load_dataarray, {}, 0)},
            )
        with pytest.raises(ValueError, match="only describe one prepared array"):
            manager._data_ingress.receive_data(
                [test_data, test_data],
                {"load_func": (xr.load_dataarray, {}, selection)},
            )
        with pytest.raises(ValueError, match="requires explicit load_selections"):
            manager._data_ingress.receive_data(
                [test_data],
                {"load_func": (xr.load_dataarray, {})},
            )
        with pytest.raises(TypeError, match="must be a sequence"):
            manager._data_ingress.receive_data(
                [test_data],
                {"load_func": (xr.load_dataarray, {}), "load_selections": 0},
            )
        with pytest.raises(ValueError, match="one selection per prepared array"):
            manager._data_ingress.receive_data(
                [test_data],
                {"load_func": (xr.load_dataarray, {}), "load_selections": ()},
            )
        with pytest.raises(TypeError, match="FileDataSelection instances"):
            manager._data_ingress.receive_data(
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
        node.set_detached_provenance(
            full_data(AverageOperation(dims=("alpha",))),
            replay_source_data=None,
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

        manager._data_ingress.receive_data(
            [test_data], {}, watched_var=("my_data", "kernel-0")
        )
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

        manager._data_ingress.receive_data(
            [test_data], {}, watched_var=("my_data", "kernel-0")
        )
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

        manager._data_ingress.receive_data(
            [test_data], {}, watched_var=("my_data", "kernel-0")
        )
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


def test_manager_open_weighted_ftool_binds_replays_and_refreshes_inputs(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    x = np.linspace(-1.0, 1.0, 21)
    data = xr.DataArray(
        np.exp(-((x / 0.25) ** 2)),
        dims=("x",),
        coords={"x": x},
        name="data",
    )
    uncertainty = xr.DataArray(
        np.linspace(0.1, 0.2, x.size),
        dims=("x",),
        coords={"x": x},
        name="uncertainty",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        manager._data_ingress.receive_data(
            [data], {}, watched_var=("data", "kernel-data")
        )
        manager._data_ingress.receive_data(
            [uncertainty], {}, watched_var=("uncertainty", "kernel-uncertainty")
        )
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        data_node = manager._tool_graph.root_wrappers[0]
        uncertainty_node = manager._tool_graph.root_wrappers[1]
        child_uid = manager.open_weighted_ftool(data_node.uid, uncertainty_node.uid)

        assert child_uid is not None
        assert data_node._childtool_indices == [child_uid]
        child = manager.get_childtool(child_uid)
        assert isinstance(child, Fit1DTool)
        assert [(item.name, item.node_uid) for item in child.script_inputs] == [
            ("data", data_node.uid),
            ("uncertainty", uncertainty_node.uid),
        ]
        copied = copy_full_code_for_uid(monkeypatch, manager, child_uid)
        replayed = _exec_generated_code(
            copied,
            {
                "data": data.copy(deep=True),
                "uncertainty": uncertainty.copy(deep=True),
            },
        )["result"]
        assert isinstance(replayed, xr.Dataset)
        xr.testing.assert_identical(
            replayed.modelfit_data, data.rename("modelfit_data")
        )
        xr.testing.assert_allclose(replayed.modelfit_weights, 1.0 / uncertainty)

        manager._flush_idle_work(force=True)
        qtbot.wait(0)
        assert child.source_state == "fresh"

        updates: list[dict[str, xr.DataArray]] = []
        update_inputs = child.update_inputs

        def _update_inputs(inputs: dict[str, xr.DataArray]) -> bool | None:
            updates.append(dict(inputs))
            return update_inputs(inputs)

        monkeypatch.setattr(child, "update_inputs", _update_inputs)
        child._set_source_auto_update(True)
        updated_uncertainty = uncertainty * 2.0
        manager._data_watched_update(
            "uncertainty", "kernel-uncertainty", updated_uncertainty
        )
        manager._flush_idle_work(force=True)
        qtbot.wait(0)

        assert len(updates) == 1
        assert set(updates[0]) == {"data", "uncertainty"}
        xr.testing.assert_equal(updates[0]["data"], data)
        xr.testing.assert_equal(updates[0]["uncertainty"], updated_uncertainty)
        assert child.uncertainty is not None
        xr.testing.assert_equal(child.uncertainty, updated_uncertainty)

        manager.remove_imagetool(1)
        qtbot.wait_until(lambda: child.source_state == "unavailable", timeout=5000)
        assert data_node._childtool_indices == [child_uid]
        assert child.uncertainty is not None


def test_manager_open_weighted_ftool_rejects_invalid_uncertainty_input(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    uncertainty = xr.ones_like(test_data).rename("uncertainty")
    invalid_uncertainty = uncertainty.isel(alpha=slice(1, None))

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        monkeypatch.setattr(
            manager._actions_controller, "_show_operation_error", lambda *_: None
        )
        manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(test_data, _in_manager=True),
            show=False,
        )
        manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(
                invalid_uncertainty, _in_manager=True
            ),
            show=False,
        )

        data_node = manager._tool_graph.root_wrappers[0]
        invalid_node = manager._tool_graph.root_wrappers[1]
        assert manager.open_weighted_ftool(data_node.uid, invalid_node.uid) is None
        assert data_node._childtool_indices == []


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

        manager._data_ingress.receive_data(
            [test_data], {}, watched_var=("my_data", "kernel-0")
        )
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
        assert manager._metadata_full_code_available
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
        child.set_script_inputs(
            (ScriptInput(name="data", data_role="displayed"),),
            primary_input="data",
        )
        child_uid = manager.add_childtool(
            child,
            script_inputs={"data": 0},
            show=False,
        )
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
        child.set_script_inputs(
            (ScriptInput(name="data", data_role="displayed"),),
            primary_input="data",
        )
        child_uid = manager.add_childtool(
            child,
            script_inputs={"data": 0},
            show=False,
        )
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

        child = ResolutionTool(gold.copy(deep=True), data_name="gold_input")
        child.set_script_inputs(
            (ScriptInput(name="data", data_role="displayed"),),
            primary_input="data",
        )
        child_uid = manager.add_childtool(
            child,
            script_inputs={"data": 0},
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

        child = ResolutionTool(gold.copy(deep=True), data_name="gold_input")
        child.set_script_inputs(
            (ScriptInput(name="data", data_role="displayed"),),
            primary_input="data",
        )
        child_uid = manager.add_childtool(
            child,
            script_inputs={"data": 0},
            show=False,
        )

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

        child = MeshTool(test_data.copy(deep=True), data_name="mesh_input")
        child.set_script_inputs(
            (ScriptInput(name="data", data_role="displayed"),),
            primary_input="data",
        )
        child_uid = manager.add_childtool(
            child,
            script_inputs={"data": 0},
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

        child = MeshTool(test_data.copy(deep=True), data_name="mesh_input")
        child.set_script_inputs(
            (ScriptInput(name="data", data_role="displayed"),),
            primary_input="data",
        )
        child_uid = manager.add_childtool(
            child,
            script_inputs={"data": 0},
            show=False,
        )

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

        manager._data_ingress.receive_data(
            [data], {}, watched_var=("my_1d", "kernel-0")
        )
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


def test_manager_watched_1d_profile_ftool_copy_code_omits_synthetic_selection(
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

        manager._data_ingress.receive_data(
            [data], {}, watched_var=("my_1d", "kernel-0")
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.profiles[0].open_in_ftool()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[0]._childtool_indices) == 1,
            timeout=5000,
        )

        child_uid = manager._tool_graph.root_wrappers[0]._childtool_indices[0]
        child_tool = manager.get_childtool(child_uid)
        assert isinstance(child_tool, Fit1DTool)

        copied: list[str] = []
        monkeypatch.setattr(
            erlab.interactive.utils,
            "copy_to_clipboard",
            lambda text: copied.append(text) or text,
        )
        child_tool.copy_code()

        assert copied
        assert "stack_dim" not in copied[-1]
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

        manager._data_ingress.receive_data(
            [test_data], {}, watched_var=("my_data", "kernel-0")
        )
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

        manager._data_ingress.receive_data(
            [data], {}, watched_var=("my_1d", "kernel-0")
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        replay_source = manager._tool_graph.root_wrappers[
            0
        ].resolved_replay_source_data()
        assert replay_source is not None

        duplicated = manager.duplicate_imagetool(0)
        assert isinstance(duplicated, int)
        duplicated_wrapper = manager._tool_graph.root_wrappers[duplicated]
        assert duplicated_wrapper.replay_source_data is None
        xr.testing.assert_identical(
            manager._provenance_revisions.resolve_data(
                duplicated_wrapper.provenance_spec.replay_inputs[0].revision_id
            ),
            replay_source,
        )

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

        manager._data_ingress.receive_data(
            [data], {}, watched_var=("my_1d", "kernel-0")
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        tree = manager._workspace_controller.saving._to_datatree()
        assert tree["0/imagetool"].attrs["manager_node_source_input_ndim"] == 1

        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        for node in tree.values():
            manager._workspace_controller.loading._load_workspace_node(
                typing.cast("xr.DataTree", node)
            )

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
    source = ScriptInput(
        name="data",
        label="data",
        node_uid="closed-parent",
    )
    manager = types.SimpleNamespace(
        _tool_graph=types.SimpleNamespace(
            nodes={"derived": types.SimpleNamespace(tool_script_inputs=(source,))}
        ),
    )
    controller = manager_lineage._LineageController(typing.cast("typing.Any", manager))

    assert controller.dependency_input_summary_for_uid("derived") == ("data: Missing")


def test_manager_dntool_image_menu_source_and_result(
    qtbot, manager_context, monkeypatch, tmp_path
):
    from qtpy import QtWidgets

    monkeypatch.setattr(
        "erlab.interactive.imagetool.manager._logging.get_log_file_path",
        lambda: tmp_path / "manager.log",
    )
    pytest.importorskip("onnxruntime")
    from erlab.interactive._dntool import DenoiseTool
    from erlab.interactive.imagetool._provenance._operations import DenoiseOperation

    data = xr.DataArray(
        np.arange(1.0, 31.0).reshape(5, 6),
        dims=("beta", "eV"),
        coords={"beta": np.arange(5.0), "eV": np.arange(6.0)},
        name="spectrum",
    )
    with manager_context() as manager:
        parent_tool = itool(data, manager=False, execute=False)
        manager.add_imagetool(parent_tool, show=False)
        image = parent_tool.slicer_area.images[0]
        menu = image.vb.getMenu(None)
        action = next(
            a for a in menu.actions() if a.objectName() == "itool_dntool_action"
        )
        expected_input = image.current_data.T
        action.trigger()
        parent_node = manager._tool_graph.root_wrappers[0]
        qtbot.wait_until(lambda: len(parent_node._childtool_indices) == 1)
        uid = parent_node._childtool_indices[0]
        tool = manager.get_childtool(uid)
        assert isinstance(tool, DenoiseTool)
        xr.testing.assert_identical(tool.tool_data, expected_input)
        assert tool.script_inputs[0].node_uid == parent_node.uid
        assert tool.source_state == "fresh"

        tool.energy_spin.setValue(8)
        tool.angle_spin.setValue(10)
        tool.tta_check.setChecked(False)
        tool.denoise()
        qtbot.wait_until(tool.open_btn.isEnabled)
        tool.open_btn.click()
        node = manager._child_node(uid)
        qtbot.wait_until(lambda: len(node._childtool_indices) == 1)
        output_uid = node._childtool_indices[0]
        output_node = manager._child_node(output_uid)
        assert output_node.output_id == tool.Output.RESULT
        output = manager.get_imagetool(output_uid)
        xr.testing.assert_identical(output.slicer_area.data, tool.result)
        spec = tool.output_imagetool_provenance(tool.Output.RESULT, tool.result)
        assert spec is not None
        assert any(isinstance(op, DenoiseOperation) for op in spec.operations)
        namespace = _exec_generated_code(
            spec.display_code(), {"data": data.copy(deep=True)}
        )
        xr.testing.assert_identical(namespace["result"], output.slicer_area.data)

        manager.tree_view.clearSelection()
        select_child_tool(manager, output_uid)
        manager._update_info(uid=output_uid)
        denoise_rows = []
        for index in range(manager.metadata_derivation_list.conceptual_count()):
            item = manager.metadata_derivation_list.conceptual_item(index)
            assert item is not None
            row = item.data(0, manager_details_panel._METADATA_DERIVATION_ROW_ROLE)
            if row.edit_ref is not None:
                row_spec = manager._provenance_edit_controller._display_spec_for_row(
                    output_node, row
                )
                if row_spec is not None and isinstance(
                    row_spec._operation_for_ref(row.edit_ref), DenoiseOperation
                ):
                    denoise_rows.append(index)
        assert len(denoise_rows) == 1
        select_metadata_rows(manager, denoise_rows)
        code = manager._selected_derivation_code()
        assert code is not None
        namespace = _exec_generated_code(code, {"data": tool.tool_data})
        xr.testing.assert_identical(namespace["result"], tool.result)
        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        assert manager._metadata_copy_selected_action.isEnabled()
        assert manager._metadata_edit_step_action.isEnabled()
        manager._metadata_copy_selected_action.trigger()
        clipboard = QtWidgets.QApplication.clipboard()
        assert clipboard.text() == code
        copied = manager_details_panel._provenance_step_clipboard_payload(
            clipboard.mimeData()
        )
        assert copied is not None
        assert len(copied.operations) == 1
        assert isinstance(copied.operations[0], DenoiseOperation)

        output_node._set_source_auto_update(True)
        tool.undo()
        qtbot.wait_until(
            lambda: output_node.source_state == "unavailable", timeout=5000
        )
        tool.redo()
        qtbot.wait_until(lambda: output_node.source_state == "fresh", timeout=5000)
        xr.testing.assert_identical(output.slicer_area.data, tool.result)

        updated = data * 2
        parent_tool.slicer_area.replace_source_data(updated)
        qtbot.wait_until(lambda: tool.source_state == "stale", timeout=5000)
        assert not manager._lineage_controller._refresh_source_chain_to_uid(output_uid)
        qtbot.wait_until(lambda: tool.source_state == "fresh", timeout=5000)
        xr.testing.assert_identical(
            tool.tool_data, updated.transpose(*tool.tool_data.dims)
        )
        expected = erlab.analysis.image.ml.denoise(
            tool.tool_data, along="beta", resize=(8, 10), tta=False
        )
        xr.testing.assert_identical(tool.result, expected)
        qtbot.wait_until(lambda: output.slicer_area.data.identical(expected))
        xr.testing.assert_identical(output.slicer_area.data, expected)

        tool.energy_spin.setValue(12)
        workspace = tmp_path / "denoising.itws"
        manager._workspace_controller.saving._save_workspace_document(workspace)
        assert manager._workspace_controller.loading._load_workspace_file(
            workspace, replace=True, associate=False, mark_dirty=False, select=False
        )
        restored_parent = manager._tool_graph.root_wrappers[0]
        restored_uid = restored_parent._childtool_indices[0]
        restored = manager.get_childtool(restored_uid)
        assert isinstance(restored, DenoiseTool)
        assert restored.energy_spin.value() == 12
        assert restored._result_provenance()[0].resize == (8, 10)
        xr.testing.assert_identical(restored.result, expected)


def test_manager_dntool_image_menu_requires_energy(
    qtbot, manager_context, monkeypatch, tmp_path
):
    from qtpy import QtWidgets

    from erlab.interactive._dntool import DenoiseTool

    monkeypatch.setattr(
        "erlab.interactive.imagetool.manager._logging.get_log_file_path",
        lambda: tmp_path / "manager.log",
    )
    errors = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "critical", lambda *args: errors.append(args)
    )
    data = xr.DataArray(np.ones((5, 6)), dims=("x", "y"))
    with manager_context() as manager:
        parent_tool = itool(data, manager=False, execute=False)
        manager.add_imagetool(parent_tool, show=False)
        image = parent_tool.slicer_area.images[0]
        menu = image.vb.getMenu(None)
        action = next(
            a for a in menu.actions() if a.objectName() == "itool_dntool_action"
        )
        assert not action.isEnabled()
        action.trigger()
        assert errors == []
        assert manager._tool_graph.root_wrappers[0]._childtool_indices == []

        # The direct handler still rejects invalid data if called without the menu.
        image.open_in_dntool()
        assert len(errors) == 1
        assert manager._tool_graph.root_wrappers[0]._childtool_indices == []

        updated = xr.DataArray(np.ones((5, 6)), dims=("eV", "y"))
        parent_tool.slicer_area.replace_source_data(updated)
        assert image.vb.getMenu(None) is menu
        assert action.isEnabled()
        action.trigger()
        parent_node = manager._tool_graph.root_wrappers[0]
        qtbot.wait_until(lambda: len(parent_node._childtool_indices) == 1)
        assert isinstance(
            manager.get_childtool(parent_node._childtool_indices[0]), DenoiseTool
        )


def test_manager_dntool_cropped_selection(
    qtbot, manager_context, monkeypatch, tmp_path
):
    from qtpy import QtCore, QtWidgets

    from erlab.interactive._dntool import DenoiseTool

    monkeypatch.setattr(
        "erlab.interactive.imagetool.manager._logging.get_log_file_path",
        lambda: tmp_path / "manager.log",
    )
    data = xr.DataArray(
        np.arange(1.0, 43.0).reshape(6, 7),
        dims=("beta", "eV"),
        coords={"beta": np.arange(6.0), "eV": np.arange(7.0)},
    )
    with manager_context() as manager:
        parent_tool = itool(data, manager=False, execute=False)
        manager.add_imagetool(parent_tool, show=False)
        area = parent_tool.slicer_area
        area.set_manual_limits({"beta": (1.0, 3.0), "eV": (2.0, 5.0)})
        image = area.main_image
        monkeypatch.setattr(
            QtWidgets.QApplication,
            "queryKeyboardModifiers",
            lambda: QtCore.Qt.KeyboardModifier.AltModifier,
        )
        menu = image.vb.getMenu(None)
        action = next(
            a for a in menu.actions() if a.objectName() == "itool_dntool_action"
        )
        expected = image.current_data.T
        assert expected.size < data.size
        action.trigger()
        parent_node = manager._tool_graph.root_wrappers[0]
        qtbot.wait_until(lambda: len(parent_node._childtool_indices) == 1)
        tool = manager.get_childtool(parent_node._childtool_indices[0])
        assert isinstance(tool, DenoiseTool)
        xr.testing.assert_identical(tool.tool_data, expected)
        source = tool.script_inputs[0].parsed_source_spec()
        assert source is not None
        xr.testing.assert_identical(
            source.apply(area._tool_source_parent_data()), expected
        )


@pytest.mark.parametrize("detached", [False, True])
def test_manager_dntool_output_provenance_edit_and_copy(
    qtbot, manager_context, monkeypatch, tmp_path, detached
):
    from qtpy import QtWidgets

    from erlab.interactive._dntool import DenoiseTool
    from erlab.interactive.imagetool import dialogs
    from erlab.interactive.imagetool._provenance._operations import DenoiseOperation

    monkeypatch.setattr(
        "erlab.interactive.imagetool.manager._logging.get_log_file_path",
        lambda: tmp_path / "manager.log",
    )

    def denoise(data, resize, **kwargs):
        del kwargs
        return data + resize[0]

    monkeypatch.setattr(erlab.analysis.image.ml, "denoise", denoise)
    data = xr.DataArray(
        np.arange(1.0, 31.0).reshape(5, 6),
        dims=("beta", "eV"),
        coords={"beta": np.arange(5.0), "eV": np.arange(6.0)},
        name="data",
    )
    with manager_context() as manager:
        manager._data_ingress.receive_data([data], {}, watched_var=("data", "kernel-0"))
        qtbot.wait_until(lambda: manager.ntools == 1)
        parent_tool = manager.get_imagetool(0)
        menu = parent_tool.slicer_area.images[0].vb.getMenu(None)
        next(
            action
            for action in menu.actions()
            if action.objectName() == "itool_dntool_action"
        ).trigger()
        parent_node = manager._tool_graph.root_wrappers[0]
        qtbot.wait_until(lambda: len(parent_node._childtool_indices) == 1)
        tool = manager.get_childtool(parent_node._childtool_indices[0])
        assert isinstance(tool, DenoiseTool)
        tool.energy_spin.setValue(8)
        tool.angle_spin.setValue(10)
        tool.tta_check.setChecked(False)
        tool.denoise()
        qtbot.wait_until(tool.open_btn.isEnabled)
        before_uids = set(manager._tool_graph.nodes)
        if detached:
            tool.show()
            tool.images[1].open_itool()
        else:
            tool.open_btn.click()
        qtbot.wait_until(lambda: len(manager._tool_graph.nodes) == len(before_uids) + 1)
        output_uid = (set(manager._tool_graph.nodes) - before_uids).pop()
        output_node = manager._tool_graph.nodes[output_uid]
        output = manager.get_imagetool(output_uid)
        before = output.slicer_area.data.copy(deep=True)

        manager.tree_view.clearSelection()
        select_child_tool(manager, output_uid)
        manager._update_info(uid=output_uid)
        denoise_rows = []
        for index in range(manager.metadata_derivation_list.conceptual_count()):
            item = manager.metadata_derivation_list.conceptual_item(index)
            assert item is not None
            row = item.data(0, manager_details_panel._METADATA_DERIVATION_ROW_ROLE)
            if row.edit_ref is None:
                continue
            row_spec = manager._provenance_edit_controller._display_spec_for_row(
                output_node, row
            )
            if row_spec is not None and isinstance(
                row_spec._operation_for_ref(row.edit_ref), DenoiseOperation
            ):
                denoise_rows.append(index)
        assert len(denoise_rows) == 1
        select_metadata_rows(manager, denoise_rows)
        assert manager._build_metadata_derivation_menu() is not None
        assert manager._metadata_edit_step_action.isEnabled()

        edited = []

        def edit_denoise(dialog):
            assert dialog.energy_spin.value() == 8
            assert dialog.angle_spin.value() == 10
            assert dialog.along_combo.currentData() == "beta"
            assert not dialog.tta_check.isChecked()
            dialog.energy_spin.setValue(12)
            edited.append(True)
            return int(QtWidgets.QDialog.DialogCode.Accepted)

        monkeypatch.setattr(dialogs._DenoiseDialog, "exec", edit_denoise)
        manager._metadata_edit_step_action.trigger()
        assert edited == [True]
        expected = tool.tool_data + 12
        if detached:
            expected = expected.T
        assert not output.slicer_area.data.identical(before)
        xr.testing.assert_identical(output.slicer_area.data, expected)
        spec = output_node.displayed_provenance_spec
        assert spec is not None
        operations = [op for op in spec.operations if isinstance(op, DenoiseOperation)]
        assert len(operations) == 1
        assert operations[0].resize == (12, 10)
        copied = []
        monkeypatch.setattr(
            erlab.interactive.utils,
            "copy_to_clipboard",
            lambda code: copied.append(code) or code,
        )
        monkeypatch.setattr(manager, "_prompt_replay_input_name", lambda _node: "data")
        manager._update_info(uid=output_uid)
        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        assert manager._metadata_copy_full_action.isEnabled()
        trigger_menu_action(menu, manager._metadata_copy_full_action)
        assert copied
        namespace = _exec_generated_code(copied[-1], {"data": data})
        assert spec.active_name is not None
        xr.testing.assert_identical(namespace[spec.active_name], expected)


@pytest.mark.parametrize("succeeds", [True, False], ids=["commit", "abort"])
def test_manager_dntool_source_refresh_is_responsive_and_atomic(
    qtbot, manager_context, monkeypatch, tmp_path, succeeds
):
    from qtpy import QtCore, QtWidgets

    from erlab.interactive._dntool import DenoiseTool
    from erlab.interactive.imagetool._provenance._operations import DenoiseOperation

    monkeypatch.setattr(
        "erlab.interactive.imagetool.manager._logging.get_log_file_path",
        lambda: tmp_path / "manager.log",
    )
    data = xr.DataArray(
        np.arange(20.0).reshape(4, 5),
        dims=("beta", "eV"),
        coords={"beta": np.arange(4), "eV": np.arange(5)},
        name="data",
    )
    entered = threading.Event()
    release = threading.Event()
    calls = []
    errors = []
    gui_thread = threading.get_ident()

    def apply(_operation, input_data):
        calls.append((input_data.copy(deep=True), threading.get_ident()))
        if len(calls) > 1:
            entered.set()
            if not release.wait(10):
                raise TimeoutError("Denoise test worker was not released")
            if not succeeds:
                raise ValueError("Failed source refresh")
        return input_data * 2

    monkeypatch.setattr(DenoiseOperation, "apply", apply)
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "critical", lambda *args: errors.append(args)
    )
    with manager_context() as manager:
        parent = itool(data, manager=False, execute=False)
        parent_index = manager.add_imagetool(parent, show=False)
        tool = DenoiseTool(data)
        tool.set_script_inputs(
            (ScriptInput(name="data", data_role="source"),), primary_input="data"
        )
        uid = manager.add_childtool(
            tool, script_inputs={"data": parent_index}, show=False
        )
        tool.denoise()
        qtbot.wait_until(tool.open_btn.isEnabled)
        tool.open_btn.click()
        node = manager._child_node(uid)
        qtbot.wait_until(lambda: len(node._childtool_indices) == 1)
        output_uid = node._childtool_indices[0]
        output_node = manager._child_node(output_uid)
        output_node._set_source_auto_update(True)
        output = manager.get_imagetool(output_uid)
        previous_result = tool.result.copy(deep=True)
        previous_inputs = tool.script_inputs
        previous_snapshot = node.snapshot_token

        updated = data + 100
        parent.slicer_area.replace_source_data(updated)
        qtbot.wait_until(lambda: tool.source_state == "stale")
        try:
            assert not manager._lineage_controller._refresh_tool_inputs(
                uid, allow_recorded=True
            )
            qtbot.wait_until(entered.is_set)
            assert tool._source_refresh_deferred
            assert tool._input_state().pending is not None
            assert tool.script_inputs == previous_inputs
            assert node.snapshot_token == previous_snapshot
            xr.testing.assert_identical(tool.tool_data, data)
            xr.testing.assert_identical(tool.result, previous_result)
            xr.testing.assert_identical(output.slicer_area.data, previous_result)
            processed = []
            QtCore.QTimer.singleShot(0, lambda: processed.append(threading.get_ident()))
            qtbot.wait_until(lambda: bool(processed))
            assert processed == [gui_thread]
            assert calls[-1][1] != gui_thread
        finally:
            release.set()

        qtbot.wait_until(lambda: not tool._busy)
        qtbot.wait_until(lambda: not tool._source_refresh_deferred)
        assert tool._input_state().pending is None
        assert len(calls) == 2
        if succeeds:
            qtbot.wait_until(lambda: tool.source_state == "fresh")
            expected = updated * 2
            qtbot.wait_until(lambda: output.slicer_area.data.identical(expected))
            xr.testing.assert_identical(tool.tool_data, updated)
            xr.testing.assert_identical(tool.result, expected)
            assert node.snapshot_token != previous_snapshot
            assert errors == []
        else:
            qtbot.wait_until(lambda: len(errors) == 1)
            assert tool.source_state == "stale"
            assert tool.script_inputs == previous_inputs
            assert node.snapshot_token == previous_snapshot
            xr.testing.assert_identical(tool.tool_data, data)
            xr.testing.assert_identical(tool.result, previous_result)
            xr.testing.assert_identical(output.slicer_area.data, previous_result)
        qtbot.wait_until(
            lambda: not manager._dependency_tracker.has_pending_source_refreshes()
        )


@pytest.mark.parametrize("completion", ["finish", "restore_state", "close"])
def test_manager_dntool_source_refresh_uses_latest_queued_input(
    qtbot, manager_context, monkeypatch, tmp_path, completion
):
    from erlab.interactive._dntool import DenoiseTool, _denoise_coordinator
    from erlab.interactive.imagetool._provenance._operations import DenoiseOperation

    monkeypatch.setattr(
        "erlab.interactive.imagetool.manager._logging.get_log_file_path",
        lambda: tmp_path / "manager.log",
    )
    data = xr.DataArray(
        np.arange(20.0).reshape(4, 5),
        dims=("beta", "eV"),
        coords={"beta": np.arange(4), "eV": np.arange(5)},
        name="data",
    )
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def apply(_operation, input_data):
        calls.append(input_data.copy(deep=True))
        if len(calls) == 2:
            entered.set()
            if not release.wait(10):
                raise TimeoutError("Denoise test worker was not released")
        return input_data * 2

    monkeypatch.setattr(DenoiseOperation, "apply", apply)
    with manager_context() as manager:
        parent = itool(data, manager=False, execute=False)
        parent_index = manager.add_imagetool(parent, show=False)
        tool = DenoiseTool(data)
        tool.set_script_inputs(
            (ScriptInput(name="data", data_role="source"),), primary_input="data"
        )
        uid = manager.add_childtool(
            tool, script_inputs={"data": parent_index}, show=False
        )
        pre_run_status = tool.tool_status.model_copy(deep=True)
        tool.denoise()
        qtbot.wait_until(tool.open_btn.isEnabled)
        tool.open_btn.click()
        node = manager._child_node(uid)
        qtbot.wait_until(lambda: len(node._childtool_indices) == 1)
        output_uid = node._childtool_indices[0]
        manager._child_node(output_uid)._set_source_auto_update(True)
        output = manager.get_imagetool(output_uid)
        accepted_inputs = tool.script_inputs
        accepted_provenance = tool._result_provenance()

        first = data + 100
        parent.slicer_area.replace_source_data(first)
        qtbot.wait_until(lambda: tool.source_state == "stale")
        latest = data + 200
        try:
            assert not manager._lineage_controller._refresh_tool_inputs(
                uid, allow_recorded=True
            )
            qtbot.wait_until(entered.is_set)
            tool._set_source_auto_update(True)
            parent.slicer_area.replace_source_data(latest)
            qtbot.wait_until(
                lambda: manager._dependency_tracker.source_refresh_queued(uid, uid)
            )
            assert tool._source_refresh_deferred
            xr.testing.assert_identical(tool.tool_data, data)
            xr.testing.assert_identical(output.slicer_area.data, data * 2)
            task = tool._denoise_task
            assert task is not None
            if completion == "restore_state":
                tool.tool_status = pre_run_status
            elif completion == "close":
                assert tool.close()
            if completion != "finish":
                assert not tool._source_refresh_deferred
                assert tool._input_state().pending is None
        finally:
            release.set()

        qtbot.wait_until(lambda: task not in _denoise_coordinator()._tasks)
        if completion != "finish":
            qtbot.wait_until(lambda: not manager._interaction_gate.pending_keys)
            assert len(calls) == 2
            assert not tool._busy
            assert tool._denoise_task is None
            assert tool.script_inputs == accepted_inputs
            xr.testing.assert_identical(tool.tool_data, data)
            xr.testing.assert_identical(output.slicer_area.data, data * 2)
            if completion == "restore_state":
                assert tool.tool_status == pre_run_status
                assert tool._result_provenance() == ()
                with pytest.raises(ValueError, match="Denoise the data first"):
                    _ = tool.result
            else:
                xr.testing.assert_identical(tool.result, data * 2)
                assert tool._result_provenance() == accepted_provenance
            assert not manager._dependency_tracker.has_pending_source_refreshes()
            return

        qtbot.wait_until(lambda: len(calls) == 3 and not tool._busy)
        qtbot.wait_until(lambda: tool.source_state == "fresh")
        expected = latest * 2
        qtbot.wait_until(lambda: output.slicer_area.data.identical(expected))
        xr.testing.assert_identical(calls[1], first)
        xr.testing.assert_identical(calls[2], latest)
        xr.testing.assert_identical(tool.tool_data, latest)
        xr.testing.assert_identical(tool.result, expected)
        assert not tool._source_refresh_deferred
        assert tool._input_state().pending is None
        qtbot.wait_until(
            lambda: not manager._dependency_tracker.has_pending_source_refreshes()
        )
