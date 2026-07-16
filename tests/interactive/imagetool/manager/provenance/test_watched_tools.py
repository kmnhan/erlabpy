import json
import pathlib
import types
import typing
from collections.abc import Callable

import numpy as np
import pytest
import xarray as xr

import erlab
import erlab.interactive.imagetool._highdim as imagetool_highdim
import erlab.interactive.imagetool.manager._lineage as manager_lineage
import erlab.interactive.imagetool.manager._widgets as manager_widgets
from erlab.interactive._fit2d import Fit2DTool
from erlab.interactive._mesh import MeshTool
from erlab.interactive.derivative import DerivativeTool
from erlab.interactive.fermiedge import GoldTool, ResolutionTool
from erlab.interactive.imagetool import itool
from erlab.interactive.imagetool._provenance._code import uses_default_replay_input
from erlab.interactive.imagetool._provenance._model import FileDataSelection, full_data
from erlab.interactive.imagetool._provenance._operations import AverageOperation
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
        tree.attrs.update(
            manager._workspace_controller.saving._workspace_root_attrs_payload(
                delta_save_count=0
            )
        )
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

        manager._data_ingress.receive_data(
            [test_data], {}, watched_var=("my_data", "kernel-0")
        )
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

        manager._data_ingress.receive_data([test_data], {})
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

        manager._data_ingress.receive_data([test_data], {})
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
    ref = types.SimpleNamespace(
        name="data",
        label="data",
        node_uid="closed-parent",
        node_snapshot_token=None,
        data_role="displayed",
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
