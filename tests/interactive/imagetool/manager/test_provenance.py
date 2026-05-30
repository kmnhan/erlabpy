# ruff: noqa: E501,F403,F405,RUF012
from ._shared import *


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
    operation = (
        erlab.interactive.imagetool.provenance_framework.GaussianFilterOperation(
            sigma={"alpha": 1.0}
        )
    )
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
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 1,
            timeout=5000,
        )

        child_uid = manager._imagetool_wrappers[0]._childtool_indices[0]
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
    prov = erlab.interactive.imagetool.provenance_framework
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["alpha", "eV"],
        coords={"alpha": np.arange(5, dtype=float), "eV": np.arange(5, dtype=float)},
    )
    operation = prov.GaussianFilterOperation(sigma={"alpha": 1.0})
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
            source_spec=prov.full_data(),
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
    prov = erlab.interactive.imagetool.provenance_framework
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["alpha", "eV"],
        coords={"alpha": np.arange(5, dtype=float), "eV": np.arange(5, dtype=float)},
    )
    updated = data + 100.0
    operation = prov.GaussianFilterOperation(sigma={"alpha": 1.0})
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
            source_spec=prov.full_data(),
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
    prov = erlab.interactive.imagetool.provenance_framework
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
    operation = prov.GaussianFilterOperation(sigma={"x": 1.0})
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
            source_spec=prov.full_data(),
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
    prov = erlab.interactive.imagetool.provenance_framework
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["alpha", "eV"],
        coords={"alpha": np.arange(5, dtype=float), "eV": np.arange(5, dtype=float)},
    )
    operation = prov.GaussianFilterOperation(sigma={"alpha": 1.0})
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
            source_spec=prov.full_data(),
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
    prov = erlab.interactive.imagetool.provenance_framework
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(float),
        dims=["alpha", "eV"],
        coords={"alpha": np.arange(5, dtype=float), "eV": np.arange(5, dtype=float)},
    )
    operation = prov.GaussianFilterOperation(sigma={"alpha": 1.0})

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
            source_spec=prov.full_data(),
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
    prov = erlab.interactive.imagetool.provenance_framework

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
        ) -> erlab.interactive.imagetool.provenance_framework.ToolProvenanceSpec | None:
            assert output_id == "out"
            del data
            return prov.script(
                prov.ScriptCodeOperation(label="Use output", code="result = data + 10"),
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
        operation = prov.GaussianFilterOperation(sigma={"x": 1.0})
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
    prov = erlab.interactive.imagetool.provenance_framework

    class _StaticToolState(pydantic.BaseModel):
        value: int = 0

    class _StaticTool(erlab.interactive.utils.ToolWindow[_StaticToolState]):
        StateModel = _StaticToolState
        tool_name = "static-dummy"

        def __init__(
            self,
            data: xr.DataArray,
            provenance_spec: erlab.interactive.imagetool.provenance_framework.ToolProvenanceSpec,
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
        ) -> erlab.interactive.imagetool.provenance_framework.ToolProvenanceSpec | None:
            return self._provenance_spec

    data = xr.DataArray(np.arange(4.0), dims=("x",))
    provenance_spec = prov.script(
        prov.ScriptCodeOperation(label="Double data", code="result = data * 2"),
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
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 1, timeout=5000
        )

        child_uid = manager._imagetool_wrappers[0]._childtool_indices[0]
        child = typing.cast("typing.Any", manager.get_childtool(child_uid))
        assert child.main_image.data_array is not None

        child.main_image.open_itool()

        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        child_node = manager._child_node(child_uid)
        assert child_node._childtool_indices == []
        output_node = manager._imagetool_wrappers[1]
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
        second_output_node = manager._imagetool_wrappers[2]
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
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 1, timeout=5000
        )

        child_uid = manager._imagetool_wrappers[0]._childtool_indices[0]
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
            for index, wrapper in manager._imagetool_wrappers.items()
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
            lambda: len(manager._imagetool_wrappers[0]._childtool_indices) == 1,
            timeout=5000,
        )

        child_uid = manager._imagetool_wrappers[0]._childtool_indices[0]
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
    prov = erlab.interactive.imagetool.provenance_framework
    base = xr.DataArray(
        np.arange(16, dtype=float).reshape((4, 4)),
        dims=["x", "y"],
        coords={"x": np.arange(4), "y": np.arange(4)},
        name="scan",
    )
    initial_root_spec = prov.selection(prov.IselOperation(kwargs={"x": slice(0, 2)}))
    updated_root_spec = prov.selection(prov.IselOperation(kwargs={"x": slice(1, 3)}))

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
            source_spec=prov.full_data(),
            source_auto_update=True,
        )

        grandchild_data = root_data.isel(y=slice(0, 2))
        grandchild_tool = itool(grandchild_data, manager=False, execute=False)
        assert isinstance(grandchild_tool, erlab.interactive.imagetool.ImageTool)
        grandchild_uid = manager.add_imagetool_child(
            grandchild_tool,
            child_uid,
            show=False,
            source_spec=prov.selection(prov.IselOperation(kwargs={"y": slice(0, 2)})),
            source_auto_update=True,
        )

        root_node = manager._imagetool_wrappers[0]
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
    prov = erlab.interactive.imagetool.provenance_framework
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
            provenance_spec=prov.selection(
                prov.IselOperation(kwargs={"x": slice(0, 2)})
            ),
        )

        child_tool = itool(root_data.copy(deep=False), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=prov.full_data(),
            source_auto_update=False,
        )
        child_node = manager._child_node(child_uid)

        updated = base.isel(x=slice(2, 4))
        manager._imagetool_wrappers[0].set_detached_provenance(
            prov.selection(prov.IselOperation(kwargs={"x": slice(2, 4)}))
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
        manager._imagetool_wrappers[0].set_detached_provenance(
            prov.selection(prov.IselOperation(kwargs={"x": slice(4, 6)}))
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
    prov = erlab.interactive.imagetool.provenance_framework
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
            provenance_spec=prov.selection(
                prov.IselOperation(kwargs={"x": slice(0, 2)})
            ),
        )

        child_tool = itool(root_data.copy(deep=False), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=prov.full_data(),
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
            source_spec=prov.selection(prov.IselOperation(kwargs={"y": slice(0, 2)})),
            source_auto_update=True,
        )

        root_node = manager._imagetool_wrappers[0]
        child_node = manager._child_node(child_uid)
        grandchild_node = manager._child_node(grandchild_uid)

        root_node.set_detached_provenance(
            prov.selection(prov.IselOperation(kwargs={"x": slice(1, 3)}))
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
    prov = erlab.interactive.imagetool.provenance_framework
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
            provenance_spec=prov.selection(
                prov.IselOperation(kwargs={"x": slice(0, 2)})
            ),
        )

        child_tool = itool(root_data.copy(deep=False), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=prov.full_data(),
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
            source_spec=prov.selection(prov.IselOperation(kwargs={"y": slice(0, 2)})),
            source_auto_update=False,
        )

        child_node = manager._child_node(child_uid)
        grandchild_node = manager._child_node(grandchild_uid)
        updated_root = base.isel(x=slice(2, 4))

        manager._imagetool_wrappers[0].set_detached_provenance(
            prov.selection(prov.IselOperation(kwargs={"x": slice(2, 4)}))
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
    prov = erlab.interactive.imagetool.provenance_framework

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
        parent_tool.set_source_binding(prov.full_data(), auto_update=False)

        leaf_tool = itool(root_data.isel(y=slice(0, 2)), manager=False, execute=False)
        assert isinstance(leaf_tool, erlab.interactive.imagetool.ImageTool)
        leaf_uid = manager.add_imagetool_child(
            leaf_tool,
            parent_uid,
            show=False,
            source_spec=prov.selection(prov.IselOperation(kwargs={"y": slice(0, 2)})),
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
        assert manager._pending_source_refresh_targets == {}


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
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 1, timeout=5000
        )

        child_uid = manager._imagetool_wrappers[0]._childtool_indices[0]
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

        parent = manager._imagetool_wrappers[0]
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
    prov = erlab.interactive.imagetool.provenance_framework

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.open_in_meshtool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 1, timeout=5000
        )

        child_uid = manager._imagetool_wrappers[0]._childtool_indices[0]
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
            source_spec=prov.selection(
                prov.QSelOperation(kwargs={"alpha": 1, "alpha_width": 1})
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
    prov = erlab.interactive.imagetool.provenance_framework
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        parent_tool = manager.get_imagetool(0)
        parent_tool.set_provenance_spec(
            prov.script(
                prov.ScriptCodeOperation(
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
        child.timeout_spin.setValue(30.0)
        child.nfev_spin.setValue(0)
        child.y_index_spin.setValue(child.y_min_spin.value())
        child._run_fit_2d("up")
        qtbot.wait_until(
            lambda: all(ds is not None for ds in child._result_ds_full),
            timeout=10000,
        )

        child.param_plot_combo.setCurrentIndex(0)
        param_name = child.param_plot_combo.currentText()
        assert param_name

        child.param_plot._show_parameter_values()
        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)

        child.param_plot._show_parameter_stderr()
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 2, timeout=5000)

        values_uid, stderr_uid = child_node._childtool_indices
        values_node = manager._child_node(values_uid)
        stderr_node = manager._child_node(stderr_uid)
        assert manager.ntools == 1
        assert values_node.parent_uid == child_uid
        assert stderr_node.parent_uid == child_uid
        assert values_node.output_id == "fit2d.param_plot.values"
        assert stderr_node.output_id == "fit2d.param_plot.stderr"
        assert values_node.source_spec is None
        assert values_node.provenance_spec is not None
        assert stderr_node.source_spec is None
        assert stderr_node.provenance_spec is not None
        xr.testing.assert_identical(
            fetch(values_uid), child._param_plot_dataarray(param_name, stderr=False)
        )
        xr.testing.assert_identical(
            fetch(stderr_uid), child._param_plot_dataarray(param_name, stderr=True)
        )
        values_code = copy_full_code_for_uid(monkeypatch, manager, values_uid)
        stderr_code = copy_full_code_for_uid(monkeypatch, manager, stderr_uid)
        assert "prepared_parent = data + 1" in values_code
        assert "prepared_parent = data + 1" in stderr_code
        assert ".modelfit_coefficients.sel(param=" in values_code
        assert ".modelfit_stderr.sel(param=" in stderr_code


def test_manager_output_refresh_updates_stale_parent_source(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    prov = erlab.interactive.imagetool.provenance_framework

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
        ) -> erlab.interactive.imagetool.provenance_framework.ToolProvenanceSpec | None:
            assert output_id == "out"
            return prov.script(
                prov.ScriptCodeOperation(label="Use output", code="result = data + 10"),
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
        child.set_source_binding(prov.full_data(), auto_update=False)

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
        first_output_node = manager._imagetool_wrappers[1]
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
        second_output_node = manager._imagetool_wrappers[2]
        assert second_output_node.parent_uid is None
        assert second_output_node.output_id is None
        assert second_output_node.source_spec is None
        assert second_output_node.provenance_spec is None
        assert not second_output_node.reloadable
        xr.testing.assert_identical(fetch(2), updated)


def test_manager_open_in_new_window_nests_imagetool_children(
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

        parent = manager._imagetool_wrappers[0]
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
        qtbot.waitUntil(
            lambda: (
                getattr(file_label, "full_text", file_label.text()) != file_label.text()
            ),
            timeout=2000,
        )

        def _inspect_source_dialog(dialog: QtWidgets.QDialog) -> None:
            assert dialog.path_edit.text() == str(file_path)  # type: ignore[attr-defined]
            assert (
                dialog.loader_edit.text().endswith("xarray.load_dataarray")  # type: ignore[attr-defined]
            )
            assert (
                dialog.kwargs_edit.toPlainText() == 'engine="h5netcdf"'  # type: ignore[attr-defined]
            )
            dialog.copy_code_button.click()  # type: ignore[attr-defined]

        accept_dialog(
            lambda: qtbot.mouseClick(file_label, QtCore.Qt.MouseButton.LeftButton),
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
        assert not erlab.interactive.imagetool.provenance_framework.uses_default_replay_input(
            copied[-1]
        )
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

        parent = manager._imagetool_wrappers[0]
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

        parent = manager._imagetool_wrappers[0]
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

        parent = manager._imagetool_wrappers[0]
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

        parent = manager._imagetool_wrappers[0]
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
        promoted = manager._imagetool_wrappers[promoted_index]
        assert promoted.uid == child_uid
        assert child_uid not in parent._childtool_indices
        assert promoted.parent_uid is None
        assert promoted.source_spec is None
        assert promoted.provenance_spec is not None
        assert promoted._childtool_indices == [nested_uid]
        assert manager._child_node(nested_uid).parent_uid == child_uid
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

        root = manager._imagetool_wrappers[0]
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

        parent = manager._imagetool_wrappers[0]
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
    prov = erlab.interactive.imagetool.provenance_framework
    data = xr.DataArray(
        np.arange(12).reshape((3, 4)).astype(float),
        dims=["x", "y"],
        coords={"x": np.arange(3, dtype=float), "y": np.arange(4, dtype=float)},
        name="scan",
    )
    operation = prov.GaussianFilterOperation(sigma={"x": 1.0})

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
            source_spec=prov.full_data(),
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

        root = manager._imagetool_wrappers[0]
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
        detached = manager._imagetool_wrappers[1]
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
        loaded = manager._imagetool_wrappers[0]
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

        parent = manager._imagetool_wrappers[0]
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

        parent = manager._imagetool_wrappers[0]
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
                "erlab.interactive.imagetool.provenance_framework.ToolProvenanceSpec",
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
        select_metadata_rows(manager, [0])
        qtbot.keyClick(
            manager.metadata_derivation_list,
            QtCore.Qt.Key.Key_C,
            QtCore.Qt.KeyboardModifier.ControlModifier,
        )
        assert copied == []

        select_metadata_rows(manager, [1, 2])
        qtbot.keyClick(
            manager.metadata_derivation_list,
            QtCore.Qt.Key.Key_C,
            QtCore.Qt.KeyboardModifier.ControlModifier,
        )
        selected_namespace = _exec_generated_code(
            copied[-1],
            {"derived": data.copy(deep=True)},
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
            copied[-1],
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
        assert not erlab.interactive.imagetool.provenance_framework.uses_default_replay_input(
            copied[-1]
        )
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

        detached = manager._imagetool_wrappers[1]
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
        duplicated_detached = manager._imagetool_wrappers[duplicated_detached_index]
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

        parent = manager._imagetool_wrappers[0]
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
        assert not erlab.interactive.imagetool.provenance_framework.uses_default_replay_input(
            copied[-1]
        )
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

        parent = manager._imagetool_wrappers[0]
        qtbot.wait_until(lambda: len(parent._childtool_indices) == 1, timeout=5000)
        child_uid = parent._childtool_indices[0]
        child_node = manager._child_node(child_uid)
        child_tool = manager.get_imagetool(child_uid)

        operation = (
            erlab.interactive.imagetool.provenance_framework.AffineCoordOperation(
                coord_name="y",
                scale=2.0,
                offset=0.5,
            )
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

        parent = manager._imagetool_wrappers[0]
        qtbot.wait_until(lambda: len(parent._childtool_indices) == 1, timeout=5000)
        child_uid = parent._childtool_indices[0]
        child_node = manager._child_node(child_uid)
        child_tool = manager.get_imagetool(child_uid)

        operation = (
            erlab.interactive.imagetool.provenance_framework.AssignAttrsOperation(
                attrs={"source": "new", "flag": True}
            )
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

        assert manager._displayed_indices == [0, 1, 2]

        # Remove tool at index 1
        manager.remove_imagetool(1)
        qtbot.wait_until(lambda: manager.ntools == 2)

        assert manager._displayed_indices == [0, 2]

        # Reindex
        manager.reindex_action.trigger()
        qtbot.wait_until(lambda: manager._displayed_indices == [0, 1], timeout=5000)


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
                "workspace_link_id": manager._workspace_link_id,
                "source_label": "notebook-a",
                "source_uid": "kernel-a",
                "connected": True,
            },
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        workspace_link_id = manager._workspace_link_id
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
        manager._workspace_link_id = "different-workspace-link"

        manager._load_workspace_node(typing.cast("xr.DataTree", tree["0"]))
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        wrapper = manager._imagetool_wrappers[0]
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
        assert manager._workspace_link_id == workspace_link_id


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

        left = manager._imagetool_wrappers[0]
        right = manager._imagetool_wrappers[1]
        other = manager._imagetool_wrappers[2]

        assert (
            manager.color_for_watched_var_source(left)
            == manager_mainwindow._WATCHED_VAR_COLORS[0]
        )
        assert (
            manager.color_for_watched_var_source(right)
            == manager_mainwindow._WATCHED_VAR_COLORS[0]
        )
        assert (
            manager.color_for_watched_var_source(other)
            == manager_mainwindow._WATCHED_VAR_COLORS[1]
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

        left = manager._imagetool_wrappers[0]
        right = manager._imagetool_wrappers[1]
        other = manager._imagetool_wrappers[2]
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

        left = manager._imagetool_wrappers[0]
        right = manager._imagetool_wrappers[1]
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

        node = manager._imagetool_wrappers[0]
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

        node = manager._imagetool_wrappers[0]
        node.set_detached_provenance(
            erlab.interactive.imagetool.provenance_framework.full_data(
                erlab.interactive.imagetool.provenance_framework.AverageOperation(
                    dims=("alpha",)
                )
            )
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
        assert not erlab.interactive.imagetool.provenance_framework.uses_default_replay_input(
            copied[-1]
        )
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

        node = manager._imagetool_wrappers[0]
        node.set_detached_provenance(
            erlab.interactive.imagetool.provenance_framework.full_data(
                erlab.interactive.imagetool.provenance_framework.AverageOperation(
                    dims=("alpha",)
                )
            )
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

        node = manager._imagetool_wrappers[0]
        node.set_detached_provenance(
            erlab.interactive.imagetool.provenance_framework.full_data(
                erlab.interactive.imagetool.provenance_framework.AverageOperation(
                    dims=("alpha",)
                )
            )
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
        assert not erlab.interactive.imagetool.provenance_framework.uses_default_replay_input(
            copied[-1]
        )
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

        node = manager._imagetool_wrappers[0]
        node.set_detached_provenance(
            erlab.interactive.imagetool.provenance_framework.full_data(
                erlab.interactive.imagetool.provenance_framework.AverageOperation(
                    dims=("alpha",)
                )
            )
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
            lambda: len(manager._imagetool_wrappers[0]._childtool_indices) == 1,
            timeout=5000,
        )

        child_uid = manager._imagetool_wrappers[0]._childtool_indices[0]
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
            lambda: len(manager._imagetool_wrappers[0]._childtool_indices) == 1,
            timeout=5000,
        )

        child_uid = manager._imagetool_wrappers[0]._childtool_indices[0]
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
            lambda: len(manager._imagetool_wrappers[0]._childtool_indices) == 1,
            timeout=5000,
        )

        child_uid = manager._imagetool_wrappers[0]._childtool_indices[0]
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
            lambda: len(manager._imagetool_wrappers[0]._childtool_indices) == 1,
            timeout=5000,
        )

        child_uid = manager._imagetool_wrappers[0]._childtool_indices[0]
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
            lambda: len(manager._imagetool_wrappers[0]._childtool_indices) == 1,
            timeout=5000,
        )

        child_uid = manager._imagetool_wrappers[0]._childtool_indices[0]
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
            lambda: len(manager._imagetool_wrappers[0]._childtool_indices) == 1,
            timeout=5000,
        )

        child_uid = manager._imagetool_wrappers[0]._childtool_indices[0]
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
                len(manager._imagetool_wrappers[duplicated]._childtool_indices) == 1
            ),
            timeout=5000,
        )

        child_uid = manager._imagetool_wrappers[duplicated]._childtool_indices[0]
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
            lambda: len(manager._imagetool_wrappers[0]._childtool_indices) == 1,
            timeout=5000,
        )

        child_uid = manager._imagetool_wrappers[0]._childtool_indices[0]
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
