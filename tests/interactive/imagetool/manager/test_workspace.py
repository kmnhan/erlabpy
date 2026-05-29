# ruff: noqa: F403, F405
from ._shared import *


def test_manager_duplicate(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        # Open a tool with the manager
        itool([test_data, test_data], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        select_tools(manager, [0, 1])
        manager.duplicate_selected()
        qtbot.wait_until(lambda: manager.ntools == 4, timeout=5000)

        # Check if the duplicated tools have the same data
        for i in range(2):
            original_tool = manager.get_imagetool(i)
            duplicated_tool = manager.get_imagetool(i + 2)

            assert original_tool.slicer_area._data.equals(
                duplicated_tool.slicer_area._data
            )
            assert (
                manager._imagetool_wrappers[i].name
                == manager._imagetool_wrappers[i + 2].name
            )


def test_workspace_backing_uses_persistence_data_for_filtered_file_data(
    qtbot,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    file_path = tmp_path / "scan.h5"
    data = xr.DataArray(
        np.arange(25, dtype=float).reshape((5, 5)),
        dims=["x", "y"],
        coords={"x": np.arange(5, dtype=float), "y": np.arange(5, dtype=float)},
        name="scan",
    )
    data.to_netcdf(file_path, engine="h5netcdf")
    operation = erlab.interactive.imagetool.provenance.GaussianFilterOperation(
        sigma={"x": 1.0}
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        opened = xr.open_dataarray(file_path, engine="h5netcdf")
        try:
            itool(opened, manager=True)
            qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
            tool = manager.get_imagetool(0)
            tool.slicer_area.apply_filter_operation(operation, emit_edited=True)

            uid = manager._imagetool_wrappers[0].uid
            entry = next(
                item
                for item in manager._workspace_node_manifest_entries()
                if item["uid"] == uid
            )
            assert entry["data_backing"] == "file_lazy"
            snapshot = manager._workspace_data_backing_snapshot()
            assert snapshot[uid][0] == "file_lazy"
            assert str(file_path.resolve()) in snapshot[uid][1]
        finally:
            opened.close()


def test_manager_duplicate_goldtool_child(
    qtbot,
    monkeypatch,
    gold,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()

        itool(gold, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child = GoldTool(gold.copy(deep=True), data_name="gold_input")
        child_uid = manager.add_childtool(child, 0, show=False)
        configure_goldtool_child(child, fitted=True, spline=True)
        child.open_itool()

        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)

        select_child_tool(manager, child_uid)
        manager.duplicate_selected()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 2,
            timeout=5000,
        )

        duplicate_uid = next(
            uid
            for uid in manager._imagetool_wrappers[0]._childtool_indices
            if uid != child_uid
        )
        duplicated = manager.get_childtool(duplicate_uid)

        assert isinstance(duplicated, GoldTool)
        assert duplicated is not child
        assert duplicated.tool_status == child.tool_status
        xr.testing.assert_identical(duplicated.corrected, child.corrected)

        duplicate_node = manager._child_node(duplicate_uid)
        qtbot.wait_until(
            lambda: len(duplicate_node._childtool_indices) == 1, timeout=5000
        )
        duplicate_output_uid = duplicate_node._childtool_indices[0]
        duplicate_output_node = manager._child_node(duplicate_output_uid)
        assert duplicate_output_node.output_id == "goldtool.corrected"
        assert duplicate_output_node.source_spec is None
        assert duplicate_output_node.provenance_spec is not None
        xr.testing.assert_identical(fetch(duplicate_output_uid), duplicated.corrected)

        monkeypatch.setattr(
            duplicated, "_prompt_existing_output_imagetool", lambda: "update"
        )
        duplicated.open_itool()
        assert duplicate_node._childtool_indices == [duplicate_output_uid]


def test_manager_sync(
    qtbot,
    move_and_compare_values,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool([test_data, test_data], link=True, link_colors=True, manager=True)

        qtbot.wait_until(lambda: manager.ntools == 2)

        win0, win1 = manager.get_imagetool(0), manager.get_imagetool(1)
        win0.show()
        win1.show()

        win1.slicer_area.set_colormap("RdYlBu", gamma=1.5)
        assert (
            win0.slicer_area._colormap_properties
            == win1.slicer_area._colormap_properties
        )

        move_and_compare_values(qtbot, win0, [12.0, 7.0, 6.0, 11.0], target_win=win1)

        # Transpose
        win0.slicer_area.transpose_main_image()
        move_and_compare_values(qtbot, win0, [12.0, 11.0, 6.0, 7.0], target_win=win1)

        # Set bin
        win1.slicer_area.set_bin(0, 2, update=False)
        win1.slicer_area.set_bin(1, 2, update=True)

        # Set all bins, same effect as above since we only have 1 cursor
        win1.slicer_area.set_bin_all(1, 2, update=True)

        move_and_compare_values(qtbot, win0, [9.0, 8.0, 3.0, 4.0], target_win=win1)

        # Change limits
        win0.slicer_area.main_image.getViewBox().setRange(xRange=[2, 3], yRange=[1, 2])
        # Trigger manual range propagation
        win0.slicer_area.main_image.getViewBox().sigRangeChangedManually.emit(
            win0.slicer_area.main_image.getViewBox().state["mouseEnabled"][:]
        )
        assert win1.slicer_area.main_image.getViewBox().viewRange() == [[2, 3], [1, 2]]


def test_manager_link_action_links_colors(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool([test_data, test_data], link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        select_tools(manager, [0, 1])
        qtbot.wait_until(lambda: manager.link_action.isEnabled())
        manager.link_action.trigger()

        proxy = manager.get_imagetool(0).slicer_area._linking_proxy
        assert proxy is not None
        assert proxy.link_colors is True

        control = (
            manager.get_imagetool(0).docks[1].widget().findChild(ItoolColormapControls)
        )
        assert control is not None
        control._set_gamma(1.5)
        assert manager.get_imagetool(1).slicer_area.colormap_properties[
            "gamma"
        ] == pytest.approx(1.5)

        manager.get_imagetool(0).slicer_area.undo()
        assert manager.get_imagetool(0).slicer_area.colormap_properties[
            "gamma"
        ] == pytest.approx(0.5)
        assert manager.get_imagetool(1).slicer_area.colormap_properties[
            "gamma"
        ] == pytest.approx(0.5)


def test_manager_workspace_io(
    qtbot,
    accept_dialog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()

        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        # Add two tools
        itool([data, data], link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        # Open dtool for first tool
        manager.get_imagetool(0).slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 1, timeout=5000
        )

        # Save and load workspace
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            filename = f"{tmp_dir_name}/workspace.itws"

            def _go_to_file(dialog: QtWidgets.QFileDialog):
                dialog.setDirectory(tmp_dir_name)
                dialog.selectFile(filename)
                focused = dialog.focusWidget()
                if isinstance(focused, QtWidgets.QLineEdit):
                    focused.setText("workspace.itws")

            # Save workspace
            accept_dialog(
                lambda: manager.save(native=False),
                pre_call=_go_to_file,
            )
            assert manager.workspace_path == str(pathlib.Path(filename).resolve())
            assert not manager.is_workspace_modified

            # Load workspace
            accept_dialog(lambda: manager.load(native=False), pre_call=_go_to_file)

            # Check if the data is loaded
            assert manager.ntools == 2

            # Check if the child dtool is also loaded
            assert len(manager._imagetool_wrappers[0]._childtools) == 1

            select_tools(manager, list(manager._imagetool_wrappers.keys()))
            accept_dialog(manager.remove_action.trigger)
            qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)


def test_manager_workspace_preserves_link_groups(
    qtbot,
    tmp_path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool([test_data, test_data, test_data], link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)

        fname = tmp_path / "linked.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert not manager.is_workspace_modified

        manager.link_imagetools(0, 1, link_colors=False)
        assert manager.is_workspace_modified
        manager._save_workspace_document(fname, force_full=True)

        manifest = manager_workspace._workspace_manifest_from_attrs(
            manager_workspace._read_workspace_root_attrs_h5py(fname)
        )
        linked_entries = [entry for entry in manifest["nodes"] if "link_group" in entry]
        assert {entry["path"] for entry in linked_entries} == {"0", "1"}
        assert {entry["link_group"] for entry in linked_entries} == {0}
        assert {entry["link_colors"] for entry in linked_entries} == {False}

        assert manager._load_workspace_file(
            fname,
            replace=True,
            associate=True,
            mark_dirty=False,
            select=False,
        )
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)

        proxy0 = manager.get_imagetool(0).slicer_area._linking_proxy
        proxy1 = manager.get_imagetool(1).slicer_area._linking_proxy
        assert proxy0 is not None
        assert proxy0 is proxy1
        assert proxy0.link_colors is False
        assert not manager.get_imagetool(2).slicer_area.is_linked
        assert not manager.is_workspace_modified


def test_manager_workspace_unlink_removes_saved_link_group(
    qtbot,
    tmp_path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool([test_data, test_data], link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        fname = tmp_path / "unlinked.itws"
        manager.link_imagetools(0, 1)
        manager._save_workspace_document(fname, force_full=True)

        select_tools(manager, [0, 1])
        manager.unlink_selected()
        assert manager.is_workspace_modified
        manager._save_workspace_document(fname, force_full=True)

        manifest = manager_workspace._workspace_manifest_from_attrs(
            manager_workspace._read_workspace_root_attrs_h5py(fname)
        )
        assert all("link_group" not in entry for entry in manifest["nodes"])

        assert manager._load_workspace_file(
            fname,
            replace=True,
            associate=True,
            mark_dirty=False,
            select=False,
        )
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        assert not manager.get_imagetool(0).slicer_area.is_linked
        assert not manager.get_imagetool(1).slicer_area.is_linked
        assert not manager.is_workspace_modified


def test_manager_workspace_save_selection_cancel_does_not_write(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    class _RejectedChooseDialog:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def exec(self) -> QtWidgets.QDialog.DialogCode:
            return QtWidgets.QDialog.DialogCode.Rejected

    monkeypatch.setattr(
        erlab.interactive.imagetool.manager._mainwindow,
        "_ChooseFromDataTreeDialog",
        _RejectedChooseDialog,
    )
    closed_trees: list[xr.DataTree] = []
    original_close = xr.DataTree.close

    def _close_spy(tree: xr.DataTree) -> None:
        closed_trees.append(tree)
        original_close(tree)

    monkeypatch.setattr(xr.DataTree, "close", _close_spy)

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])
        itool(data, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            filename = pathlib.Path(tmp_dir_name) / "workspace.itws"
            manager._save_to_file(str(filename))

            assert not filename.exists()
            assert len(closed_trees) == 1


def test_manager_workspace_load_selection_skips_unchecked_children(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    class _SelectedChooseDialog(
        erlab.interactive.imagetool.manager._mainwindow._ChooseFromDataTreeDialog
    ):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            root_item = self._tree_widget.topLevelItem(0)
            assert root_item is not None
            unchecked_child = root_item.child(1)
            assert unchecked_child is not None
            unchecked_child.setCheckState(0, QtCore.Qt.CheckState.Unchecked)

        def exec(self) -> QtWidgets.QDialog.DialogCode:
            return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(
        erlab.interactive.imagetool.manager._mainwindow,
        "_ChooseFromDataTreeDialog",
        _SelectedChooseDialog,
    )

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root_tool = itool(data, manager=False, execute=False)
        assert isinstance(root_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root_tool, show=False)

        child_uids: list[str] = []
        for offset in (1.0, 2.0):
            child_tool = itool(data + offset, manager=False, execute=False)
            assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
            child_uids.append(manager.add_imagetool_child(child_tool, 0, show=False))

        tree = manager._to_datatree()
        try:
            manager.remove_all_tools()
            qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

            manager._from_datatree(tree)
        finally:
            tree.close()

        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        assert manager._imagetool_wrappers[0]._childtool_indices == [child_uids[0]]
        assert child_uids[1] not in manager._all_nodes


def _open_external_hdf5_imagetool_data(
    fname: pathlib.Path, *, chunks: str | None = None
) -> xr.DataArray:
    open_kwargs: dict[str, typing.Any] = {
        "engine": "h5netcdf",
        "phony_dims": "sort",
    }
    if chunks is not None:
        open_kwargs["chunks"] = chunks
    tree = xr.open_datatree(fname, **open_kwargs)
    try:
        ds = typing.cast("xr.DataTree", tree["0/imagetool"]).to_dataset(inherit=False)
        return ds[next(iter(ds.data_vars))]
    finally:
        tree.close()


def _open_external_lazy_hdf5_imagetool_data(fname: pathlib.Path) -> xr.DataArray:
    return _open_external_hdf5_imagetool_data(fname, chunks="auto")


def _open_external_file_backed_hdf5_imagetool_data(
    fname: pathlib.Path,
) -> xr.DataArray:
    return _open_external_hdf5_imagetool_data(fname)


def _compute_first_value(darr: xr.DataArray) -> object:
    return darr.isel(dict.fromkeys(darr.dims, 0)).compute().item()


def _hdf5_filter_ids(dataset) -> list[int]:
    create_plist = dataset.id.get_create_plist()
    return [create_plist.get_filter(i)[0] for i in range(create_plist.get_nfilters())]


def _transaction_test_root_attrs(delta_save_count: int = 0) -> dict[str, object]:
    manifest: dict[str, object] = {
        "schema_version": 4,
        "root_order": [0],
        "nodes": [],
    }
    if delta_save_count > 0:
        manifest["transaction_protocol"] = (
            manager_workspace._WORKSPACE_TRANSACTION_PROTOCOL
        )
        manifest["delta_save_count"] = delta_save_count
    return {
        "imagetool_workspace_schema_version": 4,
        manager_workspace._WORKSPACE_MANIFEST_ATTR: json.dumps(manifest),
    }


def _transaction_test_dataset(value: float, *, title: str) -> xr.Dataset:
    ds = xr.Dataset({"data": ("x", np.array([value], dtype=np.float64))})
    ds.attrs["itool_title"] = title
    return ds


def _write_transaction_test_workspace(fname: pathlib.Path, value: float = 1.0) -> None:
    tree = xr.DataTree.from_dict(
        {"0/imagetool": _transaction_test_dataset(value, title="old")}
    )
    try:
        manager_workspace._write_full_workspace_tree_file(
            fname, tree, _transaction_test_root_attrs()
        )
    finally:
        tree.close()


def _read_transaction_test_value(fname: pathlib.Path) -> float:
    opened = manager_xarray.open_workspace_datatree(fname, chunks=None)
    try:
        ds = typing.cast("xr.DataTree", opened["/0/imagetool"]).to_dataset(
            inherit=False
        )
        return float(ds["data"].item())
    finally:
        opened.close()


def _assert_no_workspace_internal_groups(fname: pathlib.Path) -> None:
    import h5py

    with h5py.File(fname, "r") as h5_file:
        assert not any(
            manager_workspace._is_workspace_internal_group_name(name)
            for name in h5_file
        )


def test_workspace_dataset_encoding_compresses_only_large_numeric_payloads() -> None:
    import hdf5plugin

    ds = xr.Dataset(
        {
            "large": (
                ("x", "y"),
                np.arange(512 * 512, dtype=np.float64).reshape(512, 512),
            ),
            "small": ("x", np.arange(512, dtype=np.float64)),
            "metadata": ("label", np.array(["a", "b"], dtype=object)),
        },
        coords={
            "x": np.linspace(-1.0, 1.0, 512),
            "y": np.linspace(-2.0, 2.0, 512),
            "label": ["a", "b"],
        },
    )

    encoding = manager_xarray.workspace_dataset_encoding(ds)

    assert set(encoding) == {"large"}
    assert encoding["large"] == dict(
        hdf5plugin.Blosc2(
            cname="blosclz",
            clevel=3,
            filters=hdf5plugin.Blosc2.SHUFFLE,
        )
    )


def test_workspace_dataset_encoding_respects_compression_preference() -> None:
    ds = xr.Dataset(
        {
            "large": (
                ("x", "y"),
                np.arange(512 * 512, dtype=np.float64).reshape(512, 512),
            )
        }
    )
    old_value = erlab.interactive.options["io/workspace/compress"]
    try:
        erlab.interactive.options["io/workspace/compress"] = False
        assert manager_xarray.workspace_dataset_encoding(ds) == {}

        erlab.interactive.options["io/workspace/compress"] = True
        assert set(manager_xarray.workspace_dataset_encoding(ds)) == {"large"}
    finally:
        erlab.interactive.options["io/workspace/compress"] = old_value


def test_workspace_dataset_encoding_persists_dask_chunksizes() -> None:
    data = xr.DataArray(
        np.arange(25, dtype=np.float64).reshape(5, 5),
        dims=("x", "y"),
    ).chunk({"x": (2, 3), "y": (4, 1)})
    ds = xr.Dataset({"data": data})

    assert manager_xarray.workspace_dataset_encoding(ds, compress=False) == {
        "data": {"chunksizes": (2, 4)}
    }


def test_workspace_chunksizes_rejects_invalid_chunk_shapes() -> None:
    assert (
        manager_xarray._workspace_chunksizes_for_dataarray(
            types.SimpleNamespace(chunks=((1,),), ndim=1, shape=(0,))
        )
        is None
    )
    assert (
        manager_xarray._workspace_chunksizes_for_dataarray(
            types.SimpleNamespace(chunks=((0,),), ndim=1, shape=(5,))
        )
        is None
    )


def test_workspace_datatree_encoding_uses_group_paths() -> None:
    large_ds = xr.Dataset(
        {
            "data": (
                ("x", "y"),
                np.arange(512 * 512, dtype=np.float64).reshape(512, 512),
            )
        },
        coords={"x": np.arange(512, dtype=np.float64), "y": np.arange(512)},
    )
    small_ds = xr.Dataset({"data": ("x", np.arange(4, dtype=np.float64))})
    tree = xr.DataTree.from_dict({"0/imagetool": large_ds, "1/imagetool": small_ds})
    try:
        encoding = manager_xarray.workspace_datatree_encoding(tree)
    finally:
        tree.close()

    assert set(encoding) == {"/0/imagetool"}
    assert set(encoding["/0/imagetool"]) == {"data"}


def test_workspace_datatree_encoding_can_be_disabled() -> None:
    tree = xr.DataTree.from_dict(
        {
            "0/imagetool": xr.Dataset(
                {
                    "data": (
                        ("x", "y"),
                        np.arange(512 * 512, dtype=np.float64).reshape(512, 512),
                    )
                }
            )
        }
    )
    try:
        assert manager_xarray.workspace_datatree_encoding(tree, compress=False) == {}
    finally:
        tree.close()


def test_workspace_xarray_path_helpers_cover_fallbacks(monkeypatch, tmp_path) -> None:
    class _BadPath(os.PathLike):
        def __fspath__(self) -> str:
            raise TypeError

    assert manager_xarray._normalized_file_path(object()) is None
    assert manager_xarray._normalized_file_path(_BadPath()) is None
    assert manager_xarray._normalized_file_path("") is None

    def _raise_oserror(_path: pathlib.Path) -> pathlib.Path:
        raise OSError("resolve failed")

    monkeypatch.setattr(pathlib.Path, "resolve", _raise_oserror)
    assert manager_xarray._normalized_file_path(tmp_path / "workspace.itws") == str(
        tmp_path / "workspace.itws"
    )

    monkeypatch.setattr(manager_xarray, "_normalized_file_path", lambda _path: None)
    lock = manager_xarray._workspace_file_lock("fallback.itws")
    assert lock is manager_xarray._workspace_file_lock("fallback.itws")

    def _raise_stat_oserror(_path: str):
        raise OSError

    monkeypatch.setattr(manager_xarray.os, "stat", _raise_stat_oserror)
    assert manager_xarray._workspace_file_identity("missing.itws") == (
        "missing.itws",
        0,
        0,
        0,
    )


def test_workspace_file_manager_uses_fsdecode_fallback(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_init(self, opener, *args, **kwargs):
        captured["opener"] = opener
        captured["args"] = args
        captured["kwargs"] = kwargs
        self._key = "fake-key"
        self._ref_counter = types.SimpleNamespace(decrement=lambda _key: None)
        self._cache = {}

    monkeypatch.setattr(
        manager_xarray, "ensure_workspace_hdf5_filters_registered", lambda: None
    )
    monkeypatch.setattr(manager_xarray, "_normalized_file_path", lambda _path: None)
    monkeypatch.setattr(
        manager_xarray, "_workspace_file_identity", lambda path: (path, 0, 0, 0)
    )
    monkeypatch.setattr(manager_xarray.CachingFileManager, "__init__", _fake_init)

    file_manager = manager_xarray.WorkspaceFileManager("fallback.itws")

    assert file_manager.workspace_path == "fallback.itws"
    assert captured["args"][0] == "fallback.itws"


def test_open_workspace_dataset_uses_fsdecode_fallback(monkeypatch) -> None:
    calls: list[tuple[object, str, str | None]] = []

    class _FakeFileManager:
        def __init__(self, path: str) -> None:
            self.workspace_path = path

    def _fake_open(file_manager, group: str, *, chunks: str | None):
        calls.append((file_manager, group, chunks))
        return "dataset"

    monkeypatch.setattr(manager_xarray, "_normalized_file_path", lambda _path: None)
    monkeypatch.setattr(manager_xarray, "WorkspaceFileManager", _FakeFileManager)
    monkeypatch.setattr(
        manager_xarray, "_open_workspace_dataset_from_manager", _fake_open
    )

    assert (
        manager_xarray.open_workspace_dataset("fallback.itws", "/0", chunks=None)
        == "dataset"
    )
    file_manager, group, chunks = calls[0]
    assert isinstance(file_manager, _FakeFileManager)
    assert file_manager.workspace_path == "fallback.itws"
    assert group == "/0"
    assert chunks is None


def test_open_workspace_datatree_closes_partial_groups_on_error(monkeypatch) -> None:
    closed: list[str] = []

    class _FakeDataset:
        def __init__(self, group_path: str) -> None:
            self.group_path = group_path

        def close(self) -> None:
            closed.append(self.group_path)

    class _FakeFileManager:
        workspace_path = "fallback.itws"

        def __init__(self, _path: str) -> None:
            pass

        def acquire_context(self):
            return contextlib.nullcontext(object())

    def _fake_open(_file_manager, group_path: str, *, chunks: str | None):
        if group_path == "/broken":
            raise RuntimeError("broken group")
        return _FakeDataset(group_path)

    monkeypatch.setattr(manager_xarray, "_normalized_file_path", lambda _path: None)
    monkeypatch.setattr(manager_xarray, "WorkspaceFileManager", _FakeFileManager)
    monkeypatch.setattr(
        manager_xarray, "_iter_h5netcdf_group_paths", lambda _h5_file: ("/", "/broken")
    )
    monkeypatch.setattr(
        manager_xarray, "_open_workspace_dataset_from_manager", _fake_open
    )

    with pytest.raises(RuntimeError, match="broken group"):
        manager_xarray.open_workspace_datatree("fallback.itws", chunks="auto")

    assert closed == ["/"]


def test_write_full_workspace_tree_file_compresses_payload_not_coords(
    tmp_path,
) -> None:
    import h5py
    import hdf5plugin

    ds = xr.Dataset(
        {
            "data": (
                ("x", "y"),
                np.arange(512 * 512, dtype=np.float64).reshape(512, 512),
            ),
            "small": ("x", np.arange(512, dtype=np.int64)),
        },
        coords={
            "x": np.linspace(-1.0, 1.0, 512),
            "y": np.linspace(-2.0, 2.0, 512),
        },
    )
    tree = xr.DataTree.from_dict({"0/imagetool": ds})
    fname = tmp_path / "compressed.itws"
    try:
        manager_workspace._write_full_workspace_tree_file(
            fname, tree, {"imagetool_workspace_schema_version": 4}
        )
    finally:
        tree.close()

    with h5py.File(fname, "r") as h5_file:
        assert hdf5plugin.Blosc2.filter_id in _hdf5_filter_ids(
            h5_file["0/imagetool/data"]
        )
        assert _hdf5_filter_ids(h5_file["0/imagetool/x"]) == []
        assert _hdf5_filter_ids(h5_file["0/imagetool/y"]) == []
        assert _hdf5_filter_ids(h5_file["0/imagetool/small"]) == []

    opened = manager_xarray.open_workspace_datatree(fname, chunks=None)
    try:
        loaded = typing.cast("xr.DataTree", opened["/0/imagetool"]).to_dataset(
            inherit=False
        )
        xarray.testing.assert_equal(loaded["data"], ds["data"])
        xarray.testing.assert_equal(loaded["x"], ds["x"])
        xarray.testing.assert_equal(loaded["y"], ds["y"])
    finally:
        opened.close()


def test_open_workspace_datatree_reads_uncompressed_workspace(tmp_path) -> None:
    ds = xr.Dataset(
        {"data": (("x", "y"), np.arange(12, dtype=np.float64).reshape(3, 4))},
        coords={"x": np.arange(3), "y": np.arange(4)},
    )
    tree = xr.DataTree.from_dict({"0/imagetool": ds})
    fname = tmp_path / "uncompressed.itws"
    try:
        tree.to_netcdf(fname, engine="h5netcdf", invalid_netcdf=True)
    finally:
        tree.close()

    opened = manager_xarray.open_workspace_datatree(fname, chunks=None)
    try:
        loaded = typing.cast("xr.DataTree", opened["/0/imagetool"]).to_dataset(
            inherit=False
        )
        xarray.testing.assert_equal(loaded["data"], ds["data"])
    finally:
        opened.close()


def test_imagetool_private_coord_serialization_edge_cases() -> None:
    private_attr = imagetool_serialization._PRIVATE_COORDS_ATTR
    private_prefix = imagetool_serialization._PRIVATE_COORD_VAR_PREFIX
    data_name = imagetool_serialization.ITOOL_DATA_NAME
    valid_payload = json.dumps(
        [{"coord_name": "Fake Motor", "variable_name": "private", "dims": ["x"]}]
    )

    assert imagetool_serialization.private_coord_records_from_attrs(
        {private_attr: valid_payload.encode()}
    ) == ({"coord_name": "Fake Motor", "variable_name": "private", "dims": ("x",)},)
    assert (
        imagetool_serialization.private_coord_records_from_attrs({private_attr: 1})
        == ()
    )
    assert (
        imagetool_serialization.private_coord_records_from_attrs(
            {private_attr: "{not-json"}
        )
        == ()
    )
    assert (
        imagetool_serialization.private_coord_records_from_attrs(
            {private_attr: json.dumps([[]])}
        )
        == ()
    )
    assert (
        imagetool_serialization.private_coord_records_from_attrs(
            {private_attr: json.dumps([{"coord_name": "Fake Motor", "dims": ["x"]}])}
        )
        == ()
    )
    assert (
        imagetool_serialization.private_coord_variable_names(
            xr.Dataset({"other": ("x", [1.0])})
        )
        == ()
    )

    ds = xr.Dataset(
        {
            data_name: ("x", np.arange(2.0)),
            f"{private_prefix}0": ("x", np.arange(2.0) + 10.0),
        },
        coords={"x": np.arange(2.0), "Fake Motor": ("x", np.arange(2.0) + 20.0)},
    )
    encoded = imagetool_serialization.encode_private_coords(ds)

    assert imagetool_serialization.private_coord_variable_names(encoded) == (
        f"{private_prefix}1",
    )
    restored = imagetool_serialization.restore_private_coords(encoded)
    xr.testing.assert_equal(restored.coords["Fake Motor"], ds.coords["Fake Motor"])


def test_imagetool_private_coord_restore_ignores_invalid_records() -> None:
    private_attr = imagetool_serialization._PRIVATE_COORDS_ATTR
    data_name = imagetool_serialization.ITOOL_DATA_NAME
    missing_data = xr.Dataset({"other": ("x", [1.0])})

    assert imagetool_serialization.restore_private_coords(missing_data) is missing_data

    payload = json.dumps(
        [
            {"coord_name": "Missing", "variable_name": "missing", "dims": ["x"]},
            {"coord_name": "Bad Dims", "variable_name": "present", "dims": ["z"]},
        ]
    )
    encoded = xr.Dataset(
        {
            data_name: ("x", [1.0]),
            "present": ("z", [2.0]),
        },
        attrs={"root": "kept"},
    )
    encoded[data_name].attrs[private_attr] = payload

    restored = imagetool_serialization.restore_private_coords(encoded)

    assert private_attr not in restored[data_name].attrs
    assert "Missing" not in restored.coords
    assert "Bad Dims" not in restored.coords
    assert "present" in restored.data_vars

    legacy = xr.Dataset(
        {
            data_name: ("x", [1.0]),
            "plain": ("x", [2.0]),
            "Fake Motor": ("z", [3.0]),
        }
    )

    legacy_restored = imagetool_serialization.restore_private_coords(legacy)

    assert "plain" in legacy_restored.data_vars
    assert "Fake Motor" in legacy_restored.data_vars


def test_workspace_h5py_attrs_and_root_validation(tmp_path) -> None:
    import h5py

    assert manager_workspace._h5py_attrs_to_dict({"name": b"value"}) == {
        "name": "value"
    }

    fname = tmp_path / "plain.h5"
    with h5py.File(fname, "w"):
        pass

    with pytest.raises(ValueError, match="Not a valid workspace file"):
        manager_workspace._read_workspace_root_attrs_h5py(fname)


def _assert_workspace_h5py_roundtrip(
    tmp_path: pathlib.Path, label: str, data: xr.DataArray
) -> tuple[xr.Dataset, xr.Dataset, pathlib.Path]:
    data_name = manager_mainwindow._ITOOL_DATA_NAME
    fname = tmp_path / f"{label}.itws"
    ds = data.rename(data_name).to_dataset()

    assert manager_workspace._workspace_dataset_can_write_h5py(ds)
    assert manager_workspace._write_workspace_dataset_group_h5py(
        fname, "0/imagetool", ds
    )
    loaded = manager_workspace._read_workspace_dataset_group_h5py(
        fname,
        "0/imagetool",
        preferred_data_name=data_name,
    )
    assert loaded is not None

    opened = manager_xarray.open_workspace_dataset(fname, "0/imagetool", chunks=None)
    try:
        opened_loaded = opened.load()
    finally:
        opened.close()
    xr.testing.assert_equal(loaded, opened_loaded)
    return loaded, opened_loaded, fname


def test_workspace_h5py_fast_path_roundtrips_scalar_coords(tmp_path) -> None:
    import h5py

    fname = tmp_path / "scalar-fast-path.itws"
    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("x", "y"),
        coords={"x": np.arange(2.0), "y": np.arange(3.0), "temperature": 20.0},
        attrs={"coordinates": b""},
        name=manager_mainwindow._ITOOL_DATA_NAME,
    )
    ds = data.to_dataset()

    assert manager_workspace._write_workspace_dataset_group_h5py(
        fname, "0/imagetool", ds
    )
    loaded = manager_workspace._read_workspace_dataset_group_h5py(
        fname,
        "0/imagetool",
        preferred_data_name=manager_mainwindow._ITOOL_DATA_NAME,
    )

    assert loaded is not None
    expected = data.copy()
    expected.attrs.pop("coordinates")
    xr.testing.assert_equal(
        loaded[manager_mainwindow._ITOOL_DATA_NAME],
        expected,
    )
    assert loaded.coords["temperature"].item() == 20.0
    with h5py.File(fname, "r") as h5_file:
        coordinates = h5_file["0/imagetool"][manager_mainwindow._ITOOL_DATA_NAME].attrs[
            "coordinates"
        ]
    if isinstance(coordinates, bytes):
        coordinates = coordinates.decode()
    assert coordinates == "temperature"


def test_workspace_writer_encodes_saved_tool_spaced_associated_coord(
    tmp_path,
) -> None:
    data_name = imagetool_serialization.SAVED_TOOL_DATA_NAME
    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("x", "y"),
        coords={
            "x": np.arange(2.0),
            "y": np.arange(3.0),
            "Fake Motor": ("x", np.linspace(10.0, 20.0, 2)),
        },
        name=data_name,
    )
    fname = tmp_path / "saved-tool-spaced-coord.itws"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        manager_workspace._write_workspace_dataset_group_to_file(
            fname, "0/tool", data.to_dataset()
        )

    assert not any("space in its name" in str(item.message) for item in caught)
    loaded = manager_workspace._read_workspace_dataset_group_h5py(
        fname, "0/tool", preferred_data_name=data_name
    )

    assert loaded is not None
    xr.testing.assert_equal(
        loaded[data_name].coords["Fake Motor"], data.coords["Fake Motor"]
    )


def test_workspace_h5py_fast_path_roundtrips_associated_coords_and_xarray(
    tmp_path,
) -> None:
    import h5py

    data_name = manager_mainwindow._ITOOL_DATA_NAME
    base = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("x", "y"),
        coords={"x": np.arange(2.0), "y": np.arange(3.0)},
    )

    divided = base.assign_coords(mesh_current=("x", [1.0, 2.0]))
    divided = divided / divided.coords["mesh_current"]
    loaded, _opened, _fname = _assert_workspace_h5py_roundtrip(
        tmp_path, "divide-by-coord", divided
    )
    assert loaded.coords["mesh_current"].dims == ("x",)
    np.testing.assert_allclose(loaded.coords["mesh_current"], [1.0, 2.0])

    loaded, _opened, _fname = _assert_workspace_h5py_roundtrip(
        tmp_path,
        "two-dimensional-associated-coord",
        base.assign_coords(
            detector_norm=(("x", "y"), np.arange(6.0).reshape(2, 3) + 1.0)
        ),
    )
    assert loaded.coords["detector_norm"].dims == ("x", "y")

    loaded, _opened, _fname = _assert_workspace_h5py_roundtrip(
        tmp_path,
        "unicode-scalar-coord",
        base.assign_coords(label="sample"),
    )
    assert loaded.coords["label"].item() == "sample"

    loaded, _opened, _fname = _assert_workspace_h5py_roundtrip(
        tmp_path,
        "unicode-associated-coord",
        base.assign_coords(label=("x", np.array(["left", "right"]))),
    )
    assert loaded.coords["label"].dtype.kind == "U"

    loaded, _opened, _fname = _assert_workspace_h5py_roundtrip(
        tmp_path,
        "bytes-associated-coord",
        base.assign_coords(raw=("x", np.array([b"a", b"bb"], dtype="S2"))),
    )
    assert loaded.coords["raw"].dtype.kind == "S"

    loaded, _opened, _fname = _assert_workspace_h5py_roundtrip(
        tmp_path,
        "datetime-associated-coord",
        xr.DataArray(
            np.arange(2.0),
            dims=("time",),
            coords={
                "time": np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[D]"),
                "event_time": (
                    "time",
                    np.array(["2024-02-01", "2024-02-02"], dtype="datetime64[D]"),
                ),
            },
        ),
    )
    assert loaded.coords["time"].dtype.kind == "M"
    assert loaded.coords["event_time"].dtype.kind == "M"

    loaded, _opened, _fname = _assert_workspace_h5py_roundtrip(
        tmp_path,
        "timedelta-associated-coord",
        xr.DataArray(
            np.arange(2.0),
            dims=("delay",),
            coords={
                "delay": np.array([0, 5], dtype="timedelta64[ms]"),
                "exposure": (
                    "delay",
                    np.array([1, 2], dtype="timedelta64[s]"),
                ),
            },
        ),
    )
    assert loaded.coords["delay"].dtype == np.dtype("timedelta64[ms]")
    assert loaded.coords["exposure"].dtype == np.dtype("timedelta64[s]")

    with h5py.File(_fname, "r") as h5_file:
        coordinates = h5_file["0/imagetool"][data_name].attrs["coordinates"]
    if isinstance(coordinates, bytes):
        coordinates = coordinates.decode()
    assert coordinates == "exposure"


def test_workspace_h5py_fast_path_keeps_numeric_since_units(tmp_path) -> None:
    data_name = manager_mainwindow._ITOOL_DATA_NAME
    fname = tmp_path / "numeric-since-units.itws"
    data = xr.DataArray(
        [1.0, 2.0],
        dims=("x",),
        coords={
            "x": [0.0, 1.0],
            "elapsed": xr.DataArray(
                [0.0, 1.0],
                dims=("x",),
                attrs={"units": "seconds since start"},
            ),
        },
        name=data_name,
    )
    ds = data.to_dataset()

    assert manager_workspace._workspace_dataset_can_write_h5py(ds)
    assert manager_workspace._write_workspace_dataset_group_h5py(
        fname, "0/imagetool", ds
    )
    loaded = manager_workspace._read_workspace_dataset_group_h5py(
        fname,
        "0/imagetool",
        preferred_data_name=data_name,
    )

    assert loaded is not None
    xr.testing.assert_equal(loaded[data_name], data)
    assert loaded.coords["elapsed"].attrs["units"] == "seconds since start"


def test_workspace_h5py_fast_path_rejects_invalid_payloads(
    monkeypatch, tmp_path
) -> None:
    data_name = manager_mainwindow._ITOOL_DATA_NAME
    private_attr = imagetool_serialization._PRIVATE_COORDS_ATTR

    assert not manager_workspace._workspace_dataset_can_write_h5py(
        xr.Dataset(
            {
                data_name: ("x", [1.0]),
                "extra": ("x", [2.0]),
            },
            coords={"x": [0.0]},
        )
    )

    missing_private = xr.Dataset({data_name: ("x", [1.0])}, coords={"x": [0.0]})
    missing_private[data_name].attrs[private_attr] = json.dumps(
        [{"coord_name": "Fake Motor", "variable_name": "missing", "dims": ["x"]}]
    )
    assert not manager_workspace._workspace_dataset_can_write_h5py(missing_private)

    bad_private_dims = xr.Dataset(
        {
            data_name: ("x", [1.0]),
            "private": ("z", [2.0]),
        },
        coords={"x": [0.0], "z": [0.0]},
    )
    bad_private_dims[data_name].attrs[private_attr] = json.dumps(
        [{"coord_name": "Fake Motor", "variable_name": "private", "dims": ["z"]}]
    )
    assert not manager_workspace._workspace_dataset_can_write_h5py(bad_private_dims)

    assert not manager_workspace._workspace_dataset_can_write_h5py(
        xr.Dataset(
            {data_name: ("x", [1.0])},
            coords={"x": np.array([object()], dtype=object)},
        )
    )

    bad_associated_dims = xr.Dataset(
        {data_name: ("x", [1.0])},
        coords={"x": [0.0], "z": [0.0], "bad": ("z", [1.0])},
    )
    assert not manager_workspace._workspace_dataset_can_write_h5py(bad_associated_dims)

    import dask.array as da

    chunked_coord = xr.Dataset(
        {data_name: ("x", [1.0, 2.0])},
        coords={
            "x": [0.0, 1.0],
            "chunked": ("x", da.from_array(np.array([1.0, 2.0]), chunks=(1,))),
        },
    )
    assert not manager_workspace._workspace_dataset_can_write_h5py(chunked_coord)

    monkeypatch.setattr(
        manager_workspace, "_workspace_dataset_can_write_h5py", lambda _ds: True
    )
    assert not manager_workspace._write_workspace_dataset_group_h5py(
        tmp_path / "no-data-name.itws", "0/imagetool", xr.Dataset()
    )

    bad_attrs = xr.Dataset({data_name: ("x", [1.0])}, coords={"x": [0.0]})
    bad_attrs.attrs["bad"] = object()
    fname = tmp_path / "bad-attrs.itws"
    assert not manager_workspace._write_workspace_dataset_group_h5py(
        fname, "0/imagetool", bad_attrs
    )
    import h5py

    with h5py.File(fname, "r") as h5_file:
        assert "0/imagetool" not in h5_file


def test_workspace_h5py_reader_rejects_malformed_groups(tmp_path) -> None:
    import h5py

    data_name = manager_mainwindow._ITOOL_DATA_NAME
    private_attr = imagetool_serialization._PRIVATE_COORDS_ATTR
    fname = tmp_path / "malformed-reader.itws"

    with h5py.File(fname, "w") as h5_file:
        h5_file.create_dataset("not-a-group", data=np.arange(2.0))
        multi = h5_file.create_group("multi")
        multi.create_dataset("a", data=np.arange(2.0))
        multi.create_dataset("b", data=np.arange(2.0))
        no_dims = h5_file.create_group("no-dims")
        no_dims.create_dataset(data_name, data=np.arange(2.0))
        bad_scale = h5_file.create_group("bad-scale")
        scale = bad_scale.create_dataset("x", data=np.arange(4.0).reshape(2, 2))
        scale.make_scale("x")
        bad_data = bad_scale.create_dataset(data_name, data=np.arange(2.0))
        bad_data.dims[0].attach_scale(scale)
        missing_scalar = h5_file.create_group("missing-scalar")
        x = missing_scalar.create_dataset("x", data=np.arange(2.0))
        x.make_scale("x")
        missing_data = missing_scalar.create_dataset(data_name, data=np.arange(2.0))
        missing_data.dims[0].attach_scale(x)
        missing_data.attrs["coordinates"] = np.bytes_("missing")
        missing_private = h5_file.create_group("missing-private")
        x = missing_private.create_dataset("x", data=np.arange(2.0))
        x.make_scale("x")
        private_data = missing_private.create_dataset(data_name, data=np.arange(2.0))
        private_data.dims[0].attach_scale(x)
        private_data.attrs[private_attr] = json.dumps(
            [{"coord_name": "Fake Motor", "variable_name": "missing", "dims": ["x"]}]
        )
        bad_private = h5_file.create_group("bad-private")
        x = bad_private.create_dataset("x", data=np.arange(2.0))
        x.make_scale("x")
        private_data = bad_private.create_dataset(data_name, data=np.arange(2.0))
        private_data.dims[0].attach_scale(x)
        bad_coord = bad_private.create_dataset("private", data=np.arange(2.0))
        bad_coord.dims[0].attach_scale(x)
        private_data.attrs[private_attr] = json.dumps(
            [{"coord_name": "Fake Motor", "variable_name": "private", "dims": ["z"]}]
        )
        bad_associated_no_scale = h5_file.create_group("bad-associated-no-scale")
        x = bad_associated_no_scale.create_dataset("x", data=np.arange(2.0))
        x.make_scale("x")
        data = bad_associated_no_scale.create_dataset(data_name, data=np.arange(2.0))
        data.dims[0].attach_scale(x)
        data.attrs["coordinates"] = "associated"
        bad_associated_no_scale.create_dataset("associated", data=np.arange(2.0))
        bad_associated_length = h5_file.create_group("bad-associated-length")
        x = bad_associated_length.create_dataset("x", data=np.arange(2.0))
        x.make_scale("x")
        data = bad_associated_length.create_dataset(data_name, data=np.arange(2.0))
        data.dims[0].attach_scale(x)
        data.attrs["coordinates"] = "associated"
        associated = bad_associated_length.create_dataset(
            "associated", data=np.arange(3.0)
        )
        associated.dims[0].attach_scale(x)
        bad_associated_foreign_dim = h5_file.create_group("bad-associated-foreign-dim")
        x = bad_associated_foreign_dim.create_dataset("x", data=np.arange(2.0))
        x.make_scale("x")
        z = bad_associated_foreign_dim.create_dataset("z", data=np.arange(2.0))
        z.make_scale("z")
        data = bad_associated_foreign_dim.create_dataset(data_name, data=np.arange(2.0))
        data.dims[0].attach_scale(x)
        data.attrs["coordinates"] = "associated"
        associated = bad_associated_foreign_dim.create_dataset(
            "associated", data=np.arange(2.0)
        )
        associated.dims[0].attach_scale(z)
        bad_time = h5_file.create_group("bad-time-metadata")
        x = bad_time.create_dataset("x", data=np.arange(2.0))
        x.make_scale("x")
        data = bad_time.create_dataset(data_name, data=np.arange(2.0))
        data.dims[0].attach_scale(x)
        data.attrs["coordinates"] = "time"
        time = bad_time.create_dataset("time", data=np.arange(2, dtype=np.int64))
        time.dims[0].attach_scale(x)
        time.attrs["units"] = "days since not-a-date"
        time.attrs["calendar"] = "proleptic_gregorian"

    assert (
        manager_workspace._read_workspace_dataset_group_h5py(fname, "missing") is None
    )
    assert (
        manager_workspace._read_workspace_dataset_group_h5py(fname, "not-a-group")
        is None
    )
    assert manager_workspace._read_workspace_dataset_group_h5py(fname, "multi") is None
    assert (
        manager_workspace._read_workspace_dataset_group_h5py(fname, "no-dims") is None
    )
    assert (
        manager_workspace._read_workspace_dataset_group_h5py(fname, "bad-scale") is None
    )
    assert (
        manager_workspace._read_workspace_dataset_group_h5py(fname, "missing-scalar")
        is None
    )
    assert (
        manager_workspace._read_workspace_dataset_group_h5py(
            fname, "missing-private", preferred_data_name=data_name
        )
        is None
    )
    assert (
        manager_workspace._read_workspace_dataset_group_h5py(
            fname, "bad-private", preferred_data_name=data_name
        )
        is None
    )
    assert (
        manager_workspace._read_workspace_dataset_group_h5py(
            fname, "bad-associated-no-scale", preferred_data_name=data_name
        )
        is None
    )
    assert (
        manager_workspace._read_workspace_dataset_group_h5py(
            fname, "bad-associated-length", preferred_data_name=data_name
        )
        is None
    )
    assert (
        manager_workspace._read_workspace_dataset_group_h5py(
            fname, "bad-associated-foreign-dim", preferred_data_name=data_name
        )
        is None
    )
    assert (
        manager_workspace._read_workspace_dataset_group_h5py(
            fname, "bad-time-metadata", preferred_data_name=data_name
        )
        is None
    )


def test_workspace_h5py_reader_restores_legacy_spaced_coords(tmp_path) -> None:
    import h5py

    data_name = manager_mainwindow._ITOOL_DATA_NAME
    private_attr = imagetool_serialization._PRIVATE_COORDS_ATTR
    fname = tmp_path / "legacy-spaced-coord.itws"

    with h5py.File(fname, "w") as h5_file:
        group = h5_file.create_group("valid")
        x = group.create_dataset("x", data=np.arange(2.0))
        x.make_scale("x")
        data = group.create_dataset(data_name, data=np.arange(2.0))
        data.dims[0].attach_scale(x)
        data.attrs["coordinates"] = "missing"
        fake = group.create_dataset("Fake Motor", data=np.arange(2.0) + 10.0)
        fake.dims[0].attach_scale(x)
        duplicate = h5_file.create_group("duplicate")
        x = duplicate.create_dataset("x", data=np.arange(2.0))
        x.make_scale("x")
        data = duplicate.create_dataset(data_name, data=np.arange(2.0))
        data.dims[0].attach_scale(x)
        data.attrs[private_attr] = json.dumps(
            [
                {
                    "coord_name": "Fake Motor",
                    "variable_name": "Fake Motor",
                    "dims": ["x"],
                }
            ]
        )
        fake = duplicate.create_dataset("Fake Motor", data=np.arange(2.0) + 20.0)
        fake.dims[0].attach_scale(x)
        invalid = h5_file.create_group("invalid")
        x = invalid.create_dataset("x", data=np.arange(2.0))
        x.make_scale("x")
        data = invalid.create_dataset(data_name, data=np.arange(2.0))
        data.dims[0].attach_scale(x)
        invalid.create_dataset("Fake Motor", data=np.arange(2.0) + 30.0)

    loaded = manager_workspace._read_workspace_dataset_group_h5py(
        fname, "valid", preferred_data_name=data_name
    )
    assert loaded is not None
    np.testing.assert_allclose(loaded.coords["Fake Motor"].values, [10.0, 11.0])
    duplicate_loaded = manager_workspace._read_workspace_dataset_group_h5py(
        fname, "duplicate", preferred_data_name=data_name
    )
    assert duplicate_loaded is not None
    np.testing.assert_allclose(
        duplicate_loaded.coords["Fake Motor"].values, [20.0, 21.0]
    )
    invalid_loaded = manager_workspace._read_workspace_dataset_group_h5py(
        fname, "invalid", preferred_data_name=data_name
    )
    assert invalid_loaded is not None
    assert "Fake Motor" not in invalid_loaded


def test_workspace_h5py_writer_replaces_groups_and_preserves_attrs(tmp_path) -> None:
    import h5py

    data_name = manager_mainwindow._ITOOL_DATA_NAME
    private_attr = imagetool_serialization._PRIVATE_COORDS_ATTR
    fname = tmp_path / "writer-attrs.itws"
    ds = xr.Dataset(
        {
            data_name: (
                ("x", "y"),
                np.arange(4.0).reshape(2, 2),
                {"coordinates": "legacy"},
            ),
            "private": (("x",), np.arange(2.0), {"private_attr": "kept"}),
        },
        coords={
            "x": ("x", np.arange(2.0), {"axis_attr": "x"}),
            "y": ("y", np.arange(2.0), {"axis_attr": "y"}),
            "temperature": ((), 20.0, {"units": "K"}),
        },
    )
    ds[data_name].attrs[private_attr] = json.dumps(
        [{"coord_name": "Fake Motor", "variable_name": "private", "dims": ["x"]}]
    )

    assert manager_workspace._write_workspace_dataset_group_h5py(
        fname, "0/imagetool", ds
    )
    assert manager_workspace._write_workspace_dataset_group_h5py(
        fname, "0/imagetool", ds
    )

    with h5py.File(fname, "r") as h5_file:
        group = h5_file["0/imagetool"]
        assert group["x"].attrs["axis_attr"] == "x"
        assert group["temperature"].attrs["units"] == "K"
        assert group["private"].attrs["private_attr"] == "kept"
        coordinates = group[data_name].attrs["coordinates"]
        if isinstance(coordinates, bytes):
            coordinates = coordinates.decode()
        assert coordinates == "legacy temperature"


def test_manager_load_workspace_dataset_ignores_invalid_saved_metadata(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": np.arange(2.0), "y": np.arange(2.0)},
    )
    saved = itool(data, manager=False, execute=False)
    qtbot.addWidget(saved)
    assert isinstance(saved, erlab.interactive.imagetool.ImageTool)
    ds = saved.to_dataset()
    ds.attrs["manager_node_uid"] = "loaded"
    ds.attrs["manager_node_provenance_spec"] = "{not-json"
    ds.attrs["manager_node_live_source_spec"] = "{not-json"
    ds.attrs["manager_node_live_source_binding"] = "{not-json"

    with manager_context() as manager:
        target = manager._load_workspace_imagetool_dataset(
            ds, parent_target=None, node_path="-1"
        )

        assert target in manager._imagetool_wrappers
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)


def test_manager_load_workspace_tool_dataset_rejects_root_tool(
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with (
        manager_context() as manager,
        pytest.raises(ValueError, match="Workspace tool node has no parent"),
    ):
        manager._load_workspace_tool_dataset(xr.Dataset(), parent_target=None)


def test_manager_from_h5py_workspace_manifest_validation(
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    fname = tmp_path / "missing.itws"

    with manager_context() as manager:
        with pytest.raises(TypeError, match="missing node ordering"):
            manager._from_h5py_workspace_file(
                fname, {}, replace=False, mark_dirty=False
            )
        with pytest.raises(ValueError, match="no loadable nodes"):
            manager._from_h5py_workspace_file(
                fname,
                {
                    "nodes": [
                        [],
                        {"path": 0, "kind": "imagetool"},
                        {"path": "0", "kind": "unknown"},
                    ],
                    "root_order": [],
                },
                replace=False,
                mark_dirty=False,
            )
        with pytest.raises(ValueError, match="no root ImageTool nodes"):
            manager._from_h5py_workspace_file(
                fname,
                {
                    "nodes": [{"path": "0/childtools/tool", "kind": "tool"}],
                    "root_order": [],
                },
                replace=False,
                mark_dirty=False,
            )


def test_manager_from_h5py_workspace_falls_back_after_fast_read_error(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25.0).reshape(5, 5), dims=("x", "y"))
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        fname = tmp_path / "fallback-load.itws"
        manager._save_workspace_document(fname, force_full=True)
        with h5py.File(fname, "r") as h5_file:
            manifest = manager_workspace._workspace_manifest_from_attrs(h5_file.attrs)

        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        def _raise_fast_read(*_args, **_kwargs):
            raise RuntimeError("fast path failed")

        monkeypatch.setattr(
            manager_workspace,
            "_read_workspace_dataset_group_h5py",
            _raise_fast_read,
        )

        assert manager._from_h5py_workspace_file(
            fname, manifest, replace=True, mark_dirty=False
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)


def test_manager_from_h5py_workspace_logs_restore_failure(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25.0).reshape(5, 5), dims=("x", "y"))
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        fname = tmp_path / "restore-failure.itws"
        manager._save_workspace_document(fname, force_full=True)
        with h5py.File(fname, "r") as h5_file:
            manifest = manager_workspace._workspace_manifest_from_attrs(h5_file.attrs)

        def _raise_load(*_args, **_kwargs):
            raise RuntimeError("load failed")

        def _raise_restore(*_args, **_kwargs):
            raise RuntimeError("restore failed")

        monkeypatch.setattr(manager, "_load_workspace_imagetool_dataset", _raise_load)
        monkeypatch.setattr(manager, "_restore_replaced_workspace", _raise_restore)

        with pytest.raises(RuntimeError, match="load failed"):
            manager._from_h5py_workspace_file(
                fname, manifest, replace=True, mark_dirty=False
            )


def test_manager_workspace_rebind_skips_missing_snapshot_and_keeps_chunks(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(16.0).reshape(4, 4), dims=("x", "y")).chunk(
            {"x": 2}
        )
        root = itool(data, manager=False, execute=False, auto_compute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._imagetool_wrappers[0].uid
        calls: list[typing.Any] = []

        def _fake_rebind_data(_fname, _uid, *, chunks):
            calls.append(chunks)
            return data

        monkeypatch.setattr(
            manager, "_workspace_rebind_data_for_uid", _fake_rebind_data
        )

        manager._rebind_workspace_backed_imagetools(
            tmp_path / "workspace.itws", backing_snapshot={}
        )
        assert calls == []

        manager._rebind_workspace_backed_imagetools(tmp_path / "workspace.itws")

        assert uid in manager._all_nodes
        assert calls == [{}]


def test_manager_workspace_full_save_copy_group_edge_cases(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    workspace_path = tmp_path / "copy-groups.itws"
    workspace_path.touch()

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager._workspace_path = workspace_path
        monkeypatch.setattr(
            manager_workspace,
            "_read_workspace_root_attrs_h5py",
            lambda _path: (_ for _ in ()).throw(RuntimeError("metadata failed")),
        )
        assert manager._workspace_full_save_copy_groups(xr.DataTree()) == (None, ())

        monkeypatch.setattr(
            manager_workspace,
            "_read_workspace_root_attrs_h5py",
            lambda _path: {"imagetool_workspace_schema_version": 1},
        )
        assert manager._workspace_full_save_copy_groups(xr.DataTree()) == (None, ())

        data = xr.DataArray(np.arange(25.0).reshape(5, 5), dims=("x", "y"))
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._imagetool_wrappers[0].uid
        tool = DerivativeTool(data)
        monkeypatch.setattr(tool, "can_save_and_load", lambda: False)
        manager.add_childtool(tool, 0, show=False)
        manager._mark_workspace_clean()
        tree = manager._to_datatree()
        try:
            manifest_without_identities = {
                "schema_version": manager_workspace._current_workspace_schema_version(),
                "nodes": [[]],
                "root_order": [0],
            }
            monkeypatch.setattr(
                manager_workspace,
                "_read_workspace_root_attrs_h5py",
                lambda _path: {
                    "imagetool_workspace_schema_version": (
                        manager_workspace._current_workspace_schema_version()
                    ),
                    manager_workspace._WORKSPACE_MANIFEST_ATTR: json.dumps(
                        manifest_without_identities
                    ),
                },
            )
            assert manager._workspace_full_save_copy_groups(tree) == (
                str(workspace_path),
                (),
            )

            manifest_with_missing_tree_payload = {
                "schema_version": manager_workspace._current_workspace_schema_version(),
                "nodes": [
                    [],
                    {"uid": uid, "kind": "imagetool", "path": "0"},
                ],
                "root_order": [0],
            }
            monkeypatch.setattr(
                manager_workspace,
                "_read_workspace_root_attrs_h5py",
                lambda _path: {
                    "imagetool_workspace_schema_version": (
                        manager_workspace._current_workspace_schema_version()
                    ),
                    manager_workspace._WORKSPACE_MANIFEST_ATTR: json.dumps(
                        manifest_with_missing_tree_payload
                    ),
                },
            )
            assert manager._workspace_full_save_copy_groups(xr.DataTree()) == (
                str(workspace_path),
                (),
            )
        finally:
            tree.close()


def test_prepare_workspace_transaction_promotes_missing_attr_fallback(
    tmp_path,
) -> None:
    fname = tmp_path / "fallback.itws"
    _write_transaction_test_workspace(fname)
    fallback = (
        "0",
        {"0/imagetool": _transaction_test_dataset(2.0, title="fallback")},
    )
    rewrite_map: dict[str, tuple[str, dict[str, xr.Dataset]]] = {}

    group_operations, attr_updates = manager_workspace._prepare_workspace_transaction(
        fname,
        f"{manager_workspace._WORKSPACE_TRANSACTION_GROUP_PREFIX}fallback",
        f"{manager_workspace._WORKSPACE_PENDING_GROUP_PREFIX}fallback",
        f"{manager_workspace._WORKSPACE_BACKUP_GROUP_PREFIX}fallback",
        rewrite_map,
        (("0/missing", {"itool_title": "new"}, fallback),),
        _transaction_test_root_attrs(delta_save_count=1),
    )

    assert rewrite_map == {"0": fallback}
    assert attr_updates == []
    assert group_operations[0]["group_path"] == "0"
    manager_workspace._recover_workspace_transactions(fname)
    _assert_no_workspace_internal_groups(fname)


def test_write_full_workspace_tree_file_skips_missing_copy_source_group(
    tmp_path,
) -> None:
    fname = tmp_path / "missing-copy-group.itws"
    _write_transaction_test_workspace(fname)
    tree = xr.DataTree.from_dict(
        {"0/imagetool": _transaction_test_dataset(3.0, title="rewritten")}
    )
    try:
        manager_workspace._write_full_workspace_tree_file(
            fname,
            tree,
            _transaction_test_root_attrs(),
            copy_source=fname,
            copy_groups=(("missing/source", "0/imagetool", None),),
        )
    finally:
        tree.close()

    assert _read_transaction_test_value(fname) == 3.0


def test_write_full_workspace_tree_file_replaces_stale_root_attrs(tmp_path) -> None:
    import h5py

    fname = tmp_path / "root-attrs.itws"
    tree = xr.DataTree.from_dict(
        {"0/imagetool": _transaction_test_dataset(1.0, title="old")}
    )
    tree.attrs["stale_workspace_attr"] = "remove me"
    try:
        manager_workspace._write_full_workspace_tree_file(
            fname, tree, _transaction_test_root_attrs()
        )
    finally:
        tree.close()

    with h5py.File(fname, "r") as h5_file:
        assert "stale_workspace_attr" not in h5_file.attrs
        manifest = manager_workspace._workspace_manifest_from_attrs(h5_file.attrs)
        assert manifest == {"schema_version": 4, "root_order": [0], "nodes": []}


def test_write_full_workspace_tree_file_local_path_uses_destination_temp(
    monkeypatch, tmp_path
) -> None:
    fname = tmp_path / "local.itws"
    tree = xr.DataTree.from_dict(
        {"0/imagetool": _transaction_test_dataset(1.0, title="local")}
    )
    write_targets: list[pathlib.Path] = []
    original_write = manager_workspace._write_workspace_dataset_group_to_file

    def _record_write(target, *args, **kwargs):
        write_targets.append(pathlib.Path(target))
        return original_write(target, *args, **kwargs)

    monkeypatch.setattr(
        manager_workspace, "_write_workspace_dataset_group_to_file", _record_write
    )
    monkeypatch.setattr(
        manager_workspace, "_workspace_path_is_likely_network_path", lambda _path: False
    )
    monkeypatch.setattr(
        manager_workspace, "_workspace_path_is_likely_cloud_path", lambda _path: False
    )
    try:
        manager_workspace._write_full_workspace_tree_file(
            fname, tree, _transaction_test_root_attrs()
        )
    finally:
        tree.close()

    assert write_targets
    assert all(target.parent == fname.parent for target in write_targets)
    assert all(target.name.startswith(f"{fname.name}.tmp-") for target in write_targets)


def test_write_full_workspace_tree_file_cloud_path_uses_scratch_and_replace_first(
    monkeypatch, tmp_path
) -> None:
    fname = tmp_path / "Dropbox" / "cloud.itws"
    fname.parent.mkdir()
    tree = xr.DataTree.from_dict(
        {"0/imagetool": _transaction_test_dataset(2.0, title="cloud")}
    )
    write_targets: list[pathlib.Path] = []
    replace_calls: list[tuple[pathlib.Path, pathlib.Path]] = []
    original_write = manager_workspace._write_workspace_dataset_group_to_file

    def _record_write(target, *args, **kwargs):
        write_targets.append(pathlib.Path(target))
        return original_write(target, *args, **kwargs)

    def _replace_by_copy(src, dst):
        src_path = pathlib.Path(src)
        dst_path = pathlib.Path(dst)
        replace_calls.append((src_path, dst_path))
        dst_path.write_bytes(src_path.read_bytes())
        src_path.unlink()

    monkeypatch.setattr(
        manager_workspace, "_write_workspace_dataset_group_to_file", _record_write
    )
    monkeypatch.setattr(
        manager_workspace, "_workspace_path_is_likely_network_path", lambda _path: False
    )
    monkeypatch.setattr(
        manager_workspace, "_workspace_path_is_likely_cloud_path", lambda _path: True
    )
    monkeypatch.setattr(manager_workspace.os, "replace", _replace_by_copy)
    try:
        manager_workspace._write_full_workspace_tree_file(
            fname, tree, _transaction_test_root_attrs()
        )
    finally:
        tree.close()

    assert write_targets
    assert all(target.parent != fname.parent for target in write_targets)
    assert replace_calls == [(write_targets[0], fname)]
    assert _read_transaction_test_value(fname) == 2.0


def test_write_full_workspace_tree_file_copies_unchanged_payload_groups(
    monkeypatch,
    tmp_path,
) -> None:
    import h5py

    fname = tmp_path / "copy.itws"
    ds = xr.Dataset(
        {
            manager_mainwindow._ITOOL_DATA_NAME: (
                ("x", "y"),
                np.arange(12, dtype=np.float64).reshape(3, 4),
            )
        },
        coords={
            "x": np.arange(3, dtype=np.float64),
            "y": np.arange(4, dtype=np.float64),
        },
        attrs={
            "itool_title": "old",
            "manager_node_uid": "n0",
            "manager_node_kind": "imagetool",
        },
    )
    tree = xr.DataTree.from_dict({"0/imagetool": ds})
    try:
        manager_workspace._write_full_workspace_tree_file(
            fname, tree, _transaction_test_root_attrs()
        )
    finally:
        tree.close()

    rewritten = ds.assign_attrs(
        {
            "itool_title": "new",
            "manager_node_uid": "n0",
            "manager_node_kind": "imagetool",
        }
    )
    tree = xr.DataTree.from_dict({"0/imagetool": rewritten})

    def _fail_to_netcdf(*_args, **_kwargs):
        raise AssertionError("unchanged payload should be copied with h5py")

    monkeypatch.setattr(xr.Dataset, "to_netcdf", _fail_to_netcdf)
    try:
        manager_workspace._write_full_workspace_tree_file(
            fname,
            tree,
            _transaction_test_root_attrs(),
            copy_source=fname,
            copy_groups=(("0/imagetool", "0/imagetool", dict(rewritten.attrs)),),
        )
    finally:
        tree.close()

    with h5py.File(fname, "r") as h5_file:
        group = h5_file["0/imagetool"]
        assert group.attrs["itool_title"] == "new"
        np.testing.assert_array_equal(
            group[manager_mainwindow._ITOOL_DATA_NAME][...],
            np.arange(12, dtype=np.float64).reshape(3, 4),
        )


def test_write_full_workspace_tree_file_network_scratch_skips_copy_reuse(
    monkeypatch, tmp_path
) -> None:
    import shutil

    fname = tmp_path / "network-copy-reuse.itws"
    _write_transaction_test_workspace(fname)
    tree = xr.DataTree.from_dict(
        {"0/imagetool": _transaction_test_dataset(3.0, title="rewritten")}
    )

    def _fail_copyfile(*_args, **_kwargs):
        raise AssertionError("network scratch save should not copy old workspace")

    def _replace_by_copy(src, dst):
        src_path = pathlib.Path(src)
        dst_path = pathlib.Path(dst)
        dst_path.write_bytes(src_path.read_bytes())
        src_path.unlink()

    monkeypatch.setattr(
        manager_workspace, "_workspace_path_is_likely_network_path", lambda _path: True
    )
    monkeypatch.setattr(
        manager_workspace, "_workspace_path_is_likely_cloud_path", lambda _path: False
    )
    monkeypatch.setattr(shutil, "copyfile", _fail_copyfile)
    monkeypatch.setattr(manager_workspace.os, "replace", _replace_by_copy)
    try:
        manager_workspace._write_full_workspace_tree_file(
            fname,
            tree,
            _transaction_test_root_attrs(),
            copy_source=fname,
            copy_groups=(("0/imagetool", "0/imagetool", None),),
        )
    finally:
        tree.close()

    assert _read_transaction_test_value(fname) == 3.0


def test_write_full_workspace_tree_file_scratch_exdev_fallback(
    monkeypatch, tmp_path
) -> None:
    fname = tmp_path / "fallback.itws"
    _write_transaction_test_workspace(fname)
    tree = xr.DataTree.from_dict(
        {"0/imagetool": _transaction_test_dataset(4.0, title="fallback")}
    )
    original_replace = manager_workspace.os.replace
    replace_calls: list[tuple[pathlib.Path, pathlib.Path]] = []
    scratch_path: pathlib.Path | None = None

    def _replace_with_exdev(src, dst):
        nonlocal scratch_path
        src_path = pathlib.Path(src)
        dst_path = pathlib.Path(dst)
        replace_calls.append((src_path, dst_path))
        if dst_path == fname and src_path.parent != fname.parent:
            scratch_path = src_path
            raise OSError(errno.EXDEV, "cross-device link")
        return original_replace(src, dst)

    monkeypatch.setattr(
        manager_workspace, "_workspace_path_is_likely_network_path", lambda _path: False
    )
    monkeypatch.setattr(
        manager_workspace, "_workspace_path_is_likely_cloud_path", lambda _path: True
    )
    monkeypatch.setattr(manager_workspace.os, "replace", _replace_with_exdev)
    try:
        manager_workspace._write_full_workspace_tree_file(
            fname, tree, _transaction_test_root_attrs()
        )
    finally:
        tree.close()

    assert _read_transaction_test_value(fname) == 4.0
    assert scratch_path is not None
    assert not scratch_path.exists()
    assert len(replace_calls) == 2
    assert replace_calls[0] == (scratch_path, fname)
    assert replace_calls[1][0].parent == fname.parent
    assert replace_calls[1][1] == fname
    assert not list(fname.parent.glob(f"{fname.name}.tmp-*"))


def test_write_full_workspace_tree_file_scratch_replace_failure_preserves_old(
    monkeypatch, tmp_path
) -> None:
    fname = tmp_path / "replace-failure.itws"
    _write_transaction_test_workspace(fname)
    tree = xr.DataTree.from_dict(
        {"0/imagetool": _transaction_test_dataset(5.0, title="failure")}
    )
    scratch_paths: list[pathlib.Path] = []

    def _fail_replace(src, dst):
        src_path = pathlib.Path(src)
        if pathlib.Path(dst) == fname:
            scratch_paths.append(src_path)
        raise OSError(errno.EPERM, "replace failed")

    monkeypatch.setattr(
        manager_workspace, "_workspace_path_is_likely_network_path", lambda _path: False
    )
    monkeypatch.setattr(
        manager_workspace, "_workspace_path_is_likely_cloud_path", lambda _path: True
    )
    monkeypatch.setattr(manager_workspace.os, "replace", _fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        try:
            manager_workspace._write_full_workspace_tree_file(
                fname, tree, _transaction_test_root_attrs()
            )
        finally:
            tree.close()

    assert _read_transaction_test_value(fname) == 1.0
    assert scratch_paths
    assert all(not scratch_path.exists() for scratch_path in scratch_paths)
    assert not list(fname.parent.glob(f"{fname.name}.tmp-*"))


def test_write_full_workspace_tree_file_scratch_copy_failure_cleans_destination_tmp(
    monkeypatch, tmp_path
) -> None:
    import shutil

    fname = tmp_path / "copy-failure.itws"
    _write_transaction_test_workspace(fname)
    tree = xr.DataTree.from_dict(
        {"0/imagetool": _transaction_test_dataset(6.0, title="failure")}
    )
    original_replace = manager_workspace.os.replace
    scratch_paths: list[pathlib.Path] = []

    def _replace_with_exdev(src, dst):
        src_path = pathlib.Path(src)
        dst_path = pathlib.Path(dst)
        if dst_path == fname and src_path.parent != fname.parent:
            scratch_paths.append(src_path)
            raise OSError(errno.EXDEV, "cross-device link")
        return original_replace(src, dst)

    def _fail_copyfile(src, dst):
        pathlib.Path(dst).write_bytes(b"partial")
        raise OSError(errno.EIO, "copy failed")

    monkeypatch.setattr(
        manager_workspace, "_workspace_path_is_likely_network_path", lambda _path: False
    )
    monkeypatch.setattr(
        manager_workspace, "_workspace_path_is_likely_cloud_path", lambda _path: True
    )
    monkeypatch.setattr(manager_workspace.os, "replace", _replace_with_exdev)
    monkeypatch.setattr(shutil, "copyfile", _fail_copyfile)
    with pytest.raises(OSError, match="copy failed"):
        try:
            manager_workspace._write_full_workspace_tree_file(
                fname, tree, _transaction_test_root_attrs()
            )
        finally:
            tree.close()

    assert _read_transaction_test_value(fname) == 1.0
    assert scratch_paths
    assert all(not scratch_path.exists() for scratch_path in scratch_paths)
    assert not list(fname.parent.glob(f"{fname.name}.tmp-*"))


def test_workspace_recovery_discards_pending_only_transaction(tmp_path) -> None:
    fname = tmp_path / "pending-only.itws"
    _write_transaction_test_workspace(fname)
    rewrite = ("0", {"0/imagetool": _transaction_test_dataset(2.0, title="new")})
    rewrite_map = {"0": rewrite}
    txn_id = "pendingonly"
    txn_path = f"{manager_workspace._WORKSPACE_TRANSACTION_GROUP_PREFIX}{txn_id}"
    pending_root = f"{manager_workspace._WORKSPACE_PENDING_GROUP_PREFIX}{txn_id}"
    backup_root = f"{manager_workspace._WORKSPACE_BACKUP_GROUP_PREFIX}{txn_id}"

    manager_workspace._prepare_workspace_transaction(
        fname,
        txn_path,
        pending_root,
        backup_root,
        rewrite_map,
        (),
        _transaction_test_root_attrs(delta_save_count=1),
    )
    manager_workspace._write_workspace_transaction_pending_groups(
        fname, rewrite_map, pending_root
    )

    manager_workspace._recover_workspace_transactions(fname)

    assert _read_transaction_test_value(fname) == 1.0
    _assert_no_workspace_internal_groups(fname)


def test_workspace_recovery_restores_backup_before_pending_move(tmp_path) -> None:
    import h5py

    fname = tmp_path / "backup-before-pending.itws"
    _write_transaction_test_workspace(fname)
    rewrite = ("0", {"0/imagetool": _transaction_test_dataset(2.0, title="new")})
    rewrite_map = {"0": rewrite}
    txn_id = "backuponly"
    txn_path = f"{manager_workspace._WORKSPACE_TRANSACTION_GROUP_PREFIX}{txn_id}"
    pending_root = f"{manager_workspace._WORKSPACE_PENDING_GROUP_PREFIX}{txn_id}"
    backup_root = f"{manager_workspace._WORKSPACE_BACKUP_GROUP_PREFIX}{txn_id}"
    group_operations, _ = manager_workspace._prepare_workspace_transaction(
        fname,
        txn_path,
        pending_root,
        backup_root,
        rewrite_map,
        (),
        _transaction_test_root_attrs(delta_save_count=1),
    )
    manager_workspace._write_workspace_transaction_pending_groups(
        fname, rewrite_map, pending_root
    )

    with h5py.File(fname, "a") as h5_file:
        manager_workspace._set_workspace_transaction_status(
            h5_file,
            txn_path,
            "committing",
        )
        operation = group_operations[0]
        manager_workspace._move_h5_path(
            h5_file,
            typing.cast("str", operation["group_path"]),
            typing.cast("str", operation["backup_path"]),
        )

    manager_workspace._recover_workspace_transactions(fname)

    assert _read_transaction_test_value(fname) == 1.0
    _assert_no_workspace_internal_groups(fname)


def test_workspace_recovery_rolls_back_active_moved_before_commit(tmp_path) -> None:
    import h5py

    fname = tmp_path / "active-before-commit.itws"
    _write_transaction_test_workspace(fname)
    rewrite = ("0", {"0/imagetool": _transaction_test_dataset(2.0, title="new")})
    rewrite_map = {"0": rewrite}
    txn_id = "activemoved"
    txn_path = f"{manager_workspace._WORKSPACE_TRANSACTION_GROUP_PREFIX}{txn_id}"
    pending_root = f"{manager_workspace._WORKSPACE_PENDING_GROUP_PREFIX}{txn_id}"
    backup_root = f"{manager_workspace._WORKSPACE_BACKUP_GROUP_PREFIX}{txn_id}"
    group_operations, _ = manager_workspace._prepare_workspace_transaction(
        fname,
        txn_path,
        pending_root,
        backup_root,
        rewrite_map,
        (),
        _transaction_test_root_attrs(delta_save_count=1),
    )
    manager_workspace._write_workspace_transaction_pending_groups(
        fname, rewrite_map, pending_root
    )

    with h5py.File(fname, "a") as h5_file:
        manager_workspace._set_workspace_transaction_status(
            h5_file,
            txn_path,
            "committing",
        )
        operation = group_operations[0]
        manager_workspace._move_h5_path(
            h5_file,
            typing.cast("str", operation["group_path"]),
            typing.cast("str", operation["backup_path"]),
        )
        manager_workspace._move_h5_path(
            h5_file,
            typing.cast("str", operation["pending_path"]),
            typing.cast("str", operation["group_path"]),
        )

    manager_workspace._recover_workspace_transactions(fname)

    assert _read_transaction_test_value(fname) == 1.0
    _assert_no_workspace_internal_groups(fname)


def test_workspace_recovery_accepts_committed_before_cleanup(tmp_path) -> None:
    fname = tmp_path / "committed-before-cleanup.itws"
    _write_transaction_test_workspace(fname)
    rewrite = ("0", {"0/imagetool": _transaction_test_dataset(2.0, title="new")})
    rewrite_map = {"0": rewrite}
    txn_id = "committed"
    txn_path = f"{manager_workspace._WORKSPACE_TRANSACTION_GROUP_PREFIX}{txn_id}"
    pending_root = f"{manager_workspace._WORKSPACE_PENDING_GROUP_PREFIX}{txn_id}"
    backup_root = f"{manager_workspace._WORKSPACE_BACKUP_GROUP_PREFIX}{txn_id}"
    root_attrs = _transaction_test_root_attrs(delta_save_count=1)
    group_operations, attr_updates = manager_workspace._prepare_workspace_transaction(
        fname,
        txn_path,
        pending_root,
        backup_root,
        rewrite_map,
        (),
        root_attrs,
    )
    manager_workspace._write_workspace_transaction_pending_groups(
        fname, rewrite_map, pending_root
    )
    manager_workspace._commit_workspace_transaction(
        fname, txn_path, group_operations, attr_updates, root_attrs
    )

    manager_workspace._recover_workspace_transactions(fname)

    assert _read_transaction_test_value(fname) == 2.0
    _assert_no_workspace_internal_groups(fname)


def test_workspace_recovery_rolls_back_attr_only_transaction(tmp_path) -> None:
    import h5py

    fname = tmp_path / "attrs-before-commit.itws"
    _write_transaction_test_workspace(fname)
    fallback = (
        "0",
        {"0/imagetool": _transaction_test_dataset(2.0, title="fallback")},
    )
    attr_update = ("0/imagetool", {"itool_title": "new"}, fallback)
    txn_id = "attrrollback"
    txn_path = f"{manager_workspace._WORKSPACE_TRANSACTION_GROUP_PREFIX}{txn_id}"
    pending_root = f"{manager_workspace._WORKSPACE_PENDING_GROUP_PREFIX}{txn_id}"
    backup_root = f"{manager_workspace._WORKSPACE_BACKUP_GROUP_PREFIX}{txn_id}"
    root_attrs = _transaction_test_root_attrs(delta_save_count=1)
    _, attr_updates = manager_workspace._prepare_workspace_transaction(
        fname, txn_path, pending_root, backup_root, {}, (attr_update,), root_attrs
    )

    with h5py.File(fname, "a") as h5_file:
        manager_workspace._set_workspace_transaction_status(
            h5_file,
            txn_path,
            "committing",
        )
        manager_workspace._replace_h5_attrs(
            h5_file["0/imagetool"].attrs, attr_updates[0][1]
        )
        manager_workspace._write_root_attrs_to_open_workspace_file(h5_file, root_attrs)
        h5_file.flush()

    manager_workspace._recover_workspace_transactions(fname)

    with h5py.File(fname, "r") as h5_file:
        assert h5_file["0/imagetool"].attrs["itool_title"] == "old"
        assert (
            manager_workspace._workspace_delta_save_count_from_attrs(h5_file.attrs) == 0
        )
    _assert_no_workspace_internal_groups(fname)


def test_workspace_recovery_cleans_orphan_internal_groups(tmp_path) -> None:
    import h5py

    fname = tmp_path / "orphan-internal.itws"
    _write_transaction_test_workspace(fname)
    with h5py.File(fname, "a") as h5_file:
        h5_file.create_group(
            f"{manager_workspace._WORKSPACE_PENDING_GROUP_PREFIX}orphan"
        )
        h5_file.create_group(
            f"{manager_workspace._WORKSPACE_BACKUP_GROUP_PREFIX}orphan"
        )

    manager_workspace._recover_workspace_transactions(fname)

    _assert_no_workspace_internal_groups(fname)


def test_workspace_lock_path_uses_hidden_sidecar(tmp_path) -> None:
    fname = tmp_path / "example.itws"

    assert manager_workspace._workspace_lock_path(fname) == str(
        (tmp_path / ".example.itws.lock").resolve()
    )


def test_workspace_lock_conflict_is_reported(tmp_path) -> None:
    fname = tmp_path / "locked.itws"
    _write_transaction_test_workspace(fname)
    hidden_lock_path = pathlib.Path(manager_workspace._workspace_lock_path(fname))
    visible_lock_path = pathlib.Path(f"{fname.resolve()}.lock")
    lock = manager_workspace._acquire_workspace_document_lock(fname)
    try:
        assert lock.staleLockTime() == 0
        assert hidden_lock_path.exists()
        assert not visible_lock_path.exists()
        with pytest.raises(BlockingIOError):
            manager_workspace._acquire_workspace_document_lock(fname)
    finally:
        lock.unlock()


def test_hide_workspace_lock_file_sets_macos_hidden_flag(monkeypatch) -> None:
    calls: list[tuple[str, int]] = []
    lock_path = "/workspace/.workspace.itws.lock"
    regular_stat = types.SimpleNamespace(st_mode=0o100600)

    monkeypatch.setattr(manager_workspace.sys, "platform", "darwin")
    monkeypatch.setattr(manager_workspace.os, "lstat", lambda _path: regular_stat)
    monkeypatch.setattr(
        manager_workspace.os,
        "chflags",
        lambda path, flags: calls.append((path, flags)),
        raising=False,
    )

    manager_workspace._hide_workspace_lock_file(lock_path)

    assert calls == [(lock_path, 0x8000)]


def test_hide_workspace_lock_file_skips_macos_symlink(monkeypatch) -> None:
    calls: list[tuple[str, int]] = []
    symlink_stat = types.SimpleNamespace(st_mode=0o120777)

    monkeypatch.setattr(manager_workspace.sys, "platform", "darwin")
    monkeypatch.setattr(manager_workspace.os, "lstat", lambda _path: symlink_stat)
    monkeypatch.setattr(
        manager_workspace.os,
        "chflags",
        lambda path, flags: calls.append((path, flags)),
        raising=False,
    )

    manager_workspace._hide_workspace_lock_file("/workspace/.workspace.itws.lock")

    assert calls == []


def test_workspace_lock_error_message_names_owner(monkeypatch, tmp_path) -> None:
    fname = tmp_path / "busy-message.itws"
    _write_transaction_test_workspace(fname)
    lock = manager_workspace._acquire_workspace_document_lock(fname)
    lock_info = manager_workspace._workspace_document_lock_info(fname)
    calls: list[dict[str, object]] = []

    def _critical(*args, **kwargs) -> int:
        calls.append({"args": args, "kwargs": kwargs})
        return int(QtWidgets.QDialog.DialogCode.Accepted)

    monkeypatch.setattr(erlab.interactive.utils.MessageDialog, "critical", _critical)
    try:
        manager_mainwindow._show_workspace_file_lock_error(None, fname)
    finally:
        lock.unlock()

    assert len(calls) == 1
    args = calls[0]["args"]
    assert isinstance(args, tuple)
    assert args[1] == "Workspace Already Open"
    assert args[2] == "This workspace is already open somewhere else."
    informative_text = args[3]
    assert isinstance(informative_text, str)
    assert fname.name in informative_text
    assert "lock" not in informative_text.casefold()
    if lock_info.owner:
        assert lock_info.owner in informative_text
    if lock_info.hostname:
        assert lock_info.hostname in informative_text
    detailed_text = calls[0]["kwargs"]["detailed_text"]
    assert isinstance(detailed_text, str)
    assert "Temporary workspace ownership marker:" in detailed_text
    assert lock_info.path in detailed_text


def test_workspace_lock_text_variants(tmp_path) -> None:
    app_only = manager_workspace._WorkspaceDocumentLockInfo(
        path="marker",
        owner="user",
        hostname="",
        appname="ImageTool",
        pid=None,
    )
    pid_only = manager_workspace._WorkspaceDocumentLockInfo(
        path="marker",
        owner="",
        hostname="",
        appname="",
        pid=123,
    )
    full_info = manager_workspace._WorkspaceDocumentLockInfo(
        path="marker",
        owner="user",
        hostname="workstation",
        appname="ImageTool",
        pid=123,
    )

    assert manager_mainwindow._workspace_lock_owner_text(app_only) == (
        "user using ImageTool"
    )
    assert manager_mainwindow._workspace_lock_owner_text(pid_only) == (
        "using process 123"
    )
    assert manager_mainwindow._workspace_lock_owner_text(full_info) == (
        "user on workstation using ImageTool (process 123)"
    )

    def _raise_owner_details_failed() -> None:
        raise RuntimeError("owner details failed")

    def _details_from_active_exception() -> str:
        try:
            _raise_owner_details_failed()
        except RuntimeError:
            return manager_mainwindow._workspace_lock_details_text(
                tmp_path / "workspace.itws", full_info
            )

    details = _details_from_active_exception()

    assert "owner details failed" in details
    assert "Temporary workspace ownership marker: marker" in details


def test_workspace_window_title_placeholder_non_macos(monkeypatch) -> None:
    monkeypatch.setattr(manager_mainwindow.sys, "platform", "linux")

    assert manager_mainwindow._strip_workspace_modified_placeholder("Name[*]") == "Name"
    assert manager_mainwindow._window_title_with_modified_placeholder("Name[*]") == (
        "Name[*]"
    )


def test_manager_workspace_window_title_sets_file_path_normally(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        workspace = tmp_path / "normal.itws"
        manager._workspace_path = workspace
        manager._workspace_structure_modified = True
        file_path_calls: list[str] = []

        with monkeypatch.context() as patch:
            patch.setattr(
                ImageToolManager,
                "setWindowFilePath",
                lambda _manager, path: file_path_calls.append(path),
            )
            manager._update_workspace_window_title()

        assert file_path_calls == [str(workspace)]
        assert workspace.name in manager.windowTitle()
        assert manager.isWindowModified()


def test_manager_workspace_window_title_skips_file_path_during_macos_close(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        workspace = tmp_path / "close-save.itws"
        manager._workspace_path = workspace
        manager._workspace_structure_modified = True
        previous_closing = manager._closing_workspace_document
        manager._closing_workspace_document = True

        def _fail_if_called(_manager, _path: str) -> None:
            raise AssertionError("setWindowFilePath should be skipped")

        try:
            with monkeypatch.context() as patch:
                patch.setattr(manager_mainwindow.sys, "platform", "darwin")
                patch.setattr(ImageToolManager, "setWindowFilePath", _fail_if_called)
                manager._update_workspace_window_title()
        finally:
            manager._closing_workspace_document = previous_closing

        assert workspace.name in manager.windowTitle()
        assert manager.isWindowModified()


def test_manager_workspace_window_title_sets_file_path_for_non_macos_close(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        workspace = tmp_path / "linux-close.itws"
        manager._workspace_path = workspace
        previous_closing = manager._closing_workspace_document
        manager._closing_workspace_document = True
        file_path_calls: list[str] = []

        try:
            with monkeypatch.context() as patch:
                patch.setattr(manager_mainwindow.sys, "platform", "linux")
                patch.setattr(
                    ImageToolManager,
                    "setWindowFilePath",
                    lambda _manager, path: file_path_calls.append(path),
                )
                manager._update_workspace_window_title()
        finally:
            manager._closing_workspace_document = previous_closing

        assert file_path_calls == [str(workspace)]


def test_manager_close_cancel_restores_workspace_document_closing_state(
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager._workspace_structure_modified = True
        manager._closing_workspace_document = False
        event = QtGui.QCloseEvent()
        with monkeypatch.context() as patch:
            patch.setattr(
                manager, "_confirm_save_dirty_workspace", lambda _message: False
            )
            manager.closeEvent(event)

        assert not event.isAccepted()
        assert not manager._closing_workspace_document


def test_manager_close_save_path_skips_macos_file_path_update(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager._workspace_path = tmp_path / "close-save.itws"
        manager._workspace_structure_modified = True
        save_closing_states: list[bool] = []

        def _fail_during_close_file_path_update(
            window: ImageToolManager, _path: str
        ) -> None:
            if window._closing_workspace_document:
                raise AssertionError("setWindowFilePath should be skipped")

        def _save(*, native: bool = True) -> bool:
            save_closing_states.append(manager._closing_workspace_document)
            manager._mark_workspace_clean()
            return True

        with monkeypatch.context() as patch:
            patch.setattr(manager_mainwindow.sys, "platform", "darwin")
            patch.setattr(
                ImageToolManager,
                "setWindowFilePath",
                _fail_during_close_file_path_update,
            )
            patch.setattr(
                QtWidgets.QMessageBox,
                "exec",
                lambda _msg_box: QtWidgets.QMessageBox.StandardButton.Save,
            )
            patch.setattr(manager, "save", _save)
            assert manager.close()

        assert save_closing_states == [True]
        assert not manager._closing_workspace_document


def test_workspace_lock_error_message_without_owner(monkeypatch, tmp_path) -> None:
    fname = tmp_path / "busy-message.itws"
    calls: list[dict[str, object]] = []
    lock_info = manager_workspace._WorkspaceDocumentLockInfo(
        path=str(tmp_path / ".busy-message.itws.lock"),
        owner="",
        hostname="",
        appname="",
        pid=None,
    )

    def _critical(*args, **kwargs) -> int:
        calls.append({"args": args, "kwargs": kwargs})
        return int(QtWidgets.QDialog.DialogCode.Accepted)

    monkeypatch.setattr(
        manager_workspace, "_workspace_document_lock_info", lambda _fname: lock_info
    )
    monkeypatch.setattr(erlab.interactive.utils.MessageDialog, "critical", _critical)

    manager_mainwindow._show_workspace_file_lock_error(None, fname)

    args = calls[0]["args"]
    assert isinstance(args, tuple)
    informative_text = args[3]
    assert isinstance(informative_text, str)
    assert informative_text == (
        "Close the other ImageTool Manager that has busy-message.itws open, "
        "then try again."
    )


def test_application_quit_filter_routes_quit_events(qtbot) -> None:
    manager = QtWidgets.QWidget()
    qtbot.addWidget(manager)
    calls: list[str] = []
    manager.close = lambda: calls.append("close") or False
    event_filter = manager_mainwindow._ApplicationQuitFilter(
        typing.cast("ImageToolManager", manager)
    )

    assert not event_filter.eventFilter(None, None)
    assert event_filter.eventFilter(None, QtCore.QEvent(QtCore.QEvent.Type.Quit))

    class _QuitKeyEvent(QtGui.QKeyEvent):
        def matches(self, key: QtGui.QKeySequence.StandardKey) -> bool:
            return key == QtGui.QKeySequence.StandardKey.Quit

    shortcut_event = _QuitKeyEvent(
        QtCore.QEvent.Type.ShortcutOverride,
        QtCore.Qt.Key.Key_Q,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    assert not event_filter.eventFilter(None, shortcut_event)

    key_event = _QuitKeyEvent(
        QtCore.QEvent.Type.KeyPress,
        QtCore.Qt.Key.Key_Q,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )

    assert event_filter.eventFilter(None, key_event)
    assert key_event.isAccepted()
    assert calls == ["close", "close"]


def test_workspace_document_access_releases_lock(tmp_path) -> None:
    class _FakeLock:
        def __init__(self) -> None:
            self.unlock_count = 0

        def unlock(self) -> None:
            self.unlock_count += 1

    lock = _FakeLock()
    access = manager_mainwindow._WorkspaceDocumentAccess(
        tmp_path / "workspace.itws", lock
    )

    assert access.take_lock() is lock
    access.release()
    assert lock.unlock_count == 0

    access = manager_mainwindow._WorkspaceDocumentAccess(
        tmp_path / "workspace.itws", lock
    )
    access.release()
    access.release()
    assert lock.unlock_count == 1


def test_choose_from_datatree_dialog_root_keys_skip_missing(qtbot) -> None:
    manager = QtWidgets.QWidget()
    manager.next_idx = 7
    qtbot.addWidget(manager)
    tree = xr.DataTree.from_dict(
        {
            "0/imagetool": xr.Dataset(
                attrs={"itool_title": "Loaded"},
            )
        }
    )
    try:
        dialog = _ChooseFromDataTreeDialog(
            typing.cast("ImageToolManager", manager),
            tree,
            mode="load",
            root_keys=("missing", "0"),
        )
        qtbot.addWidget(dialog)

        assert dialog._tree_widget.topLevelItemCount() == 1
        assert dialog._tree_widget.topLevelItem(0).text(0) == "7: Loaded"
    finally:
        tree.close()


def test_manager_workspace_save_as_locked_target_does_not_write(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    fname = tmp_path / "locked-save-as.itws"
    _write_transaction_test_workspace(fname)
    lock = manager_workspace._acquire_workspace_document_lock(fname)
    operation_errors: list[tuple[typing.Any, ...]] = []
    try:
        with manager_context() as manager:
            monkeypatch.setattr(
                manager, "_workspace_save_dialog", lambda *args, **kwargs: str(fname)
            )
            monkeypatch.setattr(
                manager,
                "_save_workspace_document",
                lambda *args, **kwargs: pytest.fail(
                    "Save As should lock the target before writing"
                ),
            )
            monkeypatch.setattr(
                manager,
                "_show_operation_error",
                lambda *args, **kwargs: operation_errors.append(args),
            )

            assert not manager.save_as(native=False)
    finally:
        lock.unlock()

    assert operation_errors


def test_manager_workspace_load_locks_before_recovery(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    fname = tmp_path / "locked-load.itws"
    _write_transaction_test_workspace(fname)
    lock = manager_workspace._acquire_workspace_document_lock(fname)
    recovery_calls: list[pathlib.Path] = []
    try:
        monkeypatch.setattr(
            manager_workspace,
            "_recover_workspace_transactions",
            lambda path: (
                recovery_calls.append(pathlib.Path(path))
                or pytest.fail("Load should lock the workspace before recovery")
            ),
        )
        with manager_context() as manager, pytest.raises(BlockingIOError):
            manager._load_workspace_file(
                fname,
                replace=True,
                associate=True,
                mark_dirty=False,
                select=False,
            )
    finally:
        lock.unlock()

    assert recovery_calls == []


def test_manager_workspace_path_lock_contract(
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    class _FakeLock:
        def __init__(self) -> None:
            self.unlock_count = 0

        def unlock(self) -> None:
            self.unlock_count += 1

    with manager_context() as manager:
        current = (tmp_path / "current.itws").resolve()
        manager._workspace_path = current
        lock = _FakeLock()

        manager._set_workspace_path(current, workspace_lock=lock)

        assert lock.unlock_count == 1
        with pytest.raises(RuntimeError, match="pre-acquired document lock"):
            manager._set_workspace_path(tmp_path / "other.itws")


def test_manager_open_recent_menu_state_labels_and_clear(
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    workspace_a = tmp_path / "alpha" / "workspace.itws"
    workspace_b = tmp_path / "beta" / "workspace.itws"
    workspace_a.parent.mkdir()
    workspace_b.parent.mkdir()
    workspace_a.touch()
    workspace_b.touch()

    with manager_context() as manager:
        assert not manager.open_recent_menu.isEnabled()

        manager._record_recent_workspace(workspace_a)
        manager._record_recent_workspace(workspace_b)
        manager._populate_open_recent_menu()

        assert manager.open_recent_menu.isEnabled()
        actions = action_map_by_object_name(manager.open_recent_menu)
        first_action = actions["manager_recent_workspace_action_0"]
        second_action = actions["manager_recent_workspace_action_1"]
        assert first_action.data() == str(workspace_b.resolve())
        assert second_action.data() == str(workspace_a.resolve())
        assert first_action.toolTip() == str(workspace_b.resolve())
        assert first_action.statusTip() == str(workspace_b.resolve())
        assert "manager_clear_recent_workspaces_action" in actions

        actions["manager_clear_recent_workspaces_action"].trigger()
        assert manager._recent_workspace_paths() == []
        assert not manager.open_recent_menu.isEnabled()


def test_manager_recent_workspaces_dedupe_move_to_top_and_cap(
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    paths = [tmp_path / f"workspace-{idx}.itws" for idx in range(12)]
    for path in paths:
        path.touch()

    with manager_context() as manager:
        for path in paths:
            manager._record_recent_workspace(path)

        assert manager._recent_workspace_paths() == [
            path.resolve() for path in reversed(paths[2:])
        ]

        manager._record_recent_workspace(paths[6])

        assert manager._recent_workspace_paths() == [
            paths[6].resolve(),
            paths[11].resolve(),
            paths[10].resolve(),
            paths[9].resolve(),
            paths[8].resolve(),
            paths[7].resolve(),
            paths[5].resolve(),
            paths[4].resolve(),
            paths[3].resolve(),
            paths[2].resolve(),
        ]


def test_manager_recent_workspace_normalization_and_settings(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    workspace = tmp_path / "workspace.itws"
    data_file = tmp_path / "data.h5"
    workspace.touch()
    data_file.touch()

    assert ImageToolManager._normalize_recent_workspace_paths(
        [data_file, workspace, workspace]
    ) == [workspace.resolve()]

    with manager_context() as manager:
        settings = manager_mainwindow._manager_settings()
        settings.setValue(
            manager_mainwindow._RECENT_WORKSPACES_SETTINGS_KEY, str(workspace)
        )
        settings.sync()
        assert manager._recent_workspace_paths() == [workspace.resolve()]

        class _ObjectSettings:
            def sync(self) -> None:
                pass

            def value(self, _key, _default):
                return object()

        monkeypatch.setattr(
            manager_mainwindow, "_manager_settings", lambda: _ObjectSettings()
        )
        assert manager._recent_workspace_paths() == []


def test_manager_open_recent_workspace_flow(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    older = tmp_path / "older.itws"
    newer = tmp_path / "newer.itws"
    missing = tmp_path / "missing.itws"
    older.touch()
    newer.touch()

    with manager_context() as manager:
        manager._set_recent_workspace_paths([newer.resolve(), older.resolve()])
        load_calls: list[tuple[pathlib.Path, dict[str, typing.Any]]] = []

        def _record_load(path, **kwargs):
            load_calls.append((pathlib.Path(path), kwargs))
            return True

        monkeypatch.setattr(manager, "_load_workspace_file", _record_load)
        monkeypatch.setattr(
            manager, "_confirm_save_dirty_workspace", lambda _message: False
        )

        assert not manager.open_recent_workspace(older)
        assert load_calls == []
        assert manager._recent_workspace_paths() == [
            newer.resolve(),
            older.resolve(),
        ]

        monkeypatch.setattr(
            manager, "_confirm_save_dirty_workspace", lambda _message: True
        )
        assert manager.open_recent_workspace(older)
        assert load_calls == [
            (
                older.resolve(),
                {
                    "replace": True,
                    "associate": True,
                    "mark_dirty": False,
                    "select": False,
                },
            )
        ]
        assert manager._recent_workspace_paths() == [
            older.resolve(),
            newer.resolve(),
        ]

        missing_warnings: list[tuple[typing.Any, ...]] = []
        manager._set_recent_workspace_paths([missing.resolve(), older.resolve()])
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "warning",
            lambda *args: missing_warnings.append(args),
        )

        assert not manager.open_recent_workspace(missing)
        assert len(missing_warnings) == 1
        assert manager._recent_workspace_paths() == [older.resolve()]


def test_manager_open_recent_workspace_reports_load_errors(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    workspace = tmp_path / "broken.itws"
    workspace.touch()

    with manager_context() as manager:
        monkeypatch.setattr(
            manager, "_confirm_save_dirty_workspace", lambda _message: True
        )
        monkeypatch.setattr(
            manager,
            "_load_workspace_file",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        lock_errors: list[pathlib.Path] = []
        monkeypatch.setattr(
            manager_workspace,
            "_is_workspace_file_lock_error",
            lambda _exc: True,
        )
        monkeypatch.setattr(
            manager_mainwindow,
            "_show_workspace_file_lock_error",
            lambda _parent, path: lock_errors.append(pathlib.Path(path)),
        )
        assert not manager.open_recent_workspace(workspace)
        assert lock_errors == [workspace.resolve()]

        critical_messages: list[tuple[str, str]] = []
        monkeypatch.setattr(
            manager_workspace,
            "_is_workspace_file_lock_error",
            lambda _exc: False,
        )
        monkeypatch.setattr(
            erlab.interactive.utils.MessageDialog,
            "critical",
            lambda _parent, title, message: critical_messages.append((title, message)),
        )
        assert not manager.open_recent_workspace(workspace)
        assert critical_messages == [
            ("Error", "An error occurred while loading the workspace file.")
        ]


def test_manager_records_recent_workspace_accesses(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    opened = tmp_path / "opened.itws"
    saved = tmp_path / "saved.itws"
    imported = tmp_path / "imported.itws"
    for path in (opened, saved, imported):
        path.touch()

    with manager_context() as manager:
        with manager._workspace_document_access_context(opened) as access:
            manager._associate_loaded_workspace_file(
                opened,
                manager_workspace._current_workspace_schema_version(),
                workspace_access=access,
            )

        monkeypatch.setattr(
            manager, "_workspace_save_dialog", lambda **_kwargs: str(saved)
        )
        monkeypatch.setattr(
            manager, "_save_workspace_document", lambda *_args, **_kwargs: None
        )
        monkeypatch.setattr(
            manager,
            "_rebind_workspace_backed_imagetools",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            erlab.interactive.utils,
            "wait_dialog",
            lambda *_args, **_kwargs: contextlib.nullcontext(),
        )
        assert manager.save_as(native=False)

        monkeypatch.setattr(QtWidgets.QFileDialog, "exec", lambda _dialog: True)
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "selectedFiles",
            lambda _dialog: [str(imported)],
        )
        monkeypatch.setattr(
            manager, "_load_workspace_file", lambda *_args, **_kwargs: True
        )
        assert manager.import_workspace(native=False)

        assert manager._recent_workspace_paths()[:3] == [
            imported.resolve(),
            saved.resolve(),
            opened.resolve(),
        ]


def test_manager_records_packaged_workspace_with_desktop_shell(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    workspace = tmp_path / "workspace.itws"
    data_file = tmp_path / "data.h5"
    workspace.touch()
    data_file.touch()
    recorded: list[pathlib.Path] = []

    monkeypatch.setattr(
        manager_desktop,
        "record_recent_workspace",
        lambda path: recorded.append(pathlib.Path(path)),
    )

    with manager_context() as manager:
        monkeypatch.setattr(erlab.utils.misc, "_IS_PACKAGED", True)
        manager._record_recent_workspace(workspace)
        manager._record_recent_workspace(data_file)

    assert recorded == [workspace.resolve()]


def test_manager_startup_args_parse_flags_and_file_paths(tmp_path) -> None:
    workspace = tmp_path / "workspace.itws"
    data_file = tmp_path / "data.h5"
    workspace.touch()
    data_file.touch()

    file_args, open_workspace_dialog = manager_module._parse_startup_args(
        [
            manager_desktop.OPEN_WORKSPACE_DIALOG_ARG,
            str(workspace),
            manager_desktop.NEW_MANAGER_WINDOW_ARG,
            "--ignored",
            str(tmp_path / "missing.itws"),
            str(data_file),
        ]
    )

    assert open_workspace_dialog
    assert file_args == [workspace, data_file]


def test_manager_startup_open_workspace_dialog_schedules_load(
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    calls: list[tuple[int, weakref.ReferenceType[object] | None, str]] = []

    def record_single_shot(interval: int, callback: typing.Callable[[], None]) -> None:
        receiver = getattr(callback, "__self__", None)
        calls.append(
            (
                interval,
                weakref.ref(receiver) if receiver is not None else None,
                getattr(callback, "__name__", ""),
            )
        )

    monkeypatch.setattr(
        manager_module.sys,
        "argv",
        ["erlab-imagetool-manager", manager_desktop.OPEN_WORKSPACE_DIALOG_ARG],
    )
    monkeypatch.setattr(
        manager_module.QtCore.QTimer,
        "singleShot",
        record_single_shot,
    )

    with manager_context() as manager:
        assert any(
            interval == 0
            and receiver_ref is not None
            and receiver_ref() is manager
            and name == "load"
            for interval, receiver_ref, name in calls
        )


def test_manager_active_window_and_focus_restore_guards(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        active = QtWidgets.QWidget()
        origin = QtWidgets.QWidget()
        other = QtWidgets.QWidget()
        qtbot.addWidget(active)
        qtbot.addWidget(origin)
        qtbot.addWidget(other)
        origin.show()
        other.show()

        monkeypatch.setattr(
            QtWidgets.QApplication, "activeWindow", staticmethod(lambda: active)
        )
        monkeypatch.setattr(manager, "_node_uid_from_window", lambda _window: "uid")
        monkeypatch.setattr(erlab.interactive.utils, "qt_is_valid", lambda _obj: False)
        assert manager._active_managed_window() is None

        monkeypatch.setattr(
            QtWidgets.QApplication, "activeWindow", staticmethod(lambda: other)
        )
        monkeypatch.setattr(erlab.interactive.utils, "qt_is_valid", lambda _obj: True)
        manager._restore_focus_after_workspace_save(origin)


def test_manager_compact_workspace_edge_paths(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        monkeypatch.setattr(manager, "save_as", lambda: True)
        assert manager.compact_workspace()

        manager._workspace_path = tmp_path / "workspace.itws"
        manager._workspace_save_in_progress = True
        assert not manager.compact_workspace()
        manager._workspace_save_in_progress = False

        operation_errors: list[tuple[typing.Any, ...]] = []
        focus_restores: list[QtWidgets.QWidget | None] = []
        monkeypatch.setattr(
            erlab.interactive.utils,
            "wait_dialog",
            lambda *args, **kwargs: contextlib.nullcontext(),
        )
        monkeypatch.setattr(
            manager,
            "_save_workspace_document",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("compact failed")
            ),
        )
        monkeypatch.setattr(
            manager,
            "_show_operation_error",
            lambda *args: operation_errors.append(args),
        )
        monkeypatch.setattr(
            manager,
            "_restore_focus_after_workspace_save",
            lambda origin: focus_restores.append(origin),
        )

        assert not manager.compact_workspace()
        assert operation_errors == [
            (
                "Error while compacting workspace",
                "An error occurred while compacting the workspace file.",
            )
        ]
        assert focus_restores == [None]


def test_manager_shutdown_compaction_logs_failure(
    caplog,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager._workspace_path = tmp_path / "workspace.itws"
        manager._workspace_delta_save_count = 1

        def _fail_worker(
            _fname: str | os.PathLike[str],
            snapshot: manager_workspace._WorkspaceSaveSnapshot,
            _origin: QtWidgets.QWidget | None,
            **_kwargs,
        ) -> tuple[bool, float, str]:
            snapshot.close()
            return False, 0.0, "compact failed"

        monkeypatch.setattr(
            manager,
            "_run_workspace_save_worker",
            _fail_worker,
        )

        with caplog.at_level(logging.ERROR, logger=manager_mainwindow.logger.name):
            manager._compact_workspace_before_shutdown()

    assert "Failed to compact workspace before shutdown" in caplog.text
    assert "compact failed" in caplog.text


def test_manager_workspace_save_dialog_paths(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    calls: list[tuple[str, object]] = []
    original_file_dialog = QtWidgets.QFileDialog

    class _FakeFileDialog:
        AcceptMode = original_file_dialog.AcceptMode
        FileMode = original_file_dialog.FileMode
        Option = original_file_dialog.Option
        exec_result = 0

        def __init__(self, _parent, caption: str) -> None:
            calls.append(("caption", caption))

        def setAcceptMode(self, mode) -> None:
            calls.append(("accept", mode))

        def setFileMode(self, mode) -> None:
            calls.append(("file_mode", mode))

        def setNameFilter(self, name_filter: str) -> None:
            calls.append(("filter", name_filter))

        def setDefaultSuffix(self, suffix: str) -> None:
            calls.append(("suffix", suffix))

        def selectFile(self, fname: str) -> None:
            calls.append(("select", fname))

        def setDirectory(self, directory: str) -> None:
            calls.append(("directory", directory))

        def setOption(self, option) -> None:
            calls.append(("option", option))

        def exec(self) -> int:
            return self.exec_result

        def selectedFiles(self) -> list[str]:
            return [str(tmp_path / "selected.itws")]

    monkeypatch.setattr(QtWidgets, "QFileDialog", _FakeFileDialog)
    with manager_context() as manager:
        assert (
            manager._workspace_save_dialog(
                native=False, selected_file=tmp_path / "explicit.itws"
            )
            is None
        )
        assert ("select", str(tmp_path / "explicit.itws")) in calls

        _FakeFileDialog.exec_result = 1
        manager._workspace_path = tmp_path / "bound.itws"
        assert manager._workspace_save_dialog(native=True) == str(
            tmp_path / "selected.itws"
        )
        assert ("select", str(tmp_path / "bound.itws")) in calls

        manager._workspace_path = None
        manager._recent_directory = str(tmp_path)
        assert manager._workspace_save_dialog(native=True) == str(
            tmp_path / "selected.itws"
        )
        assert ("directory", str(tmp_path)) in calls


def test_manager_confirm_save_dirty_workspace_save_branch(
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager._workspace_structure_modified = True
        monkeypatch.setattr(manager, "save", lambda: True)
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "exec",
            lambda _msg_box: QtWidgets.QMessageBox.StandardButton.Save,
        )

        assert manager._confirm_save_dirty_workspace("Save before continuing.")


def test_manager_legacy_workspace_save_helpers(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "exec",
            lambda _msg_box: QtWidgets.QMessageBox.StandardButton.Ok,
        )
        manager._show_legacy_workspace_upgrade_message(tmp_path / "legacy.itws")

        monkeypatch.setattr(manager, "_workspace_save_dialog", lambda **_kwargs: None)
        assert manager._save_legacy_workspace_as_v4(tmp_path / "legacy.itws") is None

        dirty_reasons: list[str] = []
        monkeypatch.setattr(
            manager,
            "_save_legacy_workspace_as_v4",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            manager, "_mark_workspace_structure_dirty", dirty_reasons.append
        )
        manager._associate_loaded_workspace_file(
            tmp_path / "legacy.itws",
            manager_workspace._WORKSPACE_LEGACY_SCHEMA_VERSION - 1,
        )

        assert manager._workspace_path is None
        assert manager._workspace_needs_full_save
        assert dirty_reasons == ["Legacy workspace needs conversion"]


def test_manager_save_and_wait_dialog_error_paths(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    critical_calls: list[tuple[typing.Any, ...]] = []

    def _critical(*args, **kwargs) -> int:
        critical_calls.append(args)
        return int(QtWidgets.QDialog.DialogCode.Accepted)

    monkeypatch.setattr(erlab.interactive.utils.MessageDialog, "critical", _critical)
    with manager_context() as manager:
        wait_dialog = manager._open_workspace_save_wait_dialog(manager)
        qtbot.addWidget(wait_dialog)
        assert wait_dialog.windowTitle() == "Saving Workspace"
        assert typing.cast("QtWidgets.QProgressDialog", wait_dialog).labelText() == (
            "Saving workspace..."
        )
        wait_dialog.close()

        manager._show_workspace_save_worker_error("Traceback text")
        assert critical_calls[-1][2] == (
            "An error occurred while saving the workspace file."
        )

        manager._workspace_path = tmp_path / "workspace.itws"
        manager._workspace_save_in_progress = True
        assert not manager.save()

        manager._workspace_save_in_progress = False
        monkeypatch.setattr(
            manager,
            "_workspace_save_snapshot",
            lambda _path: (_ for _ in ()).throw(RuntimeError("snapshot failed")),
        )
        monkeypatch.setattr(
            manager, "_restore_focus_after_workspace_save", lambda _origin: None
        )
        assert not manager.save()

        monkeypatch.setattr(manager, "_workspace_save_dialog", lambda **_kwargs: None)
        assert not manager.save_as()


def test_open_multiple_files_workspace_locks_before_recovery(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    fname = tmp_path / "locked-dropped.itws"
    _write_transaction_test_workspace(fname)
    lock = manager_workspace._acquire_workspace_document_lock(fname)
    lock_calls: list[pathlib.Path] = []
    recovery_calls: list[pathlib.Path] = []
    try:
        monkeypatch.setattr(
            manager_workspace,
            "_recover_workspace_transactions",
            lambda path: (
                recovery_calls.append(pathlib.Path(path))
                or pytest.fail("Dropped workspace should lock before recovery")
            ),
        )
        monkeypatch.setattr(
            manager_mainwindow,
            "_show_workspace_file_lock_error",
            lambda _parent, locked_fname: lock_calls.append(pathlib.Path(locked_fname)),
        )
        monkeypatch.setattr(
            erlab.interactive.utils,
            "file_loaders",
            lambda *args, **kwargs: pytest.fail(
                "locked workspace should not fall through to loaders"
            ),
        )

        with manager_context() as manager:
            manager.open_multiple_files([fname], try_workspace=True)
    finally:
        lock.unlock()

    assert recovery_calls == []
    assert lock_calls == [fname]


def test_open_multiple_files_loads_workspace_and_reads_metadata(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25.0).reshape(5, 5), dims=("x", "y"))
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        fname = tmp_path / "open-multiple.itws"
        manager._save_workspace_document(fname, force_full=True)

        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)
        manager.open_multiple_files([fname], try_workspace=True)

        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        assert manager.workspace_path == str(fname.resolve())


def test_workspace_high_risk_path_detection() -> None:
    assert manager_workspace._workspace_path_is_high_risk(
        pathlib.Path.home() / "OneDrive" / "workspace.itws"
    )
    assert manager_workspace._workspace_path_is_high_risk(
        pathlib.Path.home()
        / "Library"
        / "Mobile Documents"
        / "com~apple~CloudDocs"
        / "workspace.itws"
    )
    assert manager_workspace._workspace_path_is_high_risk(
        pathlib.Path("//server/share/workspace.itws")
    )


def test_workspace_lock_error_detection_message_variants() -> None:
    transient = OSError(errno.EACCES, "resource temporarily unavailable")
    assert manager_workspace._is_workspace_file_lock_error(transient)
    assert manager_workspace._is_workspace_file_lock_error(
        RuntimeError("file is already open by another process")
    )
    assert manager_workspace._is_workspace_file_lock_error(
        RuntimeError("unable to lock file")
    )
    assert not manager_workspace._is_workspace_file_lock_error(
        OSError(errno.EINVAL, "resource temporarily unavailable")
    )


def test_hide_workspace_lock_file_windows_paths(monkeypatch) -> None:
    import ctypes

    calls: list[tuple[str, int]] = []

    class _Kernel32:
        @staticmethod
        def SetFileAttributesW(path: str, attrs: int) -> None:
            calls.append((path, attrs))

    monkeypatch.setattr(manager_workspace.sys, "platform", "win32")
    monkeypatch.setattr(manager_workspace.os, "name", "nt")
    monkeypatch.setattr(ctypes, "windll", None, raising=False)
    manager_workspace._hide_workspace_lock_file("missing-windll.itws.lock")
    assert calls == []

    monkeypatch.setattr(
        ctypes, "windll", types.SimpleNamespace(kernel32=_Kernel32()), raising=False
    )
    manager_workspace._hide_workspace_lock_file("hidden.itws.lock")
    assert calls == [("hidden.itws.lock", 0x2)]


def test_workspace_document_lock_info_without_lock(tmp_path) -> None:
    info = manager_workspace._workspace_document_lock_info(tmp_path / "free.itws")

    assert info.pid is None
    assert info.hostname == ""
    assert info.appname == ""


def test_workspace_metadata_helpers_cover_invalid_payloads() -> None:
    manifest_attrs = manager_workspace._workspace_root_attrs_payload(
        root_order=["1"],
        nodes=[{"path": "1"}],
        delta_save_count=2,
        erlab_version="test",
    )
    raw_manifest = manifest_attrs[manager_workspace._WORKSPACE_MANIFEST_ATTR]

    assert (
        manager_workspace._workspace_manifest_from_attrs(
            {manager_workspace._WORKSPACE_MANIFEST_ATTR: raw_manifest.encode()}
        )["delta_save_count"]
        == 2
    )
    assert (
        manager_workspace._workspace_manifest_from_attrs(
            {manager_workspace._WORKSPACE_MANIFEST_ATTR: "{not-json"}
        )
        == {}
    )
    assert (
        manager_workspace._workspace_delta_save_count_from_attrs(
            {
                manager_workspace._WORKSPACE_MANIFEST_ATTR: (
                    '{"delta_save_count": "not-an-int"}'
                )
            }
        )
        == 0
    )


def test_workspace_path_risk_detection_fallbacks(monkeypatch, tmp_path) -> None:
    def _raise_oserror(_path: pathlib.Path) -> pathlib.Path:
        raise OSError("resolve failed")

    monkeypatch.setattr(pathlib.Path, "resolve", _raise_oserror)
    assert manager_workspace._workspace_path_is_likely_cloud_path(
        tmp_path / "Dropbox" / "workspace.itws"
    )
    assert manager_workspace._workspace_path_is_likely_network_path(
        pathlib.Path("/net/server/workspace.itws")
    )

    monkeypatch.setattr(manager_workspace.sys, "platform", "darwin")
    assert manager_workspace._workspace_path_is_likely_network_path(
        pathlib.Path("/Volumes/share/workspace.itws")
    )


def test_workspace_requires_full_save_reasons(tmp_path) -> None:
    options = erlab.interactive.options
    old_incremental = options["io/workspace/use_incremental"]
    old_remote = options["io/workspace/incremental_save_on_remote"]
    existing = tmp_path / "existing.itws"
    existing.touch()
    try:
        options["io/workspace/use_incremental"] = False
        assert manager_workspace._workspace_requires_full_save(
            existing,
            needs_full_save=False,
            schema_version=manager_workspace._current_workspace_schema_version(),
            structure_modified=False,
            has_dirty_added=False,
            has_dirty_removed=False,
        )

        options["io/workspace/use_incremental"] = True
        options["io/workspace/incremental_save_on_remote"] = True
        assert manager_workspace._workspace_requires_full_save(
            tmp_path / "missing.itws",
            needs_full_save=False,
            schema_version=manager_workspace._current_workspace_schema_version(),
            structure_modified=False,
            has_dirty_added=False,
            has_dirty_removed=False,
        )
        for kwargs in (
            {"needs_full_save": True},
            {
                "schema_version": (
                    manager_workspace._current_workspace_schema_version() - 1
                )
            },
            {"structure_modified": True},
            {"has_dirty_added": True},
            {"has_dirty_removed": True},
        ):
            call_kwargs = {
                "needs_full_save": False,
                "schema_version": manager_workspace._current_workspace_schema_version(),
                "structure_modified": False,
                "has_dirty_added": False,
                "has_dirty_removed": False,
            }
            call_kwargs.update(kwargs)
            assert manager_workspace._workspace_requires_full_save(
                existing, **call_kwargs
            )
    finally:
        options["io/workspace/use_incremental"] = old_incremental
        options["io/workspace/incremental_save_on_remote"] = old_remote


def test_workspace_h5_transaction_helper_edge_cases(tmp_path) -> None:
    import h5py

    fname = tmp_path / "transaction-helpers.itws"
    with h5py.File(fname, "w") as h5_file:
        h5_file.attrs["imagetool_workspace_schema_version"] = (
            manager_workspace._current_workspace_schema_version()
        )
        assert manager_workspace._workspace_txn_attr_target(h5_file, "/missing") is None

        txn = h5_file.create_group(
            f"{manager_workspace._WORKSPACE_TRANSACTION_GROUP_PREFIX}x"
        )
        txn_name = txn.name.strip("/")
        manager_workspace._restore_workspace_attr_backups(h5_file, txn)

        txn.attrs["operations"] = b'{"group_replacements": []}'
        assert manager_workspace._workspace_transaction_operations(txn) == {
            "group_replacements": []
        }
        txn.attrs["operations"] = "{not-json"
        assert manager_workspace._workspace_transaction_operations(txn) == {}

        txn.attrs["pending_root"] = b"__itws_pending_x"
        txn.attrs["backup_root"] = b"__itws_backup_x"
        assert manager_workspace._workspace_transaction_roots(txn) == (
            "__itws_pending_x",
            "__itws_backup_x",
        )

        manager_workspace._rollback_workspace_group_operations(
            h5_file, {"group_replacements": "not-a-list"}
        )
        manager_workspace._rollback_workspace_group_operations(
            h5_file,
            {"group_replacements": [None, {"group_path": 1, "backup_path": "x"}]},
        )

        target = h5_file.create_group("target")
        target.attrs["value"] = "old"
        txn.attrs["status"] = b"committing"
        txn.attrs["operations"] = json.dumps(
            {
                "group_replacements": [
                    {
                        "group_path": "target",
                        "backup_path": "missing-backup",
                        "old_exists": False,
                    }
                ]
            }
        )
        pending = h5_file.create_group("__itws_pending_x")
        pending.attrs["unused"] = True
        backup = h5_file.create_group("__itws_backup_x")
        backup.attrs["unused"] = True

        manager_workspace._recover_open_workspace_transaction(h5_file, txn.name)

        assert "target" not in h5_file
        assert "__itws_pending_x" not in h5_file
        assert "__itws_backup_x" not in h5_file
        assert txn_name not in h5_file


def test_recover_workspace_transactions_ignores_non_workspace_file(tmp_path) -> None:
    import h5py

    fname = tmp_path / "plain.h5"
    with h5py.File(fname, "w") as h5_file:
        h5_file.create_group(
            f"{manager_workspace._WORKSPACE_TRANSACTION_GROUP_PREFIX}x"
        )

    manager_workspace._recover_workspace_transactions(fname)

    with h5py.File(fname, "r") as h5_file:
        assert f"{manager_workspace._WORKSPACE_TRANSACTION_GROUP_PREFIX}x" in h5_file


def test_validate_workspace_h5_file_rejects_non_workspace(tmp_path) -> None:
    import h5py

    fname = tmp_path / "invalid.h5"
    with h5py.File(fname, "w"):
        pass

    with pytest.raises(ValueError, match="not valid"):
        manager_workspace._validate_workspace_h5_file(fname)


def test_fsync_parent_directory_skips_non_posix(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(manager_workspace.os, "name", "nt")
    monkeypatch.setattr(
        manager_workspace.os,
        "open",
        lambda *args, **kwargs: pytest.fail("non-posix platforms should not fsync"),
    )

    manager_workspace._fsync_parent_directory(tmp_path / "workspace.itws")


def test_manager_workspace_v4_save_open_replaces_and_binds_path(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        child = itool(data + 1, manager=False, execute=False)
        assert isinstance(child, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(child, 0, show=False)

        fname = tmp_path / "bound.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        assert manager.workspace_path == str(fname.resolve())
        assert not manager.is_workspace_modified

        with h5py.File(fname, "r") as h5_file:
            assert h5_file.attrs["imagetool_workspace_schema_version"] == 4
            manifest = json.loads(h5_file.attrs["imagetool_workspace_manifest"])
        assert manifest["schema_version"] == 4
        assert {node["uid"] for node in manifest["nodes"]} >= {
            manager._imagetool_wrappers[0].uid,
            child_uid,
        }

        extra = itool(data + 2, manager=False, execute=False)
        assert isinstance(extra, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(extra, show=False)
        assert manager.ntools == 2

        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        assert manager.workspace_path == str(fname.resolve())
        assert not manager.is_workspace_modified
        assert manager._imagetool_wrappers[0]._childtool_indices == [child_uid]
        assert manager.get_imagetool(0).slicer_area._data.chunks is None
        assert _compute_first_value(manager.get_imagetool(0).slicer_area._data) == 0


def test_manager_workspace_h5py_fast_path_preserves_spaced_associated_coord(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25.0).reshape((5, 5)),
            dims=["x", "y"],
            coords={
                "x": np.arange(5.0),
                "y": np.arange(5.0),
                "Fake Motor": (("x", "y"), np.arange(25.0).reshape((5, 5))),
            },
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "spaced-coord.itws"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            manager._save_workspace_document(fname, force_full=True)
        assert not any("space in its name" in str(item.message) for item in caught)

        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)
        monkeypatch.setattr(
            manager_xarray,
            "open_workspace_dataset",
            lambda *args, **kwargs: pytest.fail(
                "spaced numeric coords should stay on the h5py fast path"
            ),
        )

        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        loaded = manager.get_imagetool(0).slicer_area._data
        assert loaded.chunks is None
        assert "Fake Motor" in loaded.coords
        xarray.testing.assert_equal(
            loaded.coords["Fake Motor"], data.coords["Fake Motor"]
        )


def test_manager_workspace_import_appends_without_reassociation(
    qtbot,
    monkeypatch,
    accept_dialog,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    choose_dialog_calls = {"count": 0}

    class _SelectSecondDialog(_ChooseFromDataTreeDialog):
        def __init__(self, *args, **kwargs) -> None:
            choose_dialog_calls["count"] += 1
            super().__init__(*args, **kwargs)
            first_item = self._tree_widget.topLevelItem(0)
            assert first_item is not None
            first_item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)

        def exec(self) -> QtWidgets.QDialog.DialogCode:
            return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(
        erlab.interactive.imagetool.manager._mainwindow,
        "_ChooseFromDataTreeDialog",
        _SelectSecondDialog,
    )

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        base_tool = itool(data, manager=False, execute=False)
        assert isinstance(base_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(base_tool, show=False)

        current_fname = tmp_path / "current.itws"
        manager._save_workspace_document(current_fname, force_full=True)
        manager._adopt_workspace_path(current_fname)
        manager._mark_workspace_clean()

        import_tool = itool(data + 1, manager=False, execute=False)
        assert isinstance(import_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(import_tool, show=False)
        import_fname = tmp_path / "import.itws"
        manager._save_workspace_document(import_fname, force_full=True)

        manager.remove_imagetool(1)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        manager._mark_workspace_clean()

        def _go_to_file(dialog: QtWidgets.QFileDialog):
            dialog.setDirectory(str(tmp_path))
            dialog.selectFile(str(import_fname))
            focused = dialog.focusWidget()
            if isinstance(focused, QtWidgets.QLineEdit):
                focused.setText(import_fname.name)

        accept_dialog(
            lambda: manager.import_workspace(native=False),
            pre_call=_go_to_file,
        )

        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        assert choose_dialog_calls["count"] == 1
        assert manager.workspace_path == str(current_fname.resolve())
        assert manager.is_workspace_modified


def test_manager_workspace_save_as_preserves_live_in_memory_windows(
    qtbot,
    accept_dialog,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    dask_options = erlab.interactive.options.model.io.dask
    old_threshold = dask_options.compute_threshold
    object.__setattr__(dask_options, "compute_threshold", 0)
    try:
        with manager_context() as manager:
            qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
            data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])

            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)
            child = itool(data + 1.0, manager=False, execute=False)
            assert isinstance(child, erlab.interactive.imagetool.ImageTool)
            child_uid = manager.add_imagetool_child(child, 0, show=False)

            def _load_workspace_file_should_not_run(*args, **kwargs):
                raise AssertionError("Save As should not reload the saved workspace")

            monkeypatch.setattr(
                manager, "_load_workspace_file", _load_workspace_file_should_not_run
            )

            new_fname = tmp_path / "new.itws"

            def _go_to_file(dialog: QtWidgets.QFileDialog):
                dialog.setDirectory(str(tmp_path))
                dialog.selectFile(str(new_fname))
                focused = dialog.focusWidget()
                if isinstance(focused, QtWidgets.QLineEdit):
                    focused.setText(new_fname.name)

            accept_dialog(lambda: manager.save_as(native=False), pre_call=_go_to_file)

            assert manager.workspace_path == str(new_fname.resolve())
            assert not manager.is_workspace_modified
            assert manager.get_imagetool(0) is root
            assert manager._child_node(child_uid).imagetool is child
            assert manager._imagetool_wrappers[0]._childtool_indices == [child_uid]
            assert root.slicer_area._data.chunks is None
            assert child.slicer_area._data.chunks is None
    finally:
        object.__setattr__(dask_options, "compute_threshold", old_threshold)


def test_manager_offload_to_workspace_save_as_rebinds_root_as_dask(
    qtbot,
    accept_dialog,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25.0).reshape((5, 5)), dims=["x", "y"], name="source"
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        select_tools(manager, [0])
        manager._update_actions()
        assert manager.offload_action.isEnabled()
        assert root.slicer_area._data.chunks is None

        fname = tmp_path / "offload.itws"

        def _go_to_file(dialog: QtWidgets.QFileDialog):
            dialog.setDirectory(str(tmp_path))
            dialog.selectFile(str(fname))
            focused = dialog.focusWidget()
            if isinstance(focused, QtWidgets.QLineEdit):
                focused.setText(fname.name)

        results: list[bool] = []
        accept_dialog(
            lambda: results.append(manager.offload_to_workspace([0], native=False)),
            pre_call=_go_to_file,
        )

        assert results == [True]
        assert manager.workspace_path == str(fname.resolve())
        assert not manager.is_workspace_modified

        rebound = manager.get_imagetool(0).slicer_area._data
        assert rebound.chunks is not None
        assert manager_xarray._normalized_file_path(rebound.encoding.get("source")) == (
            str(fname.resolve())
        )
        assert _compute_first_value(rebound) == 0.0

        manager._update_actions()
        assert not manager.offload_action.isEnabled()


def test_manager_workspace_load_reopens_offloaded_data_as_dask(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "offload-reopen.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()

        assert manager.offload_to_workspace([0], native=False)
        assert root.slicer_area._data.chunks is not None
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        loaded = manager.get_imagetool(0).slicer_area._data
        assert loaded.chunks is not None
        assert manager_xarray._normalized_file_path(loaded.encoding.get("source")) == (
            str(fname.resolve())
        )
        assert _compute_first_value(loaded) == 0.0


def test_manager_workspace_import_reopens_offloaded_data_as_dask(
    qtbot,
    accept_dialog,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    fname = tmp_path / "offload-import.itws"
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()
        assert manager.offload_to_workspace([0], native=False)
        assert root.slicer_area._data.chunks is not None

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        loaded: list[bool] = []
        accept_dialog(
            lambda: loaded.append(
                manager._load_workspace_file(
                    fname,
                    replace=False,
                    associate=False,
                    mark_dirty=True,
                    select=True,
                )
            )
        )

        assert loaded == [True]
        loaded_data = manager.get_imagetool(0).slicer_area._data
        assert loaded_data.chunks is not None
        assert manager_xarray._normalized_file_path(
            loaded_data.encoding.get("source")
        ) == str(fname.resolve())
        assert _compute_first_value(loaded_data) == 0.0


def test_manager_workspace_load_reopens_offloaded_spaced_coord_data_as_dask(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25.0).reshape((5, 5)),
            dims=["x", "y"],
            coords={
                "x": np.arange(5.0),
                "y": np.arange(5.0),
                "Fake Motor": ("x", np.linspace(10.0, 20.0, 5)),
            },
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "offload-spaced-coord.itws"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            manager._save_workspace_document(fname, force_full=True)
        assert not any("space in its name" in str(item.message) for item in caught)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()

        assert manager.offload_to_workspace([0], native=False)
        assert root.slicer_area._data.chunks is not None
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        loaded = manager.get_imagetool(0).slicer_area._data
        assert loaded.chunks is not None
        assert "Fake Motor" in loaded.coords
        np.testing.assert_allclose(
            np.asarray(loaded.coords["Fake Motor"]),
            np.asarray(data.coords["Fake Motor"]),
        )


def test_manager_offload_to_workspace_saves_dirty_workspace_before_rebind(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "dirty-offload.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()

        updated = data + 10.0
        root.slicer_area.replace_source_data(
            updated, auto_compute=False, emit_edited=True
        )
        assert manager.is_workspace_modified

        original_save = manager.save
        save_calls: list[bool] = []

        def _save(*, native: bool = True) -> bool:
            save_calls.append(native)
            return original_save(native=native)

        monkeypatch.setattr(manager, "save", _save)

        assert manager.offload_to_workspace([0], native=False)
        assert save_calls == [False]

        rebound = manager.get_imagetool(0).slicer_area._data
        assert rebound.chunks is not None
        assert _compute_first_value(rebound) == 10.0
        assert not manager.is_workspace_modified

        with h5py.File(fname, "r") as h5_file:
            saved = h5_file["0/imagetool"][manager_mainwindow._ITOOL_DATA_NAME]
            assert saved[0, 0] == 10.0


def test_manager_compute_offloaded_workspace_data_marks_backing_dirty(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._imagetool_wrappers[0].uid

        fname = tmp_path / "compute-offloaded.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()

        assert manager.offload_to_workspace([0], native=False)
        assert root.slicer_area._data.chunks is not None
        assert not manager.is_workspace_modified

        root.slicer_area._compute_chunked()

        assert root.slicer_area._data.chunks is None
        assert uid in manager._workspace_dirty_data
        assert manager.is_workspace_modified

        select_tools(manager, [0])
        manager._update_actions()
        assert manager.offload_action.isEnabled()

        assert manager.save()
        assert not manager.is_workspace_modified

        with h5py.File(fname, "r") as h5_file:
            saved = h5_file["0/imagetool"][manager_mainwindow._ITOOL_DATA_NAME]
            assert saved.chunks is None


def test_manager_offload_to_workspace_save_cancel_or_failure_noop(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        monkeypatch.setattr(manager, "_workspace_save_dialog", lambda **_kwargs: None)
        assert not manager.offload_to_workspace([0], native=False)
        assert manager.workspace_path is None
        assert root.slicer_area._data.chunks is None

        fname = tmp_path / "failure-offload.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()
        manager._mark_node_data_dirty(manager._imagetool_wrappers[0].uid)

        monkeypatch.setattr(manager, "save", lambda *, native=True: False)
        assert not manager.offload_to_workspace([0], native=False)
        assert root.slicer_area._data.chunks is None


def test_manager_offload_to_workspace_edge_paths(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        calls: list[list[int]] = []
        monkeypatch.setattr(manager, "_selected_imagetool_targets", lambda: [0])
        monkeypatch.setattr(
            manager,
            "offload_to_workspace",
            lambda targets: calls.append(list(targets)) or True,
        )
        manager.offload_selected_to_workspace()
        assert calls == [[0]]

    with manager_context() as manager:
        assert not manager.offload_to_workspace([])

    fake_node = types.SimpleNamespace(
        is_imagetool=True,
        imagetool=object(),
        slicer_area=types.SimpleNamespace(data_chunked=False),
    )

    with manager_context() as manager:
        monkeypatch.setattr(manager, "_node_for_target", lambda _target: fake_node)
        monkeypatch.setattr(manager, "save_as", lambda *, native=True: True)
        assert not manager.offload_to_workspace([0], native=False)

    with manager_context() as manager:
        workspace = tmp_path / "offload-error.itws"
        manager._workspace_path = workspace
        manager._workspace_needs_full_save = False
        monkeypatch.setattr(manager, "_node_for_target", lambda _target: fake_node)
        monkeypatch.setattr(
            manager, "_active_managed_window", lambda: typing.cast("typing.Any", None)
        )
        monkeypatch.setattr(
            erlab.interactive.utils,
            "wait_dialog",
            lambda *_args, **_kwargs: contextlib.nullcontext(),
        )
        monkeypatch.setattr(
            manager,
            "_rebind_workspace_backed_imagetools",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        errors: list[tuple[str, str]] = []
        restored: list[object | None] = []
        monkeypatch.setattr(
            manager,
            "_show_operation_error",
            lambda title, message: errors.append((title, message)),
        )
        monkeypatch.setattr(
            manager, "_restore_focus_after_workspace_save", restored.append
        )

        assert not manager.offload_to_workspace([0], native=False)
        assert errors == [
            (
                "Error while offloading to workspace",
                "An error occurred while reconnecting data from the workspace file.",
            )
        ]
        assert restored == [None]


def test_manager_offload_to_workspace_preserves_child_source_state(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    prov = erlab.interactive.imagetool.provenance
    data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False, provenance_spec=prov.full_data())

        child = itool(data.copy(deep=False), manager=False, execute=False)
        assert isinstance(child, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child,
            0,
            show=False,
            source_spec=prov.full_data(),
            source_auto_update=True,
        )
        child_node = manager._child_node(child_uid)

        fname = tmp_path / "child-offload.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()

        assert manager.offload_to_workspace([0], native=False)
        assert manager.get_imagetool(0).slicer_area._data.chunks is not None
        assert child_node.source_state == "fresh"
        assert child.slicer_area._data.chunks is None

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_actions()
        assert manager.offload_action.isEnabled()

        assert manager.offload_to_workspace([child_uid], native=False)
        assert child.slicer_area._data.chunks is not None
        assert child_node.source_state == "fresh"
        assert _compute_first_value(child.slicer_area._data) == 0.0

        manager._update_actions()
        assert not manager.offload_action.isEnabled()


def test_manager_manual_chunk_edits_persist_on_next_workspace_save(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._imagetool_wrappers[0].uid

        fname = tmp_path / "manual-chunks.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()

        root.slicer_area._set_chunks({"x": 2, "y": 3})

        assert root.slicer_area._data.chunks == ((2, 2, 1), (3, 2))
        assert uid in manager._workspace_dirty_data
        assert manager.is_workspace_modified

        with h5py.File(fname, "r") as h5_file:
            saved = h5_file["0/imagetool"][manager_mainwindow._ITOOL_DATA_NAME]
            assert saved.chunks is None

        assert manager.save()
        assert not manager.is_workspace_modified

        with h5py.File(fname, "r") as h5_file:
            saved = h5_file["0/imagetool"][manager_mainwindow._ITOOL_DATA_NAME]
            assert saved.chunks == (2, 3)

        opened = manager_xarray.open_workspace_dataset(
            fname, manager._workspace_payload_path(uid), chunks={}
        )
        try:
            rebound = opened[manager_mainwindow._ITOOL_DATA_NAME]
            assert rebound.chunks == ((2, 2, 1), (3, 2))
        finally:
            opened.close()


def test_manager_workspace_full_save_preserves_non_dask_data(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    dask_options = erlab.interactive.options.model.io.dask
    old_threshold = dask_options.compute_threshold
    object.__setattr__(dask_options, "compute_threshold", 0)
    try:
        with manager_context() as manager:
            qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
            data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])

            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)
            assert root.slicer_area._data.chunks is None

            fname = tmp_path / "full-save.itws"
            manager._save_workspace_document(fname, force_full=True)
            manager._adopt_workspace_path(fname)
            manager._mark_workspace_clean()
            manager._workspace_needs_full_save = True

            assert manager.save()
            assert root.slicer_area._data.chunks is None
            assert _compute_first_value(root.slicer_area._data) == 0.0
    finally:
        object.__setattr__(dask_options, "compute_threshold", old_threshold)


def test_manager_workspace_save_as_preserves_external_non_dask_file_backed_data(
    qtbot,
    accept_dialog,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])

        old_fname = tmp_path / "old.h5"
        new_fname = tmp_path / "new.itws"
        xr.DataTree.from_dict({"0/imagetool": data.to_dataset(name="data")}).to_netcdf(
            old_fname, engine="h5netcdf", invalid_netcdf=True
        )
        source = _open_external_file_backed_hdf5_imagetool_data(old_fname)
        assert source.chunks is None

        root = itool(source, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        live_data = manager.get_imagetool(0).slicer_area._data
        old_source = str(old_fname.resolve())
        assert (
            manager_xarray._normalized_file_path(live_data.encoding.get("source"))
            == old_source
        )

        def _go_to_file(dialog: QtWidgets.QFileDialog):
            dialog.setDirectory(str(tmp_path))
            dialog.selectFile(str(new_fname))
            focused = dialog.focusWidget()
            if isinstance(focused, QtWidgets.QLineEdit):
                focused.setText(new_fname.name)

        accept_dialog(lambda: manager.save_as(native=False), pre_call=_go_to_file)

        rebound = manager.get_imagetool(0).slicer_area._data
        new_source = str(new_fname.resolve())
        rebound_source = manager_xarray._normalized_file_path(
            rebound.encoding.get("source")
        )
        assert rebound_source == old_source
        assert rebound_source != new_source
        assert rebound.chunks is None

        assert _compute_first_value(rebound) == 0.0


def test_manager_workspace_save_as_rebinds_workspace_non_dask_file_backed_data(
    qtbot,
    accept_dialog,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        old_fname = tmp_path / "old.itws"
        new_fname = tmp_path / "new.itws"
        manager._save_workspace_document(old_fname, force_full=True)
        manager._adopt_workspace_path(old_fname)
        manager._rebind_workspace_backed_imagetools(old_fname, targets=[0], chunks=None)

        live_data = manager.get_imagetool(0).slicer_area._data
        old_source = str(old_fname.resolve())
        assert (
            manager_xarray._normalized_file_path(live_data.encoding.get("source"))
            == old_source
        )
        assert live_data.chunks is None

        def _go_to_file(dialog: QtWidgets.QFileDialog):
            dialog.setDirectory(str(tmp_path))
            dialog.selectFile(str(new_fname))
            focused = dialog.focusWidget()
            if isinstance(focused, QtWidgets.QLineEdit):
                focused.setText(new_fname.name)

        accept_dialog(lambda: manager.save_as(native=False), pre_call=_go_to_file)

        rebound = manager.get_imagetool(0).slicer_area._data
        new_source = str(new_fname.resolve())
        assert (
            manager_xarray._normalized_file_path(rebound.encoding.get("source"))
            == new_source
        )
        assert rebound.chunks is None

        old_fname.unlink()
        assert _compute_first_value(rebound) == 0.0


def test_manager_workspace_save_as_preserves_manually_chunked_file_backed_data(
    qtbot,
    accept_dialog,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])

        old_fname = tmp_path / "old.h5"
        new_fname = tmp_path / "new.itws"
        xr.DataTree.from_dict({"0/imagetool": data.to_dataset(name="data")}).to_netcdf(
            old_fname, engine="h5netcdf", invalid_netcdf=True
        )
        source = _open_external_file_backed_hdf5_imagetool_data(old_fname)
        assert source.chunks is None

        root = itool(source, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root.slicer_area.replace_source_data(
            root.slicer_area._data.chunk({"x": 2, "y": 2}),
            auto_compute=False,
        )
        assert root.slicer_area._data.chunks is not None

        def _go_to_file(dialog: QtWidgets.QFileDialog):
            dialog.setDirectory(str(tmp_path))
            dialog.selectFile(str(new_fname))
            focused = dialog.focusWidget()
            if isinstance(focused, QtWidgets.QLineEdit):
                focused.setText(new_fname.name)

        accept_dialog(lambda: manager.save_as(native=False), pre_call=_go_to_file)

        rebound = manager.get_imagetool(0).slicer_area._data
        assert rebound.chunks is not None
        assert manager_xarray._normalized_file_path(
            rebound.encoding.get("source")
        ) == str(new_fname.resolve())

        old_fname.unlink()
        assert _compute_first_value(rebound) == 0.0


def test_manager_workspace_save_as_rebinds_lazy_data_to_new_document(
    qtbot,
    accept_dialog,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    dask_options = erlab.interactive.options.model.io.dask
    old_threshold = dask_options.compute_threshold
    object.__setattr__(dask_options, "compute_threshold", 0)
    try:
        with manager_context() as manager:
            qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
            data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)

            old_fname = tmp_path / "old.h5"
            new_fname = tmp_path / "new.itws"
            xr.DataTree.from_dict(
                {"0/imagetool": data.to_dataset(name="data")}
            ).to_netcdf(old_fname, engine="h5netcdf", invalid_netcdf=True)
            old_lazy = _open_external_lazy_hdf5_imagetool_data(old_fname)
            root.slicer_area.replace_source_data(old_lazy + 0, auto_compute=False)
            assert _compute_first_value(old_lazy) == 0

            def _load_workspace_file_should_not_run(*args, **kwargs):
                raise AssertionError("Save As should not reload the saved workspace")

            monkeypatch.setattr(
                manager, "_load_workspace_file", _load_workspace_file_should_not_run
            )

            def _go_to_file(dialog: QtWidgets.QFileDialog):
                dialog.setDirectory(str(tmp_path))
                dialog.selectFile(str(new_fname))
                focused = dialog.focusWidget()
                if isinstance(focused, QtWidgets.QLineEdit):
                    focused.setText(new_fname.name)

            accept_dialog(lambda: manager.save_as(native=False), pre_call=_go_to_file)

            assert manager.workspace_path == str(new_fname.resolve())
            rebound = manager.get_imagetool(0).slicer_area._data
            assert rebound.chunks is not None
            assert manager_xarray._normalized_file_path(
                rebound.encoding.get("source")
            ) == str(new_fname.resolve())
            old_fname.unlink()
            assert _compute_first_value(rebound) == 0
    finally:
        object.__setattr__(dask_options, "compute_threshold", old_threshold)


def test_manager_workspace_dirty_markers_are_node_scoped(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        child = itool(data + 1, manager=False, execute=False)
        assert isinstance(child, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(child, 0, show=False)

        fname = tmp_path / "dirty.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        assert not manager.is_workspace_modified
        assert not root.isWindowModified()
        assert not child.isWindowModified()

        manager._child_node(child_uid).name = "renamed child"
        assert manager.is_workspace_modified
        assert manager.isWindowModified()
        assert not root.isWindowModified()
        assert child.isWindowModified()
        details = manager._dirty_details_text()
        assert "State modified:\n- renamed child" in details
        assert "Data modified:" not in details


def test_manager_workspace_save_clears_deferred_dirty_events(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._imagetool_wrappers[0].uid

        fname = tmp_path / "deferred-dirty.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()

        QtCore.QTimer.singleShot(0, lambda: manager._mark_node_state_dirty(uid))
        manager._mark_node_state_dirty(uid)
        assert manager.is_workspace_modified
        assert root.isWindowModified()

        focus_restored: list[QtWidgets.QWidget | None] = []
        monkeypatch.setattr(manager, "_active_managed_window", lambda: root)
        monkeypatch.setattr(
            manager,
            "_restore_focus_after_workspace_save",
            lambda origin: focus_restored.append(origin),
        )
        monkeypatch.setattr(
            manager,
            "_open_workspace_save_wait_dialog",
            lambda *args, **kwargs: pytest.fail("regular Save should not be modal"),
        )

        assert manager.save()
        manager._drain_workspace_deferred_events()
        assert not manager.is_workspace_modified
        assert not root.isWindowModified()
        assert focus_restored == [root]


def test_manager_workspace_state_save_updates_attrs_without_full_rewrite(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._imagetool_wrappers[0].uid

        fname = tmp_path / "state-delta.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()
        manager._mark_node_state_dirty(uid)

        original_transaction_write = manager_workspace._write_workspace_transaction_file
        attr_write_calls: list[str] = []

        def _record_transaction_write(
            _fname: str | os.PathLike[str],
            rewrite_groups: Iterable[tuple[str, dict[str, xr.Dataset]]],
            attr_updates: Iterable[
                tuple[
                    str,
                    dict[str, typing.Any],
                    tuple[str, dict[str, xr.Dataset]],
                ]
            ],
            root_attrs: Mapping[str, typing.Any],
        ) -> None:
            rewrite_groups = tuple(rewrite_groups)
            updates = tuple(attr_updates)
            assert rewrite_groups == ()
            attr_write_calls.extend(update[0] for update in updates)
            original_transaction_write(_fname, rewrite_groups, updates, root_attrs)

        monkeypatch.setattr(
            manager_workspace,
            "_write_full_workspace_tree_file",
            lambda *args, **kwargs: pytest.fail(
                "state-only Save should not rewrite the full workspace"
            ),
        )
        monkeypatch.setattr(
            manager_workspace,
            "_write_workspace_root_attrs_to_file",
            lambda *args, **kwargs: pytest.fail(
                "state-only Save should batch root attrs with node attrs"
            ),
        )
        monkeypatch.setattr(
            manager_workspace,
            "_write_workspace_transaction_file",
            _record_transaction_write,
        )
        monkeypatch.setattr(
            manager,
            "_open_workspace_save_wait_dialog",
            lambda *args, **kwargs: pytest.fail("state-only Save should be fast"),
        )

        assert manager.save()
        assert attr_write_calls == ["0/imagetool"]
        assert not manager.is_workspace_modified
        assert not root.isWindowModified()


def test_manager_workspace_save_does_not_close_live_workspace_handles(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._imagetool_wrappers[0].uid

        fname = tmp_path / "live-handles.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()
        manager._mark_node_state_dirty(uid)

        assert manager.save()

        manager._mark_node_data_dirty(uid)
        assert manager.save()


def test_manager_workspace_save_preserves_live_lazy_readers_during_write(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "live-lazy.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        root = manager.get_imagetool(0)
        uid = manager._imagetool_wrappers[0].uid
        root.slicer_area.replace_source_data(
            manager._workspace_rebind_data_for_uid(fname, uid, chunks="auto"),
            auto_compute=False,
        )
        live_data = root.slicer_area._data
        assert live_data.chunks is not None
        assert _compute_first_value(live_data) == 0.0

        manager._mark_node_state_dirty(uid)
        original_write = manager_workspace._write_workspace_transaction_file
        computed_values: list[object] = []

        def _slow_write_workspace_transaction_file(*args, **kwargs):
            time.sleep(0.05)
            return original_write(*args, **kwargs)

        def _compute_live_data() -> None:
            computed_values.append(live_data.isel({"x": 1, "y": 1}).compute().item())

        monkeypatch.setattr(
            manager_workspace,
            "_write_workspace_transaction_file",
            _slow_write_workspace_transaction_file,
        )
        QtCore.QTimer.singleShot(10, _compute_live_data)

        assert manager.save()
        assert computed_values == [6.0]


def test_manager_workspace_save_shows_wait_dialog_when_actual_save_is_slow(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "slow-save.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()
        manager._mark_node_state_dirty(manager._imagetool_wrappers[0].uid)

        original_write = manager_workspace._write_workspace_transaction_file

        def _slow_write_workspace_transaction_file(*args, **kwargs):
            time.sleep(0.05)
            return original_write(*args, **kwargs)

        wait_calls: list[tuple[QtWidgets.QWidget, str, str]] = []

        def _fake_open_wait_dialog(
            parent: QtWidgets.QWidget,
            *,
            title: str = "Saving Workspace",
            label_text: str = "Saving workspace...",
        ) -> QtWidgets.QDialog:
            wait_calls.append((parent, title, label_text))
            return QtWidgets.QDialog(parent)

        focus_restored: list[QtWidgets.QWidget | None] = []
        monkeypatch.setattr(
            manager_mainwindow,
            "_WORKSPACE_SAVE_WAIT_DIALOG_THRESHOLD_SECONDS",
            0.01,
        )
        monkeypatch.setattr(
            manager_workspace,
            "_write_workspace_transaction_file",
            _slow_write_workspace_transaction_file,
        )
        monkeypatch.setattr(manager, "_active_managed_window", lambda: root)
        monkeypatch.setattr(
            manager,
            "_restore_focus_after_workspace_save",
            lambda origin: focus_restored.append(origin),
        )
        monkeypatch.setattr(
            manager,
            "_open_workspace_save_wait_dialog",
            _fake_open_wait_dialog,
        )

        assert manager.save()
        assert wait_calls == [(root, "Saving Workspace", "Saving workspace...")]
        assert focus_restored == [root]


def test_manager_workspace_save_keeps_post_command_changes_dirty(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._imagetool_wrappers[0].uid

        fname = tmp_path / "post-command-dirty.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()
        manager._mark_node_data_dirty(uid)

        def _fake_run_workspace_save_worker(
            _fname: str | os.PathLike[str],
            snapshot: manager_workspace._WorkspaceSaveSnapshot,
            _origin: QtWidgets.QWidget | None,
            **_kwargs,
        ) -> tuple[bool, float, str]:
            snapshot.close()
            manager._mark_node_state_dirty(uid)
            return True, 0.0, ""

        monkeypatch.setattr(
            manager,
            "_run_workspace_save_worker",
            _fake_run_workspace_save_worker,
        )

        assert manager.save()
        assert manager.is_workspace_modified
        assert root.isWindowModified()
        details = manager._dirty_details_text()
        assert "State modified:" in details
        assert "Data modified:" not in details


def test_manager_workspace_compact_resets_delta_count_and_cleans_internal_groups(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._imagetool_wrappers[0].uid

        fname = tmp_path / "compact.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()
        manager._mark_node_state_dirty(uid)
        assert manager.save()
        assert manager._workspace_delta_save_count == 1

        monkeypatch.setattr(
            erlab.interactive.utils,
            "wait_dialog",
            lambda *args, **kwargs: contextlib.nullcontext(),
        )

        assert manager.compact_workspace()
        assert manager._workspace_delta_save_count == 0
        _assert_no_workspace_internal_groups(fname)
        with h5py.File(fname, "r") as h5_file:
            assert (
                manager_workspace._workspace_delta_save_count_from_attrs(h5_file.attrs)
                == 0
            )
            manifest = manager_workspace._workspace_manifest_from_attrs(h5_file.attrs)
            assert "delta_save_count" not in manifest
            assert "transaction_protocol" not in manifest


def test_manager_workspace_shutdown_compacts_clean_delta_workspace(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._imagetool_wrappers[0].uid

        fname = tmp_path / "shutdown-compact.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()
        manager._mark_node_state_dirty(uid)
        assert manager.save()
        assert manager._workspace_delta_save_count == 1

        manager._compact_workspace_before_shutdown()

        assert manager._workspace_delta_save_count == 0
        with h5py.File(fname, "r") as h5_file:
            assert (
                manager_workspace._workspace_delta_save_count_from_attrs(h5_file.attrs)
                == 0
            )
            manifest = manager_workspace._workspace_manifest_from_attrs(h5_file.attrs)
            assert "delta_save_count" not in manifest
            assert "transaction_protocol" not in manifest


def test_manager_workspace_shutdown_compact_shows_optimization_wait_dialog(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._imagetool_wrappers[0].uid

        fname = tmp_path / "slow-shutdown-compact.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()
        manager._mark_node_state_dirty(uid)
        assert manager.save()
        assert manager._workspace_delta_save_count == 1

        original_write = manager_workspace._write_full_workspace_tree_file

        def _slow_write_full_workspace_tree_file(*args, **kwargs):
            time.sleep(0.05)
            return original_write(*args, **kwargs)

        wait_calls: list[tuple[QtWidgets.QWidget, str, str]] = []

        def _fake_open_wait_dialog(
            parent: QtWidgets.QWidget,
            *,
            title: str = "Saving Workspace",
            label_text: str = "Saving workspace...",
        ) -> QtWidgets.QDialog:
            wait_calls.append((parent, title, label_text))
            return QtWidgets.QDialog(parent)

        monkeypatch.setattr(
            manager_mainwindow,
            "_WORKSPACE_SAVE_WAIT_DIALOG_THRESHOLD_SECONDS",
            0.01,
        )
        monkeypatch.setattr(
            manager_workspace,
            "_write_full_workspace_tree_file",
            _slow_write_full_workspace_tree_file,
        )
        monkeypatch.setattr(
            manager,
            "_open_workspace_save_wait_dialog",
            _fake_open_wait_dialog,
        )

        manager._compact_workspace_before_shutdown()

        assert wait_calls == [
            (manager, "Optimizing Workspace", "Optimizing workspace file…")
        ]
        assert manager._workspace_delta_save_count == 0


def test_manager_workspace_shutdown_compact_skips_dirty_workspace(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._imagetool_wrappers[0].uid

        fname = tmp_path / "dirty-shutdown-compact.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()
        manager._workspace_delta_save_count = 1
        manager._mark_node_state_dirty(uid)

        monkeypatch.setattr(
            manager,
            "_run_workspace_save_worker",
            lambda *args, **kwargs: pytest.fail(
                "Dirty shutdown compaction should not write discarded changes"
            ),
        )

        manager._compact_workspace_before_shutdown()


def test_manager_workspace_high_risk_path_forces_full_save_snapshot(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._imagetool_wrappers[0].uid

        fname = tmp_path / "high-risk.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()
        manager._mark_node_state_dirty(uid)
        monkeypatch.setattr(
            manager_workspace, "_workspace_path_is_high_risk", lambda *_args: True
        )

        snapshot = manager._workspace_save_snapshot(fname)
        try:
            assert snapshot.full_tree is not None
            assert snapshot.delta_save_count == 0
        finally:
            snapshot.close()


def test_workspace_remote_incremental_option_allows_delta_save(
    monkeypatch,
    tmp_path,
) -> None:
    options = erlab.interactive.options
    incremental_name = "io/workspace/incremental_save_on_remote"
    use_incremental_name = "io/workspace/use_incremental"
    old_remote_value = options[incremental_name]
    old_incremental_value = options[use_incremental_name]
    fname = tmp_path / "remote-incremental.itws"
    fname.touch()
    monkeypatch.setattr(
        manager_workspace, "_workspace_path_is_high_risk", lambda *_args: True
    )
    options[incremental_name] = True
    options[use_incremental_name] = True
    try:
        assert not manager_workspace._workspace_requires_full_save(
            fname,
            needs_full_save=False,
            schema_version=manager_workspace._current_workspace_schema_version(),
            structure_modified=False,
            has_dirty_added=False,
            has_dirty_removed=False,
        )
    finally:
        options[incremental_name] = old_remote_value
        options[use_incremental_name] = old_incremental_value


def test_manager_close_suppresses_child_visibility_dirty(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=True)
        qtbot.wait_until(root.isVisible)

        fname = tmp_path / "quit-clean.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._drain_workspace_deferred_events()
        manager._mark_workspace_clean()
        assert not manager.is_workspace_modified

        manager._closing_workspace_document = True
        root.hide()
        manager._closing_workspace_document = False
        manager._drain_workspace_deferred_events()

        assert not manager.is_workspace_modified
        assert not root.isWindowModified()


def test_manager_application_quit_filter_routes_quit_to_manager(
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        calls: list[str] = []
        monkeypatch.setattr(
            manager,
            "close",
            lambda: calls.append("close") or False,
        )

        event = QtCore.QEvent(QtCore.QEvent.Type.Quit)
        assert manager._application_quit_filter is not None
        assert manager._application_quit_filter.eventFilter(None, event)
        assert calls == ["close"]


def test_manager_workspace_dirty_marker_not_saved_in_titles(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25).reshape((5, 5)), dims=["x", "y"], name="source"
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root_uid = manager._imagetool_wrappers[0].uid
        tool = DerivativeTool(data)
        tool_uid = manager.add_childtool(tool, 0, show=False)

        expect_title_placeholder = sys.platform != "darwin"
        assert ("[*]" in root.windowTitle()) is expect_title_placeholder
        assert ("[*]" in tool.windowTitle()) is expect_title_placeholder

        root.setWindowTitle("stale root title[*]")
        manager._imagetool_wrappers[0].update_title()
        assert "stale root title" in manager._imagetool_wrappers[0].label_text
        assert "[*]" not in manager._imagetool_wrappers[0].label_text

        root.setWindowTitle("stale root title[*]")
        tool.setWindowTitle("stale tool title[*]")
        manager._set_node_window_modified(root_uid, True)
        manager._set_node_window_modified(tool_uid, True)

        assert (
            root.windowTitle()
            == manager_mainwindow._window_title_with_modified_placeholder(
                manager._imagetool_wrappers[0].label_text
            )
        )
        assert (
            tool.windowTitle()
            == manager_mainwindow._window_title_with_modified_placeholder(
                f"{tool.tool_name}: {tool._tool_display_name}"
            )
        )

        fname = tmp_path / "titles.itws"
        manager._save_workspace_document(fname, force_full=True)

        import h5py

        with h5py.File(fname, "r") as h5_file:
            root_title = h5_file["0/imagetool"].attrs["itool_title"]
            tool_title = h5_file[f"0/childtools/{tool_uid}/tool"].attrs["tool_title"]

        assert "[*]" not in root_title
        assert "[*]" not in tool_title


def test_manager_workspace_save_preserves_reordered_roots(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        for value in range(3):
            data = xr.DataArray(
                np.full((5, 5), value), dims=["x", "y"], name=f"data_{value}"
            )
            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)

        fname = tmp_path / "ordered.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()

        model = manager.tree_view._model
        assert model.dropMimeData(
            model.mimeData([model.index(0, 0)]),
            QtCore.Qt.DropAction.MoveAction,
            model.rowCount(),
            0,
            QtCore.QModelIndex(),
        )
        assert manager._displayed_indices == [1, 2, 0]
        assert manager.is_workspace_modified
        assert manager.save()

        with h5py.File(fname, "r") as h5_file:
            manifest = json.loads(h5_file.attrs["imagetool_workspace_manifest"])
        assert manifest["root_order"] == [1, 2, 0]

        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)
        loaded_order = [
            int(manager.get_imagetool(index).slicer_area._data.values[0, 0])
            for index in manager._displayed_indices
        ]
        assert loaded_order == [1, 2, 0]


def test_manager_workspace_child_save_shortcuts_call_manager_save(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()
        calls: list[bool] = []

        def _fake_save(*, native: bool = True) -> bool:
            calls.append(native)
            return True

        monkeypatch.setattr(manager, "save", _fake_save)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=True)
        root_shortcuts = root.findChildren(QtWidgets.QShortcut)
        root_save = [
            shortcut
            for shortcut in root_shortcuts
            if shortcut.objectName() == "managerWorkspaceSaveShortcut"
        ]
        assert len(root_save) == 1
        root_save[0].activated.emit()

        child = itool(data + 1, manager=False, execute=False)
        assert isinstance(child, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool_child(child, 0, show=False)
        child_save = [
            shortcut
            for shortcut in child.findChildren(QtWidgets.QShortcut)
            if shortcut.objectName() == "managerWorkspaceSaveShortcut"
        ]
        assert len(child_save) == 1
        child_save[0].activated.emit()

        tool = DerivativeTool(data)
        manager.add_childtool(tool, 0, show=False)
        tool_save = [
            shortcut
            for shortcut in tool.findChildren(QtWidgets.QShortcut)
            if shortcut.objectName() == "managerWorkspaceSaveShortcut"
        ]
        assert len(tool_save) == 1
        tool_save[0].activated.emit()

        assert calls == [True, True, True]


def test_manager_workspace_delta_save_splits_state_and_data_writes(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        fname = tmp_path / "delta.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)

        dataset_writes: list[str | None] = []
        original_to_netcdf = xr.Dataset.to_netcdf

        def _to_netcdf_spy(self, *args, **kwargs):
            dataset_writes.append(kwargs.get("group"))
            return original_to_netcdf(self, *args, **kwargs)

        monkeypatch.setattr(xr.Dataset, "to_netcdf", _to_netcdf_spy)

        manager.rename_imagetool(0, "state only")
        assert manager.save()
        assert dataset_writes == []

        replacement = data.copy(deep=True)
        replacement.data = np.asarray(replacement.data) + 10
        root.slicer_area.replace_source_data(replacement)
        assert manager.save()

        import h5py

        with h5py.File(fname, "r") as h5_file:
            saved = h5_file["0/imagetool"][manager_mainwindow._ITOOL_DATA_NAME]
            assert saved[0, 0] == 10


def test_manager_workspace_full_save_preserves_in_memory_backing_after_rebind(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25, dtype=np.float64).reshape((5, 5)),
            dims=["x", "y"],
            coords={"x": np.arange(5), "y": np.arange(5)},
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "memory.itws"
        backing_snapshot = manager._workspace_data_backing_snapshot()
        manager._save_workspace_document(fname, force_full=True)
        manager._rebind_workspace_backed_imagetools(
            fname,
            backing_snapshot=backing_snapshot,
            old_workspace_path=None,
        )

        saved_data = manager.get_imagetool(0).slicer_area._data
        assert manager_xarray.dataarray_is_numpy_backed(saved_data)
        assert not manager_xarray.dataarray_is_file_backed(saved_data)


def test_manager_workspace_file_backed_data_can_load_into_memory(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source_file = tmp_path / "source.itws"
    source = xr.DataArray(
        np.arange(25, dtype=np.float64).reshape((5, 5)),
        dims=["x", "y"],
        coords={"x": np.arange(5), "y": np.arange(5)},
    )
    tree = xr.DataTree.from_dict(
        {"0/imagetool": source.to_dataset(name=manager_mainwindow._ITOOL_DATA_NAME)}
    )
    try:
        manager_workspace._write_full_workspace_tree_file(
            source_file, tree, _transaction_test_root_attrs()
        )
    finally:
        tree.close()

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        file_backed = _open_external_file_backed_hdf5_imagetool_data(source_file)
        root = itool(file_backed, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        slicer_area = manager.get_imagetool(0).slicer_area
        assert slicer_area.data_file_backed
        assert slicer_area.data_loadable
        slicer_area._compute_chunked()

        assert manager_xarray.dataarray_is_numpy_backed(slicer_area._data)
        assert not slicer_area.data_file_backed


def test_manager_workspace_load_keeps_saved_data_in_memory(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    dask_options = erlab.interactive.options.model.io.dask
    old_threshold = dask_options.compute_threshold
    object.__setattr__(dask_options, "compute_threshold", 0)
    try:
        with manager_context() as manager:
            qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
            data = xr.DataArray(
                np.arange(512 * 512, dtype=np.float64).reshape((512, 512)),
                dims=["x", "y"],
            )

            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)

            fname = tmp_path / "load-memory.itws"
            manager._save_workspace_document(fname, force_full=True)
            assert manager._load_workspace_file(
                fname, replace=True, associate=True, mark_dirty=False, select=False
            )

            loaded = manager.get_imagetool(0).slicer_area
            assert not loaded.data_chunked
            assert not loaded.data_file_backed
            assert manager_xarray.dataarray_is_numpy_backed(loaded._data)
    finally:
        object.__setattr__(dask_options, "compute_threshold", old_threshold)


def test_manager_workspace_load_uses_h5py_fast_path(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25, dtype=np.float64).reshape((5, 5)),
            dims=["x", "y"],
            coords={"x": np.arange(5), "y": np.arange(5)},
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "h5py-fast-load.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        monkeypatch.setattr(
            manager_xarray,
            "open_workspace_datatree",
            lambda *args, **kwargs: pytest.fail(
                "simple v4 load should not open the workspace DataTree"
            ),
        )

        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        loaded = manager.get_imagetool(0).slicer_area
        assert manager_xarray.dataarray_is_numpy_backed(loaded._data)
        assert not loaded.data_chunked
        assert not loaded.data_file_backed
        np.testing.assert_array_equal(loaded._data.values, data.values)


def test_manager_workspace_load_h5py_fast_path_falls_back_per_payload(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25, dtype=np.float64).reshape((5, 5)),
            dims=["x", "y"],
            coords={"x": np.arange(5), "y": np.arange(5), "label": "sample"},
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "h5py-group-fallback.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        monkeypatch.setattr(
            manager_xarray,
            "open_workspace_datatree",
            lambda *args, **kwargs: pytest.fail(
                "unsupported payload should fall back by group, not by whole tree"
            ),
        )

        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        loaded = manager.get_imagetool(0).slicer_area
        assert manager_xarray.dataarray_is_numpy_backed(loaded._data)
        assert not loaded.data_chunked
        assert not loaded.data_file_backed
        assert loaded._data.coords["label"].item() == "sample"


def test_manager_workspace_rejects_external_xarray_reader_for_active_workspace(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "lazy-state.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        lazy = _open_external_lazy_hdf5_imagetool_data(fname)
        manager.get_imagetool(0).slicer_area.set_data(lazy, auto_compute=False)
        assert _compute_first_value(manager.get_imagetool(0).slicer_area._data) == 0
        manager._mark_workspace_clean()

        errors: list[str] = []
        monkeypatch.setattr(manager, "_show_workspace_save_worker_error", errors.append)

        manager.rename_imagetool(0, "lazy state")
        assert not manager.save()
        assert errors
        with contextlib.suppress(Exception):
            lazy.close()


def test_manager_workspace_lazy_data_delta_save_uses_pending_group_before_replacing(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "lazy-data.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        replacement = data.copy(deep=True)
        replacement.data = np.asarray(replacement.data) + 10
        manager.get_imagetool(0).slicer_area.replace_source_data(
            replacement, auto_compute=False
        )
        assert manager.save()
        assert list(tmp_path.glob("lazy-data.itws.delta-*")) == []

        import h5py

        with h5py.File(fname, "r") as h5_file:
            saved = h5_file["0/imagetool"][manager_mainwindow._ITOOL_DATA_NAME]
            assert saved[0, 0] == 10


def test_manager_workspace_same_file_lazy_data_delta_save_does_not_deadlock(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(512 * 512, dtype=np.float64).reshape((512, 512)),
            dims=["x", "y"],
            coords={"x": np.arange(512), "y": np.arange(512)},
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "same-file-lazy.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        manager._rebind_workspace_backed_imagetools(fname, targets=[0], chunks={})
        assert manager.get_imagetool(0).slicer_area.data_chunked
        manager.get_imagetool(0).slicer_area._set_chunks({"x": 128, "y": 64})

        uid = manager._imagetool_wrappers[0].uid
        manager._mark_node_data_dirty(uid)
        assert manager.save()

        import h5py

        with h5py.File(fname, "r") as h5_file:
            saved = h5_file["0/imagetool"][manager_mainwindow._ITOOL_DATA_NAME]
            assert saved[0, 0] == 0
            assert saved.chunks == (128, 64)


def test_manager_workspace_lazy_data_delta_pending_failure_preserves_old_group(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "lazy-failure.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()

        replacement = data.copy(deep=True)
        replacement.data = np.asarray(replacement.data) + 10
        root.slicer_area.replace_source_data(replacement, auto_compute=False)

        def _write_partial_pending_then_raise(
            fname: str | os.PathLike[str],
            _constructor: Mapping[str, xr.Dataset],
            _group_path: str,
            pending_path: str,
        ) -> None:
            with manager_workspace._open_workspace_h5_file_for_update(fname) as h5_file:
                h5_file.create_group(pending_path)
            raise RuntimeError("pending write failed")

        monkeypatch.setattr(
            manager_workspace,
            "_write_workspace_constructor_groups_to_pending",
            _write_partial_pending_then_raise,
        )
        monkeypatch.setattr(
            manager, "_show_workspace_save_worker_error", lambda *args: None
        )

        assert not manager.save()
        import h5py

        with h5py.File(fname, "r") as h5_file:
            saved = h5_file["0/imagetool"][manager_mainwindow._ITOOL_DATA_NAME]
            assert saved[0, 0] == 0
            assert not any(
                manager_workspace._is_workspace_internal_group_name(name)
                for name in h5_file
            )


def test_manager_workspace_stale_pending_groups_do_not_poison_open_or_save(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "stale-pending.itws"
        manager._save_workspace_document(fname, force_full=True)
        with h5py.File(fname, "a") as h5_file:
            h5_file.create_group(
                f"{manager_workspace._WORKSPACE_PENDING_GROUP_PREFIX}stale"
            )
            h5_file.create_group(
                f"{manager_workspace._WORKSPACE_BACKUP_GROUP_PREFIX}stale"
            )

        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        manager.rename_imagetool(0, "cleaned")
        assert manager.save()
        with h5py.File(fname, "r") as h5_file:
            assert not any(
                manager_workspace._is_workspace_internal_group_name(name)
                for name in h5_file
            )


def test_manager_workspace_load_dialog_skips_stale_internal_groups(
    qtbot,
    accept_dialog,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "stale-dialog.itws"
        manager._save_workspace_document(fname, force_full=True)
        with h5py.File(fname, "a") as h5_file:
            h5_file.create_group(
                f"{manager_workspace._WORKSPACE_PENDING_GROUP_PREFIX}stale"
            )
            h5_file.create_group(
                f"{manager_workspace._WORKSPACE_BACKUP_GROUP_PREFIX}stale"
            )

        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        tree = manager_xarray.open_workspace_datatree(fname, chunks="auto")
        accept_dialog(
            lambda: manager._from_datatree(
                tree,
                replace=True,
                mark_dirty=False,
                select=True,
            )
        )

        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)


def test_manager_workspace_replace_load_failure_restores_previous_workspace(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        current_fname = tmp_path / "current.itws"
        manager._save_workspace_document(current_fname, force_full=True)
        manager._adopt_workspace_path(current_fname)
        manager._mark_workspace_clean()

        broken_fname = tmp_path / "broken.itws"
        with h5py.File(broken_fname, "w") as h5_file:
            h5_file.attrs["imagetool_workspace_schema_version"] = 4
            h5_file.create_group("0")

        with pytest.raises(ValueError, match="Workspace node"):
            manager._load_workspace_file(
                broken_fname,
                replace=True,
                associate=True,
                mark_dirty=False,
                select=False,
            )

        assert manager.workspace_path == str(current_fname.resolve())
        assert manager.ntools == 1
        xarray.testing.assert_equal(manager.get_imagetool(0).slicer_area._data, data)
        assert not manager.is_workspace_modified


def test_manager_workspace_load_visible_windows_stays_clean_after_events(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=True)
        qtbot.wait_until(root.isVisible)

        fname = tmp_path / "visible.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()

        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        for _ in range(3):
            QtWidgets.QApplication.sendPostedEvents(None, 0)
            QtWidgets.QApplication.processEvents()

        assert not manager.is_workspace_modified


def test_manager_workspace_roundtrip_preserves_controls_visibility(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        manager._mark_workspace_clean()
        assert not manager.is_workspace_modified

        root.mnb.action_dict["toggleControlsAct"].trigger()
        assert not root.controls_visible
        qtbot.wait_until(lambda: manager.is_workspace_modified, timeout=5000)

        fname = tmp_path / "controls.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        assert not manager.is_workspace_modified

        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        restored = manager.get_imagetool(0)
        assert not restored.controls_visible
        assert not restored.mnb.action_dict["toggleControlsAct"].isChecked()
        assert restored.slicer_area.state["controls_visible"] is False
        assert not manager.is_workspace_modified


def test_manager_workspace_delta_save_persists_geometry_changes(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "geometry.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()

        root.setGeometry(12, 34, 321, 234)
        qtbot.wait_until(lambda: manager.is_workspace_modified, timeout=5000)
        expected_rect = tuple(root.geometry().getRect())

        assert manager.save()
        with h5py.File(fname, "r") as h5_file:
            saved_rect = tuple(
                int(value) for value in h5_file["0/imagetool"].attrs["itool_rect"]
            )
        assert saved_rect == expected_rect


def test_manager_workspace_roundtrip_goldtool_child(
    qtbot,
    monkeypatch,
    gold,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()

        itool(gold, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child = GoldTool(gold.copy(deep=True), data_name="gold_input")
        child.set_source_binding(erlab.interactive.imagetool.provenance.full_data())
        child_uid = manager.add_childtool(child, 0, show=False)
        configure_goldtool_child(child, fitted=True, spline=True)

        expected_status = child.tool_status.model_copy(deep=True)
        expected_corrected = child.corrected.copy(deep=True)
        expected_source_spec = child.source_spec
        child.open_itool()
        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)
        output_uid = child_node._childtool_indices[0]

        tree = manager._to_datatree()
        assert (
            tree[f"0/childtools/{child_uid}/tool"].attrs["manager_node_uid"]
            == child_uid
        )

        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        for node in tree.values():
            manager._load_workspace_node(typing.cast("xr.DataTree", node))

        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        loaded_root = manager._imagetool_wrappers[0]
        assert loaded_root._childtool_indices == [child_uid]

        loaded_child = manager.get_childtool(child_uid)
        assert isinstance(loaded_child, GoldTool)
        assert loaded_child.source_spec == expected_source_spec
        assert loaded_child.tool_status == expected_status
        xr.testing.assert_identical(loaded_child.corrected, expected_corrected)
        loaded_child_node = manager._child_node(child_uid)
        assert loaded_child_node._childtool_indices == [output_uid]
        loaded_output_node = manager._child_node(output_uid)
        assert loaded_output_node.output_id == "goldtool.corrected"
        assert loaded_output_node.source_spec is None
        assert loaded_output_node.provenance_spec is not None
        assert loaded_output_node.provenance_spec.active_name == "corrected"
        xr.testing.assert_identical(fetch(output_uid), expected_corrected)

        monkeypatch.setattr(
            loaded_child, "_prompt_existing_output_imagetool", lambda: "update"
        )
        loaded_child.open_itool()
        assert loaded_child_node._childtool_indices == [output_uid]


def test_manager_workspace_roundtrip_dtool_child(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()

        itool(test_data, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 1, timeout=5000
        )

        child_uid = manager._imagetool_wrappers[0]._childtool_indices[0]
        child = manager.get_childtool(child_uid)
        assert isinstance(child, DerivativeTool)

        expected_status = child.tool_status.model_copy(deep=True)
        expected_result = child.result.T.copy(deep=True)
        expected_source_spec = child.source_spec
        child.open_itool()
        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)
        output_uid = child_node._childtool_indices[0]

        tree = manager._to_datatree()
        assert (
            tree[f"0/childtools/{child_uid}/tool"].attrs["manager_node_uid"]
            == child_uid
        )

        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        for node in tree.values():
            manager._load_workspace_node(typing.cast("xr.DataTree", node))

        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        loaded_root = manager._imagetool_wrappers[0]
        assert loaded_root._childtool_indices == [child_uid]

        loaded_child = manager.get_childtool(child_uid)
        assert isinstance(loaded_child, DerivativeTool)
        assert loaded_child.source_spec == expected_source_spec
        assert loaded_child.tool_status == expected_status
        xr.testing.assert_identical(loaded_child.result.T, expected_result)
        loaded_child_node = manager._child_node(child_uid)
        assert loaded_child_node._childtool_indices == [output_uid]
        loaded_output_node = manager._child_node(output_uid)
        assert loaded_output_node.output_id == "dtool.result"
        assert loaded_output_node.source_spec is None
        assert loaded_output_node.provenance_spec is not None
        assert loaded_output_node.provenance_spec.active_name == "result"
        xr.testing.assert_identical(fetch(output_uid), expected_result)

        monkeypatch.setattr(
            loaded_child, "_prompt_existing_output_imagetool", lambda: "update"
        )
        loaded_child.open_itool()
        assert loaded_child_node._childtool_indices == [output_uid]


def test_manager_workspace_roundtrip_fit1d_child(
    qtbot,
    monkeypatch,
    exp_decay_model,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()

        itool(test_data, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        t = np.linspace(0.0, 4.0, 25)
        data = xr.DataArray(
            3.0 * np.exp(-t / 2.0), dims=("t",), coords={"t": t}, name="decay"
        )
        params = exp_decay_model.make_params(n0=2.0, tau=1.0)
        child = erlab.interactive.ftool(
            data, model=exp_decay_model, params=params, execute=False
        )
        assert isinstance(child, Fit1DTool)
        child_uid = manager.add_childtool(child, 0, show=False)

        assert child._run_fit()
        qtbot.wait_until(lambda: child._last_result_ds is not None, timeout=10000)
        assert child._last_result_ds is not None
        expected_fit_ds = child._last_result_ds.copy(deep=True)
        expected_status = child.tool_status.model_dump()

        tree = manager._to_datatree()
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        for node in tree.values():
            manager._load_workspace_node(typing.cast("xr.DataTree", node))

        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        loaded_child = manager.get_childtool(child_uid)
        assert isinstance(loaded_child, Fit1DTool)
        assert loaded_child._last_result_ds is not None
        assert_fit_result_dataset_equivalent(
            loaded_child._last_result_ds, expected_fit_ds
        )
        assert loaded_child.tool_status.model_dump() == expected_status
        assert loaded_child._fit_is_current
        assert loaded_child.save_button.isEnabled()
        assert loaded_child.copy_button.isEnabled()

        warnings: list[tuple[str, str]] = []
        monkeypatch.setattr(
            loaded_child,
            "_show_warning",
            lambda title, text: warnings.append((title, text)),
        )
        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)
        copied = copy_full_code_for_uid(monkeypatch, manager, child_uid)
        assert "modelfit" in copied
        assert not warnings


def test_manager_workspace_roundtrip_fit2d_child(
    qtbot,
    monkeypatch,
    exp_decay_model,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()

        itool(test_data, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child_uid, child = make_fit2d_child(manager, 0, exp_decay_model)
        child.timeout_spin.setValue(30.0)
        child.nfev_spin.setValue(0)
        child.y_index_spin.setValue(child.y_min_spin.value())
        child._run_fit_2d("up")
        qtbot.wait_until(
            lambda: all(ds is not None for ds in child._result_ds_full),
            timeout=10000,
        )
        expected_results = [
            None if ds is None else ds.copy(deep=True) for ds in child._result_ds_full
        ]
        expected_status = child.tool_status.model_dump()

        tree = manager._to_datatree()
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        for node in tree.values():
            manager._load_workspace_node(typing.cast("xr.DataTree", node))

        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        loaded_child = manager.get_childtool(child_uid)
        assert isinstance(loaded_child, Fit2DTool)
        assert all(ds is not None for ds in loaded_child._result_ds_full)
        assert_fit_result_list_equivalent(
            loaded_child._result_ds_full, expected_results
        )
        assert loaded_child.tool_status.model_dump() == expected_status
        assert loaded_child._fit_is_current
        assert loaded_child.copy_full_button.isEnabled()
        assert loaded_child.save_full_button.isEnabled()

        warnings: list[tuple[str, str]] = []
        monkeypatch.setattr(
            loaded_child,
            "_show_warning",
            lambda title, text: warnings.append((title, text)),
        )
        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)
        copied = copy_full_code_for_uid(monkeypatch, manager, child_uid)
        assert "modelfit" in copied
        assert not warnings


def test_manager_workspace_roundtrip_fit2d_child_with_spaced_axis(
    qtbot,
    exp_decay_model,
    test_data,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()

        itool(test_data, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        t = np.linspace(0.0, 4.0, 25)
        motor = np.arange(3.0)
        data = xr.DataArray(
            np.stack(
                [((1.0 + 0.5 * idx) * np.exp(-t / 2.0)) for idx in motor],
                axis=0,
            ),
            dims=("Fake Motor", "t"),
            coords={
                "Fake Motor": motor,
                "t": t,
                "Sample Motor": ("Fake Motor", motor + 10.0),
            },
            name="decay2d",
        )
        params = exp_decay_model.make_params(n0=1.0, tau=1.0)
        child = erlab.interactive.ftool(
            data, model=exp_decay_model, params=params, execute=False
        )
        assert isinstance(child, Fit2DTool)
        child_uid = manager.add_childtool(child, 0, show=False)
        child.timeout_spin.setValue(30.0)
        child.nfev_spin.setValue(0)
        child.y_index_spin.setValue(child.y_min_spin.value())
        child._run_fit_2d("up")
        qtbot.wait_until(
            lambda: all(ds is not None for ds in child._result_ds_full),
            timeout=10000,
        )
        assert child.current_provenance_spec() is not None

        fname = tmp_path / "fit2d-spaced-axis.itws"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            manager._save_workspace_document(fname, force_full=True)

        assert not any("space in its name" in str(item.message) for item in caught)
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        loaded_child = manager.get_childtool(child_uid)
        assert isinstance(loaded_child, Fit2DTool)
        assert loaded_child.tool_data.dims[0] == "Fake Motor"
        xr.testing.assert_equal(
            loaded_child.tool_data.coords["Fake Motor"], data.coords["Fake Motor"]
        )
        xr.testing.assert_equal(
            loaded_child.tool_data.coords["Sample Motor"], data.coords["Sample Motor"]
        )
        assert loaded_child.current_provenance_spec() is not None


def test_manager_workspace_roundtrip_recursive_nested_imagetools(
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
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()

        itool(data, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        def _nest_average(dialog) -> None:
            dialog.dim_checks["x"].setChecked(True)
            set_transform_launch_mode(dialog, "nest")

        accept_dialog(manager.get_imagetool(0).mnb._average, pre_call=_nest_average)

        root_wrapper = manager._imagetool_wrappers[0]
        qtbot.wait_until(
            lambda: len(root_wrapper._childtool_indices) == 1, timeout=5000
        )
        child_uid = root_wrapper._childtool_indices[0]
        child_node = manager._child_node(child_uid)
        child_spec = child_node.source_spec

        child_tool = manager.get_imagetool(child_uid)
        child_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)
        tool_uid = child_node._childtool_indices[0]

        tree = manager._to_datatree()
        assert tree.attrs["imagetool_workspace_schema_version"] == 3
        assert (
            tree[f"0/childtools/{child_uid}/imagetool"].attrs["manager_node_uid"]
            == child_uid
        )
        assert (
            tree[f"0/childtools/{child_uid}/childtools/{tool_uid}/tool"].attrs[
                "manager_node_uid"
            ]
            == tool_uid
        )

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            filename = pathlib.Path(tmp_dir_name) / "workspace.itws"
            tree.to_netcdf(filename, engine="h5netcdf", invalid_netcdf=True)

            manager.remove_all_tools()
            qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

            loaded = manager_xarray.open_workspace_datatree(filename, chunks="auto")
            try:
                assert manager._is_datatree_workspace(loaded)
                assert loaded.attrs["imagetool_workspace_schema_version"] == 3
                for node in loaded.values():
                    manager._load_workspace_node(typing.cast("xr.DataTree", node))
            finally:
                loaded.close()

        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        loaded_root = manager._imagetool_wrappers[0]
        assert loaded_root._childtool_indices == [child_uid]

        loaded_child = manager._child_node(child_uid)
        assert loaded_child.source_spec == child_spec
        assert loaded_child._childtool_indices == [tool_uid]

        updated = data.copy(deep=True)
        updated.data = np.asarray(updated.data) * 4

        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated)

        qtbot.wait_until(lambda: loaded_child.source_state == "stale", timeout=5000)
        assert loaded_child._update_from_parent_source() is True
        xr.testing.assert_identical(
            manager.get_imagetool(child_uid).slicer_area._data.rename(None),
            updated.qsel.mean("x").rename(None),
        )


def test_manager_workspace_load_legacy(
    qtbot,
    accept_dialog,
    datadir,
    monkeypatch,
    tmp_path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    legacy_workspace = tmp_path / "manager_workspace_legacy.h5"
    legacy_workspace.write_bytes((datadir / "manager_workspace_legacy.h5").read_bytes())

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()

        def _go_to_file(dialog: QtWidgets.QFileDialog):
            dialog.setDirectory(str(tmp_path))
            dialog.selectFile(str(legacy_workspace))
            focused = dialog.focusWidget()
            if isinstance(focused, QtWidgets.QLineEdit):
                focused.setText("manager_workspace_legacy.h5")

        legacy_notices: list[pathlib.Path] = []
        save_dialog_calls: list[tuple[bool, str, pathlib.Path | None]] = []

        def _record_legacy_notice(fname: str | os.PathLike[str]) -> None:
            legacy_notices.append(pathlib.Path(fname))

        def _record_save_dialog(
            *,
            native: bool = True,
            caption: str = "Save Workspace",
            selected_file: str | os.PathLike[str] | None = None,
        ) -> str:
            selected_path = (
                None if selected_file is None else pathlib.Path(selected_file)
            )
            save_dialog_calls.append((native, caption, selected_path))
            return str(legacy_workspace)

        monkeypatch.setattr(
            manager, "_show_legacy_workspace_upgrade_message", _record_legacy_notice
        )
        monkeypatch.setattr(manager, "_workspace_save_dialog", _record_save_dialog)

        # Load workspace
        accept_dialog(lambda: manager.load(native=False), pre_call=_go_to_file)

        assert legacy_notices == [legacy_workspace]
        assert save_dialog_calls == [
            (False, "Save Converted Workspace", legacy_workspace)
        ]

        # Check if the data is loaded
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        # Check data
        xr.testing.assert_identical(
            manager.get_imagetool(0).slicer_area._data,
            test_data,
        )

        assert not manager.is_workspace_modified
        assert manager.save()
        import h5py

        with h5py.File(legacy_workspace, "r") as h5_file:
            assert h5_file.attrs["imagetool_workspace_schema_version"] == 4

        select_tools(manager, list(manager._imagetool_wrappers.keys()))
        accept_dialog(manager.remove_action.trigger)
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        assert manager._load_workspace_file(
            legacy_workspace,
            replace=True,
            associate=True,
            mark_dirty=False,
            select=False,
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
