from dataclasses import asdict

import numpy as np
import pytest
import xarray as xr

import erlab
from erlab._provenance import FileVersion
from erlab.io.utils import save_as_hdf5


def test_save_as_hdf5_squeezes_size_one_dims_with_igor_compat(tmp_path) -> None:
    data = xr.DataArray(np.array([1.0]), dims=["x"], coords={"x": [0.0]})
    out = tmp_path / "single.h5"

    with pytest.warns(UserWarning, match="Dimensions with length 1 were squeezed"):
        save_as_hdf5(data, out, igor_compat=True)

    loaded = xr.load_dataarray(out, engine="h5netcdf")
    assert loaded.ndim == 0
    assert "IGORWaveScaling" in loaded.attrs


def test_file_replaced_during_versioned_read_closes_result(tmp_path) -> None:
    path = tmp_path / "scan.nc"
    original = xr.DataArray([1, 2], dims="x")
    original.to_netcdf(path, engine="h5netcdf")
    version = FileVersion.capture(path)
    changed = tmp_path / "next.nc"
    (original + 100).to_netcdf(changed, engine="h5netcdf")
    closed = []

    def read(filename):
        array = xr.load_dataarray(filename, engine="h5netcdf")
        array.set_close(lambda: closed.append(True))
        changed.replace(path)
        return array

    with pytest.raises(OSError, match=r"has changed.*checkpoint"):
        erlab.io.read_file_version(read=read, **asdict(version))
    assert closed == [True]


@pytest.mark.parametrize("selection_fails", [False, True])
def test_versioned_tree_selection_releases_file_owners(tmp_path, selection_fails):
    import gc
    import weakref

    path = tmp_path / "tree.nc"
    path.touch()
    closed = []
    refs = []

    def read(_filename):
        unrelated = np.arange(1000)
        tree = xr.DataTree.from_dict(
            {
                "/": xr.Dataset({"unrelated": ("other", unrelated)}),
                "/scan": xr.Dataset({"counts": ("x", np.array([1, 3]))}),
            }
        )
        refs.append(weakref.ref(tree.dataset["unrelated"].data))
        tree.set_close(lambda: closed.append("root"))
        tree["scan"].set_close(lambda: closed.append("scan"))
        return tree

    def selected(tree):
        return tree["scan"].dataset["absent" if selection_fails else "counts"]

    kwargs = dict(
        read=read, select=selected, dtype="float64", **asdict(FileVersion.capture(path))
    )
    if selection_fails:
        with pytest.raises(KeyError):
            erlab.io.read_file_version(**kwargs)
    else:
        result = erlab.io.read_file_version(**kwargs)
        np.testing.assert_array_equal(result.values, [1.0, 3.0])
        gc.collect()
        assert refs[0]() is None
        assert closed == []
        result.close()
        result.close()
    assert sorted(closed) == ["root", "scan"]


def test_versioned_sequence_closes_repeated_owner_once(tmp_path):
    path = tmp_path / "scan.bin"
    path.touch()
    first, second = xr.DataArray([1]), xr.DataArray([2])
    closed = []
    first.set_close(lambda: closed.append("first"))
    second.set_close(lambda: closed.append("second"))
    result = erlab.io.read_file_version(
        **asdict(FileVersion.capture(path)),
        read=lambda _: (first, first, second),
        select=lambda arrays: xr.DataArray(arrays[2]),
    )
    assert result.item() == 2
    result.close()
    first.close()
    second.close()
    assert sorted(closed) == ["first", "second"]


def test_versioned_tree_file_can_reopen_after_pickle(tmp_path):
    import pickle

    import psutil

    path = tmp_path / "tree.nc"
    xr.DataTree.from_dict(
        {
            "/": xr.Dataset({"a": ("x", [1, 2])}),
            "/scan": xr.Dataset({"b": ("y", [3, 4])}),
        }
    ).to_netcdf(path, engine="h5netcdf")
    with erlab.io.read_file_version(
        **asdict(FileVersion.capture(path)),
        read=lambda filename: xr.open_datatree(filename, engine="h5netcdf", chunks={}),
        select=lambda tree: tree["scan"].dataset["b"],
    ) as selected:
        payload = pickle.dumps(selected)
    with pickle.loads(payload) as reopened:
        assert reopened.isel(y=1).compute().item() == 4
    assert str(path.resolve()) not in {f.path for f in psutil.Process().open_files()}


@pytest.mark.parametrize("key", [None, ("counts", "cleaned")])
def test_named_file_variable_preserves_hashable_keys(tmp_path, key) -> None:
    path = tmp_path / "named.dat"
    path.touch()
    data = xr.Dataset({None: ("x", [1, 2]), ("counts", "cleaned"): ("x", [4, 6])})
    options = asdict(FileVersion.capture(path))
    with erlab.io.read_file_version(
        read=lambda _: data, variable=key, **options
    ) as selected:
        xr.testing.assert_identical(selected, data[key])
    with pytest.raises(TypeError, match="select a DataArray"):
        erlab.io.read_file_version(read=lambda _: data, **options)


def test_versioned_named_selection_rejects_incompatible_options(tmp_path):
    path = tmp_path / "selection.dat"
    path.touch()
    options = asdict(FileVersion.capture(path))
    with pytest.raises(ValueError, match="either select"):
        erlab.io.read_file_version(
            **options,
            read=lambda _: xr.Dataset(),
            select=lambda ds: ds[None],
            variable=None,
        )
    with pytest.raises(ValueError, match="requires a variable"):
        erlab.io.read_file_version(
            **options, read=lambda _: xr.DataTree(), node_path="/scan"
        )
    with pytest.raises(TypeError, match="requires a DataTree"):
        erlab.io.read_file_version(
            **options, read=lambda _: xr.Dataset(), variable="x", node_path="/"
        )
    with pytest.raises(TypeError, match="requires a Dataset"):
        erlab.io.read_file_version(
            **options, read=lambda _: xr.DataArray([1]), variable=None
        )


def test_storage_chunk_file_checks_preserve_slices_and_shared_reads(
    tmp_path, monkeypatch
):
    import dask
    import psutil
    from xarray.backends.h5netcdf_ import H5NetCDFArrayWrapper

    from erlab._provenance._checkpoint import _read_versioned_file

    path = tmp_path / "spectra.nc"
    values = np.arange(48.0).reshape(6, 8)
    source = xr.Dataset(
        {"signal": (("x", "y"), values), "background": (("x", "y"), values + 10)},
        coords={"weights": (("x", "y"), values * 2)},
    )
    encoding = {
        "signal": {"chunksizes": (2, 4), "add_offset": 2.0},
        "weights": {"chunksizes": (3, 2)},
    }
    source.to_netcdf(path, engine="h5netcdf", encoding=encoding)
    (source + 100).to_netcdf(
        path, engine="h5netcdf", group="scan", mode="a", encoding=encoding
    )
    version = asdict(FileVersion.capture(path))
    loader_args = {"engine": "h5netcdf", "chunks": {}}

    def read(loader, variable, *, node_path=None, **kwargs):
        return erlab.io.read_file_version(
            **version,
            read=loader,
            read_kwargs=loader_args | kwargs,
            variable=variable,
            node_path=node_path,
        )

    signal = read(xr.open_dataset, "signal")
    same = read(xr.open_dataset, "signal")
    other = read(xr.open_dataset, "background")
    group = read(xr.open_dataset, "signal", group="scan")
    root = read(xr.open_datatree, "signal", node_path="/")
    child = read(xr.open_datatree, "signal", node_path="/scan")
    tree = _read_versioned_file(
        xr.open_datatree, path, loader_args, FileVersion.capture(path)
    )
    live_root, live_child = (node.dataset["signal"] for node in tree.subtree)
    raw = read(xr.open_dataset, "signal", mask_and_scale=False)
    arrays = [signal, same, other, group, root, child, live_root, live_child, raw]
    assert signal.chunks == ((2, 2, 2), (4, 4))
    assert signal.weights.chunks == ((3, 3), (2, 2, 2, 2))
    assert signal.data.name == same.data.name
    assert len({a.data.name for a in arrays}) == 6
    assert live_root.data.name == root.data.name
    assert live_child.data.name == child.data.name

    reads = []
    backend_read = H5NetCDFArrayWrapper._getitem

    def count_read(self, key):
        value = backend_read(self, key)
        reads.append((self.variable_name, np.asarray(value).nbytes))
        return value

    monkeypatch.setattr(H5NetCDFArrayWrapper, "_getitem", count_read)
    cells = [a.data[1, 3] for a in arrays]
    assert dask.compute(*cells) == (11, 11, 21, 111, 11, 111, 11, 111, 9)
    assert sorted(reads) == [("background", 8)] + [("signal", 8)] * 5
    tree.close()
    for array in arrays:
        array.close()
    assert str(path.resolve()) not in {f.path for f in psutil.Process().open_files()}

    replaced = tmp_path / "replacement.nc"
    (source + 300).to_netcdf(replaced, engine="h5netcdf")
    replaced.replace(path)
    reads.clear()
    with pytest.raises(OSError, match="has changed"):
        cells[0].compute()
    assert reads == []


@pytest.mark.parametrize(
    "options",
    [
        {"chunks": {"x": 2}},
        {"engine": "h5netcdf", "chunks": {"x": 2}},
        {"engine": "h5netcdf", "chunks": {}, "inline_array": True},
    ],
)
def test_versioned_reader_forwards_other_chunk_settings(tmp_path, monkeypatch, options):
    path = tmp_path / "options.nc"
    data = xr.DataArray(np.arange(8), dims="x", name="spectrum")
    data.to_netcdf(path, engine="h5netcdf", encoding={"spectrum": {"chunksizes": (2,)}})
    native = xr.open_dataarray
    calls = []

    def open_array(path, **kwargs):
        calls.append(kwargs)
        return native(path, **kwargs)

    monkeypatch.setattr(xr, "open_dataarray", open_array)
    with erlab.io.read_file_version(
        read=xr.open_dataarray,
        read_kwargs=options,
        **asdict(FileVersion.capture(path)),
    ) as selected:
        assert calls == [options]
        assert selected.chunks == ((2, 2, 2, 2),)
        xr.testing.assert_identical(selected.compute(), data)


def test_versioned_file_default_uses_actual_engine_in_task_keys(tmp_path, monkeypatch):
    from erlab._provenance._checkpoint import _native_file_options

    path = tmp_path / "data.nc"
    path.touch()
    backing = FileVersion.capture(path)
    current = "h5netcdf"
    guessed = []

    def choose(filename):
        guessed.append(filename)
        return current

    monkeypatch.setattr(xr.backends.plugins, "guess_engine", choose)
    args, first = _native_file_options(xr.open_dataset, {"chunks": {}}, backing)
    assert args == {"engine": "h5netcdf", "chunks": None, "cache": False}
    current = "netcdf4"
    args, second = _native_file_options(xr.open_dataset, {"chunks": {}}, backing)
    assert args["engine"] == current
    assert first != second
    current = "custom"
    args, third = _native_file_options(xr.open_dataset, {"chunks": {}}, backing)
    assert third is None
    assert args == {"engine": "custom", "chunks": {}}
    assert guessed == [str(path)] * 3


@pytest.mark.parametrize("chunks", [{}, "auto"])
def test_versioned_tree_uses_default_engine_without_aliasing_groups(tmp_path, chunks):
    import dask

    from erlab._provenance._checkpoint import _read_versioned_file

    path = tmp_path / "tree.nc"
    ds = xr.Dataset({"signal": ("x", [3, 7])})
    ds.to_netcdf(path, engine="h5netcdf")
    (ds + 100).to_netcdf(path, engine="h5netcdf", group="scan", mode="a")
    backing = FileVersion.capture(path)
    options = {"chunks": chunks}
    with (
        _read_versioned_file(xr.open_datatree, path, options, backing) as live,
        erlab.io.read_file_version(
            read=xr.open_datatree,
            read_kwargs=options,
            node_path="/scan",
            variable="signal",
            **asdict(backing),
        ) as copied,
    ):
        root, child = (node.dataset["signal"] for node in live.subtree)
        assert child.data.name == copied.data.name
        assert dask.compute(*(a.data[0] for a in (root, child, copied))) == (
            3,
            103,
            103,
        )


def test_automatic_file_chunks_match_native_layout_and_task_identity(
    tmp_path, monkeypatch
):
    import dask
    from xarray.backends.h5netcdf_ import H5NetCDFArrayWrapper

    from erlab._provenance._checkpoint import _read_versioned_file

    path = tmp_path / "automatic.h5"
    values = np.arange(128 * 192, dtype="f8").reshape(128, 192)
    grid = xr.DataArray(
        values,
        dims=("x", "y"),
        coords={"weights": (("x", "y"), values + 2)},
        name="grid",
    )
    grid.to_netcdf(
        path,
        engine="h5netcdf",
        encoding={"grid": {"chunksizes": (4, 8)}, "weights": {"chunksizes": (8, 2)}},
    )
    version = FileVersion.capture(path)
    options = {"chunks": "auto", "engine": "h5netcdf"}
    reads = []
    read = H5NetCDFArrayWrapper._getitem

    def track(self, key):
        result = read(self, key)
        reads.append((self.variable_name, result.size))
        return result

    monkeypatch.setattr(H5NetCDFArrayWrapper, "_getitem", track)
    layouts, names = [], []
    for limit in ("2 KiB", "32 KiB"):
        with (
            dask.config.set({"array.chunk-size": limit}),
            xr.open_dataarray(path, **options) as native,
            _read_versioned_file(xr.open_dataarray, path, options, version) as live,
            erlab.io.read_file_version(
                read=xr.open_dataarray, read_kwargs=options, **asdict(version)
            ) as copied,
        ):
            assert live.chunks == copied.chunks == native.chunks
            assert live.weights.chunks == copied.weights.chunks == native.weights.chunks
            layouts.append((live.chunks, live.weights.chunks))
            names.append((live.data.name, live.weights.data.name))
            reads.clear()
            values = dask.compute(
                live.data[15, 3], copied.data[15, 3], copied.weights.data[15, 3]
            )
            assert values == (2883, 2883, 2885)
            assert sorted(reads) == [("grid", 1), ("weights", 1)]
    assert layouts[0][0] != layouts[1][0]
    assert layouts[0][1] != layouts[1][1]
    assert names[0][0] != names[1][0]
    assert names[0][1] != names[1][1]


def test_automatic_file_chunks_open_mixed_dataset_once(tmp_path, monkeypatch):
    from erlab._provenance._checkpoint import _read_versioned_file

    path = tmp_path / "mixed.h5"
    ds = xr.Dataset({"counts": ("x", [1, 2]), "labels": ("x", ["small", "large"])})
    ds.to_netcdf(path, engine="h5netcdf")
    options = {"engine": "h5netcdf", "chunks": "auto"}
    version = FileVersion.capture(path)
    with xr.open_dataset(path, **options) as expected:
        read = xr.open_dataset
        opened = []

        def record(filename, **kwargs):
            opened.append(filename)
            return read(filename, **kwargs)

        monkeypatch.setattr(xr, "open_dataset", record)
        with _read_versioned_file(xr.open_dataset, path, options, version) as source:
            assert opened == [str(path)]
            assert source.counts.chunks == expected.counts.chunks
            assert source.labels.chunks == expected.labels.chunks
            xr.testing.assert_identical(source.compute(), expected.compute())


def test_automatic_file_chunks_use_native_calendar_handling(tmp_path):
    import re

    from erlab._provenance._checkpoint import _read_versioned_file

    pytest.importorskip("cftime")
    path = tmp_path / "calendar.nc"
    data = xr.Dataset({"time": ("x", np.arange(12)), "counts": ("x", np.arange(12.0))})
    data.time.attrs.update(units="days since 2000-01-01", calendar="360_day")
    data.to_netcdf(path, engine="h5netcdf")
    version = FileVersion.capture(path)
    options = {"engine": "h5netcdf", "chunks": "auto"}
    try:
        native = xr.open_dataset(path, **options)
    except NotImplementedError as exc:
        # Old Dask does not estimate automatic chunk sizes for object arrays.
        with pytest.raises(NotImplementedError, match=re.escape(str(exc))):
            _read_versioned_file(xr.open_dataset, path, options, version)
        return
    with native, _read_versioned_file(xr.open_dataset, path, options, version) as live:
        assert live.time.chunks == native.time.chunks
        xr.testing.assert_identical(live.compute(), native.compute())
