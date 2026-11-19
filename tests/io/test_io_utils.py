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

    path = tmp_path / "spectra.nc"
    values = np.arange(48.0).reshape(6, 8)
    source = xr.Dataset(
        {"signal": (("x", "y"), values), "background": (("x", "y"), values + 10)},
        coords={"weights": (("x", "y"), values * 2)},
    )
    encoding = {"signal": {"chunksizes": (2, 4)}, "weights": {"chunksizes": (3, 2)}}
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
    arrays = [signal, same, other, group, root, child]
    assert signal.chunks == ((2, 2, 2), (4, 4))
    assert signal.weights.chunks == ((3, 3), (2, 2, 2, 2))
    assert signal.data.name == same.data.name
    assert len({a.data.name for a in arrays}) == 5

    reads = []
    backend_read = H5NetCDFArrayWrapper._getitem

    def count_read(self, key):
        value = backend_read(self, key)
        reads.append((self.variable_name, np.asarray(value).nbytes))
        return value

    monkeypatch.setattr(H5NetCDFArrayWrapper, "_getitem", count_read)
    cells = [a.data[1, 3] for a in arrays]
    assert dask.compute(*cells) == (11, 11, 21, 111, 11, 111)
    assert sorted(reads) == [("background", 8)] + [("signal", 8)] * 4
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
        {"chunks": {}},
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
