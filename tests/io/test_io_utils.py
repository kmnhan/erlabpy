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
