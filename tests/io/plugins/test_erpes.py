import pytest
import xarray as xr

import erlab
from erlab.io.plugins.da30 import DA30Loader
from erlab.io.plugins.erpes import ERPESLoader, get_cache_file


@pytest.fixture(scope="module")
def data_dir(test_data_dir):
    erlab.io.set_loader("erpes")
    erlab.io.set_data_dir(test_data_dir / "erpes")
    return test_data_dir / "erpes"


@pytest.fixture(scope="module")
def expected_dir(data_dir):
    return data_dir / "expected"


@pytest.mark.parametrize(
    ("arg", "index"),
    [
        (1, 1),  # Single cut
        (2, 2),  # Multi-region cut (DataTree)
        (3, 3),  # DA map (DataTree)
        (4, 4),  # Single motor cut
        (5, 5),  # Single motor DA map
        (6, 6),  # Single motor DA map with first slice missing coords
        ("test0002.pxt", 2),
        ("test0004_S00001.pxt", 4),
        ("test0005_S00002.zip", 5),
    ],
)
def test_load(expected_dir, arg, index) -> None:
    loaded = erlab.io.load(arg)

    expected_file = expected_dir / f"{index}.h5"

    if index in {2, 3}:
        assert isinstance(loaded, xr.DataTree)
        expected_data = xr.load_datatree(expected_file, engine="h5netcdf")
        xr.testing.assert_equal(loaded, expected_data)
        xr.testing.assert_isomorphic(loaded, expected_data)
    else:
        assert isinstance(loaded, xr.DataArray)
        xr.testing.assert_identical(
            loaded, xr.load_dataarray(expected_file, engine="h5netcdf")
        )


def test_summarize(data_dir) -> None:
    erlab.io.summarize(cache=False)


def test_load_single_use_cache_false_ignores_existing_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    file_path = tmp_path / "test0001.zip"
    file_path.touch()

    cache_file = get_cache_file(file_path)
    cache_file.parent.mkdir()
    cache_file.write_text("invalid cache contents")

    expected = xr.DataArray([1.0, 2.0], dims=("eV",), name="spectrum")
    calls: list[str] = []

    def fake_super_load(self, file_path, *, without_values=False):
        calls.append("super")
        return expected.copy()

    def fail_open_datatree(*args, **kwargs):
        raise AssertionError("existing cache should not be opened when use_cache=False")

    def fail_to_netcdf(self, *args, **kwargs):
        raise AssertionError("cache should not be written when use_cache=False")

    monkeypatch.setattr(DA30Loader, "load_single", fake_super_load)
    monkeypatch.setattr(xr, "open_datatree", fail_open_datatree)
    monkeypatch.setattr(xr.DataArray, "to_netcdf", fail_to_netcdf)

    loaded = ERPESLoader().load_single(file_path, use_cache=False)

    assert calls == ["super"]
    xr.testing.assert_identical(loaded.compute(), expected)
    assert cache_file.read_text() == "invalid cache contents"


def test_load_single_use_cache_false_does_not_write_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    file_path = tmp_path / "test0001.zip"
    file_path.touch()

    cache_file = get_cache_file(file_path)
    expected = xr.DataArray([1.0, 2.0], dims=("eV",), name="spectrum")
    calls: list[str] = []

    def fake_super_load(self, file_path, *, without_values=False):
        calls.append("super")
        return expected.copy()

    def fail_to_netcdf(self, *args, **kwargs):
        raise AssertionError("cache should not be written when use_cache=False")

    monkeypatch.setattr(DA30Loader, "load_single", fake_super_load)
    monkeypatch.setattr(xr.DataArray, "to_netcdf", fail_to_netcdf)

    loaded = ERPESLoader().load_single(file_path, use_cache=False)

    assert calls == ["super"]
    xr.testing.assert_identical(loaded.compute(), expected)
    assert not cache_file.exists()
