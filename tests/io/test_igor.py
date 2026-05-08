import numpy as np
import pytest
import xarray as xr
import xarray.testing

from erlab.io.igor import _parse_wave_shape, load_text, load_wave, save_wave, set_scale

WAVE0 = xr.DataArray(
    np.linspace(-0.3, 0.7, 6, dtype=np.float32),
    coords={"W": np.arange(6, dtype=np.float64)},
)
WAVE1 = xr.DataArray(
    np.linspace(-0.5, 1.5, 6, dtype=np.float32),
    coords={"W": np.arange(6, dtype=np.float64)},
)


def test_backend_dataarray(datadir) -> None:
    xarray.testing.assert_equal(xr.load_dataarray(datadir / "wave0.ibw"), WAVE0)


def test_backend_dataarray_itx(datadir) -> None:
    wave = xr.load_dataarray(datadir / "wave0.itx").rename(x="W").astype(np.float32)
    xarray.testing.assert_equal(wave, WAVE0)


def test_backend_dataset(datadir) -> None:
    xarray.testing.assert_equal(xr.load_dataset(datadir / "exp0.pxt")["wave0"], WAVE0)


def test_backend_datatree(datadir) -> None:
    dt = xr.open_datatree(datadir / "exp1.pxt")
    assert dt.groups == ("/", "/wave0", "/testfolder", "/testfolder/wave1")

    xr.testing.assert_equal(dt["wave0/wave0"], WAVE0)
    xr.testing.assert_equal(dt["testfolder/wave1/wave1"], WAVE1)


def test_load_wave(datadir) -> None:
    xarray.testing.assert_equal(load_wave(datadir / "wave0.ibw"), WAVE0)


def test_set_scale_noargs() -> None:
    data = xr.DataArray(np.arange(5))
    scaled = set_scale(data, None, "x", 0, 5, units_str="new_name")
    np.testing.assert_allclose(scaled["new_name"], np.linspace(0, 4, 5))


def test_set_scale_endpoints() -> None:
    data = xr.DataArray(np.arange(5), dims=["x"])
    scaled = set_scale(data, "/I", "x", 0, 1)
    np.testing.assert_allclose(scaled["x"], np.linspace(0, 1, 5))


def test_set_scale_step() -> None:
    data = xr.DataArray(np.arange(5), dims=["x"])
    scaled = set_scale(data, "/P", "x", 0, 2)
    np.testing.assert_allclose(scaled["x"], np.linspace(0, 8, 5))


def test_set_scale_invalid_dimension() -> None:
    data = xr.DataArray(np.arange(5), dims=["x"])
    with pytest.raises(ValueError, match="Invalid dimension"):
        set_scale(data, None, "invalid", 0, 4)


def test_parse_wave_shape() -> None:
    wave_str = r"WAVES/S/N=(100,200) 'wave_name'"
    shape, name = _parse_wave_shape(wave_str)
    assert shape == (100, 200)
    assert name == "wave_name"

    wave_str_invalid = r"non-matching string"
    with pytest.raises(ValueError, match="Invalid format"):
        _parse_wave_shape(wave_str_invalid)


def test_load_text_basic(datadir) -> None:
    wave = load_text(datadir / "wave0.itx", dtype=np.float32).rename(x="W")
    xarray.testing.assert_equal(wave, WAVE0)

    wave = load_text(
        datadir / "wave0.itx", dtype=np.float32, without_values=True
    ).rename(x="W")
    xarray.testing.assert_equal(wave, xr.zeros_like(WAVE0))


def test_load_text_invalid_file(datadir) -> None:
    with pytest.raises(ValueError, match="No valid wave definition found"):
        load_text(datadir / "invalid.itx")


def test_load_text_multiple_waves(datadir) -> None:
    with pytest.warns(UserWarning, match="Multiple wave definitions found"):
        wave = load_text(datadir / "multiple_waves.itx", dtype=np.float32).rename(x="W")

    xarray.testing.assert_equal(wave, WAVE0)


def test_load_text_empty_unit_setscale_and_notes(tmp_path) -> None:
    path = tmp_path / "note_wave.itx"
    path.write_text(
        """IGOR
WAVES/D/N=(2,3,2) testwave
BEGIN
1 2 3
4 5 6
7 8 9
10 11 12
END
X SetScale/P x 10,0.5,"", testwave
X SetScale/P y -1,1,"", testwave
X SetScale/P z 4,2,"", testwave
X Note testwave, "Photon Energy = 80.5 eV"
X Note testwave, "Acquired from 2025-10-30 14:30:28 To 2025-10-30 14:41:39"
X Note testwave, "Category: sample"
X Note testwave, "Compact:sample"
""",
        encoding="utf-8",
    )

    wave = load_text(path)

    assert wave.dims == ("x", "y", "z")
    assert wave.shape == (2, 3, 2)
    np.testing.assert_allclose(wave["x"], [10.0, 10.5])
    np.testing.assert_allclose(wave["y"], [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(wave["z"], [4.0, 6.0])
    assert wave.attrs["Photon Energy"] == "80.5 eV"
    assert wave.attrs["Category"] == "sample"
    assert wave.attrs["Compact"] == "sample"
    assert not any(key.startswith("Acquired from") for key in wave.attrs)
    assert wave.attrs["IGORWaveNote"] == (
        "Photon Energy = 80.5 eV\n"
        "Acquired from 2025-10-30 14:30:28 To 2025-10-30 14:41:39\n"
        "Category: sample\n"
        "Compact:sample"
    )

    wave_without_values = load_text(path, without_values=True)

    assert wave_without_values.dims == wave.dims
    assert wave_without_values.shape == wave.shape
    for dim in wave.dims:
        np.testing.assert_allclose(wave_without_values[dim], wave[dim])
    assert wave_without_values.attrs == wave.attrs
    np.testing.assert_array_equal(wave_without_values.values, np.zeros(wave.shape))


@pytest.mark.parametrize(
    ("data", "dims", "coords", "name"),
    [
        (np.arange(5, dtype=np.float32), ["x"], {"x": np.linspace(0, 4, 5)}, "wave0"),
        (
            np.arange(6, dtype=np.float64).reshape(2, 3),
            ["x", "y"],
            {"x": np.linspace(0, 1, 2), "y": np.linspace(0, 2, 3)},
            "mywave",
        ),
    ],
)
def test_save_wave_roundtrip(tmp_path, data, dims, coords, name):
    arr = xr.DataArray(data, dims=dims, coords=coords, name=name)
    path = tmp_path / "test.ibw"
    save_wave(arr, path)
    loaded = load_wave(path)
    xr.testing.assert_allclose(loaded, arr, atol=1e-6)


def test_save_wave_non_uniform_coord(tmp_path):
    arr = xr.DataArray(
        np.arange(5).astype(np.float64), dims=["x"], coords={"x": [0, 1, 3, 6, 10]}
    )
    path = tmp_path / "fail.ibw"
    with pytest.warns(UserWarning, match="not evenly spaced"):
        save_wave(arr, path)
    loaded = load_wave(path)
    assert loaded.dims == ("x_idx",)
    np.testing.assert_allclose(loaded["x_idx"], np.arange(5))


def test_save_wave_extra_coord_warns(tmp_path):
    arr = xr.DataArray(
        np.arange(5).astype(np.float64),
        dims=["x"],
        coords={"x": np.arange(5), "foo": 3},
    )
    path = tmp_path / "warn.ibw"
    with pytest.warns(UserWarning, match="not dimension scales"):
        save_wave(arr, path)


def test_save_wave_attrs_roundtrip(tmp_path):
    arr = xr.DataArray(
        np.arange(3).astype(np.float64),
        dims=["x"],
        coords={"x": np.arange(3)},
        name="attrwave",
        attrs={"foo": 42, "bar": "baz"},
    )
    path = tmp_path / "attr.ibw"
    save_wave(arr, path)
    loaded = load_wave(path)
    assert loaded.attrs["foo"] == 42
    assert loaded.attrs["bar"] == "baz"
