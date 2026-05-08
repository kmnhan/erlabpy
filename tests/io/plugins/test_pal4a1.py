import pathlib

import numpy as np
import pytest
import xarray as xr

import erlab


@pytest.fixture(scope="module")
def data_dir(test_data_dir):
    erlab.io.set_loader("pal4a1")
    erlab.io.set_data_dir(test_data_dir / "pal4a1")
    return test_data_dir / "pal4a1"


@pytest.fixture(scope="module")
def expected_dir(data_dir):
    return data_dir / "expected"


@pytest.mark.parametrize(
    "filename", ["core_level.itx", "fine_cut.itx", "rough_map.itx"]
)
def test_load(filename, expected_dir) -> None:
    xr.testing.assert_identical(
        erlab.io.load(filename),
        xr.load_dataarray(expected_dir / f"{pathlib.Path(filename).stem}.h5"),
    )


def test_core_level_coordinates(data_dir) -> None:
    data = erlab.io.load("core_level.itx")

    assert data.dims == ("eV",)
    np.testing.assert_allclose(data["eV"], [10.0, 10.02, 10.04, 10.06])
    assert float(data["alpha"]) == pytest.approx(-3.0)
    assert float(data["beta"]) == pytest.approx(1.0)
    assert float(data["hv"]) == pytest.approx(80.5)
    assert float(data["x"]) == pytest.approx(1.8701)
    assert float(data["y"]) == pytest.approx(5.9799)
    assert float(data["z"]) == pytest.approx(18.92)
    assert data.attrs["configuration"] == 1
    assert data.attrs["Pass Energy"] == 10
    assert data.attrs["I0"] == "25.913 nA ... 25.763 nA"


def test_fine_cut_coordinates(data_dir) -> None:
    data = erlab.io.load("fine_cut.itx")

    assert data.dims == ("eV", "alpha")
    np.testing.assert_allclose(data["eV"], [62.0, 62.005, 62.01, 62.015])
    np.testing.assert_allclose(data["alpha"], [-16.6, -16.5534286, -16.5068572])
    assert float(data["beta"]) == pytest.approx(1.0)
    assert float(data["hv"]) == pytest.approx(80.5)
    assert float(data["x"]) == pytest.approx(1.8701)
    assert float(data["y"]) == pytest.approx(5.9799)
    assert float(data["z"]) == pytest.approx(18.92)
    assert data.attrs["configuration"] == 1
    assert data.attrs["Pass Energy"] == 50


def test_rough_map_coordinates(data_dir) -> None:
    data = erlab.io.load("rough_map.itx")

    assert data.dims == ("eV", "alpha", "beta")
    np.testing.assert_allclose(data["eV"], [63.0, 63.1])
    np.testing.assert_allclose(data["alpha"], [-13.8, -13.7534286, -13.7068572])
    np.testing.assert_allclose(data["beta"], [-19.0, -18.0])
    assert float(data["hv"]) == pytest.approx(80.5)
    assert float(data["x"]) == pytest.approx(1.8701)
    assert float(data["y"]) == pytest.approx(5.9799)
    assert float(data["z"]) == pytest.approx(18.94)
    assert data.attrs["configuration"] == 1
    assert data.attrs["Pass Energy"] == 50


def test_without_values(data_dir) -> None:
    data = erlab.io.load("rough_map.itx", load_kwargs={"without_values": True})

    assert data.dims == ("eV", "alpha", "beta")
    assert data.shape == (2, 3, 2)
    np.testing.assert_array_equal(data.values, np.zeros(data.shape))
    np.testing.assert_allclose(data["eV"], [63.0, 63.1])
    np.testing.assert_allclose(data["beta"], [-19.0, -18.0])
    assert float(data["hv"]) == pytest.approx(80.5)
    assert data.attrs["configuration"] == 1


def test_integer_loading_raises(data_dir) -> None:
    with pytest.raises(
        NotImplementedError,
        match="does not support loading data by scan index",
    ):
        erlab.io.load(1)
