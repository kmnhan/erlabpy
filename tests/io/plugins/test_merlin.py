import pytest
import xarray as xr

import erlab


@pytest.fixture(scope="module")
def data_dir(test_data_dir):
    erlab.io.set_loader("merlin")
    erlab.io.set_data_dir(test_data_dir / "merlin")
    return test_data_dir / "merlin"


@pytest.fixture(scope="module")
def expected_dir(data_dir):
    return data_dir / "expected"


def test_load_xps(expected_dir) -> None:
    xr.testing.assert_identical(
        erlab.io.load("core.pxt"), xr.load_dataarray(expected_dir / "core.h5")
    )


def test_load_extend(expected_dir) -> None:
    with erlab.io.extend_loader(
        name_map={"ta_test": "Temperature Sensor A", "sc_test": "Sample Current"},
        coordinate_attrs=("Temperature Sensor A", "Sample Current"),
    ):
        dat = erlab.io.load(5)

    for new_coord in ("ta_test", "sc_test"):
        assert new_coord in dat.coords

    assert float(dat["ta_test"][0]) != float(dat["ta_test"][1])

    # Check if overridden attrs are restored
    xr.testing.assert_identical(
        erlab.io.load(5), xr.load_dataarray(expected_dir / "5.h5")
    )


def test_load_multiple(expected_dir) -> None:
    xr.testing.assert_identical(
        erlab.io.load("f_005_S001.pxt"), xr.load_dataarray(expected_dir / "5.h5")
    )
    xr.testing.assert_identical(
        erlab.io.load(5), xr.load_dataarray(expected_dir / "5.h5")
    )


def test_load_live(expected_dir) -> None:
    for live in ("lp", "lxy"):
        xr.testing.assert_identical(
            erlab.io.load(f"{live}.ibw"), xr.load_dataarray(expected_dir / f"{live}.h5")
        )


def test_corrupt(data_dir) -> None:
    with pytest.warns(
        UserWarning,
        match=r"Loading f_001_S001 with inferred index 1 resulted in an error[\s\S]*",
    ):
        erlab.io.load("f_001_S001.pxt")


def test_summarize(data_dir) -> None:
    with pytest.warns(
        UserWarning,
        match=r"Loading f_001_S001 with inferred index 1 resulted in an error[\s\S]*",
    ):
        erlab.io.summarize(cache=False)


def test_qinfo(data_dir) -> None:
    data = erlab.io.load(5)
    assert (
        data.qinfo.__repr__()
        == """time: 2022-03-27 07:53:26\ntype: map\nlens mode (Lens Mode): A30
mode (Acquisition Mode): Dither\ntemperature (sample_temp): 110.67
pass energy (Pass Energy): 10\nanalyzer slit (Slit Plate): 7\npol (polarization): LH
hv (hv): 100\nentrance slit (Entrance Slit): 70\nexit slit (Exit Slit): 70
polar (beta): [-15.5, -15]\ntilt (xi): 0\nazi (delta): 3\nx (x): 2.487\ny (y): 0.578
z (z): -1.12"""
    )
