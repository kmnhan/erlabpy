import json
import shutil

import numpy as np
import pytest
import xarray as xr

import erlab
from erlab.io.plugins.merlin import load_bcs


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


def test_load_multiregion(data_dir, expected_dir) -> None:
    expected = xr.load_datatree(expected_dir / "8.h5")

    xr.testing.assert_identical(erlab.io.load("f_008_R0_S001.pxt"), expected)
    xr.testing.assert_identical(erlab.io.load(8), expected)

    shutil.copy(data_dir / "f_008_R0_S001.pxt", data_dir / "f_009_R0.pxt")
    shutil.copy(data_dir / "f_008_R1_S001.pxt", data_dir / "f_009_R1.pxt")


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


def _write_bcs_scan(
    tmp_path,
    *,
    image_columns: tuple[str, ...] = ("DiagOn YAG",),
) -> tuple:
    image_module = pytest.importorskip("PIL.Image")
    scan_path = tmp_path / "Single Motor Scan 000001.txt"
    image_dir = tmp_path / "Single Motor Scan 000001 Images"
    image_dir.mkdir()

    header = {
        "Scan Type": "Single Motor Scan",
        "Version": "4",
        "General": {"Date": "Apr 22 2026, 02:44:50.789 -07:00:00"},
        "Single Motor Scan": {"X Motor": "EPU Gap", "Start": 20.0, "Stop": 20.2},
        "Motors": {
            "EPU Gap": 999.0,
            "Beamline Energy": 30.0,
            "DiagOn Z": -48.0,
        },
        "Notes": {"operator": "test"},
    }
    columns = [
        "Time (s)",
        "EPU Gap Goal",
        "EPU Gap Actual",
        "EPU Gap",
        "Beam Current",
        "Const Trace",
        *image_columns,
    ]
    rows = []
    expected_images = {}
    for i in range(2):
        row = [
            f"{float(i):.1f}",
            f"{20.0 + 0.2 * i:.1f}",
            f"{20.01 + 0.2 * i:.2f}",
            "0.0",
            f"{500.0 + i:.1f}",
            "7.0",
        ]
        for image_column in image_columns:
            values = np.arange(6, dtype=np.uint16).reshape(2, 3) + i * 10
            if len(image_columns) > 1:
                values = values + 100 * image_columns.index(image_column)
            image_name = f"Single Motor Scan 000001 {image_column} {i:03d}.png"
            image_module.fromarray(values).save(image_dir / image_name)
            row.append(rf"..\Single Motor Scan 000001 Images\{image_name}")
            expected_images.setdefault(image_column, []).append(values)
        rows.append(row)

    data_table = "\t".join(columns) + "\n" + "\n".join("\t".join(row) for row in rows)
    scan_path.write_text(
        f"HEADER\n{json.dumps(header, indent=4)}\nDATA\n{data_table}",
        encoding="utf-8",
    )
    return scan_path, expected_images


def test_load_bcs_dataarray(tmp_path) -> None:
    scan_path, expected_images = _write_bcs_scan(tmp_path)

    data = load_bcs(scan_path)

    assert isinstance(data, xr.DataArray)
    assert data.dims == ("EPU Gap", "y", "x")
    assert data.dtype == np.uint16
    np.testing.assert_array_equal(data.values, np.stack(expected_images["DiagOn YAG"]))
    np.testing.assert_allclose(data["EPU Gap"].values, [20.0, 20.2])
    np.testing.assert_allclose(data["EPU Gap Actual"].values, [20.01, 20.21])
    np.testing.assert_allclose(data["Beam Current"].values, [500.0, 501.0])

    assert data["Const Trace"].dims == ()
    assert float(data["Const Trace"]) == 7.0
    assert data["Beamline Energy"].dims == ()
    assert float(data["Beamline Energy"]) == 30.0
    assert data["DiagOn Z"].dims == ()
    assert float(data["DiagOn Z"]) == -48.0
    assert "EPU Gap raw" in data.coords
    assert "General" not in data.attrs
    assert "Motors" not in data.attrs
    assert data.attrs["Scan Type"] == "Single Motor Scan"
    assert data.attrs["Notes"] == {"operator": "test"}


def test_load_bcs_multiple_image_columns(tmp_path) -> None:
    scan_path, expected_images = _write_bcs_scan(
        tmp_path, image_columns=("DiagOn YAG", "Camera 2")
    )

    data = load_bcs(scan_path)

    assert isinstance(data, xr.DataTree)
    assert set(data.children) == {"DiagOn YAG", "Camera 2"}
    for image_column in expected_images:
        arr = data[image_column].to_dataset()[image_column]
        np.testing.assert_array_equal(
            arr.values, np.stack(expected_images[image_column])
        )
        np.testing.assert_allclose(arr["EPU Gap"].values, [20.0, 20.2])
        assert "General" not in arr.attrs
        assert "Motors" not in arr.attrs
