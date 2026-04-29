import builtins
import json
import shutil

import numpy as np
import pytest
import xarray as xr

import erlab
from erlab.io.plugins import _merlin_bcs
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
        "Empty Trace",
        "Text Trace",
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
            "",
            "bad",
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


def _write_bcs_text(scan_path, header, columns, rows) -> None:
    data_table = "\t".join(columns) + "\n" + "\n".join("\t".join(row) for row in rows)
    scan_path.write_text(
        f"HEADER\n{json.dumps(header, indent=4)}\nDATA\n{data_table}",
        encoding="utf-8",
    )


def _save_png(path, values) -> None:
    image_module = pytest.importorskip("PIL.Image")
    path.parent.mkdir(exist_ok=True)
    image_module.fromarray(np.asarray(values)).save(path)


def _save_text_payload(path, columns, rows) -> None:
    path.parent.mkdir(exist_ok=True)
    data_table = "\t".join(columns) + "\n" + "\n".join("\t".join(row) for row in rows)
    path.write_text(data_table, encoding="utf-8")


def test_load_bcs_dataarray(tmp_path) -> None:
    scan_path, expected_images = _write_bcs_scan(tmp_path)

    data = load_bcs(scan_path)

    assert isinstance(data, xr.DataArray)
    assert data.dims == ("y", "x", "EPU Gap")
    assert data.dtype == np.uint16
    np.testing.assert_array_equal(
        data.values, np.moveaxis(np.stack(expected_images["DiagOn YAG"]), 0, -1)
    )
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
    assert "Empty Trace" not in data.coords
    assert "Text Trace" not in data.coords
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
            arr.values, np.moveaxis(np.stack(expected_images[image_column]), 0, -1)
        )
        np.testing.assert_allclose(arr["EPU Gap"].values, [20.0, 20.2])
        assert "General" not in arr.attrs
        assert "Motors" not in arr.attrs


def test_load_bcs_scan_axis_fallbacks(tmp_path) -> None:
    image_path = tmp_path / "fallback.png"
    _save_png(image_path, np.arange(6, dtype=np.uint16).reshape(2, 3))

    scan_path = tmp_path / "readback.txt"
    _write_bcs_text(
        scan_path,
        {"Scan Type": "Single Motor Scan", "Single Motor Scan": {"X Motor": "Motor"}},
        ["Motor", "Image"],
        [["1.5", str(image_path)]],
    )
    data = load_bcs(scan_path)
    assert data.dims == ("y", "x", "Motor")
    np.testing.assert_allclose(data["Motor"].values, [1.5])

    scan_path = tmp_path / "step.txt"
    _write_bcs_text(
        scan_path,
        {
            "Scan Type": "Single Motor Scan",
            "Single Motor Scan": {"X Motor": "Missing"},
            "Missing": "already a coord",
            "Motors": None,
        },
        ["Image"],
        [[str(image_path)]],
    )
    data = load_bcs(scan_path)
    assert data.dims == ("y", "x", "Missing")
    np.testing.assert_allclose(data["Missing"].values, [0.0])
    assert "Missing" not in data.attrs
    assert "Motors" not in data.attrs


def test_load_bcs_absolute_rgb_images(tmp_path) -> None:
    paths = [tmp_path / f"rgb_{i}.png" for i in range(2)]
    for i, path in enumerate(paths):
        _save_png(path, np.full((2, 3, 3), i, dtype=np.uint8))

    scan_path = tmp_path / "rgb.txt"
    _write_bcs_text(
        scan_path,
        {"Scan Type": "Single Motor Scan"},
        ["Image"],
        [[str(path)] for path in paths],
    )

    data = load_bcs(scan_path)

    assert data.dims == ("y", "x", "channel", "step")
    np.testing.assert_array_equal(data["step"].values, [0.0, 1.0])
    np.testing.assert_array_equal(data["channel"].values, [0, 1, 2])
    assert data.shape == (2, 3, 3, 2)


def test_load_bcs_single_row_text_payload(tmp_path) -> None:
    payload_path = tmp_path / "f 000097 Spectrums" / "f 000097 001.txt"
    _save_text_payload(
        payload_path,
        ["Kinetic Energy [eV]", "Counts [a.u.]"],
        [["88.633", "10.0"], ["88.635", "12.0"]],
    )

    scan_path = tmp_path / "f 000097.txt"
    _write_bcs_text(
        scan_path,
        {
            "Scan Type": "Image Single Motor Scan",
            "Image Single Motor Scan": {
                "Instrument": "Scienta",
                "X Motor": "Fake Motor",
            },
            "Motors": {"Fake Motor": 0.0, "BL Energy": 105.0},
        },
        ["Time (s)", "Fake Motor Goal", "Fake Motor Actual", "Scienta"],
        [["0.294", "0.0", "0.0", r"..\f 000097 Spectrums\f 000097 001.txt"]],
    )

    data = load_bcs(scan_path)

    assert data.name == "Scienta"
    assert data.dims == ("Kinetic Energy [eV]", "Fake Motor")
    assert data.shape == (2, 1)
    np.testing.assert_allclose(data["Kinetic Energy [eV]"].values, [88.633, 88.635])
    np.testing.assert_allclose(data["Fake Motor"].values, [0.0])
    np.testing.assert_allclose(data.values, [[10.0], [12.0]])
    assert "eV" not in data.coords
    assert data.attrs["BCS value column"] == "Counts [a.u.]"


def test_load_bcs_multi_row_text_payload(tmp_path) -> None:
    payload_dir = tmp_path / "spectra"
    _save_text_payload(
        payload_dir / "a.txt",
        ["energy", "counts"],
        [["1.0", "10.0"], ["2.0", "20.0"]],
    )
    _save_text_payload(
        payload_dir / "b.txt",
        ["energy", "counts"],
        [["1.0", "30.0"], ["2.0", "40.0"]],
    )

    scan_path = tmp_path / "scan.txt"
    _write_bcs_text(
        scan_path,
        {"Scan Type": "Single Motor Scan", "Single Motor Scan": {"X Motor": "Motor"}},
        ["Motor Goal", "Spectrum"],
        [
            ["0.0", r"..\spectra\a.txt"],
            ["1.0", r"..\spectra\b.txt"],
        ],
    )

    data = load_bcs(scan_path)

    assert data.dims == ("energy", "Motor")
    np.testing.assert_allclose(data["Motor"].values, [0.0, 1.0])
    np.testing.assert_allclose(data.values, [[10.0, 30.0], [20.0, 40.0]])


def test_load_bcs_text_payload_multiple_value_columns(tmp_path) -> None:
    payload_path = tmp_path / "tables" / "a.txt"
    _save_text_payload(
        payload_path,
        ["axis", "A", "B"],
        [["1.0", "10.0", "20.0"], ["2.0", "30.0", "40.0"]],
    )

    scan_path = tmp_path / "scan.txt"
    _write_bcs_text(
        scan_path,
        {"Scan Type": "Single Motor Scan"},
        ["Table"],
        [[r"..\tables\a.txt"]],
    )

    data = load_bcs(scan_path)

    assert data.dims == ("axis", "column", "step")
    np.testing.assert_array_equal(data["column"].values, ["A", "B"])
    np.testing.assert_allclose(data.values[:, :, 0], [[10.0, 20.0], [30.0, 40.0]])


def test_load_bcs_mixed_payload_columns_return_datatree(tmp_path) -> None:
    image_path = tmp_path / "camera.png"
    _save_png(image_path, np.arange(6, dtype=np.uint16).reshape(2, 3))
    payload_path = tmp_path / "spectra" / "a.txt"
    _save_text_payload(
        payload_path,
        ["energy", "counts"],
        [["1.0", "10.0"], ["2.0", "20.0"]],
    )

    scan_path = tmp_path / "scan.txt"
    _write_bcs_text(
        scan_path,
        {"Scan Type": "Single Motor Scan", "Single Motor Scan": {"X Motor": "Motor"}},
        ["Motor Goal", "Camera", "Spectrum"],
        [["0.0", str(image_path), r"..\spectra\a.txt"]],
    )

    data = load_bcs(scan_path)

    assert isinstance(data, xr.DataTree)
    assert set(data.children) == {"Camera", "Spectrum"}
    assert data["Camera"].to_dataset()["Camera"].dims == ("y", "x", "Motor")
    assert data["Spectrum"].to_dataset()["Spectrum"].dims == ("energy", "Motor")


def test_load_bcs_text_payload_mismatched_axes(tmp_path) -> None:
    payload_dir = tmp_path / "spectra"
    _save_text_payload(
        payload_dir / "a.txt",
        ["energy", "counts"],
        [["1.0", "10.0"], ["2.0", "20.0"]],
    )
    _save_text_payload(
        payload_dir / "b.txt",
        ["energy", "counts"],
        [["1.0", "30.0"], ["3.0", "40.0"]],
    )

    scan_path = tmp_path / "scan.txt"
    _write_bcs_text(
        scan_path,
        {"Scan Type": "Single Motor Scan", "Single Motor Scan": {"X Motor": "Motor"}},
        ["Motor Goal", "Spectrum"],
        [
            ["0.0", r"..\spectra\a.txt"],
            ["1.0", r"..\spectra\b.txt"],
        ],
    )

    with pytest.raises(ValueError, match="text payload axes differ"):
        load_bcs(scan_path)


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("DATA\nx\n1\n", "not a valid BCS data file"),
        ("HEADER\n{}\nDATA\n", "missing BCS data table"),
        ("HEADER\n{}\nDATA\nx\n", "contains no BCS data rows"),
        ("HEADER\n{}\nDATA\nx\n1\n", "contains no BCS payload columns"),
    ],
)
def test_load_bcs_invalid_files(tmp_path, text, match) -> None:
    scan_path = tmp_path / "invalid.txt"
    scan_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_bcs(scan_path)


def test_load_bcs_missing_image(tmp_path) -> None:
    scan_path = tmp_path / "missing_image.txt"
    _write_bcs_text(
        scan_path,
        {"Scan Type": "Single Motor Scan"},
        ["Image"],
        [["missing.png"]],
    )

    with pytest.raises(FileNotFoundError, match="Could not find BCS image"):
        load_bcs(scan_path)


def test_load_bcs_mismatched_image_shapes(tmp_path) -> None:
    image_dir = tmp_path / "shape Images"
    _save_png(image_dir / "a.png", np.zeros((2, 3), dtype=np.uint16))
    _save_png(image_dir / "b.png", np.zeros((3, 3), dtype=np.uint16))
    scan_path = tmp_path / "shape.txt"
    _write_bcs_text(
        scan_path,
        {"Scan Type": "Single Motor Scan"},
        ["Image"],
        [
            [r"..\shape Images\a.png"],
            [r"..\shape Images\b.png"],
        ],
    )

    with pytest.raises(ValueError, match="All BCS images must have the same shape"):
        load_bcs(scan_path)


def test_load_bcs_invalid_image_dimensions(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "image.png"
    image_path.touch()
    scan_path = tmp_path / "invalid_image_dims.txt"
    _write_bcs_text(
        scan_path,
        {"Scan Type": "Single Motor Scan"},
        ["Image"],
        [[str(image_path)]],
    )

    image_module = pytest.importorskip("PIL.Image")

    class ScalarImage:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def __array__(self):
            return np.asarray(1, dtype=np.uint16)

    monkeypatch.setattr(image_module, "open", lambda _: ScalarImage())
    with pytest.raises(
        ValueError, match="BCS images must be two-dimensional or RGB/RGBA images"
    ):
        load_bcs(scan_path)


def test_load_bcs_pillow_missing(tmp_path, monkeypatch) -> None:
    scan_path = tmp_path / "no_pillow.txt"
    _write_bcs_text(
        scan_path,
        {"Scan Type": "Single Motor Scan"},
        ["Image"],
        [["image.png"]],
    )

    real_import = builtins.__import__

    def fake_import(name, globalns=None, localns=None, fromlist=(), level=0):
        if name == "PIL":
            raise ImportError("blocked")
        return real_import(name, globalns, localns, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="PIL is required"):
        load_bcs(scan_path)


def test_load_bcs_private_name_deduplication() -> None:
    existing = {"name", "name raw", "name raw 1"}

    assert _merlin_bcs._unique_name("name", existing) == "name raw 2"
    assert "name raw 2" in existing
