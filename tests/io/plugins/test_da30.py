import zipfile

import numpy as np
import pytest
import xarray as xr

import erlab
from erlab.io.plugins.da30 import InvalidDA30ZipError, _parse_value, load_zip


def _make_da30_region(
    region: str,
    *,
    name: str,
    values: np.ndarray,
    dims: tuple[str, str, str] = ("depth", "height", "width"),
) -> dict[str, bytes]:
    shape = values.shape
    assert len(shape) == 3

    spectrum_ini = (
        "[spectrum]\n"
        f"depth = {shape[0]}\n"
        "depthoffset = 0.0\n"
        "depthdelta = 1.0\n"
        f"depthlabel = {dims[0]}\n"
        f"height = {shape[1]}\n"
        "heightoffset = 1.0\n"
        "heightdelta = 0.5\n"
        f"heightlabel = {dims[1]}\n"
        f"width = {shape[2]}\n"
        "widthoffset = -1.0\n"
        "widthdelta = 0.25\n"
        f"widthlabel = {dims[2]}\n"
        f"name = {name}\n"
    ).encode()
    attrs_ini = b"[attrs]\nPass Energy = 20\nLens Mode = Angular\n"

    return {
        f"Spectrum_{region}.ini": spectrum_ini,
        f"{region}.ini": attrs_ini,
        f"Spectrum_{region}.bin": values.astype(np.float32).tobytes(),
    }


def _write_da30_directory(path, regions: dict[str, dict[str, bytes]]) -> None:
    path.mkdir()
    for files in regions.values():
        for name, content in files.items():
            (path / name).write_bytes(content)


def _write_da30_zip(path, regions: dict[str, dict[str, bytes]]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for files in regions.values():
            for name, content in files.items():
                zf.writestr(name, content)


@pytest.fixture(scope="module")
def data_dir(test_data_dir):
    erlab.io.set_loader("da30")
    erlab.io.set_data_dir(test_data_dir / "da30")
    return test_data_dir / "da30"


@pytest.fixture(scope="module")
def expected_dir(data_dir):
    return data_dir / "expected"


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ("f0001f_001.ibw", "f0001f_001.h5"),
        (1, "f0001f_001.h5"),
        ("f0002.zip", "f0002.h5"),
        (2, "f0002.h5"),
        ("f0003.pxt", "f0003.h5"),
        (3, "f0003.h5"),
    ],
)
def test_load(expected_dir, args, expected) -> None:
    loaded = erlab.io.load(**args) if isinstance(args, dict) else erlab.io.load(args)

    expected_darr = xr.load_dataarray(expected_dir / expected, engine="h5netcdf")

    xr.testing.assert_equal(loaded, expected_darr)
    assert loaded.name == expected_darr.name

    if loaded.attrs != expected_darr.attrs:
        # Attrs may be np.float('nan'), which do not compare equal
        for key in list(expected_darr.attrs.keys()):
            if isinstance(expected_darr.attrs[key], np.floating) and np.isnan(
                expected_darr.attrs[key]
            ):
                assert np.isnan(loaded.attrs.get(key, None))
                del loaded.attrs[key]
                del expected_darr.attrs[key]

    assert loaded.attrs == expected_darr.attrs


def test_load_zip_multiregion_directory_without_values(tmp_path) -> None:
    region1 = _make_da30_region(
        "R1",
        name="region_one",
        values=np.arange(6, dtype=np.float32).reshape(1, 2, 3),
    )
    region2 = _make_da30_region(
        "R2",
        name="region_two",
        values=np.arange(6, 12, dtype=np.float32).reshape(1, 2, 3),
    )
    data_dir = tmp_path / "unzipped"
    _write_da30_directory(data_dir, {"R1": region1, "R2": region2})

    loaded = load_zip(data_dir, without_values=True)

    assert isinstance(loaded, xr.DataTree)
    assert set(loaded.groups) == {"/", "/region_one", "/region_two"}

    for node_name in ("region_one", "region_two"):
        leaf = loaded[node_name]
        assert isinstance(leaf, xr.DataTree)
        arr = next(iter(leaf.dataset.data_vars.values()))
        assert arr.dtype == np.float32
        assert arr.shape == (1, 2, 3)
        assert np.count_nonzero(arr.values) == 0
        assert arr.attrs["Pass Energy"] == 20
        assert arr.attrs["Lens Mode"] == "Angular"


def test_load_zip_invalid_zip_raises(tmp_path) -> None:
    bad_zip = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("README.txt", "not a DA30 archive")

    with pytest.raises(InvalidDA30ZipError):
        load_zip(bad_zip)


def test_load_zip_invalid_directory_raises(tmp_path) -> None:
    bad_dir = tmp_path / "bad_dir"
    bad_dir.mkdir()
    (bad_dir / "README.txt").write_text("not a DA30 directory")

    with pytest.raises(InvalidDA30ZipError):
        load_zip(bad_dir)


def test_parse_value_passthrough_non_string() -> None:
    assert _parse_value(1.5) == 1.5
