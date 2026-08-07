import shutil

import h5py
import pytest
import xarray as xr

import erlab


@pytest.fixture(scope="module")
def data_dir(test_data_dir):
    erlab.io.set_loader("ssrl52")
    erlab.io.set_data_dir(test_data_dir / "ssrl52")
    return test_data_dir / "ssrl52"


@pytest.fixture(scope="module")
def expected_dir(data_dir):
    return data_dir / "expected"


@pytest.mark.parametrize("chunks", [None, "auto"], ids=["no_chunks", "auto_chunks"])
@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ("f_0002.h5", "f_0002.h5"),
        (2, "f_0002.h5"),
        ("f_zap_0002.h5", "f_zap_0002.h5"),
        ({"identifier": 2, "zap": True}, "f_zap_0002.h5"),
    ],
)
def test_load(expected_dir, args, expected, chunks) -> None:
    loaded = (
        erlab.io.load(**args, chunks=chunks)
        if isinstance(args, dict)
        else erlab.io.load(args, chunks=chunks)
    )

    if chunks is not None:
        assert loaded.chunks is not None

    xr.testing.assert_identical(
        loaded, xr.load_dataarray(expected_dir / expected, engine="h5netcdf")
    )


def test_load_legacy_compat_avoids_xarray_reduction(
    data_dir, expected_dir, tmp_path, monkeypatch
) -> None:
    legacy_file = tmp_path / "legacy_ssrl52.h5"
    shutil.copyfile(data_dir / "f_0002.h5", legacy_file)
    with h5py.File(legacy_file, "r+") as file:
        file.move("Data", "data")
        file["data"].move("Count", "counts")
        file["data"].move("Time", "exposure")

    def unexpected_xarray_reduction(*_args, **_kwargs):
        raise AssertionError("SSRL52 loading must not use DataArray.min()")

    with monkeypatch.context() as patch:
        patch.setattr(xr.DataArray, "min", unexpected_xarray_reduction)
        loaded = erlab.io.load(legacy_file)

    xr.testing.assert_identical(
        loaded,
        xr.load_dataarray(expected_dir / "f_0002.h5", engine="h5netcdf"),
    )


def test_summarize(data_dir) -> None:
    erlab.io.summarize()
