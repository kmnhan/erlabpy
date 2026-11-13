import shutil

import h5py
import numpy as np
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

    reference = xr.load_dataarray(expected_dir / expected, engine="h5netcdf")
    reference.attrs["analyzer_work_function"] = reference.attrs.pop(
        "sample_workfunction"
    )
    xr.testing.assert_identical(loaded, reference)


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

    reference = xr.load_dataarray(expected_dir / "f_0002.h5", engine="h5netcdf")
    reference.attrs["analyzer_work_function"] = reference.attrs.pop(
        "sample_workfunction"
    )
    xr.testing.assert_identical(loaded, reference)


@pytest.mark.parametrize("chunks", [None, "auto"], ids=["no_chunks", "auto_chunks"])
@pytest.mark.parametrize("stored_beta", [False, True], ids=["scalar_beta", "beta_dim"])
def test_load_sample_angle_axes(tmp_path, chunks, stored_beta) -> None:
    file_path = tmp_path / "sample_angles.h5"
    beta = -4.996145248413086
    shape = (4, 5, 1) if stored_beta else (4, 5)
    counts = np.arange(1, np.prod(shape) + 1, dtype=float).reshape(shape)
    dwell_time = np.full(shape, 2.0)
    dwell_time.flat[0] = 0.0
    with h5py.File(file_path, "w") as file:
        file.attrs["FileName"] = file_path.name
        file.create_group("Beamline").attrs["energy"] = 111.0
        file.create_group("Measurement").attrs["LensThetaY"] = -5.25
        file.create_group("UserSettings").attrs["WorkFunction"] = 4.465
        data = file.create_group("Data")
        data.attrs.update(Dimension=len(shape), Shape=shape)
        data.create_dataset("Count", data=counts)
        data.create_dataset("Time", data=dwell_time)
        for index, (label, offset, delta, count) in enumerate(
            [
                ("Kinetic Energy", 105.6646, 0.001, 4),
                ("SampleThetaX", -2.0, 1.0, 5),
                ("SampleThetaY", beta, 0.538116455078125, 1),
            ]
        ):
            data.create_group(f"Axes{index}").attrs.update(
                Label=label, Offset=offset, Delta=delta, Count=count
            )
        if stored_beta:
            data.create_group("Axes3").attrs.update(
                Label="None", Offset=0, Delta=1, Count=10
            )

    loaded = erlab.io.loaders["ssrl52"].load(file_path, chunks=chunks)

    assert loaded.dims == (("eV", "alpha", "beta") if stored_beta else ("eV", "alpha"))
    assert loaded.shape == shape
    assert (loaded.chunks is not None) == (chunks is not None)
    np.testing.assert_allclose(loaded.eV, 105.6646 + np.arange(4) * 0.001 - 111 + 4.465)
    np.testing.assert_allclose(loaded.alpha, np.arange(-2.0, 3.0))
    np.testing.assert_allclose(loaded.beta, [beta] if stored_beta else beta)
    assert loaded.beta.dims == (("beta",) if stored_beta else ())
    assert loaded.beta.attrs == {
        "label": "SampleThetaY",
        "offset": beta,
        "delta": 0.538116455078125,
        "count": 1,
    }
    assert loaded.attrs["LensThetaY"] == -5.25
    expected_values = counts / 2.0
    expected_values.flat[0] = np.nan
    np.testing.assert_allclose(loaded.values, expected_values)
    assert "None" not in loaded.coords

    preview = erlab.io.loaders["ssrl52"].load(
        file_path, chunks=chunks, without_values=True
    )
    assert preview.attrs["LensThetaY"] == -5.25
    xr.testing.assert_identical(preview.beta, loaded.beta)


@pytest.mark.parametrize("chunks", [None, "auto"], ids=["no_chunks", "auto_chunks"])
def test_load_sample_angle_mapped_metadata(tmp_path, chunks) -> None:
    file_path = tmp_path / "sample_angle_map.h5"
    counts = np.arange(1, 61, dtype=float).reshape(4, 5, 3)
    monitor = np.array([0.1, 0.2, 0.3])
    with h5py.File(file_path, "w") as file:
        file.attrs["FileName"] = file_path.name
        beamline = file.create_group("Beamline")
        beamline.attrs.update(energy=111.0, I0="MapInfo:Beamline:I0")
        file.create_group("Measurement").attrs["LensThetaY"] = -5.25
        file.create_group("UserSettings").attrs["WorkFunction"] = 4.465
        data = file.create_group("Data")
        data.create_dataset("Count", data=counts)
        data.create_dataset("Time", data=np.full(counts.shape, 2.0))
        for index, (label, offset, delta, count) in enumerate(
            [
                ("Kinetic Energy", 105.6646, 0.001, 4),
                ("SampleThetaX", -2.0, 1.0, 5),
                ("SampleThetaY", -5.0, 0.5, 3),
            ]
        ):
            data.create_group(f"Axes{index}").attrs.update(
                Label=label, Offset=offset, Delta=delta, Count=count
            )
        file.create_group("MapInfo").create_dataset("Beamline:I0", data=monitor)

    loaded = erlab.io.loaders["ssrl52"].load(file_path, chunks=chunks)

    assert loaded.dims == ("eV", "alpha", "beta")
    assert (loaded.chunks is not None) == (chunks is not None)
    assert loaded.I0.dims == ("beta",)
    np.testing.assert_allclose(loaded.beta, [-5.0, -4.5, -4.0])
    assert loaded.beta.attrs == {
        "label": "SampleThetaY",
        "offset": -5.0,
        "delta": 0.5,
        "count": 3,
    }
    assert loaded.attrs["LensThetaY"] == -5.25
    np.testing.assert_allclose(loaded.I0, monitor)
    np.testing.assert_allclose(loaded.values, counts / 2.0)

    preview = erlab.io.loaders["ssrl52"].load(
        file_path, chunks=chunks, without_values=True
    )
    assert preview.attrs["LensThetaY"] == -5.25
    xr.testing.assert_identical(preview.beta, loaded.beta)


def test_summarize(data_dir) -> None:
    erlab.io.summarize()
