import concurrent.futures
from typing import ClassVar

import cloudpickle
import dask.callbacks
import h5py
import numpy as np
import pytest
import scipy.ndimage
import xarray as xr

import erlab
from erlab.io.plugins.fesuma import FeSuMaLoader
from tests.io.plugins.test_fesuma import _forbid_image_reads, _write_fesuma

_ESQM_JACOBIAN_MAXIMA = {
    16: 7.312843172989124,
    28: 2.31404944,
    32: 2.8332359565230463,
}


def _calibrated_loader(configuration=4, coefficients=(0.0, 2.0)):
    class _CalibratedFeSuMaLoader(FeSuMaLoader):
        name = "_fesuma_calibration_test"
        fourier16_coefficients = coefficients
        fourier28_coefficients = coefficients
        fourier32_coefficients = coefficients
        additional_attrs: ClassVar[dict] = {"configuration": configuration}

    return _CalibratedFeSuMaLoader()


def _write_angle_file(
    path,
    *,
    x=None,
    y=None,
    units="mm",
    mode="Fourier 16°",
    warped=0,
    acquisition_mode="Fermi Surface",
    dtype=np.float64,
):
    x = np.linspace(-8, 8, 9) if x is None else np.asarray(x, dtype=float)
    y = np.linspace(6, -6, 7) if y is None else np.asarray(y, dtype=float)
    plane = 3 * x[:, None] + 5 * y[None, :]
    images = np.stack([plane + 100 * index for index in range(3)]).astype(dtype)
    _write_fesuma(path, images, acquisition_mode=acquisition_mode)
    with h5py.File(path, "r+") as file:
        for index in range(len(images)):
            group = file[f"{index:05d}"]
            group["AnalyzerMode"][0] = mode.encode()
            attributes = group["analysisImage"].attrs
            attributes["AnalyzerMode"] = np.bytes_(mode.encode())
            attributes["IsWarped"] = warped
            attributes["DimOffsetX"] = x[0]
            attributes["DimDeltaX"] = x[1] - x[0]
            attributes["DimOffsetY"] = y[0]
            attributes["DimDeltaY"] = y[1] - y[0]
            attributes["ScaleUnitsX"] = units.encode()
            attributes["ScaleUnitsY"] = units.encode()
            attributes["IGORWaveScaling"] = [
                [0, 0],
                [x[1] - x[0], x[0]],
                [y[1] - y[0], y[0]],
            ]
            attributes["IGORWaveUnits"] = np.array(
                ["counts", units, units], dtype=h5py.string_dtype()
            )
    return images, x, y


def _assert_plane_values(
    loaded, configuration, radial_factor, source_x, source_y, cap, jacobian, maximum
):
    x_name, y_name = ("alpha", "beta") if configuration == 4 else ("beta", "alpha")
    angle_x, angle_y = np.meshgrid(loaded[x_name], loaded[y_name], indexing="ij")
    radius = np.hypot(angle_x, angle_y)
    factor = radial_factor(radius)
    query_x, query_y = factor * angle_x, factor * angle_y
    inside = (
        (query_x > np.min(source_x) + 1e-5)
        & (query_x < np.max(source_x) - 1e-5)
        & (query_y > np.min(source_y) + 1e-5)
        & (query_y < np.max(source_y) - 1e-5)
        & (radius < cap - 1e-5)
    )
    outside = (
        (query_x < np.min(source_x) - 1e-5)
        | (query_x > np.max(source_x) + 1e-5)
        | (query_y < np.min(source_y) - 1e-5)
        | (query_y > np.max(source_y) + 1e-5)
        | (radius > cap + 1e-5)
    )
    values = loaded.transpose("eV", x_name, y_name).values
    assert inside.any()
    for index, image in enumerate(values):
        expected = (
            (3 * query_x + 5 * query_y + 100 * index) * jacobian(radius) / maximum
        )
        assert np.isfinite(image[inside]).all()
        np.testing.assert_allclose(image[inside], expected[inside], atol=1e-9)
        assert np.isnan(image[outside]).all()


@pytest.mark.parametrize("configuration", [3, 4])
def test_linear_radial_calibration(tmp_path, configuration):
    path = tmp_path / "Data_00000.h5"
    images, source_x, source_y = _write_angle_file(path)
    loaded = _calibrated_loader(configuration).load(path)
    x_name, y_name = ("alpha", "beta") if configuration == 4 else ("beta", "alpha")

    assert loaded.name == "Fermi Surface"
    assert loaded.dims == ("eV", x_name, y_name)
    np.testing.assert_allclose(loaded[x_name], source_x / 2, atol=1e-12)
    np.testing.assert_allclose(loaded[y_name], source_y / 2, atol=1e-12)
    np.testing.assert_allclose(loaded.values, images, atol=1e-10)
    assert loaded[x_name].attrs["units"] == "degree"
    assert loaded[y_name].attrs["units"] == "degree"
    assert loaded.attrs["units"] == "counts"
    assert loaded.attrs["configuration"] == configuration
    assert "detector_x" not in loaded.coords
    assert "detector_y" not in loaded.coords


def test_nonlinear_radial_calibration_resamples_both_axes(tmp_path):
    path = tmp_path / "Data_00000.h5"
    _, source_x, source_y = _write_angle_file(path)
    loaded = _calibrated_loader(coefficients=(0.0, 2.0, 1.0)).load(path)

    np.testing.assert_allclose(loaded.alpha, np.linspace(-2, 2, 9), atol=1e-6)
    np.testing.assert_allclose(
        loaded.beta, np.linspace(np.sqrt(7) - 1, 1 - np.sqrt(7), 7), atol=1e-6
    )
    _assert_plane_values(
        loaded,
        4,
        lambda radius: 2 + radius,
        source_x,
        source_y,
        8,
        lambda radius: (2 + 2 * radius) * (2 + radius),
        180,
    )


def test_nonlinear_flat_image_uses_radial_and_tangential_jacobian(tmp_path):
    path = tmp_path / "Data_00000.h5"
    _write_angle_file(path)
    with h5py.File(path, "r+") as file:
        for index in range(3):
            file[f"{index:05d}/analysisImage"][:] = 1.0
    loaded = _calibrated_loader(coefficients=(0.0, 2.0, 1.0)).load(path)

    angle_x, angle_y = np.meshgrid(loaded.alpha, loaded.beta, indexing="ij")
    radius = np.hypot(angle_x, angle_y)
    expected = (2 + 2 * radius) * (2 + radius) / 180
    valid = np.isfinite(loaded.isel(eV=0).values)
    assert valid.any()
    assert (~valid).any()
    np.testing.assert_allclose(
        loaded.values[:, valid], np.tile(expected[valid], (3, 1))
    )
    np.testing.assert_allclose(loaded.isel(alpha=4, beta=3), 4 / 180)
    np.testing.assert_allclose(loaded.isel(alpha=6, beta=3), 12 / 180, atol=1e-6)


def test_jacobian_normalization_uses_interior_maximum_independent_of_crop(tmp_path):
    path = tmp_path / "Data_00000.h5"
    loader = _calibrated_loader(coefficients=(0.0, 2.0, 53 / 272, -5 / 272))
    # J increases up to theta=4 and decreases from theta=4 to theta=8.
    maximum = 15379 / 2312
    centers = []
    for extent in (2.0, 4.0):
        detector_extent = 2 * extent + (53 * extent**2 - 5 * extent**3) / 272
        _write_angle_file(
            path,
            x=np.linspace(-detector_extent, detector_extent, 9),
            y=np.linspace(detector_extent, -detector_extent, 9),
        )
        with h5py.File(path, "r+") as file:
            for index in range(3):
                file[f"{index:05d}/analysisImage"][:] = 1.0
        loaded = loader.load(path)

        center = loaded.isel(alpha=4, beta=4)
        np.testing.assert_allclose(center, 4 / maximum)
        assert np.nanmax(loaded.values) <= 1 + 1e-12
        centers.append(center.values)

    np.testing.assert_array_equal(*centers)
    np.testing.assert_allclose(loaded.isel(alpha=-1, beta=4), 1.0)


@pytest.mark.parametrize("configuration", [3, 4])
@pytest.mark.parametrize("jacobian", [False, True])
@pytest.mark.parametrize(
    ("acquisition_mode", "dtype"),
    [("LockIn", np.int32), ("Fermi Surface", np.float64)],
)
def test_angular_resampling_matches_scipy(
    tmp_path, configuration, acquisition_mode, dtype, jacobian
):
    path = tmp_path / "Data_00000.h5"
    images, source_x, source_y = _write_angle_file(
        path, acquisition_mode=acquisition_mode, dtype=dtype
    )
    loaded = _calibrated_loader(configuration, coefficients=(0.0, 2.0, 1.0)).load(
        path, jacobian=jacobian
    )
    x_name, y_name = ("alpha", "beta") if configuration == 4 else ("beta", "alpha")
    angle_x, angle_y = np.meshgrid(loaded[x_name], loaded[y_name], indexing="ij")
    radius = np.hypot(angle_x, angle_y)
    sampling = ((2 + radius) * angle_x, (2 + radius) * angle_y)
    valid = radius <= 8
    coordinates = []
    for values, source in zip(sampling, (source_x, source_y), strict=True):
        valid &= (values >= source.min() - 1e-9) & (values <= source.max() + 1e-9)
        coordinates.append(
            np.clip((values - source[0]) / (source[1] - source[0]), 0, len(source) - 1)
        )
    expected = np.stack(
        [
            scipy.ndimage.map_coordinates(
                image,
                coordinates,
                output=np.float64,
                order=1,
                mode="constant",
                cval=np.nan,
                prefilter=False,
            )
            for image in images
        ]
    )
    expected[:, ~valid] = np.nan
    if jacobian:
        expected *= (2 + 2 * radius) * (2 + radius) / 180

    assert images.dtype == dtype
    assert loaded.dtype == np.float64
    assert loaded.name == acquisition_mode
    assert valid.any()
    assert (~valid).any()
    assert (expected < 0).any()
    np.testing.assert_array_equal(np.isnan(loaded.values), np.isnan(expected))
    np.testing.assert_allclose(loaded.values, expected, atol=1e-12)


@pytest.mark.parametrize("scheduler", ["threads", "synchronous"])
@pytest.mark.parametrize("jacobian", [False, True])
def test_lazy_angular_resampling_uses_serial_kernel(
    tmp_path, monkeypatch, scheduler, jacobian
):
    path = tmp_path / "Data_00000.h5"
    _write_angle_file(path)
    loader = _calibrated_loader(coefficients=(0.0, 2.0, 1.0))
    expected = loader.load(path, jacobian=jacobian)

    def fail_parallel_kernel(*args, **kwargs):
        raise AssertionError("Parallel interpolation was used inside a Dask task")

    monkeypatch.setattr(erlab.analysis.interpolate, "_interp2", fail_parallel_kernel)
    construction_tasks = []
    with (
        monkeypatch.context() as patch,
        dask.callbacks.Callback(
            pretask=lambda key, *args: construction_tasks.append(key)
        ),
    ):
        _forbid_image_reads(patch)
        lazy = loader.load(path, chunks={"eV": 1}, jacobian=jacobian)

    assert construction_tasks == []
    assert lazy.chunks is not None
    xr.testing.assert_allclose(lazy.compute(scheduler=scheduler), expected)


def test_eager_angular_resampling_uses_serial_kernel_in_worker(tmp_path, monkeypatch):
    path = tmp_path / "Data_00000.h5"
    _write_angle_file(path)
    loader = _calibrated_loader(coefficients=(0.0, 2.0, 1.0))
    parallel_calls = []
    parallel_kernel = erlab.analysis.interpolate._interp2

    def record_parallel_kernel(*args, **kwargs):
        parallel_calls.append(True)
        return parallel_kernel(*args, **kwargs)

    monkeypatch.setattr(erlab.analysis.interpolate, "_interp2", record_parallel_kernel)
    expected = loader.load(path)
    assert parallel_calls

    def fail_parallel_kernel(*args, **kwargs):
        raise AssertionError("Parallel interpolation was used inside a worker thread")

    monkeypatch.setattr(erlab.analysis.interpolate, "_interp2", fail_parallel_kernel)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        actual = executor.submit(loader.load, path).result()

    xr.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("extent", [5.0, 10.0])
@pytest.mark.parametrize("mode", [16, 28, 32])
def test_esqm_calibration_preserves_valid_source_corners(tmp_path, extent, mode):
    path = tmp_path / "Data_00000.h5"
    _write_angle_file(
        path,
        x=np.linspace(-extent, extent, 41),
        y=np.linspace(extent, -extent, 41),
        mode=f"Fourier {mode}°",
    )
    with h5py.File(path, "r+") as file:
        for index in range(3):
            file[f"{index:05d}/analysisImage"][:] = 1.0
    loader = erlab.io.loaders["esqm"]
    loaded = loader.load(path)

    corners = loaded.isel(alpha=[0, -1], beta=[0, -1])
    coefficients = getattr(loader, f"fourier{mode}_coefficients")
    radius = np.hypot(corners.alpha, corners.beta)
    expected = (
        np.polynomial.polynomial.polyval(radius, coefficients[1:])
        * np.polynomial.polynomial.polyval(
            radius, np.polynomial.polynomial.polyder(coefficients)
        )
        / _ESQM_JACOBIAN_MAXIMA[mode]
    )
    assert np.isfinite(corners).all()
    np.testing.assert_allclose(corners, np.broadcast_to(expected, corners.shape))
    np.testing.assert_allclose(
        loaded.isel(alpha=20, beta=20),
        coefficients[1] ** 2 / _ESQM_JACOBIAN_MAXIMA[mode],
        atol=1e-12,
    )


@pytest.mark.parametrize("slope", [0.0, 2.0])
def test_radial_jacobian_has_finite_physical_origin(tmp_path, slope):
    path = tmp_path / "Data_00000.h5"
    _write_angle_file(path)
    with h5py.File(path, "r+") as file:
        for index in range(3):
            file[f"{index:05d}/analysisImage"][:] = 5.0
    loader = _calibrated_loader(coefficients=(0.0, slope, 1.0))
    loaded = loader.load(path)

    center = loaded.isel(alpha=4, beta=3)
    assert center.alpha.item() == 0.0
    assert center.beta.item() == 0.0
    assert np.isfinite(center).all()
    np.testing.assert_allclose(
        center, 5 * slope**2 / ((slope + 16) * (slope + 8)), atol=1e-12
    )
    np.testing.assert_allclose(
        loader.load(path, jacobian=False).isel(alpha=4, beta=3), 5.0, atol=1e-12
    )


@pytest.mark.parametrize(("mode", "slope"), [(16, 2.0), (28, 3.0), (32, 4.0)])
def test_selects_mode_specific_calibration(tmp_path, mode, slope):
    loader = _calibrated_loader()
    type(loader).fourier16_coefficients = (0.0, 2.0)
    type(loader).fourier28_coefficients = (0.0, 3.0)
    type(loader).fourier32_coefficients = (0.0, 4.0)
    path = tmp_path / "Data_00000.h5"
    images, source_x, source_y = _write_angle_file(path, mode=f"Fourier {mode}°")
    loaded = loader.load(path)

    np.testing.assert_allclose(loaded.alpha, source_x / slope, atol=1e-12)
    np.testing.assert_allclose(loaded.beta, source_y / slope, atol=1e-12)
    np.testing.assert_allclose(loaded.values, images, atol=1e-10)


@pytest.mark.parametrize(("mode", "cap"), [(16, 8), (28, 15), (32, 17)])
def test_mode_acceptance_masks_angles_outside_domain(tmp_path, mode, cap):
    path = tmp_path / "Data_00000.h5"
    _, source_x, source_y = _write_angle_file(
        path,
        x=np.linspace(-40, 40, 81),
        y=np.linspace(40, -40, 81),
        mode=f"Fourier {mode}°",
    )
    loaded = _calibrated_loader().load(path)

    np.testing.assert_allclose(loaded.alpha[[0, -1]], [-cap, cap], atol=1e-12)
    np.testing.assert_allclose(loaded.beta[[0, -1]], [cap, -cap], atol=1e-12)
    _assert_plane_values(
        loaded, 4, lambda radius: 2, source_x, source_y, cap, lambda radius: 4, 4
    )


@pytest.mark.parametrize("configuration", [3, 4])
@pytest.mark.parametrize("warped", [0, 1])
@pytest.mark.parametrize("convert_angles", [False, True])
def test_degree_axes_only_rename(tmp_path, configuration, warped, convert_angles):
    path = tmp_path / "Data_00000.h5"
    images, source_x, source_y = _write_angle_file(path, units="degree", warped=warped)
    loaded = _calibrated_loader(configuration).load(path, convert_angles=convert_angles)
    x_name, y_name = ("alpha", "beta") if configuration == 4 else ("beta", "alpha")

    assert loaded.dims == ("eV", x_name, y_name)
    assert loaded.dtype == images.dtype
    np.testing.assert_array_equal(loaded.values, images)
    np.testing.assert_array_equal(loaded[x_name], source_x)
    np.testing.assert_array_equal(loaded[y_name], source_y)


@pytest.mark.parametrize("mode", ["Direct", "Direct Low"])
def test_direct_mode_keeps_spatial_image(tmp_path, mode):
    path = tmp_path / "Data_00000.h5"
    images, source_x, source_y = _write_angle_file(path, mode=mode)
    loaded = _calibrated_loader().load(path)

    assert loaded.dims == ("eV", "detector_x", "detector_y")
    np.testing.assert_array_equal(loaded.values, images)
    np.testing.assert_array_equal(loaded.detector_x, source_x)
    np.testing.assert_array_equal(loaded.detector_y, source_y)
    assert loaded.detector_x.attrs["units"] == "mm"


def test_varying_analyzer_mode_prevents_angular_conversion(tmp_path):
    path = tmp_path / "Data_00000.h5"
    _write_angle_file(path)
    with h5py.File(path, "r+") as file:
        file["00001/AnalyzerMode"][0] = "Fourier 28°".encode()
        file["00001/analysisImage"].attrs["AnalyzerMode"] = np.bytes_(
            "Fourier 28°".encode()
        )
    with pytest.raises(ValueError, match=r"(?i)lens mode"):
        _calibrated_loader().load(path)


@pytest.mark.parametrize(
    "modes",
    [
        ("Direct", "Direct Low", "Direct"),
        ("Fourier 16°", "Fourier 28°", "Fourier 16°"),
        ("Direct", "Fourier 16°", "Direct Low"),
    ],
)
@pytest.mark.parametrize("chunks", [None, {"eV": 1}])
def test_uncalibrated_varying_modes_keep_native_images(
    tmp_path, monkeypatch, modes, chunks
):
    path = tmp_path / "Data_00000.h5"
    images, source_x, source_y = _write_angle_file(path)
    with h5py.File(path, "r+") as file:
        for index, mode in enumerate(modes):
            group = file[f"{index:05d}"]
            group["AnalyzerMode"][0] = mode.encode()
            group["analysisImage"].attrs["AnalyzerMode"] = np.bytes_(mode.encode())
    loader = _calibrated_loader(coefficients=None)
    type(loader).fourier32_coefficients = (0.0, 2.0)
    with monkeypatch.context() as patch:
        if chunks is not None:
            _forbid_image_reads(patch)
        loaded = loader.load(path, chunks=chunks)

    assert loaded.dims == ("eV", "detector_x", "detector_y")
    assert (loaded.chunks is None) == (chunks is None)
    assert loaded.AnalyzerMode.dims == ("eV",)
    np.testing.assert_array_equal(loaded.AnalyzerMode, modes)
    np.testing.assert_array_equal(loaded.compute().values, images)
    np.testing.assert_array_equal(loaded.detector_x, source_x)
    np.testing.assert_array_equal(loaded.detector_y, source_y)
    assert loaded.detector_x.attrs["units"] == "mm"
    assert loaded.detector_y.attrs["units"] == "mm"


def test_can_disable_resampling(tmp_path):
    path = tmp_path / "Data_00000.h5"
    images, source_x, source_y = _write_angle_file(path)
    loaded = _calibrated_loader().load(path, convert_angles=False)

    assert loaded.dims == ("eV", "detector_x", "detector_y")
    np.testing.assert_array_equal(loaded.values, images)
    np.testing.assert_array_equal(loaded.detector_x, source_x)
    np.testing.assert_array_equal(loaded.detector_y, source_y)


def test_missing_optional_calibration_keeps_native_image(tmp_path):
    path = tmp_path / "Data_00000.h5"
    images, source_x, source_y = _write_angle_file(path)
    loaded = _calibrated_loader(coefficients=None).load(path)

    assert loaded.dims == ("eV", "detector_x", "detector_y")
    np.testing.assert_array_equal(loaded.values, images)
    np.testing.assert_array_equal(loaded.detector_x, source_x)
    np.testing.assert_array_equal(loaded.detector_y, source_y)


@pytest.mark.parametrize(
    "coefficients", [(0.0,), (0.0, np.nan), (1.0, 2.0), (0.0, 2.0, -1.0)]
)
def test_invalid_radial_calibration_is_rejected(tmp_path, coefficients):
    path = tmp_path / "Data_00000.h5"
    _write_angle_file(path)
    with pytest.raises(ValueError, match=r"(?i)calibration|increase|coefficient"):
        _calibrated_loader(coefficients=coefficients).load(path)


def test_invalid_angular_configuration_is_rejected(tmp_path):
    path = tmp_path / "Data_00000.h5"
    _write_angle_file(path)
    with pytest.raises(ValueError, match=r"(?i)configuration"):
        _calibrated_loader(configuration=1).load(path)


@pytest.mark.parametrize("configuration", [3, 4])
@pytest.mark.parametrize(
    "chunks",
    [
        None,
        1,
        {},
        {"alpha": 3, "beta": 2, "eV": 1},
        {"detector_x": 3, "detector_y": 2, "eV": 1},
    ],
)
def test_configuration_evaluation_is_independent_of_chunks(
    tmp_path, configuration, chunks
):
    path = tmp_path / "Data_00000.h5"
    _write_angle_file(path)
    with h5py.File(path, "r+") as file:
        file["presentHardware"][0] = 1
        for index in range(3):
            energy = file[f"{index:05d}"].create_dataset(
                "MonochromatorEnergy", data=[21.2]
            )
            energy.attrs["IGORWaveUnits"] = np.array(
                ["[eV]", ""], dtype=h5py.string_dtype()
            )
    calls = []

    def fixed_configuration(data):
        calls.append(data.hv.item())
        return configuration

    loader = _calibrated_loader(configuration=fixed_configuration)
    raw = loader.load_single(path, chunks=chunks)
    assert calls == []
    assert raw.dims == ("eV", "detector_x", "detector_y")

    loaded = loader.load(path, chunks=chunks)
    assert calls == [21.2]
    assert loaded.attrs["configuration"] == configuration
    assert "_fesuma_chunks" not in loaded.encoding
    expected = _calibrated_loader(configuration).load(path)
    xr.testing.assert_identical(loaded.compute(), expected)
    if isinstance(chunks, dict):
        angle_names = (
            {"detector_x": "alpha", "detector_y": "beta"}
            if configuration == 4
            else {"detector_x": "beta", "detector_y": "alpha"}
        )
        for name, size in chunks.items():
            assert max(loaded.chunksizes[angle_names.get(name, name)]) <= size


@pytest.mark.parametrize("configuration", [3, 4])
@pytest.mark.parametrize("jacobian", [False, True])
def test_angular_metadata_lazy_selection_and_roundtrip(
    tmp_path, monkeypatch, configuration, jacobian
):
    path = tmp_path / "Data_00000.h5"
    _write_angle_file(path)
    loader = _calibrated_loader(configuration, coefficients=(0.0, 2.0, 1.0))
    expected = loader.load(path, jacobian=jacobian)
    with monkeypatch.context() as patch:
        _forbid_image_reads(patch)
        metadata = loader.load(
            path,
            without_values=True,
            chunks={"alpha": 3, "beta": 2, "eV": 1},
            jacobian=jacobian,
        )
        lazy = loader.load(
            path, chunks={"alpha": 3, "beta": 2, "eV": 1}, jacobian=jacobian
        )
    assert expected.name == "Fermi Surface"
    assert lazy.name == "Fermi Surface"
    xr.testing.assert_identical(metadata, expected.copy(data=metadata.data))
    np.testing.assert_array_equal(metadata, 0.0)
    assert metadata.chunks is None
    assert metadata.data.strides == (0,) * metadata.ndim
    assert max(lazy.chunksizes["alpha"]) <= 3
    assert max(lazy.chunksizes["beta"]) <= 2
    assert max(lazy.chunksizes["eV"]) == 1
    restored = cloudpickle.loads(cloudpickle.dumps(lazy))
    xr.testing.assert_identical(restored.compute(), expected)
    output = tmp_path / "converted.h5"
    expected.to_netcdf(output, engine="h5netcdf")
    xr.testing.assert_identical(xr.load_dataarray(output, engine="h5netcdf"), expected)

    reads = []
    original_read_direct = h5py.Dataset.read_direct

    def read_direct(dataset, *args, **kwargs):
        if dataset.name.endswith("/analysisImage"):
            reads.append(dataset.name)
        return original_read_direct(dataset, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "read_direct", read_direct)
    xr.testing.assert_identical(lazy.isel(eV=1).compute(), expected.isel(eV=1))
    assert set(reads) == {"/00001/analysisImage"}


@pytest.mark.parametrize("mode", [16, 28, 32])
def test_esqm_uses_bundled_calibrations(tmp_path, mode):
    from erlab.io.plugins.esqm import ESQMLoader

    assert ESQMLoader.fourier16_coefficients == (0.0, 2.6953, 0.0099912, -0.0031449)
    assert ESQMLoader.fourier28_coefficients == (0.0, 1.5212, -0.00075013, -0.001731)
    assert ESQMLoader.fourier32_coefficients == (0.0, 1.6609, 0.013249, -0.0022113)
    path = tmp_path / "Data_00000.h5"
    _, source_x, source_y = _write_angle_file(path, mode=f"Fourier {mode}°")
    loaded = erlab.io.loaders["esqm"].load(path)

    assert loaded.name == "Fermi Surface"
    assert loaded.dims == ("eV", "alpha", "beta")
    assert loaded.attrs["configuration"] == 4
    assert loaded.attrs["data_loader_name"] == "esqm"
    coefficients = getattr(ESQMLoader, f"fourier{mode}_coefficients")
    _assert_plane_values(
        loaded,
        4,
        lambda radius: np.polynomial.polynomial.polyval(radius, coefficients[1:]),
        source_x,
        source_y,
        {16: 8, 28: 15, 32: 17}[mode],
        lambda radius: (
            np.polynomial.polynomial.polyval(radius, coefficients[1:])
            * np.polynomial.polynomial.polyval(
                radius, np.polynomial.polynomial.polyder(coefficients)
            )
        ),
        _ESQM_JACOBIAN_MAXIMA[mode],
    )
