import cloudpickle
import h5py
import numpy as np
import pytest
import xarray as xr

import erlab

_ANALYZER_CHANNELS = ("ANA1", "ANA2", "ANA3", "ANA4", "MCP", "SCREEN")
_ANALYZER_VECTORS = (
    "AnalyzerCurrentVoltages",
    "AnalyzerUserSetVoltages",
    "AnalyzerUserSetAmplitudes",
)


def _write_fesuma(
    path,
    images,
    *,
    steps=None,
    sweeps=None,
    group_names=None,
    expected_steps=None,
    expected_sweeps=1,
    acquisition_coordinate="EKIN",
    acquisition_mode="LockIn",
    exposure_times=None,
):
    """Write a small export with the same layout as actual data."""
    images = np.asarray(images)
    count = len(images)
    steps = list(range(count)) if steps is None else steps
    sweeps = [0] * count if sweeps is None else sweeps
    group_names = (
        [f"{index:05d}" for index in range(count)]
        if group_names is None
        else group_names
    )
    exposure_times = [100.0] * count if exposure_times is None else exposure_times
    expected_steps = count if expected_steps is None else expected_steps
    with h5py.File(path, "w") as file:
        for key, value in {
            "wSaveType": "Acquisition",
            "AcquisitionMode": acquisition_mode,
            "AcquisitionCoordinate": acquisition_coordinate,
            "AcquisitionCoordinateNice": "Kinetic Energy",
            "AcquisitionStartTimeNice": "2026-06-30T15:49:06.2Z",
        }.items():
            file[key] = [value.encode()]
        for key, value in {
            "AcquisitionType": int(acquisition_mode == "LockIn"),
            "AcquisitionNumberOfSweeps": expected_sweeps,
            "AcquisitionNumberOfExposures": 20,
            "IntensityConservvation": 0,
        }.items():
            file[key] = [float(value)]
        if acquisition_coordinate == "EKIN":
            for key, value in {
                "AcquisitionEkinStart": 10.0,
                "AcquisitionEkinStop": 10.0 + 0.5 * expected_steps,
                "AcquisitionEkinStep": 0.5,
            }.items():
                file[key] = [value]
        file["CommonData"] = np.array(
            [[b"Program", b"Igor Pro 64bit"], [b"Package Version", b"1.1"]]
        )
        file["presentHardware"] = np.array([0, 0, 1], dtype=np.uint8)
        for index, (name, step, sweep, exposure, image) in enumerate(
            zip(
                group_names,
                steps,
                sweeps,
                exposure_times,
                images,
                strict=True,
            )
        ):
            group = file.create_group(name)
            group["AcquisitionCurrentStep"] = [float(step)]
            group["AcquisitionCurrentSweep"] = [float(sweep)]
            group["AnalyzerMode"] = [b"Fourier 16\xc2\xb0"]
            group["SaveTimestamp"] = [f"2026-06-30T06:49:{index:02d}.0Z".encode()]
            group["AnalyzerUserSetVoltages"] = np.array(
                [-8.25, 90.75, -8.25, -15.54, 1380, 3400], dtype=np.float32
            )
            group["AnalyzerCurrentVoltages"] = group["AnalyzerUserSetVoltages"][()]
            group["AnalyzerUserSetAmplitudes"] = np.array(
                [-0.16, 1.83, -0.16, -0.3, 0, 0], dtype=np.float32
            )
            for name in _ANALYZER_VECTORS:
                group[name].attrs["IGORWaveDimensionLabels"] = np.array(
                    [[""], *[[channel] for channel in _ANALYZER_CHANNELS]],
                    dtype=h5py.string_dtype(),
                )
                group[name].attrs["IGORWaveType"] = 2
                group[name].attrs["IGORWaveUnits"] = np.array(
                    ["[V]", ""], dtype=h5py.string_dtype()
                )
            dataset = group.create_dataset("analysisImage", data=image)
            dataset.attrs.update(
                {
                    "DimOffsetX": -2.0,
                    "DimDeltaX": 0.5,
                    "ScaleUnitsX": np.bytes_(b"mm"),
                    "DimOffsetY": 3.0,
                    "DimDeltaY": -1.0,
                    "ScaleUnitsY": np.bytes_(b"mm"),
                    "IGORWaveScaling": [[0, 0], [0.5, -2], [-1, 3]],
                    "IGORWaveType": 32,
                    "IGORWaveUnits": np.array(
                        ["", "mm", "mm"], dtype=h5py.string_dtype()
                    ),
                    "AccumulationCount": 1.0,
                    "ExposureTime": float(exposure),
                    "ExposureTime_UNIT": np.bytes_(b"[ms]"),
                    "IsWarped": np.uint8(0),
                    "AnalyzerMode": np.bytes_(b"Fourier 16\xc2\xb0"),
                    "AcquisitionEndTimestamp": np.bytes_(b"TZ"),
                }
            )

        for dataset in file.values():
            if isinstance(dataset, h5py.Dataset):
                dataset.attrs["IGORWaveType"] = 4
    return path


def _images(count):
    return (np.arange(count * 6).reshape(count, 2, 3) - 11).astype(np.int32)


def _scan_layout(layout):
    if layout == "sweeps":
        return (
            4,
            {
                "steps": [0, 1, 0, 1],
                "sweeps": [0, 0, 1, 1],
                "expected_steps": 2,
                "expected_sweeps": 2,
            },
            ("sweep", "eV"),
            (2, 2),
        )
    if layout == "partial":
        return (
            3,
            {
                "steps": [0, 2, 4],
                "group_names": ["00000", "00002", "00004"],
                "expected_steps": 6,
            },
            ("frame",),
            (3,),
        )
    return 3, {}, ("eV",), (3,)


def _load(path, **kwargs):
    with erlab.io.loader_context("fesuma"):
        return erlab.io.load(path, **kwargs)


def _forbid_image_reads(monkeypatch):
    original_getitem = h5py.Dataset.__getitem__
    original_read_direct = h5py.Dataset.read_direct

    def getitem(dataset, *args, **kwargs):
        if dataset.name.endswith("/analysisImage"):
            raise AssertionError("Image payload was read before computation")
        return original_getitem(dataset, *args, **kwargs)

    def read_direct(dataset, *args, **kwargs):
        if dataset.name.endswith("/analysisImage"):
            raise AssertionError("Image payload was read before computation")
        return original_read_direct(dataset, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", getitem)
    monkeypatch.setattr(h5py.Dataset, "read_direct", read_direct)


def test_load_detector_order_and_energy(tmp_path):
    images = _images(4)
    path = _write_fesuma(tmp_path / "Data_00007.h5", images)
    loaded = _load(path)

    assert loaded.name == "LockIn"
    assert loaded.dims == ("eV", "detector_x", "detector_y")
    assert loaded.dtype == np.int32
    np.testing.assert_array_equal(loaded.values, images)
    np.testing.assert_array_equal(loaded.eV, [10, 10.5, 11, 11.5])
    np.testing.assert_array_equal(loaded.detector_x, [-2, -1.5])
    np.testing.assert_array_equal(loaded.detector_y, [3, 2, 1])
    assert loaded.detector_x.attrs["units"] == "mm"
    assert loaded.detector_y.attrs["units"] == "mm"
    assert loaded.eV.attrs["units"] == "eV"
    assert loaded.attrs["AcquisitionMode"] == "LockIn"
    assert loaded.attrs["data_loader_name"] == "fesuma"
    assert "alpha" not in loaded.coords
    assert "beta" not in loaded.coords
    assert "step" not in loaded.coords
    assert "frame_group" not in loaded.coords
    assert loaded.attrs["configuration"] == 4


def test_identify_and_scan_index(tmp_path):
    images = _images(2)
    path = _write_fesuma(tmp_path / "Data_00007.h5", images)
    _write_fesuma(tmp_path / "Other_00007.h5", images + 100)
    loader = erlab.io.loaders["fesuma"]

    paths, coordinates = loader.identify(7, tmp_path)
    assert [str(value) for value in paths] == [str(path)]
    assert coordinates == {}
    with erlab.io.loader_context("fesuma", tmp_path):
        by_index = erlab.io.load(7)
    xr.testing.assert_identical(by_index, _load(path))


def test_numeric_group_order(tmp_path):
    images = _images(3)
    path = _write_fesuma(
        tmp_path / "Data_00000.h5",
        images,
        group_names=["00001", "00002", "00010"],
        steps=[1, 2, 10],
        expected_steps=11,
    )
    loaded = _load(path)
    np.testing.assert_array_equal(loaded.values, images)
    np.testing.assert_array_equal(loaded.eV, [10.5, 11, 15])


def test_preserves_sweeps_and_intensity(tmp_path):
    images = _images(4)
    path = _write_fesuma(
        tmp_path / "Data_00000.h5",
        images,
        steps=[0, 1, 0, 1],
        sweeps=[0, 0, 1, 1],
        expected_steps=2,
        expected_sweeps=2,
        exposure_times=[100, 100, 200, 200],
    )
    loaded = _load(path)

    assert loaded.dims == ("sweep", "eV", "detector_x", "detector_y")
    np.testing.assert_array_equal(loaded.sweep, [0, 1])
    np.testing.assert_array_equal(loaded.eV, [10, 10.5])
    np.testing.assert_array_equal(loaded.values, images.reshape(2, 2, 2, 3))
    assert loaded.dtype == np.int32
    assert "step" not in loaded.coords
    assert "frame_group" not in loaded.coords


@pytest.mark.parametrize("chunks", [None, {}, {"eV": 2}])
def test_partial_scan_uses_recorded_energy_axis(tmp_path, chunks):
    images = _images(4)
    path = _write_fesuma(tmp_path / "Data_00000.h5", images, expected_steps=8)
    loaded = _load(path, chunks=chunks)

    assert loaded.dims == ("eV", "detector_x", "detector_y")
    assert loaded.eV.dims == ("eV",)
    np.testing.assert_array_equal(loaded.eV, [10, 10.5, 11, 11.5])
    np.testing.assert_array_equal(loaded.values, images)
    assert "step" not in loaded.coords
    assert "frame_group" not in loaded.coords


def test_partial_sweep_retains_recorded_energy_and_sweeps(tmp_path):
    images = _images(5)
    path = _write_fesuma(
        tmp_path / "Data_00000.h5",
        images,
        steps=[0, 1, 2, 0, 2],
        sweeps=[0, 0, 0, 1, 1],
        group_names=["00000", "00001", "00002", "00003", "00005"],
        expected_steps=3,
        expected_sweeps=2,
    )
    loaded = _load(path)

    assert loaded.dims == ("frame", "detector_x", "detector_y")
    np.testing.assert_array_equal(loaded.sweep, [0, 0, 0, 1, 1])
    np.testing.assert_array_equal(loaded.eV, [10, 10.5, 11, 10, 11])
    np.testing.assert_array_equal(loaded.values, images)
    assert "step" not in loaded.coords
    assert "frame_group" not in loaded.coords


def test_metadata_only_reads_no_images(tmp_path, monkeypatch):
    path = _write_fesuma(tmp_path / "Data_00000.h5", _images(4))
    expected = _load(path)
    with monkeypatch.context() as patch:
        _forbid_image_reads(patch)
        metadata = _load(path, without_values=True)

    assert metadata.sizes == expected.sizes
    assert metadata.dtype == expected.dtype
    xr.testing.assert_identical(metadata, expected.copy(data=metadata.data))
    backing = metadata.data
    while isinstance(backing.base, np.ndarray):
        backing = backing.base
    assert backing.size == 1


def test_loaded_array_hdf5_roundtrip(tmp_path):
    path = _write_fesuma(tmp_path / "Data_00000.h5", _images(3))
    loaded = _load(path)
    output = tmp_path / "loaded.h5"

    loaded.to_netcdf(output, engine="h5netcdf")
    restored = xr.load_dataarray(output, engine="h5netcdf")

    assert loaded.name == "LockIn"
    xr.testing.assert_identical(restored, loaded)
    assert "fesuma_metadata" not in loaded.attrs
    voltage_values = [-8.25, 90.75, -8.25, -15.54, 1380, 3400]
    for name, values in zip(
        _ANALYZER_VECTORS,
        [voltage_values, voltage_values, [-0.16, 1.83, -0.16, -0.3, 0, 0]],
        strict=True,
    ):
        assert name not in loaded.attrs
        for channel, value in zip(_ANALYZER_CHANNELS, values, strict=True):
            coordinate = loaded.coords[f"{name}_{channel}"]
            assert coordinate.dims == ()
            assert coordinate.item() == pytest.approx(value)
            assert coordinate.attrs["units"] == "V"


def test_omits_redundant_metadata(tmp_path):
    path = _write_fesuma(tmp_path / "Data_00000.h5", _images(3))
    loaded = _load(path)

    assert "source_file" not in loaded.attrs
    assert "frame_count" not in loaded.attrs
    assert "IGORWaveType" not in loaded.coords
    assert "IGORWaveUnits" not in loaded.coords
    assert "IGORWaveScaling" not in loaded.coords
    assert "fesuma_metadata" not in loaded.attrs

    def check_metadata(value):
        if isinstance(value, dict):
            assert "IGORWaveType" not in value
            assert "IGORWaveUnits" not in value
            assert "IGORWaveScaling" not in value
            assert "dataset_attributes" not in value
            for child in value.values():
                check_metadata(child)
        elif isinstance(value, list):
            for child in value:
                check_metadata(child)

    check_metadata(loaded.attrs)
    for coordinate in loaded.coords.values():
        check_metadata(coordinate.attrs)


@pytest.mark.parametrize("layout", ["regular", "sweeps", "partial"])
def test_generic_vectors_follow_scan_layout(tmp_path, monkeypatch, layout):
    count, scan, scan_dims, scan_shape = _scan_layout(layout)
    path = _write_fesuma(tmp_path / "Data_00000.h5", _images(count), **scan)
    positions = np.tile([1.0, 2.0, 3.0, 4.0], (count, 1))
    positions[:, 0] += np.arange(count)
    with h5py.File(path, "r+") as file:
        group_names = scan.get(
            "group_names", [f"{index:05d}" for index in range(count)]
        )
        for group_name, position in zip(group_names, positions, strict=True):
            group = file[group_name]
            group["ManipulatorPosition"] = position
            group["ManipulatorPosition"].attrs["IGORWaveDimensionLabels"] = np.array(
                [[""], ["X"], ["Y"], ["Z"], ["POLAR"]],
                dtype=h5py.string_dtype(),
            )
            group["ManipulatorScaleFactors"] = np.ones(6)
    loaded = _load(path)

    np.testing.assert_array_equal(loaded.attrs["ManipulatorScaleFactors"], np.ones(6))
    assert "ManipulatorPosition" not in loaded.attrs
    assert "fesuma_metadata" not in loaded.attrs
    for index, label in enumerate(("X", "Y", "Z", "POLAR")):
        coordinate = loaded.coords[f"ManipulatorPosition_{label}"]
        assert coordinate.dims == scan_dims
        assert "IGORWaveDimensionLabels" not in coordinate.attrs
        np.testing.assert_array_equal(
            coordinate, positions[:, index].reshape(scan_shape)
        )
    assert loaded.SaveTimestamp.dims == scan_dims
    np.testing.assert_array_equal(
        loaded.SaveTimestamp,
        np.array(
            [f"2026-06-30T06:49:{index:02d}.0Z" for index in range(count)]
        ).reshape(scan_shape),
    )
    with monkeypatch.context() as patch:
        _forbid_image_reads(patch)
        metadata = _load(path, without_values=True)
        lazy = _load(path, chunks=1)
    xr.testing.assert_identical(metadata, loaded.copy(data=metadata.data))
    xr.testing.assert_identical(lazy.compute(), loaded)
    output = tmp_path / "loaded.h5"
    loaded.to_netcdf(output, engine="h5netcdf")
    xr.testing.assert_identical(xr.load_dataarray(output, engine="h5netcdf"), loaded)


@pytest.mark.parametrize("layout", ["regular", "sweeps", "partial"])
def test_missing_generic_vector_preserves_scan_layout(tmp_path, layout):
    count, scan, scan_dims, scan_shape = _scan_layout(layout)
    path = _write_fesuma(tmp_path / "Data_00000.h5", _images(count), **scan)
    positions = np.full((count, 4), np.nan)
    positions[1] = [1.0, 2.0, 3.0, 4.0]
    group_names = scan.get("group_names", [f"{index:05d}" for index in range(count)])
    with h5py.File(path, "r+") as file:
        dataset = file[group_names[1]].create_dataset(
            "ManipulatorPosition", data=positions[1]
        )
        dataset.attrs["IGORWaveDimensionLabels"] = np.array(
            [[""], ["X"], ["Y"], ["Z"], ["POLAR"]],
            dtype=h5py.string_dtype(),
        )
    loaded = _load(path)

    assert "ManipulatorPosition" not in loaded.attrs
    assert "fesuma_metadata" not in loaded.attrs
    for index, label in enumerate(("X", "Y", "Z", "POLAR")):
        coordinate = loaded.coords[f"ManipulatorPosition_{label}"]
        assert coordinate.dims == scan_dims
        np.testing.assert_allclose(
            coordinate, positions[:, index].reshape(scan_shape), equal_nan=True
        )


def test_unlabeled_generic_vectors_use_component_coordinates(tmp_path):
    path = _write_fesuma(tmp_path / "Data_00000.h5", _images(3))
    scale_factors = 1.0 + 0.1 * np.arange(18).reshape(3, 6)
    with h5py.File(path, "r+") as file:
        for index, factors in enumerate(scale_factors):
            file[f"{index:05d}/ManipulatorScaleFactors"] = factors
    loaded = _load(path)

    assert "ManipulatorScaleFactors" not in loaded.attrs
    assert "fesuma_metadata" not in loaded.attrs
    for index in range(6):
        coordinate = loaded.coords[f"ManipulatorScaleFactors_{index}"]
        assert coordinate.dims == ("eV",)
        np.testing.assert_array_equal(coordinate, scale_factors[:, index])


def test_missing_timestamp_uses_nan_and_preserves_roundtrip(tmp_path):
    path = _write_fesuma(tmp_path / "Data_00000.h5", _images(3))
    with h5py.File(path, "r+") as file:
        del file["00001/SaveTimestamp"]
    loaded = _load(path)

    assert loaded.SaveTimestamp.dims == ("eV",)
    assert np.isnan(loaded.SaveTimestamp.values[1])
    assert "fesuma_metadata" not in loaded.attrs
    output = tmp_path / "loaded.h5"
    loaded.to_netcdf(output, engine="h5netcdf")
    xr.testing.assert_identical(xr.load_dataarray(output, engine="h5netcdf"), loaded)


@pytest.mark.parametrize("layout", ["regular", "sweeps", "partial"])
def test_analyzer_vectors_follow_scan_layout(tmp_path, monkeypatch, layout):
    count, scan, scan_dims, scan_shape = _scan_layout(layout)
    path = _write_fesuma(tmp_path / "Data_00000.h5", _images(count), **scan)
    with h5py.File(path, "r+") as file:
        group_names = scan.get(
            "group_names", [f"{index:05d}" for index in range(count)]
        )
        for index, group_name in enumerate(group_names):
            group = file[group_name]
            for vector_index, name in enumerate(_ANALYZER_VECTORS):
                group[name][:] = 1000 * vector_index + 100 * index + np.arange(6)
    loaded = _load(path)

    for vector_index, name in enumerate(_ANALYZER_VECTORS):
        assert name not in loaded.attrs
        for channel_index, channel in enumerate(_ANALYZER_CHANNELS):
            coordinate = loaded.coords[f"{name}_{channel}"]
            expected = (
                1000 * vector_index + 100 * np.arange(count) + channel_index
            ).reshape(scan_shape)
            assert coordinate.dims == scan_dims
            assert coordinate.attrs["units"] == "V"
            np.testing.assert_array_equal(coordinate, expected)
    assert "fesuma_metadata" not in loaded.attrs
    with monkeypatch.context() as patch:
        _forbid_image_reads(patch)
        metadata = _load(path, without_values=True)
        lazy = _load(path, chunks=1)
    xr.testing.assert_identical(metadata, loaded.copy(data=metadata.data))
    xr.testing.assert_identical(lazy.compute(), loaded)
    output = tmp_path / "loaded.h5"
    loaded.to_netcdf(output, engine="h5netcdf")
    xr.testing.assert_identical(xr.load_dataarray(output, engine="h5netcdf"), loaded)


def test_analyzer_hardware_disconnect_preserves_missing_points(tmp_path):
    path = _write_fesuma(tmp_path / "Data_00000.h5", _images(3))
    with h5py.File(path, "r+") as file:
        for index in range(3):
            for name in _ANALYZER_VECTORS:
                file[f"{index:05d}/{name}"][:] = index
        for name in (*_ANALYZER_VECTORS, "AnalyzerMode"):
            del file[f"00001/{name}"]
    loaded = _load(path)

    for name in _ANALYZER_VECTORS:
        for channel in _ANALYZER_CHANNELS:
            coordinate = loaded.coords[f"{name}_{channel}"]
            assert coordinate.dims == ("eV",)
            assert coordinate.attrs["units"] == "V"
            np.testing.assert_allclose(coordinate, [0, np.nan, 2], equal_nan=True)


@pytest.mark.parametrize("varying", [False, True])
def test_source_value_units_follow_coordinates(tmp_path, varying):
    path = _write_fesuma(tmp_path / "Data_00000.h5", _images(3))
    photon_energies = [20.0, 20.5, 21.0] if varying else [20.0] * 3
    with h5py.File(path, "r+") as file:
        for index, photon_energy in enumerate(photon_energies):
            group = file[f"{index:05d}"]
            group["MonochromatorEnergy"] = [photon_energy]
            group["MonochromatorEnergy"].attrs.update(
                {
                    "IGORWaveType": 4,
                    "IGORWaveUnits": np.array(["[eV]", ""], dtype=h5py.string_dtype()),
                }
            )
    loaded = _load(path)

    assert loaded.hv.attrs["units"] == "eV"
    if varying:
        np.testing.assert_array_equal(loaded.hv, photon_energies)
        assert loaded.hv.dims == ("eV",)
    else:
        assert loaded.hv.item() == 20.0
        assert loaded.hv.dims == ()


@pytest.mark.parametrize("layout", ["regular", "sweeps", "partial"])
def test_missing_photon_energy_preserves_scan_layout(tmp_path, monkeypatch, layout):
    count, scan, scan_dims, scan_shape = _scan_layout(layout)
    path = _write_fesuma(tmp_path / "Data_00000.h5", _images(count), **scan)
    photon_energies = np.full(count, 20.0)
    photon_energies[1] = np.nan
    with h5py.File(path, "r+") as file:
        group_names = scan.get(
            "group_names", [f"{index:05d}" for index in range(count)]
        )
        for group_name, photon_energy in zip(group_names, photon_energies, strict=True):
            if np.isnan(photon_energy):
                continue
            dataset = file[group_name].create_dataset(
                "MonochromatorEnergy", data=[photon_energy]
            )
            dataset.attrs["IGORWaveUnits"] = np.array(
                ["[eV]", ""], dtype=h5py.string_dtype()
            )
    loaded = _load(path)

    assert loaded.hv.dims == scan_dims
    assert loaded.hv.attrs["units"] == "eV"
    assert "MonochromatorEnergy" not in loaded.attrs
    assert "fesuma_metadata" not in loaded.attrs
    np.testing.assert_allclose(
        loaded.hv, photon_energies.reshape(scan_shape), equal_nan=True
    )
    with monkeypatch.context() as patch:
        _forbid_image_reads(patch)
        metadata = _load(path, without_values=True)
        lazy = _load(path, chunks=1)
    xr.testing.assert_identical(metadata, loaded.copy(data=metadata.data))
    xr.testing.assert_identical(lazy.compute(), loaded)
    output = tmp_path / "loaded.h5"
    loaded.to_netcdf(output, engine="h5netcdf")
    xr.testing.assert_identical(xr.load_dataarray(output, engine="h5netcdf"), loaded)


@pytest.mark.parametrize("chunks", [{}, "auto", 2, -1, {"eV": 2, "detector_x": 1}])
def test_lazy_loading_and_serialization(tmp_path, monkeypatch, chunks):
    path = _write_fesuma(tmp_path / "Data_00000.h5", _images(4))
    expected = _load(path)
    with monkeypatch.context() as patch:
        _forbid_image_reads(patch)
        loaded = _load(path, chunks=chunks)

    assert loaded.chunks is not None
    restored = cloudpickle.loads(cloudpickle.dumps(loaded))
    xr.testing.assert_identical(loaded.compute(), expected)
    xr.testing.assert_identical(restored.isel(eV=1).compute(), expected.isel(eV=1))


def test_lazy_selection_reads_only_selected_frame(tmp_path, monkeypatch):
    images = _images(4)
    path = _write_fesuma(tmp_path / "Data_00000.h5", images)
    loaded = _load(path, chunks=1)
    image_reads = []
    original_getitem = h5py.Dataset.__getitem__
    original_read_direct = h5py.Dataset.read_direct

    def getitem(dataset, *args, **kwargs):
        if dataset.name.endswith("/analysisImage"):
            image_reads.append(dataset.name)
        return original_getitem(dataset, *args, **kwargs)

    def read_direct(dataset, *args, **kwargs):
        if dataset.name.endswith("/analysisImage"):
            image_reads.append(dataset.name)
        return original_read_direct(dataset, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", getitem)
    monkeypatch.setattr(h5py.Dataset, "read_direct", read_direct)
    selected = loaded.isel(eV=1).compute()

    np.testing.assert_array_equal(selected.values, images[1])
    assert set(image_reads) == {"/00001/analysisImage"}


@pytest.mark.parametrize("invalid", ["shape", "scale"])
def test_invalid_frame_metadata(tmp_path, invalid):
    path = _write_fesuma(tmp_path / "Data_00000.h5", _images(2))
    with h5py.File(path, "r+") as file:
        if invalid == "shape":
            attributes = dict(file["00001/analysisImage"].attrs)
            del file["00001/analysisImage"]
            dataset = file.create_dataset("00001/analysisImage", data=np.zeros((3, 3)))
            dataset.attrs.update(attributes)
        else:
            file["00001/analysisImage"].attrs["DimDeltaX"] = 0.75
    with pytest.raises(ValueError, match=r"(?i)shape|dimension|image|scale"):
        _load(path)


def test_rejects_unrelated_hdf5_file(tmp_path):
    path = tmp_path / "unrelated.h5"
    with h5py.File(path, "w") as file:
        file["counts"] = [1, 2, 3]
    with pytest.raises(ValueError, match=r"(?i)fesuma|analysisImage|acquisition"):
        _load(path)


def test_floating_fermi_surface_values_are_preserved(tmp_path):
    images = _images(3).astype(np.float64) + 0.125
    path = _write_fesuma(
        tmp_path / "Data_00000.h5", images, acquisition_mode="Fermi Surface"
    )
    with h5py.File(path, "r+") as file:
        for group in ("00000", "00001", "00002"):
            file[f"{group}/analysisImage"].attrs["AccumulationCount"] = 20.0
    loaded = _load(path)

    assert loaded.name == "Fermi Surface"
    assert loaded.dtype == np.float64
    np.testing.assert_array_equal(loaded.values, images)
    assert loaded.attrs["AcquisitionMode"] == "Fermi Surface"


def test_adjustment_returns_one_detector_image(tmp_path):
    images = _images(1)
    path = _write_fesuma(tmp_path / "Data_00000.h5", images)
    with h5py.File(path, "r+") as file:
        file["wSaveType"][0] = b"Adjustment"
        for key in list(file):
            if key.startswith("Acquisition"):
                del file[key]
        del file["00000/AcquisitionCurrentStep"]
        del file["00000/AcquisitionCurrentSweep"]
    loaded = _load(path)

    assert loaded.name == "analysisImage"
    assert loaded.dims == ("detector_x", "detector_y")
    np.testing.assert_array_equal(loaded.values, images[0])
    assert "eV" not in loaded.coords


def test_non_ekin_monochromator_readback_is_auxiliary(tmp_path, monkeypatch):
    images = _images(3)
    path = _write_fesuma(
        tmp_path / "Data_00000.h5", images, acquisition_coordinate="MONO"
    )
    with h5py.File(path, "r+") as file:
        for index in range(3):
            file[f"{index:05d}/MonochromatorEnergy"] = [20 + 0.5 * index]
            file[f"{index:05d}/MonochromatorEnergy"].attrs["IGORWaveUnits"] = np.array(
                ["[eV]", ""], dtype=h5py.string_dtype()
            )
    loaded = _load(path)

    assert loaded.dims == ("frame", "detector_x", "detector_y")
    assert loaded.hv.dims == ("frame",)
    np.testing.assert_array_equal(loaded.hv, [20, 20.5, 21])
    assert loaded.hv.attrs["units"] == "eV"
    assert "eV" not in loaded.coords
    np.testing.assert_array_equal(loaded.values, images)
    with monkeypatch.context() as patch:
        _forbid_image_reads(patch)
        metadata = _load(path, without_values=True)
        lazy = _load(path, chunks={})
    xr.testing.assert_identical(metadata, loaded.copy(data=metadata.data))
    xr.testing.assert_identical(lazy.compute(), loaded)


@pytest.mark.parametrize("mode", ["MONO", "Manipulator", "TIMED", "TABLE"])
def test_non_ekin_modes_preserve_record_order(tmp_path, mode):
    images = _images(4)
    path = _write_fesuma(
        tmp_path / "Data_00000.h5",
        images,
        acquisition_coordinate=mode,
        steps=[0, 1, 0, 1],
        sweeps=[0, 0, 1, 1],
        expected_steps=2,
        expected_sweeps=2,
    )
    readbacks = [0.2, 0.6, 0.4, 0.8]
    with h5py.File(path, "r+") as file:
        for index, value in enumerate(readbacks):
            file[f"{index:05d}/MonochromatorEnergy"] = [value]
            file[f"{index:05d}/MonochromatorEnergy"].attrs["IGORWaveUnits"] = np.array(
                ["[eV]", ""], dtype=h5py.string_dtype()
            )
    loaded = _load(path)

    assert loaded.dims == ("frame", "detector_x", "detector_y")
    np.testing.assert_array_equal(loaded.frame, [0, 1, 2, 3])
    np.testing.assert_array_equal(loaded.sweep, [0, 0, 1, 1])
    np.testing.assert_array_equal(loaded.hv, readbacks)
    assert loaded.hv.dims == ("frame",)
    assert loaded.hv.attrs["units"] == "eV"
    assert loaded.attrs["AcquisitionCoordinate"] == mode
    assert loaded.attrs["AcquisitionNumberOfSweeps"] == 2
    assert "eV" not in loaded.coords
    assert "time" not in loaded.coords
    assert "manipulator_x" not in loaded.coords
    assert "step" not in loaded.coords
    assert "frame_group" not in loaded.coords
    np.testing.assert_array_equal(loaded.values, images)


@pytest.mark.parametrize("steps", [[0, 1, 2], [0, 2, 4]])
def test_descending_and_gapped_energy(tmp_path, steps):
    images = _images(3)
    path = _write_fesuma(
        tmp_path / "Data_00000.h5",
        images,
        steps=steps,
        group_names=[f"{step:05d}" for step in steps],
        expected_steps=5 if steps == [0, 2, 4] else 3,
    )
    with h5py.File(path, "r+") as file:
        file["AcquisitionEkinStep"][0] = -0.5
        file["AcquisitionEkinStop"][0] = 7.5 if steps == [0, 2, 4] else 8.5
    loaded = _load(path)

    np.testing.assert_array_equal(loaded.eV, 10 - 0.5 * np.asarray(steps))
    np.testing.assert_array_equal(loaded.values, images)
    assert "step" not in loaded.coords
    assert "frame_group" not in loaded.coords
    if steps == [0, 1, 2]:
        assert loaded.dims[0] == "eV"
    else:
        assert loaded.dims[0] == "frame"


def test_lazy_multiple_sweeps(tmp_path):
    images = _images(6)
    path = _write_fesuma(
        tmp_path / "Data_00000.h5",
        images,
        steps=[0, 1, 2, 0, 1, 2],
        sweeps=[0, 0, 0, 1, 1, 1],
        expected_steps=3,
        expected_sweeps=2,
    )
    expected = _load(path)
    loaded = _load(path, chunks={"eV": 2, "sweep": 1})

    assert loaded.chunks is not None
    assert loaded.dims == ("sweep", "eV", "detector_x", "detector_y")
    xr.testing.assert_identical(loaded.compute(), expected)


@pytest.mark.parametrize("chunks", [0, -2, True, {"unknown": 1}, {"eV": 0}])
def test_invalid_chunks(tmp_path, chunks):
    path = _write_fesuma(tmp_path / "Data_00000.h5", _images(2))
    with pytest.raises(ValueError, match=r"(?i)chunk|dimension"):
        _load(path, chunks=chunks)
