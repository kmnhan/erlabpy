import pathlib

import numpy as np
import pytest
import xarray as xr

import erlab.io.dataloader as dataloader
from erlab.interactive.imagetool.manager._tutorial.data import (
    TutorialDataGenerationCancelled,
    generate_tutorial_data_files,
    tutorial_loader_registration,
)


@pytest.fixture(scope="module")
def tutorial_files(tmp_path_factory):
    published = []
    files = generate_tutorial_data_files(
        tmp_path_factory.mktemp("tutorial-data"),
        on_file_published=published.append,
    )
    assert published == [files.map, files.cut]
    return files


def test_tutorial_raw_files(tutorial_files) -> None:
    raw_map = xr.load_dataarray(tutorial_files.map, engine="h5netcdf")
    raw_cut = xr.load_dataarray(tutorial_files.cut, engine="h5netcdf")

    assert raw_map.name == "example_map"
    assert raw_map.dims == ("ThetaX", "Polar", "KineticEnergy")
    assert raw_map.shape == (160, 48, 128)
    assert raw_map.dtype == np.float32
    np.testing.assert_allclose(raw_map.ThetaX[[0, -1]], (-18.0, 18.0))
    np.testing.assert_allclose(raw_map.Polar[[0, -1]], (-13.5, 13.5))
    np.testing.assert_allclose(raw_map.KineticEnergy[[0, -1]], (45.05, 45.62))
    assert float(raw_map.PhotonEnergy) == 50.0
    assert float(raw_map.Tilt) == 0.0
    assert float(raw_map.Azimuth) == 0.0

    assert raw_cut.name == "dispersion_cut"
    assert raw_cut.dims == ("ThetaX", "KineticEnergy")
    assert raw_cut.shape == (160, 128)
    assert raw_cut.dtype == np.float32
    np.testing.assert_allclose(raw_cut.ThetaX[[0, -1]], (-18.0, 18.0))
    np.testing.assert_allclose(raw_cut.KineticEnergy[[0, -1]], (45.05, 45.62))
    assert float(raw_cut.Polar) == -1.5
    assert tutorial_files.map.read_bytes()[:8] == b"\x89HDF\r\n\x1a\n"
    assert tutorial_files.cut.read_bytes()[:8] == b"\x89HDF\r\n\x1a\n"


def test_tutorial_loader_and_registry_cleanup(tutorial_files) -> None:
    registry = dataloader.LoaderRegistry.instance()
    previous_loaders = registry._loaders.copy()
    previous_aliases = registry._alias_mapping.copy()

    with tutorial_loader_registration() as loader:
        assert loader.name == "tutorial"
        assert loader.display_name == "ERLab tutorial data"
        loaded = loader.load(tutorial_files.map)
        assert loaded.dims == ("alpha", "beta", "eV")
        np.testing.assert_allclose(loaded.eV[[0, -1]], (45.05, 45.62))
        assert loaded.attrs["sample_temp"] == 20.0
        assert loaded.attrs["sample_workfunction"] == 4.5
        assert loaded.attrs["angle_resolution"] == 0.1
        assert dict(loaded.kspace.offsets) == {
            "delta": 0.0,
            "xi": 0.0,
            "beta": 0.0,
        }
        corrected = loaded.assign_coords(eV=loaded.eV - 45.5)
        corrected.kspace.set_normal(2.0, -1.5, delta=-4.0)
        np.testing.assert_allclose(
            corrected.kspace._forward_func(2.0, -1.5), 0.0, atol=1e-14
        )

    assert registry._loaders == previous_loaders
    assert registry._alias_mapping == previous_aliases
    if "tutorial" not in previous_aliases:
        with pytest.raises(dataloader.LoaderNotFoundError):
            registry.get("tutorial")


def test_tutorial_loader_registry_cleanup_after_failure() -> None:
    registry = dataloader.LoaderRegistry.instance()
    previous_loaders = registry._loaders.copy()
    previous_aliases = registry._alias_mapping.copy()

    with pytest.raises(RuntimeError, match="stop"), tutorial_loader_registration():
        raise RuntimeError("stop")

    assert registry._loaders == previous_loaders
    assert registry._alias_mapping == previous_aliases


def test_tutorial_generation_is_deterministic(tmp_path: pathlib.Path) -> None:
    first = generate_tutorial_data_files(tmp_path / "first")
    second = generate_tutorial_data_files(tmp_path / "second")

    xr.testing.assert_identical(
        xr.load_dataarray(first.map, engine="h5netcdf"),
        xr.load_dataarray(second.map, engine="h5netcdf"),
    )
    xr.testing.assert_identical(
        xr.load_dataarray(first.cut, engine="h5netcdf"),
        xr.load_dataarray(second.cut, engine="h5netcdf"),
    )


def test_tutorial_generation_cancellation(tmp_path: pathlib.Path) -> None:
    with pytest.raises(TutorialDataGenerationCancelled):
        generate_tutorial_data_files(tmp_path, is_cancelled=lambda: True)

    assert list(tmp_path.iterdir()) == []
