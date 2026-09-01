"""Deterministic data files for the ImageTool Manager tutorial."""

from __future__ import annotations

import contextlib
import os
import pathlib
import tempfile
import typing
from dataclasses import dataclass

import numpy as np
import xarray as xr

from erlab.io.dataloader import LoaderBase, LoaderRegistry
from erlab.io.exampledata import generate_data_angles

if typing.TYPE_CHECKING:
    import datetime
    from collections.abc import Callable, Iterable, Iterator


class TutorialDataGenerationCancelled(Exception):
    """Raised when tutorial data generation is cancelled."""


@dataclass(frozen=True)
class TutorialDataFiles:
    """Paths of the generated tutorial data files."""

    map: pathlib.Path
    cut: pathlib.Path


_loader_registry = LoaderRegistry.instance()
with _loader_registry._lock:
    _loaders_before_class_definition = _loader_registry._loaders.copy()
    _aliases_before_class_definition = _loader_registry._alias_mapping.copy()


class _TutorialLoader(LoaderBase):
    """Load the raw files that are used only by the tutorial."""

    name = "tutorial"
    display_name = "ERLab tutorial data"
    description = "ERLab tutorial data"
    extensions: typing.ClassVar[set[str]] = {".h5"}
    name_map: typing.ClassVar[dict[str, str | Iterable[str]]] = {
        "alpha": "ThetaX",
        "beta": "Polar",
        "eV": "KineticEnergy",
        "hv": "PhotonEnergy",
        "xi": "Tilt",
        "delta": "Azimuth",
    }
    additional_attrs: typing.ClassVar[
        dict[
            str,
            str
            | float
            | datetime.datetime
            | Callable[[xr.DataArray], str | float | datetime.datetime],
        ]
    ] = {
        "configuration": 1,
        "sample_temp": 20.0,
        "sample_workfunction": 4.5,
        "angle_resolution": 0.1,
    }

    def load_single(
        self, file_path: str | os.PathLike, *, without_values: bool = False
    ) -> xr.DataArray:
        with xr.open_dataarray(file_path, engine="h5netcdf") as source:
            data = source.load()
        if without_values:
            data = xr.zeros_like(data)
        return data


# Class creation registers LoaderBase subclasses. Restore the exact state immediately
# so that importing this module does not retain the tutorial-only loader.
with _loader_registry._lock:
    _loader_registry._loaders.clear()
    _loader_registry._loaders.update(_loaders_before_class_definition)
    _loader_registry._alias_mapping.clear()
    _loader_registry._alias_mapping.update(_aliases_before_class_definition)
del _loaders_before_class_definition, _aliases_before_class_definition


def _check_cancelled(is_cancelled: Callable[[], bool] | None) -> None:
    if is_cancelled is not None and is_cancelled():
        raise TutorialDataGenerationCancelled


def _raw_tutorial_data(*, is_map: bool) -> xr.DataArray:
    beta_range = (-13.5, 13.5) if is_map else (-1.5, -1.5)
    data = generate_data_angles(
        (160, 48 if is_map else 1, 128),
        angrange={"alpha": (-18.0, 18.0), "beta": beta_range},
        Erange=(-0.45, 0.12),
        hv=50.0,
        configuration=1,
        normal_emission=(2.0, -1.5),
        delta_offset=-4.0,
        band_rotation=-30.0,
        seed=1,
        assign_attributes=False,
    ).astype(np.float32)
    data = data.assign_coords(eV=data.eV + 45.5).rename(
        {
            "alpha": "ThetaX",
            "beta": "Polar",
            "eV": "KineticEnergy",
            "hv": "PhotonEnergy",
            "xi": "Tilt",
            "delta": "Azimuth",
        }
    )
    return data.rename("example_map" if is_map else "example_cut")


def _write_atomic(
    data: xr.DataArray,
    destination: pathlib.Path,
    *,
    is_cancelled: Callable[[], bool] | None,
    on_file_published: Callable[[pathlib.Path], None] | None,
) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary_path = pathlib.Path(temporary_name)
    try:
        data.to_netcdf(temporary_path, engine="h5netcdf")
        _check_cancelled(is_cancelled)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    if on_file_published is not None:
        on_file_published(destination)


def generate_tutorial_data_files(
    directory: str | pathlib.Path,
    *,
    is_cancelled: Callable[[], bool] | None = None,
    on_file_published: Callable[[pathlib.Path], None] | None = None,
) -> TutorialDataFiles:
    """Generate and atomically publish the tutorial data files."""
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = TutorialDataFiles(
        map=directory / "tutorial_map.h5",
        cut=directory / "tutorial_cut.h5",
    )

    _check_cancelled(is_cancelled)
    map_data = _raw_tutorial_data(is_map=True)
    _check_cancelled(is_cancelled)
    _write_atomic(
        map_data,
        files.map,
        is_cancelled=is_cancelled,
        on_file_published=on_file_published,
    )

    _check_cancelled(is_cancelled)
    cut_data = _raw_tutorial_data(is_map=False)
    _check_cancelled(is_cancelled)
    _write_atomic(
        cut_data,
        files.cut,
        is_cancelled=is_cancelled,
        on_file_published=on_file_published,
    )
    return files


@contextlib.contextmanager
def tutorial_loader_registration() -> Iterator[_TutorialLoader]:
    """Register the tutorial loader and restore the prior registry state on exit."""
    registry = LoaderRegistry.instance()
    with registry._lock:
        previous_loaders = registry._loaders.copy()
        previous_aliases = registry._alias_mapping.copy()
        registry._register(_TutorialLoader)
    previous_current = registry.current_loader
    try:
        yield typing.cast("_TutorialLoader", registry["tutorial"])
    finally:
        with registry._lock:
            registry._loaders.clear()
            registry._loaders.update(previous_loaders)
            registry._alias_mapping.clear()
            registry._alias_mapping.update(previous_aliases)
        registry.current_loader = previous_current
