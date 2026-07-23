from __future__ import annotations

import typing

import numpy as np
import pytest
import xarray as xr

import erlab.interactive.imagetool.manager._widgets as manager_widgets

if typing.TYPE_CHECKING:
    import pathlib


@pytest.fixture(scope="module")
def test_data() -> xr.DataArray:
    return xr.DataArray(
        np.arange(25).reshape((5, 5)),
        dims=["alpha", "eV"],
        coords={"alpha": np.arange(5), "eV": np.arange(5)},
    )


@pytest.fixture(autouse=True)
def isolated_recent_workspace_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> pathlib.Path:
    settings_path = tmp_path / "recent-workspaces.ini"
    monkeypatch.setenv(
        manager_widgets._MANAGER_SETTINGS_PATH_ENV_VAR,
        str(settings_path),
    )
    return settings_path
