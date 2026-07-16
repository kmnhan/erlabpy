from __future__ import annotations

import typing

import numpy as np
import pytest
import xarray as xr
from qtpy import QtCore

import erlab.interactive.imagetool.manager._widgets as manager_widgets
from erlab.interactive.imagetool.manager._figurecomposer import _collection
from erlab.interactive.imagetool.manager._workspace import (
    _controller as workspace_controller,
)

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

    def _settings() -> QtCore.QSettings:
        return QtCore.QSettings(str(settings_path), QtCore.QSettings.Format.IniFormat)

    _settings().clear()
    monkeypatch.setattr(_collection, "_manager_settings", _settings)
    monkeypatch.setattr(manager_widgets, "_manager_settings", _settings)
    monkeypatch.setattr(workspace_controller, "_manager_settings", _settings)
    return settings_path
