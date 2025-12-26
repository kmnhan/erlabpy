import IPython
import numpy as np
import pytest
import xarray as xr

import erlab.interactive as interactive_mod


def _identity(data):
    return data


def _sel_eV1(data):
    return data.sel(eV=1)


def _isel_alpha1(data):
    return data.isel(alpha=1)


def _isel_eV0(data):
    return data.isel(eV=0)


@pytest.mark.parametrize(
    ("magic_name", "line", "expected_name", "expected_fn"),
    [
        ("ktool", "--cmap plasma darr.sel(eV=1)", "darr.sel(eV=1)", _sel_eV1),
        ("dtool", "darr", "darr", _identity),
        ("goldtool", "darr.isel(alpha=1)", "darr.isel(alpha=1)", _isel_alpha1),
        ("restool", "darr.isel(eV=0)", "darr.isel(eV=0)", _isel_eV0),
        ("meshtool", "darr", "darr", _identity),
        ("ftool", "darr", "darr", _identity),
    ],
)
def test_interactive_tool_magics_forward_data(
    ip_shell: IPython.InteractiveShell,
    monkeypatch,
    magic_name,
    line,
    expected_name,
    expected_fn,
):
    darr = xr.DataArray(
        np.arange(12).reshape((3, 4)),
        dims=["alpha", "eV"],
        coords={"alpha": np.arange(3), "eV": np.arange(4)},
    )
    ip_shell.user_ns["darr"] = darr

    calls = []

    def fake_tool(**kwargs):
        calls.append(kwargs)
        return f"{magic_name}-result"

    monkeypatch.setattr(interactive_mod, magic_name, fake_tool, raising=False)

    result = ip_shell.run_line_magic(magic_name, line)

    assert result == f"{magic_name}-result"
    call_kwargs = calls[-1]
    xr.testing.assert_identical(call_kwargs["data"], expected_fn(darr))
    assert call_kwargs["data_name"] == expected_name

    if magic_name == "ktool":
        assert call_kwargs["cmap"] == "plasma"
    else:
        assert "cmap" not in call_kwargs
