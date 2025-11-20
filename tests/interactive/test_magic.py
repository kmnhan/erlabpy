import contextlib

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


@pytest.fixture
def ip_shell():
    from IPython.testing.globalipapp import start_ipython

    ip_session = start_ipython()
    ip_session.run_line_magic("load_ext", "erlab.interactive")

    yield ip_session

    ip_session.run_line_magic("unload_ext", "erlab.interactive")
    ip_session.user_ns.clear()
    ip_session.clear_instance()
    with contextlib.suppress(AttributeError):
        del start_ipython.already_called


@pytest.mark.parametrize(
    ("magic_name", "tool_attr", "line", "expected_name", "expected_fn"),
    [
        (
            "ktool",
            "ktool",
            "--cmap plasma darr.sel(eV=1)",
            "darr.sel(eV=1)",
            _sel_eV1,
        ),
        ("dtool", "dtool", "darr", "darr", _identity),
        (
            "goldtool",
            "goldtool",
            "darr.isel(alpha=1)",
            "darr.isel(alpha=1)",
            _isel_alpha1,
        ),
        ("restool", "restool", "darr.isel(eV=0)", "darr.isel(eV=0)", _isel_eV0),
    ],
)
def test_interactive_tool_magics_forward_data(
    ip_shell, monkeypatch, magic_name, tool_attr, line, expected_name, expected_fn
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
        return f"{tool_attr}-result"

    monkeypatch.setattr(interactive_mod, tool_attr, fake_tool, raising=False)

    result = ip_shell.run_line_magic(magic_name, line)

    assert result == f"{tool_attr}-result"
    call_kwargs = calls[-1]
    xr.testing.assert_identical(call_kwargs["data"], expected_fn(darr))
    assert call_kwargs["data_name"] == expected_name

    if tool_attr == "ktool":
        assert call_kwargs["cmap"] == "plasma"
    else:
        assert "cmap" not in call_kwargs
