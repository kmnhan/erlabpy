from __future__ import annotations

import contextlib
import inspect
import os
import pathlib
import types

import pytest
import xarray as xr

import erlab
import erlab.utils.misc
from erlab.utils.misc import accepts_kwarg


@pytest.mark.parametrize("package", [erlab, xr], ids=["erlab", "xarray"])
def test_user_level_warning_skips_package_and_stdlib_wrappers(package) -> None:
    filename = pathlib.Path(package.__file__).parent / "_warning_wrapper.py"

    def issue_warning():
        erlab.utils.misc.emit_user_level_warning("User warning", UserWarning)
        yield

    wrapped = contextlib.contextmanager(
        types.FunctionType(
            issue_warning.__code__.replace(co_filename=str(filename)), globals()
        )
    )

    expected_line = inspect.currentframe().f_lineno + 1
    with pytest.warns(UserWarning, match="User warning") as recorded, wrapped():
        pass

    assert recorded[0].filename == __file__
    assert recorded[0].lineno == expected_line


def test_user_level_warning_preserves_third_party_source() -> None:
    filename = pathlib.Path(os.__file__).parent / "site-packages" / "other.py"

    def issue_warning():
        erlab.utils.misc.emit_user_level_warning("Third-party warning", FutureWarning)

    wrapped = types.FunctionType(
        issue_warning.__code__.replace(co_filename=str(filename), co_firstlineno=1),
        globals(),
    )

    with pytest.warns(FutureWarning, match="Third-party warning") as recorded:
        wrapped()

    assert recorded[0].filename == str(filename)
    assert recorded[0].lineno == 2


def test_user_level_warning_without_stdlib_location(monkeypatch) -> None:
    erlab.utils.misc._warning_skip_file_prefixes.cache_clear()
    monkeypatch.setattr(os, "__file__", None)
    try:
        with pytest.warns(UserWarning, match="User warning") as recorded:
            erlab.utils.misc.emit_user_level_warning("User warning")
        assert recorded[0].filename == __file__
    finally:
        erlab.utils.misc._warning_skip_file_prefixes.cache_clear()


def test_python_identifier_utilities() -> None:
    assert erlab.utils.misc._is_valid_identifier("valid_name")
    assert not erlab.utils.misc._is_valid_identifier("with space")
    assert not erlab.utils.misc._is_valid_identifier("for")
    assert not erlab.utils.misc._is_valid_identifier(None)

    assert erlab.utils.misc._normalize_identifier("1 invalid") == "_1_invalid"
    assert erlab.utils.misc._normalize_identifier("for") == "for_"
    assert erlab.utils.misc._normalize_identifier("___") == "var"
    assert erlab.utils.misc._normalize_identifier("a²") == "a2"
    assert erlab.utils.misc._normalize_identifier(None) == "var"


def test_accepts_kwarg_explicit_positional_or_keyword() -> None:
    def func(a, b=1) -> None:
        return None

    assert accepts_kwarg(func, "b")


def test_accepts_kwarg_explicit_keyword_only() -> None:
    def func(a, *, c=1) -> None:
        return None

    assert accepts_kwarg(func, "c")


def test_accepts_kwarg_positional_only() -> None:
    def func(a, /, b) -> None:
        return None

    assert not accepts_kwarg(func, "a")
    assert accepts_kwarg(func, "b")


def test_accepts_kwarg_requires_explicit_when_strict() -> None:
    def func(**kwargs) -> None:
        return None

    assert not accepts_kwarg(func, "missing")


def test_accepts_kwarg_allows_kwargs_when_not_strict() -> None:
    def func(**kwargs) -> None:
        return None

    assert accepts_kwarg(func, "missing", strict=False)


def test_accepts_kwarg_missing_parameter() -> None:
    def func(a, b) -> None:
        return None

    assert not accepts_kwarg(func, "c")
