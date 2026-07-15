from __future__ import annotations

import erlab.utils.misc
from erlab.utils.misc import accepts_kwarg


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
