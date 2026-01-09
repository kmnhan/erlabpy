from __future__ import annotations

from erlab.utils.misc import accepts_kwarg


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
