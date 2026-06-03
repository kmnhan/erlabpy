"""Matplotlib stylesheet availability helpers for interactive tools."""

from __future__ import annotations

import contextlib
import functools
import importlib
import pathlib
import typing

from matplotlib import style as mpl_style

if typing.TYPE_CHECKING:
    from collections.abc import Iterable

_ERLAB_REGISTERED_STYLESHEETS: set[str] = set()


@functools.cache
def _erlab_stylesheet_names() -> frozenset[str]:
    style_dir = pathlib.Path(__file__).resolve().parent.parent / "plotting" / "stylelib"
    return frozenset(path.stem for path in style_dir.glob("*.mplstyle"))


def _stylesheet_name_set() -> frozenset[str]:
    return frozenset(str(name) for name in mpl_style.available)


@functools.cache
def load_erlab_plotting_stylesheets() -> None:
    """Import ERLab plotting once so its bundled stylesheets are registered."""
    with contextlib.suppress(Exception):
        importlib.import_module("erlab.plotting")


def available_stylesheets(names: Iterable[str] = ()) -> frozenset[str]:
    """Return available stylesheets, loading ERLab plotting if requested names miss."""
    names = tuple(names)
    available = _stylesheet_name_set()
    if names and any(name not in available for name in names):
        before_load = available
        load_erlab_plotting_stylesheets()
        available = _stylesheet_name_set()
        _ERLAB_REGISTERED_STYLESHEETS.update(
            name for name in names if name not in before_load and name in available
        )
    return available


def stylesheets_require_erlab_plotting(names: Iterable[str]) -> bool:
    """Return whether ERLab plotting registers a requested stylesheet."""
    names = tuple(names)
    if not names:
        return False
    if any(name in _erlab_stylesheet_names() for name in names):
        return True
    if any(name in _ERLAB_REGISTERED_STYLESHEETS for name in names):
        return True
    available = _stylesheet_name_set()
    missing = tuple(name for name in names if name not in available)
    if not missing:
        return False
    load_erlab_plotting_stylesheets()
    available = _stylesheet_name_set()
    registered = tuple(name for name in missing if name in available)
    _ERLAB_REGISTERED_STYLESHEETS.update(registered)
    return bool(registered)


def sorted_available_stylesheets(names: Iterable[str] = ()) -> list[str]:
    """Return sorted available stylesheets for option editors."""
    return sorted(available_stylesheets(names))
