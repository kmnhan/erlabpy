"""Matplotlib stylesheet availability helpers for interactive tools."""

from __future__ import annotations

import contextlib
import functools
import importlib
import typing

from matplotlib import style as mpl_style

if typing.TYPE_CHECKING:
    from collections.abc import Iterable


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
        load_erlab_plotting_stylesheets()
        available = _stylesheet_name_set()
    return available


def sorted_available_stylesheets(names: Iterable[str] = ()) -> list[str]:
    """Return sorted available stylesheets for option editors."""
    return sorted(available_stylesheets(names))
