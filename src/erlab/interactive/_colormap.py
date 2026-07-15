"""Colormap name resolution without a Qt dependency."""

from __future__ import annotations

import contextlib
import contextvars
import importlib

_ALL_COLORMAPS_LOADED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "all_colormaps_loaded", default=False
)


def _matplotlib_has_colormap(name: str) -> bool:
    import matplotlib as mpl

    return name in mpl.colormaps


def _colorcet_matplotlib_candidate(name: str) -> str:
    return name if name.startswith("cet_") else f"cet_{name}"


def matplotlib_colormap_name(name: str) -> str:
    """Return the Matplotlib colormap name matching a display name."""
    if not name or _matplotlib_has_colormap(name):
        return name

    candidate = _colorcet_matplotlib_candidate(name)
    if _matplotlib_has_colormap(candidate):
        return candidate

    if not _ALL_COLORMAPS_LOADED.get():
        with contextlib.suppress(Exception):
            load_all_colormaps()
        if _matplotlib_has_colormap(name):
            return name
        if _matplotlib_has_colormap(candidate):
            return candidate

    return name


def load_all_colormaps() -> None:
    """Load colormaps from packages enabled in interactive preferences."""
    if _ALL_COLORMAPS_LOADED.get():
        return

    import erlab.plotting

    for package in erlab.interactive.options.model.colors.cmap.packages:
        if importlib.util.find_spec(package):
            importlib.import_module(package)
    _ALL_COLORMAPS_LOADED.set(True)
