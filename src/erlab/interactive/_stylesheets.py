"""Matplotlib stylesheet availability helpers for interactive tools."""

from __future__ import annotations

import contextlib
import functools
import importlib
import pathlib
import typing
import warnings

import matplotlib as mpl
from matplotlib import style as mpl_style
from qtpy import QtCore

if typing.TYPE_CHECKING:
    from collections.abc import Iterable

_ERLAB_REGISTERED_STYLESHEETS: set[str] = set()
_USER_REGISTERED_STYLESHEETS: set[str] = set()
_USER_STYLESHEET_NAMES: frozenset[str] = frozenset()

_STYLE_APPLICATION_NAME = "ImageTool Manager"


@functools.cache
def _erlab_stylesheet_names() -> frozenset[str]:
    style_dir = pathlib.Path(__file__).resolve().parent.parent / "plotting" / "stylelib"
    return frozenset(path.stem for path in style_dir.glob("*.mplstyle"))


def _stylesheet_name_set() -> frozenset[str]:
    return frozenset(str(name) for name in mpl_style.available)


def _style_library_paths() -> list[str]:
    paths: list[str] | None = getattr(mpl_style, "USER_LIBRARY_PATHS", None)
    if paths is not None:
        return paths
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=mpl.MatplotlibDeprecationWarning)
        from matplotlib.style import core as mpl_style_core  # pragma: no cover

    return mpl_style_core.USER_LIBRARY_PATHS  # pragma: no cover


@contextlib.contextmanager
def _style_path_application_context():
    application_name = QtCore.QCoreApplication.applicationName()
    try:
        QtCore.QCoreApplication.setApplicationName(_STYLE_APPLICATION_NAME)
        yield
    finally:
        QtCore.QCoreApplication.setApplicationName(application_name)


def _app_data_directory() -> pathlib.Path | None:
    with _style_path_application_context():
        location = QtCore.QStandardPaths.writableLocation(
            QtCore.QStandardPaths.StandardLocation.AppDataLocation
        )
    return pathlib.Path(location) if location else None


def _generic_data_directory() -> pathlib.Path | None:
    location = QtCore.QStandardPaths.writableLocation(
        QtCore.QStandardPaths.StandardLocation.GenericDataLocation
    )
    if not location:
        return None
    return pathlib.Path(location) / "erlabpy" / _STYLE_APPLICATION_NAME


def user_stylesheet_directory(*, create: bool = True) -> pathlib.Path:
    """Return the ERLab user stylesheet directory."""
    base_path = _app_data_directory() or _generic_data_directory()
    if base_path is None:
        raise RuntimeError(
            "Could not determine a writable application data directory for "
            "custom Matplotlib stylesheets."
        )
    style_dir = base_path / "stylelib"
    if create:
        try:
            style_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(
                "Could not create the custom Matplotlib stylesheet directory "
                f"at {style_dir}."
            ) from exc
    return style_dir


def _stylesheet_names_in_directory(style_dir: pathlib.Path) -> frozenset[str]:
    if not style_dir.is_dir():
        return frozenset()
    return frozenset(path.stem for path in style_dir.glob("*.mplstyle"))


def load_user_stylesheets(*, reload: bool = False) -> None:
    """Register and load user-provided Matplotlib stylesheets."""
    global _USER_STYLESHEET_NAMES

    style_dir = user_stylesheet_directory()
    library_paths = _style_library_paths()
    style_dir_str = str(style_dir)
    path_added = style_dir_str not in library_paths
    if path_added:
        library_paths.append(style_dir_str)

    stylesheet_names = _stylesheet_names_in_directory(style_dir)
    if path_added or reload or stylesheet_names != _USER_STYLESHEET_NAMES:
        mpl_style.reload_library()
        stylesheet_names = _stylesheet_names_in_directory(style_dir)

    _USER_STYLESHEET_NAMES = stylesheet_names
    available = _stylesheet_name_set()
    _USER_REGISTERED_STYLESHEETS.clear()
    _USER_REGISTERED_STYLESHEETS.update(
        name for name in stylesheet_names if name in available
    )


def reload_stylesheets() -> None:
    """Reload bundled and user-provided Matplotlib stylesheets."""
    load_erlab_plotting_stylesheets()
    load_user_stylesheets(reload=True)


@functools.cache
def load_erlab_plotting_stylesheets() -> None:
    """Import ERLab plotting once so its bundled stylesheets are registered."""
    with contextlib.suppress(Exception):
        importlib.import_module("erlab.plotting")


def available_stylesheets(names: Iterable[str] = ()) -> frozenset[str]:
    """Return available stylesheets, loading ERLab plotting if requested names miss."""
    names = tuple(names)
    load_user_stylesheets()
    available = _stylesheet_name_set()
    if names and any(name not in available for name in names):
        before_load = available
        load_erlab_plotting_stylesheets()
        available = _stylesheet_name_set()
        _ERLAB_REGISTERED_STYLESHEETS.update(
            name for name in names if name not in before_load and name in available
        )
    return available


def stylesheets_require_user_stylesheets(names: Iterable[str]) -> bool:
    """Return whether a requested stylesheet is registered from the user directory."""
    names = tuple(names)
    if not names:
        return False
    if any(name in _USER_REGISTERED_STYLESHEETS for name in names):
        return True
    load_user_stylesheets()
    return any(name in _USER_REGISTERED_STYLESHEETS for name in names)


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
