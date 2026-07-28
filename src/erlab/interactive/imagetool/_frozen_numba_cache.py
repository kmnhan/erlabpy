"""Numba cache compatibility for the standalone PyInstaller application."""

import pathlib
import sys
import typing
from collections.abc import Callable

from numba.core import config
from numba.core.caching import (
    CacheImpl,
    UserProvidedCacheLocator,
    UserWideCacheLocator,
    _CacheLocator,
)


def _is_bundled_source(py_file: str) -> bool:
    """Return whether a source path belongs to the running PyInstaller bundle."""
    if not (getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")):
        return False

    path = pathlib.Path(py_file)
    if not path.is_absolute():
        # Keep synthetic paths such as ``<ipython-input-...>`` on Numba's
        # standard locator path; PyInstaller PYZ entries are regular relative
        # module paths.
        return not py_file.startswith("<")

    return path.is_relative_to(pathlib.Path(sys._MEIPASS))


class _PyInstallerCacheSubpathMixin:
    """Generate stable subpaths for source files inside a PyInstaller bundle."""

    @classmethod
    def get_suitable_cache_subpath(cls, py_file: str) -> str:
        path = pathlib.Path(py_file)
        extraction_root = pathlib.Path(typing.cast("str", vars(sys)["_MEIPASS"]))

        if path.is_absolute():
            path = path.relative_to(extraction_root)

        # Treat the executable as a synthetic directory. Including its name
        # prevents unrelated frozen applications installed together from
        # sharing a cache identity for the same relative module path.
        path = pathlib.Path(sys.executable) / path

        return _CacheLocator.get_suitable_cache_subpath(str(path))


class _PyInstallerUserProvidedCacheLocator(
    _PyInstallerCacheSubpathMixin, UserProvidedCacheLocator
):
    """Honor ``NUMBA_CACHE_DIR`` for bundled sources with a stable subpath."""

    @classmethod
    def from_function(
        cls, py_func: Callable[..., typing.Any], py_file: str
    ) -> _CacheLocator | None:
        if not (_is_bundled_source(py_file) and config.CACHE_DIR):
            return None

        # UserProvidedCacheLocator normally rejects source paths that do not
        # exist. Relative PyInstaller PYZ filenames are valid in a frozen app,
        # so use the same frozen-source exception as UserWideCacheLocator.
        self = cls(py_func, py_file)
        try:
            self.ensure_cache_path()
        except OSError:
            return None
        return self


class _PyInstallerUserWideCacheLocator(
    _PyInstallerCacheSubpathMixin, UserWideCacheLocator
):
    """Use Numba's user-wide root for bundled sources with a stable subpath."""

    @classmethod
    def from_function(
        cls, py_func: Callable[..., typing.Any], py_file: str
    ) -> _CacheLocator | None:
        if not _is_bundled_source(py_file):
            return None
        return super().from_function(py_func, py_file)


class PyInstallerCacheLocator:
    """Preserve Numba's locator policy while stabilizing bundled source paths.

    ``NUMBA_CACHE_LOCATOR_CLASSES`` replaces Numba's locator chain instead of
    extending it. This dispatcher handles PyInstaller sources with stable
    variants of Numba's user-provided and user-wide locators, then delegates
    every other source to Numba's original locator chain. It is selected only
    by the frozen Windows ImageTool Manager entry point and can be removed once
    Numba preserves frozen cache identities itself.
    """

    @classmethod
    def from_function(
        cls, py_func: Callable[..., typing.Any], py_file: str
    ) -> _CacheLocator | None:
        for locator_cls in (
            _PyInstallerUserProvidedCacheLocator,
            _PyInstallerUserWideCacheLocator,
        ):
            if locator := locator_cls.from_function(py_func, py_file):
                return locator

        # Do not copy Numba's current list here. Delegating to the class-owned
        # chain preserves its ordering and any locators added by future releases.
        for locator_cls in CacheImpl._locator_classes:
            if locator := locator_cls.from_function(py_func, py_file):
                return locator
        return None
