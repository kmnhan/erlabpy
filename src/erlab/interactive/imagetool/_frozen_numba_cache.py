"""Numba cache compatibility for the standalone PyInstaller application."""

import pathlib
import sys

from numba.core.caching import UserWideCacheLocator


class PyInstallerCacheLocator(UserWideCacheLocator):
    """Use a stable cache identity for modules loaded from a PyInstaller bundle.

    PyInstaller stores relative ``co_filename`` values for modules in its PYZ
    archive. Numba 0.66 resolves those paths against the current working directory
    before hashing them, so otherwise equivalent Windows launches can select
    different user-wide cache folders. One-file bundles can have the same problem
    when an absolute source path contains a new ``sys._MEIPASS`` extraction root.

    Keep Numba's user-wide cache location and source validation unchanged, but
    anchor bundled source paths to the executable so their hashes remain stable.
    This class is selected only by the frozen ImageTool Manager entry point and can
    be removed once Numba preserves frozen cache identities itself.
    """

    @classmethod
    def get_suitable_cache_subpath(cls, py_file: str) -> str:
        path = pathlib.Path(py_file)
        extraction_root = getattr(sys, "_MEIPASS", None)

        if (
            path.is_absolute()
            and extraction_root
            and path.is_relative_to(extraction_root)
        ):
            path = path.relative_to(extraction_root)

        if not path.is_absolute():
            # Treat the executable as a synthetic directory. Including its name
            # prevents unrelated frozen applications installed together from
            # sharing a cache identity for the same relative module path.
            path = pathlib.Path(sys.executable) / path

        return super().get_suitable_cache_subpath(str(path))
