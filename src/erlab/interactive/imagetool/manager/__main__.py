import multiprocessing
import os
import pathlib
import sys

from qtpy import QtCore


def _cache_directory() -> pathlib.Path:
    location = QtCore.QStandardPaths.writableLocation(
        QtCore.QStandardPaths.StandardLocation.CacheLocation
    )
    if not location:  # pragma: no cover
        location = str(pathlib.Path.home() / ".cache")

    path = pathlib.Path(location) / "dev.kmnhan.erlabpy.imagetoolmanager"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_subdirectory(name: str) -> pathlib.Path:
    path = _cache_directory() / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _mpl_cache_directory() -> pathlib.Path:
    return _cache_subdirectory("matplotlib")


def _pycache_directory() -> pathlib.Path:
    return _cache_subdirectory("python-pycache")


def _configure_packaged_runtime_caches() -> None:
    if not (getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")):
        return
    os.environ["MPLCONFIGDIR"] = str(_mpl_cache_directory())
    pycache_dir = _pycache_directory()
    os.environ["PYTHONPYCACHEPREFIX"] = str(pycache_dir)
    sys.pycache_prefix = str(pycache_dir)
    sys.dont_write_bytecode = False


if __name__ == "__main__":
    multiprocessing.freeze_support()
    _configure_packaged_runtime_caches()

    from erlab.interactive.imagetool.manager import main

    main()
