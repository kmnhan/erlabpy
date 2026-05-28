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


def _mpl_cache_directory() -> pathlib.Path:
    path = _cache_directory() / "matplotlib"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _configure_packaged_matplotlib_cache() -> None:
    if not (getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")):
        return
    os.environ["MPLCONFIGDIR"] = str(_mpl_cache_directory())


if __name__ == "__main__":
    multiprocessing.freeze_support()
    _configure_packaged_matplotlib_cache()

    from erlab.interactive.imagetool.manager import main

    main()
