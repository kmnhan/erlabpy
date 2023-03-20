"""Functions for manipulating colors in Qt.

"""
from __future__ import annotations
from typing import Literal

import matplotlib
import matplotlib.colors as mcolors
import numpy as np
from qtpy import QtGui
import pyqtgraph as pg

__all__ = [
    "color_to_QColor",
    "pg_colormap_names",
    "pg_colormap_from_name",
    "pg_colormap_powernorm",
    "pg_colormap_to_QPixmap",
]


def color_to_QColor(
    c: str | tuple[float, ...], alpha: float | None = None
) -> QtGui.QColor:
    """Convert a matplotlib color to a :class:`PySide6.QtGui.QColor`.

    Parameters
    ----------
    c
        A valid matplotlib color. See the `matplotlib documentation
        <https://matplotlib.org/stable/tutorials/colors/colors.html>` for more
        information.
    alpha
        If supplied, applies transparency to the color.

    Returns
    -------
    PySide6.QtGui.QColor

    """
    return QtGui.QColor.fromRgbF(*mcolors.to_rgba(c, alpha=alpha))


def pg_colormap_names(
    source: Literal["local", "all", "matplotlib"] = "all"
) -> list[str]:
    """Get all valid `pyqtgraph` colormap names.

    Parameters
    ----------
    source
        If ``'all'``, includes all colorcet colormaps.

    Returns
    -------
    list of str

    """
    local = sorted(pg.colormap.listMaps())
    if source == "local":
        return local
    else:
        _mpl = sorted(pg.colormap.listMaps(source="matplotlib"))
        for cmap in _mpl:
            if cmap.startswith("cet_"):
                _mpl = list(filter((cmap).__ne__, _mpl))
            elif cmap.endswith("_r"):
                # _mpl_r.append(cmap)
                _mpl = list(filter((cmap).__ne__, _mpl))
        if source == "all":
            cet = sorted(pg.colormap.listMaps(source="colorcet"))
            # if (_mpl != []) and (cet != []):
            # local = []
            # _mpl_r = []
            all_cmaps = local + cet + _mpl  # + _mpl_r
        else:
            all_cmaps = local + _mpl
    return list({value: None for value in all_cmaps})


def pg_colormap_from_name(name: str, skipCache: bool = True) -> pg.ColorMap:
    """Gets a :class:`pyqtgraph.ColorMap` from its name.

    Parameters
    ----------
    name
        A valid colormap name.
    skipCache
        Whether to skip cache, by default `True`. Passed onto
        :func:`pyqtgraph.colormap.get`.

    Returns
    -------
    pyqtgraph.ColorMap

    """
    try:
        return pg.colormap.get(name, skipCache=skipCache)
    except FileNotFoundError:
        try:
            return pg.colormap.get(name, source="matplotlib", skipCache=skipCache)
        except ValueError:
            return pg.colormap.get(name, source="colorcet", skipCache=skipCache)


def pg_colormap_powernorm(
    cmap: str | pg.ColorMap,
    gamma: float,
    reverse: bool = False,
    highContrast: bool = False,
    zeroCentered: bool = False,
) -> pg.ColorMap:
    if isinstance(cmap, str):
        cmap = pg_colormap_from_name(cmap, skipCache=True)
    N = 4096

    if gamma == 1:
        mapping_fn = lambda x: x
    elif highContrast:
        mapping_fn = lambda x: 1 - np.power(np.flip(x), 1.0 / gamma)
    else:
        if gamma < 1:
            N = 65536  # maximum uint16
        mapping_fn = lambda x: np.power(x, gamma)

    x = np.linspace(0, 1, N)
    if zeroCentered:
        mapping = np.piecewise(
            x,
            [x < 0.5, x >= 0.5],
            [
                lambda x: 0.5 - 0.5 * mapping_fn(-2 * x + 1),
                lambda x: 0.5 + 0.5 * mapping_fn(2 * x - 1),
            ],
        )
    else:
        mapping = mapping_fn(x)
    mapping[mapping > 1] = 1
    mapping[mapping < 0] = 0

    cmap.color = cmap.mapToFloat(mapping)
    cmap.pos = np.linspace(0, 1, N)
    if reverse:
        cmap.reverse()
    pg.colormap._mapCache = {}  # disable cache to reduce memory usage
    return cmap


def pg_colormap_to_QPixmap(
    cmap: str | pg.ColorMap, w: int = 64, h: int = 16, skipCache: bool = True
) -> QtGui.QPixmap:
    """Converts a :class:`pyqtgraph.ColorMap` to a ``w``-by-``h`` QPixmap thumbnail.

    Parameters
    ----------
    cmap
        The colormap.
    w, h
        Specifies the dimension of the pixmap.
    skipCache : bool, optional
        Whether to skip cache, by default `True`. Passed onto
        :func:`pg_colormap_from_name`.

    Returns
    -------
    qtpy.QtGui.QPixmap

    """

    if isinstance(cmap, str):
        cmap = pg_colormap_from_name(cmap, skipCache=skipCache)
    # cmap_arr = np.reshape(cmap.getColors()[:, None], (1, -1, 4), order='C')
    # cmap_arr = np.reshape(
    # cmap.getLookupTable(0, 1, w, alpha=True)[:, None], (1, -1, 4),
    # order='C')
    cmap_arr = cmap.getLookupTable(0, 1, w, alpha=True)[:, None]

    # print(cmap_arr.shape)
    img = QtGui.QImage(cmap_arr, w, 1, QtGui.QImage.Format_RGBA8888)
    return QtGui.QPixmap.fromImage(img).scaled(w, h)
