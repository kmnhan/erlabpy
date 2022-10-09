import matplotlib.colors as colors
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui

__all__ = [
    "mpl_color_to_QColor",
    "pg_colormap_names",
    "pg_colormap_from_name",
    "pg_colormap_powernorm",
    "pg_colormap_to_QPixmap",
]


def mpl_color_to_QColor(c, alpha=None):
    """Convert matplotlib color to QtGui.Qcolor."""
    return QtGui.QColor.fromRgbF(*colors.to_rgba(c, alpha=alpha))


def pg_colormap_names(source="all"):
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


def pg_colormap_from_name(name: str, skipCache=True):
    try:
        return pg.colormap.get(name, skipCache=skipCache)
    except FileNotFoundError:
        try:
            return pg.colormap.get(name, source="matplotlib", skipCache=skipCache)
        except ValueError:
            return pg.colormap.get(name, source="colorcet", skipCache=skipCache)


def pg_colormap_powernorm(
    cmap, gamma, reverse=False, highContrast=False, zeroCentered=False
):
    if isinstance(cmap, str):
        cmap = pg_colormap_from_name(cmap, skipCache=True)
    if reverse:
        cmap.reverse()
    N = 4096
    if gamma == 1:
        mapping = np.linspace(0, 1, N)
    elif highContrast and (gamma < 1):
        if zeroCentered:
            map_half = (
                1 - np.power(np.linspace(1, 0, int(N / 2)), 1.0 / gamma)
            ) * 0.5 + 0.5
            mapping = np.concatenate((-np.flip(map_half) + 1, map_half))
        else:
            mapping = 1 - np.power(np.linspace(1, 0, N), 1.0 / gamma)
    else:
        if gamma < 1:
            N = 65536  # maximum uint16
        if zeroCentered:
            map_half = np.power(np.linspace(0, 1, int(N / 2)), gamma) * 0.5 + 0.5
            mapping = np.concatenate((-np.flip(map_half) + 1, map_half))
        else:
            mapping = np.power(np.linspace(0, 1, N), gamma)
    cmap.color = cmap.mapToFloat(mapping)
    cmap.pos = np.linspace(0, 1, N)
    pg.colormap._mapCache = {}  # disable cache to reduce memory usage
    return cmap


def pg_colormap_to_QPixmap(cmap, w=64, h=16, skipCache=True):
    """Convert pyqtgraph colormap to a `w`-by-`h` QPixmap thumbnail."""
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
