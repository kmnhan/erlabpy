"""Plotting utilities.

"""
import os
import io
import pkgutil

import cmasher
import cmocean
import colorcet
import matplotlib
import matplotlib.style
import matplotlib.colors
import matplotlib.font_manager
import numpy as np

import erlab.io

__all__ = [
    "annotations",
    "bz",
    "colors",
    "general",
    "plot3d",
]


def load_igor_ct(fname, name):
    file = pkgutil.get_data(__package__, "IgorCT/" + fname)
    if fname.endswith(".txt"):
        values = np.genfromtxt(io.StringIO(file.decode()))
    elif fname.endswith(".ibw"):
        values = erlab.io.load_ibw(io.BytesIO(file)).values

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(name, values / 65535)
    matplotlib.colormaps.register(cmap)
    matplotlib.colormaps.register(cmap.reversed())


load_igor_ct("CTBlueWhite.ibw", "BuWh")
load_igor_ct("CTRainbowLIght.ibw", "RainbowLight")
load_igor_ct("CTRedTemperature.ibw", "RedTemperature")

matplotlib.style.core.USER_LIBRARY_PATHS.append(
    os.path.join(os.path.dirname(__file__), "stylelib")
)
matplotlib.style.core.reload_library()
