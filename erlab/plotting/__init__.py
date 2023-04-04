"""Plotting utilities.

"""
import os
import pkgutil
from io import StringIO

import cmasher
import cmocean
import colorcet
import matplotlib
import matplotlib.style
import matplotlib.colors
import matplotlib.font_manager
import numpy as np

os.environ["QT_API"] = "pyside6"

__all__ = [
    "annotations",
    "bz",
    "colors",
    "general",
    "interactive",
]


def load_igor_ct(file, name):
    file = pkgutil.get_data(__package__, "IgorCT/" + file)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        name, np.genfromtxt(StringIO(file.decode())) / 65535
    )
    matplotlib.colormaps.register(cmap)
    matplotlib.colormaps.register(cmap.reversed())


load_igor_ct("Blue-White.txt", "BuWh")

matplotlib.style.core.USER_LIBRARY_PATHS.append(
    os.path.join(os.path.dirname(__file__), "stylelib")
)
matplotlib.style.core.reload_library()
